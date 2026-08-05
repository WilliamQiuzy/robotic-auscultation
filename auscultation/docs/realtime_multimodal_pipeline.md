# Real-time multimodal auscultation pipeline (audio + contact force + robot)

How the **stethoscope audio** and the **Hall contact-force** signal are brought into one
continuously-monitored, time-aligned pipeline — for **live observation**, **real-time
inference**, and **imitation-learning data collection + annotation**.

> Scope: posterior **lung** auscultation. Acoustic quality model **Q** = `P(usable) ∈ [0,1]`
> (frozen OPERA-CT encoder + logistic head; bandpass 90–1100 Hz, 8 s windows, window-mean).
> See [acoustic_quality_strategy.md](acoustic_quality_strategy.md), [model_architecture.md](model_architecture.md).

---

## 0. Hardware topology — everything on one host, one clock

All sensors and the robot connect to **one machine: linbox (the local RTX 5090 Linux hardware and
real-time-inference host)**. Single host ⇒ a single monotonic clock ⇒ synchronization is a software
problem, not a network one. OPERA-CT on a 5090 is <10 ms/window, so real-time Q is trivial.

```
ThinkLabs One ──3.5mm analog──> Scarlett Solo (sound card) ──USB──>┐
SS49E Hall (GP28) ──> RP2040 (MicroPython, ~94 Hz "t_ms,raw") ──USB──>┤   linbox / RTX 5090
OpenArm follower (joint states) ───────────────────────────USB/EtherCAT──>┤   Linux host
OpenArm leader (teleop targets) ──────────────────────────USB/EtherCAT──>┘   (one clock)
```

Heavy training and offline evaluation never run on linbox; completed immutable LeRobot releases
are transferred to H200 for those jobs. The commissioned Hall calibration is persistent linbox data
at `~/wam_data/config/calibration/force_feedback_calibration_veo6_tared.json` (or the explicit
`WAM_HALL_CALIBRATION` override), and its SHA-256 is recorded in each session.

- **Audio:** continuous, 48 kHz in → resampled to 16 kHz for the model.
- **Force:** `force_feedback/force_sensor.py` already gives a **non-blocking, host-timestamped**
  `ForceSample(t, force_n, displacement_mm, hall_v, status)` at ~94 Hz. **Reuse as-is.**
- **Robot:** OpenArm in **leader–follower** teleop (see §7); follower states = observation,
  leader targets = action.

**Naming:** the arm is **OpenArm** (formerly Trossen/WidowX). The `mechanical_sensor/` CAD,
generated files, and tests are all renamed to `openarm`; the VEO6 flange already matches
OpenArm's measured 4×M3 adapter pattern.

---

## 1. Clock & synchronization (the multimodal命门)

Everything is stamped with `time.monotonic()` on the host at capture. Two corrections make the
streams co-registered to sub-window accuracy:

1. **Serial drift fit (force).** RP2040 sends its own `t_ms`. Fit a line `host_recv ≈ a·t_ms + b`
   over a rolling buffer ⇒ convert device time → host time, removing per-sample jitter.
2. **Fixed audio↔force offset, calibrated by taps.** Before collection, **tap the diaphragm**:
   it produces a sharp transient in audio **and** a force impulse in Hall. Cross-correlate the
   two envelopes ⇒ a constant `Δ_audio↔force = force_t - audio_t` (sound-card buffering + serial
   latency). Apply it exactly once as `audio_aligned = audio_host + Δ_audio↔force`.
3. **Camera exposure latency, measured by LED flash.** Fit camera-device drift to host receipt time,
   then subtract the measured `host_arrival - exposure` latency. Receipt time alone is not exposure
   time and is never accepted as a training timestamp.

The persisted quality gates are intentionally strict:

```bash
# Requires >=4 one-to-one diaphragm-tap matches and offset std <=5 ms.
python ausc/sync_calib.py --seconds 10 \
  --audio-device "Scarlett Solo USB" --audio-channel 1 \
  --hall-port /dev/ttyACM0 --hall-cal /path/force_calibration.json \
  --out wam_sync_v1.json

# Supply >=20 repeated hardware LED-flash measurements; std must be <=5 ms.
python ausc/camera_latency_calib.py --samples-ms 12.1 11.8 12.4 ... \
  --backend orbbec --camera-id FEMTO_BOLT_SERIAL \
  --width 640 --height 480 --fps 30 --color-format RGB \
  --depth-width 640 --depth-height 576 --depth-fps 30 --depth-format Y16 \
  --align-mode software_depth_to_color --frame-aggregate-mode full_frame_require \
  --out wam_camera_latency_v1.json
```

Both JSON files carry the measurement method, sample count, uncertainty, and an exact
`capture_mode`. Sync keeps the requested audio name and Hall path only as diagnostics. Its actual
output is immutable: `--out` must not already exist, publication uses atomic no-replace semantics,
and device/tap/uncertainty failure exits nonzero even when no output path was requested. The identity
contract binds the selected PortAudio index, host-API name, resolved device name, maximum
input channels, audio channel, Hall USB serial/VID/PID, and force-calibration hash. On Linux ALSA it
also maps the PortAudio device through sysfs to the audio interface's USB serial/VID/PID and USB
topology path. A serialless integrated Linux audio device is eligible only when VID/PID and the
fixed sysfs USB topology all match; moving it to another USB path fails closed. A changed ALSA card
or PortAudio index is accepted only when the stable serial identity still matches. Hall always
requires a stable USB serial. Therefore run tap
calibration on the acquisition host, with the exact sound-card mode used for collection;
the calibration CLI enforces this by default. `--allow-unstable-audio-identity` exists only for
non-arm rehearsal and writes an artifact that arm collection will reject. With matching audio USB
serial/VID/PID, ALSA `hw:N,M` name and PortAudio-index remapping are treated as diagnostics rather
than a hardware change. With serialless topology identity, the USB path is part of the immutable
identity. In both cases host API, input-channel capacity, and selected channel must match;
camera latency is bound to backend, stable camera ID, color/depth profiles, pixel formats,
alignment, frame aggregation, FPS, and timestamp semantics. The recorder verifies the live RGB-D
frame shape/time source and the backend's negotiated pipeline. A
hand-entered zero, reused tap, high-jitter result, mismatched device/mode, missing file, or
post-capture modification makes the session ineligible.

After this, any time `t` maps to an aligned `(audio_sample, force_sample, robot_state)`.

---

## 2. Acquisition layer — three independent producers, one buffer

Each sensor runs its own thread/producer writing into a **ring buffer**; the model/loggers are
consumers. Capture is **never blocked** by inference (else audio drops).

- **AudioStream** (new, mirrors `ForceSensor`'s ergonomics): `sounddevice` callback on the
  Scarlett device, 48 kHz → ring buffer with per-block host timestamps. The callback **only
  copies**; resample-to-16 kHz and feature extraction happen downstream.
  ```python
  with AudioStream("Scarlett Solo USB", sr_in=48000, sr_model=16000) as audio:
      win = audio.latest_window(seconds=8)   # -> (samples_16k, t_start, t_end), non-blocking
  ```
- **ForceSensor** (existing): `force.read()` → latest `ForceSample`. Background thread already.
- **Robot**: the left unilateral OpenArm controller publishes exact 1 kHz follower observations
  and the exact MIT parameters consumed at each control tick; the hardware mapping is
  `can1` leader → `can3` follower.

---

## 3. Multi-rate fusion — don't put 16 kHz into every frame

Three native rates: **robot/control 1 kHz**, **force ~94 Hz**, **audio 16 kHz**. Rule:
**audio is a continuous side-stream**, not per-frame samples.

| Stream | Native rate | How it enters the pipeline |
|---|---|---|
| Robot state / action | 1 kHz | canonical v2 exact-control sidecars; observation alone may be interpolated at exposure nominal, while action is the last real tick whose full arm+gripper call completed by nominal and is never interpolated |
| Contact force | 94 Hz | raw `force.csv`; past-only value plus validity and age at each exposure |
| RGB camera | device FPS | one canonical LeRobot row per corrected exposure timestamp |
| Audio | 16 kHz | immutable continuous `audio.wav` sidecar; never duplicated into every frame |
| Q score / contact gate | segment rate | label/reward in `segments.jsonl` and `next.reward`, never an observation |

Canonical per-camera frame (LeRobot 0.3.3 uses a separate `task=` argument):

```python
obs = {
  "observation.state":                 follower_joint_pos_at_exposure_nominal,
  "observation.robot_state_age_s":     0.0,
  "observation.contact_force_n":       past_force_n,
  "observation.contact_force_valid":   True,
  "observation.contact_force_age_s":   force_age_s,
  "observation.images.top":            rgb_at_exposure,
  "observation.target_point_2d":        target_uv,
  "observation.target_point_3d":        target_xyz_robot_base,
}
frame = {**obs, "action": exact_40_parameter_arm_and_gripper_MIT_action,
         "next.reward": segment_reward, "next.done": target_interval_done}
dataset.add_frame(frame, task="auscultate the R1")
```

The live control loop remains 1 kHz, but the canonical dataset FPS is the camera FPS. Every force
and target join is past-only and bounded (force 50 ms, target 250 ms); missing data is rejected,
never replaced with zero. **Format = LeRobot** (`LeRobotDataset`); audio remains a session sidecar WAV
with `audio.t0_monotonic` plus the hashed sync calibration. The converter maps an exposure query back
to raw WAV time; no per-frame future-looking audio-quality feature is required.

---

## 4. Real-time inference — contact-gated Q

The Hall force turns "is this recording usable?" into two easier questions: **is contact good?**
(Hall, trivial) **and given good contact, is the audio clean?** (Q).

```
force.read() ──> gate = (F_min ≤ force_n ≤ F_max) AND stable(force, window)
                   │ gate == False ──> P_usable = 0   (skip the model — no contact, no point)
                   │ gate == True  ──> OPERA-CT(audio 8s window) ──> Q head ──> Q_audio
P_usable = gate · Q_audio   (or a small learned fusion head)
```

- **Sliding window:** 8 s window, **hop 1 s** ⇒ a `P(usable)` update ~1 Hz; mean-pool the last
  2–3 windows (matches training's window-mean), then EMA/median-smooth to de-flicker.
- **Targets:** derive every threshold from the active force calibration. The current mount targets
  4.0 N, enters `SCREENED_LIMIT` at 4.6 N, and reserves 5.5 N for the physical hard limit.
- **Latency:** on the 5090, OPERA forward <10 ms ⇒ end-to-end well under the 1 s hop. Real-time
  is not a constraint here; we can even hop faster (e.g., 0.5 s) if desired.
- **Safety:** the autonomous `AuscultationController` can command a bounded retract because it owns
  motion history and records a validated base-frame approach direction. The leader–follower
  **collection** path has no verified retract/ack interface: `SCREENED_LIMIT`, `STALE`, and `NO_DATA`
  request teleop shutdown and start the independently commissioned direct-disable command
  immediately; `HARD_LIMIT` uses the same immediate energy-isolation path. Every such demo is
  aborted, and the operator retracts only after isolation is confirmed. This is not proof of passive
  force relief; phantom/HIL testing across arm poses remains mandatory before patient-adjacent use.

---

## 5. Continuous monitoring (live observation)

A streaming view + log, updated every hop:

- **Force** trace with the 4 N target band shaded; `STATUS` (OK / adjust force / screened-limit /
  hard-limit).
- **Rolling mel-spectrogram** of the bandpassed audio.
- **`P(usable)`** gauge + history; **gate** state.
- **Respiratory phase** (insp/exp) and **heart-beat markers** (§6).
- **Operator hint:** "good contact" / "press harder" / "too noisy — hold still" / "reposition".

This live multimodal stream **is** the imitation-learning data (just logged).

---

## 6. Heart vs lung identification

Three cross-checking cues — frequency, periodicity, and the Hall signal:

| Cue | Heart | Lung |
|---|---|---|
| Frequency | low, **20–150 Hz** (S1/S2 "lub-dub") | broadband, **100–1000+ Hz** |
| Periodicity | ~1 Hz, regular sharp double-thump | respiration **0.2–0.33 Hz** slow envelope modulation |
| Site | strong anteriorly | **posterior = lung-dominant, heart attenuated** (our case) |

- **Heart rate / contamination:** take the **20–150 Hz band envelope** → autocorrelation peak at
  ~0.5–1.5 s gives HR; the **low-band/total energy ratio** is a "heart-contamination" metric.
  Too high ⇒ probe sits too anterior ⇒ **reposition**.
- **Respiratory phase (insp/exp):** the **Hall force directly encodes breathing** — chest-wall
  rise/fall modulates contact force at 0.2–0.33 Hz. Use **Hall (primary) + lung-band envelope
  (secondary)**; rising edge = inspiration. (Adventitious sounds are phase-specific: crackles
  late-inspiratory, wheezes expiratory — useful for annotation.)
- Our **90–1100 Hz bandpass already suppresses most heart energy (<90 Hz)**, so heart detection
  is used mainly to **judge placement**, not to separate signals.

---

## 7. Processing/annotation unit = 2–3 respiratory cycles

A respiratory cycle (inspiration + expiration) ≈ **3–5 s** (12–20 breaths/min). Decision:

- **Real-time decision window:** 8 s sliding (for the Q score).
- **Storage / annotation unit:** **2–3 full respiratory cycles (~10–15 s), cycle-boundary-aligned.**
  - 1 cycle is too short/variable — crackles appear in only some cycles; need ≥2–3 to catch them.
  - 2–3 cycles give Q **2–3 eight-second windows to mean-pool** (matches training) ⇒ stable score.
  - Aligns with clinical practice (listen ≥1 full breath) and with **ICBHI's respiratory-cycle
    annotations**.
- **Segmentation:** detect inspiration onsets from **Hall respiratory oscillation (primary) +
  lung-band envelope (secondary)** → cut the continuous stream into cycle-aligned clips; store
  boundaries in metadata.

---

## 8. Teleoperation & data collection (imitation learning)

**Leader–follower teleop (confirmed).** The **stethoscope is mounted on the follower** OpenArm;
a human moves the **leader** OpenArm and the follower mirrors it to perform the auscultation.
Recorded per control tick: **follower joint states → observation stream**, and the exact follower
arm/gripper MIT `(kp, kd, q, dq, tau)` parameters consumed by the controller → action stream.
The converter emits one camera-rate action by selecting a single real tick; copied follower
observations and interpolated actions are explicitly forbidden.

**Per-session multimodal container**, all host-clock aligned:

```
session_YYYYMMDD_HHMMSS/
  manifest.json      # lifecycle + wam_promotion_v1 contract, identities and calibration/raw hashes
  FINALIZATION.json  # external SHA-256 seal for the exact promoted manifest bytes
  calibrations/      # immutable force calibration snapshot
  audio.wav          # continuous 16 kHz; ADC clock mapped to host monotonic
  force.csv          # host/device clocks, raw Hall values, calibrated force and status (~94 Hz)
  robot_control_envelopes.jsonl # canonical publisher bytes; 100 Hz envelopes of 1 kHz ticks
  robot_exact_control.csv       # every exact observation/action tick once
  robot_frame_index.jsonl       # RGB-D exposure bracket and exact action-range hashes
  robot_capture_summary.json    # commit marker, counts/hashes/source closure/blockers
  rgb.mp4            # clean RGB frames
  rgb_times.csv      # frame index, host arrival, device timestamp and explicit time source
  depth/              # one depth array per RGB-D frameset plus depth_times.csv
  obs.csv            # live fused monitoring rows (not the canonical training join)
  targets.jsonl      # camera-bound target, robot_base xyz and target identity
```

Promotion decodes and validates the payloads, hashes every raw file before and after that validation,
writes `artifact_hashes` plus `promotion_contract_version: wam_promotion_v1`, and externally seals the
exact manifest bytes in `FINALIZATION.json`. Conversion verifies the seal, binds the same manifest
bytes through discovery and snapshot, and checks every expected raw hash while freezing inputs. Any
manifest or raw file changed after promotion is rejected before a dataset is published.

`session_to_lerobot_v2.py` canonical-replays all four robot artifacts, requires one robot frame row
for every RGB-D frame, and writes `robot_frame_action_selection.jsonl`. Its fixed selection rule is
`latest_exact_control_tick_completed_at_or_before_exposure_nominal_ns`; completion means the full
arm+gripper control call ended before the exposure nominal. Each row stores the selected tick ID,
tick time, action hash, source action-range hash, and the full-sidecar hashes used by WorldAction
training. Tick IDs/timestamps remain audit metadata rather than policy observations, preventing
absolute session-position leakage. The original 1 kHz files remain in the immutable dataset snapshot.

**Current training gate:** the audited OpenArm SDK's `mit_control_all` returns `void`, so consumed
action send status is not observable. Current v2 summaries therefore keep
`training_eligible=false` with `robot_action_send_status_unobservable`; the converter preserves the
evidence for commissioning but refuses to publish it as VLA/WAM training data. This blocker must be
resolved in the controller/SDK and represented by a newly validated sidecar contract—it must not be
bypassed in conversion.

Arm-connected collection is launched only through the fail-closed supervisor. It requires the
exact commissioning artifacts and a direct motor-disable command:

```bash
export SYNC_CAL=/path/wam_sync_v1.json
export CAMERA_LATENCY_CAL=/path/wam_camera_latency_v1.json
export CAMERA_ID=FEMTO_BOLT_SERIAL
export TARGETS_FILE=/path/robot_base_targets.jsonl
export TELEOP_STOP_CMD_JSON='["/path/direct_disable"]'
scripts/record_teleop_session.sh left 15 posterior-lung demo_001
```

**Smart collection** using the real-time `Q + contact gate`:
1. low-quality segment ⇒ prompt **re-record on the spot**;
2. **save/annotate only good-contact segments** ⇒ saves annotation budget;
3. **stratify by quality** (reuse the earlier stratified-sampling策略) ⇒ balanced usable/unusable.

---

## 9. Annotation logic

Send clinicians the **cycle-aligned 2–3-cycle clips**:

- **Primary label:** `usable / unusable` (the Q target).
- **Optional:** respiratory phase, adventitious sounds (crackle/wheeze), heart-contamination.
- Labels keyed to `clip_id` + cycle boundaries ⇒ feed directly into Q retraining on **in-domain,
  first-party** data (the real unlock for tightening the AUC CI and upgrading the head to an MLP).

---

## 10. Open parameters & follow-ups

| Item | Default chosen | Note |
|---|---|---|
| Control loop | **50 Hz** | 30 Hz if data size matters |
| Data format | **LeRobot** + sidecar WAV | audio aligned by `audio_t` |
| Hop / window | 1 s hop / 8 s window | can hop 0.5 s on the 5090 |
| Force band gate | 3–5 N around 4 N target | from force calibration |
| Annotation unit | 2–3 resp. cycles (~10–15 s) | cycle-aligned |
| Inference host | **local RTX 5090 Linux** | OPERA <10 ms/window |

**Status / follow-ups:**
- ✅ `AudioStream` built — `ausc/audio/audio_stream.py` (Scarlett stethoscope on `pick_channel=1`).
- ✅ `CameraStream` built — `ausc/perception/camera_stream.py` (Orbbec Femto Bolt RGB, avfoundation idx 1, 1920×1080).
- ✅ The software path is fail-closed: device/mode-bound calibration, sensor-first start barrier,
  concurrent safety supervision, hash-bound artifact validation, causal conversion and provenance.
- ✅ Teleop = leader–follower, stethoscope on the follower arm.
- ✅ **Trossen → OpenArm rename** done across `mechanical_sensor/` CAD, generated files, and tests.
- ⏳ Hardware commissioning remains: bench-scale force calibration, exact-device sync/camera
  calibrations, measured direct-disable latency plus phantom validation that torque-off cannot increase
  patient-directed force in any collection pose, robot-base targets, and a real LeRobot 0.3.3 reopen
  smoke. Until then, direct disable is an energy-isolation fallback—not a proven passive unload or
  automatic retract.
