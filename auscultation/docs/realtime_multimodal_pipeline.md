# Real-time multimodal auscultation pipeline (audio + contact force + robot)

How the **stethoscope audio** and the **Hall contact-force** signal are brought into one
continuously-monitored, time-aligned pipeline — for **live observation**, **real-time
inference**, and **imitation-learning data collection + annotation**.

> Scope: posterior **lung** auscultation. Acoustic quality model **Q** = `P(usable) ∈ [0,1]`
> (frozen OPERA-CT encoder + logistic head; bandpass 90–1100 Hz, 8 s windows, window-mean).
> See [acoustic_quality_strategy.md](acoustic_quality_strategy.md), [model_architecture.md](model_architecture.md).

---

## 0. Hardware topology — everything on one host, one clock

All sensors and the robot connect to **one machine: the local RTX 5090 Linux box** (the GPU
inference host). Single host ⇒ a single monotonic clock ⇒ synchronization is a software
problem, not a network one. OPERA-CT on a 5090 is <10 ms/window, so real-time Q is trivial.

```
ThinkLabs One ──3.5mm analog──> Scarlett Solo (sound card) ──USB──>┐
SS49E Hall (GP28) ──> RP2040 (MicroPython, ~94 Hz "t_ms,raw") ──USB──>┤   RTX 5090
OpenArm follower (joint states) ───────────────────────────USB/EtherCAT──>┤   Linux host
OpenArm leader (teleop targets) ──────────────────────────USB/EtherCAT──>┘   (one clock)
```

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
2. **Fixed audio↔force offset, calibrated by a tap.** At session start, **tap the diaphragm**:
   it produces a sharp transient in audio **and** a force impulse in Hall. Cross-correlate the
   two envelopes ⇒ a constant `Δ_audio↔force` (sound-card buffering + serial latency). Subtract it.

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
- **Robot**: `robot.get_observation()` (joint pos/vel, EE pose) and the leader's commanded
  `action` at the control rate.

---

## 3. Multi-rate fusion — don't put 16 kHz into every frame

Three native rates: **robot/control 50 Hz**, **force ~94 Hz**, **audio 16 kHz**. Rule:
**audio is a continuous side-stream**, not per-frame samples.

| Stream | Native rate | How it enters the pipeline |
|---|---|---|
| Robot state / action | 50 Hz | one LeRobot frame per step |
| Contact force | 94 Hz | logged raw to `force.parquet`; **latest value** sampled into each 50 Hz frame |
| Audio | 16 kHz | continuous `episode_XXX.wav` + `t0`; per frame stores only a **timestamp pointer** + the latest `q_usable` |
| Q score | ~1–5 Hz | sliding-window output, **held** between updates in the frame |

Per-frame observation (LeRobot `add_frame`):

```python
obs = {
  "observation.state":          robot_joint_pos,        # OpenArm follower
  "observation.contact_force_n": force.read().force_n,   # 94 Hz -> latest
  "observation.q_usable":        q_stream.latest,        # ~1-5 Hz -> held
  "observation.resp_phase":      resp.phase,             # insp/exp (§6)
  "observation.audio_t":         audio.now_t,            # pointer into episode wav
  # "observation.images.cam":    frame,                  # if cameras present
}
action = leader_joint_targets                            # §7
dataset.add_frame(obs, action)
```

**Control loop = 50 Hz** (ALOHA/ACT/LeRobot convention; captures force dynamics; 94 Hz force
decimates cleanly). **Format = LeRobot** (`LeRobotDataset`); audio as a per-episode sidecar WAV
aligned by `t0` + `observation.audio_t`, because tabular per-frame storage can't hold raw audio.

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
- **Targets:** `F_min..F_max` from the force calibration (target 4 N, hard limit 8 N).
- **Latency:** on the 5090, OPERA forward <10 ms ⇒ end-to-end well under the 1 s hop. Real-time
  is not a constraint here; we can even hop faster (e.g., 0.5 s) if desired.
- **Safety:** force `HARD_LIMIT` (>8 N) trips the robot e-stop independently of the model
  (already in `ForceSensor(on_hard_limit=...)`).

---

## 5. Continuous monitoring (live observation)

A streaming view + log, updated every hop:

- **Force** trace with the 4 N target band shaded; `STATUS` (OK / adjust force / hard-limit).
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
Recorded per 50 Hz step: **follower joint states → `observation.state`**, **leader joint targets
→ `action`**.

**Per-session multimodal container**, all host-clock aligned:

```
session_YYYYMMDD_HHMMSS/
  meta.json          # subject, site (e.g. LLL/RUL), device settings, gains, calib, t0
  audio.wav          # continuous 16 kHz + t0 in meta
  force.parquet      # t, raw, hall_v, displacement_mm, force_n, status   (~94 Hz)
  robot.parquet      # t, follower joint pos/vel, EE pose, leader targets (50 Hz)
  events.jsonl       # contact on/off, probe-move, tap-sync, operator marks
  cycles.jsonl       # respiratory-cycle boundaries + insp/exp labels
  lerobot/           # LeRobotDataset frames (obs+action), audio referenced by audio_t
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
- ✅ **All three streams verified co-running** on one host-monotonic clock (force + audio + camera, 3/3 live).
- ✅ Teleop = leader–follower, stethoscope on the follower arm.
- ✅ **Trossen → OpenArm rename** done across `mechanical_sensor/` CAD, generated files, and tests.
- ⏳ Depth from the Femto Bolt (Orbbec SDK) — RGB-only via avfoundation for now.
