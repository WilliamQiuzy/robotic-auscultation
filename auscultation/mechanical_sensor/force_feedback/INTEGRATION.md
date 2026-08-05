# Force feedback — integrating into the core pipeline

The VEO6 mount streams contact force over USB serial. `force_sensor.ForceSensor`
turns that into a **non-blocking, thread-safe** reading you call from any loop.
Two uses, **one class**: it is learned during the imitation study and monitored
during deployment.

```
SS49E (GP28) ──> RP2040 (firmware/main.py, ~100 Hz "t_ms,raw") ──USB──>
    ForceSensor.read() ──> ForceSample(force_n, displacement_mm, hall_v, status)
```

## One-time setup

1. Flash MicroPython, then install the streamer so it runs on boot:
   ```bash
   PORT=$(ls /dev/cu.usbmodem*)
   python -m mpremote connect $PORT cp firmware/main.py :main.py
   python -m mpremote connect $PORT reset
   ```
   (Remove with `mpremote connect $PORT rm :main.py`; Ctrl-C in a REPL stops the
   stream to reflash.)
2. `pip install pyserial` in the env that runs the policy/recorder.
3. Live check / standalone deployment monitor:
   ```bash
   python force_sensor.py --port $PORT          # prints force at 20 Hz
   ```

## 1. Imitation study — force as a LEARNED observation

Add force to the observation dict every control step so the policy sees it. With
a LeRobot-style recorder (LingBot-VA `lerobot_latent_dataset`), force is just
another low-dim observation key:

```python
from force_sensor import DEFAULT_CALIBRATION_PATH, ForceSensor, ForceCalibration

cal = ForceCalibration.from_json(DEFAULT_CALIBRATION_PATH)
with ForceSensor(PORT, cal, log_csv="episode_force.csv") as force:
    while collecting_episode:
        obs    = robot.get_observation()          # images, joint state, ...
        action = teleop.get_action()
        obs["observation.contact_force_n"] = force.read().force_n   # <-- one line
        dataset.add_frame(observation=obs, action=action)
```

- Use a **stable key** (`observation.contact_force_n`) and add it to the dataset
  feature spec (1-D float32) so training picks it up as a policy input.
- The reader is decoupled from the control rate: `read()` returns the latest
  ~100 Hz sample without blocking your loop.
- `log_csv` keeps a raw force trace per episode for debugging/calibration.

## 2. Deployment and collection safety

The safety response depends on which process owns the motion history:

- The autonomous `AuscultationController` records the last finite base-frame approach direction,
  stops approach at the screened limit (≥4.6 N), and commands a direction-checked, distance-bounded
  retract. Missing/stale direction, actuator errors, excess travel, or the physical hard limit
  (≥5.5 N) hold and latch ESTOP.
- The leader–follower **collection** teleop is a separate binary and currently exposes no verified
  retract/ack interface. The recorder therefore never guesses a retract direction or starts a second
  arm controller. On `SCREENED_LIMIT`, `HARD_LIMIT`, `STALE`, or `NO_DATA`, the supervisor requests
  teleop shutdown and starts the independently commissioned direct motor-disable command immediately;
  the demo is aborted and the operator retracts after isolation is confirmed.

Direct disable proves only commanded energy isolation. It does **not** prove that gravity or mechanism
geometry unloads the patient. Before patient-adjacent collection, phantom/HIL commissioning must bound
detection-to-disable latency and verify across all intended dual-arm poses that torque-off cannot
increase patient-directed force. Until then, do not describe this path as passive unload or automatic
retract. A future automatic collection retract must live inside the long-running teleop process, latch
out leader-follow updates, and return measured FK/hold evidence; an independent motor helper would race
the teleop controller on the same bus.

`ForceSample.status`: `OK` (within ±0.5 N of 4 N target) · `WARN` ·
`SCREENED_LIMIT` (≥4.6 N and <5.5 N) · `HARD_LIMIT` (≥5.5 N) · `STALE`
(no fresh sample within 200 ms) · `NO_DATA`.

## Calibration (do before trusting absolute newtons)

`force_feedback_calibration_veo6_tared.json` is the default for the as-built,
inverted mount, but it still ships with **provisional** points
(`calibrated:false`). To bench-calibrate the assembled module:

1. Assemble the module (magnet on the carriage at the 7.75 mm rest gap).
2. Stream raw with the firmware; press the contact face onto a digital scale.
3. Record `(hall_v, force_n)` at ~1, 2, 4, 5 N (read `hall_v` from the live
   monitor; force from the scale).
4. Replace `calibration_points` with these pairs (ascending `hall_v`), set
   `calibrated: true`. The host conversion updates with no reflash.
