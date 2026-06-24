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
from force_sensor import ForceSensor, ForceCalibration

cal = ForceCalibration.from_json("force_feedback_calibration_veo6.json")
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

## 2. Deployment — continuous monitoring + safety

Same reader, plus a hard-stop callback that fires the instant force crosses the
screened limit (5.5 N) — independent of the policy:

```python
def on_force_limit(sample):
    robot.hold()                                  # or e-stop / retract
    log.warning("contact force %.2f N >= limit", sample.force_n)

with ForceSensor(PORT, cal, on_hard_limit=on_force_limit) as force:
    while running:
        s = force.read()
        obs["observation.contact_force_n"] = s.force_n   # same key as training
        if s.status in ("STALE", "NO_DATA"):
            robot.hold()                          # lost the sensor -> fail safe
        action = policy(obs)
        robot.step(action)
```

`ForceSample.status`: `OK` (within ±0.5 N of 4 N target) · `WARN` · `HARD_LIMIT`
(≥5.5 N) · `STALE` (no fresh sample within 200 ms) · `NO_DATA`.

## Calibration (do before trusting absolute newtons)

`force_feedback_calibration_veo6.json` ships with **design-nominal** points
(`calibrated:false`). To bench-calibrate the assembled module:

1. Assemble the module (magnet on the carriage at the 7.75 mm rest gap).
2. Stream raw with the firmware; press the contact face onto a digital scale.
3. Record `(hall_v, force_n)` at ~1, 2, 4, 5 N (read `hall_v` from the live
   monitor; force from the scale).
4. Replace `calibration_points` with these pairs (ascending `hall_v`), set
   `calibrated: true`. The host conversion updates with no reflash.
