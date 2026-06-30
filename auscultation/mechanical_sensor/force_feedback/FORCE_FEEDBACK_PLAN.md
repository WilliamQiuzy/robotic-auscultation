# ThinkLabs Passive Mount Force Feedback Plan

## What Is Being Measured

The current mount is a passive spring carriage. The robot moves the fixed frame, the ThinkLabs One sits in the sliding carriage, and the contact force is estimated from spring compression:

```text
force_n = spring_rate_n_per_mm * compression_mm
```

For the current compact v8B target spring, MISUMI `WF6-25`, `spring_rate_n_per_mm = 0.5`. A 4 N contact target corresponds to about `8.0 mm` spring compression. The checked spring limit is `11.25 mm` allowable deflection and `5.5 N` maximum load, so v8B uses a `5.5 N` hard force screen. The older v6/v7A baseline used `0.38 N/mm`, which required about `10.53 mm` spring compression.

The ThinkLabs One audio cable stays separate. We do not need to solder or modify the stethoscope. The force-feedback wiring belongs to the printed mount:

```mermaid
flowchart LR
  A["Moving carriage"] --> B["Magnet or sensor target"]
  B --> C["Fixed-frame displacement sensor"]
  C --> D["Microcontroller ADC/I2C"]
  D --> E["USB serial"]
  E --> F["Computer force monitor"]
```

## V7 Sensor Variants

### v7A Hall Sensor Plus Magnet, Recommended

- Put a small magnet in the moving carriage pocket.
- Mount a linear Hall sensor or magnetic linear encoder on the fixed frame rail.
- Measure carriage compression without contact, friction, or wiring on the moving stethoscope.
- Convert displacement to force with the calibrated spring constant.

Why this is the recommended first hardware version:

- No wire is attached to the ThinkLabs One.
- The sensor does not sit in the acoustic/contact path.
- It keeps the passive compliance simple.
- It is easy to waterproof or cover because it sits on the side rail.

Expected wiring:

```text
Hall sensor VCC/GND/signal -> microcontroller analog or I2C input
microcontroller USB -> laptop
ThinkLabs One audio/USB path -> separate audio acquisition path
```

### v7B Linear Potentiometer

- Attach a small linear potentiometer or spring-loaded slider between fixed frame and moving carriage.
- Read analog voltage with a microcontroller.
- Convert voltage to compression, then force.

This is mechanically easy to debug but less clean than Hall sensing because the pot can add friction, wear, and cable drag.

### v7C Inline Load Cell

- Place a button load cell or miniature compression load cell behind the contact disk or spring seat.
- Read through HX711, NAU7802, or another bridge amplifier.
- Stream force directly instead of estimating from spring compression.

This gives the most direct measurement, but it changes stack height and stiffness. It is better as a second validation build after v7A is working.

## How We Judge Good Or Bad

The module is good if it controls force without losing acoustic contact, without bottoming out, and without printed plastic touching the patient before the ThinkLabs contact face.

Current numeric gates:

| Metric | Gate | Why |
| --- | ---: | --- |
| Steady contact force | `4.0 +/- 0.5 N` | enough seal without excessive pressure |
| Peak contact force | `< 5.5 N hard screen` | stays inside the checked WF6-25 spring load |
| Spring travel use | `< 8.5 mm preferred`, `< 11.0 mm hard screen` | leaves margin before the WF6-25 deflection limit |
| Contact during hold | `> 98%` | no bouncing or loss of acoustic seal |
| Settling time | `< 1.0 s` | robot can move point to point efficiently |
| Force noise | `< 0.1 N RMS` after filtering | computer display is stable |
| Calibration error | `< 0.25 N` over 0-5.5 N | displacement-to-force mapping is trustworthy |
| Sensor update rate | `>= 50 Hz`, target `100 Hz` | enough for robot feedback loop |
| End-to-end latency | `< 50 ms` | usable for safety stop or approach control |

## Computer Feedback

The computer should show a live numeric stream like:

```text
t=1.23s  compression=10.52mm  force=4.00N  status=OK
```

The first implementation can run over USB serial. The microcontroller sends either JSON lines:

```json
{"t_ms": 1234, "disp_mm": 10.52}
```

or CSV lines:

```text
1234,10.52
```

The laptop monitor computes force, logs CSV, and can later publish the same values to ROS or the OpenArm control process.

## Calibration Procedure

1. Assemble the spring, carriage, and sensor.
2. With no skin contact, press zero and record `zero_compression_mm`.
3. Push against a small digital scale at 1 N, 2 N, 4 N, and 5 N.
4. Record sensor displacement and scale force.
5. Fit `force_n = k * compression_mm + b`.
6. Replace the nominal `0.5 N/mm` value in `v8b_force_feedback_calibration.json` with the fitted spring rate.
7. Run the monitor replay and live tests; reject the build if calibration error exceeds `0.25 N`.

## Design Recommendation

Use v8B for the next printed module:

- Use the compact printed slide and mini-spring geometry in `generated/v8b_compact/`.
- Use `v8b_force_feedback_calibration.json` as the starting computer-side force calibration.
- Keep the magnet pocket on the floating carriage.
- Keep the fixed sensor rail with slots so the Hall sensor can be moved during calibration.
- Route the sensor cable along the fixed robot-side frame, not the moving stethoscope holder.
- Use v7C load cell only as a validation fixture if v8B repeatability is not good enough.
