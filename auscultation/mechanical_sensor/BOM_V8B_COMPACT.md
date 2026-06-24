# V8B Compact Mount BOM and Sourcing Notes

This BOM is tied to the current V8B CAD parameters in
`generated/v8b_compact/v8b_design_metadata.json`.

Checked date: 2026-06-08.

## CAD Constraints That Drive The BOM

- Spring seat OD: `10.0 mm`
- Spring cup inner clearance diameter: `6.8 mm`
- Nominal spring installed envelope in preview: `25 mm`
- Real WF6-25 spring seat face gap at rest: `25.0 mm`
- Passive travel limit: `11.0 mm`
- Nominal simulation spring rate: `0.5 N/mm`
- Target contact force: `4.0 N`
- Target compression for nominal spring: `4.0 / 0.5 = 8.0 mm`
- MISUMI WF6-25 allowable deflection: `11.25 mm`
- MISUMI WF6-25 maximum load: `5.5 N`
- Magnet pocket in moving carriage: about `5.8 x 2.4 x 5.8 mm`
- Hall body pocket/rail on fixed frame: about `7 x 2.2 x 12 mm`
- M3 clearance in CAD: `3.5 mm`
- M2 clearance in CAD: `2.3 mm`

## Buy First

| Item | Qty | Recommended spec | Suggested source | Why |
| --- | ---: | --- | --- | --- |
| Compression spring | 3-10 | OD `6-8 mm`, free length `25 mm`, rate `0.45-0.65 N/mm`, max usable deflection >= `8 mm` | MISUMI `WF6-25`, or nearby McMaster spring | This is the part that sets passive contact force. The ideal V8B target is `0.5 N/mm`; buy a few close rates and calibrate the real one. |
| Linear Hall sensor | 2-5 | Analog linear Hall, 3-wire, 3.3 V or 5 V compatible | Honeywell `SS49E` from Digi-Key, or TI `DRV5053` | Reads magnet displacement without mechanical contact. |
| Magnet | 5-10 | Neodymium disc `5 mm OD x 2 mm thick`, axial magnetization | K&J/Supreme/McMaster equivalent | Fits the V8B magnet pocket and is strong enough for Hall displacement sensing. |
| Microcontroller | 1-2 | USB serial plus at least one analog input | Seeed XIAO RP2040, Adafruit QT Py RP2040, Arduino Nano, Raspberry Pi Pico | Converts Hall analog voltage into displacement and streams it to the Mac. |
| Sensor cable | 1-3 | 3-conductor, flexible, `26-30 AWG`; JST-PH 2.0 3-pin optional | Adafruit silicone wire or JST-PH cable pair | Hall sensor needs `VCC`, `GND`, and `OUT`. Use flexible wire to avoid adding force to the sliding carrier. |
| Clamp screw | 2-4 | M3 socket head, `12-16 mm` long, with M3 nut or heat-set insert | McMaster, Amazon, local hardware | Tightens the split ThinkLabs cradle. |
| Sensor rail screws | 4-8 | M2 socket head, `5-8 mm` long, with nuts or inserts | McMaster, Amazon, local hardware | Holds the tiny Hall board/printed sensor bracket. |
| Robot-side mounting screws | 4-8 | Usually M3, exact length depends on OpenArm flange stack | OpenArm kit hardware or McMaster | The V8B top plate has generic M3 slots; final screw length must be chosen after measuring the actual wrist flange. |

## Amazon Shortlist

Amazon inventory and sellers change often, so these are exact search/buy links
for the current prototype categories rather than permanent part approvals.

- Spring target search:
  https://www.amazon.com/s?k=6mm+OD+25mm+compression+spring+0.5+N%2Fmm
- Hall sensor matching the screenshot:
  https://www.amazon.com/s?k=49E+OH49E+SS49E+S49E+TO-92S+Hall+Effect+Sensor
- `5 x 2 mm` neodymium disc magnets matching the screenshot:
  https://www.amazon.com/s?k=D5x2mm+N42+neodymium+disc+magnets
- Seeed Studio XIAO RP2040 matching the screenshot:
  https://www.amazon.com/s?k=Seeed+Studio+XIAO+RP2040
- Flexible sensor wire, not 3D-printer filament:
  https://www.amazon.com/dp/B07G2SWB19
- JST-PH 2.0 3-pin pigtails, optional connectorized sensor cable:
  https://www.amazon.com/s?k=JST+PH+2.0+3+pin+cable+26AWG
- M2/M3/M4 screw/nut/washer assortment:
  https://www.amazon.com/dp/B07H4MG7TC
- M3 socket-head screw-only search, useful if the assortment is out of stock:
  https://www.amazon.com/s?k=M3+socket+head+cap+screw+assortment+M3+x+12+M3+x+16
- M2 socket-head screw-only search for the Hall rail:
  https://www.amazon.com/s?k=M2+socket+head+cap+screw+assortment+M2+x+6

## Spring Selection

The clean target is:

```text
k = 0.5 N/mm
target force = 4 N
target compression = 8 mm
travel limit = 11 mm
hard force screen = 5.5 N
```

Recommended order strategy:

1. Buy the closest exact target spring first: MISUMI `WF6-25` is listed as a
   `6 mm` OD, `25 mm` free-length compression spring with `0.5 N/mm` spring
   constant, `11.25 mm` allowable deflection, and `5.5 N` maximum load. It is
   the best match to the V8B simulation target.
2. Also buy one stiffer backup spring, around `0.65-0.75 N/mm`. A McMaster
   nearby option is a `25.5 mm` long, `6.93 mm` OD, `5.67 mm` ID spring with
   about `0.15 lbf/mm` rate, which is about `0.67 N/mm`.
3. Avoid springs above about `1.0 N/mm` for the first patient-contact
   prototype. They reach 4 N in less than 4 mm and make the contact much less
   forgiving.

The CAD can physically tolerate a range of springs as long as:

- OD is less than the `10 mm` printed spring cup OD; `6 mm` is the checked fit.
- Free length is around `22-26 mm`.
- Solid height is low enough that `8 mm` compression does not coil-bind.
- The real measured spring rate is written into
  `generated/v8b_compact/v8b_force_feedback_calibration.json`.

## Force Feedback Collection Chain

The current V8B design does not need a load cell. It estimates contact force
from spring compression:

```text
Hall voltage -> magnet displacement -> spring compression -> force
force_N = spring_rate_N_per_mm * max(0, displacement_mm - zero_mm)
```

Mechanical placement:

1. Glue or press-fit the `5 x 2 mm` magnet into the moving carriage pocket.
2. Mount the Hall sensor on the fixed rail so the magnet moves toward/away from
   the sensor along the passive travel axis.
3. Keep the sensor wire on the fixed frame as much as possible. If a wire must
   cross the moving interface, leave a soft service loop so it does not behave
   like an extra spring.

Electrical wiring for an analog Hall sensor:

```text
Hall VCC  -> microcontroller 3V3 or 5V, depending on the sensor
Hall GND  -> microcontroller GND
Hall OUT  -> microcontroller analog input, for example A0
Microcontroller USB -> Mac
```

Computer protocol:

The microcontroller should send one line per sample at about `100 Hz`:

```json
{"t_ms": 1234, "disp_mm": 8.0}
```

Then run:

```bash
/private/tmp/wam-mujoco-venv/bin/python auscultation/mechanical_sensor/force_feedback/force_feedback_monitor.py \
  --calibration auscultation/mechanical_sensor/generated/v8b_compact/v8b_force_feedback_calibration.json \
  --serial-port /dev/cu.usbmodemXXXX \
  --out auscultation/mechanical_sensor/generated/v8b_compact/hardware/force_feedback_log.csv
```

The monitor writes a CSV log plus a summary JSON with peak force, final force,
max compression, and hard-limit count.

## Calibration Procedure

1. Assemble the printed V8B frame, spring, magnet, Hall sensor, and
   microcontroller.
2. With the mount in free air and no patient/contact load, record the raw ADC
   value. Put this into the microcontroller sketch as `ADC_ZERO`.
3. Compress the moving carriage by a measured `8.0 mm` using calipers or a
   fixture. Record the raw ADC value and put it into the sketch as
   `ADC_AT_8MM`.
4. Measure the real spring rate. The quick bench method is to press the module
   against a small scale and record force at several measured compressions.
   Fit `force_N = k * compression_mm`.
5. Replace `spring_rate_n_per_mm` in the V8B calibration JSON with the fitted
   value.
6. Run the serial monitor and confirm:
   - no contact: near `0 N`
   - gentle target contact: about `4 N`
   - hard push: flagged before or near `5.5 N`

## Notes Before Robot Use

- Treat this as a force estimate, not a clinical force sensor, until the
  assembly is calibrated against a scale.
- Add strain relief for both cables: ThinkLabs audio cable and Hall sensor
  cable.
- Do not route the Hall cable where it pulls on the moving carriage.
- The ThinkLabs audio cable is separate from force feedback. Do not splice or
  modify the ThinkLabs cable for this prototype.
- The final robot-side hole pattern still needs measurement from the exact
  OpenArm wrist/end-effector adapter.

## Source Links

- MISUMI `WF6-25` spring search result:
  https://uk.misumi-ec.com/vona2/detail/110100184720/?HissuCode=WF6-25
- McMaster 25.5 mm compression springs:
  https://www.mcmaster.com/products/springs/length~25-5-mm/
- Honeywell SS49E at Digi-Key:
  https://www.digikey.com/en/products/detail/honeywell-sensing-and-productivity-solutions/SS49E/701361
- TI DRV5053 product page:
  https://www.ti.com/product/DRV5053
- Seeed XIAO RP2040:
  https://wiki.seeedstudio.com/XIAO-RP2040/
- 5 mm x 2 mm neodymium magnet example:
  https://suprememagnets.com/products/neodymium-magnet-5x2mm-disc
- Adafruit 30 AWG silicone wire:
  https://www.adafruit.com/product/3166
- Adafruit JST-PH 2.0 3-pin cable pair:
  https://www.adafruit.com/product/5087
