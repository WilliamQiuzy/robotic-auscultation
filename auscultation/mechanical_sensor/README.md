# ThinkLabs One passive force mount

This folder contains a first-pass CAD concept for the auscultation robot's end-effector module.  The design is intentionally simple and parametric:

- fixed printed frame: robot-side top plate, guide-rod supports, spring seat, and displacement-sensor rail
- floating printed carriage: split circular side cradle for the ThinkLabs One, two guide bushings, spring seat, and magnet pocket
- non-print proxies: ThinkLabs One body/contact envelope, two 6 mm guide rods, and compression-spring envelope

The mechanism follows the passive-force idea in `passive_mechanism_sample.png`: the robot positions the top frame, while the stethoscope carrier can slide along the contact axis against a compression spring.  Force can be estimated from displacement after calibration: `F = k * x`.

In the preview image, tan is the robot-side fixed frame, blue is the moving printed cradle, and dark gray/black is the ThinkLabs One envelope.  The ThinkLabs One does **not** go into the central spring cylinder.  It sits inside the blue circular side cradle.  The bottom dark circular face protrudes below the blue cradle and is the only part intended to touch skin.

## Recommended validated iteration: VEO5

VEO5 is the current preferred concept. It keeps the VEO3/VEO4 clamshell
architecture but fixes four defects found in VEO4 and redesigns the magnet
sensing module:

- clamshell halves now split with a true 3 mm air gap (VEO4's split cutters
  were swapped, so the two blue halves overlapped by ~471 mm³ at the centerline)
- guide bushings, M3 lugs, and the magnet tower are fused into the ring and
  every printed part is audited as a single connected solid (VEO4's halves
  exported as 3-4 loose pieces)
- magnet sensing is coaxial: Z-magnetized D5x2 disc in a pocket on the moving
  ring, SS49E face-down in a pillar on the fixed bridge directly above it; the
  axial air gap closes 1:1 with spring compression so the Hall output is
  monotonic over the full stroke (VEO4's Y-axis magnet crossed the Hall plane
  at 5 mm compression, making the readout non-monotonic), and the gap never
  closes below 1.5 mm even at the coil-bind hard stop
- spring bridge lowered for ~0.9 mm clearance to the bottom coil (was 0.1 mm
  interference), the compressed spring helix is audited against the printed
  parts at 4/8/9.2/11.25 mm travel, and the motion-video spring squash preserves
  seating offsets so the spring no longer sinks into the blue ring visually
- clamp screws are real M3 x 30 SHCS with the head seated on the left lug face
  and the nut in a slotted pocket inside the right lug (they were modeled as
  56 mm rods skewering the spring bridge slab, with the M3 nut buried inside
  the 3 mm plate — the visible blue-plate/screw interpenetration)
- the Hall sensor is modeled as the purchased Honeywell SS49E TO-92S including
  its three ~13 mm straight leads; the pillar leg channel clears them with
  >=0.5 mm margin and the sensor wires splice at the lead tips
- guide rods are standard 5 x 25 mm dowel pins pulled inboard to x=±34 so the
  shortened 14 mm bushing barrels hug the ring (lateral protrusion 24.5 ->
  11.5 mm) and the top plate shrinks from 108 x 48 to 88 x 48; rod tips stay
  7.75 mm above the skin plane even at the coil-bind hard stop (the inherited
  47 mm rods reached the patient's skin at the 4 N working compression — a
  safety defect)
- purchased-part fidelity: spring helix at the true WF6-25 6.0 mm OD with the
  catalog ~20 coils (was 5.2 mm / 10 coils), sensor wires at the real ~1.2 mm
  silicone OD (was 0.76). The optional JST-PH connector was dropped (user
  decision): the three hall wires solder directly onto the XIAO's right-edge
  pads, passing through a notch in the cradle rail; the routing is verified
  against the rising device and magnet tower at 4 compression states
- **Thinklabs One re-modeled from the user's product photos**: flared
  diaphragm ring (official 46 max), an upward-NARROWING cone, a camera-lens
  style ROTATING volume dial on top, and two low side sockets (audio/charge).
  Consequences engineered in: the clamp bore is now CONICAL and
  self-energizing (contact force wedges the cone into the narrowing bore — no
  lip anywhere near the rotating dial, which is audited free at +-0.8 mm),
  drop-out is held by clamp friction (device is 50 g), and a round jack
  window at the 215 deg azimuth lets the 3.5 mm audio plug STAY PLUGGED
  during auscultation (a right-angle plug with the cable turning upward is
  recommended and modeled; verified at 4 travel states). The legacy front
  relief and rear cable window are gone. Cone/dial/socket numbers are PHOTO
  ESTIMATES — measure the real unit before the final print
- guide-rod clearance is now bored through the fused bushing bridge blocks
  (every prior version only bored the bushing cylinder before the union, so
  the rods silently skewered the bridge blocks; rods-vs-ring is now audited
  statically and at 4 compression states)
- XIAO RP2040 rides in a printed rail cradle on a wing off the fixed bridge
  (board has no mounting holes; USB-C faces outward for cable access)
- **OpenArm mount (scheme A — gripper removed)**: the arm changed to the
  open-source **OpenArm** (Enactic). The gripper is removed and the plate bolts
  to an **adapter plate on the OpenArm side**, so the module becomes the
  end-effector. The **MEASURED hole pattern** (user calipers, 2026-06-20) is a
  **rectangle of 4x M3**: centers **18 mm (long edge) x 11 mm (short edge)** ->
  holes at **(±9, ±5.5) mm**, overall envelope ~25 x 18 mm. This **supersedes
  the earlier assumed Ø50 6x M3 circle**, which did not match the real hardware.
  Hole **size unchanged** (M3 screws thread into the existing holes fine): Ø3.5
  clearance + bottom-face Ø6.2 x 3.2 counterbores; screws drive up into the
  adapter taps -> **M3 x 8 x4**. The plate **centre is solid** (no boss relief)
  so it mates flat on the adapter.
- the OpenArm **actuator body behind the adapter is large**, so the XIAO rail
  cradle stays **outboard (y -41, USB-C outward)** to clear it; the -y wing
  extends ~7 mm past the Ø88 plate as a result (can move back to centre if a
  thick adapter spaces the module clear of the actuator — confirm on the bench)
- wires now POSITIVELY join their solder partners in the audit (min
  intersection volume vs the hall leads and the XIAO edge) — a v9.cyl_y
  displacement bug (CadQuery "XZ" plane normal is -Y) once left all three
  wires floating disconnected; y-axis cylinders use veo4.cyl_y_centered
- travel decision (user): keep the in-hand 2x WF6-25 — 4 N @ 4 mm with the
  hard stop at 5.5 mm leaves only 1.5 mm of passive margin, so the robot
  controller MUST servo-track breathing (2-5 mm chest excursion); softer
  springs (2x WY10-25, k 0.4 total -> 13.75 mm stroke) remain the drop-in
  upgrade if passive compliance proves insufficient on the bench
- **moment-free dual-spring suspension**: the single rear spring (46 mm moment
  arm from the device axis) side-loaded the bushings with friction comparable
  to the 4 N working force. TWO WF6-25 springs sit point-symmetric beside the
  guide rods at (+34,+10)/(-34,-10) — force centroid = device axis, zero
  moment. k doubles to 1.0 N/mm (4 N @ 4 mm); hard-stop sleeves on the rod
  bosses cap travel at 5.5 mm so the maximum possible force equals the 5.5 N
  patient limit mechanically; per-spring load at the stop is only 2.75 N
- **round top plate, 88 dia**: the rear spring tab and cantilever bridge are
  gone; the center now carries the OpenArm 4x M3 / 18 x 11 mm rectangle, the
  XIAO wing keeps rounded front corners — the printed bridge is a tidy disc
- Hall pillar deepened so the rest air gap is 7.75 mm, re-sized to the 5.5 mm
  stroke: predicted span ~0.4 V over 0..4.6 N (better than the old 12.75 mm
  gap gave over the longer stroke)

### BOM corrections (vs the original purchase list)

| item | original BOM | corrected |
| --- | --- | --- |
| ThinkLabs One | "66 mm lower / 48 mm upper disc (CAD approx.)" | **official 46 dia x 28 mm, 50 g**; upper-tier 37 dia is an estimate — bench-measure |
| clamp screws | M3 x 12-16 mm | **M3 x 30 mm SHCS x4** (head face x=-17 to nut x=11.5 needs ~28.5 mm grip) |
| guide posts | 5 mm, length unspecified | **2x Ø5 shoulder screw, ~33 mm shoulder, M4 thread (head DOWN, threads up into the plate)** at x=±34. The Ø8 head is the rest stop + drop-out catch under spring preload |
| robot-side screws | "M3, length TBD" | **OpenArm (gripper removed): M3 x 8 SHCS x4** into the adapter plate. Holes in the **measured 18 x 11 mm rectangle** (centers ±9 / ±5.5). Bottom-face counterbores Ø6.2 x 3.2. Hole size unchanged (M3 fits) — only the positions were corrected from the assumed Ø50 circle |
| M2 screws (Hall board) | M2 x 5-8 mm | **not needed** — SS49E is slot-mounted in the pillar, no breakout PCB |
| spring | WF6-25 | **two required (already ordered)** — dual point-symmetric springs (k 1.0 N/mm total), **PRELOADED 1.5 mm at assembly (1.5 N preload)** for a stable zero; each sees ≤2.75 N at the 4 mm hard stop (5.5 N total) |
| XIAO RP2040 | — | **on the mount**: printed rail cradle on the front-center wing, USB-C outward; hall wires solder to its right-edge pads |
| JST-PH 3-pin pigtail | optional | **not needed** — direct soldering replaces the connector |

Generate VEO5 (any Python 3.11 with `cadquery==2.7.0` works):

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache XDG_CACHE_HOME=/private/tmp/cadquery-cache \
  python3.11 auscultation/mechanical_sensor/cad/thinklab_openarm_veo5_coaxial_hall_mount.py
```

Key VEO5 outputs are written to `auscultation/mechanical_sensor/generated/veo5_coaxial_hall/`:

- `thinklab_openarm_veo5_{fixed_bridge,left_clamshell,right_clamshell}_print.step` / `.stl`
- `thinklab_openarm_veo5_assembly.step` / `thinklab_openarm_veo5_preview.step`
- `veo5_{geometry,mechanical,material}_audit.json`, `veo5_design_metadata.json`
- `VEO5_COAXIAL_HALL_REPORT.md` (includes the predicted Hall response table)
- `simulation/veo5_hall_response.{csv,png}`: first-order SS49E output vs compression
- `simulation/veo5_contact_motion.mp4`, `presentation/veo5_actual_cad_motion.mp4`

Regression tests: `tests/test_veo5_coaxial_hall_mount_outputs.py` asserts the
split air gap, single-solid connectivity, compressed-spring clearance, and
monotonic Hall output, so the VEO4 defects cannot silently come back.

## Previous iteration: VEO3

VEO3 keeps the V12 two-piece clamshell
layout, but the CAD generator now produces a validation package instead of only
visual STEP files. The VEO3 script checks:

- geometry: static and max-compression CadQuery boolean collision checks
- mechanics: spring force, travel, screened contact force, and spring deflection margin
- material: first-order printed-part stress estimates and safety factors for a PETG/PA12 baseline

Generate VEO3:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache XDG_CACHE_HOME=/private/tmp/cadquery-cache \
  /private/tmp/wam-cadquery-venv/bin/python \
  auscultation/mechanical_sensor/cad/thinklab_openarm_veo3_validated_mount.py
```

Key VEO3 outputs are written to `auscultation/mechanical_sensor/generated/veo3_validated/`:

- `thinklab_openarm_veo3_fixed_bridge_print.step` / `.stl`
- `thinklab_openarm_veo3_left_clamshell_print.step` / `.stl`
- `thinklab_openarm_veo3_right_clamshell_print.step` / `.stl`
- `thinklab_openarm_veo3_assembly.step`
- `thinklab_openarm_veo3_preview.step`
- `veo3_geometry_audit.json`
- `veo3_mechanical_audit.json`
- `veo3_material_audit.json`
- `veo3_design_metadata.json`
- `VEO3_VALIDATED_REPORT.md`
- `simulation/veo3_contact_motion.mp4`
- `simulation/veo3_contact_response.png`
- `simulation/veo3_contact_summary.json`
- `presentation/veo3_cad_clean.png`: clean CAD render for slides.
- `presentation/veo3_clamshell_detail.png`: clamshell and sensor detail render.
- `presentation/veo3_actual_cad_motion.mp4`: report video using the current VEO3 CAD geometry.

Current VEO3 assumptions remain approximate until the physical device is
measured: lower bottom diameter `66 mm`, lower top diameter `56 mm`, upper disc
diameter `48 mm`, `11 mm` travel, MISUMI `WF6-25` spring, two `5 mm` guide rods,
four total M3 clamshell screws, and Hall sensor plus magnet force feedback.
The VEO3 validation is an engineering screen for CAD feasibility; final use
still requires measurement, calibration, and physical load testing.

## Previous clamshell iteration: v12

V12 is the previous preferred concept. It keeps the V11 stepped ThinkLabs One
profile, but replaces the single split ring with two separately printed blue
clamshell halves. The halves are split on `X=0` and bolted together at both the
front and rear seams with M3 screws, so the printed plastic is not used as a
living hinge. No silicone, TPU, or other soft pad is required.

Generate v12:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache XDG_CACHE_HOME=/private/tmp/cadquery-cache \
  /private/tmp/wam-cadquery-venv/bin/python \
  auscultation/mechanical_sensor/cad/thinklab_openarm_v12_clamshell_mount.py
```

Key v12 outputs are written to `auscultation/mechanical_sensor/generated/v12_clamshell/`:

- `thinklab_openarm_v12_fixed_bridge_print.step` / `.stl`
- `thinklab_openarm_v12_left_clamshell_print.step` / `.stl`
- `thinklab_openarm_v12_right_clamshell_print.step` / `.stl`
- `thinklab_openarm_v12_clamshell_assembly.step`
- `thinklab_openarm_v12_clamshell_preview.step`
- `thinklab_openarm_v12_clamshell_preview.png`
- `thinklab_v12_stepped_profile_proxy.step`
- `V12_CLAMSHELL_REPORT.md`
- `simulation/v12_contact_motion.mp4`
- `simulation/v12_contact_response.png`
- `simulation/v12_contact_summary.json`
- `v12_force_feedback_calibration.json`
- `presentation/v12_cad_clean.png`: clean CAD render for slides.
- `presentation/v12_clamshell_detail.png`: clamshell split and two-sided M3 detail render.
- `presentation/v12_actual_cad_motion.mp4`: report video using the current V12 CAD geometry.

Current v12 assumptions remain approximate until the physical device is
measured: lower bottom diameter `66 mm`, lower top diameter `56 mm`, upper disc
diameter `48 mm`, `11 mm` travel, MISUMI `WF6-25` spring, two `5 mm` guide rods,
four total M3 clamshell screws, and Hall sensor plus magnet force feedback.

## Recommended stepped-profile iteration: v11

V11 is the previous stepped-profile concept because it corrects the ThinkLabs One
geometry. The stethoscope proxy is no longer a plain cylinder: it is modeled as
a larger lower frustum plus a smaller upper disc/control face, with front jack
proxies and a rear cable stub. The blue cradle is also changed from a
single-diameter clamp to a stepped holder: the lower lip supports the larger
bottom rim, and three upper retaining pads lightly capture the smaller upper
disc/shoulder without blocking the front jacks or rear cable.

Generate v11:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache XDG_CACHE_HOME=/private/tmp/cadquery-cache \
  /private/tmp/wam-cadquery-venv/bin/python \
  auscultation/mechanical_sensor/cad/thinklab_openarm_v11_stepped_thinklabs_mount.py
```

Key v11 outputs are written to `auscultation/mechanical_sensor/generated/v11_stepped_thinklabs/`:

- `thinklab_openarm_v11_fixed_bridge_print.step` / `.stl`
- `thinklab_openarm_v11_stepped_cradle_print.step` / `.stl`
- `thinklab_openarm_v11_stepped_assembly.step`
- `thinklab_openarm_v11_stepped_preview.step`
- `thinklab_openarm_v11_stepped_preview.png`
- `thinklab_v11_stepped_profile_proxy.step`
- `V11_STEPPED_THINKLABS_REPORT.md`
- `simulation/v11_contact_motion.mp4`
- `simulation/v11_contact_response.png`
- `simulation/v11_contact_summary.json`
- `v11_force_feedback_calibration.json`
- `presentation/v11_cad_clean.png`: clean CAD render for slides.
- `presentation/v11_profile_clamp_detail.png`: side-profile and stepped-clamp detail render.
- `presentation/v11_actual_cad_motion.mp4`: report video using the current V11 CAD geometry.

Current v11 assumptions are still approximate until the physical device is
measured: lower bottom diameter `66 mm`, lower top diameter `56 mm`, upper disc
diameter `48 mm`, `11 mm` travel, MISUMI `WF6-25` spring, two `5 mm` guide rods,
two M3 clamp screws, and Hall sensor plus magnet force feedback.

## Recommended detailed iteration: v10

V10 is the previous detailed concept. It keeps the simple two-rail/single-spring
V9 architecture, but draws the practical assembly details that were ambiguous:
two visible M3 clamp screws across the blue split clamp, nut pockets, spring top
and bottom seats, visible green spring coil, red magnet pocket on the moving
blue cradle, purple Hall sensor on the fixed side, a small local sensor
connector with short wires, and a top robot flange proxy with four bolt slots.

Generate v10:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache XDG_CACHE_HOME=/private/tmp/cadquery-cache \
  /private/tmp/wam-cadquery-venv/bin/python \
  auscultation/mechanical_sensor/cad/thinklab_openarm_v10_detailed_clamp_mount.py
```

Key v10 outputs are written to `auscultation/mechanical_sensor/generated/v10_detailed/`:

- `thinklab_openarm_v10_fixed_bridge_print.step` / `.stl`
- `thinklab_openarm_v10_blue_cradle_print.step` / `.stl`
- `thinklab_openarm_v10_detailed_assembly.step`
- `thinklab_openarm_v10_detailed_preview.step`
- `thinklab_openarm_v10_detailed_preview.png`
- `V10_DETAILED_REPORT.md`
- `simulation/v10_contact_motion.mp4`
- `simulation/v10_contact_response.png`
- `simulation/v10_contact_summary.json`
- `v10_force_feedback_calibration.json`
- `presentation/v10_cad_clean.png`: clean CAD render for slides.
- `presentation/v10_detail_layout.png`: detail layout showing clamp, spring, sensor, and robot interface.
- `presentation/v10_actual_cad_motion.mp4`: report video using the current V10 CAD geometry.

Current v10 assumptions: `62 mm` ThinkLabs body diameter, `11 mm` travel,
MISUMI `WF6-25` spring (`6 mm OD`, `25 mm` free length, `0.5 N/mm`,
`5.5 N` max load), `4 N` target contact force, two `5 mm` guide rods, two
M3 clamp screws, and Hall sensor plus magnet force feedback. The top flange is
a non-print placeholder for the OpenArm interface and should be replaced with
the measured robot end-effector pattern.

## Recommended simplified iteration: v9

V9 is the previous simplified candidate. It removes the bulky v8B printed slide
frame and keeps only the essentials: a blue split cradle that clamps the
ThinkLabs One, two vertical guide rods, one MISUMI `WF6-25` compression spring,
two M3 clamp screws, and a Hall sensor/magnet pair for displacement-based force
feedback. The tan printed part is now only a small robot-side bridge for the
rod seats, spring top seat, and generic robot mounting slots.

Generate v9:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache XDG_CACHE_HOME=/private/tmp/cadquery-cache \
  /private/tmp/wam-cadquery-venv/bin/python \
  auscultation/mechanical_sensor/cad/thinklab_openarm_v9_simplified_rail_mount.py
```

Key v9 outputs are written to `auscultation/mechanical_sensor/generated/v9_simplified/`:

- `thinklab_openarm_v9_fixed_bridge_print.step` / `.stl`
- `thinklab_openarm_v9_blue_cradle_print.step` / `.stl`
- `thinklab_openarm_v9_simplified_assembly.step`
- `thinklab_openarm_v9_simplified_preview.step`
- `thinklab_openarm_v9_simplified_preview.png`
- `V9_SIMPLIFIED_REPORT.md`
- `simulation/v9_contact_motion.mp4`
- `simulation/v9_contact_response.png`
- `simulation/v9_contact_summary.json`
- `v9_force_feedback_calibration.json`
- `presentation/v9_cad_clean.png`: clean no-label CAD render for slides.
- `presentation/v9_simple_schematic.png`: simple no-label schematic for slides.
- `presentation/v9_actual_cad_motion.mp4`: report video using the current V9 CAD geometry.

Current v9 assumptions: `62 mm` ThinkLabs body diameter, `11 mm` travel,
MISUMI `WF6-25` spring (`6 mm OD`, `25 mm` free length, `0.5 N/mm`,
`5.5 N` max load), `4 N` target contact force, two `5 mm` guide rods, and
Hall sensor plus magnet force feedback. The XIAO RP2040 board and longer wires
are external electronics and are intentionally omitted from the mechanical STEP.

## Recommended compact iteration: v8B

V8B is the previous compact iteration. It removes the v6 pair of long 6 mm guide rods and keeps one small standard compression spring so the force constant can still be calibrated. Printed slide cheeks provide short-travel anti-rotation. The ThinkLabs contact face remains the lowest part, so the printed holder should not touch the patient.

Generate v8B:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache /private/tmp/wam-cadquery-venv/bin/python \
  auscultation/mechanical_sensor/cad/thinklab_openarm_v8b_compact_mount.py
```

Key v8B outputs are written to `auscultation/mechanical_sensor/generated/v8b_compact/`:

- `thinklab_openarm_v8b_fixed_frame_print.step` / `.stl`
- `thinklab_openarm_v8b_floating_carriage_print.step` / `.stl`
- `thinklab_openarm_v8b_compact_mount_preview.png`
- `V8B_COMPACT_REPORT.md`
- `simulation/v8b_contact_motion.mp4`
- `simulation/v8b_contact_response.png`
- `simulation/v8b_contact_summary.json`
- `v8b_force_feedback_calibration.json`
- `presentation/v8b_cad_clean.png`: clean no-label CAD render for slides, including non-print wiring proxies.
- `presentation/v8b_simple_schematic.png`: simple no-label schematic for slides, including sensor/audio wiring.
- `presentation/v8b_actual_cad_motion.mp4`: report video using the current V8B CAD geometry and wiring proxies.

Current v8B assumptions: `11 mm` travel, MISUMI `WF6-25` nominal mini-spring (`6 mm OD`, `25 mm` free length, `0.5 N/mm`, `5.5 N` max load), `4 N` target contact force, and Hall sensor plus magnet for computer-readable displacement/force feedback.

For reporting, use `presentation/v8b_actual_cad_motion.mp4`. The older
`simulation/v8b_contact_motion.mp4` remains a force-response handoff animation
and does not render the actual CAD mesh.

The V8B assembly/preview CAD includes non-print wiring proxies: red Hall VCC,
yellow Hall signal, black Hall ground, a small external XIAO RP2040 board proxy,
and a black ThinkLabs audio cable exiting the cradle cable window. The two
print-part STEP files remain clean and do not include electronics or cables.

V8B non-print BOM and force-feedback wiring are documented in
`BOM_V8B_COMPACT.md`. A starter microcontroller sketch is available at
`force_feedback/v8b_hall_reader/v8b_hall_reader.ino`; it streams JSON lines that
`force_feedback/force_feedback_monitor.py` can read directly.

## Generate CAD

```bash
python -m venv /tmp/wam-cadquery-venv
/tmp/wam-cadquery-venv/bin/pip install -r auscultation/mechanical_sensor/cad/requirements.txt
/tmp/wam-cadquery-venv/bin/python auscultation/mechanical_sensor/cad/thinklab_openarm_passive_mount.py
```

Generated files are written to `auscultation/mechanical_sensor/generated/`.

Key outputs:

- `thinklab_openarm_fixed_frame_print.step` / `.stl`: print this as the robot-side fixed frame.
- `thinklab_openarm_floating_carriage_print.step` / `.stl`: print this as the moving ThinkLabs cradle.
- `thinklab_openarm_passive_force_mount_preview.step`: open this for the full mechanical preview.
- `thinklab_openarm_passive_force_mount_preview.png`: quick visual preview generated every time the CAD script runs.
- `thinklab_openarm_passive_force_mount_assembly.step`: colored assembly-style STEP, if your CAD tool supports it.
- `thinklab_one_envelope_proxy.step`: non-print proxy for checking holder clearance.

## Current assumptions

- ThinkLabs One body is approximated as a 62 mm diameter, 24 mm tall circular device.
- The robot-side plate bolts to an adapter plate on the OpenArm (4x M3 in a measured 18 x 11 mm rectangle, scheme A: gripper removed).
- The guide axis uses two 6 mm smooth rods and one central compression spring.
- The side sensor rail is sized as a placeholder for a small linear potentiometer or Hall sensor/magnet pair.
- Current default CAD is the combined local-simulation winner `v6_balanced_high_clearance`: 24 mm travel, 16 mm contact-face projection, and a recommended spring rate near 0.38 N/mm.

## Run local contact simulation

The repository's high-fidelity simulator is Isaac Sim/Isaac Lab under `sim/`, which requires the cloud GPU workflow described in `sim/README.md`.  For fast local iteration, run the deterministic contact-response screen:

```bash
/private/tmp/wam-cadquery-venv/bin/python auscultation/mechanical_sensor/sim/contact_response_sim.py
```

Outputs are written to `auscultation/mechanical_sensor/generated/simulation/`:

- `SIM_REPORT.md`: variant ranking and limitations.
- `summary_scores.csv`: one-row-per-variant score table.
- `all_trials.csv`: all tissue/speed trials.
- `force_profiles_medium_30mms.png`: force and spring-compression curves.
- `variant_score_summary.png`: score, peak force, and travel-use comparison.
- `robustness_trials.csv`: Monte Carlo tolerance/off-axis/friction trials.
- `robustness_summary.csv`: robust ranking after geometry penalties.
- `robustness_summary.png`: robust score, P95 peak force, binding-risk, and seal-risk comparison.
- `contact_patch_trials.csv`: 2D tilted/off-center contact-patch trials.
- `contact_patch_summary.csv`: pressure, contact-area, centroid, moment, and tilted-clearance ranking.
- `contact_patch_summary.png`: 2D contact-patch comparison.
- `contact_patch_pressure_maps_final.png`: pressure maps for the final recommended candidate.
- `combined_recommendation.csv`: final combined ranking across deterministic, robustness, and 2D patch screens.

## Isaac Sim handoff

The local simulator is a screening tool, not the final contact proof.  Once the repo's cloud Isaac Sim instance is available, run the handoff scene:

```bash
/isaac-sim/python.sh auscultation/mechanical_sensor/sim/isaac_mount_contact_scene.py \
  --metadata auscultation/mechanical_sensor/generated/design_metadata.json \
  --out auscultation/mechanical_sensor/generated/isaac_contact \
  --headless
```

Add `--render-preview` only when the Isaac machine has a working render stack and you want a camera preview. The default headless path disables viewport updates and still writes the force-response PNG.

Expected Isaac outputs:

- `isaac_contact_trace.csv`: commanded approach, contact-face position, compression, and contact force over time.
- `isaac_contact_summary.json`: peak force, final force, and max spring compression.
- `isaac_contact_response.png`: force, spring-compression, and contact-position curves from the run.
- `isaac_contact_motion.mp4`: synchronized motion video with live force and compression curves.
- `isaac_contact_preview.png`: optional rendered scene preview when Replicator/image output is available.

Use the current STEP/STL files plus `design_metadata.json` as the import package for a later full rigid-body/contact-sensor Isaac validation pass.  The handoff script builds a simplified Isaac scene and logs the same passive contact model in Isaac time; it is meant to make the cloud test reproducible, not to replace final high-fidelity validation.

## Generate calibration and acceptance package

After running the CAD and simulation scripts, generate the pre-hardware calibration gates:

```bash
/private/tmp/wam-cadquery-venv/bin/python auscultation/mechanical_sensor/sim/generate_acceptance_package.py
```

Outputs are written to `auscultation/mechanical_sensor/generated/acceptance/`:

- `ACCEPTANCE_REPORT.md`: practical go/no-go gates before robot testing.
- `acceptance_gates.json`: machine-readable force, travel, clearance, and command limits.
- `force_displacement_calibration.csv` / `.png`: displacement-to-force calibration target.
- `robot_command_envelope.csv` / `.png`: estimated robot overdrive for soft, medium, firm, and bony tissue.

## Run local MuJoCo contact smoke test

MuJoCo can run locally on macOS and gives an actual physics-engine contact check before the cloud Isaac Lab pass:

```bash
/opt/homebrew/bin/python3.11 -m venv /private/tmp/wam-mujoco-venv
/private/tmp/wam-mujoco-venv/bin/python -m pip install -r auscultation/mechanical_sensor/sim/requirements-mujoco.txt
/private/tmp/wam-mujoco-venv/bin/python auscultation/mechanical_sensor/sim/mujoco_mount_contact_sim.py
```

Outputs are written to `auscultation/mechanical_sensor/generated/mujoco/`:

- `MUJOCO_REPORT.md`: physics-engine contact smoke-test report.
- `mujoco_contact_trace.csv`: per-step command depth, compression, spring force, and MuJoCo contact force.
- `mujoco_contact_summary.csv` / `.json`: scenario-level pass/fail metrics.
- `mujoco_contact_profiles.png`: force and compression traces.
- `mujoco_contact_summary.png`: steady force, peak force, travel, and contact-presence comparison.
- `mujoco_random_sweep.csv`: randomized MuJoCo compression-gated robustness scenarios.
- `mujoco_random_sweep_summary.json`: aggregate randomized-sweep pass/fail metrics.
- `mujoco_random_sweep.png`: distributions for peak force, force error, travel use, and tilt/speed response.
- `mujoco_scene_*.xml`: generated MuJoCo scene files for each contact scenario.

## Generate Isaac/Omniverse USD handoff

The USD handoff scene is a simplified physics proxy for Isaac/Omniverse. It carries the v6 CAD metadata, MuJoCo evidence, simple colliders, a dynamic sliding carrier, and a prismatic passive-axis joint:

```bash
/private/tmp/wam-mujoco-venv/bin/python -m pip install -r auscultation/mechanical_sensor/sim/requirements-usd.txt
/private/tmp/wam-mujoco-venv/bin/python auscultation/mechanical_sensor/sim/export_isaac_usd_scene.py
```

Outputs are written to `auscultation/mechanical_sensor/generated/isaac_usd/`:

- `thinklabs_v6_passive_mount_physics.usda`: Isaac/Omniverse physics proxy scene.
- `isaac_usd_handoff_manifest.json`: generated artifact manifest and carried MuJoCo metrics.
- `thinklabs_v6_usd_handoff_preview.png`: quick visual preview of the proxy scene and force gates.
- `ISAAC_USD_HANDOFF_REPORT.md`: import notes and suggested Isaac contact-sensor validation path.

## Prototype hardware

- 2x 6 mm smooth guide rods, about 90-100 mm long.
- For v8B: 1x MISUMI `WF6-25` compression spring, `6 mm OD`, `25 mm` free length, `0.5 N/mm`. Older v6 prototypes used larger springs near `0.38 N/mm`.
- 1x M3 pinch screw and nut/insert for the ThinkLabs cradle split clamp.
- 4x M3 x 8 SHCS for the OpenArm DM-J4310 output-face mount (scheme A, gripper removed).
- Optional linear potentiometer or Hall sensor plus magnet for displacement-to-force calibration.

Before printing for hardware use, measure the actual ThinkLabs One diameter/height, cable exit, the OpenArm DM-J4310 output-face hub (for a centering pilot), and target spring constant.

## Force feedback v7

The recommended next hardware iteration is v7A: keep the passive spring carriage, add a magnet to the moving carriage, and mount a linear Hall sensor on the fixed frame. The sensor cable routes to a microcontroller, and the microcontroller streams displacement over USB serial to the computer. The ThinkLabs One audio cable remains separate and does not need to be modified.

Force feedback files:

- `force_feedback/FORCE_FEEDBACK_PLAN.md`: sensor variants, wiring, calibration, and acceptance gates.
- `force_feedback/force_feedback_calibration.json`: current nominal spring/displacement calibration.
- `force_feedback/force_feedback_monitor.py`: replay or USB-serial monitor that converts displacement to force.

Replay the current Isaac trace as a computer-side force readout:

```bash
/private/tmp/wam-mujoco-venv/bin/python auscultation/mechanical_sensor/force_feedback/force_feedback_monitor.py \
  --calibration auscultation/mechanical_sensor/force_feedback/force_feedback_calibration.json \
  --replay-csv auscultation/mechanical_sensor/generated/isaac_contact_vast/isaac_contact_trace.csv \
  --out auscultation/mechanical_sensor/generated/force_feedback/replay_force_feedback_log.csv
```

For real hardware, stream either JSON lines such as `{"t_ms": 1234, "disp_mm": 10.52}` or CSV lines such as `1234,10.52` from the microcontroller. The monitor writes a CSV log plus a `.summary.json` file.
