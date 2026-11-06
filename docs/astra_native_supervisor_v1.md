# Astra native supervisor v1

This is an opt-in **Astra supervisor + analytic GivenContactExpert local motor skills** arm. Astra reads one approved RGB-D camera, joints and calibrated simulated Hall ADC feedback and selects actually dispatched skills at safe boundaries. It does not generate motor actions, independently estimate an anatomical target from depth, or replace the original local force controller. CPU tests and the synthetic client smoke are not native physical qualification.

## Native entry point and paired configuration

Use the existing owner entry point:

```sh
python -m sim.ausc_policy.run --config /absolute/path/astra-case.json --run-id astra-case-unique-id
```

Start from the same original single-target `experiment: geometry_qualification` / `fresh_expert` / `m2_given_contact_control` configuration used by the analytic arm. Merge [the machine-readable delta](../configs/astra_supervisor_given_contact_v1.merge.json) at the top level. Keep the scene, initial home, physics, body, prescribed reference, plan, preload and controller settings identical across arms. Use separate output directories. No teleport, state setter, or mid-episode reset is introduced.

The given-contact route must retain `schema: m2_given_contact_pose_control_v1`, its original hash-bound `metadata`, an explicit `episode_id`, and `handoff_policy_id: m2_given_interior_parking_handoff_v3` at both route and metadata levels. The native plan must have the original 23 mm local plane and 3 degree limit; relaxed 10 degree plans are rejected. The original 25-pose / 24-interval parking, speed/path/rotation/5-second budget, joint/collision/return checks, continuous normal 3.5–4.5 N for at least 3 seconds, total force at most 6 N, Hall below 5.5 N, stroke at most 6 mm and zero forbidden forces remain in the original executor and independent audit. The supervisor cannot change these values. The existing optional narrower Hall proxy HOLD band also remains unchanged.

Do not add ACT/deployment/recovery/formal-evaluation route keys to the Astra arm. Expert raw/compact training capture is rejected because its labels would misattribute supervised motion. The default old routes are unchanged when `astra_supervisor` is absent. This arm is a new diagnostic route and does not claim the historical formal cohort qualification.

## CPU inference service and tunnel

On the authenticated Mac, run this source checkout with Python (the service itself uses only the standard library):

```sh
python3 -m sim.ausc_policy.astra_service --port 8123
```

Supply a fresh random `AUSC_ASTRA_RUN_TOKEN` environment variable, at least 24 characters, to the Mac service and Linux native owner. It is a per-run bridge capability, never an API/account token. Keep it out of configuration JSON and logs. Bind a reverse SSH tunnel from remote `127.0.0.1:8123` to Mac `127.0.0.1:8123`. The service only binds loopback and requires the token on every request. Never copy `auth.json` to the Pod.

The service invokes the already authenticated bundled Codex CLI with exact model `gpt-6-astra`, `--ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check --sandbox read-only --json --output-schema ... -o ... -`. It has a 120-second subprocess timeout, no model fallback or retries, and structured last-response validation. The explicit `--codex` argument can select an equivalent authenticated CLI location. Temporary image/schema/last-response files are removed after each call, including failure. Prompts and image bytes are not logged on the Mac by this service. Existing Codex account state remains local.

## Request and dispatch contract

`POST /decision` with `Authorization: Bearer <per-run-token>` accepts exactly:

```json
{"observation": {"schema": "ausc_astra_native_observation_v1"}, "rgb_png_base64": "..."}
```

The complete observation is validated by `astra_supervisor.validate_observation`. It includes episode and observation IDs, current native physics row/time, q/dq, phase, compression, the hash-bound prescribed full pose, RGB SHA256 and `sensor_contract: ausc_astra_rgbd_hall_adc_v1`. Requests are bounded to 2 MiB, decoded RGB to 1 MiB, and responses to 128 KiB. The service remains standard-library-only; it receives summaries and the compressed RGB preview, never raw numeric depth arrays.

The native sensor extension uses the existing `M2CameraSampler(env, camera_bytes=..., native_image_size=(640,360))` and `capture(state) -> (rgb, depth_m, depth_valid)` API. Its owned RGB/depth render product points to the same single approved `/World/RobotCam` sensor as the diagnostic video. The exact annotators are `rgb` and `distance_to_image_plane`; the depth is metric optical Z in meters, not Euclidean range. The existing backend renders with `rep.orchestrator.step(rt_subframes=2, delta_time=0., pause_timeline=False)` and rejects changed world time, physics time, measured q/dq/compression, camera transform, optics, projection or stage units during setup/read/render. It never advances physics to obtain a sensor reading.

Exact 640x360 raw depth, original invalid values and boolean validity masks are stored only under Linux `astra/sensors/frame_0000.npz`, alongside the corresponding RGB. Astra receives a finite 32x18 valid-pixel median depth grid, per-cell validity counts, global depth range/quantiles, units, native intrinsics/extrinsics and artifact hashes. NaN, infinity and nonpositive depths are invalid; empty cells become JSON null. An entirely invalid map rejects the observation. An optional backprojected depth probe is explicitly selected by projecting the prescribed GT pose; it is labeled `anatomical_target_estimator: false` and does not replace the prescribed reference. This gives depth-derived measurements without claiming independent visual target recognition.

The passive Hall observer uses the existing `HallSimulator.update(time_s, measured_compression_m)`, `read(time_s)` and `HallCalibration.decode(packet, now_s=...)`. It observes the initial state and each already-executed 240 Hz native row; the simulator emits nominal 100 Hz RP2040-style quantized/averaged u16 packets. `hall_voltage_v` and `hall_force_N` in the Astra observation are the voltage and spring-load estimate decoded from those **simulated_adc** packets. They are not hardware ADC readings, physical normal-force labels or bench-verified patient pressure. The immutable calibration SHA is `ce6bcda3bef2282e07be0f44686dc007756ede57cdfcd732d35e28c1d469d054`: its endpoint voltages were measured, while the operating force/displacement curve is spring-anchored and `bench_verified` remains false. The original local force controller still reads its original native axial proxy and all force/safety thresholds remain unchanged. Per-row Hall evidence and a since-boundary voltage/decoded-force summary preserve what happened during each local skill without blocking active force control.

The evaluation observer shares `hall_adc_boundary.py` with the existing compact capture path. This extracts the already-used `deterministic_adc_equivalent_lower_endpoint_v1` rule without changing its bound: a half native 12-bit ADC LSB and the pinned tare slope give exactly `3.809856306074158e-6 m`. A negative lower-stop solver residual within that interval maps to zero **only as the forward sensor-model input**, preserving the same zero-travel ADC code. Raw measured compression, native state, physical gates and the upper travel domain remain untouched; values beyond the existing lower bound still reject. Every Hall evidence row retains raw compression, sensor compression and their delta. The sensor manifest and teardown receipt retain the helper source SHA, original policy, conversion count and maximum delta. The Mac service/protocol bytes and observation contract remain unchanged.

Each model-visible boundary bundle also carries explicit interpretation fields. The configured nominal 1.5 N preload is bound to the existing runtime configuration, passive-preload authoring receipt and live USD startup readback by SHA256; it is not an applied-effort measurement. A decoded rest value near 1.5 N includes spring preload and does not establish patient normal contact. The decoded value is retained without subtracting preload. Existing `WARN` classification can mean an off-target force at unloaded rest; it is not itself a screened/hard-limit trip. The payload states the original classifier's target window and screened/hard thresholds without changing any of them. The prescribed-pose-projected depth pixel may instead show the arm, tool or other foreground surface: finite positive pixel validity never verifies an anatomical patient surface. These factual fields are in the sensor manifest and observation payload; the inference service and protocol SYSTEM bytes stay unchanged.

Camera, Hall simulator, host decoder, calibration and sensor-observer source SHAs are saved in the native sensor manifest and original runtime source inventory. Sensor time, physics-row identity, simulated provenance, decoded voltage/estimate and depth units/mask summary are validated before dispatch. Both camera annotators are finalized before the world stops; a sensor read or teardown failure revokes acceptance. These APIs are source-verified and CPU-tested here; actual Isaac annotator output and native clock preservation still require a Linux run. Restart the Mac service after deploying updated protocol source so its source hashes match Linux.

The service returns `model: gpt-6-astra`, `backend: codex_exec`, inference wall time, filtered turn events including token usage, and a strict decision with exactly `observation_id`, `skill`, `target_id`, `reason`. The three safe boundaries are:

| Boundary | Motion choice | Existing executed skill |
| --- | --- | --- |
| HOME | approach_target | GivenContactExpert L2, strict measured parking |
| PARKED | continue_local_control | Full measured ALIGN + original contact/HOLD/release |
| RELEASED | retract | Existing checked return home |

Each boundary also permits `abort`, which stops simulation and leaves the episode unsuccessful. A missing image, model/transport error, stale ID, changed physics state, unexpected target/field or illegal skill fails closed without invoking another local motion skill. There is no remote inference during the contact skill. Native physics state/rows are checked unchanged across observation and inference; simulation time pauses while inference wall time elapses.

Authoritative Linux artifacts are written before/after each call under `output/.../astra/0000/`: `observation.json`, `rgb.png`, `request.json`, `proposal.json`, `execution.json`, and `record.json`. `astra/sensors/` holds exact RGB-D NPZs, Hall packet/decoder ledgers, pinned camera/calibration bytes, source manifest and teardown receipt; raw depth/masks never enter the HTTP payload or Mac. Records bind request/observation/image hashes and retain proposals separately from execution acknowledgments. Acknowledgments carry actual physics row ranges, completed/failed/aborted/rejected status and failure reasons. Skill completion does not set `physical_success`; the original independent native audit decides physical acceptance. Native rows retain their original motor `command_source_id` and add Astra decision/model IDs plus explicit analytic motor assistance. Results are excluded from expert training data and never labeled as an ACT learned motor policy.

## Verification scope

CPU tests cover decision identity/phase/target admission, real skill dispatch and refusal, failure acknowledgments, no physics advance during inference, preserved motor provenance, loopback token/size checks, HTTP image binding, pinned CLI invocation and temporary-file cleanup. A synthetic real Astra CLI smoke returned an explicit abort for missing RGB; it proves model-client invocation/structured parsing only. Native RGB/model dispatch, original physical audits and paired Astra/ACT/analytic results must be executed on the rebuilt Linux runtime before any model or physical success claim.
