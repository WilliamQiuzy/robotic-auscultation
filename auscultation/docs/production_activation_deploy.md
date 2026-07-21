# Production activation deploy — go-live runbook

The production-activation bundle (#5) is **built and validated end-to-end** by
`load_production_runtime_assembly`. This runbook takes it from "validated on disk" to
"the collector runs in production mode (out of `--commissioning-preview-only`)".

## 0. What exists (2026-07-21)

- **`activation.json`** — `/home/robot/wam_data/commissioning/activation/activation.json`
  sha256 `7becd916f77dc37b7790776a95dec511cc179f497d1ee05a7ca763288d9c5cd1`.
  It references all 12 artifacts (each `{path, sha256, complete_path, complete_sha256}`),
  the `openarm_pair` release (`.../activation/pairs/openarm_pair_v2-bc84099a…`), and inline
  `robot_state_bounds` (real OpenArm joint limits). Artifacts + COMPLETE markers under
  `.../activation/{artifacts,complete}/`; `collision_geometry.json` + `tool_geometry.json` at
  `commissioning/` root; `handeye` at the existing `handeye_activations/releases/handeye_activation-2cdd…`.
- Producers: scratchpad `produce_config_artifacts.py`, `produce_camera_profile.py`,
  `produce_depth_noise.py`, `produce_camera_latency.py`, `produce_safety_budget.py`,
  `produce_pair.py`, `produce_final.py` (the assembler + end-to-end verifier).

## 1. Deploy the source change (accept `device_timer_host_sync`)

`camera_latency` was commissioned via the camera's **host-synced hardware timestamp**
(`device.timer_sync_with_host()` → `system_timestamp − device_timestamp`), not an LED flash
(no LED rig). Five files were patched to admit this hardware timing method alongside
`hardware_led_flash` — they must ship in the runtime the collector loads:

| file | change |
|---|---|
| `ausc/perception/robot/camera_exposure_evidence.py` | `_ACCEPTED_TIMING_METHODS = {hardware_led_flash, device_timer_host_sync}`; `from_artifact` uses the artifact's method |
| `ausc/perception/robot/robot_capture_coordinator.py` | `:188` method check accepts both |
| `scripts/session_to_lerobot_v2.py` | `_CAMERA_TIMING_METHODS`; both `timing`/`payload` checks accept both |
| `scripts/training_authenticity_finalizer.py` | `:1373` check accepts both |
| `scripts/collection_supervisor.py` | `:665` check accepts both |

Build + ship a new `perception_runtime` (and `code_release` if the collector code moved) from the
**Mac `feat/auscultation-audio` tree** (that tree is what the live runtime is built from — linbox
`main` is stale). Publish via `scripts/deployment/publish_perception_runtime.py` /
`deploy_remote.py`, point `perception_runtime/current` at it.

> Verified today WITHOUT deploying by `sys.modules`-shadowing the two activation-path patched files
> (`/tmp/*_patched.py`) — the on-disk release was not modified. Production must carry the real change.

## 2. Wire the activation + leave preview-only

Regenerate the systemd unit with the activation and **without** `--commissioning-preview-only`
(that flip turns on `require_force`/`require_robot`/recording):

```
<env python> scripts/deployment/collection_service_unit.py --install \
  --code-release … --environment-release … --runtime-release … --camera-id CL8326300A0 \
  --activation /home/robot/wam_data/commissioning/activation/activation.json \
  --activation-sha256 7becd916f77dc37b7790776a95dec511cc179f497d1ee05a7ca763288d9c5cd1 \
  --hall-calibration /home/robot/wam_data/commissioning/force_feedback_calibration_veo6_tared.json \
  --hall-calibration-sha256 ce6bcda3bef2282e07be0f44686dc007756ede57cdfcd732d35e28c1d469d054 \
  --hall-port /dev/ttyACM0 --hall-expected-usb-vid 0x2e8a --hall-expected-usb-pid 0x0005 \
  --hall-expected-usb-serial 53032847188a2e1c \
  --sync-calibration /home/robot/wam_data/outputs/perception/calibration/sync/wam_sync_v1.json \
  --sync-calibration-sha256 6b539d5557b50ad9ec1ea6fe31006290676e4f2e414ecf6fec0c375e7cd0f001 \
  --audio-device usb-serial:Y77BQVG17CD6E1 --audio-expected-usb-serial Y77BQVG17CD6E1 \
  --expected-unit-sha256 <current unit sha>
```
(Confirm the collector's exact activation flag — `--activation` vs `--production-activation` — against
the running `collect_server.py` argparse before installing.)

Operator (manual — the generator never calls systemctl):
```
systemctl --user daemon-reload
systemctl --user restart wam-collector.service
<env python> scripts/deployment/collection_service_unit.py --production-check …   # ActiveState=active, cmdline byte-exact
```

## 3. Bounds / caveats

- **hand-eye is provisional** (`USER_ACCEPTED_BELOW_GATE`, ~5 mm, `autonomous_targeting_authorized=false`).
  The occlusion budget was built for it (`handeye_bound_m = 6.5 mm`). → **Supervised teleop data
  collection is fully authorised.** Full autonomous targeting needs a re-calibration with a rigid
  visible tip to < 3 mm (rebuild `occlusion_runtime` + `occlusion_identity` after).
- `camera_latency` = 69.55 ± 0.29 ms (device_timer_host_sync). If the camera pipeline (MJPG 1080p)
  is later made faster, re-measure; the value must stay ≤ 100 ms and its 3σ interval ≤ 40 ms.
- `depth_noise` (σ = 1.117 mm + 0.150 mm·z²) was fit from flat patches already in the fixed scene
  (1.09–1.89 m). Floor-dominated at nearer range, so valid across the working depth.
- Every artifact validated through its real loader + the whole bundle through
  `load_production_runtime_assembly`; re-run `produce_final.py` (with the two shadows) to re-verify.
