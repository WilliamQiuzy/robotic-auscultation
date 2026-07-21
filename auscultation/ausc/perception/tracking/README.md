# `ausc.perception.tracking` — streaming auscultation-point 3D tracker

Turns single-frame perception (RGB-D back → 10 posterior-lung auscultation points) into a
**continuous, long-duration 3D tracker** that keeps each point locked to the same anatomy while the
patient breathes/shifts and the camera moves — so a robot arm can place a stethoscope and a doctor
teleoperates from a stable target.

Design + results: [`docs/perception_redesign.md`](../../../docs/perception_redesign.md) (esp. §9).

## Production line (recommended)
**`NormalsDetector` → `ModelBasedTracker`.**

- **`NormalsDetector`** (`detector.py`): the detector reads the **depth-derived surface-normal map**
  (DINOv2 ViT-B + LoRA + 10-landmark head), so garment colour/texture is *absent from the input* —
  garment-invariant by construction. Weights: `perc_lmh_v7_adapt` (synthetic HuMMan + 16 real frames).
  Measured: held-out real PCK@5 = 98%; on an unseen garment the 3D widths match clinical labels
  within ~1 cm (zero labels on that garment).
- **`ModelBasedTracker`** (`model_tracker.py`): the 10 points are ONE near-rigid anatomical
  constellation, not 10 independent detections. Each frame it fits a **single similarity pose**
  (rotation + scale + translation) to all 10 detector points via weighted Umeyama + **IRLS** outlier
  rejection, temporally low-passes the pose, and outputs `canonical_template ∘ pose + small
  regularized residual`. The 10-noisy-obs → 7-DOF collapse + temporal filter cut per-frame lateral
  jitter **2.8 → 1.0 mm** at only ~33 ms lag; 100% tracking on the real clip.
  - **Per-point safety (critical for a robot):** every `MTPoint` carries `measured` (was it actually
    observed this frame vs silently extrapolated from the rest of the constellation) and
    `position_std` (grows while a point coasts). When the arm occludes part of the back, the hidden
    points come back `measured=False` / `COASTING` with inflated `position_std` — **never** a
    confident `TRACKING`. The consumer must refuse/defer placement on any non-`measured` point.

The original **`StreamingAuscultationTracker`** (`tracker.py`: canonical frame + 4-channel EKF —
detector / LK-flow / point-to-plane-ICP+RGBD-odometry / IUV-semantic) remains for reference and for
the IUV pose-invariant anchor channel. It was built when the RGB detector was the least-reliable
component (track-primary policy); with the strong garment-invariant `NormalsDetector` the
`ModelBasedTracker`'s detector-primary design is preferred.

## Depth handling (measured, not assumed)
True high-frequency depth NOISE is **sub-mm** (detector 0.7 mm, model 0.2–0.3 mm) — an early "16 mm
depth jitter" was a *measurement artifact* (a frame-to-frame metric that hadn't removed real body
rotation). Consequently **`surface_snap` (ray ∩ local back-plane) is OFF by default**: it only
improves depth 0.3→0.2 mm but triples per-frame cost (52 ms → 152 ms at 1080p). Enable it for
offline/quality runs via `ModelConfig(surface_snap=True)`.

## Usage
```python
from ausc.perception.tracking import (
    NormalsDetector, ModelBasedTracker, ModelConfig,
    orbbec_intrinsics, default_intrinsics,
)
from ausc.perception.camera_stream import CameraStream

detector = NormalsDetector("outputs/perc_lmh_v7_adapt/best.pt")
with CameraStream(backend="orbbec") as cam:
    intr = orbbec_intrinsics(cam._pipe) or default_intrinsics(1920, 1080)
    tracker = ModelBasedTracker(detector, intr, ModelConfig(), extrinsics=T_cam_base)
    while True:
        f = cam.latest_frame()
        if f.status != "OK":
            continue
        out = tracker.process(f.frame, f.depth, t=f.t, depth_scale=f.depth_scale)
        for p in out.points:                      # 10 MTPoint
            if p.measured and p.position_std < 0.01:   # gate: only act on observed, low-uncertainty pts
                send_to_arm(p.xyz_world)               # world frame requires extrinsics
```

## Files
| File | Role |
|------|------|
| `detector.py` | `NormalsDetector` (geometry/normals input) + `DenseIUVDetector` (RGB) + `iuv_locate` |
| `model_tracker.py` | **`ModelBasedTracker`** — shared-pose constellation fit + per-point `measured`/`position_std` |
| `tracker.py` | `StreamingAuscultationTracker` — 4-channel EKF (reference / IUV channel) |
| `geometry.py` | intrinsics, invertible `back_square_crop`, unproject/project, normals, Kabsch, ICP, `_snap_rays` |
| `canonical.py` | EKF-line canonical model: pose estimation, bounded surfels, breathing snap |
| `ekf.py` | per-point constant-velocity EKF + Mahalanobis gating |
| `flow.py` / `rgbd_odometry.py` | LK flow / RGB-D feature odometry (EKF line) |
| `intrinsics.py` | Orbbec intrinsics / JSON IO / FOV approximation |
| `replay.py` | synthetic-scene + recorded-`Session` harness (incl. `occlude=` for arm-occlusion tests) |
| `tests/` | `test_tracking.py` (EKF line) + `test_model_tracker.py` (production line incl. occlusion safety) |

Reproducible training/data-engine/eval pipeline: [`scripts/perception/`](../../../scripts/perception/).

## Validation
`pytest ausc/perception/tracking/tests` — **25/25 pass** (geometry / ICP / EKF / IUV / RGB-D
odometry / e2e-synthetic / memory / identity; plus model-tracker GT-accuracy, smoothness,
surface-snap regression, **arm-occlusion safety**, extrinsics/world-frame, coasting).

Measured on real Orbbec clips: lateral jitter 1.0 mm @ ~33 ms lag, sub-mm depth noise, 100%
tracking, ~19 Hz/frame at 1080p (`surface_snap=False`). Synthetic GT (EKF line): median 1.3 mm.

**Remaining for live robot use (needs hardware/data):** camera↔arm **hand-eye calibration**
(`solvePnP`/ArUco → `extrinsics`); multi-subject/garment benchmark with CIs; palpation validation of
the `LUNG_ADDR_V6A` anatomical addresses. See `docs/perception_redesign.md` §9 roadmap.
