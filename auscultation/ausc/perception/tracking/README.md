# `ausc.perception.tracking` — streaming auscultation-point tracker

Turns the **single-frame** perception model (RGB → 10 posterior-lung auscultation points) into a
**continuous, long-duration (≤10 min) 3D tracker** that keeps each point locked to the same
anatomy while the **patient and the camera move**. Accuracy first; real-time.

Full design + prior-art survey: [`docs/perception_streaming_tracker.md`](../../../docs/perception_streaming_tracker.md).

## Why
A 10-minute auscultation means the patient breathes / shifts and the camera vibrates. Re-running
the detector per frame and back-projecting gives a jittery, identity-less point stream. This
package fuses evidence over time into a **stable, drift-free, anatomically anchored** 3D estimate.

## How (Approach B — hybrid canonical tracker)
Each auscultation point is fixed once in a **canonical patient frame** (anchored to anatomy). Per
frame we estimate the rigid `T_cam←canon` and fuse four independent channels into a per-point EKF:

| Ch | Source | Role |
|----|--------|------|
| **D** detector | `perc_lmh_v4` 2D pts → depth → 3D (throttled ~2 Hz) | drift-free re-grounding |
| **F** flow | LK optical flow 2D → depth → 3D (per frame) | smooth, high-rate |
| **G** geometry | **RGB-D feature odometry** + point-to-plane ICP → `T_cam←canon` (per frame) | camera/patient motion + tangential lock |
| **S** semantic | IUV `(part,U,V)` address → 3D (with detect) | pose-invariant anchor |

**Aligned RGB-D specifics** (Orbbec, per-pixel registered): every landmark/feature pixel has an
*exact* 3D point, so (1) ego-motion uses **RGB-D feature odometry** (`ransac_rigid` on tracked
skin features) to pin the tangential DOFs a smooth back hides from ICP; (2) the bed/background is
removed by a depth body-mask; (3) EKF noise is scaled by axial depth noise (~z²); (4) the back's
mean depth yields a **respiratory signal**; (5) detections must pass a depth surface-consistency
gate to anchor. See design doc §7b.

EKF lives in the canonical frame; output camera points = `T_cam←canon · ekf.position` then
**snapped to the live surface** for the current breathing phase. Mahalanobis gating rejects bad
detections / ICP steps. Memory is constant (voxel-capped surfels + one-frame LK state).

## Usage
```python
from ausc.perception.tracking import (
    StreamingAuscultationTracker, TrackerConfig, DenseIUVDetector,
    orbbec_intrinsics, default_intrinsics,
)
from ausc.perception.camera_stream import CameraStream

detector = DenseIUVDetector("outputs/perc_lmh_v4/best.pt")   # production weights
with CameraStream(backend="orbbec") as cam:
    intr = orbbec_intrinsics(cam._pipe) or default_intrinsics(1280, 720)
    tracker = StreamingAuscultationTracker(detector, intr, TrackerConfig())
    while True:
        f = cam.latest_frame()
        if f.status != "OK":
            continue
        out = tracker.process(f.frame, f.depth, t=f.t, depth_scale=f.depth_scale)
        for p in out.points:        # 10 TrackedPoint: name, xyz_camera, pixel, status, conf
            ...                     # feed xyz_camera (+world via extrinsics) to the arm controller
```

## Files
| File | Role |
|------|------|
| `geometry.py` | intrinsics, crop-affine (518↔full), unproject/project, normals, Kabsch, point-to-plane ICP |
| `ekf.py` | per-point constant-velocity EKF + Mahalanobis gating |
| `detector.py` | `DenseIUVDetector` (wraps `perc_lmh_v4`) + `iuv_locate` (semantic anchor) |
| `flow.py` | `LKFlow` short-term tracker (+ CoTracker3 upgrade hook) |
| `rgbd_odometry.py` | `FeatureField` — RGB-D 3D-3D feature odometry (tangential lock) |
| `canonical.py` | canonical model: `T_cam←canon` estimation, bounded surfel memory, breathing snap |
| `tracker.py` | `StreamingAuscultationTracker` orchestrator + output dataclasses |
| `intrinsics.py` | Orbbec intrinsics / JSON IO / FOV approximation |
| `replay.py` | synthetic-scene + recorded-`Session` harness (`python -m ...replay --synthetic`) |
| `tests/test_tracking.py` | geometry / ICP / EKF / IUV / e2e-synthetic / memory / identity |

## Validation (synthetic GT; sensors not yet on the box)
`pytest ausc/perception/tracking/tests` — **16/16 pass** (geometry / ICP / EKF / IUV / RGB-D
odometry / body-mask / axial-noise / respiratory / surface-gate / e2e / memory / identity).
Synthetic sim with a background wall (camera motion + 6 mm breathing + 2 mm depth noise,
detector throttled to 2 Hz):

- **median 1.3–1.6 mm**, p90 ≤ 2.8 mm — under the 2.5–4.5 mm robotic-ultrasound clinical bar.
- **99 %** of point-frames in `TRACKING`; no identity swaps; memory bounded over long runs.
- core ~31 ms/frame (~32 Hz) on a laptop CPU; the DINOv2 detector is the throttled cost.
- torch detector seam smoke-tested on the RTX 5090 (loads production-format ckpt; confidence
  gate refuses to anchor on a bad detection).
- VO ablation: parity with ICP-only on the (easy) curved/flow-alive scene; worst-case edge on
  flat+occluded. Consumers must gate on `status`/`confidence` — occluded points drop to
  `COASTING`/`LOST` rather than reporting a confident wrong 3D point.

**Remaining for live use:** plug the Orbbec into the box → real `orbbec_intrinsics` + RGBD →
calibrate camera↔arm extrinsics → tune `TrackerConfig` thresholds on real depth noise.
