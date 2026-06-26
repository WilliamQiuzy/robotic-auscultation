# Streaming Auscultation-Point Tracker — Design

> Goal: turn the **single-frame** perception model (RGB → 10 posterior-lung auscultation
> points) into a **continuous, long-duration (≤10 min) 3D tracker** that keeps each
> auscultation point locked to the *same anatomical location* on the patient's back while
> **both the patient and the camera move**. Accuracy first; must keep up with the stream
> (real-time), latency-tolerant.

Code: `auscultation/ausc/perception/tracking/`. Entry point: `StreamingAuscultationTracker`.

---

## 1. Problem statement

The existing pipeline ([scripts/deploy_landmark.py](../scripts/deploy_landmark.py),
[ausc/models/iuv_model.py](../ausc/models/iuv_model.py)) is **deliberately single-frame**:
RGB → DINOv2+LoRA → landmark heatmap head → `heatmap_xy` → `structure_points` → 10 points in
518-image space. There is **no temporal state, no depth back-projection, no 3D**.

A real auscultation is a *process*: the robot dwells at each point for several respiratory
cycles, the whole session can run ~10 minutes. Over that window:

- **The patient moves** — breathing (non-rigid, periodic, mm–cm), postural drift, occasional
  shifts (rigid, cm).
- **The camera moves** — vibration from the arm, table knocks (rigid, small).
- **Detection is noisy & occasionally wrong** — per-frame jitter, and hard frames
  (occlusion by the probe/arm, OOD pose) where the detector fails.

Re-running the single-frame detector every frame and back-projecting would give a **jittery,
identity-less** stream: points flicker, occasionally jump to wrong anatomy, and there's no
notion that "point R3 at t=0 is the same skin location as R3 at t=9 min". We need a tracker
that fuses evidence over time into a **stable, drift-free, anatomically-anchored** 3D estimate.

### Requirements
1. **Streaming / bounded memory** — process an unbounded stream at constant memory (no growth
   with session length).
2. **Long-term identity** — point *k* keeps its anatomical meaning for the whole session.
3. **Motion separation** — distinguish camera ego-motion from patient motion; absorb both.
4. **Non-rigid robustness** — breathing must not move the anatomical target.
5. **Accuracy first** — target ≤ a few mm (cf. robotic-ultrasound surface tracking achieves
   2.5–4.5 mm clinically, our reference bar).
6. **Real-time** — keep up with the camera stream; heavy parts may run throttled/async.

---

## 2. Prior art surveyed

| Family | Examples | What we take |
|---|---|---|
| **Point tracking (TAP)** | CoTracker3 (online, sliding-window), TAPIR, TAPTRv3, Track-On | Short-term frame-to-frame 2D tracking robust to occlusion. Online/windowed → real-time. We use it as the **short-term layer** (or LK optical flow as the dependency-light fallback). |
| **Non-rigid fusion** | DynamicFusion, VolumeDeform, SurfelWarp, BodyFusion, OcclusionFusion | The **canonical-model + per-frame warp** idea: keep a reference ("canonical") model of the surface and estimate the deformation that maps it to the live frame. We adopt the *canonical frame* concept (without full volumetric TSDF). |
| **RGBD odometry** | Generalized/point-to-plane ICP, RGB-D visual odometry | Estimate the rigid transform between the canonical surface and the live depth cloud → **ego + rigid-patient motion**. |
| **Robotic-US motion comp.** | Liver/lung US surface-surrogate tracking, optical-flow vascular comp. | The **closest analog**: track an anatomical surface point on a breathing patient with an RGB-D camera and drive a probe. Validates the surface-surrogate + online-model approach (mm-level). |
| **Neural 4D ("VDPM")** | Dynamic Point Maps (DPM), CoTracker3 | Feed-forward 4D reconstruction + scene flow. Powerful but heavy/opaque; hard to *guarantee* anatomical accuracy. We borrow components (CoTracker3), not the whole net. |

References: Dynamic Point Maps (arXiv:2503.16318); CoTracker3 (arXiv:2410.11831); DynamicFusion
(Newcombe 2015); robotic lung-US surface tracking (PMC9373068); RGB-D respiratory PCA (PMC5579577).

---

## 3. Key insight → the design

**An auscultation point is a fixed location on the body surface.** It does not move *relative
to the anatomy*. Everything that changes is a transform:

```
p_camera(t)  =  T_cam←canon(t)  ·  warp_breathing(t)  ·  p_canonical
                └─ rigid (ego + patient shift) ─┘  └ small non-rigid ┘
```

So instead of re-detecting an absolute point every frame, we:
1. fix each point **once** in a **canonical patient frame** (`p_canonical`, anchored to anatomy),
2. estimate the **rigid transform** `T_cam←canon(t)` each frame (camera + patient motion), and
3. apply a **small local non-rigid correction** for breathing,
4. **fuse** this prediction with multiple *independent* per-frame measurements in a per-point
   filter, so no single failure mode breaks the track.

This is DynamicFusion's philosophy specialized to **10 points** (not a full surface), made
robust by fusing 4 evidence channels with different failure modes.

### The four evidence channels (independent → robust)
| Channel | What it gives | Drift? | Fails when… | Rate |
|---|---|---|---|---|
| **D — Detector** | absolute 2D pts (perc_lmh_v4) → depth → 3D | drift-free | occlusion, OOD pose, clothing | throttled (~1–3 Hz) |
| **F — Flow** | frame→frame 2D track (LK / CoTracker3) → 3D | drifts slowly | fast motion, textureless, long gaps | per-frame |
| **G — Geometry/ICP** | rigid `T_cam←canon` from depth-cloud ICP | drift-free vs canonical | textureless depth, large motion | per-frame |
| **S — Semantic/IUV** | pixel matching canonical (part,U,V) addr → 3D | drift-free | IUV noise, OOD | with each detect |

Each emits a **3D measurement (in the canonical frame) + covariance**. A per-point EKF fuses
them with **Mahalanobis gating** (reject outliers). Result: D & G & S re-ground absolutely
(kill drift), F gives smooth high-rate motion between detections, and the EKF + gating means a
wrong detection or a bad ICP step is rejected rather than corrupting the track.

---

## 4. Three candidate architectures (and the choice)

### Approach A — Detect-and-track + per-point EKF (lightweight, classical)
Throttled detector + LK optical flow + depth back-projection + per-point EKF in the **camera**
frame. No canonical model.
- ➕ Simplest, CPU-only, real-time, few deps.
- ➖ No drift-free long-term anchor (EKF in camera frame conflates camera & patient motion);
  weak under camera motion; identity can wander over 10 min.
- **Role:** built-in **fallback mode** (`mode="ab"`), and the baseline we measure against.

### Approach B — Hybrid canonical tracker  ★ CHOSEN (production)
Canonical patient frame + rigid `T_cam←canon` (Kabsch on landmark correspondences → ICP
refine) + local breathing snap + the **4-channel EKF fusion** above + bounded keyframe/surfel
memory + confidence-gated re-acquisition.
- ➕ Drift-free & anatomically anchored for the whole session; separates camera/patient
  motion; non-rigid-robust; degrades gracefully (any 1 channel can drop out); constant memory.
- ➖ More moving parts; ICP needs depth (we have it, aligned).
- **Role:** the production pipeline (`mode="canonical"`).

### Approach C — Neural 4D (Dynamic Point Maps / CoTracker3 end-to-end)
Feed the RGBD stream into a learned 4D reconstruction + point tracker.
- ➕ SOTA generic tracking; handles texture/occlusion well.
- ➖ Heavy (GPU, large model), opaque, **hard to guarantee anatomical accuracy/units**, weak
  anatomical prior. Anatomy is exactly what our trained detector + IUV give for free.
- **Role:** CoTracker3 is available as the **F-channel backend** inside B; we do not adopt C
  wholesale.

**Decision:** Build **B**, with **A** as a fallback mode and **CoTracker3** as an optional
F-channel upgrade. B reuses every existing asset (perc_lmh_v4 detector, IUV head, aligned
depth) and adds exactly the temporal/3D machinery that's missing.

---

## 5. Streaming & memory management (constant memory over 10 min)

- **Frames:** only the *latest* frame is held (the existing `CameraStream` one-slot buffer);
  the tracker keeps the **previous** frame's gray image + tracked pixels for LK. O(1).
- **Canonical model:** a **voxel-downsampled surfel cloud** of the back surface with a fixed
  max point budget (e.g. 20k surfels); new geometry is fused in / voxel-merged, so it never
  grows past the budget. O(1).
- **Keyframes:** a small ring (e.g. 8) of (pose, sparse features) for re-acquisition; oldest
  marginalized. O(1).
- **Per-point state:** EKF (6 floats) + short history ring (e.g. last 30 estimates for
  velocity/health). O(1).
- **No full-session buffering.** Offline replay ([replay.py](../ausc/perception/tracking/replay.py))
  streams a recorded `Session` frame-by-frame exactly as live.

## 6. Real-time strategy
- Per-frame (fast, ~15–30 Hz): LK flow, depth back-project, ICP refine (init from last pose →
  few iterations), EKF predict+update. All numpy/cv2/scipy.
- Throttled / async (~1–3 Hz): the DINOv2 detector (the only heavy part). Between detections the
  G+F channels carry the track; each detection re-grounds D+S. This is the standard
  detect-then-track split (SLAM front/back-end, MOT). Accuracy↑ → detect more often.

## 7. Module layout
```
ausc/perception/tracking/
  geometry.py   intrinsics, crop-affine (518↔full frame), unproject/project,
                normals, Kabsch/Umeyama, point-to-plane ICP            [numpy/scipy]
  ekf.py        PointEKF: per-point constant-velocity EKF, Mahalanobis gating  [numpy]
  detector.py   LandmarkDetector protocol + DenseIUVDetector(perc_lmh_v4)  [torch, injectable]
  flow.py       ShortTermTracker: pyramidal LK (+ optional CoTracker3)      [cv2]
  canonical.py  CanonicalModel: canonical pts + IUV addrs + surfels,
                T_cam←canon estimation, bounded fusion, breathing snap   [numpy/scipy]
  tracker.py    StreamingAuscultationTracker (orchestrator) + dataclasses
  replay.py     run over a recorded Session or a synthetic generator (test/sim harness)
  __init__.py
  tests/test_tracking.py  geometry round-trips, ICP recovers pose, EKF converges, e2e synth
```
The detector is **dependency-injected** (a `Protocol`), so the whole tracker is unit-testable
with a synthetic detector — no GPU/model needed for the core logic.

## 7b. Exploiting the aligned RGB-D camera (Orbbec)

The camera gives **per-pixel-aligned RGB + depth every frame**. That changes the design in five
concrete ways (all on top of the canonical tracker, all config-gated):

1. **RGB-D feature odometry → tangential lock** ([rgbd_odometry.py](../ausc/perception/tracking/rgbd_odometry.py)).
   A human back is *geometrically* smooth, so point-to-plane ICP nails the surface normal but
   **slides tangentially** — over 10 min the anchor could creep sideways across the skin, which
   depth alone can't detect. RGB texture (pores, moles, hair) pins those DOFs, and because depth
   is aligned, every tracked RGB corner has an **exact** 3D coordinate → frame↔canonical motion
   is a clean 3D-3D problem (`ransac_rigid`). `FeatureField` maintains canonical-frame 3D
   features on the patient surface, LK-tracks them, and produces a tangentially-locked
   `T_cam←canon` that initializes the Kabsch+ICP refine. Most valuable under **probe/arm
   occlusion** (constant in auscultation): off-probe skin features keep the global pose honest.

2. **Depth body-masking** (`geometry.body_band_mask`). The patient is the nearest blob, so a
   depth band around the central-region median strips the bed/background from ICP, surfels, and
   feature seeding — for free, no segmentation network.

3. **Depth-aware measurement noise** (`geometry.axial_noise_var`). Structured-light axial noise
   grows ~z², so EKF measurement variance is floored by the per-measurement depth noise instead
   of a constant — principled fusion across near/far points.

4. **Respiratory signal from depth.** The back's mean depth *is* the breathing waveform; the
   tracker exposes `respiratory_offset_m` / `respiratory_amplitude_m` (ties into the
   respiratory-cycle auscultation cadence).

5. **Depth-consistency gate at init** (`_on_surface`). A detection is only allowed to anchor the
   canonical model if its 10 back-projected points form a coherent surface patch — rejects
   garbage detections before they poison the anchor (complements flip-consistency).

**Honest ablation note.** On the synthetic scene (curved surface, landmark-flow alive) VO and
ICP-only are at parity (~2.6 mm) — landmark-LK + curvature already suffice there. VO's measured
edge shows up on **flat + occluded** sequences (worst-case error ↓), and its real-world value is
expected on flat real backs + probe occlusion, which synthetic data under-represents. It is
kept (correct RGB-D design, `use_feature_odometry=True`) but not overclaimed.

## 7c. Track-primary policy (the detector is the weak link)

`perc_lmh_v4` is the **least reliable** component — it is not robustly trained for arbitrary
live poses, so on an unseen pose it can return a large-displacement (garbage) detection. We must
**not** lean on it per-frame. The policy: *trust the training-free geometric/photometric tracking
(VO + ICP + flow, registered to the frame-1 canonical model — drift-resistant); use the detector
only defensively.*

- **Confirmed init** — anchor the canonical model only after `init_confirm_frames` consecutive,
  structurally-consistent, confident detections (a single bad frame-1 cannot poison the anchor).
- **Sparing, gated steady state** — the detector re-runs only every `drift_audit_s` (default 1 s)
  or when points are LOST. Each detection passes a **whole-detection agreement gate** (`_detection_agrees`):
  it is used only if it is confident *and* agrees with the current track — *or* the track is
  already LOST and we need it to re-acquire identity. This sits on top of the per-point EKF
  Mahalanobis gate → **defense in depth**.
- **Re-acquisition** — when ≥`force_redetect_lost` points are LOST, the gate is bypassed and a
  confident detection re-anchors them (the one time we lean on the detector).

Effect (synthetic): garbage detections injected for 40 frames are rejected; the track holds at
~1.7 mm. The agreement gate makes detector cadence a *safe* knob — more frequent only improves
drift correction (audit 0.5 s→1.6 mm, 1 s→1.7 mm, 4 s→4.3 mm dynamic), bad detections rejected
either way. **Honest caveat:** the synthetic detector is GT-perfect, so synthetic *undersells*
this policy — on a real unreliable detector, leaning on it less is exactly what protects you, and
that benefit can't be measured until real-data testing.

## 8. Output contract
Per frame the tracker returns, for each of the 10 points:
`name, xyz_canonical, xyz_camera, xyz_world(if extrinsics), pixel(full-frame), confidence,
status ∈ {INIT, TRACKING, COASTING, LOST}, velocity`. Plus session-level
`T_cam←canon`, ICP fitness, detector age. This is what the arm controller consumes to servo
the stethoscope onto the live anatomical target.
