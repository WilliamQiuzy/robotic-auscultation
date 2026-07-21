# Perception training / data-engine / eval pipeline

Reproducible scripts behind the geometry-primary auscultation perception model and the dynamic
3D point tracker. Host ownership is explicit and fail-closed:

- **H200** owns geometry generation, detector training, and offline evaluation. Defaults are active
  code `/home/ssh-ziyue/wam_ausc/current/auscultation`, persistent data
  `/home/ssh-ziyue/wam_ausc/data`, and outputs under `data/outputs/perception/`.
- **linbox** owns calibration, recorded-clip replay, real-time inference, and hardware-in-the-loop.
  Defaults are active code `/home/robot/Desktop/stethoscope/.wam-current/auscultation`, sessions
  `/home/robot/wam_data/sessions`, and outputs `/home/robot/wam_data/outputs/perception/`.
- **Mac** is development/test only. A wrong-host launch is rejected before `chdir` or model loading.

`AUSC_REPO`, `AUSC_DATA`, and `AUSC_OUTPUT` remain available as absolute-path overrides. Local tests
must additionally set both `WAM_TEST_ROLE_OVERRIDE=1` and `WAM_TESTING=1`.

H200 jobs that consume real recordings no longer scan mutable names under `data/`. Set
`AUSC_PERCEPTION_LABELED_RELEASE` and/or `AUSC_PERCEPTION_EVAL_RELEASE` to a direct completed release
under `~/wam_ausc/data/raw_sessions/releases/raw_session-<hash>`. Before model imports, the resolver
rechecks `manifest.json`, `FINALIZATION.json`, the exact artifact set, and every size/SHA-256; labels
must already be part of the sealed artifact inventory. Synthetic Stage-0 data is generated under
`data/outputs/perception/stage0`, not inside an immutable real-session release.

## Data engine (zero manual labels)
- **`geom_export.py`** — HuMMan GT-SMPL → virtual back cameras → painter-rasterized IUV →
  `LUNG_ADDR_V6A` 10-point readout + per-view depth + surface normals. Produces the synthetic
  Stage-0 dataset (3628 samples / 907 sequences) under H200 `data/stage0/`.

## Detector training (geometry / surface-normals input → garment-invariant)
- **`stage0_train_normals.py`** — Stage-0: DINOv2+LoRA on the normal map, synthetic-only → `perc_lmh_v6_normals`.
- **`stage1_cotrain.py`** — Stage-1: co-train v6 + 16 real (dark) frames (garment-blind) → `perc_lmh_v7_adapt`.
- **`stage2_drape_train.py`** — Stage-2: procedural garment-drape augmentation → `perc_lmh_v8_drape`
  (drape did NOT reduce the loose-garment low-back spread — kept for the record, not the default).

## Dynamic tracking on linbox (real RGB-D clips)
- **`track_final.py`** — run the `ModelBasedTracker` (v7 detector) on a recorded clip → overlay video
  + canonical-frame data.
- **`run_compare.py`** — before/after: per-frame detector vs model-based tracker (jitter metrics).
- **`run_closed_loop.py`** — hardware-in-the-loop closed-loop tracking/inference; never a training job.
- **`clean_depth_noise.py`** — linbox recorded-clip/realtime depth-noise diagnostic (>2 Hz noise vs
  real <1 Hz motion). It revealed that the prior “16 mm depth jitter” measurement mixed motion with
  sensor noise; it does not fit or update model weights.

## Honest offline evaluation on H200 (separate noise from real motion)
- **`eval_model_effect.py`** — lag, deviation-from-zero-phase-ideal, motion-split error.
- **`eval_detection.py`** and **`latency_bench.py`** — released-model detection quality and offline
  performance gates. They read completed H200 artifacts and never touch linbox hardware.

Weights live under H200 `~/wam_ausc/data/outputs/perception/perc_lmh_*`. Linbox only consumes a
verified released weight; it does not train or overwrite the active code release.
