# Perception training / data-engine / eval pipeline

Reproducible scripts behind the geometry-primary auscultation perception model and the dynamic
3D point tracker. These run on the GPU box (RTX 5090, `~/envs/ausc` torch venv) against capture +
HuMMan data under `~/ausc_test/` and `data/`. See `docs/perception_redesign.md` (§9) for the full
rationale and measured results.

## Data engine (zero manual labels)
- **`geom_export.py`** — HuMMan GT-SMPL → virtual back cameras → painter-rasterized IUV →
  `LUNG_ADDR_V6A` 10-point readout + per-view depth + surface normals. Produces the synthetic
  Stage-0 dataset (3628 samples / 907 sequences) under `~/ausc_test/stage0/`.

## Detector training (geometry / surface-normals input → garment-invariant)
- **`stage0_train_normals.py`** — Stage-0: DINOv2+LoRA on the normal map, synthetic-only → `perc_lmh_v6_normals`.
- **`stage1_cotrain.py`** — Stage-1: co-train v6 + 16 real (dark) frames (garment-blind) → `perc_lmh_v7_adapt`.
- **`stage2_drape_train.py`** — Stage-2: procedural garment-drape augmentation → `perc_lmh_v8_drape`
  (drape did NOT reduce the loose-garment low-back spread — kept for the record, not the default).

## Dynamic tracking (on real RGB-D clips)
- **`track_final.py`** — run the `ModelBasedTracker` (v7 detector) on a recorded clip → overlay video
  + canonical-frame data.
- **`run_compare.py`** — before/after: per-frame detector vs model-based tracker (jitter metrics).

## Honest evaluation (separate noise from real motion)
- **`eval_model_effect.py`** — lag, deviation-from-zero-phase-ideal, motion-split error.
- **`clean_depth_noise.py`** — >2 Hz depth NOISE (vs real <1 Hz motion). This is the metric that
  revealed the "16mm depth jitter" was a motion artifact (true depth noise is sub-mm).

Weights live in `outputs/perc_lmh_v{6_normals,7_adapt,8_drape}/` on the GPU box (git-ignored).
