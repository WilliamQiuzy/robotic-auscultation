# Landmark Head Architecture + Evaluation (2026-06)

## TL;DR

The auscultation-point readout was upgraded from **IUV-address soft-argmax** (which inherits the
biased V-field) to a **dedicated landmark heatmap head** that directly regresses each point. This is
the current production architecture (`perc_v9a_landmark`): it beats the previous best (`perc_v7`) on
both accuracy and left-right symmetry, and decouples front (heart) from back (lung) points.

## Model

RGB → DINOv2 ViT-B/14 (frozen) + LoRA(r16/a32) → two parallel heads:
1. **IUV head** (existing): per-pixel DensePose `(I, U, V)`. Still trained/used for the body-surface map.
2. **Landmark head** (new, `iuv_model.LandmarkHead`): conv decoder → **K=15 heatmap channels**
   (10 posterior lung `R1,L1..R5,L5` + 5 anterior heart `APETM`). Each point reads out by
   soft-argmax of its own channel (`scripts/landmark_readout.py`, window=8 / temp=0.1).

Why it helps:
- **Bypasses the biased V-field.** Points come from a directly-supervised heatmap, not from looking up
  a fixed `(I,U,V)` address in a V-field known to be biased.
- **Decouples front/back.** Each landmark owns a channel, so adding heart points cannot dilute lung
  points (this is what hurt the shared-IUV model `perc_v8`).

Training: warm-start from `perc_v7`, **freeze encoder LoRA + IUV head, train only the landmark head**
(`--landmark_head 1 --freeze_iuv 1 --init_from <v7>`). 2.36M trainable params, ~44 min (12 epochs).
The frozen IUV path guarantees the back IUV map stays identical to v7 (no regression).

## Evaluation protocol (textbook-anchored)

**The textbook image `data/lab_sample_data/lung_auscul_points_ORIG_backup.png` (Posterior view) is the
gold standard.** All targets and tests must stay consistent with it. The V6A `(I,U,V)` addresses encode
this textbook pattern (large vertical span T3→T10, lateral widening to the base, level-2 narrowest);
see `data/lab_sample_data/_gold/textbook_standard.json`.

Three complementary measures (the **3 lab back photos are the deployment judge**, eyeballed against the
textbook — we deliberately do NOT hand-label a gold, which is biased):
1. **Held-out realback PCK@10** vs V6A(=textbook) targets — `scripts/eval_gold.py` / training metric.
   Internal ckpt-selection only.
2. **Flip-consistency** (`scripts/flip_consistency.py`) — automatic left-right symmetry on lab images
   (predict on image vs its flip). The most robust deployment metric; pose-invariant.
3. **Anatomical plausibility** (`scripts/anat_plausibility.py`) — on-body %, level monotonicity,
   L/R pairing/symmetry (the pairing/symmetry parts are only valid on upright subjects).

Lab overlays + comparison panels live in `data/lab_sample_data/_bakeoff/`.

## Results (held-out PCK is NOT the deployment metric — lab is)

| Model | readout | held-out PCK@10 | lab flip-consistency | verdict |
|---|---|---:|---:|---|
| perc_v3 | IUV-address | 0.604 | — | old baseline |
| perc_v6 | IUV-address | 0.829 | 2.25% | textbook V6A + hflip fix |
| perc_v7 | IUV-address | 0.835 | 1.75% | + multi-view back data |
| **perc_v9a** | **landmark head (frozen)** | **0.883** | **1.28%** | **production** |
| perc_v9b | landmark head (joint FT) | 0.893 | 1.85% | dead end (overfits PCK, hurts lab) |

## Key learnings

1. **The readout was a real bottleneck.** PCK sat at ~0.83 across several data changes; swapping the
   readout (IUV-address → landmark head) jumped it to 0.883 — a gain no data change achieved.
2. **Held-out PCK ≠ deployment quality.** `perc_v9b` (joint fine-tune) got the *highest* held-out PCK
   (0.893) but the *worst* lab symmetry (1.85%) — it overfit the V6A targets. The lab photos are the judge.
3. **Don't dump heterogeneous data into a shared head.** `perc_v8` added front (heart) data to the
   shared IUV model and *regressed* the back (front samples don't hflip-augment, diluting back symmetry).
   The per-channel landmark head fixes this structurally.
4. **More clothed-back data has diminishing returns** (v8). The real data bottleneck is bare-back
   deployment-domain images, which needs first-party collection (see `perception_data_strategy.md`).

## Infra notes

- RunPod A100 pod; `/workspace` has a ~150 GB **quota** (df shows the moosefs pool, not the quota).
- **Never run GPU rendering/inference during training** — it kills the training process (GPU contention).
- THuman2.1 (2445 textured scans, no SMPL-X) rendering pipeline is validated (`render_thuman.py`,
  pyrender+EGL) but **shelved**: IUV labeling needs HMR2 (heavy install, env risk) and the data is
  clothed (≈ HuMMan). Revisit with SMPL-X params (exact labels) or an isolated env.
