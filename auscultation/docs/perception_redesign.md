# Auscultation Perception — Authoritative Design & History

**Module:** RGB(-D) → 10 posterior-lung auscultation points → live full-frame pixels + 3D camera + robot frame
**Status:** Production weights `perc_lmh_v4` (bare-trained); clothed Orbbec deployment is out-of-distribution and the active redesign target.
**Repo root:** `/Users/williamqiu/Desktop/Harvard/WAM/auscultation`
**Owners:** WAM perception. Last revised 2026-06-26.

---

## 1. Goal & Hard Requirements

### 1.1 Functional goal

From a back-facing camera view of a (clothed) patient, detect the **10 posterior-lung auscultation points** in a deterministic order — `R1, L1, R2, L2, R3, L3, R4, L4, R5, L5` (R = image-right, L = image-left) — and expose each point, every frame, in three coordinate frames so that (a) a 6-DOF arm can place a stethoscope and (b) a doctor can teleoperate by reference to the points (imitation learning).

### 1.2 Hard requirements

| # | Requirement | Acceptance signal |
|---|---|---|
| R1 | **Image-coordinate keypoints.** Points must be defined, labeled, and ultimately readable in **original full-frame pixels** (1920×1080 Orbbec), not only inside a crop. | Labels stored in native-frame coords; output xy in native pixels. |
| R2 | **Real-time projection chain.** Every frame, each point → full-frame pixel → 3D camera metres → robot/world 3D. | `TrackedPoint.pixel`, `.xyz_camera`, `.xyz_world` populated live at camera rate. |
| R3 | **Whole back coverage.** The detector input must contain the entire posterior thorax; no lung column may be cropped off. | Crop-coverage audit ≈ 0 frames with any column outside the crop. |
| R4 | **Clothed deployment.** Robust on T-shirts (incl. dark fabric + high-contrast graphics), wrinkles, distance 0.6–1.2 m, lateral patient motion. | Clothed-subject PCK with bootstrap CIs at anatomical tolerance. |
| R5 | **Highest medical standard.** Sub-cm placement; calibrated per-point uncertainty surfaced to the robot/teleop layer; geometry metrics (parallelism, width-consistency) reported, not just PCK. | 3D placement error in mm; per-point confidence; fanning metrics. |
| R6 | **Anatomical structure.** Output must respect bilateral L/R symmetry about the spine, even vertical spacing, and **parallel** (non-fanning) L/R columns. | Width-consistency + parallelism metrics within tolerance. |

### 1.3 Deployment context

Orbbec Femto Bolt: 1920×1080 RGB + per-pixel-aligned depth; intrinsics `fx=fy≈1123, cx≈956, cy≈533`. Camera fixed-ish; patient moves 0.6–1.2 m and laterally. Patients are clothed.

---

## 2. What We Built and Tried — Measured Results & Lessons

| Attempt / Artifact | What it is | Files | Measured result | Lesson |
|---|---|---|---|---|
| **`perc_lmh_v4` (production head)** | DINOv2 ViT-B/14 frozen + LoRA (also frozen) warm-started from `perc_v7_multiview` IUV; only the 10-channel `LandmarkHead` trained. | `iuv_model.py:111-126`; `train_landmark_human.py:181-190` | PCK@10≈**96%** on bare held-out (`thr=0.10·res`≈51.8 px — very loose). | Headline metric measures near-train bare distribution; says nothing about clothed accuracy. Detector is a thin probe on frozen, bare-tuned features. |
| **Training data composition** | `gold_all.json` = **88 items: 80 scoliosis (bare) + 8 HuMMan (bare), 0 clothed**; +874 self-training pseudo-labels (bare HuMMan); clothed lab frames only in the **test** set. | `data/lab_sample_data/_GOLD_ALL/gold_all.json`; `_GOLD_TESTSET/gold_points.json` | Clothed appearance never in the loss. | Clothed deployment is **OOD by construction** — the dominant root cause (R4). |
| **Self-training scale-up (v3→v4)** | 874 pseudo-labels from the lmh_v3 bare teacher, filtered by `flip_incon`<8% + `structure_points` consistency. | `train_landmark_human.py` | "Holds accuracy, does not exceed teacher ceiling." | Confirmation bias: filtering selects samples the bare teacher already agrees with; reinforces bare prior, never injects clothing. |
| **`structure_points` (geometric projector)** | PCA spine midline on 5 midpoints → 5 even levels → L/R symmetric about midline with **per-level** half-widths. Applied at **both** inference paths. | `train_landmark_human.py:110-128`; `deploy_landmark.py:125`; `detector.py:140` | Enforces symmetry + even spacing; **does NOT equalize column widths**. | Band-aid: it launders bad detections into plausible-looking grids and **passes fanning straight through** (R6 unmet). Removing it → collapse, proving raw heatmaps are unreliable. |
| **Blind center-square crop** | `min(h,w)` square at image center, resized to 518. | `infer_overlay.py:93-100`; `geometry.py:82-88` | Frame-to-frame **collapse**: vertical spread swung 0.04–0.56; head-heavy frames crammed all 10 points onto shoulders. | On 1920×1080 landscape, `side=1080` discards columns `[0,420)∪[1500,1920)` = **~44% of width, off-center**. `up_bias` is a **no-op** on landscape (`h−s=0`). |
| **`iuv2pass` smart crop (deploy default)** | Pass-1 center-square → IUV `part==1` bbox → `max(w,h)·(1+margin)` square, **clamped** to frame, resized. | `deploy_landmark.py:53-73` | Clean clothed back (Oxford shirt): spread stabilized 0.47–0.56 but points imprecise; **L/R columns NOT parallel** (per-level widths fan 50→150 px). | Clamp-not-pad shifts body off-center + anisotropic squash; `max(w,h)` over-includes width; bbox driven by **OOD bare IUV mask**. Also **`:73` returns only the resized crop — non-invertible** (seed of the projection misconception). |
| **Real Orbbec clothed test** | Dark T-shirt with large colorful graphic. | — | All crop strategies → similar spread (~0.35–0.48); points **cluster ON the graphic** (feature magnet); not anatomically accurate. | Textbook texture-bias OOD failure. No crop trick fixes it — it is a data/feature problem (R4). |
| **`StreamingAuscultationTracker`** | EKF + LK flow + ICP; `TrackedPoint` carries `.pixel` (full-frame), `.xyz_camera`, `.xyz_world`. | `tracker.py:270-281,431-450`; `geometry.py:47-53,90-94` | Runs end-to-end on real Orbbec RGB-D; inits fast; holds **9–10/10 TRACKING**; follows patient; outputs full-frame px + 3D per point. | Tracking machinery is **solid and proves R2 is already met** — but it tracks whatever the weak detector emits. Its detector `_preprocess` uses the **dumb center-square** (`detector.py:109-115`), so it inherits the half-body crop bug. |

**Net lessons:** (i) the crop deletes the body and is the visible failure; (ii) the detector is weak by construction (bare-only data, frozen features, no geometry, MSE-collapse loss, biased readout); (iii) `structure_points` hides the weakness; (iv) the **projection chain already works** — the perceived blocker is a misconception plus one isolated non-invertible offline crop.

---

## 3. Diagnosis of the Real Problems

### 3.1 Crop root cause (R3, Concern #1) — CONFIRMED

There are **three crop paths**, sharing one fatal choice: **a square anchored to the image center (or an OOD IUV bbox), with overruns clamped to the frame edge instead of padded.**

- **Center-square (paths A & C, the live tracker).** `infer_overlay.center_square` and `CropTransform.center_square` (`geometry.py:84-88`) take `s=min(1080,1920)=1080`, keep columns `[420,1500)`, discard **840 px (~44%)** off-center. `up_bias` does nothing (`h−s=0`). When the patient is not horizontally centered, one shoulder/lung column falls outside the window and is sliced off. The tracker's `DenseIUVDetector._preprocess` (`detector.py:112`) uses this path.
- **`iuv2pass` (path B, deploy default).** `side=max(w,h)·(1+margin)` (`deploy_landmark.py:70`) over-includes width (back is taller than wide → pulls in arms/chair/background); `max(0,·)`/`min(w,·)` **clamp one edge only**, shifting the body off-center and making the box **non-square** → `cv2.resize` distorts it anisotropically; the bbox is `part==1` from a **bare-trained IUV head that mis-segments clothed backs**; and pass-1 already discarded half the body via center-square before the bbox was even measured.

**Fix axis:** person/depth-centered, square-in-source-pixels, **pad-not-clamp**, one shared function for all paths.

### 3.2 Model limitations (R5, Concern #3) — PARTLY TRUE

The **paradigm is sound** (DINOv2 + heatmap head, top-down; ViT is inherently shape-biased and DINOv2 frozen features transfer well). The weaknesses are specific:

- **Frozen backbone AND frozen LoRA** during landmark training (`train_landmark_human.py:187-188`): only a 2-layer conv head learns on features tuned for bare-back IUV. No mechanism for clothed-texture invariance → graphic feature-magnet.
- **RGB-only.** Depth/normals (already computed in `geometry.py:125-182`) — the most clothing-invariant signal — are unused for **detection** (only tracking).
- **MSE-collapse loss.** `weighted_hm_loss` with `pos_w=100` (`:25-33`) is a brittle hack against "纯 MSE 会塌成全零"; no inter-channel geometric coupling.
- **Biased readout.** argmax + ±5 window-centroid (`:86-107`) on a `res//4=129` heatmap → quantization-limited; biased hflip uses `res−1−x` (`flip_incon`), not the unbiased continuous `1−x`.
- **Encoder floor-div.** `DinoV2Encoder.forward` silently drops the non-mult-of-14 remainder (`iuv_model.py:38-39`).

### 3.3 Training flaws (R4, Concern #4) — TRUE & DOMINANT

- **~100% bare, tiny, low-diversity, self-referential** supervision (§2 table). The 80 scoliosis "backs" are ~86 single photos upsampled `n_aug=12`. 874 pseudo-labels = bare HuMMan from a bare teacher (confirmation bias).
- **Rich augmentation misrouted.** `ausc/augment.py` (lighting/hue/specular/shadow/occlusion/sensor_degrade) + `geom_aug` are baked into the **IUV** path; the production head's `HandLabelDS` (`:59-78`) does **only resize + 50% hflip**. No clothing/graphic overlays anywhere.
- **A clothed generator exists but is unwired.** `clinical_domainify.py` feeds the IUV path, not the landmark head, and is a flat-gray + sinusoidal-fold placeholder — it cannot model a printed graphic tee.
- **Train↔serve skew.** Train = perfect 518 square canvas; deploy = `iuv2pass` of landscape Orbbec; tracker = center-square. No crop-jitter/off-center/partial-body aug in training.
- **Evaluation blind to the real failure.** PCK@10 (≤51.8 px) on near-train bare data; cannot see fanning, width inconsistency, or clothed error; tiny clothed test set is statistically meaningless.

### 3.4 Projection misconception (R2, Concern #2) — RESOLVED: the math already works

The belief that the 518 square makes real-time back-projection impossible is **false**. The crop is a fully **invertible affine** `(x0, y0, side, res)`:

```
det → full :  u_full = x0 + x_det · (side/res),   v_full = y0 + y_det · (side/res)   # geometry.py:90-94
full → det :  x_det  = (u_full − x0) · (res/side), y_det  = (v_full − y0) · (res/side) # geometry.py:96-99
full → 3D  :  Z = sample_depth_m(u,v); X = (u−cx)Z/fx; Y = (v−cy)Z/fy                 # Intrinsics.unproject :47-53
cam → world:  p_w = T_cam→world · [X Y Z 1]ᵀ                                          # transform_points :210-212
```

`det_to_full` is pure `x0 + x·scale`, so it is **correct even with a negative origin** (required by pad-not-clamp). The live tracker already runs this chain: `det.crop.det_to_full(det.xy_det)` → `_unproject_pixels` → 3D (`tracker.py:271-272`); `TrackedPoint.pixel` is projected from the EKF 3D state through the **true intrinsics** (`tracker.py:445`), so it never lives in crop space. The square does **not** delete the mapping; it deletes the **off-crop data** (a column in the discarded `[0,420)` simply does not exist to be mapped). So the projection concern reduces to the **crop-coverage** problem (§3.1) plus **one real, isolated bug**: `deploy_landmark.py:73` returns only `cv2.resize(rgb[by0:by1, bx0:bx1], …)` — discards `(bx0,by0,side)` and resizes a non-square box anisotropically → non-invertible. That one offline line, not the architecture, seeded the misconception. The remaining genuine gap for robot-frame output is **hand-eye extrinsics calibration** (a deliverable, not a code flaw).

---

## 4. Recommended Architecture

**One-line thesis:** Keep top-down (canonical scale = sub-cm accuracy), but make the crop **depth-centered, pad-not-clamp, invertible**; feed **geometry (surface normals)** alongside RGB; **predict the spine structure** instead of bolting it on; and **train on clothed data** with crop-jitter and garment/graphic overlays, **labeled on the full native frame**.

### 4.1 Why top-down (not full-frame letterbox)

The crop normalizes the body to a canonical scale — the single biggest reason top-down pose (HRNet/ViTPose/RTMPose) beats bottom-up. For 10 points on a back at 0.6–1.2 m needing sub-cm precision, a far/small patient occupies few tokens and full-frame precision collapses; a depth person-box "scale normalizer" *is* a crop. Multi-person advantage is irrelevant (one patient). Crop-then-inverse-affine is exactly the standard MMPose pipeline ([MMPose `get_warp_matrix`](https://mmpose.readthedocs.io/en/latest/_modules/mmpose/structures/bbox/transforms.html); [ViTPose++](https://arxiv.org/abs/2212.04246)).

### 4.2 Single shared crop — `back_square_crop()`

Replaces `center_square`, `iuv_back_crop`, and detector `_preprocess` (kills train↔serve skew), used by label-gen, train, deploy, and the live tracker:

```python
def back_square_crop(rgb, depth, intr, margin=0.20) -> (img518, CropTransform):
    P    = pointmap_from_depth(depth, intr)           # geometry.py:125-154
    mask = body_band_mask(P)                           # geometry.py:168-182  (clothing/lighting-invariant)
    box  = bbox(mask) if mask.sum() >= MIN_PIX else person_detector(rgb)   # YOLO/RTMDet fallback; NEVER center-square
    cx, cy = center(box)                               # PERSON center, not image center
    side   = back_extent(box) * (1 + margin)
    x0, y0 = cx - side/2, cy - side/2                  # may be NEGATIVE — fine for det_to_full
    img    = pad_and_extract(rgb, x0, y0, side)        # cv2.copyMakeBorder — NEVER clamp
    return cv2.resize(img, (518,518)), CropTransform(int(x0), int(y0), int(side), 518)
```

Square stays (`518=37×14`, patch-aligned, avoids floor-div); pad keeps it square (no anisotropic resize) and the anatomical center centered; depth box replaces the OOD IUV `part==1` oracle. `det_to_full` is unchanged.

### 4.3 Model

- **Backbone:** DINOv2 ViT-B/14, weights frozen, **LoRA A/B trainable** (change `train_landmark_human.py:187-188`) — cheapest clothed-texture adaptation; small set can't destroy a frozen backbone. Sapiens swap is a documented post-MVP option (`iuv_model.py:51`).
- **Dual input (late fusion):** RGB → frozen DINOv2 patch tokens; a 3-channel **surface-normal map** (from `pointmap_from_depth`→`normals_from_pointmap`, cropped by the *same* `CropTransform`) → small conv stem → concat before `LandmarkHead`. Zero normals where `body_band_mask` is invalid (RGB-only fallback). **Single biggest clothing-invariance lever** — back curvature / spine groove / scapular angles survive a T-shirt; the graphic does not change geometry ([SimPose, ECCV 2020](https://arxiv.org/abs/2007.15506); [Kinect body-parts review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7957572/)).
- **Heads:** keep per-channel Gaussian heatmaps at **`res//2=259`** (FPN fuse / upsample); add a **structured spine-decoder** that predicts **midline polyline + 5 level positions + per-level symmetric half-widths** as model outputs, supervised jointly (teaches the bilateral/spacing prior so a graphic detection can't be laundered into a grid — fixes fanning at the source, R6). **Demote `structure_points` to a confidence gate** (raw-vs-structured residual = uncertainty); keep `flip_incon` as a second gate; surface per-point confidence to teleop.
- **Bug fixes:** encoder floor-div → assert/pad to mult-of-14; `deploy_landmark.py:73` → return the `CropTransform` and apply `det_to_full`+`unproject`.

### 4.4 Output coordinate chain

```
heatmap (DARK sub-pixel decode) → xy_det (518, sub-pixel)
CropTransform.det_to_full       → (u,v) FULL-FRAME ORIGINAL PIXELS          [R1, R2 — lossless]
Intrinsics.unproject(u,v,Z)     → (X,Y,Z) 3D CAMERA METRES (Orbbec frame)
transform_points(T_cam→robot,·) → (Xw,Yw,Zw) ROBOT/WORLD 3D                 [needs hand-eye calib]
```

Sub-pixel readout upgrades: **DARK** decode ([arXiv 1910.06278](https://arxiv.org/abs/1910.06278)) and **UDP** unbiased flip/transform ([arXiv 1911.07524](https://arxiv.org/abs/1911.07524v2)) replace argmax+window-centroid and the biased `res−1−x` flip.

### 4.5 Training recipe

- **Data (priority):** (1) **real clothed Orbbec, 150–300 subjects** (graphic/dark/wrinkled shirts, 0.6–1.2 m, off-center/partial framings) — non-negotiable dominant fix; usable signal at 50–100 ([frozen-backbone few-shot](https://link.springer.com/article/10.1007/s11263-025-02671-5)). (2) **`humman_v2`** 717 clothed real backs with GT-SMPL (derive 10 pts from `LUNG_ADDR_V6A`, zero labeling) — up-weight. (3) **clothed synthetic** rendered with physics/scan cloth + paired depth+normals — reverses the "abandon synthetic" decision, which failed only because it was *bare* pyrender ([BEDLAM, CVPR 2023](https://arxiv.org/abs/2306.16940); [SURREAL](https://openaccess.thecvf.com/content_cvpr_2017/papers/Varol_Learning_From_Synthetic_CVPR_2017_paper.pdf); THuman/AGORA). (4) bare HuMMan+scoliosis down-weighted. (5) **drop the 874 bare pseudo-labels** (confirmation bias, [ST3D](https://arxiv.org/pdf/2103.05346)); if semi-sup kept, pseudo-label *clothed unlabeled Orbbec* only, gated by `flip_incon`+`anat_plausibility`+human review.
- **Augmentation into `HandLabelDS`:** crop/framing jitter matching the deploy cropper's error distribution (off-center, partial-body, scale 0.5–1.5×, aspect); **garment/graphic/logo/wrinkle overlays** on the back mask (keypoints fixed) — direct antidote to the magnet ([texture bias, Geirhos 2019](https://arxiv.org/abs/1811.12231); [BlanketGen2](https://arxiv.org/pdf/2501.12318)); style randomization; wire in the existing `ausc/augment.py`; UDP-correct hflip.
- **Losses:** adaptive-wing / focal heatmap + **DSNT integral coordinate loss** (replaces `pos_w=100` MSE hack) + **bilateral-symmetry + width-consistency** on the structured head + optional 3D metric-spacing loss; **σ scale-normalized to back-height in px**.
- **Schedule:** two-stage — heads+normal-stem first, then very-low-LR LoRA; select checkpoint by **clothed-subject PCK**, never the bare metric.

### 4.6 Full-frame labeling

Label **anchors on the native full frame** in original pixels: C7, both acromions, both scapular inferior angles, 2–3 spine-midline points, waistline. A deterministic builder (PCA midline → 5 even levels → per-level R/L symmetric offsets) derives `R1…L5` — and *is* the structured-head supervision. Store `{stem:{anchors,points}, frame_wh, depth_path, K}`. Train-time `back_square_crop`→`full_to_det` maps labels into 518-space on the fly, so the **same labels support any crop/resolution** without re-labeling (DINOv2 pos-embeds interpolate to any mult-of-14 size; [DINOv3](https://arxiv.org/html/2508.10104v1)). Strict **subject-level** splits; freeze a clothed test split; show the annotator a depth/normal overlay to place spine/scapula under cloth.

### 4.7 Evaluation

Frozen **clothed-Orbbec benchmark, ≥20–30 subjects**, subject-disjoint, original-coord GT. Report: **PCK@5 / @2.5 in back-height-normalized units** (drop the loose `0.10·res`); **L/R parallelism** + **per-level width-consistency** (PCK is blind to fanning); **3D placement error in mm** (what the robot cares about); **crop-coverage audit** (fraction of frames with any column outside the crop — must be ~0); flip-inconsistency / raw-vs-structured residual as calibrated confidence; **bootstrap CIs on everything** (a few clips are meaningless, [MedIA 2025](https://arxiv.org/html/2307.10926v3)). Live regression: tracker TRACKING rate (≥9–10/10), full-frame px drift, 3D breathing stability.

---

## 5. Phased Roadmap to Highest Standard

| Phase | Window | Steps | Success measure |
|---|---|---|---|
| **0 — De-risk + crop fix** (no retrain) | Week 1 | Implement `back_square_crop` (depth box, pad-not-clamp, returns `CropTransform`) in `geometry.py`; wire into deploy + tracker `_preprocess`; **fix `deploy_landmark.py:73` invertibility**; fix encoder floor-div; plumb normal-map cropping. Run **existing v4** through it and overlay `det_to_full` points on the live 1920×1080. | Visual proof **R1, R2, R3** solved before any training: whole back in crop; points land in native pixels + 3D; crop-coverage ≈ 0. |
| **1 — Readout / loss / aug + LoRA unfreeze** | Weeks 1–2 (∥) | DARK+UDP readout; adaptive-wing+DSNT loss; heatmap `res/2`; wire augmentation + garment/graphic overlays + crop-jitter into `HandLabelDS`; unfreeze LoRA; retrain on existing assets (humman_v2 up-weight, bare down-weight). | First measurable clothed-PCK gain over v4; reduced graphic-magnet on real Orbbec clip; fanning metric down. |
| **2 — Dual-input + structured head + clothed data** | Weeks 2–5 | Add normal-map branch; add structured spine-decoder + symmetry/width losses (demote `structure_points` to gate); collect + anchor-label 50–100 → 300 clothed subjects; add clothed-synthetic render path; two-stage fine-tune. | Clothed-subject **PCK@5** clears bar; **parallelism + width-consistency** within tolerance; per-point confidence calibrated. |
| **3 — Eval + calibration** | Weeks 2–6 (∥) | Build frozen clothed benchmark (parallelism/width/3D-mm + bootstrap CIs); **hand-eye extrinsics calibration**. | **3D placement error in mm** with CIs meets stethoscope-contact spec; **R5 + robot-frame R2** closed-loop ready. |

**Critical path:** honest clothed model ≈ **2 weeks** (Phase 0+1 + 50–100 subjects); full accuracy ≈ **5–6 weeks**, gated by clothed data collection (parallelizable). Keep the top-down square (accuracy asset, never the projection blocker); make it depth-centered, pad-not-clamp, invertible; feed geometry; train on clothed data; predict the structure.

---

## 6. Direct Resolution of the Four Concerns

1. **Crop cuts the body — CONFIRMED.** Root cause = image-center square + clamp-not-pad on 1920×1080 landscape (discards ~44% off-center; one-sided when off-center), compounded by an OOD IUV `part==1` oracle. **Fix:** `back_square_crop` (depth-centered, pad-not-clamp, square in source px), shared by all four paths.
2. **Can't project to live frame — MISCONCEPTION.** `CropTransform.det_to_full` + `Intrinsics.unproject` already give live full-frame px + 3D (verified; works with negative origin). The square is an invertible affine. The one real bug is the non-invertible offline crop at `deploy_landmark.py:73` (one-line fix). Robot-frame output additionally needs hand-eye extrinsics (calibration deliverable).
3. **Model flawed — PARTLY TRUE.** Paradigm sound (DINOv2 + heatmap, top-down, shape-biased); weaknesses are RGB-only input, frozen LoRA, MSE-collapse `pos_w=100`, biased argmax/flip readout, inference-time `structure_points`. **Fix:** normals input + trained LoRA + adaptive-wing/DSNT + DARK/UDP + structured-output head.
4. **Training flawed — TRUE & DOMINANT.** ~100% bare, tiny, self-referential (874 bare pseudo-labels), clothed only in test, no crop-jitter, rich aug misrouted to the IUV path. **Fix:** real clothed data + humman_v2 + clothed synthetic, full-frame anchor labels, crop-jitter + garment overlays into `HandLabelDS`, subject-level clothed benchmark with parallelism/width/3D-mm + CIs.

---

## 7. Key File References

| File | Role |
|---|---|
| `ausc/models/iuv_model.py` | `:18-44` encoder (floor-div bug), `:72-87` LoRA, `:111-126` LandmarkHead / dual-input point, `:143-155` `forward_with_landmarks` |
| `scripts/train_landmark_human.py` | `:25-33` MSE+pos_w=100 loss, `:40-56` heatmaps, `:59-78` `HandLabelDS` (aug gap), `:86-107` biased readout, `:110-128` `structure_points` (width fanning), `:181-190` freeze recipe, `:147-159` eval |
| `scripts/deploy_landmark.py` | `:53-73` `iuv_back_crop` (clamp/max(w,h)/OOD bbox), **`:73` non-invertible crop**, `:125` structure at inference |
| `ausc/perception/tracking/geometry.py` | `:47-53` `Intrinsics.unproject/project`, `:70-99` `CropTransform` (the working back-projection), `:103-182` depth/pointmap/normals/`body_band_mask` (build `back_square_crop` here), `:210-212` `transform_points` |
| `ausc/perception/tracking/detector.py` | `:109-115` `_preprocess` (live tracker uses dumb center-square) |
| `ausc/perception/tracking/tracker.py` | `:270-281` `_detection_3d` (live chain, correct), `:431-450` `_emit` (`TrackedPoint.pixel/.xyz_camera/.xyz_world`) |
| `ausc/humman_label.py` | `:55-128` `pick_back_pair` / `LUNG_ADDR_V6A` (heuristic vertex definition) |
| `data/lab_sample_data/_GOLD_ALL/gold_all.json` | Production supervision: 88 items, 100% bare |
| `scripts/clinical_domainify.py` | `:63-112,198` clothed generator (unwired to head; flat-gray placeholder) |
| `ausc/augment.py`, `ausc/datasets/common.py:102-141`, `ausc/datasets/dataset.py:108-116` | Rich aug, misrouted to the IUV path |

### Citations

- Top-down pose / inverse affine: [MMPose transforms](https://mmpose.readthedocs.io/en/latest/_modules/mmpose/structures/bbox/transforms.html) · [ViTPose++](https://arxiv.org/abs/2212.04246) · [RTMPose](https://arxiv.org/abs/2303.07399) · [RTMO, CVPR 2024](https://arxiv.org/abs/2312.07526)
- Sub-pixel / unbiased readout: [DARK](https://arxiv.org/abs/1910.06278) · [UDP](https://arxiv.org/abs/1911.07524v2)
- Geometry / clothing-invariance: [SimPose](https://arxiv.org/abs/2007.15506) · [Kinect body-parts review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7957572/) · [SPiKE](https://arxiv.org/html/2409.01879v1) · [PointHPS](https://arxiv.org/pdf/2308.14492)
- Texture bias / shape bias: [Geirhos et al., ICLR 2019](https://arxiv.org/abs/1811.12231) · [Intriguing Properties of ViTs](https://arxiv.org/pdf/2105.10497)
- Synthetic clothed humans: [BEDLAM, CVPR 2023](https://arxiv.org/abs/2306.16940) · [SURREAL, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Varol_Learning_From_Synthetic_CVPR_2017_paper.pdf) · [ShapeBoost](https://arxiv.org/html/2403.01345) · [BlanketGen2](https://arxiv.org/pdf/2501.12318)
- Backbones / frozen-feature transfer: [DINOv2](https://arxiv.org/pdf/2304.07193) · [DINOv3](https://arxiv.org/pdf/2508.10104) · [Sapiens](https://arxiv.org/html/2408.12569v1) · [few-shot keypoint, frozen DINO ViT](https://link.springer.com/article/10.1007/s11263-025-02671-5)
- Pseudo-label pitfalls / evaluation: [ST3D](https://arxiv.org/pdf/2103.05346) · [CIs for medical segmentation, MedIA 2025](https://arxiv.org/html/2307.10926v3)