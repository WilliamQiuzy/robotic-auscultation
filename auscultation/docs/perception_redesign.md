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
| **`StreamingAuscultationTracker`** (LEGACY/reference — NOT the deployment path; `ModelBasedTracker` is) | EKF + LK flow + ICP; `TrackedPoint` carries `.pixel` (full-frame), `.xyz_camera`, `.xyz_world`. | `tracker.py` `_detection_3d`/`_emit`; `geometry.py` `Intrinsics`/`CropTransform` | Runs end-to-end on real Orbbec RGB-D; inits fast; holds **9–10/10 TRACKING**; follows patient; outputs full-frame px + 3D per point. | Tracking machinery is **solid and proves R2 is already met** — but it tracks whatever the weak detector emits. Its detector `_preprocess` uses the **dumb center-square** (`detector.py:109-115`), so it inherits the half-body crop bug. |

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
cam → world:  p_w = T_cam→world · [X Y Z 1]ᵀ                                          # transform_points (geometry.py)
```

`det_to_full` is pure `x0 + x·scale`, so it is **correct even with a negative origin** (required by pad-not-clamp). The live tracker already runs this chain: `det.crop.det_to_full(det.xy_det)` → `_unproject_pixels` → 3D (`tracker.py` `_detection_3d`); `TrackedPoint.pixel` is projected from the EKF 3D state through the **true intrinsics** (`tracker.py` `_emit`, `intr.project`), so it never lives in crop space. The square does **not** delete the mapping; it deletes the **off-crop data** (a column in the discarded `[0,420)` simply does not exist to be mapped). So the projection concern reduces to the **crop-coverage** problem (§3.1) plus **one real, isolated bug**: `deploy_landmark.py:73` returns only `cv2.resize(rgb[by0:by1, bx0:bx1], …)` — discards `(bx0,by0,side)` and resizes a non-square box anisotropically → non-invertible. That one offline line, not the architecture, seeded the misconception. The remaining genuine gap for robot-frame output is **hand-eye extrinsics calibration** (a deliverable, not a code flaw).

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

## 6b. Progress log (executed)

| Date | Step | Result |
|---|---|---|
| 2026-06-26 | **Phase 0** `back_square_crop` (depth-centered, pad-not-clamp, invertible) | Proven on real Orbbec: whole back in crop, 10/10 points → full-frame px + 3D. Concerns #1 (crop) + #2 (projection) resolved. Committed. |
| 2026-06-26 | **Phase 0b** wired crop into live detector+tracker (`detect(rgb,depth,intr)`) | Tracker re-runs on Orbbec at depth-centered crop; 7–10/10 TRACKING. Committed/pushed. |
| 2026-06-26 | **Phase 1 data check** | Prebuilt HuMMan has only **91 back** samples, SMPL-derived (uncertain) + off-domain → insufficient alone. Pivot to real Orbbec labels. |
| 2026-06-26 | **Phase 2 labeling** | New 20s Orbbec clip with corrected framing (whole back + R5/L5 margin); **22 frames** anchor-labeled on the full native frame (anchor tool → 5×2 grid → drag). |
| 2026-06-26 | **Phase 2a clothed fine-tune** (v4→v5): `back_square_crop` canvases + affine/flip/color/**graphic-overlay** aug + bare gold mix; head + LoRA, 60 ep | **Held-out clothed PCK@5 32%→98%, PCK@10 57%→100%, median 45→8 px.** Detection (concerns #3/#4) fixed **for this domain**. |

**Honest caveat:** the held-out set is from the *same* 20s clip (one subject, one shirt, one lighting). v5 proves the **fine-tune pipeline works and this domain is solved**; it does **not** yet prove generalization to new people/garments. Next: collect + anchor-label more subjects/garments (Phase 2b) → re-fine-tune → subject-disjoint clothed benchmark.

## 7. Key File References

| File | Role |
|---|---|
| `ausc/models/iuv_model.py` | `:18-44` encoder (floor-div bug), `:72-87` LoRA, `:111-126` LandmarkHead / dual-input point, `:143-155` `forward_with_landmarks` |
| `scripts/train_landmark_human.py` | `:25-33` MSE+pos_w=100 loss, `:40-56` heatmaps, `:59-78` `HandLabelDS` (aug gap), `:86-107` biased readout, `:110-128` `structure_points` (width fanning), `:181-190` freeze recipe, `:147-159` eval |
| `scripts/deploy_landmark.py` | `:53-73` `iuv_back_crop` (clamp/max(w,h)/OOD bbox), **`:73` non-invertible crop**, `:125` structure at inference |
| `ausc/perception/tracking/geometry.py` | `:47-53` `Intrinsics.unproject/project`, `:70-99` `CropTransform` (the working back-projection), `:103-182` depth/pointmap/normals/`body_band_mask` (build `back_square_crop` here), `transform_points` |
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
---

## 9. Generalization strategy (beyond per-scenario fine-tuning)

### 9.1 The problem: per-scenario fine-tune overfits (v5 evidence)

The GOAL is a **generalizable** detector of the 10 posterior-lung points (R1,L1..R5,L5) that works across **people / garments / scenes / cameras without per-scenario retraining**, so others can reuse the model and it is defensible in a Nature-level paper. The two prior regimes both fail this bar in opposite ways:

- **v4 — trained ~entirely on bare backs → fails on clothed.** RGB-only on a frozen bare-tuned backbone is a textbook **texture-magnet** ([Geirhos et al., ICLR 2019](https://arxiv.org/abs/1811.12231)): points cluster on the highest-saliency print (graphic-magnet), 45 px median error on clothed.
- **v5 — fine-tune on 22 real clothed frames (1 dark graphic shirt, 1 subject).** Held-out-**on-that-clip** PCK@5 32→98%, median 45→8 px — *the domain is solved*. But on an **unseen yellow shirt** it only partially generalizes: a coherent bilateral 5-level grid appears (no collapse), yet **the lower points fan out / lose precision**. A full fine-tune memorized one garment/lighting/camera, and the 10 free heatmap outputs had **no structural constraint** to forbid the fanned grid.

**Conclusion: per-scenario fine-tuning learns a pixel→point map conditioned on one scene and cannot be the path to generalization.** The fix is to remove appearance from the job in three orthogonal ways and to make invalid grids structurally impossible.

### 9.2 Candidate approaches compared

The 10 points already have an **anatomical definition** (DensePose-style `LUNG_ADDR_V6A = {name:(part,U,V)}` + SMPL vertex selection via `define_ausc_vertices`), so the target is body-model-addressable, not merely a pixel. Four candidate generalization directions were evaluated:

| Approach | Generalization mechanism | Real-data need | Accuracy on clothed back | Latency | Paper-explainability |
|---|---|---|---|---|---|
| **(1) Body-model anchoring** — fit SMPL/SMPL-X to RGB-D, read off fixed anatomical addresses | **By construction** (appearance never enters); no collapse, always a valid bilateral grid | **Minimal** — HuMMan gives 907 GT-SMPL fits free; addresses need 0 human labels | **~1–3 cm floor** (definition error + clothing offset + partial back-only cloud) — **above sub-cm** if standalone | 30–120 ms feed-forward init + throttled depth-refine | Strong narrative; reviewers hit the clothing-offset / definition-error gap |
| **(2) Synthetic scale** — large clothed domain-randomized render (BEDLAM/AGORA/SynBody/THuman/HuMMan-render), GT points projected from SMPL | Coverage of garment/pose/camera; **synthetic≈real** when clothing is *physics-simulated*, not painted skin | **Tiny** real label-adaptation; unlimited synthetic incl. graphics | Sub-cm achievable; depth/normal sim-to-real gap ≪ RGB gap | Offline (render) + real-time inference | Strong (Shotton/FaceSynthetics/BEDLAM pillars); must model Femto-Bolt depth noise |
| **(3) Geometry-primary** — surface normals/point-cloud detection; a printed graphic cannot move back geometry | **Physics** (depth/normals illumination- & texture-invariant) | Low (synthetic geom labels free + correct) | Anchors structure sub-cm (spine groove, scapular angles); **weak on vertical intercostal level** alone | Real-time (cheap organized normals) | Strong; direct precedent for back landmarks from depth alone |
| **(4) Foundation + data-efficient** — frozen human-foundation backbone + small head + LoRA + strong aug + correctly-gated SSL | Pretrained human priors + appearance randomized away; few labels can't memorize a garment | **Tens** of frames (label-adaptation, not full fine-tune) | Sub-cm with depth + structured head | Real-time (throttle heavy backbone) | Best ROI; clean ablation table is the generalization argument |

**Key evidence per direction:**
- (1) Fit-then-read-off floors at 1–3 cm on a clothed back ([ETCH](https://arxiv.org/html/2503.10624v3); [SOSORT consensus, PMC2794256](https://pmc.ncbi.nlm.nih.gov/articles/PMC2794256/); [scoliosis surface-topography ~9 mm, Sci Reports 2025](https://www.nature.com/articles/s41598-025-92551-2)); depth makes the *fit* good (~1–2 cm vertex, [ETCH](https://arxiv.org/html/2503.10624v3), [WACV 2021 RGBD pose](https://openaccess.thecvf.com/content/WACV2021/papers/Bashirov_Real-Time_RGBD-Based_Extended_Body_Pose_Estimation_WACV_2021_paper.pdf)) but the clothing-offset + definition error remain.
- (2) **Physics-draped clothing beats painted skin** ([BEDLAM, CVPR 2023](https://arxiv.org/abs/2306.16940): synthetic PA-MPJPE 46.6 vs real 46.4 on 3DPW; "realistic clothing simulation … significantly better than textured bodies"); synthetic-only landmarks via parametric template + label-adaptation ([FaceSynthetics, ICCV 2021](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/10/ICCV_2021_FaceSynthetics.pdf)); depth render garment-invariant ([Shotton et al., CVPR 2011](https://web.cs.ucdavis.edu/~yjlee/teaching/ecs289h-fall2014/kinect.pdf)). Our earlier abandoned synthetic rendered **naked SMPL** — the wrong step before the one that works.
- (3) **Back landmarks from depth alone** already published (~80% PCK / 86.9% PDJ, 105 subjects, [Delgarmi et al., bioRxiv 2021](https://www.biorxiv.org/content/10.1101/2021.02.04.429842v2.full)); normals generalize better than raw depth ([Wu et al., Adv. Intelligent Systems 2024](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202400159); [ECCV 2016 surface normals](https://link.springer.com/chapter/10.1007/978-3-319-46487-9_34)); geometry shrinks sim-to-real gap ([SimPose, ECCV 2020](https://arxiv.org/abs/2007.15506)). Both precedents report **scapula/lower region as hardest** — exactly our fan-out.
- (4) Frozen foundation features support few-shot keypoints ([few-shot keypoint on frozen DINO ViT, IJCV 2025](https://link.springer.com/article/10.1007/s11263-025-02671-5)); human-specific [Sapiens](https://arxiv.org/html/2408.12569v1) ships native normals/pointmap heads; style-randomization breaks texture bias ([SagNet](https://arxiv.org/abs/1910.11645)); SSL must avoid the single-teacher confirmation-bias trap via dual-teacher position-inconsistency correction ([SSPCM, CVPR 2023](https://arxiv.org/abs/2303.04346); [ST3D pitfalls](https://arxiv.org/pdf/2103.05346)).

### 9.3 Recommended strategy + why it generalizes

**Adopt (4) as the production real-time detector, fold in (1) as the offline label factory + throttled 3D teacher/regularizer, and use (2) as the pretraining data engine that supplies (3)'s geometry channel. Per-scenario fine-tuning is explicitly rejected.** The runtime detector is the foundation+geometry head; (1) and (2) are demoted to **supervision and a slow consistency prior** — the roles the evidence says they are actually good at.

**Core principle (the paper's thesis in one sentence):**

> Our 10 points generalize because they are **not pixels to be learned** — they are **fixed anatomical addresses on a body manifold** (`LUNG_ADDR_V6A` → barycentric SMPL addresses), detected through **three stacked appearance-invariant mechanisms**: (1) a **frozen human-foundation backbone** that has already seen millions of clothed humans so tens of our labels cannot memorize a garment; (2) a **depth-derived surface-normal stream** that a printed graphic physically cannot move; and (3) a **structured bilateral spine decoder** that makes asymmetry / fan-out impossible by construction.

Generalization is the **product of four orthogonal invariances** (frozen prior × geometry × structural inductive bias × appearance randomization + gated SSL), so **no single mechanism has to carry the load**. This removes both prior failure modes *by construction*: v4's texture-magnet (appearance is out of the backbone's job, off the geometry channel, and absent from the output structure) and v5's fan-out (the spine parametrization forbids invalid grids).

### 9.4 Concrete pipeline (runtime)

```
Orbbec RGB-D ─► back_square_crop (depth-centered, pad-not-clamp, invertible CropTransform)
  ├─ RGB 518 ─► FROZEN backbone (DINOv2 ViT-B/14 now; Sapiens-2 ablation) + rank-8 LoRA (last 2-4 blocks)
  │                                                          ─► tokens 768×37×37
  └─ depth ─► pointmap_from_depth ─► normals_from_pointmap ─► (mask by body_band_mask) ─► NormalStem
                                                                                            │
                            late-fuse 1×1 conv ─► StructuredSpineHead  (PRIMARY: midline polyline
                                                   + 5 levels + per-level symmetric half-widths)
                                                ─► LandmarkHead heatmap (AUX: DSNT + DARK/UDP readout;
                                                   raw−structured residual = per-point uncertainty σ)
  readout: structured 10 pts (518) ─► det_to_full ─► PIXELS (1920×1080)
                                    ─► sample depth ─► unproject ─► 3D CAMERA (m) [+ garment standoff along normal]
                                    ─► T_robot_cam (hand-eye) ─► ROBOT (m), with σ
  between heavy calls: StreamingAuscultationTracker (EKF+LK+ICP) propagates at camera rate
  throttled (5–15 Hz, off real-time path): SMPL depth-fit teacher → metric anchor grid → gate/teach the head
```

### 9.5 Training / fitting recipe (almost zero new human labels)

The GT machinery already exists (`define_ausc_vertices`, `LUNG_ADDR_V6A`, `project`, `rasterize_iuv`); for every source you only repoint verts/camera and **add a depth+normal render pass**.

- **Stage 0 — geometry pretrain (no real labels).** Train NormalStem + StructuredSpineHead + LandmarkHead on: (a) **HuMMan 907 GT-SMPL** clothed backs with extra virtual back-cameras → RGB+depth+normals+exact 10 pts; (b) **clothed synthetic** = `synthetic.py` upgraded naked→clothed (swap skin→clothing albedo + a logo/graphic texture bank in `_textured_mesh`; dump pyrender depth → normals; match Femto-Bolt fx=fy≈1123 + depth-noise sim). Losses: adaptive-wing + **DSNT** (replaces the `pos_w=100` MSE-collapse hack) + structured losses (bilateral symmetry, even level-spacing, width-consistency, midline-on-spine-groove). *Generalization is born here.*
- **Stage 1 — label-adaptation on the ~44 real Orbbec frames (NOT full fine-tune).** Freeze backbone + most heads; train only StructuredSpineHead (+ very-low-LR rank-8 LoRA). The structured head's **low DoF (≈16 scalars vs 10 free xy)** is why tens of frames suffice. Checkpoint-select on a **subject-disjoint clothed** split — never in-clip (the v5 mistake).
- **Stage 2 — gated SSL on unlabeled clothed Orbbec video.** Dual-teacher = student-EMA ⊕ **SAM-3D-Body mesh-fit** (already on disk at `sam-3d-body-dinov3/model.ckpt`) projecting the anatomical addresses — appearance-independent by construction, injecting exactly what the bare RGB teacher could not (fixes the v3→v4 confirmation-bias trap). Weak-aug teacher vs strong garment/style-aug student; accept a pseudo-label only if **all** gates pass: dual-teacher spatial agreement, low flip-inconsistency, low structured-residual, low mesh-reprojection error, **temporal consistency across the clip**. Human spot-review each round.
- **Do first, highest ROI:** rewire `augment.py` into the landmark dataloader (currently misrouted to the IUV path) and add **graphic/logo overlay** (shaded by normals) + **AdaIN style-randomization** — the cheapest texture-bias breaker, measurable against the yellow-shirt held-out set in week 1.

**Minimal real data.** Train weights: the **~44 frames** already collected (Stage 1 only). Calibration: ~10–20 frames with anchor GT + depth to calibrate σ and the garment-standoff term. **Benchmark (the only place more humans are needed):** ≥20–30 subjects × ≥3 garments × ≥1 unseen camera, subject-disjoint (test subjects are cheaper than train subjects and don't risk overfit). **One palpation/marker validation** of the address definition on ≥1 subject — otherwise reviewers call the SMPL-address→intercostal mapping unvalidated (it currently rests on one reference photo).

### 9.6 Cross-subject / cross-garment evaluation protocol

Fixed, **subject-level, non-augmented** real-back split; select checkpoints by real clothed-back PCK; report front/back separately. Report with **bootstrap CIs**:

- **PCK@2.5 / @5** (back-height-normalized),
- **3D placement error in mm** (the sub-cm claim, via `unproject` through real depth),
- **L/R parallelism** + **per-level width-consistency** (quantifies "no fan-out"),
- **leave-one-garment-out / leave-one-subject-out** tables.

**The killer ablation grid** = backbone (DINOv2 / DINOv3 / Sapiens-2) × normals{on/off} × structured-head{on/off} × aug{on/off} × SSL{on/off}. Expected, defensible result: **fusion ≥ normals-only ≥ RGB-only on unseen garments, RGB-only collapses on the graphic shirt** — that ablation *is* the generalization argument. Also report **fit-only (1) read-off on HuMMan-GT** to quantify the irreducible definition+clothing floor and show the learned head + depth-snap closes the last sub-cm.

### 9.7 Paper narrative

> **Anatomically-defined auscultation targets (fixed SMPL/barycentric addresses) + large-scale physically-clothed domain-randomized synthesis + a depth/normal-primary detector on a frozen human-foundation backbone + a structured bilateral spine decoder + cheap label-adaptation and correctly-gated semi-supervision = garment/subject/scene-generalizing sub-cm 3D point detection from ~zero per-scene real data.**

Three citable pillars: **Shotton** (depth garment-invariance), **FaceSynthetics** (synthetic-only dense landmarks via parametric template + label-adaptation), **BEDLAM** (synthetic≈real for clothed humans). Reusability is concrete: ship a **frozen backbone + released heads + a JSON address table + rendering scripts** (no redistribution of BEDLAM/AGORA assets — research-license only); others drop it onto new garments/subjects **without retraining**. That is the difference between a scene-tuned demo and a defensible, reusable model.

### 9.8 Phased roadmap from v5

| Phase | Window | Steps | Risk | Expected outcome |
|---|---|---|---|---|
| **A — cheapest wins** | Week 1 | Plumb normals into dataloader + NormalStem + late fusion; rewire `augment.py` + graphic-overlay + style-randomize; DARK/UDP/DSNT readout at res//2 | Low | Biggest single cross-garment jump; **measured vs the yellow-shirt held-out set** |
| **B — structure + data engine** | Weeks 1–2 | StructuredSpineHead + symmetry/even-spacing/width losses (demote inference `structure_points` to a gate); `synthetic.py` naked→clothed with depth/normals + Femto intrinsics/noise; HuMMan depth/normal export + virtual back-cameras; run **Stage-0 pretrain** | Medium (synthetic clothing fidelity, depth sim-to-real — mitigate with calibrated Femto noise from the 44 frames) | Heaviest lift; clothed-PCK clears bar, fanning structurally removed |
| **C — adaptation + SSL** | Weeks 2–3 | Stage-1 **label-adaptation** on 44 frames; SAM-3D-Body dual-teacher SSL with multi-gate acceptance | SSL drift — mitigated by gates + temporal filter + human spot-review | Generalizes to unseen garments without per-scene fine-tune |
| **D — fit-as-teacher + benchmark** | Weeks 3–4 | Build `LUNG_ADDR_BARY` (barycentric, validate on HuMMan-GT); SMPL depth-fit as throttled anchor/consistency prior + fallback; cross-subject/garment benchmark + ablation grid + palpation validation | Back-only partial cloud under-constrains the fit (regression init + priors); MHR↔SMPL topology (SAM-3D-Body init-only or Multi-HMR) | Paper-grade generalization table + sub-cm 3D mm with CIs |

**Total ≈ 3.5–4 weeks** to a paper-grade first pass, front-loaded so the highest-leverage, lowest-risk changes (normals + augmentation, Phase A) land and are measurable in week 1.

### 9.9 Key trade-offs justified

- **Frozen vs fine-tuned backbone:** with tens of labels, frozen + thin head is the only regime that doesn't overfit (v5 proved the alternative fails). Cost: slightly lower in-domain ceiling, bought back by the structured head + depth.
- **Geometry-primary vs RGB-only:** normals are noisy at range / under loose cloth (mitigate: bilateral-denoise depth, keep 0.6–1.2 m, late-fusion so RGB refines vertical level where geometry is ambiguous). Net: citable garment-invariance ([Shotton](https://web.cs.ucdavis.edu/~yjlee/teaching/ecs289h-fall2014/kinect.pdf), [Delgarmi](https://www.biorxiv.org/content/10.1101/2021.02.04.429842v2.full), [SimPose](https://arxiv.org/abs/2007.15506)) outweighs the noise.
- **Fit as teacher, not detector:** accepts the 1–3 cm fit floor but harvests its appearance-independence for free supervision + a 3D plausibility gate, while the learned head + depth-snap delivers sub-cm where depth is valid.
- **Loose garments + SMPL≠ribs are modeled, not hidden:** garment standoff measured along the surface normal (SMPL-fit surface vs observed depth); definition error quantified against HuMMan-GT and palpation. Reporting these as modeled error terms is what makes it Nature-defensible rather than a reviewer kill-shot.

### 9.11 Executed results — geometry generalizes (Stage-0 → Stage-1)

Ran the geometry-primary line end-to-end on the 5090. **Headline: a normals-only model trained on *zero* real labels already beats the bare-RGB v4 cross-domain, and a garment-blind real-adapt then reaches in-tolerance on held-out real poses.** This is the empirical backbone of §9.3.

**Data engine (zero human labels).** `geom_export.py` — HuMMan GT-SMPL → virtual back-cameras (azimuth ±28°, dist 0.95–1.35 m) → painter-rasterized IUV → V6A 10-point readout + per-view depth + surface normals. **3628 samples / 907 sequences**, fully automatic.

**Stage-0 (synthetic-only, `perc_lmh_v6_normals`).** DINOv2 ViT-B + LoRA on the **surface-normal map** → 10 heatmaps (warm-start v4; flip-aug flips nₓ + L/R-swaps). Held-out *synthetic* PCK@2.5 = 100, median 1.2 px. The decisive test is **real Orbbec, never seen in training**:

| Model | Real data seen | Real-dark PCK@5 | PCK@10 | median (518-px) |
|---|---|---|---|---|
| v4 (bare RGB) | none | 32 % | — | graphic-magnet |
| v5 (RGB, fine-tuned **on** the 22) | those 22 (in-domain) | 98 % | — | overfits 1 garment, fans on yellow |
| **v6 (normals, synthetic-only)** | **none** | **56 %** | **94 %** | 22.4 px |

Geometry **doubles** v4's cross-domain PCK@5 with no real labels, and is garment-colour-blind *by construction* (input is normals).

**Sim-to-real gap diagnosis.** Ruled out by experiment: effective-focal mismatch (K560 vs measured fₓ,eff 315–518 — re-deriving K *hurt*), axial sensor noise (calibrated **0.5 mm**, negligible), and depth holes (25 % invalid, but test-time fill/bilateral/median moved PCK@5 by ≤1). The residual 22 px decomposes (per-point signed error over the 22 frames) into **~15 px structured bias** — R-columns biased +x, L-columns −x, i.e. the V6A U-addresses sit slightly *more lateral* than the clinician's labels (the columns over-spread) — plus **~22 px random scatter** that grows from 12 px (upper back) to 32 px (low back R5/L5), the genuine normals-only localization limit where the low back is least distinctive. Global translation bias ≈ 6 px (no framing error).

**Stage-1 (co-train, `perc_lmh_v7_adapt`).** Co-trained v6 on synthetic (3628) **+ 16 real dark** normal-maps (oversampled ×60, heavy affine/flip aug), tested on **6 held-out dark** frames (unseen poses):

| | held-out-dark PCK@5 | PCK@10 | median (518-px) |
|---|---|---|---|
| v6 (synthetic-only) | 63 % | 95 % | 19.7 px |
| **v7 (co-train)** | **98 %** | **100 %** | **9.0 px (~0.5 cm)** |

Real adaptation halves the median on poses it never trained on, with no overfitting collapse across 8 epochs. Because the **only input is the depth-derived normal map**, this adaptation cannot key on garment colour/texture — so it is expected to transfer to other garments. Qualitatively, v7 (trained on dark only) places coherent bilateral columns on the **unseen yellow** clip; precision is pose/drape-dependent (loose garments perturb *geometry*, which normals **do** see — the honest residual limitation, §9.9).

**Open item (the one sanctioned human step): cross-garment quantification.** Built a 20-frame yellow (session3) anchor-labeling set (`data/lab_back_video_test/labeling_yellow/`) to measure v6 (zero-real) and v7 (dark-adapted) on yellow vs dark — the definitive "generalized" number + the paper's cross-garment table. Until then the yellow claim is architectural (input is garment-blind) + qualitative, not yet measured.

**Artifacts:** weights `outputs/perc_lmh_v6_normals/`, `outputs/perc_lmh_v7_adapt/` (5090); engine/train/eval scripts in scratchpad (`geom_export.py`, `stage0_train_normals.py`, `stage1_cotrain.py`, `geom_*` eval); overlays `data/lab_back_video_test/{GEOM,V7}_generalize.png`.

### 9.12 Dynamic 3D tracking + cross-garment validation (executed)

Plugged the v7 geometry detector into the streaming tracker ([[streaming-tracker]]) and ran it on the **real yellow clip** (the unlabeled test patient — kept label-free to avoid contamination).

**Policy inversion (the key fix).** The tracker was built "track-primary, detector-sparse" because the old RGB v4 was the least-reliable component. v7 inverts that premise: it is strong and garment-invariant, while LK-flow has no corners on a textureless shirt and ICP slides tangentially on a smooth loose back (fit reads 0.98 while sliding). Switching to **detector-primary** (detect every frame, trust it, down-weight flow) took the real-clip TRACKING rate **72%→100%** and canonical-frame stability **17.6→10.6mm**.

**Robust 3D lift.** Added `_snap_rays` to the tracker: lift each landmark by intersecting its camera ray with the **local back-surface plane** (KD-tree neighbourhood) instead of reading single-pixel depth — removes the along-ray depth error that's invisible in 2D reprojection but corrupts the EKF's lateral estimate. All 18 synthetic tracker tests still pass. `NormalsDetector` promoted into `tracking/detector.py` (drops the IUV/S channel; D+F+G).

**Breathing isolation (label-free physiology check).** Removing the rigid body motion via the canonical transform drops the mid-back depth residual from **11.5mm (gross sway) → 4.1mm** periodic — consistent with respiration. The tracker cleanly separates "patient shifts" (rigid, removed) from "surface motion" (breathing) — exactly what a held stethoscope contact needs.

**Cross-garment width validation (the generalization payoff, quantified, zero yellow labels).** dark (session2, fitted) and yellow (session3, loose) are the same subject, so the true 3D L-R widths are garment-invariant. Comparing the **model's yellow widths** to the **dark clinical-label widths**:

| level | dark clinical | yellow model | Δ |
|---|---|---|---|
| 1 | 165mm | 177mm | +12 |
| 2 | 70mm | 74mm | +4 |
| 3 | 70mm | 76mm | +6 |
| 4 | 122mm | 143mm | +21 |
| 5 | 217mm | 282mm | +65 |

On an unseen garment with no yellow labels, the geometry tracker reproduces the clinical 3D widths within **~1cm for the upper 3 levels** and 2–6cm at the bases. The hourglass profile (wide shoulders, narrow mid, wide bases) is itself clinical — the wide bases are **not** fan-out. Residual: the lung-base columns sit ~3cm/side too lateral on the loose garment (hardest point: low back + loose hem). A V6A base-address recalibration was scoped then **rejected** — the dark-vs-synthetic per-point gaps are mixed/one-subject-noisy and the clinical widths already match, so retraining would chase noise and risk the good levels.

**Artifacts (dynamic):** `track_final.mp4` (clean overlay), `track_cmp_frame.png` (raw-2D≈tracked-3D), `v8_lowback_fanout.png`; tracker edit in `ausc/perception/tracking/{tracker.py,detector.py}`; runners in scratchpad (`track_real.py`, `track_final.py`, `track_breath.py`, `dark_width.py`).

### 9.13 Post-review hardening (T5/T2/T7 executed)

After the full-module review (verdict: viable-with-blockers), executed the autonomous improvements:

- **T5 — reproducibility.** All `scripts/perception/*.py` paths are now `AUSC_REPO`/`AUSC_DATA` env-overridable (box defaults preserved). Runs from a fresh clone without path edits.
- **T2 — arm-occlusion robustness (measured win).** The arm occludes the back at placement time. Measured the gap: v7 PCK@5 drops **99→45** under a gray occluder band (occluded points 27, and even *visible* points fall 99→79 — the untrained band contaminates the whole heatmap). Added env-gated occlusion domain-randomization to the co-train (`AUSC_OCCLUSION=1`) → **v9** (`perc_lmh_v9_occ`): under occlusion **PCK@5 45→97, occluded points 27→95, visible 79→100**, clean 99 unchanged, held-out dark 98% unchanged. v9 is the recommended deployment detector; it complements the tracker's per-point `measured`/`COASTING` flags (R1).
- **T7 — SMPL-fit auto-label teacher (attempted, does NOT work as a quick build).** Built a depth-fit SMPL teacher (`scripts/perception/teacher_smpl_fit.py`): fit SMPL to the Orbbec back cloud → rasterize IUV → V6A pseudo-labels. On the real clothed partial-back it **fails**: single-view front/back symmetry is unresolved even with a back-facing prior (`back_facing=False`), Chamfer 2–8cm, and V6A reads out ~448px off vs manual — worse than the expected 1–3cm clothed floor. Conclusion: a usable teacher needs robust pose+disambiguation or a learned predictor (SAM-3D-Body, not on the box), and its payoff is **gated on T6** (more unlabeled subjects, which don't exist yet). Deferred; for now hand-labeling ~16 frames/subject (the v7 recipe) is cheaper and proven.

**Robot-safety (R1/R5) + calibration (R4) landed earlier this pass:** per-point `measured`/`position_std`, occlusion regression test, production-line exports + `xyz_world`, `surface_snap` off by default (19Hz vs 7Hz), and the eye-to-hand / tip-touch hand-eye calibration software (`scripts/perception/handeye_calibrate.py`, sim-validated; physical capture pending a lab session).

### 9.14 Industrial-grade assessment (scored, adversarially de-inflated)

Scored against a contact-medical-device bar (7 dimensions, each adversarially checked for inflation).
**Verdict: Research-prototype — mean 3.4/10.** Strong camera-frame perception, but **open-loop,
single-subject-validated, and not yet a system**. NOT pre-clinical; far from industrial-grade.

| dimension | score | one-line |
|---|---|---|
| tracking accuracy/stability | **5** | strongest: 1.0mm jitter @33ms, sub-mm depth, honest measured/COASTING flags, 25 tests — but n=1, camera-frame |
| safety / failure-handling | 4 | per-point COASTING + occlusion tests good; fast-motion/multi-person/lighting/dropout untested |
| reproducibility/traceability | 4 | reproducible scripts + tests; no CI, no per-run SHA/seed/lockfile |
| detection accuracy | 3 | clever garment-blind + occlusion-robust (v9), but all real numbers = 1 subject vs self-generated labels |
| real-time/compute | 3 | 19Hz@1080p is a doc assertion, no committed bench/p95; 53ms breaches a 20ms control deadline |
| generalization/validation rigor | 2.5 | weakest-but-one: no multi-subject set, no CIs, no LOSO/LOGO tables |
| closed-loop integration | **2** | gating: xyz_world=None, hand-eye sim-only, no arm-control node / E-stop / force loop |

**Genuinely strong:** tracking accuracy/stability + tracking-level failure semantics.
**Gating industrial-grade:** (1) closed-loop integration (no loop to close yet), (2) validation rigor
(n=1, no CIs vs the project's own ≥20–30-subject R4/R5 bar), (3) real anatomical accuracy (labels are
self-generated, never palpation/clinician-validated).

**Single most important next step:** run hand-eye on real hardware + stand up a minimal arm-control
node so `xyz_world` becomes real and the loop closes in robot frame — until then every perception
metric is camera-frame. (The arm-control node + a sim safety state machine can be built autonomously
now, ready for the moment the extrinsics land.)

**Autonomous-doable gap-closure** (no hardware): latency benchmark harness (p50/p95/p99 + per-stage);
codify real-clip jitter/fps as regression tests + CI; per-run git-SHA/data-hash/seed + lockfile;
bootstrap-CI + LOSO/LOGO table scaffolding; arm-control node + sim safety state machine; synthetic
failure-mode tests (motion blur, multi-person, lighting, frame dropout).
**Needs user (hardware/patients/clinician):** hand-eye on hardware; ≥20–30-subject clothed test set
with CIs; palpation/clinician-validated anatomical ground truth; closed-loop force regulation.

### 9.15 Post-assessment autonomous gap-closure (executed)

Closed every autonomous gap from §9.14 (calibration capture excepted — needs the lab), all committed
on `feat/auscultation-perception-tracker`, 41 hardware-free tests green + GitHub CI:

- **Closed-loop (gate #1, was 2/10):** `ausc/perception/robot/` — a safety state machine
  (IDLE/ACQUIRING/APPROACH/CONTACT/HOLD/RETRACT/ESTOP) that only drives toward a point the tracker
  reports `measured` + low `position_std` in a valid world frame, HALTS when a target degrades (arm
  occludes it), RETRACTs on persistent loss, ESTOPs on over-force; + OpenArm FK/DLS-IK + SimArm
  (kinematic, reaches 1 mm CONTACT, joint-limit-respecting) + RealArm (openarm_can, motors-off until
  `enable()`). 8/8 hardware-free safety tests. Ready for hardware: RealArm + R4 extrinsics → real loop.
- **Real-time (was 3/10):** `scripts/perception/latency_bench.py` (p50/p95/p99 + per-stage); detector
  forward = 98% of cost. **fp16 inference made default for NormalsDetector → p50 47 ms (~21 Hz)
  at 1080p, accuracy identical** (PCK@5 100 / median 9.0 px). Cites the committed
  `perception_metrics.json` (canonical), and 21 Hz clears the 20 Hz control rate.
- **Reproducibility (was 4/10):** GitHub Actions CI (`.github/workflows/perception-ci.yml`, CPU,
  runs tracker+safety tests on every change), `requirements-perception.txt` pin, `_provenance.py`
  (git-SHA/seed/env run stamp).
- **Generalization scaffolding (was 2.5/10):** `ausc/perception/eval/benchmark.py` — bootstrap-CI +
  per-subject/per-garment (LOSO/LOGO) tables + L/R width/fan-out metric; auto-produces the §9.6
  medical-grade report (and warns below the ≥20-subject bar) the moment T6 data lands.
- **Safety/robustness (was 4/10):** hard-inlier refit in the pose fit (transient detector outliers —
  2 random 9 cm spikes/frame — now rejected without corrupting the other points, which stay <2 cm) +
  failure-mode tests (transient outliers, sensor frame-dropout coast+recover). Lighting is
  architecturally mitigated (detector reads depth-normals, not RGB).

Still gating (need the user): hand-eye on hardware (R4), ≥20–30-subject clothed test set + palpation-
validated ground truth (T6/T3), closed-loop force regulation with a clinician. Detector occlusion
robustness = v9 (`perc_lmh_v9_occ`, the deployment weight).

### 9.16 Re-assessment after gap-closure (3.4 → 3.9/10)

Re-scored, adversarially de-inflated. **Mean 3.4 → 3.9 (+0.5)** — "advanced research prototype with a
wired-but-unproven closed loop." The autonomous work moved the **engineering scaffolding**, not the
**evidence**: every load-bearing accuracy/safety number is still n=1 subject, synthetic, or replay-only.

| dim | before | after | note |
|---|---|---|---|
| real-time | 3.0 | 4.5 | fp16 ~21Hz (perception_metrics.json) + latency harness (replay-only, no live p99) |
| closed-loop | 2.0 | 3.5 | full safety machine + IK + 591-frame sim demo; never on hardware, stand-in extrinsic |
| safety | 4.0 | 4.5 | state machine + hard-inlier + occlusion v9 + dropout tests; force/E-stop/hardware unproven |
| generalization | 2.5 | 3.5 | LOSO/LOGO + bootstrap-CI harness built+tested, but empty of real data (gated on T6) |
| detection | 3.0 | 3.5 | v9 occlusion-robust; still n=1, synthetic cross-subject, no clinician GT |
| tracking | 5.0 | 3.8 | **de-inflated**: 1.0mm headline not in a committed test (rests on one un-committed real clip) |
| reproducibility | 4.0 | 4.1 | CI + provenance + reqs pin; no lockfile/weight-hash, CI hardware-free only |

**Remaining gates — ALL now need the user** (a self-driving agent cannot clear them): T6 multi-subject
clothed dataset (≥20–30, varied bodies/garments) — unblocks every empty CI/LOSO/LOGO cell; clinician/
palpation anatomical ground truth (the only thing that can catch a self-consistent-but-wrong V6A);
R4 hand-eye on hardware (real arm-base accuracy); independent metrology (mocap/fiducials); real-arm
safety bring-up (force sensing + hardware E-stop + real IK); live-loop p99 under load. **Single
highest-value next step: run T6 collection AND pair it with clinician landmark labels on the same
subjects** — it simultaneously fills the two lowest-scoring, most-evidence-starved dimensions.

### 9.17 Production stress-testing → fixes (executed)

Actively probed for real-clinic failure modes and fixed what broke (each measured-before-built):

- **Depth holes — CATASTROPHIC, now FIXED.** A stress battery showed depth holes crater the detector:
  PCK@5 **100→30 at the real ~25% hole rate, →6 at 40%, →4 at 55%** (the normal map becomes noise).
  Real Orbbec already drops ~25%, worse with dark clothing/range/lighting — so the *current* deploy
  was silently degraded. **Fix: nearest-valid depth-hole fill before normals** (`_fill_depth_holes`,
  `fill_holes=True` default in NormalsDetector) → recovers **PCK@5 to 100% up to 55% holes**, verified
  through the production `detect()` path. Before/after video: `data/lab_back_video_test/production_eval/`.
- **Per-point uncertainty calibration — probed, no fix needed.** Flip-consistency disagreement does NOT
  predict per-point error (corr 0.08); concluded the existing `measured`/COASTING flag + hard-inlier
  rejection is the correct safety signal (it catches the *large* occlusion/anomaly errors; residual
  few-px on good frames is sub-cm noise). Avoided a wasted "fix."
- **Off-center patient — false alarm.** The initial 0% was a test artifact (image shifted but labels
  not); the depth-centered `back_square_crop` re-centers fine. **Bystander @ same depth** left PCK@5 98.
- **OOD fail-safe — probed + HARDENED.** The detector already fails reasonably safe (real-back conf
  0.94–0.98 vs OOD: noise 0.02, sphere 0.54, flat/empty-wall 0.84), but flat/empty at 0.84 is a thin
  margin (the tracker's `min_conf=0.3` would have accepted an empty wall and built a canonical on it).
  Added a **depth-plausibility gate** (`_back_plausible`, default on): a real back has body-distance
  median (0.4–1.6 m) AND a real depth-extent (>80 mm; measured back ~620 mm vs flat/empty 0 mm). It is
  **hole-robust** (range computed on valid points, so a back at 40% holes still passes — verified
  conf 0.97). Result: flat/empty/noise → conf 0.00–0.02 (rejected); real backs (even high-hole) pass.
- **Garment-colour depth quality:** dark clothing → **18% back-crop holes vs 10% for yellow** (IR
  absorption) — confirms the hole-fill matters most exactly where it's most common (dark clothing).

### 9.18 Production-hardening wrap-up (artifacts)

All autonomous gap-closure + production-hardening committed on `feat/auscultation-perception-tracker`,
**44 hardware-free tests green** (→ **57 after the §9.19 audit**; CI now runs all of them — see §9.19).
Net detector = **v9** (`perc_lmh_v9_occ`, occlusion-robust)
with fp16 + hole-fill + OOD-plausibility gate; net tracker = `ModelBasedTracker` (shared-pose + per-point
`measured`/`position_std`, hard-inlier robustness, robust canonical init); closed loop = `robot/`
(safety state machine + staleness/control-deadline + over-force ESTOP + FK/IK + SimArm/RealArm).

- Robustness coverage added: arm-occlusion safety, transient-outlier rejection, sensor frame-dropout,
  fast-motion (3×, bounded), long-duration (no-drift), staleness HALT — all in CI.
- Committed metric record: `scripts/perception/perception_metrics.json` (PCK / occlusion / hole / latency
  + git-SHA & weight/data hashes + explicit n=1 caveat); reproducible eval `scripts/perception/eval_detection.py`.
- Videos: `data/lab_back_video_test/production_eval/` — `hole_fix_before_after.mp4` (depth-hole fix),
  `hardened_tracker_yellow.mp4` (production-hardened tracker with per-point safety colours).

Industrial-grade is still gated on **user-only** items (hardware/patients/clinician): R4 hand-eye on
hardware, ≥20–30-subject clothed dataset + palpation/clinician landmarks (T6/T3), independent metrology,
real-arm force/E-stop bring-up. The harness (LOSO/LOGO + bootstrap CIs) auto-produces the §9.6 tables
the moment that data lands.

### 9.19 Adversarial quality audit + fixes (defense-in-depth pass)

A 9-dimension multi-agent audit (correctness / numerical / robot-safety / kinematics-IK / legacy-dead-code
/ test-coverage / integration-contract / scripts-repro / docs-accuracy) re-read the whole module; **every
finding was adversarially re-verified against the code** (33 confirmed real, 5 false positives rejected,
1 hardware-gated). All autonomously-fixable items are fixed + regression-tested (**44 → 57 hardware-free
tests, all green, all in CI**). Highlights:

**Safety-critical**
- **Staleness guard was inert in the deployment call form.** `controller.tick(points)` without
  `perception_t` defaulted the timestamp to the wall clock, which advances every tick → the frozen-pipeline
  HOLD guard never fired in production (only in the unit test that passed explicit timestamps). Fixed:
  freshness now comes from `perception_t` **or** a content fingerprint of the tracker frame, so a hung
  detector re-emitting identical points goes stale → HOLD. `run_closed_loop.py` now passes `perception_t=out.t`.
- **RETRACT / CONTACT issued no arm command** (fell through the dispatch). Now: APPROACH servos toward the
  target, **RETRACT actively backs the TCP away to a standoff then self-completes**, every other state
  positively `hold()`s (exhaustive — no state leaves the arm uncommanded). `servo_toward` is wrapped:
  an IK/driver exception → `hold()` + latched ESTOP (never crashes the loop).
- **CONTACT scoped honestly.** It is deliberately NOT re-gated on perception (the placed probe occludes
  its own landmark — `measured→False` is expected), so auto-retract there would abort every placement;
  CONTACT relies on the over-force ESTOP + teleop. Docstrings corrected to match.
- **Flicker-proof give-up.** The HOLD grace counter is now **leaky** (carried, not reset by a single good
  frame), so an intermittently-recovering target still times out to RETRACT.
- **Fail-closed boundaries:** NaN/inf detector confidence is now rejected (was `nan < min_conf == False`
  → fail-open); `_view` reads point fields via `getattr` (a legacy point type missing `measured` fails
  safe instead of crashing mid-tick); `ik_step` and `RealArm.servo_toward` refuse non-finite targets/joints.

**Correctness**
- **`NormalsDetector` ignored `intr.depth_scale`** (hardcoded `/1000`), mis-scaling both the normal-map
  input and the metric OOD plausibility gate on any Orbbec profile with `depth_scale != 1`. Now uses the
  camera scale, consistent with `sample_depth_m` / `pointmap_from_depth`.
- **IK hardening:** joint-space slew bounded by the URDF per-joint velocity limits, adaptive damping near
  singularities, and fail-loud joint limits (no silent ±π fallback). Frame contract documented: tracker
  `xyz_world`, `tcp_xyz`, and IK targets are all the arm-base (`link0`) frame — verified consistent with
  the hand-eye calibration's base link (the audit's "world-vs-link0" flag was a naming issue, **not** a
  numeric bug; changing the FK frame would have *introduced* one).
- **Hand-eye ChArUco mode crashed** on the first capture (`fk(q7)` on a non-callable `ArmFK` → `fk.fk(q7)`);
  stillness check now uses full-window peak-to-peak spread (an endpoint diff missed swing-out-and-return).

**Honest-metric / clarity:** committed `med_px` relabeled `med_px_518crop` (it is detector-crop, not
full-frame, pixels); legacy EKF `tracker.py` docstring corrected (it is NOT the deployment path —
`ModelBasedTracker` is); `run_compare.py` BEFORE baseline labeled as a de-safetied worst case (not a
production-uplift claim); doc fp16 figure reconciled to the committed 47 ms / ~21 Hz.

The one **hardware-gated** finding (cannot fix autonomously): the contact over-force ESTOP is real logic
but **not wired to any force source on the real arm** — needs the RP2040/FT sensor (or joint-effort
estimate) on hardware. Until then CONTACT-phase force safety is unbacked; this folds into the existing
real-arm force/E-stop bring-up gate.

**Second-pass self-review** (adversarial review of the fixes above, since that code was new): found 6
real issues in the just-written code, all fixed. The notable one was a CRITICAL regression I had
introduced — the new content-fingerprint staleness key embedded raw floats, and `NaN != NaN`, so a
frozen frame carrying a NaN coordinate on *any* point (routine for LOST/INIT landmarks) compared unequal
to itself every tick → staleness never fired → arm would servo on frozen data. Fixed by canonicalizing
non-finite values in the fingerprint and rejecting a non-finite `perception_t`. Also fixed: RETRACT
servo-failure now latches ESTOP (was silently swallowed → could strand the machine in RETRACT); `ik_step`
slew `dt` tied to the real ~30 Hz loop; leaky-counter docstring corrected (an exact 1:1 flicker
deliberately does not time out — still safe, each good frame is measured). **61 hardware-free tests green.**

### 9.20 Property / fuzz testing of under-covered modules (empirical bug hunt)

Static review converged after §9.19, so this round switched to **empirical property/invariant/fuzz
testing** of the modules with thin coverage (geometry, intrinsics, tracker pose-math, the torch-free
detector helpers, eval/benchmark, the legacy EKF). 120 new tests authored; running them surfaced **8
real bugs** (the rest of the asserts confirmed correct behavior and stay as regression guards):

- **`_fit_pose` corrupted by asymmetric outliers (HIGH).** The single hard-inlier refit could not
  reject 2/10 GROSS ASYMMETRIC detector glitches — they survived 3 IRLS iters and collapsed the
  centroid-sensitive Umeyama scale, smearing the 8 good points up to 83 mm and marking them
  `measured=False` (the exact failure the refit was meant to prevent; the old spike test passed only
  because symmetric spikes cancel). Fixed with an **LTS-concentration refit** (repeatedly refit on the
  h=70% smallest-residual points, then a final all-inlier refit for precision). Re-scan of 200 random
  configs with 2/10 asymmetric outliers: **worst inlier error 0.00 mm, 0 corruptions** (was ~13%).
- **Sensor-NaN robustness (deployment depth→normals path).** `_fill_depth_holes` treated NaN as valid
  (`NaN <= 0` is False) → NaN leaked through the fill; `normals_from_depth_m` only zeroed `==0` not
  NaN → non-finite normals reached the model input. Both now treat non-finite depth as a hole.
- **`normals_from_pointmap`** assigned a valid normal to a depth-hole pixel (the cross-product never
  reads the pixel's own point) → now NaN'd out where the point is non-finite.
- **Legacy `PointEKF.predict` poisoned by a NaN/inf `dt`** (`max(nan,1e-3)` returns `nan`) — `update()`
  sanitized its inputs but `predict()` did not. Now clamps `dt` to a sane finite band.
- **`eval/benchmark`**: `median_ci` let a single NaN poison the median → now drops non-finite; a record
  missing the grouping key crashed `breakdown` → now skipped (consistent with `report()`). (And a
  self-caught over-correction: `pck_ci` must KEEP non-finite errors as MISSES — dropping them would
  inflate PCK and hide failures — so that one was left as-is, guarded by a test.)

A follow-on round added an **end-to-end integration fuzz** (real `NormalsDetector`-stub → real
`ModelBasedTracker` → real `AuscultationController`/safety, driven over a synthetic sequence with
injected occlusion bursts / depth-NaN bands / dropout) asserting the chain never crashes, never emits
a non-finite `xyz_world`, and the arm only servos when the commanded point is genuinely valid — plus
property tests for the legacy torch-free primitives (`voxel_downsample`, `LKFlow`, `FeatureField`),
which surfaced 2 more real bugs: `voxel_downsample` leaked a NaN input row into the canonical surface
(`floor(nan).astype(int64)` = a garbage bucket) and `LKFlow.track` crashed on 0 input points
(`cv2.calcOpticalFlowPyrLK` assertion) — both fixed.

The deployment **camera-ingest path** (`camera_stream.py`, previously untested) got SDK-mocked tests:
the per-frameset ingest was extracted to a pure `_orbbec_frame()` (testability refactor) and covered
with a fake Orbbec SDK — color decode for every format (RGB/BGR/MJPG/YUYV/UYVY/NV12 + unknown/malformed
→ None, not crash), depth reshape + `depth_scale` extraction (incl. the no-`get_depth_scale` fallback),
alignment, frame-skip logic, the `latest_frame` NO_DATA/OK/STALE freshness contract, and the threaded
lifecycle via the built-in `fake` backend. No bug found there — the ingest was sound — but it is now
regression-covered in CI.

**230 hardware-free tests green** (61 → 181 → 210 → 230); all in CI.

**Real-clip end-to-end regression** (the real v9 `NormalsDetector` → `ModelBasedTracker` →
`AuscultationController` → `SimArm`, GPU): ran **both 602-frame Orbbec clips × 6 targets, full length**.
Result **PASS** — **0 servo-on-degraded-target violations** (the arm never moved toward a degraded
target) and **0 non-finite `xyz_world`/`position_std` emissions** across all 12 runs, confirming the
NaN-robustness + safety fixes hold on real data, not just synthetic stubs. Detector ~19 fps fp16
(consistent with the committed 47 ms). Behavioral signal worth noting: the hardest lowest points
(R5/L5) correctly degrade → HOLD → IDLE/retract (measured-rate 0.43–0.70) instead of being serviced on
bad estimates, while the easier upper/mid points (R1/R3/L1/L3) track at ~0.99–1.00 and stay in
APPROACH — the safety machine halting exactly on the points that are genuinely harder to see.

### 9.10 Citations (Section 9)

- Texture/shape bias: [Geirhos et al., ICLR 2019](https://arxiv.org/abs/1811.12231) · [SagNet style-agnostic](https://arxiv.org/abs/1910.11645)
- Body-model anchoring / RGB-D fit: [SMPL](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf) · [ETCH 2025](https://arxiv.org/html/2503.10624v3) · [WACV 2021 RGBD pose](https://openaccess.thecvf.com/content/WACV2021/papers/Bashirov_Real-Time_RGBD-Based_Extended_Body_Pose_Estimation_WACV_2021_paper.pdf) · [DoubleFusion, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_DoubleFusion_Real-Time_Capture_CVPR_2018_paper.pdf) · [Multi-HMR 2024](https://arxiv.org/abs/2402.14654) · [PointHPS](https://arxiv.org/pdf/2308.14492)
- Clinical surface-topography floor: [Scoliosis surface topography, Sci Reports 2025](https://www.nature.com/articles/s41598-025-92551-2) · [SOSORT consensus, PMC2794256](https://pmc.ncbi.nlm.nih.gov/articles/PMC2794256/)
- Synthetic scale: [BEDLAM, CVPR 2023](https://arxiv.org/abs/2306.16940) · [FaceSynthetics, ICCV 2021](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/10/ICCV_2021_FaceSynthetics.pdf) · [Shotton et al., CVPR 2011](https://web.cs.ucdavis.edu/~yjlee/teaching/ecs289h-fall2014/kinect.pdf) · [AGORA, CVPR 2021](https://arxiv.org/abs/2104.14643) · [SynBody, ICCV 2023](https://arxiv.org/pdf/2303.17368) · [SURREAL, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Varol_Learning_From_Synthetic_CVPR_2017_paper.pdf)
- Geometry-primary: [Delgarmi et al., bioRxiv 2021](https://www.biorxiv.org/content/10.1101/2021.02.04.429842v2.full) · [Normal Maps, Adv. Intelligent Systems 2024](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202400159) · [Surface normals, ECCV 2016](https://link.springer.com/chapter/10.1007/978-3-319-46487-9_34) · [SPiKE 2024](https://arxiv.org/html/2409.01879v1) · [SimPose, ECCV 2020](https://arxiv.org/abs/2007.15506)
- Foundation + data-efficient + SSL: [DINOv2](https://arxiv.org/pdf/2304.07193) · [DINOv3](https://arxiv.org/pdf/2508.10104) · [Sapiens](https://arxiv.org/html/2408.12569v1) · [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3) · [few-shot keypoint, frozen DINO ViT, IJCV 2025](https://link.springer.com/article/10.1007/s11263-025-02671-5) · [SSPCM, CVPR 2023](https://arxiv.org/abs/2303.04346) · [ST3D pitfalls](https://arxiv.org/pdf/2103.05346) · [TENT, ICLR 2021](https://arxiv.org/abs/2006.10726)
- Readout / loss: [DARK](https://arxiv.org/abs/1910.06278) · [UDP](https://arxiv.org/abs/1911.07524v2) · [DSNT](https://arxiv.org/abs/1801.07372) · [Adaptive Wing, ICCV 2019](https://arxiv.org/abs/1904.07399)

## 10. TAPIP3D ground-truth oracle (P0) — what the per-frame jitter actually is

Date: 2026-06-29. Ran **TAPIP3D** (NeurIPS 2025, RGB-D long-term 3D point tracking) as an **offline
ground-truth oracle** for our 10 lung landmarks. Compute: **RTX A6000, RunPod secure cloud, torch 2.4.1/cu124**
(the local Blackwell RTX 5090 cannot run that toolchain — sm_120 needs torch≥2.7/cu128; resolved by an
ephemeral cloud GPU, terminated after the run). TAPIP3D was **seeded with our detector's 10 landmark 3D
positions at the highest-confidence anchor frame** (identity extrinsics — camera is fixed), so it returns the
*smooth true 3D trajectory* of those exact physical points. Two 8-s clips @640×360: `orbbec_session2`
(trained subject) and `orbbec_session3` ("yellow", held-out). Alignment sanity: GT-vs-seed at anchor = **0.1 mm**.

### 10.1 Findings (per-point 3D, camera frame, mm)

| metric | TAPIP3D truth | raw detector (P1) | EKF (P2) | LK-prop (P3) |
|---|---|---|---|---|
| yellow frame-to-frame median | 1.5 | 5.0 | 2.3 | 2.3 |
| yellow frame-to-frame p95 | 11.0 | 16.1 | 10.8 | 11.6 |
| yellow frame-to-frame max | 24.5 | 47.0 | 21.6 | **194** |
| validated frame-to-frame median | 2.3 | 3.5 | 2.5 | 2.5 |

Detector-vs-smooth-truth deviation, binned by distance from the seeded anchor (separates *instantaneous*
error from *accumulated* drift):

| \|frame−anchor\| | yellow median | validated median |
|---|---|---|
| 0–15 | **3.6** | **6.3** |
| 15–60 | 11.6 | 18.9 |
| 60–150 | 9.2 | 25.3 |
| 150+ | — | 24.3 |

### 10.2 What this proves

1. **Per-frame jitter is mostly NOISE, not motion.** The back truly moves ~1.5 mm/frame (breathing) but the
   raw detector moves 5.0 mm/frame on held-out yellow → ~70 % of the per-frame motion is detector noise.
2. **The EKF (P2) is already near the smoothness floor.** Post-EKF frame-to-frame (2.3 mm) ≈ true motion
   (1.5 mm); negligible headroom remains for *smoothing/propagation* to win. The EKF de-jitter is validated.
3. **Instantaneous detector accuracy is ~4–6 mm** (near-anchor deviation). The larger 9–20 mm overall numbers
   are *accumulated divergence with time from the anchor* (confounded between TAPIP drift and detector wander)
   — **not a static bias**, and not the operative metric for a re-perceive-before-touch arm.
4. **P3 (LK optical-flow propagation between low-rate detector anchors) is REFUTED.** On low-texture
   skin/garment, LK drifts and the periodic re-anchor snaps back → spikes to 50–225 mm, *worse* than plain
   EKF. Drop LK-P3; keep the EKF.
5. **The real lever is detector accuracy on the lower points** (yellow per-point: R1/L1/R2/L2 ≈ 5–6 mm but
   R4/R5/L4/L5 = 12–18 mm) → **generalization**, not the filter. Re-confirms §3/§9: the bottleneck is the
   detector on held-out subjects; the fix is data/generalization, not more tracking machinery.

### 10.3 Deliverables / reproduction

- `data/lab_back_video_test/production_eval/gt_overlay_yellow.mp4` — 3-panel [raw | EKF | TAPIP3D truth];
  `…/gt_overlay_validated.mp4`; `…/gt_oracle_summary.png`; `…/p3_orbbec_session3.mp4` (refuted-P3 evidence).
- Recipe: build RGB-D npz (`video`/`depths`/`intrinsics`/identity `extrinsics` + `query_point=[anchor,X,Y,Z]`),
  run `inference.py --resolution_factor 2`; **`pointops2` must be compiled for the GPU arch** (sm_86 here).
  Cloud GPU is ephemeral (terminate after) and **off the critical path** now that the lever is known.
