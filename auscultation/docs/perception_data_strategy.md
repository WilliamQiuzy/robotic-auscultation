# Perception Data Strategy

## Current Finding

The Perception bottleneck is real back data quality, not model size.

Current usable training data on the Vast training box:

| Dataset | Front rows | Back rows | Main use | Main risk |
|---|---:|---:|---|---|
| HuMMan | 4053 | 91 | Real clothed human robustness, front/heart support | Back is sparse and clothed, weak proxy for bare-back lung placement |
| syn2/synthetic | 9600 | 9600 | Dense IUV pretraining and pose/camera diversity | Synthetic domain does not close real bare-back label noise |
| Scoliosis | 0 | 1118 augmented rows from 86 subjects | Real bare-back in-domain data | Small true subject count, low-res images, HMR2/SMPL fitting noise |

Latest verified model direction:

- v3 is still the best local checkpoint for real back landmarks.
- v4a back oversampling did not beat v3.
- v4b/v4L ViT-L runs showed unstable or weak early real-back PCK.
- Therefore the next high-ROI path is data curation and better labels, not larger encoders.

## Review Artifacts

Generated local review packs:

- `data/_qc/realback_scoliosis_review/review_sheet.csv`
- `data/_qc/realback_scoliosis_review/contact_sheet_*.jpg`
- `data/_qc/realback_scoliosis_review/overlays/*.jpg`
- `data/_qc/realback_humman_review/review_sheet.csv`
- `data/_qc/realback_humman_review/contact_sheet_*.jpg`

Clinical deployment-domain artifacts:

- `data/lab_sample_data/back/*.jpg`: local backup of the three lab-domain back photos.
- `assets/clinical_bg/*.jpg`: local background plates cropped from the lab-domain photos.
- `docs/runs/lab_domain/*.png`: v3 inference overlays on lab-domain photos.
- `scripts/clinical_domainify.py`: generator for clothed clinical-domain back samples.
- `data/dataset/scoliosis_canon_iuv/`: locally backed-up canonical scoliosis source rows with image + IUV.
- `data/dataset/scoliosis_clinical/`: generated clothed clinical-domain training slice.
- `docs/runs/scoliosis_clinical_data_report.md`: generation, QC, backup, and visual-review record.

The review CSV has `review_status`, `exclude_reason`, and `notes` fields for manual curation. Use these statuses to exclude bad subjects:

- `exclude`
- `bad_label`
- `bad_image`
- `not_back`
- `uncertain`

Automatic flags are advisory only:

- `tiny`: min image dimension < 128 px
- `low_res`: min dimension < 180 px or area < 40,000 px
- `few_visible`: fewer than 8 visible back points
- `low_side_conf`: side confidence < 0.9

Current scoliosis review summary:

- 86 true subjects
- 6 `tiny`
- 15 `low_res`
- 1 `few_visible`
- At least 21/86 subjects need close review before they are trusted as gold data.

Current clinical-domain generated slice:

- Source: `data/dataset/scoliosis_canon_iuv`
- Filter: exclude `tiny`, `low_res`, and `few_visible`
- Kept source rows: 64/86
- Generated rows: 256 (`n_per=4`)
- Manifest validation errors: 0
- Lung point schema: `posterior_lung_standard_v3_safe`
  - Old issue: posterior lung points were an equal-width 4-by-2 grid.
  - Root bug: strict posterior normal filtering clamped lower lateral lung points back toward the spine.
  - New alignment: left/right are selected as symmetric pairs; lower lung rows can use lateral chest-wall vertices; manifest realignment requires points to be on the IUV body foreground with a 5 px safety-margin QC pass.
  - Reference: `data/lab_sample_data/lung_auscul_points.png`
- Trusted backups:
  - `data/_backups/scoliosis_canon_iuv_20260605.tgz`
  - `data/_backups/scoliosis_canon_iuv_local_20260605.tgz`
  - `data/_backups/scoliosis_clinical_20260605.tgz`
  - `data/_backups/scoliosis_canon_iuv_lung_standard_v2_20260605.tgz`
  - `data/_backups/scoliosis_clinical_lung_standard_v2_20260605.tgz`
  - `data/_backups/scoliosis_canon_iuv_lung_standard_v3_safe_20260605.tgz`
  - `data/_backups/scoliosis_clinical_lung_standard_v3_safe_20260605.tgz`
- Do not use incomplete transfer artifacts:
  - `data/_backups/scoliosis_dataset_20260605.tgz`
  - `data/_backups/scoliosis_dataset_20260605_http.tgz`

## Recommended Data Changes

1. Build `realback_gold` from scoliosis only after manual review.
   - Mark each subject in `review_sheet.csv`.
   - Exclude low-quality labels at subject level, not row level, because augmented rows share the same original label error.
   - Keep a fixed, non-augmented real-back test split.

2. Treat HuMMan back as robustness data, not gold lung supervision.
   - It is mostly clothed and sparse.
   - Use it at low weight for pose/domain variation.

3. Reduce reliance on augmented duplicates for validation.
   - Validation must be subject-level and original-image-level.
   - Augmentation increases training diversity but does not increase true evidence.

4. Add an anchor-based label repair path for real backs.
   - Minimum anchors: C7/neck base, left/right shoulder or acromion, spine midline, scapular lower angle, waist/pelvis line.
   - Derive the 8 lung points from these anchors.
   - This should be preferred over pure HMR2/SMPL fitting when the fit is visually off.

5. Add a clinical deployment-domain training slice.
   - Generate it locally from reviewed scoliosis canonical rows.
   - Keep IUV and points unchanged, because auscultation points are anatomical locations under clothing.
   - Use lab drape/wall background plates from `assets/clinical_bg/`.
   - Exclude `tiny`, `low_res`, and `few_visible` review subjects before generation.
   - Treat this as domain adaptation data, not as new anatomical evidence.

## Useful Commands

Build a review pack:

```bash
python scripts/qc/realback_review.py \
  --root data/dataset/scoliosis \
  --out data/_qc/realback_scoliosis_review
```

Filter a manifest after manual review:

```bash
python scripts/qc/filter_review_manifest.py \
  --manifest_in data/dataset/scoliosis/manifest.jsonl \
  --review_csv data/_qc/realback_scoliosis_review/review_sheet.csv \
  --manifest_out data/dataset/scoliosis_gold/manifest.jsonl
```

Generate clinical-domain scoliosis data:

```bash
python scripts/clinical_domainify.py \
  --src data/dataset/scoliosis_canon_iuv \
  --bg assets/clinical_bg \
  --out data/dataset/scoliosis_clinical \
  --review_csv data/_qc/realback_scoliosis_canon_iuv_review/review_sheet.csv \
  --exclude_auto_flags tiny,low_res,few_visible \
  --regions back \
  --canon_only 1 \
  --n_per 4 \
  --seed 17
```

Validate a generated dataset:

```bash
python scripts/qc/check_dataset_manifest.py \
  --root data/dataset/scoliosis_clinical \
  --json_out data/_qc/scoliosis_clinical/manifest_check.json
```

## External Data Priority

Preferred order:

1. Collect our own real back images.
   - Target first batch: 50-100 people.
   - Target useful batch: 150-300 people.
   - Capture 6-10 back images/person: standing, sitting, slight rotation, normal phone distance, different lighting.
   - Add either physical markers or manual anchor labels.

2. Use public scoliosis/back images only as candidates.
   - Roboflow Universe has scoliosis/back datasets, but licensing, duplicates, watermarks, and label quality must be checked.
   - Treat them as pseudo-label candidates until manually reviewed.

3. Use scan/render datasets for synthetic diversity.
   - THuman2.0/2.1: textured scans plus SMPL/SMPL-X fitting, useful for back rendering.
   - AGORA/BEDLAM: photorealistic humans with SMPL-X style ground truth, useful for pose/camera/domain variation.
   - 3DPW/MPII/MPI-INF-3DHP: useful for general body pose robustness, not direct auscultation landmark truth.

## Next Training Route

After `realback_gold` is created:

1. Pretrain on synthetic + HuMMan.
2. Fine-tune LoRA on curated realback gold with low LR.
3. Mix a small amount of front data to prevent front/heart regression.
4. Select checkpoint by fixed realback gold PCK, not mixed validation loss.
5. Report front and back separately.
