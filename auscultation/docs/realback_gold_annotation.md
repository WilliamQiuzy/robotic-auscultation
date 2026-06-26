# Real Back Gold Annotation Path

Purpose: create a small in-domain validation set for posterior lung landmarks without requiring dense IUV labels.

## First Batch Target

- Minimum useful batch: 50 people.
- Preferred batch: 150-300 people.
- Per person: 2-4 standard back-facing frames first; add slight sitting/lighting variation after the standard view is covered.
- Keep a fixed subject-level split. Do not let different frames from the same person cross train/validation/test.

## Required Points

Use the current posterior schema: `posterior_lung_standard_v3_safe`.

Required labels per image:

`R1, L1, R2, L2, R3, L3, R4, L4`

Side convention is patient-side:

- `R*`: patient's right lung field.
- `L*`: patient's left lung field.

The target shape should follow `data/lab_sample_data/lung_auscul_points.png`: upper points are closer to the paraspinal/scapular region, lower points expand laterally toward the posterior/lower lung field.

## CSV Formats

Wide format is easiest for spreadsheets:

```csv
image,subject_id,frame_id,split,R1_x,R1_y,L1_x,L1_y,R2_x,R2_y,L2_x,L2_y,R3_x,R3_y,L3_x,L3_y,R4_x,R4_y,L4_x,L4_y
subject01/frame001.jpg,s01,f001,gold,120,180,250,180,145,230,225,230,130,290,245,290,105,350,270,350
```

Long format is easiest for labeling-tool exports:

```csv
image,subject_id,frame_id,split,point,x,y,visible
subject01/frame001.jpg,s01,f001,gold,R1,120,180,1
subject01/frame001.jpg,s01,f001,gold,L1,250,180,1
```

All eight points must be present for a gold row. A visible point outside image bounds is rejected during import.

## Import

```bash
python scripts/lab/import_realback_annotations.py \
  --csv data/lab_realback_gold/annotations.csv \
  --image_root data/lab_realback_gold/raw \
  --out data/dataset/lab_realback_gold
```

Outputs:

- `data/dataset/lab_realback_gold/manifest_landmarks.jsonl`
- `data/dataset/lab_realback_gold/img/*.jpg`
- `data/dataset/lab_realback_gold/overlays/*.jpg`
- `data/dataset/lab_realback_gold/contact_sheet.jpg`

Manually inspect `contact_sheet.jpg` before treating the set as a gold validation set.

## Landmark-Only Evaluation

```bash
python ausc/training/eval_realback_landmarks.py \
  --ckpt outputs/perc_lung_v3safe_clinical_v1/best.pt \
  --root data/dataset/lab_realback_gold \
  --tag lab_realback_gold_lung_v3safe_clinical_v1 \
  --encoder dinov2_vitb14 \
  --tta 1.0,1.14,1.27 \
  --dump outputs/perc_lung_v3safe_clinical_v1/eval/lab_realback_gold_dump
```

This reports real-domain posterior landmark PCK without dense IUV labels. It is the first metric to trust for deployment-domain back lung positioning.

## Current Limitation

This path is evaluation-ready, not training-ready. Current dense-IUV training still needs `iuv` labels. Once the first gold set exists, the next engineering step is a landmark-only fine-tuning loss that uses the same soft readout as this evaluator.
