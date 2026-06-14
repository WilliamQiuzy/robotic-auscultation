#!/usr/bin/env python3
"""Import manually annotated lab back-lung landmarks into a local gold set.

Supported CSV formats:

Long format:
  image,subject_id,frame_id,point,x,y,visible,split,notes

Wide format:
  image,subject_id,frame_id,R1_x,R1_y,L1_x,L1_y,...,R4_x,R4_y,L4_x,L4_y

The output manifest is landmark-only on purpose.  It can be used by
`ausc/training/eval_realback_landmarks.py` without requiring dense IUV labels.
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np


BACK_POINTS = ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4")


def _truthy(v):
    if v is None:
        return True
    s = str(v).strip().lower()
    if s == "":
        return True
    return s not in {"0", "false", "no", "n", "hidden", "occluded", "nan"}


def _safe_id(s):
    s = os.path.splitext(os.path.basename(str(s)))[0]
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s or "frame"


def _read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty annotation CSV: {path}")
    return rows


def _parse_float(row, key, image):
    try:
        return float(row[key])
    except Exception as exc:
        raise ValueError(f"{image}: invalid numeric field {key}={row.get(key)!r}") from exc


def parse_annotations(csv_path):
    rows = _read_csv(csv_path)
    if "image" not in rows[0]:
        raise ValueError("annotation CSV must contain an 'image' column")

    grouped = defaultdict(lambda: {"points": {}, "meta": {}})
    if "point" in rows[0]:
        for row in rows:
            image = row["image"].strip()
            point = row.get("point", "").strip()
            if point not in BACK_POINTS:
                raise ValueError(f"{image}: unexpected point {point!r}; expected one of {BACK_POINTS}")
            grouped[image]["points"][point] = {
                "xy": [_parse_float(row, "x", image), _parse_float(row, "y", image)],
                "visible": _truthy(row.get("visible")),
            }
            grouped[image]["meta"].update({k: v for k, v in row.items()
                                           if k not in {"point", "x", "y", "visible"}})
    else:
        missing = [f"{p}_{axis}" for p in BACK_POINTS for axis in ("x", "y")
                   if f"{p}_{axis}" not in rows[0]]
        if missing:
            raise ValueError(f"wide annotation CSV missing columns: {', '.join(missing)}")
        for row in rows:
            image = row["image"].strip()
            for point in BACK_POINTS:
                grouped[image]["points"][point] = {
                    "xy": [_parse_float(row, f"{point}_x", image),
                           _parse_float(row, f"{point}_y", image)],
                    "visible": _truthy(row.get(f"{point}_visible", row.get("visible"))),
                }
            grouped[image]["meta"].update({k: v for k, v in row.items()
                                           if not re.match(r"^[LR][1-4]_(x|y|visible)$", k)})

    out = []
    for image, item in grouped.items():
        missing = [p for p in BACK_POINTS if p not in item["points"]]
        if missing:
            raise ValueError(f"{image}: missing required back points: {', '.join(missing)}")
        out.append({"image": image, **item})
    return out


def validate_and_write(entries, image_root, out_root, copy_images=True):
    img_dir = os.path.join(out_root, "img")
    ov_dir = os.path.join(out_root, "overlays")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ov_dir, exist_ok=True)

    manifest = []
    for entry in entries:
        src = entry["image"]
        src_path = src if os.path.isabs(src) else os.path.join(image_root, src)
        img = cv2.imread(src_path)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {src_path}")
        h, w = img.shape[:2]

        for point, payload in entry["points"].items():
            x, y = payload["xy"]
            if payload.get("visible", True) and not (0 <= x < w and 0 <= y < h):
                raise ValueError(f"{entry['image']}: {point} out of image bounds ({x:.1f}, {y:.1f}) for {w}x{h}")

        meta = entry["meta"]
        subject = meta.get("subject_id") or meta.get("subject") or _safe_id(entry["image"])
        frame = meta.get("frame_id") or meta.get("frame") or _safe_id(entry["image"])
        row_id = f"lab_{_safe_id(subject)}_{_safe_id(frame)}"

        ext = os.path.splitext(src_path)[1].lower() or ".jpg"
        rel_img = os.path.join("img", f"{row_id}{ext}")
        dst = os.path.join(out_root, rel_img)
        if copy_images:
            shutil.copy2(src_path, dst)
        elif not os.path.exists(dst):
            os.symlink(os.path.abspath(src_path), dst)

        row = {
            "id": row_id,
            "image": rel_img,
            "iuv": None,
            "region": "back",
            "is_valid": True,
            "domain": "lab_realback_gold",
            "point_schema": "posterior_lung_standard_v3_safe",
            "subject_id": str(subject),
            "frame_id": str(frame),
            "split": meta.get("split", "gold"),
            "points": entry["points"],
            "width": w,
            "height": h,
        }
        if meta.get("notes"):
            row["notes"] = meta["notes"]
        manifest.append(row)
        _write_overlay(img, row, os.path.join(ov_dir, f"{row_id}.jpg"))

    with open(os.path.join(out_root, "manifest_landmarks.jsonl"), "w") as f:
        for row in manifest:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    _write_contact_sheet(out_root, manifest)
    return manifest


def _write_overlay(img, row, path):
    canvas = img.copy()
    for point in BACK_POINTS:
        payload = row["points"][point]
        if not payload.get("visible", True):
            continue
        x, y = [int(round(v)) for v in payload["xy"]]
        cv2.circle(canvas, (x, y), 8, (0, 255, 0), 2)
        cv2.putText(canvas, point, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imwrite(path, canvas)


def _write_contact_sheet(out_root, manifest, max_n=24):
    thumbs = []
    for row in manifest[:max_n]:
        ov = cv2.imread(os.path.join(out_root, "overlays", f"{row['id']}.jpg"))
        if ov is None:
            continue
        h, w = ov.shape[:2]
        scale = 280 / max(h, w)
        thumb = cv2.resize(ov, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
    if not thumbs:
        return
    cell_h = max(t.shape[0] for t in thumbs)
    cell_w = max(t.shape[1] for t in thumbs)
    cols = min(4, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * cell_h:r * cell_h + thumb.shape[0], c * cell_w:c * cell_w + thumb.shape[1]] = thumb
    cv2.imwrite(os.path.join(out_root, "contact_sheet.jpg"), sheet)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Manual annotation CSV, long or wide format")
    ap.add_argument("--image_root", required=True, help="Directory used to resolve relative image paths")
    ap.add_argument("--out", required=True, help="Output dataset directory")
    ap.add_argument("--copy_images", type=int, default=1, help="1=copy images, 0=symlink absolute source images")
    args = ap.parse_args(argv)

    entries = parse_annotations(args.csv)
    manifest = validate_and_write(entries, args.image_root, args.out, copy_images=bool(args.copy_images))
    subjects = {r["subject_id"] for r in manifest}
    print(json.dumps({
        "out": args.out,
        "rows": len(manifest),
        "subjects": len(subjects),
        "manifest": os.path.join(args.out, "manifest_landmarks.jsonl"),
        "contact_sheet": os.path.join(args.out, "contact_sheet.jpg"),
    }, indent=2))


if __name__ == "__main__":
    main()
