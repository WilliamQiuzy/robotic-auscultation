#!/usr/bin/env python3
"""Build a subject-level review pack for real back perception data.

The perception bottleneck is real back landmark quality. This script makes that
auditable: one representative row per subject, a CSV review sheet, and contact
sheets with image/IUV/landmark overlays.
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

import cv2
import numpy as np


REVIEW_FIELDS = [
    "subject",
    "id",
    "image",
    "iuv",
    "region",
    "side_conf",
    "visible_points",
    "width",
    "height",
    "min_dim",
    "area_px",
    "auto_flags",
    "overlay",
    "review_status",
    "exclude_reason",
    "notes",
]


def subject_key(sid: str) -> str:
    """Match training subject split keys so QC and train filtering agree."""
    m = re.match(r"(syn\d+)", sid)
    if m:
        return m.group(1)
    m = re.match(r"(p\d+)_a\d+", sid)
    if m:
        return m.group(1)
    prev = None
    cur = sid
    while cur != prev:
        prev = cur
        cur = re.sub(r"_(aug\d+|canon|neg|clin\d+)$", "", cur)
    return cur


def load_valid_rows(root: str, region: str = "back"):
    rows = []
    with open(os.path.join(root, "manifest.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if not (r.get("is_valid") and r.get("iuv")):
                continue
            if region and r.get("region") != region:
                continue
            rows.append(r)
    return rows


def choose_representatives(rows):
    """Prefer canonical rows; otherwise choose the row with most visible points."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[subject_key(r["id"])].append(r)

    chosen = []
    for sub, rs in sorted(grouped.items()):
        def score(r):
            visible = sum(1 for p in (r.get("points") or {}).values() if p.get("visible"))
            canon = 1 if r["id"].endswith("_canon") or "_canon" in r["id"] else 0
            conf = float(r.get("side_conf", 0.0) or 0.0)
            return canon, visible, conf, r["id"]

        best = max(rs, key=score)
        chosen.append((sub, best))
    return chosen


def draw_overlay(root: str, row: dict, out_path: str, max_side: int = 640):
    img_path = os.path.join(root, row["image"])
    iuv_path = os.path.join(root, row["iuv"])
    img = cv2.imread(img_path)
    iuv = cv2.imread(iuv_path)
    if img is None:
        raise FileNotFoundError(img_path)
    if iuv is None:
        raise FileNotFoundError(iuv_path)

    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale != 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        iuv = cv2.resize(iuv, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    vis = img.copy()
    mask = iuv[:, :, 0] > 0
    tint = np.zeros_like(vis)
    tint[mask] = (255, 180, 50)
    vis = cv2.addWeighted(vis, 0.82, tint, 0.18, 0)

    sx, sy = img.shape[1] / float(w), img.shape[0] / float(h)
    for name, point in sorted((row.get("points") or {}).items()):
        if not point.get("visible"):
            continue
        x = int(round(point["xy"][0] * sx))
        y = int(round(point["xy"][1] * sy))
        cv2.circle(vis, (x, y), 6, (30, 230, 30), 2)
        cv2.putText(vis, name, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 230, 30), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)
    return w, h


def make_contact_sheets(overlay_paths, out_dir, cols=4, thumb_w=320):
    os.makedirs(out_dir, exist_ok=True)
    sheets = []
    thumbs = []
    for p in overlay_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        scale = thumb_w / float(img.shape[1])
        thumb = cv2.resize(img, (thumb_w, max(1, round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        thumbs.append((p, thumb))

    for start in range(0, len(thumbs), cols * cols):
        batch = thumbs[start:start + cols * cols]
        if not batch:
            continue
        max_h = max(t.shape[0] for _, t in batch)
        rows = []
        for r0 in range(0, len(batch), cols):
            cells = []
            for _, thumb in batch[r0:r0 + cols]:
                pad = np.full((max_h, thumb_w, 3), 245, dtype=np.uint8)
                pad[:thumb.shape[0], :thumb.shape[1]] = thumb
                cells.append(pad)
            while len(cells) < cols:
                cells.append(np.full((max_h, thumb_w, 3), 245, dtype=np.uint8))
            rows.append(np.concatenate(cells, axis=1))
        sheet = np.concatenate(rows, axis=0)
        path = os.path.join(out_dir, f"contact_sheet_{len(sheets):03d}.jpg")
        cv2.imwrite(path, sheet)
        sheets.append(path)
    return sheets


def auto_flags(row: dict, width: int, height: int, visible_points: int):
    flags = []
    min_dim = min(width, height)
    area = width * height
    side_conf = float(row.get("side_conf", 1.0) or 0.0)
    if min_dim < 128:
        flags.append("tiny")
    elif min_dim < 180 or area < 40000:
        flags.append("low_res")
    if visible_points < 8:
        flags.append("few_visible")
    if side_conf < 0.9:
        flags.append("low_side_conf")
    return ";".join(flags)


def build_review(root: str, out: str, max_subjects=None):
    rows = load_valid_rows(root, region="back")
    reps = choose_representatives(rows)
    if max_subjects is not None:
        reps = reps[:max_subjects]

    overlay_dir = os.path.join(out, "overlays")
    records = []
    overlay_paths = []
    for subject, row in reps:
        overlay_rel = os.path.join("overlays", f"{subject}.jpg")
        overlay_abs = os.path.join(out, overlay_rel)
        width, height = draw_overlay(root, row, overlay_abs)
        visible = sum(1 for p in (row.get("points") or {}).values() if p.get("visible"))
        flags = auto_flags(row, width, height, visible)
        records.append({
            "subject": subject,
            "id": row["id"],
            "image": row["image"],
            "iuv": row["iuv"],
            "region": row.get("region", ""),
            "side_conf": row.get("side_conf", ""),
            "visible_points": visible,
            "width": width,
            "height": height,
            "min_dim": min(width, height),
            "area_px": width * height,
            "auto_flags": flags,
            "overlay": overlay_rel,
            "review_status": "needs_review",
            "exclude_reason": "",
            "notes": "",
        })
        overlay_paths.append(overlay_abs)

    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "review_sheet.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    make_contact_sheets(overlay_paths, out)
    return csv_path, len(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset directory containing manifest.jsonl/img/iuv")
    ap.add_argument("--out", required=True, help="Output review directory")
    ap.add_argument("--max_subjects", type=int, default=None)
    args = ap.parse_args()
    csv_path, n = build_review(args.root, args.out, max_subjects=args.max_subjects)
    print(f"[realback_review] wrote {n} subject rows -> {csv_path}")


if __name__ == "__main__":
    main()
