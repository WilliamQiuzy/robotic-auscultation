#!/usr/bin/env python3
"""Render auscultation landmark overlays from the current SMPL/IUV addresses.

This is useful when manifest point coordinates may be stale after the canonical
cardiac/lung vertex definitions change.  It reads each landmark back from the
row's dense IUV label, then draws the point on the RGB image.
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ausc"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "training"))

from eval import landmark_addresses  # noqa: E402


FRONT_ORDER = ("aortic", "pulmonary", "erb", "tricuspid", "mitral")
BACK_ORDER = ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4")


def readout_iuv(iuv, addr):
    part, u, v = addr
    part_map = iuv[:, :, 0].astype(np.int32)
    ys, xs = np.where(part_map == int(part))
    if len(xs) == 0:
        return None
    uv = iuv[:, :, 1:3].astype(np.float32) / 255.0
    d2 = (uv[ys, xs, 0] - float(u)) ** 2 + (uv[ys, xs, 1] - float(v)) ** 2
    j = int(np.argmin(d2))
    return float(xs[j]), float(ys[j])


def load_rows(root, region, max_rows):
    rows = []
    with open(os.path.join(root, "manifest.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            if not (row.get("is_valid") and row.get("region") == region and row.get("iuv")):
                continue
            if "_canon" not in row.get("id", ""):
                continue
            rows.append(row)
            if len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"no valid canon {region} rows found under {root}")
    return rows


def draw_overlay(root, row, addr, order, out_path):
    img = cv2.imread(os.path.join(root, row["image"]))
    iuv = cv2.imread(os.path.join(root, row["iuv"]), cv2.IMREAD_COLOR)
    if img is None or iuv is None:
        raise FileNotFoundError(row["id"])
    points = {}
    for name in order:
        if name not in addr:
            continue
        p = readout_iuv(iuv, addr[name])
        if p is None:
            continue
        points[name] = p

    canvas = img.copy()
    for name in order:
        if name not in points:
            continue
        x, y = [int(round(v)) for v in points[name]]
        color = (0, 220, 255) if name.startswith("R") else (60, 220, 60)
        if name in FRONT_ORDER:
            color = (0, 255, 0)
        cv2.circle(canvas, (x, y), 9, color, 2)
        cv2.putText(canvas, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imwrite(out_path, canvas)
    return points


def make_contact_sheet(paths, out_path, max_side=360, cols=3):
    thumbs = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        thumbs.append(cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                                 interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    cell_h = max(t.shape[0] for t in thumbs)
    cell_w = max(t.shape[1] for t in thumbs)
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * cell_h:r * cell_h + thumb.shape[0], c * cell_w:c * cell_w + thumb.shape[1]] = thumb
    cv2.imwrite(out_path, sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--region", required=True, choices=["front", "back"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rows", type=int, default=9)
    ap.add_argument("--smpl", default=os.path.join(ROOT, "data", "smpl", "SMPL_NEUTRAL.pkl"))
    ap.add_argument("--uv", default=os.path.join(ROOT, "data", "densepose_coco", "uv_data", "UV_Processed.mat"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    addr, _ = landmark_addresses(args.uv, args.smpl)
    order = FRONT_ORDER if args.region == "front" else BACK_ORDER
    rows = load_rows(args.root, args.region, args.max_rows)
    overlay_paths = []
    csv_rows = []
    for row in rows:
        out_path = os.path.join(args.out, f"{row['id']}.jpg")
        points = draw_overlay(args.root, row, addr, order, out_path)
        overlay_paths.append(out_path)
        for name, xy in points.items():
            csv_rows.append({"id": row["id"], "point": name, "x": round(xy[0], 1), "y": round(xy[1], 1)})
    with open(os.path.join(args.out, "points.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "point", "x", "y"])
        writer.writeheader()
        writer.writerows(csv_rows)
    make_contact_sheet(overlay_paths, os.path.join(args.out, "contact_sheet.jpg"))
    print(json.dumps({"root": args.root, "region": args.region, "rows": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
