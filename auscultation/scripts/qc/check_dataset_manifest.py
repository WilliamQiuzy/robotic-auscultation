#!/usr/bin/env python3
"""Validate a generated perception dataset manifest."""
import argparse
import json
import os
from collections import Counter

import cv2


def check_dataset(root, max_rows=0, point_margin_px=0):
    manifest = os.path.join(root, "manifest.jsonl")
    rows = [json.loads(line) for line in open(manifest)]
    if max_rows:
        rows_to_check = rows[:max_rows]
    else:
        rows_to_check = rows
    errors = []
    regions = Counter()
    domains = Counter()
    visible_points = 0
    for idx, row in enumerate(rows_to_check):
        regions[row.get("region", "")] += 1
        domains[row.get("domain", "original")] += 1
        img_path = os.path.join(root, row["image"])
        iuv_path = os.path.join(root, row["iuv"])
        img = cv2.imread(img_path)
        iuv = cv2.imread(iuv_path)
        if img is None:
            errors.append(f"{idx}:{row['id']}: missing/unreadable image {row['image']}")
            continue
        if iuv is None:
            errors.append(f"{idx}:{row['id']}: missing/unreadable iuv {row['iuv']}")
            continue
        if img.shape[:2] != iuv.shape[:2]:
            errors.append(f"{idx}:{row['id']}: shape mismatch image={img.shape[:2]} iuv={iuv.shape[:2]}")
        part = iuv[:, :, 0]
        if int(part.max()) > 24:
            errors.append(f"{idx}:{row['id']}: invalid part max {int(part.max())}")
        if int((part > 0).sum()) == 0:
            errors.append(f"{idx}:{row['id']}: empty foreground IUV")
        dist = None
        for name, p in (row.get("points") or {}).items():
            if not p.get("visible"):
                continue
            visible_points += 1
            x = int(round(float(p["xy"][0])))
            y = int(round(float(p["xy"][1])))
            if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
                errors.append(f"{idx}:{row['id']}:{name}: point outside image ({x},{y})")
                continue
            if int(part[y, x]) == 0:
                errors.append(f"{idx}:{row['id']}:{name}: visible point not on body foreground ({x},{y})")
                continue
            if point_margin_px > 0:
                if dist is None:
                    dist = cv2.distanceTransform((part > 0).astype("uint8"), cv2.DIST_L2, 3)
                if float(dist[y, x]) < point_margin_px:
                    errors.append(
                        f"{idx}:{row['id']}:{name}: visible point too close to body boundary "
                        f"({dist[y, x]:.1f}px < {point_margin_px}px)"
                    )
    return {
        "root": root,
        "rows": len(rows),
        "checked": len(rows_to_check),
        "regions": dict(regions),
        "domains": dict(domains),
        "visible_points_checked": visible_points,
        "errors": errors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--point_margin_px", type=float, default=0.0,
                    help="Require visible points to be at least this far inside the IUV foreground.")
    ap.add_argument("--json_out", default="")
    args = ap.parse_args()
    result = check_dataset(args.root, args.max_rows, point_margin_px=args.point_margin_px)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
