#!/usr/bin/env python3
"""Realign posterior lung points in dataset manifests from the current SMPL/IUV addresses.

Training supervises dense IUV; landmark points in manifest are used for QC and
evaluation. When the canonical posterior lung vertex definitions change, this
script rewrites back-region manifest points by reading each point back from the
row's IUV label.
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ausc"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "training"))

from eval import landmark_addresses  # noqa: E402


def _safe_margin_px(shape, frac=0.018, min_px=3, max_px=10):
    h, w = shape
    return int(np.clip(round(min(h, w) * frac), min_px, max_px))


def readout_iuv(iuv, addr, safe_margin_px=0):
    part, u, v = addr
    part_map = iuv[:, :, 0].astype(np.int32)
    candidate = part_map == int(part)
    if safe_margin_px > 0:
        fg = (part_map > 0).astype(np.uint8)
        dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
        safe_candidate = candidate & (dist >= float(safe_margin_px))
        if not safe_candidate.any():
            # Low-resolution or narrow masks may not support the full margin.
            # Relax gradually, but still prefer an interior body pixel.
            for margin in range(safe_margin_px - 1, 0, -1):
                safe_candidate = candidate & (dist >= float(margin))
                if safe_candidate.any():
                    break
        if safe_candidate.any():
            candidate = safe_candidate
    ys, xs = np.where(candidate)
    if len(xs) == 0:
        return None
    uv = iuv[:, :, 1:3].astype(np.float32) / 255.0
    d2 = (uv[ys, xs, 0] - float(u)) ** 2 + (uv[ys, xs, 1] - float(v)) ** 2
    j = int(np.argmin(d2))
    return float(xs[j]), float(ys[j])


def realign_rows(rows, default_root, addr, schema_name, margin_frac):
    back_names = [f"{side}{i}" for i in range(1, 6) for side in ("R", "L")]
    stats = Counter()
    out_rows = []

    for row in rows:
        row = dict(row)
        if not (row.get("is_valid") and row.get("region") == "back" and row.get("iuv")):
            out_rows.append(row)
            stats["unchanged"] += 1
            continue
        root = row.get("_root") or default_root
        iuv_path = os.path.join(root, row["iuv"])
        iuv = cv2.imread(iuv_path, cv2.IMREAD_COLOR)
        if iuv is None:
            stats["missing_iuv"] += 1
            out_rows.append(row)
            continue
        safe_margin = _safe_margin_px(iuv.shape[:2], frac=margin_frac)
        stats[f"safe_margin_px_{safe_margin}"] += 1
        points = {}
        for name in back_names:
            p = readout_iuv(iuv, addr[name], safe_margin_px=safe_margin)
            if p is None:
                points[name] = {"xy": [0.0, 0.0], "visible": False}
                stats["missing_point"] += 1
            else:
                points[name] = {"xy": [round(p[0], 1), round(p[1], 1)], "visible": True}
        row["points"] = points
        row["point_schema"] = schema_name
        out_rows.append(row)
        stats["realigned"] += 1
    return out_rows, dict(stats)


def realign_manifest(root, manifest_name, backup_suffix, smpl, uv, schema_name, margin_frac, dry_run=False):
    manifest = os.path.join(root, manifest_name)
    rows = [json.loads(line) for line in open(manifest)]
    addr, _ = landmark_addresses(uv, smpl)
    out_rows, stats = realign_rows(rows, root, addr, schema_name, margin_frac)

    if not dry_run:
        backup = manifest + backup_suffix
        if not os.path.exists(backup):
            shutil.copyfile(manifest, backup)
        with open(manifest, "w") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def realign_json(json_in, json_out, default_root, smpl, uv, schema_name, margin_frac, dry_run=False):
    rows = json.load(open(json_in))
    addr, _ = landmark_addresses(uv, smpl)
    out_rows, stats = realign_rows(rows, default_root, addr, schema_name, margin_frac)
    if not dry_run:
        with open(json_out, "w") as f:
            json.dump(out_rows, f, indent=2, ensure_ascii=False)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="", help="Dataset root containing manifest.jsonl")
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--json_in", default="", help="Optional val_split.json to realign instead of a manifest")
    ap.add_argument("--json_out", default="", help="Output path for --json_in")
    ap.add_argument("--backup_suffix", default=".pre_lung_standard_v3_safe_20260605")
    ap.add_argument("--schema_name", default="posterior_lung_standard_v6a")
    ap.add_argument("--safe_margin_frac", type=float, default=0.018,
                    help="Minimum body-mask distance as a fraction of min image side.")
    ap.add_argument("--smpl", default=os.path.join(ROOT, "data", "smpl", "SMPL_NEUTRAL.pkl"))
    ap.add_argument("--uv", default=os.path.join(ROOT, "data", "densepose_coco", "uv_data", "UV_Processed.mat"))
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()
    if args.json_in:
        if not args.json_out:
            raise SystemExit("--json_out is required with --json_in")
        stats = realign_json(
            args.json_in,
            args.json_out,
            args.root,
            args.smpl,
            args.uv,
            args.schema_name,
            args.safe_margin_frac,
            dry_run=bool(args.dry_run),
        )
        payload = {"json_in": args.json_in, "json_out": args.json_out, "stats": stats, "dry_run": bool(args.dry_run)}
    else:
        if not args.root:
            raise SystemExit("--root is required for manifest realignment")
        stats = realign_manifest(
            args.root,
            args.manifest,
            args.backup_suffix,
            args.smpl,
            args.uv,
            args.schema_name,
            args.safe_margin_frac,
            dry_run=bool(args.dry_run),
        )
        payload = {"root": args.root, "stats": stats, "dry_run": bool(args.dry_run)}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
