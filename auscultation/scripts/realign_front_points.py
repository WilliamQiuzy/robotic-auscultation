#!/usr/bin/env python3
"""Realign anterior (heart) landmark points in manifests from current SMPL/IUV addresses.

Counterpart to scripts/realign_lung_points.py (which handles posterior lung points).
When the canonical cardiac vertex definitions change (define_ausc_vertices['front']),
the manifest's stored front points become stale. Training supervises dense IUV, so
this only affects QC/eval; this script rewrites front-region points by reading each
point back from the row's IUV label at the new (part,U,V) address. IUV labels unchanged.

Two modes:
  --root DIR        : DIR/manifest.jsonl, IUV at DIR/<row.iuv>
  --rows_json FILE  : flat list (e.g. a fixed val_split.json), IUV at <row._root>/<row.iuv>
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ausc"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "training"))

from eval import landmark_addresses  # noqa: E402

FRONT_NAMES = ["aortic", "pulmonary", "erb", "tricuspid", "mitral"]


def readout_iuv(iuv, addr):
    """IUV png(BGR): part=ch0(B), U=ch1(G), V=ch2(R). addr=(part,u,v)。返回最近匹配像素 (x,y)。"""
    part, u, v = addr
    part_map = iuv[:, :, 0].astype(np.int32)
    ys, xs = np.where(part_map == int(part))
    if len(xs) == 0:
        return None
    uv = iuv[:, :, 1:3].astype(np.float32) / 255.0
    d2 = (uv[ys, xs, 0] - float(u)) ** 2 + (uv[ys, xs, 1] - float(v)) ** 2
    j = int(np.argmin(d2))
    return float(xs[j]), float(ys[j])


def iuv_path_for(row, root, rows_json_mode):
    if rows_json_mode:
        base = row.get("_root", root)
        return os.path.join(base, row["iuv"])
    return os.path.join(root, row["iuv"])


def realign(rows, addr, root, rows_json_mode, stats):
    out = []
    for row in rows:
        row = dict(row)
        if not (row.get("is_valid") and row.get("region") == "front" and row.get("iuv")):
            out.append(row); stats["unchanged"] += 1; continue
        iuv = cv2.imread(iuv_path_for(row, root, rows_json_mode), cv2.IMREAD_COLOR)
        if iuv is None:
            out.append(row); stats["missing_iuv"] += 1; continue
        pts = {}
        for nm in FRONT_NAMES:
            if nm not in addr:
                continue
            p = readout_iuv(iuv, addr[nm])
            if p is None:
                pts[nm] = {"xy": [0.0, 0.0], "visible": False}; stats["missing_point"] += 1
            else:
                pts[nm] = {"xy": [round(p[0], 1), round(p[1], 1)], "visible": True}
        row["points"] = pts
        row["front_point_schema"] = "cardiac_apetm_v2"
        row["point_schema"] = "cardiac_apetm_v2"
        out.append(row); stats["realigned"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="数据集目录(含 manifest.jsonl)")
    ap.add_argument("--rows_json", default=None, help="扁平行列表(如 fixed val_split.json), iuv 用 _root")
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--backup_suffix", default=".pre_cardiac_apetm_v2_20260605")
    ap.add_argument("--smpl", default=os.path.join(ROOT, "data", "smpl", "SMPL_NEUTRAL.pkl"))
    ap.add_argument("--uv", default=os.path.join(ROOT, "data", "densepose_coco", "uv_data", "UV_Processed.mat"))
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()
    addr, _ = landmark_addresses(args.uv, args.smpl)
    from collections import Counter
    stats = Counter()

    if args.rows_json:
        target = args.rows_json; rows = json.load(open(target))
        out = realign(rows, addr, args.root or "", True, stats)
        if not args.dry_run:
            bak = target + args.backup_suffix
            if not os.path.exists(bak):
                shutil.copyfile(target, bak)
            json.dump(out, open(target, "w"))
    else:
        assert args.root, "需 --root 或 --rows_json"
        target = os.path.join(args.root, args.manifest)
        rows = [json.loads(l) for l in open(target)]
        out = realign(rows, addr, args.root, False, stats)
        if not args.dry_run:
            bak = target + args.backup_suffix
            if not os.path.exists(bak):
                shutil.copyfile(target, bak)
            with open(target, "w") as f:
                for r in out:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"target": target, "stats": dict(stats), "dry_run": bool(args.dry_run)}, indent=2))


if __name__ == "__main__":
    main()
