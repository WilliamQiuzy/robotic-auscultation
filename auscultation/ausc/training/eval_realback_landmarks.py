#!/usr/bin/env python3
"""Evaluate Perception landmarks on real lab back images with manual point labels.

This evaluator intentionally does not require dense IUV ground truth.  It uses
the model's predicted IUV maps to read out the standard posterior lung addresses
and compares them against manually labeled 2D points.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "models"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "training"))

from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from eval import _forward_maps, landmark_addresses, readout, readout_soft  # noqa: E402
from iuv_model import DenseIUVModel  # noqa: E402


BACK_POINTS = ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4")


def load_model(ckpt_path, encoder, device):
    ck = torch.load(ckpt_path, map_location=device)
    if ck.get("kind") == "lora":
        from iuv_model import apply_lora
        model = DenseIUVModel(encoder=encoder, frozen_encoder=False).to(device)
        apply_lora(model, r=ck["lora_rank"], alpha=ck["lora_alpha"])
        model.load_state_dict(ck["lora_head"], strict=False)
    else:
        model = DenseIUVModel(encoder=encoder, frozen_encoder=True).to(device)
        if "model" in ck:
            model.load_state_dict(ck["model"])
        else:
            model.head.load_state_dict(ck["head"])
    model.to(device)
    model.eval()
    return model


def load_rows(root, manifest_name="manifest_landmarks.jsonl"):
    manifest = os.path.join(root, manifest_name)
    with open(manifest) as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r.get("is_valid", True) and r.get("region") == "back"]
    if not rows:
        raise ValueError(f"no valid back rows in {manifest}")
    return rows


def preprocess_image(path, res):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    rgb = bgr[:, :, ::-1]
    h0, w0 = rgb.shape[:2]
    img = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)
    return rgb, x, w0, h0


def point_summary(errors, res):
    arr = np.asarray(errors, np.float32)
    if len(arr) == 0:
        return {}
    return {
        "n_pts": int(len(arr)),
        "med_px": float(np.median(arr)),
        "mean_px": float(arr.mean()),
        "pct_img": float(np.median(arr) / res * 100),
        "pck@5%": float((arr < 0.05 * res).mean()),
        "pck@10%": float((arr < 0.10 * res).mean()),
    }


@torch.no_grad()
def evaluate_landmark_manifest(model, rows, root, addr, res, device, tta_scales=None,
                               readout_mode="soft", tau=0.01, dump=None, max_n=None):
    if dump:
        os.makedirs(dump, exist_ok=True)

    gy, gx = np.mgrid[0:res, 0:res].astype(np.float32)
    all_err = []
    by_point = defaultdict(list)
    per_image = []

    for idx, row in enumerate(rows[:max_n] if max_n else rows):
        img_path = row["image"] if os.path.isabs(row["image"]) else os.path.join(root, row["image"])
        rgb, x, w0, h0 = preprocess_image(img_path, res)
        x = x.to(device)
        part_prob, pred_uv = _forward_maps(model, x, res, tta_scales)
        pred_part = part_prob.argmax(0).astype(np.int32)

        sx = res / float(w0)
        sy = res / float(h0)
        image_err = []
        pred_points = {}
        for name in BACK_POINTS:
            payload = row.get("points", {}).get(name)
            if not payload or not payload.get("visible", True) or name not in addr:
                continue
            if readout_mode == "soft":
                pred = readout_soft(part_prob[addr[name][0]], pred_uv, addr[name], gx, gy, tau=tau)
            else:
                pred = readout(pred_part, pred_uv, addr[name])
            if pred is None:
                continue
            gt = (float(payload["xy"][0]) * sx, float(payload["xy"][1]) * sy)
            err = float(((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2) ** 0.5)
            all_err.append(err)
            by_point[name].append(err)
            image_err.append(err)
            pred_points[name] = {"pred_xy_res": [float(pred[0]), float(pred[1])],
                                 "gt_xy_res": [float(gt[0]), float(gt[1])],
                                 "err_px": err}

        per_image.append({
            "id": row["id"],
            "subject_id": row.get("subject_id"),
            "n_pts": len(image_err),
            "med_px": float(np.median(image_err)) if image_err else None,
            "mean_px": float(np.mean(image_err)) if image_err else None,
            "points": pred_points,
        })

        if dump and idx < 12:
            _dump_overlay(rgb, row, pred_points, os.path.join(dump, f"{idx:03d}_{row['id']}.jpg"), res)

    return {
        "n_images": len(per_image),
        "overall": point_summary(all_err, res),
        "by_point": {name: point_summary(errs, res) for name, errs in by_point.items()},
        "per_image": per_image,
    }


def _dump_overlay(rgb, row, pred_points, path, res):
    canvas = cv2.resize(rgb[:, :, ::-1], (res, res), interpolation=cv2.INTER_AREA)
    for name, payload in pred_points.items():
        gx, gy = [int(round(v)) for v in payload["gt_xy_res"]]
        px, py = [int(round(v)) for v in payload["pred_xy_res"]]
        cv2.circle(canvas, (gx, gy), 7, (0, 255, 0), 2)
        cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)
        cv2.line(canvas, (gx, gy), (px, py), (255, 255, 255), 1)
        cv2.putText(canvas, name, (gx + 7, gy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.imwrite(path, canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True, help="Lab gold dataset root")
    ap.add_argument("--manifest", default="manifest_landmarks.jsonl")
    ap.add_argument("--tag", default="realback_gold")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--smpl", default="data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--readout", default="soft", choices=["soft", "argmin"])
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--tta", default=None)
    ap.add_argument("--max_n", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(args.root, args.manifest)
    model = load_model(args.ckpt, args.encoder, device)
    addr, _ = landmark_addresses(args.uv, args.smpl)
    tta_scales = tuple(float(x) for x in args.tta.split(",")) if args.tta else None
    result = evaluate_landmark_manifest(
        model, rows, args.root, addr, args.res, device,
        tta_scales=tta_scales, readout_mode=args.readout, tau=args.tau,
        dump=args.dump, max_n=args.max_n,
    )

    print(f"\n===== [{args.tag}] landmark-only realback n={result['n_images']} =====")
    if result["overall"]:
        d = result["overall"]
        print(f"  med {d['med_px']:.1f}px ({d['pct_img']:.1f}% img) mean {d['mean_px']:.1f}px "
              f"| PCK@5% {d['pck@5%']:.2f} PCK@10% {d['pck@10%']:.2f} (n={d['n_pts']})")

    out = args.out or os.path.join("outputs", f"eval_{args.tag}_landmark_only.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
