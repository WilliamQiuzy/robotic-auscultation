#!/usr/bin/env python3
"""Generate clothed clinical-domain perception samples from baked IUV data.

This is a data-domain augmentation for the expected lab deployment setting:
clothed back, indoor clinical drape/wall background, and unchanged anatomical
landmarks under clothing. It intentionally does not alter IUV labels or points.
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
from collections import Counter

import cv2
import numpy as np


TORSO_PARTS = {1, 2}
ARM_PARTS = set(range(15, 23))
CLOTHE_PARTS = TORSO_PARTS | ARM_PARTS


def subject_key(sid: str) -> str:
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


def load_review_exclusions(path, exclude_auto_flags=(), exclude_statuses=()):
    if not path:
        return set()
    flags = {x.strip() for x in exclude_auto_flags if x.strip()}
    statuses = {x.strip().lower() for x in exclude_statuses if x.strip()}
    excluded = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row_flags = {x.strip() for x in (row.get("auto_flags") or "").split(";") if x.strip()}
            status = (row.get("review_status") or "").strip().lower()
            if (flags and row_flags & flags) or (statuses and status in statuses):
                excluded.add(row["subject"])
    return excluded


def load_iuv_part(path):
    iuv = cv2.imread(path, cv2.IMREAD_COLOR)
    if iuv is None:
        raise FileNotFoundError(path)
    return iuv[:, :, 0].astype(np.int32), iuv


def fabric_texture(h, w, rng, base):
    grain = rng.normal(0, 7, (h, w, 1))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    freq = rng.uniform(0.04, 0.09)
    folds = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * freq + rng.uniform(0, 6)) * 6
    tex = base.reshape(1, 1, 3) + grain + folds[..., None]
    return np.clip(tex, 0, 255)


def clothe(rgb, part, rng):
    h, w = part.shape
    mask = np.isin(part, list(CLOTHE_PARTS))
    if int(mask.sum()) < 50:
        return rgb, mask

    palette = np.array(
        [
            [112, 112, 118],
            [96, 100, 110],
            [130, 128, 126],
            [80, 86, 96],
            [150, 148, 150],
        ],
        np.float32,
    )
    base = palette[rng.integers(len(palette))]
    tex = fabric_texture(h, w, rng, base)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    fg = gray[mask]
    if fg.size:
        lo, hi = np.percentile(fg, 5), np.percentile(fg, 95)
        shade = np.clip((gray - lo) / max(1.0, hi - lo), 0.0, 1.0) * 0.5 + 0.7
    else:
        shade = np.ones_like(gray)
    shirt = np.clip(tex * shade[..., None], 0, 255)
    out = np.where(mask[..., None], shirt, rgb.astype(np.float32))

    if rng.random() < 0.3:
        ys, xs = np.where(np.isin(part, list(TORSO_PARTS)))
        if len(ys):
            y0 = int(np.percentile(ys, 18))
            cv2.line(
                out,
                (int(xs.min()), y0),
                (int(xs.max()), y0 + int(rng.integers(-6, 6))),
                (170, 40, 50),
                thickness=max(1, h // 160),
            )
    return np.clip(out, 0, 255).astype(np.uint8), mask


def put_background(rgb, fg_mask, plate, rng):
    h, w = rgb.shape[:2]
    ph, pw = plate.shape[:2]
    scale = max(h / ph, w / pw) * rng.uniform(1.0, 1.4)
    bg_big = cv2.resize(plate, (max(w, int(pw * scale)), max(h, int(ph * scale))))
    y0 = int(rng.integers(0, bg_big.shape[0] - h + 1))
    x0 = int(rng.integers(0, bg_big.shape[1] - w + 1))
    bg = bg_big[y0:y0 + h, x0:x0 + w].astype(np.float32) * rng.uniform(0.85, 1.12)
    bg = np.clip(bg, 0, 255)

    fg = fg_mask.astype(np.float32)
    alpha = cv2.GaussianBlur(fg, (0, 0), 1.2)[..., None]
    out = rgb.astype(np.float32) * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_rows(src, regions, canon_only, limit, excluded_subjects):
    with open(os.path.join(src, "manifest.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r.get("is_valid") and r.get("iuv")]
    if regions:
        rows = [r for r in rows if r.get("region") in regions]
    if canon_only:
        rows = [r for r in rows if r.get("id", "").endswith("_canon")]
    if excluded_subjects:
        rows = [r for r in rows if subject_key(r["id"]) not in excluded_subjects]
    if limit:
        rows = rows[:limit]
    return rows


def generate(args):
    os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "iuv"), exist_ok=True)
    plates = []
    for path in sorted(glob.glob(os.path.join(args.bg, "*.jpg"))):
        img = cv2.imread(path)
        if img is not None:
            plates.append(img[:, :, ::-1])
    if not plates:
        raise RuntimeError(f"no background plates found in {args.bg}")

    excluded = load_review_exclusions(
        args.review_csv,
        exclude_auto_flags=[x for x in args.exclude_auto_flags.split(",") if x],
        exclude_statuses=[x for x in args.exclude_statuses.split(",") if x],
    )
    regions = {x.strip() for x in args.regions.split(",") if x.strip()}
    rows = load_rows(args.src, regions, bool(args.canon_only), args.limit, excluded)
    rng = np.random.default_rng(args.seed)
    out_rows = []
    skipped = Counter()

    for idx, row in enumerate(rows):
        img_path = os.path.join(args.src, row["image"])
        iuv_path = os.path.join(args.src, row["iuv"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            skipped["missing_image"] += 1
            continue
        part, iuv_bgr = load_iuv_part(iuv_path)
        rgb = img_bgr[:, :, ::-1]
        if rgb.shape[:2] != part.shape:
            skipped["shape_mismatch"] += 1
            if args.strict_shapes:
                raise ValueError(f"image/IUV shape mismatch for {row['id']}: {rgb.shape[:2]} vs {part.shape}")
            continue
        fg = part > 0
        for k in range(args.n_per):
            clothed, _ = clothe(rgb, part, rng)
            plate = plates[int(rng.integers(len(plates)))]
            final = put_background(clothed, fg, plate, rng)
            new_id = f"{row['id']}_clin{k}"
            img_rel = f"img/{new_id}.jpg"
            iuv_rel = f"iuv/{new_id}.png"
            cv2.imwrite(os.path.join(args.out, img_rel), final[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            shutil.copyfile(iuv_path, os.path.join(args.out, iuv_rel))
            out_row = dict(row)
            out_row.update({
                "id": new_id,
                "image": img_rel,
                "iuv": iuv_rel,
                "source_id": row["id"],
                "domain": "clinical_clothed_drape",
                "clinical_variant": k,
            })
            out_rows.append(out_row)
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(rows)} source rows", flush=True)

    manifest = os.path.join(args.out, "manifest.jsonl")
    with open(manifest, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "src": args.src,
        "bg": args.bg,
        "out": args.out,
        "regions": sorted(regions),
        "canon_only": bool(args.canon_only),
        "n_per": args.n_per,
        "seed": args.seed,
        "review_csv": args.review_csv,
        "exclude_auto_flags": args.exclude_auto_flags,
        "exclude_statuses": args.exclude_statuses,
        "excluded_subjects": len(excluded),
        "source_rows": len(rows),
        "generated_rows": len(out_rows),
        "skipped": dict(skipped),
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[done] generated_rows={len(out_rows)} source_rows={len(rows)} skipped={dict(skipped)} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Dataset directory with manifest.jsonl/img/iuv")
    ap.add_argument("--bg", default="assets/clinical_bg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", default="back", help="Comma-separated regions; empty means all")
    ap.add_argument("--n_per", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--canon_only", type=int, default=1)
    ap.add_argument("--review_csv", default="")
    ap.add_argument("--exclude_auto_flags", default="", help="Comma-separated review auto flags to drop")
    ap.add_argument("--exclude_statuses", default="exclude,bad_label,bad_image,not_back,uncertain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg_quality", type=int, default=92)
    ap.add_argument("--strict_shapes", type=int, default=1)
    args = ap.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
