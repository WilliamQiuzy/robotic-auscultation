#!/usr/bin/env python3
"""Create corrected posterior lung relabeling candidates for CalibC review images.

This is intentionally a review/annotation utility, not a training data generator.
It reads the original back images plus the old CalibC overlays, estimates a
coarse torso center/scale, lays out posterior lung points from a clinical
reference template, draws new overlays, and emits machine-readable labels.
"""
import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


POINT_ORDER = ["L1", "R1", "L2", "R2", "L3", "R3", "L4", "R4"]
POINT_SCHEMA = "posterior_lung_standard_v4_review"


def image_id(path):
    m = re.match(r"(\d+)", path.name)
    return m.group(1) if m else path.stem


def detect_color_components(bgr, color):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if color == "blue":
        mask = cv2.inRange(hsv, (85, 55, 40), (135, 255, 255))
    elif color == "red":
        mask1 = cv2.inRange(hsv, (0, 55, 40), (13, 255, 255))
        mask2 = cv2.inRange(hsv, (165, 55, 40), (179, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        raise ValueError(color)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    h, w = bgr.shape[:2]
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if not (12 <= area <= 180 and 3 <= ww <= 24 and 3 <= hh <= 24):
            continue
        cx, cy = cents[i]
        # Ignore page index text and obvious borders/watermarks.
        if cy < 0.08 * h or cy > 0.96 * h or cx < 0.02 * w or cx > 0.98 * w:
            continue
        comps.append((float(cx), float(cy), int(area), int(ww), int(hh)))
    return comps


def choose_four(comps):
    """Pick four likely dot components sorted vertically.

    The old overlays contain text labels in the same color. Dot components are
    usually compact and among the larger same-color components. We choose a
    combination with increasing y and plausible row spacing.
    """
    if len(comps) < 4:
        return None
    comps = sorted(comps, key=lambda c: (-c[2], c[1]))[:14]
    best = None
    from itertools import combinations

    for cand in combinations(comps, 4):
        cand = sorted(cand, key=lambda c: c[1])
        ys = [c[1] for c in cand]
        if min(np.diff(ys)) < 12:
            continue
        span = ys[-1] - ys[0]
        if span < 60:
            continue
        # Prefer rows with roughly monotonic, not wildly irregular spacing.
        diffs = np.diff(ys)
        reg = float(np.std(diffs) / (np.mean(diffs) + 1e-6))
        area_score = -sum(c[2] for c in cand) / 100.0
        score = reg + area_score
        if best is None or score < best[0]:
            best = (score, cand)
    if best is None:
        cand = sorted(comps, key=lambda c: c[1])[:4]
        return [(c[0], c[1]) for c in cand]
    return [(c[0], c[1]) for c in best[1]]


def fallback_points(w, h):
    cx = w * 0.50
    top = h * 0.26
    ys = [top, top + h * 0.10, top + h * 0.22, top + h * 0.36]
    offs = [w * 0.17, w * 0.11, w * 0.14, w * 0.21]
    return {
        "L1": [cx - offs[0], ys[0]],
        "R1": [cx + offs[0], ys[0]],
        "L2": [cx - offs[1], ys[1]],
        "R2": [cx + offs[1], ys[1]],
        "L3": [cx - offs[2], ys[2]],
        "R3": [cx + offs[2], ys[2]],
        "L4": [cx - offs[3], ys[3]],
        "R4": [cx + offs[3], ys[3]],
    }


def estimate_body_mask_and_bbox(bgr):
    """Estimate visible torso/skin region for clamping review points.

    The images are heterogeneous web/lab crops, so this is deliberately
    conservative: it combines non-background and skin-color cues, then keeps
    the largest plausible central component. The result is only a guardrail;
    labels remain review candidates rather than training gold.
    """
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    corners = np.concatenate(
        [
            rgb[: max(2, h // 20), : max(2, w // 20)].reshape(-1, 3),
            rgb[: max(2, h // 20), -max(2, w // 20) :].reshape(-1, 3),
            rgb[-max(2, h // 20) :, : max(2, w // 20)].reshape(-1, 3),
            rgb[-max(2, h // 20) :, -max(2, w // 20) :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    dist_bg = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
    non_bg = dist_bg > 24

    y, cr, cb = cv2.split(ycrcb)
    skin = (y > 45) & (cr > 125) & (cr < 185) & (cb > 70) & (cb < 150)
    sat = hsv[:, :, 1] > 18
    mask = ((non_bg & sat) | skin).astype(np.uint8) * 255
    mask[: max(1, int(0.02 * h)), :] = 0
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask.astype(bool), (w * 0.15, h * 0.05, w * 0.85, h * 0.92), "bbox_fallback"

    best = None
    for i in range(1, n):
        x, yy, ww, hh, area = stats[i]
        if area < max(40, 0.015 * w * h):
            continue
        cx = x + ww / 2.0
        central = 1.0 - min(abs(cx - w / 2.0) / max(w / 2.0, 1), 1.0)
        score = area * (0.6 + 0.4 * central)
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        return mask.astype(bool), (w * 0.15, h * 0.05, w * 0.85, h * 0.92), "bbox_fallback"

    keep = labels == best[1]
    ys, xs = np.where(keep)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return keep, (x0, y0, x1, y1), "bbox_mask"


def row_segment(mask, y, cx):
    h, w = mask.shape[:2]
    yy = int(np.clip(round(y), 0, h - 1))
    band = mask[max(0, yy - 2) : min(h, yy + 3), :].any(axis=0)
    if not band.any():
        return None
    cx_i = int(np.clip(round(cx), 0, w - 1))
    idxs = np.where(band)[0]
    runs = []
    start = int(idxs[0])
    prev = int(idxs[0])
    for x in idxs[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
            continue
        runs.append((start, prev))
        start = prev = x
    runs.append((start, prev))
    runs.sort(key=lambda r: (0 if r[0] <= cx_i <= r[1] else 1, abs((r[0] + r[1]) / 2.0 - cx_i)))
    return runs[0]


def clamp_point_to_body(x, y, cx, mask, bbox, margin):
    seg = row_segment(mask, y, cx)
    if seg is None:
        x0, _, x1, _ = bbox
    else:
        x0, x1 = seg
    lo = x0 + margin
    hi = x1 - margin
    if hi <= lo:
        lo, hi = x0, x1
    return float(np.clip(x, lo, hi))


def relabel_from_old(old_overlay, raw_size):
    bgr = cv2.imread(str(old_overlay), cv2.IMREAD_COLOR)
    w_raw, h_raw = raw_size
    if bgr is None:
        return fallback_points(w_raw, h_raw), "fallback_no_old_overlay"
    h_old, w_old = bgr.shape[:2]
    blue = choose_four(detect_color_components(bgr, "blue"))
    red = choose_four(detect_color_components(bgr, "red"))
    if blue is None or red is None:
        return fallback_points(w_raw, h_raw), "fallback_detection_failed"

    raw_bgr = None
    if old_overlay.exists():
        # The raw image is loaded by the caller in v4 mode; this fallback keeps
        # the older call path functional when only overlay data is available.
        raw_bgr = np.zeros((h_raw, w_raw, 3), dtype=np.uint8)

    # Scale old overlay coords to raw image coords.
    sx, sy = w_raw / w_old, h_raw / h_old
    left = [(x * sx, y * sy) for x, y in blue]
    right = [(x * sx, y * sy) for x, y in red]
    old_rows = [((left[i][1] + right[i][1]) / 2.0) for i in range(4)]
    row1, row4 = old_rows[0], old_rows[-1]
    span = max(row4 - row1, h_raw * 0.20)

    # New posterior lung rows: upper/mid/lower lung field, not full torso.
    # This pulls the old lower rows upward while preserving image-specific scale.
    new_rows = [
        row1 - 0.02 * span,
        row1 + 0.22 * span,
        row1 + 0.43 * span,
        row1 + 0.64 * span,
    ]
    # Keep row 4 safely above the old waist-proximal row.
    new_rows[3] = min(new_rows[3], row4 - 0.16 * span)

    # Preserve side-specific horizontal placement, but smooth width by row to
    # match posterior reference: row2 narrower, row4 wider.
    centers = [((left[i][0] + right[i][0]) / 2.0) for i in range(4)]
    cx = float(np.median(centers))
    old_offs = [abs(right[i][0] - left[i][0]) / 2.0 for i in range(4)]
    base = float(np.median(old_offs))
    offs = [
        max(old_offs[0], base * 0.95),
        min(old_offs[1], base * 0.78),
        max(old_offs[2], base * 0.90),
        max(old_offs[3], base * 1.18),
    ]
    # Do not push points beyond the likely torso frame.
    max_off = w_raw * 0.34
    min_off = w_raw * 0.09
    offs = [float(np.clip(o, min_off, max_off)) for o in offs]

    pts = {}
    for i in range(4):
        y = float(np.clip(new_rows[i], h_raw * 0.10, h_raw * 0.86))
        pts[f"L{i+1}"] = [round(cx - offs[i], 1), round(y, 1)]
        pts[f"R{i+1}"] = [round(cx + offs[i], 1), round(y, 1)]
    return pts, "old_overlay_adjusted"


def relabel_template(old_overlay, raw_path):
    raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if raw is None:
        with Image.open(raw_path) as im:
            return fallback_points(*im.size), "fallback_no_raw"
    h_raw, w_raw = raw.shape[:2]
    mask, bbox, bbox_method = estimate_body_mask_and_bbox(raw)

    bgr = cv2.imread(str(old_overlay), cv2.IMREAD_COLOR)
    old_method = "old_overlay_unused"
    old_rows = None
    old_center = None
    old_width = None
    if bgr is not None:
        h_old, w_old = bgr.shape[:2]
        blue = choose_four(detect_color_components(bgr, "blue"))
        red = choose_four(detect_color_components(bgr, "red"))
        if blue is not None and red is not None:
            sx, sy = w_raw / w_old, h_raw / h_old
            left = [(x * sx, y * sy) for x, y in blue]
            right = [(x * sx, y * sy) for x, y in red]
            old_rows = [((left[i][1] + right[i][1]) / 2.0) for i in range(4)]
            old_center = float(np.median([(left[i][0] + right[i][0]) / 2.0 for i in range(4)]))
            old_width = float(np.median([abs(right[i][0] - left[i][0]) for i in range(4)]))
            old_method = "old_overlay_center_scale"

    x0, y0, x1, y1 = bbox
    body_w = max(x1 - x0, w_raw * 0.45)
    body_h = max(y1 - y0, h_raw * 0.55)
    cx = old_center if old_center is not None else (x0 + x1) / 2.0
    cx = float(np.clip(cx, x0 + 0.25 * body_w, x1 - 0.25 * body_w))

    if old_rows is not None:
        old_span = max(old_rows[-1] - old_rows[0], h_raw * 0.22)
        top = min(old_rows[0] - 0.22 * old_span, y0 + 0.32 * body_h)
        bottom = min(old_rows[0] + 0.40 * old_span, y0 + 0.55 * body_h, y1 - 0.25 * body_h)
        bottom = max(bottom, top + 0.26 * body_h)
    else:
        tall_crop = body_h / max(body_w, 1.0) > 1.55
        top = y0 + (0.17 if tall_crop else 0.22) * body_h
        bottom = y0 + (0.42 if tall_crop else 0.50) * body_h

    top = float(np.clip(top, y0 + 0.14 * body_h, y0 + 0.46 * body_h))
    bottom = float(np.clip(bottom, top + 0.20 * body_h, y0 + 0.58 * body_h))
    lung_h = bottom - top
    ys = [
        top,
        top + 0.28 * lung_h,
        top + 0.58 * lung_h,
        bottom,
    ]

    # Posterior reference layout: not two vertical paraspinal columns. The
    # second row is medial; the fourth row is lower and more lateral.
    torso_w = max(w_raw * 0.42, body_w * 0.58)
    if old_width is not None:
        torso_w = max(torso_w, old_width * 1.35)
    torso_w = float(np.clip(torso_w, w_raw * 0.38, min(body_w * 0.72, w_raw * 0.66)))
    offs = [
        0.27 * torso_w,
        0.17 * torso_w,
        0.22 * torso_w,
        0.34 * torso_w,
    ]
    margin = max(3.0, min(w_raw, h_raw) * 0.035)

    pts = {}
    for i, (y, off) in enumerate(zip(ys, offs), 1):
        y = float(np.clip(y, y0 + 0.06 * body_h, y1 - 0.08 * body_h))
        lx = clamp_point_to_body(cx - off, y, cx, mask, bbox, margin)
        rx = clamp_point_to_body(cx + off, y, cx, mask, bbox, margin)
        # If segmentation collapses to one side, keep symmetric review points
        # inside the broader torso box instead of inheriting a one-sided error.
        if rx - lx < 0.16 * torso_w:
            lx = float(np.clip(cx - off, x0 + margin, x1 - margin))
            rx = float(np.clip(cx + off, x0 + margin, x1 - margin))
        pts[f"L{i}"] = [round(lx, 1), round(y, 1)]
        pts[f"R{i}"] = [round(rx, 1), round(y, 1)]
    return pts, f"{POINT_SCHEMA}:{old_method}:{bbox_method}"


def draw_overlay(raw_path, points, out_path, usable=True):
    im = Image.open(raw_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    colors = {
        "L": (0, 150, 255),
        "R": (255, 60, 60),
    }
    for name in POINT_ORDER:
        x, y = points[name]
        side = name[0]
        c = colors[side]
        r = max(4, int(min(im.size) * 0.014))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=c, outline=(255, 255, 255), width=1)
        draw.text((x + r + 2, y - r - 1), name, fill=c)
    if not usable:
        draw.rectangle((0, 0, im.width - 1, 24), fill=(255, 240, 190))
        draw.text((4, 4), "review-only / not gold", fill=(130, 70, 0))
    im.save(out_path, quality=94)


def make_contact_sheets(files, out_dir):
    thumb_w, thumb_h = 260, 348
    cols, rows = 4, 4
    for page, start in enumerate(range(0, len(files), cols * rows), 1):
        page_files = files[start : start + cols * rows]
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 26)), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, f in enumerate(page_files):
            im = Image.open(f).convert("RGB")
            im.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            x = (idx % cols) * thumb_w + (thumb_w - im.width) // 2
            y = (idx // cols) * (thumb_h + 26)
            sheet.paste(im, (x, y + 22))
            draw.rectangle(
                [(idx % cols) * thumb_w, y, (idx % cols + 1) * thumb_w - 1, y + 21],
                fill=(245, 245, 245),
            )
            draw.text(((idx % cols) * thumb_w + 4, y + 4), f.stem[:30], fill=(0, 0, 0))
        sheet.save(out_dir / f"contact_page_{page:02d}.jpg", quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--old_overlay_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--triage_csv", default="")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    old_dir = Path(args.old_overlay_dir)
    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "imgs"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    triage = {}
    if args.triage_csv and Path(args.triage_csv).exists():
        with open(args.triage_csv) as f:
            for row in csv.DictReader(f):
                triage[row["image"]] = row

    rows = []
    overlay_files = []
    raw_files = []
    for pat in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        raw_files.extend(raw_dir.glob(pat))
    for raw in sorted(raw_files):
        idx = image_id(raw)
        old = old_dir / f"{idx}.jpg"
        pts, method = relabel_template(old, raw)
        tr = triage.get(idx, {})
        usable = bool(tr) and tr.get("usable_as_gold") == "1"
        out_img = overlay_dir / f"{idx}.jpg"
        draw_overlay(raw, pts, out_img, usable=usable)
        overlay_files.append(out_img)
        row = {
            "id": idx,
            "image": str(raw.relative_to(raw_dir.parent.parent)),
            "overlay": str(out_img),
            "point_schema": POINT_SCHEMA,
            "usable_as_gold": usable,
            "method": method,
            "points": {k: {"xy": pts[k], "visible": True} for k in POINT_ORDER},
            "triage": tr,
        }
        rows.append(row)

    with open(out_dir / "labels.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(out_dir / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "point", "x", "y", "usable_as_gold", "method"])
        for row in rows:
            for name in POINT_ORDER:
                x, y = row["points"][name]["xy"]
                writer.writerow([row["id"], name, x, y, int(row["usable_as_gold"]), row["method"]])
    make_contact_sheets(overlay_files, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows), "overlays": len(overlay_files)}, indent=2))


if __name__ == "__main__":
    main()
