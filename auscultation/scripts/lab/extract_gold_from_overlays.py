#!/usr/bin/env python3
"""Recover THE posterior-lung gold from the user-designated overlay images.

The `_testset_v6a/imgs/*.jpg` overlays are the authoritative gold: each shows the
10 V6A points hand-designated as blue (L1-L5) and red (R1-R5) dots. Earlier we
mistakenly *re-derived* points from the current (wider) V6A addresses, which did
NOT match these overlays. This script instead reads the points straight off the
designated overlays via colour-blob detection, so the gold == what is drawn.

Output: <out>/v6a_gold_labels.json  (id -> 10 points, normalised + pixel) and
        <out>/verify/<id>.png       (extracted crosses over the overlay, to check).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

ORDER_L = ["L1", "L2", "L3", "L4", "L5"]
ORDER_R = ["R1", "R2", "R3", "R4", "R5"]


def _dots(mask, W, H):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _lab, st, ce = cv2.connectedComponentsWithStats(mask, 8)
    cand = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        ar, fill = w / max(1, h), a / max(1, w * h)
        x, y = ce[i]
        if x < W * 0.12 and y < H * 0.12:          # drop the top-left "NN" index text
            continue
        if 10 < a < 300 and 0.5 < ar < 2.0 and fill > 0.45:   # filled, near-circular dot
            cand.append((float(x), float(y), a))
    cand.sort(key=lambda p: -p[2])                 # biggest blobs first (dots > line bits)
    cand = cand[:5]
    cand.sort(key=lambda p: p[1])                  # top-to-bottom -> level 1..5
    return cand


def extract(path):
    im = cv2.imread(path)
    H, W = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 90, 70), (135, 255, 255))
    red = cv2.inRange(hsv, (0, 140, 90), (8, 255, 255)) | \
        cv2.inRange(hsv, (170, 140, 90), (180, 255, 255))
    return _dots(blue, W, H), _dots(red, W, H), (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True, help="dir of designated overlay jpgs (00..85)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=None, help="calibD v8 split json for tiers")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "verify"), exist_ok=True)

    tier = {}
    if args.split and os.path.exists(args.split):
        sp = json.load(open(args.split))
        for t in ("manual_candidate", "low_quality_candidate", "reject_for_gold"):
            for i in sp.get(t, []):
                tier[i] = t

    gold = {"_meta": {
        "schema": "posterior_lung_v6a_designated_overlay",
        "standard": "THE gold; 10pt read directly off user-designated _testset_v6a/imgs overlays",
        "space": "xy_norm = xy / image WH of the overlay (resolution-independent)",
        "source": "blue=L1-L5, red=R1-R5 dots in the overlays; missing dots -> visible:false",
    }}
    files = sorted(glob.glob(os.path.join(args.imgs, "*.jpg")))
    n_full = 0
    for f in files:
        tid = os.path.splitext(os.path.basename(f))[0]
        L, R, (W, H) = extract(f)
        pts = {}
        for names, dots in ((ORDER_L, L), (ORDER_R, R)):
            for k, nm in enumerate(names):
                if k < len(dots):
                    x, y, _a = dots[k]
                    pts[nm] = {"xy": [round(x, 1), round(y, 1)],
                               "xy_norm": [round(x / W, 4), round(y / H, 4)], "visible": True}
                else:
                    pts[nm] = {"xy": None, "xy_norm": None, "visible": False}
        n_vis = sum(1 for p in pts.values() if p["visible"])
        n_full += int(n_vis == 10)
        gold[tid] = {"tier": tier.get(tid, "unknown"), "n_visible": n_vis, "wh": [W, H], "points": pts}
        # verify overlay
        im = cv2.imread(f)
        for nm, p in pts.items():
            if not p["visible"]:
                continue
            x, y = int(p["xy"][0]), int(p["xy"][1])
            col = (0, 255, 0) if nm[0] == "R" else (0, 255, 255)
            cv2.drawMarker(im, (x, y), col, cv2.MARKER_CROSS, 12, 2)
        cv2.imwrite(os.path.join(args.out, "verify", f"{tid}.png"), im)
    json.dump(gold, open(os.path.join(args.out, "v6a_gold_labels.json"), "w"), indent=1)
    print(f"extracted {len(files)} overlays | full-10pt: {n_full} | -> {args.out}/v6a_gold_labels.json")


if __name__ == "__main__":
    main()
