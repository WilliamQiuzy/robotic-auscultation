#!/usr/bin/env python3
"""把 86 张兜底后点位图拼成带标签的总览 contact sheet + 问题图单独大图。"""
import os, re, glob
import numpy as np, cv2

D = "data/lab_sample_data/_REVIEW_86_v2"
PTS = os.path.join(D, "points")

# index → stem (0-based, 文件名排序)
idx2stem = {}
for line in open(os.path.join(D, "index_map.txt")):
    i, stem = line.split(maxsplit=1)
    idx2stem[int(i)] = stem.strip()

# stem 短号(IMG_1536) → flag 类型
flag = {}
for line in open(os.path.join(D, "run.log")):
    m = re.search(r"(scol_IMG_\d+).*FLAGS=(\w+)", line)
    if m:
        flag[m.group(1)] = m.group(2)

def short(stem):
    m = re.search(r"(IMG_\d+)", stem)
    return m.group(1) if m else stem[:10]

def find_png(stem):
    g = glob.glob(os.path.join(PTS, stem + "*_3points.png"))
    return g[0] if g else None

BORDER = {"v_respread": (0, 150, 255), "lowfov": (235, 0, 235)}   # 橙 / 品红
PROBLEM = {21, 25, 29, 60, 68, 71, 73, 74, 78, 85}               # 你之前点名的

def tile(idx, sz):
    stem = idx2stem[idx]
    sh = short(stem)
    short_key = "scol_" + sh
    f = find_png(stem)
    img = cv2.imread(f) if f else np.full((sz, sz, 3), 60, np.uint8)
    img = cv2.resize(img, (sz, sz))
    fl = flag.get(short_key)
    bcol = BORDER.get(fl, (90, 90, 90))
    bw = 5 if fl else (4 if idx in PROBLEM else 1)
    if idx in PROBLEM and not fl:
        bcol = (0, 255, 255)                                     # 黄: 你点名但本次无 flag
    img = cv2.copyMakeBorder(img, bw, bw, bw, bw, cv2.BORDER_CONSTANT, value=bcol)
    img = cv2.resize(img, (sz, sz))
    lab = f"{idx:02d} {sh}" + (f" [{fl}]" if fl else "")
    cv2.rectangle(img, (0, 0), (sz, 16), (0, 0, 0), -1)
    cv2.putText(img, lab, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return img

# ---- contact sheet 9x10 ----
SZ, COLS = 150, 9
rows = []
for r0 in range(0, 86, COLS):
    rowtiles = [tile(i, SZ) for i in range(r0, min(r0 + COLS, 86))]
    while len(rowtiles) < COLS:
        rowtiles.append(np.full((SZ, SZ, 3), 30, np.uint8))
    rows.append(np.hstack(rowtiles))
sheet = np.vstack(rows)
banner = np.full((34, sheet.shape[1], 3), 20, np.uint8)
cv2.putText(banner, "86 backs w/ deploy fallback | orange=V-respread  magenta=lowfov(lower back absent)  yellow=you-flagged(no trigger)",
            (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
sheet = np.vstack([banner, sheet])
cv2.imwrite(os.path.join(D, "_CONTACT_SHEET.png"), sheet)
print("contact sheet:", sheet.shape, "->", os.path.join(D, "_CONTACT_SHEET.png"))

# ---- 问题图单独大图 (你点名的 10 张) ----
probs = sorted(PROBLEM)
SZ2, COLS2 = 300, 5
rows = []
for r0 in range(0, len(probs), COLS2):
    rowtiles = [tile(probs[k], SZ2) for k in range(r0, min(r0 + COLS2, len(probs)))]
    while len(rowtiles) < COLS2:
        rowtiles.append(np.full((SZ2, SZ2, 3), 30, np.uint8))
    rows.append(np.hstack(rowtiles))
big = np.vstack(rows)
cv2.imwrite(os.path.join(D, "_PROBLEM_10_big.png"), big)
print("problem big:", big.shape, "->", os.path.join(D, "_PROBLEM_10_big.png"))
