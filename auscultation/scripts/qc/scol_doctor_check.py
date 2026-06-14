#!/usr/bin/env python3
"""Scoliosis 临床准确性复核图(给医生看): 每个受试者一行
   [背部原图 | 稠密 IUV | 8 点听诊图(L1-L4 左肺 / R1-R4 右肺, 标注)]。
用于确认 HMR2 拟合的稠密 IUV 与后背 8 个肺听诊点定位是否解剖准确。
后视: 患者背对相机 → 患者左 = 图像左。

用法(服务器): python scripts/qc/scol_doctor_check.py [n_subjects]
"""
import os
import sys
import glob
import pickle

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "datasets"))

import smplx  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from humman_label import define_ausc_vertices  # noqa: E402
from render_iuv import load_uv_mapping, rasterize_iuv  # noqa: E402

DATA = os.path.join(ROOT, "data")
SMPL = os.path.join(DATA, "smpl", "SMPL_NEUTRAL.pkl")
UV = os.path.join(DATA, "densepose_coco", "uv_data", "UV_Processed.mat")
OUT = os.path.join(DATA, "_viz_synthetic", "SCOL_doctor_check.png")

PKL = "/root/scol_refit.pkl" if os.path.exists("/root/scol_refit.pkl") else os.path.join(DATA, "scol_refit.pkl")
IMG_ROOT = "/root/scol86" if os.path.isdir("/root/scol86") else os.path.join(DATA, "scoliosis_back")

# 8 个后背肺听诊点的临床含义(便于医生核对)
POINT_DESC = {
    "L1": "L upper (apex)", "L2": "L upper-mid", "L3": "L lower-mid", "L4": "L lower (base)",
    "R1": "R upper (apex)", "R2": "R upper-mid", "R3": "R lower-mid", "R4": "R lower (base)",
}


def colorize_iuv(iuv):
    part = iuv[:, :, 0]
    cmap = (np.array(plt.get_cmap("tab20").colors) * 255).astype(np.uint8)
    out = np.zeros((*part.shape, 3), np.uint8)
    for p in range(1, 25):
        out[part == p] = cmap[p % 20]
    return out


def find_image(name):
    direct = os.path.join(IMG_ROOT, name)
    if os.path.exists(direct):
        return direct
    for r, _d, fs in os.walk(IMG_ROOT):
        if name in fs:
            return os.path.join(r, name)
    return None


def build_sample(lab, name, back_idx, uvmap):
    path = find_image(name)
    bgr = cv2.imread(path) if path else None
    if bgr is None or "verts_cam" not in lab:
        return None
    rgb = bgr[:, :, ::-1]; H, W = rgb.shape[:2]
    lw, lh = lab["wh"]; sx, sy = W / float(lw), H / float(lh)
    f = lab["focal"] * (sx + sy) / 2.0
    cam = {"K": np.array([[f, 0, W / 2.], [0, f, H / 2.], [0, 0, 1.]], np.float32),
           "R": np.eye(3, dtype=np.float32), "T": np.zeros(3, np.float32)}
    I, U, V, _ = rasterize_iuv(np.asarray(lab["verts_cam"], np.float32).copy(), cam, uvmap, H, W)
    iuv = np.dstack([I, (U * 255).astype(np.uint8), (V * 255).astype(np.uint8)])
    v2d = np.asarray(lab["verts2d"], np.float32)
    pts = {nm: (float(v2d[vi, 0] * sx), float(v2d[vi, 1] * sy)) for nm, vi in back_idx.items()}
    return rgb, iuv, pts


def draw_points(img, pts):
    """8 点叠加: 左肺(L*)蓝, 右肺(R*)橙; 标号置于点外侧(L在左/R在右)避免叠字。"""
    im = img.copy()
    H, W = im.shape[:2]
    r = max(4, int(0.012 * max(H, W)))
    fs = max(0.4, 0.0011 * max(H, W))
    for nm, (x, y) in pts.items():
        x, y = int(x), int(y)
        is_L = nm.startswith("L")
        col = (40, 90, 230) if is_L else (240, 140, 30)
        cv2.circle(im, (x, y), r, col, -1, cv2.LINE_AA)
        cv2.circle(im, (x, y), r, (255, 255, 255), 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(nm, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        tx = x - r - 4 - tw if is_L else x + r + 4          # L 标左侧, R 标右侧
        ty = y + th // 2
        tx = int(np.clip(tx, 2, W - tw - 2))
        cv2.putText(im, nm, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, nm, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 0), 1, cv2.LINE_AA)
    return im


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    model = smplx.SMPL(model_path=SMPL, gender="neutral", batch_size=1)
    uvmap = load_uv_mapping(UV)
    back_idx = define_ausc_vertices(model)["back"]
    labs = pickle.load(open(PKL, "rb"))

    samples = []
    for name, lab in labs.items():
        s = build_sample(lab, name, back_idx, uvmap)
        if s is not None:
            samples.append((name, s))
        if len(samples) >= n:
            break
    if not samples:
        raise RuntimeError("no scoliosis sample with verts_cam + image")

    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4.2 * rows))
    axes = np.array(axes).reshape(rows, 3)
    for i, (name, (rgb, iuv, pts)) in enumerate(samples):
        axes[i, 0].imshow(rgb); axes[i, 0].set_title(f"{name[:18]}  back RGB", fontsize=9)
        axes[i, 1].imshow(colorize_iuv(iuv)); axes[i, 1].set_title("dense IUV (HMR2 fit)", fontsize=9)
        axes[i, 2].imshow(draw_points(rgb, pts)); axes[i, 2].set_title("8 lung auscultation points", fontsize=9)
        for j in range(3):
            axes[i, j].axis("off")
    legend = "  |  ".join(f"{k}={v}" for k, v in POINT_DESC.items())
    plt.suptitle("Scoliosis clinical check: bare back / dense IUV / 8 lung points\n"
                 "Posterior view (patient L = image L).  Blue=Left lung (L1-L4), Orange=Right lung (R1-R4)\n"
                 + legend, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", OUT, "rows:", rows)


if __name__ == "__main__":
    main()
