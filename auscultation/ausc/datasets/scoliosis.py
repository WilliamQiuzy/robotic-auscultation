#!/usr/bin/env python3
"""脊柱侧弯裸背 → 可训练样本(后背 8 肺点 + **稠密 IUV**)。

真实裸背照, 无 SMPL 参数 → HMR2.0 单目拟合(YOLO 检测正确 bbox, 见 scripts/gpu/refit_scoliosis.py)
得相机系 3D 顶点 `verts_cam`(6890,3) → 光栅化稠密 IUV + 投影读 8 个后背听诊点。
价值: **真实皮肤纹理 / 深肤色 / 真实背**(域适应主力)。

⚠️ 标签来源限制: SMPL 直脊柱拟合侧弯有**小曲率误差**(听诊容差内, 可接受)。
   重拟合用 YOLO bbox(非整图)修掉了旧版的坏拟合。labels 格式(scol_refit.pkl):
   {name: {verts_cam:(6890,3), verts2d:(6890,2), wh:(W,H), focal:f}}。
"""
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import ManifestWriter, augment_once, write_sample  # noqa: E402
from humman_label import define_ausc_vertices  # noqa: E402
from render_iuv import load_uv_mapping, rasterize_iuv  # noqa: E402


def _back_vertex_indices(smpl_model):
    import smplx
    model = smplx.SMPL(model_path=smpl_model, gender="neutral", batch_size=1)
    return define_ausc_vertices(model)["back"]   # {L1..L4,R1..R4: vertex_idx}


def _find_image(img_root, name):
    direct = os.path.join(img_root, name)
    if os.path.exists(direct):
        return direct
    for r, _d, fs in os.walk(img_root):
        if name in fs:
            return os.path.join(r, name)
    return None


def _cam_from(focal, W, H):
    K = np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]], np.float32)
    return {"K": K, "R": np.eye(3, dtype=np.float32), "T": np.zeros(3, np.float32)}


def generate(labels_pkl, img_root, smpl_model, uv, out, n_aug=12,
             target=(384, 384), seed=0):
    """86 张裸背 → 每张 1 规范 + n_aug 随机组合增强(后背 8 点 + 稠密 IUV)。"""
    labels = pickle.load(open(labels_pkl, "rb"))
    back_idx = _back_vertex_indices(smpl_model)
    uvmap = load_uv_mapping(uv)
    rng = np.random.RandomState(seed)
    miss = 0
    with ManifestWriter(out) as mw:
        for name, lab in labels.items():
            path = _find_image(img_root, name)
            bgr = cv2.imread(path) if path else None
            if bgr is None:
                miss += 1
                continue
            rgb = bgr[:, :, ::-1]
            H, W = rgb.shape[:2]
            lw, lh = lab["wh"]
            sx, sy = W / float(lw), H / float(lh)
            cam = _cam_from(lab["focal"] * (sx + sy) / 2.0, W, H)        # 缩放到实图尺度
            vc = np.asarray(lab["verts_cam"], np.float32).copy()
            # verts_cam 投影是相对 lw,lh; 用缩放 K 对齐到实图(focal 已按比例缩放)
            I, U, V, _ = rasterize_iuv(vc, cam, uvmap, H, W)
            iuv = np.dstack([I, (U * 255).astype(np.uint8), (V * 255).astype(np.uint8)])
            v2d = np.asarray(lab["verts2d"], np.float32)
            pts = {}
            for nm, vi in back_idx.items():
                x, y = float(v2d[vi, 0] * sx), float(v2d[vi, 1] * sy)
                pts[nm] = {"xy": (x, y), "visible": bool(0 <= x < W and 0 <= y < H)}
            base = {"image": rgb, "iuv": iuv, "points": pts, "region": "back"}
            stem = os.path.splitext(name)[0][:40]
            mw.add(write_sample(out, f"scol_{stem}_canon", base))
            for k in range(n_aug):
                mw.add(write_sample(out, f"scol_{stem}_aug{k}", augment_once(base, rng, target=target)))
        s = mw.summary()
    print(f"[scoliosis] {s}" + (f"  (missing imgs: {miss})" if miss else ""))
