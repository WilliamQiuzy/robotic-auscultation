#!/usr/bin/env python3
"""脊柱侧弯裸背 → 可训练样本(后背 8 肺点, 点监督)。

真实裸背照, 无 SMPL 参数 → 用 HMR2.0 单目拟合得 verts2d(6890,2, 已存 labels_hmr2.pkl),
取 8 个肺点顶点 = 后背听诊点。真实皮肤/深肤色/近景, 作后背 in-domain 微调主力。
珍贵(仅 86 张)→ 增强榨满。

⚠️ HMR2.0 只给 verts2d, 无稠密 IUV → 这些样本为【点监督】(iuv=null)。
   稠密 IUV 主力来自 HuMMan + synthetic。
"""
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import ManifestWriter, augment_once, write_sample  # noqa: E402
from humman_label import define_ausc_vertices  # noqa: E402


def _back_vertex_indices(smpl_model):
    import smplx
    model = smplx.SMPL(model_path=smpl_model, gender="neutral", batch_size=1)
    return define_ausc_vertices(model)["back"]   # {L1..L4,R1..R4: vertex_idx}


def _find_image(img_root, name):
    direct = os.path.join(img_root, name)
    if os.path.exists(direct):
        return direct
    hits = []
    for r, _d, fs in os.walk(img_root):
        if name in fs:
            hits.append(os.path.join(r, name))
    return hits[0] if hits else None


def generate(labels_pkl, img_root, smpl_model, out, n_aug=14,
             target=(384, 384), seed=0):
    """86 张裸背 → 每张 1 规范 + n_aug 随机组合增强(后背点监督)。"""
    labels = pickle.load(open(labels_pkl, "rb"))
    back_idx = _back_vertex_indices(smpl_model)
    rng = np.random.RandomState(seed)
    miss = 0
    with ManifestWriter(out) as mw:
        for name, lab in labels.items():
            path = _find_image(img_root, name)
            if path is None:
                miss += 1
                continue
            bgr = cv2.imread(path)
            if bgr is None:
                miss += 1
                continue
            rgb = bgr[:, :, ::-1]
            H, W = rgb.shape[:2]
            v2d = np.asarray(lab["verts2d"], dtype=np.float32)
            lw, lh = lab["wh"]
            sx, sy = W / float(lw), H / float(lh)            # verts2d 标在 wh 坐标 → 缩放到实图
            pts = {}
            for nm, vi in back_idx.items():
                x, y = float(v2d[vi, 0] * sx), float(v2d[vi, 1] * sy)
                pts[nm] = {"xy": (x, y), "visible": bool(0 <= x < W and 0 <= y < H)}
            base = {"image": rgb, "iuv": None, "points": pts, "region": "back"}
            stem = os.path.splitext(name)[0][:40]
            mw.add(write_sample(out, f"scol_{stem}_canon", base))
            for k in range(n_aug):
                mw.add(write_sample(out, f"scol_{stem}_aug{k}", augment_once(base, rng, target=target)))
        s = mw.summary()
    print(f"[scoliosis] {s}" + (f"  (missing imgs: {miss})" if miss else ""))
