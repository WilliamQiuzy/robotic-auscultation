#!/usr/bin/env python3
"""感知训练 dataloader: 读 manifest.jsonl + img/iuv, 出 (图, part 标签, UV 标签, 听诊点)。

⚠️ 数据集里的图【已经过增强】(生成器把增强烘焙进去了), 所以这里只做 load+resize+归一化,
   不再二次增强。
IUV png 通道(cv2 BGR 读回): [:,:,0]=part I(0..24) · [:,:,1]=U*255 · [:,:,2]=V*255。
resize 一律 NEAREST 保 part 标签离散 + 点同步缩放。
"""
import json
import os
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def subject_key(sid: str) -> str:
    """提取【真正的受试者/生成主体】键, 防 train/val 同人泄漏(关键!):
       - synthetic: `syn00000_...`        → `syn00000`(同一生成体的所有姿态/视角/取景/增强)
       - HuMMan:    `p000438_a000040_...`  → `p000438`(同一真人, 跨序列/帧/视角/增强)
       - scoliosis: `scol_<stem>_aug3`     → `scol_<stem>`(同一张真实裸背)
    仅去 _aug/_canon 后缀是错的: 同一人的不同帧/视角会被当成不同受试者 → 泄漏。"""
    m = re.match(r"(syn\d+)", sid)
    if m:
        return m.group(1)
    m = re.match(r"(p\d+)_a\d+", sid)
    if m:
        return m.group(1)
    return re.sub(r"_(aug\d+|canon|neg)$", "", sid)


def split_manifest(root: str, val_frac: float = 0.1, seed: int = 0, min_side_conf: float = 0.5):
    """按【真受试者】划分 train/val(同一人不跨集); 过滤 side_conf 过低的坏 IUV 样本。
    每行附 _root。返回 (train_rows, val_rows)。"""
    rows = [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl"))]
    rows = [{**r, "_root": root} for r in rows
            if r.get("is_valid") and r.get("iuv") and r.get("side_conf", 1.0) >= min_side_conf]
    subs = sorted({subject_key(r["id"]) for r in rows})
    rng = np.random.RandomState(seed)
    val = set(rng.choice(subs, size=max(1, int(len(subs) * val_frac)), replace=False))
    tr = [r for r in rows if subject_key(r["id"]) not in val]
    va = [r for r in rows if subject_key(r["id"]) in val]
    return tr, va


def split_manifests(roots, val_frac: float = 0.1, seed: int = 0, min_side_conf: float = 0.5):
    """多数据集混合: 各自按【真受试者】划分后合并(每行带 _root, 加载时按各自 root 读)。"""
    tr_all, va_all = [], []
    for rt in roots:
        tr, va = split_manifest(rt, val_frac=val_frac, seed=seed, min_side_conf=min_side_conf)
        tr_all += tr; va_all += va
    return tr_all, va_all


class AuscIUVDataset(Dataset):
    def __init__(self, root: str, rows: list, res: int = 518):
        self.root = root
        self.rows = rows
        self.res = res

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        res = self.res
        root = r.get("_root", self.root)                                   # 多数据集混训: 按各行 root 读
        img = cv2.imread(os.path.join(root, r["image"]))[:, :, ::-1]       # BGR→RGB
        iuv = cv2.imread(os.path.join(root, r["iuv"]))                     # BGR: 0=I,1=U255,2=V255
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        iuv = cv2.resize(iuv, (res, res), interpolation=cv2.INTER_NEAREST)
        # 图像归一化
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        # 标签
        part = torch.from_numpy(iuv[:, :, 0].astype(np.int64))
        uv = torch.from_numpy(np.stack([iuv[:, :, 1], iuv[:, :, 2]], 0).astype(np.float32) / 255.0)
        # 听诊点(缩放到 res)
        sx, sy = res / float(w0), res / float(h0)
        pts = {nm: (float(p["xy"][0]) * sx, float(p["xy"][1]) * sy, bool(p["visible"]))
               for nm, p in (r.get("points") or {}).items()}
        return {"image": x, "part": part, "uv": uv, "points": pts,
                "region": r.get("region", "none"), "id": r["id"]}


def collate(batch):
    """图/part/uv stack; points/region/id 保留为 list(变长, 评测用)。"""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "part": torch.stack([b["part"] for b in batch]),
        "uv": torch.stack([b["uv"] for b in batch]),
        "points": [b["points"] for b in batch],
        "region": [b["region"] for b in batch],
        "id": [b["id"] for b in batch],
    }
