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

from tasks import LANDMARK_INDEX, N_LANDMARKS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _gaussian_heatmaps(pts, hm_res, res, sigma=2.5):
    """把听诊点(res 空间 xy)渲染成 K 通道高斯热图 + 通道 mask。
    pts: {name:(x,y,visible)}; 仅在场+可见的点激活其通道。返回 (hm[K,hm,hm], mask[K])。"""
    hm = np.zeros((N_LANDMARKS, hm_res, hm_res), np.float32)
    mask = np.zeros((N_LANDMARKS,), np.float32)
    s = hm_res / float(res)
    rad = int(3 * sigma)
    for nm, (x, y, vis) in pts.items():
        c = LANDMARK_INDEX.get(nm)
        if c is None or not vis:
            continue
        cx, cy = x * s, y * s
        if not (0 <= cx < hm_res and 0 <= cy < hm_res):
            continue
        x0, x1 = max(0, int(cx) - rad), min(hm_res, int(cx) + rad + 1)
        y0, y1 = max(0, int(cy) - rad), min(hm_res, int(cy) + rad + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        gx = np.arange(x0, x1, dtype=np.float32)
        gy = np.arange(y0, y1, dtype=np.float32)[:, None]
        hm[c, y0:y1, x0:x1] = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
        mask[c] = 1.0
    return hm, mask


def _fg_bbox(iuv):
    """前景(part>0)包围盒 + 占比。返回 (x0,y0,x1,y1,frac) 或 None(前景过小)。"""
    fg = iuv[:, :, 0] > 0
    ys, xs = np.where(fg)
    if len(xs) < 50:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), float(fg.mean())


def _move_pts(pts, ox, oy, scale, res):
    """点同步搬运: 先平移(-ox,-oy)再缩放; 出框标 invisible。pts={name:(x,y,vis)}。"""
    out = {}
    for nm, (x, y, vis) in pts.items():
        nx, ny = (x - ox) * scale, (y - oy) * scale
        out[nm] = (nx, ny, bool(vis and 0 <= nx < res and 0 <= ny < res))
    return out


def _upper_crop(img, iuv, pts, res, rng):
    """保留人物上半(上背)的方形窗 → resize 回 res。模拟只拍到上背的半身照(29/68/74/78 型)。"""
    bb = _fg_bbox(iuv)
    if bb is None:
        return img, iuv, pts
    x0, y0, x1, y1, _ = bb
    bw, bh = x1 - x0, y1 - y0
    if bh < 0.30 * res:                                  # 人物本就小, 不再裁
        return img, iuv, pts
    frac = rng.uniform(0.45, 0.72)                       # 保留上 45-72%
    side = min(max(frac * bh, bw * 1.05), res)
    s = int(round(side))
    cx = (x0 + x1) / 2.0
    wx0 = int(np.clip(cx - side / 2, 0, res - s))
    wy0 = int(np.clip(y0 - 0.04 * bh, 0, res - s))       # 顶部略留头/肩
    img2 = cv2.resize(img[wy0:wy0 + s, wx0:wx0 + s], (res, res), interpolation=cv2.INTER_AREA)
    iuv2 = cv2.resize(iuv[wy0:wy0 + s, wx0:wx0 + s], (res, res), interpolation=cv2.INTER_NEAREST)
    return img2, iuv2, _move_pts(pts, wx0, wy0, res / float(s), res)


def _shrink(img, iuv, pts, res, rng):
    """整图缩小贴进画布(图填边框中位色, IUV 填背景0)。模拟受试者小/远 → IUV 场需在小后背上展开(85 型)。"""
    s = rng.uniform(0.40, 0.68)
    ns = int(round(res * s))
    small_img = cv2.resize(img, (ns, ns), interpolation=cv2.INTER_AREA)
    small_iuv = cv2.resize(iuv, (ns, ns), interpolation=cv2.INTER_NEAREST)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]], 0)
    bg = np.median(border, axis=0).astype(np.uint8)
    canvas_img = np.empty((res, res, 3), np.uint8); canvas_img[:] = bg
    canvas_iuv = np.zeros((res, res, 3), np.uint8)
    ox = int(rng.randint(0, res - ns + 1)); oy = int(rng.randint(0, res - ns + 1))
    canvas_img[oy:oy + ns, ox:ox + ns] = small_img
    canvas_iuv[oy:oy + ns, ox:ox + ns] = small_iuv
    # 缩放后再平移: 等价 _move_pts 但顺序相反(先缩放后平移)
    out = {}
    for nm, (x, y, vis) in pts.items():
        nx, ny = x * s + ox, y * s + oy
        out[nm] = (nx, ny, bool(vis and 0 <= nx < res and 0 <= ny < res))
    return canvas_img, canvas_iuv, out


def geom_augment(img, iuv, pts, res, rng, p_crop=0.18, p_shrink=0.18):
    """部署取景增强(在线, 仅训练): 互斥地以 p_crop 上背裁切 / p_shrink 缩小贴画布 / 否则原样。
    img,iuv: res×res×3 uint8(iuv 通道0=part); pts:{name:(x,y,vis)} res 空间。"""
    u = rng.rand()
    if u < p_crop:
        return _upper_crop(img, iuv, pts, res, rng)
    if u < p_crop + p_shrink:
        return _shrink(img, iuv, pts, res, rng)
    return img, iuv, pts


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
    # Scoliosis and other single-image real datasets can accumulate generated
    # suffixes, e.g. `_canon_clin0` or `_aug3_clin2`.  Strip all terminal
    # derivation suffixes so generated variants cannot cross train/val splits.
    prev = None
    cur = sid
    while cur != prev:
        prev = cur
        cur = re.sub(r"_(aug\d+|canon|neg|clin\d+)$", "", cur)
    return cur


def valid_manifest_rows(root: str, min_side_conf: float = 0.5):
    """Load valid rows and attach _root for mixed-dataset training/eval."""
    with open(os.path.join(root, "manifest.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    return [{**r, "_root": root} for r in rows
            if r.get("is_valid") and r.get("iuv") and r.get("side_conf", 1.0) >= min_side_conf]


def split_manifest(root: str, val_frac: float = 0.1, seed: int = 0, min_side_conf: float = 0.5):
    """按【真受试者】划分 train/val(同一人不跨集); 过滤 side_conf 过低的坏 IUV 样本。
    每行附 _root。返回 (train_rows, val_rows)。"""
    rows = valid_manifest_rows(root, min_side_conf=min_side_conf)
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


def train_rows_with_fixed_val(roots, val_rows, min_side_conf: float = 0.5):
    """Load train rows from roots while excluding every subject in fixed val rows."""
    val_subjects = {subject_key(r["id"]) for r in val_rows}
    train_rows = []
    for root in roots:
        for row in valid_manifest_rows(root, min_side_conf=min_side_conf):
            if subject_key(row["id"]) not in val_subjects:
                train_rows.append(row)
    return train_rows


class AuscIUVDataset(Dataset):
    def __init__(self, root: str, rows: list, res: int = 518,
                 with_landmarks: bool = False, hm_res: int = 0, hm_sigma: float = 2.5,
                 geom_aug: bool = False, geom_crop_p: float = 0.18, geom_shrink_p: float = 0.18):
        self.root = root
        self.rows = rows
        self.res = res
        self.with_landmarks = with_landmarks
        self.hm_res = hm_res or (res // 4)
        self.hm_sigma = hm_sigma
        self.geom_aug = geom_aug                          # 仅训练开; 部署取景(上背裁切/小远缩放)在线增强
        self.geom_crop_p = geom_crop_p
        self.geom_shrink_p = geom_shrink_p

    def __len__(self):
        return len(self.rows)

    def _sample_seed(self, i: int) -> int:
        """每(worker,index,epoch)独立随机种子: 防多worker下 np.random 重复同一增强。
        torch worker.seed 每epoch变 → 同一样本各epoch增强不同。"""
        wi = torch.utils.data.get_worker_info()
        base = int(wi.seed) if wi is not None else 0
        return (base ^ ((i + 1) * 2654435761)) & 0xFFFFFFFF

    def __getitem__(self, i):
        r = self.rows[i]
        res = self.res
        root = r.get("_root", self.root)                                   # 多数据集混训: 按各行 root 读
        img = cv2.imread(os.path.join(root, r["image"]))[:, :, ::-1]       # BGR→RGB
        iuv = cv2.imread(os.path.join(root, r["iuv"]))                     # BGR: 0=I,1=U255,2=V255
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        iuv = cv2.resize(iuv, (res, res), interpolation=cv2.INTER_NEAREST)
        # 听诊点(缩放到 res)
        sx, sy = res / float(w0), res / float(h0)
        pts = {nm: (float(p["xy"][0]) * sx, float(p["xy"][1]) * sy, bool(p["visible"]))
               for nm, p in (r.get("points") or {}).items()}
        # 部署取景在线增强(仅训练): 上背半身裁切 / 受试者小远缩放, img/iuv/pts 几何同步
        if self.geom_aug:
            rng = np.random.RandomState(self._sample_seed(i))
            img, iuv, pts = geom_augment(img, iuv, pts, res, rng,
                                         self.geom_crop_p, self.geom_shrink_p)
        # 图像归一化
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        # 标签
        part = torch.from_numpy(iuv[:, :, 0].astype(np.int64))
        uv = torch.from_numpy(np.stack([iuv[:, :, 1], iuv[:, :, 2]], 0).astype(np.float32) / 255.0)
        out = {"image": x, "part": part, "uv": uv, "points": pts,
               "region": r.get("region", "none"), "id": r["id"]}
        if self.with_landmarks:
            hm, mask = _gaussian_heatmaps(pts, self.hm_res, res, self.hm_sigma)
            out["landmark_hm"] = torch.from_numpy(hm)
            out["landmark_mask"] = torch.from_numpy(mask)
        return out


def collate(batch):
    """图/part/uv stack; points/region/id 保留为 list(变长, 评测用)。"""
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "part": torch.stack([b["part"] for b in batch]),
        "uv": torch.stack([b["uv"] for b in batch]),
        "points": [b["points"] for b in batch],
        "region": [b["region"] for b in batch],
        "id": [b["id"] for b in batch],
    }
    if "landmark_hm" in batch[0]:
        out["landmark_hm"] = torch.stack([b["landmark_hm"] for b in batch])
        out["landmark_mask"] = torch.stack([b["landmark_mask"] for b in batch])
    return out
