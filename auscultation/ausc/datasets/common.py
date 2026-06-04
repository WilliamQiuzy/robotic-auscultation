#!/usr/bin/env python3
"""数据集生成共用件: 标签 schema / manifest 写入 / 增强的随机组合编排。

所有 generator(humman/scoliosis/synthetic)都把单帧整理成统一 **base sample**:
    {"image": HxWx3 uint8(RGB),
     "iuv":   HxWx3 uint8(R=part0..24,G=U*255,B=V*255) 或 None(无稠密标签, 仅点监督),
     "points":{name:{"xy":(x,y),"visible":bool}},
     "region":"front"|"back"}
再由 `augment_once` 施加【随机组合】的增强, 由 `write_sample` 落盘 + 出 manifest 行。

增强编排(匹配第三人称固定相机 + 实验室 WidowX/Trossen 桌面侧装臂的部署分布):
  几何  perspective_warp 透视抖动 或 _rotate 对称小角度 ±8°(二选一)
        hflip+LR_SWAP 水平翻转换 L/R —— 仅后背(肺双侧对称), 前胸心点不对称禁翻
        crop_to_framing 取景 F1远/F2中/F3近(正方形真像素裁剪, 不补边)
  外观  clinical_lighting / hue_shift / specular / cast_shadow
  遮挡  occlude_medical → 机械臂从右下方够向听诊点(电极已弃用)
  退化  sensor_degrade 运动模糊/高斯模糊/JPEG/噪声
所有几何算子同步变换 image/iuv/points; IUV 用 NEAREST 缩放保标签离散。
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # 让 ausc/ 顶层模块可导入
from augment import (_xform_points, cast_shadow, clinical_lighting, hflip, hue_shift,  # noqa: E402
                     occlude_medical, perspective_warp, sensor_degrade, specular, LR_SWAP)

ROT_CHOICES = (0.0, -8.0, 8.0)  # 旋转仅 ±8°(±15太过), 对称

# 取景三档(2D 真实图裁剪): side = 听诊点纵向跨度 × 倍数。远→近。
# 与 synthetic 相机三档(F1远/F2中/F3近)一致, 见 datasets_and_processing.md 取景分类。
# side = 躯干尺寸 × 倍数。F1>1宽景(缩小/多上下文) → F3近景; 不再做低分辨率真实图极近放大。
FRAMINGS_2D = {"F1_upper": 1.15, "F2_torso": 0.88, "F3_near": 0.74}


def crop_to_framing(sample, framing, rng, target=(384, 384)):
    """以【躯干中心】为中心的正方形裁剪(身体居中, 不偏心), 边长按取景档×躯干尺寸 → 远/中/近景。
    中心从 IUV 躯干块(前胸part2/后背part1)质心取; 保证全部听诊点在框内。"""
    img = sample["image"]; H, W = img.shape[:2]
    iuv = sample.get("iuv"); region = sample.get("region")
    part = 2 if region == "front" else 1
    mask = None
    if iuv is not None:
        m = iuv[:, :, 0] == part
        if m.sum() < 50:
            m = iuv[:, :, 0] > 0                      # 兜底: 全身体素
        if m.sum() >= 50:
            mask = m
    if mask is not None:                              # 躯干中心 + 躯干尺寸
        ys, xs = np.where(mask); cx, cy = float(xs.mean()), float(ys.mean())
        extent = float(max(xs.ptp(), ys.ptp()))
    else:                                             # 无IUV兜底: 用点
        p0 = np.array([p["xy"] if isinstance(p, dict) else p for p in sample["points"].values()])
        cx, cy = p0.mean(0); extent = max(float(p0.ptp(0).max()), 8.0)
    side = extent * FRAMINGS_2D[framing]
    pts = np.array([(p["xy"] if isinstance(p, dict) else p) for p in sample["points"].values()])
    if len(pts):                                      # 下限: 仅保证点的跨度+小边距在框内(温和, 不压垮缩放分级)
        side = max(side, 1.4 * float(max(pts[:, 0].ptp(), pts[:, 1].ptp())))
    side = int(np.clip(side, 24, min(H, W)))
    cx += rng.uniform(-0.04, 0.04) * side; cy += rng.uniform(-0.04, 0.04) * side
    x0 = int(np.clip(cx - side / 2, 0, W - side)); y0 = int(np.clip(cy - side / 2, 0, H - side))
    tw, th = target
    out = {"image": cv2.resize(img[y0:y0 + side, x0:x0 + side], (tw, th), interpolation=cv2.INTER_LINEAR),
           "region": sample.get("region")}
    if sample.get("iuv") is not None:
        out["iuv"] = cv2.resize(sample["iuv"][y0:y0 + side, x0:x0 + side], (tw, th), interpolation=cv2.INTER_NEAREST)
    out["points"] = _xform_points(sample.get("points"), x0, y0, tw / side, th / side, tw, th)
    return out


def side_conf(part_map: np.ndarray, region: str) -> float:
    """躯干块 part2=前胸 / part1=后背, 取目标区域占比作为前后判别置信。"""
    f = int((part_map == 2).sum())
    b = int((part_map == 1).sum())
    if f + b == 0:
        return 0.0
    return round((f if region == "front" else b) / (f + b), 3)


def _rotate(sample: dict, ang_deg: float) -> dict:
    """绕图心旋转 ang_deg; 图像/IUV/点同步。小角度(±8/±15), 不缩放。"""
    img = sample["image"]
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang_deg, 1.0)
    # 真实图旋转: 边缘复制(避免黑角; 比纯色填充自然)。IUV 露出区填 0(背景)。
    out = {"image": cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
           "region": sample.get("region")}
    if sample.get("iuv") is not None:
        out["iuv"] = cv2.warpAffine(sample["iuv"], M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    pts = {}
    for nm, p in sample["points"].items():
        xy = p["xy"] if isinstance(p, dict) else p
        v = M @ np.array([xy[0], xy[1], 1.0])
        pts[nm] = dict(xy=(float(v[0]), float(v[1])), visible=bool(0 <= v[0] < w and 0 <= v[1] < h))
    out["points"] = pts
    return out


def augment_once(sample: dict, rng, target=(384, 384)) -> dict:
    """对 base sample 施加【一次随机组合】的 5 种核心增强 → target 尺寸样本。"""
    region = sample["region"]
    s = {"image": sample["image"].copy(),
         "iuv": None if sample.get("iuv") is None else sample["iuv"].copy(),
         "points": {k: dict(v) for k, v in sample["points"].items()},
         "region": region}

    # 几何: 透视视角抖动 或 对称旋转(二选一)
    if rng.random() < 0.45:
        f = perspective_warp(s, rng); f["region"] = region; s = f
    else:
        ang = float(rng.choice(ROT_CHOICES))
        if ang != 0.0:
            s = _rotate(s, ang)

    # 水平翻转 + 换 L/R —— 仅后背(肺双侧对称); 前胸心点不对称, 永不翻转
    if region == "back" and rng.random() < 0.5:
        f = hflip(s, lr_swap=LR_SWAP); f["region"] = region; s = f

    # 取景 F1远/F2中/F3近(随机一档, 以听诊区居中正方形裁剪)
    s = crop_to_framing(s, rng.choice(list(FRAMINGS_2D)), rng, target=target)
    s["region"] = region

    # 光照/外观
    s["image"] = clinical_lighting(s["image"], rng)
    if rng.random() < 0.4:
        s["image"] = hue_shift(s["image"], rng)
    if rng.random() < 0.3:
        s["image"] = specular(s["image"], rng)
    if rng.random() < 0.25:
        s["image"] = cast_shadow(s["image"], rng)
    # 遮挡(听诊器臂/ECG导线/电极/检查服 + 头发) + 随机擦除
    if rng.random() < 0.6:                                       # 机械臂遮挡(从右下方够向听诊点; 电极已弃用)
        s["image"] = occlude_medical(s["image"], s["points"], rng,
                                     region=region, iuv=s.get("iuv"))
    # 相机退化(运动模糊/JPEG/噪声): 低分辨率真实图要保纹理, 只温和少量加入。
    if rng.random() < 0.2:
        s["image"] = sensor_degrade(s["image"], rng, severity="mild")
    return s


def _ensure_dirs(out: str) -> None:
    for d in ("img", "iuv"):
        os.makedirs(os.path.join(out, d), exist_ok=True)


def write_sample(out: str, sid_id: str, sample: dict, is_valid: bool = True) -> dict:
    """落盘 image(+iuv) 并返回 manifest 记录(dict)。iuv=None → 仅点监督(iuv 字段 null)。"""
    cv2.imwrite(os.path.join(out, "img", sid_id + ".jpg"), sample["image"][:, :, ::-1])
    iuv_path = None
    conf = 0.0
    if sample.get("iuv") is not None:
        cv2.imwrite(os.path.join(out, "iuv", sid_id + ".png"), sample["iuv"])
        iuv_path = f"iuv/{sid_id}.png"
        conf = side_conf(sample["iuv"][:, :, 0], sample.get("region", "")) if is_valid else 0.0
    pts = {nm: {"xy": [round(float(p["xy"][0]), 1), round(float(p["xy"][1]), 1)],
                "visible": bool(p["visible"])}
           for nm, p in (sample.get("points") or {}).items()} if is_valid else {}
    return {"id": sid_id, "image": f"img/{sid_id}.jpg", "iuv": iuv_path,
            "region": sample.get("region") if is_valid else "none",
            "is_valid": bool(is_valid), "side_conf": conf, "points": pts}


class ManifestWriter:
    """逐行写 manifest.jsonl, 统计计数。with 语境管理。"""

    def __init__(self, out: str):
        _ensure_dirs(out)
        self.out = out
        self._fh = open(os.path.join(out, "manifest.jsonl"), "w")
        self.n = {"front": 0, "back": 0, "neg": 0}

    def add(self, rec: dict) -> None:
        self._fh.write(json.dumps(rec) + "\n")
        if not rec["is_valid"]:
            self.n["neg"] += 1
        else:
            self.n[rec["region"]] = self.n.get(rec["region"], 0) + 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._fh.close()

    def summary(self) -> str:
        tot = sum(self.n.values())
        neg_pct = self.n["neg"] / max(1, tot)
        return (f"front={self.n['front']} back={self.n['back']} neg={self.n['neg']} "
                f"(neg {neg_pct:.0%}, total {tot}) -> {self.out}/manifest.jsonl")
