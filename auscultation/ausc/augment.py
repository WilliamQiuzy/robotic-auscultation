#!/usr/bin/env python3
"""
增强 pipeline: 让训练取景分布 ≈ 机器人部署的两视角。

样本 dict: {image:HxWx3, iuv:HxWx3(I,U,V) 可选, points:{name:(x,y)} 可选, depth 可选}
所有增强同步变换 image / iuv / points; 出框点标 visible=False(不计损失)。

核心(对应五点处理):
  1+2+3 random_resized_crop : 裁剪+缩放+平移, 模拟近景/上半身/各种远近
  4      hflip             : 水平翻转(需换 L/R 解剖, 见说明)
  6      photometric       : 亮度/对比/模糊/噪声(抗光照, 不动标签)
  out-of-frame mask        : 裁掉的点弃权

两视角预设(见 robot 设计, docs/engineering.md):
  VIEW_PRESETS["thirdperson"] : 中景全/上身, 感知主视角
  VIEW_PRESETS["arm"]         : 近景躯干特写(注:贴太近会丢上下文, 仅局部伺服)
"""
import numpy as np
import cv2

VIEW_PRESETS = {
    "thirdperson": dict(frac=(0.62, 0.88)),   # 躯干+较多身体, 主感知视角
    "arm":         dict(frac=(0.34, 0.48)),   # 胸/背特写近景(仍含全部听诊点)
}


def _xform_points(points, x0, y0, sx, sy, out_w, out_h):
    out = {}
    for nm, p in (points or {}).items():
        px, py = p["xy"] if isinstance(p, dict) else p          # 兼容 dict / 原始 (x,y)
        nx, ny = (px - x0) * sx, (py - y0) * sy
        out[nm] = dict(xy=(float(nx), float(ny)),
                       visible=bool(0 <= nx < out_w and 0 <= ny < out_h))
    return out


def random_resized_crop(sample, scale=(0.3, 1.0), ratio=(0.75, 1.33),
                        target=(384, 384), rng=None, center=None):
    """裁剪(随机尺度/比例/位置)+缩放到 target。center=(cx,cy) 可把裁剪偏向某点(如躯干)。"""
    rng = rng or np.random
    H, W = sample["image"].shape[:2]
    area = H * W
    s = rng.uniform(*scale); r = rng.uniform(*ratio)
    cw = int(round(np.sqrt(area * s * r))); ch = int(round(np.sqrt(area * s / r)))
    cw, ch = min(cw, W), min(ch, H)
    if center is not None:
        cx, cy = center
        x0 = int(np.clip(cx - cw / 2, 0, W - cw)); y0 = int(np.clip(cy - ch / 2, 0, H - ch))
    else:
        x0 = rng.randint(0, W - cw + 1); y0 = rng.randint(0, H - ch + 1)
    tw, th = target
    img = cv2.resize(sample["image"][y0:y0+ch, x0:x0+cw], (tw, th), interpolation=cv2.INTER_LINEAR)
    out = {"image": img}
    if sample.get("iuv") is not None:
        out["iuv"] = cv2.resize(sample["iuv"][y0:y0+ch, x0:x0+cw], (tw, th), interpolation=cv2.INTER_NEAREST)
    if sample.get("depth") is not None:
        out["depth"] = cv2.resize(sample["depth"][y0:y0+ch, x0:x0+cw], (tw, th), interpolation=cv2.INTER_NEAREST)
    out["points"] = _xform_points(sample.get("points"), x0, y0, tw / cw, th / ch, tw, th)
    return out


def photometric(image, rng=None):
    """亮度/对比/gamma/模糊/高斯噪声(抗临床光照差异); 不影响标签。"""
    rng = rng or np.random
    img = image.astype(np.float32)
    # 温和: 避免把浅肤色曝成全白(原 ×1.3+gamma0.7 会过曝)
    img = img * rng.uniform(0.82, 1.12) + rng.uniform(-12, 12)    # 对比+亮度
    img = np.clip(img, 0, 255)
    g = rng.uniform(0.85, 1.18)                                   # gamma(不再大幅提亮)
    img = 255 * (img / 255) ** g
    if rng.random() < 0.3:
        k = rng.choice([3, 5]); img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:
        img = img + rng.normal(0, rng.uniform(3, 12), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def hflip(sample, lr_swap=None, uv_sym=None):
    """水平翻转。⚠️ 翻转会交换患者左右:
       - points: 用 lr_swap 把成对点改名(如 aortic↔? 无对; R_upper↔L_upper)。
       - iuv: 需用 DensePose 对称变换(uv_symmetry_transforms.mat)交换左右 part 并镜像 U。
       这里只做几何翻转 + 点改名; iuv 对称交换留给生成端用 uv_sym。"""
    H, W = sample["image"].shape[:2]
    out = {"image": sample["image"][:, ::-1].copy()}
    if sample.get("iuv") is not None:
        out["iuv"] = sample["iuv"][:, ::-1].copy()   # ⚠️仅几何; part左右需uv_sym再换
    pts = {}
    for nm, p in (sample.get("points") or {}).items():
        x, y = p["xy"] if isinstance(p, dict) else p
        nm2 = (lr_swap or {}).get(nm, nm)
        pts[nm2] = dict(xy=(W - 1 - x, y), visible=p.get("visible", True) if isinstance(p, dict) else True)
    out["points"] = pts
    return out


# 后背肺点的左右对(用于 hflip 改名)。R{i}↔L{i}, i=1..4。
# ⚠️ 仅后背(肺双侧近似对称)可翻转; 前胸心点(aortic/pulmonary/tricuspid/mitral)
#    因心脏左右不对称, 翻转+改名会得到解剖错误标签 → 前胸禁止水平翻转(见 datasets/common.augment_once)。
LR_SWAP = {f"{a}{i}": f"{b}{i}" for i in range(1, 5) for a, b in (("R", "L"), ("L", "R"))}


def crop_around_points(sample, frac=(0.6, 0.9), jitter=0.08, target=(384, 384), rng=None):
    """以听诊点为中心的【正方形】裁剪(避免拉伸), 边长=frac*身体高度, 保证听诊区在画面内。
    模拟机器人相机锁定躯干取景, 而非全身乱裁。"""
    rng = rng or np.random
    img = sample["image"]; H, W = img.shape[:2]
    pts = np.array([(p["xy"] if isinstance(p, dict) else p) for p in sample["points"].values()])
    cx, cy = pts.mean(0)
    f = rng.uniform(*frac)
    side = int(min(f * H, W, H))                          # 正方形, 不超图像
    side = max(side, int((pts[:, :2].ptp(0).max()) * 1.3) + 4)  # 至少容纳所有点
    side = min(side, W, H)
    cx += rng.uniform(-jitter, jitter) * side; cy += rng.uniform(-jitter, jitter) * side
    x0 = int(np.clip(cx - side / 2, 0, W - side)); y0 = int(np.clip(cy - side / 2, 0, H - side))
    tw, th = target
    out = {"image": cv2.resize(img[y0:y0+side, x0:x0+side], (tw, th), interpolation=cv2.INTER_LINEAR)}
    if sample.get("iuv") is not None:
        out["iuv"] = cv2.resize(sample["iuv"][y0:y0+side, x0:x0+side], (tw, th), interpolation=cv2.INTER_NEAREST)
    out["points"] = _xform_points(sample.get("points"), x0, y0, tw / side, th / side, tw, th)
    return out


def augment_for_view(sample, view="thirdperson", target=(384, 384), rng=None):
    """组合: 锁定听诊区域的正方形裁剪(按视角预设缩放) + 光照增强。"""
    rng = rng or np.random
    s = crop_around_points(sample, frac=VIEW_PRESETS[view]["frac"], target=target, rng=rng)
    s["image"] = photometric(s["image"], rng)
    s["view"] = view
    return s


def upright(img, pelvis_uv, neck_uv, points=None, iuv=None):
    """按 SMPL 脊柱(pelvis→neck)方向把图旋转到躯干竖直(头朝上)。图像/IUV/点同步。"""
    dx, dy = neck_uv[0] - pelvis_uv[0], neck_uv[1] - pelvis_uv[1]
    # 解: cv2 角度 a 使脊柱向量(dx,dy)→(0,-1)朝上, 推导得 a=atan2(dx,-dy)(直接用,不取负)
    ang = np.degrees(np.arctan2(dx, -dy))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
    out = {"image": cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))}
    if iuv is not None:
        out["iuv"] = cv2.warpAffine(iuv, M, (w, h), flags=cv2.INTER_NEAREST)
    if points is not None:
        op = {}
        for nm, p in points.items():
            xy = p["xy"] if isinstance(p, dict) else p
            v = M @ np.array([xy[0], xy[1], 1.0])
            op[nm] = dict(xy=(float(v[0]), float(v[1])),
                          visible=(0 <= v[0] < w and 0 <= v[1] < h))
        out["points"] = op
    return out


def occlude_arm(img, target_uv, rng=None, color=(120, 125, 130)):
    """贴一个合成"机械臂+听诊器"遮挡: 从随机边缘伸向 target_uv 的粗杆 + 末端圆头(听诊器)。
    模拟第三人称相机里手臂挡住躯干。只改图像, 不动标签(让模型学会透过遮挡推断)。"""
    rng = rng or np.random
    img = img.copy(); H, W = img.shape[:2]
    tx, ty = int(target_uv[0]), int(target_uv[1])
    edge = rng.choice(["top", "left", "right"])
    sx = {"top": tx + rng.randint(-W // 6, W // 6), "left": 0, "right": W - 1}[edge]
    sy = {"top": 0, "left": ty + rng.randint(-H // 6, H // 6), "right": ty + rng.randint(-H // 6, H // 6)}[edge]
    thick = rng.randint(int(0.05 * W), int(0.11 * W))
    cv2.line(img, (sx, sy), (tx, ty), color, thick, cv2.LINE_AA)              # 臂杆
    cv2.circle(img, (sx, sy), thick // 2, color, -1, cv2.LINE_AA)
    # 末端听诊器头(深色圆 + 金属环)
    r = rng.randint(int(0.04 * W), int(0.07 * W))
    cv2.circle(img, (tx, ty), r, (60, 60, 65), -1, cv2.LINE_AA)
    cv2.circle(img, (tx, ty), r, (180, 185, 190), max(2, r // 6), cv2.LINE_AA)
    return img


def clinical_lighting(image, rng=None):
    """临床光照(保皮肤纹理): 色温暖/冷 + 不均匀方向光(检查灯) + 渐晕。比纯 photometric 更贴临床。"""
    rng = rng or np.random
    img = image.astype(np.float32)
    img = img * rng.uniform(0.85, 1.12) + rng.uniform(-10, 10)            # 整体亮度/对比(温和)
    t = rng.uniform(-1, 1)                                                # 色温
    img[:, :, 0] *= (1 + 0.07 * t); img[:, :, 2] *= (1 - 0.07 * t)
    H, W = img.shape[:2]
    if rng.random() < 0.6:                                               # 单侧检查灯(方向梯度)
        a = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = (xx / W - 0.5) * np.cos(a) + (yy / H - 0.5) * np.sin(a)
        img *= (1 + rng.uniform(0.10, 0.30) * g[:, :, None])
    if rng.random() < 0.5:                                               # 渐晕
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = ((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) ** 0.5
        img *= (1 - rng.uniform(0.15, 0.38) * (rr / rr.max())[:, :, None] ** 2)
    g = rng.uniform(0.85, 1.18)                                          # gamma
    img = 255 * np.clip(img, 0, 255) / 255
    img = 255 * (img / 255) ** g
    return np.clip(img, 0, 255).astype(np.uint8)


def occlude_medical(img, target_uv, rng=None):
    """临床遮挡(只改图不动标签): 随机 听诊器臂 / ECG导线 / 电极 / 检查服边。"""
    rng = rng or np.random
    kind = rng.choice(["arm", "wires", "electrodes", "gown"])
    if kind == "arm":
        return occlude_arm(img, target_uv, rng)
    img = img.copy(); H, W = img.shape[:2]
    if kind == "wires":                                                  # ECG 导线: 几根细线
        for _ in range(rng.randint(2, 4)):
            p0 = (rng.randint(0, W), rng.randint(0, H // 3))
            p1 = (int(target_uv[0]) + rng.randint(-W // 8, W // 8), int(target_uv[1]) + rng.randint(-H // 8, H // 8))
            pal = [(40, 40, 40), (220, 220, 60), (60, 60, 220), (220, 60, 60)]
            col = pal[rng.randint(len(pal))]
            cv2.line(img, p0, p1, col, rng.randint(2, 4), cv2.LINE_AA)
    elif kind == "electrodes":                                           # 电极贴片: 浅色圆+中心点
        for _ in range(rng.randint(2, 4)):
            c = (rng.randint(int(0.2 * W), int(0.8 * W)), rng.randint(int(0.2 * H), int(0.8 * H)))
            r = rng.randint(int(0.03 * W), int(0.06 * W))
            cv2.circle(img, c, r, (235, 235, 230), -1, cv2.LINE_AA)
            cv2.circle(img, c, max(2, r // 3), (120, 120, 120), -1, cv2.LINE_AA)
    else:                                                                # 检查服/铺巾边: 一侧浅色块
        side = rng.choice(["top", "bottom", "left", "right"])
        col = tuple(int(c) for c in (np.array([225, 225, 235]) * rng.uniform(0.85, 1.0)))
        f = rng.uniform(0.12, 0.28)
        if side == "top": cv2.rectangle(img, (0, 0), (W, int(H * f)), col, -1)
        elif side == "bottom": cv2.rectangle(img, (0, int(H * (1 - f))), (W, H), col, -1)
        elif side == "left": cv2.rectangle(img, (0, 0), (int(W * f), H), col, -1)
        else: cv2.rectangle(img, (int(W * (1 - f)), 0), (W, H), col, -1)
    return img


def sensor_degrade(image, rng=None):
    """相机退化(贴真实采集, 保纹理为主): 随机 运动模糊 / 轻高斯模糊 / JPEG / 传感器噪声。"""
    rng = rng or np.random
    img = image.copy()
    k = rng.choice(["motion", "blur", "jpeg", "noise", "none"])
    if k == "motion":
        ks = rng.choice([5, 7, 9]); ker = np.zeros((ks, ks), np.float32)
        if rng.random() < 0.5: ker[ks // 2, :] = 1.0
        else: ker[:, ks // 2] = 1.0
        img = cv2.filter2D(img, -1, ker / ks)
    elif k == "blur":
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif k == "jpeg":
        q = int(rng.randint(35, 75))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok: img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    elif k == "noise":
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(3, 10), img.shape), 0, 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    # 自测: 造一张假样本跑通
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    samp = {"image": img, "points": {"aortic": (300, 200), "mitral": (320, 260)}}
    for v in ("thirdperson", "arm"):
        o = augment_for_view(samp, view=v, center=(310, 230))
        print(v, "->", o["image"].shape, {k: (round(p["xy"][0]), round(p["xy"][1]), p["visible"]) for k, p in o["points"].items()})
