#!/usr/bin/env python3
"""增强算子库: 让训练取景/外观分布 ≈ 机器人第三人称固定相机的部署分布。

样本 dict: {image:HxWx3(RGB), iuv:HxWx3(I,U,V) 可选, points:{name:{xy,visible}} 可选, region}。
所有几何算子同步变换 image / iuv / points; 出框点标 visible=False(不计损失)。
IUV 一律 INTER_NEAREST 缩放(保 part 标签离散, 不插值串类)。

随机组合的编排见 datasets/common.augment_once, 当前在用的算子:
  几何   perspective_warp  : 小透视/视角抖动(相机非正对)
         (common._rotate)  : 对称小角度旋转 ±8°(在 common 内, 与裁剪同模块)
         hflip + LR_SWAP   : 水平翻转+换 L/R —— 仅后背(肺双侧对称), 前胸心点不对称禁翻
         (common.crop_to_framing): 取景 F1远/F2中/F3近
  外观   clinical_lighting : 色温 + 单侧检查灯方向光 + 渐晕 + gamma
         hue_shift         : 轻色相/白平衡偏移
         specular          : 皮肤高光/油光
         cast_shadow       : 软投影阴影
  遮挡   occlude_medical → occlude_robot_arm : 桌面侧装机械臂从右下方够向听诊点
  退化   sensor_degrade    : 运动模糊 / 高斯模糊 / JPEG / 传感器噪声
"""
import numpy as np
import cv2


def _xform_points(points, x0, y0, sx, sy, out_w, out_h):
    """裁剪+缩放后同步搬运点; 出框标 visible=False。"""
    out = {}
    for nm, p in (points or {}).items():
        px, py = p["xy"] if isinstance(p, dict) else p          # 兼容 dict / 原始 (x,y)
        nx, ny = (px - x0) * sx, (py - y0) * sy
        out[nm] = dict(xy=(float(nx), float(ny)),
                       visible=bool(0 <= nx < out_w and 0 <= ny < out_h))
    return out


# DensePose 24-part 水平镜像的 part 互换(1-indexed; 0=背景)。躯干 1,2 自身; 肢体左右互换。
_INDEX_SYM = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]
_UV_SYM_CACHE = None


def _get_uv_symmetry():
    """懒加载 DensePose UV_symmetry_transforms.mat → (U_transforms[24], V_transforms[24], Index_Symmetry)。
    每个 transform 是 (256,256) 查表, 按 [V*255, U*255] 索引, 给出镜像后的 U/V ∈ [0,1]。"""
    global _UV_SYM_CACHE
    if _UV_SYM_CACHE is not None:
        return _UV_SYM_CACHE
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "..", "data/densepose_coco/uv_data/UV_symmetry_transforms.mat"),
             os.path.join(here, "data/densepose_coco/uv_data/UV_symmetry_transforms.mat")]
    path = next((c for c in cands if os.path.exists(c)), None)
    if path is None:
        _UV_SYM_CACHE = (None, None, None)
        return _UV_SYM_CACHE
    import scipy.io as sio
    d = sio.loadmat(path)
    Ut = [d["U_transforms"][0, i] for i in range(24)]
    Vt = [d["V_transforms"][0, i] for i in range(24)]
    _UV_SYM_CACHE = (Ut, Vt, _INDEX_SYM)
    return _UV_SYM_CACHE


def apply_uv_symmetry(iuv):
    """对(已几何镜像的) IUV 做 DensePose 对称: part 左右互换 + (U,V) 按对称表重映射。
    iuv: HxWx3 uint8 (I, U*255, V*255)。这是水平翻转的正确做法——只翻几何会让 V(左右表面坐标)错位。"""
    Ut, Vt, idx_sym = _get_uv_symmetry()
    if Ut is None:
        return iuv.copy()
    I = iuv[:, :, 0].astype(np.int32)
    U = np.clip(iuv[:, :, 1].astype(np.int32), 0, 255)
    V = np.clip(iuv[:, :, 2].astype(np.int32), 0, 255)
    newI = I.astype(np.uint8).copy()
    newU = iuv[:, :, 1].copy()
    newV = iuv[:, :, 2].copy()
    for part in range(1, 25):
        m = I == part
        if not m.any():
            continue
        u_idx = U[m]
        v_idx = V[m]
        nu = Ut[part - 1][v_idx, u_idx]            # ∈[0,1]
        nv = Vt[part - 1][v_idx, u_idx]
        newU[m] = np.clip(nu * 255.0, 0, 255).astype(np.uint8)
        newV[m] = np.clip(nv * 255.0, 0, 255).astype(np.uint8)
        newI[m] = idx_sym[part]
    return np.dstack([newI, newU, newV])


def hflip(sample, lr_swap=None, uv_sym=True):
    """水平翻转(镜像患者左右)。
       - points: 用 lr_swap 把成对点改名(后背 R{i}↔L{i}, i=1..5)。
       - iuv: 几何镜像 + DensePose UV 对称(part 互换 + U,V 对称表重映射)。
         uv_sym=True(默认)才正确; 否则 V(左右表面坐标)会与翻转后的图像错位(历史 bug)。"""
    H, W = sample["image"].shape[:2]
    out = {"image": sample["image"][:, ::-1].copy()}
    if sample.get("iuv") is not None:
        flipped = sample["iuv"][:, ::-1]
        out["iuv"] = apply_uv_symmetry(flipped) if uv_sym else flipped.copy()
    pts = {}
    for nm, p in (sample.get("points") or {}).items():
        x, y = p["xy"] if isinstance(p, dict) else p
        nm2 = (lr_swap or {}).get(nm, nm)
        pts[nm2] = dict(xy=(W - 1 - x, y), visible=p.get("visible", True) if isinstance(p, dict) else True)
    out["points"] = pts
    return out


# 后背肺点的左右对(用于 hflip 改名)。R{i}↔L{i}, i=1..5(V6A 5 层)。
# ⚠️ 仅后背(肺双侧近似对称)可翻转; 前胸心点(aortic/pulmonary/tricuspid/mitral)
#    因心脏左右不对称, 翻转+改名会得到解剖错误标签 → 前胸禁止水平翻转(见 datasets/common.augment_once)。
LR_SWAP = {f"{a}{i}": f"{b}{i}" for i in range(1, 6) for a, b in (("R", "L"), ("L", "R"))}


def clinical_lighting(image, rng=None):
    """临床光照(保皮肤纹理): 色温暖/冷 + 不均匀方向光(检查灯) + 渐晕 + gamma。"""
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


def hue_shift(image, rng=None):
    """轻色相偏移(不同相机/白平衡)。"""
    rng = rng or np.random
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-7, 7)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.85, 1.15), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def specular(img, rng=None):
    """皮肤高光(油光/出汗在检查灯下反光): 软亮斑。只改图。"""
    rng = rng or np.random
    f = img.astype(np.float32); H, W = img.shape[:2]
    m = np.zeros((H, W), np.float32)
    for _ in range(rng.randint(1, 3)):
        cv2.circle(m, (rng.randint(0, W), rng.randint(int(0.2 * H), int(0.75 * H))),
                   rng.randint(int(0.05 * W), int(0.14 * W)), 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), rng.uniform(14, 30))
    f += rng.uniform(0.12, 0.32) * m[:, :, None] * 130
    return np.clip(f, 0, 255).astype(np.uint8)


def cast_shadow(img, rng=None):
    """软投影阴影(检查灯下手臂/设备投影到身上)。只改图。"""
    rng = rng or np.random
    f = img.astype(np.float32); H, W = img.shape[:2]
    m = np.zeros((H, W), np.float32)
    poly = np.array([[rng.randint(0, W), rng.randint(0, H)] for _ in range(rng.randint(3, 6))], np.int32)
    cv2.fillConvexPoly(m, cv2.convexHull(poly), 1.0)
    m = cv2.GaussianBlur(m, (0, 0), rng.uniform(20, 55))
    f *= (1 - rng.uniform(0.22, 0.48) * m[:, :, None])
    return np.clip(f, 0, 255).astype(np.uint8)


def perspective_warp(sample, rng=None, mag=0.055):
    """小透视/视角抖动(相机非正对患者): image/iuv/points 同步。比纯旋转更真实的视角变化。"""
    rng = rng or np.random
    img = sample["image"]; H, W = img.shape[:2]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = src + rng.uniform(-mag, mag, (4, 2)).astype(np.float32) * np.float32([W, H])
    M = cv2.getPerspectiveTransform(src, dst)
    bgc = tuple(int(c) for c in img[0, 0])
    out = {"region": sample.get("region"),
           "image": cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=bgc)}
    if sample.get("iuv") is not None:
        out["iuv"] = cv2.warpPerspective(sample["iuv"], M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    pts = {}
    for nm, p in (sample.get("points") or {}).items():
        xy = p["xy"] if isinstance(p, dict) else p
        v = M @ np.array([xy[0], xy[1], 1.0]); v = v / v[2]
        pts[nm] = dict(xy=(float(v[0]), float(v[1])), visible=bool(0 <= v[0] < W and 0 <= v[1] < H))
    out["points"] = pts
    return out


def occlude_robot_arm(img, target_uv, rng=None):
    """机器人机械臂(分节金属连杆+关节)伸向听诊点。模拟第三人称看机器人作业(无末端听诊头)。"""
    rng = rng or np.random
    img = img.copy(); H, W = img.shape[:2]
    tx, ty = int(target_uv[0]), int(target_uv[1])
    # 桌面侧装机械臂从【右下方】够向听诊点(实验室 OpenArm 布局), 不从上方下来
    if rng.random() < 0.65:                                              # 右下角进
        sx = int(rng.uniform(0.62, 1.0) * W); sy = H - 1
    else:                                                                # 右侧偏下进
        sx = W - 1; sy = int(rng.uniform(0.55, 0.98) * H)
    n_mid = rng.randint(2, 3)                                            # 中间关节数(2)→稳定多节, 不退化成线
    pts = [(sx, sy)]
    for i in range(1, n_mid + 1):
        t = i / (n_mid + 1)
        pts.append((int(sx + (tx - sx) * t + rng.randint(-W // 16, W // 16)),   # 偏移更小→更规整, 不乱折
                    int(sy + (ty - sy) * t + rng.randint(-H // 16, H // 16))))
    pts.append((tx, ty))
    th = int(rng.uniform(0.06, 0.085) * W)                               # 连杆够粗且稳定
    cv2.circle(img, (sx, sy), int(th * 0.9), (55, 57, 64), -1, cv2.LINE_AA)   # 入口基座
    for a, b in zip(pts[:-1], pts[1:]):                                  # 金属连杆(暗边+金属+高光)
        cv2.line(img, a, b, (62, 64, 72), th + 6, cv2.LINE_AA)
        cv2.line(img, a, b, (152, 155, 162), th, cv2.LINE_AA)
        cv2.line(img, a, b, (205, 207, 213), max(2, th // 4), cv2.LINE_AA)
    for p in pts[1:-1]:                                                  # 关节(暗座+亮轴)
        cv2.circle(img, p, th // 2 + 4, (52, 54, 62), -1, cv2.LINE_AA)
        cv2.circle(img, p, max(3, th // 3), (130, 133, 140), -1, cv2.LINE_AA)
    cv2.circle(img, (tx, ty), th // 2 + 4, (52, 54, 62), -1, cv2.LINE_AA)  # 末端夹爪节点(非听诊头)
    cv2.circle(img, (tx, ty), max(3, th // 3), (140, 143, 150), -1, cv2.LINE_AA)
    return img


def occlude_medical(img, points, rng=None, region="front", iuv=None):
    """部署期遮挡(只改图不动标签): 机器人机械臂横跨听诊区。
    (电极贴片已移除——医学位置在合成图上不可靠, 宁缺毋滥。)"""
    rng = rng or np.random
    vis = [(nm, (p["xy"] if isinstance(p, dict) else p)) for nm, p in points.items()
           if (p.get("visible", True) if isinstance(p, dict) else True)]
    if not vis:
        return img
    return occlude_robot_arm(img, vis[rng.randint(len(vis))][1], rng)


def sensor_degrade(image, rng=None, severity="mild"):
    """相机退化(贴真实采集, 保纹理为主): 随机运动模糊 / 轻高斯模糊 / JPEG / 传感器噪声。

    severity="mild" 用于真实裸背/HuMMan增强, 避免把本来就低分辨率的皮肤纹理打碎;
    severity="strong" 仅保留给未来明确需要重退化鲁棒性的实验。
    """
    rng = rng or np.random
    img = image.copy()
    k = rng.choice(["motion", "blur", "jpeg", "noise"])      # 去 none, 调用即出效果且明显
    strong = severity == "strong"
    if k == "motion":
        ks = int(rng.choice([7, 9, 11] if not strong else [9, 13, 17])); ker = np.zeros((ks, ks), np.float32)
        if rng.random() < 0.5: ker[ks // 2, :] = 1.0
        else: ker[:, ks // 2] = 1.0
        img = cv2.filter2D(img, -1, ker / ks)
    elif k == "blur":
        ks = int(rng.choice([3, 5] if not strong else [5, 7, 9])); img = cv2.GaussianBlur(img, (ks, ks), 0)
    elif k == "jpeg":
        q = int(rng.randint(45, 76) if not strong else rng.randint(18, 45))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok: img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    elif k == "noise":
        sigma = rng.uniform(4, 10) if not strong else rng.uniform(10, 25)
        img = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return img
