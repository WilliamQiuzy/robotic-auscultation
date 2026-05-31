#!/usr/bin/env python3
"""合成 SMPL → 可训练样本(前胸4/后背8 + 稠密IUV)。**纯仿真 pyrender 后端**。

渲染 = pyrender 平滑Phong + 连续程序化皮肤(单一多人种肤色+极轻变化, 无纹理拼接) + 三点柔光。
取景 = 相机距离三档 F1远/F2中/F3近(以听诊区居中); 姿态 = 站/坐; 体型/肤色随机。
每受试者 × {站,坐} × {前,后} × {F1,F2,F3} = 12 base, 每 base 再叠加 2D 增强(光照/遮挡/对称旋转/翻转[仅后背])。

⚠️ **需 GPU + pyrender(EGL)**: 依赖 pyrender/trimesh, 环境变量 PYOPENGL_PLATFORM=egl。
   本机(无EGL)跑不了; 在 GPU 服务器上经 scripts/gen_dataset.py --dataset synthetic 调用。
   取景相机/姿态/肤色定义见下; 与 docs/datasets_and_processing.md 的取景分类一致。
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from augment import LR_SWAP, hflip, occlude_arm, photometric  # noqa: E402
from common import ManifestWriter, write_sample  # noqa: E402
from humman_label import define_ausc_vertices  # noqa: E402
from render_iuv import load_uv_mapping, project, rasterize_iuv  # noqa: E402

# 多人种肤色调色板 (RGB, 浅→深)
SKIN = np.array([[235, 201, 178], [222, 183, 156], [206, 158, 124],
                 [178, 126, 96], [141, 96, 70], [104, 70, 52], [78, 52, 40]], np.float32)

# 取景三档(全部近距, 部署相机离得近/人拍得大): span倍数小=更近。
# torso画面占比≈fill/k → F1上半身填满~85% / F2躯干填满~105% / F3胸背特写~140%。
FRAMINGS = {
    "F1_upper": (1.00, 0.85, 0.05),   # 上半身填满 (~85%)
    "F2_torso": (0.80, 0.86, 0.05),   # 躯干填满 (~105%)
    "F3_near":  (0.60, 0.90, 0.10),   # 胸/背特写 (~150%, 听诊区居中)
}
ROT_CHOICES = (0.0, -8.0, 8.0, -15.0, 15.0)


def _rx(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def _rz(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


# 相机朝向(实测): 正面=绕x转180°(正面朝相机+头朝上); 背面=绕z转180°
VIEW_R = {"front": _rx(180.0), "back": _rz(180.0)}


def _natural_pose(rng):
    """站姿: 肩关节下旋到身侧(非T-pose)。"""
    bp = np.zeros((23, 3), np.float32)
    bp[15] = [0, 0, -1.35]; bp[16] = [0, 0, 1.35]
    bp[17] = [0, 0, -0.2]; bp[18] = [0, 0, 0.2]
    bp += rng.normal(0, 0.04, bp.shape).astype(np.float32)
    return bp.reshape(1, 69)


def _seated_pose(rng):
    bp = _natural_pose(rng).reshape(23, 3)
    bp[0] = [-1.35, 0, 0.05]; bp[1] = [-1.35, 0, -0.05]   # 屈髋
    bp[3] = [1.5, 0, 0]; bp[4] = [1.5, 0, 0]              # 屈膝
    return bp.reshape(1, 69).astype(np.float32)


def _skin_colors(verts, skin, rng):
    V = verts.shape[0]
    col = np.tile(skin, (V, 1))
    yy = (verts[:, 1] - verts[:, 1].min()) / (verts[:, 1].ptp() + 1e-6)
    col[:, 0] *= (1 + 0.04 * yy)                          # 上身略红
    col *= (1 + rng.normal(0, 0.015, (V, 1)))             # 极轻噪声(连续不色块)
    return np.clip(col, 0, 255).astype(np.uint8)


def _make_cam(cx, cy, span, H, W, focal, fill, R):
    d = focal * span / (fill * H)
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], np.float32)
    return {"K": K, "R": R, "T": np.array([-cx, -cy, d], np.float32)}


def _render_rgb(verts, faces, cam, H, W, skin, rng):
    """pyrender 平滑Phong + 三点柔光 → RGB(纯色柔背景合成)。"""
    import pyrender
    import trimesh
    mesh = trimesh.Trimesh(verts, faces, process=False)
    mesh.visual.vertex_colors = _skin_colors(verts, skin, rng)
    pm = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4])
    scene.add(pm)
    pose4 = np.linalg.inv(np.vstack([np.hstack([cam["R"], cam["T"][:, None]]), [0, 0, 0, 1]])) @ np.diag([1.0, -1, -1, 1])
    K = cam["K"]
    scene.add(pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.05, zfar=20), pose=pose4)
    # 柔光(收敛强度: 原 2.8/1.5/1.0 会把浅肤色曝白)
    for d, it in [((-0.5, -0.6, -1.0), 0.9), ((0.7, -0.2, -0.8), 0.5), ((0, 0.4, 1.0), 0.3)]:
        lp = np.eye(4); z = -np.array(d, float); z /= np.linalg.norm(z)
        x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x); lp[:3, :3] = np.c_[x, np.cross(z, x), z]
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=it), pose=pose4 @ lp)
    r = pyrender.OffscreenRenderer(W, H)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA); r.delete()
    rgb, a = color[:, :, :3], color[:, :, 3]
    bg = np.full((H, W, 3), int(rng.uniform(150, 205)), np.uint8)
    return np.where((a > 10)[:, :, None], rgb, bg).astype(np.uint8)


def _render_base(model, faces, uvmap, ausc, torso_idx, rng, view, posture, framing, H, W, focal):
    """渲染一个 base sample: RGB + 稠密IUV + 听诊点(相机取景 F1/F2/F3)。"""
    import torch
    betas = np.clip(rng.normal(0, 1.4, (1, 10)), -3, 3).astype(np.float32)
    pose = _seated_pose(rng) if posture == "seated" else _natural_pose(rng)
    with torch.no_grad():
        o = model(betas=torch.tensor(betas), body_pose=torch.tensor(pose),
                  global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3))
    verts = o.vertices[0].numpy()
    R = VIEW_R[view]; P = (R @ verts.T).T
    ac = P[list(ausc[view].values())].mean(0); torso_h = float(P[torso_idx, 1].ptp())
    k, fill, cyoff = FRAMINGS[framing]
    cam = _make_cam(ac[0], ac[1] + cyoff * torso_h, torso_h * k, H, W, focal, fill, R)
    skin = SKIN[rng.randint(len(SKIN))] * rng.uniform(0.9, 1.02)
    img = _render_rgb(verts, faces, cam, H, W, skin, rng)
    I, U, V, _ = rasterize_iuv(verts, cam, uvmap, H, W)
    iuv = np.dstack([I, (U * 255).astype(np.uint8), (V * 255).astype(np.uint8)])
    uv, _ = project(verts, cam)
    pts = {nm: {"xy": (float(uv[vi, 0]), float(uv[vi, 1])),
                "visible": bool(0 <= uv[vi, 0] < W and 0 <= uv[vi, 1] < H)} for nm, vi in ausc[view].items()}
    return {"image": img, "iuv": iuv, "points": pts, "region": view}


def _augment(base, rng):
    """synthetic 2D 增强: 对称旋转 + 翻转(仅后背) + 光照 + 遮挡。取景已由相机做, 不再2D裁剪。"""
    region = base["region"]; H, W = base["image"].shape[:2]
    s = {"image": base["image"].copy(), "iuv": base["iuv"].copy(),
         "points": {k: dict(v) for k, v in base["points"].items()}, "region": region}
    ang = float(rng.choice(ROT_CHOICES))
    if ang != 0.0:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        bgc = tuple(int(c) for c in s["image"][0, 0])   # 旋转露出的角填背景灰(非黑)
        s["image"] = cv2.warpAffine(s["image"], M, (W, H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=bgc)
        s["iuv"] = cv2.warpAffine(s["iuv"], M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        for nm, p in s["points"].items():
            v = M @ np.array([p["xy"][0], p["xy"][1], 1.0])
            s["points"][nm] = {"xy": (float(v[0]), float(v[1])), "visible": bool(0 <= v[0] < W and 0 <= v[1] < H)}
    if region == "back" and rng.random() < 0.5:
        f = hflip(s, lr_swap=LR_SWAP); f["region"] = region; s = f
    s["image"] = photometric(s["image"], rng)
    if rng.random() < 0.6:
        vis = [p["xy"] for p in s["points"].values() if p.get("visible")]
        if vis:
            s["image"] = occlude_arm(s["image"], vis[rng.randint(len(vis))], rng)
    return s


def generate(smpl_model, uv, out, n_subjects=400, n_aug=3, res=512, focal=1100.0, seed=0):
    """纯仿真 pyrender 量产。需 GPU+EGL。每受试者 × 站/坐 × 前/后 × F1/F2/F3 × (1+n_aug)。"""
    import smplx
    model = smplx.SMPL(model_path=smpl_model, gender="neutral", batch_size=1)
    faces = model.faces
    uvmap = load_uv_mapping(uv)
    ausc = define_ausc_vertices(model)
    from humman_label import infer_side  # noqa: F401 (确保模块就绪)
    torso_idx = _torso_indices(uvmap)
    with ManifestWriter(out) as mw:
        for si in range(n_subjects):
            for posture in ("standing", "seated"):
                for view in ("front", "back"):
                    for fr in FRAMINGS:
                        r = np.random.RandomState(seed * 100003 + si * 97 + (hash((posture, view, fr)) % 1000))
                        base = _render_base(model, faces, uvmap, ausc, torso_idx, r, view, posture, fr, res, res, focal)
                        bid = f"syn{si:05d}_{posture}_{view}_{fr}"
                        mw.add(write_sample(out, bid + "_canon", base))
                        for k in range(n_aug):
                            mw.add(write_sample(out, f"{bid}_aug{k}", _augment(base, r)))
        print("[synthetic]", mw.summary())


def _torso_indices(uvmap):
    """DensePose 躯干块(part1后背/part2前胸)对应的 SMPL 顶点索引。"""
    v2s, U, V, faces, fpart = uvmap
    torso_dp = np.unique(faces[(fpart == 1) | (fpart == 2)].reshape(-1))
    return np.array(sorted(set(v2s[torso_dp].tolist())))
