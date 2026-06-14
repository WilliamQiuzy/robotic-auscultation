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
import zlib

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from augment import (LR_SWAP, cast_shadow, clinical_lighting, hflip, hue_shift,  # noqa: E402
                     occlude_medical, perspective_warp, sensor_degrade, specular)
from common import ManifestWriter, write_sample  # noqa: E402
from humman_label import define_ausc_vertices  # noqa: E402
from render_iuv import load_uv_mapping, project, rasterize_iuv  # noqa: E402

# 真实肤色 albedo: 据 ITA/Xiao2017 在 CIELAB 沿真实肤色轨迹取值再转 sRGB。
# a*≈13-16(红度,各人种近似), b*黄度(东亚偏黄), L*主导(白人浅→非洲深棕)。浅→深。
SKIN = np.array([[212, 170, 154], [201, 156, 138], [193, 148, 121], [179, 132, 106],
                 [164, 116, 90], [146, 101, 79], [128, 88, 68], [110, 74, 57],
                 [91, 62, 49], [78, 54, 43]], np.float32)

# 取景三档(全部近距, 部署相机离得近/人拍得大): span倍数小=更近。
# torso画面占比≈fill/k → F1上半身填满~85% / F2躯干填满~105% / F3胸背特写~140%。
# (span倍数, 画面占比, 中心上移) 躯干占比≈fill/k。保持三档部署取景, 不生成极近局部裁切。
FRAMINGS = {
    "F1_upper": (1.45, 0.86, 0.05),   # 远: 上半身(躯干~60%, 多上下文)
    "F2_torso": (1.00, 0.86, 0.05),   # 中: 躯干填满 (~86%)
    "F3_near":  (0.70, 0.90, 0.08),   # 近: 胸/背特写 (~128%)
}
ROT_CHOICES = (0.0, -8.0, 8.0)   # 旋转仅 ±8°(±15太过)


def variant_seed_offset(posture, view, framing):
    """稳定的 variant seed offset; 避免 Python 内置 hash 的跨进程随机 salt。"""
    return zlib.crc32(f"{posture}|{view}|{framing}".encode("utf-8")) % 1000


def _rx(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def _rz(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


# 相机朝向(实测): 正面=绕x转180°(正面朝相机+头朝上); 背面=绕z转180°
VIEW_R = {"front": _rx(180.0), "back": _rz(180.0)}


def _jitter_view(R0, rng, yaw=18.0, pitch=12.0, roll=7.0):
    """相机视角随机抖动(方位/俯仰/滚转, 域随机化): 保持正/背面可见前提下变视角。"""
    a, b, c = (np.radians(rng.uniform(-yaw, yaw)), np.radians(rng.uniform(-pitch, pitch)),
               np.radians(rng.uniform(-roll, roll)))
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]], np.float32)
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]], np.float32)
    return (Rz @ Rx @ Ry @ R0).astype(np.float32)


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


# ───── BEDLAM 真实皮肤纹理渲染(SMPL UV) ─────
def parse_obj(path):
    """解析 smpl_uv_full.obj → vt(Nvt,2), face_v(F,3 SMPL顶点idx), face_vt(F,3 vt idx)。"""
    vt, fv, fvt = [], [], []
    for ln in open(path):
        if ln.startswith("vt "):
            _, a, b = ln.split()[:3]; vt.append((float(a), float(b)))
        elif ln.startswith("f "):
            t = ln.split()[1:4]
            fv.append([int(x.split("/")[0]) - 1 for x in t])
            fvt.append([int(x.split("/")[1]) - 1 for x in t])
    return np.array(vt, np.float32), np.array(fv, np.int64), np.array(fvt, np.int64)


def _vnormals(verts, faces):
    n = np.zeros_like(verts); tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for i in range(3):
        np.add.at(n, faces[:, i], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def _textured_mesh(verts, obj, albedo_rgb):
    """用 BEDLAM albedo(SMPL UV) 建带纹理的 pyrender.Mesh(平滑法线)。"""
    import pyrender
    vt, face_v, face_vt = obj
    nrm = _vnormals(verts, face_v)
    pos = verts[face_v].reshape(-1, 3).astype(np.float32)
    nor = nrm[face_v].reshape(-1, 3).astype(np.float32)
    uv = vt[face_vt].reshape(-1, 2).astype(np.float32).copy()    # BEDLAM 纹理: 不翻 v(实测 no-flip 对齐)
    idx = np.arange(len(pos), dtype=np.int64).reshape(-1, 3)
    tex = pyrender.Texture(source=albedo_rgb, source_channels="RGB")
    mat = pyrender.MetallicRoughnessMaterial(baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=0.7)
    return pyrender.Mesh(primitives=[pyrender.Primitive(positions=pos, normals=nor, texcoord_0=uv, indices=idx, material=mat)])


def _render_rgb(verts, faces, cam, H, W, skin, rng, tex_rgb=None, obj=None):
    """pyrender 平滑Phong + 三点柔光 → RGB。tex_rgb给定→用BEDLAM真皮纹理, 否则程序化肤色。"""
    import pyrender
    import trimesh
    textured = tex_rgb is not None and obj is not None
    if textured:
        pm = _textured_mesh(verts, obj, tex_rgb)
    else:
        mesh = trimesh.Trimesh(verts, faces, process=False)
        mesh.visual.vertex_colors = _skin_colors(verts, skin, rng)
        pm = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    amb = rng.uniform(0.45, 0.62) if textured else rng.uniform(0.35, 0.48)   # 环境光随机
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[amb, amb, amb])
    scene.add(pm)
    pose4 = np.linalg.inv(np.vstack([np.hstack([cam["R"], cam["T"][:, None]]), [0, 0, 0, 1]])) @ np.diag([1.0, -1, -1, 1])
    K = cam["K"]
    scene.add(pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.05, zfar=20), pose=pose4)
    # 随机光源(方向/强度/色温都转一转, 域随机化): 1主光 + 1-2补光, 朝向被摄体(-z)一侧
    base_it = 1.5 if textured else 0.9
    n_l = rng.randint(2, 4)
    for li in range(n_l):
        d = np.array([rng.uniform(-1, 1), rng.uniform(-0.8, 0.8), -abs(rng.uniform(0.4, 1.2))], np.float32)
        it = base_it * (rng.uniform(0.7, 1.3) if li == 0 else rng.uniform(0.2, 0.6))   # 主光强/补光弱
        ct = rng.uniform(-0.12, 0.12)                                                  # 色温(暖/冷)
        col = np.clip([1 + ct, 1.0, 1 - ct], 0.6, 1.3)
        z = -d / np.linalg.norm(d); x = np.cross([0, 1, 0], z)
        x = x / (np.linalg.norm(x) + 1e-9); lp = np.eye(4); lp[:3, :3] = np.c_[x, np.cross(z, x), z]
        scene.add(pyrender.DirectionalLight(color=col, intensity=float(it)), pose=pose4 @ lp)
    r = pyrender.OffscreenRenderer(W, H)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA); r.delete()
    rgb, a = color[:, :, :3], color[:, :, 3]
    bg = np.full((H, W, 3), int(rng.uniform(150, 205)), np.uint8)
    return np.where((a > 10)[:, :, None], rgb, bg).astype(np.uint8)


def _render_base(model, faces, uvmap, ausc, torso_idx, rng, view, posture, framing, H, W, focal,
                 obj=None, textures=None):
    """渲染一个 base sample: RGB + 稠密IUV + 听诊点(相机取景 F1/F2/F3)。
    textures给定(BEDLAM真皮路径列表)→用真实纹理; 否则程序化肤色。"""
    import torch
    betas = np.clip(rng.normal(0, 1.4, (1, 10)), -3, 3).astype(np.float32)
    pose = _seated_pose(rng) if posture == "seated" else _natural_pose(rng)
    with torch.no_grad():
        o = model(betas=torch.tensor(betas), body_pose=torch.tensor(pose),
                  global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3))
    verts = o.vertices[0].numpy()
    R = _jitter_view(VIEW_R[view], rng); P = (R @ verts.T).T   # 随机相机视角(方位/俯仰/滚转)
    tc = P[torso_idx].mean(0); torso_h = float(P[torso_idx, 1].ptp())   # 对准躯干中心(身体居中, 不偏心)
    k, fill, cyoff = FRAMINGS[framing]
    cam = _make_cam(tc[0], tc[1] + cyoff * torso_h, torso_h * k, H, W, focal, fill, R)
    tex_rgb = None
    if textures:
        tex_rgb = cv2.imread(textures[rng.randint(len(textures))])[:, :, ::-1].copy()  # BEDLAM真皮 albedo
    skin = SKIN[rng.randint(len(SKIN))] * rng.uniform(0.95, 1.05)   # 程序化兜底
    img = _render_rgb(verts, faces, cam, H, W, skin, rng, tex_rgb=tex_rgb, obj=obj)
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
    # ── 几何: 透视视角抖动 或 对称旋转(二选一) ──
    if rng.random() < 0.45:
        f = perspective_warp(s, rng); f["region"] = region; s = f
    else:
        ang = float(rng.choice(ROT_CHOICES))
        if ang != 0.0:
            M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
            bgc = tuple(int(c) for c in s["image"][0, 0])
            s["image"] = cv2.warpAffine(s["image"], M, (W, H), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=bgc)
            s["iuv"] = cv2.warpAffine(s["iuv"], M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
            for nm, p in s["points"].items():
                v = M @ np.array([p["xy"][0], p["xy"][1], 1.0])
                s["points"][nm] = {"xy": (float(v[0]), float(v[1])), "visible": bool(0 <= v[0] < W and 0 <= v[1] < H)}
    if region == "back" and rng.random() < 0.5:
        f = hflip(s, lr_swap=LR_SWAP); f["region"] = region; s = f
    # ── 光照/外观(只改图) ──
    s["image"] = clinical_lighting(s["image"], rng)
    if rng.random() < 0.4:
        s["image"] = hue_shift(s["image"], rng)
    if rng.random() < 0.3:
        s["image"] = specular(s["image"], rng)
    if rng.random() < 0.25:
        s["image"] = cast_shadow(s["image"], rng)
    # ── 遮挡 ──
    if rng.random() < 0.6:                                       # 机械臂遮挡(从右下方够向听诊点)
        s["image"] = occlude_medical(s["image"], s["points"], rng,
                                     region=region, iuv=s.get("iuv"))
    if rng.random() < 0.2:
        s["image"] = sensor_degrade(s["image"], rng, severity="mild")
    return s


def generate(smpl_model, uv, out, n_subjects=400, n_aug=3, res=512, focal=1100.0, seed=0,
             texture_dir=None, smpl_uv_obj=None):
    """纯仿真 pyrender 量产。需 GPU+EGL。每受试者 × 站/坐 × 前/后 × F1/F2/F3 × (1+n_aug)。
    texture_dir(BEDLAM真皮 SMPL albedo 目录)+ smpl_uv_obj 给定 → 用真实皮肤纹理; 否则程序化肤色。"""
    import glob
    import smplx
    model = smplx.SMPL(model_path=smpl_model, gender="neutral", batch_size=1)
    faces = model.faces
    uvmap = load_uv_mapping(uv)
    ausc = define_ausc_vertices(model)
    from humman_label import infer_side  # noqa: F401 (确保模块就绪)
    torso_idx = _torso_indices(uvmap)
    obj = textures = None
    if texture_dir and smpl_uv_obj:
        obj = parse_obj(smpl_uv_obj)
        textures = sorted(glob.glob(os.path.join(texture_dir, "*.png")))
        print(f"[synthetic] BEDLAM 真皮纹理: {len(textures)} 张, UV={os.path.basename(smpl_uv_obj)}")
    with ManifestWriter(out) as mw:
        for si in range(n_subjects):
            for posture in ("standing", "seated"):
                for view in ("front", "back"):
                    for fr in FRAMINGS:
                        r = np.random.RandomState(seed * 100003 + si * 97 + variant_seed_offset(posture, view, fr))
                        base = _render_base(model, faces, uvmap, ausc, torso_idx, r, view, posture, fr, res, res, focal,
                                            obj=obj, textures=textures)
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
