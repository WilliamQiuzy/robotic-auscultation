#!/usr/bin/env python3
"""数据增强总览 (X-OR): 取一帧真实 base sample, 把【每一种增强手段】单独施加一次, 拼成一张大图。
三数据集共用同一套算子(augment.py); 取景语义按数据集:
  humman / scoliosis : 2D 真实图 → crop_to_framing 裁出 F1/F2/F3/F4(common.augment_once 路径)
  synthetic          : 3D 渲染 → 相机距离渲出 F1/F2/F3/F4(synthetic._augment 路径, 不二次裁剪)
后背(scoliosis)额外含 hflip+换 L/R(肺双侧对称); 前胸禁翻。

用法:
  python scripts/qc/aug_showcase.py humman       # 本机(HuMMan 数据在本地)
  python scripts/qc/aug_showcase.py scoliosis    # 服务器(需 scol_refit.pkl)
  python scripts/qc/aug_showcase.py synthetic    # 服务器(需 EGL + BEDLAM + smpl_uv obj)
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "datasets"))

import glob  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402

import smplx  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import augment as A  # noqa: E402
import common as C  # noqa: E402
from humman_label import define_ausc_vertices  # noqa: E402
from render_iuv import load_uv_mapping, rasterize_iuv  # noqa: E402

DATA = os.path.join(ROOT, "data")
SMPL = os.path.join(DATA, "smpl", "SMPL_NEUTRAL.pkl")
UV = os.path.join(DATA, "densepose_coco", "uv_data", "UV_Processed.mat")


def colorize_iuv(iuv):
    """稠密 IUV → 可视化: part 用 tab20 上色, 背景黑。"""
    part = iuv[:, :, 0]
    cmap = (np.array(plt.get_cmap("tab20").colors) * 255).astype(np.uint8)
    out = np.zeros((*part.shape, 3), np.uint8)
    for p in range(1, 25):
        out[part == p] = cmap[p % 20]
    return out


def letterbox(img, size=384, pad=(235, 235, 235)):
    """保比例缩放 + 补边到正方形(仅展示用, 不拉伸)。"""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), pad, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


class ModeRNG:
    """包装 RandomState: sensor_degrade 的模式 choice 强制为 mode, 其余照常。"""
    def __init__(self, mode, seed=0):
        self.mode = mode
        self.r = np.random.RandomState(seed)

    def choice(self, seq, *a, **k):
        if isinstance(seq, (list, tuple)) and "motion" in seq:
            return self.mode
        return self.r.choice(seq, *a, **k)

    def __getattr__(self, n):
        return getattr(self.r, n)


# ── base sample 构造(按数据集) ───────────────────────────────────────────────

def base_humman(model, faces, uvmap, ausc):
    """本机 HuMMan: 扫序列取首个正面 base。"""
    from frame_filter import front_core_indices
    import humman as HM
    core = front_core_indices(model)
    for sq in sorted(glob.glob(os.path.join(DATA, "humman", "smpl_params", "*"))):
        sid = os.path.basename(sq)
        try:
            params = dict(np.load(os.path.join(sq, "smpl_params.npz")))
            cams = json.load(open(os.path.join(DATA, "humman", "point_cameras", sid, "cameras.json")))
        except (FileNotFoundError, OSError):
            continue
        for region, ck, fr, vid in HM._pick_views(model, faces, core, params, cams, os.path.join(DATA, "humman"), sid):
            if region != "front":
                continue
            cam = dict(cams["iphone"]); rgb = HM._read_frame(vid, fr)
            if rgb is None:
                continue
            cam["_HW"], cam["_RGB"] = rgb.shape[:2], rgb
            base = HM._base_sample(model, faces, uvmap, ausc, params, fr, cam, region)
            if base is not None and base["image"].size:
                print(f"[humman] base {sid} f{fr}")
                return base, "front", None
    raise RuntimeError("no humman front base")


def base_scoliosis(model, faces, uvmap, ausc):
    """服务器 scoliosis: 用 scol_refit.pkl + 图, 构 1 个后背 base(同 scoliosis.generate)。"""
    pkl = "/root/scol_refit.pkl" if os.path.exists("/root/scol_refit.pkl") else os.path.join(DATA, "scol_refit.pkl")
    img_root = "/root/scol86" if os.path.isdir("/root/scol86") else os.path.join(DATA, "scoliosis_back")
    labs = pickle.load(open(pkl, "rb"))
    back_idx = ausc["back"]
    for name, lab in labs.items():
        if "verts_cam" not in lab:
            continue
        path = None
        direct = os.path.join(img_root, name)
        if os.path.exists(direct):
            path = direct
        else:
            for r, _d, fs in os.walk(img_root):
                if name in fs:
                    path = os.path.join(r, name); break
        bgr = cv2.imread(path) if path else None
        if bgr is None:
            continue
        rgb = bgr[:, :, ::-1]; H, W = rgb.shape[:2]
        lw, lh = lab["wh"]; sx, sy = W / float(lw), H / float(lh)
        f = lab["focal"] * (sx + sy) / 2.0
        cam = {"K": np.array([[f, 0, W / 2.], [0, f, H / 2.], [0, 0, 1.]], np.float32),
               "R": np.eye(3, dtype=np.float32), "T": np.zeros(3, np.float32)}
        I, U, V, _ = rasterize_iuv(np.asarray(lab["verts_cam"], np.float32).copy(), cam, uvmap, H, W)
        iuv = np.dstack([I, (U * 255).astype(np.uint8), (V * 255).astype(np.uint8)])
        v2d = np.asarray(lab["verts2d"], np.float32)
        pts = {nm: {"xy": (float(v2d[vi, 0] * sx), float(v2d[vi, 1] * sy)),
                    "visible": bool(0 <= v2d[vi, 0] * sx < W and 0 <= v2d[vi, 1] * sy < H)}
               for nm, vi in back_idx.items()}
        print(f"[scoliosis] base {name[:24]}")
        return {"image": rgb, "iuv": iuv, "points": pts, "region": "back"}, "back", None
    raise RuntimeError("no scoliosis base with verts_cam")


def base_synthetic(model, faces, uvmap, ausc):
    """服务器 synthetic: 渲 F1-F4(前胸站姿), 返回 F2 base + 各 framing 渲染图(3D取景)。"""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import synthetic as SY
    torso = SY._torso_indices(uvmap)
    obj = SY.parse_obj(os.path.join(DATA, "smpl_assets", "smpl_uv_full.obj"))
    tex = sorted(glob.glob(os.path.join(DATA, "bedlam_skin", "*.png")))
    framings = {}
    for fk in ["F1_upper", "F2_torso", "F3_near", "F4_zoom"]:
        r = np.random.RandomState(7)
        framings[fk] = SY._render_base(model, faces, uvmap, ausc, torso, r, "front", "standing",
                                       fk, 512, 512, 1100.0, obj=obj, textures=tex)
    print("[synthetic] rendered F1-F4")
    return framings["F2_torso"], "front", framings


# ── 增强格构造 ───────────────────────────────────────────────────────────────

def build_cells(base, region, framings, dataset):
    """返回 [(title, image)] 列表: 原图 + IUV + F1-F4 + 各算子(同一套 augment.py)。"""
    cells = [("original (uprighted + crop)", letterbox(base["image"])),
             ("dense IUV (perception label)", letterbox(colorize_iuv(base["iuv"]), pad=(0, 0, 0)))]

    if framings is not None:                                  # synthetic: 3D 渲染取景
        names = {"F1_upper": "framing F1: far (cam)", "F2_torso": "framing F2: mid (cam)",
                 "F3_near": "framing F3: near (cam)", "F4_zoom": "framing F4: zoom (cam)"}
        for fk in ["F1_upper", "F2_torso", "F3_near", "F4_zoom"]:
            cells.append((names[fk], framings[fk]["image"]))
        ref = {"image": base["image"].copy(), "iuv": base["iuv"].copy(),
               "points": {k: dict(v) for k, v in base["points"].items()}, "region": region}
    else:                                                     # humman / scoliosis: 2D 裁剪取景
        names = {"F1_upper": "framing F1: far (more context)", "F2_torso": "framing F2: mid",
                 "F3_near": "framing F3: near", "F4_zoom": "framing F4: zoom (into chest)"}
        for fk in ["F1_upper", "F2_torso", "F3_near", "F4_zoom"]:
            c = C.crop_to_framing(base, fk, np.random.RandomState(0), target=(384, 384))
            cells.append((names[fk], c["image"]))
        ref = C.crop_to_framing(base, "F2_torso", np.random.RandomState(0), target=(384, 384))
        ref["region"] = region
    refimg = ref["image"]

    # 几何
    cells.append(("rotation +8 deg", C._rotate(dict(ref), 8.0)["image"]))
    cells.append(("perspective jitter", A.perspective_warp(dict(ref), np.random.RandomState(1))["image"]))
    if region == "back":                                      # 仅后背: 水平翻转 + 换 L/R
        fl = A.hflip(dict(ref), lr_swap=A.LR_SWAP)
        cells.append(("hflip + L/R swap (back only)", fl["image"]))
    # 外观
    cells.append(("clinical lighting", A.clinical_lighting(refimg.copy(), np.random.RandomState(2))))
    cells.append(("hue shift", A.hue_shift(refimg.copy(), np.random.RandomState(5))))
    cells.append(("specular highlight", A.specular(refimg.copy(), np.random.RandomState(4))))
    cells.append(("cast shadow", A.cast_shadow(refimg.copy(), np.random.RandomState(6))))
    # 遮挡 + 退化
    cells.append(("robot arm occlusion",
                  A.occlude_medical(refimg.copy(), ref["points"], np.random.RandomState(8),
                                    region=region, iuv=ref.get("iuv"))))
    cells.append(("motion blur", A.sensor_degrade(refimg.copy(), ModeRNG("motion", 1))))
    cells.append(("JPEG compression", A.sensor_degrade(refimg.copy(), ModeRNG("jpeg", 2))))
    cells.append(("sensor noise", A.sensor_degrade(refimg.copy(), ModeRNG("noise", 3))))
    return cells


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "humman"
    out = os.path.join(DATA, "_viz_synthetic", f"AUG_{dataset}_all.png")
    model = smplx.SMPL(model_path=SMPL, gender="neutral", batch_size=1)
    faces = model.faces
    uvmap = load_uv_mapping(UV)
    ausc = define_ausc_vertices(model)

    builder = {"humman": base_humman, "scoliosis": base_scoliosis, "synthetic": base_synthetic}[dataset]
    base, region, framings = builder(model, faces, uvmap, ausc)
    cells = build_cells(base, region, framings, dataset)

    n = len(cells); cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.6))
    for i, ax in enumerate(np.array(axes).flat):
        if i < n:
            title, im = cells[i]
            ax.imshow(im); ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.suptitle(f"{dataset} augmentation operators (each cell = one operator, same source)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=95, bbox_inches="tight"); plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
