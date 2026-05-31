#!/usr/bin/env python3
"""HuMMan → 可训练样本(稠密 IUV + 听诊点)。

每序列选一帧最正的正面(iPhone, 4 心点)和/或一帧最正的背面(Kinect 后向, 8 肺点),
渲染稠密 IUV、转正、裁到躯干 → base sample; 再随机组合增强出 n_aug 张, 并按比例派生负样本。

⚠️ U,V 为每面均值(粗); 全量精确连续 UV 用 GPU+pytorch3d(见 datasets_and_processing.md §五)。
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json  # noqa: E402

from augment import occlude_arm  # noqa: E402,F401  (增强经 common 调用)
from common import ManifestWriter, augment_once, write_sample  # noqa: E402
from frame_filter import classify_frame, front_core_indices  # noqa: E402
from humman_io import find_kinect_videos  # noqa: E402
from humman_label import define_ausc_vertices  # noqa: E402
from render_iuv import load_uv_mapping, project, rasterize_iuv  # noqa: E402


def _posed(model, params, fr):
    import torch
    with torch.no_grad():
        o = model(betas=torch.tensor(params["betas"][fr:fr + 1], dtype=torch.float32),
                  body_pose=torch.tensor(params["body_pose"][fr:fr + 1], dtype=torch.float32),
                  global_orient=torch.tensor(params["global_orient"][fr:fr + 1], dtype=torch.float32),
                  transl=torch.tensor(params["transl"][fr:fr + 1], dtype=torch.float32))
    return o.vertices[0].numpy(), o.joints[0].numpy()


def _read_frame(vid, fr):
    cap = cv2.VideoCapture(vid)
    nfk = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(fr, nfk - 1))
    ok, bgr = cap.read()
    cap.release()
    return (bgr[:, :, ::-1] if ok else None)


def _base_sample(model, faces, uvmap, ausc, params, fr, cam, region):
    """渲染+转正+裁躯干 → base sample dict, 或 None。"""
    verts, joints = _posed(model, params, fr)
    H, W = cam["_HW"]
    I_map, U_map, V_map, _ = rasterize_iuv(verts, cam, uvmap, H, W)
    iuv = np.dstack([I_map, (U_map * 255).astype(np.uint8), (V_map * 255).astype(np.uint8)])
    uv, _ = project(verts, cam)
    juv, _ = project(joints, cam)
    # 转正: 脊柱 pelvis(0)→neck(12) 竖直
    dx, dy = juv[12, 0] - juv[0, 0], juv[12, 1] - juv[0, 1]
    ang = np.degrees(np.arctan2(dx, -dy))
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    rgb = cam["_RGB"]
    img_u = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR)
    iuv_u = cv2.warpAffine(iuv, M, (W, H), flags=cv2.INTER_NEAREST)
    ruv = (M @ np.c_[uv, np.ones(len(uv))].T).T
    g = np.isfinite(ruv).all(1)
    if g.sum() < 10:
        return None
    x0, y0 = ruv[g].min(0)
    x1, y1 = ruv[g].max(0)
    mx, my = (x1 - x0) * .15, (y1 - y0) * .12
    bx0, by0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    bx1, by1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
    img_c, iuv_c = img_u[by0:by1, bx0:bx1], iuv_u[by0:by1, bx0:bx1]
    if img_c.size == 0:
        return None
    pts = {}
    ch, cw = img_c.shape[:2]
    for nm, vi in ausc[region].items():
        v3 = M @ np.array([uv[vi, 0], uv[vi, 1], 1.0])
        px, py = v3[0] - bx0, v3[1] - by0
        pts[nm] = {"xy": (float(px), float(py)), "visible": bool(0 <= px < cw and 0 <= py < ch)}
    return {"image": img_c, "iuv": iuv_c, "points": pts, "region": region,
            "_extra": {"img_u": img_u, "iuv_u": iuv_u, "juv": juv, "M": M, "bbox": (bx0, by0, bx1, by1)}}


def _pick_views(model, faces, core, params, cams, root, sid):
    """返回 [(region, cam_key|'iphone', frame, video_path)], 最多前/后各一个。"""
    out = []
    nfp = params["betas"].shape[0]
    # 正面: iPhone 选最正帧
    if "iphone" in cams:
        R = np.asarray(cams["iphone"]["R"])
        best = (-1.0, -1)
        for f in range(0, nfp, max(1, nfp // 10)):
            v, jj = _posed(model, params, f)
            lab, z = classify_frame(v, faces, R, core, joints=jj)
            if lab == "front" and abs(z) > best[0]:
                best = (abs(z), f)
        mp4 = glob.glob(os.path.join(root, "iphone_color", sid, "**", "*.mp4"), recursive=True)
        if best[1] >= 0 and best[0] > 0.6 and mp4:
            out.append(("front", "iphone", best[1], mp4[0]))
    # 背面: Kinect 后向相机选最正
    vids = find_kinect_videos(root, sid)
    if vids:
        f0 = nfp // 2
        v, jj = _posed(model, params, f0)
        best = (-1.0, None)
        for ck in vids:
            if ck in cams:
                lab, z = classify_frame(v, faces, np.asarray(cams[ck]["R"]), core, joints=jj)
                if lab == "back" and z > best[0]:
                    best = (z, ck)
        if best[1]:
            out.append(("back", best[1], f0, vids[best[1]]))
    return out


def _emit_negative(base, kind, rng):
    """从同帧派生一个负样本 sample(region=none)。kind: limb/bg。"""
    ex = base["_extra"]
    img_u, iuv_u, juv = ex["img_u"], ex["iuv_u"], ex["juv"]
    H, W = img_u.shape[:2]
    if kind == "limb":
        aj = ex["M"] @ np.array([juv[7, 0], juv[7, 1], 1.0])  # 左脚踝
        s = max(20, (ex["bbox"][3] - ex["bbox"][1]) // 4)
        cx, cy = int(aj[0]), int(aj[1])
        sub_i = iuv_u[max(0, cy - s):cy + s, max(0, cx - s):cx + s]
        if sub_i.size and (sub_i[:, :, 0] == 0).mean() < 0.95:
            return {"image": img_u[max(0, cy - s):cy + s, max(0, cx - s):cx + s],
                    "iuv": sub_i, "points": {}, "region": "none"}
    s = min(H, W) // 3                                          # 背景角落
    for yy, xx in [(0, 0), (0, W - s), (H - s, 0), (H - s, W - s)]:
        bi = iuv_u[yy:yy + s, xx:xx + s]
        if (bi[:, :, 0] > 0).mean() < 0.05:
            return {"image": img_u[yy:yy + s, xx:xx + s], "iuv": bi, "points": {}, "region": "none"}
    return None


def generate(root, smpl_model, uv, out, limit=None, n_aug=6, neg_ratio=0.25,
             target=(384, 384), seed=0):
    """全量(或前 limit 个序列)生成 HuMMan 训练样本。"""
    import smplx
    model = smplx.SMPL(model_path=smpl_model, gender="neutral", batch_size=1)
    faces = model.faces
    core = front_core_indices(model)
    uvmap = load_uv_mapping(uv)
    ausc = define_ausc_vertices(model)
    rng = np.random.RandomState(seed)
    seqs = sorted(glob.glob(os.path.join(root, "smpl_params", "*")))
    if limit:
        seqs = seqs[:limit]
    n_pos = 0
    with ManifestWriter(out) as mw:
        for sq in seqs:
            sid = os.path.basename(sq)
            try:
                params = dict(np.load(os.path.join(sq, "smpl_params.npz")))
                cams = json.load(open(os.path.join(root, "point_cameras", sid, "cameras.json")))
            except (FileNotFoundError, OSError):
                continue
            for region, cam_key, fr, vid in _pick_views(model, faces, core, params, cams, root, sid):
                cam = dict(cams[cam_key] if cam_key != "iphone" else cams["iphone"])
                rgb = _read_frame(vid, fr)
                if rgb is None:
                    continue
                cam["_HW"], cam["_RGB"] = rgb.shape[:2], rgb
                base = _base_sample(model, faces, uvmap, ausc, params, fr, cam, region)
                if base is None:
                    continue
                sid_id = f"{sid}_{region}_f{fr}"
                mw.add(write_sample(out, sid_id + "_canon", base))           # 规范样本
                for k in range(n_aug):                                       # 随机组合增强
                    aug = augment_once(base, rng, target=target)
                    mw.add(write_sample(out, f"{sid_id}_aug{k}", aug))
                n_pos += 1
                if neg_ratio > 0 and rng.random() < neg_ratio:              # 负样本
                    neg = _emit_negative(base, "limb" if rng.random() < 0.5 else "bg", rng)
                    if neg is not None and neg["image"].size:
                        mw.add(write_sample(out, f"{sid_id}_neg", neg, is_valid=False))
        print("[humman]", mw.summary())
