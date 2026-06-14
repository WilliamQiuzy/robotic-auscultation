#!/usr/bin/env python3
"""
HuMMan 帧级 正面/背面 过滤器。

按 SMPL 躯干法向相对相机的朝向, 把每帧分:
  front  : 前胸朝相机 → 可用作心脏点标签
  back   : 后背朝相机 → 可用作肺点标签
  reject : 侧身/朝向不明 → 丢弃
(临床域: 病人躯干面向相机; 侧身帧标签无意义)

用法(扫描+渲染筛选后的样本):
  python frame_filter.py --root ../data/humman --smpl_model ../data/smpl/SMPL_NEUTRAL.pkl \
      --uv ../data/densepose_coco/uv_data/UV_Processed.mat --out ../data/_viz_humman/filtered --n 8
"""
import argparse, glob, json, os
import numpy as np


def front_core_indices(model):
    """前胸躯干核心顶点(模板, 去手臂/腿/头) —— 用于算躯干朝向。"""
    import torch
    from humman_label import _vertex_normals
    with torch.no_grad():
        V = model(betas=torch.zeros(1, 10), body_pose=torch.zeros(1, 69),
                  global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3)).vertices[0].numpy()
    N = _vertex_normals(V, model.faces)
    x, y = V[:, 0], V[:, 1]
    core = (N[:, 2] > 0.2) & (np.abs(x) < 0.16) & (y > -0.25) & (y < 0.19)
    return np.where(core)[0]


def classify_frame(posed_verts, faces, R, core_idx, joints=None, thresh=0.6, bend_thresh=0.5):
    """返回 (label, z)。
    - 朝向: chest 法向相机系 z 分量, z<-thresh=front(前胸朝相机), z>thresh=back, 之间=侧身reject。
      thresh 收紧到 0.6 → 剔除 3/4 侧转。
    - 弯腰/前缩(传 joints 时): 脊柱(neck-pelvis)在相机系若大幅指向相机轴(|z|>bend_thresh),
      说明躯干前倾/后仰/下蹲(前缩),标注会压缩 → reject。"""
    from humman_label import _vertex_normals
    N = _vertex_normals(posed_verts, faces)
    n = N[core_idx].mean(0); n = n / (np.linalg.norm(n) + 1e-9)
    z = float((R @ n)[2])
    if joints is not None:
        spine = np.asarray(joints[12]) - np.asarray(joints[0])      # neck - pelvis (world)
        sc = R @ spine; sc = sc / (np.linalg.norm(sc) + 1e-9)
        if abs(float(sc[2])) > bend_thresh:                         # 躯干前缩(弯腰/下蹲/前倾)
            return "reject", z
    if z < -thresh:
        return "front", z
    if z > thresh:
        return "back", z
    return "reject", z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/humman")
    ap.add_argument("--smpl_model", default="../data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="../data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--out", default="../data/_viz_humman/filtered"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--scan", type=int, default=60, help="扫描多少序列")
    args = ap.parse_args()
    import smplx, cv2, torch, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from render_iuv import load_uv_mapping, smpl_posed, rasterize_iuv, project
    from humman_label import define_ausc_vertices

    model = smplx.SMPL(model_path=args.smpl_model, gender="neutral", batch_size=1)
    faces = model.faces
    core = front_core_indices(model)
    uvmap = load_uv_mapping(args.uv)
    ausc = define_ausc_vertices(model)
    os.makedirs(args.out, exist_ok=True)

    seqs = sorted(glob.glob(os.path.join(args.root, "smpl_params", "*")))[:args.scan]
    stats = {"front": 0, "back": 0, "reject": 0}
    rendered = {"front": 0, "back": 0}
    for sq in seqs:
        sid = os.path.basename(sq)
        params = dict(np.load(os.path.join(sq, "smpl_params.npz")))
        cams = json.load(open(os.path.join(args.root, "point_cameras", sid, "cameras.json")))
        cam = cams["iphone"]; R = np.asarray(cam["R"])
        nfp = params["betas"].shape[0]
        # 找该序列里朝向最"正"的 front 帧 和 back 帧
        best = {"front": (0, 1e9, -1), "back": (0, 1e9, -1)}   # (|z|score, frame)
        for fr in range(0, nfp, max(1, nfp // 12)):
            v = smpl_posed(model, params, fr)
            lab, z = classify_frame(v, faces, R, core)
            stats[lab] += 1
            if lab in best and abs(z) > 0.6 and abs(z) < best[lab][1] * 0 + 99:
                if abs(z) > best[lab][0]:
                    best[lab] = (abs(z), z, fr)
        # 渲染该序列最强的一个朝向(够样本即停)
        for lab in ("front", "back"):
            if rendered[lab] >= args.n:
                continue
            score, z, fr = best[lab]
            if fr < 0 or score < 0.6:
                continue
            mp4 = glob.glob(os.path.join(args.root, "iphone_color", sid, "**", "*.mp4"), recursive=True)
            if not mp4:
                continue
            cap = cv2.VideoCapture(mp4[0]); cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, bgr = cap.read(); cap.release()
            if not ok:
                continue
            H, W = bgr.shape[:2]
            verts = smpl_posed(model, params, fr)
            I_map, U_map, V_map, mask = rasterize_iuv(verts, cam, uvmap, H, W)
            rgb = bgr[:, :, ::-1].copy()
            colr = (plt.cm.tab20(I_map / 24.0)[:, :, :3] * 255).astype(np.uint8)
            rgb[mask] = (0.45 * rgb[mask] + 0.55 * colr[mask]).astype(np.uint8)
            uv, _ = project(verts, cam)
            for nm, vi in ausc[lab].items():
                p = uv[vi]
                if np.all(np.isfinite(p)):
                    cv2.drawMarker(rgb, (int(p[0]), int(p[1])), (0, 255, 255) if lab == "front" else (255, 150, 0),
                                   cv2.MARKER_TILTED_CROSS, 20, 3)
            cv2.imwrite(os.path.join(args.out, f"{lab}_{sid}_f{fr}_z{z:+.2f}.png"), rgb[:, :, ::-1])
            rendered[lab] += 1
            print(f"rendered {lab} {sid} frame{fr} z={z:+.2f}")
    print("frame-level stats:", stats, "(front=前胸朝相机, back=后背, reject=侧身)")


if __name__ == "__main__":
    main()
