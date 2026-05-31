#!/usr/bin/env python3
"""
质检渲染器: 把 稠密IUV标签 + 听诊点 + 自动转正 + 手臂遮挡增强 串起来,
为人工逐张检查质量(位置/效果/转正/场景有用性)而生。

每个样本出一张三联图:
  (1) 转正后 IUV稠密标签叠加 + 听诊点  → 看标签位置/对齐/转正
  (2) 转正后 原图 + 听诊点              → 看点是否贴对身体
  (3) 转正 + 手臂听诊器遮挡             → 看遮挡增强是否真实(标签仍在)

用法:
  python qc_render.py --root ../data/humman --color iphone_color --region front \
      --out ../data/_viz_humman/qc --n 8
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ausc"))


def posed(model, params, fr):
    import torch
    with torch.no_grad():
        o = model(betas=torch.tensor(params["betas"][fr:fr+1], dtype=torch.float32),
                  body_pose=torch.tensor(params["body_pose"][fr:fr+1], dtype=torch.float32),
                  global_orient=torch.tensor(params["global_orient"][fr:fr+1], dtype=torch.float32),
                  transl=torch.tensor(params["transl"][fr:fr+1], dtype=torch.float32))
    return o.vertices[0].numpy(), o.joints[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/humman")
    ap.add_argument("--smpl_model", default="../data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="../data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--color", default="iphone_color")
    ap.add_argument("--region", choices=["front", "back"], default="front")
    ap.add_argument("--out", default="../data/_viz_humman/qc"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--scan", type=int, default=80)
    args = ap.parse_args()
    import smplx, cv2, torch, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from render_iuv import load_uv_mapping, rasterize_iuv, project
    from humman_label import define_ausc_vertices
    from frame_filter import front_core_indices, classify_frame
    import augment as A

    model = smplx.SMPL(model_path=args.smpl_model, gender="neutral", batch_size=1)
    faces = model.faces; core = front_core_indices(model)
    uvmap = load_uv_mapping(args.uv); ausc = define_ausc_vertices(model)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(0)
    seqs = sorted(glob.glob(os.path.join(args.root, "smpl_params", "*")))[:args.scan]
    done = 0
    for sq in seqs:
        if done >= args.n:
            break
        sid = os.path.basename(sq)
        params = dict(np.load(os.path.join(sq, "smpl_params.npz")))
        cams = json.load(open(os.path.join(args.root, "point_cameras", sid, "cameras.json")))
        # 选相机: iphone(只有front); kinect 选朝向匹配 region 的
        camkey = "iphone" if args.color == "iphone_color" else None
        nfp = params["betas"].shape[0]
        # 选朝向匹配 region 的最佳帧
        bestfr, bestz = -1, 0
        for fr in range(0, nfp, max(1, nfp // 12)):
            v, jj = posed(model, params, fr)
            R = np.asarray(cams[camkey or "kinect_color_000"]["R"])
            lab, z = classify_frame(v, faces, R, core, joints=jj)
            if lab == args.region and abs(z) > bestz:
                bestz, bestfr = abs(z), fr
        if bestfr < 0 or bestz < 0.6:
            continue
        mp4 = glob.glob(os.path.join(args.root, args.color, sid, "**", "*.mp4"), recursive=True)
        if not mp4:
            continue
        cap = cv2.VideoCapture(mp4[0]); cap.set(cv2.CAP_PROP_POS_FRAMES, bestfr); ok, bgr = cap.read(); cap.release()
        if not ok:
            continue
        cam = cams[camkey]
        H, W = bgr.shape[:2]
        verts, joints = posed(model, params, bestfr)
        I_map, U_map, V_map, mask = rasterize_iuv(verts, cam, uvmap, H, W)
        uv, _ = project(verts, cam)
        juv, _ = project(joints, cam)
        pelvis_uv, neck_uv = juv[0], juv[12]
        rgb = bgr[:, :, ::-1].copy()
        colr = (plt.cm.tab20(I_map / 24.0)[:, :, :3] * 255).astype(np.uint8)
        iuv_ov = rgb.copy(); iuv_ov[mask] = (0.45 * iuv_ov[mask] + 0.55 * colr[mask]).astype(np.uint8)
        pts = {nm: uv[vi] for nm, vi in ausc[args.region].items()}
        c = (0, 255, 255) if args.region == "front" else (255, 150, 0)
        def draw(im):
            im = im.copy()
            for nm, p in pts.items():
                if np.all(np.isfinite(p)):
                    cv2.drawMarker(im, (int(p[0]), int(p[1])), c, cv2.MARKER_TILTED_CROSS, 22, 3)
            return im
        # 遮挡图: 在原图上对一个心脏点贴手臂遮挡
        tgt = pts.get("mitral", list(pts.values())[0])
        occ = A.occlude_arm(bgr[:, :, ::-1], tgt, rng)
        # 统一转正 M (脊柱→竖直), warp 三联 + 用顶点uv算人体bbox裁剪
        dx, dy = neck_uv[0] - pelvis_uv[0], neck_uv[1] - pelvis_uv[1]
        ang = np.degrees(np.arctan2(dx, -dy))
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
        warp = lambda im: cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        ruv = (M @ np.c_[uv, np.ones(len(uv))].T).T            # 旋转后顶点
        good = np.isfinite(ruv).all(1)
        x0, y0 = ruv[good].min(0); x1, y1 = ruv[good].max(0)
        mx, my = (x1 - x0) * 0.15, (y1 - y0) * 0.12
        x0, y0, x1, y1 = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx)), int(min(H, y1 + my))
        crop = lambda im: im[y0:y1, x0:x1]
        panels = [crop(warp(draw(iuv_ov))), crop(warp(draw(rgb))), crop(warp(draw(occ)))]
        titles = ["(1) IUV label + points (upright)", "(2) image + points", "(3) +arm occlusion"]
        fig, ax = plt.subplots(1, 3, figsize=(16, 7))
        for a, im, t in zip(ax, panels, titles):
            a.imshow(im); a.set_title(t, fontsize=10); a.axis("off")
        plt.suptitle(f"{sid} f{bestfr} [{args.region}, z={bestz:.2f}]", fontsize=10)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, f"{args.region}_{sid}.png"), dpi=85, bbox_inches="tight"); plt.close()
        print("qc", sid, "frame", bestfr, "z", round(bestz, 2)); done += 1
    print(f"done {done} -> {args.out}")


if __name__ == "__main__":
    main()
