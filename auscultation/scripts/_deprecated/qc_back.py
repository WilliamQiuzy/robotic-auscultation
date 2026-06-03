#!/usr/bin/env python3
"""
后背/肺 质检渲染: 从 Kinect 多相机里自动挑【后背朝相机】的视角, 渲染肺点 IUV 标签。

Kinect 有 10 个环绕相机(kinect_color_000..009), 自动对每个相机分类朝向, 选 back 的。
输出三联图(转正): IUV标签+肺点 / 原图+肺点 / +手臂遮挡。

用法:
  python qc_back.py --root ../data/humman --out ../data/_viz_humman/qc_back --n 6
"""
import argparse, glob, json, os, re, sys
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


from humman_io import find_kinect_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/humman")
    ap.add_argument("--smpl_model", default="../data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="../data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--out", default="../data/_viz_humman/qc_back"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--scan", type=int, default=80)
    args = ap.parse_args()
    import smplx, cv2, matplotlib
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
        vids = find_kinect_videos(args.root, sid)
        if not vids:
            continue
        params = dict(np.load(os.path.join(sq, "smpl_params.npz")))
        cams = json.load(open(os.path.join(args.root, "point_cameras", sid, "cameras.json")))
        nfp = params["betas"].shape[0]; fr = nfp // 2
        verts, joints = posed(model, params, fr)
        # 在已有视频的相机里挑 back 朝向最强的
        best = (-1, None)
        for ck, path in vids.items():
            if ck not in cams:
                continue
            lab, z = classify_frame(verts, faces, np.asarray(cams[ck]["R"]), core, joints=joints)
            if lab == "back" and z > best[0]:
                best = (z, ck)
        if best[1] is None:
            continue
        ck = best[1]; cam = cams[ck]
        cap = cv2.VideoCapture(vids[ck]); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ff = min(fr, nf - 1); cap.set(cv2.CAP_PROP_POS_FRAMES, ff); ok, bgr = cap.read(); cap.release()
        if not ok:
            continue
        H, W = bgr.shape[:2]
        I_map, U_map, V_map, mask = rasterize_iuv(verts, cam, uvmap, H, W)
        uv, _ = project(verts, cam); juv, _ = project(joints, cam)
        pts = {nm: uv[vi] for nm, vi in ausc["back"].items()}
        rgb = bgr[:, :, ::-1].copy()
        colr = (plt.cm.tab20(I_map / 24.0)[:, :, :3] * 255).astype(np.uint8)
        iuv_ov = rgb.copy(); iuv_ov[mask] = (0.45 * iuv_ov[mask] + 0.55 * colr[mask]).astype(np.uint8)
        def draw(im):
            im = im.copy()
            for nm, p in pts.items():
                if np.all(np.isfinite(p)):
                    cv2.drawMarker(im, (int(p[0]), int(p[1])), (255, 150, 0), cv2.MARKER_TILTED_CROSS, 22, 3)
            return im
        tgt = pts.get("L_mid", list(pts.values())[0])
        occ = A.occlude_arm(rgb, tgt, rng)
        dx, dy = juv[12, 0] - juv[0, 0], juv[12, 1] - juv[0, 1]
        ang = np.degrees(np.arctan2(dx, -dy))
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
        warp = lambda im: cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR)
        ruv = (M @ np.c_[uv, np.ones(len(uv))].T).T
        g = np.isfinite(ruv).all(1); x0, y0 = ruv[g].min(0); x1, y1 = ruv[g].max(0)
        mx, my = (x1 - x0) * .15, (y1 - y0) * .1
        x0, y0, x1, y1 = int(max(0, x0-mx)), int(max(0, y0-my)), int(min(W, x1+mx)), int(min(H, y1+my))
        crop = lambda im: im[y0:y1, x0:x1]
        panels = [crop(warp(draw(iuv_ov))), crop(warp(draw(rgb))), crop(warp(draw(occ)))]
        fig, ax = plt.subplots(1, 3, figsize=(16, 7))
        for a, im, t in zip(ax, panels, ["(1) IUV+lung pts", "(2) image+pts", "(3) +occlusion"]):
            a.imshow(im); a.set_title(t, fontsize=10); a.axis("off")
        plt.suptitle(f"{sid} {ck} f{ff} [back z={best[0]:.2f}]", fontsize=10)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, f"back_{sid}.png"), dpi=85, bbox_inches="tight"); plt.close()
        print("qc_back", sid, ck, "z", round(best[0], 2)); done += 1
    print(f"done {done} -> {args.out}")


if __name__ == "__main__":
    main()
