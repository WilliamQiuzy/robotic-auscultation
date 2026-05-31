#!/usr/bin/env python3
"""
样图画廊: 每张原图 → 一串处理后的图(含手臂遮挡增强), 放同一文件夹, 直观看"1原图→N处理图"。

每个原图输出到 data/sample_gallery/<id>/:
  0_original.jpg          原始帧(相机直出)
  1_canonical.jpg         转正+裁剪到人体(规范样本)
  2_iuv_label.jpg         稠密IUV标签叠加
  3_points.jpg            听诊点叠加
  4_aug_view1.jpg 5_..2   第三人称随机裁剪+缩放+光照增强
  6_occlusion1.jpg 7_..2  ★手臂+听诊器遮挡增强★
  8_arm_closeup.jpg       近景(arm视角)裁剪
  _montage.png            上述拼到一张, 一眼看全
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
    ap.add_argument("--out", default="../data/sample_gallery")
    args = ap.parse_args()
    import smplx, cv2, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from render_iuv import load_uv_mapping, rasterize_iuv, project
    from humman_label import define_ausc_vertices
    from humman_io import find_kinect_videos
    import augment as A

    model = smplx.SMPL(model_path=args.smpl_model, gender="neutral", batch_size=1)
    uvmap = load_uv_mapping(args.uv); ausc = define_ausc_vertices(model)
    # (sid, region, frame, camera_key 或 None=iphone)
    jobs = [("p000438_a000040", "front", 84, "iphone"),
            ("p000441_a000062", "front", 88, "iphone"),
            ("p000446_a000036", "back", None, None)]
    for sid, region, frame, ckey in jobs:
        params = dict(np.load(os.path.join(args.root, "smpl_params", sid, "smpl_params.npz")))
        cams = json.load(open(os.path.join(args.root, "point_cameras", sid, "cameras.json")))
        if region == "back":
            vids = find_kinect_videos(args.root, sid); ckey = list(vids)[0]; vid = vids[ckey]
        else:
            vid = glob.glob(os.path.join(args.root, "iphone_color", sid, "**", "*.mp4"), recursive=True)[0]
        cam = cams[ckey]
        cap = cv2.VideoCapture(vid); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fr = (nf // 2) if frame is None else min(frame, nf - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, bgr = cap.read(); cap.release()
        rgb = bgr[:, :, ::-1]; H, W = rgb.shape[:2]
        verts, joints = posed(model, params, fr)
        I_map, U_map, V_map, mask = rasterize_iuv(verts, cam, uvmap, H, W)
        uv, _ = project(verts, cam); juv, _ = project(joints, cam)
        # 转正 M + 裁剪 bbox
        dx, dy = juv[12, 0] - juv[0, 0], juv[12, 1] - juv[0, 1]
        ang = np.degrees(np.arctan2(dx, -dy)); M = cv2.getRotationMatrix2D((W/2, H/2), ang, 1.0)
        warp = lambda im, ip=cv2.INTER_LINEAR: cv2.warpAffine(im, M, (W, H), flags=ip)
        ruv = (M @ np.c_[uv, np.ones(len(uv))].T).T; g = np.isfinite(ruv).all(1)
        x0, y0 = ruv[g].min(0); x1, y1 = ruv[g].max(0); mx, my = (x1-x0)*.15, (y1-y0)*.12
        x0, y0, x1, y1 = int(max(0, x0-mx)), int(max(0, y0-my)), int(min(W, x1+mx)), int(min(H, y1+my))
        cano = warp(rgb)[y0:y1, x0:x1]
        iuv_full = np.dstack([I_map, U_map, V_map])
        colr = (plt.cm.tab20(I_map / 24.0)[:, :, :3] * 255).astype(np.uint8)
        iuv_ov = rgb.copy(); iuv_ov[mask] = (0.45*iuv_ov[mask] + 0.55*colr[mask]).astype(np.uint8)
        cano_iuv = warp(iuv_ov)[y0:y1, x0:x1]
        # 规范坐标系下的点
        pts = {}
        for nm, vi in ausc[region].items():
            v = M @ np.array([uv[vi, 0], uv[vi, 1], 1.0]); pts[nm] = (v[0]-x0, v[1]-y0)
        col = (0, 255, 255) if region == "front" else (255, 150, 0)
        def mark(im, pd):
            im = np.ascontiguousarray(im.copy())
            for nm, (px, py) in pd.items():
                if 0 <= px < im.shape[1] and 0 <= py < im.shape[0]:
                    cv2.drawMarker(im, (int(px), int(py)), col, cv2.MARKER_TILTED_CROSS, 18, 3)
            return im
        d = os.path.join(args.out, f"{sid}_{region}"); os.makedirs(d, exist_ok=True)
        def save(name, im): cv2.imwrite(os.path.join(d, name), im[:, :, ::-1])
        rng = np.random.RandomState(7)
        outs = [("0_original.jpg", rgb),
                ("1_canonical.jpg", cano),
                ("2_iuv_label.jpg", cano_iuv),
                ("3_points.jpg", mark(cano, pts))]
        samp = {"image": cano, "points": pts}
        a1 = A.augment_for_view(samp, "thirdperson", target=(384, 384), rng=rng)
        a2 = A.augment_for_view(samp, "thirdperson", target=(384, 384), rng=rng)
        outs += [("4_aug_view1.jpg", mark(a1["image"], {k: v["xy"] for k, v in a1["points"].items() if v["visible"]})),
                 ("5_aug_view2.jpg", mark(a2["image"], {k: v["xy"] for k, v in a2["points"].items() if v["visible"]}))]
        tgt = pts.get("mitral", pts.get("L_mid", list(pts.values())[0]))
        outs += [("6_occlusion1.jpg", mark(A.occlude_arm(cano, tgt, np.random.RandomState(1)), pts)),
                 ("7_occlusion2.jpg", mark(A.occlude_arm(cano, list(pts.values())[0], np.random.RandomState(5)), pts))]
        ac = A.augment_for_view(samp, "arm", target=(384, 384), rng=np.random.RandomState(3))
        outs += [("8_arm_closeup.jpg", mark(ac["image"], {k: v["xy"] for k, v in ac["points"].items() if v["visible"]}))]
        for name, im in outs:
            save(name, im)
        # montage
        fig, ax = plt.subplots(3, 3, figsize=(13, 12))
        for a, (name, im) in zip(ax.ravel(), outs):
            a.imshow(im); a.set_title(name, fontsize=10); a.axis("off")
        ax.ravel()[-1].axis("off")
        plt.suptitle(f"{sid} [{region}]  1 original -> {len(outs)-1} processed", fontsize=12)
        plt.tight_layout(); plt.savefig(os.path.join(d, "_montage.png"), dpi=85, bbox_inches="tight"); plt.close()
        print(f"{sid} [{region}]: 1 original + {len(outs)-1} processed -> {d}")
    print("done ->", args.out)


if __name__ == "__main__":
    main()
