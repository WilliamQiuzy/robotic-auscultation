#!/usr/bin/env python3
"""
抽查: 生成 N 个样本(前胸4点/后背8点混合, 跨受试者)的单图 IUV+点 叠加, 供人工逐张查标注错误。
前胸用 iPhone(自动选最正front帧); 后背用已解压的 Kinect 后向视频。
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
    ap.add_argument("--out", default="../data/_viz_humman/spotcheck")
    ap.add_argument("--n_front", type=int, default=5); ap.add_argument("--n_back", type=int, default=5)
    ap.add_argument("--foff", type=int, default=0, help="前胸受试者取样偏移(换一批人)")
    ap.add_argument("--targets", default="_back_targets.json", help="后背targets文件名")
    args = ap.parse_args()
    import smplx, cv2, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from render_iuv import load_uv_mapping, rasterize_iuv, project
    from humman_label import define_ausc_vertices
    from frame_filter import front_core_indices, classify_frame
    from humman_io import find_kinect_videos
    model = smplx.SMPL(model_path="../data/smpl/SMPL_NEUTRAL.pkl", gender="neutral", batch_size=1)
    faces = model.faces; core = front_core_indices(model)
    uvmap = load_uv_mapping("../data/densepose_coco/uv_data/UV_Processed.mat")
    ausc = define_ausc_vertices(model)
    os.makedirs(args.out, exist_ok=True)

    allseq = sorted(glob.glob(os.path.join(args.root, "smpl_params", "*")))
    # 前胸: 跨受试者按 subject 取样
    bysub = {}
    for s in allseq:
        bysub.setdefault(os.path.basename(s).split("_")[0], []).append(s)
    front_subs = list(bysub)[args.foff::3]                 # 充足候选(动作帧可能没正脸,需冗余)
    back_targets = json.load(open(os.path.join(args.root, args.targets)))
    jobs = [(os.path.basename(bysub[s][0]), "front", None) for s in front_subs]
    jobs += [(t["sid"], "back", t["cam"]) for t in back_targets]

    panels = []; cnt = {"front": 0, "back": 0}
    for sid, region, ckey in jobs:
        if cnt[region] >= (args.n_front if region == "front" else args.n_back):
            continue
        sq = os.path.join(args.root, "smpl_params", sid)
        if not os.path.exists(sq):
            continue
        params = dict(np.load(os.path.join(sq, "smpl_params.npz")))
        cams = json.load(open(os.path.join(args.root, "point_cameras", sid, "cameras.json")))
        nfp = params["betas"].shape[0]
        if region == "front":
            R = np.asarray(cams["iphone"]["R"]); best = (-1, -1)
            for fr in range(0, nfp, max(1, nfp // 10)):
                v, jj = posed(model, params, fr); lab, z = classify_frame(v, faces, R, core, joints=jj)
                if lab == "front" and abs(z) > best[0]:
                    best = (abs(z), fr)
            if best[1] < 0:
                continue
            fr = best[1]; cam = cams["iphone"]
            vid = glob.glob(os.path.join(args.root, "iphone_color", sid, "**", "*.mp4"), recursive=True)
        else:
            vids = find_kinect_videos(args.root, sid)
            if ckey not in vids:
                continue
            fr = nfp // 2; cam = cams[ckey]; vid = [vids[ckey]]
        if not vid:
            continue
        cap = cv2.VideoCapture(vid[0]); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ff = min(fr, nf - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, ff); ok, bgr = cap.read(); cap.release()
        if not ok:
            continue
        verts, joints = posed(model, params, ff); H, W = bgr.shape[:2]
        I_map, U_map, V_map, mask = rasterize_iuv(verts, cam, uvmap, H, W)
        uv, _ = project(verts, cam); juv, _ = project(joints, cam)
        rgb = bgr[:, :, ::-1].copy()
        colr = (plt.cm.tab20(I_map / 24.0)[:, :, :3] * 255).astype(np.uint8)
        rgb[mask] = (0.5 * rgb[mask] + 0.5 * colr[mask]).astype(np.uint8)
        c = (0, 255, 255) if region == "front" else (255, 150, 0)
        for nm, vi in ausc[region].items():
            p = uv[vi]
            if np.all(np.isfinite(p)):
                cv2.drawMarker(rgb, (int(p[0]), int(p[1])), c, cv2.MARKER_TILTED_CROSS, 22, 4)
        dx, dy = juv[12, 0] - juv[0, 0], juv[12, 1] - juv[0, 1]; ang = np.degrees(np.arctan2(dx, -dy))
        M = cv2.getRotationMatrix2D((W/2, H/2), ang, 1.0); img = cv2.warpAffine(rgb, M, (W, H))
        ruv = (M @ np.c_[uv, np.ones(len(uv))].T).T; g = np.isfinite(ruv).all(1)
        x0, y0 = ruv[g].min(0); x1, y1 = ruv[g].max(0); mx, my = (x1-x0)*.18, (y1-y0)*.12
        x0, y0, x1, y1 = int(max(0, x0-mx)), int(max(0, y0-my)), int(min(W, x1+mx)), int(min(H, y1+my))
        crop = img[y0:y1, x0:x1]
        cv2.imwrite(os.path.join(args.out, f"{region}_{sid}.png"), crop[:, :, ::-1])
        panels.append((f"{region} {sid}\n{len(ausc[region])} pts", crop))
        cnt[region] += 1
        print(f"spotcheck {region} {sid} ({len(ausc[region])} pts)")
    # contact sheet
    n = len(panels); cols = 5; rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    for a, (t, im) in zip(ax.ravel(), panels):
        a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
    for a in ax.ravel()[n:]:
        a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "_contact_sheet.png"), dpi=70, bbox_inches="tight"); plt.close()
    print(f"done {n} -> {args.out}")


if __name__ == "__main__":
    main()
