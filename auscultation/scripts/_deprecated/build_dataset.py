#!/usr/bin/env python3
"""
统一数据处理流水线(质检通过版) —— 直接可训练的数据集(含负样本)。

每序列产出:
  - 正样本: 前胸(iPhone, 选最正front帧, 4心点) 或 后背(Kinect后向, 8肺点)
  - 负样本(同批): 背景crop / 非躯干(四肢)crop / 侧身帧 → region=none, is_valid=false, 不放点

导出:
  <out>/img/<id>.jpg, <out>/iuv/<id>.png (R=part0..24,G=U*255,B=V*255), <out>/manifest.jsonl
manifest 每行(见 docs/model_architecture.md):
  {id,image,iuv,region(front|back|none),is_valid,side_conf,points:{name:{xy,visible}}}

⚠️ U,V 为每面均值(粗); 全量精确连续UV用 GPU+pytorch3d。
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))


def posed(model, params, fr):
    import torch
    with torch.no_grad():
        o = model(betas=torch.tensor(params["betas"][fr:fr+1], dtype=torch.float32),
                  body_pose=torch.tensor(params["body_pose"][fr:fr+1], dtype=torch.float32),
                  global_orient=torch.tensor(params["global_orient"][fr:fr+1], dtype=torch.float32),
                  transl=torch.tensor(params["transl"][fr:fr+1], dtype=torch.float32))
    return o.vertices[0].numpy(), o.joints[0].numpy()


def side_conf(part_crop, region):
    f = int((part_crop == 2).sum()); b = int((part_crop == 1).sum())
    if f + b == 0:
        return 0.0
    return round((f if region == "front" else b) / (f + b), 3)


def write(out, sid_id, img, iuv):
    import cv2
    cv2.imwrite(os.path.join(out, "img", sid_id + ".jpg"), img[:, :, ::-1])
    cv2.imwrite(os.path.join(out, "iuv", sid_id + ".png"), iuv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/humman")
    ap.add_argument("--smpl_model", default="../data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="../data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--out", default="../data/dataset_sample")
    ap.add_argument("--n_front", type=int, default=4); ap.add_argument("--n_back", type=int, default=4)
    ap.add_argument("--neg_ratio", type=float, default=0.25)
    ap.add_argument("--scan", type=int, default=200)
    args = ap.parse_args()
    import smplx, cv2
    from render_iuv import load_uv_mapping, rasterize_iuv, project
    from humman_label import define_ausc_vertices
    from frame_filter import front_core_indices, classify_frame
    from qc_back import find_kinect_videos
    model = smplx.SMPL(model_path=args.smpl_model, gender="neutral", batch_size=1)
    faces = model.faces; core = front_core_indices(model)
    uvmap = load_uv_mapping(args.uv); ausc = define_ausc_vertices(model)
    for d in ("img", "iuv"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    rng = np.random.RandomState(0)
    man = open(os.path.join(args.out, "manifest.jsonl"), "w")
    recs = []

    def render_view(params, fr, cam, region):
        """渲染一帧某相机: 返回 (img_u, iuv_u, pts_canonical, crop_bbox, juv_rot, full_iuv_u)。转正。"""
        verts, joints = posed(model, params, fr)
        capH = None
        I_map, U_map, V_map, mask = rasterize_iuv(verts, cam, uvmap, *cam["_HW"])
        uv, _ = project(verts, cam); juv, _ = project(joints, cam)
        H, W = cam["_HW"]
        dx, dy = juv[12, 0] - juv[0, 0], juv[12, 1] - juv[0, 1]
        ang = np.degrees(np.arctan2(dx, -dy)); M = cv2.getRotationMatrix2D((W/2, H/2), ang, 1.0)
        iuv = np.dstack([I_map, (U_map*255).astype(np.uint8), (V_map*255).astype(np.uint8)])
        return verts, joints, uv, juv, M, iuv, (H, W)

    seqs = sorted(glob.glob(os.path.join(args.root, "smpl_params", "*")))[:args.scan]
    nf = nb = nneg = 0
    neg_kinds = ["bg", "limb", "side"]
    for sq in seqs:
        sid = os.path.basename(sq)
        params = dict(np.load(os.path.join(sq, "smpl_params.npz")))
        cams = json.load(open(os.path.join(args.root, "point_cameras", sid, "cameras.json")))
        nfp = params["betas"].shape[0]
        # 选定帧/相机/区域
        region = None; cam = None; vid = None; fr = None; side_reject_fr = None
        if nf < args.n_front and "iphone" in cams:
            R = np.asarray(cams["iphone"]["R"]); best = (-1, -1)
            for f in range(0, nfp, max(1, nfp // 10)):
                v, jj = posed(model, params, f); lab, z = classify_frame(v, faces, R, core, joints=jj)
                if lab == "front" and abs(z) > best[0]:
                    best = (abs(z), f)
                elif lab == "reject" and side_reject_fr is None:
                    side_reject_fr = f
            mp4 = glob.glob(os.path.join(args.root, "iphone_color", sid, "**", "*.mp4"), recursive=True)
            if best[1] >= 0 and best[0] > 0.6 and mp4:
                region, cam, vid, fr = "front", cams["iphone"], mp4[0], best[1]
        if region is None and nb < args.n_back:
            vids = find_kinect_videos(args.root, sid)
            if vids:
                f0 = nfp // 2; v, jj = posed(model, params, f0); best = (-1, None)
                for ck in vids:
                    if ck in cams:
                        lab, z = classify_frame(v, faces, np.asarray(cams[ck]["R"]), core, joints=jj)
                        if lab == "back" and z > best[0]:
                            best = (z, ck)
                if best[1]:
                    region, cam, vid, fr = "back", cams[best[1]], vids[best[1]], f0
        if region is None:
            continue
        cap = cv2.VideoCapture(vid); nfk = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ff = min(fr, nfk - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, ff); ok, bgr = cap.read(); cap.release()
        if not ok:
            continue
        rgb = bgr[:, :, ::-1]; H, W = rgb.shape[:2]; cam["_HW"] = (H, W)
        verts, joints, uv, juv, M, iuv, _ = render_view(params, ff, cam, region)
        warp = lambda im, ip: cv2.warpAffine(im, M, (W, H), flags=ip)
        img_u = warp(rgb, cv2.INTER_LINEAR); iuv_u = warp(iuv, cv2.INTER_NEAREST)
        ruv = (M @ np.c_[uv, np.ones(len(uv))].T).T; g = np.isfinite(ruv).all(1)
        x0, y0 = ruv[g].min(0); x1, y1 = ruv[g].max(0); mx, my = (x1-x0)*.15, (y1-y0)*.12
        bx0, by0 = int(max(0, x0-mx)), int(max(0, y0-my)); bx1, by1 = int(min(W, x1+mx)), int(min(H, y1+my))
        img_c = img_u[by0:by1, bx0:bx1]; iuv_c = iuv_u[by0:by1, bx0:bx1]
        # 正样本点
        pts = {}
        for nm, vi in ausc[region].items():
            v3 = M @ np.array([uv[vi, 0], uv[vi, 1], 1.0]); px, py = v3[0]-bx0, v3[1]-by0
            ch, cw = img_c.shape[:2]
            pts[nm] = {"xy": [round(float(px), 1), round(float(py), 1)], "visible": bool(0 <= px < cw and 0 <= py < ch)}
        sid_id = f"{sid}_{region}_f{ff}"; write(args.out, sid_id, img_c, iuv_c)
        recs.append({"id": sid_id, "image": f"img/{sid_id}.jpg", "iuv": f"iuv/{sid_id}.png",
                     "region": region, "is_valid": True, "side_conf": side_conf(iuv_c[:, :, 0], region), "points": pts})
        nf += region == "front"; nb += region == "back"

        # ---- 负样本(同帧派生, 每3个正样本出1个, ~25%) ----
        if (nf + nb) % 3 == 0:
            kind = neg_kinds[(nf + nb) // 3 % 3]; neg = None
            if kind == "limb":                            # 裁到脚踝/小腿(非躯干)
                aj = M @ np.array([juv[7, 0], juv[7, 1], 1.0])   # 左脚踝
                s = max(20, (by1 - by0) // 4); cx, cy = int(aj[0]), int(aj[1])
                sub_i = iuv_u[max(0, cy-s):cy+s, max(0, cx-s):cx+s]
                if sub_i.size and (sub_i[:, :, 0] == 0).mean() < 0.95:  # 确有肢体(非纯背景)
                    neg = (img_u[max(0, cy-s):cy+s, max(0, cx-s):cx+s], sub_i)
            elif kind == "side" and side_reject_fr is not None and region == "front":
                cap = cv2.VideoCapture(vid); cap.set(cv2.CAP_PROP_POS_FRAMES, min(side_reject_fr, nfk-1))
                ok2, b2 = cap.read(); cap.release()
                if ok2:
                    neg = (b2[:, :, ::-1][by0:by1, bx0:bx1], np.zeros((by1-by0, bx1-bx0, 3), np.uint8))
            if neg is None:                               # 默认/兜底: 背景角落(无人)
                s = min(H, W) // 3
                for yy, xx in [(0, 0), (0, W-s), (H-s, 0), (H-s, W-s)]:
                    bi = iuv_u[yy:yy+s, xx:xx+s]
                    if (bi[:, :, 0] > 0).mean() < 0.05:   # 基本无身体
                        neg = (img_u[yy:yy+s, xx:xx+s], bi); kind = "bg"; break
            if neg and neg[0].size > 0:
                nid = f"{sid}_neg_{kind}"; write(args.out, nid, neg[0], neg[1])
                recs.append({"id": nid, "image": f"img/{nid}.jpg", "iuv": f"iuv/{nid}.png",
                             "region": "none", "is_valid": False, "side_conf": 0.0, "points": {}})
                nneg += 1
        if nf >= args.n_front and nb >= args.n_back:
            break
    for r in recs:
        man.write(json.dumps(r) + "\n")
    man.close()
    print(f"exported front={nf} back={nb} neg={nneg} (neg比例={nneg/max(1,len(recs)):.0%}) -> {args.out}/manifest.jsonl")


if __name__ == "__main__":
    main()
