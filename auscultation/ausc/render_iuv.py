#!/usr/bin/env python3
"""
稠密标签生成器: HuMMan SMPL mesh → 逐像素 DensePose (I,U,V) 图。

这是【稠密感知架构】的训练标签:模型学"图像→逐像素身体表面坐标(I,U,V)"。
听诊点读出: 推理时预测稠密 IUV, 找 (I,U,V) 匹配目标顶点的像素 → 2D 点(+深度→3D)。
抗噪/抗遮挡/抗近景, 远优于稀疏关键点。

依赖: uv_data/UV_Processed.mat (SMPL顶点→IUV映射, 已有) + SMPL模型 + HuMMan序列。
本脚本用 CPU painter 算法(cv2.fillConvexPoly)出 part-segmentation 可视化;
全量/连续UV标签建议 GPU 服务器用 pytorch3d 光栅化(更快更准)。
"""
import argparse, glob, json, os
import numpy as np
import scipy.io as sio


def load_uv_mapping(mat_path):
    d = sio.loadmat(mat_path)
    v2s = d["All_vertices"].reshape(-1).astype(int) - 1     # (7829,) DPvert→SMPL idx(0based)
    U = d["All_U_norm"].reshape(-1)                          # (7829,)
    V = d["All_V_norm"].reshape(-1)
    faces = d["All_Faces"].astype(int) - 1                   # (13774,3) 0based into DPverts
    fpart = d["All_FaceIndices"].reshape(-1).astype(int)     # (13774,) part 1..24
    return v2s, U, V, faces, fpart


def smpl_posed(model, params, frame):
    import torch
    with torch.no_grad():
        o = model(betas=torch.tensor(params["betas"][frame:frame+1], dtype=torch.float32),
                  body_pose=torch.tensor(params["body_pose"][frame:frame+1], dtype=torch.float32),
                  global_orient=torch.tensor(params["global_orient"][frame:frame+1], dtype=torch.float32),
                  transl=torch.tensor(params["transl"][frame:frame+1], dtype=torch.float32))
    return o.vertices[0].numpy()


def project(verts, cam):
    K = np.asarray(cam["K"]); R = np.asarray(cam["R"]); T = np.asarray(cam.get("T", cam.get("t"))).reshape(3)
    P = (R @ verts.T).T + T
    uv = (K @ P.T).T
    return uv[:, :2] / uv[:, 2:3], P[:, 2]                   # 2D, 深度


def rasterize_iuv(smpl_verts, cam, uvmap, H, W):
    """返回 (I_map HxW uint8 0..24, U_map, V_map float[0,1], mask). painter 算法(远→近)。
    U,V 用【逐像素重心插值】(非逐面均值): 三角面内 U,V 按重心坐标连续插值,
    使 UV→像素 的映射连续可分辨(关键: 后背单 part 内 8 个肺点全靠 UV 精度区分)。"""
    v2s, U, V, faces, fpart = uvmap
    dp_xyz = smpl_verts[v2s]                                 # (7829,3)
    uv2d, depth = project(dp_xyz, cam)                       # (7829,2),(7829,)
    fdepth = depth[faces].mean(1)
    order = np.argsort(-fdepth)                              # 远的先画(painter: 近的覆盖远的)
    I_map = np.zeros((H, W), np.uint8)
    U_map = np.zeros((H, W), np.float32); V_map = np.zeros((H, W), np.float32)
    for fi in order:
        tri = uv2d[faces[fi]]                                # (3,2)
        if np.any(~np.isfinite(tri)):
            continue
        (ax, ay), (bx, by), (cx, cy) = tri
        x0 = max(int(np.floor(min(ax, bx, cx))), 0); x1 = min(int(np.ceil(max(ax, bx, cx))), W - 1)
        y0 = max(int(np.floor(min(ay, by, cy))), 0); y1 = min(int(np.ceil(max(ay, by, cy))), H - 1)
        if x1 < x0 or y1 < y0:
            continue
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-9:
            continue
        xs = np.arange(x0, x1 + 1); ys = np.arange(y0, y1 + 1)
        gx, gy = np.meshgrid(xs, ys)                         # bbox 像素网格
        l0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / denom   # 重心坐标
        l1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / denom
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)         # 三角内
        if not inside.any():
            continue
        vi = faces[fi]; Uv = U[vi]; Vv = V[vi]
        yy = gy[inside]; xx = gx[inside]
        I_map[yy, xx] = int(fpart[fi])
        U_map[yy, xx] = (l0[inside] * Uv[0] + l1[inside] * Uv[1] + l2[inside] * Uv[2]).astype(np.float32)
        V_map[yy, xx] = (l0[inside] * Vv[0] + l1[inside] * Vv[1] + l2[inside] * Vv[2]).astype(np.float32)
    return I_map, U_map, V_map, I_map > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/humman")
    ap.add_argument("--smpl_model", default="../data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="../data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--out", default="../data/_viz_humman"); ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    import smplx, cv2, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    model = smplx.SMPL(model_path=args.smpl_model, gender="neutral", batch_size=1)
    uvmap = load_uv_mapping(args.uv)
    os.makedirs(args.out, exist_ok=True)
    seqs = sorted(glob.glob(os.path.join(args.root, "smpl_params", "*")))[:args.n]
    for sq in seqs:
        sid = os.path.basename(sq)
        params = dict(np.load(os.path.join(sq, "smpl_params.npz")))
        cams = json.load(open(os.path.join(args.root, "point_cameras", sid, "cameras.json")))
        cam = cams.get("iphone", cams[list(cams)[0]])
        mp4 = glob.glob(os.path.join(args.root, "iphone_color", sid, "**", "*.mp4"), recursive=True)
        if not mp4:
            continue
        cap = cv2.VideoCapture(mp4[0]); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fr = nf // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, bgr = cap.read(); cap.release()
        if not ok:
            continue
        H, W = bgr.shape[:2]
        verts = smpl_posed(model, params, min(fr, params["betas"].shape[0] - 1))
        I_map, U_map, V_map, mask = rasterize_iuv(verts, cam, uvmap, H, W)
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        ax[0].imshow(bgr[:, :, ::-1]); ax[0].set_title(f"{sid} image"); ax[0].axis("off")
        ov = bgr[:, :, ::-1].copy()
        col = (plt.cm.tab20(I_map / 24.0)[:, :, :3] * 255).astype(np.uint8)
        ov[mask] = (0.5 * ov[mask] + 0.5 * col[mask]).astype(np.uint8)
        ax[1].imshow(ov); ax[1].set_title("dense part-index (I) overlay"); ax[1].axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, f"_IUV_{sid}.png"), dpi=90, bbox_inches="tight"); plt.close()
        print("saved _IUV_" + sid)


if __name__ == "__main__":
    main()
