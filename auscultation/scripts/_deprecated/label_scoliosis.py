#!/usr/bin/env python3
"""
对真实裸背图(scoliosis)用 DensePose 预测稠密IUV, 读出我们定义的 8 个后背肺点, 检查精度。
桥接: humman_label 的后背肺点(SMPL顶点) → 各自的 DensePose (I,U,V) → 在预测IUV上 map_point。
"""
import glob, os, sys, importlib.machinery
import numpy as np
import scipy.io as sio
# av 桩(densepose 的 video 模块导入 av; 本机 av dylib 坏, 不跑视频→桩掉)
if "av" not in sys.modules:
    try:
        import av  # noqa
    except Exception:
        from unittest.mock import MagicMock
        _av = MagicMock(); _av.__spec__ = importlib.machinery.ModuleSpec("av", loader=None)
        sys.modules["av"] = _av
sys.path.insert(0, "/tmp/d2repo/projects/DensePose")
sys.path.insert(0, os.path.dirname(__file__))

D2_CFG = "/tmp/d2repo/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
WEIGHTS = "../data/densepose_model/model_final_162be9.pkl"


def smpl_to_iuv_lookup(mat):
    d = sio.loadmat(mat)
    v2s = d["All_vertices"].reshape(-1).astype(int) - 1     # DPvert→SMPL(0based)
    U = d["All_U_norm"].reshape(-1); V = d["All_V_norm"].reshape(-1)
    faces = d["All_Faces"].astype(int) - 1; fpart = d["All_FaceIndices"].reshape(-1).astype(int)
    dp_part = np.zeros(len(v2s), int)
    for fi in range(len(faces)):
        for k in faces[fi]:
            dp_part[k] = fpart[fi]
    # smpl顶点 → 候选 DP顶点列表
    from collections import defaultdict
    s2dp = defaultdict(list)
    for j, s in enumerate(v2s):
        s2dp[s].append(j)
    def lookup(smpl_idx, prefer_part=None):
        cand = s2dp.get(smpl_idx, [])
        if not cand:
            return None
        if prefer_part is not None:
            pc = [j for j in cand if dp_part[j] == prefer_part]
            if pc:
                cand = pc
        j = cand[0]
        return (int(dp_part[j]), float(U[j]), float(V[j]))
    return lookup


def main():
    import smplx, cv2, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from humman_label import define_ausc_vertices
    from ausc_points import BACK_PART
    from densepose_predictor import DensePosePredictor, map_point
    model = smplx.SMPL(model_path="../data/smpl/SMPL_NEUTRAL.pkl", gender="neutral", batch_size=1)
    ausc = define_ausc_vertices(model)
    lookup = smpl_to_iuv_lookup("../data/densepose_coco/uv_data/UV_Processed.mat")
    # 8 后背肺点的 (I,U,V)
    targets = {}
    for nm, vi in ausc["back"].items():
        iuv = lookup(vi, prefer_part=BACK_PART)
        if iuv:
            targets[nm] = iuv
    print("后背点 (I,U,V):", {k: tuple(round(x, 2) for x in v) for k, v in targets.items()})

    pred = DensePosePredictor(D2_CFG, WEIGHTS, device="cpu", score_thresh=0.7)
    imgs = sorted(glob.glob("../data/scoliosis_back/train/*.JPG") + glob.glob("../data/scoliosis_back/train/*.jpg"))[:6]
    os.makedirs("../data/_viz_humman/scoliosis", exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 11))
    for a, p in zip(ax.ravel(), imgs):
        img = cv2.imread(p)
        dp = pred.predict_largest(img)
        a.imshow(img[:, :, ::-1]); a.axis("off"); a.set_title(os.path.basename(p)[:18], fontsize=8)
        if dp is None:
            a.set_title("no person", fontsize=8); continue
        for nm, t in targets.items():
            r = map_point(dp, t)
            if r:
                a.scatter([r[0]], [r[1]], c="orange", s=80, marker="x", lw=2.5)
    plt.suptitle("真实裸背 DensePose → 我们的8后背肺点 (精度检查)", fontsize=13)
    plt.tight_layout(); plt.savefig("../data/_viz_humman/scoliosis/_LABELED.png", dpi=85, bbox_inches="tight"); plt.close()
    print("saved _LABELED.png")


if __name__ == "__main__":
    main()
