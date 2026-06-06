#!/usr/bin/env python3
"""
HuMMan 自动标注流水线: SMPL 参数 + 相机 → 听诊点 2D 投影（稠密精确, 不塌缩）。

依赖 SMPL 模型文件(免费注册 smpl.is.tue.mpg.de, 与 AGORA 同机构):
  放到 data/smpl/SMPL_NEUTRAL.pkl  (或传 --smpl_model)

流程:
  1. SMPL(betas,pose,transl) → 6890 顶点
  2. 听诊点 = 模板上按【解剖比例】自动选的固定顶点 (define_ausc_vertices)
  3. 用序列相机 K/R/T 投影顶点 → 精确 2D 点
  4. 叠加到 HuMMan 彩色图预览

用法:
  python humman_label.py --root ../data/humman --smpl_model ../data/smpl/SMPL_NEUTRAL.pkl \
      --color iphone_color --out ../data/_viz_humman --n 8
"""
import argparse, os, glob, json
import numpy as np


def load_smpl(model_path):
    try:
        import smplx, torch
    except ImportError as e:
        raise ImportError(f"需要 smplx+torch: {e}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SMPL 模型缺失: {model_path}\n"
            "→ 注册 https://smpl.is.tue.mpg.de 下载 SMPL_NEUTRAL.pkl 放到该路径。")
    import smplx
    return smplx.SMPL(model_path=model_path, gender="neutral", batch_size=1)


def define_ausc_vertices(model):
    """从 SMPL 模板(T-pose,中性体型)按解剖比例选听诊点顶点索引。
    返回 {name: vertex_idx}。前胸=心脏点(法向 +z), 后背=肺点(法向 -z)。
    标定(中性模板): +x=患者左, -x=患者右; 躯干核心 |x|<0.16(排除T-pose手臂)。"""
    import torch
    with torch.no_grad():
        V = model(betas=torch.zeros(1, 10), body_pose=torch.zeros(1, 69),
                  global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3)).vertices[0].numpy()
    N = _vertex_normals(V, model.faces)
    x, y, z = V[:, 0], V[:, 1], V[:, 2]               # x=左右(+左), y=上下(+上), z=前后(+前)

    # 用绝对解剖坐标(由模板关节标定): 锁骨 collar y≈0.19, 颈根 neck y≈0.29,
    # 脊柱 spine1 y≈-0.11, 骨盆 y≈-0.22; 锁骨中线 x≈±0.08, 胸骨旁 x≈±0.04。
    def pick(tx, ty, front):
        face = N[:, 2] > 0.2 if front else N[:, 2] < -0.2
        core = face & (np.abs(x) < 0.16) & (y > -0.25) & (y < 0.19)   # 躯干核心(锁骨以下~骨盆以上)
        cand = np.where(core)[0]
        d = (y[cand] - ty) ** 2 + (x[cand] - tx) ** 2
        return int(cand[np.argmin(d)])

    def pick_back_pair(off, ty, nz_max):
        """Pick symmetric posterior lung vertices for one level.

        The lateral lower lung field has side-facing normals, so a strict
        N_z < -0.2 posterior-only filter clamps points toward the spine.  Pick
        the two sides jointly with a relaxed per-level normal threshold, then
        penalize left/right height and mirror asymmetry.
        """
        core = (
            (np.abs(x) < 0.18)
            & (y > -0.25)
            & (y < 0.19)
            & (z < 0.02)
            & (N[:, 2] < nz_max)
        )
        r_cand = np.where(core & (x < 0))[0]
        l_cand = np.where(core & (x > 0))[0]
        if len(r_cand) == 0 or len(l_cand) == 0:
            raise RuntimeError("no posterior lung candidates for symmetric pair")
        r_cand = r_cand[np.argsort((x[r_cand] + off) ** 2 + (y[r_cand] - ty) ** 2)[:220]]
        l_cand = l_cand[np.argsort((x[l_cand] - off) ** 2 + (y[l_cand] - ty) ** 2)[:220]]
        best = None
        for ri in r_cand:
            for li in l_cand:
                score = (
                    (x[ri] + off) ** 2
                    + (x[li] - off) ** 2
                    + 2.0 * ((y[ri] - ty) ** 2 + (y[li] - ty) ** 2)
                    + 8.0 * (y[ri] - y[li]) ** 2
                    + 5.0 * (x[ri] + x[li]) ** 2
                )
                if best is None or score < best[0]:
                    best = (score, int(ri), int(li))
        return best[1], best[2]

    # 【前胸心脏 5 区】Bates' 标准, 助记 APETM(含 Erb 第三听诊区)。x<0=患者右,x>0=患者左。
    # dict 顺序 = 听诊顺序(心底→心尖)。任务→点集路由见 ausc/tasks.py。
    # 按标准 APETM 图(heart_auscul_points.png)沿真实肋间高度铺开: 锁骨 y≈0.19, 每肋间≈0.035,
    # 2nd→5th ICS 纵跨~0.12; 原定义压在上中胸 8cm 内、心尖偏高 → 已拉开纵向、心尖下移更外侧。
    cardiac = {
        "aortic":    pick(-0.045, 0.135, True),  # 主动脉瓣区: 第2肋间 患者右胸骨旁
        "pulmonary": pick(+0.045, 0.135, True),  # 肺动脉瓣区: 第2肋间 患者左胸骨旁
        "erb":       pick(+0.040, 0.100, True),  # Erb 第三听诊区: 第3肋间 左胸骨缘(S1+S2均清)
        "tricuspid": pick(+0.040, 0.055, True),  # 三尖瓣区: 第4肋间 左下胸骨缘
        "mitral":    pick(+0.095, 0.015, True),  # 二尖瓣区/心尖: 第5肋间 左锁骨中线(↓更低更外, 对齐标准心尖)
    }
    # 【后背肺点】权威读出地址 = 模块常量 LUNG_ADDR_V6A(直接 (part,U,V), 见下方)。
    # 这里返回的 back 顶点仅供数据生成器(humman/scoliosis/frame_filter)初值投影,
    # 会被 scripts/realign_lung_points(按 LUNG_ADDR_V6A 从稠密IUV读出)覆盖。
    # 注: 后背 SMPL 顶点的 (U,V) 太稀疏, 无法表达 V6A 连续位置, 故读出走 LUNG_ADDR_V6A 而非顶点。
    lung = {}
    for i, (ty, off, nz_max) in enumerate([(0.145, 0.085, 0.05), (0.065, 0.050, 0.08),
                                           (-0.005, 0.070, 0.10), (-0.075, 0.120, 0.12)]):
        ri, li = pick_back_pair(off, ty, nz_max)
        lung[f"R{i+1}"] = ri; lung[f"L{i+1}"] = li
    return {"front": cardiac, "back": lung}


# DensePose 躯干块: part2=前胸, part1=后背 (与 render_iuv 的 I 通道一致)
FRONT_TORSO_PART, BACK_TORSO_PART = 2, 1

# 【后背肺点标准 V6A】= (part=1, U, V) 直接地址, 5 层双侧, R=患者右(低V)/L=患者左(高V)。
# 由临床医生在干净参考图上手标 → 取其稠密 IUV 的 (U,V) → 对称化得到(见 docs)。
# U=垂直(0顶1底, 围肩胛骨/上中背), V=水平(0.5脊柱). 单一权威, 改这里即改全管线读点。
_V6A_U = (0.200, 0.300, 0.390, 0.470, 0.540)
_V6A_DV = (0.212, 0.135, 0.096, 0.159, 0.216)
LUNG_ADDR_V6A = {}
for _i in range(5):
    LUNG_ADDR_V6A[f"R{_i+1}"] = (BACK_TORSO_PART, _V6A_U[_i], 0.5 - _V6A_DV[_i])  # 患者右=低V
    LUNG_ADDR_V6A[f"L{_i+1}"] = (BACK_TORSO_PART, _V6A_U[_i], 0.5 + _V6A_DV[_i])  # 患者左=高V


def infer_side(part_map):
    """推理时前胸/后背判别: 稠密IUV的躯干块 part2(前)vs part1(后), 谁的像素多算谁。
    → 决定读 4 心脏点(front) 还是 8 肺点(back)。实测极悬殊(前87209:770/后544:45278)。"""
    front = int((np.asarray(part_map) == FRONT_TORSO_PART).sum())
    back = int((np.asarray(part_map) == BACK_TORSO_PART).sum())
    return "front" if front >= back else "back"


def _vertex_normals(V, faces):
    N = np.zeros_like(V)
    tris = V[faces]
    fn = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0])
    for i in range(3):
        np.add.at(N, faces[:, i], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True); n[n == 0] = 1
    return N / n


def smpl_vertices(model, params, frame):
    import torch
    with torch.no_grad():
        out = model(betas=torch.tensor(params["betas"][frame:frame+1], dtype=torch.float32),
                    body_pose=torch.tensor(params["body_pose"][frame:frame+1], dtype=torch.float32),
                    global_orient=torch.tensor(params["global_orient"][frame:frame+1], dtype=torch.float32),
                    transl=torch.tensor(params["transl"][frame:frame+1], dtype=torch.float32))
    return out.vertices[0].numpy()


def project(verts, cam):
    K = np.asarray(cam["K"]); R = np.asarray(cam["R"])
    T = np.asarray(cam.get("T", cam.get("t"))).reshape(3)
    P = (R @ verts.T).T + T
    uv = (K @ P.T).T
    return uv[:, :2] / uv[:, 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--smpl_model", default="../data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--color", default="iphone_color")
    ap.add_argument("--cam_key", default=None, help="相机名; 默认自动挑 iphone/第一个")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="iPhone mp4 帧可能旋转; 投影对不上时调这个")
    ap.add_argument("--out", default="../data/_viz_humman"); ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    model = load_smpl(args.smpl_model)
    ausc = define_ausc_vertices(model)
    print("听诊点顶点:", {r: list(v.values()) for r, v in ausc.items()})

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    os.makedirs(args.out, exist_ok=True)
    seqs = sorted(glob.glob(os.path.join(args.root, "smpl_params", "*")))
    done = 0
    for sq in seqs:
        if done >= args.n:
            break
        sid = os.path.basename(sq)
        params = dict(np.load(os.path.join(sq, "smpl_params.npz")))
        cams = json.load(open(os.path.join(args.root, "point_cameras", sid, "cameras.json")))
        ckey = args.cam_key or next((k for k in cams if "iphone" in k.lower()), list(cams)[0])
        # HuMMan iphone color 存为 mp4；读中间帧
        import cv2
        mp4 = glob.glob(os.path.join(args.root, args.color, sid, "**", "*.mp4"), recursive=True)
        if not mp4:
            continue
        cap = cv2.VideoCapture(mp4[0]); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fr = nf // 2; cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, bgr = cap.read(); cap.release()
        if not ok:
            continue
        im = bgr[:, :, ::-1]
        if args.rotate:
            im = np.rot90(im, args.rotate // 90)
        nfp = params["betas"].shape[0]
        verts = smpl_vertices(model, params, min(int(fr / max(1, nf) * nfp), nfp - 1))
        uv = project(verts, cams[ckey])
        plt.figure(figsize=(6, 8)); plt.imshow(im); plt.axis("off"); plt.title(f"{sid} [{ckey}]", fontsize=8)
        for region, col in [("front", "cyan"), ("back", "orange")]:
            for name, vi in ausc[region].items():
                px, py = uv[vi]
                plt.scatter([px], [py], c=col, s=90, marker="x", linewidths=2)
        plt.savefig(os.path.join(args.out, f"{sid}.png"), dpi=95, bbox_inches="tight"); plt.close()
        print("viz", sid); done += 1
    print(f"done {done} -> {args.out}")


if __name__ == "__main__":
    main()
