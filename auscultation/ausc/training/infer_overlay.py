#!/usr/bin/env python3
"""无 GT 推理 + 叠加可视化(部署域实测)。

对任意 RGB 图: 中心方裁 → resize res → 模型预测稠密IUV → soft-argmax 读出听诊点 → 叠加。
区分前/后: 由预测 part 像素多数决(part1=后背/part2=前胸, 见 humman_label.infer_side)。
用法: python infer_overlay.py --ckpt <pt> --imgs <dir或图列表> --region back --out <dir> [--encoder ...] [--tta 1.0,1.14,1.27]
"""
import argparse, glob, os, sys
import numpy as np, cv2, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in ("ausc", "ausc/models", "ausc/training"):
    sys.path.insert(0, os.path.join(ROOT, p))
from iuv_model import DenseIUVModel, N_PARTS                      # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD                  # noqa: E402
from eval import readout_soft, _forward_maps                     # noqa: E402
from render_iuv import load_uv_mapping                           # noqa: E402
from tasks import task_region, task_landmarks, list_tasks        # noqa: E402


def region_names(addr, region):
    """后背肺点=名字形如 L1..R4(字母+数字); 其余=前胸心点。"""
    back = [n for n in addr if len(n) >= 2 and n[0] in "LR" and n[1:].isdigit()]
    return sorted(back) if region == "back" else [n for n in addr if n not in back]


def posture_confidence(points, pred_part, res, region):
    """全自动置信度(无人在回路): 据 后背占比/肺点双侧展开/对称/纵向覆盖 给 [0,1] 分。
    用途: 多帧时自动挑最高分那帧喂机械臂; 分数作机器信号(低分→更保守/换帧)。无"叫人重摆"。
    返回 (conf, metrics, flags)。"""
    flags = []
    fg = pred_part > 0
    torso = np.isin(pred_part, [1, 2])
    back_px = int((pred_part == 1).sum()); front_px = int((pred_part == 2).sum())
    back_frac = back_px / max(1, back_px + front_px)
    fg_frac = float(fg.sum()) / (res * res)
    ys, xs = np.where(torso if torso.sum() > 50 else fg)
    tw = (xs.max() - xs.min() + 1) if len(xs) else res
    th = (ys.max() - ys.min() + 1) if len(ys) else res
    cx = (xs.min() + xs.max()) / 2 if len(xs) else res / 2
    pts = np.array(list(points.values())) if points else np.zeros((0, 2))
    xspread = float((pts[:, 0].max() - pts[:, 0].min()) / max(1, tw)) if len(pts) >= 4 else 0.0
    yspread = float((pts[:, 1].max() - pts[:, 1].min()) / max(1, th)) if len(pts) >= 4 else 0.0
    L = np.array([points[n] for n in points if n[0] == "L"]) if points else np.zeros((0, 2))
    R = np.array([points[n] for n in points if n[0] == "R"]) if points else np.zeros((0, 2))
    balanced = 1.0 if (len(L) and len(R) and (L[:, 0].mean() - cx) * (R[:, 0].mean() - cx) < 0) else 0.0

    def soft(v, lo, hi):                                  # lo→0, hi→1 线性饱和
        return float(np.clip((v - lo) / max(1e-6, hi - lo), 0, 1))
    s_side = soft(back_frac, 0.5, 0.9) if region == "back" else soft(1 - back_frac, 0.5, 0.9)
    s_size = soft(fg_frac, 0.03, 0.12)
    s_x = soft(xspread, 0.12, 0.30)
    s_y = soft(yspread, 0.10, 0.25)
    s_n = soft(len(pts), 4, 8)
    conf = float(np.clip(0.30 * s_side + 0.12 * s_size + 0.22 * s_x + 0.14 * s_y + 0.10 * s_n + 0.12 * balanced, 0, 1))
    if back_frac < 0.8 and region == "back":
        flags.append("low_back_frac")
    if xspread < 0.18:
        flags.append("points_collapsed")
    if balanced < 1:
        flags.append("LR_imbalance")
    if len(pts) < 8:
        flags.append("missing_points")
    metrics = {"conf": round(conf, 3), "back_frac": round(back_frac, 3), "fg_frac": round(fg_frac, 3),
               "xspread": round(xspread, 3), "yspread": round(yspread, 3), "n_pts": int(len(pts))}
    return conf, metrics, flags


def landmark_addresses(uv_path, smpl_path):
    from eval import landmark_addresses as la
    return la(uv_path, smpl_path)


def load_model(ckpt, encoder, device):
    ck = torch.load(ckpt, map_location=device)
    if ck.get("kind") == "lora":
        from iuv_model import apply_lora
        m = DenseIUVModel(encoder=encoder, frozen_encoder=False).to(device)
        apply_lora(m, r=ck["lora_rank"], alpha=ck["lora_alpha"])
        m.load_state_dict(ck["lora_head"], strict=False); m.to(device)
    else:
        m = DenseIUVModel(encoder=encoder, frozen_encoder=True).to(device)
        if "model" in ck:
            m.load_state_dict(ck["model"])
        else:
            m.head.load_state_dict(ck["head"])
    m.eval()
    return m


def center_square(img, up_bias=0.0):
    """中心方裁(可上偏 up_bias∈[0,1) 把人/躯干往画面中上对齐)。"""
    h, w = img.shape[:2]
    s = min(h, w)
    x0 = (w - s) // 2
    y0 = int((h - s) * (0.5 - up_bias))
    y0 = max(0, min(h - s, y0))
    return img[y0:y0 + s, x0:x0 + s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--imgs", required=True, help="目录或逗号分隔图路径")
    ap.add_argument("--region", default="back", choices=["back", "front"])
    ap.add_argument("--task", default=None, help="任务路由(优先于--region): heart→前胸5点 / lung→后背8点(见 ausc/tasks.py)")
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--tta", default=None)
    ap.add_argument("--up_bias", type=float, default=0.12, help="中心裁上偏(把坐姿躯干居中)")
    ap.add_argument("--gate", type=int, default=0, help="1=全自动置信度+多帧自动挑最优(无人在回路)")
    ap.add_argument("--conf_thr", type=float, default=0.6, help="置信度阈值(标注色; 低于→机械臂更保守/换帧)")
    ap.add_argument("--smpl", default="data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--out", default="infer_out")
    args = ap.parse_args()
    device = "cpu" if os.environ.get("FORCE_CPU") else "cuda"
    os.makedirs(args.out, exist_ok=True)
    tta = tuple(float(x) for x in args.tta.split(",")) if args.tta else None

    m = load_model(args.ckpt, args.encoder, device)
    addr, _ = landmark_addresses(args.uv, args.smpl)
    if args.task:                                          # 任务路由(单一权威 ausc/tasks.py): 任务→区域+有序点集
        region = task_region(args.task)
        names = [n for n in task_landmarks(args.task) if n in addr]
        print(f"[task] {args.task} → region={region} | 听诊点(顺序)={names}")
    else:
        region = args.region
        names = region_names(addr, region)
    res = args.res
    gy, gx = np.mgrid[0:res, 0:res].astype(np.float32)

    files = args.imgs.split(",") if "," in args.imgs or args.imgs.endswith((".jpg", ".png", ".JPG")) else sorted(glob.glob(os.path.join(args.imgs, "*")))
    ranking = []
    for f in files:
        raw = cv2.imread(f)
        if raw is None:
            print("skip", f); continue
        crop = center_square(raw[:, :, ::-1], args.up_bias)            # RGB
        im = cv2.resize(crop, (res, res), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(((im / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]).float().to(device)
        part_prob, pred_uv = _forward_maps(m, x, res, tta)             # (25,H,W),(2,H,W)
        pred_part = part_prob.argmax(0)
        # 前/后侧:part1=后背 part2=前胸 像素多数
        back_px = int((pred_part == 1).sum()); front_px = int((pred_part == 2).sum())
        side = "back" if back_px >= front_px else "front"
        ov = im.copy()
        pts = {}
        for nm in names:
            if nm not in addr:
                continue
            p = readout_soft(part_prob[addr[nm][0]], pred_uv, addr[nm], gx, gy, args.tau)
            if p is None:
                continue
            pts[nm] = (float(p[0]), float(p[1]))
            cv2.circle(ov, (int(p[0]), int(p[1])), 6, (255, 40, 40), -1)
            cv2.circle(ov, (int(p[0]), int(p[1])), 7, (255, 255, 255), 1)
        conf, metrics, flags = (None, {}, [])
        if args.gate:
            conf, metrics, flags = posture_confidence(pts, pred_part, res, region)
            # 视角与任务区域一致性校验(全自动: 不匹配→flag, 机械臂换帧/换视角)
            if (region == "back") != (side == "back"):
                flags = ["view_task_mismatch"] + flags; conf = min(conf, 0.3)
            col = (40, 200, 40) if conf >= args.conf_thr else (230, 160, 40)
            cv2.putText(ov, f"conf={conf:.2f}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            for j, fl in enumerate(flags[:3]):
                cv2.putText(ov, fl, (8, res - 12 - 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 120, 40), 1)
        # part 分割图
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            cmap = (np.array(plt.get_cmap("tab20").colors) * 255).astype(np.uint8)
        except Exception:
            cmap = (np.random.RandomState(0).rand(20, 3) * 255).astype(np.uint8)
        seg = np.zeros((res, res, 3), np.uint8)
        for c in range(1, N_PARTS):
            seg[pred_part == c] = cmap[c % 20]
        panel = np.concatenate([im, seg, ov], axis=1)
        out = os.path.join(args.out, os.path.splitext(os.path.basename(f))[0] + f"_{side}.png")
        cv2.imwrite(out, panel[:, :, ::-1])
        gate_str = f" | {metrics} flags={flags}" if args.gate else ""
        print(f"{os.path.basename(f)} side={side}{gate_str} -> {out}")
        if args.gate:
            ranking.append((conf, os.path.basename(f), out))
    if args.gate and ranking:
        ranking.sort(reverse=True)
        best = ranking[0]
        print(f"\n[auto-select] 最优帧(喂机械臂)= {best[1]} conf={best[0]:.3f}  | 全部: " +
              ", ".join(f"{b[1]}:{b[0]:.2f}" for b in ranking))


if __name__ == "__main__":
    main()
