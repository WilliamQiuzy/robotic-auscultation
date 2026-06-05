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


def region_names(addr, region):
    """后背肺点=名字形如 L1..R4(字母+数字); 其余=前胸心点。"""
    back = [n for n in addr if len(n) >= 2 and n[0] in "LR" and n[1:].isdigit()]
    return sorted(back) if region == "back" else [n for n in addr if n not in back]


def posture_gate(points, pred_part, res, region):
    """部署姿态质量门: 据 后背占比/肺点双侧展开/对称/纵向覆盖 判 PASS / REPOSITION。
    points: {name:(x,y)}; pred_part: (H,W) argmax part。返回 (ok, reasons, metrics)。"""
    reasons = []
    fg = pred_part > 0
    torso = np.isin(pred_part, [1, 2])
    back_px = int((pred_part == 1).sum()); front_px = int((pred_part == 2).sum())
    back_frac = back_px / max(1, back_px + front_px)
    if fg.sum() < 0.04 * res * res:
        reasons.append("人太小/太远(前景占比低)→ 靠近, 让躯干填中部")
    if region == "back" and back_frac < 0.80:
        reasons.append(f"后背占比低({back_frac:.2f})→ 后背正对相机, 不要转身/前倾")
    # 躯干宽高(像素 bbox)
    ys, xs = np.where(torso if torso.sum() > 50 else fg)
    tw = (xs.max() - xs.min() + 1) if len(xs) else res
    th = (ys.max() - ys.min() + 1) if len(ys) else res
    cx = (xs.min() + xs.max()) / 2 if len(xs) else res / 2
    pts = np.array(list(points.values())) if points else np.zeros((0, 2))
    if len(pts) >= 4:
        xspread = (pts[:, 0].max() - pts[:, 0].min()) / max(1, tw)
        yspread = (pts[:, 1].max() - pts[:, 1].min()) / max(1, th)
        if xspread < 0.20:
            reasons.append(f"肺点横向未展开({xspread:.2f})→ 坐直、肩部正对(转身会聚成一条)")
        if yspread < 0.18:
            reasons.append(f"肺点纵向跨度小({yspread:.2f})→ 取景纳入上背到下背")
        # 左右平衡: L* 应在中线一侧, R* 另一侧
        L = np.array([points[n] for n in points if n[0] == "L"])
        R = np.array([points[n] for n in points if n[0] == "R"])
        if len(L) and len(R):
            if not (L[:, 0].mean() - cx) * (R[:, 0].mean() - cx) < 0:
                reasons.append("左右肺点未分居中线两侧→ 后背正对、不要侧身")
    else:
        reasons.append("有效肺点过少→ 检查后背是否清晰可见")
    metrics = {"back_frac": round(back_frac, 3),
               "xspread": round(float((pts[:, 0].max() - pts[:, 0].min()) / max(1, tw)), 3) if len(pts) >= 4 else None,
               "n_pts": int(len(pts))}
    return (len(reasons) == 0), reasons, metrics


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
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--tta", default=None)
    ap.add_argument("--up_bias", type=float, default=0.12, help="中心裁上偏(把坐姿躯干居中)")
    ap.add_argument("--gate", type=int, default=0, help="1=姿态质量门(PASS/REPOSITION + 原因)")
    ap.add_argument("--smpl", default="data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--out", default="infer_out")
    args = ap.parse_args()
    device = "cpu" if os.environ.get("FORCE_CPU") else "cuda"
    os.makedirs(args.out, exist_ok=True)
    tta = tuple(float(x) for x in args.tta.split(",")) if args.tta else None

    m = load_model(args.ckpt, args.encoder, device)
    addr, _ = landmark_addresses(args.uv, args.smpl)
    names = region_names(addr, args.region)
    res = args.res
    gy, gx = np.mgrid[0:res, 0:res].astype(np.float32)

    files = args.imgs.split(",") if "," in args.imgs or args.imgs.endswith((".jpg", ".png", ".JPG")) else sorted(glob.glob(os.path.join(args.imgs, "*")))
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
        gate_ok, reasons, metrics = (None, [], {})
        if args.gate:
            gate_ok, reasons, metrics = posture_gate(pts, pred_part, res, args.region)
            verdict = "PASS" if gate_ok else "REPOSITION"
            col = (40, 200, 40) if gate_ok else (230, 60, 60)
            cv2.putText(ov, verdict, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            for j, rs in enumerate(reasons[:3]):
                cv2.putText(ov, rs[:22], (8, res - 12 - 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 60, 60), 1)
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
        gate_str = ""
        if args.gate:
            gate_str = f" | GATE={'PASS' if gate_ok else 'REPOSITION'} {metrics}" + ("" if gate_ok else f" reasons={reasons}")
        print(f"{os.path.basename(f)} side={side}(back_px={back_px},front_px={front_px}){gate_str} -> {out}")


if __name__ == "__main__":
    main()
