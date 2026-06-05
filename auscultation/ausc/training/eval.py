#!/usr/bin/env python3
"""感知模型评测 harness。
指标(三层):
  1) 稠密 IUV: part mIoU + 像素准确率(前景 1..24); 前景 UV 平均 L1。
  2) ★Headline 听诊点定位: 从预测 IUV 按各 landmark 的标准 (I,U,V) 地址反查 2D 点,
     与 GT 点比 → 像素误差(@res) / 占图宽% / PCK@τ。区分前胸(心5)/后背(肺8)。
  3) 跨域 gap: 对不同数据集(如 synthetic-val vs scoliosis-real)分别报上面指标。
用法:
  python ausc/training/eval.py --ckpt outputs/perc_v0/best.pt --root data/dataset/synthetic --tag syn-val
  python ausc/training/eval.py --ckpt outputs/perc_v0/best.pt --root data/dataset/scoliosis --tag scol-real --all
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "models"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "training"))

from iuv_model import DenseIUVModel, N_PARTS          # noqa: E402
from dataset import AuscIUVDataset, split_manifest, collate, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from render_iuv import load_uv_mapping                # noqa: E402
from humman_label import define_ausc_vertices         # noqa: E402


def landmark_addresses(uv_mat, smpl_model):
    """每个听诊点 landmark 的标准 (part, U, V) 地址(常量, 由 uvmap+顶点离线算)。"""
    import smplx
    v2s, U, V, faces, fpart = load_uv_mapping(uv_mat)
    vert_parts = defaultdict(list)
    for fi in range(len(faces)):
        for dv in faces[fi]:
            vert_parts[int(dv)].append(int(fpart[fi]))
    m = smplx.SMPL(model_path=smpl_model, gender="neutral", batch_size=1)
    ausc = define_ausc_vertices(m)
    addr = {}
    for region in ("front", "back"):
        for nm, vi in ausc[region].items():
            dp = np.where(v2s == vi)[0]
            if len(dp) == 0:
                continue
            parts = [p for d in dp for p in vert_parts.get(int(d), [])]
            if not parts:
                continue
            addr[nm] = (Counter(parts).most_common(1)[0][0], float(U[dp].mean()), float(V[dp].mean()))
    return addr, ausc


def readout(pred_part, pred_uv, addr):
    """硬 argmin: 在 part==p 的像素里找 (U,V) 最接近地址的单像素。返回 (x,y) 或 None。"""
    part, u, v = addr
    ys, xs = np.where(pred_part == part)
    if len(xs) == 0:
        return None
    d = (pred_uv[0, ys, xs] - u) ** 2 + (pred_uv[1, ys, xs] - v) ** 2
    j = int(np.argmin(d))
    return float(xs[j]), float(ys[j])


def readout_soft(part_prob_p, pred_uv, addr, gx, gy, tau=0.01):
    """soft-argmax 地址亲和热图: A = P(part=p) · exp(-||UV-地址||²/τ), 取期望位置(亚像素抗噪)。"""
    _, u, v = addr
    d2 = (pred_uv[0] - u) ** 2 + (pred_uv[1] - v) ** 2
    A = part_prob_p * np.exp(-d2 / tau)
    s = A.sum()
    if s < 1e-8:
        return None
    return float((A * gx).sum() / s), float((A * gy).sum() / s)


def _forward_maps(model, img, res, tta_scales=None):
    """前向得 part 概率(softmax 后) + UV, 全部在 res 分辨率。
    tta_scales: 多尺度 TTA(如 (1.0,1.14,1.27)); 各尺度 softmax(part)+uv 上采样到 res 后平均(纯几何, 不动 part/UV 语义, 安全)。"""
    scales = tta_scales or (1.0,)
    prob_sum = None; uv_sum = None
    for sc in scales:
        s = max(1, round(res * sc / 14)) * 14            # DINOv2 patch=14 对齐
        x = F.interpolate(img, size=(s, s), mode="bilinear", align_corners=False) if s != res else img
        with torch.autocast("cuda", dtype=torch.float16):
            pl, puv = model(x)
        prob = torch.softmax(pl.float(), 1)
        if s != res:
            prob = F.interpolate(prob, size=(res, res), mode="bilinear", align_corners=False)
            puv = F.interpolate(puv.float(), size=(res, res), mode="bilinear", align_corners=False)
        prob_sum = prob if prob_sum is None else prob_sum + prob
        uv_sum = puv.float() if uv_sum is None else uv_sum + puv.float()
    n = len(scales)
    return (prob_sum / n)[0].cpu().numpy(), (uv_sum / n)[0].cpu().numpy()


@torch.no_grad()
def evaluate(model, root, rows, addr, res, device, max_n=None, dump=None, readout_mode="soft", tau=0.01, tta_scales=None):
    model.eval()
    ds = AuscIUVDataset(root, rows if max_n is None else rows[:max_n], res)
    inter = np.zeros(N_PARTS); union = np.zeros(N_PARTS)
    pix_ok = pix_tot = 0
    uv_err = uv_n = 0.0
    lm_err = {"front": [], "back": []}                  # 像素误差(@res)
    gy, gx = np.mgrid[0:res, 0:res].astype(np.float32)  # soft-argmax 用的坐标网格
    dumped = 0
    for k in range(len(ds)):
        s = ds[k]
        img = s["image"].unsqueeze(0).to(device)
        part_prob_full, pred_uv = _forward_maps(model, img, res, tta_scales)   # (25,H,W),(2,H,W)
        pred_part = part_prob_full.argmax(0).astype(np.int32)
        part_prob = part_prob_full if readout_mode == "soft" else None
        gt_part = s["part"].numpy(); gt_uv = s["uv"].numpy()
        # 稠密: mIoU + acc(前景)
        fg = gt_part > 0
        pix_ok += int((pred_part[fg] == gt_part[fg]).sum()); pix_tot += int(fg.sum())
        for c in range(1, N_PARTS):
            pc = pred_part == c; gc = gt_part == c
            inter[c] += int((pc & gc).sum()); union[c] += int((pc | gc).sum())
        m = fg
        uv_err += float(np.abs(pred_uv[:, m] - gt_uv[:, m]).sum()); uv_n += int(m.sum()) * 2
        # landmark 定位
        reg = s["region"]
        for nm, (x, y, vis) in s["points"].items():
            if not vis or nm not in addr:
                continue
            if readout_mode == "soft":
                p = readout_soft(part_prob[addr[nm][0]], pred_uv, addr[nm], gx, gy, tau)
            else:
                p = readout(pred_part, pred_uv, addr[nm])
            if p is None:
                continue
            e = ((p[0] - x) ** 2 + (p[1] - y) ** 2) ** 0.5
            lm_err[reg if reg in lm_err else "front"].append(e)
        # 定性叠加图
        if dump and dumped < 6:
            _dump_overlay(s, pred_part, pred_uv, addr, os.path.join(dump, f"eval_{dumped}.png"),
                          part_prob=part_prob, gx=gx, gy=gy, tau=tau, mode=readout_mode)
            dumped += 1
    iou = inter[1:] / np.maximum(union[1:], 1)
    valid = union[1:] > 0
    res_d = {
        "miou": float(iou[valid].mean()) if valid.any() else 0.0,
        "pix_acc": pix_ok / max(1, pix_tot),
        "uv_l1": uv_err / max(1.0, uv_n),
        "n": len(ds),
    }
    for reg in ("front", "back"):
        e = np.array(lm_err[reg])
        if len(e):
            res_d[reg] = {"n_pts": len(e), "med_px": float(np.median(e)), "mean_px": float(e.mean()),
                          "pct_img": float(np.median(e) / res * 100),
                          "pck@5%": float((e < 0.05 * res).mean()), "pck@10%": float((e < 0.10 * res).mean())}
    return res_d


def _dump_overlay(s, pred_part, pred_uv, addr, path, part_prob=None, gx=None, gy=None, tau=0.01, mode="soft"):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    img = ((s["image"].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255).clip(0, 255).astype(np.uint8)
    cmap = (np.array(plt.get_cmap("tab20").colors) * 255).astype(np.uint8)
    seg = np.zeros((*pred_part.shape, 3), np.uint8)
    for c in range(1, N_PARTS):
        seg[pred_part == c] = cmap[c % 20]
    ov = img.copy()
    for nm, (x, y, vis) in s["points"].items():
        if vis and nm in addr:
            cv2.circle(ov, (int(x), int(y)), 6, (40, 230, 40), 2)   # GT 绿
            if mode == "soft" and part_prob is not None:
                p = readout_soft(part_prob[addr[nm][0]], pred_uv, addr[nm], gx, gy, tau)
            else:
                p = readout(pred_part, pred_uv, addr[nm])
            if p:
                cv2.circle(ov, (int(p[0]), int(p[1])), 6, (230, 40, 40), -1)  # 预测 红
    fig, ax = plt.subplots(1, 3, figsize=(13, 5))
    ax[0].imshow(img); ax[0].set_title("input"); ax[1].imshow(seg); ax[1].set_title("pred part")
    ax[2].imshow(ov); ax[2].set_title("GT(绿) vs pred(红) landmarks")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=80, bbox_inches="tight"); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=None, help="单数据集目录(配合 --all / --val 划分)")
    ap.add_argument("--val_split", default=None, help="训练存的 val_split.json: 精确复现 held-out 行(无泄漏)")
    ap.add_argument("--only_root", default=None, help="配合 --val_split: 只评 _root 含该子串的数据集(分域报)")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--smpl", default="data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--all", action="store_true", help="评整个数据集(否则只评 val 划分)")
    ap.add_argument("--max_n", type=int, default=400)
    ap.add_argument("--dump", default=None, help="定性叠加图输出目录")
    ap.add_argument("--readout", default="soft", choices=["soft", "argmin"], help="听诊点读出: soft-argmax(默认) / 硬argmin")
    ap.add_argument("--tau", type=float, default=0.01, help="soft-argmax UV 亲和温度")
    ap.add_argument("--tta", default=None, help="多尺度TTA, 逗号分隔缩放因子(如 '1.0,1.14,1.27'); 留空=单尺度")
    args = ap.parse_args()
    device = "cuda"

    ck = torch.load(args.ckpt, map_location=device)
    if ck.get("kind") == "lora":               # LoRA: 原DINOv2基座 + 注入低秩 + 载 A/B+头
        from iuv_model import apply_lora
        model = DenseIUVModel(encoder=args.encoder, frozen_encoder=False).to(device)
        apply_lora(model, r=ck["lora_rank"], alpha=ck["lora_alpha"])
        model.load_state_dict(ck["lora_head"], strict=False)
        model.to(device)                           # A/B 注入后重新搬到 GPU
    else:
        model = DenseIUVModel(encoder=args.encoder, frozen_encoder=True).to(device)
        if "model" in ck:                      # 全解冻: 整模型
            model.load_state_dict(ck["model"])
        else:                                  # 冻结: 仅 head
            model.head.load_state_dict(ck["head"])
    model.eval()
    addr, _ = landmark_addresses(args.uv, args.smpl)
    if args.val_split:                          # 精确复现训练的 held-out(无泄漏); 行内自带 _root
        rows = json.load(open(args.val_split))
        if args.only_root:
            rows = [r for r in rows if args.only_root in r.get("_root", "")]
    elif args.all:
        rows = [json.loads(l) for l in open(os.path.join(args.root, "manifest.jsonl"))]
        rows = [{**r, "_root": args.root} for r in rows if r.get("is_valid") and r.get("iuv")]
    else:
        _, rows = split_manifest(args.root)
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
    tta_scales = tuple(float(x) for x in args.tta.split(",")) if args.tta else None
    r = evaluate(model, args.root or "", rows, addr, args.res, device, max_n=args.max_n, dump=args.dump,
                 readout_mode=args.readout, tau=args.tau, tta_scales=tta_scales)
    print(f"\n===== [{args.tag}] n={r['n']} (readout={args.readout}) =====")
    print(f"  part mIoU      : {r['miou']:.3f}")
    print(f"  part pix-acc   : {r['pix_acc']:.3f}")
    print(f"  UV L1 (fg)     : {r['uv_l1']:.4f}")
    for reg in ("front", "back"):
        if reg in r:
            d = r[reg]
            print(f"  [{reg}] landmark: med {d['med_px']:.1f}px ({d['pct_img']:.1f}% img) "
                  f"mean {d['mean_px']:.1f}px | PCK@5% {d['pck@5%']:.2f} PCK@10% {d['pck@10%']:.2f} (n={d['n_pts']})")
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/eval_{args.tag}.json", "w") as f:
        json.dump(r, f, indent=2)
    print(f"  → outputs/eval_{args.tag}.json")


if __name__ == "__main__":
    main()
