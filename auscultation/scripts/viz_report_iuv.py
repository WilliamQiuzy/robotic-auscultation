#!/usr/bin/env python3
"""报告可视化: 对实验室真实后背照片跑当前最佳感知模型, 出
  ① 对齐裁切原图  ② 稠密 IUV 可视化(part 色相 + U,V 连续梯度)  ③ 听诊点标注图
每张照片产 3 个独立文件(原图/IUV/点位), 供 PPT 排版。

用法:
  FORCE_CPU=1 python scripts/viz_report_iuv.py \
    --ckpt docs/runs/perc_lung_v3safe_clinical_v1/best.pt \
    --imgs data/lab_sample_data/back --task lung \
    --tta 1.0,1.14,1.27 --out outputs/report_viz
"""
import argparse, glob, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in ("ausc", "ausc/models", "ausc/training"):
    sys.path.insert(0, os.path.join(ROOT, p))
from iuv_model import N_PARTS                                   # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD                 # noqa: E402
from eval import _forward_maps, readout_soft                    # noqa: E402
from infer_overlay import load_model, center_square, landmark_addresses  # noqa: E402
from tasks import task_region, task_landmarks                   # noqa: E402
from readout_fallback import readout_with_fallback             # noqa: E402


def flip_inconsistency(model, im, res, tta, addr, names, gx, gy, tau, device):
    """部署置信门: 同一张背正图 vs 水平镜像图各读一次点, 镜像点 un-flip(x→res-1-x)+换L/R 后
    与正图点的平均距离(占图宽%)。模型对该背把握高→两者应镜像一致(低); 不确定(侧弯/非典型)→高。
    金标实测: 正常lab背 2-4%(可信), scoliosis 18-45%(模型自知不确定, 该人工复核)。"""
    def _pred(img_rgb):
        x = torch.from_numpy(((img_rgb / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
                             .transpose(2, 0, 1)[None]).float().to(device)
        pp, puv = _forward_maps(model, x, res, tta)
        return {nm: readout_soft(pp[addr[nm][0]], puv, addr[nm], gx, gy, tau) for nm in names}
    p0 = _pred(im)
    pf = _pred(im[:, ::-1, :].copy())
    swap = lambda nm: (("L" if nm[0] == "R" else "R") + nm[1:])
    pu = {swap(nm): (res - 1 - pf[nm][0], pf[nm][1]) for nm in pf if pf[nm]}
    d = [((p0[nm][0] - pu[nm][0]) ** 2 + (p0[nm][1] - pu[nm][1]) ** 2) ** 0.5
         for nm in names if p0.get(nm) and pu.get(nm)]
    return (sum(d) / len(d) / res * 100.0) if d else 0.0

import torch


def iuv_visual(pred_part, pred_uv, bg=(12, 12, 12)):
    """DensePose 风格 IUV 可视化: 每个 part 一个色相, part 内用 (U,V) 做连续梯度
    (V→饱和度, U→明度), 背景纯色。直观展示"逐像素体表坐标"。返回 BGR uint8。"""
    H, W = pred_part.shape
    U = np.clip(pred_uv[0], 0, 1)
    V = np.clip(pred_uv[1], 0, 1)
    # 24 个身体块均匀铺色相(避开纯红/纯绿便于和点位区分), 跳着取以让相邻 part 对比明显
    hue = np.zeros((H, W), np.float32)
    order = (np.arange(1, N_PARTS) * 7) % 24                    # 打散相邻 part 的色相
    for c in range(1, N_PARTS):
        hue[pred_part == c] = order[c - 1] / 24.0 * 179.0       # OpenCV H ∈ [0,179]
    sat = (0.45 + 0.55 * V) * 255.0
    val = (0.45 + 0.55 * U) * 255.0
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    fg = pred_part > 0
    out = np.empty((H, W, 3), np.uint8)
    out[:] = np.array(bg, np.uint8)
    out[fg] = bgr[fg]
    return out


def draw_points(im_rgb, pts_ordered):
    """在 RGB 裁切图上画有序听诊点。R=红/L=蓝(分色), 标签朝外侧避免叠字。
    兜底状态用外环标记: 黄=夹回体内, 橙=V场塌缩重展, 品红=下背不在画面(不可信)。返回 BGR uint8。"""
    ov = im_rgb[:, :, ::-1].copy()                             # → BGR
    R = (40, 40, 235); L = (235, 150, 40)                      # BGR: 患者右=红, 患者左=蓝
    RING = {"clamped": (0, 220, 255), "respread": (0, 150, 255),
            "lowfov": (235, 0, 235)}                           # BGR: 黄/橙/品红
    for nm, (x, y), status in pts_ordered:
        x, y = int(round(x)), int(round(y))
        col = R if nm[0] == "R" else L
        if status in RING:
            cv2.circle(ov, (x, y), 10, RING[status], 2)        # 外环: 兜底状态
        cv2.circle(ov, (x, y), 7, (255, 255, 255), -1)
        cv2.circle(ov, (x, y), 5, col, -1)
        # 标签朝外: R 点(图像左)标在左侧, L 点(图像右)标在右侧 → 不往中间挤
        tx = x - 30 if nm[0] == "R" else x + 9
        cv2.putText(ov, nm, (tx, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(ov, nm, (tx, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--task", default="lung")
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--tta", default="1.0,1.14,1.27")
    ap.add_argument("--up_bias", type=float, default=0.12)
    ap.add_argument("--review_thr", type=float, default=12.0,
                    help="翻转不一致>该%→标 NEEDS REVIEW(模型在该背不确定, 多见scoliosis/非典型, 需人工核对)")
    ap.add_argument("--smpl", default="data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--out", default="outputs/report_viz")
    args = ap.parse_args()
    device = "cpu" if os.environ.get("FORCE_CPU") else "cuda"
    os.makedirs(args.out, exist_ok=True)
    tta = tuple(float(x) for x in args.tta.split(",")) if args.tta else None
    res = args.res
    gy, gx = np.mgrid[0:res, 0:res].astype(np.float32)

    m = load_model(args.ckpt, args.encoder, device)
    addr, _ = landmark_addresses(args.uv, args.smpl)
    region = task_region(args.task)
    names = [n for n in task_landmarks(args.task) if n in addr]
    print(f"[task] {args.task} region={region} points={names}")

    files = sorted(glob.glob(os.path.join(args.imgs, "*")))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for f in files:
        raw = cv2.imread(f)
        if raw is None:
            print("skip", f); continue
        crop = center_square(raw[:, :, ::-1], args.up_bias)            # RGB
        im = cv2.resize(crop, (res, res), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(((im / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
                             .transpose(2, 0, 1)[None]).float().to(device)
        part_prob, pred_uv = _forward_maps(m, x, res, tta)             # (25,H,W),(2,H,W)
        pred_part = part_prob.argmax(0)
        back_px = int((pred_part == 1).sum()); front_px = int((pred_part == 2).sum())
        side = "back" if back_px >= front_px else "front"

        # soft-argmax 读出 + 部署兜底: 体外夹回 / V场塌缩重展 / 下背不在画面标记
        pts, flags, _meta = readout_with_fallback(
            part_prob, pred_uv, addr, names, gx, gy, args.tau, back_part=1)
        pts_ordered = [(nm, (pts[nm][0], pts[nm][1]), pts[nm][2])
                       for nm in names if nm in pts]

        # 置信门: 翻转一致性(模型自评), 高=不可信→需人工复核
        flip_pct = flip_inconsistency(m, im, res, tta, addr, names, gx, gy, args.tau, device)
        review = flip_pct > args.review_thr
        if review:
            flags = flags + [f"NEEDS_REVIEW(翻转不一致{flip_pct:.0f}%>{args.review_thr:.0f}, 模型不确定)"]

        iuv = iuv_visual(pred_part, pred_uv)
        pov = draw_points(im, pts_ordered)
        if review:                                                 # 红色复核横幅
            cv2.rectangle(pov, (0, 0), (pov.shape[1], 26), (0, 0, 200), -1)
            cv2.putText(pov, f"NEEDS REVIEW  flip={flip_pct:.0f}%  (model uncertain - verify on patient)",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        stem = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(args.out, f"{stem}_1photo.png"), im[:, :, ::-1])
        cv2.imwrite(os.path.join(args.out, f"{stem}_2iuv.png"), iuv)
        cv2.imwrite(os.path.join(args.out, f"{stem}_3points.png"), pov)
        flag_str = (" FLAGS=" + "|".join(flags)) if flags else ""
        print(f"{os.path.basename(f)} side={side} n_pts={len(pts_ordered)}"
              f"{flag_str} -> {stem}_{{1photo,2iuv,3points}}.png")


if __name__ == "__main__":
    main()
