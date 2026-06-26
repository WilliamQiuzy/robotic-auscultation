#!/usr/bin/env python3
"""部署推理(landmark 头生产管线): RGB 后背照 → center_square 518 → 一次前向出
  ① 10 个后背肺听诊点(landmark 热图头直接读, 不走有偏的 IUV 地址)
  ② IUV 可视化(v7 冻结骨干, 仅参考)
  ③ 翻转一致性置信门 → 高=NEEDS REVIEW(模型不确定, 多见非典型/穿衣域外, 需人工核对)
每张出 3 文件(_1photo/_2iuv/_3points)。听诊点读出彻底脱离 HMR2/V6A 地址(金标证明地址法仅57%)。

用法:
  python scripts/deploy_landmark.py \
    --imgs data/lab_sample_data/back --out outputs/deploy_lmh --review_thr 12
  # 默认 --ckpt = outputs/perc_lmh_v4/best.pt (生产模型, 后背肺 10 点)
"""
import argparse, glob, os, sys
import numpy as np, cv2, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in ("ausc", "ausc/models", "ausc/training"):
    sys.path.insert(0, os.path.join(ROOT, p))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from iuv_model import DenseIUVModel, apply_lora, N_PARTS                  # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD                          # noqa: E402
from infer_overlay import center_square                                  # noqa: E402
from train_landmark_human import heatmap_xy, structure_points, ORDER, IDX  # noqa: E402


def load_landmark_model(ckpt, encoder, device):
    ck = torch.load(ckpt, map_location=device)
    m = DenseIUVModel(encoder=encoder, frozen_encoder=False, n_landmarks=len(ORDER)).to(device)
    apply_lora(m, r=ck["lora_rank"], alpha=ck["lora_alpha"])
    m.load_state_dict(ck["lora_head"], strict=False); m.to(device).eval()
    return m


def iuv_visual(part, uv, bg=(12, 12, 12)):
    H, W = part.shape; U = np.clip(uv[0], 0, 1); V = np.clip(uv[1], 0, 1)
    hue = np.zeros((H, W), np.float32); order = (np.arange(1, N_PARTS) * 7) % 24
    for c in range(1, N_PARTS):
        hue[part == c] = order[c - 1] / 24.0 * 179.0
    hsv = np.stack([hue, (0.45 + 0.55 * V) * 255, (0.45 + 0.55 * U) * 255], -1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR); out = np.empty((H, W, 3), np.uint8)
    out[:] = bg; fg = part > 0; out[fg] = bgr[fg]; return out


def _points(m, im, res, hm_res, device):
    x = torch.from_numpy(((im / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
                         .transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    with torch.no_grad():
        pl, uv, hl = m.forward_with_landmarks(x, (hm_res, hm_res))
    xy = heatmap_xy(hl.float(), res)[0].cpu().numpy()
    return xy, pl[0].argmax(0).cpu().numpy(), uv[0].cpu().numpy()


def iuv_back_crop(m, rgb, res, hm_res, device, up_bias=0.12, margin=0.18, back_part=1):
    """2-pass smart crop: pass-1 center-square → IUV back-torso(part=1) bbox → tight square
    re-crop so the back fills the 518² input (model is scale/translation-sensitive — this
    normalizes framing and removes the frame-to-frame collapse). Falls back to the center
    square if the back isn't found. Validated on lab video: vertical spread 0.04–0.56 (blind)
    → 0.47–0.56 (this). For Orbbec, the depth-based back crop is the production equivalent."""
    h, w = rgb.shape[:2]
    s = min(h, w); x0 = (w - s) // 2; y0 = max(0, min(h - s, int((h - s) * (0.5 - up_bias))))
    im1 = cv2.resize(rgb[y0:y0 + s, x0:x0 + s], (res, res), interpolation=cv2.INTER_AREA)
    _, part, _ = _points(m, im1, res, hm_res, device)
    ys, xs = np.nonzero(part == back_part)
    if len(ys) < 200:
        return im1
    u0, u1, v0, v1 = xs.min(), xs.max(), ys.min(), ys.max()
    ox0, ox1 = x0 + u0 * s / res, x0 + u1 * s / res
    oy0, oy1 = y0 + v0 * s / res, y0 + v1 * s / res
    cx, cy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
    side = max(ox1 - ox0, oy1 - oy0) * (1 + margin)
    bx0, by0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    bx1, by1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return cv2.resize(rgb[by0:by1, bx0:bx1], (res, res), interpolation=cv2.INTER_AREA)


def flip_incon(m, im, res, hm_res, device):
    xy0, _, _ = _points(m, im, res, hm_res, device)
    xyf, _, _ = _points(m, im[:, ::-1, :].copy(), res, hm_res, device)
    swap = {IDX[nm]: IDX[("L" + nm[1:] if nm[0] == "R" else "R" + nm[1:])] for nm in ORDER}
    d = [np.hypot(xy0[k, 0] - (res - 1 - xyf[swap[k], 0]), xy0[k, 1] - xyf[swap[k], 1]) for k in range(len(ORDER))]
    return float(np.mean(d) / res * 100)


def draw_points(im_rgb, xy):
    ov = im_rgb[:, :, ::-1].copy()
    R = (40, 40, 235); L = (235, 150, 40)
    for nm in ORDER:
        x, y = int(xy[IDX[nm], 0]), int(xy[IDX[nm], 1]); col = R if nm[0] == "R" else L
        cv2.circle(ov, (x, y), 7, (255, 255, 255), -1); cv2.circle(ov, (x, y), 5, col, -1)
        tx = x - 30 if nm[0] == "R" else x + 9
        cv2.putText(ov, nm, (tx, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(ov, nm, (tx, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/perc_lmh_v4/best.pt",
                    help="生产模型: landmark 头(纯人手标种子+自训练放量); 结构化读出")
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--out", default="outputs/deploy_lmh")
    ap.add_argument("--encoder", default="dinov2_vitb14"); ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--up_bias", type=float, default=0.12); ap.add_argument("--review_thr", type=float, default=12.0)
    ap.add_argument("--structured", type=int, default=1, help="1=结构化读出(脊柱中线+对称宽度, 保证R/L平行+层均匀)")
    ap.add_argument("--crop", default="iuv2pass", choices=["iuv2pass", "center"],
                    help="iuv2pass=智能裁剪(2-pass IUV 框后背→紧贴方裁, 消除取景塌缩, 默认); center=旧中心方裁")
    ap.add_argument("--crop_margin", type=float, default=0.18)
    args = ap.parse_args()
    device = "cpu" if os.environ.get("FORCE_CPU") else "cuda"
    hm_res = args.res // 4; os.makedirs(args.out, exist_ok=True)
    m = load_landmark_model(args.ckpt, args.encoder, device)
    files = [f for f in sorted(glob.glob(os.path.join(args.imgs, "*")))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for f in files:
        raw = cv2.imread(f)
        if raw is None:
            print("skip", f); continue
        rgb = raw[:, :, ::-1]
        if args.crop == "iuv2pass":
            im = iuv_back_crop(m, rgb, args.res, hm_res, device, args.up_bias, args.crop_margin)
        else:
            im = cv2.resize(center_square(rgb, args.up_bias), (args.res, args.res), interpolation=cv2.INTER_AREA)
        xy, part, uv = _points(m, im, args.res, hm_res, device)
        if args.structured:
            xy = structure_points(xy)                          # 保证 R/L 平行 + 层间距均匀
        flip = flip_incon(m, im, args.res, hm_res, device); review = flip > args.review_thr
        pov = draw_points(im, xy)
        if review:
            cv2.rectangle(pov, (0, 0), (pov.shape[1], 26), (0, 0, 200), -1)
            cv2.putText(pov, f"NEEDS REVIEW flip={flip:.0f}% (verify on patient)", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        stem = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(args.out, f"{stem}_1photo.png"), im[:, :, ::-1])
        cv2.imwrite(os.path.join(args.out, f"{stem}_2iuv.png"), iuv_visual(part, uv))
        cv2.imwrite(os.path.join(args.out, f"{stem}_3points.png"), pov)
        print(f"{os.path.basename(f)} flip={flip:.0f}%{' NEEDS_REVIEW' if review else ''} -> {stem}_{{1photo,2iuv,3points}}.png")


if __name__ == "__main__":
    main()
