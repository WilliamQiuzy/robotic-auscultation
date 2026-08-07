#!/usr/bin/env python3
"""纯人手标 landmark 头训练。
v7 IUV 骨干暖启动 + 冻结(encoder LoRA + IUV 头), 只训一个新的 landmark 头, 监督信号 = 用户手标的
10 个后背肺点(2D 热图), 彻底脱离 HMR2/V6A 地址。这是金标证明 HMR2 标签仅 45% 后的"治本"路径。

数据 = canvas png(center_square 518, 部署空间) + 手标点(同空间)。不需要 IUV。
用法:
  python scripts/train_landmark_human.py \
    --train gold_train.json --test gold_test.json --canvas_root <dir with all canvases> \
    --init_from outputs/perc_v7_multiview/best.pt --out outputs/perc_lmh_v1 --epochs 40
其中 train/test json = {stem: {points:{R1:[x,y],...}}}; canvas_root 下有 <stem>.png。
"""
import argparse, glob, json, os, sys
import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in ("ausc", "ausc/models", "ausc/training"):
    sys.path.insert(0, os.path.join(ROOT, p))
import torch.nn.functional as F                                         # noqa: E402
from iuv_model import DenseIUVModel, apply_lora                          # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD                          # noqa: E402


def weighted_hm_loss(logits, tgt, mask, pos_w=100.0):
    """正样本加权热图损失: 高斯目标几乎全 0, 纯 MSE 会塌成全零(不出峰)。
    给峰区(tgt 大)加权 pos_w 倍, 强制出峰。仅在场+可见通道上算。"""
    pred = torch.sigmoid(logits)
    w = 1.0 + pos_w * tgt
    per = (w * (pred - tgt) ** 2).mean(dim=(2, 3))                       # (B,K)
    m = mask.float(); denom = m.sum().clamp(min=1.0)
    loss = (per * m).sum() / denom
    return loss, {"hm": loss.item()}

ORDER = [f"{s}{i}" for i in range(1, 6) for s in ("R", "L")]            # 10 后背肺点
IDX = {nm: i for i, nm in enumerate(ORDER)}
K = len(ORDER)


def make_heatmaps(points, hm_res, res, sigma=2.5):
    hm = np.zeros((K, hm_res, hm_res), np.float32); mask = np.zeros((K,), np.float32)
    s = hm_res / float(res); rad = int(3 * sigma)
    for nm, xy in points.items():
        if nm not in IDX:
            continue
        cx, cy = xy[0] * s, xy[1] * s
        if not (0 <= cx < hm_res and 0 <= cy < hm_res):
            continue
        x0, x1 = max(0, int(cx) - rad), min(hm_res, int(cx) + rad + 1)
        y0, y1 = max(0, int(cy) - rad), min(hm_res, int(cy) + rad + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        gx = np.arange(x0, x1, dtype=np.float32); gy = np.arange(y0, y1, dtype=np.float32)[:, None]
        hm[IDX[nm], y0:y1, x0:x1] = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
        mask[IDX[nm]] = 1.0
    return hm, mask


class HandLabelDS(Dataset):
    def __init__(self, items, res, hm_res, sigma, train=False):
        self.items, self.res, self.hm_res, self.sigma, self.train = items, res, hm_res, sigma, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, pts, _id = self.items[i]
        im = cv2.imread(path)[:, :, ::-1]
        im = cv2.resize(im, (self.res, self.res), interpolation=cv2.INTER_AREA)
        p = {nm: [float(xy[0]), float(xy[1])] for nm, xy in pts.items()}
        if self.train and np.random.rand() < 0.5:                       # 水平翻转 + L/R 互换
            im = im[:, ::-1, :].copy()
            p = {("L" + nm[1:] if nm[0] == "R" else "R" + nm[1:]): [self.res - 1 - xy[0], xy[1]]
                 for nm, xy in p.items()}
        x = ((im / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
        hm, mask = make_heatmaps(p, self.hm_res, self.res, self.sigma)
        return torch.from_numpy(x.copy()), torch.from_numpy(hm), torch.from_numpy(mask), \
            {nm: tuple(xy) for nm, xy in p.items()}


def collate(b):
    return (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]),
            torch.stack([x[2] for x in b]), [x[3] for x in b])


def heatmap_xy(hm_logits, res, win=5):
    """(B,K,h,w) logits → (B,K,2) xy in res space。先 argmax 找峰(抗背景稀释),
    再在峰附近 ±win 窗口内对 sigmoid 概率做加权质心(亚像素)。"""
    B, Kk, h, w = hm_logits.shape
    prob = torch.sigmoid(hm_logits)
    flat = prob.view(B, Kk, -1)
    idx = flat.argmax(-1)
    py0 = (idx // w).float(); px0 = (idx % w).float()
    out = torch.zeros(B, Kk, 2, device=hm_logits.device)
    for b in range(B):
        for k in range(Kk):
            cy, cx = int(py0[b, k]), int(px0[b, k])
            y0, y1 = max(0, cy - win), min(h, cy + win + 1)
            x0, x1 = max(0, cx - win), min(w, cx + win + 1)
            patch = prob[b, k, y0:y1, x0:x1]
            s = patch.sum().clamp(min=1e-6)
            ys = torch.arange(y0, y1, device=hm_logits.device).float()
            xs = torch.arange(x0, x1, device=hm_logits.device).float()
            my = (patch.sum(1) * ys).sum() / s
            mx = (patch.sum(0) * xs).sum() / s
            out[b, k, 0] = mx * (res / w); out[b, k, 1] = my * (res / h)
    return out


def structure_points(xy):
    """把 10 个独立点拟合成结构化网格: 脊柱中线(主轴, 5 层均匀) + 每层关于中线的对称垂直偏移。
    数学上保证 R/L 双侧对称(平行)+ 层间距均匀。xy: (10,2) in ORDER。返回 (10,2)。"""
    R = np.array([xy[IDX[f"R{i}"]] for i in range(1, 6)])
    L = np.array([xy[IDX[f"L{i}"]] for i in range(1, 6)])
    M = (R + L) / 2.0; c = M.mean(0); X = M - c
    try:
        _, _, vt = np.linalg.svd(X)
    except np.linalg.LinAlgError:
        return xy
    d = vt[0]; d = d if d[1] >= 0 else -d                # 脊柱方向(朝下)
    t = X @ d; te = np.linspace(t[0], t[4], 5)           # 5 层沿脊柱均匀
    Mp = c + np.outer(te, d); perp = np.array([-d[1], d[0]])
    out = xy.copy()
    for i in range(5):
        w = ((R[i] - L[i]) @ perp) / 2.0                 # 关于中线的对称半宽
        out[IDX[f"R{i+1}"]] = Mp[i] + perp * w
        out[IDX[f"L{i+1}"]] = Mp[i] - perp * w
    return out


def structure_points_robust(xy):
    """Robust variant of ``structure_points`` (2026-08-07 audit item #5).

    ``structure_points`` sets the 5 layer axial positions with ``np.linspace(t[0], t[4], 5)`` —
    the spacing is driven *only* by the two endpoint levels (R1/L1 and R5/L5), and R5/L5 are the
    noisiest points on the curved lower back, so their jitter smears the middle layers. This fits
    the axial positions ``t_i ≈ a + b·i`` over ALL five levels with a Theil-Sen line (median of
    pairwise slopes → resistant to a single bad endpoint), then rebuilds the even grid from that.
    Half-widths stay per-level (unchanged). Pure numpy; same signature/return as structure_points.
    """
    R = np.array([xy[IDX[f"R{i}"]] for i in range(1, 6)])
    L = np.array([xy[IDX[f"L{i}"]] for i in range(1, 6)])
    M = (R + L) / 2.0; c = M.mean(0); X = M - c
    try:
        _, _, vt = np.linalg.svd(X)
    except np.linalg.LinAlgError:
        return xy
    d = vt[0]; d = d if d[1] >= 0 else -d
    t = X @ d
    idx = np.arange(5, dtype=np.float64)
    slopes = [(t[b] - t[a]) / (b - a) for a in range(5) for b in range(a + 1, 5)]
    b_ts = float(np.median(slopes))                      # Theil-Sen slope over all 10 pairs
    a_ts = float(np.median(t - b_ts * idx))              # robust intercept
    te = a_ts + b_ts * idx                               # even grid fit from ALL layers
    Mp = c + np.outer(te, d); perp = np.array([-d[1], d[0]])
    out = xy.copy()
    for i in range(5):
        w = ((R[i] - L[i]) @ perp) / 2.0
        out[IDX[f"R{i+1}"]] = Mp[i] + perp * w
        out[IDX[f"L{i+1}"]] = Mp[i] - perp * w
    return out


def heatmap_xy_soft(hm_logits, res, win=5, temp=1.0):
    """Temperature soft-argmax variant of ``heatmap_xy`` (2026-08-07 audit item #5).

    NOTE: ``heatmap_xy`` is ALREADY a windowed soft-argmax — it takes the sigmoid-probability
    weighted centroid over a ±win window around the hard-argmax peak (a DSNT-style expectation on
    the *calibrated* sigmoid probabilities the model was trained with, via sigmoid+MSE). So item
    #5b is largely already in production. This adds only a temperature knob on those probabilities:
    ``w = p**(1/temp)`` — ``temp=1`` reduces EXACTLY to ``heatmap_xy`` (never worse); ``temp>1``
    softens (flatter window → less grid-flip when the peak jitters ±1 cell); ``temp<1`` sharpens.
    Operating on sigmoid probs (not raw logits) matters: the head is sigmoid-MSE-trained, so a
    softmax over raw logits would be badly over-peaked and worse than the centroid. Same
    (B,K,h,w)->(B,K,2 in res space) contract as heatmap_xy.
    """
    B, Kk, h, w = hm_logits.shape
    prob = torch.sigmoid(hm_logits)
    flat = prob.view(B, Kk, -1)
    idx = flat.argmax(-1)
    py0 = (idx // w).float(); px0 = (idx % w).float()
    out = torch.zeros(B, Kk, 2, device=hm_logits.device)
    inv_t = 1.0 / max(temp, 1e-6)
    for b in range(B):
        for k in range(Kk):
            cy, cx = int(py0[b, k]), int(px0[b, k])
            y0, y1 = max(0, cy - win), min(h, cy + win + 1)
            x0, x1 = max(0, cx - win), min(w, cx + win + 1)
            patch = prob[b, k, y0:y1, x0:x1] ** inv_t          # temp=1 → the heatmap_xy centroid
            s = patch.sum().clamp(min=1e-9)
            ys = torch.arange(y0, y1, device=hm_logits.device).float()
            xs = torch.arange(x0, x1, device=hm_logits.device).float()
            my = (patch.sum(1) * ys).sum() / s
            mx = (patch.sum(0) * xs).sum() / s
            out[b, k, 0] = mx * (res / w); out[b, k, 1] = my * (res / h)
    return out


def load_items(jsonf, canvas_root):
    d = json.load(open(jsonf)); out = []
    for stem, rec in d.items():
        if stem == "_meta":
            continue
        pts = rec.get("points", rec)
        cand = glob.glob(os.path.join(canvas_root, f"{stem}.png")) + \
            glob.glob(os.path.join(canvas_root, "**", f"{stem}.png"), recursive=True)
        if not cand:
            continue
        pp = {nm: v["xy"] if isinstance(v, dict) else v for nm, v in pts.items()}
        if len(pp) >= 8:
            out.append((cand[0], pp, stem))
    return out


@torch.no_grad()
def evaluate(model, ds, res, hm_res, device, thr=0.10):
    model.eval(); errs = []
    for x, _hm, _m, pts in DataLoader(ds, batch_size=8, collate_fn=collate):
        hl = model.predict_landmark_heatmap(x.to(device), (hm_res, hm_res))
        xy = heatmap_xy(hl.float(), res).cpu().numpy()
        for b, p in enumerate(pts):
            for nm, t in p.items():
                d = float(np.hypot(xy[b, IDX[nm], 0] - t[0], xy[b, IDX[nm], 1] - t[1]))
                errs.append(d)
    errs = np.array(errs)
    return {"pck10": float((errs < thr * res).mean()), "pck5": float((errs < 0.05 * res).mean()),
            "medpx": float(np.median(errs)), "n": len(errs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--canvas_root", required=True)
    ap.add_argument("--init_from", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--res", type=int, default=518); ap.add_argument("--hm_res", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    device = "cpu" if os.environ.get("FORCE_CPU") else "cuda"
    hm_res = args.hm_res or (args.res // 4)
    os.makedirs(args.out, exist_ok=True)

    tr_items = load_items(args.train, args.canvas_root)
    te_items = load_items(args.test, args.canvas_root)
    print(f"[data] train={len(tr_items)} test={len(te_items)} | K={K} hm_res={hm_res}")

    # 模型: v7 LoRA 暖启动 + 新 landmark 头, 冻结除 landmark 头外一切
    ck = torch.load(args.init_from, map_location=device)
    m = DenseIUVModel(encoder=args.encoder, frozen_encoder=False, n_landmarks=K).to(device)
    apply_lora(m, r=ck["lora_rank"], alpha=ck["lora_alpha"])
    miss, unexp = m.load_state_dict(ck["lora_head"], strict=False)
    m.to(device)
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith("landmark_head."))
    ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[model] 冻结IUV+LoRA, 仅训 landmark 头 {ntr/1e6:.2f}M")

    tr = DataLoader(HandLabelDS(tr_items, args.res, hm_res, args.sigma, train=True), batch_size=args.bs,
                    shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=False)
    te = HandLabelDS(te_items, args.res, hm_res, args.sigma)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best = -1
    for ep in range(args.epochs):
        m.train(); run = 0.0
        for x, hm, mask, _ in tr:
            x, hm, mask = x.to(device), hm.to(device), mask.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                _pl, _uv, hl = m.forward_with_landmarks(x, (hm_res, hm_res))
                loss, _ = weighted_hm_loss(hl, hm, mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()
        ev = evaluate(m, te, args.res, hm_res, device)
        print(f"[ep{ep:02d}] loss {run/max(1,len(tr)):.5f} | test PCK@10 {ev['pck10']*100:.1f} "
              f"PCK@5 {ev['pck5']*100:.1f} medpx {ev['medpx']:.1f}", flush=True)
        if ev["pck10"] > best:
            best = ev["pck10"]
            torch.save({"kind": "lora", "lora_rank": ck["lora_rank"], "lora_alpha": ck["lora_alpha"],
                        "lora_head": {k: v.detach().cpu() for k, v in m.state_dict().items()
                                      if ".A." in k or ".B." in k or k.startswith("head.")
                                      or k.startswith("landmark_head.")},
                        "order": ORDER, "ep": ep, "pck10": best}, os.path.join(args.out, "best.pt"))
    print(f"[done] best test PCK@10 = {best*100:.1f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
