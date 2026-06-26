#!/usr/bin/env python3
"""S2 with OPERA — 用 OPERA-CT(呼吸专用 foundation model)的 768-d embedding 替换基线编码器,
在同样的合成 clean/degraded 集上训质量头, 和 LogMelStats(0.939)/手工Q(0.785)比 AUC。

跑在 pod(需 /workspace/OPERA 已 setup + operaCT ckpt)。临时 wav 写 /tmp(快), 不写 moosefs。
  python scripts/audio/run_s2_opera.py --n_sources 100 --out /workspace/data/audio/_s2opera
"""
import argparse, glob, os, sys, json, wave
import numpy as np

ROOT = "/workspace/code/auscultation"
sys.path.insert(0, os.path.join(ROOT, "ausc", "audio"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "audio"))
sys.path.insert(0, "/workspace/OPERA")
import proxies as P            # noqa: E402
import calibrate_proxies as C  # noqa: E402  (退化算子)
from quality_head import QualityHead  # noqa: E402
from src.benchmark.model_util import extract_opera_feature  # noqa: E402
import torch, torch.nn as nn   # noqa: E402


def writew(x, path):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_glob", default="/tmp/hflung/**/*.wav")
    ap.add_argument("--n_sources", type=int, default=100)
    ap.add_argument("--tmp", default="/tmp/_s2opera_wavs")
    ap.add_argument("--out", default="/workspace/data/audio/_s2opera")
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(args.tmp, exist_ok=True); os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(17)
    cands = sorted(glob.glob(args.clean_glob, recursive=True)); rng.shuffle(cands)

    paths, y, g, n = [], [], [], 0
    for src in cands:
        if n >= args.n_sources:
            break
        try:
            xn, srn = P.read_wav(src); x, sr = P.resample(xn, srn)
        except Exception:
            continue
        if len(x) < sr:
            continue
        x = (x / (np.max(np.abs(x)) + 1e-9) * 0.3).astype(np.float32)
        m0 = P.metrics_from_signal(x, sr)
        if m0["snr_db"] < 12 or m0["silence_ratio"] > 0.4:
            continue
        n += 1
        p = os.path.join(args.tmp, f"s{n}_clean.wav"); writew(x, p)
        paths.append(p); y.append(1); g.append(n)
        for mode, tier, xv in C.variants(x, rng):
            if tier < 2:
                continue
            p = os.path.join(args.tmp, f"s{n}_{mode}{tier}.wav"); writew(xv.astype(np.float32), p)
            paths.append(p); y.append(0); g.append(n)
    print(f"[s2-opera] built {len(paths)} wavs from {n} sources", flush=True)

    feats = extract_opera_feature(paths, pretrain="operaCT", input_sec=8, dim=768)
    X = np.array(feats, np.float32); y = np.array(y); g = np.array(g)
    print(f"[s2-opera] OPERA features {X.shape}", flush=True)

    groups = np.unique(g); r2 = np.random.default_rng(0); r2.shuffle(groups)
    val_g = set(groups[: max(1, len(groups) // 5)].tolist())
    va = np.array([gg in val_g for gg in g]); tr = ~va
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; Xs = (X - mu) / sd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xs, device=dev); yt = torch.tensor(y, dtype=torch.float32, device=dev)
    head = QualityHead(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    best = -1
    for ep in range(args.epochs):
        head.train(); opt.zero_grad(); loss = lossf(head(Xt[tr]), yt[tr]); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == args.epochs - 1:
            head.eval()
            with torch.no_grad():
                pv = torch.sigmoid(head(Xt[va])).cpu().numpy()
            a = P.auc(pv, y[va]); best = max(best, a)
            print(f"  ep{ep} loss {loss.item():.3f} val_AUC {a:.3f}", flush=True)
    rep = {"encoder": "operaCT", "n": int(len(y)), "val_auc_opera": round(float(best), 3),
           "ref_logmel_stats": 0.939, "ref_handcrafted_q": 0.785}
    json.dump(rep, open(os.path.join(args.out, "s2_opera_report.json"), "w"), indent=2)
    print(f"[S2-OPERA done] {json.dumps(rep)}", flush=True)


if __name__ == "__main__":
    main()
