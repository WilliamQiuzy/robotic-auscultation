#!/usr/bin/env python3
"""S2 — 训练质量头(冻结编码器 + 轻头)。
bootstrap 模式: 合成 clean/degraded → 训学习头, 和手工 Q_proxy 比 AUC。
  python ausc/audio/train.py --mode synth --clean_glob "/tmp/hflung/**/*.wav" \
    --encoder logmel_stats --n_sources 150 --epochs 300 --out outputs/s2_v0
真训练模式(等一手标注): --mode manifest --manifest annots.jsonl --label_key quality.overall
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc", "audio"))
import proxies as P  # noqa: E402
from encoders import build_encoder  # noqa: E402
from quality_head import QualityHead  # noqa: E402
import dataset as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="synth", choices=["synth", "manifest"])
    ap.add_argument("--clean_glob", default="/tmp/hflung/**/*.wav")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--label_key", default="q_proxy")
    ap.add_argument("--encoder", default="logmel_stats")
    ap.add_argument("--n_sources", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="outputs/s2_v0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    enc = build_encoder(args.encoder)

    qh = None
    if args.mode == "synth":
        X, y, g, qh = D.build_synth_xy(args.clean_glob, enc, args.n_sources)
        print(f"[data] synth X={X.shape} clean={int((y==1).sum())} bad={int((y==0).sum())} "
              f"sources={len(np.unique(g))}", flush=True)
    else:
        X, y = D.load_manifest_xy(args.manifest, enc, args.label_key)
        g = np.arange(len(y))                                   # 无分组信息时按样本
        print(f"[data] manifest X={X.shape}", flush=True)
    if len(X) < 20:
        print("[abort] too few examples", flush=True); return

    # 按组切 train/val(防泄漏)
    groups = np.unique(g); rng = np.random.default_rng(0); rng.shuffle(groups)
    val_g = set(groups[: max(1, len(groups) // 5)].tolist())
    va = np.array([gg in val_g for gg in g]); tr = ~va
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xs, device=dev); yt = torch.tensor(y, dtype=torch.float32, device=dev)
    head = QualityHead(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    binary = set(np.unique(y).tolist()) <= {0.0, 1.0}
    lossf = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()

    best = -1
    for ep in range(args.epochs):
        head.train(); opt.zero_grad()
        out = head(Xt[tr])
        loss = lossf(out, yt[tr] if binary else yt[tr])
        loss.backward(); opt.step()
        if ep % 50 == 0 or ep == args.epochs - 1:
            head.eval()
            with torch.no_grad():
                pv = (torch.sigmoid(head(Xt[va])) if binary else head(Xt[va])).cpu().numpy()
            metric = P.auc(pv, y[va]) if binary else float(np.corrcoef(pv, y[va])[0, 1])
            best = max(best, metric)
            print(f"  ep{ep} loss {loss.item():.3f} val_{'AUC' if binary else 'corr'} {metric:.3f}", flush=True)

    report = {"mode": args.mode, "encoder": args.encoder, "n": int(len(y)),
              "val_metric_learned": round(float(best), 3), "binary": binary}
    if qh is not None and binary:
        report["val_auc_qproxy_handcrafted"] = round(float(P.auc(qh[va], y[va])), 3)
    torch.save({"state": head.state_dict(), "mu": mu, "sd": sd, "dim": int(X.shape[1]),
                "encoder": args.encoder}, os.path.join(args.out, "head.pt"))
    json.dump(report, open(os.path.join(args.out, "s2_report.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[S2 done] {json.dumps(report, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
