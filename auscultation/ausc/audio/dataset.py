#!/usr/bin/env python3
"""S2 — 数据装配: 把肺音变成 (embedding, 质量标)。两种来源:
- build_synth_xy: 合成 clean/degraded(复用 calibrate 的退化算子)→ 学习头的 bootstrap 训练集,
  可直接和手工 Q_proxy 比 AUC, 看学习头能不能超过手工规则。
- load_manifest_xy: 从弱标 manifest(q_proxy)或一手标注(overall)读 → 真训练。
按【源/受试者】分组返回, 切分必须按组防泄漏。
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc", "audio"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "audio"))
import proxies as P  # noqa: E402
import calibrate_proxies as C  # noqa: E402  (退化算子)


def build_synth_xy(clean_glob, encoder, n_sources=120, seed=17):
    """合成 clean(label 1)/degraded(label 0) → (X embedding, y, group=源id, q_hand=手工Q)。"""
    rng = np.random.default_rng(seed)
    cands = sorted(glob.glob(clean_glob, recursive=True))
    rng.shuffle(cands)
    X, y, g, qh = [], [], [], []
    n = 0
    for path in cands:
        if n >= n_sources:
            break
        try:
            xn, srn = P.read_wav(path); x, sr = P.resample(xn, srn)
        except Exception:
            continue
        if len(x) < sr:
            continue
        x = (x / (np.max(np.abs(x)) + 1e-9) * 0.3).astype(np.float32)
        m0 = P.metrics_from_signal(x, sr)
        if m0["snr_db"] < 12 or m0["silence_ratio"] > 0.4:      # 源不够干净则跳过
            continue
        n += 1
        X.append(encoder.embed_signal(x, sr)); y.append(1); g.append(n); qh.append(m0["q_proxy"])
        for mode, tier, xv in C.variants(x, rng):
            if tier < 2:
                continue
            xv = xv.astype(np.float32)
            mv = P.metrics_from_signal(xv, sr)
            X.append(encoder.embed_signal(xv, sr)); y.append(0); g.append(n); qh.append(mv["q_proxy"])
    return (np.array(X, np.float32), np.array(y), np.array(g), np.array(qh, np.float32))


def load_manifest_xy(manifest, encoder, label_key="q_proxy"):
    """从 jsonl(每行含 wav + label_key)读 → (X, y)。一手标注用 label_key='quality.overall' 归一化。"""
    X, y = [], []
    for line in open(manifest):
        r = json.loads(line)
        lab = r
        for k in label_key.split("."):
            lab = lab.get(k) if isinstance(lab, dict) else None
        if lab is None:
            continue
        try:
            X.append(encoder.embed(r["wav"]))
        except Exception:
            continue
        y.append(float(lab))
    return np.array(X, np.float32), np.array(y, np.float32)
