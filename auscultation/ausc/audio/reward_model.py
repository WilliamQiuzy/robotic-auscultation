"""Head-Reward — a Bradley-Terry preference reward model on the shared AST embedding.

docs/audio_reward_design.md §4 (LOCKED): the learned reward model is **preference-based** (doctors
rank pairs; absolute usable/unusable is noisy at κ0.5). This trains `R_φ(embedding) → scalar` so that
`P(a ≻ b) = σ(R(a) − R(b))` matches the preferences (Bradley-Terry). It plugs into the segment reward
once real doctor pairwise data exists (Preference rows in reward_schema.py); until then the __main__
validates the machinery on SYNTHETIC preferences (prefer the cleaner of two clips) and checks it
learns a quality ranking that GENERALIZES to unseen clips.

    python ausc/audio/reward_model.py            # synthetic-preference validation (runs the AST encoder)

Deps: torch, transformers (via q_head), numpy, soundfile. GPU (H200) speeds the embedding extraction.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for q_head


def _torch():
    import torch
    return torch


class BTReward:
    """MLP reward head R_φ: embedding(768) -> scalar score. Wraps a tiny torch net."""

    def __init__(self, dim: int = 768, hidden: int = 128, seed: int = 0):
        torch = _torch(); torch.manual_seed(seed)
        import torch.nn as nn
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def fit(self, emb: np.ndarray, pairs: np.ndarray, prefs: np.ndarray, epochs: int = 300, lr: float = 1e-3):
        """emb (N,768); pairs (M,2) index into emb; prefs (M,) 1.0 if pairs[:,0] preferred else 0.0."""
        torch = _torch()
        E = torch.tensor(np.asarray(emb, np.float32))
        A = torch.tensor(pairs[:, 0]); B = torch.tensor(pairs[:, 1])
        P = torch.tensor(np.asarray(prefs, np.float32))
        opt = torch.optim.Adam(self.net.parameters(), lr)
        bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            s = self.net(E).squeeze(1)                          # (N,)
            logit = s[A] - s[B]                                 # Bradley-Terry: σ(R_a - R_b)
            loss = bce(logit, P)
            loss.backward(); opt.step()
        return float(loss.item())

    def score(self, emb: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            return self.net(torch.tensor(np.asarray(emb, np.float32))).squeeze(1).cpu().numpy()


def train_from_preferences(pref_path: str, clip_dir: str, epochs: int = 400):
    """Train the BT reward model from a preferences.jsonl (reward_schema.Preference rows, e.g. from
    preference_annotator.py) over a pool of clip wavs. 'tie' rows are dropped. Returns (model, stats)."""
    import glob  # noqa: F401
    import json
    import soundfile as sf
    from q_head import make_embed_fn

    emb_fn = make_embed_fn()
    prefs = [json.loads(ln) for ln in open(pref_path) if ln.strip()]
    names = sorted({p["segment_a"] for p in prefs} | {p["segment_b"] for p in prefs})
    idx = {n: i for i, n in enumerate(names)}
    emb = []
    for n in names:
        y, sr = sf.read(os.path.join(clip_dir, n)); y = y.mean(1) if getattr(y, "ndim", 1) > 1 else y
        emb.append(emb_fn(y, sr))
    emb = np.stack(emb)
    pairs, labels = [], []
    for p in prefs:
        if p["preferred"] == "tie":
            continue
        pairs.append((idx[p["segment_a"]], idx[p["segment_b"]]))
        labels.append(1.0 if p["preferred"] == "a" else 0.0)
    if not pairs:
        return None, {"n_clips": len(names), "n_pairs": 0}
    model = BTReward()
    loss = model.fit(emb, np.array(pairs), np.array(labels), epochs=epochs)
    return model, {"n_clips": len(names), "n_pairs": len(pairs), "loss": loss,
                   "scores": {n: float(s) for n, s in zip(names, model.score(emb))}}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra @ rb) / (np.sqrt((ra @ ra) * (rb @ rb)) + 1e-12))


def _synthetic_validation():
    """Build clips at known quality (clean→noisy), preferences prefer the cleaner one, train BT on a
    subset of BASE clips, test the learned ranking on UNSEEN base clips."""
    import glob
    import soundfile as sf
    from q_head import make_embed_fn

    emb_fn = make_embed_fn()
    base_wavs = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "data",
                                              "audio", "_samples", "hf_lung_v1", "*.wav")))
    if len(base_wavs) < 4:
        print("need >=4 sample clips; found", len(base_wavs)); return
    rng = np.random.RandomState(0)
    noise = [0.0, 0.05, 0.1, 0.2, 0.35]
    items = []                                                  # (base_idx, gt_quality, embedding)
    for bi, w in enumerate(base_wavs):
        y, sr = sf.read(w); y = y.mean(1) if y.ndim > 1 else y
        y = (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)
        for nl in noise:
            yn = y + nl * rng.randn(len(y)).astype(np.float32)
            items.append((bi, -nl, emb_fn(yn, sr)))            # gt_quality = -noise (cleaner=better)
    emb = np.stack([it[2] for it in items]); gt = np.array([it[1] for it in items])
    base = np.array([it[0] for it in items])
    n_base = len(base_wavs); test_bases = set(range(n_base))  # split by base clip
    tr_bases = set(list(test_bases)[: max(2, n_base - 2)]); te_bases = test_bases - tr_bases
    tr_idx = np.where(np.isin(base, list(tr_bases)))[0]; te_idx = np.where(np.isin(base, list(te_bases)))[0]

    # training preference pairs: within train items with a clear quality gap; prefer higher gt
    pairs, prefs = [], []
    for i in tr_idx:
        for j in tr_idx:
            if i < j and abs(gt[i] - gt[j]) > 1e-3:
                pairs.append((i, j)); prefs.append(1.0 if gt[i] > gt[j] else 0.0)
    pairs = np.array(pairs); prefs = np.array(prefs)

    model = BTReward()
    loss = model.fit(emb, pairs, prefs, epochs=400)
    sc = model.score(emb)
    # held-out ranking quality on UNSEEN base clips
    rho = _spearman(sc[te_idx], gt[te_idx])
    # pairwise accuracy on unseen items
    acc, tot = 0, 0
    for i in te_idx:
        for j in te_idx:
            if i < j and abs(gt[i] - gt[j]) > 1e-3:
                tot += 1; acc += int((sc[i] > sc[j]) == (gt[i] > gt[j]))
    print(f"BT reward: train loss={loss:.4f} | HELD-OUT(unseen clips) spearman(score,quality)={rho:.3f} "
          f"pairwise_acc={acc/max(tot,1)*100:.0f}% (n={tot})")
    print("PASS" if rho > 0.6 and acc / max(tot, 1) > 0.75 else "WEAK")


if __name__ == "__main__":
    _synthetic_validation()
