"""Batch 7 — does hum+noise augmentation give GENERAL robustness or only to seen corruptions?
Train AST (clean vs aug=hum+noise) -> test on UNSEEN corruptions (reverb, clipping, band-limit) +
seen (hum, noise) for reference. Subsample ~1800. Logs to /workspace/out/results.jsonl."""
import os, sys, time
import numpy as np
sys.path.insert(0, "/workspace")
from audio_loop import load_audio, add_hum, add_noise, log, CACHE, SR   # noqa

d = np.load(CACHE, allow_pickle=True)
y = d["y"]; paths = list(d["paths"])
rng = np.random.RandomState(7)
pj = [i for i in range(len(y)) if y[i] == 1]
uj = [i for i in range(len(y)) if y[i] == 0]; rng.shuffle(uj); uj = uj[:1340]
sub = np.array(pj + uj); ys = y[sub]; sp = [paths[i] for i in sub]
_A = {}


def reverb(yy):
    ir = (np.random.RandomState(0).randn(int(0.05 * SR)) * np.exp(-np.linspace(0, 6, int(0.05 * SR)))).astype(np.float32)
    out = np.convolve(yy, ir)[:len(yy)]
    return (out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)


def clip_dist(yy):
    return np.clip(yy * 4.0, -1, 1).astype(np.float32)


def bandlimit(yy):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [300 / (SR / 2), 3400 / (SR / 2)], "bandpass")
    return filtfilt(b, a, yy).astype(np.float32)


CORR = {"clean": None, "hum": lambda z: add_hum(z, 0.1), "noise": lambda z: add_noise(z, 10),
        "reverb*": reverb, "clip*": clip_dist, "bandlimit*": bandlimit}   # * = UNSEEN in training


def ast(paths_, fn=None):
    import torch
    if "m" not in _A:
        from transformers import AutoFeatureExtractor, ASTModel
        _A["fe"] = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        _A["m"] = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").cuda().eval()
    fe, m = _A["fe"], _A["m"]; out = []; B = 16
    for i in range(0, len(paths_), B):
        batch = [(fn(load_audio(p)) if fn else load_audio(p)).astype(np.float32) for p in paths_[i:i + B]]
        inp = fe(batch, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out.append(m(inp.input_values.cuda()).last_hidden_state.mean(1).cpu().numpy().astype(np.float32))
    return np.concatenate(out)


t0 = time.time()
X = {c: ast(sp, fn) for c, fn in CORR.items()}
log({"stage": "unseen_corruption_feats", "n": len(sub), "conds": list(CORR), "secs": round(time.time() - t0)})
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
res = {"clean_train": {c: [] for c in CORR}, "aug_train": {c: [] for c in CORR}}
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X["clean"], ys):
    c1 = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08).fit(X["clean"][tr], ys[tr])
    Xa = np.vstack([X["clean"][tr], X["hum"][tr], X["noise"][tr]]); ya = np.concatenate([ys[tr]] * 3)
    c2 = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08).fit(Xa, ya)
    for c in CORR:
        res["clean_train"][c].append(roc_auc_score(ys[te], c1.predict_proba(X[c][te])[:, 1]))
        res["aug_train"][c].append(roc_auc_score(ys[te], c2.predict_proba(X[c][te])[:, 1]))
for regime in res:
    for c in CORR:
        log({"stage": "unseen_robust", "train": regime, "test_cond": c,
             "auc": round(float(np.mean(res[regime][c])), 3)})
print("STAGE_COMPLETE batch7", flush=True)
