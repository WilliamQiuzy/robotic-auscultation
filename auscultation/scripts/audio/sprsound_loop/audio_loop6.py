"""Batch 6 — finalize the recommended encoder (AST, best patient-level generalizer) as a ROBUST
deployable model: AST features under clean/hum/noise x clean-train vs augment-train. Mirrors batch4
but with AST. Subsampled ~1800 clips. Logs to /workspace/out/results.jsonl."""
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
_AST = {}


def ast_corrupt(paths_, corrupt=None):
    import torch
    if "m" not in _AST:
        from transformers import AutoFeatureExtractor, ASTModel
        _AST["fe"] = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        _AST["m"] = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").cuda().eval()
    fe, m = _AST["fe"], _AST["m"]
    out = []; B = 16
    for i in range(0, len(paths_), B):
        batch = []
        for p in paths_[i:i + B]:
            yy = load_audio(p)
            if corrupt == "hum": yy = add_hum(yy, 0.1)
            elif corrupt == "noise": yy = add_noise(yy, 10)
            batch.append(yy.astype(np.float32))
        inp = fe(batch, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            h = m(inp.input_values.cuda()).last_hidden_state.mean(1)
        out.append(h.cpu().numpy().astype(np.float32))
    return np.concatenate(out)


t0 = time.time()
Ac = ast_corrupt(sp, None); Ah = ast_corrupt(sp, "hum"); An = ast_corrupt(sp, "noise")
log({"stage": "ast_robust_feats", "n": len(sub), "secs": round(time.time() - t0)})
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
res = {r: {c: [] for c in ("clean", "hum", "noise")} for r in ("clean_train", "aug_train")}
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Ac, ys):
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08).fit(Ac[tr], ys[tr])
    for c, A in (("clean", Ac), ("hum", Ah), ("noise", An)):
        res["clean_train"][c].append(roc_auc_score(ys[te], clf.predict_proba(A[te])[:, 1]))
    Xtr = np.vstack([Ac[tr], Ah[tr], An[tr]]); ytr = np.concatenate([ys[tr]] * 3)
    clf2 = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08).fit(Xtr, ytr)
    for c, A in (("clean", Ac), ("hum", Ah), ("noise", An)):
        res["aug_train"][c].append(roc_auc_score(ys[te], clf2.predict_proba(A[te])[:, 1]))
for regime in res:
    for c in res[regime]:
        log({"stage": "ast_robust", "train": regime, "test_cond": c, "auc": round(float(np.mean(res[regime][c])), 3)})
print("STAGE_COMPLETE batch6", flush=True)
