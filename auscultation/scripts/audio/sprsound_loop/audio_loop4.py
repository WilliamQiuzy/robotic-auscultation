"""Iteration batch 4 — the KEY deployable artifact: a robustness-augmented WavLM quality model.
The clean-trained handcrafted model collapsed under noise (AUC 0.50) and hum (0.79). Here:
  (A) is WavLM itself more robust than handcrafted?  (B) does train-time hum+noise augmentation
  restore robustness while keeping clean AUC?  -> 2x3 AUC table (train regime x test condition).
Subsampled (~1800 clips) for the slow-CPU/GPU budget. Logs to /workspace/out/results.jsonl."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, "/workspace")
import audio_loop as AL
from audio_loop import load_audio, add_hum, add_noise, log, CACHE, SR   # noqa

d = np.load(CACHE, allow_pickle=True)
y = d["y"]; paths = list(d["paths"])
rng = np.random.RandomState(7)
pj = [i for i in range(len(y)) if y[i] == 1]
uj = [i for i in range(len(y)) if y[i] == 0]; rng.shuffle(uj); uj = uj[:1340]
sub = np.array(pj + uj); ys = y[sub]; sp = [paths[i] for i in sub]


def wavlm_corrupt(paths_, corrupt=None):
    import torch
    if "m" not in AL._WAVLM:
        AL.wavlm_embed(paths_[:1])                       # init model once
    fe, m = AL._WAVLM["fe"], AL._WAVLM["m"]
    out = []; B = 16
    for i in range(0, len(paths_), B):
        batch = []
        for p in paths_[i:i + B]:
            yy = load_audio(p)
            if corrupt == "hum":
                yy = add_hum(yy, 0.1)
            elif corrupt == "noise":
                yy = add_noise(yy, 10)
            batch.append(yy.astype(np.float32))
        inp = fe(batch, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            h = m(inp.input_values.cuda()).last_hidden_state.mean(1)
        out.append(h.cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def main():
    t0 = time.time()
    Wc = wavlm_corrupt(sp, None)
    Wh = wavlm_corrupt(sp, "hum")
    Wn = wavlm_corrupt(sp, "noise")
    log({"stage": "wavlm_robust_feats", "n": len(sub), "secs": round(time.time() - t0)})
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    # 5-fold: same split across conditions; train regimes = clean-only vs aug(clean+hum+noise)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {r: {c: [] for c in ("clean", "hum", "noise")} for r in ("clean_train", "aug_train")}
    for tr, te in skf.split(Wc, ys):
        # clean-trained
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08).fit(Wc[tr], ys[tr])
        for c, W in (("clean", Wc), ("hum", Wh), ("noise", Wn)):
            res["clean_train"][c].append(roc_auc_score(ys[te], clf.predict_proba(W[te])[:, 1]))
        # aug-trained (clean+hum+noise train)
        Xtr = np.vstack([Wc[tr], Wh[tr], Wn[tr]]); ytr = np.concatenate([ys[tr], ys[tr], ys[tr]])
        clf2 = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08).fit(Xtr, ytr)
        for c, W in (("clean", Wc), ("hum", Wh), ("noise", Wn)):
            res["aug_train"][c].append(roc_auc_score(ys[te], clf2.predict_proba(W[te])[:, 1]))
    for regime in res:
        for c in res[regime]:
            log({"stage": "wavlm_robust", "train": regime, "test_cond": c,
                 "auc": round(float(np.mean(res[regime][c])), 3)})
    print("STAGE_COMPLETE batch4", flush=True)


if __name__ == "__main__":
    main()
