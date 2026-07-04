"""Iteration batch 2 on the cached SPRSound features (from audio_loop.py):
  - AST encoder embedding (2nd pretrained audio model) + ensemble with WavLM
  - lung-band sweep (handcrafted)
  - cross-YEAR generalization (BioCAS2022->2023/24/25 domain shift) = the real generalization test
  - calibration (Brier/ECE) + operating point at target precision
Reuses feats.npz; caches AST. Logs to /workspace/out/results.jsonl."""
import os, sys, json, time, glob
import numpy as np
sys.path.insert(0, "/workspace")
from audio_loop import load_audio, feats_handcrafted, cv_auc, log, CACHE, OUT, SR   # noqa

d = np.load(CACHE, allow_pickle=True)
y = d["y"]; paths = list(d["paths"]); Xw = d["Xw"]; Xh_band = d["Xh_band"]


# ---- AST embedding ----
def ast_embed_all():
    cache = f"{OUT}/ast.npy"
    if os.path.exists(cache):
        return np.load(cache)
    import torch
    from transformers import AutoFeatureExtractor, ASTModel
    fe = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    m = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").cuda().eval()
    out = []; B = 16; t0 = time.time()
    for i in range(0, len(paths), B):
        ys = [load_audio(p) for p in paths[i:i + B]]
        inp = fe(ys, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            h = m(inp.input_values.cuda()).last_hidden_state.mean(1)
        out.append(h.cpu().numpy().astype(np.float32))
        if i % 640 == 0:
            print(f"  ast {i}/{len(paths)} {time.time()-t0:.0f}s", flush=True)
    Xa = np.concatenate(out); np.save(cache, Xa)
    log({"stage": "ast_done", "dim": int(Xa.shape[1]), "secs": round(time.time() - t0)})
    return Xa


def band_sweep():
    # subsample (slow-CPU re-extraction): all poor + equal usable ~= 900 clips
    rng = np.random.RandomState(3)
    pj = [i for i in range(len(y)) if y[i] == 1]
    uj = [i for i in range(len(y)) if y[i] == 0]; rng.shuffle(uj); uj = uj[:len(pj)]
    sub = pj + uj; ys = y[sub]; ps = [paths[i] for i in sub]
    for band in [(70, 1000), (90, 1100), (110, 1300), (90, 2000), (150, 1000), (60, 2500)]:
        X = np.stack([feats_handcrafted(load_audio(p, band)) for p in ps])
        m, s, ci = cv_auc(X, ys, "gbm")
        log({"stage": "band_sweep", "band": list(band), "n": len(sub), "auc": round(m, 3),
             "ci95": [round(ci[0], 3), round(ci[1], 3)]})


def ensembles(Xa):
    combos = {"wavlm": Xw, "ast": Xa, "wavlm+ast": np.hstack([Xw, Xa]),
              "wavlm+ast+hc": np.hstack([Xw, Xa, Xh_band])}
    best = None
    for name, X in combos.items():
        m, s, ci = cv_auc(X, y, "gbm")
        rec = {"stage": "ensemble", "feat": name, "dim": int(X.shape[1]), "auc": round(m, 3),
               "ci95": [round(ci[0], 3), round(ci[1], 3)]}
        log(rec)
        if best is None or m > best["auc"]:
            best = rec
    return best


def cross_year(Xbest):
    # BioCAS year from path = domain/recording shift
    def yr(p):
        for t in ("2022", "2023", "2024", "2025"):
            if t in p:
                return t
        return "?"
    years = np.array([yr(p) for p in paths])
    from collections import Counter
    log({"stage": "year_dist", "dist": dict(Counter(years))})
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    for test_year in ("2023", "2024", "2025"):
        tr = years != test_year; te = years == test_year
        if te.sum() < 20 or len(set(y[te])) < 2 or y[tr].sum() < 10:
            continue
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06).fit(Xbest[tr], y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(Xbest[te])[:, 1])
        log({"stage": "cross_year", "test_year": test_year, "n_test": int(te.sum()),
             "auc": round(float(auc), 3)})


def calibrate(Xbest):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(Xbest[tr], y[tr])
    p = clf.predict_proba(Xbest[te])[:, 1]
    auc = roc_auc_score(y[te], p); brier = brier_score_loss(y[te], p)
    # operating point: highest recall of POOR at precision>=0.6 (flagging bad clips)
    best_op = None
    for thr in np.linspace(0.1, 0.9, 33):
        pred = (p >= thr).astype(int)
        if pred.sum() == 0:
            continue
        pr = precision_score(y[te], pred); rc = recall_score(y[te], pred)
        if pr >= 0.6 and (best_op is None or rc > best_op[2]):
            best_op = (round(float(thr), 2), round(float(pr), 2), round(float(rc), 2))
    log({"stage": "calibrate", "auc": round(float(auc), 3), "brier": round(float(brier), 3),
         "op_thr_prec_rec": best_op})


if __name__ == "__main__":
    Xa = ast_embed_all()
    best = ensembles(Xa)
    Xbest = {"wavlm": Xw, "ast": Xa, "wavlm+ast": np.hstack([Xw, Xa]),
             "wavlm+ast+hc": np.hstack([Xw, Xa, Xh_band])}[best["feat"]]
    band_sweep()
    cross_year(Xbest)
    calibrate(Xbest)
    json.dump(best, open(f"{OUT}/best2.json", "w"), indent=2)
    print("STAGE_COMPLETE batch2", flush=True)
