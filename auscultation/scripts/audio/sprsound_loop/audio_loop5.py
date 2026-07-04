"""Batch 5 — HONEST generalization: patient-grouped split (no same-patient leakage across folds),
vs the random-split AUC. SPRSound filename = <patientID>_... . Uses cached features (fast)."""
import os, sys, json
import numpy as np
sys.path.insert(0, "/workspace")
from audio_loop import log, CACHE, OUT   # noqa
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

d = np.load(CACHE, allow_pickle=True)
y = d["y"]; paths = list(d["paths"]); Xw = d["Xw"]; Xh = d["Xh_band"]
Xa = np.load(f"{OUT}/ast.npy")
groups = np.array([os.path.basename(p).split("_")[0] for p in paths])
log({"stage": "patient_split", "n_clips": len(y), "n_patients": int(len(set(groups)))})

feats = {"handcrafted_band": Xh, "wavlm": Xw, "ast": Xa, "wavlm+ast": np.hstack([Xw, Xa])}
for name, X in feats.items():
    # patient-grouped (honest)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    g = roc_auc_score(y, oof)
    # random (leaky) for comparison
    oof2 = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08).fit(X[tr], y[tr])
        oof2[te] = clf.predict_proba(X[te])[:, 1]
    r = roc_auc_score(y, oof2)
    log({"stage": "patient_grouped_auc", "feat": name, "auc_patient": round(float(g), 3),
         "auc_random": round(float(r), 3), "leakage_gap": round(float(r - g), 3)})
print("STAGE_COMPLETE batch5", flush=True)
