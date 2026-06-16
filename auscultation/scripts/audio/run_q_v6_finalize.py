#!/usr/bin/env python3
"""Q v2 固化 — 窗级 OPERA-CT + bandpass(100-2000Hz) + 校准头(clear-consensus 0.794)。
全套评测(binary / soft-ordinal / strong-agree)+ 存模型。用 q5 缓存的 bandpass 特征(秒级)。"""
import json, os, sys, glob, re
import numpy as np

sys.path.insert(0, "/workspace/code/auscultation/ausc/audio")
CACHE = "/workspace/data/audio/_q_v1"; TMP = "/tmp/_q2win"; OUT = "/workspace/data/audio/_q_v2"
os.makedirs(OUT, exist_ok=True)
gold = json.load(open("/workspace/data/audio/_doctor/gold_merged.json"))
wavs = sorted(glob.glob(f"{TMP}/c*_w*.wav"), key=lambda p: (int(re.search(r"c(\d+)_", p).group(1)), int(re.search(r"_w(\d+)", p).group(1))))
owner = np.array([int(re.search(r"c(\d+)_", p).group(1)) for p in wavs])
F = np.load(f"{CACHE}/winfeat_operaCT_bp.npy")
X = np.array([F[owner == ci].mean(0) for ci in range(len(gold))])

s = np.array([g["sungeun"] for g in gold]); j = np.array([g["jason"] for g in gold]); mean_ov = (s + j) / 2
cons = [g["consensus"] for g in gold]
idx = [i for i in range(len(gold)) if cons[i] in ("usable", "unusable")]
yb = np.array([1 if cons[i] == "usable" else 0 for i in idx])
strong = [i for i in range(len(gold)) if (s[i] >= 2 and j[i] >= 2) or (s[i] <= 1 and j[i] <= 1)]
ys = np.array([1 if (s[i] >= 2 and j[i] >= 2) else 0 for i in strong])

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import spearmanr
import joblib

LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
cvauc = lambda Xi, y: float(np.mean([roc_auc_score(y, cross_val_predict(LR(), Xi, y, cv=StratifiedKFold(5, shuffle=True, random_state=sd), method="predict_proba")[:, 1]) for sd in range(8)]))

bin_auc = cvauc(X[idx], yb)
strong_auc = cvauc(X[strong], ys)
pred = np.mean([cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=50.0)), X, mean_ov, cv=KFold(5, shuffle=True, random_state=sd)) for sd in range(8)], axis=0)
both_usable = np.array([1 if (s[i] >= 2 and j[i] >= 2) else 0 for i in range(len(gold))])
soft_sp = spearmanr(pred, mean_ov).correlation; soft_auc = roc_auc_score(both_usable, pred)

final = make_pipeline(StandardScaler(), CalibratedClassifierCV(LogisticRegression(C=0.05, max_iter=5000), method="isotonic", cv=5)).fit(X[idx], yb)
joblib.dump({"model": final, "encoder": "operaCT", "preprocess": "bandpass_100_2000",
             "agg": "window_mean", "win_s": 8, "hop_s": 4}, f"{OUT}/q_v2.joblib")
rep = {"clear_consensus_auc": round(bin_auc, 3), "strong_agree_auc": round(strong_auc, 3),
       "soft_spearman": round(float(soft_sp), 3), "soft_auc_usable": round(float(soft_auc), 3),
       "n_clear": len(idx), "n_strong": len(strong), "q_v1_baseline": 0.72, "preprocess": "bandpass100-2000"}
json.dump(rep, open(f"{OUT}/q_v2_report.json", "w"), indent=2)
print(f"[Q v2 finalized] {json.dumps(rep)}", flush=True)
