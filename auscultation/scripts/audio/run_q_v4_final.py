#!/usr/bin/env python3
"""Q v4 最终消融 + 固化 — Plan A 收尾(用 q3 缓存的窗级特征, 秒级)。
- A4 软标签: 目标=两医生平均 overall(0-3, 全 90 条)→ Ridge → Spearman + AUC。
- A4 强一致训练: 只在强一致子集上训(干净标签)→ 看是否泛化更好。
- A6 分类器 + 概率校准。
- 选最佳配置固化 Q v2(OPERA 集成 + 校准头)+ 存模型/报告。
"""
import json, os, sys, glob, re
import numpy as np

sys.path.insert(0, "/workspace/code/auscultation/ausc/audio")
CACHE = "/workspace/data/audio/_q_v1"; TMP = "/tmp/_q2win"; OUT = "/workspace/data/audio/_q_v2"
os.makedirs(OUT, exist_ok=True)
gold = json.load(open("/workspace/data/audio/_doctor/gold_merged.json"))

wavs = sorted(glob.glob(f"{TMP}/c*_w*.wav"),
              key=lambda p: (int(re.search(r"c(\d+)_", p).group(1)), int(re.search(r"_w(\d+)", p).group(1))))
owner = np.array([int(re.search(r"c(\d+)_", p).group(1)) for p in wavs])
avail = [pt for pt in ["operaCT", "operaGT", "operaCE"] if os.path.exists(f"{CACHE}/winfeat_{pt}.npy")]
feats = {pt: np.load(f"{CACHE}/winfeat_{pt}.npy") for pt in avail}
agg = {pt: np.array([F[owner == ci].mean(0) for ci in range(len(gold))]) for pt, F in feats.items()}
ENS = np.hstack([agg[pt] for pt in avail])
print(f"encoders used: {avail} → feature dim {ENS.shape[1]}", flush=True)

s = np.array([g["sungeun"] for g in gold]); j = np.array([g["jason"] for g in gold])
mean_ov = (s + j) / 2.0
cons = [g["consensus"] for g in gold]
idx = [i for i in range(len(gold)) if cons[i] in ("usable", "unusable")]
yb = np.array([1 if cons[i] == "usable" else 0 for i in idx])
strong = [i for i in range(len(gold)) if (s[i] >= 2 and j[i] >= 2) or (s[i] <= 1 and j[i] <= 1)]
ys = np.array([1 if (s[i] >= 2 and j[i] >= 2) else 0 for i in strong])

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr

LOGREG = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
SVM = lambda: make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, gamma="scale"))


def cvauc(X, y, mk, reps=8):
    return float(np.mean([roc_auc_score(y, cross_val_predict(mk(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=sd),
                          method="predict_proba")[:, 1]) for sd in range(reps)]))


print("=== A4 binary, clear-consensus (ENS) ===", flush=True)
print(f"  logreg {cvauc(ENS[idx], yb, LOGREG):.3f}   svm {cvauc(ENS[idx], yb, SVM):.3f}", flush=True)

print("=== A4 soft ordinal (mean doctor 0-3, ALL 90, Ridge) ===", flush=True)
pred = np.zeros(len(gold))
for sd in range(8):
    pred += cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=50.0)), ENS, mean_ov,
                              cv=KFold(5, shuffle=True, random_state=sd))
pred /= 8
sp = spearmanr(pred, mean_ov).correlation
both_usable = np.array([1 if (s[i] >= 2 and j[i] >= 2) else 0 for i in range(len(gold))])
auc_soft = roc_auc_score(both_usable, pred)
print(f"  Spearman(pred, mean_doctor)={sp:.3f}   AUC(pred → both-usable)={auc_soft:.3f}", flush=True)

print("=== A4 train-on-strong-agree, eval on full clear-consensus (LOGREG) ===", flush=True)
# 在强一致上训, 在 full clear-consensus 的留出上评(用 strong∩idx 做 CV 训练源)
clf = LOGREG().fit(ENS[strong], ys)
# honest: CV within strong
print(f"  strong-subset CV AUC: {cvauc(ENS[strong], ys, LOGREG):.3f}  (n={len(strong)}, good {int(ys.sum())})", flush=True)

# 固化: ENS + 校准 logreg, 用 binary clear-consensus 训
from sklearn.calibration import CalibratedClassifierCV
import joblib
final = make_pipeline(StandardScaler(), CalibratedClassifierCV(
    LogisticRegression(C=0.05, max_iter=5000), method="isotonic", cv=5)).fit(ENS[idx], yb)
joblib.dump({"model": final, "encoders": avail,
             "agg": "window_mean", "win_s": 8, "hop_s": 4}, f"{OUT}/q_v2.joblib")
report = {"binary_clearconsensus_auc": round(cvauc(ENS[idx], yb, LOGREG), 3),
          "soft_spearman": round(float(sp), 3), "soft_auc_usable": round(float(auc_soft), 3),
          "strong_agree_auc": round(cvauc(ENS[strong], ys, LOGREG), 3),
          "n_clear": len(idx), "n_strong": len(strong), "q_v1_baseline": 0.72}
json.dump(report, open(f"{OUT}/q_v2_report.json", "w"), indent=2)
print(f"\n[Q v2 final] {json.dumps(report)}", flush=True)
