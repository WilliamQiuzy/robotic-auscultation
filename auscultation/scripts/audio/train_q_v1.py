#!/usr/bin/env python3
"""Q v1 — 用 90 条医生标签 + OPERA 特征训一个【临床质量】分类器(部署质检门 v1)。
数据少 → 强正则线性探针 + 重复交叉验证诚实报数; 固化最佳特征集, 存模型 + 阈值。
特征候选: OPERA(语义) / OPERA+q_proxy子分(语义+粗伪影互补) / q_proxy子分。
目标: 双医生一致 usable(42) vs unusable(26)。

跑在 pod。 输出: /workspace/data/audio/_q_v1/{q_v1.joblib, q_v1_report.json}
"""
import json, os, sys
import numpy as np

sys.path.insert(0, "/workspace/code/auscultation/ausc/audio")
sys.path.insert(0, "/workspace/OPERA")
import proxies as P  # noqa: E402
from src.benchmark.model_util import extract_opera_feature  # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

CLIPS = "/workspace/data/audio/_doctor/clips"
OUT = "/workspace/data/audio/_q_v1"; os.makedirs(OUT, exist_ok=True)
gold = json.load(open("/workspace/data/audio/_doctor/gold_merged.json"))
paths = [f"{CLIPS}/{g['clip']}.wav" for g in gold]

# 特征: OPERA(全 90) + q_proxy 子分(全 90)
Xo_all = np.array(extract_opera_feature(paths, pretrain="operaCT", input_sec=8, dim=768))
Xs_all = np.array([[P.analyze(p)["subscores"][k] for k in P._METRIC_KEY] for p in paths])

idx = [i for i, g in enumerate(gold) if g["consensus"] in ("usable", "unusable")]
y = np.array([1 if gold[i]["consensus"] == "usable" else 0 for i in idx])
feats = {"OPERA": Xo_all[idx], "OPERA+qproxy": np.hstack([Xo_all, Xs_all])[idx], "qproxy": Xs_all[idx]}
print(f"clear-consensus n={len(y)} (usable {int(y.sum())}, unusable {int((1-y).sum())})", flush=True)


def make_clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))


def repeated_cv(X, y, reps=10):
    aucs, oof = [], np.zeros((reps, len(y)))
    for s in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        pp = cross_val_predict(make_clf(), X, y, cv=skf, method="predict_proba")[:, 1]
        oof[s] = pp; aucs.append(roc_auc_score(y, pp))
    return float(np.mean(aucs)), float(np.std(aucs)), oof.mean(0)


results = {}
for name, X in feats.items():
    m, sd, _ = repeated_cv(X, y)
    results[name] = round(m, 3)
    print(f"[{name:14s}] CV AUC {m:.3f} ± {sd:.3f}", flush=True)

best = max(results, key=results.get)
Xb = feats[best]
_, _, oof = repeated_cv(Xb, y)
# 阈值: 最大化 Youden's J(sens+spec-1)
ths = np.linspace(0.1, 0.9, 81)
def sens_spec(t):
    pred = (oof >= t).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    return tp / max(1, tp + fn), tn / max(1, tn + fp)
J = [(t, *sens_spec(t)) for t in ths]
thr, sens, spec = max(J, key=lambda x: x[1] + x[2] - 1)

# 用最佳特征集在全部 clear-consensus 上拟合最终模型
final = make_clf().fit(Xb, y)
joblib.dump({"model": final, "feature_set": best, "threshold": round(float(thr), 3),
             "metric_keys": list(P._METRIC_KEY.keys())}, os.path.join(OUT, "q_v1.joblib"))
report = {"n": int(len(y)), "n_usable": int(y.sum()), "cv_auc_by_feature": results,
          "chosen_feature_set": best, "cv_auc": results[best],
          "threshold": round(float(thr), 3), "sensitivity": round(float(sens), 3),
          "specificity": round(float(spec), 3),
          "interrater_kappa_ref": 0.58, "q_proxy_baseline_auc": 0.636}
json.dump(report, open(os.path.join(OUT, "q_v1_report.json"), "w"), indent=2)
print(f"[Q v1 done] {json.dumps(report)}", flush=True)
