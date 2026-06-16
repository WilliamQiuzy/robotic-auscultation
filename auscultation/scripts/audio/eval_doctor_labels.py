#!/usr/bin/env python3
"""用真实医生标签评估自动质量评分: q_proxy(无训练基线) vs OPERA/LogMel 线性探针(交叉验证)。
目标: 双医生一致的 usable(42) / unusable(26) → 看谁能预测【临床质量】。

跑在 pod(需 OPERA + 90 个 clip wav + gold_merged.json)。
"""
import json, os, sys
import numpy as np

sys.path.insert(0, "/workspace/code/auscultation/ausc/audio")
sys.path.insert(0, "/workspace/OPERA")
import proxies as P  # noqa: E402
from encoders import build_encoder  # noqa: E402
from src.benchmark.model_util import extract_opera_feature  # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CLIPS = "/workspace/data/audio/_doctor/clips"
gold = json.load(open("/workspace/data/audio/_doctor/gold_merged.json"))
clear = [g for g in gold if g["consensus"] in ("usable", "unusable")]
paths = [f"{CLIPS}/{g['clip']}.wav" for g in clear]
y = np.array([1 if g["consensus"] == "usable" else 0 for g in clear])
qproxy = np.array([g["q_proxy"] for g in clear])
print(f"clear-consensus: {len(clear)} (usable {int(y.sum())}, unusable {int((1-y).sum())})", flush=True)


def cv_auc(X, y):
    aucs = []
    for seed in range(5):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        pp = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, pp))
    return float(np.mean(aucs))


print(f"[q_proxy   no-train baseline] AUC = {roc_auc_score(y, qproxy):.3f}", flush=True)
Xo = np.array(extract_opera_feature(paths, pretrain="operaCT", input_sec=8, dim=768))
enc = build_encoder("logmel_stats"); Xl = np.array([enc.embed(p) for p in paths])
oa, la = cv_auc(Xo, y), cv_auc(Xl, y)
print(f"[OPERA-CT  5-fold CV probe]  AUC = {oa:.3f}", flush=True)
print(f"[LogMel    5-fold CV probe]  AUC = {la:.3f}", flush=True)
print(f"[EVAL done] {json.dumps({'q_proxy': round(roc_auc_score(y,qproxy),3), 'opera_cv': round(oa,3), 'logmel_cv': round(la,3), 'n': len(clear)})}", flush=True)
