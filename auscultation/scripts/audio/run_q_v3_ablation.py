#!/usr/bin/env python3
"""Q v3 消融 — 把 Plan A 走到极限(现有 90 条数据上)。
- 窗级特征(8s/hop4s, 覆盖全 clip) × 3 个 OPERA 变体(CT/GT/CE)→ mean-pool 聚合(缓存)。
- 编码器单用 + 集成(CT+GT+CE 拼接)。
- 分类器: logreg / svm-rbf / gbm。
- 评测两套: 全 clear-consensus(68) + 强一致子集(both>=2 vs both<=1)。
"""
import json, os, sys, glob, re, wave
import numpy as np

sys.path.insert(0, "/workspace/code/auscultation/ausc/audio")
sys.path.insert(0, "/workspace/OPERA")
import proxies as P  # noqa: E402
from src.benchmark.model_util import extract_opera_feature, get_encoder_path  # noqa: E402

CLIPS = "/workspace/data/audio/_doctor/clips"
TMP = "/tmp/_q2win"; CACHE = "/workspace/data/audio/_q_v1"
os.makedirs(TMP, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
SR = 16000; WIN = 8 * SR; HOP = 4 * SR
gold = json.load(open("/workspace/data/audio/_doctor/gold_merged.json"))


def windows_of(wav):
    xn, srn = P.read_wav(wav); x, _ = P.resample(xn, srn, SR)
    if len(x) <= WIN:
        return [x]
    out, i = [], 0
    while i + WIN <= len(x):
        out.append(x[i:i + WIN]); i += HOP
    if i < len(x):
        out.append(x[-WIN:])
    return out


def writew(x, p):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(p, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(pcm.tobytes())


# 重建窗 temp wav(若缺)
if len(glob.glob(f"{TMP}/c*_w*.wav")) < 100:
    for ci, g in enumerate(gold):
        for wi, w in enumerate(windows_of(f"{CLIPS}/{g['clip']}.wav")):
            writew(w, f"{TMP}/c{ci}_w{wi}.wav")
wavs = sorted(glob.glob(f"{TMP}/c*_w*.wav"),
              key=lambda p: (int(re.search(r"c(\d+)_", p).group(1)), int(re.search(r"_w(\d+)", p).group(1))))
owner = np.array([int(re.search(r"c(\d+)_", p).group(1)) for p in wavs])
print(f"windows={len(wavs)} clips={len(gold)}", flush=True)

# 3 个 OPERA 变体窗级特征(缓存)
feats = {}
for pt, dim in [("operaCT", 768), ("operaGT", 384), ("operaCE", 1280)]:
    cf = f"{CACHE}/winfeat_{pt}.npy"
    if os.path.exists(cf) and np.load(cf).shape[0] == len(wavs):
        F = np.load(cf)
    else:
        get_encoder_path(pt)
        F = np.array(extract_opera_feature(wavs, pretrain=pt, input_sec=8, dim=dim)); np.save(cf, F)
    feats[pt] = F; print(f"{pt} {F.shape}", flush=True)

agg = {pt: np.array([F[owner == ci].mean(0) for ci in range(len(gold))]) for pt, F in feats.items()}

s = np.array([g["sungeun"] for g in gold]); j = np.array([g["jason"] for g in gold])
cons = [g["consensus"] for g in gold]
idx = [i for i in range(len(gold)) if cons[i] in ("usable", "unusable")]
yb = np.array([1 if cons[i] == "usable" else 0 for i in idx])
strong = [i for i in range(len(gold)) if (s[i] >= 2 and j[i] >= 2) or (s[i] <= 1 and j[i] <= 1)]
ys = np.array([1 if (s[i] >= 2 and j[i] >= 2) else 0 for i in strong])
print(f"clear-consensus n={len(idx)} (usable {int(yb.sum())}); strong-agree n={len(strong)} (good {int(ys.sum())})", flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

clfs = {"logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)),
        "svm": lambda: make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, gamma="scale")),
        "gbm": lambda: GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05)}


def cvauc(X, y, mk, reps=6):
    a = []
    for sd in range(reps):
        pp = cross_val_predict(mk(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=sd), method="predict_proba")[:, 1]
        a.append(roc_auc_score(y, pp))
    return float(np.mean(a))


FS = {"CT": agg["operaCT"], "GT": agg["operaGT"], "CE": agg["operaCE"],
      "CT+GT+CE": np.hstack([agg["operaCT"], agg["operaGT"], agg["operaCE"]])}
for title, (ix, yy) in [("clear-consensus (n=68)", (idx, yb)), ("strong-agreement", (strong, ys))]:
    print(f"\n=== {title} ===", flush=True)
    best = ("", 0)
    for fn, X in FS.items():
        for cn, mk in clfs.items():
            a = cvauc(X[ix], yy, mk)
            if a > best[1]:
                best = (f"{fn}/{cn}", a)
            print(f"  {fn:10s} {cn:7s} AUC {a:.3f}", flush=True)
    print(f"  BEST: {best[0]} = {best[1]:.3f}", flush=True)
print("\n[Q v3 ablation done]", flush=True)
