#!/usr/bin/env python3
"""Q v2 实验 — 分窗 OPERA 嵌入 + 聚合(修 Q v1 只吃前 8s 截断长 clip 的问题)+ MIL。
对每条 clip 切 8s 窗(hop 4s)覆盖全长 → 每窗一个 OPERA 嵌入 → 聚合(mean/max/meanmax/top-k)。
"a clip is usable if it has clean breaths" → max/top-k 池化天然贴合。
和 Q v1 单嵌入基线(0.72)比。
"""
import json, os, sys, wave
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

CLIPS = "/workspace/data/audio/_doctor/clips"
TMP = "/tmp/_q2win"; os.makedirs(TMP, exist_ok=True)
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


paths, owner, nwin = [], [], []
for ci, g in enumerate(gold):
    ws = windows_of(f"{CLIPS}/{g['clip']}.wav"); nwin.append(len(ws))
    for wi, w in enumerate(ws):
        p = f"{TMP}/c{ci}_w{wi}.wav"; writew(w, p); paths.append(p); owner.append(ci)
owner = np.array(owner)
print(f"clips={len(gold)} windows={len(paths)} (avg {np.mean(nwin):.1f}/clip, max {max(nwin)})", flush=True)

F = np.array(extract_opera_feature(paths, pretrain="operaCT", input_sec=8, dim=768))


def agg(method):
    X = []
    for ci in range(len(gold)):
        wf = F[owner == ci]
        if method == "first":   X.append(wf[0])                                   # ≈ Q v1 基线
        elif method == "mean":  X.append(wf.mean(0))
        elif method == "max":   X.append(wf.max(0))
        elif method == "meanmax": X.append(np.concatenate([wf.mean(0), wf.max(0)]))
        elif method == "topk":  X.append(np.sort(wf, 0)[-2:].mean(0) if len(wf) >= 2 else wf.mean(0))
    return np.array(X)


idx = [i for i, g in enumerate(gold) if g["consensus"] in ("usable", "unusable")]
y = np.array([1 if gold[i]["consensus"] == "usable" else 0 for i in idx])


def cvauc(X, y):
    a = []
    for s in range(10):
        pp = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)),
                               X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), method="predict_proba")[:, 1]
        a.append(roc_auc_score(y, pp))
    return float(np.mean(a)), float(np.std(a))


print(f"clear-consensus n={len(y)} (usable {int(y.sum())})", flush=True)
res = {}
for m in ["first", "mean", "max", "meanmax", "topk"]:
    mu, sd = cvauc(agg(m)[idx], y); res[m] = round(mu, 3)
    print(f"  window-{m:8s} CV AUC {mu:.3f} ± {sd:.3f}", flush=True)
print(f"[Q v2 win-agg done] {json.dumps(res)}  (Q v1 baseline 0.72)", flush=True)
