#!/usr/bin/env python3
"""Plan A 最后两根杠杆 — CE 集成 + bandpass 前处理。用完即认定 A 走到极限。"""
import json, os, sys, glob, re, wave
import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, "/workspace/code/auscultation/ausc/audio"); sys.path.insert(0, "/workspace/OPERA")
import proxies as P  # noqa
from src.benchmark.model_util import extract_opera_feature, get_encoder_path  # noqa
CACHE = "/workspace/data/audio/_q_v1"; TMP = "/tmp/_q2win"; BP = "/tmp/_q2bp"; SR = 16000
os.makedirs(BP, exist_ok=True)
gold = json.load(open("/workspace/data/audio/_doctor/gold_merged.json"))
wavs = sorted(glob.glob(f"{TMP}/c*_w*.wav"), key=lambda p: (int(re.search(r"c(\d+)_", p).group(1)), int(re.search(r"_w(\d+)", p).group(1))))
owner = np.array([int(re.search(r"c(\d+)_", p).group(1)) for p in wavs])


def meanpool(F):
    return np.array([F[owner == ci].mean(0) for ci in range(len(gold))])


CT = meanpool(np.load(f"{CACHE}/winfeat_operaCT.npy"))

# CE 集成
sets = {"CT": CT}
try:
    cf = f"{CACHE}/winfeat_operaCE.npy"
    F = np.load(cf) if os.path.exists(cf) and np.load(cf).shape[0] == len(wavs) else None
    if F is None:
        get_encoder_path("operaCE"); F = np.array(extract_opera_feature(wavs, pretrain="operaCE", input_sec=8, dim=1280)); np.save(cf, F)
    CE = meanpool(F); sets["CT+CE"] = np.hstack([CT, CE]); print("CE ok", F.shape, flush=True)
except Exception as e:
    print("CE failed:", repr(e)[:120], flush=True)

# bandpass 前处理 → 重抽 CT
try:
    b, a = butter(4, [100 / (SR / 2), 2000 / (SR / 2)], btype="band")
    for w in wavs:
        xn, srn = P.read_wav(w); x, _ = P.resample(xn, srn, SR)
        y = filtfilt(b, a, x).astype(np.float32)
        pcm = (np.clip(y / (np.max(np.abs(y)) + 1e-9) * 0.9, -1, 1) * 32767).astype("<i2")
        with wave.open(f"{BP}/{os.path.basename(w)}", "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(pcm.tobytes())
    bpw = [f"{BP}/{os.path.basename(w)}" for w in wavs]
    cf = f"{CACHE}/winfeat_operaCT_bp.npy"
    F = np.load(cf) if os.path.exists(cf) else np.array(extract_opera_feature(bpw, pretrain="operaCT", input_sec=8, dim=768))
    np.save(cf, F); sets["CT_bandpass"] = meanpool(F); print("bandpass ok", flush=True)
except Exception as e:
    print("bandpass failed:", repr(e)[:120], flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
cons = [g["consensus"] for g in gold]
idx = [i for i in range(len(gold)) if cons[i] in ("usable", "unusable")]
y = np.array([1 if cons[i] == "usable" else 0 for i in idx])


def cvauc(X):
    return float(np.mean([roc_auc_score(y, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)),
                          X[idx], y, cv=StratifiedKFold(5, shuffle=True, random_state=sd), method="predict_proba")[:, 1]) for sd in range(8)]))


print("=== final A exhaustion (clear-consensus binary AUC) ===", flush=True)
res = {k: round(cvauc(X), 3) for k, X in sets.items()}
for k, v in res.items():
    print(f"  {k:14s} {v:.3f}", flush=True)
print(f"[A exhaust done] {json.dumps(res)}  (window-mean CT was 0.736)", flush=True)
