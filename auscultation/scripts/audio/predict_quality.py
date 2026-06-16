#!/usr/bin/env python3
"""Q v1 推理 — 一条/多条肺音 wav → 临床质量可用概率 P(usable) + 判定。
模型 = 冻结 OPERA-CT 编码器 + 医生标签训的线性头(q_v1.joblib, CV AUC 0.72)。
部署质检门 v1: P>=阈值 → usable; 否则 NEEDS REVIEW(让机械臂重录/换位)。需 OPERA 仓库在 path。

  python scripts/audio/predict_quality.py --model /workspace/data/audio/_q_v1/q_v1.joblib clip.wav ...
"""
import argparse, json, os, sys
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ausc", "audio"))
sys.path.insert(0, "/workspace/OPERA")


def featurize(wavs, feature_set):
    from src.benchmark.model_util import extract_opera_feature
    Xo = np.array(extract_opera_feature(list(wavs), pretrain="operaCT", input_sec=8, dim=768))
    if feature_set == "OPERA":
        return Xo
    import proxies as P                                  # OPERA+qproxy 时补子分
    Xs = np.array([[P.analyze(w)["subscores"][k] for k in P._METRIC_KEY] for w in wavs])
    return np.hstack([Xo, Xs])


def predict(wavs, model_path):
    M = joblib.load(model_path)
    X = featurize(wavs, M["feature_set"])
    p = M["model"].predict_proba(X)[:, 1]
    thr = M["threshold"]
    return [{"wav": w, "p_usable": round(float(pi), 3),
             "verdict": "usable" if pi >= thr else "NEEDS_REVIEW"} for w, pi in zip(wavs, p)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/data/audio/_q_v1/q_v1.joblib")
    ap.add_argument("wavs", nargs="+")
    a = ap.parse_args()
    for r in predict(a.wavs, a.model):
        print(json.dumps(r, ensure_ascii=False))
