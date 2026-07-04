"""Iteration batch 3 — deployment-critical + interpretability:
  - HUM-SEVERITY: inject 60Hz hum at increasing levels into USABLE clips; does the quality model's
    P(poor) rise? (i.e., will it flag our real 60Hz-hummed recordings as low quality?)
  - FEATURE IMPORTANCE: which handcrafted spectral features define 'Poor Quality'.
  - HUM-AWARE TRAINING: relabel heavily-hummed usable clips as poor -> retrain -> does it flag hum
    without losing base AUC.
Reuses feats.npz. Logs to /workspace/out/results.jsonl."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, "/workspace")
from audio_loop import load_audio, feats_handcrafted, log, CACHE, OUT, SR   # noqa

d = np.load(CACHE, allow_pickle=True)
y = d["y"]; paths = list(d["paths"]); Xh = d["Xh_band"]
FNAMES = ([f"mel_mean_{i}" for i in range(64)] + [f"mel_std_{i}" for i in range(64)] +
          ["cent_m", "cent_s", "bw_m", "bw_s", "roll_m", "roll_s", "flat_m", "flat_s",
           "zcr_m", "zcr_s", "rms"] + [f"bandratio_{b}" for b in
           ["0-100", "100-250", "250-600", "600-1200", "1200-2500", "2500+"]])


def hum_band(p, level):
    from scipy.signal import butter, filtfilt
    yy = load_audio(p)                                  # raw
    t = np.arange(len(yy)) / SR
    yy = yy + level * sum(np.sin(2 * np.pi * 60 * k * t) / k for k in range(1, 5)).astype(np.float32)
    b, a = butter(4, [90 / (SR / 2), 1100 / (SR / 2)], "bandpass")
    return filtfilt(b, a, yy).astype(np.float32)


def train_clean():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(Xh, y)


def hum_severity():
    clf = train_clean()
    usable = [i for i in range(len(y)) if y[i] == 0]
    rng = np.random.RandomState(1); rng.shuffle(usable); usable = usable[:300]
    for level in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
        X = np.stack([feats_handcrafted(hum_band(paths[i], level)) for i in usable])
        pmean = float(clf.predict_proba(X)[:, 1].mean())
        log({"stage": "hum_severity", "hum_level": level, "mean_P_poor": round(pmean, 3),
             "flagged_poor@0.5": round(float((clf.predict_proba(X)[:, 1] >= 0.5).mean()), 3)})


def feat_importance():
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(Xh[tr], y[tr])
    imp = permutation_importance(clf, Xh[te], y[te], n_repeats=5, random_state=0, scoring="roc_auc")
    order = np.argsort(imp.importances_mean)[::-1][:12]
    top = [(FNAMES[i] if i < len(FNAMES) else f"f{i}", round(float(imp.importances_mean[i]), 4)) for i in order]
    log({"stage": "feat_importance", "top12": top})


def hum_aware_training():
    # relabel usable clips that (with injected hum) look poor -> add as poor examples; check base AUC + hum flagging
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)
    usable_tr = [i for i in tr if y[i] == 0]
    rng = np.random.RandomState(2); rng.shuffle(usable_tr); aug = usable_tr[:400]
    Xaug = np.stack([feats_handcrafted(hum_band(paths[i], 0.2)) for i in aug])   # heavy hum = poor
    Xtr = np.vstack([Xh[tr], Xaug]); ytr = np.concatenate([y[tr], np.ones(len(aug), int)])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(Xtr, ytr)
    base_auc = roc_auc_score(y[te], clf.predict_proba(Xh[te])[:, 1])
    # does it now flag hummed usable test clips as poor?
    ute = [i for i in te if y[i] == 0][:200]
    Xhum = np.stack([feats_handcrafted(hum_band(paths[i], 0.2)) for i in ute])
    flag = float((clf.predict_proba(Xhum)[:, 1] >= 0.5).mean())
    log({"stage": "hum_aware", "base_auc_kept": round(float(base_auc), 3), "hum_clips_flagged_poor": round(flag, 3)})


if __name__ == "__main__":
    feat_importance()
    hum_severity()
    hum_aware_training()
    print("STAGE_COMPLETE batch3", flush=True)
