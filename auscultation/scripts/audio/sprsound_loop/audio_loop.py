"""Autonomous lung-sound QUALITY loop on SPRSound (open, 'Poor Quality' labels).

Task: binary quality = Poor Quality (bad) vs usable (Normal/CAS/DAS/CAS&DAS) — the open-data
analogue of our doctor 'usable' target. Pipeline: cache features (handcrafted SQI + WavLM
embedding, optional 90-1100Hz lung band) -> sweep configs -> repeated-CV AUC -> 60Hz-hum/noise
robustness study -> keep best. Resumable (feature cache). Logs to /workspace/out.

Stages:  python audio_loop.py features   # extract + cache (long, GPU)
         python audio_loop.py sweep      # config sweep on cached feats (fast)
         python audio_loop.py robust      # 60Hz-hum / noise robustness study
         python audio_loop.py all         # features -> sweep -> robust
"""
import os, sys, json, glob, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = "/workspace/data/SPRSound"; OUT = "/workspace/out"; os.makedirs(OUT, exist_ok=True)
SR = 16000
MAX_USABLE = int(os.environ.get("MAX_USABLE", "4500"))   # cap usable for a balanced-ish, fast first pass
CACHE = f"{OUT}/feats.npz"
LOG = f"{OUT}/results.jsonl"


def log(rec):
    rec["t"] = time.strftime("%H:%M:%S")
    open(LOG, "a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


# ---------------- 1. label table ----------------
def build_table():
    wavs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(f"{ROOT}/**/*.wav", recursive=True)}
    rows = []
    for jp in glob.glob(f"{ROOT}/**/*.json", recursive=True):
        stem = os.path.splitext(os.path.basename(jp))[0]
        if stem not in wavs:
            continue
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        ra = d.get("record_annotation") if isinstance(d, dict) else None
        if ra is None:
            continue
        y_poor = 1 if ra.strip().lower() == "poor quality" else 0
        rows.append((wavs[stem], y_poor))
    poor = [r for r in rows if r[1] == 1]
    good = [r for r in rows if r[1] == 0]
    rng = np.random.RandomState(0); rng.shuffle(good)
    good = good[:MAX_USABLE]
    tab = poor + good
    rng.shuffle(tab)
    return tab, len(poor), len(good)


# ---------------- 2. audio + features ----------------
def load_audio(path, band=None):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < SR // 2:
        y = np.pad(y, (0, SR // 2 - len(y)))
    y = y[: SR * 12]                      # cap 12s
    if band:
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [band[0] / (SR / 2), band[1] / (SR / 2)], "bandpass")
        y = filtfilt(b, a, y).astype(np.float32)
    m = np.max(np.abs(y)) + 1e-9
    return (y / m).astype(np.float32)


def feats_handcrafted(y):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=64)
    lm = librosa.power_to_db(mel + 1e-9)
    f = [lm.mean(1), lm.std(1)]                                  # 128
    for fn in (librosa.feature.spectral_centroid, librosa.feature.spectral_bandwidth,
               librosa.feature.spectral_rolloff, librosa.feature.spectral_flatness):
        v = fn(S=S) if fn is librosa.feature.spectral_flatness else fn(S=S, sr=SR)
        f.append([v.mean(), v.std()])
    z = librosa.feature.zero_crossing_rate(y); f.append([z.mean(), z.std()])
    f.append([np.sqrt(np.mean(y ** 2))])                        # rms
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    bands = [(0, 100), (100, 250), (250, 600), (600, 1200), (1200, 2500), (2500, 8000)]
    tot = S.sum() + 1e-9
    f.append([S[(freqs >= lo) & (freqs < hi)].sum() / tot for lo, hi in bands])  # band ratios
    return np.concatenate([np.atleast_1d(x).ravel() for x in f]).astype(np.float32)


_WAVLM = {}
def wavlm_embed(batch_paths, band=None):
    import torch
    if "m" not in _WAVLM:
        from transformers import AutoFeatureExtractor, WavLMModel
        _WAVLM["fe"] = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
        _WAVLM["m"] = WavLMModel.from_pretrained("microsoft/wavlm-base").cuda().eval()
    fe, m = _WAVLM["fe"], _WAVLM["m"]
    ys = [load_audio(p, band) for p in batch_paths]
    inp = fe(ys, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        h = m(inp.input_values.cuda()).last_hidden_state       # (B,T,768)
    return h.mean(1).cpu().numpy().astype(np.float32)


def stage_features():
    tab, npoor, ngood = build_table()
    log({"stage": "table", "poor": npoor, "usable_sampled": ngood, "total": len(tab)})
    paths = [p for p, _ in tab]; y = np.array([l for _, l in tab])
    # handcrafted, two bands
    Xh_full = np.stack([feats_handcrafted(load_audio(p)) for p in paths])
    Xh_band = np.stack([feats_handcrafted(load_audio(p, (90, 1100))) for p in paths])
    log({"stage": "handcrafted_done", "dim": int(Xh_full.shape[1])})
    # wavlm (batched)
    Xw = []
    B = 16
    t0 = time.time()
    for i in range(0, len(paths), B):
        Xw.append(wavlm_embed(paths[i:i + B]))
        if i % 320 == 0:
            print(f"  wavlm {i}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
    Xw = np.concatenate(Xw)
    log({"stage": "wavlm_done", "dim": int(Xw.shape[1]), "secs": round(time.time() - t0)})
    np.savez_compressed(CACHE, y=y, Xh_full=Xh_full, Xh_band=Xh_band, Xw=Xw,
                        paths=np.array(paths))
    log({"stage": "features_cached", "path": CACHE})


# ---------------- 3. eval ----------------
def cv_auc(X, y, head="logreg", reps=1):
    # FAST: single 5-fold out-of-fold preds + bootstrap CI (pod CPU is slow; keeps iterations ~1min)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        if head == "logreg":
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08).fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    rng = np.random.RandomState(0); bs = []
    for _ in range(200):
        idx = rng.randint(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            bs.append(roc_auc_score(y[idx], oof[idx]))
    return float(auc), float(np.std(bs)), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def stage_sweep():
    d = np.load(CACHE, allow_pickle=True); y = d["y"]
    feats = {"handcrafted_full": d["Xh_full"], "handcrafted_band": d["Xh_band"], "wavlm": d["Xw"],
             "wavlm+band_hc": np.hstack([d["Xw"], d["Xh_band"]])}
    best = None
    for name, X in feats.items():
        for head in ("logreg", "gbm"):
            m, s, ci = cv_auc(X, y, head)
            rec = {"stage": "sweep", "feat": name, "head": head, "dim": int(X.shape[1]),
                   "auc": round(m, 3), "auc_std": round(s, 3), "ci95": [round(ci[0], 3), round(ci[1], 3)]}
            log(rec)
            if best is None or m > best["auc"]:
                best = rec
    json.dump(best, open(f"{OUT}/best.json", "w"), indent=2)
    log({"stage": "sweep_done", "best": best})


# ---------------- 4. robustness (60Hz hum / noise) ----------------
def add_hum(y, level=0.05):
    t = np.arange(len(y)) / SR
    h = sum(np.sin(2 * np.pi * 60 * k * t) / k for k in range(1, 5))
    return y + level * h.astype(np.float32)


def add_noise(y, snr_db=10):
    p = np.mean(y ** 2); n = np.random.randn(len(y)).astype(np.float32)
    n *= np.sqrt(p / (10 ** (snr_db / 10)) / (np.mean(n ** 2) + 1e-9))
    return y + n


def stage_robust():
    # measure clean vs 60Hz-hum vs noise on wavlm+band_hc (the deployable feature); then hum-aug training
    d = np.load(CACHE, allow_pickle=True); y = d["y"]; paths = d["paths"]
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)

    def featset(idx, corrupt=None):
        Xh = [];
        for i in idx:
            yy = load_audio(paths[i], (90, 1100))
            if corrupt == "hum": yy = load_audio_raw_then(paths[i], "hum")
            Xh.append(feats_handcrafted(yy))
        return np.stack(Xh)

    def load_audio_raw_then(p, c):
        yy = load_audio(p)                       # raw (no band) then corrupt then band
        yy = add_hum(yy) if c == "hum" else add_noise(yy)
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [90 / (SR / 2), 1100 / (SR / 2)], "bandpass")
        return filtfilt(b, a, yy).astype(np.float32)

    # clean-trained model, eval on clean / hum / noise (handcrafted_band for speed)
    Xtr = featset(tr); sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), y[tr])
    for cond in ("clean", "hum", "noise"):
        Xte = np.stack([feats_handcrafted(load_audio(paths[i], (90, 1100)) if cond == "clean"
                                          else load_audio_raw_then(paths[i], "hum" if cond == "hum" else "noise"))
                        for i in te])
        auc = roc_auc_score(y[te], clf.predict_proba(sc.transform(Xte))[:, 1])
        log({"stage": "robust_cleanmodel", "test_cond": cond, "auc": round(float(auc), 3)})
    # hum-augmented training -> recovery on hum test
    Xtr_aug = np.vstack([Xtr, np.stack([feats_handcrafted(load_audio_raw_then(paths[i], "hum")) for i in tr])])
    ytr_aug = np.concatenate([y[tr], y[tr]])
    sc2 = StandardScaler().fit(Xtr_aug)
    clf2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc2.transform(Xtr_aug), ytr_aug)
    Xte_hum = np.stack([feats_handcrafted(load_audio_raw_then(paths[i], "hum")) for i in te])
    auc_h = roc_auc_score(y[te], clf2.predict_proba(sc2.transform(Xte_hum))[:, 1])
    log({"stage": "robust_humaug", "test_cond": "hum", "auc": round(float(auc_h), 3)})
    log({"stage": "robust_done"})


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("features", "all"):
        stage_features()
    if st in ("sweep", "all"):
        stage_sweep()
    if st in ("robust", "all"):
        stage_robust()
    print("STAGE_COMPLETE", st, flush=True)
