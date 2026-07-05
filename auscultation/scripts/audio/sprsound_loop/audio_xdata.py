"""Cross-dataset generalization of the lung-sound quality gate. Train on SPRSound, test transfer to
KAUH (King Abdullah Hospital) + COPD (different hospitals/devices).
  Test A: does the SPRSound real-quality model falsely rate a NEW domain's clean clips as poor? (domain shift)
  Test B: does SPRSound-trained CORRUPTION detection (clean vs hum+noise) transfer to KAUH/COPD? (AUC)
AST encoder (the winner). Logs to /workspace/out/xdata_results.jsonl."""
import os, sys, glob, json, time
import numpy as np
OUT = "/workspace/out"; os.makedirs(OUT, exist_ok=True); SR = 16000


def log(r):
    r["t"] = time.strftime("%H:%M:%S")
    open(f"{OUT}/xdata_results.jsonl", "a").write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True)


def load_audio(p):
    import librosa
    try:
        y, _ = librosa.load(p, sr=SR, mono=True)
    except Exception:
        return None
    if len(y) < SR // 2:
        y = np.pad(y, (0, SR // 2 - len(y)))
    y = y[: SR * 12]
    return (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def add_hum(y, l=0.08):
    t = np.arange(len(y)) / SR
    return (y + l * sum(np.sin(2 * np.pi * 60 * k * t) / k for k in range(1, 5))).astype(np.float32)


def add_noise(y, snr=12):
    p = np.mean(y ** 2); n = np.random.randn(len(y)).astype(np.float32)
    n *= np.sqrt(p / (10 ** (snr / 10)) / (np.mean(n ** 2) + 1e-9))
    return (y + n).astype(np.float32)


def corrupt(y):
    return add_noise(add_hum(y))


def sprsound():
    R = "/workspace/data/SPRSound"
    wavs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{R}/**/*.wav", recursive=True)}
    poor = []; good = []
    for jp in glob.glob(f"{R}/**/*.json", recursive=True):
        s = os.path.splitext(os.path.basename(jp))[0]
        if s not in wavs:
            continue
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        ra = d.get("record_annotation") if isinstance(d, dict) else None
        if ra is None:
            continue
        (poor if ra.strip().lower() == "poor quality" else good).append(wavs[s])
    rng = np.random.RandomState(0); rng.shuffle(good)
    return poor, good


def ext(d, n=300):
    ps = glob.glob(f"/workspace/data/{d}/files/**/*.wav", recursive=True)
    rng = np.random.RandomState(0); rng.shuffle(ps)
    return ps[:n]


_A = {}
def ast(paths, fn=None):
    import torch
    if "m" not in _A:
        from transformers import AutoFeatureExtractor, ASTModel
        _A["fe"] = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        _A["m"] = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").cuda().eval()
    fe, m = _A["fe"], _A["m"]; out = []; B = 16
    for i in range(0, len(paths), B):
        batch = []
        for p in paths[i:i + B]:
            y = load_audio(p)
            if y is not None:
                batch.append((fn(y) if fn else y).astype(np.float32))
        if not batch:
            continue
        inp = fe(batch, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out.append(m(inp.input_values.cuda()).last_hidden_state.mean(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 768), np.float32)


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

poor, good = sprsound(); kauh = ext("KAUH"); copd = ext("COPD")
log({"stage": "xdata_tables", "sprsound_poor": len(poor), "sprsound_good": len(good),
     "kauh": len(kauh), "copd": len(copd)})

# ---- Test A: real-quality model domain shift ----
sp_tr = poor + good[:1540]; sp_y = np.array([1] * len(poor) + [0] * min(1540, len(good)))
Xsp = ast(sp_tr)
clf_real = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(Xsp, sp_y[:len(Xsp)])
for name, ps in [("SPRSound-clean", good[1540:1840]), ("KAUH-clean", kauh), ("COPD-clean", copd)]:
    Xc = ast(ps)
    log({"stage": "domain_shift_Ppoor", "dataset": name, "n": len(Xc),
         "mean_P_poor": round(float(clf_real.predict_proba(Xc)[:, 1].mean()), 3)})

# ---- Test B: cross-dataset corruption detection ----
tr = good[:800]
Xcl = ast(tr); Xco = ast(tr, corrupt)
Xtr = np.vstack([Xcl, Xco]); ytr = np.concatenate([np.zeros(len(Xcl)), np.ones(len(Xco))])
clf_c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(Xtr, ytr)
for name, ps in [("SPRSound", good[800:1100]), ("KAUH", kauh), ("COPD", copd)]:
    A = ast(ps); B = ast(ps, corrupt)
    X = np.vstack([A, B]); yy = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    log({"stage": "xdata_corruption_auc", "test_dataset": name, "n_pairs": len(A),
         "auc": round(float(roc_auc_score(yy, clf_c.predict_proba(X)[:, 1])), 3)})
print("STAGE_COMPLETE xdata", flush=True)
