"""Train + SAVE the deployable lung-sound QUALITY gate: AST encoder + GBM head, hum/noise-augmented
(the recommended recipe from the study). Saves quality_gbm.joblib + quality_meta.json.
Run on a GPU pod with SPRSound at /workspace/data/SPRSound."""
import os, glob, json, time, joblib
import numpy as np
ROOT = "/workspace/data/SPRSound"; OUT = "/workspace/out"; os.makedirs(OUT, exist_ok=True); SR = 16000


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


def table(nusable=4000):
    wavs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{ROOT}/**/*.wav", recursive=True)}
    poor = []; good = []
    for jp in glob.glob(f"{ROOT}/**/*.json", recursive=True):
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
    paths = poor + good[:nusable]
    return paths, np.array([1] * len(poor) + [0] * min(nusable, len(good)))


_A = {}
def ast(paths, fn=None):
    import torch
    if "m" not in _A:
        from transformers import AutoFeatureExtractor, ASTModel
        _A["fe"] = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        _A["m"] = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").cuda().eval()
    fe, m = _A["fe"], _A["m"]; out = []; B = 16; t0 = time.time()
    for i in range(0, len(paths), B):
        b = []
        for p in paths[i:i + B]:
            y = load_audio(p)
            if y is not None:
                b.append((fn(y) if fn else y).astype(np.float32))
        if not b:
            continue
        inp = fe(b, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out.append(m(inp.input_values.cuda()).last_hidden_state.mean(1).cpu().numpy())
        if i % 1600 == 0:
            print(f"  ast {i}/{len(paths)} {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(out).astype(np.float32)


paths, y = table()
print("table", len(paths), "poor", int(y.sum()), flush=True)
Xc = ast(paths); Xh = ast(paths, add_hum); Xn = ast(paths, add_noise)
from sklearn.ensemble import HistGradientBoostingClassifier
X = np.vstack([Xc, Xh, Xn]); Y = np.concatenate([y, y, y])
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08).fit(X, Y)
# Portability: drop the incidental numpy Generator (predict never uses it) so the saved model
# loads across numpy 1.x/2.x hosts. Without this, a numpy-2.x-pickled Generator won't unpickle
# on a numpy-1.x deployment box (BitGenerator state format changed between numpy 1 and 2).
if hasattr(clf, "_feature_subsample_rng"):
    clf._feature_subsample_rng = None
joblib.dump(clf, f"{OUT}/quality_gbm.joblib")
json.dump({"encoder": "MIT/ast-finetuned-audioset-10-10-0.4593", "pool": "mean_last_hidden", "sr": SR,
           "train": "SPRSound Poor-vs-usable + 60Hz-hum/noise augmentation", "n_clips": len(paths),
           "n_poor": int(y.sum()), "op_threshold": 0.15,
           "output": "P(poor); flag clip as low-quality if P(poor) >= op_threshold",
           "honest_patient_grouped_auc": 0.94},
          open(f"{OUT}/quality_meta.json", "w"), indent=2)
print("SAVED_MODEL", flush=True)
