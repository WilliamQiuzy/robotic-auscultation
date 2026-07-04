"""OPERA (respiratory-specific foundation model) vs AST (general AudioSet) on SPRSound lung-sound
quality — patient-grouped (honest) + random AUC. User-authorized to run the OPERA repo code."""
import os, sys, glob, json, time
import numpy as np
ROOT = "/workspace/data/SPRSound"; OUT = "/workspace/out"; os.makedirs(OUT, exist_ok=True); SR = 16000
MAX_USABLE = 4500


def log(r):
    r["t"] = time.strftime("%H:%M:%S")
    open(f"{OUT}/opera_results.jsonl", "a").write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True)


def load_audio(p):
    import librosa
    y, _ = librosa.load(p, sr=SR, mono=True)
    if len(y) < SR // 2:
        y = np.pad(y, (0, SR // 2 - len(y)))
    y = y[: SR * 12]; return (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def build_table():
    wavs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{ROOT}/**/*.wav", recursive=True)}
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
        rows.append((wavs[stem], 1 if ra.strip().lower() == "poor quality" else 0))
    poor = [r for r in rows if r[1] == 1]; good = [r for r in rows if r[1] == 0]
    rng = np.random.RandomState(0); rng.shuffle(good); good = good[:MAX_USABLE]
    tab = poor + good; rng.shuffle(tab)
    return [t[0] for t in tab], np.array([t[1] for t in tab])


def ast_feat(paths):
    import torch
    from transformers import AutoFeatureExtractor, ASTModel
    fe = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    m = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").cuda().eval()
    out = []; B = 16; t0 = time.time()
    for i in range(0, len(paths), B):
        batch = [load_audio(p) for p in paths[i:i + B]]
        inp = fe(batch, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out.append(m(inp.input_values.cuda()).last_hidden_state.mean(1).cpu().numpy())
        if i % 1600 == 0:
            print(f"  ast {i}/{len(paths)} {time.time()-t0:.0f}s", flush=True)
    log({"stage": "ast_feat_done", "secs": round(time.time() - t0)})
    return np.concatenate(out).astype(np.float32)


def opera_feat(paths):
    sys.path.insert(0, "/workspace/OPERA")
    from src.benchmark.model_util import extract_opera_feature
    t0 = time.time()
    X = np.array(extract_opera_feature(paths, pretrain="operaCT", input_sec=8, dim=768))
    log({"stage": "opera_feat_done", "secs": round(time.time() - t0), "dim": int(X.shape[1])})
    return X.astype(np.float32)


def evals(name, X, y, groups):
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    og = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        og[te] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    orr = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        orr[te] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    log({"stage": "compare", "encoder": name, "dim": int(X.shape[1]),
         "auc_patient": round(float(roc_auc_score(y, og)), 3), "auc_random": round(float(roc_auc_score(y, orr)), 3)})


if __name__ == "__main__":
    paths, y = build_table()
    groups = np.array([os.path.basename(p).split("_")[0] for p in paths])
    log({"stage": "table", "n": len(y), "poor": int(y.sum()), "patients": int(len(set(groups)))})
    Xo = opera_feat(paths); evals("OPERA-CT", Xo, y, groups)
    Xa = ast_feat(paths); evals("AST", Xa, y, groups)
    print("STAGE_COMPLETE opera", flush=True)
