"""Batch 9 — does MORE DATA lift honest generalization? (the premise). Full SPRSound usable set
(~14k) vs the 4.5k subset, AST encoder, patient-grouped AUC. Plus selective prediction (reject the
most-uncertain X% -> AUC on the kept). Extracts AST on the full set. Logs to results.jsonl."""
import os, sys, glob, json, time
import numpy as np
sys.path.insert(0, "/workspace")
from audio_loop import load_audio, log, SR, ROOT   # noqa
_A = {}


def full_table():
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
    return [r[0] for r in rows], np.array([r[1] for r in rows])


def ast(paths_):
    import torch
    if "m" not in _A:
        from transformers import AutoFeatureExtractor, ASTModel
        _A["fe"] = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        _A["m"] = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").cuda().eval()
    fe, m = _A["fe"], _A["m"]; out = []; bs = 16; t0 = time.time()
    for i in range(0, len(paths_), bs):
        batch = [load_audio(p) for p in paths_[i:i + bs]]
        inp = fe(batch, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out.append(m(inp.input_values.cuda()).last_hidden_state.mean(1).cpu().numpy().astype(np.float32))
        if i % 3200 == 0:
            print(f"  ast {i}/{len(paths_)} {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(out)


paths, y = full_table()
groups = np.array([os.path.basename(p).split("_")[0] for p in paths])
log({"stage": "fulldata_table", "n": len(y), "poor": int(y.sum()), "patients": int(len(set(groups)))})
t0 = time.time(); X = ast(paths); log({"stage": "fulldata_feats", "secs": round(time.time() - t0)})
np.savez_compressed("/workspace/out/ast_full.npz", X=X, y=y, groups=groups)
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
oof = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, groups):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(X[tr], y[tr])
    oof[te] = clf.predict_proba(X[te])[:, 1]
auc = roc_auc_score(y, oof)
log({"stage": "fulldata_patient_auc", "encoder": "AST", "n": len(y), "auc_patient": round(float(auc), 3),
     "vs_4500subset": 0.941})
# selective prediction: reject most-uncertain (|p-0.5| smallest), AUC on kept
unc = -np.abs(oof - 0.5)
for keep in (1.0, 0.9, 0.75, 0.5):
    thr = np.quantile(unc, keep)
    m = unc <= thr
    if len(set(y[m])) == 2:
        log({"stage": "selective_pred", "keep_frac": keep, "n_kept": int(m.sum()),
             "auc_kept": round(float(roc_auc_score(y[m], oof[m])), 3)})
print("STAGE_COMPLETE batch9", flush=True)
