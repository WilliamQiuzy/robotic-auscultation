"""Batch 8 — SPRSound OFFICIAL BioCAS2022 split: train on train2022, eval inter_test (held-out
patients) vs intra_test (seen patients). Benchmark-comparable + a 2nd generalization measure.
Extracts AST on the official sets. Logs to /workspace/out/results.jsonl."""
import os, sys, glob, json, time
import numpy as np
sys.path.insert(0, "/workspace")
from audio_loop import load_audio, log, SR   # noqa
B = "/workspace/data/SPRSound/BioCAS2022"
_A = {}


def table(json_dir, wav_dir):
    rows = []
    for jp in glob.glob(f"{json_dir}/*.json"):
        stem = os.path.splitext(os.path.basename(jp))[0]
        wp = f"{wav_dir}/{stem}.wav"
        if not os.path.exists(wp):
            continue
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        ra = d.get("record_annotation") if isinstance(d, dict) else None
        if ra is None:
            continue
        rows.append((wp, 1 if ra.strip().lower() == "poor quality" else 0))
    return [r[0] for r in rows], np.array([r[1] for r in rows])


def ast(paths_):
    import torch
    if "m" not in _A:
        from transformers import AutoFeatureExtractor, ASTModel
        _A["fe"] = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        _A["m"] = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").cuda().eval()
    fe, m = _A["fe"], _A["m"]; out = []; bs = 16
    for i in range(0, len(paths_), bs):
        batch = [load_audio(p) for p in paths_[i:i + bs]]
        inp = fe(batch, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out.append(m(inp.input_values.cuda()).last_hidden_state.mean(1).cpu().numpy().astype(np.float32))
    return np.concatenate(out)


ptr, ytr = table(f"{B}/train2022_json", f"{B}/train2022_wav")
pin, yin = table(f"{B}/test2022_json/inter_test_json", f"{B}/test2022_wav")
pia, yia = table(f"{B}/test2022_json/intra_test_json", f"{B}/test2022_wav")
log({"stage": "official_tables", "train": [len(ytr), int(ytr.sum())],
     "inter": [len(yin), int(yin.sum())], "intra": [len(yia), int(yia.sum())]})
t0 = time.time()
Xtr, Xin, Xia = ast(ptr), ast(pin), ast(pia)
log({"stage": "official_feats", "secs": round(time.time() - t0)})
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(Xtr, ytr)
log({"stage": "official_benchmark", "encoder": "AST",
     "inter_test_auc": round(float(roc_auc_score(yin, clf.predict_proba(Xin)[:, 1])), 3),
     "intra_test_auc": round(float(roc_auc_score(yia, clf.predict_proba(Xia)[:, 1])), 3)})
print("STAGE_COMPLETE batch8", flush=True)
