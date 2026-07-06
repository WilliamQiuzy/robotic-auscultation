"""Deployable lung-sound QUALITY gate (SPRSound recipe).

    python sprsound_quality_gate.py clip.wav [clip2.wav ...]

Loads the AST encoder (MIT/ast-finetuned-audioset) + the trained GBM head (quality_gbm.joblib,
sits next to this script) and prints P(poor) per clip. A clip is flagged low-quality when
P(poor) >= op_threshold (from quality_meta.json, default 0.15 = recall-0.99 operating point).

This is the OUTPUT of the SPRSound quality study (honest patient-grouped AUC ~0.94, robust to
60 Hz hum + noise via train-time augmentation). It detects gross recording artifacts
(noise/SNR/hum/clipping). It is DISTINCT from scripts/audio/predict_quality.py, which is the
doctor-label Q v1 (OPERA-CT + clinician "usable" grading, subtler task, AUC 0.72).

Deps: torch, transformers, librosa (+ soundfile), joblib, scikit-learn. CPU is fine for a few clips.
The bundled quality_gbm.joblib is numpy 1.x/2.x-portable (rng stripped); a compat shim below also
normalizes any numpy-2.x-pickled BitGenerator so the model loads on a numpy-1.x host.
"""
import sys, os, json, joblib
import numpy as np

# numpy 1.x <-> 2.x pickle compat: a Generator pickled under numpy 2.x hands the BitGenerator
# *class* to __bit_generator_ctor, but numpy 1.x expects its string name. Normalize before load
# so the model (trained under numpy 2.x on the pod) loads on a numpy-1.x host too.
try:
    import numpy.random._pickle as _npp
    _orig_ctor = getattr(_npp, "__bit_generator_ctor")

    def _compat_ctor(bit_generator_name="MT19937"):
        if not isinstance(bit_generator_name, str):
            bit_generator_name = getattr(bit_generator_name, "__name__", bit_generator_name)
        return _orig_ctor(bit_generator_name)

    setattr(_npp, "__bit_generator_ctor", _compat_ctor)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(os.path.join(HERE, "quality_meta.json")))
clf = joblib.load(os.path.join(HERE, "quality_gbm.joblib"))
SR = meta["sr"]; THR = meta["op_threshold"]

import librosa, torch
from transformers import AutoFeatureExtractor, ASTModel

_fe = AutoFeatureExtractor.from_pretrained(meta["encoder"])
_dev = "cuda" if torch.cuda.is_available() else "cpu"
_m = ASTModel.from_pretrained(meta["encoder"]).to(_dev).eval()


def embed(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = y[: SR * 12]
    if len(y) < SR // 2:
        y = np.pad(y, (0, SR // 2 - len(y)))
    y = (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)
    inp = _fe([y], sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        return _m(inp.input_values.to(_dev)).last_hidden_state.mean(1).cpu().numpy()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for p in sys.argv[1:]:
        pp = float(clf.predict_proba(embed(p))[:, 1][0])
        verdict = "LOW-QUALITY (flag)" if pp >= THR else "usable"
        print(f"{os.path.basename(p):40s}  P(poor)={pp:.3f}  ->  {verdict}")
