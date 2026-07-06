"""Deployable lung-sound QUALITY gate (SPRSound recipe).

    python sprsound_quality_gate.py clip.wav [clip2.wav ...]

Prints TWO independent signals per clip:
  1. P(poor) — the learned quality gate (AST encoder + GBM head, quality_gbm.joblib next to this
     script). Flagged LOW-QUALITY when P(poor) >= op_threshold (quality_meta.json, default 0.15).
     This gate is trained to be ROBUST TO (invariant to) 60 Hz hum — it judges the underlying lung
     content and will NOT flag a clip just for mains hum.
  2. 60Hz-hum% — a cheap, model-free mains-hum detector (energy in 60/120/180 Hz). Flagged HUM when
     >= 20% (clean lung recordings measure ≤11.5% here, any real hum ≥29%). Use it to catch a
     ground-loop / mains problem so the setup can be fixed or the clip re-recorded. The two flags
     answer different questions: "is the lung sound usable?" vs "is there a hardware hum to fix?"

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
HUM_FLAG_PCT = 20.0  # 60Hz-family energy %% above which we raise a (separate) mains-hum flag

import librosa, torch
from transformers import AutoFeatureExtractor, ASTModel

_fe = AutoFeatureExtractor.from_pretrained(meta["encoder"])
_dev = "cuda" if torch.cuda.is_available() else "cpu"
_m = ASTModel.from_pretrained(meta["encoder"]).to(_dev).eval()


def load_wave(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = y[: SR * 12]
    if len(y) < SR // 2:
        y = np.pad(y, (0, SR // 2 - len(y)))
    return (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def embed(y):
    inp = _fe([y], sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        return _m(inp.input_values.to(_dev)).last_hidden_state.mean(1).cpu().numpy()


def hum_energy_pct(y):
    """Percent of spectral energy in the 60 Hz mains family (60/120/180 Hz ±3). A cheap, model-free
    hardware-hum detector run in PARALLEL to the quality gate. The gate itself is trained to be
    *invariant* to hum (it judges the underlying lung content), so it will NOT flag hum on its own;
    this separate flag catches ground-loop / mains hum so the setup can be fixed or the clip re-recorded.
    Calibration (this repo): clean lung recordings ≤11.5%, even mild injected hum ≥29% → 20% threshold."""
    w = np.hanning(len(y))
    X = np.abs(np.fft.rfft(y * w))
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    tot = float(np.sum(X ** 2)) + 1e-12
    fam = sum(float(np.sum(X[(f >= h - 3) & (f <= h + 3)] ** 2)) for h in (60, 120, 180))
    return 100.0 * fam / tot


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for p in sys.argv[1:]:
        y = load_wave(p)
        pp = float(clf.predict_proba(embed(y))[:, 1][0])
        verdict = "LOW-QUALITY (flag)" if pp >= THR else "usable"
        hp = hum_energy_pct(y)
        hum = "HUM (60Hz)!" if hp >= HUM_FLAG_PCT else "ok"
        print(f"{os.path.basename(p):34s}  P(poor)={pp:.3f} -> {verdict:18s} | 60Hz={hp:4.1f}% -> {hum}")
