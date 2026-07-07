"""The Q head — real learned audio-quality model as a plug-in `q_fn` for the IL pipeline.

This is the "Head-Q" of docs/audio_reward_design.md's one-encoder-three-heads design: AST embedding
→ the SPRSound-trained GBM → P(poor), returned as **P(usable) = 1 − P(poor)** ∈ [0,1] (higher =
better), which is exactly what `session_to_lerobot.py` wants for `q_fn(audio, sr) → q_usable` and
what the segment reward `gate · Q` maximizes.

    from q_head import make_quality_fn
    q = make_quality_fn()                 # lazy-loads AST + quality_gbm.joblib on first call
    p_usable = q(audio_16k, 16000)

Deps: torch, transformers, librosa/soundfile not needed (audio comes in as an array), joblib, sklearn.
The model is the deployable SPRSound gate (honest patient-grouped AUC ≈0.94); see sprsound_loop/.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import numpy as np

_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "scripts", "audio", "sprsound_loop")


def _numpy_compat():
    """numpy 1.x<->2.x BitGenerator pickle shim (model was trained under numpy2 on the pod)."""
    try:
        import numpy.random._pickle as _npp
        orig = getattr(_npp, "__bit_generator_ctor")

        def ctor(name="MT19937"):
            return orig(name if isinstance(name, str) else getattr(name, "__name__", name))
        setattr(_npp, "__bit_generator_ctor", ctor)
    except Exception:
        pass


def make_quality_fn(model_dir: Optional[str] = None) -> Callable[[np.ndarray, int], float]:
    """Return `q(audio, sr) -> P(usable) in [0,1]`. Lazily loads AST + the GBM on first call so
    importing this module (e.g. in the converter) is cheap when --real-q is not used."""
    model_dir = model_dir or _DEFAULT_DIR
    state: dict = {}

    def _load():
        import joblib
        import torch
        from transformers import AutoFeatureExtractor, ASTModel
        _numpy_compat()
        meta = json.load(open(os.path.join(model_dir, "quality_meta.json")))
        state["meta"] = meta
        state["clf"] = joblib.load(os.path.join(model_dir, "quality_gbm.joblib"))
        state["fe"] = AutoFeatureExtractor.from_pretrained(meta["encoder"])
        state["dev"] = "cuda" if torch.cuda.is_available() else "cpu"
        state["m"] = ASTModel.from_pretrained(meta["encoder"]).to(state["dev"]).eval()
        state["torch"] = torch

    def q(audio: np.ndarray, sr: int) -> float:
        if "m" not in state:
            _load()
        meta, torch = state["meta"], state["torch"]
        y = np.asarray(audio, np.float32)
        target = meta["sr"]
        if sr != target:                                   # linear resample to the model rate
            y = np.interp(np.linspace(0, len(y) - 1, int(len(y) * target / sr)), np.arange(len(y)), y)
        y = y[: target * 12]
        if len(y) < target // 2:
            y = np.pad(y, (0, target // 2 - len(y)))
        y = (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)
        inp = state["fe"]([y], sampling_rate=target, return_tensors="pt")
        with torch.no_grad():
            emb = state["m"](inp.input_values.to(state["dev"])).last_hidden_state.mean(1).cpu().numpy()
        p_poor = float(state["clf"].predict_proba(emb)[:, 1][0])
        return float(np.clip(1.0 - p_poor, 0.0, 1.0))       # P(usable)

    return q


if __name__ == "__main__":  # smoke: score a clean lung sample vs a noised copy
    import glob
    import soundfile as sf
    q = make_quality_fn()
    samples = sorted(glob.glob(os.path.join(os.path.dirname(_DEFAULT_DIR), "..", "..",
                                             "data", "audio", "_samples", "hf_lung_v1", "*.wav")))
    if samples:
        y, sr = sf.read(samples[0]); y = y.mean(1) if y.ndim > 1 else y
        clean = q(y, sr)
        noisy = q(y + 0.3 * np.random.RandomState(0).randn(len(y)), sr)
        print(f"P(usable) clean={clean:.3f}  noised={noisy:.3f}  (expect clean > noised)")
    else:
        print("no sample wav found; q_fn built OK")
