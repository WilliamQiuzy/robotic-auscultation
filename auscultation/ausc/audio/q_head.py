"""The shared audio encoder + the Q head — one AST encoder feeding multiple heads.

docs/audio_reward_design.md's "one encoder, three heads": a single frozen AST encoder produces the
embedding; `make_embed_fn` exposes it (for the reward model, Head-Reward), and `make_quality_fn`
puts the SPRSound GBM on top → **P(usable) = 1 − P(poor)** ∈ [0,1] (Head-Q, what `session_to_lerobot`
wants for `q_fn(audio, sr) → q_usable`).

    from q_head import make_embed_fn, make_quality_fn
    emb = make_embed_fn()                  # emb(audio_16k, 16000) -> 768-d AST embedding
    q   = make_quality_fn()                # q(audio_16k, 16000)   -> P(usable)

Deps: torch, transformers, joblib, sklearn. Model = the deployable SPRSound gate (AUC ≈0.94).
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import numpy as np

DEFAULT_ENCODER = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "scripts", "audio", "sprsound_loop")
_ENC: dict = {}                                            # shared lazy encoder cache (one encoder)


def _numpy_compat():
    """numpy 1.x<->2.x BitGenerator pickle shim (GBM was trained under numpy2 on the pod)."""
    try:
        import numpy.random._pickle as _npp
        orig = getattr(_npp, "__bit_generator_ctor")

        def ctor(name="MT19937"):
            return orig(name if isinstance(name, str) else getattr(name, "__name__", name))
        setattr(_npp, "__bit_generator_ctor", ctor)
    except Exception:
        pass


def _encoder(encoder_id: str = DEFAULT_ENCODER):
    if "m" not in _ENC:
        import torch
        from transformers import AutoFeatureExtractor, ASTModel
        _ENC["fe"] = AutoFeatureExtractor.from_pretrained(encoder_id)
        _ENC["dev"] = "cuda" if torch.cuda.is_available() else "cpu"
        _ENC["m"] = ASTModel.from_pretrained(encoder_id).to(_ENC["dev"]).eval()
        _ENC["torch"] = torch
    return _ENC


def embed(audio: np.ndarray, sr: int) -> np.ndarray:
    """AST embedding (768-d) of one clip. Resamples to 16 kHz, 12 s cap, peak-normalized."""
    enc = _encoder(); torch = enc["torch"]
    y = np.asarray(audio, np.float32)
    if sr != SR:
        y = np.interp(np.linspace(0, len(y) - 1, int(len(y) * SR / sr)), np.arange(len(y)), y)
    y = y[: SR * 12]
    if len(y) < SR // 2:
        y = np.pad(y, (0, SR // 2 - len(y)))
    y = (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)
    inp = enc["fe"]([y], sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        return enc["m"](inp.input_values.to(enc["dev"])).last_hidden_state.mean(1).cpu().numpy()[0]


def make_embed_fn() -> Callable[[np.ndarray, int], np.ndarray]:
    """Head-shared encoder: `emb(audio, sr) -> 768-d`. Used to train the reward model."""
    return embed


def make_quality_fn(model_dir: Optional[str] = None) -> Callable[[np.ndarray, int], float]:
    """Head-Q: `q(audio, sr) -> P(usable) ∈ [0,1]` = 1 − P(poor) from the SPRSound GBM on the embedding."""
    model_dir = model_dir or _DEFAULT_DIR
    st: dict = {}

    def q(audio: np.ndarray, sr: int) -> float:
        if "clf" not in st:
            import joblib
            _numpy_compat()
            st["clf"] = joblib.load(os.path.join(model_dir, "quality_gbm.joblib"))
        p_poor = float(st["clf"].predict_proba(embed(audio, sr)[None, :])[:, 1][0])
        return float(np.clip(1.0 - p_poor, 0.0, 1.0))

    return q


if __name__ == "__main__":  # smoke: embedding dim + Q clean vs noised
    import glob
    import soundfile as sf
    emb, q = make_embed_fn(), make_quality_fn()
    s = sorted(glob.glob(os.path.join(_DEFAULT_DIR, "..", "..", "..", "data", "audio",
                                      "_samples", "hf_lung_v1", "*.wav")))
    if s:
        y, sr = sf.read(s[0]); y = y.mean(1) if y.ndim > 1 else y
        print(f"embed dim = {emb(y, sr).shape}  | P(usable) clean={q(y, sr):.3f} "
              f"noised={q(y + 0.3 * np.random.RandomState(0).randn(len(y)), sr):.3f}")
    else:
        print("no sample; encoder built OK")
