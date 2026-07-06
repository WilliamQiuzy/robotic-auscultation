"""Controlled 60Hz-hum A/B on our own clean ThinkLabs capture, through the DEPLOYED gate.
Base = session_20260624_145256/audio.wav (clean, gate P(poor)=0.008). Inject the study's 60Hz
hum at several severities, then de-hum with 60/120/180Hz notch filters. Mirrors train_final's
pipeline: normalize base ONCE, add hum to the normalized signal, feed to AST without renorm."""
import sys, os
GATE_DIR = os.path.dirname(os.path.abspath(__file__))  # this script lives next to the gate
sys.path.insert(0, GATE_DIR)
import numpy as np
import soundfile as sf
from scipy.signal import iirnotch, filtfilt, resample_poly
from math import gcd
import torch
import sprsound_quality_gate as G  # importing loads AST encoder + the GBM head

SR = G.SR


def embed_arr(y):
    y = y[: SR * 12]
    if len(y) < SR // 2:
        y = np.pad(y, (0, SR // 2 - len(y)))
    inp = G._fe([y.astype(np.float32)], sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        return G._m(inp.input_values.to(G._dev)).last_hidden_state.mean(1).cpu().numpy()


def P(y):
    return float(G.clf.predict_proba(embed_arr(y))[:, 1][0])


def add_hum(y, l):
    t = np.arange(len(y)) / SR
    return (y + l * sum(np.sin(2 * np.pi * 60 * k * t) / k for k in range(1, 5))).astype(np.float32)


def dehum(y):
    out = y.astype(np.float64)
    for f0 in (60, 120, 180):
        b, a = iirnotch(f0, Q=30, fs=SR)
        out = filtfilt(b, a, out)
    return out.astype(np.float32)


# base = any clean recording; default to our clean ThinkLabs session capture (gate P(poor)=0.008)
BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    GATE_DIR, "..", "..", "..", "data", "sessions", "session_20260624_145256", "audio.wav")
y, sr = sf.read(BASE)
if y.ndim > 1:
    y = y.mean(1)
y = y.astype(np.float32)
if sr != SR:
    g = gcd(sr, SR); y = resample_poly(y, SR // g, sr // g).astype(np.float32)
y = y / (np.max(np.abs(y)) + 1e-9)  # normalize ONCE (load_audio style)

thr = G.meta["op_threshold"]
def tag(p):
    return "FLAG " if p >= thr else "pass "

print(f"base clean          P(poor)={P(y):.3f}  {tag(P(y))}")
h02 = None
for l in [0.05, 0.10, 0.20, 0.40]:
    yl = add_hum(y, l); p = P(yl)
    print(f"+60Hz hum  l={l:<4}    P(poor)={p:.3f}  {tag(p)}")
    if abs(l - 0.20) < 1e-9:
        h02 = yl
p_hum = P(h02); p_deh = P(dehum(h02))
print(f"--- de-hum recovery (from l=0.20) ---")
print(f"hummed (l=0.20)     P(poor)={p_hum:.3f}  {tag(p_hum)}")
print(f"de-hummed (notch)   P(poor)={p_deh:.3f}  {tag(p_deh)}")
