"""Streaming contact-gated quality inference (docs/realtime_multimodal_pipeline.md §4).

Sliding 8 s window over the audio stream -> a per-hop P(usable). The Hall contact force GATES
it: when contact is out of band, P(usable)=0 (no point scoring no-contact noise) and we skip
the encoder. The scorer is pluggable:
  - on the 5090: the real Q = frozen OPERA-CT + logistic head (GPU);
  - here / CI: `band_energy_proxy` (lung-band spectral concentration) — a lightweight STAND-IN
    so the streaming structure + gating can be validated without OPERA. It is NOT the real Q.

    python ausc/realtime_q.py --seconds 12      # tap/breathe on the diaphragm to drive the score
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(HERE, "audio"), os.path.join(ROOT, "mechanical_sensor", "force_feedback")]


def band_energy_proxy(samples: np.ndarray, sr: int = 16000, lo: float = 90.0, hi: float = 1100.0) -> float:
    """Placeholder scorer: fraction of spectral energy in the lung band [lo,hi]. NOT OPERA-Q."""
    if samples.size < sr // 4:
        return 0.0
    x = samples - samples.mean()
    f = np.fft.rfftfreq(x.size, 1.0 / sr)
    p = np.abs(np.fft.rfft(x)) ** 2
    tot = p[(f >= 20) & (f <= sr / 2)].sum()
    return float(p[(f >= lo) & (f <= hi)].sum() / tot) if tot > 0 else 0.0


class StreamingQuality:
    """Per-hop contact-gated quality, EMA-smoothed."""

    def __init__(self, scorer=band_energy_proxy, force_band=(3.0, 5.0),
                 window_s: float = 8.0, ema: float = 0.5) -> None:
        self.scorer = scorer
        self.force_lo, self.force_hi = force_band
        self.window_s = window_s
        self.ema = ema
        self.p_smoothed = 0.0

    def update(self, audio_samples: np.ndarray, sr: int, force_n: float | None) -> dict:
        gate = force_n is not None and self.force_lo <= force_n <= self.force_hi
        q_raw = self.scorer(audio_samples, sr)            # audio quality (gate-independent, for visibility)
        p = q_raw if gate else 0.0                        # fused: no contact -> not usable
        self.p_smoothed = self.ema * self.p_smoothed + (1 - self.ema) * p
        return {"force_n": force_n, "gate": gate, "q_raw": round(q_raw, 3),
                "p_usable": round(self.p_smoothed, 3)}


def run_live(seconds: float = 12.0, hop_s: float = 1.0, window_s: float = 8.0,
             force_band=(3.0, 5.0), audio_device: str = "Scarlett Solo USB", audio_channel: int | None = 1,
             hall_port: str | None = None, hall_cal: str | None = None) -> None:
    from audio_stream import AudioStream
    from force_sensor import ForceCalibration, ForceSensor

    hall_port = hall_port or (sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*")) or [None])[0]
    hall_cal = hall_cal or os.path.join(ROOT, "mechanical_sensor", "force_feedback",
                                        "force_feedback_calibration_veo6.json")
    sq = StreamingQuality(force_band=force_band, window_s=window_s)
    from contextlib import ExitStack
    with ExitStack() as st:
        au = st.enter_context(AudioStream(audio_device, pick_channel=audio_channel, buffer_seconds=window_s + 3))
        fs = None
        try:
            fs = st.enter_context(ForceSensor(hall_port, ForceCalibration.from_json(hall_cal)))
        except Exception as e:  # noqa: BLE001
            print(f"[force] not available ({e}); gate will be open")
        print(f">>> breathe/tap on the diaphragm — streaming P(usable) @ {1/hop_s:.0f} Hz <<<", flush=True)
        time.sleep(window_s * 0.3)
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            w = au.latest_window(window_s)
            force_n = fs.read().force_n if fs else 4.0   # no force -> assume in-band for the demo
            out = sq.update(w.samples, w.sr, force_n)
            bar = "#" * int(out["p_usable"] * 30)
            print(f"  force={out['force_n']:.2f}N gate={'OPEN ' if out['gate'] else 'shut '} "
                  f"q_raw={out['q_raw']:.2f}  P(usable)={out['p_usable']:.2f} |{bar}", flush=True)
            time.sleep(hop_s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--force-lo", type=float, default=3.0)
    ap.add_argument("--force-hi", type=float, default=5.0)
    a = ap.parse_args()
    run_live(seconds=a.seconds, hop_s=a.hop, force_band=(a.force_lo, a.force_hi))
