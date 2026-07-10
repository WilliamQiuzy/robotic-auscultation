"""Device fault detection shared by the acquisition tools.

A device can be *open and streaming* yet producing **no useful signal** — the classic case is
the ThinkLabs stethoscope being powered OFF: the sound card still streams, but it's only the
input's electrical noise floor. Tools must catch this and SAY which device is dead, instead of
silently computing garbage (e.g. "taps" detected in noise).

  audio_status(samples, sr)   -> OK | SILENT | NO_DATA   (SILENT = peak at noise floor)
  force_status(force_values)  -> OK | FROZEN | NO_DATA   (FROZEN = not moving = loose/off)
"""

from __future__ import annotations

import numpy as np

# a real tap/breath on a powered stethoscope peaks well above this; pure input noise sits below
AUDIO_FLOOR_DB = -40.0
FORCE_FROZEN_N = 0.05      # peak-to-peak below this over an "active" window => not responding


def _db(amp: float) -> float:
    return 20.0 * np.log10(max(float(amp), 1e-9))


def audio_status(samples, sr: int, floor_db: float = AUDIO_FLOOR_DB) -> dict:
    if samples is None or np.asarray(samples).size < sr // 4:
        return {"status": "NO_DATA", "peak_db": -120.0, "msg": "no audio captured (device not open?)"}
    s = np.asarray(samples, dtype=np.float64)
    peak = _db(np.max(np.abs(s)))
    if peak < floor_db:
        return {"status": "SILENT", "peak_db": round(peak, 1),
                "msg": f"audio peak {peak:.0f} dB ~ noise floor — stethoscope OFF? gain down? wrong channel? not making sound?"}
    return {"status": "OK", "peak_db": round(peak, 1), "msg": "audio signal present"}


def force_status(force_values, frozen_n: float = FORCE_FROZEN_N) -> dict:
    a = np.asarray(force_values, dtype=np.float64)
    if a.size < 5:
        return {"status": "NO_DATA", "swing_n": 0.0, "msg": "no force samples (sensor/serial down?)"}
    swing = float(a.max() - a.min())
    if swing < frozen_n:
        return {"status": "FROZEN", "swing_n": round(swing, 3),
                "msg": f"force not moving (swing {swing:.2f} N) — Hall loose/off, or no compression"}
    return {"status": "OK", "swing_n": round(swing, 2), "msg": "force responding"}


def warn(label: str, st: dict) -> bool:
    """Print a prominent warning if a device status is not OK; return True if faulted. Tolerates a
    status dict without 'msg' (e.g. an absent device) — don't crash the end of a recording."""
    if st.get("status", "OK") != "OK":
        print(f"  ⚠️  {label}: {st.get('status', '?')} — {st.get('msg', '(no detail)')}", flush=True)
        return True
    return False
