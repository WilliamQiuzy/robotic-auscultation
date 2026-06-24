"""Cross-modal time-sync calibration (docs/realtime_multimodal_pipeline.md §1).

Both AudioStream and ForceSensor stamp on `time.monotonic()`, but the sound card buffers
audio (the sound is older than its host stamp) and the serial link delays force. A physical
**tap on the stethoscope diaphragm** is one simultaneous event in both: a transient in audio
and an impulse in force. Tapping a few times and matching the two lets us measure the fixed
offset  Delta = (force tap time) - (audio tap time)  to add when fusing the streams.

    python ausc/sync_calib.py --seconds 10        # tap the diaphragm ~4-5x during the window
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(HERE, "audio"),
                os.path.join(ROOT, "mechanical_sensor", "force_feedback")]


def _peaks(x: np.ndarray, fs: float, min_gap_s: float, k: float) -> np.ndarray:
    """Indices of prominent peaks: above mean + k*std, separated by >= min_gap_s."""
    from scipy.signal import find_peaks
    thr = float(x.mean() + k * x.std())
    idx, _ = find_peaks(x, height=thr, distance=int(min_gap_s * fs))
    return idx


def calibrate_tap_sync(seconds: float = 10.0, grid_hz: float = 1000.0,
                       audio_device: str = "Scarlett Solo USB", audio_channel: int | None = 1,
                       hall_port: str | None = None, hall_cal: str | None = None) -> dict:
    from audio_stream import AudioStream
    from force_sensor import ForceCalibration, ForceSensor

    hall_port = hall_port or (sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*")) or [None])[0]
    hall_cal = hall_cal or os.path.join(ROOT, "mechanical_sensor", "force_feedback",
                                        "force_feedback_calibration_veo6.json")
    tmp_csv = os.path.join("/tmp", f"sync_force_{int(time.monotonic()*1000)}.csv")

    cal = ForceCalibration.from_json(hall_cal)
    with AudioStream(audio_device, pick_channel=audio_channel, buffer_seconds=seconds + 3.0) as au, \
         ForceSensor(hall_port, cal, log_csv=tmp_csv) as _fs:
        time.sleep(0.6)
        print(f">>> TAP the diaphragm SHARPLY ~4-5 times over the next {seconds:.0f}s <<<", flush=True)
        time.sleep(seconds)
        win = au.latest_window(seconds)

    # ---- audio envelope on the common grid ----
    a = win.samples.astype(np.float64)
    sr = win.sr
    step = max(1, int(sr / grid_hz))
    env = np.abs(a)[: len(a) // step * step].reshape(-1, step).max(axis=1)  # peak-hold downsample
    t_env = win.t_start + np.arange(len(env)) * (step / sr)

    # ---- force impulse on the common grid (high-pass: remove slow baseline) ----
    ft, ff = [], []
    with open(tmp_csv) as f:
        r = csv.reader(f); next(r)
        for row in r:
            try:
                ft.append(float(row[0])); ff.append(float(row[4]))
            except (ValueError, IndexError):
                continue
    ft, ff = np.array(ft), np.array(ff)
    t_lo, t_hi = max(t_env[0], ft[0]), min(t_env[-1], ft[-1])
    tg = np.arange(t_lo, t_hi, 1.0 / grid_hz)
    fg = np.interp(tg, ft, ff)
    fg_hp = np.abs(fg - np.convolve(fg, np.ones(int(0.25 * grid_hz)) / int(0.25 * grid_hz), mode="same"))
    eg = np.interp(tg, t_env, env)

    # ---- detect taps in each, match nearest ----
    a_idx = _peaks(eg, grid_hz, 0.4, 3.5)
    f_idx = _peaks(fg_hp, grid_hz, 0.4, 2.5)
    a_t, f_t = tg[a_idx], tg[f_idx]
    deltas = []
    for at in a_t:
        if len(f_t):
            j = int(np.argmin(np.abs(f_t - at)))
            if abs(f_t[j] - at) < 0.25:        # within 250 ms -> same tap
                deltas.append(f_t[j] - at)
    deltas = np.array(deltas)
    res = {
        "n_audio_taps": int(len(a_t)), "n_force_taps": int(len(f_t)),
        "n_matched": int(len(deltas)),
        "offset_ms": round(float(np.median(deltas) * 1000), 1) if len(deltas) else None,
        "offset_std_ms": round(float(np.std(deltas) * 1000), 1) if len(deltas) > 1 else None,
        "note": "offset = force_t - audio_t; add to audio timestamps to align force.",
    }
    try:
        os.remove(tmp_csv)
    except OSError:
        pass
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    r = calibrate_tap_sync(seconds=a.seconds)
    import json
    print(json.dumps(r, indent=2))
    if r["n_matched"] >= 2:
        print(f"\nRESULT: audio<->force offset = {r['offset_ms']:+} ms "
              f"(+/- {r['offset_std_ms']} ms over {r['n_matched']} taps)")
    else:
        print("\nRESULT: not enough matched taps - tap sharper / more times and retry")
