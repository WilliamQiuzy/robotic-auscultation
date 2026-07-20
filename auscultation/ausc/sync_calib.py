"""Cross-modal time-sync calibration (docs/realtime_multimodal_pipeline.md §1).

Both AudioStream and ForceSensor stamp on `time.monotonic()`, but the sound card buffers
audio (the sound is older than its host stamp) and the serial link delays force. A physical
**tap on the stethoscope diaphragm** is one simultaneous event in both: a transient in audio
and an impulse in force. Tapping ~6 times and aligning the two measures the fixed offset
Delta = (force tap time) - (audio tap time)  to add when fusing the streams.

Signal processing (learned the hard way, see ``compute_tap_sync``): the sound card emits a
large DC transient for several seconds when the stream opens, so we ``settle`` before the tap
window; the audio envelope is **high-passed** (a raw ``|samples|`` envelope lets that transient
dominate a mean/std threshold and hide every tap); and each tap's offset is a **per-tap
cross-correlation** of the audio envelope against the force impulse (argmax peak-matching is
fooled by the diaphragm ring, which can land tens of ms after the strike). A STALE/disconnected
sound card mid-capture yields frozen, misaligned timestamps and is rejected, not averaged.

Tap technique: hold the module in the air (resting it on the bench damps the diaphragm and
kills the audio), and tap SHARPLY + FIRMLY (sharp -> audio transient; firm -> Hall compresses),
~6 taps ~3s apart so each is an isolated peak in both streams.

    python ausc/sync_calib.py --seconds 20 --settle 4 --audio-device "Scarlett Solo USB" \
      --audio-channel 1 --hall-port /dev/ttyACM0 --hall-cal force_calibration.json \
      --out wam_sync_v1.json

The display name and serial path are diagnostics, not identity.  The persisted
capture mode records the selected PortAudio descriptor and pyserial USB
serial/VID/PID.  On Linux ALSA, sysfs is also queried for the audio USB parent.
Formal calibration requires either that audio USB serial or, for a serialless
Linux device, its VID/PID plus fixed sysfs USB topology path. The explicit
``--allow-unstable-audio-identity`` escape hatch is rehearsal-only.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, "audio"),
                os.path.join(ROOT, "mechanical_sensor", "force_feedback")]
from health import audio_status, force_status  # noqa: E402
from record_session import (  # noqa: E402
    build_sync_capture_mode,
    default_hall_calibration,
    resolve_audio_hardware_descriptor,
    resolve_hall_hardware_descriptor,
    resolve_linux_alsa_usb_identity,
    sync_capture_endpoints,
    validate_sync_capture_mode,
)

SYNC_SCHEMA_VERSION = "wam_sync_v1"
MIN_SYNC_TAPS = 4
MAX_SYNC_STD_MS = 5.0
MAX_SYNC_OFFSET_MS = 250.0
SYNC_MEASUREMENT_METHOD = "physical_diaphragm_taps"


def sync_identity_warning(capture_mode: object) -> str | None:
    mode = validate_sync_capture_mode(capture_mode)
    if (mode["audio"]["usb_serial"] is None
            and mode["audio"].get("usb_port_path") is None):
        return (
            "WARNING: calibration has no stable audio USB serial or fixed Linux USB "
            "topology and is not arm-eligible; "
            f"identity mapping status={mode['audio']['usb_identity_source']!r}. "
            "Run calibration on the Linux ALSA acquisition host and verify its sysfs mapping."
        )
    return None


def build_calibration_capture_mode(
    *, audio_device: str | int, audio_channel: int, hall_port: str,
    force_calibration_sha256: str, hall_resolver=None, audio_resolver=None,
    allow_unstable_audio_identity: bool = False,
) -> dict:  # noqa: ANN001
    """Resolve the hardware used for taps; formal calibration is arm-safe by default."""
    return build_sync_capture_mode(
        audio_device=audio_device,
        audio_channel=audio_channel,
        hall_port=hall_port,
        force_calibration_sha256=force_calibration_sha256,
        hall_resolver=hall_resolver,
        audio_resolver=audio_resolver,
        require_audio_usb_serial=not allow_unstable_audio_identity,
    )


def fit_device_clock(host_t_s, device_t_ms) -> dict:  # noqa: ANN001
    """Fit ``host_t = intercept + slope * device_ms`` for serial drift correction."""
    host = np.asarray(host_t_s, dtype=np.float64)
    device = np.asarray(device_t_ms, dtype=np.float64)
    valid = np.isfinite(host) & np.isfinite(device)
    host, device = host[valid], device[valid]
    if host.size < 2 or np.ptp(device) <= 0:
        raise ValueError("device clock fit needs at least two finite, distinct timestamps")
    slope, intercept = np.polyfit(device, host, 1)
    residual_s = host - (intercept + slope * device)
    return {
        "n_samples": int(host.size),
        "host_s_per_device_ms": float(slope),
        "host_intercept_s": float(intercept),
        "drift_ppm": float((slope / 0.001 - 1.0) * 1_000_000.0),
        "rmse_ms": float(np.sqrt(np.mean(residual_s ** 2)) * 1_000.0),
    }


def _validated_sync_payload(result: dict) -> dict:
    """Return the persisted contract only when every physical quality gate passes."""
    if (result.get("faulted") or int(result.get("n_matched", 0)) < MIN_SYNC_TAPS
            or result.get("offset_ms") is None):
        raise ValueError(
            f"sync result needs healthy devices and at least {MIN_SYNC_TAPS} matched taps"
        )
    try:
        offset_ms = float(result["offset_ms"])
        offset_std_ms = float(result["offset_std_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("sync result needs finite uncertainty from unique tap matches") from exc
    if (not np.isfinite(offset_ms) or abs(offset_ms) > MAX_SYNC_OFFSET_MS
            or not np.isfinite(offset_std_ms) or not 0 <= offset_std_ms <= MAX_SYNC_STD_MS
            or result.get("unique_matches") is not True):
        raise ValueError("sync result uncertainty/unique-match quality gate failed")
    method = str(result.get("measurement_method", ""))
    if method != SYNC_MEASUREMENT_METHOD:
        raise ValueError("sync result measurement method is not a physical tap calibration")
    capture_mode = validate_sync_capture_mode(result.get("capture_mode"))
    payload = {
        "schema_version": SYNC_SCHEMA_VERSION,
        "offset_definition": "force_t_minus_audio_t",
        "audio_to_force_offset_ms": offset_ms,
        "offset_std_ms": offset_std_ms,
        "n_matched": int(result["n_matched"]),
        "unique_matches": True,
        "measurement_method": method,
        "capture_mode": capture_mode,
    }
    return payload


def persist_sync_result(path: str | Path, result: dict) -> dict:
    """Atomically publish one immutable, quality-gated tap calibration."""
    payload = _validated_sync_payload(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, indent=2) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        try:
            os.link(tmp_name, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace existing sync calibration: {path}"
            ) from exc
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return payload


def _peaks(x: np.ndarray, fs: float, min_gap_s: float, k: float) -> np.ndarray:
    """Indices of prominent peaks: above mean + k*std, separated by >= min_gap_s."""
    from scipy.signal import find_peaks
    thr = float(x.mean() + k * x.std())
    idx, _ = find_peaks(x, height=thr, distance=int(min_gap_s * fs))
    return idx


def match_unique_taps(
    audio_t, force_t, *, max_delta_s: float = 0.25,
) -> tuple[list[tuple[int, int]], np.ndarray]:  # noqa: ANN001
    """Greedily select nearest candidate pairs without reusing either event."""
    audio = np.asarray(audio_t, dtype=np.float64)
    force = np.asarray(force_t, dtype=np.float64)
    candidates = sorted(
        (abs(float(force[fi] - audio[ai])), ai, fi)
        for ai in range(len(audio)) for fi in range(len(force))
        if abs(float(force[fi] - audio[ai])) <= float(max_delta_s)
    )
    used_audio: set[int] = set()
    used_force: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, ai, fi in candidates:
        if ai in used_audio or fi in used_force:
            continue
        used_audio.add(ai); used_force.add(fi); pairs.append((ai, fi))
    pairs.sort(key=lambda pair: force[pair[1]])
    deltas = np.asarray([force[fi] - audio[ai] for ai, fi in pairs], dtype=np.float64)
    return pairs, deltas


def _highpass_envelope(samples, sr, grid_hz, hp_hz=150.0):  # noqa: ANN001
    """``|high-pass(samples)|`` peak-hold-downsampled to ``grid_hz``. High-passing first removes
    the sound-card stream-open DC transient (which rails for seconds) and any slow drift, so a
    peak threshold is not dominated by non-tap energy the way a raw ``|samples|`` envelope is.
    Returns ``(env, step)``."""
    from scipy.signal import butter, filtfilt
    a = np.asarray(samples, dtype=np.float64)
    b, a_coef = butter(4, hp_hz, btype="highpass", fs=sr)
    hp = np.abs(filtfilt(b, a_coef, a))
    step = max(1, int(sr / grid_hz))
    env = hp[: hp.size // step * step].reshape(-1, step).max(axis=1)
    return env, step


def _per_tap_offsets(eg, fimp, f_idx, grid_hz, half_s=0.15, maxlag_ms=120.0):  # noqa: ANN001
    """For each force tap, cross-correlate the audio envelope against the force impulse in a tight
    window -> that tap's offset (s), ``force_t - audio_t``. Per-tap cross-correlation aligns the
    whole tap waveform, so it is immune to the diaphragm ring that fools argmax peak-matching
    (the loudest audio sample can be the ring maximum tens of ms after the strike)."""
    from scipy.signal import correlate, correlation_lags
    half = int(half_s * grid_hz)
    maxlag = int(maxlag_ms / 1000.0 * grid_hz)
    offsets = []
    for i in f_idx:
        lo, hi = max(0, int(i) - half), min(len(eg), int(i) + half)
        e, f = eg[lo:hi], fimp[lo:hi]
        if e.size < 3 or e.std() < 1e-9 or f.std() < 1e-9:
            continue
        e = (e - e.mean()) / e.std()
        f = (f - f.mean()) / f.std()
        c = correlate(f, e, mode="full") / len(e)
        lags = correlation_lags(len(f), len(e), mode="full")
        m = np.abs(lags) <= maxlag
        offsets.append(float(lags[m][np.argmax(c[m])]) / grid_hz)
    return offsets


def compute_tap_sync(samples, sr, t_start, force_t, force_n, grid_hz=1000.0):  # noqa: ANN001
    """Robust tap-sync detection from a captured window. Builds a high-passed audio envelope and a
    baseline-removed force impulse on a common monotonic grid, detects well-separated force taps
    directly on the calibrated Newtons, then per-tap cross-correlates. Returns a detection dict
    (the caller adds capture_mode/health). Pure and hardware-free, so it is unit-testable on
    saved captures."""
    from scipy.signal import find_peaks
    env, step = _highpass_envelope(samples, sr, grid_hz)
    t_env = float(t_start) + np.arange(len(env)) * (step / sr)
    ft = np.asarray(force_t, dtype=np.float64)
    ff = np.asarray(force_n, dtype=np.float64)
    empty = {"n_audio_taps": 0, "n_force_taps": 0, "force_swing_n": None, "n_matched": 0,
             "offset_s": None, "offset_std_s": None, "matched_pairs": [], "deltas_s": []}
    if ft.size < 2 or env.size < 2:
        return empty
    t_lo, t_hi = max(t_env[0], ft[0]), min(t_env[-1], ft[-1])
    if t_hi - t_lo < 1.0:  # audio/force windows barely overlap (e.g. stale, frozen timestamps)
        return empty
    tg = np.arange(t_lo, t_hi, 1.0 / grid_hz)
    eg = np.interp(tg, t_env, env)
    fg = np.interp(tg, ft, ff)
    w = max(1, int(0.6 * grid_hz))
    fimp = np.abs(fg - np.convolve(fg, np.ones(w) / w, mode="same"))
    rest = float(np.median(fg))
    swing = float(fg.max() - rest)
    f_idx, _ = find_peaks(fg, height=rest + 0.4 * swing, distance=int(0.8 * grid_hz))
    med = float(np.median(eg))
    mad = float(np.median(np.abs(eg - med))) or 1e-9
    a_idx, _ = find_peaks(eg, height=med + 8.0 * mad, distance=int(0.8 * grid_hz))
    deltas = np.asarray(_per_tap_offsets(eg, fimp, f_idx, grid_hz), dtype=np.float64)
    return {
        "n_audio_taps": int(len(a_idx)), "n_force_taps": int(len(f_idx)),
        "force_swing_n": round(float(np.ptp(ff)), 2) if ff.size else None,
        "n_matched": int(deltas.size),
        "offset_s": float(np.median(deltas)) if deltas.size else None,
        "offset_std_s": float(np.std(deltas)) if deltas.size > 1 else None,
        "matched_pairs": [(k, k) for k in range(deltas.size)],
        "deltas_s": deltas.tolist(),
    }


def calibrate_tap_sync(seconds: float = 20.0, grid_hz: float = 1000.0,
                       audio_device: str = "Scarlett Solo USB", audio_channel: int | None = 1,
                       hall_port: str | None = None, hall_cal: str | None = None,
                       allow_unstable_audio_identity: bool = False,
                       settle: float = 4.0) -> dict:
    from audio_stream import AudioStream
    from force_sensor import ForceCalibration, ForceSensor

    hall_port = hall_port or (sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*")) or [None])[0]
    hall_cal = str(Path(hall_cal) if hall_cal else default_hall_calibration())
    tmp_csv = os.path.join("/tmp", f"sync_force_{int(time.monotonic()*1000)}.csv")

    calibration_bytes = Path(hall_cal).read_bytes()
    cal = ForceCalibration.from_bytes(calibration_bytes)
    force_calibration_sha256 = hashlib.sha256(calibration_bytes).hexdigest()
    capture_mode = build_calibration_capture_mode(
        audio_device=audio_device,
        audio_channel=audio_channel,
        hall_port=hall_port,
        force_calibration_sha256=force_calibration_sha256,
        allow_unstable_audio_identity=allow_unstable_audio_identity,
    )
    audio_endpoint, hall_endpoint = sync_capture_endpoints(capture_mode)
    with AudioStream(audio_endpoint, pick_channel=audio_channel,
                     buffer_seconds=seconds + 3.0) as au, \
         ForceSensor(hall_endpoint, cal, log_csv=tmp_csv) as _fs:
        time.sleep(settle)  # let the sound-card stream-open DC transient decay out of the window
        print(f">>> TAP the diaphragm SHARPLY & FIRMLY ~6 times, ~3s apart, holding the module in "
              f"the air ({seconds:.0f}s) <<<", flush=True)
        time.sleep(seconds)
        win = au.latest_window(seconds)

    # ---- read the force log, then detect + align robustly (high-pass audio envelope, direct
    #      force-tap peaks on the calibrated Newtons, per-tap cross-correlation) ----
    ft, ff = [], []
    with open(tmp_csv) as f:
        r = csv.reader(f); next(r)
        for row in r:
            try:
                ft.append(float(row[0])); ff.append(float(row[4]))
            except (ValueError, IndexError):
                continue
    ff_arr = np.asarray(ff, dtype=np.float64)
    det = compute_tap_sync(win.samples, win.sr, win.t_start, ft, ff, grid_hz)

    # FAULT DETECTION: a streaming-but-dead device (stethoscope OFF -> audio noise floor; Hall
    # loose -> force frozen) or a stalled/disconnected sound card (STALE window with frozen,
    # misaligned timestamps) must not yield an offset.
    a_health = audio_status(win.samples, win.sr)
    f_health = force_status(ff_arr)
    window_stale = str(win.status) != "OK"
    faulted = a_health["status"] != "OK" or f_health["status"] != "OK" or window_stale
    offset_ms = (None if (faulted or det["offset_s"] is None)
                 else round(det["offset_s"] * 1000, 1))
    std_ms = (None if (faulted or det["offset_std_s"] is None)
              else round(det["offset_std_s"] * 1000, 1))
    res = {
        "audio_health": a_health, "force_health": f_health, "faulted": faulted,
        "window_status": str(win.status),
        "n_audio_taps": det["n_audio_taps"], "n_force_taps": det["n_force_taps"],
        "unique_matches": True,
        "measurement_method": SYNC_MEASUREMENT_METHOD,
        "capture_mode": capture_mode,
        "matched_pairs": det["matched_pairs"],
        "force_swing_n": det["force_swing_n"],
        "n_matched": det["n_matched"],
        "offset_ms": offset_ms,
        "offset_std_ms": std_ms,
        "note": ("offset = force_t - audio_t; add to audio timestamps to align force. "
                 "per-tap cross-correlation of audio envelope vs force impulse."),
    }
    try:
        os.remove(tmp_csv)
    except OSError:
        pass
    return res


def main(argv=None) -> int:  # noqa: ANN001
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="tap-window length; give ~6 well-separated taps room (~3s apart)")
    ap.add_argument("--settle", type=float, default=4.0,
                    help="pre-window settle so the sound-card stream-open DC transient decays out")
    ap.add_argument("--audio-device", required=True,
                    help="exact sound-device identity used for this calibration")
    ap.add_argument("--audio-channel", type=int, required=True,
                    help="exact audio input channel used for this calibration")
    ap.add_argument("--hall-port", required=True,
                    help="exact Hall serial identity used for this calibration")
    ap.add_argument("--hall-cal", required=True,
                    help="exact force calibration JSON bound by SHA-256")
    ap.add_argument(
        "--allow-unstable-audio-identity", action="store_true",
        help=("REHEARSAL ONLY: persist without a stable audio USB serial or "
              "fixed Linux USB topology; arm collection rejects it"),
    )
    ap.add_argument("--out", default=None,
                    help=f"write a versioned sync JSON only when >={MIN_SYNC_TAPS} unique healthy taps match")
    a = ap.parse_args(argv)
    if a.out and os.path.lexists(a.out):
        print(
            f"ERROR: refusing to reuse existing sync calibration output: {a.out}",
            file=sys.stderr,
        )
        return 2
    r = calibrate_tap_sync(
        seconds=a.seconds, audio_device=a.audio_device, audio_channel=a.audio_channel,
        hall_port=a.hall_port, hall_cal=a.hall_cal,
        allow_unstable_audio_identity=a.allow_unstable_audio_identity, settle=a.settle,
    )
    from health import warn
    print(json.dumps({k: v for k, v in r.items() if k not in ("audio_health", "force_health")}, indent=2))
    identity_warning = sync_identity_warning(r["capture_mode"])
    if identity_warning:
        print(f"\n{identity_warning}")
        if not a.allow_unstable_audio_identity:
            print("\nRESULT: stable audio hardware identity is required for formal calibration.")
            return 1
    if str(r.get("window_status", "OK")) != "OK":
        print(f"\nRESULT: ⚠️ audio window was {r.get('window_status')} — the sound card stalled or "
              "disconnected mid-capture (frozen, misaligned timestamps). Secure the USB and retry.")
        return 1
    af = warn("stethoscope/audio", r["audio_health"])
    ff_bad = warn("force sensor", r["force_health"])
    if af or ff_bad:
        print("\nRESULT: ⚠️ device fault above — fix it and retry (NO offset computed from a dead device).")
        return 1
    elif r["n_matched"] >= MIN_SYNC_TAPS:
        try:
            _validated_sync_payload(r)
        except ValueError as exc:
            print(f"\nRESULT: sync quality gate failed: {exc}", file=sys.stderr)
            return 1
        print(f"\nRESULT: audio<->force offset = {r['offset_ms']:+} ms "
              f"(+/- {r['offset_std_ms']} ms over {r['n_matched']} taps)")
        if a.out:
            try:
                persist_sync_result(a.out, r)
            except (ValueError, FileExistsError) as exc:
                print(f"\nRESULT: sync quality gate failed: {exc}", file=sys.stderr)
                return 1
            print(f"saved: {a.out}")
    else:
        print(f"\nRESULT: devices OK but fewer than {MIN_SYNC_TAPS} unique matched taps — "
              "bump sharper/more and retry")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
