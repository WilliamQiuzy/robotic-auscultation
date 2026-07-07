#!/usr/bin/env python3
"""Convert a recorded session -> LeRobot-style frames + a segment/reward layer (filtered-BC ready).

Implements Phase 2 of docs/audio_reward_design.md: split the continuous session into
respiratory-cycle segments (2-3 cycles) using the Hall force oscillation, compute per-segment
reward INGREDIENTS (contact stats + cheap audio quality proxies), write them via
ausc/audio/reward_schema.py, and emit one LeRobot frame per control step carrying a
`segment_index` that joins back to a Segment. Reward/weight stay DERIVED (recompute, don't
re-collect); a real Q model plugs in via `q_fn` on the 5090.

    python scripts/session_to_lerobot.py data/sessions/session_XXXX --out data/lerobot/session_XXXX

Output:
    <out>/segments.jsonl   reward_schema Segments (boundaries, contact, ingredients, doctor_label?)
    <out>/frames.parquet   per-step obs+action+segment_index (frames.jsonl if no pandas)
    <out>/info.json        schema, streams present, control_hz, what is MISSING (e.g. robot state)

Note: current sessions have NO robot state/action (teleop unwired) — the converter reports this in
info.json rather than fabricating it, so the gap is explicit before large-scale collection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Callable, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # auscultation/
sys.path[:0] = [os.path.join(ROOT, "ausc"), os.path.join(ROOT, "ausc", "audio")]
from reward_schema import (  # noqa: E402
    Contact, RewardIngredients, Segment, attach_reward, reward_qgate, save_segments, validate,
)
from session_io import Session  # noqa: E402

F_MIN, F_MAX = 3.0, 5.0        # contact gate band around the 4 N target (design §10)
TARGET_CYCLES = 3              # segment = up to this many respiratory cycles (2-3)
FALLBACK_SEG_S = 10.0          # if cycle detection is too weak, tile fixed-duration segments


# -- respiratory cycle detection from the Hall force oscillation (design §6-7) ---------------------

def detect_respiratory_onsets(ft: list[float], fn: list[float]) -> list[float]:
    """Inspiration onsets = rising zero-crossings of the breathing-band (0.1-0.5 Hz) force signal."""
    if len(ft) < 8:
        return []
    t = np.asarray(ft, float); f = np.asarray(fn, float)
    fs = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
    if fs < 1.0:
        return []
    x = f - np.mean(f)
    # zero-phase-ish band-pass ~0.1-0.5 Hz via difference of moving averages (breathing envelope)
    def movavg(sig: np.ndarray, win_s: float) -> np.ndarray:
        w = max(1, int(round(win_s * fs)))
        k = np.ones(w) / w
        return np.convolve(sig, k, mode="same")
    band = movavg(x, 1.0) - movavg(x, 6.0)         # keep ~0.15-1 Hz, drop DC drift and fast noise
    onsets: list[float] = []
    for i in range(1, len(band)):
        if band[i - 1] <= 0.0 < band[i]:           # rising zero crossing = inspiration onset
            if not onsets or t[i] - onsets[-1] > 1.5:   # refractory: cycles are >= ~1.5 s apart
                onsets.append(float(t[i]))
    return onsets


def _segment_spans(onsets: list[float], t0: float, t1: float) -> list[tuple[float, float, list[float]]]:
    """Group onsets into <=TARGET_CYCLES-cycle spans; fall back to fixed-duration tiling if too few."""
    onsets = [o for o in onsets if t0 <= o <= t1]
    if len(onsets) >= 3:
        spans = []
        for i in range(0, len(onsets) - 1, TARGET_CYCLES):
            grp = onsets[i:i + TARGET_CYCLES + 1]
            if len(grp) >= 2:
                spans.append((grp[0], grp[-1], grp[:-1]))
        if spans:  # clamp to fully cover [t0, t1] so every control frame lands in a segment
            spans[0] = (t0, spans[0][1], spans[0][2])
            spans[-1] = (spans[-1][0], t1, spans[-1][2])
            return spans
    n = max(1, int(math.ceil((t1 - t0) / FALLBACK_SEG_S)))
    step = (t1 - t0) / n
    return [(t0 + k * step, t0 + (k + 1) * step, []) for k in range(n)]


# -- cheap, model-free audio ingredients (no torch; real Q plugs in via q_fn) ----------------------

def _audio_ingredients(audio: Optional[np.ndarray], sr: int,
                       q_fn: Optional[Callable[[np.ndarray, int], float]]) -> RewardIngredients:
    if audio is None or len(audio) < sr // 4:
        return RewardIngredients(q_usable=0.0, heart_contamination=1.0, snr_db=None)
    w = np.hanning(len(audio))
    X = np.abs(np.fft.rfft(audio * w)) ** 2
    fr = np.fft.rfftfreq(len(audio), 1.0 / sr)
    tot = float(np.sum(X)) + 1e-12
    heart = float(np.sum(X[(fr >= 20) & (fr < 150)])) / tot          # low-band heart contamination
    lung = float(np.sum(X[(fr >= 90) & (fr < 1100)])) / tot          # lung band share
    snr_db = 10.0 * math.log10(max(lung, 1e-6) / max(1.0 - lung, 1e-6))
    if q_fn is not None:
        q = float(q_fn(audio, sr))                                   # real Q model (e.g. AST head)
    else:                                                            # cheap proxy: lung-dominant + low heart
        q = max(0.0, min(1.0, lung * (1.0 - heart)))
    return RewardIngredients(q_usable=q, heart_contamination=heart, snr_db=round(snr_db, 2))


def _contact(ft: list[float], fn: list[float], t0: float, t1: float) -> Contact:
    vals = [v for t, v in zip(ft, fn) if t0 <= t <= t1]
    if not vals:
        return Contact(force_mean_n=0.0, force_in_band_frac=0.0, force_stable=False, gate=False)
    a = np.asarray(vals, float)
    in_band = float(np.mean((a >= F_MIN) & (a <= F_MAX)))
    stable = bool(np.std(a) < 1.0)                                   # < 1 N wobble over the segment
    return Contact(force_mean_n=round(float(np.mean(a)), 3), force_in_band_frac=round(in_band, 3),
                   force_stable=stable, gate=bool(in_band > 0.6 and stable))


# -- build segments & frames -----------------------------------------------------------------------

def _read_force(sdir: str) -> tuple[list[float], list[float]]:
    ft, fn = [], []
    fp = os.path.join(sdir, "force.csv")
    if os.path.exists(fp):
        import csv
        with open(fp) as f:
            for r in list(csv.reader(f))[1:]:
                try:
                    ft.append(float(r[0])); fn.append(float(r[4]))
                except (ValueError, IndexError):
                    continue
    return ft, fn


def build_segments(s: Session, ft: list[float], fn: list[float], session_id: str,
                   q_fn: Optional[Callable[[np.ndarray, int], float]]) -> list[Segment]:
    onsets = detect_respiratory_onsets(ft, fn)
    spans = _segment_spans(onsets, s.t0, s.t0 + s.duration)
    segs: list[Segment] = []
    for i, (a, b, cyc) in enumerate(spans):
        audio = s.audio_window(b, seconds=b - a)
        seg = Segment(
            segment_id=f"{session_id}#seg{i:03d}", session_id=session_id,
            t_start=round(a, 4), t_end=round(b, 4), cycle_onsets=[round(c, 4) for c in cyc],
            n_cycles=max(1, len(cyc) if cyc else int(round((b - a) / 4.0))),
            audio_t_start=round(a, 4), audio_t_end=round(b, 4),
            contact=_contact(ft, fn, a, b),
            ingredients=_audio_ingredients(audio, s.audio_sr, q_fn),
        )
        segs.append(attach_reward(seg, reward_qgate, version="qgate-1"))
    for seg in segs:
        validate(seg)
    return segs


def _seg_index_at(segs: list[Segment], t: float) -> int:
    if not segs or t < segs[0].t_start:
        return 0
    for i, seg in enumerate(segs):
        if seg.t_start <= t < seg.t_end:
            return i
    return len(segs) - 1


def build_frames(s: Session, segs: list[Segment], control_hz: float, has_robot: bool) -> list[dict]:
    dt = 1.0 / control_hz
    n = int(s.duration * control_hz)
    frames = []
    for k in range(n):
        t = s.t0 + k * dt
        fnv, _ = s.force_at(t)
        si = _seg_index_at(segs, t)
        frame = {
            "episode_index": 0, "frame_index": k, "timestamp": round(k * dt, 4),
            "segment_index": si,
            "observation.contact_force_n": round(fnv, 3) if fnv is not None else None,
            "observation.contact_gate": bool(segs[si].contact.gate) if segs else False,
            "observation.q_usable": round(segs[si].ingredients.q_usable, 4) if segs else None,
            "observation.audio_t": round(t - s.t0, 4),      # pointer into the sidecar wav
            "observation.cam_t": round(t - s.t0, 4),        # pointer into rgb.mp4 (video sidecar)
            "reward.weight": round(segs[si].weight, 4) if segs and segs[si].weight is not None else None,
        }
        if has_robot:  # observation.state / action slot in when teleop is wired (design §8)
            frame["observation.state"] = None
            frame["action"] = None
        frames.append(frame)
    return frames


def convert(session_dir: str, out_dir: str,
            q_fn: Optional[Callable[[np.ndarray, int], float]] = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    s = Session(session_dir)
    session_id = os.path.basename(session_dir.rstrip("/"))
    ft, fn = _read_force(session_dir)
    lo, hi = s.t0 - 0.5, s.t0 + s.duration + 0.5   # window force to the session (drop log over-run)
    win = [(t, v) for t, v in zip(ft, fn) if lo <= t <= hi]
    ft = [t for t, _ in win]; fn = [v for _, v in win]
    has_robot = os.path.exists(os.path.join(session_dir, "robot.csv"))  # not produced yet
    segs = build_segments(s, ft, fn, session_id, q_fn)
    frames = build_frames(s, segs, s.manifest.get("control_hz", 50.0), has_robot)
    save_segments(os.path.join(out_dir, "segments.jsonl"), segs)
    frames_path = _write_frames(out_dir, frames)
    info = {
        "schema": "lerobot-frames+segments/1.0", "source_session": session_id,
        "control_hz": s.manifest.get("control_hz", 50.0), "n_frames": len(frames),
        "n_segments": len(segs), "audio_sidecar": "audio.wav", "video_sidecar": "rgb.mp4",
        "q_source": "real_q_fn" if q_fn else "PROXY (lung-band*·no-heart); plug real AST Q on the 5090",
        "streams_present": sorted(s.manifest.get("streams", [])),
        "MISSING": ([] if has_robot else ["observation.state (follower joints)", "action (leader targets)"])
                   + ([] if "audio" in s.manifest.get("streams", []) else ["audio"]),
        "frames_file": os.path.basename(frames_path),
    }
    json.dump(info, open(os.path.join(out_dir, "info.json"), "w"), indent=2)
    return info


def _write_frames(out_dir: str, frames: list[dict]) -> str:
    try:
        import pandas as pd
        p = os.path.join(out_dir, "frames.parquet")
        pd.DataFrame(frames).to_parquet(p)          # LeRobot packs frames as parquet
        return p
    except Exception:                               # no pandas/pyarrow -> inspectable jsonl fallback
        p = os.path.join(out_dir, "frames.jsonl")
        with open(p, "w") as f:
            for fr in frames:
                f.write(json.dumps(fr) + "\n")
        return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "data", "lerobot", os.path.basename(a.session_dir.rstrip("/")))
    info = convert(a.session_dir, out)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"-> {out}")
    if info["MISSING"]:
        print(f"[!] MISSING for training-ready IL data: {info['MISSING']}")
