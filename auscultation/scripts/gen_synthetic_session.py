#!/usr/bin/env python3
"""Phase-0 synthetic session generator — a hardware-free dry run of the IL data pipeline.

Writes a session in the EXACT `record_session.py` on-disk format, but fills the three holes the
real sessions still have (docs/audio_reward_design.md): robot state/action, real contact, and
state-dependent audio. No physics engine — a scripted reach trajectory drives a phase variable that
couples the arm motion, the contact force, and the audio quality, so the whole converter → segments
→ reward chain can be validated end-to-end (and a tiny policy trained) before any MuJoCo/hardware.

Follower = OpenArm LEFT arm (7 DOF, joint1..7; the auscultation arm). action = leader joint targets,
observation.state = follower joints (targets + tracking lag/noise). Contact rises into the 3-5 N band
during the hold; audio is a real lung clip that we DEGRADE outside contact → good-contact window gets
high proxy-Q → reward>0 there. Phases: approach → hold-over-breathing (2-3 cycles) → retract.

    python scripts/gen_synthetic_session.py --seconds 20 --out data/sessions/session_sim0001
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # auscultation/
SR = 16000
NJ = 7                                   # OpenArm left arm joints 1..7
Q_HOME = np.array([0.0, -0.5, 0.0, 1.2, 0.0, 0.7, 0.0])
Q_CONTACT = np.array([0.15, 0.10, -0.10, 1.55, 0.05, 0.95, -0.05])
BREATH_HZ = 0.25                          # ~4 s per respiratory cycle
F_TARGET = 4.0                            # N, mid of the 3-5 N gate band


def _phase(t: np.ndarray, seconds: float) -> np.ndarray:
    """0 during approach → 1 across the hold → back toward 0 on retract (drives motion+force+audio)."""
    approach, retract = 0.25 * seconds, 0.85 * seconds
    p = np.clip((t - 0.0) / max(approach, 1e-6), 0, 1)          # ramp up over the approach
    down = np.clip((seconds - t) / max(seconds - retract, 1e-6), 0, 1)
    return np.minimum(p, down)


def _write_wav(path: str, y: np.ndarray) -> None:
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


def _clean_clip(seconds: float) -> np.ndarray:
    """Tile a real lung recording to `seconds` (fallback: filtered noise if no sample present)."""
    import soundfile as sf
    cands = sorted(glob.glob(os.path.join(ROOT, "data", "audio", "_samples", "hf_lung_v1", "*.wav")))
    n = int(seconds * SR)
    if cands:
        y, sr = sf.read(cands[0])
        if y.ndim > 1:
            y = y.mean(1)
        if sr != SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), SR); y = resample_poly(y, SR // g, int(sr) // g)
        y = np.tile(y, int(np.ceil(n / len(y))))[:n]
        return (y / (np.max(np.abs(y)) + 1e-9) * 0.6).astype(np.float32)
    return (0.05 * np.random.randn(n)).astype(np.float32)


def generate(out_dir: str, seconds: float = 20.0, control_hz: float = 50.0, seed: int = 0) -> str:
    import cv2
    rng = np.random.RandomState(seed)
    os.makedirs(out_dir, exist_ok=True)
    depth = None
    t0 = 1000.0                                              # synthetic monotonic origin
    n = int(seconds * control_hz); dt = 1.0 / control_hz
    tc = t0 + np.arange(n) * dt
    ph = _phase(tc - t0, seconds)
    breath = 0.5 * (1 - np.cos(2 * np.pi * BREATH_HZ * (tc - t0)))   # 0..1 chest-wall rise

    # robot: leader targets = smooth reach; follower = targets with 1st-order lag + noise
    tgt = Q_HOME[None, :] + ph[:, None] * (Q_CONTACT - Q_HOME)[None, :]
    foll = np.zeros_like(tgt); foll[0] = tgt[0]
    for k in range(1, n):
        foll[k] = 0.8 * foll[k - 1] + 0.2 * tgt[k] + rng.normal(0, 0.002, NJ)   # lag + jitter
    with open(os.path.join(out_dir, "robot.csv"), "w") as fp:
        fp.write("t," + ",".join(f"q{i+1}" for i in range(NJ)) + "," + ",".join(f"a{i+1}" for i in range(NJ)) + "\n")
        for k in range(n):
            fp.write(f"{tc[k]:.4f}," + ",".join(f"{v:.5f}" for v in foll[k]) + "," +
                     ",".join(f"{v:.5f}" for v in tgt[k]) + "\n")

    # force at ~100 Hz: in-band 4 N during contact, breathing-modulated; ~0 otherwise
    fhz = 100.0; nf = int(seconds * fhz)
    tf = t0 + np.arange(nf) / fhz
    phf = _phase(tf - t0, seconds); brf = 0.5 * (1 - np.cos(2 * np.pi * BREATH_HZ * (tf - t0)))
    force = phf * (F_TARGET + 0.8 * (brf - 0.5) * 2) + rng.normal(0, 0.05, nf)   # +/-0.8 N breathing
    force = np.clip(force, 0, None)
    hall_v = 1.6 - 0.05 * force                              # inverted-face plausible mapping
    with open(os.path.join(out_dir, "force.csv"), "w") as fp:
        fp.write("t,raw,hall_v,displacement_mm,force_n,status\n")
        for k in range(nf):
            raw = int(hall_v[k] / 3.3 * 65535)
            fp.write(f"{tf[k]:.4f},{raw},{hall_v[k]:.4f},{force[k]*0.4:.3f},{force[k]:.3f},OK\n")

    # audio: clean lung clip, degraded (noisy) wherever contact is poor -> state-dependent quality
    y = _clean_clip(seconds)
    contact_env = np.interp(np.arange(len(y)) / SR, tc - t0, ph)     # phase along the audio timeline
    noise = rng.normal(0, 1, len(y)).astype(np.float32)
    y = y * (0.3 + 0.7 * contact_env) + noise * 0.25 * (1 - contact_env)   # bad contact -> noisy
    _write_wav(os.path.join(out_dir, "audio.wav"), y / (np.max(np.abs(y)) + 1e-9) * 0.9)

    # camera: tiny procedural RGB (background + a marker that tracks the approach phase)
    H, W = 180, 240
    writer = cv2.VideoWriter(os.path.join(out_dir, "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             control_hz, (W, H))
    rgb_times = []
    for k in range(n):
        img = np.full((H, W, 3), (40, 30, 30), np.uint8)
        img[H // 3:, :] = (60, 45, 90)                       # a "back" region
        cx = int(W * 0.5); cy = int(H * (0.15 + 0.6 * ph[k]))  # probe descends with approach
        cv2.circle(img, (cx, cy), 8, (0, 200, 255), -1)
        writer.write(img); rgb_times.append((k, tc[k]))
    writer.release()
    with open(os.path.join(out_dir, "rgb_times.csv"), "w") as fp:
        fp.write("frame_index,t\n"); fp.writelines(f"{i},{t:.4f}\n" for i, t in rgb_times)

    # obs.csv (50 Hz fused precursor) + manifest, matching record_session.py
    with open(os.path.join(out_dir, "obs.csv"), "w") as fp:
        fp.write("t,force_n,force_status,audio_rms_db,audio_peak_db,cam_index,cam_t\n")
        for k in range(n):
            fi = min(int((tc[k] - t0) * fhz), nf - 1)
            fp.write(f"{tc[k]:.4f},{force[fi]:.3f},OK,-30.0,-12.0,{k},{tc[k]:.4f}\n")
    manifest = {
        "t0_monotonic": t0, "t0_wall": "SYNTHETIC", "duration_s": seconds, "control_hz": control_hz,
        "streams": ["audio", "cam", "force", "robot"], "synthetic": True,
        "audio": {"sr": SR, "t0_monotonic": t0, "device": "synthetic", "channel": 1},
        "force": {"port": "synthetic", "calibration": "synthetic", "n_rows_hint": f"{nf} @100Hz"},
        "camera": {"backend": "synthetic", "index": 0, "n_rgb_frames": n, "n_depth_frames": 0,
                   "has_depth": False},
        "robot": {"arm": "openarm_left", "n_joints": NJ, "state": "follower q1..7", "action": "leader targets"},
        "n_obs": n, "health": {"audio": {"status": "OK"}, "force": {"status": "OK"}},
    }
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(f"[sim] wrote {out_dir}: {n} steps @ {control_hz}Hz, {nf} force rows, robot 7-DOF, "
          f"contact hold ~{0.6*seconds:.0f}s")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sessions", "session_sim0001"))
    a = ap.parse_args()
    generate(a.out, seconds=a.seconds, control_hz=a.control_hz, seed=a.seed)
