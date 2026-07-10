"""Canonical LeRobot dataset from a recorded session — VLA-ready + WAM/RL, highest standard.

Produces the REAL LeRobot v2 dataset via `LeRobotDataset.create/add_frame/save_episode` (so stats,
tasks, and video encoding are exactly what VLA tooling — ACT / pi0 / OpenVLA-style — expects), rather
than a hand-rolled parquet. What each modality becomes:

  ④ arm     -> observation.state (joint pos) + action (leader joint targets)     [float32]
  ③ video   -> observation.images.<cam>  (encoded mp4, one row per camera frame)  [video]  ← VLA core
  ① force   -> observation.contact_force_n + observation.contact_gate             [float32]
  ② audio   -> observation.q_usable (quality scalar); raw wav stays a session sidecar (LeRobot has no
               native audio), recomputable reward lives in segments.jsonl
  language  -> `task` per frame (the auscultation-site instruction) — the "L" in VLA
  WAM/RL    -> next.reward (segment gate*Q) + next.done (episode end)

Dataset fps = CAMERA fps (one row per image, the VLA convention); the higher-rate force/control
streams are sampled in and also kept raw in the session. Runs in linbox's lerobot conda env.

    ~/miniconda3/envs/lerobot/bin/python scripts/session_to_lerobot_v2.py <session_dir> --repo-id ausc/demo --root <out>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from bisect import bisect_left

import numpy as np

DEFAULT_TASK = "auscultate the posterior lung"


def _read_csv_cols(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _read_robot(sdir):
    hdr, rows = _read_csv_cols(os.path.join(sdir, "robot.csv"))
    nj = (len(hdr) - 1) // 2
    t = np.array([float(r[0]) for r in rows])
    q = np.array([[float(x) for x in r[1:1 + nj]] for r in rows], np.float32)
    a = np.array([[float(x) for x in r[1 + nj:1 + 2 * nj]] for r in rows], np.float32)
    return t, q, a, nj


def _read_force(sdir):
    p = os.path.join(sdir, "force.csv")
    if not os.path.exists(p):
        return np.array([0.0]), np.array([0.0])
    _, rows = _read_csv_cols(p)
    t = np.array([float(r[0]) for r in rows]); f = np.array([float(r[4]) for r in rows])
    return t, f


def _nearest(ts, t):
    i = min(bisect_left(ts, t), len(ts) - 1)
    return min((i - 1, i), key=lambda k: abs(ts[k] - t)) if 0 <= i - 1 else i


def _read_targets(sdir):
    """Read targets.jsonl (point-conditioned collection): the active auscultation target over time.
    Returns (times, uv[N,2], xyz[N,3], names) or None if not present."""
    p = os.path.join(sdir, "targets.jsonl")
    if not os.path.exists(p):
        return None
    t, uv, xyz, names = [], [], [], []
    for line in open(p):
        if not line.strip():
            continue
        d = json.loads(line)
        t.append(float(d["t"])); uv.append(d.get("uv") or [0.0, 0.0])
        xyz.append(d.get("xyz") or [0.0, 0.0, 0.0]); names.append(d.get("name", "?"))
    if not t:
        return None
    return t, np.asarray(uv, np.float32), np.asarray(xyz, np.float32), names


def _segments(sdir):
    """Segment reward/gate/q over time, from segments.jsonl if present (via session_to_lerobot)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ausc"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ausc", "audio"))
    from session_io import Session
    from session_to_lerobot import _read_force as rf, build_segments
    s = Session(sdir)
    ft, fn = rf(sdir)
    lo, hi = s.t0 - 0.5, s.t0 + s.duration + 0.5
    win = [(t, v) for t, v in zip(ft, fn) if lo <= t <= hi]
    ft = [t for t, _ in win]; fn = [v for _, v in win]
    segs = build_segments(s, ft, fn, os.path.basename(sdir.rstrip("/")), q_fn=None)
    return segs, s.t0


def _seg_at(segs, t):
    for seg in segs:
        if seg.t_start <= t < seg.t_end:
            return seg
    return segs[-1] if segs else None


def build(session_dir, repo_id, root, fps=None, task=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import cv2

    manifest = json.load(open(os.path.join(session_dir, "manifest.json")))
    task = task or (manifest.get("task") or {}).get("instruction") or DEFAULT_TASK
    joints = [f"j{i+1}" for i in range(7)]
    rt, q, a, nj = _read_robot(session_dir)
    ftv, fnv = _read_force(session_dir)
    segs, t0 = _segments(session_dir)
    tgt = _read_targets(session_dir)                        # point-conditioned goal (optional)
    tgt_t = tgt[0] if tgt else None

    # camera frames (one dataset row per image = VLA convention)
    _, cam_rows = _read_csv_cols(os.path.join(session_dir, "rgb_times.csv"))
    cam_t = [(int(r[0]), float(r[1])) for r in cam_rows]
    cap = cv2.VideoCapture(os.path.join(session_dir, "rgb.mp4"))
    ok, fr = cap.read(); H, W = fr.shape[:2]
    cam_fps = fps or round(len(cam_t) / max(manifest.get("duration_s", 1), 1e-6))

    features = {
        "observation.state": {"dtype": "float32", "shape": (nj,), "names": joints},
        "observation.images.top": {"dtype": "video", "shape": [H, W, 3],
                                    "names": ["height", "width", "channels"]},
        "observation.contact_force_n": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.contact_gate": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.q_usable": {"dtype": "float32", "shape": (1,), "names": None},
        "action": {"dtype": "float32", "shape": (nj,), "names": joints},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    if tgt:  # point-conditioned: the auscultation target the doctor was directed to (goal for the policy)
        features["observation.target_point_2d"] = {"dtype": "float32", "shape": (2,), "names": ["u", "v"]}
        features["observation.target_point_3d"] = {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]}
    ds = LeRobotDataset.create(repo_id, cam_fps, features=features, root=root,
                               robot_type="openarm_left", use_videos=True)

    rtl = rt.tolist(); ftl = ftv.tolist()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n = len(cam_t)
    for k, (fi, t) in enumerate(cam_t):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ir = _nearest(rtl, t); iff = _nearest(ftl, t)
        seg = _seg_at(segs, t)
        rew = float(seg.reward) if seg and seg.reward is not None else 0.0
        gate = 1.0 if (seg and seg.contact.gate) else 0.0
        qv = float(seg.ingredients.q_usable) if seg else 0.0
        frame = {
            "observation.state": q[ir],
            "observation.images.top": rgb,
            "observation.contact_force_n": np.array([fnv[iff]], np.float32),
            "observation.contact_gate": np.array([gate], np.float32),
            "observation.q_usable": np.array([qv], np.float32),
            "action": a[ir],
            "next.reward": np.array([rew], np.float32),
            "next.done": np.array([k == n - 1], bool),
            "task": task,
        }
        if tgt:  # attach the active auscultation target at this time → point-conditioned goal + task
            it = _nearest(tgt_t, t)
            frame["observation.target_point_2d"] = tgt[1][it]
            frame["observation.target_point_3d"] = tgt[2][it]
            frame["task"] = f"auscultate the {tgt[3][it]}"    # the point the doctor was directed to
        ds.add_frame(frame)
    cap.release()
    ds.save_episode()
    print(f"[v2] {repo_id}: {n} frames @ {cam_fps}fps, task={task!r}, state/action {nj}-DOF, "
          f"images.top {H}x{W}, next.reward/done + stats -> {root}")
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--repo-id", default="ausc/demo")
    ap.add_argument("--root", required=True)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    build(a.session_dir, a.repo_id, a.root, a.fps, a.task)
