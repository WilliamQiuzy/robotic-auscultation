"""Per-frame latency benchmark for the production perception stack on real 1080p frames.
Reports p50/p95/p99 + a per-stage breakdown (detector forward vs tracker overhead), with vs without
surface-snap, on H200. The honest offline benchmark the industrial-grade assessment asked for.

    python scripts/perception/latency_bench.py --frames 120
"""
from __future__ import annotations
import argparse, json, os, sys, time
from runtime_paths import activate_runtime, require_h200_raw_release

RUNTIME_ROLE = "h200_evaluation"
PATHS = activate_runtime(RUNTIME_ROLE)
ROOT, DATA, OUTPUTS = str(PATHS.repo_root), str(PATHS.data_root), str(PATHS.output_root)
EVAL_DATA = str(require_h200_raw_release("AUSC_PERCEPTION_EVAL_RELEASE"))

import numpy as np, cv2, torch
from tracking.detector import NormalsDetector
from tracking.geometry import Intrinsics
from tracking.model_tracker import ModelBasedTracker, ModelConfig


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--ckpt", default=os.path.join(OUTPUTS, "perc_lmh_v9_occ/best.pt"))
    args = ap.parse_args()
    S = EVAL_DATA; d = json.load(open(f"{S}/intrinsics.json"))
    intr = Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], 1.0)
    det = NormalsDetector(args.ckpt)
    frames = []
    cap = cv2.VideoCapture(f"{S}/rgb.mp4")
    for fi in range(args.frames):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append((bgr[:, :, ::-1].copy(), np.load(f"{S}/depth/d{fi:04d}.npy")))
    cap.release()
    print(f"[latency] {len(frames)} frames @ {d['width']}x{d['height']}, ckpt={os.path.basename(os.path.dirname(args.ckpt))}")

    for snap in (False, True):
        tr = ModelBasedTracker(det, intr, ModelConfig(surface_snap=snap))
        tot, dt_det = [], []
        for i, (rgb, dep) in enumerate(frames):
            torch.cuda.synchronize(); t0 = time.time()
            det.detect(rgb, dep, intr)                       # stage: detector forward
            torch.cuda.synchronize(); t1 = time.time()
            tr.process(rgb, dep, t=i / 30.0, depth_scale=1.0)  # full tick (detect again + cloud/fit/smooth)
            torch.cuda.synchronize(); t2 = time.time()
            dt_det.append((t1 - t0) * 1000); tot.append((t2 - t1) * 1000)
        tot = np.array(tot[10:]); dt_det = np.array(dt_det[10:])
        overhead = tot - dt_det
        print(f"  surface_snap={str(snap):5s}: total p50 {pct(tot,50):.0f} p95 {pct(tot,95):.0f} "
              f"p99 {pct(tot,99):.0f} ms ({1000/pct(tot,50):.0f} Hz) | detector p50 {pct(dt_det,50):.0f} ms"
              f" | tracker-overhead p50 {max(pct(overhead,50),0):.0f} ms")


if __name__ == "__main__":
    main()
