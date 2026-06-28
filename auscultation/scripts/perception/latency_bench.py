"""Per-frame latency benchmark for the production perception stack on real 1080p frames.
Reports p50/p95/p99 + a per-stage breakdown (detector forward vs tracker overhead), with vs without
surface-snap, on the GPU box. The honest real-time number the industrial-grade assessment asked for.

    AUSC_DATA=~/ausc_test python scripts/perception/latency_bench.py --frames 120
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, cv2, torch

ROOT = os.environ.get("AUSC_REPO", os.path.expanduser("~/Desktop/wam-vla-medical/auscultation"))
DATA = os.path.expanduser(os.environ.get("AUSC_DATA", "~/ausc_test"))
os.chdir(ROOT); sys.path[:0] = ["ausc/perception", "ausc", "ausc/models", "ausc/training", "scripts"]
from tracking.detector import NormalsDetector
from tracking.geometry import Intrinsics
from tracking.model_tracker import ModelBasedTracker, ModelConfig


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="orbbec_session3")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--ckpt", default="outputs/perc_lmh_v9_occ/best.pt")
    args = ap.parse_args()
    S = os.path.join(DATA, args.session); d = json.load(open(f"{S}/intrinsics.json"))
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
