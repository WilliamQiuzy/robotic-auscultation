"""Consolidated detection eval → a committed, traceable metrics artifact (scripts/perception/
perception_metrics.json) with provenance (git SHA + weight/data hashes). Runs the v9 deployment
detector on the real dark labels for: nominal PCK, occlusion robustness, depth-hole robustness, and
latency. Box-only (GPU). The JSON is the reproducible record the CI/repo otherwise lacked.

    AUSC_DATA=~/ausc_test python scripts/perception/eval_detection.py
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, cv2, torch

ROOT = os.environ.get("AUSC_REPO", os.path.expanduser("~/Desktop/wam-vla-medical/auscultation"))
DATA = os.path.expanduser(os.environ.get("AUSC_DATA", "~/ausc_test"))
os.chdir(ROOT); sys.path[:0] = ["ausc/perception", "ausc", "ausc/models", "ausc/training", "scripts"]
from tracking.detector import NormalsDetector
from tracking.geometry import Intrinsics
from train_landmark_human import ORDER
from _provenance import run_stamp, data_hash

CKPT = "outputs/perc_lmh_v9_occ/best.pt"
LABELS = os.path.join(DATA, "clothed_labels_a.json")


def pck(det, hole=0.0, occ=False):
    lab = json.load(open(LABELS)); S = os.path.join(DATA, "orbbec_session2")
    d = json.load(open(f"{S}/intrinsics.json"))
    intr = Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], 1.0)
    cap = cv2.VideoCapture(f"{S}/rgb.mp4"); E = []
    for stem, rec in lab.items():
        if stem == "_meta" or rec.get("clip") != "orbbec_session2":
            continue
        fr = rec["frame"]; cap.set(1, fr); ok, bgr = cap.read()
        if not ok:
            continue
        depth = np.load(f"{S}/depth/d{fr:04d}.npy").copy()
        if hole > 0:
            depth[np.random.RandomState(fr).rand(*depth.shape) < hole] = 0
        if occ:
            h = depth.shape[0]; depth[int(h * 0.35):int(h * 0.6)] = 0
        r = det.detect(bgr[:, :, ::-1], depth, intr)
        if r is None or r.confidence < 0.05:
            continue
        full = r.crop.det_to_full(r.xy_det); sc = 518.0 / r.crop.side
        for k, nm in enumerate(ORDER):
            E.append(np.hypot(full[k, 0] - rec["points"][nm][0], full[k, 1] - rec["points"][nm][1]) * sc)
    cap.release(); E = np.array(E)
    return round(float(np.mean(E < 25.9) * 100), 1), round(float(np.median(E)), 1)


def latency_p50(det, n=40):
    S = os.path.join(DATA, "orbbec_session3"); d = json.load(open(f"{S}/intrinsics.json"))
    intr = Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], 1.0)
    cap = cv2.VideoCapture(f"{S}/rgb.mp4"); ts = []
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        dep = np.load(f"{S}/depth/d{i:04d}.npy")
        torch.cuda.synchronize(); t0 = time.time(); det.detect(bgr[:, :, ::-1], dep, intr)
        torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000)
    cap.release(); return round(float(np.percentile(ts[5:], 50)), 1)


def main():
    det = NormalsDetector(CKPT, half=True)
    det_off = NormalsDetector(CKPT, half=True, fill_holes=False)
    m = {
        "weight": os.path.basename(os.path.dirname(CKPT)),
        "nominal": dict(zip(("pck5", "med_px"), pck(det))),
        "occlusion_band": dict(zip(("pck5", "med_px"), pck(det, occ=True))),
        "holes40_fill_off": dict(zip(("pck5", "med_px"), pck(det_off, hole=0.40))),
        "holes40_fill_on": dict(zip(("pck5", "med_px"), pck(det, hole=0.40))),
        "detector_latency_p50_ms_fp16_1080p": latency_p50(det),
        "provenance": {**run_stamp(repo=ROOT, ckpt_sha=data_hash(CKPT), labels_sha=data_hash(LABELS)),
                       "note": "n=1 subject/2 garments — single-subject record, NOT population-validated"},
    }
    out = "scripts/perception/perception_metrics.json"
    json.dump(m, open(out, "w"), indent=2)
    print(json.dumps(m, indent=2)); print("WROTE", out)


if __name__ == "__main__":
    main()
