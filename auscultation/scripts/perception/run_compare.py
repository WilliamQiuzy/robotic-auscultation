"""Before/after on the real yellow clip: LEFT = current per-point detector-primary tracker (jittery),
RIGHT = new ModelBasedTracker (shared-pose model). Side-by-side video + per-point jitter metric
(frame-to-frame, gross-motion removed)."""
import json, os, sys
from runtime_paths import activate_runtime

RUNTIME_ROLE = "linbox_realtime"
PATHS = activate_runtime(RUNTIME_ROLE)
ROOT, DATA, OUTPUTS = str(PATHS.repo_root), str(PATHS.data_root), str(PATHS.output_root)

import numpy as np, cv2
from tracking.detector import NormalsDetector, LUNG_ORDER, LUNG_INDEX
from tracking.geometry import Intrinsics
from tracking.tracker import StreamingAuscultationTracker, TrackerConfig
from tracking.model_tracker import ModelBasedTracker, ModelConfig

S = os.path.join(DATA, "orbbec_session3"); d = json.load(open(f"{S}/intrinsics.json"))
intr = Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], 1.0)
det = NormalsDetector(os.path.join(OUTPUTS, "perc_lmh_v7_adapt/best.pt"))

old = StreamingAuscultationTracker(det, intr, TrackerConfig(
    drift_audit_s=0.0, detection_agree_mm=1000.0, detect_conf_accept=0.0, var_flow=4e-3, init_confirm_frames=2))
new = ModelBasedTracker(det, intr, ModelConfig())


def draw(bgr, pts_pix, status_ok, title):
    ov = bgr.copy()
    proj = {}
    for k, nm in enumerate(LUNG_ORDER):
        p = pts_pix[k]
        if np.isfinite(p).all():
            proj[k] = p; x0, y0 = int(p[0]), int(p[1])
            c = (40, 40, 235) if nm[0] == "R" else (40, 220, 40)
            cv2.circle(ov, (x0, y0), 10, (255, 255, 255), -1); cv2.circle(ov, (x0, y0), 6, c, -1)
    for side in ("R", "L"):
        pl = [proj[LUNG_INDEX[f"{side}{i}"]] for i in range(1, 6) if LUNG_INDEX[f"{side}{i}"] in proj]
        if len(pl) > 1:
            cv2.polylines(ov, [np.array(pl, np.int32)], False, (40, 40, 235) if side == "R" else (40, 220, 40), 2)
    cv2.putText(ov, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return ov


cap = cv2.VideoCapture(f"{S}/rgb.mp4"); N = int(cap.get(7)); fps = 30.0
os.makedirs(OUTPUTS, exist_ok=True)
vw = cv2.VideoWriter(os.path.join(OUTPUTS, "track_compare.mp4"),
                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 360))
pix_old, pix_new = [], []
for fi in range(N):
    ok, bgr = cap.read()
    if not ok:
        break
    depth = np.load(f"{S}/depth/d{fi:04d}.npy")
    oo = old.process(bgr[:, :, ::-1], depth, t=fi / fps, depth_scale=1.0)
    nn = new.process(bgr[:, :, ::-1], depth, t=fi / fps, depth_scale=1.0)
    po = np.array([p.pixel if oo.initialized else (np.nan, np.nan) for p in oo.points]) if oo.initialized else np.full((10, 2), np.nan)
    pn = np.array([p.pixel for p in nn.points]) if nn.initialized else np.full((10, 2), np.nan)
    pix_old.append(po); pix_new.append(pn)
    L = draw(bgr, po, oo.initialized, "BEFORE: per-frame detector")
    R = draw(bgr, pn, nn.initialized, "AFTER: model-based (shared pose)")
    vw.write(np.hstack([cv2.resize(L, (640, 360)), cv2.resize(R, (640, 360))]))
cap.release(); vw.release()


def jitter(pix):
    A = np.array(pix)                                  # (T,10,2)
    dA = np.diff(A, axis=0)
    cm = np.nanmedian(dA, axis=1, keepdims=True)       # gross common-mode motion per frame
    j = np.linalg.norm(dA - cm, axis=2)               # (T-1,10) px
    j = j[np.isfinite(j)]
    # convert px->mm at ~1.15m
    return np.median(j) * 1.15 / d["fx"] * 1000, np.percentile(j, 90) * 1.15 / d["fx"] * 1000


jo = jitter(pix_old); jn = jitter(pix_new)
print(f"BEFORE per-point jitter: median {jo[0]:.1f}mm  p90 {jo[1]:.1f}mm")
print(f"AFTER  per-point jitter: median {jn[0]:.1f}mm  p90 {jn[1]:.1f}mm")
print(f"jitter reduction: median {jo[0]/max(jn[0],1e-6):.1f}x  p90 {jo[1]/max(jn[1],1e-6):.1f}x")
print("WROTE track_compare.mp4")
