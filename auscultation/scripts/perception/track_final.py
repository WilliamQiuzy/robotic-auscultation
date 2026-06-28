"""Final dynamic-tracking run on REAL yellow: clean overlay video (no diagnostic markers) +
per-frame canonical transform & 3D points saved, so breathing can be isolated by removing the
rigid body motion (live point in canonical frame = T^-1 @ xyz_camera; its residual = non-rigid)."""
import os, sys, json, numpy as np, cv2
ROOT = os.environ.get("AUSC_REPO", os.path.expanduser("~/Desktop/wam-vla-medical/auscultation")); os.chdir(ROOT)
sys.path[:0] = ["ausc/perception", "ausc", "ausc/models", "ausc/training", "scripts"]
sys.path.insert(0, os.path.expanduser(os.environ.get("AUSC_DATA","~/ausc_test")+""))
from track_real import NormalsDetector
from tracking.geometry import Intrinsics, invert_T
from tracking.detector import LUNG_ORDER, LUNG_INDEX
from tracking.tracker import StreamingAuscultationTracker, TrackerConfig

S = os.path.expanduser(os.environ.get("AUSC_DATA","~/ausc_test")+"/orbbec_session3"); d = json.load(open(f"{S}/intrinsics.json"))
intr = Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], 1.0)
det = NormalsDetector("outputs/perc_lmh_v7_adapt/best.pt")
cfg = TrackerConfig(drift_audit_s=0.0, detection_agree_mm=1000.0, detect_conf_accept=0.0,
                    var_flow=4e-3, init_confirm_frames=2)
tr = StreamingAuscultationTracker(det, intr, cfg)
cap = cv2.VideoCapture(f"{S}/rgb.mp4"); N = int(cap.get(7)); fps = 30.0
vw = cv2.VideoWriter(os.path.expanduser(os.environ.get("AUSC_DATA","~/ausc_test")+"/track_final.mp4"),
                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
STC = {"TRACKING": (40, 220, 40), "COASTING": (40, 200, 235), "LOST": (120, 120, 120),
       "INIT": (200, 200, 200)}
Ts, camP, ts = [], [], []
for fi in range(N):
    ok, bgr = cap.read()
    if not ok:
        break
    depth = np.load(f"{S}/depth/d{fi:04d}.npy")
    out = tr.process(bgr[:, :, ::-1], depth, t=fi / fps, depth_scale=1.0)
    ov = bgr.copy()
    if out.initialized:
        ts.append(fi / fps); Ts.append(out.T_cam_canon.copy())
        cam = np.array([p.xyz_camera for p in out.points])
        camP.append(cam.copy())
        proj = {}
        for k, tp in enumerate(out.points):
            if np.isfinite(tp.xyz_camera).all():
                uv = intr.project(tp.xyz_camera[None])[0]; proj[k] = uv
                x0, y0 = int(uv[0]), int(uv[1])
                c = (40, 40, 235) if tp.name[0] == "R" else (40, 220, 40)
                cv2.circle(ov, (x0, y0), 11, STC.get(tp.status, (180, 180, 180)), -1)
                cv2.circle(ov, (x0, y0), 7, c, -1)
                cv2.putText(ov, tp.name, (x0 + 10, y0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        for side in ("R", "L"):
            pl = [proj[LUNG_INDEX[f"{side}{i}"]] for i in range(1, 6) if LUNG_INDEX[f"{side}{i}"] in proj]
            if len(pl) > 1:
                cv2.polylines(ov, [np.array(pl, np.int32)], False,
                              (40, 40, 235) if side == "R" else (40, 220, 40), 2)
    cv2.putText(ov, f"v7 dynamic 3D track  f{fi}  {out.n_tracking}/10 tracking",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    vw.write(cv2.resize(ov, (960, 540)))
cap.release(); vw.release()
np.savez(os.path.expanduser(os.environ.get("AUSC_DATA","~/ausc_test")+"/track_canon.npz"),
         Ts=np.array(Ts), camP=np.array(camP), ts=np.array(ts), order=np.array(LUNG_ORDER))
print(f"WROTE track_final.mp4 ({len(ts)} frames) + track_canon.npz")
