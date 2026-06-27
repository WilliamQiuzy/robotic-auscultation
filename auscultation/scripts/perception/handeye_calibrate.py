"""Eye-to-hand calibration: Orbbec (fixed) → OpenArm base, so perception 3D points can be
commanded in the arm's base frame (closes the R4 loop; feeds ModelBasedTracker(extrinsics=)).

SETUP (camera fixed, target on the end-effector):
  1. Rigidly attach a ChArUco board to the arm's hand/TCP (any rigid offset is solved for).
  2. Close OrbbecViewer (it locks the camera).
  3. Float the auscultation arm in gravity-comp so it can be hand-guided AND logs joints, e.g.:
        cd ~/Desktop/Openarm_project && REC_CSV=demos/handeye.csv \
            openarm_teleop/build/gravity_comp right_arm can2 openarm_v10.urdf
  4. Run this script (capture mode). Hand-guide the arm to ~15-20 VARIED poses (vary all 3 rotation
     axes + position), pausing ~1.5 s at each with the board fully in the Orbbec view. It
     auto-captures a (board-in-camera, TCP-in-base) pair whenever the arm is still + board visible.
  5. It solves cv2.calibrateHandEye, reports a consistency residual, and writes the extrinsic JSON.

NO autonomous arm motion: the human floats + guides; this script only READS the camera and the
gravity-comp joint CSV. Safe over SSH (capture loop needs the 5090's display, or use --headless).

    python handeye_calibrate.py --arm right --joint-csv ~/Desktop/Openarm_project/demos/handeye.csv \
        --urdf ~/Desktop/Openarm_project/openarm_v10.urdf \
        --squares-x 5 --squares-y 7 --square-len 0.030 --marker-len 0.022 --dict 5X5_50
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np


# ── OpenArm forward kinematics (joints → TCP pose in arm base) ────────────────
class ArmFK:
    def __init__(self, urdf_path: str, arm: str = "right"):
        from yourdfpy import URDF
        self.robot = URDF.load(os.path.expanduser(urdf_path), load_meshes=False,
                               build_collision_scene_graph=False)
        self.joints = [f"openarm_{arm}_joint{i}" for i in range(1, 8)]
        self.base = f"openarm_{arm}_link0"
        self.tcp = f"openarm_{arm}_hand_tcp"
        assert all(j in self.robot.actuated_joint_names for j in self.joints), "joint names mismatch"

    def fk(self, q7: np.ndarray) -> np.ndarray:
        """7 joint angles → 4x4 T_tcp_base (TCP pose in the arm base frame)."""
        cfg = {j: float(a) for j, a in zip(self.joints, q7)}
        self.robot.update_cfg(cfg)
        return np.asarray(self.robot.get_transform(self.tcp, self.base), dtype=np.float64)


def read_latest_joints(csv_path: str, settle_rows: int = 8, vel_eps: float = 2e-3):
    """Return (q7, is_still) from the tail of the gravity-comp CSV (t,a1..a7,grip). is_still =
    the arm has been ~stationary over the last `settle_rows` samples (so a clean pose to capture)."""
    with open(os.path.expanduser(csv_path)) as f:
        lines = f.readlines()
    rows = [l.strip().split(",") for l in lines[-settle_rows:] if l and l[0].isdigit() or (l[:1] in "-0123456789")]
    rows = [r for r in rows if len(r) >= 8]
    if len(rows) < 2:
        return None, False
    arr = np.array([[float(x) for x in r[1:8]] for r in rows])
    q7 = arr[-1]
    still = float(np.abs(arr[-1] - arr[0]).max()) < vel_eps
    return q7, still


# ── ChArUco board pose in camera frame ───────────────────────────────────────
class CharucoPoseEstimator:
    def __init__(self, sx, sy, square_len, marker_len, dict_name, K, dist):
        d = getattr(cv2.aruco, f"DICT_{dict_name}")
        self.adict = cv2.aruco.getPredefinedDictionary(d)
        self.board = cv2.aruco.CharucoBoard((sx, sy), square_len, marker_len, self.adict)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.K, self.dist = K, dist

    def pose(self, rgb):
        """Return (T_target_cam 4x4, n_corners) or (None, 0)."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cc, cids, _, _ = self.detector.detectBoard(gray)
        if cids is None or len(cids) < 6:
            return None, 0
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            cc, cids, self.board, self.K, self.dist, None, None)
        if not ok:
            return None, 0
        T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rvec)[0]; T[:3, 3] = tvec.ravel()
        return T, len(cids)


def _split(T):
    return T[:3, :3], T[:3, 3]


def solve_eye_to_hand(T_tcp_base_list, T_target_cam_list):
    """Camera fixed, target on TCP. Returns T_cam_base (4x4): camera pose in the arm base frame.
    Eye-to-hand uses base→gripper (inverse of FK) with cv2.calibrateHandEye."""
    R_b2g, t_b2g, R_t2c, t_t2c = [], [], [], []
    for Ttb, Ttc in zip(T_tcp_base_list, T_target_cam_list):
        Tbg = np.linalg.inv(Ttb)                       # base→gripper(TCP)
        R, t = _split(Tbg); R_b2g.append(R); t_b2g.append(t)
        R, t = _split(Ttc); R_t2c.append(R); t_t2c.append(t)
    R_cb, t_cb = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c,
                                      method=cv2.CALIB_HAND_EYE_PARK)
    T_cam_base = np.eye(4); T_cam_base[:3, :3] = R_cb; T_cam_base[:3, 3] = t_cb.ravel()
    return T_cam_base


def consistency_residual(T_cam_base, T_tcp_base_list, T_target_cam_list):
    """T_base_target = T_base_cam · T_cam_target must be a CONSTANT rigid offset (target↔TCP).
    Report the spread of the implied TCP→target offset across poses = calibration quality (mm)."""
    offs = []
    for Ttb, Ttc in zip(T_tcp_base_list, T_target_cam_list):
        T_base_target = T_cam_base @ Ttc               # target in base (T_cam_base maps cam→base)
        T_tcp_target = np.linalg.inv(Ttb) @ T_base_target  # TCP→target offset; constant if calib is right
        offs.append(T_tcp_target[:3, 3])
    offs = np.array(offs)
    return float(np.linalg.norm(offs - offs.mean(0), axis=1).mean() * 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="right")
    ap.add_argument("--joint-csv", required=True)
    ap.add_argument("--urdf", default="~/Desktop/Openarm_project/openarm_v10.urdf")
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-len", type=float, default=0.030)   # metres
    ap.add_argument("--marker-len", type=float, default=0.022)
    ap.add_argument("--dict", default="5X5_50")
    ap.add_argument("--n-poses", type=int, default=18)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out", default="ausc/perception/calib/cam2base.json")
    args = ap.parse_args()

    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [os.path.join(HERE, "..", "..", "ausc", "perception")]
    from camera_stream import CameraStream
    from tracking.intrinsics import orbbec_intrinsics  # noqa

    fk = ArmFK(args.urdf, args.arm)
    cam = CameraStream(backend="orbbec", align_depth_to_color=True).start()
    time.sleep(1.0)
    intr = orbbec_intrinsics(cam._pipe)
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1.0]])
    dist = np.zeros(5)
    est = CharucoPoseEstimator(args.squares_x, args.squares_y, args.square_len,
                               args.marker_len, args.dict, K, dist)
    print(f"[handeye] camera {intr.width}x{intr.height} fx={intr.fx:.0f}; guide the arm to "
          f"{args.n_poses} varied poses (pause ~1.5s each, board in view).")

    T_tcp_base, T_target_cam = [], []
    last_capture = 0.0
    while len(T_tcp_base) < args.n_poses:
        f = cam.latest_frame()
        if f is None or f.frame is None:
            time.sleep(0.05); continue
        q7, still = read_latest_joints(args.joint_csv)
        Ttc, ncorn = est.pose(f.frame)
        ready = (q7 is not None and still and Ttc is not None
                 and time.time() - last_capture > 2.0)
        if ready:
            T_tcp_base.append(fk(q7)); T_target_cam.append(Ttc); last_capture = time.time()
            print(f"  captured pose {len(T_tcp_base)}/{args.n_poses} ({ncorn} corners)")
        if not args.headless:
            vis = cv2.cvtColor(f.frame, cv2.COLOR_RGB2BGR)
            cv2.putText(vis, f"poses {len(T_tcp_base)}/{args.n_poses}  corners={ncorn}  still={still}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("handeye (q to abort)", cv2.resize(vis, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            time.sleep(0.05)
    cam.stop() if hasattr(cam, "stop") else None
    if len(T_tcp_base) < 6:
        print("[handeye] too few poses; abort."); return

    T_cam_base = solve_eye_to_hand(T_tcp_base, T_target_cam)
    resid = consistency_residual(T_cam_base, T_tcp_base, T_target_cam)
    print(f"[handeye] solved T_cam_base from {len(T_tcp_base)} poses; consistency residual {resid:.1f} mm")
    print(T_cam_base)
    out = os.path.expanduser(args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"T_cam_base": T_cam_base.tolist(), "arm": args.arm, "n_poses": len(T_tcp_base),
               "residual_mm": resid, "convention": "eye-to-hand; X maps camera->arm base"},
              open(out, "w"), indent=2)
    print(f"[handeye] wrote {out}  (feed to ModelBasedTracker(extrinsics=T_cam_base) for base-frame points)")
    if resid > 10:
        print("[handeye] WARNING residual >10mm: add more varied poses / check board size & rigid mount.")


if __name__ == "__main__":
    main()
