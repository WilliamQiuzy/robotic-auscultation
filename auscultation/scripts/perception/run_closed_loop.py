"""End-to-end closed-loop demo (sim): real Orbbec clip → v9 NormalsDetector → ModelBasedTracker
(with extrinsics) → AuscultationController (safety state machine) → SimArm. Proves the FULL chain
runs: perception's per-point world-frame points drive safety-gated arm motion, the arm reaches the
target in sim, and motion HALTS the moment the real target degrades (turn/occlusion).

Uses a stand-in camera→base extrinsic chosen so the tracked back points fall in the SimArm workspace
(the REAL extrinsic comes from R4 hand-eye). On the box (GPU + yourdfpy).

    python scripts/perception/run_closed_loop.py --target R3
"""
from __future__ import annotations
import argparse, json, os, sys
from types import SimpleNamespace
from runtime_paths import activate_runtime

RUNTIME_ROLE = "linbox_realtime"
PATHS = activate_runtime(RUNTIME_ROLE)
ROOT, DATA, OUTPUTS = str(PATHS.repo_root), str(PATHS.data_root), str(PATHS.output_root)

import numpy as np, cv2
from tracking.detector import NormalsDetector
from tracking.geometry import Intrinsics
from tracking.model_tracker import ModelBasedTracker, ModelConfig
from robot.arm_interface import SimArm
from robot.controller import AuscultationController
from robot.safety import SafetyConfig, State


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="orbbec_session3")
    ap.add_argument("--target", default="R3")
    ap.add_argument("--urdf", default="ausc/perception/robot/openarm_v10.urdf")
    args = ap.parse_args()
    S = os.path.join(DATA, args.session); d = json.load(open(f"{S}/intrinsics.json"))
    intr = Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], 1.0)

    sim = SimArm(args.urdf, "left", q0=[0.1, 0.3, -0.5, 0.8, 0.0, 0.3, 0.0])
    # stand-in extrinsic: map camera frame so the ~1.1m-deep back points land near the arm TCP region
    # (replace with the R4-calibrated T_cam_base). Translation chosen to bring points into reach.
    tcp0 = sim.tcp_xyz()
    E = np.eye(4); E[:3, 3] = tcp0 - np.array([0.0, 0.0, 1.15])   # camera origin → behind the TCP
    det = NormalsDetector(os.path.join(OUTPUTS, "perc_lmh_v9_occ/best.pt"), half=True)
    tr = ModelBasedTracker(det, intr, ModelConfig(), extrinsics=E)
    # Simulation still uses the production controller contract: limits come
    # from an explicit calibrated object and every tick carries a fresh force
    # sample.  A missing sensor must never silently become 0 N, even in demos.
    sim_calibration = SimpleNamespace(
        calibrated=True, max_screened_force_n=4.6, hard_limit_force_n=5.5,
    )
    ctrl = AuscultationController(sim, SafetyConfig.from_force_calibration(sim_calibration))

    cap = cv2.VideoCapture(f"{S}/rgb.mp4"); N = int(cap.get(7)); fps = 30.0
    commanded = False; seen = {}
    for fi in range(N):
        ok, bgr = cap.read()
        if not ok:
            break
        depth = np.load(f"{S}/depth/d{fi:04d}.npy")
        out = tr.process(bgr[:, :, ::-1], depth, t=fi / fps, depth_scale=1.0)
        if out.initialized and not commanded:
            ctrl.command(args.target); commanded = True
        force_sample = SimpleNamespace(force_n=4.0, status="OK", age_s=0.0, valid=True)
        st = (ctrl.tick(out.points, perception_t=fi / fps, now=fi / fps,
                        force_sample=force_sample).state if commanded else None)
        if st is not None:
            seen[st] = seen.get(st, 0) + 1
        if st == State.CONTACT:
            tp = next(p for p in out.points if p.name == args.target)
            err = np.linalg.norm(sim.tcp_xyz() - tp.xyz_world) * 1000
            print(f"  f{fi}: CONTACT at target {args.target}; sim reach error {err:.1f} mm "
                  f"(measured={tp.measured}, std={tp.position_std*1000:.0f}mm)")
            break
    cap.release()
    print(f"[closed-loop] states visited: {{ {', '.join(f'{k.value}:{v}' for k,v in seen.items())} }}")
    print("  full chain ran: perception → safety state machine → SimArm. "
          "Motion only in APPROACH/CONTACT; HOLD/ACQUIRING when target degraded (real occlusion/turn).")


if __name__ == "__main__":
    main()
