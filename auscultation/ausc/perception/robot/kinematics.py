"""OpenArm forward kinematics + damped-least-squares IK (position) from the URDF (yourdfpy).

Runtime/box module (needs yourdfpy + the URDF). The control LOGIC (safety/controller) is tested
without this via a mock arm; this provides the real motion on the box.

Frame: all positions returned/consumed here are in the ARM-BASE frame ``openarm_{arm}_link0``. The
hand-eye extrinsic (scripts/perception/handeye_calibrate.py) is calibrated to this SAME base link,
so ``MTPoint.xyz_world`` and ``tcp_pos`` are directly comparable. (The URDF also has a ``world`` root
above link0 via two fixed joints: do NOT mix it in; that would offset every target by the link0
mounting transform.)
"""
from __future__ import annotations

import os
import numpy as np


class ArmKinematics:
    def __init__(self, urdf_path: str, arm: str = "left"):
        from yourdfpy import URDF
        self.robot = URDF.load(os.path.expanduser(urdf_path), load_meshes=False,
                               build_collision_scene_graph=False)
        self.joints = [f"openarm_{arm}_joint{i}" for i in range(1, 8)]
        self.base = f"openarm_{arm}_link0"
        self.tcp = f"openarm_{arm}_hand_tcp"
        lim, vel = [], []
        for j in self.joints:
            jt = self.robot.joint_map[j]
            jtype = getattr(jt, "type", "revolute")
            if jtype == "continuous":
                lim.append((-np.inf, np.inf))             # genuinely unbounded → clip is a no-op
            else:
                L = jt.limit
                lo = getattr(L, "lower", None) if L else None
                hi = getattr(L, "upper", None) if L else None
                if lo is None or hi is None or (lo == 0.0 and hi == 0.0):
                    # Fail LOUD: never invent ±π / [0,0] bounds for a safety clamp (a URDF swap that
                    # changes joint typing or drops limits must error, not silently mis-clamp).
                    raise ValueError(f"joint {j} (type={jtype}) has no usable position limits in the "
                                     f"URDF; refusing to guess bounds for a safety clamp")
                lim.append((float(lo), float(hi)))
            v = getattr(jt.limit, "velocity", None) if jt.limit else None
            vel.append(float(v) if v else 5.0)            # rad/s; fallback only if URDF omits velocity
        self.limits = np.array(lim)                       # (7,2)
        self.vel_limits = np.array(vel)                   # (7,) per-joint max angular velocity

    def fk(self, q: np.ndarray) -> np.ndarray:
        self.robot.update_cfg({j: float(a) for j, a in zip(self.joints, q)})
        return np.asarray(self.robot.get_transform(self.tcp, self.base), float)

    def tcp_pos(self, q: np.ndarray) -> np.ndarray:
        return self.fk(q)[:3, 3]

    def jacobian_pos(self, q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        p0 = self.tcp_pos(q); J = np.zeros((3, 7))
        for i in range(7):
            dq = q.copy(); dq[i] += eps
            J[:, i] = (self.tcp_pos(dq) - p0) / eps
        return J

    def ik_step(self, q: np.ndarray, target_xyz: np.ndarray, max_step_m: float = 0.02,
                lam: float = 0.01, dt: float = 1.0 / 30.0) -> np.ndarray:
        """One damped-least-squares step of the TCP toward target_xyz; returns new joints (clamped).

        Fails CLOSED on non-finite input/output (raises), grows damping near singularities so a weak
        Jacobian direction cannot whip a joint, and bounds the joint-space slew to the URDF per-joint
        velocity limits. ``dt`` is a fixed conservative per-tick slew bound matching the ~30 Hz control
        loop (not plumbed from the caller); keep it ≤ the real loop period so the cap can't exceed the
        true per-tick budget. The slew bound scales ``dq`` uniformly, preserving the Cartesian direction."""
        q = np.asarray(q, float); tx = np.asarray(target_xyz, float)
        if q.shape != (7,) or tx.shape != (3,) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(tx)):
            raise ValueError("ik_step: non-finite or wrong-shape joints/target")
        p = self.tcp_pos(q)
        dx = tx - p
        n = np.linalg.norm(dx)
        if n > max_step_m:
            dx = dx / n * max_step_m                       # clamp Cartesian step (speed limit)
        J = self.jacobian_pos(q)
        smin = float(np.linalg.svd(J, compute_uv=False).min())
        lam_eff = lam if smin > 0.02 else lam * (1.0 + (0.02 - smin) / 0.02 * 9.0)  # up to ~10× near singular
        dq = J.T @ np.linalg.solve(J @ J.T + (lam_eff ** 2) * np.eye(3), dx)
        if not np.all(np.isfinite(dq)):
            raise ValueError("ik_step: non-finite joint delta (degenerate Jacobian)")
        cap = self.vel_limits * dt                         # max per-joint slew this tick
        m = float((np.abs(dq) / np.maximum(cap, 1e-9)).max())
        if m > 1.0:
            dq = dq / m                                    # uniform scale: bound slew, keep direction
        return np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])
