"""OpenArm forward kinematics + damped-least-squares IK (position) from the URDF (yourdfpy).

Runtime/box module (needs yourdfpy + the URDF). The control LOGIC (safety/controller) is tested
without this via a mock arm; this provides the real motion on the box.
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
        lim = []
        for j in self.joints:
            jt = self.robot.joint_map[j]
            lo = getattr(jt.limit, "lower", -np.pi) if jt.limit else -np.pi
            hi = getattr(jt.limit, "upper", np.pi) if jt.limit else np.pi
            lim.append((lo, hi))
        self.limits = np.array(lim)                       # (7,2)

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
                lam: float = 0.01) -> np.ndarray:
        """One damped-least-squares step of the TCP toward target_xyz; returns new joints (clamped)."""
        p = self.tcp_pos(q)
        dx = np.asarray(target_xyz, float) - p
        n = np.linalg.norm(dx)
        if n > max_step_m:
            dx = dx / n * max_step_m                       # clamp Cartesian step (speed limit)
        J = self.jacobian_pos(q)
        dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(3), dx)
        return np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])
