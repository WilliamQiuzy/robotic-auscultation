"""ArmInterface implementations: a kinematic Sim arm (box, for closed-loop sim) and a Real OpenArm
wrapper (box, openarm_can). The controller/safety logic depends only on the Protocol in
``controller.py`` and is tested with a mock — so these heavier impls are import-light and box-only.

SAFETY: RealArm does NOT energize on construction. Motors are enabled only via ``enable()`` (explicit),
and ``servo_toward`` is a clamped single step. Never run RealArm over SSH without a human present.
"""
from __future__ import annotations

import numpy as np

from .kinematics import ArmKinematics


class SimArm:
    """Kinematic sim: holds joint state, moves via DLS IK. No physics (contact/force come on hardware)."""

    def __init__(self, urdf_path: str, arm: str = "left", q0=None):
        self.kin = ArmKinematics(urdf_path, arm)
        self.q = np.zeros(7) if q0 is None else np.asarray(q0, float)

    def tcp_xyz(self) -> np.ndarray:
        return self.kin.tcp_pos(self.q)

    def servo_toward(self, target_xyz: np.ndarray, max_step_m: float) -> np.ndarray:
        self.q = self.kin.ik_step(self.q, target_xyz, max_step_m)
        return self.tcp_xyz()

    def hold(self) -> None:
        pass

    def retract(self, direction_base: np.ndarray, max_step_m: float) -> np.ndarray:
        direction = np.asarray(direction_base, dtype=float)
        norm = float(np.linalg.norm(direction))
        if direction.shape != (3,) or not np.isfinite(direction).all() or norm <= 1e-9:
            raise ValueError("retract direction must be a finite nonzero 3-vector")
        return self.servo_toward(self.tcp_xyz() + direction / norm * max_step_m, max_step_m)


class RealArm:
    """OpenArm (Damiao/CAN) wrapper. Reads joints via openarm_can; commands via posvel after IK.
    Motors stay OFF until enable() is called explicitly."""

    def __init__(self, urdf_path: str, can: str = "can3", arm: str = "left",
                 motor_types=None, send_ids=None, recv_ids=None):
        self.kin = ArmKinematics(urdf_path, arm)
        self.can, self.motor_types, self.send_ids, self.recv_ids = can, motor_types, send_ids, recv_ids
        self._arm = None
        self._enabled = False

    def _connect(self):
        import openarm_can as oa
        self._arm = oa.OpenArm(self.can, True)
        self._arm.init_arm_motors(self.motor_types, self.send_ids, self.recv_ids)

    def enable(self):
        if self._arm is None:
            self._connect()
        self._arm.enable_all(); self._enabled = True

    def read_q(self) -> np.ndarray:
        self._arm.refresh_all(); self._arm.recv_all()
        return np.array([m.get_position() for m in self._arm.get_arm().get_motors()][:7], float)

    def tcp_xyz(self) -> np.ndarray:
        return self.kin.tcp_pos(self.read_q())

    def servo_toward(self, target_xyz: np.ndarray, max_step_m: float) -> np.ndarray:
        if not self._enabled:
            raise RuntimeError("RealArm not enabled; call enable() (human present!) before motion")
        q = self.read_q()
        qn = self.kin.ik_step(q, target_xyz, max_step_m)
        import openarm_can as oa
        self._arm.get_arm().posvel_control_all([oa.PosVelParam(float(a), 5.0) for a in qn])
        return self.kin.tcp_pos(qn)

    def hold(self) -> None:
        if self._arm is not None and self._enabled:
            q = self.read_q()
            import openarm_can as oa
            self._arm.get_arm().posvel_control_all([oa.PosVelParam(float(a), 0.0) for a in q])

    def retract(self, direction_base: np.ndarray, max_step_m: float) -> np.ndarray:
        if not self._enabled:
            raise RuntimeError("RealArm not enabled; cannot retract")
        direction = np.asarray(direction_base, dtype=float)
        norm = float(np.linalg.norm(direction))
        if direction.shape != (3,) or not np.isfinite(direction).all() or norm <= 1e-9:
            raise ValueError("retract direction must be a finite nonzero 3-vector")
        return self.servo_toward(self.tcp_xyz() + direction / norm * max_step_m, max_step_m)
