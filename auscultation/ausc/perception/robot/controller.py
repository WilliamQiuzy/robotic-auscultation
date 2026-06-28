"""Auscultation arm controller: turns the perception stream into safe stethoscope placement.

Each tick it reads the ModelBasedTracker output (10 points with ``xyz_world`` / ``measured`` /
``position_std`` / ``status``), runs the safety state machine, and — only in APPROACH/CONTACT —
commands a clamped Cartesian step toward the target via an injected ``ArmInterface``. The arm
interface is a Protocol (real OpenArm, kinematic Sim, or a Mock for tests), so all control LOGIC is
hardware-free testable; only IK/motors are hardware/box-specific.
"""
from __future__ import annotations

from typing import List, Optional, Protocol

import numpy as np

from . import safety as sf


class ArmInterface(Protocol):
    def tcp_xyz(self) -> np.ndarray: ...                 # current TCP position in arm-base frame (3,)
    def servo_toward(self, target_xyz: np.ndarray, max_step_m: float) -> np.ndarray: ...  # one step
    def hold(self) -> None: ...                          # stop / hold current pose


def _view(points, name: str) -> Optional[sf.TargetView]:
    for p in points:
        if p.name == name:
            xw = getattr(p, "xyz_world", None)
            return sf.TargetView(name=p.name, xyz_world=(None if xw is None else np.asarray(xw, float)),
                                 measured=bool(p.measured), position_std_m=float(p.position_std),
                                 status=str(p.status))
    return None


class AuscultationController:
    def __init__(self, arm: ArmInterface, cfg: Optional[sf.SafetyConfig] = None):
        self.arm = arm
        self.cfg = cfg or sf.SafetyConfig()
        self.state = sf.SafetyState()
        self._last_t = None              # last perception timestamp seen
        self._last_advance_wall = None   # wall time the perception frame last advanced

    def command(self, target_name: str) -> None:
        self.state = sf.command(self.state, target_name)

    def estop(self, reason: str = "manual") -> None:
        self.arm.hold(); self.state = sf.estop(self.state, reason)

    def reset(self) -> None:
        self.state = sf.reset(self.state)

    def tick(self, tracker_points: List, perception_t: Optional[float] = None,
             now: Optional[float] = None, force_n: float = 0.0) -> sf.SafetyState:
        """One control cycle. ``tracker_points`` = MTPoint-like list from ModelBasedTracker.process().
        ``perception_t`` = the tracker output timestamp; ``now`` = wall clock. If the perception frame
        stops advancing (frozen pipeline) for > max_target_age_s of wall time, the target is treated as
        STALE → HOLD (never act on frozen data). Both default to the wall clock (live, never stale)."""
        import time
        now = time.monotonic() if now is None else now
        pt = now if perception_t is None else perception_t
        if self._last_t is None or pt > self._last_t + 1e-9:
            self._last_advance_wall = now          # a fresh perception frame arrived
        self._last_t = pt
        stale = self._last_advance_wall is None or (now - self._last_advance_wall) > self.cfg.max_target_age_s

        tv = _view(tracker_points, self.state.target_name) if self.state.target_name else None
        if stale and tv is not None:               # frozen pipeline → force HOLD via a degraded view
            tv = sf.TargetView(tv.name, tv.xyz_world, measured=False,
                               position_std_m=tv.position_std_m, status="COASTING")
        t = tv
        dist = np.inf
        if t is not None and t.xyz_world is not None and np.all(np.isfinite(t.xyz_world)):
            dist = float(np.linalg.norm(self.arm.tcp_xyz() - t.xyz_world))
        self.state = sf.step(self.state, t, self.cfg, dist_to_target_m=dist, force_n=force_n)

        if self.state.state in (sf.State.APPROACH,) and t is not None and t.xyz_world is not None:
            self.arm.servo_toward(t.xyz_world, self.cfg.max_step_m)   # safe clamped step toward target
        elif self.state.state in (sf.State.HOLD, sf.State.ESTOP, sf.State.IDLE, sf.State.ACQUIRING):
            self.arm.hold()                                          # never drive on a degraded target
        return self.state
