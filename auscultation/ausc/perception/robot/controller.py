"""Auscultation arm controller: turns the perception stream into safe stethoscope placement.

Each tick it reads the ModelBasedTracker output (10 points with ``xyz_world`` / ``measured`` /
``position_std`` / ``status``), runs the safety state machine, and commands the arm via an injected
``ArmInterface``. Per state: APPROACH → one clamped Cartesian step toward the target; RETRACT → a
clamped step AWAY from the last target until a standoff is reached; every other state (CONTACT /
HOLD / ESTOP / IDLE / ACQUIRING) → a positive ``hold()`` so the arm is never left uncommanded.

Frame contract: ``xyz_world`` (tracker), ``tcp_xyz()`` (arm) and the IK targets are ALL in the arm
base frame (``openarm_{arm}_link0``); the hand-eye extrinsic is calibrated to that same link, so they
are directly comparable (the field name "world" is historical — it is the arm-base frame).

The arm interface is a Protocol (real OpenArm, kinematic Sim, or a Mock for tests), so all control
LOGIC is hardware-free testable; only IK/motors are hardware/box-specific.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

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
            # Every field read defensively: a legacy/foreign point type missing ``measured`` /
            # ``position_std`` / ``status`` reads as NOT-measured / infinite-uncertainty / LOST
            # (fail-safe) instead of raising mid-tick on a robot near a patient.
            return sf.TargetView(name=p.name, xyz_world=(None if xw is None else np.asarray(xw, float)),
                                 measured=bool(getattr(p, "measured", False)),
                                 position_std_m=float(getattr(p, "position_std", np.inf)),
                                 status=str(getattr(p, "status", "LOST")))
    return None


def _fingerprint(points) -> Tuple:
    """Content key of a tracker frame — used to detect a FROZEN pipeline when the caller does not
    thread ``perception_t``. A hung detector re-emits byte-identical points, so an unchanging key
    over > max_target_age_s of wall time is treated as stale (never act on frozen data)."""
    key = []
    for p in points:
        xw = getattr(p, "xyz_world", None)
        xw_t = None if xw is None else tuple(np.asarray(xw, float).ravel().tolist())
        key.append((getattr(p, "name", None), str(getattr(p, "status", "")),
                    bool(getattr(p, "measured", False)),
                    round(float(getattr(p, "position_std", 0.0)), 9), xw_t))
    return tuple(key)


class AuscultationController:
    def __init__(self, arm: ArmInterface, cfg: Optional[sf.SafetyConfig] = None):
        self.arm = arm
        self.cfg = cfg or sf.SafetyConfig()
        self.state = sf.SafetyState()
        self._last_key = None             # last freshness key (perception_t, or content fingerprint)
        self._last_advance_wall = None    # wall time the perception frame last advanced
        self._last_target_world = None    # last finite target position (for the retreat direction)

    def command(self, target_name: str) -> None:
        self.state = sf.command(self.state, target_name)

    def estop(self, reason: str = "manual") -> None:
        self.arm.hold(); self.state = sf.estop(self.state, reason)

    def reset(self) -> None:
        self.state = sf.reset(self.state)

    def tick(self, tracker_points: List, perception_t: Optional[float] = None,
             now: Optional[float] = None, force_n: float = 0.0) -> sf.SafetyState:
        """One control cycle. ``tracker_points`` = MTPoint-like list from ModelBasedTracker.process().
        ``perception_t`` = the tracker output timestamp (PASS THIS in production: ``out.t``);
        ``now`` = wall clock. Freshness: if ``perception_t`` advances we treat the frame as fresh; if
        the caller omits ``perception_t`` we fall back to a content fingerprint so a frozen pipeline
        (identical points re-emitted) still goes stale → HOLD after ``max_target_age_s``. We never
        assume "no timestamp ⇒ live"."""
        import time
        now = time.monotonic() if now is None else now
        # Prefer the perception timestamp; else use a content fingerprint as the freshness signal.
        frame_key = float(perception_t) if perception_t is not None else _fingerprint(tracker_points)
        if self._last_key is None or frame_key != self._last_key:
            self._last_advance_wall = now          # a fresh (advanced/changed) perception frame
        self._last_key = frame_key
        stale = (self._last_advance_wall is None
                 or (now - self._last_advance_wall) > self.cfg.max_target_age_s)

        tv = _view(tracker_points, self.state.target_name) if self.state.target_name else None
        if stale and tv is not None:               # frozen pipeline → force HOLD via a degraded view
            tv = sf.TargetView(tv.name, tv.xyz_world, measured=False,
                               position_std_m=tv.position_std_m, status="COASTING")
        t = tv
        dist = np.inf
        if t is not None and t.xyz_world is not None and np.all(np.isfinite(t.xyz_world)):
            dist = float(np.linalg.norm(self.arm.tcp_xyz() - t.xyz_world))
            self._last_target_world = np.asarray(t.xyz_world, float)   # remember for the retreat axis
        self.state = sf.step(self.state, t, self.cfg, dist_to_target_m=dist, force_n=force_n)

        st = self.state.state
        if st == sf.State.APPROACH and t is not None and t.xyz_world is not None \
                and np.all(np.isfinite(t.xyz_world)):
            try:
                self.arm.servo_toward(t.xyz_world, self.cfg.max_step_m)   # clamped step toward target
            except Exception as e:  # noqa: BLE001 — IK/unreachable/driver failure must not crash the loop
                self.arm.hold(); self.state = sf.estop(self.state, f"servo failed: {e}")
        elif st == sf.State.RETRACT:
            self._retract_step()                                         # commanded back-off
        else:
            self.arm.hold()       # CONTACT / HOLD / ESTOP / IDLE / ACQUIRING (and any unforeseen) → hold
        return self.state

    def _retract_step(self) -> None:
        """Command a clamped step AWAY from the last target until ``standoff_m`` is cleared, then
        self-complete the retract → IDLE. Never leave the arm parked against the patient."""
        tgt = self._last_target_world
        if tgt is None or not np.all(np.isfinite(tgt)):
            self.arm.hold(); self.state = sf.retract_done(self.state); return
        tcp = np.asarray(self.arm.tcp_xyz(), float)
        away = tcp - tgt
        gap = float(np.linalg.norm(away))
        if gap >= self.cfg.standoff_m:
            self.arm.hold(); self.state = sf.retract_done(self.state); return   # backed off → done
        direction = away / gap if gap > 1e-6 else np.array([0.0, 0.0, -1.0])
        retreat = tcp + direction * self.cfg.standoff_m
        try:
            self.arm.servo_toward(retreat, self.cfg.max_step_m)
        except Exception:  # noqa: BLE001
            self.arm.hold()
