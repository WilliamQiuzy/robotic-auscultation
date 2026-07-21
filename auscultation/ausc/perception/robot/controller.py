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
    def retract(self, direction_base: np.ndarray, max_step_m: float) -> np.ndarray: ...
    def hold(self) -> None: ...                          # stop / hold current pose


class ForceSampleLike(Protocol):
    force_n: float
    status: str
    age_s: float


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
        if (cfg is None or cfg.max_screened_force_n is None or cfg.hard_limit_force_n is None
                or not np.isfinite(cfg.max_screened_force_n)
                or not np.isfinite(cfg.hard_limit_force_n)
                or not np.isfinite(cfg.max_force_age_s) or cfg.max_force_age_s <= 0):
            raise ValueError("controller requires SafetyConfig derived from force calibration")
        self.arm = arm
        self.cfg = cfg
        self.state = sf.SafetyState()
        self._last_t = None              # last perception timestamp seen
        self._last_advance_wall = None   # wall time the perception frame last advanced
        self._last_approach_direction: Optional[np.ndarray] = None
        self._retract_origin: Optional[np.ndarray] = None

    def command(self, target_name: str) -> None:
        self.state = sf.command(self.state, target_name)
        self._last_approach_direction = None
        self._retract_origin = None

    def estop(self, reason: str = "manual") -> None:
        self._controller_estop(reason)

    def reset(self) -> None:
        self.state = sf.reset(self.state)
        self._last_approach_direction = None
        self._retract_origin = None

    def _controller_estop(self, reason: str) -> None:
        self.state = sf.estop(self.state, reason)
        try:
            self.arm.hold()
        except Exception as exc:  # noqa: BLE001 - keep ESTOP latched even if hold transport failed
            self.state = sf.estop(self.state, f"{reason}; hold command failed: {exc}")

    def _safe_hold(self) -> None:
        try:
            self.arm.hold()
        except Exception as exc:  # noqa: BLE001 - a failed hold command is itself an emergency
            self.state = sf.estop(self.state, f"hold command failed: {exc}")

    def _force_value(
        self, sample: Optional[ForceSampleLike],
    ) -> tuple[Optional[float], Optional[str]]:
        """Validate the complete force-reading contract without coupling to its concrete class."""
        if sample is None:
            return None, "force sample missing"
        try:
            force_n = float(sample.force_n)
            age_s = float(sample.age_s)
            status = str(sample.status)
            valid = bool(getattr(sample, "valid", True))
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            return None, f"force sample invalid: {exc}"
        if not valid:
            return None, "force sample marked invalid"
        if status not in {"OK", "WARN", "SCREENED_LIMIT", "HARD_LIMIT"}:
            return None, f"force sample invalid status {status!r}"
        if not np.isfinite(force_n):
            return None, "force sample force_n is not finite"
        if not np.isfinite(age_s) or age_s < 0.0:
            return None, "force sample age_s is invalid"
        if age_s > self.cfg.max_force_age_s:
            return None, (f"force sample is stale ({age_s:.3f}s > "
                          f"{self.cfg.max_force_age_s:.3f}s)")
        # Status is a safety signal, not decoration. Preserve the more severe
        # interpretation when a corrupted/inconsistent numeric field is lower.
        if status == "HARD_LIMIT":
            force_n = max(force_n, float(self.cfg.hard_limit_force_n))
        elif status == "SCREENED_LIMIT":
            force_n = max(force_n, float(self.cfg.max_screened_force_n))
        return force_n, None

    def _retract_tick(self) -> None:
        direction = self._last_approach_direction
        if direction is None or direction.shape != (3,) or not np.all(np.isfinite(direction)):
            self._controller_estop("retract has no valid recorded approach direction")
            return
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            self._controller_estop("retract approach direction is zero")
            return
        retreat = -direction / norm
        try:
            now_xyz = np.asarray(self.arm.tcp_xyz(), dtype=float)
        except Exception as exc:  # noqa: BLE001
            self._controller_estop(f"retract TCP read failed: {exc}")
            return
        if now_xyz.shape != (3,) or not np.all(np.isfinite(now_xyz)):
            self._controller_estop("retract TCP read returned an invalid position")
            return
        if self._retract_origin is None:
            self._retract_origin = now_xyz.copy()
        travelled = float(np.linalg.norm(now_xyz - self._retract_origin))
        remaining = self.cfg.max_retract_distance_m - travelled
        if remaining <= 1e-6:
            self._safe_hold()
            if self.state.state == sf.State.ESTOP:
                return
            self.state = sf.retract_done(self.state)
            self._retract_origin = None
            return
        step_m = min(self.cfg.max_step_m, remaining)
        try:
            new_xyz = np.asarray(self.arm.retract(retreat, step_m), dtype=float)
        except Exception as exc:  # noqa: BLE001 - any actuator/IK fault must latch safe
            self._controller_estop(f"retract command failed: {exc}")
            return
        motion = new_xyz - now_xyz
        motion_n = float(np.linalg.norm(motion))
        if (not np.all(np.isfinite(new_xyz)) or motion_n > step_m + 1e-6
                or (motion_n > 1e-9 and float(np.dot(motion / motion_n, retreat)) <= 0.0)):
            self._controller_estop("retract motion violated direction or step bound")
            return
        total = float(np.linalg.norm(new_xyz - self._retract_origin))
        if total > self.cfg.max_retract_distance_m + 1e-6:
            self._controller_estop("retract exceeded maximum travel")
        elif total >= self.cfg.max_retract_distance_m - 1e-6:
            self._safe_hold()
            if self.state.state == sf.State.ESTOP:
                return
            self.state = sf.retract_done(self.state)
            self._retract_origin = None

    def tick(self, tracker_points: List, perception_t: Optional[float] = None,
             now: Optional[float] = None,
             force_sample: Optional[ForceSampleLike] = None) -> sf.SafetyState:
        """One control cycle. ``tracker_points`` = MTPoint-like list from ModelBasedTracker.process().
        ``perception_t`` = the tracker output timestamp; ``now`` = wall clock. If the perception frame
        stops advancing (frozen pipeline) for > max_target_age_s of wall time, the target is treated as
        STALE → HOLD (never act on frozen data). ``force_sample`` must be an explicit fresh sensor
        reading with a recognized status; missing/invalid/stale readings latch ESTOP. Perception times
        default to the wall clock (live, never stale)."""
        import time
        force_n, force_fault = self._force_value(force_sample)
        if force_fault is not None:
            self._controller_estop(force_fault)
            return self.state
        assert force_n is not None
        now = time.monotonic() if now is None else now
        pt = now if perception_t is None else perception_t
        if self._last_t is None or pt > self._last_t + 1e-9:
            self._last_advance_wall = now          # a fresh perception frame arrived
        self._last_t = pt
        stale = self._last_advance_wall is None or (now - self._last_advance_wall) > self.cfg.max_target_age_s

        try:
            tv = _view(tracker_points, self.state.target_name) if self.state.target_name else None
        except Exception as exc:  # noqa: BLE001 - corrupt perception payloads must fail closed
            self._controller_estop(f"target view failed: {exc}")
            return self.state
        if stale and tv is not None:               # frozen pipeline → force HOLD via a degraded view
            tv = sf.TargetView(tv.name, tv.xyz_world, measured=False,
                               position_std_m=tv.position_std_m, status="COASTING")
        t = tv
        dist = np.inf
        try:
            tcp_xyz = np.asarray(self.arm.tcp_xyz(), dtype=float)
        except Exception as exc:  # noqa: BLE001 - encoder/kinematics faults must latch safe
            self._controller_estop(f"TCP read failed: {exc}")
            return self.state
        if tcp_xyz.shape != (3,) or not np.all(np.isfinite(tcp_xyz)):
            self._controller_estop("TCP read returned an invalid position")
            return self.state

        target_xyz = None
        if t is not None and t.xyz_world is not None:
            try:
                candidate = np.asarray(t.xyz_world, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                self._controller_estop(f"target view position invalid: {exc}")
                return self.state
            if candidate.shape != (3,):
                self._controller_estop("target view position must be a 3-vector")
                return self.state
            if np.all(np.isfinite(candidate)):
                target_xyz = candidate
                dist = float(np.linalg.norm(tcp_xyz - target_xyz))
        self.state = sf.step(self.state, t, self.cfg, dist_to_target_m=dist, force_n=force_n)

        if self.state.state in (sf.State.APPROACH,) and target_xyz is not None:
            delta = target_xyz - tcp_xyz
            delta_n = float(np.linalg.norm(delta))
            if delta_n > 1e-9:
                self._last_approach_direction = delta / delta_n
            try:
                new_xyz = np.asarray(
                    self.arm.servo_toward(target_xyz, self.cfg.max_step_m), dtype=float,
                )
            except Exception as exc:  # noqa: BLE001 - actuator/IK faults must latch safe
                self._controller_estop(f"approach servo failed: {exc}")
                return self.state
            motion = new_xyz - tcp_xyz if new_xyz.shape == (3,) else np.full(3, np.nan)
            motion_n = float(np.linalg.norm(motion))
            max_motion = min(self.cfg.max_step_m, delta_n)
            direction_ok = (motion_n > 1e-9 and delta_n > 1e-9
                            and float(np.dot(motion / motion_n, delta / delta_n)) > 0.0)
            if (new_xyz.shape != (3,) or not np.all(np.isfinite(new_xyz))
                    or not np.isfinite(motion_n) or motion_n > max_motion + 1e-6
                    or not direction_ok
                    or float(np.linalg.norm(target_xyz - new_xyz)) > delta_n + 1e-6):
                self._controller_estop("approach motion violated direction or step bound")
                return self.state
        elif self.state.state == sf.State.RETRACT:
            self._retract_tick()
        elif self.state.state in (sf.State.HOLD, sf.State.ESTOP, sf.State.IDLE,
                                  sf.State.ACQUIRING, sf.State.CONTACT):
            self._safe_hold()                                        # never drive on a degraded target
        return self.state
