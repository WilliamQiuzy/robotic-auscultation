"""Safety state machine for stethoscope placement — the gate between perception and arm motion.

The robot must NEVER drive toward a point the perception layer did not actually observe. The
ModelBasedTracker already flags each point ``measured`` (observed this frame) / ``position_std``
(uncertainty, grows while extrapolated) / ``status``. This state machine consumes those and only
permits APPROACH motion toward a target that is currently measured, low-uncertainty, in a valid
base frame, with no fault — and HALTS the instant the target degrades during ACQUIRING/APPROACH/HOLD
(e.g. the arm/probe occludes it), escalating to RETRACT if it stays degraded.

CONTACT is deliberately exempt from the perception re-gate: once the stethoscope is placed it
occludes its own landmark (measured→False is EXPECTED), so auto-retracting on that would abort every
successful placement. At CONTACT the arm is positively held (no servo) and safety relies on the
over-force ESTOP (hardware force source — see controller gate) plus doctor teleop.

Pure logic, no robot/torch deps → fully unit-testable without hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class State(str, Enum):
    IDLE = "IDLE"              # nothing commanded
    ACQUIRING = "ACQUIRING"   # have a command; waiting for a valid, stable target
    APPROACH = "APPROACH"     # servoing toward the target
    CONTACT = "CONTACT"       # at the target (placing / regulating force)
    HOLD = "HOLD"             # target degraded mid-motion → paused, not retreating yet
    RETRACT = "RETRACT"       # backing off safely
    ESTOP = "ESTOP"           # latched emergency stop; manual reset only


@dataclass
class SafetyConfig:
    max_position_std_m: float = 0.010   # refuse/halt if target uncertainty exceeds this (10 mm)
    reach_tol_m: float = 0.008          # within this of the target ⇒ CONTACT
    hold_grace_frames: int = 15         # frames a target may stay degraded before we RETRACT
    require_world_frame: bool = True    # require camera→base extrinsics (no extrinsics ⇒ never move)
    max_force_n: float = 8.0            # (hardware) contact force ceiling ⇒ ESTOP if exceeded
    max_step_m: float = 0.02            # max commanded Cartesian step per tick (servo speed clamp)
    max_target_age_s: float = 0.25      # HALT if perception hasn't produced a fresh frame within this
    #                                     (frozen camera/pipeline must not be acted on as live data)
    standoff_m: float = 0.08            # RETRACT backs the TCP this far from the last target, then stops


@dataclass
class TargetView:
    """What the controller knows about the commanded target this tick (from the tracker)."""
    name: str
    xyz_world: Optional[np.ndarray]     # target in arm-base frame (None if no extrinsics)
    measured: bool                      # observed this frame (not extrapolated)
    position_std_m: float
    status: str                         # TRACKING / COASTING / LOST / INIT


@dataclass
class SafetyState:
    state: State = State.IDLE
    target_name: Optional[str] = None
    _degraded: int = 0
    fault: Optional[str] = None

    def target_ok(self, t: TargetView, cfg: SafetyConfig) -> bool:
        if t is None or t.name != self.target_name:
            return False
        if cfg.require_world_frame and (t.xyz_world is None or not np.all(np.isfinite(t.xyz_world))):
            return False
        return (t.measured and t.status == "TRACKING"
                and np.isfinite(t.position_std_m) and t.position_std_m <= cfg.max_position_std_m)


def command(s: SafetyState, target_name: str) -> SafetyState:
    """Operator/planner requests placement at a target. Ignored while ESTOP is latched."""
    if s.state == State.ESTOP:
        return s
    return SafetyState(State.ACQUIRING, target_name)


def estop(s: SafetyState, reason: str = "manual") -> SafetyState:
    return SafetyState(State.ESTOP, s.target_name, fault=reason)


def reset(s: SafetyState) -> SafetyState:
    """Manual recovery from ESTOP/RETRACT back to IDLE."""
    return SafetyState(State.IDLE)


def step(s: SafetyState, t: Optional[TargetView], cfg: SafetyConfig,
         dist_to_target_m: float = np.inf, force_n: float = 0.0) -> SafetyState:
    """Advance the state machine one tick given the latest target view + (hardware) sensors.

    Returns the next state. The controller issues servo motion only in APPROACH (toward the target)
    and RETRACT (away from it); CONTACT / HOLD / IDLE / ACQUIRING / ESTOP are positively held.
    The ``_degraded`` counter is leaky (carried, not reset, on a single recovered frame): it nets
    upward and eventually RETRACTs only when degraded frames OUTNUMBER recovered frames. An exact
    1:1 occlusion/recovery alternation is a deliberate non-timeout — each good frame is genuinely
    measured/low-uncertainty, so servoing on it stays safe.
    """
    if s.state == State.ESTOP:
        return s
    if force_n > cfg.max_force_n:                       # hardware over-force ⇒ latch ESTOP
        return estop(s, f"force {force_n:.1f}N > {cfg.max_force_n}N")

    if s.state == State.IDLE:
        return s

    if s.state == State.ACQUIRING:
        if s.target_ok(t, cfg):
            return SafetyState(State.APPROACH, s.target_name)
        return s                                        # keep waiting for a valid target

    if s.state == State.APPROACH:
        if not s.target_ok(t, cfg):                     # target degraded (e.g. arm occludes it)
            return SafetyState(State.HOLD, s.target_name, _degraded=s._degraded + 1)
        if dist_to_target_m <= cfg.reach_tol_m:
            return SafetyState(State.CONTACT, s.target_name)
        return SafetyState(State.APPROACH, s.target_name, _degraded=max(0, s._degraded - 1))

    if s.state == State.HOLD:
        if s.target_ok(t, cfg):                         # recovered → resume (leaky decay, NOT reset:
            #                                             net-positive degradation still times out;
            #                                             a sustained 1:1 flicker deliberately won't)
            return SafetyState(State.APPROACH, s.target_name, _degraded=max(0, s._degraded - 1))
        if s._degraded >= cfg.hold_grace_frames:        # gave up waiting → back off safely
            return SafetyState(State.RETRACT, s.target_name)
        return SafetyState(State.HOLD, s.target_name, _degraded=s._degraded + 1)

    if s.state == State.CONTACT:
        # NOT re-gated on perception: the placed probe occludes its own landmark (measured→False is
        # EXPECTED at contact), so auto-retracting on that would abort every placement. Arm is held
        # (no servo); safety here = the over-force ESTOP checked above + doctor teleop. Caller signals
        # placement_done() when finished.
        return s

    if s.state == State.RETRACT:
        return s                                        # caller signals retract-complete → reset

    return s


def placement_done(s: SafetyState) -> SafetyState:
    return SafetyState(State.RETRACT, s.target_name) if s.state == State.CONTACT else s


def retract_done(s: SafetyState) -> SafetyState:
    return SafetyState(State.IDLE) if s.state == State.RETRACT else s
