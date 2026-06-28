"""Closed-loop safety/controller tests (mock arm; no hardware/IK/torch). These encode the core
guarantee: the arm only moves toward a currently-measured, low-uncertainty, world-framed target and
halts the instant it degrades — the gate that makes a stethoscope-placing robot safe."""
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from ausc.perception.robot.controller import AuscultationController  # noqa: E402
from ausc.perception.robot.safety import SafetyConfig, State  # noqa: E402


@dataclass
class P:
    name: str
    xyz_world: Optional[np.ndarray]
    measured: bool = True
    position_std: float = 0.003
    status: str = "TRACKING"


class MockArm:
    def __init__(self, tcp=(0., 0., 0.)):
        self.p = np.array(tcp, float); self.servo_calls = 0; self.hold_calls = 0

    def tcp_xyz(self):
        return self.p.copy()

    def servo_toward(self, target_xyz, max_step_m):
        self.servo_calls += 1
        d = np.asarray(target_xyz, float) - self.p; n = np.linalg.norm(d)
        self.p = self.p + (d if n <= max_step_m else d / n * max_step_m)
        return self.p.copy()

    def hold(self):
        self.hold_calls += 1


def _drive(ctrl, points_fn, n, force_fn=lambda i: 0.0):
    states = []
    for i in range(n):
        states.append(ctrl.tick(points_fn(i), force_n=force_fn(i)).state)
    return states


def test_valid_target_reaches_contact():
    arm = MockArm(); c = AuscultationController(arm)
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    c.command("R3")
    states = _drive(c, lambda i: tgt, 20)
    assert State.CONTACT in states
    assert arm.servo_calls > 0
    assert states[-1] in (State.CONTACT,)


def test_no_extrinsics_never_moves():
    arm = MockArm(); c = AuscultationController(arm)
    c.command("R3")
    _drive(c, lambda i: [P("R3", None)], 10)          # xyz_world None ⇒ no world frame
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_high_uncertainty_blocks_motion():
    arm = MockArm(); c = AuscultationController(arm)
    c.command("R3")
    _drive(c, lambda i: [P("R3", np.array([0, 0, .1]), position_std=0.05)], 10)  # 50mm std > 10mm
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_unmeasured_target_never_silently_driven():
    arm = MockArm(); c = AuscultationController(arm)
    c.command("R3")
    _drive(c, lambda i: [P("R3", np.array([0, 0, .1]), measured=False, status="COASTING")], 10)
    assert arm.servo_calls == 0


def test_occlusion_midapproach_halts_then_recovers():
    arm = MockArm(); c = AuscultationController(arm); c.command("R3")
    good = [P("R3", np.array([0., 0., 0.10]))]
    occ = [P("R3", np.array([0., 0., 0.10]), measured=False, status="COASTING")]
    # approach a bit, then occlude → HOLD (no further servo), then recover → APPROACH
    for _ in range(2):
        c.tick(good)
    assert c.state.state == State.APPROACH
    s_before = arm.servo_calls
    c.tick(occ)
    assert c.state.state == State.HOLD
    c.tick(occ)
    assert arm.servo_calls == s_before                # NO motion while target degraded
    c.tick(good)
    assert c.state.state == State.APPROACH            # recovered → resume


def test_persistent_occlusion_retracts():
    arm = MockArm(); c = AuscultationController(arm, SafetyConfig(hold_grace_frames=3)); c.command("R3")
    good = [P("R3", np.array([0., 0., 0.10]))]
    occ = [P("R3", np.array([0., 0., 0.10]), measured=False, status="COASTING")]
    c.tick(good); c.tick(good)
    states = _drive(c, lambda i: occ, 10)
    assert State.RETRACT in states                    # gave up safely after grace


def test_estop_latches_and_ignores_commands():
    arm = MockArm(); c = AuscultationController(arm); c.command("R3")
    c.tick([P("R3", np.array([0, 0, .1]))])
    c.estop("test")
    assert c.state.state == State.ESTOP
    c.command("L2")                                   # ignored while latched
    s = c.tick([P("L2", np.array([0, 0, .1]))])
    assert s.state == State.ESTOP and arm.servo_calls == 1   # only the one pre-estop tick moved...
    # (servo happened on the first tick in APPROACH; none after estop)


def test_stale_perception_halts():
    """A frozen perception pipeline (timestamp stops advancing) must HALT, even though it keeps
    reporting the last 'measured' target — never act on stale data."""
    arm = MockArm(); c = AuscultationController(arm, SafetyConfig(max_target_age_s=2.5)); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    for i in range(3):                                  # live: perception_t advances with wall clock
        c.tick(tgt, perception_t=float(i), now=float(i))
    assert c.state.state == State.APPROACH
    sc = arm.servo_calls
    c.tick(tgt, perception_t=2.0, now=10.0)             # frozen perception, wall jumped 8s >> 2.5s
    assert c.state.state == State.HOLD and arm.servo_calls == sc   # halted, no motion on stale data
    for i in range(11, 14):                             # perception resumes
        c.tick(tgt, perception_t=float(i), now=float(i))
    assert c.state.state in (State.APPROACH, State.CONTACT)   # resumed motion


def test_overforce_triggers_estop():
    arm = MockArm(); c = AuscultationController(arm); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    c.tick(tgt); c.tick(tgt)
    s = c.tick(tgt, force_n=20.0)                      # over-force ⇒ latch ESTOP
    assert s.state == State.ESTOP


# ── hardening regression tests (added after the production quality audit) ──────────────────

def test_nan_world_coord_never_moves():
    """A finite-but-NaN target coordinate must be rejected exactly like a missing one (distinct path
    from xyz_world=None)."""
    arm = MockArm(); c = AuscultationController(arm); c.command("R3")
    _drive(c, lambda i: [P("R3", np.array([0., 0., np.nan]))], 10)
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_inf_world_coord_never_moves():
    arm = MockArm(); c = AuscultationController(arm); c.command("R3")
    _drive(c, lambda i: [P("R3", np.array([0., 0., np.inf]))], 10)
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_empty_detector_output_never_moves():
    """Routine occlusion/init/dropout yields an empty point list → never act."""
    arm = MockArm(); c = AuscultationController(arm); c.command("R3")
    _drive(c, lambda i: [], 10)
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_commanded_landmark_absent_never_moves():
    """The commanded landmark is missing from the output (a different point is present) → never act."""
    arm = MockArm(); c = AuscultationController(arm); c.command("R3")
    _drive(c, lambda i: [P("L2", np.array([0., 0., 0.10]))], 10)
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_high_uncertainty_midapproach_halts_then_recovers():
    """position_std blowing past the ceiling DURING approach (still measured/TRACKING) must HALT —
    the continuous-uncertainty branch, not just the entry gate."""
    arm = MockArm(); c = AuscultationController(arm); c.command("R3")
    good = [P("R3", np.array([0., 0., 0.10]))]
    hi = [P("R3", np.array([0., 0., 0.10]), position_std=0.05)]   # 50 mm > 10 mm, measured+TRACKING
    c.tick(good); c.tick(good)
    assert c.state.state == State.APPROACH
    sc = arm.servo_calls
    c.tick(hi); assert c.state.state == State.HOLD
    c.tick(hi); assert arm.servo_calls == sc           # no motion while uncertain
    c.tick(good); assert c.state.state == State.APPROACH


def test_frozen_pipeline_no_timestamp_halts():
    """DEPLOYMENT signature: tick(points) with NO perception_t. A frozen pipeline re-emitting
    identical points must still go stale → HOLD (the staleness guard must not be inert by default)."""
    arm = MockArm(); c = AuscultationController(arm, SafetyConfig(max_target_age_s=2.5)); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    c.tick(tgt, now=0.0); c.tick(tgt, now=0.1)         # identical content, wall barely advanced
    assert c.state.state == State.APPROACH
    sc = arm.servo_calls
    c.tick(tgt, now=10.0)                               # 10 s with unchanged content ⇒ stale ⇒ HOLD
    assert c.state.state == State.HOLD and arm.servo_calls == sc


def test_overforce_during_stale_latches_estop():
    """Over-force must win over staleness: a freeze + over-force in the same tick ⇒ latched ESTOP,
    never a recoverable HOLD."""
    arm = MockArm(); c = AuscultationController(arm, SafetyConfig(max_target_age_s=2.5)); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    for i in range(3):
        c.tick(tgt, perception_t=float(i), now=float(i))
    assert c.state.state == State.APPROACH
    s = c.tick(tgt, perception_t=2.0, now=10.0, force_n=20.0)   # frozen AND over-force
    assert s.state == State.ESTOP
    s2 = c.tick(tgt, perception_t=11.0, now=11.0, force_n=0.0)  # latches even after recovery
    assert s2.state == State.ESTOP


class _FailingArm(MockArm):
    """servo_toward always raises (simulates IK divergence / unreachable / driver fault)."""
    def servo_toward(self, target_xyz, max_step_m):
        self.servo_calls += 1
        raise RuntimeError("simulated IK/servo failure")


def test_servo_failure_estops_without_crashing():
    arm = _FailingArm(); c = AuscultationController(arm); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    s = c.tick(tgt)                                     # APPROACH → servo raises → caught → ESTOP
    assert s.state == State.ESTOP
    s2 = c.tick(tgt)                                    # latched, no crash on the next tick
    assert s2.state == State.ESTOP


def test_retract_commands_backoff():
    """RETRACT must actively back the TCP AWAY from the target (not leave it parked at the patient)."""
    arm = MockArm(); c = AuscultationController(arm, SafetyConfig(hold_grace_frames=2)); c.command("R3")
    tgt = np.array([0., 0., 0.10])
    good = [P("R3", tgt)]
    occ = [P("R3", tgt, measured=False, status="COASTING")]
    c.tick(good); c.tick(good)
    d0 = float(np.linalg.norm(arm.tcp_xyz() - tgt))
    states = _drive(c, lambda i: occ, 12)
    assert State.RETRACT in states
    d1 = float(np.linalg.norm(arm.tcp_xyz() - tgt))
    assert d1 > d0                                      # actually backed off, not stationary


def test_contact_degradation_holds_not_retracts():
    """At CONTACT the probe occludes its own landmark (measured→False is EXPECTED): the machine must
    STAY in CONTACT with the arm held (no servo), NOT auto-retract."""
    arm = MockArm(); c = AuscultationController(arm); c.command("R3")
    near = np.array([0., 0., 0.005])                    # within reach_tol (8 mm) after one step
    c.tick([P("R3", near)]); c.tick([P("R3", near)])
    assert c.state.state == State.CONTACT
    sc = arm.servo_calls
    c.tick([P("R3", near, measured=False, status="COASTING")])
    assert c.state.state == State.CONTACT and arm.servo_calls == sc


def test_flickering_occlusion_eventually_retracts():
    """A target that recovers for one frame per window must NOT indefinitely defer RETRACT (leaky
    degraded counter). Tests the state-machine logic directly (dist=1.0 ⇒ no CONTACT; no arm ⇒ the
    RETRACT state is observable rather than instantly self-completing)."""
    from ausc.perception.robot.safety import (SafetyState, TargetView, command, step)  # noqa: E402
    cfg = SafetyConfig(hold_grace_frames=4)
    p = np.array([0., 0., 0.10])
    good = TargetView("R3", p, measured=True, position_std_m=0.003, status="TRACKING")
    occ = TargetView("R3", p, measured=False, position_std_m=0.003, status="COASTING")
    s = command(SafetyState(), "R3")
    s = step(s, good, cfg, dist_to_target_m=1.0)
    s = step(s, good, cfg, dist_to_target_m=1.0)
    assert s.state == State.APPROACH
    states = []
    for _ in range(20):
        s = step(s, occ, cfg, dist_to_target_m=1.0); states.append(s.state)
        s = step(s, occ, cfg, dist_to_target_m=1.0); states.append(s.state)
        s = step(s, good, cfg, dist_to_target_m=1.0); states.append(s.state)   # 1 good frame per 3
    assert State.RETRACT in states                      # leaky count still times out despite recovery


class _LegacyPoint:
    """A legacy/foreign tracker point lacking the ``measured`` field the safety gate keys on."""
    def __init__(self, name, xyz_world):
        self.name = name; self.xyz_world = xyz_world
        self.position_std = 0.003; self.status = "TRACKING"   # NO `measured` attribute


def test_legacy_point_without_measured_fails_safe():
    """Handing a point type without ``measured`` to tick() must fail SAFE (no motion, no crash),
    not raise AttributeError mid-tick."""
    arm = MockArm(); c = AuscultationController(arm); c.command("R3")
    for _ in range(6):
        c.tick([_LegacyPoint("R3", np.array([0., 0., 0.10]))])
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING
