"""Closed-loop safety/controller tests (mock arm; no hardware/IK/torch). These encode the core
guarantee: the arm only moves toward a currently-measured, low-uncertainty, world-framed target and
halts the instant it degrades — the gate that makes a stethoscope-placing robot safe."""
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from ausc.perception.robot.controller import AuscultationController  # noqa: E402
from ausc.perception.robot.safety import SafetyConfig, SafetyState, State  # noqa: E402


@dataclass
class P:
    name: str
    xyz_world: Optional[np.ndarray]
    measured: bool = True
    position_std: float = 0.003
    status: str = "TRACKING"


@dataclass(frozen=True)
class ForceReading:
    force_n: float = 0.0
    status: str = "WARN"
    age_s: float = 0.0
    valid: bool = True


def _force(force_n=0.0, status="WARN", age_s=0.0, valid=True):
    return ForceReading(force_n=force_n, status=status, age_s=age_s, valid=valid)


class MockArm:
    def __init__(self, tcp=(0., 0., 0.)):
        self.p = np.array(tcp, float); self.servo_calls = 0; self.hold_calls = 0
        self.retract_calls = 0; self.retract_directions = []; self.total_retract_distance_m = 0.0

    def tcp_xyz(self):
        return self.p.copy()

    def servo_toward(self, target_xyz, max_step_m):
        self.servo_calls += 1
        d = np.asarray(target_xyz, float) - self.p; n = np.linalg.norm(d)
        self.p = self.p + (d if n <= max_step_m else d / n * max_step_m)
        return self.p.copy()

    def hold(self):
        self.hold_calls += 1

    def retract(self, direction_base, max_step_m):
        direction = np.asarray(direction_base, float)
        direction /= np.linalg.norm(direction)
        self.retract_calls += 1
        self.retract_directions.append(direction.copy())
        self.p = self.p + direction * max_step_m
        self.total_retract_distance_m += max_step_m
        return self.p.copy()


class ServoFaultArm(MockArm):
    def __init__(self, fault):
        super().__init__()
        self.fault = fault

    def tcp_xyz(self):
        if self.fault == "tcp-exception":
            raise RuntimeError("encoder read failed")
        return super().tcp_xyz()

    def servo_toward(self, target_xyz, max_step_m):
        self.servo_calls += 1
        if self.fault == "servo-exception":
            raise RuntimeError("IK failed")
        direction = np.asarray(target_xyz, float) - self.p
        direction /= np.linalg.norm(direction)
        if self.fault == "wrong-direction":
            self.p = self.p - direction * (max_step_m / 2.0)
        elif self.fault == "oversized-step":
            self.p = self.p + direction * (max_step_m * 1.5)
        else:
            return super().servo_toward(target_xyz, max_step_m)
        return self.p.copy()


class SecondTcpFaultArm(MockArm):
    def __init__(self):
        super().__init__(); self.tcp_calls = 0

    def tcp_xyz(self):
        self.tcp_calls += 1
        if self.tcp_calls == 2:
            raise RuntimeError("encoder failed during retract")
        return super().tcp_xyz()


class HoldFaultArm(MockArm):
    def hold(self):
        self.hold_calls += 1
        raise RuntimeError("hold transport failed")


class BrokenTargetView:
    name = "R3"

    @property
    def xyz_world(self):
        raise RuntimeError("tracker payload corrupt")


def _cfg(**overrides):
    cal = SimpleNamespace(
        calibrated=True, max_screened_force_n=4.6, hard_limit_force_n=5.5,
    )
    return SafetyConfig.from_force_calibration(cal, **overrides)


def _controller(arm, **overrides):
    return AuscultationController(arm, _cfg(**overrides))


def _drive(ctrl, points_fn, n, force_fn=lambda i: _force()):
    states = []
    for i in range(n):
        states.append(ctrl.tick(points_fn(i), force_sample=force_fn(i)).state)
    return states


def test_valid_target_reaches_contact():
    arm = MockArm(); c = _controller(arm)
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    c.command("R3")
    states = _drive(c, lambda i: tgt, 20)
    assert State.CONTACT in states
    assert arm.servo_calls > 0
    assert states[-1] in (State.CONTACT,)


def test_no_extrinsics_never_moves():
    arm = MockArm(); c = _controller(arm)
    c.command("R3")
    _drive(c, lambda i: [P("R3", None)], 10)          # xyz_world None ⇒ no world frame
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_high_uncertainty_blocks_motion():
    arm = MockArm(); c = _controller(arm)
    c.command("R3")
    _drive(c, lambda i: [P("R3", np.array([0, 0, .1]), position_std=0.05)], 10)  # 50mm std > 10mm
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_unmeasured_target_never_silently_driven():
    arm = MockArm(); c = _controller(arm)
    c.command("R3")
    _drive(c, lambda i: [P("R3", np.array([0, 0, .1]), measured=False, status="COASTING")], 10)
    assert arm.servo_calls == 0


def test_occlusion_midapproach_halts_then_recovers():
    arm = MockArm(); c = _controller(arm); c.command("R3")
    good = [P("R3", np.array([0., 0., 0.10]))]
    occ = [P("R3", np.array([0., 0., 0.10]), measured=False, status="COASTING")]
    # approach a bit, then occlude → HOLD (no further servo), then recover → APPROACH
    for _ in range(2):
        c.tick(good, force_sample=_force())
    assert c.state.state == State.APPROACH
    s_before = arm.servo_calls
    c.tick(occ, force_sample=_force())
    assert c.state.state == State.HOLD
    c.tick(occ, force_sample=_force())
    assert arm.servo_calls == s_before                # NO motion while target degraded
    c.tick(good, force_sample=_force())
    assert c.state.state == State.APPROACH            # recovered → resume


def test_persistent_occlusion_retracts():
    arm = MockArm(); c = _controller(
        arm, hold_grace_frames=3, max_retract_distance_m=0.03,
    ); c.command("R3")
    good = [P("R3", np.array([0., 0., 0.10]))]
    occ = [P("R3", np.array([0., 0., 0.10]), measured=False, status="COASTING")]
    c.tick(good, force_sample=_force()); c.tick(good, force_sample=_force())
    approach_direction = np.array([0.0, 0.0, 1.0])
    states = _drive(c, lambda i: occ, 10)
    assert State.RETRACT in states                    # gave up safely after grace
    assert arm.retract_calls > 0
    assert np.dot(arm.retract_directions[-1], approach_direction) < 0
    assert arm.total_retract_distance_m <= c.cfg.max_retract_distance_m + 1e-9
    assert c.state.state == State.IDLE


def test_estop_latches_and_ignores_commands():
    arm = MockArm(); c = _controller(arm); c.command("R3")
    c.tick([P("R3", np.array([0, 0, .1]))], force_sample=_force())
    c.estop("test")
    assert c.state.state == State.ESTOP
    c.command("L2")                                   # ignored while latched
    s = c.tick([P("L2", np.array([0, 0, .1]))], force_sample=_force())
    assert s.state == State.ESTOP and arm.servo_calls == 1   # only the one pre-estop tick moved...
    # (servo happened on the first tick in APPROACH; none after estop)


def test_stale_perception_halts():
    """A frozen perception pipeline (timestamp stops advancing) must HALT, even though it keeps
    reporting the last 'measured' target — never act on stale data."""
    arm = MockArm(); c = _controller(arm, max_target_age_s=2.5); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    for i in range(3):                                  # live: perception_t advances with wall clock
        c.tick(tgt, perception_t=float(i), now=float(i), force_sample=_force())
    assert c.state.state == State.APPROACH
    sc = arm.servo_calls
    c.tick(tgt, perception_t=2.0, now=10.0, force_sample=_force())  # frozen perception
    assert c.state.state == State.HOLD and arm.servo_calls == sc   # halted, no motion on stale data
    for i in range(11, 14):                             # perception resumes
        c.tick(tgt, perception_t=float(i), now=float(i), force_sample=_force())
    assert c.state.state in (State.APPROACH, State.CONTACT)   # resumed motion


def test_overforce_triggers_estop():
    arm = MockArm(); c = _controller(arm); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    c.tick(tgt, force_sample=_force()); c.tick(tgt, force_sample=_force())
    holds = arm.hold_calls
    s = c.tick(tgt, force_sample=_force(5.5, "HARD_LIMIT"))  # calibrated ceiling ⇒ ESTOP
    assert s.state == State.ESTOP
    assert arm.hold_calls == holds + 1


def test_screened_force_retracts_without_latching_estop():
    arm = MockArm(); c = _controller(arm); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    c.tick(tgt, force_sample=_force()); c.tick(tgt, force_sample=_force())
    state = c.tick(tgt, force_sample=_force(4.6, "SCREENED_LIMIT"))
    assert state.state == State.RETRACT
    assert state.fault is None
    assert arm.retract_calls == 1


def test_persistent_screened_force_continues_bounded_retract_without_estop():
    arm = MockArm(); c = _controller(arm); c.command("R3")
    target = [P("R3", np.array([0.0, 0.0, 0.10]))]
    c.tick(target, force_sample=_force()); c.tick(target, force_sample=_force())

    first = c.tick(target, force_sample=_force(4.6, "SCREENED_LIMIT"))
    second = c.tick(target, force_sample=_force(4.6, "SCREENED_LIMIT"))

    assert first.state == State.RETRACT
    assert second.state in (State.RETRACT, State.IDLE)
    assert second.state != State.ESTOP
    assert arm.retract_calls >= 2


def test_hard_limit_status_estops_even_if_numeric_payload_is_inconsistent():
    arm = MockArm(); c = _controller(arm); c.command("R3")
    state = c.tick(
        [P("R3", np.array([0.0, 0.0, 0.10]))],
        force_sample=_force(0.0, "HARD_LIMIT"),
    )
    assert state.state == State.ESTOP
    assert arm.servo_calls == 0


def test_safety_config_copies_both_calibration_thresholds():
    cfg = _cfg()
    assert cfg.max_screened_force_n == pytest.approx(4.6)
    assert cfg.hard_limit_force_n == pytest.approx(5.5)


def test_controller_requires_calibration_derived_force_limit():
    with pytest.raises(ValueError, match="force calibration"):
        AuscultationController(MockArm())
    with pytest.raises(ValueError, match="calibrated=true"):
        SafetyConfig.from_force_calibration(
            SimpleNamespace(calibrated=False, max_screened_force_n=4.6, hard_limit_force_n=5.5),
        )


def test_retract_without_recorded_approach_direction_estops_and_holds():
    arm = MockArm(); c = _controller(arm)
    c.state = SafetyState(State.RETRACT, "R3")
    holds = arm.hold_calls
    state = c.tick([], force_sample=_force())
    assert state.state == State.ESTOP
    assert "approach direction" in state.fault
    assert arm.retract_calls == 0
    assert arm.hold_calls == holds + 1


@pytest.mark.parametrize(
    "sample",
    [
        None,
        _force(status="STALE"),
        _force(status="NO_DATA"),
        _force(age_s=0.201),
        _force(status="INVALID"),
        _force(valid=False),
        _force(force_n=float("nan")),
    ],
    ids=["missing", "stale", "no-data", "too-old", "bad-status", "invalid", "nan"],
)
def test_force_sample_must_be_explicit_fresh_valid_and_finite(sample):
    arm = MockArm(); c = _controller(arm); c.command("R3")
    holds = arm.hold_calls

    state = c.tick([P("R3", np.array([0.0, 0.0, 0.1]))], force_sample=sample)

    assert state.state == State.ESTOP
    assert "force sample" in state.fault
    assert arm.servo_calls == 0
    assert arm.hold_calls == holds + 1


def test_approach_target_view_exception_holds_and_estops():
    arm = MockArm(); c = _controller(arm)
    c.state = SafetyState(State.APPROACH, "R3")
    holds = arm.hold_calls

    state = c.tick([BrokenTargetView()], force_sample=_force())

    assert state.state == State.ESTOP
    assert "target view" in state.fault
    assert arm.servo_calls == 0
    assert arm.hold_calls == holds + 1


@pytest.mark.parametrize(
    ("fault", "fault_text"),
    [
        ("tcp-exception", "TCP"),
        ("servo-exception", "servo"),
        ("wrong-direction", "direction or step bound"),
        ("oversized-step", "direction or step bound"),
    ],
)
def test_approach_actuator_faults_and_motion_contract_violations_estop(fault, fault_text):
    arm = ServoFaultArm(fault); c = _controller(arm)
    c.state = SafetyState(State.APPROACH, "R3")
    holds = arm.hold_calls

    state = c.tick(
        [P("R3", np.array([0.0, 0.0, 0.1]))],
        force_sample=_force(),
    )

    assert state.state == State.ESTOP
    assert fault_text in state.fault
    assert arm.hold_calls == holds + 1


def test_retract_second_tcp_read_failure_latches_estop():
    arm = SecondTcpFaultArm(); c = _controller(arm)
    c.state = SafetyState(State.RETRACT, "R3")
    c._last_approach_direction = np.array([0.0, 0.0, 1.0])

    state = c.tick([], force_sample=_force())

    assert state.state == State.ESTOP
    assert "retract TCP" in state.fault


def test_hold_transport_failure_still_latches_estop_without_escaping():
    arm = HoldFaultArm(); c = _controller(arm)

    state = c.tick([], force_sample=_force())

    assert state.state == State.ESTOP
    assert "hold command failed" in state.fault


# ── June audit regressions (28a6f30, d267326) adapted to the force-sample contract ────────────

def test_nan_world_coord_never_moves():
    """A NaN target coordinate must be rejected exactly like a missing one."""
    arm = MockArm(); c = _controller(arm); c.command("R3")
    _drive(c, lambda i: [P("R3", np.array([0., 0., np.nan]))], 10)
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_inf_world_coord_never_moves():
    arm = MockArm(); c = _controller(arm); c.command("R3")
    _drive(c, lambda i: [P("R3", np.array([0., 0., np.inf]))], 10)
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_empty_detector_output_never_moves():
    arm = MockArm(); c = _controller(arm); c.command("R3")
    _drive(c, lambda i: [], 10)
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_commanded_landmark_absent_never_moves():
    arm = MockArm(); c = _controller(arm); c.command("R3")
    _drive(c, lambda i: [P("L2", np.array([0., 0., 0.10]))], 10)
    assert arm.servo_calls == 0 and c.state.state == State.ACQUIRING


def test_high_uncertainty_midapproach_halts_then_recovers():
    arm = MockArm(); c = _controller(arm); c.command("R3")
    good = [P("R3", np.array([0., 0., 0.10]))]
    hi = [P("R3", np.array([0., 0., 0.10]), position_std=0.05)]   # 50 mm > 10 mm, still measured
    c.tick(good, force_sample=_force()); c.tick(good, force_sample=_force())
    assert c.state.state == State.APPROACH
    sc = arm.servo_calls
    c.tick(hi, force_sample=_force()); assert c.state.state == State.HOLD
    c.tick(hi, force_sample=_force()); assert arm.servo_calls == sc
    c.tick(good, force_sample=_force()); assert c.state.state == State.APPROACH


def test_frozen_pipeline_no_timestamp_halts():
    """tick(points) with NO perception_t: identical points re-emitted by a frozen pipeline must still
    go stale -> HOLD (the staleness guard must not be inert by default)."""
    arm = MockArm(); c = _controller(arm, max_target_age_s=2.5); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    c.tick(tgt, now=0.0, force_sample=_force()); c.tick(tgt, now=0.1, force_sample=_force())
    assert c.state.state == State.APPROACH
    sc = arm.servo_calls
    c.tick(tgt, now=10.0, force_sample=_force())        # 10 s of unchanged content => stale
    assert c.state.state == State.HOLD and arm.servo_calls == sc


def test_frozen_pipeline_with_nan_sibling_still_halts():
    """A NaN coord on an unrelated (LOST) point must not defeat the content-fingerprint staleness
    guard (nan != nan would make a frozen frame compare unequal to itself)."""
    arm = MockArm(); c = _controller(arm, max_target_age_s=2.5); c.command("R3")
    def frame():
        return [P("R3", np.array([0., 0., 0.10])),
                P("L2", np.array([0., 0., np.nan]), measured=False, status="LOST")]
    c.tick(frame(), now=0.0, force_sample=_force()); c.tick(frame(), now=0.1, force_sample=_force())
    assert c.state.state == State.APPROACH
    sc = arm.servo_calls
    c.tick(frame(), now=10.0, force_sample=_force())
    assert c.state.state == State.HOLD and arm.servo_calls == sc


def test_nan_perception_t_halts():
    """A non-finite perception timestamp must never count as a fresh frame, not even the first."""
    arm = MockArm(); c = _controller(arm, max_target_age_s=2.5); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    for bad, now in ((float("nan"), 0.0), (float("inf"), 0.1), (float("nan"), 10.0)):
        c.tick(tgt, perception_t=bad, now=now, force_sample=_force())
        assert c.state.state == State.ACQUIRING and arm.servo_calls == 0


def test_controller_recovers_when_perception_t_is_finite_again():
    """One corrupt timestamp halts motion while it lasts, but must not poison every later
    comparison (nan < x is always False): finite, increasing timestamps resume the approach."""
    arm = MockArm(); c = _controller(arm, max_target_age_s=2.5); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    for i in range(3):
        c.tick(tgt, perception_t=float(i), now=float(i), force_sample=_force())
    assert c.state.state == State.APPROACH
    c.tick(tgt, perception_t=float("nan"), now=5.0, force_sample=_force())
    assert c.state.state == State.HOLD
    sc = arm.servo_calls
    c.tick(tgt, perception_t=6.0, now=6.0, force_sample=_force())
    assert c.state.state == State.APPROACH and arm.servo_calls == sc + 1


def test_reset_starts_a_fresh_perception_history():
    """After reset() a restarted tracker clock (smaller timestamps) counts as fresh at once."""
    arm = MockArm(); c = _controller(arm, max_target_age_s=2.5); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    c.tick(tgt, perception_t=100.0, now=0.0, force_sample=_force())
    c.reset(); c.command("R3")
    c.tick(tgt, perception_t=1.0, now=5.0, force_sample=_force())
    assert c.state.state == State.APPROACH


def test_overforce_during_stale_latches_estop():
    """Hard-limit force wins over staleness: freeze + over-force in one tick => latched ESTOP."""
    arm = MockArm(); c = _controller(arm, max_target_age_s=2.5); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    for i in range(3):
        c.tick(tgt, perception_t=float(i), now=float(i), force_sample=_force())
    assert c.state.state == State.APPROACH
    s = c.tick(tgt, perception_t=2.0, now=10.0, force_sample=_force(5.5, "HARD_LIMIT"))
    assert s.state == State.ESTOP
    s2 = c.tick(tgt, perception_t=11.0, now=11.0, force_sample=_force())
    assert s2.state == State.ESTOP


def test_contact_degradation_holds_not_retracts():
    """At CONTACT the probe occludes its own landmark (measured=False is expected): stay in CONTACT
    with the arm positively held, no servo, no auto-retract."""
    arm = MockArm(); c = _controller(arm); c.command("R3")
    near = np.array([0., 0., 0.005])                    # within reach_tol (8 mm) after one step
    c.tick([P("R3", near)], force_sample=_force()); c.tick([P("R3", near)], force_sample=_force())
    assert c.state.state == State.CONTACT
    sc, holds = arm.servo_calls, arm.hold_calls
    c.tick([P("R3", near, measured=False, status="COASTING")], force_sample=_force())
    assert c.state.state == State.CONTACT and arm.servo_calls == sc and arm.hold_calls == holds + 1


def test_every_non_servo_tick_positively_holds():
    """Exhaustive dispatch: a tick that cannot servo must still command hold(), never leave the arm
    on its last motion command. Reachable with require_world_frame=False and no extrinsics."""
    arm = MockArm(); c = _controller(arm, require_world_frame=False); c.command("R3")
    holds = arm.hold_calls
    for _ in range(3):
        c.tick([P("R3", None)], force_sample=_force())
    assert c.state.state == State.APPROACH and arm.servo_calls == 0
    assert arm.hold_calls == holds + 3


def _flicker_views():
    from ausc.perception.robot.safety import TargetView
    p = np.array([0., 0., 0.10])
    return (TargetView("R3", p, measured=True, position_std_m=0.003, status="TRACKING"),
            TargetView("R3", p, measured=False, position_std_m=0.003, status="COASTING"))


def test_flickering_occlusion_eventually_retracts():
    """A target that recovers for one frame per window must not defer RETRACT forever (leaky
    degraded counter). State machine only: dist=1.0 => no CONTACT."""
    from ausc.perception.robot.safety import command, step
    cfg = _cfg(hold_grace_frames=4); good, occ = _flicker_views()
    s = command(SafetyState(), "R3")
    s = step(s, good, cfg, dist_to_target_m=1.0); s = step(s, good, cfg, dist_to_target_m=1.0)
    assert s.state == State.APPROACH
    states = []
    for _ in range(20):
        s = step(s, occ, cfg, dist_to_target_m=1.0); states.append(s.state)
        s = step(s, occ, cfg, dist_to_target_m=1.0); states.append(s.state)
        s = step(s, good, cfg, dist_to_target_m=1.0); states.append(s.state)   # 1 good frame per 3
    assert State.RETRACT in states


def test_one_to_one_flicker_does_not_retract():
    """Documented boundary: an exact 1:1 occlusion/recovery alternation deliberately does not time
    out (each good frame is genuinely measured, so servoing on it is safe)."""
    from ausc.perception.robot.safety import command, step
    cfg = _cfg(hold_grace_frames=4); good, occ = _flicker_views()
    s = command(SafetyState(), "R3")
    s = step(s, good, cfg, dist_to_target_m=1.0); s = step(s, good, cfg, dist_to_target_m=1.0)
    states = set()
    for _ in range(200):
        s = step(s, occ, cfg, dist_to_target_m=1.0); states.add(s.state)
        s = step(s, good, cfg, dist_to_target_m=1.0); states.add(s.state)
    assert State.RETRACT not in states and states <= {State.APPROACH, State.HOLD}


class _LegacyPoint:
    """A legacy/foreign tracker point lacking the ``measured`` field the safety gate keys on."""
    def __init__(self, name, xyz_world):
        self.name = name; self.xyz_world = xyz_world
        self.position_std = 0.003; self.status = "TRACKING"   # NO `measured` attribute


def test_legacy_point_without_measured_never_moves():
    """June expected a fail-safe ACQUIRING here; the current controller latches ESTOP on any
    unreadable target view (see test_approach_target_view_exception_holds_and_estops). No motion."""
    arm = MockArm(); c = _controller(arm); c.command("R3")
    for _ in range(6):
        c.tick([_LegacyPoint("R3", np.array([0., 0., 0.10]))], force_sample=_force())
    assert arm.servo_calls == 0 and c.state.state == State.ESTOP


class RetractFaultArm(MockArm):
    def retract(self, direction_base, max_step_m):
        self.retract_calls += 1
        raise RuntimeError("retract servo failure")


def test_retract_servo_failure_estops():
    """A fault during RETRACT latches ESTOP (symmetric with APPROACH), never strands the machine."""
    arm = RetractFaultArm(); c = _controller(arm, hold_grace_frames=2); c.command("R3")
    near = np.array([0., 0., 0.12])
    good = [P("R3", near)]; occ = [P("R3", near, measured=False, status="COASTING")]
    for _ in range(3):
        c.tick(good, force_sample=_force())
    states = [c.tick(occ, force_sample=_force()).state for _ in range(12)]
    assert State.ESTOP in states and arm.retract_calls >= 1
    assert "retract command failed" in c.state.fault


# ── kinematics / RealArm hardening (28a6f30, d267326); fake yourdfpy, no hardware ─────────────

def _fake_kinematics(monkeypatch, gains=None, limit=(-2.0, 2.0), velocity=3.0):
    """ArmKinematics over a stand-in URDF with linear FK p = gains @ q (analytically checkable)."""
    from ausc.perception.robot.kinematics import ArmKinematics
    A = np.eye(3, 7) * 0.5 if gains is None else np.asarray(gains, float)
    lim = None if limit is None else SimpleNamespace(lower=limit[0], upper=limit[1], velocity=velocity)
    names = [f"openarm_left_joint{i}" for i in range(1, 8)]

    class Robot:
        joint_map = {n: SimpleNamespace(type="revolute", limit=lim) for n in names}
        q = np.zeros(7)

        def update_cfg(self, cfg):
            self.q = np.array([cfg[n] for n in names], float)

        def get_transform(self, frame_to, frame_from):
            T = np.eye(4); T[:3, 3] = A @ self.q
            return T

    monkeypatch.setitem(sys.modules, "yourdfpy",
                        SimpleNamespace(URDF=SimpleNamespace(load=lambda path, **kw: Robot())))
    return ArmKinematics("fake.urdf", "left")


@pytest.mark.parametrize("q, target", [
    (np.zeros(7), np.array([np.nan, 0.0, 0.0])),
    (np.zeros(7), np.array([0.0, np.inf, 0.0])),
    (np.full(7, np.nan), np.array([0.0, 0.0, 0.1])),
], ids=["nan-target", "inf-target", "nan-joints"])
def test_ik_step_refuses_non_finite_input(monkeypatch, q, target):
    kin = _fake_kinematics(monkeypatch)
    with pytest.raises(ValueError, match="non-finite"):
        kin.ik_step(q, target)


def test_ik_step_slew_bounded_by_urdf_velocity_at_30hz(monkeypatch):
    kin = _fake_kinematics(monkeypatch, gains=np.eye(3, 7) * 0.05, velocity=0.3)
    q0 = np.zeros(7); dx = np.array([0.02, -0.01, 0.005])
    dq = kin.ik_step(q0, dx) - q0                       # unbounded DLS would slew ~0.38 rad here
    assert np.all(np.abs(dq) <= 0.3 / 30.0 + 1e-12)     # default dt matches the ~30 Hz loop
    moved = kin.tcp_pos(q0 + dq) - kin.tcp_pos(q0)
    assert np.allclose(moved / np.linalg.norm(moved), dx / np.linalg.norm(dx))   # direction kept


def test_ik_step_damps_near_singular_direction(monkeypatch):
    gains = np.zeros((3, 7)); gains[0, 0] = gains[1, 1] = 0.5; gains[2, 2] = 0.005   # weak z axis
    kin = _fake_kinematics(monkeypatch, gains=gains, velocity=1e4)   # slew cap out of the way
    dq = kin.ik_step(np.zeros(7), np.array([0.0, 0.0, 0.02]))
    assert abs(dq[2]) < 0.1                             # fixed lam=0.01 would whip it ~0.8 rad


@pytest.mark.parametrize("limit", [None, (0.0, 0.0)], ids=["no-limit", "zero-range"])
def test_kinematics_refuses_to_guess_joint_limits(monkeypatch, limit):
    with pytest.raises(ValueError, match="no usable position limits"):
        _fake_kinematics(monkeypatch, limit=limit)


def test_vendored_urdf_has_usable_limits_for_every_arm_joint():
    """The fail-loud limit check must not reject the URDF we actually ship."""
    import xml.etree.ElementTree as ET
    urdf = os.path.join(os.path.dirname(__file__), "..", "openarm_v10.urdf")
    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    for arm in ("left", "right"):
        for i in range(1, 8):
            j = joints[f"openarm_{arm}_joint{i}"]
            lim = j.find("limit")
            assert j.get("type") == "revolute" and lim is not None
            lo, hi, vel = (float(lim.get(k)) for k in ("lower", "upper", "velocity"))
            assert lo < hi and vel > 0


def test_real_arm_refuses_non_finite_joint_command(monkeypatch):
    from ausc.perception.robot.arm_interface import RealArm
    sent = []
    monkeypatch.setitem(sys.modules, "openarm_can",
                        SimpleNamespace(PosVelParam=lambda pos, vel: (pos, vel)))
    motors = [SimpleNamespace(get_position=lambda: 0.0) for _ in range(7)]
    bus = SimpleNamespace(get_motors=lambda: motors,
                          posvel_control_all=lambda params: sent.append(list(params)))
    real = RealArm.__new__(RealArm)
    real.kin = SimpleNamespace(ik_step=lambda q, t, m: np.full(7, np.nan), tcp_pos=lambda q: np.zeros(3))
    real._arm = SimpleNamespace(refresh_all=lambda: None, recv_all=lambda: None, get_arm=lambda: bus)
    real._enabled = True
    with pytest.raises(RuntimeError, match="non-finite"):
        real.servo_toward(np.array([0.0, 0.0, 0.1]), 0.02)
    assert sent and all(np.isfinite(pos) for params in sent for pos, _ in params)
    assert all(vel == 0.0 for _, vel in sent[-1])       # last command is a positive hold


def test_impossible_force_floor_comes_from_the_calibration_curve():
    """One full calibrated force range below the rest preload is unreachable: the carriage sits on
    its preloaded rest stop. As built: 1.5 N - (8.75 - 1.5) N = -5.75 N."""
    import sys
    from pathlib import Path
    ff = Path(__file__).resolve().parents[4] / "mechanical_sensor" / "force_feedback"
    sys.path.insert(0, str(ff))
    import force_sensor as fs
    tared = fs.ForceCalibration.from_json(ff / "force_feedback_calibration_veo6_tared.json")
    cfg = SafetyConfig.from_force_calibration(tared)
    assert cfg.min_plausible_force_n == pytest.approx(1.5 - (8.75 - 1.5))
    assert tared.sample_from_raw(65535, 0.0).force_n < cfg.min_plausible_force_n   # stuck high
    assert _cfg().min_plausible_force_n == pytest.approx(-5.5)   # no curve: minus the hard limit
    with pytest.raises(ValueError, match="derived from calibration"):
        _cfg(min_plausible_force_n=-100.0)


def test_physically_impossible_force_latches_estop():
    """A stuck-high Hall sensor reads about -14.7 N with status WARN, which would otherwise look
    like "no contact" while the arm keeps approaching."""
    arm = MockArm(); c = _controller(arm); c.command("R3")
    tgt = [P("R3", np.array([0., 0., 0.10]))]
    c.tick(tgt, perception_t=0.0, now=0.0, force_sample=_force())
    assert c.state.state == State.APPROACH
    sc = arm.servo_calls
    s = c.tick(tgt, perception_t=0.033, now=0.033, force_sample=_force(-14.7, "WARN"))
    assert s.state == State.ESTOP and "physically impossible" in s.fault
    assert arm.servo_calls == sc


@pytest.mark.parametrize("floor", [None, float("nan"), 0.5])
def test_controller_refuses_a_config_without_a_usable_force_floor(floor):
    """A hand-built SafetyConfig must not silently skip the stuck-sensor check (a missing or NaN
    floor would compare False and fail open; a positive one could trip on a light touch)."""
    import dataclasses
    cfg = dataclasses.replace(_cfg(), min_plausible_force_n=floor)
    with pytest.raises(ValueError, match="force calibration"):
        AuscultationController(MockArm(), cfg)
