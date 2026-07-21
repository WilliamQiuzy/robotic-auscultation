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
