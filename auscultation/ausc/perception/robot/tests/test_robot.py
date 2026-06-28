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
