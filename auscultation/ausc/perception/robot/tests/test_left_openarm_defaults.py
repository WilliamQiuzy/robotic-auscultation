from __future__ import annotations

import inspect

from ausc.perception.robot.arm_interface import RealArm, SimArm
from ausc.perception.robot.kinematics import ArmKinematics


def test_openarm_runtime_defaults_are_the_left_can1_to_can3_system() -> None:
    sim = inspect.signature(SimArm).parameters
    real = inspect.signature(RealArm).parameters
    kinematics = inspect.signature(ArmKinematics).parameters

    assert sim["arm"].default == "left"
    assert real["arm"].default == "left"
    assert real["can"].default == "can3"
    assert kinematics["arm"].default == "left"
