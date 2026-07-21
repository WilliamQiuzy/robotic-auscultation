"""Closed-loop arm control for auscultation-point stethoscope placement.

Perception (ModelBasedTracker) → ``AuscultationController`` (+ ``SafetyConfig`` state machine) →
``ArmInterface`` (SimArm kinematic / RealArm OpenArm). The controller acts ONLY on points the
tracker reports ``measured`` with low ``position_std`` in a valid world frame, and HALTS when a
target degrades (e.g. the arm occludes it). Safety/controller logic is hardware-free testable;
kinematics + RealArm are box-only.
"""
from .controller import ArmInterface, AuscultationController
from .safety import SafetyConfig, SafetyState, State, TargetView, command, estop, reset, step

__all__ = ["AuscultationController", "ArmInterface", "SafetyConfig", "SafetyState", "State",
           "TargetView", "command", "estop", "reset", "step"]
