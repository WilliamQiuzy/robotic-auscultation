"""Pure contracts for the official-asset OpenArm occlusion experiment.

This module deliberately contains no Isaac imports and cannot open a camera or command a
robot.  The Lingbox runner supplies rendered RGB, depth, and instance masks; the functions here
validate the official scene inventory, transfer the WAM ten-point identity to a frozen human
mesh, and measure visibility from those rendered products.  No capsule or visible-count rule is
part of this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import PurePosixPath
import re
from typing import Mapping, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np


OFFICIAL_CLEAN_URDF_SHA256 = (
    "63caf1116a844c7eca25e91f0acf025c2798cc4795cf727b23433eacde80ca12"
)
NVIDIA_HUMAN_SHA256 = (
    "2c65e15587a311c5d14cdbd6a1408e0c3d8f3bec69f483150f04ca4c8f4e3ccc"
)
POINT_NAMES = ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5")
LEFT_FOLLOWER_JOINTS = tuple(f"openarm_left_joint{index}" for index in range(1, 8))
RIGHT_REFERENCE_JOINTS = tuple(f"openarm_right_joint{index}" for index in range(1, 8))
ORBBEC_RGB_INTRINSICS = {
    "width": 1920,
    "height": 1080,
    "fx": 1123.173095703125,
    "fy": 1122.9962158203125,
    "cx": 955.7200927734375,
    "cy": 533.1757202148438,
}

_PRIMITIVES = {"box", "capsule", "cylinder", "plane", "sphere"}
_PACKAGE_PREFIX = "package://openarm_description/"
_SAFE_SCENARIO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class ExperimentContractError(ValueError):
    """The proposed official-asset experiment is incomplete or substituted."""


@dataclass(frozen=True)
class OpenArmUrdfInventory:
    robot_name: str
    link_count: int
    joint_count: int
    revolute_joint_names: Tuple[str, ...]
    left_follower_joints: Tuple[str, ...]
    right_fixed_joints: Tuple[str, ...]
    mesh_reference_count: int
    unique_mesh_count: int
    has_primitive_geometry: bool


@dataclass(frozen=True)
class MeshAnchor:
    name: str
    triangle_id: int
    barycentric: np.ndarray
    position_m: np.ndarray
    surface_normal: np.ndarray


@dataclass(frozen=True)
class RenderedVisibility:
    name: str
    visible_fraction: float
    robot_front_fraction: float
    valid_depth_fraction: float
    mask_edge_distance_px: float
    visibility_source: str = "rendered_instance_mask_and_depth"


@dataclass(frozen=True)
class PositionIkCandidate:
    """One real-chain position solution with its observed wrist attitude."""

    joint_positions: np.ndarray
    tip_rotation: np.ndarray
    position_error_m: float
    orientation_separation_from_first_rad: float


@dataclass(frozen=True)
class JointTrajectory:
    """Minimum-jerk joint samples with a provably fixed articulation root."""

    joint_positions: np.ndarray
    time_s: np.ndarray
    root_poses: np.ndarray


@dataclass(frozen=True)
class OcclusionPolicyDecision:
    """Experimental handoff state; never a production robot command gate."""

    state: str
    predicted_error_p95_m: float
    reason: str
    visible_point_count_rule: bool = False
    scope: str = "parameter_sensitivity_only"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def inspect_openarm_urdf(raw: bytes) -> OpenArmUrdfInventory:
    """Validate the exact full-v1 topology before Isaac is allowed to import it."""

    if type(raw) is not bytes:
        raise ExperimentContractError("URDF must be supplied as bytes")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ExperimentContractError("OpenArm URDF is not well-formed XML") from exc
    if _local_name(root.tag) != "robot" or root.get("name") != "openarm":
        raise ExperimentContractError("robot name must be the official openarm assembly")

    links = [item for item in root if _local_name(item.tag) == "link"]
    joints = [item for item in root if _local_name(item.tag) == "joint"]
    primitives = [
        item for item in root.iter() if _local_name(item.tag) in _PRIMITIVES
    ]
    meshes = [item for item in root.iter() if _local_name(item.tag) == "mesh"]
    mesh_paths = []
    for mesh in meshes:
        filename = mesh.get("filename")
        if not isinstance(filename, str) or not filename.startswith(_PACKAGE_PREFIX):
            raise ExperimentContractError(
                "every mesh must resolve from the official openarm_description package"
            )
        relative = filename[len(_PACKAGE_PREFIX) :]
        path = PurePosixPath(relative)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ExperimentContractError("OpenArm mesh path is unsafe")
        scale = mesh.get("scale")
        if not isinstance(scale, str) or len(scale.split()) != 3:
            raise ExperimentContractError("every OpenArm mesh must keep its explicit scale")
        try:
            values = tuple(float(value) for value in scale.split())
        except ValueError as exc:
            raise ExperimentContractError("OpenArm mesh scale is invalid") from exc
        if not all(math.isfinite(value) and value != 0.0 for value in values):
            raise ExperimentContractError("OpenArm mesh scale is invalid")
        mesh_paths.append(relative)

    revolute = tuple(
        joint.get("name", "") for joint in joints if joint.get("type") == "revolute"
    )
    failures = []
    if primitives:
        failures.append("primitive geometry is forbidden")
    if len(links) != 24:
        failures.append("official v1 dual arm must contain exactly 24 links")
    if len(joints) != 23:
        failures.append("official v1 dual arm must contain exactly 23 joints")
    if revolute != LEFT_FOLLOWER_JOINTS + RIGHT_REFERENCE_JOINTS:
        failures.append("official v1 dual arm must contain both exact seven-joint chains")
    if len(mesh_paths) != 42 or len(set(mesh_paths)) != 20:
        failures.append("official v1 dual arm must retain all 42 mesh references/20 files")
    if failures:
        raise ExperimentContractError("; ".join(failures))
    return OpenArmUrdfInventory(
        robot_name="openarm",
        link_count=len(links),
        joint_count=len(joints),
        revolute_joint_names=revolute,
        left_follower_joints=LEFT_FOLLOWER_JOINTS,
        right_fixed_joints=RIGHT_REFERENCE_JOINTS,
        mesh_reference_count=len(mesh_paths),
        unique_mesh_count=len(set(mesh_paths)),
        has_primitive_geometry=False,
    )


def _vector_attribute(element: ET.Element, name: str, default: str) -> np.ndarray:
    try:
        result = np.asarray(
            [float(value) for value in element.get(name, default).split()],
            dtype=np.float64,
        )
    except (AttributeError, ValueError) as exc:
        raise ExperimentContractError(f"URDF {name} vector is invalid") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ExperimentContractError(f"URDF {name} vector is invalid")
    return result


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise ExperimentContractError("URDF revolute axis must be nonzero")
    x, y, z = axis / norm
    cosine, sine, complement = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.asarray(
        [
            [cosine + x * x * complement, x * y * complement - z * sine, x * z * complement + y * sine],
            [y * x * complement + z * sine, cosine + y * y * complement, y * z * complement - x * sine],
            [z * x * complement - y * sine, z * y * complement + x * sine, cosine + z * z * complement],
        ],
        dtype=np.float64,
    )


def _origin_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    xyz = np.zeros(3) if origin is None else _vector_attribute(origin, "xyz", "0 0 0")
    rpy = np.zeros(3) if origin is None else _vector_attribute(origin, "rpy", "0 0 0")
    transform = np.eye(4)
    transform[:3, :3] = _rotation_from_rpy(rpy)
    transform[:3, 3] = xyz
    return transform


@lru_cache(maxsize=8)
def _left_chain(raw: bytes) -> Tuple[Tuple[ET.Element, ...], Tuple[ET.Element, ...]]:
    inspect_openarm_urdf(raw)
    root = ET.fromstring(raw)
    by_child = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None or not child.get("link"):
            raise ExperimentContractError("URDF joint is missing its child link")
        by_child[child.get("link")] = joint
    child_chain = (
        "openarm_body_link0",
        "openarm_left_link0",
        *(f"openarm_left_link{index}" for index in range(1, 8)),
    )
    try:
        all_joints = tuple(by_child[child] for child in child_chain)
    except KeyError as exc:
        raise ExperimentContractError("official left OpenArm chain is incomplete") from exc
    moving = tuple(joint for joint in all_joints if joint.get("type") == "revolute")
    if tuple(joint.get("name") for joint in moving) != LEFT_FOLLOWER_JOINTS:
        raise ExperimentContractError("official left OpenArm joint order is invalid")
    return all_joints, moving


def left_joint_limits(raw: bytes) -> np.ndarray:
    """Return the official left-follower position limits in joint order."""

    _, moving = _left_chain(raw)
    rows = []
    for joint in moving:
        limit = joint.find("limit")
        if limit is None:
            raise ExperimentContractError("official revolute joint is missing limits")
        try:
            rows.append((float(limit.get("lower")), float(limit.get("upper"))))
        except (TypeError, ValueError) as exc:
            raise ExperimentContractError("official revolute joint limits are invalid") from exc
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (7, 2) or not np.isfinite(result).all() or np.any(
        result[:, 0] >= result[:, 1]
    ):
        raise ExperimentContractError("official left joint-limit matrix is invalid")
    return result


def left_joint_velocity_limits(raw: bytes) -> np.ndarray:
    """Return the official left-follower velocity limits in joint order."""

    _, moving = _left_chain(raw)
    values = []
    for joint in moving:
        limit = joint.find("limit")
        try:
            values.append(float(limit.get("velocity")))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ExperimentContractError("official revolute velocity limit is invalid") from exc
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (7,) or not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ExperimentContractError("official left velocity-limit vector is invalid")
    return result


def _left_fk_and_jacobian(raw: bytes, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    all_joints, _ = _left_chain(raw)
    positions = np.asarray(joint_positions, dtype=np.float64)
    if positions.shape != (7,) or not np.isfinite(positions).all():
        raise ExperimentContractError("left joint positions must be a finite seven-vector")
    limits = left_joint_limits(raw)
    if np.any(positions < limits[:, 0]) or np.any(positions > limits[:, 1]):
        raise ExperimentContractError("left joint positions exceed an official joint limit")
    transform = np.eye(4)
    axes_world = []
    origins_world = []
    position_index = 0
    for joint in all_joints:
        transform = transform @ _origin_transform(joint)
        if joint.get("type") == "revolute":
            axis_element = joint.find("axis")
            if axis_element is None:
                raise ExperimentContractError("official revolute joint is missing its axis")
            axis = _vector_attribute(axis_element, "xyz", "0 0 0")
            axes_world.append(transform[:3, :3] @ axis)
            origins_world.append(transform[:3, 3].copy())
            rotation = np.eye(4)
            rotation[:3, :3] = _axis_rotation(axis, float(positions[position_index]))
            transform = transform @ rotation
            position_index += 1
    tip = transform[:3, 3]
    jacobian = np.column_stack(
        [np.cross(axis, tip - origin) for axis, origin in zip(axes_world, origins_world)]
    )
    return transform, jacobian


def left_tip_forward_kinematics(raw: bytes, joint_positions: np.ndarray) -> np.ndarray:
    """Return the official left-link7 frame at unit URDF scale."""

    return _left_fk_and_jacobian(raw, joint_positions)[0]


def solve_left_position_ik(
    raw: bytes,
    target_position_m: np.ndarray,
    *,
    initial_q: np.ndarray,
    tolerance_m: float,
    max_iterations: int,
) -> np.ndarray:
    """Solve deterministic damped-least-squares position IK on the real v1 chain."""

    target = np.asarray(target_position_m, dtype=np.float64)
    positions = np.asarray(initial_q, dtype=np.float64).copy()
    limits = left_joint_limits(raw)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ExperimentContractError("IK target must be a finite XYZ position")
    if positions.shape != (7,) or not np.isfinite(positions).all():
        raise ExperimentContractError("IK initial_q must be a finite seven-vector")
    if not math.isfinite(tolerance_m) or tolerance_m <= 0.0:
        raise ExperimentContractError("IK tolerance must be finite and positive")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ExperimentContractError("IK max_iterations must be a positive integer")
    margin = 1.0e-8
    positions = np.clip(positions, limits[:, 0] + margin, limits[:, 1] - margin)
    damping = 2.0e-4
    for _ in range(max_iterations):
        transform, jacobian = _left_fk_and_jacobian(raw, positions)
        error = target - transform[:3, 3]
        if float(np.linalg.norm(error)) <= tolerance_m:
            return positions
        normal = jacobian @ jacobian.T + damping * np.eye(3)
        step = jacobian.T @ np.linalg.solve(normal, error)
        norm = float(np.linalg.norm(step))
        if norm > 0.20:
            step *= 0.20 / norm
        # Deterministic backtracking prevents a near-limit update from increasing target error.
        baseline = float(np.linalg.norm(error))
        accepted = False
        for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = np.clip(
                positions + factor * step,
                limits[:, 0] + margin,
                limits[:, 1] - margin,
            )
            candidate_error = float(
                np.linalg.norm(target - _left_fk_and_jacobian(raw, candidate)[0][:3, 3])
            )
            if candidate_error < baseline:
                positions = candidate
                accepted = True
                break
        if not accepted:
            damping = min(damping * 10.0, 1.0)
    final_error = float(
        np.linalg.norm(target - _left_fk_and_jacobian(raw, positions)[0][:3, 3])
    )
    raise ExperimentContractError(
        f"official OpenArm IK did not converge: final_error_m={final_error:.9f}"
    )


def _rotation_separation(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def solve_left_position_ik_candidates(
    raw: bytes,
    target_position_m: np.ndarray,
    *,
    initial_seeds: np.ndarray,
    tolerance_m: float,
    max_iterations: int,
    max_solutions: int,
    min_orientation_separation_rad: float,
) -> Tuple[PositionIkCandidate, ...]:
    """Find deterministic real-chain IK branches with distinct rendered wrist attitudes.

    Position-only IK is intentional here: the experiment samples occluding arm silhouettes at
    the same target from distinct reachable wrist attitudes.  The actual resulting rotations are
    measured and persisted; they are not asserted from a hand-authored orientation target.
    """

    seeds = np.asarray(initial_seeds, dtype=np.float64)
    target = np.asarray(target_position_m, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 7 or not np.isfinite(seeds).all():
        raise ExperimentContractError("IK seeds must be a finite array with shape (N, 7)")
    if len(seeds) == 0:
        raise ExperimentContractError("at least one IK seed is required")
    if type(max_solutions) is not int or max_solutions <= 0:
        raise ExperimentContractError("max_solutions must be a positive integer")
    if (
        not math.isfinite(min_orientation_separation_rad)
        or min_orientation_separation_rad < 0.0
        or min_orientation_separation_rad > math.pi
    ):
        raise ExperimentContractError("orientation separation must be within [0, pi]")

    solved = []
    for seed_index, seed in enumerate(seeds):
        try:
            joints = solve_left_position_ik(
                raw,
                target,
                initial_q=seed,
                tolerance_m=tolerance_m,
                max_iterations=max_iterations,
            )
        except ExperimentContractError:
            continue
        transform = left_tip_forward_kinematics(raw, joints)
        error = float(np.linalg.norm(transform[:3, 3] - target))
        if error > tolerance_m:
            continue
        if any(np.linalg.norm(joints - prior[1]) <= 1.0e-5 for prior in solved):
            continue
        solved.append((seed_index, joints, transform[:3, :3].copy(), error))

    if not solved:
        return ()
    # Preserve seed order so a frozen seed table gives byte-stable branch selection.
    first = solved[0]
    selected = [
        PositionIkCandidate(
            joint_positions=first[1].copy(),
            tip_rotation=first[2].copy(),
            position_error_m=first[3],
            orientation_separation_from_first_rad=0.0,
        )
    ]
    for _seed_index, joints, rotation, error in solved[1:]:
        separation = _rotation_separation(first[2], rotation)
        if separation + 1.0e-12 < min_orientation_separation_rad:
            continue
        selected.append(
            PositionIkCandidate(
                joint_positions=joints.copy(),
                tip_rotation=rotation.copy(),
                position_error_m=error,
                orientation_separation_from_first_rad=separation,
            )
        )
        if len(selected) >= max_solutions:
            break
    return tuple(selected[:max_solutions])


def quintic_blend(unit_time: np.ndarray) -> np.ndarray:
    """Minimum-jerk 0-to-1 interpolation used between frozen joint waypoints."""

    value = np.asarray(unit_time, dtype=np.float64)
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ExperimentContractError("quintic blend input must remain in [0, 1]")
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def build_minimum_jerk_trajectory(
    joint_waypoints: np.ndarray,
    *,
    frames_per_segment: int,
    seconds_per_segment: float,
) -> JointTrajectory:
    """Interpolate frozen seven-joint waypoints without moving the robot root."""

    waypoints = np.asarray(joint_waypoints, dtype=np.float64)
    if (
        waypoints.ndim != 2
        or waypoints.shape[1] != 7
        or len(waypoints) < 2
        or not np.isfinite(waypoints).all()
    ):
        raise ExperimentContractError("joint waypoints must be finite with shape (N>=2, 7)")
    if type(frames_per_segment) is not int or frames_per_segment < 3:
        raise ExperimentContractError("frames_per_segment must be an integer >= 3")
    if not math.isfinite(seconds_per_segment) or seconds_per_segment <= 0.0:
        raise ExperimentContractError("seconds_per_segment must be finite and positive")

    unit = np.linspace(0.0, 1.0, frames_per_segment)
    blend = quintic_blend(unit)[:, None]
    position_parts = []
    time_parts = []
    for index, (start, end) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        positions = start + blend * (end - start)
        times = index * seconds_per_segment + unit * seconds_per_segment
        if index:
            positions = positions[1:]
            times = times[1:]
        position_parts.append(positions)
        time_parts.append(times)
    result_positions = np.vstack(position_parts)
    result_times = np.concatenate(time_parts)
    roots = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(result_positions), axis=0)
    return JointTrajectory(result_positions, result_times, roots)


def decide_occlusion_policy(
    *,
    geometry_accepted: bool,
    last_valid_age_s: float,
    last_target_error_p95_m: float,
    patient_motion_bound_m_s: float,
    max_freeze_age_s: float,
    max_freeze_error_p95_m: float,
) -> OcclusionPolicyDecision:
    """Choose direct update, bounded patient-frame freeze, or fail-closed LOST.

    The bound is deliberately simple and auditable: during a freeze, last-valid p95 grows by
    ``patient_motion_bound_m_s * age``.  Point count is not an input and cannot authorize a hold.
    """

    if type(geometry_accepted) is not bool:
        raise ExperimentContractError("geometry_accepted must be a bool")
    values = np.asarray(
        [
            last_valid_age_s,
            last_target_error_p95_m,
            patient_motion_bound_m_s,
            max_freeze_age_s,
            max_freeze_error_p95_m,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ExperimentContractError("occlusion policy bounds must be finite and nonnegative")
    if max_freeze_age_s <= 0.0 or max_freeze_error_p95_m <= 0.0:
        raise ExperimentContractError("occlusion policy maxima must be positive")
    if geometry_accepted:
        return OcclusionPolicyDecision(
            state="DIRECT_GEOMETRY_UPDATE",
            predicted_error_p95_m=float(last_target_error_p95_m),
            reason="current_rendered_support_passes_declared_geometry_surface",
        )
    predicted = float(last_target_error_p95_m + patient_motion_bound_m_s * last_valid_age_s)
    if last_valid_age_s <= max_freeze_age_s and predicted <= max_freeze_error_p95_m:
        return OcclusionPolicyDecision(
            state="BOUNDED_PATIENT_FRAME_FREEZE",
            predicted_error_p95_m=predicted,
            reason="short_hold_with_declared_motion_uncertainty_bound",
        )
    reasons = []
    if last_valid_age_s > max_freeze_age_s:
        reasons.append("freeze_age_exceeded")
    if predicted > max_freeze_error_p95_m:
        reasons.append("freeze_uncertainty_exceeded")
    return OcclusionPolicyDecision(
        state="LOST",
        predicted_error_p95_m=predicted,
        reason="+".join(reasons) or "geometry_unavailable",
    )


def validate_left_joint_trajectory(
    joint_positions: np.ndarray,
    time_s: np.ndarray,
    root_poses: np.ndarray,
    *,
    joint_limits: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
) -> dict:
    """Validate real-joint motion and prove that root motion was not used as a proxy."""

    positions = np.asarray(joint_positions, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    roots = np.asarray(root_poses, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    velocity_limit = np.asarray(velocity_limits, dtype=np.float64)
    acceleration_limit = np.asarray(acceleration_limits, dtype=np.float64)
    frame_count = len(positions)
    if (
        positions.ndim != 2
        or positions.shape[1] != 7
        or frame_count < 3
        or times.shape != (frame_count,)
        or roots.shape != (frame_count, 4, 4)
        or limits.shape != (7, 2)
        or velocity_limit.shape != (7,)
        or acceleration_limit.shape != (7,)
    ):
        raise ExperimentContractError("joint trajectory arrays have invalid shapes")
    if not all(
        np.isfinite(value).all()
        for value in (positions, times, roots, limits, velocity_limit, acceleration_limit)
    ):
        raise ExperimentContractError("joint trajectory arrays must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ExperimentContractError("trajectory timestamps must be strictly increasing")
    if np.max(np.abs(roots - roots[0])) > 1.0e-12:
        raise ExperimentContractError("robot root pose animation is forbidden")
    if np.any(positions < limits[:, 0]) or np.any(positions > limits[:, 1]):
        raise ExperimentContractError("trajectory exceeds an official joint limit")
    if np.any(velocity_limit <= 0.0) or np.any(acceleration_limit <= 0.0):
        raise ExperimentContractError("trajectory velocity/acceleration limits must be positive")
    velocity = np.gradient(positions, times, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
    maximum_velocity = np.max(np.abs(velocity), axis=0)
    maximum_acceleration = np.max(np.abs(acceleration), axis=0)
    if np.any(maximum_velocity > velocity_limit + 1.0e-9):
        raise ExperimentContractError("trajectory exceeds an official joint velocity limit")
    if np.any(maximum_acceleration > acceleration_limit + 1.0e-9):
        raise ExperimentContractError("trajectory exceeds the declared acceleration limit")
    return {
        "joint_names": list(LEFT_FOLLOWER_JOINTS),
        "frame_count": frame_count,
        "root_pose_animated": False,
        "within_position_limits": True,
        "within_velocity_limits": True,
        "within_acceleration_limits": True,
        "max_abs_velocity_rad_s": maximum_velocity.tolist(),
        "max_abs_acceleration_rad_s2": maximum_acceleration.tolist(),
    }


def project_camera_points(
    points_camera_m: np.ndarray,
    intrinsics: Mapping[str, float],
) -> np.ndarray:
    """Project camera-frame XYZ points with the frozen RGB pinhole contract."""

    points = np.asarray(points_camera_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ExperimentContractError("camera points must be a finite array with shape (N, 3)")
    required = {"width", "height", "fx", "fy", "cx", "cy"}
    if set(intrinsics) != required:
        raise ExperimentContractError("camera intrinsics must have exact frozen keys")
    values = {key: float(intrinsics[key]) for key in required}
    if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
        raise ExperimentContractError("camera intrinsics must be finite and positive")
    depth = points[:, 2]
    if np.any(depth <= 0.0):
        raise ExperimentContractError("projection requires positive camera depth")
    return np.column_stack(
        (
            values["fx"] * points[:, 0] / depth + values["cx"],
            values["fy"] * points[:, 1] / depth + values["cy"],
        )
    )


def _triangle_normals(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(length) & (length > 1.0e-12)
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / length[valid, None]
    return normals, valid


def anchor_ten_points_to_back_mesh(
    vertices_m: np.ndarray,
    triangle_indices: np.ndarray,
    *,
    back_axis: int,
    back_sign: int,
) -> Tuple[MeshAnchor, ...]:
    """Transfer the ten WAM identities to exact mesh triangles.

    The layout is defined in normalized torso coordinates and therefore does not reuse the old
    hand-authored metre coordinates.  Each returned anchor persists one triangle ID and its
    barycentric coordinates for deterministic replay.
    """

    vertices = np.asarray(vertices_m, dtype=np.float64)
    faces = np.asarray(triangle_indices, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ExperimentContractError("mesh vertices must be finite with shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 10:
        raise ExperimentContractError("mesh triangles must have shape (M, 3) with M >= 10")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ExperimentContractError("mesh triangle index is out of range")
    if back_axis not in (0, 1, 2) or back_sign not in (-1, 1):
        raise ExperimentContractError("back axis/sign is invalid")
    lateral_axis, vertical_axis = [axis for axis in range(3) if axis != back_axis]
    # For the NVIDIA medical character the remaining axes are X then Z.  Reject an orientation
    # that would treat height as lateral rather than silently swapping anatomical ordering.
    if {lateral_axis, vertical_axis} != {0, 2}:
        raise ExperimentContractError("back axis must leave X lateral and Z vertical")
    lateral_axis, vertical_axis = 0, 2

    normals, nondegenerate = _triangle_normals(vertices, faces)
    facing = nondegenerate & (normals[:, back_axis] * back_sign > 0.50)
    candidate_ids = np.flatnonzero(facing)
    if len(candidate_ids) < 10:
        raise ExperimentContractError("mesh has insufficient back-facing triangles")
    centroids = vertices[faces[candidate_ids]].mean(axis=1)
    lateral_min, lateral_max = np.min(centroids[:, lateral_axis]), np.max(
        centroids[:, lateral_axis]
    )
    vertical_min, vertical_max = np.min(centroids[:, vertical_axis]), np.max(
        centroids[:, vertical_axis]
    )
    if lateral_max - lateral_min <= 1.0e-6 or vertical_max - vertical_min <= 1.0e-6:
        raise ExperimentContractError("back mesh has degenerate lateral/vertical extent")
    normalized = np.column_stack(
        (
            (centroids[:, lateral_axis] - lateral_min) / (lateral_max - lateral_min),
            (centroids[:, vertical_axis] - vertical_min) / (vertical_max - vertical_min),
        )
    )
    # R/L keeps the existing WAM identity convention: R precedes L and lies at lower image X.
    lateral_targets = (0.34, 0.66)
    vertical_targets = (0.80, 0.66, 0.52, 0.38, 0.24)
    desired = [
        (lateral, vertical)
        for vertical in vertical_targets
        for lateral in lateral_targets
    ]
    used = set()
    anchors = []
    for name, target in zip(POINT_NAMES, desired):
        distance = np.sum((normalized - np.asarray(target)[None, :]) ** 2, axis=1)
        order = np.argsort(distance, kind="stable")
        selected_local = next((int(index) for index in order if int(index) not in used), None)
        if selected_local is None:
            raise ExperimentContractError("could not assign ten unique back triangles")
        used.add(selected_local)
        triangle_id = int(candidate_ids[selected_local])
        barycentric = np.full(3, 1.0 / 3.0, dtype=np.float64)
        position = barycentric @ vertices[faces[triangle_id]]
        anchors.append(
            MeshAnchor(
                name=name,
                triangle_id=triangle_id,
                barycentric=barycentric,
                position_m=position,
                surface_normal=normals[triangle_id].copy(),
            )
        )
    positions = np.asarray([anchor.position_m for anchor in anchors])
    if not (
        np.all(positions[0::2, lateral_axis] < positions[1::2, lateral_axis])
        and np.all(np.diff(positions[0::2, vertical_axis]) < 0.0)
        and np.all(np.diff(positions[1::2, vertical_axis]) < 0.0)
    ):
        raise ExperimentContractError("ten-point anatomical ordering did not survive mesh transfer")
    return tuple(anchors)


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels < 0:
        raise ExperimentContractError("mask dilation must be nonnegative")
    result = np.asarray(mask, dtype=bool).copy()
    if pixels == 0:
        return result
    source = result.copy()
    height, width = result.shape
    for dy in range(-pixels, pixels + 1):
        for dx in range(-pixels, pixels + 1):
            if dx * dx + dy * dy > pixels * pixels:
                continue
            source_y0, source_y1 = max(0, -dy), min(height, height - dy)
            source_x0, source_x1 = max(0, -dx), min(width, width - dx)
            target_y0, target_y1 = source_y0 + dy, source_y1 + dy
            target_x0, target_x1 = source_x0 + dx, source_x1 + dx
            result[target_y0:target_y1, target_x0:target_x1] |= source[
                source_y0:source_y1, source_x0:source_x1
            ]
    return result


def measure_rendered_visibility(
    *,
    point_names: Sequence[str],
    projected_pixels: np.ndarray,
    point_depth_m: np.ndarray,
    robot_mask: np.ndarray,
    human_mask: np.ndarray,
    depth_m: np.ndarray,
    support_radius_px: int,
    robot_mask_dilation_px: int,
    front_depth_margin_m: float,
) -> Tuple[RenderedVisibility, ...]:
    """Measure each landmark support from instance masks and the rendered depth buffer."""

    names = tuple(point_names)
    pixels = np.asarray(projected_pixels, dtype=np.float64)
    point_depth = np.asarray(point_depth_m, dtype=np.float64)
    robot = np.asarray(robot_mask, dtype=bool)
    human = np.asarray(human_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    if pixels.shape != (len(names), 2) or point_depth.shape != (len(names),):
        raise ExperimentContractError("point identity, projection, and depth shapes disagree")
    if robot.ndim != 2 or human.shape != robot.shape or depth.shape != robot.shape:
        raise ExperimentContractError("rendered mask/depth products must share one 2-D shape")
    if not np.isfinite(pixels).all() or not np.isfinite(point_depth).all() or np.any(
        point_depth <= 0.0
    ):
        raise ExperimentContractError("projected point products must be finite and in front")
    if type(support_radius_px) is not int or support_radius_px <= 0:
        raise ExperimentContractError("support radius must be a positive integer")
    if (
        not math.isfinite(front_depth_margin_m)
        or front_depth_margin_m < 0.0
        or type(robot_mask_dilation_px) is not int
        or robot_mask_dilation_px < 0
    ):
        raise ExperimentContractError("mask/depth tolerances are invalid")
    dilated_robot = _dilate(robot, robot_mask_dilation_px)
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    height, width = robot.shape
    yy, xx = np.ogrid[:height, :width]
    rows = []
    robot_coordinates = np.argwhere(dilated_robot)
    for name, center, expected_depth in zip(names, pixels, point_depth):
        support = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= support_radius_px**2
        total = int(np.count_nonzero(support))
        if total == 0:
            raise ExperimentContractError("projected support lies outside the render product")
        valid = support & valid_depth
        robot_front = (
            support
            & dilated_robot
            & valid_depth
            & (depth <= expected_depth - front_depth_margin_m)
        )
        human_visible = support & human & valid_depth & ~robot_front
        if len(robot_coordinates):
            dy = robot_coordinates[:, 0] - center[1]
            dx = robot_coordinates[:, 1] - center[0]
            edge_distance = float(np.sqrt(np.min(dx * dx + dy * dy)))
        else:
            edge_distance = float("inf")
        rows.append(
            RenderedVisibility(
                name=str(name),
                visible_fraction=float(np.count_nonzero(human_visible) / total),
                robot_front_fraction=float(np.count_nonzero(robot_front) / total),
                valid_depth_fraction=float(np.count_nonzero(valid) / total),
                mask_edge_distance_px=edge_distance,
            )
        )
    return tuple(rows)


def validate_stage_inventory(raw: Mapping[str, object]) -> dict:
    """Fail closed unless the rendered stage is the pinned full official scene."""

    required = {
        "meters_per_unit",
        "import_root_scale",
        "link_count",
        "joint_count",
        "revolute_joint_names",
        "animated_joint_names",
        "robot_pixel_count",
        "human_pixel_count",
        "human_asset_sha256",
        "robot_urdf_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ExperimentContractError("stage inventory must have exact keys")
    value = dict(raw)
    if value["meters_per_unit"] != 1.0 or value["import_root_scale"] != 1.0:
        raise ExperimentContractError("Isaac stage/import root scale must be exactly one metre")
    if value["link_count"] != 24 or value["joint_count"] != 23:
        raise ExperimentContractError("stage does not contain the full OpenArm v1 dual arm")
    if tuple(value["revolute_joint_names"]) != LEFT_FOLLOWER_JOINTS + RIGHT_REFERENCE_JOINTS:
        raise ExperimentContractError("stage revolute joint inventory is not the official dual arm")
    if tuple(value["animated_joint_names"]) != LEFT_FOLLOWER_JOINTS:
        raise ExperimentContractError("only the exact left follower chain may be animated")
    if type(value["robot_pixel_count"]) is not int or value["robot_pixel_count"] <= 0:
        raise ExperimentContractError("official OpenArm must produce nonzero rendered pixels")
    if type(value["human_pixel_count"]) is not int or value["human_pixel_count"] <= 0:
        raise ExperimentContractError("official human must produce nonzero rendered pixels")
    if value["human_asset_sha256"] != NVIDIA_HUMAN_SHA256:
        raise ExperimentContractError("human asset hash is not the pinned NVIDIA model")
    if value["robot_urdf_sha256"] != OFFICIAL_CLEAN_URDF_SHA256:
        raise ExperimentContractError("robot URDF hash is not the clean official v1 file")
    return value


def scenario_name(target_name: str, wrist_index: int, camera_pose: str, seed: int) -> str:
    """Build a deterministic, path-safe scenario identity."""

    if target_name not in POINT_NAMES:
        raise ExperimentContractError("scenario target is not one of the ten WAM identities")
    if type(wrist_index) is not int or wrist_index < 0 or wrist_index > 99:
        raise ExperimentContractError("wrist index must be an integer from 0 to 99")
    if type(seed) is not int or seed < 0 or seed > 9999:
        raise ExperimentContractError("seed must be an integer from 0 to 9999")
    if type(camera_pose) is not str or _SAFE_SCENARIO.fullmatch(camera_pose) is None:
        raise ExperimentContractError("camera pose label is not path-safe")
    return f"{target_name}__wrist{wrist_index:02d}__{camera_pose}__seed{seed:04d}"


def primary_experiment_contract() -> dict:
    """Return the immutable scope labels that every result must carry."""

    return {
        "robot_geometry": "official_openarm_v1_full_dual_arm",
        "animated_chain": "left_follower_openarm_left_joint1_to_7",
        "human_geometry": "official_nvidia_male_adult_medical_01",
        "visibility_source": "rendered_instance_mask_and_depth",
        "camera_pose_status": "installation_pose_uncommissioned",
        "installation_specific_pass": False,
        "result_scope": "parameter_sensitivity_only",
        "physical_accuracy_pass": False,
    }


__all__ = [
    "ExperimentContractError",
    "JointTrajectory",
    "LEFT_FOLLOWER_JOINTS",
    "MeshAnchor",
    "NVIDIA_HUMAN_SHA256",
    "OcclusionPolicyDecision",
    "OFFICIAL_CLEAN_URDF_SHA256",
    "ORBBEC_RGB_INTRINSICS",
    "OpenArmUrdfInventory",
    "POINT_NAMES",
    "PositionIkCandidate",
    "RIGHT_REFERENCE_JOINTS",
    "RenderedVisibility",
    "anchor_ten_points_to_back_mesh",
    "build_minimum_jerk_trajectory",
    "decide_occlusion_policy",
    "inspect_openarm_urdf",
    "left_joint_limits",
    "left_joint_velocity_limits",
    "left_tip_forward_kinematics",
    "measure_rendered_visibility",
    "primary_experiment_contract",
    "project_camera_points",
    "quintic_blend",
    "scenario_name",
    "solve_left_position_ik",
    "solve_left_position_ik_candidates",
    "validate_left_joint_trajectory",
    "validate_stage_inventory",
]
