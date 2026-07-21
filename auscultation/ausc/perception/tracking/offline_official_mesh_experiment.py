"""Deterministic oracle-landmark-noise core for the Mac occlusion study.

This is a method upper-bound simulation, not trained-model validation.  It has
no file-output, hardware, Isaac, or activation path.  It plans motion on the
pinned official OpenArm chain and evaluates three declared occlusion policies.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import mujoco
import numpy as np

from .experimental_visibility_geometry import (
    DecisionSurface,
    VisibilityGeometryResult,
    evaluate_visibility_geometry,
)
from .offline_mesh_simulation import (
    GroundTruthFrame,
    EXPECTED_RUNTIME_MANIFEST_SHA256,
    EXPECTED_SCENE_CONTRACT_SHA256,
    ORACLE_SIMULATION_STATUS,
    ObservationFrame,
    OfflineFrameEvidence,
    OfflineSimulationError,
    POLICY_NAMES,
    PolicyFrameOutput,
    ROBOT_WORLD_FROM_ROOT,
    RequiredStratum,
    evaluate_occlusion_policies,
    oracle_frame_evidence_sha256,
    project_cv_points,
    raw_render_sha256 as oracle_raw_render_sha256,
    scene_contract_sha256,
    score_occlusion_policy_metrics,
    validate_oracle_frame_evidence,
)
from .offline_official_mesh_assets import (
    DEFAULT_RAW_MANIFEST,
    EXPECTED_ASSET_MANIFEST_SHA256,
    EXPECTED_CONVERSION_MANIFEST_SHA256,
    canonical_json_bytes,
)
from .official_openarm_experiment import (
    ExperimentContractError,
    OFFICIAL_CLEAN_URDF_SHA256,
    POINT_NAMES,
    RenderedVisibility,
    build_minimum_jerk_trajectory,
    left_joint_limits,
    left_joint_velocity_limits,
    left_tip_forward_kinematics,
    solve_left_position_ik,
    solve_left_position_ik_candidates,
    validate_left_joint_trajectory,
)


class OfflineOfficialMeshExperimentError(RuntimeError):
    """The frozen offline experiment contract was violated."""


IK_SEED_RNG = 20260715
EXPECTED_IK_SEED_LE_F64_SHA256 = (
    "c7c3407b5f86ccffdc4761b1dd9752105f2cd051710bed7f2560d7553cc9c8f5"
)
EXPECTED_ANCHOR_CONTRACT_SHA256 = (
    "3ea89362672086acea6f7af5aaed5baa27def8a58fc07db59d1ddb69476f87d3"
)
CONTACT_NORMAL_OFFSET_M = 0.03
APPROACH_NORMAL_OFFSET_M = 0.15
IK_TOLERANCE_M = 5.0e-4
IK_MAX_ITERATIONS = 320
IK_MIN_ORIENTATION_SEPARATION_RAD = math.radians(20.0)
TRAJECTORY_FRAMES_PER_SEGMENT = 31
TRAJECTORY_SECONDS_PER_SEGMENT = 1.5
TRAJECTORY_ACCELERATION_LIMIT_RAD_S2 = 8.0
EXECUTION_STATUSES = (
    "COMPLETE",
    "IK_UNREACHABLE",
    "INSUFFICIENT_WRIST_ATTITUDES",
    "TRAJECTORY_INVALID",
)
DECISION_SURFACE = DecisionSurface(
    min_relative_geometry_weight=0.05,
    max_depth_consistency_sigma=3.0,
    min_second_spatial_eigenvalue_m2=2.0e-4,
    min_hull_area_m2=1.0e-3,
    min_spread_m=8.0e-2,
    max_condition_number=40.0,
    max_residual_m=4.0e-3,
    max_target_leverage=4.0,
    max_target_error_p95_m=7.0e-3,
)
OBSERVATION_LATERAL_STD_M = 1.0e-3
OBSERVATION_AXIAL_STD_M = 2.0e-3
OBSERVATION_ROBUST_DELTA_M = 3.0e-3
OCCLUSION_DURATIONS_S = (0.10, 0.25, 0.50, 1.00)
POST_EVENT_DELTA_S = 0.05
FRAME_PERIOD_S = 0.05
SCORING_TRAJECTORY_INDICES = tuple(range(121))
PRECONTACT_TRAJECTORY_INDICES = tuple(range(61))
DEPART_TRAJECTORY_INDICES = tuple(range(61, 121))
OCCLUSION_HOLD_FRAMES = MappingProxyType(
    {0.10: 2, 0.25: 5, 0.50: 10, 1.00: 20}
)
PATIENT_TRANSLATION_SPEED_M_S = 0.012
PATIENT_MOTION_BOUND_M_S = 0.013
OBSERVATION_SEED_ROOT = 51020260715
PATIENT_MOTION_SEED_ROOT = 51020260716
REQUIRED_STRATA_COUNT = 80
ANCHOR_QUANTIZATION_TOLERANCE_M = 1.0e-9
DEPTH_FOOTPRINT = "observed_uv_bilinear_2x2_exact_labcoat_4_of_4"
LABCOAT_GEOM_IDENTITY = "opaque__fabric__labcoat"
NOISE_SWEEP_SEED_ROOTS = (51020260715, 52020260715, 53020260715)
BIAS_SWEEP_SEED_ROOTS = (61020260715, 62020260715, 63020260715)
DELAY_SWEEP_SEED_ROOTS = (71020260715, 72020260715, 73020260715)
DECLARED_SWEEP_COUNT = 3
EXECUTED_SWEEP_INDICES = (0,)


def _immutable_array(value: np.ndarray, *, dtype: object = np.float64) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


HOME_JOINT_POSITIONS_RAD = _immutable_array(
    np.asarray([-1.0, -1.2, 0.0, 1.15, 0.0, 0.0, -0.30], dtype="<f8"),
    dtype="<f8",
)


def frozen_ik_seed_table(urdf_raw: bytes) -> np.ndarray:
    """Return the exact 37x7 official-limit seed table used by this study."""

    if type(urdf_raw) is not bytes:
        raise OfflineOfficialMeshExperimentError("official URDF must be bytes")
    if hashlib.sha256(urdf_raw).hexdigest() != OFFICIAL_CLEAN_URDF_SHA256:
        raise OfflineOfficialMeshExperimentError("official OpenArm URDF identity changed")
    limits = left_joint_limits(urdf_raw)
    if limits.shape != (7, 2) or not np.isfinite(limits).all():
        raise OfflineOfficialMeshExperimentError("official joint limits are invalid")
    lower, upper = limits[:, 0], limits[:, 1]
    margin = np.minimum(0.08, 0.08 * (upper - lower))
    random_rows = np.random.default_rng(IK_SEED_RNG).uniform(
        lower + margin,
        upper - margin,
        size=(36, 7),
    )
    table = np.vstack((HOME_JOINT_POSITIONS_RAD, random_rows)).astype(
        "<f8", copy=False
    )
    digest = hashlib.sha256(table.tobytes(order="C")).hexdigest()
    if digest != EXPECTED_IK_SEED_LE_F64_SHA256:
        raise OfflineOfficialMeshExperimentError("frozen IK seed recipe drifted")
    return _immutable_array(table, dtype="<f8")


@dataclass(frozen=True)
class OfficialAnchor:
    """One exact triangle+barycentric anchor from the pinned labcoat mesh."""

    name: str
    triangle_id: int
    source_vertex_indices: tuple[int, int, int]
    barycentric: np.ndarray
    position_m: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        if self.name not in POINT_NAMES:
            raise OfflineOfficialMeshExperimentError("anchor name is not frozen")
        if type(self.triangle_id) is not int or self.triangle_id < 0:
            raise OfflineOfficialMeshExperimentError("anchor triangle id is invalid")
        indices = tuple(self.source_vertex_indices)
        if len(indices) != 3 or any(type(value) is not int or value < 0 for value in indices):
            raise OfflineOfficialMeshExperimentError("anchor vertex identity is invalid")
        barycentric = _immutable_array(self.barycentric)
        position = _immutable_array(self.position_m)
        normal = _immutable_array(self.normal)
        if any(value.shape != (3,) or not np.isfinite(value).all() for value in (
            barycentric,
            position,
            normal,
        )):
            raise OfflineOfficialMeshExperimentError("anchor arrays are invalid")
        if not math.isclose(float(np.sum(barycentric)), 1.0, abs_tol=1.0e-12):
            raise OfflineOfficialMeshExperimentError("anchor barycentric weights changed")
        if np.any(barycentric < 0.0):
            raise OfflineOfficialMeshExperimentError("anchor barycentric weights are negative")
        if not math.isclose(float(np.linalg.norm(normal)), 1.0, abs_tol=1.0e-12):
            raise OfflineOfficialMeshExperimentError("anchor normal is not unit length")
        object.__setattr__(self, "source_vertex_indices", indices)
        object.__setattr__(self, "barycentric", barycentric)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "normal", normal)


def _anchor_json_rows(anchors: Sequence[OfficialAnchor]) -> list[dict[str, object]]:
    return [
        {
            "barycentric": anchor.barycentric.tolist(),
            "name": anchor.name,
            "normal": anchor.normal.tolist(),
            "position_m": anchor.position_m.tolist(),
            "source_vertex_indices": list(anchor.source_vertex_indices),
            "triangle_id": anchor.triangle_id,
        }
        for anchor in anchors
    ]


def anchor_contract_sha256(anchors: Sequence[OfficialAnchor]) -> str:
    """Hash the exact ten persisted triangle+barycentric anchor records."""

    rows = tuple(anchors)
    if tuple(row.name for row in rows) != POINT_NAMES:
        raise OfflineOfficialMeshExperimentError("ten-point anchor order changed")
    return hashlib.sha256(canonical_json_bytes(_anchor_json_rows(rows))).hexdigest()


def load_official_anchor_contract(
    manifest_raw: bytes | None = None,
) -> tuple[OfficialAnchor, ...]:
    """Load anchors only from the exact frozen conversion manifest bytes."""

    if manifest_raw is None:
        try:
            raw = Path(DEFAULT_RAW_MANIFEST).read_bytes()
        except OSError as exc:
            raise OfflineOfficialMeshExperimentError(
                "frozen conversion manifest is unavailable"
            ) from exc
    else:
        if type(manifest_raw) is not bytes:
            raise OfflineOfficialMeshExperimentError("conversion manifest must be bytes")
        raw = manifest_raw
    if hashlib.sha256(raw).hexdigest() != EXPECTED_CONVERSION_MANIFEST_SHA256:
        raise OfflineOfficialMeshExperimentError("conversion manifest identity changed")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfflineOfficialMeshExperimentError("conversion manifest is invalid JSON") from exc
    if canonical_json_bytes(manifest) != raw:
        raise OfflineOfficialMeshExperimentError("conversion manifest is not canonical")
    try:
        rows = manifest["human"]["labcoat"]["anchors"]
    except (KeyError, TypeError) as exc:
        raise OfflineOfficialMeshExperimentError("labcoat anchors are unavailable") from exc
    if not isinstance(rows, list):
        raise OfflineOfficialMeshExperimentError("labcoat anchor rows are invalid")
    try:
        anchors = tuple(
            OfficialAnchor(
                name=str(row["name"]),
                triangle_id=int(row["triangle_id"]),
                source_vertex_indices=tuple(int(value) for value in row["source_vertex_indices"]),
                barycentric=np.asarray(row["barycentric"], dtype=np.float64),
                position_m=np.asarray(row["position_m"], dtype=np.float64),
                normal=np.asarray(row["normal"], dtype=np.float64),
            )
            for row in rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OfflineOfficialMeshExperimentError("labcoat anchor row is malformed") from exc
    if anchor_contract_sha256(anchors) != EXPECTED_ANCHOR_CONTRACT_SHA256:
        raise OfflineOfficialMeshExperimentError("official anchor contract changed")
    return anchors


@dataclass(frozen=True)
class WristBranchPlan:
    """One valid approach/contact/depart trajectory for a distinct wrist attitude."""

    wrist_index: int
    status: str
    failure_reason: str | None
    contact_joint_positions_rad: np.ndarray
    approach_joint_positions_rad: np.ndarray
    contact_tip_rotation: np.ndarray
    position_error_m: float
    orientation_separation_from_first_rad: float
    trajectory_joint_positions: np.ndarray
    trajectory_time_s: np.ndarray
    trajectory_root_poses: np.ndarray
    trajectory_validation: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.wrist_index) is not int or self.wrist_index not in (0, 1):
            raise OfflineOfficialMeshExperimentError("wrist index is invalid")
        if self.status != "COMPLETE" or self.failure_reason is not None:
            raise OfflineOfficialMeshExperimentError("stored wrist branch is not complete")
        arrays = {
            "contact_joint_positions_rad": _immutable_array(
                self.contact_joint_positions_rad
            ),
            "approach_joint_positions_rad": _immutable_array(
                self.approach_joint_positions_rad
            ),
            "contact_tip_rotation": _immutable_array(self.contact_tip_rotation),
            "trajectory_joint_positions": _immutable_array(
                self.trajectory_joint_positions
            ),
            "trajectory_time_s": _immutable_array(self.trajectory_time_s),
            "trajectory_root_poses": _immutable_array(self.trajectory_root_poses),
        }
        if arrays["contact_joint_positions_rad"].shape != (7,):
            raise OfflineOfficialMeshExperimentError("contact joint vector is invalid")
        if arrays["approach_joint_positions_rad"].shape != (7,):
            raise OfflineOfficialMeshExperimentError("approach joint vector is invalid")
        if arrays["contact_tip_rotation"].shape != (3, 3):
            raise OfflineOfficialMeshExperimentError("contact wrist rotation is invalid")
        if arrays["trajectory_joint_positions"].shape != (121, 7):
            raise OfflineOfficialMeshExperimentError("trajectory sample count changed")
        if arrays["trajectory_time_s"].shape != (121,):
            raise OfflineOfficialMeshExperimentError("trajectory timestamps changed")
        if arrays["trajectory_root_poses"].shape != (121, 4, 4):
            raise OfflineOfficialMeshExperimentError("trajectory root poses changed")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise OfflineOfficialMeshExperimentError("wrist branch contains nonfinite data")
        if not math.isfinite(float(self.position_error_m)) or not 0.0 <= self.position_error_m <= IK_TOLERANCE_M:
            raise OfflineOfficialMeshExperimentError("wrist position error is invalid")
        if not math.isfinite(float(self.orientation_separation_from_first_rad)):
            raise OfflineOfficialMeshExperimentError("wrist separation is invalid")
        validation = MappingProxyType(dict(self.trajectory_validation))
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "trajectory_validation", validation)


@dataclass(frozen=True)
class TargetMotionPlan:
    """Explicit motion-planning status for one of the ten required targets."""

    target_name: str
    status: str
    failure_reason: str | None
    contact_target_world_m: np.ndarray
    approach_target_world_m: np.ndarray
    contact_target_robot_root_m: np.ndarray
    approach_target_robot_root_m: np.ndarray
    candidate_count: int
    branches: tuple[WristBranchPlan, ...]

    def __post_init__(self) -> None:
        if self.target_name not in POINT_NAMES or self.status not in EXECUTION_STATUSES:
            raise OfflineOfficialMeshExperimentError("target plan identity is invalid")
        for name in (
            "contact_target_world_m",
            "approach_target_world_m",
            "contact_target_robot_root_m",
            "approach_target_robot_root_m",
        ):
            value = _immutable_array(getattr(self, name))
            if value.shape != (3,) or not np.isfinite(value).all():
                raise OfflineOfficialMeshExperimentError("target waypoint is invalid")
            object.__setattr__(self, name, value)
        if type(self.candidate_count) is not int or not 0 <= self.candidate_count <= 2:
            raise OfflineOfficialMeshExperimentError("target candidate count is invalid")
        if self.status == "COMPLETE":
            if self.failure_reason is not None or len(self.branches) != 2:
                raise OfflineOfficialMeshExperimentError("complete target lacks two branches")
        elif not isinstance(self.failure_reason, str) or not self.failure_reason:
            raise OfflineOfficialMeshExperimentError("failed target lacks a reason")


def _world_point_to_robot_root(point_world_m: np.ndarray) -> np.ndarray:
    point = np.asarray(point_world_m, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise OfflineOfficialMeshExperimentError("world waypoint is invalid")
    homogeneous = np.append(point, 1.0)
    root = np.linalg.inv(ROBOT_WORLD_FROM_ROOT) @ homogeneous
    if not np.allclose(root[3], 1.0, atol=1.0e-12, rtol=0.0):
        raise OfflineOfficialMeshExperimentError("robot root inverse transform failed")
    return root[:3]


def plan_official_ik_matrix(
    urdf_raw: bytes,
    anchors: Sequence[OfficialAnchor],
) -> tuple[TargetMotionPlan, ...]:
    """Attempt every target and emit one explicit status without dropping failures."""

    anchor_rows = tuple(anchors)
    if anchor_contract_sha256(anchor_rows) != EXPECTED_ANCHOR_CONTRACT_SHA256:
        raise OfflineOfficialMeshExperimentError("IK planner requires exact ten anchors")
    seeds = frozen_ik_seed_table(urdf_raw)
    limits = left_joint_limits(urdf_raw)
    velocity_limits = left_joint_velocity_limits(urdf_raw)
    acceleration_limits = np.full(7, TRAJECTORY_ACCELERATION_LIMIT_RAD_S2)
    plans: list[TargetMotionPlan] = []
    for anchor in anchor_rows:
        contact_world = anchor.position_m + CONTACT_NORMAL_OFFSET_M * anchor.normal
        approach_world = anchor.position_m + APPROACH_NORMAL_OFFSET_M * anchor.normal
        contact_root = _world_point_to_robot_root(contact_world)
        approach_root = _world_point_to_robot_root(approach_world)
        status = "COMPLETE"
        failure_reason: str | None = None
        branches: list[WristBranchPlan] = []
        try:
            candidates = solve_left_position_ik_candidates(
                urdf_raw,
                contact_root,
                initial_seeds=seeds,
                tolerance_m=IK_TOLERANCE_M,
                max_iterations=IK_MAX_ITERATIONS,
                max_solutions=2,
                min_orientation_separation_rad=IK_MIN_ORIENTATION_SEPARATION_RAD,
            )
        except ExperimentContractError as exc:
            candidates = ()
            status = "IK_UNREACHABLE"
            failure_reason = f"contact_ik_contract_error:{exc}"
        if status == "COMPLETE" and not candidates:
            status = "IK_UNREACHABLE"
            failure_reason = "no_contact_ik_solution"
        elif status == "COMPLETE" and len(candidates) < 2:
            status = "INSUFFICIENT_WRIST_ATTITUDES"
            failure_reason = "fewer_than_two_distinct_contact_wrist_attitudes"

        if status == "COMPLETE":
            branch_failures: list[str] = []
            for wrist_index, candidate in enumerate(candidates):
                try:
                    approach_q = solve_left_position_ik(
                        urdf_raw,
                        approach_root,
                        initial_q=candidate.joint_positions,
                        tolerance_m=IK_TOLERANCE_M,
                        max_iterations=IK_MAX_ITERATIONS,
                    )
                    trajectory = build_minimum_jerk_trajectory(
                        np.vstack(
                            (
                                HOME_JOINT_POSITIONS_RAD,
                                approach_q,
                                candidate.joint_positions,
                                approach_q,
                                HOME_JOINT_POSITIONS_RAD,
                            )
                        ),
                        frames_per_segment=TRAJECTORY_FRAMES_PER_SEGMENT,
                        seconds_per_segment=TRAJECTORY_SECONDS_PER_SEGMENT,
                    )
                    validation = validate_left_joint_trajectory(
                        trajectory.joint_positions,
                        trajectory.time_s,
                        trajectory.root_poses,
                        joint_limits=limits,
                        velocity_limits=velocity_limits,
                        acceleration_limits=acceleration_limits,
                    )
                except ExperimentContractError as exc:
                    branch_failures.append(f"wrist_{wrist_index}:{exc}")
                    continue
                branches.append(
                    WristBranchPlan(
                        wrist_index=wrist_index,
                        status="COMPLETE",
                        failure_reason=None,
                        contact_joint_positions_rad=candidate.joint_positions,
                        approach_joint_positions_rad=approach_q,
                        contact_tip_rotation=candidate.tip_rotation,
                        position_error_m=float(candidate.position_error_m),
                        orientation_separation_from_first_rad=float(
                            candidate.orientation_separation_from_first_rad
                        ),
                        trajectory_joint_positions=trajectory.joint_positions,
                        trajectory_time_s=trajectory.time_s,
                        trajectory_root_poses=trajectory.root_poses,
                        trajectory_validation=validation,
                    )
                )
            if branch_failures or len(branches) != 2:
                status = "TRAJECTORY_INVALID"
                failure_reason = ";".join(branch_failures) or "missing_valid_branch"

        plans.append(
            TargetMotionPlan(
                target_name=anchor.name,
                status=status,
                failure_reason=failure_reason,
                contact_target_world_m=contact_world,
                approach_target_world_m=approach_world,
                contact_target_robot_root_m=contact_root,
                approach_target_robot_root_m=approach_root,
                candidate_count=len(candidates),
                branches=tuple(branches),
            )
        )
    if tuple(plan.target_name for plan in plans) != POINT_NAMES:
        raise OfflineOfficialMeshExperimentError("IK target attempt order changed")
    return tuple(plans)


def _motion_plan_value(plans: Sequence[TargetMotionPlan]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for plan in plans:
        rows.append(
            {
                "target_name": plan.target_name,
                "status": plan.status,
                "failure_reason": plan.failure_reason,
                "candidate_count": plan.candidate_count,
                "contact_target_world_m": plan.contact_target_world_m.tolist(),
                "approach_target_world_m": plan.approach_target_world_m.tolist(),
                "contact_target_robot_root_m": (
                    plan.contact_target_robot_root_m.tolist()
                ),
                "approach_target_robot_root_m": (
                    plan.approach_target_robot_root_m.tolist()
                ),
                "branches": [
                    {
                        "wrist_index": branch.wrist_index,
                        "contact_joint_positions_rad": (
                            branch.contact_joint_positions_rad.tolist()
                        ),
                        "approach_joint_positions_rad": (
                            branch.approach_joint_positions_rad.tolist()
                        ),
                        "contact_tip_rotation": branch.contact_tip_rotation.tolist(),
                        "position_error_m": branch.position_error_m,
                        "orientation_separation_from_first_rad": (
                            branch.orientation_separation_from_first_rad
                        ),
                        "trajectory_joint_positions": (
                            branch.trajectory_joint_positions.tolist()
                        ),
                        "trajectory_time_s": branch.trajectory_time_s.tolist(),
                        "trajectory_root_poses": branch.trajectory_root_poses.tolist(),
                        "trajectory_validation": dict(branch.trajectory_validation),
                    }
                    for branch in plan.branches
                ],
            }
        )
    return {
        "schema": "wam_offline_official_motion_plan_v2",
        "robot_urdf_sha256": OFFICIAL_CLEAN_URDF_SHA256,
        "scene_contract_sha256": EXPECTED_SCENE_CONTRACT_SHA256,
        "robot_world_from_root": ROBOT_WORLD_FROM_ROOT.tolist(),
        "plans": rows,
    }


def revalidate_official_motion_plans(
    urdf_raw: bytes,
    anchors: Sequence[OfficialAnchor],
    target_plans: Sequence[TargetMotionPlan],
) -> str:
    """Recompute FK, limits, timing, roots, and every trajectory before use."""

    if type(urdf_raw) is not bytes or (
        hashlib.sha256(urdf_raw).hexdigest() != OFFICIAL_CLEAN_URDF_SHA256
    ):
        raise OfflineOfficialMeshExperimentError("official motion URDF identity changed")
    if scene_contract_sha256() != EXPECTED_SCENE_CONTRACT_SHA256:
        raise OfflineOfficialMeshExperimentError("official scene identity changed")
    anchors_tuple = tuple(anchors)
    plans = tuple(target_plans)
    if anchor_contract_sha256(anchors_tuple) != EXPECTED_ANCHOR_CONTRACT_SHA256:
        raise OfflineOfficialMeshExperimentError("official motion anchors changed")
    if len(plans) != len(POINT_NAMES) or tuple(
        plan.target_name for plan in plans
    ) != POINT_NAMES:
        raise OfflineOfficialMeshExperimentError("official motion plan inventory changed")
    limits = left_joint_limits(urdf_raw)
    velocity_limits = left_joint_velocity_limits(urdf_raw)
    acceleration_limits = np.full(7, TRAJECTORY_ACCELERATION_LIMIT_RAD_S2)
    expected_waypoint_indices = (0, 30, 60, 90, 120)
    for anchor, plan in zip(anchors_tuple, plans, strict=True):
        if not isinstance(plan, TargetMotionPlan):
            raise OfflineOfficialMeshExperimentError("motion plan row type changed")
        expected_contact_world = (
            anchor.position_m + CONTACT_NORMAL_OFFSET_M * anchor.normal
        )
        expected_approach_world = (
            anchor.position_m + APPROACH_NORMAL_OFFSET_M * anchor.normal
        )
        expected_contact_root = _world_point_to_robot_root(expected_contact_world)
        expected_approach_root = _world_point_to_robot_root(expected_approach_world)
        for actual, expected, label in (
            (plan.contact_target_world_m, expected_contact_world, "contact world"),
            (plan.approach_target_world_m, expected_approach_world, "approach world"),
            (plan.contact_target_robot_root_m, expected_contact_root, "contact root"),
            (plan.approach_target_robot_root_m, expected_approach_root, "approach root"),
        ):
            if not np.allclose(actual, expected, atol=1.0e-12, rtol=0.0):
                raise OfflineOfficialMeshExperimentError(
                    f"official motion {label} waypoint changed"
                )
        if plan.status != "COMPLETE":
            continue
        if plan.candidate_count != 2 or tuple(
            branch.wrist_index for branch in plan.branches
        ) != (0, 1):
            raise OfflineOfficialMeshExperimentError(
                "complete motion plan lost two wrist branches"
            )
        rotations: list[np.ndarray] = []
        for branch in plan.branches:
            for joints, label in (
                (branch.contact_joint_positions_rad, "contact joint"),
                (branch.approach_joint_positions_rad, "approach joint"),
            ):
                if np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1]):
                    raise OfflineOfficialMeshExperimentError(
                        f"official motion {label} limit changed"
                    )
            contact_fk = left_tip_forward_kinematics(
                urdf_raw, branch.contact_joint_positions_rad
            )
            approach_fk = left_tip_forward_kinematics(
                urdf_raw, branch.approach_joint_positions_rad
            )
            contact_error = float(
                np.linalg.norm(contact_fk[:3, 3] - expected_contact_root)
            )
            approach_error = float(
                np.linalg.norm(approach_fk[:3, 3] - expected_approach_root)
            )
            if contact_error > IK_TOLERANCE_M or approach_error > IK_TOLERANCE_M:
                raise OfflineOfficialMeshExperimentError(
                    "official motion FK waypoint is unreachable"
                )
            if not math.isclose(
                branch.position_error_m, contact_error, abs_tol=1.0e-12, rel_tol=0.0
            ):
                raise OfflineOfficialMeshExperimentError(
                    "stored contact FK error changed"
                )
            if not np.allclose(
                branch.contact_tip_rotation,
                contact_fk[:3, :3],
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise OfflineOfficialMeshExperimentError(
                    "stored contact wrist rotation changed"
                )
            rotations.append(contact_fk[:3, :3])
            expected_trajectory = build_minimum_jerk_trajectory(
                np.vstack(
                    (
                        HOME_JOINT_POSITIONS_RAD,
                        branch.approach_joint_positions_rad,
                        branch.contact_joint_positions_rad,
                        branch.approach_joint_positions_rad,
                        HOME_JOINT_POSITIONS_RAD,
                    )
                ),
                frames_per_segment=TRAJECTORY_FRAMES_PER_SEGMENT,
                seconds_per_segment=TRAJECTORY_SECONDS_PER_SEGMENT,
            )
            if not all(
                np.array_equal(actual, expected)
                for actual, expected in (
                    (
                        branch.trajectory_joint_positions,
                        expected_trajectory.joint_positions,
                    ),
                    (branch.trajectory_time_s, expected_trajectory.time_s),
                    (branch.trajectory_root_poses, expected_trajectory.root_poses),
                )
            ):
                raise OfflineOfficialMeshExperimentError(
                    "official motion trajectory payload changed"
                )
            expected_waypoints = np.vstack(
                (
                    HOME_JOINT_POSITIONS_RAD,
                    branch.approach_joint_positions_rad,
                    branch.contact_joint_positions_rad,
                    branch.approach_joint_positions_rad,
                    HOME_JOINT_POSITIONS_RAD,
                )
            )
            if not np.allclose(
                branch.trajectory_joint_positions[list(expected_waypoint_indices)],
                expected_waypoints,
                atol=1.0e-15,
                rtol=0.0,
            ):
                raise OfflineOfficialMeshExperimentError(
                    "official motion trajectory endpoints changed"
                )
            validation = validate_left_joint_trajectory(
                branch.trajectory_joint_positions,
                branch.trajectory_time_s,
                branch.trajectory_root_poses,
                joint_limits=limits,
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
            )
            for flag in (
                "within_position_limits",
                "within_velocity_limits",
                "within_acceleration_limits",
            ):
                if validation.get(flag) is not True or (
                    branch.trajectory_validation.get(flag) is not True
                ):
                    raise OfflineOfficialMeshExperimentError(
                        f"official motion trajectory {flag} changed"
                    )
            if not np.allclose(
                np.diff(branch.trajectory_time_s),
                FRAME_PERIOD_S,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise OfflineOfficialMeshExperimentError(
                    "official motion trajectory is not 20 Hz"
                )
        relative = rotations[0].T @ rotations[1]
        separation = float(
            math.acos(
                float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
            )
        )
        if separation + 1.0e-12 < IK_MIN_ORIENTATION_SEPARATION_RAD:
            raise OfflineOfficialMeshExperimentError(
                "official wrist attitudes are insufficiently distinct"
            )
        if not math.isclose(
            plan.branches[0].orientation_separation_from_first_rad,
            0.0,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            plan.branches[1].orientation_separation_from_first_rad,
            separation,
            abs_tol=1.0e-12,
        ):
            raise OfflineOfficialMeshExperimentError(
                "stored wrist attitude separation changed"
            )
    return hashlib.sha256(canonical_json_bytes(_motion_plan_value(plans))).hexdigest()


def _noise_contract_value(
    *,
    seed: int,
    error_camera_m: np.ndarray,
    lateral_noise_std_m: np.ndarray,
    axial_noise_std_m: np.ndarray,
    robust_delta_m: float,
) -> dict[str, object]:
    return {
        "schema": "wam_offline_predeclared_observation_noise_v1",
        "seed": seed,
        "error_camera_m": np.asarray(error_camera_m, dtype=np.float64).tolist(),
        "lateral_noise_std_m": np.asarray(
            lateral_noise_std_m, dtype=np.float64
        ).tolist(),
        "axial_noise_std_m": np.asarray(axial_noise_std_m, dtype=np.float64).tolist(),
        "robust_delta_m": float(robust_delta_m),
    }


@dataclass(frozen=True)
class PredeclaredObservationNoise:
    """A realized, hashed error vector fixed before any policy evaluation."""

    seed: int
    error_camera_m: np.ndarray
    lateral_noise_std_m: np.ndarray
    axial_noise_std_m: np.ndarray
    robust_delta_m: float
    contract_sha256: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise OfflineOfficialMeshExperimentError("observation noise seed is invalid")
        error = _immutable_array(self.error_camera_m)
        lateral = _immutable_array(self.lateral_noise_std_m)
        axial = _immutable_array(self.axial_noise_std_m)
        if error.shape != (len(POINT_NAMES), 3) or not np.isfinite(error).all():
            raise OfflineOfficialMeshExperimentError("observation error realization is invalid")
        if any(
            value.shape != (len(POINT_NAMES),)
            or not np.isfinite(value).all()
            or np.any(value <= 0.0)
            for value in (lateral, axial)
        ):
            raise OfflineOfficialMeshExperimentError("observation covariance is invalid")
        robust_delta = float(self.robust_delta_m)
        if not math.isfinite(robust_delta) or robust_delta <= 0.0:
            raise OfflineOfficialMeshExperimentError("robust delta is invalid")
        expected = hashlib.sha256(
            canonical_json_bytes(
                _noise_contract_value(
                    seed=self.seed,
                    error_camera_m=error,
                    lateral_noise_std_m=lateral,
                    axial_noise_std_m=axial,
                    robust_delta_m=robust_delta,
                )
            )
        ).hexdigest()
        if self.contract_sha256 != expected:
            raise OfflineOfficialMeshExperimentError("observation noise hash changed")
        object.__setattr__(self, "error_camera_m", error)
        object.__setattr__(self, "lateral_noise_std_m", lateral)
        object.__setattr__(self, "axial_noise_std_m", axial)
        object.__setattr__(self, "robust_delta_m", robust_delta)


def make_predeclared_observation_noise(seed: int) -> PredeclaredObservationNoise:
    """Create one deterministic, clipped error realization for all ten anchors."""

    if type(seed) is not int or seed < 0:
        raise OfflineOfficialMeshExperimentError("observation noise seed is invalid")
    lateral = np.full(len(POINT_NAMES), OBSERVATION_LATERAL_STD_M)
    axial = np.full(len(POINT_NAMES), OBSERVATION_AXIAL_STD_M)
    normalized = np.clip(
        np.random.default_rng(seed).standard_normal((len(POINT_NAMES), 3)),
        -2.5,
        2.5,
    )
    scales = np.column_stack((lateral, lateral, axial))
    error = normalized * scales
    value = _noise_contract_value(
        seed=seed,
        error_camera_m=error,
        lateral_noise_std_m=lateral,
        axial_noise_std_m=axial,
        robust_delta_m=OBSERVATION_ROBUST_DELTA_M,
    )
    return PredeclaredObservationNoise(
        seed=seed,
        error_camera_m=error,
        lateral_noise_std_m=lateral,
        axial_noise_std_m=axial,
        robust_delta_m=OBSERVATION_ROBUST_DELTA_M,
        contract_sha256=hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
    )


def observation_sweep_contract() -> Mapping[str, object]:
    """Declare three frozen seeds while truthfully executing only sweep zero."""

    rows = tuple(
        MappingProxyType(
            {
                "sweep_index": index,
                "noise_seed_root": NOISE_SWEEP_SEED_ROOTS[index],
                "bias_seed_root": BIAS_SWEEP_SEED_ROOTS[index],
                "delay_seed_root": DELAY_SWEEP_SEED_ROOTS[index],
                "noise_applied": index in EXECUTED_SWEEP_INDICES,
                "bias_applied": False,
                "delay_applied": False,
            }
        )
        for index in range(DECLARED_SWEEP_COUNT)
    )
    return MappingProxyType(
        {
            "declared_sweep_count": DECLARED_SWEEP_COUNT,
            "executed_sweep_count": len(EXECUTED_SWEEP_INDICES),
            "executed_sweep_indices": EXECUTED_SWEEP_INDICES,
            "bias_model_status": "seed_declared_not_implemented_not_applied",
            "delay_model_status": "seed_declared_not_implemented_not_applied",
            "sweeps": rows,
        }
    )


def _immutable_rigid_transform(value: np.ndarray, label: str) -> np.ndarray:
    transform = _immutable_array(value)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise OfflineOfficialMeshExperimentError(f"{label} is invalid")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12, rtol=0.0):
        raise OfflineOfficialMeshExperimentError(f"{label} homogeneous row changed")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10, rtol=0.0):
        raise OfflineOfficialMeshExperimentError(f"{label} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-10):
        raise OfflineOfficialMeshExperimentError(f"{label} rotation is not proper")
    return transform


def validate_anchor_quantization(
    official_anchor_position_m: np.ndarray,
    rendered_anchor_position_m: np.ndarray,
) -> None:
    """Accept only the frozen fixed-decimal OBJ serialization envelope."""

    official = np.asarray(official_anchor_position_m, dtype=np.float64)
    rendered = np.asarray(rendered_anchor_position_m, dtype=np.float64)
    if (
        official.shape != (len(POINT_NAMES), 3)
        or rendered.shape != official.shape
        or not np.isfinite(official).all()
        or not np.isfinite(rendered).all()
        or not np.allclose(
            rendered,
            official,
            atol=ANCHOR_QUANTIZATION_TOLERANCE_M,
            rtol=0.0,
        )
    ):
        raise OfflineOfficialMeshExperimentError(
            "rendered anchor exceeds the fixed OBJ quantization tolerance"
        )


@dataclass(frozen=True)
class RenderedGeometryGateFrame:
    """Oracle-landmark-noise observation consumed by the geometry gate.

    Exact anchors plus predeclared noise define a method upper bound; this is
    not a trained-model output or validation claim.  Scoring truth is not
    retained here.  Depth is associated only through the exact labcoat
    instance at the observed 2-D location.
    """

    point_names: tuple[str, ...]
    reference_anchor_camera_m: np.ndarray
    observed_anchor_camera_m: np.ndarray
    observed_projected_pixels: np.ndarray
    depth_interval_m: np.ndarray
    depth_mid_m: np.ndarray
    axial_sigma_effective_m: np.ndarray
    depth_valid: np.ndarray
    visibility: tuple[RenderedVisibility, ...]
    target_name: str
    world_from_cv_camera: np.ndarray
    evidence_sha256: str
    raw_render_sha256: str
    predeclared_noise_sha256: str
    conversion_release_id: str
    conversion_manifest_sha256: str
    asset_release_id: str
    asset_manifest_sha256: str
    runtime_manifest_sha256: str
    scene_contract_sha256: str
    depth_footprint: str
    evidence_source: str

    def __post_init__(self) -> None:
        names = tuple(self.point_names)
        if names != POINT_NAMES or self.target_name not in POINT_NAMES:
            raise OfflineOfficialMeshExperimentError("rendered gate point identity changed")
        reference = _immutable_array(self.reference_anchor_camera_m)
        observed = _immutable_array(self.observed_anchor_camera_m)
        pixels = _immutable_array(self.observed_projected_pixels)
        depth_interval = _immutable_array(self.depth_interval_m)
        depth_mid = _immutable_array(self.depth_mid_m)
        axial_sigma = _immutable_array(self.axial_sigma_effective_m)
        depth_valid = _immutable_array(
            np.asarray(self.depth_valid, dtype=np.bool_), dtype=np.bool_
        )
        if any(
            value.shape != (len(POINT_NAMES), 3) or not np.isfinite(value).all()
            for value in (reference, observed)
        ):
            raise OfflineOfficialMeshExperimentError("camera-space anchor arrays are invalid")
        if np.any(reference[:, 2] <= 0.0) or np.any(observed[:, 2] <= 0.0):
            raise OfflineOfficialMeshExperimentError("camera-space anchors must be in front")
        if pixels.shape != (len(POINT_NAMES), 2) or not np.isfinite(pixels).all():
            raise OfflineOfficialMeshExperimentError("observed pixel array is invalid")
        if depth_interval.shape != (len(POINT_NAMES), 2):
            raise OfflineOfficialMeshExperimentError("depth interval shape changed")
        if any(value.shape != (len(POINT_NAMES),) for value in (
            depth_mid,
            axial_sigma,
            depth_valid,
        )):
            raise OfflineOfficialMeshExperimentError("depth support vector shape changed")
        valid = np.asarray(depth_valid, dtype=bool)
        if np.any(valid):
            intervals = depth_interval[valid]
            midpoint = depth_mid[valid]
            sigmas = axial_sigma[valid]
            if (
                not np.isfinite(intervals).all()
                or not np.isfinite(midpoint).all()
                or not np.isfinite(sigmas).all()
                or np.any(intervals <= 0.0)
                or np.any(intervals[:, 0] > intervals[:, 1])
                or np.any(sigmas <= 0.0)
                or not np.allclose(
                    midpoint,
                    np.mean(intervals, axis=1),
                    atol=1.0e-15,
                    rtol=0.0,
                )
            ):
                raise OfflineOfficialMeshExperimentError("valid depth interval is invalid")
        if (
            np.any(np.isfinite(depth_interval[~valid]))
            or np.any(np.isfinite(depth_mid[~valid]))
            or np.any(np.isfinite(axial_sigma[~valid]))
        ):
            raise OfflineOfficialMeshExperimentError(
                "unsupported point carried a depth observation"
            )
        visibility = tuple(self.visibility)
        if len(visibility) != len(POINT_NAMES):
            raise OfflineOfficialMeshExperimentError("rendered visibility count changed")
        for name, row in zip(POINT_NAMES, visibility, strict=True):
            if not isinstance(row, RenderedVisibility) or row.name != name:
                raise OfflineOfficialMeshExperimentError("rendered visibility identity changed")
            if row.visibility_source != "rendered_instance_mask_and_depth":
                raise OfflineOfficialMeshExperimentError("visibility provenance changed")
            values = (
                row.visible_fraction,
                row.robot_front_fraction,
                row.valid_depth_fraction,
            )
            if not all(math.isfinite(float(value)) and 0.0 <= value <= 1.0 for value in values):
                raise OfflineOfficialMeshExperimentError("rendered visibility value is invalid")
        expected_identities = {
            "conversion_release_id": (
                f"official_meshes-{EXPECTED_CONVERSION_MANIFEST_SHA256[:20]}"
            ),
            "conversion_manifest_sha256": EXPECTED_CONVERSION_MANIFEST_SHA256,
            "asset_release_id": f"official_assets-{EXPECTED_ASSET_MANIFEST_SHA256[:20]}",
            "asset_manifest_sha256": EXPECTED_ASSET_MANIFEST_SHA256,
            "runtime_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
            "scene_contract_sha256": EXPECTED_SCENE_CONTRACT_SHA256,
        }
        for field_name, expected in expected_identities.items():
            if getattr(self, field_name) != expected:
                raise OfflineOfficialMeshExperimentError(
                    f"rendered gate {field_name} changed"
                )
        for field_name in (
            "evidence_sha256",
            "raw_render_sha256",
            "predeclared_noise_sha256",
        ):
            digest = getattr(self, field_name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise OfflineOfficialMeshExperimentError(
                    f"rendered gate {field_name} is invalid"
                )
        if self.depth_footprint != DEPTH_FOOTPRINT:
            raise OfflineOfficialMeshExperimentError("depth footprint contract changed")
        if self.evidence_source != (
            "oracle_landmark_noise_simulation_exact_labcoat_segmentation_and_metric_depth"
        ):
            raise OfflineOfficialMeshExperimentError("rendered gate evidence source changed")
        world_from_cv = _immutable_rigid_transform(
            self.world_from_cv_camera, "world_from_cv_camera"
        )
        object.__setattr__(self, "point_names", names)
        object.__setattr__(self, "reference_anchor_camera_m", reference)
        object.__setattr__(self, "observed_anchor_camera_m", observed)
        object.__setattr__(self, "observed_projected_pixels", pixels)
        object.__setattr__(self, "depth_interval_m", depth_interval)
        object.__setattr__(self, "depth_mid_m", depth_mid)
        object.__setattr__(self, "axial_sigma_effective_m", axial_sigma)
        object.__setattr__(self, "depth_valid", depth_valid)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "world_from_cv_camera", world_from_cv)


@dataclass(frozen=True)
class GeometryGateDecision:
    """One oracle-simulation result bound to exact rendered evidence."""

    accepted: bool
    accepted_estimate_world_m: np.ndarray | None
    accepted_error_p95_m: float | None
    visible_point_count_report_only: int
    supported_point_names: tuple[str, ...]
    predeclared_noise_sha256: str
    evidence_sha256: str
    raw_render_sha256: str
    conversion_manifest_sha256: str
    asset_manifest_sha256: str
    runtime_manifest_sha256: str
    scene_contract_sha256: str
    geometry_result: VisibilityGeometryResult

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise OfflineOfficialMeshExperimentError("geometry decision flag is invalid")
        if (
            type(self.visible_point_count_report_only) is not int
            or not 0 <= self.visible_point_count_report_only <= len(POINT_NAMES)
        ):
            raise OfflineOfficialMeshExperimentError("visible report count is invalid")
        if any(name not in POINT_NAMES for name in self.supported_point_names):
            raise OfflineOfficialMeshExperimentError("supported point identity is invalid")
        expected_digests = {
            "predeclared_noise_sha256": None,
            "evidence_sha256": None,
            "raw_render_sha256": None,
            "conversion_manifest_sha256": EXPECTED_CONVERSION_MANIFEST_SHA256,
            "asset_manifest_sha256": EXPECTED_ASSET_MANIFEST_SHA256,
            "runtime_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
            "scene_contract_sha256": EXPECTED_SCENE_CONTRACT_SHA256,
        }
        for field_name, expected in expected_digests.items():
            digest = getattr(self, field_name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or (expected is not None and digest != expected)
            ):
                raise OfflineOfficialMeshExperimentError(
                    f"geometry decision {field_name} is invalid"
                )
        if not isinstance(self.geometry_result, VisibilityGeometryResult):
            raise OfflineOfficialMeshExperimentError("geometry result type changed")
        if self.geometry_result.accepted != self.accepted:
            raise OfflineOfficialMeshExperimentError("geometry acceptance disagrees")
        if self.accepted:
            if self.accepted_estimate_world_m is None or self.accepted_error_p95_m is None:
                raise OfflineOfficialMeshExperimentError("accepted geometry lacks estimate")
            estimate = _immutable_array(self.accepted_estimate_world_m)
            error = float(self.accepted_error_p95_m)
            if estimate.shape != (3,) or not np.isfinite(estimate).all():
                raise OfflineOfficialMeshExperimentError("accepted estimate is invalid")
            if not math.isfinite(error) or error < 0.0:
                raise OfflineOfficialMeshExperimentError("accepted error is invalid")
        else:
            if self.accepted_estimate_world_m is not None or self.accepted_error_p95_m is not None:
                raise OfflineOfficialMeshExperimentError("rejected geometry carried estimate")
            estimate = None
            error = None
        object.__setattr__(self, "accepted_estimate_world_m", estimate)
        object.__setattr__(self, "accepted_error_p95_m", error)


def evaluate_rendered_geometry_gate(
    frame: RenderedGeometryGateFrame,
) -> GeometryGateDecision:
    """Evaluate a prebuilt observation without any current-frame truth input."""

    if not isinstance(frame, RenderedGeometryGateFrame):
        raise OfflineOfficialMeshExperimentError("rendered gate frame type changed")
    # The radius-six visibility rows are retained for diagnostics only.  They
    # are centered on the simulator's nominal anchor projection and therefore
    # must not select or weight oracle geometry inputs.  Exact 4/4 labcoat
    # depth at the independently noisy observed UV is the sole support gate.
    supported = np.asarray(frame.depth_valid, dtype=np.bool_)
    supported_indices = np.flatnonzero(supported)
    target_index = POINT_NAMES.index(frame.target_name)
    result = evaluate_visibility_geometry(
        frame.reference_anchor_camera_m[supported],
        frame.observed_anchor_camera_m[supported],
        frame.reference_anchor_camera_m[target_index],
        depth_m=frame.depth_mid_m[supported],
        mask_confidence=np.ones(int(np.count_nonzero(supported))),
        lateral_noise_std_m=np.full(
            int(np.count_nonzero(supported)), OBSERVATION_LATERAL_STD_M
        ),
        axial_noise_std_m=frame.axial_sigma_effective_m[supported],
        robust_delta_m=OBSERVATION_ROBUST_DELTA_M,
        decision_surface=DECISION_SURFACE,
    )
    world_from_cv = frame.world_from_cv_camera
    if result.accepted:
        estimated_target_camera = (
            result.transform[:3, :3] @ frame.reference_anchor_camera_m[target_index]
            + result.transform[:3, 3]
        )
        estimated_target_world = (
            world_from_cv[:3, :3] @ estimated_target_camera + world_from_cv[:3, 3]
        )
        error_p95: float | None = float(result.target_error_p95_m)
    else:
        estimated_target_world = None
        error_p95 = None
    return GeometryGateDecision(
        accepted=bool(result.accepted),
        accepted_estimate_world_m=estimated_target_world,
        accepted_error_p95_m=error_p95,
        visible_point_count_report_only=int(
            np.count_nonzero([row.visible_fraction > 0.0 for row in frame.visibility])
        ),
        supported_point_names=tuple(POINT_NAMES[index] for index in supported_indices),
        predeclared_noise_sha256=frame.predeclared_noise_sha256,
        evidence_sha256=frame.evidence_sha256,
        raw_render_sha256=frame.raw_render_sha256,
        conversion_manifest_sha256=frame.conversion_manifest_sha256,
        asset_manifest_sha256=frame.asset_manifest_sha256,
        runtime_manifest_sha256=frame.runtime_manifest_sha256,
        scene_contract_sha256=frame.scene_contract_sha256,
        geometry_result=result,
    )


def offline_frame_evidence_sha256(evidence: OfflineFrameEvidence) -> str:
    """Return the oracle event/evidence hash (not the raw-render hash)."""

    if not isinstance(evidence, OfflineFrameEvidence):
        raise OfflineOfficialMeshExperimentError("offline frame evidence type changed")
    return oracle_frame_evidence_sha256(evidence)


def rendered_gate_frame_from_evidence(
    evidence: OfflineFrameEvidence,
    *,
    reference_anchor_camera_m: np.ndarray,
    target_name: str,
    noise: PredeclaredObservationNoise,
) -> RenderedGeometryGateFrame:
    """Create one oracle-landmark-noise/labcoat-depth gate input."""

    if not isinstance(evidence, OfflineFrameEvidence):
        raise OfflineOfficialMeshExperimentError("offline frame evidence type changed")
    if not isinstance(noise, PredeclaredObservationNoise):
        raise OfflineOfficialMeshExperimentError("noise contract type changed")
    try:
        validate_oracle_frame_evidence(evidence)
    except OfflineSimulationError as exc:
        raise OfflineOfficialMeshExperimentError(
            f"oracle frame evidence failed validation: {exc}"
        ) from exc
    if scene_contract_sha256() != EXPECTED_SCENE_CONTRACT_SHA256:
        raise OfflineOfficialMeshExperimentError("scene contract identity changed")
    expected_identity = {
        "conversion_release_id": f"official_meshes-{EXPECTED_CONVERSION_MANIFEST_SHA256[:20]}",
        "conversion_manifest_sha256": EXPECTED_CONVERSION_MANIFEST_SHA256,
        "asset_release_id": f"official_assets-{EXPECTED_ASSET_MANIFEST_SHA256[:20]}",
        "asset_manifest_sha256": EXPECTED_ASSET_MANIFEST_SHA256,
        "runtime_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
    }
    for field_name, expected in expected_identity.items():
        if getattr(evidence, field_name) != expected:
            raise OfflineOfficialMeshExperimentError(
                f"offline evidence {field_name} changed"
            )
    labcoat_ids = [
        int(geom_id)
        for geom_id, row in evidence.geom_provenance.items()
        if row.get("kind") == "human"
        and row.get("identity") == LABCOAT_GEOM_IDENTITY
    ]
    if len(labcoat_ids) != 1:
        raise OfflineOfficialMeshExperimentError(
            "exact labcoat geometry identity is unavailable"
        )
    labcoat_geom_id = labcoat_ids[0]
    observed = np.asarray(evidence.anchor_camera_position_m, dtype=np.float64) + (
        noise.error_camera_m
    )
    try:
        observed_pixels = project_cv_points(observed)
    except Exception as exc:
        raise OfflineOfficialMeshExperimentError(
            "noisy observed anchor projection failed"
        ) from exc
    depth = np.asarray(evidence.products.depth_m)
    segmentation = np.asarray(evidence.products.segmentation)
    if segmentation.shape != (*depth.shape, 2):
        raise OfflineOfficialMeshExperimentError("rendered segmentation shape changed")
    intervals = np.full((len(POINT_NAMES), 2), np.nan, dtype=np.float64)
    midpoint = np.full(len(POINT_NAMES), np.nan, dtype=np.float64)
    effective_sigma = np.full(len(POINT_NAMES), np.nan, dtype=np.float64)
    valid = np.zeros(len(POINT_NAMES), dtype=np.bool_)
    height, width = depth.shape
    geom_object_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    for index, center in enumerate(observed_pixels):
        x0 = int(math.floor(float(center[0])))
        y0 = int(math.floor(float(center[1])))
        if x0 < 0 or y0 < 0 or x0 + 1 >= width or y0 + 1 >= height:
            continue
        block_depth = depth[y0 : y0 + 2, x0 : x0 + 2]
        block_segmentation = segmentation[y0 : y0 + 2, x0 : x0 + 2]
        exact_labcoat = (
            (block_segmentation[..., 0] == labcoat_geom_id)
            & (block_segmentation[..., 1] == geom_object_type)
        )
        if (
            not np.all(exact_labcoat)
            or not np.isfinite(block_depth).all()
            or np.any(block_depth <= 0.0)
        ):
            continue
        lower = float(np.min(block_depth))
        upper = float(np.max(block_depth))
        intervals[index] = (lower, upper)
        midpoint[index] = 0.5 * (lower + upper)
        effective_sigma[index] = float(
            noise.axial_noise_std_m[index] + 0.5 * (upper - lower)
        )
        valid[index] = True
    evidence_sha256 = offline_frame_evidence_sha256(evidence)
    raw_render_digest = oracle_raw_render_sha256(evidence)
    return RenderedGeometryGateFrame(
        point_names=evidence.anchor_names,
        reference_anchor_camera_m=reference_anchor_camera_m,
        observed_anchor_camera_m=observed,
        observed_projected_pixels=observed_pixels,
        depth_interval_m=intervals,
        depth_mid_m=midpoint,
        axial_sigma_effective_m=effective_sigma,
        depth_valid=valid,
        visibility=evidence.visibility,
        target_name=target_name,
        world_from_cv_camera=evidence.world_from_cv_camera,
        evidence_sha256=evidence_sha256,
        raw_render_sha256=raw_render_digest,
        predeclared_noise_sha256=noise.contract_sha256,
        conversion_release_id=evidence.conversion_release_id,
        conversion_manifest_sha256=evidence.conversion_manifest_sha256,
        asset_release_id=evidence.asset_release_id,
        asset_manifest_sha256=evidence.asset_manifest_sha256,
        runtime_manifest_sha256=evidence.runtime_manifest_sha256,
        scene_contract_sha256=EXPECTED_SCENE_CONTRACT_SHA256,
        depth_footprint=DEPTH_FOOTPRINT,
        evidence_source=(
            "oracle_landmark_noise_simulation_exact_labcoat_segmentation_and_metric_depth"
        ),
    )


def rendered_gate_decision_from_evidence(
    evidence: OfflineFrameEvidence,
    *,
    reference_anchor_camera_m: np.ndarray,
    target_name: str,
    noise: PredeclaredObservationNoise,
) -> GeometryGateDecision:
    """Evaluate the oracle landmark-noise method upper bound for one frame."""

    frame = rendered_gate_frame_from_evidence(
        evidence,
        reference_anchor_camera_m=reference_anchor_camera_m,
        target_name=target_name,
        noise=noise,
    )
    return evaluate_rendered_geometry_gate(frame)


def _decision_surface_value() -> dict[str, float]:
    return {
        name: float(getattr(DECISION_SURFACE, name))
        for name in DECISION_SURFACE.__dataclass_fields__
    }


def _input_contract_value(
    motion_plan_sha256: str, *, allow_test_only: bool
) -> dict[str, object]:
    if (
        not isinstance(motion_plan_sha256, str)
        or len(motion_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in motion_plan_sha256)
    ):
        raise OfflineOfficialMeshExperimentError("motion plan SHA-256 is invalid")
    sweep_contract = observation_sweep_contract()
    return {
        "schema": "wam_oracle_landmark_noise_simulation_v1",
        "status": ORACLE_SIMULATION_STATUS,
        "scope": ORACLE_SIMULATION_STATUS,
        "production_activation": False,
        "model_validation": False,
        "isaac_equivalence": False,
        "provider_contract": (
            "test_only_synthetic_decision"
            if allow_test_only
            else "scene_owned_oracle_evidence"
        ),
        "test_only_synthetic_provider": allow_test_only,
        "landmark_source": "oracle_anchor_plus_predeclared_noise",
        "ground_truth_role": "method_upper_bound_and_policy_scoring_only",
        "metric_eligibility_contract": (
            "trajectory_index_59_geometry_accepted",
            "at_least_one_contact_hold_geometry_rejected",
            "contact_hold_exit_geometry_rejected",
            "first_post_rejection_acceptance_occurs_in_depart",
        ),
        "point_names": POINT_NAMES,
        "wrist_indices": (0, 1),
        "occlusion_durations_s": OCCLUSION_DURATIONS_S,
        "required_strata_count": REQUIRED_STRATA_COUNT,
        "frame_period_s": FRAME_PERIOD_S,
        "event_schedule": (
            "official_trajectory_indices_0_through_60_patient_identity",
            "contact_hold_2_5_10_20_frames_patient_rigid_translation",
            "official_trajectory_indices_61_through_120_shifted_by_hold",
        ),
        "base_trajectory_indices": SCORING_TRAJECTORY_INDICES,
        "precontact_trajectory_indices": PRECONTACT_TRAJECTORY_INDICES,
        "depart_trajectory_indices": DEPART_TRAJECTORY_INDICES,
        "occlusion_hold_frames": dict(OCCLUSION_HOLD_FRAMES),
        "patient_motion_model": "seeded_rigid_translation_only",
        "patient_translation_speed_m_s": PATIENT_TRANSLATION_SPEED_M_S,
        "patient_motion_bound_m_s": PATIENT_MOTION_BOUND_M_S,
        "observation_seed_root": OBSERVATION_SEED_ROOT,
        "patient_motion_seed_root": PATIENT_MOTION_SEED_ROOT,
        "ik_seed_rng": IK_SEED_RNG,
        "ik_seed_le_f64_sha256": EXPECTED_IK_SEED_LE_F64_SHA256,
        "anchor_contract_sha256": EXPECTED_ANCHOR_CONTRACT_SHA256,
        "anchor_quantization_tolerance_m": ANCHOR_QUANTIZATION_TOLERANCE_M,
        "robot_urdf_sha256": OFFICIAL_CLEAN_URDF_SHA256,
        "asset_manifest_sha256": EXPECTED_ASSET_MANIFEST_SHA256,
        "conversion_manifest_sha256": EXPECTED_CONVERSION_MANIFEST_SHA256,
        "runtime_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
        "scene_contract_sha256": EXPECTED_SCENE_CONTRACT_SHA256,
        "robot_world_from_root": ROBOT_WORLD_FROM_ROOT.tolist(),
        "motion_plan_sha256": motion_plan_sha256,
        "contact_normal_offset_m": CONTACT_NORMAL_OFFSET_M,
        "approach_normal_offset_m": APPROACH_NORMAL_OFFSET_M,
        "ik_tolerance_m": IK_TOLERANCE_M,
        "ik_max_iterations": IK_MAX_ITERATIONS,
        "ik_min_orientation_separation_rad": IK_MIN_ORIENTATION_SEPARATION_RAD,
        "trajectory_frames_per_segment": TRAJECTORY_FRAMES_PER_SEGMENT,
        "trajectory_seconds_per_segment": TRAJECTORY_SECONDS_PER_SEGMENT,
        "trajectory_acceleration_limit_rad_s2": TRAJECTORY_ACCELERATION_LIMIT_RAD_S2,
        "observation_lateral_std_m": OBSERVATION_LATERAL_STD_M,
        "observation_axial_std_m": OBSERVATION_AXIAL_STD_M,
        "observation_robust_delta_m": OBSERVATION_ROBUST_DELTA_M,
        "depth_footprint": DEPTH_FOOTPRINT,
        "observation_sweeps": {
            "declared_sweep_count": sweep_contract["declared_sweep_count"],
            "executed_sweep_count": sweep_contract["executed_sweep_count"],
            "executed_sweep_indices": sweep_contract["executed_sweep_indices"],
            "bias_model_status": sweep_contract["bias_model_status"],
            "delay_model_status": sweep_contract["delay_model_status"],
            "sweeps": tuple(dict(row) for row in sweep_contract["sweeps"]),
        },
        "decision_surface": _decision_surface_value(),
        "policy_names": POLICY_NAMES,
        "visible_point_count_role": "report_only",
    }


def build_experiment_input_contract(
    *, motion_plan_sha256: str, allow_test_only: bool = False
) -> tuple[Mapping[str, object], str]:
    """Return the complete immutable input contract and its canonical hash."""

    if type(allow_test_only) is not bool:
        raise OfflineOfficialMeshExperimentError("allow_test_only must be a bool")
    value = _input_contract_value(
        motion_plan_sha256, allow_test_only=allow_test_only
    )
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    immutable = dict(value)
    immutable["decision_surface"] = MappingProxyType(
        dict(value["decision_surface"])
    )
    immutable["observation_sweeps"] = MappingProxyType(
        dict(value["observation_sweeps"])
    )
    return MappingProxyType(immutable), digest


@dataclass(frozen=True)
class SparseEventSpec:
    """One 20 Hz rendered event on the validated motion/hold timeline."""

    scenario_id: str
    target_name: str
    wrist_index: int
    occlusion_duration_s: float
    frame_index: int
    event_name: str
    phase: str
    trajectory_index: int
    hold_frame_index: int | None
    timestamp_s: float
    left_joint_positions_rad: np.ndarray
    world_from_patient: np.ndarray
    target_ground_truth_world_m: np.ndarray
    patient_motion_direction_world: np.ndarray
    patient_motion_bound_m_s: float
    observation_seed: int
    patient_motion_seed: int
    bias_seed: int
    delay_seed: int
    sweep_index: int
    bias_applied: bool
    delay_applied: bool
    noise: PredeclaredObservationNoise
    input_contract_sha256: str
    motion_plan_sha256: str
    render_reuse_key_sha256: str

    def __post_init__(self) -> None:
        if self.target_name not in POINT_NAMES or self.wrist_index not in (0, 1):
            raise OfflineOfficialMeshExperimentError("sparse event identity is invalid")
        if self.occlusion_duration_s not in OCCLUSION_DURATIONS_S:
            raise OfflineOfficialMeshExperimentError("sparse event duration is invalid")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise OfflineOfficialMeshExperimentError("event frame index is invalid")
        hold_count = OCCLUSION_HOLD_FRAMES[self.occlusion_duration_s]
        if self.phase == "base_precontact":
            if (
                self.trajectory_index not in PRECONTACT_TRAJECTORY_INDICES
                or self.hold_frame_index is not None
                or self.event_name != f"base_{self.trajectory_index:03d}"
            ):
                raise OfflineOfficialMeshExperimentError("precontact event identity changed")
            expected_time = self.trajectory_index * FRAME_PERIOD_S
            expected_patient_elapsed = 0.0
        elif self.phase == "contact_hold":
            if (
                self.trajectory_index != 60
                or type(self.hold_frame_index) is not int
                or not 1 <= self.hold_frame_index <= hold_count
                or self.event_name != f"hold_{self.hold_frame_index:03d}"
            ):
                raise OfflineOfficialMeshExperimentError("hold event identity changed")
            expected_time = 3.0 + self.hold_frame_index * FRAME_PERIOD_S
            expected_patient_elapsed = self.hold_frame_index * FRAME_PERIOD_S
        elif self.phase == "base_depart":
            if (
                self.trajectory_index not in DEPART_TRAJECTORY_INDICES
                or self.hold_frame_index is not None
                or self.event_name != f"depart_{self.trajectory_index:03d}"
            ):
                raise OfflineOfficialMeshExperimentError("depart event identity changed")
            expected_time = (
                self.trajectory_index * FRAME_PERIOD_S + self.occlusion_duration_s
            )
            expected_patient_elapsed = self.occlusion_duration_s
        else:
            raise OfflineOfficialMeshExperimentError("event phase is invalid")
        if not math.isclose(float(self.timestamp_s), expected_time, abs_tol=1.0e-12):
            raise OfflineOfficialMeshExperimentError("20 Hz event timestamp changed")
        joints = _immutable_array(self.left_joint_positions_rad)
        patient = _immutable_rigid_transform(self.world_from_patient, "world_from_patient")
        truth = _immutable_array(self.target_ground_truth_world_m)
        direction = _immutable_array(self.patient_motion_direction_world)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise OfflineOfficialMeshExperimentError("sparse event joint pose is invalid")
        if truth.shape != (3,) or not np.isfinite(truth).all():
            raise OfflineOfficialMeshExperimentError("sparse event truth is invalid")
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise OfflineOfficialMeshExperimentError("patient direction is invalid")
        if not math.isclose(float(np.linalg.norm(direction)), 1.0, abs_tol=1.0e-12):
            raise OfflineOfficialMeshExperimentError("patient direction is not unit length")
        if float(self.patient_motion_bound_m_s) != PATIENT_MOTION_BOUND_M_S:
            raise OfflineOfficialMeshExperimentError("patient motion bound changed")
        if any(type(seed) is not int or seed < 0 for seed in (
            self.observation_seed,
            self.patient_motion_seed,
            self.bias_seed,
            self.delay_seed,
        )):
            raise OfflineOfficialMeshExperimentError("sparse event seed is invalid")
        if self.sweep_index != 0 or self.sweep_index not in EXECUTED_SWEEP_INDICES:
            raise OfflineOfficialMeshExperimentError("undeclared sweep was executed")
        if type(self.bias_applied) is not bool or type(self.delay_applied) is not bool:
            raise OfflineOfficialMeshExperimentError("bias/delay execution flags are invalid")
        if self.bias_applied or self.delay_applied:
            raise OfflineOfficialMeshExperimentError(
                "unimplemented bias/delay model was reported as applied"
            )
        if not isinstance(self.noise, PredeclaredObservationNoise):
            raise OfflineOfficialMeshExperimentError("sparse event noise is invalid")
        if self.noise.seed != self.observation_seed:
            raise OfflineOfficialMeshExperimentError("event noise seed disagrees")
        for label, digest in (
            ("input contract", self.input_contract_sha256),
            ("motion plan", self.motion_plan_sha256),
            ("render reuse key", self.render_reuse_key_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise OfflineOfficialMeshExperimentError(f"{label} hash is invalid")
        expected_translation = (
            direction * PATIENT_TRANSLATION_SPEED_M_S * expected_patient_elapsed
        )
        if not np.allclose(
            patient[:3, 3], expected_translation, atol=1.0e-12, rtol=0.0
        ):
            raise OfflineOfficialMeshExperimentError("patient translation schedule changed")
        object.__setattr__(self, "left_joint_positions_rad", joints)
        object.__setattr__(self, "world_from_patient", patient)
        object.__setattr__(self, "target_ground_truth_world_m", truth)
        object.__setattr__(self, "patient_motion_direction_world", direction)


@dataclass(frozen=True)
class OracleRenderRequest:
    """Minimal request exposed to the scene-owned evidence provider.

    No explicit scored-target identity/vector or oracle noise is passed in the
    request.  This is a non-security oracle-rendering boundary: the provider
    necessarily renders the whole scene, and returned evidence contains every
    official anchor projection.
    """

    frame_index: int
    timestamp_s: float
    left_joint_positions_rad: np.ndarray
    world_from_patient: np.ndarray
    render_reuse_key_sha256: str

    @classmethod
    def from_event(cls, event: SparseEventSpec) -> "OracleRenderRequest":
        if not isinstance(event, SparseEventSpec):
            raise OfflineOfficialMeshExperimentError("oracle render event type changed")
        return cls(
            frame_index=event.frame_index,
            timestamp_s=event.timestamp_s,
            left_joint_positions_rad=event.left_joint_positions_rad,
            world_from_patient=event.world_from_patient,
            render_reuse_key_sha256=event.render_reuse_key_sha256,
        )

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise OfflineOfficialMeshExperimentError(
                "oracle render request frame index changed"
            )
        timestamp = float(self.timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise OfflineOfficialMeshExperimentError(
                "oracle render request timestamp changed"
            )
        joints = _immutable_array(self.left_joint_positions_rad)
        patient = _immutable_rigid_transform(
            self.world_from_patient, "oracle request world_from_patient"
        )
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise OfflineOfficialMeshExperimentError(
                "oracle render request joint pose changed"
            )
        digest = self.render_reuse_key_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise OfflineOfficialMeshExperimentError(
                "oracle render reuse hash is invalid"
            )
        object.__setattr__(self, "left_joint_positions_rad", joints)
        object.__setattr__(self, "world_from_patient", patient)
        object.__setattr__(self, "timestamp_s", timestamp)


@dataclass(frozen=True)
class SparseStratumResult:
    """One retained required stratum, including explicit failed rows."""

    scenario_id: str
    target_name: str
    wrist_index: int
    occlusion_duration_s: float
    status: str
    failure_reason: str | None
    geometry_eligible: bool
    geometry_failure_reason: str | None
    events: tuple[SparseEventSpec, ...]
    gates: tuple[GeometryGateDecision, ...]
    observations: tuple[ObservationFrame, ...]
    ground_truth_frames: tuple[GroundTruthFrame, ...]
    required_stratum: RequiredStratum

    def __post_init__(self) -> None:
        if self.status not in EXECUTION_STATUSES:
            raise OfflineOfficialMeshExperimentError("stratum status is invalid")
        if type(self.geometry_eligible) is not bool:
            raise OfflineOfficialMeshExperimentError("geometry eligibility is invalid")
        if self.status == "COMPLETE":
            expected_count = 121 + OCCLUSION_HOLD_FRAMES[self.occlusion_duration_s]
            if self.failure_reason is not None or not all(
                len(value) == expected_count
                for value in (
                    self.events,
                    self.gates,
                    self.observations,
                    self.ground_truth_frames,
                )
            ):
                raise OfflineOfficialMeshExperimentError("complete stratum is incomplete")
            if self.geometry_eligible != (self.geometry_failure_reason is None):
                raise OfflineOfficialMeshExperimentError(
                    "geometry eligibility reason disagrees"
                )
        else:
            if not isinstance(self.failure_reason, str) or not self.failure_reason:
                raise OfflineOfficialMeshExperimentError("failed stratum lacks reason")
            if self.observations or self.ground_truth_frames:
                raise OfflineOfficialMeshExperimentError("failed stratum carried scoring rows")
            if self.geometry_eligible or self.geometry_failure_reason is not None:
                raise OfflineOfficialMeshExperimentError(
                    "failed motion stratum carried geometry eligibility"
                )
        if self.required_stratum.execution_status != self.status:
            raise OfflineOfficialMeshExperimentError("required stratum status disagrees")
        if self.required_stratum.metric_eligible != self.geometry_eligible:
            raise OfflineOfficialMeshExperimentError(
                "required stratum metric eligibility disagrees"
            )


@dataclass(frozen=True)
class InMemorySparseExperiment:
    """All predeclared inputs, observations, policy traces, and scores in memory."""

    input_contract: Mapping[str, object]
    input_contract_sha256: str
    target_plans: tuple[TargetMotionPlan, ...]
    strata: tuple[SparseStratumResult, ...]
    observations: tuple[ObservationFrame, ...]
    ground_truth_frames: tuple[GroundTruthFrame, ...]
    required_strata: tuple[RequiredStratum, ...]
    policy_outputs: Mapping[str, tuple[PolicyFrameOutput, ...]]
    metrics: Mapping[str, Mapping[str, object]]


def _stratum_scenario_id(
    target_name: str, wrist_index: int, duration_s: float
) -> str:
    milliseconds = int(round(duration_s * 1000.0))
    return f"{target_name}__w{wrist_index}__occ{milliseconds:04d}ms"


def _patient_direction(seed: int) -> np.ndarray:
    direction = np.random.default_rng(seed).normal(size=3)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise OfflineOfficialMeshExperimentError("patient motion direction failed")
    return direction / norm


def _event_contract_row(
    event: SparseEventSpec, gate: GeometryGateDecision
) -> dict[str, object]:
    return {
        "frame_index": event.frame_index,
        "event_name": event.event_name,
        "phase": event.phase,
        "trajectory_index": event.trajectory_index,
        "hold_frame_index": event.hold_frame_index,
        "timestamp_s": event.timestamp_s,
        "left_joint_positions_rad": event.left_joint_positions_rad.tolist(),
        "world_from_patient": event.world_from_patient.tolist(),
        "observation_seed": event.observation_seed,
        "patient_motion_seed": event.patient_motion_seed,
        "bias_seed": event.bias_seed,
        "delay_seed": event.delay_seed,
        "sweep_index": event.sweep_index,
        "bias_applied": event.bias_applied,
        "delay_applied": event.delay_applied,
        "noise_contract_sha256": event.noise.contract_sha256,
        "motion_plan_sha256": event.motion_plan_sha256,
        "render_reuse_key_sha256": event.render_reuse_key_sha256,
        "evidence_sha256": gate.evidence_sha256,
        "raw_render_sha256": gate.raw_render_sha256,
        "conversion_manifest_sha256": gate.conversion_manifest_sha256,
        "asset_manifest_sha256": gate.asset_manifest_sha256,
        "runtime_manifest_sha256": gate.runtime_manifest_sha256,
        "scene_contract_sha256": gate.scene_contract_sha256,
        "geometry_accepted": gate.accepted,
        "accepted_estimate_world_m": (
            None
            if gate.accepted_estimate_world_m is None
            else gate.accepted_estimate_world_m.tolist()
        ),
        "accepted_error_p95_m": gate.accepted_error_p95_m,
        "visible_point_count_report_only": gate.visible_point_count_report_only,
        "supported_point_names": gate.supported_point_names,
        "target_ground_truth_world_m": event.target_ground_truth_world_m.tolist(),
    }


def _stream_sha256(
    *,
    input_contract_sha256: str,
    scenario_id: str,
    target_name: str,
    wrist_index: int,
    duration_s: float,
    observation_seed: int,
    patient_motion_seed: int,
    patient_direction: np.ndarray,
    status: str,
    geometry_eligible: bool,
    geometry_failure_reason: str | None,
    events: Sequence[SparseEventSpec],
    gates: Sequence[GeometryGateDecision],
) -> str:
    value = {
        "schema": "wam_oracle_landmark_noise_observation_stream_v1",
        "input_contract_sha256": input_contract_sha256,
        "scenario_id": scenario_id,
        "target_name": target_name,
        "wrist_index": wrist_index,
        "occlusion_duration_s": duration_s,
        "observation_seed": observation_seed,
        "patient_motion_seed": patient_motion_seed,
        "patient_motion_direction_world": np.asarray(patient_direction).tolist(),
        "execution_status": status,
        "geometry_eligible": geometry_eligible,
        "geometry_failure_reason": geometry_failure_reason,
        "events": [
            _event_contract_row(event, gate)
            for event, gate in zip(events, gates, strict=True)
        ],
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _render_reuse_key_sha256(
    *,
    motion_plan_sha256: str,
    target_name: str,
    wrist_index: int,
    left_joint_positions_rad: np.ndarray,
    world_from_patient: np.ndarray,
) -> str:
    """Hash only geometry inputs so rendered products may be reused by sweeps."""

    value = {
        "schema": "wam_oracle_raw_render_reuse_key_v1",
        "motion_plan_sha256": motion_plan_sha256,
        "target_name": target_name,
        "wrist_index": wrist_index,
        "left_joint_positions_rad": np.asarray(left_joint_positions_rad).tolist(),
        "world_from_patient": np.asarray(world_from_patient).tolist(),
        "robot_urdf_sha256": OFFICIAL_CLEAN_URDF_SHA256,
        "asset_manifest_sha256": EXPECTED_ASSET_MANIFEST_SHA256,
        "conversion_manifest_sha256": EXPECTED_CONVERSION_MANIFEST_SHA256,
        "runtime_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
        "scene_contract_sha256": EXPECTED_SCENE_CONTRACT_SHA256,
        "robot_world_from_root": ROBOT_WORLD_FROM_ROOT.tolist(),
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _decision_from_gate_provider(
    provider_value: object,
    *,
    event: SparseEventSpec,
    official_anchor_positions: np.ndarray,
    allow_test_only: bool,
) -> GeometryGateDecision:
    if isinstance(provider_value, OfflineFrameEvidence):
        evidence = provider_value
        try:
            validate_oracle_frame_evidence(evidence)
        except OfflineSimulationError as exc:
            raise OfflineOfficialMeshExperimentError(
                f"oracle provider evidence is not scene-owned: {exc}"
            ) from exc
        if not math.isclose(evidence.timestamp_s, event.timestamp_s, abs_tol=1.0e-12):
            raise OfflineOfficialMeshExperimentError(
                "rendered evidence timestamp differs from the event"
            )
        if not np.array_equal(
            evidence.left_joint_positions, event.left_joint_positions_rad
        ):
            raise OfflineOfficialMeshExperimentError(
                "rendered evidence joint pose differs from the event"
            )
        if not np.allclose(
            evidence.world_from_patient,
            event.world_from_patient,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise OfflineOfficialMeshExperimentError(
                "rendered evidence patient pose differs from the event"
            )
        if tuple(evidence.anchor_names) != POINT_NAMES:
            raise OfflineOfficialMeshExperimentError(
                "rendered evidence anchor identity differs from the event"
            )
        validate_anchor_quantization(
            official_anchor_positions, evidence.anchor_local_position_m
        )
        world_from_cv = np.asarray(evidence.world_from_cv_camera, dtype=np.float64)
        reference_world = np.asarray(evidence.anchor_local_position_m, dtype=np.float64)
        reference_camera = (
            world_from_cv[:3, :3].T
            @ (reference_world - world_from_cv[:3, 3]).T
        ).T
        decision = rendered_gate_decision_from_evidence(
            evidence,
            reference_anchor_camera_m=reference_camera,
            target_name=event.target_name,
            noise=event.noise,
        )
    elif allow_test_only and isinstance(provider_value, GeometryGateDecision):
        decision = provider_value
    else:
        raise OfflineOfficialMeshExperimentError(
            "oracle provider must return scene-owned OfflineFrameEvidence; "
            "synthetic decisions require allow_test_only=True"
        )
    if decision.predeclared_noise_sha256 != event.noise.contract_sha256:
        raise OfflineOfficialMeshExperimentError(
            "gate decision used a different noise realization"
        )
    return decision


def build_in_memory_sparse_experiment(
    target_plans: Sequence[TargetMotionPlan],
    anchors: Sequence[OfficialAnchor],
    *,
    urdf_raw: bytes,
    gate_provider: Callable[[SparseEventSpec | OracleRenderRequest], object],
    allow_test_only: bool = False,
) -> InMemorySparseExperiment:
    """Build the full 20 Hz 80-stratum matrix without writing files."""

    if type(allow_test_only) is not bool:
        raise OfflineOfficialMeshExperimentError("allow_test_only must be a bool")
    plans = tuple(target_plans)
    anchor_rows = tuple(anchors)
    if not callable(gate_provider):
        raise OfflineOfficialMeshExperimentError("gate_provider must be callable")
    motion_plan_sha256 = revalidate_official_motion_plans(
        urdf_raw, anchor_rows, plans
    )
    input_contract, input_contract_sha256 = build_experiment_input_contract(
        motion_plan_sha256=motion_plan_sha256,
        allow_test_only=allow_test_only,
    )
    strata: list[SparseStratumResult] = []
    flat_observations: list[ObservationFrame] = []
    flat_truth: list[GroundTruthFrame] = []
    required_rows: list[RequiredStratum] = []
    raw_render_by_reuse_key: dict[str, str] = {}
    official_anchor_positions = np.stack(
        [row.position_m for row in anchor_rows]
    )

    for target_index, (plan, anchor) in enumerate(zip(plans, anchor_rows, strict=True)):
        for wrist_index in (0, 1):
            for duration_index, duration_s in enumerate(OCCLUSION_DURATIONS_S):
                stratum_index = target_index * 8 + wrist_index * 4 + duration_index
                scenario_id = _stratum_scenario_id(
                    plan.target_name, wrist_index, duration_s
                )
                base_observation_seed = (
                    NOISE_SWEEP_SEED_ROOTS[0] + 1000 * stratum_index
                )
                bias_seed = BIAS_SWEEP_SEED_ROOTS[0] + stratum_index
                delay_seed = DELAY_SWEEP_SEED_ROOTS[0] + stratum_index
                # One target/wrist motion direction is shared across durations;
                # the shorter holds are exact prefixes of the longer holds.
                patient_motion_seed = (
                    PATIENT_MOTION_SEED_ROOT + target_index * 2 + wrist_index
                )
                direction = _patient_direction(patient_motion_seed)
                events: list[SparseEventSpec] = []
                gates: list[GeometryGateDecision] = []
                status = plan.status
                failure_reason = plan.failure_reason
                geometry_eligible = False
                geometry_failure_reason: str | None = None

                if status == "COMPLETE":
                    branch = next(
                        row for row in plan.branches if row.wrist_index == wrist_index
                    )
                    timeline: list[
                        tuple[str, int, int | None, float, np.ndarray, float]
                    ] = []
                    for trajectory_index in PRECONTACT_TRAJECTORY_INDICES:
                        timeline.append(
                            (
                                "base_precontact",
                                trajectory_index,
                                None,
                                trajectory_index * FRAME_PERIOD_S,
                                branch.trajectory_joint_positions[trajectory_index],
                                0.0,
                            )
                        )
                    for hold_index in range(
                        1, OCCLUSION_HOLD_FRAMES[duration_s] + 1
                    ):
                        timeline.append(
                            (
                                "contact_hold",
                                60,
                                hold_index,
                                3.0 + hold_index * FRAME_PERIOD_S,
                                branch.trajectory_joint_positions[60],
                                hold_index * FRAME_PERIOD_S,
                            )
                        )
                    for trajectory_index in DEPART_TRAJECTORY_INDICES:
                        timeline.append(
                            (
                                "base_depart",
                                trajectory_index,
                                None,
                                trajectory_index * FRAME_PERIOD_S + duration_s,
                                branch.trajectory_joint_positions[trajectory_index],
                                duration_s,
                            )
                        )
                    for frame_index, (
                        phase,
                        trajectory_index,
                        hold_frame_index,
                        timestamp_s,
                        joints,
                        patient_elapsed_s,
                    ) in enumerate(timeline):
                        if phase == "base_precontact":
                            event_name = f"base_{trajectory_index:03d}"
                        elif phase == "contact_hold":
                            assert hold_frame_index is not None
                            event_name = f"hold_{hold_frame_index:03d}"
                        else:
                            event_name = f"depart_{trajectory_index:03d}"
                        world_from_patient = np.eye(4)
                        world_from_patient[:3, 3] = (
                            direction
                            * PATIENT_TRANSLATION_SPEED_M_S
                            * patient_elapsed_s
                        )
                        target_truth = (
                            world_from_patient[:3, :3] @ anchor.position_m
                            + world_from_patient[:3, 3]
                        )
                        observation_seed = base_observation_seed + frame_index
                        render_reuse_key = _render_reuse_key_sha256(
                            motion_plan_sha256=motion_plan_sha256,
                            target_name=plan.target_name,
                            wrist_index=wrist_index,
                            left_joint_positions_rad=joints,
                            world_from_patient=world_from_patient,
                        )
                        event = SparseEventSpec(
                            scenario_id=scenario_id,
                            target_name=plan.target_name,
                            wrist_index=wrist_index,
                            occlusion_duration_s=duration_s,
                            frame_index=frame_index,
                            event_name=event_name,
                            phase=phase,
                            trajectory_index=trajectory_index,
                            hold_frame_index=hold_frame_index,
                            timestamp_s=timestamp_s,
                            left_joint_positions_rad=joints,
                            world_from_patient=world_from_patient,
                            target_ground_truth_world_m=target_truth,
                            patient_motion_direction_world=direction,
                            patient_motion_bound_m_s=PATIENT_MOTION_BOUND_M_S,
                            observation_seed=observation_seed,
                            patient_motion_seed=patient_motion_seed,
                            bias_seed=bias_seed,
                            delay_seed=delay_seed,
                            sweep_index=0,
                            bias_applied=False,
                            delay_applied=False,
                            noise=make_predeclared_observation_noise(observation_seed),
                            input_contract_sha256=input_contract_sha256,
                            motion_plan_sha256=motion_plan_sha256,
                            render_reuse_key_sha256=render_reuse_key,
                        )
                        provider_request: SparseEventSpec | OracleRenderRequest = (
                            event
                            if allow_test_only
                            else OracleRenderRequest.from_event(event)
                        )
                        gate = _decision_from_gate_provider(
                            gate_provider(provider_request),
                            event=event,
                            official_anchor_positions=official_anchor_positions,
                            allow_test_only=allow_test_only,
                        )
                        prior_raw_render = raw_render_by_reuse_key.setdefault(
                            event.render_reuse_key_sha256,
                            gate.raw_render_sha256,
                        )
                        if prior_raw_render != gate.raw_render_sha256:
                            raise OfflineOfficialMeshExperimentError(
                                "render reuse key resolved to a different raw render SHA-256"
                            )
                        events.append(event)
                        gates.append(gate)
                    baseline_acquired = any(
                        gate.accepted and event.trajectory_index == 59
                        for event, gate in zip(events, gates, strict=True)
                        if event.phase == "base_precontact"
                    )
                    hold_pairs = tuple(
                        (event, gate)
                        for event, gate in zip(events, gates, strict=True)
                        if event.phase == "contact_hold"
                    )
                    hold_rejected = any(
                        not gate.accepted for _event, gate in hold_pairs
                    )
                    # A prior rejection followed by acceptance inside the hold
                    # has already reacquired before depart.  Do not count a
                    # later accepted depart frame as a second acquisition event.
                    hold_exit_rejected = bool(hold_pairs) and not hold_pairs[-1][1].accepted
                    depart_reacquired = hold_exit_rejected and any(
                        gate.accepted
                        for event, gate in zip(events, gates, strict=True)
                        if event.phase == "base_depart"
                    )
                    geometry_eligible = (
                        baseline_acquired and hold_rejected and depart_reacquired
                    )
                    reasons: list[str] = []
                    if not baseline_acquired:
                        reasons.append(
                            "no_immediate_precontact_geometry_baseline"
                        )
                    if not hold_rejected:
                        reasons.append("no_hold_geometry_rejection")
                    if hold_rejected and not depart_reacquired:
                        reasons.append("no_depart_geometry_reacquisition")
                    geometry_failure_reason = "+".join(reasons) or None

                stream_sha256 = _stream_sha256(
                    input_contract_sha256=input_contract_sha256,
                    scenario_id=scenario_id,
                    target_name=plan.target_name,
                    wrist_index=wrist_index,
                    duration_s=duration_s,
                    observation_seed=base_observation_seed,
                    patient_motion_seed=patient_motion_seed,
                    patient_direction=direction,
                    status=status,
                    geometry_eligible=geometry_eligible,
                    geometry_failure_reason=geometry_failure_reason,
                    events=events,
                    gates=gates,
                )
                observations: list[ObservationFrame] = []
                truths: list[GroundTruthFrame] = []
                if status == "COMPLETE":
                    for event, gate in zip(events, gates, strict=True):
                        observations.append(
                            ObservationFrame(
                                scenario_id=scenario_id,
                                target_name=plan.target_name,
                                wrist_index=wrist_index,
                                frame_index=event.frame_index,
                                timestamp_s=event.timestamp_s,
                                geometry_accepted=gate.accepted,
                                accepted_estimate_world_m=gate.accepted_estimate_world_m,
                                accepted_error_p95_m=gate.accepted_error_p95_m,
                                patient_motion_bound_m_s=PATIENT_MOTION_BOUND_M_S,
                                visible_point_count_report_only=(
                                    gate.visible_point_count_report_only
                                ),
                                observation_stream_sha256=stream_sha256,
                            )
                        )
                        truths.append(
                            GroundTruthFrame(
                                scenario_id=scenario_id,
                                target_name=plan.target_name,
                                wrist_index=wrist_index,
                                frame_index=event.frame_index,
                                timestamp_s=event.timestamp_s,
                                observation_stream_sha256=stream_sha256,
                                target_world_m=event.target_ground_truth_world_m,
                            )
                        )
                required = RequiredStratum(
                    scenario_id=scenario_id,
                    target_name=plan.target_name,
                    wrist_index=wrist_index,
                    observation_stream_sha256=stream_sha256,
                    observation_seed=base_observation_seed,
                    patient_motion_seed=patient_motion_seed,
                    execution_status=status,
                    declared_frame_count=len(events) if status == "COMPLETE" else 0,
                    metric_eligible=geometry_eligible,
                    metric_ineligibility_reason=geometry_failure_reason,
                )
                required_rows.append(required)
                flat_observations.extend(observations)
                flat_truth.extend(truths)
                strata.append(
                    SparseStratumResult(
                        scenario_id=scenario_id,
                        target_name=plan.target_name,
                        wrist_index=wrist_index,
                        occlusion_duration_s=duration_s,
                        status=status,
                        failure_reason=failure_reason,
                        geometry_eligible=geometry_eligible,
                        geometry_failure_reason=geometry_failure_reason,
                        events=tuple(events),
                        gates=tuple(gates),
                        observations=tuple(observations),
                        ground_truth_frames=tuple(truths),
                        required_stratum=required,
                    )
                )
    if len(strata) != REQUIRED_STRATA_COUNT or len(required_rows) != REQUIRED_STRATA_COUNT:
        raise OfflineOfficialMeshExperimentError("required 80-stratum matrix changed")
    policy_outputs = evaluate_occlusion_policies(tuple(flat_observations))
    raw_metrics = score_occlusion_policy_metrics(
        policy_outputs,
        tuple(flat_truth),
        required_strata=tuple(required_rows),
    )
    metrics: dict[str, Mapping[str, object]] = {}
    for policy_name in POLICY_NAMES:
        entry = dict(raw_metrics[policy_name])
        metrics[policy_name] = MappingProxyType(entry)
    return InMemorySparseExperiment(
        input_contract=input_contract,
        input_contract_sha256=input_contract_sha256,
        target_plans=plans,
        strata=tuple(strata),
        observations=tuple(flat_observations),
        ground_truth_frames=tuple(flat_truth),
        required_strata=tuple(required_rows),
        policy_outputs=MappingProxyType(
            {name: tuple(policy_outputs[name]) for name in POLICY_NAMES}
        ),
        metrics=MappingProxyType(metrics),
    )


__all__ = [
    "ANCHOR_QUANTIZATION_TOLERANCE_M",
    "APPROACH_NORMAL_OFFSET_M",
    "BIAS_SWEEP_SEED_ROOTS",
    "CONTACT_NORMAL_OFFSET_M",
    "DECISION_SURFACE",
    "EXECUTION_STATUSES",
    "EXPECTED_ANCHOR_CONTRACT_SHA256",
    "EXPECTED_IK_SEED_LE_F64_SHA256",
    "EXECUTED_SWEEP_INDICES",
    "FRAME_PERIOD_S",
    "HOME_JOINT_POSITIONS_RAD",
    "IK_MAX_ITERATIONS",
    "IK_MIN_ORIENTATION_SEPARATION_RAD",
    "IK_SEED_RNG",
    "IK_TOLERANCE_M",
    "OfficialAnchor",
    "OfflineOfficialMeshExperimentError",
    "OracleRenderRequest",
    "GeometryGateDecision",
    "InMemorySparseExperiment",
    "OBSERVATION_AXIAL_STD_M",
    "OBSERVATION_LATERAL_STD_M",
    "OBSERVATION_ROBUST_DELTA_M",
    "OBSERVATION_SEED_ROOT",
    "OCCLUSION_HOLD_FRAMES",
    "OCCLUSION_DURATIONS_S",
    "PATIENT_MOTION_BOUND_M_S",
    "PATIENT_MOTION_SEED_ROOT",
    "PATIENT_TRANSLATION_SPEED_M_S",
    "POLICY_NAMES",
    "POST_EVENT_DELTA_S",
    "PredeclaredObservationNoise",
    "RenderedGeometryGateFrame",
    "REQUIRED_STRATA_COUNT",
    "SCORING_TRAJECTORY_INDICES",
    "SparseEventSpec",
    "SparseStratumResult",
    "TargetMotionPlan",
    "TRAJECTORY_ACCELERATION_LIMIT_RAD_S2",
    "TRAJECTORY_FRAMES_PER_SEGMENT",
    "TRAJECTORY_SECONDS_PER_SEGMENT",
    "WristBranchPlan",
    "anchor_contract_sha256",
    "build_experiment_input_contract",
    "build_in_memory_sparse_experiment",
    "evaluate_rendered_geometry_gate",
    "frozen_ik_seed_table",
    "load_official_anchor_contract",
    "make_predeclared_observation_noise",
    "observation_sweep_contract",
    "offline_frame_evidence_sha256",
    "plan_official_ik_matrix",
    "rendered_gate_decision_from_evidence",
    "rendered_gate_frame_from_evidence",
    "revalidate_official_motion_plans",
    "validate_anchor_quantization",
]
