#!/usr/bin/env python3
"""Non-release official-mesh upper bound for two frozen-point frames.

The study compares an absolute camera-frame 3-D freeze with an ideal
simulator-GT patient-frame canonical freeze.  Both methods share one reference
confirmed from the same noisy oracle-2D and rendered RGB-D stream.  This does
not run v9, Isaac, hardware, or a production patient-pose provider.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np


AUSCULTATION_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(AUSCULTATION_ROOT) not in sys.path:
    sys.path.insert(0, str(AUSCULTATION_ROOT))

from ausc.perception.tracking.geometry import Intrinsics  # noqa: E402
from ausc.perception.tracking.offline_mesh_simulation import (  # noqa: E402
    EXPECTED_RUNTIME_MANIFEST_SHA256,
    EXPECTED_SCENE_CONTRACT_SHA256,
    HEIGHT,
    INTRINSICS,
    OfflineMeshScene,
    WIDTH,
    camera_pose_transform,
)
from ausc.perception.tracking.offline_official_mesh_assets import (  # noqa: E402
    EXPECTED_ASSET_MANIFEST_SHA256,
    EXPECTED_CONVERSION_MANIFEST_SHA256,
    verify_conversion_release,
)
from ausc.perception.tracking.offline_official_mesh_experiment import (  # noqa: E402
    FRAME_PERIOD_S,
    HOME_JOINT_POSITIONS_RAD,
    load_official_anchor_contract,
    plan_official_ik_matrix,
    revalidate_official_motion_plans,
)
from ausc.perception.tracking.official_openarm_experiment import (  # noqa: E402
    OFFICIAL_CLEAN_URDF_SHA256,
    POINT_NAMES,
)


class UpperBoundError(RuntimeError):
    """The deliberately fail-closed staging experiment violated its contract."""


SCHEMA = "wam_patient_frame_freeze_upper_bound_v1"
INPUT_SCHEMA = "wam_patient_frame_freeze_upper_bound_inputs_v1"
CONVERSION_RELEASE_ID = "official_meshes-b5e826af2539bceafea1"
ASSET_RELEASE_ID = "official_assets-05e66c94cd8b36ff79be"
TARGET_NAME = "R1"
WRIST_INDEX = 0
CAMERA_POSE_NAME = "primary"
FPS = 20
CONFIRMATION_FRAMES = 12
CONTACT_HOLD_FRAMES = 10
CONTACT_TRAJECTORY_INDEX = 60
PATIENT_TRANSLATION_SPEED_M_S = 0.012
PATIENT_MOTION_BOUND_M_S = 0.013
ORACLE_NOISE_STD_PX = 1.0
ORACLE_NOISE_SEED_ROOT = 91020260715
REFERENCE_PATCH_RADIUS_PX = 2
REFERENCE_MIN_HUMAN_PIXELS = 13
REFERENCE_MAX_DEVIATION_M = 0.015
MOTION_SPEED_SWEEP_M_S = (0.0, 0.006, 0.012, 0.013)
POSE_DELAY_SWEEP_S = (0.0, 0.05, 0.10, 0.20)
POSE_TRANSLATION_NOISE_SWEEP_M = (0.0, 0.001, 0.002, 0.005)
BANNER = "ORACLE2D / NOT V9 / NOMINAL ROOT / NOT ISAAC"
HUMAN_BANNER = "OFFICIAL NVIDIA REST"
METHOD_BANNER = (
    "IDEAL SIM PATIENT TF / UPPER BOUND / NOT AVAILABLE PRODUCTION"
)
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
EXPECTED_PYTHON = WORKSPACE_ROOT / ".wam-build/envs/offline-openarm-mac-v1/bin/python"
CONVERSION_RELEASE = (
    WORKSPACE_ROOT
    / ".wam-build/offline-official-openarm-meshes/releases"
    / CONVERSION_RELEASE_ID
)
ASSET_RELEASE = (
    WORKSPACE_ROOT
    / ".wam-build/offline-official-openarm-assets/releases"
    / ASSET_RELEASE_ID
)
RUNTIME_MANIFEST = (
    WORKSPACE_ROOT
    / "docs/runs/2026-07-15-offline-official-mesh-occlusion/raw/runtime_manifest.json"
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "docs/runs/2026-07-16-offline-patient-frame-upper-bound"
)
BOUND_SOURCE_FILES = {
    "upper_bound_runner": Path(__file__).resolve(),
    "ausc_init": AUSCULTATION_ROOT / "ausc/__init__.py",
    "tracking_init": AUSCULTATION_ROOT / "ausc/perception/tracking/__init__.py",
    "canonical": AUSCULTATION_ROOT / "ausc/perception/tracking/canonical.py",
    "detector": AUSCULTATION_ROOT / "ausc/perception/tracking/detector.py",
    "ekf": AUSCULTATION_ROOT / "ausc/perception/tracking/ekf.py",
    "experimental_visibility_geometry": (
        AUSCULTATION_ROOT
        / "ausc/perception/tracking/experimental_visibility_geometry.py"
    ),
    "flow": AUSCULTATION_ROOT / "ausc/perception/tracking/flow.py",
    "geometry": AUSCULTATION_ROOT / "ausc/perception/tracking/geometry.py",
    "intrinsics": AUSCULTATION_ROOT / "ausc/perception/tracking/intrinsics.py",
    "model_tracker": AUSCULTATION_ROOT / "ausc/perception/tracking/model_tracker.py",
    "official_asset_contract": (
        AUSCULTATION_ROOT / "ausc/perception/tracking/official_asset_contract.py"
    ),
    "official_openarm_experiment": (
        AUSCULTATION_ROOT / "ausc/perception/tracking/official_openarm_experiment.py"
    ),
    "offline_mesh_simulation": (
        AUSCULTATION_ROOT / "ausc/perception/tracking/offline_mesh_simulation.py"
    ),
    "offline_official_mesh_assets": (
        AUSCULTATION_ROOT
        / "ausc/perception/tracking/offline_official_mesh_assets.py"
    ),
    "offline_official_mesh_experiment": (
        AUSCULTATION_ROOT
        / "ausc/perception/tracking/offline_official_mesh_experiment.py"
    ),
    "rgbd_odometry": AUSCULTATION_ROOT / "ausc/perception/tracking/rgbd_odometry.py",
    "runtime": AUSCULTATION_ROOT / "ausc/perception/tracking/runtime.py",
    "tracker": AUSCULTATION_ROOT / "ausc/perception/tracking/tracker.py",
}
STAGING_WARNING_TEXT = (
    "NON-RELEASE METHOD UPPER-BOUND EXPERIMENT ONLY.\n"
    "completion=false; production_activation=false; recommendation_supported=false.\n"
    "No current pointer or completion marker is permitted in this directory.\n"
    "Linbox Isaac, real v9, patient-pose provider, hand-eye, and latency validation remain pending.\n"
)
REQUIRED_ARTIFACTS = (
    "README.md",
    "input_hashes.json",
    "metrics.json",
    "per_frame.csv",
    "report_frame.png",
    "patient_frame_upper_bound_h264.mp4",
)
PER_FRAME_FIELDS = (
    "frame_index",
    "timestamp_s",
    "phase",
    "trajectory_index",
    "patient_speed_m_s",
    "patient_translation_x_m",
    "robot_pixel_count",
    "target_robot_mask_5x5",
    "reference_candidate_valid",
    "reference_candidate_x_camera_m",
    "reference_candidate_y_camera_m",
    "reference_candidate_z_camera_m",
    "reference_ready",
    "reference_confirmed_this_frame",
    "oracle_u_px",
    "oracle_v_px",
    "gt_x_camera_m",
    "gt_y_camera_m",
    "gt_z_camera_m",
    "camera_lock_valid",
    "camera_lock_x_camera_m",
    "camera_lock_y_camera_m",
    "camera_lock_z_camera_m",
    "camera_lock_error_mm",
    "patient_lock_valid",
    "patient_lock_x_camera_m",
    "patient_lock_y_camera_m",
    "patient_lock_z_camera_m",
    "patient_lock_error_mm",
    "rendered_input_sha256",
    "frame_evidence_sha256",
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def camera_pose_transform_sha256() -> str:
    """Hash the exact primary world-from-CV-camera transform used by methods."""

    transform = np.ascontiguousarray(
        camera_pose_transform(CAMERA_POSE_NAME), dtype="<f8"
    )
    digest = hashlib.sha256()
    digest.update(b"wam_patient_frame_camera_pose_v1\0")
    digest.update(CAMERA_POSE_NAME.encode("ascii"))
    digest.update(transform.tobytes(order="C"))
    return digest.hexdigest()


def static_contract() -> dict[str, object]:
    """Return scope statements that every persisted result must retain."""

    return {
        "schema": SCHEMA,
        "oracle_2d": True,
        "real_v9_executed": False,
        "nominal_root_not_commissioned": True,
        "production_activation": False,
        "completion": False,
        "recommendation_supported": False,
        "simulator": "MuJoCo_kinematic_official_mesh_not_Isaac",
        "isaac_equivalence": False,
        "official_openarm": True,
        "official_nvidia_human_pose": "default_equals_rest",
        "target_name": TARGET_NAME,
        "wrist_index": WRIST_INDEX,
        "camera_pose_name": CAMERA_POSE_NAME,
        "comparison": (
            "camera_frame_3d_freeze_vs_ideal_patient_frame_canonical_freeze"
        ),
        "reference_source": (
            "12_frame_noisy_oracle2d_plus_rgbd_exact_simulator_mask_consensus"
        ),
        "anchor_3d_truth_used_by_methods": False,
        "anchor_3d_truth_used_for_scoring_only": True,
        "exact_simulator_human_mask_used_for_reference_gate": True,
        "reference_gate_available_in_production": False,
        "reference_consensus_max_deviation_m": REFERENCE_MAX_DEVIATION_M,
        "patient_transform_source": "simulator_ground_truth",
        "patient_transform_available_in_production": False,
        "patient_transform_method_upper_bound_only": True,
        "real_patient_pose_provider_executed": False,
        "hand_eye_validated": False,
        "latency_validated": False,
        "banner": BANNER,
        "human_banner": HUMAN_BANNER,
        "method_banner": METHOD_BANNER,
    }


def base_metrics() -> dict[str, object]:
    result = static_contract()
    result.update(
        {
            "frame_rate_hz": FPS,
            "frame_period_s": FRAME_PERIOD_S,
            "confirmation_frame_count": CONFIRMATION_FRAMES,
            "contact_hold_frame_count": CONTACT_HOLD_FRAMES,
            "patient_translation_speed_m_s": PATIENT_TRANSLATION_SPEED_M_S,
            "patient_motion_bound_m_s": PATIENT_MOTION_BOUND_M_S,
            "oracle_2d_noise_std_px": ORACLE_NOISE_STD_PX,
            "oracle_2d_noise_seed_root": ORACLE_NOISE_SEED_ROOT,
            "meaningful_staging_evidence": False,
        }
    )
    return result


def validate_metrics_contract(metrics: Mapping[str, object]) -> None:
    for key, expected in static_contract().items():
        actual = metrics.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise UpperBoundError(f"{key} violates the fixed upper-bound contract")
    if type(metrics.get("meaningful_staging_evidence")) is not bool:
        raise UpperBoundError("meaningful_staging_evidence must be boolean")
    fixed_base = base_metrics()
    for key in (
        "frame_rate_hz",
        "frame_period_s",
        "confirmation_frame_count",
        "contact_hold_frame_count",
        "patient_translation_speed_m_s",
        "patient_motion_bound_m_s",
        "oracle_2d_noise_std_px",
        "oracle_2d_noise_seed_root",
    ):
        if not _values_match(metrics.get(key), fixed_base[key]):
            raise UpperBoundError(f"{key} violates the fixed metric contract")
    if metrics.get("meaningful_staging_evidence") is True:
        if metrics.get("result_window_frame_count") != 132:
            raise UpperBoundError("meaningful evidence has the wrong result denominator")
        if metrics.get("camera_lock_coverage") != 1.0:
            raise UpperBoundError("camera lock coverage is incomplete")
        if metrics.get("patient_lock_coverage") != 1.0:
            raise UpperBoundError("patient lock coverage is incomplete")


@dataclass(frozen=True)
class TimelineFrame:
    frame_index: int
    timestamp_s: float
    phase: str
    trajectory_index: int | None
    left_joint_positions_rad: np.ndarray
    patient_speed_m_s: float
    patient_translation_x_m: float


def build_timeline(trajectory_joint_positions: np.ndarray) -> tuple[TimelineFrame, ...]:
    """Build the fixed official R1/w0 143-frame diagnostic timeline."""

    trajectory = np.asarray(trajectory_joint_positions, dtype=np.float64)
    if trajectory.shape != (121, 7) or not np.isfinite(trajectory).all():
        raise UpperBoundError("official R1/w0 trajectory must be exactly 121x7")
    rows: list[tuple[str, int | None, np.ndarray, float, float]] = []
    for _ in range(CONFIRMATION_FRAMES):
        rows.append(("confirm_home", None, HOME_JOINT_POSITIONS_RAD, 0.0, 0.0))
    for index in range(CONTACT_TRAJECTORY_INDEX + 1):
        rows.append(("approach", index, trajectory[index], 0.0, 0.0))
    for hold_index in range(1, CONTACT_HOLD_FRAMES + 1):
        elapsed_s = hold_index * FRAME_PERIOD_S
        rows.append(
            (
                "contact_hold",
                CONTACT_TRAJECTORY_INDEX,
                trajectory[CONTACT_TRAJECTORY_INDEX],
                PATIENT_TRANSLATION_SPEED_M_S,
                PATIENT_TRANSLATION_SPEED_M_S * elapsed_s,
            )
        )
    final_offset = (
        PATIENT_TRANSLATION_SPEED_M_S * CONTACT_HOLD_FRAMES * FRAME_PERIOD_S
    )
    for index in range(CONTACT_TRAJECTORY_INDEX + 1, len(trajectory)):
        rows.append(("depart", index, trajectory[index], 0.0, final_offset))
    timeline = tuple(
        TimelineFrame(
            frame_index=index,
            timestamp_s=index * FRAME_PERIOD_S,
            phase=phase,
            trajectory_index=trajectory_index,
            left_joint_positions_rad=np.asarray(joints, dtype=np.float64).copy(),
            patient_speed_m_s=float(speed),
            patient_translation_x_m=float(offset),
        )
        for index, (phase, trajectory_index, joints, speed, offset) in enumerate(rows)
    )
    if len(timeline) != 143:
        raise UpperBoundError("fixed timeline no longer has 143 frames")
    if max(row.patient_speed_m_s for row in timeline) > PATIENT_MOTION_BOUND_M_S:
        raise UpperBoundError("declared patient motion exceeds its fixed bound")
    return timeline


def noisy_oracle_pixels(projected_pixels: np.ndarray, frame_index: int) -> np.ndarray:
    projected = np.asarray(projected_pixels, dtype=np.float64)
    if projected.shape != (len(POINT_NAMES), 2) or not np.isfinite(projected).all():
        raise UpperBoundError("oracle projection is invalid")
    if type(frame_index) is not int or frame_index < 0:
        raise UpperBoundError("frame index is invalid")
    noise = np.clip(
        np.random.default_rng(ORACLE_NOISE_SEED_ROOT + frame_index).standard_normal(
            projected.shape
        ),
        -2.5,
        2.5,
    )
    return projected + ORACLE_NOISE_STD_PX * noise


def _rigid_transform(value: np.ndarray, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-10
        )
    ):
        raise UpperBoundError(f"{label} is not a rigid transform")
    return transform


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise UpperBoundError("point is invalid")
    return transform[:3, :3] @ value + transform[:3, 3]


def trusted_depth_lift(
    *,
    pixel: np.ndarray,
    depth_m: np.ndarray,
    robot_mask: np.ndarray,
    human_mask: np.ndarray,
    intrinsics: Intrinsics,
) -> np.ndarray | None:
    """Lift one simulator-gated reference candidate without anchor 3-D truth.

    The ideal robot and human masks deliberately make this an upper-bound
    reference gate.  Any robot pixel in the exact 5x5 footprint rejects the
    candidate.  Depth is the median of at least 13 human pixels in that patch.
    """

    uv = np.asarray(pixel, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    robot = np.asarray(robot_mask)
    human = np.asarray(human_mask)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise UpperBoundError("reference pixel is invalid")
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise UpperBoundError("reference depth shape changed")
    if robot.shape != depth.shape or robot.dtype != np.bool_:
        raise UpperBoundError("robot mask must be exact-shape bool")
    if human.shape != depth.shape or human.dtype != np.bool_:
        raise UpperBoundError("human mask must be exact-shape bool")
    ui, vi = int(round(float(uv[0]))), int(round(float(uv[1])))
    radius = REFERENCE_PATCH_RADIUS_PX
    if not (
        radius <= ui < intrinsics.width - radius
        and radius <= vi < intrinsics.height - radius
    ):
        return None
    ys = slice(vi - radius, vi + radius + 1)
    xs = slice(ui - radius, ui + radius + 1)
    if bool(robot[ys, xs].any()):
        return None
    human_patch = human[ys, xs]
    valid = human_patch & np.isfinite(depth[ys, xs])
    valid &= (depth[ys, xs] > 0.05) & (depth[ys, xs] < 4.0)
    if int(np.count_nonzero(valid)) < REFERENCE_MIN_HUMAN_PIXELS:
        return None
    z_m = float(np.median(depth[ys, xs][valid]))
    return intrinsics.unproject(uv[None], np.asarray([z_m]))[0]


@dataclass(frozen=True)
class FreezeOutput:
    camera_lock_xyz_camera_m: np.ndarray
    patient_lock_xyz_camera_m: np.ndarray


class FreezeComparator:
    """Freeze a single consensus point in camera and patient coordinates."""

    def __init__(self, *, required_confirmation_frames: int) -> None:
        if type(required_confirmation_frames) is not int or required_confirmation_frames < 1:
            raise UpperBoundError("confirmation frame count is invalid")
        self.required_confirmation_frames = required_confirmation_frames
        self._patient_samples: list[np.ndarray] = []
        self._reference_patient: np.ndarray | None = None
        self._reference_camera: np.ndarray | None = None
        self.reference_frame_index: int | None = None
        self.consensus_max_deviation_m: float | None = None

    @property
    def ready(self) -> bool:
        return self._reference_patient is not None

    @property
    def reference_patient_xyz_m(self) -> np.ndarray:
        if self._reference_patient is None:
            raise UpperBoundError("reference is not confirmed")
        return self._reference_patient.copy()

    @property
    def reference_camera_xyz_m(self) -> np.ndarray:
        if self._reference_camera is None:
            raise UpperBoundError("reference is not confirmed")
        return self._reference_camera.copy()

    def observe(
        self,
        xyz_camera_m: np.ndarray,
        world_from_patient: np.ndarray,
        world_from_cv_camera: np.ndarray,
    ) -> bool:
        if self.ready:
            raise UpperBoundError("comparison reference is already frozen")
        camera_point = np.asarray(xyz_camera_m, dtype=np.float64)
        if camera_point.shape != (3,) or not np.isfinite(camera_point).all():
            raise UpperBoundError("reference observation is invalid")
        patient = _rigid_transform(world_from_patient, "world_from_patient")
        camera = _rigid_transform(world_from_cv_camera, "world_from_cv_camera")
        world_point = _transform_point(camera, camera_point)
        patient_point = _transform_point(np.linalg.inv(patient), world_point)
        self._patient_samples.append(patient_point)
        if len(self._patient_samples) < self.required_confirmation_frames:
            return False
        patient_array = np.asarray(self._patient_samples, dtype=np.float64)
        patient_reference = np.median(patient_array, axis=0)
        # Both policies must start from the exact same physical consensus point.
        # Component-wise medians are not rotation-equivariant, so independently
        # taking a camera-frame median would create two subtly different references.
        reference_world = _transform_point(patient, patient_reference)
        camera_reference = _transform_point(np.linalg.inv(camera), reference_world)
        deviation = np.linalg.norm(patient_array - patient_reference[None], axis=1)
        max_deviation = float(np.max(deviation))
        if max_deviation > REFERENCE_MAX_DEVIATION_M + 1.0e-12:
            raise UpperBoundError("reference confirmation is not spatially stable")
        self._reference_patient = patient_reference
        self._reference_camera = camera_reference
        self.reference_frame_index = len(self._patient_samples) - 1
        self.consensus_max_deviation_m = max_deviation
        return True

    def predict(
        self,
        world_from_patient: np.ndarray,
        world_from_cv_camera: np.ndarray,
    ) -> FreezeOutput:
        if self._reference_patient is None or self._reference_camera is None:
            raise UpperBoundError("reference is not confirmed")
        patient = _rigid_transform(world_from_patient, "world_from_patient")
        camera = _rigid_transform(world_from_cv_camera, "world_from_cv_camera")
        patient_world = _transform_point(patient, self._reference_patient)
        patient_camera = _transform_point(np.linalg.inv(camera), patient_world)
        return FreezeOutput(
            camera_lock_xyz_camera_m=self._reference_camera.copy(),
            patient_lock_xyz_camera_m=patient_camera,
        )


def _error_mm(estimate: np.ndarray | None, truth: np.ndarray) -> float | None:
    if estimate is None:
        return None
    return float(np.linalg.norm(np.asarray(estimate) - np.asarray(truth)) * 1000.0)


def motion_sensitivity(
    *,
    reference_patient_xyz_m: np.ndarray,
    reference_camera_xyz_m: np.ndarray,
    world_from_cv_camera: np.ndarray,
) -> list[dict[str, float]]:
    """Analytic post-reference rigid-translation sweep, without rerendering."""

    reference_patient = np.asarray(reference_patient_xyz_m, dtype=np.float64)
    reference_camera = np.asarray(reference_camera_xyz_m, dtype=np.float64)
    camera = _rigid_transform(world_from_cv_camera, "world_from_cv_camera")
    if reference_patient.shape != (3,) or reference_camera.shape != (3,):
        raise UpperBoundError("sensitivity reference is invalid")
    result: list[dict[str, float]] = []
    for speed in MOTION_SPEED_SWEEP_M_S:
        patient = np.eye(4, dtype=np.float64)
        patient[0, 3] = speed * CONTACT_HOLD_FRAMES * FRAME_PERIOD_S
        truth_world = _transform_point(patient, reference_patient)
        truth_camera = _transform_point(np.linalg.inv(camera), truth_world)
        ideal_patient_camera = truth_camera.copy()
        result.append(
            {
                "patient_speed_m_s": speed,
                "final_translation_m": float(patient[0, 3]),
                "final_camera_lock_error_mm": float(
                    np.linalg.norm(reference_camera - truth_camera) * 1000.0
                ),
                "final_ideal_patient_lock_error_mm": float(
                    np.linalg.norm(ideal_patient_camera - truth_camera) * 1000.0
                ),
            }
        )
    return result


def pose_delay_sensitivity(
    *,
    reference_patient_xyz_m: np.ndarray,
    world_from_cv_camera: np.ndarray,
) -> list[dict[str, float]]:
    """Analytic nominal-motion sensitivity to a delayed patient transform."""

    reference = np.asarray(reference_patient_xyz_m, dtype=np.float64)
    camera = _rigid_transform(world_from_cv_camera, "world_from_cv_camera")
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise UpperBoundError("delay sensitivity reference is invalid")
    times = np.arange(CONTACT_HOLD_FRAMES + 1, dtype=np.float64) * FRAME_PERIOD_S
    result: list[dict[str, float]] = []
    for delay_s in POSE_DELAY_SWEEP_S:
        errors: list[float] = []
        for elapsed_s in times:
            true_patient = np.eye(4)
            true_patient[0, 3] = PATIENT_TRANSLATION_SPEED_M_S * elapsed_s
            delayed_patient = np.eye(4)
            delayed_patient[0, 3] = (
                PATIENT_TRANSLATION_SPEED_M_S * max(0.0, elapsed_s - delay_s)
            )
            truth = _transform_point(
                np.linalg.inv(camera), _transform_point(true_patient, reference)
            )
            estimate = _transform_point(
                np.linalg.inv(camera), _transform_point(delayed_patient, reference)
            )
            errors.append(float(np.linalg.norm(estimate - truth) * 1000.0))
        result.append(
            {
                "delay_s": delay_s,
                "max_error_mm": max(errors),
                "p95_error_mm": float(np.percentile(errors, 95)),
            }
        )
    return result


def pose_translation_noise_sensitivity() -> list[dict[str, float]]:
    """Exact translation-only pose-noise stress; orientation noise is not claimed."""

    return [
        {
            "pose_translation_noise_m": value,
            "resulting_point_error_mm": value * 1000.0,
        }
        for value in POSE_TRANSLATION_NOISE_SWEEP_M
    ]


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.9f}"
    return value


def per_frame_csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=PER_FRAME_FIELDS,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_value(row.get(name)) for name in PER_FRAME_FIELDS})
    return buffer.getvalue().encode("utf-8")


def semantic_frame_sha256(row: Mapping[str, object]) -> str:
    """Bind one raw render digest to every persisted non-self CSV field."""

    raw_sha = str(row.get("rendered_input_sha256", ""))
    if len(raw_sha) != 64:
        raise UpperBoundError("rendered input SHA is invalid")
    try:
        bytes.fromhex(raw_sha)
    except ValueError as exc:
        raise UpperBoundError("rendered input SHA is invalid") from exc
    normalized = [
        [name, str(_csv_value(row.get(name)))]
        for name in PER_FRAME_FIELDS
        if name != "frame_evidence_sha256"
    ]
    raw = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(b"wam_patient_frame_semantic_row_v1\0" + raw)


def _row_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in ("true", "false"):
        return value == "true"
    raise UpperBoundError(f"CSV boolean is invalid: {value!r}")


def _row_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise UpperBoundError(f"{label} is not a number") from exc
    if not math.isfinite(result):
        raise UpperBoundError(f"{label} is nonfinite")
    return result


def _optional_float(value: object, label: str) -> float | None:
    if value is None or value == "":
        return None
    return _row_float(value, label)


def _method_metrics(prefix: str, errors: Sequence[float], denominator: int) -> dict[str, object]:
    count = len(errors)
    result: dict[str, object] = {
        f"{prefix}_valid_frame_count": count,
        f"{prefix}_coverage": count / denominator,
        f"{prefix}_error_p50_mm": float(np.percentile(errors, 50)) if errors else None,
        f"{prefix}_error_p95_mm": float(np.percentile(errors, 95)) if errors else None,
    }
    for threshold in (5, 10):
        failures = sum(value > threshold for value in errors)
        result[f"{prefix}_false_valid_gt_{threshold}mm_count"] = failures
        result[f"{prefix}_false_valid_gt_{threshold}mm_rate"] = (
            failures / count if count else None
        )
    return result


def _row_xyz(
    row: Mapping[str, object], prefix: str, *, required: bool
) -> np.ndarray | None:
    values = [
        _optional_float(row.get(f"{prefix}_{axis}_camera_m"), f"{prefix} {axis}")
        for axis in ("x", "y", "z")
    ]
    present = [value is not None for value in values]
    if any(present) and not all(present):
        raise UpperBoundError(f"{prefix} validity has a partial xyz")
    if required and not all(present):
        raise UpperBoundError(f"{prefix} validity lacks xyz")
    if not any(present):
        return None
    return np.asarray(values, dtype=np.float64)


def _optional_trajectory_index(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise UpperBoundError("timeline trajectory index is invalid")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise UpperBoundError("timeline trajectory index is invalid") from exc
    if str(value).strip() not in {str(numeric), f"{numeric}.0"}:
        raise UpperBoundError("timeline trajectory index is invalid")
    return numeric


def validate_full_experiment_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Validate persisted method semantics independently of reported metrics."""

    if len(rows) != 143:
        raise UpperBoundError("full timeline must contain exactly 143 rows")
    truth_reference: np.ndarray | None = None
    camera_reference: np.ndarray | None = None
    patient_reference: np.ndarray | None = None
    world_from_camera = camera_pose_transform(CAMERA_POSE_NAME)
    camera_from_world_rotation = world_from_camera[:3, :3].T
    for index, row in enumerate(rows):
        if int(row.get("frame_index", -1)) != index:
            raise UpperBoundError("timeline frame indices changed")
        expected_timestamp = index * FRAME_PERIOD_S
        timestamp = _row_float(row.get("timestamp_s"), "timeline timestamp")
        if not math.isclose(timestamp, expected_timestamp, abs_tol=1.0e-9, rel_tol=0.0):
            raise UpperBoundError("timeline timestamp changed")
        if index < CONFIRMATION_FRAMES:
            expected_phase = "confirm_home"
            expected_trajectory = None
            expected_speed = 0.0
            expected_offset = 0.0
        elif index < CONFIRMATION_FRAMES + CONTACT_TRAJECTORY_INDEX + 1:
            expected_phase = "approach"
            expected_trajectory = index - CONFIRMATION_FRAMES
            expected_speed = 0.0
            expected_offset = 0.0
        elif index < CONFIRMATION_FRAMES + CONTACT_TRAJECTORY_INDEX + 1 + CONTACT_HOLD_FRAMES:
            expected_phase = "contact_hold"
            expected_trajectory = CONTACT_TRAJECTORY_INDEX
            hold_index = index - (CONFIRMATION_FRAMES + CONTACT_TRAJECTORY_INDEX)
            expected_speed = PATIENT_TRANSLATION_SPEED_M_S
            expected_offset = expected_speed * hold_index * FRAME_PERIOD_S
        else:
            expected_phase = "depart"
            expected_trajectory = index - 22
            expected_speed = 0.0
            expected_offset = (
                PATIENT_TRANSLATION_SPEED_M_S
                * CONTACT_HOLD_FRAMES
                * FRAME_PERIOD_S
            )
        if str(row.get("phase")) != expected_phase:
            raise UpperBoundError("timeline phase changed")
        if _optional_trajectory_index(row.get("trajectory_index")) != expected_trajectory:
            raise UpperBoundError("timeline trajectory index changed")
        speed = _row_float(row.get("patient_speed_m_s"), "timeline patient speed")
        offset = _row_float(
            row.get("patient_translation_x_m"), "timeline patient offset"
        )
        if not math.isclose(speed, expected_speed, abs_tol=1.0e-12, rel_tol=0.0):
            raise UpperBoundError("timeline patient speed changed")
        if not math.isclose(offset, expected_offset, abs_tol=1.0e-9, rel_tol=0.0):
            raise UpperBoundError("timeline patient offset changed")
        expected_ready = index >= CONFIRMATION_FRAMES - 1
        if _row_bool(row.get("reference_ready")) is not expected_ready:
            raise UpperBoundError("reference-ready schedule changed")
        if _row_bool(row.get("reference_confirmed_this_frame")) is not (
            index == CONFIRMATION_FRAMES - 1
        ):
            raise UpperBoundError("reference-confirmation schedule changed")
        candidate_valid = _row_bool(row.get("reference_candidate_valid"))
        candidate_xyz = _row_xyz(
            row, "reference_candidate", required=candidate_valid
        )
        if candidate_valid != (candidate_xyz is not None):
            raise UpperBoundError("reference candidate validity disagrees with xyz")
        if index < CONFIRMATION_FRAMES and not candidate_valid:
            raise UpperBoundError("reference confirmation lost a required candidate")
        _row_bool(row.get("target_robot_mask_5x5"))
        try:
            robot_pixels = int(row.get("robot_pixel_count", -1))
        except (TypeError, ValueError) as exc:
            raise UpperBoundError("robot pixel count is invalid") from exc
        if robot_pixels < 0:
            raise UpperBoundError("robot pixel count is invalid")
        _row_float(row.get("oracle_u_px"), "oracle u")
        _row_float(row.get("oracle_v_px"), "oracle v")
        truth = np.asarray(
            [
                _row_float(row.get("gt_x_camera_m"), "GT x"),
                _row_float(row.get("gt_y_camera_m"), "GT y"),
                _row_float(row.get("gt_z_camera_m"), "GT z"),
            ],
            dtype=np.float64,
        )
        camera_valid = _row_bool(row.get("camera_lock_valid"))
        patient_valid = _row_bool(row.get("patient_lock_valid"))
        if camera_valid is not expected_ready or patient_valid is not expected_ready:
            raise UpperBoundError("method validity schedule changed")
        camera_xyz = _row_xyz(row, "camera_lock", required=camera_valid)
        patient_xyz = _row_xyz(row, "patient_lock", required=patient_valid)
        camera_error = _optional_float(
            row.get("camera_lock_error_mm"), "camera lock error"
        )
        patient_error = _optional_float(
            row.get("patient_lock_error_mm"), "patient lock error"
        )
        if camera_valid != (camera_xyz is not None):
            raise UpperBoundError("camera validity disagrees with xyz")
        if patient_valid != (patient_xyz is not None):
            raise UpperBoundError("patient validity disagrees with xyz")
        if camera_valid != (camera_error is not None):
            raise UpperBoundError("camera validity disagrees with error")
        if patient_valid != (patient_error is not None):
            raise UpperBoundError("patient validity disagrees with error")
        if index == CONFIRMATION_FRAMES - 1:
            assert camera_xyz is not None and patient_xyz is not None
            if not np.allclose(camera_xyz, patient_xyz, atol=2.0e-9, rtol=0.0):
                raise UpperBoundError("reference methods do not share one physical point")
            truth_reference = truth.copy()
            camera_reference = camera_xyz.copy()
            patient_reference = patient_xyz.copy()
        if expected_ready:
            assert truth_reference is not None
            assert camera_reference is not None
            assert patient_reference is not None
            expected_delta = camera_from_world_rotation @ np.asarray(
                [expected_offset, 0.0, 0.0]
            )
            if not np.allclose(
                truth, truth_reference + expected_delta, atol=2.0e-8, rtol=0.0
            ):
                raise UpperBoundError("GT trajectory is inconsistent with the timeline")
            assert camera_xyz is not None and patient_xyz is not None
            if not np.allclose(
                camera_xyz, camera_reference, atol=2.0e-9, rtol=0.0
            ):
                raise UpperBoundError("camera freeze moved after confirmation")
            if not np.allclose(
                patient_xyz,
                patient_reference + expected_delta,
                atol=2.0e-8,
                rtol=0.0,
            ):
                raise UpperBoundError("patient freeze disagrees with the rigid transform")
            expected_camera_error = float(np.linalg.norm(camera_xyz - truth) * 1000.0)
            expected_patient_error = float(np.linalg.norm(patient_xyz - truth) * 1000.0)
            if not math.isclose(
                float(camera_error),
                expected_camera_error,
                abs_tol=5.0e-6,
                rel_tol=0.0,
            ):
                raise UpperBoundError("camera error is not geometrically self-consistent")
            if not math.isclose(
                float(patient_error),
                expected_patient_error,
                abs_tol=5.0e-6,
                rel_tol=0.0,
            ):
                raise UpperBoundError("patient error is not geometrically self-consistent")


def derive_row_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise UpperBoundError("per-frame result is empty")
    frame_indices = [int(row["frame_index"]) for row in rows]
    if frame_indices != list(range(frame_indices[0], frame_indices[0] + len(rows))):
        raise UpperBoundError("frame indices are not contiguous")
    first_reference: int | None = None
    for index, row in enumerate(rows):
        if _row_bool(row.get("reference_ready")):
            first_reference = index
            break
    if first_reference is None:
        raise UpperBoundError("comparison reference was never confirmed")
    result_rows = rows[first_reference:]
    denominator = len(result_rows)
    camera_errors: list[float] = []
    patient_errors: list[float] = []
    moved_camera: list[float] = []
    moved_patient: list[float] = []
    reference_offset = _row_float(
        result_rows[0].get("patient_translation_x_m"), "reference patient offset"
    )
    for row in result_rows:
        camera_valid = _row_bool(row.get("camera_lock_valid"))
        patient_valid = _row_bool(row.get("patient_lock_valid"))
        camera_error = _optional_float(
            row.get("camera_lock_error_mm"), "camera lock error"
        )
        patient_error = _optional_float(
            row.get("patient_lock_error_mm"), "patient lock error"
        )
        if camera_valid != (camera_error is not None):
            raise UpperBoundError("camera lock validity disagrees with its error")
        if patient_valid != (patient_error is not None):
            raise UpperBoundError("patient lock validity disagrees with its error")
        if camera_error is not None:
            camera_errors.append(camera_error)
        if patient_error is not None:
            patient_errors.append(patient_error)
        moved = (
            _row_float(row.get("patient_translation_x_m"), "patient offset")
            > reference_offset + 1.0e-12
        )
        if moved and camera_error is not None:
            moved_camera.append(camera_error)
        if moved and patient_error is not None:
            moved_patient.append(patient_error)
    result: dict[str, object] = {
        "frame_count": len(rows),
        "duration_s": _row_float(rows[-1]["timestamp_s"], "duration"),
        "result_window_first_frame_index": int(result_rows[0]["frame_index"]),
        "result_window_frame_count": denominator,
        "common_denominator": "all_frames_at_or_after_reference_confirmation",
        "preconfirmation_frame_count": first_reference,
        "moved_patient_result_frame_count": sum(
            _row_float(row.get("patient_translation_x_m"), "patient offset")
            > reference_offset + 1.0e-12
            for row in result_rows
        ),
        "target_robot_mask_result_frame_count": sum(
            _row_bool(row.get("target_robot_mask_5x5", False))
            for row in result_rows
        ),
    }
    result.update(_method_metrics("camera_lock", camera_errors, denominator))
    result.update(_method_metrics("patient_lock", patient_errors, denominator))
    result.update(
        {
            "moved_camera_lock_error_p50_mm": (
                float(np.percentile(moved_camera, 50)) if moved_camera else None
            ),
            "moved_camera_lock_error_p95_mm": (
                float(np.percentile(moved_camera, 95)) if moved_camera else None
            ),
            "moved_patient_lock_error_p50_mm": (
                float(np.percentile(moved_patient, 50)) if moved_patient else None
            ),
            "moved_patient_lock_error_p95_mm": (
                float(np.percentile(moved_patient, 95)) if moved_patient else None
            ),
        }
    )
    return result


def _values_match(actual: object, expected: object) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(
            _values_match(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            _values_match(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=2.0e-6
        )
    return type(actual) is type(expected) and actual == expected


def _expected_video_contract(expected_frames: int = 143) -> dict[str, object]:
    return {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": WIDTH,
        "height": HEIGHT,
        "r_frame_rate": f"{FPS}/1",
        "nb_read_frames": str(expected_frames),
    }


def derive_nonrow_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Recompute every persisted non-row metric from rows or fixed contracts."""

    validate_full_experiment_rows(rows)
    world_from_camera = camera_pose_transform(CAMERA_POSE_NAME)
    patient_candidates: list[np.ndarray] = []
    for row in rows[:CONFIRMATION_FRAMES]:
        candidate = _row_xyz(row, "reference_candidate", required=True)
        assert candidate is not None
        patient_candidates.append(_transform_point(world_from_camera, candidate))
    candidate_array = np.asarray(patient_candidates, dtype=np.float64)
    reference_patient = np.median(candidate_array, axis=0)
    consensus_max_deviation_m = float(
        np.max(np.linalg.norm(candidate_array - reference_patient[None], axis=1))
    )
    reference_row = rows[CONFIRMATION_FRAMES - 1]
    reference_camera = _row_xyz(reference_row, "camera_lock", required=True)
    assert reference_camera is not None
    row_metrics = derive_row_metrics(rows)
    meaningful = bool(
        row_metrics["result_window_frame_count"] == 132
        and row_metrics["camera_lock_coverage"] == 1.0
        and row_metrics["patient_lock_coverage"] == 1.0
        and row_metrics["moved_patient_result_frame_count"] > 0
    )
    return {
        "official_trajectory_frame_count": 121,
        "reference_confirmation_frame_index": CONFIRMATION_FRAMES - 1,
        "reference_consensus_max_deviation_mm": consensus_max_deviation_m * 1000.0,
        "patient_translation_during_hold_m": (
            PATIENT_TRANSLATION_SPEED_M_S
            * CONTACT_HOLD_FRAMES
            * FRAME_PERIOD_S
        ),
        "motion_sensitivity": motion_sensitivity(
            reference_patient_xyz_m=reference_patient,
            reference_camera_xyz_m=reference_camera,
            world_from_cv_camera=world_from_camera,
        ),
        "pose_delay_sensitivity": pose_delay_sensitivity(
            reference_patient_xyz_m=reference_patient,
            world_from_cv_camera=world_from_camera,
        ),
        "pose_translation_noise_sensitivity": pose_translation_noise_sensitivity(),
        "pose_orientation_noise_sensitivity_executed": False,
        "meaningful_staging_evidence": meaningful,
        "report_frame_index": (
            CONFIRMATION_FRAMES
            + CONTACT_TRAJECTORY_INDEX
            + 1
            + CONTACT_HOLD_FRAMES
            - 1
        ),
        "video_contract": _expected_video_contract(),
    }


def verify_metrics_against_csv(
    metrics: Mapping[str, object], csv_raw: bytes, inputs: Mapping[str, object]
) -> None:
    try:
        text = csv_raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        if tuple(reader.fieldnames or ()) != PER_FRAME_FIELDS:
            raise UpperBoundError("CSV field order changed")
        rows = list(reader)
    except UnicodeDecodeError as exc:
        raise UpperBoundError("CSV is not UTF-8") from exc
    validate_metrics_contract(metrics)
    current_schema = inputs.get("schema") == INPUT_SCHEMA
    if current_schema:
        validate_full_experiment_rows(rows)
    derived = derive_row_metrics(rows)
    for key, expected in derived.items():
        if not _values_match(metrics.get(key), expected):
            raise UpperBoundError(f"persisted {key} disagrees with CSV")
    if current_schema:
        for key, expected in derive_nonrow_metrics(rows).items():
            if not _values_match(metrics.get(key), expected):
                raise UpperBoundError(f"persisted {key} disagrees with fixed/CSV evidence")
    digest = hashlib.sha256()
    raw_digest = hashlib.sha256()
    for row in rows:
        raw_value = row.get("rendered_input_sha256", "")
        value = row.get("frame_evidence_sha256", "")
        if len(raw_value) != 64 or len(value) != 64:
            raise UpperBoundError("CSV frame evidence hash is invalid")
        try:
            raw_digest.update(bytes.fromhex(raw_value))
            digest.update(bytes.fromhex(value))
        except ValueError as exc:
            raise UpperBoundError("CSV frame evidence hash is invalid") from exc
        if semantic_frame_sha256(row) != value:
            raise UpperBoundError("CSV frame evidence does not bind row semantics")
    expected_raw_stream = inputs.get("raw_rendered_input_stream_sha256")
    if raw_digest.hexdigest() != expected_raw_stream:
        raise UpperBoundError("CSV raw rendered-input stream disagrees with input hashes")
    expected_stream = inputs.get("rendered_evidence_stream_sha256")
    if digest.hexdigest() != expected_stream:
        raise UpperBoundError("CSV evidence stream disagrees with input hashes")


def bound_source_manifest() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, path in BOUND_SOURCE_FILES.items():
        if not path.is_file() or path.is_symlink():
            raise UpperBoundError(f"bound source file is unavailable: {name}")
        result[name] = {
            "path": str(path.relative_to(WORKSPACE_ROOT)),
            "sha256": _sha256_file(path),
        }
    return result


def validate_bound_source_inputs(inputs: Mapping[str, object]) -> None:
    if inputs.get("schema") != INPUT_SCHEMA:
        raise UpperBoundError("input schema is not the fixed upper-bound schema")
    if inputs.get("camera_pose_transform_sha256") != camera_pose_transform_sha256():
        raise UpperBoundError("camera transform binding changed")
    declared = inputs.get("bound_source_files")
    if not isinstance(declared, Mapping):
        raise UpperBoundError("source binding is unavailable")
    normalized: dict[str, dict[str, str]] = {}
    try:
        for name, value in declared.items():
            if not isinstance(name, str) or not isinstance(value, Mapping):
                raise TypeError
            normalized[name] = {
                "path": str(value["path"]),
                "sha256": str(value["sha256"]),
            }
    except (KeyError, TypeError) as exc:
        raise UpperBoundError("source binding is malformed") from exc
    current = bound_source_manifest()
    if normalized != current:
        raise UpperBoundError("source binding changed after execution")
    runner = normalized.get("upper_bound_runner")
    if runner is None or inputs.get("upper_bound_runner_sha256") != runner["sha256"]:
        raise UpperBoundError("upper-bound runner SHA is not source-bound")


def _frame_digest(
    *,
    joints: np.ndarray,
    world_from_patient: np.ndarray,
    oracle_pixels: np.ndarray,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    robot_mask: np.ndarray,
    human_mask: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for label, value, dtype in (
        (b"joints\0", joints, "<f8"),
        (b"patient\0", world_from_patient, "<f8"),
        (b"oracle\0", oracle_pixels, "<f8"),
        (b"rgb\0", rgb, np.uint8),
        (b"depth\0", depth_m, "<f8"),
        (b"robot\0", robot_mask, np.uint8),
        (b"human\0", human_mask, np.uint8),
    ):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(label)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _target_masked(pixel: np.ndarray, robot_mask: np.ndarray) -> bool:
    uv = np.asarray(pixel, dtype=np.float64)
    robot = np.asarray(robot_mask, dtype=bool)
    ui, vi = int(round(float(uv[0]))), int(round(float(uv[1])))
    radius = REFERENCE_PATCH_RADIUS_PX
    if not (0 <= ui < robot.shape[1] and 0 <= vi < robot.shape[0]):
        return False
    return bool(
        robot[
            max(0, vi - radius) : min(robot.shape[0], vi + radius + 1),
            max(0, ui - radius) : min(robot.shape[1], ui + radius + 1),
        ].any()
    )


def _draw_cross(
    frame: np.ndarray,
    pixel: np.ndarray | None,
    colour: tuple[int, int, int],
    size: int,
) -> None:
    if pixel is None or not np.isfinite(pixel).all():
        return
    x, y = int(round(float(pixel[0]))), int(round(float(pixel[1])))
    if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
        return
    cv2.line(frame, (x - size, y), (x + size, y), colour, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - size), (x, y + size), colour, 2, cv2.LINE_AA)


def overlay_frame(
    *,
    rgb: np.ndarray,
    robot_mask: np.ndarray,
    oracle_pixels: np.ndarray,
    truth_pixel: np.ndarray,
    camera_lock_xyz: np.ndarray | None,
    patient_lock_xyz: np.ndarray | None,
    truth_xyz: np.ndarray,
    intrinsics: Intrinsics,
    phase: str,
    timestamp_s: float,
) -> np.ndarray:
    """Render a permanent-scope overlay and a magnified target inset."""

    frame = np.asarray(rgb, dtype=np.uint8).copy()
    robot = np.asarray(robot_mask, dtype=bool)
    if robot.any():
        tint = np.asarray([255.0, 35.0, 35.0])
        frame[robot] = np.clip(0.72 * frame[robot] + 0.28 * tint, 0, 255).astype(
            np.uint8
        )
    for pixel in np.asarray(oracle_pixels):
        if np.asarray(pixel).shape == (2,) and np.isfinite(pixel).all():
            cv2.circle(
                frame,
                (int(round(float(pixel[0]))), int(round(float(pixel[1])))),
                2,
                (255, 220, 70),
                -1,
                cv2.LINE_AA,
            )
    _draw_cross(frame, truth_pixel, (55, 255, 90), 7)
    camera_pixel = (
        None
        if camera_lock_xyz is None
        else intrinsics.project(np.asarray(camera_lock_xyz)[None])[0]
    )
    patient_pixel = (
        None
        if patient_lock_xyz is None
        else intrinsics.project(np.asarray(patient_lock_xyz)[None])[0]
    )
    if camera_pixel is not None and np.isfinite(camera_pixel).all():
        cv2.circle(
            frame,
            (int(round(float(camera_pixel[0]))), int(round(float(camera_pixel[1])))),
            7,
            (255, 70, 225),
            2,
            cv2.LINE_AA,
        )
    _draw_cross(frame, patient_pixel, (45, 225, 255), 5)

    # Magnify the small millimetre-scale separation without altering metrics.
    truth_uv = np.asarray(truth_pixel, dtype=np.float64)
    if truth_uv.shape == (2,) and np.isfinite(truth_uv).all():
        tx, ty = int(round(float(truth_uv[0]))), int(round(float(truth_uv[1])))
        x0, x1 = max(0, tx - 24), min(WIDTH, tx + 24)
        y0, y1 = max(64, ty - 18), min(HEIGHT - 58, ty + 18)
    else:
        x0 = x1 = y0 = y1 = 0
    if x1 > x0 + 4 and y1 > y0 + 4:
        inset = cv2.resize(frame[y0:y1, x0:x1], (160, 120), interpolation=cv2.INTER_NEAREST)
        frame[74:194, 472:632] = inset
        cv2.rectangle(frame, (471, 73), (633, 195), (255, 255, 255), 1)
        cv2.putText(
            frame,
            "TARGET ZOOM",
            (480, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    frame[:64] = (0.18 * frame[:64]).astype(np.uint8)
    for line, y, colour, scale in (
        (BANNER, 18, (255, 255, 255), 0.46),
        (HUMAN_BANNER, 37, (150, 220, 255), 0.43),
        (METHOD_BANNER, 57, (255, 210, 80), 0.38),
    ):
        cv2.putText(
            frame,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            colour,
            1,
            cv2.LINE_AA,
        )
    frame[HEIGHT - 58 :] = (0.20 * frame[HEIGHT - 58 :]).astype(np.uint8)
    camera_error = _error_mm(camera_lock_xyz, truth_xyz)
    patient_error = _error_mm(patient_lock_xyz, truth_xyz)
    camera_text = "--" if camera_error is None else f"{camera_error:.2f}mm"
    patient_text = "--" if patient_error is None else f"{patient_error:.2f}mm"
    cv2.putText(
        frame,
        f"R1/w0  {phase}  t={timestamp_s:.2f}s",
        (8, HEIGHT - 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"GT green | CAMERA magenta {camera_text} | PATIENT cyan {patient_text}",
        (8, HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return frame


def ffmpeg_command(destination: Path) -> list[str]:
    return [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        str(destination),
    ]


def _encode_video(frames: Iterable[np.ndarray], destination: Path) -> None:
    if not Path(FFMPEG).is_file():
        raise UpperBoundError("locked ffmpeg is unavailable")
    process = subprocess.Popen(
        ffmpeg_command(destination),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stderr is not None
    try:
        for index, frame in enumerate(frames):
            value = np.asarray(frame)
            if value.shape != (HEIGHT, WIDTH, 3) or value.dtype != np.uint8:
                raise UpperBoundError(f"video frame {index} violates 640x360 RGB8")
            process.stdin.write(np.ascontiguousarray(value).tobytes(order="C"))
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait(timeout=120)
    except BaseException:
        process.kill()
        process.wait()
        raise
    if returncode != 0:
        raise UpperBoundError(f"ffmpeg failed: {stderr.strip()}")


def _probe_video(path: Path, expected_frames: int) -> dict[str, object]:
    if not Path(FFPROBE).is_file():
        raise UpperBoundError("locked ffprobe is unavailable")
    result = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames,codec_type",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise UpperBoundError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        streams = json.loads(result.stdout)["streams"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise UpperBoundError("ffprobe output is invalid") from exc
    video = [row for row in streams if row.get("codec_type") == "video"]
    audio = [row for row in streams if row.get("codec_type") == "audio"]
    if len(video) != 1 or audio:
        raise UpperBoundError("video stream inventory changed")
    expected = _expected_video_contract(expected_frames)
    for key, value in expected.items():
        if video[0].get(key) != value:
            raise UpperBoundError(f"video {key} changed")
    return expected


def _clear_except_warning(destination: Path) -> None:
    if not destination.exists():
        return
    for path in destination.iterdir():
        if path.name == "STAGING_WARNING":
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def publish_staging_artifacts(*, scratch: Path, destination: Path) -> None:
    scratch_path = Path(scratch)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    warning = destination_path / "STAGING_WARNING"
    if not warning.exists():
        warning.write_text(STAGING_WARNING_TEXT, encoding="utf-8")
    unexpected = [path.name for path in destination_path.iterdir() if path.name != "STAGING_WARNING"]
    if unexpected:
        raise UpperBoundError("staging destination already contains artifacts")
    for name in REQUIRED_ARTIFACTS:
        source = scratch_path / name
        if not source.is_file() or source.is_symlink():
            raise UpperBoundError(f"required staging artifact is missing: {name}")
    for name in REQUIRED_ARTIFACTS:
        os.replace(scratch_path / name, destination_path / name)
    sums = "".join(
        f"{_sha256_file(destination_path / name)}  {name}\n"
        for name in sorted(REQUIRED_ARTIFACTS)
    )
    (destination_path / "SHA256SUMS").write_text(sums, encoding="utf-8")
    forbidden = ("COMPLETE", "COMPLETED", "COMPLETION.json", "current")
    if any((destination_path / name).exists() or (destination_path / name).is_symlink() for name in forbidden):
        raise UpperBoundError("completion marker or current pointer is forbidden")


def _tool_version(path: str) -> str:
    result = subprocess.run(
        [path, "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise UpperBoundError(f"tool version probe failed: {path}")
    for line in result.stdout.splitlines():
        if line.strip():
            return line.strip()
    raise UpperBoundError(f"tool version output is empty: {path}")


def _read_official_inputs() -> tuple[bytes, bytes, Mapping[str, object], Path]:
    if platform.machine() != "arm64":
        raise UpperBoundError("study requires the locked arm64 Mac runtime")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise UpperBoundError(f"run with locked Python: {EXPECTED_PYTHON}")
    runtime_raw = RUNTIME_MANIFEST.read_bytes()
    if _sha256_bytes(runtime_raw) != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise UpperBoundError("locked Mac runtime manifest changed")
    conversion_raw = (CONVERSION_RELEASE / "conversion_manifest.json").read_bytes()
    if _sha256_bytes(conversion_raw) != EXPECTED_CONVERSION_MANIFEST_SHA256:
        raise UpperBoundError("official conversion manifest changed")
    manifest = verify_conversion_release(CONVERSION_RELEASE)
    if manifest.get("asset_release_id") != ASSET_RELEASE_ID:
        raise UpperBoundError("official asset release identity changed")
    asset_manifest = ASSET_RELEASE / "asset_manifest.json"
    if _sha256_file(asset_manifest) != EXPECTED_ASSET_MANIFEST_SHA256:
        raise UpperBoundError("official asset manifest changed")
    urdf_path = ASSET_RELEASE / str(manifest["robot"]["urdf_path"])
    if _sha256_file(urdf_path) != OFFICIAL_CLEAN_URDF_SHA256:
        raise UpperBoundError("official OpenArm URDF changed")
    return runtime_raw, conversion_raw, manifest, urdf_path


def _input_hashes(
    *,
    runtime_raw: bytes,
    conversion_raw: bytes,
    urdf_path: Path,
    motion_plan_sha256: str,
    oracle_stream_sha256: str,
    raw_rendered_input_stream_sha256: str,
    evidence_stream_sha256: str,
    sources: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    runner_sha = str(sources["upper_bound_runner"]["sha256"])
    return {
        "schema": INPUT_SCHEMA,
        "architecture": platform.machine(),
        "python_executable": str(EXPECTED_PYTHON.relative_to(WORKSPACE_ROOT)),
        "python_version": platform.python_version(),
        "runtime_manifest_path": str(RUNTIME_MANIFEST.relative_to(WORKSPACE_ROOT)),
        "runtime_manifest_sha256": _sha256_bytes(runtime_raw),
        "conversion_release_id": CONVERSION_RELEASE_ID,
        "conversion_manifest_sha256": _sha256_bytes(conversion_raw),
        "asset_release_id": ASSET_RELEASE_ID,
        "asset_manifest_sha256": _sha256_file(ASSET_RELEASE / "asset_manifest.json"),
        "official_openarm_urdf_path": str(urdf_path.relative_to(WORKSPACE_ROOT)),
        "official_openarm_urdf_sha256": _sha256_file(urdf_path),
        "scene_contract_sha256": EXPECTED_SCENE_CONTRACT_SHA256,
        "camera_pose_transform_sha256": camera_pose_transform_sha256(),
        "motion_plan_sha256": motion_plan_sha256,
        "oracle_2d_stream_sha256": oracle_stream_sha256,
        "raw_rendered_input_stream_sha256": raw_rendered_input_stream_sha256,
        "rendered_evidence_stream_sha256": evidence_stream_sha256,
        "bound_source_files": {name: dict(value) for name, value in sources.items()},
        "upper_bound_runner_sha256": runner_sha,
        "ffmpeg_path": FFMPEG,
        "ffmpeg_sha256": _sha256_file(Path(FFMPEG)),
        "ffmpeg_version": _tool_version(FFMPEG),
        "ffprobe_path": FFPROBE,
        "ffprobe_sha256": _sha256_file(Path(FFPROBE)),
        "ffprobe_version": _tool_version(FFPROBE),
    }


def _readme(metrics: Mapping[str, object]) -> str:
    def mm(key: str) -> str:
        value = metrics.get(key)
        return "none" if value is None else f"{float(value):.2f}"

    def pct(key: str) -> str:
        value = metrics.get(key)
        return "none" if value is None else f"{100.0 * float(value):.1f}%"

    return f"""# Camera-frame vs patient-frame freeze (non-release upper bound)

This staging experiment answers one narrow method question: after one R1 reference is confirmed, is it better to keep the 3-D point fixed in the camera frame or keep its canonical coordinate fixed in the patient frame? It does **not** validate a production implementation or recommend activation.

The run uses 143 frames at 20 Hz: 12 HOME confirmation frames, the official R1/wrist-0 OpenArm trajectory, a 10-frame contact hold, and departure. Geometry is the pinned official OpenArm plus the official NVIDIA human in its authored `default_equals_rest` pose. This is Mac MuJoCo kinematic rendering, not Isaac. The root placement remains nominal and uncommissioned.

Both methods share the same reference. It is the median of 12 consecutive R1 lifts from deterministic 1 px noisy Oracle2D plus the same rendered metric depth. The exact simulator robot mask must be absent from the 5x5 footprint and at least 13 exact simulator human pixels must support depth. This intentionally ideal gate is **not available in production**. Anchor 3-D truth is not used by the reference or either method; it is used only for scoring and video overlays.

From confirmation frame {metrics['reference_confirmation_frame_index']} onward, both methods use the same {metrics['result_window_frame_count']}-frame denominator. Camera-lock coverage is {pct('camera_lock_coverage')} with p50/p95 {mm('camera_lock_error_p50_mm')}/{mm('camera_lock_error_p95_mm')} mm and false-valid >5/>10 mm {pct('camera_lock_false_valid_gt_5mm_rate')}/{pct('camera_lock_false_valid_gt_10mm_rate')}. Ideal patient-lock coverage is {pct('patient_lock_coverage')} with p50/p95 {mm('patient_lock_error_p50_mm')}/{mm('patient_lock_error_p95_mm')} mm and false-valid >5/>10 mm {pct('patient_lock_false_valid_gt_5mm_rate')}/{pct('patient_lock_false_valid_gt_10mm_rate')}.

On the {metrics['moved_patient_result_frame_count']} frames after patient motion begins, camera-lock p95 is {mm('moved_camera_lock_error_p95_mm')} mm and ideal patient-lock p95 is {mm('moved_patient_lock_error_p95_mm')} mm. The reference consensus maximum deviation is {mm('reference_consensus_max_deviation_mm')} mm. The declared rigid translation is 0.012 m/s during 0.50 s, below the 0.013 m/s bound.

`motion_sensitivity` is an analytic post-reference rigid-translation sweep at 0, 0.006, 0.012, and 0.013 m/s. It does not rerender or claim extra RGB-D trials. `pose_delay_sensitivity` analytically delays the ideal patient transform by 0, 50, 100, and 200 ms. `pose_translation_noise_sensitivity` is translation-only; orientation noise is not tested. These isolate frame semantics from detector and association behavior.

The patient-frame result is an **ideal simulator-GT upper bound**. A real method still needs a production patient-pose provider, camera/patient timestamp alignment, hand-eye calibration, tool and mount calibration, robot-mask provenance, and Linbox Isaac/hardware validation. None is supplied by this Mac result.

Permanent video banners: `{BANNER}`, `{HUMAN_BANNER}`, `{METHOD_BANNER}`. The target zoom is display-only and does not affect metrics.

`completion=false`; `production_activation=false`; `recommendation_supported=false`. `STAGING_WARNING` is deliberately retained. No completion marker, release, or `current` pointer is written.
"""


def _simulate() -> tuple[
    list[dict[str, object]],
    list[np.ndarray],
    int,
    dict[str, object],
    dict[str, object],
]:
    sources_before = bound_source_manifest()
    runtime_raw, conversion_raw, _manifest, urdf_path = _read_official_inputs()
    urdf_raw = urdf_path.read_bytes()
    anchors = load_official_anchor_contract(conversion_raw)
    plans = plan_official_ik_matrix(urdf_raw, anchors)
    motion_plan_sha256 = revalidate_official_motion_plans(urdf_raw, anchors, plans)
    r1_plan = next(plan for plan in plans if plan.target_name == TARGET_NAME)
    if r1_plan.status != "COMPLETE" or len(r1_plan.branches) != 2:
        raise UpperBoundError(f"official R1 motion is unavailable: {r1_plan.failure_reason}")
    branch = r1_plan.branches[WRIST_INDEX]
    timeline = build_timeline(branch.trajectory_joint_positions)
    intrinsics = Intrinsics(
        fx=float(INTRINSICS["fx"]),
        fy=float(INTRINSICS["fy"]),
        cx=float(INTRINSICS["cx"]),
        cy=float(INTRINSICS["cy"]),
        width=WIDTH,
        height=HEIGHT,
        depth_scale=1000.0,
    )
    comparator = FreezeComparator(required_confirmation_frames=CONFIRMATION_FRAMES)
    target_index = POINT_NAMES.index(TARGET_NAME)
    rows: list[dict[str, object]] = []
    frames: list[np.ndarray] = []
    oracle_stream = hashlib.sha256()
    raw_rendered_input_stream = hashlib.sha256()
    evidence_stream = hashlib.sha256()

    with OfflineMeshScene.from_verified_conversion(
        CONVERSION_RELEASE, camera_pose_name=CAMERA_POSE_NAME
    ) as scene:
        for timeline_row in timeline:
            patient = np.eye(4, dtype=np.float64)
            patient[0, 3] = timeline_row.patient_translation_x_m
            scene.set_frame(timeline_row.left_joint_positions_rad, patient)
            projection = scene.project_current_anchors()
            products = scene.render()
            oracle_pixels = noisy_oracle_pixels(
                projection.projected_pixels, timeline_row.frame_index
            )
            oracle_stream.update(
                np.ascontiguousarray(oracle_pixels, dtype="<f8").tobytes()
            )
            target_pixel = oracle_pixels[target_index]
            candidate = trusted_depth_lift(
                pixel=target_pixel,
                depth_m=products.depth_m,
                robot_mask=products.robot_mask,
                human_mask=products.human_mask,
                intrinsics=intrinsics,
            )
            confirmed = False
            if timeline_row.phase == "confirm_home":
                if candidate is None:
                    raise UpperBoundError(
                        f"HOME reference candidate failed at frame {timeline_row.frame_index}"
                    )
                confirmed = comparator.observe(
                    candidate,
                    patient,
                    projection.world_from_cv_camera,
                )
            output = (
                comparator.predict(patient, projection.world_from_cv_camera)
                if comparator.ready
                else None
            )
            truth_xyz = np.asarray(
                projection.anchor_camera_position_m[target_index], dtype=np.float64
            )
            rendered_input_sha = _frame_digest(
                joints=timeline_row.left_joint_positions_rad,
                world_from_patient=patient,
                oracle_pixels=oracle_pixels,
                rgb=products.rgb,
                depth_m=products.depth_m,
                robot_mask=products.robot_mask,
                human_mask=products.human_mask,
            )
            raw_rendered_input_stream.update(bytes.fromhex(rendered_input_sha))
            camera_xyz = None if output is None else output.camera_lock_xyz_camera_m
            patient_xyz = None if output is None else output.patient_lock_xyz_camera_m
            row: dict[str, object] = {
                "frame_index": timeline_row.frame_index,
                "timestamp_s": timeline_row.timestamp_s,
                "phase": timeline_row.phase,
                "trajectory_index": timeline_row.trajectory_index,
                "patient_speed_m_s": timeline_row.patient_speed_m_s,
                "patient_translation_x_m": timeline_row.patient_translation_x_m,
                "robot_pixel_count": int(np.count_nonzero(products.robot_mask)),
                "target_robot_mask_5x5": _target_masked(
                    target_pixel, products.robot_mask
                ),
                "reference_candidate_valid": candidate is not None,
                "reference_candidate_x_camera_m": (
                    None if candidate is None else float(candidate[0])
                ),
                "reference_candidate_y_camera_m": (
                    None if candidate is None else float(candidate[1])
                ),
                "reference_candidate_z_camera_m": (
                    None if candidate is None else float(candidate[2])
                ),
                "reference_ready": comparator.ready,
                "reference_confirmed_this_frame": confirmed,
                "oracle_u_px": float(target_pixel[0]),
                "oracle_v_px": float(target_pixel[1]),
                "gt_x_camera_m": float(truth_xyz[0]),
                "gt_y_camera_m": float(truth_xyz[1]),
                "gt_z_camera_m": float(truth_xyz[2]),
                "camera_lock_valid": camera_xyz is not None,
                "camera_lock_x_camera_m": None if camera_xyz is None else float(camera_xyz[0]),
                "camera_lock_y_camera_m": None if camera_xyz is None else float(camera_xyz[1]),
                "camera_lock_z_camera_m": None if camera_xyz is None else float(camera_xyz[2]),
                "camera_lock_error_mm": _error_mm(camera_xyz, truth_xyz),
                "patient_lock_valid": patient_xyz is not None,
                "patient_lock_x_camera_m": None if patient_xyz is None else float(patient_xyz[0]),
                "patient_lock_y_camera_m": None if patient_xyz is None else float(patient_xyz[1]),
                "patient_lock_z_camera_m": None if patient_xyz is None else float(patient_xyz[2]),
                "patient_lock_error_mm": _error_mm(patient_xyz, truth_xyz),
                "rendered_input_sha256": rendered_input_sha,
            }
            frame_sha = semantic_frame_sha256(row)
            row["frame_evidence_sha256"] = frame_sha
            evidence_stream.update(bytes.fromhex(frame_sha))
            rows.append(row)
            frames.append(
                overlay_frame(
                    rgb=products.rgb,
                    robot_mask=products.robot_mask,
                    oracle_pixels=oracle_pixels,
                    truth_pixel=projection.projected_pixels[target_index],
                    camera_lock_xyz=camera_xyz,
                    patient_lock_xyz=patient_xyz,
                    truth_xyz=truth_xyz,
                    intrinsics=intrinsics,
                    phase=timeline_row.phase,
                    timestamp_s=timeline_row.timestamp_s,
                )
            )
    if comparator.reference_frame_index != CONFIRMATION_FRAMES - 1:
        raise UpperBoundError("reference did not confirm on the fixed twelfth frame")
    row_metrics = derive_row_metrics(rows)
    metrics = base_metrics()
    metrics.update(row_metrics)
    metrics.update(derive_nonrow_metrics(rows))
    sources_after = bound_source_manifest()
    if sources_after != sources_before:
        raise UpperBoundError("execution-critical source changed during the run")
    inputs = _input_hashes(
        runtime_raw=runtime_raw,
        conversion_raw=conversion_raw,
        urdf_path=urdf_path,
        motion_plan_sha256=motion_plan_sha256,
        oracle_stream_sha256=oracle_stream.hexdigest(),
        raw_rendered_input_stream_sha256=raw_rendered_input_stream.hexdigest(),
        evidence_stream_sha256=evidence_stream.hexdigest(),
        sources=sources_after,
    )
    report_index = CONFIRMATION_FRAMES + 61 + CONTACT_HOLD_FRAMES - 1
    return rows, frames, report_index, metrics, inputs


def run(destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    existing = [path.name for path in destination_path.iterdir()]
    if any(name != "STAGING_WARNING" for name in existing):
        raise UpperBoundError("staging destination already contains artifacts")
    (destination_path / "STAGING_WARNING").write_text(
        STAGING_WARNING_TEXT, encoding="utf-8"
    )
    scratch = Path(
        tempfile.mkdtemp(prefix=".patient-frame-upper-bound-", dir=destination_path.parent)
    )
    try:
        rows, frames, report_index, metrics, inputs = _simulate()
        validate_metrics_contract(metrics)
        validate_bound_source_inputs(inputs)
        if metrics.get("meaningful_staging_evidence") is not True:
            raise UpperBoundError("structural staging-evidence gate did not pass")
        video_path = scratch / "patient_frame_upper_bound_h264.mp4"
        _encode_video(frames, video_path)
        metrics["video_contract"] = _probe_video(video_path, len(frames))
        metrics["report_frame_index"] = report_index
        report_bgr = cv2.cvtColor(frames[report_index], cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(scratch / "report_frame.png"), report_bgr):
            raise UpperBoundError("report frame could not be written")
        csv_raw = per_frame_csv_bytes(rows)
        verify_metrics_against_csv(metrics, csv_raw, inputs)
        (scratch / "per_frame.csv").write_bytes(csv_raw)
        (scratch / "input_hashes.json").write_bytes(_json_bytes(inputs))
        (scratch / "metrics.json").write_bytes(_json_bytes(metrics))
        (scratch / "README.md").write_text(_readme(metrics), encoding="utf-8")
        persisted_metrics = json.loads((scratch / "metrics.json").read_text())
        persisted_inputs = json.loads((scratch / "input_hashes.json").read_text())
        validate_metrics_contract(persisted_metrics)
        validate_bound_source_inputs(persisted_inputs)
        verify_metrics_against_csv(
            persisted_metrics,
            (scratch / "per_frame.csv").read_bytes(),
            persisted_inputs,
        )
        publish_staging_artifacts(scratch=scratch, destination=destination_path)
        return metrics
    except BaseException as exc:
        _clear_except_warning(destination_path)
        (destination_path / "STAGING_WARNING").write_text(
            STAGING_WARNING_TEXT
            + f"Fail-closed reason: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    metrics = run(args.output)
    print(
        json.dumps(
            {
                "completion": False,
                "meaningful_staging_evidence": metrics["meaningful_staging_evidence"],
                "output": str(args.output),
                "production_activation": False,
                "recommendation_supported": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
