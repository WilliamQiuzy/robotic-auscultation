#!/usr/bin/env python3
"""Non-release Mac diagnostic for exact robot-mask depth rejection.

This intentionally runs an oracle-2D, official-mesh R1/wrist-0 comparison.  It
does not run v9, Isaac, hardware, calibration, or any production activation.
No completion marker or current pointer is ever written.
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

from ausc.perception.tracking.detector import DetectorResult, LUNG_ORDER  # noqa: E402
from ausc.perception.tracking.geometry import CropTransform, Intrinsics  # noqa: E402
from ausc.perception.tracking.model_tracker import (  # noqa: E402
    ModelBasedTracker,
    ModelConfig,
    STATUS_COASTING,
    STATUS_TRACKING,
)
from ausc.perception.tracking.offline_mesh_simulation import (  # noqa: E402
    EXPECTED_RUNTIME_MANIFEST_SHA256,
    EXPECTED_SCENE_CONTRACT_SHA256,
    HEIGHT,
    INTRINSICS,
    OfflineMeshScene,
    WIDTH,
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


class DiagnosticError(RuntimeError):
    """The intentionally non-release diagnostic failed closed."""


SCHEMA = "wam_offline_robot_mask_diagnostic_v3"
INPUT_SCHEMA = "wam_offline_robot_mask_diagnostic_inputs_v3"
CONVERSION_RELEASE_ID = "official_meshes-b5e826af2539bceafea1"
ASSET_RELEASE_ID = "official_assets-05e66c94cd8b36ff79be"
TARGET_NAME = "R1"
WRIST_INDEX = 0
CAMERA_POSE_NAME = "primary"
FPS = 20
INIT_FRAMES = 12
CONTACT_HOLD_FRAMES = 10
CONTACT_TRAJECTORY_INDEX = 60
MAX_COAST_AGE_S = 0.40
MAX_COAST_BOUND_M = 0.010
PATIENT_MOTION_BOUND_M_S = 0.013
PATIENT_TRANSLATION_SPEED_M_S = 0.012
ORACLE_NOISE_STD_PX = 1.0
ORACLE_NOISE_SEED_ROOT = 91020260715
BANNER = "ORACLE 2D / NOT V9 / NOMINAL ROOT / NOT ISAAC"
METHOD_BANNER = "SIM PATIENT TF / METHOD UPPER BOUND ONLY"
HUMAN_BANNER = "OFFICIAL NVIDIA HUMAN / DEFAULT REST POSE"
OUTCOME_BANNER = "NEGATIVE: RESET / NO RECOVERY / NOT RECOMMENDED"
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
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "docs/runs/2026-07-16-offline-robot-mask-negative-v3"
)
TRACKING_ROOT = AUSCULTATION_ROOT / "ausc/perception/tracking"
BOUND_SOURCE_FILES = {
    "diagnostic_runner": Path(__file__).resolve(),
    "ausc_init": AUSCULTATION_ROOT / "ausc/__init__.py",
    "tracking_init": TRACKING_ROOT / "__init__.py",
    "canonical": TRACKING_ROOT / "canonical.py",
    "detector": TRACKING_ROOT / "detector.py",
    "ekf": TRACKING_ROOT / "ekf.py",
    "experimental_visibility_geometry": TRACKING_ROOT
    / "experimental_visibility_geometry.py",
    "flow": TRACKING_ROOT / "flow.py",
    "geometry": TRACKING_ROOT / "geometry.py",
    "intrinsics": TRACKING_ROOT / "intrinsics.py",
    "model_tracker": TRACKING_ROOT / "model_tracker.py",
    "official_asset_contract": TRACKING_ROOT / "official_asset_contract.py",
    "official_openarm_experiment": TRACKING_ROOT / "official_openarm_experiment.py",
    "offline_mesh_simulation": TRACKING_ROOT / "offline_mesh_simulation.py",
    "offline_official_mesh_assets": TRACKING_ROOT
    / "offline_official_mesh_assets.py",
    "offline_official_mesh_experiment": TRACKING_ROOT
    / "offline_official_mesh_experiment.py",
    "rgbd_odometry": TRACKING_ROOT / "rgbd_odometry.py",
    "runtime": TRACKING_ROOT / "runtime.py",
    "tracker": TRACKING_ROOT / "tracker.py",
}
RUNTIME_MANIFEST = (
    WORKSPACE_ROOT
    / "docs/runs/2026-07-15-offline-official-mesh-occlusion/raw/runtime_manifest.json"
)
STAGING_WARNING_TEXT = (
    "NON-RELEASE DIAGNOSTIC ONLY. completion=false; production_activation=false.\n"
    "This directory deliberately has no completion marker and no current pointer.\n"
)

REQUIRED_ARTIFACTS = (
    "README.md",
    "input_hashes.json",
    "metrics.json",
    "per_frame.csv",
    "report_frame.png",
    "robot_mask_diagnostic_h264.mp4",
)

PER_FRAME_FIELDS = (
    "frame_index",
    "timestamp_s",
    "phase",
    "trajectory_index",
    "patient_translation_x_m",
    "robot_pixel_count",
    "target_robot_mask_5x5",
    "masked_association_point_count",
    "masked_association_points",
    "baseline_initialized",
    "candidate_initialized",
    "baseline_measured",
    "candidate_raw_measured",
    "baseline_status",
    "candidate_raw_status",
    "candidate_status",
    "candidate_reason",
    "candidate_coast_age_s",
    "candidate_bound_m",
    "oracle_u_px",
    "oracle_v_px",
    "gt_x_camera_m",
    "gt_y_camera_m",
    "gt_z_camera_m",
    "baseline_x_camera_m",
    "baseline_y_camera_m",
    "baseline_z_camera_m",
    "baseline_error_mm",
    "candidate_raw_x_camera_m",
    "candidate_raw_y_camera_m",
    "candidate_raw_z_camera_m",
    "candidate_raw_error_mm",
    "candidate_x_camera_m",
    "candidate_y_camera_m",
    "candidate_z_camera_m",
    "candidate_error_mm",
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


def _bound_source_manifest() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, path in BOUND_SOURCE_FILES.items():
        if not path.is_file() or path.is_symlink():
            raise DiagnosticError(f"bound source file is unavailable: {name}")
        result[name] = {
            "path": str(path.relative_to(WORKSPACE_ROOT)),
            "sha256": _sha256_file(path),
        }
    return result


def validate_bound_source_inputs(inputs: Mapping[str, object]) -> None:
    """Fail if persisted execution-source bindings differ from current bytes."""

    if inputs.get("schema") != INPUT_SCHEMA:
        raise DiagnosticError("input schema is not the fixed negative-v3 schema")
    declared = inputs.get("bound_source_files")
    if not isinstance(declared, Mapping):
        raise DiagnosticError("bound source binding is unavailable")
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
        raise DiagnosticError("bound source binding is malformed") from exc
    if normalized != _bound_source_manifest():
        raise DiagnosticError("bound source binding changed after execution")
    runner = normalized.get("diagnostic_runner")
    if (
        runner is None
        or inputs.get("diagnostic_script_sha256") != runner.get("sha256")
    ):
        raise DiagnosticError("persisted diagnostic runner SHA is not source-bound")


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


def static_contract() -> dict[str, object]:
    """Return the immutable scope disclaimer carried by every output."""

    return {
        "schema": SCHEMA,
        "oracle_2d": True,
        "real_v9_executed": False,
        "nominal_root_not_commissioned": True,
        "production_activation": False,
        "completion": False,
        "target_name": TARGET_NAME,
        "wrist_index": WRIST_INDEX,
        "camera_pose_name": CAMERA_POSE_NAME,
        "banner": BANNER,
        "method_banner": METHOD_BANNER,
        "human_banner": HUMAN_BANNER,
        "outcome_banner": OUTCOME_BANNER,
        "conversion_release_id": CONVERSION_RELEASE_ID,
        "asset_release_id": ASSET_RELEASE_ID,
        "simulator": "MuJoCo_kinematic_official_mesh_not_Isaac",
        "isaac_equivalence": False,
        "isaac_surface_equivalence": False,
        "human_pose": "default_equals_rest",
        "human_rest_pose": True,
        "scope": "R1_wrist0_representative_only",
        "anchor_three_d_gt_in_trackers": False,
        "gt_depth_in_tracker": False,
        "robot_hidden_for_baseline": False,
        "baseline_robot_mask_argument": None,
        "candidate_robot_mask_argument": "exact_simulator_robot_mask",
        "robot_mask_source": "exact_simulator_instance_segmentation",
        "robot_mask_provider_commissioned": False,
        "mask_false_negative_sweep_executed": False,
        "stable_wrong_surface_defense_validated": False,
        "production_robot_mask_call_path_executed": False,
        "patient_transform_source": "simulator_ground_truth",
        "patient_transform_method_sensitivity_upper_bound_only": True,
        "patient_transform_available_in_production": False,
        "simulator_patient_transform_used_by_bounded_wrapper": True,
        "production_patient_pose_requirement": (
            "tracker_or_external_estimator_plus_latency_and_hand_eye_validation"
        ),
        "display_policy": (
            "mask_tracker_plus_simulator_gt_patient_relative_bounded_coast"
        ),
        "outcome_class": "NEGATIVE_RESET_NO_RECOVERY",
        "recommendation_supported": False,
    }


def base_metrics() -> dict[str, object]:
    result = static_contract()
    result.update(
        {
            "frame_rate_hz": FPS,
            "frame_period_s": FRAME_PERIOD_S,
            "initialization_frame_count": INIT_FRAMES,
            "contact_hold_frame_count": CONTACT_HOLD_FRAMES,
            "contact_hold_duration_s": CONTACT_HOLD_FRAMES * FRAME_PERIOD_S,
            "bounded_coast_max_age_s": MAX_COAST_AGE_S,
            "bounded_coast_max_uncertainty_m": MAX_COAST_BOUND_M,
            "patient_motion_bound_m_s": PATIENT_MOTION_BOUND_M_S,
            "oracle_2d_noise_std_px": ORACLE_NOISE_STD_PX,
            "oracle_2d_noise_seed_root": ORACLE_NOISE_SEED_ROOT,
            "meaningful_diagnostic": False,
        }
    )
    return result


def validate_metrics_contract(metrics: Mapping[str, object]) -> None:
    """Fail if a diagnostic artifact makes a production/completion claim."""

    for key, expected in static_contract().items():
        actual = metrics.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise DiagnosticError(f"{key} violates the fixed diagnostic contract")
    meaningful = metrics.get("meaningful_diagnostic")
    if type(meaningful) is not bool:
        raise DiagnosticError("meaningful_diagnostic must be a boolean")
    if meaningful and not negative_reset_no_recovery_is_meaningful(metrics):
        raise DiagnosticError(
            "meaningful_diagnostic lacks the declared negative reset/no-recovery evidence"
        )
    if (
        "negative_reset_no_recovery_observed" in metrics
        and metrics.get("negative_reset_no_recovery_observed") is not meaningful
    ):
        raise DiagnosticError("negative outcome observation disagrees with meaningful gate")


@dataclass(frozen=True)
class TimelineFrame:
    frame_index: int
    timestamp_s: float
    phase: str
    trajectory_index: int | None
    left_joint_positions_rad: np.ndarray
    patient_translation_x_m: float


def build_timeline(trajectory_joint_positions: np.ndarray) -> tuple[TimelineFrame, ...]:
    """Build 12 HOME init frames plus official 20 Hz motion and 0.5 s contact hold."""

    trajectory = np.asarray(trajectory_joint_positions, dtype=np.float64)
    if trajectory.shape != (121, 7) or not np.isfinite(trajectory).all():
        raise DiagnosticError("official R1/w0 trajectory must be exactly 121x7")
    rows: list[tuple[str, int | None, np.ndarray, float]] = []
    for _ in range(INIT_FRAMES):
        rows.append(("init_home", None, HOME_JOINT_POSITIONS_RAD, 0.0))
    for index in range(CONTACT_TRAJECTORY_INDEX + 1):
        rows.append(("approach", index, trajectory[index], 0.0))
    for hold_index in range(1, CONTACT_HOLD_FRAMES + 1):
        elapsed = hold_index * FRAME_PERIOD_S
        rows.append(
            (
                "contact_hold",
                CONTACT_TRAJECTORY_INDEX,
                trajectory[CONTACT_TRAJECTORY_INDEX],
                PATIENT_TRANSLATION_SPEED_M_S * elapsed,
            )
        )
    final_patient_offset = (
        PATIENT_TRANSLATION_SPEED_M_S * CONTACT_HOLD_FRAMES * FRAME_PERIOD_S
    )
    for index in range(CONTACT_TRAJECTORY_INDEX + 1, len(trajectory)):
        rows.append(("depart", index, trajectory[index], final_patient_offset))
    return tuple(
        TimelineFrame(
            frame_index=index,
            timestamp_s=index * FRAME_PERIOD_S,
            phase=phase,
            trajectory_index=trajectory_index,
            left_joint_positions_rad=np.asarray(joints, dtype=np.float64).copy(),
            patient_translation_x_m=float(patient_offset),
        )
        for index, (phase, trajectory_index, joints, patient_offset) in enumerate(rows)
    )


def noisy_oracle_pixels(projected_pixels: np.ndarray, frame_index: int) -> np.ndarray:
    """Add a deterministic clipped 1 px oracle-2D perturbation."""

    projected = np.asarray(projected_pixels, dtype=np.float64)
    if projected.shape != (len(POINT_NAMES), 2) or not np.isfinite(projected).all():
        raise DiagnosticError("oracle projection is invalid")
    if type(frame_index) is not int or frame_index < 0:
        raise DiagnosticError("frame index is invalid")
    normalized = np.clip(
        np.random.default_rng(ORACLE_NOISE_SEED_ROOT + frame_index).standard_normal(
            projected.shape
        ),
        -2.5,
        2.5,
    )
    return projected + ORACLE_NOISE_STD_PX * normalized


@dataclass(frozen=True)
class TrackerTargetSample:
    measured: bool
    xyz_camera_m: np.ndarray
    position_std_m: float


@dataclass(frozen=True)
class BoundedTargetOutput:
    status: str
    valid: bool
    xyz_camera_m: np.ndarray | None
    reason: str
    age_s: float | None
    bound_m: float | None


def _rigid_transform(value: np.ndarray, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-10)
    ):
        raise DiagnosticError(f"{label} is not a rigid transform")
    return transform


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise DiagnosticError("point is invalid")
    return transform[:3, :3] @ value + transform[:3, 3]


class PatientRelativeBoundedCoast:
    """Coast from the last measured patient-relative point, never raw tracker coast."""

    def __init__(
        self,
        *,
        max_age_s: float = MAX_COAST_AGE_S,
        max_bound_m: float = MAX_COAST_BOUND_M,
        drift_bound_m_s: float = PATIENT_MOTION_BOUND_M_S,
    ) -> None:
        self.max_age_s = float(max_age_s)
        self.max_bound_m = float(max_bound_m)
        self.drift_bound_m_s = float(drift_bound_m_s)
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.max_age_s, self.max_bound_m, self.drift_bound_m_s)
        ):
            raise DiagnosticError("bounded coast limits are invalid")
        self._patient_point_m: np.ndarray | None = None
        self._measured_time_s: float | None = None
        self._measured_std_m: float | None = None

    def step(
        self,
        *,
        timestamp_s: float,
        sample: TrackerTargetSample,
        world_from_patient: np.ndarray,
        world_from_cv_camera: np.ndarray,
    ) -> BoundedTargetOutput:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise DiagnosticError("timestamp is invalid")
        patient = _rigid_transform(world_from_patient, "world_from_patient")
        camera = _rigid_transform(world_from_cv_camera, "world_from_cv_camera")
        xyz = np.asarray(sample.xyz_camera_m, dtype=np.float64)
        std = float(sample.position_std_m)
        if sample.measured:
            if xyz.shape != (3,) or not np.isfinite(xyz).all() or not math.isfinite(std) or std <= 0.0:
                raise DiagnosticError("measured target sample is invalid")
            world_point = _transform_point(camera, xyz)
            patient_point = _transform_point(np.linalg.inv(patient), world_point)
            self._patient_point_m = patient_point
            self._measured_time_s = timestamp
            self._measured_std_m = std
            return BoundedTargetOutput(
                status="MEASURED",
                valid=True,
                xyz_camera_m=xyz.copy(),
                reason="current_measurement",
                age_s=0.0,
                bound_m=std,
            )
        if (
            self._patient_point_m is None
            or self._measured_time_s is None
            or self._measured_std_m is None
        ):
            return BoundedTargetOutput(
                status="LOST",
                valid=False,
                xyz_camera_m=None,
                reason="no_measured_reference",
                age_s=None,
                bound_m=None,
            )
        age = timestamp - self._measured_time_s
        if age < -1.0e-12:
            raise DiagnosticError("timestamps are not monotonic")
        bound = self._measured_std_m + max(age, 0.0) * self.drift_bound_m_s
        if age > self.max_age_s + 1.0e-12:
            return BoundedTargetOutput(
                status="LOST",
                valid=False,
                xyz_camera_m=None,
                reason="age_bound_exceeded",
                age_s=age,
                bound_m=bound,
            )
        if bound > self.max_bound_m + 1.0e-12:
            return BoundedTargetOutput(
                status="LOST",
                valid=False,
                xyz_camera_m=None,
                reason="uncertainty_bound_exceeded",
                age_s=age,
                bound_m=bound,
            )
        world_point = _transform_point(patient, self._patient_point_m)
        camera_point = _transform_point(np.linalg.inv(camera), world_point)
        return BoundedTargetOutput(
            status="PATIENT_RELATIVE_COAST",
            valid=True,
            xyz_camera_m=camera_point,
            reason="bounded_patient_relative_prediction",
            age_s=age,
            bound_m=bound,
        )


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
    """Serialize rows with a fixed schema and stable float formatting."""

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
    """Bind one raw-render digest to every persisted non-self CSV field."""

    raw_sha = str(row.get("rendered_input_sha256", ""))
    if len(raw_sha) != 64:
        raise DiagnosticError("rendered input SHA is invalid")
    try:
        bytes.fromhex(raw_sha)
    except ValueError as exc:
        raise DiagnosticError("rendered input SHA is invalid") from exc
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
    return _sha256_bytes(b"wam_robot_mask_semantic_row_v1\0" + raw)


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


def parse_tool_version(raw: str) -> str:
    """Return the exact first non-empty version line for an input manifest."""

    if not isinstance(raw, str):
        raise DiagnosticError("tool version output is invalid")
    for line in raw.splitlines():
        value = line.strip()
        if value:
            return value
    raise DiagnosticError("tool version output is empty")


def _tool_version(path: str) -> str:
    result = subprocess.run(
        [path, "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise DiagnosticError(f"tool version probe failed: {path}")
    return parse_tool_version(result.stdout)


def _encode_video(frames: Iterable[np.ndarray], destination: Path) -> None:
    if not Path(FFMPEG).is_file():
        raise DiagnosticError("locked Homebrew ffmpeg is unavailable")
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
                raise DiagnosticError(f"video frame {index} violates 640x360 RGB8")
            process.stdin.write(np.ascontiguousarray(value).tobytes(order="C"))
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait(timeout=120)
    except BaseException:
        process.kill()
        process.wait()
        raise
    if returncode != 0:
        raise DiagnosticError(f"ffmpeg failed: {stderr.strip()}")


def _probe_video(path: Path, expected_frames: int) -> dict[str, object]:
    if not Path(FFPROBE).is_file():
        raise DiagnosticError("locked Homebrew ffprobe is unavailable")
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
        raise DiagnosticError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        streams = json.loads(result.stdout)["streams"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DiagnosticError("ffprobe output is invalid") from exc
    video = [row for row in streams if row.get("codec_type") == "video"]
    audio = [row for row in streams if row.get("codec_type") == "audio"]
    if len(video) != 1 or audio:
        raise DiagnosticError("diagnostic video stream inventory changed")
    stream = video[0]
    expected = {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": WIDTH,
        "height": HEIGHT,
        "r_frame_rate": f"{FPS}/1",
        "nb_read_frames": str(expected_frames),
    }
    for key, value in expected.items():
        if stream.get(key) != value:
            raise DiagnosticError(f"diagnostic video {key} changed")
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


def publish_nonrelease_artifacts(
    *, scratch: Path, destination: Path, meaningful: bool
) -> None:
    """Publish a meaningful diagnostic without ever writing completion state."""

    scratch_path = Path(scratch)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    warning = destination_path / "STAGING_WARNING"
    if not warning.exists():
        warning.write_text(STAGING_WARNING_TEXT, encoding="utf-8")
    if not meaningful:
        _clear_except_warning(destination_path)
        raise DiagnosticError("diagnostic is not meaningful; artifacts were withheld")
    unexpected = [
        path.name
        for path in destination_path.iterdir()
        if path.name != "STAGING_WARNING"
    ]
    if unexpected:
        raise DiagnosticError("destination already contains non-staging artifacts")
    for name in REQUIRED_ARTIFACTS:
        source = scratch_path / name
        if not source.is_file() or source.is_symlink():
            raise DiagnosticError(f"required artifact is missing: {name}")
    for name in REQUIRED_ARTIFACTS:
        os.replace(scratch_path / name, destination_path / name)
    sums = "".join(
        f"{_sha256_file(destination_path / name)}  {name}\n"
        for name in sorted(REQUIRED_ARTIFACTS)
    )
    (destination_path / "SHA256SUMS").write_text(sums, encoding="utf-8")
    forbidden = (
        destination_path / "COMPLETED",
        destination_path / "COMPLETE",
        destination_path / "COMPLETION.json",
        destination_path / "current",
    )
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise DiagnosticError("completion marker or current pointer is forbidden")


class _OracleDetector:
    """One shared, declared oracle-2D source; it never receives 3-D truth."""

    def __init__(self) -> None:
        self._pixels: np.ndarray | None = None
        self._crop = CropTransform(x0=0, y0=0, side=WIDTH, res=WIDTH)

    def set_pixels(self, pixels: np.ndarray) -> None:
        value = np.asarray(pixels, dtype=np.float64)
        if value.shape != (len(POINT_NAMES), 2) or not np.isfinite(value).all():
            raise DiagnosticError("oracle detector pixels are invalid")
        self._pixels = value.copy()

    def detect(self, rgb_full: np.ndarray, depth=None, intr=None) -> DetectorResult:
        if self._pixels is None:
            raise DiagnosticError("oracle pixels were not set")
        return DetectorResult(
            xy_det=self._pixels.copy(),
            crop=self._crop,
            confidence=1.0,
            back_present=True,
            order=tuple(LUNG_ORDER),
        )


def _association_blocked_points(
    pixels: np.ndarray, robot_mask: np.ndarray, radius_px: int = 2
) -> tuple[str, ...]:
    height, width = robot_mask.shape
    blocked: list[str] = []
    for name, (u, v) in zip(POINT_NAMES, pixels, strict=True):
        ui, vi = int(round(float(u))), int(round(float(v)))
        if not (0 <= ui < width and 0 <= vi < height):
            continue
        x0, x1 = max(0, ui - radius_px), min(width, ui + radius_px + 1)
        y0, y1 = max(0, vi - radius_px), min(height, vi + radius_px + 1)
        if np.asarray(robot_mask[y0:y1, x0:x1], dtype=bool).any():
            blocked.append(name)
    return tuple(blocked)


def _xyz_or_none(point: object, initialized: bool) -> np.ndarray | None:
    if not initialized or getattr(point, "status", None) not in {
        STATUS_TRACKING,
        STATUS_COASTING,
    }:
        return None
    value = np.asarray(getattr(point, "xyz_camera"), dtype=np.float64)
    return value.copy() if value.shape == (3,) and np.isfinite(value).all() else None


def _error_mm(estimate: np.ndarray | None, truth: np.ndarray) -> float | None:
    if estimate is None:
        return None
    return float(np.linalg.norm(estimate - truth) * 1000.0)


def _frame_digest(
    *,
    joints: np.ndarray,
    world_from_patient: np.ndarray,
    oracle_pixels: np.ndarray,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    robot_mask: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for label, value, dtype in (
        (b"joints\0", joints, "<f8"),
        (b"patient\0", world_from_patient, "<f8"),
        (b"oracle\0", oracle_pixels, "<f8"),
        (b"rgb\0", rgb, np.uint8),
        (b"depth\0", depth_m, "<f8"),
        (b"robot\0", robot_mask, np.uint8),
    ):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(label)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _draw_cross(frame: np.ndarray, pixel: np.ndarray, colour: tuple[int, int, int], size: int) -> None:
    if pixel is None or not np.isfinite(pixel).all():
        return
    x, y = int(round(float(pixel[0]))), int(round(float(pixel[1])))
    if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
        return
    cv2.line(frame, (x - size, y), (x + size, y), colour, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - size), (x, y + size), colour, 2, cv2.LINE_AA)


def _overlay_frame(
    *,
    rgb: np.ndarray,
    robot_mask: np.ndarray,
    oracle_pixels: np.ndarray,
    truth_pixel: np.ndarray,
    baseline_xyz: np.ndarray | None,
    candidate: BoundedTargetOutput,
    intrinsics: Intrinsics,
    phase: str,
    timestamp_s: float,
) -> np.ndarray:
    frame = np.asarray(rgb, dtype=np.uint8).copy()
    robot = np.asarray(robot_mask, dtype=bool)
    if robot.any():
        tint = np.asarray([255.0, 38.0, 38.0])
        frame[robot] = np.clip(0.72 * frame[robot] + 0.28 * tint, 0, 255).astype(np.uint8)
    for pixel in oracle_pixels:
        if np.isfinite(pixel).all():
            cv2.circle(
                frame,
                (int(round(float(pixel[0]))), int(round(float(pixel[1])))),
                2,
                (255, 220, 70),
                -1,
                cv2.LINE_AA,
            )
    _draw_cross(frame, truth_pixel, (55, 255, 90), 7)
    if baseline_xyz is not None:
        baseline_pixel = intrinsics.project(baseline_xyz[None])[0]
        cv2.circle(
            frame,
            (int(round(float(baseline_pixel[0]))), int(round(float(baseline_pixel[1])))),
            7,
            (255, 70, 225),
            2,
            cv2.LINE_AA,
        )
    if candidate.valid and candidate.xyz_camera_m is not None:
        candidate_pixel = intrinsics.project(candidate.xyz_camera_m[None])[0]
        _draw_cross(frame, candidate_pixel, (45, 225, 255), 5)
    top = frame[:67].copy()
    frame[:67] = (0.18 * top).astype(np.uint8)
    cv2.putText(
        frame,
        BANNER,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        METHOD_BANNER,
        (8, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 210, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        HUMAN_BANNER,
        (8, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (150, 220, 255),
        1,
        cv2.LINE_AA,
    )
    bottom = frame[HEIGHT - 67 :].copy()
    frame[HEIGHT - 67 :] = (0.22 * bottom).astype(np.uint8)
    cv2.putText(
        frame,
        OUTCOME_BANNER,
        (8, HEIGHT - 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (255, 190, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"R1/w0  {phase}  t={timestamp_s:.2f}s  MASK={candidate.status}",
        (8, HEIGHT - 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "GT + green | no-mask baseline magenta | bounded mask output cyan",
        (8, HEIGHT - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    return frame


def _percentile(values: Sequence[object], percentile: float) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.percentile(finite, percentile)) if finite else None


def _row_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in ("true", "false"):
        return value == "true"
    raise DiagnosticError(f"CSV boolean is invalid: {value!r}")


def _row_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DiagnosticError(f"CSV float is invalid: {value!r}") from exc
    if not math.isfinite(result):
        raise DiagnosticError("CSV float is nonfinite")
    return result


def validate_scored_point_semantics(rows: Sequence[Mapping[str, object]]) -> None:
    """Require every scored coordinate to agree with its explicit validity state."""

    raw_specs = (
        (
            "baseline raw",
            "baseline_initialized",
            "baseline_measured",
            "baseline_status",
            "baseline",
        ),
        (
            "candidate raw",
            "candidate_initialized",
            "candidate_raw_measured",
            "candidate_raw_status",
            "candidate_raw",
        ),
    )
    for row in rows:
        truth = np.asarray(
            [
                _row_optional_float(row.get("gt_x_camera_m")),
                _row_optional_float(row.get("gt_y_camera_m")),
                _row_optional_float(row.get("gt_z_camera_m")),
            ],
            dtype=object,
        )
        if any(value is None for value in truth):
            raise DiagnosticError("ground-truth scored coordinates are incomplete")
        truth_xyz = np.asarray(truth, dtype=np.float64)
        for label, initialized_key, measured_key, status_key, prefix in raw_specs:
            initialized = _row_bool(row.get(initialized_key))
            measured = _row_bool(row.get(measured_key))
            status = str(row.get(status_key))
            if status not in {"INIT", "TRACKING", "COASTING", "LOST"}:
                raise DiagnosticError(f"{label} status is invalid: {status!r}")
            if measured != (status == "TRACKING"):
                raise DiagnosticError(f"{label} measured flag disagrees with {status}")
            valid = initialized and status in {"TRACKING", "COASTING"}
            if status in {"TRACKING", "COASTING"} and not initialized:
                raise DiagnosticError(f"{label} {status} requires initialization")
            xyz = [
                _row_optional_float(row.get(f"{prefix}_{axis}_camera_m"))
                for axis in ("x", "y", "z")
            ]
            error = _row_optional_float(row.get(f"{prefix}_error_mm"))
            if not valid:
                if any(value is not None for value in (*xyz, error)):
                    raise DiagnosticError(
                        f"{label} {status} scored coordinates must be empty"
                    )
                continue
            if any(value is None for value in (*xyz, error)):
                raise DiagnosticError(f"{label} {status} scored coordinates are incomplete")
            recomputed = float(
                np.linalg.norm(np.asarray(xyz, dtype=np.float64) - truth_xyz) * 1000.0
            )
            if not math.isclose(
                recomputed, float(error), rel_tol=0.0, abs_tol=5.0e-6
            ):
                raise DiagnosticError(f"{label} scored error disagrees with coordinates")

        display_status = str(row.get("candidate_status"))
        if display_status not in {"MEASURED", "PATIENT_RELATIVE_COAST", "LOST"}:
            raise DiagnosticError(f"display status is invalid: {display_status!r}")
        display_xyz = [
            _row_optional_float(row.get(f"candidate_{axis}_camera_m"))
            for axis in ("x", "y", "z")
        ]
        display_error = _row_optional_float(row.get("candidate_error_mm"))
        display_valid = display_status in {"MEASURED", "PATIENT_RELATIVE_COAST"}
        if not display_valid:
            if any(value is not None for value in (*display_xyz, display_error)):
                raise DiagnosticError("display LOST scored coordinates must be empty")
            continue
        if any(value is None for value in (*display_xyz, display_error)):
            raise DiagnosticError(f"display {display_status} scored coordinates are incomplete")
        recomputed = float(
            np.linalg.norm(
                np.asarray(display_xyz, dtype=np.float64) - truth_xyz
            )
            * 1000.0
        )
        if not math.isclose(
            recomputed, float(display_error), rel_tol=0.0, abs_tol=5.0e-6
        ):
            raise DiagnosticError("display scored error disagrees with coordinates")


def candidate_lifecycle_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize tracker/display lifecycle only after the first real initialization."""

    first_initialized: int | None = None
    reset_frame: int | None = None
    reinitialized_frame: int | None = None
    previously_initialized = False
    for index, row in enumerate(rows):
        initialized = _row_bool(row.get("candidate_initialized"))
        if initialized and first_initialized is None:
            first_initialized = index
        if (
            first_initialized is not None
            and previously_initialized
            and not initialized
            and reset_frame is None
        ):
            reset_frame = int(row["frame_index"])
        if (
            reset_frame is not None
            and not previously_initialized
            and initialized
            and reinitialized_frame is None
        ):
            reinitialized_frame = int(row["frame_index"])
        previously_initialized = initialized
    if first_initialized is None:
        return {
            "candidate_ever_initialized": False,
            "candidate_first_initialized_frame_index": None,
            "candidate_final_initialized": False,
            "candidate_tracker_reset_frame_index": None,
            "candidate_tracker_reinitialized_frame_index": None,
            "candidate_first_lost_age_from_last_measurement_s": None,
            "candidate_display_lost_after_first_initialization_frame_count": 0,
            "candidate_not_measured_after_first_initialization_frame_count": 0,
        }
    active_rows = rows[first_initialized:]
    last_measured_time: float | None = None
    first_lost_age: float | None = None
    for row in active_rows:
        timestamp = float(row["timestamp_s"])
        if _row_bool(row.get("candidate_raw_measured")):
            last_measured_time = timestamp
        if (
            first_lost_age is None
            and row.get("candidate_status") == "LOST"
            and last_measured_time is not None
        ):
            first_lost_age = timestamp - last_measured_time
    return {
        "candidate_ever_initialized": True,
        "candidate_first_initialized_frame_index": int(
            active_rows[0]["frame_index"]
        ),
        "candidate_final_initialized": _row_bool(
            rows[-1].get("candidate_initialized")
        ),
        "candidate_tracker_reset_frame_index": reset_frame,
        "candidate_tracker_reinitialized_frame_index": reinitialized_frame,
        "candidate_first_lost_age_from_last_measurement_s": first_lost_age,
        "candidate_display_lost_after_first_initialization_frame_count": sum(
            row.get("candidate_status") == "LOST" for row in active_rows
        ),
        "candidate_not_measured_after_first_initialization_frame_count": sum(
            not _row_bool(row.get("candidate_raw_measured")) for row in active_rows
        ),
    }


def _false_valid_metrics(
    prefix: str, errors_mm: Sequence[float]
) -> dict[str, object]:
    count = len(errors_mm)
    result: dict[str, object] = {f"{prefix}_valid_frame_count": count}
    for threshold in (5, 10):
        failures = sum(value > threshold for value in errors_mm)
        result[f"{prefix}_false_valid_gt_{threshold}mm_count"] = failures
        result[f"{prefix}_false_valid_gt_{threshold}mm_rate"] = (
            failures / count if count else None
        )
    return result


def _raw_tracker_valid_error(
    row: Mapping[str, object],
    *,
    initialized_key: str,
    status_key: str,
    error_key: str,
) -> float | None:
    """Apply one identical raw-output validity policy to either tracker."""

    initialized = _row_bool(row.get(initialized_key))
    status = str(row.get(status_key))
    if status not in {"INIT", "TRACKING", "COASTING", "LOST"}:
        raise DiagnosticError(f"raw tracker status is invalid: {status!r}")
    error = _row_optional_float(row.get(error_key))
    valid = initialized and status in {"TRACKING", "COASTING"}
    if valid and error is None:
        raise DiagnosticError(f"valid raw tracker output lacks {error_key}")
    return error if valid else None


def negative_reset_no_recovery_is_meaningful(
    metrics: Mapping[str, object],
) -> bool:
    """Recognize only the declared negative reset/no-recovery diagnostic."""

    def positive_count(key: str) -> bool:
        value = metrics.get(key)
        return type(value) is int and value > 0

    reset_frame = metrics.get("candidate_tracker_reset_frame_index")
    return bool(
        metrics.get("baseline_initialized") is True
        and metrics.get("candidate_initialized") is True
        and positive_count("target_mask_raw_dropout_overlap_frame_count")
        and positive_count(
            "target_mask_raw_dropout_overlap_while_initialized_frame_count"
        )
        and positive_count("display_bounded_coast_frame_count")
        and positive_count("display_lost_frame_count")
        and type(reset_frame) is int
        and reset_frame >= 0
        and metrics.get("candidate_tracker_reinitialized_frame_index") is None
        and metrics.get("candidate_final_initialized") is False
    )


def derive_row_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Derive every result-window count from frame rows as the single source."""

    if not rows:
        raise DiagnosticError("per-frame result is empty")
    first_both: int | None = None
    for index, row in enumerate(rows):
        if _row_bool(row.get("baseline_initialized")) and _row_bool(
            row.get("candidate_initialized")
        ):
            first_both = index
            break
    if first_both is None:
        raise DiagnosticError("trackers never initialized together")
    result_rows = rows[first_both:]
    statuses = [str(row.get("candidate_status")) for row in result_rows]
    allowed = {"MEASURED", "PATIENT_RELATIVE_COAST", "LOST"}
    if any(status not in allowed for status in statuses):
        raise DiagnosticError("candidate status inventory changed")
    measured_count = statuses.count("MEASURED")
    coast_count = statuses.count("PATIENT_RELATIVE_COAST")
    lost_count = statuses.count("LOST")
    if measured_count + coast_count + lost_count != len(result_rows):
        raise DiagnosticError("candidate status partition is incomplete")
    raw_no_mask_by_row = [
        _raw_tracker_valid_error(
            row,
            initialized_key="baseline_initialized",
            status_key="baseline_status",
            error_key="baseline_error_mm",
        )
        for row in result_rows
    ]
    raw_mask_by_row = [
        _raw_tracker_valid_error(
            row,
            initialized_key="candidate_initialized",
            status_key="candidate_raw_status",
            error_key="candidate_raw_error_mm",
        )
        for row in result_rows
    ]
    raw_no_mask_errors = [value for value in raw_no_mask_by_row if value is not None]
    raw_mask_errors = [value for value in raw_mask_by_row if value is not None]
    common_pairs = [
        (no_mask, mask)
        for no_mask, mask in zip(raw_no_mask_by_row, raw_mask_by_row, strict=True)
        if no_mask is not None and mask is not None
    ]
    common_no_mask_errors = [pair[0] for pair in common_pairs]
    common_mask_errors = [pair[1] for pair in common_pairs]
    display_errors = [
        value
        for row in result_rows
        if str(row.get("candidate_status")) != "LOST"
        for value in [_row_optional_float(row.get("candidate_error_mm"))]
        if value is not None
    ]
    if len(display_errors) != measured_count + coast_count:
        raise DiagnosticError("valid display status lacks a scored output")
    denominator = len(result_rows)
    lifecycle = candidate_lifecycle_metrics(result_rows)
    home_rows = [row for row in rows if row.get("phase") == "init_home"]
    first_blocked = (
        tuple(
            value
            for value in str(home_rows[0].get("masked_association_points", "")).split(";")
            if value
        )
        if home_rows
        else ()
    )
    home_union = tuple(
        name
        for name in POINT_NAMES
        if any(
            name
            in {
                value
                for value in str(row.get("masked_association_points", "")).split(";")
                if value
            }
            for row in home_rows
        )
    )
    derived: dict[str, object] = {
        "frame_count": len(rows),
        "duration_s": float(rows[-1]["timestamp_s"]),
        "result_window_first_frame_index": int(
            result_rows[0]["frame_index"]
        ),
        "result_window_frame_count": denominator,
        "raw_tracker_result_window_frame_count": denominator,
        "preinitialization_display_lost_frame_count": sum(
            str(row.get("candidate_status")) == "LOST" for row in rows[:first_both]
        ),
        "baseline_initialized": True,
        "candidate_initialized": True,
        "baseline_final_initialized": _row_bool(
            rows[-1].get("baseline_initialized")
        ),
        "both_trackers_currently_initialized_frame_count": sum(
            _row_bool(row.get("baseline_initialized"))
            and _row_bool(row.get("candidate_initialized"))
            for row in result_rows
        ),
        "target_robot_mask_frame_count": sum(
            _row_bool(row.get("target_robot_mask_5x5")) for row in rows
        ),
        "raw_mask_unmeasured_frame_count": sum(
            not _row_bool(row.get("candidate_raw_measured")) for row in result_rows
        ),
        "raw_mask_unmeasured_while_tracker_initialized_frame_count": sum(
            _row_bool(row.get("candidate_initialized"))
            and not _row_bool(row.get("candidate_raw_measured"))
            for row in result_rows
        ),
        "target_mask_raw_dropout_overlap_frame_count": sum(
            _row_bool(row.get("target_robot_mask_5x5"))
            and not _row_bool(row.get("candidate_raw_measured"))
            for row in result_rows
        ),
        "target_mask_raw_dropout_overlap_while_initialized_frame_count": sum(
            _row_bool(row.get("target_robot_mask_5x5"))
            and _row_bool(row.get("candidate_initialized"))
            and not _row_bool(row.get("candidate_raw_measured"))
            for row in result_rows
        ),
        "raw_no_mask_valid_coverage": len(raw_no_mask_errors) / denominator,
        "raw_mask_valid_coverage": len(raw_mask_errors) / denominator,
        "raw_no_mask_error_median_mm": _percentile(raw_no_mask_errors, 50),
        "raw_no_mask_error_p95_mm": _percentile(raw_no_mask_errors, 95),
        "raw_mask_error_median_mm": _percentile(raw_mask_errors, 50),
        "raw_mask_error_p95_mm": _percentile(raw_mask_errors, 95),
        "raw_common_valid_frame_count": len(common_pairs),
        "raw_common_valid_coverage": len(common_pairs) / denominator,
        "raw_common_no_mask_error_median_mm": _percentile(
            common_no_mask_errors, 50
        ),
        "raw_common_no_mask_error_p95_mm": _percentile(common_no_mask_errors, 95),
        "raw_common_mask_error_median_mm": _percentile(common_mask_errors, 50),
        "raw_common_mask_error_p95_mm": _percentile(common_mask_errors, 95),
        "display_measured_frame_count": measured_count,
        "display_bounded_coast_frame_count": coast_count,
        "display_lost_frame_count": lost_count,
        "display_valid_coverage": len(display_errors) / denominator,
        "display_error_median_mm": _percentile(display_errors, 50),
        "display_error_p95_mm": _percentile(display_errors, 95),
        "nominal_home_exact_mask_blocked_points_first_frame": list(first_blocked),
        "nominal_home_exact_mask_blocked_points_union": list(home_union),
        "nominal_home_candidate_depth_association_point_count_first_frame": (
            len(POINT_NAMES) - len(first_blocked)
        ),
        "nominal_home_support_is_less_than_ten": bool(first_blocked),
        **_false_valid_metrics("raw_no_mask", raw_no_mask_errors),
        **_false_valid_metrics("raw_mask", raw_mask_errors),
        **_false_valid_metrics("display", display_errors),
        **lifecycle,
    }
    masked_no_mask = [
        value
        for row, value in zip(result_rows, raw_no_mask_by_row, strict=True)
        if _row_bool(row.get("target_robot_mask_5x5")) and value is not None
    ]
    derived["raw_no_mask_target_masked_error_p95_mm"] = _percentile(
        masked_no_mask, 95
    )
    derived["negative_reset_no_recovery_observed"] = (
        negative_reset_no_recovery_is_meaningful(derived)
    )
    return derived


def verify_metrics_against_csv(
    metrics: Mapping[str, object],
    csv_raw: bytes,
    input_hashes: Mapping[str, object],
) -> None:
    """Reverse-check derived metrics and the persisted frame-evidence stream."""

    if input_hashes.get("schema") != INPUT_SCHEMA:
        raise DiagnosticError("input schema is not the fixed negative-v3 schema")
    try:
        text = csv_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DiagnosticError("per-frame CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != PER_FRAME_FIELDS:
        raise DiagnosticError("per-frame CSV field order changed")
    parsed = list(reader)
    validate_scored_point_semantics(parsed)
    derived = derive_row_metrics(parsed)
    for key, expected in derived.items():
        actual = metrics.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(
                float(actual), expected, abs_tol=1.0e-6, rel_tol=0.0
            ):
                raise DiagnosticError(f"metrics CSV mismatch: {key}")
        elif actual != expected:
            raise DiagnosticError(f"metrics CSV mismatch: {key}")
    raw_stream = hashlib.sha256()
    evidence_stream = hashlib.sha256()
    for row in parsed:
        raw_value = row.get("rendered_input_sha256")
        value = row.get("frame_evidence_sha256")
        if (
            not isinstance(raw_value, str)
            or len(raw_value) != 64
            or any(character not in "0123456789abcdef" for character in raw_value)
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise DiagnosticError("CSV frame evidence digest is invalid")
        raw_stream.update(bytes.fromhex(raw_value))
        evidence_stream.update(bytes.fromhex(value))
        if semantic_frame_sha256(row) != value:
            raise DiagnosticError("CSV frame evidence does not bind row semantics")
    expected_raw_stream = input_hashes.get("raw_rendered_input_stream_sha256")
    if (
        not isinstance(expected_raw_stream, str)
        or len(expected_raw_stream) != 64
        or raw_stream.hexdigest() != expected_raw_stream
    ):
        raise DiagnosticError("CSV raw rendered-input stream does not match input hashes")
    expected_stream = input_hashes.get("rendered_evidence_stream_sha256")
    if (
        not isinstance(expected_stream, str)
        or len(expected_stream) != 64
        or evidence_stream.hexdigest() != expected_stream
    ):
        raise DiagnosticError("CSV frame evidence stream does not match input hashes")


def _read_official_inputs() -> tuple[bytes, bytes, dict[str, object], Path]:
    if platform.machine() != "arm64":
        raise DiagnosticError("diagnostic requires the locked arm64 Mac runtime")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise DiagnosticError(f"run with locked Python: {EXPECTED_PYTHON}")
    runtime_raw = RUNTIME_MANIFEST.read_bytes()
    if _sha256_bytes(runtime_raw) != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise DiagnosticError("locked Mac runtime manifest changed")
    conversion_raw = (CONVERSION_RELEASE / "conversion_manifest.json").read_bytes()
    if _sha256_bytes(conversion_raw) != EXPECTED_CONVERSION_MANIFEST_SHA256:
        raise DiagnosticError("official conversion manifest changed")
    manifest = verify_conversion_release(CONVERSION_RELEASE)
    if manifest.get("asset_release_id") != ASSET_RELEASE_ID:
        raise DiagnosticError("official asset release identity changed")
    urdf_path = ASSET_RELEASE / str(manifest["robot"]["urdf_path"])
    urdf_raw = urdf_path.read_bytes()
    if _sha256_bytes(urdf_raw) != OFFICIAL_CLEAN_URDF_SHA256:
        raise DiagnosticError("official OpenArm URDF changed")
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
    bound_source_files: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    asset_manifest = ASSET_RELEASE / "asset_manifest.json"
    try:
        diagnostic_runner_sha256 = str(
            bound_source_files["diagnostic_runner"]["sha256"]
        )
    except (KeyError, TypeError) as exc:
        raise DiagnosticError("diagnostic runner source binding is unavailable") from exc
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
        "asset_manifest_sha256": _sha256_file(asset_manifest),
        "official_openarm_urdf_path": str(urdf_path.relative_to(WORKSPACE_ROOT)),
        "official_openarm_urdf_sha256": _sha256_file(urdf_path),
        "scene_contract_sha256": EXPECTED_SCENE_CONTRACT_SHA256,
        "motion_plan_sha256": motion_plan_sha256,
        "oracle_2d_stream_sha256": oracle_stream_sha256,
        "raw_rendered_input_stream_sha256": raw_rendered_input_stream_sha256,
        "rendered_evidence_stream_sha256": evidence_stream_sha256,
        "bound_source_files": {
            name: dict(value) for name, value in bound_source_files.items()
        },
        "diagnostic_script_sha256": diagnostic_runner_sha256,
        "ffmpeg_path": FFMPEG,
        "ffmpeg_sha256": _sha256_file(Path(FFMPEG)),
        "ffmpeg_version": _tool_version(FFMPEG),
        "ffprobe_path": FFPROBE,
        "ffprobe_sha256": _sha256_file(Path(FFPROBE)),
        "ffprobe_version": _tool_version(FFPROBE),
    }


def _readme(metrics: Mapping[str, object]) -> str:
    blocked = ", ".join(metrics["nominal_home_exact_mask_blocked_points_first_frame"]) or "none"
    support = int(metrics["nominal_home_candidate_depth_association_point_count_first_frame"])
    first_lost = metrics["candidate_first_lost_age_from_last_measurement_s"]
    first_lost_text = "none" if first_lost is None else f"{float(first_lost):.2f}"
    def p95_text(key: str) -> str:
        value = metrics[key]
        return "none" if value is None else f"{float(value):.2f}"

    def percentage_text(key: str) -> str:
        value = metrics[key]
        return "none" if value is None else f"{100.0 * float(value):.1f}%"

    recommendation = str(metrics["recommendation_supported"]).lower()
    return f"""# Offline robot-mask diagnostic (not a release)

## Negative outcome

Outcome: **{metrics['outcome_class']}**; `recommendation_supported={recommendation}`. The mask-aware tracker reset at frame {metrics['candidate_tracker_reset_frame_index']} and did not recover (`reinitialized={metrics['candidate_tracker_reinitialized_frame_index']}`, `final_initialized={str(metrics['candidate_final_initialized']).lower()}`). This is retained as negative staging evidence, not a recommended method result.

This is one representative official-mesh **R1 / wrist 0** diagnostic. It uses the locked arm64 Mac runtime, official OpenArm geometry, and the official NVIDIA human in its authored `default_equals_rest` rest pose. The converted human surface has `isaac_surface_equivalence=false`. This run is MuJoCo kinematic simulation, not Isaac; it does not run real v9, commissioned root geometry, hardware, or a production call path.

The two trackers consume the same 143 rendered RGB-D frames and the same deterministic noisy oracle-2D landmarks. The baseline sees the rendered robot but receives no mask. The candidate receives the exact simulator robot instance mask through `process(robot_mask=...)`. Anchor 3-D truth is used only for post-hoc error scoring and the overlay; truth depth is never supplied to either tracker. The bounded display wrapper separately uses the simulator patient transform as declared below.

The exact simulator mask is an idealized method input. This run does not execute or commission a production mask provider, does not inject mask false negatives, and does not validate a stable-wrong-surface defense. A missed real robot surface can still create a confident wrong depth association; this diagnostic cannot rule that out.

Nominal HOME exact-mask depth support is **{support}/10** on the first frame; blocked points: **{blocked}**. This limitation is retained, not hidden. The result is only an R1 representative diagnostic, not evidence for all ten points.

The mask-aware display policy accepts current measurements, otherwise uses the last measured point in patient coordinates for at most 0.40 s and a 10 mm uncertainty bound. Raw tracker coast is never relabeled as a display measurement. After either bound is exceeded, the display is `LOST`.

Important: this offline diagnostic gets the patient transform from simulator ground truth. Therefore the patient-relative coast is only a method-sensitivity upper bound, not a signal currently available to production. A real deployment must obtain patient pose from the tracker or an external patient-pose estimator and validate its latency and hand-eye transform before this method can be considered.

Observed summary: baseline initialized={str(metrics['baseline_initialized']).lower()}, candidate initialized at HOME={str(metrics['candidate_initialized']).lower()}, target masked frames={metrics['target_robot_mask_frame_count']}. In the {metrics['result_window_frame_count']}-frame result window, raw mask-tracker unmeasured frames={metrics['raw_mask_unmeasured_frame_count']}; tracker-still-initialized unmeasured frames={metrics['raw_mask_unmeasured_while_tracker_initialized_frame_count']}. Exact target-mask/dropout overlap is {metrics['target_mask_raw_dropout_overlap_frame_count']} frames, including {metrics['target_mask_raw_dropout_overlap_while_initialized_frame_count']} while initialized. Bounded display coast frames={metrics['display_bounded_coast_frame_count']}; display LOST frames={metrics['display_lost_frame_count']}; pre-initialization LOST frames={metrics['preinitialization_display_lost_frame_count']} are separate and excluded. First display LOST occurs {first_lost_text} s after the last measured R1.

## Metrics with explicit denominators

Raw tracker outputs use the same {metrics['result_window_frame_count']}-frame denominator and the same `TRACKING|COASTING` validity policy: no-mask coverage={percentage_text('raw_no_mask_valid_coverage')}, conditional p95={p95_text('raw_no_mask_error_p95_mm')} mm; exact-mask coverage={percentage_text('raw_mask_valid_coverage')}, conditional p95={p95_text('raw_mask_error_p95_mm')} mm.

On only the common-valid paired frames (coverage={percentage_text('raw_common_valid_coverage')}), conditional p95 is no-mask={p95_text('raw_common_no_mask_error_p95_mm')} mm and exact-mask={p95_text('raw_common_mask_error_p95_mm')} mm. The separate simulator-GT patient-relative display policy has coverage={percentage_text('display_valid_coverage')} and conditional p95={p95_text('display_error_p95_mm')} mm.

All error summaries are conditional on their explicitly reported coverage. They must not be interpreted as an improvement, recommendation, or production estimate. Post-hoc false-valid rates above 5/10 mm are raw no-mask={percentage_text('raw_no_mask_false_valid_gt_5mm_rate')}/{percentage_text('raw_no_mask_false_valid_gt_10mm_rate')}, raw exact-mask={percentage_text('raw_mask_false_valid_gt_5mm_rate')}/{percentage_text('raw_mask_false_valid_gt_10mm_rate')}, and display policy={percentage_text('display_false_valid_gt_5mm_rate')}/{percentage_text('display_false_valid_gt_10mm_rate')}.

Permanent video banners: `{BANNER}`, `{METHOD_BANNER}`, `{HUMAN_BANNER}`, and `{OUTCOME_BANNER}`.

`completion=false`; `production_activation=false`. `STAGING_WARNING` is deliberately retained. There is no completion marker and no `current` pointer. Linbox/Isaac/hardware commissioning remains pending.
"""


def _simulate() -> tuple[
    list[dict[str, object]],
    list[np.ndarray],
    int,
    dict[str, object],
    dict[str, object],
]:
    bound_source_start = _bound_source_manifest()
    runtime_raw, conversion_raw, _manifest, urdf_path = _read_official_inputs()
    urdf_raw = urdf_path.read_bytes()
    anchors = load_official_anchor_contract(conversion_raw)
    plans = plan_official_ik_matrix(urdf_raw, anchors)
    motion_plan_sha256 = revalidate_official_motion_plans(urdf_raw, anchors, plans)
    r1_plan = next(plan for plan in plans if plan.target_name == TARGET_NAME)
    if r1_plan.status != "COMPLETE" or len(r1_plan.branches) != 2:
        raise DiagnosticError(f"official R1 motion is unavailable: {r1_plan.failure_reason}")
    branch = r1_plan.branches[WRIST_INDEX]
    timeline = build_timeline(branch.trajectory_joint_positions)
    if len(timeline) != 143:
        raise DiagnosticError("diagnostic timeline count changed")

    intrinsics = Intrinsics(
        fx=float(INTRINSICS["fx"]),
        fy=float(INTRINSICS["fy"]),
        cx=float(INTRINSICS["cx"]),
        cy=float(INTRINSICS["cy"]),
        width=WIDTH,
        height=HEIGHT,
        depth_scale=1000.0,
    )
    detector = _OracleDetector()
    config = ModelConfig(
        init_frames=INIT_FRAMES,
        surface_snap=False,
        robot_mask_dilation_px=0,
    )
    baseline_tracker = ModelBasedTracker(detector, intrinsics, config)
    candidate_tracker = ModelBasedTracker(detector, intrinsics, config)
    bounded = PatientRelativeBoundedCoast()
    target_index = POINT_NAMES.index(TARGET_NAME)
    rows: list[dict[str, object]] = []
    video_frames: list[np.ndarray] = []
    oracle_stream = hashlib.sha256()
    raw_rendered_input_stream = hashlib.sha256()
    evidence_stream = hashlib.sha256()

    with OfflineMeshScene.from_verified_conversion(
        CONVERSION_RELEASE, camera_pose_name=CAMERA_POSE_NAME
    ) as scene:
        for timeline_row in timeline:
            world_from_patient = np.eye(4, dtype=np.float64)
            world_from_patient[0, 3] = timeline_row.patient_translation_x_m
            scene.set_frame(
                timeline_row.left_joint_positions_rad,
                world_from_patient,
            )
            projection = scene.project_current_anchors()
            products = scene.render()
            oracle_pixels = noisy_oracle_pixels(
                projection.projected_pixels, timeline_row.frame_index
            )
            detector.set_pixels(oracle_pixels)
            oracle_stream.update(np.ascontiguousarray(oracle_pixels, dtype="<f8").tobytes())
            blocked = _association_blocked_points(oracle_pixels, products.robot_mask)
            baseline_output = baseline_tracker.process(
                products.rgb,
                products.depth_m,
                t=timeline_row.timestamp_s,
                depth_scale=1000.0,
            )
            candidate_output = candidate_tracker.process(
                products.rgb,
                products.depth_m,
                t=timeline_row.timestamp_s,
                depth_scale=1000.0,
                robot_mask=products.robot_mask,
            )
            baseline_point = baseline_output.points[target_index]
            candidate_point = candidate_output.points[target_index]
            baseline_xyz = _xyz_or_none(baseline_point, baseline_output.initialized)
            candidate_xyz = _xyz_or_none(candidate_point, candidate_output.initialized)
            candidate_sample = TrackerTargetSample(
                measured=bool(
                    candidate_output.initialized
                    and candidate_point.measured
                    and candidate_xyz is not None
                ),
                xyz_camera_m=(
                    candidate_xyz
                    if candidate_xyz is not None
                    else np.full(3, np.nan, dtype=np.float64)
                ),
                position_std_m=float(candidate_point.position_std),
            )
            bounded_output = bounded.step(
                timestamp_s=timeline_row.timestamp_s,
                sample=candidate_sample,
                world_from_patient=world_from_patient,
                world_from_cv_camera=projection.world_from_cv_camera,
            )
            truth_xyz = np.asarray(
                projection.anchor_camera_position_m[target_index], dtype=np.float64
            )
            rendered_input_sha = _frame_digest(
                joints=timeline_row.left_joint_positions_rad,
                world_from_patient=world_from_patient,
                oracle_pixels=oracle_pixels,
                rgb=products.rgb,
                depth_m=products.depth_m,
                robot_mask=products.robot_mask,
            )
            raw_rendered_input_stream.update(bytes.fromhex(rendered_input_sha))
            row: dict[str, object] = {
                "frame_index": timeline_row.frame_index,
                "timestamp_s": timeline_row.timestamp_s,
                "phase": timeline_row.phase,
                "trajectory_index": timeline_row.trajectory_index,
                "patient_translation_x_m": timeline_row.patient_translation_x_m,
                "robot_pixel_count": int(np.count_nonzero(products.robot_mask)),
                "target_robot_mask_5x5": TARGET_NAME in blocked,
                "masked_association_point_count": len(blocked),
                "masked_association_points": ";".join(blocked),
                "baseline_initialized": bool(baseline_output.initialized),
                "candidate_initialized": bool(candidate_output.initialized),
                "baseline_measured": bool(
                    baseline_output.initialized and baseline_point.measured
                ),
                "candidate_raw_measured": bool(
                    candidate_output.initialized and candidate_point.measured
                ),
                "baseline_status": baseline_point.status,
                "candidate_raw_status": candidate_point.status,
                "candidate_status": bounded_output.status,
                "candidate_reason": bounded_output.reason,
                "candidate_coast_age_s": bounded_output.age_s,
                "candidate_bound_m": bounded_output.bound_m,
                "oracle_u_px": float(oracle_pixels[target_index, 0]),
                "oracle_v_px": float(oracle_pixels[target_index, 1]),
                "gt_x_camera_m": float(truth_xyz[0]),
                "gt_y_camera_m": float(truth_xyz[1]),
                "gt_z_camera_m": float(truth_xyz[2]),
                "baseline_x_camera_m": None if baseline_xyz is None else float(baseline_xyz[0]),
                "baseline_y_camera_m": None if baseline_xyz is None else float(baseline_xyz[1]),
                "baseline_z_camera_m": None if baseline_xyz is None else float(baseline_xyz[2]),
                "baseline_error_mm": _error_mm(baseline_xyz, truth_xyz),
                "candidate_raw_x_camera_m": (
                    None if candidate_xyz is None else float(candidate_xyz[0])
                ),
                "candidate_raw_y_camera_m": (
                    None if candidate_xyz is None else float(candidate_xyz[1])
                ),
                "candidate_raw_z_camera_m": (
                    None if candidate_xyz is None else float(candidate_xyz[2])
                ),
                "candidate_raw_error_mm": _error_mm(candidate_xyz, truth_xyz),
                "candidate_x_camera_m": (
                    None if bounded_output.xyz_camera_m is None else float(bounded_output.xyz_camera_m[0])
                ),
                "candidate_y_camera_m": (
                    None if bounded_output.xyz_camera_m is None else float(bounded_output.xyz_camera_m[1])
                ),
                "candidate_z_camera_m": (
                    None if bounded_output.xyz_camera_m is None else float(bounded_output.xyz_camera_m[2])
                ),
                "candidate_error_mm": _error_mm(bounded_output.xyz_camera_m, truth_xyz),
                "rendered_input_sha256": rendered_input_sha,
            }
            frame_sha = semantic_frame_sha256(row)
            row["frame_evidence_sha256"] = frame_sha
            evidence_stream.update(bytes.fromhex(frame_sha))
            rows.append(row)
            video_frames.append(
                _overlay_frame(
                    rgb=products.rgb,
                    robot_mask=products.robot_mask,
                    oracle_pixels=oracle_pixels,
                    truth_pixel=projection.projected_pixels[target_index],
                    baseline_xyz=baseline_xyz,
                    candidate=bounded_output,
                    intrinsics=intrinsics,
                    phase=timeline_row.phase,
                    timestamp_s=timeline_row.timestamp_s,
                )
            )

    row_metrics = derive_row_metrics(rows)
    meaningful = negative_reset_no_recovery_is_meaningful(row_metrics)
    metrics = base_metrics()
    metrics.update(
        {
            "official_trajectory_frame_count": 121,
            "patient_translation_during_hold_m": (
                PATIENT_TRANSLATION_SPEED_M_S * CONTACT_HOLD_FRAMES * FRAME_PERIOD_S
            ),
            "patient_relative_coast_uses_raw_tracker_coast": False,
            "result_window_denominator": "frames_at_or_after_first_both_initialized",
            "meaningful_diagnostic": meaningful,
            **row_metrics,
        }
    )
    bound_source_end = _bound_source_manifest()
    if bound_source_end != bound_source_start:
        raise DiagnosticError("execution-critical source changed during the run")
    input_hashes = _input_hashes(
        runtime_raw=runtime_raw,
        conversion_raw=conversion_raw,
        urdf_path=urdf_path,
        motion_plan_sha256=motion_plan_sha256,
        oracle_stream_sha256=oracle_stream.hexdigest(),
        raw_rendered_input_stream_sha256=raw_rendered_input_stream.hexdigest(),
        evidence_stream_sha256=evidence_stream.hexdigest(),
        bound_source_files=bound_source_end,
    )
    report_candidates = [
        int(row["frame_index"])
        for row in rows
        if row["phase"] == "contact_hold" and row["candidate_status"] == "LOST"
    ]
    if not report_candidates:
        report_candidates = [
            int(row["frame_index"])
            for row in rows
            if row["target_robot_mask_5x5"]
        ]
    report_index = report_candidates[0] if report_candidates else 0
    return rows, video_frames, report_index, metrics, input_hashes


def run(destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    existing = [path.name for path in destination_path.iterdir()]
    if any(name != "STAGING_WARNING" for name in existing):
        raise DiagnosticError("destination already contains diagnostic artifacts")
    (destination_path / "STAGING_WARNING").write_text(
        STAGING_WARNING_TEXT, encoding="utf-8"
    )
    scratch_path = Path(
        tempfile.mkdtemp(prefix=".robot-mask-diagnostic-", dir=destination_path.parent)
    )
    try:
        rows, frames, report_index, metrics, input_hashes = _simulate()
        validate_bound_source_inputs(input_hashes)
        validate_metrics_contract(metrics)
        if metrics.get("meaningful_diagnostic") is not True:
            publish_nonrelease_artifacts(
                scratch=scratch_path,
                destination=destination_path,
                meaningful=False,
            )
        video_path = scratch_path / "robot_mask_diagnostic_h264.mp4"
        _encode_video(frames, video_path)
        metrics["video_contract"] = _probe_video(video_path, len(frames))
        metrics["report_frame_index"] = report_index
        report_bgr = cv2.cvtColor(frames[report_index], cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(scratch_path / "report_frame.png"), report_bgr):
            raise DiagnosticError("report frame could not be written")
        csv_raw = per_frame_csv_bytes(rows)
        verify_metrics_against_csv(metrics, csv_raw, input_hashes)
        (scratch_path / "per_frame.csv").write_bytes(csv_raw)
        (scratch_path / "input_hashes.json").write_bytes(_json_bytes(input_hashes))
        (scratch_path / "metrics.json").write_bytes(_json_bytes(metrics))
        (scratch_path / "README.md").write_text(_readme(metrics), encoding="utf-8")
        persisted_metrics = json.loads(
            (scratch_path / "metrics.json").read_text(encoding="utf-8")
        )
        validate_metrics_contract(persisted_metrics)
        persisted_inputs = json.loads(
            (scratch_path / "input_hashes.json").read_text(encoding="utf-8")
        )
        verify_metrics_against_csv(
            persisted_metrics,
            (scratch_path / "per_frame.csv").read_bytes(),
            persisted_inputs,
        )
        validate_bound_source_inputs(persisted_inputs)
        publish_nonrelease_artifacts(
            scratch=scratch_path,
            destination=destination_path,
            meaningful=True,
        )
        return metrics
    except BaseException as exc:
        _clear_except_warning(destination_path)
        (destination_path / "STAGING_WARNING").write_text(
            STAGING_WARNING_TEXT + f"Fail-closed reason: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        raise
    finally:
        shutil.rmtree(scratch_path, ignore_errors=True)


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
                "meaningful_diagnostic": metrics["meaningful_diagnostic"],
                "output": str(args.output),
                "production_activation": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
