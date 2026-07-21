"""Prototype a staged Mac-only official-mesh occlusion evidence release.

This module never publishes or activates a release.  The current 240-event path
is intentionally test-only because the three-event sparse core cannot measure
the 20 Hz time-to-lost boundary.  It still binds the pinned official OpenArm and
NVIDIA-human conversion through ``OfflineMeshScene`` so serialization and
orchestration can be tested while the temporal core v2 is being integrated.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from ausc.perception.tracking.offline_official_mesh_experiment import (
    InMemorySparseExperiment,
    PATIENT_MOTION_BOUND_M_S,
    SparseEventSpec,
    build_in_memory_sparse_experiment,
    evaluate_rendered_geometry_gate,
    load_official_anchor_contract,
    plan_official_ik_matrix,
    rendered_gate_frame_from_evidence,
)
from ausc.perception.tracking.offline_mesh_simulation import (
    DEFAULT_MAX_FREEZE_AGE_S,
    DEFAULT_MAX_FREEZE_ERROR_P95_M,
    OfflineMeshScene,
    POLICY_NAMES,
)
from ausc.perception.tracking.offline_official_mesh_assets import canonical_json_bytes
from ausc.perception.tracking.official_openarm_experiment import POINT_NAMES
from scripts.perception import run_offline_official_mesh_occlusion as release_contract


class OfflineReleaseBuildError(RuntimeError):
    """A staged offline evidence build violated its fail-closed contract."""


_SCENARIO_RE = re.compile(
    r"^(?:R|L)[1-5]__w[01]__occ(?:0100|0250|0500|1000)ms$"
)
_EVENT_NAMES = ("pre", "contact", "post")
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_URDF_PATH = (
    WORKSPACE_ROOT
    / ".wam-build/offline-official-openarm-assets/releases"
    / "official_assets-05e66c94cd8b36ff79be"
    / "openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf"
)
OFFICIAL_URDF_SHA256 = (
    "63caf1116a844c7eca25e91f0acf025c2798cc4795cf727b23433eacde80ca12"
)
FROZEN_RAW_ROOT = (
    WORKSPACE_ROOT / "docs/runs/2026-07-15-offline-official-mesh-occlusion/raw"
)
DEFAULT_CONVERSION_RELEASE = (
    WORKSPACE_ROOT
    / ".wam-build/offline-official-openarm-meshes/releases"
    / "official_meshes-b5e826af2539bceafea1"
)
WIDTH = 640
HEIGHT = 360
FPS = 12
REPORT_FRAME_INDEX = 120
VIDEO_BANNER = "OFFLINE METHOD SIM - NOT ISAAC / NOT PRODUCTION"
_TEST_ONLY_WARNING = (
    "TEST ONLY - NOT A COMPLETE ARTIFACT\n"
    "20 Hz temporal core v2 and the bound evidence-depth contract "
    "are not integrated.\n"
    "This tree must not be published, activated, or used as method evidence.\n"
).encode("utf-8")


@dataclass(frozen=True)
class ReleaseBuildResult:
    destination: Path
    staging_root: Path
    required_strata_count: int
    sparse_event_count: int
    production_activation: bool
    verification_report: Mapping[str, Any]


@dataclass(frozen=True)
class _CapturedEvent:
    event: SparseEventSpec
    depth_relative: str
    segmentation_relative: str
    scratch_relative: str


@dataclass(frozen=True)
class DryRunReport:
    destination: Path
    required_strata_count: int
    official_target_count: int
    official_wrist_branch_count: int
    serialization_prototype_event_count: int
    temporal_core_v2_integrated: bool
    patient_motion_bound_m_s: float
    scenes_compiled: bool
    production_activation: bool


def _official_urdf_bytes() -> bytes:
    try:
        raw = OFFICIAL_URDF_PATH.read_bytes()
    except OSError as exc:
        raise OfflineReleaseBuildError("pinned official OpenArm URDF is unavailable") from exc
    if hashlib.sha256(raw).hexdigest() != OFFICIAL_URDF_SHA256:
        raise OfflineReleaseBuildError("pinned official OpenArm URDF identity changed")
    return raw


def _validate_complete_motion_plans(plans: Sequence[object]) -> None:
    rows = tuple(plans)
    if tuple(getattr(row, "target_name", None) for row in rows) != POINT_NAMES:
        raise OfflineReleaseBuildError("motion planner did not retain the exact ten targets")
    for row in rows:
        branches = tuple(getattr(row, "branches", ()))
        if (
            getattr(row, "status", None) != "COMPLETE"
            or tuple(getattr(branch, "wrist_index", None) for branch in branches) != (0, 1)
        ):
            raise OfflineReleaseBuildError(
                "the 240-event release requires two complete official branches per target"
            )


def _frozen_manifest_bytes() -> dict[str, bytes]:
    expected = {
        "runtime_manifest.json": release_contract.RUNTIME_MANIFEST_SHA256,
        "asset_manifest.json": release_contract.ASSET_MANIFEST_SHA256,
        "conversion_manifest.json": release_contract.CONVERSION_MANIFEST_SHA256,
    }
    result: dict[str, bytes] = {}
    for name, digest in expected.items():
        source = FROZEN_RAW_ROOT / name
        try:
            raw = source.read_bytes()
        except OSError as exc:
            raise OfflineReleaseBuildError(f"frozen {name} is unavailable") from exc
        if hashlib.sha256(raw).hexdigest() != digest:
            raise OfflineReleaseBuildError(f"frozen {name} identity changed")
        result[name] = raw
    return result


def dry_run_release(
    destination: str | Path,
    *,
    planner: Callable[[bytes, Sequence[object]], Sequence[object]] = plan_official_ik_matrix,
) -> DryRunReport:
    """Validate identities and all 20 IK branches without compiling a render scene."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise OfflineReleaseBuildError("destination already exists; refusing overwrite")
    _frozen_manifest_bytes()
    anchors = load_official_anchor_contract()
    plans = tuple(planner(_official_urdf_bytes(), anchors))
    _validate_complete_motion_plans(plans)
    return DryRunReport(
        destination=target,
        required_strata_count=80,
        official_target_count=10,
        official_wrist_branch_count=20,
        serialization_prototype_event_count=240,
        temporal_core_v2_integrated=False,
        patient_motion_bound_m_s=PATIENT_MOTION_BOUND_M_S,
        scenes_compiled=False,
        production_activation=False,
    )


def _jsonable(value: object) -> object:
    """Round-trip a contract through canonical JSON to remove proxy/tuple wrappers."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise OfflineReleaseBuildError("contract mapping keys must be strings")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise OfflineReleaseBuildError(
        f"contract contains a non-JSON value: {type(value).__name__}"
    )


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError as exc:
        raise OfflineReleaseBuildError(f"refusing to overwrite staged artifact: {path}") from exc


def _write_json(path: Path, value: object) -> None:
    _write_exclusive(path, _release_json_bytes(value))


def _release_json_bytes(value: object) -> bytes:
    """Canonical release JSON uses one final newline; core hashes do not."""

    return canonical_json_bytes(value) + b"\n"


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
    except FileExistsError as exc:
        raise OfflineReleaseBuildError(f"refusing to overwrite raw frame: {path}") from exc


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=release_contract.CSV_HEADER, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode("utf-8")
    _write_exclusive(path, raw)
    return raw


def _copy_frozen_manifests(staging: Path) -> None:
    for name, raw in _frozen_manifest_bytes().items():
        _write_exclusive(staging / "raw" / name, raw)


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _float(value: float) -> str:
    return repr(float(value))


def _optional_float(value: float | None) -> str:
    return "" if value is None else _float(value)


def _validate_rgb(value: object, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (HEIGHT, WIDTH, 3) or array.dtype != np.uint8:
        raise OfflineReleaseBuildError(f"{label} RGB contract changed")
    return np.ascontiguousarray(array)


def _validate_raw_products(products: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = _validate_rgb(getattr(products, "rgb", None), label="primary")
    depth = np.asarray(getattr(products, "depth_m", None))
    segmentation = np.asarray(getattr(products, "segmentation", None))
    if (
        depth.shape != (HEIGHT, WIDTH)
        or not np.issubdtype(depth.dtype, np.floating)
        or not np.isfinite(depth).all()
        or np.any(depth <= 0.0)
    ):
        raise OfflineReleaseBuildError("primary metric-depth contract changed")
    if segmentation.shape != (HEIGHT, WIDTH, 2) or segmentation.dtype != np.int32:
        raise OfflineReleaseBuildError("primary two-channel segmentation contract changed")
    return (
        rgb,
        np.ascontiguousarray(depth, dtype=np.float32),
        np.ascontiguousarray(segmentation, dtype=np.int32),
    )


def _put_text(
    image: np.ndarray,
    text_value: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.40,
    colour: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(
        image,
        text_value,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        1,
        cv2.LINE_AA,
    )


def _compose_base_frame(
    primary_rgb: np.ndarray,
    oblique_rgb: np.ndarray,
    event: SparseEventSpec,
) -> np.ndarray:
    primary = _validate_rgb(primary_rgb, label="primary")
    oblique = _validate_rgb(oblique_rgb, label="oblique")
    result = primary.copy()
    inset = cv2.resize(oblique, (240, 135), interpolation=cv2.INTER_AREA)
    result[HEIGHT - 135 : HEIGHT, WIDTH - 240 : WIDTH] = inset
    result[:72] = np.asarray([14, 20, 32], dtype=np.uint8)
    _put_text(result, VIDEO_BANNER, (8, 17), scale=0.46, colour=(255, 220, 80))
    _put_text(
        result,
        f"OpenArm {release_contract.OPENARM_COMMIT[:12]}  {event.scenario_id}",
        (8, 36),
    )
    _put_text(
        result,
        f"{event.event_name}  t={event.timestamp_s:.2f}s  "
        f"target={event.target_name}  wrist={event.wrist_index}",
        (8, 55),
    )
    _put_text(result, "OBLIQUE", (WIDTH - 232, HEIGHT - 116), scale=0.38)
    return result


def _policy_lookup(
    result: InMemorySparseExperiment,
) -> dict[tuple[str, int, str], object]:
    lookup: dict[tuple[str, int, str], object] = {}
    for policy_name in POLICY_NAMES:
        for row in result.policy_outputs[policy_name]:
            key = (row.scenario_id, row.frame_index, policy_name)
            if key in lookup:
                raise OfflineReleaseBuildError("duplicate policy output key")
            lookup[key] = row
    return lookup


def _overlay_policy_results(
    base_rgb: np.ndarray,
    event: SparseEventSpec,
    lookup: Mapping[tuple[str, int, str], object],
    truth_world_m: np.ndarray,
) -> np.ndarray:
    result = np.ascontiguousarray(base_rgb).copy()
    y = 91
    for policy_name in POLICY_NAMES:
        output = lookup[(event.scenario_id, event.frame_index, policy_name)]
        estimate = output.estimate_world_m
        error = (
            None
            if estimate is None
            else float(np.linalg.norm(np.asarray(estimate) - truth_world_m))
        )
        error_text = "n/a" if error is None else f"{1000.0 * error:.1f}mm"
        _put_text(
            result,
            f"{policy_name}: {output.state}  err={error_text}",
            (8, y),
            scale=0.34,
            colour=(245, 245, 245),
        )
        y += 17
    return result


class _FFmpegVideoSink:
    """Stream fixed-size RGB frames directly to locked H.264/yuv420p output."""

    def __init__(self, path: Path, *, fps: int, width: int, height: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise OfflineReleaseBuildError("ffmpeg is required for the evidence video")
        self.path = path
        self.width = width
        self.height = height
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def __enter__(self) -> "_FFmpegVideoSink":
        return self

    def write(self, frame: np.ndarray) -> None:
        rgb = _validate_rgb(frame, label="video")
        if self._process.stdin is None:
            raise OfflineReleaseBuildError("ffmpeg input pipe is unavailable")
        try:
            self._process.stdin.write(rgb.tobytes(order="C"))
        except (BrokenPipeError, OSError) as exc:
            raise OfflineReleaseBuildError("ffmpeg rejected an evidence frame") from exc

    def __exit__(self, exc_type: object, _exc: object, _traceback: object) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr = b""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read()
        return_code = self._process.wait(timeout=60)
        if exc_type is None and return_code != 0:
            raise OfflineReleaseBuildError(
                "ffmpeg failed to encode H.264 evidence: "
                + stderr.decode("utf-8", errors="replace")[-1000:]
            )


def _matrix_records(result: InMemorySparseExperiment) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for stratum in result.strata:
        required = stratum.required_stratum
        records.append(
            {
                "scenario_id": stratum.scenario_id,
                "target_name": stratum.target_name,
                "wrist_index": stratum.wrist_index,
                "occlusion_duration_s": stratum.occlusion_duration_s,
                "status": stratum.status,
                "frame_count": required.declared_frame_count,
                "observation_seed": required.observation_seed,
                "patient_motion_seed": required.patient_motion_seed,
                "observation_stream_sha256": required.observation_stream_sha256,
            }
        )
    return records


def _input_contract(
    result: InMemorySparseExperiment, records: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    strata: list[dict[str, object]] = []
    for stratum, record in zip(result.strata, records, strict=True):
        if len(stratum.events) != 3 or len(stratum.gates) != 3:
            raise OfflineReleaseBuildError("every release stratum must contain three events")
        direction = stratum.events[0].patient_motion_direction_world.tolist()
        strata.append(
            {
                **{
                    key: record[key]
                    for key in (
                        "scenario_id",
                        "target_name",
                        "wrist_index",
                        "occlusion_duration_s",
                        "status",
                        "frame_count",
                        "observation_seed",
                        "patient_motion_seed",
                    )
                },
                "patient_motion_direction_world": direction,
                "events": [
                    {
                        "frame_index": event.frame_index,
                        "event_name": event.event_name,
                        "timestamp_s": event.timestamp_s,
                        "observation_seed": event.observation_seed,
                        "noise_contract_sha256": event.noise.contract_sha256,
                    }
                    for event in stratum.events
                ],
            }
        )
    return {
        "schema": "wam_offline_official_mesh_occlusion_input_contract_v2",
        "status": release_contract.STATUS,
        "scope": release_contract.SCOPE,
        "production_activation": False,
        "identities": {
            "runtime_manifest_sha256": release_contract.RUNTIME_MANIFEST_SHA256,
            "asset_manifest_sha256": release_contract.ASSET_MANIFEST_SHA256,
            "conversion_manifest_sha256": release_contract.CONVERSION_MANIFEST_SHA256,
        },
        "core_contract": _jsonable(result.input_contract),
        "core_contract_sha256": result.input_contract_sha256,
        "policy_contract": {
            "policy_names": list(POLICY_NAMES),
            "max_freeze_age_s": DEFAULT_MAX_FREEZE_AGE_S,
            "max_freeze_error_p95_m": DEFAULT_MAX_FREEZE_ERROR_P95_M,
            "patient_motion_bound_m_s": PATIENT_MOTION_BOUND_M_S,
        },
        "matrix": {
            "point_names": list(POINT_NAMES),
            "wrist_indices": [0, 1],
            "occlusion_durations_s": [0.10, 0.25, 0.50, 1.00],
            "required_strata_count": 80,
        },
        "render_provenance": {
            "width": WIDTH,
            "height": HEIGHT,
            "raw_depth_dtype": "float32",
            "raw_segmentation_dtype": "int32",
            "raw_segmentation_channels": 2,
            "segmentation_object_type": release_contract.OFFICIAL_GEOM_OBJECT_TYPE,
            "robot_geom_ids": list(release_contract.OFFICIAL_ROBOT_GEOM_IDS),
            "human_geom_ids": list(release_contract.OFFICIAL_HUMAN_GEOM_IDS),
        },
        "strata": strata,
    }


def _csv_rows(
    result: InMemorySparseExperiment,
    captures: Mapping[tuple[str, int], _CapturedEvent],
) -> list[dict[str, str]]:
    policy = _policy_lookup(result)
    rows: list[dict[str, str]] = []
    for stratum in result.strata:
        for event, gate, truth in zip(
            stratum.events,
            stratum.gates,
            stratum.ground_truth_frames,
            strict=True,
        ):
            capture = captures[(stratum.scenario_id, event.frame_index)]
            for policy_name in POLICY_NAMES:
                output = policy[(stratum.scenario_id, event.frame_index, policy_name)]
                estimate = output.estimate_world_m
                error = (
                    None
                    if estimate is None
                    else float(
                        np.linalg.norm(
                            np.asarray(estimate, dtype=np.float64) - truth.target_world_m
                        )
                    )
                )
                accepted = gate.accepted_estimate_world_m
                row = {
                    "scenario_id": stratum.scenario_id,
                    "target_name": stratum.target_name,
                    "wrist_index": str(stratum.wrist_index),
                    "occlusion_duration_s": _float(stratum.occlusion_duration_s),
                    "frame_index": str(event.frame_index),
                    "timestamp_s": _float(event.timestamp_s),
                    "event_name": event.event_name,
                    "observation_seed": str(event.observation_seed),
                    "noise_contract_sha256": event.noise.contract_sha256,
                    "observation_stream_sha256": truth.observation_stream_sha256,
                    "geometry_accepted": _bool(gate.accepted),
                    "visible_point_count_report_only": str(
                        gate.visible_point_count_report_only
                    ),
                    "accepted_estimate_x_m": (
                        "" if accepted is None else _float(accepted[0])
                    ),
                    "accepted_estimate_y_m": (
                        "" if accepted is None else _float(accepted[1])
                    ),
                    "accepted_estimate_z_m": (
                        "" if accepted is None else _float(accepted[2])
                    ),
                    "accepted_error_p95_m": _optional_float(
                        gate.accepted_error_p95_m
                    ),
                    "patient_motion_bound_m_s": _float(PATIENT_MOTION_BOUND_M_S),
                    "policy_name": policy_name,
                    "policy_state": output.state,
                    "episode_id": output.episode_id or "",
                    "last_valid_age_s": _optional_float(output.last_valid_age_s),
                    "predicted_error_p95_m": _optional_float(
                        output.predicted_error_p95_m
                    ),
                    "reacquired": _bool(output.reacquired),
                    "deliberately_unsafe": _bool(output.deliberately_unsafe),
                    "estimate_x_m": "" if estimate is None else _float(estimate[0]),
                    "estimate_y_m": "" if estimate is None else _float(estimate[1]),
                    "estimate_z_m": "" if estimate is None else _float(estimate[2]),
                    "valid": _bool(output.valid),
                    "ground_truth_x_m": _float(truth.target_world_m[0]),
                    "ground_truth_y_m": _float(truth.target_world_m[1]),
                    "ground_truth_z_m": _float(truth.target_world_m[2]),
                    "error_m": _optional_float(error),
                    "raw_depth_path": capture.depth_relative,
                    "raw_segmentation_path": capture.segmentation_relative,
                }
                if tuple(row) != release_contract.CSV_HEADER:
                    raise OfflineReleaseBuildError("builder/verifier CSV schema drifted")
                rows.append(row)
    return rows


def _metric_payload(result: InMemorySparseExperiment) -> dict[str, object]:
    denominators = result.metrics[POLICY_NAMES[0]]["denominators"]
    if any(result.metrics[name]["denominators"] != denominators for name in POLICY_NAMES):
        raise OfflineReleaseBuildError("policy metric denominators disagree")
    return {
        "schema": "wam_offline_official_mesh_occlusion_metrics_v2",
        "status": release_contract.STATUS,
        "scope": release_contract.SCOPE,
        "production_activation": False,
        "isaac_equivalence": False,
        "identities": {
            "runtime_manifest_sha256": release_contract.RUNTIME_MANIFEST_SHA256,
            "asset_manifest_sha256": release_contract.ASSET_MANIFEST_SHA256,
            "conversion_manifest_sha256": release_contract.CONVERSION_MANIFEST_SHA256,
        },
        "denominators": _jsonable(denominators),
        "policies": {
            name: {
                key: _jsonable(result.metrics[name][key])
                for key in ("pooled_frame", "equal_weight_per_stratum", "per_stratum")
            }
            for name in POLICY_NAMES
        },
        "failed_strata": [
            row.scenario_id for row in result.strata if row.status != "COMPLETE"
        ],
        "completion_gates": dict(release_contract.FALSE_COMPLETION_GATES),
    }


def _video_timeline(
    result: InMemorySparseExperiment,
    captured: Sequence[_CapturedEvent],
) -> dict[str, object]:
    required = {
        row.scenario_id: row.observation_stream_sha256 for row in result.required_strata
    }
    return {
        "schema": "wam_offline_official_mesh_occlusion_video_timeline_v1",
        "status": release_contract.STATUS,
        "video_path": "offline_official_mesh_occlusion_h264.mp4",
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "frame_count": len(captured),
        "report_frame_video_frame_index": REPORT_FRAME_INDEX,
        "overlay_contract": {
            "region_xywh": [0, 0, WIDTH, 72],
            "banner_text": "OFFLINE METHOD SIM — NOT ISAAC / NOT PRODUCTION",
            "robot_commit": release_contract.OPENARM_COMMIT,
            "policy_names": list(POLICY_NAMES),
        },
        "frames": [
            {
                "video_frame_index": index,
                "scenario_id": row.event.scenario_id,
                "target_name": row.event.target_name,
                "wrist_index": row.event.wrist_index,
                "frame_index": row.event.frame_index,
                "timestamp_s": row.event.timestamp_s,
                "event_name": row.event.event_name,
                "observation_stream_sha256": required[row.event.scenario_id],
                "raw_depth_path": row.depth_relative,
                "raw_segmentation_path": row.segmentation_relative,
            }
            for index, row in enumerate(captured)
        ],
    }


def _experiment_manifest(
    result: InMemorySparseExperiment,
    records: Sequence[Mapping[str, object]],
    *,
    input_contract_sha256: str,
    first_csv_sha256: str,
    rerun_csv_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "wam_offline_official_mesh_occlusion_experiment_v2",
        "status": release_contract.STATUS,
        "scope": release_contract.SCOPE,
        "production_activation": False,
        "isaac_equivalence": False,
        "identities": {
            "runtime_manifest_sha256": release_contract.RUNTIME_MANIFEST_SHA256,
            "asset_manifest_sha256": release_contract.ASSET_MANIFEST_SHA256,
            "conversion_manifest_sha256": release_contract.CONVERSION_MANIFEST_SHA256,
            "asset_release_id": release_contract.ASSET_RELEASE_ID,
            "conversion_release_id": release_contract.CONVERSION_RELEASE_ID,
            "robot_repository_url": release_contract.ROBOT_REPOSITORY_URL,
            "robot_commit": release_contract.OPENARM_COMMIT,
            "robot_urdf_sha256": release_contract.OPENARM_URDF_SHA256,
            "human_asset_url": release_contract.HUMAN_ASSET_URL,
            "human_usd_sha256": release_contract.NVIDIA_HUMAN_USD_SHA256,
        },
        "render": {
            "codec": "h264",
            "pixel_format": "yuv420p",
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "frame_count": 240,
            "undistorted_pinhole_only": True,
            "raw_depth_dtype": "float32",
            "raw_segmentation_dtype": "int32",
            "raw_segmentation_channels": 2,
            "segmentation_object_type": release_contract.OFFICIAL_GEOM_OBJECT_TYPE,
            "robot_geom_ids": list(release_contract.OFFICIAL_ROBOT_GEOM_IDS),
            "human_geom_ids": list(release_contract.OFFICIAL_HUMAN_GEOM_IDS),
        },
        "matrix": {
            "point_names": list(POINT_NAMES),
            "wrist_indices": [0, 1],
            "occlusion_durations_s": [0.10, 0.25, 0.50, 1.00],
            "required_strata_count": 80,
            "records": list(records),
        },
        "policies": list(POLICY_NAMES),
        "artifacts": {
            "video_path": "offline_official_mesh_occlusion_h264.mp4",
            "report_frame_path": "report_frame.png",
            "metrics_path": "raw/metrics.json",
            "per_frame_csv_path": "raw/per_frame.csv",
            "determinism_rerun_csv_path": "raw/determinism_rerun/per_frame.csv",
            "input_contract_path": "raw/input_contract.json",
            "video_timeline_path": "raw/video_timeline.json",
            "raw_frame_count": 240,
        },
        "determinism": {
            "input_contract_sha256": input_contract_sha256,
            "core_input_contract_sha256": result.input_contract_sha256,
            "first_run_per_frame_sha256": first_csv_sha256,
            "same_input_rerun_per_frame_sha256": rerun_csv_sha256,
            "same_input_match": first_csv_sha256 == rerun_csv_sha256,
        },
        "completion_gates": dict(release_contract.FALSE_COMPLETION_GATES),
    }


def _write_sha256sums(staging: Path) -> None:
    rows: list[str] = []
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        relative = path.relative_to(staging).as_posix()
        if relative == "SHA256SUMS.txt":
            continue
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    _write_exclusive(staging / "SHA256SUMS.txt", "".join(rows).encode("utf-8"))


def _staging_path(destination: Path) -> Path:
    if not destination.name or destination.name in (".", ".."):
        raise OfflineReleaseBuildError("destination name is invalid")
    if os.path.lexists(destination):
        raise OfflineReleaseBuildError("destination already exists; refusing overwrite")
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise OfflineReleaseBuildError("destination parent must be an existing real directory")
    for _attempt in range(32):
        candidate = parent / f".{destination.name}.staging-{secrets.token_hex(8)}"
        try:
            candidate.mkdir(mode=0o700)
            return candidate
        except FileExistsError:
            continue
    raise OfflineReleaseBuildError("could not allocate a unique sibling staging directory")


def build_release_staging(
    destination: str | Path,
    *,
    allow_test_only: bool = False,
    conversion_release: str | Path = DEFAULT_CONVERSION_RELEASE,
    planner: Callable[[bytes, Sequence[object]], Sequence[object]] = plan_official_ik_matrix,
    scene_factory: Callable[..., object] = OfflineMeshScene.from_verified_conversion,
    experiment_builder: Callable[
        ..., InMemorySparseExperiment
    ] = build_in_memory_sparse_experiment,
    gate_frame_factory: Callable[..., object] = rendered_gate_frame_from_evidence,
    gate_evaluator: Callable[[object, object], object] = evaluate_rendered_geometry_gate,
    array_writer: Callable[[Path, np.ndarray], None] = _write_npy,
    video_sink_factory: Callable[..., object] = _FFmpegVideoSink,
    release_verifier: Callable[
        [Path], Mapping[str, Any]
    ] = release_contract.verify_offline_release,
) -> ReleaseBuildResult:
    """Exercise the v2 serializer in tests; real staging remains fail-closed."""

    target = Path(destination)
    if os.path.lexists(target):
        raise OfflineReleaseBuildError("destination already exists; refusing overwrite")
    if type(allow_test_only) is not bool or not allow_test_only:
        raise OfflineReleaseBuildError(
            "real staging is blocked until the 20 Hz temporal core v2 and "
            "evidence-depth contract are integrated"
        )
    if (
        scene_factory == OfflineMeshScene.from_verified_conversion
        or array_writer is _write_npy
        or video_sink_factory is _FFmpegVideoSink
        or release_verifier is release_contract.verify_offline_release
    ):
        raise OfflineReleaseBuildError(
            "test-only serialization requires injected scene, raw-writer, "
            "video-sink, and schema-validator boundaries"
        )
    staging = _staging_path(target)
    _write_exclusive(
        staging / "TEST_ONLY_SERIALIZATION_PROTOTYPE.txt", _TEST_ONLY_WARNING
    )
    scratch = staging / "._scratch"
    scratch.mkdir()
    try:
        _copy_frozen_manifests(staging)
        anchors = load_official_anchor_contract()
        plans = tuple(planner(_official_urdf_bytes(), anchors))
        _validate_complete_motion_plans(plans)
        captures: list[_CapturedEvent] = []
        captures_by_key: dict[tuple[str, int], _CapturedEvent] = {}
        conversion = Path(conversion_release)
        with ExitStack() as stack:
            primary = stack.enter_context(
                scene_factory(conversion, camera_pose_name="primary")
            )
            oblique = stack.enter_context(
                scene_factory(conversion, camera_pose_name="oblique")
            )
            initial_q = plans[0].branches[0].approach_joint_positions_rad
            primary.set_frame(initial_q, np.eye(4))
            reference = np.asarray(
                primary.project_current_anchors().anchor_camera_position_m,
                dtype=np.float64,
            )
            if reference.shape != (10, 3) or not np.isfinite(reference).all():
                raise OfflineReleaseBuildError("primary reference anchor projection changed")

            def gate_provider(event: SparseEventSpec) -> object:
                if not isinstance(event, SparseEventSpec):
                    raise OfflineReleaseBuildError("core provided an invalid sparse event")
                key = (event.scenario_id, event.frame_index)
                if key in captures_by_key:
                    raise OfflineReleaseBuildError("core requested a duplicate rendered event")
                primary.set_frame(event.left_joint_positions_rad, event.world_from_patient)
                oblique.set_frame(event.left_joint_positions_rad, event.world_from_patient)
                evidence = primary.capture_frame_evidence(event.timestamp_s)
                primary_rgb, depth, segmentation = _validate_raw_products(
                    evidence.products
                )
                oblique_rgb = _validate_rgb(oblique.render().rgb, label="oblique")
                depth_relative, segmentation_relative = raw_paths_for_event(
                    scenario_id=event.scenario_id,
                    frame_index=event.frame_index,
                    event_name=event.event_name,
                )
                array_writer(staging / depth_relative, depth)
                array_writer(staging / segmentation_relative, segmentation)
                base = _compose_base_frame(primary_rgb, oblique_rgb, event)
                scratch_relative = f"._scratch/frame_{len(captures):05d}.png"
                if not cv2.imwrite(
                    str(staging / scratch_relative),
                    cv2.cvtColor(base, cv2.COLOR_RGB2BGR),
                ):
                    raise OfflineReleaseBuildError("could not persist a scratch video frame")
                captured = _CapturedEvent(
                    event=event,
                    depth_relative=depth_relative,
                    segmentation_relative=segmentation_relative,
                    scratch_relative=scratch_relative,
                )
                captures.append(captured)
                captures_by_key[key] = captured
                frame = gate_frame_factory(
                    evidence,
                    reference_anchor_camera_m=reference,
                    target_name=event.target_name,
                )
                return gate_evaluator(frame, event.noise)

            result = experiment_builder(plans, anchors, gate_provider=gate_provider)

        if not isinstance(result, InMemorySparseExperiment):
            raise OfflineReleaseBuildError("experiment builder returned the wrong contract")
        if (
            len(result.strata) != 80
            or len(result.observations) != 240
            or len(captures) != 240
            or any(row.status != "COMPLETE" or len(row.events) != 3 for row in result.strata)
        ):
            raise OfflineReleaseBuildError(
                "release requires all 80 complete strata and exactly 240 real rendered events"
            )

        records = _matrix_records(result)
        input_value = _input_contract(result, records)
        input_raw = _release_json_bytes(input_value)
        _write_exclusive(staging / "raw/input_contract.json", input_raw)
        input_sha = hashlib.sha256(input_raw).hexdigest()
        rows_first = _csv_rows(result, captures_by_key)
        rows_rerun = _csv_rows(result, captures_by_key)
        first_raw = _write_csv(staging / "raw/per_frame.csv", rows_first)
        rerun_raw = _write_csv(
            staging / "raw/determinism_rerun/per_frame.csv", rows_rerun
        )
        first_sha = hashlib.sha256(first_raw).hexdigest()
        rerun_sha = hashlib.sha256(rerun_raw).hexdigest()
        if first_raw != rerun_raw:
            raise OfflineReleaseBuildError("same-input method serialization is not deterministic")

        lookup = _policy_lookup(result)
        truth_lookup = {
            (row.scenario_id, row.frame_index): row.target_world_m
            for row in result.ground_truth_frames
        }
        report_frame: np.ndarray | None = None
        video_path = staging / "offline_official_mesh_occlusion_h264.mp4"
        with video_sink_factory(
            video_path, fps=FPS, width=WIDTH, height=HEIGHT
        ) as sink:
            for video_index, captured in enumerate(captures):
                bgr = cv2.imread(str(staging / captured.scratch_relative), cv2.IMREAD_COLOR)
                if bgr is None or bgr.shape != (HEIGHT, WIDTH, 3):
                    raise OfflineReleaseBuildError("scratch video frame is unreadable")
                base = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                key = (captured.event.scenario_id, captured.event.frame_index)
                frame = _overlay_policy_results(
                    base, captured.event, lookup, truth_lookup[key]
                )
                sink.write(frame)
                if video_index == REPORT_FRAME_INDEX:
                    report_frame = frame.copy()
        if report_frame is None:
            raise OfflineReleaseBuildError("report frame index was not generated")
        if not cv2.imwrite(
            str(staging / "report_frame.png"),
            cv2.cvtColor(report_frame, cv2.COLOR_RGB2BGR),
        ):
            raise OfflineReleaseBuildError("could not write the report frame")

        _write_json(staging / "raw/metrics.json", _metric_payload(result))
        _write_json(
            staging / "raw/video_timeline.json", _video_timeline(result, captures)
        )
        _write_json(
            staging / "raw/experiment_manifest.json",
            _experiment_manifest(
                result,
                records,
                input_contract_sha256=input_sha,
                first_csv_sha256=first_sha,
                rerun_csv_sha256=rerun_sha,
            ),
        )
        _write_exclusive(
            staging / "README.md",
            (
                "# Offline official-mesh occlusion method experiment\n\n"
                "OFFLINE METHOD SIM — NOT ISAAC / NOT PRODUCTION.\n\n"
                "Official OpenArm and pinned NVIDIA-human derived meshes only. "
                "This staging tree is not activated or published.\n"
            ).encode("utf-8"),
        )
        shutil.rmtree(scratch)
        try:
            verification = dict(release_verifier(staging))
        except BaseException:
            (staging / "ARTIFACT_COMPLETE.json").unlink(missing_ok=True)
            (staging / "SHA256SUMS.txt").unlink(missing_ok=True)
            raise
        return ReleaseBuildResult(
            destination=target,
            staging_root=staging,
            required_strata_count=80,
            sparse_event_count=240,
            production_activation=False,
            verification_report=verification,
        )
    except BaseException:
        # Preserve expensive raw evidence for diagnosis, but never leave a
        # completion marker on a failed or interrupted build.
        (staging / "ARTIFACT_COMPLETE.json").unlink(missing_ok=True)
        (staging / "SHA256SUMS.txt").unlink(missing_ok=True)
        raise


def raw_paths_for_event(
    *, scenario_id: str, frame_index: int, event_name: str
) -> tuple[str, str]:
    """Return the unique canonical depth/segmentation paths for one event."""

    if not isinstance(scenario_id, str) or _SCENARIO_RE.fullmatch(scenario_id) is None:
        raise OfflineReleaseBuildError("scenario_id is outside the frozen 80-stratum matrix")
    if type(frame_index) is not int or frame_index not in range(3):
        raise OfflineReleaseBuildError("frame_index must be one of the three sparse events")
    if event_name != _EVENT_NAMES[frame_index]:
        raise OfflineReleaseBuildError("event_name does not match the sparse frame index")
    stem = f"raw/frames/{scenario_id}/event_{frame_index}_{event_name}"
    return f"{stem}_depth_m.npy", f"{stem}_segmentation.npy"


def main(
    argv: Sequence[str] | None = None,
    *,
    dry_runner: Callable[..., DryRunReport] = dry_run_release,
    build_runner: Callable[..., ReleaseBuildResult] = build_release_staging,
) -> int:
    """Run only an explicit dry-run or an explicit non-publishing staging build."""

    parser = argparse.ArgumentParser(
        description="Build the Mac-only official-mesh method evidence staging tree."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--build-staging", action="store_true")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--conversion-release", type=Path, default=DEFAULT_CONVERSION_RELEASE
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        report = dry_runner(args.destination)
        payload = {
            "status": "DRY_RUN_IK_ONLY_TEMPORAL_CORE_V2_PENDING",
            "destination": str(report.destination),
            "required_strata_count": report.required_strata_count,
            "official_target_count": report.official_target_count,
            "official_wrist_branch_count": report.official_wrist_branch_count,
            "serialization_prototype_event_count": (
                report.serialization_prototype_event_count
            ),
            "temporal_core_v2_integrated": report.temporal_core_v2_integrated,
            "patient_motion_bound_m_s": report.patient_motion_bound_m_s,
            "scenes_compiled": report.scenes_compiled,
            "production_activation": report.production_activation,
        }
    else:
        result = build_runner(
            args.destination, conversion_release=args.conversion_release
        )
        payload = {
            "status": "STAGING_VERIFIED_NOT_PUBLISHED",
            "destination": str(result.destination),
            "staging_root": str(result.staging_root),
            "required_strata_count": result.required_strata_count,
            "sparse_event_count": result.sparse_event_count,
            "production_activation": result.production_activation,
            "verification_report": _jsonable(result.verification_report),
        }
    print(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


__all__ = (
    "DryRunReport",
    "OfflineReleaseBuildError",
    "POINT_NAMES",
    "ReleaseBuildResult",
    "build_release_staging",
    "dry_run_release",
    "main",
    "raw_paths_for_event",
)


if __name__ == "__main__":
    raise SystemExit(main())
