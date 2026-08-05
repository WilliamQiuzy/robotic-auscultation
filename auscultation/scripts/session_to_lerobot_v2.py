"""Canonical LeRobot dataset from a recorded session — fail-closed VLA/WAM export.

Produces the REAL LeRobot v2 dataset via `LeRobotDataset.create/add_frame/save_episode` (so stats,
tasks, and video encoding are exactly what VLA tooling — ACT / pi0 / OpenVLA-style — expects), rather
than a hand-rolled parquet. What each modality becomes:

  ④ arm     -> observation.state (follower joint observation at exposure nominal)
               + action (one exact, non-interpolated MIT control tick)          [float32]
  ③ video   -> observation.images.<cam>  (encoded mp4, one row per camera frame)  [video]  ← VLA core
  ① force   -> current contact force + validity/age (past-only as-of alignment)     [float32]
  ② audio   -> raw wav stays an immutable session sidecar (LeRobot has no native audio);
               recomputable segment reward metadata lives in segments.jsonl and never enters observations
  language  -> `task` per frame (the auscultation-site instruction) — the "L" in VLA
  WAM/RL    -> next.reward + next.done, with every target interval saved as a separate episode

Dataset fps = CAMERA fps (one row per image, the VLA convention); camera exposure, force, robot,
and target rows are aligned causally. The full 1 kHz exact-control sidecar is copied into a hashed
provenance snapshot and every camera frame records which real tick supplied its action. Current
v2/v3 captures remain readable for commissioning but can never be published as training data.  V3
proves only that every Linux SocketCAN write queued its complete frame; it has no per-motor feedback
liveness evidence and is not a motor receipt or execution acknowledgement. Runs in linbox's pinned
lerobot 0.3.3 env.

    ~/miniconda3/envs/lerobot/bin/python scripts/session_to_lerobot_v2.py <session_dir> --repo-id ausc/demo --root <out>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import wave
from bisect import bisect_right
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

# Shared source of truth for the point-code -> anatomical instruction (the VLA "L"). Kept in
# scripts/ next to this converter and the collection server so BOTH fill the task identically.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auscultation_sites import phrase_site_instruction, site_instruction  # noqa: E402

DEFAULT_TASK = "auscultate the posterior lung field"
CAMERA_TIME_SOURCE_BY_BACKEND = {
    "orbbec": "orbbec_device_and_host_arrival",
    "avfoundation": "opencv_host_arrival",
    "opencv": "opencv_host_arrival",
}
ORBBEC_COLOR_FORMATS = frozenset({"RGB", "BGR", "MJPG", "YUYV", "UYVY", "NV12"})
ORBBEC_ALIGN_MODES = frozenset({"software_depth_to_color", "disabled"})
# Accepted hardware camera-timing methods (see camera_exposure_evidence._ACCEPTED_TIMING_METHODS):
# LED flash, or the camera's host-synced hardware timestamp (device_timer_host_sync).
_CAMERA_TIMING_METHODS = frozenset({"hardware_led_flash", "device_timer_host_sync"})
REQUIRED_RAW_ARTIFACTS = frozenset({
    "audio.wav", "rgb.mp4", "force.csv", "rgb_times.csv", "obs.csv",
    "targets.jsonl", "tracker.jsonl", "depth/depth_times.csv",
    "robot_control_envelopes.jsonl",
    "robot_exact_control.csv", "robot_frame_index.jsonl", "robot_capture_summary.json",
})

#: Immutable capture kinds that a production writer may stamp into the session
#: manifest and that this converter is willing to promote.  A diagnostic capture
#: (``diagnostic_perception``/``diagnostic_perception_preview``) is never a
#: member, so it can never be converted to training data even if a forged
#: manifest also claims ``training_eligible=true``.  This allowlist MUST stay in
#: agreement with ``training_authenticity_finalizer.PRODUCTION_CAPTURE_KINDS``.
PRODUCTION_CAPTURE_KINDS = frozenset({"production_teleop_v4"})
ARTIFACT_HASH_SCHEMA_VERSION = "wam_session_artifact_hashes_v1"
PROMOTION_CONTRACT_VERSION = "wam_promotion_v2"
TRUSTED_FINALIZATION_REGISTRY_ENV = "WAM_TRAINING_FINALIZATION_REGISTRY"
TRUSTED_SNAPSHOT_ROOT_ENV = "WAM_TRAINING_SNAPSHOT_ROOT"
FINALIZATION_SCHEMA_VERSION = "wam_session_finalization_v2"
TRAINING_AUTHENTICITY_NAME = "TRAINING_AUTHENTICITY.json"
TRAINING_AUTHENTICITY_SCHEMA_VERSION = "wam_training_authenticity_v2"
EXACT_ROBOT_SUMMARY_SCHEMA_V2 = "wam_openarm_exact_control_capture_summary_v2"
EXACT_ROBOT_SUMMARY_SCHEMA = "wam_openarm_exact_control_capture_summary_v3"
EXACT_ROBOT_SUMMARY_SCHEMA_V4 = "wam_openarm_exact_control_capture_summary_v4"
EXACT_ROBOT_FRAME_SCHEMA_V2 = "wam_openarm_exact_control_frame_evidence_v2"
EXACT_ROBOT_FRAME_SCHEMA = "wam_openarm_exact_control_frame_evidence_v3"
EXACT_ROBOT_FRAME_SCHEMA_V4 = "wam_openarm_exact_control_frame_evidence_v4"
EXACT_ROBOT_TX_STATUS_REASON = "openarm_can_1_2_2_full_frame_write_results"
EXACT_ROBOT_TX_STATUS_SEMANTICS = (
    "linux_socketcan_full_frame_write_tx_enqueue_not_motor_ack"
)
EXACT_ROBOT_MOTOR_ACK_REASON = (
    "socketcan_tx_enqueue_does_not_acknowledge_motor_receipt_or_execution"
)
EXACT_ACTION_SELECTION_SCHEMA = "wam_lerobot_exact_action_selection_v1"
EXACT_ACTION_SELECTION_RULE = (
    "latest_exact_control_tick_completed_at_or_before_exposure_nominal_ns"
)
EXACT_ROBOT_DATA_FILES = (
    "robot_control_envelopes.jsonl",
    "robot_exact_control.csv",
    "robot_frame_index.jsonl",
)
EXACT_ROBOT_FILES = EXACT_ROBOT_DATA_FILES + ("robot_capture_summary.json",)
CONVERTER_DERIVED_PATHS = frozenset({
    "FINALIZATION.original.json",
    "manifest.original.json",
    "provenance.json",
    "robot_frame_action_selection.jsonl",
    "segments.jsonl",
})


@dataclass(frozen=True)
class _PinnedSidecar:
    descriptor: int
    identity: tuple[int, ...]
    sha256: str
    size_bytes: int


class _PinnedExactRobotSidecars:
    """One no-follow fd bundle shared by replay, parsing, and selection."""

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self._items: dict[str, _PinnedSidecar] = {}

    @staticmethod
    def _identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_mode),
            int(metadata.st_nlink),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )

    @staticmethod
    def _hash_descriptor(descriptor: int) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.pread(descriptor, 1024 * 1024, size)
            if not block:
                break
            digest.update(block)
            size += len(block)
        return digest.hexdigest(), size

    def __enter__(self):
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        seen_inodes = set()
        try:
            for name in EXACT_ROBOT_FILES:
                descriptor = os.open(str(self.session_dir / name), flags)
                try:
                    before = os.fstat(descriptor)
                    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                        raise ValueError(
                            f"robot sidecar must be a non-empty regular file: {name}"
                        )
                    inode = (int(before.st_dev), int(before.st_ino))
                    if inode in seen_inodes:
                        raise ValueError("robot sidecar bundle contains aliased inodes")
                    seen_inodes.add(inode)
                    digest, size = self._hash_descriptor(descriptor)
                    after = os.fstat(descriptor)
                    identity = self._identity(before)
                    if self._identity(after) != identity or size != before.st_size:
                        raise ValueError(f"robot sidecar changed while pinning: {name}")
                    self._items[name] = _PinnedSidecar(
                        descriptor=descriptor,
                        identity=identity,
                        sha256=digest,
                        size_bytes=size,
                    )
                except Exception:
                    os.close(descriptor)
                    raise
        except Exception:
            self.close()
            raise
        return self

    @property
    def descriptors(self) -> Mapping[str, int]:
        return {
            name: item.descriptor for name, item in self._items.items()
        }

    @property
    def hashes(self) -> dict[str, str]:
        return {name: item.sha256 for name, item in self._items.items()}

    def read_bytes(self, name: str) -> bytes:
        item = self._items[name]
        chunks = []
        offset = 0
        while offset < item.size_bytes:
            block = os.pread(
                item.descriptor,
                min(1024 * 1024, item.size_bytes - offset),
                offset,
            )
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        if offset != item.size_bytes:
            raise ValueError(f"robot sidecar ended while reading pinned fd: {name}")
        return b"".join(chunks)

    def open_text(self, name: str):
        return os.fdopen(
            os.dup(self._items[name].descriptor),
            "r",
            newline="",
            encoding="utf-8",
        )

    def verify_unchanged(self) -> None:
        for name, item in self._items.items():
            if self._identity(os.fstat(item.descriptor)) != item.identity:
                raise ValueError(f"robot sidecar fd identity changed: {name}")
            digest, size = self._hash_descriptor(item.descriptor)
            if digest != item.sha256 or size != item.size_bytes:
                raise ValueError(f"robot sidecar fd content changed: {name}")

    def close(self) -> None:
        while self._items:
            _name, item = self._items.popitem()
            os.close(item.descriptor)

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()


@dataclass(frozen=True)
class RobotStream:
    t_s: np.ndarray
    state: np.ndarray
    action: np.ndarray
    joint_names: tuple[str, ...]


@dataclass(frozen=True)
class ExactRobotFrameStream:
    """One exact selected MIT action and one observation for every RGB-D frame."""

    frame_index: np.ndarray
    t_exposure_s: np.ndarray
    state: np.ndarray
    action: np.ndarray
    action_tick_seq: np.ndarray
    action_tick_monotonic_ns: np.ndarray
    joint_names: tuple[str, ...]
    action_names: tuple[str, ...]
    selection_records: tuple[dict, ...]
    summary: dict
    sidecar_hashes: dict[str, str]


@dataclass(frozen=True)
class ForceStream:
    t_host_s: np.ndarray
    force_n: np.ndarray
    status: tuple[str, ...]
    device_t_ms: Optional[np.ndarray]


@dataclass(frozen=True)
class TargetStream:
    t_s: np.ndarray
    names: tuple[str, ...]
    uv: np.ndarray
    xyz: tuple[Optional[tuple[float, float, float]], ...]
    xyz_frames: tuple[Optional[str], ...]


@dataclass(frozen=True)
class CameraTimeline:
    frame_index: np.ndarray
    t_exposure_s: np.ndarray
    host_arrival_t_s: np.ndarray
    device_t_ms: Optional[np.ndarray]
    clock_fit: Optional[dict]
    latency_ms: float


@dataclass(frozen=True)
class CausalFrame:
    frame: dict
    task: str
    target_name: Optional[str]


@dataclass(frozen=True)
class RewardSpan:
    t_start: float
    t_end: float
    reward: float


@dataclass(frozen=True)
class VerifiedSource:
    """A provenance input whose identity was authenticated before conversion."""

    path: Path
    sha256: str


def _strictly_increasing(values: np.ndarray, stream: str) -> None:
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{stream} timestamps must be non-empty and finite")
    if values.size > 1 and not np.all(np.diff(values) > 0):
        raise ValueError(f"{stream} timestamps must be strictly increasing")


def asof_within(ts, query_t: float, *, max_age_s: float, stream: str) -> tuple[int, float]:  # noqa: ANN001
    """Latest sample at or before query time, with a bounded nonnegative age."""
    values = np.asarray(ts, dtype=np.float64)
    _strictly_increasing(values, stream)
    i = bisect_right(values.tolist(), float(query_t)) - 1
    if i < 0:
        raise ValueError(f"no past {stream} sample for t={query_t:.6f}")
    age = float(query_t) - float(values[i])
    if age < -1e-9:  # defensive: bisect_right must make this impossible
        raise ValueError(f"{stream} alignment selected a future sample")
    if age > float(max_age_s) + 1e-12:
        raise ValueError(
            f"{stream} alignment gap {age:.6f}s exceeds max age {max_age_s:.6f}s "
            f"at t={query_t:.6f}"
        )
    return i, max(0.0, age)


def _csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    if len(rows) < 2 or not rows[0]:
        raise ValueError(f"{path.name} must have a header and at least one data row")
    return rows[0], rows[1:]


def _robot_columns(header: list[str]) -> tuple[str, list[str], list[str]]:
    if not header or header[0] not in ("t", "host_t_s"):
        raise ValueError("robot.csv header must start with t or host_t_s")
    body = header[1:]
    q_cols = [name for name in body if name.startswith("q")]
    a_cols = [name for name in body if name.startswith("a")]
    if not q_cols or len(q_cols) != len(a_cols) or body != q_cols + a_cols:
        raise ValueError("robot.csv header needs ordered, matching q*/a* column groups")

    def suffix(name: str, prefix: str) -> str:
        value = name[len(prefix):]
        return value[1:] if value.startswith("_") else value

    if [suffix(c, "q") for c in q_cols] != [suffix(c, "a") for c in a_cols]:
        raise ValueError("robot.csv header q*/a* joint identities do not match")
    return header[0], q_cols, a_cols


def _header_joint_id(column: str, prefix: str) -> str:
    """Map compact q1/a1 headers to the manifest's canonical joint_1 identity."""
    suffix = column[len(prefix):]
    suffix = suffix[1:] if suffix.startswith("_") else suffix
    if not suffix:
        raise ValueError(f"robot.csv column {column!r} has no joint identity")
    return f"joint_{int(suffix)}" if suffix.isdigit() else suffix


def read_robot(session_dir: str | Path, manifest: dict) -> RobotStream:
    path = Path(session_dir) / "robot.csv"
    header, rows = _csv_rows(path)
    _, q_cols, a_cols = _robot_columns(header)
    arm = manifest.get("arm") or {}
    joint_names = tuple(arm.get("joint_names") or ())
    if (len(joint_names) != len(q_cols) or arm.get("state_unit") != "rad"
            or arm.get("action_type") != "position_target"
            or arm.get("side") not in ("left", "right")):
        raise ValueError("robot.csv header requires matching manifest arm metadata")
    header_joint_names = tuple(_header_joint_id(column, "q") for column in q_cols)
    if len(set(joint_names)) != len(joint_names) or header_joint_names != joint_names:
        raise ValueError(
            "robot.csv header joint identities/order do not match manifest joint_names: "
            f"header={header_joint_names}, manifest={joint_names}"
        )
    expected_width = len(header)
    try:
        values = np.asarray([[float(x) for x in row] for row in rows if len(row) == expected_width],
                            dtype=np.float64)
    except ValueError as exc:
        raise ValueError("robot.csv contains non-numeric data") from exc
    if len(values) != len(rows) or values.shape != (len(rows), expected_width):
        raise ValueError("robot.csv contains a malformed row")
    if not np.isfinite(values).all():
        raise ValueError("robot.csv contains non-finite data")
    t = values[:, 0]
    _strictly_increasing(t, "robot")
    nj = len(q_cols)
    return RobotStream(
        t_s=t, state=values[:, 1:1 + nj].astype(np.float32),
        action=values[:, 1 + nj:1 + 2 * nj].astype(np.float32), joint_names=joint_names,
    )


def _strict_json_object(pairs):  # noqa: ANN001
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _decode_strict_json(raw: bytes, *, label: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value is forbidden: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_strict_json(path: Path, *, label: str) -> dict:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is not strict JSON: {exc}") from exc
    return _decode_strict_json(raw, label=label)


def _exact_action_names(joint_names: tuple[str, ...]) -> tuple[str, ...]:
    names = []
    for parameter in ("kp", "kd", "q", "dq", "tau"):
        names.extend(f"arm_mit_{parameter}.{joint}" for joint in joint_names)
    names.extend(
        f"gripper_mit_{parameter}" for parameter in ("kp", "kd", "q", "dq", "tau")
    )
    return tuple(names)


def _summary_robot_contract(summary: Mapping):  # noqa: ANN001
    """Reconstruct the hash-pinned parser contract used for canonical replay.

    Physical limits are not rediscovered by the converter.  The exact values
    committed by the capture writer are reconstructed together with the
    activated config/pair hashes, so replay verifies byte semantics, counts,
    timing, source closure, bounds, and exact actions against one contract.
    """

    if (
        summary.get("schema") not in {
            EXACT_ROBOT_SUMMARY_SCHEMA_V2,
            EXACT_ROBOT_SUMMARY_SCHEMA,
            EXACT_ROBOT_SUMMARY_SCHEMA_V4,
        }
        or summary.get("side") != "left"
        or summary.get("controller_mode") != "unilateral"
        or summary.get("leader_can_interface") != "can1"
        or summary.get("follower_can_interface") != "can3"
    ):
        raise ValueError(
            "canonical robot sidecar requires readable exact-control v2/v3/v4 "
            "left can1 leader -> can3 follower"
        )
    joint_names = tuple(summary.get("joint_names") or ())
    canonical = tuple(f"openarm_left_joint{index}" for index in range(1, 8))
    if joint_names != canonical:
        raise ValueError("canonical robot sidecar joint order is not left OpenArm")
    for field in (
        "urdf_sha256",
        "joint_mapper_sha256",
        "camera_profile_sha256",
        "timing_artifact_sha256",
        "pair_contract_sha256",
        "controller_binary_sha256",
        "controller_source_manifest_sha256",
        "follower_config_sha256",
        "openarm_sdk_sha256",
    ):
        _validate_sha256(summary.get(field), label=f"robot {field}")

    package_root = str(Path(__file__).resolve().parents[1])
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from ausc.perception.robot.robot_state_stream import RobotStateContract

    raw_limits = summary.get("joint_position_limits_rad")
    raw_gripper_limits = summary.get("gripper_motor_position_limits_rad")
    raw_velocity = summary.get("max_abs_joint_velocity_rad_s")
    if (
        not isinstance(raw_limits, list)
        or len(raw_limits) != 7
        or any(not isinstance(pair, list) or len(pair) != 2 for pair in raw_limits)
        or not isinstance(raw_gripper_limits, list)
        or len(raw_gripper_limits) != 2
        or isinstance(raw_velocity, bool)
        or not isinstance(raw_velocity, (int, float))
    ):
        raise ValueError("robot summary is missing its activated physical bounds")
    return RobotStateContract(
        side="left",
        controller_mode="unilateral",
        joint_names=joint_names,
        joint_position_limits_rad=tuple(
            (float(pair[0]), float(pair[1])) for pair in raw_limits
        ),
        leader_can_interface="can1",
        follower_can_interface="can3",
        urdf_sha256=str(summary["urdf_sha256"]),
        joint_mapper_sha256=str(summary["joint_mapper_sha256"]),
        camera_profile_sha256=str(summary["camera_profile_sha256"]),
        timing_artifact_sha256=str(summary["timing_artifact_sha256"]),
        pair_contract_sha256=str(summary["pair_contract_sha256"]),
        controller_binary_sha256=str(summary["controller_binary_sha256"]),
        controller_source_manifest_sha256=str(summary["controller_source_manifest_sha256"]),
        follower_config_sha256=str(summary["follower_config_sha256"]),
        openarm_sdk_sha256=str(summary["openarm_sdk_sha256"]),
        max_abs_joint_velocity_rad_s=float(raw_velocity),
        gripper_motor_position_limits_rad=tuple(
            float(value) for value in raw_gripper_limits
        ),
    )


def _canonical_robot_summary(
    session_dir: Path,
    manifest: Mapping,
    sidecars: _PinnedExactRobotSidecars,
) -> dict:
    summary = _decode_strict_json(
        sidecars.read_bytes("robot_capture_summary.json"),
        label="robot capture summary",
    )
    manifest_summary = manifest.get("robot")
    if not isinstance(manifest_summary, Mapping) or dict(manifest_summary) != summary:
        raise ValueError("manifest robot summary is missing or differs from commit marker")
    contract = _summary_robot_contract(summary)
    try:
        from ausc.perception.robot.robot_session_capture import (
            validate_robot_capture_artifacts,
        )

        replayed = validate_robot_capture_artifacts(
            session_dir,
            contract,
            artifact_fds=sidecars.descriptors,
        )
    except Exception as exc:  # canonical validator owns its domain exception
        raise ValueError(f"robot canonical replay failed: {exc}") from exc
    if dict(replayed) != summary:
        raise ValueError("robot canonical replay summary differs from commit marker")
    return summary


def _finite_csv_float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"exact control CSV field {field!r} is invalid") from exc
    if not np.isfinite(value):
        raise ValueError(f"exact control CSV field {field!r} is non-finite")
    return value


def _exact_action_from_csv_row(
    row: Mapping[str, str], joint_names: tuple[str, ...],
) -> tuple[np.ndarray, dict]:
    vectors = {}
    for parameter, unit in (
        ("kp", "gain"),
        ("kd", "gain"),
        ("q", "motor_rad"),
        ("dq", "motor_rad_s"),
        ("tau", "motor_torque"),
    ):
        vectors[parameter] = [
            _finite_csv_float(row, f"action.arm_mit_{parameter}.{joint}.{unit}")
            for joint in joint_names
        ]
    gripper = {
        parameter: _finite_csv_float(row, f"action.gripper_mit_{parameter}.{unit}")
        for parameter, unit in (
            ("kp", "gain"),
            ("kd", "gain"),
            ("q", "motor_rad"),
            ("dq", "motor_rad_s"),
            ("tau", "motor_torque"),
        )
    }
    try:
        tick = int(row["control_tick_seq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("exact control CSV has an invalid control_tick_seq") from exc
    payload = {
        "control_tick_seq": tick,
        "follower_arm_mit_kp": vectors["kp"],
        "follower_arm_mit_kd": vectors["kd"],
        "follower_arm_mit_q": vectors["q"],
        "follower_arm_mit_dq": vectors["dq"],
        "follower_arm_mit_tau": vectors["tau"],
        "follower_gripper_mit_kp": gripper["kp"],
        "follower_gripper_mit_kd": gripper["kd"],
        "follower_gripper_mit_q": gripper["q"],
        "follower_gripper_mit_dq": gripper["dq"],
        "follower_gripper_mit_tau": gripper["tau"],
    }
    vector = np.asarray(
        [value for parameter in ("kp", "kd", "q", "dq", "tau")
         for value in vectors[parameter]]
        + [gripper[parameter] for parameter in ("kp", "kd", "q", "dq", "tau")],
        dtype=np.float32,
    )
    return vector, payload


def _selected_action_sha256(payload: Mapping) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def read_exact_robot_frames(
    session_dir: str | Path,
    camera: CameraTimeline,
    manifest: Mapping,
) -> ExactRobotFrameStream:
    """Canonical-replay v2/v3 sidecars and select one real action per RGB-D frame.

    Observation state is the capture writer's verified interpolation at exposure
    nominal.  Action is never interpolated: the selected tick is the latest
    actual 1 kHz tick whose complete arm+gripper control call ended at or
    before exposure nominal.
    """

    session = Path(session_dir)
    with _PinnedExactRobotSidecars(session) as sidecars:
        return _read_exact_robot_frames_from_pinned(
            session, camera, manifest, sidecars,
        )


def _read_exact_robot_frames_from_pinned(
    session: Path,
    camera: CameraTimeline,
    manifest: Mapping,
    sidecars: _PinnedExactRobotSidecars,
) -> ExactRobotFrameStream:
    """Consume one already-pinned bundle without reopening its paths."""

    summary = _canonical_robot_summary(session, manifest, sidecars)
    is_v3 = summary.get("schema") == EXACT_ROBOT_SUMMARY_SCHEMA
    is_v4 = summary.get("schema") == EXACT_ROBOT_SUMMARY_SCHEMA_V4
    has_tx_status = is_v3 or is_v4
    expected_frame_schema = (
        EXACT_ROBOT_FRAME_SCHEMA_V4
        if is_v4
        else (
            EXACT_ROBOT_FRAME_SCHEMA
            if is_v3
            else EXACT_ROBOT_FRAME_SCHEMA_V2
        )
    )
    if (
        summary.get("action_semantics") != "exact_mitparam_consumed_at_control_tick"
        or summary.get("action_interpolation") != "forbidden"
    ):
        raise ValueError("robot summary does not retain exact MIT action semantics")
    arm = manifest.get("arm") or {}
    joint_names = tuple(summary["joint_names"])
    if (
        arm.get("side") != "left"
        or tuple(arm.get("joint_names") or ()) != joint_names
        or arm.get("state_unit") != "rad"
        or arm.get("action_type") != "exact_mitparam_consumed_at_control_tick"
    ):
        raise ValueError("manifest arm metadata does not match exact MIT robot sidecar")

    try:
        frame_rows = [
            json.loads(
                line,
                object_pairs_hook=_strict_json_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value is forbidden: {token}")
                ),
            )
            for line in sidecars.read_bytes(
                "robot_frame_index.jsonl"
            ).decode("utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"robot frame index is not strict JSONL: {exc}") from exc
    camera_indexes = [int(value) for value in np.asarray(camera.frame_index).tolist()]
    robot_indexes = [row.get("frame_index") for row in frame_rows]
    if (
        len(frame_rows) != int(summary.get("frame_count", -1))
        or robot_indexes != camera_indexes
    ):
        raise ValueError("robot frame index does not close on the RGB-D camera timeline")
    camera_id = (manifest.get("camera") or {}).get("camera_id")
    if not isinstance(camera_id, str) or any(
        row.get("schema") != expected_frame_schema
        or row.get("camera_id") != camera_id
        for row in frame_rows
    ):
        raise ValueError("robot frame index camera identity does not match manifest")
    if has_tx_status and any(
        row.get("send_status_semantics") != EXACT_ROBOT_TX_STATUS_SEMANTICS
        or row.get("motor_ack_observable") is not False
        or row.get("motor_ack_reason") != EXACT_ROBOT_MOTOR_ACK_REASON
        or row.get("socketcan_tx_enqueue_failure_tick_count") != 0
        or row.get("socketcan_tx_enqueue_success_all_ticks") is not True
        for row in frame_rows
    ):
        raise ValueError("robot frame index lacks clean SocketCAN TX queued evidence")
    if is_v4 and any(
        row.get("feedback_liveness_observable") is not True
        or row.get("feedback_liveness_reason")
        != "openarm_can_1_2_2_matching_motor_id_valid_state_decode"
        or row.get("feedback_liveness_semantics")
        != "counter_generation_increment_only_after_matching_motor_id_valid_state_decode"
        or row.get("feedback_max_age_ns") != 20_000_000
        or row.get("feedback_max_gap_ns") != 20_000_000
        or row.get("feedback_all_ticks_fresh") is not True
        or row.get("feedback_gap_violation_counts_first")
        != row.get("feedback_gap_violation_counts_last")
        for row in frame_rows
    ):
        raise ValueError("robot frame index lacks clean v4 feedback liveness evidence")
    camera_meta = manifest.get("camera") or {}
    camera_timing_hash = (camera_meta.get("timing") or {}).get("calibration_sha256")
    if (
        camera_timing_hash is not None
        and camera_timing_hash != summary.get("timing_artifact_sha256")
    ):
        raise ValueError("robot timing artifact hash does not match camera manifest")
    camera_profile_hash = camera_meta.get("camera_profile_sha256")
    if (
        camera_profile_hash is not None
        and camera_profile_hash != summary.get("camera_profile_sha256")
    ):
        raise ValueError("robot camera profile hash does not match camera manifest")

    inventory = summary.get("hash_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != set(EXACT_ROBOT_DATA_FILES):
        raise ValueError("robot sidecar hash inventory is incomplete")
    sidecar_hashes = {}
    for name in EXACT_ROBOT_DATA_FILES:
        digest = _validate_sha256(
            (inventory.get(name) or {}).get("sha256")
            if isinstance(inventory.get(name), Mapping) else None,
            label=f"robot sidecar {name}",
        )
        if sidecars.hashes[name] != digest:
            raise ValueError(f"robot sidecar hash changed after canonical replay: {name}")
        sidecar_hashes[name] = digest
    sidecar_hashes["robot_capture_summary.json"] = sidecars.hashes[
        "robot_capture_summary.json"
    ]

    states = []
    actions = []
    selected_ticks = []
    selected_times = []
    selections = []
    window: deque[tuple[int, int, Mapping[str, str]]] = deque()
    previous_csv_tick = None
    previous_csv_time = None
    with sidecars.open_text("robot_exact_control.csv") as fp:
        reader = csv.DictReader(fp)

        def next_control_row() -> tuple[int, int, Mapping[str, str]]:
            nonlocal previous_csv_tick, previous_csv_time
            try:
                row = next(reader)
                tick = int(row["control_tick_seq"])
                timestamp = int(row["observation_monotonic_ns"])
            except StopIteration as exc:
                raise ValueError("exact control CSV ended before robot frame index") from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("exact control CSV tick identity is invalid") from exc
            if (
                previous_csv_tick is not None
                and (tick != previous_csv_tick + 1 or timestamp <= previous_csv_time)
            ):
                raise ValueError("exact control CSV ticks are not contiguous and time ordered")
            if has_tx_status:
                try:
                    tx_clean = (
                        int(row["arm_tx_expected_frame_count"]) == 7
                        and int(row["arm_tx_full_frame_write_count"]) == 7
                        and int(row["arm_tx_failed_frame_write_count"]) == 0
                        and int(row["gripper_tx_expected_frame_count"]) == 1
                        and int(row["gripper_tx_full_frame_write_count"]) == 1
                        and int(row["gripper_tx_failed_frame_write_count"]) == 0
                        and row["socketcan_tx_enqueue_success"] == "true"
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "exact control CSV SocketCAN TX fields are invalid"
                    ) from exc
                if not tx_clean:
                    raise ValueError(
                        "exact control CSV contains a SocketCAN TX enqueue failure"
                    )
            previous_csv_tick = tick
            previous_csv_time = timestamp
            return tick, timestamp, row

        for row in frame_rows:
            try:
                first_tick = int(row["exact_action_tick_first"])
                last_tick = int(row["exact_action_tick_last"])
                nominal_ns = int(row["exposure_nominal_ns"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("robot frame exact action range is invalid") from exc
            while window and window[0][0] < first_tick:
                window.popleft()
            while not window or window[-1][0] < last_tick:
                control = next_control_row()
                if control[0] >= first_tick:
                    window.append(control)
            bracket = [item for item in window if first_tick <= item[0] <= last_tick]
            if (
                len(bracket) != last_tick - first_tick + 1
                or bracket[0][0] != first_tick
                or bracket[-1][0] != last_tick
            ):
                raise ValueError("exact control CSV does not close the frame action bracket")
            try:
                eligible = [
                    item
                    for item in bracket
                    if int(item[2]["control_call_end_monotonic_ns"]) <= nominal_ns
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "exact control CSV control-call completion time is invalid"
                ) from exc
            if not eligible:
                raise ValueError(
                    "frame action bracket has no completed real tick at or before nominal"
                )
            selected_tick, selected_time, selected_row = eligible[-1]
            selected_control_end = int(
                selected_row["control_call_end_monotonic_ns"]
            )
            action, action_payload = _exact_action_from_csv_row(selected_row, joint_names)
            observation = row.get("observation_at_nominal")
            if not isinstance(observation, Mapping):
                raise ValueError("robot frame nominal observation is missing")
            state = np.asarray(observation.get("follower_q_rad"), dtype=np.float32)
            if state.shape != (len(joint_names),) or not np.isfinite(state).all():
                raise ValueError("robot frame nominal follower observation is invalid")
            states.append(state)
            actions.append(action)
            selected_ticks.append(selected_tick)
            selected_times.append(selected_control_end)
            selections.append({
                "schema_version": EXACT_ACTION_SELECTION_SCHEMA,
                "camera_id": camera_id,
                "frame_index": int(row["frame_index"]),
                "exposure_nominal_ns": nominal_ns,
                "selection_rule": EXACT_ACTION_SELECTION_RULE,
                "selected_control_tick_seq": selected_tick,
                "selected_observation_monotonic_ns": selected_time,
                "selected_control_call_end_monotonic_ns": selected_control_end,
                "selected_exact_action_sha256": _selected_action_sha256(action_payload),
                "source_exact_action_tick_first": first_tick,
                "source_exact_action_tick_last": last_tick,
                "source_exact_action_range_sha256": row["exact_action_sha256"],
                "source_exact_control_csv_sha256": sidecar_hashes[
                    "robot_exact_control.csv"
                ],
                "source_robot_summary_sha256": sidecar_hashes[
                    "robot_capture_summary.json"
                ],
                "action_interpolation": "forbidden",
            })

    sidecars.verify_unchanged()
    if sidecars.hashes != sidecar_hashes:
        raise ValueError("robot sidecar hashes changed while selecting exact actions")
    state_array = np.asarray(states, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    tick_array = np.asarray(selected_ticks, dtype=np.int64)
    time_array = np.asarray(selected_times, dtype=np.int64)
    for array in (state_array, action_array, tick_array, time_array):
        array.setflags(write=False)
    return ExactRobotFrameStream(
        frame_index=np.asarray(camera_indexes, dtype=np.int64),
        t_exposure_s=np.asarray(camera.t_exposure_s, dtype=np.float64),
        state=state_array,
        action=action_array,
        action_tick_seq=tick_array,
        action_tick_monotonic_ns=time_array,
        joint_names=joint_names,
        action_names=_exact_action_names(joint_names),
        selection_records=tuple(selections),
        summary=dict(summary),
        sidecar_hashes=sidecar_hashes,
    )


def require_exact_robot_training_eligibility(summary: Mapping) -> None:
    """Reject exact-control schemas that cannot prove motor feedback liveness."""

    if summary.get("schema") == EXACT_ROBOT_SUMMARY_SCHEMA_V2:
        raise ValueError(
            "robot send-status eligibility blocker is active; v2 exact actions may be "
            "read for commissioning but cannot be published as training data"
        )
    if summary.get("schema") == EXACT_ROBOT_SUMMARY_SCHEMA:
        raise ValueError(
            "robot feedback liveness eligibility blocker is active for v3; its "
            "SocketCAN TX enqueue evidence is diagnostic-only and cannot be "
            "published as training data"
        )
    if summary.get("schema") == EXACT_ROBOT_SUMMARY_SCHEMA_V4:
        violation_delta = summary.get("feedback_gap_violation_count_delta")
        baseline = summary.get("feedback_start_baseline")
        if (
            summary.get("send_status_observable") is not True
            or summary.get("send_status_reason") != EXACT_ROBOT_TX_STATUS_REASON
            or summary.get("send_status_semantics")
            != EXACT_ROBOT_TX_STATUS_SEMANTICS
            or summary.get("motor_ack_observable") is not False
            or summary.get("motor_ack_reason") != EXACT_ROBOT_MOTOR_ACK_REASON
            or summary.get("socketcan_tx_enqueue_failure_tick_count") != 0
            or summary.get("feedback_liveness_observable") is not True
            or summary.get("feedback_liveness_reason")
            != "openarm_can_1_2_2_matching_motor_id_valid_state_decode"
            or summary.get("feedback_liveness_semantics")
            != "counter_generation_increment_only_after_matching_motor_id_valid_state_decode"
            or summary.get("feedback_max_age_ns") != 20_000_000
            or summary.get("feedback_max_gap_ns") != 20_000_000
            or summary.get("feedback_counter_progress_all_motors") is not True
            or summary.get("feedback_all_ticks_fresh") is not True
            or summary.get("feedback_gap_clean") is not True
            or summary.get("feedback_clean") is not True
            or not isinstance(violation_delta, list)
            or violation_delta != [0] * 8
            or not isinstance(baseline, Mapping)
            or baseline.get("schema") != "wam_openarm_feedback_start_baseline_v1"
            or summary.get("trajectory_validation_schema")
            != "wam_openarm_trajectory_validation_v1"
            or summary.get("trajectory_min_whole_7d_range_rad") != 1.0e-4
            or summary.get("command_significant_range_rad") != 1.0e-2
            or summary.get("command_response_frozen_range_rad") != 1.0e-4
            or summary.get("trajectory_nondegenerate") is not True
            or summary.get("command_response_consistent") is not True
            or summary.get("command_response_violation_joint_names") != []
            or summary.get("training_eligible") is not True
            or summary.get("eligibility_blockers") != []
        ):
            raise ValueError(
                "robot feedback v4 receipt is incomplete, stale, or not session-clean"
            )
        return
    raise ValueError(
        "robot exact-control summary schema has no accepted feedback-liveness proof"
    )


def read_force(session_dir: str | Path) -> ForceStream:
    path = Path(session_dir) / "force.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError("force.csv has no header")
        time_col = "host_t_s" if "host_t_s" in reader.fieldnames else "t" if "t" in reader.fieldnames else None
        if time_col is None or "force_n" not in reader.fieldnames:
            raise ValueError("force.csv requires host_t_s/t and force_n columns")
        rows = list(reader)
    if not rows:
        raise ValueError("force.csv has no data rows")
    try:
        t = np.asarray([float(row[time_col]) for row in rows], dtype=np.float64)
        force = np.asarray([float(row["force_n"]) for row in rows], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("force.csv contains non-numeric timestamps or force") from exc
    if not np.isfinite(force).all():
        raise ValueError("force.csv contains non-finite force")
    _strictly_increasing(t, "force")
    device = None
    if "device_t_ms" in (reader.fieldnames or []) and all(row.get("device_t_ms", "") for row in rows):
        device = np.asarray([float(row["device_t_ms"]) for row in rows], dtype=np.float64)
        _strictly_increasing(device, "force device")
    return ForceStream(
        t_host_s=t, force_n=force,
        status=tuple(row.get("status", "") for row in rows), device_t_ms=device,
    )


def read_targets(
    session_dir: str | Path, camera: Optional[CameraTimeline] = None,
) -> TargetStream:
    path = Path(session_dir) / "targets.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise ValueError("targets.jsonl has no records")
    if camera is None:
        t = np.asarray([float(item["t"]) for item in records], dtype=np.float64)
    else:
        exposure_by_frame = {
            int(index): float(exposure)
            for index, exposure in zip(camera.frame_index, camera.t_exposure_s)
        }
        try:
            t = np.asarray([
                exposure_by_frame[int(item["camera_frame_index"])] for item in records
            ], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("target record is not bound to a known camera_frame_index") from exc
    _strictly_increasing(t, "target")
    uv = np.asarray([item["uv"] for item in records], dtype=np.float32)
    if uv.shape != (len(records), 2) or not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv > 1):
        raise ValueError("targets.jsonl uv must be finite normalized pairs")
    xyz = []
    frames = []
    for item in records:
        raw = item.get("xyz")
        frame = item.get("xyz_frame")
        if raw is None:
            xyz.append(None); frames.append(None)
            continue
        point = tuple(float(x) for x in raw)
        if len(point) != 3 or not np.isfinite(point).all() or frame not in ("camera", "world", "robot_base"):
            raise ValueError("targets.jsonl xyz requires three finite values and an explicit frame")
        xyz.append(point); frames.append(frame)
    return TargetStream(
        t_s=t, names=tuple(str(item["name"]) for item in records), uv=uv,
        xyz=tuple(xyz), xyz_frames=tuple(frames),
    )


def build_causal_frame(
    query_t: float,
    robot: RobotStream,
    force: ForceStream,
    targets: Optional[TargetStream],
    *,
    reward: float,
    done: bool,
    robot_max_age_s: float,
    force_max_age_s: float,
    target_max_age_s: float,
) -> CausalFrame:
    ir, robot_age = asof_within(
        robot.t_s, query_t, max_age_s=robot_max_age_s, stream="robot",
    )
    iff, force_age = asof_within(
        force.t_host_s, query_t, max_age_s=force_max_age_s, stream="force",
    )
    force_status = force.status[iff]
    # P0-4: an over-threshold reading is a VALID physician-reference measurement, not a
    # sensor fault -- it must NOT reject the session. contact_force_valid marks whether
    # this is a genuine sensor reading (True for any native status, INCLUDING
    # over-threshold); it is never gated on the force being "safe".
    force_valid = force_status in ("OK", "WARN", "SCREENED_LIMIT", "HARD_LIMIT")
    frame = {
        "observation.state": robot.state[ir],
        "observation.robot_state_age_s": np.asarray([robot_age], np.float32),
        "observation.images.top": None,  # populated only after the corresponding video frame is decoded
        "observation.contact_force_n": np.asarray([force.force_n[iff]], np.float32),
        "observation.contact_force_valid": np.asarray([force_valid], bool),
        "observation.contact_force_age_s": np.asarray([force_age], np.float32),
        "action": robot.action[ir],
        "next.reward": np.asarray([float(reward)], np.float32),
        "next.done": np.asarray([bool(done)], bool),
    }
    task = DEFAULT_TASK
    target_name = None
    if targets is not None:
        it, target_age = asof_within(
            targets.t_s, query_t, max_age_s=target_max_age_s, stream="target",
        )
        target_name = targets.names[it]
        # The VLA "L": ground the opaque code (R1..L5) in its anatomical meaning so the
        # policy can reuse a language model's anatomy understanding. Phrasing augmentation is
        # applied once, later, seeded per (session, site).
        task = site_instruction(target_name)
        frame["observation.target_point_2d"] = targets.uv[it]
        frame["observation.target_age_s"] = np.asarray([target_age], np.float32)
        if targets.xyz[it] is not None:
            frame["observation.target_point_3d"] = np.asarray(targets.xyz[it], np.float32)
    return CausalFrame(frame=frame, task=task, target_name=target_name)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trusted_verifier_path(
    value: str | Path | None, *, environment_name: str, label: str,
) -> Path:
    configured = value
    if configured is None:
        configured = os.environ.get(environment_name)
    if not isinstance(configured, (str, Path)) or not str(configured):
        raise ValueError(
            f"{label} is required (argument or trusted process environment "
            f"{environment_name})"
        )
    path = Path(configured)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute verifier-controlled path")
    return path


def _verify_training_session_contract(
    session_dir: str | Path, *,
    trusted_finalization_registry: str | Path | None = None,
    trusted_snapshot_root: str | Path | None = None,
    trusted_session_id: str | None = None,
) -> dict:
    """Run the shared promotion-v2 outer-chain verifier and normalize failures."""
    registry = _trusted_verifier_path(
        trusted_finalization_registry,
        environment_name=TRUSTED_FINALIZATION_REGISTRY_ENV,
        label="trusted finalization registry",
    )
    snapshots = _trusted_verifier_path(
        trusted_snapshot_root,
        environment_name=TRUSTED_SNAPSHOT_ROOT_ENV,
        label="trusted training snapshot root",
    )
    try:
        from deployment.artifact_contract import (
            ArtifactError,
            load_trusted_training_finalization_record,
            verify_training_session_contract,
        )
    except ImportError as exc:  # pragma: no cover - deployment packaging failure
        raise RuntimeError(
            "promotion-v2 verifier is unavailable in the converter runtime"
        ) from exc
    try:
        source = Path(session_dir).resolve()
        session_selector = trusted_session_id if trusted_session_id is not None else source.name
        trusted_record = load_trusted_training_finalization_record(
            registry, session_id=session_selector,
        )
        verified = verify_training_session_contract(
            source,
            trusted_finalization_record=trusted_record,
            trusted_snapshot_root=snapshots,
        )
    except (ArtifactError, OSError, ValueError) as exc:
        raise ValueError(f"training promotion v2 verification failed: {exc}") from exc
    if verified.get("promotion_contract_version") != PROMOTION_CONTRACT_VERSION:
        raise ValueError("training promotion v2 verifier returned an unsupported contract")
    if trusted_session_id is None and source.name != verified.get("session_id"):
        raise ValueError(
            "training promotion v2 directory basename must equal verified session_id"
        )
    verified_snapshot = Path(str(verified.get("verified_snapshot_path", "")))
    if (
        not verified_snapshot.is_absolute()
        or not verified_snapshot.is_dir()
        or verified_snapshot.is_symlink()
    ):
        raise ValueError("training promotion v2 verifier returned an unsafe snapshot")
    return dict(verified)


def _verify_audio_capture_artifacts(session_dir: str | Path) -> dict:
    """Decode the frozen WAV and recompute the audio nondegeneracy receipt."""
    session = Path(session_dir)
    metadata = _read_strict_json(
        session / "audio_capture.json", label="audio capture metadata",
    )
    sample_rate = metadata.get("sample_rate")
    n_samples = metadata.get("n_samples")
    if type(sample_rate) is not int or type(n_samples) is not int:
        raise ValueError("audio capture sample identity is invalid")
    if (
        metadata.get("schema_version") != "wam_audio_stream_capture_v1"
        or metadata.get("status") != "sensors_finalized"
        or metadata.get("training_eligible") is not False
        or sample_rate <= 0
        or n_samples <= 0
    ):
        raise ValueError("audio capture metadata is not fail-closed finalized")
    wav_path = session / "audio.wav"
    if wav_path.is_symlink() or not wav_path.is_file():
        raise ValueError("audio.wav is missing or unsafe")
    claimed_sha = metadata.get("sha256")
    if not isinstance(claimed_sha, str) or _sha256(wav_path) != claimed_sha:
        raise ValueError("audio.wav SHA-256 does not match audio capture metadata")

    package_root = str(Path(__file__).resolve().parents[1])
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from ausc.audio.audio_stream import (
        AudioSignalAuthenticityAccumulator,
        AudioSignalAuthenticityError,
    )

    analyzer = AudioSignalAuthenticityAccumulator(sample_rate)
    decoded = 0
    try:
        with wave.open(str(wav_path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != sample_rate
                or source.getnframes() != n_samples
                or source.getcomptype() != "NONE"
            ):
                raise ValueError("audio.wav PCM identity does not match metadata")
            while decoded < n_samples:
                raw = source.readframes(min(65_536, n_samples - decoded))
                if not raw or len(raw) % 2:
                    raise ValueError("audio.wav ended before its declared sample count")
                pcm = np.frombuffer(raw, dtype="<i2")
                analyzer.update(pcm)
                decoded += int(pcm.size)
    except (wave.Error, EOFError, AudioSignalAuthenticityError) as exc:
        raise ValueError(f"audio raw authenticity verification failed: {exc}") from exc
    if decoded != n_samples:
        raise ValueError("audio.wav decoded sample count is incomplete")
    try:
        receipt = analyzer.finish()
    except AudioSignalAuthenticityError as exc:
        raise ValueError(f"audio raw authenticity verification failed: {exc}") from exc
    if metadata.get("signal_authenticity") != receipt:
        raise ValueError("audio authenticity receipt does not match recomputed PCM")
    return receipt


def _content_addressed_identity(
    release_id: object, content_sha256: object, *, label: str,
) -> dict[str, str]:
    digest = _validate_sha256(content_sha256, label=f"{label} SHA-256")
    if (
        not isinstance(release_id, str)
        or not release_id.strip()
        or not release_id.endswith(digest[:20])
    ):
        raise ValueError(f"{label} release is not content-addressed")
    return {"release_id": release_id, "sha256": digest}


def _outer_v2_authorized(outer_contract: object, *, session_id: str) -> bool:
    def is_sha(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and value == value.lower()
            and all(char in "0123456789abcdef" for char in value)
        )

    return (
        isinstance(outer_contract, Mapping)
        and outer_contract.get("promotion_contract_version")
        == PROMOTION_CONTRACT_VERSION
        and outer_contract.get("session_id") == session_id
        and set(outer_contract.get("verified_components") or ())
        == {"audio", "force", "geometry", "quality", "robot"}
        and is_sha(outer_contract.get("manifest_sha256"))
        and is_sha(outer_contract.get("training_authenticity_sha256"))
        and is_sha(outer_contract.get("finalization_sha256"))
        and type(outer_contract.get("verified_raw_files")) is int
        and outer_contract["verified_raw_files"] > 0
    )


def _reverify_component_raw_gates(
    session_dir: str | Path, *, outer_contract: object,
) -> dict:
    """Recompute raw signal gates, then apply the already-verified v2 boundary.

    Geometry and Q raw receipts deliberately cannot authorize themselves.  The
    sole accepted false flag is their exact "receipt alone" blocker after the
    complete promotion-v2 outer chain has independently verified every raw byte
    and component receipt.
    """
    package_root = str(Path(__file__).resolve().parents[1])
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from ausc import realtime_q
    from ausc.perception.tracking import geometry_channel_packet
    from mechanical_sensor.force_feedback import force_capture

    session = Path(session_dir)
    audio = _verify_audio_capture_artifacts(session)
    try:
        force = force_capture.verify_force_capture(session)
    except Exception as exc:
        raise ValueError(f"force raw authenticity verification failed: {exc}") from exc
    force_authenticity = getattr(force, "signal_authenticity", None)
    if (
        getattr(force, "status", None) != "OK"
        or not isinstance(force_authenticity, Mapping)
        or force_authenticity.get("status") != "PASS"
        or force_authenticity.get("promotion_eligible") is not True
        or force_authenticity.get("promotion_blockers") != []
    ):
        raise ValueError("force raw authenticity is not training-signal eligible")

    try:
        _geometry_entries, geometry = (
            geometry_channel_packet.verify_geometry_packet_log_files(session)
        )
    except Exception as exc:
        raise ValueError(f"geometry raw packet verification failed: {exc}") from exc
    geometry_blockers = tuple(getattr(geometry, "authorization_blockers", ()))
    geometry_receipt_only = (
        getattr(geometry, "status", None) == "PASS"
        and getattr(geometry, "production_authorized", None) is False
        and geometry_blockers
        == (geometry_channel_packet.GEOMETRY_RECEIPT_AUTHORIZATION_BLOCKER,)
    )
    geometry_self_authorized = (
        getattr(geometry, "status", None) == "PASS"
        and getattr(geometry, "production_authorized", None) is True
        and geometry_blockers == ()
    )
    if not geometry_receipt_only and not geometry_self_authorized:
        raise ValueError("geometry raw receipt has non-finalization authorization blockers")
    perception_identity = _content_addressed_identity(
        getattr(geometry, "perception_release_id", None),
        getattr(geometry, "perception_model_sha256", None),
        label="geometry perception model",
    )
    independent_identity = _content_addressed_identity(
        getattr(geometry, "independent_release_id", None),
        getattr(geometry, "independent_model_sha256", None),
        label="geometry independent model",
    )
    if perception_identity == independent_identity:
        raise ValueError("geometry model channels are not independently identified")

    try:
        quality = realtime_q.verify_quality_packet_log(session)
    except Exception as exc:
        raise ValueError(f"quality raw packet verification failed: {exc}") from exc
    quality_blockers = quality.get("promotion_blockers")
    quality_receipt_only = (
        quality.get("status") == "PASS"
        and quality.get("promotion_eligible") is False
        and quality_blockers == [realtime_q.QUALITY_PROMOTION_BLOCKER]
    )
    chain = quality.get("chain_verification")
    if (
        not isinstance(chain, Mapping)
        or chain.get("status") != "PASS"
        or chain.get("chain_integrity") != "PASS"
        or chain.get("training_signal_eligible") is not True
        or not isinstance(chain.get("output_nondegeneracy"), Mapping)
        or chain["output_nondegeneracy"].get("status") != "PASS"
    ):
        raise ValueError("quality raw packet chain is not a nondegenerate training signal")
    quality_authenticity = quality.get("quality_authenticity")
    model = (
        quality_authenticity.get("model")
        if isinstance(quality_authenticity, Mapping) else None
    )
    if not isinstance(model, Mapping):
        raise ValueError("quality model identity is missing")
    quality_identity = _content_addressed_identity(
        model.get("release_id"), model.get("model_sha256"),
        label="quality model identity",
    )
    if not isinstance(quality_authenticity, Mapping):
        raise ValueError("quality model identity is missing")
    policy_sha = _validate_sha256(
        quality_authenticity.get("policy_sha256"), label="quality policy",
    )
    quality_self_authorized = (
        quality.get("status") == "PASS"
        and quality.get("promotion_eligible") is True
        and quality_blockers == []
    )
    if not quality_receipt_only and not quality_self_authorized:
        raise ValueError("quality raw receipt has non-finalization promotion blockers")

    outer_authorized = _outer_v2_authorized(
        outer_contract, session_id=session.name,
    )
    if not outer_authorized:
        raise ValueError(
            "raw geometry/Q receipt requires outer promotion-v2 authorization"
        )
    return {
        "schema_version": "wam_converter_raw_authenticity_v2",
        "status": "PASS",
        "audio": audio,
        "force": dict(force_authenticity),
        "geometry": {
            "status": "PASS",
            "outer_authorized": geometry_receipt_only,
            "packet_count": getattr(geometry, "packet_count", None),
            "packet_log_sha256": getattr(geometry, "packet_log_sha256", None),
            "receipt_sha256": getattr(geometry, "receipt_sha256", None),
            "perception": perception_identity,
            "independent": independent_identity,
        },
        "quality": {
            "status": "PASS",
            "outer_authorized": quality_receipt_only,
            "packet_count": quality.get("packet_count"),
            "packet_log_sha256": quality.get("sha256"),
            "model": quality_identity,
            "policy_sha256": policy_sha,
        },
    }


def _training_contract_file_hashes(
    session_dir: str | Path, manifest: Mapping, verified: Mapping,
) -> dict[str, str]:
    """Pin every byte in an already verified outer contract for the snapshot copy."""
    session = Path(session_dir)
    expected = dict(_validated_artifact_hashes(manifest))
    fixed = {
        "manifest.json": verified.get("manifest_sha256"),
        "FINALIZATION.json": verified.get("finalization_sha256"),
        TRAINING_AUTHENTICITY_NAME: verified.get("training_authenticity_sha256"),
    }
    authenticity = _read_strict_json(
        session / TRAINING_AUTHENTICITY_NAME,
        label="training authenticity receipt",
    )
    components = authenticity.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("training authenticity component references are missing")
    for component, reference in components.items():
        if not isinstance(reference, Mapping):
            raise ValueError(f"training authenticity component is invalid: {component}")
        relative = reference.get("path")
        digest = reference.get("sha256")
        fixed[str(relative or "")] = digest
    for relative, digest in fixed.items():
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or path.as_posix() != relative
            or any(part in ("", ".", "..") for part in path.parts)
            or not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"training contract file reference is invalid: {relative!r}")
        prior = expected.get(relative)
        if prior is not None and prior != digest:
            raise ValueError(f"training contract file hash conflicts: {relative}")
        expected[relative] = digest
    for relative, digest in expected.items():
        path = session / relative
        if not path.is_file() or path.is_symlink() or _sha256(path) != digest:
            raise ValueError(f"training contract file changed after verification: {relative}")
    return expected


def _verify_finalization_seal(
    session_dir: str | Path,
    *,
    manifest_name: str = "manifest.json",
    seal_name: str = "FINALIZATION.json",
    authenticity_name: str = TRAINING_AUTHENTICITY_NAME,
    manifest_bytes: bytes | None = None,
) -> VerifiedSource:
    """Bind the v2 seal to both the manifest and authenticity outer receipt."""
    session = Path(session_dir).resolve()
    manifest_path = session / manifest_name
    seal_path = session / seal_name
    authenticity_path = session / authenticity_name
    if (
        manifest_path.is_symlink()
        or seal_path.is_symlink()
        or authenticity_path.is_symlink()
    ):
        raise ValueError("finalization manifest/seal/authenticity must not be symlinks")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"finalized manifest missing: {manifest_path}")
    if not seal_path.is_file():
        raise FileNotFoundError(f"finalization seal missing: {seal_path}")
    if not authenticity_path.is_file():
        raise FileNotFoundError(
            f"training authenticity receipt missing: {authenticity_path}"
        )
    seal_bytes = seal_path.read_bytes()
    try:
        payload = json.loads(seal_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("finalization seal is not valid UTF-8 JSON") from exc
    expected_fields = {
        "manifest_sha256", "promotion_contract_version", "schema_version",
        "training_authenticity_sha256",
    }

    def valid_sha(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and value == value.lower()
            and all(char in "0123456789abcdef" for char in value)
        )

    expected = payload.get("manifest_sha256") if isinstance(payload, Mapping) else None
    expected_authenticity = (
        payload.get("training_authenticity_sha256")
        if isinstance(payload, Mapping) else None
    )
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_fields
        or payload.get("schema_version") != FINALIZATION_SCHEMA_VERSION
        or payload.get("promotion_contract_version") != PROMOTION_CONTRACT_VERSION
        or not valid_sha(expected)
        or not valid_sha(expected_authenticity)
    ):
        raise ValueError("finalization seal schema or bound hashes are invalid")
    raw = manifest_path.read_bytes() if manifest_bytes is None else manifest_bytes
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(
            f"finalization seal manifest hash mismatch: expected {expected}, got {actual}"
        )
    actual_authenticity = _sha256(authenticity_path)
    if actual_authenticity != expected_authenticity:
        raise ValueError(
            "finalization seal training authenticity hash mismatch: "
            f"expected {expected_authenticity}, got {actual_authenticity}"
        )
    return VerifiedSource(seal_path, hashlib.sha256(seal_bytes).hexdigest())


def _resolve_manifest_calibration_path(
    session_dir: str | Path, value: object, *, stream: str,
) -> Path:
    """Resolve an explicit absolute path or a session-contained relative path."""
    raw = str(value or "")
    if not raw:
        raise ValueError(f"manifest {stream} calibration_path is missing")
    path = Path(raw)
    session = Path(session_dir).resolve()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (session / path).resolve()
        try:
            resolved.relative_to(session)
        except ValueError as exc:
            raise ValueError(
                f"manifest {stream} calibration_path escapes session directory: {raw}"
            ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{stream} calibration missing: {resolved}")
    return resolved


def verify_force_calibration_metadata(session_dir: str | Path, manifest: dict) -> dict:
    """Bind the force calibration path to the exact hash recorded at acquisition."""
    meta = dict(manifest.get("force") or {})
    if meta.get("calibrated") is not True:
        raise ValueError("manifest force calibration is not calibrated=true")
    cal_path = _resolve_manifest_calibration_path(
        session_dir, meta.get("calibration_path"), stream="force",
    )
    expected_hash = str(meta.get("calibration_sha256", "")).lower()
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise ValueError("manifest force calibration_sha256 is missing or invalid")
    actual_hash = _sha256(cal_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"force calibration hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    meta["calibration_path_resolved"] = str(cal_path)
    meta["calibration_sha256"] = expected_hash
    return meta


def _validate_sha256(value: object, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _validate_sync_capture_mode(value: object) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("sync capture_mode must be a JSON object")
    audio_channel = value.get("audio_channel")
    if type(audio_channel) is not int or audio_channel < 0:
        raise ValueError("sync capture_mode audio_channel must be a non-negative integer")
    audio = value.get("audio")
    hall = value.get("hall")
    if not isinstance(audio, Mapping) or not isinstance(hall, Mapping):
        raise ValueError("sync capture_mode audio/hall must be JSON objects")

    def required_text(component: Mapping, field: str) -> str:
        result = component.get(field)
        if not isinstance(result, str) or not result.strip():
            raise ValueError(f"sync capture_mode descriptor {field} must be non-empty")
        return result.strip()

    def optional_text(component: Mapping, field: str) -> str | None:
        result = component.get(field)
        if result is None:
            return None
        if not isinstance(result, str):
            raise ValueError(f"sync capture_mode descriptor {field} must be string or null")
        return result.strip() or None

    def usb_id(component: Mapping, field: str, *, required: bool) -> int | None:
        result = component.get(field)
        if result is None and not required:
            return None
        if type(result) is not int or not 0 < result <= 0xFFFF:
            raise ValueError(f"sync capture_mode descriptor {field} has invalid USB id")
        return result

    portaudio_index = audio.get("portaudio_index")
    max_inputs = audio.get("max_input_channels")
    if type(portaudio_index) is not int or portaudio_index < 0:
        raise ValueError("sync capture_mode audio portaudio_index must be non-negative")
    if type(max_inputs) is not int or max_inputs <= 0 or audio_channel >= max_inputs:
        raise ValueError("sync capture_mode audio channel/input-count contract is invalid")
    audio_serial = optional_text(audio, "usb_serial")
    audio_vid = usb_id(audio, "usb_vid", required=False)
    audio_pid = usb_id(audio, "usb_pid", required=False)
    if audio_serial is not None and (audio_vid is None or audio_pid is None):
        raise ValueError("sync capture_mode audio usb_serial requires usb_vid/usb_pid")
    normalized_audio = {
        "requested_device": required_text(audio, "requested_device"),
        "portaudio_index": portaudio_index,
        "hostapi_name": required_text(audio, "hostapi_name"),
        "device_name": required_text(audio, "device_name"),
        "max_input_channels": max_inputs,
        "usb_serial": audio_serial,
        "usb_vid": audio_vid,
        "usb_pid": audio_pid,
        "usb_identity_source": required_text(audio, "usb_identity_source"),
    }
    if "alsa_card" in audio:
        alsa_card = audio.get("alsa_card")
        if type(alsa_card) is not int or alsa_card < 0:
            raise ValueError("sync capture_mode audio alsa_card must be non-negative")
        normalized_audio["alsa_card"] = alsa_card
    normalized_hall = {
        "requested_port": required_text(hall, "requested_port"),
        "device": required_text(hall, "device"),
        "usb_serial": required_text(hall, "usb_serial"),
        "usb_vid": usb_id(hall, "usb_vid", required=True),
        "usb_pid": usb_id(hall, "usb_pid", required=True),
        "manufacturer": optional_text(hall, "manufacturer"),
        "product": optional_text(hall, "product"),
        "location": optional_text(hall, "location"),
    }
    return {
        "audio_channel": audio_channel,
        "force_calibration_sha256": _validate_sha256(
            value.get("force_calibration_sha256"),
            label="sync capture_mode force_calibration_sha256",
        ),
        "audio": normalized_audio,
        "hall": normalized_hall,
    }


def _match_sync_capture_modes(calibration: dict, runtime: dict) -> dict:
    def same(component: str, field: str) -> None:
        if calibration[component][field] != runtime[component][field]:
            raise ValueError(
                f"sync capture_mode {component} {field} does not match runtime hardware"
            )

    if calibration["audio_channel"] != runtime["audio_channel"]:
        raise ValueError("sync capture_mode audio_channel does not match runtime hardware")
    if calibration["force_calibration_sha256"] != runtime["force_calibration_sha256"]:
        raise ValueError("sync capture_mode force calibration does not match runtime hardware")
    for field in ("usb_serial", "usb_vid", "usb_pid"):
        same("hall", field)
    for field in ("hostapi_name", "max_input_channels"):
        same("audio", field)
    cal_serial = calibration["audio"]["usb_serial"]
    runtime_serial = runtime["audio"]["usb_serial"]
    if cal_serial is None or runtime_serial is None:
        raise ValueError("canonical arm sync requires stable audio USB serial identity")
    for field in ("usb_serial", "usb_vid", "usb_pid"):
        same("audio", field)
    return {
        "hall_usb": "matched",
        "audio_usb": "matched",
        "audio_portaudio_index": (
            "matched" if calibration["audio"]["portaudio_index"]
            == runtime["audio"]["portaudio_index"] else "remapped"
        ),
    }


def _validate_camera_capture_mode(value: object) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("camera capture_mode must be a JSON object")
    backend = value.get("backend")
    camera_id = value.get("camera_id")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("camera capture_mode backend must be a non-empty string")
    backend = backend.strip().lower()
    expected_source = CAMERA_TIME_SOURCE_BY_BACKEND.get(backend)
    if expected_source is None:
        raise ValueError(f"camera capture_mode backend {backend!r} is unsupported")
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError("camera capture_mode camera_id must be a non-empty string")
    width, height, fps = value.get("width"), value.get("height"), value.get("fps")
    if type(width) is not int or width <= 0:
        raise ValueError("camera capture_mode width must be a positive integer")
    if type(height) is not int or height <= 0:
        raise ValueError("camera capture_mode height must be a positive integer")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ValueError("camera capture_mode fps must be a finite positive number")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("camera capture_mode fps must be a finite positive number")
    if value.get("time_source") != expected_source:
        raise ValueError(
            "camera capture_mode time_source does not match backend semantics: "
            f"expected {expected_source!r}, got {value.get('time_source')!r}"
        )
    color_format = value.get("color_format")
    depth_mode = value.get("depth_mode")
    align_mode = value.get("align_mode")
    aggregate_mode = value.get("frame_aggregate_mode")
    if not isinstance(color_format, str) or not color_format.strip():
        raise ValueError("camera capture_mode color_format must be a non-empty string")
    color_format = color_format.strip()
    if backend == "orbbec":
        if color_format not in ORBBEC_COLOR_FORMATS:
            raise ValueError(f"camera capture_mode color_format {color_format!r} is unsupported")
        if not isinstance(depth_mode, Mapping):
            raise ValueError("camera capture_mode depth_mode must be a JSON object")
        depth_width, depth_height = depth_mode.get("width"), depth_mode.get("height")
        depth_fps, depth_format = depth_mode.get("fps"), depth_mode.get("format")
        if type(depth_width) is not int or depth_width <= 0:
            raise ValueError("camera capture_mode depth_mode width must be positive integer")
        if type(depth_height) is not int or depth_height <= 0:
            raise ValueError("camera capture_mode depth_mode height must be positive integer")
        if isinstance(depth_fps, bool) or not isinstance(depth_fps, (int, float)):
            raise ValueError("camera capture_mode depth_mode fps must be finite and positive")
        depth_fps = float(depth_fps)
        if not np.isfinite(depth_fps) or depth_fps <= 0:
            raise ValueError("camera capture_mode depth_mode fps must be finite and positive")
        if not isinstance(depth_format, str) or not depth_format.strip():
            raise ValueError("camera capture_mode depth_mode format must be non-empty")
        depth_mode = {
            "width": depth_width, "height": depth_height, "fps": depth_fps,
            "format": depth_format.strip(),
        }
        if align_mode not in ORBBEC_ALIGN_MODES:
            raise ValueError(f"camera capture_mode align_mode is invalid: {align_mode!r}")
        if aggregate_mode != "full_frame_require":
            raise ValueError(
                "camera capture_mode frame_aggregate_mode must be full_frame_require"
            )
    else:
        if color_format != "opencv_decoded_bgr":
            raise ValueError("camera capture_mode OpenCV color_format must be opencv_decoded_bgr")
        if depth_mode is not None:
            raise ValueError("camera capture_mode depth_mode must be null for OpenCV")
        if align_mode != "disabled" or aggregate_mode != "not_applicable":
            raise ValueError("camera capture_mode OpenCV pipeline modes are invalid")
    return {
        "backend": backend,
        "camera_id": camera_id.strip(),
        "width": width,
        "height": height,
        "fps": fps,
        "time_source": expected_source,
        "color_format": color_format,
        "depth_mode": depth_mode,
        "align_mode": align_mode,
        "frame_aggregate_mode": aggregate_mode,
    }


def verify_sync_metadata(session_dir: str | Path, manifest: dict) -> dict:
    """Validate the raw, unapplied tap offset and its source hash/sign contract."""
    meta = dict(manifest.get("sync") or {})
    if meta.get("applied") is not False:
        raise ValueError("sync offset is missing applied=false or was already applied")
    if (meta.get("offset_definition") != "force_t_minus_audio_t"
            or meta.get("offset_units") != "ms"):
        raise ValueError("sync offset sign/units must be force_t_minus_audio_t in ms")
    cal_path = _resolve_manifest_calibration_path(
        session_dir, meta.get("calibration_path"), stream="sync",
    )
    expected_hash = _validate_sha256(
        meta.get("calibration_sha256"), label="sync calibration_sha256",
    )
    actual_hash = _sha256(cal_path)
    if expected_hash != actual_hash:
        raise ValueError(f"sync calibration hash mismatch: expected {expected_hash}, got {actual_hash}")
    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    if (payload.get("schema_version") != "wam_sync_v1"
            or payload.get("offset_definition") != "force_t_minus_audio_t"):
        raise ValueError("sync calibration file has wrong schema or offset definition")
    manifest_mode = _validate_sync_capture_mode(meta.get("capture_mode"))
    payload_mode = _validate_sync_capture_mode(payload.get("capture_mode"))
    if manifest_mode != payload_mode:
        raise ValueError("sync capture_mode in manifest does not match calibration file")
    runtime_mode = _validate_sync_capture_mode(meta.get("runtime_capture_mode"))
    identity_match = _match_sync_capture_modes(manifest_mode, runtime_mode)
    if meta.get("identity_match") != identity_match:
        raise ValueError("sync identity_match evidence is missing or disagrees with descriptors")
    audio_meta = dict(manifest.get("audio") or {})
    force_meta = dict(manifest.get("force") or {})
    if runtime_mode["audio_channel"] != audio_meta.get("channel"):
        raise ValueError("sync runtime audio channel does not match session audio metadata")
    if runtime_mode["force_calibration_sha256"] != force_meta.get("calibration_sha256"):
        raise ValueError("sync runtime force calibration does not match session metadata")
    if runtime_mode["audio"] != audio_meta.get("hardware_descriptor"):
        raise ValueError("sync runtime audio descriptor does not match session metadata")
    if runtime_mode["hall"] != force_meta.get("hardware_descriptor"):
        raise ValueError("sync runtime Hall descriptor does not match session metadata")
    if (runtime_mode["audio"]["requested_device"] != str(audio_meta.get("device"))
            or runtime_mode["hall"]["requested_port"] != force_meta.get("port")):
        raise ValueError("sync runtime requested device paths do not match session metadata")
    offset = float(meta.get("audio_to_force_offset_ms"))
    if (not np.isfinite(offset)
            or not np.isclose(offset, float(payload.get("audio_to_force_offset_ms")), atol=1e-9)):
        raise ValueError("sync manifest offset does not match calibration file")
    try:
        manifest_std = float(meta["offset_std_ms"])
        payload_std = float(payload["offset_std_ms"])
        manifest_matches = int(meta["n_matched"])
        payload_matches = int(payload["n_matched"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("sync calibration quality metadata is incomplete") from exc
    if (not np.isfinite(manifest_std) or not 0 <= manifest_std <= 5.0
            or not np.isclose(manifest_std, payload_std, atol=1e-9)
            or manifest_matches < 4 or manifest_matches != payload_matches
            or meta.get("unique_matches") is not True or payload.get("unique_matches") is not True
            or meta.get("measurement_method") != "physical_diaphragm_taps"
            or payload.get("measurement_method") != "physical_diaphragm_taps"):
        raise ValueError("sync calibration quality gate failed or manifest/file disagree")
    meta["calibration_path_resolved"] = str(cal_path.resolve())
    meta["audio_to_force_offset_ms"] = offset
    meta["capture_mode"] = manifest_mode
    meta["runtime_capture_mode"] = runtime_mode
    meta["identity_match"] = identity_match
    return meta


def apply_audio_sync(times_s, sync_meta: dict) -> np.ndarray:  # noqa: ANN001
    """Apply the sole allowed equation: audio_aligned = audio_host + offset_ms/1000."""
    if sync_meta.get("applied") is not False:
        raise ValueError("sync offset already applied")
    return np.asarray(times_s, dtype=np.float64) + float(sync_meta["audio_to_force_offset_ms"]) / 1000.0


def fit_device_clock_checked(host_t_s, device_t_ms, *, stream: str) -> dict:  # noqa: ANN001
    """Fit and enforce the acquisition clock contract used for force/camera timelines."""
    host = np.asarray(host_t_s, dtype=np.float64)
    device = np.asarray(device_t_ms, dtype=np.float64)
    valid = np.isfinite(host) & np.isfinite(device)
    host, device = host[valid], device[valid]
    if host.size < 20 or np.ptp(device) < 2_000.0 or np.any(np.diff(device) <= 0):
        raise ValueError(
            f"{stream} device clock fit rejected: need >=20 increasing pairs spanning >=2s"
        )
    slope, intercept = np.polyfit(device, host, 1)
    predicted = intercept + slope * device
    rmse_ms = float(np.sqrt(np.mean((host - predicted) ** 2)) * 1_000.0)
    drift_ppm = float((slope / 0.001 - 1.0) * 1_000_000.0)
    accepted = abs(drift_ppm) <= 5_000.0 and rmse_ms <= 5.0
    result = {
        "accepted": accepted,
        "n_samples": int(host.size),
        "span_s": float(np.ptp(device) / 1_000.0),
        "host_s_per_device_ms": float(slope),
        "host_intercept_s": float(intercept),
        "drift_ppm": drift_ppm,
        "rmse_ms": rmse_ms,
    }
    if not accepted:
        raise ValueError(
            f"{stream} device clock fit rejected: drift={drift_ppm:.1f}ppm rmse={rmse_ms:.3f}ms"
        )
    return result


def read_camera_timeline(session_dir: str | Path, manifest: dict) -> CameraTimeline:
    """Build exposure time without pretending host-only cameras have a device clock."""
    path = Path(session_dir) / "rgb_times.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError("rgb_times.csv has no rows")
    fields = set(rows[0])
    required = {"frame_index", "host_arrival_t_s", "device_t_ms", "time_source"}
    if not required.issubset(fields):
        raise ValueError(f"rgb_times.csv needs {sorted(required)}")
    try:
        frame_index = np.asarray([int(row["frame_index"]) for row in rows], dtype=np.int64)
        host = np.asarray([float(row["host_arrival_t_s"]) for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("rgb_times.csv has invalid camera timestamps") from exc
    _strictly_increasing(frame_index, "camera frame index")
    _strictly_increasing(host, "camera host arrival")

    raw_device = [str(row.get("device_t_ms") or "").strip() for row in rows]
    if any(raw_device) and not all(raw_device):
        raise ValueError("rgb_times.csv cannot mix present and missing camera device timestamps")
    try:
        device = (np.asarray([float(value) for value in raw_device], dtype=np.float64)
                  if raw_device and all(raw_device) else None)
    except ValueError as exc:
        raise ValueError("rgb_times.csv has invalid camera device timestamps") from exc

    time_sources = {str(row.get("time_source") or "").strip() for row in rows}
    if len(time_sources) != 1 or "" in time_sources:
        raise ValueError("rgb_times.csv must use one explicit camera time_source")
    time_source = next(iter(time_sources))
    camera_meta = dict(manifest.get("camera") or {})
    backend = str(camera_meta.get("backend") or "").lower()
    if backend not in ("orbbec", "avfoundation", "opencv"):
        raise ValueError(f"unsupported or missing camera backend: {backend!r}")
    if backend == "orbbec" and device is None:
        raise ValueError("orbbec camera timeline requires real device_t_ms values")
    if device is None and (backend not in ("avfoundation", "opencv")
                           or time_source != "opencv_host_arrival"):
        raise ValueError(
            "host-arrival camera timing is allowed only for avfoundation/opencv "
            "with time_source=opencv_host_arrival"
        )

    fit = (fit_device_clock_checked(host, device, stream="camera")
           if device is not None else None)

    timing = dict(camera_meta.get("timing") or {})
    if timing.get("applied") is not False:
        raise ValueError("camera latency correction missing applied=false or already applied")
    if timing.get("latency_definition") != "host_arrival_minus_exposure":
        raise ValueError("camera latency definition must be host_arrival_minus_exposure")
    cal_path = _resolve_manifest_calibration_path(
        session_dir, timing.get("calibration_path"), stream="camera latency",
    )
    actual_hash = _sha256(cal_path)
    if actual_hash != timing.get("calibration_sha256"):
        raise ValueError("camera latency calibration hash mismatch")
    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    if (payload.get("schema_version") != "wam_camera_latency_v1"
            or payload.get("latency_definition") != "host_arrival_minus_exposure"):
        raise ValueError("camera latency calibration schema/definition invalid")
    timing_mode = _validate_camera_capture_mode(timing.get("capture_mode"))
    payload_mode = _validate_camera_capture_mode(payload.get("capture_mode"))
    if timing_mode != payload_mode:
        raise ValueError("camera capture_mode in manifest does not match calibration file")
    camera_id = camera_meta.get("camera_id")
    if (timing_mode["backend"] != backend
            or timing_mode["camera_id"] != camera_id
            or timing_mode["time_source"] != time_source):
        raise ValueError(
            "camera capture_mode does not match session identity/timestamp semantics: "
            f"calibration=({timing_mode['backend']!r}, {timing_mode['camera_id']!r}, "
            f"{timing_mode['time_source']!r}), capture=({backend!r}, {camera_id!r}, "
            f"{time_source!r})"
        )
    mode_validation = timing.get("mode_validation")
    expected_validation = {
        "width": "matched_frame_shape",
        "height": "matched_frame_shape",
        "fps": "matched_negotiated_mode",
        "time_source": "matched_frame_metadata",
        "camera_id": "matched_device_serial" if backend == "orbbec" else None,
        "pipeline": "matched_negotiated_pipeline",
        "depth": "matched_pipeline_shape" if backend == "orbbec" else None,
    }
    if not isinstance(mode_validation, Mapping) or any(
        expected is not None and mode_validation.get(field) != expected
        for field, expected in expected_validation.items()
    ):
        raise ValueError("camera capture_mode was not verified against the live negotiated mode")
    if (backend == "orbbec" and mode_validation.get("device_timestamp") != "present"):
        raise ValueError("orbbec camera capture_mode lacks verified device timestamps")
    expected_negotiated = {
        field: timing_mode[field]
        for field in (
            "width", "height", "fps", "color_format", "depth_mode", "align_mode",
            "frame_aggregate_mode",
        )
    }
    if mode_validation.get("negotiated_mode") != expected_negotiated:
        raise ValueError("camera negotiated pipeline metadata does not match capture_mode")
    latency_ms = float(timing.get("latency_ms"))
    if (not np.isfinite(latency_ms)
            or not np.isclose(latency_ms, float(payload.get("latency_ms")), atol=1e-9)):
        raise ValueError("camera latency manifest value does not match calibration")
    try:
        manifest_std = float(timing["latency_std_ms"])
        payload_std = float(payload["latency_std_ms"])
        manifest_samples = int(timing["n_samples"])
        payload_samples = int(payload["n_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("camera latency quality metadata is incomplete") from exc
    if (not 0 < latency_ms <= 500.0 or not np.isfinite(manifest_std)
            or not 0 <= manifest_std <= 5.0
            or not np.isclose(manifest_std, payload_std, atol=1e-9)
            or manifest_samples < 20 or manifest_samples != payload_samples
            # hardware timing: LED flash or host-synced device timestamp (device_timer_host_sync)
            or timing.get("measurement_method") not in _CAMERA_TIMING_METHODS
            or payload.get("measurement_method") not in _CAMERA_TIMING_METHODS):
        raise ValueError("camera latency quality gate failed or manifest/file disagree")
    mapped_host = (fit["host_intercept_s"] + fit["host_s_per_device_ms"] * device
                   if fit is not None else host)
    exposure = mapped_host - latency_ms / 1_000.0
    _strictly_increasing(exposure, "camera exposure")
    return CameraTimeline(
        frame_index=frame_index, t_exposure_s=exposure, host_arrival_t_s=host,
        device_t_ms=device, clock_fit=fit, latency_ms=latency_ms,
    )


def validate_manifest_for_conversion(manifest: dict) -> None:
    """Reject anything the combined supervisor did not certify as training-ready."""
    if manifest.get("schema_version") != "wam_session_v2":
        raise ValueError("canonical conversion requires schema_version=wam_session_v2")
    if manifest.get("promotion_contract_version") != PROMOTION_CONTRACT_VERSION:
        raise ValueError(
            f"canonical conversion requires promotion_contract_version="
            f"{PROMOTION_CONTRACT_VERSION}"
        )
    if manifest.get("status") != "complete" or manifest.get("training_eligible") is not True:
        raise ValueError("session is not complete and training_eligible")
    # Immutable production-writer gate (defense in depth with the finalizer).
    # ``training_eligible`` alone cannot distinguish a diagnostic capture from a
    # production one, so the manifest must positively declare an allowlisted
    # production ``capture_kind``; a diagnostic kind can never be converted.
    if manifest.get("capture_kind") not in PRODUCTION_CAPTURE_KINDS:
        raise ValueError(
            "canonical conversion requires an allowlisted production capture_kind; "
            "diagnostic captures can never be promoted to training data"
        )
    if manifest.get("rehearsal_overrides"):
        raise ValueError("rehearsal sessions cannot be converted as training data")
    required = {"force", "audio", "cam", "robot"}
    missing = sorted(required.difference(manifest.get("streams") or []))
    if missing:
        raise ValueError(f"manifest missing required streams: {missing}")
    arm = manifest.get("arm") or {}
    canonical_joints = [f"openarm_left_joint{index}" for index in range(1, 8)]
    if (
        arm.get("side") != "left"
        or arm.get("joint_names") != canonical_joints
        or arm.get("state_unit") != "rad"
        or arm.get("action_type") != "exact_mitparam_consumed_at_control_tick"
    ):
        raise ValueError(
            "canonical conversion requires left OpenArm exact MIT action metadata; "
            "legacy position-target/follower-observation actions are forbidden"
        )
    robot = manifest.get("robot")
    if (
        not isinstance(robot, Mapping)
        or robot.get("schema") != EXACT_ROBOT_SUMMARY_SCHEMA_V4
        or robot.get("side") != "left"
        or robot.get("leader_can_interface") != "can1"
        or robot.get("follower_can_interface") != "can3"
        or robot.get("action_semantics")
        != "exact_mitparam_consumed_at_control_tick"
        or robot.get("action_interpolation") != "forbidden"
    ):
        raise ValueError("manifest is missing the canonical exact-control v4 robot summary")
    require_exact_robot_training_eligibility(robot)
    if (manifest.get("force") or {}).get("calibrated") is not True:
        raise ValueError("manifest force calibration is not calibrated=true")
    _validate_finalizer_artifact_evidence(manifest.get("artifact_validation"))
    _validated_artifact_hashes(manifest)
    camera = manifest.get("camera") or {}
    if (camera.get("backend") != "orbbec"
            or not isinstance(camera.get("camera_id"), str)
            or not camera["camera_id"].strip()
            or camera.get("camera_id_source") != "verified_orbbec_serial"
            or camera.get("has_depth") is not True):
        raise ValueError(
            "canonical arm conversion requires a serial-verifiable Orbbec RGB-D capture"
        )


def _validate_finalizer_artifact_evidence(value: object) -> None:
    if not isinstance(value, Mapping) or any(
        not isinstance(value.get(name), Mapping)
        for name in ("audio", "video", "depth", "observations")
    ):
        raise ValueError("manifest finalizer artifact validation evidence is incomplete")
    audio = value["audio"]
    video = value["video"]
    depth = value["depth"]
    observations = value["observations"]

    def positive_int(metadata: Mapping, field: str) -> int:
        result = metadata.get(field)
        if type(result) is not int or result <= 0:
            raise ValueError(
                f"manifest finalizer artifact validation {field} must be a positive integer"
            )
        return result

    positive_int(audio, "sample_rate")
    positive_int(audio, "n_frames")
    video_frames = positive_int(video, "n_frames")
    depth_frames = positive_int(depth, "n_frames")
    positive_int(observations, "n_rows")
    if video_frames != depth_frames:
        raise ValueError("manifest finalizer artifact validation RGB/depth counts disagree")
    video_shape = video.get("shape")
    depth_shape = depth.get("shape")
    if (not isinstance(video_shape, (list, tuple)) or len(video_shape) != 3
            or any(type(value) is not int or value <= 0 for value in video_shape)
            or video_shape[2] != 3
            or not isinstance(depth_shape, (list, tuple)) or len(depth_shape) != 2
            or any(type(value) is not int or value <= 0 for value in depth_shape)):
        raise ValueError("manifest finalizer artifact validation shapes are invalid")
    encoded_fps = video.get("encoded_fps")
    if (not isinstance(encoded_fps, (int, float)) or isinstance(encoded_fps, bool)
            or not np.isfinite(encoded_fps) or encoded_fps <= 0):
        raise ValueError("manifest finalizer artifact validation video FPS is invalid")


def _validated_artifact_hashes(manifest: Mapping) -> dict[str, str]:
    """Validate the finalizer's raw-artifact inventory and return expected hashes."""
    inventory = manifest.get("artifact_hashes")
    if (not isinstance(inventory, Mapping)
            or inventory.get("schema_version") != ARTIFACT_HASH_SCHEMA_VERSION
            or inventory.get("algorithm") != "sha256"
            or not isinstance(inventory.get("files"), Mapping)):
        raise ValueError("manifest artifact hash inventory is missing or invalid")
    expected: dict[str, str] = {}
    for relative, metadata in inventory["files"].items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("manifest artifact hash path must be a non-empty relative path")
        path = Path(relative)
        if (path.is_absolute() or relative != path.as_posix()
                or any(part in ("", ".", "..") for part in path.parts)):
            raise ValueError(f"manifest artifact hash path is unsafe: {relative!r}")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"manifest artifact hash metadata is invalid for {relative}")
        digest = metadata.get("sha256")
        size = metadata.get("size_bytes")
        if (not isinstance(digest, str) or len(digest) != 64
                or digest != digest.lower()
                or any(char not in "0123456789abcdef" for char in digest)
                or type(size) is not int or size < 0):
            raise ValueError(f"manifest artifact hash metadata is invalid for {relative}")
        expected[relative] = digest
    missing = sorted(REQUIRED_RAW_ARTIFACTS.difference(expected))
    if missing:
        raise ValueError(f"manifest artifact hash inventory is missing raw files: {missing}")
    if "robot.csv" in expected:
        raise ValueError(
            "legacy robot.csv is forbidden in canonical exact-action promotion inventory"
        )
    # depth frames use depth_codec.DEPTH_SUFFIX (".npy.zlib"); legacy raw ".npy"
    # sessions stay convertible.
    if not any(
        name.startswith("depth/depth_") and name.endswith((".npy.zlib", ".npy"))
        for name in expected
    ):
        raise ValueError("manifest artifact hash inventory has no depth frames")
    return expected


def split_spans_at_boundaries(
    spans: list[tuple[float, float]], boundaries,
) -> list[tuple[float, float]]:  # noqa: ANN001
    """Clip time spans so no reward segment can cross a target/episode boundary."""
    cuts = sorted({float(value) for value in boundaries if np.isfinite(value)})
    out: list[tuple[float, float]] = []
    for start, end in spans:
        start, end = float(start), float(end)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError(f"invalid segment span {(start, end)}")
        points = [start] + [cut for cut in cuts if start < cut < end] + [end]
        out.extend((points[i], points[i + 1]) for i in range(len(points) - 1))
    return out


def _cover_reward_interval(spans: list, start: float, stop: float) -> list:  # noqa: ANN001
    """Normalize rounding noise, but never assign reward across an unlabeled time gap."""
    if not spans or not np.isfinite([start, stop]).all() or stop <= start:
        raise ValueError("reward interval needs non-empty spans and finite start < stop")
    ordered = sorted(spans, key=lambda item: float(item.t_start))
    tolerance_s = 1e-4  # Segment timestamps are persisted at four decimal places.
    endpoints = [(float(item.t_start), float(item.t_end)) for item in ordered]
    if any(not np.isfinite([left, right]).all() or right <= left
           for left, right in endpoints):
        raise ValueError("reward spans must have finite positive duration")
    if abs(endpoints[0][0] - float(start)) > tolerance_s:
        raise ValueError(
            f"uncovered reward gap at interval start: {start} -> {endpoints[0][0]}"
        )
    if abs(endpoints[-1][1] - float(stop)) > tolerance_s:
        raise ValueError(
            f"uncovered reward gap at interval end: {endpoints[-1][1]} -> {stop}"
        )
    internal_boundaries: list[float] = []
    for (_, previous_end), (next_start, _) in zip(endpoints[:-1], endpoints[1:]):
        delta = next_start - previous_end
        if delta > tolerance_s:
            raise ValueError(
                f"uncovered reward gap {previous_end:.6f} -> {next_start:.6f}"
            )
        if delta < -tolerance_s:
            raise ValueError(
                f"overlapping reward spans {previous_end:.6f} and {next_start:.6f}"
            )
        internal_boundaries.append((previous_end + next_start) / 2.0)
    boundaries = [float(start), *internal_boundaries, float(stop)]
    covered = []
    for item, left, right in zip(ordered, boundaries[:-1], boundaries[1:]):
        updates = {"t_start": left, "t_end": right}
        if hasattr(item, "audio_t_start"):
            updates.update(audio_t_start=left, audio_t_end=right)
        covered.append(replace(item, **updates))
    return covered


def write_lerobot_033(dataset, records) -> None:  # noqa: ANN001
    """Write canonical frames using LeRobot 0.3.3's separate ``task=`` contract."""
    open_episode = False
    for record in records:
        dataset.add_frame(record.frame, task=record.task)
        open_episode = True
        done = bool(np.asarray(record.frame["next.done"]).reshape(-1)[0])
        if done:
            dataset.save_episode()
            open_episode = False
    if open_episode:
        raise ValueError("final canonical frame must have next.done=true before save_episode")


def _retarget_dataset_root(dataset, staging_root: Path, final_root: Path) -> None:  # noqa: ANN001
    """Retarget the supported LeRobot root holders before atomic publication."""
    holders = [dataset, getattr(dataset, "meta", None)]
    for holder in holders:
        if holder is None or not hasattr(holder, "root"):
            continue
        current = getattr(holder, "root")
        if current is None or Path(current) != staging_root:
            continue
        try:
            setattr(holder, "root", final_root)
        except Exception as exc:  # noqa: BLE001 - fail before commit, never return a stale dataset
            raise RuntimeError(
                f"cannot retarget returned dataset root from {staging_root} to {final_root}"
            ) from exc
        if Path(getattr(holder, "root")) != final_root:
            raise RuntimeError("returned dataset retained its staging root")
    kwargs = getattr(dataset, "kwargs", None)
    if (isinstance(kwargs, dict) and kwargs.get("root") is not None
            and Path(kwargs["root"]) == staging_root):
        kwargs["root"] = final_root


def prepare_causal_records(
    camera_times_s,
    robot: RobotStream,
    force: ForceStream,
    targets: Optional[TargetStream],
    reward_spans: list[RewardSpan],
    *,
    robot_max_age_s: float,
    force_max_age_s: float,
    target_max_age_s: float,
) -> list[CausalFrame]:  # noqa: ANN001
    """Join causal observations first, then place done at each target boundary."""
    camera_times = np.asarray(camera_times_s, dtype=np.float64)
    _strictly_increasing(camera_times, "camera")
    if targets is not None:
        boundaries = [float(targets.t_s[i]) for i in range(1, len(targets.t_s))
                      if targets.names[i] != targets.names[i - 1]]
        for span in reward_spans:
            if any(span.t_start + 1e-6 < boundary < span.t_end - 1e-6
                   for boundary in boundaries):
                raise ValueError("reward span crosses a target/episode boundary")
    records: list[CausalFrame] = []
    for query_t in camera_times:
        span = next(
            (item for item in reward_spans if item.t_start <= query_t < item.t_end), None,
        )
        if span is None:
            raise ValueError(f"no reward span covers camera t={query_t:.6f}")
        records.append(build_causal_frame(
            float(query_t), robot, force, targets, reward=float(span.reward), done=False,
            robot_max_age_s=robot_max_age_s, force_max_age_s=force_max_age_s,
            target_max_age_s=target_max_age_s,
        ))
    for i, record in enumerate(records):
        done = i == len(records) - 1 or records[i + 1].target_name != record.target_name
        record.frame["next.done"] = np.asarray([done], bool)
    return records


def prepare_exact_causal_records(
    camera: CameraTimeline,
    robot: ExactRobotFrameStream,
    force: ForceStream,
    targets: Optional[TargetStream],
    reward_spans: list[RewardSpan],
    *,
    force_max_age_s: float,
    target_max_age_s: float,
) -> list[CausalFrame]:
    """Build camera-rate records from preselected real MIT control ticks."""

    camera_times = np.asarray(camera.t_exposure_s, dtype=np.float64)
    _strictly_increasing(camera_times, "camera")
    if (
        robot.frame_index.shape != camera.frame_index.shape
        or not np.array_equal(robot.frame_index, camera.frame_index)
        or robot.state.shape != (len(camera_times), len(robot.joint_names))
        or robot.action.shape != (len(camera_times), len(robot.action_names))
        or robot.action_tick_seq.shape != (len(camera_times),)
        or robot.action_tick_monotonic_ns.shape != (len(camera_times),)
    ):
        raise ValueError("exact robot frame stream does not close on camera timeline")
    if targets is not None:
        boundaries = [
            float(targets.t_s[index])
            for index in range(1, len(targets.t_s))
            if targets.names[index] != targets.names[index - 1]
        ]
        for span in reward_spans:
            if any(
                span.t_start + 1e-6 < boundary < span.t_end - 1e-6
                for boundary in boundaries
            ):
                raise ValueError("reward span crosses a target/episode boundary")

    records = []
    for index, query_t in enumerate(camera_times):
        span = next(
            (item for item in reward_spans if item.t_start <= query_t < item.t_end),
            None,
        )
        if span is None:
            raise ValueError(f"no reward span covers camera t={query_t:.6f}")
        force_index, force_age = asof_within(
            force.t_host_s,
            float(query_t),
            max_age_s=force_max_age_s,
            stream="force",
        )
        force_status = force.status[force_index]
        # P0-4: over-threshold is a VALID measurement, never a session-rejecting fault;
        # contact_force_valid marks a genuine sensor reading, not "safe" force.
        force_valid = force_status in ("OK", "WARN", "SCREENED_LIMIT", "HARD_LIMIT")
        frame = {
            "observation.state": robot.state[index],
            "observation.robot_state_age_s": np.asarray([0.0], np.float32),
            "observation.images.top": None,
            "observation.contact_force_n": np.asarray(
                [force.force_n[force_index]], np.float32,
            ),
            "observation.contact_force_valid": np.asarray([force_valid], bool),
            "observation.contact_force_age_s": np.asarray([force_age], np.float32),
            "action": robot.action[index],
            "next.reward": np.asarray([float(span.reward)], np.float32),
            "next.done": np.asarray([False], bool),
        }
        task = DEFAULT_TASK
        target_name = None
        if targets is not None:
            target_index, target_age = asof_within(
                targets.t_s,
                float(query_t),
                max_age_s=target_max_age_s,
                stream="target",
            )
            target_name = targets.names[target_index]
            # VLA "L": anatomical grounding of the code (phrasing augmentation applied later).
            task = site_instruction(target_name)
            frame["observation.target_point_2d"] = targets.uv[target_index]
            frame["observation.target_age_s"] = np.asarray(
                [target_age], np.float32,
            )
            if targets.xyz[target_index] is not None:
                frame["observation.target_point_3d"] = np.asarray(
                    targets.xyz[target_index], np.float32,
                )
        records.append(CausalFrame(frame=frame, task=task, target_name=target_name))
    for index, record in enumerate(records):
        done = (
            index == len(records) - 1
            or records[index + 1].target_name != record.target_name
        )
        record.frame["next.done"] = np.asarray([done], bool)
    return records


def write_exact_action_selection(
    path: str | Path, robot: ExactRobotFrameStream,
) -> dict:
    """Commit the auditable camera-frame to exact-control-tick selection index."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"exact action selection output already exists: {destination}")
    raw = b"".join(
        (
            json.dumps(
                record, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in robot.selection_records
    )
    if not raw or raw.count(b"\n") != len(robot.selection_records):
        raise ValueError("exact action selection index is empty or malformed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"stale exact action selection partial exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    digest = _sha256(destination)
    if digest != hashlib.sha256(raw).hexdigest():
        destination.unlink(missing_ok=True)
        raise ValueError("exact action selection changed while being committed")
    return {
        "schema_version": EXACT_ACTION_SELECTION_SCHEMA,
        "path": destination.name,
        "sha256": digest,
        "size_bytes": len(raw),
        "frame_count": len(robot.selection_records),
        "selection_rule": EXACT_ACTION_SELECTION_RULE,
        "action_interpolation": "forbidden",
        "action_names": list(robot.action_names),
        "world_action_exact_control_sidecars": {
            name: {"path": name, "sha256": robot.sidecar_hashes[name]}
            for name in EXACT_ROBOT_FILES
        },
    }


def snapshot_session_sources(
    session_dir: str | Path,
    dataset_root: str | Path,
    *,
    generated_segments: str | Path | None = None,
    extra_sources: Optional[dict[str, VerifiedSource]] = None,
    expected_source_hashes: Optional[Mapping[str, str]] = None,
    expected_contract_hashes: Optional[Mapping[str, str]] = None,
    verify_training_contract: bool = False,
    verified_training_contract: Optional[Mapping[str, object]] = None,
    expected_manifest_sha256: str | None = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Copy an immutable source snapshot and optionally reverify promotion v2.

    In promotion-v2 mode every outer metadata file and every raw inventory file
    is copied first.  The shared verifier runs against that exact frozen set
    before converter-only provenance or calibration aliases are added.
    """
    session = Path(session_dir).resolve()
    if verify_training_contract:
        if not isinstance(verified_training_contract, Mapping):
            raise ValueError(
                "promotion-v2 snapshot requires a prior trusted training contract"
            )
        verified_source = Path(str(
            verified_training_contract.get("verified_snapshot_path", "")
        )).resolve()
        if verified_source != session:
            raise ValueError(
                "promotion-v2 snapshot source must be the verified trusted snapshot"
            )
        session_id_value = verified_training_contract.get("session_id")
        if (
            not isinstance(session_id_value, str)
            or not session_id_value
            or Path(session_id_value).name != session_id_value
        ):
            raise ValueError("promotion-v2 verified session identity is invalid")
        session_id = session_id_value
    else:
        if verified_training_contract is not None:
            raise ValueError(
                "legacy snapshot cannot accept a trusted training contract"
            )
        session_id = session.name
    destination = (Path(dataset_root) / "source_sessions" / session_id).resolve()
    try:
        destination.relative_to(session)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"snapshot destination and source session overlap: {destination} inside {session}"
        )
    try:
        session.relative_to(destination)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"snapshot destination and source session overlap: {session} inside {destination}"
        )
    if (expected_contract_hashes is None) != (not verify_training_contract):
        raise ValueError(
            "promotion-v2 snapshot requires expected_contract_hashes and "
            "verify_training_contract=true together"
        )
    if expected_contract_hashes is not None and (
        expected_source_hashes is not None or expected_manifest_sha256 is not None
    ):
        raise ValueError("promotion-v2 snapshot cannot mix legacy expected hash inputs")
    if expected_contract_hashes is not None:
        collisions = sorted(
            set(expected_contract_hashes).intersection(CONVERTER_DERIVED_PATHS)
        )
        if collisions:
            raise ValueError(
                "promotion-v2 raw inventory has a converter-derived path collision: "
                f"{collisions}"
            )

    destination.mkdir(parents=True, exist_ok=True)
    base_sources: list[tuple[Path, Path, Optional[str]]] = []
    derived_sources: list[tuple[Path, Path, Optional[str]]] = []
    if expected_contract_hashes is not None:
        for name, digest in sorted(expected_contract_hashes.items()):
            relative = Path(str(name))
            if (
                relative.is_absolute()
                or relative.as_posix() != str(name)
                or any(part in ("", ".", "..") for part in relative.parts)
            ):
                raise ValueError(f"promotion-v2 snapshot path is unsafe: {name!r}")
            expected_hash = _validate_sha256(
                digest, label=f"promotion-v2 snapshot hash for {name}",
            )
            source = session / relative
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(
                    f"source session missing safe contract artifact: {source}"
                )
            base_sources.append((source, relative, expected_hash))
        for source_name, copy_name in (
            ("manifest.json", "manifest.original.json"),
            ("FINALIZATION.json", "FINALIZATION.original.json"),
        ):
            derived_sources.append((
                session / source_name,
                Path(copy_name),
                expected_contract_hashes.get(source_name),
            ))
    else:
        required = [
            "manifest.json", "audio.wav", "rgb.mp4", "force.csv",
            "rgb_times.csv", "obs.csv", "targets.jsonl", *EXACT_ROBOT_FILES,
        ]
        expected_raw = (
            {
                str(name): str(digest).lower()
                for name, digest in expected_source_hashes.items()
            }
            if expected_source_hashes is not None else None
        )
        raw_relatives: set[str] = set()
        for name in required:
            source = session / name
            if not source.is_file():
                raise FileNotFoundError(
                    f"source session missing required artifact: {source}"
                )
            expected_hash = expected_manifest_sha256 if name == "manifest.json" else None
            if name != "manifest.json":
                raw_relatives.add(name)
                if expected_raw is not None:
                    expected_hash = expected_raw.get(name)
            base_sources.append((source, Path(name), expected_hash))
            if name == "manifest.json":
                base_sources.append((
                    source, Path("manifest.original.json"), expected_hash,
                ))
        depth_dir = session / "depth"
        if depth_dir.is_dir():
            for source in sorted(path for path in depth_dir.rglob("*") if path.is_file()):
                relative = Path("depth") / source.relative_to(depth_dir)
                relative_name = relative.as_posix()
                raw_relatives.add(relative_name)
                base_sources.append((
                    source, relative,
                    expected_raw.get(relative_name) if expected_raw is not None else None,
                ))
        if expected_raw is not None:
            missing = sorted(raw_relatives.difference(expected_raw))
            unexpected = sorted(set(expected_raw).difference(raw_relatives))
            if missing or unexpected:
                raise ValueError(
                    "promoted artifact hash inventory does not match session raw files: "
                    f"missing hashes={missing}, unexpected hashes={unexpected}"
                )
    if generated_segments is not None:
        segment_source = Path(generated_segments)
        if not segment_source.is_file():
            raise FileNotFoundError(f"generated segments missing: {segment_source}")
        derived_sources.append((segment_source, Path("segments.jsonl"), None))
    for relative, source_spec in sorted((extra_sources or {}).items()):
        if not isinstance(source_spec, VerifiedSource):
            raise ValueError("extra provenance sources must be verified")
        source = Path(source_spec.path)
        expected_hash = source_spec.sha256.lower()
        if not source.is_file():
            raise FileNotFoundError(f"provenance source missing: {source}")
        derived_sources.append((source, Path(relative), expected_hash))

    files: dict[str, dict] = {}

    def copy_sources(sources: list[tuple[Path, Path, Optional[str]]]) -> None:
        for source, relative, expected_hash in sources:
            if (
                relative.is_absolute()
                or relative.as_posix() != str(relative)
                or any(part in ("", ".", "..") for part in relative.parts)
            ):
                raise ValueError(f"snapshot destination path is unsafe: {relative}")
            target = destination / relative
            source_hash = _sha256(source)
            if expected_hash is not None and source_hash != expected_hash:
                raise ValueError(
                    f"verified provenance source hash mismatch for {source}: "
                    f"expected {expected_hash}, got {source_hash}"
                )
            if target.exists():
                if not target.is_file() or _sha256(target) != source_hash:
                    raise FileExistsError(
                        f"snapshot target has different content: {target}"
                    )
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if _sha256(target) != source_hash:
                    target.unlink(missing_ok=True)
                    raise ValueError(
                        f"snapshot copy changed while reading source: {source}"
                    )
            files[relative.as_posix()] = {
                "sha256": source_hash,
                "size_bytes": target.stat().st_size,
            }

    copy_sources(base_sources)
    frozen_training_contract = None
    if verify_training_contract:
        frozen_training_contract = dict(verified_training_contract)
    copy_sources(derived_sources)
    provenance = {
        "schema_version": (
            "wam_conversion_provenance_v2"
            if verify_training_contract else "wam_conversion_provenance_v1"
        ),
        "source_session_id": session_id,
        "files": files,
    }
    if frozen_training_contract is not None:
        provenance["frozen_training_contract"] = frozen_training_contract
    provenance.update(metadata or {})
    provenance_path = destination / "provenance.json"
    tmp = provenance_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, provenance_path)
    return provenance


def _persist_replayable_snapshot_manifest(snapshot_dir: str | Path, manifest: dict) -> dict:
    """Persist the relative-path replay manifest while retaining the byte-exact original."""
    snapshot = Path(snapshot_dir).resolve()
    original_path = snapshot / "manifest.original.json"
    if not original_path.is_file():
        raise FileNotFoundError(f"snapshot is missing original manifest: {original_path}")
    _verify_finalization_seal(
        snapshot, manifest_name="manifest.original.json",
        seal_name="FINALIZATION.original.json",
    )
    calibration_paths = (
        (manifest.get("force") or {}).get("calibration_path"),
        (manifest.get("sync") or {}).get("calibration_path"),
        (((manifest.get("camera") or {}).get("timing") or {}).get("calibration_path")),
    )
    for value in calibration_paths:
        relative = Path(str(value or ""))
        if not str(value or "") or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"snapshot calibration path must be relative and contained: {value!r}")
        resolved = (snapshot / relative).resolve()
        try:
            resolved.relative_to(snapshot)
        except ValueError as exc:
            raise ValueError(f"snapshot calibration path escapes snapshot: {value!r}") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"snapshot calibration is missing: {resolved}")

    manifest_path = snapshot / "manifest.json"
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    finalization_path = snapshot / "FINALIZATION.json"
    authenticity_path = snapshot / TRAINING_AUTHENTICITY_NAME
    if not authenticity_path.is_file():
        raise FileNotFoundError(
            f"snapshot is missing training authenticity: {authenticity_path}"
        )
    tmp_finalization = finalization_path.with_suffix(".json.tmp")
    tmp_finalization.write_text(json.dumps({
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "promotion_contract_version": PROMOTION_CONTRACT_VERSION,
        "manifest_sha256": _sha256(manifest_path),
        "training_authenticity_sha256": _sha256(authenticity_path),
    }, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_finalization, finalization_path)

    provenance_path = snapshot / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    files = provenance.setdefault("files", {})
    for relative, path in (
        ("manifest.json", manifest_path),
        ("manifest.original.json", original_path),
        ("FINALIZATION.json", finalization_path),
        ("FINALIZATION.original.json", snapshot / "FINALIZATION.original.json"),
    ):
        files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    tmp_provenance = provenance_path.with_suffix(".json.tmp")
    tmp_provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_provenance, provenance_path)
    return provenance


def _update_snapshot_provenance(snapshot_dir: str | Path, *, metadata: dict) -> dict:
    """Hash newly derived snapshot artifacts and atomically finalize provenance."""
    snapshot = Path(snapshot_dir)
    provenance_path = snapshot / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    segments = snapshot / "segments.jsonl"
    if not segments.is_file():
        raise FileNotFoundError(f"generated segments missing from snapshot: {segments}")
    provenance.setdefault("files", {})["segments.jsonl"] = {
        "sha256": _sha256(segments),
        "size_bytes": segments.stat().st_size,
    }
    selection = snapshot / "robot_frame_action_selection.jsonl"
    if not selection.is_file():
        raise FileNotFoundError(
            f"exact robot action selection missing from snapshot: {selection}"
        )
    provenance["files"][selection.name] = {
        "sha256": _sha256(selection),
        "size_bytes": selection.stat().st_size,
    }
    provenance.update(metadata)
    tmp = provenance_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, provenance_path)
    return provenance


def _verify_snapshot_integrity(snapshot_dir: str | Path) -> None:
    """Recheck every content hash immediately before the first dataset write."""
    snapshot = Path(snapshot_dir)
    provenance = json.loads((snapshot / "provenance.json").read_text(encoding="utf-8"))
    for relative, metadata in (provenance.get("files") or {}).items():
        path = snapshot / relative
        if not path.is_file() or _sha256(path) != metadata.get("sha256"):
            raise ValueError(f"immutable snapshot integrity failure: {relative}")


def _build_target_split_segments(
    sdir: str | Path,
    force: ForceStream,
    targets: TargetStream,
    sync_meta: dict,
    *,
    quality_fn=None,
):  # noqa: ANN001
    """Generate reward labels independently inside each target episode.

    The only currently wired call would pass ``q_fn=None`` into
    ``build_segments``, which selects the documented spectral stand-in.  A
    proxy is useful for commissioning, but it is not a truthful training
    label and must never enter the canonical VLA/WAM export.
    """
    if not callable(quality_fn):
        raise ValueError(
            "canonical reward conversion requires an activated quality model; "
            "the q_fn=None spectral proxy is forbidden"
        )
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ausc"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ausc", "audio"))
    from session_io import Session
    from session_to_lerobot import build_segments

    base = Session(str(sdir))
    offset_s = float(sync_meta["audio_to_force_offset_ms"]) / 1_000.0
    end = base.t0 + base.duration
    boundaries = [float(targets.t_s[i]) for i in range(1, len(targets.t_s))
                  if targets.names[i] != targets.names[i - 1] and base.t0 < targets.t_s[i] < end]
    interval_edges = [base.t0] + boundaries + [end]

    class Window:
        def __init__(self, start: float, stop: float) -> None:
            self.t0 = start
            self.duration = stop - start
            self.audio_sr = base.audio_sr

        def audio_window(self, t: float, seconds: float):
            # raw audio host time + offset = force-aligned time, so invert for lookup
            return base.audio_window(t - offset_s, seconds=seconds)

    all_segments = []
    session_id = Path(sdir).name
    ft = force.t_host_s.tolist(); fn = force.force_n.tolist()
    for start, stop in zip(interval_edges[:-1], interval_edges[1:]):
        interval_segments = build_segments(
            Window(start, stop), ft, fn, session_id, q_fn=quality_fn,
        )
        all_segments.extend(_cover_reward_interval(interval_segments, start, stop))
    return [replace(seg, segment_id=f"{session_id}#seg{i:04d}")
            for i, seg in enumerate(all_segments)]


def _manifest_calibration_sources(
    session_dir: str | Path, manifest: dict,
) -> dict[str, VerifiedSource]:
    """Resolve only the files needed to freeze inputs; content is consumed after snapshot."""
    entries = (
        ("calibrations/sync.json", manifest.get("sync") or {}, "sync"),
        ("calibrations/force.json", manifest.get("force") or {}, "force"),
        ("calibrations/camera_latency.json",
         ((manifest.get("camera") or {}).get("timing") or {}), "camera latency"),
    )
    sources: dict[str, VerifiedSource] = {}
    for relative, metadata, stream in entries:
        expected_hash = str(metadata.get("calibration_sha256") or "").lower()
        if (len(expected_hash) != 64
                or any(char not in "0123456789abcdef" for char in expected_hash)):
            raise ValueError(f"manifest {stream} calibration_sha256 is missing or invalid")
        sources[relative] = VerifiedSource(
            _resolve_manifest_calibration_path(
                session_dir, metadata.get("calibration_path"), stream=stream,
            ),
            expected_hash,
        )
    return sources


def _manifest_bound_to_snapshot(manifest: dict) -> dict:
    """Redirect calibration reads to authenticated copies inside the frozen snapshot."""
    bound = json.loads(json.dumps(manifest))
    bound.setdefault("sync", {})["calibration_path"] = "calibrations/sync.json"
    bound.setdefault("force", {})["calibration_path"] = "calibrations/force.json"
    bound.setdefault("camera", {}).setdefault("timing", {})[
        "calibration_path"
    ] = "calibrations/camera_latency.json"
    return bound


def build(
    session_dir, repo_id, root, fps=None, task=None, *, dataset_cls=None,
    trusted_finalization_registry=None, trusted_snapshot_root=None,
    trusted_session_id=None,
):
    session_dir = Path(session_dir).resolve()
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(f"dataset root must not already exist: {root}")
    for child, parent in ((root, session_dir), (session_dir, root)):
        try:
            child.relative_to(parent)
        except ValueError:
            continue
        raise ValueError(f"dataset root and source session overlap: {root}, {session_dir}")
    root.parent.mkdir(parents=True, exist_ok=True)

    # Verify the complete promotion-v2 chain before trusting even the manifest's
    # training_eligible bit. Every conversion read below is then performed
    # against a second, independently verified frozen copy.
    verified_training_contract = _verify_training_session_contract(
        session_dir,
        trusted_finalization_registry=trusted_finalization_registry,
        trusted_snapshot_root=trusted_snapshot_root,
        trusted_session_id=trusted_session_id,
    )
    session_dir = Path(verified_training_contract["verified_snapshot_path"]).resolve()
    source_session_id = str(verified_training_contract["session_id"])
    for child, parent in ((root, session_dir), (session_dir, root)):
        try:
            child.relative_to(parent)
        except ValueError:
            continue
        raise ValueError(
            "dataset root and trusted source snapshot overlap: "
            f"{root}, {session_dir}"
        )
    discovery_manifest_bytes = (session_dir / "manifest.json").read_bytes()
    discovery_manifest = json.loads(discovery_manifest_bytes.decode("utf-8"))
    discovery_manifest_sha256 = hashlib.sha256(discovery_manifest_bytes).hexdigest()
    if discovery_manifest_sha256 != verified_training_contract["manifest_sha256"]:
        raise ValueError("training manifest changed after promotion-v2 verification")
    validate_manifest_for_conversion(discovery_manifest)
    expected_contract_hashes = _training_contract_file_hashes(
        session_dir, discovery_manifest, verified_training_contract,
    )
    calibration_sources = _manifest_calibration_sources(session_dir, discovery_manifest)
    _verify_finalization_seal(session_dir, manifest_bytes=discovery_manifest_bytes)

    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.input-snapshot-", dir=str(root.parent),
    ) as snapshot_workspace:
        snapshot_provenance = snapshot_session_sources(
            session_dir, snapshot_workspace, extra_sources=calibration_sources,
            expected_contract_hashes=expected_contract_hashes,
            verify_training_contract=True,
            verified_training_contract=verified_training_contract,
            metadata={
                "snapshot_state": "raw_inputs_frozen",
                "promotion_contract_version": PROMOTION_CONTRACT_VERSION,
                "source_training_contract": verified_training_contract,
            },
        )
        snapshot_dir = Path(snapshot_workspace) / "source_sessions" / source_session_id
        manifest = _manifest_bound_to_snapshot(json.loads(
            (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
        ))
        validate_manifest_for_conversion(manifest)
        raw_authenticity = _reverify_component_raw_gates(
            snapshot_dir,
            outer_contract=snapshot_provenance.get("frozen_training_contract"),
        )
        sync_meta = verify_sync_metadata(snapshot_dir, manifest)
        verify_force_calibration_metadata(snapshot_dir, manifest)
        _persist_replayable_snapshot_manifest(snapshot_dir, manifest)
        raw_force = read_force(snapshot_dir)
        if raw_force.device_t_ms is None:
            raise ValueError("canonical v2 requires force device_t_ms for drift correction")
        force_fit = fit_device_clock_checked(
            raw_force.t_host_s, raw_force.device_t_ms, stream="force",
        )
        corrected_force_t = (force_fit["host_intercept_s"]
                             + force_fit["host_s_per_device_ms"] * raw_force.device_t_ms)
        force = replace(raw_force, t_host_s=corrected_force_t)
        # P0-4: an over-threshold ("unsafe") force status is a VALID physician-reference
        # measurement, not a session-rejecting fault. Only a non-native status is
        # invalid; per-frame contact_force_valid conveys genuineness downstream.
        _NATIVE_FORCE_STATUS = ("OK", "WARN", "SCREENED_LIMIT", "HARD_LIMIT", "")
        if any(status not in _NATIVE_FORCE_STATUS for status in force.status):
            raise ValueError("force.csv contains a non-native status")
        camera = read_camera_timeline(snapshot_dir, manifest)
        robot = read_exact_robot_frames(snapshot_dir, camera, manifest)
        # The currently audited OpenArm SDK returns void from mit_control_all.
        # Preserve such captures for commissioning, but never publish them as a
        # LeRobot training dataset until a new canonical sidecar proves send
        # status and independently clears every eligibility blocker.
        require_exact_robot_training_eligibility(robot.summary)
        targets = read_targets(snapshot_dir, camera)
        if any(point is None or frame != "robot_base"
               for point, frame in zip(targets.xyz, targets.xyz_frames)):
            raise ValueError("canonical v2 requires every target xyz in robot_base frame")
        segments = _build_target_split_segments(snapshot_dir, force, targets, sync_meta)
        reward_spans = [
            RewardSpan(seg.t_start, seg.t_end, float(seg.reward or 0.0)) for seg in segments
        ]
        records = prepare_exact_causal_records(
            camera, robot, force, targets, reward_spans,
            force_max_age_s=0.050, target_max_age_s=0.250,
        )
        # Fill the VLA "L" with an anatomy-grounded, phrasing-augmented instruction. Phrasing
        # is sampled from the fixed pool deterministically per (session, site) -- reproducible,
        # yet varied across sites/sessions so the policy does not overfit to one phrasing. Zero
        # per-segment human cost. An optional --task prefix is preserved for datamix labels.
        def _augmented_task(record):
            if record.target_name is None:
                return f"{task}: {DEFAULT_TASK}" if task else DEFAULT_TASK
            phrased = phrase_site_instruction(
                record.target_name, seed=f"{source_session_id}:{record.target_name}",
            )
            return f"{task}: {phrased}" if task else phrased

        records = [replace(record, task=_augmented_task(record)) for record in records]

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ausc" / "audio"))
        from reward_schema import save_segments
        snapshot_segments = snapshot_dir / "segments.jsonl"
        save_segments(str(snapshot_segments), segments)
        action_selection = write_exact_action_selection(
            snapshot_dir / "robot_frame_action_selection.jsonl", robot,
        )
        _update_snapshot_provenance(
            snapshot_dir,
            metadata={
                "snapshot_state": "conversion_inputs_frozen",
                "converter": "session_to_lerobot_v2.py",
                "lerobot_version": "0.3.3",
                "alignment": {
                    "robot_observation": (
                        "interpolate_observation_only_at_exposure_nominal_ns"
                    ),
                    "robot_action": EXACT_ACTION_SELECTION_RULE,
                    "robot_action_interpolation": "forbidden",
                    "force_max_age_s": 0.050,
                    "target_max_age_s": 0.250,
                },
                "robot_exact_action": action_selection,
                "raw_signal_authenticity": raw_authenticity,
                "force_device_clock_fit": force_fit,
                "camera_device_clock_fit": camera.clock_fit,
                "camera_latency_ms_applied": camera.latency_ms,
                "audio_sync": {
                    "applied": True,
                    "equation": "t_audio_aligned_s = t_audio_host_s + offset_ms / 1000",
                    "audio_to_force_offset_ms": sync_meta["audio_to_force_offset_ms"],
                },
            },
        )
        _verify_snapshot_integrity(snapshot_dir)

        staging_root = Path(tempfile.mkdtemp(
            prefix=f".{root.name}.staging-", dir=str(root.parent),
        ))
        # Reserve a collision-resistant sibling name, while preserving LeRobot's
        # create-on-new-root contract.
        staging_root.rmdir()
        try:
            import cv2
            cap = cv2.VideoCapture(str(snapshot_dir / "rgb.mp4"))
            try:
                ok, first_bgr = cap.read()
                if not ok or first_bgr is None:
                    raise ValueError("rgb.mp4 has no decodable first frame")
                height, width = first_bgr.shape[:2]
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if fps is None:
                    camera_dt = np.diff(camera.t_exposure_s)
                    cam_fps = int(round(1.0 / float(np.median(camera_dt))))
                else:
                    cam_fps = int(fps)
                if cam_fps <= 0:
                    raise ValueError("dataset fps must be positive")
                joints = list(robot.joint_names); nj = len(joints)
                action_names = list(robot.action_names); na = len(action_names)
                features = {
                    "observation.state": {"dtype": "float32", "shape": (nj,), "names": joints},
                    "observation.robot_state_age_s": {
                        "dtype": "float32", "shape": (1,), "names": None,
                    },
                    "observation.images.top": {
                        "dtype": "video", "shape": [height, width, 3],
                        "names": ["height", "width", "channels"],
                    },
                    "observation.contact_force_n": {
                        "dtype": "float32", "shape": (1,), "names": None,
                    },
                    "observation.contact_force_valid": {
                        "dtype": "bool", "shape": (1,), "names": None,
                    },
                    "observation.contact_force_age_s": {
                        "dtype": "float32", "shape": (1,), "names": None,
                    },
                    "observation.target_point_2d": {
                        "dtype": "float32", "shape": (2,), "names": ["u", "v"],
                    },
                    "observation.target_point_3d": {
                        "dtype": "float32", "shape": (3,), "names": ["x", "y", "z"],
                    },
                    "observation.target_age_s": {
                        "dtype": "float32", "shape": (1,), "names": None,
                    },
                    "action": {
                        "dtype": "float32", "shape": (na,), "names": action_names,
                    },
                    "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
                    "next.done": {"dtype": "bool", "shape": (1,), "names": None},
                }

                if dataset_cls is None:
                    from lerobot.datasets.lerobot_dataset import LeRobotDataset
                    dataset_cls = LeRobotDataset
                ds = dataset_cls.create(
                    repo_id=repo_id, fps=cam_fps, features=features, root=staging_root,
                    robot_type=f"openarm_{manifest['arm']['side']}", use_videos=True,
                )

                def decoded_records():
                    for record in records:
                        ok, bgr = cap.read()
                        if not ok or bgr is None:
                            raise ValueError("rgb.mp4 ended before rgb_times.csv")
                        record.frame["observation.images.top"] = cv2.cvtColor(
                            bgr, cv2.COLOR_BGR2RGB,
                        )
                        yield record

                write_lerobot_033(ds, decoded_records())
            finally:
                cap.release()

            staging_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot_segments, staging_root / "segments.jsonl")
            staged_snapshot = staging_root / "source_sessions" / session_dir.name
            staged_snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(snapshot_dir), str(staged_snapshot))
            _verify_snapshot_integrity(staged_snapshot)
            _retarget_dataset_root(ds, staging_root, root)
        except BaseException:
            if staging_root.exists():
                shutil.rmtree(staging_root)
            raise

    # The snapshot workspace is gone and every staged artifact has been verified.
    # Publication is one same-filesystem atomic rename; no failure path above can
    # expose a partial final root.
    try:
        os.replace(staging_root, root)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    print(f"[v2] {repo_id}: {len(records)} frames @ {cam_fps}fps, {nj}-DOF, "
          f"{na}-parameter exact MIT actions, images.top {height}x{width}, "
          f"causal observations + target episodes -> {root}")
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--repo-id", default="ausc/demo")
    ap.add_argument("--root", required=True)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--trusted-finalization-registry", default=None)
    ap.add_argument("--trusted-snapshot-root", default=None)
    ap.add_argument("--trusted-session-id", default=None)
    a = ap.parse_args()
    build(
        a.session_dir, a.repo_id, a.root, a.fps, a.task,
        trusted_finalization_registry=a.trusted_finalization_registry,
        trusted_snapshot_root=a.trusted_snapshot_root,
        trusted_session_id=a.trusted_session_id,
    )
