"""Issue an immutable, explicitly provisional OpenArm hand-eye activation.

This issuer is intentionally separate from the calibration solver.  A rejected
run may only enter this path when an operator has produced the narrow
``wam_handeye_user_accepted_v1`` decision artifact.  Publication completes an
artifact lifecycle; it never changes the calibration decision to a passed
quality gate.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import uuid
from typing import Any, Mapping, Sequence

import numpy as np


RUN_RE = re.compile(r"^handeye_left_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CODE_RELEASE_RE = re.compile(r"^handeye_calibration_code-[0-9a-f]{20}$")
RESULT_NAME = "cam2base_USER_ACCEPTED_5mm.json"
ARTIFACT_NAME = "handeye.json"
ARTIFACT_COMPLETE_NAME = "handeye.COMPLETE"
RELEASE_COMPLETE_NAME = "COMPLETE"
ARTIFACT_SCHEMA = "wam_openarm_handeye_activation_v2"
SOURCE_SCHEMA = "wam_handeye_user_accepted_v1"
FAILURE_SCHEMA = "wam_handeye_failure_v1"
CALIBRATION_STATUS = "USER_ACCEPTED_BELOW_GATE"
AUTHORIZED_SCOPES = (
    "bootstrap_teleop_logging",
    "conservative_robot_mask",
)
PRODUCTION_RUNS_ROOT = Path(
    "/home/robot/wam_data/outputs/perception/calibration/runs"
)
PRODUCTION_ACTIVATION_ROOT = Path(
    "/home/robot/wam_data/outputs/perception/calibration/handeye_activations"
)
SOURCE_KEYS = {
    "T_base_to_cam",
    "T_cam_to_base",
    "arm",
    "code_release",
    "convention",
    "covariance_se3_T_cam_to_base",
    "inputs",
    "note",
    "quality",
    "run_id",
    "schema",
    "status",
    "tip_offset_tcp_m",
}
OWNER_AUTHORIZATION_KEYS = {
    "authorized_by",
    "autonomous_targeting_authorized_by_owner",
    "conditions",
    "date_utc",
    "explicit_signoff",
    "note",
}
QUALITY_KEYS = {
    "fit_count",
    "gate_thresholds_mm",
    "heldout_count",
    "heldout_errors_mm",
    "result_precision_mm_1sigma_max",
    "result_precision_mm_1sigma_median",
    "solver_condition",
    "solver_rank",
    "training_errors_mm",
    "training_max_mm",
    "training_rms_mm",
}
SAMPLE_KEYS = {
    "T_tcp_base",
    "camera_age_s",
    "camera_device_t_ms",
    "camera_frame_index",
    "camera_host_monotonic_s",
    "camera_time_source",
    "depth_m",
    "depth_patch_span_mm",
    "depth_patch_valid_count",
    "depth_patch_window_size",
    "depth_path",
    "depth_raw",
    "depth_scale_mm",
    "depth_sha256",
    "orientation",
    "phase",
    "pixel",
    "position_index",
    "q7_rad",
    "rgb_path",
    "rgb_sha256",
    "sample_id",
    "tip_cam_m",
}
EXPECTED_COVARIANCE_ORDER = (
    "rx_rad",
    "ry_rad",
    "rz_rad",
    "tx_m",
    "ty_m",
    "tz_m",
)
EXPECTED_COVARIANCE_DEFINITION = (
    "empirical bootstrap cov of delta=[Rodrigues(R_ref^T R_boot), "
    "t_boot - t_ref]; random precision only, excludes systematic bias"
)
EXPECTED_CONVENTION = (
    "p_base = T_cam_to_base @ p_cam ; a point measured in the COLOR-camera "
    "optical frame (metric xyz, meters) maps into the ARM BASE "
    "(openarm_left_link0) frame. tip_offset_tcp_m is the solved tool point in "
    "the openarm_left_hand_tcp frame."
)


class HandEyeActivationError(RuntimeError):
    """The source run cannot be honestly issued as this activation tier."""


def _canonical(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HandEyeActivationError("value is not canonical JSON") from exc


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HandEyeActivationError("duplicate JSON field: %s" % key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise HandEyeActivationError("non-finite JSON constant is forbidden: %s" % value)


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
        getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000)),
    )


def _direct_path(path: Path, *, label: str, directory: bool) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise HandEyeActivationError("%s must be a direct absolute path" % label)
    current = Path(candidate.anchor)
    for index, part in enumerate(candidate.parts[1:]):
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise HandEyeActivationError("%s is missing" % label) from exc
        if stat.S_ISLNK(info.st_mode):
            raise HandEyeActivationError("%s path contains a symlink" % label)
        final = index == len(candidate.parts[1:]) - 1
        wanted = stat.S_ISDIR if (directory or not final) else stat.S_ISREG
        if not wanted(info.st_mode):
            raise HandEyeActivationError("%s has an invalid file type" % label)
    return candidate


def _stable_read(path: Path, *, label: str, maximum: int | None = None) -> bytes:
    direct = _direct_path(path, label=label, directory=False)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(direct), flags)
    except OSError as exc:
        raise HandEyeActivationError("%s cannot be opened" % label) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HandEyeActivationError("%s is not a regular file" % label)
        if maximum is not None and before.st_size > maximum:
            raise HandEyeActivationError("%s is too large" % label)
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise HandEyeActivationError("%s was truncated" % label)
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after):
        raise HandEyeActivationError("%s changed while being read" % label)
    return b"".join(chunks)


def _decode_json(body: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except HandEyeActivationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandEyeActivationError("%s is not valid JSON" % label) from exc
    if type(value) is not dict:
        raise HandEyeActivationError("%s must be a JSON object" % label)
    return value


def _json_file(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    body = _stable_read(path, label=label, maximum=16 * 1024 * 1024)
    return _decode_json(body, label=label), body


def _finite_number(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HandEyeActivationError("%s must be finite" % label)
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise HandEyeActivationError("%s must be finite%s" % (
            label, " and positive" if positive else "",
        ))
    return result


def _rigid_transform(value: object, *, label: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HandEyeActivationError("%s is invalid" % label) from exc
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12)
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8)
        or not math.isclose(
            float(np.linalg.det(transform[:3, :3])), 1.0,
            rel_tol=0.0, abs_tol=1e-8,
        )
    ):
        raise HandEyeActivationError("%s is not a rigid transform" % label)
    return transform


def _relative_capture(value: object, *, suffix: str, label: str) -> str:
    if type(value) is not str:
        raise HandEyeActivationError("%s path is invalid" % label)
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or len(parsed.parts) != 2
        or parsed.parts[0] != "captures"
        or ".." in parsed.parts
        or not parsed.name.endswith(suffix)
    ):
        raise HandEyeActivationError("%s path is unsafe" % label)
    return value


def _require_sha(value: object, *, label: str) -> str:
    if type(value) is not str or SHA_RE.fullmatch(value) is None:
        raise HandEyeActivationError("%s must be a SHA-256" % label)
    return value


def _verify_quality(source: Mapping[str, Any], failed: Mapping[str, Any]) -> dict[str, Any]:
    quality = source.get("quality")
    if type(quality) is not dict or set(quality) != QUALITY_KEYS:
        raise HandEyeActivationError("source quality schema is invalid")
    if quality["fit_count"] != 20 or quality["heldout_count"] != 3:
        raise HandEyeActivationError("source quality sample counts are invalid")
    thresholds = quality["gate_thresholds_mm"]
    if thresholds != {"each_point": 5.0, "heldout": 5.0, "training_rms": 3.0}:
        raise HandEyeActivationError("source quality gate thresholds are invalid")
    training = quality["training_errors_mm"]
    heldout = quality["heldout_errors_mm"]
    if type(training) is not list or len(training) != 20 or type(heldout) is not list or len(heldout) != 3:
        raise HandEyeActivationError("source quality error vectors are invalid")
    training_values = [_finite_number(value, label="training error") for value in training]
    heldout_values = [_finite_number(value, label="heldout error") for value in heldout]
    rms = _finite_number(quality["training_rms_mm"], label="training RMS", positive=True)
    maximum = _finite_number(quality["training_max_mm"], label="training maximum", positive=True)
    expected_rms = math.sqrt(sum(value * value for value in training_values) / 20.0)
    if not math.isclose(rms, expected_rms, rel_tol=0.0, abs_tol=1e-9):
        raise HandEyeActivationError("training RMS is inconsistent")
    if not math.isclose(maximum, max(training_values), rel_tol=0.0, abs_tol=1e-9):
        raise HandEyeActivationError("training maximum is inconsistent")
    reasons = failed.get("failure_reasons")
    if type(reasons) is not list or not all(type(item) is str for item in reasons):
        raise HandEyeActivationError("FAILED failure reasons are invalid")
    required = {"training_rms_above_3mm", "training_max_error_above_5mm"}
    if rms <= 3.0 or maximum <= 5.0 or not required.issubset(reasons):
        raise HandEyeActivationError("source is not honestly below the 3mm quality gate")
    if max(heldout_values) > 5.0 and "heldout_error_above_5mm" not in reasons:
        raise HandEyeActivationError("heldout failure reason is missing")
    median = _finite_number(
        quality["result_precision_mm_1sigma_median"],
        label="workspace median precision", positive=True,
    )
    maximum_precision = _finite_number(
        quality["result_precision_mm_1sigma_max"],
        label="workspace maximum precision", positive=True,
    )
    if median > maximum_precision or median > 5.5 or maximum_precision > 6.5:
        raise HandEyeActivationError(
            "workspace 5mm-tier 1sigma precision claim is invalid"
        )
    if quality["solver_rank"] != 9:
        raise HandEyeActivationError("solver rank is invalid")
    _finite_number(quality["solver_condition"], label="solver condition", positive=True)
    return json.loads(json.dumps(quality, allow_nan=False))


def _verify_failed(failed: Mapping[str, Any], *, run_id: str) -> tuple[str, dict[str, Any]]:
    if set(failed) != {"failure_reasons", "provenance", "schema", "status"}:
        raise HandEyeActivationError("FAILED schema is invalid")
    if failed["schema"] != FAILURE_SCHEMA or failed["status"] != "REJECTED":
        raise HandEyeActivationError("FAILED status is invalid")
    provenance = failed["provenance"]
    if type(provenance) is not dict:
        raise HandEyeActivationError("FAILED provenance is invalid")
    if (
        provenance.get("run_id") != run_id
        or provenance.get("arm") != "left"
        or provenance.get("can_interface") != "can3"
        or provenance.get("mode") != "tiptouch"
        or provenance.get("sample_count") != 23
        or provenance.get("production_activation") is not False
    ):
        raise HandEyeActivationError("FAILED provenance identity is invalid")
    camera = provenance.get("camera")
    if type(camera) is not dict:
        raise HandEyeActivationError("FAILED camera provenance is invalid")
    camera_id = camera.get("device_identity")
    if type(camera_id) is not str or not camera_id or camera.get("requested_serial") != camera_id:
        raise HandEyeActivationError("FAILED camera identity is invalid")
    contract = provenance.get("capture_contract")
    expected_contract_keys = {
        "fit_count",
        "heldout_count",
        "pair_count",
        "rigid_nonhuman_fixture_required",
        "two_wrist_orientations_per_position",
        "fit_3d_rms_max_mm",
        "fit_each_point_max_mm",
        "heldout_each_point_max_mm",
        "minimum_pair_rotation_deg",
        "maximum_pair_contact_delta_mm",
    }
    if (
        type(contract) is not dict
        or set(contract) != expected_contract_keys
        or (
            contract.get("fit_count"), contract.get("heldout_count"),
            contract.get("fit_3d_rms_max_mm"),
            contract.get("fit_each_point_max_mm"),
            contract.get("heldout_each_point_max_mm"), contract.get("pair_count"),
            contract.get("minimum_pair_rotation_deg"),
            contract.get("maximum_pair_contact_delta_mm"),
        ) != (20, 3, 3.0, 5.0, 5.0, 10, 30.0, 5.0)
        or contract.get("rigid_nonhuman_fixture_required") is not True
        or contract.get("two_wrist_orientations_per_position") is not True
    ):
        raise HandEyeActivationError("FAILED capture contract is invalid")
    return camera_id, dict(provenance)


def _exact_run_inventory(run: Path) -> None:
    expected_top = {
        RESULT_NAME,
        "FAILED.json",
        "captures",
        "joint_stream_snapshot.csv",
        "samples.jsonl",
    }
    try:
        top = {path.name for path in run.iterdir()}
    except OSError as exc:
        raise HandEyeActivationError("run inventory cannot be enumerated") from exc
    if top != expected_top:
        raise HandEyeActivationError("run inventory is not exact")
    captures = run / "captures"
    expected_captures = {
        "%04d_%s.%s" % (index, kind, extension)
        for index in range(1, 24)
        for kind, extension in (("rgb", "png"), ("depth", "npy"))
    }
    try:
        observed_captures = {path.name for path in captures.iterdir()}
    except OSError as exc:
        raise HandEyeActivationError("capture inventory cannot be enumerated") from exc
    if observed_captures != expected_captures:
        raise HandEyeActivationError("capture inventory is not exact")
    for name in expected_top - {"captures"}:
        path = run / name
        if path.is_symlink():
            raise HandEyeActivationError("run inventory contains a symlink")
        if not path.is_file():
            raise HandEyeActivationError("run inventory contains a non-regular file")
    if captures.is_symlink() or not captures.is_dir():
        raise HandEyeActivationError("capture inventory directory is invalid")
    for name in expected_captures:
        path = captures / name
        if path.is_symlink():
            raise HandEyeActivationError("capture inventory contains a symlink")
        if not path.is_file():
            raise HandEyeActivationError("capture inventory contains a non-regular file")


def _verify_samples(
    run: Path,
    samples_body: bytes,
    *,
    capture_contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lines = samples_body.splitlines()
    if len(lines) != 23 or any(not line for line in lines):
        raise HandEyeActivationError("samples must contain exactly 23 records")
    captures: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    fit_count = heldout_count = 0
    previous_time = -math.inf
    previous_frame = -1
    for index, line in enumerate(lines, start=1):
        sample = _decode_json(line, label="sample %d" % index)
        if set(sample) != SAMPLE_KEYS or sample.get("sample_id") != index:
            raise HandEyeActivationError("sample schema/order is invalid")
        phase = sample.get("phase")
        if phase == "fit":
            fit_count += 1
            expected_position = (index - 1) // 2 + 1
            expected_orientation = "A" if index % 2 else "B"
        elif phase == "heldout":
            heldout_count += 1
            expected_position = index - 20
            expected_orientation = None
        else:
            raise HandEyeActivationError("sample phase is invalid")
        if (
            sample.get("position_index") != expected_position
            or sample.get("orientation") != expected_orientation
        ):
            raise HandEyeActivationError("sample position/orientation plan is invalid")
        q = np.asarray(sample.get("q7_rad"), dtype=np.float64)
        tcp = _rigid_transform(sample.get("T_tcp_base"), label="sample TCP transform")
        tip = np.asarray(sample.get("tip_cam_m"), dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all() or tip.shape != (3,) or not np.isfinite(tip).all():
            raise HandEyeActivationError("sample robot/camera geometry is invalid")
        frame_index = sample.get("camera_frame_index")
        camera_time = _finite_number(
            sample.get("camera_host_monotonic_s"), label="sample camera time",
            positive=True,
        )
        if (
            type(frame_index) is not int
            or frame_index < 0
            or frame_index < previous_frame
            or camera_time < previous_time
        ):
            raise HandEyeActivationError("sample camera sequence/time is invalid")
        if phase == "fit" and index % 2 == 0:
            previous_sample = samples[-1]
            previous_tcp = _rigid_transform(
                previous_sample["T_tcp_base"], label="paired sample TCP transform",
            )
            relative_rotation = previous_tcp[:3, :3].T @ tcp[:3, :3]
            cosine = float((np.trace(relative_rotation) - 1.0) / 2.0)
            pair_rotation_deg = math.degrees(
                math.acos(max(-1.0, min(1.0, cosine)))
            )
            minimum_rotation_deg = float(capture_contract["minimum_pair_rotation_deg"])
            if pair_rotation_deg + 1e-9 < minimum_rotation_deg:
                raise HandEyeActivationError(
                    "paired sample pair rotation is below the capture contract"
                )
            previous_tip = np.asarray(previous_sample["tip_cam_m"], dtype=np.float64)
            pair_contact_delta_mm = float(np.linalg.norm(previous_tip - tip) * 1000.0)
            maximum_contact_delta_mm = float(
                capture_contract["maximum_pair_contact_delta_mm"]
            )
            if pair_contact_delta_mm > maximum_contact_delta_mm + 1e-9:
                raise HandEyeActivationError(
                    "paired sample pair contact exceeds the capture contract"
                )
        previous_frame = frame_index
        previous_time = camera_time
        depth_m = _finite_number(sample.get("depth_m"), label="sample depth", positive=True)
        depth_raw = _finite_number(sample.get("depth_raw"), label="sample raw depth", positive=True)
        depth_scale_mm = _finite_number(
            sample.get("depth_scale_mm"), label="sample depth scale", positive=True,
        )
        if not math.isclose(
            depth_m, depth_raw * depth_scale_mm / 1000.0,
            rel_tol=0.0, abs_tol=1e-9,
        ):
            raise HandEyeActivationError("sample metric depth conversion is invalid")
        patch_valid = sample.get("depth_patch_valid_count")
        patch_window = sample.get("depth_patch_window_size")
        if (
            type(patch_valid) is not int
            or type(patch_window) is not int
            or patch_window != 5
            or not 1 <= patch_valid <= patch_window * patch_window
        ):
            raise HandEyeActivationError("sample depth patch evidence is invalid")
        for kind, suffix in (("rgb", "_rgb.png"), ("depth", "_depth.npy")):
            relative = _relative_capture(
                sample.get(kind + "_path"), suffix=suffix,
                label="sample %s" % kind,
            )
            expected_relative = "captures/%04d_%s%s" % (index, kind, suffix[len("_" + kind):])
            if relative != expected_relative:
                raise HandEyeActivationError("sample capture naming is invalid")
            if relative in observed_paths:
                raise HandEyeActivationError("capture path is duplicated")
            observed_paths.add(relative)
            body = _stable_read(run / Path(*PurePosixPath(relative).parts), label="capture %s" % relative)
            expected = _require_sha(sample.get(kind + "_sha256"), label="capture SHA-256")
            digest = _sha256(body)
            if digest != expected:
                raise HandEyeActivationError("capture SHA-256 mismatch: %s" % relative)
            captures.append({"path": relative, "sha256": digest, "size_bytes": len(body)})
        samples.append(sample)
    if fit_count != 20 or heldout_count != 3 or len(captures) != 46:
        raise HandEyeActivationError("sample phase/capture inventory is incomplete")
    return sorted(captures, key=lambda row: row["path"]), samples


def _revalidate_source_closure(
    run: Path,
    *,
    source_hashes: Mapping[str, str],
    captures: Sequence[Mapping[str, Any]],
) -> None:
    """Second full pass proving the issued closure did not drift while built."""

    _exact_run_inventory(run)
    regular = {
        "failed": "FAILED.json",
        "joint_stream_snapshot": "joint_stream_snapshot.csv",
        "result": RESULT_NAME,
        "samples": "samples.jsonl",
    }
    for key, relative in regular.items():
        body = _stable_read(run / relative, label="closure %s" % relative)
        if _sha256(body) != source_hashes[key]:
            raise HandEyeActivationError("source closure changed during activation")
    for row in captures:
        relative = row["path"]
        body = _stable_read(
            run / Path(*PurePosixPath(relative).parts),
            label="closure capture %s" % relative,
        )
        if len(body) != row["size_bytes"] or _sha256(body) != row["sha256"]:
            raise HandEyeActivationError(
                "source closure capture SHA-256 changed during activation"
            )


def _build(run_id: str, runs_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    if type(run_id) is not str or RUN_RE.fullmatch(run_id) is None:
        raise HandEyeActivationError("run id is invalid")
    root = _direct_path(Path(runs_root).absolute(), label="hand-eye runs root", directory=True)
    run = _direct_path(root / run_id, label="hand-eye run", directory=True)
    if run.parent != root:
        raise HandEyeActivationError("run topology is invalid")
    _exact_run_inventory(run)

    source, source_body = _json_file(run / RESULT_NAME, label="accepted result")
    source_keys = set(source)
    if source_keys not in (SOURCE_KEYS, SOURCE_KEYS | {"owner_authorization"}):
        raise HandEyeActivationError("accepted result JSON schema is invalid")
    owner_authorization = source.get("owner_authorization")
    if owner_authorization is not None:
        if (
            type(owner_authorization) is not dict
            or set(owner_authorization) != OWNER_AUTHORIZATION_KEYS
            or type(owner_authorization.get("autonomous_targeting_authorized_by_owner"))
            is not bool
            or any(
                type(owner_authorization.get(key)) is not str
                or not owner_authorization[key].strip()
                for key in OWNER_AUTHORIZATION_KEYS
                - {"autonomous_targeting_authorized_by_owner"}
            )
        ):
            raise HandEyeActivationError("owner authorization metadata is invalid")
    if (
        source.get("schema") != SOURCE_SCHEMA
        or source.get("status") != CALIBRATION_STATUS
        or source.get("run_id") != run_id
        or source.get("arm") != "left"
        or source.get("convention") != EXPECTED_CONVENTION
    ):
        raise HandEyeActivationError("accepted result identity/status is invalid")

    failed, failed_body = _json_file(run / "FAILED.json", label="FAILED provenance")
    camera_id, provenance = _verify_failed(failed, run_id=run_id)
    quality = _verify_quality(source, failed)
    code_release = source.get("code_release")
    provenance_release = (
        provenance.get("software_provenance", {})
        .get("code_release", {})
        .get("release_id")
        if type(provenance.get("software_provenance")) is dict
        else None
    )
    if (
        type(code_release) is not str
        or CODE_RELEASE_RE.fullmatch(code_release) is None
        or provenance_release != code_release
    ):
        raise HandEyeActivationError(
            "accepted code release does not match FAILED provenance"
        )

    inputs = source.get("inputs")
    expected_inputs = {
        "captures_dir": run / "captures",
        "full_provenance": run / "FAILED.json",
        "joint_stream_snapshot": run / "joint_stream_snapshot.csv",
        "samples_jsonl": run / "samples.jsonl",
    }
    if type(inputs) is not dict or set(inputs) != set(expected_inputs):
        raise HandEyeActivationError("accepted input references are invalid")
    for name, expected in expected_inputs.items():
        if Path(str(inputs[name])) != expected:
            raise HandEyeActivationError("accepted input path does not bind the source run")
    _direct_path(expected_inputs["captures_dir"], label="captures directory", directory=True)

    samples_body = _stable_read(expected_inputs["samples_jsonl"], label="samples JSONL")
    captures, samples = _verify_samples(
        run,
        samples_body,
        capture_contract=provenance["capture_contract"],
    )
    joint_body = _stable_read(expected_inputs["joint_stream_snapshot"], label="joint snapshot")
    if not joint_body:
        raise HandEyeActivationError("joint snapshot is empty")

    base_from_camera = _rigid_transform(source.get("T_cam_to_base"), label="T_cam_to_base")
    camera_from_base = _rigid_transform(source.get("T_base_to_cam"), label="T_base_to_cam")
    if not np.allclose(base_from_camera @ camera_from_base, np.eye(4), atol=1e-9):
        raise HandEyeActivationError("T_cam_to_base and T_base_to_cam are not inverse transforms")

    covariance_source = source.get("covariance_se3_T_cam_to_base")
    if type(covariance_source) is not dict or set(covariance_source) != {"definition", "matrix", "order"}:
        raise HandEyeActivationError("covariance schema is invalid")
    if tuple(covariance_source.get("order", ())) != EXPECTED_COVARIANCE_ORDER:
        raise HandEyeActivationError("covariance order is invalid")
    if covariance_source.get("definition") != EXPECTED_COVARIANCE_DEFINITION:
        raise HandEyeActivationError("covariance definition is invalid")
    try:
        covariance = np.asarray(covariance_source.get("matrix"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HandEyeActivationError("covariance matrix is invalid") from exc
    if (
        covariance.shape != (6, 6)
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, atol=1e-12)
        or float(np.min(np.linalg.eigvalsh(covariance))) < -1e-12
    ):
        raise HandEyeActivationError("covariance must be finite symmetric positive semidefinite")
    if float(np.max(np.linalg.eigvalsh(covariance))) <= 0.0:
        raise HandEyeActivationError("covariance cannot be all zero")

    propagated_sigma_mm = []
    rotation = base_from_camera[:3, :3]
    for sample in samples:
        point = np.asarray(sample["tip_cam_m"], dtype=np.float64)
        skew = np.asarray(
            (
                (0.0, -point[2], point[1]),
                (point[2], 0.0, -point[0]),
                (-point[1], point[0], 0.0),
            ),
            dtype=np.float64,
        )
        jacobian = np.hstack((-rotation @ skew, np.eye(3, dtype=np.float64)))
        point_covariance = jacobian @ covariance @ jacobian.T
        variance = float(np.trace(point_covariance))
        if not math.isfinite(variance) or variance <= 0.0:
            raise HandEyeActivationError("covariance point propagation is invalid")
        propagated_sigma_mm.append(math.sqrt(variance) * 1000.0)
    observed_median_mm = float(np.median(propagated_sigma_mm))
    observed_max_mm = float(np.max(propagated_sigma_mm))
    claimed_median_mm = float(quality["result_precision_mm_1sigma_median"])
    claimed_max_mm = float(quality["result_precision_mm_1sigma_max"])
    if (
        abs(observed_median_mm - claimed_median_mm) > 0.05
        or abs(observed_max_mm - claimed_max_mm) > 0.05
    ):
        raise HandEyeActivationError(
            "covariance does not reproduce the claimed workspace precision"
        )

    tip = np.asarray(source.get("tip_offset_tcp_m"), dtype=np.float64)
    if tip.shape != (3,) or not np.isfinite(tip).all() or not 0.10 < float(np.linalg.norm(tip)) < 0.30:
        raise HandEyeActivationError("tip offset is invalid")

    source_hashes = {
        "failed": _sha256(failed_body),
        "joint_stream_snapshot": _sha256(joint_body),
        "result": _sha256(source_body),
        "samples": _sha256(samples_body),
    }
    capture_inventory_body = _canonical(captures)
    source_hashes["captures_inventory"] = _sha256(capture_inventory_body)
    source_hashes["closure"] = _sha256(_canonical(source_hashes))
    acceptance_note = source.get("note")
    if type(acceptance_note) is not str or not acceptance_note.strip():
        raise HandEyeActivationError("explicit acceptance note is missing")
    precision = {
        "max": round(
            float(quality["result_precision_mm_1sigma_max"]) / 1000.0, 12,
        ),
        "median": round(
            float(quality["result_precision_mm_1sigma_median"]) / 1000.0, 12,
        ),
    }
    artifact = {
        "acceptance_note": acceptance_note,
        "authorized_scopes": list(AUTHORIZED_SCOPES),
        "autonomous_targeting_authorized": False,
        "bootstrap_reproducibility": {
            "missing": ["bootstrap_seed", "bootstrap_sample_count", "resample_policy"],
            "status": "INCOMPLETE",
        },
        "calibration_status": CALIBRATION_STATUS,
        "camera_id": camera_id,
        "code_release": code_release,
        "covariance_scope": "random_only",
        "covariance_se3": {
            "bound_transform": "transform_follower_base_from_camera",
            "definition": covariance_source["definition"],
            "matrix": covariance.tolist(),
            "order": list(EXPECTED_COVARIANCE_ORDER),
        },
        "failed_quality_gate_m": 0.003,
        "provisional": True,
        "quality": quality,
        "quality_gate_passed": False,
        "schema": ARTIFACT_SCHEMA,
        "side": "left",
        "source_evidence": {
            "captures": captures,
            "failed": {"path": "FAILED.json", "sha256": source_hashes["failed"], "size_bytes": len(failed_body)},
            "joint_stream_snapshot": {"path": "joint_stream_snapshot.csv", "sha256": source_hashes["joint_stream_snapshot"], "size_bytes": len(joint_body)},
            "result": {"path": RESULT_NAME, "sha256": source_hashes["result"], "size_bytes": len(source_body)},
            "sample_count": 23,
            "samples": {"path": "samples.jsonl", "sha256": source_hashes["samples"], "size_bytes": len(samples_body)},
            "captures_inventory_sha256": source_hashes["captures_inventory"],
        },
        "source_frame": "orbbec_color_optical_frame",
        "source_run_id": run_id,
        "systematic_bias_bound_m": None,
        "systematic_bias_status": "NOT_CHARACTERIZED",
        "target_frame": "openarm_left_link0",
        "tip_offset_frame": "openarm_left_hand_tcp",
        "tip_offset_tcp_m": tip.tolist(),
        "transform_camera_from_follower_base": camera_from_base.tolist(),
        "transform_follower_base_from_camera": base_from_camera.tolist(),
        "workspace_1sigma_m": precision,
    }
    _revalidate_source_closure(
        run, source_hashes=source_hashes, captures=captures,
    )
    return artifact, source_hashes


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(
        str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, body: bytes, mode: int = 0o644) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags, mode)
    try:
        view = memoryview(body)
        offset = 0
        while offset < len(view):
            count = os.write(descriptor, view[offset:])
            if count <= 0:
                raise OSError("short write")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _release_payloads(artifact: Mapping[str, Any]) -> tuple[bytes, bytes, bytes]:
    artifact_body = _canonical(artifact)
    artifact_sha = _sha256(artifact_body)
    artifact_complete = {
        "artifact_sha256": artifact_sha,
        "calibration_status": CALIBRATION_STATUS,
        "provisional": True,
        "schema": "wam_openarm_handeye_artifact_completion_v2",
        "status": "COMPLETE",
    }
    artifact_complete_body = _canonical(artifact_complete)
    release_id = "handeye_activation-" + artifact_sha[:20]
    inventory = {
        ARTIFACT_NAME: {
            "sha256": artifact_sha,
            "size_bytes": len(artifact_body),
        },
        ARTIFACT_COMPLETE_NAME: {
            "sha256": _sha256(artifact_complete_body),
            "size_bytes": len(artifact_complete_body),
        },
    }
    release_complete = {
        "artifact_sha256": artifact_sha,
        "inventory": inventory,
        "release_id": release_id,
        "schema": "wam_openarm_handeye_release_completion_v2",
        "status": "COMPLETE",
    }
    return artifact_body, artifact_complete_body, _canonical(release_complete)


def _verify_existing_release(
    release: Path, *, artifact_body: bytes, complete_body: bytes,
    release_complete_body: bytes,
) -> None:
    try:
        direct = _direct_path(release, label="existing activation release", directory=True)
        expected = {
            ARTIFACT_NAME: artifact_body,
            ARTIFACT_COMPLETE_NAME: complete_body,
            RELEASE_COMPLETE_NAME: release_complete_body,
        }
        if set(path.name for path in direct.iterdir()) != set(expected):
            raise HandEyeActivationError("existing activation release collision or tamper")
        if stat.S_IMODE(direct.stat().st_mode) != 0o555:
            raise HandEyeActivationError(
                "existing activation release immutable directory mode is invalid"
            )
        for name, body in expected.items():
            path = direct / name
            if stat.S_IMODE(path.stat().st_mode) != 0o444:
                raise HandEyeActivationError(
                    "existing activation release immutable file mode is invalid"
                )
            if _stable_read(path, label="existing %s" % name) != body:
                raise HandEyeActivationError("existing activation release collision or tamper")
    except HandEyeActivationError:
        raise
    except OSError as exc:
        raise HandEyeActivationError("existing activation release collision or tamper") from exc


def activate_handeye_run(
    run_id: str,
    *,
    runs_root: Path,
    activation_root: Path,
    execute: bool = False,
    confirm_run_id: str | None = None,
    confirm_source_sha256: str | None = None,
    test_only_allow_nonproduction_roots: bool = False,
) -> dict[str, Any]:
    """Validate one run, then optionally publish its immutable v2 activation."""

    runs = Path(runs_root).absolute()
    activation = Path(activation_root).absolute()
    if not test_only_allow_nonproduction_roots and (
        runs != PRODUCTION_RUNS_ROOT
        or activation != PRODUCTION_ACTIVATION_ROOT
    ):
        raise HandEyeActivationError(
            "hand-eye issuer requires the fixed Linbox production roots"
        )
    artifact, source_hashes = _build(run_id, runs)
    artifact_body, artifact_complete_body, release_complete_body = _release_payloads(artifact)
    artifact_sha = _sha256(artifact_body)
    release_id = "handeye_activation-" + artifact_sha[:20]
    root = activation
    release = root / "releases" / release_id
    report: dict[str, Any] = {
        "artifact": artifact,
        "artifact_sha256": artifact_sha,
        "release_id": release_id,
        "release_path": str(release),
        "semantic_source_hashes": source_hashes,
        "status": "DRY_RUN" if not execute else "COMPLETE",
    }
    if not execute:
        return report
    if confirm_run_id != run_id:
        raise HandEyeActivationError("--confirm-run-id does not match the run")
    if confirm_source_sha256 != source_hashes["closure"]:
        raise HandEyeActivationError("confirmed source SHA-256 does not match")

    # Confirmations are checked before creating even the management roots.
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise HandEyeActivationError("activation root is invalid")
    root.mkdir(parents=True, exist_ok=True)
    staging_root = root / ".staging"
    releases_root = root / "releases"
    staging_root.mkdir(exist_ok=True)
    releases_root.mkdir(exist_ok=True)
    for path, label in ((root, "activation root"), (staging_root, "staging root"), (releases_root, "releases root")):
        _direct_path(path, label=label, directory=True)
    _fsync_directory(root.parent)
    _fsync_directory(root)
    _fsync_directory(staging_root)
    _fsync_directory(releases_root)

    lock_path = root / ".publish.lock"
    lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_descriptor = os.open(str(lock_path), lock_flags, 0o600)
    except OSError as exc:
        raise HandEyeActivationError("activation publication lock is invalid") from exc
    try:
        lock_info = os.fstat(lock_descriptor)
        if not stat.S_ISREG(lock_info.st_mode):
            raise HandEyeActivationError("activation publication lock is invalid")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if release.exists() or release.is_symlink():
            _verify_existing_release(
                release,
                artifact_body=artifact_body,
                complete_body=artifact_complete_body,
                release_complete_body=release_complete_body,
            )
            report["reused"] = True
            return report

        staging = staging_root / (release_id + "." + uuid.uuid4().hex)
        staging.mkdir(mode=0o700)
        try:
            _write_new(staging / ARTIFACT_NAME, artifact_body)
            _write_new(staging / ARTIFACT_COMPLETE_NAME, artifact_complete_body)
            _write_new(staging / RELEASE_COMPLETE_NAME, release_complete_body)
            _fsync_directory(staging)
            for path in staging.iterdir():
                path.chmod(0o444)
                _fsync_file(path)
            _fsync_directory(staging)
            # Keep the private staging directory owner-writable through the
            # atomic rename.  Linbox's ext4 policy rejects renaming a 0555
            # directory even when both parent directories are writable.
            os.rename(str(staging), str(release))
            release.chmod(0o555)
            _fsync_directory(release)
            _verify_existing_release(
                release,
                artifact_body=artifact_body,
                complete_body=artifact_complete_body,
                release_complete_body=release_complete_body,
            )
            _fsync_directory(releases_root)
            _fsync_directory(root)
        except BaseException:
            candidate = staging if staging.exists() else release
            if candidate.exists() and candidate != release:
                try:
                    candidate.chmod(0o755)
                    shutil.rmtree(candidate)
                except OSError:
                    pass
            raise
    finally:
        os.close(lock_descriptor)
    report["reused"] = False
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Issue one explicit below-gate provisional hand-eye activation"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-run-id")
    parser.add_argument("--confirm-source-sha256")
    args = parser.parse_args(argv)

    # Import locally so unit tests do not mutate cwd/sys.path and production
    # cannot redirect these roots through an issuer-specific CLI option.
    from runtime_paths import activate_runtime

    paths = activate_runtime("linbox_calibration")
    if (
        sys.platform != "linux"
        or paths.host_role != "linbox"
        or paths.output_root != PRODUCTION_RUNS_ROOT.parents[1]
    ):
        raise HandEyeActivationError(
            "production activation CLI requires the real Linbox and fixed output root"
        )
    report = activate_handeye_run(
        args.run_id,
        runs_root=paths.output_root / "calibration" / "runs",
        activation_root=paths.output_root / "calibration" / "handeye_activations",
        execute=args.execute,
        confirm_run_id=args.confirm_run_id,
        confirm_source_sha256=args.confirm_source_sha256,
        test_only_allow_nonproduction_roots=False,
    )
    sys.stdout.buffer.write(_canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
