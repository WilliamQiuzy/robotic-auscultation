"""Render and install an immutable Linbox collector ``systemd --user`` unit.

The helper is intentionally standard-library-only and imports the existing
release verifiers beside it.  It never invokes systemctl and never follows a
``current`` pointer: callers must name the completed code, environment, and
perception-runtime releases explicitly.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

import artifact_contract as contract
import publish_perception_runtime as publisher


UNIT_SCHEMA = "wam_linbox_collector_unit_v2"
REQUIRED_RUNTIME_MANIFEST_V2 = "PERCEPTION_RUNTIME.json"
REQUIRED_RUNTIME_SCHEMA_V2 = "wam_perception_runtime_v5"
REQUIRED_RUNTIME_CAPABILITIES_V2 = {
    "arm_control_authority": False,
    "controller_launch_state": "DORMANT_HIL_PENDING",
    "training_promotion": "NOT_ACTIVATED",
}
VERIFY_CODE_SCHEMA = "wam_linbox_collector_code_preflight_v1"
VERIFY_RUNTIME_SCHEMA = "wam_linbox_collector_runtime_preflight_v1"
VERIFY_ACTIVATION_SCHEMA = "wam_linbox_collector_activation_preflight_v1"
VERIFY_COMMISSIONING_HANDEYE_SCHEMA = (
    "wam_linbox_collector_commissioning_handeye_preflight_v1"
)
VERIFY_HALL_MONITOR_SCHEMA = (
    "wam_linbox_collector_hall_monitor_preflight_v1"
)
PRODUCTION_ACTIVATION_SCHEMA = "wam_production_runtime_activation_v1"
PRODUCTION_ACTIVATION_COMPLETION_SCHEMA = (
    "wam_production_runtime_activation_completion_v1"
)
COLLECTOR_RELATIVE = "auscultation/scripts/collect_server.py"
GENERATOR_RELATIVE = "auscultation/scripts/deployment/collection_service_unit.py"
ENV_VERIFIER_RELATIVE = (
    "auscultation/scripts/deployment/verify_perception_environment.py"
)
_CODE_ID = re.compile(r"^code_release-[0-9a-f]{20}$")
_RUNTIME_ID = re.compile(r"^perception_runtime-[0-9a-f]{20}$")
_ENVIRONMENT_ID = re.compile(r"^linbox-perception-[0-9a-f]{20}$")
_CONTROLLER_LAUNCH_ID = re.compile(r"^controller_launch_v1-([0-9a-f]{64})$")
_HANDEYE_ID = re.compile(r"^handeye_activation-[0-9a-f]{20}$")
_CAMERA_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_AUDIO_DEVICE = re.compile(r"^[A-Za-z0-9._,:=() -]{1,128}$")
_SERIAL_PORT = re.compile(r"^/dev/[A-Za-z0-9._/-]{1,240}$")
_DYNAMIC_COMPONENTS = frozenset(
    {"current", ".wam-current", "wam-current", "latest", "active"}
)


class UnitContractError(RuntimeError):
    """The collector unit or its immutable inputs violate the contract."""


@dataclass(frozen=True)
class ServiceSpec:
    code_release: Path
    code_release_id: str
    environment_release: Path
    environment_id: str
    runtime_release: Path
    runtime_release_id: str
    camera_id: str
    sessions_root: Path
    unit_path: Path
    verifier_python: Path
    production_activation: Optional[Path]
    production_activation_sha256: Optional[str]
    hall_calibration: Optional[Path]
    hall_calibration_sha256: Optional[str]
    hall_port: Optional[str]
    sync_calibration: Optional[Path]
    sync_calibration_sha256: Optional[str]
    controller_launch_release: Optional[Path] = None
    controller_launch_trusted_root: Optional[Path] = None
    controller_launch_contract_sha256: Optional[str] = None
    # Candidate CAN disable (design (a)) the operator-witnessed safe-stop HIL validates
    # before real bring-up is authorized. Only meaningful with a controller-launch release.
    safe_stop_disable_cmd: Optional[str] = None
    commissioning_handeye_release: Optional[Path] = None
    commissioning_handeye_artifact_sha256: Optional[str] = None
    commissioning_handeye_complete_sha256: Optional[str] = None
    commissioning_hall_monitor_only: bool = False
    bind: str = "127.0.0.1"
    port: int = 8800
    diagnostic_fps: float = 20.0
    minimum_free_gib: float = 50.0
    maximum_tracking_age_ms: float = 250.0
    capture_queue_frames: int = 8
    camera_delivery_frames: int = 32
    camera_reopen_interval_s: float = 2.0
    audio_device: str = "usb-serial:Y77BQVG17CD6E1"
    audio_sample_rate: int = 48_000
    audio_channels: int = 2
    audio_pick_channel: Optional[int] = 1
    audio_expected_usb_vid: int = 0x1235
    audio_expected_usb_pid: int = 0x8211
    audio_expected_usb_serial: Optional[str] = "Y77BQVG17CD6E1"
    audio_expected_usb_port_path: Optional[str] = None
    hall_expected_usb_vid: int = 0x2E8A
    hall_expected_usb_pid: int = 0x0005
    hall_expected_usb_serial: str = "53032847188a2e1c"
    commissioning_preview_only: bool = False


@dataclass(frozen=True)
class ReleaseBindings:
    spec: ServiceSpec
    code_payload_sha256: str
    environment_lock: Mapping[str, object]
    environment_lock_path: Path
    environment_lock_sha256: str
    environment_python: Path
    ffmpeg_path: Path
    ffmpeg_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    posterior_gate_path: Path
    posterior_gate_sha256: str
    dino_checkpoint_path: Path
    dino_checkpoint_sha256: str
    dino_repository_path: Path
    torch_home: Path
    collector_path: Path
    generator_path: Path
    environment_verifier_path: Path


def replace_spec(spec: ServiceSpec, **changes: object) -> ServiceSpec:
    """Typed public convenience used by deployment callers and tests."""
    return replace(spec, **changes)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise UnitContractError("cannot hash immutable file: %s" % path) from exc
    return digest.hexdigest()


def _canonical_object(path: Path, label: str) -> Tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnitContractError("%s is missing or invalid" % label) from exc
    if not isinstance(value, dict) or raw != contract.canonical_json_bytes(value):
        raise UnitContractError("%s is not canonical JSON" % label)
    return value, raw


def _reject_dynamic(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise UnitContractError("%s must be an absolute immutable path" % label)
    for component in PurePosixPath(str(path)).parts:
        if component.lower() in _DYNAMIC_COMPONENTS:
            raise UnitContractError(
                "%s must use an immutable path, not a dynamic pointer" % label
            )


def _immutable_directory(path: Path, label: str, expected_leaf: str) -> Path:
    raw = Path(path)
    _reject_dynamic(raw, label)
    if raw.name != expected_leaf:
        raise UnitContractError("%s id must equal its path leaf" % label)
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise UnitContractError("%s is missing" % label) from exc
    if resolved != raw or not raw.is_dir() or raw.is_symlink():
        raise UnitContractError("%s must be a real immutable directory" % label)
    return raw


def _regular_no_symlink(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise UnitContractError("%s is missing" % label) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise UnitContractError("%s must be a regular non-symlink file" % label)
    return path


def _bound_environment_python(environment_release: Path) -> Path:
    """Resolve only Conda's exact internal ``bin/python -> python3.11`` alias."""

    bin_dir = environment_release / "bin"
    try:
        bin_info = bin_dir.lstat()
    except OSError as exc:
        raise UnitContractError("environment Python directory is missing") from exc
    if stat.S_ISLNK(bin_info.st_mode) or not stat.S_ISDIR(bin_info.st_mode):
        raise UnitContractError(
            "environment Python directory must be a real directory"
        )
    alias = bin_dir / "python"
    try:
        alias_info = alias.lstat()
    except OSError as exc:
        raise UnitContractError("environment Python alias is missing") from exc
    if not stat.S_ISLNK(alias_info.st_mode):
        raise UnitContractError(
            "environment Python alias must be the internal python3.11 symlink"
        )
    try:
        target = os.readlink(alias)
    except OSError as exc:
        raise UnitContractError("environment Python alias cannot be read") from exc
    if target != "python3.11":
        raise UnitContractError(
            "environment Python alias must target internal python3.11"
        )
    executable = bin_dir / "python3.11"
    _regular_no_symlink(executable, "environment Python executable")
    return alias


def _require_identifier(value: str, pattern: re.Pattern, label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise UnitContractError("%s is invalid" % label)
    return value


def _safe_unit_token(value: object, label: str) -> str:
    text = str(value)
    if (
        not text
        or any(character.isspace() and character != " " for character in text)
        or any(
            character in text
            for character in ('"', "'", "\\", "%", "$", "\x00")
        )
    ):
        raise UnitContractError("%s cannot be represented safely in systemd" % label)
    return '"%s"' % text if " " in text else text


def _stable_json_input(
    path: Path, label: str, expected_sha256: str,
) -> Tuple[dict, bytes]:
    """Read a pinned JSON input once without symlink or dynamic-pointer drift."""

    raw_path = Path(path)
    _reject_dynamic(raw_path, label)
    expected = _require_sha256(expected_sha256, f"{label} SHA-256")
    try:
        info = raw_path.lstat()
        resolved = raw_path.resolve(strict=True)
    except OSError as exc:
        raise UnitContractError(f"{label} is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or resolved != raw_path:
        raise UnitContractError(
            f"{label} must be a real immutable non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(raw_path), flags)
    try:
        before = os.fstat(descriptor)
        chunks: List[bytes] = []
        remaining = before.st_size
        if not 0 < remaining <= 64 * 1024 * 1024:
            raise UnitContractError(f"{label} has an invalid size")
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise UnitContractError(f"{label} was truncated while reading")
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns,
    ) != (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    ):
        raise UnitContractError(f"{label} changed while reading")
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise UnitContractError(f"{label} SHA-256 mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnitContractError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise UnitContractError(f"{label} must be a JSON object")
    return value, raw


def _canonical_newline_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise UnitContractError("object is not finite canonical JSON") from exc


def _verify_commissioning_handeye_binding(
    *,
    release_path: Path,
    artifact_sha256: str,
    complete_sha256: str,
    expected_camera_id: str,
) -> Path:
    """Verify the immutable v2 release topology without constructing runtime objects."""

    artifact_sha256 = _require_sha256(
        artifact_sha256, "commissioning hand-eye artifact SHA-256",
    )
    complete_sha256 = _require_sha256(
        complete_sha256, "commissioning hand-eye COMPLETE SHA-256",
    )
    release_id = "handeye_activation-" + artifact_sha256[:20]
    _require_identifier(release_id, _HANDEYE_ID, "hand-eye release id")
    release = _immutable_directory(
        Path(release_path), "commissioning hand-eye release", release_id,
    )
    try:
        names = {path.name for path in release.iterdir()}
    except OSError as exc:
        raise UnitContractError(
            "commissioning hand-eye release inventory cannot be read"
        ) from exc
    if names != {"COMPLETE", "handeye.COMPLETE", "handeye.json"}:
        raise UnitContractError(
            "commissioning hand-eye release inventory is not exact"
        )
    if stat.S_IMODE(release.stat().st_mode) != 0o555:
        raise UnitContractError(
            "commissioning hand-eye release mode must be 0555"
        )
    for name in names:
        path = release / name
        _regular_no_symlink(path, f"commissioning hand-eye {name}")
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise UnitContractError(
                f"commissioning hand-eye {name} mode must be 0444"
            )
    artifact, artifact_raw = _stable_json_input(
        release / "handeye.json",
        "commissioning hand-eye artifact",
        artifact_sha256,
    )
    if artifact_raw != _canonical_newline_bytes(artifact):
        raise UnitContractError(
            "commissioning hand-eye artifact is not canonical newline JSON"
        )
    if (
        artifact.get("schema") != "wam_openarm_handeye_activation_v2"
        or artifact.get("camera_id") != expected_camera_id
        or artifact.get("calibration_status") != "USER_ACCEPTED_BELOW_GATE"
        or artifact.get("provisional") is not True
        or artifact.get("autonomous_targeting_authorized") is not False
    ):
        raise UnitContractError(
            "commissioning hand-eye artifact semantics are invalid"
        )
    marker, marker_raw = _stable_json_input(
        release / "handeye.COMPLETE",
        "commissioning hand-eye COMPLETE marker",
        complete_sha256,
    )
    if marker_raw != _canonical_newline_bytes(marker) or marker != {
        "artifact_sha256": artifact_sha256,
        "calibration_status": "USER_ACCEPTED_BELOW_GATE",
        "provisional": True,
        "schema": "wam_openarm_handeye_artifact_completion_v2",
        "status": "COMPLETE",
    }:
        raise UnitContractError(
            "commissioning hand-eye COMPLETE marker is invalid"
        )
    release_complete_path = release / "COMPLETE"
    release_complete_sha256 = _sha256_file(release_complete_path)
    release_complete, release_complete_raw = _stable_json_input(
        release_complete_path,
        "commissioning hand-eye release COMPLETE",
        release_complete_sha256,
    )
    expected_inventory = {
        "handeye.COMPLETE": {
            "sha256": complete_sha256,
            "size_bytes": len(marker_raw),
        },
        "handeye.json": {
            "sha256": artifact_sha256,
            "size_bytes": len(artifact_raw),
        },
    }
    if (
        release_complete_raw != _canonical_newline_bytes(release_complete)
        or release_complete != {
            "artifact_sha256": artifact_sha256,
            "inventory": expected_inventory,
            "release_id": release_id,
            "schema": "wam_openarm_handeye_release_completion_v2",
            "status": "COMPLETE",
        }
    ):
        raise UnitContractError(
            "commissioning hand-eye release COMPLETE is invalid"
        )
    return release


def _manifest_row(envelope: Mapping[str, object], relative: str) -> Mapping[str, object]:
    payload = envelope.get("payload")
    rows = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise UnitContractError("code release file manifest is invalid")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("path") == relative]
    if len(matches) != 1:
        raise UnitContractError("code release lacks required file: %s" % relative)
    return matches[0]


def _verify_hall_binding(spec: ServiceSpec) -> dict:
    if _SERIAL_PORT.fullmatch(str(spec.hall_port)) is None:
        raise UnitContractError("Hall serial port is invalid")
    if (
        spec.hall_expected_usb_vid != 0x2E8A
        or spec.hall_expected_usb_pid != 0x0005
        or spec.hall_expected_usb_serial != "53032847188a2e1c"
    ):
        raise UnitContractError(
            "Hall RP2040 USB identity must match commissioned VID/PID/serial"
        )
    hall_payload, _hall_raw = _stable_json_input(
        spec.hall_calibration,
        "Hall-force calibration",
        spec.hall_calibration_sha256,
    )
    if hall_payload.get("calibrated") is not True:
        raise UnitContractError(
            "Hall-force calibration must be physically commissioned calibrated=true"
        )
    return hall_payload


def _verify_dormant_controller_launch_binding(
    spec: ServiceSpec,
) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """Validate only the immutable path pins; execution remains impossible here.

    The collector performs the semantic pair-v2/artifact verification after it
    has loaded the production activation.  This unit generator deliberately
    transports only the three all-or-none pins and never renders controller,
    preflight, stop, or gripper commands into systemd.
    """

    values = (
        spec.controller_launch_release,
        spec.controller_launch_trusted_root,
        spec.controller_launch_contract_sha256,
    )
    count = sum(value not in {None, ""} for value in values)
    if count == 0:
        return None, None, None
    if count != len(values):
        raise UnitContractError(
            "controller launch binding requires release, trusted root, and SHA-256"
        )
    if spec.commissioning_preview_only:
        raise UnitContractError(
            "commissioning preview forbids a controller launch binding"
        )
    digest = _require_sha256(
        str(spec.controller_launch_contract_sha256),
        "controller launch contract SHA-256",
    )
    root = Path(spec.controller_launch_trusted_root)
    release = Path(spec.controller_launch_release)
    _reject_dynamic(root, "controller launch trusted root")
    _reject_dynamic(release, "controller launch release")
    expected_leaf = "controller_launch_v1-" + digest
    if release.name != expected_leaf or _CONTROLLER_LAUNCH_ID.fullmatch(release.name) is None:
        raise UnitContractError(
            "controller launch release id must match the pinned SHA-256"
        )
    try:
        root_info = root.lstat()
        release_info = release.lstat()
        root_resolved = root.resolve(strict=True)
        release_resolved = release.resolve(strict=True)
    except OSError as exc:
        raise UnitContractError("controller launch immutable directory is missing") from exc
    if (
        root.is_symlink()
        or release.is_symlink()
        or not stat.S_ISDIR(root_info.st_mode)
        or not stat.S_ISDIR(release_info.st_mode)
        or root_resolved != root
        or release_resolved != release
        or stat.S_IMODE(root_info.st_mode) & 0o222
        or stat.S_IMODE(release_info.st_mode) & 0o222
    ):
        raise UnitContractError(
            "controller launch root and release must be real immutable directories"
        )
    if release.parent != root:
        raise UnitContractError(
            "controller launch release must be a direct child of its trusted root"
        )
    return release, root, digest


def bind_releases(spec: ServiceSpec) -> ReleaseBindings:
    """Verify and mutually bind all immutable collector inputs."""
    code_id = _require_identifier(spec.code_release_id, _CODE_ID, "code release id")
    runtime_id = _require_identifier(
        spec.runtime_release_id, _RUNTIME_ID, "runtime release id"
    )
    environment_id = _require_identifier(
        spec.environment_id, _ENVIRONMENT_ID, "environment id"
    )
    if _CAMERA_ID.fullmatch(str(spec.camera_id)) is None:
        raise UnitContractError("camera id is invalid")
    if _AUDIO_DEVICE.fullmatch(str(spec.audio_device)) is None:
        raise UnitContractError("audio device is invalid")
    if (
        spec.audio_device != "usb-serial:Y77BQVG17CD6E1"
        or spec.audio_sample_rate != 48_000
        or spec.audio_channels != 2
        or spec.audio_pick_channel != 1
    ):
        raise UnitContractError(
            "Scarlett audio mode must be serial-selected at 48000 Hz, stereo, "
            "channel 1"
        )
    if (
        spec.audio_expected_usb_vid != 0x1235
        or spec.audio_expected_usb_pid != 0x8211
        or spec.audio_expected_usb_serial != "Y77BQVG17CD6E1"
        or spec.audio_expected_usb_port_path is not None
    ):
        raise UnitContractError(
            "Scarlett USB identity must match commissioned VID/PID/serial"
        )
    handeye_values = (
        spec.commissioning_handeye_release,
        spec.commissioning_handeye_artifact_sha256,
        spec.commissioning_handeye_complete_sha256,
    )
    handeye_count = sum(value not in {None, ""} for value in handeye_values)
    (
        controller_launch_release,
        controller_launch_trusted_root,
        controller_launch_contract_sha256,
    ) = _verify_dormant_controller_launch_binding(spec)
    if handeye_count not in {0, len(handeye_values)}:
        raise UnitContractError(
            "commissioning hand-eye binding must provide release and both hashes"
        )
    commissioning_handeye_release = None
    if spec.commissioning_preview_only:
        if spec.production_activation is not None or (
            spec.production_activation_sha256 is not None
        ):
            raise UnitContractError(
                "commissioning preview cannot bind production activation"
            )
        if handeye_count:
            commissioning_handeye_release = _verify_commissioning_handeye_binding(
                release_path=Path(spec.commissioning_handeye_release),
                artifact_sha256=str(
                    spec.commissioning_handeye_artifact_sha256
                ),
                complete_sha256=str(spec.commissioning_handeye_complete_sha256),
                expected_camera_id=spec.camera_id,
            )
        if spec.commissioning_hall_monitor_only:
            _verify_hall_binding(spec)
            if spec.sync_calibration is not None or (
                spec.sync_calibration_sha256 is not None
            ):
                raise UnitContractError(
                    "commissioning Hall monitor must remain unsynchronized"
                )
        elif any(
            value is not None
            for value in (
                spec.hall_calibration,
                spec.hall_calibration_sha256,
                spec.hall_port,
                spec.sync_calibration,
                spec.sync_calibration_sha256,
            )
        ):
            raise UnitContractError(
                "preview Hall inputs require commissioning Hall monitor mode"
            )
    else:
        if spec.commissioning_hall_monitor_only or handeye_count:
            raise UnitContractError(
                "commissioning-only bindings are forbidden in production"
            )
        _verify_hall_binding(spec)
        sync_payload, _sync_raw = _stable_json_input(
            spec.sync_calibration,
            "audio-force sync calibration",
            spec.sync_calibration_sha256,
        )
        if (
            sync_payload.get("schema_version") != "wam_sync_v1"
            or sync_payload.get("measurement_method") != "physical_diaphragm_taps"
            or sync_payload.get("unique_matches") is not True
            or type(sync_payload.get("n_matched")) is not int
            or sync_payload["n_matched"] < 4
        ):
            raise UnitContractError(
                "audio-force sync calibration is not physically commissioned"
            )
    if spec.bind != "127.0.0.1":
        raise UnitContractError("collector bind must be loopback 127.0.0.1")
    if type(spec.port) is not int or not 1 <= spec.port <= 65535:
        raise UnitContractError("collector port is invalid")
    if float(spec.minimum_free_gib) < 50.0:
        raise UnitContractError("minimum free disk reserve must be at least 50 GiB")
    if float(spec.maximum_tracking_age_ms) != 250.0:
        raise UnitContractError("maximum tracking age must be exactly 250 ms")

    code_release = _immutable_directory(spec.code_release, "code release", code_id)
    runtime_release = _immutable_directory(
        spec.runtime_release, "runtime release", runtime_id
    )
    environment_release = _immutable_directory(
        spec.environment_release, "environment release", environment_id
    )
    try:
        code_envelope = contract.verify_release(code_release)
    except contract.ArtifactError as exc:
        raise UnitContractError("code release verification failed") from exc
    if (
        code_envelope.get("release_id") != code_id
        or code_envelope.get("payload", {}).get("artifact_kind") != "code_release"
    ):
        raise UnitContractError("code release id or artifact kind mismatch")
    for relative in (COLLECTOR_RELATIVE, GENERATOR_RELATIVE, ENV_VERIFIER_RELATIVE):
        _manifest_row(code_envelope, relative)

    try:
        runtime_envelope = publisher.verify_perception_runtime_release(runtime_release)
    except (publisher.PerceptionRuntimeError, contract.ArtifactError) as exc:
        raise UnitContractError("runtime release verification failed") from exc
    if runtime_envelope.get("release_id") != runtime_id:
        raise UnitContractError("runtime release id mismatch")
    semantic, _semantic_raw = _canonical_object(
        runtime_release / REQUIRED_RUNTIME_MANIFEST_V2,
        "perception runtime manifest",
    )
    if semantic.get("schema") != REQUIRED_RUNTIME_SCHEMA_V2:
        raise UnitContractError(
            "collector unit schema v2 requires perception runtime schema v5"
        )
    if semantic.get("capabilities") != REQUIRED_RUNTIME_CAPABILITIES_V2:
        raise UnitContractError(
            "perception runtime v5 capability boundary is invalid"
        )
    code_semantic = semantic.get("code")
    environment_semantic = semantic.get("environment")
    if not isinstance(code_semantic, Mapping) or (
        code_semantic.get("release_id") != code_id
        or code_semantic.get("payload_sha256")
        != code_envelope.get("payload_sha256")
    ):
        raise UnitContractError("runtime does not bind the standalone code release")
    if not isinstance(environment_semantic, Mapping) or (
        environment_semantic.get("environment_id") != environment_id
    ):
        raise UnitContractError("runtime does not bind the environment release")

    lock_path = runtime_release / publisher.ENVIRONMENT_LOCK_PATH
    lock, lock_raw = _canonical_object(lock_path, "runtime environment lock")
    try:
        publisher.validate_environment_lock(lock, require_posterior_gate=True)
    except publisher.PerceptionRuntimeError as exc:
        raise UnitContractError("runtime environment lock verification failed") from exc
    lock_sha256 = hashlib.sha256(lock_raw).hexdigest()
    if (
        lock.get("schema") != publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA
        or lock.get("environment_id") != environment_id
        or environment_semantic.get("lock_sha256") != lock_sha256
        or environment_semantic.get("lock_size") != len(lock_raw)
    ):
        raise UnitContractError("environment release does not match runtime v3 lock")

    environment_python = _bound_environment_python(environment_release)
    if _sha256_file(environment_python) != lock["python"]["executable_sha256"]:
        raise UnitContractError("environment Python hash mismatch")
    ffmpeg_path = _regular_no_symlink(
        environment_release / str(lock["ffmpeg"]["path"]), "environment ffmpeg"
    )
    if (
        _sha256_file(ffmpeg_path) != lock["ffmpeg"]["sha256"]
        or ffmpeg_path.stat().st_size != lock["ffmpeg"]["size"]
    ):
        raise UnitContractError("environment ffmpeg hash or size mismatch")

    sessions_root = Path(spec.sessions_root)
    _reject_dynamic(sessions_root, "sessions root")
    if not sessions_root.is_dir() or sessions_root.is_symlink():
        raise UnitContractError("sessions root must be a real directory")
    unit_path = Path(spec.unit_path)
    _reject_dynamic(unit_path, "unit path")
    verifier_python = Path(spec.verifier_python)
    _reject_dynamic(verifier_python, "verifier Python")
    if verifier_python != environment_python:
        try:
            _regular_no_symlink(verifier_python, "verifier Python")
        except UnitContractError as exc:
            raise UnitContractError(
                "verifier Python must be the exact verified internal alias"
            ) from exc
        raise UnitContractError(
            "verifier Python must belong to the bound environment release"
        )
    production_activation = None
    production_activation_sha256 = None
    if not spec.commissioning_preview_only:
        activation_report = verify_production_activation(
            spec.production_activation,
            spec.production_activation_sha256,
        )
        production_activation = Path(activation_report["production_activation"])
        production_activation_sha256 = str(
            activation_report["production_activation_sha256"]
        )

    model = semantic["model"]
    gate = semantic["posterior_gate"]
    dino = semantic["dino"]
    paths = (
        code_release,
        runtime_release,
        environment_release,
        sessions_root,
        unit_path,
        verifier_python,
        *(() if production_activation is None else (production_activation,)),
        *(
            ()
            if commissioning_handeye_release is None
            else (commissioning_handeye_release,)
        ),
        *(() if spec.hall_calibration is None else (spec.hall_calibration,)),
        *(() if spec.sync_calibration is None else (spec.sync_calibration,)),
        *(
            ()
            if controller_launch_release is None
            else (controller_launch_release, controller_launch_trusted_root)
        ),
    )
    for index, path in enumerate(paths):
        _safe_unit_token(path, "unit path %d" % index)
    _safe_unit_token(spec.camera_id, "camera id")

    return ReleaseBindings(
        spec=replace(
            spec,
            code_release=code_release,
            runtime_release=runtime_release,
            environment_release=environment_release,
            sessions_root=sessions_root,
            unit_path=unit_path,
            verifier_python=verifier_python,
            production_activation=production_activation,
            production_activation_sha256=production_activation_sha256,
            commissioning_handeye_release=commissioning_handeye_release,
            hall_calibration=(
                None if spec.hall_calibration is None else Path(spec.hall_calibration)
            ),
            sync_calibration=(
                None if spec.sync_calibration is None else Path(spec.sync_calibration)
            ),
            controller_launch_release=controller_launch_release,
            controller_launch_trusted_root=controller_launch_trusted_root,
            controller_launch_contract_sha256=(
                controller_launch_contract_sha256
            ),
            safe_stop_disable_cmd=spec.safe_stop_disable_cmd,
        ),
        code_payload_sha256=str(code_envelope["payload_sha256"]),
        environment_lock=lock,
        environment_lock_path=lock_path,
        environment_lock_sha256=lock_sha256,
        environment_python=environment_python,
        ffmpeg_path=ffmpeg_path,
        ffmpeg_sha256=str(lock["ffmpeg"]["sha256"]),
        checkpoint_path=runtime_release / str(model["checkpoint_path"]),
        checkpoint_sha256=str(model["checkpoint_sha256"]),
        posterior_gate_path=runtime_release / str(gate["checkpoint_path"]),
        posterior_gate_sha256=str(gate["checkpoint_sha256"]),
        dino_checkpoint_path=runtime_release / str(dino["checkpoint_path"]),
        dino_checkpoint_sha256=str(dino["checkpoint_sha256"]),
        dino_repository_path=runtime_release / str(dino["repository_path"]),
        torch_home=runtime_release / publisher.TORCH_HOME_PATH,
        collector_path=code_release / COLLECTOR_RELATIVE,
        generator_path=code_release / GENERATOR_RELATIVE,
        environment_verifier_path=code_release / ENV_VERIFIER_RELATIVE,
    )


def collector_argv(bindings: ReleaseBindings) -> List[str]:
    """Return the exact argv used by ``ExecStart``."""
    spec = bindings.spec
    argv = [
        str(bindings.environment_python),
        "-B",
        str(bindings.collector_path),
        "--camera-backend", "orbbec",
        "--camera-index", "0",
        "--camera-id", str(spec.camera_id),
        "--camera-width", "1920",
        "--camera-height", "1080",
        "--camera-fps", "30",
        "--camera-color-format", "MJPG",
        "--depth-width", "640",
        "--depth-height", "576",
        "--depth-fps", "30",
        "--depth-format", "Y16",
        "--out", str(spec.sessions_root),
        "--checkpoint", str(bindings.checkpoint_path),
        "--checkpoint-sha256", bindings.checkpoint_sha256,
        "--base-model", str(bindings.dino_checkpoint_path),
        "--base-model-sha256", bindings.dino_checkpoint_sha256,
        "--model-release-id", "perc_lmh_v9_occ-" + bindings.checkpoint_sha256[:16],
        "--posterior-gate-checkpoint", str(bindings.posterior_gate_path),
        "--posterior-gate-checkpoint-sha256", bindings.posterior_gate_sha256,
        "--posterior-gate-release-id", "yolo11x_pose-" + bindings.posterior_gate_sha256[:16],
        "--runtime-release-id", spec.runtime_release_id,
        "--runtime-lock", str(bindings.environment_lock_path),
        "--runtime-lock-sha256", bindings.environment_lock_sha256,
        "--code-release-id", spec.code_release_id,
        "--ffmpeg-path", str(bindings.ffmpeg_path),
        "--ffmpeg-sha256", bindings.ffmpeg_sha256,
        "--diagnostic-fps", _number(spec.diagnostic_fps),
        "--diagnostic-min-free-gib", _number(spec.minimum_free_gib),
        "--max-tracking-age-ms", _number(spec.maximum_tracking_age_ms),
        "--capture-queue-frames", str(spec.capture_queue_frames),
        "--camera-delivery-frames", str(spec.camera_delivery_frames),
        "--camera-reopen-interval-s", _number(spec.camera_reopen_interval_s),
        "--audio-device", str(spec.audio_device),
        "--audio-sample-rate", str(spec.audio_sample_rate),
        "--audio-channels", str(spec.audio_channels),
        "--audio-expected-usb-vid", str(spec.audio_expected_usb_vid),
        "--audio-expected-usb-pid", str(spec.audio_expected_usb_pid),
        "--require-audio",
        "--bind", "127.0.0.1",
        "--port", str(spec.port),
    ]
    if spec.commissioning_preview_only:
        argv.append("--commissioning-preview-only")
        if spec.commissioning_handeye_release is not None:
            argv.extend([
                "--commissioning-handeye-release",
                str(spec.commissioning_handeye_release),
                "--commissioning-handeye-artifact-sha256",
                str(spec.commissioning_handeye_artifact_sha256),
                "--commissioning-handeye-complete-sha256",
                str(spec.commissioning_handeye_complete_sha256),
            ])
        if spec.commissioning_hall_monitor_only:
            argv.extend([
                "--commissioning-hall-monitor-only",
                "--hall-calibration", str(spec.hall_calibration),
                "--hall-calibration-sha256",
                str(spec.hall_calibration_sha256),
                "--hall-port", str(spec.hall_port),
                "--hall-expected-usb-vid",
                str(spec.hall_expected_usb_vid),
                "--hall-expected-usb-pid",
                str(spec.hall_expected_usb_pid),
                "--hall-expected-usb-serial",
                spec.hall_expected_usb_serial,
            ])
    else:
        argv.extend([
            "--production-teleop",
            "--production-activation", str(spec.production_activation),
            "--production-activation-sha256", str(
                spec.production_activation_sha256
            ),
            "--hall-calibration", str(spec.hall_calibration),
            "--hall-calibration-sha256", str(spec.hall_calibration_sha256),
            "--hall-port", str(spec.hall_port),
            "--hall-expected-usb-vid", str(spec.hall_expected_usb_vid),
            "--hall-expected-usb-pid", str(spec.hall_expected_usb_pid),
            "--hall-expected-usb-serial", spec.hall_expected_usb_serial,
            "--sync-calibration", str(spec.sync_calibration),
            "--sync-calibration-sha256", str(spec.sync_calibration_sha256),
            "--require-force",
        ])
        if spec.controller_launch_release is not None:
            argv.extend([
                "--controller-launch-release",
                str(spec.controller_launch_release),
                "--controller-launch-trusted-root",
                str(spec.controller_launch_trusted_root),
                "--controller-launch-contract-sha256",
                str(spec.controller_launch_contract_sha256),
            ])
            if spec.safe_stop_disable_cmd is not None:
                argv.extend([
                    "--safe-stop-disable-cmd", str(spec.safe_stop_disable_cmd),
                ])
    if spec.audio_pick_channel is not None:
        insert_at = argv.index("--require-audio")
        argv[insert_at:insert_at] = [
            "--audio-pick-channel", str(spec.audio_pick_channel),
        ]
    insert_at = argv.index("--require-audio")
    if spec.audio_expected_usb_serial is not None:
        argv[insert_at:insert_at] = [
            "--audio-expected-usb-serial", spec.audio_expected_usb_serial,
        ]
        insert_at = argv.index("--require-audio")
    if spec.audio_expected_usb_port_path is not None:
        argv[insert_at:insert_at] = [
            "--audio-expected-usb-port-path",
            spec.audio_expected_usb_port_path,
        ]
    return argv


def _number(value: object) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, ".15g")


def preflight_commands(bindings: ReleaseBindings) -> Tuple[Tuple[str, ...], ...]:
    spec = bindings.spec
    code = (
        str(spec.verifier_python), "-B", str(bindings.generator_path),
        "verify-code",
        "--code-release", str(spec.code_release),
        "--code-release-id", spec.code_release_id,
        "--code-payload-sha256", bindings.code_payload_sha256,
    )
    runtime = (
        str(spec.verifier_python), "-B", str(bindings.generator_path),
        "verify-runtime",
        "--runtime-release", str(spec.runtime_release),
        "--runtime-release-id", spec.runtime_release_id,
        "--code-release-id", spec.code_release_id,
        "--code-payload-sha256", bindings.code_payload_sha256,
        "--environment-id", spec.environment_id,
        "--environment-lock-sha256", bindings.environment_lock_sha256,
    )
    activation = (
        str(spec.verifier_python), "-B", str(bindings.generator_path),
        "verify-activation",
        "--production-activation", str(spec.production_activation),
        "--production-activation-sha256", spec.production_activation_sha256,
    )
    environment = (
        str(bindings.environment_python), "-B", str(bindings.environment_verifier_path),
        "--lock", str(bindings.environment_lock_path),
        "--environment-root", str(spec.environment_release),
    )
    force = (
        str(bindings.environment_python), "-B", str(bindings.generator_path),
        "verify-force",
        "--hall-calibration", str(spec.hall_calibration),
        "--hall-calibration-sha256", spec.hall_calibration_sha256,
        "--sync-calibration", str(spec.sync_calibration),
        "--sync-calibration-sha256", spec.sync_calibration_sha256,
    )
    if spec.commissioning_preview_only:
        commands: List[Tuple[str, ...]] = [code, runtime, environment]
        if spec.commissioning_handeye_release is not None:
            commands.append((
                str(bindings.environment_python), "-B",
                str(bindings.generator_path),
                "verify-commissioning-handeye",
                "--commissioning-handeye-release",
                str(spec.commissioning_handeye_release),
                "--commissioning-handeye-artifact-sha256",
                str(spec.commissioning_handeye_artifact_sha256),
                "--commissioning-handeye-complete-sha256",
                str(spec.commissioning_handeye_complete_sha256),
                "--camera-id", spec.camera_id,
            ))
        if spec.commissioning_hall_monitor_only:
            commands.append((
                str(bindings.environment_python), "-B",
                str(bindings.generator_path),
                "verify-hall-monitor",
                "--hall-calibration", str(spec.hall_calibration),
                "--hall-calibration-sha256",
                str(spec.hall_calibration_sha256),
            ))
        return tuple(commands)
    return code, runtime, activation, environment, force


def _unit_line(name: str, argv: Sequence[str]) -> str:
    return name + "=" + " ".join(
        _safe_unit_token(item, "%s argument" % name) for item in argv
    )


def render_unit(bindings: ReleaseBindings) -> bytes:
    """Render deterministic UTF-8 unit bytes from verified bindings."""
    spec = bindings.spec
    environment = {
        "AUSC_DINOV2_CHECKPOINT": str(bindings.dino_checkpoint_path),
        "AUSC_DINOV2_REPO": str(bindings.dino_repository_path),
        "AUSC_OFFLINE_RUNTIME": "1",
        "AUSC_POSTERIOR_GATE_CHECKPOINT": str(bindings.posterior_gate_path),
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(spec.code_release / "auscultation"),
        "PYTHONUNBUFFERED": "1",
        "TORCH_HOME": str(bindings.torch_home),
        "TRANSFORMERS_OFFLINE": "1",
        "WAM_HOST_ROLE": "linbox",
    }
    lines = [
        "# Generated by %s; release IDs are immutable inputs." % UNIT_SCHEMA,
        "# code=%s" % spec.code_release_id,
        "# environment=%s" % spec.environment_id,
        "# runtime=%s" % spec.runtime_release_id,
        "[Unit]",
        "Description=WAM Linbox immutable Orbbec RGB-D collector",
        "Wants=network-online.target",
        "After=network-online.target",
        "StartLimitIntervalSec=0",
        "",
        "[Service]",
        "Type=simple",
        "WorkingDirectory=%s" % _safe_unit_token(
            spec.code_release / "auscultation", "working directory"
        ),
    ]
    lines.extend(
        "Environment=%s=%s" % (key, _safe_unit_token(value, "environment %s" % key))
        for key, value in sorted(environment.items())
    )
    lines.extend(_unit_line("ExecStartPre", command) for command in preflight_commands(bindings))
    lines.extend(
        [
            _unit_line("ExecStart", collector_argv(bindings)),
            (
                "Restart=always"
                if spec.commissioning_preview_only else "Restart=on-failure"
            ),
            "RestartSec=2s",
            "TimeoutStartSec=600s",
            "TimeoutStopSec=3600s",
            "KillMode=mixed",
            "NoNewPrivileges=true",
            "UMask=0077",
            "StandardOutput=journal",
            "StandardError=journal",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def load_module_from_release_source(path: Path, name: str):
    """Load a Python source file for tests that audit a real argparse contract."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise UnitContractError("cannot load Python source: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise UnitContractError("%s must be a lowercase SHA-256" % label)
    return value


def _stable_canonical_object(
    path: Path,
    label: str,
    *,
    expected_sha256: Optional[str] = None,
) -> Tuple[dict, bytes]:
    """Read one immutable JSON file through a stable non-following descriptor."""

    raw_path = Path(path)
    _reject_dynamic(raw_path, label)
    try:
        info = raw_path.lstat()
        resolved = raw_path.resolve(strict=True)
    except OSError as exc:
        raise UnitContractError("%s is missing" % label) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or resolved != raw_path
    ):
        raise UnitContractError(
            "%s must be a real immutable non-symlink file" % label
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(raw_path), flags)
    except OSError as exc:
        raise UnitContractError("%s cannot be opened" % label) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 64 * 1024 * 1024:
            raise UnitContractError("%s has an invalid size" % label)
        chunks: List[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise UnitContractError("%s was truncated while reading" % label)
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise UnitContractError("%s changed while reading" % label)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        expected_sha256 is not None
        and hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise UnitContractError("%s SHA-256 mismatch" % label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnitContractError("%s is missing or invalid" % label) from exc
    try:
        canonical = contract.canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise UnitContractError("%s is not canonical JSON" % label) from exc
    if not isinstance(value, dict) or raw != canonical:
        raise UnitContractError("%s is not canonical JSON" % label)
    return value, raw


def verify_production_activation(
    production_activation: Path,
    production_activation_sha256: str,
) -> dict:
    """Boot-time gate for the exact completed left-arm activation pointer.

    Full per-artifact construction remains in ``collect_server``.  This small
    standard-library preflight prevents systemd from starting it with a
    missing, staged, unpinned, or wrong-side top-level activation.
    """

    expected_sha256 = _require_sha256(
        production_activation_sha256, "production activation SHA-256"
    )
    path = Path(production_activation)
    if path.name != "activation.json":
        raise UnitContractError(
            "production activation must be the canonical activation.json"
        )
    if os.path.lexists(path.parent / "STAGING"):
        raise UnitContractError("production activation STAGING marker is present")
    manifest, raw = _stable_canonical_object(
        path,
        "production activation",
        expected_sha256=expected_sha256,
    )
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise UnitContractError("production activation SHA-256 mismatch")
    expected_keys = {
        "artifacts",
        "follower_can_interface",
        "leader_can_interface",
        "openarm_pair",
        "robot_state_bounds",
        "schema",
        "side",
        "status",
    }
    if set(manifest) != expected_keys:
        raise UnitContractError("production activation fields are not exact")
    if (
        manifest.get("schema") != PRODUCTION_ACTIVATION_SCHEMA
        or manifest.get("status") != "COMPLETE"
        or manifest.get("side") != "left"
        or manifest.get("leader_can_interface") != "can1"
        or manifest.get("follower_can_interface") != "can3"
    ):
        raise UnitContractError(
            "production activation must be COMPLETE left can1 leader -> can3 follower"
        )
    if not all(
        isinstance(manifest.get(name), dict)
        for name in ("artifacts", "openarm_pair", "robot_state_bounds")
    ):
        raise UnitContractError("production activation object fields are invalid")
    complete, _complete_raw = _stable_canonical_object(
        path.parent / "COMPLETE", "production activation COMPLETE marker"
    )
    if complete != {
        "activation_sha256": expected_sha256,
        "schema": PRODUCTION_ACTIVATION_COMPLETION_SCHEMA,
        "status": "COMPLETE",
    }:
        raise UnitContractError(
            "production activation COMPLETE marker does not bind activation SHA-256"
        )
    return {
        "follower_can_interface": "can3",
        "leader_can_interface": "can1",
        "production_activation": str(path),
        "production_activation_sha256": expected_sha256,
        "schema": VERIFY_ACTIVATION_SCHEMA,
        "side": "left",
        "status": "VERIFIED",
    }


def verify_force_configuration(
    *, hall_calibration: Path, hall_calibration_sha256: str,
    sync_calibration: Path, sync_calibration_sha256: str,
) -> dict:
    """Boot-time pyserial and immutable physical-calibration preflight."""

    try:
        import serial  # noqa: F401
        from serial.tools import list_ports  # noqa: F401
    except ImportError as exc:
        raise UnitContractError(
            "Linbox collector environment is missing pyserial"
        ) from exc
    hall, _hall_raw = _stable_json_input(
        hall_calibration, "Hall-force calibration", hall_calibration_sha256,
    )
    if hall.get("calibrated") is not True:
        raise UnitContractError(
            "Hall-force calibration must be physically commissioned calibrated=true"
        )
    sync, _sync_raw = _stable_json_input(
        sync_calibration, "audio-force sync calibration", sync_calibration_sha256,
    )
    if (
        sync.get("schema_version") != "wam_sync_v1"
        or sync.get("measurement_method") != "physical_diaphragm_taps"
        or sync.get("unique_matches") is not True
        or type(sync.get("n_matched")) is not int
        or sync["n_matched"] < 4
    ):
        raise UnitContractError(
            "audio-force sync calibration is not physically commissioned"
        )
    return {
        "hall_calibration": str(hall_calibration),
        "hall_calibration_sha256": hall_calibration_sha256,
        "pyserial": "importable",
        "schema": "wam_linbox_collector_force_preflight_v1",
        "status": "VERIFIED",
        "sync_calibration": str(sync_calibration),
        "sync_calibration_sha256": sync_calibration_sha256,
    }


def verify_hall_monitor_configuration(
    *, hall_calibration: Path, hall_calibration_sha256: str,
) -> dict:
    """Verify a calibrated Hall input without claiming cross-modal timing."""

    try:
        import serial  # noqa: F401
        from mechanical_sensor.force_feedback.force_sensor import ForceCalibration
    except ImportError as exc:
        raise UnitContractError(
            "Linbox collector environment lacks Hall monitor dependencies"
        ) from exc
    hall, hall_raw = _stable_json_input(
        hall_calibration, "Hall-force calibration", hall_calibration_sha256,
    )
    try:
        calibration = ForceCalibration.from_bytes(hall_raw)
    except (TypeError, ValueError) as exc:
        raise UnitContractError(
            "Hall-force calibration semantics are invalid"
        ) from exc
    if hall.get("calibrated") is not True or calibration.calibrated is not True:
        raise UnitContractError(
            "Hall-force calibration must be physically commissioned calibrated=true"
        )
    return {
        "hall_calibration": str(hall_calibration),
        "hall_calibration_sha256": hall_calibration_sha256,
        "hard_limit_force_n": float(calibration.hard_limit_force_n),
        "pyserial": "importable",
        "schema": VERIFY_HALL_MONITOR_SCHEMA,
        "status": "VERIFIED",
        "synchronized": False,
        "training_eligible": False,
    }


def verify_commissioning_handeye_release(
    *,
    release_path: Path,
    artifact_sha256: str,
    complete_sha256: str,
    camera_id: str,
) -> dict:
    """Re-run the semantic v2 hand-eye loader at every service start."""

    release = _verify_commissioning_handeye_binding(
        release_path=release_path,
        artifact_sha256=artifact_sha256,
        complete_sha256=complete_sha256,
        expected_camera_id=camera_id,
    )
    try:
        from ausc.perception.production_runtime_activation import (
            ProductionRuntimeActivationError,
            load_completed_handeye_release,
        )
    except ImportError as exc:
        raise UnitContractError(
            "commissioning hand-eye loader is unavailable"
        ) from exc
    try:
        handeye = load_completed_handeye_release(
            release,
            expected_artifact_sha256=artifact_sha256,
            expected_complete_sha256=complete_sha256,
            expected_camera_id=camera_id,
        )
    except ProductionRuntimeActivationError as exc:
        raise UnitContractError(
            "commissioning hand-eye semantic verification failed"
        ) from exc
    return {
        "artifact_sha256": artifact_sha256,
        "autonomous_targeting_authorized": False,
        "camera_id": camera_id,
        "complete_sha256": complete_sha256,
        "release_id": handeye.handeye_release_id,
        "release_path": str(release),
        "schema": VERIFY_COMMISSIONING_HANDEYE_SCHEMA,
        "status": "VERIFIED",
    }


def verify_code_release(
    code_release: Path, code_release_id: str, code_payload_sha256: str
) -> dict:
    """Boot-time verification for the standalone immutable code artifact."""
    release_id = _require_identifier(code_release_id, _CODE_ID, "code release id")
    expected_payload = _require_sha256(
        code_payload_sha256, "code payload SHA-256"
    )
    release = _immutable_directory(code_release, "code release", release_id)
    try:
        envelope = contract.verify_release(release)
    except contract.ArtifactError as exc:
        raise UnitContractError("code release verification failed") from exc
    if (
        envelope.get("release_id") != release_id
        or envelope.get("payload_sha256") != expected_payload
        or envelope.get("payload", {}).get("artifact_kind") != "code_release"
    ):
        raise UnitContractError("code release id or payload identity mismatch")
    for relative in (COLLECTOR_RELATIVE, GENERATOR_RELATIVE, ENV_VERIFIER_RELATIVE):
        _manifest_row(envelope, relative)
    return {
        "code_payload_sha256": expected_payload,
        "code_release": str(release),
        "code_release_id": release_id,
        "schema": VERIFY_CODE_SCHEMA,
        "status": "VERIFIED",
    }


def verify_runtime_release(
    *,
    runtime_release: Path,
    runtime_release_id: str,
    code_release_id: str,
    code_payload_sha256: str,
    environment_id: str,
    environment_lock_sha256: str,
) -> dict:
    """Boot-time verification for runtime bytes and code/environment bindings."""
    expected_runtime = _require_identifier(
        runtime_release_id, _RUNTIME_ID, "runtime release id"
    )
    expected_code = _require_identifier(
        code_release_id, _CODE_ID, "code release id"
    )
    expected_environment = _require_identifier(
        environment_id, _ENVIRONMENT_ID, "environment id"
    )
    expected_payload = _require_sha256(
        code_payload_sha256, "code payload SHA-256"
    )
    expected_lock = _require_sha256(
        environment_lock_sha256, "environment lock SHA-256"
    )
    release = _immutable_directory(
        runtime_release, "runtime release", expected_runtime
    )
    try:
        envelope = publisher.verify_perception_runtime_release(release)
    except (publisher.PerceptionRuntimeError, contract.ArtifactError) as exc:
        raise UnitContractError("runtime release verification failed") from exc
    if envelope.get("release_id") != expected_runtime:
        raise UnitContractError("runtime release id mismatch")
    # This boot preflight is an independent local trust boundary.  The generic
    # publisher verifier intentionally supports older runtime schemas for
    # audit/rollback tooling, so accepting its verdict alone would let a
    # self-consistent v4 (or publisher-constant reinterpretation) satisfy a v2
    # collector unit.  Pin both the filename and the exact dormant v5
    # capability contract here, just as bind_releases() does at generation time.
    semantic, _semantic_raw = _canonical_object(
        release / REQUIRED_RUNTIME_MANIFEST_V2,
        "perception runtime manifest",
    )
    if semantic.get("schema") != REQUIRED_RUNTIME_SCHEMA_V2:
        raise UnitContractError(
            "collector unit schema v2 requires perception runtime schema v5"
        )
    if semantic.get("capabilities") != REQUIRED_RUNTIME_CAPABILITIES_V2:
        raise UnitContractError(
            "perception runtime v5 capability boundary is invalid"
        )
    code = semantic.get("code")
    environment = semantic.get("environment")
    if not isinstance(code, Mapping) or (
        code.get("release_id") != expected_code
        or code.get("payload_sha256") != expected_payload
    ):
        raise UnitContractError("runtime code payload binding mismatch")
    if not isinstance(environment, Mapping) or (
        environment.get("environment_id") != expected_environment
        or environment.get("lock_sha256") != expected_lock
    ):
        raise UnitContractError("runtime environment binding mismatch")
    lock, raw = _canonical_object(
        release / publisher.ENVIRONMENT_LOCK_PATH, "runtime environment lock"
    )
    try:
        publisher.validate_environment_lock(lock, require_posterior_gate=True)
    except publisher.PerceptionRuntimeError as exc:
        raise UnitContractError("runtime environment lock is invalid") from exc
    if (
        lock.get("schema") != publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA
        or lock.get("environment_id") != expected_environment
        or hashlib.sha256(raw).hexdigest() != expected_lock
        or environment.get("lock_size") != len(raw)
    ):
        raise UnitContractError("runtime environment lock identity mismatch")
    return {
        "code_payload_sha256": expected_payload,
        "code_release_id": expected_code,
        "environment_id": expected_environment,
        "environment_lock_sha256": expected_lock,
        "runtime_release": str(release),
        "runtime_release_id": expected_runtime,
        "schema": VERIFY_RUNTIME_SCHEMA,
        "status": "VERIFIED",
    }


def _decode_verified_report(raw: object, label: str) -> dict:
    if isinstance(raw, str):
        data = raw.encode("utf-8")
    elif isinstance(raw, bytes):
        data = raw
    else:
        raise UnitContractError("%s preflight returned non-byte output" % label)
    try:
        report = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnitContractError("%s preflight output is not JSON" % label) from exc
    if (
        not isinstance(report, dict)
        or data != contract.canonical_json_bytes(report)
        or report.get("status") != "VERIFIED"
    ):
        raise UnitContractError("%s preflight did not return canonical VERIFIED output" % label)
    return report


def run_preflights(
    bindings: ReleaseBindings,
    *,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> List[dict]:
    """Execute the same five fail-closed gates used by ``ExecStartPre``."""
    execute = subprocess.run if runner is None else runner
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    reports: List[dict] = []
    if bindings.spec.commissioning_preview_only:
        labels = ["code", "runtime", "environment"]
        if bindings.spec.commissioning_handeye_release is not None:
            labels.append("commissioning hand-eye")
        if bindings.spec.commissioning_hall_monitor_only:
            labels.append("Hall monitor")
    else:
        labels = ["code", "runtime", "activation", "environment", "force"]
    for index, command in enumerate(preflight_commands(bindings)):
        try:
            completed = execute(
                list(command),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
        except OSError as exc:
            raise UnitContractError(
                "%s preflight could not execute" % labels[index]
            ) from exc
        if completed.returncode != 0:
            raise UnitContractError(
                "%s preflight failed with exit code %d"
                % (labels[index], completed.returncode)
            )
        reports.append(
            _decode_verified_report(completed.stdout, labels[index])
        )
    return reports


def unit_lock_path(unit_path: Path) -> Path:
    path = Path(unit_path)
    return path.with_name(".%s.lock" % path.name)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _existing_unit_sha256(path: Path) -> str:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise UnitContractError("installed unit is missing") from exc
    if stat.S_ISLNK(info.st_mode):
        raise UnitContractError("installed unit must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise UnitContractError("installed unit must be a regular file")
    return _sha256_file(path)


def _acquire_install_lock(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except FileExistsError as exc:
        raise UnitContractError("collector unit install lock is already held") from exc
    os.fchmod(descriptor, 0o600)
    os.fsync(descriptor)
    return descriptor


def install_unit(
    bindings: ReleaseBindings,
    *,
    expect_no_existing: bool = False,
    expected_unit_sha256: Optional[str] = None,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> dict:
    """Preflight, compare-and-swap, and atomically install one local unit."""
    if bool(expect_no_existing) == (expected_unit_sha256 is not None):
        raise UnitContractError(
            "install requires exactly one of expect-no-existing or expected unit SHA-256"
        )
    if expected_unit_sha256 is not None:
        expected_unit_sha256 = _require_sha256(
            expected_unit_sha256, "expected unit SHA-256"
        )

    # Nothing below this line may mutate the unit filesystem until all five
    # expensive/live gates have succeeded.
    preflights = run_preflights(bindings, runner=runner)
    desired = render_unit(bindings)
    desired_sha256 = hashlib.sha256(desired).hexdigest()
    unit = bindings.spec.unit_path
    parent = unit.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise UnitContractError("unit parent must be a real absolute directory")

    lock = unit_lock_path(unit)
    lock_descriptor = _acquire_install_lock(lock)
    temporary = parent / (".%s.%s.tmp" % (unit.name, uuid.uuid4().hex))
    temporary_descriptor: Optional[int] = None
    try:
        exists = os.path.lexists(str(unit))
        if expect_no_existing:
            if exists:
                raise UnitContractError(
                    "compare-and-swap expected no existing unit"
                )
        else:
            if not exists or _existing_unit_sha256(unit) != expected_unit_sha256:
                raise UnitContractError(
                    "compare-and-swap existing unit SHA-256 mismatch"
                )

        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        temporary_descriptor = os.open(str(temporary), flags, 0o600)
        with os.fdopen(temporary_descriptor, "wb", closefd=True) as handle:
            temporary_descriptor = None
            handle.write(desired)
            os.fchmod(handle.fileno(), 0o644)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(unit))
        _fsync_directory(parent)
        if _existing_unit_sha256(unit) != desired_sha256:
            raise UnitContractError("installed unit failed post-replace verification")
        if stat.S_IMODE(unit.stat().st_mode) != 0o644:
            raise UnitContractError("installed unit mode is not 0644")
        return {
            "code_release_id": bindings.spec.code_release_id,
            "environment_id": bindings.spec.environment_id,
            "preflights": len(preflights),
            "runtime_release_id": bindings.spec.runtime_release_id,
            "schema": UNIT_SCHEMA,
            "status": "INSTALLED",
            "unit_path": str(unit),
            "unit_sha256": desired_sha256,
        }
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        os.close(lock_descriptor)
        try:
            lock.unlink()
        finally:
            _fsync_directory(parent)


def check_unit(
    bindings: ReleaseBindings,
    *,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> dict:
    """Rerun every gate and require exact installed bytes and mode."""
    reports = run_preflights(bindings, runner=runner)
    unit = bindings.spec.unit_path
    actual_sha256 = _existing_unit_sha256(unit)
    desired = render_unit(bindings)
    desired_sha256 = hashlib.sha256(desired).hexdigest()
    if actual_sha256 != desired_sha256 or unit.read_bytes() != desired:
        raise UnitContractError("installed unit drift detected")
    if stat.S_IMODE(unit.stat().st_mode) != 0o644:
        raise UnitContractError("installed unit mode drift detected")
    return {
        "preflights": len(reports),
        "schema": UNIT_SCHEMA,
        "status": "VERIFIED",
        "unit_path": str(unit),
        "unit_sha256": desired_sha256,
    }


def _run_read_only_command(
    command: Sequence[str],
    *,
    label: str,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> bytes:
    execute = subprocess.run if runner is None else runner
    try:
        completed = execute(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise UnitContractError("%s check could not execute" % label) from exc
    if completed.returncode != 0:
        raise UnitContractError(
            "%s check failed with exit code %d" % (label, completed.returncode)
        )
    raw = completed.stdout
    if isinstance(raw, str):
        return raw.encode("utf-8")
    if not isinstance(raw, bytes):
        raise UnitContractError("%s check returned non-byte output" % label)
    return raw


def verify_user_linger(
    user: str,
    *,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> dict:
    """Read and fail closed on the local systemd-logind linger state."""
    if (
        not isinstance(user, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", user) is None
    ):
        raise UnitContractError("linger user name is invalid")
    command = (
        "loginctl",
        "show-user",
        user,
        "--property=Linger",
        "--value",
    )
    raw = _run_read_only_command(command, label="linger", runner=runner)
    if raw != b"yes\n":
        raise UnitContractError("user linger is not enabled")
    return {
        "schema": UNIT_SCHEMA,
        "status": "VERIFIED",
        "user": user,
        "user_linger": True,
    }


def verify_systemd_handoff(
    bindings: ReleaseBindings,
    *,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    proc_root: Path = Path("/proc"),
) -> dict:
    """Read systemd state and prove MainPID runs the exact immutable argv."""
    path = bindings.spec.unit_path
    _reject_dynamic(path, "unit path")
    unit_name = path.name
    if re.fullmatch(r"[A-Za-z0-9_.@:-]+\.service", unit_name) is None:
        raise UnitContractError("systemd unit name is invalid")
    properties = (
        "LoadState",
        "ActiveState",
        "SubState",
        "UnitFileState",
        "NeedDaemonReload",
        "MainPID",
        "FragmentPath",
    )
    command = ["systemctl", "--user", "show", unit_name, "--no-pager"]
    command.extend("--property=%s" % name for name in properties)
    raw = _run_read_only_command(command, label="systemd handoff", runner=runner)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnitContractError("systemd handoff output is not UTF-8") from exc
    observed: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            raise UnitContractError("systemd handoff output is malformed")
        key, value = line.split("=", 1)
        if key not in properties or key in observed:
            raise UnitContractError("systemd handoff property set is invalid")
        observed[key] = value
    expected = {
        "ActiveState": "active",
        "FragmentPath": str(path),
        "LoadState": "loaded",
        "NeedDaemonReload": "no",
        "SubState": "running",
        "UnitFileState": "enabled",
    }
    if set(observed) != set(expected) | {"MainPID"} or any(
        observed.get(key) != value for key, value in expected.items()
    ):
        raise UnitContractError(
            "production handoff requires daemon-reload, enable, and active running service"
        )
    try:
        main_pid = int(observed["MainPID"], 10)
    except (KeyError, TypeError, ValueError) as exc:
        raise UnitContractError("systemd handoff MainPID is invalid") from exc
    if main_pid <= 0 or str(main_pid) != observed["MainPID"]:
        raise UnitContractError("systemd handoff MainPID is invalid")
    root = Path(proc_root)
    _reject_dynamic(root, "proc root")
    try:
        actual_cmdline = (root / str(main_pid) / "cmdline").read_bytes()
    except OSError as exc:
        raise UnitContractError("cannot read collector MainPID argv") from exc
    expected_cmdline = (
        b"\0".join(
            argument.encode("utf-8") for argument in collector_argv(bindings)
        )
        + b"\0"
    )
    if actual_cmdline != expected_cmdline:
        raise UnitContractError(
            "collector MainPID argv does not match the immutable ExecStart"
        )
    return {
        "active_state": observed["ActiveState"],
        "fragment_path": observed["FragmentPath"],
        "main_pid": main_pid,
        "main_process_argv_sha256": hashlib.sha256(actual_cmdline).hexdigest(),
        "need_daemon_reload": False,
        "schema": UNIT_SCHEMA,
        "status": "VERIFIED",
        "sub_state": observed["SubState"],
        "unit_file_state": observed["UnitFileState"],
        "unit_name": unit_name,
    }


def production_handoff_check(
    bindings: ReleaseBindings,
    *,
    linger_user: str,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    proc_root: Path = Path("/proc"),
) -> dict:
    """Verify the explicit post-install operator handoff, read-only."""
    unit = check_unit(bindings, runner=runner)
    linger = verify_user_linger(linger_user, runner=runner)
    systemd = verify_systemd_handoff(
        bindings, runner=runner, proc_root=proc_root
    )
    return {
        "linger": linger,
        "schema": UNIT_SCHEMA,
        "status": (
            "COMMISSIONING_PREVIEW_HANDOFF_VERIFIED"
            if bindings.spec.commissioning_preview_only
            else "PRODUCTION_HANDOFF_VERIFIED"
        ),
        "systemd": systemd,
        "unit": unit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify, render, check, or install an immutable Linbox collector unit"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify_code_parser = commands.add_parser("verify-code")
    verify_code_parser.add_argument("--code-release", required=True)
    verify_code_parser.add_argument("--code-release-id", required=True)
    verify_code_parser.add_argument("--code-payload-sha256", required=True)
    verify_runtime_parser = commands.add_parser("verify-runtime")
    verify_runtime_parser.add_argument("--runtime-release", required=True)
    verify_runtime_parser.add_argument("--runtime-release-id", required=True)
    verify_runtime_parser.add_argument("--code-release-id", required=True)
    verify_runtime_parser.add_argument("--code-payload-sha256", required=True)
    verify_runtime_parser.add_argument("--environment-id", required=True)
    verify_runtime_parser.add_argument("--environment-lock-sha256", required=True)
    verify_activation_parser = commands.add_parser("verify-activation")
    verify_activation_parser.add_argument("--production-activation", required=True)
    verify_activation_parser.add_argument(
        "--production-activation-sha256", required=True
    )
    verify_force_parser = commands.add_parser("verify-force")
    verify_force_parser.add_argument("--hall-calibration", required=True)
    verify_force_parser.add_argument("--hall-calibration-sha256", required=True)
    verify_force_parser.add_argument("--sync-calibration", required=True)
    verify_force_parser.add_argument("--sync-calibration-sha256", required=True)
    verify_hall_parser = commands.add_parser("verify-hall-monitor")
    verify_hall_parser.add_argument("--hall-calibration", required=True)
    verify_hall_parser.add_argument(
        "--hall-calibration-sha256", required=True
    )
    verify_handeye_parser = commands.add_parser(
        "verify-commissioning-handeye"
    )
    verify_handeye_parser.add_argument(
        "--commissioning-handeye-release", required=True
    )
    verify_handeye_parser.add_argument(
        "--commissioning-handeye-artifact-sha256", required=True
    )
    verify_handeye_parser.add_argument(
        "--commissioning-handeye-complete-sha256", required=True
    )
    verify_handeye_parser.add_argument("--camera-id", required=True)
    linger_parser = commands.add_parser("check-linger")
    linger_parser.add_argument("--user", required=True)
    return parser


class _DeploymentParser(argparse.ArgumentParser):
    """Keep production inputs mandatory while preview may omit them."""

    _PRODUCTION_ONLY_FIELDS = (
        ("production_activation", "--production-activation"),
        ("production_activation_sha256", "--production-activation-sha256"),
        ("hall_calibration", "--hall-calibration"),
        ("hall_calibration_sha256", "--hall-calibration-sha256"),
        ("hall_port", "--hall-port"),
        ("sync_calibration", "--sync-calibration"),
        ("sync_calibration_sha256", "--sync-calibration-sha256"),
    )

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        preview = getattr(parsed, "commissioning_preview_only", False)
        handeye_fields = (
            "commissioning_handeye_release",
            "commissioning_handeye_artifact_sha256",
            "commissioning_handeye_complete_sha256",
        )
        handeye_count = sum(
            getattr(parsed, field, None) not in {None, ""}
            for field in handeye_fields
        )
        controller_launch_fields = (
            "controller_launch_release",
            "controller_launch_trusted_root",
            "controller_launch_contract_sha256",
        )
        controller_launch_count = sum(
            getattr(parsed, field, None) not in {None, ""}
            for field in controller_launch_fields
        )
        if controller_launch_count not in {0, len(controller_launch_fields)}:
            self.error(
                "controller launch requires release, trusted root, and SHA-256"
            )
        if handeye_count not in {0, len(handeye_fields)}:
            self.error(
                "commissioning hand-eye requires release and both SHA-256 pins"
            )
        if preview:
            if controller_launch_count:
                self.error(
                    "commissioning preview forbids controller launch binding"
                )
            if (
                getattr(parsed, "production_activation", None) is not None
                or getattr(parsed, "production_activation_sha256", None)
                is not None
            ):
                self.error(
                    "commissioning preview cannot bind production activation"
                )
            hall_monitor = getattr(
                parsed, "commissioning_hall_monitor_only", False
            )
            hall_fields = (
                "hall_calibration",
                "hall_calibration_sha256",
                "hall_port",
            )
            if hall_monitor:
                missing = [
                    field for field in hall_fields
                    if getattr(parsed, field, None) in {None, ""}
                ]
                if missing:
                    self.error(
                        "commissioning Hall monitor requires: "
                        + ", ".join("--" + field.replace("_", "-") for field in missing)
                    )
                if (
                    getattr(parsed, "sync_calibration", None) is not None
                    or getattr(parsed, "sync_calibration_sha256", None)
                    is not None
                ):
                    self.error(
                        "commissioning Hall monitor forbids sync calibration"
                    )
            elif any(
                getattr(parsed, field, None) is not None
                for field in (
                    *hall_fields,
                    "sync_calibration",
                    "sync_calibration_sha256",
                )
            ):
                self.error(
                    "preview Hall inputs require "
                    "--commissioning-hall-monitor-only"
                )
        else:
            if (
                getattr(parsed, "commissioning_hall_monitor_only", False)
                or handeye_count
            ):
                self.error("commissioning-only bindings require preview mode")
            missing = [
                option for field, option in self._PRODUCTION_ONLY_FIELDS
                if getattr(parsed, field, None) in {None, ""}
            ]
            if missing:
                self.error(
                    "production deployment requires: " + ", ".join(missing)
                )
        return parsed


def build_deployment_parser() -> argparse.ArgumentParser:
    parser = _DeploymentParser(
        description="Render, check, or atomically install an immutable collector unit"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--install", action="store_true")
    mode.add_argument("--production-check", action="store_true")
    parser.add_argument("--code-release", required=True)
    parser.add_argument("--code-release-id", required=True)
    parser.add_argument("--environment-release", required=True)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--runtime-release", required=True)
    parser.add_argument("--runtime-release-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--sessions-root", required=True)
    parser.add_argument("--unit-path", required=True)
    parser.add_argument("--verifier-python", required=True)
    parser.add_argument("--commissioning-preview-only", action="store_true")
    parser.add_argument(
        "--commissioning-hall-monitor-only", action="store_true"
    )
    parser.add_argument("--commissioning-handeye-release")
    parser.add_argument("--commissioning-handeye-artifact-sha256")
    parser.add_argument("--commissioning-handeye-complete-sha256")
    parser.add_argument("--production-activation")
    parser.add_argument("--production-activation-sha256")
    parser.add_argument("--hall-calibration")
    parser.add_argument("--hall-calibration-sha256")
    parser.add_argument("--hall-port")
    parser.add_argument("--hall-expected-usb-vid", type=lambda value: int(value, 0), default=0x2E8A)
    parser.add_argument("--hall-expected-usb-pid", type=lambda value: int(value, 0), default=0x0005)
    parser.add_argument("--hall-expected-usb-serial", default="53032847188a2e1c")
    parser.add_argument("--sync-calibration")
    parser.add_argument("--sync-calibration-sha256")
    parser.add_argument("--controller-launch-release")
    parser.add_argument("--controller-launch-trusted-root")
    parser.add_argument("--controller-launch-contract-sha256")
    parser.add_argument(
        "--safe-stop-disable-cmd", default=None,
        help="candidate CAN disable command the operator-witnessed safe-stop HIL validates "
             "(design (a)); emitted into ExecStart only with a controller-launch release",
    )
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument(
        "--audio-device", default="usb-serial:Y77BQVG17CD6E1"
    )
    parser.add_argument("--audio-sample-rate", type=int, default=48_000)
    parser.add_argument("--audio-channels", type=int, default=2)
    parser.add_argument("--audio-pick-channel", type=int, default=1)
    parser.add_argument("--audio-expected-usb-vid", type=lambda value: int(value, 0), default=0x1235)
    parser.add_argument("--audio-expected-usb-pid", type=lambda value: int(value, 0), default=0x8211)
    parser.add_argument("--audio-expected-usb-serial", default="Y77BQVG17CD6E1")
    parser.add_argument("--audio-expected-usb-port-path", default=None)
    parser.add_argument("--linger-user")
    expectation = parser.add_mutually_exclusive_group()
    expectation.add_argument("--expect-no-existing", action="store_true")
    expectation.add_argument("--expected-unit-sha256")
    return parser


def _service_spec_from_args(args: argparse.Namespace) -> ServiceSpec:
    return ServiceSpec(
        code_release=Path(args.code_release),
        code_release_id=args.code_release_id,
        environment_release=Path(args.environment_release),
        environment_id=args.environment_id,
        runtime_release=Path(args.runtime_release),
        runtime_release_id=args.runtime_release_id,
        camera_id=args.camera_id,
        sessions_root=Path(args.sessions_root),
        unit_path=Path(args.unit_path),
        verifier_python=Path(args.verifier_python),
        production_activation=(
            None
            if args.production_activation is None
            else Path(args.production_activation)
        ),
        production_activation_sha256=args.production_activation_sha256,
        commissioning_handeye_release=(
            None
            if args.commissioning_handeye_release is None
            else Path(args.commissioning_handeye_release)
        ),
        commissioning_handeye_artifact_sha256=(
            args.commissioning_handeye_artifact_sha256
        ),
        commissioning_handeye_complete_sha256=(
            args.commissioning_handeye_complete_sha256
        ),
        commissioning_hall_monitor_only=(
            args.commissioning_hall_monitor_only
        ),
        hall_calibration=(
            None if args.hall_calibration is None else Path(args.hall_calibration)
        ),
        hall_calibration_sha256=args.hall_calibration_sha256,
        hall_port=args.hall_port,
        hall_expected_usb_vid=args.hall_expected_usb_vid,
        hall_expected_usb_pid=args.hall_expected_usb_pid,
        hall_expected_usb_serial=args.hall_expected_usb_serial,
        sync_calibration=(
            None if args.sync_calibration is None else Path(args.sync_calibration)
        ),
        sync_calibration_sha256=args.sync_calibration_sha256,
        controller_launch_release=(
            None
            if args.controller_launch_release is None
            else Path(args.controller_launch_release)
        ),
        controller_launch_trusted_root=(
            None
            if args.controller_launch_trusted_root is None
            else Path(args.controller_launch_trusted_root)
        ),
        controller_launch_contract_sha256=(
            args.controller_launch_contract_sha256
        ),
        safe_stop_disable_cmd=args.safe_stop_disable_cmd,
        port=args.port,
        audio_device=args.audio_device,
        audio_sample_rate=args.audio_sample_rate,
        audio_channels=args.audio_channels,
        audio_pick_channel=args.audio_pick_channel,
        audio_expected_usb_vid=args.audio_expected_usb_vid,
        audio_expected_usb_pid=args.audio_expected_usb_pid,
        audio_expected_usb_serial=args.audio_expected_usb_serial,
        audio_expected_usb_port_path=args.audio_expected_usb_port_path,
        commissioning_preview_only=args.commissioning_preview_only,
    )


def _emit_report(stream, report: object) -> None:
    stream.write(contract.canonical_json_bytes(report).decode("utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in {
        "verify-code",
        "verify-runtime",
        "verify-activation",
        "verify-force",
        "verify-hall-monitor",
        "verify-commissioning-handeye",
        "check-linger",
    }:
        args = build_parser().parse_args(values)
        if args.command == "verify-code":
            report = verify_code_release(
                Path(args.code_release),
                args.code_release_id,
                args.code_payload_sha256,
            )
        elif args.command == "verify-runtime":
            report = verify_runtime_release(
                runtime_release=Path(args.runtime_release),
                runtime_release_id=args.runtime_release_id,
                code_release_id=args.code_release_id,
                code_payload_sha256=args.code_payload_sha256,
                environment_id=args.environment_id,
                environment_lock_sha256=args.environment_lock_sha256,
            )
        elif args.command == "verify-activation":
            report = verify_production_activation(
                Path(args.production_activation),
                args.production_activation_sha256,
            )
        elif args.command == "verify-force":
            report = verify_force_configuration(
                hall_calibration=Path(args.hall_calibration),
                hall_calibration_sha256=args.hall_calibration_sha256,
                sync_calibration=Path(args.sync_calibration),
                sync_calibration_sha256=args.sync_calibration_sha256,
            )
        elif args.command == "verify-hall-monitor":
            report = verify_hall_monitor_configuration(
                hall_calibration=Path(args.hall_calibration),
                hall_calibration_sha256=args.hall_calibration_sha256,
            )
        elif args.command == "verify-commissioning-handeye":
            report = verify_commissioning_handeye_release(
                release_path=Path(args.commissioning_handeye_release),
                artifact_sha256=(
                    args.commissioning_handeye_artifact_sha256
                ),
                complete_sha256=(
                    args.commissioning_handeye_complete_sha256
                ),
                camera_id=args.camera_id,
            )
        else:
            report = verify_user_linger(args.user)
        _emit_report(sys.stdout, report)
        return 0

    args = build_deployment_parser().parse_args(values)
    bindings = bind_releases(_service_spec_from_args(args))
    if args.render:
        sys.stdout.buffer.write(render_unit(bindings))
        return 0
    if args.check:
        if args.expect_no_existing or args.expected_unit_sha256:
            raise UnitContractError("--check does not accept install expectations")
        report = check_unit(bindings)
    elif args.install:
        report = install_unit(
            bindings,
            expect_no_existing=args.expect_no_existing,
            expected_unit_sha256=args.expected_unit_sha256,
        )
    else:
        if args.expect_no_existing or args.expected_unit_sha256:
            raise UnitContractError(
                "--production-check does not accept install expectations"
            )
        if not args.linger_user:
            raise UnitContractError("--production-check requires --linger-user")
        report = production_handoff_check(
            bindings, linger_user=args.linger_user
        )
    _emit_report(sys.stdout, report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except UnitContractError as exc:
        sys.stderr.write(
            contract.canonical_json_bytes(
                {"error": str(exc), "schema": UNIT_SCHEMA, "status": "FAILED"}
            ).decode("utf-8")
        )
        raise SystemExit(1)
