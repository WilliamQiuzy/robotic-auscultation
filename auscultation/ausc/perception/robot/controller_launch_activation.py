"""Immutable, dormant launch contract for the left auscultation controller.

The sidecar binds bytes needed to launch the commissioned controller, but it is
not an arm-authority or HIL-acceptance artifact.  Loading it performs no device
discovery, subprocess launch, or CAN I/O, and this module intentionally exposes
no execution API.

A future HIL-gated executor must close the verify-to-exec race with already-
verified ``O_NOFOLLOW`` file descriptors (or an equivalently immutable mount)
and descriptor-based exec.  Re-opening these pathnames after this dormant load
is not sufficient authorization to execute them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Tuple

from .openarm_pair_activation import (
    PAIR_SCHEMA,
    OpenArmPairActivation,
)


PROFILE_SCHEMA = "wam_controller_launch_profile_v1"
ACTIVATION_SCHEMA = "wam_controller_launch_activation_v1"
COMPLETION_SCHEMA = "wam_controller_launch_completion_v1"
STOP_TOOL_DESCRIPTOR_SCHEMA = "wam_openarm_stop_tool_descriptor_v1"

CONTRACT_FILE = "controller_launch.json"
STAGING_MARKER = "STAGING"
COMPLETE_MARKER = "COMPLETE"

CONTROLLER_PATH = "controller/unilateral_control"
LEADER_CONFIG_PATH = "config/leader.yaml"
FOLLOWER_CONFIG_PATH = "config/follower.yaml"
LEADER_URDF_PATH = "urdf/leader.urdf"
FOLLOWER_URDF_PATH = "urdf/follower.urdf"
PREFLIGHT_PATH = "tools/preflight"
STOP_TOOL_DESCRIPTOR_PATH = "stop_tool.json"

_DORMANT_STATE = "DORMANT_HIL_PENDING"
_OPEN_TARGET_STATUS = "HIL_PENDING"
_MAX_JSON_BYTES = 1_048_576
_MAX_ARTIFACT_BYTES = 1 << 31
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RELEASE_ID_RE = re.compile(r"controller_launch_v1-([0-9a-f]{64})\Z")

_ARTIFACT_SPECS = {
    CONTROLLER_PATH: ("controller_elf", "0555"),
    LEADER_CONFIG_PATH: ("leader_config", "0444"),
    FOLLOWER_CONFIG_PATH: ("follower_config", "0444"),
    LEADER_URDF_PATH: ("leader_urdf", "0444"),
    FOLLOWER_URDF_PATH: ("follower_urdf", "0444"),
    PREFLIGHT_PATH: ("preflight_elf", "0555"),
    STOP_TOOL_DESCRIPTOR_PATH: ("stop_tool_descriptor", "0444"),
}
_EXECUTABLE_PATHS = frozenset({CONTROLLER_PATH, PREFLIGHT_PATH})
_EXPECTED_ARGV = (
    CONTROLLER_PATH,
    LEADER_URDF_PATH,
    FOLLOWER_URDF_PATH,
    "left_arm",
    "can1",
    "can3",
)
_PROFILE_KEYS = {
    "schema",
    "state",
    "open_target_status",
    "openarm_pair",
    "artifacts",
    "cwd",
    "argv",
    "environment",
}
_PAIR_BINDING_KEYS = {"schema", "release_id", "contract_sha256"}
_ARTIFACT_KEYS = {"role", "path", "sha256", "size", "mode"}
_ACTIVATION_KEYS = {"schema", "release_id", "contract_sha256", "profile"}
_COMPLETION_KEYS = {
    "schema",
    "status",
    "release_id",
    "contract_sha256",
    "contract_file_sha256",
}


class ControllerLaunchActivationError(ValueError):
    """A controller-launch release is not an exact dormant sidecar."""


@dataclass(frozen=True)
class ControllerLaunchArtifact:
    role: str
    path: str
    sha256: str
    size: int
    mode: str


@dataclass(frozen=True)
class ControllerLaunchActivation:
    """Validated launch inputs; deliberately carries no authorization bit."""

    schema: str
    release_id: str
    contract_sha256: str
    state: str
    open_target_status: str
    openarm_pair_release_id: str
    openarm_pair_contract_sha256: str
    artifacts: Tuple[ControllerLaunchArtifact, ...]
    trusted_root: Path
    release_directory: Path
    cwd: Path
    argv: Tuple[str, ...]
    _environment_items: Tuple[Tuple[str, str], ...]
    _openarm_pair: OpenArmPairActivation

    @property
    def environment(self) -> Mapping[str, str]:
        """Return a read-only copy so callers cannot change future launches."""

        return MappingProxyType(dict(self._environment_items))

    @property
    def controller_path(self) -> Path:
        return self.release_directory / CONTROLLER_PATH

    @property
    def preflight_path(self) -> Path:
        return self.release_directory / PREFLIGHT_PATH

    @property
    def stop_tool_descriptor_path(self) -> Path:
        return self.release_directory / STOP_TOOL_DESCRIPTOR_PATH


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ControllerLaunchActivationError(
            "value cannot be encoded as canonical JSON"
        ) from exc


def _strict_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControllerLaunchActivationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ControllerLaunchActivationError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _decode_canonical_json(raw: bytes, *, label: str) -> Dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_JSON_BYTES:
        raise ControllerLaunchActivationError(f"{label} has an invalid size")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerLaunchActivationError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    if type(value) is not dict:
        raise ControllerLaunchActivationError(f"{label} must be a JSON object")
    if canonical_json_bytes(value) != raw:
        raise ControllerLaunchActivationError(f"{label} must use canonical JSON")
    return value


def _exact_object(value: object, keys: set[str], *, label: str) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        got = sorted(value) if type(value) is dict else type(value).__name__
        raise ControllerLaunchActivationError(
            f"{label} has wrong keys: expected {sorted(keys)!r}, got {got!r}"
        )
    return value


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ControllerLaunchActivationError(
            f"{label} must be a lowercase 64-hex SHA-256"
        )
    return value


def _require_pair_v2(pair: object) -> OpenArmPairActivation:
    if not isinstance(pair, OpenArmPairActivation):
        raise ControllerLaunchActivationError(
            "openarm_pair must be a loaded OpenArmPairActivation"
        )
    if (
        pair.schema != PAIR_SCHEMA
        or pair.side != "left"
        or pair.controller_mode != "unilateral"
        or pair.leader.can_interface != "can1"
        or pair.follower.can_interface != "can3"
        or pair.release_id != f"openarm_pair_v2-{pair.contract_sha256}"
    ):
        raise ControllerLaunchActivationError(
            "controller launch requires the exact left can1 -> can3 pair-v2 activation"
        )
    _sha256(pair.contract_sha256, label="pair contract SHA-256")
    return pair


def build_controller_environment(
    openarm_pair: OpenArmPairActivation,
) -> Dict[str, str]:
    """Build the only accepted environment without reading ``os.environ``."""

    pair = _require_pair_v2(openarm_pair)
    return {
        "GRIPPER_DECOUPLE": "1",
        "GRIPPER_HOLD_POS": "0.0",
        "GRIP_LEAD": "0.08",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        # 2026-08-07 wrist-oscillation fix, operator-validated at the arm: gain 0.5 with zero
        # damping self-drives the low-friction wrist (friction comp = positive velocity feedback;
        # huge tanh k on j5/j6 makes it bang-bang) -> violent limit-to-limit slamming. 0.32 with
        # Kd=0.5 is stable and light enough; stiction back to the code default 0.1.
        "TELEOP_FRICTION_GAIN": "0.32",
        "TELEOP_LEADER_DAMP": "0.5",
        "TELEOP_STICTION": "0.1",
        "WAM_OPENARM_CONTROLLER_BINARY_SHA256": pair.candidate_binary_sha256,
        "WAM_OPENARM_CONTROLLER_SOURCE_MANIFEST_SHA256": (
            pair.controller_source_manifest_sha256
        ),
        "WAM_OPENARM_FOLLOWER_CONFIG_SHA256": pair.follower_config_sha256,
        "WAM_OPENARM_JOINT_MAPPER_SHA256": pair.joint_mapper_header_sha256,
        "WAM_OPENARM_PAIR_CONTRACT_SHA256": pair.contract_sha256,
        "WAM_OPENARM_SDK_SHA256": pair.openarm_sdk_sha256,
        "WAM_OPENARM_TELEMETRY_SOCKET": pair.telemetry_socket,
        "WAM_OPENARM_URDF_SHA256": pair.urdf_sha256,
    }


def _parse_artifacts(value: object) -> Tuple[ControllerLaunchArtifact, ...]:
    if type(value) is not list or len(value) != len(_ARTIFACT_SPECS):
        raise ControllerLaunchActivationError(
            "artifact manifest must contain the exact launch artifact set"
        )
    parsed = []
    paths = []
    for index, item in enumerate(value):
        entry = _exact_object(item, _ARTIFACT_KEYS, label=f"artifact {index}")
        path = entry["path"]
        if type(path) is not str or path not in _ARTIFACT_SPECS:
            raise ControllerLaunchActivationError(
                f"artifact {index} path is not an approved canonical path"
            )
        pure = PurePosixPath(path)
        if pure.is_absolute() or str(pure) != path or ".." in pure.parts:
            raise ControllerLaunchActivationError(
                f"artifact {index} path is not a safe relative POSIX path"
            )
        role, mode = _ARTIFACT_SPECS[path]
        if entry["role"] != role or entry["mode"] != mode:
            raise ControllerLaunchActivationError(
                f"artifact {path} role or mode differs from the launch contract"
            )
        size = entry["size"]
        if type(size) is not int or size <= 0 or size > _MAX_ARTIFACT_BYTES:
            raise ControllerLaunchActivationError(
                f"artifact {path} has an invalid size"
            )
        paths.append(path)
        parsed.append(
            ControllerLaunchArtifact(
                role=role,
                path=path,
                sha256=_sha256(entry["sha256"], label=f"artifact {path} SHA-256"),
                size=size,
                mode=mode,
            )
        )
    if paths != sorted(_ARTIFACT_SPECS):
        raise ControllerLaunchActivationError(
            "artifact manifest paths must be unique and sorted"
        )
    return tuple(parsed)


def _parse_profile(
    value: object,
    *,
    openarm_pair: OpenArmPairActivation,
) -> Tuple[Dict[str, Any], Tuple[ControllerLaunchArtifact, ...]]:
    pair = _require_pair_v2(openarm_pair)
    profile = _exact_object(value, _PROFILE_KEYS, label="launch profile")
    if profile["schema"] != PROFILE_SCHEMA:
        raise ControllerLaunchActivationError(
            f"launch profile schema must be exactly {PROFILE_SCHEMA}"
        )
    if profile["state"] != _DORMANT_STATE:
        raise ControllerLaunchActivationError(
            "launch profile must remain dormant pending HIL"
        )
    if profile["open_target_status"] != _OPEN_TARGET_STATUS:
        raise ControllerLaunchActivationError(
            "gripper open target must remain HIL_PENDING"
        )
    binding = _exact_object(
        profile["openarm_pair"],
        _PAIR_BINDING_KEYS,
        label="OpenArm pair binding",
    )
    expected_binding = {
        "schema": PAIR_SCHEMA,
        "release_id": pair.release_id,
        "contract_sha256": pair.contract_sha256,
    }
    if binding != expected_binding:
        raise ControllerLaunchActivationError(
            "launch profile does not cross-bind the loaded pair-v2 activation"
        )
    artifacts = _parse_artifacts(profile["artifacts"])
    by_path = {artifact.path: artifact for artifact in artifacts}
    if by_path[CONTROLLER_PATH].sha256 != pair.candidate_binary_sha256:
        raise ControllerLaunchActivationError(
            "controller ELF does not match pair-v2 candidate binary"
        )
    if by_path[FOLLOWER_CONFIG_PATH].sha256 != pair.follower_config_sha256:
        raise ControllerLaunchActivationError(
            "follower config does not match pair-v2 activation"
        )
    if by_path[LEADER_URDF_PATH].sha256 != pair.urdf_sha256:
        raise ControllerLaunchActivationError(
            "leader URDF does not match pair-v2 activation"
        )
    if profile["cwd"] != ".":
        raise ControllerLaunchActivationError("launch cwd must be the release root")
    if type(profile["argv"]) is not list or tuple(profile["argv"]) != _EXPECTED_ARGV:
        raise ControllerLaunchActivationError(
            "launch argv must directly execute the left can1 -> can3 controller ELF"
        )
    expected_environment = build_controller_environment(pair)
    if type(profile["environment"]) is not dict or profile["environment"] != expected_environment:
        raise ControllerLaunchActivationError(
            "launch environment must be the exact clean commissioned environment"
        )
    return profile, artifacts


def build_controller_launch_envelope(
    profile: Mapping[str, object],
    *,
    openarm_pair: OpenArmPairActivation,
) -> Dict[str, object]:
    """Validate and content-address a candidate launch profile without writing."""

    canonical_profile = _decode_canonical_json(
        canonical_json_bytes(profile),
        label="launch profile",
    )
    _parse_profile(canonical_profile, openarm_pair=openarm_pair)
    digest = hashlib.sha256(canonical_json_bytes(canonical_profile)).hexdigest()
    return {
        "schema": ACTIVATION_SCHEMA,
        "release_id": f"controller_launch_v1-{digest}",
        "contract_sha256": digest,
        "profile": canonical_profile,
    }


def _parse_envelope(
    value: object,
    *,
    openarm_pair: OpenArmPairActivation,
) -> Tuple[Dict[str, Any], Tuple[ControllerLaunchArtifact, ...]]:
    envelope = _exact_object(value, _ACTIVATION_KEYS, label="launch activation")
    if envelope["schema"] != ACTIVATION_SCHEMA:
        raise ControllerLaunchActivationError(
            f"launch activation schema must be exactly {ACTIVATION_SCHEMA}"
        )
    digest = _sha256(envelope["contract_sha256"], label="launch contract SHA-256")
    release_id = envelope["release_id"]
    if type(release_id) is not str or _RELEASE_ID_RE.fullmatch(release_id) is None:
        raise ControllerLaunchActivationError(
            "release_id must be controller_launch_v1- followed by the full SHA-256"
        )
    profile, artifacts = _parse_profile(
        envelope["profile"],
        openarm_pair=openarm_pair,
    )
    actual = hashlib.sha256(canonical_json_bytes(profile)).hexdigest()
    if digest != actual or release_id != f"controller_launch_v1-{actual}":
        raise ControllerLaunchActivationError(
            "launch release identity does not match canonical profile bytes"
        )
    return envelope, artifacts


def build_completion_marker(
    envelope: Mapping[str, object],
    contract_file_bytes: bytes,
) -> Dict[str, object]:
    if type(envelope) is not dict:
        raise ControllerLaunchActivationError("launch activation must be an object")
    if canonical_json_bytes(envelope) != contract_file_bytes:
        raise ControllerLaunchActivationError(
            "completion requires exact canonical launch contract bytes"
        )
    digest = _sha256(envelope.get("contract_sha256"), label="launch contract SHA-256")
    release_id = envelope.get("release_id")
    if type(release_id) is not str or release_id != f"controller_launch_v1-{digest}":
        raise ControllerLaunchActivationError("completion release identity is invalid")
    if envelope.get("schema") != ACTIVATION_SCHEMA:
        raise ControllerLaunchActivationError("completion activation schema is invalid")
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "release_id": release_id,
        "contract_sha256": digest,
        "contract_file_sha256": hashlib.sha256(contract_file_bytes).hexdigest(),
    }


def _mode_string(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def _read_regular(
    path: Path,
    *,
    label: str,
    expected_mode: str,
    max_bytes: int,
) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ControllerLaunchActivationError(f"{label} is missing") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ControllerLaunchActivationError(f"{label} must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise ControllerLaunchActivationError(f"{label} must be a regular file")
    if info.st_uid != os.geteuid():
        raise ControllerLaunchActivationError(
            f"{label} must be owned by the collector user"
        )
    if _mode_string(info.st_mode) != expected_mode:
        raise ControllerLaunchActivationError(
            f"{label} mode must be exactly {expected_mode}"
        )
    if info.st_size <= 0 or info.st_size > max_bytes:
        raise ControllerLaunchActivationError(f"{label} has an invalid size")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ControllerLaunchActivationError(f"{label} cannot be read") from exc


def _require_immutable_directory(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ControllerLaunchActivationError(f"{label} is missing") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ControllerLaunchActivationError(f"{label} must not be a symlink")
    if not stat.S_ISDIR(info.st_mode):
        raise ControllerLaunchActivationError(f"{label} must be a directory")
    if info.st_uid != os.geteuid():
        raise ControllerLaunchActivationError(
            f"{label} must be owned by the collector user"
        )
    if stat.S_IMODE(info.st_mode) & 0o222:
        raise ControllerLaunchActivationError(
            f"{label} must not be owner/group/world writable"
        )


def _canonical_absolute_path(value: object, *, label: str) -> Path:
    try:
        path = Path(os.fspath(value))
    except TypeError as exc:
        raise ControllerLaunchActivationError(f"{label} must be path-like") from exc
    if not path.is_absolute() or path != Path(os.path.abspath(os.fspath(path))):
        raise ControllerLaunchActivationError(
            f"{label} must be canonical and absolute"
        )
    return path


def _verify_trusted_ancestry(release: Path, trusted_root: Path) -> None:
    """Trust only the explicitly selected, locked tree—not ancestors up to ``/``."""

    try:
        relative = release.relative_to(trusted_root)
    except ValueError as exc:
        raise ControllerLaunchActivationError(
            "launch release must be beneath the explicit trusted root"
        ) from exc
    if relative == Path(".") or not relative.parts:
        raise ControllerLaunchActivationError(
            "launch release must be a content-addressed descendant of the trusted root"
        )
    current = trusted_root
    _require_immutable_directory(current, label="trusted launch root")
    for index, component in enumerate(relative.parts):
        current = current / component
        _require_immutable_directory(
            current,
            label=(
                "launch release directory"
                if index == len(relative.parts) - 1
                else "launch release ancestor"
            ),
        )


def _expected_tree_nodes() -> Tuple[set[str], set[str]]:
    files = set(_ARTIFACT_SPECS) | {CONTRACT_FILE, COMPLETE_MARKER}
    directories = set()
    for filename in _ARTIFACT_SPECS:
        parent = PurePosixPath(filename).parent
        while str(parent) != ".":
            directories.add(str(parent))
            parent = parent.parent
    return files, directories


def _verify_exact_tree(release: Path) -> None:
    expected_files, expected_dirs = _expected_tree_nodes()
    actual_files = set()
    actual_dirs = set()
    try:
        for root, dirnames, filenames in os.walk(release, followlinks=False):
            root_path = Path(root)
            for name in dirnames:
                path = root_path / name
                relative = path.relative_to(release).as_posix()
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise ControllerLaunchActivationError(
                        f"launch release contains symlink: {relative}"
                    )
                if not stat.S_ISDIR(info.st_mode):
                    raise ControllerLaunchActivationError(
                        f"launch release has a non-directory node: {relative}"
                    )
                _require_immutable_directory(
                    path,
                    label=f"launch artifact directory {relative}",
                )
                actual_dirs.add(relative)
            for name in filenames:
                path = root_path / name
                relative = path.relative_to(release).as_posix()
                if path.is_symlink():
                    raise ControllerLaunchActivationError(
                        f"launch release contains symlink: {relative}"
                    )
                actual_files.add(relative)
    except ControllerLaunchActivationError:
        raise
    except OSError as exc:
        raise ControllerLaunchActivationError(
            "launch release tree cannot be inspected"
        ) from exc
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ControllerLaunchActivationError(
            "launch release tree does not match the exact file allowlist"
        )


def _verify_payload(release: Path, artifacts: Tuple[ControllerLaunchArtifact, ...]) -> None:
    for artifact in artifacts:
        path = release / artifact.path
        raw = _read_regular(
            path,
            label=f"artifact {artifact.path}",
            expected_mode=artifact.mode,
            max_bytes=_MAX_ARTIFACT_BYTES,
        )
        if len(raw) != artifact.size or hashlib.sha256(raw).hexdigest() != artifact.sha256:
            raise ControllerLaunchActivationError(
                f"artifact {artifact.path} size or SHA-256 does not match"
            )
        if artifact.path in _EXECUTABLE_PATHS and not raw.startswith(b"\x7fELF"):
            raise ControllerLaunchActivationError(
                f"artifact {artifact.path} must be a direct ELF, not a script or interpreter"
            )
    descriptor = _decode_canonical_json(
        (release / STOP_TOOL_DESCRIPTOR_PATH).read_bytes(),
        label="stop-tool descriptor",
    )
    if descriptor != {
        "schema": STOP_TOOL_DESCRIPTOR_SCHEMA,
        "status": "HIL_PENDING",
    }:
        raise ControllerLaunchActivationError(
            "stop-tool descriptor must remain HIL_PENDING"
        )


def load_controller_launch_activation(
    release_directory: object,
    *,
    trusted_root: object,
    expected_contract_sha256: str,
    openarm_pair: OpenArmPairActivation,
) -> ControllerLaunchActivation:
    """Load an exact completed sidecar without authorizing or executing it."""

    pair = _require_pair_v2(openarm_pair)
    pinned = _sha256(
        expected_contract_sha256,
        label="pinned launch contract SHA-256",
    )
    release = _canonical_absolute_path(
        release_directory,
        label="launch release path",
    )
    root = _canonical_absolute_path(trusted_root, label="trusted launch root")
    _verify_trusted_ancestry(release, root)
    if os.path.lexists(os.fspath(release / STAGING_MARKER)):
        raise ControllerLaunchActivationError(
            "STAGING marker is present; launch release is incomplete"
        )
    _verify_exact_tree(release)

    contract_raw = _read_regular(
        release / CONTRACT_FILE,
        label="launch contract",
        expected_mode="0444",
        max_bytes=_MAX_JSON_BYTES,
    )
    envelope_value = _decode_canonical_json(contract_raw, label="launch contract")
    envelope, artifacts = _parse_envelope(
        envelope_value,
        openarm_pair=pair,
    )
    if envelope["contract_sha256"] != pinned:
        raise ControllerLaunchActivationError(
            "launch release differs from the independently pinned SHA-256"
        )
    if release.name != envelope["release_id"]:
        raise ControllerLaunchActivationError(
            "launch release directory name does not match release_id"
        )

    complete_raw = _read_regular(
        release / COMPLETE_MARKER,
        label="launch COMPLETE marker",
        expected_mode="0444",
        max_bytes=_MAX_JSON_BYTES,
    )
    complete = _decode_canonical_json(complete_raw, label="launch COMPLETE marker")
    _exact_object(complete, _COMPLETION_KEYS, label="launch COMPLETE marker")
    expected_complete = build_completion_marker(envelope, contract_raw)
    if complete != expected_complete:
        raise ControllerLaunchActivationError(
            "launch COMPLETE marker does not bind the exact contract"
        )
    _verify_payload(release, artifacts)

    profile = envelope["profile"]
    environment = profile["environment"]
    resolved_argv = (
        str((release / CONTROLLER_PATH).resolve()),
        str((release / LEADER_URDF_PATH).resolve()),
        str((release / FOLLOWER_URDF_PATH).resolve()),
        "left_arm",
        "can1",
        "can3",
    )
    return ControllerLaunchActivation(
        schema=PROFILE_SCHEMA,
        release_id=str(envelope["release_id"]),
        contract_sha256=str(envelope["contract_sha256"]),
        state=_DORMANT_STATE,
        open_target_status=_OPEN_TARGET_STATUS,
        openarm_pair_release_id=pair.release_id,
        openarm_pair_contract_sha256=pair.contract_sha256,
        artifacts=artifacts,
        trusted_root=root,
        release_directory=release,
        cwd=release.resolve(),
        argv=resolved_argv,
        _environment_items=tuple(sorted(environment.items())),
        _openarm_pair=pair,
    )


def verify_controller_launch_activation(
    activation: ControllerLaunchActivation,
) -> None:
    """Re-read every byte so a post-load mutation fails before later use."""

    if not isinstance(activation, ControllerLaunchActivation):
        raise ControllerLaunchActivationError(
            "activation must be a loaded ControllerLaunchActivation"
        )
    reloaded = load_controller_launch_activation(
        activation.release_directory,
        trusted_root=activation.trusted_root,
        expected_contract_sha256=activation.contract_sha256,
        openarm_pair=activation._openarm_pair,
    )
    if reloaded != activation:
        raise ControllerLaunchActivationError(
            "launch activation changed after it was loaded"
        )


__all__ = [
    "ACTIVATION_SCHEMA",
    "COMPLETE_MARKER",
    "COMPLETION_SCHEMA",
    "CONTRACT_FILE",
    "CONTROLLER_PATH",
    "ControllerLaunchActivation",
    "ControllerLaunchActivationError",
    "ControllerLaunchArtifact",
    "FOLLOWER_CONFIG_PATH",
    "FOLLOWER_URDF_PATH",
    "LEADER_CONFIG_PATH",
    "LEADER_URDF_PATH",
    "PREFLIGHT_PATH",
    "PROFILE_SCHEMA",
    "STAGING_MARKER",
    "STOP_TOOL_DESCRIPTOR_PATH",
    "STOP_TOOL_DESCRIPTOR_SCHEMA",
    "build_completion_marker",
    "build_controller_environment",
    "build_controller_launch_envelope",
    "canonical_json_bytes",
    "load_controller_launch_activation",
    "verify_controller_launch_activation",
]
