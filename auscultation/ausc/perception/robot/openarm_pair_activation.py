"""Immutable, fail-closed activation contract for the left OpenArm pair.

This module deliberately does not discover CAN devices, open a CAN socket, or
write an activation.  Commissioning code may use the pure builders to create a
candidate release, while the collector uses :func:`load_activated_openarm_pair`
with an independently pinned contract digest.

The only production mapping accepted by schema v2 is the left unilateral pair:
leader ``can1`` and follower ``can3``.  The content-addressed release is not an
authorization by itself; the loader also requires a caller-supplied SHA-256 pin,
an exact COMPLETE marker, and no STAGING marker.  Schema v1 artifacts are
explicitly legacy/non-production and cannot authorize telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Dict, Iterable, Mapping, Tuple


PAIR_SCHEMA = "wam_openarm_pair_contract_v2"
ACTIVATION_SCHEMA = "wam_openarm_pair_activation_v2"
COMPLETION_SCHEMA = "wam_openarm_pair_completion_v2"

LEGACY_PAIR_SCHEMA = "wam_openarm_pair_contract_v1"
LEGACY_ACTIVATION_SCHEMA = "wam_openarm_pair_activation_v1"
LEGACY_COMPLETION_SCHEMA = "wam_openarm_pair_completion_v1"

CONTRACT_FILE = "pair_contract.json"
STAGING_MARKER = "STAGING"
COMPLETE_MARKER = "COMPLETE"

LEFT_JOINT_NAMES = tuple(f"openarm_left_joint{index}" for index in range(1, 8))
ARM_SEND_CAN_IDS = tuple(range(0x01, 0x08))
ARM_RECEIVE_CAN_IDS = tuple(range(0x11, 0x18))
GRIPPER_SEND_CAN_ID = 0x08
GRIPPER_RECEIVE_CAN_ID = 0x18

_LEFT_LEADER_CAN = "can1"
_LEFT_FOLLOWER_CAN = "can3"
_FOLLOWER_CONFIG_PATH = "config/follower.yaml"
_MAX_JSON_BYTES = 1_048_576
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_STABLE_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:+/-]{1,512}\Z")
_RELEASE_ID_RE = re.compile(r"openarm_pair_v2-([0-9a-f]{64})\Z")
_LEGACY_RELEASE_ID_RE = re.compile(r"openarm_pair-([0-9a-f]{64})\Z")

_PROFILE_KEYS = {
    "schema",
    "side",
    "controller_mode",
    "leader",
    "follower",
    "joint_names",
    "urdf_sha256",
    "joint_mapper_header_sha256",
    "controller_source_manifest",
    "controller_source_manifest_sha256",
    "follower_config_sha256",
    "openarm_sdk_manifest",
    "openarm_sdk_sha256",
    "candidate_binary_sha256",
    "telemetry_socket",
    "camera_profile_sha256",
    "timing_artifact_sha256",
}
_ENDPOINT_KEYS = {
    "role",
    "can_interface",
    "adapter",
    "expected_motor_ids",
}
_ADAPTER_KEYS = {"id_path", "serial", "channel"}
_MOTOR_ID_KEYS = {
    "arm_send_can_ids",
    "arm_receive_can_ids",
    "gripper_send_can_id",
    "gripper_receive_can_id",
}
_SOURCE_ENTRY_KEYS = {"path", "sha256"}
_SDK_ENTRY_KEYS = {"kind", "path", "sha256"}
_ACTIVATION_KEYS = {"schema", "release_id", "contract_sha256", "profile"}
_COMPLETION_KEYS = {
    "schema",
    "status",
    "release_id",
    "contract_sha256",
    "contract_file_sha256",
}


class OpenArmPairActivationError(ValueError):
    """The pair activation cannot safely authorize robot telemetry."""


@dataclass(frozen=True)
class CanAdapterIdentity:
    """Stable udev identity for one physical CAN adapter channel."""

    id_path: str
    serial: str
    channel: str


@dataclass(frozen=True)
class ExpectedMotorIds:
    """Exact OpenArm v1 CAN IDs, including the gripper motor."""

    arm_send_can_ids: Tuple[int, ...]
    arm_receive_can_ids: Tuple[int, ...]
    gripper_send_can_id: int
    gripper_receive_can_id: int


@dataclass(frozen=True)
class OpenArmCanEndpoint:
    """One immutable role-to-interface-to-adapter mapping."""

    role: str
    can_interface: str
    adapter: CanAdapterIdentity
    expected_motor_ids: ExpectedMotorIds


@dataclass(frozen=True)
class ControllerSourceEntry:
    """One source byte identity used to build the candidate controller."""

    path: str
    sha256: str


@dataclass(frozen=True)
class OpenArmSdkEntry:
    """One canonical OpenArm SDK header or linked library identity."""

    kind: str
    path: str
    sha256: str


@dataclass(frozen=True)
class OpenArmPairActivation:
    """Validated identity of the production left OpenArm pair."""

    schema: str
    release_id: str
    contract_sha256: str
    side: str
    controller_mode: str
    leader: OpenArmCanEndpoint
    follower: OpenArmCanEndpoint
    joint_names: Tuple[str, ...]
    urdf_sha256: str
    joint_mapper_header_sha256: str
    controller_source_manifest: Tuple[ControllerSourceEntry, ...]
    controller_source_manifest_sha256: str
    follower_config_sha256: str
    openarm_sdk_manifest: Tuple[OpenArmSdkEntry, ...]
    openarm_sdk_sha256: str
    candidate_binary_sha256: str
    telemetry_socket: str
    camera_profile_sha256: str
    timing_artifact_sha256: str

    @property
    def joint_mapper_sha256(self) -> str:
        """RobotStateContract-v2 name for the pinned mapper header."""

        return self.joint_mapper_header_sha256

    @property
    def pair_contract_sha256(self) -> str:
        """RobotStateContract-v2 name for this activation contract digest."""

        return self.contract_sha256

    @property
    def controller_binary_sha256(self) -> str:
        """RobotStateContract-v2 name for the commissioned candidate binary."""

        return self.candidate_binary_sha256


def canonical_json_bytes(value: object) -> bytes:
    """Encode RFC-style canonical JSON used by every pair digest."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OpenArmPairActivationError(
            "value cannot be encoded as canonical JSON"
        ) from exc


def _strict_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OpenArmPairActivationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise OpenArmPairActivationError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _decode_canonical_json(raw: bytes, *, label: str) -> Dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_JSON_BYTES:
        raise OpenArmPairActivationError(f"{label} has an invalid size")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenArmPairActivationError(f"{label} is not valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise OpenArmPairActivationError(f"{label} must be a JSON object")
    if canonical_json_bytes(value) != raw:
        raise OpenArmPairActivationError(f"{label} must use canonical JSON")
    return value


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Dict[str, Any]:
    if type(value) is not dict:
        raise OpenArmPairActivationError(
            f"{label} must be an object with exact keys"
        )
    if set(value) != expected:
        raise OpenArmPairActivationError(
            f"{label} has wrong keys: expected {sorted(expected)!r}, "
            f"got {sorted(value)!r}"
        )
    return value


def _reject_legacy_schema(
    value: object,
    *,
    legacy_schema: str,
    label: str,
) -> None:
    if type(value) is dict and value.get("schema") == legacy_schema:
        raise OpenArmPairActivationError(
            f"legacy {label} v1 is non-production; a commissioned v2 artifact is required"
        )


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise OpenArmPairActivationError(
            f"{label} must be a lowercase 64-hex SHA-256"
        )
    return value


def _require_stable_identity(value: object, *, label: str) -> str:
    if type(value) is not str or _STABLE_TOKEN_RE.fullmatch(value) is None:
        raise OpenArmPairActivationError(
            f"{label} must be a non-empty stable udev identity"
        )
    return value


def _require_safe_source_path(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise OpenArmPairActivationError(
            "controller source path must be a safe relative POSIX path"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise OpenArmPairActivationError(
            "controller source path must be a safe relative POSIX path"
        )
    return value


def _require_socket_path(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or "\x00" in value
        or not value.endswith(".sock")
    ):
        raise OpenArmPairActivationError(
            "telemetry socket must be an absolute POSIX .sock path"
        )
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(value.encode("utf-8")) > 107
    ):
        raise OpenArmPairActivationError(
            "telemetry socket must be an absolute POSIX .sock path"
        )
    return value


def _parse_adapter(value: object, *, label: str) -> CanAdapterIdentity:
    adapter = _require_exact_keys(value, _ADAPTER_KEYS, label=label)
    return CanAdapterIdentity(
        id_path=_require_stable_identity(
            adapter["id_path"], label=f"{label} ID_PATH"
        ),
        serial=_require_stable_identity(
            adapter["serial"], label=f"{label} serial"
        ),
        channel=_require_stable_identity(
            adapter["channel"], label=f"{label} channel"
        ),
    )


def _exact_int(value: object, expected: int, *, label: str) -> int:
    if type(value) is not int or value != expected:
        raise OpenArmPairActivationError(f"{label} must equal {expected}")
    return value


def _exact_id_vector(
    value: object,
    expected: Tuple[int, ...],
    *,
    label: str,
) -> Tuple[int, ...]:
    if type(value) is not list or any(type(item) is not int for item in value):
        raise OpenArmPairActivationError(f"{label} must be an exact integer list")
    result = tuple(value)
    if result != expected:
        raise OpenArmPairActivationError(
            f"{label} must match the commissioned OpenArm motor ID order"
        )
    return result


def _parse_motor_ids(value: object, *, label: str) -> ExpectedMotorIds:
    motor_ids = _require_exact_keys(value, _MOTOR_ID_KEYS, label=label)
    return ExpectedMotorIds(
        arm_send_can_ids=_exact_id_vector(
            motor_ids["arm_send_can_ids"],
            ARM_SEND_CAN_IDS,
            label=f"{label} arm send IDs",
        ),
        arm_receive_can_ids=_exact_id_vector(
            motor_ids["arm_receive_can_ids"],
            ARM_RECEIVE_CAN_IDS,
            label=f"{label} arm receive IDs",
        ),
        gripper_send_can_id=_exact_int(
            motor_ids["gripper_send_can_id"],
            GRIPPER_SEND_CAN_ID,
            label=f"{label} gripper send ID",
        ),
        gripper_receive_can_id=_exact_int(
            motor_ids["gripper_receive_can_id"],
            GRIPPER_RECEIVE_CAN_ID,
            label=f"{label} gripper receive ID",
        ),
    )


def _parse_endpoint(
    value: object,
    *,
    expected_role: str,
    expected_can_interface: str,
) -> OpenArmCanEndpoint:
    endpoint = _require_exact_keys(
        value, _ENDPOINT_KEYS, label=f"{expected_role} endpoint"
    )
    if endpoint["role"] != expected_role:
        raise OpenArmPairActivationError(
            f"{expected_role} endpoint role must equal {expected_role}"
        )
    if endpoint["can_interface"] != expected_can_interface:
        raise OpenArmPairActivationError(
            f"{expected_role} CAN interface must be {expected_can_interface}; "
            "can0/can2 and cross-side mappings are forbidden"
        )
    return OpenArmCanEndpoint(
        role=expected_role,
        can_interface=expected_can_interface,
        adapter=_parse_adapter(
            endpoint["adapter"], label=f"{expected_role} adapter"
        ),
        expected_motor_ids=_parse_motor_ids(
            endpoint["expected_motor_ids"],
            label=f"{expected_role} expected motor IDs",
        ),
    )


def _parse_source_manifest(
    value: object,
    declared_sha256: object,
) -> Tuple[Tuple[ControllerSourceEntry, ...], str]:
    if type(value) is not list or not value:
        raise OpenArmPairActivationError(
            "controller source manifest must be a non-empty list"
        )
    entries = []
    paths = []
    for index, raw_entry in enumerate(value):
        entry = _require_exact_keys(
            raw_entry,
            _SOURCE_ENTRY_KEYS,
            label=f"controller source manifest entry {index}",
        )
        path = _require_safe_source_path(entry["path"])
        digest = _require_sha256(
            entry["sha256"],
            label=f"controller source manifest entry {index} SHA-256",
        )
        paths.append(path)
        entries.append(ControllerSourceEntry(path=path, sha256=digest))
    if len(set(paths)) != len(paths):
        raise OpenArmPairActivationError(
            "controller source manifest contains a duplicate path"
        )
    if paths != sorted(paths):
        raise OpenArmPairActivationError(
            "controller source manifest paths must be sorted"
        )
    declared = _require_sha256(
        declared_sha256,
        label="controller source manifest SHA-256",
    )
    actual = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    if declared != actual:
        raise OpenArmPairActivationError(
            "controller source manifest SHA-256 does not match its canonical bytes"
        )
    return tuple(entries), declared


def _parse_sdk_manifest(
    value: object,
    declared_sha256: object,
) -> Tuple[Tuple[OpenArmSdkEntry, ...], str]:
    if type(value) is not list or not value:
        raise OpenArmPairActivationError(
            "OpenArm SDK manifest must be a non-empty list"
        )
    entries = []
    paths = []
    kinds = set()
    for index, raw_entry in enumerate(value):
        entry = _require_exact_keys(
            raw_entry,
            _SDK_ENTRY_KEYS,
            label=f"OpenArm SDK manifest entry {index}",
        )
        path = _require_safe_source_path(entry["path"])
        kind = entry["kind"]
        if kind not in {"header", "library"}:
            raise OpenArmPairActivationError(
                "OpenArm SDK manifest kind must be header or library"
            )
        digest = _require_sha256(
            entry["sha256"],
            label=f"OpenArm SDK manifest entry {index} SHA-256",
        )
        paths.append(path)
        kinds.add(kind)
        entries.append(OpenArmSdkEntry(kind=kind, path=path, sha256=digest))
    if len(set(paths)) != len(paths):
        raise OpenArmPairActivationError(
            "OpenArm SDK manifest contains a duplicate path"
        )
    if paths != sorted(paths):
        raise OpenArmPairActivationError(
            "OpenArm SDK manifest paths must be sorted"
        )
    if kinds != {"header", "library"}:
        raise OpenArmPairActivationError(
            "OpenArm SDK manifest must bind at least one header and library"
        )
    declared = _require_sha256(
        declared_sha256,
        label="OpenArm SDK manifest SHA-256",
    )
    actual = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    if declared != actual:
        raise OpenArmPairActivationError(
            "OpenArm SDK manifest SHA-256 does not match its canonical bytes"
        )
    return tuple(entries), declared


def _parse_profile(value: object) -> OpenArmPairActivation:
    _reject_legacy_schema(
        value,
        legacy_schema=LEGACY_PAIR_SCHEMA,
        label="pair contract",
    )
    profile = _require_exact_keys(value, _PROFILE_KEYS, label="pair profile")
    if profile["schema"] != PAIR_SCHEMA:
        raise OpenArmPairActivationError(
            f"pair profile schema must be exactly {PAIR_SCHEMA}"
        )
    if profile["side"] != "left":
        raise OpenArmPairActivationError(
            "production pair profile must select the left side"
        )
    if profile["controller_mode"] != "unilateral":
        raise OpenArmPairActivationError(
            "production pair profile must use unilateral controller mode"
        )
    leader = _parse_endpoint(
        profile["leader"],
        expected_role="leader",
        expected_can_interface=_LEFT_LEADER_CAN,
    )
    follower = _parse_endpoint(
        profile["follower"],
        expected_role="follower",
        expected_can_interface=_LEFT_FOLLOWER_CAN,
    )
    if leader.adapter == follower.adapter:
        raise OpenArmPairActivationError(
            "leader and follower must not resolve to the same adapter channel identity"
        )
    joint_names = profile["joint_names"]
    if type(joint_names) is not list or tuple(joint_names) != LEFT_JOINT_NAMES:
        raise OpenArmPairActivationError(
            "joint names must match canonical openarm_left_joint1..7 order"
        )
    source_manifest, source_manifest_sha256 = _parse_source_manifest(
        profile["controller_source_manifest"],
        profile["controller_source_manifest_sha256"],
    )
    follower_config_sha256 = _require_sha256(
        profile["follower_config_sha256"],
        label="follower config SHA-256",
    )
    follower_config_entries = tuple(
        entry
        for entry in source_manifest
        if entry.path == _FOLLOWER_CONFIG_PATH
    )
    if len(follower_config_entries) != 1:
        raise OpenArmPairActivationError(
            "controller source manifest must contain exactly config/follower.yaml"
        )
    if follower_config_entries[0].sha256 != follower_config_sha256:
        raise OpenArmPairActivationError(
            "follower config SHA-256 does not match config/follower.yaml"
        )
    sdk_manifest, openarm_sdk_sha256 = _parse_sdk_manifest(
        profile["openarm_sdk_manifest"],
        profile["openarm_sdk_sha256"],
    )
    return OpenArmPairActivation(
        schema=PAIR_SCHEMA,
        release_id="",
        contract_sha256="",
        side="left",
        controller_mode="unilateral",
        leader=leader,
        follower=follower,
        joint_names=LEFT_JOINT_NAMES,
        urdf_sha256=_require_sha256(
            profile["urdf_sha256"], label="URDF SHA-256"
        ),
        joint_mapper_header_sha256=_require_sha256(
            profile["joint_mapper_header_sha256"],
            label="joint mapper header SHA-256",
        ),
        controller_source_manifest=source_manifest,
        controller_source_manifest_sha256=source_manifest_sha256,
        follower_config_sha256=follower_config_sha256,
        openarm_sdk_manifest=sdk_manifest,
        openarm_sdk_sha256=openarm_sdk_sha256,
        candidate_binary_sha256=_require_sha256(
            profile["candidate_binary_sha256"],
            label="candidate binary SHA-256",
        ),
        telemetry_socket=_require_socket_path(profile["telemetry_socket"]),
        camera_profile_sha256=_require_sha256(
            profile["camera_profile_sha256"],
            label="camera profile SHA-256",
        ),
        timing_artifact_sha256=_require_sha256(
            profile["timing_artifact_sha256"],
            label="timing artifact SHA-256",
        ),
    )


def _canonical_profile_copy(profile: Mapping[str, object]) -> Dict[str, Any]:
    raw = canonical_json_bytes(profile)
    # A round trip guarantees ordinary JSON values and detaches the activation
    # envelope from all caller-owned mutable dictionaries and lists.
    return _decode_canonical_json(raw, label="pair profile")


def build_activation_envelope(
    profile: Mapping[str, object],
) -> Dict[str, object]:
    """Validate a candidate profile and return its content-addressed envelope.

    This function is pure: it neither writes nor activates anything.
    """

    canonical_profile = _canonical_profile_copy(profile)
    _parse_profile(canonical_profile)
    contract_sha256 = hashlib.sha256(
        canonical_json_bytes(canonical_profile)
    ).hexdigest()
    return {
        "schema": ACTIVATION_SCHEMA,
        "release_id": f"openarm_pair_v2-{contract_sha256}",
        "contract_sha256": contract_sha256,
        "profile": canonical_profile,
    }


def _parse_activation_envelope(value: object) -> Tuple[OpenArmPairActivation, Dict[str, Any]]:
    _reject_legacy_schema(
        value,
        legacy_schema=LEGACY_ACTIVATION_SCHEMA,
        label="pair activation",
    )
    envelope = _require_exact_keys(
        value, _ACTIVATION_KEYS, label="activation envelope"
    )
    if envelope["schema"] != ACTIVATION_SCHEMA:
        raise OpenArmPairActivationError(
            f"activation schema must be exactly {ACTIVATION_SCHEMA}"
        )
    contract_sha256 = _require_sha256(
        envelope["contract_sha256"],
        label="pair contract SHA-256",
    )
    release_id = envelope["release_id"]
    if type(release_id) is not str or _RELEASE_ID_RE.fullmatch(release_id) is None:
        if type(release_id) is str and _LEGACY_RELEASE_ID_RE.fullmatch(release_id):
            raise OpenArmPairActivationError(
                "legacy v1 release_id is non-production; openarm_pair_v2 is required"
            )
        raise OpenArmPairActivationError(
            "release_id must be openarm_pair_v2- followed by the full contract SHA-256"
        )
    profile = _require_exact_keys(
        envelope["profile"], _PROFILE_KEYS, label="pair profile"
    )
    parsed = _parse_profile(profile)
    actual_sha256 = hashlib.sha256(canonical_json_bytes(profile)).hexdigest()
    if contract_sha256 != actual_sha256:
        raise OpenArmPairActivationError(
            "pair contract hash does not match canonical profile bytes"
        )
    expected_release_id = f"openarm_pair_v2-{contract_sha256}"
    if release_id != expected_release_id:
        raise OpenArmPairActivationError(
            "release_id does not match the pair contract hash"
        )
    activated = replace(
        parsed,
        release_id=release_id,
        contract_sha256=contract_sha256,
    )
    return activated, envelope


def build_completion_marker(
    envelope: Mapping[str, object],
    contract_file_bytes: bytes,
) -> Dict[str, object]:
    """Return the exact marker for already-canonical activation bytes.

    This is pure and intentionally does not remove a STAGING marker or write a
    COMPLETE marker.  Publication code must perform those atomic operations.
    """

    parsed, canonical_envelope = _parse_activation_envelope(envelope)
    if (
        not isinstance(contract_file_bytes, bytes)
        or canonical_json_bytes(canonical_envelope) != contract_file_bytes
    ):
        raise OpenArmPairActivationError(
            "completion requires exact canonical pair contract file bytes"
        )
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "release_id": parsed.release_id,
        "contract_sha256": parsed.contract_sha256,
        "contract_file_sha256": hashlib.sha256(contract_file_bytes).hexdigest(),
    }


def _path_from_value(value: object) -> Path:
    try:
        return Path(os.fspath(value))
    except TypeError as exc:
        raise OpenArmPairActivationError(
            "pair activation release must be path-like"
        ) from exc


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_regular_file(path: Path, *, label: str) -> bytes:
    if path.is_symlink():
        raise OpenArmPairActivationError(f"{label} must not be a symlink")
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise OpenArmPairActivationError(f"{label} is missing") from exc
    if not stat.S_ISREG(mode):
        raise OpenArmPairActivationError(f"{label} must be a regular file")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OpenArmPairActivationError(f"{label} cannot be read") from exc
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise OpenArmPairActivationError(f"{label} has an invalid size")
    return raw


def load_activated_openarm_pair(
    release_directory: object,
    *,
    expected_contract_sha256: str,
) -> OpenArmPairActivation:
    """Load a completed left-pair release only when it matches an external pin.

    A valid self-consistent directory is still rejected if its contract digest
    differs from ``expected_contract_sha256``.  This prevents an operator or
    partial deployment from silently replacing every self-attested hash.
    """

    pinned_sha256 = _require_sha256(
        expected_contract_sha256,
        label="pinned pair contract SHA-256",
    )
    release = _path_from_value(release_directory)
    if release.is_symlink():
        raise OpenArmPairActivationError(
            "pair activation release directory must not be a symlink"
        )
    if not release.is_dir():
        raise OpenArmPairActivationError(
            "pair activation release directory is missing"
        )
    staging = release / STAGING_MARKER
    if _lexists(staging):
        raise OpenArmPairActivationError(
            "STAGING marker is present; pair activation is incomplete"
        )
    try:
        names = {entry.name for entry in release.iterdir()}
    except OSError as exc:
        raise OpenArmPairActivationError(
            "pair activation release directory cannot be listed"
        ) from exc
    expected_names = {CONTRACT_FILE, COMPLETE_MARKER}
    if names != expected_names:
        raise OpenArmPairActivationError(
            f"pair activation files must be exactly {sorted(expected_names)!r}"
        )

    contract_raw = _require_regular_file(
        release / CONTRACT_FILE,
        label="pair contract file",
    )
    envelope_value = _decode_canonical_json(
        contract_raw,
        label="pair contract file",
    )
    activation, envelope = _parse_activation_envelope(envelope_value)
    if release.name != activation.release_id:
        raise OpenArmPairActivationError(
            "release directory name does not match content-addressed release_id"
        )
    if activation.contract_sha256 != pinned_sha256:
        raise OpenArmPairActivationError(
            "activated pair differs from the independently pinned contract SHA-256"
        )

    complete_raw = _require_regular_file(
        release / COMPLETE_MARKER,
        label="COMPLETE marker",
    )
    complete = _decode_canonical_json(
        complete_raw,
        label="COMPLETE marker",
    )
    _reject_legacy_schema(
        complete,
        legacy_schema=LEGACY_COMPLETION_SCHEMA,
        label="completion marker",
    )
    _require_exact_keys(
        complete, _COMPLETION_KEYS, label="completion marker"
    )
    if complete["schema"] != COMPLETION_SCHEMA:
        raise OpenArmPairActivationError(
            f"completion marker schema must be exactly {COMPLETION_SCHEMA}"
        )
    expected_complete = build_completion_marker(envelope, contract_raw)
    if complete != expected_complete:
        raise OpenArmPairActivationError(
            "completion marker does not bind the exact pair contract"
        )
    return activation


__all__ = [
    "ACTIVATION_SCHEMA",
    "ARM_RECEIVE_CAN_IDS",
    "ARM_SEND_CAN_IDS",
    "COMPLETE_MARKER",
    "COMPLETION_SCHEMA",
    "CONTRACT_FILE",
    "CanAdapterIdentity",
    "ControllerSourceEntry",
    "ExpectedMotorIds",
    "GRIPPER_RECEIVE_CAN_ID",
    "GRIPPER_SEND_CAN_ID",
    "LEGACY_ACTIVATION_SCHEMA",
    "LEGACY_COMPLETION_SCHEMA",
    "LEGACY_PAIR_SCHEMA",
    "LEFT_JOINT_NAMES",
    "OpenArmCanEndpoint",
    "OpenArmPairActivation",
    "OpenArmPairActivationError",
    "OpenArmSdkEntry",
    "PAIR_SCHEMA",
    "STAGING_MARKER",
    "build_activation_envelope",
    "build_completion_marker",
    "canonical_json_bytes",
    "load_activated_openarm_pair",
]
