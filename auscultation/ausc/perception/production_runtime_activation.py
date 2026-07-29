"""Fail-closed assembly of the production left OpenArm occlusion runtime.

The collector receives one externally SHA-pinned activation manifest.  The
manifest only points at immutable, hash-pinned artifacts; it never discovers,
publishes, promotes, or activates a calibration.  Every referenced artifact
must have an independently bound ``COMPLETE`` marker.  In particular, a
hand-eye candidate or a self-consistent but uncompleted depth/timing/identity
file cannot construct a production runtime.

This module is intentionally read-only.  Constructing the returned coordinator
does not bind its telemetry socket, start a controller, open CAN, or actuate the
robot.  The trusted Linux launcher remains responsible for the two-phase
controller process attestation before reception can become ready.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Dict, Tuple, Union

import numpy as np

from .robot.camera_exposure_evidence import CameraExposureContract
from .robot.openarm_pair_activation import (
    OpenArmPairActivation,
    load_activated_openarm_pair,
)
from .robot.robot_capture_coordinator import (
    ProductionRobotCaptureCoordinator,
)
from .robot.robot_mask_provider import (
    ActivatedHandEye,
    CameraProjectionProfile,
    KinematicCollisionLink,
    OpenArmCollisionGeometry,
    OpenArmToolGeometry,
    ProvisionalRobotMaskHandEye,
    ProductionRobotMaskProvider,
    RobotMaskProviderError,
    RobotMaskUncertaintyProfile,
)
from .robot.robot_state_stream import RobotStateContract
from .tracking.depth_noise_profile import ActivatedDepthNoiseProfile
from .tracking.geometry import Intrinsics
from .tracking.occlusion_runtime import (
    ActivatedCameraProjection,
    ActivatedOcclusionIdentity,
    ActivatedOcclusionRuntimeProfile,
    OcclusionRuntime,
    OcclusionRuntimeConfig,
)
from .tracking.patient_pose_provider import (
    MetricPatientPoseProvider,
    PatientPoseProviderConfig,
)
from .tracking.posterior_back_mask_provider import (
    IndependentPosteriorBackMaskProvider,
    PosteriorBackMaskConfig,
)


ACTIVATION_SCHEMA = "wam_production_runtime_activation_v1"
ACTIVATION_COMPLETION_SCHEMA = (
    "wam_production_runtime_activation_completion_v1"
)
ARTIFACT_COMPLETION_SCHEMA = (
    "wam_production_runtime_artifact_completion_v1"
)
PROVISIONAL_HANDEYE_ARTIFACT_COMPLETION_SCHEMA = (
    "wam_openarm_handeye_artifact_completion_v2"
)
PROVISIONAL_HANDEYE_RELEASE_COMPLETION_SCHEMA = (
    "wam_openarm_handeye_release_completion_v2"
)
ACTIVATION_FILE = "activation.json"
COMPLETE_MARKER = "COMPLETE"
STAGING_MARKER = "STAGING"

_MAX_JSON_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DYNAMIC_COMPONENTS = frozenset(
    {"active", "current", "latest", ".wam-current", "wam-current"}
)
_ARTIFACT_NAMES = frozenset(
    {
        "alignment",
        "back_mask",
        "camera_latency",
        "camera_profile",
        "collision_geometry",
        "depth_noise",
        "handeye",
        "occlusion_identity",
        "occlusion_runtime",
        "patient_pose",
        "robot_mask_uncertainty",
        "tool_geometry",
    }
)
_MANIFEST_KEYS = {
    "artifacts",
    "follower_can_interface",
    "leader_can_interface",
    "openarm_pair",
    "robot_state_bounds",
    "schema",
    "side",
    "status",
}
_PAIR_KEYS = {"contract_sha256", "release_directory"}
_ENTRY_KEYS = {"complete_path", "complete_sha256", "path", "sha256"}
_BOUNDS_KEYS = {
    "buffer_duration_ns",
    "gripper_motor_position_limits_rad",
    "joint_position_limits_rad",
    "max_abs_joint_velocity_rad_s",
    "max_exposure_interval_ns",
    "max_gap_ns",
    "max_packets",
    "max_staleness_ns",
}


class ProductionRuntimeActivationError(RuntimeError):
    """No production collector may be assembled from these artifacts."""


@dataclass(frozen=True)
class ProductionRuntimeAssembly:
    """Validated dependencies injected into one production ``Collector``."""

    activation_path: str
    activation_sha256: str
    openarm_pair: OpenArmPairActivation
    camera_exposure_contract: CameraExposureContract
    handeye: Union[ActivatedHandEye, ProvisionalRobotMaskHandEye]
    activated_identity: ActivatedOcclusionIdentity
    robot_capture_coordinator: ProductionRobotCaptureCoordinator
    robot_mask_provider: ProductionRobotMaskProvider
    back_mask_provider: IndependentPosteriorBackMaskProvider
    patient_pose_provider: MetricPatientPoseProvider
    occlusion_runtime: OcclusionRuntime

    def prepare_robot_receiver(self) -> None:
        """Bind the read-only socket before a trusted launcher starts control."""

        self.robot_capture_coordinator.prepare_receiver_socket()

    def attach_and_start_robot_receiver(self, attestation: object) -> None:
        """Finish the receiver handshake with a launcher-captured process pin.

        The caller owns controller startup and must supply an independent final
        exec attestation.  This method neither launches nor commands the arm.
        """

        self.robot_capture_coordinator.attach_controller_process(attestation)
        self.robot_capture_coordinator.start_receiver()

    def close(self) -> None:
        """Stop an attached receiver without ever touching the controller."""

        self.robot_capture_coordinator.stop_receiver()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionRuntimeActivationError(
            "activation value is not canonical-JSON encodable"
        ) from exc


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProductionRuntimeActivationError(
                f"duplicate activation JSON field: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ProductionRuntimeActivationError(
        f"non-finite activation JSON constant is forbidden: {value}"
    )


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ProductionRuntimeActivationError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _require_exact_mapping(
    value: object, expected: set[str], *, label: str,
) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise ProductionRuntimeActivationError(
            f"{label} must have exact keys {sorted(expected)!r}; got {actual!r}"
        )
    return value


def _immutable_path(value: object, *, label: str, directory: bool) -> Path:
    try:
        path = Path(os.fspath(value))
    except TypeError as exc:
        raise ProductionRuntimeActivationError(f"{label} is not path-like") from exc
    if not path.is_absolute():
        raise ProductionRuntimeActivationError(f"{label} must be absolute")
    if any(
        component.lower() in _DYNAMIC_COMPONENTS
        for component in PurePosixPath(str(path)).parts
    ):
        raise ProductionRuntimeActivationError(
            f"{label} must not use a current/latest/active pointer"
        )
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProductionRuntimeActivationError(f"{label} is missing") from exc
    if stat.S_ISLNK(info.st_mode) or resolved != path:
        raise ProductionRuntimeActivationError(
            f"{label} or a parent component is a symlink"
        )
    expected_mode = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_mode(info.st_mode):
        kind = "directory" if directory else "regular file"
        raise ProductionRuntimeActivationError(f"{label} must be a {kind}")
    return path


def _read_stable_file(
    value: object,
    *,
    label: str,
    expected_sha256: str | None,
) -> bytes:
    path = _immutable_path(value, label=label, directory=False)
    expected = (
        None
        if expected_sha256 is None
        else _require_sha256(expected_sha256, label=f"{label} SHA-256")
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionRuntimeActivationError(f"{label} cannot be opened") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_JSON_BYTES:
            raise ProductionRuntimeActivationError(f"{label} has an invalid size")
        chunks = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise ProductionRuntimeActivationError(
                    f"{label} was truncated while reading"
                )
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
            raise ProductionRuntimeActivationError(
                f"{label} changed while reading"
            )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ProductionRuntimeActivationError(f"{label} SHA-256 mismatch")
    return raw


def _decode_json(raw: bytes, *, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionRuntimeActivationError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    if type(value) is not dict:
        raise ProductionRuntimeActivationError(f"{label} must be a JSON object")
    return value


def _read_json(
    value: object,
    *,
    label: str,
    expected_sha256: str | None,
    canonical: bool = False,
) -> Tuple[Dict[str, Any], bytes]:
    raw = _read_stable_file(
        value, label=label, expected_sha256=expected_sha256,
    )
    decoded = _decode_json(raw, label=label)
    # Activation/manifest canonical files are stored as canonical JSON + one
    # trailing newline, matching the artifact/complete markers (`+ b"\n"`) and the
    # deploy verifier (collection_service_unit._stable_canonical_object).
    if canonical and _canonical_json_bytes(decoded) + b"\n" != raw:
        raise ProductionRuntimeActivationError(f"{label} is not canonical JSON")
    return decoded, raw


def _load_artifact(name: str, value: object) -> Tuple[Dict[str, Any], str]:
    entry = _require_exact_mapping(
        value, _ENTRY_KEYS, label=f"{name} artifact entry",
    )
    artifact_sha256 = _require_sha256(
        entry["sha256"], label=f"{name} artifact SHA-256",
    )
    payload, artifact_raw = _read_json(
        entry["path"],
        label=f"{name} artifact",
        expected_sha256=artifact_sha256,
    )
    completion_sha256 = _require_sha256(
        entry["complete_sha256"], label=f"{name} COMPLETE SHA-256",
    )
    complete, raw = _read_json(
        entry["complete_path"],
        label=f"{name} COMPLETE marker",
        expected_sha256=completion_sha256,
    )
    if name == "handeye" and payload.get("schema") == "wam_openarm_handeye_activation_v2":
        artifact_path = Path(os.fspath(entry["path"]))
        complete_path = Path(os.fspath(entry["complete_path"]))
        release = artifact_path.parent
        expected_release_id = "handeye_activation-" + artifact_sha256[:20]
        expected_marker = {
            "artifact_sha256": artifact_sha256,
            "calibration_status": "USER_ACCEPTED_BELOW_GATE",
            "provisional": True,
            "schema": PROVISIONAL_HANDEYE_ARTIFACT_COMPLETION_SCHEMA,
            "status": "COMPLETE",
        }
        if (
            artifact_path.name != "handeye.json"
            or complete_path.name != "handeye.COMPLETE"
            or complete_path.parent != release
            or release.name != expected_release_id
            or complete != expected_marker
            or _canonical_json_bytes(payload) + b"\n" != artifact_raw
            or _canonical_json_bytes(complete) + b"\n" != raw
        ):
            raise ProductionRuntimeActivationError(
                "handeye provisional COMPLETE marker/release topology is invalid"
            )
        release = _immutable_path(
            release, label="handeye provisional release", directory=True,
        )
        release_complete_path = release / COMPLETE_MARKER
        if not release_complete_path.exists():
            raise ProductionRuntimeActivationError(
                "handeye provisional release COMPLETE is missing"
            )
        release_complete, release_raw = _read_json(
            release_complete_path,
            label="handeye provisional release COMPLETE",
            expected_sha256=None,
        )
        inventory = {
            "handeye.COMPLETE": {
                "sha256": completion_sha256,
                "size_bytes": len(raw),
            },
            "handeye.json": {
                "sha256": artifact_sha256,
                "size_bytes": len(artifact_raw),
            },
        }
        expected_release_complete = {
            "artifact_sha256": artifact_sha256,
            "inventory": inventory,
            "release_id": expected_release_id,
            "schema": PROVISIONAL_HANDEYE_RELEASE_COMPLETION_SCHEMA,
            "status": "COMPLETE",
        }
        try:
            release_names = {path.name for path in release.iterdir()}
        except OSError as exc:
            raise ProductionRuntimeActivationError(
                "handeye provisional release inventory cannot be read"
            ) from exc
        if (
            release_complete != expected_release_complete
            or _canonical_json_bytes(release_complete) + b"\n" != release_raw
            or release_names != {"COMPLETE", "handeye.COMPLETE", "handeye.json"}
        ):
            raise ProductionRuntimeActivationError(
                "handeye provisional release COMPLETE does not bind its inventory"
            )
        if stat.S_IMODE(release.stat().st_mode) != 0o555:
            raise ProductionRuntimeActivationError(
                "handeye provisional release read-only directory mode is invalid"
            )
        for path in (artifact_path, complete_path, release_complete_path):
            if stat.S_IMODE(path.stat().st_mode) != 0o444:
                raise ProductionRuntimeActivationError(
                    "handeye provisional release read-only file mode is invalid"
                )
    else:
        expected = {
            "artifact_sha256": artifact_sha256,
            "schema": ARTIFACT_COMPLETION_SCHEMA,
            "status": "COMPLETE",
        }
        if complete != expected or _canonical_json_bytes(complete) + b"\n" != raw:
            raise ProductionRuntimeActivationError(
                f"{name} COMPLETE marker does not bind the artifact"
            )
    if hashlib.sha256(raw).hexdigest() != completion_sha256:
        raise ProductionRuntimeActivationError(
            f"{name} COMPLETE marker SHA-256 does not bind the artifact"
        )
    return payload, artifact_sha256


def _load_activation_manifest(
    path_value: object, *, expected_sha256: str,
) -> Tuple[Path, Dict[str, Any], str]:
    expected = _require_sha256(
        expected_sha256, label="pinned production activation SHA-256",
    )
    path = _immutable_path(
        path_value, label="production activation manifest", directory=False,
    )
    if path.name != ACTIVATION_FILE:
        raise ProductionRuntimeActivationError(
            f"production activation manifest must be named {ACTIVATION_FILE}"
        )
    if os.path.lexists(path.parent / STAGING_MARKER):
        raise ProductionRuntimeActivationError(
            "production activation STAGING marker is present"
        )
    manifest, raw = _read_json(
        path,
        label="production activation manifest",
        expected_sha256=expected,
        canonical=True,
    )
    _require_exact_mapping(manifest, _MANIFEST_KEYS, label="activation manifest")
    if (
        manifest["schema"] != ACTIVATION_SCHEMA
        or manifest["status"] != "COMPLETE"
        or manifest["side"] != "left"
        or manifest["leader_can_interface"] != "can1"
        or manifest["follower_can_interface"] != "can3"
    ):
        raise ProductionRuntimeActivationError(
            "activation must be COMPLETE left can1 leader -> can3 follower"
        )
    complete_path = path.parent / COMPLETE_MARKER
    complete_raw = _read_stable_file(
        complete_path,
        label="production activation COMPLETE marker",
        expected_sha256=None,
    )
    complete = _decode_json(
        complete_raw, label="production activation COMPLETE marker",
    )
    if _canonical_json_bytes(complete) + b"\n" != complete_raw or complete != {
        "activation_sha256": hashlib.sha256(raw).hexdigest(),
        "schema": ACTIVATION_COMPLETION_SCHEMA,
        "status": "COMPLETE",
    }:
        raise ProductionRuntimeActivationError(
            "production activation COMPLETE marker does not bind the manifest"
        )
    return path, manifest, expected


def _camera_profile(payload: Mapping[str, object]) -> Dict[str, Any]:
    expected = {
        "alignment_sha256",
        "camera_id",
        "capture_mode",
        "intrinsics",
        "min_projection_depth_m",
        "schema",
    }
    value = _require_exact_mapping(
        dict(payload), expected, label="camera_profile artifact",
    )
    if value["schema"] != "wam_rgbd_camera_profile_activation_v1":
        raise ProductionRuntimeActivationError(
            "camera_profile artifact schema is unsupported"
        )
    _require_sha256(value["alignment_sha256"], label="camera alignment SHA-256")
    intrinsics = _require_exact_mapping(
        value["intrinsics"],
        {"cx", "cy", "depth_scale_mm", "fx", "fy", "height", "width"},
        label="camera profile intrinsics",
    )
    if type(value["capture_mode"]) is not dict:
        raise ProductionRuntimeActivationError(
            "camera profile capture_mode must be an object"
        )
    value["intrinsics"] = intrinsics
    return value


def _build_geometry(payload: Mapping[str, object]) -> OpenArmCollisionGeometry:
    value = _require_exact_mapping(
        dict(payload),
        {"joint_names", "links", "schema", "side", "source", "urdf_sha256"},
        label="collision_geometry artifact",
    )
    if value["schema"] != "wam_openarm_collision_geometry_source_v1":
        raise ProductionRuntimeActivationError(
            "collision_geometry artifact schema is unsupported"
        )
    if type(value["links"]) is not list:
        raise ProductionRuntimeActivationError(
            "collision_geometry links must be a list"
        )
    link_keys = {
        "joint_axis",
        "joint_index",
        "name",
        "parent_index",
        "transform_parent_from_joint_zero",
        "vertices_link_m",
    }
    links = []
    for index, raw_link in enumerate(value["links"]):
        link = _require_exact_mapping(
            raw_link, link_keys, label=f"collision_geometry link {index}",
        )
        links.append(
            KinematicCollisionLink(
                name=link["name"],
                parent_index=link["parent_index"],
                joint_index=link["joint_index"],
                transform_parent_from_joint_zero=np.asarray(
                    link["transform_parent_from_joint_zero"], dtype=np.float64,
                ),
                joint_axis=np.asarray(link["joint_axis"], dtype=np.float64),
                vertices_link_m=np.asarray(
                    link["vertices_link_m"], dtype=np.float64,
                ),
            )
        )
    return OpenArmCollisionGeometry.create(
        source=value["source"],
        side=value["side"],
        joint_names=value["joint_names"],
        urdf_sha256=value["urdf_sha256"],
        links=links,
    )


def _build_tool(payload: Mapping[str, object]) -> OpenArmToolGeometry:
    value = _require_exact_mapping(
        dict(payload),
        {
            "parent_link_index",
            "schema",
            "source",
            "transform_parent_from_tool",
            "vertices_tool_m",
        },
        label="tool_geometry artifact",
    )
    if value["schema"] != "wam_openarm_tool_geometry_source_v1":
        raise ProductionRuntimeActivationError(
            "tool_geometry artifact schema is unsupported"
        )
    return OpenArmToolGeometry.create(
        source=value["source"],
        parent_link_index=value["parent_link_index"],
        transform_parent_from_tool=np.asarray(
            value["transform_parent_from_tool"], dtype=np.float64,
        ),
        vertices_tool_m=np.asarray(value["vertices_tool_m"], dtype=np.float64),
    )


def _build_handeye(
    payload: Mapping[str, object], *, artifact_sha256: str,
) -> ActivatedHandEye | ProvisionalRobotMaskHandEye:
    if payload.get("schema") == "wam_openarm_handeye_activation_v2":
        return ProvisionalRobotMaskHandEye.from_activation_v2(
            dict(payload),
            artifact_sha256=artifact_sha256,
            publication_status="COMPLETE",
        )
    value = _require_exact_mapping(
        dict(payload),
        {
            "camera_id",
            "covariance_se3",
            "schema",
            "side",
            "transform_camera_from_follower_base",
        },
        label="handeye artifact",
    )
    if value["schema"] != "wam_openarm_handeye_activation_v1":
        raise ProductionRuntimeActivationError(
            "handeye artifact is not a production activation"
        )
    return ActivatedHandEye.create(
        camera_id=value["camera_id"],
        side=value["side"],
        transform_camera_from_follower_base=np.asarray(
            value["transform_camera_from_follower_base"], dtype=np.float64,
        ),
        covariance_se3=np.asarray(value["covariance_se3"], dtype=np.float64),
        status="COMPLETE",
    )


def load_completed_handeye_release(
    release_dir: object,
    *,
    expected_artifact_sha256: str,
    expected_complete_sha256: str,
    expected_camera_id: str,
) -> ProvisionalRobotMaskHandEye:
    """Load one completed provisional hand-eye release for commissioning UI.

    This is deliberately narrower than the production runtime assembly.  It
    verifies the same v2 artifact marker, release marker, exact inventory and
    read-only topology, but returns only the provisional hand-eye capability.
    It does not construct a robot mask, enable base-frame targeting, or start a
    robot receiver.
    """

    release = _immutable_path(
        release_dir, label="commissioning hand-eye release", directory=True,
    )
    artifact_sha256 = _require_sha256(
        expected_artifact_sha256,
        label="commissioning hand-eye artifact SHA-256",
    )
    complete_sha256 = _require_sha256(
        expected_complete_sha256,
        label="commissioning hand-eye COMPLETE SHA-256",
    )
    if not isinstance(expected_camera_id, str) or not expected_camera_id:
        raise ProductionRuntimeActivationError(
            "commissioning hand-eye expected camera identity is missing"
        )
    entry = {
        "complete_path": str(release / "handeye.COMPLETE"),
        "complete_sha256": complete_sha256,
        "path": str(release / "handeye.json"),
        "sha256": artifact_sha256,
    }
    payload, observed_sha256 = _load_artifact("handeye", entry)
    try:
        handeye = _build_handeye(
            payload, artifact_sha256=observed_sha256,
        )
    except RobotMaskProviderError as exc:
        raise ProductionRuntimeActivationError(
            "commissioning hand-eye capability is invalid"
        ) from exc
    if not isinstance(handeye, ProvisionalRobotMaskHandEye):
        raise ProductionRuntimeActivationError(
            "commissioning hand-eye must be the provisional v2 capability"
        )
    if handeye.camera_id != expected_camera_id:
        raise ProductionRuntimeActivationError(
            "commissioning hand-eye camera identity mismatch"
        )
    return handeye


def _validate_handeye_runtime_bound(
    handeye: ActivatedHandEye | ProvisionalRobotMaskHandEye,
    config: OcclusionRuntimeConfig,
) -> None:
    if not isinstance(config, OcclusionRuntimeConfig):
        raise ProductionRuntimeActivationError(
            "occlusion runtime config is missing for hand-eye validation"
        )
    if isinstance(handeye, ProvisionalRobotMaskHandEye):
        minimum = max(
            handeye.workspace_1sigma_max,
            handeye.provisional_policy_allowance_m,
        )
        if float(config.handeye_bound_m) + 1e-12 < minimum:
            raise ProductionRuntimeActivationError(
                "occlusion runtime hand-eye bound must be at least 6.5 mm "
                "for the provisional 5 mm tier"
            )
    fixed_total = (
        float(config.handeye_bound_m)
        + float(config.timing_sweep_bound_m)
        + float(config.projection_bound_m)
        + float(config.tool_geometry_bound_m)
    )
    if fixed_total > float(config.max_uncertainty_bound_m) + 1e-12:
        raise ProductionRuntimeActivationError(
            "occlusion runtime fixed uncertainty exceeds its total bound"
        )


def _build_uncertainty(
    payload: Mapping[str, object],
) -> RobotMaskUncertaintyProfile:
    names = {
        "base_dilation_px",
        "link_geometry_error_m",
        "projection_error_px",
        "schema",
        "sigma",
        "timing_uncertainty_s",
        "tool_geometry_error_m",
    }
    value = _require_exact_mapping(
        dict(payload), names, label="robot_mask_uncertainty artifact",
    )
    if value.pop("schema") != "wam_robot_mask_uncertainty_activation_v1":
        raise ProductionRuntimeActivationError(
            "robot_mask_uncertainty artifact schema is unsupported"
        )
    return RobotMaskUncertaintyProfile.create(**value)


def _build_back_mask(
    payload: Mapping[str, object],
) -> IndependentPosteriorBackMaskProvider:
    value = _require_exact_mapping(
        dict(payload), {"config", "schema"}, label="back_mask artifact",
    )
    if value["schema"] != "wam_posterior_back_mask_activation_v1":
        raise ProductionRuntimeActivationError(
            "back_mask artifact schema is unsupported"
        )
    config_names = set(PosteriorBackMaskConfig.__dataclass_fields__) - {
        "release_id", "semantic_sha256",
    }
    config = _require_exact_mapping(
        value["config"], config_names, label="back_mask config",
    )
    return IndependentPosteriorBackMaskProvider(
        config=PosteriorBackMaskConfig.create(**config),
    )


def _build_depth_profile(
    payload: Mapping[str, object],
) -> ActivatedDepthNoiseProfile:
    value = dict(payload)
    schema = value.pop("schema", None)
    if schema != "wam_activated_aligned_depth_noise_profile_v1":
        raise ProductionRuntimeActivationError(
            "depth_noise artifact schema is unsupported"
        )
    expected = {field.name for field in fields(ActivatedDepthNoiseProfile)}
    _require_exact_mapping(value, expected, label="depth_noise profile")
    return ActivatedDepthNoiseProfile(**value)


def _build_config(
    payload: Mapping[str, object],
    *,
    schema: str,
    config_type: type,
    label: str,
) -> object:
    value = _require_exact_mapping(
        dict(payload), {"config", "schema"}, label=f"{label} artifact",
    )
    if value["schema"] != schema:
        raise ProductionRuntimeActivationError(f"{label} artifact schema is unsupported")
    expected = {field.name for field in fields(config_type)}
    config = _require_exact_mapping(
        value["config"], expected, label=f"{label} config",
    )
    return config_type(**config)


def _build_identity(
    payload: Mapping[str, object],
    *,
    handeye: Union[ActivatedHandEye, ProvisionalRobotMaskHandEye],
) -> ActivatedOcclusionIdentity:
    value = dict(payload)
    schema = value.pop("schema", None)
    expected = {field.name for field in fields(ActivatedOcclusionIdentity)}
    if schema == "wam_activated_occlusion_identity_v2":
        _require_exact_mapping(value, expected, label="occlusion_identity")
    elif schema == "wam_activated_occlusion_identity_v1":
        legacy_expected = expected.difference({
            "handeye_capability_schema",
            "handeye_calibration_tier",
            "handeye_policy_sha256",
        })
        _require_exact_mapping(
            value, legacy_expected, label="legacy occlusion_identity",
        )
        if not isinstance(handeye, ActivatedHandEye):
            raise ProductionRuntimeActivationError(
                "provisional hand-eye requires policy-bound occlusion identity v2"
            )
        value.update({
            "handeye_capability_schema": handeye.capability_schema,
            "handeye_calibration_tier": handeye.calibration_tier,
            "handeye_policy_sha256": handeye.policy_sha256,
        })
    else:
        raise ProductionRuntimeActivationError(
            "occlusion_identity artifact schema is unsupported"
        )
    return ActivatedOcclusionIdentity(**value)


def _build_robot_contract(
    pair: OpenArmPairActivation,
    value: object,
) -> RobotStateContract:
    bounds = _require_exact_mapping(
        value, _BOUNDS_KEYS, label="robot_state_bounds",
    )
    return RobotStateContract(
        side=pair.side,
        controller_mode=pair.controller_mode,
        joint_names=pair.joint_names,
        joint_position_limits_rad=tuple(
            tuple(limit) for limit in bounds["joint_position_limits_rad"]
        ),
        leader_can_interface=pair.leader.can_interface,
        follower_can_interface=pair.follower.can_interface,
        urdf_sha256=pair.urdf_sha256,
        joint_mapper_sha256=pair.joint_mapper_sha256,
        camera_profile_sha256=pair.camera_profile_sha256,
        timing_artifact_sha256=pair.timing_artifact_sha256,
        pair_contract_sha256=pair.pair_contract_sha256,
        controller_binary_sha256=pair.controller_binary_sha256,
        max_abs_joint_velocity_rad_s=bounds["max_abs_joint_velocity_rad_s"],
        gripper_motor_position_limits_rad=tuple(
            bounds["gripper_motor_position_limits_rad"]
        ),
        buffer_duration_ns=bounds["buffer_duration_ns"],
        max_packets=bounds["max_packets"],
        max_gap_ns=bounds["max_gap_ns"],
        max_staleness_ns=bounds["max_staleness_ns"],
        max_exposure_interval_ns=bounds["max_exposure_interval_ns"],
        controller_source_manifest_sha256=(
            pair.controller_source_manifest_sha256
        ),
        follower_config_sha256=pair.follower_config_sha256,
        openarm_sdk_sha256=pair.openarm_sdk_sha256,
    )


def load_production_runtime_assembly(
    activation_manifest: object,
    *,
    expected_activation_sha256: str,
    point_names: Sequence[str],
    expected_camera_id: str,
    expected_capture_mode: Mapping[str, object],
) -> ProductionRuntimeAssembly:
    """Load every pinned artifact and assemble the production dependencies.

    The returned telemetry receiver is deliberately deferred and stopped.  A
    trusted launcher must prepare the socket, capture/attach the real controller
    process attestation, and only then start reception.
    """

    path, manifest, activation_sha256 = _load_activation_manifest(
        activation_manifest, expected_sha256=expected_activation_sha256,
    )
    artifacts = _require_exact_mapping(
        manifest["artifacts"], set(_ARTIFACT_NAMES), label="activation artifacts",
    )
    loaded: Dict[str, Tuple[Dict[str, Any], str]] = {}
    for name in sorted(_ARTIFACT_NAMES):
        loaded[name] = _load_artifact(name, artifacts[name])

    pair_entry = _require_exact_mapping(
        manifest["openarm_pair"], _PAIR_KEYS, label="openarm_pair",
    )
    pair_dir = _immutable_path(
        pair_entry["release_directory"],
        label="OpenArm pair release",
        directory=True,
    )
    try:
        pair = load_activated_openarm_pair(
            pair_dir,
            expected_contract_sha256=_require_sha256(
                pair_entry["contract_sha256"],
                label="OpenArm pair contract SHA-256",
            ),
        )
        robot_contract = _build_robot_contract(
            pair, manifest["robot_state_bounds"],
        )
        camera_profile = _camera_profile(loaded["camera_profile"][0])
        alignment_sha256 = loaded["alignment"][1]
        camera_profile_sha256 = loaded["camera_profile"][1]
        camera_latency_sha256 = loaded["camera_latency"][1]
        if (
            camera_profile["camera_id"] != expected_camera_id
            or camera_profile["alignment_sha256"] != alignment_sha256
            or camera_profile_sha256 != pair.camera_profile_sha256
            or camera_latency_sha256 != pair.timing_artifact_sha256
        ):
            raise ValueError(
                "camera profile/alignment/timing do not match the activated pair"
            )
        camera_contract = CameraExposureContract.from_artifact(
            loaded["camera_latency"][0],
            artifact_sha256=camera_latency_sha256,
            camera_profile_sha256=camera_profile_sha256,
            alignment_sha256=alignment_sha256,
            expected_camera_id=expected_camera_id,
            expected_capture_mode=expected_capture_mode,
        )
        intrinsics_values = camera_profile["intrinsics"]
        projection_kwargs = {
            "camera_id": camera_profile["camera_id"],
            "camera_profile_sha256": camera_profile_sha256,
            "width": intrinsics_values["width"],
            "height": intrinsics_values["height"],
            "fx": intrinsics_values["fx"],
            "fy": intrinsics_values["fy"],
            "cx": intrinsics_values["cx"],
            "cy": intrinsics_values["cy"],
            "min_projection_depth_m": camera_profile[
                "min_projection_depth_m"
            ],
        }
        mask_projection = CameraProjectionProfile.create(**projection_kwargs)
        runtime_projection = ActivatedCameraProjection.create(**projection_kwargs)
        if (
            mask_projection.camera_projection_sha256
            != runtime_projection.camera_projection_sha256
        ):
            raise ValueError("camera projection implementations disagree")

        geometry = _build_geometry(loaded["collision_geometry"][0])
        tool = _build_tool(loaded["tool_geometry"][0])
        handeye = _build_handeye(
            loaded["handeye"][0],
            artifact_sha256=loaded["handeye"][1],
        )
        uncertainty = _build_uncertainty(
            loaded["robot_mask_uncertainty"][0]
        )
        back_provider = _build_back_mask(loaded["back_mask"][0])
        depth_profile = _build_depth_profile(loaded["depth_noise"][0])
        patient_config = _build_config(
            loaded["patient_pose"][0],
            schema="wam_metric_patient_pose_activation_v1",
            config_type=PatientPoseProviderConfig,
            label="patient_pose",
        )
        occlusion_config = _build_config(
            loaded["occlusion_runtime"][0],
            schema="wam_occlusion_runtime_activation_v1",
            config_type=OcclusionRuntimeConfig,
            label="occlusion_runtime",
        )
        _validate_handeye_runtime_bound(handeye, occlusion_config)
        identity = _build_identity(
            loaded["occlusion_identity"][0], handeye=handeye,
        )
        patient_pose_sha256 = loaded["patient_pose"][1]
        patient_pose_release_id = "patient_pose-" + patient_pose_sha256[:20]
        runtime_profile = ActivatedOcclusionRuntimeProfile.create(
            config=occlusion_config,
            patient_motion_artifact_sha256=patient_pose_sha256,
            handeye_artifact_sha256=handeye.handeye_sha256,
            camera_latency_artifact_sha256=camera_latency_sha256,
            camera_projection_artifact_sha256=(
                runtime_projection.camera_projection_sha256
            ),
            tool_geometry_artifact_sha256=tool.tool_sha256,
        )
        expected_identity = {
            "camera_id": expected_camera_id,
            "camera_profile_sha256": camera_profile_sha256,
            "geometry_release_id": geometry.geometry_release_id,
            "geometry_sha256": geometry.geometry_sha256,
            "tool_release_id": tool.tool_release_id,
            "tool_sha256": tool.tool_sha256,
            "handeye_release_id": handeye.handeye_release_id,
            "handeye_sha256": handeye.handeye_sha256,
            "handeye_capability_schema": handeye.capability_schema,
            "handeye_calibration_tier": handeye.calibration_tier,
            "handeye_policy_sha256": handeye.policy_sha256,
            "camera_latency_sha256": camera_latency_sha256,
            "camera_projection_sha256": (
                runtime_projection.camera_projection_sha256
            ),
            "occlusion_profile_sha256": (
                uncertainty.occlusion_profile_sha256
            ),
            "occlusion_runtime_profile_sha256": (
                runtime_profile.occlusion_runtime_profile_sha256
            ),
            "alignment_sha256": alignment_sha256,
            "joint_mapper_sha256": pair.joint_mapper_sha256,
            "pair_contract_sha256": pair.pair_contract_sha256,
            "controller_binary_sha256": pair.controller_binary_sha256,
            "patient_pose_release_id": patient_pose_release_id,
            "patient_pose_sha256": patient_pose_sha256,
            "depth_noise_model_sha256": depth_profile.semantic_sha256,
            "back_mask_release_id": back_provider.config.release_id,
            "back_mask_sha256": back_provider.config.semantic_sha256,
            "max_camera_to_exposure_age_ns": (
                camera_contract.latency_ns
                + camera_contract.latency_uncertainty_ns
            ),
            "max_exposure_interval_ns": robot_contract.max_exposure_interval_ns,
        }
        for name, expected in expected_identity.items():
            if getattr(identity, name) != expected:
                raise ValueError(
                    f"activated occlusion identity mismatches {name}"
                )
        if geometry.urdf_sha256 != pair.urdf_sha256:
            raise ValueError("official geometry URDF does not match OpenArm pair")
        if depth_profile.calibration_artifact_sha256 == depth_profile.calibration_complete_sha256:
            raise ValueError("depth calibration artifact and COMPLETE identities collide")

        robot_mask_provider = ProductionRobotMaskProvider(
            activated_identity=identity,
            geometry=geometry,
            tool=tool,
            handeye=handeye,
            projection=mask_projection,
            uncertainty_profile=uncertainty,
        )
        patient_pose_provider = MetricPatientPoseProvider(
            Intrinsics(
                fx=intrinsics_values["fx"],
                fy=intrinsics_values["fy"],
                cx=intrinsics_values["cx"],
                cy=intrinsics_values["cy"],
                width=intrinsics_values["width"],
                height=intrinsics_values["height"],
                depth_scale=intrinsics_values["depth_scale_mm"],
            ),
            camera_id=expected_camera_id,
            camera_profile_sha256=camera_profile_sha256,
            alignment_sha256=alignment_sha256,
            patient_pose_release_id=patient_pose_release_id,
            patient_pose_sha256=patient_pose_sha256,
            depth_noise_profile=depth_profile,
            back_mask_release_id=back_provider.config.release_id,
            back_mask_sha256=back_provider.config.semantic_sha256,
            config=patient_config,
        )
        occlusion_runtime = OcclusionRuntime(
            point_names,
            occlusion_config,
            activated_identity=identity,
            camera_projection=runtime_projection,
            runtime_profile=runtime_profile,
        )
        coordinator = ProductionRobotCaptureCoordinator(
            contract=robot_contract,
            camera_exposure_contract=camera_contract,
            socket_path=pair.telemetry_socket,
            runtime_dir=Path(pair.telemetry_socket).parent,
            deferred_controller_attestation=True,
        )
    except ProductionRuntimeActivationError:
        raise
    except Exception as exc:
        raise ProductionRuntimeActivationError(
            f"production runtime activation is invalid: {exc}"
        ) from exc

    return ProductionRuntimeAssembly(
        activation_path=str(path),
        activation_sha256=activation_sha256,
        openarm_pair=pair,
        camera_exposure_contract=camera_contract,
        handeye=handeye,
        activated_identity=identity,
        robot_capture_coordinator=coordinator,
        robot_mask_provider=robot_mask_provider,
        back_mask_provider=back_provider,
        patient_pose_provider=patient_pose_provider,
        occlusion_runtime=occlusion_runtime,
    )


__all__ = [
    "ACTIVATION_COMPLETION_SCHEMA",
    "ACTIVATION_FILE",
    "ACTIVATION_SCHEMA",
    "ARTIFACT_COMPLETION_SCHEMA",
    "COMPLETE_MARKER",
    "ProductionRuntimeActivationError",
    "ProductionRuntimeAssembly",
    "STAGING_MARKER",
    "load_production_runtime_assembly",
]
