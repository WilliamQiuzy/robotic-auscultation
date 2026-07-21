"""Fail-closed geometry authorization for patient-relative point tracking.

The production gate never accepts caller-supplied ``depth == observed_z`` as
evidence.  Depth samples, back/robot masks, exact point arrays, uncertainty
terms, and an independent same-frame patient-model prediction are bound into an
immutable evidence digest before evaluation.

An ``accepted`` result remains a software decision against an activated
profile.  It is not a substitute for physical camera, hand-eye, or robot
commissioning.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from threading import Lock
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

import numpy as np

from .experimental_visibility_geometry import (
    DecisionSurface,
    VisibilityGeometryResult,
    evaluate_visibility_geometry,
)
from .geometry_channel_packet import (
    INDEPENDENT_PATIENT_MODEL_CHANNEL,
    PERCEPTION_OBSERVATION_CHANNEL,
    GeometryChannelPacket,
    geometry_channels_are_independent,
)

if TYPE_CHECKING:
    from .patient_frame_evidence import PatientFrameEvidence


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MATHEMATICAL_SUPPORT_FLOOR = 4
_CHI2_3D_95 = 7.814727903251179
_PROFILE_SCHEMA = "wam_production_geometry_profile_v3"
_EVIDENCE_SCHEMA = "wam_production_geometry_evidence_v2"
_PRODUCTION_EVIDENCE_SCHEMA = "wam_production_bound_geometry_evidence_v1"
_DECISION_FIELDS = tuple(DecisionSurface.__dataclass_fields__)
_MAX_SEEN_PRODUCTION_ATTEMPTS = 4096
_MAX_TRACKED_PRODUCTION_EPISODES = 256

PRODUCTION_DECISION_SURFACE = DecisionSurface(
    min_relative_geometry_weight=0.05,
    max_depth_consistency_sigma=3.0,
    min_second_spatial_eigenvalue_m2=2.0e-4,
    min_hull_area_m2=1.0e-3,
    min_spread_m=8.0e-2,
    max_condition_number=40.0,
    max_residual_m=4.0e-3,
    max_target_leverage=4.0,
    max_target_error_p95_m=1.0e-2,
)


def _finite_positive(value: object) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _float_token(value: object) -> str:
    """Stable exact token for a numeric semantic parameter."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("semantic numeric value must be a real number")
    return float(value).hex()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray, *, dtype=None) -> str:
    array = np.asarray(value, dtype=dtype)
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _profile_payload(
    *,
    camera_profile_sha256: str,
    camera_projection_sha256: str,
    depth_noise_model_sha256: str,
    independent_model_sha256: str,
    lateral_noise_std_m: float,
    axial_noise_std_m: float,
    robust_delta_m: float,
    max_scale_deviation_relative: float,
    depth_sample_window_px: int,
    min_depth_m: float,
    max_depth_m: float,
    min_depth_patch_valid_pixels: int,
    min_depth_patch_valid_fraction: float,
    max_depth_patch_mad_m: float,
    max_depth_patch_p90_p10_m: float,
    min_back_mask_support_fraction: float,
    max_independent_model_innovation_m: float,
    max_independent_model_innovation_sigma: float,
    decision_surface: DecisionSurface,
) -> Mapping[str, Any]:
    return {
        "schema": _PROFILE_SCHEMA,
        "identities": {
            "camera_profile_sha256": str(camera_profile_sha256),
            "camera_projection_sha256": str(camera_projection_sha256),
            "depth_noise_model_sha256": str(depth_noise_model_sha256),
            "independent_model_sha256": str(independent_model_sha256),
        },
        "numeric_parameters": {
            "lateral_noise_std_m": _float_token(lateral_noise_std_m),
            "axial_noise_std_m": _float_token(axial_noise_std_m),
            "robust_delta_m": _float_token(robust_delta_m),
            "max_scale_deviation_relative": _float_token(
                max_scale_deviation_relative
            ),
            "depth_sample_window_px": int(depth_sample_window_px),
            "min_depth_m": _float_token(min_depth_m),
            "max_depth_m": _float_token(max_depth_m),
            "min_depth_patch_valid_pixels": int(
                min_depth_patch_valid_pixels
            ),
            "min_depth_patch_valid_fraction": _float_token(
                min_depth_patch_valid_fraction
            ),
            "max_depth_patch_mad_m": _float_token(max_depth_patch_mad_m),
            "max_depth_patch_p90_p10_m": _float_token(
                max_depth_patch_p90_p10_m
            ),
            "min_back_mask_support_fraction": _float_token(
                min_back_mask_support_fraction
            ),
            "max_independent_model_innovation_m": _float_token(
                max_independent_model_innovation_m
            ),
            "max_independent_model_innovation_sigma": _float_token(
                max_independent_model_innovation_sigma
            ),
            "decision_surface": {
                name: _float_token(getattr(decision_surface, name))
                for name in _DECISION_FIELDS
            },
        },
        # These constants change the acceptance calculation too, so they are
        # part of the activated semantic identity rather than only code trivia.
        "fixed_acceptance_contract": {
            "mathematical_support_floor": _MATHEMATICAL_SUPPORT_FLOOR,
            "target_p95_chi2_3d": _float_token(_CHI2_3D_95),
            "uncertainty_combination": (
                "pose_plus_rotated_target_plus_worst_reference_and_model"
            ),
        },
    }


def production_geometry_profile_semantic_sha256(
    *,
    camera_profile_sha256: str,
    camera_projection_sha256: str,
    depth_noise_model_sha256: str,
    independent_model_sha256: str,
    lateral_noise_std_m: float,
    axial_noise_std_m: float,
    robust_delta_m: float,
    max_scale_deviation_relative: float,
    depth_sample_window_px: int,
    min_depth_m: float,
    max_depth_m: float,
    min_depth_patch_valid_pixels: int,
    min_depth_patch_valid_fraction: float,
    max_depth_patch_mad_m: float,
    max_depth_patch_p90_p10_m: float,
    min_back_mask_support_fraction: float,
    max_independent_model_innovation_m: float,
    max_independent_model_innovation_sigma: float,
    decision_surface: DecisionSurface = PRODUCTION_DECISION_SURFACE,
    semantic_sha256: Optional[str] = None,
) -> str:
    """Return the exact activated semantic digest.

    ``semantic_sha256`` is accepted and ignored so a serialized profile mapping
    can be re-hashed directly without first deleting its claimed digest.
    """

    del semantic_sha256
    return _canonical_sha256(
        _profile_payload(
            camera_profile_sha256=camera_profile_sha256,
            camera_projection_sha256=camera_projection_sha256,
            depth_noise_model_sha256=depth_noise_model_sha256,
            independent_model_sha256=independent_model_sha256,
            lateral_noise_std_m=lateral_noise_std_m,
            axial_noise_std_m=axial_noise_std_m,
            robust_delta_m=robust_delta_m,
            max_scale_deviation_relative=max_scale_deviation_relative,
            depth_sample_window_px=depth_sample_window_px,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            min_depth_patch_valid_pixels=min_depth_patch_valid_pixels,
            min_depth_patch_valid_fraction=min_depth_patch_valid_fraction,
            max_depth_patch_mad_m=max_depth_patch_mad_m,
            max_depth_patch_p90_p10_m=max_depth_patch_p90_p10_m,
            min_back_mask_support_fraction=min_back_mask_support_fraction,
            max_independent_model_innovation_m=(
                max_independent_model_innovation_m
            ),
            max_independent_model_innovation_sigma=(
                max_independent_model_innovation_sigma
            ),
            decision_surface=decision_surface,
        )
    )


def _surface_is_no_looser(surface: DecisionSurface) -> bool:
    reference = PRODUCTION_DECISION_SURFACE
    return bool(
        surface.min_relative_geometry_weight
        >= reference.min_relative_geometry_weight
        and surface.max_depth_consistency_sigma
        <= reference.max_depth_consistency_sigma
        and surface.min_second_spatial_eigenvalue_m2
        >= reference.min_second_spatial_eigenvalue_m2
        and surface.min_hull_area_m2 >= reference.min_hull_area_m2
        and surface.min_spread_m >= reference.min_spread_m
        and surface.max_condition_number <= reference.max_condition_number
        and surface.max_residual_m <= reference.max_residual_m
        and surface.max_target_leverage <= reference.max_target_leverage
        and surface.max_target_error_p95_m
        <= reference.max_target_error_p95_m
    )


@dataclass(frozen=True)
class ProductionGeometryProfile:
    """Activated identities plus every numeric depth/acceptance parameter."""

    camera_profile_sha256: str
    camera_projection_sha256: str
    depth_noise_model_sha256: str
    independent_model_sha256: str
    lateral_noise_std_m: float
    axial_noise_std_m: float
    robust_delta_m: float
    max_scale_deviation_relative: float
    depth_sample_window_px: int
    min_depth_m: float
    max_depth_m: float
    min_depth_patch_valid_pixels: int
    min_depth_patch_valid_fraction: float
    max_depth_patch_mad_m: float
    max_depth_patch_p90_p10_m: float
    min_back_mask_support_fraction: float
    max_independent_model_innovation_m: float
    max_independent_model_innovation_sigma: float
    semantic_sha256: str
    decision_surface: DecisionSurface = PRODUCTION_DECISION_SURFACE

    def __post_init__(self) -> None:
        for digest, label in (
            (self.camera_profile_sha256, "camera profile"),
            (self.camera_projection_sha256, "camera projection"),
            (self.depth_noise_model_sha256, "depth noise model"),
            (self.independent_model_sha256, "independent model"),
            (self.semantic_sha256, "semantic profile"),
        ):
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(f"{label} SHA-256 is invalid")
        if not isinstance(self.decision_surface, DecisionSurface):
            raise ValueError("geometry profile decision surface is invalid")
        if (
            isinstance(self.depth_sample_window_px, bool)
            or type(self.depth_sample_window_px) is not int
            or self.depth_sample_window_px < 0
        ):
            raise ValueError("depth_sample_window_px must be a nonnegative int")
        if (
            isinstance(self.min_depth_patch_valid_pixels, bool)
            or type(self.min_depth_patch_valid_pixels) is not int
            or self.min_depth_patch_valid_pixels <= 0
        ):
            raise ValueError(
                "min_depth_patch_valid_pixels must be a positive int"
            )
        if not all(
            _finite_positive(value)
            for value in (
                self.lateral_noise_std_m,
                self.axial_noise_std_m,
                self.robust_delta_m,
                self.max_scale_deviation_relative,
                self.min_depth_m,
                self.max_depth_m,
                self.min_depth_patch_valid_fraction,
                self.max_depth_patch_mad_m,
                self.max_depth_patch_p90_p10_m,
                self.min_back_mask_support_fraction,
                self.max_independent_model_innovation_m,
                self.max_independent_model_innovation_sigma,
            )
        ):
            raise ValueError("geometry profile values must be finite and positive")
        if not all(
            _finite_positive(getattr(self.decision_surface, name))
            for name in _DECISION_FIELDS
        ):
            raise ValueError("geometry decision surface values are invalid")

        expected_semantic = production_geometry_profile_semantic_sha256(
            camera_profile_sha256=self.camera_profile_sha256,
            camera_projection_sha256=self.camera_projection_sha256,
            depth_noise_model_sha256=self.depth_noise_model_sha256,
            independent_model_sha256=self.independent_model_sha256,
            lateral_noise_std_m=self.lateral_noise_std_m,
            axial_noise_std_m=self.axial_noise_std_m,
            robust_delta_m=self.robust_delta_m,
            max_scale_deviation_relative=self.max_scale_deviation_relative,
            depth_sample_window_px=self.depth_sample_window_px,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            min_depth_patch_valid_pixels=(
                self.min_depth_patch_valid_pixels
            ),
            min_depth_patch_valid_fraction=(
                self.min_depth_patch_valid_fraction
            ),
            max_depth_patch_mad_m=self.max_depth_patch_mad_m,
            max_depth_patch_p90_p10_m=self.max_depth_patch_p90_p10_m,
            min_back_mask_support_fraction=self.min_back_mask_support_fraction,
            max_independent_model_innovation_m=(
                self.max_independent_model_innovation_m
            ),
            max_independent_model_innovation_sigma=(
                self.max_independent_model_innovation_sigma
            ),
            decision_surface=self.decision_surface,
        )
        if self.semantic_sha256 != expected_semantic:
            raise ValueError(
                "geometry profile semantic SHA-256 does not bind its values"
            )

        if float(self.max_scale_deviation_relative) > 0.02:
            raise ValueError("geometry profile scale allowance exceeds safety envelope")
        if self.depth_sample_window_px != 2:
            raise ValueError(
                "production depth sample window must be exactly two pixels"
            )
        if float(self.min_depth_m) < 0.05:
            raise ValueError("minimum depth is below safety envelope")
        if float(self.max_depth_m) > 3.0:
            raise ValueError("maximum depth exceeds safety envelope")
        if float(self.min_depth_m) >= float(self.max_depth_m):
            raise ValueError("minimum depth must be below maximum depth")
        max_patch_pixels = (2 * self.depth_sample_window_px + 1) ** 2
        if self.min_depth_patch_valid_pixels < 9:
            raise ValueError(
                "minimum valid depth pixel count is below safety envelope"
            )
        if self.min_depth_patch_valid_pixels > max_patch_pixels:
            raise ValueError(
                "minimum valid depth pixels exceed the configured patch"
            )
        if float(self.min_depth_patch_valid_fraction) < 0.60:
            raise ValueError("depth valid fraction is below safety envelope")
        if float(self.min_depth_patch_valid_fraction) > 1.0:
            raise ValueError("depth valid fraction exceeds one")
        if float(self.max_depth_patch_mad_m) > 0.004:
            raise ValueError("depth patch MAD exceeds safety envelope")
        if float(self.max_depth_patch_p90_p10_m) > 0.020:
            raise ValueError("depth patch robust span exceeds safety envelope")
        if float(self.min_back_mask_support_fraction) < 0.60:
            raise ValueError("back-mask support is below safety envelope")
        if float(self.min_back_mask_support_fraction) > 1.0:
            raise ValueError("back-mask support fraction exceeds one")
        if float(self.max_independent_model_innovation_m) > 0.010:
            raise ValueError("independent-model distance exceeds safety envelope")
        if float(self.max_independent_model_innovation_sigma) > 3.0:
            raise ValueError("independent-model sigma exceeds safety envelope")
        if not _surface_is_no_looser(self.decision_surface):
            raise ValueError("geometry decision surface exceeds safety envelope")


def _nested_float_tuple(value: np.ndarray) -> tuple:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        return tuple(float(item) for item in array)
    return tuple(_nested_float_tuple(item) for item in array)


def _evidence_payload(values: Mapping[str, Any]) -> Mapping[str, Any]:
    def float_list(items):
        return [_float_token(item) for item in items]

    def nested_float_list(items):
        output = []
        for item in items:
            if isinstance(item, (tuple, list)):
                output.append(nested_float_list(item))
            else:
                output.append(_float_token(item))
        return output

    return {
        "schema": _EVIDENCE_SCHEMA,
        "profile_semantic_sha256": values["profile_semantic_sha256"],
        "frame_id": values["frame_id"],
        "camera_frame_sha256": values["camera_frame_sha256"],
        "aligned_depth_sha256": values["aligned_depth_sha256"],
        "back_mask_sha256": values["back_mask_sha256"],
        "robot_mask_sha256": values["robot_mask_sha256"],
        "depth_scale_mm_per_unit": _float_token(
            values["depth_scale_mm_per_unit"]
        ),
        "sample_pixels_uv": nested_float_list(values["sample_pixels_uv"]),
        "sampled_depth_m": float_list(values["sampled_depth_m"]),
        "depth_patch_pixel_count": [
            int(item) for item in values["depth_patch_pixel_count"]
        ],
        "depth_patch_valid_pixel_count": [
            int(item) for item in values["depth_patch_valid_pixel_count"]
        ],
        "depth_patch_valid_fraction": float_list(
            values["depth_patch_valid_fraction"]
        ),
        "depth_patch_mad_m": float_list(values["depth_patch_mad_m"]),
        "depth_patch_p90_p10_m": float_list(
            values["depth_patch_p90_p10_m"]
        ),
        "back_mask_support_fraction": float_list(
            values["back_mask_support_fraction"]
        ),
        "robot_mask_intersection": [
            bool(item) for item in values["robot_mask_intersection"]
        ],
        "mask_confidence": float_list(values["mask_confidence"]),
        "reference_points_sha256": values["reference_points_sha256"],
        "observed_points_sha256": values["observed_points_sha256"],
        "target_reference_points_sha256": values[
            "target_reference_points_sha256"
        ],
        "independent_model_points_sha256": values[
            "independent_model_points_sha256"
        ],
        "independent_model_points_m": nested_float_list(
            values["independent_model_points_m"]
        ),
        "reference_covariances_m2": nested_float_list(
            values["reference_covariances_m2"]
        ),
        "target_covariances_m2": nested_float_list(
            values["target_covariances_m2"]
        ),
        "independent_model_covariances_m2": nested_float_list(
            values["independent_model_covariances_m2"]
        ),
    }


@dataclass(frozen=True)
class ProductionGeometryEvidence:
    """OFFLINE/TEST-ONLY raw evidence; never production authorization."""

    profile_semantic_sha256: str
    frame_id: str
    camera_frame_sha256: str
    aligned_depth_sha256: str
    back_mask_sha256: str
    robot_mask_sha256: str
    depth_scale_mm_per_unit: float
    sample_pixels_uv: tuple
    sampled_depth_m: tuple
    depth_patch_pixel_count: tuple
    depth_patch_valid_pixel_count: tuple
    depth_patch_valid_fraction: tuple
    depth_patch_mad_m: tuple
    depth_patch_p90_p10_m: tuple
    back_mask_support_fraction: tuple
    robot_mask_intersection: tuple
    mask_confidence: tuple
    reference_points_sha256: str
    observed_points_sha256: str
    target_reference_points_sha256: str
    independent_model_points_sha256: str
    independent_model_points_m: tuple
    reference_covariances_m2: tuple
    target_covariances_m2: tuple
    independent_model_covariances_m2: tuple
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("geometry evidence frame_id is missing")
        tuple_fields = (
            "sample_pixels_uv",
            "sampled_depth_m",
            "depth_patch_pixel_count",
            "depth_patch_valid_pixel_count",
            "depth_patch_valid_fraction",
            "depth_patch_mad_m",
            "depth_patch_p90_p10_m",
            "back_mask_support_fraction",
            "robot_mask_intersection",
            "mask_confidence",
            "independent_model_points_m",
            "reference_covariances_m2",
            "target_covariances_m2",
            "independent_model_covariances_m2",
        )

        def tuples_only(value: object) -> bool:
            if isinstance(value, tuple):
                return all(tuples_only(item) for item in value)
            return not isinstance(value, (list, np.ndarray, Mapping))

        if any(
            not isinstance(getattr(self, name), tuple)
            or not tuples_only(getattr(self, name))
            for name in tuple_fields
        ):
            raise ValueError(
                "geometry evidence content must use immutable tuples"
            )
        for digest in (
            self.profile_semantic_sha256,
            self.camera_frame_sha256,
            self.aligned_depth_sha256,
            self.back_mask_sha256,
            self.robot_mask_sha256,
            self.reference_points_sha256,
            self.observed_points_sha256,
            self.target_reference_points_sha256,
            self.independent_model_points_sha256,
            self.evidence_sha256,
        ):
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError("geometry evidence SHA-256 is invalid")
        values = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "evidence_sha256"
        }
        expected = _canonical_sha256(_evidence_payload(values))
        if self.evidence_sha256 != expected:
            raise ValueError("geometry evidence semantic SHA-256 mismatch")

    @property
    def production_authorized(self) -> bool:
        return False


def _production_evidence_payload(
    evidence: "ProductionBoundGeometryEvidence",
) -> Mapping[str, Any]:
    perception = evidence.perception_observation_packet
    independent = evidence.independent_model_packet
    return {
        "schema": _PRODUCTION_EVIDENCE_SCHEMA,
        "profile_semantic_sha256": evidence.profile_semantic_sha256,
        "frame_bundle_id": evidence.frame_bundle_id,
        "patient_frame_evidence_id": evidence.patient_frame_evidence_id,
        "episode_id": evidence.episode_id,
        "frame_index": evidence.frame_index,
        "camera_host_monotonic_ns": evidence.camera_host_monotonic_ns,
        "sequence": evidence.sequence,
        "point_names": list(evidence.point_names),
        "perception": {
            "channel": perception.channel,
            "producer_id": perception.producer_id,
            "release_id": perception.release_id,
            "model_sha256": perception.model_sha256,
            "inference_id": perception.inference_id,
            "input_rgb_sha256": perception.input_rgb_sha256,
            "input_depth_sha256": perception.input_depth_sha256,
            "point_names": list(perception.point_names),
            "points_sha256": perception.points_sha256,
            "covariances_sha256": perception.covariances_sha256,
        },
        "independent_model": {
            "channel": independent.channel,
            "producer_id": independent.producer_id,
            "release_id": independent.release_id,
            "model_sha256": independent.model_sha256,
            "inference_id": independent.inference_id,
            "input_rgb_sha256": independent.input_rgb_sha256,
            "input_depth_sha256": independent.input_depth_sha256,
            "point_names": list(independent.point_names),
            "points_sha256": independent.points_sha256,
            "covariances_sha256": independent.covariances_sha256,
        },
        "raw_geometry_evidence_sha256": (
            evidence.raw_geometry_evidence.evidence_sha256
        ),
    }


@dataclass(frozen=True)
class ProductionBoundGeometryEvidence:
    """Typed, frame-bound envelope required by production ``evaluate``."""

    schema: str
    profile_semantic_sha256: str
    frame_bundle_id: str
    patient_frame_evidence_id: str
    episode_id: str
    frame_index: int
    camera_host_monotonic_ns: int
    sequence: int
    point_names: tuple[str, ...]
    perception_observation_packet: GeometryChannelPacket
    independent_model_packet: GeometryChannelPacket
    raw_geometry_evidence: ProductionGeometryEvidence
    perception_inference_id: str
    independent_model_inference_id: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.schema != _PRODUCTION_EVIDENCE_SCHEMA:
            raise ValueError("production geometry evidence schema is unsupported")
        if (
            not isinstance(self.episode_id, str)
            or not self.episode_id.strip()
            or type(self.frame_index) is not int
            or self.frame_index < 0
            or type(self.camera_host_monotonic_ns) is not int
            or self.camera_host_monotonic_ns <= 0
            or type(self.sequence) is not int
            or self.sequence <= 0
        ):
            raise ValueError("production geometry metadata is invalid")
        if not isinstance(
            self.perception_observation_packet, GeometryChannelPacket
        ) or not isinstance(self.independent_model_packet, GeometryChannelPacket):
            raise ValueError("production geometry channel packets are missing")
        if not geometry_channels_are_independent(
            self.perception_observation_packet,
            self.independent_model_packet,
        ):
            raise ValueError(
                "independent geometry packet is not independent of perception"
            )
        if not isinstance(self.raw_geometry_evidence, ProductionGeometryEvidence):
            raise ValueError("raw geometry evidence is missing")
        try:
            self.perception_observation_packet.__post_init__()
            self.independent_model_packet.__post_init__()
            self.raw_geometry_evidence.__post_init__()
        except (TypeError, ValueError) as exc:
            raise ValueError("production geometry nested evidence is invalid") from exc
        if self.profile_semantic_sha256 != (
            self.raw_geometry_evidence.profile_semantic_sha256
        ):
            raise ValueError("production geometry profile binding mismatch")
        if (
            not isinstance(self.point_names, tuple)
            or not self.point_names
            or len(set(self.point_names)) != len(self.point_names)
            or any(not isinstance(name, str) or not name for name in self.point_names)
        ):
            raise ValueError("production geometry point order is invalid")
        if self.perception_inference_id != (
            self.perception_observation_packet.inference_id
        ) or self.independent_model_inference_id != (
            self.independent_model_packet.inference_id
        ):
            raise ValueError("production geometry inference identity mismatch")
        outer_context = (
            self.frame_bundle_id,
            self.patient_frame_evidence_id,
            self.episode_id,
            self.frame_index,
            self.camera_host_monotonic_ns,
            self.sequence,
        )
        for packet in (
            self.perception_observation_packet,
            self.independent_model_packet,
        ):
            if (
                packet.frame_bundle_id,
                packet.patient_frame_evidence_id,
                packet.episode_id,
                packet.frame_index,
                packet.camera_host_monotonic_ns,
                packet.sequence,
            ) != outer_context:
                raise ValueError("production geometry packet context mismatch")
        perception = self.perception_observation_packet
        independent = self.independent_model_packet
        raw = self.raw_geometry_evidence
        raw_independent_covariance_sha256 = _array_sha256(
            np.asarray(raw.independent_model_covariances_m2, dtype=np.float64),
            dtype=np.float64,
        )
        if (
            perception.channel != PERCEPTION_OBSERVATION_CHANNEL
            or independent.channel != INDEPENDENT_PATIENT_MODEL_CHANNEL
            or perception.point_names != self.point_names
            or independent.point_names != self.point_names
            or perception.input_rgb_sha256 != independent.input_rgb_sha256
            or perception.input_depth_sha256 != independent.input_depth_sha256
            or raw.frame_id != self.frame_bundle_id
            or raw.observed_points_sha256 != perception.points_sha256
            or raw.independent_model_points_sha256 != independent.points_sha256
            or raw_independent_covariance_sha256
            != independent.covariances_sha256
        ):
            raise ValueError("production geometry packet payload mismatch")
        expected = _canonical_sha256(_production_evidence_payload(self))
        if self.evidence_sha256 != expected:
            raise ValueError("production geometry evidence SHA-256 mismatch")

    @property
    def frame_id(self) -> str:
        return self.frame_bundle_id

    @property
    def production_authorized(self) -> bool:
        return True


@dataclass(frozen=True)
class ProductionGeometryDecision:
    accepted: bool
    decision: str
    failure_reasons: Tuple[str, ...]
    mathematical_support_count: int
    effective_support: float
    scale_audit: float
    max_target_error_p95_m: float
    max_independent_model_innovation_m: float
    max_independent_model_innovation_sigma: float
    camera_profile_sha256: str
    camera_projection_sha256: str
    depth_noise_model_sha256: str
    independent_model_sha256: str
    frame_id: str
    geometry_evidence_sha256: str
    target_combined_covariances_m2: Tuple[np.ndarray, ...]
    target_results: Tuple[VisibilityGeometryResult, ...]


def _scale_audit(reference: np.ndarray, observed: np.ndarray) -> float:
    reference_centered = reference - np.mean(reference, axis=0)
    observed_centered = observed - np.mean(observed, axis=0)
    denominator = float(np.sum(reference_centered**2))
    numerator = float(np.sum(observed_centered**2))
    if (
        not math.isfinite(denominator)
        or not math.isfinite(numerator)
        or denominator <= 0.0
        or numerator < 0.0
    ):
        return float("inf")
    return float(math.sqrt(numerator / denominator))


def _validated_covariances(
    value: object,
    shape: tuple,
    label: str,
) -> np.ndarray:
    if value is None:
        raise ValueError(f"{label} is required")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    symmetric = 0.5 * (array + np.swapaxes(array, -1, -2))
    if not np.allclose(array, symmetric, rtol=0.0, atol=1.0e-15):
        raise ValueError(f"{label} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if (
        not np.isfinite(eigenvalues).all()
        or np.any(eigenvalues < -1.0e-15)
        or np.any(eigenvalues[:, -1] <= 0.0)
    ):
        raise ValueError(f"{label} must be positive semidefinite and nonzero")
    return symmetric


class ProductionGeometryGate:
    """Evaluate requested targets against one activated profile and evidence."""

    def __init__(self, profile: ProductionGeometryProfile) -> None:
        if not isinstance(profile, ProductionGeometryProfile):
            raise TypeError("profile must be ProductionGeometryProfile")
        self.profile = profile
        self._production_replay_lock = Lock()
        self._last_production_context_by_episode: dict[
            str, tuple[int, int, int]
        ] = {}
        self._seen_production_attempts: OrderedDict[
            str, tuple[str, str]
        ] = OrderedDict()
        self._seen_production_inference_ids: dict[str, str] = {}
        self._last_accepted_camera_frame: Optional[tuple[int, str]] = None

    def bind_evidence(
        self,
        *,
        frame_id: str,
        camera_frame: np.ndarray,
        aligned_depth_frame: np.ndarray,
        depth_scale_mm_per_unit: float,
        sample_pixels_uv: np.ndarray,
        aligned_back_mask: np.ndarray,
        aligned_robot_mask: np.ndarray,
        reference_points_m: np.ndarray,
        observed_points_m: np.ndarray,
        target_reference_points_m: np.ndarray,
        mask_confidence: np.ndarray,
        independent_model_points_m: np.ndarray,
        reference_covariances_m2: np.ndarray,
        target_covariances_m2: np.ndarray,
        independent_model_covariances_m2: np.ndarray,
    ) -> ProductionGeometryEvidence:
        """OFFLINE/TEST ONLY: derive raw depth and mask diagnostics.

        Callers cannot inject a free ``depth_m`` array.  The evidence digest
        binds the full RGB/depth/mask frames, sampling pixels, exact point
        arrays, model prediction, and every covariance used by ``evaluate``.
        Production callers must use the typed production bind API instead;
        evidence returned here is rejected by :meth:`evaluate`.
        """

        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be a nonempty string")
        camera = np.asarray(camera_frame)
        depth_frame = np.asarray(aligned_depth_frame)
        if depth_frame.ndim != 2 or not np.issubdtype(
            depth_frame.dtype, np.number
        ):
            raise ValueError("aligned_depth_frame must be a numeric 2-D array")
        if camera.ndim < 2 or camera.shape[:2] != depth_frame.shape:
            raise ValueError("camera_frame must align exactly to depth")
        back_mask = np.asarray(aligned_back_mask)
        robot_mask = np.asarray(aligned_robot_mask)
        for label, mask in (
            ("aligned_back_mask", back_mask),
            ("aligned_robot_mask", robot_mask),
        ):
            if mask.dtype != np.bool_ or mask.shape != depth_frame.shape:
                raise ValueError(f"{label} must be an aligned bool mask")
        if not _finite_positive(depth_scale_mm_per_unit):
            raise ValueError("depth_scale_mm_per_unit must be finite and positive")

        reference = np.asarray(reference_points_m, dtype=np.float64)
        observed = np.asarray(observed_points_m, dtype=np.float64)
        targets = np.asarray(target_reference_points_m, dtype=np.float64)
        pixels = np.asarray(sample_pixels_uv, dtype=np.float64)
        confidence = np.asarray(mask_confidence, dtype=np.float64)
        model_points = np.asarray(independent_model_points_m, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1:] != (3,):
            raise ValueError("reference_points_m must have shape (N, 3)")
        count = len(reference)
        if observed.shape != reference.shape:
            raise ValueError("observed_points_m must match reference_points_m")
        if targets.ndim != 2 or targets.shape[1:] != (3,) or len(targets) == 0:
            raise ValueError("target_reference_points_m must have shape (M, 3)")
        if pixels.shape != (count, 2):
            raise ValueError("sample_pixels_uv must have shape (N, 2)")
        if confidence.shape != (count,):
            raise ValueError("mask_confidence must have shape (N,)")
        if model_points.shape != reference.shape:
            raise ValueError(
                "independent_model_points_m must match reference_points_m"
            )

        reference_covariances = _validated_covariances(
            reference_covariances_m2,
            (count, 3, 3),
            "reference_covariances_m2",
        )
        target_covariances = _validated_covariances(
            target_covariances_m2,
            (len(targets), 3, 3),
            "target_covariances_m2",
        )
        model_covariances = _validated_covariances(
            independent_model_covariances_m2,
            (count, 3, 3),
            "independent_model_covariances_m2",
        )

        height, width = depth_frame.shape
        window = self.profile.depth_sample_window_px
        sampled_depth = []
        patch_pixel_counts = []
        valid_pixel_counts = []
        valid_fractions = []
        depth_mads = []
        depth_robust_spans = []
        back_fractions = []
        robot_intersections = []
        for u, v in pixels:
            if not (math.isfinite(float(u)) and math.isfinite(float(v))):
                sampled_depth.append(float("nan"))
                patch_pixel_counts.append(0)
                valid_pixel_counts.append(0)
                valid_fractions.append(0.0)
                depth_mads.append(float("nan"))
                depth_robust_spans.append(float("nan"))
                back_fractions.append(0.0)
                robot_intersections.append(False)
                continue
            ui, vi = int(round(float(u))), int(round(float(v)))
            if not (0 <= ui < width and 0 <= vi < height):
                sampled_depth.append(float("nan"))
                patch_pixel_counts.append(0)
                valid_pixel_counts.append(0)
                valid_fractions.append(0.0)
                depth_mads.append(float("nan"))
                depth_robust_spans.append(float("nan"))
                back_fractions.append(0.0)
                robot_intersections.append(False)
                continue
            y0, y1 = max(0, vi - window), min(height, vi + window + 1)
            x0, x1 = max(0, ui - window), min(width, ui + window + 1)
            depth_patch = (
                depth_frame[y0:y1, x0:x1].astype(np.float64)
                * float(depth_scale_mm_per_unit)
                / 1000.0
            )
            back_patch = back_mask[y0:y1, x0:x1]
            robot_patch = robot_mask[y0:y1, x0:x1]
            back_fractions.append(float(np.mean(back_patch)))
            robot_intersections.append(bool(np.any(robot_patch)))
            eligible = (
                back_patch
                & ~robot_patch
                & np.isfinite(depth_patch)
                & (depth_patch > self.profile.min_depth_m)
                & (depth_patch < self.profile.max_depth_m)
            )
            valid = depth_patch[eligible]
            patch_pixels = int(depth_patch.size)
            valid_pixels = int(valid.size)
            median = float(np.median(valid)) if valid.size else float("nan")
            if valid.size:
                mad = float(np.median(np.abs(valid - median)))
                p10, p90 = np.quantile(valid, (0.10, 0.90))
                robust_span = float(p90 - p10)
            else:
                mad = float("nan")
                robust_span = float("nan")
            sampled_depth.append(median)
            patch_pixel_counts.append(patch_pixels)
            valid_pixel_counts.append(valid_pixels)
            valid_fractions.append(
                valid_pixels / float(patch_pixels) if patch_pixels else 0.0
            )
            depth_mads.append(mad)
            depth_robust_spans.append(robust_span)

        values = {
            "profile_semantic_sha256": self.profile.semantic_sha256,
            "frame_id": frame_id,
            "camera_frame_sha256": _array_sha256(camera),
            "aligned_depth_sha256": _array_sha256(depth_frame),
            "back_mask_sha256": _array_sha256(back_mask, dtype=np.bool_),
            "robot_mask_sha256": _array_sha256(robot_mask, dtype=np.bool_),
            "depth_scale_mm_per_unit": float(depth_scale_mm_per_unit),
            "sample_pixels_uv": _nested_float_tuple(pixels),
            "sampled_depth_m": tuple(sampled_depth),
            "depth_patch_pixel_count": tuple(patch_pixel_counts),
            "depth_patch_valid_pixel_count": tuple(valid_pixel_counts),
            "depth_patch_valid_fraction": tuple(valid_fractions),
            "depth_patch_mad_m": tuple(depth_mads),
            "depth_patch_p90_p10_m": tuple(depth_robust_spans),
            "back_mask_support_fraction": tuple(back_fractions),
            "robot_mask_intersection": tuple(robot_intersections),
            "mask_confidence": _nested_float_tuple(confidence),
            "reference_points_sha256": _array_sha256(
                reference, dtype=np.float64
            ),
            "observed_points_sha256": _array_sha256(
                observed, dtype=np.float64
            ),
            "target_reference_points_sha256": _array_sha256(
                targets, dtype=np.float64
            ),
            "independent_model_points_sha256": _array_sha256(
                model_points, dtype=np.float64
            ),
            "independent_model_points_m": _nested_float_tuple(model_points),
            "reference_covariances_m2": _nested_float_tuple(
                reference_covariances
            ),
            "target_covariances_m2": _nested_float_tuple(target_covariances),
            "independent_model_covariances_m2": _nested_float_tuple(
                model_covariances
            ),
        }
        values["evidence_sha256"] = _canonical_sha256(_evidence_payload(values))
        return ProductionGeometryEvidence(**values)

    def bind_production_evidence(
        self,
        *,
        patient_frame_evidence: PatientFrameEvidence,
        episode_id: str,
        sequence: int,
        point_names: tuple[str, ...],
        perception_observation_packet: GeometryChannelPacket,
        independent_model_packet: GeometryChannelPacket,
        sample_pixels_uv: np.ndarray,
        reference_points_m: np.ndarray,
        target_reference_points_m: np.ndarray,
        mask_confidence: np.ndarray,
        reference_covariances_m2: np.ndarray,
        target_covariances_m2: np.ndarray,
    ) -> ProductionBoundGeometryEvidence:
        """Bind independently identified model outputs to one patient frame."""

        from .patient_frame_evidence import (
            PatientFrameEvidence,
            PatientFrameEvidenceError,
        )

        if not isinstance(patient_frame_evidence, PatientFrameEvidence):
            raise ValueError("patient frame evidence is required")
        try:
            patient_frame_evidence.__post_init__()
        except PatientFrameEvidenceError as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("production geometry episode identity is missing")
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("production geometry sequence is invalid")
        if not isinstance(perception_observation_packet, GeometryChannelPacket):
            raise ValueError("perception geometry packet is required")
        if not isinstance(independent_model_packet, GeometryChannelPacket):
            raise ValueError("independent geometry packet is required")
        if perception_observation_packet.channel != PERCEPTION_OBSERVATION_CHANNEL:
            raise ValueError("perception geometry packet channel is invalid")
        if independent_model_packet.channel != INDEPENDENT_PATIENT_MODEL_CHANNEL:
            raise ValueError("independent geometry packet channel is invalid")
        if independent_model_packet.model_sha256 != (
            self.profile.independent_model_sha256
        ):
            raise ValueError("independent model SHA-256 does not match profile")
        if not geometry_channels_are_independent(
            perception_observation_packet,
            independent_model_packet,
        ):
            raise ValueError(
                "independent geometry packet is not independent of perception"
            )

        camera = patient_frame_evidence.frame_bundle.camera_frame
        if camera.camera_profile_sha256 != self.profile.camera_profile_sha256:
            raise ValueError(
                "patient frame camera profile does not match geometry profile"
            )
        expected_context = (
            patient_frame_evidence.frame_bundle.bundle_id,
            patient_frame_evidence.evidence_id,
            episode_id,
            camera.index,
            camera.camera_host_monotonic_ns,
            sequence,
            camera.rgb_sha256,
            camera.depth_sha256,
        )
        for packet, label in (
            (perception_observation_packet, "perception"),
            (independent_model_packet, "independent model"),
        ):
            observed_context = (
                packet.frame_bundle_id,
                packet.patient_frame_evidence_id,
                packet.episode_id,
                packet.frame_index,
                packet.camera_host_monotonic_ns,
                packet.sequence,
                packet.input_rgb_sha256,
                packet.input_depth_sha256,
            )
            if observed_context != expected_context:
                raise ValueError(f"{label} packet is not bound to this frame")

        names = tuple(point_names)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
            or perception_observation_packet.point_names != names
            or independent_model_packet.point_names != names
        ):
            raise ValueError("geometry packet point names/order do not match")
        reference = np.asarray(reference_points_m, dtype=np.float64)
        targets = np.asarray(target_reference_points_m, dtype=np.float64)
        if reference.shape != perception_observation_packet.points_m.shape:
            raise ValueError("reference points do not match geometry packet order")

        raw_evidence = self.bind_evidence(
            frame_id=patient_frame_evidence.frame_bundle.bundle_id,
            camera_frame=patient_frame_evidence.rgb,
            aligned_depth_frame=patient_frame_evidence.aligned_depth,
            depth_scale_mm_per_unit=float(camera.depth_scale),
            sample_pixels_uv=sample_pixels_uv,
            aligned_back_mask=patient_frame_evidence.back_mask,
            aligned_robot_mask=patient_frame_evidence.robot_mask,
            reference_points_m=reference,
            observed_points_m=perception_observation_packet.points_m,
            target_reference_points_m=targets,
            mask_confidence=mask_confidence,
            independent_model_points_m=independent_model_packet.points_m,
            reference_covariances_m2=reference_covariances_m2,
            target_covariances_m2=target_covariances_m2,
            independent_model_covariances_m2=(
                independent_model_packet.covariances_m2
            ),
        )
        values = {
            "schema": _PRODUCTION_EVIDENCE_SCHEMA,
            "profile_semantic_sha256": self.profile.semantic_sha256,
            "frame_bundle_id": patient_frame_evidence.frame_bundle.bundle_id,
            "patient_frame_evidence_id": patient_frame_evidence.evidence_id,
            "episode_id": episode_id,
            "frame_index": camera.index,
            "camera_host_monotonic_ns": camera.camera_host_monotonic_ns,
            "sequence": sequence,
            "point_names": names,
            "perception_observation_packet": perception_observation_packet,
            "independent_model_packet": independent_model_packet,
            "raw_geometry_evidence": raw_evidence,
            "perception_inference_id": perception_observation_packet.inference_id,
            "independent_model_inference_id": independent_model_packet.inference_id,
        }
        provisional = object.__new__(ProductionBoundGeometryEvidence)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        values["evidence_sha256"] = _canonical_sha256(
            _production_evidence_payload(provisional)
        )
        return ProductionBoundGeometryEvidence(**values)

    def _production_progress_failure_reason(
        self,
        evidence: ProductionBoundGeometryEvidence,
    ) -> Optional[str]:
        """Return a temporal failure against accepted progress only.

        Callers hold ``_production_replay_lock``. Rejected geometry attempts
        deliberately do not advance either accepted watermark.
        """

        previous = self._last_production_context_by_episode.get(
            evidence.episode_id
        )
        if previous is not None:
            previous_sequence, previous_frame_index, previous_monotonic_ns = (
                previous
            )
            if evidence.sequence <= previous_sequence:
                return "production_geometry_sequence_not_monotonic"
            if (
                evidence.frame_index <= previous_frame_index
                or evidence.camera_host_monotonic_ns <= previous_monotonic_ns
            ):
                return "production_geometry_frame_not_monotonic"
        elif (
            len(self._last_production_context_by_episode)
            >= _MAX_TRACKED_PRODUCTION_EPISODES
        ):
            return "production_geometry_episode_capacity_exceeded"

        if self._last_accepted_camera_frame is not None:
            global_monotonic_ns, global_episode_id = (
                self._last_accepted_camera_frame
            )
            if (
                evidence.episode_id != global_episode_id
                and evidence.camera_host_monotonic_ns <= global_monotonic_ns
            ):
                return "production_geometry_frame_reused_across_episodes"
        return None

    def _failed(
        self,
        reasons: Tuple[str, ...],
        *,
        support_count: int,
        scale_audit: float = float("inf"),
        target_results: Tuple[VisibilityGeometryResult, ...] = (),
        target_combined_covariances_m2: Tuple[np.ndarray, ...] = (),
        max_target_error_p95_m: float = float("inf"),
        max_independent_model_innovation_m: float = float("inf"),
        max_independent_model_innovation_sigma: float = float("inf"),
        evidence: Optional[ProductionGeometryEvidence] = None,
    ) -> ProductionGeometryDecision:
        effective = (
            min(result.effective_support for result in target_results)
            if target_results
            else 0.0
        )
        return ProductionGeometryDecision(
            accepted=False,
            decision="fail_closed",
            failure_reasons=tuple(reasons),
            mathematical_support_count=int(support_count),
            effective_support=float(effective),
            scale_audit=float(scale_audit),
            max_target_error_p95_m=float(max_target_error_p95_m),
            max_independent_model_innovation_m=float(
                max_independent_model_innovation_m
            ),
            max_independent_model_innovation_sigma=float(
                max_independent_model_innovation_sigma
            ),
            camera_profile_sha256=self.profile.camera_profile_sha256,
            camera_projection_sha256=self.profile.camera_projection_sha256,
            depth_noise_model_sha256=self.profile.depth_noise_model_sha256,
            independent_model_sha256=self.profile.independent_model_sha256,
            frame_id=evidence.frame_id if evidence is not None else "",
            geometry_evidence_sha256=(
                evidence.evidence_sha256 if evidence is not None else ""
            ),
            target_combined_covariances_m2=tuple(
                target_combined_covariances_m2
            ),
            target_results=tuple(target_results),
        )

    def evaluate(
        self,
        *,
        reference_points_m: np.ndarray,
        observed_points_m: np.ndarray,
        target_reference_points_m: np.ndarray,
        evidence: object,
        patient_frame_evidence: Optional[PatientFrameEvidence] = None,
        episode_id: Optional[str] = None,
        sequence: Optional[int] = None,
    ) -> ProductionGeometryDecision:
        """Authorize only typed evidence rebound to the caller's live context."""

        from .patient_frame_evidence import (
            PatientFrameEvidence,
            PatientFrameEvidenceError,
        )

        if not isinstance(evidence, ProductionBoundGeometryEvidence):
            return self._failed(
                ("production_bound_geometry_evidence_required",),
                support_count=0,
                evidence=(
                    evidence
                    if isinstance(evidence, ProductionGeometryEvidence)
                    else None
                ),
            )
        if not isinstance(patient_frame_evidence, PatientFrameEvidence):
            return self._failed(
                ("production_geometry_context_required",),
                support_count=0,
                evidence=evidence.raw_geometry_evidence,
            )
        if (
            not isinstance(episode_id, str)
            or not episode_id.strip()
            or type(sequence) is not int
            or sequence <= 0
        ):
            return self._failed(
                ("production_geometry_context_invalid",),
                support_count=0,
                evidence=evidence.raw_geometry_evidence,
            )
        try:
            evidence.__post_init__()
            patient_frame_evidence.__post_init__()
        except (PatientFrameEvidenceError, TypeError, ValueError):
            return self._failed(
                ("production_geometry_evidence_invalid",),
                support_count=0,
                evidence=evidence.raw_geometry_evidence,
            )
        camera = patient_frame_evidence.frame_bundle.camera_frame
        expected_context = (
            self.profile.semantic_sha256,
            patient_frame_evidence.frame_bundle.bundle_id,
            patient_frame_evidence.evidence_id,
            episode_id,
            camera.index,
            camera.camera_host_monotonic_ns,
            sequence,
        )
        evidence_context = (
            evidence.profile_semantic_sha256,
            evidence.frame_bundle_id,
            evidence.patient_frame_evidence_id,
            evidence.episode_id,
            evidence.frame_index,
            evidence.camera_host_monotonic_ns,
            evidence.sequence,
        )
        if expected_context != evidence_context:
            return self._failed(
                ("production_geometry_context_mismatch",),
                support_count=0,
                evidence=evidence.raw_geometry_evidence,
            )
        if evidence.independent_model_packet.model_sha256 != (
            self.profile.independent_model_sha256
        ):
            return self._failed(
                ("independent_model_identity_mismatch",),
                support_count=0,
                evidence=evidence.raw_geometry_evidence,
            )
        raw = evidence.raw_geometry_evidence
        if (
            raw.frame_id != patient_frame_evidence.frame_bundle.bundle_id
            or evidence.perception_observation_packet.input_rgb_sha256
            != camera.rgb_sha256
            or evidence.perception_observation_packet.input_depth_sha256
            != camera.depth_sha256
            or raw.camera_frame_sha256
            != _array_sha256(patient_frame_evidence.rgb)
            or raw.aligned_depth_sha256
            != _array_sha256(patient_frame_evidence.aligned_depth)
            or raw.back_mask_sha256
            != _array_sha256(patient_frame_evidence.back_mask, dtype=np.bool_)
            or raw.robot_mask_sha256
            != _array_sha256(patient_frame_evidence.robot_mask, dtype=np.bool_)
            or raw.depth_scale_mm_per_unit != float(camera.depth_scale)
        ):
            return self._failed(
                ("production_geometry_patient_frame_payload_mismatch",),
                support_count=0,
                evidence=raw,
            )

        with self._production_replay_lock:
            if (
                evidence.evidence_sha256 in self._seen_production_attempts
                or evidence.perception_inference_id
                in self._seen_production_inference_ids
                or evidence.independent_model_inference_id
                in self._seen_production_inference_ids
            ):
                return self._failed(
                    ("production_geometry_evidence_replayed",),
                    support_count=0,
                    evidence=evidence.raw_geometry_evidence,
                )
            progress_failure = self._production_progress_failure_reason(evidence)
            if progress_failure is not None:
                return self._failed(
                    (progress_failure,),
                    support_count=0,
                    evidence=evidence.raw_geometry_evidence,
                )

            inference_ids = (
                evidence.perception_inference_id,
                evidence.independent_model_inference_id,
            )
            self._seen_production_attempts[evidence.evidence_sha256] = (
                inference_ids
            )
            for inference_id in inference_ids:
                self._seen_production_inference_ids[
                    inference_id
                ] = evidence.evidence_sha256
            while (
                len(self._seen_production_attempts)
                > _MAX_SEEN_PRODUCTION_ATTEMPTS
            ):
                evicted_evidence_sha256, evicted_inference_ids = (
                    self._seen_production_attempts.popitem(last=False)
                )
                for inference_id in evicted_inference_ids:
                    if self._seen_production_inference_ids.get(inference_id) == (
                        evicted_evidence_sha256
                    ):
                        del self._seen_production_inference_ids[inference_id]

        diagnostic = self.evaluate_diagnostic(
            reference_points_m=reference_points_m,
            observed_points_m=observed_points_m,
            target_reference_points_m=target_reference_points_m,
            evidence=evidence.raw_geometry_evidence,
        )
        if diagnostic.accepted:
            with self._production_replay_lock:
                progress_failure = self._production_progress_failure_reason(
                    evidence
                )
                if progress_failure is not None:
                    return self._failed(
                        (progress_failure,),
                        support_count=0,
                        evidence=evidence.raw_geometry_evidence,
                    )
                self._last_production_context_by_episode[evidence.episode_id] = (
                    evidence.sequence,
                    evidence.frame_index,
                    evidence.camera_host_monotonic_ns,
                )
                self._last_accepted_camera_frame = (
                    evidence.camera_host_monotonic_ns,
                    evidence.episode_id,
                )
        return replace(
            diagnostic,
            decision=(
                "accepted_by_production_geometry_surface"
                if diagnostic.accepted
                else diagnostic.decision
            ),
            frame_id=evidence.frame_bundle_id,
            geometry_evidence_sha256=evidence.evidence_sha256,
        )

    def evaluate_diagnostic(
        self,
        *,
        reference_points_m: np.ndarray,
        observed_points_m: np.ndarray,
        target_reference_points_m: np.ndarray,
        evidence: Optional[ProductionGeometryEvidence],
    ) -> ProductionGeometryDecision:
        reference = np.asarray(reference_points_m, dtype=np.float64)
        observed = np.asarray(observed_points_m, dtype=np.float64)
        targets = np.asarray(target_reference_points_m, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1:] != (3,):
            raise ValueError("reference_points_m must have shape (N, 3)")
        if observed.shape != reference.shape:
            raise ValueError("observed_points_m must match reference_points_m")
        if targets.ndim != 2 or targets.shape[1:] != (3,) or len(targets) == 0:
            raise ValueError("target_reference_points_m must have shape (M, 3)")
        if evidence is None:
            return self._failed(
                ("missing_bound_geometry_evidence",),
                support_count=0,
            )
        if not isinstance(evidence, ProductionGeometryEvidence):
            return self._failed(
                ("invalid_bound_geometry_evidence",),
                support_count=0,
            )

        identities_match = bool(
            evidence.profile_semantic_sha256 == self.profile.semantic_sha256
            and evidence.reference_points_sha256
            == _array_sha256(reference, dtype=np.float64)
            and evidence.observed_points_sha256
            == _array_sha256(observed, dtype=np.float64)
            and evidence.target_reference_points_sha256
            == _array_sha256(targets, dtype=np.float64)
        )
        if not identities_match:
            return self._failed(
                ("geometry_evidence_identity_mismatch",),
                support_count=0,
                evidence=evidence,
            )

        patch_counts_are_integers = all(
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, np.integer))
            for values in (
                evidence.depth_patch_pixel_count,
                evidence.depth_patch_valid_pixel_count,
            )
            for value in values
        )
        if not patch_counts_are_integers:
            return self._failed(
                ("depth_patch_evidence_inconsistent",),
                support_count=0,
                evidence=evidence,
            )

        depth = np.asarray(evidence.sampled_depth_m, dtype=np.float64)
        patch_pixels = np.asarray(
            evidence.depth_patch_pixel_count, dtype=np.int64
        )
        valid_pixels = np.asarray(
            evidence.depth_patch_valid_pixel_count, dtype=np.int64
        )
        valid_fraction = np.asarray(
            evidence.depth_patch_valid_fraction, dtype=np.float64
        )
        depth_mad = np.asarray(evidence.depth_patch_mad_m, dtype=np.float64)
        depth_robust_span = np.asarray(
            evidence.depth_patch_p90_p10_m, dtype=np.float64
        )
        confidence = np.asarray(evidence.mask_confidence, dtype=np.float64)
        back_support = np.asarray(
            evidence.back_mask_support_fraction, dtype=np.float64
        )
        robot_intersection = np.asarray(
            evidence.robot_mask_intersection, dtype=np.bool_
        )
        model_points = np.asarray(
            evidence.independent_model_points_m, dtype=np.float64
        )
        reference_covariances = np.asarray(
            evidence.reference_covariances_m2, dtype=np.float64
        )
        target_covariances = np.asarray(
            evidence.target_covariances_m2, dtype=np.float64
        )
        model_covariances = np.asarray(
            evidence.independent_model_covariances_m2, dtype=np.float64
        )
        count = len(reference)
        evidence_shapes_ok = bool(
            depth.shape == (count,)
            and patch_pixels.shape == (count,)
            and valid_pixels.shape == (count,)
            and valid_fraction.shape == (count,)
            and depth_mad.shape == (count,)
            and depth_robust_span.shape == (count,)
            and confidence.shape == (count,)
            and back_support.shape == (count,)
            and robot_intersection.shape == (count,)
            and model_points.shape == reference.shape
            and reference_covariances.shape == (count, 3, 3)
            and target_covariances.shape == (len(targets), 3, 3)
            and model_covariances.shape == (count, 3, 3)
        )
        if not evidence_shapes_ok:
            return self._failed(
                ("geometry_evidence_shape_mismatch",),
                support_count=0,
                evidence=evidence,
            )

        expected_patch_pixels = (2 * self.profile.depth_sample_window_px + 1) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            expected_valid_fraction = valid_pixels.astype(np.float64) / patch_pixels
        has_valid_depth = valid_pixels > 0
        missing_depth_statistics = (
            ~has_valid_depth
            & np.isnan(depth)
            & np.isnan(depth_mad)
            & np.isnan(depth_robust_span)
        )
        present_depth_statistics = (
            has_valid_depth
            & np.isfinite(depth)
            & (depth > self.profile.min_depth_m)
            & (depth < self.profile.max_depth_m)
            & np.isfinite(depth_mad)
            & (depth_mad >= 0.0)
            & np.isfinite(depth_robust_span)
            & (depth_robust_span >= 0.0)
        )
        depth_patch_evidence_consistent = bool(
            np.all(patch_pixels == expected_patch_pixels)
            and np.all(valid_pixels >= 0)
            and np.all(valid_pixels <= patch_pixels)
            and np.isfinite(valid_fraction).all()
            and np.all(valid_fraction >= 0.0)
            and np.all(valid_fraction <= 1.0)
            and np.array_equal(valid_fraction, expected_valid_fraction)
            and np.all(missing_depth_statistics | present_depth_statistics)
        )
        if not depth_patch_evidence_consistent:
            return self._failed(
                ("depth_patch_evidence_inconsistent",),
                support_count=0,
                evidence=evidence,
            )

        finite_support = (
            np.isfinite(reference).all(axis=1)
            & np.isfinite(observed).all(axis=1)
            & np.isfinite(model_points).all(axis=1)
            & np.isfinite(depth)
            & np.isfinite(valid_fraction)
            & np.isfinite(depth_mad)
            & np.isfinite(depth_robust_span)
            & np.isfinite(confidence)
            & np.isfinite(back_support)
            & (depth > self.profile.min_depth_m)
            & (depth < self.profile.max_depth_m)
            & (patch_pixels == expected_patch_pixels)
            & (valid_pixels >= self.profile.min_depth_patch_valid_pixels)
            & (valid_pixels <= patch_pixels)
            & (
                valid_fraction
                >= self.profile.min_depth_patch_valid_fraction
            )
            & (depth_mad >= 0.0)
            & (depth_mad <= self.profile.max_depth_patch_mad_m)
            & (
                (depth_robust_span >= 0.0)
                & (
                    depth_robust_span
                    <= self.profile.max_depth_patch_p90_p10_m
                )
            )
            & (confidence > 0.0)
            & (confidence <= 1.0)
            & (
                back_support
                >= self.profile.min_back_mask_support_fraction
            )
            & ~robot_intersection
        )
        support_count = int(np.count_nonzero(finite_support))
        if support_count < _MATHEMATICAL_SUPPORT_FLOOR:
            return self._failed(
                ("insufficient_mathematical_support",),
                support_count=support_count,
                evidence=evidence,
            )
        if not np.isfinite(targets).all():
            return self._failed(
                ("non_finite_target",),
                support_count=support_count,
                evidence=evidence,
            )

        reference = reference[finite_support]
        observed = observed[finite_support]
        depth = depth[finite_support]
        confidence = confidence[finite_support]
        model_points = model_points[finite_support]
        reference_covariances = reference_covariances[finite_support]
        model_covariances = model_covariances[finite_support]

        scale = _scale_audit(reference, observed)
        scale_ok = bool(
            math.isfinite(scale)
            and abs(scale - 1.0)
            <= self.profile.max_scale_deviation_relative
        )

        innovations = observed - model_points
        innovation_distances = np.linalg.norm(innovations, axis=1)
        measurement_variance = max(
            self.profile.lateral_noise_std_m**2,
            self.profile.axial_noise_std_m**2,
        )
        innovation_sigmas = []
        for innovation, covariance in zip(
            innovations, model_covariances
        ):
            combined = covariance + np.eye(3) * measurement_variance
            try:
                precision = np.linalg.inv(combined)
                squared = float(innovation @ precision @ innovation)
            except np.linalg.LinAlgError:
                squared = float("inf")
            innovation_sigmas.append(
                math.sqrt(max(squared, 0.0))
                if math.isfinite(squared)
                else float("inf")
            )
        max_innovation_m = float(np.max(innovation_distances))
        max_innovation_sigma = float(np.max(innovation_sigmas))
        independent_model_ok = bool(
            math.isfinite(max_innovation_m)
            and math.isfinite(max_innovation_sigma)
            and max_innovation_m
            <= self.profile.max_independent_model_innovation_m
            and max_innovation_sigma
            <= self.profile.max_independent_model_innovation_sigma
        )

        lateral = np.full(
            support_count,
            float(self.profile.lateral_noise_std_m),
            dtype=np.float64,
        )
        axial = np.full(
            support_count,
            float(self.profile.axial_noise_std_m),
            dtype=np.float64,
        )
        results = tuple(
            evaluate_visibility_geometry(
                reference,
                observed,
                target,
                depth_m=depth,
                mask_confidence=confidence,
                lateral_noise_std_m=lateral,
                axial_noise_std_m=axial,
                robust_delta_m=float(self.profile.robust_delta_m),
                decision_surface=self.profile.decision_surface,
            )
            for target in targets
        )

        reference_uncertainty = float(
            np.max(np.linalg.eigvalsh(reference_covariances))
        )
        model_uncertainty = float(
            np.max(np.linalg.eigvalsh(model_covariances))
        )
        combined_covariances = []
        combined_p95 = []
        for result, target_covariance in zip(results, target_covariances):
            rotation = result.transform[:3, :3]
            propagated = np.asarray(
                result.propagated_covariance, dtype=np.float64
            )
            if (
                propagated.shape != (3, 3)
                or not np.isfinite(propagated).all()
            ):
                combined = np.full((3, 3), np.inf, dtype=np.float64)
                p95 = float("inf")
            else:
                combined = (
                    propagated
                    + rotation @ target_covariance @ rotation.T
                    + np.eye(3)
                    * (reference_uncertainty + model_uncertainty)
                )
                combined = 0.5 * (combined + combined.T)
                eigenvalues = np.linalg.eigvalsh(combined)
                if (
                    not np.isfinite(eigenvalues).all()
                    or np.any(eigenvalues < -1.0e-12)
                ):
                    p95 = float("inf")
                else:
                    p95 = float(
                        math.sqrt(
                            _CHI2_3D_95
                            * max(float(eigenvalues[-1]), 0.0)
                        )
                    )
            combined_covariances.append(combined)
            combined_p95.append(p95)
        max_combined_p95 = float(max(combined_p95))

        reasons = []
        if not scale_ok:
            reasons.append("scale_audit_failed")
        if not independent_model_ok:
            reasons.append("independent_model_innovation_above_surface")
        if not all(result.accepted for result in results):
            reasons.append("geometry_surface_rejected")
        if (
            not math.isfinite(max_combined_p95)
            or max_combined_p95
            > self.profile.decision_surface.max_target_error_p95_m
        ):
            reasons.append("target_uncertainty_above_surface")
        if reasons:
            return self._failed(
                tuple(reasons),
                support_count=support_count,
                scale_audit=scale,
                target_results=results,
                target_combined_covariances_m2=tuple(combined_covariances),
                max_target_error_p95_m=max_combined_p95,
                max_independent_model_innovation_m=max_innovation_m,
                max_independent_model_innovation_sigma=max_innovation_sigma,
                evidence=evidence,
            )
        return ProductionGeometryDecision(
            accepted=True,
            decision="diagnostic_geometry_surface_pass",
            failure_reasons=(),
            mathematical_support_count=support_count,
            effective_support=float(
                min(result.effective_support for result in results)
            ),
            scale_audit=float(scale),
            max_target_error_p95_m=max_combined_p95,
            max_independent_model_innovation_m=max_innovation_m,
            max_independent_model_innovation_sigma=max_innovation_sigma,
            camera_profile_sha256=self.profile.camera_profile_sha256,
            camera_projection_sha256=self.profile.camera_projection_sha256,
            depth_noise_model_sha256=self.profile.depth_noise_model_sha256,
            independent_model_sha256=self.profile.independent_model_sha256,
            frame_id=evidence.frame_id,
            geometry_evidence_sha256=evidence.evidence_sha256,
            target_combined_covariances_m2=tuple(combined_covariances),
            target_results=results,
        )


__all__ = [
    "PRODUCTION_DECISION_SURFACE",
    "ProductionBoundGeometryEvidence",
    "ProductionGeometryDecision",
    "ProductionGeometryEvidence",
    "ProductionGeometryGate",
    "ProductionGeometryProfile",
    "production_geometry_profile_semantic_sha256",
]
