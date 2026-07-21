"""Bounded independent posterior-back segmentation from aligned metric depth.

This provider is deliberately independent of the ten-point landmark model.  A
same-frame operator decision authorizes a bounded person box; aligned metric
depth then selects one deterministic foreground component inside that box.  A
missing authorization or insufficient depth produces an explicit ``NO_BACK``
packet.  Cross-frame, malformed, or over-budget input raises and therefore
fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Optional, Tuple

import cv2
import numpy as np

from ausc.perception.robot.frame_evidence_bundle import ImmutableCameraFrame
from ausc.perception.tracking.back_mask_packet import BackMaskPacket


OPERATOR_CONFIRMED_BACK_GATE_SCHEMA = "wam_operator_confirmed_back_gate_v1"
POSTERIOR_BACK_MASK_CONFIG_SCHEMA = "wam_posterior_back_mask_config_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_FRAME_PIXELS_HARD = 4_194_304
_MAX_MORPHOLOGY_RADIUS_PX_HARD = 8
_MAX_CONNECTED_COMPONENTS_HARD = 4_096


class PosteriorBackMaskProviderError(ValueError):
    """Input cannot authorize a production-candidate posterior back mask."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PosteriorBackMaskProviderError(
            "posterior back-mask evidence is not canonical JSON"
        ) from exc


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _float_token(value: object, *, label: str) -> str:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise PosteriorBackMaskProviderError(f"{label} must be finite")
    return float(value).hex()


def _validate_person_box(
    value: object, *, frame_shape_hw: Tuple[int, int], required: bool,
) -> Optional[Tuple[int, int, int, int]]:
    if value is None:
        if required:
            raise PosteriorBackMaskProviderError(
                "confirmed posterior decision requires a person box"
            )
        return None
    if not isinstance(value, tuple) or len(value) != 4:
        raise PosteriorBackMaskProviderError(
            "person box must be an immutable four-integer tuple"
        )
    if any(type(coordinate) is not int for coordinate in value):
        raise PosteriorBackMaskProviderError(
            "person box coordinates must be integer values"
        )
    x0, y0, x1, y1 = value
    if x1 <= x0 or y1 <= y0:
        raise PosteriorBackMaskProviderError("person box must have positive area")
    height, width = frame_shape_hw
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise PosteriorBackMaskProviderError("person box exceeds frame bounds")
    return (x0, y0, x1, y1)


def _gate_payload(
    *,
    frame_index: int,
    camera_host_monotonic_ns: int,
    rgb_sha256: str,
    frame_shape_hw: Tuple[int, int],
    posterior_present: bool,
    operator_confirmed: bool,
    person_box_xyxy: Optional[Tuple[int, int, int, int]],
) -> Mapping[str, object]:
    return {
        "schema": OPERATOR_CONFIRMED_BACK_GATE_SCHEMA,
        "frame_index": frame_index,
        "camera_host_monotonic_ns": camera_host_monotonic_ns,
        "rgb_sha256": rgb_sha256,
        "frame_shape_hw": list(frame_shape_hw),
        "posterior_present": posterior_present,
        "operator_confirmed": operator_confirmed,
        "person_box_xyxy": (None if person_box_xyxy is None else list(person_box_xyxy)),
    }


@dataclass(frozen=True)
class OperatorConfirmedBackGate:
    """One operator decision cryptographically bound to one RGB frame."""

    schema: str
    frame_index: int
    camera_host_monotonic_ns: int
    rgb_sha256: str
    frame_shape_hw: Tuple[int, int]
    posterior_present: bool
    operator_confirmed: bool
    person_box_xyxy: Optional[Tuple[int, int, int, int]]
    decision_id: str

    @classmethod
    def for_frame(
        cls,
        *,
        camera_frame: ImmutableCameraFrame,
        posterior_present: bool,
        operator_confirmed: bool,
        person_box_xyxy: Optional[Tuple[int, int, int, int]],
    ) -> "OperatorConfirmedBackGate":
        if not isinstance(camera_frame, ImmutableCameraFrame):
            raise PosteriorBackMaskProviderError(
                "operator gate requires an immutable camera frame"
            )
        if type(posterior_present) is not bool:
            raise PosteriorBackMaskProviderError(
                "posterior_present must be an exact boolean"
            )
        if type(operator_confirmed) is not bool:
            raise PosteriorBackMaskProviderError(
                "operator_confirmed must be an exact boolean"
            )
        shape = tuple(int(value) for value in camera_frame.frame.shape[:2])
        box = _validate_person_box(
            person_box_xyxy,
            frame_shape_hw=shape,
            required=posterior_present and operator_confirmed,
        )
        payload = _gate_payload(
            frame_index=camera_frame.index,
            camera_host_monotonic_ns=camera_frame.camera_host_monotonic_ns,
            rgb_sha256=camera_frame.rgb_sha256,
            frame_shape_hw=shape,
            posterior_present=posterior_present,
            operator_confirmed=operator_confirmed,
            person_box_xyxy=box,
        )
        return cls(
            schema=OPERATOR_CONFIRMED_BACK_GATE_SCHEMA,
            frame_index=camera_frame.index,
            camera_host_monotonic_ns=camera_frame.camera_host_monotonic_ns,
            rgb_sha256=camera_frame.rgb_sha256,
            frame_shape_hw=shape,
            posterior_present=posterior_present,
            operator_confirmed=operator_confirmed,
            person_box_xyxy=box,
            decision_id="operator_back_gate-" + _semantic_sha256(payload),
        )

    def __post_init__(self) -> None:
        if self.schema != OPERATOR_CONFIRMED_BACK_GATE_SCHEMA:
            raise PosteriorBackMaskProviderError(
                "operator back gate schema is unsupported"
            )
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise PosteriorBackMaskProviderError("operator gate frame index is invalid")
        if (
            type(self.camera_host_monotonic_ns) is not int
            or self.camera_host_monotonic_ns <= 0
        ):
            raise PosteriorBackMaskProviderError(
                "operator gate host timestamp is invalid"
            )
        if (
            not isinstance(self.rgb_sha256, str)
            or _SHA256_RE.fullmatch(self.rgb_sha256) is None
        ):
            raise PosteriorBackMaskProviderError("operator gate RGB SHA-256 is invalid")
        if (
            not isinstance(self.frame_shape_hw, tuple)
            or len(self.frame_shape_hw) != 2
            or any(
                type(value) is not int or value <= 0 for value in self.frame_shape_hw
            )
        ):
            raise PosteriorBackMaskProviderError("operator gate frame shape is invalid")
        if type(self.posterior_present) is not bool:
            raise PosteriorBackMaskProviderError(
                "posterior_present must be an exact boolean"
            )
        if type(self.operator_confirmed) is not bool:
            raise PosteriorBackMaskProviderError(
                "operator_confirmed must be an exact boolean"
            )
        box = _validate_person_box(
            self.person_box_xyxy,
            frame_shape_hw=self.frame_shape_hw,
            required=self.posterior_present and self.operator_confirmed,
        )
        payload = _gate_payload(
            frame_index=self.frame_index,
            camera_host_monotonic_ns=self.camera_host_monotonic_ns,
            rgb_sha256=self.rgb_sha256,
            frame_shape_hw=self.frame_shape_hw,
            posterior_present=self.posterior_present,
            operator_confirmed=self.operator_confirmed,
            person_box_xyxy=box,
        )
        expected = "operator_back_gate-" + _semantic_sha256(payload)
        if self.decision_id != expected:
            raise PosteriorBackMaskProviderError(
                "operator gate decision ID does not bind its values"
            )


def _config_payload(
    *,
    minimum_depth_m: float,
    maximum_depth_m: float,
    depth_band_m: float,
    mad_scale: float,
    maximum_adaptive_depth_band_m: float,
    central_seed_fraction: float,
    minimum_valid_depth_pixels: int,
    minimum_valid_depth_fraction: float,
    minimum_component_pixels: int,
    morphology_radius_px: int,
    maximum_frame_pixels: int,
    maximum_box_pixels: int,
    maximum_connected_components: int,
) -> Mapping[str, object]:
    return {
        "schema": POSTERIOR_BACK_MASK_CONFIG_SCHEMA,
        "algorithm": {
            "name": "aligned_metric_depth_central_median_component",
            "version": 1,
            "landmark_dependency": "forbidden",
            "connectivity": 8,
            "morphology_order": "open_then_close",
            "component_selection": "largest_then_lowest_label",
            "output_support": "strictly_inside_operator_person_box",
        },
        "metric_depth": {
            "minimum_depth_m": _float_token(minimum_depth_m, label="minimum depth"),
            "maximum_depth_m": _float_token(maximum_depth_m, label="maximum depth"),
            "depth_band_m": _float_token(depth_band_m, label="depth band"),
            "mad_scale": _float_token(mad_scale, label="MAD scale"),
            "maximum_adaptive_depth_band_m": _float_token(
                maximum_adaptive_depth_band_m, label="maximum adaptive depth band",
            ),
            "central_seed_fraction": _float_token(
                central_seed_fraction, label="central seed fraction"
            ),
            "minimum_valid_depth_pixels": minimum_valid_depth_pixels,
            "minimum_valid_depth_fraction": _float_token(
                minimum_valid_depth_fraction, label="minimum valid depth fraction",
            ),
        },
        "component": {
            "minimum_component_pixels": minimum_component_pixels,
            "morphology_radius_px": morphology_radius_px,
            "maximum_connected_components": maximum_connected_components,
        },
        "memory": {
            "maximum_frame_pixels": maximum_frame_pixels,
            "maximum_box_pixels": maximum_box_pixels,
        },
    }


@dataclass(frozen=True)
class PosteriorBackMaskConfig:
    """Content-addressed algorithm and fixed runtime work budget."""

    minimum_depth_m: float
    maximum_depth_m: float
    depth_band_m: float
    mad_scale: float
    maximum_adaptive_depth_band_m: float
    central_seed_fraction: float
    minimum_valid_depth_pixels: int
    minimum_valid_depth_fraction: float
    minimum_component_pixels: int
    morphology_radius_px: int
    maximum_frame_pixels: int
    maximum_box_pixels: int
    maximum_connected_components: int
    semantic_sha256: str
    release_id: str

    @classmethod
    def create(
        cls,
        *,
        minimum_depth_m: float = 0.25,
        maximum_depth_m: float = 3.0,
        depth_band_m: float = 0.08,
        mad_scale: float = 3.0,
        maximum_adaptive_depth_band_m: float = 0.18,
        central_seed_fraction: float = 0.4,
        minimum_valid_depth_pixels: int = 256,
        minimum_valid_depth_fraction: float = 0.15,
        minimum_component_pixels: int = 512,
        morphology_radius_px: int = 2,
        maximum_frame_pixels: int = 2_097_152,
        maximum_box_pixels: int = 1_048_576,
        maximum_connected_components: int = 1_024,
    ) -> "PosteriorBackMaskConfig":
        payload = _config_payload(
            minimum_depth_m=minimum_depth_m,
            maximum_depth_m=maximum_depth_m,
            depth_band_m=depth_band_m,
            mad_scale=mad_scale,
            maximum_adaptive_depth_band_m=maximum_adaptive_depth_band_m,
            central_seed_fraction=central_seed_fraction,
            minimum_valid_depth_pixels=minimum_valid_depth_pixels,
            minimum_valid_depth_fraction=minimum_valid_depth_fraction,
            minimum_component_pixels=minimum_component_pixels,
            morphology_radius_px=morphology_radius_px,
            maximum_frame_pixels=maximum_frame_pixels,
            maximum_box_pixels=maximum_box_pixels,
            maximum_connected_components=maximum_connected_components,
        )
        digest = _semantic_sha256(payload)
        return cls(
            minimum_depth_m=minimum_depth_m,
            maximum_depth_m=maximum_depth_m,
            depth_band_m=depth_band_m,
            mad_scale=mad_scale,
            maximum_adaptive_depth_band_m=maximum_adaptive_depth_band_m,
            central_seed_fraction=central_seed_fraction,
            minimum_valid_depth_pixels=minimum_valid_depth_pixels,
            minimum_valid_depth_fraction=minimum_valid_depth_fraction,
            minimum_component_pixels=minimum_component_pixels,
            morphology_radius_px=morphology_radius_px,
            maximum_frame_pixels=maximum_frame_pixels,
            maximum_box_pixels=maximum_box_pixels,
            maximum_connected_components=maximum_connected_components,
            semantic_sha256=digest,
            release_id=f"back_mask-{digest[:20]}",
        )

    def __post_init__(self) -> None:
        numeric = (
            (self.minimum_depth_m, "minimum depth"),
            (self.maximum_depth_m, "maximum depth"),
            (self.depth_band_m, "depth band"),
            (self.mad_scale, "MAD scale"),
            (self.maximum_adaptive_depth_band_m, "maximum adaptive depth band",),
            (self.central_seed_fraction, "central seed fraction"),
            (self.minimum_valid_depth_fraction, "minimum valid depth fraction",),
        )
        for value, label in numeric:
            _float_token(value, label=label)
        integer = (
            (self.minimum_valid_depth_pixels, "minimum valid depth pixels"),
            (self.minimum_component_pixels, "minimum component pixels"),
            (self.morphology_radius_px, "morphology radius"),
            (self.maximum_frame_pixels, "maximum frame pixels"),
            (self.maximum_box_pixels, "maximum box pixels"),
            (self.maximum_connected_components, "maximum connected components",),
        )
        for value, label in integer:
            if type(value) is not int:
                raise PosteriorBackMaskProviderError(f"{label} must be integer")
        if not 0 <= self.morphology_radius_px <= _MAX_MORPHOLOGY_RADIUS_PX_HARD:
            raise PosteriorBackMaskProviderError(
                "morphology radius exceeds the fixed safety bound"
            )
        if (
            self.minimum_depth_m < 0.05
            or self.maximum_depth_m > 5.0
            or self.minimum_depth_m >= self.maximum_depth_m
            or not 0.001 <= self.depth_band_m <= 0.5
            or not 0.0 <= self.mad_scale <= 10.0
            or self.maximum_adaptive_depth_band_m < self.depth_band_m
            or self.maximum_adaptive_depth_band_m > 1.0
            or not 0.1 <= self.central_seed_fraction <= 1.0
            or self.minimum_valid_depth_pixels <= 0
            or not 0.0 < self.minimum_valid_depth_fraction <= 1.0
            or self.minimum_component_pixels <= 0
            or not 1 <= self.maximum_frame_pixels <= _MAX_FRAME_PIXELS_HARD
            or not 1 <= self.maximum_box_pixels <= self.maximum_frame_pixels
            or self.minimum_valid_depth_pixels > self.maximum_box_pixels
            or self.minimum_component_pixels > self.maximum_box_pixels
            or not 1
            <= self.maximum_connected_components
            <= _MAX_CONNECTED_COMPONENTS_HARD
        ):
            raise PosteriorBackMaskProviderError(
                "posterior back-mask configuration exceeds a safety bound"
            )
        payload = _config_payload(
            minimum_depth_m=self.minimum_depth_m,
            maximum_depth_m=self.maximum_depth_m,
            depth_band_m=self.depth_band_m,
            mad_scale=self.mad_scale,
            maximum_adaptive_depth_band_m=(self.maximum_adaptive_depth_band_m),
            central_seed_fraction=self.central_seed_fraction,
            minimum_valid_depth_pixels=self.minimum_valid_depth_pixels,
            minimum_valid_depth_fraction=self.minimum_valid_depth_fraction,
            minimum_component_pixels=self.minimum_component_pixels,
            morphology_radius_px=self.morphology_radius_px,
            maximum_frame_pixels=self.maximum_frame_pixels,
            maximum_box_pixels=self.maximum_box_pixels,
            maximum_connected_components=self.maximum_connected_components,
        )
        digest = _semantic_sha256(payload)
        if self.semantic_sha256 != digest:
            raise PosteriorBackMaskProviderError(
                "posterior back-mask semantic SHA-256 does not bind its values"
            )
        if self.release_id != f"back_mask-{digest[:20]}":
            raise PosteriorBackMaskProviderError(
                "posterior back-mask release ID does not bind its values"
            )


class IndependentPosteriorBackMaskProvider:
    """Generate one bounded posterior mask without consulting landmarks."""

    def __init__(self, *, config: PosteriorBackMaskConfig) -> None:
        if not isinstance(config, PosteriorBackMaskConfig):
            raise PosteriorBackMaskProviderError(
                "posterior back-mask configuration is required"
            )
        config.__post_init__()
        self.config = config

    def _empty_packet(
        self,
        camera_frame: ImmutableCameraFrame,
        gate: OperatorConfirmedBackGate,
    ) -> BackMaskPacket:
        return BackMaskPacket.build(
            camera_frame=camera_frame,
            mask=np.zeros(camera_frame.frame.shape[:2], dtype=np.bool_),
            back_mask_release_id=self.config.release_id,
            back_mask_model_sha256=self.config.semantic_sha256,
            operator_gate_decision_id=gate.decision_id,
        )

    def _validate_same_frame(
        self, *, camera_frame: ImmutableCameraFrame, gate: OperatorConfirmedBackGate,
    ) -> None:
        if not isinstance(camera_frame, ImmutableCameraFrame):
            raise PosteriorBackMaskProviderError(
                "posterior mask requires an immutable camera frame"
            )
        if not isinstance(gate, OperatorConfirmedBackGate):
            raise PosteriorBackMaskProviderError(
                "operator-confirmed posterior gate is required"
            )
        camera_frame.__post_init__()
        gate.__post_init__()
        shape = tuple(int(value) for value in camera_frame.frame.shape[:2])
        if (
            gate.frame_index != camera_frame.index
            or gate.camera_host_monotonic_ns != camera_frame.camera_host_monotonic_ns
            or gate.rgb_sha256 != camera_frame.rgb_sha256
            or gate.frame_shape_hw != shape
        ):
            raise PosteriorBackMaskProviderError(
                "operator gate is not bound to the same camera frame"
            )
        frame_pixels = shape[0] * shape[1]
        if frame_pixels > self.config.maximum_frame_pixels:
            raise PosteriorBackMaskProviderError(
                "camera frame pixel count exceeds the fixed memory budget"
            )

    def _build(
        self, *, camera_frame: ImmutableCameraFrame, gate: OperatorConfirmedBackGate,
    ) -> BackMaskPacket:
        self._validate_same_frame(camera_frame=camera_frame, gate=gate)
        if not gate.posterior_present or not gate.operator_confirmed:
            return self._empty_packet(camera_frame, gate)

        box = gate.person_box_xyxy
        if box is None:
            raise PosteriorBackMaskProviderError(
                "confirmed posterior gate omitted the person box"
            )
        x0, y0, x1, y1 = box
        box_pixels = (x1 - x0) * (y1 - y0)
        if box_pixels > self.config.maximum_box_pixels:
            raise PosteriorBackMaskProviderError(
                "person box pixel count exceeds the fixed work budget"
            )

        raw_depth = camera_frame.depth[y0:y1, x0:x1]
        depth_m = (
            raw_depth.astype(np.float64, copy=True)
            * float(camera_frame.depth_scale)
            / 1000.0
        )
        valid = (
            (raw_depth > 0)
            & np.isfinite(depth_m)
            & (depth_m >= self.config.minimum_depth_m)
            & (depth_m <= self.config.maximum_depth_m)
        )
        valid_count = int(np.count_nonzero(valid))
        required_valid = max(
            self.config.minimum_valid_depth_pixels,
            int(math.ceil(box_pixels * self.config.minimum_valid_depth_fraction)),
        )
        if valid_count < required_valid:
            return self._empty_packet(camera_frame, gate)

        box_height, box_width = raw_depth.shape
        seed_height = max(
            1, int(math.floor(box_height * self.config.central_seed_fraction))
        )
        seed_width = max(
            1, int(math.floor(box_width * self.config.central_seed_fraction))
        )
        seed_y0 = (box_height - seed_height) // 2
        seed_x0 = (box_width - seed_width) // 2
        seed_valid = valid[
            seed_y0 : seed_y0 + seed_height, seed_x0 : seed_x0 + seed_width,
        ]
        seed_depth = depth_m[
            seed_y0 : seed_y0 + seed_height, seed_x0 : seed_x0 + seed_width,
        ][seed_valid]
        minimum_seed_pixels = max(
            3, min(self.config.minimum_valid_depth_pixels, box_pixels) // 4
        )
        if int(seed_depth.size) < minimum_seed_pixels:
            return self._empty_packet(camera_frame, gate)

        central_depth_m = float(np.median(seed_depth))
        median_absolute_deviation_m = float(
            np.median(np.abs(seed_depth - central_depth_m))
        )
        adaptive_band_m = min(
            self.config.maximum_adaptive_depth_band_m,
            max(
                self.config.depth_band_m,
                1.4826 * self.config.mad_scale * median_absolute_deviation_m,
            ),
        )
        candidate = valid & (np.abs(depth_m - central_depth_m) <= adaptive_band_m)

        candidate_u8 = np.ascontiguousarray(candidate, dtype=np.uint8)
        radius = self.config.morphology_radius_px
        if radius > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            candidate_u8 = cv2.morphologyEx(
                candidate_u8, cv2.MORPH_OPEN, kernel, iterations=1
            )
            candidate_u8 = cv2.morphologyEx(
                candidate_u8, cv2.MORPH_CLOSE, kernel, iterations=1
            )
            # Closing may fill a nearer robot/tool-shaped hole.  Keep morphology
            # subordinate to the original metric-depth foreground predicate.
            candidate_u8 &= candidate.astype(np.uint8, copy=False)

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            candidate_u8, connectivity=8, ltype=cv2.CV_32S,
        )
        component_count = int(count) - 1
        if component_count > self.config.maximum_connected_components:
            raise PosteriorBackMaskProviderError(
                "connected-component count exceeds the fixed work budget"
            )
        if component_count <= 0:
            return self._empty_packet(camera_frame, gate)
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_offset = int(np.argmax(areas))
        if int(areas[best_offset]) < self.config.minimum_component_pixels:
            return self._empty_packet(camera_frame, gate)

        roi_mask = labels == (best_offset + 1)
        mask = np.zeros(camera_frame.frame.shape[:2], dtype=np.bool_)
        mask[y0:y1, x0:x1] = roi_mask
        return BackMaskPacket.build(
            camera_frame=camera_frame,
            mask=mask,
            back_mask_release_id=self.config.release_id,
            back_mask_model_sha256=self.config.semantic_sha256,
            operator_gate_decision_id=gate.decision_id,
        )

    def build(
        self, *, camera_frame: ImmutableCameraFrame, gate: OperatorConfirmedBackGate,
    ) -> BackMaskPacket:
        """Build one packet or raise on malformed/cross-frame/over-budget input."""

        try:
            return self._build(camera_frame=camera_frame, gate=gate)
        except PosteriorBackMaskProviderError:
            raise
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
            cv2.error,
            MemoryError,
        ) as exc:
            raise PosteriorBackMaskProviderError(
                "posterior back-mask generation failed closed"
            ) from exc


__all__ = [
    "IndependentPosteriorBackMaskProvider",
    "OPERATOR_CONFIRMED_BACK_GATE_SCHEMA",
    "OperatorConfirmedBackGate",
    "POSTERIOR_BACK_MASK_CONFIG_SCHEMA",
    "PosteriorBackMaskConfig",
    "PosteriorBackMaskProviderError",
]
