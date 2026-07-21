"""Activated aligned-depth uncertainty contract for metric pose estimation.

The profile is deliberately camera/profile/alignment specific.  Its semantic
digest binds every identity and numeric coefficient used by the covariance
calculation, while independent calibration and COMPLETE-marker digests bind
the coefficients to an external commissioning artifact.  Constructing this
object proves structural integrity; it does not replace collecting that
artifact on the physical Lingbox camera.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Optional

import numpy as np


_SCHEMA = "wam_activated_aligned_depth_noise_profile_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PIXEL_QUANTIZATION_SIGMA_FLOOR_PX = 1.0 / math.sqrt(12.0)
_SQRT_TWO = math.sqrt(2.0)


def _finite_real(value: object) -> bool:
    return bool(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _float_token(value: object) -> str:
    if not _finite_real(value):
        raise ValueError("depth-noise semantic numeric value is invalid")
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


def _profile_payload(
    *,
    camera_id: str,
    camera_profile_sha256: str,
    alignment_sha256: str,
    calibration_artifact_sha256: str,
    calibration_complete_sha256: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale_mm: float,
    calibrated_min_depth_m: float,
    calibrated_max_depth_m: float,
    axial_sigma_floor_m: float,
    axial_sigma_quadratic_m_inv: float,
    pixel_quantization_sigma_px: float,
    alignment_sigma_u_px: float,
    alignment_sigma_v_px: float,
    alignment_sigma_depth_m: float,
    fit_residual_scale: float,
    uncertainty_expansion_factor: float,
    calibration_quantile: float,
    calibration_frame_count: int,
    calibration_sample_count: int,
    calibration_depth_bin_count: int,
) -> Mapping[str, Any]:
    return {
        "schema": _SCHEMA,
        "identities": {
            "camera_id": str(camera_id),
            "camera_profile_sha256": str(camera_profile_sha256),
            "alignment_sha256": str(alignment_sha256),
            "calibration_artifact_sha256": str(calibration_artifact_sha256),
            "calibration_complete_sha256": str(calibration_complete_sha256),
        },
        "camera_contract": {
            "width": int(width),
            "height": int(height),
            "fx": _float_token(fx),
            "fy": _float_token(fy),
            "cx": _float_token(cx),
            "cy": _float_token(cy),
            "depth_scale_mm": _float_token(depth_scale_mm),
        },
        "calibration_contract": {
            "calibrated_min_depth_m": _float_token(calibrated_min_depth_m),
            "calibrated_max_depth_m": _float_token(calibrated_max_depth_m),
            "calibration_quantile": _float_token(calibration_quantile),
            "calibration_frame_count": int(calibration_frame_count),
            "calibration_sample_count": int(calibration_sample_count),
            "calibration_depth_bin_count": int(calibration_depth_bin_count),
        },
        "uncertainty_parameters": {
            "axial_sigma_floor_m": _float_token(axial_sigma_floor_m),
            "axial_sigma_quadratic_m_inv": _float_token(axial_sigma_quadratic_m_inv),
            "pixel_quantization_sigma_px": _float_token(pixel_quantization_sigma_px),
            "alignment_sigma_u_px": _float_token(alignment_sigma_u_px),
            "alignment_sigma_v_px": _float_token(alignment_sigma_v_px),
            "alignment_sigma_depth_m": _float_token(alignment_sigma_depth_m),
            "fit_residual_scale": _float_token(fit_residual_scale),
            "uncertainty_expansion_factor": _float_token(uncertainty_expansion_factor),
        },
        "fixed_covariance_contract": {
            "axial_model": "hypot(floor_m,quadratic_m_inv*depth_m^2)",
            "point_model": "pinhole_jacobian_uvz",
            "pair_model": "R*C_reference*R.T+C_current+C_fit",
            "common_alignment_model": "reference_plus_current_not_averaged",
            "fit_model": "scaled_residual_not_averaged",
            "tangent_order": "rotation_patient_rad_then_translation_camera_m",
            "pixel_quantization_sigma_floor_px": _float_token(
                _PIXEL_QUANTIZATION_SIGMA_FLOOR_PX
            ),
            "reference_current_sigma_factor": _float_token(_SQRT_TWO),
        },
    }


def depth_noise_profile_semantic_sha256(
    *,
    camera_id: str,
    camera_profile_sha256: str,
    alignment_sha256: str,
    calibration_artifact_sha256: str,
    calibration_complete_sha256: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale_mm: float,
    calibrated_min_depth_m: float,
    calibrated_max_depth_m: float,
    axial_sigma_floor_m: float,
    axial_sigma_quadratic_m_inv: float,
    pixel_quantization_sigma_px: float,
    alignment_sigma_u_px: float,
    alignment_sigma_v_px: float,
    alignment_sigma_depth_m: float,
    fit_residual_scale: float,
    uncertainty_expansion_factor: float,
    calibration_quantile: float,
    calibration_frame_count: int,
    calibration_sample_count: int,
    calibration_depth_bin_count: int,
    semantic_sha256: Optional[str] = None,
) -> str:
    """Return the exact semantic digest for a serialized profile mapping."""

    del semantic_sha256
    return _canonical_sha256(
        _profile_payload(
            camera_id=camera_id,
            camera_profile_sha256=camera_profile_sha256,
            alignment_sha256=alignment_sha256,
            calibration_artifact_sha256=calibration_artifact_sha256,
            calibration_complete_sha256=calibration_complete_sha256,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale_mm=depth_scale_mm,
            calibrated_min_depth_m=calibrated_min_depth_m,
            calibrated_max_depth_m=calibrated_max_depth_m,
            axial_sigma_floor_m=axial_sigma_floor_m,
            axial_sigma_quadratic_m_inv=axial_sigma_quadratic_m_inv,
            pixel_quantization_sigma_px=pixel_quantization_sigma_px,
            alignment_sigma_u_px=alignment_sigma_u_px,
            alignment_sigma_v_px=alignment_sigma_v_px,
            alignment_sigma_depth_m=alignment_sigma_depth_m,
            fit_residual_scale=fit_residual_scale,
            uncertainty_expansion_factor=uncertainty_expansion_factor,
            calibration_quantile=calibration_quantile,
            calibration_frame_count=calibration_frame_count,
            calibration_sample_count=calibration_sample_count,
            calibration_depth_bin_count=calibration_depth_bin_count,
        )
    )


@dataclass(frozen=True)
class ActivatedDepthNoiseProfile:
    """Camera-specific, hash-bound aligned-depth uncertainty profile."""

    camera_id: str
    camera_profile_sha256: str
    alignment_sha256: str
    calibration_artifact_sha256: str
    calibration_complete_sha256: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_mm: float
    calibrated_min_depth_m: float
    calibrated_max_depth_m: float
    axial_sigma_floor_m: float
    axial_sigma_quadratic_m_inv: float
    pixel_quantization_sigma_px: float
    alignment_sigma_u_px: float
    alignment_sigma_v_px: float
    alignment_sigma_depth_m: float
    fit_residual_scale: float
    uncertainty_expansion_factor: float
    calibration_quantile: float
    calibration_frame_count: int
    calibration_sample_count: int
    calibration_depth_bin_count: int
    semantic_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("depth-noise camera identity is invalid")
        for digest, label in (
            (self.camera_profile_sha256, "camera profile"),
            (self.alignment_sha256, "alignment"),
            (self.calibration_artifact_sha256, "calibration artifact"),
            (self.calibration_complete_sha256, "calibration COMPLETE marker"),
            (self.semantic_sha256, "semantic profile"),
        ):
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(f"depth-noise {label} SHA-256 is invalid")
        for value, label in (
            (self.width, "width"),
            (self.height, "height"),
            (self.calibration_frame_count, "calibration frame count"),
            (self.calibration_sample_count, "calibration sample count"),
            (self.calibration_depth_bin_count, "calibration depth-bin count"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"depth-noise {label} is invalid")
        finite_values = (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.depth_scale_mm,
            self.calibrated_min_depth_m,
            self.calibrated_max_depth_m,
            self.axial_sigma_floor_m,
            self.axial_sigma_quadratic_m_inv,
            self.pixel_quantization_sigma_px,
            self.alignment_sigma_u_px,
            self.alignment_sigma_v_px,
            self.alignment_sigma_depth_m,
            self.fit_residual_scale,
            self.uncertainty_expansion_factor,
            self.calibration_quantile,
        )
        if not all(_finite_real(value) for value in finite_values):
            raise ValueError("depth-noise numeric parameter is invalid")
        if (
            self.fx <= 0.0
            or self.fy <= 0.0
            or self.depth_scale_mm <= 0.0
            or not 0.0 <= self.cx <= self.width - 1
            or not 0.0 <= self.cy <= self.height - 1
            or self.calibrated_min_depth_m < 0.05
            or self.calibrated_max_depth_m > 3.0
            or self.calibrated_min_depth_m >= self.calibrated_max_depth_m
            or self.axial_sigma_floor_m < 0.0005
            or self.axial_sigma_quadratic_m_inv <= 0.0
            or self.pixel_quantization_sigma_px < _PIXEL_QUANTIZATION_SIGMA_FLOOR_PX
            or self.alignment_sigma_u_px <= 0.0
            or self.alignment_sigma_v_px <= 0.0
            or self.alignment_sigma_depth_m <= 0.0
            or self.fit_residual_scale < 1.0
            or self.uncertainty_expansion_factor < 1.0
            or not 0.95 <= self.calibration_quantile < 1.0
            or self.calibration_frame_count < 50
            or self.calibration_sample_count < 1000
            or self.calibration_depth_bin_count < 5
        ):
            raise ValueError("depth-noise values are uncommissioned or nonconservative")
        expected = depth_noise_profile_semantic_sha256(
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "semantic_sha256"
            }
        )
        if self.semantic_sha256 != expected:
            raise ValueError("depth-noise semantic SHA-256 does not bind its values")

    def validate_for_camera(
        self,
        *,
        camera_id: str,
        camera_profile_sha256: str,
        alignment_sha256: str,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale_mm: float,
    ) -> None:
        """Fail closed unless the live aligned camera contract matches exactly."""

        identities = (
            (camera_id, self.camera_id, "camera identity"),
            (camera_profile_sha256, self.camera_profile_sha256, "camera profile",),
            (alignment_sha256, self.alignment_sha256, "alignment"),
        )
        for actual, expected, label in identities:
            if actual != expected:
                raise ValueError(f"depth-noise {label} does not match activation")
        integer_values = ((width, self.width), (height, self.height))
        if any(
            type(actual) is not int or actual != expected
            for actual, expected in integer_values
        ):
            raise ValueError("depth-noise camera dimensions do not match activation")
        numeric_values = (
            (fx, self.fx),
            (fy, self.fy),
            (cx, self.cx),
            (cy, self.cy),
            (depth_scale_mm, self.depth_scale_mm),
        )
        if any(
            not _finite_real(actual) or float(actual).hex() != float(expected).hex()
            for actual, expected in numeric_values
        ):
            raise ValueError("depth-noise camera intrinsics do not match activation")

    def measurement_covariances_m2(
        self, pixels: np.ndarray, depth_m: np.ndarray
    ) -> np.ndarray:
        """Return per-point 3-D covariance from aligned ``(u,v,z)`` evidence."""

        uv = np.asarray(pixels, dtype=np.float64)
        depth = np.asarray(depth_m, dtype=np.float64).reshape(-1)
        if (
            uv.ndim != 2
            or uv.shape[1:] != (2,)
            or len(uv) != len(depth)
            or len(uv) == 0
            or not np.isfinite(uv).all()
            or not np.isfinite(depth).all()
        ):
            raise ValueError("depth-noise measurement evidence is invalid")
        if (
            np.any(uv[:, 0] < 0.0)
            or np.any(uv[:, 0] > self.width - 1)
            or np.any(uv[:, 1] < 0.0)
            or np.any(uv[:, 1] > self.height - 1)
        ):
            raise ValueError("depth-noise pixel is outside the activated profile")
        if np.any(depth < self.calibrated_min_depth_m) or np.any(
            depth > self.calibrated_max_depth_m
        ):
            raise ValueError("depth is outside the calibrated range")

        expansion = float(self.uncertainty_expansion_factor)
        sigma_u = expansion * math.hypot(
            float(self.pixel_quantization_sigma_px), float(self.alignment_sigma_u_px),
        )
        sigma_v = expansion * math.hypot(
            float(self.pixel_quantization_sigma_px), float(self.alignment_sigma_v_px),
        )
        axial = np.hypot(
            float(self.axial_sigma_floor_m),
            float(self.axial_sigma_quadratic_m_inv) * depth * depth,
        )
        sigma_z = expansion * np.hypot(axial, float(self.alignment_sigma_depth_m))

        output = np.empty((len(depth), 3, 3), dtype=np.float64)
        for index, ((u, v), z, z_sigma) in enumerate(zip(uv, depth, sigma_z)):
            jacobian = np.array(
                [
                    [z / self.fx, 0.0, (u - self.cx) / self.fx],
                    [0.0, z / self.fy, (v - self.cy) / self.fy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            uvz_covariance = np.diag(
                [sigma_u * sigma_u, sigma_v * sigma_v, z_sigma * z_sigma]
            )
            covariance = jacobian @ uvz_covariance @ jacobian.T
            output[index] = 0.5 * (covariance + covariance.T)
        if not np.isfinite(output).all():
            raise ValueError("depth-noise covariance is non-finite")
        return output

    def common_alignment_pose_covariance_se3(
        self, *, maximum_depth_m: float, target_radius_m: float
    ) -> np.ndarray:
        """Common alignment term that is not divided by feature count."""

        if (
            not _finite_real(maximum_depth_m)
            or not self.calibrated_min_depth_m
            <= float(maximum_depth_m)
            <= self.calibrated_max_depth_m
            or not _finite_real(target_radius_m)
            or float(target_radius_m) <= 0.0
        ):
            raise ValueError("alignment pose-covariance input is invalid")
        factor = _SQRT_TWO * float(self.uncertainty_expansion_factor)
        rotation_x = factor * self.alignment_sigma_v_px / self.fy
        rotation_y = factor * self.alignment_sigma_u_px / self.fx
        rotation_z = math.hypot(rotation_x, rotation_y)
        translation_x = factor * maximum_depth_m * self.alignment_sigma_u_px / self.fx
        translation_y = factor * maximum_depth_m * self.alignment_sigma_v_px / self.fy
        translation_z = factor * self.alignment_sigma_depth_m
        sigma = np.array(
            [
                rotation_x,
                rotation_y,
                rotation_z,
                translation_x,
                translation_y,
                translation_z,
            ],
            dtype=np.float64,
        )
        return np.diag(sigma * sigma)

    def fit_residual_pose_covariance_se3(
        self, *, fit_rmse_m: float, target_radius_m: float
    ) -> np.ndarray:
        """Conservative fit term that is not divided by feature count."""

        if (
            not _finite_real(fit_rmse_m)
            or float(fit_rmse_m) < 0.0
            or not _finite_real(target_radius_m)
            or float(target_radius_m) <= 0.0
        ):
            raise ValueError("fit pose-covariance input is invalid")
        sigma_m = (
            float(fit_rmse_m)
            * float(self.fit_residual_scale)
            * float(self.uncertainty_expansion_factor)
        )
        sigma_rotation = sigma_m / float(target_radius_m)
        sigma = np.array([sigma_rotation] * 3 + [sigma_m] * 3, dtype=np.float64,)
        return np.diag(sigma * sigma)


__all__ = [
    "ActivatedDepthNoiseProfile",
    "depth_noise_profile_semantic_sha256",
]
