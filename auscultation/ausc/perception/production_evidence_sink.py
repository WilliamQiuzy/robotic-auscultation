"""Single sink that composes the whole production perception-evidence write-side.

The orchestrator drives one :class:`ProductionEvidenceSink` per session.  It ties
together everything the write-side needs so the live pipeline never has to know
the internal contracts:

* the geometry recorder (dual-channel packet log + Step 2a model-input ledger),
* the two geometry-channel adapters (perception scalar-std -> isotropic 3x3;
  independent SE(3) pose -> per-landmark points + propagated 3x3), and
* the quality recorder (production StreamingQuality + packet-log writer).

The sink owns the monotonic geometry ``sequence`` counter so the caller only
supplies per-frame model outputs, and it fails closed exactly like the
components it wraps.  It is framework-free: the caller injects the already-built
:class:`PatientFrameEvidence`, the perception landmark points + scalar stds, the
independent :class:`PatientPosePacket`, the canonical patient-frame template, and
the audio windows — so the sink's composition is fully unit-testable without a
camera or a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ausc.perception.tracking.geometry_channel_adapters import (
    independent_channel_geometry,
    perception_channel_geometry,
)
from ausc.perception.tracking.production_geometry_recorder import (
    GeometryChannelIdentity,
    ProductionGeometrySessionRecorder,
)
from ausc.production_quality_recorder import ProductionQualitySessionRecorder


class ProductionEvidenceSinkError(RuntimeError):
    """The production evidence sink cannot record a trustworthy session."""


class ProductionEvidenceSink:
    """Compose geometry + quality production recording for one session."""

    def __init__(
        self,
        session_dir: "str | Path",
        *,
        intrinsics,
        geometry_profile,
        point_names: Sequence[str],
        perception_channel: GeometryChannelIdentity,
        independent_channel: GeometryChannelIdentity,
        quality_scorer,
        quality_model_release_id: str,
        quality_model_sha256: str,
        quality_policy: Mapping[str, object],
    ) -> None:
        self._session_dir = Path(session_dir).absolute()
        self._point_names = tuple(point_names)
        self._geometry = ProductionGeometrySessionRecorder(
            self._session_dir,
            intrinsics=intrinsics,
            geometry_profile=geometry_profile,
            point_names=self._point_names,
            perception_channel=perception_channel,
            independent_channel=independent_channel,
        )
        self._quality = ProductionQualitySessionRecorder(
            self._session_dir,
            scorer=quality_scorer,
            model_release_id=quality_model_release_id,
            model_sha256=quality_model_sha256,
            policy=quality_policy,
        )
        self._geometry_sequence = 0
        self._quality_windows = 0

    @property
    def geometry_frame_count(self) -> int:
        return self._geometry.frame_count

    @property
    def quality_window_count(self) -> int:
        return self._quality_windows

    def record_geometry_frame(
        self,
        patient_frame_evidence,
        *,
        episode_id: str,
        perception_points_xyz_camera_m: np.ndarray,
        perception_position_std_m: np.ndarray,
        patient_pose_packet,
        template_points_patient_m: np.ndarray,
    ) -> None:
        """Apply both channel adapters and record one geometry frame.

        The caller supplies the two model outputs; the sink turns them into the
        packet-valid ``(points_m, covariances_m2)`` pairs and advances the
        monotonic geometry sequence itself.
        """

        perception_points_m, perception_cov = perception_channel_geometry(
            perception_points_xyz_camera_m, perception_position_std_m,
        )
        independent_points_m, independent_cov = independent_channel_geometry(
            patient_pose_packet, template_points_patient_m,
        )
        if perception_points_m.shape[0] != len(self._point_names) or (
            independent_points_m.shape[0] != len(self._point_names)
        ):
            raise ProductionEvidenceSinkError(
                "geometry channel point count does not match the point names"
            )
        self._geometry_sequence += 1
        self._geometry.record_frame(
            patient_frame_evidence,
            episode_id=episode_id,
            sequence=self._geometry_sequence,
            perception_points_m=perception_points_m,
            perception_covariances_m2=perception_cov,
            independent_points_m=independent_points_m,
            independent_covariances_m2=independent_cov,
        )

    def record_quality_window(
        self,
        *,
        audio_window,
        force_sample,
        inference_sequence: int,
        inference_monotonic_s: float,
        observed_monotonic_s: float | None = None,
    ) -> dict[str, object]:
        result = self._quality.record_window(
            audio_window=audio_window,
            force_sample=force_sample,
            inference_sequence=inference_sequence,
            inference_monotonic_s=inference_monotonic_s,
            observed_monotonic_s=observed_monotonic_s,
        )
        self._quality_windows += 1
        return result

    def finalize_geometry(self):
        return self._geometry.finalize()

    def finalize_quality(self, *, now_monotonic_s: float | None = None):
        return self._quality.finalize(now_monotonic_s=now_monotonic_s)


__all__ = [
    "ProductionEvidenceSink",
    "ProductionEvidenceSinkError",
]
