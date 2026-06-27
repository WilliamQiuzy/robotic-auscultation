"""Streaming auscultation-point tracker.

Turns the single-frame perception model (RGB → 10 posterior-lung points) into a continuous,
long-duration (≤10 min) 3D tracker that keeps each point locked to the same anatomy while the
patient and camera move. Design: ``docs/perception_streaming_tracker.md``.

Recommended production components: the geometry/garment-invariant ``NormalsDetector`` (depth-derived
surface-normal input) feeding the ``ModelBasedTracker`` (shared-similarity-pose fit, per-point
visibility flags). The original ``StreamingAuscultationTracker`` (4-channel EKF) + ``DenseIUVDetector``
(RGB) remain for reference / the IUV-semantic channel.

Public API:
    from ausc.perception.tracking import (
        NormalsDetector, ModelBasedTracker, ModelConfig, MTOutput, MTPoint,   # production line
        StreamingAuscultationTracker, TrackerConfig, TrackerOutput, TrackedPoint,
        Intrinsics, DenseIUVDetector, LUNG_ORDER,
    )
"""

from .detector import DenseIUVDetector, DetectorResult, LandmarkDetector, LUNG_ORDER, NormalsDetector
from .geometry import CropTransform, Intrinsics
from .intrinsics import (default_intrinsics, load_intrinsics, orbbec_intrinsics,
                         save_intrinsics)
from .model_tracker import ModelBasedTracker, ModelConfig, MTOutput, MTPoint
from .rgbd_odometry import FeatureField
from .tracker import (StreamingAuscultationTracker, TrackedPoint, TrackerConfig,
                      TrackerOutput)

__all__ = [
    # production line (geometry-primary, garment-invariant)
    "NormalsDetector", "ModelBasedTracker", "ModelConfig", "MTOutput", "MTPoint",
    # original EKF line + shared primitives
    "StreamingAuscultationTracker", "TrackerConfig", "TrackerOutput", "TrackedPoint",
    "Intrinsics", "CropTransform", "DenseIUVDetector", "DetectorResult", "LandmarkDetector",
    "LUNG_ORDER", "FeatureField", "orbbec_intrinsics", "load_intrinsics", "save_intrinsics",
    "default_intrinsics",
]
