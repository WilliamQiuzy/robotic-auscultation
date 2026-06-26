"""Streaming auscultation-point tracker.

Turns the single-frame perception model (RGB → 10 posterior-lung points) into a continuous,
long-duration (≤10 min) 3D tracker that keeps each point locked to the same anatomy while the
patient and camera move. Design: ``docs/perception_streaming_tracker.md``.

Public API:
    from ausc.perception.tracking import (
        StreamingAuscultationTracker, TrackerConfig, TrackerOutput, TrackedPoint,
        Intrinsics, DenseIUVDetector, LUNG_ORDER,
    )
"""

from .detector import DenseIUVDetector, DetectorResult, LandmarkDetector, LUNG_ORDER
from .geometry import CropTransform, Intrinsics
from .intrinsics import (default_intrinsics, load_intrinsics, orbbec_intrinsics,
                         save_intrinsics)
from .rgbd_odometry import FeatureField
from .tracker import (StreamingAuscultationTracker, TrackedPoint, TrackerConfig,
                      TrackerOutput)

__all__ = [
    "StreamingAuscultationTracker", "TrackerConfig", "TrackerOutput", "TrackedPoint",
    "Intrinsics", "CropTransform", "DenseIUVDetector", "DetectorResult", "LandmarkDetector",
    "LUNG_ORDER", "FeatureField", "orbbec_intrinsics", "load_intrinsics", "save_intrinsics",
    "default_intrinsics",
]
