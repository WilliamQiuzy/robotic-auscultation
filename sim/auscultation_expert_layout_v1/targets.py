"""Load model-proposed surface points for simulation-only contact planning.

The displayed points were approved for visualization, not as anatomical ground
truth. Loading a prediction never grants anatomical acceptance or robot motion
authorization. A verified stage ``world_from_left_base`` transform is required.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np


DEFAULT_PREDICTIONS_PATH = Path(__file__).parents[1] / "auscultation_assets_20260922" / "model_predictions.json"


@dataclass(frozen=True)
class Target:
    """One candidate point, with outward normals in world and left-base frames."""

    name: str
    position_world: np.ndarray
    normal_world: np.ndarray
    position_base: np.ndarray
    normal_base: np.ndarray
    provenance: dict
    accepted_target: bool = field(default=False, init=False)

    @property
    def approach_axis_base(self):
        """Desired tool +Z direction: into the surface, opposite its normal."""
        return -self.normal_base


def _vector(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite 3-vector") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite 3-vector")
    return result


def _base_transform(value):
    label = "world_from_left_base"
    try:
        transform = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a rigid 4 x 4 transform") from error
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{label} must be a finite 4 x 4 transform")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)):
        raise ValueError(f"{label} must contain a proper rotation and homogeneous last row")
    return transform


def load_targets(world_from_left_base, path=None, frame="raised"):
    """Return candidates from ``frames[frame].points`` in document order.

    Positions are metres. ``normal_world_from_depth`` is normalized, rotated
    into the base frame, and retained as an outward surface normal. Malformed,
    nonfinite, zero-normal, duplicate-name, or depth-invalid points cause a
    ValueError rather than silently producing an incomplete target set.

    Provenance retains all document metadata except ``frames``, selected frame
    metadata except ``points``, the source point, source path, and frame name.
    ``accepted_target`` always remains false, including if the input claims true.
    """
    transform = _base_transform(world_from_left_base)
    source = Path(path) if path is not None else DEFAULT_PREDICTIONS_PATH
    with source.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("Prediction document must be a JSON object")
    frames = document.get("frames")
    if not isinstance(frames, dict) or not isinstance(frame, str) or frame not in frames:
        raise ValueError(f"Prediction frame {frame!r} does not exist")
    frame_data = frames[frame]
    if not isinstance(frame_data, dict) or not isinstance(frame_data.get("points"), list):
        raise ValueError(f"Prediction frame {frame!r} must contain a points list")
    rotation, origin = transform[:3, :3], transform[:3, 3]
    metadata = {key: deepcopy(value) for key, value in document.items() if key != "frames"}
    targets, names = [], set()
    for point in frame_data["points"]:
        if not isinstance(point, dict):
            raise ValueError("Every candidate point must be an object")
        name = point.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Candidate name must be a nonempty string")
        if name in names:
            raise ValueError(f"Candidate name {name!r} is duplicate")
        names.add(name)
        if point.get("depth_valid") is not True:
            raise ValueError(f"Candidate {name} requires depth_valid=true")
        position_world = _vector(point.get("world_xyz_m"), f"{name} world_xyz_m")
        normal_world = _vector(point.get("normal_world_from_depth"), f"{name} normal_world_from_depth")
        norm = np.linalg.norm(normal_world)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError(f"Candidate {name} requires a finite nonzero surface normal")
        normal_world = normal_world / norm
        provenance = deepcopy(metadata)
        provenance.update({
            "source_path": str(source.resolve()), "frame": frame,
            "frame_metadata": {key: deepcopy(value) for key, value in frame_data.items() if key != "points"},
            "source_point": deepcopy(point),
        })
        targets.append(Target(
            name=name, position_world=position_world.copy(), normal_world=normal_world,
            position_base=rotation.T @ (position_world - origin),
            normal_base=rotation.T @ normal_world, provenance=provenance,
        ))
    return targets
