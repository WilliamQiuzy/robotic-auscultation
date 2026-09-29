"""Candidate surface points retain their prediction provenance."""

import importlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def load_targets(*args, **kwargs):
    return importlib.import_module("sim.auscultation_training.targets").load_targets(*args, **kwargs)


def write_predictions(tmp_path, **point_changes):
    point = {
        "name": "R1", "world_xyz_m": [1, 4, 3],
        "normal_world_from_depth": [0, -2, 0], "depth_valid": True,
        "inside_frame": True, "accepted_target": False,
    }
    point.update(point_changes)
    document = {
        "model": "test-predictor", "checkpoint_sha256": "abc",
        "point_semantics": "model proposals, not anatomical ground truth",
        "frames": {"raised": {"points": [point]}},
    }
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(document))
    return path


def test_target_positions_and_normals_use_supplied_base_transform(tmp_path):
    world_from_base = np.eye(4)
    world_from_base[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    world_from_base[:3, 3] = [1, 2, 3]
    target, = load_targets(world_from_base, path=write_predictions(tmp_path))
    assert target.name == "R1"
    np.testing.assert_allclose(target.position_world, [1, 4, 3])
    np.testing.assert_allclose(target.normal_world, [0, -1, 0])
    np.testing.assert_allclose(target.position_base, [2, 0, 0], atol=1e-12)
    np.testing.assert_allclose(target.normal_base, [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(target.approach_axis_base, [1, 0, 0], atol=1e-12)


def test_candidate_status_is_never_promoted_to_anatomical_acceptance(tmp_path):
    target, = load_targets(np.eye(4), path=write_predictions(tmp_path, accepted_target=True))
    assert target.accepted_target is False
    assert target.provenance["frame"] == "raised"
    assert target.provenance["model"] == "test-predictor"
    assert target.provenance["checkpoint_sha256"] == "abc"
    assert target.provenance["source_point"]["accepted_target"] is True
    assert "not anatomical ground truth" in target.provenance["point_semantics"]


def test_checked_in_raised_predictions_load_all_ten_candidates():
    targets = load_targets(np.eye(4))
    assert [target.name for target in targets] == [name for i in range(1, 6) for name in (f"R{i}", f"L{i}")]
    for target in targets:
        assert not target.accepted_target
        assert np.isfinite(target.position_base).all()
        np.testing.assert_allclose(np.linalg.norm(target.normal_world), 1)
        np.testing.assert_allclose(target.position_base, target.position_world)


@pytest.mark.parametrize("changes", [
    {"depth_valid": False}, {"depth_valid": "true"},
    {"world_xyz_m": [0, np.nan, 0]}, {"world_xyz_m": [0, 1]},
    {"normal_world_from_depth": [0, 0, 0]},
    {"normal_world_from_depth": [0, np.inf, 1]}, {"name": ""},
])
def test_rejects_invalid_candidate_fields(tmp_path, changes):
    with pytest.raises(ValueError):
        load_targets(np.eye(4), path=write_predictions(tmp_path, **changes))


@pytest.mark.parametrize("transform", [
    np.eye(3), np.zeros((4, 4)), np.diag([2, 1, 1, 1]),
    np.diag([-1, 1, 1, 1]), np.diag([1, 1, np.nan, 1]),
])
def test_rejects_invalid_base_transform(tmp_path, transform):
    with pytest.raises(ValueError, match="world_from_left_base"):
        load_targets(transform, path=write_predictions(tmp_path))


def test_rejects_missing_frame(tmp_path):
    with pytest.raises(ValueError, match="frame"):
        load_targets(np.eye(4), path=write_predictions(tmp_path), frame="missing")


def test_rejects_duplicate_target_names(tmp_path):
    path = write_predictions(tmp_path)
    data = json.loads(path.read_text())
    data["frames"]["raised"]["points"] *= 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        load_targets(np.eye(4), path=path)
