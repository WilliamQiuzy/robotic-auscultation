"""Strict goal-coordinate contract tests."""

from __future__ import annotations

import json

import pytest

import ausc_targets as targets


def _write_jsonl(path, *records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_defaults_match_canonical_ten_point_lung_task():
    assert [p.name for p in targets.DEFAULT_TARGETS] == [
        "R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5",
    ]


def test_normalized_uv_is_validated(tmp_path):
    path = tmp_path / "targets.jsonl"
    _write_jsonl(path, {"name": "R1", "uv": [0.2, 0.3]})
    assert targets.load_targets(str(path))[0].uv == pytest.approx((0.2, 0.3))
    _write_jsonl(path, {"name": "R1", "uv": [1.2, 0.3]})
    with pytest.raises(ValueError, match="normalized uv"):
        targets.load_targets(str(path))


def test_pixel_coordinates_require_dimensions_and_are_normalized(tmp_path):
    path = tmp_path / "targets.jsonl"
    _write_jsonl(path, {"name": "R1", "xy_pixel": [320, 240]})
    with pytest.raises(ValueError, match="image_width.*image_height"):
        targets.load_targets(str(path))
    _write_jsonl(path, {
        "name": "R1", "xy_pixel": [320, 240], "image_width": 640, "image_height": 480,
    })
    assert targets.load_targets(str(path))[0].uv == pytest.approx((0.5, 0.5))


def test_xyz_coordinate_frame_is_never_inferred_ambiguously(tmp_path):
    path = tmp_path / "targets.jsonl"
    _write_jsonl(path, {"name": "R1", "uv": [0.2, 0.3], "xyz": [1, 2, 3]})
    with pytest.raises(ValueError, match="xyz_frame"):
        targets.load_targets(str(path))
    _write_jsonl(path, {"name": "R1", "uv": [0.2, 0.3], "xyz_camera": [1, 2, 3]})
    point = targets.load_targets(str(path))[0]
    assert point.xyz == (1.0, 2.0, 3.0)
    assert point.xyz_frame == "camera"


def test_explicit_missing_target_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        targets.load_targets(str(tmp_path / "missing.json"))


def test_target_log_binds_goal_to_camera_frame_identity(tmp_path):
    log = targets.TargetLog(str(tmp_path))
    log.log(
        10.25, targets.TargetPoint(0, "R1", (0.2, 0.3)), 640, 480,
        camera_frame_index=7, camera_device_t_ms=1234.5,
    )
    log.close()
    record = json.loads((tmp_path / "targets.jsonl").read_text())
    assert record["camera_frame_index"] == 7
    assert record["camera_host_arrival_t_s"] == pytest.approx(10.25)
    assert record["camera_device_t_ms"] == pytest.approx(1234.5)
