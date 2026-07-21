from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import handeye_calibrate as handeye  # noqa: E402
import handeye_orbbec_preflight as orbbec_preflight  # noqa: E402


def _rotation(rotvec: tuple[float, float, float]) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64))[0]


def _synthetic_tip_touch_samples(
    *,
    same_pair_orientation: bool = False,
    fit_points=None,
    pair_axes=None,
    heldout_points=None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    camera_to_base = np.eye(4, dtype=np.float64)
    camera_to_base[:3, :3] = _rotation((0.16, -0.11, 0.08))
    camera_to_base[:3, 3] = (0.42, -0.18, 0.73)
    base_to_camera = np.linalg.inv(camera_to_base)
    tip_in_tcp = np.asarray((0.014, -0.009, 0.083), dtype=np.float64)

    tip_camera: list[np.ndarray] = []
    tcp_rotations: list[np.ndarray] = []
    tcp_translations: list[np.ndarray] = []
    points = fit_points or [
        (0.20, -0.28, 0.52), (0.32, -0.24, 0.61),
        (0.47, -0.18, 0.71), (0.58, -0.05, 0.82),
        (0.24, 0.05, 0.65), (0.39, 0.12, 0.76),
        (0.54, 0.20, 0.88), (0.28, 0.27, 0.58),
        (0.45, 0.31, 0.69), (0.61, 0.25, 0.79),
    ]
    axes = pair_axes or [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
        (-1.0, 1.0, 0.5), (0.5, -1.0, 1.0), (1.0, 0.5, -1.0),
        (1.0, 1.0, 1.0),
    ]
    if len(points) != 10 or len(axes) != 10:
        raise ValueError("synthetic commissioning fixture requires ten fit points and axes")
    for index, point in enumerate(points):
        first = _rotation((0.06 * index, -0.04 * index, 0.03 * index))
        if same_pair_orientation:
            second = first
        else:
            axis = np.asarray(axes[index], dtype=np.float64)
            axis /= np.linalg.norm(axis)
            second = first @ _rotation(tuple(axis * math.radians(35.0)))
        point_base = np.asarray(point, dtype=np.float64)
        point_camera = (base_to_camera @ np.r_[point_base, 1.0])[:3]
        for rotation in (first, second):
            tip_camera.append(point_camera.copy())
            tcp_rotations.append(rotation)
            tcp_translations.append(point_base - rotation @ tip_in_tcp)

    heldout = heldout_points or [
        (0.34, -0.10, 0.57),
        (0.50, 0.03, 0.74),
        (0.30, 0.22, 0.84),
    ]
    heldout_rotations = [
        (0.20, -0.15, 0.10),
        (-0.10, 0.22, 0.14),
        (0.18, 0.12, -0.20),
    ]
    if len(heldout) != 3:
        raise ValueError("synthetic commissioning fixture requires three held-out points")
    for point, rotvec in zip(heldout, heldout_rotations):
        point_base = np.asarray(point, dtype=np.float64)
        rotation = _rotation(rotvec)
        tip_camera.append((base_to_camera @ np.r_[point_base, 1.0])[:3])
        tcp_rotations.append(rotation)
        tcp_translations.append(point_base - rotation @ tip_in_tcp)
    return tip_camera, tcp_rotations, tcp_translations


def test_clicked_depth_uses_the_camera_frame_depth_scale() -> None:
    frame = SimpleNamespace(
        depth=np.asarray([[0, 1250]], dtype=np.uint16),
        depth_scale=0.2,
    )

    assert handeye.clicked_depth_metres(frame, 1, 0) == pytest.approx(0.25)
    assert handeye.clicked_depth_metres(frame, 0, 0) is None


def test_clicked_depth_patch_is_robust_but_rejects_a_depth_edge() -> None:
    stable = np.full((5, 5), 1000, dtype=np.uint16)
    stable[0, 0] = 0
    stable[4, 4] = 1010
    frame = SimpleNamespace(depth=stable, depth_scale=0.2)

    measurement = handeye.clicked_depth_patch(frame, 2, 2)

    assert measurement["depth_m"] == pytest.approx(0.2)
    assert measurement["valid_count"] == 24
    assert measurement["span_mm"] <= 2.0

    edge = stable.copy()
    edge[:, :2] = 1400
    edge_frame = SimpleNamespace(depth=edge, depth_scale=0.2)
    assert handeye.clicked_depth_patch(edge_frame, 2, 2) is None


def test_calibration_intrinsics_are_bound_to_the_active_color_stream() -> None:
    camera = SimpleNamespace(
        color_intrinsics={
            "fx": 1000.0,
            "fy": 1001.0,
            "cx": 959.5,
            "cy": 539.5,
            "width": 1920,
            "height": 1080,
        },
        negotiated_mode={"width": 1920, "height": 1080},
        _pipe=SimpleNamespace(default_profile="must-not-be-used"),
    )

    intrinsics = handeye.calibration_intrinsics_from_stream(camera)

    assert intrinsics.fx == pytest.approx(1000.0)
    assert intrinsics.fy == pytest.approx(1001.0)
    assert intrinsics.width == 1920
    assert intrinsics.height == 1080
    assert np.allclose(
        intrinsics.unproject(
            np.asarray([[959.5, 539.5]]), np.asarray([0.75]),
        ),
        np.asarray([[0.0, 0.0, 0.75]]),
    )


def test_calibration_intrinsics_reject_profile_mismatch() -> None:
    camera = SimpleNamespace(
        color_intrinsics={
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 959.5,
            "cy": 539.5,
            "width": 1920,
            "height": 1080,
        },
        negotiated_mode={"width": 1280, "height": 720},
    )

    with pytest.raises(RuntimeError, match="active color profile"):
        handeye.calibration_intrinsics_from_stream(camera)


def test_calibration_frame_gate_rejects_stale_or_failed_delivery() -> None:
    healthy = {
        "status": "OK",
        "thread_alive": True,
        "thread_error": None,
        "frame_fresh": True,
    }
    camera = SimpleNamespace(delivery_health=lambda: dict(healthy))
    frame = SimpleNamespace(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.ones((4, 4), dtype=np.uint16),
        status="OK",
        age_s=0.01,
    )

    assert handeye.calibration_frame_ready(camera, frame, max_age_s=0.1) is True

    frame.status = "STALE"
    assert handeye.calibration_frame_ready(camera, frame, max_age_s=0.1) is False
    frame.status = "OK"
    frame.age_s = 0.101
    assert handeye.calibration_frame_ready(camera, frame, max_age_s=0.1) is False
    frame.age_s = 0.01
    healthy["thread_error"] = "RuntimeError: capture failed"
    healthy["status"] = "ERROR"
    assert handeye.calibration_frame_ready(camera, frame, max_age_s=0.1) is False


def test_orbbec_preflight_bundle_is_hash_bound_and_fail_closed(tmp_path: Path) -> None:
    report = {
        "schema": "wam_handeye_orbbec_preflight_v1",
        "serial": "CL8326300A0",
        "status": "PASS",
        "intrinsics": {
            "fx": 1000.0, "fy": 1000.0, "cx": 959.5, "cy": 539.5,
            "width": 1920, "height": 1080,
        },
        "distortion": {
            "model": "model_api_unexposed",
            "coefficient_order": ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"],
            "coefficients": [0.0] * 8,
        },
        "modes": orbbec_preflight.EXPECTED_MODES,
    }
    path = tmp_path / "preflight.json"
    orbbec_preflight.write_report_bundle(path, report)

    evidence = handeye.load_orbbec_preflight_bundle(path)

    assert evidence["report"] == report
    assert len(evidence["report_sha256"]) == 64
    assert len(evidence["pass_marker_sha256"]) == 64

    path.write_bytes(orbbec_preflight.canonical_json_bytes({**report, "serial": "tampered"}))
    with pytest.raises(RuntimeError, match="hash|serial"):
        handeye.load_orbbec_preflight_bundle(path)


def test_commissioning_arguments_require_twenty_fit_and_three_heldout() -> None:
    args = handeye.parse_arguments([
        "--arm", "right",
        "--joint-csv", "/tmp/joints.csv",
        "--mode", "tiptouch",
        "--fixture-id", "lab_fixture_v1",
    ])

    assert args.n_points == 20
    assert args.heldout_points == 3
    assert args.pair_count == 10
    assert args.camera_id == "CL8326300A0"


def test_commissioning_defaults_to_the_production_left_follower_can3() -> None:
    args = handeye.parse_arguments([
        "--joint-csv", "/tmp/joints.csv",
        "--mode", "tiptouch",
        "--fixture-id", "lab_fixture_v1",
        "--environment-lock-sha256", "a" * 64,
        "--environment-clone-marker-sha256", "b" * 64,
    ])

    handeye.validate_commissioning_arguments(args)

    assert args.arm == "left"
    assert args.can_interface == "can3"


def test_commissioning_preflight_rejects_legacy_or_unpaired_capture() -> None:
    charuco = handeye.parse_arguments([
        "--joint-csv", "/tmp/joints.csv", "--mode", "charuco",
        "--fixture-id", "lab_fixture_v1",
    ])
    with pytest.raises(ValueError, match="Charuco"):
        handeye.validate_commissioning_arguments(charuco)

    unpaired = handeye.parse_arguments([
        "--joint-csv", "/tmp/joints.csv", "--n-points", "14",
        "--fixture-id", "lab_fixture_v1",
    ])
    with pytest.raises(ValueError, match="two wrist orientations"):
        handeye.validate_commissioning_arguments(unpaired)

    headless = handeye.parse_arguments([
        "--joint-csv", "/tmp/joints.csv", "--headless",
        "--fixture-id", "lab_fixture_v1",
    ])
    with pytest.raises(ValueError, match="graphical desktop"):
        handeye.validate_commissioning_arguments(headless)


def test_tip_touch_plan_pairs_each_rigid_position_before_heldout() -> None:
    plan = handeye.tip_touch_sample_plan(pair_count=10, heldout_count=3)

    assert len(plan) == 23
    assert plan[:4] == [
        {"orientation": "A", "phase": "fit", "position_index": 1},
        {"orientation": "B", "phase": "fit", "position_index": 1},
        {"orientation": "A", "phase": "fit", "position_index": 2},
        {"orientation": "B", "phase": "fit", "position_index": 2},
    ]
    assert plan[-1] == {
        "orientation": None,
        "phase": "heldout",
        "position_index": 3,
    }


def test_build_tip_touch_sample_binds_frame_depth_and_fk() -> None:
    frame = SimpleNamespace(
        depth=np.full((5, 5), 1000, dtype=np.uint16),
        depth_scale=0.25,
        index=42,
        t=123.5,
        device_t_ms=456.0,
        time_source="device",
        age_s=0.01,
    )
    intrinsics = SimpleNamespace(
        unproject=lambda pixels, depth: np.asarray([[0.1, 0.2, depth[0]]]),
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (0.4, 0.5, 0.6)

    sample = handeye.build_tip_touch_sample(
        frame=frame,
        intrinsics=intrinsics,
        pixel=(2, 2),
        q7=np.arange(7, dtype=np.float64),
        T_tcp_base=transform,
        sample_id=1,
        plan_entry={"phase": "fit", "position_index": 1, "orientation": "A"},
    )

    assert sample["depth_m"] == pytest.approx(0.25)
    assert sample["depth_scale_mm"] == pytest.approx(0.25)
    assert sample["tip_cam_m"] == pytest.approx([0.1, 0.2, 0.25])
    assert sample["q7_rad"] == pytest.approx(np.arange(7).tolist())
    assert np.allclose(sample["T_tcp_base"], transform)
    assert sample["camera_frame_index"] == 42
    assert sample["camera_device_t_ms"] == 456.0


def test_capture_tiptouch_persists_each_accepted_click(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = handeye.create_commissioning_run(
        tmp_path, "handeye_right_20260716T190003Z_deadbeef", {"arm": "right"},
    )
    frame = SimpleNamespace(
        frame=np.zeros((5, 5, 3), dtype=np.uint8),
        depth=np.full((5, 5), 1000, dtype=np.uint16),
        depth_scale=1.0,
        index=7,
        t=12.0,
        device_t_ms=15.0,
        time_source="device",
        age_s=0.01,
        status="OK",
    )
    camera = SimpleNamespace(
        latest_frame=lambda: frame,
        delivery_health=lambda: {
            "status": "OK",
            "thread_alive": True,
            "thread_error": None,
            "frame_fresh": True,
        },
    )
    intrinsics = SimpleNamespace(
        unproject=lambda pixels, depth: np.asarray([[0.0, 0.0, depth[0]]]),
    )
    fk = SimpleNamespace(fk=lambda _q: np.eye(4, dtype=np.float64))
    callback = {}
    wait_calls = {"count": 0}

    monkeypatch.setattr(handeye.cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(
        handeye.cv2, "setMouseCallback",
        lambda _name, value: callback.setdefault("mouse", value),
    )
    monkeypatch.setattr(handeye.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(handeye.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(handeye.cv2, "putText", lambda image, *_args: image)
    monkeypatch.setattr(
        handeye,
        "read_latest_joints",
        lambda *_args, **_kwargs: (np.zeros(7), True),
    )

    def wait_key(_delay: int) -> int:
        wait_calls["count"] += 1
        if wait_calls["count"] == 1:
            callback["mouse"](handeye.cv2.EVENT_LBUTTONDOWN, 2, 2, None, None)
        return -1

    monkeypatch.setattr(handeye.cv2, "waitKey", wait_key)

    samples, aborted = handeye.capture_tiptouch(
        camera,
        "/tmp/joints.csv",
        fk,
        intrinsics,
        [{"orientation": "A", "phase": "fit", "position_index": 1}],
        run,
        display_scale=1.0,
    )

    assert aborted is False
    assert len(samples) == 1
    assert samples[0]["sample_id"] == 1
    assert (run / "captures/0001_rgb.png").is_file()
    assert (run / "captures/0001_depth.npy").is_file()


def test_joint_reader_rejects_a_stale_gravity_comp_csv(tmp_path: Path) -> None:
    path = tmp_path / "joints.csv"
    path.write_text(
        "0.00,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0\n"
        "0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0\n",
        encoding="utf-8",
    )
    os.utime(path, (100.0, 100.0))

    joints, still = handeye.read_latest_joints(
        str(path), max_file_age_s=0.5, now_s=101.0,
    )

    assert joints is None
    assert still is False


def test_joint_reader_requires_the_exact_gravity_comp_schema(tmp_path: Path) -> None:
    path = tmp_path / "joints.csv"
    path.write_text(
        "wrong,a1,a2,a3,a4,a5,a6,a7,grip\n"
        "0.00,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0\n"
        "0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0\n",
        encoding="utf-8",
    )

    joints, still = handeye.read_latest_joints(
        str(path), settle_rows=2, now_s=path.stat().st_mtime,
    )

    assert joints is None
    assert still is False


@pytest.mark.parametrize(
    "bad_row",
    [
        "0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7\n",
        "0.01,0.1,0.2,0.3,nan,0.5,0.6,0.7,0\n",
        "0.00,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0\n",
    ],
)
def test_joint_reader_rejects_bad_rows_and_nonmonotonic_time(
    tmp_path: Path, bad_row: str,
) -> None:
    path = tmp_path / "joints.csv"
    path.write_text(
        "t,a1,a2,a3,a4,a5,a6,a7,grip\n"
        "0.00,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0\n"
        + bad_row,
        encoding="utf-8",
    )

    assert handeye.read_latest_joints(
        str(path), settle_rows=2, now_s=path.stat().st_mtime,
    ) == (None, False)


def test_joint_stillness_checks_the_whole_settle_window(tmp_path: Path) -> None:
    path = tmp_path / "joints.csv"
    path.write_text(
        "t,a1,a2,a3,a4,a5,a6,a7,grip\n"
        "0.00,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0\n"
        "0.01,0.1,0.2,0.3,0.8,0.5,0.6,0.7,0\n"
        "0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0\n",
        encoding="utf-8",
    )

    joints, still = handeye.read_latest_joints(
        str(path), settle_rows=3, now_s=path.stat().st_mtime,
    )

    assert joints == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert still is False


def _fake_gravity_process(
    tmp_path: Path, *, arm: str = "right", can_interface: str = "can2",
):
    binary = tmp_path / "gravity_comp"
    binary.write_bytes(b"gravity-binary")
    binary.chmod(0o755)
    urdf = tmp_path / "openarm_v10.urdf"
    urdf.write_text("<robot name='openarm'/>", encoding="utf-8")
    joint_csv = tmp_path / "joints.csv"
    joint_csv.write_text(
        "t,a1,a2,a3,a4,a5,a6,a7,grip\n"
        "0.00,0,0,0,0,0,0,0,0\n"
        "0.01,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    proc_root = tmp_path / "proc"
    process = proc_root / "1234"
    (process / "fd").mkdir(parents=True)
    (process / "exe").symlink_to(binary)
    (process / "fd/4").symlink_to(joint_csv)
    (process / "cmdline").write_bytes(
        b"\0".join([
            os.fsencode(binary),
            os.fsencode(arm + "_arm"),
            os.fsencode(can_interface),
            os.fsencode(urdf),
            b"",
        ])
    )
    (process / "environ").write_bytes(
        b"REC_CSV=" + os.fsencode(joint_csv) + b"\0"
    )
    return proc_root, binary, urdf, joint_csv


def test_gravity_comp_process_is_bound_to_arm_can_urdf_and_csv(tmp_path: Path) -> None:
    proc_root, binary, urdf, joint_csv = _fake_gravity_process(tmp_path)

    record = handeye.verify_gravity_comp_process(
        arm="right",
        can_interface="can2",
        gravity_comp_bin=binary,
        joint_csv=joint_csv,
        urdf=urdf,
        proc_root=proc_root,
    )

    assert record["pid"] == 1234
    assert record["arm_argument"] == "right_arm"
    assert record["can_interface"] == "can2"
    assert record["binary_sha256"] == handeye._sha256(binary)
    assert record["urdf_sha256"] == handeye._sha256(urdf)
    assert record["joint_csv"] == str(joint_csv.resolve())
    assert record["joint_csv_open_fd"] is True


def test_gravity_comp_process_rejects_wrong_can_or_unopened_csv(tmp_path: Path) -> None:
    proc_root, binary, urdf, joint_csv = _fake_gravity_process(
        tmp_path, can_interface="can1",
    )

    with pytest.raises(RuntimeError, match="exactly one matching gravity-comp"):
        handeye.verify_gravity_comp_process(
            arm="right",
            can_interface="can2",
            gravity_comp_bin=binary,
            joint_csv=joint_csv,
            urdf=urdf,
            proc_root=proc_root,
        )

    (proc_root / "1234/fd/4").unlink()
    (proc_root / "1234/cmdline").write_bytes(
        b"\0".join([
            os.fsencode(binary), b"right_arm", b"can2", os.fsencode(urdf), b"",
        ])
    )
    with pytest.raises(RuntimeError, match="exactly one matching gravity-comp"):
        handeye.verify_gravity_comp_process(
            arm="right",
            can_interface="can2",
            gravity_comp_bin=binary,
            joint_csv=joint_csv,
            urdf=urdf,
            proc_root=proc_root,
        )


def test_openarm_joint_limits_reject_out_of_range_fk_input() -> None:
    urdf = HERE.parents[1] / "ausc/perception/robot/openarm_v10.urdf"
    lower, upper = handeye.openarm_joint_limits(urdf, arm="right")
    midpoint = (lower + upper) / 2.0

    handeye.validate_joint_vector(midpoint, lower, upper)
    outside = midpoint.copy()
    outside[3] = upper[3] + 0.01
    with pytest.raises(ValueError, match="URDF limits"):
        handeye.validate_joint_vector(outside, lower, upper)


def test_tip_touch_candidate_passes_varied_pairs_and_heldout_points() -> None:
    tip_camera, rotations, translations = _synthetic_tip_touch_samples()

    candidate = handeye.evaluate_tip_touch_candidate(
        tip_camera,
        rotations,
        translations,
        fit_count=20,
        pair_count=10,
        heldout_count=3,
    )

    assert candidate["accepted"] is True
    assert candidate["failure_reasons"] == []
    assert candidate["training_rms_mm"] < 1e-6
    assert candidate["heldout_max_error_mm"] < 1e-6
    assert candidate["solver_rank"] == 9
    assert min(candidate["pair_rotation_deg"]) == pytest.approx(35.0)


def test_tip_touch_candidate_rejects_same_orientation_pairs() -> None:
    tip_camera, rotations, translations = _synthetic_tip_touch_samples(
        same_pair_orientation=True,
    )

    candidate = handeye.evaluate_tip_touch_candidate(
        tip_camera,
        rotations,
        translations,
        fit_count=20,
        pair_count=10,
        heldout_count=3,
    )

    assert candidate["accepted"] is False
    assert "insufficient_pair_rotation" in candidate["failure_reasons"]

    with pytest.raises(ValueError, match="candidate rejected"):
        handeye.solve_tip_touch(tip_camera, rotations, translations)


def test_tip_touch_candidate_rejects_bad_heldout_error() -> None:
    tip_camera, rotations, translations = _synthetic_tip_touch_samples()
    tip_camera[-1] = tip_camera[-1] + np.asarray((0.020, 0.0, 0.0))

    candidate = handeye.evaluate_tip_touch_candidate(
        tip_camera,
        rotations,
        translations,
        fit_count=20,
        pair_count=10,
        heldout_count=3,
    )

    assert candidate["accepted"] is False
    assert "heldout_error_above_5mm" in candidate["failure_reasons"]


def test_tip_touch_candidate_rejects_common_wrist_axis() -> None:
    tip_camera, rotations, translations = _synthetic_tip_touch_samples(
        pair_axes=[(1.0, 0.0, 0.0)] * 10,
    )

    candidate = handeye.evaluate_tip_touch_candidate(
        tip_camera, rotations, translations,
        fit_count=20, pair_count=10, heldout_count=3,
    )

    assert candidate["accepted"] is False
    assert "insufficient_wrist_axis_excitation" in candidate["failure_reasons"]


@pytest.mark.parametrize(
    "points",
    [
        [(0.40 + i * 0.001, 0.02 + (i % 2) * 0.001, 0.70) for i in range(10)],
        [(0.20 + 0.04 * (i % 5), -0.20 + 0.10 * (i // 5), 0.70) for i in range(10)],
    ],
)
def test_tip_touch_candidate_rejects_clustered_or_planar_fit_points(points) -> None:
    tip_camera, rotations, translations = _synthetic_tip_touch_samples(fit_points=points)

    candidate = handeye.evaluate_tip_touch_candidate(
        tip_camera, rotations, translations,
        fit_count=20, pair_count=10, heldout_count=3,
    )

    assert candidate["accepted"] is False
    assert "insufficient_fit_spatial_coverage" in candidate["failure_reasons"]


def test_tip_touch_candidate_rejects_nonindependent_heldout_points() -> None:
    duplicates = [(0.34, -0.10, 0.57)] * 3
    tip_camera, rotations, translations = _synthetic_tip_touch_samples(
        heldout_points=duplicates,
    )

    candidate = handeye.evaluate_tip_touch_candidate(
        tip_camera, rotations, translations,
        fit_count=20, pair_count=10, heldout_count=3,
    )

    assert candidate["accepted"] is False
    assert "heldout_points_not_independent" in candidate["failure_reasons"]


def test_tip_touch_candidate_rejects_nonfinite_input() -> None:
    tip_camera, rotations, translations = _synthetic_tip_touch_samples()
    tip_camera[-1] = tip_camera[-1].copy()
    tip_camera[-1][0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        handeye.evaluate_tip_touch_candidate(
            tip_camera, rotations, translations,
            fit_count=20, pair_count=10, heldout_count=3,
        )


def test_tip_touch_candidate_rejects_single_fit_outlier() -> None:
    tip_camera, rotations, translations = _synthetic_tip_touch_samples()
    tip_camera[0] = tip_camera[0] + np.asarray((0.008, 0.0, 0.0))

    candidate = handeye.evaluate_tip_touch_candidate(
        tip_camera, rotations, translations,
        fit_count=20, pair_count=10, heldout_count=3,
    )

    assert candidate["accepted"] is False
    assert (
        "paired_position_mismatch" in candidate["failure_reasons"]
        or "training_max_error_above_5mm" in candidate["failure_reasons"]
    )


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError, match="JSON"):
        handeye._canonical_json({"bad": float("nan")})


def test_commissioning_run_is_unique_and_persists_raw_capture(tmp_path: Path) -> None:
    metadata = {"arm": "right", "camera_serial": "CL8326300A0"}
    run = handeye.create_commissioning_run(
        tmp_path, "handeye_right_20260716T190000Z_deadbeef", metadata,
    )
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    depth = np.arange(96, dtype=np.uint16).reshape(8, 12)
    sample = {
        "sample_id": 1,
        "phase": "fit",
        "pixel": [4, 5],
        "depth_m": 0.8,
        "q7_rad": [0.0] * 7,
        "T_tcp_base": np.eye(4).tolist(),
    }

    persisted = handeye.persist_tip_touch_sample(run, sample, rgb, depth)

    assert json.loads((run / "RUN_IN_PROGRESS.json").read_text()) == metadata
    assert persisted["sample_id"] == 1
    assert (run / "captures/0001_rgb.png").is_file()
    assert np.array_equal(np.load(run / "captures/0001_depth.npy"), depth)
    rows = [json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()]
    assert rows == [persisted]
    with pytest.raises(FileExistsError):
        handeye.create_commissioning_run(
            tmp_path, "handeye_right_20260716T190000Z_deadbeef", metadata,
        )


def test_snapshot_file_preserves_the_joint_stream_bytes(tmp_path: Path) -> None:
    run = handeye.create_commissioning_run(
        tmp_path, "handeye_right_20260716T190004Z_deadbeef", {"arm": "right"},
    )
    source = tmp_path / "live-joints.csv"
    source.write_bytes(b"0.0,1,2,3,4,5,6,7,0\n")

    record = handeye.snapshot_file(run, source, "joint_stream_snapshot.csv")

    assert (run / "joint_stream_snapshot.csv").read_bytes() == source.read_bytes()
    assert record["size_bytes"] == len(source.read_bytes())
    assert record["sha256"] == handeye._sha256(run / "joint_stream_snapshot.csv")
    with pytest.raises(FileExistsError):
        handeye.snapshot_file(run, source, "joint_stream_snapshot.csv")


def test_finalize_commissioning_run_is_fail_closed(tmp_path: Path) -> None:
    rejected = handeye.create_commissioning_run(
        tmp_path, "handeye_right_20260716T190001Z_deadbeef", {"arm": "right"},
    )
    rejected_candidate = {
        "accepted": False,
        "failure_reasons": ["heldout_error_above_5mm"],
    }

    rejected_result = handeye.finalize_commissioning_run(
        rejected, rejected_candidate, {"code_sha256": "a" * 64},
    )

    assert rejected_result["status"] == "REJECTED"
    assert (rejected / "FAILED.json").is_file()
    assert not (rejected / "COMPLETE.json").exists()
    assert not (rejected / "cam2base_candidate.json").exists()

    accepted = handeye.create_commissioning_run(
        tmp_path, "handeye_right_20260716T190002Z_deadbeef", {"arm": "right"},
    )
    accepted_candidate = {
        "accepted": True,
        "failure_reasons": [],
        "T_cam_base": np.eye(4).tolist(),
        "tip_offset_tcp_m": [0.0, 0.0, 0.1],
    }

    accepted_result = handeye.finalize_commissioning_run(
        accepted, accepted_candidate, {"code_sha256": "b" * 64},
    )

    assert accepted_result["status"] == "CANDIDATE_COMPLETE_NOT_ACTIVATED"
    assert (accepted / "cam2base_candidate.json").is_file()
    assert (accepted / "SHA256SUMS.txt").is_file()
    assert (accepted / "COMPLETE.json").is_file()
    assert not (accepted / "RUN_IN_PROGRESS.json").exists()


def test_finalize_keeps_in_progress_marker_if_completion_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = handeye.create_commissioning_run(
        tmp_path, "handeye_right_20260716T190005Z_deadbeef", {"arm": "right"},
    )
    original = handeye._write_new

    def fail_inventory(path: Path, body: bytes, mode: int = 0o644) -> None:
        if Path(path).name == "SHA256SUMS.txt":
            raise OSError("injected inventory failure")
        original(path, body, mode)

    monkeypatch.setattr(handeye, "_write_new", fail_inventory)

    with pytest.raises(OSError, match="injected inventory failure"):
        handeye.finalize_commissioning_run(
            run,
            {
                "accepted": True,
                "failure_reasons": [],
                "T_cam_base": np.eye(4).tolist(),
                "tip_offset_tcp_m": [0.0, 0.0, 0.1],
            },
            {"code_sha256": "c" * 64},
        )

    assert (run / "RUN_IN_PROGRESS.json").is_file()
    assert not (run / "COMPLETE.json").exists()


def test_write_new_handles_partial_os_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = handeye.os.write

    def partial(descriptor: int, body: bytes) -> int:
        count = max(1, len(body) // 3)
        return original(descriptor, body[:count])

    monkeypatch.setattr(handeye.os, "write", partial)
    target = tmp_path / "payload.bin"
    body = bytes(range(251)) * 20

    handeye._write_new(target, body)

    assert target.read_bytes() == body
