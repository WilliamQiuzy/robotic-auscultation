"""Pure-contract tests for the official OpenArm Isaac occlusion experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ausc.perception.tracking import official_openarm_experiment as experiment


OPENARM_URDF = (
    Path(__file__).resolve().parents[2] / "robot" / "openarm_v10.urdf"
)


def _back_plane_grid(rows: int = 9, columns: int = 9):
    """Return a triangulated X/Z plane whose normals point along negative Y."""

    xs = np.linspace(-0.30, 0.30, columns)
    zs = np.linspace(0.75, 1.55, rows)
    points = np.array([[x, 0.0, z] for z in zs for x in xs], dtype=np.float64)
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = (row + 1) * columns + column
            upper_right = upper_left + 1
            # This winding produces a -Y normal in the X/Y/Z convention.
            faces.append([lower_left, lower_right, upper_left])
            faces.append([lower_right, upper_right, upper_left])
    return points, np.asarray(faces, dtype=np.int64)


def test_pinned_v1_urdf_is_full_dual_arm_and_not_proxy_geometry() -> None:
    inventory = experiment.inspect_openarm_urdf(OPENARM_URDF.read_bytes())

    assert inventory.robot_name == "openarm"
    assert inventory.link_count == 24
    assert inventory.joint_count == 23
    assert inventory.revolute_joint_names == (
        "openarm_left_joint1",
        "openarm_left_joint2",
        "openarm_left_joint3",
        "openarm_left_joint4",
        "openarm_left_joint5",
        "openarm_left_joint6",
        "openarm_left_joint7",
        "openarm_right_joint1",
        "openarm_right_joint2",
        "openarm_right_joint3",
        "openarm_right_joint4",
        "openarm_right_joint5",
        "openarm_right_joint6",
        "openarm_right_joint7",
    )
    assert inventory.left_follower_joints == experiment.LEFT_FOLLOWER_JOINTS
    assert inventory.right_fixed_joints == experiment.RIGHT_REFERENCE_JOINTS
    assert inventory.mesh_reference_count == 42
    assert inventory.unique_mesh_count == 20
    assert not inventory.has_primitive_geometry


def test_urdf_inventory_rejects_primitive_or_incomplete_substitution() -> None:
    primitive = b"""<robot name="openarm"><link name="one"><visual><geometry>
      <cylinder radius="0.14" length="1"/>
    </geometry></visual></link></robot>"""

    with pytest.raises(experiment.ExperimentContractError, match="primitive|24 links"):
        experiment.inspect_openarm_urdf(primitive)


def test_camera_projection_uses_frozen_orbbec_intrinsics() -> None:
    points = np.array(
        [[0.0, 0.0, 1.0], [0.10, -0.05, 2.0], [-0.20, 0.10, 0.5]],
        dtype=np.float64,
    )

    pixels = experiment.project_camera_points(points, experiment.ORBBEC_RGB_INTRINSICS)

    assert np.allclose(pixels[0], [955.7200927734375, 533.1757202148438])
    assert np.allclose(
        pixels[1],
        [955.7200927734375 + 1123.173095703125 * 0.05,
         533.1757202148438 - 1122.9962158203125 * 0.025],
    )
    with pytest.raises(experiment.ExperimentContractError, match="positive camera depth"):
        experiment.project_camera_points(
            np.array([[0.0, 0.0, 0.0]]), experiment.ORBBEC_RGB_INTRINSICS
        )


def test_ten_point_layout_is_persisted_on_official_mesh_triangles() -> None:
    vertices, faces = _back_plane_grid()

    anchors = experiment.anchor_ten_points_to_back_mesh(
        vertices,
        faces,
        back_axis=1,
        back_sign=-1,
    )

    assert tuple(anchor.name for anchor in anchors) == experiment.POINT_NAMES
    assert len({anchor.triangle_id for anchor in anchors}) == 10
    assert all(np.isclose(np.sum(anchor.barycentric), 1.0) for anchor in anchors)
    assert all(np.all(anchor.barycentric >= 0.0) for anchor in anchors)
    world = np.asarray([anchor.position_m for anchor in anchors])
    # Preserve the WAM R/L identity and paired top-to-bottom ordering.
    assert np.all(world[0::2, 0] < world[1::2, 0])
    assert np.all(np.diff(world[0::2, 2]) < 0.0)
    assert np.all(np.diff(world[1::2, 2]) < 0.0)
    assert all(anchor.surface_normal[1] < -0.99 for anchor in anchors)


def test_visibility_comes_only_from_rendered_masks_and_depth() -> None:
    height = width = 80
    human = np.zeros((height, width), dtype=bool)
    robot = np.zeros((height, width), dtype=bool)
    depth = np.full((height, width), np.nan, dtype=np.float32)
    human[30:51, 30:51] = True
    depth[human] = 1.0
    robot[37:44, 37:44] = True
    human[robot] = False
    depth[robot] = 0.82
    centers = np.array([[40.0, 40.0], [34.0, 34.0]])

    results = experiment.measure_rendered_visibility(
        point_names=("R1", "L1"),
        projected_pixels=centers,
        point_depth_m=np.array([1.0, 1.0]),
        robot_mask=robot,
        human_mask=human,
        depth_m=depth,
        support_radius_px=6,
        robot_mask_dilation_px=1,
        front_depth_margin_m=0.003,
    )

    assert tuple(row.name for row in results) == ("R1", "L1")
    assert results[0].robot_front_fraction > results[1].robot_front_fraction
    assert results[0].visible_fraction < results[1].visible_fraction
    assert results[0].mask_edge_distance_px <= results[1].mask_edge_distance_px
    assert 0.0 < results[0].valid_depth_fraction <= 1.0
    assert results[0].visibility_source == "rendered_instance_mask_and_depth"


def test_visibility_contract_rejects_analytic_capsule_inputs() -> None:
    masks = np.zeros((16, 16), dtype=bool)
    depth = np.ones((16, 16), dtype=np.float32)

    with pytest.raises(TypeError):
        experiment.measure_rendered_visibility(
            point_names=("R1",),
            projected_pixels=np.array([[8.0, 8.0]]),
            point_depth_m=np.array([1.0]),
            robot_mask=masks,
            human_mask=~masks,
            depth_m=depth,
            support_radius_px=3,
            robot_mask_dilation_px=1,
            front_depth_margin_m=0.003,
            capsule_radius_m=0.14,
        )


def test_stage_inventory_requires_unit_scale_and_only_left_chain_motion() -> None:
    valid = {
        "meters_per_unit": 1.0,
        "import_root_scale": 1.0,
        "link_count": 24,
        "joint_count": 23,
        "revolute_joint_names": list(
            experiment.LEFT_FOLLOWER_JOINTS + experiment.RIGHT_REFERENCE_JOINTS
        ),
        "animated_joint_names": list(experiment.LEFT_FOLLOWER_JOINTS),
        "robot_pixel_count": 1200,
        "human_pixel_count": 8000,
        "human_asset_sha256": experiment.NVIDIA_HUMAN_SHA256,
        "robot_urdf_sha256": experiment.OFFICIAL_CLEAN_URDF_SHA256,
    }

    assert experiment.validate_stage_inventory(valid)["animated_joint_names"] == list(
        experiment.LEFT_FOLLOWER_JOINTS
    )
    for key, bad_value in (
        ("meters_per_unit", 0.001),
        ("import_root_scale", 0.001),
        ("link_count", 8),
        ("joint_count", 7),
        ("robot_pixel_count", 0),
        ("human_pixel_count", 0),
    ):
        changed = dict(valid)
        changed[key] = bad_value
        with pytest.raises(experiment.ExperimentContractError):
            experiment.validate_stage_inventory(changed)
    changed = dict(valid)
    changed["animated_joint_names"] = list(experiment.RIGHT_REFERENCE_JOINTS)
    with pytest.raises(experiment.ExperimentContractError, match="left follower"):
        experiment.validate_stage_inventory(changed)


def test_scenarios_are_deterministic_and_pose_is_never_commissioned() -> None:
    first = experiment.scenario_name("R3", 1, "cam_yaw_plus10", 7)
    second = experiment.scenario_name("R3", 1, "cam_yaw_plus10", 7)

    assert first == second == "R3__wrist01__cam_yaw_plus10__seed0007"
    contract = experiment.primary_experiment_contract()
    assert contract["robot_geometry"] == "official_openarm_v1_full_dual_arm"
    assert contract["human_geometry"] == "official_nvidia_male_adult_medical_01"
    assert contract["visibility_source"] == "rendered_instance_mask_and_depth"
    assert contract["camera_pose_status"] == "installation_pose_uncommissioned"
    assert contract["installation_specific_pass"] is False
    assert contract["result_scope"] == "parameter_sensitivity_only"
    assert "min_visible" not in contract
    assert "capsule" not in " ".join(map(str, contract.values())).lower()


def test_numerical_ik_uses_the_real_left_openarm_chain() -> None:
    raw = OPENARM_URDF.read_bytes()
    known_q = np.array([-1.0, -1.2, 0.15, 1.15, -0.10, 0.20, -0.35])
    target = experiment.left_tip_forward_kinematics(raw, known_q)[:3, 3]

    solution = experiment.solve_left_position_ik(
        raw,
        target,
        initial_q=known_q + np.array([0.08, -0.05, 0.04, -0.06, 0.03, -0.02, 0.05]),
        tolerance_m=2.0e-5,
        max_iterations=200,
    )
    solved = experiment.left_tip_forward_kinematics(raw, solution)[:3, 3]
    limits = experiment.left_joint_limits(raw)

    assert np.linalg.norm(solved - target) <= 2.0e-5
    assert np.all(solution >= limits[:, 0])
    assert np.all(solution <= limits[:, 1])


def test_joint_trajectory_rejects_limit_velocity_acceleration_or_root_motion() -> None:
    raw = OPENARM_URDF.read_bytes()
    limits = experiment.left_joint_limits(raw)
    velocity_limits = experiment.left_joint_velocity_limits(raw)
    frames = 61
    time_s = np.linspace(0.0, 2.0, frames)
    start = np.array([-1.0, -1.2, 0.0, 1.10, 0.0, 0.0, -0.3])
    end = np.array([-0.9, -1.1, 0.1, 1.20, 0.1, 0.1, 0.3])
    blend = experiment.quintic_blend(np.linspace(0.0, 1.0, frames))[:, None]
    positions = start + blend * (end - start)
    root_poses = np.repeat(np.eye(4)[None, ...], frames, axis=0)

    evidence = experiment.validate_left_joint_trajectory(
        positions,
        time_s,
        root_poses,
        joint_limits=limits,
        velocity_limits=velocity_limits,
        acceleration_limits=np.full(7, 8.0),
    )

    assert evidence["joint_names"] == list(experiment.LEFT_FOLLOWER_JOINTS)
    assert evidence["root_pose_animated"] is False
    assert evidence["within_position_limits"] is True
    assert evidence["within_velocity_limits"] is True
    assert evidence["within_acceleration_limits"] is True
    bad_root = root_poses.copy()
    bad_root[-1, 0, 3] = 0.01
    with pytest.raises(experiment.ExperimentContractError, match="root pose"):
        experiment.validate_left_joint_trajectory(
            positions,
            time_s,
            bad_root,
            joint_limits=limits,
            velocity_limits=velocity_limits,
            acceleration_limits=np.full(7, 8.0),
        )
    bad_position = positions.copy()
    bad_position[-1, 0] = limits[0, 1] + 0.01
    with pytest.raises(experiment.ExperimentContractError, match="joint limit"):
        experiment.validate_left_joint_trajectory(
            bad_position,
            time_s,
            root_poses,
            joint_limits=limits,
            velocity_limits=velocity_limits,
            acceleration_limits=np.full(7, 8.0),
        )


def test_position_ik_candidates_keep_target_and_select_distinct_wrist_attitudes() -> None:
    raw = OPENARM_URDF.read_bytes()
    known_q = np.array([-1.0, -1.2, 0.15, 1.15, -0.10, 0.20, -0.35])
    target = experiment.left_tip_forward_kinematics(raw, known_q)[:3, 3]
    seeds = np.array(
        [
            known_q + np.array([0.08, -0.05, 0.04, -0.06, 0.03, -0.02, 0.05]),
            [0.71, -2.71, 0.98, 1.15, 0.70, -0.42, -1.35],
            [0.92, -1.72, -1.24, 1.15, -1.24, 0.18, 1.08],
            [-0.99, -1.16, 0.08, 1.15, -0.88, 0.55, 1.44],
        ],
        dtype=np.float64,
    )

    candidates = experiment.solve_left_position_ik_candidates(
        raw,
        target,
        initial_seeds=seeds,
        tolerance_m=2.0e-4,
        max_iterations=250,
        max_solutions=2,
        min_orientation_separation_rad=np.deg2rad(20.0),
    )

    assert len(candidates) == 2
    assert all(candidate.position_error_m <= 2.0e-4 for candidate in candidates)
    assert all(candidate.joint_positions.shape == (7,) for candidate in candidates)
    assert candidates[1].orientation_separation_from_first_rad >= np.deg2rad(20.0)
    for candidate in candidates:
        actual = experiment.left_tip_forward_kinematics(raw, candidate.joint_positions)
        assert np.linalg.norm(actual[:3, 3] - target) <= 2.0e-4
        assert np.allclose(actual[:3, :3], candidate.tip_rotation)


def test_minimum_jerk_waypoint_trajectory_is_deterministic_and_has_fixed_root() -> None:
    waypoints = np.array(
        [
            [-1.0, -1.2, 0.0, 1.10, 0.0, 0.0, -0.30],
            [-0.9, -1.1, 0.1, 1.20, 0.1, 0.1, 0.30],
            [-1.1, -1.0, -0.1, 1.05, -0.1, -0.1, 0.10],
        ],
        dtype=np.float64,
    )

    first = experiment.build_minimum_jerk_trajectory(
        waypoints,
        frames_per_segment=25,
        seconds_per_segment=2.0,
    )
    second = experiment.build_minimum_jerk_trajectory(
        waypoints,
        frames_per_segment=25,
        seconds_per_segment=2.0,
    )

    assert np.array_equal(first.joint_positions, second.joint_positions)
    assert np.array_equal(first.time_s, second.time_s)
    assert np.array_equal(first.root_poses, second.root_poses)
    assert np.allclose(first.joint_positions[0], waypoints[0])
    assert np.allclose(first.joint_positions[24], waypoints[1])
    assert np.allclose(first.joint_positions[-1], waypoints[-1])
    assert len(first.joint_positions) == 49
    assert np.max(np.abs(first.root_poses - np.eye(4))) == 0.0
    assert np.all(np.diff(first.time_s) > 0.0)


def test_bounded_freeze_policy_uses_uncertainty_and_time_not_visible_count() -> None:
    direct = experiment.decide_occlusion_policy(
        geometry_accepted=True,
        last_valid_age_s=10.0,
        last_target_error_p95_m=0.020,
        patient_motion_bound_m_s=0.010,
        max_freeze_age_s=0.40,
        max_freeze_error_p95_m=0.010,
    )
    short_hold = experiment.decide_occlusion_policy(
        geometry_accepted=False,
        last_valid_age_s=0.20,
        last_target_error_p95_m=0.004,
        patient_motion_bound_m_s=0.010,
        max_freeze_age_s=0.40,
        max_freeze_error_p95_m=0.010,
    )
    too_old = experiment.decide_occlusion_policy(
        geometry_accepted=False,
        last_valid_age_s=0.41,
        last_target_error_p95_m=0.004,
        patient_motion_bound_m_s=0.010,
        max_freeze_age_s=0.40,
        max_freeze_error_p95_m=0.010,
    )
    too_uncertain = experiment.decide_occlusion_policy(
        geometry_accepted=False,
        last_valid_age_s=0.30,
        last_target_error_p95_m=0.008,
        patient_motion_bound_m_s=0.010,
        max_freeze_age_s=0.40,
        max_freeze_error_p95_m=0.010,
    )

    assert direct.state == "DIRECT_GEOMETRY_UPDATE"
    assert short_hold.state == "BOUNDED_PATIENT_FRAME_FREEZE"
    assert np.isclose(short_hold.predicted_error_p95_m, 0.006)
    assert too_old.state == "LOST"
    assert too_uncertain.state == "LOST"
    assert all(decision.visible_point_count_rule is False for decision in (
        direct, short_hold, too_old, too_uncertain
    ))
