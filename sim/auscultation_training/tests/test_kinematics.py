"""Pure numerical kinematics tests; no Isaac Sim or hardware is imported."""

import importlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def make_model(path=None):
    module = importlib.import_module("sim.auscultation_training.kinematics")
    return module.OpenArmKinematics() if path is None else module.OpenArmKinematics(path)


def test_local_posture_prior_avoids_redundant_jump_for_small_cartesian_step():
    model = make_model()
    q = np.array([-2.4715939439,.0324937305,-1.1968704974,1.7166386211,1.2083090574,.7798276264,.6177787680])
    pose = model.tcp(q)
    destination = pose[:3,3]+pose[:3,2]*.0006
    result = model.solve_ik(destination,orientation_base=pose[:3,:3],seed=q,
                            max_starts=1,max_nfev=80,posture_weight=.003)
    assert result.success
    assert np.max(np.abs(result.q-q)) < .0125
    assert result.position_error_m < 1e-5
    assert result.orientation_error_rad < 1e-4


def test_posture_prior_rejects_invalid_weight():
    model = make_model()
    pose = model.tcp(np.zeros(7))
    with pytest.raises(ValueError,match='posture_weight'):
        model.solve_ik(pose[:3,3],orientation_base=pose[:3,:3],posture_weight=-1)


def test_zero_configuration_matches_urdf_chain():
    model = make_model()
    transforms = model.link_transforms(np.zeros(7))
    expected_positions = [
        [0, 0, 0], [0, 0, 0.0625], [-0.0301, 0, 0.1225],
        [0, 0.06625, 0.1225], [0, 0.22, 0.091],
        [0, 0.3155, 0.1225], [0.0375, 0.436, 0.1225],
        [0, 0.436, 0.1225],
    ]
    assert list(transforms) == [f"openarm_left_link{i}" for i in range(8)]
    for i, position in enumerate(expected_positions):
        np.testing.assert_allclose(transforms[f"openarm_left_link{i}"][:3, 3], position, atol=1e-10)
    np.testing.assert_allclose(model.fk(np.zeros(7)), transforms["openarm_left_link7"])
    np.testing.assert_allclose(model.fk(np.zeros(7))[:3, :3], Rotation.from_euler("x", -np.pi / 2).as_matrix(), atol=1e-10)


def test_joint_two_negative_axis_straightens_chain():
    q = np.array([0, -np.pi / 2, 0, 0, 0, 0, 0])
    wrist = make_model().fk(q)
    np.testing.assert_allclose(wrist[:3, 3], [0, 0, 0.5585], atol=1e-10)
    np.testing.assert_allclose(wrist[:3, :3], np.eye(3), atol=1e-10)


def test_tcp_is_diaphragm_and_compression_retracts_along_tool_axis():
    model = make_model()
    q = np.zeros(7)
    tcp = model.tcp(q)
    np.testing.assert_allclose(tcp[:3, 3], [0, 0.5622, 0.1225], atol=1e-10)
    compressed = model.tcp(q, compression=0.005)
    np.testing.assert_allclose(compressed[:3, 3], tcp[:3, 3] - 0.005 * tcp[:3, 2])
    frames = importlib.import_module("sim.auscultation_training.tool_frames")
    np.testing.assert_allclose(frames.wrist_from_diaphragm(0.005)[:3, 3], [0, 0, 0.1212])
    np.testing.assert_allclose(frames.wrist_from_outer_lip(0.005)[:3, 3], [0, 0, 0.1220])


def test_geometric_jacobian_matches_pose_finite_difference():
    model = make_model()
    q = np.array([-0.8, -1.7, 0.4, 0.9, -0.2, 0.3, -0.4])
    epsilon = 1e-6
    expected = np.empty((6, 7))
    for j in range(7):
        dq = np.eye(7)[j] * epsilon
        plus = model.tcp(q + dq, compression=0.003)
        minus = model.tcp(q - dq, compression=0.003)
        expected[:3, j] = (plus[:3, 3] - minus[:3, 3]) / (2 * epsilon)
        expected[3:, j] = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / (2 * epsilon)
    np.testing.assert_allclose(model.jacobian(q, compression=0.003), expected, atol=2e-8)


def test_joint_limits_are_read_from_urdf_and_enforced():
    model = make_model()
    np.testing.assert_allclose(model.lower_limits, [-3.490659, -3.3161253267948965, -1.570796, 0, -1.570796, -0.785398, -1.570796])
    np.testing.assert_allclose(model.upper_limits, [1.396263, 0.17453267320510335, 1.570796, 2.443461, 1.570796, 0.785398, 1.570796])
    assert model.joint_names == tuple(f"openarm_left_joint{i}" for i in range(1, 8))
    model.fk(model.lower_limits)
    model.fk(model.upper_limits)
    with pytest.raises(ValueError, match="limit"):
        model.fk(model.lower_limits - 0.01)


@pytest.mark.parametrize("q", [[0] * 6, [[0] * 7], [0, 0, 0, np.nan, 0, 0, 0], [0, 0, 0, np.inf, 0, 0, 0]])
def test_rejects_malformed_joint_vectors(q):
    with pytest.raises(ValueError):
        make_model().fk(q)


@pytest.mark.parametrize("compression", [-0.001, np.nan, np.inf, 0.007251, 0.2])
def test_rejects_invalid_compression(compression):
    with pytest.raises(ValueError, match="compression"):
        make_model().tcp(np.zeros(7), compression=compression)


def test_rejects_malformed_urdf_joint_axis(tmp_path):
    source = Path(__file__).parents[1] / "openarm_v10_follower_stetho.urdf"
    root = ET.parse(source)
    root.find("./joint[@name='openarm_left_joint3']/axis").set("xyz", "0 0 0")
    destination = tmp_path / "invalid.urdf"
    root.write(destination)
    with pytest.raises(ValueError, match="axis"):
        make_model(destination)


@pytest.mark.parametrize("q", [
    [-0.8, -1.7, 0.4, 0.9, -0.2, 0.3, -0.4],
    [0.4, -0.6, -0.8, 1.7, 0.9, -0.4, 0.7],
    [-2.5, -2.8, 1.1, 0.6, 0.5, 0.4, -0.9],
])
def test_axis_ik_recovers_reachable_compressed_tcp_poses(q):
    model = make_model()
    expected = model.tcp(q, compression=0.004)
    result = model.solve_ik(expected[:3, 3], expected[:3, 2], compression=0.004)
    assert result.success, result
    assert result.position_error_m < 1e-4
    assert result.orientation_error_rad < 1e-3
    actual = model.tcp(result.q, compression=0.004)
    np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=1e-4)
    np.testing.assert_allclose(actual[:3, 2], expected[:3, 2], atol=1e-3)
    assert np.all(result.q >= model.lower_limits)
    assert np.all(result.q <= model.upper_limits)


def test_axis_ik_does_not_accept_antiparallel_seed():
    model = make_model()
    seed = np.zeros(7)
    desired = model.tcp([-np.pi, 0, 0, 0, 0, 0, 0])
    assert np.dot(desired[:3, 2], model.tcp(seed)[:3, 2]) == pytest.approx(-1)
    result = model.solve_ik(desired[:3, 3], desired[:3, 2], seed=seed)
    assert result.success, result
    assert np.dot(model.tcp(result.q)[:3, 2], desired[:3, 2]) > 0.999999


def test_full_orientation_ik_roundtrip():
    model = make_model()
    desired = model.tcp([-0.8, -1.7, 0.4, 0.9, -0.2, 0.3, -0.4])
    result = model.solve_ik(desired[:3, 3], orientation_base=desired[:3, :3])
    assert result.success, result
    np.testing.assert_allclose(model.tcp(result.q), desired, atol=1e-4)


def test_axis_ik_keeps_roll_free_and_accepts_existing_solution():
    model = make_model()
    q = np.array([-0.8, -1.7, 0.4, 0.9, -0.2, 0.3, -0.4])
    desired = model.tcp(q)
    result = model.solve_ik(desired[:3, 3], 2 * desired[:3, 2], seed=q, max_starts=1)
    assert result.success
    assert result.attempts == 1
    np.testing.assert_allclose(result.q, q, atol=1e-7)


def test_unreachable_ik_reports_failure_without_violating_limits():
    model = make_model()
    result = model.solve_ik([10, 10, 10], [0, 0, 1], max_starts=2, max_nfev=40)
    assert not result.success
    assert result.position_error_m > 1
    assert result.attempts == 2
    assert np.isfinite(result.q).all()
    assert np.all(result.q >= model.lower_limits)
    assert np.all(result.q <= model.upper_limits)


@pytest.mark.parametrize("position,axis,kwargs", [
    ([0, np.nan, 0], [0, 0, 1], {}),
    ([0, 0, 0], [0, 0, 0], {}),
    ([0, 0, 0], [0, 0, np.inf], {}),
    ([0, 0, 0], None, {}),
    ([0, 0, 0], [0, 0, 1], {"max_starts": 0}),
    ([0, 0, 0], [0, 0, 1], {"max_nfev": -1}),
    ([0, 0, 0], [0, 0, 1], {"position_tolerance": 0}),
    ([0, 0, 0], [0, 0, 1], {"orientation_tolerance": np.nan}),
    ([0, 0, 0], [0, 0, 1], {"seed": [20] * 7}),
    ([0, 0, 0], None, {"orientation_base": np.zeros((3, 3))}),
    ([0, 0, 0], [0, 0, -1], {"orientation_base": np.eye(3)}),
])
def test_rejects_malformed_ik_inputs(position, axis, kwargs):
    with pytest.raises(ValueError):
        make_model().solve_ik(position, axis, **kwargs)
