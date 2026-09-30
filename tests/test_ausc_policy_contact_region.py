import numpy as np
import pytest
from sim.ausc_policy import axis_ik
from sim.ausc_policy.geometry_epoch import contact_frame
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from sim.auscultation_expert_layout_v1.posture import PostureLimits


def test_contact_region_keeps_original_target_and_reports_residual():
    solve = getattr(axis_ik, 'solve_contact_region', None)
    assert callable(solve), 'Need bounded contact-region IK, not target relabeling'
    kin = OpenArmKinematics(contact_frame=contact_frame)
    q = (kin.lower_limits + kin.upper_limits) / 2
    pose = kin.tcp(q)
    target = pose[:3, 3] + .001 * pose[:3, 0]
    before = target.copy()
    limits = PostureLimits(10., tuple(kin.lower_limits), tuple(kin.upper_limits))
    result = solve(kin, np.eye(4), limits, target, pose[:3, 2], q, contact=False)
    assert result['success']
    np.testing.assert_array_equal(target, before)
    measured = kin.tcp(result['q'])[:3, 3] - target
    assert abs(measured @ pose[:3, 2]) < 2e-5
    assert np.linalg.norm(measured) <= .002 + 1e-8
    assert result['collision_checked'] is False
    np.testing.assert_allclose(result['position_residual_base_m'], measured)


def test_contact_region_cannot_relax_original_limits():
    solve = getattr(axis_ik, 'solve_contact_region', None)
    assert callable(solve)
    kin = OpenArmKinematics(contact_frame=contact_frame)
    q = (kin.lower_limits + kin.upper_limits) / 2
    pose = kin.tcp(q)
    limits = PostureLimits(10., tuple(kin.lower_limits), tuple(kin.upper_limits))
    for kwargs in ({'tangent_limit_m': .0031}, {'angle_limit_deg': 5.1}):
        with pytest.raises(ValueError):
            solve(kin, np.eye(4), limits, pose[:3,3], pose[:3,2], q, **kwargs)


def test_impossible_target_returns_failure_without_clipping_into_acceptance():
    solve = getattr(axis_ik, 'solve_contact_region', None)
    assert callable(solve)
    kin = OpenArmKinematics(contact_frame=contact_frame)
    q = (kin.lower_limits + kin.upper_limits) / 2
    limits = PostureLimits(10., tuple(kin.lower_limits), tuple(kin.upper_limits))
    result = solve(kin, np.eye(4), limits, [10,10,10], [0,0,1], q, contact=False)
    assert not result['success']


def test_zero_joint_stop_also_has_contact_planning_margin():
    kin = OpenArmKinematics(contact_frame=contact_frame)
    q = (kin.lower_limits + kin.upper_limits) / 2
    q[3] = 0.
    pose = kin.tcp(q)
    limits = PostureLimits(10., tuple(kin.lower_limits), tuple(kin.upper_limits))
    result = axis_ik.solve_contact_region(kin, np.eye(4), limits, pose[:3,3], pose[:3,2], q, contact=False)
    assert not result['success'] or result['q'][3] >= np.deg2rad(.2)-1e-10


def test_nonfinite_posture_limit_is_rejected():
    kin = OpenArmKinematics(contact_frame=contact_frame)
    q = (kin.lower_limits + kin.upper_limits) / 2
    pose = kin.tcp(q)
    limits = PostureLimits(float('inf'), tuple(kin.lower_limits), tuple(kin.upper_limits))
    with pytest.raises(ValueError):
        axis_ik.solve_contact_region(kin, np.eye(4), limits, pose[:3,3], pose[:3,2], q, contact=False)


@pytest.mark.parametrize('compression', [.001, .003, .006])
def test_loaded_contact_region_keeps_known_q_and_measures_loaded_tcp(compression):
    kin = OpenArmKinematics(contact_frame=contact_frame)
    q = (kin.lower_limits + kin.upper_limits) / 2
    pose = kin.tcp(q, compression=compression)
    target = pose[:3, 3].copy()
    before = target.copy()
    limits = PostureLimits(10., tuple(kin.lower_limits), tuple(kin.upper_limits))
    result = axis_ik.solve_contact_region(kin, np.eye(4), limits, target, pose[:3, 2], q,
                                         contact=False, compression=compression)
    assert result['success']
    np.testing.assert_array_equal(target, before)
    np.testing.assert_allclose(result['q'], q, atol=1e-9, rtol=0)
    np.testing.assert_allclose(result['position_residual_base_m'],
        kin.tcp(result['q'], compression=compression)[:3, 3]-target, atol=1e-12, rtol=0)
    assert result['collision_checked'] is False
    assert result['physics_executed'] is False


def test_loaded_region_derivatives_match_central_differences_and_pose_offset(monkeypatch):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation

    compression = .004
    kin = OpenArmKinematics(contact_frame=contact_frame)
    q = (kin.lower_limits + kin.upper_limits) / 2
    pose = kin.tcp(q, compression=compression)
    target = pose[:3, 3] + .0007 * pose[:3, 0]
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler('xyz', [.13, -.27, .18]).as_matrix()
    base[:3, 3] = [.1, .2, .8]
    limits = PostureLimits(10., tuple(kin.lower_limits), tuple(kin.upper_limits))
    offsets = []
    original_pose = axis_ik.PoseConstraint

    class RecordingPose(original_pose):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            offsets.append(self.offset.copy())

    checked = []

    def fake_minimize(fun, x0, *, jac, constraints, **kwargs):
        probe = x0 + np.array([.017, -.013, .011, .014, -.016, .012, -.018])
        for function, derivative in [(fun, jac)] + [(c['fun'], c['jac']) for c in constraints]:
            numerical = np.array([(function(probe + np.eye(7)[j]*1e-6)
                                   - function(probe - np.eye(7)[j]*1e-6))/(2e-6)
                                  for j in range(7)])
            analytic = np.asarray(derivative(probe))
            np.testing.assert_allclose(analytic, numerical, atol=2e-6, rtol=2e-6)
            checked.append(True)
        return SimpleNamespace(x=x0)

    monkeypatch.setattr(axis_ik, 'PoseConstraint', RecordingPose)
    monkeypatch.setattr(axis_ik, 'minimize', fake_minimize)
    axis_ik.solve_contact_region(kin, base, limits, target, pose[:3, 2], q,
                                 compression=compression, contact=True)
    assert len(checked) == 6  # objective, plane, radius, cone, elbow, wrist gap
    assert len(offsets) == 1
    np.testing.assert_allclose(offsets[0], contact_frame(compression), atol=1e-15, rtol=0)


@pytest.mark.parametrize('compression', [-.0001, .0061, np.nan, np.inf, [.001]])
def test_contact_region_rejects_invalid_compression(compression):
    kin = OpenArmKinematics(contact_frame=contact_frame)
    q = (kin.lower_limits + kin.upper_limits) / 2
    pose = kin.tcp(q)
    limits = PostureLimits(10., tuple(kin.lower_limits), tuple(kin.upper_limits))
    with pytest.raises(ValueError, match='compression'):
        axis_ik.solve_contact_region(kin, np.eye(4), limits, pose[:3, 3], pose[:3, 2], q,
                                     compression=compression, contact=False)


def test_loaded_region_rejects_retired_tcp_contract():
    kin = OpenArmKinematics()  # Retired default 126.2 mm contact frame.
    q = (kin.lower_limits + kin.upper_limits) / 2
    pose = kin.tcp(q, compression=.003)
    limits = PostureLimits(10., tuple(kin.lower_limits), tuple(kin.upper_limits))
    with pytest.raises(ValueError, match='132 mm'):
        axis_ik.solve_contact_region(kin, np.eye(4), limits, pose[:3, 3], pose[:3, 2], q,
                                     compression=.003, contact=False)
