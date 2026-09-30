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
