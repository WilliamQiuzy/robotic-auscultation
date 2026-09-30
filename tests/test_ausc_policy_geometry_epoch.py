import numpy as np
import pytest
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from sim.auscultation_expert_layout_v1.pose_ik import PoseConstraint


def frame(compression=0.):
    t = np.eye(4)
    t[2, 3] = .132 - compression
    return t


def test_new_tcp_and_jacobian_use_same_geometry():
    kin = OpenArmKinematics(contact_frame=frame)
    q = np.array([.1, -.2, -.1, .3, .1, .1, -.2])
    np.testing.assert_allclose(kin.tcp(q), kin.fk(q) @ frame())
    numeric = np.column_stack([(kin.tcp(q + np.eye(7)[i]*1e-6)[:3,3]
        - kin.tcp(q - np.eye(7)[i]*1e-6)[:3,3])/2e-6 for i in range(7)])
    np.testing.assert_allclose(kin.jacobian(q)[:3], numeric, atol=1e-8)
    pose = PoseConstraint(kin, np.eye(4), kin.tcp(q)[:3,3], kin.tcp(q)[:3,:3])
    np.testing.assert_allclose(pose.residual(q), 0., atol=1e-10)


def test_new_ik_targets_actual_new_membrane():
    kin = OpenArmKinematics(contact_frame=frame)
    q = np.array([.1, -.2, -.1, .3, .1, .1, -.2])
    target = kin.tcp(q)
    result = kin.solve_ik(target[:3,3], orientation_base=target[:3,:3], seed=q, max_starts=1)
    assert result.success
    np.testing.assert_allclose(result.q, q, atol=1e-7)


def test_old_inputs_cannot_enter_new_epoch():
    from sim.ausc_policy.geometry_epoch import require_current, GEOMETRY_ID
    for old in ({}, {'geometry_id': 'legacy_1262'}, {'geometry_id': GEOMETRY_ID}):
        with pytest.raises(ValueError):
            require_current(old)
    require_current({'geometry_id': GEOMETRY_ID, 'geometry_source_sha256': 'a'*64,
                     'freshly_generated': True})


def test_contact_gate_requires_membrane_and_zero_lip():
    from sim.ausc_policy.geometry_epoch import qualify_contact
    rows = [dict(diaphragm_N=4., lip_N=0., contact_rim_N=0., carriage_other_N=0.,
        frame_N=0., forbidden_N=0., compression_m=.001, hall_N=4.) for _ in range(720)]
    assert qualify_contact(rows)['passed']
    rows[0]['lip_N'] = 1e-8
    assert not qualify_contact(rows)['passed']
    for r in rows:
        r.update(lip_N=2., diaphragm_N=0.)
    assert not qualify_contact(rows)['passed']


def test_infinite_ground_is_not_silently_dropped():
    from sim.ausc_policy.geometry_epoch import plane_lower_clearance
    boxes = np.array([[[-1,-1,.004],[1,1,.02]], [[100,100,.002],[101,101,.03]]])
    assert plane_lower_clearance(boxes,[0,0,1],[0,0,0]) == pytest.approx(.002)
