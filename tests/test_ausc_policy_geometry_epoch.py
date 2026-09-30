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
    from sim.ausc_policy.geometry_epoch import require_current, GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
    for old in ({}, {'geometry_id': 'legacy_1262'}, {'geometry_id': GEOMETRY_ID}):
        with pytest.raises(ValueError):
            require_current(old)
    with pytest.raises(ValueError):
        require_current({'geometry_id':GEOMETRY_ID,'geometry_source_sha256':'a'*64,'freshly_generated':True})
    require_current({'geometry_id': GEOMETRY_ID, 'geometry_source_sha256': GEOMETRY_SOURCE_SHA256,
                     'freshly_generated': True})


def test_flat_membrane_5mm_predecessor_cannot_enter_recessed_retainer_epoch():
    from sim.ausc_policy.geometry_epoch import require_current
    with pytest.raises(ValueError,match='Retired'):
        require_current(dict(geometry_id='stethoscope_protruding_5mm_v1',freshly_generated=True,
                             geometry_source_sha256='4637a4e7218da2b91ef2688f45f879102a243082d11d709c61a3a953a46d8081'))


def test_export_bytes_must_match_current_receipt(tmp_path):
    import hashlib
    from sim.ausc_policy import geometry_epoch as g
    meta=dict(geometry_id=g.GEOMETRY_ID,geometry_source_sha256=g.GEOMETRY_SOURCE_SHA256,freshly_generated=True)
    for name,key,data in [('collision_geometry.json','collision_geometry_sha256',b'fresh metadata'),
                          ('collision_geometry.npz','collision_arrays_sha256',b'fresh arrays')]:
        (tmp_path/name).write_bytes(data);meta[key]=hashlib.sha256(data).hexdigest()
    result=g.require_geometry_artifacts(tmp_path,meta)
    assert result['collision_arrays_sha256']==meta['collision_arrays_sha256']
    (tmp_path/'collision_geometry.npz').write_bytes(b'retired')
    with pytest.raises(ValueError,match='artifact'):
        g.require_geometry_artifacts(tmp_path,meta)
    del meta['collision_arrays_sha256']
    with pytest.raises(ValueError,match='artifact'):
        g.require_geometry_artifacts(tmp_path,meta)


def test_contact_gate_requires_membrane_and_zero_lip():
    from sim.ausc_policy.geometry_epoch import qualify_contact
    rows = [dict(diaphragm_N=4., retainer_N=0., lip_N=0., contact_rim_N=0., carriage_other_N=0.,
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


def test_actual_state_uses_new_tcp_not_legacy_offset():
    from types import SimpleNamespace
    from sim.auscultation_expert_layout_v1.env import IsaacContactEnv
    env=IsaacContactEnv.__new__(IsaacContactEnv)
    env.robot=SimpleNamespace(get_joint_positions=lambda:np.r_[np.zeros(7),.001],
                              get_joint_velocities=lambda:np.zeros(8))
    env.wrist=SimpleNamespace(get_world_pose=lambda:(np.zeros(3),np.array([1.,0,0,0])))
    env.slider=7;env.left=np.arange(7);env.base=np.eye(4);env.t=0.;env.filtered_hall_force=0.
    env.contact_offset_m=.132;env.moving_mass_kg=.067
    np.testing.assert_allclose(env.state()['tcp_world'],[0,0,.131])


def test_axis_ik_can_use_roll_freedom_inside_safety_bounds():
    from sim.ausc_policy.axis_ik import solve_axis
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    kin=OpenArmKinematics(contact_frame=frame)
    q=(kin.lower_limits+kin.upper_limits)/2
    pose=kin.tcp(q)
    limits=PostureLimits(10.,tuple(kin.lower_limits),tuple(kin.upper_limits))
    result=solve_axis(kin,np.eye(4),limits,pose[:3,3],pose[:3,2],q+.03,contact=False)
    assert result['success']
    np.testing.assert_allclose(kin.tcp(result['q'])[:3,3],pose[:3,3],atol=2e-5)


def test_axis_cone_never_exceeds_original_five_degree_contact_tolerance():
    from sim.ausc_policy.axis_ik import solve_axis
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    kin=OpenArmKinematics(contact_frame=frame);q=(kin.lower_limits+kin.upper_limits)/2
    pose=kin.tcp(q);a=np.deg2rad(4.)
    normal_axis=np.cos(a)*pose[:3,2]+np.sin(a)*pose[:3,0]
    limits=PostureLimits(10.,tuple(kin.lower_limits),tuple(kin.upper_limits))
    result=solve_axis(kin,np.eye(4),limits,pose[:3,3],normal_axis,q,contact=False,axis_tolerance_deg=5.)
    assert result['success'] and result['orientation_error_rad']<=np.deg2rad(5.)+1e-8
    with pytest.raises(ValueError):
        solve_axis(kin,np.eye(4),limits,pose[:3,3],normal_axis,q,axis_tolerance_deg=5.1)


def test_full_pose_ik_margin_also_covers_zero_joint_limit():
    from sim.auscultation_expert_layout_v1.pose_ik import solve_posture_ik
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    kin=OpenArmKinematics(contact_frame=frame)
    lower=kin.lower_limits.copy();lower[3]=0.
    limits=PostureLimits(10.,tuple(lower),tuple(kin.upper_limits))
    q=np.zeros(7);pose=kin.tcp(q)
    solved,_,_,_=solve_posture_ik(kin,np.eye(4),limits,pose[:3,3],pose[:3,:3],q)
    assert solved[3]>=np.deg2rad(.2)-1e-12
