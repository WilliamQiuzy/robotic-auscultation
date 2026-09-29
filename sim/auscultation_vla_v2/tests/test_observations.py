import importlib
import numpy as np
import pytest
from dataclasses import replace
from sim.auscultation_vla_v2.hall import HallPacket, default_calibration


@pytest.fixture
def api():
    try:
        module = importlib.import_module('sim.auscultation_vla_v2.observations')
    except ModuleNotFoundError:
        module = None
    assert module is not None, 'deployment observation contract is not implemented'
    return module


def sample(api):
    return api.Observation(
        rgb=np.full((24, 32, 3), 120, np.uint8),
        depth_m=np.full((24, 32), .8, np.float32),
        depth_valid=np.ones((24, 32), bool),
        intrinsics=np.array([[100., 0., 16.], [0., 100., 12.], [0., 0., 1.]]),
        camera_to_base=np.eye(4), rgb_time_s=10., depth_time_s=10.,
        robot=api.RobotState(q=np.zeros(7), dq=np.zeros(7),
            wrist_position_base_m=np.array([0.,0.,-(.1262-default_calibration().decode(hall_packet(),now_s=10.).displacement_estimate_m)]),
            wrist_rotation_base=np.eye(3), hall=hall_packet(), time_s=10.),
        target=api.TargetEstimate(position_base_m=np.array([.1, .2, .8]), rotation_base=np.eye(3),
            velocity_base_m_s=np.zeros(3), confidence=.95, time_s=10., revision=3,
            source='rgbd_surface_tracking'),
        instruction='Contact the currently specified back target.')


def test_one_real_camera_and_two_explicitly_missing_views(api):
    obs=sample(api); encoded=api.encode(obs, now_s=10.05)
    assert encoded['state'].shape == (32,)
    assert encoded['image_masks'] == {'base_0_rgb': True, 'left_wrist_0_rgb': False, 'right_wrist_0_rgb': False}
    assert np.array_equal(encoded['images']['base_0_rgb'], obs.rgb)
    assert not encoded['images']['left_wrist_0_rgb'].any()
    assert encoded['prompt'] == obs.instruction
    assert not {'simulator_pose', 'future_trajectory', 'target_id', 'phase'} & set(encoded)
    assert len(api.STATE_FEATURES) == len(set(api.STATE_FEATURES)) == 32


def test_shifted_target_changes_relative_geometry_without_changing_robot(api):
    obs=sample(api); before=api.encode(obs, now_s=10.05)['state']
    obs.target.position_base_m[0] += .03
    after=api.encode(obs, now_s=10.05)['state']
    indices=[api.STATE_FEATURES.index('target_relative_x_m'), api.STATE_FEATURES.index('target_relative_y_m'), api.STATE_FEATURES.index('target_relative_z_m')]
    assert np.allclose((after-before)[indices], [.03,0,0])
    assert np.array_equal(before[:14], after[:14])


def test_relative_geometry_uses_tool_frame(api):
    obs=sample(api)
    obs.robot.wrist_rotation_base=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    encoded=api.encode(obs, now_s=10.05)['state']
    start=api.STATE_FEATURES.index('target_relative_x_m')
    assert np.allclose(encoded[start:start+3], [.2,-.1,.8])


@pytest.mark.parametrize('change', [
    lambda o: setattr(o.target,'source','isaac_ground_truth'),
    lambda o: setattr(o.target,'confidence',.1),
    lambda o: setattr(o.target,'time_s',9.),
    lambda o: setattr(o.robot,'time_s',9.),
    lambda o: setattr(o,'rgb_time_s',9.),
    lambda o: setattr(o,'depth_time_s',9.9),
    lambda o: setattr(o.robot,'time_s',11.),
    lambda o: setattr(o.robot,'hall',replace(o.robot.hall,raw=-1)),
    lambda o: setattr(o.robot,'hall',replace(o.robot.hall,calibration_sha256='wrong')),
    lambda o: setattr(o.target,'revision',-1),
    lambda o: setattr(o.target,'rotation_base',np.zeros((3,3))),
    lambda o: setattr(o,'camera_to_base',np.ones((4,4))),
    lambda o: setattr(o,'depth_valid',np.zeros((24,32),bool)),
    lambda o: o.depth_m.__setitem__((0,0),float('nan')),
    lambda o: setattr(o,'rgb',o.rgb.astype(np.float32)),
    lambda o: setattr(o,'instruction',''),
])
def test_invalid_or_privileged_observation_is_rejected(api, change):
    obs=sample(api); change(obs)
    with pytest.raises(ValueError): api.encode(obs, now_s=10.05)


def test_invalid_depth_pixels_can_be_masked_but_do_not_become_valid(api):
    obs=sample(api); obs.depth_m[0,0]=np.nan; obs.depth_valid[0,0]=False
    encoded=api.encode(obs, now_s=10.05)
    assert np.isfinite(encoded['state']).all()
    assert np.isnan(obs.depth_m[0,0]) and not obs.depth_valid[0,0]


def test_encoding_has_no_alias_that_can_mutate_camera_capture(api):
    obs=sample(api); encoded=api.encode(obs, now_s=10.05)
    encoded['images']['base_0_rgb'][:]=0
    assert np.all(obs.rgb==120)


def hall_packet(time_s=10.,source='simulated_adc'):
    return HallPacket(int(round(1.3261/3.3*65535)),time_s,time_s,default_calibration().sha256,source)


def test_policy_force_features_are_decoded_from_hall_voltage(api):
    obs=sample(api);obs.robot.hall=hall_packet()
    state=api.encode(obs,now_s=10.05)['state']
    assert 'hall_voltage_V' in api.STATE_FEATURES
    assert 'hall_N' not in api.STATE_FEATURES and 'compression_m' not in api.STATE_FEATURES
    assert state[api.STATE_FEATURES.index('hall_voltage_V')]==pytest.approx(1.3261,abs=.0001)
    assert state[api.STATE_FEATURES.index('hall_spring_estimate_N')]==pytest.approx(4.,abs=.001)


def test_stale_hall_is_rejected_even_with_fresh_robot_and_camera(api):
    obs=sample(api);obs.robot.hall=hall_packet(9.9)
    with pytest.raises(ValueError,match='stale'):api.encode(obs,now_s=10.05)


def test_scalar_force_without_adc_packet_cannot_enter_policy(api):
    obs=sample(api);obs.robot.hall=None
    with pytest.raises(ValueError,match='ADC'):api.encode(obs,now_s=10.05)


def test_hard_limit_from_actual_voltage_conversion_rejects_proposal_input(api):
    obs=sample(api);obs.robot.hall=replace(hall_packet(),raw=int(round(.9/3.3*65535)))
    with pytest.raises(ValueError,match='limit'):api.encode(obs,now_s=10.05)


def test_real_and_sim_adc_use_identical_model_features(api):
    obs=sample(api);obs.robot.hall=hall_packet()
    a=api.encode(obs,now_s=10.05)['state']
    obs.robot.hall=replace(obs.robot.hall,source='real_adc')
    assert np.array_equal(a,api.encode(obs,now_s=10.05)['state'])


def test_hall_derived_compression_updates_tip_geometry_without_joint_motion(api):
    obs=sample(api);before=api.encode(obs,now_s=10.05)['state']
    voltage=default_calibration().voltage_at_displacement(.0015)
    obs.robot.hall=replace(obs.robot.hall,raw=int(round(voltage/3.3*65535)))
    after=api.encode(obs,now_s=10.05)['state']
    index=api.STATE_FEATURES.index('target_relative_z_m')
    assert after[index]-before[index]==pytest.approx(-.001,abs=.00001)
    assert np.array_equal(after[:14],before[:14])


def test_extra_simulator_force_and_compression_cannot_override_adc(api):
    obs=sample(api);before=api.encode(obs,now_s=10.05)['state']
    obs.robot.hall_N=99.;obs.robot.compression_m=.1
    assert np.array_equal(before,api.encode(obs,now_s=10.05)['state'])
