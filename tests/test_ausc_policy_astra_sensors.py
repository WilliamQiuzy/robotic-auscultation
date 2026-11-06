"""Sensor-domain CPU checks; fake render data do not prove native sensing."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_ausc_policy_m2_capture import FakeEnv, FakeBackend, CAMERA, CALIBRATION


def make(tmp_path,backend=None,**kwargs):
    from sim.ausc_policy.astra_sensors import AstraSensors
    env=FakeEnv();env.rows=[];env.base=np.eye(4)
    backend=backend or FakeBackend(env)
    backend.env=env
    sensor=AstraSensors(env,tmp_path/'sensor',camera_bytes=CAMERA,calibration_bytes=CALIBRATION,
        backend_factory=lambda _:backend,**kwargs)
    return sensor,env,backend


def test_rgbd_and_adc_bind_same_native_state_without_advancing_physics(tmp_path):
    sensor,env,backend=make(tmp_path)
    before=deepcopy(env.state());clock=env.world.current_time
    bundle,rgb=sensor.capture(env.state(),0,np.eye(4))
    assert bundle['schema']=='ausc_astra_rgbd_hall_adc_v1' and bundle['single_camera_path']=='/World/RobotCam'
    assert bundle['hall']['packet']['source']=='simulated_adc'
    assert bundle['hall']['decoded']['calibration_sha256']==hashlib.sha256(CALIBRATION).hexdigest()
    assert bundle['hall']['decoded']['hall_voltage_v']>0
    assert bundle['hall']['bench_verified'] is False and bundle['hall']['force_label_used'] is False
    assert bundle['depth']['units']=='m' and bundle['depth']['semantics']=='metric_optical_z'
    assert env.state()['time_s']==before['time_s'] and env.world.current_time==clock and len(env.rows)==0
    assert rgb.startswith(b'\x89PNG')
    raw=np.load(sensor.output/bundle['raw_rgbd']['file'],allow_pickle=False)
    np.testing.assert_array_equal(raw['depth_m'],backend.depth)
    np.testing.assert_array_equal(raw['depth_valid'],np.isfinite(backend.depth)&(backend.depth>0))
    assert 'depth_m' not in bundle and 'depth_valid' not in bundle
    assert sensor.close()['completed']


def test_invalid_depth_is_masked_preserved_and_json_never_contains_nan(tmp_path):
    sensor,env,backend=make(tmp_path)
    backend.depth[0,:4]=[np.nan,np.inf,0.,-1.]
    bundle,_=sensor.capture(env.state(),0,np.eye(4))
    assert bundle['depth']['valid_count']==360*640-4
    assert bundle['depth']['invalid_count']==4
    assert 'NaN' not in json.dumps(bundle,allow_nan=False)
    raw=np.load(sensor.output/bundle['raw_rgbd']['file'],allow_pickle=False)
    assert np.isnan(raw['depth_m'][0,0]) and not raw['depth_valid'][0,:4].any()
    sensor.close()


def test_all_invalid_depth_is_rejected_with_no_physics_ticks(tmp_path):
    sensor,env,backend=make(tmp_path);backend.depth[:]=np.nan
    with pytest.raises(ValueError,match='valid depth'):sensor.capture(env.state(),0,np.eye(4))
    assert not env.rows and env.state()['time_s']==0
    sensor.close()


@pytest.mark.parametrize('kind',['units','optics','clock'])
def test_camera_units_intrinsics_and_render_clock_remain_checked(tmp_path,kind):
    sensor,env,backend=make(tmp_path)
    if kind=='units':backend.camera['meters_per_unit']=.001
    elif kind=='optics':backend.camera['intrinsics']['fx']*=2
    else:backend.on_render=lambda:setattr(env.world,'current_time',9.)
    with pytest.raises(ValueError):sensor.capture(env.state(),0,np.eye(4))
    assert not sensor.close()['completed']


def test_calibration_drift_nan_and_out_of_domain_cannot_manufacture_adc(tmp_path):
    from sim.ausc_policy.astra_sensors import AstraSensors
    env=FakeEnv();env.rows=[];env.base=np.eye(4)
    with pytest.raises(ValueError,match='calibration'):
        AstraSensors(env,tmp_path/'drift',camera_bytes=CAMERA,calibration_bytes=CALIBRATION+b' ',backend_factory=FakeBackend)
    sensor,env,_=make(tmp_path/'valid')
    for compression in (np.nan,-.0001,.008):
        bad=env.state();bad['compression_m']=compression
        with pytest.raises(ValueError):sensor.observe_hall(bad,0)
    sensor.close()


def test_hall_observation_uses_calibration_decode_not_native_force_label(tmp_path):
    sensor,env,_=make(tmp_path)
    for i in range(3):
        state=env.step_to(i);state['compression_m']=.0025;state['hall_N']=999.
        env.value.update(state);env.rows.append({})
        evidence=sensor.observe_hall(state,i+1)
    assert evidence['decoded']['spring_force_estimate_n']==pytest.approx(4.,abs=.01)
    assert evidence['packet']['source']=='simulated_adc'
    assert evidence['decoded']['hall_voltage_v']==pytest.approx(1.3261,abs=.002)
    assert evidence['packet']['sample_time_s']==state['time_s']
    sensor.close()


def test_depth_probe_is_explicitly_gt_projected_not_a_visual_target_estimate(tmp_path):
    sensor,env,backend=make(tmp_path)
    camera=sensor.camera.actual_camera_at_setup
    world_from_optical=np.asarray(camera['world_from_camera_usd'])@np.diag([1.,-1.,-1.,1.])
    reference=np.eye(4);reference[:3,3]=(world_from_optical@np.array([0.,0.,.5,1.]))[:3]
    bundle,_=sensor.capture(env.state(),0,reference)
    probe=bundle['depth']['prescribed_projection_probe']
    assert probe['source']=='native_depth_at_prescribed_pose_projection'
    assert probe['anatomical_target_estimator'] is False and probe['valid'] is True
    assert probe['optical_z_m']==pytest.approx(float(backend.depth[180,320]))
    assert probe['point_base_m']!=reference[:3,3].tolist()
    sensor.close()


def test_model_sensor_fields_are_bound_to_units_adc_calibration_and_native_clock(tmp_path):
    from sim.ausc_policy.astra_supervisor import validate_observation
    from test_ausc_policy_astra_supervisor import observation
    sensor,env,_=make(tmp_path);bundle,rgb=sensor.capture(env.state(),0,np.eye(4))
    obs=observation();obs.update(sensor_contract=bundle['schema'],sensor_bundle=bundle,
        hall_voltage_v=bundle['hall']['decoded']['hall_voltage_v'],hall_voltage_source='simulated_adc',
        hall_force_N=bundle['hall']['decoded']['spring_force_estimate_n'],
        hall_source='calibration_decoded_simulated_adc_spring_load',
        rgb_sha256=hashlib.sha256(rgb).hexdigest())
    assert validate_observation(obs) is obs
    for field,value in [('units','mm'),('semantics','euclidean_range')]:
        bad=deepcopy(obs);bad['sensor_bundle']['depth'][field]=value
        with pytest.raises(ValueError):validate_observation(bad)
    for mutate in [lambda b:b['sensor_bundle']['hall']['packet'].update(source='real_adc'),
                   lambda b:b['sensor_bundle']['hall']['decoded'].update(hall_voltage_v=np.nan),
                   lambda b:b['sensor_bundle'].update(observed_at_s=.001),
                   lambda b:b.update(hall_force_N=999.),
                   lambda b:b['sensor_bundle']['hall']['packet'].update(calibration_sha256='0'*64)]:
        bad=deepcopy(obs);mutate(bad)
        with pytest.raises(ValueError):validate_observation(bad)
    sensor.close()


def test_passive_hall_sampling_requires_consecutive_same_native_displacement(tmp_path):
    sensor,env,_=make(tmp_path)
    bad=env.state();bad['compression_m']+=.001
    with pytest.raises(ValueError):sensor.observe_hall(bad,0)
    state=env.step_to(1);env.rows.extend([{},{}])
    with pytest.raises(ValueError,match='consecutive'):sensor.observe_hall(state,2)
    sensor.close()


def test_native_observation_passes_rgbd_and_voltage_decode_not_controller_force(tmp_path):
    from types import SimpleNamespace
    from sim.ausc_policy.astra_native import native_observation
    sensor,env,_=make(tmp_path);env.value['hall_N']=999.
    controller=SimpleNamespace(env=env,astra_sensors=sensor,episode_id='episode',
        manager=SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='APPROACH'))),
        target=SimpleNamespace(name='L1'),reference=SimpleNamespace(sha256='a'*64,pose_base=np.eye(4)))
    obs,rgb=native_observation(controller,'HOME')
    assert obs['hall_force_N']!=999. and obs['hall_voltage_v']>0
    assert obs['sensor_contract']=='ausc_astra_rgbd_hall_adc_v1'
    assert obs['sensor_bundle']['depth']['valid_count']==360*640 and rgb.startswith(b'\x89PNG')
    sensor.close()


def test_controller_tick_observes_hall_without_replacing_native_command(tmp_path):
    from types import SimpleNamespace
    from test_ausc_policy_astra_supervisor import bridge
    c=bridge(tmp_path);seen=[]
    c.astra_sensors=SimpleNamespace(observe_hall=lambda state,row:seen.append((state,row)))
    c._tick(np.zeros(7),'contact')
    assert len(seen)==1 and seen[0][1]==1
    assert c.env.rows[0]['command_source_id']=='given_contact_original_analytic_l3_v1'


def test_sensor_cleanup_failure_is_retained_and_world_still_stops(tmp_path):
    from types import SimpleNamespace
    from sim.ausc_policy.fresh_expert import finalize_probe
    events=[]
    def close():events.append('sensor');raise ValueError('depth detach failed')
    controller=SimpleNamespace(control_rows=[],close_astra_sensors=close)
    env=SimpleNamespace(rows=[],save=lambda _:events.append('trace'),
                        world=SimpleNamespace(stop=lambda:events.append('stop')))
    _,errors=finalize_probe(env,controller,None,tmp_path)
    assert events.index('sensor')<events.index('stop')
    assert any(x['stage']=='astra_sensor_finalize' for x in errors)


def test_hall_sampling_failure_cannot_leave_a_successful_sensor_teardown(tmp_path):
    sensor,env,_=make(tmp_path)
    env.value['compression_m']=np.nan
    with pytest.raises(ValueError):sensor.observe_hall(env.state(),0)
    receipt=sensor.close()
    assert receipt['completed'] is False and receipt['failures']
