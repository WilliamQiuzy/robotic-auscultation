"""Task conditioning is measured-frame geometry, not a privileged action."""
import hashlib
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from sim.auscultation_vla_v2.hall import HallPacket, default_calibration


def adapter():
    name='sim.auscultation_pi05_v6.task_observation'
    assert importlib.util.find_spec(name) is not None, 'shared explicit task observation adapter required'
    return importlib.import_module(name)


def fixture():
    kin=OpenArmKinematics();cal=default_calibration()
    q=np.array([.2,-.4,.3,1.2,-.2,.1,.4])
    packet=HallPacket(27000,.99,.995,cal.sha256,'simulated_adc')
    decoded=cal.decode(packet,now_s=1.)
    raw=np.linspace(-.3,.4,32).astype(np.float32);raw[:7]=q
    raw[20:22]=[decoded.hall_voltage_v,decoded.spring_force_estimate_n]
    pose=kin.tcp(q,decoded.displacement_estimate_m)
    task=pose.copy();task[:3,3]+=pose[:3,:3]@np.array([.012,-.004,.023])
    task[:3,:3]=pose[:3,:3]@Rotation.from_euler('xyz',[.2,-.3,.5]).as_matrix()
    kwargs=dict(measured_q=q,hall_packet=packet,observation_time_s=1.,now_s=1.,
        body_reference_to_current_base=np.eye(4),body_time_s=1.,task_pose_in_body_reference=task,
        kinematics=kin,hall_calibration=cal)
    return raw,kwargs


def test_explicit_task_pose_appends_expected_nine_values_without_replacing_raw_geometry():
    raw,kwargs=fixture();original=raw.copy()
    result=adapter().encode_task_state(raw,**kwargs)
    expected_rotation=Rotation.from_euler('xyz',[.2,-.3,.5]).as_matrix()
    assert result.shape==(41,) and result.dtype==np.float32
    assert result[:32].tobytes()==original.tobytes()==raw.tobytes()
    np.testing.assert_allclose(result[32:35],[.012,-.004,.023],atol=1e-9,rtol=0)
    np.testing.assert_array_equal(result[35:],expected_rotation[:,:2].T.reshape(-1).astype(np.float32))


def test_global_frame_change_leaves_tool_relative_task_condition_identical():
    raw,kwargs=fixture();baseline=adapter().encode_task_state(raw,**kwargs)
    transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',[.7,.2,-.9]).as_matrix()
    transform[:3,3]=[1.2,-.3,.6]
    original=kwargs['kinematics']
    class ChangedCoordinateFrame:
        def tcp(self,q,compression):return transform@original.tcp(q,compression)
    kwargs['kinematics']=ChangedCoordinateFrame()
    kwargs['body_reference_to_current_base']=transform
    changed=adapter().encode_task_state(raw,**kwargs)
    np.testing.assert_array_equal(changed,baseline)


def test_per_point_fitted_tilt_changes_only_task_condition():
    raw,kwargs=fixture();baseline=adapter().encode_task_state(raw,**kwargs)
    kwargs['task_pose_in_body_reference']=kwargs['task_pose_in_body_reference'].copy()
    kwargs['task_pose_in_body_reference'][:3,:3] @= Rotation.from_euler('y',21.3191,degrees=True).as_matrix()
    tilted=adapter().encode_task_state(raw,**kwargs)
    np.testing.assert_array_equal(tilted[:35],baseline[:35])
    assert not np.allclose(tilted[35:],baseline[35:])


def test_actual_compression_and_supplied_calibration_are_not_overwritten():
    raw,kwargs=fixture();before=adapter().encode_task_state(raw,**kwargs)
    cal=kwargs['hall_calibration'];old=cal.decode(kwargs['hall_packet'],now_s=1.)
    packet=HallPacket(29000,.99,.995,cal.sha256,'simulated_adc')
    new=cal.decode(packet,now_s=1.);kwargs['hall_packet']=packet
    raw[20:22]=[new.hall_voltage_v,new.spring_force_estimate_n]
    after=adapter().encode_task_state(raw,**kwargs)
    # Task fixed in base: more retracted TCP makes relative task Z larger.
    assert after[34]-before[34]==pytest.approx(new.displacement_estimate_m-old.displacement_estimate_m,abs=2e-9)
    np.testing.assert_array_equal(after[:32],raw)
    from dataclasses import replace
    kwargs['hall_packet']=replace(packet,calibration_sha256='0'*64)
    with pytest.raises(ValueError,match='calibration'):adapter().encode_task_state(raw,**kwargs)


@pytest.mark.parametrize('field,value,match',[
    ('now_s',1.16,'stale'),('now_s',.999,'future'),
    ('body_time_s',.95,'same acquired'),('body_time_s',float('nan'),'clock'),
    ('observation_time_s',-1.,'clock'),
])
def test_rejects_stale_future_or_unpaired_geometry(field,value,match):
    raw,kwargs=fixture();kwargs[field]=value
    with pytest.raises(ValueError,match=match):adapter().encode_task_state(raw,**kwargs)


@pytest.mark.parametrize('name', ['body_reference_to_current_base','task_pose_in_body_reference'])
@pytest.mark.parametrize('corruption', ['reflection','bottom_row','nonfinite','scaling'])
def test_rejects_non_se3_inputs(name,corruption):
    raw,kwargs=fixture();bad=kwargs[name].copy()
    if corruption=='reflection':bad[:3,0]*=-1
    elif corruption=='bottom_row':bad[3,0]=.1
    elif corruption=='nonfinite':bad[0,3]=np.nan
    else:bad[:3,0]*=1.01
    kwargs[name]=bad
    with pytest.raises(ValueError,match='transform'):adapter().encode_task_state(raw,**kwargs)


@pytest.mark.parametrize('index', [0,20,21])
def test_rejects_raw_measurement_or_hall_feature_disagreement(index):
    raw,kwargs=fixture();raw[index]+=.001
    with pytest.raises(ValueError,match='raw'):adapter().encode_task_state(raw,**kwargs)


def test_stale_adc_is_rejected_even_with_fresh_camera():
    from dataclasses import replace
    raw,kwargs=fixture();kwargs['hall_packet']=replace(kwargs['hall_packet'],sample_time_s=.94,received_time_s=.94)
    with pytest.raises(ValueError,match='stale'):adapter().encode_task_state(raw,**kwargs)


def test_contract_binds_exact_implementation_and_width():
    module=adapter()
    assert module.STATE_WIDTH==module.STATE_DIM==41 and module.RAW_WIDTH==32
    assert module.observation_binding()==dict(version=module.VERSION,
        sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),state_width=41,raw_width=32)
