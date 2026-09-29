"""Teacher action labels must be causal measured-TCP-relative commands."""
import importlib
import numpy as np
import pytest
from sim.auscultation_vla_v2.hall import HallSimulator,default_calibration
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from dataclasses import asdict


def module():
    try:return importlib.import_module('sim.auscultation_vla_v2.demo_labels')
    except ModuleNotFoundError:pytest.fail('causal action label adapter missing')


def pair():
    sensor=HallSimulator(default_calibration());sensor.update(1.,.001)
    a=dict(time_s=1.,phase='CONTACT',target='R1',q=np.zeros(7).tolist(),hall=asdict(sensor.read(1.)),eligible=True)
    b=dict(a,time_s=1.05)
    return a,b


def test_action_is_next_command_relative_to_measured_pose():
    m=module();a,b=pair();kin=OpenArmKinematics();q=np.zeros(7);q[2]=.0001
    action=m.label_pair(a,b,q)
    from sim.auscultation_vla_v2.hall import HallPacket
    compression=default_calibration().decode(HallPacket(**a['hall']),now_s=1.).displacement_estimate_m
    expected=kin.tcp(q,compression)[:3,3]-kin.tcp(np.array(a['q']),compression)[:3,3]
    np.testing.assert_allclose(action[:3],expected,atol=1e-12)
    assert action.shape==(7,) and action[6]==4.
    # The future measured state must never replace the approved command label.
    b['q']=(np.ones(7)*.1).tolist()
    np.testing.assert_array_equal(m.label_pair(a,b,q),action)


@pytest.mark.parametrize('field,value',[('phase','RETRACT'),('target','L1'),('time_s',1.10),('eligible',False)])
def test_action_pair_rejects_temporal_or_phase_boundaries(field,value):
    m=module();a,b=pair();b[field]=value
    with pytest.raises(ValueError):m.label_pair(a,b,np.zeros(7))


def test_retraction_force_reference_is_zero_and_stale_voltage_rejected():
    m=module();a,b=pair();a['phase']=b['phase']='RETRACT'
    assert m.label_pair(a,b,np.zeros(7))[6]==0.
    a['hall']['sample_time_s']=.8
    with pytest.raises(ValueError):m.label_pair(a,b,np.zeros(7))
