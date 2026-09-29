import numpy as np
import pytest
from sim.auscultation_pi05_v5.execution import JointTrajectory
from sim.auscultation_pi05_v5.action_contract import compose_joints


def test_measured_origin_preserves_all_seven_joint_channels():
    measured=np.arange(7)/10
    raw=np.linspace(-.005,.005,7)
    got=compose_joints(measured,raw,'HOLD',np.full(7,-3.),np.full(7,3.))
    np.testing.assert_array_equal(got,measured+raw)
    assert got[6]==measured[6]+raw[6]


def test_rejects_proposal_or_physical_joint_limit_without_clipping():
    with pytest.raises(ValueError):
        compose_joints(np.zeros(7),np.full(7,.008),'HOLD',-np.ones(7),np.ones(7))
    with pytest.raises(ValueError):
        compose_joints(np.ones(7),np.full(7,.001),'FREE',-np.ones(7),np.ones(7))


def test_continuous_retargeting_respects_joint_derivatives():
    executor=JointTrajectory(np.zeros(7),dt=1/240,velocity=.25,acceleration=.5,jerk=8.)
    q=[];v=[];a=[]
    for i in range(720):
        if i%12==0:executor.set_target(np.full(7,.15 if i<360 else -.1))
        position=executor.step();q.append(position);v.append(executor.velocity.copy());a.append(executor.acceleration.copy())
    assert np.max(np.abs(v))<=.25+1e-8
    assert np.max(np.abs(a))<=.5+1e-8
    assert np.max(np.abs(np.diff(a,axis=0)))*240<=8+1e-6
    assert np.max(np.abs(np.diff(q,axis=0)))*240<=.25+1e-8
