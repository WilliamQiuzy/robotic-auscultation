import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sim.auscultation_pi05_v4.execution import compose_pose, JointTrajectory


def test_pose_offsets_are_base_frame_and_do_not_integrate_velocity():
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    pose[:3,3]=[1,2,3]
    action=np.array([.0003,.0004,.0005,.001,0,0,4.])
    goal=compose_pose(pose,action,'HOLD')
    np.testing.assert_allclose(goal[:3,3],[1.0003,2.0004,3.0005])
    np.testing.assert_allclose(goal[:3,:3],Rotation.from_rotvec(action[3:6]).as_matrix()@pose[:3,:3])


def test_rejects_out_of_bounds_pose_without_clipping():
    with pytest.raises(ValueError):compose_pose(np.eye(4),np.array([.003,0,0,0,0,0,4]),'HOLD')


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
