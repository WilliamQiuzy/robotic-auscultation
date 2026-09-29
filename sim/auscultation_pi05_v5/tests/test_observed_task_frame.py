from types import SimpleNamespace as S
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sim.auscultation_pi05_v5.learned_controller import LearnedController


def controller(tmp_path):
    actor=LearnedController.__new__(LearnedController)
    actor.env=S(out=tmp_path,base=np.eye(4));actor.task_frames={}
    actor.target=S(name='L5',position_base=np.array([.001,0,0]))
    actor.rotation_base=np.diag([1.,-1.,-1.])
    frame=dict(index=0,time_s=0.,target='L5',body_reference_to_current_base=np.eye(4).tolist(),target_estimate=dict(
        rotation_base=np.eye(3).tolist(),position_base_m=[0.,0.,0.]))
    hall=S(displacement_estimate_m=0.,spring_force_estimate_n=0.)
    actor.sensor=S(rows=[frame],cal=S(decode=lambda *a,**kw:hall),hall=S(read=lambda t:None))
    pose=np.eye(4);pose[:3,:3]=actor.rotation_base;pose[:3,3]=[.001,0,.03]
    actor.kin=S(tcp=lambda *a:pose.copy())
    return actor,pose


def test_all_initial_tasks_share_first_body_registration_before_second_target_use(tmp_path):
    actor,pose=controller(tmp_path)
    second=S(name='R4',position_base=np.array([.10,.04,0.]))
    rotation=actor.rotation_base.copy()
    actor._prepare_task_frames([(actor.target,dict(command_rotation_world=rotation)),
                               (second,dict(command_rotation_world=rotation))])
    # The body-reference transform need not be identity at the first camera frame.
    first=np.eye(4);first[:3,3]=[.02,-.01,0.]
    actor.sensor.rows[0]['body_reference_to_current_base']=first.tolist()
    actor._tracked_metrics(dict(time_s=0.,q=np.zeros(7)))
    assert set(actor.task_frames)=={'L5','R4'}
    assert {item['source_frame'] for item in actor.task_frames.values()}=={0}
    assert actor.task_frames['R4']['initial_body_reference_to_current_base']==first.tolist()
    body=np.eye(4);body[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    body[:3,3]=[.03,.02,0.]
    initial_task=np.eye(4);initial_task[:3,:3]=rotation;initial_task[:3,3]=second.position_base
    expected=body@np.linalg.inv(first)@initial_task
    # Mutating future task objects cannot replace the copied initial task pose.
    second.position_base[:]=[8,8,8]
    actor.target=second;actor.rotation_base=np.eye(3)
    actor.sensor.rows.append(dict(actor.sensor.rows[0],index=1,time_s=.05,target='R4',
                                 body_reference_to_current_base=body.tolist()))
    pose[:]=expected;pose[:3,3]-=expected[:3,2]*.03
    metric,_,_=actor._tracked_metrics(dict(time_s=.05,q=np.zeros(7)))
    assert metric['gap_m']==pytest.approx(.03)
    assert metric['tangent_m']<1e-10
    assert metric['angle_deg']<1e-5
    assert actor.task_frames['R4']['source_frame']==0


def test_initial_task_registration_follows_rgbd_without_truth_refresh(tmp_path):
    actor,pose=controller(tmp_path);state={'time_s':0.,'q':np.zeros(7)}
    actor._prepare_task_frames([(actor.target,dict(command_rotation_world=actor.rotation_base.copy()))])
    metric,_,_=actor._tracked_metrics(state)
    assert metric['gap_m']==pytest.approx(.03)
    assert metric['tangent_m']==pytest.approx(0)
    rotation=Rotation.from_euler('z',90,degrees=True).as_matrix()
    frame=actor.sensor.rows[-1];frame['time_s']=.05
    frame['target_estimate'].update(rotation_base=rotation.tolist(),position_base_m=[0,.02,0])
    body=np.eye(4);body[:3,:3]=rotation;body[:3,3]=[0,.02,0]
    frame['body_reference_to_current_base']=body.tolist()
    pose[:3,3]=[0,.021,.03];pose[:3,:3]=rotation@np.diag([1.,-1.,-1.])
    actor.target.position_base[:]=[8,8,8] # simulator/task object mutations cannot refresh registration
    actor.rotation_base[:]=np.eye(3)
    metric,_,_=actor._tracked_metrics(dict(state,time_s=.05))
    assert metric['gap_m']==pytest.approx(.03)
    assert metric['tangent_m']<1e-10
    assert metric['angle_deg']<1e-5


def test_wrong_target_or_old_geometry_is_rejected(tmp_path):
    actor,_=controller(tmp_path)
    with pytest.raises(RuntimeError,match='stale'):actor._tracked_metrics(dict(time_s=.2,q=np.zeros(7)))
    actor.sensor.rows[-1]['target']='R1'
    with pytest.raises(RuntimeError,match='Target'):actor._tracked_metrics(dict(time_s=0.,q=np.zeros(7)))
