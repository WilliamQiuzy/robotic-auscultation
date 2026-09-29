from types import SimpleNamespace as S
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sim.auscultation_pi05_v4.learned_controller import LearnedController


def controller(tmp_path):
    actor=LearnedController.__new__(LearnedController)
    actor.env=S(out=tmp_path);actor.task_frames={}
    actor.target=S(name='L5',position_base=np.array([.001,0,0]))
    actor.rotation_base=np.diag([1.,-1.,-1.])
    frame=dict(index=0,time_s=0.,target='L5',body_reference_to_current_base=np.eye(4).tolist(),target_estimate=dict(
        rotation_base=np.eye(3).tolist(),position_base_m=[0.,0.,0.]))
    hall=S(displacement_estimate_m=0.,spring_force_estimate_n=0.)
    actor.sensor=S(rows=[frame],cal=S(decode=lambda *a,**kw:hall),hall=S(read=lambda t:None))
    pose=np.eye(4);pose[:3,:3]=actor.rotation_base;pose[:3,3]=[.001,0,.03]
    actor.kin=S(tcp=lambda *a:pose.copy())
    return actor,pose


def test_initial_task_registration_follows_rgbd_without_truth_refresh(tmp_path):
    actor,pose=controller(tmp_path);state={'time_s':0.,'q':np.zeros(7)}
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
