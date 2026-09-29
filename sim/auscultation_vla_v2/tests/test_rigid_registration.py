"""Geometric candidate tests; synthetic inputs do not constitute task data."""
import importlib
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def api():
    try:return importlib.import_module('sim.auscultation_vla_v2.rigid_registration')
    except ModuleNotFoundError:pytest.fail('rigid RGB-D registration missing')


def surface():
    y,x=np.mgrid[-.18:.18:40j,-.2:.2:45j]
    z=.8+1.2*x*x+.8*y*y+.018*np.sin(22*x)*np.cos(17*y)
    dx=2.4*x+.018*22*np.cos(22*x)*np.cos(17*y)
    dy=1.6*y-.018*17*np.sin(22*x)*np.sin(17*y)
    n=np.stack([-dx,-dy,np.ones_like(x)],-1).reshape(-1,3)
    n/=np.linalg.norm(n,axis=1,keepdims=True)
    return np.stack([x,y,z],-1).reshape(-1,3),n


def test_rigid_motion_recovered_and_inputs_preserved():
    m=api();p,n=surface();original=p.copy();R=Rotation.from_rotvec([.015,-.01,.012]).as_matrix();t=np.array([.006,-.004,.003])
    q=p@R.T+t;result=m.register(p,n,q,n@R.T)
    np.testing.assert_allclose(result['transform'][:3,:3],R,atol=.001)
    np.testing.assert_allclose(result['transform'][:3,3],t,atol=.001)
    np.testing.assert_array_equal(original,p)


def test_planar_geometry_cannot_claim_six_axis_observability():
    m=api();p,n=surface();p[:,2]=.8;n[:]=[0,0,1]
    with pytest.raises(ValueError,match='observable'):m.register(p,n,p,n)


def test_partial_front_occluder_does_not_drag_pose():
    m=api();p,n=surface();q=p.copy();q[q[:,0]>.1,2]-=.06
    result=m.register(p,n,q,n)
    assert result['support_fraction']>.6
    assert np.linalg.norm(result['transform']-np.eye(4))<.003


def test_half_visible_curved_torso_retains_observable_motion():
    m=api();p,n=surface();R=Rotation.from_rotvec([.005,-.006,.004]).as_matrix();t=np.array([.002,-.001,.001])
    q=p@R.T+t;qn=n@R.T;q[p[:,0]>0,2]-=.06
    result=m.register(p,n,q,qn)
    assert .4<=result['support_fraction']<.6
    transformed=p@result['transform'][:3,:3].T+result['transform'][:3,3]
    assert np.max(np.linalg.norm(transformed-(p@R.T+t),axis=1))<.0003


def test_large_occlusion_and_invalid_points_fail():
    m=api();p,n=surface();q=p.copy();q[q[:,0]>-.1,2]-=.06
    with pytest.raises(ValueError):m.register(p,n,q,n)
    q=p.copy();q[0,0]=np.nan
    with pytest.raises(ValueError):m.register(p,n,q,n)


def test_cloud_unprojection_preserves_skewed_camera_geometry():
    m=api();y,x=np.mgrid[:120,:160]
    rgb=np.stack([x,y,np.zeros_like(x)],-1).astype(np.uint8)
    k=np.array([[180.,22.,80.],[0.,190.,60.],[0.,0.,1.]])
    points,_,colors=m.rgbd_cloud(rgb,np.full((120,160),.8),np.ones((120,160),bool),k,
                                np.array([[10,10],[149,10],[149,109],[10,109]]))
    projected=points@k.T;projected=projected[:,:2]/projected[:,2,None]
    np.testing.assert_allclose(projected,colors[:,:2],atol=1e-8)


@pytest.mark.parametrize('warm_start',[False,True])
def test_recorded_occlusion_boundary_cycle_converges_at_sensor_precision(warm_start):
    # Actual Isaac RGB-D clouds, frames 0/383 of the failed R1_v3 diagnostic.
    # This fixed sensor fixture is a numerical regression, never task training data.
    with np.load(Path(__file__).parent/'fixtures/registration_boundary_cycle.npz') as f:
        p=f['reference_points'];n=f['reference_normals']
        result=api().register(p,n,f['occluded_points'],f['occluded_normals'],
                              initial=f['previous_transform'] if warm_start else None)
    transform=result['transform']
    displacement=np.linalg.norm(p@transform[:3,:3].T+transform[:3,3]-p,axis=1)
    assert displacement.max()<.0002
    assert result['rmse_m']<.00015
    assert result['support_fraction']>.8
