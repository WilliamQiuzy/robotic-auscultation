"""Rigid-body tracking candidate: explicitly bounded synthetic RGB-D tests."""
import importlib
import numpy as np
import pytest


def api():
    try:return importlib.import_module('sim.auscultation_vla_v2.body_tracking')
    except ModuleNotFoundError:pytest.fail('body tracking missing')


K=np.array([[220.,0,120],[0,220.,100],[0,0,1.]])
POLY=np.array([[25,25],[214,25],[214,174],[25,174]])
SEEDS={'R1':[85.,75.],'L1':[155.,75.]}


def frame():
    y,x=np.mgrid[:200,:240];u=(x-120)/220;v=(y-100)/220
    d=(.8+.4*u*u+.3*v*v+.015*np.sin(16*u)*np.cos(19*v)).astype(np.float32)
    return np.full((200,240,3),160,np.uint8),d,np.ones(d.shape,bool)


def test_low_texture_surface_initializes_all_points_and_tracks_static():
    m=api();t=m.BodyTracker(K,np.eye(4))
    a=t.initialize(*frame(),SEEDS,POLY,rgb_time_s=1.,depth_time_s=1.,now_s=1.)
    b=t.update(*frame(),rgb_time_s=1.05,depth_time_s=1.05,now_s=1.05)
    assert set(a)==set(SEEDS)
    for key in a:
        np.testing.assert_allclose(a[key].position_base_m,b[key].position_base_m,atol=1e-5)
        assert b[key].revision==a[key].revision==0


def test_metric_translation_changes_all_marked_points_without_reseeding():
    m=api();t=m.BodyTracker(K,np.eye(4));rgb,d,v=frame()
    a=t.initialize(rgb,d,v,SEEDS,POLY,rgb_time_s=1.,depth_time_s=1.,now_s=1.)
    y,x=np.mgrid[:200,:240];u=(x-120)/220;w=(y-100)/220
    tx,tz=.004,.003;z=d.astype(float)
    # Analytic ray intersection with the translated synthetic surface.
    for _ in range(30):
        uu=(u*z-tx)/(z-tz);vv=w*z/(z-tz)
        z=.8+.4*uu*uu+.3*vv*vv+.015*np.sin(16*uu)*np.cos(19*vv)+tz
    b=t.update(rgb,z.astype(np.float32),v,rgb_time_s=1.05,depth_time_s=1.05,now_s=1.05)
    for key in a:
        np.testing.assert_allclose(b[key].position_base_m-a[key].position_base_m,[tx,0,tz],atol=.0015)
        np.testing.assert_allclose(b[key].velocity_base_m_s,[tx/.05,0,tz/.05],atol=.03)


@pytest.mark.parametrize('noise',[.0001,.001])
def test_depth_noise_cannot_make_a_plane_observable(noise):
    m=api();rgb,d,v=frame();d[:]=.8
    d+=np.random.default_rng(9).normal(0,noise,d.shape).astype(np.float32)
    t=m.BodyTracker(K,np.eye(4))
    with pytest.raises(ValueError):t.initialize(rgb,d,v,SEEDS,POLY,rgb_time_s=1.,depth_time_s=1.,now_s=1.)


def test_rgb_clock_reversal_latches_independently():
    m=api();t=m.BodyTracker(K,np.eye(4));t.initialize(*frame(),SEEDS,POLY,rgb_time_s=1.,depth_time_s=1.,now_s=1.)
    with pytest.raises(ValueError):t.update(*frame(),rgb_time_s=.995,depth_time_s=1.01,now_s=1.01)
    assert t.lost


def test_frame_resize_latches_and_fresh_explicit_reinitialization_allows_gap():
    m=api();t=m.BodyTracker(K,np.eye(4));t.initialize(*frame(),SEEDS,POLY,rgb_time_s=1.,depth_time_s=1.,now_s=1.)
    rgb,d,v=frame()
    with pytest.raises(ValueError):t.update(rgb[:,1:],d[:,1:],v[:,1:],rgb_time_s=1.05,depth_time_s=1.05,now_s=1.05)
    assert t.lost
    t=m.BodyTracker(K,np.eye(4));t.initialize(*frame(),SEEDS,POLY,rgb_time_s=1.,depth_time_s=1.,now_s=1.)
    result=t.initialize(*frame(),SEEDS,POLY,rgb_time_s=1.3,depth_time_s=1.3,now_s=1.3)
    assert result['R1'].revision==1 and not t.lost


@pytest.mark.parametrize('fault',['missing_depth','color_replacement','stale','large_occlusion','planar'])
def test_unobservable_or_invalid_input_latches(fault):
    m=api();t=m.BodyTracker(K,np.eye(4));t.initialize(*frame(),SEEDS,POLY,rgb_time_s=1.,depth_time_s=1.,now_s=1.)
    rgb,d,v=frame();stamp=1.05
    if fault=='missing_depth':v[:]=False
    elif fault=='color_replacement':rgb[:]=20
    elif fault=='stale':stamp=.9
    elif fault=='large_occlusion':d[:,:180]-=.1
    elif fault=='planar':d[:]=.8
    with pytest.raises(ValueError):t.update(rgb,d,v,rgb_time_s=stamp,depth_time_s=stamp,now_s=1.05)
    assert t.lost
    with pytest.raises(ValueError):t.update(*frame(),rgb_time_s=1.1,depth_time_s=1.1,now_s=1.1)
