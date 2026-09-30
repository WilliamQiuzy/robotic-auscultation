"""Orchestration faults must preserve physical rows and close each resource."""
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest

from sim.ausc_policy.m2_sensor_probe import capture_home, close_resources


class Environment:
    def __init__(self):
        self.rows=[];self.last_contacts={};self.video_recorder=None
        self.world=SimpleNamespace(current_time=7.)
        self.value=dict(time_s=0.,q=np.zeros(7),dq=np.zeros(7),compression_m=0.)
    def state(self): return deepcopy(self.value)
    def step(self,command,*args):
        self.value['time_s']=(len(self.rows)+1)/240.
        self.world.current_time=7.+self.value['time_s']
        self.rows.append(self.state());return self.state()


def test_every_real_step_precedes_its_same_step_camera_and_video():
    env=Environment();events=[];checks=[]
    def check(q,**kwargs):
        checks.append(kwargs);return dict(clear=True)
    def append(row,stamp,compression):
        assert len(env.rows)==row+1;events.append(('hall',row))
        return dict(row=row,time=stamp,compression=compression)
    def capture(state,**kwargs):
        row=kwargs['physics_row'];assert events[-1]==('hall',row)
        assert kwargs['hall']['row']==row
        events.append(('camera',row))
    def video():
        row=len(env.rows)-1
        assert events[-1]==(('camera' if row%12==11 else 'hall'),row)
        events.append(('video',row))
    capture_home(env,SimpleNamespace(check=check),lambda q:dict(clear=True),
                 SimpleNamespace(append=append),SimpleNamespace(capture=capture),video_step=video)
    assert len(env.rows)==240 and len(checks)==480
    assert len([e for e in events if e[0]=='camera'])==20
    assert all(c==dict(clearance=.003,allow_patient_contact=False,compression=0.) for c in checks)


def test_observed_failure_still_preserves_real_sensor_step_and_due_frame():
    env=Environment();calls=[]
    def check(q,**kw): return dict(clear=len(env.rows)<12)
    def append(row,*args):calls.append(('hall',row));return {}
    def capture(state,**kw):
        assert kw['context']['physical_audit_error'];calls.append(('camera',kw['physics_row']))
    with pytest.raises(ValueError,match='observed'):
        capture_home(env,SimpleNamespace(check=check),lambda q:dict(clear=True),
                     SimpleNamespace(append=append),SimpleNamespace(capture=capture))
    assert len(env.rows)==12 and calls[-2:]==[('hall',11),('camera',11)]


def test_video_cannot_silently_advance_physics_after_camera():
    env=Environment()
    def video():env.world.current_time+=1/240.
    with pytest.raises(ValueError,match='video render advanced physics clock'):
        capture_home(env,SimpleNamespace(check=lambda *a,**k:dict(clear=True)),lambda q:dict(clear=True),
                     SimpleNamespace(append=lambda *a:{}),SimpleNamespace(capture=lambda *a,**k:None),
                     video_step=video)
    assert len(env.rows)==1


@pytest.mark.parametrize('invalid',[float('nan'),float('inf')])
def test_last_video_frame_rejects_nonfinite_world_clock(invalid):
    env=Environment()
    def video():
        if len(env.rows)==240:env.world.current_time=invalid
    with pytest.raises(ValueError,match='world clock'):
        capture_home(env,SimpleNamespace(check=lambda *a,**k:dict(clear=True)),lambda q:dict(clear=True),
            SimpleNamespace(append=lambda *a:{}),SimpleNamespace(capture=lambda *a,**k:None),video_step=video)


def test_cleanup_attempts_all_independent_resources_and_keeps_failures():
    events=[];initiating=ValueError('physical failure')
    def fail(name):
        def op(*args):
            events.append(name);raise RuntimeError(name+' failure')
        return op
    env=SimpleNamespace(video_recorder=object(),save=fail('trace'),world=SimpleNamespace(stop=fail('world')))
    receipts,errors=close_resources(SimpleNamespace(finalize=fail('camera')),
        SimpleNamespace(finalize=fail('hall')),SimpleNamespace(close=fail('video')),env,initiating)
    assert events==['camera','hall','trace','video','world']
    assert len(errors)==5 and not receipts and env.video_recorder is None


@pytest.mark.parametrize('capture_fails',[False,True])
def test_native_assembly_uses_flat_geometry_hashes_and_single_calibration_snapshot(tmp_path, monkeypatch,capture_fails):
    import json
    from pathlib import Path
    from sim.ausc_policy import m2_sensor_probe as probe, fresh_expert, contact_collision
    from sim.ausc_policy.geometry_epoch import require_geometry_artifacts
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    config=json.loads(Path('configs/ausc_policy/m2_sensor_probe_v1.json').read_text())
    camera=Path('sim/camera_framing_20260923/recommended_camera.json')
    calibration=Path('sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json')
    approved=calibration.read_bytes()
    config['m2_sensor_probe'].update(camera_file=str(camera),calibration_file=str(calibration))
    geometry=tmp_path/'geometry';geometry.mkdir()
    (geometry/'collision_geometry.json').write_bytes(b'{}')
    (geometry/'collision_geometry.npz').write_bytes(b'fixture')
    read_bytes=Path.read_bytes;reads=[];received=[]
    def changing_read(path):
        if path==calibration:
            reads.append(path)
            return approved if len(reads)==1 else b'replaced'
        return read_bytes(path)
    monkeypatch.setattr(Path,'read_bytes',changing_read)
    monkeypatch.setattr(probe.sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    class Env(Environment):
        def __init__(self,*args,**kw):
            super().__init__();self.base=np.eye(4)
            self.world.stop=lambda:None
        def save(self,*args):return {}
    class Checker:
        def __init__(self,kin,base,directory,metadata):
            require_geometry_artifacts(directory,metadata)
        def check(self,*args,**kwargs):return dict(clear=True)
    class Recorder:
        def __init__(self,*args):pass
        def after_step(self):pass
        def close(self):return dict(frames=10)
    class Hall:
        def __init__(self,output,**kwargs):
            self.output=output;received.append(kwargs['calibration_bytes'])
        def append(self,*args):return {}
        def finalize(self,error):
            assert error is None or isinstance(error,str)
            return dict(completed=True,physics_rows=240)
    class Camera:
        def __init__(self,*args,**kwargs):received.append(kwargs['calibration_bytes'])
        def capture(self,*args,**kwargs):
            if capture_fails:raise RuntimeError('native capture failed')
        def finalize(self,error):
            assert error is None or isinstance(error,str)
            return dict(completed=True,frames=20)
    monkeypatch.setattr(fresh_expert,'FreshContactEnv',Env)
    monkeypatch.setattr(fresh_expert,'_VideoRecorder',Recorder)
    monkeypatch.setattr(contact_collision,'ContactCollisionChecker',Checker)
    monkeypatch.setattr(PostureLimits,'from_dict',lambda *a:SimpleNamespace(evaluate=lambda *a,**k:dict(clear=True)))
    monkeypatch.setattr(probe,'M2HallStream',Hall);monkeypatch.setattr(probe,'M2RGBDCapture',Camera)
    monkeypatch.setattr(probe,'audit_hall_stream',lambda *a:dict(passed=True))
    result=probe.run_probe(None,object(),tmp_path,config,dict(mass_kg=.066))
    if capture_fails:
        assert not result['capture_completed'] and result['physics_steps']==12
        assert [e['stage'] for e in result['errors']]==['capture']
        assert 'RuntimeError: native capture failed' in (tmp_path/'sensor_probe_failure.txt').read_text()
    else:
        assert result['capture_completed'],result['errors']
        assert result['physics_steps']==240
    assert len(reads)==1 and received==[approved,approved]
    assert not result['physical_validation']


def test_camera_failure_does_not_mask_simultaneous_physical_failure():
    env=Environment()
    def broken_camera(*a,**k):raise RuntimeError('camera broken')
    with pytest.raises(ValueError,match='observed home') as caught:
        capture_home(env,SimpleNamespace(check=lambda *a,**k:dict(clear=len(env.rows)<12)),
            lambda q:dict(clear=True),SimpleNamespace(append=lambda *a:{}),
            SimpleNamespace(capture=broken_camera))
    assert isinstance(caught.value.__cause__,RuntimeError)


@pytest.mark.parametrize('camera_fails',[False,True])
def test_physical_failure_does_not_retain_environment_in_traceback_cycle(camera_fails):
    import gc
    import traceback
    import weakref
    def scope():
        env=Environment();reference=weakref.ref(env)
        def camera(*a,**k):
            if camera_fails:raise RuntimeError('camera failed too')
        try:
            capture_home(env,SimpleNamespace(check=lambda *a,**k:dict(clear=len(env.rows)<12)),
                lambda q:dict(clear=True),SimpleNamespace(append=lambda *a:{}),SimpleNamespace(capture=camera))
        except Exception as error:
            assert 'observed home' in repr(error)
            assert 'observed home' in traceback.format_exc()
        return reference
    enabled=gc.isenabled();gc.disable()
    try: assert scope()() is None
    finally:
        if enabled:gc.enable()
