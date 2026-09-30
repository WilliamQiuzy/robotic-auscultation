"""CPU acquisition integration: native rendering is deliberately injected."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
from sim.ausc_policy.m2_capture import APPROVED_CAMERA_SHA256
from sim.ausc_policy.m2_sensor_probe import CALIBRATION_SHA256

ROOT = Path(__file__).resolve().parents[1]
CAMERA = ROOT/'sim/camera_framing_20260923/recommended_camera.json'
CALIBRATION = ROOT/'sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json'
GEOMETRY = dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
                freshly_generated=True, geometry_artifacts={
                    'collision_geometry_sha256': 'a'*64, 'collision_arrays_sha256': 'b'*64})


def module():
    assert importlib.util.find_spec('sim.ausc_policy.expert_capture'), 'Expert capture integration is missing'
    return importlib.import_module('sim.ausc_policy.expert_capture')


def binding():
    return dict(enabled=True, episode_id='m2-expert-cpu-1', camera_file=str(CAMERA),
        camera_sha256=APPROVED_CAMERA_SHA256, calibration_file=str(CALIBRATION),
        calibration_sha256=CALIBRATION_SHA256)


class Env:
    def __init__(self):
        self.rows=[]; self.world=SimpleNamespace(current_time=7.)
        self.value=dict(time_s=0., q=np.zeros(7), dq=np.zeros(7), compression_m=.001)
    def state(self): return deepcopy(self.value)
    def advance(self, audited=True):
        self.value['time_s']=(len(self.rows)+1)/240.
        self.world.current_time=7.+self.value['time_s']
        row=dict(self.state(),phase='APPROACH',target='R3',subphase='contact_cycle')
        if audited: row['physical_violations']=[]
        self.rows.append(row)


class Backend:
    def __init__(self,env):
        self.env=env; self.events=[]
        approved=json.loads(CAMERA.read_bytes())
        self.camera=dict(camera_path='/World/RobotCam',
            world_from_camera_usd=approved['world_from_camera_usd'],
            intrinsics={k:approved['intrinsics'][k]/3 for k in ('fx','fy','cx','cy')},
            width=640,height=360,meters_per_unit=1.,projection='perspective')
    def snapshot_camera(self): return deepcopy(self.camera)
    def setup(self): pass
    def render(self): self.events.append('render')
    def read(self): return np.zeros((360,640,3),np.uint8),np.full((360,640),.5,np.float32)
    def detach(self,name): self.events.append('detach:'+name)
    def destroy_product(self): self.events.append('destroy')


def labels(phase='APPROACH'):
    return dict(phase=phase,target='R3',subphase='contact_cycle',active=True,recovery=False)


def new_capture(tmp_path,env):
    capture=module().ExpertCapture(tmp_path/'raw',binding=binding(),geometry_metadata=GEOMETRY)
    capture.start(env,backend_factory=Backend)
    return capture


@pytest.mark.parametrize('bad',[None,True,{},dict(enabled=1),dict(enabled=True),
    dict(binding(),episode_id='../escape'),dict(binding(),camera_sha256='0'*64),
    dict(binding(),calibration_sha256='0'*64),dict(binding(),extra=True)])
def test_strict_opt_in_settings(bad):
    with pytest.raises(ValueError,match='raw_multimodal_capture'):
        module().capture_settings(dict(raw_multimodal_capture=bad))


def test_missing_or_explicit_off_remain_disabled():
    assert module().capture_settings({}) is None
    assert module().capture_settings(dict(raw_multimodal_capture=dict(enabled=False))) is None
    assert module().capture_settings(dict(raw_multimodal_capture=binding()))==binding()


def test_raw_hall_and_frames_bind_physics_and_postupdate_labels(tmp_path):
    env=Env(); capture=new_capture(tmp_path,env)
    for row in range(24):
        env.advance(); capture.after_physics(env,row)
        capture.finish_tick(env,row,labels(),labels('HOLD'))
    capture.close()
    result=capture.result(24,execution_completed=True)
    assert result['raw_capture_complete'] and not result['training_eligible']
    assert not result['physical_qualification'] and not result['randomization_qualified']
    assert result['receipts']['hall_replay']['passed']
    assert result['receipts']['hall']['physics_rows']==24
    frames=[json.loads(line) for line in (capture.output/'rgbd/frames.jsonl').read_text().splitlines()]
    assert [row['physics_row'] for row in frames]==[11,23]
    for frame in frames:
        assert frame['phase']=='HOLD' and frame['target']=='R3'
        assert frame['hall_physics']['compression_m']==.001
        assert frame['context']['controller_before']['phase']=='APPROACH'
        assert frame['context']['controller_after']['phase']=='HOLD'
        assert frame['context']['physics_labels']['phase']=='APPROACH'
        assert frame['context']['manager_update']=='completed'
    assert json.loads((capture.output/'receipt.json').read_text())==result
    assert (capture.output/'approved_camera.json').read_bytes()==CAMERA.read_bytes()
    assert (capture.output/'calibration.json').read_bytes()==CALIBRATION.read_bytes()


def test_due_frame_preserved_with_incomplete_audit_and_original_error(tmp_path):
    env=Env(); capture=new_capture(tmp_path,env)
    for row in range(12):
        env.advance(audited=row!=11); capture.after_physics(env,row)
        capture.finish_tick(env,row,labels(),labels('APPROACH'),
                            error='gravity readback failed' if row==11 else None)
    capture.close('gravity readback failed')
    result=capture.result(12,execution_completed=False)
    assert not result['raw_capture_complete']
    frame=json.loads((capture.output/'rgbd/frames.jsonl').read_text())
    assert frame['phase']=='APPROACH'
    assert frame['context']['manager_update']=='not_completed'
    assert frame['context']['tick_error']=='gravity readback failed'


@pytest.mark.parametrize('change',['clock','q','dq','compression','nan'])
def test_video_guard_rejects_any_physics_change(change):
    env=Env()
    def render():
        if change=='clock':env.world.current_time+=1/240.
        elif change=='nan':env.world.current_time=float('nan')
        elif change=='compression':env.value['compression_m']+=1e-6
        else:env.value[change][0]+=.01
    with pytest.raises(ValueError,match='video'):
        module().checked_render(env,render,'video')


def test_cleanup_is_independent_and_aggregate_failure_revokes_completion(tmp_path):
    env=Env();capture=new_capture(tmp_path,env);events=[]
    original_hall=capture.hall
    def camera_close(error): events.append('camera'); raise OSError('camera close failed')
    def hall_close(error): events.append('hall'); return original_hall.finalize(error)
    original_camera=capture.camera
    capture.camera=SimpleNamespace(finalize=camera_close)
    capture.hall=SimpleNamespace(finalize=hall_close,output=original_hall.output)
    try:
        capture.close('initiating failure')
        assert events==['camera','hall']
        result=capture.result(0,execution_completed=False,errors=[dict(stage='video',error='failed')])
        assert not result['raw_capture_complete']
        assert {r['stage'] for r in result['errors']} >= {'camera_finalize','video'}
        assert capture.camera is None and capture.hall is None
    finally:original_camera.finalize('test cleanup')


def test_exact_frame_count_and_source_fingerprints_are_required(tmp_path):
    env=Env(); capture=new_capture(tmp_path,env)
    for row in range(12):
        env.advance();capture.after_physics(env,row);capture.finish_tick(env,row,labels(),labels())
    capture.close()
    result=capture.result(24,execution_completed=True)
    assert not result['raw_capture_complete']
    assert any(item['stage']=='counts' for item in result['errors'])


def test_changed_runtime_source_revokes_complete_acquisition(tmp_path):
    env=Env();source=tmp_path/'source.py';source.write_text('original')
    capture=module().ExpertCapture(tmp_path/'raw',binding=binding(),geometry_metadata=GEOMETRY,
                                  source_paths=[source])
    capture.start(env,backend_factory=Backend)
    for row in range(12):
        env.advance();capture.after_physics(env,row);capture.finish_tick(env,row,labels(),labels())
    capture.close();source.write_text('replacement')
    result=capture.result(12,execution_completed=True)
    assert not result['raw_capture_complete']
    assert any(error['stage']=='source' for error in result['errors'])


def test_partial_camera_setup_keeps_its_receipt_and_closes_hall(tmp_path):
    env=Env();capture=module().ExpertCapture(tmp_path/'raw',binding=binding(),geometry_metadata=GEOMETRY)
    class FailingBackend(Backend):
        def setup(self):raise RuntimeError('backend setup failed')
    with pytest.raises(RuntimeError,match='backend setup'):
        capture.start(env,backend_factory=FailingBackend)
    capture.close('backend setup failed')
    result=capture.result(0,execution_completed=False)
    assert not result['raw_capture_complete']
    assert result['receipts']['camera']['failures']
    assert result['receipts']['hall']['failure']
    assert capture.camera is None and capture.hall is None


def test_tick_error_cannot_be_forgotten_by_the_completion_caller(tmp_path):
    env=Env();capture=new_capture(tmp_path,env)
    for row in range(12):
        env.advance();capture.after_physics(env,row)
        capture.finish_tick(env,row,labels(),labels(),error='physical gate rejected' if row==11 else None)
    capture.close()
    result=capture.result(12,execution_completed=True)
    assert not result['raw_capture_complete']
    assert any(error['stage']=='controller_tick' for error in result['errors'])


def test_step_and_tick_labels_cannot_bind_different_targets(tmp_path):
    env=Env();capture=new_capture(tmp_path,env)
    env.advance();capture.after_physics(env,0)
    try:
        with pytest.raises(ValueError,match='labels'):
            capture.finish_tick(env,0,labels(),dict(labels(),target='L3'))
    finally:capture.close('test rejection')


def test_capture_receipt_write_failure_is_reported_and_not_complete(tmp_path,monkeypatch):
    env=Env();capture=new_capture(tmp_path,env)
    for row in range(12):
        env.advance();capture.after_physics(env,row);capture.finish_tick(env,row,labels(),labels())
    capture.close()
    def fail(*args):raise OSError('receipt unavailable')
    monkeypatch.setattr(capture,'_write',fail)
    result=capture.result(12,execution_completed=True)
    assert not result['raw_capture_complete']
    assert any(error['stage']=='receipt_write' for error in result['errors'])


def test_changed_calibration_rejected_before_creating_resources(tmp_path):
    config=binding(); path=tmp_path/'changed.json';path.write_bytes(b'{}')
    config['calibration_file']=str(path)
    with pytest.raises(ValueError,match='Pinned'):
        module().ExpertCapture(tmp_path/'raw',binding=config,geometry_metadata=GEOMETRY)
    assert not (tmp_path/'raw').exists()


@pytest.mark.parametrize('fault',['none','trace','video','setup_clock','video_setup','result'])
def test_run_probe_owns_capture_and_revokes_for_any_cleanup_failure(tmp_path,monkeypatch,fault):
    import sys
    from sim.ausc_policy import fresh_expert as fresh, contact_collision
    capture_module=module();events=[]
    original_camera=capture_module.M2RGBDCapture
    class CPUCamera(original_camera):
        def __init__(self,*args,**kwargs):
            kwargs['backend_factory']=Backend
            super().__init__(*args,**kwargs)
    class RunEnv(fresh.FreshContactEnv,Env):
        def __init__(self,output,**kwargs):
            Env.__init__(self);self.out=Path(output);self.out.mkdir()
            self.base=np.eye(4);self.slider=7;self.robot=SimpleNamespace(apply_action=lambda _:None)
            self.world.stop=lambda:events.append('stop')
        def save(self,name):
            events.append('trace')
            if fault=='trace':raise OSError('trace failed')
            (self.out/(name+'_trace.jsonl')).write_text('retained physical trace\n')
        state=Env.state
    def parent_step(env,command,label,target,meta):
        env.advance();env.rows[-1].update(meta,phase=label,target=target)
        return env.state()
    class Recorder:
        def __init__(self,env,*args):
            self.env=env
            if fault=='setup_clock':env.world.current_time+=1/240.
            if fault=='video_setup':raise OSError('video setup failed')
        def after_step(self):pass
        def close(self):
            events.append('video')
            if fault=='video':raise OSError('video failed')
            return dict(frames=len(self.env.rows)//24)
    class Controller:
        def __init__(self,env,*args,**kwargs):self.env=env;self.control_rows=[]
        def run(self,points):
            for index in range(24):
                self.env.step(np.zeros(7),'APPROACH','R3',dict(subphase='contact_cycle'))
                self.env.raw_capture.finish_tick(self.env,index,labels(),labels('HOLD'))
            return dict(success=True,points=[])
    metadata=dict(GEOMETRY,world_from_left_base=np.eye(4).tolist(),plans=[])
    monkeypatch.setattr(sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    monkeypatch.setitem(sys.modules,'isaacsim.core.utils.types',SimpleNamespace(ArticulationAction=SimpleNamespace))
    monkeypatch.setattr(capture_module,'M2RGBDCapture',CPUCamera)
    monkeypatch.setattr(fresh.IsaacContactEnv,'step',parent_step)
    monkeypatch.setattr(fresh,'FreshContactEnv',RunEnv);monkeypatch.setattr(fresh,'FreshExpert',Controller)
    monkeypatch.setattr(fresh,'_VideoRecorder',Recorder)
    monkeypatch.setattr(fresh,'read_plans',lambda *args:([object()],metadata))
    monkeypatch.setattr(contact_collision,'ContactCollisionChecker',lambda *args:SimpleNamespace(static_bundle={}))
    monkeypatch.setattr(fresh,'require_geometry_artifacts',lambda *args:{})
    original_write=fresh._write
    def write(path,value):
        if fault=='result' and path.name=='fresh_expert_result.json':raise OSError('final receipt failed')
        return original_write(path,value)
    monkeypatch.setattr(fresh,'_write',write)
    config=dict(GEOMETRY,fresh_expert=dict(passive_preload_N=1.5,raw_multimodal_capture=binding()))
    result=fresh.run_probe(None,None,tmp_path,config,dict(mass_kg=.067))
    assert events==['trace','video','stop']
    assert result['raw_capture_complete']==(fault=='none')
    assert result['raw_multimodal_capture']['raw_capture_complete']==(fault=='none')
    assert not result['training_eligible'] and not result['raw_multimodal_training_capture']
    receipt=json.loads((tmp_path/'raw_multimodal/receipt.json').read_text())
    assert receipt==result['raw_multimodal_capture']
    if fault=='none':assert receipt['receipts']['hall_replay']['passed']


@pytest.mark.parametrize('camera_fails',[False,True])
def test_rejected_tick_does_not_keep_isaac_objects_in_an_exception_cycle(tmp_path,monkeypatch,camera_fails):
    import gc
    import weakref
    from sim.ausc_policy import fresh_expert as fresh
    def parent(controller,*args,**kwargs):
        before=len(controller.env.rows)
        controller.env.advance();controller.env.raw_capture.after_physics(controller.env,before)
        raise ValueError('observed collision rejected')
    monkeypatch.setattr(fresh.SkillExpert,'_tick',parent)
    def scope():
        env=Env();capture=new_capture(tmp_path,env);env.raw_capture=capture
        for row in range(11):
            env.advance();capture.after_physics(env,row);capture.finish_tick(env,row,labels(),labels())
        controller=fresh.FreshExpert.__new__(fresh.FreshExpert);controller.env=env
        controller.target=SimpleNamespace(name='R3')
        controller.manager=SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='APPROACH')))
        if camera_fails:
            def failed_render():raise RuntimeError('camera render failed')
            capture.camera._backend.render=failed_render
        references=weakref.ref(env),weakref.ref(controller)
        try:controller._tick(np.zeros(7),'contact_cycle',active=True)
        except ValueError as error:assert 'observed collision' in str(error)
        capture.close('test rejected tick')
        return references
    was_enabled=gc.isenabled();gc.disable()
    try:
        env_ref,controller_ref=scope()
        assert controller_ref() is None and env_ref() is None
    finally:
        if was_enabled:gc.enable()
