"""Injected CPU camera protocol tests; never Isaac or settling qualification."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CAMERA = (Path(__file__).resolve().parents[1]/'sim/camera_framing_20260923/recommended_camera.json').read_bytes()
DIGEST = hashlib.sha256(CAMERA).hexdigest()


class FakeEnv:
    def __init__(self):
        self.world = SimpleNamespace(current_time=7.)
        self.value = dict(time_s=0.,q=np.arange(7)/10.,dq=np.zeros(7),compression_m=.001)

    def state(self): return deepcopy(self.value)


class FakeBackend:
    def __init__(self,env):
        self.env = env
        original = json.loads(CAMERA)
        self.camera = dict(camera_path='/World/RobotCam',
            world_from_camera_usd=original['world_from_camera_usd'],
            intrinsics={key:original['intrinsics'][key]/3 for key in ('fx','fy','cx','cy')},
            width=640,height=360,meters_per_unit=1.,projection='perspective',time_varying=False)
        self.rgb = np.full((360,640,4),71,np.uint8)
        self.depth = np.full((360,640),.54321,np.float32)
        self.events = []
        self.hooks = {}
        self.product = False
        self.attached = set()
        self.release_errors = set()

    def event(self,name):
        self.events.append(name)
        if name in self.hooks: self.hooks[name]()

    def snapshot_camera(self):
        self.event('camera')
        return deepcopy(self.camera)

    def setup(self):
        self.product = True
        self.attached.add('rgb')
        self.event('setup')
        self.attached.add('distance_to_image_plane')

    def render(self): self.event('render')

    def read(self):
        self.event('read')
        return self.rgb,self.depth

    def detach(self,name):
        self.event('detach:'+name)
        if 'detach:'+name in self.release_errors: raise RuntimeError('cannot detach '+name)
        self.attached.discard(name)

    def destroy_product(self):
        self.event('destroy_product')
        if 'destroy_product' in self.release_errors: raise RuntimeError('cannot destroy product')
        self.product = False


def module():
    assert importlib.util.find_spec('sim.ausc_policy.training_camera') is not None, 'guarded camera stream is missing'
    return importlib.import_module('sim.ausc_policy.training_camera')


def make(*,env=None,backend=None,**kwargs):
    env = env or FakeEnv();backend = backend or FakeBackend(env)
    options = dict(camera_bytes=CAMERA,camera_sha256=DIGEST,backend_factory=lambda current:backend)
    options.update(kwargs)
    return module().GuardedRGBDStream(env,**options),env,backend


def assert_released(backend):
    assert backend.events[-3:]==['detach:rgb','detach:distance_to_image_plane','destroy_product']
    assert not backend.product and not backend.attached


def test_native_arrays_and_measured_snapshot_are_owned_metric_data(tmp_path):
    stream,env,backend=make()
    backend.depth[0,:5]=[np.nan,np.inf,-np.inf,0.,-.1]
    expected=backend.depth.copy()
    result=stream.read()
    np.testing.assert_array_equal(result['depth_m'],expected)
    np.testing.assert_array_equal(result['depth_valid'],np.isfinite(expected)&(expected>0))
    assert result['depth_valid'].dtype==bool
    assert result['rgb'].dtype==np.uint8 and result['rgb'].shape==(360,640,4)
    assert result['actual_camera']==backend.camera
    assert result['measured_state']['time_s']==0. and result['world_time_s']==7.
    np.testing.assert_array_equal(result['measured_state']['q'],env.value['q'])
    assert result['acquisition_wall_time_s']>=0
    backend.rgb[:]=0;backend.depth[:]=0;env.value['q'][:]=0
    backend.camera['intrinsics']['fx']=0
    assert result['rgb'][0,0,0]==71 and result['depth_m'][1,1]>.5
    assert result['measured_state']['q'][6]==.6 and result['actual_camera']['intrinsics']['fx']>0
    assert not list(tmp_path.iterdir())
    receipt=stream.close();assert receipt['cleanup_complete'] and receipt['read_frames']==1
    assert receipt['training_eligible'] is False and receipt['physical_qualification'] is False
    assert receipt['leak_check_performed'] is False
    assert_released(backend)
    events=list(backend.events);assert stream.close()==receipt and backend.events==events


@pytest.mark.parametrize('change',[dict(camera_bytes=CAMERA+b' '),dict(camera_sha256='0'*64)])
def test_approved_camera_hash_is_required_before_backend_setup(change):
    env=FakeEnv();backend=FakeBackend(env)
    with pytest.raises(ValueError,match='camera'):make(env=env,backend=backend,**change)
    assert not backend.events


def mutate(env,backend,field):
    if field=='episode_time':env.value['time_s']+=2e-8
    elif field=='world_time':env.world.current_time+=2e-8
    elif field in ('q','dq'):env.value[field][0]+=2e-8
    elif field=='compression':env.value['compression_m']+=1e-14
    elif field=='pose':backend.camera['world_from_camera_usd'][0][3]+=2e-7
    elif field=='intrinsics':backend.camera['intrinsics']['fx']+=640*2e-7


@pytest.mark.parametrize('operation',['setup','render','read','camera'])
@pytest.mark.parametrize('field',['episode_time','world_time','q','dq','compression','pose','intrinsics'])
def test_each_operation_preserves_physics_clocks_and_camera_or_latches_failure(operation,field):
    env=FakeEnv();backend=FakeBackend(env)
    if operation in ('setup','camera'):
        backend.hooks[operation]=lambda:mutate(env,backend,field)
        with pytest.raises(ValueError) as raised:make(env=env,backend=backend)
        assert raised.value.training_camera_cleanup['cleanup_complete']
    else:
        stream,_,_=make(env=env,backend=backend)
        backend.hooks[operation]=lambda:mutate(env,backend,field)
        with pytest.raises(ValueError):stream.read()
        with pytest.raises(RuntimeError,match='failed|closed'):stream.read()
        with pytest.raises(RuntimeError,match='failed|closed'):stream.warmup(1)
        assert not stream.close()['completed']
    assert_released(backend)


@pytest.mark.parametrize('field,value',[
    ('camera_path','/World/Other'),('width',1920),('height',1080),
    ('meters_per_unit',.01),('projection','orthographic'),('time_varying',True),
])
def test_actual_camera_contract_is_checked_before_setup(field,value):
    env=FakeEnv();backend=FakeBackend(env);backend.camera[field]=value
    with pytest.raises(ValueError,match='camera'):make(env=env,backend=backend)
    assert 'setup' not in backend.events
    assert_released(backend)


@pytest.mark.parametrize('field',['episode_time','world_time','q','dq','compression'])
def test_initial_nonfinite_state_is_rejected(field):
    env=FakeEnv();backend=FakeBackend(env)
    if field=='episode_time':env.value['time_s']=np.nan
    elif field=='world_time':env.world.current_time=np.nan
    elif field=='compression':env.value['compression_m']=np.nan
    else:env.value[field][0]=np.nan
    with pytest.raises(ValueError):make(env=env,backend=backend)
    assert 'setup' not in backend.events
    assert_released(backend)


@pytest.mark.parametrize('problem',['rgb_float','rgb_shape','depth_integer','depth_shape','missing'])
def test_bad_native_frames_fail_and_release_resources(problem):
    stream,_,backend=make()
    if problem=='rgb_float':backend.rgb=backend.rgb.astype(float)
    elif problem=='rgb_shape':backend.rgb=backend.rgb[:100]
    elif problem=='depth_integer':backend.depth=backend.depth.astype(int)
    elif problem=='depth_shape':backend.depth=backend.depth[:,:,None]
    else:backend.depth=None
    with pytest.raises(ValueError):stream.read()
    assert_released(backend)
    with pytest.raises(RuntimeError):stream.warmup(1)


def test_read_returns_copies_before_post_read_camera_snapshot_can_reuse_buffers():
    stream,_,backend=make()
    backend.hooks['read']=lambda:backend.hooks.update(camera=lambda:backend.rgb.fill(99))
    result=stream.read()
    assert result['rgb'][0,0,0]==71 and backend.rgb[0,0,0]==99
    stream.close()


@pytest.mark.parametrize('count',[0,129,-1,True,1.0,None])
def test_warmup_requires_predeclared_integer_bound_and_never_retries(count):
    stream,_,backend=make()
    with pytest.raises(ValueError,match='1.*128'):stream.warmup(count)
    assert 'render' not in backend.events
    assert_released(backend)


def test_warmup_reports_each_render_without_arrays_or_convergence_claim():
    stream,env,backend=make()
    def brighten():backend.rgb[:,:,:3]+=2
    backend.hooks['render']=brighten
    report=stream.warmup(3)
    assert report['frame_count']==3 and len(report['renders'])==3
    assert [row['rgb_mean'] for row in report['renders']]==[73.,75.,77.]
    assert [row['adjacent_rgb_mean_abs_difference'] for row in report['renders']]==[None,2.,2.]
    assert [row['depth_valid_fraction'] for row in report['renders']]==[1.,1.,1.]
    assert all(row['time_s']==0. and row['world_time_s']==7. for row in report['renders'])
    assert all(row['actual_camera']==backend.camera for row in report['renders'])
    assert report['final_rgb_sha256']==hashlib.sha256(backend.rgb).hexdigest()
    assert report['warmup_wall_time_s']>=0
    assert report['training_eligible'] is False and report['physical_qualification'] is False
    assert not {'converged','accuracy','recommended_warmup_count'}&set(report)
    json.dumps(report,allow_nan=False)
    assert env.value['time_s']==0. and env.world.current_time==7.
    receipt=stream.close()
    assert receipt['read_frames']==0 and receipt['warmup_frames']==3
    assert receipt['warmup_wall_time_s']>=report['warmup_wall_time_s']
    assert receipt['acquisition_wall_time_s']==0.


def test_warmup_never_skips_missing_frames_or_retries_to_reach_count():
    stream,_,backend=make();renders=[0]
    def fail_second():
        renders[0]+=1
        if renders[0]==2:backend.depth=None
    backend.hooks['render']=fail_second
    with pytest.raises(ValueError):stream.warmup(5)
    assert renders[0]==2 and stream.close()['warmup_frames']==1
    assert_released(backend)


def test_partial_setup_and_cleanup_errors_preserve_original_exception_and_all_attempts():
    env=FakeEnv();backend=FakeBackend(env)
    initiating=RuntimeError('original attach failure')
    def fail():raise initiating
    backend.hooks['setup']=fail
    backend.release_errors={'detach:rgb','detach:distance_to_image_plane','destroy_product'}
    with pytest.raises(RuntimeError) as raised:make(env=env,backend=backend)
    assert raised.value is initiating
    receipt=raised.value.training_camera_cleanup
    assert receipt['cleanup_complete'] is False and receipt['completed'] is False
    assert len(receipt['failures'])==4
    assert backend.events[-3:]==['detach:rgb','detach:distance_to_image_plane','destroy_product']
    assert backend.product and 'rgb' in backend.attached


@pytest.mark.parametrize('failure',['detach:rgb','detach:distance_to_image_plane','destroy_product'])
def test_each_release_failure_is_explicit_and_does_not_skip_other_resources(failure):
    stream,_,backend=make();backend.release_errors={failure}
    receipt=stream.close()
    assert not receipt['cleanup_complete'] and not receipt['completed']
    assert receipt['release_attempts'][failure] is False
    assert len(receipt['failures'])==1
    assert backend.events[-3:]==['detach:rgb','detach:distance_to_image_plane','destroy_product']
    events=list(backend.events);assert stream.close()==receipt and backend.events==events


def test_native_backend_default_retains_linux_owned_process_gate(monkeypatch):
    from sim.ausc_policy import m2_capture
    monkeypatch.setattr(m2_capture.sys,'platform','darwin')
    with pytest.raises(RuntimeError,match='owned Linux'):
        module().GuardedRGBDStream(FakeEnv(),camera_bytes=CAMERA,camera_sha256=DIGEST)


def test_separate_reads_allow_actual_physics_progress_between_frames():
    stream,env,_=make();first=stream.read()
    env.value['time_s']=.05;env.world.current_time=7.05;env.value['q'][0]=.2
    second=stream.read()
    assert first['measured_state']['time_s']==0. and second['measured_state']['time_s']==.05
    assert second['measured_state']['q'][0]==.2 and second['world_time_s']==7.05
    assert stream.close()['read_frames']==2


def test_rgb_three_channels_and_invalid_depth_are_retained_without_quality_gate():
    stream,_,backend=make();backend.rgb=backend.rgb[:,:,:3];backend.depth[:]=np.nan
    frame=stream.read()
    assert frame['rgb'].shape==(360,640,3)
    assert np.isnan(frame['depth_m']).all() and not frame['depth_valid'].any()
    report=stream.warmup(1)
    assert report['renders'][0]['depth_valid_fraction']==0.
    assert stream.close()['completed']


def test_warmup_maximum_is_exact_and_keeps_only_scalar_summaries():
    stream,_,backend=make();report=stream.warmup(128)
    assert report['frame_count']==128 and len(report['renders'])==128
    assert backend.events.count('render')==128 and backend.events.count('read')==128
    json.dumps(report,allow_nan=False)
    assert all(not isinstance(value,np.ndarray) for value in vars(stream).values())
    assert stream.close()['warmup_frames']==128


def test_warmup_rejects_cumulative_state_drift_between_individually_tolerated_renders():
    stream,env,backend=make()
    backend.hooks['render']=lambda:env.value.__setitem__('time_s',env.value['time_s']+6e-9)
    with pytest.raises(ValueError,match='warmup'):stream.warmup(3)
    assert backend.events.count('render')==2
    assert stream.close()['warmup_frames']==1
    assert_released(backend)


def test_read_preserves_original_error_when_cleanup_also_fails():
    stream,_,backend=make();initiating=RuntimeError('original read failure')
    def fail():raise initiating
    backend.hooks['read']=fail;backend.release_errors={'detach:rgb','destroy_product'}
    with pytest.raises(RuntimeError) as raised:stream.read()
    assert raised.value is initiating
    receipt=stream.close()
    assert receipt==raised.value.training_camera_cleanup
    assert not receipt['cleanup_complete'] and not receipt['completed']
    assert len(receipt['failures'])==3 and receipt['read_frames']==0
    assert receipt['acquisition_wall_time_s']>0 and receipt['warmup_wall_time_s']==0


def test_failure_receipt_is_bounded_and_callers_cannot_mutate_idempotent_result():
    stream,_,backend=make()
    def fail():raise RuntimeError('x'*100_000)
    backend.hooks['read']=fail
    with pytest.raises(RuntimeError):stream.read()
    receipt=stream.close()
    assert len(receipt['failures'][0]['error'])<=1024
    receipt['failures'].clear();receipt['release_attempts'].clear()
    assert stream.close()['failures'] and len(stream.close()['release_attempts'])==3


@pytest.mark.parametrize('field',['time_s','world','q','dq'])
def test_complete_read_rejects_cumulative_render_and_read_drift(field):
    stream,env,backend=make()
    def drift():
        if field=='world':env.world.current_time+=6e-9
        elif field in ('q','dq'):env.value[field][0]+=6e-9
        else:env.value[field]+=6e-9
    backend.hooks['render']=drift;backend.hooks['read']=drift
    with pytest.raises(ValueError):stream.read()
    assert_released(backend)
    assert not stream.close()['completed'] and stream.close()['read_frames']==0


class UnprintableError(RuntimeError):
    def __str__(self):raise RuntimeError('secondary error formatting failure')


def test_unprintable_initiating_error_is_preserved_and_does_not_skip_cleanup():
    stream,_,backend=make();initiating=UnprintableError()
    def fail():raise initiating
    backend.hooks['read']=fail
    with pytest.raises(UnprintableError) as raised:stream.read()
    assert raised.value is initiating
    assert_released(backend)
    receipt=stream.close()
    assert receipt['cleanup_complete'] and not receipt['completed']
    assert receipt==initiating.training_camera_cleanup
    assert 'UnprintableError' in receipt['failures'][0]['error']
    assert len(receipt['failures'][0]['error'])<=1024


def test_unprintable_release_error_is_recorded_and_remaining_releases_are_attempted():
    stream,_,backend=make();detach=backend.detach
    def fail_first(name):
        if name=='rgb':
            backend.event('detach:rgb')
            raise UnprintableError()
        detach(name)
    backend.detach=fail_first
    receipt=stream.close()
    assert backend.events[-3:]==['detach:rgb','detach:distance_to_image_plane','destroy_product']
    assert not backend.product and not receipt['cleanup_complete'] and not receipt['completed']
    assert receipt['release_attempts']=={'detach:rgb':False,'detach:distance_to_image_plane':True,'destroy_product':True}
    assert 'UnprintableError' in receipt['failures'][0]['error']
