"""CPU protocol probes only; no native renderer, physics or settling evidence."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
CAMERA=(ROOT/'sim/camera_framing_20260923/recommended_camera.json').read_bytes()


def module():
    assert importlib.util.find_spec('sim.ausc_policy.render_settling_probe'), 'settling diagnostic is missing'
    return importlib.import_module('sim.ausc_policy.render_settling_probe')


def config():
    return json.loads((ROOT/'configs/ausc_policy/m2_render_settling_retainer_v1.json').read_text())


class Env:
    def __init__(self):
        self.rows=[];self.stopped=False;self.stop_error=False
        self.world=SimpleNamespace(current_time=7.,stop=self.stop,step=self.forbidden,reset=self.forbidden)
        self.value=dict(time_s=0.,q=np.zeros(7),dq=np.zeros(7),compression_m=.001)
    def state(self):return deepcopy(self.value)
    def stop(self):
        self.stopped=True
        if self.stop_error:raise RuntimeError('world stop failed')
    def forbidden(self,*args,**kwargs):raise AssertionError('physics/controller/write operation forbidden')
    step=reset=save=set_joint_positions=forbidden


class Camera:
    def __init__(self,env):
        self.env=env;self.count=0;self.closed=False;self.hook=None;self.close_error=False
        approved=json.loads(CAMERA)
        self.actual=dict(camera_path='/World/RobotCam',world_from_camera_usd=approved['world_from_camera_usd'],
            intrinsics={key:approved['intrinsics'][key]/3 for key in ('fx','fy','cx','cy')},
            width=640,height=360,meters_per_unit=1.,projection='perspective',time_varying=False)
    def snapshot(self):return deepcopy(self.actual)
    _verify_camera=snapshot
    def read(self):
        self.count+=1
        if self.hook:self.hook(self.count)
        depth=np.full((360,640),.5,np.float32);depth[0,0]=np.nan
        return dict(rgb=np.full((360,640,3),self.count,np.uint8),depth_m=depth,
            depth_valid=np.isfinite(depth)&(depth>0),actual_camera=self.snapshot(),
            measured_state=self.env.state(),world_time_s=self.env.world.current_time)
    def close(self):
        self.closed=True
        return dict(completed=not self.close_error,cleanup_complete=not self.close_error,
                    failures=[dict(stage='detach',error='failed')] if self.close_error else [])


def run(tmp_path,*,env=None,camera=None,initialize=None,**kwargs):
    env=env or Env();camera=camera or Camera(env)
    def init(register):
        register(env)
        return dict(scope='injected CPU initialization',native_execution=False)
    options=dict(initialize=initialize or init,camera_factory=lambda current:camera,
        camera_bytes=CAMERA,provenance=dict(execution_kind='cpu_synthetic_test',source_hashes={}),
        settings_reader=lambda key:None,limit_bytes=8_000_000)
    options.update(kwargs)
    result=module().run_diagnostic(tmp_path,**options)
    return result,env,camera


def test_exact_128_zero_delta_reads_five_panels_and_bounded_json_chart_only(tmp_path):
    result,env,camera=run(tmp_path)
    assert result['completed'] and camera.count==128 and camera.closed and env.stopped
    assert result['observation_count']==128 and result['panel_render_numbers']==[1,8,32,64,128]
    assert result['measurement']['initial']['rows']==result['measurement']['final']['rows']==0
    assert result['measurement']['initial']['time_s']==result['measurement']['final']['time_s']==0.
    assert result['measurement']['initial']['world_time_s']==result['measurement']['final']['world_time_s']==7.
    assert result['render_protocol']==dict(planned_calls=128,rt_subframes=2,delta_time_s=0.,physics_frames=False)
    assert result['read_attempts']==128
    assert [row['rgb_mean'] for row in result['observations']]==list(range(1,129))
    assert result['observations'][0]['adjacent_rgb_mean_abs_difference'] is None
    assert result['observations'][-1]['adjacent_rgb_mean_abs_difference']==1.
    assert result['observations'][-1]['joint_valid_depth_mean_abs_difference_m']==0.
    assert result['initialization']['native_execution'] is False
    assert not any(result[key] for key in ('native_qualification','physical_qualification','training_eligible','settling_qualified'))
    assert {p.name for p in tmp_path.iterdir()}=={'render_settling_panels.png','render_settling_result.json','COMPLETE.json'}
    assert result['diagnostic_total_bytes']==sum(p.stat().st_size for p in tmp_path.iterdir())
    assert result['diagnostic_total_bytes']<=8_000_000
    assert all(row['status']=='missing' for row in result['renderer_settings'].values())
    assert 'disabled' not in json.dumps(result['renderer_settings'])
    for name,record in result['artifacts'].items():
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==record['sha256']


@pytest.mark.parametrize('change',['episode','world','q','dq','compression','rows','camera','shape'])
def test_fixed_anchor_rejects_cumulative_changes_and_keeps_partial_evidence(tmp_path,change):
    env=Env();camera=Camera(env)
    def mutate(number):
        if change=='episode':env.value['time_s']+=6e-9
        elif change=='world':env.world.current_time+=6e-9
        elif change in ('q','dq'):env.value[change][0]+=6e-9
        elif change=='compression':env.value['compression_m']+=1e-14
        elif change=='rows':env.rows.append({})
        elif change=='camera':camera.actual['intrinsics']['fx']+=.001
    camera.hook=mutate
    if change=='shape':
        original=camera.read
        def malformed():
            row=original();row['rgb']=row['rgb'][:100];return row
        camera.read=malformed
    result,_,_=run(tmp_path,env=env,camera=camera)
    assert not result['completed'] and result['errors'] and camera.count<128
    assert camera.closed and env.stopped and (tmp_path/'FAILED.json').exists()
    assert not (tmp_path/'COMPLETE.json').exists()


def test_absent_depth_overlap_is_explicit_not_zero_difference(tmp_path):
    env=Env();camera=Camera(env);original=camera.read
    def invalid_depth():
        row=original();row['depth_m'][:]=np.nan;row['depth_valid'][:]=False;return row
    camera.read=invalid_depth
    result,_,_=run(tmp_path,env=env,camera=camera)
    assert result['completed']
    assert result['observations'][-1]['joint_valid_depth_fraction']==0.
    assert result['observations'][-1]['joint_valid_depth_mean_abs_difference_m'] is None


@pytest.mark.parametrize('failure',['read','camera_close','world_stop','chart_save','report_save'])
def test_operation_output_and_cleanup_failures_never_publish_complete(tmp_path,failure,monkeypatch):
    env=Env();camera=Camera(env)
    if failure=='read':
        def missing(number):
            if number==9:raise RuntimeError('original missing frame')
        camera.hook=missing
    elif failure=='camera_close':camera.close_error=True
    elif failure=='world_stop':env.stop_error=True
    else:
        method='_save_chart' if failure=='chart_save' else '_save_json'
        original=getattr(module(),method)
        def fail(path,*args,**kwargs):
            if failure=='chart_save' or path.name=='render_settling_result.json':raise OSError('original disk error')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(module(),method,fail)
    result,_,_=run(tmp_path,env=env,camera=camera)
    assert camera.closed and env.stopped and not result['completed'] and result['errors']
    assert (tmp_path/'FAILED.json').exists() and not (tmp_path/'COMPLETE.json').exists()
    if failure=='read':
        assert result['observation_count']==8 and result['panel_render_numbers']==[1,8]
        assert result['read_attempts']==9


def test_partial_camera_constructor_cleanup_receipt_is_not_lost(tmp_path):
    env=Env()
    def camera_factory(current):
        error=RuntimeError('original camera setup error')
        error.training_camera_cleanup=dict(completed=False,cleanup_complete=False,
                                          failures=[dict(stage='detach',error='still attached')])
        raise error
    result,_,_=run(tmp_path,env=env,camera_factory=camera_factory)
    assert not result['completed'] and env.stopped
    assert result['cleanup']['camera']['cleanup_complete'] is False
    assert result['errors'][0]['stage']=='camera_setup'


@pytest.mark.parametrize('created_world',[False,True])
def test_partial_environment_initialization_retains_error_and_stops_owned_world(tmp_path,created_world):
    env=Env()
    partial=env if created_world else SimpleNamespace()
    def initialize(register):
        register(partial);raise RuntimeError('original environment initialization error')
    result,_,_=run(tmp_path,initialize=initialize)
    assert not result['completed'] and result['observation_count']==0
    assert result['errors'][0]['stage']=='initialization'
    assert result['cleanup']['world']['status']==('stopped' if created_world else 'not_created')
    assert env.stopped is created_world


@pytest.mark.parametrize('extra',[
    {'m2_sensor_probe':{}},{'patient_asset':{}},{'patient_preview':True},{'contact_probe':{}},
    {'fresh_expert':{'plans':[]}},{'fresh_expert':{'raw_multimodal_capture':{'enabled':False}}},
])
def test_mode_conflicts_and_controller_keys_are_rejected_before_runtime(extra):
    cfg=config();cfg.update(extra)
    with pytest.raises(ValueError):module().validate_mode(cfg)


def test_config_derivation_preserves_approved_geometry_physics_preload_and_no_plan():
    cfg=config();source=ROOT/'configs/ausc_policy/fresh_expert_retainer_r3_single_preload_v3.json'
    original=json.loads(source.read_text())
    assert module().validate_mode(cfg)
    for key in ('geometry_id','geometry_source','geometry_source_sha256','physics','component_priors'):
        assert cfg[key]==original[key]
    assert set(cfg['fresh_expert'])=={'passive_preload_N','passive_preload_model'}
    assert cfg['derived_from']['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert cfg['time_limit_seconds']>0 and cfg['time_limit_seconds']<=1800
    assert module().validate_mode({'fresh_expert':{'plans':[]}}) is None


def test_geometry_dispatch_preflight_precedes_simulation_app_and_default_remains(tmp_path,monkeypatch):
    import sys
    from sim.ausc_policy import geometry_qualification as entry
    monkeypatch.setattr(entry.sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    cfg=config();cfg['contact_probe']={};path=tmp_path/'conflict.json';path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys,'argv',['probe','--config',str(path),'--output',str(tmp_path/'out')])
    class ReachedRuntime(Exception):pass
    def forbidden(*args,**kwargs):raise ReachedRuntime()
    monkeypatch.setitem(sys.modules,'isaacsim',SimpleNamespace(SimulationApp=forbidden))
    with pytest.raises(ValueError):entry.main()
    raw=b'CPU source';source=tmp_path/'source.usdc';source.write_bytes(raw)
    path.write_text(json.dumps(dict(geometry_source=str(source),geometry_source_sha256=hashlib.sha256(raw).hexdigest())))
    monkeypatch.setattr(entry,'require_current',lambda value:None)
    with pytest.raises(ReachedRuntime):entry.main()


def test_mutated_approved_camera_bytes_are_rejected_without_measurement(tmp_path):
    result,env,camera=run(tmp_path,camera_bytes=CAMERA+b' ')
    assert not result['completed'] and camera.count==0
    assert result['errors'][0]['stage']=='input_validation'
    assert result['cleanup']['world']['status']=='not_created'


def test_noisy_chart_budget_failure_preserves_bounded_partial_report(tmp_path):
    env=Env();camera=Camera(env);original=camera.read;rng=np.random.default_rng(5)
    def noisy():
        row=original();row['rgb']=rng.integers(0,256,(360,640,3),dtype=np.uint8);return row
    camera.read=noisy
    result,_,_=run(tmp_path,env=env,camera=camera,limit_bytes=250_000)
    assert not result['completed'] and any('budget' in error['error'] for error in result['errors'])
    assert (tmp_path/'FAILED.json').exists() and not (tmp_path/'COMPLETE.json').exists()
    assert sum(p.stat().st_size for p in tmp_path.iterdir())<=250_000


def test_short_chart_write_is_not_successful_output(tmp_path,monkeypatch):
    original=Path.open
    class ShortWrite:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):self.stream.close()
        def write(self,payload):self.stream.write(payload[:10]);return 10
    def open_path(path,*args,**kwargs):
        stream=original(path,*args,**kwargs)
        if path.name=='render_settling_panels.png' and args and args[0]=='xb':return ShortWrite(stream)
        return stream
    monkeypatch.setattr(Path,'open',open_path)
    result,env,camera=run(tmp_path)
    assert not result['completed'] and env.stopped and camera.closed
    assert (tmp_path/'FAILED.json').exists() and not (tmp_path/'COMPLETE.json').exists()


def test_read_only_renderer_settings_distinguish_missing_unavailable_and_false(tmp_path):
    def read(key):
        if key.endswith('autoExposure/enabled'):return False
        if key.endswith('rendermode'):raise RuntimeError('setting unavailable')
        return None
    result,_,_=run(tmp_path,settings_reader=read)
    assert result['completed']
    assert result['renderer_settings']['/rtx/post/autoExposure/enabled']==dict(status='available',value=False)
    assert result['renderer_settings']['/rtx/rendermode']['status']=='unavailable'


@pytest.mark.parametrize('changed',['physics','component_priors','camera'])
def test_preflight_rejects_changed_inputs_before_native_initialization(tmp_path,monkeypatch,changed):
    cfg=config();camera_path=tmp_path/'camera.json';camera_path.write_bytes(CAMERA)
    cfg['m2_render_settling_probe']['camera_file']=str(camera_path)
    if changed=='physics':cfg['physics']['position_iterations']=1
    elif changed=='component_priors':cfg['component_priors']['parts'][0]['mass_kg_estimate']*=2
    else:camera_path.write_bytes(CAMERA+b' ')
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    original=module()._sha
    monkeypatch.setattr(module(),'_sha',lambda path:cfg['geometry_source_sha256'] if str(path)==cfg['geometry_source'] else original(path))
    with pytest.raises(ValueError):module().preflight(cfg,path)
