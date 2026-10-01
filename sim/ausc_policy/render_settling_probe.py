"""Opt-in, owned-Linux render settling diagnostic; no motion or warmup policy.

Initialization may reset/start the FreshContactEnv world and is reported apart
from measurement. Measurement performs exactly 128 GuardedRGBDStream.read calls
(NativeBackend rt_subframes=2, delta_time=0), not 128 physics frames or 20 Hz
simulation time. Full-FOV statistics and panels 1/8/32/64/128 are diagnostic
evidence only. No ROI, tracker, controller, raw frame persistence or env.save.

run_diagnostic injects CPU protocol fixtures explicitly. Its initialize(register)
callback registers partial environment ownership before construction can fail.
Only the native run_probe wrapper imports FreshContactEnv/renderer settings.
"""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from . import m2_capture
from .geometry_epoch import require_current
from .passive_preload import passive_preload_settings
from .training_camera import GuardedRGBDStream

PANELS=(1,8,32,64,128)
READS=128
MAX_DIAGNOSTIC_BYTES=8_000_000
JSON_RESERVE=200_000
QUALIFICATIONS=dict(native_qualification=False,physical_qualification=False,
                    training_eligible=False,settling_qualified=False)
SETTING_KEYS=('/rtx/rendermode','/rtx/post/autoExposure/enabled','/rtx/post/tonemap/op',
    '/rtx/pathtracing/spp','/rtx/pathtracing/totalSpp','/rtx/pathtracing/optixDenoiser/enabled',
    '/rtx/post/taa/enabled')


def _bytes(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _error(stage,error):
    try:message=str(error)
    except BaseException:message='<exception message unavailable>'
    return dict(stage=stage,error=(type(error).__name__+': '+message)[:1024])


def validate_mode(config):
    """Absence preserves legacy dispatch; opt-in is strict before Isaac import."""
    if 'm2_render_settling_probe' not in config:return None
    require_current(config)
    for key in ('patient_asset','patient_preview','m2_sensor_probe','contact_probe','contact_test'):
        if key in config:raise ValueError('Render settling is mutually exclusive with '+key)
    fresh=config.get('fresh_expert')
    if type(fresh) is not dict or set(fresh)!={'passive_preload_N','passive_preload_model'}:
        raise ValueError('Render settling permits only the two single-preload settings; no expert configuration')
    if passive_preload_settings(fresh) is None:raise ValueError('Single explicit preload required')
    mode=config['m2_render_settling_probe']
    required={'enabled','render_calls','panel_render_numbers','camera_file','camera_sha256','diagnostic_limit_bytes'}
    if (type(mode) is not dict or set(mode)!=required or mode['enabled'] is not True or
            type(mode['render_calls']) is not int or mode['render_calls']!=READS or
            type(mode['panel_render_numbers']) is not list or
            any(type(number) is not int for number in mode['panel_render_numbers']) or
            mode['panel_render_numbers']!=list(PANELS) or
            mode['camera_sha256']!=m2_capture.APPROVED_CAMERA_SHA256 or
            not isinstance(mode['camera_file'],str) or not mode['camera_file'] or
            type(mode['diagnostic_limit_bytes']) is not int or
            not JSON_RESERVE < mode['diagnostic_limit_bytes']<=MAX_DIAGNOSTIC_BYTES):
        raise ValueError('Render settling requires the fixed bounded 128-call approved-camera profile')
    duration=m2_capture._scalar(config.get('time_limit_seconds'),'diagnostic time limit')
    if not 0<duration<=1800:raise ValueError('Render settling needs a finite diagnostic time limit <=1800s')
    return deepcopy(mode)


def preflight(config,config_path):
    """Verify bound input bytes and inspect source files before SimulationApp."""
    mode=validate_mode(config)
    if mode is None:return None
    camera=Path(mode['camera_file']).read_bytes()
    if hashlib.sha256(camera).hexdigest()!=m2_capture.APPROVED_CAMERA_SHA256:
        raise ValueError('Approved camera bytes changed')
    if _sha(config['geometry_source'])!=config['geometry_source_sha256']:raise ValueError('Geometry source changed')
    config_bytes=Path(config_path).read_bytes()
    if json.loads(config_bytes)!=config:raise ValueError('Configuration changed during preflight')
    root=Path(__file__).resolve().parents[2]
    origin=config['derived_from']
    origin_path=root/origin['file']
    if _sha(origin_path)!=origin['sha256']:raise ValueError('Successful source configuration changed')
    original=json.loads(origin_path.read_text())
    for key in ('geometry_id','geometry_source','geometry_source_sha256','physics','component_priors'):
        if config[key]!=original[key]:raise ValueError('Diagnostic changed source configuration '+key)
    paths=[Path(__file__).with_name(name+'.py') for name in
           ('render_settling_probe','training_camera','m2_capture','geometry_qualification',
            'geometry_epoch','passive_preload','approved_camera','fresh_expert')]
    paths.append(root/'sim/auscultation_expert_layout_v1/env.py')
    sources={str(path):_sha(path) for path in paths}
    return dict(camera_bytes=camera,provenance=dict(execution_kind='isaac_native',
        config_file=str(Path(config_path).resolve()),config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        geometry_source=config['geometry_source'],geometry_source_sha256=config['geometry_source_sha256'],
        camera_sha256=mode['camera_sha256'],derived_from=deepcopy(origin),source_hashes=sources,
        source_binding_scope='listed diagnostic/camera/environment source files; not a full dependency attestation',
        python_executable=sys.executable,python_version=sys.version,source_inspection_only=False))


def _physical(env):
    state=m2_capture._state(env.state())
    return state,m2_capture._scalar(env.world.current_time,'world clock'),len(env.rows)


def _physical_json(value):
    state,clock,rows=value
    return dict(GuardedRGBDStream._state_dict(state,json_compatible=True),world_time_s=clock,rows=rows)


def _unchanged(anchor,current,label):
    m2_capture._same_state(anchor[0],current[0],label)
    if abs(anchor[1]-current[1])>1e-8 or anchor[2]!=current[2]:
        raise ValueError(label+' changed world clock or physics row count')


def _camera_unchanged(anchor,current):
    for key in ('camera_path','width','height','meters_per_unit','projection','time_varying'):
        if current.get(key,False if key=='time_varying' else None)!=anchor.get(key,False if key=='time_varying' else None):
            raise ValueError('Measurement changed camera '+key)
    if not np.allclose(anchor['world_from_camera_usd'],current['world_from_camera_usd'],atol=1e-7,rtol=0):
        raise ValueError('Measurement changed camera transform')
    for key,dimension in (('fx','width'),('fy','height'),('cx','width'),('cy','height')):
        if abs(m2_capture._scalar(current['intrinsics'][key],'camera intrinsics')/current[dimension]-
               anchor['intrinsics'][key]/anchor[dimension])>1e-7:
            raise ValueError('Measurement changed camera intrinsics')


def _settings(reader):
    result={}
    for key in SETTING_KEYS:
        try:
            value=reader(key)
            if value is None:result[key]=dict(status='missing',value=None)
            elif type(value) in (str,bool,int,float) and len(_bytes(value))<=1024:
                result[key]=dict(status='available',value=value)
            else:result[key]=dict(status='unavailable',reason='unsupported or unbounded setting value')
        except BaseException as error:result[key]=dict(status='unavailable',reason=_error('setting',error)['error'])
    return result


def _summary(frame,previous,number):
    rgb,depth,valid=frame['rgb'],frame['depth_m'],frame['depth_valid']
    if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape not in ((360,640,3),(360,640,4)) or
            not isinstance(depth,np.ndarray) or depth.shape!=(360,640) or depth.dtype.kind!='f' or
            not isinstance(valid,np.ndarray) or valid.shape!=depth.shape or valid.dtype!=bool or
            not np.array_equal(valid,np.isfinite(depth)&(depth>0))):
        raise ValueError('Missing or invalid native frame/validity')
    if previous is not None and rgb.shape!=previous['rgb'].shape:raise ValueError('Native RGB shape changed')
    row=GuardedRGBDStream._rgb_statistics(rgb,None if previous is None else previous['rgb'])
    row.update(render_number=number,time_s=float(frame['measured_state']['time_s']),
        world_time_s=float(frame['world_time_s']),depth_valid_fraction=float(valid.mean()),
        actual_camera_sha256=hashlib.sha256(_bytes(frame['actual_camera'])).hexdigest(),
        joint_valid_depth_fraction=None,joint_valid_depth_mean_abs_difference_m=None,
        joint_valid_depth_max_abs_difference_m=None)
    if previous is not None:
        overlap_count=0;total=0.;largest=0.
        for index in range(depth.shape[0]):
            overlap=valid[index]&previous['depth_valid'][index]
            count=int(overlap.sum());overlap_count+=count
            if count:
                delta=np.abs(depth[index,overlap].astype(float)-previous['depth_m'][index,overlap].astype(float))
                total+=float(delta.sum());largest=max(largest,float(delta.max()))
        row['joint_valid_depth_fraction']=overlap_count/depth.size
        if overlap_count:
            row['joint_valid_depth_mean_abs_difference_m']=total/overlap_count
            row['joint_valid_depth_max_abs_difference_m']=largest
    return row


def _save_json(path,value):
    payload=_bytes(value);temporary=path.with_suffix(path.suffix+'.partial')
    if path.exists():raise FileExistsError(str(path))
    try:
        with temporary.open('xb') as stream:
            if stream.write(payload)!=len(payload):raise OSError('Short diagnostic JSON write')
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()


def _save_chart(path,panels,limit_bytes):
    chart=Image.new('RGB',(640*len(PANELS),392),'white');draw=ImageDraw.Draw(chart)
    for index,number in enumerate(PANELS):
        draw.text((index*640+8,8),f'Render call {number} - zero physics delta',fill='black')
        if number in panels:chart.paste(Image.fromarray(panels[number][:,:,:3]),(index*640,32))
        else:draw.text((index*640+8,60),'Not captured',fill='black')
    encoded=io.BytesIO();chart.save(encoded,format='PNG');payload=encoded.getvalue()
    if len(payload)>limit_bytes:raise ValueError('Diagnostic chart exceeds byte budget')
    with path.open('xb') as stream:
        if stream.write(payload)!=len(payload):raise OSError('Short diagnostic chart write')


def _persist(output,result,panels,limit):
    result['artifacts']={}
    chart=output/'render_settling_panels.png'
    if panels:
        try:
            _save_chart(chart,panels,limit-JSON_RESERVE)
            result['artifacts'][chart.name]=dict(bytes=chart.stat().st_size,sha256=_sha(chart))
        except BaseException as error:result['errors'].append(_error('chart_save',error))
    result['completed']=bool(result['completed'] and not result['errors'])
    result['diagnostic_total_bytes']=0
    report=output/'render_settling_result.json'
    try:
        chart_size=chart.stat().st_size if chart.exists() else 0
        while True:
            raw=_bytes(result)
            marker=dict(completed=result['completed'],report=dict(file=report.name,bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest()),diagnostic_total_bytes=result['diagnostic_total_bytes'],**QUALIFICATIONS)
            total=chart_size+len(raw)+len(_bytes(marker))
            if total==result['diagnostic_total_bytes']:break
            result['diagnostic_total_bytes']=total
        if total>limit:raise ValueError('Diagnostic JSON/chart/marker exceeds byte budget')
        _save_json(report,result)
        _save_json(output/('COMPLETE.json' if result['completed'] else 'FAILED.json'),marker)
    except BaseException as error:
        result['completed']=False;result['errors'].append(_error('result_save',error))
        fallback=dict(result,diagnostic_total_bytes=None)
        try:
            used=sum(path.stat().st_size for path in (chart,report) if path.exists())
            if used+len(_bytes(fallback))>limit:raise ValueError('Partial failure report exceeds diagnostic byte budget')
            _save_json(output/'FAILED.json',fallback)
        except BaseException as failure:result['errors'].append(_error('failure_report_save',failure))
    return result


def run_diagnostic(output, *, initialize,camera_factory,camera_bytes,provenance,settings_reader,limit_bytes):
    """Bounded core with explicit injected initialization/camera protocols for CPU tests."""
    output=Path(output)
    if type(limit_bytes) is not int or not JSON_RESERVE<limit_bytes<=MAX_DIAGNOSTIC_BYTES:
        raise ValueError('Invalid bounded diagnostic byte limit')
    env=camera=None;panels={};previous=None;anchor=None;initial_camera=None;stage='input_validation'
    result=dict(completed=False,observation_count=0,read_attempts=0,observations=[],panel_render_numbers=[],errors=[],
        initialization=dict(completed=False),measurement={},renderer_settings={},cleanup={},
        provenance=deepcopy(provenance),camera_sha256=hashlib.sha256(camera_bytes).hexdigest(),
        render_protocol=dict(planned_calls=READS,rt_subframes=2,delta_time_s=0.,physics_frames=False),
        statistic_scope='full native FOV RGB excluding alpha and jointly valid optical-Z depth; no ROI',
        renderer_settings_scope='read-only values; missing/unavailable does not establish a disabled feature',
        **QUALIFICATIONS)
    def register(value):
        nonlocal env
        if env is not None:raise ValueError('Only one environment may be owned')
        env=value
    began=time.perf_counter()
    try:
        if (not isinstance(camera_bytes,bytes) or
                hashlib.sha256(camera_bytes).hexdigest()!=m2_capture.APPROVED_CAMERA_SHA256):
            raise ValueError('Approved camera bytes changed')
        stage='initialization'
        initialization=initialize(register)
        result['initialization']=dict(initialization,completed=True,wall_time_s=time.perf_counter()-began)
        if env is None:raise ValueError('Initialization did not register its environment')
        anchor=_physical(env)
        if np.max(np.abs(anchor[0][1]))>.03:raise ValueError('Diagnostic must start at the original home state')
        result['measurement']['initial']=_physical_json(anchor)
        stage='settings';result['renderer_settings']=_settings(settings_reader)
        _unchanged(anchor,_physical(env),'renderer settings reads')
        stage='camera_setup';camera=camera_factory(env)
        _unchanged(anchor,_physical(env),'camera setup')
        initial_camera=camera._verify_camera();result['actual_camera']=deepcopy(initial_camera)
        stage='measurement';measurement_start=time.perf_counter()
        for number in range(1,READS+1):
            _unchanged(anchor,_physical(env),'before camera read')
            result['read_attempts']+=1
            frame=camera.read()
            _unchanged(anchor,_physical(env),'after camera read')
            observed=(m2_capture._state(frame['measured_state']),m2_capture._scalar(frame['world_time_s'],'frame world clock'),len(env.rows))
            _unchanged(anchor,observed,'returned observation')
            _camera_unchanged(initial_camera,frame['actual_camera'])
            row=_summary(frame,previous,number)
            result['observations'].append(row);result['observation_count']=number
            if number in PANELS:panels[number]=frame['rgb'].copy()
            previous=frame
        stage='final_measurement'
        _camera_unchanged(initial_camera,camera._verify_camera())
        final=_physical(env);_unchanged(anchor,final,'final measurement')
        result['measurement'].update(final=_physical_json(final),unchanged=True,
            wall_time_s=time.perf_counter()-measurement_start)
        result['completed']=True
    except BaseException as error:
        result['errors'].append(_error(stage,error))
        if camera is None and hasattr(error,'training_camera_cleanup'):
            result['cleanup']['camera']=deepcopy(error.training_camera_cleanup)
        if stage=='initialization':result['initialization']['wall_time_s']=time.perf_counter()-began
    finally:
        if camera is not None:
            try:
                receipt=camera.close();result['cleanup']['camera']=receipt
                if receipt.get('cleanup_complete') is not True or receipt.get('completed') is not True:
                    raise RuntimeError('Camera cleanup or acquisition did not complete')
            except BaseException as error:result['errors'].append(_error('camera_close',error))
        world=None if env is None else getattr(env,'world',None)
        if world is None:result['cleanup']['world']=dict(status='not_created')
        else:
            try:world.stop();result['cleanup']['world']=dict(status='stopped')
            except BaseException as error:
                result['cleanup']['world']=dict(status='release_failed')
                result['errors'].append(_error('world_stop',error))
    result['panel_render_numbers']=sorted(panels)
    for path,expected in provenance.get('source_hashes',{}).items():
        try:
            if _sha(path)!=expected:raise ValueError('Bound source changed during diagnostic')
        except BaseException as error:result['errors'].append(_error('source_recheck',error))
    result['completed']=bool(result['completed'] and not result['errors'])
    return _persist(output,result,panels,limit_bytes)


def run_probe(app,stage,output,config,mass, *, checked_inputs):
    """Owned entrypoint; initialization performs no controller or expert plan."""
    if sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1':
        raise RuntimeError('Render settling requires the owned Linux runner')
    mode=validate_mode(config)
    from .fresh_expert import FreshContactEnv
    from .passive_preload import verify_preload_readback
    import carb
    output=Path(output)
    def initialize(register):
        mass_kg=m2_capture._scalar(mass['mass_kg'],'moving mass')
        if mass_kg<=0:raise ValueError('Invalid moving mass')
        receipt=json.loads((output/'passive_preload_authoring.json').read_text())
        before=verify_preload_readback(stage,config['fresh_expert'],receipt)
        def stage_factory(destination):stage.GetRootLayer().Export(str(destination));return stage
        env=FreshContactEnv.__new__(FreshContactEnv);register(env)
        env.__init__(output/'render_settling_initialization',stage_factory=stage_factory,
                     contact_offset_m=.132,moving_mass_kg=mass_kg)
        after=verify_preload_readback(env.stage,config['fresh_expert'],receipt,robot=env.robot,slider=env.slider)
        return dict(native_execution=True,scope='FreshContactEnv initialization, separate from zero-delta measurement',
            preload_before=before,preload_after=after,explicit_effort_applied_during_measurement=False)
    return run_diagnostic(output,initialize=initialize,
        camera_factory=lambda env:GuardedRGBDStream(env,camera_bytes=checked_inputs['camera_bytes'],
                                                   camera_sha256=mode['camera_sha256']),
        camera_bytes=checked_inputs['camera_bytes'],provenance=checked_inputs['provenance'],
        settings_reader=carb.settings.get_settings().get,limit_bytes=mode['diagnostic_limit_bytes'])
