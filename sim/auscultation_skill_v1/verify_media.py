"""Independent, read-only full-ten video and saved-pose RGB-D verification.

Only the explicitly requested new receipt is written. No renderer or simulator
is initialized. Media fidelity does not replace the independent physical audit.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .config import PROJECT, TARGETS
from .quality import validate_episode_provenance
from sim.upper_pair_posture_20260924.verify_video import decode_video

CAMERA = 'sim/camera_framing_20260923/recommended_camera.json'
LAYOUT = dict(posture=[0,0,1280,720], robotcam=[1280,0,640,360], metadata=[1280,360,640,360])
RENDER_SCHEMA = 'auscultation_skill_recorded_full10_review_v1'
RGBD_SCHEMA = 'auscultation_saved_pose_rgbd_v1'
PRIVATE_RENDERER = 'sim/auscultation_skill_v1/private_renderer.py'
TEARDOWN = 'sim/auscultation_skill_v1/media_teardown.py'
POST_CLOSE = dict(apps_created=1, close_returned=True,
    completion_stage='post_app_close_pre_process_exit', exit_policy='controlled_os_exit_after_close')


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def read_json_text(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value): raise ValueError('nonfinite JSON: '+value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(path): return read_json_text(Path(path).read_text())


def child(root, relative):
    root = Path(root).resolve()
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and '..' not in Path(relative).parts, 'unsafe relative path')
    result = (root/relative).resolve()
    require(root in result.parents, 'unsafe resolved path')
    return result


def number(actual, expected, label, tolerance=1e-7):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and abs(actual-float(expected)) <= tolerance, 'invalid '+label)


def integer(actual, expected, label):
    require(type(actual) is int and actual == int(expected), 'invalid '+label)


def verify_sources(hashes, project=PROJECT):
    require(isinstance(hashes, dict) and hashes, 'missing source hashes')
    result = {}
    for relative, digest in hashes.items():
        path = child(project, relative)
        require(isinstance(digest, str) and len(digest)==64
                and all(x in '0123456789abcdef' for x in digest), 'invalid source digest')
        require(path.is_file() and sha(path)==digest, 'source hash mismatch: '+relative)
        result[relative] = path
    require(len(set(result.values()))==len(result), 'duplicate source aliases')
    return result


def verify_process_completion(directory, data, *, guard_path, project=PROJECT, module):
    """Bind post-close evidence to the actual exit of the owned invocation.

    The guard is a separate, final parent record. A manifest written before
    process exit cannot establish that exit. The launcher hash records the
    current owned script; media source hashes independently bind the renderer.
    """
    directory, project = Path(directory).resolve(), Path(project).resolve()
    require(module in ('sim.auscultation_skill_v1.render', 'sim.auscultation_skill_v1.rgbd'),
            'unsupported media process module')
    kind = module.rsplit('.', 1)[1]
    lifecycle = data.get('render_lifecycle' if kind=='render' else 'capture_lifecycle')
    require(isinstance(lifecycle, dict) and type(lifecycle.get('apps_created')) is int
            and lifecycle.get('close_returned') is True and lifecycle == POST_CLOSE,
            'missing or invalid post-close media lifecycle')
    teardown = data.get('resource_teardown')
    expected_teardown = dict(policy='capture_scope_end_then_stage_close_before_normal_close_v1',
        replicator_wait_completed=True, annotators_detached=2,
        render_products_destroyed=2 if kind=='render' else 1, close_returned=True,
        capture_scope_ended=True, gc_before_stage_close=True, pre_stage_close_updates=5,
        explicit_stage_close_checked=True, stage_absent_after_drain=True,
        post_stage_close_updates=5, gc_before_app_close=True, errors=[])
    require(isinstance(teardown, dict) and teardown == expected_teardown
            and all(type(teardown.get(key)) is type(value) for key, value in expected_teardown.items()),
            'incomplete scoped media resource teardown')
    require(guard_path is not None, 'explicit final media guard is required')
    original_guard = Path(guard_path)
    require(not original_guard.is_symlink(), 'media guard must not be a symlink')
    guard_path = original_guard.resolve()
    episode = child(project, data.get('episode'))
    require(guard_path != directory and directory not in guard_path.parents
            and guard_path != episode and episode not in guard_path.parents,
            'media guard must be outside immutable media and episode')
    require(guard_path.is_file(), 'final media guard is missing')
    guard_hash = sha(guard_path); guard = read(guard_path)
    require(isinstance(guard, dict), 'invalid media guard')
    require(type(guard.get('returncode')) is int and guard['returncode']==0
            and type(guard.get('guard_returncode')) is int and guard['guard_returncode']==0,
            'media process or guard did not exit successfully')
    require(guard.get('timed_out') is False and guard.get('aborted_low_memory', False) is False
            and 'interrupted_signal' not in guard, 'media guard timed out, aborted or was interrupted')
    require(type(guard.get('owned_process_group')) is int and guard['owned_process_group']>0,
            'media guard lacks an owned process group')
    for field in ('minimum_required_gib', 'minimum_observed_gib'):
        value = guard.get(field)
        require(type(value) in (int, float) and math.isfinite(value) and value>=8.,
                'invalid media guard memory evidence')
    require(guard['minimum_observed_gib']>=guard['minimum_required_gib'],
            'media guard violated its memory reserve')
    launcher = project.parent/'run_isaac_linbox.sh'
    require(launcher.is_file() and not launcher.is_symlink(), 'owned media launcher missing or replaced')
    command = guard.get('command')
    require(isinstance(command, list) and all(isinstance(arg, str) and arg for arg in command)
            and len(command)>=8 and command[:4]==['bash', str(launcher), '-m', module],
            'media guard command is not the owned launcher/module')
    arguments = command[4:]; allowed = {'--episode', '--output'} | ({'--label'} if kind=='render' else set())
    require(len(arguments)%2==0, 'invalid media command arguments')
    options = {}
    for flag, value in zip(arguments[::2], arguments[1::2]):
        require(flag in allowed and flag not in options, 'unknown or duplicate media command argument')
        options[flag] = value
    require({'--episode', '--output'}<=set(options), 'missing media command episode/output')
    def resolved(value):
        path = Path(value)
        return (path if path.is_absolute() else project/path).resolve()
    require(resolved(options['--episode'])==episode and resolved(options['--output'])==directory,
            'media guard episode/output differs from verified media')
    launcher_hash = sha(launcher)
    require(sha(guard_path)==guard_hash, 'media guard changed during verification')
    return dict(status='passed', guard_path=str(guard_path), guard_sha256=guard_hash,
        launcher_path=str(launcher), launcher_sha256=launcher_hash, command=command,
        module=module, episode=str(episode), output=str(directory),
        owned_process_group=guard['owned_process_group'], returncode=0, guard_returncode=0,
        post_close_lifecycle_verified=True, owned_resources_released=True,
        command_identity_verified=True)


def validate_episode_rows(outcome, rows, indices, poses, body_paths):
    require(outcome.get('requested_points')==list(TARGETS), 'full ten-point order required')
    require(outcome.get('success') is True, 'incomplete source episode')
    count = len(rows)
    integer(outcome.get('physics_steps'), count, 'physics row count')
    require(count>=120, 'insufficient source rows')
    times = np.asarray([r['time_s'] for r in rows], float)
    require(np.isfinite(times).all() and np.allclose(times, np.arange(1,count+1)/240,
            atol=1e-7, rtol=0), 'source clock shifted or discontinuous')
    points = outcome.get('points', [])
    require([p.get('target') for p in points]==list(TARGETS), 'point inventory/order mismatch')
    number(outcome.get('phase_config', {}).get('hold_duration_s'), 3., 'HOLD configuration')
    end = 0
    for point in points:
        integer(point.get('start_physics_row'), end, 'point start')
        next_end = point.get('end_physics_row')
        require(type(next_end) is int and end<next_end<=count and point.get('success') is True,
                'invalid completed point interval')
        segment = rows[end:next_end]
        require(all(r['target']==point['target'] for r in segment), 'point trace coverage mismatch')
        require(any(r['phase']=='HOLD' for r in segment), 'missing visible HOLD state')
        # HOLD may contain extra settling rows. Force-qualified continuous 3 s is
        # checked by the physical audit, never inferred from an exact row count.
        end = next_end
    require(end==count, 'incomplete point coverage')
    require(rows[0]['target']==TARGETS[0] and rows[0]['subphase']=='initial_home', 'initial home missing')
    require(all(r['target']==TARGETS[-1] and r['phase']=='DONE' and r['subphase']=='home_settle'
                for r in rows[-120:]), 'incomplete final home settle')
    require(indices.ndim==1 and np.issubdtype(indices.dtype,np.integer)
            and np.array_equal(indices,np.arange(7,count,8)), 'saved-pose row inventory mismatch')
    require(body_paths.ndim==1 and len(body_paths)>0 and body_paths.dtype.kind in ('U','S')
            and len(set(map(str,body_paths)))==len(body_paths), 'invalid replay body paths')
    require(poses.shape==(len(indices),len(body_paths),7) and np.isfinite(poses).all(), 'invalid saved poses')
    require(np.allclose(np.linalg.norm(poses[:,:,3:],axis=2),1.,atol=1e-4,rtol=0), 'nonunit saved rotation')
    return times


def validate_schedule(data, outcome, rows, indices, poses, body_paths):
    times = validate_episode_rows(outcome,rows,indices,poses,body_paths)
    count = len(rows); frames_count = (count+15)//16
    integer(data.get('fps'),15,'fps'); number(data.get('playback_speed'),1.,'playback speed')
    require(data.get('resolution')==[1920,720], 'unexpected video resolution')
    integer(data.get('frame_count'),frames_count,'frame count')
    integer(data.get('source_physics_rows'),count,'source physics rows')
    for name, expected in dict(source_duration_s=count/240,duration_s=frames_count/15,
        terminal_video_padding_s=frames_count/15-count/240,source_physics_dt_s=1/240,
        source_pose_dt_s=1/30,first_saved_pose_s=times[indices[0]],last_saved_pose_s=times[indices[-1]]).items():
        number(data.get(name),expected,name)
    require(data.get('point_intervals')==[{k:p[k] for k in ('target','start_physics_row','end_physics_row')}
                for p in outcome['points']], 'point intervals differ')
    frames = data.get('frames')
    require(isinstance(frames,list) and len(frames)==frames_count, 'incomplete frame inventory')
    shown_holds = set()
    for index, frame in enumerate(frames):
        # Independent integer derivation: 16 physics ticks / video frame,
        # 8 ticks / saved pose, and the final partial exposure is retained.
        end = min((index+1)*16,count); row=end-1; pose=end//8-1
        for key, value in dict(frame_index=index,measurement_row=row,pose_index=pose,pose_row=indices[pose]).items():
            integer(frame.get(key),value,'frame '+str(index)+' '+key)
        for key, value in dict(video_time_s=index/15,simulation_interval_end_s=end/240,
            measurement_time_s=times[row],pose_time_s=times[indices[pose]]).items():
            number(frame.get(key),value,'frame '+str(index)+' '+key)
        for key in ('target','phase','subphase'):
            require(frame.get(key)==rows[row][key], 'frame/source label mismatch')
        if frame['phase']=='HOLD': shown_holds.add(frame['target'])
    require(shown_holds==set(TARGETS) and frames[-1]['subphase']=='home_settle', 'missing HOLD/final home frames')
    return dict(frame_count=frames_count,physics_rows=count,targets=list(TARGETS),
                final_home_included=True,all_frame_source_bindings_verified=True,
                hold_validation='HOLD states shown; continuous force-qualified duration belongs to the physical audit')


def _context(directory, manifest_name, schema, project, *, guard_path, module):
    directory, project = Path(directory).resolve(), Path(project).resolve()
    manifest_path = directory/manifest_name; manifest_hash=sha(manifest_path); data=read(manifest_path)
    require(data.get('schema')==schema, 'unsupported media schema')
    process = verify_process_completion(directory,data,guard_path=guard_path,project=project,module=module)
    sources = verify_sources(data.get('source_sha256'),project)
    episode = child(project,data.get('episode'))
    required = {str((episode/name).relative_to(project)) for name in
                ('outcome.json','training_scene.usdc','expert_trace.jsonl','expert_replay.npz')}
    require(required|{CAMERA} <= set(sources), 'incomplete media source inventory')
    require(data.get('camera_config')==CAMERA, 'unapproved camera configuration')
    camera=read(sources[CAMERA]); number(camera.get('downward_pitch_delta_deg'),7.,'camera pitch')
    number(camera.get('camera_height_delta_m'),.05,'camera height')
    require(data.get('source_files_unchanged') is True, 'source stability receipt absent')
    provenance=validate_episode_provenance(episode)
    require(provenance.get('status')=='passed', 'episode provenance rejected: '+str(provenance.get('reasons')))
    recorded=data.get('episode_provenance',{})
    require(recorded.get('status')=='passed' and recorded.get('artifact_sha256')==provenance.get('artifact_sha256'),
            'media/episode provenance mismatch')
    verify_sources(provenance['artifact_sha256'],episode)
    outcome=read(episode/'outcome.json')
    require(sha(episode/'training_scene.usdc')==outcome.get('training_scene_sha256'), 'scene hash mismatch')
    with (episode/'expert_trace.jsonl').open() as stream: rows=[read_json_text(line) for line in stream]
    with np.load(episode/'expert_replay.npz',allow_pickle=False) as archive:
        indices,poses,paths=(archive[k].copy() for k in ('row_indices','poses','paths'))
    validate_episode_rows(outcome,rows,indices,poses,paths)
    return dict(directory=directory,project=project,manifest_path=manifest_path,manifest_hash=manifest_hash,
                data=data,sources=sources,episode=episode,camera=camera,provenance=provenance,
                outcome=outcome,rows=rows,indices=indices,poses=poses,paths=paths,
                process_completion=process,module=module)


def _unchanged(context):
    verify_sources(context['data']['source_sha256'],context['project'])
    verify_sources(context['provenance']['artifact_sha256'],context['episode'])
    require(sha(context['manifest_path'])==context['manifest_hash'], 'media manifest changed during verification')
    require(verify_process_completion(context['directory'],context['data'],
        guard_path=context['process_completion']['guard_path'],project=context['project'],
        module=context['module'])==context['process_completion'], 'media process evidence changed during verification')


def verify_render(directory, project=PROJECT, *, guard_path=None):
    c=_context(directory,'review_manifest.json',RENDER_SCHEMA,project,
               guard_path=guard_path,module='sim.auscultation_skill_v1.render');data=c['data']
    implementation=verify_sources(data.get('renderer_implementation_sha256'),c['project'])
    required={'sim/auscultation_skill_v1/'+name for name in ('render.py','quality.py','provenance.py','contract.py')}
    require(required<=set(implementation), 'missing skill renderer implementation binding')
    retained='sim/upper_pair_posture_20260924/render_episode.py'
    require(retained in c['sources'] and data.get('inherited_renderer')==retained, 'renderer source mismatch')
    require({PRIVATE_RENDERER,TEARDOWN}<=set(c['sources'])
            and data.get('executed_renderer')==PRIVATE_RENDERER, 'private renderer/teardown source mismatch')
    require('sim/route_b_full_video_20260924/timeline.py' in c['sources'], 'timeline source missing')
    require(data.get('source_binding_validator')=='sim.auscultation_skill_v1.quality.validate_episode_provenance',
            'wrong episode provenance validator')
    require(data.get('executed_controller_class')==c['outcome'].get('executed_controller_class'), 'controller identity mismatch')
    require(data.get('layout')==LAYOUT, 'unapproved whole-arm/camera layout')
    require(data.get('source_outcome_accepted') is (c['outcome'].get('training_accepted') is True), 'source acceptance label differs')
    schedule=validate_schedule(data,c['outcome'],c['rows'],c['indices'],c['poses'],c['paths'])
    video=child(c['directory'],data.get('video'))
    require(video.is_file() and sha(video)==data.get('video_sha256'), 'video hash mismatch')
    decoded=decode_video(video,schedule['frame_count'],resolution=(1920,720),fps=15)
    _unchanged(c); verify_sources(data['renderer_implementation_sha256'],c['project'])
    require(sha(video)==data['video_sha256'], 'video changed during verification')
    return dict(status='passed',kind='full10_video',manifest=str(c['manifest_path']),manifest_sha256=c['manifest_hash'],
                video_sha256=data['video_sha256'],episode=str(c['episode']),schedule=schedule,decode=decoded,
                process_completion=c['process_completion'],
                scope='Full source timeline, frame/pose bindings, decoded dimensions/PTS and hashes verified; no independent pixel-to-3D reconstruction or physical acceptance claim.')


def expected_rgbd_samples(rows, indices, targets):
    require(list(targets)==list(TARGETS), 'full ten-point RGB-D capture required')
    result=[]
    def choose(kind,target,predicate,middle=False):
        choices=[(k,int(r)) for k,r in enumerate(indices) if predicate(rows[int(r)])]
        require(bool(choices),'missing saved '+kind+' pose for '+target)
        k,r=choices[len(choices)//2] if middle else choices[-1]
        result.append(dict(kind=kind,target=target,physics_row=r,pose_index=k,
            simulation_time_s=rows[r]['time_s'],phase=rows[r]['phase'],subphase=rows[r]['subphase']))
    choose('initial_home',targets[0],lambda r:r['target']==targets[0] and r['subphase']=='settle')
    for target in targets:
        choose('prepose',target,lambda r:r['target']==target and r['subphase']=='align')
        choose('hold',target,lambda r:r['target']==target and r['phase']=='HOLD',True)
        choose('release',target,lambda r:r['target']==target and r['subphase']=='normal_release_confirm')
    choose('final_home',targets[-1],lambda r:r['target']==targets[-1] and r['subphase']=='home_settle')
    return result


def validate_rgbd_bindings(samples, expected):
    require(isinstance(samples,list) and len(samples)==len(expected)==32, 'RGB-D inventory must contain 32 samples')
    for index,(sample,reference) in enumerate(zip(samples,expected)):
        integer(sample.get('capture_id'),index,'capture id')
        for key,value in reference.items():
            if key in ('physics_row','pose_index'): integer(sample.get(key),value,key)
            elif key=='simulation_time_s': number(sample.get(key),value,key)
            else: require(sample.get(key)==value,'RGB-D pose binding differs: '+key)
        for key in ('rgb_source_time_s','depth_source_time_s'):
            number(sample.get(key),reference['simulation_time_s'],key)


def validate_rgbd_triplet(directory, sample, intrinsics):
    from PIL import Image
    files=sample.get('files',{}); hashes=sample.get('file_sha256',{})
    require(set(files)==set(hashes)=={'rgb','depth_m','valid_mask'}, 'incomplete RGB-D file triplet')
    paths={name:child(directory,relative) for name,relative in files.items()}
    require(len(set(paths.values()))==3, 'RGB-D file aliases')
    for name,path in paths.items():require(path.is_file() and sha(path)==hashes[name], 'RGB-D file hash mismatch: '+name)
    with Image.open(paths['rgb']) as image:
        require(image.format=='PNG' and image.mode=='RGB', 'RGB must be lossless three-channel PNG')
        pixels=np.asarray(image)
    shape=(intrinsics['height'],intrinsics['width'])
    require(pixels.shape==(*shape,3) and pixels.dtype==np.uint8, 'RGB dimensions/type mismatch')
    require(float(pixels.std())>=5., 'RGB appears blank')
    depth=np.load(paths['depth_m'],allow_pickle=False); mask=np.load(paths['valid_mask'],allow_pickle=False)
    require(depth.shape==shape and depth.dtype==np.float32, 'metric depth dimensions/type mismatch')
    require(mask.shape==shape and mask.dtype==np.bool_, 'valid mask dimensions/type mismatch')
    expected=np.isfinite(depth)&(depth>0)&(depth<100.)
    require(np.array_equal(mask,expected), 'valid mask differs from raw metric depth')
    require(bool(expected.any()), 'RGB-D capture has no valid depth')
    number(sample.get('valid_depth_fraction'),float(expected.mean()),'valid depth fraction')
    for name,path in paths.items():require(sha(path)==hashes[name], 'RGB-D file changed during verification')
    return depth


def _array(actual, expected, label, tolerance=1e-8):
    try: actual=np.asarray(actual,float);expected=np.asarray(expected,float)
    except (TypeError,ValueError):raise ValueError('invalid '+label)
    require(actual.shape==expected.shape and np.isfinite(actual).all()
            and np.allclose(actual,expected,atol=tolerance,rtol=0), 'invalid '+label)


def validate_projections(records, targets, depth, k, world_from_optical):
    require(isinstance(records,list) and [r.get('name') for r in records]==list(targets), 'projection target inventory differs')
    transform=np.asarray(world_from_optical,float); rotation=transform[:3,:3];origin=transform[:3,3]
    all_matching=True
    for record,(name,target) in zip(records,targets.items()):
        target=np.asarray(target,float);p=rotation.T@(target-origin);z=float(p[2]);forward=z>0
        _array(record.get('target_world_m'),target,'projection target')
        number(record.get('optical_depth_m'),z,'target optical depth')
        uv=np.array([k['fx']*p[0]/z+k['cx'],k['fy']*p[1]/z+k['cy']]) if forward else None
        inside=bool(forward and 0<=uv[0]<k['width'] and 0<=uv[1]<k['height'])
        pixel=np.rint(uv).astype(int) if inside else None
        valid=bool(inside and 0<=pixel[0]<k['width'] and 0<=pixel[1]<k['height'])
        measured=float(depth[pixel[1],pixel[0]]) if valid else None
        valid=bool(valid and np.isfinite(measured) and 0<measured<100.)
        residual=measured-z if valid else None
        for key,value in dict(inside=inside,valid_depth=valid,occluded_by_10mm=bool(valid and residual<-.01),
            depth_matches_within_5mm=bool(valid and abs(residual)<=.005)).items():
            require(record.get(key) is value,'incorrect projection '+key)
        all_matching=all_matching and bool(valid and abs(residual)<=.005)
        for key,value in [('uv_px',uv),('sample_pixel_xy',pixel)]:
            if value is None:require(record.get(key) is None,'invalid behind/offscreen projection')
            else:_array(record.get(key),value,key)
        for key,value in [('measured_depth_m',measured if valid else None),('depth_residual_m',residual)]:
            if value is None:require(record.get(key) is None,'invalid depth must remain null')
            else:number(record.get(key),value,key)
        if valid:
            optical=np.array([(pixel[0]-k['cx'])*measured/k['fx'],(pixel[1]-k['cy'])*measured/k['fy'],measured])
            reconstructed=rotation@optical+origin
            _array(record.get('backprojected_world_m'),reconstructed,'measured backprojection')
            number(record.get('measured_backprojection_residual_m'),np.linalg.norm(reconstructed-target),'backprojection residual')
        else:
            require(record.get('backprojected_world_m') is None and record.get('measured_backprojection_residual_m') is None,'invalid backprojection must remain null')
        if forward:number(record.get('mathematical_roundtrip_residual_m'),0.,'mathematical roundtrip',1e-9)
        else:require(record.get('mathematical_roundtrip_residual_m') is None,'behind-camera roundtrip must remain null')
    return all_matching


def verify_rgbd(directory, project=PROJECT, *, guard_path=None):
    c=_context(directory,'rgbd_manifest.json',RGBD_SCHEMA,project,
               guard_path=guard_path,module='sim.auscultation_skill_v1.rgbd');data=c['data'];camera=c['camera']
    require({'sim/auscultation_skill_v1/'+n for n in ('rgbd.py','quality.py','provenance.py','contract.py')}<=set(c['sources']), 'capture implementation binding missing')
    require(TEARDOWN in c['sources'], 'capture teardown source binding missing')
    require(data.get('physics_advanced') is False and data.get('raw_rgb_has_overlays') is False,'RGB-D capture scope differs')
    require(data.get('depth_semantics')=='distance_to_image_plane: metric optical-Z, not radial range','wrong depth semantics')
    k=data.get('intrinsics',{}); expected=camera['intrinsics']
    require(set(k)==set(expected) and (k.get('width'),k.get('height'))==(1920,1080),'unapproved RGB-D dimensions/calibration')
    for key,value in expected.items():number(k.get(key),value,'camera '+key,1e-6)
    _array(data.get('world_from_optical'),camera['world_from_optical'],'optical extrinsics')
    _array(data.get('world_from_camera_usd'),camera['world_from_camera_usd'],'USD extrinsics')
    transform=np.asarray(data['world_from_optical'],float)
    _array(transform[3],[0,0,0,1],'extrinsic homogeneous row')
    _array(transform[:3,:3].T@transform[:3,:3],np.eye(3),'extrinsic rotation')
    number(float(np.linalg.det(transform[:3,:3])),1.,'extrinsic handedness',1e-8)
    _array(np.asarray(data['world_from_camera_usd'])@np.diag([1,-1,-1,1]),transform,'USD/optical convention')
    samples=data.get('samples');expected_samples=expected_rgbd_samples(c['rows'],c['indices'],list(TARGETS))
    integer(data.get('sample_count'),32,'RGB-D sample count');validate_rgbd_bindings(samples,expected_samples)
    targets={p['target']:p['world_target_m'] for p in c['outcome']['points']}
    seen=set();report=[]
    for sample in samples:
        depth=validate_rgbd_triplet(c['directory'],sample,k)
        for filename in sample['files'].values():
            path=child(c['directory'],filename);require(path not in seen,'reused RGB-D sample file');seen.add(path)
        stamp=datetime.datetime.fromisoformat(sample['capture_utc'])
        require(stamp.tzinfo is not None,'capture UTC timezone absent')
        require(sample.get('synchronization')=='both annotators on one render product after one completed orchestrator capture at a fixed recorded pose','capture synchronization receipt differs')
        visible=validate_projections(sample.get('target_projection'),targets,depth,k,transform)
        require(sample.get('all_targets_visible') is visible,'visibility summary differs')
        report.append(dict(capture_id=sample['capture_id'],target=sample['target'],kind=sample['kind'],
                           physics_row=sample['physics_row'],all_targets_visible=visible))
    _unchanged(c)
    for sample in samples:
        for name,filename in sample['files'].items():require(sha(child(c['directory'],filename))==sample['file_sha256'][name],'RGB-D changed during verification')
    return dict(status='passed',kind='saved_pose_rgbd',manifest=str(c['manifest_path']),manifest_sha256=c['manifest_hash'],
                episode=str(c['episode']),samples=report,sample_count=32,file_count=len(seen),
                process_completion=c['process_completion'],
                scope='Paired saved-pose sensor-interface proof; calibration, metric depth, masks, hashes and projections verified. Occlusions are retained; not RGB-D policy training.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render',type=Path);parser.add_argument('--rgbd',type=Path)
    parser.add_argument('--render-guard',type=Path);parser.add_argument('--rgbd-guard',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.render is None and args.rgbd is None:parser.error('provide --render and/or --rgbd')
    if (args.render is None)!=(args.render_guard is None):parser.error('--render and --render-guard are required together')
    if (args.rgbd is None)!=(args.rgbd_guard is None):parser.error('--rgbd and --rgbd-guard are required together')
    output=args.output.resolve()
    for directory in (args.render,args.rgbd):
        if directory is not None and (output==directory.resolve() or directory.resolve() in output.parents):
            parser.error('receipt must be outside immutable media directories')
    if output.exists():parser.error('receipt path already exists')
    reports={}
    if args.render is not None:reports['render']=verify_render(args.render,guard_path=args.render_guard)
    if args.rgbd is not None:reports['rgbd']=verify_rgbd(args.rgbd,guard_path=args.rgbd_guard)
    if len(reports)==2:require(reports['render']['episode']==reports['rgbd']['episode'],'video and RGB-D episodes differ')
    for report in reports.values():require(Path(report['episode']) not in output.parents,'receipt must be outside source episode')
    result=dict(schema='auscultation_skill_media_verification_v1',status='passed',reports=reports,
                verifier_sha256=sha(__file__),decoder_sha256=sha(Path(__file__).parents[1]/'upper_pair_posture_20260924/verify_video.py'))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='passed',receipt=str(output),receipt_sha256=sha(output))))
    return result


if __name__=='__main__':main()
