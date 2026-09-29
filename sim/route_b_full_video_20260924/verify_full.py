"""Read-only source and full decode verification for the canonical complete B video.

Run: python -B -m sim.route_b_full_video_20260924.verify_full MANIFEST --output RECEIPT
The integer schedule here is independent of the renderer and timeline.py.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sim.curobo_expert_v1.quality import validate_episode_provenance, verify_hashes

PROJECT = Path(__file__).resolve().parents[2]
EPISODE_REL = 'sim/curobo_expert_v1/results/physics_revised_all10_v1'
CAMERA_REL = 'sim/camera_framing_20260923/recommended_camera.json'
EXPECTED_ROWS = 95484
HOLD_STEPS = 720
HOME_SETTLE_STEPS = 120
TARGETS = ('R1','L1','R2','L2','R3','L3','R4','L4','R5','L5')
FPS, PHYSICS_HZ, POSE_STRIDE, FRAME_STEPS = 15, 240, 8, 16
RESOLUTION = (1920, 720)
EXPECTED_LAYOUT = {'posture':[0,0,1280,720], 'robotcam':[1280,0,640,360],
                   'metadata':[1280,360,640,360]}
REQUIRED_STATIC_SOURCES = (
    CAMERA_REL, 'sim/route_b_full_video_20260924/render_full.py',
    'sim/route_b_full_video_20260924/timeline.py',
    'sim/curobo_expert_v1/render_review.py', 'sim/curobo_expert_v1/quality.py',
    'sim/curobo_expert_v1/provenance.py', 'sim/expert_quality_20260924/quality.py')
TRACE_FIELDS = ('time_s','target','phase','subphase','hall_N','lip_N',
                'diaphragm_N','contact_rim_N','hold_elapsed_s')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite JSON number: '+value)


def _json(text):
    return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)


def _read(path):
    return _json(Path(path).read_text())


def _number(actual, expected, label, tolerance=1e-8):
    _require(type(actual) in (int, float) and math.isfinite(actual)
             and abs(actual-expected) <= tolerance, 'invalid '+label)


def _integer(actual, expected, label):
    _require(type(actual) is int and actual == expected, 'invalid '+label)


def _safe_child(root, relative, label):
    _require(isinstance(relative,str) and relative and not Path(relative).is_absolute()
             and '..' not in Path(relative).parts, 'unsafe '+label+' path')
    path = (root/relative).resolve()
    _require(root in path.parents, 'unsafe '+label+' path')
    return path


def _digest(value, label):
    _require(isinstance(value,str) and len(value)==64
             and all(c in '0123456789abcdef' for c in value), 'invalid '+label+' SHA256')


def required_sources():
    return set(REQUIRED_STATIC_SOURCES) | {
        EPISODE_REL+'/'+name for name in ('training_scene.usdc','outcome.json',
                                          'expert_trace.jsonl','expert_replay.npz')}


def _source_bindings(data, project):
    sources = data.get('source_sha256')
    _require(isinstance(sources,dict), 'invalid source inventory')
    paths = {}
    for relative, expected in sources.items():
        path = _safe_child(project,relative,'source')
        _digest(expected,'source')
        _require(path.is_file() and sha256(path)==expected, 'source hash mismatch: '+relative)
        paths[relative] = path
    _require(required_sources().issubset(sources), 'incomplete source inventory')
    _require(len(set(paths.values()))==len(paths), 'duplicate source path aliases')
    return paths


def _read_trace(path, points):
    rows = []
    point_cursor = 0
    with path.open() as stream:
        for index, line in enumerate(stream):
            _require(index < EXPECTED_ROWS, 'extra physics row')
            row = _json(line)
            _require(isinstance(row,dict) and set(TRACE_FIELDS).issubset(row), 'invalid trace record')
            _number(row['time_s'], (index+1)/PHYSICS_HZ, 'trace clock at row '+str(index),1e-7)
            for key in ('hall_N','lip_N','diaphragm_N','contact_rim_N','hold_elapsed_s'):
                _require(type(row[key]) in (int,float) and math.isfinite(row[key]), 'invalid trace '+key)
            for key in ('target','phase','subphase'):
                _require(isinstance(row[key],str) and row[key], 'invalid trace '+key)
            while index >= points[point_cursor]['end_physics_row']:
                point_cursor += 1
            _require(row['target']==points[point_cursor]['target'], 'trace point interval target mismatch')
            rows.append({key:row[key] for key in TRACE_FIELDS})
    _integer(len(rows), EXPECTED_ROWS, 'physics row count')
    for point in points:
        hold = [i for i in range(point['start_physics_row'],point['end_physics_row'])
                if rows[i]['phase']=='HOLD']
        _require(len(hold)==HOLD_STEPS and hold[-1]-hold[0]+1==HOLD_STEPS,
                 'missing or noncontinuous full HOLD: '+point['target'])
    _require(all(row['target']=='L5' and row['phase']=='DONE' and row['subphase']=='home_settle'
                 for row in rows[-HOME_SETTLE_STEPS:]), 'incomplete final home_settle')
    _require(rows[-HOME_SETTLE_STEPS-1]['subphase']!='home_settle', 'unexpected home_settle interval')
    return rows


def validate_full_sources(manifest, project=PROJECT):
    """Validate metadata against actual hashed sources, without reading video pixels.

    project is injectable only to support isolated tests/relocation; the canonical
    relative episode, source inventory and complete route contract stay fixed.
    """
    manifest = Path(manifest).resolve(); project = Path(project).resolve()
    data = _read(manifest)
    _require(isinstance(data,dict) and data.get('schema')=='continuous_recorded_route_v1', 'invalid manifest schema')
    _require(data.get('episode')==EPISODE_REL, 'wrong canonical episode')
    episode = _safe_child(project, EPISODE_REL, 'episode')
    paths = _source_bindings(data,project)
    video = _safe_child(manifest.parent,data.get('video'),'video')
    _digest(data.get('video_sha256'),'video')
    _require(video.is_file() and sha256(video)==data['video_sha256'], 'video hash mismatch')
    _require(data.get('source_files_unchanged') is True, 'render source stability not recorded')
    _require(data.get('camera_config')==CAMERA_REL and data.get('layout')==EXPECTED_LAYOUT,
             'unexpected camera/layout')
    camera = _read(paths[CAMERA_REL])
    _number(camera.get('downward_pitch_delta_deg'),7,'approved camera pitch')
    _number(camera.get('camera_height_delta_m'),.05,'approved camera height')
    transform = np.asarray(camera.get('world_from_camera_usd'),dtype=float)
    _require(transform.shape==(4,4) and np.isfinite(transform).all(), 'invalid camera transform')
    _require(data.get('resolution')==list(RESOLUTION), 'unexpected video resolution')
    _integer(data.get('fps'),FPS,'fps')
    _number(data.get('playback_speed'),1,'playback speed')
    _number(data.get('source_physics_dt_s'),1/PHYSICS_HZ,'physics sampling rate',1e-12)
    _number(data.get('source_pose_dt_s'),POSE_STRIDE/PHYSICS_HZ,'pose sampling rate',1e-12)
    count = (EXPECTED_ROWS+FRAME_STEPS-1)//FRAME_STEPS
    _integer(data.get('frame_count'),count,'frame count')
    _integer(data.get('source_physics_rows'),EXPECTED_ROWS,'source physics rows')
    _number(data.get('source_duration_s'),EXPECTED_ROWS/PHYSICS_HZ,'source duration')
    _number(data.get('duration_s'),count/FPS,'video duration')
    padding = (count*FRAME_STEPS-EXPECTED_ROWS)/PHYSICS_HZ
    _number(data.get('terminal_video_padding_s'),padding,'terminal padding')
    outcome = _read(episode/'outcome.json')
    _require(outcome.get('success') is True and outcome.get('training_accepted') is True,
             'canonical source is not a completed accepted engineering episode')
    _require(data.get('source_outcome_accepted') is True, 'source outcome acceptance mismatch')
    _integer(outcome.get('physics_steps'),EXPECTED_ROWS,'outcome physics rows')
    _require(outcome.get('training_scene_sha256')==data['source_sha256'][EPISODE_REL+'/training_scene.usdc'],
             'scene does not match bound outcome')
    points = outcome.get('points')
    _require(isinstance(points,list) and [p.get('target') for p in points]==list(TARGETS),
             'invalid ten point inventory/order')
    previous = 0
    for point in points:
        _require(point.get('success') is True and point.get('training_accepted') is True,
                 'point not successful: '+point['target'])
        _integer(point.get('start_physics_row'),previous,'point interval start')
        end = point.get('end_physics_row')
        _require(type(end) is int and previous<end<=EXPECTED_ROWS, 'invalid point interval end')
        previous = end
    _integer(previous,EXPECTED_ROWS,'complete point interval coverage')
    intervals = [{key:p[key] for key in ('target','start_physics_row','end_physics_row')} for p in points]
    _require(data.get('point_intervals')==intervals, 'manifest point interval mismatch')
    recorded_acceptance = {}
    for name in ('independent_audit.json','posture_audit.json'):
        recorded_acceptance[name] = _read(episode/name).get('accepted')
        _require(recorded_acceptance[name] is True,'source engineering audit rejected: '+name)
    rows = _read_trace(episode/'expert_trace.jsonl',points)
    with np.load(episode/'expert_replay.npz', allow_pickle=False) as archive:
        indices, poses, body_paths = (archive[key] for key in ('row_indices','poses','paths'))
        _require(indices.ndim==1 and np.issubdtype(indices.dtype,np.integer)
                 and np.array_equal(indices,np.arange(7,EXPECTED_ROWS,8)), 'invalid exact replay pose row inventory')
        _require(body_paths.ndim==1 and len(body_paths)>0
                 and body_paths.dtype.kind in ('U','S')
                 and len(set(map(str,body_paths)))==len(body_paths), 'invalid replay body paths')
        _require(poses.shape==(len(indices),len(body_paths),7) and np.isfinite(poses).all(), 'invalid replay poses')
        _require(np.all(np.abs(np.linalg.norm(poses[:,:,3:],axis=2)-1)<1e-4), 'invalid replay rotation')
    _number(data.get('first_saved_pose_s'),rows[int(indices[0])]['time_s'],'first saved pose time')
    _number(data.get('last_saved_pose_s'),rows[int(indices[-1])]['time_s'],'last saved pose time')
    frames = data.get('frames')
    _require(isinstance(frames,list) and len(frames)==count, 'incomplete frame inventory')
    hold_frames = Counter(); home_frames = 0; max_pose_lag = 0; previous_end = 0
    frame_bindings_digest = hashlib.sha256()
    for index, frame in enumerate(frames):
        _require(isinstance(frame,dict),'invalid frame record')
        end = min((index+1)*FRAME_STEPS,EXPECTED_ROWS)
        measurement_row = end-1
        # Replay is exactly every eighth step; this independently determines its
        # most recent preceding sample, without renderer searchsorted/floats.
        pose_index = (measurement_row+1)//POSE_STRIDE-1
        pose_row = (pose_index+1)*POSE_STRIDE-1
        _require(end>previous_end and end-previous_end<=FRAME_STEPS, 'invalid integer interval coverage')
        for key,expected in (('frame_index',index),('measurement_row',measurement_row),
                             ('pose_index',pose_index),('pose_row',pose_row)):
            _integer(frame.get(key),expected,'frame '+str(index)+' '+key)
        _integer(int(indices[pose_index]),pose_row,'archive pose row')
        row = rows[measurement_row]
        for key, expected in (('video_time_s',index/FPS),('simulation_interval_end_s',end/PHYSICS_HZ),
                              ('measurement_time_s',row['time_s']),('pose_time_s',rows[pose_row]['time_s'])):
            _number(frame.get(key),expected,'frame '+str(index)+' '+key)
        for key in ('target','phase','subphase'):
            _require(frame.get(key)==row[key], 'frame source mismatch: '+key+' at '+str(index))
        if row['phase']=='HOLD':hold_frames[row['target']]+=1
        if row['subphase']=='home_settle':home_frames+=1
        max_pose_lag = max(max_pose_lag,measurement_row-pose_row)
        bound = dict(frame_index=index, interval_start_physics_row=previous_end,
                     interval_end_physics_row=end, measurement_row=measurement_row,
                     pose_index=pose_index,pose_row=pose_row,measurement=row)
        frame_bindings_digest.update(json.dumps(bound,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        previous_end = end
    _require(set(hold_frames)==set(TARGETS), 'not all ten HOLD periods appear in video')
    _require(frames[-1]['target']=='L5' and frames[-1]['phase']=='DONE'
             and frames[-1]['subphase']=='home_settle' and home_frames>0, 'last home_settle missing from video')
    provenance = validate_episode_provenance(episode)
    _require(provenance.get('status')=='passed', 'episode provenance rejected: '+str(provenance.get('reasons')))
    recorded = data.get('episode_provenance')
    _require(isinstance(recorded,dict) and recorded.get('status')=='passed'
             and recorded.get('artifact_sha256')==provenance.get('artifact_sha256'),
             'render provenance artifact binding differs from fresh validation')
    _require(isinstance(provenance.get('artifact_sha256'),dict) and bool(provenance['artifact_sha256']),
             'missing provenance artifact hashes')
    _require(not verify_hashes(episode,provenance['artifact_sha256']), 'episode artifact hashes changed')
    semantic = dict(status='passed', frames_verified=count, source_rows_verified=len(rows),
        coverage=dict(physics_rows=EXPECTED_ROWS,first_interval_start_row=0,last_interval_end_row=previous_end,
                      frame_count=count,full_interval_steps=FRAME_STEPS,full_interval_count=count-1,
                      last_interval_steps=EXPECTED_ROWS-(count-1)*FRAME_STEPS,
                      source_duration_s=EXPECTED_ROWS/PHYSICS_HZ,video_duration_s=count/FPS,padding_s=padding),
        source_pose_count=len(indices),body_count=len(body_paths),max_pose_lag_physics_steps=max_pose_lag,
        pose_sampling='latest preceding saved 30 Hz pose, no interpolation',
        hold_source_step_counts={target:HOLD_STEPS for target in TARGETS},
        hold_video_frame_counts=dict(hold_frames),home_settle_source_steps=HOME_SETTLE_STEPS,
        home_settle_video_frames=home_frames,first_frame=frames[0],last_frame=frames[-1],
        frame_source_bindings_sha256=frame_bindings_digest.hexdigest(),
        checks=['all_source_hashes','fresh_episode_provenance','scene_outcome_binding',
                'all_integer_frame_intervals','all_trace_clocks','all_replay_poses',
                'measurement_row_time','pose_index_row_time','trace_target_phase_subphase',
                'all_ten_continuous_HOLDs','complete_terminal_home_settle'])
    return dict(manifest=data,video_path=video,source_paths=paths,episode_path=episode,
                episode_provenance=provenance,recorded_engineering_acceptance=recorded_acceptance,
                semantic_binding=semantic)


def decode_full(video, expected_count, fps=FPS, resolution=RESOLUTION):
    """Decode sequentially to EOF; never seek or rely only on container totals."""
    import cv2
    capture = cv2.VideoCapture(str(video))
    count = 0; pixel_digest = hashlib.sha256(); max_pts_error_s = 0.0
    try:
        _require(capture.isOpened(),'video cannot be opened')
        _number(float(capture.get(cv2.CAP_PROP_FPS)),fps,'decoded fps',1e-6)
        _number(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),resolution[0],'decoded width',0)
        _number(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),resolution[1],'decoded height',0)
        _number(float(capture.get(cv2.CAP_PROP_FRAME_COUNT)),expected_count,'container frame count',0)
        while True:
            ok, pixels = capture.read()
            if not ok:break
            _require(count<expected_count,'extra decoded frame')
            _require(pixels is not None and pixels.shape==(resolution[1],resolution[0],3)
                     and pixels.dtype==np.uint8, 'invalid decoded frame dimensions/type at '+str(count))
            timestamp_s = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            _number(timestamp_s,count/fps,'decoded presentation timestamp at frame '+str(count),1e-5)
            max_pts_error_s = max(max_pts_error_s,abs(timestamp_s-count/fps))
            pixel_digest.update(pixels.tobytes())
            count += 1
        _integer(count,expected_count,'decoded frame count (possibly truncated)')
    finally:
        capture.release()
    return dict(decoded_full=True,decoded_frame_count=count,fps=fps,resolution=list(resolution),
                presentation_timestamps_verified=True,max_pts_error_s=max_pts_error_s,
                last_frame_pts_s=(count-1)/fps,constant_rate_duration_s=count/fps,
                opencv_version=cv2.__version__,
                decoded_bgr_stream_sha256=pixel_digest.hexdigest(),
                pixel_hash_scope='All sequential OpenCV BGR bytes; codec/platform dependent, not the encoded file SHA.')


def verify_full(manifest, output, project=PROJECT):
    """Write a passed receipt only after bindings and every frame decode pass."""
    manifest=Path(manifest).resolve(); output=Path(output).resolve()
    before=sha256(manifest); verifier_before=sha256(__file__)
    result=validate_full_sources(manifest,project)
    episode=result['episode_path']; data=result['manifest']; video=result['video_path']
    protected={manifest,video,Path(__file__).resolve(),*result['source_paths'].values()}
    # The canonical renderer is a new package; receipts must never mutate any
    # prior package, even a file that happens not to be among this video's inputs.
    new_package=(Path(project)/'sim/route_b_full_video_20260924').resolve()
    simulation_root=(Path(project)/'sim').resolve()
    outside_old_packages=(simulation_root not in output.parents
                          or new_package in output.parents)
    _require(output not in protected and output!=episode and episode not in output.parents,
             'receipt must not overwrite evidence or immutable source episode')
    _require(outside_old_packages,'receipt must not write into a frozen historical simulation package')
    decoded=decode_full(video,data['frame_count'])
    _require(sha256(manifest)==before,'manifest changed during verification')
    _require(sha256(__file__)==verifier_before,'verifier changed during verification')
    _require(sha256(video)==data['video_sha256'],'video changed during verification')
    for relative,path in result['source_paths'].items():
        _require(sha256(path)==data['source_sha256'][relative], 'source changed during verification: '+relative)
    _require(not verify_hashes(episode,result['episode_provenance']['artifact_sha256']),
             'episode changed during verification')
    report=dict(schema='continuous_recorded_route_verification_v1',status='passed',
        manifest_sha256=before,verifier_sha256=verifier_before,episode=EPISODE_REL,
        video=data['video'],video_sha256=data['video_sha256'],source_sha256=data['source_sha256'],
        episode_provenance=result['episode_provenance'],source_files_unchanged=True,
        recorded_engineering_acceptance=result['recorded_engineering_acceptance'],
        semantic_binding=result['semantic_binding'],video_decode=decoded,
        scope='Media/source verification only: metadata-to-trace/replay bindings and full sequential decode. '
              'Coverage means all physics intervals are represented by end-of-frame samples, not a visible frame for every physics step. '
              'No independent RGB pose estimation/OCR, new physical audit, expert endorsement, clinical or acoustic acceptance.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=verify_full(args.manifest,args.output)
    print(json.dumps({'status':report['status'],'frames_verified':report['semantic_binding']['frames_verified'],
                      'decoded_frames':report['video_decode']['decoded_frame_count'],'receipt':str(args.output.resolve())}))


if __name__=='__main__':main()
