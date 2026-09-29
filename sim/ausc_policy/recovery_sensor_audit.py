"""Independent CPU replay of raw recovery-pilot modalities, never data sealing.

Reconstruct Hall packets, RGB-D tracking and encoded state from acquisition.
Keep all original eligibility flags false; no arrays, labels or model exported.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def bind_sensor_frames(physics,frames):
    from sim.auscultation_vla_v2.hall import HallPacket, HallSimulator, default_calibration
    from sim.auscultation_vla_v2.multimodal import INSTRUCTIONS
    if not physics or not frames:
        raise ValueError('Empty actual physics/sensor history')
    times=np.asarray([r['time_s'] for r in physics],float)
    if (not np.isfinite(times).all() or abs(times[0]-1/240)>1e-8
            or not np.allclose(np.diff(times),1/240,atol=1e-8,rtol=0)
            or [r['physics_row'] for r in frames] != list(range(11,len(physics),12))
            or [r['index'] for r in frames] != list(range(len(frames)))):
        raise ValueError('Nonconsecutive acquired physics or camera clocks')
    by_row={r['physics_row']:r for r in frames}
    simulator=HallSimulator(default_calibration())
    roles={};volts=[];adc=[]
    for i,row in enumerate(physics):
        stamp=row['time_s'];simulator.update(stamp,row['compression_m'])
        if i not in by_row: continue
        frame=by_row[i];role=row['oracle_role']
        roles[role]=roles.get(role,0)+1
        if role not in ('PREPARATION','EXPERT_RECOVERY','MEASURED_ANALYTIC_L3','FINAL_RETURN'):
            raise ValueError('Unknown acquisition role')
        if (frame['eligible'] is not False or frame['time_s'] != stamp
                or frame['target'] != row['target'] or frame['subphase'] != row['subphase']):
            raise ValueError('Acquisition identity or explicit ineligibility changed')
        for key in ('q','dq'):
            q=np.asarray(frame[key],float);actual=np.asarray(row[key],float)
            if q.shape != (7,) or not np.isfinite(q).all() or not np.array_equal(q,actual):
                raise ValueError('Acquired '+key+' does not bind same physical row')
        if frame['hall'] != asdict(simulator.read(stamp)):
            raise ValueError('ADC differs from full 240 Hz Hall replay')
        decoded=default_calibration().decode(HallPacket(**frame['hall']),now_s=stamp)
        volts.append(decoded.hall_voltage_v);adc.append(frame['hall']['raw'])
        candidates={'FREE'}
        if role=='MEASURED_ANALYTIC_L3':
            phases=[row['phase']]
            if i+1<len(physics): phases.append(physics[i+1]['phase'])
            candidates={'RETRACT' if p=='DONE' else p for p in phases}
        if (frame['phase'] not in candidates or frame['prompt'] != INSTRUCTIONS[frame['phase']]
                or not np.isfinite(frame['render_physics_delta_s'])
                or abs(frame['render_physics_delta_s'])>1e-8):
            raise ValueError('Phase, instruction or render clock differs from recovery collector')
    return dict(physics_rows_replayed=len(physics),hall_packets_matched=len(frames),
        role_frame_counts=roles,training_dataset_eligible=False,
        acquired_voltage_range_v=[min(volts),max(volts)],adc_range=[min(adc),max(adc)],
        hall_model='Full 240 Hz displacement to calibrated voltage/ADC; default engineering parameters')


def validate_pixels(arrays,shape):
    if set(arrays) != {'rgb','depth_m','depth_valid'}:
        raise ValueError('Unexpected or oracle raw modalities')
    rgb,depth,valid=[arrays[k] for k in ('rgb','depth_m','depth_valid')]
    if (rgb.shape != (*shape,3) or rgb.dtype != np.uint8
            or depth.shape != shape or depth.dtype.kind != 'f'
            or valid.shape != shape or valid.dtype != bool
            or not np.array_equal(valid,np.isfinite(depth)&(depth>0))):
        raise ValueError('RGB/depth dtype, shape or validity mask differs from raw capture')


def snapshot_binding(case,outcome):
    root=Path(__file__).resolve().parents[2]
    manifest=case/'input_snapshot/manifest.json'
    if sha(manifest) != outcome['input_snapshot_manifest_sha256']:
        raise ValueError('Acquisition snapshot identity changed')
    files=read(manifest)['files'];bound={manifest:sha(manifest)}
    wanted=[]
    for package,names in {
        'auscultation_vla_v2':('body_tracking.py','rigid_registration.py','tracking.py',
            'observations.py','hall.py','multimodal.py','live_capture.py'),
        'auscultation_pi05_v6':('live_capture.py','hall_conditions.py'),
        'auscultation_expert_layout_v1':('kinematics.py','tool_frames.py'),
        'ausc_policy':('recovery_teacher.py',),
    }.items():
        wanted.extend(('source/code/'+package+'/'+n,root/'sim'/package/n) for n in names)
    for name in ('sim/camera_framing_20260923/recommended_camera.json',
                 'sim/auscultation_vla_v2/sensor_pilot_annotation.json',
                 'sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json',
                 'sim/auscultation_expert_layout_v1/openarm_v10_follower_stetho.urdf'):
        wanted.append(('source/'+name,root/name))
    from auscultation.mechanical_sensor.force_feedback import force_sensor
    wanted.append(('source/calibration/force_sensor.py',Path(force_sensor.__file__)))
    for key,path in wanted:
        item=files[key];obj=manifest.parent/'objects'/item['sha256']
        if (not obj.is_file() or obj.is_symlink() or obj.stat().st_size!=item['bytes']
                or sha(obj)!=item['sha256'] or sha(path)!=item['sha256']):
            raise ValueError('Replay source differs from actual acquisition: '+key)
        bound[path]=item['sha256'];bound[obj]=item['sha256']
    return bound


def bind_calibration_runtime(case):
    """Locate the same host calibration used at capture before importing Hall.

    The thin sim checkout intentionally omits the host auscultation package.
    The acquisition owner and snapshot bind its approved external source.
    """
    owner=read(case.parent.parent/'result.json');outcome=read(case/'outcome.json')
    manifest=case/'input_snapshot/manifest.json'
    if sha(manifest)!=outcome['input_snapshot_manifest_sha256']:
        raise ValueError('Acquisition snapshot identity changed')
    source=read(manifest)['files']['source/calibration/force_sensor.py']
    asset=Path(owner['config']['asset_root']).resolve()
    calibration=asset/'auscultation/mechanical_sensor/force_feedback/force_sensor.py'
    if sha(calibration)!=source['sha256']:
        raise ValueError('Approved external Hall calibration source changed')
    if str(asset) not in sys.path: sys.path.append(str(asset))
    from auscultation.mechanical_sensor.force_feedback import force_sensor
    if sha(force_sensor.__file__)!=source['sha256']:
        raise ValueError('Imported Hall calibration differs from acquisition')


def audit_case(case,deadline):
    bind_calibration_runtime(case)
    import cv2
    import scipy
    from sim.auscultation_vla_v2.hall import HallPacket, default_calibration
    from sim.auscultation_vla_v2.multimodal import INSTRUCTIONS
    from sim.auscultation_vla_v2.body_tracking import BodyTracker
    from sim.auscultation_vla_v2.observations import Observation,RobotState,encode
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_pi05_v6.hall_conditions import DEFAULT
    root=Path(__file__).resolve().parents[2]
    owner=read(case.parent.parent/'result.json');outcome=read(case/'outcome.json');result=read(case/'result.json')
    if not (owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
            and owner['stage']['exit_code']==0 and owner['config']['experiment']=='recovery_run'
            and result['execution_completed'] and outcome['expert_demonstration']
            and outcome['learned_policy'] is False and outcome['training_dataset_eligible'] is False):
        raise ValueError('Completed immutable owned expert acquisition required')
    bound=snapshot_binding(case,outcome)
    bound[case.parent.parent/'result.json']=sha(case.parent.parent/'result.json')
    for name in ('outcome.json','result.json','expert_trace.jsonl','sensors/frames.jsonl',
                 'sensors/camera.json','sensors/body_poses.jsonl','world_from_left_base.json','oracle_query.json'):
        bound[case/name]=sha(case/name)
    physics=[json.loads(x) for x in (case/'expert_trace.jsonl').open()]
    frames=[json.loads(x) for x in (case/'sensors/frames.jsonl').open()]
    ledger=[json.loads(x) for x in (case/'sensors/body_poses.jsonl').open()]
    statistics=bind_sensor_frames(physics,frames)
    from .recovery_labels import audit_query_labels
    query=read(case/'oracle_query.json')
    if sha(case/'oracle_query.json')!=outcome['oracle_receipt']['query_sha256']:
        raise ValueError('Actual oracle receipt binding changed')
    labels=audit_query_labels(query,physics)
    query_frame=[r for r in frames if r['physics_row']==labels['query_physics_row']]
    if (len(query_frame)!=1 or query_frame[0]['target_estimate']['revision']
            !=query['measured_query']['context']['revision']):
        raise ValueError('Oracle query revision differs from acquired camera frame')
    metadata=read(case/'sensors/camera.json')
    camera_path=root/'sim/camera_framing_20260923/recommended_camera.json'
    annotation_path=root/'sim/auscultation_vla_v2/sensor_pilot_annotation.json'
    approved=read(camera_path);annotation=read(annotation_path)
    if (metadata['camera_sha256']!=sha(camera_path) or metadata['annotation_sha256']!=sha(annotation_path)
            or metadata['calibration_sha256']!=default_calibration().sha256
            or metadata['audio']!='absent' or metadata['depth_semantics']!='metric_optical_z'
            or metadata['target_source']!='rgbd_surface_tracking' or metadata['hall_condition']!=DEFAULT):
        raise ValueError('Approved camera, Hall condition or modalities changed')
    width,height=metadata['width'],metadata['height'];original=approved['intrinsics']
    if type(width) is not int or type(height) is not int or min(width,height)<2:
        raise ValueError('Invalid capture dimensions')
    scale=width/original['width']
    K=np.array([[original['fx']*scale,0,original['cx']*scale],
                [0,original['fy']*scale,original['cy']*scale],[0,0,1.]])
    base=np.asarray(read(case/'world_from_left_base.json'))
    transform=np.linalg.inv(base)@np.asarray(approved['world_from_optical'])
    if (height!=round(original['height']*scale)
            or not np.allclose(metadata['intrinsics'],K,atol=1e-10,rtol=0)
            or not np.allclose(metadata['camera_to_base'],transform,atol=1e-10,rtol=0)):
        raise ValueError('Capture calibration differs from approved camera')
    seeds={k:np.asarray(v)*scale for k,v in annotation['target_pixels'].items()}
    polygon=np.asarray(annotation['support_polygon_px'])*scale
    raw_names=[r['raw_file'] for r in frames]
    if (raw_names!=[f'{i:05d}.npz' for i in range(len(frames))]
            or {p.name for p in (case/'sensors').glob('*.npz')}!=set(raw_names)
            or len(ledger)!=len(frames)):
        raise ValueError('Missing, duplicated or unlogged acquired raw frames')
    tracker=BodyTracker(K,transform);kin=OpenArmKinematics();maximum=0.;depth_counts=[]
    for row,body in zip(frames,ledger):
        if time.monotonic()>deadline: raise TimeoutError('Raw replay time budget exhausted')
        raw=case/'sensors'/row['raw_file']
        if raw.is_symlink() or sha(raw)!=row['raw_sha256']:
            raise ValueError('Raw RGB-D identity changed')
        bound[raw]=row['raw_sha256']
        with np.load(raw,allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        validate_pixels(arrays,(height,width))
        rgb,depth,valid=[arrays[k] for k in ('rgb','depth_m','depth_valid')]
        depth_counts.append(int(valid.sum()));stamp=row['time_s']
        clocks=dict(rgb_time_s=stamp,depth_time_s=stamp,now_s=stamp)
        estimates=(tracker.initialize(rgb,depth,valid,seeds,polygon,**clocks) if row['index']==0
                   else tracker.update(rgb,depth,valid,**clocks))
        estimate=estimates[row['target']];recorded=row['target_estimate']
        for name,value in asdict(estimate).items():
            old=recorded[name]
            if isinstance(value,np.ndarray):
                if not np.allclose(old,value,atol=1e-5,rtol=0):
                    raise ValueError('Target geometry differs from raw RGB-D replay: '+name)
            elif isinstance(value,float):
                if not np.isclose(old,value,atol=1e-8,rtol=0):
                    raise ValueError('Target scalar differs from raw RGB-D replay: '+name)
            elif old!=value: raise ValueError('Target provenance differs: '+name)
        if (body['frame_index']!=row['index'] or body['time_s']!=stamp
                or not np.allclose(body['body_reference_to_current_base'],
                    transform@tracker._T@np.linalg.inv(transform),atol=1e-8,rtol=0)):
            raise ValueError('Body pose ledger differs from actual image registration')
        q,dq=np.asarray(row['q']),np.asarray(row['dq']);wrist=kin.fk(q)
        robot=RobotState(q,dq,wrist[:3,3],wrist[:3,:3],HallPacket(**row['hall']),stamp)
        obs=Observation(rgb,depth,valid,K,transform,stamp,stamp,robot,estimate,INSTRUCTIONS[row['phase']])
        state=encode(obs,now_s=stamp,hall_calibration=default_calibration())['state']
        captured=np.asarray(row['state']);error=float(np.max(abs(state-captured)))
        maximum=max(maximum,error)
        if captured.shape!=(32,) or not np.isfinite(captured).all() or not np.allclose(state,captured,atol=1e-5,rtol=0):
            raise ValueError('Encoded observation differs from raw measured modalities')
        if row['index']%200==0:
            print(json.dumps(dict(case=case.name,replayed_frames=row['index']+1,total=len(frames))),flush=True)
    for path,digest in bound.items():
        if sha(path)!=digest: raise ValueError('Source changed during read-only replay: '+str(path))
    return dict(passed=True,raw_frames_replayed=len(frames),maximum_state_error=maximum,
        audit_source_sha256=sha(__file__),
        minimum_valid_depth_pixels=min(depth_counts),sensor_bindings=statistics,
        first_stationary_query_label_audit=labels,
        runtime=dict(python=sys.version,numpy=np.__version__,opencv=cv2.__version__,scipy=scipy.__version__),
        source_hashes={str(k):v for k,v in bound.items()},
        source_case_passed=result['case_passed'],m0_passed=False,training_dataset_eligible=False,
        learned_policy=False,labels_exported=0,
        scope='Raw modality reproducibility only; no full dataset qualification, model sensor-use or generalization proof')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Owned CPU-only Linux audit required')
    config=read(args.config)
    if config['experiment']!='recovery_sensor_audit' or not 1<=len(config['cases'])<=3:
        raise ValueError('Bounded explicit cases required')
    args.output.mkdir(exist_ok=False,parents=True);reports=[];deadline=time.monotonic()+1800
    for item in config['cases']:
        case=Path(item['path']).resolve();owner=case.parent.parent/'result.json'
        if sha(owner)!=item['owner_sha256'] or sha(case/'result.json')!=item['case_sha256']:
            raise ValueError('Selected actual case changed')
        result=dict(passed=False,m0_passed=False,training_dataset_eligible=False)
        try: result=audit_case(case,deadline)
        except Exception as error: result['failure_reason']=type(error).__name__+': '+str(error)
        name=case.parent.parent.name+'--'+case.name
        write(args.output/(name+'.json'),result)
        reports.append(dict(case=str(case),passed=result['passed'],failure_reason=result.get('failure_reason')))
    write(args.output/'COMPLETE.json',dict(completed=True,all_passed=all(r['passed'] for r in reports),
        cases=reports,m0_passed=False,training_dataset_eligible=False,model_loaded=False))
    print(json.dumps(reports),flush=True)


if __name__=='__main__': main()
