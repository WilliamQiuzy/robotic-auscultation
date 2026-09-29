"""Append supplied, RGB-D-transported task poses to immutable sensor states.

This derives task inputs, not sensor data or expert actions. Original 32-channel
states and all other NPZ members remain bound to the qualified acquisition.
"""
import argparse
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from .export_joint_dataset import bound_path, json_bytes, read_json, sha256
from .task_observation import encode_task_state, observation_binding

SCHEMA='wam_registered_task_state_derivation_v1'
INPUTS=('sensors/frames.jsonl','sensors/body_poses.jsonl','task_frame_registration.json',
        'input_snapshot/manifest.json','outcome.json')
DEPENDENCIES=('source/urdf','source/vla/hall_calibration.json','source/code/kinematics.py',
              'source/code/tool_frames.py','source/vla/hall.py','source/vla/force_sensor.py')


def required_inputs(snapshot,registration):
    keys=list(DEPENDENCIES)+[name+'/plan' for name in registration]
    return set(INPUTS)|{'observed_supervisor_audit.json'}|{
        'input_snapshot/objects/'+snapshot['files'][key]['sha256'] for key in keys}


def qualified_audit(root,report,raw_root):
    """The derivation cannot substitute a new audit for the qualified one."""
    binding=report['evidence']['joint_action_derivation']['phase_derivation']['independent_supervisor_audit']
    audit=read_json(bound_path(root,binding['path'],binding['sha256']))
    raw_path=bound_path(raw_root,'observed_supervisor_audit.json',binding['sha256'])
    if (audit!=read_json(raw_path) or audit.get('schema')!='wam_observed_supervisor_replay_audit_v1'
            or audit.get('passed') is not True or audit.get('artifact_sha256')!=binding['artifact_sha256']
            or audit.get('audit_source_sha256')!=sha256(Path(__file__).with_name('audit_observed_supervisor.py'))):
        raise ValueError('Task source audit is not the original qualified replay')
    return audit,binding


def derive_states(raw_state,pairs,root):
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_vla_v2.hall import HallCalibration, HallPacket
    root=Path(root)
    frames=[json.loads(line) for line in (root/'sensors/frames.jsonl').read_text().splitlines()]
    bodies=[json.loads(line) for line in (root/'sensors/body_poses.jsonl').read_text().splitlines()]
    if ([f['index'] for f in frames]!=list(range(len(frames)))
            or len({b['frame_index'] for b in bodies})!=len(frames)):
        raise ValueError('Missing or duplicate acquired frame/body identity')
    body_map={b['frame_index']:b for b in bodies}
    registration=read_json(root/'task_frame_registration.json')
    snapshot=read_json(root/'input_snapshot/manifest.json')
    outcome=read_json(root/'outcome.json')
    if (registration!=outcome['task_frame_registration']
            or sha256(root/'input_snapshot/manifest.json')!=outcome['input_snapshot_manifest_sha256']):
        raise ValueError('Task registration/acquisition snapshot differs')
    def obj(key,module=None):
        digest=snapshot['files'][key]['sha256']
        path=bound_path(root,'input_snapshot/objects/'+digest,digest)
        if module and sha256(importlib.import_module(module).__file__)!=digest:
            raise ValueError('Task derivation dependency differs: '+key)
        return path
    for key,module in [('source/code/kinematics.py','sim.auscultation_expert_layout_v1.kinematics'),
        ('source/code/tool_frames.py','sim.auscultation_expert_layout_v1.tool_frames'),
        ('source/vla/hall.py','sim.auscultation_vla_v2.hall'),
        ('source/vla/force_sensor.py','auscultation.mechanical_sensor.force_feedback.force_sensor')]:obj(key,module)
    kin=OpenArmKinematics(obj('source/urdf'))
    cal=HallCalibration.from_path(obj('source/vla/hall_calibration.json'))
    first=frames[0];first_body=np.asarray(body_map[first['index']]['body_reference_to_current_base'],float)
    def close(a,b,label):
        a,b=np.asarray(a,float),np.asarray(b,float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=1e-8,rtol=0):
            raise ValueError('Task approval binding differs: '+label)
    for name,own in registration.items():
        if own['source_frame']!=first['index']:raise ValueError('Task must share first observed body frame')
        close(own['source_time_s'],first['time_s'],'registration time')
        close(own['initial_body_reference_to_current_base'],first_body,'registration body')
        initial=np.eye(4);initial[:3,3]=own['initial_task_position'];initial[:3,:3]=own['initial_task_rotation']
        close(first_body@np.asarray(own['task_pose_in_body_reference']),initial,'registration transform')
        plan=read_json(obj(name+'/plan'));base=np.asarray(plan['world_from_left_base'],float)
        point=next(p for p in plan['points'] if p['name']==name)
        approved_position=(np.linalg.inv(base)@np.r_[point['position_world'],1.])[:3]
        close(own['initial_task_position'],approved_position,'approved position')
        close(own['initial_task_rotation'],base[:3,:3].T@np.asarray(point['command_rotation_world']),'approved rotation')
    if len(raw_state)!=len(pairs):raise ValueError('Task state/pair count differs')
    result=[]
    for raw,pair in zip(raw_state,pairs):
        frame=frames[pair['frame_index']];body=body_map[frame['index']]
        if not np.array_equal(raw,np.asarray(frame['state'],np.float32)):
            raise ValueError('Raw state differs from qualified acquired frame')
        own=registration[frame['target']]
        result.append(encode_task_state(raw,measured_q=frame['q'],hall_packet=HallPacket(**frame['hall']),
            observation_time_s=frame['time_s'],now_s=frame['time_s'],
            body_reference_to_current_base=body['body_reference_to_current_base'],body_time_s=body['time_s'],
            task_pose_in_body_reference=own['task_pose_in_body_reference'],kinematics=kin,hall_calibration=cal))
    return np.asarray(result,np.float32)


def replace_state(parent,destination,state):
    payload=io.BytesIO();np.save(payload,state,allow_pickle=False)
    evidence={}
    with zipfile.ZipFile(parent) as source,zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            before=source.read(name);after=payload.getvalue() if name=='state.npy' else before
            target.writestr(name,after)
            evidence[name]=dict(parent_sha256=hashlib.sha256(before).hexdigest(),
                derived_sha256=hashlib.sha256(after).hexdigest(),unchanged=before==after)
    return evidence


def verify_task_derivation(root,record,report,arrays):
    root=Path(root)
    derivation=report.get('evidence',{}).get('task_observation_derivation',{})
    if (derivation.get('schema')!=SCHEMA or derivation.get('observation_contract')!=observation_binding()
            or derivation.get('source_sha256')!=sha256(__file__)
            or derivation.get('arrays_sha256')!=record['sha256']
            or report.get('checks',{}).get('task_observation_verified') is not True):
        raise ValueError('Missing or incompatible task observation derivation')
    parent=bound_path(root,derivation['parent_episode']['path'],derivation['parent_episode']['sha256'])
    evidence_root=root/derivation['evidence_root']
    if not evidence_root.resolve().is_relative_to(root.resolve()):raise ValueError('Task evidence escapes dataset')
    if not set(INPUTS)|{'observed_supervisor_audit.json'}<=set(derivation['input_sha256']):
        raise ValueError('Task input hash set omits required acquisition evidence')
    for name,digest in derivation['input_sha256'].items():bound_path(evidence_root,name,digest)
    required=required_inputs(read_json(evidence_root/'input_snapshot/manifest.json'),
                             read_json(evidence_root/'task_frame_registration.json'))
    if set(derivation['input_sha256'])!=required:raise ValueError('Incomplete task snapshot object bindings')
    audit,binding=qualified_audit(root,report,evidence_root)
    if derivation.get('qualified_supervisor_audit_sha256')!=binding['sha256']:
        raise ValueError('Task derivation replaced the qualified supervisor audit')
    for name,digest in derivation['input_sha256'].items():
        if name=='observed_supervisor_audit.json':expected=binding['sha256']
        else:expected=audit['artifact_sha256'].get(name)
        if digest!=expected:raise ValueError('Task input not bound by original qualified replay')
    pairs=report['evidence']['label_pairs']
    if hashlib.sha256(json_bytes(pairs)).hexdigest()!=derivation['label_pairs_sha256']:
        raise ValueError('Task label pair selection changed')
    with np.load(parent,allow_pickle=False) as original:
        if not np.array_equal(arrays['state'][:,:32],original['state']):
            raise ValueError('Task conditioning altered acquired sensor state')
        derived=derive_states(original['state'],pairs,evidence_root)
    if not np.array_equal(derived,arrays['state']):raise ValueError('Task observation cannot be independently reproduced')
    with zipfile.ZipFile(parent) as a,zipfile.ZipFile(root/record['path']) as b:
        if set(a.namelist())!=set(b.namelist()):raise ValueError('Task derivation altered array contract')
        for name in a.namelist():
            expected=derivation['array_members'][name]
            if (hashlib.sha256(a.read(name)).hexdigest()!=expected['parent_sha256']
                    or hashlib.sha256(b.read(name)).hexdigest()!=expected['derived_sha256']
                    or (name!='state.npy' and a.read(name)!=b.read(name))):
                raise ValueError('Non-state acquired array or member binding changed')
    return True


def derive_dataset(parent_manifest,trace_root,output):
    source=Path(parent_manifest).resolve();output=Path(output).resolve();trace_root=Path(trace_root).resolve()
    if output.exists() or output.is_relative_to(source.parent):raise ValueError('New external task dataset required')
    original={str(p.relative_to(source.parent)):sha256(p) for p in source.parent.rglob('*') if p.is_file()}
    manifest=read_json(source);shutil.copytree(source.parent,output)
    parent=output/'task_parent';parent.mkdir();shutil.copyfile(source,parent/'manifest.json')
    for record in manifest['episodes']:
        identity=record['episode_id'];raw=trace_root/('expert_rgbd_'+identity)
        report=read_json(output/record['qualification_path'])
        audit,binding=qualified_audit(output,report,raw)
        own=output/'task_evidence'/identity;own.mkdir(parents=True)
        snapshot=read_json(raw/'input_snapshot/manifest.json')
        names=sorted(required_inputs(snapshot,read_json(raw/'task_frame_registration.json')))
        inputs={}
        for name in names:
            digest=binding['sha256'] if name=='observed_supervisor_audit.json' else audit['artifact_sha256'][name]
            src=bound_path(raw,name,digest);dest=own/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest);inputs[name]=digest
        original_npz=output/record['path'];backup=parent/original_npz.name
        original_npz.rename(backup)
        shutil.copyfile(output/record['qualification_path'],parent/Path(record['qualification_path']).name)
        with np.load(backup,allow_pickle=False) as archive:
            derived=derive_states(archive['state'],report['evidence']['label_pairs'],own)
        members=replace_state(backup,original_npz,derived);new_hash=sha256(original_npz)
        derivation=dict(schema=SCHEMA,source_sha256=sha256(__file__),observation_contract=observation_binding(),
            parent_episode=dict(path=str(backup.relative_to(output)),sha256=record['sha256']),
            evidence_root=str(own.relative_to(output)),input_sha256=inputs,array_members=members,
            qualified_supervisor_audit_sha256=binding['sha256'],
            arrays_sha256=new_hash,label_pairs_sha256=hashlib.sha256(json_bytes(report['evidence']['label_pairs'])).hexdigest(),
            scope='Explicit supplied task pose transported only by acquired RGB-D body motion; original sensors/actions unchanged')
        report['evidence']['task_observation_derivation']=derivation
        report['checks']['task_observation_verified']=True
        report.update(arrays_sha256=new_hash,observation_contract=observation_binding())
        (output/record['qualification_path']).write_bytes(json_bytes(report))
        record.update(sha256=new_hash,qualification_sha256=sha256(output/record['qualification_path']))
    manifest['observation_contract']=observation_binding()
    (output/'manifest.json').write_bytes(json_bytes(manifest))
    for name,digest in original.items():
        if sha256(source.parent/name)!=digest:raise ValueError('Task derivation changed original dataset')
    (output/'task_derivation_receipt.json').write_bytes(json_bytes(dict(schema=SCHEMA,
        parent_manifest_sha256=sha256(source),manifest_sha256=sha256(output/'manifest.json'),
        episodes=len(manifest['episodes']),acquired_sources_unchanged=True,policy_acceptance=False)))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-manifest','trace-root','output'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();result=derive_dataset(args.parent_manifest,args.trace_root,args.output)
    print(json.dumps(dict(episodes=len(result['episodes']),observation_contract=result['observation_contract'])))
