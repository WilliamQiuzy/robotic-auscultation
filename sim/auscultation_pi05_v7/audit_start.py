"""Independent measured-start audit over immutable qualified raw acquisitions.

This does not qualify sensor/mechanics data or generate observations. It binds
the first actual selected label pair to the frozen start-family experiment and
fingerprints numeric sensor content independently of target/episode metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NOMINAL_FAMILY='legacy_nominal_v6_diagnostic'
PREPARATION={'initial_home','settle'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def read(path):return json.loads(Path(path).read_text())


def bound(root,name,digest):
    root=Path(root).resolve();relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Relative bound artifact required')
    p=root/relative
    if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file() or sha(p)!=digest:
        raise ValueError('Artifact/source changed: '+str(name))
    return p


def q(value):
    x=np.asarray(value,float)
    if x.shape!=(7,) or not np.isfinite(x).all():raise ValueError('Finite seven-joint vector required')
    return x


def numeric_fingerprint(root,frames):
    """Actual RGB, metric depth, valid mask, q/dq, Hall and times, no IDs."""
    h=hashlib.sha256()
    for frame in frames:
        p=bound(root/'sensors',frame['raw_file'],frame['raw_sha256'])
        with np.load(p,allow_pickle=False) as arrays:
            if set(arrays.files)!={'rgb','depth_m','depth_valid'}:raise ValueError('Raw sensor fields changed')
            for name in ('rgb','depth_m','depth_valid'):
                a=np.asarray(arrays[name])
                if a.dtype.hasobject:raise ValueError('Object sensor array')
                h.update(canonical(dict(channel=name,dtype=a.dtype.str,shape=list(a.shape))))
                h.update(a.tobytes(order='C'))
        h.update(canonical({key:frame[key] for key in ('time_s','q','dq','hall')}))
    return h.hexdigest()


def audit(episode,qualification,protocol_path,start_family,receipt_path,*,legacy=False):
    episode=Path(episode).resolve();qualification=Path(qualification).resolve()
    protocol_path=Path(protocol_path).resolve();receipt_path=Path(receipt_path).resolve()
    protocol=read(protocol_path);qualification_data=read(qualification)
    if protocol.get('schema')!='wam_v7_frozen_start_family_protocol_v1':raise ValueError('Frozen start protocol required')
    if protocol_path.read_bytes()!=canonical(protocol):raise ValueError('Protocol must use exact frozen canonical bytes')
    if protocol.get('start_auditor_sha256')!=sha(__file__):raise ValueError('Start auditor source changed')
    if (qualification_data.get('qualified') is not True or qualification_data.get('data_kind')!='closed_loop_rollout'
            or qualification_data.get('schema')!='wam_vla_episode_qualification_v1'):
        raise ValueError('Independent qualified sensor/mechanics acquisition required first')
    identity=qualification_data['episode_id'];evidence=qualification_data['evidence'];hashes=evidence['artifact_sha256']
    outcome=read(bound(episode,'outcome.json',hashes['outcome.json']))
    if len(outcome.get('requested_points',[]))!=1:raise ValueError('Single target acquisition required')
    target=outcome['requested_points'][0];case=protocol['cases'][start_family+'::'+target]
    families={r['family_id']:r for r in protocol['families']};family=families[start_family]
    if legacy!=(start_family==NOMINAL_FAMILY):raise ValueError('Explicit legacy provenance required')
    frames=[json.loads(x) for x in bound(episode,'sensors/frames.jsonl',hashes['sensors/frames.jsonl']).read_text().splitlines()]
    physics=[json.loads(x) for x in bound(episode,'expert_trace.jsonl',hashes['expert_trace.jsonl']).read_text().splitlines()]
    if [r['index'] for r in frames]!=list(range(len(frames))):raise ValueError('Sensor frame identities changed')
    if [r['physics_row'] for r in frames]!=list(range(11,len(physics),12)):raise ValueError('Sensor cadence differs')
    pairs=evidence['label_pairs']
    if not pairs:raise ValueError('No qualified label pairs')
    for pair in pairs:
        first_i,following_i=(pair.get(k) for k in ('frame_index','following_frame_index'))
        interval=pair.get('physics_interval_inclusive')
        if (type(first_i) is not int or type(following_i) is not int or
                not 0<=first_i<following_i<len(frames) or following_i!=first_i+1 or
                not isinstance(interval,list) or len(interval)!=2 or
                any(type(i) is not int for i in interval) or not 0<=interval[0]<interval[1]<len(physics) or
                interval!=[frames[first_i]['physics_row'],frames[following_i]['physics_row']] or
                interval[1]-interval[0]!=12 or pair.get('command_physics_row')!=interval[1]):
            raise ValueError('Qualified pair interval/endpoints differ from actual frames')
    first=frames[pairs[0]['frame_index']];following=frames[pairs[0]['following_frame_index']]
    if (following['index']!=first['index']+1 or first.get('eligible') is not True or
            following.get('eligible') is not True or first['phase']!='FREE' or
            first['physics_row']!=pairs[0]['physics_interval_inclusive'][0]):
        raise ValueError('First selected pair is not the actual eligible FREE start')
    earlier=[f for f in frames[:first['index']] if f.get('eligible')]
    if earlier:raise ValueError('Eligible acquisition preceded the recorded start')
    row=physics[first['physics_row']]
    if (first['time_s']!=row['time_s'] or not np.array_equal(q(first['q']),q(row['q'])) or
            not np.array_equal(q(first['dq']),q(row['dq']))):
        raise ValueError('Actual start state not bound to physics')
    snapshot_path=bound(episode,outcome['input_snapshot'],outcome['input_snapshot_manifest_sha256'])
    snapshot=read(snapshot_path);plan=snapshot['files'][target+'/plan']['sha256']
    if plan!=case['source_plan_sha256']:raise ValueError('Approved source plan differs')
    bound(snapshot_path.parent,'objects/'+plan,plan)
    if not legacy:
        if case['start_manifest_sha256'] not in {v['sha256'] for v in snapshot['files'].values()}:
            raise ValueError('Start geometry not snapshot bound')
        bound(snapshot_path.parent,'objects/'+case['start_manifest_sha256'],case['start_manifest_sha256'])
    first_record=dict(frame_index=first['index'],physics_row=first['physics_row'],time_s=first['time_s'],
        measured_q_rad=first['q'],measured_dq_rad_s=first['dq'],retained_command_q_rad=row['q_command'])
    if legacy:
        if receipt_path.exists():raise ValueError('New retrospective receipt required')
        receipt=dict(schema='wam_v7_legacy_start_replay_receipt_v1',episode_id=identity,target=target,
            start_family=start_family,experiment_protocol_sha256=sha(protocol_path),source_plan_sha256=plan,
            start_manifest_sha256=case['start_manifest_sha256'],sensor_frames_sha256=hashes['sensors/frames.jsonl'],
            first_eligible=first_record,provenance='Retrospective replay of preserved nominal acquisition; not a new collector receipt',
            auditor_source_sha256=sha(__file__))
    else:
        recorded=outcome['measured_start_receipt']
        original=bound(episode,recorded['path'],recorded['sha256'])
        if original!=receipt_path:raise ValueError('Must audit original producer receipt')
        receipt=read(original)
        if receipt.get('schema')!='wam_v7_measured_start_receipt_v1':raise ValueError('Wrong producer receipt schema')
    for key,expected in dict(episode_id=identity,target=target,start_family=start_family,
        experiment_protocol_sha256=sha(protocol_path),source_plan_sha256=plan,
        start_manifest_sha256=case['start_manifest_sha256'],sensor_frames_sha256=hashes['sensors/frames.jsonl']).items():
        if receipt.get(key)!=expected:raise ValueError('Measured receipt mismatch: '+key)
    for key,expected in first_record.items():
        if receipt.get('first_eligible',{}).get(key)!=expected:raise ValueError('First eligible measurement mismatch: '+key)
    if np.max(abs(q(first['q'])-q(family['requested_q_rad'])))>protocol['actual_start_tolerance_rad']+1e-12:
        raise ValueError('Measured start misses declared tolerance')
    if not legacy:
        prep,settle=(receipt.get(k) for k in ('preparation_physics_range','settle_physics_range'))
        for interval in (prep,settle):
            if (not isinstance(interval,list) or len(interval)!=2 or any(type(i) is not int for i in interval)
                    or not 0<=interval[0]<interval[1]<=first['physics_row']):
                raise ValueError('Invalid actual preparation interval')
        if prep[1]>settle[0]:raise ValueError('Preparation/settle overlap')
        if any(r['subphase']!='initial_home' for r in physics[prep[0]:prep[1]]):raise ValueError('Preparation not marked excluded')
        if any(r['subphase']!='settle' for r in physics[settle[0]:settle[1]]):raise ValueError('Settle not marked excluded')
        if np.max(abs(q(first['q'])))<=.03:raise ValueError('Declared nonzero start washed out to home')
        route=receipt.get('actual_start_route',{})
        planning=route.get('planning_physics_row');stop=route.get('end_physics_row_exclusive')
        if (route.get('planning_accepted') is not True or route.get('motion_complete') is not True or
                type(planning) is not int or type(stop) is not int or
                not first['physics_row']<planning<stop<=len(physics)):
            raise ValueError('Incomplete or invalid actual initial route interval')
        begin=settle[1]
        actual_planning=next((i for i in range(begin,len(physics))
            if physics[i].get('preparation_role')!='start_observation'),len(physics))
        actual_stop=next((i for i in range(actual_planning,len(physics))
            if physics[i].get('preparation_role')!='eligible_start_route'),len(physics))
        if (actual_planning!=planning or actual_stop!=stop or
                not begin<=first['physics_row']<planning or planning-begin!=12):
            raise ValueError('Actual initial route interval differs from per-tick preparation roles')
        for r in physics[begin:stop]:
            if np.max(abs(q(r['q_command'])))<=.03 or np.max(abs(q(r['q'])))<=.03:
                raise ValueError('Actual eligible route returned to home')
    for pair in pairs:
        start,end=pair['physics_interval_inclusive']
        if any(r['subphase'] in PREPARATION for r in physics[start:end+1]):
            raise ValueError('Preparation entered qualified training labels')
    fingerprint=numeric_fingerprint(episode,frames)
    if legacy:
        receipt_path.parent.mkdir(parents=True,exist_ok=True)
        with receipt_path.open('xb') as stream:stream.write(canonical(receipt))
    checks=dict(raw_artifacts_verified=True,first_eligible_bound_to_label_pairs=True,
        measured_state_recomputed=True,start_receipt_bound=True,source_plan_snapshot_bound=True,
        raw_sensor_content_recomputed=True,preparation_excluded_from_labels=True,
        eligible_route_no_home_washout=not legacy)
    return dict(schema='wam_v7_start_independent_audit_v1',episode_id=identity,target=target,start_family=start_family,
        passed=True,checks=checks,start_receipt_sha256=sha(receipt_path),qualification_sha256=sha(qualification),
        protocol_sha256=sha(protocol_path),auditor_source_sha256=sha(__file__),
        raw_sensor_content_sha256=fingerprint,raw_sensor_fingerprint_semantics='Numeric RGB/depth/valid arrays and q/dq/Hall/times in acquisition order; excludes IDs, labels, prompts and filenames',
        artifact_sha256={k:hashes[k] for k in ('outcome.json','sensors/frames.jsonl','expert_trace.jsonl')},
        frame_count=len(frames),legacy_retrospective=legacy,first_eligible=first_record,
        scope='Actual start ancestry only; independent sensor/mechanics qualification required; no learned-policy acceptance')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('episode','qualification','protocol','receipt','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--start-family',required=True);p.add_argument('--legacy',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh independent audit output required')
    result=audit(a.episode,a.qualification,a.protocol,a.start_family,a.receipt,legacy=a.legacy)
    with a.output.open('xb') as f:f.write(canonical(result))
    print(json.dumps(dict(passed=result['passed'],episode_id=result['episode_id'],audit_sha256=sha(a.output))))


if __name__=='__main__':main()
