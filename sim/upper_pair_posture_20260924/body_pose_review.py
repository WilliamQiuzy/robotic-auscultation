"""Recorded HOLD body-pose diagnostics; explicitly not an elbow-flip classifier.

python -B -m sim.upper_pair_posture_20260924.body_pose_review \
  --episode baseline=PATH --episode A=PATH --episode B=PATH --output REPORT.json
Full10 inputs use the same R1-start through R2-HOLD-end window by default.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sim.curobo_expert_v1.quality import validate_episode_provenance, verify_hashes

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(__file__).resolve().parent
TARGETS=tuple(side+str(level) for level in range(1,6) for side in ('R','L'))
LINKS={'shoulder':'openarm_left_link1','elbow':'openarm_left_link4','wrist':'openarm_left_link7'}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def _json(text):
    def bad(value):raise ValueError('nonfinite JSON number: '+value)
    return json.loads(text,parse_constant=bad)


def unit_quaternions(values):
    # Normalize AFTER float64 conversion: float32 acos(dot) creates spurious
    # approximately 0.04 degree increments in near-stationary HOLD samples.
    values=np.asarray(values,dtype=np.float64)
    require(values.ndim>=1 and values.shape[-1]==4 and np.isfinite(values).all(),'invalid quaternion array')
    norm=np.linalg.norm(values,axis=-1,keepdims=True)
    require(np.all(norm>0) and np.all(np.abs(norm-1)<1e-4),'nonunit/zero recorded quaternion')
    return values/norm


def rotation_angle_deg(a,b):
    """Shortest SO(3) angle; wxyz quaternion sign invariant and small-angle stable."""
    a,b=unit_quaternions(a),unit_quaternions(b)
    sign=np.where(np.sum(a*b,axis=-1,keepdims=True)<0,-1.,1.)
    return np.degrees(4*np.arcsin(np.clip(np.linalg.norm(a-sign*b,axis=-1)/2,0,1)))


def orientation_summary(quaternions,indices,dt,pairwise=False):
    q=unit_quaternions(quaternions);indices=np.asarray(indices)
    require(q.ndim==2 and len(q)>0 and len(q)==len(indices),'orientation sample/index mismatch')
    require(indices.ndim==1 and np.issubdtype(indices.dtype,np.integer)
            and np.all(np.diff(indices)>0),'invalid orientation sample order')
    _,vectors=np.linalg.eigh(q.T@q);mean=vectors[:,-1]
    if mean[0]<0:mean=-mean
    increments=rotation_angle_deg(q[1:],q[:-1])
    event=None
    if len(increments):
        i=int(increments.argmax())
        event=dict(angle_deg=float(increments[i]),from_physics_row=int(indices[i]),to_physics_row=int(indices[i+1]),
                   from_time_s=(int(indices[i])+1)*dt,to_time_s=(int(indices[i+1])+1)*dt)
    return dict(sample_count=len(q),mean_orientation_wxyz=mean.tolist(),
        maximum_pairwise_angle_deg=float(rotation_angle_deg(q[:,None,:],q[None,:,:]).max()) if pairwise else None,
        maximum_angle_from_first_deg=float(rotation_angle_deg(q,q[0]).max()),
        maximum_angle_from_mean_deg=float(rotation_angle_deg(q,mean).max()),
        sampled_path_length_deg=float(increments.sum()),maximum_increment_event=event,
        scope='Shortest SO(3) rotation between recorded link4 body orientations. '
              'Path length depends on sampling and noise; neither this nor actuator q4 classifies a spatial elbow flip.')


def read_prefix_trace(episode,outcome,hold_targets,prefix_end):
    dt=outcome['physics_dt_s'];require(type(dt) in (int,float) and abs(dt-1/240)<1e-12,'unexpected physics dt')
    points={p['target']:p for p in outcome['points']}
    require(all(t in points for t in (*hold_targets,prefix_end)),'requested review points absent from outcome')
    final=points[prefix_end];holds={t:[] for t in hold_targets};qs={t:[] for t in hold_targets}
    stop=None;last_hold=[]
    with (episode/'expert_trace.jsonl').open() as stream:
        for i,line in enumerate(stream):
            row=_json(line)
            require(type(row['time_s']) in (int,float) and abs(row['time_s']-(i+1)*dt)<=1e-7,'trace clock gap/origin mismatch')
            point=points.get(row['target'])
            require(point and point['start_physics_row']<=i<point['end_physics_row'],'trace point interval mismatch')
            if row['target']==prefix_end and row['phase']=='HOLD':last_hold.append(i)
            elif last_hold:
                stop=i;break
            if row['target'] in holds and row['phase']=='HOLD':
                value=np.asarray(row['q'],float)
                require(value.shape==(7,) and np.isfinite(value).all(),'invalid measured HOLD q')
                holds[row['target']].append(i);qs[row['target']].append(value)
    require(stop is not None and last_hold,'requested terminal HOLD did not complete')
    require(stop==last_hold[-1]+1 and len(last_hold)==last_hold[-1]-last_hold[0]+1,'noncontiguous terminal HOLD')
    all_holds=dict(holds);all_holds[prefix_end]=last_hold
    for target,indices in all_holds.items():
        require(indices and indices[-1]-indices[0]+1==len(indices),'missing/noncontiguous HOLD: '+target)
        expected=points[target].get('phase_config',outcome['phase_config'])['hold_duration_s']
        require(abs(len(indices)*dt-expected)<1e-7,'HOLD shorter/longer than recorded contract: '+target)
    return stop,holds,qs


def review_episode(path,hold_targets=('R1','L1'),prefix_end='R2'):
    path=Path(path).resolve();runner_path=path/'runner_summary.json'
    runner_hash=sha(runner_path);runner=_json(runner_path.read_text())
    require(type(runner.get('child_returncode')) is int,'runner has no completed child return code')
    # Reject a partial copy before parsing a trace or opening its NPZ.
    for name in ('expert_trace.jsonl','expert_replay.npz','outcome.json'):
        require(sha(path/name)==runner.get('evidence_sha256',{}).get(name),'runner artifact hash mismatch: '+name)
    provenance=validate_episode_provenance(path)
    require(provenance.get('status')=='passed','episode provenance rejected: '+str(provenance.get('reasons')))
    outcome=_json((path/'outcome.json').read_text());dt=outcome['physics_dt_s']
    require(outcome['requested_points'][:TARGETS.index(prefix_end)+1]==list(TARGETS[:TARGETS.index(prefix_end)+1]),'noncanonical requested prefix')
    end,holds,qs=read_prefix_trace(path,outcome,hold_targets,prefix_end)
    with np.load(path/'expert_replay.npz',allow_pickle=False) as replay:
        indices=replay['row_indices'];paths=list(map(str,replay['paths']));poses=replay['poses']
        require(indices.ndim==1 and np.issubdtype(indices.dtype,np.integer)
                and np.array_equal(indices,np.arange(7,outcome['physics_steps'],8)),'noncanonical replay cadence/coverage')
        require(poses.shape==(len(indices),len(paths),7) and np.isfinite(poses).all(),'invalid replay body poses')
        require(len(set(paths))==len(paths),'duplicate replay body path')
        unit_quaternions(poses[:,:,3:])
        body_indices={}
        for label,link in LINKS.items():
            found=[i for i,p in enumerate(paths) if p.endswith('/'+link)]
            require(len(found)==1,'missing/ambiguous replay landmark '+link);body_indices[label]=found[0]
        result={}
        for target,rows in holds.items():
            mask=(indices>=rows[0])&(indices<=rows[-1]);sample_rows=indices[mask]
            require(len(sample_rows)>0,'HOLD has no saved rigid body samples')
            bodies={name:poses[mask,k].astype(np.float64) for name,k in body_indices.items()}
            q=np.asarray(qs[target],dtype=np.float64)
            geometry={name:dict(body_path=paths[body_indices[name]],median_xyz_m=np.median(v[:,:3],axis=0).tolist(),
                               minimum_z_m=float(v[:,2].min()),maximum_z_m=float(v[:,2].max())) for name,v in bodies.items()}
            gap=bodies['wrist'][:,2]-bodies['elbow'][:,2];minimum=int(gap.argmin())
            result[target]=dict(physics_start_row=rows[0],physics_end_row_exclusive=rows[-1]+1,
                interval_start_s=rows[0]*dt,interval_end_s=(rows[-1]+1)*dt,hold_duration_s=len(rows)*dt,
                measured_q_sample_count=len(q),replay_body_sample_count=len(sample_rows),
                median_measured_q_rad=np.median(q,axis=0).tolist(),median_measured_q_deg=np.degrees(np.median(q,axis=0)).tolist(),
                recorded_body_landmarks=geometry,
                median_wrist_minus_elbow_z_m=float(np.median(gap)),minimum_wrist_minus_elbow_z_m=float(gap[minimum]),
                minimum_wrist_gap_event=dict(physics_row=int(sample_rows[minimum]),time_s=(int(sample_rows[minimum])+1)*dt),
                median_elbow_above_shoulder_z_m=float(np.median(bodies['elbow'][:,2]-bodies['shoulder'][:,2])),
                link4_orientation=orientation_summary(bodies['elbow'][:,3:],sample_rows,dt,True))
        mask=indices<end
        prefix=orientation_summary(poses[mask,body_indices['elbow'],3:],indices[mask],dt)
    require(sha(runner_path)==runner_hash and not verify_hashes(path,provenance['artifact_sha256']),
            'source changed during body-pose review')
    return dict(episode=str(path),runner_sha256=runner_hash,provenance=provenance,
        recorded_runner_child_returncode=runner['child_returncode'],recorded_runner_accepted=runner.get('accepted'),
        recorded_outcome_success=outcome.get('success'),recorded_training_accepted=outcome.get('training_accepted'),
        matched_window=dict(definition='R1 first row through '+prefix_end+' completed HOLD end; terminal home and later points excluded',
                            start_row=0,end_row_exclusive=end,interval_start_s=0.,interval_end_s=end*dt,duration_s=end*dt),
        HOLD=result,matched_window_link4_orientation=prefix)


def create_review(episodes,output,hold_targets=('R1','L1'),prefix_end='R2'):
    require('baseline' in episodes and len(episodes)>1,'baseline and at least one alternative label required')
    require(prefix_end in TARGETS and hold_targets and all(t in TARGETS[:TARGETS.index(prefix_end)+1] for t in hold_targets),
            'HOLD targets must lie within the chosen prefix')
    output=Path(output).resolve();episode_paths={label:Path(path).resolve() for label,path in episodes.items()}
    require(all(output!=path and path not in output.parents for path in episode_paths.values()),'output must be outside episodes')
    require((PROJECT/'sim') not in output.parents or ROOT in output.parents,'output must not modify frozen packages')
    sources=[Path(__file__),PROJECT/'sim/curobo_expert_v1/quality.py',PROJECT/'sim/curobo_expert_v1/provenance.py',
             PROJECT/'sim/expert_quality_20260924/quality.py']
    source_hashes={str(p.resolve()):sha(p) for p in sources}
    reviews={label:review_episode(path,hold_targets,prefix_end) for label,path in episode_paths.items()}
    baseline=reviews['baseline'];comparisons={}
    for label,review in reviews.items():
        if label=='baseline':continue
        comparison={}
        for target in hold_targets:
            b,a=baseline['HOLD'][target],review['HOLD'][target]
            comparison[target]=dict(
                delta_median_q_rad=(np.asarray(a['median_measured_q_rad'])-b['median_measured_q_rad']).tolist(),
                delta_median_elbow_z_m=a['recorded_body_landmarks']['elbow']['median_xyz_m'][2]-b['recorded_body_landmarks']['elbow']['median_xyz_m'][2],
                delta_minimum_wrist_minus_elbow_z_m=a['minimum_wrist_minus_elbow_z_m']-b['minimum_wrist_minus_elbow_z_m'],
                HOLD_mean_link4_orientation_difference_deg=float(rotation_angle_deg(a['link4_orientation']['mean_orientation_wxyz'],b['link4_orientation']['mean_orientation_wxyz'])))
        comparison['matched_window_delta_sampled_link4_path_deg']=review['matched_window_link4_orientation']['sampled_path_length_deg']-baseline['matched_window_link4_orientation']['sampled_path_length_deg']
        comparisons[label]=comparison
    for label,review in reviews.items():
        require(sha(episode_paths[label]/'runner_summary.json')==review['runner_sha256']
                and not verify_hashes(episode_paths[label],review['provenance']['artifact_sha256']),
                'episode changed before report save: '+label)
    require(all(sha(path)==digest for path,digest in source_hashes.items()),'analysis source changed')
    report=dict(schema='upper_pair_recorded_body_pose_review_v1',episodes=reviews,alternative_minus_baseline=comparisons,
        analysis_sources=source_hashes,numpy_version=np.__version__,
        joint_order='left URDF joints 1..7; q4 is actuator flexion and is not a spatial elbow-flip label',
        geometry='World-frame recorded rigid bodies: link1 shoulder, link4 elbow, link7 wrist; no new FK or interpolation.',
        rotations='Recorded wxyz quaternions normalized in float64; shortest sign-invariant SO(3) angle by quaternion difference; Markley eigenvector mean.',
        sampling='240 Hz measured q and 30 Hz body poses; no inference about unsampled intermediate body orientations.',
        physics_acceptance_inferred=None,elbow_flip_classifier=None,
        scope='Descriptive spatial posture and sampled orientation diagnostics, not a clinical/comfort/assembly threshold or a proof that another posture is impossible.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',action='append',required=True,metavar='LABEL=PATH')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hold-targets',nargs='+',default=['R1','L1'])
    parser.add_argument('--prefix-end',default='R2',choices=TARGETS)
    args=parser.parse_args();episodes={}
    for entry in args.episode:
        label,separator,path=entry.partition('=')
        require(separator and label and path and label not in episodes,'use unique nonempty LABEL=PATH arguments')
        episodes[label]=Path(path)
    result=create_review(episodes,args.output,tuple(args.hold_targets),args.prefix_end)
    print(json.dumps({'output':str(args.output.resolve()),'labels':list(result['episodes']),
                      'scope':'recorded body-pose metrics only; not an elbow-flip classifier'}))


if __name__=='__main__':main()
