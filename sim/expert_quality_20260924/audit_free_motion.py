"""Audit saved free-motion generator claims independently of quality averages.

This module never imports or changes the candidate generator. Actual forward
finite differences use the saved 240 Hz command stream, including adjacent
free-segment boundaries. Contact IK windows are explicitly excluded and counted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .quality import sha256_file, validate_episode_provenance, verify_hashes

DT=1/240.
FREE_SUBPHASES=frozenset(('initial_home','transfer','inter_point_transfer','lower_approach','return_bridge','return_home'))
STATIC_SUBPHASES=frozenset(('settle','align','home_settle'))
METRICS=(('velocity_rad_s',1,.25,'velocity_limit_rad_s','joint_velocity_rad_s'),
         ('acceleration_rad_s2',2,2.,'acceleration_limit_rad_s2','joint_acceleration_rad_s2'),
         ('jerk_rad_s3',3,20.,'jerk_limit_rad_s3','joint_jerk_rad_s3'))


def _scalar(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
        raise ValueError('invalid finite number:'+name)
    return float(value)


def _peak(values,eligible,times,segment_boundary):
    selected=np.flatnonzero(eligible)
    if not len(selected):return None
    absolute=np.abs(values[selected]);local,joint=np.unravel_index(absolute.argmax(),absolute.shape)
    index=int(selected[local]);order=len(times)-len(values)
    return dict(value=float(absolute[local,joint]),signed_value=float(values[index,joint]),joint=int(joint+1),
        start_row=index-1,end_row=index+order-1,time_s=float(times[index+order]),
        start_time_s=float(times[index]),cross_segment_boundary=bool(segment_boundary[index]))


def analyze_free_motion(rows,outcome,initial_q=None,source_binding=None):
    """Pure saved-row numerical analysis; source binding is supplied by file audit.

    `initial_q` means the retained command BEFORE physics row 0, not measured q.
    Omitting it leaves the first padded segment pending and excludes that single
    invented predecessor from real-stream windows; zero is never assumed.
    """
    if not isinstance(outcome,dict) or not isinstance(rows,list) or len(rows)<4:
        raise ValueError('invalid outcome or insufficient trace')
    q=np.asarray([row['q_command'] for row in rows],float)
    times=np.asarray([row['time_s'] for row in rows],float);n=len(rows)
    if q.shape!=(n,7) or times.shape!=(n,) or not np.isfinite(q).all() or not np.isfinite(times).all():
        raise ValueError('nonfinite or malformed saved commands/time')
    if not np.allclose(times,np.arange(1,n+1)*DT,atol=1e-7,rtol=0):raise ValueError('timestamp gap or origin')
    subphase=np.asarray([row['subphase'] for row in rows]);targets=np.asarray([row['target'] for row in rows])
    if any(not isinstance(x,str) for x in list(subphase)+list(targets)):raise ValueError('invalid labels')
    initial_available=initial_q is not None
    if initial_available:
        initial_q=np.asarray(initial_q,float)
        if initial_q.shape!=(7,) or not np.isfinite(initial_q).all():raise ValueError('invalid initial retained command')
    else:initial_q=q[0].copy() # excluded, never evidence for a first-segment padded check
    report=dict(schema='saved_free_motion_audit_v1',failures=[],pending=[],segments=[],actual_windows={},
        derivative_method='forward finite differences on saved commands, no filtering and no unwrap',
        dt_s=DT,limits={name:limit for name,_,limit,_,_ in METRICS},
        scope='saved free-motion command bounds only; no C2, measured-joint, collision, clinical or contact-IK acceptance')
    failures=report['failures'];pending=report['pending'];segment_ids=np.full(n,-1,int)
    expected_free=np.isin(subphase,list(FREE_SUBPHASES))
    is_candidate=any(k in outcome for k in ('candidate_policy','candidate_method','free_motion_segments'))
    records=outcome.get('free_motion_segments')
    declared=isinstance(records,list) and bool(records)
    if is_candidate and not declared:failures.append('candidate_missing_free_motion_segments')
    if declared:
        report['metadata_scope']='candidate_declared_segments'
        source_binding=source_binding or {'status':'pending','reason':'no file-level source binding supplied to pure analyzer'}
        if source_binding.get('status')=='failed':failures.append('candidate_source_binding_failed')
        elif source_binding.get('status')!='passed':pending.append('candidate_source_binding_not_verified')
        bounds=outcome.get('free_motion_command_bounds',{})
        for name,_,limit,_,top_key in METRICS:
            try:
                value=_scalar(bounds[top_key],top_key)
                if not 0<value<=limit+1e-9:raise ValueError('declared bound weaker than fixed audit limit')
                report['limits'][name]=min(value,limit)
            except (KeyError,TypeError,ValueError) as error:failures.append('invalid_declared_bound:'+name+':'+str(error))
    else:
        report['metadata_scope']='subphase_only_baseline_diagnostic'
        report['controller_claims_verified']=False
        pending.append('segment_metadata_missing_no_controller_peak_or_count_claims_can_be_verified')
        records=[];start=None
        for i in range(n+1):
            in_free=i<n and expected_free[i]
            if start is not None and (not in_free or (subphase[i],targets[i])!=(subphase[start],targets[start])):
                records.append(dict(start_physics_row=start,end_physics_row_exclusive=i,subphase=str(subphase[start]),target=str(targets[start])))
                start=None
            if in_free and start is None:start=i
    report['source_binding']=source_binding or {'status':'not_applicable','scope':'baseline has no derived-controller claims'}
    previous_end=0
    for number,record in enumerate(records):
        entry=dict(segment_index=number,reported_peaks_match=None,padded_limits_passed=None,
            source_binding_reference='source_binding',source_binding_status=report['source_binding'].get('status'))
        report['segments'].append(entry)
        try:
            start=record['start_physics_row'];end=record['end_physics_row_exclusive']
            if type(start)!=int or type(end)!=int or not 0<=start<end<=n or start<previous_end:
                raise ValueError('invalid segment bounds, order, overlap or missing completion')
            previous_end=end;segment_ids[start:end]=number
            entry.update(start_row=start,end_row_exclusive=end,start_time_s=float(times[start]),end_time_s=float(times[end-1]),
                saved_command_count=end-start,subphase=record.get('subphase'),target=record.get('target'),
                command_array_sha256=hashlib.sha256(q[start:end].astype('<f8').tobytes()).hexdigest(),
                command_digest_format='little-endian float64 row-major (samples, 7), radians')
            if (record.get('subphase') not in FREE_SUBPHASES or not np.all(subphase[start:end]==record.get('subphase'))
                    or not np.all(targets[start:end]==record.get('target'))):
                raise ValueError('segment label differs from saved trace or includes contact IK')
            segment_limits=dict(report['limits'])
            if declared:
                if record.get('motion_complete') is not True:raise ValueError('segment motion incomplete')
                dt=_scalar(record.get('physics_dt_s'),'physics_dt_s')
                if not np.isclose(dt,DT,atol=1e-12,rtol=0) or record.get('endpoint_stationary_padding_samples')!=3:
                    raise ValueError('segment derivative method differs from audited generator claim')
                duration=_scalar(record.get('duration_s'),'duration_s');expected_count=duration/DT
                entry['duration_derived_command_count']=expected_count
                if not np.isclose(expected_count,end-start,atol=1e-7,rtol=0) or (end-start)%12:
                    failures.append('segment_'+str(number)+':command_count_mismatch')
                for name,_,limit,record_key,_ in METRICS:
                    bound=_scalar(record.get(record_key),record_key)
                    if not 0<bound<=report['limits'][name]+1e-9:raise ValueError('segment bound differs from global declared/fixed bound:'+name)
                    segment_limits[name]=min(bound,report['limits'][name])
            if start==0 and not initial_available:
                pending.append('initial_command_before_first_row_unavailable');continue
            retained=initial_q if start==0 else q[start-1]
            padded=np.vstack([np.repeat(retained[None],3,axis=0),q[start:end],np.repeat(q[end-1:end],3,axis=0)])
            peaks={name:float(np.abs(np.diff(padded,n=order,axis=0)).max()/DT**order) for name,order,_,_,_ in METRICS}
            entry['recomputed_padded_peaks']=peaks
            entry['effective_segment_limits']=segment_limits
            entry['padded_limits_passed']=all(peaks[name]<=segment_limits[name]+1e-9 for name,_,_,_,_ in METRICS)
            if not entry['padded_limits_passed']:failures.append('segment_'+str(number)+':padded_command_limit')
            if declared:
                reported=record['peaks'];match=True
                for name,_,_,_,_ in METRICS:
                    value=_scalar(reported[name],name)
                    if not np.isclose(value,peaks[name],atol=1e-7,rtol=1e-8):match=False
                entry['reported_peaks_match']=match;entry['reported_peaks']=reported
                if not match:failures.append('segment_'+str(number)+':reported_peak_mismatch')
        except (KeyError,TypeError,ValueError) as error:
            failures.append('segment_'+str(number)+':'+str(error));entry['metadata_error']=str(error)
    uncovered=expected_free & (segment_ids<0)
    report['uncovered_free_subphase_row_count']=int(uncovered.sum())
    if declared and uncovered.any():failures.append('uncovered_free_subphase_rows')
    prior=np.vstack([initial_q,q[:-1]])
    static=np.isin(subphase,list(STATIC_SUBPHASES))
    stationary=static & np.all(np.abs(q-prior)<=1e-10,axis=1)
    if not initial_available and static[0]:stationary[0]=False
    if np.any(static & ~stationary):failures.append('nonstationary_command_in_free_hold_subphase')
    free=expected_free | stationary
    command=np.vstack([initial_q,q]);time=np.r_[times[0]-DT,times]
    initial_free=initial_available and bool(free[0])
    free=np.r_[initial_free,free]
    ids=np.r_[segment_ids[0] if initial_free else -1,segment_ids]
    boundary=(ids[1:]>=0)&(ids[:-1]>=0)&(ids[1:]!=ids[:-1])
    report['free_dynamic_row_count']=int(expected_free.sum())
    report['free_stationary_row_count']=int(stationary.sum())
    report['excluded_nonfree_row_count']=int(n-expected_free.sum()-stationary.sum())
    nonfree=~(expected_free|stationary)
    report['excluded_nonfree_rows_by_subphase']={str(label):int(np.count_nonzero(nonfree & (subphase==label))) for label in np.unique(subphase[nonfree])}
    report['initial_retained_command_available']=initial_available
    report['boundary_policy']='all actual windows consisting solely of named free commands or verified stationary free holds; any contact/other command excludes its overlapping windows'
    report['worst_events']=[]
    for name,order,_,_,_ in METRICS:
        values=np.diff(command,n=order,axis=0)/DT**order
        free_counts=np.convolve(free.astype(int),np.ones(order+1,dtype=int),'valid')
        eligible=free_counts==order+1;mixed=(free_counts>0)&~eligible
        cross=np.convolve(boundary.astype(int),np.ones(order,dtype=int),'valid')>0
        limit=report['limits'][name];violated=eligible & np.any(np.abs(values)>limit+1e-9,axis=1)
        peak=_peak(values,eligible,time,cross);cross_peak=_peak(values,eligible&cross,time,cross)
        metric=dict(order=order,limit=limit,total_windows=len(values),evaluated_windows=int(eligible.sum()),
            evaluated_cross_segment_windows=int((eligible&cross).sum()),excluded_mixed_free_and_nonfree_windows=int(mixed.sum()),
            excluded_all_nonfree_windows=int((free_counts==0).sum()),violation_window_count=int(violated.sum()),
            cross_segment_violation_window_count=int((violated&cross).sum()),peak=peak,cross_segment_peak=cross_peak,
            max_abs_per_joint=np.abs(values[eligible]).max(axis=0).tolist() if eligible.any() else None)
        report['actual_windows'][name]=metric
        if violated.any():
            failures.append('actual_free_motion_'+name+'_limit')
            report['worst_events'].append(dict(peak,reason=name+'_limit',limit=limit))
            if cross_peak and cross_peak['value']>limit+1e-9 and (not peak['cross_segment_boundary']):
                report['worst_events'].append(dict(cross_peak,reason='cross_segment_'+name+'_limit',limit=limit))
        if not eligible.any():pending.append('no_evaluable_free_windows:'+name)
    failures[:]=list(dict.fromkeys(failures));pending[:]=list(dict.fromkeys(pending))
    report['controller_claims_verified']=bool(declared and not failures and not pending)
    report['numerical_status']='failed' if failures else 'pending' if pending else 'passed'
    report['status']=report['numerical_status'] if is_candidate else 'baseline_diagnostic'
    return report


def _initial_retained_command(path):
    """Read only the first control JSON object, avoiding the full command log."""
    decoder=json.JSONDecoder();buffer=''
    with Path(path).open() as stream:
        while len(buffer)<1024*1024:
            chunk=stream.read(8192)
            if not chunk:break
            buffer+=chunk;content=buffer.lstrip()
            if not content.startswith('['):raise ValueError('control_commands must be a list')
            try:first,_=decoder.raw_decode(content[1:].lstrip())
            except json.JSONDecodeError:continue
            return first['previous_q_command']
    raise ValueError('first retained control command unavailable or exceeds read limit')


def audit_free_motion(episode_path,output_path):
    """Validate provenance, stream only needed trace fields, and write outside input."""
    source=Path(episode_path).resolve();output=Path(output_path).resolve()
    if output==source or source in output.parents:raise ValueError('output must be outside input directory')
    own_hash=sha256_file(__file__);quality_source=Path(__file__).with_name('quality.py');quality_hash=sha256_file(quality_source)
    provenance=validate_episode_provenance(source)
    report=dict(schema='saved_free_motion_audit_v1',status='failed',failures=[],pending=[],provenance=provenance)
    try:
        if provenance['status']!='passed':raise ValueError('saved artifact/source provenance failed')
        outcome=json.loads((source/'outcome.json').read_text())
        rows=[]
        with (source/'expert_trace.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line);rows.append({key:row[key] for key in ('time_s','q_command','phase','subphase','target')})
        initial=_initial_retained_command(source/'control_commands.json')
        binding=provenance.get('candidate_source_binding',{})
        if binding.get('status')=='passed':
            manifest=json.loads((source/outcome['candidate_sources_manifest']).read_text())
            binding=dict(binding,candidate_source_sha256={name:item['sha256'] for name,item in manifest['files'].items()},
                scope='each segment executed by the same controller/planner/runner sources bound to the episode snapshot; no separate per-segment source hash is declared')
        report=analyze_free_motion(rows,outcome,initial_q=initial,source_binding=binding)
        report['provenance']=provenance
    except (OSError,ValueError,KeyError,TypeError) as error:report['failures'].append('audit_unavailable:'+str(error));report['status']='failed'
    changed=verify_hashes(source,provenance.get('artifact_sha256',{}))
    report['failures'].extend(changed)
    if sha256_file(__file__)!=own_hash or sha256_file(quality_source)!=quality_hash:report['failures'].append('audit_source_changed')
    if changed or 'audit_source_changed' in report['failures']:
        report['status']='failed';report['controller_claims_verified']=False
    report['audit_sources']={'audit_free_motion.py':own_hash,'quality.py':quality_hash}
    report['episode']=str(source)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);report=audit_free_motion(args.episode,args.output)
    print(json.dumps({key:report.get(key) for key in ('status','numerical_status','metadata_scope','controller_claims_verified','failures','pending')},indent=2))
    return 1 if report['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
