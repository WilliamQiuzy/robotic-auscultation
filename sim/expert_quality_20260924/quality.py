"""CPU-only, read-only quality audit for immutable expert replay evidence.

No scalar score combines hard failures with motion preferences. The legacy
mechanical/posture gates are recomputed from samples, while additional limits
are explicitly a draft engineering profile. Anatomical/acoustic/assembly truth
cannot be established by these simulation logs.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.signal import savgol_filter

DEFAULT_PROFILE = Path(__file__).with_name('quality_profile.json')
FRONT = ('diaphragm_N', 'lip_N', 'contact_rim_N')
FORBIDDEN = ('carriage_other_N', 'frame_N', 'forbidden_N')
SCALARS = ('time_s', 'hall_N', 'compression_m', 'gap_m', 'tangent_m',
           'angle_deg', 'tcp_speed_m_s') + FRONT + FORBIDDEN
REQUIRED_ARTIFACTS = ('expert_trace.jsonl', 'outcome.json', 'expert_replay.npz',
                      'world_from_left_base.json', 'control_commands.json',
                      'independent_audit.json', 'posture_audit.json')


def load_profile(path=None):
    profile = json.loads(Path(path or DEFAULT_PROFILE).read_text())
    for key in ('reversal_excursion_deg', 'filter_window_s', 'joint_step_rad',
                'joint_velocity_rad_s', 'joint_acceleration_rad_s2', 'joint_jerk_rad_s3',
                'loaded_force_N', 'loaded_drag_max_event_m', 'loaded_drag_max_speed_m_s'):
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid profile: '+key)
    if type(profile.get('user_selected')) is not bool:
        raise ValueError('user_selected must be boolean')
    if profile.get('filter_polynomial_order') != 4:
        raise ValueError('filter_polynomial_order must be 4 for this audited method')
    return profile


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def verify_hashes(directory, expected):
    directory = Path(directory).resolve()
    reasons = []
    for name, digest in expected.items():
        path = (directory/name).resolve()
        if directory not in path.parents:
            reasons.append('unsafe_artifact_path:'+name)
        elif not path.is_file():
            reasons.append('missing_artifact:'+name)
        elif sha256_file(path) != digest:
            reasons.append('artifact_sha256_mismatch:'+name)
    return reasons


def _finite_tree(value, path='input'):
    if isinstance(value, (float,int)) and not isinstance(value,bool):
        if not math.isfinite(value): raise ValueError('nonfinite:'+path)
    elif isinstance(value, dict):
        for key, child in value.items(): _finite_tree(child, path+'.'+str(key))
    elif isinstance(value, (list,tuple)):
        for i, child in enumerate(value): _finite_tree(child, path+'['+str(i)+']')


def validate_trace(rows, expected_dt_s=1/240., *, check_all=True):
    if len(rows) < 8: raise ValueError('insufficient_trace')
    if check_all:
        for i,row in enumerate(rows): _finite_tree(row, 'row['+str(i)+']')
    try:
        arrays = {k:np.asarray([r[k] for r in rows],float) for k in SCALARS}
        for k,width in (('q',7),('q_command',7),('dq',7),('tcp_world',3),('axis_world',3)):
            arrays[k]=np.asarray([r[k] for r in rows],float)
            if arrays[k].shape != (len(rows),width): raise ValueError('shape:'+k)
    except (KeyError, TypeError) as error:
        raise ValueError('malformed_trace:'+str(error)) from error
    if any(not np.isfinite(x).all() for x in arrays.values()): raise ValueError('nonfinite_trace')
    if any(arrays[k].shape != (len(rows),) for k in SCALARS): raise ValueError('scalar_shape')
    times=arrays['time_s']; dt=np.diff(times)
    if (not np.isfinite(expected_dt_s) or expected_dt_s<=0 or
            not np.allclose(dt,expected_dt_s,atol=1e-7,rtol=0) or
            not np.isclose(times[0],expected_dt_s,atol=1e-7,rtol=0)):
        raise ValueError('timestamp_gap_or_origin')
    if any(r.get('phase') not in ('APPROACH','CONTACT','HOLD','RETRACT','DONE','ABORT')
           or not isinstance(r.get('target'),str) or not r['target']
           or not isinstance(r.get('subphase'),str) for r in rows):
        raise ValueError('invalid_phase_target_subphase')
    if any((arrays[k] < 0).any() for k in FRONT+FORBIDDEN+('hall_N','tcp_speed_m_s','tangent_m')):
        raise ValueError('negative_force_speed_or_distance')
    return arrays


def reversal_events(q, times, threshold_rad=np.deg2rad(1.)):
    """Confirm reversal after > threshold excursion from running extreme.

    Many small increments accumulate; jitter around the extreme does not count.
    Events retain both the turning point and the later confirmation timestamp.
    """
    q=np.asarray(q,float);times=np.asarray(times,float)
    if q.ndim != 2 or len(q)!=len(times) or not len(q) or not np.isfinite(q).all() or threshold_rad<=0:
        raise ValueError('invalid reversal input')
    direction=np.zeros(q.shape[1],int); extreme=q[0].copy(); extreme_row=np.zeros(q.shape[1],int)
    events=[]
    for i in range(1,len(q)):
        value=q[i]
        for j in range(q.shape[1]):
            if direction[j]==0:
                if abs(value[j]-extreme[j]) > threshold_rad:
                    direction[j]=1 if value[j]>extreme[j] else -1; extreme[j]=value[j];extreme_row[j]=i
            elif direction[j]*(value[j]-extreme[j]) > 0:
                extreme[j]=value[j];extreme_row[j]=i
            elif direction[j]*(extreme[j]-value[j]) > threshold_rad:
                events.append(dict(row=i,time_s=float(times[i]),joint=j+1,extreme_row=int(extreme_row[j]),
                    extreme_time_s=float(times[extreme_row[j]]),excursion_rad=float(abs(value[j]-extreme[j])),
                    from_direction=int(direction[j]),reason='confirmed_direction_reversal'))
                direction[j]*=-1; extreme[j]=value[j];extreme_row[j]=i
    return events


def _peak(values,times,indices=None):
    values=np.asarray(values,float)
    if values.ndim==1: values=values[:,None]
    local=np.abs(values).argmax(axis=0); joints=np.arange(values.shape[1])
    row=local if indices is None else np.asarray(indices)[local]
    return dict(max_abs_per_joint=np.abs(values)[local,joints].tolist(),
        signed_value_per_joint=values[local,joints].tolist(),row_per_joint=row.tolist(),
        time_s_per_joint=times[row].tolist(),rms_per_joint=np.sqrt(np.mean(values**2,axis=0)).tolist())


def _derivatives(values,dt,profile):
    raw_v=np.gradient(values,dt,axis=0,edge_order=2)
    raw_a=np.gradient(raw_v,dt,axis=0,edge_order=2)
    raw_j=np.gradient(raw_a,dt,axis=0,edge_order=2)
    requested=int(round(profile['filter_window_s']/dt)); requested += (requested%2==0)
    window=min(max(requested,7),len(values) if len(values)%2 else len(values)-1)
    filtered={name:savgol_filter(values,window,4,deriv=order,delta=dt,axis=0,mode='interp')
              for name,order in (('velocity',1),('acceleration',2),('jerk',3))}
    return {'raw':dict(velocity=raw_v,acceleration=raw_a,jerk=raw_j),'filtered':filtered},window


def _preference_status(exceeded,profile):
    if exceeded:return 'failed' if profile['user_selected'] else 'exceeded_pending_user_choice'
    return 'passed' if profile['user_selected'] else 'within_draft_pending_user_choice'


def analyze_motion(rows,outcome,profile=None,expected_dt_s=1/240., *, arrays=None):
    profile=load_profile() if profile is None else profile
    a=validate_trace(rows,expected_dt_s) if arrays is None else arrays
    times=a['time_s'];n=len(rows);labels={name:np.asarray([r.get(name,'') for r in rows]) for name in ('target','subphase','phase')}
    states={};windows={}
    for name,key in (('measured','q'),('commanded','q_command')):
        values=a[key];derivatives,window=_derivatives(values,expected_dt_s,profile)
        states[name]=(values,np.vstack((np.zeros((1,7)),np.abs(np.diff(values,axis=0)))),
                      reversal_events(values,times,np.deg2rad(profile['reversal_excursion_deg'])),derivatives)
        windows[name]=window
    def summarize(start,end):
        report=dict(start_row=start,end_row_exclusive=end,start_time_s=float(times[start]),end_time_s=float(times[end-1]))
        indices=np.arange(start,end)
        for label,(values,steps,events,derivatives) in states.items():
            travel=steps[start:end].sum(axis=0);selected=[e for e in events if start<=e['row']<end]
            metric=dict(travel_rad=travel.tolist(),total_travel_rad=float(travel.sum()),
                peak_to_peak_rad=np.ptp(values[start:end],axis=0).tolist(),
                min_rad=values[start:end].min(axis=0).tolist(),max_rad=values[start:end].max(axis=0).tolist(),
                max_step_rad=steps[start:end].max(axis=0).tolist(),
                step_peak=_peak(steps[start:end],times,indices),reversal_count_per_joint=[sum(e['joint']==j+1 for e in selected) for j in range(7)],
                reversal_count=len(selected),reversal_events=selected,
                derivatives={method:{kind:_peak(x[start:end],times,indices) for kind,x in ds.items()} for method,ds in derivatives.items()})
            report[label]=metric
        return report
    global_metrics=summarize(0,n)
    point_segments=[];subphase_segments=[]
    for keys,output in ((('target',),point_segments),(('target','phase','subphase'),subphase_segments)):
        cuts=[0]+[i for i in range(1,n) if any(labels[k][i]!=labels[k][i-1] for k in keys)]+[n]
        for start,end in zip(cuts[:-1],cuts[1:]):
            item=summarize(start,end);item.update({key:str(labels[key][start]) for key in keys});output.append(item)
    transitions=[]
    for previous,current in zip(point_segments[:-1],point_segments[1:]):
        boundary=current['start_row'];before=np.flatnonzero((labels['phase'][:boundary]=='HOLD') & (labels['target'][:boundary]==previous['target']))
        after=np.flatnonzero((labels['phase'][boundary:]=='HOLD') & (labels['target'][boundary:]==current['target']))
        start=int(before[-1])+1 if len(before) else previous['start_row']
        end=boundary+int(after[0])+1 if len(after) else current['end_row_exclusive']
        item=summarize(start,end);item.update(from_target=previous['target'],to_target=current['target'],boundary_row=boundary,
            definition='after last source HOLD through first destination HOLD; includes release and incoming boundary step')
        transitions.append(item)
    preferences=[]
    for label,metric in ((k,global_metrics[k]) for k in states):
        checks=[('joint_step_rad',metric['step_peak'])]+[(key,metric['derivatives']['raw'][kind]) for key,kind in
               (('joint_velocity_rad_s','velocity'),('joint_acceleration_rad_s2','acceleration'),('joint_jerk_rad_s3','jerk'))]
        for key,peak in checks:
            for j,value in enumerate(peak['max_abs_per_joint']):
                if value>profile[key]:preferences.append(dict(reason=key,label=label,joint=j+1,value=value,limit=profile[key],
                    row=peak['row_per_joint'][j],time_s=peak['time_s_per_joint'][j],status=_preference_status(True,profile)))
    return dict(**{'global':global_metrics},points=point_segments,subphases=subphase_segments,transitions=transitions,
        preference_events=preferences,preference_status=_preference_status(bool(preferences),profile),
        measured_recorded_velocity=_peak(a['dq'],times),sampling=dict(dt_s=expected_dt_s,hz=1/expected_dt_s,sample_count=n,
        duration_s=float(times[-1]),differentiation='central finite difference raw; edge_order=2; derivative chains amplify simulation noise',
        filtering='Savitzky-Golay polynomial derivative, order 4, centered window, interp edges; no unwrap; raw jumps retained',
        requested_filter_window_s=profile['filter_window_s'],actual_filter_samples=windows,
        actual_filter_span_s={k:(v-1)*expected_dt_s for k,v in windows.items()},
        boundary_accounting='each increment assigned to arriving row, so disjoint point/subphase travel sums equal global travel'))


def analyze_contact(rows,outcome,profile=None, *, arrays=None):
    profile=load_profile() if profile is None else profile
    a=validate_trace(rows) if arrays is None else arrays
    from sim.auscultation_expert_layout_v1.audit_episode import audit_sequence,audit_rows
    legacy=audit_sequence(rows,outcome) if 'points' in outcome else audit_rows(rows,outcome)
    reasons=[]
    # Legacy reason strings may combine per-point failures. Keep machine-readable individual reasons.
    for reason in legacy['reasons']:
        if ':' in reason:
            target,rest=reason.split(':',1);reasons.extend(target+':'+x for x in rest.split(','))
        else:reasons.append(reason)
    events=[];times=a['time_s'];front=sum(a[k] for k in FRONT);total=front+sum(a[k] for k in FORBIDDEN)
    from sim.auscultation_expert_layout_v1.phase_manager import PhaseConfig
    configured_points=outcome.get('points') or [dict(outcome,start_physics_row=0,contact_end_physics_row=len(rows))]
    for point in configured_points:
        config=asdict(PhaseConfig());config.update(point.get('phase_config',{}));PhaseConfig(**config)
        start=point.get('start_physics_row');end=point.get('contact_end_physics_row')
        if type(start)!=int or type(end)!=int or not 0<=start<end<=len(rows):continue
        for key,limit_key,reason in (('hall_N','hard_force_limit_N','hall_force_limit'),('compression_m','max_compression_m','stroke_limit')):
            v=a[key][start:end];i=start+int(v.argmax());limit=config[limit_key]
            if a[key][i]>=limit:
                events.append(dict(row=i,time_s=float(times[i]),target=point.get('target'),reason=reason,value=float(a[key][i]),limit=limit))
    for key,limit,reason in [(k,.5,k+'_contact') for k in FORBIDDEN]+[('total',6.,'physical_force_limit')]:
        v=total if key=='total' else a[key]
        if v.max()>limit:
            i=int(v.argmax());events.append(dict(row=i,time_s=float(times[i]),reason=reason,value=float(v[i]),limit=limit))
    normals={p['target']:np.asarray(p.get('command_normal_world',[]),float) for p in outcome.get('points',[])}
    normals_ok=all(r['target'] in normals and normals[r['target']].shape==(3,) and np.isfinite(normals[r['target']]).all()
                   and abs(np.linalg.norm(normals[r['target']])-1)<1e-5 for r in rows)
    if normals_ok:
        # Previous target's normal governs a cross-target increment; no target-frame re-labeling can erase drag.
        normal=np.asarray([normals[r['target']] for r in rows[:-1]])
        displacement=np.diff(a['tcp_world'],axis=0)
        tangent=displacement-np.sum(displacement*normal,axis=1)[:,None]*normal
        distance=np.linalg.norm(tangent,axis=1);speed=distance/np.diff(times)
        loaded=(np.maximum.reduce((front[:-1],front[1:],a['hall_N'][:-1],a['hall_N'][1:]))>=profile['loaded_force_N'])
        categories=[]
        for i in range(len(rows)-1):
            left,right=rows[i],rows[i+1]
            if left['target']!=right['target']:category='point_change'
            elif (left['phase']=='RETRACT' or right['phase']=='RETRACT' or
                    any(r['subphase'].startswith(('withdraw','retract')) for r in (left,right))):category='retraction'
            elif left['phase'] not in ('HOLD','CONTACT') or right['phase'] not in ('HOLD','CONTACT'):category='free_transfer'
            else:category=''
            categories.append(category)
        categories=np.asarray(categories);mask=loaded & (categories!='');intervals=[]
        normal_step=np.sum(displacement*normal,axis=1)
        # A category change splits an event even when force remains continuous.
        labels=np.where(mask,categories,'');cuts=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
        for start,end in zip(cuts[:-1],cuts[1:]):
            if not labels[start]:continue
            worst=start+int(speed[start:end].argmax());length=float(distance[start:end].sum());maximum_speed=float(speed[worst])
            exceeded=bool(length>profile['loaded_drag_max_event_m'] or maximum_speed>profile['loaded_drag_max_speed_m_s'])
            category=str(labels[start])
            intervals.append(dict(start_row=int(start),end_row=int(end),start_time_s=float(times[start]),end_time_s=float(times[end]),
                from_target=rows[start]['target'],to_target=rows[end]['target'],category=category,
                distance_m=length,tangential_distance_m=length,max_speed_m_s=maximum_speed,
                normal_signed_distance_m=float(normal_step[start:end].sum()),
                normal_absolute_distance_m=float(np.abs(normal_step[start:end]).sum()),
                tangential_net_displacement_world_m=tangent[start:end].sum(axis=0).tolist(),
                reference_normal_start_world=normal[start].tolist(),reference_normal_end_world=normal[end-1].tolist(),
                loaded_duration_s=float(times[end]-times[start]),
                max_front_force_N=float(front[start:end+1].max()),max_hall_force_N=float(a['hall_N'][start:end+1].max()),
                worst_row=int(worst+1),worst_time_s=float(times[worst+1]),
                reason='loaded_'+category+'_tangential_motion',exceeded=exceeded,
                distance_limit_m=profile['loaded_drag_max_event_m'],speed_limit_m_s=profile['loaded_drag_max_speed_m_s'],
                severity='preference_exceedance' if exceeded else 'diagnostic'))
        exceeded=any(x['exceeded'] for x in intervals)
        by_category={}
        for category in ('retraction','point_change','free_transfer'):
            group=[x for x in intervals if x['category']==category]
            by_category[category]=dict(event_count=len(group),exceedance_count=sum(x['exceeded'] for x in group),
                total_tangential_distance_m=sum(x['distance_m'] for x in group),
                total_normal_absolute_distance_m=sum(x['normal_absolute_distance_m'] for x in group),
                loaded_duration_s=sum(x['loaded_duration_s'] for x in group),
                max_event_distance_m=max((x['distance_m'] for x in group),default=0.))
        drag=dict(status=_preference_status(exceeded,profile),exceeded=exceeded,total_tangential_distance_m=float(distance[mask].sum()),
            max_event_distance_m=max((x['distance_m'] for x in intervals),default=0.),
            max_speed_m_s=float(speed[mask].max()) if mask.any() else 0.,events=intervals,by_category=by_category,
            loaded_force_N=profile['loaded_force_N'],scope='loaded-motion diagnostics separated into normal retraction, point change and free transfer; TCP projection is a slip proxy, not tissue shear force; within-threshold retraction is not an unsafe transfer')
    else:drag=dict(status='pending',reason='missing_valid_target_normals')
    points=[]
    for point in outcome.get('points',[]):
        start=point.get('start_physics_row');end=point.get('contact_end_physics_row')
        if type(start)!=int or type(end)!=int or not 0<=start<end<=len(rows):continue
        hold=np.asarray([r['phase']=='HOLD' for r in rows[start:end]])
        parts={key:dict(mean_N=float(a[key][start:end][hold].mean()),max_N=float(a[key][start:end][hold].max())) for key in FRONT} if hold.any() else {}
        points.append(dict(target=point['target'],hold_force_by_site=parts,
            anatomical_target_status='pending',acoustic_coupling_status='pending',
            reason='front ring/lip load and detector-proposed site do not establish diaphragm coupling or anatomical correctness'))
    return dict(legacy=legacy,hard_failures=reasons,events=events,loaded_drag=drag,points=points,
        clinical_contact_site=dict(status='pending',reason='independent anatomical/diaphragm coupling truth absent'))


def _read_trace(path):
    rows=[]
    with path.open('rb') as stream:
        for i,line in enumerate(stream):
            if not line.strip():raise ValueError('blank_trace_row:'+str(i))
            row=json.loads(line);_finite_tree(row,'row['+str(i)+']')
            # Raw contact points are validated before dropping to bound RAM use.
            rows.append({k:v for k,v in row.items() if k not in ('contact_points','contact_pairs_N')})
    return rows


CANDIDATE_CLASS='sim.expert_quality_20260924.candidate_controller.CandidateController'
CANDIDATE_SOURCE_NAMES=('candidate_controller.py','candidate_planner.py','run_candidate.py')


def verify_candidate_sources(source,outcome):
    """Require complete derived-source evidence whenever the outcome is a candidate."""
    source=Path(source).resolve()
    if not isinstance(outcome,dict):return dict(status='failed',artifact_sha256={},reasons=['candidate_sources:outcome must be an object'])
    name=outcome.get('candidate_sources_manifest')
    if not any(key in outcome for key in ('candidate_sources_manifest','candidate_sources_manifest_sha256','candidate_policy','candidate_method')):
        return dict(status='not_applicable',artifact_sha256={},reasons=[])
    reasons=[];bindings={}
    try:
        if not isinstance(name,str):raise ValueError('missing manifest path for candidate outcome')
        path=(source/name).resolve()
        if source not in path.parents:raise ValueError('unsafe manifest path')
        expected=outcome.get('candidate_sources_manifest_sha256')
        if not expected or sha256_file(path)!=expected:raise ValueError('manifest sha256 mismatch')
        bindings[name]=expected;manifest=json.loads(path.read_text())
        if (not isinstance(manifest,dict) or manifest.get('schema')!='wam_expert_candidate_sources_v1'
                or not isinstance(manifest.get('files'),dict) or set(manifest['files'])!=set(CANDIDATE_SOURCE_NAMES)):
            raise ValueError('candidate manifest must bind exactly the three implementation sources')
        policy=outcome.get('candidate_policy')
        if policy not in ('staged_continuous','checked_shortcut') or manifest.get('candidate_policy')!=policy:
            raise ValueError('candidate policy mismatch')
        if outcome.get('candidate_method')!=CANDIDATE_CLASS or manifest.get('derived_controller')!=CANDIDATE_CLASS:
            raise ValueError('candidate controller class mismatch')
        snapshot_name=outcome['input_snapshot'];snapshot_path=(source/snapshot_name).resolve()
        if source not in snapshot_path.parents:raise ValueError('unsafe snapshot path')
        if sha256_file(snapshot_path)!=outcome['input_snapshot_manifest_sha256']:raise ValueError('snapshot manifest sha256 mismatch')
        bindings[snapshot_name]=outcome['input_snapshot_manifest_sha256'];snapshot=json.loads(snapshot_path.read_text())
        if not isinstance(snapshot,dict) or not isinstance(snapshot.get('files'),dict):raise ValueError('snapshot files must be an object')
        for key,item in manifest['files'].items():
            if not isinstance(item,dict):raise ValueError('candidate source record must be an object')
            relative=item['path'];digest=item['sha256'];captured=snapshot['files']['source/candidate/'+key]['sha256']
            if digest!=captured:raise ValueError('candidate source differs from original snapshot:'+key)
            expected_path=snapshot_path.parent/'objects'/captured
            if (source/relative).resolve()!=expected_path.resolve():raise ValueError('candidate source path differs from snapshot object:'+key)
            if relative in bindings and bindings[relative]!=digest:raise ValueError('conflicting source hash')
            bindings[relative]=digest
        reasons.extend(verify_hashes(source,bindings))
    except (OSError,ValueError,KeyError,TypeError) as error:reasons.append('candidate_sources:'+str(error))
    return dict(status='failed' if reasons else 'passed',artifact_sha256=bindings,reasons=reasons)


def validate_episode_provenance(episode_path):
    """Read-only render preflight: validate old audit anchors without numerical audit.

    Status describes content binding only, never mechanical or clinical validity.
    The caller can render failed runs for diagnosis using recorded_acceptance.
    Recheck returned artifact_sha256 with verify_hashes after loading/rendering
    to reject evidence changed since this validation. No trace parsing, NumPy
    replay loading, FK, simulation, or filesystem writes occur here.
    """
    source=Path(episode_path).resolve();reasons=[];hashes={};acceptance={};candidate={}
    urdf=Path(__file__).resolve().parents[1]/'auscultation_expert_layout_v1'/'openarm_v10_follower_stetho.urdf'
    try:
        missing=[n for n in REQUIRED_ARTIFACTS if not (source/n).is_file()]
        if missing:raise ValueError('missing_artifacts:'+','.join(missing))
        hashes.update({n:sha256_file(source/n) for n in REQUIRED_ARTIFACTS})
        outcome=json.loads((source/'outcome.json').read_text());_finite_tree(outcome,'outcome')
        if not isinstance(outcome,dict):raise ValueError('outcome must be an object')
        anchors={n:json.loads((source/n).read_text()) for n in ('independent_audit.json','posture_audit.json')}
        for name,anchor in anchors.items():
            if not isinstance(anchor,dict):raise ValueError('audit anchor must be an object:'+name)
            required=('expert_trace.jsonl','outcome.json','control_commands.json') if name=='independent_audit.json' else ('expert_trace.jsonl','outcome.json','expert_replay.npz')
            binding=anchor.get('artifact_sha256',{});acceptance[name]=anchor.get('accepted')
            for key in required:
                if binding.get(key)!=hashes[key]:reasons.append(name+':artifact_sha256_mismatch:'+key)
        sources=anchors['posture_audit.json'].get('source_sha256',{})
        if sources.get('world_from_left_base',{}).get('sha256')!=hashes['world_from_left_base.json']:
            reasons.append('posture_audit.json:transform_sha256_mismatch')
        urdf_hash=sha256_file(urdf)
        if sources.get('kinematics_urdf',{}).get('sha256')!=urdf_hash:reasons.append('kinematics_urdf_anchor_mismatch')
        manifest_path=(source/outcome.get('input_snapshot','missing')).resolve()
        if source not in manifest_path.parents or not manifest_path.is_file():raise ValueError('missing_or_unsafe_input_snapshot')
        relative=str(manifest_path.relative_to(source));hashes[relative]=sha256_file(manifest_path)
        if hashes[relative]!=outcome.get('input_snapshot_manifest_sha256'):reasons.append('snapshot_manifest_mismatch')
        manifest=json.loads(manifest_path.read_text())
        if not isinstance(manifest,dict) or not isinstance(manifest.get('files'),dict):raise ValueError('snapshot files must be an object')
        for name,item in manifest['files'].items():
            digest=item['sha256']
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('invalid_snapshot_digest')
            relative=str((manifest_path.parent/'objects'/digest).relative_to(source));hashes[relative]=digest
        if outcome.get('collision_regeneration_sha256')!=manifest['files']['source/collision_regeneration']['sha256']:
            reasons.append('geometry_source_binding_mismatch')
        candidate=verify_candidate_sources(source,outcome);reasons.extend(candidate['reasons']);hashes.update(candidate['artifact_sha256'])
        if sha256_file(urdf)!=urdf_hash:reasons.append('kinematics_changed_during_validation')
    except (OSError,ValueError,KeyError,TypeError) as error:reasons.append('invalid_provenance:'+str(error))
    reasons.extend(verify_hashes(source,hashes));reasons=list(dict.fromkeys(reasons))
    return dict(status='failed' if reasons else 'passed',reasons=reasons,artifact_sha256=hashes,
        candidate_source_binding=candidate,recorded_acceptance=acceptance,
        scope='content binding against saved audit anchors and snapshot only; numerical acceptance is not recomputed')


def select_worst_events(motion,contact):
    """Within-profile loaded release observations are diagnostics, not hazards."""
    return (motion.get('preference_events',[])+contact.get('events',[])+
            [event for event in contact.get('loaded_drag',{}).get('events',[]) if event.get('exceeded') is True])


def summarize_landmarks(elbow_z,wrist_z,shoulder_z,times,start,end,contact_mask):
    """Raw FK heights per segment; integration uses right-sample rectangle dt."""
    elbow_z=np.asarray(elbow_z);wrist_z=np.asarray(wrist_z);times=np.asarray(times)
    heights=elbow_z[start:end];gap=wrist_z[start:end]-heights
    above=heights-shoulder_z;row=start+int(heights.argmax());dt=float(np.median(np.diff(times)))
    contact=np.asarray(contact_mask)[start:end];minimum=int(gap.argmin())
    return dict(max_elbow_z_m=float(heights.max()),max_elbow_row=row,max_elbow_time_s=float(times[row]),
        shoulder_z_m=float(shoulder_z),max_elbow_above_shoulder_m=float(above.max()),
        time_above_shoulder_s=float(np.count_nonzero(above>0)*dt),
        above_shoulder_integral_m_s=float(np.maximum(above,0).sum()*dt),
        minimum_elbow_below_wrist_m=float(gap.min()),minimum_wrist_gap_row=start+minimum,
        minimum_wrist_gap_time_s=float(times[start+minimum]),
        minimum_contact_elbow_below_wrist_m=float(gap[contact].min()) if contact.any() else None)


def _legacy_modules():
    # Never create __pycache__ inside frozen source packages.
    prior=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:
        from sim.auscultation_expert_layout_v1 import audit_episode
        from sim.auscultation_expert_layout_v1 import audit_low_elbow,dataset,controller,kinematics
        return audit_low_elbow,dataset,controller,kinematics
    finally:sys.dont_write_bytecode=prior


def simulation_quality_decision(hard_failures,motion_status,drag_status,profile):
    """Required pending evaluations cannot silently become a selected-profile pass."""
    if hard_failures:return 'reject_hard_failure'
    statuses=(motion_status,drag_status)
    if 'failed' in statuses:return 'reject_selected_preference'
    known=('passed','failed','within_draft_pending_user_choice','exceeded_pending_user_choice')
    if any(status not in known for status in statuses):return 'pending_evaluation'
    if 'exceeded_pending_user_choice' in statuses:return 'review_draft_preference_exceedances'
    return 'passes_selected_engineering_profile' if profile['user_selected'] else 'passes_legacy_gates_draft_preferences_within_bounds'


def audit_episode(episode_path,output_path,profile_path=None):
    """Audit exact immutable files and write one JSON outside the input tree.

    Returns a report even for missing/invalid evidence. Output is never an input
    artifact. Overall accepted is deliberately false without external clinical
    and assembly truth; engineering gates and preferences have separate states.
    """
    source=Path(episode_path).resolve();destination=Path(output_path).resolve()
    if destination==source or source in destination.parents:
        raise ValueError('output must be outside input directory')
    profile=load_profile(profile_path);profile_file=Path(profile_path or DEFAULT_PROFILE).resolve()
    report=dict(schema='expert_quality_audit_v1',accepted=False,overall_status='pending',episode=str(source),
        scope='offline simulation quality; no acoustic, clinical, assembly or real-robot acceptance',
        profile=profile,profile_sha256=sha256_file(profile_file),hard_failures=[],pending=[
            'clinical_anatomical_targets_and_acoustic_coupling','physical_assembly_motor_and_cable_limits'],artifact_sha256={})
    reasons=report['hard_failures'];hashes=report['artifact_sha256']
    try:
        provenance=validate_episode_provenance(source)
        reasons.extend(provenance['reasons']);hashes.update(provenance['artifact_sha256'])
        report['candidate_source_binding']=provenance['candidate_source_binding']
        report['provenance_status']=provenance['status']
        outcome=json.loads((source/'outcome.json').read_text());_finite_tree(outcome,'outcome')
        base=json.loads((source/'world_from_left_base.json').read_text());_finite_tree(base,'base')
        anchors={n:json.loads((source/n).read_text()) for n in ('independent_audit.json','posture_audit.json')}
        posture,dataset,controller,kinematics=_legacy_modules()
        source_paths={name:Path(module.__file__) for name,module in (('audit_low_elbow',posture),('dataset',dataset),('controller',controller),('kinematics',kinematics))}
        from sim.auscultation_expert_layout_v1 import audit_episode as contact_module
        source_paths['audit_episode']=Path(contact_module.__file__);source_paths['kinematics_urdf']=kinematics.DEFAULT_URDF_PATH
        for module_name in ('phase_manager','posture','tool_frames','targets'):
            source_paths[module_name]=Path(kinematics.__file__).with_name(module_name+'.py')
        source_paths['quality_evaluator']=Path(__file__).resolve()
        source_hashes={name:sha256_file(path) for name,path in source_paths.items()}
        report['audit_sources']={name:dict(path=str(source_paths[name]),sha256=digest) for name,digest in source_hashes.items()}
        expected_urdf=anchors['posture_audit.json'].get('source_sha256',{}).get('kinematics_urdf',{}).get('sha256')
        if expected_urdf!=source_hashes['kinematics_urdf']:reasons.append('kinematics_urdf_anchor_mismatch')
        rows=_read_trace(source/'expert_trace.jsonl');a=validate_trace(rows,check_all=False)
        report['motion']=analyze_motion(rows,outcome,profile,arrays=a)
        report['contact']=analyze_contact(rows,outcome,profile,arrays=a);reasons.extend(report['contact']['hard_failures'])
        controls=json.loads((source/'control_commands.json').read_text());_finite_tree(controls,'controls')
        try:report['control_trace_binding']=dataset.validate_control_trace(controls,rows)
        except (KeyError,TypeError,ValueError) as error:reasons.append('control_trace_binding:'+str(error))
        del controls
        with np.load(source/'expert_replay.npz',allow_pickle=False) as loaded:
            replay={key:loaded[key] for key in loaded.files}
        report['posture']=posture.audit_rows(rows,outcome,base,replay)
        reasons.extend('posture:'+r for r in report['posture']['reasons'])
        shoulder=report['posture'].get('shoulder_z_m')
        for label in ('measured','commanded'):
            metric=report['posture'].get(label,{})
            if 'max_elbow_row' in metric:
                row=metric['max_elbow_row'];metric['max_elbow_time_s']=rows[row]['time_s']
                metric['max_elbow_above_shoulder_m']=metric['max_elbow_z_m']-shoulder
        if shoulder is not None:
            kin=kinematics.OpenArmKinematics();contact_mask=np.asarray([r['phase'] in posture.CONTACT_PHASES for r in rows])
            for label,key in (('measured','q'),('commanded','q_command')):
                elbows=posture.elbow_positions_world(kin,base,a[key])[:,2]
                wrists=posture.landmark_positions_world(kin,base,a[key],'openarm_left_link7')[:,2]
                for segment in [report['motion']['global']]+report['motion']['points']+report['motion']['subphases']+report['motion']['transitions']:
                    segment[label]['posture']=summarize_landmarks(elbows,wrists,shoulder,a['time_s'],segment['start_row'],segment['end_row_exclusive'],contact_mask)
        report['worst_events']=select_worst_events(report['motion'],report['contact'])
        for name,path in source_paths.items():
            if sha256_file(path)!=source_hashes[name]:reasons.append('audit_source_changed:'+name)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
        reasons.append('invalid_evidence:'+str(error))
    reasons.extend(verify_hashes(source,hashes))
    if sha256_file(profile_file)!=report['profile_sha256']:reasons.append('profile_changed_during_audit')
    reasons[:]=list(dict.fromkeys(reasons))
    report['legacy_engineering_status']='failed' if reasons else 'passed'
    preference_failed=(report.get('motion',{}).get('preference_status')=='failed' or report.get('contact',{}).get('loaded_drag',{}).get('status')=='failed')
    report['overall_status']='rejected' if reasons or preference_failed else 'pending'
    report['simulation_quality_decision']=simulation_quality_decision(reasons,
        report.get('motion',{}).get('preference_status'),report.get('contact',{}).get('loaded_drag',{}).get('status'),profile)
    if report['simulation_quality_decision']=='pending_evaluation':report['pending'].append('required_simulation_quality_evaluation_incomplete')
    report['contact_site_calibration_status']='pending'
    report['training_demonstration_readiness']='rejected' if reasons or preference_failed else 'pending_external_protocol_and_assembly_confirmation'
    if not profile['user_selected']:report['pending'].append('engineering_preference_profile_user_choice')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
