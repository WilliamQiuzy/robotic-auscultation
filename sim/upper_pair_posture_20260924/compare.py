"""Hash-bound actual-trace metrics for the upper-pair posture experiment.

No simulation, collision checking, acceptance decision or source writes. Raw and
filtered derivatives are computed on each FULL trace before any comparison slice.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sim.curobo_expert_v1.quality import validate_episode_provenance, verify_hashes
from sim.expert_quality_20260924 import quality as equations

PROJECT=Path(__file__).resolve().parents[2]
BASELINE=PROJECT/'sim/curobo_expert_v1/results/physics_revised_all10_v1'
TARGETS=tuple(side+str(level) for level in range(1,6) for side in ('R','L'))
JOINT_CONVENTION={
    'order':'left arm URDF chain joint 1 through joint 7',
    'one_based_joint_numbers':list(range(1,8)),
    'wrist_zero_based_indices':[4,5,6], 'wrist_one_based_joints':[5,6,7],
    'elbow_flexion_zero_based_index':3,'elbow_flexion_one_based_joint':4,
    'elbow_flexion_scope':'Actuator q4 excursion/travel, not a spatial elbow flip.',
    'elbow_landmark':'openarm_left_link4','wrist_landmark':'openarm_left_link7'}
CONTACT_PHASES=('CONTACT','HOLD','RETRACT')
FILTER_WINDOW_S=.1
TRANSITION_HALF_WINDOW_S=.025


def require(condition,message):
    if not condition:raise ValueError(message)


def sha256_bytes(data):return hashlib.sha256(data).hexdigest()


def sha256(path):
    return equations.sha256_file(path)


def read(path):
    def reject(value):raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def _point(outcome,target):
    return next((p for p in outcome['points'] if p['target']==target),None)


def _peak(values,times,indices):
    values=np.asarray(values,float)
    if values.ndim==1:values=values[:,None]
    local=np.abs(values).argmax(axis=0);columns=np.arange(values.shape[1])
    absolute_rows=np.asarray(indices)[local]
    return dict(max_abs_per_joint=np.abs(values)[local,columns].tolist(),
        signed_value_per_joint=values[local,columns].tolist(),row_per_joint=absolute_rows.tolist(),
        time_s_per_joint=times[absolute_rows].tolist(),rms_per_joint=np.sqrt(np.mean(values**2,axis=0)).tolist())


def _scalar_peak(values,times,indices,maximum=True):
    local=int(np.argmax(values) if maximum else np.argmin(values));row=int(indices[local])
    return dict(value=float(values[local]),row=row,time_s=float(times[row]))


def _window(ep,start,end):
    return dict(start_row=start,end_row_exclusive=end,sample_count=end-start,
        interval_start_s=start*ep['dt'],interval_end_s=end*ep['dt'],duration_s=(end-start)*ep['dt'],
        first_sample_time_s=float(ep['a']['time_s'][start]),last_sample_time_s=float(ep['a']['time_s'][end-1]))


def prepare_episode(rows,outcome,base,replay=None):
    """Pure in-memory preparation. IO callers must first validate provenance."""
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.audit_low_elbow import landmark_positions_world
    equations._finite_tree(outcome,'outcome')
    dt=outcome.get('physics_dt_s')
    require(type(dt) in (int,float) and dt>0,'invalid physics dt')
    a=equations.validate_trace(rows,dt,check_all=True)
    n=len(rows);require(outcome.get('physics_steps')==n,'outcome physics row count mismatch')
    requested=outcome.get('requested_points')
    require(isinstance(requested,list) and requested and requested==list(TARGETS[:len(requested)]),
            'requested points must be the canonical prefix')
    points=outcome.get('points')
    require(isinstance(points,list) and points and [p.get('target') for p in points]==requested[:len(points)],
            'outcome point inventory differs from requested prefix')
    previous=0
    for p in points:
        start,end=p.get('start_physics_row'),p.get('end_physics_row')
        require(type(start) is int and type(end) is int and start==previous and start<=end<=n,
                'invalid point intervals')
        require(all(r['target']==p['target'] for r in rows[start:end]),'point/trace target mismatch')
        previous=end
    require(previous==n,'point intervals do not cover trace')
    kin=OpenArmKinematics()
    elbow=landmark_positions_world(kin,base,a['q'],'openarm_left_link4')[:,2]
    wrist=landmark_positions_world(kin,base,a['q'],'openarm_left_link7')[:,2]
    shoulder=float(landmark_positions_world(kin,base,np.zeros((1,7)),'openarm_left_link1')[0,2])
    limits=outcome.get('posture_limits',{})
    lower=np.asarray(limits.get('joint_lower_rad',kin.lower_limits),float)
    upper=np.asarray(limits.get('joint_upper_rad',kin.upper_limits),float)
    require(lower.shape==upper.shape==(7,) and np.isfinite([lower,upper]).all() and np.all(lower<upper),
            'invalid posture joint limits')
    states={};filter_samples={}
    for label,key in (('actual','q'),('commanded','q_command')):
        derivatives,window=equations._derivatives(a[key],dt,{'filter_window_s':FILTER_WINDOW_S})
        states[label]=dict(q=a[key],steps=np.vstack((np.zeros((1,7)),np.abs(np.diff(a[key],axis=0)))),
                           derivatives=derivatives,reversals=equations.reversal_events(a[key],a['time_s']))
        filter_samples[label]=window
    result=dict(rows=rows,outcome=outcome,a=a,n=n,dt=dt,requested=requested,states=states,
                lower=lower,upper=upper,urdf_lower=kin.lower_limits,urdf_upper=kin.upper_limits,
                elbow=elbow,wrist=wrist,shoulder=shoulder,joint_names=list(kin.joint_names),
                filter_samples=filter_samples,tracking=a['q']-a['q_command'],replay=None,sources={})
    if replay is not None:
        indices=np.asarray(replay['row_indices']);poses=np.asarray(replay['poses']);paths=np.asarray(replay['paths'])
        require(indices.ndim==1 and np.issubdtype(indices.dtype,np.integer) and len(indices)>0
                and np.array_equal(indices,np.arange(7,n,8)),'invalid recorded pose indices')
        require(paths.ndim==1 and poses.shape==(len(indices),len(paths),7)
                and np.isfinite(poses).all(),'invalid recorded body poses')
        require(np.all(np.abs(np.linalg.norm(poses[:,:,3:],axis=2)-1)<1e-4),'invalid replay rotations')
        landmarks=[]
        for name in ('openarm_left_link4','openarm_left_link7'):
            matches=[i for i,path in enumerate(paths) if str(path).endswith('/'+name)]
            require(len(matches)==1,'ambiguous recorded landmark '+name);landmarks.append(matches[0])
        result['replay']=dict(indices=indices,elbow=poses[:,landmarks[0],2],wrist=poses[:,landmarks[1],2])
    return result


def _posture(ep,indices,elbow,wrist):
    times=ep['a']['time_s'];gap=wrist-elbow
    contact=np.asarray([ep['rows'][int(i)]['phase'] in CONTACT_PHASES for i in indices])
    high=_scalar_peak(elbow,times,indices);low_gap=_scalar_peak(gap,times,indices,False)
    return dict(elbow_max_z_m=high['value'],elbow_max_event=high,shoulder_z_m=ep['shoulder'],
        elbow_max_above_shoulder_m=high['value']-ep['shoulder'],
        minimum_wrist_minus_elbow_z_m=low_gap['value'],minimum_wrist_gap_event=low_gap,
        minimum_contact_wrist_minus_elbow_z_m=float(gap[contact].min()) if contact.any() else None,
        contact_samples=int(contact.sum()),contact_phase_definition=list(CONTACT_PHASES),
        elbow_peak_to_peak_z_m=float(np.ptp(elbow)))


def summarize(ep,start,end):
    require(type(start) is int and type(end) is int and 0<=start<end<=ep['n'],'invalid metric window')
    indices=np.arange(start,end);times=ep['a']['time_s'];result={'window':_window(ep,start,end)}
    for label,state in ep['states'].items():
        values=state['q'][start:end];travel=state['steps'][start:end].sum(axis=0)
        events=[e for e in state['reversals'] if start<=e['row']<end]
        result[label]=dict(travel_rad=travel.tolist(),total_travel_rad=float(travel.sum()),
            wrist_joints_5_7_travel_rad=float(travel[4:7].sum()),elbow_flexion_joint_4_travel_rad=float(travel[3]),
            excursion_rad=np.ptp(values,axis=0).tolist(),minimum_q_rad=values.min(axis=0).tolist(),
            maximum_q_rad=values.max(axis=0).tolist(),max_step_rad=state['steps'][start:end].max(axis=0).tolist(),
            reversal_count=len(events),reversal_count_per_joint=[sum(e['joint']==j+1 for e in events) for j in range(7)],
            raw={key:_peak(array[start:end],times,indices) for key,array in state['derivatives']['raw'].items()},
            filtered={key:_peak(array[start:end],times,indices) for key,array in state['derivatives']['filtered'].items()})
    q=ep['a']['q'][start:end];margin=np.minimum(q-ep['lower'],ep['upper']-q)
    urdf_margin=np.minimum(q-ep['urdf_lower'],ep['urdf_upper']-q)
    tolerance=ep['outcome'].get('posture_limits',{}).get('observed_joint_tolerance_rad',0.)
    result['joint_limits']=dict(minimum_declared_envelope_margin_rad=float(margin.min()),
        minimum_envelope_margin_per_joint_rad=margin.min(axis=0).tolist(),
        minimum_urdf_margin_rad=float(urdf_margin.min()),declared_observed_tolerance_rad=tolerance,
        samples_below_declared_envelope_minus_tolerance=int(np.any(margin < -tolerance-1e-9,axis=1).sum()))
    result['recorded_actual_dq']=_peak(ep['a']['dq'][start:end],times,indices)
    result['tracking']=_peak(ep['tracking'][start:end],times,indices)
    result['tracking']['overall_rms_rad']=float(np.sqrt(np.mean(ep['tracking'][start:end]**2)))
    result['posture_measured_q_fk']=_posture(ep,indices,ep['elbow'][start:end],ep['wrist'][start:end])
    result['posture_recorded_bodies']=None
    if ep['replay'] is not None:
        r=ep['replay'];mask=(r['indices']>=start)&(r['indices']<end)
        if mask.any():result['posture_recorded_bodies']=_posture(ep,r['indices'][mask],r['elbow'][mask],r['wrist'][mask])
    force=ep['a']['hall_N'][start:end]
    front=sum(ep['a'][name][start:end] for name in ('diaphragm_N','lip_N','contact_rim_N'))
    result['force']=dict(hall_peak_N=float(force.max()),front_sum_peak_N=float(front.max()),
        hall_peak_event=_scalar_peak(force,times,indices),front_peak_event=_scalar_peak(front,times,indices),
        component_peak_N={key:float(ep['a'][key][start:end].max()) for key in
                         ('diaphragm_N','lip_N','contact_rim_N','carriage_other_N','frame_N','forbidden_N')})
    result['phase_sample_counts']=dict(Counter(ep['rows'][i]['phase'] for i in indices))
    result['raw_derivative_global_endpoint_samples']=int(((indices<3)|(indices>=ep['n']-3)).sum())
    return result


def _hold_end(ep,target):
    p=_point(ep['outcome'],target)
    if p is None:return None
    hold=[i for i in range(p['start_physics_row'],p['end_physics_row']) if ep['rows'][i]['phase']=='HOLD']
    if not hold:return None
    # A truncated HOLD cannot define a completed prefix even if it has samples.
    if hold[-1]+1>=ep['n'] or ep['rows'][hold[-1]+1]['phase']=='HOLD':return None
    if hold[-1]-hold[0]+1!=len(hold):return None
    config=p.get('phase_config',ep['outcome'].get('phase_config',{}))
    required=config.get('hold_duration_s')
    if type(required) not in (int,float) or required<=0 or len(hold)*ep['dt']+1e-7<required:return None
    return hold[-1]+1


def _max_value(metric,kind,method='raw'):
    return max(metric['actual'][method][kind]['max_abs_per_joint'])


def _delta(baseline,alternative):
    if baseline is None or alternative is None:return None
    def values(x):return dict(actual_total_travel_rad=x['actual']['total_travel_rad'],
        actual_wrist_joints_5_7_travel_rad=x['actual']['wrist_joints_5_7_travel_rad'],
        actual_joint_4_travel_rad=x['actual']['elbow_flexion_joint_4_travel_rad'],
        actual_max_raw_velocity_rad_s=_max_value(x,'velocity'),actual_max_raw_acceleration_rad_s2=_max_value(x,'acceleration'),
        actual_max_raw_jerk_rad_s3=_max_value(x,'jerk'),actual_max_filtered_jerk_rad_s3=_max_value(x,'jerk','filtered'),
        actual_max_elbow_z_m=x['posture_measured_q_fk']['elbow_max_z_m'],
        actual_minimum_wrist_minus_elbow_z_m=x['posture_measured_q_fk']['minimum_wrist_minus_elbow_z_m'],
        tracking_rms_rad=x['tracking']['overall_rms_rad'],hall_peak_N=x['force']['hall_peak_N'],
        front_peak_N=x['force']['front_sum_peak_N'],duration_s=x['window']['duration_s'])
    left,right=values(baseline),values(alternative)
    result={key:right[key]-value for key,value in left.items()}
    result['actual_travel_per_joint_rad']=(np.asarray(alternative['actual']['travel_rad'])-baseline['actual']['travel_rad']).tolist()
    result['convention']='alternative minus baseline; no combined score or automatic winner'
    return result


def _pair(b,a,left,right,**extra):
    bm=summarize(b,*left) if left and left[0]<left[1] else None
    am=summarize(a,*right) if right and right[0]<right[1] else None
    return dict(status='matched' if bm is not None and am is not None else 'unavailable_or_one_sided',
                baseline=bm,alternative=am,delta=_delta(bm,am),**extra)


def _stages(ep,end):
    end=ep['n'] if end is None else end;segments={};counts=Counter();start=0
    for i in range(1,end+1):
        previous=tuple(ep['rows'][start][key] for key in ('target','phase','subphase'))
        if i==end or tuple(ep['rows'][i][key] for key in ('target','phase','subphase'))!=previous:
            key=previous+(counts[previous],);segments[key]=(start,i);counts[previous]+=1;start=i
    return segments


def _event(ep,row):
    r=ep['rows'][row];indices=np.array([row])
    return dict(row=row,time_s=float(ep['a']['time_s'][row]),target=r['target'],phase=r['phase'],subphase=r['subphase'],
        hall_N=float(ep['a']['hall_N'][row]),front_sum_N=float(sum(ep['a'][k][row] for k in ('lip_N','diaphragm_N','contact_rim_N'))),
        compression_m=float(ep['a']['compression_m'][row]))


def _event_dynamics(ep,row):
    half=max(1,int(round(TRANSITION_HALF_WINDOW_S/ep['dt'])))
    start,end=max(0,row-half),min(ep['n'],row+half+1);indices=np.arange(start,end);times=ep['a']['time_s']
    return dict(window=_window(ep,start,end),actual_raw_jerk=_peak(ep['states']['actual']['derivatives']['raw']['jerk'][start:end],times,indices),
                commanded_raw_jerk=_peak(ep['states']['commanded']['derivatives']['raw']['jerk'][start:end],times,indices),
                scope='Full-trace raw derivatives within +/-25 ms; descriptive timing, not causal attribution.')


def contact_point(ep,target):
    point=_point(ep['outcome'],target)
    if point is None or point['start_physics_row']==point['end_physics_row']:return None
    start=point['start_physics_row'];end=point.get('contact_end_physics_row',point['end_physics_row'])
    require(type(end) is int and start<end<=point['end_physics_row'],'invalid contact_end interval')
    indices=np.arange(start,end);hold=np.array([i for i in indices if ep['rows'][i]['phase']=='HOLD'],dtype=int)
    config=point.get('phase_config',ep['outcome'].get('phase_config',{}))
    target_force=config.get('target_force_N');tolerance=config.get('force_tolerance_N')
    front=sum(ep['a'][key] for key in ('lip_N','diaphragm_N','contact_rim_N'))
    def force_stats(values):
        selected=values[hold]
        if not len(selected):return None
        return dict(mean_N=float(selected.mean()),std_N=float(selected.std()),peak_N=float(selected.max()),
                    minimum_N=float(selected.min()),peak_to_peak_N=float(np.ptp(selected)),
                    rms_error_to_declared_target_N=float(np.sqrt(np.mean((selected-target_force)**2))) if target_force is not None else None,
                    outside_declared_band_samples=int((np.abs(selected-target_force)>tolerance).sum()) if target_force is not None and tolerance is not None else None)
    runs=[]
    for row in hold:
        if not runs or row!=runs[-1][1]:runs.append([int(row),int(row)+1])
        else:runs[-1][1]+=1
    hold_metric=dict(duration_s=len(hold)*ep['dt'],sample_count=len(hold),runs=[{'start_row':s,'end_row_exclusive':e} for s,e in runs],
        longest_continuous_s=max((e-s for s,e in runs),default=0)*ep['dt'],declared_hold_duration_s=config.get('hold_duration_s'),
        controller_recorded_hold_s=point.get('observed_hold_s'),declared_force_target_N=target_force,
        declared_force_tolerance_N=tolerance,hall=force_stats(ep['a']['hall_N']),front_sum=force_stats(front))
    if len(hold):
        hold_metric.update(tangent_peak_m=float(ep['a']['tangent_m'][hold].max()),
                           angle_peak_deg=float(ep['a']['angle_deg'][hold].max()),
                           compression_mean_m=float(ep['a']['compression_m'][hold].mean()),
                           compression_std_m=float(ep['a']['compression_m'][hold].std()),
                           recorded_tcp_speed_peak_m_s=float(ep['a']['tcp_speed_m_s'][hold].max()))
    after=indices[indices>hold[-1]] if len(hold) else np.array([],int)
    threshold=config.get('release_force_limit_N',.2)
    below=[int(i) for i in after if front[i]<=threshold and ep['a']['hall_N'][i]<=threshold]
    zero=[int(i) for i in after if front[i]==0 and ep['a']['hall_N'][i]==0]
    release=dict(force_threshold_N=threshold,first_below_both_force_thresholds=_event(ep,below[0]) if below else None,
                 first_exact_zero_both_forces=_event(ep,zero[0]) if zero else None,
                 raw_dynamics_near_first_exact_zero=_event_dynamics(ep,zero[0]) if zero else None)
    loaded=[]
    for i in after:
        if i>0 and (ep['rows'][i]['phase']=='RETRACT' or ep['rows'][i]['subphase'].startswith('normal_')) and max(front[i-1],front[i],ep['a']['hall_N'][i-1],ep['a']['hall_N'][i])>.2:
            loaded.append(int(i))
    normal=point.get('command_normal_world')
    drag=dict(status='unavailable_normal',loaded_threshold_N=.2,sample_count=len(loaded),
              load_definition='max of raw Hall and front sum at either displacement endpoint',
              scope='Raw TCP displacement while withdrawing under front load; no safety/clinical threshold decision.')
    if normal is not None:
        normal=np.asarray(normal,float)
        require(normal.shape==(3,) and np.isfinite(normal).all() and np.linalg.norm(normal)>0,'invalid contact normal')
        normal=normal/np.linalg.norm(normal)
        displacement=ep['a']['tcp_world'][loaded]-ep['a']['tcp_world'][np.asarray(loaded,dtype=int)-1]
        signed=displacement@normal; tangent=np.linalg.norm(displacement-signed[:,None]*normal,axis=1)
        drag.update(status='measured',tangent_travel_m=float(tangent.sum()),normal_absolute_travel_m=float(np.abs(signed).sum()),
                    signed_normal_travel_m=float(signed.sum()),tangent_peak_speed_m_s=float(tangent.max()/ep['dt']) if len(tangent) else 0.)
    return dict(target=target,window=_window(ep,start,end),window_scope='through contact_end_physics_row when recorded; excludes terminal home path',
                contact_end_bound_recorded='contact_end_physics_row' in point,hold=hold_metric,release=release,loaded_withdrawal=drag,
                force_component_peaks_N={key:float(ep['a'][key][indices].max()) for key in
                    ('hall_N','lip_N','diaphragm_N','contact_rim_N','carriage_other_N','frame_N','forbidden_N')})


def geometry_diagnostics(ep):
    records=[]
    def visit(value,path,gate):
        if isinstance(value,dict):
            if 'minimum_distance_m' in value:
                records.append(dict(source_field=path,reported_check_value_m=value['minimum_distance_m'],
                    clear=value.get('clear'),gate_m=gate,clear_values_are_capped_at_gate=value.get('clear') is True,
                    sample_count=value.get('samples')))
            for key,child in value.items():visit(child,path+'.'+key,gate)
        elif isinstance(value,list):
            for i,child in enumerate(value):visit(child,path+'['+str(i)+']',gate)
    for i,route in enumerate(ep['outcome'].get('inter_point_routes',[])):
        visit(route,'inter_point_routes['+str(i)+']',route.get('free_clearance_m'))
    return dict(exact_clearance_margin_m=None,records=records,
        scope='Recorded discrete surface-check evidence only. Passing legacy minimum_distance_m is capped at the gate, '
              'so it is not an exact clearance or continuous/solid collision certificate; no check rerun here.')


def _diagnostics(ep):
    failures=[];previous=None
    for i,row in enumerate(ep['rows']):
        reasons={key:row.get(key) for key in ('physical_violations','posture_violations','recovery') if row.get(key)}
        if row['phase']=='ABORT':reasons['phase']='ABORT'
        if reasons:
            identity=(row['target'],json.dumps(reasons,sort_keys=True))
            if identity==previous and failures[-1]['end_row_exclusive']==i:
                failures[-1]['end_row_exclusive']=i+1;failures[-1]['last_time_s']=row['time_s']
            else:failures.append(dict(target=row['target'],start_row=i,end_row_exclusive=i+1,time_s=row['time_s'],last_time_s=row['time_s'],reasons=reasons))
            previous=identity
        else:previous=None
    point_records=[]
    for point in ep['outcome']['points']:
        record={key:point.get(key) for key in ('target','success','training_accepted','start_physics_row','end_physics_row','final_phase')}
        record['failure_fields']={key:value for key,value in point.items() if any(s in key.lower() for s in ('fail','error','reason','abort','recover'))}
        point_records.append(record)
    return dict(recorded_success=ep['outcome'].get('success'),recorded_training_accepted=ep['outcome'].get('training_accepted'),
        recorded_completed_points=sum(p.get('success') is True for p in ep['outcome']['points']),requested_points=ep['requested'],
        source_row_count=ep['n'],source_duration_s=ep['n']*ep['dt'],point_records=point_records,trace_failure_events=failures,
        recorded_audits=ep.get('recorded_audits',{}),geometry=geometry_diagnostics(ep),
        scope='Recorded outcome/audit flags and failures retained as evidence; metrics do not re-run or infer acceptance.')


def _complete_full10(ep):
    return (ep['requested']==list(TARGETS) and len(ep['outcome']['points'])==10
            and all(p.get('success') is True for p in ep['outcome']['points'])
            and ep['outcome'].get('success') is True and ep['rows'][-1]['subphase']=='home_settle')


def compare_prepared(baseline,alternative):
    b,a=baseline,alternative;requested=a['requested']
    require(requested==b['requested'][:len(requested)],'baseline does not cover requested prefix')
    require(abs(b['dt']-a['dt'])<1e-12,'sampling rates differ between comparison inputs')
    final=requested[-1];be,ae=_hold_end(b,final),_hold_end(a,final)
    full=_complete_full10(b) and _complete_full10(a)
    if full:be,ae=b['n'],a['n']
    primary=_pair(b,a,(0,be) if be else None,(0,ae) if ae else None,
        window_kind='full10_complete_route' if full else 'requested_prefix_through_final_HOLD',requested_points=requested)
    if be is None or ae is None:primary.update(status='unavailable_incomplete_requested_window',delta=None)
    upper_targets=requested[:3];upper_last=upper_targets[-1]
    upper_b,upper_a=_hold_end(b,upper_last),_hold_end(a,upper_last)
    upper_pair=_pair(b,a,(0,upper_b) if upper_b else None,(0,upper_a) if upper_a else None,
                    window_kind='upper_pair_through_R2_HOLD' if len(upper_targets)==3 else 'available_upper_prefix_through_HOLD',
                    requested_points=upper_targets)
    if upper_b is None or upper_a is None:upper_pair.update(status='unavailable_incomplete_requested_window',delta=None)
    points=[]
    for target in requested:
        left,right=_point(b['outcome'],target),_point(a['outcome'],target)
        def bounds(p,end,ep):
            end=ep['n'] if end is None else end
            return (p['start_physics_row'],min(p['end_physics_row'],end)) if p and p['start_physics_row']<end else None
        pair=_pair(b,a,bounds(left,be,b),bounds(right,ae,a),target=target,
                   scope='primary-window point' if be and ae else 'partial point diagnostic')
        if not left or not right or left.get('success') is not True or right.get('success') is not True:
            pair.update(status='incomplete_point_diagnostic',delta=None)
        if target==final and (be is None or ae is None):
            pair.update(status='incomplete_requested_terminal_point',delta=None)
        points.append(pair)
    bs,ass=_stages(b,be),_stages(a,ae)
    # If the requested terminal HOLD was never reached, use only recorded rows
    # for secondary diagnostics, and never substitute their total for primary.
    stages=[]
    for key in sorted(set(bs)|set(ass)):
        if key[0] in requested:
            pair=_pair(b,a,bs.get(key),ass.get(key),key=list(key),
                       scope='primary-window stages' if be and ae else 'partial stage diagnostic')
            left,right=_point(b['outcome'],key[0]),_point(a['outcome'],key[0])
            if not left or not right or left.get('success') is not True or right.get('success') is not True:
                pair.update(status='incomplete_point_stage_diagnostic',delta=None)
            stages.append(pair)
    contacts={side:{target:contact_point(ep,target) for target in requested} for side,ep in (('baseline',b),('alternative',a))}
    transitions={}
    for side,ep,end in (('baseline',b,be),('alternative',a,ae)):
        events=[]
        for i in range(1,ep['n'] if end is None else end):
            prev,row=ep['rows'][i-1],ep['rows'][i]
            if row['target'] not in requested:continue
            if any(row[k]!=prev[k] for k in ('target','phase','subphase')):
                events.append(dict(before=_event(ep,i-1),after=_event(ep,i),dynamics=_event_dynamics(ep,i)))
        transitions[side]=events
    terminal={}
    for side,ep,end in (('baseline',b,be),('alternative',a,ae)):
        p=_point(ep['outcome'],final);tail_end=p['end_physics_row'] if p else ep['n']
        terminal[side]=summarize(ep,end,tail_end) if end and end<tail_end else None
    same={};changed=[]
    bp=b.get('sources',{}).get('captured_plans',{});ap=a.get('sources',{}).get('captured_plans',{})
    for target in requested:
        if target in bp and target in ap:
            same[target]=bp[target]['sha256']==ap[target]['sha256']
            if target not in ('R1','L1') and not same[target]:changed.append(target)
    return dict(schema='upper_pair_actual_trace_comparison_v1',joint_convention=JOINT_CONVENTION,
        primary_comparison=primary,upper_pair_comparison=upper_pair,full10_comparison=primary if full else None,point_comparisons=points,
        stage_comparisons=stages,contact_by_point=contacts,transition_events=transitions,
        terminal_diagnostics=terminal,terminal_scope='Unmatched final requested-point tail only; not ranked. Baseline later points excluded.',
        episode_diagnostics={'baseline':_diagnostics(b),'alternative':_diagnostics(a)},
        source_bindings={'baseline':b.get('sources',{}),'alternative':a.get('sources',{})},
        shared_plan_hashes_equal=same,unexpected_nonupper_plan_changes=changed,
        fixed_plan_comparison_status='mismatch' if changed else 'matching_captured_nonupper_plans' if same else 'not_checked_in_memory_only',
        methods=dict(raw='Repeated central finite differences of q with NumPy gradient edge_order=2, computed on full trace BEFORE slicing; no unwrap/clipping.',
            actual='q is measured joint position; raw derivatives are not the separately recorded dq.',
            commanded='q_command sampled at every physics row; repeated controller commands remain in raw derivatives.',
            filtering='Savitzky-Golay polynomial 4 derivatives, centered window, interp global ends; full trace BEFORE slicing.',
            filter_requested_window_s=FILTER_WINDOW_S,filter_samples={'baseline':b['filter_samples'],'alternative':a['filter_samples']},
            filter_actual_span_s={'baseline':(b['filter_samples']['actual']-1)*b['dt'],'alternative':(a['filter_samples']['actual']-1)*a['dt']},
            dt_s=b['dt'],travel='Absolute increments assigned to arriving row. Window first increment includes preceding row; first trace increment zero. Disjoint stage sums conserve travel.',
            reversals='Original >1 degree cumulative excursion detector on full trace, assigned to confirmation row.',
            posture='240 Hz URDF FK of measured q plus separately labeled saved body positions at 30 Hz when available; q4 is not spatial elbow orientation.',
            endpoint_context='Raw derivative chains use up to 3 neighboring rows; filtered estimates use a centered halo. No differentiation after slicing.',
            peak_events='Per-joint absolute peaks preserve signed value, original global row and time; no clipping.'),
        physics_acceptance_inferred=None,clinical_acceptance_inferred=None,automatic_winner=None,
        scope='Descriptive matched-window metrics, not new geometry/physics acceptance or clinical/acoustic validity. '
              'Failed and incomplete runs retain their evidence; a shorter failed trace cannot win by lower total travel.')


def load_episode(directory):
    directory=Path(directory).resolve();provenance=validate_episode_provenance(directory)
    require(provenance.get('status')=='passed','episode provenance rejected: '+str(provenance.get('reasons')))
    hashes=provenance.get('artifact_sha256')
    require(isinstance(hashes,dict) and hashes and not verify_hashes(directory,hashes),'episode artifact hash mismatch')
    outcome=read(directory/'outcome.json');base=read(directory/'world_from_left_base.json')
    keep=set(equations.SCALARS)|{'target','phase','subphase','q','q_command','dq','tcp_world','axis_world',
                                'hold_elapsed_s','physical_violations','posture_violations','recovery'}
    rows=[]
    with (directory/'expert_trace.jsonl').open() as stream:
        for line in stream:
            require(line.strip(),'empty trace record')
            row=json.loads(line,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite '+value)))
            rows.append({key:row[key] for key in keep if key in row})
    with np.load(directory/'expert_replay.npz',allow_pickle=False) as archive:
        replay={key:archive[key] for key in ('row_indices','poses','paths')}
    prepared=prepare_episode(rows,outcome,base,replay)
    manifest_path=(directory/outcome['input_snapshot']).resolve()
    require(directory in manifest_path.parents,'unsafe captured snapshot')
    snapshot=read(manifest_path);plans={}
    for target in outcome['requested_points']:
        entry=snapshot['files'].get(target+'/plan')
        require(isinstance(entry,dict),'missing captured plan for '+target)
        digest=entry['sha256'];obj=manifest_path.parent/'objects'/digest
        require(directory in obj.resolve().parents and hashes.get(str(obj.relative_to(directory)))==digest
                and sha256(obj)==digest,'captured plan hash mismatch: '+target)
        plan=read(obj);point=plan['points'][0]
        require(point['name']==target,'captured plan target mismatch')
        plans[target]=dict(sha256=digest,snapshot_key=target+'/plan',object_path=str(obj.relative_to(directory)),
            declared_free_clearance_m=point.get('geometric_clearance_m'),
            declared_loaded_noncontact_clearance_m=point.get('loaded_noncontact_clearance_m'))
    prepared['sources']=dict(episode=str(directory),artifact_sha256=hashes,provenance=provenance,captured_plans=plans)
    prepared['recorded_audits']={name:{key:read(directory/name).get(key) for key in ('accepted','reasons','scope')}
                                for name in ('independent_audit.json','posture_audit.json')}
    require(not verify_hashes(directory,hashes),'episode artifact changed during analysis')
    return prepared


def _markdown(report):
    primary=report['primary_comparison'];lines=['# 上部点位姿态：实测轨迹比较','',
        '仅比较实际记录的对应窗口；不推断新的物理通过、专家认可或临床/声学有效性。', '',
        '主窗口：'+primary['window_kind']+'；状态：'+primary['status']+'。',
        '三点筛查截至 R2 HOLD 结束。R2 回 HOME 与基线继续下一点的不同末段不计入主比较。', '',
        '| 指标 | 基线 | 替代 | 替代−基线 |','|---|---:|---:|---:|']
    b,a=primary['baseline'],primary['alternative']
    if b is not None and a is not None:
        getters=[('时长 (s)',lambda x:x['window']['duration_s']),
            ('实测总绝对行程 (rad)',lambda x:x['actual']['total_travel_rad']),
            ('实测腕关节 5–7 行程 (rad)',lambda x:x['actual']['wrist_joints_5_7_travel_rad']),
            ('实测关节 4 行程 (rad)',lambda x:x['actual']['elbow_flexion_joint_4_travel_rad']),
            ('实测 q 原始速度峰 (rad/s)',lambda x:_max_value(x,'velocity')),
            ('实测 q 原始加速度峰 (rad/s²)',lambda x:_max_value(x,'acceleration')),
            ('实测 q 原始 jerk 峰 (rad/s³)',lambda x:_max_value(x,'jerk')),
            ('实测 q 滤波 jerk 峰 (rad/s³)',lambda x:_max_value(x,'jerk','filtered')),
            ('实测 q FK 肘最高 (m)',lambda x:x['posture_measured_q_fk']['elbow_max_z_m']),
            ('腕高−肘高最小 (m)',lambda x:x['posture_measured_q_fk']['minimum_wrist_minus_elbow_z_m']),
            ('q−命令 跟踪 RMS (rad)',lambda x:x['tracking']['overall_rms_rad'])]
        for label,getter in getters:
            left,right=getter(b),getter(a);lines.append(f'| {label} | {left:.9g} | {right:.9g} | {right-left:+.9g} |')
        lines+=['','窗口行号（左闭右开）及物理时段：']
        for label,value in [('基线',b),('替代',a)]:
            w=value['window'];lines.append(f'- {label}：[{w["start_row"]}, {w["end_row_exclusive"]})；{w["interval_start_s"]:.9g}–{w["interval_end_s"]:.9g} s。')
        lines+=['','| 关节 | 基线实测行程 rad | 替代实测行程 rad |','|---|---:|---:|']
        for j in range(7):lines.append(f'| {j+1} | {b["actual"]["travel_rad"][j]:.9g} | {a["actual"]["travel_rad"][j]:.9g} |')
    else:lines+=['| 主窗口未完成 | 不提供总量优劣排名 | 保留部分阶段诊断 | — |']
    lines+=['','HOLD 与接触记录：','','| 点 | 基线 HOLD s | 替代 HOLD s | 基线 Hall 均值/标准差 N | 替代 Hall 均值/标准差 N |','|---|---:|---:|---:|---:|']
    def hold_text(point):
        if point is None or point['hold']['hall'] is None:return '缺失','缺失'
        hold=point['hold'];return f'{hold["duration_s"]:.9g}',f'{hold["hall"]["mean_N"]:.6g} / {hold["hall"]["std_N"]:.6g}'
    for target in primary['requested_points']:
        bt,bf=hold_text(report['contact_by_point']['baseline'][target]);at,af=hold_text(report['contact_by_point']['alternative'][target])
        lines.append(f'| {target} | {bt} | {at} | {bf} | {af} |')
    lines+=['','记录结果（沿用来源标记，不是本比较重新验收）：']
    for side in ('baseline','alternative'):
        d=report['episode_diagnostics'][side]
        lines.append(f'- {side}：控制器成功={d["recorded_success"]}，完成点数={d["recorded_completed_points"]}/{len(d["requested_points"])}；原始 trace {d["source_row_count"]} 行。')
        for p in d['point_records']:
            if p['success'] is not True:lines.append(f'  - {p["target"]} 未完成；源失败字段：{json.dumps(p["failure_fields"],ensure_ascii=False)}')
    if report['full10_comparison'] is not None:
        upper=report['upper_pair_comparison']
        lines+=['','另保留 R1 开始至 R2 HOLD 结束的固定上部窗口（不被完整十点总量替代）：']
        for side in ('baseline','alternative'):
            metric=upper[side]
            if metric:
                w=metric['window'];lines.append(f'- {side}：[{w["start_row"]}, {w["end_row_exclusive"]})，{w["duration_s"]:.9g} s；实测总行程 {metric["actual"]["total_travel_rad"]:.9g} rad，腕组 {metric["actual"]["wrist_joints_5_7_travel_rad"]:.9g} rad。')
    lines+=['','方法与边界：', '',
        '- 全轨迹先算原始中心差分及 Savitzky–Golay 4 阶、约 0.1 s 窗的滤波导数，再切片。原始 jerk 放大采样噪声；命令、实测 q 导数、记录 dq 分列。',
        '- 腕组为关节 5–7（数组索引 4–6）；关节 4（索引 3）的屈曲行程不等于空间翻肘。',
        '- 每个绝对增量归于到达的行；不包角、不削峰。阶段起点导数保留邻近真实样本，不重新在小窗求导。',
        '- JSON 含逐点/阶段行号、峰值时刻、力与释放转换、带载撤回、原始失败、输入与计划 hash。阶段覆盖不同时单列缺失，不用平均分掩盖。',
        '- FCL 通过记录中的 minimum_distance_m 可能封顶于门槛；它不是精确净空裕量，也不是连续或实体碰撞证书。',
        '- 仅双方完整十点时提供完整路线总量；不自动选择赢家。', '',
        '来源与分析文件 SHA256 见 comparison.json 的 source_bindings / analysis_sources。','']
    return '\n'.join(lines)


def _analysis_sources():
    files=[Path(__file__),Path(equations.__file__),PROJECT/'sim/curobo_expert_v1/quality.py',
           PROJECT/'sim/curobo_expert_v1/provenance.py']
    files += [PROJECT/'sim/auscultation_expert_layout_v1'/name for name in
              ('kinematics.py','audit_low_elbow.py','targets.py','posture.py','tool_frames.py','openarm_v10_follower_stetho.urdf')]
    return {str(path.resolve()):sha256(path) for path in files}


def compare_episodes(baseline,alternative,output_dir):
    baseline=Path(baseline).resolve();alternative=Path(alternative).resolve();output=Path(output_dir).resolve()
    require(all(output!=source and source not in output.parents for source in (baseline,alternative)),
            'output must be outside source episodes')
    sim=PROJECT/'sim';own=Path(__file__).resolve().parent
    require(sim not in output.parents or own==output or own in output.parents,'output must not modify old simulation packages')
    analysis_hashes=_analysis_sources()
    b=load_episode(baseline);a=b if baseline==alternative else load_episode(alternative)
    report=compare_prepared(b,a)
    for ep in (b,a):require(not verify_hashes(Path(ep['sources']['episode']),ep['sources']['artifact_sha256']),
                           'episode artifact changed during comparison')
    require(all(sha256(path)==digest for path,digest in analysis_hashes.items()),'analysis source changed')
    report['analysis_sources']=analysis_hashes
    import scipy
    report['library_versions']={'numpy':np.__version__,'scipy':scipy.__version__}
    markdown=_markdown(report)
    output.mkdir(parents=True,exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'comparison.md').write_text(markdown)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--alternative',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();report=compare_episodes(args.baseline,args.alternative,args.output_dir)
    print(json.dumps({'comparison':report['primary_comparison']['status'],'window':report['primary_comparison']['window_kind'],
                      'output_dir':str(args.output_dir.resolve()),'automatic_winner':None}))


if __name__=='__main__':main()
