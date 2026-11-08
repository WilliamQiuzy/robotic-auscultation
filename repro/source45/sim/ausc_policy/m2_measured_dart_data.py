"""Fixed twenty-request TRAIN-only measured late-DART recovery pilot.

Explicit Source34/Source37 profiles retain every terminal collection outcome.
No training integration, owner invocation, query-prefix promotion or stats fit.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np

from . import m2_given_contact_dataset as data, m2_given_contact_act as given
from . import m2_given_contact_recovery as recovery, m2_given_contact_recovery_dataset as roles
from . import m2_given_contact_runtime as rt, m2_teacher_stop_reader as teacher
from . import m2_measured_recovery_data as measured
from .retainer_contact import ALL_FORCE_GROUPS

need=rt.need
SCHEMA='m2_measured_late_dart_pilot_bank_v1'
MANIFEST_SCHEMA='m2_measured_late_dart_pilot_manifest_v1'
ADMISSION_SCHEMA='m2_measured_late_dart_pilot_admission_v1'
CPU_PLAN_SCHEMA='m2_measured_dart_precollection_cpu_plan_v1'
PROFILE='production_source34_late_dart_compact_teacher_v1'
PROFILE_SOURCE37='production_source37_late_dart_compact_teacher_v1'
SUBJECTS=('p000758','p001031','p000833','p000961','p000742')
POSES=('p000474_a001230_f0000','p000476_a001231_f0023')
TARGETS=('R3','L3')
BASE29=deepcopy(measured.BASE29)
BASE_COUNTS=deepcopy(measured.BASE_COUNTS)
NORMALIZATION=dict(path='/opt/wam-runtime/fast-wam/datasets/act-student-recovery-masked-20261106-v1/normalization.json',
    bytes=2173,sha256='6302de1c94e34b193249081e1ddf4a8e1234a6361829e23d28567fd8d342651b')
ARRAY_KEYS=measured.ARRAY_KEYS
OUTCOMES=('collected','cpu_unavailable','preview_rejected','physical_negative','infrastructure_error')


class PhysicalNegative(ValueError):
    """Independently observed physical negative after provenance validation."""


def planned_requests():
    return [dict(slot_id='dart-'+str(i).zfill(2),split='train',subject_id=s,pose_id=p,target_id=t,
        seed=20261107+i,amplitude_rad=.015,trigger_distance_m=.035)
        for i,(s,p,t) in enumerate((s,p,t) for s in SUBJECTS for p in POSES for t in TARGETS)]


def _profile(manifest):
    if manifest.get('profile')==PROFILE:
        teacher.validate_dart_profile(manifest.get('evidence_profile'),manifest.get('recorded_source'))
        return teacher.validate_source34_header,teacher._source34_configuration_and_sources,teacher.decode_source34_dart_capture
    need(manifest.get('profile')==PROFILE_SOURCE37,'Explicit approved DART bank profile required')
    teacher.validate_dart37_profile(manifest.get('evidence_profile'),manifest.get('recorded_source'))
    return teacher.validate_source37_header,teacher._source37_configuration_and_sources,teacher.decode_source37_dart_capture


def validate_manifest(manifest,base):
    """Validate ALL declared identities and references before ANY payload read."""
    need(isinstance(manifest,dict) and set(manifest)=={'schema','profile','evidence_profile','recorded_source',
        'base_dataset','normalization','attempts','precollection'} and manifest.get('schema')==MANIFEST_SCHEMA,
        'Explicit separate DART pilot manifest/profile required')
    _profile(manifest)
    need(manifest['base_dataset']==BASE29 and manifest['normalization']==NORMALIZATION,
        'Exact unchanged Base29 dataset/normalization references required')
    attempts=manifest['attempts'];planned=planned_requests()
    need(isinstance(attempts,list) and len(attempts)==20,'Exactly twenty ordered planned requests required')
    # First pass contains no references, filesystem reads, or callbacks.
    for attempt,request in zip(attempts,planned):
        need(isinstance(attempt,dict) and set(attempt)=={'request','nominal_source_config','episode','outcome','reason','evidence'} and
            attempt.get('request')==request,'Exact canonical TRAIN request identity/order/settings required; no replacement')
        row=attempt['episode']
        if row is not None:
            need(isinstance(row,dict) and all(row.get(k)==request[k] for k in ('split','subject_id','pose_id','target_id')),
                'Episode must match original planned TRAIN identity/pose/target')
    for attempt in attempts:
        outcome=attempt['outcome'];row=attempt['episode'];nominal=attempt['nominal_source_config']
        need(outcome in OUTCOMES and isinstance(attempt['reason'],str) and
            (outcome=='collected' or bool(attempt['reason'].strip())) and isinstance(attempt['evidence'],dict),
            'Explicit terminal unavailable/error/physical outcome and reason required')
        need(outcome not in ('collected','physical_negative','preview_rejected') or row is not None,'Native outcome requires actual episode declaration')
        need(nominal is not None or outcome in ('cpu_unavailable','infrastructure_error'),
            'Original nominal source config reference required')
        for ref in ([nominal] if nominal is not None else [])+list(attempt['evidence'].values()):data._ref_path(ref,base)
        if row is not None and outcome!='infrastructure_error':data._declarations([row],base)
    data._ref_path(manifest['precollection'],base)
    return attempts


def _utc(stamp):
    need(isinstance(stamp,str),'Explicit frozen UTC timestamp required')
    value=datetime.fromisoformat(stamp.replace('Z','+00:00'))
    need(value.tzinfo is not None and value.utcoffset()==timezone.utc.utcoffset(value),'Timezone-aware UTC timestamp required')
    return value


def validate_precollection(plan):
    need(isinstance(plan,dict) and set(plan)=={'schema','frozen_utc','frozen_before_first_native','slots'} and
        plan.get('schema')==CPU_PLAN_SCHEMA and plan.get('frozen_before_first_native') is True,
        'Explicit complete CPU preparation plan frozen before first Native required')
    _utc(plan['frozen_utc']);slots=plan['slots']
    need(isinstance(slots,list) and len(slots)==20,'All twenty precollection CPU preparation statuses required')
    for slot,request in zip(slots,planned_requests()):
        need(isinstance(slot,dict) and set(slot)=={'request','status','reason','evidence'} and slot.get('request')==request and
            slot.get('status') in ('prepared','cpu_unavailable') and isinstance(slot.get('reason'),str) and
            (slot['status']=='prepared' or bool(slot['reason'].strip())) and isinstance(slot.get('evidence'),dict),
            'Exact precollection CPU identity/status/unavailable reason required')
    return plan


def _precollection(manifest,base):
    plan,_=data._bound(manifest['precollection'],base);validate_precollection(plan)
    # Validate every CPU identity/status before opening referenced CPU metadata.
    for slot,attempt in zip(plan['slots'],manifest['attempts']):
        need((attempt['outcome']=='infrastructure_error' or attempt['outcome']=='cpu_unavailable' and attempt['episode'] is None) if slot['status']=='cpu_unavailable' else
            attempt['outcome']!='cpu_unavailable','Frozen prepared/unavailable classification cannot change after Native/preview outcomes')
        for ref in slot['evidence'].values():data._bound(ref,base)
    return plan


def admission(ledger,*,precollection):
    validate_precollection(precollection)
    need(isinstance(ledger,list),'Explicit terminal DART outcome ledger required')
    planned=planned_requests()
    order=isinstance(ledger,list) and len(ledger)==20 and all(e.get('request')==r for e,r in zip(ledger,planned))
    if order:
        for entry,slot in zip(ledger,precollection['slots']):
            need((entry.get('status') in ('cpu_unavailable','infrastructure_error')) if slot['status']=='cpu_unavailable' else
                entry.get('status')!='cpu_unavailable','Native/preview outcomes cannot be relabeled CPU-unavailable')
    qualified=[e for e in ledger if e.get('status')=='qualified' and type(e.get('windows')) is int and e['windows']>0]
    prepared=sum(s['status']=='prepared' for s in precollection['slots'])
    try:
        identities=[_reset(e) for e in qualified]
        independent=len(set(identities))==len(identities)
    except (ValueError,KeyError,TypeError):independent=False
    covered=all({e['request'][k] for e in qualified}==set(values) for k,values in
        (('subject_id',SUBJECTS),('pose_id',POSES),('target_id',TARGETS))) if order else False
    clean=order and independent and all(e.get('status') in ('qualified','cpu_unavailable','preview_rejected','physical_negative','infrastructure_error') and
        e.get('infrastructure_contamination') is False and e.get('status')!='infrastructure_error' and
        (e.get('status')!='qualified' or e in qualified) for e in ledger)
    eligible=bool(order and clean and prepared>=16 and len(qualified)*100>=95*prepared and covered)
    return dict(schema=ADMISSION_SCHEMA,pilot_optimization_eligible=eligible,fixed_attempt_denominator=20,
        prepared_episode_denominator=prepared,minimum_prepared_episodes=16,minimum_qualified_episodes=(95*prepared+99)//100,
        minimum_prepared_success_percent=95,qualified_episodes=len(qualified),all_identity_pose_target_coverage=covered,
        no_infrastructure_contamination=bool(clean),bulk_expert_qualification=False,model_qualified=False,
        actual_student_failure_data=False,formal_acceptance=False)


def validate_samples(samples):
    measured.validate_samples(samples)
    need(np.array_equal(samples['velocities'],(samples['histories'][:,1,:7]-samples['histories'][:,0,:7])/.05),
        'Actual H2 finite-difference velocity differs')
    windows=samples.get('derived_index',{}).get('windows')
    need(isinstance(windows,list) and len(windows)==len(samples['states']),'Actual ordered DART window provenance required')
    for i,window in enumerate(windows):
        ticks=window.get('inclusive_native_ticks');n=int((~samples['action_is_pad'][i]).sum())
        need(isinstance(ticks,list) and len(ticks)==2 and all(type(t) is int for t in ticks),'Actual native window interval required')
        first=ticks[0];observations=window.get('observation_indices',[])
        need(first>=23 and (first+1)%12==0 and ticks[1]==first+12*n and
            window.get('history_native_ticks')==[first-12,first] and
            window.get('raw_intent_native_ticks')==[first+1,first+12*n] and
            abs(float(samples['observation_times'][i])-(first+1)/240.)<=1e-8 and
            window.get('actual_control_owner')=='TEACHER_L2' and window.get('execution_role')=='given_l2_motion' and
            len(observations)==2 and observations[1]==observations[0]+1 and
            window.get('action_record_indices')==list(range(observations[1],observations[1]+n)),
            'Actual DART window owner/history/native clock/future boundary differs')
    return samples


def _base(manifest):
    base=rt.read_bound(BASE29)
    need(base.get('complete') is True and base.get('counts')==BASE_COUNTS and base.get('normalization')==NORMALIZATION,
        'Immutable Base29 counts/normalization reference differs')
    rt.validate_normalization(rt.read_bound(NORMALIZATION))
    return base


def _config(attempt,base):
    row=attempt['episode'];request=attempt['request'];config,_=data._bound(row['config'],base)
    binding=recovery.preflight(config);settings=binding['settings'];meta=binding['metadata']
    need(settings==dict(schema=recovery.LATE_SCHEMA,source_config=attempt['nominal_source_config'],
        episode_id=row['episode_id'],**{k:request[k] for k in ('seed','amplitude_rad','trigger_distance_m')}),
        'Exact planned late-DART source projection/settings required')
    need(all(meta.get(k)==request[k] for k in ('subject_id','split','target_id')),'Configured planned TRAIN identity differs')
    patient=rt.pose.bound_json(binding['reference'].snapshot()['sources']['patient_receipt'])
    need(patient.get('identity')==request['subject_id'] and patient.get('split')=='train' and
        patient.get('proposal')==request['subject_id']+'_'+request['pose_id'],'Actual installed TRAIN body/pose differs')
    need(teacher.residual.CONFIG_KEY in config,'Original nominal source must already carry STOP8')
    teacher.residual.native_options(config,metadata=meta)
    return config,binding


def _headers(manifest,base):
    attempts=validate_manifest(manifest,base);precollection=_precollection(manifest,base);errors={};seen={}
    # All configured/compact identity metadata precedes reference verification,
    # asset preflights, Base29 statistics, and the first native payload.
    for attempt in attempts:
        row=attempt['episode'];request=attempt['request'];slot=request['slot_id']
        if row is None or attempt['outcome']=='infrastructure_error':continue
        refs=[('config',row['config']),('config',attempt['nominal_source_config'])]
        if attempt['outcome']=='collected':
            compact=row.get('artifacts',{}).get('compact/manifest.json',{})
            refs.append(('compact',dict(compact,path=str(data._path(row['capture_root'],base)/'compact/manifest.json'))))
        for kind,ref in refs:
            try:value,_=data._bound(ref,base)
            except (ValueError,KeyError,TypeError,OSError) as error:
                errors[slot]=type(error).__name__+': '+str(error)[:2000];continue
            meta=value.get('metadata',{}) if kind=='compact' else value.get('m2_given_contact_control',{}).get('metadata',{})
            need(all(meta.get(k)==request[k] for k in ('split','subject_id','target_id')),
                'All configured/compact identities must match canonical TRAIN metadata before payloads')
    _base(manifest)
    for attempt in attempts:
        slot=attempt['request']['slot_id'];row=attempt['episode']
        try:
            need(slot not in errors,errors.get(slot,''))
            for ref in attempt['evidence'].values():rt.bound_path(ref)
            if row is None:continue
            if attempt['outcome']=='infrastructure_error':continue
            for key,token in (('episode',row['episode_id']),('reset',row['reset_id']),
                    ('capture',str(data._path(row['capture_root'],base))),('result',row['run_result']['sha256']),
                    ('job',(row['run_id'],row['job_id']))):
                need(token not in seen.setdefault(key,set()),'Duplicate independent native declaration: '+key)
                seen[key].add(token)
            _config(attempt,base)
            owner,_=data._bound(row['owner_terminal'],base)
            need(_utc(owner.get('started_utc'))>_utc(precollection['frozen_utc']),
                'CPU preparation denominator was not frozen before actual Native owner launch')
            need(owner.get('config',{}).get('measured_dart_precollection')==manifest['precollection'],
                'Original owner config must hash-bind the frozen precollection CPU denominator')
            declared=[a['episode'] for a in attempts if a['episode'] is not None and a['episode'].get('run_id')==row['run_id']]
            jobs=owner.get('config',{}).get('jobs',[])
            need(len(jobs)==len(declared) and {j.get('id') for j in jobs}=={r['job_id'] for r in declared} and
                all(any(j.get('id')==r['job_id'] and j.get('sha256')==r['config']['sha256'] and
                    data._path(j.get('config'),base)==data._ref_path(r['config'],base) for j in jobs) for r in declared),
                'Original owner contains undeclared/replaced/missing pilot workers')
            if attempt['outcome']=='collected':data._metadata(row,base)
        except (ValueError,KeyError,TypeError,OSError) as error:errors[slot]=type(error).__name__+': '+str(error)[:2000]
    return attempts,errors,precollection


def _context(row,base,manifest,freeze):
    header,_,_=_profile(manifest)
    header(freeze,evidence_profile=manifest['evidence_profile'],recorded_source=manifest['recorded_source'])
    return dict(evidence_profile=manifest['evidence_profile'],recorded_source=deepcopy(manifest['recorded_source']),
        source_freeze=freeze,run_id=row['run_id'],source_declaration=deepcopy(row),evidence_base=str(Path(base).resolve()))


def _reset(receipt):
    identity=receipt.get('native_reset_identity',{})
    need(all(type(identity.get(k)) is int and identity[k]>0 for k in ('pid','sid')) and
        isinstance(identity.get('start_ticks'),str) and identity['start_ticks'].isdigit(),
        'Actual independent native reset process identity required')
    return identity['pid'],identity['sid'],identity['start_ticks']


def _cumulative_l2(run,audit):
    rows=[i for i,e in enumerate(audit) if e.get('actual_control_owner')=='TEACHER_L2']
    handoffs=run.get('l2_handoffs',[])
    need(rows and len(handoffs)==1,'Actual original L2 ownership/parking interval missing')
    first=rows[0];end=handoffs[0].get('parking_end_physics_row')
    need(type(end) is int and first<end<=len(audit) and all(e.get('actual_control_owner') in ('TEACHER_L2','L2_STOP')
        for e in audit[first:end]),'Actual cumulative L2 ownership/parking interval changed')
    duration=(end-first)/240.
    if duration>120.:
        error=PhysicalNegative('Measured cumulative L2 through parking exceeds120s; no clock reset allowed')
        error.evidence=dict(first_native_interval_start_row=first,parking_end_row_exclusive=end,duration_s=duration,maximum_s=120.)
        raise error
    return dict(first_native_interval_start_row=first,parking_end_row_exclusive=end,duration_s=duration,
        maximum_s=120.,enforcement='independent_offline_reconstruction_of_inherited_runtime_deadline')


def _validate_dart_trace(sidecar,audit,records,metadata,reference,settings,*,trajectory_factory=None,kin=None):
    """Bind the actual compact fields; no absent native q/dq mirror is assumed.

    The original sidecar records full measured dq. Compact native evidence
    independently records its maximum magnitude, plus complete captured q/H2.
    Replaying actual setters checks the retained command q/v/a across DART.
    """
    roles.validate_sidecar(sidecar,audit,metadata)
    need(settings.get('schema')==recovery.LATE_SCHEMA and sidecar.get('declaration')==settings,
        'Only actual late-DART declaration may enter this trace verifier')
    queries=sidecar.get('queries')
    need(isinstance(queries,list) and [q.get('kind') for q in queries]==['perturbation_preview','recovery_proposal'],
        'Exactly actual DART perturbation and measured expert replan required; no query replay terminal')
    before,after=queries;pre=before['query'];post=after['query']
    for query in (pre,post):
        recovery._validate_query(query,reference)
        row=query['physics_row'];stamp=query['time_s']
        need(query['metadata']==metadata and 3<=row<len(audit),'Actual DART query metadata/native row differs')
        observations=[]
        for time in (stamp-.05,stamp):
            matches=[r for r in records if abs(r['observed_at_s']-time)<=1e-8]
            need(len(matches)==1,'Actual captured causal H2 missing or duplicate at DART query');observations.append(matches[0])
        need(np.array_equal(query['history'],[r['state17'] for r in observations]) and
            np.array_equal(query['measured_q'],observations[-1]['measured_q']),
            'Actual DART measured q/H2 differs from real compact capture')
        actual=audit[row-1].get('measurements',{})
        contacts=sum(actual.get(k+'_N',float('nan')) for k in ALL_FORCE_GROUPS)
        need(query['hall_N']==actual.get('hall_N') and query['compression_m']==actual.get('compression_m') and
            query['physical_contact_N']==contacts and
            float(np.max(abs(np.asarray(query['measured_dq']))))==actual.get('max_abs_joint_velocity_rad_s'),
            'Actual existing native DART release/velocity scalar witness differs')
    start=pre['physics_row'];cut=post['physics_row'];perturb=[e for e in sidecar['events'] if e['kind']=='perturbation']
    need(perturb[0]['physics_row_start']==start and perturb[-1]['physics_row_end_exclusive']==cut and
        cut-start==12*recovery.PERTURB_CELLS and after.get('query_sha256')==rt.digest(post),
        'Actual complete perturbation interval/postperturb measured query binding differs')
    rng=np.random.default_rng(settings['seed']);offset=rng.normal(size=7);offset*=settings['amplitude_rad']/max(abs(offset))
    first_preview=before['preview'];second_preview=after['preview'];state=first_preview['input_state']
    target=np.asarray(state['fields']['target_position'])+offset
    need(np.array_equal(before.get('offset_rad'),offset) and np.array_equal(before.get('target_q'),target),
        'Actual DART target differs from frozen seed/max-absolute-component radians')
    rt.geometry.require_current(metadata)
    if kin is None:
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        kin=OpenArmKinematics(contact_frame=rt.geometry.contact_frame)
    distance=float(np.linalg.norm(kin.tcp(pre['measured_q'])[:3,3]-reference.pose_base[:3,3]))
    need(distance<=settings['trigger_distance_m'],'Actual late-DART trigger is 3D skin-target distance, not prepose error')
    for preview in (first_preview,second_preview):
        need(preview.get('input_sha256')==rt.digest(preview.get('input_state')) and
            preview.get('all_240hz_commands_verified') is True and preview.get('physics_executed') is False and
            type(preview.get('steps')) is int and preview['steps']>0,
            'Actual original full retained-state preview report required')
    h=rt.handoff;clone=h.restore_trajectory(state,trajectory_factory)
    need(np.array_equal(state['fields']['current_position'],audit[start-1]['q_command']),
        'Actual DART retained command seam differs')
    route=np.asarray(after.get('route'),dtype=float)
    need(route.ndim==2 and route.shape[1:]==(7,) and len(route)>=2 and np.isfinite(route).all() and
        np.array_equal(route[0],second_preview['input_state']['fields']['current_position']) and
        np.array_equal(route[-1],after.get('destination_q')),'Actual measured recovery route/retained seam differs')
    plan=rt.pose.bound_json(reference.snapshot()['sources']['fresh_plan'])
    need(any(np.array_equal(route[-1],q) for q in plan.get('transfer_q',[])[1:]),
        'Actual measured recovery destination differs from authenticated original plan')
    actual_recovery=[e for e in sidecar['events'] if e['kind']=='recovery_teacher']
    need(actual_recovery and actual_recovery[0]['physics_row_start']==cut,'Measured postperturb recovery has no executed teacher setters')
    end=actual_recovery[-1]['physics_row_end_exclusive'];route_index=1
    history=np.asarray([e['q_command'] for e in audit[start-3:start]],dtype=float)
    for i in range(start,end):
        event=audit[i];raw=event.get('raw_intent') or {};setter=raw.get('setter') or {}
        if i==cut:
            need(h.input_snapshot(clone)==second_preview['input_state'],'Actual retained q/v/a reset or history seam changed after DART')
        need(event.get('actual_control_owner')=='TEACHER_L2' and setter.get('execution_role')=='given_l2_motion' and
            setter.get('offgrid_stop') is False and setter.get('control_owner')=='TEACHER_L2',
            'DART/recovery trajectory contains STOP/ALIGN/nonteacher control')
        if raw.get('native_tick_in_cell')==0:
            value=setter['target_q']
            if i<cut:need(np.array_equal(value,target),'Actual perturbation setter target differs from seeded DART')
            else:
                while route_index<len(route) and not np.array_equal(route[route_index],value):route_index+=1
                need(route_index<len(route),'Actual recovery setter is not an ordered measured-plan waypoint')
            clone.set_target(value)
        command=clone.step()
        need(teacher._bits(command)==teacher._bits(event.get('q_command')),'Actual DART retained-q/v/a independent command replay differs')
        values=recovery.command_derivatives(history,command)
        need(values['velocity']<=.175 and values['acceleration']<=.5+1e-8 and values['jerk']<=1.5+1e-8,
            'Actual DART derivative/history seam exceeds original limits')
        history=np.vstack([history[1:],command])
    commands=np.asarray([r['q_command'] for r in audit[start:cut]],dtype=np.float64)
    need(first_preview['steps']==len(commands) and first_preview.get('commands_sha256')==hashlib.sha256(commands.tobytes()).hexdigest(),
        'Perturbation preview differs from actually executed independent 240Hz commands')
    return dict(preperturb_physics_row=start,postperturb_physics_row=cut,actual_perturbation_steps=cut-start,
        measured_h2_capture_bound=True,retained_command_qva_replayed=True,source_bound_dq_with_native_maximum_witness=True,
        actual_recovery_setter_end_row=end,trigger_skin_target_distance_m=distance,
        trigger_geometry=dict(geometry_id=metadata['geometry_id'],geometry_source_sha256=metadata['geometry_source_sha256'],
            contact_frame=rt.geometry.__name__+'.contact_frame',contact_frame_source=rt.reference(rt.geometry.__file__)))


def read_episode(attempt,base,manifest):
    """One complete episode audit; only full-bank admission can permit loading."""
    _,sources_check,decode=_profile(manifest)
    row=attempt['episode'];request=attempt['request']
    need(request in planned_requests() and all(row.get(k)==request[k] for k in ('split','subject_id','pose_id','target_id')),
        'Only exact planned TRAIN DART episode may open payloads')
    config,binding=_config(attempt,base);context=data._metadata(row,base)
    run,actual_config,freeze=data._native_evidence(row,base,context[2],physical=False)
    need(config==actual_config,'Bound original config changed')
    teacher_context=_context(row,base,manifest,freeze)
    sources_check(run,context[2],config,teacher_context)
    child,_=data._bound(row['child_result'],base);need(child.get('physical_pass') is True,'Actual child full physical pass required')
    ref=run.get('recovery_collection');sidecar=rt.read_bound(ref)
    roles.require_sidecar_location(ref,data._ref_path(row['run_result'],base).parent)
    need(sidecar.get('config_sha256')==rt.digest(config) and sidecar.get('declaration')==binding['settings'],
        'Actual DART sidecar/config/declaration binding differs')
    samples,receipt,meta=decode(row,base,context,run,config,freeze,teacher_context=teacher_context)
    outer=rt.shared._json(context[0]/'receipt.json')
    audit=data.read_native_capture_audit(context[0],outer,transport=row['capture_audit'])
    records=[{k:v for k,v in r.items() if k in ('observed_at_s','state17','measured_q')}
        for r in data.iter_training_records(context[0]/'compact')]
    trace=_validate_dart_trace(sidecar,audit,records,meta,binding['reference'],binding['settings'])
    clock=_cumulative_l2(run,audit)
    selected=roles.filter_recovery_samples(samples,audit,meta,sidecar);validate_samples(selected)
    need(len(selected['states'])>0,'Zero clean actual recovery teacher windows cannot be a positive episode')
    receipt.update(windows=len(selected['states']),recovery_collection=ref,actual_dart_trace=trace,
        cumulative_l2=clock,capture_audit=deepcopy(row['capture_audit']),evidence_profile=manifest['evidence_profile'],
        recorded_source=deepcopy(manifest['recorded_source']),physical_verifier=rt.reference(teacher.__file__),
        producer_sources=deepcopy(freeze),label_protocol=rt.shared.MASKED_LABEL_PROTOCOL,
        profile=manifest['profile'],qualification_scope='single_episode_only',bank_optimization_eligible=False)
    return selected,receipt,meta


def _read_negative(attempt,base,manifest):
    """A completed source/owner-bound physical negative contributes no labels."""
    _,sources_check,_=_profile(manifest)
    row=attempt['episode'];config,binding=_config(attempt,base);meta=binding['metadata']
    run,actual,freeze=data._native_evidence(row,base,meta,physical=False)
    need(config==actual,'Negative actual config changed')
    sources_check(run,meta,config,_context(row,base,manifest,freeze),positive=False)
    child,_=data._bound(row['child_result'],base)
    diagnostic=bool(run.get('phase_failure_diagnostics') or run.get('collision_failure_diagnostics') or
        any(p.get('success') is False for p in run.get('points',[])) or
        any(isinstance(h.get('failure_reason'),str) and h['failure_reason'] for h in run.get('l2_handoffs',[])))
    need(child.get('physical_pass') is False and run.get('success') is False and
        run.get('physical_validation') is False and diagnostic,'Original completed physical-negative diagnostics required')
    if attempt['outcome']=='preview_rejected':
        ref=run.get('recovery_collection');roles.require_sidecar_location(ref,data._ref_path(row['run_result'],base).parent)
        sidecar=rt.read_bound(ref)
        need(sidecar.get('schema')==recovery.SIDECAR_SCHEMA and sidecar.get('completed') is False and
            sidecar.get('config_sha256')==rt.digest(config) and sidecar.get('declaration')==binding['settings'] and
            any(q.get('kind')=='abstain' and isinstance(q.get('reason'),str) and q['reason'] for q in sidecar.get('queries',[])),
            'Original measured expert preview-abstention evidence required')
    return dict(run_result=row['run_result'],child_result=row['child_result'],owner_terminal=row['owner_terminal'],
        source_freeze=row['source_freeze'],physical_negative=True,classification=attempt['outcome'])


def code_binding():
    return dict(bank=rt.reference(__file__),capture_decoder=rt.reference(data.__file__),
        role_filter=rt.reference(roles.__file__),teacher_reader=rt.reference(teacher.__file__),
        sample_validator=rt.reference(measured.__file__),collector=rt.reference(recovery.__file__))


def _horizons(arrays):
    result={}
    for h in range(20):
        values=np.concatenate([a['actions'][~a['action_is_pad'][:,h],h].reshape(-1) for a in arrays]) if arrays else np.array([])
        result[str(h+1)]=dict(valid_cells=int(len(values)//7),mean_absolute_action_rad=float(np.mean(abs(values))) if len(values) else None)
    return result


def export_dataset(manifest_ref,output):
    manifest=rt.read_bound(manifest_ref);base=Path(manifest_ref['path']).parent
    attempts,errors,precollection=_headers(manifest,base)
    output=Path(output).resolve();need(not output.exists(),'Fresh separate DART pilot output directory required');output.mkdir(parents=True)
    ledger=[];sources=[];arrays=[];windows=[];offset=0;resets=set()
    for attempt in attempts:
        row=attempt['episode'];slot=attempt['request']['slot_id'];status=attempt['outcome']
        entry=dict(request=deepcopy(attempt['request']),episode=deepcopy(row),nominal_source_config=deepcopy(attempt['nominal_source_config']),
            status=status,reason=attempt['reason'],evidence=deepcopy(attempt['evidence']),windows=0,
            infrastructure_contamination=status=='infrastructure_error')
        try:
            need(slot not in errors,errors.get(slot,''))
            if status=='collected':
                sample,receipt,meta=read_episode(attempt,base,manifest);validate_samples(sample)
                reset=_reset(receipt);need(reset not in resets,'Duplicate native reset across independent episodes');resets.add(reset)
                count=len(sample['states']);need(count>0,'Zero-window physical positive cannot qualify')
                source=dict(slot_id=slot,declaration=deepcopy(row),metadata=deepcopy(meta),receipt=deepcopy(receipt),
                    windows=count,split_sample_range=[offset,offset+count])
                sources.append(source);arrays.append(sample)
                windows.extend(dict(slot_id=slot,sample_index=offset+i,window=deepcopy(w))
                    for i,w in enumerate(sample['derived_index']['windows']))
                offset+=count;entry.update(status='qualified',windows=count,receipt=receipt,native_reset_identity=receipt['native_reset_identity'])
            elif status in ('physical_negative','preview_rejected'):entry['negative_receipt']=_read_negative(attempt,base,manifest)
        except PhysicalNegative as error:
            entry.update(status='physical_negative',infrastructure_contamination=False,error=str(error),
                negative_receipt=deepcopy(getattr(error,'evidence',{})))
        except (ValueError,KeyError,TypeError,OSError,RuntimeError) as error:
            entry.update(status='infrastructure_error',infrastructure_contamination=True,error=type(error).__name__+': '+str(error)[:2000])
        ledger.append(entry)
    packed={key:np.concatenate([a[key] for a in arrays]) for key in ARRAY_KEYS} if arrays else {}
    array_ref=None
    if packed:
        path=output/'TRAIN_MEASURED_DART.npz'
        with path.open('xb') as stream:np.savez_compressed(stream,**packed)
        array_ref=rt.reference(path)
    index=rt.write_new(output/'INDEX.json',dict(samples=windows,order_sha256=rt.digest(windows),label_protocol=rt.shared.MASKED_LABEL_PROTOCOL))
    gate=admission(ledger,precollection=precollection)
    meta=dict(schema=SCHEMA,profile=manifest['profile'],evidence_profile=manifest['evidence_profile'],recorded_source=manifest['recorded_source'],
        source_manifest=manifest_ref,precollection=manifest['precollection'],precollection_plan=precollection,
        base_dataset=BASE29,base_counts=BASE_COUNTS,normalization=NORMALIZATION,normalization_refitted=False,
        attempt_ledger=ledger,fixed_denominator=20,sources=sources,windows=offset,arrays=array_ref,
        array_proofs={k:measured.array_proof(v) for k,v in packed.items()},index=index,optimization_admission=gate,data_code=code_binding(),
        horizon_diagnostics=_horizons(arrays),label_protocol=rt.shared.MASKED_LABEL_PROTOCOL,training_started=False,model_qualified=False,
        validation_payloads_opened=False,fresh_payloads_opened=False,base_arrays_modified=False)
    metadata_ref=rt.write_new(output/'METADATA.json',meta)
    return rt.write_new(output/'COMPLETE.json',dict(schema=SCHEMA,complete=True,profile=manifest['profile'],metadata=metadata_ref,
        optimization_admission=gate,arrays=array_ref,normalization=NORMALIZATION,windows=offset))


def metadata(complete_ref):
    complete=rt.read_bound(complete_ref)
    need(complete.get('schema')==SCHEMA and complete.get('complete') is True and complete.get('profile') in (PROFILE,PROFILE_SOURCE37),
        'Explicit complete DART pilot bank required')
    meta=rt.read_bound(complete['metadata']);manifest=rt.read_bound(meta['source_manifest'])
    attempts,errors,precollection=_headers(manifest,Path(meta['source_manifest']['path']).parent)
    need(complete.get('profile')==meta.get('profile')==manifest['profile'] and
        all(meta.get(k)==manifest[k] for k in ('profile','evidence_profile','recorded_source','base_dataset','normalization','precollection')) and
        meta.get('precollection_plan')==precollection and
        meta.get('schema')==SCHEMA and meta.get('normalization')==complete.get('normalization')==NORMALIZATION and
        meta.get('base_counts')==BASE_COUNTS and meta.get('normalization_refitted') is False and meta.get('fixed_denominator')==20 and
        meta.get('data_code')==code_binding(),'DART metadata/source/code/unchanged Base29 binding differs')
    ledger=meta.get('attempt_ledger');gate=admission(ledger,precollection=precollection)
    need(complete.get('optimization_admission')==meta.get('optimization_admission')==gate and
        complete.get('arrays')==meta.get('arrays') and complete.get('windows')==meta.get('windows'),
        'DART complete admission/arrays/counts differ')
    need(isinstance(ledger,list) and len(ledger)==20,'All twenty terminal outcomes required')
    for entry,attempt in zip(ledger,attempts):
        need(all(entry.get(k)==attempt[k] for k in ('request','episode','nominal_source_config','reason','evidence')) and
            (entry.get('status') in ('qualified','physical_negative','infrastructure_error') if attempt['outcome']=='collected' else
             entry.get('status') in (attempt['outcome'],'infrastructure_error')) and
            (entry['request']['slot_id'] not in errors or entry['status']=='infrastructure_error'),
            'Original canonical terminal outcome/source declaration changed')
    sources=meta.get('sources');need(isinstance(sources,list),'Ordered actual DART sources required');cursor=0;expected=[]
    for entry in ledger:
        if entry['status']=='qualified':expected.append(entry)
    need([s.get('slot_id') for s in sources]==[e['request']['slot_id'] for e in expected],
        'Ordered qualified sources omitted, duplicated or reordered')
    index=rt.read_bound(meta['index']);samples=index.get('samples')
    need(isinstance(samples,list) and index.get('order_sha256')==rt.digest(samples) and
        [w.get('sample_index') for w in samples]==list(range(meta['windows'])),'Actual DART ordered sample index differs')
    for source,entry in zip(sources,expected):
        n=entry['windows'];m=source.get('metadata',{});request=entry['request']
        need(source.get('declaration')==entry['episode'] and source.get('receipt')==entry.get('receipt') and
            source.get('windows')==n and source.get('split_sample_range')==[cursor,cursor+n] and
            source['receipt'].get('native_reset_identity')==entry.get('native_reset_identity') and
            all(m.get(k)==request[k] for k in ('split','subject_id','target_id')) and
            m.get('episode_id')==entry['episode']['episode_id'] and
            source['receipt'].get('evidence_profile')==manifest['evidence_profile'] and
            source['receipt'].get('recorded_source')==manifest['recorded_source'] and
            all(w.get('slot_id')==source['slot_id'] for w in samples[cursor:cursor+n]),
            'Actual DART source identity/receipt/mask/sample range differs')
        _reset(source['receipt']);cursor+=n
    need(cursor==meta['windows'] and len(samples)==cursor and set(meta.get('array_proofs',{}))==(set(ARRAY_KEYS) if cursor else set()),
        'Complete DART array proofs/counts required')
    return complete,meta


def load_arrays(complete_ref):
    complete,meta=metadata(complete_ref)
    need(complete['optimization_admission']['pilot_optimization_eligible'] is True,
        'Only eligible independently qualified frozen-prepared-denominator pilot arrays may load')
    with np.load(rt.bound_path(complete['arrays']),allow_pickle=False) as archive:
        need(set(archive.files)==set(ARRAY_KEYS),'Exactly seven existing masked arrays required')
        arrays={k:archive[k].copy() for k in ARRAY_KEYS}
    need({k:measured.array_proof(v) for k,v in arrays.items()}==meta['array_proofs'],'DART ordered array/mask bytes differ')
    index=rt.read_bound(meta['index'])
    arrays.update(schema=given.MASKED_DATASET_SCHEMA,task_id=rt.pose.TASK_ID,action_semantics_id=given.ACTION_SEMANTICS_ID,
        observation_contract=rt.pose.observation_contract(),label_protocol=rt.shared.MASKED_LABEL_PROTOCOL,
        derived_index=dict(windows=[w['window'] for w in index['samples']]),normalization_reference=NORMALIZATION,
        bank_schema=SCHEMA,optimization_admission=deepcopy(complete['optimization_admission']))
    validate_samples(arrays)
    need(len(arrays['states'])==meta['windows'] and _horizons([arrays])==meta.get('horizon_diagnostics'),
        'DART supported horizon/count diagnostics differ')
    return arrays,meta
