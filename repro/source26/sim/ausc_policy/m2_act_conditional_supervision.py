"""Finite, frozen, static ideal-state continuation producer; no measured API.

The original full checker is constructed once per case. No approximate cache,
resampling, clipping, live model, simulator, or fresh-cohort payload is used.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import numpy as np

from . import m2_act_conditional_data as data, m2_act_planned_supervision as planned
from . import m2_given_contact_recovery as recovery

rt=data.rt
old=data.old
need=rt.need
REQUEST_SCHEMA='m2_act_conditional_frozen_request_v3'
GENERATION_SCHEMA='m2_act_conditional_generation_v3'
CANDIDATE_SCHEMA='m2_act_conditional_candidate_v3'
ANCHORS=('near60mm','near40mm','near30mm','final_approach')
POLICIES=('mask_absent_canonical_suffix','computed_constant_coarse_target_hold')
SAMPLING='PCG64 SeedSequence([seed,slot_index]); unit Gaussian orthogonal to local tangent; no clipping/resampling'
TIE_RULE='nearest distance then lowest original input_index among unused APPROACH windows; final latest unused APPROACH'


def anchor_indices(histories,windows):
    h=np.asarray(histories);need(h.shape==(len(windows),2,17) and np.isfinite(h).all(),'Finite aligned ideal H2 required')
    available=[i for i,w in enumerate(windows) if w['phase']=='APPROACH']
    need(len(available)>=4,'Four distinct canonical APPROACH windows required')
    distances=np.linalg.norm(h[:,1,7:10],axis=1);selected=[]
    for target in (.060,.040,.030):
        i=min((i for i in available if i not in selected),key=lambda i:(abs(distances[i]-target),windows[i]['input_index']))
        selected.append(i)
    selected.append(max((i for i in available if i not in selected),key=lambda i:windows[i]['input_index']))
    return selected


def perturb_seed(q,tangent,*,seed,slot_index,scale_rad):
    q=rt.shared.contract._array(q,(7,),'canonical ideal q');t=rt.shared.contract._array(tangent,(7,),'local teacher tangent')
    need(type(seed) is int and 0<=seed<2**32 and type(slot_index) is int and slot_index>=0 and
        type(scale_rad) in (float,int) and np.isfinite(scale_rad) and scale_rad>=0,'Explicit frozen finite seed/scale required')
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,slot_index])))
    delta=rng.standard_normal(7);norm=float(np.linalg.norm(t));rule='stationary_tangent_isotropic_fallback'
    if norm>1e-12:
        unit=t/norm;delta-=unit*np.dot(delta,unit);rule='orthogonal_local_teacher_tangent'
    length=float(np.linalg.norm(delta));need(length>1e-12,'Degenerate perturbation seed rejected; do not resample')
    return q+delta*(scale_rad/length),rule


def continuation(raw,start,policy):
    raw=np.asarray(raw);need(raw.ndim==2 and raw.shape[1:]==(7,) and np.isfinite(raw).all() and
        type(start) is int and 0<=start<len(raw) and policy in POLICIES,'Bound canonical raw L2 prefix and explicit suffix policy required')
    cells=raw[start:start+20].copy()
    if policy==POLICIES[1] and len(cells)<20:cells=np.vstack([cells,np.repeat(raw[-1:],20-len(cells),axis=0)])
    return cells,np.arange(20)>=len(cells)


def _code():
    return {name:rt.reference(Path(__file__).with_name(name)) for name in
        ('m2_act_conditional_supervision.py','m2_act_conditional_data.py','m2_act_planned_supervision.py',
         'm2_given_contact_recovery.py','m2_given_contact_handoff.py','m2_contract.py')}


def _request(scene_manifest_ref,parent,allowed,*,seed,scales_rad,suffix_policy):
    need(type(seed) is int and 0<=seed<2**32 and isinstance(scales_rad,list) and len(scales_rad)==4 and
        all(type(x) in (float,int) and np.isfinite(x) and x>=0 for x in scales_rad) and suffix_policy in POLICIES,
        'Root must explicitly freeze seed, all four scales and suffix policy before outcomes')
    need(scene_manifest_ref in parent['request_refs'],'Original canonical scene manifest must belong to unchanged V2 parent')
    manifest=rt.read_bound(scene_manifest_ref);slots=old.requested_slots([scene_manifest_ref],allowed)
    need(len(slots)==64 and manifest.get('requested_target_slots')==64,'All 64 canonical body/pose/target slots required')
    s=parent['samples'];cases=[]
    for case_index,slot in enumerate(slots):
        rows=[r for r in parent['slot_ledger'] if r['slot_id']==slot['slot_id']]
        need(len(rows)==1,'Every canonical case needs its unchanged parent outcome')
        outcome=rows[0];case={k:slot[k] for k in ('slot_id','subject_id','pose_id','target_id')}
        case.update(case_index=case_index,canonical_status=outcome['status'],anchors=[])
        if outcome['status']=='rejected':case['canonical_rejection']=outcome['rejection']
        else:
            matches=[(i,source) for i,source in enumerate(s['sources']) if source['source_id']==outcome['source_id']]
            need(len(matches)==1 and matches[0][1]['kind']=='planned_ideal_tracking','Bound original planned canonical source required')
            source_index,source=matches[0];ids=np.flatnonzero(s['source_indices']==source_index)
            histories=s['source_histories'][ids];selected=anchor_indices(histories,source['windows'])
            scene=next(scene for scene in manifest['scenes'] if scene['subject_id']==case['subject_id'] and scene['pose_id']==case['pose_id'])
            case.update(parent_source_id=source['source_id'],metadata=deepcopy(source['metadata']),scene=deepcopy(scene),
                plan_config=source['bindings']['geometry'],raw_schedule=source['bindings']['raw_schedule'])
            for anchor_index,i in enumerate(selected):
                q=histories[i,1,:7];before=histories[max(0,i-1),1,:7];after=histories[min(len(ids)-1,i+1),1,:7]
                tangent=after-before;seeds=[]
                for perturbation,scale in enumerate(scales_rad):
                    index=case_index*16+anchor_index*4+perturbation
                    seed_slot=dict(slot_index=index,perturbation=perturbation,scale_rad=scale)
                    try:
                        generated,rule=perturb_seed(q,tangent,seed=seed,slot_index=index,scale_rad=scale)
                        seed_slot.update(q=generated.tolist(),rule=rule)
                    except ValueError as error:seed_slot['seed_rejection']='ValueError: '+str(error)
                    seeds.append(seed_slot)
                case['anchors'].append(dict(anchor=ANCHORS[anchor_index],input_index=source['windows'][i]['input_index'],
                    parent_row=int(ids[i]),q=q.tolist(),tangent=tangent.tolist(),seeds=seeds))
        cases.append(case)
    need(sum(c['canonical_status']=='rejected' for c in cases)==4,'Retain all four unavailable canonical cases')
    return dict(schema=REQUEST_SCHEMA,parent_complete=deepcopy(data.PARENT_COMPLETE),scene_manifest=deepcopy(scene_manifest_ref),
        seed=seed,scales_rad=deepcopy(scales_rad),perturbation_scale_semantics='joint_l2_radius_rad',
        suffix_policy=suffix_policy,sampling_rule=SAMPLING,anchor_tie_rule=TIE_RULE,
        anchors=list(ANCHORS),case_denominator=64,slot_denominator=1024,cases=cases,code=_code(),
        frozen_before_preview=True,selection_validation_used=False,fresh_payloads_opened=False,**old.FLAGS)


def freeze_request(output,scene_manifest_ref,*,allowed_train_ids,seed,scales_rad,suffix_policy):
    # Identity-only request expansion precedes any parent numerical access.
    old.requested_slots([scene_manifest_ref],allowed_train_ids)
    parent=data.load_parent(allowed_train_ids)
    return rt.write_new(output,_request(scene_manifest_ref,parent,allowed_train_ids,seed=seed,
        scales_rad=scales_rad,suffix_policy=suffix_policy))


def request_metadata(request_ref,allowed):
    request=rt.read_bound(request_ref);old._flags(request)
    need(request.get('schema')==REQUEST_SCHEMA and request.get('parent_complete')==data.PARENT_COMPLETE and
        request.get('case_denominator')==64 and request.get('slot_denominator')==1024 and len(request.get('cases',[]))==64 and
        request.get('frozen_before_preview') is True and request.get('selection_validation_used') is False and
        request.get('fresh_payloads_opened') is False and request.get('perturbation_scale_semantics')=='joint_l2_radius_rad',
        'Full frozen TRAIN-only 1024-slot conditional request with explicit joint L2 radius semantics required')
    for case in request['cases']:
        old._train(case.get('subject_id'),'train',allowed)
        if case.get('canonical_status')=='accepted':
            old._train(case.get('metadata',{}).get('subject_id'),case.get('metadata',{}).get('split'),allowed)
        else:need(case.get('canonical_status')=='rejected' and bool(case.get('canonical_rejection')),'Canonical rejection must remain explicit')
    need(request.get('code')==_code(),'Frozen conditional numerical/data/producer bytes changed')
    return request


def validate_request(request_ref,allowed):
    request=request_metadata(request_ref,allowed);parent=data.load_parent(allowed)
    want=_request(request['scene_manifest'],parent,allowed,seed=request['seed'],scales_rad=request['scales_rad'],
        suffix_policy=request['suffix_policy'])
    need(request==want,'Frozen conditional cases/anchors/seeds/policies differ from canonical parent')
    return request


def _slot(case,anchor_index,perturbation):
    index=case['case_index']*16+anchor_index*4+perturbation
    return dict(slot_id='conditional:'+str(index),slot_index=index,case_index=case['case_index'],
        subject_id=case['subject_id'],pose_id=case['pose_id'],target_id=case['target_id'],kind=data.CONDITIONAL_KIND,
        anchor=ANCHORS[anchor_index],perturbation=perturbation,status='rejected',**old.FLAGS)


def _static_arrays(q,metadata,cells,mask):
    reference=rt.pose.require_route(metadata)
    need(reference.snapshot()['available_at_s']==0.,'Generated known-target reference must be available at generated time zero')
    encoded=rt.shared.contract.encode_state(q,reference.pose_base,1.,metadata=metadata)
    relative=np.zeros((1,20,7));relative[0,:len(cells)]=cells-q
    return dict(source_histories=np.repeat(encoded[None,None],2,axis=1),source_times=np.array([[0.,.05]]),
        relative_setters=relative,setter_is_pad=mask[None],source_indices=np.zeros(1,dtype=np.int64))


def generate_case(request_ref,case_index,output,*,allowed_train_ids,production=True,trajectory_factory=None,verify_factory=None):
    """Generate exactly sixteen outcomes; every unsafe seed/cell rejects its slot.

    Fixture factories require explicit production=False and can never be loaded
    as production generation or computational optimization data.
    """
    injected=[k for k,v in (('trajectory_factory',trajectory_factory),('verify_factory',verify_factory)) if v is not None]
    need(not production or not injected,'Production producer rejects injected fixture factories')
    request=request_metadata(request_ref,allowed_train_ids)
    need(type(case_index) is int and 0<=case_index<64,'Declared canonical case index required')
    case=request['cases'][case_index];need(case['case_index']==case_index,'Canonical case order differs')
    slots=[_slot(case,a,p) for a in range(4) for p in range(4)];parts=[];output=Path(output);check_count=0
    need(not output.exists(),'Fresh per-case conditional artifact directory required');output.mkdir(parents=True)
    def finish():
        bound=rt.write_new(output/'OUTCOMES.json',dict(schema='m2_act_conditional_case_outcomes_v3',request_ref=request_ref,
            case_index=case_index,slot_ledger=slots,original_verification_calls=check_count,exact_command_cache=False,**old.FLAGS))
        return dict(parts=parts,slot_ledger=slots,outcomes=bound)
    if case['canonical_status']=='rejected':
        for slot in slots:slot['rejection']='canonical unavailable: '+case['canonical_rejection']
        return finish()
    try:
        reference,plan,config,_,geometry=planned._scene(case['scene'],case['target_id'],case['metadata'],
            case['plan_config'],allowed_train_ids,cpu_fixture=not production)
        original_verify=(verify_factory or planned._original_verifier)(config,plan,reference)  # exactly once per case
        def verify(q):
            nonlocal check_count
            check_count+=1
            return original_verify(q)
        raw=old.load_planned_schedule(case['raw_schedule'],allowed_train_ids)['absolute_raw_setters']
        if trajectory_factory is None:
            from sim.auscultation_pi05_v6.execution import JointTrajectory
            trajectory_factory=JointTrajectory
    except (ValueError,KeyError,TypeError,RuntimeError,TimeoutError) as error:
        for slot in slots:slot['rejection']=type(error).__name__+': '+str(error)
        return finish()
    for slot in slots:
        anchor=case['anchors'][ANCHORS.index(slot['anchor'])];seed=anchor['seeds'][slot['perturbation']]
        try:
            need('seed_rejection' not in seed,seed.get('seed_rejection',''))
            q=rt.shared.contract._array(seed['q'],(7,),'generated ideal seed');verify(q)
            trajectory=trajectory_factory(q,dt=rt.handoff.DT,**rt.handoff.PLANNING_LIMITS)
            snapshot=rt.handoff.input_snapshot(trajectory)
            need(snapshot['fields']['current_velocity']==[0.]*7 and snapshot['fields']['current_acceleration']==[0.]*7,
                'Static generated ideal input must have zero retained v/a')
            cells,mask=continuation(raw,anchor['input_index'],request['suffix_policy'])
            history=np.repeat(q[None],3,axis=0)
            preview=recovery.preview_cells(trajectory,history,cells,verify,trajectory_factory=trajectory_factory,
                deadline=time.monotonic()+recovery.MAX_PLAN_CPU_S)
            need(preview['cells']==cells.tolist() and preview['steps']==12*len(cells),'Every requested raw cell must be computed and checked')
            candidate=dict(schema=CANDIDATE_SCHEMA,slot_index=slot['slot_index'],q=q.tolist(),generated_command_history=history.tolist(),
                generated_history_times_s=[0.,.05],generated_velocity=[0.]*7,generated_acceleration=[0.]*7,
                synthetic_compression_m=0.,suffix_policy=request['suffix_policy'],canonical_prefix_cells=min(20,len(raw)-anchor['input_index']),
                computed_hold_cells=max(0,len(cells)-(len(raw)-anchor['input_index'])),preview=preview,
                verification_profile='injected_cpu_fixture_unqualified' if injected else 'original_full_cpu_preview',
                injected_factories=injected,logical_geometry_checks=1+preview['steps'],actual_geometry_checks=1+preview['steps'],
                exact_command_cache=False,**old.FLAGS)
            bound=rt.write_new(output/f"candidate-{slot['slot_index']}.json",candidate)
            source=dict(schema=data.SOURCE_SCHEMA,kind=data.CONDITIONAL_KIND,
                **{k:case[k] for k in ('subject_id','pose_id','target_id')},split='train',metadata=deepcopy(case['metadata']),
                observation_origin=data.ORIGIN,generated_static_input=True,future_measured_histories=False,synthetic_compression_m=0.,
                history_clock='generated H2 [0,.05]s; static q; no native ticks',anchor=slot['anchor'],slot_index=slot['slot_index'],
                windows=[dict(input_index=1,phase='APPROACH',anchor=slot['anchor'])],planning_limits=deepcopy(rt.handoff.PLANNING_LIMITS),
                verification_profile=candidate['verification_profile'],injected_factories=injected,geometry_artifacts=geometry,
                bindings=dict(source=case['scene']['plans'][case['target_id']]['plan'],geometry=case['plan_config'],
                    generator=rt.reference(__file__),data_protocol=rt.reference(data.__file__),
                    numerical_preview=rt.reference(recovery.__file__),retained_dynamics=rt.reference(rt.handoff.__file__),
                    raw_schedule=case['raw_schedule'],frozen_request=request_ref,candidate=bound),**old.FLAGS)
            part=data.bind_samples(_static_arrays(q,case['metadata'],cells,mask),[source],allowed_train_ids,production=production)
            parts.append(part);slot.update(status='accepted',source_id=part['sources'][0]['source_id'],candidate=bound)
        except (ValueError,KeyError,TypeError,RuntimeError,TimeoutError) as error:
            slot['rejection']=type(error).__name__+': '+str(error)
    need(rt.geometry.require_geometry_artifacts(config['collision_directory'],config)==geometry,'Geometry changed during conditional case')
    reference.verify_sources()
    return finish()


def validate_candidate(source,arrays,rows,allowed,*,production):
    request=request_metadata(source['bindings']['frozen_request'],allowed);index=source['slot_index']
    need(type(index) is int and 0<=index<1024 and len(rows)==1,'One declared conditional request slot per source required')
    case=request['cases'][index//16];a=(index%16)//4;p=index%4
    need(case['canonical_status']=='accepted' and source['anchor']==ANCHORS[a] and
        all(source[k]==case[k] for k in ('subject_id','pose_id','target_id','metadata')),'Conditional source/request identity/anchor differs')
    anchor=case['anchors'][a];seed=anchor['seeds'][p]
    q,rule=perturb_seed(anchor['q'],anchor['tangent'],seed=request['seed'],slot_index=index,scale_rad=request['scales_rad'][p])
    need(seed==dict(slot_index=index,perturbation=p,q=q.tolist(),rule=rule,scale_rad=request['scales_rad'][p]),
        'Conditional frozen seed changed or resampled')
    candidate=rt.read_bound(source['bindings']['candidate']);old._flags(candidate);preview=candidate.get('preview',{})
    keys={'schema','slot_index','q','generated_command_history','generated_history_times_s','generated_velocity','generated_acceleration',
        'synthetic_compression_m','suffix_policy','canonical_prefix_cells','computed_hold_cells','preview','verification_profile',
        'injected_factories','logical_geometry_checks','actual_geometry_checks','exact_command_cache'}|set(old.FLAGS)
    need(set(candidate)==keys,'Exact generated conditional candidate fields required; no measured claims')
    need(candidate.get('schema')==CANDIDATE_SCHEMA and candidate.get('slot_index')==index and candidate.get('q')==q.tolist() and
        candidate.get('generated_command_history')==np.repeat(q[None],3,axis=0).tolist() and
        candidate.get('generated_history_times_s')==[0.,.05] and candidate.get('generated_velocity')==[0.]*7 and
        candidate.get('generated_acceleration')==[0.]*7 and candidate.get('synthetic_compression_m')==0. and
        candidate.get('suffix_policy')==request['suffix_policy'],'Generated static provenance differs; never measured H2')
    if production:
        need(candidate.get('verification_profile')=='original_full_cpu_preview' and candidate.get('injected_factories')==[],
            'Production candidate fixture/profile rejected')
    need(source['verification_profile']==candidate['verification_profile'] and source['injected_factories']==candidate['injected_factories'],
        'Candidate/source full verification profile differs')
    need(source['bindings']['raw_schedule']==case['raw_schedule'] and source['bindings']['geometry']==case['plan_config'] and
        source['bindings']['source']==case['scene']['plans'][case['target_id']]['plan'],'Conditional canonical source bindings differ')
    raw=old.load_planned_schedule(case['raw_schedule'],allowed)['absolute_raw_setters']
    cells,mask=continuation(raw,anchor['input_index'],request['suffix_policy'])
    need(candidate.get('canonical_prefix_cells')==min(20,len(raw)-anchor['input_index']) and
        candidate.get('computed_hold_cells')==max(0,len(cells)-(len(raw)-anchor['input_index'])),
        'Computed hold extension must remain explicit; no recorded future claim')
    need(preview.get('all_240hz_commands_verified') is True and preview.get('physics_executed') is False and
        preview.get('cells')==cells.tolist() and preview.get('steps')==12*len(cells),
        'Every raw canonical/hold cell requires full original preview; failed cells cannot be masked')
    initial=preview.get('input_state',{});fields=initial.get('fields',{})
    need(preview.get('input_sha256')==rt.digest(initial) and fields.get('current_position')==q.tolist() and
        fields.get('current_velocity')==fields.get('current_acceleration')==[0.]*7 and initial.get('dt')==rt.handoff.DT and
        all(fields.get('max_'+key)==[value]*7 for key,value in rt.handoff.PLANNING_LIMITS.items()),
        'Generated original static command q/v/a/limits seam differs')
    commands=np.asarray(preview.get('commands'),dtype=np.float64)
    need(commands.shape==(12*len(cells),7) and np.isfinite(commands).all() and
        preview.get('commands_sha256')==hashlib.sha256(commands.tobytes()).hexdigest() and
        candidate.get('logical_geometry_checks')==candidate.get('actual_geometry_checks')==1+len(commands) and
        candidate.get('exact_command_cache') is False,'Full exact uncached command/check-count binding required')
    from sim.auscultation_skill_v1.executor import command_derivatives
    history=np.repeat(q[None],3,axis=0);peaks={k:0. for k in recovery.GIVEN_TRAJECTORY_LIMITS}
    for command in commands:
        values=command_derivatives(history,command)
        need(values['velocity']<=.175 and values['acceleration']<=.5+1e-8 and values['jerk']<=1.5+1e-8,
            'Original FD seam/limits failed; reject entire conditional candidate')
        for key in peaks:peaks[key]=max(peaks[key],values[key])
        history=np.vstack([history[1:],command])
    need(preview.get('command_derivative_maxima')==peaks,'Original per-command FD maxima differ')
    expected=_static_arrays(q,case['metadata'],cells,mask);row=int(rows[0])
    for key in old.ARRAY_KEYS[:-1]:need(np.array_equal(arrays[key][row:row+1],expected[key]),'Static generated H2/fixed-origin labels/mask differs: '+key)


def generate(request_ref,output,*,allowed_train_ids,case_indices=None,production=True,trajectory_factory=None,verify_factory=None):
    """Full 64-case generation or explicit finite qualification subset.

    Subsets retain their declared sixteen slots per case, are marked incomplete,
    and cannot be admitted by load_generation or a training preparation.
    """
    request=validate_request(request_ref,allowed_train_ids);indices=list(range(64)) if case_indices is None else list(case_indices)
    need(indices and len(indices)==len(set(indices)) and all(type(i) is int and 0<=i<64 for i in indices),'Finite unique declared case indices required')
    need(not production or (trajectory_factory is None and verify_factory is None),'Production fixture injection rejected')
    output=Path(output);need(not output.exists(),'Fresh finite conditional generation directory required');output.mkdir(parents=True)
    parts=[];ledger=[];outcomes=[];started=time.monotonic()
    for index in indices:
        got=generate_case(request_ref,index,output/f'case-{index}',allowed_train_ids=allowed_train_ids,production=production,
            trajectory_factory=trajectory_factory,verify_factory=verify_factory)
        parts.extend(got['parts']);ledger.extend(got['slot_ledger']);outcomes.append(got['outcomes'])
    if parts:samples=data.concatenate_samples(parts,allowed_train_ids,production=production)
    else:
        shapes=dict(source_histories=(0,2,17),source_times=(0,2),relative_setters=(0,20,7),setter_is_pad=(0,20),source_indices=(0,))
        arrays={k:np.empty(shape,dtype=bool if k=='setter_is_pad' else np.int64 if k=='source_indices' else float) for k,shape in shapes.items()}
        samples=data.bind_samples(arrays,[],allowed_train_ids,production=production)
    data._ledger(ledger,samples['sources'],allowed_train_ids)
    path=output/'CONDITIONAL_ARRAYS.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,**{k:samples[k] for k in old.ARRAY_KEYS})
    meta={k:v for k,v in samples.items() if k not in old.ARRAY_KEYS}
    meta.update(slot_ledger=ledger,request_ref=request_ref,case_indices=indices,case_outcomes=outcomes,counts=data.counts(samples,ledger),
        original_verification_calls=sum(rt.read_bound(ref)['original_verification_calls'] for ref in outcomes),
        production=production,elapsed_wall_s=time.monotonic()-started)
    bound=rt.write_new(output/'METADATA.json',meta)
    return rt.write_new(output/'COMPLETE.json',dict(schema=GENERATION_SCHEMA,complete=indices==list(range(64)) and production,
        metadata=bound,arrays=rt.reference(path),metadata_sha256=bound['sha256'],request_ref=request_ref,
        fixed_slot_denominator=1024,attempted_slots=len(ledger),**old.FLAGS))


def generation_metadata(ref,allowed):
    complete=rt.read_bound(ref);old._flags(complete)
    need(complete.get('schema')==GENERATION_SCHEMA and complete.get('complete') is True and
        complete.get('fixed_slot_denominator')==complete.get('attempted_slots')==1024,
        'Production complete 1024-slot generation required; qualification subsets/fixtures rejected')
    request=request_metadata(complete['request_ref'],allowed);meta=rt.read_bound(complete['metadata'])
    need(meta.get('production') is True and meta.get('case_indices')==list(range(64)) and
        meta.get('request_ref')==complete['request_ref'] and complete.get('metadata_sha256')==complete['metadata']['sha256'],
        'Full production request/case/metadata binding required')
    data._sources(meta['sources'],allowed);data._ledger(meta['slot_ledger'],meta['sources'],allowed)
    need(len(meta.get('case_outcomes',[]))==64,'All original per-case outcome bindings required')
    actual_checks=0
    for index,bound in enumerate(meta['case_outcomes']):
        outcome=rt.read_bound(bound);old._flags(outcome)
        need(outcome.get('schema')=='m2_act_conditional_case_outcomes_v3' and outcome.get('request_ref')==complete['request_ref'] and
            outcome.get('case_index')==index and outcome.get('slot_ledger')==meta['slot_ledger'][index*16:(index+1)*16] and
            type(outcome.get('original_verification_calls')) is int and outcome['original_verification_calls']>=0 and
            outcome.get('exact_command_cache') is False,'Every accepted/rejected per-case outcome and original check count must be retained')
        actual_checks+=outcome['original_verification_calls']
    need(meta.get('original_verification_calls')==actual_checks,'Total actual original verifier calls differ')
    expected=[_slot(case,a,p) for case in request['cases'] for a in range(4) for p in range(4)]
    need(len(meta['slot_ledger'])==1024,'Every conditional slot must survive')
    for got,want in zip(meta['slot_ledger'],expected):
        need(all(got.get(k)==v for k,v in want.items() if k!='status'),'Conditional requested slot order/identity differs')
        case=request['cases'][want['case_index']]
        if case['canonical_status']=='rejected':
            need(got['status']=='rejected' and got['rejection']=='canonical unavailable: '+case['canonical_rejection'],
                'Unavailable canonical case needs all sixteen unchanged rejected slots')
        elif got['status']=='accepted':
            source=next(s for s in meta['sources'] if s['source_id']==got['source_id'])
            need(source.get('slot_index')==got['slot_index'] and source['bindings']['candidate']==got.get('candidate') and
                source['bindings']['frozen_request']==complete['request_ref'],'Accepted conditional outcome binding differs')
    return complete,meta


def load_generation(ref,allowed):
    complete,meta=generation_metadata(ref,allowed);validate_request(complete['request_ref'],allowed)
    with np.load(rt.bound_path(complete['arrays']),allow_pickle=False) as archive:
        need(set(archive.files)==set(old.ARRAY_KEYS),'Exact separate conditional array keys required')
        samples={**{k:v for k,v in meta.items() if k not in ('slot_ledger','request_ref','case_indices','case_outcomes','counts','production','elapsed_wall_s','original_verification_calls')},
            **{k:archive[k].copy() for k in old.ARRAY_KEYS}}
    data.validate_samples(samples,meta['sources'],allowed)
    need(meta['counts']==data.counts(samples,meta['slot_ledger']),'Conditional computational denominator differs')
    return dict(samples=samples,sources=samples['sources'],slot_ledger=meta['slot_ledger'],request_ref=meta['request_ref'])
