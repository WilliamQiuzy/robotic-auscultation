"""Hash-bound, explicitly unqualified ACT diagnostic auxiliary supervision.

No native ARRAY_KEYS, native ticks, contacts, settling or successful episode
receipts are created. Normalization is supplied by the real TRAIN bank. The
frozen-oracle importer never replays a model or imports its numerical generator.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from . import m2_given_contact_runtime as rt

need=rt.need
SCHEMA='m2_act_auxiliary_preparation_v1'
SOURCE_SCHEMA='m2_act_auxiliary_source_v1'
PLANNED_SCHEDULE_SCHEMA='m2_act_planned_raw_setter_schedule_v1'
PLANNED_SCHEDULE_KEY='planned_absolute_raw_setters'
ARRAY_KEYS=('source_histories','source_times','relative_setters','setter_is_pad','source_indices')
KINDS=('planned_ideal_tracking','student_oracle_unexecuted')
ORIGINS=dict(planned_ideal_tracking='causal_generated_20hz_command_states',
    student_oracle_unexecuted='actual_authenticated_TRAIN_student_query')
FLAGS=dict(actions_executed=False,physics_qualified=False,native_demonstration_eligible=False,training_eligible=False)


def _train(subject,split,allowed):
    need(split=='train' and subject in allowed,'Eligible TRAIN-only auxiliary source required before numeric reads')
    rt.shared.require_membership('train',subject)


def _flags(value):
    need(all(value.get(k) is False for k in FLAGS),'Auxiliary physical/admission flags must remain false')


def make_source(kind,metadata,pose_id,bindings,windows):
    """Describe one body/pose/target source; bind_samples seals its numeric rows."""
    need(kind in KINDS,'Unknown auxiliary kind')
    return dict(schema=SOURCE_SCHEMA,kind=kind,subject_id=metadata['subject_id'],split=metadata['split'],
        pose_id=pose_id,target_id=metadata['target_id'],metadata=deepcopy(metadata),bindings=deepcopy(bindings),
        observation_origin=ORIGINS[kind],windows=deepcopy(windows),**FLAGS)


def installed_pose_id(metadata):
    """Exact observed pose from the bound installed patient, never a reset ID."""
    from .m2_rebuild_capture_dataset import POSE_IDS
    reference=rt.pose.require_route(metadata)
    patient=rt.pose.bound_json(reference.snapshot()['sources']['patient_receipt'])
    prefix=metadata['subject_id']+'_';proposal=patient.get('proposal')
    need(patient.get('identity')==metadata['subject_id'] and isinstance(proposal,str) and
         proposal.startswith(prefix) and proposal[len(prefix):] in POSE_IDS,
         'Bound installed patient must declare an exact known observed pose')
    return proposal[len(prefix):]


def _planned_schedule_header(schedule_ref,allowed):
    header=rt.read_bound(schedule_ref);meta=header.get('metadata',{})
    _train(header.get('subject_id'),header.get('split'),allowed)
    _train(meta.get('subject_id'),meta.get('split'),allowed)
    need(header.get('schema')==PLANNED_SCHEDULE_SCHEMA and header.get('kind')=='planned_ideal_tracking',
         'Separate planned raw setter schedule manifest required')
    _flags(header);rt.geometry.require_current(meta)
    reference=rt.pose.require_route(meta)
    need(all(header.get(k)==meta.get(k) for k in ('subject_id','split','target_id')) and
         header.get('pose_id')==installed_pose_id(meta) and header.get('reference_sha256')==reference.sha256 and
         header.get('source_binding')==rt.reference(reference.snapshot()['sources']['fresh_plan']['file']),
         'Planned schedule identity/pose/target/reference binding differs')
    initial=header.get('initial_command_state',{});fields=initial.get('fields',{})
    need(initial.get('dt')==rt.handoff.DT and set(fields)==set(rt.handoff.INPUT_FIELDS) and
         header.get('initial_command_state_sha256')==rt.digest(initial),'Planned schedule initial command state binding differs')
    for key in ('current_position','current_velocity','current_acceleration'):
        rt.shared.contract._array(fields[key],(7,),'planned schedule '+key)
    need(np.array_equal(initial.get('velocity'),fields['current_velocity']) and
         np.array_equal(initial.get('acceleration'),fields['current_acceleration']) and
         all(fields['max_'+key]==[value]*7 for key,value in rt.handoff.PLANNING_LIMITS.items()),
         'Planned schedule must retain original command q/v/a and planning limits')
    need(type(header.get('raw_setter_cells')) is int and 0<header['raw_setter_cells']<=2400 and
         isinstance(header.get('raw_setter_sha256'),str) and len(header['raw_setter_sha256'])==64,
         'Bounded complete raw setter schedule count/hash required')
    return header


def _planned_schedule_array(header):
    with np.load(rt.bound_path(header['arrays']),allow_pickle=False) as archive:
        need(archive.files==[PLANNED_SCHEDULE_KEY],'Separate complete planned raw setter array required')
        raw=np.asarray(archive[PLANNED_SCHEDULE_KEY]).copy()
    need(raw.shape==(header['raw_setter_cells'],7) and raw.dtype.kind in 'fiu' and np.isfinite(raw).all() and
         hashlib.sha256(np.ascontiguousarray(raw,dtype='<f8').tobytes()).hexdigest()==header['raw_setter_sha256'],
         'Complete planned raw setter schedule shape/finite/hash binding differs')
    return raw


def load_planned_schedule(schedule_ref,allowed_train_ids):
    """Read the complete raw 20Hz schedule after TRAIN metadata validation.

    This includes cells whose origins were omitted from causal H2 windows at a
    phase transition. The retained initial command q/v/a is returned verbatim.
    """
    header=_planned_schedule_header(schedule_ref,allowed_train_ids)
    return dict(absolute_raw_setters=_planned_schedule_array(header),
        initial_command_state=deepcopy(header['initial_command_state']),**FLAGS)


def write_planned_schedule(output,preview,metadata,pose_id,allowed_train_ids):
    """Persist an unexecuted preview's complete raw schedule in a fresh directory."""
    _train(metadata.get('subject_id'),metadata.get('split'),allowed_train_ids)
    reference=rt.pose.require_route(metadata)
    need(pose_id==installed_pose_id(metadata),'Planned schedule pose must match installed patient')
    need(preview.get('all_240hz_commands_verified') is True and preview.get('physics_executed') is False,
         'Unexecuted computational preview required for planned schedule')
    raw=np.asarray(preview.get('cells'),dtype='<f8')
    need(raw.ndim==2 and raw.shape[1:]==(7,) and 0<len(raw)<=2400 and np.isfinite(raw).all(),
         'Complete bounded finite raw setter schedule required')
    output=Path(output);need(not output.exists(),'Fresh planned schedule artifact directory required');output.mkdir(parents=True)
    path=output/'PLANNED_RAW_SETTERS.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,**{PLANNED_SCHEDULE_KEY:raw})
    header=dict(schema=PLANNED_SCHEDULE_SCHEMA,kind='planned_ideal_tracking',
        subject_id=metadata['subject_id'],split=metadata['split'],pose_id=pose_id,target_id=metadata['target_id'],
        metadata=deepcopy(metadata),reference_sha256=reference.sha256,
        source_binding=rt.reference(reference.snapshot()['sources']['fresh_plan']['file']),
        initial_command_state=deepcopy(preview['input_state']),initial_command_state_sha256=rt.digest(preview['input_state']),
        raw_setter_cells=len(raw),raw_setter_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        arrays=rt.reference(path),schedule_clock='generated_20hz_raw_setters; no native ticks',**FLAGS)
    bound=rt.write_new(output/'PLANNED_RAW_SETTERS.json',header)
    _planned_schedule_header(bound,allowed_train_ids)
    return bound


def _sources(sources,allowed,*,source_schema=SOURCE_SCHEMA):
    need(isinstance(sources,list) and isinstance(allowed,(list,tuple,set,frozenset)),'Explicit source table and TRAIN IDs required')
    # Entire table is gated before even a hash read of source/native references.
    for source in sources:
        need(isinstance(source,dict),'Explicit auxiliary source required')
        _train(source.get('subject_id'),source.get('split'),allowed)
        meta=source.get('metadata',{})
        _train(meta.get('subject_id'),meta.get('split'),allowed)
    # Gate every separate schedule manifest before any numeric archive is read.
    schedules={}
    for i,source in enumerate(sources):
        if source.get('kind')=='planned_ideal_tracking':
            bound=source.get('bindings',{}).get('raw_schedule')
            need(bound is not None,'Complete planned raw setter schedule reference required')
            schedules[i]=_planned_schedule_header(bound,allowed)
    for i,source in enumerate(sources):
        kind=source.get('kind');meta=source['metadata']
        need(source.get('schema')==source_schema and kind in KINDS and
             source.get('observation_origin')==ORIGINS[kind],'Separate auxiliary source schema/origin required')
        _flags(source);rt.geometry.require_current(meta)
        reference=rt.pose.require_route(meta)
        need(all(source.get(k)==meta.get(k) for k in ('subject_id','split','target_id')) and
             isinstance(source.get('pose_id'),str) and bool(source['pose_id']), 'Auxiliary identity/pose/target alignment differs')
        need(isinstance(source.get('windows'),list) and len(source['windows'])>0 and
             all(type(w.get('input_index')) is int and w['input_index']>=0 and w.get('phase') in ('FREE','APPROACH')
                 for w in source['windows']),'Explicit causal source windows required')
        indices=[w['input_index'] for w in source['windows']]
        need(len(indices)==len(set(indices)),'Duplicate source window input indices')
        bindings=source.get('bindings',{})
        need(isinstance(bindings,dict) and {'source','geometry','generator'}<=set(bindings),
             'Bound source/generator/geometry references required')
        for bound in bindings.values():rt.bound_path(bound)
        reference.verify_sources()
        patient=rt.pose.bound_json(reference.snapshot()['sources']['patient_receipt'])
        need(patient.get('identity')==source['subject_id'],'Installed patient identity differs from TRAIN source')
        need(source['pose_id']==installed_pose_id(meta),'Declared source pose differs from installed patient pose')
        if i in schedules:
            schedule=schedules[i];preview=source.get('preview',{})
            need(schedule['metadata']==meta and schedule['source_binding']==bindings['source'] and
                 schedule['initial_command_state']==preview.get('input_state') and
                 schedule['initial_command_state_sha256']==preview.get('input_sha256') and
                 all(schedule[k]==preview.get(k) for k in ('raw_setter_cells','raw_setter_sha256')),
                 'Source differs from complete planned raw setter schedule binding')
    return schedules


def _numeric(samples,sources,*,full_oracle_prefix=False):
    n=len(np.asarray(samples['source_indices']))
    shapes=dict(source_histories=(n,2,17),source_times=(n,2),relative_setters=(n,20,7),
        setter_is_pad=(n,20),source_indices=(n,))
    arrays={}
    for key,shape in shapes.items():
        value=np.asarray(samples[key]);need(value.shape==shape,'Auxiliary '+key+' shape differs')
        if key=='setter_is_pad':need(value.dtype==np.dtype(bool),'Boolean auxiliary mask required')
        elif key=='source_indices':need(value.dtype.kind in 'iu','Integer auxiliary source indices required')
        else:need(value.dtype.kind in 'fiu' and np.isfinite(value).all(),'Finite auxiliary '+key+' required')
        arrays[key]=value
    mask=arrays['setter_is_pad'];times=arrays['source_times'];ids=arrays['source_indices']
    need((n==0 and not sources) or (n>0 and sources and np.all(ids>=0) and np.all(ids<len(sources))),
         'Auxiliary source table/index alignment differs')
    need(np.all(~mask[:,0]) and np.all(np.diff(mask.astype(int),axis=1)>=0) and
         np.all(arrays['relative_setters'][mask]==0),'Nonempty contiguous supervision and zero padding required')
    need(np.all(times>=0) and np.all(abs(times[:,1]-times[:,0]-.05)<=1e-8), 'Causal H2 20Hz clocks required')
    for i,source in enumerate(sources):
        rows=np.flatnonzero(ids==i)
        need(len(rows)==len(source['windows']),'Source table/window count differs')
        inputs=np.asarray([w['input_index'] for w in source['windows']])
        expected=(inputs+1)/20. if source['kind']=='planned_ideal_tracking' else inputs/20.
        need(np.all(abs(times[rows,1]-expected)<=1e-8),'Source window/current clock binding differs')
        if source['kind']=='student_oracle_unexecuted' and not full_oracle_prefix:
            need(np.all(mask[rows,1:]),'Unobserved student future must mask all but first raw setter')
        need(np.all(np.diff(times[rows,1])>0),'Source clocks must increase in window order')
    return arrays


def _row_digest(arrays,rows):
    digest=hashlib.sha256()
    for key in ARRAY_KEYS[:-1]:
        value=np.ascontiguousarray(arrays[key][rows])
        # Normalize numeric storage so float32/int encodings cannot alter meaning silently.
        value=value.astype(np.bool_ if key=='setter_is_pad' else '<f8')
        digest.update(key.encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def _source_id(source):
    return rt.digest({k:v for k,v in source.items() if k!='source_id'})


def bind_samples(arrays,sources,allowed_train_ids):
    """Seal new producer output, without admitting it as native demonstration data."""
    sources=deepcopy(sources);_sources(sources,allowed_train_ids)
    arrays=_numeric(arrays,sources)
    for i,source in enumerate(sources):
        source['sample_sha256']=_row_digest(arrays,np.flatnonzero(arrays['source_indices']==i))
        source['source_id']=_source_id(source)
    result=dict(schema=SCHEMA,sources=sources,source_table_sha256=rt.digest(sources),
        allowed_train_ids=sorted(allowed_train_ids),**FLAGS,**{k:v.copy() for k,v in arrays.items()})
    validate_samples(result,sources,allowed_train_ids)
    return result


def validate_samples(samples,sources,allowed_train_ids):
    """Validate all source eligibility/bindings before accessing numerical arrays."""
    schedules=_sources(sources,allowed_train_ids)
    need(samples.get('schema')==SCHEMA,'Separate auxiliary preparation schema required');_flags(samples)
    need(samples.get('source_table_sha256')==rt.digest(sources) and
         samples.get('sources')==sources,'Auxiliary source table binding differs')
    arrays=_numeric(samples,sources)
    for i,source in enumerate(sources):
        rows=np.flatnonzero(arrays['source_indices']==i)
        need(source.get('source_id')==_source_id(source) and source.get('sample_sha256')==
            _row_digest(arrays,rows), 'Source numeric/window binding changed')
        if i in schedules:
            raw=_planned_schedule_array(schedules[i])
            for row,window in zip(rows,source['windows']):
                start=window['input_index'];count=int(np.count_nonzero(~arrays['setter_is_pad'][row]))
                need(start+count<=len(raw) and np.array_equal(arrays['relative_setters'][row,:count],
                     raw[start:start+count]-arrays['source_histories'][row,-1,:7]),
                     'Supervised labels differ from the complete planned raw setter schedule')
    return samples


def pack_batch(samples,real_native_stats,indices):
    """Independent no-noise auxiliary packing with real TRAIN normalization only."""
    validate_samples(samples,samples['sources'],samples['allowed_train_ids'])
    stats=rt.validate_normalization(real_native_stats)
    ids=np.asarray(indices);need(ids.ndim==1 and ids.dtype.kind in 'iu' and
        np.all(ids>=0) and np.all(ids<len(samples['source_indices'])),'Valid auxiliary batch indices required')
    h=(samples['source_histories'][ids]-np.asarray(stats['state_mean']))/np.asarray(stats['state_std'])
    action=(samples['relative_setters'][ids]-np.asarray(stats['action_mean']))/np.asarray(stats['action_std'])
    return {'observation.state':h[:,1,:7].astype(np.float32),
        'observation.environment_state':np.concatenate([h[:,1,7:],h[:,0]],axis=1).astype(np.float32),
        'action':action.astype(np.float32),'action_is_pad':samples['setter_is_pad'][ids].copy()}


def sample_indices(rng,samples,sources,count):
    """Uniform available body, then pose, then target; windows are sampled last."""
    validate_samples(samples,sources,samples['allowed_train_ids'])
    need(type(count) is int and count>0 and len(sources)>0,'Positive auxiliary sample count and available sources required')
    tree={}
    for i,s in enumerate(sources):tree.setdefault(s['subject_id'],{}).setdefault(s['pose_id'],{}).setdefault(s['target_id'],[]).append(i)
    def choose(values):return values[int(rng.integers(len(values)))]
    selected=[]
    for _ in range(count):
        poses=tree[choose(sorted(tree))];targets=poses[choose(sorted(poses))]
        group=targets[choose(sorted(targets))]
        rows=np.flatnonzero(np.isin(samples['source_indices'],group));selected.append(int(choose(rows)))
    return np.asarray(selected,dtype=np.int64)


def _ledger(ledger,sources,allowed):
    need(isinstance(ledger,list),'All attempted auxiliary slots must be retained')
    seen=set();accepted=[]
    for row in ledger:
        _train(row.get('subject_id'),'train',allowed)
        need(isinstance(row.get('slot_id'),str) and row['slot_id'] not in seen and
            row.get('kind') in KINDS and row.get('status') in ('accepted','rejected'), 'Unique explicit attempted slot required')
        seen.add(row['slot_id'])
        if row['status']=='accepted':
            matches=[s for s in sources if s['source_id']==row.get('source_id')]
            need(len(matches)==1 and all(matches[0][k]==row.get(k) for k in ('subject_id','pose_id','target_id','kind')),
                'Accepted slot differs from bound auxiliary source')
            accepted.append(row['source_id'])
        else:need(isinstance(row.get('rejection'),str) and bool(row['rejection']),'Retained rejection reason required')
    need(sorted(accepted)==sorted(s['source_id'] for s in sources),'Every available source needs exactly one retained accepted slot')


def _counts(samples,ledger):
    return {kind:dict(attempted_slots=sum(r['kind']==kind for r in ledger),
        accepted_sources=sum(s['kind']==kind for s in samples['sources']),
        samples=sum(len(s['windows']) for s in samples['sources'] if s['kind']==kind),
        supervised_cells=int(sum(np.count_nonzero(~samples['setter_is_pad'][samples['source_indices']==i])
            for i,s in enumerate(samples['sources']) if s['kind']==kind))) for kind in KINDS}


def requested_slots(request_refs,allowed_train_ids):
    """Expand bound scene manifests and frozen oracle requests without numerics."""
    need(isinstance(request_refs,(list,tuple)) and request_refs,'Bound original auxiliary slot requests required')
    slots=[];pending=[]
    for bound in request_refs:
        request=rt.read_bound(bound)
        if request.get('schema')=='m2_rebuild_scene_manifest_v1':
            scenes=request.get('scenes',[]);targets=request.get('targets',[])
            need(request.get('purpose')=='capture' and scenes and targets and len(targets)==len(set(targets)) and
                 request.get('requested_scenes')==len(scenes) and
                 request.get('requested_target_slots')==len(scenes)*len(targets) and
                 set(request.get('subject_ids',[]))=={s.get('subject_id') for s in scenes},
                 'Original TRAIN scene manifest slot denominator required')
            for scene in scenes:
                _train(scene.get('subject_id'),scene.get('split'),allowed_train_ids)
                need(scene.get('cohort')=='train' and set(scene.get('plans',{}))==set(targets),
                     'Every requested TRAIN body/pose/target slot must be retained')
                for target in targets:
                    pending.append(dict(slot_id='planned:'+scene['subject_id']+':'+scene['pose_id']+':'+target,
                        subject_id=scene['subject_id'],pose_id=scene['pose_id'],target_id=target,
                        kind='planned_ideal_tracking',source_binding=scene['plans'][target].get('plan')))
        else:
            cases=request.get('cases',[]);_old_flags(request)
            need(cases and request.get('fixed_query_denominator')==sum(len(c.get('physics_rows',[])) for c in cases),
                 'Original frozen oracle slot denominator required')
            for case in cases:
                config=rt.read_bound(case['source']['config'])
                meta=config.get('m2_given_contact_expanded_pilot_deployment',{}).get('metadata',{})
                _train(meta.get('subject_id'),meta.get('split'),allowed_train_ids)
                pending.append((case,meta,bound))
    # Entire requested cohort is TRAIN-gated before installed patient payloads.
    for entry in pending:
        if isinstance(entry,dict):
            slots.append(entry)
            continue
        case,meta,frozen_request=entry
        pose=installed_pose_id(meta)
        for row in case['physics_rows']:
            need(type(row) is int and row>=12 and row%12==0,'Original frozen query grid required')
            slots.append(dict(slot_id='oracle:'+case['job']+':'+str(row),subject_id=meta['subject_id'],
                pose_id=pose,target_id=meta['target_id'],kind='student_oracle_unexecuted',source_binding=case['source']['config'],
                physics_row=row,frozen_request=frozen_request,frozen_case=deepcopy(case)))
    need(len({s['slot_id'] for s in slots})==len(slots),'Duplicate requested auxiliary slots')
    return slots


def _requested_ledger(ledger,sources,request_refs,allowed):
    expected=requested_slots(request_refs,allowed)
    need([r['slot_id'] for r in ledger]==[r['slot_id'] for r in expected],
         'Every requested auxiliary slot must survive in original order')
    for row,want in zip(ledger,expected):
        need(all(row.get(k)==want[k] for k in ('subject_id','pose_id','target_id','kind')),
             'Requested slot identity/pose/target differs')
        if row['status']=='accepted':
            source=next(s for s in sources if s['source_id']==row['source_id'])
            need(want['source_binding'] is not None and source['bindings']['source']==want['source_binding'],
                 'Accepted source differs from the requested source binding')
            if want['kind']=='student_oracle_unexecuted':
                bindings=source['bindings'];case=want['frozen_case']
                need(row.get('physics_row')==want['physics_row']==source.get('frozen_physics_row') and
                     bindings.get('frozen_request')==want['frozen_request'],
                     'Accepted oracle query differs from the exact frozen request/physics row')
                need(row.get('outcome')==bindings.get('outcome'),
                     'Accepted oracle query differs from its frozen outcome reference')
                outcome=rt.read_bound(row['outcome']);request=rt.read_bound(bindings['request'])
                need(outcome.get('physics_row')==want['physics_row'] and outcome.get('proposal_accepted') is True and
                     outcome.get('computational_proposal')==bindings.get('proposal'),
                     'Accepted oracle proposal differs from the exact frozen query outcome')
                need(request.get('source')==case['source'] and request.get('plan_config')==case['plan_config'] and
                     request.get('physics_rows')==case['physics_rows'],
                     'Accepted oracle query differs from the saved frozen REQUEST')
                proposal=rt.read_bound(bindings['proposal'])
                need(proposal.get('query',{}).get('physics_row')==want['physics_row'] and
                     proposal.get('query',{}).get('metadata')==source['metadata'],
                     'Accepted oracle proposal belongs to a different frozen query')


def export_preparation(output,samples,slot_ledger,allowed_train_ids,*,request_refs):
    """Write immutable auxiliary bytes; requests enforce all attempted slots.

    request_refs lists original scene_manifest.json and/or FROZEN_REQUEST.json
    refs in ledger order. Thus a 64-slot request cannot export 63 retained slots.
    """
    validate_samples(samples,samples['sources'],allowed_train_ids);_ledger(slot_ledger,samples['sources'],allowed_train_ids)
    _requested_ledger(slot_ledger,samples['sources'],request_refs,allowed_train_ids)
    output=Path(output);need(not output.exists(),'Fresh auxiliary export directory required');output.mkdir(parents=True)
    path=output/'AUXILIARY_ARRAYS.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,**{k:samples[k] for k in ARRAY_KEYS})
    meta={k:v for k,v in samples.items() if k not in ARRAY_KEYS}
    meta.update(slot_ledger=deepcopy(slot_ledger),counts=_counts(samples,slot_ledger),request_refs=deepcopy(request_refs))
    metadata=rt.write_new(output/'METADATA.json',meta)
    return rt.write_new(output/'COMPLETE.json',dict(schema=SCHEMA,complete=True,metadata=metadata,
        arrays=rt.reference(path),metadata_sha256=metadata['sha256'],**FLAGS))


def load_preparation(complete_ref,allowed_train_ids):
    """Gate all source metadata before opening the separate numerical archive."""
    complete=rt.read_bound(complete_ref)
    need(complete.get('schema')==SCHEMA and complete.get('complete') is True,'Completed auxiliary preparation required');_flags(complete)
    metadata=rt.read_bound(complete['metadata']);_sources(metadata['sources'],allowed_train_ids)
    need(complete.get('metadata_sha256')==complete['metadata']['sha256'],'Metadata hash binding differs')
    _ledger(metadata['slot_ledger'],metadata['sources'],allowed_train_ids)
    _requested_ledger(metadata['slot_ledger'],metadata['sources'],metadata['request_refs'],allowed_train_ids)
    with np.load(rt.bound_path(complete['arrays']),allow_pickle=False) as archive:
        need(set(archive.files)==set(ARRAY_KEYS),'Separate auxiliary numerical keys required')
        samples={**{k:v for k,v in metadata.items() if k not in ('slot_ledger','counts','request_refs')},**{k:archive[k].copy() for k in ARRAY_KEYS}}
    validate_samples(samples,metadata['sources'],allowed_train_ids)
    need(metadata['counts']==_counts(samples,metadata['slot_ledger']),'Source-specific counts changed')
    return dict(samples=samples,sources=samples['sources'],slot_ledger=metadata['slot_ledger'],counts=metadata['counts'])


def concatenate_samples(parts,allowed_train_ids):
    """Join separately bound producer outputs, retaining each source table entry."""
    sources=[];arrays={k:[] for k in ARRAY_KEYS}
    for part in parts:
        validate_samples(part,part['sources'],allowed_train_ids)
        offset=len(sources);sources.extend(part['sources'])
        for key in ARRAY_KEYS:
            arrays[key].append(part[key]+offset if key=='source_indices' else part[key])
    shapes=dict(source_histories=(0,2,17),source_times=(0,2),relative_setters=(0,20,7),
        setter_is_pad=(0,20),source_indices=(0,))
    joined={k:np.concatenate(v,axis=0) if v else np.empty(shapes[k],dtype=bool if k=='setter_is_pad' else
        np.int64 if k=='source_indices' else float) for k,v in arrays.items()}
    return bind_samples(joined,sources,allowed_train_ids)


def _old_flags(value):
    need(all(value.get(k) is False for k in ('actions_executed','physics_qualified','training_eligible')),
         'Frozen diagnostic must remain unexecuted and unqualified')


def _oracle_preflight(complete_ref,frozen_request_ref,allowed):
    root=rt.read_bound(complete_ref);frozen=rt.read_bound(frozen_request_ref)
    _old_flags(root);_old_flags(frozen)
    need(root.get('gpu_launched') is False,'Frozen CPU-only diagnostic required')
    cases=frozen.get('cases',[]);results=root.get('results',[])
    need(isinstance(cases,list) and cases and isinstance(results,list) and len(cases)==len(results) and
         len({c['job'] for c in cases})==len(cases) and
         [r.get('job') for r in results]==[c['job'] for c in cases] and
         root.get('fixed_query_denominator')==frozen.get('fixed_query_denominator')==
            sum(len(c.get('physics_rows',[])) for c in cases), 'Frozen oracle slot denominator/order differs')
    pending=[]
    # Read EVERY source config first. Never open query JSON, numeric/native
    # payloads, or a frozen generator until all jobs are eligible TRAIN.
    for case,result in zip(cases,results):
        directory=Path(result['result']['path']).parent
        per=rt.read_bound(result['result'])
        request_ref=rt.reference(directory/'REQUEST.json');request=rt.read_bound(request_ref)
        source=request.get('source',{})
        need(isinstance(source,dict) and set(source)=={'config','owner','child','native'},
             'Exact frozen original student source references required')
        config=rt.read_bound(source['config'])
        meta=config.get('m2_given_contact_expanded_pilot_deployment',{}).get('metadata',{})
        _train(meta.get('subject_id'),meta.get('split'),allowed)
        _train(per.get('source_subject_id'),per.get('source_split'),allowed)
        pending.append(dict(case=case,result=result,directory=directory,complete=per,
            request=request,request_ref=request_ref,metadata=meta))
    manifest=rt.read_bound(frozen['source_manifest'])
    need(isinstance(manifest,list),'Frozen source manifest inventory required')
    expected=[r for r in manifest if r.get('file')=='sim/ausc_policy/m2_student_oracle_proposals.py']
    need(len(expected)==1,'One frozen oracle generator in source manifest required')
    generator_path=Path(frozen['source_manifest']['path']).parent/'sim/ausc_policy/m2_student_oracle_proposals.py'
    for item in pending:
        case=item['case'];request=item['request'];per=item['complete'];meta=item['metadata']
        _old_flags(request);_old_flags(per)
        generator=request['generator'];rt.bound_path(generator)
        need(Path(generator['path'])==generator_path and
             all(generator.get(k)==expected[0].get(k) for k in ('sha256','bytes')),
             'Oracle generator differs from explicitly bound frozen source manifest')
        need(request.get('source')==case.get('source') and request.get('plan_config')==case.get('plan_config') and
             request.get('physics_rows')==case.get('physics_rows') and
             isinstance(request.get('physics_rows'),list) and request['physics_rows'] and
             len(request['physics_rows'])==len(set(request['physics_rows'])) and
             all(type(r) is int and r>=12 and r%12==0 for r in request['physics_rows']),
             'Frozen oracle REQUEST source/config/query schedule changed')
        need(per.get('schema')=='m2_student_cpu_oracle_diagnostic_v1' and per.get('complete') is True and
             per.get('gpu_launched') is False and per.get('source_episode_verdict_unchanged') is True and
             per.get('source_subject_id')==meta['subject_id'] and per.get('source_split')=='train' and
             per.get('attempted_queries')==len(request['physics_rows']) and
             [r.get('physics_row') for r in per.get('outcomes',[])]==request['physics_rows'],
             'Complete fixed-denominator frozen TRAIN diagnostic required')
        rt.geometry.require_current(meta);reference=rt.pose.require_route(meta);reference.verify_sources()
        item['pose_id']=installed_pose_id(meta)
        for bound in [*request['source'].values(),request['plan_config']]:rt.bound_path(bound)
        native=rt.read_bound(request['source']['native'])
        need(native.get('success') is False,'Original negative student source verdict must remain negative')
        item['native_source_verdict']=False
    return root,frozen,pending


def _oracle_query(proposal,metadata,row):
    need(proposal.get('kind')=='recovery_proposal' and proposal.get('physics_executed') is False and
         proposal.get('training_eligible') is False,'Unexecuted CPU recovery proposal required')
    query=proposal.get('query',{});preview=proposal.get('preview',{})
    # Inspect metadata before accessing the saved genuine history values.
    need(query.get('metadata')==metadata and query.get('reference_sha256')==metadata['reference_sha256'] and
         proposal.get('query_sha256')==rt.digest(query),'Frozen query metadata/reference/hash changed')
    need(preview.get('all_240hz_commands_verified') is True and preview.get('physics_executed') is False,
         'Original full 240Hz computational proposal required')
    h=rt.shared.contract._array(query.get('history'),(2,17),'genuine frozen H2')
    times=rt.shared.contract._array(query.get('history_times'),(2,),'genuine frozen H2 times')
    q=rt.shared.contract._array(query.get('measured_q'),(7,),'fixed actual input origin')
    need(query.get('physics_row')==row and type(query.get('time_s')) in (int,float) and
         abs(query['time_s']-row/240.)<=1e-8 and times[-1]==query['time_s'] and
         abs(times[1]-times[0]-.05)<=1e-8 and np.array_equal(h[-1,:7],q) and
         query.get('context')==[metadata['episode_id'],metadata['target_id'],0,'APPROACH'],
         'Frozen genuine input origin/context/clocks differ')
    cells=np.asarray(preview.get('cells'),float)
    need(cells.ndim==2 and cells.shape[1:]==(7,) and 0<len(cells)<=2400 and np.isfinite(cells).all(),
         'Bounded finite proposed raw setter cells required')
    count=min(20,len(cells));relative=np.zeros((20,7));relative[:count]=cells[:count]-q
    return h,times,relative,np.arange(20)>=count


def import_student_oracle(complete_ref,frozen_request_ref,*,allowed_train_ids):
    """Import frozen source20 diagnostics without current model authentication.

    complete_ref is the top-level diagnostic COMPLETE.json; frozen_request_ref
    is its original FROZEN_REQUEST.json with the frozen source manifest and
    exact per-job inputs. Every attempted query survives in slot_ledger. Saved
    full proposal arrays are checked against the actual proposal, then ONLY
    the first raw setter is supervised because later feedback is unobserved.
    """
    root,frozen,pending=_oracle_preflight(complete_ref,frozen_request_ref,allowed_train_ids)
    parts=[];ledger=[]
    original_keys={'source_observed_histories','source_observation_times','source_physics_rows',
        'proposed_relative_setters','proposed_is_pad'}
    for item in pending:
        request=item['request'];per=item['complete'];meta=item['metadata'];directory=item['directory']
        outcomes=per['outcomes'];accepted=[o for o in outcomes if o.get('proposal_accepted') is True]
        need(per.get('accepted_proposals')==len(accepted),'Frozen accepted proposal count differs')
        queried=[]
        for outcome in outcomes:
            row=outcome['physics_row'];_old_flags(outcome)
            outcome_ref=rt.reference(directory/f'outcome-{row}.json')
            need(rt.read_bound(outcome_ref)==outcome,'Saved per-query outcome differs from COMPLETE ledger')
            slot=dict(slot_id='oracle:'+item['case']['job']+':'+str(row),subject_id=meta['subject_id'],
                pose_id=item['pose_id'],target_id=meta['target_id'],kind='student_oracle_unexecuted',
                status='rejected',physics_row=row,outcome=outcome_ref,**FLAGS)
            ledger.append(slot)
            if outcome.get('proposal_accepted') is not True:
                need(outcome.get('proposal_accepted') is False,'Explicit accepted/rejected oracle slot required')
                slot['rejection']=outcome.get('rejection','Frozen CPU oracle rejected query')
                continue
            proposal_ref=outcome.get('computational_proposal',{})
            need(Path(proposal_ref.get('path',''))==directory/f'query-{row}.json',
                 'Frozen per-query proposal path differs')
            proposal=rt.read_bound(proposal_ref)
            h,t,relative,mask=_oracle_query(proposal,meta,row)
            need(outcome.get('valid_proposed_cells')==int(np.count_nonzero(~mask)),
                 'Frozen proposed cell count differs from saved raw cells')
            queried.append((outcome,proposal_ref,h,t,relative,mask,slot))
        if accepted:
            with np.load(rt.bound_path(per['arrays']),allow_pickle=False) as archive:
                need(set(archive.files)==original_keys,'Exact frozen UNEXECUTED proposal keys required')
                payload={k:archive[k].copy() for k in original_keys}
            n=len(queried)
            shapes=dict(source_observed_histories=(n,2,17),source_observation_times=(n,2),source_physics_rows=(n,),
                proposed_relative_setters=(n,20,7),proposed_is_pad=(n,20))
            for key,shape in shapes.items():
                a=payload[key]
                need(a.shape==shape and (a.dtype==np.dtype(bool) if key=='proposed_is_pad' else
                     a.dtype.kind in 'iu' if key=='source_physics_rows' else a.dtype.kind in 'fiu' and np.isfinite(a).all()),
                     'Frozen proposal array shape/dtype/finite values differ: '+key)
            for i,(outcome,proposal_ref,h,t,relative,mask,slot) in enumerate(queried):
                need(payload['source_physics_rows'][i]==outcome['physics_row'] and
                     np.array_equal(payload['source_observed_histories'][i],h) and
                     np.array_equal(payload['source_observation_times'][i],t) and
                     np.array_equal(payload['proposed_relative_setters'][i],relative) and
                     np.array_equal(payload['proposed_is_pad'][i],mask),'Frozen NPZ differs from genuine query/raw proposed setters')
                # Existing source20 H20 proposal is not H20 executed supervision.
                first=np.zeros((1,20,7));first[0,0]=relative[0]
                arrays=dict(source_histories=h[None],source_times=t[None],relative_setters=first,
                    setter_is_pad=(np.arange(20)>=1)[None],source_indices=np.zeros(1,dtype=np.int64))
                bindings=dict(source=request['source']['config'],geometry=request['plan_config'],
                    generator=request['generator'],native=request['source']['native'],owner=request['source']['owner'],
                    child=request['source']['child'],proposal=proposal_ref,outcome=slot['outcome'],
                    request=item['request_ref'],diagnostic=item['result']['result'],
                    diagnostic_root=complete_ref,frozen_request=frozen_request_ref,frozen_manifest=frozen['source_manifest'],
                    original_arrays=per['arrays'])
                source=make_source('student_oracle_unexecuted',meta,item['pose_id'],bindings,
                    [dict(input_index=outcome['physics_row']//12,phase='APPROACH')])
                source.update(source_episode_verdict_unchanged=True,original_source_success=False,
                    supervision='first_raw_setter_only; later physical feedback/handoff unobserved',
                    frozen_physics_row=outcome['physics_row'])
                part=bind_samples(arrays,[source],allowed_train_ids);parts.append(part)
                slot.update(status='accepted',source_id=part['sources'][0]['source_id'])
        else:need(per.get('arrays') is None,'Rejected-only diagnostic must not invent numeric labels')
    samples=concatenate_samples(parts,allowed_train_ids)
    _ledger(ledger,samples['sources'],allowed_train_ids)
    return dict(samples=samples,slot_ledger=ledger,counts=_counts(samples,ledger))
