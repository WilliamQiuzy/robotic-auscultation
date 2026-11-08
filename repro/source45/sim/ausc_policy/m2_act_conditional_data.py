"""Typed V3 computational data. The complete ordered V2 parent stays verbatim.

Generated static ideal inputs and genuine source24 queries have separate origins.
Neither becomes native, executed, physical, or measured-future supervision.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import m2_act_auxiliary_data as old, m2_act_chunk_data as v2

rt=old.rt
need=rt.need
SCHEMA='m2_act_auxiliary_preparation_v3'
SOURCE_SCHEMA='m2_act_auxiliary_source_v3'
CONDITIONAL_KIND='conditional_ideal_continuation'
KINDS=old.KINDS+(CONDITIONAL_KIND,)
ORIGIN='generated_static_ideal_initial_state; not measured or executed'
ADMISSION=dict(schema='m2_act_conditional_computational_admission_v3',computational_optimization_eligible=True,
    oracle_labels='bound_full_computed_prefix_up_to20',conditional_labels='canonical_L2_raw_setter_continuation',
    future_measured_histories=False,**old.FLAGS)
PARENT_COMPLETE=dict(path='/opt/wam-runtime/fast-wam/datasets/act-posture-full-chunks-20261106-v1/COMPLETE.json',
    bytes=1129,sha256='6b08048ff52454254eeda57cf161fdfcdf37324cdfba53fa11c501999d087c14')
PARENT_ROWS=35120
PARENT_SOURCES=72
PARENT_SLOTS=76
SOURCE24_MANIFEST=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v24/SOURCE_MANIFEST.json',
    bytes=286136,sha256='169a593ac33b2d14625b580868faca37c1d34e52697a5412565f2f76c37e1b01')


def require_source24(manifest,generator):
    need(manifest==SOURCE24_MANIFEST,'Pinned immutable original source24 manifest required')
    expected=dict(path=str(Path(SOURCE24_MANIFEST['path']).parent/'sim/ausc_policy/m2_student_oracle_proposals.py'),
        bytes=v2.GENERATOR_BYTES,sha256=v2.GENERATOR_SHA256)
    need(generator==expected,'Unchanged pinned original source24 generator required; arbitrary Python rejected')
    rt.bound_path(manifest);rt.bound_path(generator)


def require_oracle_archive(source):
    refs=source['bindings']
    if source['schema']==v2.SOURCE_SCHEMA:v2.require_frozen_archive(refs['frozen_manifest'],refs['generator'])
    else:require_source24(refs['frozen_manifest'],refs['generator'])


def _require_closed_owner(ref):
    owner=rt.read_bound(ref);stage=owner.get('stage',{})
    need(owner.get('completed') is True and owner.get('gpu_released') is True and stage.get('passed') is True and
        type(stage.get('exit_code')) is int and stage['exit_code']==0 and stage.get('process_group_drained') is True and
        all(stage.get(k)==[] for k in ('errors','signals','deferred_signals')),
        'Entire original source24 owner must be closed normally, drained and released before oracle arrays')


def _sources(sources,allowed,*,production=True):
    need(isinstance(sources,list),'Explicit V3 source table required')
    # All identities precede ANY reference, schedule, or numerical reads.
    for source in sources:
        old._train(source.get('subject_id'),source.get('split'),allowed)
        old._train(source.get('metadata',{}).get('subject_id'),source.get('metadata',{}).get('split'),allowed)
    schedules={};new_seen=False
    for i,source in enumerate(sources):
        schema=source.get('schema');kind=source.get('kind');old._flags(source)
        need(kind in KINDS,'Known explicitly typed V3 computational source required')
        if schema==v2.SOURCE_SCHEMA:
            need(not new_seen,'Unchanged V2 parent source table must precede all V3 sources')
            checked=old._sources([source],allowed,source_schema=v2.SOURCE_SCHEMA)
            if checked:schedules[i]=checked[0]
            if kind=='student_oracle_unexecuted':require_oracle_archive(source)
        else:
            new_seen=True;need(schema==SOURCE_SCHEMA,'Explicit new V3 source schema required')
            if kind=='student_oracle_unexecuted':
                old._sources([source],allowed,source_schema=SOURCE_SCHEMA);require_oracle_archive(source)
                _require_closed_owner(source['bindings']['owner'])
                need(source.get('source_episode_verdict_unchanged') is True and source.get('original_source_success') is False,
                    'Original genuine negative source verdict must remain unchanged')
            else:
                need(kind==CONDITIONAL_KIND,'Only new conditional or genuine source24 sources may append to V2')
                _conditional_header(source,allowed,production=production)
        if kind=='planned_ideal_tracking' and production:
            need(source.get('verification_profile')=='original_full_cpu_preview' and source.get('injected_factories')==[],
                'Production full original preview required; fixture injection rejected')
    return schedules


def _conditional_header(source,allowed,*,production):
    keys={'schema','kind','subject_id','pose_id','target_id','split','metadata','observation_origin','generated_static_input',
        'future_measured_histories','synthetic_compression_m','history_clock','anchor','slot_index','windows','planning_limits',
        'verification_profile','injected_factories','geometry_artifacts','bindings'}|set(old.FLAGS)
    need(set(source)-{'source_id','sample_sha256'}==keys,'Exact generated conditional source fields required; no measured claims')
    meta=source['metadata'];rt.geometry.require_current(meta);reference=rt.pose.require_route(meta)
    need(source.get('observation_origin')==ORIGIN and source.get('generated_static_input') is True and
        source.get('future_measured_histories') is False and source.get('synthetic_compression_m')==0. and
        source.get('history_clock')=='generated H2 [0,.05]s; static q; no native ticks',
        'Conditional ideal provenance must never claim actual/measured observations')
    need(all(source.get(k)==meta.get(k) for k in ('subject_id','split','target_id')) and
        source.get('pose_id')==old.installed_pose_id(meta),'Conditional identity/pose/target differs')
    need(source.get('windows')==[dict(input_index=1,phase='APPROACH',anchor=source.get('anchor'))],
        'One generated static conditional H2 per source required')
    refs=source.get('bindings',{})
    need({'source','geometry','generator','data_protocol','numerical_preview','retained_dynamics','raw_schedule',
        'frozen_request','candidate'}<=set(refs),'Complete conditional source/request/code/preview bindings required')
    for bound in refs.values():rt.bound_path(bound)
    need(refs['generator']==rt.reference(Path(__file__).with_name('m2_act_conditional_supervision.py')) and
        refs['data_protocol']==rt.reference(__file__),'Conditional producer/data module bytes changed')
    from . import m2_given_contact_recovery as recovery
    need(refs['numerical_preview']==rt.reference(recovery.__file__) and refs['retained_dynamics']==rt.reference(rt.handoff.__file__),
        'Conditional preview/dynamics code bindings differ')
    if production:
        need(source.get('verification_profile')=='original_full_cpu_preview' and source.get('injected_factories')==[],
            'Production conditional full verification required; injected fixture rejected')
    need(source.get('planning_limits')==rt.handoff.PLANNING_LIMITS,'Original conditional planning limits required')
    reference.verify_sources()


def validate_samples(samples,sources,allowed_train_ids,*,production=True):
    schedules=_sources(sources,allowed_train_ids,production=production)
    need(samples.get('schema')==SCHEMA and samples.get('optimization_admission')==ADMISSION,
        'Explicit V3 computational optimization admission required');old._flags(samples)
    need(samples.get('sources')==sources and samples.get('source_table_sha256')==rt.digest(sources),
        'V3 source table binding differs')
    arrays=old._numeric(samples,sources,full_oracle_prefix=True)
    for i,source in enumerate(sources):
        rows=np.flatnonzero(arrays['source_indices']==i)
        need(source.get('source_id')==old._source_id(source) and
            source.get('sample_sha256')==old._row_digest(arrays,rows),'V3 numeric/source binding changed')
        if i in schedules:
            raw=old._planned_schedule_array(schedules[i])
            for row,w in zip(rows,source['windows']):
                start=w['input_index'];count=int(np.count_nonzero(~arrays['setter_is_pad'][row]))
                need(start+count<=len(raw) and np.array_equal(arrays['relative_setters'][row,:count],
                    raw[start:start+count]-arrays['source_histories'][row,-1,:7]),'Original parent planned labels changed')
        elif source['kind']=='student_oracle_unexecuted':
            proposal=rt.read_bound(source['bindings']['proposal'])
            v2.validate_oracle_sample(samples,i,proposal,proposal['query'])
        else:
            from . import m2_act_conditional_supervision as producer
            producer.validate_candidate(source,arrays,rows,allowed_train_ids,production=production)
    return samples


def bind_samples(arrays,sources,allowed_train_ids,*,production=True):
    sources=deepcopy(sources);_sources(sources,allowed_train_ids,production=production)
    arrays=old._numeric(arrays,sources,full_oracle_prefix=True)
    for i,source in enumerate(sources):
        rows=np.flatnonzero(arrays['source_indices']==i)
        digest=old._row_digest(arrays,rows)
        if source['schema']==v2.SOURCE_SCHEMA:
            need(source.get('sample_sha256')==digest and source.get('source_id')==old._source_id(source),
                'Do not reseal or relabel unchanged V2 parent sources')
        else:source['sample_sha256']=digest;source['source_id']=old._source_id(source)
    samples=dict(schema=SCHEMA,sources=sources,source_table_sha256=rt.digest(sources),allowed_train_ids=sorted(allowed_train_ids),
        optimization_admission=deepcopy(ADMISSION),**old.FLAGS,**{k:v.copy() for k,v in arrays.items()})
    return validate_samples(samples,sources,allowed_train_ids,production=production)


def import_student_oracle(complete_ref,frozen_request_ref,*,allowed_train_ids):
    """Full source24 prefixes; actual history authentication runs in frozen code later."""
    _,frozen,pending=old._oracle_preflight(complete_ref,frozen_request_ref,allowed_train_ids)
    for item in pending:
        require_source24(frozen['source_manifest'],item['request']['generator'])
        _require_closed_owner(item['request']['source']['owner'])
    imported=old.import_student_oracle(complete_ref,frozen_request_ref,allowed_train_ids=allowed_train_ids)
    samples=imported['samples'];arrays={k:samples[k].copy() for k in old.ARRAY_KEYS};sources=[]
    for i,source in enumerate(samples['sources']):
        source=deepcopy(source);source.pop('source_id');source.pop('sample_sha256');source['schema']=SOURCE_SCHEMA
        source['supervision']='full_computed_oracle_prefix_up_to20; genuine H2 only; no future measured histories'
        proposal=rt.read_bound(source['bindings']['proposal'])
        _,_,relative,mask=old._oracle_query(proposal,source['metadata'],source['frozen_physics_row'])
        row=int(np.flatnonzero(arrays['source_indices']==i)[0]);arrays['relative_setters'][row]=relative;arrays['setter_is_pad'][row]=mask
        sources.append(source)
    result=bind_samples(arrays,sources,allowed_train_ids);ledger=deepcopy(imported['slot_ledger'])
    mapping={a['source_id']:b['source_id'] for a,b in zip(samples['sources'],result['sources'])}
    for row in ledger:
        if row['status']=='accepted':row['source_id']=mapping[row['source_id']]
    return dict(samples=result,sources=result['sources'],slot_ledger=ledger)


def concatenate_samples(parts,allowed_train_ids,*,production=True):
    need(bool(parts),'Explicit V3 parts required');sources=[];arrays={k:[] for k in old.ARRAY_KEYS}
    for part in parts:
        if part['schema']==v2.SCHEMA:v2.validate_samples(part,part['sources'],allowed_train_ids)
        else:validate_samples(part,part['sources'],allowed_train_ids,production=production)
        offset=len(sources);sources.extend(part['sources'])
        for k in arrays:arrays[k].append(part[k]+offset if k=='source_indices' else part[k])
    return bind_samples({k:np.concatenate(v) for k,v in arrays.items()},sources,allowed_train_ids,production=production)


def prove_parent(parent,candidate):
    before=parent['samples'];after=candidate['samples']
    proof=v2.prove_ordered_prefix(before,after,len(before['source_indices']))
    need(after['sources'][:len(before['sources'])]==before['sources'],'Exact ordered V2 parent sources/source IDs changed')
    need(candidate['slot_ledger'][:len(parent['slot_ledger'])]==parent['slot_ledger'],'Exact ordered V2 parent ledger changed')
    need(candidate['request_refs'][:len(parent['request_refs'])]==parent['request_refs'],'Exact ordered V2 parent requests changed')
    return proof


def load_parent(allowed_train_ids):
    parent=v2.load_preparation(PARENT_COMPLETE,allowed_train_ids);v2.validate_lineage(parent,allowed_train_ids)
    need(len(parent['samples']['source_indices'])==PARENT_ROWS and len(parent['sources'])==PARENT_SOURCES and
        len(parent['slot_ledger'])==PARENT_SLOTS,'Exact complete V2 35120-row/72-source/76-slot parent required')
    return parent


def counts(samples,ledger):
    return {kind:dict(attempted_slots=sum(r['kind']==kind for r in ledger),
        accepted_sources=sum(s['kind']==kind for s in samples['sources']),
        samples=sum(len(s['windows']) for s in samples['sources'] if s['kind']==kind),
        supervised_cells=int(sum(np.count_nonzero(~samples['setter_is_pad'][samples['source_indices']==i])
            for i,s in enumerate(samples['sources']) if s['kind']==kind))) for kind in KINDS}


def _ledger(ledger,sources,allowed):
    seen=set();accepted=[]
    for row in ledger:
        old._train(row.get('subject_id'),'train',allowed);old._flags(row)
        need(isinstance(row.get('slot_id'),str) and row['slot_id'] not in seen and row.get('kind') in KINDS and
            row.get('status') in ('accepted','rejected'),'Unique explicit V3 attempted slot required')
        seen.add(row['slot_id'])
        if row['status']=='accepted':
            matches=[s for s in sources if s['source_id']==row.get('source_id')]
            need(len(matches)==1 and all(matches[0][k]==row.get(k) for k in ('subject_id','pose_id','target_id','kind')),
                'Accepted V3 slot differs from its source');accepted.append(row['source_id'])
        else:need(isinstance(row.get('rejection'),str) and bool(row['rejection']),'Every rejected slot needs its original reason')
    need(sorted(accepted)==sorted(s['source_id'] for s in sources),'Every accepted V3 source needs exactly one slot')


def validate_lineage(auxiliary,allowed_train_ids):
    from . import m2_act_conditional_supervision as producer
    p=auxiliary.get('provenance',{});need(p.get('parent_complete')==PARENT_COMPLETE,'Pinned complete V2 parent required')
    parent=load_parent(allowed_train_ids);proof=prove_parent(parent,auxiliary)
    need(p.get('ordered_parent_proof')==proof,'Exact full V2 parent proof differs')
    runs=p.get('new_oracle_runs');need(isinstance(runs,list) and runs,'New pinned source24 oracle runs required')
    parts=[parent['samples']];ledger=deepcopy(parent['slot_ledger']);refs=deepcopy(parent['request_refs'])
    for run in runs:
        need(set(run)=={'complete','frozen_request'},'Exact source24 completed diagnostic/request refs required')
        got=import_student_oracle(run['complete'],run['frozen_request'],allowed_train_ids=allowed_train_ids)
        parts.append(got['samples']);ledger.extend(got['slot_ledger']);refs.append(run['frozen_request'])
    conditional=producer.load_generation(p['conditional_complete'],allowed_train_ids)
    parts.append(conditional['samples']);ledger.extend(conditional['slot_ledger']);refs.append(conditional['request_ref'])
    expected=concatenate_samples(parts,allowed_train_ids)
    need(auxiliary['samples']['sources']==expected['sources'] and auxiliary['slot_ledger']==ledger and
        auxiliary['request_refs']==refs,'Every V3 accepted/rejected source/request/outcome must survive in order')
    v2.prove_ordered_prefix(expected,auxiliary['samples'],len(expected['source_indices']))
    need(len(expected['source_indices'])==len(auxiliary['samples']['source_indices']),'No unbound V3 tail rows allowed')
    return proof


def export_from_parent(output,*,oracle_runs,conditional_complete_ref,allowed_train_ids):
    from . import m2_act_conditional_supervision as producer
    # All source identity metadata, including rejected requests, precede parent numerics.
    for run in oracle_runs:
        _,frozen,pending=old._oracle_preflight(run['complete'],run['frozen_request'],allowed_train_ids)
        for item in pending:
            require_source24(frozen['source_manifest'],item['request']['generator'])
            _require_closed_owner(item['request']['source']['owner'])
    producer.generation_metadata(conditional_complete_ref,allowed_train_ids)
    parent=load_parent(allowed_train_ids);parts=[parent['samples']];ledger=deepcopy(parent['slot_ledger']);refs=deepcopy(parent['request_refs'])
    for run in oracle_runs:
        got=import_student_oracle(run['complete'],run['frozen_request'],allowed_train_ids=allowed_train_ids)
        parts.append(got['samples']);ledger.extend(got['slot_ledger']);refs.append(run['frozen_request'])
    conditional=producer.load_generation(conditional_complete_ref,allowed_train_ids)
    parts.append(conditional['samples']);ledger.extend(conditional['slot_ledger']);refs.append(conditional['request_ref'])
    samples=concatenate_samples(parts,allowed_train_ids)
    aux=dict(samples=samples,sources=samples['sources'],slot_ledger=ledger,request_refs=refs)
    aux['provenance']=dict(parent_complete=deepcopy(PARENT_COMPLETE),ordered_parent_proof=prove_parent(parent,aux),
        new_oracle_runs=deepcopy(oracle_runs),conditional_complete=deepcopy(conditional_complete_ref))
    return export_preparation(output,aux,allowed_train_ids)


def export_preparation(output,auxiliary,allowed_train_ids):
    samples=auxiliary['samples'];validate_samples(samples,samples['sources'],allowed_train_ids)
    _ledger(auxiliary['slot_ledger'],samples['sources'],allowed_train_ids);validate_lineage(auxiliary,allowed_train_ids)
    output=Path(output);need(not output.exists(),'Fresh V3 preparation directory required');output.mkdir(parents=True)
    path=output/'AUXILIARY_ARRAYS.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,**{k:samples[k] for k in old.ARRAY_KEYS})
    meta={k:v for k,v in samples.items() if k not in old.ARRAY_KEYS}
    meta.update(slot_ledger=deepcopy(auxiliary['slot_ledger']),request_refs=deepcopy(auxiliary['request_refs']),
        provenance=deepcopy(auxiliary['provenance']),counts=counts(samples,auxiliary['slot_ledger']))
    bound=rt.write_new(output/'METADATA.json',meta)
    return rt.write_new(output/'COMPLETE.json',dict(schema=SCHEMA,complete=True,metadata=bound,arrays=rt.reference(path),
        metadata_sha256=bound['sha256'],optimization_admission=deepcopy(ADMISSION),**old.FLAGS))


def metadata(ref,allowed_train_ids):
    complete=rt.read_bound(ref)
    need(complete.get('schema')==SCHEMA and complete.get('complete') is True and complete.get('optimization_admission')==ADMISSION,
        'Complete explicit V3 preparation required');old._flags(complete)
    meta=rt.read_bound(complete['metadata']);_sources(meta['sources'],allowed_train_ids)
    need(meta.get('optimization_admission')==ADMISSION and complete.get('metadata_sha256')==complete['metadata']['sha256'],
        'Explicit V3 metadata binding/admission required');old._flags(meta)
    _ledger(meta['slot_ledger'],meta['sources'],allowed_train_ids)
    return complete,meta


def load_preparation(complete_ref,allowed_train_ids):
    complete,meta=metadata(complete_ref,allowed_train_ids)
    with np.load(rt.bound_path(complete['arrays']),allow_pickle=False) as archive:
        need(set(archive.files)==set(old.ARRAY_KEYS),'Exact separate V3 numerical array keys required')
        samples={**{k:v for k,v in meta.items() if k not in ('slot_ledger','counts','request_refs','provenance')},
            **{k:archive[k].copy() for k in old.ARRAY_KEYS}}
    validate_samples(samples,meta['sources'],allowed_train_ids)
    auxiliary=dict(samples=samples,sources=samples['sources'],**{k:meta[k] for k in ('slot_ledger','counts','request_refs','provenance')})
    need(meta['counts']==counts(samples,meta['slot_ledger']),'V3 separate computational denominators differ')
    validate_lineage(auxiliary,allowed_train_ids);return auxiliary


def pack_batch(samples,real_native_stats,indices):
    validate_samples(samples,samples['sources'],samples['allowed_train_ids']);stats=rt.validate_normalization(real_native_stats)
    ids=np.asarray(indices);need(ids.ndim==1 and ids.dtype.kind in 'iu' and np.all(ids>=0) and
        np.all(ids<len(samples['source_indices'])),'Valid V3 batch indices required')
    h=(samples['source_histories'][ids]-np.asarray(stats['state_mean']))/np.asarray(stats['state_std'])
    a=(samples['relative_setters'][ids]-np.asarray(stats['action_mean']))/np.asarray(stats['action_std'])
    return {'observation.state':h[:,1,:7].astype(np.float32),
        'observation.environment_state':np.concatenate([h[:,1,7:],h[:,0]],axis=1).astype(np.float32),
        'action':a.astype(np.float32),'action_is_pad':samples['setter_is_pad'][ids].copy()}
