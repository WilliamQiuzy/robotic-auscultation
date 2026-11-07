"""V2 full computed oracle prefixes, separate from native demonstrations.

The original planned array prefix is copied, never regenerated. Genuine H2 is
observed once; later cells are computed targets and never future measurements.
Legacy v1 APIs retain first-cell-only admission.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from . import m2_act_auxiliary_data as old

rt=old.rt
need=rt.need
SCHEMA='m2_act_auxiliary_preparation_v2'
SOURCE_SCHEMA='m2_act_auxiliary_source_v2'
ADMISSION=dict(schema='m2_act_full_computed_oracle_prefix_admission_v1',computational_optimization_eligible=True,
    oracle_labels='bound_full_computed_prefix_up_to20',future_measured_histories=False,**old.FLAGS)
LEGACY_COMPLETE=dict(path='/opt/wam-runtime/fast-wam/datasets/act-train16-auxiliary-20261106-v1/COMPLETE.json',
    bytes=739,sha256='2bf5a9715712425325881d90e8c5e518d9dad311f105dbc8c25c64482198d110')
PLANNED_COUNT=35108
SOURCE20_MANIFEST=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v20/SOURCE_MANIFEST.json',
    bytes=282448,sha256='34a722d283db0dfa78718e037b2473739eaf066fc5c455343b8740b7b929c844')
SOURCE23_MANIFEST=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v23/SOURCE_MANIFEST.json',
    bytes=284304,sha256='17099fb13c44590a7bd108402dfe9708a91ac625b387b8f171e880e7d4f36175')
GENERATOR_BYTES=11637
GENERATOR_SHA256='b507ed21c339d2ca371ccb9b08472aeca4a1de2abb89080b45cf9ac1fb594e5e'


def require_frozen_archive(manifest,generator):
    need(manifest in (SOURCE20_MANIFEST,SOURCE23_MANIFEST),'Pinned original source20/source23 frozen manifest required')
    expected=dict(path=str(Path(manifest['path']).parent/'sim/ausc_policy/m2_student_oracle_proposals.py'),
        bytes=GENERATOR_BYTES,sha256=GENERATOR_SHA256)
    need(generator==expected,'Pinned original frozen generator required; arbitrary Python cannot self-authenticate')
    rt.bound_path(manifest);rt.bound_path(generator)


def _source(source):
    value=deepcopy(source)
    value.pop('source_id',None);value.pop('sample_sha256',None)
    value['schema']=SOURCE_SCHEMA
    if value['kind']=='student_oracle_unexecuted':
        value['supervision']='full_computed_oracle_prefix_up_to20; no future measured histories'
    return value


def validate_oracle_sample(samples,source_index,proposal,authenticated_query,*,snapshot=None):
    source=samples['sources'][source_index]
    need(proposal.get('query')==authenticated_query,'Computed oracle query differs from genuine authenticated input')
    h,t,relative,mask=old._oracle_query(proposal,source['metadata'],source['frozen_physics_row'])
    rows=np.flatnonzero(samples['source_indices']==source_index)
    need(len(rows)==1,'Exactly one genuine H2 per computed oracle source required')
    row=int(rows[0])
    need(np.array_equal(samples['source_histories'][row],h) and np.array_equal(samples['source_times'][row],t) and
         np.array_equal(samples['relative_setters'][row],relative) and np.array_equal(samples['setter_is_pad'][row],mask),
         'Full computed prefix must equal every bound preview cell relative to the genuine H2 current origin')
    if snapshot is not None:
        preview=proposal.get('preview',{})
        need(preview.get('input_state')==snapshot and preview.get('input_sha256')==rt.digest(snapshot),
             'Computed oracle retained actual command q/v/a differs from the authenticated pre-query seam')


def bind_samples(arrays,sources,allowed_train_ids):
    sources=deepcopy(sources);old._sources(sources,allowed_train_ids,source_schema=SOURCE_SCHEMA)
    arrays=old._numeric(arrays,sources,full_oracle_prefix=True)
    for i,source in enumerate(sources):
        source['sample_sha256']=old._row_digest(arrays,np.flatnonzero(arrays['source_indices']==i))
        source['source_id']=old._source_id(source)
    samples=dict(schema=SCHEMA,sources=sources,source_table_sha256=rt.digest(sources),
        allowed_train_ids=sorted(allowed_train_ids),optimization_admission=deepcopy(ADMISSION),**old.FLAGS,
        **{k:v.copy() for k,v in arrays.items()})
    return validate_samples(samples,sources,allowed_train_ids)


def validate_samples(samples,sources,allowed_train_ids):
    schedules=old._sources(sources,allowed_train_ids,source_schema=SOURCE_SCHEMA)
    need(samples.get('schema')==SCHEMA and samples.get('optimization_admission')==ADMISSION,
         'Explicit V2 full computed-prefix optimization admission required')
    old._flags(samples)
    need(samples.get('sources')==sources and samples.get('source_table_sha256')==rt.digest(sources),
         'V2 source table binding differs')
    arrays=old._numeric(samples,sources,full_oracle_prefix=True)
    for i,source in enumerate(sources):
        rows=np.flatnonzero(arrays['source_indices']==i)
        need(source.get('source_id')==old._source_id(source) and
             source.get('sample_sha256')==old._row_digest(arrays,rows),'V2 numeric source binding changed')
        if i in schedules:
            raw=old._planned_schedule_array(schedules[i])
            for row,w in zip(rows,source['windows']):
                start=w['input_index'];count=int(np.count_nonzero(~arrays['setter_is_pad'][row]))
                need(start+count<=len(raw) and np.array_equal(arrays['relative_setters'][row,:count],
                     raw[start:start+count]-arrays['source_histories'][row,-1,:7]),'Planned prefix differs from original bound schedule')
        else:
            refs=source['bindings'];require_frozen_archive(refs['frozen_manifest'],refs['generator'])
            proposal=rt.read_bound(refs['proposal'])
            validate_oracle_sample(samples,i,proposal,proposal['query'])
    return samples


def import_student_oracle(complete_ref,frozen_request_ref,*,allowed_train_ids):
    """Validate original full NPZ through v1, then explicitly retain all K cells."""
    # Entire new cohort is gated before archive bytes or proposal arrays.
    _,frozen,pending=old._oracle_preflight(complete_ref,frozen_request_ref,allowed_train_ids)
    for item in pending:require_frozen_archive(frozen['source_manifest'],item['request']['generator'])
    imported=old.import_student_oracle(complete_ref,frozen_request_ref,allowed_train_ids=allowed_train_ids)
    original=imported['samples'];arrays={k:original[k].copy() for k in old.ARRAY_KEYS}
    sources=[_source(s) for s in original['sources']]
    for i,source in enumerate(sources):
        proposal=rt.read_bound(source['bindings']['proposal'])
        _,_,relative,mask=old._oracle_query(proposal,source['metadata'],source['frozen_physics_row'])
        row=int(np.flatnonzero(arrays['source_indices']==i)[0])
        arrays['relative_setters'][row]=relative;arrays['setter_is_pad'][row]=mask
    samples=bind_samples(arrays,sources,allowed_train_ids)
    mapping={s['source_id']:new['source_id'] for s,new in zip(original['sources'],samples['sources'])}
    ledger=deepcopy(imported['slot_ledger'])
    for row in ledger:
        if row['status']=='accepted':row['source_id']=mapping[row['source_id']]
    return dict(samples=samples,sources=samples['sources'],slot_ledger=ledger,counts=old._counts(samples,ledger))


def concatenate_samples(parts,allowed_train_ids):
    sources=[];arrays={k:[] for k in old.ARRAY_KEYS}
    for part in parts:
        validate_samples(part,part['sources'],allowed_train_ids);offset=len(sources);sources.extend(part['sources'])
        for key in arrays:arrays[key].append(part[key]+offset if key=='source_indices' else part[key])
    need(bool(parts),'Explicit V2 sample parts required')
    return bind_samples({k:np.concatenate(values) for k,values in arrays.items()},sources,allowed_train_ids)


def pack_batch(samples,real_native_stats,indices):
    validate_samples(samples,samples['sources'],samples['allowed_train_ids'])
    stats=rt.validate_normalization(real_native_stats);ids=np.asarray(indices)
    need(ids.ndim==1 and ids.dtype.kind in 'iu' and np.all(ids>=0) and np.all(ids<len(samples['source_indices'])),
         'Valid V2 batch indices required')
    h=(samples['source_histories'][ids]-np.asarray(stats['state_mean']))/np.asarray(stats['state_std'])
    a=(samples['relative_setters'][ids]-np.asarray(stats['action_mean']))/np.asarray(stats['action_std'])
    return {'observation.state':h[:,1,:7].astype(np.float32),
        'observation.environment_state':np.concatenate([h[:,1,7:],h[:,0]],axis=1).astype(np.float32),
        'action':a.astype(np.float32),'action_is_pad':samples['setter_is_pad'][ids].copy()}


def export_preparation(output,samples,slot_ledger,allowed_train_ids,*,request_refs,provenance):
    validate_samples(samples,samples['sources'],allowed_train_ids)
    old._ledger(slot_ledger,samples['sources'],allowed_train_ids)
    old._requested_ledger(slot_ledger,samples['sources'],request_refs,allowed_train_ids)
    output=Path(output);need(not output.exists(),'Fresh V2 export directory required');output.mkdir(parents=True)
    path=output/'AUXILIARY_ARRAYS.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,**{k:samples[k] for k in old.ARRAY_KEYS})
    meta={k:v for k,v in samples.items() if k not in old.ARRAY_KEYS}
    meta.update(slot_ledger=deepcopy(slot_ledger),counts=old._counts(samples,slot_ledger),
        request_refs=deepcopy(request_refs),provenance=deepcopy(provenance))
    bound=rt.write_new(output/'METADATA.json',meta)
    return rt.write_new(output/'COMPLETE.json',dict(schema=SCHEMA,complete=True,metadata=bound,
        arrays=rt.reference(path),metadata_sha256=bound['sha256'],optimization_admission=deepcopy(ADMISSION),**old.FLAGS))


def load_preparation(complete_ref,allowed_train_ids):
    complete=rt.read_bound(complete_ref)
    need(complete.get('schema')==SCHEMA and complete.get('complete') is True and complete.get('optimization_admission')==ADMISSION,
         'Completed explicit V2 computed-prefix preparation required');old._flags(complete)
    meta=rt.read_bound(complete['metadata']);old._sources(meta['sources'],allowed_train_ids,source_schema=SOURCE_SCHEMA)
    need(complete.get('metadata_sha256')==complete['metadata']['sha256'],'V2 metadata hash differs')
    old._ledger(meta['slot_ledger'],meta['sources'],allowed_train_ids)
    old._requested_ledger(meta['slot_ledger'],meta['sources'],meta['request_refs'],allowed_train_ids)
    with np.load(rt.bound_path(complete['arrays']),allow_pickle=False) as archive:
        need(set(archive.files)==set(old.ARRAY_KEYS),'Separate V2 arrays required')
        samples={**{k:v for k,v in meta.items() if k not in ('slot_ledger','counts','request_refs','provenance')},
            **{k:archive[k].copy() for k in old.ARRAY_KEYS}}
    validate_samples(samples,meta['sources'],allowed_train_ids)
    need(meta['counts']==old._counts(samples,meta['slot_ledger']),'V2 source counts differ')
    return dict(samples=samples,sources=samples['sources'],slot_ledger=meta['slot_ledger'],counts=meta['counts'],
        provenance=meta['provenance'],request_refs=meta['request_refs'])


def prove_ordered_prefix(original,candidate,count):
    proof={}
    for key in old.ARRAY_KEYS:
        before=np.asarray(original[key])[:count];after=np.asarray(candidate[key])[:count]
        need(before.dtype==after.dtype and np.array_equal(before,after),'Exact original ordered planned array prefix changed: '+key)
        proof[key]=dict(shape=list(before.shape),dtype=str(before.dtype),sha256=hashlib.sha256(np.ascontiguousarray(before).tobytes()).hexdigest())
    return proof


def require_current_targets(sources):
    current={s['target_id'] for s in sources if s['kind']=='student_oracle_unexecuted' and
        s['bindings'].get('frozen_manifest')==SOURCE23_MANIFEST}
    need(current=={'R3','L3'},'Accepted new current-policy source23 TRAIN oracle query required for both R3 and L3; keep rejections')


def export_from_legacy(output,legacy_complete_ref,*,oracle_runs,allowed_train_ids):
    """Root API: convert old six automatically; oracle_runs are NEW source23 only.

    oracle_runs=[{'complete': top_COMPLETE_ref, 'frozen_request': FROZEN_REQUEST_ref}].
    All 70 original request slots plus every new slot survive in original order.
    """
    need(legacy_complete_ref==LEGACY_COMPLETE,'Pinned unchanged original auxiliary bank required')
    complete=rt.read_bound(legacy_complete_ref);meta=rt.read_bound(complete['metadata'])
    old._sources(meta['sources'],allowed_train_ids)
    need(isinstance(oracle_runs,list) and oracle_runs,'New source23 frozen diagnostics required')
    # Reject all new VAL/TEST/fresh source headers before opening old arrays too.
    for run in oracle_runs:
        _,frozen,pending=old._oracle_preflight(run['complete'],run['frozen_request'],allowed_train_ids)
        need(frozen['source_manifest']==SOURCE23_MANIFEST,'Only new source23 runs may be appended')
        for item in pending:require_frozen_archive(frozen['source_manifest'],item['request']['generator'])
    legacy=old.load_preparation(legacy_complete_ref,allowed_train_ids);s=legacy['samples']
    planned_sources=[source for source in s['sources'] if source['kind']=='planned_ideal_tracking']
    n=len(planned_sources)
    need(len(s['source_indices'])==PLANNED_COUNT+6 and len(meta['slot_ledger'])==70 and
         np.all(s['source_indices'][:PLANNED_COUNT]<n) and np.all(s['source_indices'][PLANNED_COUNT:]>=n) and
         s['sources'][:n]==planned_sources,'Original 35108 ordered planned prefix and six old oracle queries required')
    planned=bind_samples({k:s[k][:PLANNED_COUNT].copy() for k in old.ARRAY_KEYS},[_source(v) for v in planned_sources],allowed_train_ids)
    parts=[planned];old_oracles=[source for source in s['sources'] if source['kind']=='student_oracle_unexecuted']
    runs=[]
    for source in old_oracles:
        run=dict(complete=source['bindings']['diagnostic_root'],frozen_request=source['bindings']['frozen_request'])
        if run not in runs:runs.append(run)
    need(len(runs)==1,'One original old-six frozen oracle diagnostic required')
    converted=import_student_oracle(**{'complete_ref':runs[0]['complete'],'frozen_request_ref':runs[0]['frozen_request']},allowed_train_ids=allowed_train_ids)
    need(len(converted['samples']['source_indices'])==6,'All six original proposals must be retained')
    parts.append(converted['samples'])
    mapping={a['source_id']:b['source_id'] for a,b in zip(planned_sources,planned['sources'])}
    mapping.update({a['source_id']:b['source_id'] for a,b in zip(old_oracles,converted['sources'])})
    ledger=deepcopy(meta['slot_ledger'])
    for row in ledger:
        if row['status']=='accepted':row['source_id']=mapping[row['source_id']]
    request_refs=deepcopy(meta['request_refs'])
    for run in oracle_runs:
        got=import_student_oracle(run['complete'],run['frozen_request'],allowed_train_ids=allowed_train_ids)
        parts.append(got['samples']);ledger.extend(got['slot_ledger']);request_refs.append(run['frozen_request'])
    combined=concatenate_samples(parts,allowed_train_ids);require_current_targets(combined['sources'])
    proof=prove_ordered_prefix(s,combined,PLANNED_COUNT)
    provenance=dict(legacy_complete=legacy_complete_ref,legacy_arrays=complete['arrays'],planned_count=PLANNED_COUNT,
        planned_ordered_prefix=proof,legacy_source_ids=[v['source_id'] for v in planned_sources],
        legacy_converted_oracles=6,original_request_slots=70,new_oracle_runs=deepcopy(oracle_runs))
    return export_preparation(output,combined,ledger,allowed_train_ids,request_refs=request_refs,provenance=provenance)


def validate_lineage(auxiliary,allowed_train_ids):
    """Reproduce the ordered-array/ref proof from the pinned original bank once."""
    p=auxiliary.get('provenance',{})
    need(p.get('legacy_complete')==LEGACY_COMPLETE and p.get('planned_count')==PLANNED_COUNT and
         p.get('legacy_converted_oracles')==6 and p.get('original_request_slots')==70,
         'Original planned-prefix and old-six provenance required')
    complete=rt.read_bound(LEGACY_COMPLETE)
    need(p.get('legacy_arrays')==complete['arrays'],'Original planned array reference differs')
    original=old.load_preparation(LEGACY_COMPLETE,allowed_train_ids)['samples']
    planned=[s for s in original['sources'] if s['kind']=='planned_ideal_tracking']
    need(p.get('legacy_source_ids')==[s['source_id'] for s in planned], 'Original planned source order differs')
    actual=auxiliary['samples'];proof=prove_ordered_prefix(original,actual,PLANNED_COUNT)
    need(p.get('planned_ordered_prefix')==proof,'Original ordered-prefix proof differs')
    # All source bindings, metadata and causal windows must remain exactly old.
    need(actual['sources'][:len(planned)]==[_reseal_planned(s) for s in planned],
         'Original planned source/reference prefix changed')
    metadata=rt.read_bound(complete['metadata']) if 'metadata' in complete else complete
    need(auxiliary['request_refs'][:len(metadata['request_refs'])]==metadata['request_refs'],
         'All original auxiliary request slots must precede appended requests')
    converted=_upgrade_legacy_samples(original,allowed_train_ids)
    prove_ordered_prefix(converted,actual,len(original['source_indices']))
    need(actual['sources'][:len(converted['sources'])]==converted['sources'],
         'Every original planned source and converted old oracle query must remain in order')
    mapping={a['source_id']:b['source_id'] for a,b in zip(original['sources'],converted['sources'])}
    ledger=deepcopy(metadata['slot_ledger'])
    for row in ledger:
        if row['status']=='accepted':row['source_id']=mapping[row['source_id']]
    need(auxiliary['slot_ledger'][:len(ledger)]==ledger,'All original70 accepted/rejected outcomes must remain unchanged')
    for source in actual['sources'][len(converted['sources']):]:
        need(source['kind']=='student_oracle_unexecuted' and source['bindings']['frozen_manifest']==SOURCE23_MANIFEST,
             'Only new source23 genuine oracle queries may follow the original bank')
    validate_new_runs(auxiliary,metadata,original,allowed_train_ids)
    require_current_targets(actual['sources'])
    return proof


def _reseal_planned(source):
    value=_source(source)
    value['sample_sha256']=source['sample_sha256'];value['source_id']=old._source_id(value)
    return value


def _upgrade_legacy_samples(original,allowed_train_ids):
    arrays={k:original[k].copy() for k in old.ARRAY_KEYS}
    sources=[_source(s) for s in original['sources']]
    for i,source in enumerate(sources):
        if source['kind']!='student_oracle_unexecuted':continue
        proposal=rt.read_bound(source['bindings']['proposal'])
        _,_,relative,mask=old._oracle_query(proposal,source['metadata'],source['frozen_physics_row'])
        rows=np.flatnonzero(arrays['source_indices']==i)
        need(len(rows)==1,'One original actual student query per old oracle source required')
        arrays['relative_setters'][rows[0]]=relative;arrays['setter_is_pad'][rows[0]]=mask
    return bind_samples(arrays,sources,allowed_train_ids)


def validate_new_runs(auxiliary,original_metadata,original_samples,allowed_train_ids):
    """Bind appended accepted AND rejected slots to their completed diagnostics."""
    runs=auxiliary.get('provenance',{}).get('new_oracle_runs')
    need(isinstance(runs,list) and bool(runs),'Bound new source23 completed diagnostic runs required')
    need(auxiliary['request_refs']==original_metadata['request_refs']+[r['frozen_request'] for r in runs],
         'New frozen diagnostic request order differs')
    parts=[];ledger=[]
    for run in runs:
        need(set(run)=={'complete','frozen_request'},'Exact new completed diagnostic/request references required')
        frozen=rt.read_bound(run['frozen_request'])
        need(frozen.get('source_manifest')==SOURCE23_MANIFEST,'New diagnostic must retain original source23 code')
        imported=import_student_oracle(run['complete'],run['frozen_request'],allowed_train_ids=allowed_train_ids)
        parts.append(imported['samples']);ledger.extend(imported['slot_ledger'])
    expected=concatenate_samples(parts,allowed_train_ids)
    actual=auxiliary['samples'];offset=len(original_samples['source_indices']);sources=len(original_samples['sources'])
    tail={k:actual[k][offset:].copy() for k in old.ARRAY_KEYS};tail['source_indices']-=sources
    need(len(tail['source_indices'])==len(expected['source_indices']) and
         actual['sources'][sources:]==expected['sources'],'All new diagnostic accepted labels/sources must remain in order')
    prove_ordered_prefix(expected,tail,len(expected['source_indices']))
    need(auxiliary['slot_ledger'][len(original_metadata['slot_ledger']):]==ledger,
         'Every new accepted/rejected diagnostic outcome must match its frozen ledger')
