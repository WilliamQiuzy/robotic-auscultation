"""Separate TRAIN-only physically measured query-recovery supplement.

This never changes Base29 arrays/statistics or computational V5 admission. Four
frozen attempts, including every failure/missing/duplicate, own the denominator.
No trainer, model, optimizer, sampler or evaluation integration is provided.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from . import m2_given_contact_dataset as data, m2_given_contact_act as given
from . import m2_given_contact_recovery as recovery, m2_given_contact_recovery_dataset as roles
from . import m2_given_contact_runtime as rt, m2_query_recovery_prefix as query
from . import m2_stop_residual_trajectory as residual
from . import m2_teacher_stop_reader as teacher

need=rt.need
SCHEMA='m2_measured_query_recovery_bank_v1'
MANIFEST_SCHEMA='m2_measured_query_recovery_manifest_v1'
ADMISSION_SCHEMA='m2_measured_query_recovery_admission_v1'
SCHEMA_V2='m2_measured_query_recovery_bank_v2'
MANIFEST_SCHEMA_V2='m2_measured_query_recovery_manifest_v2'
ADMISSION_SCHEMA_V2='m2_measured_query_recovery_admission_v2'
PROFILE_V2='production_source32_teacher_compact_v2'
ARRAY_KEYS=rt.shared.ARRAY_KEYS+('action_is_pad',)
BASE29=dict(path='/opt/wam-runtime/fast-wam/datasets/act-student-recovery-masked-20261106-v1/COMPLETE.json',
    bytes=164297,sha256='265495e293b4ba95ca779cdc9693762716a3ea0010c6430ad2fde3b456f30cee')
BASE_COUNTS=dict(train=8373,validation=2345)


def admission(ledger,*,profile):
    need(profile in ('production_source30','local_synthetic_fixture',PROFILE_V2,'local_synthetic_fixture_v2'),
        'Explicit measured supplement profile required')
    order=isinstance(ledger,list) and [r.get('slot_id') for r in ledger]==list(query.SLOT_IDS)
    keys=[rt.digest(r['identity']) for r in ledger if r.get('identity') is not None]
    qualified=order and len(keys)==4 and len(set(keys))==4 and all(
        row.get('status')=='qualified' and type(row.get('windows')) is int and row['windows']>0 and
        row.get('target_id')==('R3' if i<2 else 'L3') for i,row in enumerate(ledger))
    return dict(schema=ADMISSION_SCHEMA_V2 if profile in (PROFILE_V2,'local_synthetic_fixture_v2') else ADMISSION_SCHEMA,
        measured_native_optimization_eligible=bool(qualified and profile in ('production_source30',PROFILE_V2)),
        fixed_attempt_denominator=4,all_four_required=True,population_generalization_evidence=False,
        new_body_diversity_evidence=False,formal_acceptance=False)


def array_proof(array):
    a=np.asarray(array)
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def code_binding():
    return dict(bank=rt.reference(__file__),capture_decoder=rt.reference(data.__file__),
        role_filter=rt.reference(roles.__file__),native_physical=rt.reference(residual.__file__),
        query_prefix=rt.reference(query.__file__),teacher_reader=rt.reference(teacher.__file__))


def validate_samples(samples):
    given.require_samples(samples)
    need(samples['schema']==given.MASKED_DATASET_SCHEMA and samples.get('label_protocol')==rt.shared.MASKED_LABEL_PROTOCOL,
        'Actual fixed-origin H2/H20/E10 native padding protocol required')
    n=len(samples['states'])
    shapes=dict(states=(n,17),histories=(n,2,17),actions=(n,20,7),origins=(n,7),velocities=(n,7),observation_times=(n,))
    for key,shape in shapes.items():rt.shared.contract._array(samples[key],shape,'measured '+key)
    need(np.array_equal(samples['states'],samples['histories'][:,1]) and
        np.array_equal(samples['origins'],samples['states'][:,:7]),'Actual current H2/fixed measured q7 origin differs')
    if 'derived_index' in samples:
        windows=samples['derived_index']['windows'];need(len(windows)==n,'Actual ordered window proof missing')
        need(all(w.get('label_role')=='actual_clean_recovery_teacher_raw_intents' and
            w.get('valid_future_cells')==int((~samples['action_is_pad'][i]).sum()) for i,w in enumerate(windows)),
            'Only positive actual clean teacher windows and exact masks required')
    return samples


def validate_replay_terminal(terminal,prefix,samples,metadata,*,captured_history=None):
    """Bind takeover to real captured H2, without requiring deterministic tracking."""
    actual=terminal.get('query',{});source=prefix['source_cut_query'];stamp=actual.get('time_s')
    need(terminal.get('kind')=='measured_query_replay_terminal' and terminal.get('saved_state_injected') is False and
        terminal.get('source_cut_query_sha256')==rt.digest(source) and
        terminal.get('l2_started_at_s')==prefix['l2_started_at_s'] and terminal.get('l2_deadline_s')==prefix['l2_deadline_s'] and
        actual.get('physics_row')==source['physics_row'] and type(stamp) in (int,float) and np.isfinite(stamp) and
        abs(stamp-source['time_s'])<=1e-8 and 0<terminal.get('remaining_l2_budget_s',0)==prefix['l2_deadline_s']-stamp and
        actual.get('metadata')==metadata and actual.get('reference_sha256')==metadata['reference_sha256'] and
        actual.get('context')==[metadata['episode_id'],metadata['target_id'],0,'APPROACH'],
        'Actual replay terminal/source/cumulative clock binding differs')
    if captured_history is None:
        anchors=np.flatnonzero(abs(np.asarray(samples['observation_times'])-stamp)<=1e-8)
        need(len(anchors)==1,'Actual captured replay-terminal teacher origin missing or duplicate')
        index=int(anchors[0]);window=samples['derived_index']['windows'][index]
        captured_history=dict(physics_row=window['inclusive_native_ticks'][0]+1,history=samples['histories'][index],
            measured_q=samples['origins'][index],history_times=[stamp-.05,stamp])
    need(captured_history.get('physics_row')==actual['physics_row'] and
        np.array_equal(captured_history.get('history'),actual.get('history')) and
        np.array_equal(captured_history.get('measured_q'),actual.get('measured_q')) and
        np.allclose(captured_history.get('history_times'),[stamp-.05,stamp],atol=1e-8,rtol=0) and
        np.allclose(actual.get('history_times'),[stamp-.05,stamp],atol=1e-8,rtol=0),
        'Replay terminal differs from actual fully captured causal H2/q7 origin')
    for key,out in (('measured_q','replay_minus_source_q'),('measured_dq','replay_minus_source_dq')):
        current=rt.shared.contract._array(actual.get(key),(7,),'actual replay '+key)
        original=rt.shared.contract._array(source.get(key),(7,),'source cut '+key)
        need(np.array_equal(current-original,terminal.get(out)),'Actual replay minus source '+key+' differs')
    current=rt.shared.contract._pose(actual.get('measured_pose_base'),'actual replay native pose')
    original=rt.shared.contract._pose(source.get('measured_pose_base'),'source cut native pose')
    need(np.array_equal(current-original,terminal.get('replay_minus_source_tcp_pose')),
        'Actual replay minus source native full TCP pose differs')


def _identity_rows(manifest):
    attempts=manifest.get('attempts')
    need(isinstance(attempts,list) and [a.get('slot_id') for a in attempts]==list(query.SLOT_IDS),
        'Exactly FOUR predeclared ordered attempts required; no replacement or denominator shrink')
    # Gate ALL source identities before opening even the first capture/NPZ.
    for i,attempt in enumerate(attempts):
        row=attempt.get('episode')
        if row is None:continue
        need(isinstance(row,dict) and row.get('split')=='train' and row.get('subject_id')=='p001120',
            'Measured recovery supplement is original TRAIN only; no VAL/fresh payloads')
        need(row.get('target_id')==('R3' if i<2 else 'L3') and row.get('pose_id')=='p000474_a001230_f0000',
            'Exact original TRAIN pose/target slots required')
    return attempts


def _base(manifest):
    need(manifest.get('base_dataset')==BASE29,'Original unchanged29 dataset reference required')
    base=rt.read_bound(BASE29)
    need(base.get('complete') is True and base.get('counts')==BASE_COUNTS and
        manifest.get('normalization')==base.get('normalization'),'Original29 counts/normalization reference differs')
    rt.validate_normalization(rt.read_bound(manifest['normalization']))
    return base


def _prefix_producer(ref):
    value=rt.read_bound(ref)
    need(value.get('schema')==query.BRIDGE_SCHEMA and value.get('complete') is True and value.get('fixed_denominator')==4,
        'Closed original four-slot query-prefix producer required')
    _,proof=query.validate_bridge(value['bridge'])
    need(value.get('slots')==proof['slots'] and set(value.get('prefixes',{}))==set(proof['prefixes']) and
        value.get('collection_eligible')==all(s['status']=='selected' for s in proof['slots']),
        'Immutable source selector ledger differs')
    for slot,ref in value['prefixes'].items():
        payload=rt.read_bound(ref)
        need(payload.get('bridge')==value['bridge'] and payload.get('payload')==proof['prefixes'][slot],
            'Prefix wrapper differs from frozen selector producer')
    return value


def _manifest_profile(manifest):
    """Version/schema/profile opt-in before opening any source payload."""
    keys={'schema','profile','attempts','prefix_producer','base_dataset','normalization'}
    if manifest.get('schema')==MANIFEST_SCHEMA_V2:
        need(set(manifest)==keys|{'evidence_profile','recorded_source'} and manifest.get('profile')==PROFILE_V2,
            'Explicit V2 teacher manifest/schema/evidence profile required')
        teacher.validate_profile(manifest.get('evidence_profile'),manifest.get('recorded_source'))
        return 2
    need(set(manifest)==keys and manifest.get('schema')==MANIFEST_SCHEMA and manifest.get('profile')=='production_source30',
        'Explicit production measured TRAIN supplement manifest required')
    return 1


def _teacher_context(row,base,*,evidence_profile,recorded_source):
    teacher.validate_profile(evidence_profile,recorded_source)
    need(row.get('run_id')=='act9-query-recovery-20261107-v1','Only original Source32 whole-owner run is approved')
    freeze,_=data._bound(row['source_freeze'],base)
    teacher.validate_source32_header(freeze,evidence_profile=evidence_profile,recorded_source=recorded_source)
    return dict(evidence_profile=evidence_profile,recorded_source=deepcopy(recorded_source),source_freeze=freeze,run_id=row['run_id'],
        source_declaration=deepcopy(row),evidence_base=str(Path(base).resolve()))


def _headers(manifest,base):
    version=_manifest_profile(manifest);attempts=_identity_rows(manifest)
    _base(manifest);producer=_prefix_producer(manifest['prefix_producer'])
    # Duplicate declarations remain failed attempts, never silently removed.
    errors={};seen={}
    for i,attempt in enumerate(attempts):
        row=attempt.get('episode');slot=producer['slots'][i]
        if row is None:continue
        try:
            need(slot['status']=='selected','Missing/duplicate source selector is not collectable')
            if version==2:
                _teacher_context(row,base,evidence_profile=manifest['evidence_profile'],recorded_source=manifest['recorded_source'])
            config=rt.read_bound(row['config']);binding=recovery.preflight(config)
            need(binding['settings']['schema']==recovery.QUERY_SCHEMA and
                binding['settings']['student_prefix']==producer['prefixes'][attempt['slot_id']],
                'Exact typed source query-prefix config required')
            payload=binding['student_prefix'];need(payload['slot_id']==attempt['slot_id'],'Attempt changed frozen selector')
            tokens=(row.get('episode_id'),row.get('reset_id'),rt.digest(row.get('run_result')),row.get('capture_root'))
            need(all(token not in seen.setdefault(k,set()) for k,token in enumerate(tokens)),
                'Duplicate native recovery episode/reset/source/capture declaration')
            for k,token in enumerate(tokens):seen[k].add(token)
            data._declarations([row],base);data._metadata(row,base)
        except (ValueError,KeyError,TypeError,OSError) as error:errors[attempt['slot_id']]=type(error).__name__+': '+str(error)[:2000]
    return attempts,producer,errors


def read_episode(row,base):
    """V1 strict native observer route; compact teacher traces never fallback."""
    return _read_episode(row,base)


def read_teacher_episode(row,base,*,evidence_profile,recorded_source):
    """V2 only: original owner/source/capture gates plus fixed teacher proof."""
    need(row.get('split')=='train' and row.get('subject_id')=='p001120','Measured labels are original TRAIN only')
    context=_teacher_context(row,base,evidence_profile=evidence_profile,recorded_source=recorded_source)
    return _read_episode(row,base,teacher_context=context)


def _read_episode(row,base,*,teacher_context=None):
    """Full original Native physical/capture/process/source validation, typed only."""
    need(row.get('split')=='train' and row.get('subject_id')=='p001120','Measured labels are original TRAIN only')
    data._declarations([row],base);context=data._metadata(row,base)
    run,config,freeze=data._native_evidence(row,base,context[2],physical=False)
    if teacher_context is not None:
        need(freeze==teacher_context['source_freeze'],'Authenticated original source freeze changed')
    binding=recovery.preflight(config)
    need(binding['settings']['schema']==recovery.QUERY_SCHEMA,'Explicit measured query-recovery route required')
    residual.native_options(config,metadata=context[2])
    need(residual.CONFIG_KEY in config,'Explicit native STOP route required')
    child,_=data._bound(row['child_result'],base);rt.shared._require_run_identity(run,row)
    need(child.get('physical_pass') is True and run.get('success') is True and run.get('physical_validation') is True and
        run.get('errors')==[] and run.get('controller_motion_complete') is True and run.get('compact_capture_complete') is True and
        run.get('executed_controller_class')==recovery.__name__+'.GivenContactRecoveryExpert' and
        run.get('learned_policy_executed') is False and run.get('actual_student_command_prefix_replayed') is True,
        'Complete physically passing actual command-replay/recovery teacher required')
    ref=run.get('recovery_collection');sidecar=rt.read_bound(ref)
    roles.require_sidecar_location(ref,data._ref_path(row['run_result'],base).parent)
    need(sidecar.get('config_sha256')==rt.digest(config) and sidecar.get('declaration')==binding['settings'],
        'Actual recovery sidecar/exact config binding differs')
    terminal=[q for q in sidecar.get('queries',[]) if q.get('kind')=='measured_query_replay_terminal']
    need(len(terminal)==1 and terminal[0].get('saved_state_injected') is False and
        terminal[0].get('source_cut_query_sha256')==rt.digest(binding['student_prefix']['source_cut_query']) and
        terminal[0].get('l2_started_at_s')==binding['student_prefix']['l2_started_at_s'] and
        terminal[0].get('l2_deadline_s')==binding['student_prefix']['l2_deadline_s'] and
        terminal[0].get('remaining_l2_budget_s',0)>0,'New measured replay terminal and original clock evidence required')
    if teacher_context is None:
        samples,receipt,meta=data.decode_native_query_recovery_capture(row,base,context,run,config,freeze,
            replay_cut_row=binding['student_prefix']['end_row_exclusive'])
    else:
        samples,receipt,meta=data.decode_teacher_query_recovery_capture(row,base,context,run,config,freeze,
            teacher_context=teacher_context,replay_cut_row=binding['student_prefix']['end_row_exclusive'])
    outer=rt.shared._json(context[0]/'receipt.json')
    audit=data.read_native_capture_audit(context[0],outer,transport=row['capture_audit'])
    samples=roles.filter_recovery_samples(samples,audit,meta,sidecar);validate_samples(samples)
    need(len(samples['states'])>0,'Positive completed clean recovery teacher future windows required')
    validate_replay_terminal(terminal[0],binding['student_prefix'],samples,meta,captured_history=receipt['replay_cut_observations'])
    receipt.update(windows=len(samples['states']),recovery_collection=ref,query_prefix=binding['settings']['student_prefix'],
        capture_audit=deepcopy(row['capture_audit']),physical_verifier=rt.reference(residual.__file__),
        decoder=rt.reference(data.__file__),producer_sources=deepcopy(freeze),
        measured_replay_terminal=deepcopy(terminal[0]),label_protocol=rt.shared.MASKED_LABEL_PROTOCOL)
    if teacher_context is not None:
        receipt.update(evidence_profile=teacher_context['evidence_profile'],recorded_source=deepcopy(teacher_context['recorded_source']),
            physical_verifier=rt.reference(teacher.__file__))
    return samples,receipt,meta


def export_dataset(manifest_ref,output):
    manifest=rt.read_bound(manifest_ref);_identity_rows(manifest)
    base=Path(manifest_ref['path']).parent;attempts,producer,errors=_headers(manifest,base)
    output=Path(output).resolve();need(not output.exists(),'Fresh separate measured supplement directory required');output.mkdir(parents=True)
    arrays=[];ledger=[];sources=[];windows=[];offset=0;resets=set()
    for attempt,slot in zip(attempts,producer['slots']):
        entry=deepcopy(slot);entry.update(status=slot['status'] if slot['status']!='selected' else 'missing',windows=0,
            episode=deepcopy(attempt.get('episode')))
        row=attempt.get('episode')
        if slot['status']=='selected' and row is not None:
            try:
                need(attempt['slot_id'] not in errors,errors.get(attempt['slot_id'],''))
                if manifest['schema']==MANIFEST_SCHEMA_V2:
                    sample,receipt,meta=read_teacher_episode(row,base,evidence_profile=manifest['evidence_profile'],recorded_source=manifest['recorded_source'])
                else:sample,receipt,meta=read_episode(row,base)
                validate_samples(sample)
                identity=receipt['native_reset_identity'];reset=(identity['pid'],identity['sid'],identity['start_ticks'])
                need(reset not in resets,'Native process reused across recovery resets');resets.add(reset)
                count=len(sample['states']);need(count>0,'Empty positive recovery windows')
                sources.append(dict(slot_id=slot['slot_id'],metadata=deepcopy(meta),declaration=deepcopy(row),
                    receipt=deepcopy(receipt),split_sample_range=[offset,offset+count],windows=count))
                windows.extend(dict(slot_id=slot['slot_id'],sample_index=offset+i,window=deepcopy(w))
                    for i,w in enumerate(sample['derived_index']['windows']))
                arrays.append(sample);offset+=count;entry.update(status='qualified',windows=count,receipt=receipt)
            except Exception as error:entry.update(status='failed',error=type(error).__name__+': '+str(error)[:2000])
        ledger.append(entry)
    refs={};proofs={}
    if arrays:
        samples={key:np.concatenate([a[key] for a in arrays]) for key in ARRAY_KEYS}
        path=output/'TRAIN_MEASURED.npz'
        with path.open('xb') as stream:np.savez_compressed(stream,**samples)
        refs['arrays']=rt.reference(path);proofs={k:array_proof(v) for k,v in samples.items()}
    qualified=admission(ledger,profile=manifest['profile'])
    index=rt.write_new(output/'INDEX.json',dict(samples=windows,order_sha256=rt.digest(windows),label_protocol=rt.shared.MASKED_LABEL_PROTOCOL))
    bank_schema=SCHEMA_V2 if manifest['schema']==MANIFEST_SCHEMA_V2 else SCHEMA
    extra=({k:deepcopy(manifest[k]) for k in ('evidence_profile','recorded_source')} if bank_schema==SCHEMA_V2 else {})
    metadata=rt.write_new(output/'METADATA.json',dict(schema=bank_schema,profile=manifest['profile'],source_manifest=manifest_ref,
        base_dataset=manifest['base_dataset'],base_counts=BASE_COUNTS,normalization=manifest['normalization'],normalization_refitted=False,
        prefix_producer=manifest['prefix_producer'],attempt_ledger=ledger,fixed_denominator=4,sources=sources,windows=offset,
        arrays=refs.get('arrays'),array_proofs=proofs,index=index,optimization_admission=qualified,data_code=code_binding(),
        actions_executed=bool(arrays),physics_qualified=bool(arrays),native_eligible=qualified['measured_native_optimization_eligible'],
        label_protocol=rt.shared.MASKED_LABEL_PROTOCOL,training_started=False,model_qualified=False,
        validation_payloads_opened=False,fresh_payloads_opened=False,base_arrays_modified=False,**extra))
    return rt.write_new(output/'COMPLETE.json',dict(schema=bank_schema,complete=True,profile=manifest['profile'],metadata=metadata,
        optimization_admission=qualified,arrays=refs.get('arrays'),normalization=manifest['normalization'],windows=offset,**extra))


def metadata(complete_ref):
    """All source identities/sealed refs before NPZ; no historical model replay."""
    complete=rt.read_bound(complete_ref)
    bank_schema=complete.get('schema')
    need(bank_schema in (SCHEMA,SCHEMA_V2) and complete.get('complete') is True and
        complete.get('profile')==('production_source30' if bank_schema==SCHEMA else PROFILE_V2),
        'Explicit complete production measured supplement required')
    meta=rt.read_bound(complete['metadata']);manifest=rt.read_bound(meta['source_manifest']);_identity_rows(manifest)
    attempts,producer,errors=_headers(manifest,Path(meta['source_manifest']['path']).parent)
    need(bank_schema==(SCHEMA_V2 if manifest['schema']==MANIFEST_SCHEMA_V2 else SCHEMA),'No measured bank schema promotion allowed')
    if bank_schema==SCHEMA_V2:
        need(all(complete.get(k)==meta.get(k)==manifest[k] for k in ('evidence_profile','recorded_source')),
            'Explicit V2 source/evidence profile bindings differ')
    need(meta.get('schema')==bank_schema and meta.get('profile')==manifest['profile'] and meta.get('data_code')==code_binding() and
        meta.get('normalization')==manifest['normalization']==complete.get('normalization') and
        meta.get('normalization_refitted') is False and meta.get('base_dataset')==BASE29 and meta.get('base_counts')==BASE_COUNTS and
        meta.get('prefix_producer')==manifest['prefix_producer'] and meta.get('fixed_denominator')==4,
        'Measured source/code/unchanged29 normalization binding differs')
    ledger=meta.get('attempt_ledger');expected_admission=admission(ledger,profile=manifest['profile'])
    need(meta.get('optimization_admission')==complete.get('optimization_admission')==expected_admission and
        meta.get('native_eligible')==expected_admission['measured_native_optimization_eligible'] and
        meta.get('arrays')==complete.get('arrays') and meta.get('windows')==complete.get('windows'),
        'Measured bank admission/counts differ')
    for slot,attempt,entry in zip(producer['slots'],attempts,ledger):
        need(all(entry.get(k)==slot[k] for k in ('slot_id','target_id','physics_row','identity','source')) and
            entry.get('episode')==attempt.get('episode') and
            (entry.get('status')==slot['status'] if slot['status']!='selected' else entry.get('status') in ('qualified','failed','missing')),
            'Complete actual attempt ledger changed')
        if entry.get('status')=='qualified':need(entry['slot_id'] not in errors,'Qualified entry has invalid metadata')
    cursor=0
    for source in meta.get('sources',[]):
        m=source.get('metadata',{});need(m.get('split')=='train' and m.get('subject_id')=='p001120','All measured sources must be original TRAIN')
        row=source['declaration'];need(m.get('episode_id')==row['episode_id'] and m.get('target_id')==row['target_id'],
            'Measured source identity/target differs')
        if bank_schema==SCHEMA_V2:
            need(all(source['receipt'].get(k)==manifest[k] for k in ('evidence_profile','recorded_source')) and
                source['receipt'].get('physical_verifier')==rt.reference(teacher.__file__),
                'Qualified V2 receipt must bind the fixed teacher reader/source profile')
        entries=[e for e in ledger if e['slot_id']==source['slot_id']]
        need(len(entries)==1 and entries[0]['status']=='qualified' and entries[0]['episode']==row and entries[0]['receipt']==source['receipt'] and
            source['split_sample_range']==[cursor,cursor+source['windows']] and source['windows']==entries[0]['windows'],
            'Exact qualified source receipt/order required');cursor+=source['windows']
    need(cursor==meta['windows'] and len(meta.get('sources',[]))==sum(e['status']=='qualified' for e in ledger),
        'No omitted or duplicate positive sources')
    index=rt.read_bound(meta['index']);need(index.get('order_sha256')==rt.digest(index.get('samples')) and
        len(index['samples'])==cursor and [w['sample_index'] for w in index['samples']]==list(range(cursor)),
        'Actual sample order/mask index differs')
    return complete,meta


def load_arrays(complete_ref):
    complete,meta=metadata(complete_ref)
    need(complete['optimization_admission']['measured_native_optimization_eligible'] is True,
        'All FOUR actual distinct qualified attempts required before optimization arrays')
    with np.load(rt.bound_path(complete['arrays']),allow_pickle=False) as archive:
        need(set(archive.files)==set(ARRAY_KEYS),'Exact measured array fields required')
        arrays={key:archive[key].copy() for key in ARRAY_KEYS}
    need({k:array_proof(v) for k,v in arrays.items()}==meta['array_proofs'],'Measured ordered array/mask bytes differ')
    index=rt.read_bound(meta['index'])
    arrays.update(schema=given.MASKED_DATASET_SCHEMA,task_id=rt.pose.TASK_ID,action_semantics_id=given.ACTION_SEMANTICS_ID,
        observation_contract=rt.pose.observation_contract(),label_protocol=rt.shared.MASKED_LABEL_PROTOCOL,
        measured_bank=complete_ref,normalization_reference=meta['normalization'],bank_schema=meta['schema'],
        optimization_admission=deepcopy(complete['optimization_admission']),
        derived_index=dict(windows=[w['window'] for w in index['samples']]))
    validate_samples(arrays)
    need(len(arrays['states'])==meta['windows'],'Measured sample count differs')
    return arrays,meta


def pack_batch(samples,normalization_ref,indices):
    need(samples.get('bank_schema') in (SCHEMA,SCHEMA_V2) and
        samples.get('optimization_admission',{}).get('schema')==(ADMISSION_SCHEMA_V2 if samples['bank_schema']==SCHEMA_V2 else ADMISSION_SCHEMA) and
        samples.get('optimization_admission',{}).get('measured_native_optimization_eligible') is True,
        'Explicit complete measured bank optimization admission required')
    need(samples.get('normalization_reference')==normalization_ref,'Only original29 normalization reference may pack measured samples')
    stats=rt.validate_normalization(rt.read_bound(normalization_ref));validate_samples(samples)
    ids=np.asarray(indices)
    need(ids.ndim==1 and ids.dtype.kind in 'iu' and np.all(ids>=0) and np.all(ids<len(samples['states'])),
        'Valid measured sample indices required')
    return given.pack_batch(samples,stats,ids,training=False)
