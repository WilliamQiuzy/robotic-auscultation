"""Data-only V4 terminal-query admission with an immutable complete ACT5 parent.

Actual causal H2 is measured once. Future raw setter cells remain unexecuted
computational labels. This module neither fits normalization nor dispatches a
trainer, model, native capture, GPU owner, or fresh evaluation.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import m2_act_auxiliary_data as old, m2_act_frozen26_data as frozen26
from . import m2_act_recovery_scope as scope, m2_act_chunk_data as v2

rt = old.rt
need = rt.need
ARRAY_KEYS = old.ARRAY_KEYS
SCHEMA = 'm2_act_auxiliary_preparation_v4'
SOURCE_SCHEMA = 'm2_act_auxiliary_source_v4'
REQUEST_SCHEMA = 'm2_act_terminal_request_v4'
SEMANTICS = ('at_or_before_60s', 'at_or_before_90s', 'last_eligible')
PARENT_COMPLETE = dict(path='/opt/wam-runtime/fast-wam/datasets/act-conditional-full-chunks-20261107-v1/COMPLETE.json',
    bytes=1202, sha256='d84afc9f705f25de34bc2aeb253ee79b43a52981ad6fdc57e3f19265da47b075')
PARENT_ROWS, PARENT_SOURCES, PARENT_SLOTS = 35982, 934, 1109
TRAIN16_SCOPE = dict(path='/opt/wam-runtime/fast-wam/diagnostics/act-auxiliary-cohort-20261106-v2/COHORT_FREEZE.json',
    bytes=1521, sha256='adf6b0bd95669719989f6795c177f0a8419921dd79a257ca0c725f99b3220e71')
ADMISSION = dict(schema='m2_act_terminal_computational_admission_v4', computational_optimization_eligible=True,
    oracle_labels='bound_full_computed_prefix_up_to20', future_measured_histories=False, **old.FLAGS)


def _identity_gate(sources, allowed):
    need(isinstance(sources, list) and bool(allowed) and set(allowed) <= set(scope.TRAIN16_IDS), 'Explicit TRAIN16 sources/IDs required')
    for source in sources:
        old._train(source.get('subject_id'), source.get('split'), allowed)
        meta = source.get('metadata', {})
        old._train(meta.get('subject_id'), meta.get('split'), allowed)
        need(source['subject_id'] == meta['subject_id'], 'TRAIN identity metadata differs')


def assert_ordered_parent(parent, candidate):
    """Storage bit equality, including signed zero, plus all historical ledgers."""
    before, after = parent['samples'], candidate['samples']
    count = len(before['source_indices']); proof = {}
    for key in ARRAY_KEYS:
        proof[key] = frozen26.array_proof(before[key])
        need(frozen26.array_proof(after[key][:count]) == proof[key], 'Exact ordered parent array changed: '+key)
    need(after['sources'][:len(before['sources'])] == before['sources'], 'Exact ordered parent sources/source IDs changed')
    for key in ('slot_ledger', 'request_refs'):
        need(candidate[key][:len(parent[key])] == parent[key], 'Exact ordered parent '+key+' changed')
    return proof


def _terminal_request(request, allowed):
    need(request.get('schema') == REQUEST_SCHEMA and set(request)=={'schema','cases','slots','oracle_request_refs'}|set(old.FLAGS),
        'Exact explicit V4 semantic terminal request fields required')
    old._flags(request); slots = request.get('slots')
    need(isinstance(slots, list) and len(slots) in (3, 6), 'Three semantic slots per failed TRAIN target required')
    seen = set(); groups = {}
    for slot in slots:
        old._train(slot.get('subject_id'), slot.get('split'), allowed)
        need(slot.get('subject_id') == 'p001120' and slot.get('target_id') in ('R3', 'L3'), 'Original TRAIN terminal targets required')
        need(isinstance(slot.get('slot_id'), str) and slot['slot_id'] and slot['slot_id'] not in seen and
            isinstance(slot.get('job'), str) and slot['job'] and isinstance(slot.get('pose_id'), str) and slot['pose_id'],
            'Unique named semantic terminal slots required')
        row = slot.get('physics_row')
        need(row is None or (type(row) is int and row >= 12 and row % 12 == 0), 'Actual eligible query row or explicit missing slot required')
        limit = {'at_or_before_60s':14400,'at_or_before_90s':21600}.get(slot.get('semantic'))
        need(row is None or limit is None or row<=limit, 'Frozen terminal time slot exceeds its declared native time')
        seen.add(slot['slot_id']); key = (slot['subject_id'],slot['pose_id'],slot['target_id'],slot['job'])
        groups.setdefault(key, []).append(slot.get('semantic'))
    need(len(groups) <= 2 and len({k[2] for k in groups}) == len(groups) and
        all(tuple(v) == SEMANTICS for v in groups.values()), 'Exact ordered 60s/90s/last request triplet per target required')
    cases = request.get('cases')
    need(isinstance(cases,list) and len(cases)==len(groups), 'Every failed target including all-missing cases must survive')
    seen_cases = set()
    for case in cases:
        need(set(case)=={'job','subject_id','split','pose_id','target_id','source'} and
            isinstance(case['source'],dict) and set(case['source'])=={'config','owner','child','native'}, 'Exact failed original case/source fields required')
        old._train(case.get('subject_id'),case.get('split'),allowed)
        key = (case['subject_id'],case['pose_id'],case['target_id'],case['job'])
        need(key in groups and key not in seen_cases, 'Every semantic target must bind its own failed original source exactly once')
        seen_cases.add(key)
    return slots


def validate_terminal_ledger(request, ledger, imported_ledger, allowed_train_ids):
    slots = _terminal_request(request, allowed_train_ids)
    need(isinstance(ledger, list) and len(ledger) == len(slots), 'Every semantic attempted slot must survive in order')
    unique = {}; imported = {}
    for row in imported_ledger:
        key = row['slot_id']
        need(key not in imported, 'Unique original oracle rows required'); imported[key] = row
    for slot, result in zip(slots, ledger):
        old._flags(result)
        need(all(result.get(k) == v for k,v in slot.items()), 'Frozen semantic slot changed')
        row = slot['physics_row']; key = (slot['job'],row)
        if row is None:
            need(result.get('status') == 'missing' and isinstance(result.get('rejection'), str) and result['rejection'] and
                'source_id' not in result, 'Missing actual query must remain missing with its reason')
        elif key in unique:
            need(result.get('status') == 'duplicate' and result.get('duplicate_of') == unique[key] and
                'source_id' not in result, 'Duplicate semantic slot must not duplicate source/weight')
        else:
            unique[key] = slot['slot_id']; original = imported.get('oracle:'+slot['job']+':'+str(row))
            need(original is not None and all(result.get(k) == original.get(k) for k in ('subject_id','pose_id','target_id','status')),
                'Semantic slot differs from original unique oracle outcome')
            if original['status'] == 'accepted':
                need(result.get('source_id') == original['source_id'], 'Accepted semantic source differs')
            else:
                need(result.get('rejection') == original.get('rejection') and 'source_id' not in result,
                    'Original oracle rejection/abstention reason must survive')
    need(set(imported) == {'oracle:'+job+':'+str(row) for job,row in unique}, 'No omitted or extra unique actual query allowed')
    return ledger


def _run_headers(runs, allowed):
    """Inspect EVERY case/config identity before any native/proposal/archive read."""
    need(bool(allowed) and set(allowed)<=set(scope.TRAIN16_IDS), 'Original TRAIN16 allowed identity pool required')
    need(isinstance(runs, list), 'Explicit original source26 oracle runs required')
    checked = []
    for run in runs:
        need(set(run) == {'complete', 'frozen_request'}, 'Exact original aggregate complete/frozen request refs required')
        root = rt.read_bound(run['complete']); frozen = rt.read_bound(run['frozen_request'])
        cases, results = frozen.get('cases'), root.get('results')
        need(isinstance(cases,list) and cases and isinstance(results,list) and len(cases)==len(results),
            'Original source26 request/result case inventory required')
        pending = []
        for case, result in zip(cases, results):
            per = rt.read_bound(result['result']); directory = Path(result['result']['path']).parent
            request_ref = rt.reference(directory/'REQUEST.json'); request = rt.read_bound(request_ref)
            metas = []
            for source in (case.get('source',{}),request.get('source',{})):
                need(isinstance(source,dict) and set(source)=={'config','owner','child','native'}, 'Original student source refs required')
                config = rt.read_bound(source['config'])
                meta = config.get('m2_given_contact_expanded_pilot_deployment',{}).get('metadata',{})
                old._train(meta.get('subject_id'),meta.get('split'),allowed); metas.append(meta)
            old._train(per.get('source_subject_id'),per.get('source_split'),allowed)
            need(metas[0] == metas[1] and per['source_subject_id']==metas[0]['subject_id'], 'TRAIN source config metadata differs')
            pending.append(dict(request=request,request_ref=request_ref,metadata=metas[0],complete=per,case=case))
        checked.append(dict(run=run,root=root,frozen=frozen,pending=pending))
    return checked


def _closed_owner(ref):
    owner = rt.read_bound(ref); stage = owner.get('stage', {})
    need(owner.get('completed') is True and owner.get('gpu_released') is True and stage.get('passed') is True and
        type(stage.get('exit_code')) is int and stage['exit_code']==0 and stage.get('process_group_drained') is True and
        all(stage.get(k)==[] for k in ('errors','signals','deferred_signals')),
        'Whole original owner must close normally, drain and release before oracle payloads')
    return owner


def _preflight_runs(runs, allowed):
    checked = _run_headers(runs,allowed)
    for header in checked:
        need(header['frozen'].get('source_manifest')==frozen26.SOURCE26_MANIFEST, 'Pinned original source26 generator archive required')
    if checked:
        files = frozen26.require_archive(); generator = files['sim/ausc_policy/m2_student_oracle_proposals.py']
        for header in checked:
            for item in header['pending']:
                need(item['request'].get('generator')==generator, 'Generator must derive from full original source26 manifest')
                _closed_owner(item['request']['source']['owner'])
        # The unchanged generic preflight retains all original full geometry,
        # raw/source/request/outcome bindings after cohort-wide identity gating.
        for header in checked:
            old._oracle_preflight(header['run']['complete'],header['run']['frozen_request'],allowed)
    return checked


def _new_source_headers(sources, allowed):
    _identity_gate(sources,allowed)
    # A forged source table cannot conceal a VAL/fresh source config. All
    # configs pass before geometry, actual native refs, proposal JSON or NPZ.
    requests = []
    for source in sources:
        refs = source.get('bindings',{}); request = rt.read_bound(refs['request'])
        config = rt.read_bound(request['source']['config'])
        meta = config.get('m2_given_contact_expanded_pilot_deployment',{}).get('metadata',{})
        old._train(meta.get('subject_id'),meta.get('split'),allowed)
        need(meta==source['metadata'], 'New source metadata differs from original actual source config')
        requests.append(request)
    for source, request in zip(sources,requests):
        refs = source['bindings']; old._flags(source)
        need(source.get('schema')==SOURCE_SCHEMA and source.get('kind')=='student_oracle_unexecuted' and
            source.get('observation_origin')==old.ORIGINS['student_oracle_unexecuted'] and
            source.get('future_measured_histories') is False and source.get('generated_static_input') is False and
            source.get('source_episode_verdict_unchanged') is True and source.get('original_source_success') is False,
            'Explicit genuine-H2/unexecuted computational V4 source required')
        need(refs.get('frozen_manifest')==frozen26.SOURCE26_MANIFEST and
            refs.get('generator')==request.get('generator'), 'Original source26 manifest/generator binding required')
        for key in ('config','owner','child','native'):
            need(refs.get('source' if key=='config' else key)==request['source'][key], 'Original source owner/child/native binding differs')
        _closed_owner(refs['owner'])
    if sources:
        generator = frozen26.generator_reference()
        need(all(s['bindings']['generator']==generator for s in sources), 'Fully frozen source26 generator required')
        old._sources(sources,allowed,source_schema=SOURCE_SCHEMA)
    return requests


def authenticate_new_genuine_groups(auxiliary, allowed_train_ids):
    """Full source26 authentication for new queries, never a stale export seal."""
    samples = auxiliary['samples']; sources = auxiliary['sources']
    _identity_gate(sources,allowed_train_ids)
    selected = [(i,s) for i,s in enumerate(sources) if s.get('schema')==SOURCE_SCHEMA]
    requests = _new_source_headers([s for _,s in selected],allowed_train_ids)
    groups = {}
    for (index,source),request in zip(selected,requests):
        key = rt.digest(request['source'])
        group = groups.setdefault(key,dict(source=request['source'],rows=[],proposals={},members=[]))
        row = source['frozen_physics_row']
        need(row not in group['rows'], 'Unique actual query may contribute only one V4 source')
        group['rows'].append(row); group['proposals'][str(row)] = source['bindings']['proposal']; group['members'].append((index,source))
    ordered = list(groups.values())
    result = frozen26.authenticate_queries([{k:g[k] for k in ('source','rows','proposals')} for g in ordered]) if ordered else []
    need(len(result)==len(ordered), 'Complete original26 genuine authentication group inventory required')
    receipts = []
    for group,records in zip(ordered,result):
        need(len(records)==len(group['members']), 'Complete original26 genuine query inventory required')
        for (index,source),record in zip(group['members'],records):
            proposal = rt.read_bound(source['bindings']['proposal'])
            need(record.get('proposal_authenticated_sha256')==source['bindings']['proposal']['sha256'],
                'Original26 full computed prefix/seam authentication receipt differs')
            v2.validate_oracle_sample(samples,index,proposal,record['query'],snapshot=record['snapshot'])
        receipts.append(dict(source=group['source'],physics_rows=group['rows'],manifest=frozen26.SOURCE26_MANIFEST,
            actual_history_authenticated=True,retained_command_seam_authenticated=True,future_measurements_claimed=False))
    return receipts


def import_student_oracle(complete_ref, frozen_request_ref, *, allowed_train_ids):
    checked = _preflight_runs([dict(complete=complete_ref,frozen_request=frozen_request_ref)],allowed_train_ids)
    # Authenticate rejected actual queries too; they remain attempts, not labels.
    groups = []
    for item in checked[0]['pending']:
        groups.append(dict(source=item['request']['source'],rows=item['request']['physics_rows'],
            proposals={str(o['physics_row']):o['computational_proposal'] for o in item['complete']['outcomes']
                if 'computational_proposal' in o}))
    records = frozen26.authenticate_queries(groups)
    need(len(records)==len(groups) and all(len(r)==len(g['rows']) for r,g in zip(records,groups)),
        'All new eligible actual query rows require original26 authentication')
    imported = old.import_student_oracle(complete_ref,frozen_request_ref,allowed_train_ids=allowed_train_ids)
    original = imported['samples']; arrays = {k:original[k].copy() for k in ARRAY_KEYS}; sources=[]
    authenticated = {(rt.digest(g['source']),row):record for g,values in zip(groups,records) for row,record in zip(g['rows'],values)}
    for index,source in enumerate(original['sources']):
        source = deepcopy(source); source.pop('source_id'); source.pop('sample_sha256'); source['schema']=SOURCE_SCHEMA
        source.update(supervision='full_computed_oracle_prefix_up_to20; genuine H2 only; future unexecuted',
            future_measured_histories=False,generated_static_input=False)
        proposal = rt.read_bound(source['bindings']['proposal'])
        _,_,relative,mask = old._oracle_query(proposal,source['metadata'],source['frozen_physics_row'])
        row = int(np.flatnonzero(arrays['source_indices']==index)[0])
        arrays['relative_setters'][row] = relative; arrays['setter_is_pad'][row] = mask
        source['sample_sha256'] = old._row_digest(arrays,[row]); source['source_id'] = old._source_id(source); sources.append(source)
    samples = dict(schema=SCHEMA,sources=sources,source_table_sha256=rt.digest(sources),allowed_train_ids=sorted(allowed_train_ids),
        optimization_admission=deepcopy(ADMISSION),**old.FLAGS,**arrays)
    _new_source_headers(sources,allowed_train_ids); old._numeric(samples,sources,full_oracle_prefix=True)
    for index,source in enumerate(sources):
        request = rt.read_bound(source['bindings']['request']); record = authenticated[(rt.digest(request['source']),source['frozen_physics_row'])]
        need(record.get('proposal_authenticated_sha256')==source['bindings']['proposal']['sha256'], 'Original26 computed proposal receipt differs')
        v2.validate_oracle_sample(samples,index,rt.read_bound(source['bindings']['proposal']),record['query'],snapshot=record['snapshot'])
    mapping = {a['source_id']:b['source_id'] for a,b in zip(original['sources'],sources)}
    ledger = deepcopy(imported['slot_ledger'])
    for slot in ledger:
        if slot['status']=='accepted': slot['source_id']=mapping[slot['source_id']]
    return dict(samples=samples,sources=sources,slot_ledger=ledger)


def pack_batch(samples, real_native_stats, indices):
    """Same native normalization and H2 layout; never refit or replay archives."""
    need(samples.get('schema')==SCHEMA and samples.get('optimization_admission')==ADMISSION, 'Explicit V4 samples required')
    stats = rt.validate_normalization(real_native_stats); ids = np.asarray(indices)
    need(ids.ndim==1 and ids.dtype.kind in 'iu' and np.all(ids>=0) and np.all(ids<len(samples['source_indices'])), 'Valid V4 batch indices required')
    h = (samples['source_histories'][ids]-np.asarray(stats['state_mean']))/np.asarray(stats['state_std'])
    mask = samples['setter_is_pad'][ids].copy()
    raw = np.where(mask[...,None],0.,samples['relative_setters'][ids])
    a = (raw-np.asarray(stats['action_mean']))/np.asarray(stats['action_std'])
    need(np.isfinite(h).all() and np.isfinite(a).all(), 'Finite valid V4 packed data required')
    return {'observation.state':h[:,1,:7].astype(np.float32),
        'observation.environment_state':np.concatenate([h[:,1,7:],h[:,0]],axis=1).astype(np.float32),
        'action':a.astype(np.float32),'action_is_pad':mask}


def code_binding():
    """Bind the modules actually loaded by the eventual numerical source freeze."""
    return dict(schema='m2_act_terminal_data_code_v4',files=dict(data=rt.reference(__file__),
        source26_bridge=rt.reference(frozen26.__file__),generic_import=rt.reference(old.__file__),
        numeric_protocol=rt.reference(v2.__file__)))


def _scope(ref, allowed):
    need(ref==TRAIN16_SCOPE, 'Pinned original metadata-only TRAIN16 scope required')
    value = rt.read_bound(ref); cohorts = scope.validate_train16_metadata(value)
    need(bool(allowed) and set(allowed)<=set(cohorts['train']), 'Explicit allowed TRAIN16 identities required')


def _parent_header(allowed):
    complete = rt.read_bound(PARENT_COMPLETE); meta = rt.read_bound(complete['metadata'])
    _identity_gate(meta['sources'],allowed)
    need(complete.get('schema')=='m2_act_auxiliary_preparation_v3' and complete.get('complete') is True and
        complete.get('metadata_sha256')==complete['metadata']['sha256'], 'Pinned complete ACT5 V3 parent required')
    need(len(meta['sources'])==PARENT_SOURCES and len(meta['slot_ledger'])==PARENT_SLOTS and
        sum(len(s['windows']) for s in meta['sources'])==PARENT_ROWS, 'Entire ACT5 parent row/source/attempt inventory required')
    return complete,meta


def _hash_references(value):
    if isinstance(value,dict):
        path = value.get('path',value.get('file'))
        if isinstance(path,str) and 'sha256' in value:
            ref = dict(path=path,sha256=value['sha256'])
            if 'bytes' in value: ref['bytes']=value['bytes']
            rt.bound_path(ref)
        for child in value.values(): _hash_references(child)
    elif isinstance(value,list):
        for child in value: _hash_references(child)


def load_parent(allowed_train_ids):
    _parent_header(allowed_train_ids)
    parent = frozen26.load_parent(PARENT_COMPLETE,allowed_train_ids)
    _identity_gate(parent['sources'],allowed_train_ids)
    need(len(parent['samples']['source_indices'])==PARENT_ROWS and len(parent['sources'])==PARENT_SOURCES and
        len(parent['slot_ledger'])==PARENT_SLOTS, 'Entire original26-validated ACT5 parent required')
    validation = parent.get('archive_validation',{})
    need(validation.get('manifest')==frozen26.SOURCE26_MANIFEST and validation.get('code_commit')==frozen26.SOURCE26_COMMIT and
        validation.get('array_proofs')=={k:frozen26.array_proof(parent['samples'][k]) for k in ARRAY_KEYS},
        'Original26 full validation and per-array parent bit proofs required')
    return parent


def _append(parent, additions, allowed):
    sources = deepcopy(parent['sources']); parts = [parent['samples'],*[v['samples'] for v in additions]]
    arrays = {k:[parent['samples'][k]] for k in ARRAY_KEYS}
    for part in parts[1:]:
        offset = len(sources); sources.extend(deepcopy(part['sources']))
        for key in ARRAY_KEYS:
            need(part[key].dtype==parts[0][key].dtype, 'New rows must preserve exact parent array storage dtype')
            values = part[key]
            if key=='source_indices':
                need(len(values)==0 or int(values.max())+offset<=np.iinfo(values.dtype).max, 'V4 source-index storage overflow')
                values = values+offset
            arrays[key].append(values)
    need(len({s['source_id'] for s in sources})==len(sources), 'No duplicate source IDs or repeated actual query weight allowed')
    return dict(schema=SCHEMA,sources=sources,source_table_sha256=rt.digest(sources),allowed_train_ids=sorted(allowed),
        optimization_admission=deepcopy(ADMISSION),**old.FLAGS,**{k:np.concatenate(v) for k,v in arrays.items()})


def contribution_inventory(auxiliary):
    """Semantic data groups only; this API declares no sampler weights/sizes."""
    samples = auxiliary['samples']; sources = auxiliary['sources']; groups = {}
    groups['native'] = dict(origin='separate_original_native_bank',included_in_auxiliary=False)
    for name,kind in (('planned','planned_ideal_tracking'),('conditional','conditional_ideal_continuation'),('oracle','student_oracle_unexecuted')):
        selected = [(i,s) for i,s in enumerate(sources) if s['kind']==kind]
        groups[name] = dict(kind=kind,accepted_sources=len(selected),samples=sum(len(s['windows']) for _,s in selected),
            subject_ids=sorted({s['subject_id'] for _,s in selected}),source_ids=[s['source_id'] for _,s in selected],
            supervised_cells=int(sum(np.count_nonzero(~samples['setter_is_pad'][samples['source_indices']==i]) for i,_ in selected)),
            actions_executed=False,physics_qualified=False,future_measured_histories=False)
    new = [s for s in sources if s.get('schema')==SOURCE_SCHEMA]
    return dict(groups=groups,parent=dict(rows=PARENT_ROWS,sources=PARENT_SOURCES,attempted_slots=PARENT_SLOTS),
        new_source26=dict(accepted_sources=len(new),samples=sum(len(s['windows']) for s in new),
            attempted_slots=len(auxiliary['slot_ledger'])-PARENT_SLOTS,
            subject_ids=sorted({s['subject_id'] for s in new})))


def _assemble(parent, additions, runs, terminal_ref, request, ledger, allowed):
    unique_ledger = [slot for added in additions for slot in added['slot_ledger']]
    validate_terminal_ledger(request,ledger,unique_ledger,allowed)
    samples = _append(parent,additions,allowed)
    auxiliary = dict(samples=samples,sources=samples['sources'],slot_ledger=deepcopy(parent['slot_ledger'])+deepcopy(ledger),
        request_refs=deepcopy(parent['request_refs'])+[r['frozen_request'] for r in runs]+[terminal_ref])
    proof = assert_ordered_parent(parent,auxiliary)
    auxiliary['provenance'] = dict(parent_complete=deepcopy(PARENT_COMPLETE),train_scope=deepcopy(TRAIN16_SCOPE),
        parent_archive_validation=deepcopy(parent['archive_validation']),ordered_parent_proof=proof,
        new_oracle_runs=deepcopy(runs),terminal_request=deepcopy(terminal_ref),
        new_unique_slot_ledger=deepcopy(unique_ledger),data_code=code_binding())
    auxiliary['counts'] = contribution_inventory(auxiliary)
    return auxiliary


def _input_headers(runs, terminal_ref, ledger, allowed, train_scope_ref):
    _scope(train_scope_ref,allowed)
    request = rt.read_bound(terminal_ref); _terminal_request(request,allowed)
    for case in request['cases']:
        config = rt.read_bound(case['source']['config'])
        meta = config.get('m2_given_contact_expanded_pilot_deployment',{}).get('metadata',{})
        old._train(meta.get('subject_id'),meta.get('split'),allowed)
        need(all(case[k]==meta.get(k) for k in ('subject_id','split','target_id')), 'Frozen terminal case differs from original TRAIN source config')
    _parent_header(allowed); _run_headers(runs,allowed)
    need(request.get('oracle_request_refs')==[r['frozen_request'] for r in runs], 'Outer semantic request must bind every original frozen query request in order')
    need(isinstance(ledger,list) and len(ledger)==len(request['slots']), 'Complete outer semantic ledger required')
    for row in ledger: old._train(row.get('subject_id'),row.get('split'),allowed)
    owners = set()
    for case in request['cases']:
        source = case['source']; owners.add(rt.digest(source['owner'])); owner = _closed_owner(source['owner'])
        need(owner.get('config',{}).get('experiment')=='isaac_batch' and len(owner['config'].get('jobs',[]))==4,
            'Whole original four-case DEV batch owner required before terminal data treatment')
        need(rt.read_bound(source['native']).get('success') is False, 'Original actual TRAIN target must remain failed before V4 data treatment')
        for ref in source.values(): rt.bound_path(ref)
    need(len(owners)==1, 'Whole original four-case batch must retain its single shared owner')
    by_job = {c['job']:c for c in request['cases']}
    for header in _run_headers(runs,allowed):
        for item in header['pending']:
            case = by_job.get(item['case']['job'])
            need(case is not None and case['source']==item['request']['source'], 'Original diagnostic source differs from frozen terminal case')
    _preflight_runs(runs,allowed)
    return request


def _authenticate_terminal_sources(request):
    cases = request['cases']; records = frozen26.authenticate_sources([c['source'] for c in cases])
    need(len(records)==len(cases), 'Original26 authentication required for every failed case, including missing queries')
    for case,record in zip(cases,records):
        meta = record['metadata']
        need(all(case[k]==meta.get(k) for k in ('subject_id','split','target_id')) and case['pose_id']==record.get('pose_id'),
            'Actual authenticated original source differs from frozen terminal target/pose')


def export_from_parent(output, *, oracle_runs, terminal_request_ref, slot_ledger, allowed_train_ids, train_scope_ref=None):
    """Append unique actual source26 queries; preserve all 3/6 semantic attempts.

    terminal_request_ref binds REQUEST_SCHEMA, slots, oracle_request_refs and
    false physical/training flags. Each slot has slot_id/job/subject_id/split/
    pose_id/target_id/semantic/physics_row (None means missing). slot_ledger has
    those exact fields plus status accepted/rejected/duplicate/missing. Duplicate
    slots name duplicate_of and carry no source_id; canonical accepted rows name
    their one source_id. Root freezes these inputs after original owner closure.
    """
    train_scope_ref = TRAIN16_SCOPE if train_scope_ref is None else train_scope_ref
    request = _input_headers(oracle_runs,terminal_request_ref,slot_ledger,allowed_train_ids,train_scope_ref)
    _authenticate_terminal_sources(request)
    parent = load_parent(allowed_train_ids)
    additions = [import_student_oracle(r['complete'],r['frozen_request'],allowed_train_ids=allowed_train_ids) for r in oracle_runs]
    auxiliary = _assemble(parent,additions,oracle_runs,terminal_request_ref,request,slot_ledger,allowed_train_ids)
    samples = auxiliary['samples']; old._numeric(samples,samples['sources'],full_oracle_prefix=True)
    output = Path(output); need(not output.exists(), 'Fresh V4 preparation directory required'); output.mkdir(parents=True)
    path = output/'AUXILIARY_ARRAYS.npz'
    with path.open('xb') as stream: np.savez_compressed(stream,**{k:samples[k] for k in ARRAY_KEYS})
    meta = {k:v for k,v in samples.items() if k not in ARRAY_KEYS}
    meta.update({k:deepcopy(auxiliary[k]) for k in ('slot_ledger','request_refs','provenance','counts')})
    bound = rt.write_new(output/'METADATA.json',meta)
    return rt.write_new(output/'COMPLETE.json',dict(schema=SCHEMA,complete=True,metadata=bound,arrays=rt.reference(path),
        metadata_sha256=bound['sha256'],optimization_admission=deepcopy(ADMISSION),**old.FLAGS))


def metadata(ref, allowed_train_ids):
    """Admission metadata and bound hashes only; never replay/load training NPZ."""
    complete = rt.read_bound(ref)
    need(complete.get('schema')==SCHEMA and complete.get('complete') is True and complete.get('optimization_admission')==ADMISSION,
        'Complete explicit V4 preparation required'); old._flags(complete)
    meta = rt.read_bound(complete['metadata']); _identity_gate(meta['sources'],allowed_train_ids)
    need(meta.get('schema')==SCHEMA and meta.get('optimization_admission')==ADMISSION and
        complete.get('metadata_sha256')==complete['metadata']['sha256'] and
        meta.get('source_table_sha256')==rt.digest(meta['sources']) and meta.get('allowed_train_ids')==sorted(allowed_train_ids),
        'V4 metadata/source table binding differs'); old._flags(meta)
    for source in meta['sources']: old._flags(source)
    p = meta['provenance']
    need(p.get('parent_complete')==PARENT_COMPLETE and p.get('train_scope')==TRAIN16_SCOPE and
        p.get('data_code')==code_binding(), 'Pinned V4 parent/scope/actual numerical code bindings required')
    before, parent_meta = _parent_header(allowed_train_ids)
    for key in ('sources','slot_ledger','request_refs'):
        need(meta[key][:len(parent_meta[key])]==parent_meta[key], 'Exact ordered parent metadata prefix changed: '+key)
    runs = p['new_oracle_runs']; tail = meta['slot_ledger'][PARENT_SLOTS:]
    request = _input_headers(runs,p['terminal_request'],tail,allowed_train_ids,p['train_scope'])
    validate_terminal_ledger(request,tail,p['new_unique_slot_ledger'],allowed_train_ids)
    need(meta['request_refs']==parent_meta['request_refs']+[r['frozen_request'] for r in runs]+[p['terminal_request']],
        'Exact ordered parent/new request refs required')
    _new_source_headers(meta['sources'][PARENT_SOURCES:],allowed_train_ids)
    need(all(s.get('schema')==SOURCE_SCHEMA for s in meta['sources'][PARENT_SOURCES:]), 'Only explicit V4 sources may append to complete parent')
    _hash_references(complete); _hash_references(meta); _hash_references(before); _hash_references(parent_meta)
    return complete,meta


def load_preparation(complete_ref, allowed_train_ids):
    """Authoritative full admission: rerun original parent and actual new queries."""
    complete,meta = metadata(complete_ref,allowed_train_ids)
    with np.load(rt.bound_path(complete['arrays']),allow_pickle=False) as archive:
        need(set(archive.files)==set(ARRAY_KEYS), 'Exact separate V4 array keys required')
        samples = {k:v for k,v in meta.items() if k not in ('slot_ledger','request_refs','provenance','counts')}
        samples.update({k:archive[k].copy() for k in ARRAY_KEYS})
    actual = dict(samples=samples,sources=samples['sources'],**{k:meta[k] for k in ('slot_ledger','request_refs','provenance','counts')})
    old._numeric(samples,samples['sources'],full_oracle_prefix=True)
    parent = load_parent(allowed_train_ids); p = meta['provenance']; runs = p['new_oracle_runs']
    additions = [import_student_oracle(r['complete'],r['frozen_request'],allowed_train_ids=allowed_train_ids) for r in runs]
    request = rt.read_bound(p['terminal_request'])
    _authenticate_terminal_sources(request)
    expected = _assemble(parent,additions,runs,p['terminal_request'],request,meta['slot_ledger'][PARENT_SLOTS:],allowed_train_ids)
    proof = assert_ordered_parent(parent,actual)
    need(p.get('ordered_parent_proof')==proof and p.get('parent_archive_validation',{}).get('array_proofs')==proof,
        'Revalidated original26 parent bit proof differs')
    assert_ordered_parent(expected,actual)
    need(len(samples['source_indices'])==len(expected['samples']['source_indices']) and
        actual['sources']==expected['sources'] and actual['slot_ledger']==expected['slot_ledger'] and
        p['new_unique_slot_ledger']==expected['provenance']['new_unique_slot_ledger'] and meta['counts']==expected['counts'],
        'Exact V4 full source/query/attempt inventory differs')
    return actual
