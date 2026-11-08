"""V9 three-bank TRAIN sampling with unchanged native32 and auxiliary groups.

No schema aliases, normalization fitting or preview-to-native promotion. The
immutable inventory describes every original source and every packed mask.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import m2_given_contact_runtime as rt
from . import m2_measured_recovery_data as query
from . import m2_measured_dart_data as dart
from . import m2_act_measured_sampling as previous
from . import m2_act_corridor_sampling as corridor
from . import m2_act_refinement_data as data
from . import m2_act_onpolicy_collection as collection

need=rt.need
SCHEMA='m2_act_refinement_sampling_inventory_v9'
BANKS=('query32','dart_enriched','onpolicy8')
APIS={name:data for name in BANKS}
PROFILES=dict(query32=(data.QUERY_SCHEMA,data.QUERY_PROFILE),
    dart_enriched=(data.ENRICHED_SCHEMA,data.ENRICHED_PROFILE),onpolicy8=(data.ONPOLICY_SCHEMA,data.ONPOLICY_PROFILE))
GROUP_WEIGHTS=dict(native=.5,planned=.125,conditional=.125,oracle=.25)
NATIVE_ORDER=previous.NATIVE_ORDER
DRAWS=previous.DRAWS
LABELS=(*previous.LABELS,'bank')


def _admission(meta):
    if meta['schema']==data.QUERY_SCHEMA:
        original=query.admission(meta.get('attempt_ledger'),profile=query.PROFILE_V2)
        need(original['measured_native_optimization_eligible'], 'All original four qualified query episodes required')
        return dict(original,schema='m2_act_refinement_query32_admission_v1',data_optimization_eligible=True,benchmark_act_eligible=False)
    if meta['schema']==data.ENRICHED_SCHEMA:
        original=dart.admission(meta.get('attempt_ledger'),precollection=meta.get('precollection'))
        return dict(original,schema='m2_act_refinement_enriched_dart_admission_v1',
            data_optimization_eligible=bool(original['pilot_optimization_eligible'] and original['qualified_episodes']==17),benchmark_act_eligible=False)
    return collection.admission(meta.get('attempt_ledger'))


def _headers(complete,meta,name,*,dataset,normalization):
    schema,profile=PROFILES[name]
    need(dataset==query.BASE29 and complete.get('schema')==meta.get('schema')==schema and
        complete.get('profile')==meta.get('profile')==profile and complete.get('complete') is True and
        meta.get('normalization')==normalization and meta.get('normalization_refitted') is False and
        not meta.get('single_episode_only',False),'Exact distinct V9 schemas and unchanged Base29 normalization required')
    gate=_admission(meta)
    need(gate.get('data_optimization_eligible') is True and complete.get('optimization_admission')==meta.get('optimization_admission')==gate,
        'Three complete independently admitted positive TRAIN banks required')
    _identities(meta,name)
    return gate


def _identities(meta,name):
    sources=meta.get('sources');need(isinstance(sources,list) and sources,'Complete ordered positive sources required')
    if name=='query32':
        expected=list(query.query.SLOT_IDS)
    else:expected=[e['request']['slot_id'] for e in meta['attempt_ledger'] if e['status']=='qualified']
    if name=='onpolicy8':need(expected==[r['slot_id'] for r in collection.planned_requests()], 'All fixed eight on-policy slots required')
    need([s.get('slot_id') for s in sources]==expected,'Every ordered original source slot required')
    cursor=0;episodes=set()
    requests={r['slot_id']:r for r in dart.planned_requests()}
    for i,source in enumerate(sources):
        m=source.get('metadata',{});row=source.get('declaration',{});slot=source['slot_id']
        request=(dict(split='train',subject_id='p001120',pose_id=dart.POSES[0],target_id='R3' if i<2 else 'L3')
            if name=='query32' else requests[slot])
        need(all(row.get(k)==request[k] and m.get(k,row[k] if k=='pose_id' else None)==request[k]
            for k in ('split','subject_id','pose_id','target_id')) and
            isinstance(m.get('episode_id'),str) and m['episode_id']==row.get('episode_id') and m['episode_id'] not in episodes,
            'Only original independently admitted TRAIN body/pose/target/episode identities allowed')
        n=source.get('windows');need(type(n) is int and n>0 and source.get('split_sample_range')==[cursor,cursor+n],
            'Complete contiguous original source sample boundaries required')
        cursor+=n;episodes.add(m['episode_id'])
    need(cursor==meta.get('windows'),'Complete positive source window count required')


def admitted_metadata(refs,*,dataset,normalization):
    need(isinstance(refs,dict) and set(refs)==set(BANKS),'All three distinct query32/dart_enriched/onpolicy8 references mandatory')
    # Gate ALL THREE typed headers, identities and fixed-denominator admissions
    # before any deeper consumer API (enrichment reaudits its historical parent).
    headers={}
    for name in BANKS:
        rt._reference_shape(refs[name])
        need(set(refs[name])=={'path','bytes','sha256'} and type(refs[name].get('bytes')) is int and refs[name]['bytes']>0,
            'Full immutable bank path/bytes/SHA256 references required')
        value=rt.read_bound(refs[name])
        schema,profile=PROFILES[name]
        need(value.get('schema')==schema and value.get('profile')==profile and value.get('complete') is True,
            'Explicit separate bank schema/profile required; no old negative promotion')
        meta=rt.read_bound(value['metadata']);_headers(value,meta,name,dataset=dataset,normalization=normalization)
        headers[name]=dict(reference=deepcopy(refs[name]),complete=value,metadata=meta)
    need(len({rt.digest(refs[n]) for n in BANKS})==3,'Distinct immutable measured banks required')
    _unique_sources(headers)
    result={}
    for name in BANKS:
        complete,meta=APIS[name].metadata(refs[name]);_headers(complete,meta,name,dataset=dataset,normalization=normalization)
        need(complete==headers[name]['complete'] and meta==headers[name]['metadata'],'Bank header changed during consumer admission')
        result[name]=dict(reference=deepcopy(refs[name]),complete=complete,metadata=meta)
    _unique_sources(result)
    return result


def _unique_sources(banks):
    episodes=set();windows=set()
    for name in BANKS:
        meta=banks[name]['metadata'];index=rt.read_bound(meta['index']);items=index.get('samples')
        need(isinstance(items,list) and len(items)==meta['windows'] and index.get('order_sha256')==rt.digest(items) and
            [w.get('sample_index') for w in items]==list(range(meta['windows'])), 'Exact complete ordered V9 bank index required')
        for source in meta['sources']:
            episode=source['metadata']['episode_id'];need(episode not in episodes,'Duplicate raw source episode across banks')
            episodes.add(episode);lo,hi=source['split_sample_range'];ticks=[]
            for item in items[lo:hi]:
                need(item.get('slot_id')==source['slot_id'],'Bank window slot/source order differs')
                tick=item['window'].get('inclusive_native_ticks',[])
                need(isinstance(tick,list) and len(tick)==2 and all(type(t) is int for t in tick),'Actual raw native window ticks required')
                identity=(episode,tick[0]);need(identity not in windows,'Duplicate raw source window across banks')
                windows.add(identity);ticks.append(tick[0])
            need(ticks==sorted(set(ticks)), 'Strict chronological raw source windows required')


def _mandatory_targets(rows):
    need({'R3','L3'}<={r['target'] for r in rows},'Both mandatory target pools must be nonempty')


def _binding(bank,records,offset):
    meta=bank['metadata']
    return dict(reference=deepcopy(bank['reference']),metadata=deepcopy(bank['complete']['metadata']),
        schema=meta['schema'],profile=meta['profile'],consumer_metadata=deepcopy(meta),
        arrays=deepcopy(meta['arrays']),array_proofs=deepcopy(meta['array_proofs']),index=deepcopy(meta['index']),
        optimization_admission=deepcopy(meta['optimization_admission']),sources=deepcopy(records),windows=meta['windows'],
        composite_sample_range=[offset,offset+meta['windows']])


def load_banks(refs,*,dataset,normalization):
    result=admitted_metadata(refs,dataset=dataset,normalization=normalization)
    for name in BANKS:
        samples,meta=APIS[name].load_arrays(refs[name]);need(meta==result[name]['metadata'],'Bank metadata changed during loading')
        result[name]['samples']=samples
    return result


def source_records(meta,name):
    _identities(meta,name)
    if name=='query32':records=previous.source_records(meta)
    else:
        records=[];base=Path(meta['source_manifest']['path']).parent
        for source in meta['sources']:
            m=source['metadata'];row=source['declaration'];run,report=query.data._bound(row['run_result'],base)
            _,config=query.data._bound(row['config'],base)
            points=[p for p in run.get('points',[]) if p.get('target')==m['target_id']]
            need(len(points)==1,'One actual DART recovered target point required')
            point=points[0];gap=point.get('phase_config',{}).get('precontact_gap_m')
            reference=rt.pose.require_route(m);fresh=reference.snapshot()['sources']['fresh_plan']
            plan_ref=rt.reference(fresh['file']);plan=rt.read_bound(plan_ref)
            need(type(gap) in (int,float) and np.isfinite(gap) and gap>0 and
                gap==point.get('checked_prepose_gap_m')==plan.get('free_approach_start_gap_m') and
                plan_ref['sha256']==fresh['sha256'] and plan.get('name',plan.get('target'))==reference.snapshot()['target_id']==m['target_id'],
                'Actual DART dynamic PREPOSE gap/checked plan differs')
            records.append(dict(slot=source['slot_id'],body=m['subject_id'],pose=row['pose_id'],target=m['target_id'],
                episode=m['episode_id'],actual_gap_m=float(gap),split_sample_range=deepcopy(source['split_sample_range']),
                config=rt.reference(config),report=rt.reference(report),canonical_plan=plan_ref))
    for record in records:
        record.update(bank=name,source='measured:'+name+':'+record['slot']+':'+record['episode'])
    return records


def _pool(rows,*,threshold=None,slots=()):
    result=previous._pool(rows,threshold=threshold)
    result['missing_slots']=sorted(set(slots)-{r.get('bank','')+':'+r.get('slot','') for r in rows})
    return result


def measured_tree(rows):
    tree={}
    for row in rows:
        tree.setdefault(row['target'],{}).setdefault(row['body'],{}).setdefault(row['pose'],{}).setdefault(
            row['bank']+':'+row['slot'],[]).append(row['row_index'])
    return tree


def draw_measured(rng,tree):
    active=tree
    for _ in range(4):active=active[previous._choose(rng,sorted(active))]
    return previous._choose(rng,active)


def configure_cache(cache,prepared,auxiliary,*,torch=None,device='cpu'):
    banks=prepared.get('measured_banks');need(isinstance(banks,dict) and set(banks)==set(BANKS),'All three measured TRAIN banks required')
    authenticated=admitted_metadata({n:banks[n]['reference'] for n in BANKS},dataset=query.BASE29,normalization=prepared['preparation']['normalization'])
    need(all(authenticated[n]['metadata']==banks[n]['metadata'] and authenticated[n]['complete']==banks[n]['complete'] for n in BANKS),
        'Prepared bank metadata differs from authenticated consumer')
    base_count=cache['native_count'];offset=base_count;rows=[];bindings={}
    for name in BANKS:
        bank=banks[name];meta=bank['metadata'];samples=bank['samples']
        _headers(bank['complete'],meta,name,dataset=query.BASE29,normalization=prepared['preparation']['normalization'])
        query.validate_samples(samples)
        need(samples.get('derived_index',{}).get('windows')==[w['window'] for w in rt.read_bound(meta['index'])['samples']],
            'Prepared native window order/semantics differ from exact admitted index')
        need({k:query.array_proof(samples[k]) for k in query.ARRAY_KEYS}==meta.get('array_proofs'),
            'Measured ordered arrays/masks proof bytes changed before packing')
        need(samples.get('bank_schema')==meta['schema'] and samples.get('normalization_reference')==meta['normalization'] and
            rt.validate_normalization(rt.read_bound(meta['normalization']))==prepared['normalization'],
            'Exact source-specific admission and unchanged native normalization required')
        # Each strict bank loader has admitted these seven arrays; the original
        # numerical packer consumes them unchanged. No bank schema is relabeled.
        packed=rt.given.pack_batch(samples,prepared['normalization'],np.arange(len(samples['states'])),training=False)
        for key,value in packed.items():
            cache['packed']['native'][key]=(torch.cat([cache['packed']['native'][key],torch.from_numpy(value).to(device)],dim=0)
                if torch is not None else np.concatenate([cache['packed']['native'][key],value]))
        records=source_records(meta,name);actual=previous._native_rows(samples,records,offset=offset)
        for row in actual:row['bank']=name
        rows.extend(actual)
        bindings[name]=_binding(bank,records,offset)
        offset+=meta['windows']
    proof=corridor.build_inventory(prepared,auxiliary)
    base_rows=previous._native_rows(prepared['train'],proof['pools']['native']['sources'])
    slots=[r['bank']+':'+r['slot'] for r in rows]
    pools=dict(base_full=_pool(base_rows),base_corridor=_pool([r for r in base_rows if r['error_m']<=.012],threshold=.012),
        measured_full=_pool(rows,slots=slots),measured_corridor=_pool([r for r in rows if r['error_m']<=.012],threshold=.012,slots=slots))
    for name in NATIVE_ORDER:
        _mandatory_targets(pools[name]['rows'])
    for name,kind in (('planned','planned_ideal_tracking'),('conditional','conditional_ideal_continuation'),('oracle','student_oracle_unexecuted')):
        actual=previous._auxiliary_rows(auxiliary,kind);pools[name+'_full']=_pool(actual)
        if name!='oracle':
            allowed=set(proof['pools'][name]['eligible_row_indices'])
            pools[name+('_near' if name=='conditional' else '_corridor')]=_pool([r for r in actual if r['row_index'] in allowed],threshold=.006 if name=='conditional' else .012)
    inventory=dict(schema=SCHEMA,data_version=5,model_optimizer_version=9,native_order=list(NATIVE_ORDER),
        base_counts=deepcopy(prepared['preparation']['counts']),base_train_windows=base_count,measured_windows=len(rows),native_index=deepcopy(prepared['preparation']['index']),
        measured_banks=bindings,normalization=deepcopy(prepared['preparation']['normalization']),normalization_sha256=rt.digest(prepared['normalization']),
        corridors=proof,pools=pools,base_array_proofs={k:query.array_proof(prepared['train'][k]) for k in query.ARRAY_KEYS if k in prepared['train']},
        auxiliary_array_proofs={k:query.array_proof(auxiliary['samples'][k]) for k in ('source_histories','relative_setters','setter_is_pad','source_indices')},
        physical_measured_subject_ids=sorted({r['body'] for r in base_rows+rows}),computed_auxiliary_subject_ids=sorted({s['subject_id'] for s in auxiliary['sources']}),
        normalization_fitted=False,physical_gates_changed=False,labels_changed=False,
        group_weights=deepcopy(GROUP_WEIGHTS),native_loss='one native32 valid-cell mean; equal example draws are not equal gradient shares')
    for field,key in (('source_ids','source'),('subjects','body')):cache[field]['native']=np.r_[cache[field]['native'],[r[key] for r in rows]]
    cache['kinds']['native']=np.r_[cache['kinds']['native'],np.full(len(rows),'measured_recovery')]
    cache.update(data_version=5,base_native_count=base_count,measured_count=len(rows),refinement_inventory=inventory,
        refinement_labels={name:{r['row_index']:r for r in pool['rows']} for name,pool in pools.items()},
        refinement_trees={name:measured_tree(pools[name]['rows']) for name in ('measured_full','measured_corridor')},
        planned_corridor_tree=corridor.filtered_tree(cache['trees']['planned'],set(proof['pools']['planned']['eligible_row_indices'])))
    return cache


def native_indices(rng,cache):
    result=[];pools=cache['refinement_inventory']['pools']
    for name in NATIVE_ORDER:
        if name.startswith('measured'):result.extend(draw_measured(rng,cache['refinement_trees'][name]) for _ in range(8))
        else:
            rows=pools[name]['eligible_row_indices'];result.extend(np.asarray(rows)[rng.integers(len(rows),size=8)].tolist())
    return np.asarray(result,dtype=np.int64)


def new_draws():
    return {name:dict(draws=0,denominator=0,valid_cells=0,valid_joints=0,rows={},**{key:{} for key in LABELS}) for name in DRAWS}


def record_draws(value,cache,indices):
    for name,(group,start,end) in DRAWS.items():
        counts=value['refinement_draws'][name];lookup=cache['refinement_labels'][name];selected=indices[group][start:end]
        need(len(selected)==end-start,'Exact V9 subgroup size required')
        for index in selected:
            index=int(index);need(index in lookup,'V9 draw outside declared pool');row=lookup[index]
            counts['draws']+=1;counts['denominator']+=1;counts['valid_cells']+=row['valid_cells'];counts['valid_joints']+=row['valid_cells']*7
            key=str(index);counts['rows'][key]=counts['rows'].get(key,0)+1
            for key in LABELS:
                if key in row:counts[key][row[key]]=counts[key].get(row[key],0)+1


def validate_draws(value,*,steps,inventory):
    need(inventory.get('schema')==SCHEMA and inventory.get('model_optimizer_version')==9 and inventory.get('data_version')==5 and
        inventory.get('native_order')==list(NATIVE_ORDER) and inventory.get('group_weights')==GROUP_WEIGHTS and set(inventory.get('pools',{}))==set(DRAWS) and
        set(value.get('refinement_draws',{}))==set(DRAWS),'Explicit complete V9 subgroup proof required')
    combined={g:{} for g in ('native','planned','conditional','oracle')}
    for name,(group,start,end) in DRAWS.items():
        pool=inventory['pools'][name];rows=pool['rows'];lookup={str(r['row_index']):r for r in rows}
        threshold=.012 if name in ('base_corridor','measured_corridor','planned_corridor') else .006 if name=='conditional_near' else None
        need(rows and pool['row_count']==len(rows)==len(lookup) and pool['order_sha256']==rt.digest(rows) and
            pool['eligible_row_indices']==[r['row_index'] for r in rows] and pool['threshold_m']==threshold,
            'Exact V9 pool order/threshold proof required')
        for row in rows:
            need(type(row['row_index']) is int and type(row['valid_cells']) is int and 1<=row['valid_cells']<=20,'Positive valid-cell row required')
            if group=='native':
                low,high=(0,inventory['base_train_windows']) if name.startswith('base') else (inventory['base_train_windows'],inventory['base_train_windows']+inventory['measured_windows'])
                need(low<=row['row_index']<high and np.isfinite(row['error_m']) and row['error_m']>=0 and row['actual_gap_m']>0 and
                    (threshold is None or row['error_m']<=threshold),'V9 native row/gap eligibility differs')
        counts=value['refinement_draws'][name];n=steps*(end-start);draws=counts.get('rows',{})
        need(counts.get('draws')==counts.get('denominator')==n and draws and set(draws)<=set(lookup) and
            all(type(c) is int and c>0 for c in draws.values()) and sum(draws.values())==n,'Exact V9 realized row draw counts required')
        labels={k:{} for k in LABELS};valid=0
        for index,count in draws.items():
            row=lookup[index];valid+=count*row['valid_cells']
            for key in LABELS:
                if key in row:labels[key][row[key]]=labels[key].get(row[key],0)+count
        need(all(counts.get(k)==v for k,v in labels.items()) and counts.get('valid_cells')==valid and counts.get('valid_joints')==valid*7,
            'V9 realized source/bank/body/pose/valid-cell exposures must reconcile')
        for source,count in labels['source'].items():combined[group][source]=combined[group].get(source,0)+count
        if name=='conditional_near':
            old=value.get('conditional_terminal_draws',{})
            need(old.get('draws')==n and old.get('denominator')==steps*8 and all(old.get(k)==labels[k] for k in ('source','body','pose','target','anchor')),
                'Original conditional4/8 counters differ')
    need(combined==value.get('gradient_source_examples'),'Every V9 optimized source must reconcile')


def validate_binding(inventory,*,config,native,auxiliary_metadata,auxiliary_arrays,frozen_scope=None):
    """Authenticate current consumers and original producers; no model reads.

    The enriched data API also reaudits its original274 parent arrays. This is
    the existing strict data admission, not a metadata-only compatibility path.
    """
    banks=admitted_metadata(config['measured_banks'],dataset=config['dataset'],normalization=native['normalization'])
    need(inventory.get('schema')==SCHEMA and inventory.get('data_version')==5 and inventory.get('model_optimizer_version')==9 and
        inventory.get('native_order')==list(NATIVE_ORDER) and inventory.get('group_weights')==GROUP_WEIGHTS and set(inventory.get('measured_banks',{}))==set(BANKS) and
        all(inventory.get(k) is False for k in ('normalization_fitted','physical_gates_changed','labels_changed')) and
        inventory.get('base_counts')==native['counts'] and inventory.get('base_train_windows')==native['counts']['train'] and inventory.get('native_index')==native['index'] and
        inventory.get('normalization')==native['normalization'] and
        inventory.get('normalization_sha256')==rt.digest(rt.validate_normalization(rt.read_bound(native['normalization']))),
        'Exact V9 composite/native/statistics binding required')
    corridor.validate_binding(inventory['corridors'],native=native,auxiliary_metadata=auxiliary_metadata,auxiliary_arrays=auxiliary_arrays,frozen_scope=frozen_scope)
    records=deepcopy(inventory['corridors']['pools']['native']['sources']);offset=inventory['base_train_windows'];measured=[]
    for name in BANKS:
        bank=banks[name];meta=bank['metadata'];actual=source_records(meta,name);binding=inventory['measured_banks'][name]
        expected=_binding(bank,actual,offset)
        need(binding==expected,'Exact source-specific V9 bank/order/array/mask binding required')
        index=rt.read_bound(meta['index'])['samples'];selected=[r for r in inventory['pools']['measured_full']['rows'] if r.get('bank')==name]
        need([r['row_index'] for r in selected]==list(range(offset,offset+meta['windows'])) and
            [r['valid_cells'] for r in selected]==[w['window']['valid_future_cells'] for w in index],
            'V9 measured original sample order and admitted masks differ')
        for record in actual:
            record=deepcopy(record);record['split_sample_range']=[n+offset for n in record['split_sample_range']];measured.append(record)
        offset+=meta['windows']
    need(inventory['measured_windows']==offset-inventory['base_train_windows'],'All measured windows required')
    pools=inventory['pools'];slots=[r['bank']+':'+r['slot'] for r in measured]
    native_windows=[row for row in rt.read_bound(native['index']).get('samples',[]) if row.get('split')=='train']
    base_rows=pools['base_full']['rows']
    need(len(native_windows)==len(base_rows)==inventory['base_train_windows'] and
        [w.get('sample_index') for w in native_windows]==list(range(len(base_rows))) and
        all(row['episode']==window.get('episode_id') and row['valid_cells']==window.get('window',{}).get('valid_future_cells',20)
            for row,window in zip(base_rows,native_windows)), 'V9 native masks/order must match the immutable Base29 window index')
    for name,recs,low,high in (('base',records,0,inventory['base_train_windows']),('measured',measured,inventory['base_train_windows'],offset)):
        rows=pools[name+'_full']['rows'];lookup={r['source']:r for r in recs}
        need([r['row_index'] for r in rows]==list(range(low,high)),'Complete V9 full native range required')
        for row in rows:
            record=lookup.get(row.get('source'));need(record is not None,'Unknown V9 source')
            need(all(row.get(k)==record.get(k) for k in (*LABELS,'actual_gap_m')) and
                record['split_sample_range'][0]<=row['row_index']<record['split_sample_range'][1] and
                type(row.get('valid_cells')) is int and 1<=row['valid_cells']<=20 and np.isfinite(row.get('error_m',np.nan)) and row['error_m']>=0,
                'V9 source/sample/gap/mask identity differs')
        need(pools[name+'_full']==_pool(rows,slots=slots if name=='measured' else ()) and
            pools[name+'_corridor']==_pool([r for r in rows if r['error_m']<=.012],threshold=.012,slots=slots if name=='measured' else ()) and
            all({'R3','L3'}<={r['target'] for r in pools[name+suffix]['rows']} for suffix in ('_full','_corridor')),
            'Exact full/corridor membership and both targets required')
    mapping={'planned_ideal_tracking':'planned','conditional_ideal_continuation':'conditional','student_oracle_unexecuted':'oracle'}
    sources={s['source_id']:s for s in auxiliary_metadata['sources']};all_aux=[]
    for group in ('planned','conditional','oracle'):
        rows=pools[group+'_full']['rows'];counts={};all_aux.extend(r['row_index'] for r in rows)
        for row in rows:
            source=sources.get(row.get('source'))
            need(source is not None and mapping[source['kind']]==group and all(row.get(k)==source.get(field) for k,field in
                (('body','subject_id'),('pose','pose_id'),('target','target_id'),('anchor','anchor'))) and type(row.get('valid_cells')) is int and 1<=row['valid_cells']<=20,
                'V9 auxiliary source/identity/mask binding differs')
            counts[source['source_id']]=counts.get(source['source_id'],0)+1
        need(counts=={s['source_id']:len(s['windows']) for s in sources.values() if mapping[s['kind']]==group} and pools[group+'_full']==_pool(rows),
            'Complete unchanged auxiliary full pools required')
        if group!='oracle':
            allowed=set(inventory['corridors']['pools'][group]['eligible_row_indices']);name=group+('_near' if group=='conditional' else '_corridor')
            need(pools[name]==_pool([r for r in rows if r['row_index'] in allowed],threshold=.006 if group=='conditional' else .012),'Exact unchanged auxiliary corridor subset required')
    need(sorted(all_aux)==list(range(inventory['auxiliary_array_proofs']['source_indices']['shape'][0])), 'Every auxiliary row exactly once required')
    need(inventory.get('physical_measured_subject_ids')==sorted({r['body'] for r in records+measured}) and
        inventory.get('computed_auxiliary_subject_ids')==sorted({s['subject_id'] for s in sources.values()}), 'Physical measured and computed scopes must remain distinct')
    _validate_observed_pools(inventory,banks=banks,native=native,auxiliary_metadata=auxiliary_metadata,
        auxiliary_arrays=auxiliary_arrays,frozen_scope=frozen_scope)


def _validate_observed_pools(inventory,*,banks,native,auxiliary_metadata,auxiliary_arrays,frozen_scope):
    """Rederive every pool from authenticated observations, not stored errors.

    All three bank admissions, source identities and indices, plus native and
    computational source metadata, have passed before these numerical reads.
    """
    base=corridor.phase._archive(native['arrays']['train'],query.ARRAY_KEYS)
    base.update(schema=rt.given.MASKED_DATASET_SCHEMA,task_id=rt.pose.TASK_ID,
        observation_contract=rt.pose.observation_contract(),action_semantics_id=rt.given.ACTION_SEMANTICS_ID,
        label_protocol=rt.shared.MASKED_LABEL_PROTOCOL)
    query.validate_samples(base)
    need({k:query.array_proof(base[k]) for k in query.ARRAY_KEYS}==inventory['base_array_proofs'],
        'Observed Base29 arrays differ from bound native proofs')
    auxiliary=dict(samples=corridor.phase._archive(auxiliary_arrays,corridor.phase.ARRAY_KEYS),
        sources=auxiliary_metadata['sources'],array_reference=auxiliary_arrays)
    samples=auxiliary['samples'];indices=np.asarray(samples['source_indices']);mask=np.asarray(samples['setter_is_pad'])
    need(indices.ndim==1 and indices.dtype.kind in 'iu' and np.all(indices>=0) and np.all(indices<len(auxiliary['sources'])) and
        mask.dtype.kind=='b' and mask.shape==(len(indices),20) and (~mask).any(axis=1).all() and
        np.all(np.diff(mask.astype(int),axis=1)>=0),'Bound computational source indices/valid masks required')
    need({k:query.array_proof(samples[k]) for k in ('source_histories','relative_setters','setter_is_pad','source_indices')}==inventory['auxiliary_array_proofs'],
        'Observed computational arrays/source indices differ from bound proofs')
    prepared=dict(train=base,preparation=native,optimization_scope=frozen_scope or {})
    proof=corridor.build_inventory(prepared,auxiliary)
    need(proof==inventory['corridors'],'Observed source states/gaps disagree with rebuilt corridor inventory')
    base_rows=previous._native_rows(base,proof['pools']['native']['sources']);rows=[];offset=len(base['states'])
    for name in BANKS:
        bank=banks[name];samples,meta=APIS[name].load_arrays(bank['reference'])
        need(meta==bank['metadata'],'Measured metadata changed during observed array loading')
        rt.bound_path(meta['arrays']);query.validate_samples(samples)
        need({k:query.array_proof(samples[k]) for k in query.ARRAY_KEYS}==meta['array_proofs'] and
            samples.get('derived_index',{}).get('windows')==[w['window'] for w in rt.read_bound(meta['index'])['samples']],
            'Observed measured arrays/masks/order differ from admitted proofs')
        actual=previous._native_rows(samples,source_records(meta,name),offset=offset)
        for row in actual:row['bank']=name
        rows.extend(actual);offset+=len(samples['states'])
    slots=[r['bank']+':'+r['slot'] for r in rows]
    pools=dict(base_full=_pool(base_rows),base_corridor=_pool([r for r in base_rows if r['error_m']<=.012],threshold=.012),
        measured_full=_pool(rows,slots=slots),measured_corridor=_pool([r for r in rows if r['error_m']<=.012],threshold=.012,slots=slots))
    for name,kind in (('planned','planned_ideal_tracking'),('conditional','conditional_ideal_continuation'),('oracle','student_oracle_unexecuted')):
        actual=previous._auxiliary_rows(auxiliary,kind);pools[name+'_full']=_pool(actual)
        if name!='oracle':
            allowed=set(proof['pools'][name]['eligible_row_indices'])
            pools[name+('_near' if name=='conditional' else '_corridor')]=_pool([r for r in actual if r['row_index'] in allowed],
                threshold=.006 if name=='conditional' else .012)
    need(pools==inventory['pools'],'Observed source states/gaps/indices disagree with rebuilt sampling pools')
