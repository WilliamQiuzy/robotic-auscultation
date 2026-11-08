"""V7's separate measured TRAIN supplement inside the unchanged native32 loss.

Inventories are built before optimization and validated from bounded sealed
metadata at deployment. No capture decoding or historical policy replay occurs
here. Equal subgroup example counts do not imply equal gradient shares.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import m2_given_contact_runtime as rt
from . import m2_measured_recovery_data as bank
from . import m2_act_corridor_sampling as corridor

need=rt.need
SCHEMA='m2_act_measured_sampling_inventory_v7'
LABELS=('source','slot','body','pose','target','episode','anchor')
NATIVE_ORDER=('base_corridor','base_full','measured_corridor','measured_full')
# group, first position, exclusive last position; original64 and coefficients unchanged.
DRAWS={name:('native',i*8,(i+1)*8) for i,name in enumerate(NATIVE_ORDER)}
DRAWS.update(planned_corridor=('planned',0,4),planned_full=('planned',4,8),
    conditional_near=('conditional',0,4),conditional_full=('conditional',4,8),oracle_full=('oracle',0,16))


def admitted_metadata(ref,*,dataset,normalization):
    complete,meta=bank.metadata(ref)
    need(meta.get('base_dataset')==dataset and meta.get('base_counts')==bank.BASE_COUNTS and
        meta.get('normalization')==normalization and meta.get('normalization_refitted') is False,
        'Measured supplement must retain exact Base29 counts and normalization')
    expected=bank.admission(meta.get('attempt_ledger'),profile=meta.get('profile'))
    need(expected.get('measured_native_optimization_eligible') is True and
        complete.get('optimization_admission')==meta.get('optimization_admission')==expected and
        all(meta.get(key) is True for key in ('actions_executed','physics_qualified','native_eligible')),
        'All FOUR distinct physically qualified TRAIN teacher attempts required before optimization')
    source_records(meta)
    return complete,meta


def source_records(meta):
    """Bind every measured gap to recovered point, checked PREPOSE and plan."""
    sources=meta.get('sources',[])
    need(len(sources)==4 and [s.get('slot_id') for s in sources]==list(bank.query.SLOT_IDS),
        'All FOUR ordered distinct measured slots required')
    # Gate ALL identities before opening even bounded config/report/plan metadata.
    for i,source in enumerate(sources):
        m=source.get('metadata',{});row=source.get('declaration',{})
        need(m.get('split')=='train' and m.get('subject_id')=='p001120' and
            row.get('pose_id')=='p000474_a001230_f0000' and m.get('pose_id',row['pose_id'])==row['pose_id'] and m.get('target_id')==('R3' if i<2 else 'L3') and
            all(m.get(k)==row.get(k) for k in ('subject_id','split','target_id','episode_id')),
            'Only exact original TRAIN measured identities/pose/targets allowed')
    base=Path(meta['source_manifest']['path']).parent;result=[];cursor=0;episodes=set()
    for source in sources:
        m=source['metadata'];row=source['declaration'];n=source.get('windows');eid=m.get('episode_id')
        need(type(n) is int and n>0 and isinstance(eid,str) and eid not in episodes and
            source.get('split_sample_range')==[cursor,cursor+n], 'Unique complete positive measured episode ranges required')
        episodes.add(eid);cursor+=n
        run,report_path=bank.data._bound(row['run_result'],base)
        _,config_path=bank.data._bound(row['config'],base)
        points=[p for p in run.get('points',[]) if p.get('target')==m['target_id']]
        need(len(points)==1,'Exactly one actual recovered target point required')
        point=points[0];gap=point.get('phase_config',{}).get('precontact_gap_m')
        reference=rt.pose.require_route(m);fresh=reference.snapshot()['sources']['fresh_plan']
        plan_ref=rt.reference(fresh['file']);plan=rt.read_bound(plan_ref)
        need(type(gap) in (int,float) and np.isfinite(gap) and gap>0 and
            point.get('checked_prepose_gap_m')==gap==plan.get('free_approach_start_gap_m') and
            plan_ref['sha256']==fresh['sha256'] and
            plan.get('name',plan.get('target'))==reference.snapshot()['target_id']==m['target_id'],
            'Measured actual recovered phase/checked PREPOSE/canonical plan gap differs')
        result.append(dict(source='measured:'+source['slot_id']+':'+eid,slot=source['slot_id'],body=m['subject_id'],
            pose=row['pose_id'],target=m['target_id'],episode=eid,actual_gap_m=float(gap),
            split_sample_range=deepcopy(source['split_sample_range']),config=rt.reference(config_path),
            report=rt.reference(report_path),canonical_plan=plan_ref))
    need(cursor==meta.get('windows'),'Complete measured window count required')
    return result


def _pool(rows,*,threshold=None,slots=()):
    rows=sorted(rows,key=lambda r:r['row_index']);ids=[r['row_index'] for r in rows]
    need(rows and len(ids)==len(set(ids)), 'Nonempty unique required V7 sampling pool')
    return dict(rows=rows,eligible_row_indices=ids,row_count=len(rows),threshold_m=threshold,
        missing_slots=sorted(set(slots)-{r.get('slot') for r in rows}),order_sha256=rt.digest(rows))


def _native_rows(samples,records,*,offset=0):
    states=np.asarray(samples['states']);histories=np.asarray(samples['histories']);count=len(states)
    need(states.shape==(count,17) and histories.shape==(count,2,17) and np.isfinite(states).all() and
        np.array_equal(states,histories[:,1]),'Actual native H2/current state binding required')
    valid=rt.shared.valid_action_mask(samples).sum(1);rows=[]
    for record in records:
        start,end=record['split_sample_range']
        for index in range(start,end):
            rows.append(dict(row_index=offset+index,valid_cells=int(valid[index]),actual_gap_m=record['actual_gap_m'],
                error_m=float(np.linalg.norm(states[index,7:10]+[0,0,record['actual_gap_m']])),
                **{k:record[k] for k in LABELS if k in record}))
    need([r['row_index'] for r in rows]==list(range(offset,offset+count)),'Full positive native source coverage required')
    return rows


def _auxiliary_rows(auxiliary,kind):
    samples=auxiliary['samples'];valid=(~samples['setter_is_pad']).sum(1);rows=[]
    for row,i in enumerate(samples['source_indices']):
        source=auxiliary['sources'][int(i)]
        if source['kind']!=kind:continue
        entry=dict(row_index=row,valid_cells=int(valid[row]),source=source['source_id'],body=source['subject_id'],
            pose=source['pose_id'],target=source['target_id'])
        if 'anchor' in source:entry['anchor']=source['anchor']
        rows.append(entry)
    return rows


def configure_cache(cache,prepared,auxiliary,*,torch=None,device='cpu'):
    measured=prepared.get('measured');need(isinstance(measured,dict),'Separate admitted measured bank required for V7')
    samples=measured['samples'];meta=measured['metadata'];bank.validate_samples(samples)
    need({k:bank.array_proof(samples[k]) for k in bank.ARRAY_KEYS}==meta.get('array_proofs'),
        'Measured ordered array/mask proof bytes changed before packing')
    expected=bank.admission(meta.get('attempt_ledger'),profile=meta.get('profile'))
    need(expected.get('measured_native_optimization_eligible') is True and
        samples.get('optimization_admission')==meta.get('optimization_admission')==expected and
        all(meta.get(k) is True for k in ('actions_executed','physics_qualified','native_eligible')),
        'All FOUR measured attempts and physical admission required')
    records=source_records(meta);base_count=cache['native_count']
    # Same packer/stats and mask semantics; only temporary tensor indexing is combined.
    need(rt.validate_normalization(rt.read_bound(meta['normalization']))==prepared['normalization'],
        'Measured packer statistics must equal unchanged Base29 statistics')
    packed=bank.pack_batch(samples,meta['normalization'],np.arange(len(samples['states'])))
    for key,value in packed.items():
        cache['packed']['native'][key]=(torch.cat([cache['packed']['native'][key],torch.from_numpy(value).to(device)],dim=0)
            if torch is not None else np.concatenate([cache['packed']['native'][key],value]))
    proof=corridor.build_inventory(prepared,auxiliary)
    base_records=proof['pools']['native']['sources'];base_rows=_native_rows(prepared['train'],base_records)
    measured_rows=_native_rows(samples,records,offset=base_count)
    pools=dict(base_full=_pool(base_rows),base_corridor=_pool([r for r in base_rows if r['error_m']<=.012],threshold=.012),
        measured_full=_pool(measured_rows,slots=bank.query.SLOT_IDS),
        measured_corridor=_pool([r for r in measured_rows if r['error_m']<=.012],threshold=.012,slots=bank.query.SLOT_IDS))
    for name in ('measured_full','measured_corridor'):
        need({r['target'] for r in pools[name]['rows']}=={'R3','L3'},'Both measured target pools must be nonempty')
    for name,kind in (('planned','planned_ideal_tracking'),('conditional','conditional_ideal_continuation'),('oracle','student_oracle_unexecuted')):
        rows=_auxiliary_rows(auxiliary,kind);pools[name+'_full']=_pool(rows)
        if name!='oracle':
            allowed=set(proof['pools'][name]['eligible_row_indices'])
            pools[name+('_near' if name=='conditional' else '_corridor')]=_pool([r for r in rows if r['row_index'] in allowed],threshold=.006 if name=='conditional' else .012)
    inventory=dict(schema=SCHEMA,data_version=5,model_optimizer_version=7,native_order=list(NATIVE_ORDER),
        base_counts=deepcopy(prepared['preparation']['counts']),base_train_windows=base_count,measured_windows=len(samples['states']),
        measured_bank=deepcopy(measured['reference']),measured_metadata=deepcopy(measured['complete']['metadata']),
        measured_arrays=deepcopy(meta['arrays']),measured_array_proofs=deepcopy(meta['array_proofs']),
        measured_index=deepcopy(meta['index']),measured_sources=records,normalization=deepcopy(meta['normalization']),
        normalization_sha256=rt.digest(prepared['normalization']),corridors=proof,pools=pools,
        base_array_proofs={key:bank.array_proof(prepared['train'][key]) for key in bank.ARRAY_KEYS if key in prepared['train']},
        auxiliary_array_proofs={key:bank.array_proof(auxiliary['samples'][key]) for key in
            ('source_histories','relative_setters','setter_is_pad','source_indices')},
        normalization_fitted=False,physical_gates_changed=False,labels_changed=False,
        native_loss='one native32 valid-cell mean; subgroup example counts are not gradient shares')
    for field,key in (('source_ids','source'),('subjects','body')):
        cache[field]['native']=np.r_[cache[field]['native'],[r[key] for r in measured_rows]]
    cache['kinds']['native']=np.r_[cache['kinds']['native'],np.full(len(measured_rows),'measured_recovery')]
    cache.update(data_version=5,base_native_count=base_count,measured_count=len(measured_rows),measured_inventory=inventory,
        measured_labels={name:{r['row_index']:r for r in pool['rows']} for name,pool in pools.items()},
        measured_trees={name:_measured_tree(pools[name]['rows']) for name in ('measured_full','measured_corridor')},
        planned_corridor_tree=corridor.filtered_tree(cache['trees']['planned'],set(proof['pools']['planned']['eligible_row_indices'])))
    return cache


def _measured_tree(rows):
    tree={}
    for row in rows:tree.setdefault(row['target'],{}).setdefault(row['slot'],[]).append(row['row_index'])
    return tree


def _choose(rng,values):return values[int(rng.integers(len(values)))]


def native_indices(rng,cache):
    pools=cache['measured_inventory']['pools'];result=[]
    for name in NATIVE_ORDER:
        if name.startswith('measured'):
            tree=cache['measured_trees'][name]
            for _ in range(8):
                slots=tree[_choose(rng,sorted(tree))];rows=slots[_choose(rng,sorted(slots))]
                result.append(_choose(rng,rows))
        else:
            rows=pools[name]['eligible_row_indices'];result.extend(np.asarray(rows)[rng.integers(len(rows),size=8)].tolist())
    return np.asarray(result,dtype=np.int64)


def new_draws():
    return {name:dict(draws=0,denominator=0,valid_cells=0,valid_joints=0,rows={},**{key:{} for key in LABELS}) for name in DRAWS}


def record_draws(value,cache,indices):
    for name,(group,start,end) in DRAWS.items():
        counts=value['measured_draws'][name];lookup=cache['measured_labels'][name]
        selected=indices[group][start:end];need(len(selected)==end-start,'Exact V7 subgroup size required')
        for index in selected:
            index=int(index);need(index in lookup,'V7 draw outside declared pool');row=lookup[index]
            counts['draws']+=1;counts['denominator']+=1;counts['valid_cells']+=row['valid_cells'];counts['valid_joints']+=row['valid_cells']*7
            key=str(index);counts['rows'][key]=counts['rows'].get(key,0)+1
            for key in LABELS:
                if key in row:
                    label=row[key];counts[key][label]=counts[key].get(label,0)+1


def validate_draws(value,*,steps,inventory):
    need(inventory.get('schema')==SCHEMA and inventory.get('data_version')==5 and inventory.get('model_optimizer_version')==7 and
        inventory.get('native_order')==list(NATIVE_ORDER) and set(inventory.get('pools',{}))==set(DRAWS) and
        set(value.get('measured_draws',{}))==set(DRAWS),'Explicit complete V7 subgroup proof required')
    combined={group:{} for group in ('native','planned','conditional','oracle')}
    for name,(group,start,end) in DRAWS.items():
        pool=inventory['pools'][name];rows=pool['rows'];lookup={str(r['row_index']):r for r in rows}
        expected_threshold=.012 if name in ('base_corridor','measured_corridor','planned_corridor') else .006 if name=='conditional_near' else None
        need(rows and len(lookup)==len(rows)==pool['row_count'] and pool['eligible_row_indices']==[r['row_index'] for r in rows] and
            pool['order_sha256']==rt.digest(rows) and pool['threshold_m']==expected_threshold,
            'Exact V7 pool rows/order/threshold required')
        for row in rows:
            need(type(row['row_index']) is int and type(row['valid_cells']) is int and 1<=row['valid_cells']<=20,
                'V7 row range and positive valid-cell count required')
            if group=='native':
                low,high=(0,inventory['base_train_windows']) if name.startswith('base') else (inventory['base_train_windows'],inventory['base_train_windows']+inventory['measured_windows'])
                need(low<=row['row_index']<high and np.isfinite(row['error_m']) and row['error_m']>=0 and row['actual_gap_m']>0 and
                    (expected_threshold is None or row['error_m']<=expected_threshold), 'V7 native row/source coordinate eligibility differs')
        counts=value['measured_draws'][name];row_counts=counts.get('rows',{});n=steps*(end-start)
        need(counts.get('draws')==counts.get('denominator')==n and row_counts and set(row_counts)<=set(lookup) and
            all(type(c) is int and c>0 for c in row_counts.values()) and sum(row_counts.values())==n,
            'Exact realized V7 subgroup draw counts/denominators required')
        labels={key:{} for key in LABELS};valid=0
        for index,count in row_counts.items():
            row=lookup[index];valid+=count*row['valid_cells']
            for key in LABELS:
                if key in row:labels[key][row[key]]=labels[key].get(row[key],0)+count
        need(all(counts.get(k)==v for k,v in labels.items()) and counts.get('valid_cells')==valid and counts.get('valid_joints')==valid*7,
            'V7 realized row/source/slot/identity/valid-cell counts must reconcile')
        for source,count in labels['source'].items():combined[group][source]=combined[group].get(source,0)+count
        if name=='conditional_near':
            old=value.get('conditional_terminal_draws',{})
            need(old.get('draws')==n and old.get('denominator')==steps*8 and
                all(old.get(k)==labels[k] for k in ('source','body','pose','target','anchor')), 'Original conditional4/8 counts differ')
    need(combined==value.get('gradient_source_examples'),'Every V7 optimized source must match subgroup realized counts')


def validate_binding(inventory,*,config,native,auxiliary_metadata,auxiliary_arrays,frozen_scope=None):
    """Check sealed metadata/row proofs only; no NPZ or historical trace reads."""
    complete,meta=admitted_metadata(config['measured_bank'],dataset=config['dataset'],normalization=inventory.get('normalization'))
    need(inventory.get('schema')==SCHEMA and inventory.get('data_version')==5 and inventory.get('model_optimizer_version')==7 and
        inventory.get('native_order')==list(NATIVE_ORDER) and set(inventory.get('pools',{}))==set(DRAWS) and
        all(inventory.get(k) is False for k in ('normalization_fitted','physical_gates_changed','labels_changed')),
        'Explicit bound sampler-only V7/data5 proof required')
    need(inventory.get('measured_bank')==config['measured_bank'] and inventory.get('measured_metadata')==complete['metadata'] and
        inventory.get('measured_arrays')==meta['arrays'] and inventory.get('measured_array_proofs')==meta['array_proofs'] and
        inventory.get('measured_index')==meta['index'] and inventory.get('measured_sources')==source_records(meta) and
        inventory.get('measured_windows')==meta['windows'] and inventory.get('base_counts')==native['counts'] and
        inventory.get('base_train_windows')==native['counts']['train'] and
        inventory.get('normalization_sha256')==rt.digest(rt.validate_normalization(rt.read_bound(meta['normalization']))),
        'Exact V7 separate bank/source/arrays/masks/base/statistics binding differs')
    if 'normalization' in native:need(native['normalization']==meta['normalization'],'Original Base29 normalization differs')
    corridor.validate_binding(inventory['corridors'],native=native,auxiliary_metadata=auxiliary_metadata,
        auxiliary_arrays=auxiliary_arrays,frozen_scope=frozen_scope)
    pools=inventory['pools'];base_count=inventory['base_train_windows']
    for name,records,offset,count in (('base_full',inventory['corridors']['pools']['native']['sources'],0,base_count),
        ('measured_full',inventory['measured_sources'],base_count,meta['windows'])):
        rows=pools[name]['rows']
        need([r['row_index'] for r in rows]==list(range(offset,offset+count)),'Exact full native subgroup row range required')
        lookup={r['source']:r for r in records}
        for row in rows:
            record=lookup.get(row.get('source'));need(record is not None,'Unknown V7 native source')
            need(all(row.get(k)==record.get(k) for k in (*LABELS,'actual_gap_m')) and
                offset+record['split_sample_range'][0]<=row['row_index']<offset+record['split_sample_range'][1] and
                type(row.get('valid_cells')) is int and 1<=row['valid_cells']<=20 and
                np.isfinite(row.get('error_m',np.nan)) and row['error_m']>=0,
                'V7 native actual row/source/gap/mask binding differs')
        for key in (name,name.replace('_full','_corridor')):
            selected=rows if key==name else [r for r in rows if r['error_m']<=.012]
            need(pools[key]==_pool(selected,threshold=None if key==name else .012,
                slots=bank.query.SLOT_IDS if name=='measured_full' else ()), 'Exact V7 full/corridor inventory differs')
        if name=='measured_full':
            need({r['slot'] for r in rows}==set(bank.query.SLOT_IDS) and
                {r['target'] for r in pools['measured_corridor']['rows']}=={'R3','L3'}, 'All4 full slots and both corridor targets required')
    sources=auxiliary_metadata['sources'];all_aux=[]
    mapping={'planned_ideal_tracking':'planned','conditional_ideal_continuation':'conditional','student_oracle_unexecuted':'oracle'}
    source_lookup={s['source_id']:s for s in sources}
    for group in ('planned','conditional','oracle'):
        rows=pools[group+'_full']['rows'];all_aux.extend(r['row_index'] for r in rows);counts={}
        for row in rows:
            source=source_lookup.get(row.get('source'))
            need(source is not None and mapping.get(source['kind'])==group and
                all(row.get(k)==source.get(field) for k,field in
                    (('body','subject_id'),('pose','pose_id'),('target','target_id'),('anchor','anchor'))) and
                type(row.get('valid_cells')) is int and 1<=row['valid_cells']<=20,
                'Exact V5 auxiliary full subgroup source/identity/mask required')
            counts[source['source_id']]=counts.get(source['source_id'],0)+1
        need(counts=={s['source_id']:len(s['windows']) for s in sources if mapping[s['kind']]==group},'Complete original auxiliary rows required')
        need(pools[group+'_full']==_pool(rows),'Exact auxiliary full-pool order required')
        if group!='oracle':
            allowed=set(inventory['corridors']['pools'][group]['eligible_row_indices'])
            name=group+('_near' if group=='conditional' else '_corridor')
            need(pools[name]==_pool([r for r in rows if r['row_index'] in allowed],threshold=.006 if group=='conditional' else .012),
                'Exact original planned/conditional corridor subset required')
    aux_count=inventory['auxiliary_array_proofs']['source_indices']['shape'][0]
    need(sorted(all_aux)==list(range(aux_count)),'Every V5 auxiliary row exactly once in its full group required')
    # Masks are sealed in both the archive proof and the per-window admission index.
    index=rt.read_bound(meta['index']);windows=index.get('samples',[])
    need(len(windows)==meta['windows'] and all(r['valid_cells']==w['window']['valid_future_cells']
        for r,w in zip(pools['measured_full']['rows'],windows)), 'Measured subgroup masks differ from admitted clean window index')
