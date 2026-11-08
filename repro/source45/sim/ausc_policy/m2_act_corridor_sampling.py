"""V6 sampling inventories over unchanged native and typed V5 TRAIN arrays.

Eligibility uses skin-relative state17 and each bound source's actual PREPOSE.
No physical threshold, target, label, normalization or data admission changes.
"""
from copy import deepcopy

import numpy as np

from . import m2_given_contact_runtime as rt
from . import m2_act_phase_data as phase

need=rt.need
SCHEMA='m2_act_corridor_sampling_inventory_v6'
DRAWS=dict(native=(16,32),planned=(4,8),conditional=(4,8))
LABELS=('source','body','pose','target','episode','anchor')


def native_metadata(complete,*,allowed_train_ids):
    """Gate the entire TRAIN identity/range table before reading any payload.

    Read only bound per-episode config and COMPLETE JSON, never captures, VAL
    arrays or future observations. No pose is inferred when absent in metadata.
    """
    episodes=complete.get('episodes');count=complete.get('counts',{}).get('train')
    need(isinstance(episodes,list) and type(count) is int and count>0,'Native TRAIN episode/count table required')
    selected=[e for e in episodes if e.get('split')=='train'];cursor=0;seen=set()
    for episode in selected:
        need(episode.get('subject_id') in allowed_train_ids,'Only admitted TRAIN identities may supply native corridor rows')
        eid=episode.get('episode_id');n=episode.get('windows');target=episode.get('target_id')
        need(isinstance(eid,str) and eid and eid not in seen and isinstance(target,str) and target and
            type(n) is int and n>0 and episode.get('split_sample_range')==[cursor,cursor+n],
            'Unique contiguous nonoverlapping full native TRAIN episode ranges required')
        cursor+=n;seen.add(eid)
    need(cursor==count,'Full native TRAIN coverage required')
    result=[]
    for episode in selected:
        for key in ('config','run_result'):rt._reference_shape(episode.get(key))
        config=rt.read_bound(episode['config']);run=rt.read_bound(episode['run_result'])
        route=config.get('m2_given_contact_control',{});meta=route.get('metadata',{})
        need(all(meta.get(k)==episode[k] for k in ('subject_id','split','target_id','episode_id')) and
            route.get('episode_id')==episode['episode_id'],'Native config/episode TRAIN identity and target binding differs')
        points=[p for p in run.get('points',[]) if p.get('target')==episode['target_id']]
        need(len(points)==1,'Exactly one actual native target point required')
        point=points[0];gap=point.get('phase_config',{}).get('precontact_gap_m')
        need(type(gap) in (int,float) and np.isfinite(gap) and gap>0 and
            point.get('checked_prepose_gap_m')==gap,'Actual target point phase/checked prepose gap must agree and be positive')
        entry=dict(source='native:'+episode['episode_id'],body=episode['subject_id'],target=episode['target_id'],
            episode=episode['episode_id'],split_sample_range=deepcopy(episode['split_sample_range']),actual_gap_m=float(gap),
            config=deepcopy(episode['config']),report=deepcopy(episode['run_result']))
        if 'pose_id' in episode:entry['pose']=episode['pose_id']
        if 'given_reference' in meta:
            reference=rt.pose.require_route(meta);fresh=reference.snapshot()['sources']['fresh_plan']
            plan_ref=rt.reference(fresh['file']);plan=rt.read_bound(plan_ref)
            need(plan_ref['sha256']==fresh['sha256'] and plan.get('name',plan.get('target'))==episode['target_id'] and
                plan.get('free_approach_start_gap_m')==gap,'Native canonical plan/target/actual point gap differs')
            entry['canonical_plan']=plan_ref
        result.append(entry)
    return result


def _pool(rows,*,threshold,requested=(),sources=(),arrays=None):
    rows=sorted(rows,key=lambda r:r['row_index']);buckets={key:{} for key in LABELS};present=set()
    for row in rows:
        for key in LABELS:
            if key in row:
                label=row[key];buckets[key][label]=buckets[key].get(label,0)+1
        if 'pose' in row:present.add((row['body'],row['pose'],row['target']))
    indices=[r['row_index'] for r in rows]
    need(len(indices)==len(set(indices)),'Unique eligible corridor row indices required')
    return dict(threshold_m=threshold,coordinate_error='norm(state17[7:10]+[0,0,actual_source_gap])',
        eligible_row_indices=indices,rows=rows,row_count=len(rows),source_count=len(buckets['source']),
        available_source_count=len(sources),missing_sources=sorted(s['source'] for s in sources if s['source'] not in buckets['source']),
        bucket_counts=buckets,missing_strata=[list(s) for s in sorted(set(requested)-present)],
        sources=deepcopy(sources),arrays=deepcopy(arrays),order_sha256=rt.digest(rows))


def native_inventory(samples,metadata,*,array_reference):
    states=np.asarray(samples['states']);histories=np.asarray(samples['histories'])
    count=metadata[-1]['split_sample_range'][1]
    need(states.shape==(count,17) and histories.shape==(count,2,17) and np.isfinite(states).all() and
        np.isfinite(histories).all() and np.array_equal(states,histories[:,1]),
        'Native current state must equal actual H2 current frame throughout full TRAIN population')
    rows=[]
    for source in metadata:
        start,end=source['split_sample_range'];errors=states[start:end,7:10]+[0,0,source['actual_gap_m']]
        for local,error in enumerate(np.linalg.norm(errors,axis=1)):
            if error<=.012:
                rows.append(dict(row_index=start+local,error_m=float(error),**{k:v for k,v in source.items() if k in LABELS},actual_gap_m=source['actual_gap_m']))
    need(rows,'Nonempty actual native 12mm corridor required')
    return _pool(rows,threshold=.012,sources=metadata,arrays=array_reference,
        requested=[(s['body'],s['pose'],s['target']) for s in metadata if 'pose' in s])


def auxiliary_inventory(auxiliary,*,kind,threshold,allowed_train_ids,array_reference=None,expected_strata=()):
    """Apply existing bound-plan checks without changing V5's 6mm default."""
    sources=auxiliary['sources'];phase._identities(sources,allowed_train_ids)
    samples=auxiliary['samples'];rows=[];requested=set(expected_strata);source_records=[]
    indices=np.asarray(samples['source_indices']);histories=np.asarray(samples['source_histories'])
    need(indices.ndim==1 and histories.shape==(len(indices),2,17) and np.isfinite(histories).all(),
        'Finite existing V5 source H2 arrays required')
    for i,source in enumerate(sources):
        if source.get('kind')!=kind:continue
        stratum=(source['subject_id'],source['pose_id'],source['target_id']);requested.add(stratum)
        gap,_,plan_ref=phase._plan(source['metadata'],source)
        source_rows=np.flatnonzero(indices==i)
        need(len(source_rows)==len(source['windows']),'Existing source/window inventory differs')
        labels=dict(source=source['source_id'],body=stratum[0],pose=stratum[1],target=stratum[2])
        if 'anchor' in source:labels['anchor']=source['anchor']
        source_records.append(dict(source_index=i,actual_gap_m=gap,plan=deepcopy(plan_ref),row_indices=source_rows.tolist(),**labels))
        errors=np.linalg.norm(histories[source_rows,1,7:10]+[0,0,gap],axis=1)
        for row,error in zip(source_rows,errors):
            if error<=threshold:rows.append(dict(row_index=int(row),source_index=i,actual_gap_m=gap,error_m=float(error),**labels))
    need(rows,'Nonempty '+kind+' corridor pool required')
    return _pool(rows,threshold=threshold,requested=requested,sources=source_records,arrays=array_reference)


def build_inventory(prepared,auxiliary):
    header=prepared['preparation']
    metadata=prepared.get('corridor_native_metadata')
    if metadata is None:metadata=native_metadata(header,allowed_train_ids=phase.scope.TRAIN16_IDS)
    reference=auxiliary.get('array_reference')
    strata=expected_strata(prepared.get('optimization_scope',{}))
    return dict(schema=SCHEMA,data_version=5,model_optimizer_version=6,
        expected_strata=[list(s) for s in strata],
        pools=dict(native=native_inventory(prepared['train'],metadata,array_reference=header['arrays']['train']),
            planned=auxiliary_inventory(auxiliary,kind='planned_ideal_tracking',threshold=.012,
                allowed_train_ids=phase.scope.TRAIN16_IDS,array_reference=reference,expected_strata=strata),
            conditional=auxiliary_inventory(auxiliary,kind='conditional_ideal_continuation',threshold=.006,
                allowed_train_ids=phase.scope.TRAIN16_IDS,array_reference=reference,expected_strata=strata)),
        normalization_fitted=False,physical_gates_changed=False,labels_changed=False)


def expected_strata(frozen):
    """Record missing declared strata; never require all sixteen bodies to exist."""
    return sorted((body,pose,target) for body in frozen.get('cohorts',{}).get('train',[])
        for pose in frozen.get('pose_ids',[]) for target in frozen.get('target_ids',[]))


def filtered_tree(tree,eligible):
    """Retain available body/pose/target/(anchor)/window strata only."""
    if isinstance(tree,list):return [row for row in tree if row in eligible]
    result={}
    for key,child in tree.items():
        subset=filtered_tree(child,eligible)
        if subset:result[key]=subset
    return result


def new_draws():
    return {name:dict(draws=0,denominator=0,rows={},**{key:{} for key in LABELS}) for name in DRAWS}


def record_draws(value,indices,inventory,*,lookups=None):
    for name,(biased,_) in DRAWS.items():
        counts=value['corridor_draws'][name];counts['denominator']+=len(indices[name])
        lookup=lookups[name] if lookups is not None else {row['row_index']:row for row in inventory['pools'][name]['rows']}
        for index in indices[name][:biased]:
            row=int(index);need(row in lookup,'Biased draw outside eligible corridor inventory')
            counts['draws']+=1;label=str(row);counts['rows'][label]=counts['rows'].get(label,0)+1
            for key in LABELS:
                if key in lookup[row]:
                    label=lookup[row][key];counts[key][label]=counts[key].get(label,0)+1


def validate_binding(inventory,*,native,auxiliary_metadata,auxiliary_arrays,frozen_scope=None):
    """Authenticate source/gap/array provenance without numerical data decoding."""
    need(inventory.get('schema')==SCHEMA and inventory.get('data_version')==5 and
        inventory.get('model_optimizer_version')==6 and set(inventory.get('pools',{}))==set(DRAWS) and
        all(inventory.get(key) is False for key in ('normalization_fitted','physical_gates_changed','labels_changed')),
        'Explicit sampler-only V6/data5 inventory required')
    strata=expected_strata(frozen_scope or {})
    need(inventory.get('expected_strata')==[list(s) for s in strata],'Declared corridor strata differ from frozen scope')
    metadata=native_metadata(native,allowed_train_ids=phase.scope.TRAIN16_IDS)
    need(inventory['pools']['native']['sources']==metadata and inventory['pools']['native']['arrays']==native['arrays']['train'],
        'Native corridor source/config/report/gap/array binding differs')
    sources=auxiliary_metadata['sources'];phase._identities(sources,phase.scope.TRAIN16_IDS)
    for name,kind in (('native',None),('planned','planned_ideal_tracking'),('conditional','conditional_ideal_continuation')):
        pool=inventory['pools'][name];records=pool['sources'];rows=pool['rows']
        requested={(s['body'],s['pose'],s['target']) for s in metadata if 'pose' in s} if name=='native' else set(strata)
        if name!='native':
            need(pool['arrays']==auxiliary_arrays,'V5 corridor array binding differs')
            expected=[(i,s) for i,s in enumerate(sources) if s['kind']==kind]
            need([r['source_index'] for r in records]==[i for i,_ in expected],'Exact available planned/conditional source inventory required')
            for record,(i,source) in zip(records,expected):
                gap,_,plan_ref=phase._plan(source['metadata'],source)
                need(record['source']==source['source_id'] and record['actual_gap_m']==gap and record['plan']==plan_ref and
                    all(record.get(key)==source.get(field) for key,field in
                        (('body','subject_id'),('pose','pose_id'),('target','target_id'),('anchor','anchor'))) and
                    len(record['row_indices'])==len(source['windows']) and len(set(record['row_indices']))==len(record['row_indices']),
                    'Actual V5 source/stratum/window/plan/gap binding differs')
                requested.add((source['subject_id'],source['pose_id'],source['target_id']))
        by_source={r['source']:r for r in records}
        for row in rows:
            source=by_source.get(row.get('source'))
            need(source is not None and all(row.get(key)==source.get(key) for key in (*LABELS,'actual_gap_m')),
                'Eligible corridor row/source labels or gap differ')
            need((source['split_sample_range'][0]<=row['row_index']<source['split_sample_range'][1]) if name=='native' else
                (row['row_index'] in source['row_indices'] and row.get('source_index')==source['source_index']),
                'Eligible corridor row outside actual source range')
        expected_pool=_pool(rows,threshold=.006 if name=='conditional' else .012,requested=requested,sources=records,arrays=pool['arrays'])
        need(pool==expected_pool,'Exact corridor row/source/bucket/missing-stratum/order inventory differs')


def validate_draws(value,*,steps,inventory):
    need(isinstance(inventory,dict) and inventory.get('schema')==SCHEMA and
        inventory.get('data_version')==5 and inventory.get('model_optimizer_version')==6 and
        set(inventory.get('pools',{}))==set(DRAWS),'Explicit bound V6 corridor inventory required')
    need(set(value.get('corridor_draws',{}))==set(DRAWS),'All realized corridor counters required')
    for name,(biased,total) in DRAWS.items():
        pool=inventory['pools'][name];rows=pool['rows'];lookup={str(r['row_index']):r for r in rows}
        need(pool['order_sha256']==rt.digest(rows) and pool['eligible_row_indices']==[r['row_index'] for r in rows] and
            len(lookup)==pool['row_count']==len(rows) and pool['source_count']==len({r['source'] for r in rows}) and
            pool['threshold_m']==(.006 if name=='conditional' else .012) and
            all(type(r['row_index']) is int and r['row_index']>=0 and np.isfinite(r['error_m']) and
                0<=r['error_m']<=pool['threshold_m'] and r['actual_gap_m']>0 for r in rows),
            'Corridor numerical eligibility/order/count binding differs')
        counts=value['corridor_draws'][name];row_counts=counts.get('rows',{})
        need(counts.get('draws')==steps*biased and counts.get('denominator')==steps*total and
            isinstance(row_counts,dict) and row_counts and set(row_counts)<=set(lookup) and
            all(type(n) is int and n>0 for n in row_counts.values()) and sum(row_counts.values())==steps*biased,
            'Exact realized eligible corridor row counts and denominator required')
        derived={key:{} for key in LABELS}
        for index,count in row_counts.items():
            for key in LABELS:
                if key in lookup[index]:
                    label=lookup[index][key];derived[key][label]=derived[key].get(label,0)+count
        need(all(counts.get(key)==expected for key,expected in derived.items()),
            'Corridor row/source/body/pose/target/episode/anchor counts must reconcile exactly')
        need(all(count<=value['gradient_source_examples'][name].get(source,0) for source,count in derived['source'].items()),
            'Corridor source draws exceed actual group source contributions')
        if name=='conditional':
            old=value.get('conditional_terminal_draws',{})
            need(all(old.get(key)==counts[key] for key in ('draws','denominator','source','body','pose','target','anchor')),
                'Unchanged conditional terminal and corridor counters must agree exactly')
