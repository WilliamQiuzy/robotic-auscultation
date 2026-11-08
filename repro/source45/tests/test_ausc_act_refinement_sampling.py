"""Invented CPU banks only. No historical payloads, model or Native execution."""
from copy import deepcopy
import importlib

import numpy as np
import pytest


def module():
    name='sim.ausc_policy.m2_act_refinement_sampling'
    assert importlib.util.find_spec(name), 'V9 three-bank sampler missing'
    return importlib.import_module(name)


def fixture(tmp_path,monkeypatch):
    from test_ausc_act_diverse_sampling import fixture as old_fixture
    from sim.ausc_policy import m2_act_refinement_data as data
    from sim.ausc_policy import m2_act_onpolicy_collection as collection
    p,a,banks=old_fixture(tmp_path,monkeypatch)
    # Materialize invented bound numerical files for independent V9 validation.
    p['train'].update(schema=data.given.MASKED_DATASET_SCHEMA,label_protocol=data.rt.shared.MASKED_LABEL_PROTOCOL,
        action_is_pad=np.zeros((len(p['train']['states']),20),dtype=bool))
    p['train']['origins']=p['train']['states'][:,:7].copy()
    with (tmp_path/'v9-native-TRAIN.npz').open('xb') as stream:
        np.savez_compressed(stream,**{k:p['train'][k] for k in data.ARRAY_KEYS})
    p['preparation']['arrays']['train']=data.rt.reference(tmp_path/'v9-native-TRAIN.npz')
    a['samples']['source_times']=np.tile([0.,.05],(len(a['samples']['source_indices']),1))
    a['samples']['source_histories'][0,:,7]=.013
    with (tmp_path/'v9-auxiliary.npz').open('xb') as stream:
        np.savez_compressed(stream,**{k:a['samples'][k] for k in module().corridor.phase.ARRAY_KEYS})
    a['array_reference']=data.rt.reference(tmp_path/'v9-auxiliary.npz')
    q=banks['query32'];d=banks.pop('dart37');banks['dart_enriched']=d
    q['metadata'].update(schema=data.QUERY_SCHEMA,profile=data.QUERY_PROFILE)
    q['metadata']['optimization_admission']=dict(data_optimization_eligible=True)
    for name,b in list(banks.items()):
        if name=='query32':continue
        meta=b['metadata'];meta.update(schema=data.ENRICHED_SCHEMA,profile=data.ENRICHED_PROFILE,
            precollection=meta['precollection_plan'])
        old=meta['optimization_admission'];meta['optimization_admission']=dict(old,
            data_optimization_eligible=True,benchmark_act_eligible=False)
    o=deepcopy(d);o['metadata'].update(schema=data.ONPOLICY_SCHEMA,profile=data.ONPOLICY_PROFILE)
    requests=collection.planned_requests();o['metadata']['attempt_ledger']=[];o['metadata']['sources']=[]
    for i,request in enumerate(requests):
        row=deepcopy(d['metadata']['sources'][i]);row['slot_id']=request['slot_id']
        for key in ('subject_id','split','target_id','pose_id'):
            row['metadata'][key]=request[key];row['declaration'][key]=request[key]
        row['metadata']['episode_id']=row['declaration']['episode_id']='onpolicy-fixture-'+str(i)
        row.update(split_sample_range=[i*3,i*3+3])
        o['metadata']['sources'].append(row)
        o['metadata']['attempt_ledger'].append(dict(request=request,status='qualified',windows=3))
    o['metadata']['optimization_admission']=dict(data_optimization_eligible=True)
    for key in data.ARRAY_KEYS:o['samples'][key]=o['samples'][key][:24].copy()
    o['metadata']['windows']=24;banks['onpolicy8']=o
    d['samples']['states'][:3,7]=.02;d['samples']['histories'][:3,:,7]=.02
    for name,b in banks.items():
        meta=b['metadata'];samples=b['samples'];samples.update(bank_schema=meta['schema'],
            normalization_reference=meta['normalization'])
        samples['derived_index']['windows']=[dict(label_role='actual_clean_recovery_teacher_raw_intents',
            valid_future_cells=int((~mask).sum()),inclusive_native_ticks=[12*i+23,12*i+35])
            for i,mask in enumerate(samples['action_is_pad'])]
        items=[]
        for source in meta['sources']:
            lo,hi=source['split_sample_range']
            items.extend(dict(sample_index=i,slot_id=source['slot_id'],window=samples['derived_index']['windows'][i]) for i in range(lo,hi))
        with (tmp_path/(name+'-TRAIN.npz')).open('xb') as stream:
            np.savez_compressed(stream,**{k:samples[k] for k in data.ARRAY_KEYS})
        meta.update(arrays=data.rt.reference(tmp_path/(name+'-TRAIN.npz')),
            index=data.rt.write_new(tmp_path/(name+'-index.json'),dict(samples=items,order_sha256=data.rt.digest(items))),
            array_proofs={k:data.dart.measured.array_proof(samples[k]) for k in data.ARRAY_KEYS})
        b['complete'].update(schema=meta['schema'],profile=meta['profile'],complete=True,arrays=meta['arrays'],
            optimization_admission=meta['optimization_admission'],metadata=data.rt.write_new(tmp_path/(name+'-metadata.json'),meta))
        b['reference']=data.rt.write_new(tmp_path/(name+'-complete.json'),b['complete'])
    monkeypatch.setattr(data,'metadata',lambda ref:next((b['complete'],b['metadata']) for b in banks.values() if ref==b['reference']))
    monkeypatch.setattr(data,'load_arrays',lambda ref:next((b['samples'],b['metadata']) for b in banks.values() if ref==b['reference']))
    # Real source record and numerical packing; data admission is separately tested.
    monkeypatch.setattr(module(),'_admission',lambda meta:meta['optimization_admission'])
    p['measured_banks']=banks
    return p,a,banks


def cache_fixture(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_act_auxiliary_train as train
    p,a,banks=fixture(tmp_path,monkeypatch)
    stripped=dict(p);stripped.pop('measured_banks')
    cache=train.build_cache(stripped,a,optimizer_version=5)
    return module().configure_cache(cache,p,a),p,a,banks


def test_v9_requires_exact_three_positive_banks_before_arrays(tmp_path,monkeypatch):
    m=module();p,a,b=fixture(tmp_path,monkeypatch);refs={n:x['reference'] for n,x in b.items()}
    assert set(m.load_banks(refs,dataset=m.query.BASE29,normalization=b['query32']['metadata']['normalization']))==set(m.BANKS)
    calls=[];monkeypatch.setattr(m.data,'load_arrays',lambda ref:calls.append(ref))
    for fault in ('missing','old','negative','identity','normalization','source_order','ref'):
        bad=deepcopy(refs);meta=b['dart_enriched']['metadata'];saved=deepcopy(meta)
        if fault=='missing':bad.pop('onpolicy8')
        elif fault=='old':bad['dart37']=bad.pop('dart_enriched')
        elif fault=='negative':meta['optimization_admission']['data_optimization_eligible']=False
        elif fault=='identity':meta['sources'][0]['metadata']['split']='validation'
        elif fault=='normalization':meta['normalization']=dict(meta['normalization'],sha256='0'*64)
        elif fault=='source_order':meta['sources'].reverse()
        else:bad['query32'].pop('bytes')
        with pytest.raises(ValueError):m.load_banks(bad,dataset=m.query.BASE29,normalization=b['query32']['metadata']['normalization'])
        assert not calls
        meta.clear();meta.update(saved)


def test_three_banks_pack_masks_balanced_draws_absent_slots_and_exposure(tmp_path,monkeypatch):
    m=module();cache,p,a,banks=cache_fixture(tmp_path,monkeypatch);inv=cache['refinement_inventory']
    assert inv['model_optimizer_version']==9 and set(inv['measured_banks'])==set(m.BANKS)
    assert inv['group_weights']==dict(native=.5,planned=.125,conditional=.125,oracle=.25)
    assert len(inv['physical_measured_subject_ids'])==6
    assert inv['physical_measured_subject_ids']!=inv['computed_auxiliary_subject_ids']
    assert 'dart_enriched:dart-00' in inv['pools']['measured_corridor']['missing_slots']
    assert not inv['pools']['measured_full']['missing_slots']
    from sim.ausc_policy import m2_act_auxiliary_train as train
    rng=np.random.default_rng(87);value=train.new_optimization(version=5);value['refinement_draws']=m.new_draws()
    # Exercise all declared pools; native uses the target/body/available-pose hierarchy.
    for _ in range(200):
        ids={'native':m.native_indices(rng,cache)}
        for group in ('planned','conditional','oracle'):
            ids[group]=np.concatenate([rng.choice(inv['pools'][n]['eligible_row_indices'],size=end-start)
                for n,(g,start,end) in m.DRAWS.items() if g==group])
        m.record_draws(value,cache,ids)
        train.record_draws(value,cache,ids)
    m.validate_draws(value,steps=200,inventory=inv)
    assert set(value['refinement_draws']['measured_full']['bank'])==set(m.BANKS)
    bad=deepcopy(value);bad['refinement_draws']['measured_full']['valid_cells']+=1
    with pytest.raises(ValueError):m.validate_draws(bad,steps=200,inventory=inv)
    config=dict(dataset=m.query.BASE29,measured_banks={n:b['reference'] for n,b in banks.items()})
    kwargs=dict(config=config,native=p['preparation'],auxiliary_metadata=dict(sources=a['sources']),auxiliary_arrays=a.get('array_reference'))
    m.validate_binding(inv,**kwargs)
    for fault in ('mask','order','gap','absent','source','range'):
        bad=deepcopy(inv)
        if fault=='mask':bad['pools']['measured_full']['rows'][0]['valid_cells']+=1
        elif fault=='order':bad['pools']['measured_full']['rows'].reverse()
        elif fault=='gap':bad['pools']['measured_full']['rows'][0]['actual_gap_m']+=.001
        elif fault=='absent':bad['pools']['measured_corridor']['missing_slots']=['invented']
        elif fault=='source':bad['pools']['measured_full']['rows'][0]['source']='invented'
        else:bad['measured_banks']['query32']['composite_sample_range'][0]+=1
        with pytest.raises(ValueError):m.validate_binding(bad,**kwargs)


def test_target_body_available_pose_then_bank_slot_is_not_window_weighted():
    m=module();rows=[]
    for target in ('R3','L3'):
        for body,n,poses in (('small',1,('a',)),('large',99,('a','b'))):
            for pose in poses:
                for bank in m.BANKS:
                    for _ in range(n):rows.append(dict(row_index=len(rows),target=target,body=body,pose=pose,bank=bank,slot='slot'))
    rng=np.random.default_rng(3);tree=m.measured_tree(rows);counts={b:0 for b in ('small','large')}
    for _ in range(10000):counts[rows[m.draw_measured(rng,tree)]['body']]+=1
    assert 4700<counts['small']<5300


def test_duplicate_raw_episode_windows_and_empty_mandatory_target_reject(tmp_path,monkeypatch):
    m=module();p,a,banks=fixture(tmp_path,monkeypatch)
    banks['onpolicy8']['metadata']['sources'][0]['metadata']['episode_id']=banks['query32']['metadata']['sources'][0]['metadata']['episode_id']
    banks['onpolicy8']['metadata']['sources'][0]['declaration']['episode_id']=banks['query32']['metadata']['sources'][0]['metadata']['episode_id']
    with pytest.raises(ValueError):m.admitted_metadata({n:b['reference'] for n,b in banks.items()},dataset=m.query.BASE29,normalization=banks['query32']['metadata']['normalization'])
    with pytest.raises(ValueError):m._mandatory_targets([dict(target='R3')])


def test_all_three_lightweight_admissions_precede_deeper_consumer_api(tmp_path,monkeypatch):
    m=module();p,a,banks=fixture(tmp_path,monkeypatch);bad=banks['onpolicy8']
    meta=deepcopy(bad['metadata']);meta['optimization_admission']['data_optimization_eligible']=False
    complete=deepcopy(bad['complete']);complete['metadata']=m.rt.write_new(tmp_path/'negative-meta.json',meta)
    refs={n:b['reference'] for n,b in banks.items()};refs['onpolicy8']=m.rt.write_new(tmp_path/'negative-complete.json',complete)
    monkeypatch.setattr(m.data,'metadata',lambda ref:pytest.fail('Negative mandatory bank reached deeper consumer API'))
    with pytest.raises(ValueError):m.admitted_metadata(refs,dataset=m.query.BASE29,normalization=banks['query32']['metadata']['normalization'])


def test_actual_v9_admission_requires_all17_original_dart_and_all8_onpolicy():
    m=module()
    from test_ausc_measured_dart_data import qualified,preparation
    rows=qualified(m.dart);plan=preparation(m.dart,(4,5,6))
    for i in (4,5,6):rows[i].update(status='cpu_unavailable',windows=0)
    meta=dict(schema=m.data.ENRICHED_SCHEMA,attempt_ledger=rows,precollection=plan)
    assert m._admission(meta)['data_optimization_eligible']
    rows[0].update(status='physical_negative',windows=0)
    assert not m._admission(meta)['data_optimization_eligible']
    rows=[]
    for i,request in enumerate(m.collection.planned_requests()):
        identity=dict(pid=i+1,pgid=i+1,sid=i+1,start_ticks=str(i+1))
        rows.append(dict(request=request,status='qualified',windows=1,infrastructure_contamination=False,native_reset_identity=identity))
    assert m._admission(dict(schema=m.data.ONPOLICY_SCHEMA,attempt_ledger=rows))['data_optimization_eligible']
    rows[-1]['status']='preview_rejected'
    assert not m._admission(dict(schema=m.data.ONPOLICY_SCHEMA,attempt_ledger=rows))['data_optimization_eligible']


def test_missing_global_measured_corridor_target_cannot_be_invented(tmp_path,monkeypatch):
    m=module();p,a,banks=fixture(tmp_path,monkeypatch)
    for name,bank in banks.items():
        meta=bank['metadata'];samples=bank['samples']
        for source in meta['sources']:
            if source['metadata']['target_id']!='L3':continue
            lo,hi=source['split_sample_range'];samples['states'][lo:hi,7]=.04;samples['histories'][lo:hi,:,7]=.04
        meta['array_proofs']={k:m.query.array_proof(samples[k]) for k in m.query.ARRAY_KEYS}
        bank['complete']['metadata']=m.rt.write_new(tmp_path/(name+'-noL3meta.json'),meta)
        bank['reference']=m.rt.write_new(tmp_path/(name+'-noL3complete.json'),bank['complete'])
    from sim.ausc_policy import m2_act_auxiliary_train as train
    stripped=dict(p);stripped.pop('measured_banks');cache=train.build_cache(stripped,a,optimizer_version=5)
    with pytest.raises(ValueError,match='target'):m.configure_cache(cache,p,a)


def test_prepared_window_semantics_must_match_exact_admitted_index(tmp_path,monkeypatch):
    m=module();p,a,banks=fixture(tmp_path,monkeypatch)
    banks['dart_enriched']['samples']['derived_index']['windows'].reverse()
    from sim.ausc_policy import m2_act_auxiliary_train as train
    stripped=dict(p);stripped.pop('measured_banks');cache=train.build_cache(stripped,a,optimizer_version=5)
    with pytest.raises(ValueError,match='index|order|semantics'):m.configure_cache(cache,p,a)


@pytest.mark.parametrize('group',['base','measured','planned','conditional'])
def test_rebuilt_self_consistent_pools_cannot_promote_unobserved_corridor_rows(tmp_path,monkeypatch,group):
    m=module();cache,p,a,banks=cache_fixture(tmp_path,monkeypatch);bad=deepcopy(cache['refinement_inventory'])
    config=dict(dataset=m.query.BASE29,measured_banks={n:b['reference'] for n,b in banks.items()})
    kwargs=dict(config=config,native=p['preparation'],auxiliary_metadata=dict(sources=a['sources']),auxiliary_arrays=a['array_reference'])
    m.validate_binding(bad,**kwargs)
    if group in ('base','measured'):
        full=bad['pools'][group+'_full']['rows'];outside=next(r for r in full if r['error_m']>.012)
        assert outside['error_m']>=.013
        outside['error_m']=0.
        slots=[r['bank']+':'+r['slot'] for r in bad['pools']['measured_full']['rows']] if group=='measured' else ()
        bad['pools'][group+'_full']=m._pool(full,slots=slots)
        bad['pools'][group+'_corridor']=m._pool([r for r in full if r['error_m']<=.012],threshold=.012,slots=slots)
    else:
        pool=bad['corridors']['pools'][group];allowed=set(pool['eligible_row_indices'])
        full=bad['pools'][group+'_full']['rows'];outside=next(r for r in full if r['row_index'] not in allowed)
        source=next(s for s in pool['sources'] if outside['row_index'] in s['row_indices'])
        # Rebuild every self-reported row digest/count/missing-stratum field.
        row=dict(row_index=outside['row_index'],error_m=0.,**{k:v for k,v in source.items() if k in (*m.corridor.LABELS,'actual_gap_m','source_index')})
        requested=[(s['body'],s['pose'],s['target']) for s in pool['sources']]
        threshold=.006 if group=='conditional' else .012
        fake=m.corridor._pool(pool['rows']+[row],threshold=threshold,requested=requested,sources=pool['sources'],arrays=pool['arrays'])
        bad['corridors']['pools'][group]=fake
        name=group+('_near' if group=='conditional' else '_corridor')
        bad['pools'][name]=m._pool([r for r in full if r['row_index'] in fake['eligible_row_indices']],threshold=threshold)
    with pytest.raises(ValueError,match='observed|corridor|pool|array'):m.validate_binding(bad,**kwargs)
