"""V8 synthetic bank/sampling contracts. No actual capture or model is opened."""
from copy import deepcopy
import importlib

import numpy as np
import pytest

from sim.ausc_policy import m2_act_auxiliary_train as train
from sim.ausc_policy import m2_measured_recovery_data as query
from sim.ausc_policy import m2_measured_dart_data as dart


def module():
    name='sim.ausc_policy.m2_act_diverse_sampling'
    assert importlib.util.find_spec(name), 'Explicit separate V8 diverse sampler required'
    return importlib.import_module(name)


def fixture(tmp_path,monkeypatch):
    from test_ausc_act_training_v7 import v7_cache
    from test_ausc_measured_dart_data import qualified,preparation
    p,a=v7_cache(tmp_path,monkeypatch);q=p.pop('measured');qm=q['metadata'];qs=q['samples']
    qm.update(schema=query.SCHEMA_V2,profile=query.PROFILE_V2,
        evidence_profile=query.teacher.EVIDENCE_PROFILE,recorded_source=deepcopy(query.teacher.SOURCE32))
    gate=query.admission(qm['attempt_ledger'],profile=query.PROFILE_V2)
    qm['optimization_admission']=gate;qs.update(bank_schema=query.SCHEMA_V2,optimization_admission=deepcopy(gate))
    q['complete'].update(schema=query.SCHEMA_V2,profile=query.PROFILE_V2,complete=True,optimization_admission=deepcopy(gate))
    q['complete']['metadata']=query.rt.write_new(tmp_path/'query-metadata-v8.json',qm)
    q['reference']=query.rt.write_new(tmp_path/'query-complete-v8.json',q['complete'])
    ledger=qualified(dart);sources=[];arrays={k:[] for k in query.ARRAY_KEYS}
    for i,entry in enumerate(ledger):
        row=deepcopy(qm['sources'][i%4]);request=entry['request'];meta=row['metadata']
        for key in ('split','subject_id','pose_id','target_id'):meta[key]=request[key]
        meta['episode_id']='dart-fixture-'+str(i)
        gap=.035;target=request['target_id']
        meta['fixture_plan']=query.rt.write_new(tmp_path/('dart-plan-'+str(i)+'.json'),dict(name=target,free_approach_start_gap_m=gap))
        config=query.rt.write_new(tmp_path/('dart-config-'+str(i)+'.json'),dict(metadata=meta))
        report=query.rt.write_new(tmp_path/('dart-report-'+str(i)+'.json'),dict(points=[dict(target=target,checked_prepose_gap_m=gap,phase_config=dict(precontact_gap_m=gap))]))
        row.update(slot_id=request['slot_id'],metadata=meta,declaration=dict(meta,config=config,run_result=report),
            windows=3,split_sample_range=[i*3,i*3+3])
        entry.update(windows=3,episode=row['declaration']);sources.append(row)
        for key in arrays:arrays[key].append(qs[key][:3].copy())
        arrays['states'][-1][:,9]=-gap;arrays['histories'][-1][:,:,9]=-gap
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    dg=dart.admission(ledger,precollection=preparation(dart))
    ds={**{k:deepcopy(v) for k,v in qs.items() if k not in query.ARRAY_KEYS},**arrays}
    ds.update(bank_schema=dart.SCHEMA,optimization_admission=deepcopy(dg),derived_index=dict(windows=[dict(label_role='actual_clean_recovery_teacher_raw_intents',valid_future_cells=int((~m).sum())) for m in arrays['action_is_pad']]))
    dm=dict(qm,schema=dart.SCHEMA,profile=dart.PROFILE_SOURCE37,evidence_profile=dart.teacher.DART37_EVIDENCE_PROFILE,
        recorded_source=deepcopy(dart.teacher.SOURCE37),sources=sources,windows=60,attempt_ledger=ledger,
        precollection_plan=preparation(dart),optimization_admission=deepcopy(dg),
        arrays=dict(path='/synthetic/dart.npz',bytes=1,sha256='d'*64),array_proofs={k:query.array_proof(v) for k,v in arrays.items()},
        index=query.rt.write_new(tmp_path/'dart-index-v8.json',dict(samples=[dict(window=w) for w in ds['derived_index']['windows']])))
    dc=dict(schema=dart.SCHEMA,profile=dart.PROFILE_SOURCE37,complete=True,optimization_admission=deepcopy(dg),metadata=query.rt.write_new(tmp_path/'dart-metadata-v8.json',dm))
    d=dict(reference=query.rt.write_new(tmp_path/'dart-complete-v8.json',dc),complete=dc,metadata=dm,samples=ds)
    banks=dict(query32=q,dart37=d);p['measured_banks']=banks
    p['preparation']['normalization']=qm['normalization']
    p['preparation']['index']=query.rt.write_new(tmp_path/'native-index-v8.json',dict(samples=[dict(split='train',sample_index=i,
        episode_id=p['preparation']['episodes'][i//3]['episode_id'],window=dict(valid_future_cells=int(mask.sum()))) for i,mask in enumerate(query.rt.shared.valid_action_mask(p['train']))]))
    monkeypatch.setattr(query,'metadata',lambda ref:(q['complete'],q['metadata']))
    monkeypatch.setattr(dart,'metadata',lambda ref:(d['complete'],d['metadata']))
    # Synthetic origins and index format are deliberately not claimed as Native.
    monkeypatch.setattr(dart,'validate_samples',query.validate_samples)
    return p,a,banks


def test_required_typed_banks_and_full_admission_before_numerical_payload(tmp_path,monkeypatch):
    m=module();p,a,banks=fixture(tmp_path,monkeypatch);refs={k:b['reference'] for k,b in banks.items()}
    calls=[]
    monkeypatch.setattr(query,'load_arrays',lambda ref:calls.append('query') or (banks['query32']['samples'],banks['query32']['metadata']))
    monkeypatch.setattr(dart,'load_arrays',lambda ref:calls.append('dart') or (banks['dart37']['samples'],banks['dart37']['metadata']))
    result=m.load_banks(refs,dataset=query.BASE29,normalization=banks['query32']['metadata']['normalization'])
    assert set(result)=={'query32','dart37'} and calls==['query','dart']
    for fault in ('missing_query','old_dart','pending','negative','below95_17','norm','validation','fresh','unknown','source','reordered','smoke'):
        calls.clear();bad=deepcopy(refs);meta=banks['dart37']['metadata'];saved=deepcopy(meta)
        if fault=='missing_query':bad.pop('query32')
        elif fault=='old_dart':meta['profile']=dart.PROFILE
        elif fault=='pending':meta['attempt_ledger'][0]['status']='pending'
        elif fault=='negative':meta['attempt_ledger'][0].update(status='physical_negative',windows=0);meta['attempt_ledger'][1].update(status='physical_negative',windows=0)
        elif fault=='below95_17':
            from test_ausc_measured_dart_data import preparation
            meta['precollection_plan']=preparation(dart,(1,5,13))
            for i in (1,5,13):meta['attempt_ledger'][i].update(status='cpu_unavailable',windows=0)
            meta['attempt_ledger'][0].update(status='physical_negative',windows=0)
        elif fault=='norm':meta['normalization']=dict(meta['normalization'],sha256='f'*64)
        elif fault=='validation':meta['sources'][-1]['metadata'].update(subject_id='p000923',split='validation')
        elif fault=='fresh':meta['sources'][-1]['metadata'].update(subject_id='p000519',split='validation')
        elif fault=='source':meta['recorded_source']=deepcopy(dart.teacher.SOURCE34)
        elif fault=='unknown':meta['sources'][-1]['metadata']['subject_id']='unknown'
        elif fault=='reordered':meta['sources'].reverse()
        else:meta['single_episode_only']=True
        with pytest.raises(ValueError):m.load_banks(bad,dataset=query.BASE29,normalization=banks['query32']['metadata']['normalization'])
        assert not calls
        meta.clear();meta.update(saved)


def test_v8_balanced_composite_masks_exposure_and_absent_pools(tmp_path,monkeypatch):
    m=module();p,a,banks=fixture(tmp_path,monkeypatch)
    # A real-source pool can be absent in the corridor while retained in full.
    samples=banks['dart37']['samples'];samples['states'][:3,7]=.02;samples['histories'][:3,:,7]=.02
    banks['dart37']['metadata']['array_proofs']={k:query.array_proof(samples[k]) for k in query.ARRAY_KEYS}
    before=deepcopy(p)
    cache=train.build_cache(p,a,optimizer_version=8);proof=cache['diverse_inventory']
    assert proof['model_optimizer_version']==8 and proof['data_version']==5
    assert proof['measured_windows']==72 and cache['base_native_count']==9
    assert 'dart37:dart-00' in proof['pools']['measured_corridor']['missing_slots']
    for i,b in enumerate(banks.values()):
        start=9+(0 if i==0 else 12);end=start+len(b['samples']['states'])
        np.testing.assert_array_equal(cache['packed']['native']['action_is_pad'][start:end],b['samples']['action_is_pad'])
    left=np.random.default_rng(41);right=np.random.default_rng(41);value=train.new_optimization(version=8)
    for _ in range(300):
        ids=train.sample_cached_indices(left,cache);same=train.sample_cached_indices(right,cache)
        assert {k:len(v) for k,v in ids.items()}==dict(native=32,planned=8,conditional=8,oracle=16)
        assert all(np.array_equal(ids[k],same[k]) for k in ids)
        train.record_draws(value,cache,ids)
    m.validate_draws(value,steps=300,inventory=proof)
    draws=value['diverse_draws']['measured_full']
    assert set(draws['bank'])=={'query32','dart37'} and len(draws['body'])==6 and len(draws['pose'])==2
    assert len(draws['source'])==24 and sum(draws['body'].values())==2400
    assert draws['valid_cells']!=draws['draws']*20
    for key in query.ARRAY_KEYS:np.testing.assert_array_equal(p['measured_banks']['dart37']['samples'][key],before['measured_banks']['dart37']['samples'][key])
    bad=deepcopy(value);bad['diverse_draws']['measured_full']['bank']={'query32':2400}
    with pytest.raises(ValueError):m.validate_draws(bad,steps=300,inventory=proof)
    # Order/mask proofs cannot be recomputed from a corrupt cache silently.
    samples['actions'][0,0,0]=.1
    with pytest.raises(ValueError,match='proof|bytes'):train.build_cache(p,a,optimizer_version=8)


def test_hierarchy_is_target_body_pose_slot_window_not_window_weighted():
    m=module();rows=[]
    for target in ('R3','L3'):
        for body,n in (('small',1),('large',100)):
            for pose in ('a','b'):
                for i in range(n):rows.append(dict(row_index=len(rows),target=target,body=body,pose=pose,bank='dart37',slot='slot'))
    tree=m.measured_tree(rows);rng=np.random.default_rng(13);counts={key:0 for key in ('small','large')}
    lookup={r['row_index']:r for r in rows}
    for _ in range(12000):counts[lookup[m.draw_measured(rng,tree)]['body']]+=1
    assert 5700<counts['small']<6300


def test_bound_inventory_rejects_source_mask_range_and_bank_corruption_without_payloads(tmp_path,monkeypatch):
    m=module();p,a,banks=fixture(tmp_path,monkeypatch);a['array_reference']=dict(path='/synthetic/aux.npz',bytes=1,sha256='c'*64)
    cache=train.build_cache(p,a,optimizer_version=8);proof=cache['diverse_inventory']
    config=dict(dataset=query.BASE29,measured_banks={name:b['reference'] for name,b in banks.items()})
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Deployment reopened raw bank arrays'))
    kwargs=dict(config=config,native=p['preparation'],auxiliary_metadata=dict(sources=a['sources']),auxiliary_arrays=a['array_reference'])
    m.validate_binding(proof,**kwargs)
    for fault in ('mask','base_mask','bank','range','source','missing','scope','gap'):
        bad=deepcopy(proof)
        if fault=='mask':bad['pools']['measured_full']['rows'][0]['valid_cells']=20
        elif fault=='base_mask':
            for pool in ('base_full','base_corridor'):
                bad['pools'][pool]['rows'][0]['valid_cells']=1
                bad['pools'][pool]['order_sha256']=query.rt.digest(bad['pools'][pool]['rows'])
        elif fault=='bank':bad['measured_banks']['dart37']['profile']=dart.PROFILE
        elif fault=='range':bad['measured_banks']['dart37']['composite_sample_range'][0]+=1
        elif fault=='source':bad['pools']['measured_full']['rows'][0]['source']='native:invented'
        elif fault=='missing':bad['pools']['measured_full']['rows'].pop()
        elif fault=='scope':bad['physical_measured_subject_ids']=bad['computed_auxiliary_subject_ids']
        else:bad['pools']['measured_corridor']['rows'][0]['error_m']=.035
        with pytest.raises(ValueError):m.validate_binding(bad,**kwargs)


def test_v8_requires_full_both_bank_references_before_metadata(tmp_path,monkeypatch):
    m=module();p,a,banks=fixture(tmp_path,monkeypatch);refs={k:b['reference'] for k,b in banks.items()}
    refs['dart37'].pop('bytes')
    monkeypatch.setattr(dart,'metadata',lambda *a:pytest.fail('Missing full reference reached deeper metadata'))
    with pytest.raises(ValueError):m.admitted_metadata(refs,dataset=query.BASE29,normalization=banks['query32']['metadata']['normalization'])


def test_v8_keeps_all_original_base_targets_while_requiring_both_recovery_targets(tmp_path,monkeypatch):
    m=module();p,a,banks=fixture(tmp_path,monkeypatch);episode=p['preparation']['episodes'][2]
    config=query.rt.read_bound(episode['config']);report=query.rt.read_bound(episode['run_result'])
    config['m2_given_contact_control']['metadata']['target_id']='R1';report['points'][0]['target']='R1';episode['target_id']='R1'
    episode['config']=query.rt.write_new(tmp_path/'base-extra-target-config.json',config)
    episode['run_result']=query.rt.write_new(tmp_path/'base-extra-target-report.json',report)
    cache=train.build_cache(p,a,optimizer_version=8)
    assert {row['target'] for row in cache['diverse_inventory']['pools']['base_full']['rows']}=={'R1','R3','L3'}
