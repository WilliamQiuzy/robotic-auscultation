"""Synthetic V7 integration; no SDK, real labels, weights or GPU execution."""
from copy import deepcopy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy import m2_act_auxiliary_train as train
from sim.ausc_policy import m2_measured_recovery_data as bank


def sampler():
    name='sim.ausc_policy.m2_act_measured_sampling'
    assert importlib.util.find_spec(name), 'Separate V7 measured sampler required'
    return importlib.import_module(name)


def measured_fixture(tmp_path,monkeypatch,prepared):
    from test_ausc_measured_recovery_data import qualified_ledger
    from sim.ausc_policy import m2_given_contact_act as g
    n=12;states=np.zeros((n,17));sources=[]
    gaps=(.035,.047,.055,.029)
    for i,slot in enumerate(bank.query.SLOT_IDS):
        target='R3' if i<2 else 'L3';episode='measured-'+str(i)
        meta=dict(subject_id='p001120',split='train',pose_id=train.runtime_scope_poses()[0],target_id=target,episode_id=episode)
        plan=bank.rt.write_new(tmp_path/('plan-m'+str(i)+'.json'),dict(name=target,free_approach_start_gap_m=gaps[i]))
        report=bank.rt.write_new(tmp_path/('report-m'+str(i)+'.json'),dict(points=[dict(target=target,
            checked_prepose_gap_m=gaps[i],phase_config=dict(precontact_gap_m=gaps[i]))]))
        config=bank.rt.write_new(tmp_path/('config-m'+str(i)+'.json'),dict(metadata=meta))
        sources.append(dict(slot_id=slot,metadata=meta,declaration=dict(meta,config=config,run_result=report),receipt={},
            windows=3,split_sample_range=[i*3,i*3+3]))
        meta['fixture_plan']=plan
        states[i*3:i*3+3,7:10]=[[0,0,-gaps[i]],[.011,0,-gaps[i]],[.013,0,-gaps[i]]]
    # Bound canonical plans are synthetic; all sampler arithmetic and packers are real.
    monkeypatch.setattr(bank.rt.pose,'require_route',lambda meta:SimpleNamespace(snapshot=lambda:dict(
        target_id=meta['target_id'],sources={'fresh_plan':dict(file=meta['fixture_plan']['path'],sha256=meta['fixture_plan']['sha256'])})))
    mask=np.tile(np.arange(20)>=4,(n,1));mask[1::3]=np.arange(20)>=1
    samples=dict(schema=g.MASKED_DATASET_SCHEMA,task_id=bank.rt.pose.TASK_ID,action_semantics_id=g.ACTION_SEMANTICS_ID,
        observation_contract=bank.rt.pose.observation_contract(),label_protocol=bank.rt.shared.MASKED_LABEL_PROTOCOL,
        states=states,histories=np.stack([states,states],axis=1),actions=np.zeros((n,20,7)),origins=states[:,:7].copy(),
        velocities=np.zeros((n,7)),observation_times=np.arange(n,dtype=float),action_is_pad=mask,
        derived_index={'windows':[dict(label_role='actual_clean_recovery_teacher_raw_intents',valid_future_cells=int((~m).sum())) for m in mask]})
    norm=bank.rt.write_new(tmp_path/'base-norm.json',prepared['normalization'])
    ref=bank.rt.write_new(tmp_path/'measured.json',{'fixture':True})
    samples.update(bank_schema=bank.SCHEMA,normalization_reference=norm,measured_bank=ref,
        optimization_admission=bank.admission(qualified_ledger(bank),profile='production_source30'))
    meta=dict(schema=bank.SCHEMA,profile='production_source30',sources=sources,windows=n,normalization=norm,
        base_dataset=bank.BASE29,base_counts=deepcopy(bank.BASE_COUNTS),normalization_refitted=False,
        arrays=dict(path='/synthetic/measured.npz',bytes=1,sha256='a'*64),
        array_proofs={k:bank.array_proof(samples[k]) for k in bank.ARRAY_KEYS},attempt_ledger=qualified_ledger(bank),
        index=bank.rt.write_new(tmp_path/'index-m.json',dict(samples=[dict(window=w) for w in samples['derived_index']['windows']])),source_manifest=bank.rt.write_new(tmp_path/'manifest-m.json',{}),
        data_code=bank.code_binding(),prefix_producer={},actions_executed=True,physics_qualified=True,native_eligible=True,
        optimization_admission=deepcopy(samples['optimization_admission']))
    complete=dict(optimization_admission=deepcopy(samples['optimization_admission']),metadata=bank.rt.write_new(tmp_path/'metadata-m.json',meta),arrays=meta['arrays'])
    prepared['measured']=dict(reference=ref,samples=samples,metadata=meta,complete=complete)
    return prepared['measured']


def v7_cache(tmp_path,monkeypatch):
    from test_ausc_act_training_v6 import cache_fixture
    p,a,_=cache_fixture(tmp_path,monkeypatch);measured_fixture(tmp_path,monkeypatch,p)
    return p,a


def test_v7_explicit_schema_data5_fixed_schedule_and_no_new_loss():
    assert hasattr(train,'CONFIG_SCHEMA_V7'), 'Explicit V7 schema family required'
    for key in ('CONFIG','PILOT','SELECTION','CPU','PROTOCOL'):
        assert train._version({'schema':getattr(train,key+'_SCHEMA_V7')})==7
    assert train._api(7) is train.phase_data and train._groups(7)==train.GROUPS_V3
    assert train.SCHEDULE_V7==train.SCHEDULE_V6 and train.INITIALIZATION_V7==train.INITIALIZATION_V6
    assert train.selected_step([{'step':500},{'step':1000}],version=7)==1000
    with pytest.raises(ValueError):train._groups(10)


def test_v7_exact_four_native_subgroups_stats_masks_and_realized_draws(tmp_path,monkeypatch):
    m=sampler();p,a=v7_cache(tmp_path,monkeypatch);before=deepcopy(p['normalization'])
    monkeypatch.setattr(bank.given,'fit_normalization',lambda *a,**k:pytest.fail('Normalization refitted'))
    cache=train.build_cache(p,a,optimizer_version=7);proof=cache['measured_inventory']
    assert cache['native_count']==9 and cache['base_native_count']==9 and cache['measured_count']==12
    assert proof['base_counts']==dict(train=9) and proof['measured_windows']==12
    assert proof['native_order']==['base_corridor','base_full','measured_corridor','measured_full']
    assert proof['pools']['measured_corridor']['eligible_row_indices']==[9,10,12,13,15,16,18,19]
    np.testing.assert_array_equal(cache['packed']['native']['action_is_pad'][9:],p['measured']['samples']['action_is_pad'])
    rng=np.random.default_rng(71);value=train.new_optimization(version=7);seen=set()
    for _ in range(200):
        ids=train.sample_cached_indices(rng,cache)
        assert {k:len(v) for k,v in ids.items()}==dict(native=32,planned=8,conditional=8,oracle=16)
        for i,name in enumerate(proof['native_order']):
            assert set(ids['native'][i*8:(i+1)*8])<=set(proof['pools'][name]['eligible_row_indices'])
        seen.update(ids['native'][24:].tolist());train.record_draws(value,cache,ids)
    assert set(range(9,21))<=seen and p['normalization']==before
    m.validate_draws(value,steps=200,inventory=proof)
    assert sum(value['gradient_source_examples']['native'].values())==6400
    for name in proof['native_order']:
        count=value['measured_draws'][name]
        assert count['draws']==count['denominator']==1600 and sum(count['rows'].values())==1600
        assert count['valid_cells']==sum(proof['pools'][name]['rows'][proof['pools'][name]['eligible_row_indices'].index(int(i))]['valid_cells']*n for i,n in count['rows'].items())
    assert value['measured_draws']['measured_corridor']['valid_cells']!=value['measured_draws']['measured_full']['valid_cells']
    for field in ('rows','slot','source','valid_cells','denominator'):
        bad=deepcopy(value);counts=bad['measured_draws']['measured_full']
        counts[field]=0 if field in ('valid_cells','denominator') else {'forged':1600}
        with pytest.raises(ValueError):m.validate_draws(bad,steps=200,inventory=proof)


def test_v7_target_slot_balancing_and_missing_corridor_slots(tmp_path,monkeypatch):
    m=sampler();p,a=v7_cache(tmp_path,monkeypatch)
    # One R3 slot has no corridor windows; full draws must still use it.
    p['measured']['samples']['states'][:3,7]=.02;p['measured']['samples']['histories'][:3,:,7]=.02
    p['measured']['metadata']['array_proofs']={k:bank.array_proof(p['measured']['samples'][k]) for k in bank.ARRAY_KEYS}
    cache=train.build_cache(p,a,optimizer_version=7);pool=cache['measured_inventory']['pools']['measured_corridor']
    assert pool['missing_slots']==[bank.query.SLOT_IDS[0]]
    value=train.new_optimization(version=7);rng=np.random.default_rng(321)
    for _ in range(500):train.record_draws(value,cache,train.sample_cached_indices(rng,cache))
    counts=value['measured_draws']['measured_corridor']
    assert abs(counts['target']['R3']-2000)<150 and abs(counts['target']['L3']-2000)<150
    assert abs(counts['slot'][bank.query.SLOT_IDS[2]]-1000)<125
    assert set(value['measured_draws']['measured_full']['slot'])==set(bank.query.SLOT_IDS)
    m.validate_draws(value,steps=500,inventory=cache['measured_inventory'])


@pytest.mark.parametrize('fault',['empty','missing_target','gap','checked_gap','plan_gap','identity','duplicate'])
def test_v7_measured_pools_fail_closed_before_optimizer(tmp_path,monkeypatch,fault):
    m=sampler();p,a=v7_cache(tmp_path,monkeypatch);measured=p['measured'];sources=measured['metadata']['sources']
    if fault in ('empty','missing_target'):
        ids=slice(None) if fault=='empty' else slice(6,None)
        measured['samples']['states'][ids,7]=.1;measured['samples']['histories'][ids,:,7]=.1
    elif fault=='identity':sources[-1]['metadata']['split']='validation'
    elif fault=='duplicate':sources[-1]['slot_id']=sources[0]['slot_id']
    else:
        source=sources[0];key='fixture_plan' if fault=='plan_gap' else None
        ref=source['metadata'][key] if key else source['declaration']['run_result'];value=bank.rt.read_bound(ref)
        if key:value['free_approach_start_gap_m']+=.001
        elif fault=='gap':value['points'][0]['phase_config']['precontact_gap_m']=0.
        else:value['points'][0]['checked_prepose_gap_m']+=.001
        new=bank.rt.write_new(tmp_path/'bad-gap.json',value)
        if key:source['metadata'][key]=new
        else:source['declaration']['run_result']=new
    with pytest.raises(ValueError):train.build_cache(p,a,optimizer_version=7)


@pytest.mark.parametrize('fault',['unqualified','nontrain','duplicate','normalization','base_counts','unsafe','incomplete'])
def test_v7_bank_admission_rejects_before_arrays_or_optimizer(tmp_path,monkeypatch,fault):
    m=sampler();p,a=v7_cache(tmp_path,monkeypatch);b=p['measured'];meta=b['metadata'];complete=b['complete']
    monkeypatch.setattr(bank,'metadata',lambda ref:(complete,meta))
    monkeypatch.setattr(bank,'load_arrays',lambda *a:pytest.fail('Arrays opened before admission'))
    if fault=='unqualified':meta['attempt_ledger'][0]['status']='failed'
    elif fault=='nontrain':meta['sources'][-1]['metadata']['split']='validation'
    elif fault=='duplicate':meta['attempt_ledger'][-1]['identity']=deepcopy(meta['attempt_ledger'][0]['identity'])
    elif fault=='normalization':meta['normalization']=dict(meta['normalization'],sha256='f'*64)
    elif fault=='base_counts':meta['base_counts']['train']+=1
    elif fault=='unsafe':meta['physics_qualified']=False
    else:meta['sources'].pop()
    with pytest.raises(ValueError):m.admitted_metadata(b['reference'],dataset=bank.BASE29,normalization=b['samples']['normalization_reference'])


def test_v7_bound_inventory_rejects_proof_and_row_forgery_without_raw_payloads(tmp_path,monkeypatch):
    m=sampler();p,a=v7_cache(tmp_path,monkeypatch)
    a['array_reference']=dict(path='/synthetic/v5.npz',bytes=1,sha256='b'*64)
    cache=train.build_cache(p,a,optimizer_version=7);proof=cache['measured_inventory'];b=p['measured']
    monkeypatch.setattr(bank,'metadata',lambda ref:(b['complete'],b['metadata']))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Payload opened at deployment'))
    monkeypatch.setattr(bank,'read_episode',lambda *a:pytest.fail('Historical capture reopened'))
    monkeypatch.setattr(bank.query,'authenticate_source',lambda *a:pytest.fail('Old model/trace authentication'),raising=False)
    config=dict(dataset=bank.BASE29,measured_bank=b['reference'])
    def validate(value):m.validate_binding(value,config=config,native=p['preparation'],auxiliary_metadata=dict(sources=a['sources']),auxiliary_arrays=a['array_reference'])
    validate(proof)
    for fault in ('source','gap','row','mask','arrays','order','missing','base_counts','normalization'):
        bad=deepcopy(proof)
        if fault=='source':bad['pools']['measured_full']['rows'][0]['source']='forged'
        elif fault=='gap':bad['measured_sources'][0]['actual_gap_m']+=.001
        elif fault=='row':bad['pools']['measured_corridor']['rows'][0]['row_index']=0
        elif fault=='mask':bad['pools']['measured_full']['rows'][0]['valid_cells']=20
        elif fault=='arrays':bad['measured_array_proofs']['actions']['sha256']='a'*64
        elif fault=='order':bad['native_order'].reverse()
        elif fault=='missing':bad['pools']['measured_corridor']['missing_slots']=['invented']
        elif fault=='base_counts':bad['base_train_windows']+=1
        else:bad['normalization_sha256']='b'*64
        with pytest.raises(ValueError):validate(bad)


def test_v7_dispatch_actual_factory_sources_ipc_and_head(monkeypatch):
    from pathlib import Path
    assert hasattr(train,'CONFIG_SCHEMA_V7')
    from sim.ausc_policy import m2_given_contact_expanded_pilot_train as expanded
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep,m2_rebuild_paired_eval as paired
    from sim.ausc_policy import m2_given_contact_pilot_native_audit as audit
    sentinel=object()
    for name in ('validate_config','training_protocol','load_selected_policy','verify_selected','fit','validate_complete','model_binding'):
        monkeypatch.setattr(train,name,lambda *a,**k:sentinel)
    assert expanded.validate_config({'schema':train.CONFIG_SCHEMA_V7}) is sentinel
    assert expanded.training_protocol({'schema':train.CONFIG_SCHEMA_V7}) is sentinel
    assert expanded.load_selected_policy({'schema':train.SELECTION_SCHEMA_V7}) is sentinel
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.SELECTION_SCHEMA_V7 if ref=='selection' else train.CONFIG_SCHEMA_V7))
    assert expanded.verify_selected('selection','receipt') is sentinel and expanded.fit('config','out') is sentinel
    assert dep.validate_complete({'schema':train.PILOT_SCHEMA_V7}) is sentinel
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.PILOT_SCHEMA_V7))
    assert dep.model_binding('ref') is sentinel
    for adapter in (dep,paired):
        files=adapter.model_sources({'checkpoint_schema':train.PILOT_SCHEMA_V7})['files']
        assert {'m2_act_measured_sampling.py','m2_measured_recovery_data.py','m2_act_corridor_sampling.py'}<=set(files)
        assert all(Path(ref['path']).parent==Path(train.__file__).parent for ref in files.values())
        assert audit.model_sources(adapter,{'checkpoint_schema':train.PILOT_SCHEMA_V7})['files']==files
        descriptor=dict(model={'checkpoint_schema':train.PILOT_SCHEMA_V7})
        if adapter is dep:assert adapter._checkpoint_binding(descriptor)['checkpoint_schema']==train.PILOT_SCHEMA_V7


def test_v7_protocol_same_act6_warm_tensors_zero_optimizer_and_no_head_reset(tmp_path,monkeypatch):
    from test_ausc_act_training_v5 import initial_fixture
    m,torch,ref,receipt,Tiny,_=initial_fixture(tmp_path,monkeypatch)
    config=dict(train.SCHEDULE_V7,schema=train.CONFIG_SCHEMA_V7,initialization=ref,posture_head=receipt['posture_head'],measured_bank=bank.rt.write_new(tmp_path/'protocol-bank.json',dict(schema=bank.SCHEMA,complete=True,fixture_profile='local_synthetic_fixture')))
    protocol=train.training_protocol(config)
    assert protocol['model_optimizer_version']==7 and protocol['data_version']==5
    assert protocol['objective']=={name:dict(**v,loss='official forward masked valid-cell/joint mean') for name,v in train.GROUPS_V3.items()}
    assert 'ACT9' in protocol['treatment'] and 'ACT9' in protocol['posture_initialization']['method']
    assert train._warm_start(7) is m
    policy=Tiny();initial=m.initialize_policy(torch,policy,ref,normalization=receipt['normalization'],posture_head=receipt['posture_head'])
    assert initial['initial_tensor_proofs']==receipt['tensor_proofs'] and initial['optimizer_step']==0
    optimizer,_=m.fresh_optimizer(torch,policy,lr=config['lr'],weight_decay=config['weight_decay'])
    assert not optimizer.state and all(p.device.type=='cpu' for p in policy.parameters())
    monkeypatch.setattr(train.runtime,'make_model',lambda *a,**k:(torch,policy,{}))
    monkeypatch.setattr(train.posture_head,'install',lambda *a:None)
    monkeypatch.setattr(train.posture_head,'initialize_training',lambda *a:pytest.fail('V7 head reset'))
    assert train.make_model(dict(config,lerobot_root='/official'),device='cpu',training=True)[1] is policy


def test_v7_cpu_groups_padding_poison_positive_gradient_without_optimizer_steps(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');p,a=v7_cache(tmp_path,monkeypatch)
    stats=deepcopy(p['normalization']);cache=train.build_cache(p,a,torch=torch,device='cpu',optimizer_version=7)
    ids=train.sample_cached_indices(np.random.default_rng(8),cache);batches=train.cached_batches(cache,ids)
    class MaskedOfficialObjective(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
        def forward(self,batch):
            valid=~batch['action_is_pad'];loss=((self.weight-batch['action']).abs()*valid[...,None]).sum()/(valid.sum()*7)
            return loss,{}
    policy=MaskedOfficialObjective();optimizer=torch.optim.AdamW(policy.parameters());before=policy.weight.detach().clone()
    loss,logged=train.mixed_loss(policy,batches,version=7)
    loss.backward();assert policy.weight.grad.abs()>0
    poison={g:{k:v.clone() for k,v in batch.items()} for g,batch in batches.items()}
    for batch in poison.values():batch['action'][batch['action_is_pad']]=123456.
    again,_=train.mixed_loss(policy,poison,version=7)
    assert torch.equal(loss,again) and not optimizer.state and torch.equal(before,policy.weight.detach())
    assert logged['native']['examples']==32 and all(p.device.type=='cpu' for p in policy.parameters())
    assert p['normalization']==stats


def test_v7_pose_comes_from_admitted_declaration_without_inventing_compact_metadata(tmp_path,monkeypatch):
    m=sampler();p,a=v7_cache(tmp_path,monkeypatch);meta=p['measured']['metadata']
    for source in meta['sources']:source['metadata'].pop('pose_id')
    before=deepcopy(meta)
    records=m.source_records(meta)
    assert {r['pose'] for r in records}=={'p000474_a001230_f0000'} and meta==before
    meta['sources'][-1]['metadata']['pose_id']='invented'
    with pytest.raises(ValueError):m.source_records(meta)


def test_v7_cache_rejects_measured_array_or_mask_proof_changes(tmp_path,monkeypatch):
    sampler();p,a=v7_cache(tmp_path,monkeypatch)
    p['measured']['samples']['actions'][0,0,0]=.001
    with pytest.raises(ValueError,match='proof|bytes'):train.build_cache(p,a,optimizer_version=7)


def test_v7_fit_saves_both_and_final_even_when500_better(tmp_path,monkeypatch):
    from pathlib import Path
    from contextlib import nullcontext
    from types import SimpleNamespace
    from test_ausc_act_training_v5 import initial_fixture
    m,torch,init,receipt,Tiny,_=initial_fixture(tmp_path,monkeypatch);policy=Tiny()
    config=dict(train.SCHEDULE_V7,schema=train.CONFIG_SCHEMA_V7,base='/fixture',dataset={},auxiliary={},evaluation_cohort={},measured_bank=bank.rt.write_new(tmp_path/'protocol-bank.json',dict(schema=bank.SCHEMA,complete=True,fixture_profile='local_synthetic_fixture')),posture_head=receipt['posture_head'],initialization=init)
    config_ref=m.rt.write_new(tmp_path/'fitconfig.json',config)
    prepared=dict(normalization=receipt['normalization'],scope={},optimization_scope={},native_scope={},window_order={'train':'a'*64,'validation':'b'*64},train={'fixture':'train'},validation={'fixture':'validation'},auxiliary=dict(counts={},samples={}),oracle_authentication=[],posture_target_audit={})
    monkeypatch.setattr(train,'validate_config',lambda c:c);monkeypatch.setattr(train,'load_inputs',lambda c:prepared)
    monkeypatch.setattr(train.runtime,'owned_gpu',lambda:nullcontext({}));monkeypatch.setattr(train.runtime,'_identity',lambda:dict(pid=1,pgid=1,sid=1))
    monkeypatch.setattr(train,'make_model',lambda *a,**k:(torch,policy,{}));monkeypatch.setattr(train,'sum',lambda *a:2658823,raising=False)
    monkeypatch.setattr(train,'validate_official_forward',lambda *a:None);monkeypatch.setattr(train.runtime,'execution_settings',lambda *a,**k:{})
    def cache(*a,**kwargs):
        assert kwargs['optimizer_version']==7
        return dict(terminal_pool_inventory={'old6mm':True},measured_inventory={'fixture':True},measured_count=12)
    monkeypatch.setattr(train,'build_cache',cache);monkeypatch.setattr(train,'sample_cached_indices',lambda *a:{})
    monkeypatch.setattr(train,'cached_batches',lambda *a:{});monkeypatch.setattr(train,'record_draws',lambda *a:None)
    monkeypatch.setattr(train,'mixed_loss',lambda *a,**k:(policy.weight.square().mean(),{}))
    monkeypatch.setattr(train.posture_head,'diagnostics',lambda *a:{});monkeypatch.setattr(train.scope,'validate_train16_metadata',lambda *a,**k:None)
    calls=[];monkeypatch.setattr(train.runtime,'infer',lambda *a,**k:np.zeros((1,20,7)))
    def metrics(*a):
        calls.append(None);return {'act':{'full_chunk_mae_rad':.01 if len(calls)<=2 else .02}}
    monkeypatch.setattr(train.runtime,'metrics',metrics)
    def child(argv,**kwargs):
        assert '--verify-selected' in argv;train.runtime.write_new(Path(argv[-1]),{'passed':True})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(train,'code_binding',lambda **k:{'fixture_only':True});monkeypatch.setattr(train.subprocess,'run',child)
    result=train.fit(config_ref,tmp_path/'fit')
    assert result['schema']==train.PILOT_SCHEMA_V7 and result['selected_step']==result['completed_steps']==1000
    assert [r['step'] for r in result['evaluation_candidates']]==[500,1000]
    assert result['data_version']==5 and result['model_optimizer_version']==7
    assert train.runtime.read_bound(result['measured_inventory'])=={'fixture':True}
    for row in result['evaluation_candidates']:
        packet=torch.load(train.runtime.bound_path(row['checkpoint']),weights_only=True)
        assert packet['schema']==train.PILOT_SCHEMA_V7 and packet['step']==row['step']
        assert packet['measured_inventory']==result['measured_inventory'] and 'optimizer_state_dict' not in packet
    train._validate_candidates(result)


@pytest.mark.parametrize('failure_point',['warm_start','cache'])
def test_v7_fit_failure_before_inventory_preserves_original_error_and_artifacts(tmp_path,monkeypatch,failure_point):
    from pathlib import Path
    from contextlib import nullcontext
    from test_ausc_act_training_v5 import initial_fixture
    m,torch,init,receipt,Tiny,_=initial_fixture(tmp_path,monkeypatch);policy=Tiny()
    config=dict(train.SCHEDULE_V7,schema=train.CONFIG_SCHEMA_V7,base='/fixture',dataset={},auxiliary={},evaluation_cohort={},measured_bank=bank.rt.write_new(tmp_path/'protocol-bank.json',dict(schema=bank.SCHEMA,complete=True,fixture_profile='local_synthetic_fixture')),posture_head=receipt['posture_head'],initialization=init)
    config_ref=m.rt.write_new(tmp_path/'failureconfig.json',config)
    prepared=dict(normalization=receipt['normalization'],scope={},optimization_scope={},native_scope={},window_order={'train':'a'*64,'validation':'b'*64},
        auxiliary=dict(counts={}),oracle_authentication=[],posture_target_audit={})
    monkeypatch.setattr(train,'validate_config',lambda c:c);monkeypatch.setattr(train,'load_inputs',lambda c:prepared)
    monkeypatch.setattr(train.runtime,'owned_gpu',lambda:nullcontext({}));monkeypatch.setattr(train.runtime,'_identity',lambda:dict(pid=1,pgid=1,sid=1))
    monkeypatch.setattr(train,'make_model',lambda *a,**k:(torch,policy,{}));monkeypatch.setattr(train,'sum',lambda *a:2658823,raising=False)
    monkeypatch.setattr(train,'validate_official_forward',lambda *a:None);monkeypatch.setattr(train.runtime,'execution_settings',lambda *a,**k:{})
    monkeypatch.setattr(train.posture_head,'diagnostics',lambda *a:{});monkeypatch.setattr(train,'code_binding',lambda **k:{'fixture_only':True})
    message='original '+failure_point+' failure'
    def fail(*a,**k):raise ValueError(message)
    monkeypatch.setattr(m if failure_point=='warm_start' else train,'initialize_policy' if failure_point=='warm_start' else 'build_cache',fail)
    output=tmp_path/'failure'
    with pytest.raises(ValueError,match=message):train.fit(config_ref,output)
    failed=m.rt.read_bound(m.rt.reference(output/'FAILED.json'))
    assert failed['schema']==train.PILOT_SCHEMA_V7 and failed['complete'] is False and failed['completed_steps']==0
    assert failed['error']=='ValueError: '+message and not (output/'COMPLETE.json').exists()
    trace=m.rt.bound_path(failed['traceback']).read_text()
    assert 'ValueError: '+message in trace and 'KeyError' not in trace
    partial=torch.load(m.rt.bound_path(failed['partial_checkpoint']),weights_only=True)
    assert partial['complete'] is False and partial['step']==0 and partial['error']==failed['error']
    assert partial.get('measured_inventory') is None
    with pytest.raises(ValueError):train._measured_inventory(partial,{})


def completed_fixture(tmp_path,monkeypatch):
    """Completed metadata fixture; sampler admission itself is tested separately."""
    from test_ausc_act_training_v5 import completed_fixture as previous
    m,torch,value,receipt=previous(tmp_path,monkeypatch);sampling=sampler()
    config=m.rt.read_bound(value['inputs']['config']);config.update(schema=train.CONFIG_SCHEMA_V7,measured_bank=m.rt.write_new(tmp_path/'bank7.json',dict(schema=bank.SCHEMA,complete=True,fixture_profile='local_synthetic_fixture')))
    value.update(schema=train.PILOT_SCHEMA_V7,schedule=config,data_version=5,model_optimizer_version=7,protocol=train.training_protocol(config))
    value['inputs'].update(config=m.rt.write_new(tmp_path/'V7CONFIG.json',config),code=train.code_binding(version=7),measured_bank=config['measured_bank'])
    sources=train.contribution_inventory(config)
    base_rows=[];measured_rows=[];aux_rows=[];group_rows={}
    for i,(sid,entry) in enumerate(sources['native'].items()):
        base_rows.append(dict(row_index=i,source=sid,body=entry['subject_id'],pose=entry['pose_id'],target=entry['target_id'],anchor=entry['anchor'],valid_cells=20,error_m=0.,actual_gap_m=.025))
    for i,slot in enumerate(bank.query.SLOT_IDS):
        sid='measured:'+slot;target='R3' if i<2 else 'L3'
        sources['native'][sid]=dict(subject_id='p001120',kind='measured_recovery',pose_id=train.runtime_scope_poses()[0],target_id=target,slot=slot)
        measured_rows.append(dict(row_index=len(base_rows)+i,source=sid,body='p001120',pose=train.runtime_scope_poses()[0],target=target,slot=slot,episode=slot,valid_cells=i+1,error_m=0.,actual_gap_m=.025))
    for group in ('planned','conditional','oracle'):
        rows=[]
        for sid,entry in sources[group].items():
            row=dict(row_index=len(aux_rows),source=sid,body=entry['subject_id'],pose=entry['pose_id'],target=entry['target_id'],anchor=entry['anchor'],valid_cells=1)
            aux_rows.append(row);rows.append(row)
        group_rows[group]=rows
    pools=dict(base_corridor=sampling._pool(base_rows,threshold=.012),base_full=sampling._pool(base_rows),
        measured_corridor=sampling._pool(measured_rows,threshold=.012),measured_full=sampling._pool(measured_rows))
    for group in ('planned','conditional','oracle'):
        pools[group+'_full']=sampling._pool(group_rows[group])
        if group!='oracle':pools[group+('_near' if group=='conditional' else '_corridor')]=sampling._pool(group_rows[group],threshold=.006 if group=='conditional' else .012)
    proof=dict(schema=sampling.SCHEMA,data_version=5,model_optimizer_version=7,native_order=list(sampling.NATIVE_ORDER),pools=pools,
        base_train_windows=len(base_rows),measured_windows=len(measured_rows))
    value['measured_inventory']=m.rt.write_new(tmp_path/'MEASURED7.json',proof)
    value['measured_counts']=dict(episodes=4,train_windows=4,base_counts=deepcopy(train.NATIVE_COUNTS),normalization_refitted=False)
    monkeypatch.setattr(train,'_measured_inventory',lambda *a:proof)
    native=base_rows+measured_rows
    cache=dict(version=7,source_ids={name:np.asarray([r['source'] for r in rows],object) for name,rows in (('native',native),('auxiliary',aux_rows))},
        subjects={name:np.asarray([r['body'] for r in rows],object) for name,rows in (('native',native),('auxiliary',aux_rows))},
        kinds=dict(native=np.asarray(['native']*len(base_rows)+['measured_recovery']*4,object),auxiliary=np.asarray([sources[g][r['source']]['kind'] for g in ('planned','conditional','oracle') for r in group_rows[g]],object)),
        terminal_labels={r['row_index']:{k:r[k] for k in ('source','body','pose','target','anchor')} for r in group_rows['conditional']},
        measured_labels={name:{r['row_index']:r for r in pool['rows']} for name,pool in pools.items()})
    optimization=train.new_optimization(version=7);rng=np.random.default_rng(99)
    for _ in range(1000):
        ids={g:np.asarray(rng.choice([r['row_index'] for r in group_rows[g]],size=train.GROUPS_V3[g]['size'])) for g in group_rows}
        ids['native']=np.r_[rng.choice(len(base_rows),16),rng.choice(np.arange(len(base_rows),len(native)),16)]
        train.record_draws(optimization,cache,ids)
    value.update(optimization=optimization,selected_optimization=deepcopy(optimization))
    selected=dict(value,schema=train.SELECTION_SCHEMA_V7)
    selected_ref=m.rt.write_new(tmp_path/'V7SELECTED.json',selected);oldcpu=m.rt.read_bound(value['cpu_equivalence'])
    value['cpu_equivalence']=m.rt.write_new(tmp_path/'V7CPU.json',dict(oldcpu,schema=train.CPU_SCHEMA_V7,selection=selected_ref,
        data_version=5,model_optimizer_version=7,measured_inventory=value['measured_inventory'],measured_counts=value['measured_counts']))
    return m,torch,value,receipt


def test_v7_complete_cpu_binding_and_every_native_subgroup_counter(tmp_path,monkeypatch):
    actual=train._measured_inventory;m,_,value,_=completed_fixture(tmp_path,monkeypatch)
    assert train.validate_complete(value) is value
    assert train.model_binding(m.rt.write_new(tmp_path/'V7COMPLETE.json',value))[0]['checkpoint_schema']==train.PILOT_SCHEMA_V7
    for fault in ('old_cpu','lost500','data_version','native_subgroup','planned_rows','measured_slot','head_reset','cpu_inventory'):
        bad=deepcopy(value)
        if fault in ('old_cpu','cpu_inventory'):
            cpu=m.rt.read_bound(value['cpu_equivalence'])
            if fault=='old_cpu':cpu['schema']=train.CPU_SCHEMA_V6
            else:cpu['measured_inventory']=dict(cpu['measured_inventory'],sha256='f'*64)
            bad['cpu_equivalence']=m.rt.write_new(tmp_path/('BADCPU-'+fault+'.json'),cpu)
        elif fault=='lost500':bad['evaluation_candidates'].pop(0)
        elif fault=='data_version':bad['data_version']=7
        elif fault=='native_subgroup':bad['optimization']['measured_draws']['base_full']['draws']-=1
        elif fault=='planned_rows':bad['optimization']['measured_draws']['planned_corridor']['rows']={'99999':4000}
        elif fault=='measured_slot':bad['optimization']['measured_draws']['measured_full']['slot']={'invented':8000}
        else:bad['initialization']['optimizer']['initial_step']=18000
        with pytest.raises(ValueError):
            if fault in ('old_cpu','cpu_inventory'):train.model_binding(m.rt.write_new(tmp_path/('BAD-'+fault+'.json'),bad))
            elif fault=='data_version':actual(bad,{})
            else:train.validate_complete(bad)


def test_v7_inference_loads_only_own_checkpoint_and_never_historical_models_or_payloads(tmp_path,monkeypatch):
    m,torch,value,receipt=completed_fixture(tmp_path,monkeypatch)
    software={key:{} for key in ('vendor','dependencies','python','prefix','python_version','source_bindings')}
    value['inputs']['software']=m.rt.write_new(tmp_path/'OWN_SOFTWARE7.json',software)
    packet=dict(value,step=1000,state_dict={'own':'V7'},normalization=receipt['normalization'],optimization=value['selected_optimization'])
    loaded=[];policy=SimpleNamespace(load_state_dict=lambda state,strict:loaded.append((state,strict)),eval=lambda:None)
    monkeypatch.setattr(train,'make_model',lambda *a,**k:(torch,policy,software));monkeypatch.setattr(train,'validate_official_forward',lambda *a:None)
    for name in ('initialize_policy','export','require_archive'):
        monkeypatch.setattr(m,name,lambda *a,**k:pytest.fail('V7 inference replayed old model initialization'))
    monkeypatch.setattr(bank,'load_arrays',lambda *a:pytest.fail('Inference decoded measured numerical payload'))
    monkeypatch.setattr(train.phase_data,'load_preparation',lambda *a,**k:pytest.fail('Inference decoded numerical training payload'))
    calls=[];monkeypatch.setattr(torch,'load',lambda *a,**k:calls.append(a) or packet)
    assert train.load_selected_policy(value)[2] is packet
    assert len(calls)==1 and loaded==[({'own':'V7'},True)]
    packet['schema']=train.PILOT_SCHEMA_V6
    with pytest.raises(ValueError):train.load_selected_policy(value)


def test_v7_configuration_requires_separate_admitted_bank_and_exact_act6_budget(tmp_path,monkeypatch):
    from test_ausc_act_training_v5 import initial_fixture
    from test_ausc_act_training_v6 import native_fixture
    from test_ausc_act_auxiliary_train import scope_value,source
    from sim.ausc_policy import rebuild_gpu_binding as binding
    m,torch,init,receipt,_,_=initial_fixture(tmp_path,monkeypatch)
    frozen=scope_value();scope_ref=m.rt.write_new(tmp_path/'scope7.json',frozen)
    header,_=native_fixture(tmp_path)
    for i,e in enumerate(header['episodes']):e.update(windows=2791,split_sample_range=[i*2791,(i+1)*2791])
    measured=measured_fixture(tmp_path,monkeypatch,dict(normalization=receipt['normalization']))
    header.update(schema='m2_given_contact_masked_preparation_v1',complete=True,rebuild_epoch=binding.EPOCH,original_cases=0,
        diagnostic_only=True,formal_acceptance=False,training_eligible=False,test_payloads_opened=False,
        task_contract=m.rt.task_contract(),counts=deepcopy(train.NATIVE_COUNTS),normalization=measured['metadata']['normalization'],
        split_subject_ids=dict(train=frozen['cohorts']['train'][:4],validation=['p000923']))
    native_ref=m.rt.write_new(tmp_path/'native7.json',header);measured['metadata']['base_dataset']=native_ref
    sources=[source(kind,subject=subject) for kind in train.conditional_data.KINDS for subject in frozen['optimized_train_ids']]
    metadata=dict(sources=sources,input_profile='production_closed_inputs',optimization_admission=train.phase_data._admission(True))
    aux=m.rt.write_new(tmp_path/'aux7.json',{'schema':train.phase_data.SCHEMA})
    monkeypatch.setattr(train,'NATIVE_DATASET',native_ref);monkeypatch.setattr(train,'TRAIN16_SCOPE',scope_ref)
    monkeypatch.setattr(train,'INITIALIZATION_V7',init);monkeypatch.setattr(train.scope,'validate',lambda ref:m.rt.read_bound(ref)['cohorts'])
    monkeypatch.setattr(train.phase_data,'metadata',lambda *a,**k:({},metadata))
    monkeypatch.setattr(bank,'metadata',lambda *a:(measured['complete'],measured['metadata']))
    monkeypatch.setattr(train.posture_head,'validate_bound_spec',lambda *a,**k:None)
    monkeypatch.setattr(train,'native_normalization',lambda config:receipt['normalization'])
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Configuration opened numerical data'))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Configuration loaded old model'))
    config=dict(train.SCHEDULE_V7,schema=train.CONFIG_SCHEMA_V7,base='/fixture',dataset=native_ref,auxiliary=aux,
        measured_bank=measured['reference'],evaluation_cohort=scope_ref,posture_head=receipt['posture_head'],initialization=init)
    assert train.validate_config(config) is config
    for patch in (dict(steps=20000),dict(steps=1001),dict(eval_every=100),dict(batch_size=32),dict(resume=True),dict(initialization={'path':'/ACT8/selected.pt','bytes':1,'sha256':'a'*64})):
        with pytest.raises(ValueError):train.validate_config(dict(config,**patch))
    missing=dict(config);missing.pop('measured_bank')
    with pytest.raises(ValueError):train.validate_config(missing)
    measured['metadata']['physics_qualified']=False
    with pytest.raises(ValueError,match='FOUR|qualified'):train.validate_config(config)


def test_v7_unqualified_cache_bank_cannot_inherit_old_true_eligibility(tmp_path,monkeypatch):
    sampler();p,a=v7_cache(tmp_path,monkeypatch)
    p['measured']['metadata']['attempt_ledger'][0]['status']='failed'
    with pytest.raises(ValueError,match='FOUR|admission'):train.build_cache(p,a,optimizer_version=7)


def test_v7_base_corridor_uses_actual_gap_without_obsolete_global_near60_gate(tmp_path,monkeypatch):
    sampler();p,a=v7_cache(tmp_path,monkeypatch)
    p['train']['states'][:,9]=-.08;p['train']['histories'][:,:,9]=-.08
    for i,e in enumerate(p['preparation']['episodes']):
        report=bank.rt.read_bound(e['run_result']);point=report['points'][0]
        point['checked_prepose_gap_m']=point['phase_config']['precontact_gap_m']=.08
        e['run_result']=bank.rt.write_new(tmp_path/('non60-report'+str(i)+'.json'),report)
    cache=train.build_cache(p,a,optimizer_version=7)
    assert cache['measured_inventory']['pools']['base_corridor']['eligible_row_indices']==[0,1,3,4,6,7]


def test_v7_native_ipc_uses_v7_tag_and_keeps_original_runtime_posture_gate(tmp_path,monkeypatch):
    from test_ausc_given_contact_pose import reference,metadata
    from test_ausc_act_training_v4 import head_spec
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep,m2_rebuild_paired_eval as paired
    ref=reference(tmp_path);meta=dict(metadata(ref),episode_id='v7',target_id='L1');q=np.zeros(7)
    state=dep.runtime.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    descriptor=dict(metadata=meta,episode_id='v7',source_episode_id='native',model=dict(checkpoint={'sha256':'a'*64},checkpoint_schema=train.PILOT_SCHEMA_V7),source_config={'sha256':'b'*64},protocol={'sha256':'c'*64},scenario_id='s')
    for adapter in (dep,paired):
        request=adapter.make_request(descriptor,1,q,ref.pose_base,1.,metadata=meta,history=np.stack([state,state]),history_times=[0.,.05],context=dict(observed_at_s=.05,physics_row=12,context=['v7','L1',0,'FREE'],source_id='given-contact-'+ref.sha256))
        adapter.validate_request(request,descriptor,sequence=1)
        assert adapter.decode_response(adapter.response_for(request,np.zeros((20,7))),request).metadata['checkpoint_policy']==train.PILOT_SCHEMA_V7
    spec=head_spec();calls=[]
    monkeypatch.setattr(train.posture_head,'validate_bound_spec',lambda *a,**k:calls.append(k))
    complete=dict(schema=train.PILOT_SCHEMA_V7,protocol=dict(posture_head=spec))
    train.posture_head.validate_runtime_head(complete,dict(posture_limits=spec['physical_posture_profile']))
    assert calls==[{'version':2}]
    with pytest.raises(ValueError,match='posture profile'):train.posture_head.validate_runtime_head(complete,dict(posture_limits={}))


def test_v2_array_adapter_and_v7_bounded_metadata_keep_sampler_and_masks(tmp_path,monkeypatch):
    """Numerical V2 harness with explicit synthetic admission stub only.

    The production loader rejects this fixture profile. Here admission is
    isolated so that real packing, sampling and bounded binding can be tested
    without inventing Source32 captures or opening a historical factory.
    """
    m=sampler();p,a=v7_cache(tmp_path,monkeypatch);b=p['measured'];meta=b['metadata'];samples=b['samples']
    meta.update(schema=bank.SCHEMA_V2,profile='local_synthetic_fixture_v2')
    assert bank.admission(meta['attempt_ledger'],profile=meta['profile'])['measured_native_optimization_eligible'] is False
    fixture_admission=dict(bank.admission(meta['attempt_ledger'],profile=meta['profile']),measured_native_optimization_eligible=True)
    samples.update(bank_schema=bank.SCHEMA_V2,optimization_admission=deepcopy(fixture_admission))
    meta['optimization_admission']=deepcopy(fixture_admission);b['complete']['optimization_admission']=deepcopy(fixture_admission)
    original=bank.admission
    monkeypatch.setattr(bank,'admission',lambda ledger,*,profile:deepcopy(fixture_admission)
        if profile=='local_synthetic_fixture_v2' else original(ledger,profile=profile))
    a['array_reference']=dict(path='/synthetic/v5.npz',bytes=1,sha256='b'*64)
    cache=train.build_cache(p,a,optimizer_version=7);proof=cache['measured_inventory']
    assert proof['native_order']==list(m.NATIVE_ORDER) and proof['data_version']==5 and proof['model_optimizer_version']==7
    np.testing.assert_array_equal(cache['packed']['native']['action_is_pad'][9:],samples['action_is_pad'])
    monkeypatch.setattr(bank,'metadata',lambda ref:(b['complete'],meta)) # Sealed metadata layer isolated from admission.
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Raw arrays read during metadata binding'))
    monkeypatch.setattr(bank,'read_episode',lambda *a,**k:pytest.fail('V1 raw capture opened'))
    monkeypatch.setattr(bank,'read_teacher_episode',lambda *a,**k:pytest.fail('V2 raw capture opened'))
    monkeypatch.setattr(bank.data,'_source_freeze',lambda *a,**k:pytest.fail('Whole source/trace authentication at inference'))
    monkeypatch.setattr(bank.rt,'make_model',lambda *a,**k:pytest.fail('Historical model factory'))
    m.validate_binding(proof,config=dict(dataset=bank.BASE29,measured_bank=b['reference']),native=p['preparation'],
        auxiliary_metadata=dict(sources=a['sources']),auxiliary_arrays=a['array_reference'])
    assert 'm2_teacher_stop_reader.py' in train.code_binding(version=7)['files']


@pytest.mark.parametrize('schema',[bank.SCHEMA,bank.SCHEMA_V2])
def test_v7_protocol_reports_actual_already_validated_bank_schema(tmp_path,monkeypatch,schema):
    from test_ausc_act_training_v5 import initial_fixture
    m,_,ref,receipt,_,_=initial_fixture(tmp_path,monkeypatch)
    complete=m.rt.write_new(tmp_path/'synthetic-protocol-bank.json',dict(schema=schema,complete=True,fixture_profile='local_synthetic_fixture'))
    config=dict(train.SCHEDULE_V7,schema=train.CONFIG_SCHEMA_V7,initialization=ref,posture_head=receipt['posture_head'],measured_bank=complete)
    monkeypatch.setattr(bank,'metadata',lambda *a,**k:pytest.fail('Protocol requalified bank instead of reading validated COMPLETE'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Protocol opened numerical payload'))
    assert train.training_protocol(config)['measured_data_schema']==schema
