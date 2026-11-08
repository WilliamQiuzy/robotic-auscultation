"""ACT8 sampler-only contracts on synthetic closed inputs; no runtime claims."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sim.ausc_policy import m2_act_auxiliary_train as train
from sim.ausc_policy import m2_act_phase_data as data


def corridor():
    name='sim.ausc_policy.m2_act_corridor_sampling'
    assert importlib.util.find_spec(name), 'Separate sampler-only corridor inventory required'
    return importlib.import_module(name)


def native_fixture(tmp_path):
    states=np.zeros((9,17));episodes=[]
    for i,(body,gap) in enumerate(zip(train.scope.TRAIN16_IDS[:3],(.025,.04,.025))):
        episode='native-'+str(i);target=('R3','L3','R3')[i]
        meta=dict(subject_id=body,split='train',target_id=target,episode_id=episode)
        config=train.runtime.write_new(tmp_path/('config'+str(i)+'.json'),dict(m2_given_contact_control=dict(metadata=meta,episode_id=episode)))
        report=train.runtime.write_new(tmp_path/('report'+str(i)+'.json'),dict(points=[dict(target=target,checked_prepose_gap_m=gap,phase_config=dict(precontact_gap_m=gap))]))
        episodes.append(dict(episode_id=episode,subject_id=body,split='train',target_id=target,config=config,run_result=report,windows=3,split_sample_range=[i*3,i*3+3]))
        states[i*3:i*3+3,7:10]=[[0,0,-gap],[.012,0,-gap],[.013,0,-gap]]
    return dict(counts={'train':9},episodes=episodes,arrays={'train':dict(path='/bound/train.npz',bytes=12,sha256='a'*64)}),dict(states=states,histories=np.stack([states,states],axis=1))


def test_native_gap_coordinates_boundary_and_bound_rows(tmp_path):
    m=corridor();header,samples=native_fixture(tmp_path)
    metadata=m.native_metadata(header,allowed_train_ids=train.scope.TRAIN16_IDS)
    pool=m.native_inventory(samples,metadata,array_reference=header['arrays']['train'])
    assert pool['eligible_row_indices']==[0,1,3,4,6,7]
    assert pool['source_count']==3 and pool['row_count']==6
    assert [e['actual_gap_m'] for e in metadata]==[.025,.04,.025]
    assert pool['arrays']==header['arrays']['train'] and len(pool['order_sha256'])==64
    assert pool['rows'][2]['episode']=='native-1' and 'pose' not in pool['rows'][2]
    assert all(row['error_m']<=.012 for row in pool['rows'])


@pytest.mark.parametrize('fault',['identity','range','overlap','gap','target','binding','history','nan'])
def test_native_rejects_bad_identity_range_point_bindings_and_histories(tmp_path,fault):
    m=corridor();header,samples=native_fixture(tmp_path)
    if fault=='identity':header['episodes'][-1]['subject_id']='p000923'
    elif fault=='range':header['episodes'][-1]['split_sample_range']=[6,8]
    elif fault=='overlap':header['episodes'][1]['split_sample_range']=[2,5]
    elif fault in ('gap','target','binding'):
        e=header['episodes'][0];value=train.runtime.read_bound(e['run_result'] if fault!='binding' else e['config'])
        if fault=='gap':value['points'][0]['phase_config']['precontact_gap_m']=.04
        elif fault=='target':value['points'][0]['target']='other'
        else:value['m2_given_contact_control']['metadata']['subject_id']='p000923'
        e['run_result' if fault!='binding' else 'config']=train.runtime.write_new(tmp_path/'bad.json',value)
    elif fault=='history':samples['histories'][0,1,7]+=.01
    else:samples['states'][0,7]=np.nan
    with pytest.raises(ValueError):
        metadata=m.native_metadata(header,allowed_train_ids=train.scope.TRAIN16_IDS)
        m.native_inventory(samples,metadata,array_reference=header['arrays']['train'])


def test_native_gates_all_train_identities_before_any_bound_payload(tmp_path,monkeypatch):
    m=corridor();header,_=native_fixture(tmp_path);header['episodes'][-1]['subject_id']='p000923'
    monkeypatch.setattr(train.runtime,'read_bound',lambda *a:pytest.fail('Payload before TRAIN identity gate'))
    with pytest.raises(ValueError,match='TRAIN'):m.native_metadata(header,allowed_train_ids=train.scope.TRAIN16_IDS)


def cache_fixture(tmp_path,monkeypatch):
    from test_ausc_act_training_v5 import v5_cache
    prepared,auxiliary,start=v5_cache(tmp_path,monkeypatch)
    header,native=native_fixture(tmp_path)
    # Retain the existing native actions, padding and statistics fixture.
    prepared['train'].update(native);prepared['preparation']=header
    auxiliary['samples']['source_histories'][:42,:,7:10]=[.009,0,-.04]
    for i,source in enumerate(auxiliary['sources']):
        source['windows']=[{} for _ in np.flatnonzero(auxiliary['samples']['source_indices']==i)]
    return prepared,auxiliary,start


def test_v6_explicit_optimizer6_data5_and_fixed_final():
    assert hasattr(train,'CONFIG_SCHEMA_V6'), 'Explicit V6 optimization schema family required'
    for key in ('CONFIG','PILOT','SELECTION','CPU','PROTOCOL'):
        assert train._version({'schema':getattr(train,key+'_SCHEMA_V6')})==6
    assert train._version({'schema':data.SCHEMA})==5 and train._api(6) is data
    assert train._groups(6)==train.GROUPS_V3 and train.SCHEDULE_V6==train.SCHEDULE_V5
    assert train.selected_step([{'step':500},{'step':1000}],version=6)==1000
    with pytest.raises(ValueError):train.selected_step([{'step':500}],version=6)
    with pytest.raises(ValueError):train._groups(10)


def test_v6_native16_planned4_conditional4_exact_draws_and_unmodified_packing(tmp_path,monkeypatch):
    m=corridor();prepared,auxiliary,start=cache_fixture(tmp_path,monkeypatch)
    cache=train.build_cache(prepared,auxiliary,optimizer_version=6)
    assert cache['version']==6 and cache['data_version']==5
    pools=cache['corridor_inventory']['pools']
    assert pools['native']['eligible_row_indices']==[0,1,3,4,6,7]
    assert pools['planned']['row_count']==42 and pools['planned']['source_count']==2
    assert pools['conditional']['eligible_row_indices']==list(range(start,start+8))
    assert pools['planned']['threshold_m']==.012 and pools['conditional']['threshold_m']==.006
    old=data.terminal_pool_inventory(auxiliary,allowed_train_ids=train.scope.TRAIN16_IDS)
    assert not set(old['eligible_row_indices'])&set(range(42)) and old['diagnostic_threshold_m']==.006
    stats=prepared['normalization'];before=deepcopy(stats)
    expected=data.pack_batch(auxiliary['samples'],stats,np.arange(len(auxiliary['samples']['source_indices'])))
    for key,value in expected.items():np.testing.assert_array_equal(cache['packed']['auxiliary'][key],value)
    result=train.new_optimization(version=6);rng=np.random.default_rng(27)
    for _ in range(50):
        ids=train.sample_cached_indices(rng,cache)
        assert {key:len(v) for key,v in ids.items()}==dict(native=32,planned=8,conditional=8,oracle=16)
        for key,n in (('native',16),('planned',4),('conditional',4)):
            assert set(ids[key][:n])<=set(pools[key]['eligible_row_indices'])
        train.record_draws(result,cache,ids)
    assert stats==before and result['gradient_examples']==dict(native=1600,planned=400,conditional=400,oracle=800)
    m.validate_draws(result,steps=50,inventory=cache['corridor_inventory'])
    for key,n in (('native',16),('planned',4),('conditional',4)):
        counts=result['corridor_draws'][key]
        assert counts['draws']==50*n and sum(counts['rows'].values())==50*n
        assert counts['denominator']==result['gradient_examples'][key]
    for fault in ('rows','body','target','source','denominator'):
        bad=deepcopy(result);count=bad['corridor_draws']['native']
        if fault=='denominator':count[fault]-=1
        elif fault=='rows':count[fault]={'2':800} # ineligible row in actual native episode
        else:count[fault]={'forged':800}
        with pytest.raises(ValueError):m.validate_draws(bad,steps=50,inventory=cache['corridor_inventory'])


def test_v6_empty_planned_pool_rejected_and_missing_strata_visible(tmp_path,monkeypatch):
    corridor();p,a,_=cache_fixture(tmp_path,monkeypatch)
    a['samples']['source_histories'][:42,:,7:10]=[.013,0,-.04]
    with pytest.raises(ValueError,match='planned|corridor'):train.build_cache(p,a,optimizer_version=6)


def test_v6_missing_available_strata_recorded_without_requiring_all16(tmp_path,monkeypatch):
    m=corridor();p,a,_=cache_fixture(tmp_path,monkeypatch)
    p['optimization_scope']=dict(cohorts={'train':list(train.scope.TRAIN16_IDS)},pose_ids=list(train.runtime_scope_poses()),target_ids=['R3','L3'])
    inventory=m.build_inventory(p,a);pool=inventory['pools']['planned']
    assert pool['source_count']==2 and pool['row_count']==42
    assert pool['available_source_count']==2 and len(pool['missing_strata'])==62
    assert pool['missing_sources']==[]


def test_v6_exact_configuration_schedule_and_act6_pin_before_payload(tmp_path,monkeypatch):
    from test_ausc_act_training_v5 import initial_fixture
    from test_ausc_act_auxiliary_train import scope_value,source
    from sim.ausc_policy import rebuild_gpu_binding as binding
    m,torch,init,receipt,_,_=initial_fixture(tmp_path,monkeypatch)
    frozen=scope_value();scope_ref=m.rt.write_new(tmp_path/'scopev6.json',frozen)
    header,_=native_fixture(tmp_path)
    for i,e in enumerate(header['episodes']):e.update(windows=2791,split_sample_range=[i*2791,(i+1)*2791])
    header.update(schema='m2_given_contact_masked_preparation_v1',complete=True,rebuild_epoch=binding.EPOCH,original_cases=0,
        diagnostic_only=True,formal_acceptance=False,training_eligible=False,test_payloads_opened=False,
        task_contract=m.rt.task_contract(),counts=train.NATIVE_COUNTS,split_subject_ids=dict(train=frozen['cohorts']['train'][:4],validation=['p000923']))
    native_ref=m.rt.write_new(tmp_path/'nativev6.json',header)
    sources=[source(kind,subject=subject) for kind in train.conditional_data.KINDS for subject in frozen['optimized_train_ids']]
    metadata=dict(sources=sources,input_profile='production_closed_inputs',optimization_admission=data._admission(True))
    aux=m.rt.write_new(tmp_path/'auxv6.json',{'schema':data.SCHEMA})
    monkeypatch.setattr(train,'NATIVE_DATASET',native_ref);monkeypatch.setattr(train,'TRAIN16_SCOPE',scope_ref)
    monkeypatch.setattr(train,'INITIALIZATION_V6',init);monkeypatch.setattr(train.scope,'validate',lambda ref:m.rt.read_bound(ref)['cohorts'])
    monkeypatch.setattr(data,'metadata',lambda *a,**k:({},metadata));monkeypatch.setattr(train.posture_head,'validate_bound_spec',lambda *a,**k:None)
    monkeypatch.setattr(train,'native_normalization',lambda config:receipt['normalization'])
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Configuration opened numerical data'))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Configuration loaded old model'))
    config=dict(train.SCHEDULE_V6,schema=train.CONFIG_SCHEMA_V6,base='/fixture',dataset=native_ref,auxiliary=aux,evaluation_cohort=scope_ref,posture_head=receipt['posture_head'],initialization=init)
    assert train.validate_config(config) is config
    for patch in (dict(steps=20000),dict(steps=1001),dict(eval_every=100),dict(batch_size=32),dict(resume=True),dict(initialization={'path':'/ACT7/selected.pt','bytes':1,'sha256':'a'*64})):
        with pytest.raises(ValueError,match='schedule|ACT6'):train.validate_config(dict(config,**patch))
    metadata['input_profile']='local_synthetic_fixture'
    with pytest.raises(ValueError,match='production'):train.validate_config(config)


def test_v5_rng_is_unchanged_with_explicit_optimizer_version(tmp_path,monkeypatch):
    from test_ausc_act_training_v5 import v5_cache
    p,a,_=v5_cache(tmp_path,monkeypatch)
    implicit=train.build_cache(p,a);explicit=train.build_cache(p,a,optimizer_version=5)
    first=np.random.default_rng(54);second=np.random.default_rng(54);digest=hashlib.sha256()
    for _ in range(12):
        x=train.sample_cached_indices(first,implicit);y=train.sample_cached_indices(second,explicit)
        for name in x:
            np.testing.assert_array_equal(x[name],y[name]);digest.update(x[name].tobytes())
    # Captured independently from baseline71abb2bb's original sampler function.
    assert digest.hexdigest()=='b42d228c39c895c1e1822d08d9de62364d10fd06491bed016b19e9cb66a7f336'
    with pytest.raises(ValueError):train.build_cache(p,a,optimizer_version=4)


def test_v6_planned_balances_available_body_pose_target_and_retains_full_draws(tmp_path,monkeypatch):
    p,a,_=cache_fixture(tmp_path,monkeypatch);samples=a['samples']
    samples['source_histories'][0,:,7:10]=[.013,0,-.04] # retained only in full planned tree
    for body in ('p001120','p000758'):
        for pose in train.runtime_scope_poses():
            for target in ('R3','L3'):
                entry=deepcopy(a['sources'][0]);entry.update(subject_id=body,pose_id=pose,target_id=target,
                    source_id='planned:'+body+pose+target,windows=[{}])
                entry['metadata'].update(subject_id=body,target_id=target)
                index=len(a['sources']);a['sources'].append(entry)
                for key in ('source_histories','relative_setters','setter_is_pad'):
                    samples[key]=np.concatenate([samples[key],samples[key][1:2]])
                samples['source_indices']=np.r_[samples['source_indices'],index]
    cache=train.build_cache(p,a,optimizer_version=6);rng=np.random.default_rng(53)
    counts={name:{} for name in ('body','pose','target')};seen=set();native_full=set();planned_full=set()
    for _ in range(1000):
        ids=train.sample_cached_indices(rng,cache);native_full.update(ids['native'][16:]);planned_full.update(ids['planned'][4:])
        assert 0 not in ids['planned'][:4]
        for row in ids['planned'][:4]:
            source=a['sources'][samples['source_indices'][row]];labels=(source['subject_id'],source['pose_id'],source['target_id']);seen.add(labels)
            for name,label in zip(counts,labels):counts[name][label]=counts[name].get(label,0)+1
    assert len(seen)==8 and 0 in planned_full and {2,5,8}<=native_full
    assert all(len(values)==2 and all(abs(n-2000)<150 for n in values.values()) for values in counts.values())


def test_v6_dispatch_ipc_sources_buffered_audit_and_head(tmp_path,monkeypatch):
    assert hasattr(train,'CONFIG_SCHEMA_V6'), 'V6 dispatch required'
    from sim.ausc_policy import m2_given_contact_expanded_pilot_train as expanded
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep,m2_rebuild_paired_eval as paired
    sentinel=object()
    for name in ('validate_config','training_protocol','load_selected_policy','verify_selected','fit','validate_complete','model_binding'):
        monkeypatch.setattr(train,name,lambda *a,**k:sentinel)
    assert expanded.validate_config({'schema':train.CONFIG_SCHEMA_V6}) is sentinel
    assert expanded.training_protocol({'schema':train.CONFIG_SCHEMA_V6}) is sentinel
    assert expanded.load_selected_policy({'schema':train.SELECTION_SCHEMA_V6}) is sentinel
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.SELECTION_SCHEMA_V6 if ref=='selection' else train.CONFIG_SCHEMA_V6))
    assert expanded.verify_selected('selection','receipt') is sentinel and expanded.fit('config','out') is sentinel
    assert dep.validate_complete({'schema':train.PILOT_SCHEMA_V6}) is sentinel
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.PILOT_SCHEMA_V6))
    assert dep.model_binding('ref') is sentinel
    for adapter in (dep,paired):
        files=adapter.model_sources({'checkpoint_schema':train.PILOT_SCHEMA_V6})['files']
        assert 'm2_act_corridor_sampling.py' in files
        assert all(Path(ref['path']).parent==Path(train.__file__).parent for ref in files.values())
    from test_ausc_given_contact_pose import reference,metadata
    ref=reference(tmp_path);meta=dict(metadata(ref),episode_id='v6',target_id='L1');q=np.zeros(7)
    state=dep.runtime.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    descriptor=dict(metadata=meta,episode_id='v6',source_episode_id='native',model=dict(checkpoint={'sha256':'a'*64},checkpoint_schema=train.PILOT_SCHEMA_V6),source_config={'sha256':'b'*64},protocol={'sha256':'c'*64},scenario_id='s')
    for adapter in (dep,paired):
        request=adapter.make_request(descriptor,1,q,ref.pose_base,1.,metadata=meta,history=np.stack([state,state]),history_times=[0.,.05],context=dict(observed_at_s=.05,physics_row=12,context=['v6','L1',0,'FREE'],source_id='given-contact-'+ref.sha256))
        adapter.validate_request(request,descriptor,sequence=1)
        assert adapter.decode_response(adapter.response_for(request,np.zeros((20,7))),request).metadata['checkpoint_policy']==train.PILOT_SCHEMA_V6


def test_v6_code_is_additive_actual_paths_and_model_does_not_reset(monkeypatch):
    from test_ausc_act_training_v4 import head_spec
    assert hasattr(train,'CONFIG_SCHEMA_V6'), 'V6 source binding required'
    old=train.code_binding(version=5);new=train.code_binding(version=6)
    assert set(new['files'])-set(old['files'])=={'m2_act_corridor_sampling.py'}
    monkeypatch.setattr(train.runtime,'make_model',lambda *a,**k:('torch','policy',{}))
    monkeypatch.setattr(train.posture_head,'install',lambda *a:None)
    monkeypatch.setattr(train.posture_head,'initialize_training',lambda *a:pytest.fail('Head reset'))
    assert train.make_model(dict(schema=train.CONFIG_SCHEMA_V6,lerobot_root='/official',posture_head=head_spec()),device='cpu',training=True)[1]=='policy'


def test_v6_protocol_truthful_act8_and_authenticated_act6_step0(tmp_path,monkeypatch):
    from test_ausc_act_training_v5 import initial_fixture
    m,torch,ref,receipt,Tiny,_=initial_fixture(tmp_path,monkeypatch)
    config=dict(train.SCHEDULE_V6,schema=train.CONFIG_SCHEMA_V6,initialization=ref,posture_head=receipt['posture_head'])
    protocol=train.training_protocol(config)
    assert protocol['data_version']==5 and protocol['model_optimizer_version']==6
    assert protocol['data_schema']==data.SCHEMA and 'ACT8' in protocol['treatment'] and 'ACT7' not in protocol['treatment']
    initial=protocol['posture_initialization']
    assert 'ACT8' in initial['method'] and 'ACT7' not in initial['method']
    assert initial['inputs']['weights']==m.WEIGHTS and 'act6-terminal-train' in m.WEIGHTS['path']
    assert initial['historical_equivalence_schema']==m.EQUIVALENCE_SCHEMA
    assert train._warm_start(6) is m
    policy=Tiny();proof=m.initialize_policy(torch,policy,ref,normalization=receipt['normalization'],posture_head=receipt['posture_head'])
    assert len(proof['initial_tensor_proofs'])==54 and proof['optimizer_step']==0 and proof['final_linear_reset'] is False
    optimizer,settings=m.fresh_optimizer(torch,policy,lr=config['lr'],weight_decay=config['weight_decay'])
    assert not optimizer.state and settings['initial_step']==0


def test_v6_fit_saves_both_and_final_even_when500_better(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from test_ausc_act_training_v5 import initial_fixture
    m,torch,init,receipt,Tiny,_=initial_fixture(tmp_path,monkeypatch);policy=Tiny()
    config=dict(train.SCHEDULE_V6,schema=train.CONFIG_SCHEMA_V6,base='/fixture',dataset={},auxiliary={},evaluation_cohort={},posture_head=receipt['posture_head'],initialization=init)
    config_ref=m.rt.write_new(tmp_path/'fitconfig.json',config)
    prepared=dict(normalization=receipt['normalization'],scope={},optimization_scope={},native_scope={},window_order={'train':'a'*64,'validation':'b'*64},train={'fixture':'train'},validation={'fixture':'validation'},auxiliary=dict(counts={},samples={}),oracle_authentication=[],posture_target_audit={})
    monkeypatch.setattr(train,'validate_config',lambda c:c);monkeypatch.setattr(train,'load_inputs',lambda c:prepared)
    monkeypatch.setattr(train.runtime,'owned_gpu',lambda:nullcontext({}));monkeypatch.setattr(train.runtime,'_identity',lambda:dict(pid=1,pgid=1,sid=1))
    monkeypatch.setattr(train,'make_model',lambda *a,**k:(torch,policy,{}));monkeypatch.setattr(train,'sum',lambda *a:2658823,raising=False)
    monkeypatch.setattr(train,'validate_official_forward',lambda *a:None);monkeypatch.setattr(train.runtime,'execution_settings',lambda *a,**k:{})
    def cache(*a,**kwargs):
        assert kwargs['optimizer_version']==6
        return dict(terminal_pool_inventory={'old6mm':True},corridor_inventory={'fixture':True})
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
    assert result['schema']==train.PILOT_SCHEMA_V6 and result['selected_step']==result['completed_steps']==1000
    assert [r['step'] for r in result['evaluation_candidates']]==[500,1000]
    assert result['data_version']==5 and result['model_optimizer_version']==6
    assert train.runtime.read_bound(result['corridor_inventory'])=={'fixture':True}
    for row in result['evaluation_candidates']:
        packet=torch.load(train.runtime.bound_path(row['checkpoint']),weights_only=True)
        assert packet['schema']==train.PILOT_SCHEMA_V6 and packet['step']==row['step']
        assert packet['corridor_inventory']==result['corridor_inventory'] and 'optimizer_state_dict' not in packet
    train._validate_candidates(result)


@pytest.mark.parametrize('failure_point',['warm_start','cache'])
def test_v6_fit_failure_before_inventory_preserves_original_error_and_artifacts(tmp_path,monkeypatch,failure_point):
    from contextlib import nullcontext
    from test_ausc_act_training_v5 import initial_fixture
    m,torch,init,receipt,Tiny,_=initial_fixture(tmp_path,monkeypatch);policy=Tiny()
    config=dict(train.SCHEDULE_V6,schema=train.CONFIG_SCHEMA_V6,base='/fixture',dataset={},auxiliary={},evaluation_cohort={},posture_head=receipt['posture_head'],initialization=init)
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
    assert failed['schema']==train.PILOT_SCHEMA_V6 and failed['complete'] is False and failed['completed_steps']==0
    assert failed['error']=='ValueError: '+message and not (output/'COMPLETE.json').exists()
    trace=m.rt.bound_path(failed['traceback']).read_text()
    assert 'ValueError: '+message in trace and 'KeyError' not in trace
    partial=torch.load(m.rt.bound_path(failed['partial_checkpoint']),weights_only=True)
    assert partial['complete'] is False and partial['step']==0 and partial['error']==failed['error']
    assert partial.get('corridor_inventory') is None
    with pytest.raises(ValueError):train._corridor_inventory(partial,{})


def test_v6_bound_inventory_rejects_arrays_and_source_eligibility_forgery(tmp_path,monkeypatch):
    m=corridor();prepared,auxiliary,_=cache_fixture(tmp_path,monkeypatch)
    arrays=dict(path='/existing/v5.npz',bytes=1,sha256='b'*64);auxiliary['array_reference']=arrays
    inventory=m.build_inventory(prepared,auxiliary)
    assert hasattr(m,'validate_binding'), 'Metadata-only deployment inventory binding required'
    m.validate_binding(inventory,native=prepared['preparation'],auxiliary_metadata=dict(sources=auxiliary['sources']),auxiliary_arrays=arrays)
    for fault in ('arrays','gap','source','row','bucket'):
        bad=deepcopy(inventory);pool=bad['pools']['native']
        if fault=='arrays':pool['arrays']['sha256']='0'*64
        elif fault=='gap':pool['sources'][0]['actual_gap_m']=.04
        elif fault=='source':pool['rows'][0]['source']='unknown'
        elif fault=='row':pool['rows'][0]['row_index']=8
        else:pool['bucket_counts']['body']={'invented':pool['row_count']}
        with pytest.raises(ValueError):m.validate_binding(bad,native=prepared['preparation'],auxiliary_metadata=dict(sources=auxiliary['sources']),auxiliary_arrays=arrays)


def completed_fixture(tmp_path,monkeypatch):
    """Reuse the existing completed-run fixture; source metadata admission is isolated above."""
    from test_ausc_act_training_v5 import completed_fixture as previous
    m,torch,value,receipt=previous(tmp_path,monkeypatch)
    config=m.rt.read_bound(value['inputs']['config']);config['schema']=train.CONFIG_SCHEMA_V6
    value.update(schema=train.PILOT_SCHEMA_V6,schedule=config,data_version=5,model_optimizer_version=6,protocol=train.training_protocol(config))
    value['inputs'].update(config=m.rt.write_new(tmp_path/'V6CONFIG.json',config),code=train.code_binding(version=6))
    source_inventory=train.contribution_inventory(config);pools={}
    for name in ('native','planned','conditional'):
        rows=[]
        for i,(source,entry) in enumerate(source_inventory[name].items()):
            rows.append(dict(row_index=i,source=source,body=entry['subject_id'],pose=entry['pose_id'],target=entry['target_id'],anchor=entry['anchor'],error_m=0.,actual_gap_m=.04))
        pools[name]=corridor()._pool(rows,threshold=.006 if name=='conditional' else .012)
    inventory=dict(schema=corridor().SCHEMA,data_version=5,model_optimizer_version=6,pools=pools)
    value['corridor_inventory']=m.rt.write_new(tmp_path/'CORRIDOR.json',inventory)
    monkeypatch.setattr(train,'_corridor_inventory',lambda *a:inventory)
    for key in ('optimization','selected_optimization'):
        optimization=value[key];optimization['corridor_draws']=corridor().new_draws()
        for name,(n,size) in corridor().DRAWS.items():
            records={str(i):1000*n-len(pools[name]['rows'])+1 if i==0 else 1 for i in range(len(pools[name]['rows']))}
            counts=optimization['corridor_draws'][name];counts.update(draws=1000*n,denominator=1000*size,rows=records)
            for index,count in records.items():
                row=pools[name]['rows'][int(index)]
                for label in corridor().LABELS:
                    if label in row:counts[label][row[label]]=counts[label].get(row[label],0)+count
    selected=dict(value,schema=train.SELECTION_SCHEMA_V6)
    selected_ref=m.rt.write_new(tmp_path/'V6SELECTED.json',selected)
    oldcpu=m.rt.read_bound(value['cpu_equivalence'])
    value['cpu_equivalence']=m.rt.write_new(tmp_path/'V6CPU.json',dict(oldcpu,schema=train.CPU_SCHEMA_V6,selection=selected_ref,
        data_version=5,model_optimizer_version=6,corridor_inventory=value['corridor_inventory']))
    return m,torch,value,receipt


def test_v6_complete_and_typed_cpu_binding_reject_counter_forgery(tmp_path,monkeypatch):
    actual_binding=train._corridor_inventory
    m,_,value,_=completed_fixture(tmp_path,monkeypatch)
    assert train.validate_complete(value) is value
    assert train.model_binding(m.rt.write_new(tmp_path/'V6COMPLETE.json',value))[0]['checkpoint_schema']==train.PILOT_SCHEMA_V6
    for fault in ('old_cpu','lost500','data_version','biased_native','planned_rows','conditional_body','head_reset'):
        bad=deepcopy(value)
        if fault=='old_cpu':bad['cpu_equivalence']=m.rt.write_new(tmp_path/'OLDCPU.json',dict(m.rt.read_bound(value['cpu_equivalence']),schema=train.CPU_SCHEMA_V5))
        elif fault=='lost500':bad['evaluation_candidates'].pop(0)
        elif fault=='data_version':bad['data_version']=6
        elif fault=='biased_native':bad['optimization']['corridor_draws']['native']['draws']-=1
        elif fault=='planned_rows':bad['optimization']['corridor_draws']['planned']['rows']={'99999':4000}
        elif fault=='conditional_body':bad['optimization']['corridor_draws']['conditional']['body']={'p000923':4000}
        else:bad['initialization']['optimizer']['initial_step']=18000
        with pytest.raises(ValueError):
            if fault=='old_cpu':train.model_binding(m.rt.write_new(tmp_path/'BADCOMPLETE.json',bad))
            elif fault=='data_version':actual_binding(bad,{})
            else:train.validate_complete(bad)


def test_v6_inference_loads_only_own_v6_checkpoint_once(tmp_path,monkeypatch):
    from types import SimpleNamespace
    m,torch,value,receipt=completed_fixture(tmp_path,monkeypatch)
    software={key:{} for key in ('vendor','dependencies','python','prefix','python_version','source_bindings')}
    value['inputs']['software']=m.rt.write_new(tmp_path/'OWN_SOFTWARE.json',software)
    packet=dict(value,step=1000,state_dict={'own':'V6'},normalization=receipt['normalization'],optimization=value['selected_optimization'])
    loaded=[];policy=SimpleNamespace(load_state_dict=lambda state,strict:loaded.append((state,strict)),eval=lambda:None)
    monkeypatch.setattr(train,'make_model',lambda *a,**k:(torch,policy,software));monkeypatch.setattr(train,'validate_official_forward',lambda *a:None)
    for name in ('initialize_policy','export','require_archive'):
        monkeypatch.setattr(m,name,lambda *a,**k:pytest.fail('V6 inference replayed historical model initialization'))
    monkeypatch.setattr(data,'load_preparation',lambda *a,**k:pytest.fail('Inference decoded numerical training data'))
    calls=[];monkeypatch.setattr(torch,'load',lambda *a,**k:calls.append(a) or packet)
    assert train.load_selected_policy(value)[2] is packet
    assert len(calls)==1 and loaded==[({'own':'V6'},True)]


def test_v6_buffered_native_and_runtime_head_preserve_current_inventory_and_gates(tmp_path,monkeypatch):
    from test_ausc_given_contact_pilot_native_audit import setup
    from test_ausc_act_training_v4 import head_spec
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep
    m,previous,c,*_=setup(tmp_path,monkeypatch);previous.close()
    model=dict(checkpoint_schema=train.PILOT_SCHEMA_V6,checkpoint={'sha256':'b'*64});c.policy_receipt['model']=model
    expected={'v6':'actual-current-source'};monkeypatch.setattr(dep,'model_sources',lambda model:expected)
    audit=m.BufferedNativeAudit(tmp_path/'v6.jsonl.gz',c,dep,{dep.ROUTE_KEY:dict(model=model,source_config={'sha256':'c'*64})})
    assert audit.sources==expected and audit.close()['complete'] is True
    spec=head_spec();calls=[];monkeypatch.setattr(train.posture_head,'validate_bound_spec',lambda *a,**k:calls.append(k))
    complete=dict(schema=train.PILOT_SCHEMA_V6,protocol=dict(posture_head=spec))
    train.posture_head.validate_runtime_head(complete,dict(posture_limits=spec['physical_posture_profile']))
    assert calls==[{'version':2}]
    with pytest.raises(ValueError,match='posture profile'):train.posture_head.validate_runtime_head(complete,dict(posture_limits={}))


def test_v6_cpu_verification_binds_data_model_and_inventory(tmp_path,monkeypatch):
    m,torch,value,receipt=completed_fixture(tmp_path,monkeypatch)
    prepared={key:value[key] for key in ('scope','native_scope','optimization_scope','window_order')}
    prepared.update(train={},validation={})
    monkeypatch.setattr(train,'load_inputs',lambda *a:prepared)
    monkeypatch.setattr(train,'load_selected_policy',lambda *a:(torch,None,dict(normalization=receipt['normalization'])))
    monkeypatch.setattr(train.runtime,'execution_settings',lambda *a,**k:{})
    monkeypatch.setattr(train.runtime,'infer',lambda *a,**k:np.zeros((1,20,7)))
    monkeypatch.setattr(train.runtime,'load_predictions',lambda *a,**k:np.zeros((1,20,7)))
    monkeypatch.setattr(train.runtime,'metrics',lambda *a:{})
    monkeypatch.setattr(train.runtime,'compare_devices',lambda *a:{'passed':True})
    selected=m.rt.write_new(tmp_path/'OWN_SELECTED.json',dict(value,schema=train.SELECTION_SCHEMA_V6,selected_predictions={'train':{},'validation':{}}))
    result=m.rt.read_bound(train.verify_selected(selected,tmp_path/'OWN_CPU.json'))
    assert result['schema']==train.CPU_SCHEMA_V6 and result['data_version']==5
    assert result['model_optimizer_version']==6 and result['corridor_inventory']==value['corridor_inventory']


def test_v6_old_and_new_conditional_counts_cannot_disagree(tmp_path,monkeypatch):
    m,_,value,_=completed_fixture(tmp_path,monkeypatch)
    bad=deepcopy(value);old=bad['optimization']['conditional_terminal_draws'];ids=list(old['source'])
    old['source'][ids[0]]-=1;old['source'][ids[1]]+=1
    inventory=train.contribution_inventory({})
    old['body']={inventory['conditional'][source]['subject_id']:count for source,count in old['source'].items()}
    totals=bad['optimization']['gradient_source_examples']['conditional'];totals[ids[0]]-=1;totals[ids[1]]+=1
    bad['optimization']['gradient_subject_examples']['conditional']={inventory['conditional'][source]['subject_id']:count for source,count in totals.items()}
    with pytest.raises(ValueError,match='conditional|corridor|Corridor'):train.validate_complete(bad)
