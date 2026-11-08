"""Synthetic ACT7 integration contracts; no production artifacts or runtime."""
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy import m2_act_auxiliary_train as train
from sim.ausc_policy import m2_act_phase_data as data


def warm():
    name='sim.ausc_policy.m2_act_phase_warm_start'
    assert importlib.util.find_spec(name), 'Separate authentic ACT6 initializer required'
    return importlib.import_module(name)


def test_v5_explicit_schema_schedule_and_fixed_final_selection():
    assert hasattr(train,'CONFIG_SCHEMA_V5'), 'Explicit V5 trainer required'
    for key in ('CONFIG','PILOT','SELECTION','CPU','PROTOCOL'):
        assert train._version({'schema':getattr(train,key+'_SCHEMA_V5')})==5
    assert train._api(5) is data and train._groups(5)==train.GROUPS_V3
    assert train.SCHEDULE_V5==dict(train.SCHEDULE,steps=1000)
    assert train.expanded.evaluation_steps(train.SCHEDULE_V5)==[500,1000]
    rows=[dict(step=500,validation={'act':{'full_chunk_mae_rad':.01}}),dict(step=1000,validation={'act':{'full_chunk_mae_rad':.02}})]
    assert train.selected_step(rows,version=5)==1000
    assert train.selected_step(rows,version=4)==500 and train.SCHEDULE['steps']==20000
    with pytest.raises(ValueError):train.selected_step(rows[:1],version=5)


@pytest.mark.parametrize('schema',['m2_act_auxiliary_training_config_v99','m2_act_auxiliary_act_pilot_v0','unknown'])
def test_unknown_schema_never_falls_back(schema):
    with pytest.raises(ValueError,match='schema|version'):train._version({'schema':schema})


def test_act6_warm_exact_pins_and_not_act5():
    m=warm()
    assert m.SCHEMA=='m2_act6_selected_cpu_initialization_v1'
    assert m.SOURCE28_COMMIT==data.SOURCE28_COMMIT and m.SOURCE28_FILES==1673
    assert m.SOURCE28_MANIFEST==data.PRODUCTION_INPUTS.source28_manifest
    expected=[(m.COMPLETE,562776,'c4b145d38e4374de317cf4c070ed5a74c16cc9a54d6c992b2d75c210a5d672f1'),
        (m.WEIGHTS,10843905,'822bbaf5298695a3ceeb3ab00e741e49a220bca7d458824e20beabb9ef542efe'),
        (m.CPU_EQUIVALENCE,13263,'aa73d61b8a4e3fbe4cc03d7454693419e6bd587d094f6ac8d18a7f3b12bb3138'),
        (m.SELECTED,299637,'c1c0ad76d3173ce2fe0d421b18aafb1d068141610266a73e6f0f86b735ef1292'),
        (m.OWNER,8157,'0b6d1943fa936296f955946f80dea635a23677e826efde177d9825c2b892b494')]
    assert all(ref['bytes']==size and ref['sha256']==sha for ref,size,sha in expected)
    assert m.PROBE_TRAIN_INDICES==[0,64,4186,8372]
    assert m.OWNER['path']=='/opt/wam-runtime/fast-wam/runs/ausc_policy/act6-terminal-train-20261107-v1/result.json'
    assert 'original.model_binding(' in m._PROGRAM and 'original.load_selected_policy(' in m._PROGRAM
    assert '18000' in m._PROGRAM and '19000' not in m._PROGRAM and 'owner' in m._PROGRAM
    assert 'initialize_training' not in m._PROGRAM


def test_v5_model_head_never_reset(monkeypatch):
    from test_ausc_act_training_v4 import head_spec
    p=SimpleNamespace();monkeypatch.setattr(train.runtime,'make_model',lambda *a,**k:('torch',p,{}))
    monkeypatch.setattr(train.posture_head,'install',lambda *a:None)
    monkeypatch.setattr(train.posture_head,'initialize_training',lambda *a:pytest.fail('ACT6 head reset'))
    assert train.make_model(dict(schema=train.CONFIG_SCHEMA_V5,lerobot_root='/official',posture_head=head_spec()),device='cpu',training=True)[1] is p


def v5_cache(tmp_path,monkeypatch):
    from test_ausc_act_training_v4 import v4_cache_fixture
    p,a,n=v4_cache_fixture(tmp_path)
    # Unequal windows per body and both eligible and ineligible windows.
    s=a['samples'];s.update(schema=data.SCHEMA,optimization_admission=data._admission(True))
    sources=a['sources'];base=deepcopy(sources[3]);sources[3]['windows']=[{}]
    sources[3]['pose_id']='p000476_a001231_f0023'
    additions=[]
    for body in ('p000758','p001120'):
        for target in ('L3','R3'):
            for anchor in ('near60mm','near30mm'):
                entry=deepcopy(base);entry.update(subject_id=body,target_id=target,anchor=anchor,source_id=body+target+anchor,windows=[{}])
                entry['metadata'].update(subject_id=body,target_id=target)
                additions.append(entry)
    start=len(s['source_indices']);sources.extend(additions)
    for key in ('source_histories','relative_setters','setter_is_pad'):
        s[key]=np.concatenate([s[key],np.repeat(s[key][-1:],len(additions),axis=0)])
    s['source_indices']=np.r_[s['source_indices'],np.arange(5,5+len(additions))]
    for source in sources:
        source['metadata'].update(subject_id=source['subject_id'],split='train',target_id=source['target_id'])
    # Use the real pure inventory calculation, supplying only synthetic bound plans.
    s['source_histories'][:,:,7:10]=[0.,0.,0.]
    for i in range(len(additions)):
        s['source_histories'][start+i,:,7:10]=[0.,0.,-.03 if i%2 else -.05]
    gaps={entry['source_id']:(.03 if i%2 else .05) for i,entry in enumerate(additions)}
    monkeypatch.setattr(data,'_plan',lambda meta,source:(gaps.get(source['source_id'],.04),None,None))
    monkeypatch.setattr(train,'_api',lambda v:data)
    return p,a,start


def test_terminal_4_plus_4_balanced_available_strata_and_exact_draw_counts(tmp_path,monkeypatch):
    p,a,start=v5_cache(tmp_path,monkeypatch)
    cache=train.build_cache(p,a);inventory=cache['terminal_pool_inventory']
    assert len(inventory['eligible_row_indices'])==8 and inventory['missing_strata']
    counts={key:{} for key in ('source','body','pose','target','anchor')}
    optimization=train.new_optimization(version=5)
    rng=np.random.default_rng(123)
    for _ in range(1000):
        ids=train.sample_cached_indices(rng,cache)
        assert {k:len(v) for k,v in ids.items()}==dict(native=32,planned=8,conditional=8,oracle=16)
        assert np.all(ids['conditional'][:4]>=start)
        assert set(ids)==set(train.GROUPS_V3)
        train.record_draws(optimization,cache,ids)
    terminal=optimization['conditional_terminal_draws']
    assert terminal['draws']==4000 and terminal['denominator']==8000
    for key in counts:assert sum(terminal[key].values())==4000
    assert abs(terminal['body']['p001120']-2000)<150
    assert abs(terminal['target']['L3']-2000)<150
    assert set(terminal['source'])=={s['source_id'] for s in a['sources'][5:]}
    assert optimization['gradient_examples']['oracle']==16000


def test_terminal_empty_fails_but_missing_individual_strata_does_not(tmp_path,monkeypatch):
    p,a,start=v5_cache(tmp_path,monkeypatch)
    a['samples']['source_histories'][:,:,7:10]=0.
    with pytest.raises(ValueError,match='terminal|prepose'):train.build_cache(p,a)


def test_production_trainer_rejects_fixture_profile_before_loading(monkeypatch):
    monkeypatch.setattr(data,'metadata',lambda *a,**k:({},dict(input_profile='local_synthetic_fixture',sources=[])))
    with pytest.raises(ValueError,match='production|Production'):train.auxiliary_metadata({},version=5)


def test_v5_dispatch_ipc_and_schema_specific_inventory(tmp_path,monkeypatch):
    # Reuse the meaningful V4 IPC exercise with the explicitly new schema family.
    from sim.ausc_policy import m2_given_contact_expanded_pilot_train as expanded
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep,m2_rebuild_paired_eval as paired
    sentinel=object()
    for name in ('validate_config','training_protocol','load_selected_policy','verify_selected','fit','validate_complete','model_binding'):
        monkeypatch.setattr(train,name,lambda *a,**k:sentinel)
    assert expanded.validate_config({'schema':train.CONFIG_SCHEMA_V5}) is sentinel
    assert expanded.training_protocol({'schema':train.CONFIG_SCHEMA_V5}) is sentinel
    assert expanded.load_selected_policy({'schema':train.SELECTION_SCHEMA_V5}) is sentinel
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.SELECTION_SCHEMA_V5 if ref=='selection' else train.CONFIG_SCHEMA_V5))
    assert expanded.verify_selected('selection','receipt') is sentinel and expanded.fit('config','out') is sentinel
    assert dep.validate_complete({'schema':train.PILOT_SCHEMA_V5}) is sentinel
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.PILOT_SCHEMA_V5))
    assert dep.model_binding('ref') is sentinel
    for adapter in (dep,paired):
        assert 'm2_act_phase_warm_start.py' in adapter.model_sources({'checkpoint_schema':train.PILOT_SCHEMA_V5})['files']
    from test_ausc_given_contact_pose import reference,metadata
    ref=reference(tmp_path);meta=dict(metadata(ref),episode_id='v5',target_id='L1');q=np.zeros(7)
    state=dep.runtime.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    descriptor=dict(metadata=meta,episode_id='v5',source_episode_id='native',model=dict(checkpoint={'sha256':'a'*64},checkpoint_schema=train.PILOT_SCHEMA_V5),source_config={'sha256':'b'*64},protocol={'sha256':'c'*64},scenario_id='s')
    for adapter in (dep,paired):
        request=adapter.make_request(descriptor,1,q,ref.pose_base,1.,metadata=meta,history=np.stack([state,state]),history_times=[0.,.05],context=dict(observed_at_s=.05,physics_row=12,context=['v5','L1',0,'FREE'],source_id='given-contact-'+ref.sha256))
        adapter.validate_request(request,descriptor,sequence=1)
        assert adapter.decode_response(adapter.response_for(request,np.zeros((20,7))),request).metadata['checkpoint_policy']==train.PILOT_SCHEMA_V5


def test_v5_code_additive_actual_origins_v4_inventory_unchanged():
    old=train.code_binding(version=4);new=train.code_binding(version=5)
    assert set(old['files'])<set(new['files'])
    assert {'m2_act_phase_data.py','m2_act_phase_warm_start.py'}<=set(new['files'])-set(old['files'])
    root=Path(train.__file__).resolve().parent
    assert all(Path(ref['path'])==root/name for name,ref in new['files'].items())
    changed=deepcopy(new);changed['code_commit']='copied'
    with pytest.raises(ValueError):train.validate_code(changed,version=5)


def initial_fixture(tmp_path,monkeypatch):
    from test_ausc_act_training_v4 import initialization_fixture
    _,torch,_,value,Tiny,selection,complete,cpu=initialization_fixture(tmp_path,monkeypatch)
    m=warm()
    class AllTensors(Tiny):
        def __init__(self):
            super().__init__(7,7)
            for i in range(52):self.register_buffer('buffer'+str(i),torch.tensor(float(i)))
    model=AllTensors();state=torch.load(m.rt.bound_path(value['state_dict']),weights_only=True)
    model.load_state_dict(state,strict=False)
    path=tmp_path/'ALL54.pt';torch.save(model.state_dict(),path)
    software=m.rt.read_bound(value['inputs']['software'])
    software['source_bindings']={m.SOURCE28_ROOT+'/sim/ausc_policy/m2_act.py':'a'*64}
    software_ref=m.rt.write_new(tmp_path/'software28.json',software)
    selection.update(schema=train.SELECTION_SCHEMA_V4,selected_step=18000,best=m.WEIGHTS)
    selection['inputs'].update(software=software_ref,code=dict(code_commit=m.SOURCE28_COMMIT,files={'m2_act.py':dict(path=m.SOURCE28_ROOT+'/sim/ausc_policy/m2_act.py',bytes=1,sha256='a'*64)}))
    complete=dict(selection,schema=train.PILOT_SCHEMA_V4,complete=True,completed_steps=20000,cpu_equivalence=m.CPU_EQUIVALENCE)
    cpu=dict(selection=m.SELECTED,checkpoint=m.WEIGHTS,selected_step=18000,passed=True)
    owner=dict(run_id='act6-terminal-train-20261107-v1',code_commit=m.SOURCE28_COMMIT,completed=True,gpu_released=True,
        stage=dict(stage='m2_given_contact_expanded_pilot_train',passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[]))
    value.update(schema=m.SCHEMA,origin='ACT6 selected18000; original28 CPU authenticated',selected_step=18000,
        source28=dict(code_commit=m.SOURCE28_COMMIT,file_count=1673,root=m.SOURCE28_ROOT),
        state_dict=m.rt.reference(path),tensor_proofs=m.tensor_proofs(model.state_dict()))
    value.pop('source26')
    value['inputs'].update(complete=m.COMPLETE,weights=m.WEIGHTS,cpu_equivalence=m.CPU_EQUIVALENCE,selection=m.SELECTED,
        owner=m.OWNER,source_manifest=m.SOURCE28_MANIFEST,software=software_ref)
    original=m.rt.read_bound
    pinned={m.COMPLETE['path']:complete,m.CPU_EQUIVALENCE['path']:cpu,m.SELECTED['path']:selection,m.OWNER['path']:owner}
    monkeypatch.setattr(m.rt,'read_bound',lambda ref:deepcopy(pinned[ref['path']]) if ref.get('path') in pinned else original(ref))
    ref=m.rt.write_new(tmp_path/'INIT28.json',value)
    return m,torch,ref,value,AllTensors,owner


def test_act6_all54_bits_step0_probes_and_fresh_optimizer(tmp_path,monkeypatch):
    m,torch,ref,value,Tiny,owner=initial_fixture(tmp_path,monkeypatch)
    model=Tiny();result=m.initialize_policy(torch,model,ref,normalization=value['normalization'],posture_head=value['posture_head'])
    assert result['prediction_max_abs_rad']==0 and result['initial_tensor_proofs']==value['tensor_proofs']
    optimizer,settings=m.fresh_optimizer(torch,model,lr=1e-4,weight_decay=1e-4)
    assert not optimizer.state and settings['initial_step']==0
    with pytest.raises(ValueError,match='repeat|reset'):m.initialize_policy(torch,model,ref,normalization=value['normalization'],posture_head=value['posture_head'])
    bad=deepcopy(value);bad['tensor_proofs']['buffer0']['sha256']='0'*64
    badref=m.rt.write_new(tmp_path/'BADBITS.json',bad)
    with pytest.raises(ValueError,match='tensor'):m.initialize_policy(torch,Tiny(),badref,normalization=value['normalization'],posture_head=value['posture_head'])
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Receipt metadata loaded tensors'))
    monkeypatch.setattr(m,'require_archive',lambda:pytest.fail('Inference replayed full original archive'))
    assert m.validate_receipt(ref)==value
    owner['stage']['exit_code']=1
    with pytest.raises(ValueError,match='owner|normal'):m.validate_receipt(ref)


def test_v5_fit_saves_both_states_and_selects_final_even_when500_better(tmp_path,monkeypatch):
    from contextlib import nullcontext
    m,torch,init,value,Tiny,_=initial_fixture(tmp_path,monkeypatch)
    model=Tiny()
    config=dict(train.SCHEDULE_V5,schema=train.CONFIG_SCHEMA_V5,base='/fixture',dataset={},auxiliary={},evaluation_cohort={},posture_head=value['posture_head'],initialization=init)
    ref=m.rt.write_new(tmp_path/'fitconfig.json',config)
    prepared=dict(normalization=value['normalization'],scope={},optimization_scope={},native_scope={},window_order={'train':'a'*64,'validation':'b'*64},train={'fixture':'train'},validation={'fixture':'validation'},auxiliary=dict(counts={},samples={}),oracle_authentication=[],posture_target_audit={})
    monkeypatch.setattr(train,'validate_config',lambda c:c);monkeypatch.setattr(train,'load_inputs',lambda c:prepared)
    monkeypatch.setattr(train.runtime,'owned_gpu',lambda:nullcontext({}))
    monkeypatch.setattr(train.runtime,'_identity',lambda:dict(pid=1,pgid=1,sid=1))
    monkeypatch.setattr(train,'make_model',lambda *a,**k:(torch,model,{}))
    monkeypatch.setattr(train,'sum',lambda *a:2658823,raising=False)
    monkeypatch.setattr(train,'validate_official_forward',lambda *a:None)
    monkeypatch.setattr(train.runtime,'execution_settings',lambda *a,**k:{})
    monkeypatch.setattr(train,'build_cache',lambda *a,**k:dict(terminal_pool_inventory={'fixture':True}))
    monkeypatch.setattr(train,'sample_cached_indices',lambda *a:{})
    monkeypatch.setattr(train,'cached_batches',lambda *a:{})
    monkeypatch.setattr(train,'record_draws',lambda *a:None)
    monkeypatch.setattr(train,'mixed_loss',lambda *a,**k:(model.weight.square().mean(),{}))
    monkeypatch.setattr(train.posture_head,'diagnostics',lambda *a:{})
    monkeypatch.setattr(train.scope,'validate_train16_metadata',lambda *a,**k:None)
    calls=[]
    monkeypatch.setattr(train.runtime,'infer',lambda *a,**k:np.zeros((1,20,7)))
    def metrics(*a):
        calls.append(None)
        return {'act':{'full_chunk_mae_rad':.01 if len(calls)<=2 else .02}}
    monkeypatch.setattr(train.runtime,'metrics',metrics)
    def child(argv,**kw):
        assert '--verify-selected' in argv
        train.runtime.write_new(Path(argv[-1]),{'passed':True})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(train,'code_binding',lambda **k:{'fixture_only':True})
    monkeypatch.setattr(train.subprocess,'run',child)
    result=train.fit(ref,tmp_path/'fit')
    assert result['selected_step']==result['completed_steps']==1000
    assert [r['step'] for r in result['evaluation_candidates']]==[500,1000]
    assert result['evaluations'][0]['validation']['act']['full_chunk_mae_rad']<result['evaluations'][1]['validation']['act']['full_chunk_mae_rad']
    for row in result['evaluation_candidates']:
        packet=torch.load(train.runtime.bound_path(row['checkpoint']),weights_only=True)
        assert packet['step']==row['step'] and packet['schema']==train.PILOT_SCHEMA_V5
        assert 'optimizer_state_dict' not in packet
    train._validate_candidates(result)
    assert result['selection_policy']==train.FINAL_SELECTION_V5


def test_v5_buffered_native_audit_uses_v5_inventory(tmp_path,monkeypatch):
    from test_ausc_given_contact_pilot_native_audit import setup
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep
    m,previous,c,*_=setup(tmp_path,monkeypatch);previous.close()
    model=dict(checkpoint_schema=train.PILOT_SCHEMA_V5,checkpoint={'sha256':'b'*64})
    c.policy_receipt['model']=model
    expected={'v5':'actual-source'}
    monkeypatch.setattr(dep,'model_sources',lambda model:expected)
    audit=m.BufferedNativeAudit(tmp_path/'v5.jsonl.gz',c,dep,{dep.ROUTE_KEY:dict(model=model,source_config={'sha256':'c'*64})})
    assert audit.sources==expected and audit.close()['complete'] is True


def completed_fixture(tmp_path,monkeypatch):
    from test_ausc_act_auxiliary_train import completed_run
    value=completed_run(tmp_path,monkeypatch)
    legacy=tmp_path/'old';legacy.mkdir()
    m,torch,init,receipt,_,_=initial_fixture(legacy,monkeypatch)
    config=m.rt.read_bound(value['inputs']['config'])
    config.update(train.SCHEDULE_V5,schema=train.CONFIG_SCHEMA_V5,initialization=init,posture_head=receipt['posture_head'])
    initial=dict(schema=m.EQUIVALENCE_SCHEMA,passed=True,initialization=init,initial_tensor_proofs=receipt['tensor_proofs'],prediction_max_abs_rad=0.,prediction_rtol=1e-5,prediction_atol=1e-6,prediction_device='cuda:0',probe_split='train',probe_indices=m.PROBE_TRAIN_INDICES,optimizer_step=0,optimizer_imported=False,scheduler_imported=False,final_linear_reset=False)
    initial_ref=m.rt.write_new(tmp_path/'STEP0.json',initial)
    value.update(schema=train.PILOT_SCHEMA_V5,schedule=config,completed_steps=1000,selected_step=1000,
        protocol=train.training_protocol(config),selection_policy=deepcopy(train.FINAL_SELECTION_V5),
        initialization=dict(receipt=init,equivalence=initial_ref,optimizer=dict(name='AdamW',initial_step=0,initial_state_entries=0,lr=config['lr'],weight_decay=config['weight_decay'],betas=[.9,.999],eps=1e-8,optimizer_imported=False,scheduler_imported=False)))
    value['inputs'].update(config=m.rt.write_new(tmp_path/'V5CONFIG.json',config),initialization=init,initialization_equivalence=initial_ref,code=train.code_binding(version=5))
    subjects=value['optimization_scope']['optimized_train_ids'];native=value['native_scope']['train']['subject_ids']
    kinds=dict(native='native',planned='planned_ideal_tracking',conditional='conditional_ideal_continuation',oracle='student_oracle_unexecuted')
    inventory={name:{name+':'+subject:dict(subject_id=subject,kind=kinds[name],pose_id='p000474_a001230_f0000',target_id='R3',anchor='near60mm') for subject in (native if name=='native' else subjects)} for name in train.GROUPS_V3}
    def draws(steps):
        result=train.new_optimization(version=5)
        for name,g in train.GROUPS_V3.items():
            counts={source:steps*g['size']-len(inventory[name])+1 if i==0 else 1 for i,source in enumerate(inventory[name])}
            result['gradient_examples'][name]=steps*g['size'];result['gradient_source_examples'][name]=counts
            result['gradient_subject_examples'][name]={inventory[name][source]['subject_id']:count for source,count in counts.items()}
            result['gradient_kind_examples'][name]={kinds[name]:steps*g['size']}
        source={key:steps*4-len(inventory['conditional'])+1 if i==0 else 1 for i,key in enumerate(inventory['conditional'])}
        result.update(actual_optimized_train_ids=subjects,conditional_terminal_draws=dict(draws=steps*4,denominator=steps*8,source=source,body={inventory['conditional'][key]['subject_id']:count for key,count in source.items()},pose={'p000474_a001230_f0000':steps*4},target={'R3':steps*4},anchor={'near60mm':steps*4}))
        return result
    value.update(optimization=draws(1000),selected_optimization=draws(1000))
    rows=deepcopy(value['evaluations'][:2]);rows[1]['validation']['act']['full_chunk_mae_rad']=.1
    value.update(evaluations=rows,selected_metrics={key:rows[1][key] for key in train.NATIVE_COUNTS})
    candidates=[]
    for step in (500,1000):
        path=tmp_path/('candidate-'+str(step)+'.pt');path.write_bytes(str(step).encode())
        candidates.append(dict(step=step,checkpoint=m.rt.reference(path)))
    value.update(evaluation_candidates=candidates,best=candidates[-1]['checkpoint'],terminal_pool_inventory={'fixture_only':True})
    counts=dict(inherited_sources=940,inherited_origins=35988,new_measured_query_origins=4,new_valid_computed_future_cells=58,new_physical_demonstrations=0)
    value['auxiliary_counts']=counts
    monkeypatch.setattr(train,'native_normalization',lambda c:receipt['normalization'])
    monkeypatch.setattr(train,'contribution_inventory',lambda c:inventory)
    monkeypatch.setattr(train,'auxiliary_metadata',lambda *a,**k:dict(counts=counts))
    selected=dict(value,schema=train.SELECTION_SCHEMA_V5)
    selected_ref=m.rt.write_new(tmp_path/'V5SELECTED.json',selected)
    comparison=dict(passed=True,prediction_rtol=1e-5,prediction_atol=1e-6,metric_rtol=1e-5,metric_atol=1e-7)
    value['cpu_equivalence']=m.rt.write_new(tmp_path/'V5CPU.json',dict(schema=train.CPU_SCHEMA_V5,passed=True,checkpoint=value['best'],selected_step=1000,scope=value['scope'],optimization_scope=value['optimization_scope'],selection=selected_ref,initialization=value['initialization'],comparisons={s:deepcopy(comparison) for s in train.NATIVE_COUNTS},**train.FLAGS))
    return m,torch,value,receipt


def test_v5_complete_fixed_final_and_bound_candidates_and_counts(tmp_path,monkeypatch):
    m,_,value,_=completed_fixture(tmp_path,monkeypatch)
    assert train.validate_complete(value) is value
    assert train.model_binding(m.rt.write_new(tmp_path/'V5COMPLETE.json',value))[0]['checkpoint_schema']==train.PILOT_SCHEMA_V5
    for fault in ('minimum','old_steps','lost500','policy','denominator','body','head_reset'):
        bad=deepcopy(value)
        if fault=='minimum':bad['selected_step']=500
        elif fault=='old_steps':bad['completed_steps']=21000
        elif fault=='lost500':bad['evaluation_candidates'].pop(0)
        elif fault=='policy':bad['selection_policy']['validation_used_for_selection']=True
        elif fault=='denominator':bad['optimization']['conditional_terminal_draws']['denominator']-=1
        elif fault=='body':bad['optimization']['conditional_terminal_draws']['body']={'p000923':4000}
        else:bad['initialization']['optimizer']['initial_step']=18000
        with pytest.raises(ValueError):train.validate_complete(bad)


def test_v5_inference_loads_own_state_once_never_old_initializer(tmp_path,monkeypatch):
    m,torch,value,receipt=completed_fixture(tmp_path,monkeypatch)
    software={k:{} for k in ('vendor','dependencies','python','prefix','python_version','source_bindings')}
    value['inputs']['software']=m.rt.write_new(tmp_path/'ownsoftware.json',software)
    packet=dict(value,step=1000,state_dict={'own':'V5'},normalization=receipt['normalization'],optimization=value['selected_optimization'])
    loaded=[];policy=SimpleNamespace(load_state_dict=lambda state,strict:loaded.append((state,strict)),eval=lambda:None)
    monkeypatch.setattr(train,'make_model',lambda *a,**k:(torch,policy,software))
    monkeypatch.setattr(train,'validate_official_forward',lambda *a:None)
    monkeypatch.setattr(m,'initialize_policy',lambda *a,**k:pytest.fail('Inference reinitialized ACT6'))
    monkeypatch.setattr(m,'export',lambda *a,**k:pytest.fail('Inference executed original28'))
    monkeypatch.setattr(m,'require_archive',lambda:pytest.fail('Inference replayed original28 closure'))
    calls=[]
    monkeypatch.setattr(torch,'load',lambda *a,**k:calls.append(a) or packet)
    assert train.load_selected_policy(value)[2] is packet and len(calls)==1
    assert loaded==[({'own':'V5'},True)]


def test_v5_real_masked_loss_poison_and_backward():
    torch=pytest.importorskip('torch');parameter=torch.nn.Parameter(torch.tensor(.5));batches={}
    class Official:
        def __call__(self,batch):
            valid=~batch['action_is_pad'];error=(parameter-batch['action']).abs()
            return (error*valid[...,None]).sum()/(valid.sum()*7),{}
    for i,(name,g) in enumerate(train.GROUPS_V3.items(),1):
        mask=torch.arange(20)[None,:].expand(g['size'],-1)>=i
        action=torch.full((g['size'],20,7),float(i));action[mask]=1234.
        batches[name]=dict(action=action,action_is_pad=mask)
    loss,_=train.mixed_loss(Official(),batches,version=5);expected=float(loss.detach())
    loss.backward();assert torch.isfinite(parameter.grad) and parameter.grad!=0
    for b in batches.values():b['action'][b['action_is_pad']]=-6789.
    again,_=train.mixed_loss(Official(),batches,version=5)
    assert float(again.detach())==expected


def test_v5_terminal_source_body_pose_target_counts_must_reconcile(tmp_path,monkeypatch):
    _,_,value,_=completed_fixture(tmp_path,monkeypatch)
    bad=deepcopy(value)
    bad['optimization']['conditional_terminal_draws']['pose']={'invented':4000}
    with pytest.raises(ValueError,match='terminal|Terminal'):train.validate_complete(bad)


def test_v5_runtime_head_keeps_original_physical_profile_gate(monkeypatch):
    from test_ausc_act_training_v4 import head_spec
    spec=head_spec();calls=[]
    monkeypatch.setattr(train.posture_head,'validate_bound_spec',lambda *a,**k:calls.append(k))
    complete=dict(schema=train.PILOT_SCHEMA_V5,protocol=dict(posture_head=spec))
    train.posture_head.validate_runtime_head(complete,dict(posture_limits=spec['physical_posture_profile']))
    assert calls==[{'version':2}]
    with pytest.raises(ValueError,match='posture profile'):train.posture_head.validate_runtime_head(complete,dict(posture_limits={}))


def test_deployment_inventory_rejects_unknown_schema():
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep,m2_rebuild_paired_eval as paired
    for adapter in (dep,paired):
        with pytest.raises(ValueError,match='schema'):adapter.model_sources({'checkpoint_schema':'m2_act_auxiliary_act_pilot_v99'})
        with pytest.raises(ValueError,match='schema'):adapter.sources(model_schema='m2_act_auxiliary_act_pilot_v99')


def test_v5_config_exact1000_rejects_old_budget_resume_and_fixture(tmp_path,monkeypatch):
    m,torch,init,receipt,_,_=initial_fixture(tmp_path,monkeypatch)
    from test_ausc_act_auxiliary_train import scope_value,source
    from sim.ausc_policy import m2_act_recovery_scope as scope,rebuild_gpu_binding as binding
    frozen=scope_value();scope_ref=m.rt.write_new(tmp_path/'scope.json',frozen)
    native=dict(schema='m2_given_contact_masked_preparation_v1',complete=True,rebuild_epoch=binding.EPOCH,original_cases=0,diagnostic_only=True,formal_acceptance=False,training_eligible=False,test_payloads_opened=False,task_contract=m.rt.task_contract(),counts=train.NATIVE_COUNTS,split_subject_ids=dict(train=frozen['cohorts']['train'][:4],validation=['p000923']))
    native_ref=m.rt.write_new(tmp_path/'nativeheader.json',native)
    sources=[source(kind,subject=subject) for kind in train.conditional_data.KINDS for subject in frozen['optimized_train_ids']]
    metadata=dict(sources=sources,input_profile='production_closed_inputs',optimization_admission=data._admission(True))
    aux=m.rt.write_new(tmp_path/'v5aux.json',{'schema':data.SCHEMA})
    monkeypatch.setattr(train,'NATIVE_DATASET',native_ref);monkeypatch.setattr(train,'TRAIN16_SCOPE',scope_ref)
    monkeypatch.setattr(scope,'validate',lambda ref:m.rt.read_bound(ref)['cohorts'])
    monkeypatch.setattr(data,'metadata',lambda *a,**k:({},metadata))
    monkeypatch.setattr(train.posture_head,'validate_bound_spec',lambda *a,**k:None)
    monkeypatch.setattr(train,'native_normalization',lambda config:receipt['normalization'])
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Config opened arrays'))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Config loaded historical tensors'))
    config=dict(train.SCHEDULE_V5,schema=train.CONFIG_SCHEMA_V5,base='/fixture',dataset=native_ref,auxiliary=aux,evaluation_cohort=scope_ref,posture_head=receipt['posture_head'],initialization=init)
    assert train.validate_config(config) is config
    for patch in (dict(steps=20000),dict(steps=1001),dict(eval_every=100),dict(batch_size=32),dict(resume=True)):
        with pytest.raises(ValueError,match='schedule'):train.validate_config(dict(config,**patch))
    metadata['input_profile']='local_synthetic_fixture'
    with pytest.raises(ValueError,match='production'):train.validate_config(config)


def test_export_isolated_original28_cpu_and_archive_checked_twice(tmp_path,monkeypatch):
    m=warm();calls=[]
    monkeypatch.setattr(m.sys,'platform','linux')
    monkeypatch.setattr(m,'require_archive',lambda:calls.append('archive'))
    monkeypatch.setattr(m.rt,'bound_path',lambda ref:calls.append(ref))
    monkeypatch.setattr(m,'validate_receipt',lambda *a,**k:{})
    def run(argv,**kwargs):
        assert argv[:3]==[m.OFFICIAL_PYTHON,'-I','-B'] and argv[-1]==m.SOURCE28_ROOT
        assert kwargs['cwd']==m.SOURCE28_ROOT
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and kwargs['env']['OMP_NUM_THREADS']=='1'
        assert 'PYTHONPATH' not in kwargs['env']
        import json
        payload=json.loads(kwargs['input'])
        assert payload['owner']==m.OWNER and payload['selection']==m.SELECTED
        assert payload['code_commit']==m.SOURCE28_COMMIT
        return SimpleNamespace(returncode=0,stdout='{"state_dict":{}}',stderr='')
    monkeypatch.setattr(m.subprocess,'run',run)
    m.export(tmp_path/'export')
    assert calls.count('archive')==2


def phase_posture_fixture(tmp_path):
    """Actual reviewed V5 data path, with four appended bound query sources."""
    from test_ausc_act_phase_data_v5 import fixture,export,mutate
    from test_ausc_act_training_v4 import head_spec
    m,f=fixture(tmp_path);spec=head_spec();profile=spec['physical_posture_profile']
    mutate(f,m,f.source['config']['path'],lambda value:value.update(posture_limits=profile))
    frozen=m.rt.read_bound(f.inputs.frozen_request)
    mutate(f,m,frozen['plan_config']['path'],lambda value:value.update(posture_limits=profile))
    admitted=m.load_preparation(export(m,f),f.allowed,inputs=f.inputs)
    sources=admitted['sources'][1:];samples=admitted['samples']
    assert len(sources)==4 and all(source['schema']==m.SOURCE_SCHEMA and 'geometry' not in source['bindings'] for source in sources)
    auxiliary=dict(sources=sources,samples={key:samples[key][2:] for key in ('source_histories','relative_setters','setter_is_pad')})
    native=dict(histories=samples['source_histories'][:1],actions=samples['relative_setters'][:1],action_is_pad=samples['setter_is_pad'][:1])
    prepared=dict(preparation=dict(episodes=[dict(config=sources[0]['bindings']['source'])]),train=deepcopy(native),validation=deepcopy(native))
    return m,dict(schema=train.CONFIG_SCHEMA_V5,posture_head=spec),prepared,auxiliary


def test_v5_posture_audit_uses_all_four_authentic_sources_and_bound_plan_config(tmp_path):
    m,config,prepared,auxiliary=phase_posture_fixture(tmp_path)
    train._audit_posture_inputs(config,prepared,auxiliary)
    assert set(prepared['posture_target_audit'])=={'train','validation','auxiliary'}
    assert prepared['posture_target_audit']['auxiliary']['valid_cells']==76
    assert [source['frozen_physics_row'] for source in auxiliary['sources']]==[5760,5880,14280,14400]


@pytest.mark.parametrize('fault',['plan_posture','source_binding','source_metadata'])
def test_v5_posture_audit_rejects_changed_plan_profile_or_query_binding(tmp_path,fault):
    m,config,prepared,auxiliary=phase_posture_fixture(tmp_path)
    source=auxiliary['sources'][-1]
    frozen=m.rt.read_bound(source['bindings']['request'])
    if fault=='plan_posture':
        plan=m.rt.read_bound(frozen['plan_config']);plan['posture_limits']['joint_upper_rad'][0]-=.1
        frozen['plan_config']=m.rt.write_new(tmp_path/'bad-plan.json',plan)
    elif fault=='source_binding':frozen['source']['config']=m.rt.write_new(tmp_path/'other-native.json',{'posture_limits':config['posture_head']['physical_posture_profile']})
    else:frozen['source_metadata']['target_id']='unbound-target'
    source['bindings']['request']=m.rt.write_new(tmp_path/'bad-request.json',frozen)
    with pytest.raises(ValueError,match='profile|source|metadata|binding'):train._audit_posture_inputs(config,prepared,auxiliary)
