"""Explicit V8 optimizer/selection/dispatch; synthetic fixtures only."""
from copy import deepcopy

import pytest
from sim.ausc_policy import m2_act_auxiliary_train as train


def metric(count,score):
    return dict(sample_count=count,horizon=20,selection_metric='act.full_chunk_mae_rad',
        **{name:dict(full_chunk_mae_rad=score,per_horizon_mae_rad=[score]*20) for name in ('act','hold','constant_velocity')})


def rows():
    return [dict(step=step,train=metric(8373,.2),validation=metric(2345,.1 if step in (1500,17500) else 1.)) for step in range(500,20001,500)]


def test_v8_explicit_long_fit_earliest_full_validation_selection_and_legacy_rules():
    assert hasattr(train,'CONFIG_SCHEMA_V8'), 'Explicit V8 model/optimizer schema required'
    for kind in ('CONFIG','PILOT','SELECTION','CPU','PROTOCOL'):
        assert train._version({'schema':getattr(train,kind+'_SCHEMA_V8')})==8
    assert train._api(8) is train.phase_data and train._groups(8)==train.GROUPS_V3
    assert train.SCHEDULE_V8['steps']==20000 and train.SCHEDULE_V8['eval_every']==500
    assert train.SCHEDULE_V8['batch_size']==64 and train.SCHEDULE_V8['seed']==20261029
    assert train.selected_step(rows(),version=8)==1500
    for bad in (rows()[:-1],rows()+[rows()[-1]],list(reversed(rows()))):
        with pytest.raises(ValueError):train.selected_step(bad,version=8)
    assert train.selected_step([{'step':500},{'step':1000}],version=7)==1000
    with pytest.raises(ValueError):train._groups(10)


def test_v8_candidates_bind_all40_predictions_and_selected_not_final(tmp_path):
    assert hasattr(train,'SELECTION_POLICY_V8'), 'V8 full validation selection policy required'
    candidates=[]
    for row in rows():
        step=row['step'];checkpoint=train.runtime.write_new(tmp_path/('checkpoint-'+str(step)+'.json'),dict(step=step))
        predictions={split:dict(reference=train.runtime.write_new(tmp_path/(split+str(step)+'.json'),dict(step=step)),
            window_order_sha256=('a' if split=='train' else 'b')*64,
            array=dict(shape=[train.NATIVE_COUNTS[split],20,7],dtype='<f8',order='C',sha256='c'*64)) for split in ('train','validation')}
        candidates.append(dict(row,checkpoint=checkpoint,predictions=predictions))
    value=dict(schema=train.SELECTION_SCHEMA_V8,completed_steps=20000,selected_step=1500,selection_policy=deepcopy(train.SELECTION_POLICY_V8),
        window_order=dict(train='a'*64,validation='b'*64),evaluation_candidates=candidates,best=deepcopy(candidates[2]['checkpoint']),selected_predictions=deepcopy(candidates[2]['predictions']),
        selected_metrics={s:deepcopy(candidates[2].get(s,{})) for s in ('train','validation')})
    train._validate_candidates(value)
    for fault in ('missing','wrong_selected','finalonly','swapped','predictions','metric','completed'):
        bad=deepcopy(value)
        if fault=='missing':bad['evaluation_candidates'].pop(0)
        elif fault=='wrong_selected':bad['selected_step']=17500
        elif fault=='finalonly':bad['evaluation_candidates']=bad['evaluation_candidates'][-1:]
        elif fault=='swapped':bad['best']=deepcopy(candidates[-1]['checkpoint'])
        elif fault=='predictions':bad['selected_predictions']=deepcopy(candidates[-1]['predictions'])
        elif fault=='metric':bad['selected_metrics']['validation']['act']['full_chunk_mae_rad']=0.
        else:bad['completed_steps']=1500
        with pytest.raises(ValueError):train._validate_candidates(bad)


def test_v8_config_requires_both_real_profiles_long_schedule_and_train_only_before_arrays(tmp_path,monkeypatch):
    from test_ausc_act_diverse_sampling import fixture
    from test_ausc_act_auxiliary_train import scope_value,source
    from test_ausc_act_training_v4 import head_spec
    from sim.ausc_policy import rebuild_gpu_binding as binding
    from sim.ausc_policy import m2_act_diverse_warm_start as warm
    p,a,banks=fixture(tmp_path,monkeypatch);frozen=scope_value();frozen.update(optimized_train_ids=sorted(train.scope.TRAIN16_IDS),unavailable_train_ids=[]);native=p['preparation'];norm=banks['query32']['metadata']['normalization']
    native.update(schema=train.expanded.MASKED_PREPARATION_SCHEMA,complete=True,rebuild_epoch=binding.EPOCH,original_cases=0,
        diagnostic_only=True,formal_acceptance=False,training_eligible=False,test_payloads_opened=False,task_contract=train.runtime.task_contract(),
        counts=deepcopy(train.NATIVE_COUNTS),split_subject_ids=dict(train=frozen['cohorts']['train'][:4],validation=['p000923']))
    for i,e in enumerate(native['episodes']):e.update(windows=2791,split_sample_range=[i*2791,(i+1)*2791])
    native_ref=train.runtime.write_new(tmp_path/'native-config-v8.json',native)
    for b in banks.values():b['metadata']['base_dataset']=native_ref
    scope_ref=train.runtime.write_new(tmp_path/'scope-config-v8.json',frozen)
    aux=train.runtime.write_new(tmp_path/'aux-config-v8.json',dict(schema=train.phase_data.SCHEMA))
    init=train.runtime.write_new(tmp_path/'init-config-v8.json',dict(schema=warm.SCHEMA,fixture=True))
    sources=[source(kind,subject=body) for kind in train.conditional_data.KINDS for body in frozen['optimized_train_ids']]
    monkeypatch.setattr(train,'NATIVE_DATASET',native_ref);monkeypatch.setattr(train,'TRAIN16_SCOPE',scope_ref)
    monkeypatch.setattr(train.scope,'validate',lambda ref:train.runtime.read_bound(ref)['cohorts'])
    monkeypatch.setattr(train.phase_data,'metadata',lambda *a,**k:({},dict(sources=sources,input_profile='production_closed_inputs',optimization_admission=train.phase_data._admission(True))))
    monkeypatch.setattr(warm,'validate_receipt',lambda *a,**k:dict(fixture=True))
    monkeypatch.setattr(train.posture_head,'validate_bound_spec',lambda *a,**k:None)
    monkeypatch.setattr(train,'native_normalization',lambda *a:p['normalization'])
    import numpy as np
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Config opened raw arrays'))
    config=dict(train.SCHEDULE_V8,schema=train.CONFIG_SCHEMA_V8,base='/fixture',dataset=native_ref,auxiliary=aux,
        evaluation_cohort=scope_ref,initialization=init,posture_head=head_spec(),measured_banks={k:b['reference'] for k,b in banks.items()})
    assert train.validate_config(config) is config
    for patch in (dict(steps=1000),dict(eval_every=1000),dict(seed=1),dict(batch_size=32),dict(q_dropout=.1),dict(resume=True),
        dict(measured_banks={'dart37':banks['dart37']['reference']}),dict(measured_bank=banks['query32']['reference'])):
        with pytest.raises(ValueError):train.validate_config(dict(config,**patch))
    banks['dart37']['metadata']['attempt_ledger'][0]['status']='pending'
    with pytest.raises(ValueError):train.validate_config(config)


def test_v8_public_dispatch_registry_and_original_ipc_profile(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_expanded_pilot_train as expanded
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep,m2_rebuild_paired_eval as paired
    from test_ausc_act_training_v4 import head_spec
    sentinel=object()
    for name in ('validate_config','training_protocol','load_selected_policy','verify_selected','fit','validate_complete','model_binding'):
        monkeypatch.setattr(train,name,lambda *a,**k:sentinel)
    config={'schema':train.CONFIG_SCHEMA_V8}
    assert expanded.validate_config(config) is sentinel and expanded.training_protocol(config) is sentinel
    assert expanded.load_selected_policy({'schema':train.SELECTION_SCHEMA_V8}) is sentinel
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.SELECTION_SCHEMA_V8 if ref=='selection' else train.CONFIG_SCHEMA_V8))
    assert expanded.verify_selected('selection','receipt') is sentinel and expanded.fit('config','output') is sentinel
    assert dep.validate_complete({'schema':train.PILOT_SCHEMA_V8}) is sentinel
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.PILOT_SCHEMA_V8))
    assert dep.model_binding('ref') is sentinel
    monkeypatch.undo()
    from test_ausc_given_contact_pose import reference,metadata
    import numpy as np
    ref=reference(tmp_path);meta=dict(metadata(ref),episode_id='v8',target_id='L1');q=np.zeros(7)
    state=dep.runtime.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    descriptor=dict(metadata=meta,episode_id='v8',source_episode_id='native',model=dict(checkpoint={'sha256':'a'*64},checkpoint_schema=train.PILOT_SCHEMA_V8),source_config={'sha256':'b'*64},protocol={'sha256':'c'*64},scenario_id='s')
    for adapter in (dep,paired):
        request=adapter.make_request(descriptor,1,q,ref.pose_base,1.,metadata=meta,history=np.stack([state,state]),history_times=[0.,.05],context=dict(observed_at_s=.05,physics_row=12,context=['v8','L1',0,'FREE'],source_id='given-contact-'+ref.sha256))
        adapter.validate_request(request,descriptor,sequence=1)
        assert adapter.decode_response(adapter.response_for(request,np.zeros((20,7))),request).metadata['checkpoint_policy']==train.PILOT_SCHEMA_V8
        assert 'm2_act_diverse_sampling.py' in adapter.sources(model_schema=train.PILOT_SCHEMA_V8)['files']
    calls=[];spec=head_spec()
    monkeypatch.setattr(train.posture_head,'validate_bound_spec',lambda *a,**k:calls.append(k))
    train.posture_head.validate_runtime_head(dict(schema=train.PILOT_SCHEMA_V8,protocol=dict(posture_head=spec)),dict(posture_limits=spec['physical_posture_profile']))
    assert calls==[{'version':2}]
    with pytest.raises(ValueError):train.posture_head.validate_runtime_head(dict(schema=train.PILOT_SCHEMA_V8,protocol=dict(posture_head=spec)),dict(posture_limits={}))


def test_v8_padding_poison_has_zero_gradient_and_same_weighted_loss():
    torch=pytest.importorskip('torch');parameter=torch.nn.Parameter(torch.tensor(.5));batches={}
    class Forward:
        def __call__(self,batch):
            valid=~batch['action_is_pad'];error=(parameter-batch['action']).abs()
            return (error*valid[...,None]).sum()/(valid.sum()*7),{}
    for i,(name,g) in enumerate(train.GROUPS_V3.items(),1):
        mask=torch.arange(20)[None,:].expand(g['size'],-1)>=i
        actions=torch.full((g['size'],20,7),float(i),requires_grad=True)
        batches[name]=dict(action=actions,action_is_pad=mask)
    loss,_=train.mixed_loss(Forward(),batches,version=8);expected=float(loss.detach());loss.backward()
    assert parameter.grad is not None and torch.isfinite(parameter.grad) and parameter.grad!=0
    for b in batches.values():
        assert bool(torch.all(b['action'].grad[b['action_is_pad']]==0))
        with torch.no_grad():b['action'][b['action_is_pad']]=123456.
    again,_=train.mixed_loss(Forward(),batches,version=8)
    assert float(again.detach())==expected


def test_v8_official_cpu_factory_head_loss_and_warm_import_when_available(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    if Path(sys.prefix).resolve()!=(Path(train.runtime.VENDOR_ROOT)/'.venv').resolve():
        pytest.skip('Pinned official LeRobot environment is Linux-only; no substitute runtime')
    import torch
    import numpy as np
    from test_ausc_act_training_v4 import head_spec
    from sim.ausc_policy import m2_act_diverse_warm_start as warm
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    spec=head_spec()
    with monkeypatch.context() as isolated:
        isolated.setattr(sys,'path',[p for p in sys.path if '/test-deps' not in p])
        _,policy,software=train.make_model(dict(schema=train.CONFIG_SCHEMA_V8,lerobot_root=train.runtime.VENDOR_ROOT,posture_head=spec),device='cpu',training=True)
    train.validate_official_forward(software)
    assert sum(p.numel() for p in policy.parameters())==2658823 and len(policy.state_dict())==54
    batch={'observation.state':torch.zeros(4,7),'observation.environment_state':torch.zeros(4,27)}
    policy.eval()
    with torch.no_grad():prediction=policy.predict_action_chunk(batch).numpy().astype('float64')
    prediction=prediction*np.asarray(spec['normalization']['action_std'])+spec['normalization']['action_mean']
    path=tmp_path/'synthetic-official.pt';torch.save(policy.state_dict(),path)
    receipt=dict(state_dict=train.runtime.reference(path),tensor_proofs=warm.tensor_proofs(policy.state_dict()),probes=dict(observations={k:v.tolist() for k,v in batch.items()},raw_predictions=prediction.tolist()))
    monkeypatch.setattr(warm,'validate_receipt',lambda *a,**k:receipt)
    assert warm.initialize_policy(torch,policy,{},normalization=spec['normalization'],posture_head=spec)['passed']
    assert not warm.fresh_optimizer(torch,policy,lr=1e-4,weight_decay=1e-4)[0].state
    batches={}
    for name,g in train.GROUPS_V3.items():
        actions=torch.ones(g['size'],20,7,requires_grad=True);mask=torch.arange(20)[None,:].expand(g['size'],-1)>=5
        batches[name]={'observation.state':torch.zeros(g['size'],7),'observation.environment_state':torch.zeros(g['size'],27),'action':actions,'action_is_pad':mask}
    loss,_=train.mixed_loss(policy,batches,version=8);loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and bool(torch.any(p.grad!=0)) for p in policy.parameters())
    for batch in batches.values():assert bool(torch.all(batch['action'].grad[batch['action_is_pad']]==0))


def completed_fixture(tmp_path,monkeypatch):
    """Completed metadata fixture; admission/numerical import tested separately."""
    import numpy as np
    from test_ausc_act_training_v7 import completed_fixture as previous
    from sim.ausc_policy import m2_act_diverse_warm_start as warm,m2_act_diverse_sampling as sampling
    m,torch,value,receipt=previous(tmp_path,monkeypatch)
    monkeypatch.setattr(warm,'validate_receipt',lambda *a,**k:receipt)
    config=m.rt.read_bound(value['inputs']['config']);config.pop('measured_bank')
    config.update(train.SCHEDULE_V8,schema=train.CONFIG_SCHEMA_V8,measured_banks={name:m.rt.write_new(tmp_path/(name+'-synthetic.json'),dict(fixture=True)) for name in sampling.BANKS})
    frozen=deepcopy(value['optimization_scope']);frozen.update(optimized_train_ids=sorted(train.scope.TRAIN16_IDS[:6]),unavailable_train_ids=sorted(train.scope.TRAIN16_IDS[6:]))
    config['evaluation_cohort']=m.rt.write_new(tmp_path/'scope8-completed.json',frozen)
    value['scope']['train']['subject_ids']=frozen['optimized_train_ids'];value['optimization_scope']=frozen
    value.update(schema=train.PILOT_SCHEMA_V8,schedule=config,completed_steps=20000,selected_step=1500,
        protocol=train.training_protocol(config),selection_policy=deepcopy(train.SELECTION_POLICY_V8),data_version=5,model_optimizer_version=8)
    value['inputs'].update(config=m.rt.write_new(tmp_path/'CONFIG8.json',config),code=train.code_binding(version=8),measured_banks=config['measured_banks'])
    eq=m.rt.read_bound(value['initialization']['equivalence']);eq['schema']=warm.EQUIVALENCE_SCHEMA
    value['initialization']['equivalence']=value['inputs']['initialization_equivalence']=m.rt.write_new(tmp_path/'EQ8.json',eq)
    proof=m.rt.read_bound(value['measured_inventory']);proof.update(schema=sampling.SCHEMA,model_optimizer_version=8)
    base=proof['pools']['base_full']['rows'];measured=deepcopy(proof['pools']['measured_full']['rows'])
    for row in measured:row['bank']='query32'
    sources=train.contribution_inventory(config)
    for request in sampling.dart.planned_requests():
        sid='measured:dart37:'+request['slot_id'];i=len(base)+len(measured)
        measured.append(dict(row_index=i,source=sid,bank='dart37',body=request['subject_id'],pose=request['pose_id'],target=request['target_id'],
            slot=request['slot_id'],episode=sid,valid_cells=3,error_m=0.,actual_gap_m=.025))
        sources['native'][sid]=dict(subject_id=request['subject_id'],kind='measured_recovery',pose_id=request['pose_id'],target_id=request['target_id'],slot=request['slot_id'],bank='dart37')
    for name in ('measured_full','measured_corridor'):proof['pools'][name]=sampling._pool(measured,threshold=.012 if name.endswith('corridor') else None)
    proof['measured_windows']=len(measured)
    value['diverse_inventory']=m.rt.write_new(tmp_path/'DIVERSE8.json',proof);value.pop('measured_inventory')
    value['measured_counts']=dict(fixture=True)
    monkeypatch.setattr(train,'_diverse_inventory',lambda *a:proof)
    all_aux=[];group_rows={}
    for group in ('planned','conditional','oracle'):
        group_rows[group]=proof['pools'][group+'_full']['rows'];all_aux.extend(group_rows[group])
    native=base+measured
    cache=dict(version=8,source_ids={name:np.asarray([r['source'] for r in actual],object) for name,actual in (('native',native),('auxiliary',all_aux))},
        subjects={name:np.asarray([r['body'] for r in actual],object) for name,actual in (('native',native),('auxiliary',all_aux))},
        kinds=dict(native=np.asarray(['native']*len(base)+['measured_recovery']*len(measured),object),auxiliary=np.asarray([sources[g][r['source']]['kind'] for g in ('planned','conditional','oracle') for r in group_rows[g]],object)),
        terminal_labels={r['row_index']:{k:r[k] for k in ('source','body','pose','target','anchor')} for r in group_rows['conditional']},
        diverse_labels={name:{r['row_index']:r for r in pool['rows']} for name,pool in proof['pools'].items()})
    optimization=train.new_optimization(version=8);rng=np.random.default_rng(88)
    for _ in range(500):
        indices={g:rng.choice([r['row_index'] for r in group_rows[g]],size=train.GROUPS_V3[g]['size']) for g in group_rows}
        indices['native']=np.r_[rng.choice(len(base),16),rng.choice(np.arange(len(base),len(native)),16)]
        train.record_draws(optimization,cache,indices)
    def scale(value,factor):
        if type(value) is int:return value*factor
        if isinstance(value,dict):return {k:scale(v,factor) for k,v in value.items()}
        return deepcopy(value)
    value.update(optimization=scale(optimization,40),selected_optimization=scale(optimization,3),evaluations=rows())
    candidates=[]
    for row in value['evaluations']:
        step=row['step'];ref=m.rt.write_new(tmp_path/('STATE8-'+str(step)+'.json'),dict(step=step))
        predictions={split:dict(reference=m.rt.write_new(tmp_path/(split+'-PRED8-'+str(step)+'.json'),dict(step=step)),
            window_order_sha256=value['window_order'][split],array=dict(shape=[train.NATIVE_COUNTS[split],20,7],dtype='<f8',order='C',sha256='c'*64)) for split in train.NATIVE_COUNTS}
        candidates.append(dict(row,checkpoint=ref,predictions=predictions))
    selected=candidates[2]
    value.update(evaluation_candidates=candidates,best=selected['checkpoint'],selected_predictions=selected['predictions'],selected_metrics={s:selected[s] for s in train.NATIVE_COUNTS})
    selected_ref=m.rt.write_new(tmp_path/'SELECTED8.json',dict(value,schema=train.SELECTION_SCHEMA_V8))
    cpu=m.rt.read_bound(value['cpu_equivalence']);cpu.update(schema=train.CPU_SCHEMA_V8,selection=selected_ref,checkpoint=value['best'],selected_step=1500,
        scope=value['scope'],optimization_scope=frozen,initialization=value['initialization'],data_version=5,model_optimizer_version=8,diverse_inventory=value['diverse_inventory'],measured_counts=value['measured_counts'])
    value['cpu_equivalence']=m.rt.write_new(tmp_path/'CPU8.json',cpu)
    return warm,torch,value,receipt


def test_v8_complete_own_cpu_and_independent_selected_and_completed_exposures(tmp_path,monkeypatch):
    m,_,value,_=completed_fixture(tmp_path,monkeypatch)
    assert train.validate_complete(value) is value
    assert train.model_binding(m.rt.write_new(tmp_path/'COMPLETE8.json',value))[0]['checkpoint_schema']==train.PILOT_SCHEMA_V8
    assert value['selected_optimization']['gradient_examples']['native']==1500*32
    assert value['optimization']['gradient_examples']['native']==20000*32
    for fault in ('completed','selected_draws','final_draws','pose','source','bank','cpu','cpu_inventory','code','init','count'):
        bad=deepcopy(value)
        if fault=='completed':bad['completed_steps']=1500
        elif fault=='selected_draws':bad['selected_optimization']=deepcopy(bad['optimization'])
        elif fault=='final_draws':bad['optimization']=deepcopy(bad['selected_optimization'])
        elif fault=='pose':bad['optimization']['diverse_draws']['measured_full']['pose']={'fake':160000}
        elif fault=='source':bad['optimization']['gradient_source_examples']['native']={'fake':640000}
        elif fault=='bank':bad['optimization']['diverse_draws']['measured_full']['bank']={'query32':160000}
        elif fault in ('cpu','cpu_inventory'):
            cpu=m.rt.read_bound(bad['cpu_equivalence'])
            if fault=='cpu':cpu['schema']=train.CPU_SCHEMA_V7
            else:cpu['diverse_inventory']=dict(cpu['diverse_inventory'],sha256='f'*64)
            bad['cpu_equivalence']=m.rt.write_new(tmp_path/('CPU8-bad-'+fault+'.json'),cpu)
        elif fault=='code':bad['inputs']['code']['code_commit']='f'*40
        elif fault=='init':bad['initialization']['optimizer']['initial_step']=1000
        else:bad['evaluations'][0]['validation']['sample_count']=32
        with pytest.raises(ValueError):
            if fault.startswith('cpu') or fault=='code':train.model_binding(m.rt.write_new(tmp_path/('bad8-'+fault+'.json'),bad))
            else:train.validate_complete(bad)


def test_v8_selected_loads_only_own_state_rejects_final_or_changed_source(tmp_path,monkeypatch):
    from types import SimpleNamespace
    m,torch,value,receipt=completed_fixture(tmp_path,monkeypatch)
    software={k:{} for k in ('vendor','dependencies','python','prefix','python_version','source_bindings')}
    value['inputs']['software']=m.rt.write_new(tmp_path/'software8.json',software)
    packet=dict(value,step=1500,state_dict={'own':'V8'},normalization=receipt['normalization'],optimization=value['selected_optimization'])
    loaded=[];policy=SimpleNamespace(load_state_dict=lambda state,strict:loaded.append((state,strict)),eval=lambda:None)
    monkeypatch.setattr(train,'make_model',lambda *a,**k:(torch,policy,software));monkeypatch.setattr(train,'validate_official_forward',lambda *a:None)
    for name in ('initialize_policy','export','require_archive'):
        monkeypatch.setattr(m,name,lambda *a,**k:pytest.fail('Inference reexecuted historical model export'))
    monkeypatch.setattr(train.diverse_sampling,'load_banks',lambda *a,**k:pytest.fail('Inference decoded raw training arrays'))
    calls=[];monkeypatch.setattr(torch,'load',lambda *a,**k:calls.append(k) or packet)
    assert train.load_selected_policy(value)[2] is packet and len(calls)==1 and calls[0]['weights_only'] is True
    assert loaded==[({'own':'V8'},True)]
    for patch in (dict(step=20000),dict(schema=train.PILOT_SCHEMA_V7),dict(diverse_inventory={'sha256':'f'*64})):
        saved=deepcopy(packet);packet.update(patch)
        with pytest.raises(ValueError):train.load_selected_policy(value)
        packet.clear();packet.update(saved)


def test_v8_original_fit_runs20000_new_synthetic_updates_and_saves_all40(tmp_path,monkeypatch):
    """Actual CPU toy optimization through the original loop; no GPU/real data.

    The vendor/owner/inference boundaries are synthetic. Packing, hierarchy,
    losses, backward, warm tensor import, AdamW, draw accounting, checkpoint
    writing, selection and the own-CPU dispatch remain exercised.
    """
    from contextlib import nullcontext
    import numpy as np
    from test_ausc_act_diverse_sampling import fixture as sampling_fixture
    from test_ausc_act_diverse_warm_start import fixture as warm_fixture
    from test_ausc_act_auxiliary_train import scope_value
    p,a,banks=sampling_fixture(tmp_path,monkeypatch)
    wdir=tmp_path/'toy-warm';wdir.mkdir();warm,torch,policy,init,receipt=warm_fixture(wdir,monkeypatch)
    def forward(batch):
        valid=~batch['action_is_pad'];error=(policy.values[0]-batch['action']).abs()
        return (error*valid[...,None]).sum()/(valid.sum()*7),{}
    policy.forward=forward
    receipt['normalization']=p['normalization'];receipt['posture_head']['normalization']=p['normalization']
    receipt['probes']['raw_predictions']=np.broadcast_to(np.asarray(p['normalization']['action_mean']),(4,20,7)).tolist()
    config=dict(train.SCHEDULE_V8,schema=train.CONFIG_SCHEMA_V8,base='/synthetic',dataset=train.NATIVE_DATASET,
        auxiliary={},evaluation_cohort={},initialization=init,posture_head=receipt['posture_head'],measured_banks={name:b['reference'] for name,b in banks.items()})
    frozen=scope_value();frozen.update(optimized_train_ids=sorted(train.scope.TRAIN16_IDS[:6]),unavailable_train_ids=sorted(train.scope.TRAIN16_IDS[6:]))
    p.update(auxiliary=a,optimization_scope=frozen,native_scope=deepcopy(p['scope']),window_order=dict(train='a'*64,validation='b'*64),oracle_authentication=[],posture_target_audit={})
    cfg=train.runtime.write_new(tmp_path/'toy-config.json',config)
    monkeypatch.setattr(train,'validate_config',lambda c:c);monkeypatch.setattr(train,'load_inputs',lambda c:p)
    monkeypatch.setattr(train.runtime,'owned_gpu',lambda:nullcontext({'synthetic_cpu_only':True}));monkeypatch.setattr(train.runtime,'_identity',lambda:dict(pid=1,pgid=1,sid=1))
    monkeypatch.setattr(train,'make_model',lambda *a,**k:(torch,policy,{}));monkeypatch.setattr(train,'sum',lambda *a:2658823,raising=False)
    monkeypatch.setattr(train,'validate_official_forward',lambda *a:None);monkeypatch.setattr(train.runtime,'execution_settings',lambda *a,**k:{})
    monkeypatch.setattr(train.posture_head,'diagnostics',lambda *a:{})
    build=train.build_cache
    monkeypatch.setattr(train,'build_cache',lambda prepared,auxiliary,**k:build(prepared,auxiliary,torch=torch,device='cpu',optimizer_version=8))
    # Full population metric shape is authenticated independently in other tests;
    # this toy inference emits only three rows and marks that boundary explicitly.
    calls={'metrics':0,'cpu':0}
    monkeypatch.setattr(train.runtime,'infer',lambda *a,**k:np.zeros((3,20,7)))
    def metrics(pred,samples):
        calls['metrics']+=1
        eval_index=(calls['metrics']-1)//2+1;split='train' if calls['metrics']%2 else 'validation'
        return metric(train.NATIVE_COUNTS[split],.1 if eval_index in (3,35,41) else 1.)
    monkeypatch.setattr(train.runtime,'metrics',metrics)
    def cpu_run(command,**kwargs):
        calls['cpu']+=1
        selected=train.runtime.read_bound(train.runtime.reference(command[command.index('--verify-selected')+1]))
        assert selected['schema']==train.SELECTION_SCHEMA_V8 and selected['selected_step']==1500
        train.runtime.write_new(command[command.index('--receipt')+1],dict(passed=True,fixture_only=True))
    monkeypatch.setattr(train.subprocess,'run',cpu_run)
    # code_binding calls git subprocess.check_output internally; bind a visible
    # fixture inventory before replacing subprocess.run for own-CPU dispatch.
    monkeypatch.setattr(train,'code_binding',lambda **k:dict(synthetic_fixture=True))
    result=train.fit(cfg,tmp_path/'toy-output')
    assert result['completed_steps']==20000 and result['selected_step']==1500 and calls['cpu']==1
    assert len(result['evaluations'])==len(result['evaluation_candidates'])==40 and calls['metrics']==82
    assert result['optimization']['gradient_examples']['native']==20000*32
    assert result['selected_optimization']['gradient_examples']['native']==1500*32
    selected=result['evaluation_candidates'][2]
    assert result['best']['sha256']==selected['checkpoint']['sha256']
    assert result['selected_predictions']==selected['predictions']
    for row in result['evaluation_candidates']:
        packet=torch.load(train.runtime.bound_path(row['checkpoint']),weights_only=True)
        assert packet['schema']==train.PILOT_SCHEMA_V8 and packet['step']==row['step']
        for binding in row['predictions'].values():assert train.runtime.bound_path(binding['reference']).exists()
    last=torch.load(train.runtime.bound_path(result['last']),weights_only=True)
    assert last['step']==20000 and last['optimizer_state_dict']['state']


@pytest.mark.parametrize('adapter_name',['expanded','paired'])
def test_v8_native_audit_retains_exact_model_specific_source_inventory(adapter_name):
    from pathlib import Path
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as expanded,m2_rebuild_paired_eval as paired
    from sim.ausc_policy import m2_given_contact_pilot_native_audit as audit
    adapter={'expanded':expanded,'paired':paired}[adapter_name]
    model={'checkpoint_schema':train.PILOT_SCHEMA_V8}
    expected=adapter.model_sources(model)
    assert {'m2_act_diverse_sampling.py','m2_act_diverse_warm_start.py','m2_measured_dart_data.py'}<=set(expected['files'])
    assert all(Path(ref['path']).parent==Path(train.__file__).parent for ref in expected['files'].values())
    assert audit.model_sources(adapter,model)==expected
