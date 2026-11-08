"""V9 integration contracts. All local arrays and checkpoints are synthetic."""
from copy import deepcopy
import sys
from pathlib import Path

import numpy as np
import pytest

from sim.ausc_policy import m2_act_auxiliary_train as train


def test_v9_explicit_versions_schedule_and_task_selection_preserve_legacy():
    assert hasattr(train,'CONFIG_SCHEMA_V9'), 'Explicit V9 training/model/protocol/CPU schemas required'
    for kind in ('CONFIG','PILOT','SELECTION','CPU','PROTOCOL'):
        assert train._version({'schema':getattr(train,kind+'_SCHEMA_V9')})==9
    assert train._api(9) is train.phase_data and train._groups(9)==train.GROUPS_V3
    assert train._warm_start(9).__name__.endswith('m2_act_refinement_warm_start')
    assert train.SCHEDULE_V9==train.SCHEDULE_V8
    assert train.SCHEDULE_V9['steps']==20000 and train.SCHEDULE_V9['eval_every']==500
    from test_ausc_act_training_v8 import rows
    candidates=rows()
    for row in candidates:row['score']=dict(task_score=.1 if row['step'] in (2500,3000) else 1.)
    assert train.selected_step(candidates,version=9)==2500
    assert train.selected_step(candidates,version=8)==1500
    assert train.selected_step([dict(step=500),dict(step=1000)],version=7)==1000
    for bad in (candidates[:-1],list(reversed(candidates)),candidates+[candidates[-1]]):
        with pytest.raises(ValueError):train.selected_step(bad,version=9)
    with pytest.raises(ValueError):train._version(dict(schema='m2_act10_collection_only_bootstrap_v1'))
    with pytest.raises(ValueError):train._groups(10)


def test_v9_saved_canonical_arrays_are_scored_after_storage(tmp_path,monkeypatch):
    from test_ausc_act_task_selection import fixture
    m,ref,samples,_=fixture(tmp_path,monkeypatch)
    subset=m.freeze_subset(ref,tmp_path/'subset')
    prepared=dict(validation=samples,window_order=dict(validation=m.rt.read_bound(subset)['original_window_order_sha256']),task_subset=subset)
    prediction=np.full(samples['actions'].shape,.000123456789,dtype=np.float32)
    # runtime's canonical representation is the sole score input, never the
    # pre-save float64 tensor. This small fixture does not admit a production run.
    assert hasattr(train,'save_task_candidate'), 'V9 storage-before-score integration missing'
    binding,score=train.save_task_candidate(tmp_path/'saved.npy',prediction,prepared)
    stored=train.runtime.load_predictions(binding,window_order_sha256=prepared['window_order']['validation'])
    assert score==m.score(stored,samples,subset)
    assert score['prediction_proof']==m.measured.array_proof(stored)
    assert np.array_equal(stored,train.runtime.canonical_predictions(prediction))


def test_v9_config_pins_three_banks_initialization_and_preoptimizer_subset_before_arrays(tmp_path,monkeypatch):
    from test_ausc_act_refinement_sampling import fixture
    from test_ausc_act_auxiliary_train import scope_value,source
    from test_ausc_act_training_v4 import head_spec
    from sim.ausc_policy import rebuild_gpu_binding as binding
    p,a,banks=fixture(tmp_path,monkeypatch);frozen=scope_value()
    frozen.update(optimized_train_ids=sorted(train.scope.TRAIN16_IDS),unavailable_train_ids=[])
    native=p['preparation'];native.update(schema=train.expanded.MASKED_PREPARATION_SCHEMA,complete=True,rebuild_epoch=binding.EPOCH,original_cases=0,
        diagnostic_only=True,formal_acceptance=False,training_eligible=False,test_payloads_opened=False,task_contract=train.runtime.task_contract(),
        counts=deepcopy(train.NATIVE_COUNTS),split_subject_ids=dict(train=frozen['cohorts']['train'][:4],validation=['p000923']))
    for i,e in enumerate(native['episodes']):e.update(windows=2791,split_sample_range=[i*2791,(i+1)*2791])
    native['arrays']['validation']=native['arrays']['train']
    native_ref=train.runtime.write_new(tmp_path/'native-config-v9.json',native)
    for name,b in banks.items():
        b['metadata']['base_dataset']=native_ref
        b['complete']['metadata']=train.runtime.write_new(tmp_path/(name+'-config-metadata.json'),b['metadata'])
        b['reference']=train.runtime.write_new(tmp_path/(name+'-config-complete.json'),b['complete'])
    scope_ref=train.runtime.write_new(tmp_path/'scope-config-v9.json',frozen)
    aux=train.runtime.write_new(tmp_path/'aux-config-v9.json',dict(schema=train.phase_data.SCHEMA))
    init=train.runtime.write_new(tmp_path/'init-config-v9.json',dict(schema=train._warm_start(9).SCHEMA,synthetic=True))
    cache=train.runtime.write_new(tmp_path/'teacher-cache-synthetic.json',dict(synthetic=True))
    # Metadata-only config fixture; full observed population/cache validation is
    # exercised by the independent numerical selection fixture before fit.
    subset=dict(schema=train.task_selection.SUBSET_SCHEMA,dataset=native_ref,index=native['index'],normalization=native['normalization'],
        validation_arrays=native['arrays']['validation'],metric=deepcopy(train.task_selection.METRIC),selection_rule=train.task_selection.RULE,
        selector_source=train.runtime.reference(train.task_selection.__file__),geometry_sources=train.task_selection._geometry_sources(),teacher_cache=cache)
    subset_ref=train.runtime.write_new(tmp_path/'subset-config-v9.json',subset)
    sources=[source(kind,subject=body) for kind in train.conditional_data.KINDS for body in frozen['optimized_train_ids']]
    monkeypatch.setattr(train,'NATIVE_DATASET',native_ref);monkeypatch.setattr(train.refinement_sampling.query,'BASE29',native_ref)
    monkeypatch.setattr(train,'TRAIN16_SCOPE',scope_ref);monkeypatch.setattr(train.scope,'validate',lambda ref:train.runtime.read_bound(ref)['cohorts'])
    monkeypatch.setattr(train.phase_data,'metadata',lambda *a,**k:({},dict(sources=sources,input_profile='production_closed_inputs',optimization_admission=train.phase_data._admission(True))))
    calls=[];monkeypatch.setattr(train._warm_start(9),'validate_receipt',lambda *a,**k:calls.append((a,k)) or dict(synthetic=True))
    monkeypatch.setattr(train._warm_start(9),'protocol',lambda ref:dict(receipt=ref,original_evaluations='initialization provenance only'))
    monkeypatch.setattr(train.posture_head,'validate_bound_spec',lambda *a,**k:None)
    monkeypatch.setattr(train,'native_normalization',lambda *a:p['normalization'])
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Config opened raw arrays'))
    config=dict(train.SCHEDULE_V9,schema=train.CONFIG_SCHEMA_V9,base='/fixture',dataset=native_ref,auxiliary=aux,
        evaluation_cohort=scope_ref,initialization=init,posture_head=head_spec(),measured_banks={k:b['reference'] for k,b in banks.items()},task_subset=subset_ref)
    assert train.validate_config(config) is config and calls
    protocol=train.training_protocol(config)
    assert train.runtime.read_bound(train.runtime.write_new(tmp_path/'protocol-v9.json',protocol))==protocol
    assert protocol['model_optimizer_version']==9 and protocol['task_subset']==subset_ref
    assert protocol['teacher_cache']==cache and protocol['selection_policy']==train.SELECTION_POLICY_V9
    assert protocol['attention_heads']==4 and protocol['execution_prefix']==10 and protocol['history_frames']==2
    assert protocol['optimizer']==dict(name='AdamW',initial_step=0,lr=1e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8)
    for patch in (dict(steps=1000),dict(eval_every=1000),dict(seed=1),dict(batch_size=32),dict(q_dropout=.1),dict(resume=True),
        dict(measured_banks={k:b['reference'] for k,b in banks.items() if k!='onpolicy8'}),dict(task_subset=subset),dict(task_subset=dict(subset_ref,sha256='0'*64)),dict(schema=train.CONFIG_SCHEMA_V8)):
        with pytest.raises(ValueError):train.validate_config(dict(config,**patch))
    monkeypatch.setattr(train._warm_start(9),'validate_receipt',lambda *a,**k:(_ for _ in ()).throw(ValueError('original41 authentication revoked')))
    with pytest.raises(ValueError,match='original41'):train.validate_config(config)


def test_v9_public_dispatch_sources_head_and_bootstrap_rejection(tmp_path,monkeypatch):
    assert hasattr(train,'PILOT_SCHEMA_V9'), 'Explicit V9 normal-model admission missing'
    from sim.ausc_policy import m2_given_contact_expanded_pilot_train as expanded
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep,m2_rebuild_paired_eval as paired
    from sim.ausc_policy import m2_given_contact_pilot_native_audit as native
    from test_ausc_act_training_v4 import head_spec
    sentinel=object()
    with monkeypatch.context() as patch:
        for name in ('validate_config','training_protocol','load_selected_policy','verify_selected','fit','validate_complete','model_binding'):
            patch.setattr(train,name,lambda *a,**k:sentinel)
        config=dict(schema=train.CONFIG_SCHEMA_V9)
        assert expanded.validate_config(config) is sentinel and expanded.training_protocol(config) is sentinel
        assert expanded.load_selected_policy(dict(schema=train.SELECTION_SCHEMA_V9)) is sentinel
        patch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.SELECTION_SCHEMA_V9 if ref=='selection' else train.CONFIG_SCHEMA_V9))
        assert expanded.verify_selected('selection','receipt') is sentinel and expanded.fit('config','output') is sentinel
        assert dep.validate_complete(dict(schema=train.PILOT_SCHEMA_V9)) is sentinel
        patch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.PILOT_SCHEMA_V9))
        assert dep.model_binding('ref') is sentinel
    required={'m2_act_refinement_sampling.py','m2_act_refinement_data.py','m2_act_refinement_warm_start.py','m2_act_task_selection.py','m2_act_onpolicy_collection.py'}
    for adapter in (dep,paired):
        source=adapter.model_sources(dict(checkpoint_schema=train.PILOT_SCHEMA_V9))
        assert required<=set(source['files'])
        assert native.model_sources(adapter,dict(checkpoint_schema=train.PILOT_SCHEMA_V9))==source
        with pytest.raises(ValueError):adapter.model_sources(dict(checkpoint_schema=train._warm_start(9).COLLECTION_SCHEMA))
        with pytest.raises(ValueError):adapter.sources(model_schema=train._warm_start(9).COLLECTION_SCHEMA)
    code=train.code_binding(version=9);train.validate_code(code,version=9)
    for fault in ('bytes','origin','missing'):
        bad=deepcopy(code);name='m2_act_task_selection.py'
        if fault=='bytes':bad['files'][name]['sha256']='0'*64
        elif fault=='origin':bad['files'][name]['path']='/synthetic/old-selector.py'
        else:bad['files'].pop(name)
        with pytest.raises(ValueError):train.validate_code(bad,version=9)
    with pytest.raises(ValueError):dep.validate_complete(dict(schema=train._warm_start(9).COLLECTION_SCHEMA,complete=True))
    with pytest.raises(ValueError):train.load_selected_policy(dict(schema=train._warm_start(9).COLLECTION_SCHEMA))
    from test_ausc_given_contact_pose import reference,metadata
    ref=reference(tmp_path);meta=dict(metadata(ref),episode_id='v9',target_id='L1');q=np.zeros(7)
    state=dep.runtime.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    descriptor=dict(metadata=meta,episode_id='v9',source_episode_id='native',model=dict(checkpoint={'sha256':'a'*64},checkpoint_schema=train.PILOT_SCHEMA_V9),source_config={'sha256':'b'*64},protocol={'sha256':'c'*64},scenario_id='s')
    for adapter in (dep,paired):
        request=adapter.make_request(descriptor,1,q,ref.pose_base,1.,metadata=meta,history=np.stack([state,state]),history_times=[0.,.05],context=dict(observed_at_s=.05,physics_row=12,context=['v9','L1',0,'FREE'],source_id='given-contact-'+ref.sha256))
        adapter.validate_request(request,descriptor,sequence=1)
        assert adapter.decode_response(adapter.response_for(request,np.zeros((20,7))),request).metadata['checkpoint_policy']==train.PILOT_SCHEMA_V9
        bad=deepcopy(descriptor);bad['model']['checkpoint_schema']=train._warm_start(9).COLLECTION_SCHEMA
        with pytest.raises(ValueError):adapter.make_request(bad,1,q,ref.pose_base,1.,metadata=meta)
    calls=[];spec=head_spec()
    monkeypatch.setattr(train.posture_head,'validate_bound_spec',lambda *a,**k:calls.append(k))
    train.posture_head.validate_runtime_head(dict(schema=train.PILOT_SCHEMA_V9,protocol=dict(posture_head=spec)),dict(posture_limits=spec['physical_posture_profile']))
    assert calls==[dict(version=2)]


def test_v9_sampler_draws_and_masked_loss_use_unchanged_groups(tmp_path,monkeypatch):
    assert hasattr(train,'CONFIG_SCHEMA_V9'), 'V9 sampler integration missing'
    from test_ausc_act_refinement_sampling import fixture
    p,a,banks=fixture(tmp_path,monkeypatch)
    cache=train.build_cache(p,a,optimizer_version=9)
    assert cache['version']==9 and cache['refinement_inventory']['model_optimizer_version']==9
    rng=np.random.default_rng(9);counts=train.new_optimization(version=9)
    for i in range(5):
        train.record_draws(counts,cache,train.sample_cached_indices(rng,cache))
        if i==2:selected_counts=deepcopy(counts)
    train.refinement_sampling.validate_draws(counts,steps=5,inventory=cache['refinement_inventory'])
    assert counts['gradient_examples']==dict(native=160,planned=40,conditional=40,oracle=80)
    assert 'diverse_draws' not in counts
    # Five actual synthetic draw steps and the earlier three-step snapshot
    # remain distinct; neither is labeled a completed production optimization.
    train.refinement_sampling.validate_draws(selected_counts,steps=3,inventory=cache['refinement_inventory'])
    with pytest.raises(ValueError):train.refinement_sampling.validate_draws(selected_counts,steps=5,inventory=cache['refinement_inventory'])
    from test_ausc_act_auxiliary_train import scope_value
    frozen=scope_value();frozen.update(optimized_train_ids=counts['actual_optimized_train_ids'],unavailable_train_ids=sorted(set(train.scope.TRAIN16_IDS)-set(counts['actual_optimized_train_ids'])))
    inventory={g:{} for g in train.GROUPS_V3}
    for episode in p['preparation']['episodes']:
        inventory['native']['native:'+episode['episode_id']]=dict(subject_id=episode['subject_id'],kind='native')
    for name,bank in banks.items():
        for row in train.refinement_sampling.source_records(bank['metadata'],name):inventory['native'][row['source']]=dict(subject_id=row['body'],kind='measured_recovery')
    mapping=dict(planned_ideal_tracking='planned',conditional_ideal_continuation='conditional',student_oracle_unexecuted='oracle')
    for s in a['sources']:inventory[mapping[s['kind']]][s['source_id']]=dict(s,subject_id=s['subject_id'])
    native_ids=[e['subject_id'] for e in p['preparation']['episodes']]
    train._validate_draws(counts,5,frozen,native_ids,version=9,inventory=inventory,refinement_inventory=cache['refinement_inventory'])
    bad=deepcopy(counts);bad['refinement_draws']['measured_full']['valid_cells']+=1
    with pytest.raises(ValueError):train._validate_draws(bad,5,frozen,native_ids,version=9,inventory=inventory,refinement_inventory=cache['refinement_inventory'])
    torch=pytest.importorskip('torch');parameter=torch.nn.Parameter(torch.tensor(.5));batches={}
    class Forward:
        def __call__(self,batch):
            valid=~batch['action_is_pad'];error=(parameter-batch['action']).abs()
            return (error*valid[...,None]).sum()/(valid.sum()*7),{}
    for i,(name,g) in enumerate(train.GROUPS_V3.items(),1):
        batches[name]=dict(action=torch.full((g['size'],20,7),float(i),requires_grad=True),action_is_pad=torch.arange(20)[None,:].expand(g['size'],-1)>=i)
    loss,_=train.mixed_loss(Forward(),batches,version=9);expected=float(loss.detach());loss.backward()
    assert torch.isfinite(parameter.grad) and parameter.grad!=0
    for b in batches.values():
        assert bool(torch.all(b['action'].grad[b['action_is_pad']]==0))
        with torch.no_grad():b['action'][b['action_is_pad']]=123456.
    assert float(train.mixed_loss(Forward(),batches,version=9)[0].detach())==expected


def test_v9_training_inventory_boundary_preserves_all_bank_and_subset_bindings(tmp_path,monkeypatch):
    from test_ausc_act_refinement_sampling import fixture
    p,a,banks=fixture(tmp_path,monkeypatch);cache=train.build_cache(p,a,optimizer_version=9);inventory=cache['refinement_inventory']
    config=dict(schema=train.CONFIG_SCHEMA_V9,dataset=train.refinement_sampling.query.BASE29,
        measured_banks={n:b['reference'] for n,b in banks.items()},task_subset=train.runtime.write_new(tmp_path/'subset-binding.json',dict(synthetic=True)),
        auxiliary=train.runtime.write_new(tmp_path/'aux-binding.json',dict(synthetic=True)),evaluation_cohort=train.runtime.write_new(tmp_path/'scope-binding.json',dict(synthetic=True)))
    read=train.runtime.read_bound
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:p['preparation'] if ref==config['dataset'] else read(ref))
    monkeypatch.setattr(train.phase_data,'metadata',lambda *args,**k:(dict(arrays=a['array_reference']),dict(sources=a['sources'])))
    value=dict(data_version=5,model_optimizer_version=9,inputs=dict(measured_banks=config['measured_banks'],task_subset=config['task_subset']),
        task_subset=config['task_subset'],refinement_inventory=train.runtime.write_new(tmp_path/'refinement-binding.json',inventory),measured_counts=train.measured_counts(inventory))
    assert train._refinement_inventory(value,config)==inventory
    for fault in ('old_version','subset','bank','count','pool'):
        bad=deepcopy(value)
        if fault=='old_version':bad['model_optimizer_version']=8
        elif fault=='subset':bad['task_subset']=dict(bad['task_subset'],sha256='0'*64)
        elif fault=='bank':bad['inputs']['measured_banks'].pop('onpolicy8')
        elif fault=='count':bad['measured_counts']['train_windows']+=1
        else:
            corrupt=deepcopy(inventory);corrupt['pools']['measured_full']['rows'][0]['valid_cells']+=1
            bad['refinement_inventory']=train.runtime.write_new(tmp_path/'refinement-pool-corruption.json',corrupt)
        with pytest.raises(ValueError):train._refinement_inventory(bad,config)


def synthetic_selection(tmp_path,monkeypatch):
    """Small invented populations; production counts are patched only here."""
    torch=pytest.importorskip('torch')
    from test_ausc_act_task_selection import fixture,analytic
    m,ref,samples,header=fixture(tmp_path,monkeypatch);analytic(m,monkeypatch)
    samples['action_is_pad'][0]=False
    index=m.rt.read_bound(header['index']);index['samples'][0]['window']['valid_future_cells']=20
    header['index']=m.rt.write_new(tmp_path/'full-horizon-index.json',index)
    path=tmp_path/'full-horizon-arrays.npz'
    with path.open('xb') as stream:np.savez(stream,**{k:samples[k] for k in m.ARRAY_KEYS})
    header['arrays']['validation']=m.rt.reference(path)
    ref=m.rt.write_new(tmp_path/'full-horizon-header.json',header);monkeypatch.setattr(m,'BASE29',ref)
    subset=m.freeze_subset(ref,tmp_path/'subset');order=m.rt.read_bound(subset)['original_window_order_sha256']
    monkeypatch.setattr(m.expanded,'load_dataset',lambda r:dict(train=deepcopy(samples),validation=deepcopy(samples),preparation=header,window_order=dict(train=order,validation=order)))
    monkeypatch.setattr(train,'NATIVE_COUNTS',dict(train=len(samples['states']),validation=len(samples['states'])))
    warm=train._warm_start(9)
    # Invented tensor names/shapes exercise packet consistency only, never an
    # Official model factory substitute or a parameter-count qualification.
    state={str(i):torch.zeros(1) for i in range(54)}
    proofs=warm.tensor_proofs(state)
    monkeypatch.setattr(warm,'validate_receipt',lambda *a,**k:dict(tensor_proofs=proofs))
    init=m.rt.write_new(tmp_path/'initialization.json',dict(synthetic=True))
    config=dict(schema=train.CONFIG_SCHEMA_V9,dataset=ref,task_subset=subset,initialization=init)
    inputs=dict(config=m.rt.write_new(tmp_path/'config.json',config),dataset=ref,task_subset=subset,initialization=init)
    value=dict(schema=train.SELECTION_SCHEMA_V9,completed_steps=20000,selection_policy=deepcopy(train.SELECTION_POLICY_V9),
        inputs=inputs,window_order=dict(train=order,validation=order),task_subset=subset,protocol=dict(synthetic=True),
        data_version=5,model_optimizer_version=9,refinement_inventory=init,measured_counts=dict(synthetic=True),
        initialization=dict(receipt=init),terminal_pool_inventory=dict(synthetic=True))
    candidates=[]
    for step in m.CANDIDATE_STEPS:
        prediction=np.zeros_like(samples['actions'],dtype=np.float32)
        prediction[:,:,6 if step in (2500,3000) else 0]=.001 if step!=1500 else .0000005
        bindings={s:m.rt.save_predictions(tmp_path/(str(step)+'-'+s+'.npy'),prediction,window_order_sha256=order) for s in train.NATIVE_COUNTS}
        saved=m.rt.load_predictions(bindings['validation'],window_order_sha256=order)
        metric=m.rt.metrics(saved,samples);row=dict(step=step,train=metric,validation=deepcopy(metric),score=m.score(saved,samples,subset))
        packet={k:deepcopy(value[k]) for k in ('inputs','protocol','task_subset','refinement_inventory','measured_counts','initialization','terminal_pool_inventory','data_version','model_optimizer_version','selection_policy')}
        packet.update(schema=train.PILOT_SCHEMA_V9,step=step,metrics=deepcopy(row),state_dict=state,
            task_contract=train.runtime.task_contract(),model=train.runtime.shared.native_spec(),**train.FLAGS)
        path=tmp_path/('state-'+str(step)+'.pt');torch.save(packet,path)
        candidates.append(dict(row,checkpoint=m.rt.reference(path),predictions=bindings,state_proofs=proofs))
    value.update(evaluation_candidates=candidates,selected_step=2500,best=candidates[4]['checkpoint'],selected_predictions=candidates[4]['predictions'],
        selected_metrics={s:deepcopy(candidates[4][s]) for s in train.NATIVE_COUNTS},evaluations=[{k:deepcopy(r[k]) for k in ('step','train','validation','score')} for r in candidates])
    value['task_selection']=dict(schema=m.SELECTION_SCHEMA,dataset=ref,subset=subset,selected_step=2500,
        candidates=[dict(step=r['step'],predictions=r['predictions']['validation'],score=r['score']) for r in candidates])
    return m,samples,value


def test_v9_all40_saved_task_ranking_states_predictions_and_exact_corruption(tmp_path,monkeypatch):
    assert hasattr(train,'SELECTION_POLICY_V9'), 'V9 exact retained candidate integration missing'
    m,samples,value=synthetic_selection(tmp_path,monkeypatch)
    train._validate_candidates(value)
    assert train.selected_step(value['evaluations'],version=9)==2500
    assert train.expanded.selected_step(value['evaluations'])==1500
    proof=train.verify_task_candidates(value)
    assert proof==dict(selected_step=2500,task_score=0.,verified_candidates=40)
    for fault in ('missing','selected','state','prediction','score','subset','source','metric'):
        bad=deepcopy(value)
        if fault=='missing':bad['evaluation_candidates'].pop()
        elif fault=='selected':bad['selected_step']=3000
        elif fault=='state':bad['evaluation_candidates'][0]['checkpoint']=bad['evaluation_candidates'][1]['checkpoint']
        elif fault=='prediction':bad['evaluation_candidates'][0]['predictions']['validation']=bad['evaluation_candidates'][2]['predictions']['validation']
        elif fault=='score':bad['evaluation_candidates'][0]['score']['task_score']+=1e-15
        elif fault=='subset':bad['task_subset']=dict(bad['task_subset'],sha256='0'*64)
        elif fault=='source':bad['inputs']['dataset']=dict(bad['inputs']['dataset'],sha256='0'*64)
        else:bad['evaluation_candidates'][0]['validation']['act']['full_chunk_mae_rad']=0.
        with pytest.raises(ValueError):train._validate_candidates(bad)


def test_v9_cpu_receipt_keeps_exact_saved_ranking_and_checks_own_full_elements(tmp_path,monkeypatch):
    """Orchestration only; no substitute model/factory or runtime claim."""
    m,samples,value=synthetic_selection(tmp_path,monkeypatch)
    proof=train._validate_candidates(value)
    value.update(scope=dict(synthetic=True),native_scope=dict(synthetic=True),optimization_scope=dict(synthetic=True),**train.FLAGS)
    selection_ref=m.rt.write_new(tmp_path/'selection-for-cpu.json',value)
    prepared={k:value[k] for k in ('scope','native_scope','optimization_scope','window_order')}
    prepared.update(train=samples,validation=samples)
    monkeypatch.setattr(train,'load_inputs',lambda c:prepared)
    packet=dict(normalization=dict(synthetic=True),_task_candidate_verification=proof)
    monkeypatch.setattr(train,'load_selected_policy',lambda selection:(None,None,packet))
    monkeypatch.setattr(train.runtime,'execution_settings',lambda *a,**k:dict(synthetic=True,device='cpu'))
    saved=m.rt.load_predictions(value['selected_predictions']['validation'],window_order_sha256=value['window_order']['validation'])
    own=saved.copy();own[:,:,0]+=1e-8;calls=[]
    def infer(*a,**kwargs):
        assert kwargs['device']=='cpu';calls.append(len(a[2]['states']));return own.copy()
    monkeypatch.setattr(train.runtime,'infer',infer)
    monkeypatch.setattr(train.task_selection,'select_candidate',lambda *a:pytest.fail('Fresh CPU rounding attempted to select again'))
    receipt=train.verify_selected(selection_ref,tmp_path/'cpu.json');cpu=m.rt.read_bound(receipt)
    assert calls==[len(samples['states'])]*2 and cpu['schema']==train.CPU_SCHEMA_V9
    assert cpu['task_candidate_verification']==proof and cpu['selected_step']==2500
    assert cpu['task_selection']==value['task_selection'] and cpu['task_subset']==value['task_subset']
    assert all(c['passed'] and c['prediction_rtol']==1e-5 and c['prediction_atol']==1e-6 and
        c['metric_rtol']==1e-5 and c['metric_atol']==1e-7 for c in cpu['comparisons'].values())
    own[-1,-1,-1]+=.1
    with pytest.raises(ValueError,match='elementwise'):train.verify_selected(selection_ref,tmp_path/'bad-cpu.json')


def test_v9_normal_admission_aggregates_bound_own_cpu_and_exact_task_replay(tmp_path,monkeypatch):
    """Admission aggregation only; numerical candidates were separately verified."""
    m,samples,value=synthetic_selection(tmp_path,monkeypatch);proof=train._validate_candidates(value)
    value.update(scope=dict(synthetic=True),optimization_scope=dict(synthetic=True),**train.FLAGS)
    value['inputs']['code']=train.code_binding(version=9)
    selection_ref=m.rt.write_new(tmp_path/'bound-selected.json',value)
    saved=m.rt.load_predictions(value['selected_predictions']['validation'],window_order_sha256=value['window_order']['validation'])
    metrics=m.rt.metrics(saved,samples);comparison=m.rt.compare_devices(saved,saved,metrics,metrics)
    cpu=dict(schema=train.CPU_SCHEMA_V9,passed=True,selection=selection_ref,checkpoint=value['best'],selected_step=value['selected_step'],
        scope=value['scope'],optimization_scope=value['optimization_scope'],initialization=value['initialization'],
        comparisons={s:deepcopy(comparison) for s in train.NATIVE_COUNTS},task_candidate_verification=proof,**train.FLAGS)
    cpu.update({k:deepcopy(value[k]) for k in ('refinement_inventory','task_subset','task_selection','measured_counts','data_version','model_optimizer_version')})
    value.update(schema=train.PILOT_SCHEMA_V9,cpu_equivalence=m.rt.write_new(tmp_path/'bound-cpu.json',cpu))
    # Full config, scope and optimization validation have their own independent
    # fixtures; this test targets the final CPU/selection/COMPLETE equality gate.
    monkeypatch.setattr(train,'validate_complete',lambda v:v)
    assert train.model_binding(m.rt.write_new(tmp_path/'bound-complete.json',value))[0]['checkpoint_schema']==train.PILOT_SCHEMA_V9
    for fault in ('legacy_cpu','missing_replay','wrong_replay','subset','inventory','initialization','source'):
        bad=deepcopy(value);bad_cpu=deepcopy(cpu)
        if fault=='legacy_cpu':bad_cpu['schema']=train.CPU_SCHEMA_V8
        elif fault=='missing_replay':bad_cpu.pop('task_candidate_verification')
        elif fault=='wrong_replay':bad_cpu['task_candidate_verification']['selected_step']=3000
        elif fault=='subset':bad_cpu['task_subset']=dict(bad_cpu['task_subset'],sha256='0'*64)
        elif fault=='inventory':bad_cpu['refinement_inventory']=dict(bad_cpu['refinement_inventory'],sha256='0'*64)
        elif fault=='initialization':bad_cpu['initialization']={}
        else:bad['inputs']['code']['files']['m2_act_task_selection.py']['sha256']='0'*64
        bad['cpu_equivalence']=m.rt.write_new(tmp_path/('cpu-fault-'+fault+'.json'),bad_cpu)
        with pytest.raises(ValueError):train.model_binding(m.rt.write_new(tmp_path/('complete-fault-'+fault+'.json'),bad))


def test_v9_official_cpu_factory_gradients_padding_and_exact_warm_start_when_available(tmp_path,monkeypatch):
    if Path(sys.prefix).resolve()!=(Path(train.runtime.VENDOR_ROOT)/'.venv').resolve():
        pytest.skip('Pinned official LeRobot environment is Linux-only; no substitute runtime')
    from test_ausc_act_training_v4 import head_spec
    torch=pytest.importorskip('torch');warm=train._warm_start(9);spec=head_spec()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with monkeypatch.context() as isolated:
        isolated.setattr(sys,'path',[p for p in sys.path if '/test-deps' not in p])
        _,policy,software=train.make_model(dict(schema=train.CONFIG_SCHEMA_V9,lerobot_root=train.runtime.VENDOR_ROOT,posture_head=spec),device='cpu',training=True)
    train.validate_official_forward(software)
    assert sum(p.numel() for p in policy.parameters())==2658823 and len(policy.state_dict())==54
    batch={'observation.state':torch.zeros(4,7),'observation.environment_state':torch.zeros(4,27)}
    policy.eval()
    with torch.no_grad():prediction=policy.predict_action_chunk(batch).numpy().astype('float64')
    prediction=prediction*np.asarray(spec['normalization']['action_std'])+spec['normalization']['action_mean']
    path=tmp_path/'synthetic-official.pt';torch.save(policy.state_dict(),path)
    receipt=dict(state_dict=train.runtime.reference(path),tensor_proofs=warm.tensor_proofs(policy.state_dict()),posture_head=spec,normalization=spec['normalization'],
        probes=dict(observations={k:v.tolist() for k,v in batch.items()},raw_predictions=prediction.tolist()))
    monkeypatch.setattr(warm,'validate_receipt',lambda *a,**k:receipt)
    assert warm.initialize_policy(torch,policy,{},normalization=spec['normalization'],posture_head=spec)['passed']
    assert not warm.fresh_optimizer(torch,policy,lr=1e-4,weight_decay=1e-4)[0].state
    batches={}
    for name,g in train.GROUPS_V3.items():
        actions=torch.ones(g['size'],20,7,requires_grad=True);mask=torch.arange(20)[None,:].expand(g['size'],-1)>=5
        batches[name]={'observation.state':torch.zeros(g['size'],7),'observation.environment_state':torch.zeros(g['size'],27),'action':actions,'action_is_pad':mask}
    policy.eval();torch.manual_seed(19);loss,_=train.mixed_loss(policy,batches,version=9);loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and bool(torch.any(p.grad!=0)) for p in policy.parameters())
    for b in batches.values():
        assert bool(torch.all(b['action'].grad[b['action_is_pad']]==0))
        with torch.no_grad():b['action'][b['action_is_pad']]=123456.
    torch.manual_seed(19);assert torch.equal(train.mixed_loss(policy,batches,version=9)[0],loss)


def test_v9_original_fit_real20000_toy_updates_saved_states_own_cpu_and_complete(tmp_path,monkeypatch,request):
    """Actual V9 loop and admission, with a tiny CPU policy and invented data.

    Only source IO, the Official factory/parameter-count barrier and GPU device
    ownership are local fixtures. No fit, selector, complete/selected validator,
    metric, scorer, optimizer, checkpoint, or own-CPU verifier is substituted.
    Toy population sizes are explicit test-local constants, never production
    counts or evidence of an Official architecture/native qualification.
    """
    import builtins
    from contextlib import nullcontext
    from types import SimpleNamespace
    torch=pytest.importorskip('torch');threads=torch.get_num_threads();torch.set_num_threads(1)
    request.addfinalizer(lambda:torch.set_num_threads(threads))
    from test_ausc_act_refinement_sampling import fixture as sampling_fixture
    from test_ausc_act_task_selection import fixture as selection_fixture
    from test_ausc_act_auxiliary_train import scope_value
    from test_ausc_act_training_v4 import head_spec
    rt=train.runtime;sampler=train.refinement_sampling;warm=train._warm_start(9)
    load_dataset=train.expanded.load_dataset
    d=tmp_path/'banks';d.mkdir();p,a,banks=sampling_fixture(d,monkeypatch)
    v=tmp_path/'validation';v.mkdir();selector,_,validation,val_header=selection_fixture(v,monkeypatch)
    monkeypatch.setattr(train.expanded,'load_dataset',load_dataset)
    # Original four native TRAIN identities, with small actual array populations.
    native=p['train'];header=p['preparation'];old=len(native['states'])
    for key in sampler.query.ARRAY_KEYS:native[key]=np.concatenate([native[key],native[key][:3].copy()])
    episode=deepcopy(header['episodes'][0]);episode.update(episode_id='toy-native-fourth',subject_id='p000833',
        split_sample_range=[old,old+3],windows=3)
    header['episodes'].append(episode)
    native['actions'][:]=0.;native['origins']=native['states'][:,:7].copy()
    validation['action_is_pad'][0]=False
    # These actual observed31mm rows are outside the frozen30mm task subset.
    # Their majority of valid cells makes the full raw-MAE minimum later than
    # the task minimum, using actual predictions/labels rather than fake scores.
    validation['action_is_pad'][[2,5,8]]=False
    validation['actions'][[2,5,8]]=1.5e-6
    counts=dict(train=len(native['states']),validation=len(validation['states']))
    monkeypatch.setattr(train,'NATIVE_COUNTS',counts)
    norm=rt.given.fit_normalization(native,split='train')
    norm_ref=rt.write_new(tmp_path/'TOY_NORMALIZATION.json',norm)
    spec=head_spec();spec['normalization']=norm
    # Bind local profile sources; actual target representability audits remain.
    episodes=header['episodes']+val_header['episodes'];index=[]
    for i,e in enumerate(episodes):
        config=rt.read_bound(e['config']);config['posture_limits']=spec['physical_posture_profile']
        if i==3:
            config['m2_given_contact_control']['metadata'].update(subject_id=e['subject_id'],episode_id=e['episode_id'])
            config['m2_given_contact_control']['episode_id']=e['episode_id']
        e['config']=rt.write_new(tmp_path/('TOY_SOURCE-'+str(i)+'.json'),config)
        samples=native if e['split']=='train' else validation
        lo,hi=e['split_sample_range']
        index.extend(dict(split=e['split'],sample_index=j,episode_id=e['episode_id'],
            window=dict(valid_future_cells=int((~samples['action_is_pad'][j]).sum()))) for j in range(lo,hi))
    arrays={}
    for split,samples in (('train',native),('validation',validation)):
        path=tmp_path/('TOY_'+split+'.npz')
        with path.open('xb') as stream:np.savez(stream,**{k:samples[k] for k in sampler.query.ARRAY_KEYS})
        arrays[split]=rt.reference(path)
    header.update(schema=train.expanded.MASKED_PREPARATION_SCHEMA,complete=True,rebuild_epoch=train.binding.EPOCH,original_cases=0,
        diagnostic_only=True,formal_acceptance=False,training_eligible=False,test_payloads_opened=False,task_contract=rt.task_contract(),
        counts=counts,episodes=episodes,split_subject_ids=dict(train=sorted(train.binding.TRAIN_IDS),validation=['p000923']),
        arrays=arrays,normalization=norm_ref,index=rt.write_new(tmp_path/'TOY_INDEX.json',dict(schema=rt.dataset.INDEX_SCHEMA,
            task_id=rt.pose.TASK_ID,action_semantics_id=rt.given.ACTION_SEMANTICS_ID,samples=index)))
    dataset=rt.write_new(tmp_path/'TOY_DATASET.json',header)
    monkeypatch.setattr(train,'NATIVE_DATASET',dataset);monkeypatch.setattr(sampler.query,'BASE29',dataset)
    monkeypatch.setattr(selector,'BASE29',dataset);monkeypatch.setattr(selector,'BASE_COUNTS',counts)
    monkeypatch.setattr(selector,'NORMALIZATION',norm_ref)
    # Three strictly distinct bank headers and actual array/index proofs remain
    # admitted through the real V9 consumer/sampler. Historical producer IO is
    # already a fixture in sampling_fixture; it never opens the old archives.
    for name,bank in banks.items():
        bank['metadata'].update(base_dataset=dataset,base_counts=counts,normalization=norm_ref)
        bank['samples']['normalization_reference']=norm_ref
        bank['complete']['metadata']=rt.write_new(tmp_path/(name+'-TOY_METADATA.json'),bank['metadata'])
        bank['reference']=rt.write_new(tmp_path/(name+'-TOY_COMPLETE.json'),bank['complete'])
    # Positive raw auxiliary labels give this toy a persistent real gradient;
    # observed task-subset labels stay zero, so task selection precedes20k.
    a['samples']['relative_setters'][:]=.01
    for i,s in enumerate(a['sources']):
        ref=rt.write_new(tmp_path/('TOY_AUX_SOURCE-'+str(i)+'.json'),dict(posture_limits=spec['physical_posture_profile']))
        s.setdefault('bindings',{}).update(source=ref,geometry=ref)
    path=tmp_path/'TOY_AUXILIARY.npz'
    with path.open('xb') as stream:np.savez(stream,**{k:a['samples'][k] for k in train.phase_data.ARRAY_KEYS})
    a['array_reference']=rt.reference(path)
    a['provenance']=dict(input_profile='production_closed_inputs',
        parent_admission=dict(complete=train.phase_data.PRODUCTION_INPUTS.parent_complete),
        producer_admission=dict(producer=train.phase_data.PRODUCTION_INPUTS.producer_complete))
    auxiliary=rt.write_new(tmp_path/'TOY_AUXILIARY.json',dict(arrays=a['array_reference']))
    meta=dict(sources=a['sources'],counts=a['counts'],input_profile='production_closed_inputs',optimization_admission=train.phase_data._admission(True))
    monkeypatch.setattr(train.phase_data,'metadata',lambda ref,*args,**kwargs:(rt.read_bound(ref),deepcopy(meta)))
    monkeypatch.setattr(train.phase_data,'load_preparation',lambda *args,**kwargs:deepcopy(a))
    frozen=scope_value();frozen.update(optimized_train_ids=sorted(train.scope.TRAIN16_IDS[:6]),unavailable_train_ids=sorted(train.scope.TRAIN16_IDS[6:]))
    scope_ref=rt.write_new(tmp_path/'TOY_SCOPE.json',frozen);monkeypatch.setattr(train,'TRAIN16_SCOPE',scope_ref)
    monkeypatch.setattr(train.scope,'validate',lambda ref:train.scope.validate_train16_metadata(rt.read_bound(ref)))
    monkeypatch.setattr(train.posture_head,'pinned_spec',lambda normalization,**kwargs:deepcopy(spec))
    subset=selector.freeze_subset(dataset,tmp_path/'TOY_FROZEN_SUBSET')

    class ToyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.values=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([i/100.])) for i in range(54)])
            self._posture_head_spec=deepcopy(spec)
            self._posture_head_diagnostics=dict(synthetic_cpu_toy=True)
        def predict_action_chunk(self,batch):
            return self.values[0].reshape(1,1,1).expand(len(batch['observation.state']),20,7)
        def forward(self,batch):
            prediction=self.predict_action_chunk(batch);valid=~batch['action_is_pad']
            loss=((prediction-batch['action']).square()*valid[...,None]).sum()/(valid.sum()*7)
            return loss,{}
    initial=ToyPolicy();initial_state={k:t.clone() for k,t in initial.state_dict().items()}
    path=tmp_path/'TOY_INITIAL_STATE.pt';torch.save(initial_state,path)
    receipt=dict(schema=warm.SCHEMA,normalization=norm,posture_head=spec,state_dict=rt.reference(path),
        inputs=dict(synthetic_cpu_toy=True),tensor_proofs=warm.tensor_proofs(initial_state),
        probes=dict(observations={'observation.state':np.zeros((4,7)).tolist(),'observation.environment_state':np.zeros((4,27)).tolist()},
            raw_predictions=np.broadcast_to(np.asarray(norm['action_mean']),(4,20,7)).tolist()))
    init=rt.write_new(tmp_path/'TOY_INITIALIZATION.json',receipt)
    def local_initialization(ref,*,normalization=None,posture_head=None):
        assert ref==init
        value=rt.read_bound(ref)
        if normalization is not None:assert value['normalization']==normalization
        if posture_head is not None:assert value['posture_head']==posture_head
        return value
    monkeypatch.setattr(warm,'validate_receipt',local_initialization)
    config=dict(train.SCHEDULE_V9,schema=train.CONFIG_SCHEMA_V9,base=str(tmp_path),dataset=dataset,auxiliary=auxiliary,
        evaluation_cohort=scope_ref,initialization=init,posture_head=spec,task_subset=subset,measured_banks={n:b['reference'] for n,b in banks.items()})
    config_ref=rt.write_new(tmp_path/'TOY_CONFIG.json',config)
    software=dict(synthetic_cpu_toy=True,**{k:{} for k in ('vendor','dependencies','python','prefix','python_version','source_bindings')})
    models=[]
    def factory(*args,**kwargs):
        policy=ToyPolicy();models.append(policy);return torch,policy,deepcopy(software)
    monkeypatch.setattr(train,'make_model',factory)
    # Isolate the existing Official factory's count barrier exactly as the V8
    # loop harness does; all actual draw/metric/candidate sums use builtins.sum.
    monkeypatch.setattr(train,'sum',lambda values:2658823 if (v:=list(values))==[1]*54 else builtins.sum(v),raising=False)
    monkeypatch.setattr(train,'validate_official_forward',lambda value:rt.need(value.get('synthetic_cpu_toy') is True,'Expected toy factory boundary'))
    monkeypatch.setattr(rt,'owned_gpu',lambda:nullcontext(dict(synthetic_cpu_toy=True)))
    monkeypatch.setattr(rt,'_identity',lambda:dict(pid=1,pgid=1,sid=1))
    monkeypatch.setattr(rt,'execution_settings',lambda *args,**kwargs:dict(device='cpu',synthetic_cpu_toy=True))
    build=train.build_cache
    monkeypatch.setattr(train,'build_cache',lambda prepared,aux,**kwargs:build(prepared,aux,torch=torch,device='cpu',optimizer_version=9))
    infer=rt.infer;inferences=[]
    def cpu_device(torch_arg,policy,samples,stats,*,device):
        inferences.append((device,len(samples['states'])));return infer(torch_arg,policy,samples,stats,device='cpu')
    monkeypatch.setattr(rt,'infer',cpu_device)
    fresh_optimizer=warm.fresh_optimizer;updates=[];optimizers=[]
    def tracked_optimizer(*args,**kwargs):
        optimizer,proof=fresh_optimizer(*args,**kwargs);assert not optimizer.state
        actual_step=optimizer.step
        def step(*args,**kwargs):
            result=actual_step(*args,**kwargs);updates.append(float(models[0].values[0].detach()));return result
        optimizer.step=step;optimizers.append(optimizer);return optimizer,proof
    monkeypatch.setattr(warm,'fresh_optimizer',tracked_optimizer)
    run=train.subprocess.run;cpu_dispatch=[]
    def own_cpu(command,*args,**kwargs):
        if '--verify-selected' not in command:return run(command,*args,**kwargs)
        cpu_dispatch.append(command);selection=rt.reference(command[command.index('--verify-selected')+1])
        train.verify_selected(selection,command[command.index('--receipt')+1])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(train.subprocess,'run',own_cpu)
    result=train.fit(config_ref,tmp_path/'TOY_FIT')
    assert len(updates)==20000 and len(optimizers)==1 and cpu_dispatch
    assert len(set(updates))==20000 and result['completed_steps']==20000
    assert result['selected_step']<20000 and len(result['evaluation_candidates'])==40
    assert len(models)>=2 and models[0] is not models[1] # cold selected CPU policy
    assert inferences.count(('cuda',counts['train']))==inferences.count(('cuda',counts['validation']))==40
    assert ('cpu',counts['train']) in inferences and ('cpu',counts['validation']) in inferences
    for row in result['evaluation_candidates']:
        packet=torch.load(rt.bound_path(row['checkpoint']),weights_only=True)
        policy=ToyPolicy();policy.load_state_dict(packet['state_dict'],strict=True)
        assert float(policy.values[0].detach())==updates[row['step']-1]
        for split in counts:
            saved=rt.load_predictions(row['predictions'][split],window_order_sha256=result['window_order'][split])
            actual=infer(torch,policy,native if split=='train' else validation,norm,device='cpu')
            assert saved.shape==(counts[split],20,7) and saved.dtype==np.float64 and np.array_equal(saved,actual)
        saved=rt.load_predictions(row['predictions']['validation'],window_order_sha256=result['window_order']['validation'])
        assert row['score']==selector.score(saved,validation,subset)
    assert train.selected_step(result['evaluations'],version=9)==result['selected_step']
    assert train.expanded.selected_step(result['evaluations'])!=result['selected_step']
    assert result['optimization']['gradient_examples']['native']==20000*32
    assert result['selected_optimization']['gradient_examples']['native']==result['selected_step']*32
    assert result['selected_optimization']!=result['optimization']
    last=torch.load(rt.bound_path(result['last']),weights_only=True)
    assert {int(s['step']) for s in last['optimizer_state_dict']['state'].values()}=={20000}
    assert last['initialization']['optimizer']['initial_step']==0
    assert train.validate_complete(result) is result
    assert train.model_binding(rt.reference(tmp_path/'TOY_FIT/COMPLETE.json'))[0]['checkpoint_schema']==train.PILOT_SCHEMA_V9
    cpu=rt.read_bound(result['cpu_equivalence'])
    assert cpu['task_candidate_verification']==train.verify_task_candidates(result)
    for fault in ('completed','selected_exposure','completed_exposure','candidate_state','saved_prediction','task_score','cpu'):
        bad=deepcopy(result)
        if fault=='completed':bad['completed_steps']=19999
        elif fault=='selected_exposure':bad['selected_optimization']=deepcopy(bad['optimization'])
        elif fault=='completed_exposure':bad['optimization']=deepcopy(bad['selected_optimization'])
        elif fault=='candidate_state':bad['evaluation_candidates'][0]['checkpoint']=bad['evaluation_candidates'][1]['checkpoint']
        elif fault=='saved_prediction':bad['evaluation_candidates'][0]['predictions']['validation']=bad['evaluation_candidates'][1]['predictions']['validation']
        elif fault=='task_score':
            bad['evaluation_candidates'][0]['score']['task_score']+=1e-15
            bad['task_selection']['candidates'][0]['score']=deepcopy(bad['evaluation_candidates'][0]['score'])
        else:
            bad_cpu=deepcopy(cpu);bad_cpu['task_candidate_verification']['verified_candidates']=39
            bad['cpu_equivalence']=rt.write_new(tmp_path/'TOY_BAD_CPU.json',bad_cpu)
        with pytest.raises(ValueError):
            if fault=='cpu':train.model_binding(rt.write_new(tmp_path/'TOY_BAD_COMPLETE.json',bad))
            else:train.validate_complete(bad)
