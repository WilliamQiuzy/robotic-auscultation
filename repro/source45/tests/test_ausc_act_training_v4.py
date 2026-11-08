"""V4 software fixtures: no production weights, native runtime or GPU ownership."""
from copy import deepcopy
import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy import m2_act_auxiliary_train as train


def warm():
    name='sim.ausc_policy.m2_act_warm_start'
    assert importlib.util.find_spec(name), 'Explicit isolated ACT5 warm-start API required'
    return importlib.import_module(name)


def head_spec():
    from test_ausc_act_posture_head import spec
    from sim.ausc_policy import m2_act_posture_head as head
    old=spec()
    return head.make_spec(old['physical_posture_profile'],old['profile_sources'],old['normalization'],version=2)


def test_v4_schema_family_and_unchanged_four_group_protocol():
    assert hasattr(train,'CONFIG_SCHEMA_V4'), 'Explicit V4 config/model/selection/CPU/protocol family required'
    from sim.ausc_policy import m2_act_terminal_data as data
    for key in ('CONFIG','PILOT','SELECTION','CPU','PROTOCOL'):
        assert getattr(train,key+'_SCHEMA_V4').endswith('_v4')
    assert train._api(4) is data and train._groups(4)==train.GROUPS_V3
    assert set(train._kinds(4))==set(train.conditional_data.KINDS)
    assert train._version({'schema':data.SCHEMA})==4
    assert train._schemas({'schema':train.CONFIG_SCHEMA_V4})==(train.PILOT_SCHEMA_V4,train.SELECTION_SCHEMA_V4,train.CPU_SCHEMA_V4,train.PROTOCOL_SCHEMA_V4)
    assert train.SCHEDULE['steps']==20000 and train.expanded.evaluation_steps(train.SCHEDULE)==list(range(500,20001,500))
    assert train._groups(1)==train._groups(2)==train.GROUPS


def test_v4_make_model_never_zeroes_act5_head(monkeypatch):
    assert hasattr(train,'CONFIG_SCHEMA_V4'), 'Explicit V4 model required'
    spec=head_spec();policy=SimpleNamespace();calls=[]
    monkeypatch.setattr(train.runtime,'make_model',lambda *a,**k:('torch',policy,'software'))
    monkeypatch.setattr(train.posture_head,'install',lambda p,s:calls.append((p,s)))
    monkeypatch.setattr(train.posture_head,'initialize_training',lambda *a:pytest.fail('V4 reset the ACT5 learned final layer'))
    config=dict(schema=train.CONFIG_SCHEMA_V4,lerobot_root=train.runtime.VENDOR_ROOT,posture_head=spec)
    assert train.make_model(config,device='cpu',training=True)[1] is policy
    assert calls==[(policy,spec)]
    with pytest.raises(ValueError,match='head version'):
        train.make_model(dict(config,posture_head=dict(spec,schema=train.posture_head.SCHEMA,softness_rad=.001)),device='cpu')


def test_v4_public_entrypoints_dispatch_and_ipc_retains_marker(tmp_path,monkeypatch):
    assert hasattr(train,'CONFIG_SCHEMA_V4'), 'Explicit V4 dispatch required'
    from sim.ausc_policy import m2_given_contact_expanded_pilot_train as expanded
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep,m2_rebuild_paired_eval as paired
    sentinel=object()
    for name in ('validate_config','training_protocol','load_selected_policy','verify_selected','fit','validate_complete','model_binding'):
        monkeypatch.setattr(train,name,lambda *a,**k:sentinel)
    config={'schema':train.CONFIG_SCHEMA_V4}
    assert expanded.validate_config(config) is sentinel and expanded.training_protocol(config) is sentinel
    assert expanded.load_selected_policy({'schema':train.SELECTION_SCHEMA_V4}) is sentinel
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.SELECTION_SCHEMA_V4 if ref=='selection' else train.CONFIG_SCHEMA_V4))
    assert expanded.verify_selected('selection','receipt') is sentinel and expanded.fit('config','output') is sentinel
    assert dep.validate_complete({'schema':train.PILOT_SCHEMA_V4}) is sentinel
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:dict(schema=train.PILOT_SCHEMA_V4))
    assert dep.model_binding('ref') is sentinel
    from test_ausc_given_contact_pose import reference,metadata
    ref=reference(tmp_path);meta=dict(metadata(ref),episode_id='v4',target_id='L1');q=np.zeros(7)
    state=dep.runtime.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    descriptor=dict(metadata=meta,episode_id='v4',source_episode_id='native',model=dict(checkpoint={'sha256':'a'*64},
        checkpoint_schema=train.PILOT_SCHEMA_V4),source_config={'sha256':'b'*64},protocol={'sha256':'c'*64},scenario_id='s')
    for adapter in (dep,paired):
        request=adapter.make_request(descriptor,1,q,ref.pose_base,1.,metadata=meta,history=np.stack([state,state]),
            history_times=[0.,.05],context=dict(observed_at_s=.05,physics_row=12,context=['v4','L1',0,'FREE'],
                source_id='given-contact-'+ref.sha256))
        adapter.validate_request(request,descriptor,sequence=1)
        block=adapter.decode_response(adapter.response_for(request,np.zeros((20,7))),request)
        assert block.metadata['checkpoint_policy']==train.PILOT_SCHEMA_V4


def test_warm_start_pins_exact_act5_files_and_original_archive():
    m=warm()
    assert m.COMPLETE['bytes']==471093 and m.COMPLETE['sha256']=='f6a6c8f1956fd067e708580b1eee3b24d624967015b11ece8482b2459c2b4b37'
    assert m.WEIGHTS['bytes']==10762689 and m.WEIGHTS['sha256']=='8258390271fb23719b718468181e71a864c07fb1b8cc2d3584e4f1af2cbd6384'
    assert m.CPU_EQUIVALENCE['bytes']==12449 and m.CPU_EQUIVALENCE['sha256']=='2d975906dc3fdbeaa9920b9bba76689d6c2312334275e01085f330e6bb9e1a5b'
    assert m.frozen26.SOURCE26_COMMIT=='05e972ca184383a525aff293fc9f3590e546da9c'
    assert m.OFFICIAL_PYTHON=='/opt/wam-runtime/vendor/lerobot-eb7a54c/.venv/bin/python'
    assert m.PROBE_TRAIN_INDICES==[0,64,4186,8372]


def test_v4_code_inventory_is_additive_and_has_actual_module_paths():
    old=train.code_binding(version=3);current=train.code_binding(version=4)
    for name in ('m2_act_terminal_data.py','m2_act_frozen26_data.py','m2_act_warm_start.py'):
        assert name in current['files'], 'V4 new numerical files must be frozen'
        assert name not in old['files']
    assert set(old['files'])<set(current['files'])
    from pathlib import Path
    root=Path(train.__file__).resolve().parent
    assert all(Path(ref['path'])==root/name for name,ref in current['files'].items())


def initialization_fixture(tmp_path,monkeypatch):
    """Small tensor artifact with explicit fake legacy headers for local tests only."""
    torch=pytest.importorskip('torch');m=warm();spec=head_spec();rt=m.rt
    policy=torch.nn.Linear(7,7)
    with torch.no_grad():
        policy.weight.copy_(torch.arange(49,dtype=torch.float32).reshape(7,7)/100)
        policy.bias.copy_(torch.arange(7,dtype=torch.float32)/10)
    state={k:v.detach().clone() for k,v in policy.state_dict().items()}
    state_file=tmp_path/'INITIAL_STATE.pt';torch.save(state,state_file)
    norm_ref=rt.write_new(tmp_path/'normalization.json',spec['normalization'])
    native_ref=rt.write_new(tmp_path/'native.json',dict(normalization=norm_ref))
    forward='src/lerobot/policies/act/modeling_act.py'
    software=dict(vendor=dict(root=rt.VENDOR_ROOT,upstream_commit=rt.UPSTREAM_COMMIT,
        files={forward:dict(path=rt.VENDOR_ROOT+'/'+forward,bytes=1,sha256=train.OFFICIAL_FORWARD_SHA256)}),
        prefix=rt.VENDOR_ROOT+'/.venv',source_bindings={m.frozen26.SOURCE26_ROOT+'/sim/ausc_policy/m2_act.py':'a'*64})
    software_ref=rt.write_new(tmp_path/'software.json',software)
    source_code=dict(code_commit=m.frozen26.SOURCE26_COMMIT,files={'m2_act.py':dict(path=m.frozen26.SOURCE26_ROOT+'/sim/ausc_policy/m2_act.py',bytes=1,sha256='a'*64)})
    selection=dict(schema=train.SELECTION_SCHEMA_V3,selected_step=19000,best=m.WEIGHTS,
        inputs=dict(dataset=native_ref,software=software_ref,code=source_code),
        protocol=dict(posture_head=spec),task_contract=rt.task_contract())
    selected_ref=rt.write_new(tmp_path/'SELECTED.json',selection)
    complete=dict(selection,schema=train.PILOT_SCHEMA_V3,complete=True,completed_steps=20000,cpu_equivalence=m.CPU_EQUIVALENCE)
    cpu=dict(selection=selected_ref,checkpoint=m.WEIGHTS,selected_step=19000,passed=True)
    original_read=rt.read_bound;original_bound=rt.bound_path
    monkeypatch.setattr(rt,'read_bound',lambda ref:deepcopy(complete) if ref==m.COMPLETE else deepcopy(cpu) if ref==m.CPU_EQUIVALENCE else original_read(ref))
    monkeypatch.setattr(rt,'bound_path',lambda ref:state_file if ref==m.WEIGHTS else original_bound(ref))
    batch={'observation.state':np.arange(28,dtype=np.float32).reshape(4,7)/100,
        'observation.environment_state':np.zeros((4,27),np.float32)}
    class Tiny(torch.nn.Linear):
        def predict_action_chunk(self,b):return self(b['observation.state'])[:,None,:].expand(-1,20,-1)
    actual=Tiny(7,7);actual.load_state_dict(state)
    with torch.no_grad():prediction=actual.predict_action_chunk({k:torch.from_numpy(v) for k,v in batch.items()}).numpy()
    raw=prediction*np.array(spec['normalization']['action_std'])+np.array(spec['normalization']['action_mean'])
    receipt=dict(schema=m.SCHEMA,complete=True,origin='ACT5 selected19000; original26 CPU authenticated',
        inputs=dict(complete=m.COMPLETE,weights=m.WEIGHTS,cpu_equivalence=m.CPU_EQUIVALENCE,selection=selected_ref,
            software=software_ref,normalization=norm_ref,source_manifest=m.frozen26.SOURCE26_MANIFEST),
        source26=dict(code_commit=m.frozen26.SOURCE26_COMMIT,file_count=1660,root=m.frozen26.SOURCE26_ROOT),
        selected_step=19000,completed_steps=20000,parameter_count=2658823,model=rt.shared.native_spec(),
        task_contract=rt.task_contract(),normalization=spec['normalization'],posture_head=spec,
        state_dict=rt.reference(state_file),tensor_proofs=m.tensor_proofs(state),optimizer_imported=False,scheduler_imported=False,
        probes=dict(split='train',dataset=native_ref,indices=m.PROBE_TRAIN_INDICES,
            observations={k:v.tolist() for k,v in batch.items()},raw_predictions=raw.tolist()),
        execution=dict(device='cpu',intraop_threads=1,interop_threads=1,cuda_visible_devices=''))
    ref=rt.write_new(tmp_path/'INITIALIZATION.json',receipt)
    return m,torch,ref,receipt,Tiny,selection,complete,cpu


@pytest.mark.parametrize('fault',['complete','weights','cpu','manifest','head','norm','vendor','origin','selection','source_path','optimizer'])
def test_initialization_rejects_wrong_legacy_identity_metadata_without_loading_weights(tmp_path,monkeypatch,fault):
    m,torch,ref,value,Tiny,selection,complete,cpu=initialization_fixture(tmp_path,monkeypatch)
    assert m.validate_receipt(ref,normalization=value['normalization'],posture_head=value['posture_head'])==value
    changed=deepcopy(value)
    if fault in ('complete','weights','manifest','cpu'):
        key={'manifest':'source_manifest','cpu':'cpu_equivalence'}.get(fault,fault)
        changed['inputs'][key]['sha256']='f'*64
    elif fault=='head':changed['posture_head']['softness_rad']=.001
    elif fault=='norm':changed['normalization']['action_mean'][0]+=.1
    elif fault=='vendor':
        software=m.rt.read_bound(value['inputs']['software']);software['vendor']['upstream_commit']='a'*40
        changed['inputs']['software']=m.rt.write_new(tmp_path/'badsoftware.json',software)
    elif fault=='origin':changed['source26']['root']='/copied/source27'
    elif fault=='selection':changed['selected_step']=20000
    elif fault=='source_path':
        selection['inputs']['code']['files']['m2_act.py']['path']='/copy/m2_act.py'
        changed['inputs']['selection']=m.rt.write_new(tmp_path/'badselected.json',selection)
        cpu['selection']=changed['inputs']['selection']
    else:changed['optimizer_imported']=True
    bad=m.rt.write_new(tmp_path/'bad-init.json',changed)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Metadata validation loaded old or exported tensors'))
    with pytest.raises(ValueError):m.validate_receipt(bad,normalization=value['normalization'],posture_head=value['posture_head'])


def test_strict_initial_tensors_prediction_proof_and_fresh_optimizer(tmp_path,monkeypatch):
    m,torch,ref,value,Tiny,*_=initialization_fixture(tmp_path,monkeypatch)
    policy=Tiny(7,7);policy._posture_head_spec=deepcopy(value['posture_head'])
    result=m.initialize_policy(torch,policy,ref,normalization=value['normalization'],posture_head=value['posture_head'])
    assert result['passed'] is True and result['initial_tensor_proofs']==value['tensor_proofs']
    assert result['prediction_max_abs_rad']==0. and result['optimizer_step']==0
    assert result['optimizer_imported'] is False
    optimizer,settings=m.fresh_optimizer(torch,policy,lr=1e-4,weight_decay=1e-4)
    assert not optimizer.state and settings['initial_state_entries']==settings['initial_step']==0
    # A meaningful update confirms gradients use the imported values.
    before={k:v.clone() for k,v in policy.state_dict().items()}
    optimizer.zero_grad();policy(torch.ones((4,7))).square().mean().backward();optimizer.step()
    assert any(not torch.equal(v,policy.state_dict()[k]) for k,v in before.items())
    assert {int(row['step']) for row in optimizer.state.values()}=={1}
    changed=deepcopy(value);changed['tensor_proofs']['weight']['sha256']='0'*64
    bad=m.rt.write_new(tmp_path/'badtensor.json',changed)
    with pytest.raises(ValueError,match='tensor'):m.initialize_policy(torch,Tiny(7,7),bad,normalization=value['normalization'],posture_head=value['posture_head'])
    with pytest.raises(ValueError,match='tensor|shape'):m.initialize_policy(torch,torch.nn.Linear(8,7),ref,normalization=value['normalization'],posture_head=value['posture_head'])
    reset=Tiny(7,7);reset._posture_training_initialization={}
    with pytest.raises(ValueError,match='reset|zero'):m.initialize_policy(torch,reset,ref,normalization=value['normalization'],posture_head=value['posture_head'])


def test_export_runs_only_pinned_official_cpu_original26_and_checks_archive_twice(tmp_path,monkeypatch):
    m=warm();calls=[]
    monkeypatch.setattr(m.sys,'platform','linux')
    monkeypatch.setattr(m.frozen26,'require_archive',lambda:calls.append('archive'))
    monkeypatch.setattr(m.rt,'bound_path',lambda ref:calls.append(ref))
    monkeypatch.setattr(m,'validate_receipt',lambda ref,**k:{'complete':True})
    def run(argv,**kwargs):
        assert argv[:3]==[m.OFFICIAL_PYTHON,'-I','-B'] and argv[-1]==m.frozen26.SOURCE26_ROOT
        assert kwargs['cwd']==m.frozen26.SOURCE26_ROOT
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and kwargs['env']['OMP_NUM_THREADS']=='1'
        assert 'PYTHONPATH' not in kwargs['env']
        program=argv[4]
        assert 'original.model_binding(' in program and "cpu['selection']" in program and 'original.load_selected_policy(' in program
        assert 'initialize_training' not in program
        return SimpleNamespace(returncode=0,stdout='{"schema":"fixture-child","state_dict":{}}',stderr='')
    monkeypatch.setattr(m.subprocess,'run',run)
    ref=m.export(tmp_path/'export')
    assert calls.count('archive')==2 and ref['path']==str(tmp_path/'export'/'INITIALIZATION.json')


def test_v4_protocol_names_warm_treatment_and_excludes_old_evaluations(tmp_path,monkeypatch):
    m,_,ref,value,*_=initialization_fixture(tmp_path,monkeypatch)
    config=dict(train.SCHEDULE,schema=train.CONFIG_SCHEMA_V4,posture_head=value['posture_head'],initialization=ref)
    p=train.training_protocol(config)
    assert p['posture_initialization']==m.protocol(ref) and p['posture_initialization']['final_linear_reset'] is False
    assert p['posture_initialization']['optimizer_initial_step']==0
    assert 'fine-tuned ACT6' in p['treatment'] and '20000 new' in p['treatment']
    assert p['auxiliary_admission']['preparation_schema']==train.terminal_data.SCHEMA
    assert p['evaluation_steps']==list(range(500,20001,500)) and len(p['evaluation_steps'])==40
    assert 'excluded' in p['posture_initialization']['original_evaluations']
    assert p['objective']==train.training_protocol(dict(train.SCHEDULE,schema=train.CONFIG_SCHEMA_V3,posture_head=value['posture_head']))['objective']


def v4_cache_fixture(tmp_path):
    from test_ausc_act_auxiliary_train import cache_inputs,source
    p,a=cache_inputs(tmp_path);s=a['samples'];n=len(s['source_indices'])
    conditional=source('planned_ideal_tracking');conditional.update(kind='conditional_ideal_continuation',anchor='near60mm')
    new=source('student_oracle_unexecuted');new.update(schema=train.terminal_data.SOURCE_SCHEMA)
    a['sources'] += [conditional,new]
    for i,entry in enumerate(a['sources']):entry['source_id']='aux:'+str(i)
    for key in ('source_histories','relative_setters','setter_is_pad'):
        s[key]=np.concatenate([s[key],s[key][-2:]])
    s['source_indices']=np.r_[s['source_indices'],3,4]
    s.update(schema=train.terminal_data.SCHEMA,optimization_admission=deepcopy(train.terminal_data.ADMISSION))
    return p,a,n


def test_v4_cache_same_groups_new_genuine_sources_in_oracle_and_actual_draws(tmp_path,monkeypatch):
    p,a,n=v4_cache_fixture(tmp_path)
    cache=train.build_cache(p,a)
    rng=np.random.default_rng(42);seen=[]
    value=dict(gradient_examples={k:0 for k in train.GROUPS_V3},gradient_subject_examples={k:{} for k in train.GROUPS_V3},
        gradient_source_examples={k:{} for k in train.GROUPS_V3},gradient_kind_examples={k:{} for k in train.GROUPS_V3},actual_optimized_train_ids=[])
    for _ in range(30):
        ids=train.sample_cached_indices(rng,cache);seen.extend(ids['oracle'])
        assert {k:len(v) for k,v in ids.items()}==dict(native=32,planned=8,conditional=8,oracle=16)
        assert np.all(ids['conditional']==n)
        train.record_draws(value,cache,ids)
    assert n+1 in seen and set(seen)<=set([42,43,n+1])
    assert value['gradient_source_examples']['oracle']['aux:4']>0
    assert value['gradient_kind_examples']['oracle']=={'student_oracle_unexecuted':30*16}
    assert value['gradient_examples']=={k:30*v['size'] for k,v in train.GROUPS_V3.items()}


def test_v4_loss_padding_poison_is_invariant_and_backward_finite():
    torch=pytest.importorskip('torch');parameter=torch.nn.Parameter(torch.tensor(.5));batches={}
    class Official:
        def __call__(self,batch):
            # A genuine valid-cell mean implementation, independent of trainer.
            valid=~batch['action_is_pad'];error=(parameter-batch['action']).abs()
            return (error*valid[...,None]).sum()/(valid.sum()*7),{}
    for i,(name,g) in enumerate(train.GROUPS_V3.items(),1):
        mask=torch.arange(20)[None,:].expand(g['size'],-1)>=i
        action=torch.full((g['size'],20,7),float(i));action[mask]=1234.
        batches[name]=dict(action=action,action_is_pad=mask)
    loss,logged=train.mixed_loss(Official(),batches,version=4);original=float(loss.detach())
    assert original==pytest.approx(.5*.5+.125*1.5+.125*2.5+.25*3.5)
    loss.backward();assert torch.isfinite(parameter.grad) and parameter.grad!=0
    for b in batches.values():b['action'][b['action_is_pad']]=-6789.
    again,_=train.mixed_loss(Official(),batches,version=4);assert float(again.detach())==original
    assert logged['oracle']['valid_cells']==16*4 and logged['conditional']['examples']==8


def test_v4_authenticates_parent_in_original26_and_only_new_queries_in_current_bridge(monkeypatch):
    parent_proof=dict(manifest=train.terminal_data.frozen26.SOURCE26_MANIFEST,array_proofs={'proof':'actual'})
    auxiliary=dict(samples=dict(schema=train.terminal_data.SCHEMA),sources=[],
        provenance=dict(parent_archive_validation=parent_proof,parent_complete=train.terminal_data.PARENT_COMPLETE))
    calls=[]
    monkeypatch.setattr(train.terminal_data,'authenticate_new_genuine_groups',lambda a,ids:calls.append(a) or [dict(new=True)])
    monkeypatch.setattr(train.conditional_data,'require_oracle_archive',lambda *a:pytest.fail('Current28 tried to authenticate parent V3 data'))
    result=train.authenticate_oracles(auxiliary)
    assert calls==[auxiliary] and result[-1]==dict(new=True)
    assert result[0]['parent_original26_validation']==parent_proof and result[0]['actual_history_authenticated'] is True


def test_v4_inventory_uses_v4_metadata_and_keeps_new_source_counts(tmp_path,monkeypatch):
    p,a,_=v4_cache_fixture(tmp_path);called=[]
    monkeypatch.setattr(train,'auxiliary_metadata',lambda ref,**kw:called.append(kw['version']) or {'sources':a['sources']})
    monkeypatch.setattr(train.runtime,'read_bound',lambda ref:p['preparation'])
    inventory=train.contribution_inventory(dict(schema=train.CONFIG_SCHEMA_V4,dataset={},auxiliary={}))
    assert called==[4] and set(inventory)==set(train.GROUPS_V3)
    assert set(inventory['oracle'])=={'aux:2','aux:4'} and inventory['conditional']['aux:3']['kind']=='conditional_ideal_continuation'


def test_native_sources_select_v4_files_without_changing_legacy_inventory(monkeypatch):
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep,m2_rebuild_paired_eval as paired
    old=dep.sources();new=dep.sources(model_schema=train.PILOT_SCHEMA_V4)
    assert 'm2_act_warm_start.py' not in old['files'] and 'm2_act_warm_start.py' in new['files']
    # Avoid camera file access on Mac while exercising actual paired composition.
    from sim.ausc_policy import approved_camera
    monkeypatch.setattr(paired.runtime,'reference',lambda path:dict(path=str(path),bytes=1,sha256=approved_camera.CAMERA_SHA256) if 'camera' in str(path).lower() else dict(path=str(path),bytes=1,sha256='a'*64))
    assert 'm2_act_warm_start.py' in paired.sources(model_schema=train.PILOT_SCHEMA_V4)['files']


def test_v4_config_requires_bound_initialization_and_only_metadata_admission(tmp_path,monkeypatch):
    m,torch,init,value,*_=initialization_fixture(tmp_path,monkeypatch)
    from test_ausc_act_auxiliary_train import scope_value,source
    from sim.ausc_policy import m2_act_recovery_scope as scope,rebuild_gpu_binding as binding
    frozen=scope_value();scope_ref=m.rt.write_new(tmp_path/'scope.json',frozen)
    native=dict(schema='m2_given_contact_masked_preparation_v1',complete=True,rebuild_epoch=binding.EPOCH,
        original_cases=0,diagnostic_only=True,formal_acceptance=False,training_eligible=False,test_payloads_opened=False,
        task_contract=m.rt.task_contract(),counts=train.NATIVE_COUNTS,
        split_subject_ids=dict(train=frozen['cohorts']['train'][:4],validation=['p000923']))
    native_ref=m.rt.write_new(tmp_path/'newnative-header.json',native)
    sources=[source(kind,subject=subject) for kind in train.conditional_data.KINDS for subject in frozen['optimized_train_ids']]
    aux=m.rt.write_new(tmp_path/'v4aux.json',{'schema':train.terminal_data.SCHEMA})
    monkeypatch.setattr(train,'NATIVE_DATASET',native_ref);monkeypatch.setattr(train,'TRAIN16_SCOPE',scope_ref)
    monkeypatch.setattr(scope,'validate',lambda ref:m.rt.read_bound(ref)['cohorts'])
    calls=[];monkeypatch.setattr(train.terminal_data,'metadata',lambda ref,ids:calls.append(ref) or ({},{'sources':sources}))
    monkeypatch.setattr(train.posture_head,'validate_bound_spec',lambda *a,**k:None)
    monkeypatch.setattr(train,'native_normalization',lambda config:value['normalization'])
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('V4 config opened numerical arrays'))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('V4 config loaded historical or initial tensors'))
    config=dict(train.SCHEDULE,schema=train.CONFIG_SCHEMA_V4,base='/workspace/wam',dataset=native_ref,auxiliary=aux,
        evaluation_cohort=scope_ref,posture_head=value['posture_head'],initialization=init)
    assert train.validate_config(config) is config and calls==[aux]
    noinit=dict(config);noinit.pop('initialization')
    with pytest.raises(ValueError,match='schedule|initialization'):train.validate_config(noinit)
    changed=deepcopy(config);changed['posture_head']['softness_rad']=.001
    with pytest.raises(ValueError):train.validate_config(changed)


def v4_completed_fixture(tmp_path,monkeypatch):
    from test_ausc_act_auxiliary_train import completed_run
    complete=completed_run(tmp_path,monkeypatch)
    legacy=tmp_path/'legacy';legacy.mkdir()
    m,torch,init,receipt,Tiny,*_=initialization_fixture(legacy,monkeypatch)
    config=m.rt.read_bound(complete['inputs']['config'])
    config.update(schema=train.CONFIG_SCHEMA_V4,initialization=init,posture_head=receipt['posture_head'])
    config_ref=m.rt.write_new(tmp_path/'v4config.json',config)
    initial=dict(schema=m.EQUIVALENCE_SCHEMA,passed=True,initialization=init,initial_tensor_proofs=receipt['tensor_proofs'],
        prediction_max_abs_rad=0.,prediction_rtol=1e-5,prediction_atol=1e-6,prediction_device='cuda:0',probe_split='train',
        probe_indices=m.PROBE_TRAIN_INDICES,optimizer_step=0,optimizer_imported=False,scheduler_imported=False,final_linear_reset=False)
    initial_ref=m.rt.write_new(tmp_path/'INITIALIZATION_EQUIVALENCE.json',initial)
    optimizer=dict(name='AdamW',initial_step=0,initial_state_entries=0,lr=1e-4,weight_decay=1e-4,
        betas=[.9,.999],eps=1e-8,optimizer_imported=False,scheduler_imported=False)
    complete.update(schema=train.PILOT_SCHEMA_V4,schedule=config,protocol=train.training_protocol(config),
        initialization=dict(receipt=init,equivalence=initial_ref,optimizer=optimizer))
    complete['inputs'].update(config=config_ref,initialization=init,initialization_equivalence=initial_ref,code=train.code_binding(version=4))
    actual=complete['optimization_scope']['optimized_train_ids'];native=complete['native_scope']['train']['subject_ids']
    inventory={name:{name+':'+subject:dict(subject_id=subject,kind={'native':'native','planned':'planned_ideal_tracking',
        'conditional':'conditional_ideal_continuation','oracle':'student_oracle_unexecuted'}[name]) for subject in (native if name=='native' else actual)} for name in train.GROUPS_V3}
    def draws(steps):
        source={name:{key:steps*train.GROUPS_V3[name]['size']-len(items)+1 if i==0 else 1 for i,key in enumerate(items)} for name,items in inventory.items()}
        subjects={name:{inventory[name][key]['subject_id']:v for key,v in items.items()} for name,items in source.items()}
        kinds={name:{next(iter(items.values()))['kind']:steps*train.GROUPS_V3[name]['size']} for name,items in inventory.items()}
        return dict(gradient_examples={name:steps*g['size'] for name,g in train.GROUPS_V3.items()},gradient_subject_examples=subjects,
            gradient_source_examples=source,gradient_kind_examples=kinds,actual_optimized_train_ids=actual)
    complete.update(optimization=draws(20000),selected_optimization=draws(500))
    counts=dict(groups={name:dict(accepted_sources=len(items),samples=len(items),supervised_cells=len(items)) for name,items in inventory.items()},
        parent=dict(rows=35982,sources=934,attempted_slots=1109),new_source26=dict(accepted_sources=0,samples=0,attempted_slots=6,subject_ids=[]))
    complete['auxiliary_counts']=counts
    monkeypatch.setattr(train,'native_normalization',lambda config:receipt['normalization'])
    monkeypatch.setattr(train,'contribution_inventory',lambda config:inventory)
    monkeypatch.setattr(train,'auxiliary_metadata',lambda *a,**k:dict(counts=counts))
    keys=('schema','inputs','best','selected_step','scope','native_scope','optimization_scope','auxiliary_counts','oracle_authentication',
        'selected_optimization','optimization','protocol','selected_metrics','window_order','initialization')
    selection={k:deepcopy(complete[k]) for k in keys};selection.update(schema=train.SELECTION_SCHEMA_V4,**train.FLAGS)
    complete['cpu_equivalence']=m.rt.write_new(tmp_path/'v4cpu.json',dict(schema=train.CPU_SCHEMA_V4,passed=True,checkpoint=complete['best'],
        selected_step=500,scope=complete['scope'],optimization_scope=complete['optimization_scope'],initialization=complete['initialization'],
        selection=m.rt.write_new(tmp_path/'V4SELECTED.json',selection),comparisons={s:dict(passed=True,prediction_rtol=1e-5,prediction_atol=1e-6,
            metric_rtol=1e-5,metric_atol=1e-7) for s in ('train','validation')},**train.FLAGS))
    return m,torch,complete,receipt,initial_ref


def test_v4_complete_and_model_binding_require_initial_bit_proof_and_new_40_evaluations(tmp_path,monkeypatch):
    m,_,value,receipt,initial_ref=v4_completed_fixture(tmp_path,monkeypatch)
    assert train.validate_complete(value) is value
    ref=m.rt.write_new(tmp_path/'V4COMPLETE.json',value)
    assert train.model_binding(ref)[0]['checkpoint_schema']==train.PILOT_SCHEMA_V4
    for fault in ('old_evaluations','initial_tensors','initial_ref','optimizer','counts'):
        bad=deepcopy(value)
        if fault=='old_evaluations':bad['evaluations']+=deepcopy(bad['evaluations'])
        elif fault=='initial_tensors':
            proof=m.rt.read_bound(initial_ref);proof['initial_tensor_proofs']['weight']['sha256']='0'*64
            proof_ref=m.rt.write_new(tmp_path/'badproof.json',proof)
            bad['initialization']['equivalence']=proof_ref;bad['inputs']['initialization_equivalence']=proof_ref
        elif fault=='initial_ref':bad['inputs']['initialization']['sha256']='a'*64
        elif fault=='optimizer':bad['initialization']['optimizer']['initial_step']=19000
        else:bad['auxiliary_counts']['new_source26']['accepted_sources']=6
        with pytest.raises(ValueError):train.validate_complete(bad)


def test_v4_selected_loads_only_own_checkpoint_without_reinitializing(tmp_path,monkeypatch):
    m,torch,value,receipt,_=v4_completed_fixture(tmp_path,monkeypatch)
    normalization=receipt['normalization'];policy=SimpleNamespace(eval=lambda:None)
    loaded=[];policy.load_state_dict=lambda state,strict:loaded.append((state,strict))
    software={k:{} for k in ('vendor','dependencies','python','prefix','python_version','source_bindings')}
    value['inputs']['software']=m.rt.write_new(tmp_path/'v4software.json',software)
    packet=dict(value,schema=train.PILOT_SCHEMA_V4,step=value['selected_step'],state_dict={'own':'v4'},normalization=normalization,
        optimization=value['selected_optimization'])
    monkeypatch.setattr(train,'make_model',lambda *a,**k:(torch,policy,software))
    monkeypatch.setattr(train,'validate_official_forward',lambda *a:None)
    monkeypatch.setattr(train,'native_normalization',lambda config:normalization)
    monkeypatch.setattr(m,'initialize_policy',lambda *a,**k:pytest.fail('Selected inference warm-started again'))
    monkeypatch.setattr(m,'export',lambda *a,**k:pytest.fail('Selected inference executed original26'))
    paths=[]
    def load(source,**kwargs):
        paths.append(source)
        return packet
    monkeypatch.setattr(torch,'load',load)
    assert train.load_selected_policy(value)[2] is packet and len(paths)==1
    assert loaded==[({'own':'v4'},True)]
    packet['initialization']['optimizer']['initial_step']=19000
    with pytest.raises(ValueError):train.load_selected_policy(value)


def test_v4_buffered_native_audit_binds_model_specific_sources(tmp_path,monkeypatch):
    from test_ausc_given_contact_pilot_native_audit import setup
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep
    m,previous,c,*_=setup(tmp_path,monkeypatch);previous.close()
    model=dict(checkpoint_schema=train.PILOT_SCHEMA_V4,checkpoint={'sha256':'b'*64})
    descriptor=dict(model=model,source_config={'sha256':'c'*64})
    c.policy_receipt['model']=model
    expected={'v4':'actual28'};calls=[]
    monkeypatch.setattr(dep,'sources',lambda **kw:calls.append(kw) or expected if kw else {'v3':'legacy'})
    audit=m.BufferedNativeAudit(tmp_path/'v4.jsonl.gz',c,dep,{dep.ROUTE_KEY:descriptor})
    assert audit.sources==expected and calls==[{'model_schema':train.PILOT_SCHEMA_V4}]
    assert audit.close()['complete'] is True


def test_v4_code_rejects_relocated_or_relabelled_numerical_binding_but_legacy_hash_rule_unchanged():
    for version in (1,2,3):
        old=train.code_binding(version=version);old['code_commit']='historical-annotation'
        for ref in old['files'].values():ref['path']='/historical/'+ref['path'].split('/')[-1]
        train.validate_code(old,version=version)
    value=train.code_binding(version=4);train.validate_code(value,version=4)
    moved=deepcopy(value);next(iter(moved['files'].values()))['path']='/copied/numerical.py'
    with pytest.raises(ValueError,match='source|origin'):train.validate_code(moved,version=4)
    relabelled=deepcopy(value);relabelled['code_commit']='a'*40
    with pytest.raises(ValueError,match='source|origin'):train.validate_code(relabelled,version=4)


def test_v4_fit_exports_real_step0_proof_before_any_training_batch(tmp_path,monkeypatch):
    """CPU-only tensor fixture exercises fit ordering, then intentionally aborts."""
    from contextlib import nullcontext
    m,torch,init,value,Tiny,*_=initialization_fixture(tmp_path,monkeypatch)
    class LargeTiny(Tiny):
        def __init__(self):
            super().__init__(7,7)
            self.unused_backbone_fixture=torch.nn.Parameter(torch.zeros(2658823-56))
    original=LargeTiny();original.load_state_dict(torch.load(m.rt.bound_path(value['state_dict']),weights_only=True),strict=False)
    state_file=tmp_path/'LARGE_INITIAL_STATE.pt';torch.save(original.state_dict(),state_file)
    value['state_dict']=m.rt.reference(state_file);value['tensor_proofs']=m.tensor_proofs(original.state_dict())
    init=m.rt.write_new(tmp_path/'LARGE_INITIALIZATION.json',value)
    config=dict(train.SCHEDULE,schema=train.CONFIG_SCHEMA_V4,base='/fixture',dataset={},auxiliary={},evaluation_cohort={},
        posture_head=value['posture_head'],initialization=init)
    config_ref=m.rt.write_new(tmp_path/'fitconfig.json',config)
    prepared=dict(normalization=value['normalization'],scope={},optimization_scope={},native_scope={},
        auxiliary=dict(counts={},samples={}),oracle_authentication=[],posture_target_audit={})
    monkeypatch.setattr(train,'validate_config',lambda c:c);monkeypatch.setattr(train,'load_inputs',lambda c:prepared)
    monkeypatch.setattr(train.runtime,'owned_gpu',lambda:nullcontext({'fixture_only':True}))
    monkeypatch.setattr(train.runtime,'_identity',lambda:dict(pid=123,pgid=123,sid=123))
    model=LargeTiny();model._posture_head_diagnostics={}
    monkeypatch.setattr(train,'make_model',lambda *a,**k:(torch,model,{}))
    monkeypatch.setattr(train,'validate_official_forward',lambda s:None)
    monkeypatch.setattr(train.runtime,'execution_settings',lambda *a,**k:{'fixture_only':True})
    events=[];real_fresh=m.fresh_optimizer
    def fresh(*a,**kw):
        assert model._act6_initialization['passed'] is True
        optimizer,settings=real_fresh(*a,**kw);assert not optimizer.state
        events.append('fresh_optimizer_step0')
        return optimizer,settings
    monkeypatch.setattr(m,'fresh_optimizer',fresh)
    def stop(*a,**kw):
        events.append('first_batch_boundary')
        assert prepared['initialization']['optimizer']['initial_step']==0
        raise ValueError('deliberate CPU fixture boundary before numerical batches')
    monkeypatch.setattr(train,'build_cache',stop)
    output=tmp_path/'fit-output'
    with pytest.raises(ValueError,match='deliberate CPU fixture'):train.fit(config_ref,output)
    assert events==['fresh_optimizer_step0','first_batch_boundary']
    failed=m.rt.read_bound(m.rt.reference(output/'FAILED.json'))
    assert failed['completed_steps']==0
    packet=torch.load(m.rt.bound_path(failed['partial_checkpoint']),weights_only=True)
    assert packet['schema']==train.PILOT_SCHEMA_V4 and packet['step']==0
    assert packet['initialization']['receipt']==init and not packet['optimizer_state_dict']['state']
    proof=m.rt.read_bound(packet['initialization']['equivalence'])
    assert proof['initial_tensor_proofs']==value['tensor_proofs'] and proof['prediction_max_abs_rad']==0.


def test_initialization_metadata_validation_never_reads_any_tensor_artifact(tmp_path,monkeypatch):
    m,torch,ref,value,*_=initialization_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m.rt,'bound_path',lambda *a:pytest.fail('Metadata validation accessed old/exported tensor artifact'))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Metadata validation loaded tensors'))
    assert m.validate_receipt(ref,normalization=value['normalization'],posture_head=value['posture_head'])==value
