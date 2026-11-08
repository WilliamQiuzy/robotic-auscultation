"""V9 warm/collection admission uses invented CPU tensors and provenance only."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def module():
    name='sim.ausc_policy.m2_act_refinement_warm_start'
    assert importlib.util.find_spec(name), 'Separate authenticated ACT10 to V9 initializer required'
    return importlib.import_module(name)


def test_pins_exact_selected17000_original41_and_original_official_child():
    m=module()
    expected={
        'CONTROL':(2757,'68422ea72f34087ee6c19c78a034e03fded828ab130f29ff3510cdcf5318247c'),
        'COMPLETE':(4016445,'f54c68cf03c883610617cf9beaac0a8d8f20d59d473e1458b07ded031730ae5e'),
        'WEIGHTS':(11805569,'9de4ffff75bdf01f3855a872bb0315256ee23c240a6c5c3d37221e4cb35c764d'),
        'SELECTED':(3752763,'55b09e463e8101771cea8cfe819f200e238dd537574f7b60ef50a2aaf15f3146'),
        'CPU_EQUIVALENCE':(15458,'5909b1e21ba17b201792012951f10b24f21d60a722e26e1264f2528c4fe0fdb2'),
        'OWNER':(8471,'20d8c9c2c919f85507db8598b39da0d3f6f8513361df747966dcb24e07227195')}
    assert all((getattr(m,k)['bytes'],getattr(m,k)['sha256'])==v for k,v in expected.items())
    assert m.WEIGHTS['path'].endswith('/best_auxiliary_act.pt')
    assert m.SOURCE41['root']=='/opt/wam-runtime/code/wam-act-recovery-20261106-v41'
    assert m.SOURCE41['code_commit']=='c6590db44ff94448a7da34488b5ceaa7810f1134'
    assert m.SOURCE41['manifest_files']==1706
    assert m.SOURCE41['source_manifest']['bytes']==295288
    assert m.SOURCE41['source_manifest']['sha256']=='f11af1049afabcd7c1d17bf724b86f6d3157897b01ae4a6c73db749bfebdd1ff'
    assert m.SCHEMA.endswith('_v9') and m.EQUIVALENCE_SCHEMA.endswith('_v9')
    assert 'original.model_binding(' in m._PROGRAM and 'original.load_selected_policy(' in m._PROGRAM
    assert 'm2_act_auxiliary_act_pilot_v8' in m._PROGRAM and 'weights_only=True' in m._PROGRAM
    assert '__file__=' not in m._PROGRAM and 'monkeypatch' not in m._PROGRAM


def provenance_fixture(tmp_path,monkeypatch):
    m=module()
    from test_ausc_act_training_v4 import head_spec
    spec=head_spec();norm=spec['normalization'];norm_ref=m.rt.write_new(tmp_path/'norm.json',norm)
    native=m.rt.write_new(tmp_path/'native.json',dict(normalization=norm_ref))
    software_ref=dict(path='/invented/software.json',bytes=1,sha256='a'*64)
    software=dict(vendor=dict(root=m.rt.VENDOR_ROOT,upstream_commit=m.rt.UPSTREAM_COMMIT,files={
        'src/lerobot/policies/act/modeling_act.py':dict(sha256='83d954f79ccd3eaa6774dbea92524939c8ac08359bb355ba7cd37b8601e1e3af')}),
        prefix=m.rt.VENDOR_ROOT+'/.venv',source_bindings={m.SOURCE41['root']+'/sim/ausc_policy/invented.py':'b'*64})
    inputs=dict(dataset=native,software=software_ref,code=dict(code_commit=m.SOURCE41['code_commit'],files={
        'invented.py':dict(path=m.SOURCE41['root']+'/sim/ausc_policy/invented.py',bytes=1,sha256='b'*64)}))
    candidates=[dict(step=i) for i in range(500,20001,500)]
    complete=dict(schema='m2_act_auxiliary_act_pilot_v8',complete=True,completed_steps=20000,selected_step=17000,best=m.WEIGHTS,
        cpu_equivalence=m.CPU_EQUIVALENCE,inputs=inputs,protocol=dict(posture_head=spec),data_version=5,model_optimizer_version=8,
        evaluation_candidates=candidates,optimization=dict(total_steps=20000,draws=640000),selected_optimization=dict(total_steps=17000,draws=544000))
    selected=dict(complete,schema='m2_act_auxiliary_act_selection_v8')
    cpu=dict(schema='m2_act_auxiliary_act_cpu_equivalence_v8',passed=True,checkpoint=m.WEIGHTS,selection=m.SELECTED,selected_step=17000,
        data_version=5,model_optimizer_version=8)
    source=dict(root=m.SOURCE41['root'],code_commit=m.SOURCE41['code_commit'],files=m.SOURCE41['manifest_files'],manifest=m.SOURCE41['source_manifest'])
    control=dict(schema='act10_diverse_v8_single_owned_training_execution_v1',complete=True,normal_owner=True,training=m.COMPLETE,training_owner=m.OWNER,
        model=dict(checkpoint=m.WEIGHTS,checkpoint_schema='m2_act_auxiliary_act_pilot_v8',complete=m.COMPLETE),cpu_equivalence=m.CPU_EQUIVALENCE,
        selection=m.SELECTED,completed_steps=20000,selected_step=17000,evaluation_candidates=40,source=source)
    owner=dict(run_id='act10-diverse-v8-train-20261107-v1',code_commit=m.SOURCE41['code_commit'],completed=True,gpu_released=True,
        stage=dict(stage='m2_given_contact_expanded_pilot_train',passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[]))
    graph={r['path']:v for r,v in ((m.CONTROL,control),(m.COMPLETE,complete),(m.SELECTED,selected),(m.CPU_EQUIVALENCE,cpu),(m.OWNER,owner),(software_ref,software))}
    read=m.rt.read_bound
    monkeypatch.setattr(m.rt,'read_bound',lambda ref:deepcopy(graph[ref['path']]) if ref['path'] in graph else read(ref))
    receipt=dict(schema=m.SCHEMA,complete=True,origin=m.ORIGIN,source41=deepcopy(m.SOURCE41),completed_steps=20000,selected_step=17000,
        parent_model=dict(checkpoint_schema='m2_act_auxiliary_act_pilot_v8',data_version=5,model_optimizer_version=8,evaluation_candidates=40),
        historical_optimization=deepcopy(dict(completed=complete['optimization'],selected=selected['selected_optimization'])),
        parameter_count=2658823,model=m.rt.shared.native_spec(),task_contract=m.rt.task_contract(),normalization=norm,posture_head=spec,
        optimizer_imported=False,scheduler_imported=False,inputs=dict(control=m.CONTROL,complete=m.COMPLETE,weights=m.WEIGHTS,cpu_equivalence=m.CPU_EQUIVALENCE,
            owner=m.OWNER,selection=m.SELECTED,software=software_ref,normalization=norm_ref,source_manifest=m.SOURCE41['source_manifest']),
        state_dict=dict(path='/invented/tensor-only.pt',bytes=100,sha256='e'*64),
        tensor_proofs={str(i):dict(shape=[1],dtype='torch.float32',bytes=4,sha256='f'*64) for i in range(54)},
        probes=dict(split='train',dataset=native,indices=m.PROBE_TRAIN_INDICES,observations={'observation.state':np.zeros((4,7)).tolist(),
            'observation.environment_state':np.zeros((4,27)).tolist()},raw_predictions=np.zeros((4,20,7)).tolist()),
        execution=dict(device='cpu',intraop_threads=1,interop_threads=1,cuda_visible_devices=''))
    return m,receipt,graph


def test_metadata_only_admission_preserves_v8_counts_and_excludes_parent_evaluations(tmp_path,monkeypatch):
    m,value,graph=provenance_fixture(tmp_path,monkeypatch);torch=pytest.importorskip('torch')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Metadata admission loaded tensors'))
    ref=m.rt.write_new(tmp_path/'receipt.json',value)
    assert m.validate_receipt(ref,normalization=value['normalization'],posture_head=value['posture_head'])==value
    protocol=m.protocol(ref)
    assert protocol['schema']==m.SCHEMA and protocol['final_linear_reset'] is False
    assert protocol['optimizer_imported'] is False and protocol['optimizer_initial_step']==0
    assert protocol['initial_tensor_proofs']==value['tensor_proofs']
    assert 'excluded from V9 evaluations' in protocol['original_evaluations']
    assert m.validate_receipt(ref)['historical_optimization']==dict(completed=dict(total_steps=20000,draws=640000),selected=dict(total_steps=17000,draws=544000))


@pytest.mark.parametrize('fault',['control','final20000','missing_evaluation','cpu','owner','owner_drain','source_alias','optimizer','scheduler',
    'normalization','head','head_norm','probe_split','probe_indices','probe_nan','tensor_count','parent_version','historical_counts','requested_norm','requested_head'])
def test_metadata_rejects_wrong_parent_probes_head_norm_optimizer_and_counts(tmp_path,monkeypatch,fault):
    m,value,graph=provenance_fixture(tmp_path,monkeypatch);kwargs={}
    if fault=='control':graph[m.CONTROL['path']]['model']['checkpoint']=dict(m.WEIGHTS,sha256='0'*64)
    elif fault=='final20000':graph[m.SELECTED['path']]['selected_step']=20000
    elif fault=='missing_evaluation':graph[m.COMPLETE['path']]['evaluation_candidates'].pop()
    elif fault=='cpu':graph[m.CPU_EQUIVALENCE['path']]['passed']=False
    elif fault=='owner':graph[m.OWNER['path']]['stage']['signals']=['SIGTERM']
    elif fault=='owner_drain':graph[m.OWNER['path']]['stage']['process_group_drained']=False
    elif fault=='source_alias':graph[m.SELECTED['path']]['inputs']['code']['files']['invented.py']['path']='/foreign/invented.py'
    elif fault=='optimizer':value['optimizer_imported']=True
    elif fault=='scheduler':value['scheduler_imported']=True
    elif fault=='normalization':value['normalization']['action_mean'][0]+=.01
    elif fault=='head':value['posture_head']['schema']='m2_act_smooth_posture_head_v1'
    elif fault=='head_norm':value['posture_head']['normalization']['action_std'][0]+=.01
    elif fault=='probe_split':value['probes']['split']='validation'
    elif fault=='probe_indices':value['probes']['indices']=[0,1,2,3]
    elif fault=='probe_nan':value['probes']['observations']['observation.state'][0][0]=float('nan')
    elif fault=='tensor_count':value['tensor_proofs'].pop('53')
    elif fault=='parent_version':value['parent_model']['model_optimizer_version']=9
    elif fault=='historical_counts':value['historical_optimization']['selected']['draws']+=1
    elif fault=='requested_norm':kwargs['normalization']=dict(value['normalization'],action_std=[9.]*7)
    else:kwargs['posture_head']=dict(value['posture_head'],schema='foreign')
    # NaN is deliberately supplied through the bound reader, because write_new
    # itself correctly refuses non-JSON-finite artifacts.
    ref=dict(path='/invented/bad.json',bytes=1,sha256='9'*64);graph[ref['path']]=value
    with pytest.raises(ValueError):m.validate_receipt(ref,**kwargs)


def numerical_fixture(tmp_path,monkeypatch):
    m,value,graph=provenance_fixture(tmp_path,monkeypatch);torch=pytest.importorskip('torch')
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.values=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([i/100.])) for i in range(54)])
            self._posture_head_spec=deepcopy(value['posture_head'])
        def predict_action_chunk(self,batch):
            return self.values[0].reshape(1,1,1).expand(len(batch['observation.state']),20,7)
    policy=Policy();path=tmp_path/'state.pt';torch.save(policy.state_dict(),path)
    value['state_dict']=m.rt.reference(path);value['tensor_proofs']=m.tensor_proofs(policy.state_dict())
    value['probes']['raw_predictions']=np.broadcast_to(np.asarray(value['normalization']['action_mean']),(4,20,7)).tolist()
    ref=m.rt.write_new(tmp_path/'initialization.json',value)
    return m,torch,policy,ref,value,graph,Policy


def test_all54_bit_identical_tensor_shapes_dtypes_probes_and_fresh_adamw0(tmp_path,monkeypatch):
    m,torch,policy,ref,value,_,_=numerical_fixture(tmp_path,monkeypatch)
    with torch.no_grad():
        for p in policy.parameters():p.fill_(19.)
    result=m.initialize_policy(torch,policy,ref,normalization=value['normalization'],posture_head=value['posture_head'])
    assert result['schema']==m.EQUIVALENCE_SCHEMA and result['passed'] is True
    assert m.tensor_proofs(policy.state_dict())==value['tensor_proofs']==result['initial_tensor_proofs']
    assert result['prediction_rtol']==1e-5 and result['prediction_atol']==1e-6 and result['prediction_max_abs_rad']==0.
    optimizer,receipt=m.fresh_optimizer(torch,policy,lr=1e-4,weight_decay=1e-4)
    assert not optimizer.state and receipt==dict(name='AdamW',initial_step=0,initial_state_entries=0,lr=1e-4,weight_decay=1e-4,
        betas=[.9,.999],eps=1e-8,optimizer_imported=False,scheduler_imported=False)
    assert optimizer.defaults['betas']==(.9,.999) and optimizer.defaults['eps']==1e-8
    with pytest.raises(ValueError):m.initialize_policy(torch,policy,ref,normalization=value['normalization'],posture_head=value['posture_head'])


@pytest.mark.parametrize('fault',['hash','missing','changed','shape','dtype','optimizer_packet','prediction','prior_reset','prior_v6','prior_v8','missing_head','wrong_installed_head','optimizer_before','wrong_lr','wrong_decay'])
def test_import_rejects_any_tensor_or_probe_change_and_never_imports_optimizer(tmp_path,monkeypatch,fault):
    m,torch,policy,ref,value,graph,_=numerical_fixture(tmp_path,monkeypatch)
    if fault.startswith('optimizer_before'):
        with pytest.raises(ValueError):m.fresh_optimizer(torch,policy,lr=1e-4,weight_decay=1e-4)
        return
    if fault in ('wrong_lr','wrong_decay'):
        m.initialize_policy(torch,policy,ref,normalization=value['normalization'],posture_head=value['posture_head'])
        with pytest.raises(ValueError):m.fresh_optimizer(torch,policy,lr=2e-4 if fault=='wrong_lr' else 1e-4,weight_decay=2e-4 if fault=='wrong_decay' else 1e-4)
        return
    state=torch.load(m.rt.bound_path(value['state_dict']),weights_only=True)
    if fault=='hash':value['tensor_proofs']['values.0']['sha256']='0'*64
    elif fault=='missing':state.pop('values.0')
    elif fault=='changed':state['values.0']+=1
    elif fault=='shape':state['values.0']=torch.zeros(2);value['tensor_proofs']=m.tensor_proofs(state)
    elif fault=='dtype':state['values.0']=state['values.0'].double();value['tensor_proofs']=m.tensor_proofs(state)
    elif fault=='optimizer_packet':state={'state_dict':state,'optimizer':{}}
    elif fault=='prediction':value['probes']['raw_predictions'][0][0][0]+=.01
    elif fault=='prior_reset':policy._posture_training_initialization={}
    elif fault=='prior_v6':policy._act6_initialization={}
    elif fault=='prior_v8':policy._act8_initialization={}
    elif fault=='missing_head':del policy._posture_head_spec
    else:policy._posture_head_spec['inset_rad']+=.01
    path=tmp_path/'changed.pt';torch.save(state,path);value['state_dict']=m.rt.reference(path)
    ref=m.rt.write_new(tmp_path/'changed.json',value)
    with pytest.raises(ValueError):m.initialize_policy(torch,policy,ref,normalization=value['normalization'],posture_head=value['posture_head'])


def test_collection_uses_same_kernel_one_original_head_and_cannot_create_optimizer_or_enter_model_admission(tmp_path,monkeypatch):
    m,torch,_,ref,value,_,Policy=numerical_fixture(tmp_path,monkeypatch);policy=Policy();del policy._posture_head_spec
    calls=[];software=dict(source_bindings={'/invented/current42/runtime.py':'c'*64})
    def make_model(config,*,device):
        assert config==dict(lerobot_root=m.rt.VENDOR_ROOT) and device=='cpu';calls.append('factory');return torch,policy,software
    def install(p,spec):
        assert p is policy and spec==value['posture_head'];calls.append('head');p._posture_head_spec=deepcopy(spec)
    monkeypatch.setattr(m.rt,'make_model',make_model);monkeypatch.setattr(m.head,'install',install)
    monkeypatch.setattr(m.head,'initialize_training',lambda *a:pytest.fail('Collection reset original head'))
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('Collection constructed an optimizer'))
    actual_torch,actual,packet=m.make_collection_policy(ref)
    assert actual_torch is torch and actual is policy and calls==['factory','head']
    assert packet['schema']==m.COLLECTION_SCHEMA and packet['collection_only'] is True and packet['benchmark_model'] is False and packet['optimizer_created'] is False
    assert packet['normalization']==value['normalization'] and packet['tensor_proofs']==value['tensor_proofs']
    assert packet['source_evidence']['parent']==m.SOURCE41 and packet['source_evidence']['current']['software']==software
    assert m.rt.bound_path(packet['source_evidence']['current']['helper'])==Path(m.__file__).resolve()
    assert not hasattr(policy,'_act9_initialization')
    assert packet['parent_model']==value['parent_model'] and packet['historical_optimization']==value['historical_optimization']
    with pytest.raises(ValueError):m.fresh_optimizer(torch,policy,lr=1e-4,weight_decay=1e-4)
    with pytest.raises(ValueError):m.initialize_policy(torch,policy,ref,normalization=value['normalization'],posture_head=value['posture_head'])
    from sim.ausc_policy import m2_act_auxiliary_train as train,m2_given_contact_expanded_pilot_deployment as deployment
    packet_ref=m.rt.write_new(tmp_path/'collection.json',packet)
    for admit in (train.model_binding,deployment.model_binding):
        with pytest.raises(ValueError):admit(packet_ref)


def archive_fixture(tmp_path,monkeypatch):
    m=module();root=tmp_path/'source41';root.mkdir();code=root/'sim';code.mkdir();(code/'invented.py').write_text('value=41\n')
    commit='c'*40;(root/'CODE_COMMIT').write_text(commit+'\n')
    manifest=m.rt.write_new(root/'SOURCE_MANIFEST.json',[dict(file='sim/invented.py',**{k:v for k,v in m.rt.reference(code/'invented.py').items() if k!='path'})])
    source=dict(root=str(root),code_commit=commit,manifest_files=1,source_manifest=manifest,code_commit_file=m.rt.reference(root/'CODE_COMMIT'))
    monkeypatch.setattr(m,'SOURCE41',source)
    return m,root


@pytest.mark.parametrize('fault',['changed','unlisted','missing','symlink','hardlink','commit_alias','manifest_alias','directory_alias'])
def test_archive_authenticates_every_file_exact_inventory_and_rejects_aliases(tmp_path,monkeypatch,fault):
    m,root=archive_fixture(tmp_path,monkeypatch);m.require_archive();path=root/'sim/invented.py'
    if fault=='changed':path.write_text('changed\n')
    elif fault=='unlisted':(root/'extra.py').write_text('extra\n')
    elif fault=='missing':path.unlink()
    elif fault=='symlink':other=tmp_path/'alias.py';path.rename(other);path.symlink_to(other)
    elif fault=='hardlink':(root/'extra.py').hardlink_to(path)
    elif fault=='commit_alias':other=tmp_path/'commit';(root/'CODE_COMMIT').rename(other);(root/'CODE_COMMIT').symlink_to(other)
    elif fault=='manifest_alias':other=tmp_path/'manifest';(root/'SOURCE_MANIFEST.json').rename(other);(root/'SOURCE_MANIFEST.json').symlink_to(other)
    else:other=tmp_path/'directory';(root/'sim').rename(other);(root/'sim').symlink_to(other,target_is_directory=True)
    with pytest.raises((ValueError,FileNotFoundError)):m.require_archive()


def test_export_reauthenticates_original_source_and_parent_before_and_after_isolated_child(tmp_path,monkeypatch):
    m,value,_=provenance_fixture(tmp_path,monkeypatch);events=[];output=tmp_path/'export'
    monkeypatch.setattr(m.sys,'platform','linux');monkeypatch.setenv('PYTHONPATH','/must/not/inherit')
    monkeypatch.setattr(m,'require_archive',lambda:events.append('archive'))
    graph=m._parent_graph
    def parent():events.append('graph');return graph()
    monkeypatch.setattr(m,'_parent_graph',parent)
    def child(args,**kwargs):
        events.append('child');assert args==[m.OFFICIAL_PYTHON,'-I','-B','-c',m._PROGRAM,m.SOURCE41['root']]
        assert kwargs['cwd']==m.SOURCE41['root'] and kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and 'PYTHONPATH' not in kwargs['env']
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE']=='1' and kwargs['env']['PYTHONNOUSERSITE']=='1'
        assert json.loads(kwargs['input'])['weights']==m.WEIGHTS
        path=output/'INITIAL_STATE.pt';path.write_bytes(b'invented tensor bytes');value['state_dict']=m.rt.reference(path)
        return SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr='')
    monkeypatch.setattr(m.subprocess,'run',child)
    assert m.export(output)==m.rt.reference(output/'INITIALIZATION.json')
    assert events[:5]==['archive','graph','child','archive','graph']
    with pytest.raises(ValueError):m.export(output)


def test_failed_child_still_reauthenticates_original_archive_and_cannot_publish_receipt(tmp_path,monkeypatch):
    m=module();events=[];output=tmp_path/'failed-export'
    monkeypatch.setattr(m.sys,'platform','linux')
    monkeypatch.setattr(m,'require_archive',lambda:events.append('archive'))
    monkeypatch.setattr(m,'_parent_graph',lambda:events.append('graph'))
    def child(*a,**k):events.append('child');return SimpleNamespace(returncode=1,stdout='',stderr='invented failure')
    monkeypatch.setattr(m.subprocess,'run',child)
    with pytest.raises(ValueError,match='export failed'):m.export(output)
    assert events==['archive','graph','child','archive','graph']
    assert not (output/'INITIALIZATION.json').exists()


def test_official_cpu_factory_random_tensors_when_available(tmp_path,monkeypatch):
    m=module()
    if Path(sys.prefix).resolve()!=(Path(m.rt.VENDOR_ROOT)/'.venv').resolve():
        pytest.skip('Pinned official LeRobot environment unavailable; no substitute factory')
    import torch
    from test_ausc_act_training_v4 import head_spec
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');spec=head_spec()
    with monkeypatch.context() as isolated:
        isolated.setattr(sys,'path',[p for p in sys.path if '/test-deps' not in p])
        _,policy,_=m.rt.make_model(dict(lerobot_root=m.rt.VENDOR_ROOT),device='cpu')
    m.head.install(policy,spec)
    assert len(policy.state_dict())==54 and sum(p.numel() for p in policy.parameters())==2658823
    observations={'observation.state':torch.zeros(4,7),'observation.environment_state':torch.zeros(4,27)}
    policy.eval()
    with torch.no_grad():raw=policy.predict_action_chunk(observations).numpy().astype('float64')
    raw=raw*np.asarray(spec['normalization']['action_std'])+np.asarray(spec['normalization']['action_mean'])
    state=tmp_path/'random-official.pt';torch.save(policy.state_dict(),state)
    value=dict(normalization=spec['normalization'],posture_head=spec,state_dict=m.rt.reference(state),tensor_proofs=m.tensor_proofs(policy.state_dict()),
        probes=dict(observations={k:v.tolist() for k,v in observations.items()},raw_predictions=raw.tolist()))
    monkeypatch.setattr(m,'validate_receipt',lambda *a,**k:deepcopy(value))
    result=m.initialize_policy(torch,policy,{},normalization=value['normalization'],posture_head=spec)
    assert result['passed'] and result['initial_tensor_proofs']==value['tensor_proofs']
