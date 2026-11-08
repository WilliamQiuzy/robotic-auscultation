"""Tensor-only V8 warm import. All tensors and provenance are synthetic."""
from copy import deepcopy
import importlib

import numpy as np
import pytest


def module():
    name='sim.ausc_policy.m2_act_diverse_warm_start'
    assert importlib.util.find_spec(name), 'Separate typed ACT9 to V8 initializer required'
    return importlib.import_module(name)


def test_warm_pins_actual_act9_and_truthful_original_source34_export():
    m=module()
    assert m.CONTROL['sha256']=='93ecf09940471ed62e23f757f9f885cf9f6efa08b6433a4deed4a84ae90bf6fe'
    assert m.WEIGHTS['bytes']==11278593 and m.WEIGHTS['sha256']=='7523475be10b1ad9edb1b2080513c748d7d3d0dcbc90b19fbefc5539ac27baec'
    assert m.COMPLETE['sha256']=='3be44244e867846db274a6f558d583377a8d8fe33d95846e06d8b81c7f2489f7'
    assert m.SOURCE34['code_commit']=='ac2bc918f733e25f419d73bc83894be7e4d01545'
    assert 'original.model_binding(' in m._PROGRAM and 'original.load_selected_policy(' in m._PROGRAM
    assert "m2_act_auxiliary_act_pilot_v7" in m._PROGRAM and 'weights_only=True' in m._PROGRAM
    assert '__file__=' not in m._PROGRAM and 'monkeypatch' not in m._PROGRAM


def fixture(tmp_path,monkeypatch):
    m=module();torch=pytest.importorskip('torch')
    from test_ausc_act_training_v4 import head_spec
    spec=head_spec();norm=spec['normalization']
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.values=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([i/100.])) for i in range(54)])
        def predict_action_chunk(self,batch):
            return self.values[0].reshape(1,1,1).expand(len(batch['observation.state']),20,7)
    policy=Policy();state={k:v.detach().clone() for k,v in policy.state_dict().items()}
    path=tmp_path/'state.pt';torch.save(state,path)
    observations={'observation.state':np.zeros((4,7)).tolist(),'observation.environment_state':np.zeros((4,27)).tolist()}
    raw=np.broadcast_to(np.asarray(norm['action_mean']),(4,20,7)).copy()
    value=dict(schema=m.SCHEMA,complete=True,origin=m.ORIGIN,inputs={},source34=deepcopy(m.SOURCE34),
        completed_steps=1000,selected_step=1000,parameter_count=2658823,model=m.rt.shared.native_spec(),task_contract=m.rt.task_contract(),
        normalization=norm,posture_head=spec,state_dict=m.rt.reference(path),tensor_proofs=m.tensor_proofs(state),
        optimizer_imported=False,scheduler_imported=False,probes=dict(split='train',dataset={},indices=m.PROBE_TRAIN_INDICES,
            observations=observations,raw_predictions=raw.tolist()),execution=dict(device='cpu',intraop_threads=1,interop_threads=1,cuda_visible_devices=''))
    ref=m.rt.write_new(tmp_path/'receipt.json',value)
    # Isolate the already tested provenance gate so numerical imports use small
    # synthetic tensors; no production receipt or loader can consume this fixture.
    monkeypatch.setattr(m,'validate_receipt',lambda *a,**k:deepcopy(value))
    return m,torch,policy,ref,value


def test_all54_tensors_bit_identical_probe_equivalence_fresh_adam0(tmp_path,monkeypatch):
    m,torch,policy,ref,value=fixture(tmp_path,monkeypatch)
    expected=deepcopy(value['tensor_proofs'])
    with torch.no_grad():
        for p in policy.parameters():p.fill_(19.)
    result=m.initialize_policy(torch,policy,ref,normalization=value['normalization'],posture_head=value['posture_head'])
    assert m.tensor_proofs(policy.state_dict())==expected and result['passed'] and result['optimizer_step']==0
    optimizer,receipt=m.fresh_optimizer(torch,policy,lr=1e-4,weight_decay=1e-4)
    assert not optimizer.state and receipt['initial_step']==0 and receipt['initial_state_entries']==0
    with pytest.raises(ValueError):m.initialize_policy(torch,policy,ref,normalization=value['normalization'],posture_head=value['posture_head'])


@pytest.mark.parametrize('fault',['hash','missing_tensor','prediction','prior_reset','optimizer'])
def test_import_fails_closed_for_any_tensor_probe_or_reset_mismatch(tmp_path,monkeypatch,fault):
    m,torch,policy,ref,value=fixture(tmp_path,monkeypatch)
    if fault=='hash':value['tensor_proofs']['values.0']['sha256']='f'*64
    elif fault=='missing_tensor':value['tensor_proofs'].pop('values.0')
    elif fault=='prediction':value['probes']['raw_predictions'][0][0][0]+=.01
    elif fault=='prior_reset':policy._posture_training_initialization={}
    else:
        with pytest.raises(ValueError):m.fresh_optimizer(torch,policy,lr=1e-4,weight_decay=1e-4)
        return
    with pytest.raises(ValueError):m.initialize_policy(torch,policy,ref,normalization=value['normalization'],posture_head=value['posture_head'])


def provenance_fixture(tmp_path,monkeypatch):
    m=module()
    from test_ausc_act_training_v4 import head_spec
    spec=head_spec();norm=spec['normalization'];normalization=m.rt.write_new(tmp_path/'norm-proof.json',norm)
    native=m.rt.write_new(tmp_path/'native-proof.json',dict(normalization=normalization))
    software_ref=dict(path='/synthetic/software-proof.json',bytes=1,sha256='a'*64)
    software=dict(vendor=dict(root=m.rt.VENDOR_ROOT,upstream_commit=m.rt.UPSTREAM_COMMIT,files={
        'src/lerobot/policies/act/modeling_act.py':dict(sha256='83d954f79ccd3eaa6774dbea92524939c8ac08359bb355ba7cd37b8601e1e3af')}),
        prefix=m.rt.VENDOR_ROOT+'/.venv',source_bindings={m.SOURCE34['root']+'/sim/ausc_policy/fixture.py':{}})
    inputs=dict(dataset=native,software=software_ref,code=dict(code_commit=m.SOURCE34['code_commit'],files={
        'fixture.py':dict(path=m.SOURCE34['root']+'/sim/ausc_policy/fixture.py',bytes=1,sha256='a'*64)}))
    complete=dict(schema='m2_act_auxiliary_act_pilot_v7',complete=True,completed_steps=1000,selected_step=1000,best=m.WEIGHTS,
        cpu_equivalence=m.CPU_EQUIVALENCE,inputs=inputs,protocol=dict(posture_head=spec))
    selected=dict(complete,schema='m2_act_auxiliary_act_selection_v7')
    cpu=dict(schema='m2_act_auxiliary_act_cpu_equivalence_v7',passed=True,checkpoint=m.WEIGHTS,selection=m.SELECTED,selected_step=1000)
    control=dict(schema='act9_single_owned_training_execution_v1',complete=True,normal_owner=True,training=m.COMPLETE,training_owner=m.OWNER,
        model=dict(checkpoint=m.WEIGHTS,checkpoint_schema='m2_act_auxiliary_act_pilot_v7',complete=m.COMPLETE),cpu_equivalence=m.CPU_EQUIVALENCE,
        completed_steps=1000,selected_step=1000,source=dict(code_commit=m.SOURCE34['code_commit'],files=m.SOURCE34['manifest_files'],
        manifest=m.SOURCE34['source_manifest'],root=m.SOURCE34['root']))
    owner=dict(run_id='act9-measured-train-20261107-v1',code_commit=m.SOURCE34['code_commit'],completed=True,gpu_released=True,
        stage=dict(stage='m2_given_contact_expanded_pilot_train',passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[]))
    graph={ref['path']:value for ref,value in ((m.CONTROL,control),(m.COMPLETE,complete),(m.SELECTED,selected),(m.CPU_EQUIVALENCE,cpu),(m.OWNER,owner),(software_ref,software))}
    read=m.rt.read_bound
    monkeypatch.setattr(m.rt,'read_bound',lambda ref:deepcopy(graph[ref['path']]) if ref['path'] in graph else read(ref))
    receipt=dict(schema=m.SCHEMA,complete=True,origin=m.ORIGIN,source34=deepcopy(m.SOURCE34),completed_steps=1000,selected_step=1000,
        parameter_count=2658823,model=m.rt.shared.native_spec(),task_contract=m.rt.task_contract(),normalization=norm,posture_head=spec,
        optimizer_imported=False,scheduler_imported=False,inputs=dict(control=m.CONTROL,complete=m.COMPLETE,weights=m.WEIGHTS,cpu_equivalence=m.CPU_EQUIVALENCE,
        owner=m.OWNER,selection=m.SELECTED,software=software_ref,normalization=normalization,source_manifest=m.SOURCE34['source_manifest']),
        state_dict=dict(path='/synthetic/tensor-only.pt',bytes=100,sha256='e'*64),tensor_proofs={str(i):dict(shape=[1],dtype='torch.float32',bytes=4,sha256='f'*64) for i in range(54)},
        probes=dict(split='train',dataset=native,indices=m.PROBE_TRAIN_INDICES,observations={'observation.state':np.zeros((4,7)).tolist(),
        'observation.environment_state':np.zeros((4,27)).tolist()},raw_predictions=np.zeros((4,20,7)).tolist()),
        execution=dict(device='cpu',intraop_threads=1,interop_threads=1,cuda_visible_devices=''))
    return m,receipt,graph


def test_receipt_metadata_authenticates_exact_act9_graph_without_loading_tensors(tmp_path,monkeypatch):
    m,receipt,graph=provenance_fixture(tmp_path,monkeypatch);torch=pytest.importorskip('torch')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Metadata admission loaded weights'))
    ref=m.rt.write_new(tmp_path/'admitted.json',receipt)
    assert m.validate_receipt(ref)==receipt
    for fault in ('control','cpu','owner','source','norm','probes','optimizer','tensor_count','head'):
        bad=deepcopy(receipt);saved=deepcopy(graph)
        if fault=='control':graph[m.CONTROL['path']]['model']['checkpoint']=dict(m.WEIGHTS,sha256='0'*64)
        elif fault=='cpu':graph[m.CPU_EQUIVALENCE['path']]['selected_step']=500
        elif fault=='owner':graph[m.OWNER['path']]['stage']['signals']=['SIGTERM']
        elif fault=='source':graph[m.SELECTED['path']]['inputs']['code']['files']['fixture.py']['path']='/foreign/fixture.py'
        elif fault=='norm':bad['normalization']['action_mean'][0]+=.01
        elif fault=='probes':bad['probes']['split']='validation'
        elif fault=='optimizer':bad['optimizer_imported']=True
        elif fault=='tensor_count':bad['tensor_proofs'].pop('53')
        else:bad['posture_head']['schema']='m2_act_posture_head_v1'
        with pytest.raises(ValueError):m.validate_receipt(m.rt.write_new(tmp_path/('bad-'+fault+'.json'),bad))
        graph.clear();graph.update(saved)
