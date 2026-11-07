"""Finite fit preflight tests; CPU fixtures never qualify physical data."""
from copy import deepcopy
import importlib
from pathlib import Path

import pytest

from test_ausc_given_contact_runtime import module as runtime_module, batch
from test_ausc_given_contact_pose import reference


def module():
    name='sim.ausc_policy.m2_given_contact_train'
    assert importlib.util.find_spec(name) is not None, 'Formal finite trainer missing'
    return importlib.import_module(name)


def config_fixture(tmp_path,monkeypatch):
    r=runtime_module();m=module();value,stats=batch(tmp_path);ref=reference(tmp_path)
    import numpy as np
    value.update(origins=np.zeros((2,7)),observation_times=np.array([.05,.1]))
    prepared=dict(train=value,validation=deepcopy(value),training_eligible=True,test_payloads_opened=False,
        qualification=dict(passed=True),normalization=stats,task_contract=r.task_contract(),
        episodes=[dict(episode_id='one')],reference_inventory=[dict(episode_id='one',reference=ref.snapshot(),reference_sha256=ref.sha256)],
        derived_index=dict(samples=[dict(split='train',sample_index=i) for i in range(2)]+[
            dict(split='validation',sample_index=i) for i in range(2)]),preparer_sha256='a'*64,reconstruction_sources={})
    refs={key:r.write_new(tmp_path/('dataset-'+key+'.json'),{}) for key in ('plan','manifest','qualification')}
    refs['reconstruction']=r.write_new(tmp_path/'reconstruction.json',r.dataset._fingerprint(prepared))
    refs['seal']=r.write_new(tmp_path/'seal.json',dict(manifest=refs['manifest'],qualification=refs['qualification'],fingerprint=refs['reconstruction']))
    freeze=r.write_new(tmp_path/'freeze.json',{})
    calls=[]
    def verify(plan,seal):
        calls.append(('seal',plan,seal));return dict(passed=True,training_eligible=True,fingerprint=refs['reconstruction'])
    def reconstruct(path,sha,qualification):
        calls.append(('reconstruct',path,sha,qualification));return prepared
    monkeypatch.setattr(r.dataset,'verify_final_seal',verify)
    monkeypatch.setattr(r.dataset,'prepare_dataset',reconstruct)
    monkeypatch.setattr(r,'processing_audit',lambda binding:dict(descriptor=freeze,observed_at=str(len(calls))))
    monkeypatch.setattr(r,'verify_fixture',lambda ref:dict(passed=True,processing=dict(plan=refs['plan'],processing_freeze=freeze)))
    config=dict(schema=m.CONFIG_SCHEMA,lerobot_root=r.VENDOR_ROOT,device='cuda',task_contract=r.task_contract(),
        model=r.shared.native_spec(),numerics=r.numerical_protocol(),fixture_spec=r.fixture_spec(),dataset=refs,
        processing=dict(plan=refs['plan'],processing_freeze=freeze),
        runtime_qualification=r.write_new(tmp_path/'runtime-qualified.json',{}))
    monkeypatch.setattr(m,'_freeze_inputs',lambda config_ref,config,prepared:m._InputClosure({},[('config',config_ref,False)]))
    return m,r,config,prepared,calls


def test_fit_preflight_reopens_original_seal_and_new_dataset_reader(tmp_path,monkeypatch):
    m,r,config,prepared,calls=config_fixture(tmp_path,monkeypatch)
    config_ref=r.write_new(tmp_path/'config.json',config)
    result=m.preflight(config_ref)
    assert result['prepared'] is prepared and calls[0][0]=='seal' and calls[1][0]=='reconstruct'
    assert calls[1][1:]==(config['dataset']['manifest']['path'],config['dataset']['manifest']['sha256'],config['dataset']['qualification'])
    assert result['window_order']['train']!=result['window_order']['validation']


@pytest.mark.parametrize('fault',['unqualified','test_opened','changed_stats','changed_fingerprint','old_contract','fixture_root'])
def test_fit_preflight_rejects_qualification_source_or_split_drift(tmp_path,monkeypatch,fault):
    m,r,config,prepared,_=config_fixture(tmp_path,monkeypatch)
    if fault=='unqualified':prepared['training_eligible']=False
    elif fault=='test_opened':prepared['test_payloads_opened']=True
    elif fault=='changed_stats':prepared['normalization']['action_mean'][0]+=.1
    elif fault=='changed_fingerprint':prepared['train']['actions'][0,0,0]+=.1
    elif fault=='old_contract':prepared['task_contract']['history_steps']=1
    else:monkeypatch.setattr(r,'verify_fixture',lambda ref:dict(passed=True,processing={}))
    with pytest.raises(ValueError):m.preflight(r.write_new(tmp_path/'config.json',config))


@pytest.mark.parametrize('fault',['steps','batch','augmentation','resume','device','model','test','missing_seal'])
def test_formal_config_has_one_exact_schedule_without_resume_or_test(tmp_path,monkeypatch,fault):
    m,r,config,_,_=config_fixture(tmp_path,monkeypatch);m.validate_config(config)
    if fault=='steps':config['numerics']['steps']=19999
    elif fault=='batch':config['numerics']['batch_size']=65
    elif fault=='augmentation':config['numerics']['q_dropout']=.1
    elif fault=='resume':config['resume']='last.pt'
    elif fault=='device':config['device']='cpu'
    elif fault=='model':config['model']['dropout']=.05
    elif fault=='test':config['dataset']['test']={}
    else:config['dataset'].pop('seal')
    with pytest.raises(ValueError):m.validate_config(config)


def test_fit_cannot_construct_cuda_before_qualified_preflight(tmp_path,monkeypatch):
    m,r,config,prepared,_=config_fixture(tmp_path,monkeypatch);prepared['training_eligible']=False
    from contextlib import contextmanager
    @contextmanager
    def lock():yield {}
    monkeypatch.setattr(r,'owned_gpu',lock)
    config_ref=r.write_new(tmp_path/'config.json',config)
    monkeypatch.setattr(r,'start_owned_stage',lambda *a,**k:dict(input=config_ref))
    monkeypatch.setattr(r,'make_model',lambda *a,**k:pytest.fail('CUDA constructed before qualification'))
    with pytest.raises(ValueError):m.fit(config_ref,tmp_path/'out')
    assert not (tmp_path/'out/COMPLETE.json').exists()


def toy_loop_fixture(tmp_path,monkeypatch):
    import torch
    m,r,config,prepared,_=config_fixture(tmp_path,monkeypatch)
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.value=torch.nn.Parameter(torch.zeros(()))
        def forward(self,b):return ((self.value-b['action'])**2).mean(),{}
        def predict_action_chunk(self,b):return self.value.expand(len(b['observation.state']),20,7)
    inputs=dict(dataset=config['dataset'],config=r.write_new(tmp_path/'config.json',config),
        **{name:r.write_new(tmp_path/(name+'-input.json'),{}) for name in ('software','normalization','reference_inventory')})
    output=tmp_path/'loop';output.mkdir()
    return m,r,torch,Model(),prepared,inputs,output


def test_finite_loop_keeps_all_forty_snapshots_and_exact_terminal_optimizer_rng(tmp_path,monkeypatch):
    import io
    m,r,torch,policy,prepared,inputs,output=toy_loop_fixture(tmp_path,monkeypatch)
    checks=[]
    evaluations,last=m._optimize(torch,policy,prepared,inputs,output,m.window_order(prepared),
        lambda:checks.append(True),io.StringIO(),device='cpu')
    assert len(checks)==20000 and [row['step'] for row in evaluations]==r.EVAL_STEPS
    assert len(list(output.glob('step-*.pt')))==40 and not (output/'best.pt').exists()
    packet=torch.load(r.bound_path(last),map_location='cpu',weights_only=True)
    assert packet['step']==packet['training_state']['completed_steps']==20000
    assert float(packet['training_state']['optimizer']['state'][0]['step'])==20000
    assert packet['training_state']['rng']['minibatch']['bit_generator']=='PCG64'
    assert set(packet['training_state']['rng'])=={'minibatch','augmentation','python','numpy','torch_cpu','torch_cuda'}
    assert not (output/'COMPLETE.json').exists()


def test_partial_retains_actual_completed_step_without_formal_completion(tmp_path,monkeypatch):
    import io
    m,r,torch,policy,prepared,inputs,output=toy_loop_fixture(tmp_path,monkeypatch)
    count=[0]
    def abort():
        count[0]+=1
        if count[0]==11:raise ValueError('resource abort')
    with pytest.raises(ValueError,match='resource abort'):
        m._optimize(torch,policy,prepared,inputs,output,m.window_order(prepared),abort,io.StringIO(),device='cpu')
    partial=torch.load(output/'partial.pt',map_location='cpu',weights_only=True)
    assert partial['completed_steps']==10 and partial['formal_complete'] is False
    assert not (output/'last.pt').exists() and not (output/'COMPLETE.json').exists()


def test_committed_template_locks_schedule_and_cannot_run_without_actual_seal():
    import json
    m=module();r=runtime_module()
    config=json.loads((Path(m.__file__).parents[2]/'configs/ausc_policy/m2_given_contact_act_formal_v1.json').read_text())
    assert config['numerics']==r.numerical_protocol() and config['fixture_spec']==r.fixture_spec()
    assert config['model']['dropout']==.1 and config['numerics']['q_dropout']==.05
    with pytest.raises(ValueError):m.validate_config(config)


def closure_fixture(tmp_path):
    m=module();r=runtime_module();root=tmp_path/'sealed';root.mkdir()
    ref=r.write_new(root/'receipt.json',dict(source='original'))
    external=tmp_path/'runtime.so';external.write_bytes(b'installed runtime')
    return m,r,root,ref,external


@pytest.mark.parametrize('fault',['add','empty_dir','delete','bytes','replace_same_bytes','symlink','hardlink','declaration_same_sha'])
def test_fixed_input_closure_rejects_terminal_tree_or_role_mutation(tmp_path,fault):
    import os
    m,r,root,ref,external=closure_fixture(tmp_path)
    declaration=r.write_new(root/'declaration.json',dict(reference=r.reference(external)))
    closure=m._InputClosure({'sealed_metadata':[str(root)]},[('other_runtime',r.reference(external),True)])
    if fault=='add':(root/'new.json').write_text('{}')
    elif fault=='empty_dir':(root/'new').mkdir()
    elif fault=='delete':Path(ref['path']).unlink()
    elif fault=='bytes':Path(ref['path']).write_text('changed')
    elif fault=='replace_same_bytes':
        p=Path(ref['path']);replacement=root/'replacement';replacement.write_bytes(p.read_bytes());os.replace(replacement,p)
    elif fault=='symlink':Path(ref['path']).unlink();Path(ref['path']).symlink_to(external)
    elif fault=='hardlink':os.link(ref['path'],tmp_path/'outside-link')
    else:
        twin=tmp_path/'runtime-copy.so';twin.write_bytes(external.read_bytes())
        Path(declaration['path']).write_text(__import__('json').dumps(dict(reference=r.reference(twin))))
    with pytest.raises(ValueError):closure.verify()


def test_fixed_input_closure_preserves_existing_authenticated_runtime_hardlink(tmp_path):
    import os
    m,r,root,ref,external=closure_fixture(tmp_path);os.link(external,tmp_path/'installed-copy')
    closure=m._InputClosure({'metadata':[str(root)]},[('other_runtime',r.reference(external),True)])
    before=closure.verify();assert before['passed'] and before['file_count']==2
    os.link(external,tmp_path/'new-link')
    with pytest.raises(ValueError):closure.verify()


@pytest.mark.parametrize('role',['source','prepared','config','sdk_wrong_role'])
def test_readonly_inputs_do_not_broadly_accept_runtime_link_exception(tmp_path,role):
    import os
    m,r,root,ref,external=closure_fixture(tmp_path);os.link(external,tmp_path/'installed-copy')
    with pytest.raises(ValueError):m._InputClosure({},[(role,r.reference(external),False)])


@pytest.mark.parametrize('location',['inside','ancestor','samefile','symlink_parent'])
def test_current_outputs_cannot_overlap_readonly_inputs(tmp_path,location):
    m,r,root,ref,external=closure_fixture(tmp_path)
    closure=m._InputClosure({'metadata':[str(root)]},[('other_runtime',r.reference(external),True)])
    output=root/'new-output' if location=='inside' else tmp_path if location=='ancestor' else external
    if location=='symlink_parent':
        alias=tmp_path/'alias';alias.symlink_to(root,target_is_directory=True);output=alias/'new'
    with pytest.raises(ValueError):closure.assert_output(output)
    closure.assert_output(tmp_path/'independent-output')


def test_postflight_preserves_initial_two_decodes_and_rebinds_without_another_decode(tmp_path,monkeypatch):
    m,r,config,prepared,calls=config_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_freeze_inputs',lambda config_ref,config,prepared:m._InputClosure({},[('config',config_ref,False)]))
    pre=m.preflight(r.write_new(tmp_path/'config.json',config))
    before=deepcopy(calls);result=m.postflight(pre,tmp_path/'new-output')
    assert calls==before and len(calls)==2 and result['passed']
    assert result['fingerprint']==r.dataset._fingerprint(prepared)
    assert (tmp_path/'new-output/input_rebinding.json').is_file()


@pytest.mark.parametrize('fault',['array','order','stats','reference_inventory','preflight_baseline'])
def test_postflight_refuses_in_memory_arrays_order_stats_and_mutated_preflight_baseline(tmp_path,monkeypatch,fault):
    m,r,config,prepared,calls=config_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_freeze_inputs',lambda config_ref,config,prepared:m._InputClosure({},[('config',config_ref,False)]))
    pre=m.preflight(r.write_new(tmp_path/'config.json',config))
    if fault=='array':prepared['train']['actions'][0,0,0]+=.1
    elif fault=='order':prepared['derived_index']['samples'].reverse()
    elif fault=='stats':prepared['normalization']['action_mean'][0]+=.1
    elif fault=='reference_inventory':prepared['reference_inventory'][0]['episode_id']='another'
    else:pre['fingerprint']['arrays']['train']['actions']['sha256']='b'*64
    with pytest.raises(ValueError):m.postflight(pre,tmp_path/'new-output')
    assert len(calls)==2


def population_closure_fixture(tmp_path):
    """Tiny CPU files with the real finite role graph, never native qualification."""
    m=module();r=runtime_module()
    roots={name:tmp_path/name for name in ('metadata','configs','collector','final','archives','historical','smoke')}
    for root in roots.values():root.mkdir()
    def write(root,name,value):return r.write_new(roots[root]/name,value)
    (roots['collector']/'CODE_COMMIT').write_text('f'*40)
    source=tmp_path/'patient.json';source.write_text('{}')
    archived=write('configs','archived.json',{});source.write_bytes(Path(archived['path']).read_bytes())
    freeze=write('configs','source-freeze.json',dict(files={str(source):archived}))
    cfg=write('configs','config.json',dict(compact_scene_storage=dict(sources=dict(patient=dict(file=str(source),**{k:v for k,v in r.reference(source).items() if k!='path'})))))
    owner=write('smoke','result.json',{})
    (roots['smoke']/'frame.bin').write_bytes(b'original-capture-frame')
    artifacts={p.relative_to(roots['smoke']).as_posix():{k:v for k,v in r.reference(p).items() if k!='path'} for p in roots['smoke'].iterdir()}
    smoke=dict(config=cfg,source_freeze=freeze,owner_terminal=owner,capture_root=str(roots['smoke']),artifacts=artifacts)
    qualification=write('metadata','smokes.json',dict(episodes=[deepcopy(smoke) for _ in range(5)]))
    manifest=write('metadata','manifest.json',dict(episodes=[dict(split='train',**smoke)]))
    owners=[]
    for i in range(3):
        root=tmp_path/('owner'+str(i));root.mkdir();(root/'failed.log').write_text('complete outcome');owners.append(str(root))
    refs={key:write('metadata',key+'.json',{}) for key in ('allocation_proposal','allocation_mapping','source_recipe','previous_owner')}
    excluded=tmp_path/'whole_cpu_bank';excluded.mkdir();(excluded/'TEST-secret').write_text('never walk')
    plan=dict(attempts=[deepcopy(smoke) for _ in range(240)],qualification=qualification,
        config_inventory=write('configs','inventory.json',{}),phase_batches={},collector=dict(root=str(roots['collector'])),
        accounting_scope=dict(owner_roots=owners,operational_metadata=dict(root=str(roots['metadata'])),
            storage_roots=[dict(path=str(excluded))],extra_input_refs=[dict(category='prepared',reference=r.reference(source))],
            processing_sources=dict(live_root=str(roots['final']),archive_root=str(roots['archives']),
                descriptor_path=str(roots['metadata']/'descriptor.json'),original_execution_freeze=dict(root=str(roots['historical'])))),**refs)
    plan_ref=write('metadata','plan.json',plan)
    (roots['metadata']/'coverage.log').write_text('original-unreferenced-metadata')
    metadata_inventory={p.relative_to(roots['metadata']).as_posix():{k:v for k,v in r.reference(p).items() if k!='path'} for p in roots['metadata'].iterdir()}
    seal=r.write_new(tmp_path/'external-seal.json',dict(metadata_inventory=metadata_inventory))
    config=dict(dataset=dict(plan=plan_ref,manifest=manifest,seal=seal))
    config_ref=r.write_new(tmp_path/'training-config.json',config)
    prepared=dict(qualification=dict(accounting=dict(all_accounted_jobs_within_budget=True,
        unique_union=dict(shared_inputs=[dict(file=str(source),bytes=source.stat().st_size,sha256=r.reference(source)['sha256'])]),
        additional_inputs=[])))
    return m,r,config_ref,config,prepared,roots,excluded,source


def test_authenticated_population_closure_covers_all_finite_roles_and_excludes_storage_bank(tmp_path):
    m,r,ref,config,prepared,roots,excluded,source=population_closure_fixture(tmp_path)
    closure=m._freeze_inputs(ref,config,prepared);inventory=closure.inventory()
    assert all(str(root) in inventory['entries'] for root in roots.values())
    assert str(excluded) not in inventory['entries'] and str(source) in inventory['entries']
    roles=inventory['roles'][str(source)]
    assert any(row['role']=='original_source' for row in roles)
    (excluded/'unrelated-new-file').write_text('not a dataset input');assert closure.verify()['passed']
    (roots['historical']/'new.py').write_text('changed historical code')
    with pytest.raises(ValueError):closure.verify()


def test_completed_model_closure_binds_every_snapshot_prediction_and_current_outputs_separately(tmp_path):
    m=module();r=runtime_module();files=[]
    for i in range(40):
        path=tmp_path/('weight'+str(i));path.write_bytes(bytes([i]));files.append(r.reference(path))
    complete=dict(evaluations=[dict(checkpoint=ref) for ref in files],inputs={})
    model=dict(checkpoint=files[0],complete=r.write_new(tmp_path/'COMPLETE.json',complete))
    closure=m._freeze_model_inputs(model,complete)
    assert all(row['path'] in closure.inventory()['entries'] for row in files)
    closure.assert_output(tmp_path/'new-worker-output')
    Path(files[-1]['path']).write_bytes(b'changed-last-snapshot')
    with pytest.raises(ValueError):closure.verify()


def test_completed_model_freeze_reopens_original_owner_chain_before_inference(tmp_path,monkeypatch):
    m=module();r=runtime_module();owner=tmp_path/'training-owner';owner.mkdir()
    run=r.write_new(owner/'run.json',{});terminal=r.write_new(owner/'result.json',{})
    resources=r.write_new(owner/'resources.json',dict(identity=dict(pid=7)))
    complete=dict(owned_stage=dict(run=run),resources=resources)
    model=dict(complete=r.write_new(owner/'COMPLETE.json',complete))
    calls=[];monkeypatch.setattr(r,'verify_owned_stage',lambda *a,**k:calls.append((a,k)))
    closure=m._freeze_model_inputs(model,complete)
    assert calls and calls[0][0]==(terminal,) and calls[0][1]['child_identity']==dict(pid=7)
    assert closure.inventory()['roots']['completed_training_owner']==[str(owner)]
    (owner/'late-owner.log').write_text('must remain closed')
    with pytest.raises(ValueError):closure.verify()


@pytest.mark.parametrize('role',['sdk','other_runtime'])
def test_installed_link_exception_never_applies_inside_closed_source_or_capture_tree(tmp_path,role):
    import os
    m,r,root,ref,external=closure_fixture(tmp_path);os.link(ref['path'],tmp_path/'duplicate')
    with pytest.raises(ValueError):m._InputClosure({'closed_capture':[str(root)]},[(role,ref,True)])


def test_special_file_refused_without_opening_or_blocking(tmp_path):
    import os
    m,r,root,ref,external=closure_fixture(tmp_path);os.mkfifo(root/'pipe')
    with pytest.raises(ValueError):m._InputClosure({'metadata':[str(root)]},[])


@pytest.mark.parametrize('scope,fault',[(s,f) for s in ('capture','metadata') for f in
    ('bytes','add','missing','rename_same_sha','empty_directory')])
def test_population_baseline_authenticates_original_seal_and_capture_inventory_before_accepting_current_bytes(tmp_path,scope,fault):
    import os
    m,r,ref,config,prepared,roots,excluded,source=population_closure_fixture(tmp_path)
    root=roots['smoke'] if scope=='capture' else roots['metadata']
    candidate=root/('frame.bin' if scope=='capture' else 'coverage.log')
    if fault=='bytes':
        data=candidate.read_bytes();candidate.write_bytes(b'X'+data[1:])
    elif fault=='add':(root/'extra.json').write_text('{}')
    elif fault=='missing':candidate.unlink()
    elif fault=='rename_same_sha':os.rename(candidate,root/'renamed.json')
    else:(root/'new_empty_dir').mkdir()
    with pytest.raises(ValueError):m._freeze_inputs(ref,config,prepared)
