"""Formal runtime CPU contracts; these fixtures never qualify native data."""
from copy import deepcopy
import importlib
from pathlib import Path
import subprocess

import numpy as np
import pytest

from sim.ausc_policy import m2_given_contact_act as given, m2_act as shared
from test_ausc_given_contact_act import samples
from test_ausc_given_contact_pose import reference, metadata


def module():
    name='sim.ausc_policy.m2_given_contact_runtime'
    assert importlib.util.find_spec(name) is not None, 'Formal official runtime missing'
    return importlib.import_module(name)


def batch(tmp_path):
    value=samples(reference(tmp_path));value['velocities']=np.full((2,7),.1)
    value['actions']=np.tile(np.arange(1,21)[None,:,None]*.005,(2,1,7))
    return value,given.fit_normalization(value,split='train')


def test_full_target_metrics_use_dual_unrelaxed_gate_and_zero_baseline(tmp_path):
    m=module();value,_=batch(tmp_path)
    metrics=m.metrics(value['actions'],value)
    assert m.admission_gate(metrics)['passed'] is True
    assert metrics['metric_space']=='raw target-space MAE; not measured tracking'
    zero=dict(value,actions=np.zeros((2,20,7)),velocities=np.zeros((2,7)))
    gate=m.admission_gate(m.metrics(np.zeros((2,20,7)),zero))
    assert gate['passed'] is True and gate['act_over_hold'] is None and gate['act_over_cv'] is None
    assert m.admission_gate(m.metrics(np.full((2,20,7),1e-20),zero))['passed'] is False


@pytest.mark.parametrize('fault',['nan','negative','short_horizon','missing','count'])
def test_gate_refuses_nonfinite_partial_or_missing_metrics(tmp_path,fault):
    m=module();value,_=batch(tmp_path);result=m.metrics(value['actions'],value)
    if fault=='nan':result['act']['full_chunk_mae_rad']=float('nan')
    elif fault=='negative':result['hold']['full_chunk_mae_rad']=-1.
    elif fault=='short_horizon':result['act']['per_horizon_mae_rad'].pop()
    elif fault=='missing':del result['constant_velocity']
    else:result['sample_count']=0
    with pytest.raises(ValueError):m.admission_gate(result)


def test_earliest_argmin_requires_exact_complete_forty_snapshot_schedule(tmp_path):
    m=module();value,_=batch(tmp_path);metric=m.metrics(value['actions'],value)
    rows=[dict(step=step,metrics=deepcopy(metric)) for step in range(500,20001,500)]
    assert m.earliest_argmin(rows)==500
    rows[1]['metrics']['act']['full_chunk_mae_rad']=-1.
    with pytest.raises(ValueError):m.earliest_argmin(rows)
    with pytest.raises(ValueError):m.earliest_argmin(rows[:-1])


def test_device_tolerance_checks_every_cell_and_never_changes_gate(tmp_path):
    m=module();value,_=batch(tmp_path);a=value['actions'].copy();b=a.copy()
    b[-1,-1,-1]+=5e-7
    assert m.compare_devices(a,b,m.metrics(a,value),m.metrics(b,value))['passed'] is True
    b[-1,-1,-1]+=1e-3
    with pytest.raises(ValueError,match='prediction'):m.compare_devices(a,b,m.metrics(a,value),m.metrics(b,value))
    with pytest.raises(ValueError):m.compare_devices(a,b[:-1],m.metrics(a,value),m.metrics(b[:-1],dict(value,actions=value['actions'][:-1],velocities=value['velocities'][:-1])))


def test_canonical_prediction_roundtrip_rejects_changed_file(tmp_path):
    m=module();a=np.arange(280,dtype=np.float32).reshape(2,20,7)[:,::-1]
    ref=m.save_predictions(tmp_path/'prediction.npy',a,window_order_sha256='a'*64)
    assert ref['array']['dtype']=='<f8' and ref['array']['order']=='C'
    assert np.array_equal(m.load_predictions(ref,window_order_sha256='a'*64),a)
    with pytest.raises(ValueError):m.load_predictions(ref,window_order_sha256='b'*64)
    with (tmp_path/'prediction.npy').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError):m.load_predictions(ref,window_order_sha256='a'*64)


def test_inference_packing_never_reads_action_or_padding_labels(tmp_path):
    m=module();value,stats=batch(tmp_path)
    clean=m.pack_observations(value,stats,[1,0])
    assert set(clean)=={'observation.state','observation.environment_state'}
    assert clean['observation.state'].shape==(2,7) and clean['observation.environment_state'].shape==(2,27)
    del value['actions'];value['action_is_pad']=object()
    for key,result in m.pack_observations(value,stats,[1,0]).items():assert np.array_equal(result,clean[key])


def test_runtime_reference_is_common_contract_plus_fresh_independent_reference(tmp_path):
    m=module();(tmp_path/'train').mkdir();(tmp_path/'new-runtime').mkdir()
    a=reference(tmp_path/'train');b=reference(tmp_path/'new-runtime')
    cp=dict(task_contract=m.task_contract(),reference_inventory=[dict(episode_id='first',reference=a.snapshot(),reference_sha256=a.sha256)])
    current=dict(metadata(b),handoff_policy_id=m.handoff.POLICY_ID)
    m.validate_runtime_reference(cp,current)
    bad=dict(current);bad['action_semantics_id']='old_applied_q_command'
    with pytest.raises(ValueError):m.validate_runtime_reference(cp,bad)


def test_dependency_inventory_keeps_duplicate_distribution_names_and_locations(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m.sys,'prefix',str(tmp_path))
    class Dist:
        def __init__(self,path):
            import base64,hashlib
            self._path=path;path.mkdir();(path/'METADATA').write_text('same distribution\n')
            body=(path/'METADATA').read_bytes();sha=base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip('=')
            (path/'RECORD').write_text(path.name+'/METADATA,sha256='+sha+','+str(len(body))+'\n'+path.name+'/RECORD,,\n')
            self.metadata={'Name':'same-package'};self.version='1.0'
    first=Dist(tmp_path/'first.dist-info');second=Dist(tmp_path/'second.dist-info')
    result=m._dependency_inventory([second,first,first])
    assert len(result)==3 and len({r['metadata_location'] for r in result})==2
    assert all(r['name']=='same-package' for r in result)


def dependency_fixture(tmp_path,monkeypatch,*,editable=False):
    import base64,csv,hashlib
    m=module();prefix=tmp_path/'venv';site=prefix/'lib/python3.12/site-packages';site.mkdir(parents=True)
    monkeypatch.setattr(m.sys,'prefix',str(prefix))
    location=site/'tiny-1.0.dist-info';location.mkdir();(location/'METADATA').write_text('Name: tiny\nVersion: 1.0\n')
    payload=site/'tiny.py';payload.write_bytes(b'alpha=1\n')
    script=prefix/'bin/tiny';script.parent.mkdir();script.write_text('entrypoint\n')
    def digest(path):return 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
    rows=[['tiny.py',digest(payload),str(payload.stat().st_size)],
        ['../../../bin/tiny',digest(script),str(script.stat().st_size)],
        [location.name+'/METADATA',digest(location/'METADATA'),str((location/'METADATA').stat().st_size)],
        [location.name+'/RECORD','','']]
    with (location/'RECORD').open('w') as stream:csv.writer(stream).writerows(rows)
    class Dist:
        _path=location;metadata={'Name':'tiny'};version='1.0'
    return m,Dist(),payload,rows


@pytest.mark.parametrize('fault',[None,'same_size_payload','record_hash','record_size','escape','unknown_blank','missing'])
def test_runtime_inventory_authenticates_record_payload_and_legitimate_installer_paths(tmp_path,monkeypatch,fault):
    import csv
    m,dist,payload,rows=dependency_fixture(tmp_path,monkeypatch)
    baseline=m._dependency_inventory([dist])
    if fault=='same_size_payload':payload.write_bytes(b'alpha=2\n')
    elif fault=='record_hash':rows[0][1]='sha256='+'A'*43
    elif fault=='record_size':rows[0][2]='1'
    elif fault=='escape':rows[0][0]='../../../../../../outside.py'
    elif fault=='unknown_blank':rows[0][1:]=['','']
    elif fault=='missing':payload.unlink()
    if fault not in (None,'same_size_payload','missing'):
        with (dist._path/'RECORD').open('w') as stream:csv.writer(stream).writerows(rows)
    if fault is None:
        payloads=baseline[0]['payloads']
        assert str(payload) in payloads and any(row['record_path']=='../../../bin/tiny' for row in payloads.values())
        own=payloads[str(dist._path/'RECORD')]
        assert own['record_scope']=='installer_record_self' and own['reference']['sha256']==m.reference(dist._path/'RECORD')['sha256']
    else:
        with pytest.raises(ValueError,match='Dependency|RECORD'):m._dependency_inventory([dist])


def test_only_exact_editable_vendor_occurrence_can_omit_record(tmp_path,monkeypatch):
    m=module();root=tmp_path/'vendor';location=root/'src/lerobot.egg-info';location.mkdir(parents=True)
    (location/'PKG-INFO').write_text('Name: lerobot\nVersion: 1\n');(location/'SOURCES.txt').write_text('src/lerobot/model.py\n')
    monkeypatch.setattr(m,'VENDOR_ROOT',str(root))
    class Dist:
        _path=location;metadata={'Name':'lerobot'};version='1'
    result=m._dependency_inventory([Dist()])
    assert result[0]['payload_source']['kind']=='editable_pinned_vendor'
    assert len(result[0]['files'])==2 and result[0]['payloads']=={}
    Dist.metadata={'Name':'unreviewed-package'}
    with pytest.raises(ValueError,match='RECORD'):m._dependency_inventory([Dist()])


def test_binary_dependency_payload_and_explicit_blank_installer_metadata_are_frozen(tmp_path,monkeypatch):
    import base64,csv,hashlib
    m,dist,payload,rows=dependency_fixture(tmp_path,monkeypatch)
    binary=payload.with_suffix('.so');payload.rename(binary);binary.write_bytes(b'\x7fELF1234')
    rows[0]=['tiny.so','sha256='+base64.urlsafe_b64encode(hashlib.sha256(binary.read_bytes()).digest()).decode().rstrip('='),'8']
    installer=dist._path/'INSTALLER';installer.write_text('uv\n');rows.append([dist._path.name+'/INSTALLER','',''])
    with (dist._path/'RECORD').open('w') as stream:csv.writer(stream).writerows(rows)
    before=m._dependency_inventory([dist]);installer.write_text('py\n');after=m._dependency_inventory([dist])
    assert after[0]['payloads'][str(installer)]['record_scope']=='installer_generated_metadata'
    with pytest.raises(ValueError):m.validate_software(before,after)
    binary.write_bytes(b'\x7fELF5678')
    with pytest.raises(ValueError,match='payload hash/size'):m._dependency_inventory([dist])


def test_official_constructor_refuses_wrong_prefix_before_torch_or_cuda(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m.sys,'prefix',str(tmp_path/'untrusted'))
    with pytest.raises(ValueError,match='official'):m.make_model(dict(lerobot_root=m.VENDOR_ROOT),device='cpu')


def vendor_fixture(tmp_path,monkeypatch):
    m=module();root=tmp_path/'vendor';root.mkdir();(root/'src').mkdir()
    for i in range(m.VENDOR_PYTHON_FILES):(root/'src'/('source'+str(i)+'.py')).write_text('x=1\n')
    for name in ('pyproject.toml','uv.lock'):(root/name).write_text('frozen\n')
    subprocess.run(['git','init','-q',str(root)],check=True)
    subprocess.run(['git','-C',str(root),'add','.'],check=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=CPU Fixture','-c','user.email=cpu@fixture.invalid','commit','-qm','synthetic vendor'],check=True)
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    monkeypatch.setattr(m,'UPSTREAM_COMMIT',commit)
    return m,root


def test_vendor_inventory_closes_all_880_python_and_build_lock_files(tmp_path,monkeypatch):
    m,root=vendor_fixture(tmp_path,monkeypatch)
    result=m.vendor_inventory(root)
    assert len(result['files'])==882 and {'pyproject.toml','uv.lock'}<=set(result['files'])
    (root/'src/untracked_import.py').write_text('unreviewed=1\n')
    with pytest.raises(ValueError,match='untracked'):m.vendor_inventory(root)


def checkpoint_fixture(tmp_path):
    m=module();a=reference(tmp_path);value,stats=batch(tmp_path)
    return dict(schema=given.CHECKPOINT_SCHEMA,formal_run_schema=m.FORMAL_RUN_SCHEMA,upstream_commit=shared.UPSTREAM_COMMIT,
        model=shared.native_spec(),task_contract=m.task_contract(),step=500,selection_split='validation',state_only=True,
        action_representation='block_relative_joint',cpu_fixture_only=False,normalization=stats,
        reference_inventory=[dict(episode_id='one',reference=a.snapshot(),reference_sha256=a.sha256)],
        inputs=dict(dataset={name:dict(path='/immutable/'+name+'.json',sha256='a'*64) for name in
            ('plan','seal','manifest','qualification','reconstruction')},
            **{name:dict(path='/immutable/'+name+'.json',sha256='b'*64) for name in
                ('config','software','normalization','reference_inventory')}),
        metrics=m.metrics(value['actions'],value),state_dict={})


@pytest.mark.parametrize('fault',['old_schema','pilot','fixture','one_history','bad_stats','unbound_reference','partial_step','not_val','old_action'])
def test_formal_checkpoint_cannot_promote_pilot_fixture_or_unbound_metadata(tmp_path,fault):
    m=module();cp=checkpoint_fixture(tmp_path);m.validate_checkpoint(cp)
    if fault=='old_schema':cp['schema']=shared.SCHEMA
    elif fault=='pilot':cp.pop('formal_run_schema')
    elif fault=='fixture':cp['cpu_fixture_only']=True
    elif fault=='one_history':cp['task_contract']['history_steps']=1
    elif fault=='bad_stats':cp['normalization']['fitted_split']='validation'
    elif fault=='unbound_reference':cp['reference_inventory'][0]['reference_sha256']='c'*64
    elif fault=='partial_step':cp['step']=19999
    elif fault=='not_val':cp['selection_split']='train'
    else:cp['task_contract']['action_semantics_id']='applied_q_command'
    with pytest.raises(ValueError):m.validate_checkpoint(cp)


def test_fixture_generation_consumes_reviewed_draw_order_without_native_claim():
    m=module()
    class Draws:
        calls=[]
        def uniform(self,low,high,size):self.calls.append(('uniform',low,high,size));return np.zeros(size)
        def normal(self,mean,std,size):self.calls.append(('normal',mean,std,size));return np.zeros(size)
        def standard_normal(self,size):self.calls.append(('standard_normal',size));return np.zeros(size)
    rng=Draws();samples=m._fixture_samples(rng)
    assert rng.calls==[('uniform',-.8,.8,(65,7)),('normal',0,.01,(65,7)),
        ('uniform',-.06,.06,(65,3)),('normal',0,.002,(65,3)),
        ('uniform',-.4,.4,(65,3)),('normal',0,.01,(65,3)),('standard_normal',(65,20,7))]
    assert samples['histories'].shape==(65,2,17) and samples['actions'].shape==(65,20,7)
    assert np.array_equal(samples['histories'][:,1],samples['states'])
    assert samples['training_eligible'] is False and samples['synthetic'] is True
    assert samples['actions'][0,-1].tolist()==[.1,-.1,.1,-.1,.1,-.1,.1]


def test_fixture_freeze_refuses_nonofficial_process_before_generation(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m.sys,'prefix',str(tmp_path))
    config=m.write_new(tmp_path/'config.json',dict(lerobot_root=m.VENDOR_ROOT))
    with pytest.raises(ValueError,match='official'):m.freeze_fixture(config,tmp_path/'fixture')
    assert not (tmp_path/'fixture').exists()


def test_inference_batches_64_then_one_and_ignores_all_labels(tmp_path):
    import torch
    m=module();value,stats=batch(tmp_path)
    samples={key:np.repeat(value[key][:1],65,axis=0) for key in ('states','histories')}
    class Model:
        batches=[]
        def eval(self):pass
        def predict_action_chunk(self,b):
            assert set(b)==set(m.OBSERVATION_KEYS)
            n=len(b['observation.state']);self.batches.append(n);return torch.zeros((n,20,7))
    model=Model();output=m.infer(torch,model,samples,stats,device='cpu')
    assert model.batches==[64,1] and output.dtype.str=='<f8' and output.shape==(65,20,7)
    samples['action']=object();samples['action_is_pad']=object()
    assert m.infer(torch,model,samples,stats,device='cpu').tobytes()==output.tobytes()


def test_fresh_reference_predictor_uses_full_pose_and_two_real_frames(tmp_path):
    import torch
    m=module();cp=checkpoint_fixture(tmp_path);(tmp_path/'fresh').mkdir();ref=reference(tmp_path/'fresh')
    meta=dict(metadata(ref),handoff_policy_id=m.handoff.POLICY_ID)
    state=shared.contract.encode_state(np.zeros(7),ref.pose_base,1,metadata=meta)
    class Model:
        def eval(self):pass
        def predict_action_chunk(self,b):return torch.zeros((1,20,7))
    predictor=m.ChunkPredictor(torch,Model(),cp,meta)
    kwargs=dict(metadata=meta,history=np.stack([state,state]),history_times=[0,.05])
    result=predictor(np.zeros(7),ref.pose_base,1,**kwargs)
    assert result.values.shape==(20,7)
    kwargs['history_times']=[0,0]
    with pytest.raises(ValueError):predictor(np.zeros(7),ref.pose_base,1,**kwargs)


def numeric_result_fixture(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'processing_audit',lambda binding:dict(passed=True))
    monkeypatch.setattr(m,'software_inventory',lambda root:{})
    class Draws:
        def uniform(self,low,high,size):return np.zeros(size)
        def normal(self,mean,std,size):return np.zeros(size)
        def standard_normal(self,size):return np.zeros(size)
    value=m._fixture_samples(Draws());stats=given.fit_normalization(value,split='train')
    (tmp_path/'initial.pt').write_bytes(b'synthetic contract fixture; not a model')
    keys=('states','histories','actions','velocities')
    np.savez(tmp_path/'samples.npz',**{key:value[key] for key in keys})
    manifest=dict(spec=m.fixture_spec(),samples=m.reference(tmp_path/'samples.npz'),
        arrays={key:m._array_binding(value[key]) for key in keys},
        normalization=m.write_new(tmp_path/'norm.json',stats),initial_state=m.reference(tmp_path/'initial.pt'),
        software=m.write_new(tmp_path/'software.json',{}),processing={},window_order_sha256=m.digest(list(range(65))))
    fixture=m.write_new(tmp_path/'fixture.json',manifest);refs=[]
    for i,device in enumerate(('cpu','cpu','cuda')):
        predictions=np.zeros((65,20,7))
        result=dict(schema=m.FIXTURE_RESULT_SCHEMA,fixture=fixture,device=device,identity=dict(pid=i+100,start_ticks=str(i)),
            execution=dict(device=device,parameter_devices=[device],parameter_dtypes=['torch.float32'],default_dtype='torch.float32',
                intraop_threads=2,interop_threads=1,deterministic_algorithms=True,matmul_precision='highest',matmul_tf32=False,
                cudnn_tf32=False,cudnn_deterministic=True,cudnn_benchmark=False,cublas_workspace_config=':4096:8',
                torch_version='CPU unit fixture',cuda_version='CPU unit fixture',device_name='CPU unit fixture'),
            software=manifest['software'],initial_state=manifest['initial_state'],batches=[64,1],
            predictions=m.save_predictions(tmp_path/(str(i)+'.npy'),predictions,window_order_sha256=manifest['window_order_sha256']),
            metrics=m.metrics(predictions,value),runtime_only=True,synthetic=True,training_eligible=False)
        refs.append(m.write_new(tmp_path/(str(i)+'.json'),result))
    owner_dir=tmp_path/'owner';owner_dir.mkdir()
    _,terminal,complete,stub,child=owner_fixture(owner_dir,input_ref=fixture)
    owner=m.read_bound(stub);cuda=m.read_bound(refs[2]);cuda.update(identity=child,owned_stage=owner['owned_stage'],ownership=owner['ownership'])
    Path(stub['path']).write_text(__import__('json').dumps(cuda));refs[2]=m.reference(stub['path'])
    end=m.read_bound(complete);end['result']=refs[2]
    Path(complete['path']).write_text(__import__('json').dumps(end))
    return m,fixture,refs,terminal


@pytest.mark.parametrize('fault',[None,'same_process','negative_zero','reordered','partial','metrics','recipe','actual_device'])
def test_fixture_qualification_reopens_arrays_and_requires_two_fresh_byte_identical_cpus(tmp_path,monkeypatch,fault):
    m,fixture,refs,terminal=numeric_result_fixture(tmp_path,monkeypatch)
    if fault in ('same_process','negative_zero','partial','metrics','actual_device'):
        result=m.read_bound(refs[1])
        if fault=='same_process':result['identity']=m.read_bound(refs[0])['identity']
        elif fault=='negative_zero':
            a=m.load_predictions(result['predictions'],window_order_sha256=m.digest(list(range(65))));a[-1,-1,-1]=-0.
            result['predictions']=m.save_predictions(tmp_path/'changed.npy',a,window_order_sha256=m.digest(list(range(65))))
        elif fault=='partial':result['batches']=[64]
        elif fault=='actual_device':result['execution']['parameter_devices']=['cuda']
        else:result['metrics']['act']['per_horizon_mae_rad'][-1]+=.01
        refs[1]=m.write_new(tmp_path/'changed.json',result)
    elif fault=='reordered':refs=refs[::-1]
    elif fault=='recipe':
        value=m.read_bound(fixture);value['spec']['seed']+=1
        fixture=m.write_new(tmp_path/'changed-fixture.json',value)
    if fault is None:
        result=m.qualify_fixture(fixture,refs,tmp_path/'qualified.json',cuda_owner_terminal=terminal)
        assert result['passed'] is True and result['training_eligible'] is False
        assert result['comparison']['prediction_max_abs_rad']==0
    else:
        with pytest.raises(ValueError):m.qualify_fixture(fixture,refs,tmp_path/'qualified.json',cuda_owner_terminal=terminal)
        assert m.read_bound(m.reference(tmp_path/'qualified.json'))['passed'] is False


def test_complete_requires_cuda_earliest_argmin_all_forty_and_source_audit(tmp_path):
    m=module();value,_=batch(tmp_path);metric=m.metrics(value['actions'],value)
    ref=dict(path='/immutable/snapshot.pt',sha256='a'*64)
    rows=[dict(step=step,metrics=deepcopy(metric),checkpoint=ref,predictions={}) for step in m.EVAL_STEPS]
    complete=dict(schema=m.FORMAL_RUN_SCHEMA,completed_steps=20000,complete=True,source_audit_passed=True,
        cpu_fixture_only=False,test_payloads_opened=False,task_contract=m.task_contract(),evaluations=rows,
        selected_step=500,best=ref,last=ref,selected_predictions={s:{} for s in ('train','validation')},
        selected_metrics={s:deepcopy(metric) for s in ('train','validation')})
    m.validate_complete(complete)
    for field,new_value in [('completed_steps',19500),('selected_step',1000),('source_audit_passed',False),('cpu_fixture_only',True)]:
        bad=deepcopy(complete);bad[field]=new_value
        with pytest.raises(ValueError):m.validate_complete(bad)
    complete['evaluations'].pop()
    with pytest.raises(ValueError):m.validate_complete(complete)


@pytest.mark.parametrize('index_flag',['--assume-unchanged','--skip-worktree'])
def test_vendor_pinned_blobs_cannot_be_hidden_by_git_index_flags(tmp_path,monkeypatch,index_flag):
    m,root=vendor_fixture(tmp_path,monkeypatch)
    subprocess.run(['git','-C',str(root),'update-index',index_flag,'src/source0.py'],check=True)
    (root/'src/source0.py').write_text('x=2\n')
    with pytest.raises(ValueError,match='pinned'):m.vendor_inventory(root)


def test_fixture_source_audits_accept_real_accounting_observation_time_changes(tmp_path,monkeypatch):
    from test_ausc_given_processing_source_envelope import source_scope
    m=module();scope=source_scope(tmp_path,monkeypatch)
    reports=[]
    def audit(binding):
        report=m.dataset.processing_source_accounting(scope,tmp_path,snapshot_phase='pre_export')
        reports.append(report);return report
    monkeypatch.setattr(m,'processing_audit',audit)
    monkeypatch.setattr(m.sys,'prefix',str(Path(m.VENDOR_ROOT)/'.venv'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    class Torch:
        def save(self,value,stream):stream.write(b'unit-test state only; never official model')
    class Policy:
        def state_dict(self):return {}
    monkeypatch.setattr(m,'make_model',lambda *a,**k:(Torch(),Policy(),{}))
    monkeypatch.setattr(m,'software_inventory',lambda root:{})
    # Stub generation: this test does not execute the actual reviewed PCG draw.
    class Draws:
        def uniform(self,lo,hi,size):return np.zeros(size)
        def normal(self,mean,std,size):return np.zeros(size)
        def standard_normal(self,size):return np.zeros(size)
    samples=m._fixture_samples(Draws());monkeypatch.setattr(m,'_fixture_samples',lambda rng:samples)
    config=m.write_new(tmp_path/'runtime-config.json',dict(lerobot_root=m.VENDOR_ROOT,fixture_spec=m.fixture_spec(),processing={}))
    result=m.freeze_fixture(config,tmp_path/'unit-fixture')
    assert result['training_eligible'] is False and len(reports)>=2
    assert reports[0]['observed_at']!=reports[1]['observed_at']


def owner_fixture(tmp_path,*,input_ref=None):
    m=module();folder=tmp_path/'runs/ausc_policy/numeric-fixed';out=folder/'output';out.mkdir(parents=True)
    module_name='sim.ausc_policy.m2_given_contact_runtime';experiment='m2_given_contact_numeric_fixture'
    child=dict(pid=300,pgid=300,sid=300,start_ticks='7')
    source_guard=m.reference(Path(m.__file__).with_name('process_guard.py'))
    config=m.write_new(tmp_path/'stage.json',dict(schema='m2_given_contact_owned_stage_v1',experiment=experiment,
        base=str(tmp_path),time_limit_seconds=3600,input=input_ref or m.write_new(tmp_path/'numeric-manifest.json',{})))
    cfg=m.read_bound(config)
    command=[m.sys.executable,'-u','-B','-m',module_name,'--config',config['path'],'--output',str(out)]
    run=dict(run_id=folder.name,code_commit='a'*40,config=cfg,config_sha256=config['sha256'],command=command,
        resource_guard=dict(path=source_guard['path'],sha256=source_guard['sha256'],gpu_exit_grace_s=15.),started_utc='2026-10-28T01:00:00Z',completed=False)
    start=dict(run_id=folder.name,run=m.write_new(folder/'run.json',run),config=config,input=cfg['input'],output=str(out),
        experiment=experiment,module=module_name,child_identity=child,guard=source_guard)
    lock=dict(path=m.GPU_LOCK,device=m.GPU_LOCK_ID[0],inode=m.GPU_LOCK_ID[1],pid=child['pid'],checks=dict(gpu_pids_before=[]))
    result=m.write_new(out/'result.json',dict(schema=m.FIXTURE_RESULT_SCHEMA,identity=child,owned_stage=start,ownership=lock))
    complete=m.write_new(out/'COMPLETE.json',dict(schema='m2_given_contact_owned_fixture_complete_v1',result=result,
        identity=child,owned_stage=start,ownership=lock))
    stage=dict(passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[],process_identity=child)
    (folder/'lifecycle').mkdir();m.write_new(folder/'lifecycle'/(experiment+'.process.json'),stage)
    terminal=m.write_new(folder/'result.json',dict(run,stage=stage,completed=True,gpu_released=True,ended_utc='2026-10-28T01:00:01Z'))
    return m,terminal,complete,result,child


@pytest.mark.parametrize('fault',[None,'no_terminal','exit','drain','release','guard','identity','config','result','lock'])
def test_cuda_requires_original_guard_owner_terminal_and_exact_child_complete(tmp_path,fault):
    m,terminal,complete,result,child=owner_fixture(tmp_path)
    if fault in ('exit','drain','release','guard','identity','config'):
        value=m.read_bound(terminal)
        if fault=='exit':value['stage']['exit_code']=-9
        elif fault=='drain':value['stage']['process_group_drained']=False
        elif fault=='release':value['gpu_released']=False
        elif fault=='guard':value['resource_guard']['sha256']='f'*64
        elif fault=='identity':value['stage']['process_identity']['start_ticks']='8'
        else:value['config']['input']['sha256']='f'*64
        Path(terminal['path']).write_text(__import__('json').dumps(value));terminal=m.reference(terminal['path'])
    elif fault in ('result','lock'):
        value=m.read_bound(complete)
        if fault=='result':value['result']['sha256']='f'*64
        else:value['ownership']['inode']+=1
        Path(complete['path']).write_text(__import__('json').dumps(value));complete=m.reference(complete['path'])
    if fault=='no_terminal':terminal=None
    if fault is None:
        assert m.verify_owned_stage(terminal,complete_ref=complete,child_identity=child,
            experiment='m2_given_contact_numeric_fixture',child_result_ref=result)['passed'] is True
    else:
        with pytest.raises(ValueError):m.verify_owned_stage(terminal,complete_ref=complete,child_identity=child,
            experiment='m2_given_contact_numeric_fixture',child_result_ref=result)


def test_actual_owned_stage_start_requires_original_parent_not_only_env(tmp_path,monkeypatch):
    m,terminal,complete,_,child=owner_fixture(tmp_path);start=m.read_bound(complete)['owned_stage']
    from sim.ausc_policy import process_guard as guard
    monkeypatch.setattr(m.sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    monkeypatch.setattr(m,'_identity',lambda:child)
    monkeypatch.setattr(guard,'process_identity',lambda pid:dict(pid=pid,start_ticks='parent'))
    cmd=[m.sys.executable,'-B','-m','sim.ausc_policy.run','--config',start['config']['path'],'--run-id',start['run_id']]
    monkeypatch.setattr(m,'_process_command',lambda pid:cmd)
    value=m.start_owned_stage(start['config'],start['output'],experiment=start['experiment'])
    assert value['child_identity']==child and value['parent_command']==cmd
    cmd[3]='unrelated.module'
    with pytest.raises(ValueError,match='parent'):m.start_owned_stage(start['config'],start['output'],experiment=start['experiment'])


def test_cuda_fixture_direct_env_injection_cannot_bypass_original_stage(tmp_path,monkeypatch):
    m=module();monkeypatch.setenv('AUSC_OWNED_RUN','1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='original'):m.run_fixture({},tmp_path/'out',device='cuda')
