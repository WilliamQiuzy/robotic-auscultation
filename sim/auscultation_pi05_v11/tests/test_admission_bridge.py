"""Public synthetic CPU bridge tests; no original evidence reconstruction."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import pytest


def bridge():
    name='sim.auscultation_pi05_v11.admission_bridge'
    assert importlib.util.find_spec(name) is not None,'Missing V11 external-producer bridge'
    return importlib.import_module(name)


def test_external_command_keeps_original_producer():
    B=bridge();contract=dict(entry_binding={'path':'/original/external/postlock.py','sha256':'a'*64},worker_python={'path':'/original/python'})
    assert B.worker_command(contract,'/run/postlock/admission.json','b'*64)==['/original/python','-B','/original/external/postlock.py','--verify-admission','/run/postlock/admission.json','--admission-sha256','b'*64]
    env=B.owned.child_environment(Path('/original392'))
    assert env['PYTHONPATH']=='/original392' and env['CUDA_VISIBLE_DEVICES']=='' and env['JAX_PLATFORMS']=='cpu'


def test_artifact_rebind_or_symlink_rejected(tmp_path):
    B=bridge();p=tmp_path/'x';p.write_text('original');h=B.file_sha(p);a={str(p):h}
    B.add_artifacts(a,{str(p):h});p.write_text('changed')
    with pytest.raises(ValueError):B.add_artifacts(a,{str(p):B.file_sha(p)})
    q=tmp_path/'link';q.symlink_to(p)
    with pytest.raises(ValueError):B.read_bound(q,B.file_sha(p))


def test_negative_scientific_result_never_becomes_runtime_admission():
    B=bridge()
    with pytest.raises(ValueError,match='Positive'):B.build_runtime_admission({'candidate_admitted':False,'service_eligible':False},'/synthetic','a'*64,{})


def test_frozen_tree_rejects_extra_file_fifo_and_writable_dir(tmp_path):
    B=bridge();root=tmp_path/'tree';root.mkdir();f=root/'x';f.write_text('x');h=B.file_sha(f);f.chmod(0o444);root.chmod(0o555)
    try:
        B.verify_tree(root,{'x':h})
        root.chmod(0o755)
        with pytest.raises(ValueError):B.verify_tree(root,{'x':h})
    finally:root.chmod(0o755)


@pytest.mark.parametrize('kind',['extra','fifo','hardlink','symlink'])
def test_source_tree_is_independent_regular_and_exact(tmp_path,kind):
    B=bridge();root=tmp_path/'tree';root.mkdir();f=root/'x';f.write_text('x');f.chmod(0o444);files={'x':B.file_sha(f)}
    if kind=='extra':(root/'unlisted').write_text('extra');(root/'unlisted').chmod(0o444)
    elif kind=='fifo':os.mkfifo(root/'unlisted',0o444)
    elif kind=='hardlink':os.link(f,tmp_path/'alias')
    else:(root/'unlisted').symlink_to(tmp_path/'absent')
    root.chmod(0o555)
    try:
        with pytest.raises(ValueError):B.verify_tree(root,files)
    finally:root.chmod(0o755)


def test_new_bridge_does_not_import_scientific_modules():
    B=bridge();text=Path(B.__file__).read_text()
    assert 'verify_pair_admission(' not in text and 'from . import train' not in text and 'import sim.auscultation_pi05_v11.postlock' not in text



def synthetic_normalization():
    import numpy as np
    B=bridge();A=B.codec;bounds=A.build_bounds_contract(Path(B.__file__).resolve().parents[2]);limits=bounds['targets']['R4']
    q=((np.asarray(limits['lower_rad'])+np.asarray(limits['upper_rad']))/2).astype(np.float32);episodes=[]
    for phase in A.PHASES:
        state=np.zeros((10,41),np.float32);state[:,:7]=q
        episodes.append(dict(episode_id='synthetic_'+phase,split='train',actions=np.tile(q,(10,1)),state=state,phases=np.full(10,phase),targets=np.full(10,'R4'),starts=[0]))
    return dict(schema='wam_pi05_v11_bounded_normalization_v1',mode=A.MODE,residual_actions=A.fit_stats(episodes,bounds),action_encoding_binding=A.action_encoding_binding(bounds))

def admitted_fixture(tmp_path,monkeypatch):
    """Real JSON/file bindings with V11 make_config/D identity field shapes; synthetic content only."""
    B=bridge();run=tmp_path/'run';cpdir=run/'model/step-000000000500';cpdir.mkdir(parents=True)
    def put(p,v):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(B.json_bytes(v));return dict(path=str(p),sha256=B.file_sha(p))
    manifest=put(tmp_path/'manifest.json',dict(schema='wam_pi05_v10_recovery_composite_manifest_v1',synthetic=True))
    norm=synthetic_normalization();nb=put(tmp_path/'normalization.json',norm)
    obj=hashlib.sha256(B.json_bytes(norm).rstrip(b'\n')).hexdigest()
    training_source={'sim/synthetic.py':'a'*64};producer=tmp_path/'original392';producer.mkdir();entry=put(tmp_path/'external/postlock.py',{'source':'synthetic'})
    identity=dict(schema='wam_pi05_v11_bounded_dataset_binding_v1',mode=B.codec.MODE,manifest_binding=manifest,source_manifests=[],design_binding={'path':'/synthetic/design','sha256':'b'*64},normalization_file_binding=nb,normalization_sha256=nb['sha256'],normalization_object_sha256=obj,action_encoding_binding=norm['action_encoding_binding'],source_sha256=training_source,training_episode_ids=['SYNTHETIC16'],normalization_fit_episode_ids=['SYNTHETIC8'],split_episode_ids={'train':['SYNTHETIC16'],'validation':['R5'],'test':['L5']})
    identity['data_hash']=B.digest(identity)
    cfg=dict(training_schema='wam_pi05_v11_bounded_training_v1',training_run_path=str(run),source_sha256=training_source,design_binding=identity['design_binding'],dataset_binding=identity,schedule_binding={'path':str(run/'training_schedule.json'),'sha256':'b'*64},parent_checkpoint_binding={'synthetic_parent_B7300':True},assets={'synthetic':True},seed=61,learning_rate=5e-5,batch_size=4,total_steps=4000,candidate_steps=list(range(0,4001,500)),action_dim=7,action_horizon=10,state_width=41,openpi_commit='synthetic',initialization='synthetic warm lineage',action_encoding_binding=norm['action_encoding_binding'],normalization_fit_episode_ids=['SYNTHETIC8'],observation_contract={'state_width':41},model_architecture='unmodified official pi05 seven-joint10-horizon LoRA/projection partition',**{k:'c'*64 for k in ('dynamics_recipe_sha256','dynamics_contract_sha256','execution_contract_sha256')})
    md=put(cpdir/'metadata.json',dict(config=cfg,normalization=norm));saved=put(run/'configuration.json',dict(config=cfg,normalization=norm,data_hash=identity['data_hash'],manifest=manifest['path'],normalization_path=nb['path']))
    cp=dict(schema='wam_pi05_v11_bounded_checkpoint_binding_v1',checkpoint_path=str(cpdir),local_step=500,optimizer_step=500,schedule_cursor=2000,cumulative_trainable_update_age=7800,metadata_sha256=md['sha256'],arrays_sha256='d'*64,complete_sha256='e'*64,data_hash=identity['data_hash'],normalization_sha256=nb['sha256'],config_sha256=B.digest(cfg),schedule_sha256='b'*64,source_sha256=training_source)
    a={x['path']:x['sha256'] for x in (manifest,nb,md,saved)}
    def pin(p,v):
        b=put(p,v);a[b['path']]=b['sha256'];return b
    terminal=dict(segment=str(run/'training_segments/final.json'));pin(run/'training_run.json',terminal);pin(Path(terminal['segment']),terminal)
    life=pin(tmp_path/'lifecycle/lifecycle.json',{'synthetic_owner':True});pin(run/'TRAINING_OWNER.json',{'lifecycle':life})
    collections={}
    for kind in ('probes','test'):
        folder=run/'postlock'/kind
        guard=pin(tmp_path/('guard_'+kind)/'collect.process.json',{'synthetic':True})
        pin(folder/'execution.json',{'guard_evidence_binding':guard})
        collections[kind]=pin(folder/'receipt.json',{'synthetic':True,'kind':kind})
    lock=pin(run/'selection/selection.lock.json',{'synthetic_selected':True})
    source_map={str(producer/k):h for k,h in training_source.items()};source_map[entry['path']]=entry['sha256']
    science=dict(schema='wam_pi05_v11_bounded_admission_v1',candidate_admitted=True,service_eligible=True,admitted_checkpoint_binding=cp,dataset_binding=identity,lock_binding=lock,probes_binding=collections['probes'],test_binding=collections['test'],source_binding=source_map,artifact_sha256=a.copy(),bounded_by_codec=True,learned_safety_validated=False,physical_acceptance=False,population_generalization=False,no_reselection=True)
    ab=put(run/'postlock/admission.json',science);verified=dict(science,receipt_binding=ab,artifact_sha256=dict(a,**{ab['path']:ab['sha256']}))
    contract=dict(artifact_sha256={},verifier_binding={'path':'/synthetic/verifier.json','sha256':'f'*64},entry_binding=entry,runtime_inventory_binding={'path':'/synthetic/runtime401.json','sha256':'e'*64})
    monkeypatch.setattr(B,'PRODUCER',producer)
    return B,verified,ab,contract


def test_positive_runtime_admission_uses_actual_v10_metadata_shape(tmp_path,monkeypatch):
    B,v,b,c=admitted_fixture(tmp_path,monkeypatch);a=B.build_runtime_admission(v,b['path'],b['sha256'],c)
    assert a['service_eligible'] and a['checkpoint_binding']['local_step']==500 and a['dataset_binding']==v['dataset_binding']
    assert a['training_source_sha256']==v['admitted_checkpoint_binding']['source_sha256']
    assert not a['physical_accepted'] and a['admission_binding']==b
    B.verify_artifacts(a)
    Path(b['path']).with_suffix('.invalid.json').symlink_to(tmp_path/'absent')
    with pytest.raises(ValueError):B.verify_artifacts(a)


@pytest.mark.parametrize('changed',['metadata','config','normalization','terminal_cancel','probe_failure','guard_cancel'])
def test_admitted_bytes_and_current_failure_markers_reject(tmp_path,monkeypatch,changed):
    B,v,b,c=admitted_fixture(tmp_path,monkeypatch)
    cp=Path(v['admitted_checkpoint_binding']['checkpoint_path']);run=cp.parent.parent
    if changed=='metadata':(cp/'metadata.json').write_text('{}')
    elif changed=='config':(run/'configuration.json').write_text('{}')
    elif changed=='normalization':Path(v['dataset_binding']['normalization_file_binding']['path']).write_text('{}')
    elif changed=='terminal_cancel':(run/'training_segments/final.cancelled.json').symlink_to(tmp_path/'absent')
    elif changed=='probe_failure':(run/'postlock/probes/failure.json').write_text('{}')
    else:(tmp_path/'guard_test/collect.late_interrupt.json').write_text('{}')
    with pytest.raises(ValueError):B.build_runtime_admission(v,b['path'],b['sha256'],c)


def process_fixture(tmp_path,monkeypatch):
    B,v,b,c=admitted_fixture(tmp_path,monkeypatch)
    c['worker_python']={'path':str(B.owned.OPENPI_PYTHON)}
    monkeypatch.setattr(B,'source_contract',lambda descriptor:c)
    result=dict(exit_code=0,process_group_drained=True,process_identity=dict(pid=12345,pgid=12345,sid=12345,start_ticks='123'),cancelled=False,elapsed_s=0.1,errors=[],signals=[],deferred_signals=[],stdout=(json.dumps(v,allow_nan=False)+'\n').encode(),stderr=b'')
    calls=[]
    def bounded(command,data,*,env,cwd):
        calls.append((command,json.loads(data),env,cwd));return result
    monkeypatch.setattr(B.owned,'run_bounded',bounded)
    return B,b,c,result,calls


def test_bridge_consumes_actual_owned_result_and_keeps_two_argvs(tmp_path,monkeypatch):
    B,b,c,r,calls=process_fixture(tmp_path,monkeypatch)
    a,receipt=B.selected_admission_with_receipt(b['path'],b['sha256'],verifier_binding=c['verifier_binding'])
    assert len(calls)==1 and calls[0][0]==receipt['guarded_command']
    assert calls[0][0]!=receipt['verifier_exec_command'] and receipt['process_identity']==r['process_identity']
    assert receipt['stdout_bytes']==len(r['stdout']) and receipt['stdout_sha256']==hashlib.sha256(r['stdout']).hexdigest()
    assert calls[0][2]['CUDA_VISIBLE_DEVICES']=='' and calls[0][2]['PYTHONPATH']==str(B.PRODUCER)
    assert B.verify_receipt(receipt,a)


@pytest.mark.parametrize('field,value',[('exit_code',1),('exit_code',False),('process_group_drained',False),('errors',['failure']),('signals',['TERM']),('deferred_signals',[15]),('stdout',b'{}\n')])
def test_owned_failures_never_return_success(tmp_path,monkeypatch,field,value):
    B,b,c,r,calls=process_fixture(tmp_path,monkeypatch);r[field]=value
    with pytest.raises(B.AdmissionFailure):B.selected_admission_with_receipt(b['path'],b['sha256'],verifier_binding=c['verifier_binding'])
    assert len(calls)==1


def test_owned_cancellation_preserves_keyboard_interrupt(tmp_path,monkeypatch):
    B,b,c,r,calls=process_fixture(tmp_path,monkeypatch);r['cancelled']=True
    with pytest.raises(B.AdmissionInterrupted) as caught:B.selected_admission_with_receipt(b['path'],b['sha256'],verifier_binding=c['verifier_binding'])
    assert isinstance(caught.value,KeyboardInterrupt) and caught.value.receipt['cancelled'] is True


@pytest.mark.parametrize('field,value',[('sdk_numpy_unchanged',False),('private_configuration_used',True),('stdout_bytes',0),('process_identity',dict(pid=12345,pgid=12345.0,sid=12345,start_ticks='123'))])
def test_receipt_environment_and_stream_identity_are_not_just_assertions(tmp_path,monkeypatch,field,value):
    B,b,c,r,calls=process_fixture(tmp_path,monkeypatch)
    a,receipt=B.selected_admission_with_receipt(b['path'],b['sha256'],verifier_binding=c['verifier_binding']);receipt[field]=value
    with pytest.raises(ValueError):B.verify_receipt(receipt,a)


def test_exact_392_plus_9_contract_and_legitimate_venv_symlink(tmp_path,monkeypatch):
    """Full-sized synthetic namespace; no real runtime/data/interpreter launched."""
    B=bridge();base=tmp_path/'base';base.mkdir();producer=base/'cloud-runtime-v11/original392';runtime=producer.with_name('physical401')
    original={f'sim/synthetic/{i:03d}.json':hashlib.sha256(str(i).encode()).hexdigest() for i in range(392)}
    files=dict(original,**{'sim/auscultation_pi05_v11/'+n:hashlib.sha256(n.encode()).hexdigest() for n in B.EXTENSIONS})
    def put(p,raw):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);return B.file_sha(p)
    for root,names in ((producer,original),(runtime,files)):
        for name in names:
            raw=Path(name).stem.encode() if name.endswith('.py') else str(int(Path(name).stem)).encode()
            if name.endswith('.py'):raw=Path(name).name.encode()
            assert put(root/name,raw)==names[name]
    inv=producer.with_name(producer.name+'.source_inventory.json');ih=put(inv,B.json_bytes(dict(source_project=str(producer),source_files=original)))
    entry=base/'diagnostics/external/postlock.py';eh=put(entry,b'# synthetic external')
    prefix=base/'venv';resolved=base/'bin/python-real';ph=put(resolved,b'synthetic executable bytes');requested=prefix/'bin/python';requested.parent.mkdir(parents=True);requested.symlink_to(resolved)
    guard=base/'guard.py';gh=put(guard,b'# synthetic guard')
    for name in ('numpy','scipy'):
        h=put(prefix/('lib/python3.11/site-packages/'+name+'/__init__.py'),name.encode());monkeypatch.setattr(B.owned,name.upper()+'_SHA',h)
    for k,v in dict(BASE=base,PRODUCER=producer,PARENT_INVENTORY=inv,ENTRY=entry).items():monkeypatch.setattr(B,k,v)
    for k,v in dict(OPENPI_PYTHON=requested,OPENPI_PREFIX=prefix,PYTHON_SHA=ph,GUARD_PATH=guard,GUARD_SHA=gh).items():monkeypatch.setattr(B.owned,k,v)
    monkeypatch.setattr(B,'root_path',lambda:runtime)
    descriptor=base/'descriptor.json';dh=put(descriptor,B.json_bytes(dict(schema=B.DESCRIPTOR_SCHEMA,entry_binding=dict(path=str(entry),sha256=eh),producer_source_project=str(producer),producer_inventory_binding=dict(path=str(inv),sha256=ih),interpreter_binding=dict(path=str(requested),resolved_path=str(resolved),sha256=ph,prefix=str(prefix)))))
    binding=dict(path=str(descriptor),sha256=dh);ri=runtime.with_name(runtime.name+'.source_inventory.json')
    put(ri,B.json_bytes(dict(schema='wam_pi05_v11_bounded_physical_runtime_inventory_v1',source_project=str(runtime),parent_inventory=dict(path=str(inv),sha256=ih),admission_verifier_binding=binding,source_files=files)))
    for root in (producer,runtime):
        for p in root.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        root.chmod(0o555)
    try:
        c=B.source_contract(binding)
        assert c==B.source_contract() and c['worker_python']['path']!=c['worker_python']['resolved_path']
        assert len([p for p in c['artifact_sha256'] if p.startswith(str(runtime)+'/')])==401
        (runtime/'sim/synthetic/000.json').chmod(0o644)
        with pytest.raises(ValueError):B.source_contract(binding)
    finally:
        for root in (producer,runtime):
            root.chmod(0o755)
            for p in root.rglob('*'):p.chmod(0o755 if p.is_dir() else 0o644)


def test_no_old_normalization_or_guessed_parent_hash_constants():
    B=bridge()
    assert not any(hasattr(B,n) for n in ('NORM_SHA','OBJECT_SHA','PARENT_SHA'))
    assert str(B.PRODUCER).endswith('/cloud-runtime-v11/bounded-B7300-20260928-v1')
    assert str(B.ENTRY).endswith('/diagnostics/pi05_v11_postlock_runtime_v1/postlock.py')


def test_exact_bridge_import_as_file_never_imports_scientific_context(tmp_path):
    import sys
    B=bridge();p=tmp_path/'bridge.py';p.write_bytes(Path(B.__file__).read_bytes());before=set(sys.modules)
    spec=importlib.util.spec_from_file_location('synthetic_external_v11_bridge',p);copy=importlib.util.module_from_spec(spec);spec.loader.exec_module(copy)
    assert copy.codec is B.codec and copy.owned is B.owned
    forbidden={'sim.auscultation_pi05_v11.'+n for n in ('train','training_data','evaluation','parity','postlock')}
    assert not (set(sys.modules)-before)&forbidden


@pytest.mark.parametrize('field,value',[('bounded_by_codec',False),('learned_safety_validated',True)])
def test_scientific_architectural_support_must_be_truthful(tmp_path,monkeypatch,field,value):
    B,v,b,c=admitted_fixture(tmp_path,monkeypatch)
    # Reseal only the synthetic external science document, so rejection must be
    # semantic rather than merely detecting stale JSON bytes.
    v[field]=value;raw={k:x for k,x in v.items() if k!='receipt_binding'};raw['artifact_sha256']=dict(v['artifact_sha256']);raw['artifact_sha256'].pop(b['path'])
    Path(b['path']).write_bytes(B.json_bytes(raw));b['sha256']=B.file_sha(b['path']);v['receipt_binding']=dict(b);v['artifact_sha256'][b['path']]=b['sha256']
    with pytest.raises(ValueError,match='Architectural'):B.build_runtime_admission(v,b['path'],b['sha256'],c)
