"""Synthetic postlock tests: no real selected run, L5 data, model or guard use."""
import importlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

def module():
    try:return importlib.import_module('sim.auscultation_pi05_v11.postlock')
    except ModuleNotFoundError as e:
        if e.name!='sim.auscultation_pi05_v11.postlock':raise
        pytest.fail('V11 postlock module not implemented')

def test_module_declares_new_identity_and_original_guard():
    P=module()
    assert P.ADMISSION_SCHEMA=='wam_pi05_v11_bounded_admission_v1'
    assert P.GUARD_SHA=='48c47cf0d6d8f3d770028931a61d2882c5281684053a6149dacfcbe7b2483ff2'
    assert callable(P.collect_probes) and callable(P.collect_test) and callable(P.verify_admission)

@pytest.fixture
def numerical():
    P=module();N=5;rows=[dict(row_id=i,phase=p,target='R4',episode_index=0,start=i,train_index=i,split='train') for i,p in enumerate(P.E.PHASES)]
    q=np.zeros((N,41),np.float32);q[:,3]=1
    episode=dict(state=q,actions=np.tile(q[0,:7],(N+9,1)),phases=np.array(['FREE']*(N+9)),targets=np.array(['R4']*(N+9)),target='R4',episode_id='synthetic',split='train')
    dataset=SimpleNamespace(episodes=[episode]);ref=np.tile(q[0,:7],(N,10,1));anchors=q[:,:7].copy()
    plans={'R4':{'joint_lower_rad':[-2]*7,'joint_upper_rad':[2]*7}}
    return P,rows,dataset,ref,anchors,plans

def test_full_two_pass_numeric_same_anchor_and_all70(numerical,monkeypatch):
    P,rows,data,ref,anchors,plans=numerical
    monkeypatch.setattr(P.D,'decode_prediction',lambda d,z,s,p,t:np.asarray(z+s[:7],np.float32))
    monkeypatch.setattr(P,'row_anchor',lambda d,r:d.episodes[r['episode_index']]['state'][r['start']])
    monkeypatch.setattr(P.E,'coordinate_diagnostics',lambda *a:{'synthetic':True})
    arrays={'reference':ref,'anchors':anchors};statuses=[]
    for pi in range(2):
        for i in range(len(rows)):
            arrays[P.key(pi,i,'normalized')]=np.zeros((1,10,7),np.float32)
            arrays[P.key(pi,i,'raw')]=ref[i].copy();statuses.append('finite')
    result=P.summarize(data,rows,arrays,statuses,plans,2)
    assert result['passed'] and result['replay_equal'] and result['coordinate_count']==350
    arrays[P.key(1,4,'raw')][9,0]+=.01
    with pytest.raises(ValueError):P.summarize(data,rows,arrays,statuses,plans,2)

def test_nonfinite_negative_and_replay_mismatch_no_finite_fabrication(numerical,monkeypatch):
    P,rows,data,ref,anchors,plans=numerical
    monkeypatch.setattr(P.D,'decode_prediction',lambda d,z,s,p,t:np.asarray(z+s[:7],np.float32))
    monkeypatch.setattr(P,'row_anchor',lambda d,r:d.episodes[r['episode_index']]['state'][r['start']])
    monkeypatch.setattr(P.E,'coordinate_diagnostics',lambda *a:{'synthetic':True})
    arrays={'reference':ref,'anchors':anchors};statuses=[]
    for pi in range(2):
        for i in range(len(rows)):
            z=np.zeros((1,10,7),np.float32)
            if i==0:z[0,0,0]=np.nan;status='nonfinite_return'
            else:arrays[P.key(pi,i,'raw')]=ref[i].copy();status='finite'
            arrays[P.key(pi,i,'normalized')]=z;statuses.append(status)
    result=P.summarize(data,rows,arrays,statuses,plans,2)
    assert not result['passed'] and result['nonfinite_rows']==[0] and result['metrics'] is None
    arrays.pop(P.key(1,4,'normalized'))
    with pytest.raises(ValueError):P.summarize(data,rows,arrays,statuses,plans,2)

def test_raw_envelope_negative_never_clips(numerical,monkeypatch):
    P,rows,data,ref,anchors,plans=numerical
    monkeypatch.setattr(P.D,'decode_prediction',lambda d,z,s,p,t:np.asarray(z+s[:7],np.float32))
    monkeypatch.setattr(P,'row_anchor',lambda d,r:d.episodes[r['episode_index']]['state'][r['start']])
    monkeypatch.setattr(P.E,'coordinate_diagnostics',lambda *a:{'synthetic':True})
    z=np.zeros((1,10,7),np.float32);z[0,9,0]=10
    a={'reference':ref,'anchors':anchors};statuses=[]
    for i in range(5):a[P.key(0,i,'normalized')]=z.copy();a[P.key(0,i,'raw')]=ref[i]+z[0];statuses.append('finite')
    before={k:v.tobytes() for k,v in a.items()};result=P.summarize(data,rows,a,statuses,plans,1)
    assert not result['passed'] and result['raw_gates']['urdf']['violating_coordinates']==5
    assert before=={k:v.tobytes() for k,v in a.items()}

def test_plain_predict_callback_cannot_be_actual_model():
    P=module()
    with pytest.raises(ValueError):P.require_predictor(lambda obs:np.zeros((1,10,7)),None,{'checkpoint_binding':{}})

@pytest.mark.parametrize('kind',['probes','test'])
def test_paths_cannot_be_renamed_or_inside_other_run(tmp_path,kind):
    P=module();run=tmp_path.resolve()
    assert P.output_path(run,kind)==run/'postlock'/kind
    with pytest.raises(ValueError):P.require_output(run,kind,run/'another')

@pytest.fixture
def collection(tmp_path,monkeypatch,numerical):
    import copy
    P,rows,data,ref,anchors,plans=numerical;run=tmp_path.resolve()/'run';run.mkdir()
    source=run/'original.json';P.write_new(source,{'synthetic':True})
    lock={'path':str(source),'sha256':P.file_sha(source)};cp={'checkpoint_path':str(run/'model/selected'),'data_hash':'synthetic'}
    info=dict(run=run,lock_binding=lock,checkpoint_binding=cp,dataset_binding={'data_hash':'synthetic'},config={},artifact_sha256={str(source):P.file_sha(source)})
    data.indices={'train':[(0,i) for i in range(5)],'test':[(0,i) for i in range(5)]}
    context=dict(info,dataset=data,rows=rows,reference=ref,anchors=anchors,plans=plans,posture={},kind='probes')
    def ctx(lb,kind,reservation=None,*,locked=None):
        value=dict(context,kind=kind,rows=[dict(r,split='test',split_index=r['train_index']) if kind=='test' else r for r in rows])
        value['reservation_binding']=reservation
        return value
    monkeypatch.setattr(P,'_locked',lambda binding:copy.deepcopy(info))
    monkeypatch.setattr(P,'_context',ctx)
    monkeypatch.setattr(P,'source_binding',lambda:{str(source):P.file_sha(source)})
    monkeypatch.setattr(P,'_tokenizer',lambda context:None)
    monkeypatch.setattr(P,'noncontent_predicates',lambda context:None)
    monkeypatch.setattr(P.D,'decode_prediction',lambda d,z,s,p,t:np.asarray(z+s[:7],np.float32))
    monkeypatch.setattr(P,'row_anchor',lambda d,r:d.episodes[r['episode_index']]['state'][r['start']])
    monkeypatch.setattr(P.E,'coordinate_diagnostics',lambda *a:{'synthetic':True})
    monkeypatch.setattr(P.D,'make_observation_batch',lambda d,s,idx,t:{'state':data.episodes[0]['state'][idx].copy()})
    monkeypatch.setattr(P,'row_anchor',lambda d,r:d.episodes[r['episode_index']]['state'][r['start']])
    monkeypatch.setattr(P.E,'coordinate_diagnostics',lambda *a:{'synthetic':True})
    monkeypatch.setattr(P,'_passed_probes',lambda info:{})
    monkeypatch.setattr(P,'_probe_end',lambda *a:None)
    monkeypatch.setattr(P,'_producer',lambda lb,kind,out:dict(process_identity={'pid':123,'start_ticks':45},command=P.collection_command(lb,kind,out),source_binding=P.source_binding()))
    model=object.__new__(P.SelectedPredictor);model.checkpoint_binding=cp;model.tokenizer=object();model.source_binding=P.source_binding()
    calls=[]
    def infer(self,obs):calls.append(obs['state'].copy());return np.zeros((1,10,7),np.float32)
    monkeypatch.setattr(P.SelectedPredictor,'__call__',infer)
    return SimpleNamespace(P=P,run=run,lock=lock,info=info,model=model,calls=calls,context=context,source=source)

def test_collection_two_pass_readonly_verify_and_no_retry(collection):
    f=collection;out=f.P.output_path(f.run,'probes')
    rec=f.P.collect_probes(f.lock,out,f.model.tokenizer,f.model,cancelled=lambda:False)
    assert rec['result']['passed'] and rec['model_call_count']==10 and len(f.calls)==10
    before={str(p):p.read_bytes() for p in f.run.rglob('*') if p.is_file()}
    verified=f.P.verify_probes(out/'receipt.json')
    assert verified['receipt_binding']['sha256']==f.P.file_sha(out/'receipt.json')
    assert before=={str(p):p.read_bytes() for p in f.run.rglob('*') if p.is_file()} and len(f.calls)==10
    with pytest.raises((ValueError,FileExistsError)):f.P.collect_probes(f.lock,out,f.model.tokenizer,f.model,cancelled=lambda:False)

@pytest.mark.parametrize('kind',['exception','cancel','input_mutation','output_shape','late_receipt','late_failure'])
def test_failure_retains_prefix_and_never_verifies(collection,monkeypatch,kind):
    f=collection;P=f.P;out=P.output_path(f.run,'probes');stop=[False];count=[0];original=P.SelectedPredictor.__call__;write=P.write_new
    def infer(self,obs):
        count[0]+=1
        if count[0]==3:
            if kind=='exception':raise RuntimeError('synthetic failure')
            if kind=='cancel':stop[0]=True
            if kind=='input_mutation':obs['state'][0,0]+=.2
            if kind=='output_shape':return np.zeros((1,9,7),np.float32)
        return original(self,obs)
    def late(path,value):
        result=write(path,value)
        if Path(path)==out/'receipt.json':
            if kind=='late_receipt':Path(path).write_bytes(Path(path).read_bytes()+b' ')
            if kind=='late_failure':(out/'failure.json').symlink_to(out/'missing')
        return result
    monkeypatch.setattr(P.SelectedPredictor,'__call__',infer);monkeypatch.setattr(P,'write_new',late)
    with pytest.raises((ValueError,RuntimeError,InterruptedError)):P.collect_probes(f.lock,out,f.model.tokenizer,f.model,cancelled=lambda:stop[0])
    assert (out/'failed_numeric.npz').exists()
    with pytest.raises((ValueError,FileNotFoundError)):P.verify_probes(out/'receipt.json')

def test_test_reserved_once_and_claim_cannot_use_new_output(collection):
    f=collection;P=f.P;reservation=P.reserve_test(f.lock)
    value=P.read_binding(reservation,{})
    assert value['schema']=='wam_pi05_v11_bounded_final_test_reservation_v1' and value['test_split']=='test'
    with pytest.raises(FileExistsError):P.reserve_test(f.lock)
    out=P.output_path(f.run,'test');r=P.collect_test(reservation,out,f.model.tokenizer,f.model,cancelled=lambda:False)
    assert r['model_call_count']==5 and len(f.calls)==5
    assert P.verify_test(out/'receipt.json')['result']['passed']
    with pytest.raises((ValueError,FileExistsError)):P.collect_test(reservation,f.run/'new-test',f.model.tokenizer,f.model,cancelled=lambda:False)

@pytest.mark.parametrize('kind',['exit','pid','signals','late','command'])
def test_actual_exit_requires_clean_original_owned_guard(tmp_path,monkeypatch,kind):
    P=module();folder=tmp_path.resolve();receipt=folder/'receipt.json';lock={'path':str(folder/'lock'),'sha256':'a'*64}
    producer=dict(process_identity={'pid':123,'start_ticks':45},command=['synthetic'],source_binding={})
    P.write_new(receipt,{'producer':producer});guard=folder/'guard.py';guard.write_text('# synthetic guard')
    monkeypatch.setattr(P,'GUARD_SHA',P.file_sha(guard));gp=folder/'collection.process.json'
    g=dict(command=['synthetic'],process_identity=dict(pid=123,pgid=123,sid=123,start_ticks='45'),passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[])
    value=dict(schema=P.EXECUTION_SCHEMA,process_identity=producer['process_identity'],command=producer['command'],exit_code=0,process_group_drained=True,errors=[],deferred_signals=[],receipt_sha256=P.file_sha(receipt),guard_source_binding=P.binding(guard))
    if kind=='exit':g['exit_code']=19
    if kind=='pid':g['process_identity']['pid']=124
    if kind=='signals':g['signals']=['SIGTERM']
    if kind=='command':g['command']=['foreign']
    P.write_new(gp,g);value['guard_evidence_binding']=P.binding(gp);P.write_new(folder/'execution.json',value)
    if kind=='late':(folder/'collection.late_interrupt.json').symlink_to(folder/'missing')
    with pytest.raises(ValueError):P.verify_execution(receipt)

def attach_exit(f,out,monkeypatch):
    P=f.P;receipt=out/'receipt.json';producer=P.read_binding(P.binding(receipt),{})['producer']
    guard=f.run/'guard.py'
    if not guard.exists():guard.write_text('# explicit synthetic guard')
    monkeypatch.setattr(P,'GUARD_SHA',P.file_sha(guard));gp=out/'collection.process.json'
    identity=producer['process_identity'];pid=identity['pid']
    g=dict(command=producer['command'],process_identity=dict(pid=pid,pgid=pid,sid=pid,start_ticks=str(identity['start_ticks'])),passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[])
    P.write_new(gp,g);P.write_new(out/'execution.json',dict(schema=P.EXECUTION_SCHEMA,process_identity=identity,command=producer['command'],exit_code=0,process_group_drained=True,errors=[],deferred_signals=[],receipt_sha256=P.file_sha(receipt),guard_source_binding=P.binding(guard),guard_evidence_binding=P.binding(gp)))

def completed_pair(f,monkeypatch,negative=False):
    P=f.P
    if negative:monkeypatch.setattr(P.SelectedPredictor,'__call__',lambda self,obs:np.full((1,10,7),10,np.float32))
    probes=P.output_path(f.run,'probes');P.collect_probes(f.lock,probes,f.model.tokenizer,f.model,cancelled=lambda:False)
    reservation=P.reserve_test(f.lock);test=P.output_path(f.run,'test');P.collect_test(reservation,test,f.model.tokenizer,f.model,cancelled=lambda:False)
    attach_exit(f,probes,monkeypatch);attach_exit(f,test,monkeypatch)
    return probes/'receipt.json',test/'receipt.json'

@pytest.mark.parametrize('negative',[False,True])
def test_full_admission_reconstructed_and_negative_bars_service(collection,monkeypatch,negative):
    f=collection;P=f.P;pr,tr=completed_pair(f,monkeypatch,negative);out=f.run/'postlock/admission.json'
    result=P.artifact_admission(pr,tr,out)
    assert result['service_eligible'] is not negative and result['candidate_admitted'] is not negative
    assert P.verify_admission(out,P.file_sha(out))['status']==result['status']
    assert not result['physical_acceptance'] and not result['population_generalization']
    with pytest.raises(FileExistsError):P.artifact_admission(pr,tr,out)

def test_missing_actual_exit_cannot_be_admitted(collection,monkeypatch):
    f=collection;P=f.P;pr,tr=completed_pair(f,monkeypatch);tr.with_name('execution.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):P.artifact_admission(pr,tr,f.run/'postlock/admission.json')

@pytest.mark.parametrize('mutation',['receipt','guard_late','source'])
def test_admission_publish_rechecks_original_evidence(collection,monkeypatch,mutation):
    f=collection;P=f.P;pr,tr=completed_pair(f,monkeypatch);out=f.run/'postlock/admission.json';old=P.write_new
    def late(path,value):
        value=old(path,value)
        if Path(path)==out:
            if mutation=='receipt':Path(path).write_bytes(Path(path).read_bytes()+b' ')
            elif mutation=='guard_late':tr.with_name('collection.late_interrupt.json').symlink_to(f.run/'absent')
            else:f.source.write_text('{}\n')
        return value
    monkeypatch.setattr(P,'write_new',late)
    with pytest.raises(ValueError):P.artifact_admission(pr,tr,out)
    assert out.with_suffix('.invalid.json').exists()
    with pytest.raises(ValueError):P.verify_admission(out)

def test_started_evidence_missing_rejects(collection):
    f=collection;P=f.P;out=P.output_path(f.run,'probes');P.collect_probes(f.lock,out,f.model.tokenizer,f.model,cancelled=lambda:False)
    (out/'started.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):P.verify_probes(out/'receipt.json')

def test_test_claim_must_be_original_run_path(collection):
    f=collection;P=f.P;rb=P.reserve_test(f.lock);out=P.output_path(f.run,'test');P.collect_test(rb,out,f.model.tokenizer,f.model,cancelled=lambda:False)
    claim=f.run/'FINAL_TEST_INFERENCE_STARTED.json';copy=f.run/'copied_claim.json';copy.write_bytes(claim.read_bytes())
    receipt=out/'receipt.json';r=P.read_binding(P.binding(receipt),{});r['test_claim_binding']=P.binding(copy)
    r['artifact_sha256'].pop(str(claim));r['artifact_sha256'][str(copy)]=P.file_sha(copy);receipt.write_bytes(P.E.json_bytes(r));claim.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):P.verify_test(receipt)

def test_both_pass_gate_reports_include_later_violations(numerical,monkeypatch):
    P,rows,data,ref,anchors,plans=numerical;monkeypatch.setattr(P.D,'decode_prediction',lambda d,z,s,p,t:np.asarray(z+s[:7],np.float32))
    monkeypatch.setattr(P,'row_anchor',lambda d,r:d.episodes[r['episode_index']]['state'][r['start']])
    monkeypatch.setattr(P.E,'coordinate_diagnostics',lambda *a:{'synthetic':True})
    arrays={'reference':ref,'anchors':anchors};statuses=['finite']*10
    for pi in range(2):
        for i in range(5):
            z=np.zeros((1,10,7),np.float32)
            if pi==1 and i==4:z[0,9,0]=10
            arrays[P.key(pi,i,'normalized')]=z;arrays[P.key(pi,i,'raw')]=ref[i]+z[0]
    r=P.summarize(data,rows,arrays,statuses,plans,2)
    assert not r['passed'] and not r['replay_equal']
    assert r['raw_gates']['urdf']['violating_coordinates']==1 and r['total_coordinate_count']==700

def test_late_failure_after_readonly_collection_rebuild_bars_admission(collection,monkeypatch):
    f=collection;P=f.P;pr,tr=completed_pair(f,monkeypatch);original=P.verify_execution;calls=[0]
    def late(path):
        result=original(path);calls[0]+=1
        if calls[0]==4:pr.with_name('cancelled.json').symlink_to(f.run/'absent')
        return result
    monkeypatch.setattr(P,'verify_execution',late)
    with pytest.raises(ValueError):P._admission(pr,tr)

def test_actual_nonfinite_complete_negative_keeps_all_calls(collection,monkeypatch):
    f=collection;P=f.P;count=[0]
    def infer(self,obs):
        count[0]+=1;v=np.zeros((1,10,7),np.float32)
        if count[0] in (1,6):v[0,0,0]=np.nan
        return v
    monkeypatch.setattr(P.SelectedPredictor,'__call__',infer);out=P.output_path(f.run,'probes')
    r=P.collect_probes(f.lock,out,f.model.tokenizer,f.model,cancelled=lambda:False)
    assert count[0]==10 and not r['result']['passed'] and r['result']['replay_equal']
    with np.load(out/'numeric.npz',allow_pickle=False) as z:
        assert P.key(0,0,'raw') not in z and np.isnan(z[P.key(0,0,'normalized')][0,0,0])
    assert not P.verify_probes(out/'receipt.json')['result']['passed']

def test_cpu_default_missing_receipt_never_initializes_or_reserves(tmp_path,monkeypatch,capsys):
    import sys
    P=module();out=tmp_path/'absent';monkeypatch.setattr(sys,'argv',['postlock.py','--lock',str(tmp_path/'lock'),'--lock-sha256','a'*64,'--kind','test','--output',str(out)])
    monkeypatch.setattr(P,'_locked',lambda *a:pytest.fail('Missing receipt default must not reconstruct or reserve'))
    with pytest.raises(SystemExit) as e:P.main()
    assert e.value.code==2 and not out.exists() and 'not_ready' in capsys.readouterr().out

def test_external_file_import_keeps_original_training_package(tmp_path):
    import importlib.util
    P=module();p=tmp_path/'postlock.py';p.write_bytes(Path(P.__file__).read_bytes())
    spec=importlib.util.spec_from_file_location('outside_training_project_postlock',p);external=importlib.util.module_from_spec(spec);spec.loader.exec_module(external)
    assert external.T is P.T and external.E is P.E and external.D is P.D
    assert external.collection_command({'path':'/lock','sha256':'a'*64},'probes','/out')[:3]==[str(P.ROOT/'vendor/openpi/.venv/bin/python'),'-B',str(P.PRODUCER_FILE)]

@pytest.mark.parametrize('kind',['manifest','checkpoint','source'])
def test_official_predictor_identity_rechecked_before_actual_calls(collection,kind):
    f=collection;P=f.P
    if kind=='manifest':f.model.checkpoint_binding=dict(f.model.checkpoint_binding,data_hash='foreign')
    elif kind=='checkpoint':f.model.checkpoint_binding=dict(f.model.checkpoint_binding,checkpoint_path='/other')
    else:f.model.source_binding={'foreign':'a'*64}
    with pytest.raises(ValueError):P.collect_probes(f.lock,P.output_path(f.run,'probes'),f.model.tokenizer,f.model,cancelled=lambda:False)
    assert not f.calls

def test_guard_positive_and_final_source_replacement_rejected(collection,monkeypatch):
    f=collection;P=f.P;pr,tr=completed_pair(f,monkeypatch)
    assert P.verify_execution(pr)['binding']['path']==str(pr.with_name('execution.json'))
    old=P.D._recheck;changed=[False]
    def mutate(artifacts):
        if str(pr.with_name('execution.json')) in artifacts and not changed[0]:
            changed[0]=True;pr.write_bytes(pr.read_bytes()+b' ')
        return old(artifacts)
    monkeypatch.setattr(P.D,'_recheck',mutate)
    with pytest.raises(ValueError):P.verify_execution(pr)

def test_external_admission_cli_is_readonly_and_uses_expected_sha(tmp_path,monkeypatch,capsys):
    import sys
    P=module();p=tmp_path/'admission.json';calls=[]
    def verify(path,expected_sha=None):calls.append((Path(path),expected_sha));return {'service_eligible':False,'artifact_sha256':{'synthetic':'early'}}
    monkeypatch.setattr(P,'verify_admission',verify)
    monkeypatch.setattr(P,'_locked',lambda *a:pytest.fail('CLI must delegate only the readonly admission verifier'))
    monkeypatch.setattr(sys,'argv',['postlock.py','--verify-admission',str(p),'--admission-sha256','a'*64])
    P.main();assert calls==[(p,'a'*64)] and not p.exists() and 'service_eligible' in capsys.readouterr().out

def test_guard_late_marker_during_final_admission_hash_is_rejected(collection,monkeypatch):
    f=collection;P=f.P;pr,tr=completed_pair(f,monkeypatch);out=f.run/'postlock/admission.json';original=P.D._recheck;mutated=[False]
    def late(artifacts):
        result=original(artifacts)
        if out.exists() and str(pr.with_name('execution.json')) in artifacts and str(tr.with_name('execution.json')) in artifacts and not mutated[0]:
            mutated[0]=True;pr.with_name('collection.late_interrupt.json').symlink_to(f.run/'absent')
        return result
    monkeypatch.setattr(P.D,'_recheck',late)
    with pytest.raises(ValueError):P.artifact_admission(pr,tr,out)
    assert out.with_suffix('.invalid.json').exists()

def test_end_check_uses_existing_pins_not_another_scientific_rebuild(collection,monkeypatch):
    f=collection;P=f.P;out=P.output_path(f.run,'probes');out.parent.mkdir();out.mkdir()
    checked=[]
    monkeypatch.setattr(P,'_locked',lambda *a:pytest.fail('No repeated full selection scientific reconstruction at end boundary'))
    monkeypatch.setattr(P,'noncontent_predicates',lambda context:checked.append('fresh'),raising=False)
    P._fresh(f.context,out,f.lock,'probes',None,None)
    assert checked
    f.source.write_text('{}\n')
    with pytest.raises(ValueError):P._fresh(f.context,out,f.lock,'probes',None,None)

@pytest.mark.parametrize('mutation',['none','terminal_cancel','owner_cancel','terminal_alias','environment'])
def test_light_end_predicates_keep_actual_cancellation_and_path_checks(tmp_path,monkeypatch,mutation):
    P=module();run=tmp_path.resolve();segments=run/'training_segments';segments.mkdir();seg=segments/'current.json'
    terminal=dict(schema=P.T.VERSION,training_run_path=str(run),training_complete=True,interrupted=False,error=None,segment=str(seg),**P.T.clock(4000))
    P.write_new(seg,terminal);P.write_new(run/'training_run.json',terminal)
    owner=run/'owner';owner.mkdir();life=owner/'lifecycle.json';P.write_new(life,{'synthetic_scientific_prerequisite':True});P.write_new(run/'TRAINING_OWNER.json',{'lifecycle':P.binding(life)})
    artifacts={str(p):P.file_sha(p) for p in (seg,run/'training_run.json',run/'TRAINING_OWNER.json',life)}
    context=dict(run=run,artifact_sha256=artifacts,runtime={'environment':'original'},producer_sources={})
    monkeypatch.setattr(P,'runtime_predicates',lambda:{'environment':'changed' if mutation=='environment' else 'original'});monkeypatch.setattr(P,'source_binding',lambda:{})
    if mutation=='terminal_cancel':seg.with_suffix('.cancelled.json').symlink_to(run/'absent')
    elif mutation=='owner_cancel':(owner/'late_interrupt.json').symlink_to(run/'absent')
    elif mutation=='terminal_alias':p=run/'training_run.json';p.unlink();p.symlink_to(seg)
    if mutation=='none':P.noncontent_predicates(context)
    else:
        with pytest.raises(ValueError):P.noncontent_predicates(context)

def test_public_receipt_first_pin_before_scientific_reconstruction(tmp_path,monkeypatch):
    P=module();base=tmp_path.resolve();run=base/'run';(run/'selection').mkdir(parents=True);source=base/'source';source.write_text('# synthetic')
    public=base/'public_checkpoint_verified.json';P.write_new(public,{'files':[]})
    token=base/'public-assets/paligemma_tokenizer.model';token.parent.mkdir();token.write_bytes(b'synthetic tokenizer')
    tr=base/'public_tokenizer_verified.json';P.write_new(tr,{'sha256':P.file_sha(token)})
    config={'dataset_binding':{'data_hash':'test'},'assets':{'checkpoint_receipt_sha256':P.file_sha(public),'tokenizer_sha256':P.file_sha(token)}}
    cp={'checkpoint_path':str(run/'model/selected'),'config_sha256':P.digest(config)}
    configpath=run/'configuration.json';P.write_new(configpath,{'config':config})
    selection={'run':str(run),'selected':{'checkpoint_binding':cp},'dataset_binding':config['dataset_binding'],'artifact_sha256':{str(configpath):P.file_sha(configpath)}}
    lock=run/'selection/selection.lock.json';P.write_new(lock,{'selection':selection});P.write_new(run/'selection/selection.json',selection);P.write_new(run/P.E.MARKER,{'synthetic':True})
    lb=P.binding(lock)
    monkeypatch.setattr(P,'ROOT',base);monkeypatch.setattr(P,'runtime_predicates',lambda:{});monkeypatch.setattr(P,'source_binding',lambda:{str(source):P.file_sha(source)});monkeypatch.setattr(P,'noncontent_predicates',lambda c:None);monkeypatch.setattr(P.T,'checkpoint_binding',lambda p:cp)
    def reconstruct(*a):tr.write_bytes(tr.read_bytes()+b' ');return {'selection':selection}
    monkeypatch.setattr(P.E,'verify_selection_lock',reconstruct)
    with pytest.raises(ValueError):P._locked(lb)


def test_each_admission_operation_reconstructs_science_once(collection,monkeypatch):
    f=collection;P=f.P;pr,tr=completed_pair(f,monkeypatch);out=f.run/'postlock/admission.json';old=P._locked;calls=[]
    def counted(lb):calls.append(lb);return old(lb)
    monkeypatch.setattr(P,'_locked',counted)
    P.artifact_admission(pr,tr,out)
    assert calls==[f.lock]
    calls.clear();P.verify_admission(out,P.file_sha(out));assert calls==[f.lock]


def test_admission_rechecks_noncontent_after_publication(collection,monkeypatch):
    f=collection;P=f.P;pr,tr=completed_pair(f,monkeypatch);out=f.run/'postlock/admission.json'
    def predicate(context):
        if out.exists():raise ValueError('Synthetic late terminal cancellation')
    monkeypatch.setattr(P,'noncontent_predicates',predicate)
    with pytest.raises(ValueError,match='terminal cancellation'):P.artifact_admission(pr,tr,out)
    assert out.with_suffix('.invalid.json').exists()


def test_cli_owned_operation_reuses_initial_context_without_second_lock_verifier(collection,monkeypatch):
    f=collection;P=f.P
    monkeypatch.setattr(P,'_locked',lambda *a:pytest.fail('Same CLI operation must not repeat full scientific reconstruction'))
    reservation=P._reserve_test(f.info)
    out=P.output_path(f.run,'test')
    r=P._collect(f.lock,'test',out,f.model.tokenizer,f.model,lambda:False,reservation,locked=f.info)
    assert r['result']['passed'] and r['model_call_count']==5


@pytest.mark.parametrize('when',['before','publish','end'])
@pytest.mark.parametrize('dangling',[False,True])
def test_admission_invalid_marker_is_authoritative_and_preserved(collection,monkeypatch,when,dangling):
    f=collection;P=f.P;pr,tr=completed_pair(f,monkeypatch);out=f.run/'postlock/admission.json'
    invalid=out.with_suffix('.invalid.json');original_write=P.write_new;original_end=P._end_admission
    def mark():
        if dangling:invalid.symlink_to(f.run/'missing')
        else:invalid.write_text('original invalid evidence\n')
    if when=='before':mark()
    def publish(path,value):
        result=original_write(path,value)
        if Path(path)==out and when=='publish':mark()
        return result
    def end(value,info):
        result=original_end(value,info)
        if out.exists() and when=='end':mark()
        return result
    monkeypatch.setattr(P,'write_new',publish);monkeypatch.setattr(P,'_end_admission',end)
    with pytest.raises(ValueError,match='invalidation|publication invalid'):P.artifact_admission(pr,tr,out)
    assert invalid.is_symlink() if dangling else invalid.read_text()=='original invalid evidence\n'
    if when=='before':assert not out.exists()


def test_v11_external_identity_and_observation_only_source():
    import ast
    P=module();tree=ast.parse(Path(P.__file__).read_text())
    assert P.SCHEMA=='wam_pi05_v11_postlock_collection_v1'
    assert P.EXECUTION_SCHEMA=='wam_pi05_v11_postlock_process_exit_v1'
    assert P.RESERVATION_SCHEMA=='wam_pi05_v11_bounded_final_test_reservation_v1'
    assert str(P.PRODUCER_FILE).endswith('/diagnostics/pi05_v11_postlock_runtime_v1/postlock.py')
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='D']
    assert not any(n.func.attr=='make_batch' for n in calls)
    assert sum(n.func.attr=='make_observation_batch' for n in calls)==2
    assert not any(isinstance(n,ast.Attribute) and n.attr=='NORMALIZATION_SHA256' for n in ast.walk(tree))
    assert all(len(n.args)==5 for n in calls if n.func.attr=='decode_prediction')


@pytest.mark.parametrize('mutation',['none','target','episode_target','later_target','phase','later_phase','episode','split','anchor'])
def test_row_anchor_matches_full_chunk_without_loading_labels(mutation):
    P=module();e=dict(episode_id='R4_fixture',split='test',target='R4',targets=np.array(['R4']*10),phases=np.array(['FREE']*10),state=np.zeros((10,41),np.float32))
    r=dict(episode_index=0,episode_id=e['episode_id'],start=0,phase='FREE',target='R4',split='test');d=SimpleNamespace(episodes=[e],loaded_scopes=('train','test'))
    if mutation=='target':r['target']='L4'
    if mutation=='episode_target':e['target']='L4'
    if mutation=='later_target':e['targets'][9]='L4'
    if mutation=='phase':r['phase']='HOLD'
    if mutation=='later_phase':e['phases'][9]='HOLD'
    if mutation=='episode':r['episode_id']='other'
    if mutation=='split':r['split']='validation'
    if mutation=='anchor':e['state'][0,0]=np.nan
    if mutation=='none':assert P.row_anchor(d,r).tobytes()==e['state'][0].tobytes()
    else:
        with pytest.raises(ValueError):P.row_anchor(d,r)


@pytest.mark.parametrize('passed',[False,True])
def test_L5_reservation_requires_passing_actual_probes(tmp_path,monkeypatch,passed):
    P=module();run=tmp_path.resolve();probe=run/'postlock/probes/receipt.json';probe.parent.mkdir(parents=True);P.write_new(probe,{'synthetic':True})
    lock={'path':'/synthetic/lock','sha256':'a'*64};cp={'synthetic':'checkpoint'};db={'synthetic':'data'};source=run/'source';source.write_text('original')
    info=dict(run=run,lock_binding=lock,checkpoint_binding=cp,dataset_binding=db,artifact_sha256={str(source):P.file_sha(source)})
    monkeypatch.setattr(P,'_verify',lambda path,kind,locked=None:dict(result={'passed':passed},lock_binding=lock,checkpoint_binding=cp,dataset_binding=db,artifact_sha256={str(probe):P.file_sha(probe)},receipt_binding=P.binding(probe)))
    called=[];monkeypatch.setattr(P,'verify_execution',lambda path:called.append(path) or {'artifact_sha256':{}})
    if passed:
        result=P._reserve_test(info);assert result['path']==str(run/'FINAL_TEST_STARTED.json') and called
    else:
        with pytest.raises(ValueError):P._reserve_test(info)
        assert not (run/'FINAL_TEST_STARTED.json').exists()


def test_L5_probe_prerequisite_rejects_missing_exit(tmp_path,monkeypatch):
    P=module();run=tmp_path.resolve();info=dict(run=run,lock_binding={},checkpoint_binding={},dataset_binding={},artifact_sha256={})
    monkeypatch.setattr(P,'_verify',lambda *a,**k:dict(result={'passed':True},lock_binding={},checkpoint_binding={},dataset_binding={},artifact_sha256={}))
    with pytest.raises((ValueError,FileNotFoundError)):P._reserve_test(info)
    assert not (run/'FINAL_TEST_STARTED.json').exists()


def test_context_uses_bound_saved_norm_and_honest_scopes(tmp_path,monkeypatch):
    P=module();run=tmp_path.resolve();identity=dict(manifest_binding={'path':'/manifest'},normalization_file_binding={'path':'/v11norm','sha256':'f'*64})
    info=dict(run=run,lock_binding={'synthetic':1},dataset_binding=identity,artifact_sha256={})
    dataset=SimpleNamespace(qualification_scope_binding={'artifact_sha256':{}});calls=[]
    monkeypatch.setattr(P,'_reservation',lambda *a:None);monkeypatch.setattr(P,'_passed_probes',lambda *a:{})
    monkeypatch.setattr(P.D,'load_dataset',lambda *a,**k:calls.append((a,k)) or dataset)
    monkeypatch.setattr(P.D,'dataset_binding',lambda d:identity);monkeypatch.setattr(P.E,'build_report_schedule',lambda *a:[])
    monkeypatch.setattr(P.E,'references',lambda *a:(None,None));monkeypatch.setattr(P.E,'posture_contract',lambda:({},{}))
    P._context(info['lock_binding'],'test',{'reservation':1},locked=info)
    args,kwargs=calls[0];assert args==('/manifest',) and kwargs['scopes']==('test',)
    assert kwargs['expected_normalization_sha256']=='f'*64 and kwargs['test_authorization']==dict(selection_lock_binding=info['lock_binding'],reservation_binding={'reservation':1})


def test_original_guard_and_publication_functions_ast_identical():
    import ast
    P=module();new=ast.parse(Path(P.__file__).read_text());oldpath=Path(P.__file__).parents[1]/'auscultation_pi05_v10/postlock.py'
    text=oldpath.read_text().replace('pi05_v10','pi05_v11').replace('V10','V11').replace('wam_pi05_v11_recovery_','wam_pi05_v11_bounded_')
    old=ast.parse(text)
    names=('binding','read_binding','merge','output_path','require_output','no_failure','check_cancel','runtime_predicates','noncontent_predicates','collection_command','_producer','require_predictor','numeric_write','numeric_read','_claim','verify_execution','_end_admission','artifact_admission','verify_admission','main')
    def functions(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=functions(old),functions(new)
    assert all(a[n]==b[n] for n in names)


def test_bound_codec_result_never_claims_learned_safety(collection):
    f=collection;P=f.P;out=P.output_path(f.run,'probes');r=P.collect_probes(f.lock,out,f.model.tokenizer,f.model,cancelled=lambda:False)
    assert r['result']['bounded_by_codec'] is True and r['result']['learned_safety_validated'] is False
    assert r['result']['coordinate_diagnostics']=={'synthetic':True}
    assert r['result']['clipping'] is False and r['result']['physical_acceptance'] is False


def test_reservation_publication_cannot_rebind_earlier_probe(collection,monkeypatch):
    f=collection;P=f.P;probe=f.run/'probe_evidence.json';P.write_new(probe,{'original':True});h=P.file_sha(probe)
    monkeypatch.setattr(P,'_passed_probes',lambda info:{str(probe):h});old=P.write_new
    def late(path,value):
        result=old(path,value)
        if Path(path)==f.run/'FINAL_TEST_STARTED.json':probe.write_bytes(probe.read_bytes()+b' ')
        return result
    monkeypatch.setattr(P,'write_new',late)
    with pytest.raises(ValueError):P._reserve_test(f.info)


def test_actual_codec_full70_decode_byte_accounting(monkeypatch):
    from sim.auscultation_pi05_v11 import action_encoding as A
    P=module();bounds=A.build_bounds_contract(Path(P.__file__).resolve().parents[2]);limits=bounds['targets']['R4']
    q=((np.asarray(limits['lower_rad'])+np.asarray(limits['upper_rad']))/2).astype(np.float32)
    episodes=[];rows=[]
    for i,phase in enumerate(P.E.PHASES):
        state=np.zeros((10,41),np.float32);state[:,:7]=q
        e=dict(episode_id='R4_synthetic_'+phase,split='train',target='R4',targets=np.full(10,'R4'),phases=np.full(10,phase),state=state,actions=np.tile(q,(10,1)),starts=[0])
        episodes.append(e);rows.append(dict(row_id=i,episode_index=i,episode_id=e['episode_id'],start=0,phase=phase,target='R4',split='train'))
    stats=A.fit_stats(episodes,bounds);data=SimpleNamespace(episodes=episodes,loaded_scopes=('train',),normalization={'residual_actions':stats})
    # Numerical adapter fixture only: no claim to production source qualification.
    monkeypatch.setattr(P.D,'_state_ok',lambda data:None)
    plans={'R4':{'joint_lower_rad':limits['original_lower_rad'],'joint_upper_rad':limits['original_upper_rad']}}
    arrays=dict(reference=np.tile(q,(5,10,1)),anchors=np.tile(q,(5,1)))
    for pi in range(2):
        for i,r in enumerate(rows):
            z=A.encode_chunk(episodes[i]['actions'],q,r['phase'],'R4',stats)[None]
            arrays[P.key(pi,i,'normalized')]=z;arrays[P.key(pi,i,'raw')]=P.D.decode_prediction(data,z[0],episodes[i]['state'][0],r['phase'],'R4')
    result=P.summarize(data,rows,arrays,['finite']*10,plans,2)
    assert result['passed'] and result['bounded_by_codec'] and result['coordinate_diagnostics']['coordinate_count']==700
    for h in range(10):
        for j in range(7):
            raw=arrays[P.key(1,4,'raw')];old=raw[h,j];raw[h,j]=np.nextafter(old,np.float32(np.inf))
            with pytest.raises(ValueError,match='decode differs'):P.summarize(data,rows,arrays,['finite']*10,plans,2)
            raw[h,j]=old


def test_observation_only_collection_never_inverse_encodes_heldout(collection,monkeypatch):
    f=collection;P=f.P
    monkeypatch.setattr(P.D,'make_batch',lambda *a:pytest.fail('Heldout model input must not encode labels'))
    rb=P.reserve_test(f.lock);out=P.output_path(f.run,'test');P.collect_test(rb,out,f.model.tokenizer,f.model,cancelled=lambda:False)
    assert P.verify_test(out/'receipt.json')['result']['passed'] and len(f.calls)==5


@pytest.mark.parametrize('mutation',['foreign_lock','foreign_checkpoint','foreign_dataset','cancelled_probe'])
def test_prerequisite_probes_cannot_be_substituted(tmp_path,monkeypatch,mutation):
    P=module();run=tmp_path.resolve();probe=run/'postlock/probes/receipt.json';probe.parent.mkdir(parents=True);P.write_new(probe,{'synthetic':True})
    info=dict(run=run,lock_binding={'original':'lock'},checkpoint_binding={'original':'cp'},dataset_binding={'original':'data'},artifact_sha256={})
    value=dict(result={'passed':True},lock_binding=info['lock_binding'],checkpoint_binding=info['checkpoint_binding'],dataset_binding=info['dataset_binding'],artifact_sha256={str(probe):P.file_sha(probe)})
    if mutation.startswith('foreign_'):value[{'foreign_lock':'lock_binding','foreign_checkpoint':'checkpoint_binding','foreign_dataset':'dataset_binding'}[mutation]]={'foreign':True}
    else:(probe.parent/'cancelled.json').symlink_to(run/'absent')
    monkeypatch.setattr(P,'_verify',lambda *a,**k:value);monkeypatch.setattr(P,'verify_execution',lambda p:{'artifact_sha256':{}})
    with pytest.raises(ValueError):P._reserve_test(info)
    assert not (run/'FINAL_TEST_STARTED.json').exists()


def test_complete_parent_function_delta_is_explicit():
    import ast
    P=module();oldpath=Path(P.__file__).parents[1]/'auscultation_pi05_v10/postlock.py'
    text=oldpath.read_text().replace('pi05_v10','pi05_v11').replace('V10','V11').replace('wam_pi05_v11_recovery_','wam_pi05_v11_bounded_')
    def bodies(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old,new=bodies(text),bodies(Path(P.__file__).read_text())
    assert set(new)-set(old)=={'row_anchor','_passed_probes','_probe_end'} and not set(old)-set(new)
    assert {n for n in old if old[n]!=new[n]}=={'_reserve_test','_context','summarize','_fresh','_collect','_verify','_admission'}
