"""Synthetic numerical/receipt tests; qualification/checkpoint/model prerequisites are stubs."""
import copy,hashlib,importlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');return p

def module():
    try:return importlib.import_module('sim.auscultation_pi05_v10.parity')
    except ModuleNotFoundError as e:
        if e.name!='sim.auscultation_pi05_v10.parity':raise
        pytest.fail('V10 parity API not implemented')

@pytest.fixture
def fixture(tmp_path,monkeypatch):
    P=module();D=P.D;actual_context=P._context
    from sim.auscultation_pi05_v7 import training_data as O,coverage_pilot_evaluation as Q
    from sim.auscultation_pi05_v8 import pair_admission as F
    design=json.loads((ROOT/'results/diagnostics/pi05_recovery_data_training_design_v1/machine_spec_v2.json').read_text())
    norm=json.loads((ROOT/'results/pi05_v7_repair/horizon_pair_B_completed_mirror_v1/verified_files/runs/horizon-pair-original8-v1-B/configuration.json').read_text())['normalization']
    episodes=[];indices=[]
    for ei,eid in enumerate(design['data']['train_original8']):
        q=np.array([.01,-.5,-.3,1.,.2,.1,.2],np.float32);state=np.zeros((250,41),np.float32);state[:,:7]=q;state[:,7]=np.arange(250)*np.float32(1e-5)
        episodes.append(dict(episode_id=eid,split='train',state=state,actions=np.tile(q,(250,1)),phases=np.repeat(D.PHASES,50),targets=np.repeat(eid.split('_')[0],250)))
        indices.extend((ei,s) for s in range(241) if s//50==(s+9)//50)
    view=SimpleNamespace(episodes=episodes,indices={'train':indices},normalization=norm)
    dataset=SimpleNamespace(episodes=episodes+[{'episode_id':'R5_observed_armature02_v1','split':'validation'}],indices={'train':indices,'validation':[(8,0)],'test':[]},normalization=norm,loaded_scopes=('train','validation'),data_hash='d'*64)
    dataset.rgb=lambda ei:np.full((250,2,2,3),ei,np.uint8)
    identity={'schema':D.BINDING_SCHEMA,'data_hash':dataset.data_hash,'normalization_sha256':D.NORMALIZATION_SHA256}
    monkeypatch.setattr(D,'dataset_binding',lambda d:copy.deepcopy(identity))
    monkeypatch.setattr(D,'original_train_view',lambda d:view)
    monkeypatch.setattr(O,'observation',lambda rgb,state,phase,target,tokenizer,normalization:{'state':np.asarray(state,np.float32),'rgb':rgb,'phase':np.asarray([D.PHASES.index(phase)],np.int32)})
    calls=[]
    def batch(d,split,rows,tokenizer):
        assert split=='train' and len(rows)==1
        calls.append(rows[0]);ei,start=indices[rows[0]];e=episodes[ei]
        return O.stack_observations([O.observation(d.rgb(ei)[start],e['state'][start],str(e['phases'][start]),str(e['targets'][start]),tokenizer,norm)]),np.zeros((1,10,7),np.float32)
    monkeypatch.setattr(D,'make_batch',batch)
    monkeypatch.setattr(D,'decode_prediction',lambda d,z,state,phase:D.codec.decode_chunk(z,np.asarray(state)[:7],phase,norm['residual_actions'],mode=D.codec.B))
    rows120=Q.build_diagnostic_schedule(view);rows160=F.fixed_rows(view)
    refs120,anchors120=Q.references(view,rows120);refs160,anchors160=F.fixed_references(view,rows160)
    z=np.zeros((120,10,7),np.float32);raw=np.stack([D.decode_prediction(dataset,v,episodes[r['episode_index']]['state'][r['start']],r['phase']) for v,r in zip(z,rows120)])
    history=tmp_path/'history';history.mkdir();numeric120=history/'predictions.npz';np.savez(numeric120,normalized_predictions=z,predictions=raw,reference=refs120,anchors=anchors120)
    parent=copy.deepcopy(design['parent_checkpoint_binding']);parent['checkpoint_path']=str(tmp_path/'B/model/step-000000007300')
    report=write(history/'report.json',dict(schema='wam_horizon_pair_train_numeric_report_v1',mode=D.codec.B,checkpoint_binding=parent,sampler=Q.SAMPLER,rows=rows120,diagnostic_schedule_sha256=Q.digest(rows120),numeric_sha256=sha(numeric120),local_step=7300,optimizer_step=7300,train_predictions_only=True,heldout_predictions_generated=False,closed_loop_validated=False,initialization_baseline_only=False))
    fixed=tmp_path/'fixed';fixed.mkdir();plan=write(fixed/'plan.json',dict(schema=F.PLAN_SCHEMA,b_checkpoint_binding=parent,fixed160_rows=rows160,fixed160_rows_sha256=Q.digest(rows160),sampler=F.SAMPLER))
    arrays={'reference':refs160,'anchors':anchors160};contracts=[]
    for i,r in enumerate(rows160):
        obs,_=batch(dataset,'train',[r['train_index']],None);contracts.append(F.input_contract(obs));zz=np.zeros((10,7),np.float32);e=episodes[r['episode_index']];rr=D.decode_prediction(dataset,zz,e['state'][r['start']],'FREE')
        for prefix in ('pass','replay'):arrays[f'{prefix}_{i:03d}_normalized']=zz;arrays[f'{prefix}_{i:03d}_raw']=rr
    numeric=fixed/'numeric.npz';np.savez(numeric,**arrays)
    producer={'command':['synthetic_only'],'process_identity':{'pid':1,'start_ticks':2},'source_file_binding':{'path':str(Path(F.__file__).resolve()),'sha256':sha(F.__file__)}}
    receipt=write(fixed/'receipt.json',dict(schema=F.FIXED_SCHEMA,role='B_candidate',mode=D.codec.B,checkpoint_binding=parent,sampler=F.SAMPLER,plan_binding={'path':str(plan),'sha256':sha(plan)},rows_sha256=Q.digest(rows160),numeric_file_binding={'path':str(numeric),'sha256':sha(numeric)},numeric_array_contracts={k:F.array_contract(v) for k,v in arrays.items()},reference_array_binding=F.array_contract(refs160),anchors_array_binding=F.array_contract(anchors160),actual_consumed_input_leaf_hashes=contracts+copy.deepcopy(contracts),per_row_pass_and_replay_status=[dict(pass_name=p,row_id=i,status='finite') for p in ('pass','replay') for i in range(160)],producer_execution_binding=producer,status='finite',errors=[],replay_byte_comparison=True,canonical_codec_decode_comparison=True,frozen120_first_free_overlap_comparison=True))
    guard=write(fixed/'collection.process.json',{'synthetic_only':True});gs=write(fixed/'guard.py.json',{'synthetic_only':True})
    execution=write(fixed/'execution.json',dict(guard_evidence_binding={'path':str(guard),'sha256':sha(guard)},guard_source_binding={'path':str(gs),'sha256':sha(gs)}))
    monkeypatch.setattr(F,'verify_execution',lambda receipt,producer:{'path':str(execution),'sha256':sha(execution)})
    monkeypatch.setattr(P,'_history_paths',lambda parent:(report,receipt))
    design['parent_checkpoint_binding']=parent;classes=design['warm_start']['step0_evidence_classes'];classes['original120']['historical_output_reference'].update(report_sha256=sha(report),numeric_sha256_from_report=sha(numeric120));classes['fixed160']['historical_input_receipt_binding']['sha256']=sha(receipt)
    source=write(tmp_path/'context_source.json',{'synthetic_only':True});artifacts={str(source):sha(source)}
    run=tmp_path/'run';run.mkdir();binding=dict(schema='wam_pi05_v10_recovery_checkpoint_binding_v1',checkpoint_path=str(run/'model/step-000000000000'),local_step=0,optimizer_step=0,schedule_cursor=0,cumulative_trainable_update_age=7300,data_hash=dataset.data_hash,source_sha256={})
    context=dict(dataset_binding=identity,checkpoint_binding=binding,parent_checkpoint_binding=parent,design=design,artifact_sha256=artifacts,tokenizer=None)
    monkeypatch.setattr(P,'_context',lambda d,b:copy.deepcopy(context))
    calls.clear();predictcalls=[]
    def predict(obs):predictcalls.append(copy.deepcopy(obs));return np.zeros((1,10,7),np.float32)
    return SimpleNamespace(P=P,D=D,dataset=dataset,binding=binding,run=run,output=run/'step0_parity.json',predict=predict,predictcalls=predictcalls,calls=calls,report=report,fixed=receipt,design=design,context=context,source=source,guard=guard,rows120=rows120,rows160=rows160,batch=batch,actual_context=actual_context)

def test_complete_two_evidence_classes_and_readonly_verifier(fixture):
    f=fixture;r=f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:False)
    assert r['schema']==f.P.SCHEMA and r['passed'] is True and r['model_call_count']==280 and len(f.predictcalls)==280
    assert len(r['overlap'])==8 and r['historical_fixed160_input_records']==320
    reconstruction=json.loads((f.run/'step0_parity/original120_reconstruction.json').read_text())
    assert reconstruction['historical_input_consumption_receipt'] is False and len(reconstruction['rows'])==120
    before={str(p):sha(p) for p in f.run.rglob('*') if p.is_file()}
    verified=f.P.verify_receipt(f.output,f.dataset,f.binding)
    assert verified['receipt_binding']=={'path':str(f.output),'sha256':sha(f.output)}
    assert verified['artifact_sha256']==dict(r['artifact_sha256'],**{str(f.output):sha(f.output)})
    assert {k:v for k,v in verified.items() if k not in ('artifact_sha256','receipt_binding')}=={k:v for k,v in r.items() if k!='artifact_sha256'}
    assert before=={str(p):sha(p) for p in f.run.rglob('*') if p.is_file()} and len(f.predictcalls)==280

@pytest.mark.parametrize('kind',['different','nonfinite','float64','wrong_shape','exception'])
def test_actual_bad_output_retained_and_blocks_success(fixture,kind):
    f=fixture;count=[0]
    def predict(obs):
        count[0]+=1
        if count[0]!=5:return f.predict(obs)
        if kind=='exception':raise FloatingPointError('actual synthetic numeric failure')
        x=np.zeros((1,10,7),np.float32)
        if kind=='different':x[0,0,0]=.1
        elif kind=='nonfinite':x[0,0,0]=np.nan
        elif kind=='float64':x=x.astype(np.float64)
        else:x=x[:,:,:6]
        return x
    with pytest.raises((ValueError,FloatingPointError)):f.P.run_step0(f.output,f.dataset,f.binding,None,predict,cancelled=lambda:False)
    assert not f.output.exists() and (f.run/'step0_parity/failure.json').is_file()
    with np.load(f.run/'step0_parity/failed_numeric.npz',allow_pickle=False) as z:
        assert 'original120_003_normalized' in z
        if kind!='exception':assert 'original120_004_normalized' in z

@pytest.mark.parametrize('kind',['reference','normalized','raw','replay_input','status','missing_numeric','checkpoint','sampler','rows'])
def test_rebound_incompatible_history_fails_before_model(fixture,kind):
    f=fixture;r=json.loads(f.fixed.read_text())
    if kind=='replay_input':r['actual_consumed_input_leaf_hashes'][160][0]['sha256']='0'*64
    elif kind=='status':r['per_row_pass_and_replay_status'][200]['status']='nonfinite_return'
    elif kind=='checkpoint':r['checkpoint_binding']['local_step']=7301
    elif kind=='sampler':r['sampler']['seed']=918
    elif kind=='rows':r['rows_sha256']='0'*64
    else:
        n=Path(r['numeric_file_binding']['path'])
        with np.load(n,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        key={'reference':'reference','normalized':'replay_001_normalized','raw':'pass_001_raw','missing_numeric':'pass_001_raw'}[kind]
        if kind=='missing_numeric':a.pop(key)
        else:a[key]=a[key].copy();a[key].flat[0]+=.001
        np.savez(n,**a);r['numeric_file_binding']['sha256']=sha(n);r['numeric_array_contracts']={k:f.P.F.array_contract(v) for k,v in a.items()}
    write(f.fixed,r);f.design['warm_start']['step0_evidence_classes']['fixed160']['historical_input_receipt_binding']['sha256']=sha(f.fixed)
    with pytest.raises(ValueError):f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:False)
    assert not f.predictcalls

def test_new_input_disagreement_not_relabelled_historical(fixture,monkeypatch):
    f=fixture
    def changed(*a,**kw):
        obs,z=f.batch(*a,**kw);obs['state']=obs['state'].copy();obs['state'][0,7]+=.01;return obs,z
    monkeypatch.setattr(f.D,'make_batch',changed)
    with pytest.raises(ValueError,match='input'):f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:False)
    assert not f.output.exists() and not f.predictcalls

def test_cancellation_before_work_and_during_final_write(fixture,monkeypatch):
    f=fixture
    with pytest.raises(InterruptedError):f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:True)
    assert not f.predictcalls and not (f.run/'step0_parity').exists()
    stop=[False];original=f.P.write_new
    def late(path,value):
        result=original(path,value)
        if Path(path)==f.output:stop[0]=True
        return result
    monkeypatch.setattr(f.P,'write_new',late)
    with pytest.raises(InterruptedError):f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:stop[0])
    assert f.output.with_suffix('.cancelled.json').exists()
    with pytest.raises(ValueError,match='cancel'):f.P.verify_receipt(f.output,f.dataset,f.binding)

@pytest.mark.parametrize('kind',['history','source','input_mutation'])
def test_consumption_rebind_is_rejected(fixture,kind):
    f=fixture;count=[0]
    def predict(obs):
        count[0]+=1
        if count[0]==3:
            if kind=='history':f.report.write_text('{}\n')
            elif kind=='source':f.source.write_text('{}\n')
            else:obs['state'][0,7]+=.01
        return f.predict(obs)
    with pytest.raises(ValueError):f.P.run_step0(f.output,f.dataset,f.binding,None,predict,cancelled=lambda:False)
    assert not f.output.exists()

@pytest.mark.parametrize('kind',['numeric','input','reconstruction','late_guard','late_cancel','symlink_cancel'])
def test_readonly_verify_rejects_tampered_or_cancelled_evidence(fixture,kind):
    f=fixture;f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:False)
    if kind=='numeric':(f.run/'step0_parity/numeric.npz').write_bytes(b'changed')
    elif kind=='input':(f.run/'step0_parity/consumed_inputs.json').write_text('{}\n')
    elif kind=='reconstruction':(f.run/'step0_parity/original120_reconstruction.json').write_text('{}\n')
    elif kind=='late_guard':f.guard.with_name('collection.late_interrupt.json').write_text('{}\n')
    elif kind=='late_cancel':f.output.with_suffix('.cancelled.json').write_text('{}\n')
    else:f.output.with_suffix('.cancelled.json').symlink_to(f.run/'missing')
    with pytest.raises(ValueError):f.P.verify_receipt(f.output,f.dataset,f.binding)

def test_no_reuse_or_cross_run_output(fixture):
    f=fixture
    with pytest.raises(ValueError):f.P.run_step0(f.run/'foreign.json',f.dataset,f.binding,None,f.predict,cancelled=lambda:False)
    f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:False)
    with pytest.raises(ValueError):f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:False)


@pytest.mark.parametrize('kind',['consumed_json','reconstruction_json','numeric'])
def test_actual_persisted_bytes_rechecked_before_pass(fixture,monkeypatch,kind):
    f=fixture;write_original=f.P.write_new;numeric_original=f.P._numeric_write
    def tamper_json(path,value):
        result=write_original(path,value)
        target='consumed_inputs.json' if kind=='consumed_json' else 'original120_reconstruction.json'
        if Path(path).name==target and kind!='numeric':Path(path).write_text('{}\n')
        return result
    def tamper_numeric(path,arrays):
        result=numeric_original(path,arrays)
        if Path(path).name=='numeric.npz' and kind=='numeric':
            changed=dict(arrays);changed['original120_000_normalized']=changed['original120_000_normalized'].copy();changed['original120_000_normalized'].flat[0]+=.01
            with Path(path).open('wb') as out:np.savez(out,**changed)
        return result
    monkeypatch.setattr(f.P,'write_new',tamper_json);monkeypatch.setattr(f.P,'_numeric_write',tamper_numeric)
    with pytest.raises(ValueError):f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:False)
    assert not f.output.exists()

@pytest.mark.parametrize('key,value',[('local_step',1),('optimizer_step',True),('schedule_cursor',4),('cumulative_trainable_update_age',7301),('schema','old'),('data_hash','changed')])
def test_real_context_rejects_clock_schema_data_before_checkpoint(fixture,monkeypatch,key,value):
    f=fixture;from sim.auscultation_pi05_v10 import train as T
    monkeypatch.setattr(T,'checkpoint_binding',lambda *a:pytest.fail('Invalid context reached checkpoint'))
    b=dict(f.binding,normalization_sha256=f.D.digest(f.dataset.normalization));b[key]=value
    with pytest.raises(ValueError):f.actual_context(f.dataset,b)

def test_real_context_rejects_fresh_checkpoint_mismatch(fixture,monkeypatch):
    f=fixture;from sim.auscultation_pi05_v10 import train as T
    b=dict(f.binding,normalization_sha256=f.D.digest(f.dataset.normalization))
    monkeypatch.setattr(T,'checkpoint_binding',lambda *a:dict(b,arrays_sha256='changed'))
    with pytest.raises(ValueError,match='weights'):f.actual_context(f.dataset,b)

def test_no_old_full_loader_or_pair_reconstruction_and_source_cycle(fixture,monkeypatch):
    f=fixture
    from sim.auscultation_pi05_v6 import train_openpi as old
    from sim.auscultation_pi05_v8 import training_data as bd
    from sim.auscultation_pi05_v10 import train as T
    for module,name in [(old,'load_dataset'),(bd,'load_dataset'),(f.P.F,'_verify_plan'),(f.P.F,'verify_pair_admission')]:
        monkeypatch.setattr(module,name,lambda *a,**k:pytest.fail('Forbidden full reconstruction'))
    assert T.source_hashes()['sim/auscultation_pi05_v10/parity.py']==sha(f.P.__file__)
    f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:False)
    f.P.verify_receipt(f.output,f.dataset,f.binding)


def test_verifier_returns_early_receipt_pin_not_rebound_file(fixture,monkeypatch):
    f=fixture;f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:False)
    original=f.P._context;called=[False]
    def rebound(*a):
        result=original(*a)
        if not called[0]:
            called[0]=True;v=json.loads(f.output.read_text());v['scope']='late change';write(f.output,v)
        return result
    monkeypatch.setattr(f.P,'_context',rebound)
    with pytest.raises(ValueError):f.P.verify_receipt(f.output,f.dataset,f.binding)


@pytest.mark.parametrize('kind',['main_receipt','failure','dangling_failure'])
def test_final_publication_cannot_adopt_replaced_receipt_or_late_failure(fixture,monkeypatch,kind):
    f=fixture;original=f.P.write_new
    def late(path,value):
        result=original(path,value)
        if Path(path)==f.output:
            if kind=='main_receipt':
                changed=dict(value,passed=False);write(path,changed)
            elif kind=='failure':(f.run/'step0_parity/failure.json').write_text('{}\n')
            else:(f.run/'step0_parity/failure.json').symlink_to(f.run/'missing')
        return result
    monkeypatch.setattr(f.P,'write_new',late)
    with pytest.raises(ValueError):f.P.run_step0(f.output,f.dataset,f.binding,None,f.predict,cancelled=lambda:False)
    assert len(f.predictcalls)==280 and (f.run/'step0_parity/failed_numeric.npz').is_file()
    with pytest.raises(ValueError):f.P.verify_receipt(f.output,f.dataset,f.binding)
