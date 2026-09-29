"""Synthetic software contracts only; no actual condition, model or GPU."""
import copy,importlib,json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from sim.auscultation_pi05_v9 import action_encoding as C


def api():
    try:return importlib.import_module('sim.auscultation_pi05_v9.comparison_admission')
    except ModuleNotFoundError:pytest.fail('Conditional comparison/admission is not implemented')


@lru_cache(maxsize=1)
def _numeric_fixture_base():
    M=api();episodes=[];indices=[]
    for i in range(8):
        n=250;q=np.array([0,-.5,0,1,0,0,0],np.float32)
        state=np.zeros((n,41),np.float32);state[:,:7]=q+np.arange(n,dtype=np.float32)[:,None]*np.float32(.0001)
        e=dict(episode_id='synthetic_'+str(i),split='train',state=state,actions=(state[:,:7]+np.float32(.00005)),
            phases=np.repeat(('FREE','APPROACH','CONTACT','HOLD','RETRACT'),50),targets=np.repeat(('R' if i<4 else 'L')+str(i%4+1),n),starts=[p*50+j for p in range(5) for j in range(41)])
        episodes.append(e);indices.extend((i,j) for j in e['starts'])
    stats=C.fit_stats(episodes,C.C)
    d=SimpleNamespace(episodes=episodes,indices={'train':indices},mode=C.C,normalization={'residual_actions':stats})
    rows=M.fixed_rows(d);ref,anchors=M.fixed_references(d,rows);arrays={'reference':ref,'anchors':anchors};statuses=[];inputs=[]
    for label in ('pass','replay'):
        for i,row in enumerate(rows):
            z=C.encode_chunk(ref[i],anchors[i],'FREE',stats,mode=C.C)
            arrays[f'{label}_{i:03d}_normalized']=z
            arrays[f'{label}_{i:03d}_raw']=C.decode_chunk(z,anchors[i],'FREE',stats,mode=C.C)
            statuses.append(dict(pass_name=label,row_id=i,status='finite'))
            inputs.append([dict(path='state',**M.array_contract(np.asarray([i],np.float32)))])
    overlap={i:dict(normalized=arrays[f'pass_{i:03d}_normalized'],raw=arrays[f'pass_{i:03d}_raw']) for i in range(0,160,20)}
    return M,d,rows,arrays,statuses,inputs,overlap


def numeric_fixture():
    value=_numeric_fixture_base();return (value[0],*copy.deepcopy(value[1:]))


def test_new_schemas_and_explicit_collection_command(tmp_path):
    M=api();assert M.PLAN_SCHEMA=='wam_rational_q4_comparison_plan_v1'
    assert M.FIXED_SCHEMA=='wam_rational_q4_fixed160_receipt_v1'
    assert M.ADMISSION_SCHEMA=='wam_rational_q4_comparison_admission_v1'
    p=tmp_path/'plan.json';p.write_text('{}');b=M.binding(p)
    argv=M.collection_command(b,'C_candidate',tmp_path/'output')
    assert argv[1:4]==['-B','-m','sim.auscultation_pi05_v9.comparison_admission'] and argv[-1]=='--collect'
    assert argv[argv.index('--plan-sha256')+1]==b['sha256']
    with pytest.raises(M.UnsupportedRoute):M.collection_command(b,'B_comparator',tmp_path/'output')


def test_exact160_rational_decode_all70_replay_overlap_and_bounds():
    M,d,rows,a,s,inputs,overlap=numeric_fixture()
    result=M.verify_numeric(d,rows,a,s,inputs,inputs[:160],overlap)
    assert result['status']=='finite' and result['bounds']['coordinate_count']==11200
    assert result['bounds']['all_passed'] and result['replay_byte_comparison']
    assert len(rows)==160 and [r['start'] for r in rows[:20]]==list(range(20))


@pytest.mark.parametrize('joint',range(7))
def test_all70_coordinates_reconstructed_not_only_q4(joint):
    M,d,rows,a,s,inputs,overlap=numeric_fixture()
    for label in ('pass','replay'):
        key=f'{label}_159_raw';a[key]=a[key].copy();a[key][9,joint]=np.nextafter(a[key][9,joint],np.float32(np.inf))
    with pytest.raises(ValueError):M.verify_numeric(d,rows,a,s,inputs,inputs[:160],overlap)


@pytest.mark.parametrize('kind',['input','anchor','label','replay','missing','overlap','shape'])
def test_invalid_evidence_cannot_be_a_negative_result(kind):
    M,d,rows,a,s,inputs,overlap=numeric_fixture();canonical=copy.deepcopy(inputs[:160])
    if kind=='input':inputs[319]=[]
    elif kind=='anchor':a['anchors']=a['anchors']+.0001
    elif kind=='label':a['reference']=a['reference']+.0001
    elif kind=='replay':a['replay_159_normalized']=a['replay_159_normalized']+.001
    elif kind=='missing':del a['pass_019_raw']
    elif kind=='overlap':overlap[20]['raw']=overlap[20]['raw']+.001
    else:a['pass_019_normalized']=a['pass_019_normalized'][:9]
    with pytest.raises(ValueError):M.verify_numeric(d,rows,a,s,inputs,canonical,overlap)


def test_C_actual_numerical_failure_is_negative_but_all8_overlap_mandatory():
    M,d,rows,a,s,inputs,overlap=numeric_fixture()
    for label,offset in (('pass',0),('replay',160)):
        a[f'{label}_019_normalized']=np.full((10,7),np.nan,np.float32);del a[f'{label}_019_raw'];s[offset+19]['status']='nonfinite_return'
    r=M.verify_numeric(d,rows,a,s,inputs,inputs[:160],overlap)
    assert r['status']=='numerical_failure' and r['bounds'] is None and r['metrics'] is None
    for label,offset in (('pass',0),('replay',160)):
        a[f'{label}_020_normalized']=np.full((10,7),np.nan,np.float32);del a[f'{label}_020_raw'];s[offset+20]['status']='nonfinite_return'
    with pytest.raises(ValueError):M.verify_numeric(d,rows,a,s,inputs,inputs[:160],overlap)


def test_original_strict_bounds_and_research_are_separate():
    M=api();p=np.zeros((160,10,7),np.float32);p[:,:,1]=-.5;p[:,:,3]=1
    assert M.describe(p,p,p[:,0])['bounds']['all_passed']
    p[159,9,0]=np.float32(M.UPPER[0]+.001)
    assert M.describe(p,np.zeros_like(p),p[:,0])['bounds']['violation_count']==1
    zeros={k:0. for k in ('worst_initial_first_max_abs_rad','macro_first_mae_rad','macro_full_mae_rad')}
    comparison=M.research_comparison(zeros,{k:1. for k in zeros})
    assert not comparison['affects_admission'] and comparison['macro_first_mae_rad']['ratio'] is None


def test_strict_binding_conflict_and_consumed_bytes(tmp_path):
    M=api();p=tmp_path/'e.json';p.write_text('{"a":1}');b=M.binding(p);closure={}
    assert M.read_pinned(b)=={'a':1};M.add_binding(closure,b)
    with pytest.raises(ValueError):M.add_binding(closure,dict(b,sha256='0'*64))
    p.write_text('{}')
    with pytest.raises(ValueError):M.read_pinned(b)
    with pytest.raises(ValueError):M.recheck(closure)


def test_negative_B_route_fails_closed_before_native_relabel(tmp_path,monkeypatch):
    M=api();p=tmp_path/'condition.json';p.write_text(json.dumps(dict(schema=M.trigger.SCHEMA,condition_satisfied=True,route='negative_B_selection')))
    monkeypatch.setattr(M.trigger,'verify_receipt',lambda *a:json.loads(p.read_text()))
    with pytest.raises(M.UnsupportedRoute):M.verify_condition(M.binding(p))


def test_output_tree_overlap_and_existing_output_rejected(tmp_path):
    M=api();source=tmp_path/'immutable/source.py';source.parent.mkdir();source.write_text('# bound')
    closure={str(source):M.file_sha(source)}
    with pytest.raises(ValueError):M.fresh_output(source.parent/'new',closure,protected_trees=[source.parent])
    safe=tmp_path/'fresh';assert M.fresh_output(safe,closure,protected_trees=[source.parent])==safe
    with pytest.raises(ValueError):M.fresh_output(safe,closure,protected_trees=[source.parent])


def test_actual_guard_contract_rejects_flags_pid_or_clock_types():
    M=api();cmd=['synthetic'];ident={'pid':123,'start_ticks':456}
    value=dict(command=cmd,passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[],process_identity=dict(ident,pgid=123,sid=123))
    assert M.verify_guard(value,cmd,ident)==ident
    for key,bad in [('passed',False),('exit_code',False),('process_group_drained',False),('deferred_signals',['SIGTERM']),('signals',['SIGINT'])]:
        with pytest.raises(ValueError):M.verify_guard(dict(value,**{key:bad}),cmd,ident)
    with pytest.raises(ValueError):M.verify_guard(value,['other'],ident)


def test_external_exit_is_required_source_bound_and_cancel_safe(tmp_path,monkeypatch):
    M=api();r=tmp_path/'receipt.json';r.write_text('{}');producer=dict(process_identity={'pid':123,'start_ticks':456},command=['synthetic'])
    with pytest.raises(ValueError):M.verify_execution(M.binding(r),producer,{})
    guard_source=tmp_path/'guard.py';guard_source.write_text('# synthetic guard');monkeypatch.setattr(M,'GUARD_SHA256',M.file_sha(guard_source))
    guard=dict(command=producer['command'],passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[],process_identity=dict(producer['process_identity'],pgid=123,sid=123))
    gp=tmp_path/'collection.process.json';gp.write_text(json.dumps(guard))
    exit_value=dict(schema=M.PROCESS_EXIT_SCHEMA,**producer,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[],receipt_sha256=M.file_sha(r),guard_evidence_binding=M.binding(gp),guard_source_binding=M.binding(guard_source))
    ep=tmp_path/'execution.json';ep.write_text(json.dumps(exit_value));closure={}
    assert M.verify_execution(M.binding(r),producer,closure)==M.binding(ep)
    assert str(gp) in closure and str(guard_source) in closure
    (tmp_path/'collection.late_interrupt.json').symlink_to(tmp_path/'absent')
    with pytest.raises(ValueError):M.verify_execution(M.binding(r),producer,{})


def test_native_B_receipt_uses_original_verifier_and_retains_role(tmp_path,monkeypatch):
    M=api();p=tmp_path/'original_plan.json';p.write_text('{}');r=tmp_path/'receipt.json';r.write_text('{}');ex=tmp_path/'execution.json';ex.write_text('{}')
    condition=dict(fixed_plan_binding=M.binding(p),fixed_receipt_binding=M.binding(r),artifact_sha256={str(ex):M.file_sha(ex)})
    checkpoint={'local_step':7300};native=dict(receipt={'schema':M.P.FIXED_SCHEMA,'role':'B_candidate','checkpoint_binding':checkpoint,'status':'finite'},receipt_binding=M.binding(r),execution_binding=M.binding(ex))
    calls=[]
    def verify(plan,receipt):calls.append((plan,receipt));return copy.deepcopy(native)
    monkeypatch.setattr(M.P,'verify_fixed160',verify)
    value=M.verify_native_b(condition,checkpoint,{})
    assert calls==[(p,r)] and value==native and value['receipt']['role']=='B_candidate'
    with pytest.raises(ValueError):M.verify_native_b(condition,{'local_step':14600},{})


def test_candidate_gate_ignores_research_and_B_bounds_but_never_C_failure():
    M=api();selected=dict(local_step=1217,eligible=True,bounds={'all_passed':True,'coordinate_count':8400,'violating_coordinates':0},moving_buckets={'passed':16})
    good=dict(status='finite',bounds={'all_passed':True,'coordinate_count':11200,'violation_count':0})
    assert M.admission_decision(selected,good)==[]
    assert M.admission_decision(None,None)==['rejected_no_eligible_C']
    with pytest.raises(ValueError):M.admission_decision(None,good)
    assert M.admission_decision(selected,dict(good,status='numerical_failure',bounds=None))==['rejected_C_numerical_output']
    assert M.admission_decision(selected,dict(good,bounds=dict(good['bounds'],all_passed=False,violation_count=1)))==['rejected_C_fixed160_raw_bound']
    for bad in (dict(selected,local_step=0),dict(selected,eligible=False),dict(selected,moving_buckets={'passed':15})):
        with pytest.raises(ValueError):M.admission_decision(bad,good)


def test_prepare_requires_explicit_lifecycle_and_producer_inventory(tmp_path):
    M=api()
    with pytest.raises(TypeError):M.prepare_comparison({}, {}, tmp_path/'no')
    assert not (tmp_path/'no').exists()


def test_physical_summary_keeps_all13_and_decoded_axes():
    M=api();p=np.zeros((120,10,7),np.float32);ref=p.copy();a=p[:,0].copy()
    rows=[dict(episode_id='e'+str(i//15),phase='FREE',position='first',start=i) for i in range(120)]
    p[119,9,3]=np.float32(-.000001)
    r=M.physical_summary(p,ref,a,rows)
    assert r['per_joint']['full_max_abs_rad'][3]>0
    assert r['raw_bound_violations'][-1]['joint']==3 and r['raw_bound_violations'][-1]['side']=='lower'
    assert r['raw_bound_violations'][-1]['horizon']==9 and r['normalized_values_compared'] is False


def test_default_cli_cannot_collect_or_write(monkeypatch,capsys,tmp_path):
    M=api();p=tmp_path/'p.json';p.write_text('{}');b=M.binding(p);calls=[]
    monkeypatch.setattr(M,'_verify_plan',lambda value:(calls.append(value) or {'status':'ready_for_owned_collection'},{},{},{}))
    monkeypatch.setattr(M,'collect_fixed160',lambda *a:pytest.fail('default initialized collection'))
    M.main(['--plan',b['path'],'--plan-sha256',b['sha256']])
    assert calls==[b] and json.loads(capsys.readouterr().out)['model_initialized'] is False
    assert list(tmp_path.iterdir())==[p]


def test_final_verifier_reconstructs_not_stored_positive(monkeypatch,tmp_path):
    M=api();p=tmp_path/'admission.json'
    r=dict(schema=M.ADMISSION_SCHEMA,plan_binding={'path':'/plan','sha256':'a'*64},b_fixed160_receipt_binding=None,c_fixed160_receipt_binding=None,candidate_admitted=True)
    p.write_text(json.dumps(r));b=M.binding(p)
    monkeypatch.setattr(M,'_reconstruct_final',lambda *a:dict(r,candidate_admitted=False))
    with pytest.raises(ValueError):M.verify_comparison_admission(p,b['sha256'])


def test_final_verifier_rejects_postconsume_rebind(monkeypatch,tmp_path):
    M=api();p=tmp_path/'admission.json'
    r=dict(schema=M.ADMISSION_SCHEMA,plan_binding={},b_fixed160_receipt_binding=None,c_fixed160_receipt_binding=None)
    p.write_text(json.dumps(r));b=M.binding(p)
    def reconstruct(*args):p.write_text(json.dumps(dict(r,changed=True)));return r
    monkeypatch.setattr(M,'_reconstruct_final',reconstruct)
    with pytest.raises(ValueError):M.verify_comparison_admission(p,b['sha256'])


def test_late_current_cancel_after_hash_sweep_rejects(tmp_path,monkeypatch):
    M=api();cancel=tmp_path/'current.cancelled.json';life=tmp_path/'C'/'lifecycle.json';life.parent.mkdir();life.write_text('{}')
    blife=tmp_path/'B'/'lifecycle.json';blife.parent.mkdir();blife.write_text('{}')
    plan=dict(condition_binding={},b_selection_lock_binding={},c_training_lifecycle_binding=M.binding(life),b_training_lifecycle_binding=M.binding(blife),b_evidence_reuse=None)
    contexts={'C':dict(config={'training_run_path':str(tmp_path)},terminal_segment_binding={'cancellation_path':str(cancel)},verified_input_sha256={})}
    monkeypatch.setattr(M,'verify_condition',lambda *a:{'b_selection_lock_binding':{}})
    monkeypatch.setattr(M,'verify_training_lifecycle',lambda *a:None)
    monkeypatch.setattr(M.E,'verify_terminal_segment',lambda *a:None)
    calls=[]
    def check(*a):
        calls.append(1)
        if len(calls)==2:cancel.symlink_to(tmp_path/'missing')
    monkeypatch.setattr(M,'recheck',check)
    with pytest.raises(ValueError):M._finish_checks(plan,contexts,{},[])


def test_all13_primary14600_secondary_original_locks(monkeypatch):
    M=api();steps=list(M.E.EXPECTED_STEPS);locks={arm:{'selection':{'candidates':[{'local_step':s} for s in steps],'selected':{'local_step':7300 if arm=='B' else 1217}}} for arm in ('B','C')}
    def diag(ctx,candidate,arm,closure):
        return dict(arm=arm,step=candidate['local_step'],reference_contract={},anchors_contract={},rows_sha256='same',metrics={'scores':{k:1. if arm=='B' else 2. for k in ('worst_initial_first_max_abs_rad','macro_first_mae_rad','macro_full_mae_rad')}})
    monkeypatch.setattr(M,'_diagnostic',diag)
    p,curve,s=M._diagnostic_comparisons({'B':{},'C':{}},locks,{})
    assert len(curve)==13 and [x['local_step'] for x in curve]==steps
    assert p['B']['step']==p['C']['step']==14600 and s['B']['step']==7300 and s['C']['step']==1217
    assert p['research_claim_criteria']['criteria_passed'] is False and not p['affects_admission']
    locks['C']['selection']['selected']=None
    assert M._diagnostic_comparisons({'B':{},'C':{}},locks,{})[2] is None


def test_negative_C_final_has_no_fallback_no_service(tmp_path,monkeypatch):
    M=api();p=tmp_path/'plan.json';p.write_text('{}');pb=M.binding(p)
    plan=dict(artifact_sha256={},condition_binding={},b_selection_lock_binding={},c_selection_lock_binding={},b_comparator_kind='original_locked_selected',b_checkpoint_binding={'local_step':7300},c_checkpoint_binding=None)
    monkeypatch.setattr(M,'_diagnostic_comparisons',lambda *a:({'fixed':14600},list(range(13)),None))
    monkeypatch.setattr(M,'_finish_checks',lambda *a:None)
    ctx={'C':{'dataset':None}};locks={'C':{'selection':{'selected':None}}}
    r=M._final(pb,plan,ctx,locks,None,None,None)
    assert r['status']=='rejected_no_eligible_C' and r['candidate_admitted'] is r['service_eligible'] is False
    assert r['admitted_checkpoint_binding'] is r['admitted_dataset_codec_binding'] is None
    assert r['all13_step_matched_comparison']==list(range(13)) and not any(r[k] for k in ('physical_accepted','closed_loop_validated','generalization_validated','authorization_granted'))
    with pytest.raises(ValueError):M._final(pb,plan,ctx,locks,None,{},None)


def test_bound_evidence_parent_protected_without_caller_extra(tmp_path):
    M=api();p=tmp_path/'input'/'proof.json';p.parent.mkdir();p.write_text('{}')
    with pytest.raises(ValueError):M.fresh_output(p.parent/'hidden-output',{str(p):M.file_sha(p)})
    assert not (p.parent/'hidden-output').exists()


def _worker_fixture(tmp_path,monkeypatch):
    # All producer/guard identities below are synthetic fixtures, never run evidence.
    M,d,rows,a,s,inputs,overlap=numeric_fixture()
    planpath=tmp_path/'plan.json';planpath.write_text('{}');pb=M.binding(planpath)
    inv=tmp_path/'inventory.json';inv.write_text('{}');source=M.binding(M.__file__)
    plan=dict(artifact_sha256={},c_checkpoint_binding={'local_step':1217},c_dataset_binding={'mode':C.C},c_selection_lock_binding={},fixed160_rows=rows,
        fixed160_rows_sha256=M.digest(rows),diagnostic_source_binding=source,frozen_producer_inventory_binding=M.binding(inv))
    ctx=dict(dataset=d,config={'assets':{},'openpi_commit':'synthetic'},schedule={'numpy_version':np.__version__})
    monkeypatch.setattr(M,'_canonical_inputs',lambda *args:inputs[:160]);monkeypatch.setattr(M,'_overlap',lambda *args:overlap)
    out=tmp_path/'collection';out.mkdir();producer=dict(process_identity={'pid':123,'start_ticks':456},command=M.collection_command(pb,'C_candidate',out),source_file_binding=source)
    env=dict(numpy_version=np.__version__,openpi_python_prefix=str((M.T.ROOT/'vendor/openpi/.venv').resolve()),python_executable='/synthetic/python')
    rb=M._record(pb,plan,ctx,out,a,s,inputs,producer,env)
    guard=tmp_path/'synthetic_guard.py';guard.write_text('# synthetic');monkeypatch.setattr(M,'GUARD_SHA256',M.file_sha(guard))
    gp=out/'collection.process.json';gp.write_text(json.dumps(dict(command=producer['command'],passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[],process_identity=dict(producer['process_identity'],pgid=123,sid=123))))
    ex=dict(schema=M.PROCESS_EXIT_SCHEMA,process_identity=producer['process_identity'],command=producer['command'],exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[],receipt_sha256=rb['sha256'],guard_evidence_binding=M.binding(gp),guard_source_binding=M.binding(guard))
    (out/'execution.json').write_text(json.dumps(ex))
    return M,pb,plan,ctx,rb,out


def test_C_record_reconstructs_actual_numeric_and_external_exit(tmp_path,monkeypatch):
    M,pb,plan,ctx,rb,out=_worker_fixture(tmp_path,monkeypatch);closure={}
    result=M._verify_c_fixed(pb,plan,ctx,rb,closure)
    assert result['receipt']['bounds']['all_passed'] and result['receipt']['role']=='C_candidate'
    assert len(result['receipt']['per_row_pass_and_replay_status'])==320 and (out/'receipt.json').stat().st_mode&0o222==0
    assert str(out/'execution.json') in closure and str(out/'numeric.npz') in closure
    (out/'execution.json').unlink()
    with pytest.raises(ValueError):M._verify_c_fixed(pb,plan,ctx,rb,{})


@pytest.mark.parametrize('kind',['source','numeric','exit','environment','role','checkpoint'])
def test_C_record_rejects_rebound_or_false_provenance(kind,tmp_path,monkeypatch):
    M,pb,plan,ctx,rb,out=_worker_fixture(tmp_path,monkeypatch)
    if kind=='numeric':
        with (out/'numeric.npz').open('ab') as f:f.write(b'tamper')
    elif kind=='exit':
        p=out/'execution.json';v=json.loads(p.read_text());v['exit_code']=False;p.write_text(json.dumps(v))
    else:
        p=out/'receipt.json';v=json.loads(p.read_text());p.chmod(0o644)
        if kind=='source':v['producer_source_binding']['sha256']='f'*64
        elif kind=='environment':v['runtime_environment']['numpy_version']='wrong'
        elif kind=='role':v['role']='B_candidate'
        else:v['checkpoint_binding']={'local_step':14600}
        p.write_text(json.dumps(v));rb=M.binding(p)
    with pytest.raises(ValueError):M._verify_c_fixed(pb,plan,ctx,rb,{})


def _lifecycle_fixture(tmp_path,monkeypatch):
    # Source review and scientific run-context qualification are isolated here;
    # these synthetic process records are not evidence that a process ran.
    M=api();root=tmp_path/'base';root.mkdir();monkeypatch.setattr(M.T,'ROOT',root)
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return M.binding(path)
    run=root/'runs/rational-q4-original8-v1-C';run.mkdir(parents=True);folder=root/'diagnostics/owner';folder.mkdir(parents=True)
    cond=put(root/'inputs/condition.json',{});norm=put(root/'inputs/norm.json',{});spec=put(root/'inputs/spec.json',{})
    manifest=put(root/'datasets/pi05_v6_task_pose_armature02_v1/task_conditioned_bound/manifest.json',{})
    monkeypatch.setattr(M.T,'SPEC_SHA256',spec['sha256']);monkeypatch.setattr(M.T,'NORMALIZATION_FILE_SHA256',norm['sha256']);monkeypatch.setattr(M.D,'MANIFEST_SHA256',manifest['sha256'])
    inventory=put(root/'inventory.json',{})
    guard=put(root/'diagnostics/coverage_pilot_cloud_launcher/process_guard.py',{});monkeypatch.setattr(M,'GUARD_SHA256',guard['sha256'])
    launcher=put(root/'launcher/launch.py',{})
    segment=run/'training_segments/current.json';main=dict(current_segment_path=str(segment),training_complete=True,interrupted=False)
    sb=put(segment,main);terminal=dict(sb,cancellation_path=str(segment.with_suffix('.cancelled.json')))
    core={}
    for name in M.E.CONTEXT_FILES:core[str(run/name)]=put(run/name,main if name=='training_run.json' else {})['sha256']
    cfg=dict(training_run_path=str(run),saved_normalization_binding=norm,source_sha256={'synthetic':'source'})
    ctx=dict(config=cfg,terminal_segment_binding=terminal,verified_input_sha256=core)
    prep=dict(schema='wam_rational_q4_launch_preparation_v1',mode=C.C,output=str(run),condition_receipt=cond,saved_normalization=norm,
        training_source_sha256=cfg['source_sha256'],training_steps=14600,sample_draws=58400,candidate_steps=list(M.E.EXPECTED_STEPS),seed=61,batch_size=4,learning_rate=5e-5,
        launcher_sha256=launcher['sha256'],guard_sha256=guard['sha256'],experiment_spec=spec,manifest=manifest,inventory_path=inventory['path'],inventory_sha256=inventory['sha256'])
    pb=put(root/'inputs/preparation.json',prep);monkeypatch.setattr(M,'_review_sources',lambda *a:None)
    python=str(root/'vendor/openpi/.venv/bin/python')
    cmd=[python,'-u','-B','-m','sim.auscultation_pi05_v9.train','--manifest',manifest['path'],'--mode',C.C,'--output',str(run),'--condition-receipt',cond['path'],'--condition-sha256',cond['sha256'],'--normalization',norm['path'],'--normalization-sha256',norm['sha256']]
    final=[python,'-u','-B',launcher['path'],'--preparation',pb['path'],'--preparation-sha256',pb['sha256'],'--evidence',str(folder),'--verify-completion']
    stages=[]
    for i,(name,command) in enumerate(zip(('cpu_preflight','training','completion_verification'),(cmd+['--verify-only'],cmd,final))):
        v=dict(stage=name,command=command,passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[],process_identity=dict(pid=123+i,start_ticks=str(456+i),pgid=123+i,sid=123+i))
        put(folder/(name+'.process.json'),v);stages.append(v)
    artifacts=dict(core);artifacts[sb['path']]=sb['sha256']
    for step in M.E.EXPECTED_STEPS:
        for name in ('metadata.json','arrays.npz','COMPLETE.json'):
            b=put(run/'model'/('step-%012d'%step)/name,{});artifacts[b['path']]=b['sha256']
        for name in ('report.json','predictions.npz'):
            b=put(run/'diagnostics'/('local_%06d'%step)/name,{});artifacts[b['path']]=b['sha256']
    complete=dict(schema='wam_rational_q4_full_context_completion_v1',verified=True,output=str(run),preparation_sha256=pb['sha256'],source_sha256=cfg['source_sha256'],candidate_steps=list(M.E.EXPECTED_STEPS),terminal_segment_binding=terminal,training_receipt=main,selected=False,model_accepted=False,artifact_sha256=artifacts)
    cb=put(folder/'completion.json',complete)
    put(folder/'dispatch.json',dict(schema='wam_rational_q4_owned_dispatch_v1',evidence=str(folder),preparation_sha256=pb['sha256'],output=str(run),launcher_sha256=launcher['sha256']))
    life=dict(schema='wam_rational_q4_training_lifecycle_v1',mode=C.C,output=str(run),execute=True,completed=True,training_complete=True,errors=[],selected=False,model_accepted=False,gpu_released=True,outer_flock=False,trainer_owns_shared_lock=True,
        stages=stages,preparation=prep,preparation_sha256=pb['sha256'],command=cmd,training_receipt_sha256=core[str(run/'training_run.json')],training_receipt=main,completion_sha256=cb['sha256'])
    lb=put(folder/'lifecycle.json',life);return M,lb,ctx,cond,put


def test_training_owner_full_context_and_current_segment(tmp_path,monkeypatch):
    M,lb,ctx,cond,put=_lifecycle_fixture(tmp_path,monkeypatch);closure={}
    assert M.verify_training_lifecycle(lb,ctx,cond,closure)==lb
    assert len([p for p in closure if '/model/step-' in p])==39
    # Historical cancellation is retained but does not invalidate the clean current segment.
    put(Path(ctx['terminal_segment_binding']['path']).with_name('old.cancelled.json'),{'cancelled':True})
    assert M.verify_training_lifecycle(lb,ctx,cond,{})==lb
    Path(ctx['terminal_segment_binding']['cancellation_path']).symlink_to(tmp_path/'absent')
    with pytest.raises(ValueError):M.verify_training_lifecycle(lb,ctx,cond,{})


@pytest.mark.parametrize('kind',['completed_flag','missing_stage','guard_exit','guard_signal','completion_closure','late_owner','main_rebind'])
def test_training_owner_cannot_be_proved_by_completed_flag(kind,tmp_path,monkeypatch):
    M,lb,ctx,cond,put=_lifecycle_fixture(tmp_path,monkeypatch);life=M.read_pinned(lb);folder=Path(lb['path']).parent
    if kind=='completed_flag':life['completed']=False
    elif kind=='missing_stage':life['stages']=life['stages'][:2]
    elif kind in ('guard_exit','guard_signal'):
        life['stages'][1]['exit_code' if kind=='guard_exit' else 'signals']=1 if kind=='guard_exit' else ['SIGTERM']
        put(folder/'training.process.json',life['stages'][1])
    elif kind=='completion_closure':
        p=folder/'completion.json';v=json.loads(p.read_text());v['artifact_sha256'].pop(next(k for k in v['artifact_sha256'] if '/model/' in k));life['completion_sha256']=put(p,v)['sha256']
    elif kind=='late_owner':(folder/'late_interrupt.json').symlink_to(tmp_path/'absent')
    else:put(Path(ctx['config']['training_run_path'])/'training_run.json',{})
    lb=put(Path(lb['path']),life)
    with pytest.raises(ValueError):M.verify_training_lifecycle(lb,ctx,cond,{})


@pytest.mark.parametrize('bad',[None,'fifo','extra','writable','ancestor','wrong_parent'])
def test_exact375_independent_readonly_inventory(bad,tmp_path,monkeypatch):
    M=api();old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir();files={}
    for i in range(373):
        name='source_%03d.py'%i;(old/name).write_text('# original '+str(i));(new/name).write_bytes((old/name).read_bytes());files[name]=M.file_sha(old/name)
    parent=tmp_path/'parent.json';parent.write_text(json.dumps(dict(source_project=str(old),source_files=files)));pb=M.binding(parent)
    monkeypatch.setattr(M,'PARENT_INVENTORY',parent);monkeypatch.setattr(M,'PARENT_INVENTORY_SHA',pb['sha256']);monkeypatch.setattr(M.T,'source_hashes',lambda:{})
    current=dict(files)
    for name,source in ((M.SOURCE_KEY,Path(M.__file__)),(M.TEST_KEY,Path(__file__))):
        p=new/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(source.read_bytes());current[name]=M.file_sha(p)
    inventory=tmp_path/'inventory.json';value=dict(schema=M.INVENTORY_SCHEMA,parent_inventory=pb,source_project=str(new),source_files=current)
    if bad=='wrong_parent':value['parent_inventory']=dict(pb,sha256='0'*64)
    inventory.write_text(json.dumps(value));ib=M.binding(inventory)
    if bad=='extra':(new/'extra').write_text('unlisted')
    if bad=='fifo':
        import os
        os.mkfifo(new/'unlisted.pipe')
    if bad=='ancestor':(old/'source_000.py').write_text('mutated')
    paths=[new,*new.rglob('*')]
    for p in paths:p.chmod(0o555 if p.is_dir() else 0o444)
    if bad=='writable':new.chmod(0o755)
    try:
        if bad is None:
            inv,own=M.verify_inventory(ib,{})
            assert len(inv['source_files'])==375 and own==M.binding(new/M.SOURCE_KEY)
        else:
            with pytest.raises(ValueError):M.verify_inventory(ib,{})
    finally:
        for p in paths:p.chmod(0o755 if p.is_dir() else 0o644)
