"""Synthetic CPU pair diagnostics; never qualified data, GPU or admission evidence."""
import copy, importlib, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from sim.auscultation_pi05_v8 import action_encoding as C

@pytest.fixture
def P():return importlib.import_module('sim.auscultation_pi05_v8.pair_admission')

def dataset(mode=C.A):
    episodes=[];indices=[];records=[]
    for i in range(8):
        q=np.array([0,-.5,0,1,0,0,0],np.float32);n=35
        state=np.zeros((n,41),np.float32);state[:,:7]=q
        actions=np.tile(q,(n,1))+np.arange(n,dtype=np.float32)[:,None]*np.full((1,7),.0001,np.float32)
        ep=dict(episode_id='synthetic_'+str(i),split='train',state=state,actions=actions,phases=np.repeat('FREE',n),targets=np.repeat(('R' if i<4 else 'L')+str(i%4+1),n),starts=list(range(26)))
        episodes.append(ep);indices.extend((i,j) for j in range(26));records.append(ep)
    # Fit full valid codec five-phase support separately, preserving FREE shape.
    from test_action_encoding import episodes as codec_episodes
    stats=C.fit_stats(codec_episodes(),mode);stats['fit_episode_ids']=[e['episode_id'] for e in episodes]
    return SimpleNamespace(episodes=episodes,indices={'train':indices},mode=mode,normalization={'residual_actions':stats})

def test_rows_exact_first20_train_fit_order_and_fixed_anchor(P):
    d=dataset();rows=P.fixed_rows(d);ref,q=P.fixed_references(d,rows)
    assert len(rows)==160 and ref.shape==(160,10,7) and q.shape==(160,7)
    assert [r['start'] for r in rows[:20]]==list(range(20))
    assert rows[20]['episode_id']==d.episodes[1]['episode_id'] and rows[-1]['ordinal']==19
    assert ref[0,9,0]!=ref[0,0,0] and q[0,0]==0
    rows[0]['start']=1
    with pytest.raises(ValueError):P.fixed_references(d,rows)

@pytest.mark.parametrize('mutation',['duplicate','crossphase','heldout','too_few','fit_order','bad_index'])
def test_bad_qualified_origins_rejected(P,mutation):
    d=dataset()
    if mutation=='duplicate':d.indices['train'].append(d.indices['train'][0])
    if mutation=='crossphase':d.episodes[0]['phases'][5]='HOLD'
    if mutation=='heldout':d.episodes[0]['split']='test'
    if mutation=='too_few':d.indices['train']=[p for p in d.indices['train'] if p[0]!=0 or p[1]<19]
    if mutation=='fit_order':d.normalization['residual_actions']['fit_episode_ids'].reverse()
    if mutation=='bad_index':d.indices['train'][0]=(0,True)
    with pytest.raises(ValueError):P.fixed_rows(d)

def test_array_contracts_preserve_raw_bytes_dtype_shape_and_inputs(P):
    x=np.array([1,np.nan],np.float32);a=P.array_contract(x)
    assert a['dtype']=='float32' and a['shape']==[2]
    assert a!=P.array_contract(x.astype(np.float64))
    got=P.input_contract({'state':np.ones((1,41),np.float32),'images':{'rgb':np.zeros((1,2,2,3),np.uint8)}})
    assert [x['path'] for x in got]==['images/rgb','state']
    with pytest.raises(ValueError):P.input_contract({'bad':np.array([np.inf])})

def numeric_fixture(P):
    d=dataset();rows=P.fixed_rows(d);ref,q=P.fixed_references(d,rows)
    # The stats fixture uses distinct fitting IDs; undo metadata-only change before codec calls.
    from test_action_encoding import episodes
    d.normalization['residual_actions']=C.fit_stats(episodes(),C.A)
    z=np.stack([C.encode_chunk(r,a,'FREE',d.normalization['residual_actions'],mode=C.A) for r,a in zip(ref,q)])
    raw=np.stack([C.decode_chunk(x,a,'FREE',d.normalization['residual_actions'],mode=C.A) for x,a in zip(z,q)])
    arrays={'reference':ref,'anchors':q};statuses=[];inputs=[]
    for label in ('pass','replay'):
        for i in range(160):
            arrays[f'{label}_{i:03d}_normalized']=z[i];arrays[f'{label}_{i:03d}_raw']=raw[i]
            statuses.append(dict(pass_name=label,row_id=i,status='finite'))
            inputs.append([dict(path='state',**P.array_contract(np.array([i],np.float32)))])
    overlap={i:{'normalized':z[i],'raw':raw[i]} for i in range(0,160,20)}
    return d,rows,arrays,statuses,inputs,overlap

def test_exact_decode_full_replay_and_all8_overlap(P):
    d,rows,arrays,statuses,inputs,overlap=numeric_fixture(P)
    result=P.verify_numeric(d,rows,arrays,statuses,inputs,inputs[:160],overlap,'A_comparator')
    assert result['status']=='finite' and result['replay_byte_comparison'] is True
    assert result['bounds']['coordinate_count']==11200

@pytest.mark.parametrize('mutation',['replay','decode','overlap','inputs','reference','missing','statuses'])
def test_numeric_tampering_is_not_negative_control(P,mutation):
    d,rows,arrays,statuses,inputs,overlap=numeric_fixture(P)
    if mutation=='replay':arrays['replay_159_raw']=arrays['replay_159_raw']+.001
    if mutation=='decode':arrays['pass_015_raw']=arrays['pass_015_raw']+.001;arrays['replay_015_raw']=arrays['pass_015_raw']
    if mutation=='overlap':overlap[20]['raw']=overlap[20]['raw']+.001
    if mutation=='inputs':inputs[159]=[]
    if mutation=='reference':arrays['reference']=arrays['reference']+.001
    if mutation=='missing':del arrays['pass_154_normalized']
    if mutation=='statuses':statuses.pop()
    with pytest.raises(ValueError):P.verify_numeric(d,rows,arrays,statuses,inputs,inputs[:160] if mutation!='inputs' else [inputs[0]]*160,overlap,'A_comparator')

def test_bounds_all_coordinates_and_research_not_gate(P):
    ref=np.zeros((160,10,7),np.float32);ref[:,:,1]=-.5;ref[:,:,3]=1;q=ref[:,0]
    result=P.describe(ref,ref,q);assert result['bounds']['all_passed']
    bad=ref.copy();bad[159,9,3]=np.nextafter(np.float32(3.1),np.float32(np.inf))
    # Use exact original upper, no new tolerance.
    bad[159,9,3]=P.UPPER[3]+.01
    result=P.describe(bad,ref,q);assert result['bounds']['violation_count']==1 and len(result['bounds']['violations'])==1
    a=dict(worst_initial_first_max_abs_rad=0.,macro_first_mae_rad=0.,macro_full_mae_rad=0.)
    research=P.research_comparison(a,dict(a,macro_first_mae_rad=1.))
    assert research['macro_first_mae_rad']['ratio'] is None and not research['criteria_passed']
    assert research['affects_admission'] is False
    json.dumps(research,allow_nan=False)

def test_execution_requires_real_matching_external_exit(P,tmp_path):
    receipt=tmp_path/'receipt.json';receipt.write_text('{}')
    producer={'process_identity':{'pid':123,'start_ticks':456},'command':['synthetic','--collect']}
    with pytest.raises(ValueError):P.verify_execution(receipt,producer)
    record=dict(schema=P.PROCESS_EXIT_SCHEMA,**producer,exit_code=0,process_group_drained=True,errors=[],deferred_signals=[],receipt_sha256=P.file_sha(receipt))
    record=with_guard(P,tmp_path,record);path=tmp_path/'execution.json';path.write_text(json.dumps(record));P.verify_execution(receipt,producer)
    for field,value in [('exit_code',-15),('process_group_drained',False),('receipt_sha256','bad'),('deferred_signals',[15])]:
        path.write_text(json.dumps(dict(record,**{field:value})))
        with pytest.raises(ValueError):P.verify_execution(receipt,producer)

def test_numeric_failure_control_accounted_without_fabricating_outputs(P):
    d,rows,arrays,statuses,inputs,overlap=numeric_fixture(P)
    for label,offset in [('pass',0),('replay',160)]:
        arrays[f'{label}_019_normalized']=np.full((10,7),np.nan,np.float32);del arrays[f'{label}_019_raw']
        statuses[offset+19]['status']='nonfinite_return'
    a=P.verify_numeric(d,rows,arrays,statuses,inputs,inputs[:160],overlap,'A_comparator')
    b=P.verify_numeric(d,rows,arrays,statuses,inputs,inputs[:160],overlap,'B_candidate')
    assert a['status']=='numerical_failure' and a['metrics'] is None and b['status']=='numerical_failure'
    assert a['numerical_failure_evidence']==[19]
    statuses[19]['status']='arbitrary_exception'
    with pytest.raises(ValueError):P.verify_numeric(d,rows,arrays,statuses,inputs,inputs[:160],overlap,'A_comparator')

def test_collection_command_is_source_bound_and_explicit(P,tmp_path):
    command=P.collection_command(tmp_path/'plan.json','B_candidate',tmp_path/'out')
    assert command[0]==str(P.T.ROOT/'vendor/openpi/.venv/bin/python')
    assert command[1:4]==['-B','-m','sim.auscultation_pi05_v8.pair_admission'] and command[-1]=='--collect'
    assert str((tmp_path/'plan.json').resolve()) in command
    with pytest.raises(ValueError):P.collection_command(tmp_path/'plan','fallback',tmp_path/'out')

def test_all8_overlaps_cannot_be_waived_by_numeric_failure(P):
    d,rows,arrays,statuses,inputs,overlap=numeric_fixture(P)
    for label,offset in [('pass',0),('replay',160)]:
        arrays[f'{label}_020_normalized']=np.full((10,7),np.nan,np.float32);del arrays[f'{label}_020_raw'];statuses[offset+20]['status']='nonfinite_return'
    with pytest.raises(ValueError,match='overlap'):P.verify_numeric(d,rows,arrays,statuses,inputs,inputs[:160],overlap,'A_comparator')

def pair_context(P,tmp_path,monkeypatch):
    a=dataset();b=dataset(C.B)
    cfg=dict(source_sha256={'unchanged':'24'},assets={'tokenizer_sha256':'synthetic'},pair_spec={'path':'/approved','sha256':'spec'},seed=61,learning_rate=5e-5,batch_size=4,observation_contract={'rgb':True},parent_data_hash='parent')
    ac=dict(cfg,mode=C.A,training_run_path=str(tmp_path/'A'));bc=dict(cfg,mode=C.B,training_run_path=str(tmp_path/'B'))
    path=Path(ac['training_run_path'])/'initialization.json';path.parent.mkdir();path.write_text('{}');bc['paired_initialization']=P.binding(path)
    numeric={'synthetic':'fresh'};ai={'numerical_initialization':numeric,'integrity_sha256':'A'};bi=dict(ai,paired_a_numerics_equal=True,paired_a_receipt_integrity='A')
    monkeypatch.setattr(P.T,'_load_initialization',lambda c:(ai,P.file_sha(path)) if c['mode']==C.A else (bi,'bh'))
    monkeypatch.setattr(P.T,'records_for',lambda d:('same','train'))
    schedule=dict(total_steps=14600,total_draws=58400,candidate_steps=list(P.E.EXPECTED_STEPS))
    return {'A_comparator':dict(dataset=a,config=ac,schedule=schedule),'B_candidate':dict(dataset=b,config=bc,schedule=copy.deepcopy(schedule))},bi

@pytest.mark.parametrize('field',['source','assets','observation','schedule','numerical','a_link','mode'])
def test_pair_integrity_rejects_foreign_arm_sources_init_schedule(P,tmp_path,monkeypatch,field):
    contexts,bi=pair_context(P,tmp_path,monkeypatch);P.validate_pair(contexts);b=contexts['B_candidate']
    if field=='source':b['config']['source_sha256']={}
    if field=='assets':b['config']['assets']={}
    if field=='observation':b['config']['observation_contract']={}
    if field=='schedule':b['schedule']['total_draws']-=4
    if field=='numerical':bi['numerical_initialization']={}
    if field=='a_link':b['config']['paired_initialization']['sha256']='bad'
    if field=='mode':b['config']['mode']=C.A
    with pytest.raises(ValueError):P.validate_pair(contexts)

def final_fixture(P,tmp_path,monkeypatch,*,b_selected=True,a_unsafe=False,b_unsafe=False,numeric_a=False):
    plan_path=tmp_path/'plan.json';plan_path.write_text('{}')
    contexts={r:dict(dataset=SimpleNamespace(normalization={'action_encoding_binding':{'mode':m}}),config={}) for r,m in zip(P.ROLES,(C.A,C.B))}
    p=dict(a_comparator_role='final14600_descriptive_only',a_checkpoint_binding={'local_step':14600},b_checkpoint_binding={'local_step':1217} if b_selected else None)
    for k in ('a_selection_lock_binding','b_selection_lock_binding','a_initialization_binding','b_initialization_binding'):
        q=tmp_path/k;q.write_text('{}');p[k]=P.binding(q)
    selected={'bounds':{'all_passed':True},'moving_buckets':{'passed':16}} if b_selected else None
    locks={P.ROLES[1]:{'selection':{'selected':selected}}}
    base={'worst_initial_first_max_abs_rad':1.,'macro_first_mae_rad':1.,'macro_full_mae_rad':1.}
    monkeypatch.setattr(P,'_scores',lambda context,step:dict(base,worst_initial_first_max_abs_rad=2. if context is contexts[P.ROLES[1]] else 1.))
    monkeypatch.setattr(P.D,'dataset_binding',lambda d:{'synthetic':True})
    evidence={}
    for role,unsafe in [(P.ROLES[0],a_unsafe),(P.ROLES[1],b_unsafe)]:
        q=tmp_path/(role+'.json');q.write_text('{}');x=tmp_path/(role+'.exit.json');x.write_text('{}')
        evidence[str(q)]=dict(receipt=dict(role=role,status='numerical_failure' if role==P.ROLES[0] and numeric_a else 'finite',bounds={'all_passed':not unsafe},metrics=None if numeric_a else {'first_mae_rad':1.},actual_consumed_input_leaf_hashes=['same'],artifact_sha256={}),receipt_binding=P.binding(q),execution_binding=P.binding(x))
    monkeypatch.setattr(P,'_verify_fixed',lambda pp,plan,ctx,receipt:evidence[str(receipt)])
    plan_path.write_text(json.dumps(p))
    monkeypatch.setattr(P,'artifact_closure',lambda *args:{})
    return p,contexts,locks,plan_path,tmp_path/(P.ROLES[0]+'.json'),tmp_path/(P.ROLES[1]+'.json')

@pytest.mark.parametrize('negative_a',['raw','numeric'])
def test_safe_B_admits_despite_negative_A_and_failed_research(P,tmp_path,monkeypatch,negative_a):
    p,ctx,locks,plan,a,b=final_fixture(P,tmp_path,monkeypatch,a_unsafe=negative_a=='raw',numeric_a=negative_a=='numeric')
    value=P._final(plan,p,ctx,locks,a,b)
    assert value['candidate_admitted'] and value['service_eligible'] and not value['research_primary_final14600']['criteria_passed']
    assert value['physical_accepted_false'] and value['no_reselection']
    assert value['admitted_dataset_codec_binding']['normalization_sha256']==P.digest(ctx[P.ROLES[1]]['dataset'].normalization)


def test_negative_B_cannot_fallback_and_one_violation_rejects(P,tmp_path,monkeypatch):
    p,ctx,locks,plan,a,b=final_fixture(P,tmp_path,monkeypatch,b_selected=False)
    value=P._final(plan,p,ctx,locks,None,None);assert value['status']=='rejected_no_eligible_B' and not value['candidate_admitted']
    with pytest.raises(ValueError):P._final(plan,p,ctx,locks,a,b)
    locks[P.ROLES[1]]['selection']['selected']={'bounds':{'all_passed':True},'moving_buckets':{'passed':16}};p['b_checkpoint_binding']={'local_step':1217}
    with pytest.raises(ValueError):P._final(plan,p,ctx,locks,None,b)


def test_resealed_final_boolean_cannot_override_reconstruction(P,tmp_path,monkeypatch):
    p,ctx,locks,plan,a,b=final_fixture(P,tmp_path,monkeypatch,b_unsafe=True)
    expected=P._final(plan,p,ctx,locks,a,b);assert not expected['candidate_admitted']
    path=tmp_path/'admission.json';path.write_text(json.dumps(expected))
    monkeypatch.setattr(P,'_verify_plan',lambda path:(p,ctx,locks))
    P.verify_pair_admission(path,P.file_sha(path))
    expected['candidate_admitted']=True;expected['service_eligible']=True;path.write_text(json.dumps(expected))
    with pytest.raises(ValueError):P.verify_pair_admission(path,P.file_sha(path))

def test_verifier_source_identity_is_portable_but_original_artifact_stays_bound(P,tmp_path,monkeypatch):
    original=P.binding(P.__file__);expected=P.source_binding()
    q=tmp_path/'pair_admission.py';q.write_bytes(Path(P.__file__).read_bytes());monkeypatch.setattr(P,'__file__',str(q))
    assert P.source_binding()==expected
    P.diagnostic_artifact(original)
    q.write_text('changed')
    with pytest.raises(ValueError):P.diagnostic_artifact(original)

def record_fixture(P,tmp_path,monkeypatch):
    d,rows,arrays,statuses,inputs,overlap=numeric_fixture(P)
    ctx={'dataset':d,'config':{'assets':{'tokenizer_sha256':'synthetic'},'openpi_commit':'synthetic'},'schedule':{'numpy_version':'1.26.4'}}
    plan=tmp_path/'plan.json';plan.write_text('{}');out=tmp_path/'collection';out.mkdir()
    p=dict(a_checkpoint_binding={'synthetic':True},b_checkpoint_binding={'synthetic':True},fixed160_rows=rows,fixed160_rows_sha256=P.digest(rows),a_selection_lock_binding={'synthetic':'lock'},frozen_runtime_inventory_binding={'synthetic':'343'},diagnostic_source_artifact_binding=P.binding(P.__file__))
    plan.write_text(json.dumps(p))
    producer=dict(process_identity={'pid':123,'start_ticks':456},command=P.collection_command(plan,P.ROLES[0],out),source_file_binding=p['diagnostic_source_artifact_binding'])
    monkeypatch.setattr(P,'_canonical_inputs',lambda *a:inputs[:160]);monkeypatch.setattr(P,'_tokenizer',lambda *a:None);monkeypatch.setattr(P,'_overlap',lambda *a:overlap)
    monkeypatch.setattr(P.D,'dataset_binding',lambda *a:{'synthetic':'dataset'})
    env=dict(numpy_version='1.26.4',openpi_python_prefix=str((P.T.ROOT/'vendor/openpi/.venv').resolve()),python_executable='/synthetic/python')
    receipt=P._record(plan,p,ctx,P.ROLES[0],out,statuses,inputs,arrays,producer,env)
    path=Path(receipt['path']);execution=dict(schema=P.PROCESS_EXIT_SCHEMA,process_identity=producer['process_identity'],command=producer['command'],exit_code=0,process_group_drained=True,errors=[],deferred_signals=[],receipt_sha256=P.file_sha(path))
    execution=with_guard(P,out,execution);(out/'execution.json').write_text(json.dumps(execution))
    return plan,p,{P.ROLES[0]:ctx},path,execution

@pytest.mark.parametrize('mutation',['metric','command','source','exit_missing','numeric_hash','mode'])
def test_real_numeric_receipt_cpu_reconstruction_rejects_tamper(P,tmp_path,monkeypatch,mutation):
    plan,p,ctx,path,execution=record_fixture(P,tmp_path,monkeypatch)
    P._verify_fixed(plan,p,ctx,path)
    r=json.loads(path.read_text())
    if mutation=='metric':r['metrics']['first_mae_rad']=99
    if mutation=='command':r['producer_execution_binding']['command']=['uncontrolled']
    if mutation=='source':r['producer_execution_binding']['source_file_binding']['sha256']='bad'
    if mutation=='mode':r['mode']=C.B
    if mutation=='numeric_hash':r['numeric_file_binding']['sha256']='bad'
    path.write_text(json.dumps(r));execution['receipt_sha256']=P.file_sha(path);path.with_name('execution.json').write_text(json.dumps(execution))
    if mutation=='exit_missing':path.with_name('execution.json').unlink()
    with pytest.raises(ValueError):P._verify_fixed(plan,p,ctx,path)

def test_negative_B_collection_stops_before_output_or_model(P,tmp_path,monkeypatch):
    monkeypatch.setattr(P,'_verify_plan',lambda *a:({'b_checkpoint_binding':None},{},{}))
    out=tmp_path/'never-created'
    with pytest.raises(ValueError,match='no fallback'):P.collect_fixed160(tmp_path/'plan','A_comparator',out)
    assert not out.exists()

def test_default_cpu_cli_verification_never_collects_or_writes(P,tmp_path,monkeypatch,capsys):
    out=tmp_path/'absent'
    monkeypatch.setattr(P.sys,'argv',['pair_admission','--plan',str(tmp_path/'plan'),'--output',str(out),'--verify-only'])
    monkeypatch.setattr(P,'_verify_plan',lambda *a:({'status':'synthetic_cpu_fixture'},{},{}))
    def forbidden(*args):raise AssertionError('Collection/model initialization forbidden')
    monkeypatch.setattr(P,'collect_fixed160',forbidden);P.main()
    value=json.loads(capsys.readouterr().out);assert value['verified'] and value['model_initialized'] is False and not out.exists()

def test_self_asserted_exit_without_actual_guard_evidence_is_rejected(P,tmp_path):
    path=tmp_path/'receipt.json';path.write_text('{}');producer=dict(process_identity={'pid':123,'start_ticks':456},command=['synthetic'])
    value=dict(schema=P.PROCESS_EXIT_SCHEMA,**producer,exit_code=0,process_group_drained=True,errors=[],deferred_signals=[],receipt_sha256=P.file_sha(path))
    path.with_name('execution.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='guard|schema'):P.verify_execution(path,producer)


def with_guard(P,directory,execution):
    identity=execution['process_identity']
    guard=dict(command=execution['command'],process_identity=dict(pid=identity['pid'],start_ticks=str(identity['start_ticks']),pgid=identity['pid'],sid=identity['pid']),passed=True,exit_code=0,process_group_drained=True,errors=[],deferred_signals=[],signals=[])
    p=directory/'synthetic.process.json';p.write_text(json.dumps(guard))
    source=Path(__file__).resolve().parents[3]/'results/diagnostics/coverage_pilot_cloud_launcher/process_guard.py'
    if not source.exists():
        source=P.T.ROOT/'diagnostics/coverage_pilot_cloud_launcher/process_guard.py'
    assert P.file_sha(source)==P.GUARD_SHA256
    return dict(execution,guard_evidence_binding=P.binding(p),guard_source_binding=P.binding(source))


@pytest.mark.parametrize('change',['late','guard_false','session','source'])
def test_guard_evidence_is_reconstructed_not_only_hashed(P,tmp_path,change):
    path=tmp_path/'receipt.json';path.write_text('{}');producer=dict(process_identity={'pid':123,'start_ticks':456},command=['synthetic'])
    record=with_guard(P,tmp_path,dict(schema=P.PROCESS_EXIT_SCHEMA,**producer,exit_code=0,process_group_drained=True,errors=[],deferred_signals=[],receipt_sha256=P.file_sha(path)))
    gp=Path(record['guard_evidence_binding']['path']);guard=json.loads(gp.read_text())
    if change=='late':gp.with_name('synthetic.late_interrupt.json').write_text('{}')
    if change=='guard_false':guard['passed']=False
    if change=='session':guard['process_identity']['sid']=999
    if change=='source':record['guard_source_binding']=P.binding(path)
    gp.write_text(json.dumps(guard));record['guard_evidence_binding']=P.binding(gp);path.with_name('execution.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):P.verify_execution(path,producer)

def test_finalization_is_write_once_and_readonly_then_reconstructed(P,tmp_path,monkeypatch):
    p,ctx,locks,plan,a,b=final_fixture(P,tmp_path,monkeypatch)
    monkeypatch.setattr(P,'_verify_plan',lambda *args:(p,ctx,locks))
    def fresh(output,*args):
        output=Path(output);output.mkdir();return output
    monkeypatch.setattr(P,'_fresh',fresh)
    out=tmp_path/'final';value=P.finalize_pair_admission(plan,a,b,out);path=Path(value['path'])
    assert not(path.stat().st_mode&0o222)
    assert P.verify_pair_admission(path,value['sha256'])['candidate_admitted']
    with pytest.raises(FileExistsError):P.finalize_pair_admission(plan,a,b,out)


def test_fresh_output_cannot_write_training_or_dataset_trees(P,tmp_path):
    run=tmp_path/'run';run.mkdir();manifest=tmp_path/'data/manifest.json';manifest.parent.mkdir();manifest.write_text('{}')
    (run/'configuration.json').write_text(json.dumps({'manifest':str(manifest)}))
    ctx={P.ROLES[0]:{'config':{'training_run_path':str(run)}}}
    for out in (run/'new',manifest.parent/'new'):
        with pytest.raises(ValueError):P._fresh(out,ctx)
        assert not out.exists()
    assert P._fresh(tmp_path/'separate',ctx).is_dir()

def test_complete_descriptive_metrics_never_new_gate(P):
    d=dataset();rows=P.fixed_rows(d);ref,q=P.fixed_references(d,rows);pred=ref.copy();pred[139,0,0]+=.123
    result=P.describe(pred,ref,q,rows)
    assert result['metrics']['worst_first_max_abs_rad']>.1229
    assert len(result['metrics']['per_episode'])==8
    assert result['metrics']['per_episode'][6]['episode_id']==d.episodes[6]['episode_id']
    assert result['metrics']['per_episode'][6]['worst_first_max_abs_rad']>.1229
    d.episodes[-1]['targets'][:]='R1'
    with pytest.raises(ValueError):P.fixed_rows(d)

def test_artifact_add_rejects_conflicts_changes_and_canonical_aliases(P,tmp_path):
    p=tmp_path/'data';p.write_bytes(b'bound');expected=P.file_sha(p);result={}
    P.add_artifact(result,p,expected);P.add_artifact(result,p,expected);assert result=={str(p.resolve()):expected}
    with pytest.raises(ValueError):P.add_artifact(result,p,'0'*64)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError):P.add_artifact({},p,expected)

def test_read_pinned_rejects_rebinding_after_parse(P,tmp_path):
    p=tmp_path/'bound.json';p.write_text('{"value":1}');b=P.binding(p)
    assert P.read_pinned(b)=={'value':1}
    p.write_text('{"value":2}')
    with pytest.raises(ValueError):P.read_pinned(b)

def test_full_artifact_closure_covers26checkpoints_qualified_inputs_and_guard(P,tmp_path,monkeypatch):
    # Small text files exercise closure paths/hashes only: no real model or qualification.
    def file(path,value=b'synthetic'):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value);return P.binding(path)
    def js(path,value):return file(path,json.dumps(value).encode())
    frozen=tmp_path/'frozen';diag=tmp_path/'diag';rel='sim/auscultation_pi05_v8/train.py'
    source=file(frozen/rel);file(diag/rel)
    diagnostic=file(diag/'sim/auscultation_pi05_v8/pair_admission.py')
    inventory=js(tmp_path/'inventory.json',{'source_project':str(frozen),'source_files':{rel:source['sha256']}})
    data=tmp_path/'data';leaf=file(data/'array.npz');original=file(data/'parent.npz');raw=file(data/'raw/input.json');audit=file(data/'audit.json');dyn=file(data/'dynamics.json');outcome=file(data/'outcome.json')
    qualification=js(data/'qualification.json',{'evidence':{'task_observation_derivation':{'parent_episode':{'path':'parent.npz','sha256':original['sha256']},'evidence_root':'raw','input_sha256':{'input.json':raw['sha256']}},'joint_action_derivation':{'phase_derivation':{'independent_supervisor_audit':{'path':'audit.json','sha256':audit['sha256']}}}}})
    manifest=js(data/'manifest.json',{'dynamics_contract_path':'dynamics.json','dynamics_contract_sha256':dyn['sha256'],'episodes':[{'path':'array.npz','sha256':leaf['sha256'],'dynamics_contract_path':'dynamics.json','dynamics_contract_sha256':dyn['sha256'],'outcome_path':'outcome.json','outcome_sha256':outcome['sha256'],'qualification_path':'qualification.json','qualification_sha256':qualification['sha256']}]})
    public=tmp_path/'root';monkeypatch.setattr(P.T,'ROOT',public);base=file(public/'public_checkpoint_verified.json');token=file(public/'public-assets/paligemma_tokenizer.model')
    p={'approved_experiment_binding':file(tmp_path/'spec.json'),'frozen_runtime_inventory_binding':inventory,'diagnostic_source_artifact_binding':diagnostic};ctx={};locks={};triples=[]
    for role,prefix in zip(P.ROLES,('a','b')):
        run=tmp_path/prefix;initial=file(run/'initialization.json');schedule=file(run/'training_schedule.json')
        config={'training_run_path':str(run),'schedule_binding':schedule,'assets':{'checkpoint_receipt_sha256':base['sha256'],'tokenizer_sha256':token['sha256']}}
        cb=js(run/'configuration.json',{'config':config,'manifest':manifest['path'],'manifest_sha256':manifest['sha256']});artifacts={cb['path']:cb['sha256']};candidates=[]
        for step in P.E.EXPECTED_STEPS:
            cp=run/'model'/('step-%012d'%step);ck={'checkpoint_path':str(cp)}
            for name,key in [('metadata.json','metadata_sha256'),('arrays.npz','arrays_sha256'),('COMPLETE.json','complete_sha256')]:
                item=file(cp/name);ck[key]=item['sha256'];triples.append(item['path'])
            report=file(run/'diagnostics'/str(step)/'report.json');artifacts[report['path']]=report['sha256'];candidates.append({'local_step':step,'checkpoint_binding':ck})
        selection={'artifact_sha256':artifacts,'candidates':candidates};sp=js(run/'pair_selection/selection.json',selection);lock={'selection':selection,'selection_path':sp['path'],'selection_sha256':sp['sha256']};lp=js(run/'pair_selection/selection.lock.json',lock)
        js(run/P.E.MARKER,P.E._marker(Path(lp['path'])));p[prefix+'_selection_lock_binding']=lp;p[prefix+'_initialization_binding']=initial;ctx[role]={'config':config};locks[role]=lock
    plan=js(tmp_path/'plan.json',p);guard=file(tmp_path/'stage.process.json');guard_source=file(tmp_path/'guard.py');execution=js(tmp_path/'execution.json',{'guard_evidence_binding':guard,'guard_source_binding':guard_source});receipt=file(tmp_path/'receipt.json');numeric=file(tmp_path/'numeric.npz')
    fixed={'receipt_binding':receipt,'execution_binding':execution,'receipt':{'producer_execution_binding':{'source_file_binding':diagnostic},'artifact_sha256':{numeric['path']:numeric['sha256']}}}
    result=P.artifact_closure(plan,p,ctx,locks,(fixed,None))
    assert len(triples)==78 and all(name in result for name in triples)
    for item in (plan,inventory,diagnostic,source,manifest,qualification,raw,original,guard,guard_source,execution,numeric,token,base):assert result[item['path']]==item['sha256']
    Path(raw['path']).write_text('changed')
    with pytest.raises(ValueError):P.artifact_closure(plan,p,ctx,locks,(fixed,None))

def test_selected_binding_cannot_be_freshly_rebound_after_lock(P,monkeypatch):
    expected={'checkpoint_path':'/synthetic/checkpoint','local_step':1217,'arrays_sha256':'original'}
    selection={'selected':{'checkpoint_binding':expected},'candidates':[{'local_step':14600,'checkpoint_binding':dict(expected,local_step=14600)}]}
    monkeypatch.setattr(P.T,'checkpoint_binding',lambda p:dict(expected))
    assert P.chosen_binding(selection,'B_candidate')==expected
    monkeypatch.setattr(P.T,'checkpoint_binding',lambda p:dict(expected,arrays_sha256='rebound'))
    with pytest.raises(ValueError):P.chosen_binding(selection,'B_candidate')
    selection['selected']=None
    assert P.chosen_binding(selection,'B_candidate') is None
    final=selection['candidates'][0]['checkpoint_binding'];monkeypatch.setattr(P.T,'checkpoint_binding',lambda p:final)
    assert P.chosen_binding(selection,'A_comparator')==final
