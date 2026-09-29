"""Synthetic CPU contracts only: no actual B trigger, model or GPU admission."""
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from sim.auscultation_pi05_v7 import coverage_pilot_schedule as S
from sim.auscultation_pi05_v8 import train as OLD, action_encoding as B
from sim.auscultation_pi05_v9 import action_encoding as C

@pytest.fixture
def train():
    try:return importlib.import_module('sim.auscultation_pi05_v9.train')
    except ModuleNotFoundError:
        def absent(*a,**kw):pytest.fail('Conditional C trainer is not implemented')
        return SimpleNamespace(fingerprint_leaves=absent)

def ensure(t):t.fingerprint_leaves([('x',np.zeros(1,np.float32))])
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(S.json_bytes(v))
def numerical(t,value=1.):
    return dict(trainable=t.fingerprint_leaves([('kernel',np.array([value],np.float32))]),
        frozen=t.fingerprint_leaves([('backbone',np.array([2.],np.float32))]),
        optimizer=t.fingerprint_leaves([('adam.count',np.array(0,np.int32)),('adam.mu',np.zeros(1,np.float32))]),
        jax_rng=t.fingerprint_leaves([('key_data',np.array([0,61],np.uint32))]),optimizer_step=0,rng_seed=61)

def source_map(t):
    # E may still be authored independently. Only its inventory read is stubbed
    # until that file exists; no fake module or source is placed on disk.
    p=Path(t.__file__).with_name('evaluation.py')
    if p.exists():return t.source_hashes()
    return dict(t.ORIGINAL_B_SOURCES,**{'v8/pair_admission.py':t.PAIR_SOURCE_SHA256,
        **{f'v9/{n}':S.file_sha(Path(t.__file__).with_name(n)) for n in ('action_encoding.py','training_data.py','trigger.py','train.py')},
        'v9/evaluation.py':'e'*64})

def fixture(t,tmp_path,monkeypatch):
    ensure(t);b=tmp_path/'B';b.mkdir();monkeypatch.setattr(t.trigger,'B_RUN',b)
    assets={'public':'synthetic'};pair={'path':'/synthetic/original_pair_spec','sha256':'p'*64}
    common=dict(seed=61,learning_rate=5e-5,batch_size=4,source_sha256=t.ORIGINAL_B_SOURCES,assets=assets,pair_spec=pair)
    a=OLD.make_initialization(dict(common,mode=B.A,training_run_path=str(tmp_path/'A')),numerical(t),None)
    ap=tmp_path/'A/initialization.json';write(ap,a);ab={'path':str(ap),'sha256':S.file_sha(ap)}
    schedule=dict(schema=S.VERSION,seed=61,batch_size=4,records_sha256='records',total_steps=14600,total_draws=58400,valid_window_count=12759,candidate_steps=list(t.EXPECTED_STEPS),rows=[])
    sb=S.save_schedule(b/'training_schedule.json',schedule)
    bc=dict(common,mode=B.B,training_schema=OLD.VERSION,recipe=OLD.RECIPE,training_run_path=str(b),schedule_binding=sb,paired_initialization=ab)
    initial=OLD.make_initialization(bc,numerical(t),a);write(b/'initialization.json',initial)
    write(b/'configuration.json',dict(config=bc,data_hash='B-data',normalization={'synthetic':True},manifest='/synthetic/manifest'))
    spec=Path(__file__).resolve().parents[3]/'results/diagnostics/horizon_q4_rational_conditional_design_v1/experiment_spec.json'
    condition=dict(schema=t.trigger.SCHEMA,condition_satisfied=True,specification_binding={'path':str(spec),'sha256':t.SPEC_SHA256},
        artifact_sha256={str(p):S.file_sha(p) for p in (b/'configuration.json',b/'initialization.json',b/'training_schedule.json',ap)})
    cp=tmp_path/'inputs/condition.json';write(cp,condition);binding={'path':str(cp),'sha256':S.file_sha(cp)}
    calls=[]
    def verified(path,sha):
        calls.append((path,sha));assert path==binding['path'] and sha==binding['sha256'];return copy.deepcopy(condition)
    monkeypatch.setattr(t.trigger,'verify_receipt',verified)
    norm_path=tmp_path/'inputs/normalization.json';write(norm_path,{'synthetic':True})
    nb={'path':str(norm_path),'sha256':S.file_sha(norm_path)};monkeypatch.setattr(t,'NORMALIZATION_FILE_SHA256',nb['sha256'])
    reference=t.reference_from_condition(binding,reconstruct=True)
    cfg=dict(training_schema=t.VERSION,recipe=t.RECIPE,mode=C.C,training_run_path=str(tmp_path/'C'),seed=61,learning_rate=5e-5,batch_size=4,
        source_sha256=source_map(t),assets=assets,conditional_spec=condition['specification_binding'],condition_binding=binding,
        saved_normalization_binding=nb,b_reference=reference,schedule_binding=dict(sb,path=str(tmp_path/'C/training_schedule.json')),
        dataset_binding={'mode':C.C,'data_hash':'C-data'},total_steps=14600)
    return SimpleNamespace(b=b,initial=initial,schedule=schedule,condition=condition,condition_binding=binding,normalization_binding=nb,reference=reference,config=cfg,calls=calls)

def test_exact_initial_fingerprints_and_optimizer_reuse(train):
    ensure(train)
    assert train.fingerprint_leaves is OLD.fingerprint_leaves
    assert train.verify_optimizer_age is OLD.verify_optimizer_age
    assert train.verify_step_zero_arrays is OLD.verify_step_zero_arrays
    assert train._numeric_initialization is OLD._numeric_initialization
    with pytest.raises(ValueError):train._validate_numerical(dict(numerical(train),optimizer_step=4000))

def test_C_requires_identical_B_numeric_init_but_distinct_truthful_sources(train,tmp_path,monkeypatch):
    f=fixture(train,tmp_path,monkeypatch)
    result=train.make_initialization(f.config,numerical(train),f.initial)
    assert result['schema']==train.INITIALIZATION_SCHEMA and result['mode']==C.C and result['paired_B_numerics_equal'] is True
    assert result['source_sha256']!=f.initial['source_sha256']
    assert result['numerical_initialization']==f.initial['numerical_initialization']
    for changed in (None,dict(f.initial,mode=B.A),dict(f.initial,source_sha256={}),dict(f.initial,assets={})):
        with pytest.raises(ValueError):train.make_initialization(f.config,numerical(train),changed)
    for partition in ('trainable','frozen','optimizer','jax_rng'):
        numeric=copy.deepcopy(numerical(train));numeric[partition][0]['sha256']='f'*64
        with pytest.raises(ValueError):train.make_initialization(f.config,numeric,f.initial)
    cfg=copy.deepcopy(f.config);cfg['source_sha256']['v7/model.py']='0'*64
    with pytest.raises(ValueError):train.make_initialization(cfg,numerical(train),f.initial)

def test_reference_requires_fresh_condition_and_pinned_B_closure(train,tmp_path,monkeypatch):
    f=fixture(train,tmp_path,monkeypatch);assert len(f.calls)==1
    assert f.reference['source_sha256']==train.ORIGINAL_B_SOURCES
    assert f.reference['initialization_binding']['sha256']==S.file_sha(f.b/'initialization.json')
    monkeypatch.setattr(train.trigger,'verify_receipt',lambda *a:dict(f.condition,condition_satisfied=False))
    with pytest.raises(ValueError):train.reference_from_condition(f.condition_binding,reconstruct=True)
    monkeypatch.setattr(train.trigger,'verify_receipt',lambda *a:f.condition)
    (f.b/'training_schedule.json').write_bytes(b'changed')
    with pytest.raises(ValueError):train.reference_from_condition(f.condition_binding,reconstruct=True)

def test_condition_and_saved_norm_fail_before_loader(train,tmp_path,monkeypatch):
    ensure(train);seen=[]
    monkeypatch.setattr(train.trigger,'verify_receipt',lambda *a:(_ for _ in ()).throw(ValueError('Not satisfied')))
    monkeypatch.setattr(train.data_api,'load_dataset',lambda *a,**k:seen.append('loader'))
    with pytest.raises(ValueError):train.load_inputs(tmp_path/'manifest',C.C,condition_binding={'path':str(tmp_path/'no'),'sha256':'0'*64},saved_normalization_binding={'path':str(tmp_path/'norm'),'sha256':'0'*64})
    assert seen==[]

def test_schedule_copy_preserves_exact_B_bytes_and_rejects_rebinding(train,tmp_path,monkeypatch):
    f=fixture(train,tmp_path,monkeypatch);target=tmp_path/'C/training_schedule.json';target.parent.mkdir()
    got=train.save_reference_schedule(target,f.schedule,f.reference)
    assert target.read_bytes()==(f.b/'training_schedule.json').read_bytes()
    assert got['sha256']==f.reference['schedule_binding']['sha256']
    bad=copy.deepcopy(f.schedule);bad['seed']=0
    with pytest.raises(ValueError):train.save_reference_schedule(target.with_name('bad.json'),bad,f.reference)
    assert not target.with_name('bad.json').exists()

def test_load_inputs_passes_saved_normalization_and_exact_schedule(train,tmp_path,monkeypatch):
    f=fixture(train,tmp_path,monkeypatch);seen=[]
    data=SimpleNamespace(parent=SimpleNamespace(data_hash='B-data'),normalization={'synthetic':True})
    monkeypatch.setattr(train,'source_hashes',lambda:source_map_for_stub(train))
    def load(manifest,mode,**kwargs):seen.append(kwargs);return data
    monkeypatch.setattr(train.data_api,'load_dataset',load)
    monkeypatch.setattr(train,'records_for',lambda d:(['synthetic'],list(range(8))))
    monkeypatch.setattr(S,'verify_schedule',lambda saved,*args:saved)
    monkeypatch.setattr(train,'parent_module',lambda:SimpleNamespace(_verify_public_assets=lambda root:(None,None,{'public':'synthetic'})))
    out=train.load_inputs(tmp_path/'manifest',C.C,condition_binding=f.condition_binding,saved_normalization_binding=f.normalization_binding)
    assert out[0] is data and out[1]==f.schedule and len(f.calls)==2
    assert seen==[{'normalization':{'synthetic':True},'expected_normalization_sha256':train.data_api._digest({'synthetic':True})}]

def source_map_for_stub(t):return dict(t.ORIGINAL_B_SOURCES,**{'synthetic/new':'hash'})

def test_verify_only_is_readonly_and_has_no_model_import(train,tmp_path,monkeypatch):
    f=fixture(train,tmp_path,monkeypatch)
    data=SimpleNamespace(mode=C.C,data_hash='C-data',normalization={})
    monkeypatch.setattr(train,'load_inputs',lambda *a,**k:(data,f.schedule,{},f.condition['specification_binding'],f.reference))
    monkeypatch.setattr(train,'source_hashes',lambda:source_map_for_stub(train))
    args=SimpleNamespace(manifest=tmp_path/'inputs/manifest',mode=C.C,output=tmp_path/'unused',condition_receipt=Path(f.condition_binding['path']),condition_sha256=f.condition_binding['sha256'],normalization=Path(f.normalization_binding['path']),normalization_sha256=f.normalization_binding['sha256'],resume=None,stop_at=None,verify_only=True)
    result=train.run(args)
    assert result['verified'] and result['gpu_initialized'] is False and result['output_changed'] is False
    assert not args.output.exists()

def test_resume_uses_latest_complete_same_run_and_new_negative_lock(train,tmp_path,monkeypatch):
    ensure(train);run=tmp_path/'run';old=run/'model/step-000000000000';new=run/'recovery/step-000000000027'
    for p in (old,new):p.mkdir(parents=True);write(p/'COMPLETE.json',{})
    m=dict(step=27,optimizer_step=27,schedule_cursor=108,leaf_contract=[])
    np.savez(new/'arrays.npz',jax_rng=np.array([0,61],np.uint32))
    monkeypatch.setattr(train,'_verified_metadata',lambda *a,**k:m)
    monkeypatch.setattr(train,'verify_optimizer_age',lambda *a:None)
    kwargs=dict(data_hash='C',config={'training_run_path':str(run)},normalization={})
    assert train.resolve_same_run_resume(run,new,**kwargs)==m
    with pytest.raises(ValueError):train.resolve_same_run_resume(run,old,**kwargs)
    for marker in ('RATIONAL_SELECTION_LOCKED.json','rational_selection'):
        (run/marker).touch()
        with pytest.raises(ValueError):train.resolve_same_run_resume(run,new,**kwargs)
        (run/marker).unlink()

def test_source_closure_exact_B24_plus_pair_and_new_five(train):
    ensure(train)
    assert len(train.ORIGINAL_B_SOURCES)==24 and train.ORIGINAL_B_SOURCES==OLD.source_hashes()
    if not Path(train.__file__).with_name('evaluation.py').exists():pytest.skip('Independent root E module pending')
    values=train.source_hashes();assert len(values)==30
    assert {k:values[k] for k in train.ORIGINAL_B_SOURCES}==train.ORIGINAL_B_SOURCES
    assert values['v8/pair_admission.py']==train.PAIR_SOURCE_SHA256
    assert all('v9/'+x in values for x in ('action_encoding.py','training_data.py','trigger.py','train.py','evaluation.py'))

def test_official_loss_update_sampler_and_loop_ast_unchanged(train):
    ensure(train)
    def function(path,name):
        return next(n for n in ast.walk(ast.parse(Path(path).read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    for name in ('objective','update'):
        assert ast.dump(function(train.__file__,name),include_attributes=False)==ast.dump(function(OLD.__file__,name),include_attributes=False)
    current=Path(train.__file__).read_text()
    assert 'implementation.initialize(61,5e-5)' in current and 'implementation.sampler(runtime,steps=10)' in current
    assert 'jax.random.key(917)' in current and 'schedule_api.batch_indices(schedule,step)' in current
    assert 'RATIONAL_SELECTION_LOCKED.json' in current and 'rational_selection' in current
    for bad in (-1,14601,True):
        with pytest.raises(ValueError):train.validate_limit(bad,0)

def test_B_reference_rejects_dependency_replaced_after_pinned_parse(train,tmp_path,monkeypatch):
    f=fixture(train,tmp_path,monkeypatch);original=train._pinned
    def replace(binding):
        value=original(binding)
        if binding==f.reference['configuration_binding']:
            Path(binding['path']).write_bytes(b'{"replacement":true}\n')
        return value
    monkeypatch.setattr(train,'_pinned',replace)
    with pytest.raises(ValueError):train.reference_from_condition(f.condition_binding,reconstruct=True)

def synthetic_checkpoint(t,tmp_path,monkeypatch):
    f=fixture(t,tmp_path,monkeypatch)
    from test_action_encoding import episodes
    norm={'mode':C.C,'residual_actions':C.fit_stats(episodes())}
    npth=Path(f.normalization_binding['path']);write(npth,norm)
    nb={'path':str(npth),'sha256':S.file_sha(npth)};monkeypatch.setattr(t,'NORMALIZATION_FILE_SHA256',nb['sha256'])
    c=f.config;c['saved_normalization_binding']=nb;c['dataset_binding'].update(normalization_sha256=t.data_api._digest(norm),residual_stats_sha256=C.normalization_digest(norm['residual_actions']))
    run=Path(c['training_run_path']);run.mkdir()
    c['schedule_binding']=t.save_reference_schedule(run/'training_schedule.json',f.schedule,f.reference)
    initial=t.make_initialization(c,numerical(t),f.initial);write(run/'initialization.json',initial)
    path=run/'model/step-000000000001';path.mkdir(parents=True)
    arrays={'leaf_00000':np.array([1.],np.float32),'leaf_00001':np.array(1,np.int32),'jax_rng':np.array([0,61],np.uint32)}
    np.savez(path/'arrays.npz',**arrays)
    m=dict(schema='wam_pi05_resume_v1',training_schema=t.VERSION,step=1,optimizer_step=1,schedule_cursor=4,
        sampler_rng_state={'cursor':4,'schedule_sha256':c['schedule_binding']['sha256']},config=c,data_hash='C-data',normalization=norm,
        arrays_sha256=S.file_sha(path/'arrays.npz'),initialization_sha256=S.file_sha(run/'initialization.json'),
        leaf_contract=[dict(path='[0]kernel',shape=[1],dtype='float32'),dict(path='[1]adam.count',shape=[],dtype='int32')])
    write(path/'metadata.json',m);write(path/'COMPLETE.json',{'metadata_sha256':S.file_sha(path/'metadata.json')})
    write(run/'configuration.json',{'config':c,'data_hash':'C-data','normalization':norm})
    return f,path,m

def test_checkpoint_real_bytes_new_schema_and_saved_norm_pin(train,tmp_path,monkeypatch):
    f,path,m=synthetic_checkpoint(train,tmp_path,monkeypatch)
    bound=train.checkpoint_binding(path)
    assert bound['schema']=='wam_rational_q4_checkpoint_binding_v1' and bound['mode']==C.C
    assert bound['local_step']==bound['optimizer_step']==1
    from sim.auscultation_pi05_v9.evaluation import BINDING_KEYS
    assert set(bound)==BINDING_KEYS
    Path(m['config']['saved_normalization_binding']['path']).write_bytes(b'{}')
    with pytest.raises(ValueError):train.checkpoint_binding(path)

def test_checkpoint_rejects_metadata_replaced_during_verification(train,tmp_path,monkeypatch):
    f,path,m=synthetic_checkpoint(train,tmp_path,monkeypatch)
    original=train._verified_metadata
    def changed(p,**kw):
        value=original(p,**kw)
        changed=copy.deepcopy(value);changed['extra_unbound']='replacement'
        write(p/'metadata.json',changed);write(p/'COMPLETE.json',{'metadata_sha256':S.file_sha(p/'metadata.json')})
        return value
    monkeypatch.setattr(train,'_verified_metadata',changed)
    with pytest.raises(ValueError):train.checkpoint_binding(path)

def test_context_requires_complete_C_budget_and_all13_reports(train,tmp_path,monkeypatch):
    ensure(train)
    from sim.auscultation_pi05_v9 import evaluation as E
    run=tmp_path/'C';run.mkdir();data=SimpleNamespace(mode=C.C,data_hash='C',normalization={'synthetic':True})
    config={'training_run_path':str(run),'mode':C.C,'condition_binding':{'path':'synthetic','sha256':'synthetic'},'saved_normalization_binding':{'path':'synthetic','sha256':'synthetic'}}
    schedule={'total_steps':14600,'candidate_steps':list(train.EXPECTED_STEPS)}
    write(run/'configuration.json',dict(config=config,manifest='synthetic',manifest_sha256=train.data_api.MANIFEST_SHA256,data_hash='C',normalization=data.normalization))
    receipt=dict(schema=train.VERSION,mode=C.C,training_complete=True,local_step=14600,optimizer_step=14600,schedule_cursor=58400,error=None,interrupted=False,current_segment_path=str(run/'training_segments/current.json'))
    write(Path(receipt['current_segment_path']),receipt);write(run/'training_run.json',receipt);write(run/'diagnostic_schedule.json',dict(data_hash='C',rows=['synthetic']))
    monkeypatch.setattr(train,'load_inputs',lambda *a,**k:(data,schedule,{}, {},{}))
    monkeypatch.setattr(train,'verify_configuration',lambda *a:None);monkeypatch.setattr(train,'_load_initialization',lambda *a:None)
    monkeypatch.setattr(E,'build_diagnostic_schedule',lambda *a:['synthetic'])
    monkeypatch.setattr(train,'checkpoint_binding',lambda p:dict(local_step=int(p.name[5:])))
    seen=[]
    def report(p,*a):
        if not (p/'report.json').is_file():raise ValueError('Missing report')
        seen.append(p)
    monkeypatch.setattr(E,'verify_checkpoint_report',report)
    for step in train.EXPECTED_STEPS:write(run/'diagnostics'/('local_%06d'%step)/'report.json',{'synthetic':True})
    assert train.verified_run_context(run)['completed'] and len(seen)==13
    for field,value in [('mode',B.B),('training_complete',False),('optimizer_step',18600),('interrupted',True),('error','failure')]:
        write(run/'training_run.json',dict(receipt,**{field:value}))
        with pytest.raises(ValueError):train.verified_run_context(run)
    write(run/'training_run.json',receipt);(seen[0]/'report.json').unlink()
    with pytest.raises(ValueError):train.verified_run_context(run)


@pytest.mark.parametrize('window',['last_write','handler_restore'])
def test_actual_CPU_SIGTERM_during_finalization_fails_with_evidence(train,tmp_path,window):
    """Run exact finalizer AST in a short owned child; never run trainer/model."""
    import os,subprocess,sys
    script=tmp_path/'owned_signal_probe.py'
    script.write_text(r'''import ast,json,os,signal,sys
from pathlib import Path
from types import SimpleNamespace
source=Path(sys.argv[1]);output=Path(sys.argv[2]);window=sys.argv[3]
tree=ast.parse(source.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
save=next(n for n in ast.walk(run) if isinstance(n,ast.FunctionDef) and n.name=='save_receipt')
guard=next(n for n in run.body if isinstance(n,ast.Try))
fn=ast.FunctionDef(name='exact_finalizer',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[save,*guard.finalbody,ast.Return(ast.Name('receipt',ast.Load()))],decorator_list=[])
helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_finish_training']
code=compile(ast.fix_missing_locations(ast.Module(body=helpers+[fn],type_ignores=[])),str(source)+'::finalizer','exec')
stop=[False];handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
for sig in handlers:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
receipt=dict(training_complete=True,requested_segment_complete=True,interrupted=False,local_step=14600,optimizer_step=14600,error=None)
original_signal=signal.signal;fired=[False]
def cancel_once():
    if not fired[0]:
        fired[0]=True
        os.kill(os.getpid(),signal.SIGTERM)
        os.kill(os.getpid(),signal.SIGTERM)  # repeated handled cancellation

def atomic(path,value):
    path.write_text(json.dumps(value))
    if window=='last_write' and path==output/'training_run.json':cancel_once()
def restore(sig,handler):
    if window=='handler_restore' and sig==signal.SIGTERM:cancel_once()
    return original_signal(sig,handler)
signal.signal=restore
ns=dict(parent=SimpleNamespace(_atomic_json=atomic),receipt=receipt,step=14600,stop=stop,segment=output/'segment.json',output=output,utc=lambda:'synthetic-terminal-time',handlers=handlers,signal=signal)
exec(code,ns)
ns['exact_finalizer']()
raise SystemExit(0)
''')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,str(script),train.__file__,str(tmp_path),window],env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode!=0,result.stdout+result.stderr
    assert 'InterruptedError' in result.stderr
    saved=json.loads((tmp_path/'training_run.json').read_text())
    assert saved['training_complete'] is False and saved['interrupted'] is True
    assert saved['requested_segment_complete'] is False and saved['error']
    marker=json.loads((tmp_path/'segment.cancelled.json').read_text())
    assert marker['interrupted'] is True and marker['training_complete'] is False
    assert marker['local_step']==14600


def test_finalizer_restores_handlers_even_when_write_fails(train,monkeypatch):
    restored=[];monkeypatch.setattr(train.signal,'signal',lambda sig,old:restored.append((sig,old)))
    receipt={'training_complete':True,'error':None};original=OSError('synthetic storage failure')
    def save():raise original
    with pytest.raises(OSError) as caught:
        train._finish_training(receipt,[False],{1:'one',2:'two'},save,lambda value:None)
    assert caught.value is original and restored==[(1,'one'),(2,'two')]


def test_finalizer_cancellation_persists_marker_despite_receipt_write_error(train,monkeypatch):
    restored=[];monkeypatch.setattr(train.signal,'signal',lambda sig,old:restored.append((sig,old)))
    receipt={'training_complete':True,'error':None,'local_step':14600};markers=[]
    def save():raise OSError('synthetic receipt failure')
    with pytest.raises(InterruptedError) as caught:
        train._finish_training(receipt,[True],{1:'one',2:'two'},save,lambda value:markers.append(copy.deepcopy(value)))
    assert restored==[(1,'one'),(2,'two')] and len(markers)==1
    assert markers[0]['training_complete'] is False and markers[0]['interrupted'] is True
    assert isinstance(caught.value.__cause__,OSError)


def test_finalizer_reentry_and_marker_error_still_fail_closed(train,monkeypatch):
    monkeypatch.setattr(train.signal,'signal',lambda *a:None)
    receipt={'training_complete':True,'error':None,'local_step':14600};saved=[]
    def save():saved.append(copy.deepcopy(receipt))
    def marker(value):raise OSError('synthetic marker failure')
    for _ in range(2):
        with pytest.raises(InterruptedError):train._finish_training(receipt,[True],{},save,marker)
    assert saved[-1]['training_complete'] is False and saved[-1]['interrupted'] is True


def test_finalizer_attempts_all_handler_restorations(train,monkeypatch):
    restored=[]
    def restore(sig,old):
        restored.append(sig)
        if sig==1:raise RuntimeError('synthetic handler failure')
    monkeypatch.setattr(train.signal,'signal',restore)
    with pytest.raises(RuntimeError,match='synthetic handler failure'):
        train._finish_training({'error':None},[False],{1:'one',2:'two'},lambda:None,lambda value:None)
    assert restored==[1,2]


def complete_context_fixture(train,tmp_path,monkeypatch):
    from sim.auscultation_pi05_v9 import evaluation as E
    run=tmp_path/'C';run.mkdir();data=SimpleNamespace(mode=C.C,data_hash='C',normalization={'synthetic':True})
    config={'training_run_path':str(run),'mode':C.C,'condition_binding':{},'saved_normalization_binding':{}}
    schedule={'total_steps':14600,'candidate_steps':list(train.EXPECTED_STEPS)}
    receipt=dict(schema=train.VERSION,mode=C.C,training_complete=True,local_step=14600,optimizer_step=14600,schedule_cursor=58400,error=None,interrupted=False,current_segment_path=str(run/'training_segments/current.json'))
    write(run/'configuration.json',dict(config=config,manifest='synthetic',manifest_sha256=train.data_api.MANIFEST_SHA256,data_hash='C',normalization=data.normalization))
    write(run/'diagnostic_schedule.json',dict(data_hash='C',rows=['synthetic']))
    write(run/'training_run.json',receipt);write(Path(receipt['current_segment_path']),receipt)
    monkeypatch.setattr(train,'load_inputs',lambda *a,**k:(data,schedule,{}, {},{}))
    monkeypatch.setattr(train,'verify_configuration',lambda *a:None);monkeypatch.setattr(train,'_load_initialization',lambda *a:None)
    monkeypatch.setattr(E,'build_diagnostic_schedule',lambda *a:['synthetic'])
    monkeypatch.setattr(train,'checkpoint_binding',lambda p:dict(local_step=int(p.name[5:])))
    monkeypatch.setattr(E,'verify_checkpoint_report',lambda *a:None)
    return run,receipt,E


def test_context_rejects_cancel_marker_when_negative_receipt_rewrite_failed(train,tmp_path,monkeypatch):
    run,receipt,E=complete_context_fixture(train,tmp_path,monkeypatch)
    stop=[False];attempts=[];segment=Path(receipt['current_segment_path'])
    def save():
        attempts.append(1)
        if len(attempts)==1:
            write(segment,receipt);write(run/'training_run.json',receipt);stop[0]=True
        else:raise OSError('synthetic negative receipt rewrite failure')
    with pytest.raises(InterruptedError):
        train._finish_training(receipt,stop,{},save,lambda value:write(segment.with_suffix('.cancelled.json'),value))
    assert json.loads((run/'training_run.json').read_text())['training_complete'] is True
    with pytest.raises(ValueError):train.verified_run_context(run)


def test_context_allows_later_clean_segment_but_rejects_current_mismatch(train,tmp_path,monkeypatch):
    run,receipt,E=complete_context_fixture(train,tmp_path,monkeypatch)
    old=Path(receipt['current_segment_path']);write(old.with_suffix('.cancelled.json'),{'interrupted':True})
    new=run/'training_segments/resumed.json';receipt['current_segment_path']=str(new)
    write(new,receipt);write(run/'training_run.json',receipt)
    result=train.verified_run_context(run)
    assert result['completed'] and result['terminal_segment_binding']==dict(path=str(new),sha256=S.file_sha(new),cancellation_path=str(new.with_suffix('.cancelled.json')))
    write(new,dict(receipt,interrupted=True,training_complete=False))
    with pytest.raises(ValueError):train.verified_run_context(run)


def test_context_rechecks_terminal_segment_and_cancel_after_reports(train,tmp_path,monkeypatch):
    run,receipt,E=complete_context_fixture(train,tmp_path,monkeypatch);segment=Path(receipt['current_segment_path'])
    def cancelled(*args):write(segment.with_suffix('.cancelled.json'),{'interrupted':True})
    monkeypatch.setattr(E,'verify_checkpoint_report',cancelled)
    with pytest.raises(ValueError):train.verified_run_context(run)
    segment.with_suffix('.cancelled.json').unlink()
    def changed(*args):write(segment,dict(receipt,extra='changed after consumption'))
    monkeypatch.setattr(E,'verify_checkpoint_report',changed)
    with pytest.raises(ValueError):train.verified_run_context(run)


def test_terminal_segment_rejects_foreign_and_missing_binding(train,tmp_path,monkeypatch):
    run,receipt,E=complete_context_fixture(train,tmp_path,monkeypatch)
    for value in (None,str(tmp_path/'foreign.json')):
        changed=dict(receipt,current_segment_path=value)
        if value:write(Path(value),changed)
        write(run/'training_run.json',changed)
        with pytest.raises(ValueError):train.verified_run_context(run)


@pytest.mark.parametrize('output_kind',['A_child','A_ancestor','B_child','separate_C'])
def test_verify_only_protects_original_A_and_B_trees(train,tmp_path,monkeypatch,output_kind):
    f=fixture(train,tmp_path,monkeypatch);data=SimpleNamespace(mode=C.C,data_hash='C',normalization={})
    monkeypatch.setattr(train,'load_inputs',lambda *a,**k:(data,f.schedule,{},f.condition['specification_binding'],f.reference))
    monkeypatch.setattr(train,'source_hashes',lambda:source_map_for_stub(train))
    outputs={'A_child':tmp_path/'A/new_C','A_ancestor':tmp_path,'B_child':f.b/'new_C','separate_C':tmp_path/'runs/rational-q4-original8-v1-C'}
    args=SimpleNamespace(manifest=tmp_path/'inputs/manifest',mode=C.C,output=outputs[output_kind],condition_receipt=Path(f.condition_binding['path']),condition_sha256=f.condition_binding['sha256'],normalization=Path(f.normalization_binding['path']),normalization_sha256=f.normalization_binding['sha256'],resume=None,stop_at=None,verify_only=True)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    if output_kind=='separate_C':assert train.run(args)['verified']
    else:
        with pytest.raises(ValueError):train.run(args)
    assert {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before
    if output_kind!='A_ancestor':assert not args.output.exists()


def test_reference_requires_bound_original_A_initialization(train,tmp_path,monkeypatch):
    f=fixture(train,tmp_path,monkeypatch)
    assert f.reference['paired_A_initialization_binding']=={'path':str(tmp_path/'A/initialization.json'),'sha256':S.file_sha(tmp_path/'A/initialization.json')}
    (tmp_path/'A/initialization.json').write_text('{}')
    with pytest.raises(ValueError):train.reference_from_condition(f.condition_binding,reconstruct=True)


def test_verify_only_rechecks_A_and_condition_before_output_mutation(train,tmp_path,monkeypatch):
    f=fixture(train,tmp_path,monkeypatch);data=SimpleNamespace(mode=C.C,data_hash='C',normalization={})
    original_A=(tmp_path/'A/initialization.json').read_bytes()
    args=SimpleNamespace(manifest=tmp_path/'inputs/manifest',mode=C.C,output=tmp_path/'runs/rational-q4-original8-v1-C',condition_receipt=Path(f.condition_binding['path']),condition_sha256=f.condition_binding['sha256'],normalization=Path(f.normalization_binding['path']),normalization_sha256=f.normalization_binding['sha256'],resume=None,stop_at=None,verify_only=True)
    monkeypatch.setattr(train,'source_hashes',lambda:source_map_for_stub(train))
    for target in (tmp_path/'A/initialization.json',Path(f.condition_binding['path'])):
        original=target.read_bytes()
        def changed(*a,**k):
            target.write_text('{}')
            return data,f.schedule,{},f.condition['specification_binding'],f.reference
        monkeypatch.setattr(train,'load_inputs',changed)
        with pytest.raises(ValueError):train.run(args)
        assert not args.output.exists()
        target.write_bytes(original)
