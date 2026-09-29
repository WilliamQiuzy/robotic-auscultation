"""CPU synthetic boundaries, never a real model/qualification/training result."""
import copy,hashlib,importlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from sim.auscultation_pi05_v7 import coverage_pilot_schedule as S
from sim.auscultation_pi05_v8 import action_encoding as C

@pytest.fixture
def train():
    name='sim.auscultation_pi05_v8.train'
    if importlib.util.find_spec(name) is None:
        return SimpleNamespace(fingerprint_leaves=lambda *a,**k:pytest.fail('Fresh paired trainer is not implemented'))
    return importlib.import_module(name)

def ensure_module(t):t.fingerprint_leaves([('x',np.zeros(1,np.float32))])

def numerical(t,value=1.):
    return dict(trainable=t.fingerprint_leaves([('kernel',np.array([value],np.float32))]),
        frozen=t.fingerprint_leaves([('backbone',np.array([2.],np.float32))]),
        optimizer=t.fingerprint_leaves([('adam.count',np.array(0,np.int32)),('adam.mu',np.zeros(1,np.float32))]),
        jax_rng=t.fingerprint_leaves([('key_data',np.array([0,61],np.uint32))]),optimizer_step=0,rng_seed=61)

def config(tmp_path,mode=C.A):
    return dict(mode=mode,seed=61,learning_rate=5e-5,batch_size=4,source_sha256={'synthetic/source':'hash'},
        assets={'public':'hash'},training_run_path=str(tmp_path.resolve()),pair_spec={'path':'/synthetic/spec','sha256':'spec'},
        dataset_binding={'mode':mode,'data_hash':'data_'+mode},total_steps=14600,schedule_binding={'sha256':'schedule'},
        recipe='pi05_v8_horizon_pair_fresh_base_v1')

def test_leaf_fingerprints_use_exact_path_shape_dtype_and_bytes(train):
    a=np.array([[1.,2.]],np.float32);got=train.fingerprint_leaves(iter([('kernel',a)]))
    assert got==[dict(path='kernel',shape=[1,2],dtype='float32',sha256=hashlib.sha256(a.tobytes()).hexdigest())]
    assert got!=train.fingerprint_leaves([('kernel',a.astype(np.float64))])
    assert got!=train.fingerprint_leaves([('other',a)])
    with pytest.raises(ValueError):train.fingerprint_leaves([('x',a),('x',a)])
    with pytest.raises(ValueError):train.fingerprint_leaves([('x',np.array([np.nan]))])

def test_b_requires_a_receipt_and_exact_independent_initialization(train,tmp_path):
    ensure_module(train);a=config(tmp_path/'A');b=config(tmp_path/'B',C.B)
    initial=train.make_initialization(a,numerical(train),None)
    assert initial['optimizer_step']==0 and initial['mode']==C.A
    result=train.make_initialization(b,numerical(train),initial)
    assert result['numerical_initialization']==initial['numerical_initialization']
    assert result['paired_a_numerics_equal'] is True
    for changed in (None,dict(initial,mode=C.B),dict(initial,seed=9),dict(initial,source_sha256={}),dict(initial,assets={})):
        with pytest.raises(ValueError):train.make_initialization(b,numerical(train),changed)
    with pytest.raises(ValueError):train.make_initialization(b,numerical(train,1.1),initial)
    with pytest.raises(ValueError):train.make_initialization(a,numerical(train),initial)
    bad=numerical(train);bad['optimizer_step']=4000
    with pytest.raises(ValueError):train.make_initialization(a,bad,None)

def test_fresh_clock_rejects_old4000_and_bad_rng(train):
    ensure_module(train)
    contract=[dict(path='[0].kernel',shape=[1],dtype='float32'),dict(path='[1].count',shape=[],dtype='int32')]
    arrays={'leaf_00000':np.array([1.],np.float32),'leaf_00001':np.array(0,np.int32),'jax_rng':np.array([0,61],np.uint32)}
    train.verify_optimizer_age(arrays,contract,0)
    arrays['leaf_00001']=np.array(4000,np.int32)
    with pytest.raises(ValueError):train.verify_optimizer_age(arrays,contract,0)
    arrays['leaf_00001']=np.array(1217,np.int32);train.verify_optimizer_age(arrays,contract,1217)
    arrays['jax_rng']=np.array([0.,61.],np.float32)
    with pytest.raises(ValueError):train.verify_optimizer_age(arrays,contract,1217)

def write_json(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(S.json_bytes(value))

def checkpoint(t,run,step,*,mode=C.A,clock=None,folder='model'):
    p=run/folder/('step-%012d'%step);p.mkdir(parents=True)
    arrays={'leaf_00000':np.array([1.],np.float32),'leaf_00001':np.array(step if clock is None else clock,np.int32),'jax_rng':np.array([0,61],np.uint32)}
    np.savez(p/'arrays.npz',**arrays)
    cfg=config(run,mode);cfg['training_schema']=t.VERSION
    meta=dict(schema='wam_pi05_resume_v1',training_schema=t.VERSION,step=step,optimizer_step=step if clock is None else clock,
        schedule_cursor=4*step,sampler_rng_state={'cursor':4*step,'schedule_sha256':'schedule'},config=cfg,
        data_hash='data_'+mode,normalization={'synthetic':mode},arrays_sha256=S.file_sha(p/'arrays.npz'),
        initialization_sha256='init',leaf_contract=[dict(path='[0].kernel',shape=[1],dtype='float32'),dict(path='[1].count',shape=[],dtype='int32')])
    write_json(p/'metadata.json',meta);write_json(p/'COMPLETE.json',{'metadata_sha256':S.file_sha(p/'metadata.json')})
    return p,meta

def patch_checkpoint_gate(monkeypatch,t):
    # Isolate path/clock/resume from source/config validation, tested separately.
    def verified(p,**expected):
        m=json.loads((p/'metadata.json').read_text())
        for k,v in expected.items():
            if m[k]!=v:raise ValueError('mismatched '+k)
        return m
    monkeypatch.setattr(t,'_verified_metadata',verified)

def test_resume_latest_same_run_only_and_selection_markers(train,tmp_path,monkeypatch):
    ensure_module(train);patch_checkpoint_gate(monkeypatch,train)
    run=tmp_path/'run';old,meta=checkpoint(train,run,0);new,meta=checkpoint(train,run,27,folder='recovery')
    kwargs=dict(data_hash=meta['data_hash'],config=meta['config'],normalization=meta['normalization'])
    assert train.resolve_same_run_resume(run,new,**kwargs)['step']==27
    with pytest.raises(ValueError):train.resolve_same_run_resume(run,old,**kwargs)
    with pytest.raises(ValueError):train.resolve_same_run_resume(tmp_path/'copy',new,**kwargs)
    for name in ('PAIR_SELECTION_LOCKED.json','pair_selection'):
        p=run/name;p.touch()
        with pytest.raises(ValueError):train.resolve_same_run_resume(run,new,**kwargs)
        p.unlink()
    incomplete=run/'model/step-000000000028';incomplete.mkdir()
    with pytest.raises(ValueError):train.resolve_same_run_resume(run,new,**kwargs)

def test_resume_foreign_arm_or_data_clock_rejected(train,tmp_path,monkeypatch):
    ensure_module(train);patch_checkpoint_gate(monkeypatch,train)
    run=tmp_path/'run';p,m=checkpoint(train,run,4,clock=4004)
    with pytest.raises(ValueError):train.resolve_same_run_resume(run,p,data_hash=m['data_hash'],config=m['config'],normalization=m['normalization'])
    for key,value in (('data_hash','other'),('config',dict(m['config'],mode=C.B)),('normalization',{})):
        expected={k:m[k] for k in ('data_hash','config','normalization')};expected[key]=value
        with pytest.raises(ValueError):train.resolve_same_run_resume(run,p,**expected)

def test_source_map_includes_complete_dependencies(train):
    ensure_module(train);sources=train.source_hashes()
    for name in ('v8/action_encoding.py','v8/training_data.py','v8/train.py','v8/evaluation.py','v7/model.py',
                 'v7/coverage_pilot_schedule.py','v7/coverage_pilot_evaluation.py','v7/coverage_pilot_selection.py'):
        assert name in sources
    assert all(sources[k]==v for k,v in train.data_api.EXPECTED_PARENT_SOURCES.items())

def test_verify_only_does_not_initialize_or_write(train,tmp_path,monkeypatch):
    ensure_module(train)
    fake=SimpleNamespace(mode=C.A,data_hash='synthetic',normalization={},manifest={},dynamics={})
    schedule={'total_steps':14600,'total_draws':58400,'valid_window_count':12759,'candidate_steps':[0]}
    monkeypatch.setattr(train,'load_inputs',lambda *a:(fake,schedule,{'public':'hash'}, {'path':'/spec','sha256':'spec'}))
    monkeypatch.setattr(train,'source_hashes',lambda:{'synthetic':'hash'})
    monkeypatch.setattr(train,'validate_pair_input',lambda *a:None)
    args=SimpleNamespace(manifest=tmp_path/'input/manifest',mode=C.A,output=tmp_path/'absent',paired_initialization=None,resume=None,stop_at=None,verify_only=True)
    r=train.run(args)
    assert r['verified'] and r['mode']==C.A and r['gpu_initialized'] is False
    assert not args.output.exists()

def test_stop_limits_and_pair_mode_admission(train,tmp_path):
    ensure_module(train)
    with pytest.raises(ValueError):train.validate_pair_input(C.B,None)
    with pytest.raises(ValueError):train.validate_pair_input(C.A,tmp_path/'anything')
    for value in (-1,14601,True):
        with pytest.raises(ValueError):train.validate_limit(value,0)
    assert train.validate_limit(None,0)==14600 and train.validate_limit(0,0)==0

def test_source_inventory_does_not_import_qualified_loader(train,monkeypatch):
    ensure_module(train)
    def prohibited():raise AssertionError('Source-only inventory must not import the qualified loader')
    monkeypatch.setattr(train.data_api,'_parent_module',prohibited)
    assert len(train.source_hashes())==24

def test_step_zero_arrays_match_fresh_hashes_not_reset_old_counter(train):
    ensure_module(train)
    leaves=[dict(path='[0]kernel',shape=[1],dtype='float32'),dict(path='[1]adam.count',shape=[],dtype='int32'),dict(path='[1]adam.mu',shape=[1],dtype='float32')]
    arrays={'leaf_00000':np.array([1.],np.float32),'leaf_00001':np.array(0,np.int32),'leaf_00002':np.array([0.],np.float32),'jax_rng':np.array([0,61],np.uint32)}
    train.verify_step_zero_arrays(arrays,leaves,numerical(train))
    arrays['leaf_00000']=np.array([2.],np.float32)
    with pytest.raises(ValueError):train.verify_step_zero_arrays(arrays,leaves,numerical(train))

def test_checkpoint_binding_verifies_real_files_and_new_identity(train,tmp_path,monkeypatch):
    ensure_module(train)
    from test_action_encoding import episodes
    p,m=checkpoint(train,tmp_path/'run',1)
    c=m['config'];c['source_sha256']=train.source_hashes();c['paired_initialization']=None
    m['normalization']={'mode':C.A,'residual_actions':C.fit_stats(episodes(),C.A)}
    c['dataset_binding']['normalization_sha256']=train.data_api._digest(m['normalization'])
    c['dataset_binding']['residual_stats_sha256']=C.normalization_digest(m['normalization']['residual_actions'],mode=C.A)
    initial=train.make_initialization(c,numerical(train),None)
    write_json(p.parents[1]/'initialization.json',initial);m['initialization_sha256']=S.file_sha(p.parents[1]/'initialization.json')
    write_json(p/'metadata.json',m);write_json(p/'COMPLETE.json',{'metadata_sha256':S.file_sha(p/'metadata.json')})
    write_json(p.parents[1]/'configuration.json',{'config':c,'data_hash':m['data_hash'],'normalization':m['normalization']})
    b=train.checkpoint_binding(p)
    from sim.auscultation_pi05_v8.evaluation import BINDING_KEYS
    assert set(b)==BINDING_KEYS and b['schema']=='wam_horizon_pair_checkpoint_binding_v1'
    assert b['mode']==C.A and b['local_step']==b['optimizer_step']==1
    original=copy.deepcopy(m)
    m['normalization']['unapproved_change']=True
    write_json(p/'metadata.json',m);write_json(p/'COMPLETE.json',{'metadata_sha256':S.file_sha(p/'metadata.json')})
    write_json(p.parents[1]/'configuration.json',{'config':c,'data_hash':m['data_hash'],'normalization':m['normalization']})
    with pytest.raises(ValueError):train.checkpoint_binding(p)
    m=original
    write_json(p/'metadata.json',m);write_json(p/'COMPLETE.json',{'metadata_sha256':S.file_sha(p/'metadata.json')})
    write_json(p.parents[1]/'configuration.json',{'config':c,'data_hash':m['data_hash'],'normalization':m['normalization']})
    monkeypatch.setattr(train,'source_hashes',lambda:{'changed':'source'})
    with pytest.raises(ValueError):train.checkpoint_binding(p)
    monkeypatch.undo()
    (p/'arrays.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):train.checkpoint_binding(p)


def test_same_run_context_requires_budget_and_all13_reports(train,tmp_path,monkeypatch):
    ensure_module(train)
    from sim.auscultation_pi05_v8 import evaluation as E
    run=tmp_path/'run';run.mkdir();c=config(run);data=SimpleNamespace(mode=C.A,data_hash='data',normalization={'synthetic':True})
    schedule={'total_steps':14600,'candidate_steps':list(E.EXPECTED_STEPS)}
    write_json(run/'configuration.json',dict(config=c,manifest='/synthetic/manifest',manifest_sha256=train.data_api.MANIFEST_SHA256,data_hash='data',normalization=data.normalization))
    receipt=dict(schema=train.VERSION,mode=C.A,training_complete=True,local_step=14600,optimizer_step=14600,schedule_cursor=58400,error=None,interrupted=False)
    write_json(run/'training_run.json',receipt);write_json(run/'diagnostic_schedule.json',dict(data_hash='data',rows=['synthetic']))
    monkeypatch.setattr(train,'load_inputs',lambda *args:(data,schedule,{},{}))
    monkeypatch.setattr(train,'verify_configuration',lambda *args:None)
    monkeypatch.setattr(train,'_load_initialization',lambda *args:None)
    monkeypatch.setattr(E,'build_diagnostic_schedule',lambda *args:['synthetic'])
    monkeypatch.setattr(train,'checkpoint_binding',lambda p:dict(local_step=int(p.name[5:])))
    seen=[]
    def verify(p,*args):
        if not (p/'report.json').is_file():raise ValueError('Missing report')
        seen.append(p)
    monkeypatch.setattr(E,'verify_checkpoint_report',verify)
    for s in E.EXPECTED_STEPS:write_json(run/'diagnostics'/('local_%06d'%s)/'report.json',{'synthetic':True})
    assert train.verified_run_context(run)['completed'] and len(seen)==13
    (seen[0]/'report.json').unlink()
    with pytest.raises(ValueError):train.verified_run_context(run)
    for field,value in (('mode',C.B),('local_step',1217),('optimizer_step',18600),('training_complete',False),('interrupted',True),('error','failed')):
        write_json(run/'training_run.json',dict(receipt,**{field:value}))
        with pytest.raises(ValueError):train.verified_run_context(run)


def test_actual_cpu_jax_leaf_layout_and_fresh_adam_hashes(train):
    ensure_module(train)
    import os
    if os.environ.get('JAX_PLATFORMS')!='cpu':pytest.skip('Only explicit CPU JAX environment allowed')
    jax=pytest.importorskip('jax');jnp=pytest.importorskip('jax.numpy');optax=pytest.importorskip('optax')
    assert all(d.platform=='cpu' for d in jax.devices())
    parameters={'kernel':jnp.ones((2,3),jnp.float32)}
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(5e-5));state=optimizer.init(parameters)
    runtime=dict(trainable=parameters,frozen={'backbone':jnp.ones((2,2),jnp.bfloat16)},opt_state=state)
    key=jax.random.key(61);numeric=train._numeric_initialization(runtime,key,jax);train._validate_numerical(numeric)
    leaves=jax.tree_util.tree_flatten_with_path((parameters,state))[0]
    contracts=[dict(path=jax.tree_util.keystr(p),shape=list(v.shape),dtype=str(v.dtype)) for p,v in leaves]
    arrays={'leaf_%05d'%i:np.asarray(v) for i,(p,v) in enumerate(leaves)};arrays['jax_rng']=np.asarray(jax.random.key_data(key))
    train.verify_optimizer_age(arrays,contracts,0);train.verify_step_zero_arrays(arrays,contracts,numeric)
