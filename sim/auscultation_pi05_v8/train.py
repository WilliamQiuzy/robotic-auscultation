"""Fresh official-base paired codec training; CPU verification is read-only.

The original model, loss, optimizer, schedule and sampler are reused. This is
neither the old4000 continuation nor a renamed V7 checkpoint. Actual execution
requires the existing OpenPI GPU environment and owns the shared GPU lock.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import uuid
import numpy as np
from sim.auscultation_pi05_v7 import coverage_pilot_schedule as schedule_api
from . import action_encoding as codec,training_data as data_api

VERSION='wam_horizon_pair_training_v1'
RECIPE='pi05_v8_horizon_pair_fresh_base_v1'
INITIALIZATION_SCHEMA='wam_horizon_pair_fresh_initialization_v1'
ROOT=Path('/home/william/wam-vla-v2')
PAIR_SPEC_PATH=ROOT/'diagnostics/coverage_repair_design_v1/experiment_spec_v2.json'
PAIR_SPEC_SHA256='f6f23887cc49932e1bd9ba51c42af90365350ff019ae1278249fcb0c8b4cf162'
SEED=61
LEARNING_RATE=5e-5
TOTAL_STEPS=14600
LOCK_NAMES=('PAIR_SELECTION_LOCKED.json','pair_selection','PILOT_SELECTION_LOCKED.json','SELECTION_LOCKED.json','FINAL_TEST_STARTED.json')
ANCESTRY_SOURCES={'coverage_pilot_schedule.py':'ff8d09b8c4d66e51341021b3ba8485de4a1e735a6559a31e4e717420af6d3b65',
 'coverage_pilot_evaluation.py':'87410026c3c07167badd6feee8c19c075ba2a90624289a1a0218dcf8070d820c',
 'coverage_pilot_selection.py':'c678909c90ede4b2c526bfa90bf3de742124cdc0fd2d1030ba76122803bb02eb'}


def need(value,message):
    if not value:raise ValueError(message)


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def parent_module():
    from sim.auscultation_pi05_v6 import train_openpi
    return train_openpi


def source_hashes():
    own=Path(__file__).parent;old=own.with_name('auscultation_pi05_v7');sources={}
    # Same 19-file data inventory, without importing the qualified loader.
    for name,expected in data_api.EXPECTED_PARENT_SOURCES.items():
        actual=schedule_api.file_sha(own.parent/('auscultation_pi05_'+name))
        need(actual==expected,'Original 17 training sources changed');sources[name]=actual
    sources.update({'v8/'+n:schedule_api.file_sha(own/n) for n in ('action_encoding.py','training_data.py')})
    for name,expected in ANCESTRY_SOURCES.items():
        actual=schedule_api.file_sha(old/name);need(actual==expected,'Frozen schedule/evaluation/selection source changed')
        sources['v7/'+name]=actual
    sources.update({'v8/'+name:schedule_api.file_sha(own/name) for name in ('train.py','evaluation.py')})
    return sources


def freeze(path,value):
    path=Path(path);raw=schedule_api.json_bytes(value)
    if path.exists():need(path.is_file() and not path.is_symlink() and path.read_bytes()==raw,'Frozen file differs: '+str(path));return
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def separate(output,protected):
    output=Path(output).resolve()
    for p in protected:
        p=Path(p).resolve();need(output!=p and p not in output.parents and output not in p.parents,'Output overlaps protected input: '+str(p))


def records_for(dataset):
    ids=[e['episode_id'] for e in dataset.episodes if e['split']=='train']
    need(len(ids)==8 and ids==dataset.normalization['residual_actions']['fit_episode_ids'],'Exact eight TRAIN identities required')
    records=[]
    for ti,(ei,start) in enumerate(dataset.indices['train']):
        e=dataset.episodes[ei];need(e['split']=='train','Heldout row in training schedule')
        records.append(dict(train_index=ti,episode_id=e['episode_id'],episode_index=int(ei),start=int(start),phase=str(e['phases'][start]),split='train'))
    return records,ids


def load_inputs(manifest,mode):
    source_hashes()
    dataset=data_api.load_dataset(manifest,mode)  # canonical qualification; never a caller-supplied view
    records,ids=records_for(dataset);schedule=schedule_api.build_schedule(records,ids)
    need(schedule['total_steps']==TOTAL_STEPS and schedule['total_draws']==58400 and schedule['valid_window_count']==12759,'Original complete schedule differs')
    need(not PAIR_SPEC_PATH.is_symlink() and schedule_api.file_sha(PAIR_SPEC_PATH)==PAIR_SPEC_SHA256,'Approved paired specification differs')
    spec=dict(path=str(PAIR_SPEC_PATH),sha256=PAIR_SPEC_SHA256)
    _,_,assets=parent_module()._verify_public_assets(ROOT)
    return dataset,schedule,assets,spec


def fingerprint_leaves(pairs):
    """Hash one materialized leaf at a time; never duplicate the full backbone."""
    result=[];seen=set()
    for path,value in pairs:
        need(isinstance(path,str) and path and path not in seen,'Unique leaf paths required');seen.add(path)
        a=np.asarray(value)
        need(a.dtype.kind in 'biuf' or str(a.dtype)=='bfloat16','Numeric model leaf required')
        need(np.isfinite(a).all(),'Nonfinite initialization leaf')
        # uint8 view avoids a second leaf-sized tobytes allocation (including bf16).
        raw=np.ascontiguousarray(a).reshape(-1).view(np.uint8)
        result.append(dict(path=path,shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(memoryview(raw)).hexdigest()))
    need(result,'Nonempty initialization partition required')
    return result


def _validate_numerical(numerical):
    need(set(numerical)=={'trainable','frozen','optimizer','jax_rng','optimizer_step','rng_seed'},'Exact initialization numerical partitions required')
    need(type(numerical['optimizer_step']) is int and numerical['optimizer_step']==0 and numerical['rng_seed']==61,'Fresh Adam0/key61 required')
    for name in ('trainable','frozen','optimizer','jax_rng'):
        leaves=numerical[name];need(isinstance(leaves,list) and leaves,'Nonempty initialization partition')
        paths=set()
        for leaf in leaves:
            need(set(leaf)=={'path','shape','dtype','sha256'} and isinstance(leaf['path'],str) and leaf['path'] not in paths,'Invalid or duplicate leaf contract')
            paths.add(leaf['path'])
            need(isinstance(leaf['shape'],list) and all(type(x) is int and x>=0 for x in leaf['shape']) and isinstance(leaf['dtype'],str)
                 and isinstance(leaf['sha256'],str) and re.fullmatch('[0-9a-f]{64}',leaf['sha256']),'Invalid leaf shape/dtype/hash')
    counts=[x for x in numerical['optimizer'] if x['path'].endswith('.count')]
    need(len(counts)==1 and counts[0]['shape']==[] and counts[0]['dtype']=='int32'
         and counts[0]['sha256']==hashlib.sha256(np.asarray(0,np.int32).tobytes()).hexdigest(),'Exactly one zero Adam counter required')
    need(numerical['jax_rng']==fingerprint_leaves([('key_data',np.array([0,61],np.uint32))]),'Initial JAX key differs from key61')


def _verify_initialization(receipt):
    need(isinstance(receipt,dict) and receipt.get('schema')==INITIALIZATION_SCHEMA,'Fresh initialization receipt required')
    codec.action_encoding_binding(receipt.get('mode'))
    need(receipt.get('seed')==61 and receipt.get('optimizer_step')==0,'Fresh initialization seed/age differs')
    _validate_numerical(receipt['numerical_initialization'])
    need(receipt.get('integrity_sha256')==schedule_api.digest({k:v for k,v in receipt.items() if k!='integrity_sha256'}),'Initialization integrity differs')
    return receipt


def make_initialization(config,numerical,paired_a):
    _validate_numerical(numerical)
    need(config['seed']==61 and config['learning_rate']==5e-5 and config['batch_size']==4,'Original initialization settings required')
    mode=config['mode'];codec.action_encoding_binding(mode)
    if mode==codec.A:need(paired_a is None,'A must be independently initialized without a pair input')
    else:
        need(paired_a is not None,'B requires the actual A initialization receipt');_verify_initialization(paired_a)
        need(paired_a['mode']==codec.A,'B must compare an A initialization')
        for key in ('seed','source_sha256','assets','pair_spec'):
            need(paired_a[key]==config[key],'Paired initialization '+key+' differs')
        need(paired_a['numerical_initialization']==numerical,'Fresh A/B numerical initialization differs')
    result=dict(schema=INITIALIZATION_SCHEMA,mode=mode,training_run_path=config['training_run_path'],seed=61,optimizer_step=0,
        source_sha256=config['source_sha256'],assets=config['assets'],pair_spec=config['pair_spec'],config_sha256=schedule_api.digest(config),
        numerical_initialization=numerical,paired_a_numerics_equal=mode==codec.B,
        paired_a_receipt_integrity=None if paired_a is None else paired_a['integrity_sha256'])
    result['integrity_sha256']=schedule_api.digest(result)
    return result


def validate_pair_input(mode,path):
    codec.action_encoding_binding(mode)
    if mode==codec.A:need(path is None,'A forbids paired initialization input');return None
    need(path is not None,'B requires explicit --paired-initialization from A')
    path=Path(path);need(path.is_file() and not path.is_symlink() and path.name=='initialization.json','Canonical A initialization file required')
    value=_verify_initialization(read(path));need(value['mode']==codec.A and Path(value['training_run_path']).resolve()==path.resolve().parent,'Original A run initialization required')
    need(value['source_sha256']==source_hashes(),'A initialization source differs')
    return dict(path=str(path.resolve()),sha256=schedule_api.file_sha(path))


def validate_limit(stop_at,step):
    limit=TOTAL_STEPS if stop_at is None else stop_at
    need(type(limit) is int and step<=limit<=TOTAL_STEPS,'Invalid bounded stop-at')
    return limit


def make_config(dataset,schedule,schedule_binding,assets,spec,output,paired):
    parent=parent_module()
    return dict(training_schema=VERSION,recipe=RECIPE,mode=dataset.mode,training_run_path=str(Path(output).resolve()),
        source_sha256=source_hashes(),trainer_sha256=schedule_api.file_sha(__file__),dataset_binding=data_api.dataset_binding(dataset),
        parent_data_hash=dataset.parent.data_hash,openpi_commit=parent.OPENPI_COMMIT,assets=assets,pair_spec=spec,paired_initialization=paired,
        seed=61,learning_rate=5e-5,batch_size=4,total_steps=14600,candidate_steps=schedule['candidate_steps'],schedule_binding=schedule_binding,
        evaluate_every='predeclared_candidate_steps',save_every='predeclared_candidates_plus_interruption_recovery',
        action_dim=7,action_horizon=10,state_width=41,max_token_len=256,
        model_architecture='unmodified official pi05 with V6 seven-channel LoRA/projection partition',
        initialization='independent fresh official base; Adam0 and JAX key61; no trained checkpoint bootstrap',
        action_semantics='absolute_joint_command_v1',model_action_semantics='future_command_minus_fixed_observation_measured_q',
        canonical_phase_instructions=dict(parent.INSTRUCTIONS),instruction_policy='canonical-phase-and-target-v1',
        observation_contract=dataset.manifest['observation_contract'],action_encoding_binding=dataset.normalization['action_encoding_binding'],
        dynamics_contract_sha256=dataset.dynamics['contract_sha256'],dynamics_recipe_sha256=dataset.dynamics['recipe_sha256'],
        execution_contract_sha256=dataset.dynamics['execution_contract_sha256'])


def verify_configuration(dataset,config,schedule,assets,spec):
    run=config.get('training_run_path');need(isinstance(run,str) and str(Path(run).resolve())==run,'Canonical original run required')
    paired=validate_pair_input(dataset.mode,None if config.get('paired_initialization') is None else config['paired_initialization']['path'])
    expected=make_config(dataset,schedule,config['schedule_binding'],assets,spec,run,paired)
    need(config==expected,'Paired configuration, source, mode or dataset differs')
    need(schedule_api.load_schedule(config['schedule_binding']['path'],config['schedule_binding'])==schedule,'Saved complete schedule differs')
    return expected


def verify_optimizer_age(arrays,leaf_contract,local_step):
    need(type(local_step) is int and 0<=local_step<=TOTAL_STEPS,'Invalid local step')
    counts=[i for i,c in enumerate(leaf_contract) if c['path'].startswith('[1]') and c['path'].endswith('.count')]
    need(len(counts)==1,'Exactly one Adam count required');i=counts[0];c=leaf_contract[i];v=np.asarray(arrays['leaf_%05d'%i])
    need(c['dtype']=='int32' and c['shape']==[] and v.dtype==np.int32 and v.shape==() and int(v)==local_step,'Fresh optimizer age must equal local step')
    key=np.asarray(arrays['jax_rng']);need(key.dtype==np.uint32 and key.shape==(2,),'Actual JAX key data required')


def verify_step_zero_arrays(arrays,leaf_contract,numerical):
    """A reset Adam counter cannot relabel previously trained step0 weights."""
    _validate_numerical(numerical)
    need(set(arrays)=={'leaf_%05d'%i for i in range(len(leaf_contract))}|{'jax_rng'},'Step0 array keys differ')
    for prefix,group in (('[0]','trainable'),('[1]','optimizer')):
        def pairs():
            for i,leaf in enumerate(leaf_contract):
                if leaf['path'].startswith(prefix):
                    value=np.asarray(arrays['leaf_%05d'%i])
                    need(list(value.shape)==leaf['shape'] and str(value.dtype)==leaf['dtype'],'Step0 leaf shape/dtype differs')
                    yield leaf['path'][3:],value
        need(fingerprint_leaves(pairs())==numerical[group],'Step0 '+group+' differs from fresh initialization')
    need(fingerprint_leaves([('key_data',arrays['jax_rng'])])==numerical['jax_rng'],'Step0 key differs')


def _load_initialization(config):
    path=Path(config['training_run_path'])/'initialization.json';need(path.is_file() and not path.is_symlink(),'Initialization receipt required')
    receipt=_verify_initialization(read(path))
    for key in ('mode','training_run_path','seed','source_sha256','assets','pair_spec'):
        need(receipt[key]==config[key],'Initialization/config '+key+' differs')
    need(receipt['config_sha256']==schedule_api.digest(config),'Initialization configuration changed')
    paired=config['paired_initialization']
    if paired is not None:
        need(schedule_api.file_sha(paired['path'])==paired['sha256'],'A initialization file changed')
        a=_verify_initialization(read(paired['path']))
        need(make_initialization(config,receipt['numerical_initialization'],a)==receipt,'Paired initialization proof differs')
    else:need(make_initialization(config,receipt['numerical_initialization'],None)==receipt,'Initialization proof differs')
    return receipt,schedule_api.file_sha(path)


def _verified_metadata(path,*,data_hash,config,normalization):
    path=Path(path)
    need(not path.is_symlink() and all((path/n).is_file() and not (path/n).is_symlink() for n in ('metadata.json','arrays.npz','COMPLETE.json')),'Regular complete checkpoint required')
    m=parent_module()._verified_checkpoint_metadata(path,data_hash=data_hash,config=config,normalization=normalization)
    need(config.get('training_schema')==VERSION and config.get('recipe')==RECIPE and config.get('source_sha256')==source_hashes(),'New truthful paired training sources required')
    need(config.get('mode')==normalization.get('mode') and config['dataset_binding']['mode']==config['mode']
         and config['dataset_binding']['data_hash']==data_hash,'Foreign arm/data checkpoint')
    stats_sha=codec.validate_stats(normalization['residual_actions'],mode=config['mode'])
    need(config['dataset_binding'].get('normalization_sha256')==data_api._digest(normalization)
         and config['dataset_binding'].get('residual_stats_sha256')==stats_sha,'Checkpoint normalization differs from qualified dataset binding')
    run=Path(config['training_run_path']);need(path.resolve().parent.parent==run and str(run.resolve())==str(run),'Foreign/copied checkpoint run')
    need(re.fullmatch(r'step-[0-9]{12}',path.name) is not None and m['step']==int(path.name[5:]),'Checkpoint directory clock differs')
    need(m.get('training_schema')==VERSION and type(m.get('optimizer_step')) is int and m['optimizer_step']==m['step']
         and m.get('schedule_cursor')==4*m['step'] and m.get('sampler_rng_state')==dict(cursor=4*m['step'],schedule_sha256=config['schedule_binding']['sha256']),
         'Fresh checkpoint clock/cursor differs')
    initial,ih=_load_initialization(config);need(m.get('initialization_sha256')==ih,'Checkpoint initialization lineage differs')
    with np.load(path/'arrays.npz',allow_pickle=False) as arrays:
        verify_optimizer_age(arrays,m['leaf_contract'],m['step'])
        if m['step']==0:verify_step_zero_arrays(arrays,m['leaf_contract'],initial['numerical_initialization'])
    return m


def checkpoint_binding(path):
    path=Path(path).resolve(strict=True);m=read(path/'metadata.json');c=m['config'];run=Path(c['training_run_path'])
    saved=read(run/'configuration.json')
    need(saved['config']==c and saved['data_hash']==m['data_hash'] and saved['normalization']==m['normalization'],'Original run configuration differs')
    _verified_metadata(path,data_hash=saved['data_hash'],config=c,normalization=saved['normalization'])
    return dict(schema='wam_horizon_pair_checkpoint_binding_v1',mode=c['mode'],checkpoint_path=str(path),local_step=m['step'],optimizer_step=m['optimizer_step'],
        metadata_sha256=schedule_api.file_sha(path/'metadata.json'),arrays_sha256=m['arrays_sha256'],complete_sha256=schedule_api.file_sha(path/'COMPLETE.json'),
        data_hash=m['data_hash'],normalization_sha256=schedule_api.digest(m['normalization']),config_sha256=schedule_api.digest(c),
        schedule_sha256=c['schedule_binding']['sha256'],source_sha256=c['source_sha256'])


def resolve_same_run_resume(output,resume,*,data_hash,config,normalization):
    output=Path(output).resolve();resume=Path(resume).resolve()
    need(config.get('training_run_path')==str(output),'Copied/different run cannot resume')
    need(not any((output/n).exists() for n in LOCK_NAMES),'Selected/consumed run cannot resume')
    paths=[]
    for name in ('model','recovery'):
        folder=output/name
        if not folder.exists():continue
        for p in folder.iterdir():
            if re.fullmatch(r'step-[0-9]{12}',p.name):
                need(p.is_dir() and not p.is_symlink() and (p/'COMPLETE.json').is_file(),'Incomplete/corrupt declared checkpoint');paths.append(p.resolve())
    need(paths,'No complete same-run checkpoint');latest=max(paths,key=lambda p:int(p.name[5:]))
    need(resume==latest and sum(p.name==latest.name for p in paths)==1,'Only unambiguous latest complete same-run checkpoint can resume')
    m=_verified_metadata(latest,data_hash=data_hash,config=config,normalization=normalization)
    need(m['step']==int(latest.name[5:]) and m.get('optimizer_step')==m['step'] and m.get('schedule_cursor')==4*m['step'],'Fresh checkpoint age/cursor differs')
    with np.load(latest/'arrays.npz',allow_pickle=False) as arrays:verify_optimizer_age(arrays,m['leaf_contract'],m['step'])
    return m


def verified_run_context(run):
    run=Path(run).resolve(strict=True);saved=read(run/'configuration.json');config=saved['config']
    need(config.get('training_run_path')==str(run),'Copied run cannot select')
    dataset,schedule,assets,spec=load_inputs(saved['manifest'],config['mode']);verify_configuration(dataset,config,schedule,assets,spec)
    need(saved['data_hash']==dataset.data_hash and saved['normalization']==dataset.normalization and saved['manifest_sha256']==data_api.MANIFEST_SHA256,'Saved dataset binding changed')
    _load_initialization(config)
    receipt=read(run/'training_run.json')
    need(receipt.get('schema')==VERSION and receipt.get('mode')==dataset.mode and receipt.get('training_complete') is True
         and receipt.get('local_step')==receipt.get('optimizer_step')==TOTAL_STEPS and receipt.get('schedule_cursor')==58400
         and receipt.get('error') is None and receipt.get('interrupted') is False,'Complete uninterrupted 14600-update receipt required')
    paths=[run/'model'/('step-%012d'%s) for s in schedule['candidate_steps']]
    from . import evaluation
    rows=evaluation.build_diagnostic_schedule(dataset)
    need(read(run/'diagnostic_schedule.json')==dict(data_hash=dataset.data_hash,rows=rows),'Diagnostic rows differ')
    for p in paths:
        binding=checkpoint_binding(p);evaluation.verify_checkpoint_report(run/'diagnostics'/('local_%06d'%binding['local_step']),dataset,binding)
    return dict(dataset=dataset,config=config,schedule=schedule,candidate_paths=paths,completed=True)


def _numeric_initialization(runtime,rng,jax):
    def leaves(tree):
        for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
            yield jax.tree_util.keystr(path),np.asarray(jax.device_get(value))
    return dict(trainable=fingerprint_leaves(leaves(runtime['trainable'])),frozen=fingerprint_leaves(leaves(runtime['frozen'])),
        optimizer=fingerprint_leaves(leaves(runtime['opt_state'])),jax_rng=fingerprint_leaves([('key_data',np.asarray(jax.random.key_data(rng)))]),optimizer_step=0,rng_seed=61)


def run(args):
    output=Path(args.output).resolve();manifest=Path(args.manifest).resolve()
    protected=[Path(__file__).resolve().parents[2],manifest.parent,ROOT/'public-assets',PAIR_SPEC_PATH.parent]
    if args.paired_initialization is not None:protected.append(Path(args.paired_initialization).resolve().parent)
    separate(output,protected);validate_limit(args.stop_at,0)
    need(not any((output/n).exists() for n in LOCK_NAMES),'Locked run')
    need(not output.exists() or args.resume is not None,'Fresh output or explicit same-run resume required')
    paired=validate_pair_input(args.mode,args.paired_initialization)
    dataset,schedule,assets,spec=load_inputs(manifest,args.mode)
    if paired is not None:
        a=_verify_initialization(read(paired['path']));need(a['assets']==assets and a['pair_spec']==spec,'A initialization assets/spec differ')
    if args.verify_only:
        if args.resume:
            saved=read(output/'configuration.json');verify_configuration(dataset,saved['config'],schedule,assets,spec)
            need(saved['config']['paired_initialization']==paired,'Resume pair input changed')
            resolve_same_run_resume(output,args.resume,data_hash=dataset.data_hash,config=saved['config'],normalization=dataset.normalization)
        return dict(schema=VERSION,mode=args.mode,verified=True,train_episodes=8,valid_windows=schedule['valid_window_count'],total_steps=schedule['total_steps'],total_draws=schedule['total_draws'],schedule_sha256=schedule_api.digest(schedule),source_sha256=source_hashes(),gpu_initialized=False,output_changed=False)
    output.mkdir(parents=True,exist_ok=True);schedule_path=output/'training_schedule.json'
    if args.resume:
        saved=read(output/'configuration.json');config=saved['config'];verify_configuration(dataset,config,schedule,assets,spec)
        need(config['paired_initialization']==paired,'Resume pair input changed')
        restored=resolve_same_run_resume(output,args.resume,data_hash=dataset.data_hash,config=config,normalization=dataset.normalization);step=restored['step']
    else:
        sb=schedule_api.save_schedule(schedule_path,schedule);config=make_config(dataset,schedule,sb,assets,spec,output,paired);step=0
        freeze(output/'configuration.json',dict(config=config,data_hash=dataset.data_hash,normalization=dataset.normalization,manifest=str(manifest),manifest_sha256=data_api.MANIFEST_SHA256))
    limit=validate_limit(args.stop_at,step)
    from . import evaluation
    diagnostic=evaluation.build_diagnostic_schedule(dataset);freeze(output/'diagnostic_schedule.json',dict(data_hash=dataset.data_hash,rows=diagnostic))
    stop=[False];handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for sig in handlers:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    segments=output/'training_segments';segments.mkdir(exist_ok=True);segment=segments/(utc().replace(':','-')+'-'+uuid.uuid4().hex+'.json')
    receipt=dict(schema=VERSION,mode=args.mode,started_utc=utc(),pid=os.getpid(),training_run_path=str(output),local_step=step,optimizer_step=step,
        schedule_cursor=4*step,training_complete=False,interrupted=False,error=None,resume=None if args.resume is None else str(Path(args.resume).resolve()),source_sha256=config['source_sha256'],closed_loop_validated=False)
    parent=parent_module()
    if (output/'training_run.json').exists():
        with segment.with_suffix('.previous.json').open('xb') as f:f.write((output/'training_run.json').read_bytes())
    def save_receipt():
        receipt.update(local_step=step,optimizer_step=step,schedule_cursor=4*step,interrupted=stop[0]);parent._atomic_json(segment,receipt);parent._atomic_json(output/'training_run.json',receipt)
    save_receipt()
    try:
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        import optax
        from openpi.models import model as model_api
        from sim.auscultation_pi05_v7 import model as implementation
        runtime=implementation.initialize(61,5e-5);need(runtime['assets']==assets,'Official public assets changed during initialization')
        rng=jax.random.key(61);numeric=_numeric_initialization(runtime,rng,jax)
        initial=make_initialization(config,numeric,None if paired is None else read(paired['path']))
        if args.resume:
            saved_initial,initial_sha=_load_initialization(config);need(saved_initial==initial,'Independent fresh initialization changed before restore')
            restored,key=implementation.restore(runtime,args.resume,dataset.data_hash,config,dataset.normalization);rng=jax.random.wrap_key_data(key)
        else:
            freeze(output/'initialization.json',initial);initial_sha=schedule_api.file_sha(output/'initialization.json')
        sample=implementation.sampler(runtime,steps=10)
        def objective(parameters,frozen,obs,actions,key):
            model=nnx.merge(runtime['graph'],parameters,frozen);model.train();return model.compute_loss(key,obs,actions,train=True).mean()
        gradient=jax.jit(jax.value_and_grad(objective,argnums=0))
        @jax.jit
        def update(parameters,state,grads):
            changes,state=runtime['optimizer'].update(grads,state,parameters);return optax.apply_updates(parameters,changes),state
        def save_checkpoint():
            need(source_hashes()==config['source_sha256'] and schedule_api.file_sha(schedule_path)==config['schedule_binding']['sha256'],'Training source or schedule changed')
            leaves=jax.tree.leaves((runtime['trainable'],runtime['opt_state']));arrays={'leaf_%05d'%i:np.asarray(v) for i,v in enumerate(jax.device_get(leaves))};arrays['jax_rng']=np.asarray(jax.random.key_data(rng))
            verify_optimizer_age(arrays,runtime['leaf_contract'],step)
            if step==0:verify_step_zero_arrays(arrays,runtime['leaf_contract'],initial['numerical_initialization'])
            metadata=dict(training_schema=VERSION,data_hash=dataset.data_hash,config=config,normalization=dataset.normalization,leaf_contract=runtime['leaf_contract'],
                optimizer_step=step,schedule_cursor=4*step,sampler_rng_state=dict(cursor=4*step,schedule_sha256=config['schedule_binding']['sha256']),
                initialization_sha256=initial_sha,best_step=None,best_validation_loss=None,heldout_axes=['diagnostic_original_train_only'])
            return parent.save_checkpoint(output/('model' if step in schedule['candidate_steps'] else 'recovery'),step=step,arrays=arrays,metadata=metadata)
        def diagnose(checkpoint):
            dest=output/'diagnostics'/('local_%06d'%step);binding=checkpoint_binding(checkpoint)
            if dest.exists():evaluation.verify_checkpoint_report(dest,dataset,binding);return
            predicted=[];normalized=[]
            for row in diagnostic:
                obs,_=data_api.make_batch(dataset,'train',[row['train_index']],runtime['tokenizer'])
                value=np.asarray(sample(runtime['trainable'],runtime['frozen'],model_api.Observation(**jax.device_put(obs)),jax.random.key(917)))
                need(value.shape==(1,10,7) and np.isfinite(value).all(),'Malformed diagnostic output')
                e=dataset.episodes[row['episode_index']];state=e['state'][row['start']]
                normalized.append(value[0]);predicted.append(data_api.decode_prediction(dataset,value[0],state,row['phase']))
            evaluation.save_checkpoint_report(dest,dataset,np.asarray(predicted),np.asarray(normalized),binding)
        last_saved=step
        if not args.resume:
            checkpoint=save_checkpoint();receipt['latest_checkpoint']=str(checkpoint)
            if not stop[0]:diagnose(checkpoint)
        elif step in schedule['candidate_steps'] and not stop[0]:diagnose(Path(args.resume))
        while step<limit and not stop[0]:
            indices=schedule_api.batch_indices(schedule,step);obs,actions=data_api.make_batch(dataset,'train',indices,runtime['tokenizer']);rng,key=jax.random.split(rng)
            loss,grads=gradient(runtime['trainable'],runtime['frozen'],model_api.Observation(**jax.device_put(obs)),jax.device_put(actions),key)
            need(bool(jax.device_get(jnp.all(jnp.stack([jnp.isfinite(g).all() for g in jax.tree.leaves(grads)])))) and np.isfinite(float(loss)),'Nonfinite training update')
            runtime['trainable'],runtime['opt_state']=update(runtime['trainable'],runtime['opt_state'],grads);jax.block_until_ready((runtime['trainable'],runtime['opt_state']));step+=1
            metrics=dict(mode=args.mode,local_step=step,optimizer_step=step,train_loss=float(loss),schedule_cursor=4*step,train_indices=indices.tolist(),segment=str(segment))
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
            if step in schedule['candidate_steps'] or step==limit or stop[0]:
                checkpoint=save_checkpoint();last_saved=step;receipt['latest_checkpoint']=str(checkpoint)
                if step in schedule['candidate_steps'] and not stop[0]:diagnose(checkpoint)
            if step%25==0:print(json.dumps(metrics),flush=True)
            save_receipt()
        if step>last_saved:receipt['latest_checkpoint']=str(save_checkpoint())
        receipt.update(training_complete=step==TOTAL_STEPS and not stop[0],requested_segment_complete=step==limit)
    except BaseException as error:
        receipt['error']=type(error).__name__+': '+str(error);raise
    finally:
        receipt['finished_utc']=utc();save_receipt()
        for sig,handler in handlers.items():signal.signal(sig,handler)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--mode',choices=codec.MODES,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--paired-initialization',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--stop-at',type=int);p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    if args.verify_only:print(json.dumps(run(args),indent=2));return
    with (ROOT/'locks/isaac_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU workload is active')
        print(json.dumps(run(args),indent=2),flush=True)


if __name__=='__main__':main()
