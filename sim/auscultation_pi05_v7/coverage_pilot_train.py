"""Explicit original-eight coverage continuation from the immutable old4000.

This is a new diagnostic run with local steps; it is not fresh-base training,
expanded-50 training, a changed model/codec, or a generalization experiment.
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
from sim.auscultation_pi05_v6 import train_openpi as parent
from .training_data import load_dataset,make_batch,decode_prediction,core_sources
from . import model as implementation
from . import coverage_pilot_schedule as schedule_api

RECIPE='pi05_v7_original8_coverage_continuation_v1'
VERSION='wam_pi05_v7_original8_coverage_pilot_v1'
BOOTSTRAP_STEP=4000
BOOTSTRAP_METADATA_SHA256='43d7cbf06243a76a860212ba7f337ce806ae8166132be2d586d1c79aa4a511d2'
BOOTSTRAP_ARRAYS_SHA256='f6e3f086adbb4dc98d0a756b047443d05c2d0191d60c14378fc0dd1683a31c9a'
MANIFEST_SHA256='ce5da81634870e79cc8442d4ec166387966cc1e7c125942a16ebe98d73dcc9b0'
DATA_HASH='9c40deac72e4b54c2c921867a488022c18c36c34e2cf310ed27bace8d40cf34f'
LOCK_NAMES=('PILOT_SELECTION_LOCKED.json','pilot_selection','SELECTION_LOCKED.json','FINAL_TEST_STARTED.json')

def utc():return datetime.now(timezone.utc).isoformat()
def need(condition,message):
 if not condition:raise ValueError(message)
def _read(path):return parent._read_json(Path(path))
def _freeze(path,value):
 path=Path(path);raw=schedule_api.json_bytes(value)
 if path.exists():need(path.read_bytes()==raw,'Frozen file differs: '+str(path));return
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def _separate(output,protected):
 output=Path(output).resolve()
 for q in protected:
  q=Path(q).resolve();need(output!=q and q not in output.parents and output not in q.parents,'Output overlaps protected input: '+str(q))

def assert_array_identity(original,current):
 need(set(original)==set(current),'Bootstrap checkpoint keys differ')
 hashes={}
 for key in original:
  a,b=np.asarray(original[key]),np.asarray(current[key])
  need(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'Bootstrap array differs: '+key)
  hashes[key]=dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
 return dict(schema='wam_old4000_step0_array_identity_v1',passed=True,arrays=hashes)

def verify_optimizer_age(arrays,leaf_contract,local_step):
 need(type(local_step) is int and local_step>=0,'Invalid local step')
 counts=[(i,c) for i,c in enumerate(leaf_contract) if c['path'].startswith('[1]') and c['path'].endswith('.count')]
 need(len(counts)==1,'Exactly one inherited Adam count required')
 i,c=counts[0];value=np.asarray(arrays['leaf_%05d'%i])
 need(c['dtype']=='int32' and c['shape']==[] and value.dtype==np.int32 and value.shape==() and int(value)==BOOTSTRAP_STEP+local_step,'Optimizer age/reset differs')

def records_for(dataset):
 ids=[e['episode_id'] for e in dataset.episodes if e['split']=='train']
 need(ids==dataset.normalization['residual_actions']['fit_episode_ids'] and len(ids)==8,'Original eight normalization identities required')
 records=[]
 for ti,(ei,start) in enumerate(dataset.indices['train']):
  ep=dataset.episodes[ei];need(ep['split']=='train','Heldout index in training schedule')
  records.append(dict(train_index=ti,episode_id=ep['episode_id'],episode_index=int(ei),start=int(start),phase=str(ep['phases'][start]),split='train'))
 return records,ids

def load_inputs(manifest,bootstrap):
 manifest=Path(manifest).resolve();bootstrap=Path(bootstrap).resolve()
 need(parent._sha(manifest)==MANIFEST_SHA256,'Pilot requires exact original dataset manifest')
 need(parent._sha(bootstrap/'metadata.json')==BOOTSTRAP_METADATA_SHA256,'Pilot requires exact original4000 metadata')
 meta=_read(bootstrap/'metadata.json')
 need(meta['step']==BOOTSTRAP_STEP and meta['arrays_sha256']==BOOTSTRAP_ARRAYS_SHA256,'Wrong bootstrap checkpoint')
 dataset=load_dataset(manifest)
 need(dataset.data_hash==DATA_HASH==meta['data_hash'] and dataset.normalization==meta['normalization'],'Original data/normalization changed')
 old=meta['config'];need(old['source_sha256']==core_sources(),'Inherited 17 training sources changed')
 need(old['trainer_sha256']==parent._sha(Path(__file__).with_name('train_openpi.py')),'Original trainer changed')
 need(old['seed']==61 and old['batch_size']==4 and old['learning_rate']==5e-5,'Inherited optimizer configuration changed')
 parent._verified_checkpoint_metadata(bootstrap,data_hash=dataset.data_hash,config=old,normalization=dataset.normalization)
 records,ids=records_for(dataset);schedule=schedule_api.build_schedule(records,ids)
 return dataset,meta,schedule

def pilot_binding(dataset,bootstrap,schedule_binding,schedule):
 return dict(schema=VERSION,training_recipe=RECIPE,manifest_sha256=MANIFEST_SHA256,data_hash=dataset.data_hash,
  normalization_sha256=schedule_api.digest(dataset.normalization),bootstrap_metadata_sha256=BOOTSTRAP_METADATA_SHA256,
  bootstrap_arrays_sha256=BOOTSTRAP_ARRAYS_SHA256,bootstrap_complete_sha256=parent._sha(Path(bootstrap)/'COMPLETE.json'),
  bootstrap_optimizer_step=4000,schedule_sha256=schedule_binding['sha256'],records_sha256=schedule['records_sha256'],
  additional_steps=schedule['total_steps'],train_episode_ids=schedule['episode_ids'],train_episodes=8,diagnostic_only=True,
  model_representation='unchanged fixed observed q residual decoded to absolute command',normalization_refit=False,
  initialization='exact old4000 parameters, Adam state/count, and JAX flow RNG; new frozen schedule cursor zero',
  sampling_steps=10,sampling_seed=917,heldout_prediction=False,closed_loop_acceptance=False)

def make_config(dataset,meta,manifest,bootstrap,output,schedule_binding,schedule):
 config=dict(meta['config'])
 config.update(recipe=RECIPE,inherited_model_recipe=meta['config']['recipe'],training_run_path=str(Path(output).resolve()),
  trainer_sha256=parent._sha(__file__),pilot_source_sha256=schedule_api.source_hashes(),
  pilot_binding=pilot_binding(dataset,bootstrap,schedule_binding,schedule),
  bootstrap_checkpoint=str(Path(bootstrap).resolve()),bootstrap_configuration=meta['config'],
  schedule_binding=schedule_binding,total_steps=schedule['total_steps'],candidate_steps=schedule['candidate_steps'],
  evaluate_every='predeclared_candidate_steps',save_every='predeclared_candidates_plus_interruption_recovery')
 return config

def verify_configuration(dataset,config,schedule):
 need(config.get('recipe')==RECIPE and config.get('source_sha256')==core_sources(),'Pilot recipe/ancestry differs')
 run=config.get('training_run_path');need(isinstance(run,str) and str(Path(run).resolve())==run,'Canonical original run required')
 need(config.get('pilot_source_sha256')==schedule_api.source_hashes(),'Pilot source changed')
 need(config.get('trainer_sha256')==parent._sha(__file__),'Truthful pilot trainer source required')
 bootstrap=Path(config['bootstrap_checkpoint']);meta=_read(bootstrap/'metadata.json')
 need(parent._sha(bootstrap/'metadata.json')==BOOTSTRAP_METADATA_SHA256,'Bootstrap metadata changed')
 expected=make_config(dataset,meta,None,bootstrap,run,config['schedule_binding'],schedule)
 need(config==expected,'Pilot configuration differs from exact inherited contract')
 need(schedule_api.load_schedule(config['schedule_binding']['path'],config['schedule_binding'])==schedule,'Schedule bytes changed')
 records,ids=records_for(dataset);schedule_api.verify_schedule(schedule,records,ids)

def checkpoint_binding(path):
 path=Path(path).resolve();m=_read(path/'metadata.json')
 parent._verified_checkpoint_metadata(path,data_hash=m['data_hash'],config=m['config'],normalization=m['normalization'])
 need(m.get('optimizer_step')==4000+m['step'] and m.get('schedule_cursor')==4*m['step'],'Checkpoint clock differs')
 return dict(schema='wam_coverage_pilot_checkpoint_binding_v1',checkpoint_path=str(path),local_step=m['step'],optimizer_step=m['optimizer_step'],
  metadata_sha256=parent._sha(path/'metadata.json'),arrays_sha256=m['arrays_sha256'],complete_sha256=parent._sha(path/'COMPLETE.json'),
  data_hash=m['data_hash'],normalization_sha256=schedule_api.digest(m['normalization']),config_sha256=schedule_api.digest(m['config']),
  schedule_sha256=m['config']['schedule_binding']['sha256'],pilot_binding=m['config']['pilot_binding'],pilot_source_sha256=m['config']['pilot_source_sha256'])

def resolve_same_run_resume(output,resume,*,data_hash,config,normalization):
 output=Path(output).resolve();resume=Path(resume).resolve()
 need(config.get('training_run_path')==str(output),'Copied/different run cannot resume')
 need(not any((output/n).exists() for n in LOCK_NAMES),'Selected/test-started run cannot resume')
 paths=[]
 for name in ('model','recovery'):
  folder=output/name
  if not folder.exists():continue
  for p in folder.iterdir():
   if re.fullmatch(r'step-[0-9]{12}',p.name):
    need(p.is_dir() and not p.is_symlink() and (p/'COMPLETE.json').is_file(),'Incomplete/corrupt declared checkpoint')
    paths.append(p.resolve())
 need(paths,'No complete same-run checkpoint')
 latest=max(paths,key=lambda p:int(p.name[5:]));need(resume==latest,'Only latest complete same-run checkpoint can resume')
 need(len([p for p in paths if p.name==latest.name])==1,'Ambiguous same-step checkpoint')
 meta=parent._verified_checkpoint_metadata(latest,data_hash=data_hash,config=config,normalization=normalization)
 step=meta['step'];need(step==int(latest.name[5:]) and 0<=step<=config['total_steps'],'Checkpoint step differs')
 need(meta.get('optimizer_step')==4000+step and meta.get('schedule_cursor')==4*step,'Checkpoint age/cursor differs')
 need(meta.get('sampler_rng_state')==dict(cursor=4*step,schedule_sha256=config['schedule_binding']['sha256']),'Frozen schedule cursor differs')
 with np.load(latest/'arrays.npz',allow_pickle=False) as a:verify_optimizer_age(a,meta['leaf_contract'],step)
 return meta

def verified_run_context(run):
 run=Path(run).resolve();saved=_read(run/'configuration.json');config=saved['config']
 need(config.get('training_run_path')==str(run),'Copied run cannot select/check training completion')
 dataset,meta,schedule=load_inputs(saved['manifest'],config['bootstrap_checkpoint']);verify_configuration(dataset,config,schedule)
 need(saved['normalization']==dataset.normalization and saved['data_hash']==dataset.data_hash,'Run data changed')
 receipt=_read(run/'training_run.json')
 need(receipt.get('training_complete') is True and receipt.get('local_step')==schedule['total_steps'],'Complete full pilot budget required')
 paths=[run/'model'/('step-%012d'%s) for s in schedule['candidate_steps']]
 for p in paths:
  m=parent._verified_checkpoint_metadata(p,data_hash=dataset.data_hash,config=config,normalization=dataset.normalization)
  need(m['step']==int(p.name[5:]) and m['optimizer_step']==4000+m['step'] and m.get('schedule_cursor')==4*m['step'],'Candidate clock differs')
  with np.load(p/'arrays.npz',allow_pickle=False) as a:verify_optimizer_age(a,m['leaf_contract'],m['step'])
 with np.load(Path(config['bootstrap_checkpoint'])/'arrays.npz',allow_pickle=False) as a, np.load(paths[0]/'arrays.npz',allow_pickle=False) as b:
  parity=assert_array_identity(a,b)
 need(_read(run/'bootstrap_array_parity.json')==parity,'Step0 bootstrap parity receipt differs')
 return dict(dataset=dataset,config=config,schedule=schedule,schedule_binding=config['schedule_binding'],pilot_binding=config['pilot_binding'],
             pilot_source_sha256=config['pilot_source_sha256'],candidate_paths=paths,completed=True)

def run(args):
 output=Path(args.output).resolve();manifest=Path(args.manifest).resolve();bootstrap=Path(args.bootstrap).resolve()
 _separate(output,[Path(__file__).resolve().parents[2],manifest.parent,bootstrap.parents[1]])
 need(not any((output/n).exists() for n in LOCK_NAMES),'Locked run')
 need(not output.exists() or args.resume is not None,'Fresh output or explicit same-run resume required')
 dataset,old_meta,schedule=load_inputs(manifest,bootstrap)
 if args.verify_only:
  return dict(schema=VERSION,verified=True,old4000_metadata_sha256=BOOTSTRAP_METADATA_SHA256,train_episodes=8,valid_windows=schedule['valid_window_count'],
   additional_steps=schedule['total_steps'],additional_draws=schedule['total_draws'],schedule_content_sha256=schedule_api.digest(schedule),gpu_initialized=False)
 output.mkdir(parents=True,exist_ok=True)
 schedule_path=output/'training_schedule.json'
 if args.resume:
  saved=_read(output/'configuration.json');config=saved['config'];verify_configuration(dataset,config,schedule)
  resolve_same_run_resume(output,args.resume,data_hash=dataset.data_hash,config=config,normalization=dataset.normalization)
 else:
  sb=schedule_api.save_schedule(schedule_path,schedule);config=make_config(dataset,old_meta,manifest,bootstrap,output,sb,schedule)
  _freeze(output/'configuration.json',dict(config=config,data_hash=dataset.data_hash,normalization=dataset.normalization,manifest=str(manifest),manifest_sha256=parent._sha(manifest)))
 from .coverage_pilot_evaluation import build_diagnostic_schedule,save_checkpoint_report,verify_checkpoint_report
 diagnostic=build_diagnostic_schedule(dataset);_freeze(output/'diagnostic_schedule.json',dict(data_hash=dataset.data_hash,rows=diagnostic))
 import jax
 import jax.numpy as jnp
 import flax.nnx as nnx
 import optax
 from openpi.models import model as model_api
 runtime=implementation.initialize(61,5e-5);need(runtime['assets']==config['assets'],'Official public assets changed')
 if args.resume:
  restored,saved_rng=implementation.restore(runtime,args.resume,dataset.data_hash,config,dataset.normalization);step=restored['step']
 else:
  restored,saved_rng=implementation.restore(runtime,bootstrap,dataset.data_hash,old_meta['config'],dataset.normalization);step=0
 rng=jax.random.wrap_key_data(saved_rng);sample=implementation.sampler(runtime,steps=10);stop=[False]
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
 def objective(parameters,frozen,obs,actions,key):
  model=nnx.merge(runtime['graph'],parameters,frozen);model.train();return model.compute_loss(key,obs,actions,train=True).mean()
 gradient=jax.jit(jax.value_and_grad(objective,argnums=0))
 @jax.jit
 def update(parameters,state,grads):
  changes,state=runtime['optimizer'].update(grads,state,parameters);return optax.apply_updates(parameters,changes),state
 segments=output/'training_segments';segments.mkdir(exist_ok=True);segment=segments/(utc().replace(':','-')+'-'+uuid.uuid4().hex+'.json')
 receipt=dict(schema=VERSION,started_utc=utc(),pid=os.getpid(),training_run_path=str(output),local_step=step,optimizer_step=4000+step,
  training_complete=False,resume=None if not args.resume else str(Path(args.resume).resolve()),pilot_binding=config['pilot_binding'],pilot_source_sha256=config['pilot_source_sha256'],closed_loop_validated=False)
 if (output/'training_run.json').exists():
  with segment.with_suffix('.previous.json').open('xb') as f:f.write((output/'training_run.json').read_bytes())
 def event(kind):
  with (output/'training_events.jsonl').open('a') as f:f.write(json.dumps(dict(event=kind,utc=utc(),local_step=step,optimizer_step=4000+step,segment=str(segment)))+'\n');f.flush();os.fsync(f.fileno())
 def save_receipt():
  receipt.update(local_step=step,optimizer_step=4000+step,schedule_cursor=4*step);parent._atomic_json(segment,receipt);parent._atomic_json(output/'training_run.json',receipt)
 def save_checkpoint():
  need(schedule_api.source_hashes()==config['pilot_source_sha256'] and core_sources()==config['source_sha256'],'Training source changed')
  need(parent._sha(schedule_path)==config['schedule_binding']['sha256'],'Schedule changed')
  leaves=jax.tree.leaves((runtime['trainable'],runtime['opt_state']));arrays={'leaf_%05d'%i:np.asarray(v) for i,v in enumerate(jax.device_get(leaves))};arrays['jax_rng']=np.asarray(jax.random.key_data(rng))
  verify_optimizer_age(arrays,runtime['leaf_contract'],step)
  if step==0:
   old_arrays,_=parent.load_checkpoint(bootstrap,data_hash=dataset.data_hash,config=old_meta['config'],normalization=dataset.normalization)
   parity=assert_array_identity(old_arrays,arrays);_freeze(output/'bootstrap_array_parity.json',parity)
  metadata=dict(data_hash=dataset.data_hash,config=config,normalization=dataset.normalization,leaf_contract=runtime['leaf_contract'],
   optimizer_step=4000+step,schedule_cursor=4*step,sampler_rng_state=dict(cursor=4*step,schedule_sha256=config['schedule_binding']['sha256']),
   best_step=None,best_validation_loss=None,heldout_axes=['diagnostic_original_train_only'])
  folder='model' if step in schedule['candidate_steps'] else 'recovery'
  return parent.save_checkpoint(output/folder,step=step,arrays=arrays,metadata=metadata)
 def diagnose(checkpoint):
  dest=output/'diagnostics'/('local_%06d'%step);binding=checkpoint_binding(checkpoint)
  if dest.exists():verify_checkpoint_report(dest,dataset,binding);return
  predicted=[];normalized=[]
  for row in diagnostic:
   obs,_=make_batch(dataset,'train',[row['train_index']],runtime['tokenizer'])
   value=np.asarray(sample(runtime['trainable'],runtime['frozen'],model_api.Observation(**jax.device_put(obs)),jax.random.key(917)))
   need(value.shape==(1,10,7) and np.isfinite(value).all(),'Malformed diagnostic output')
   ep=dataset.episodes[row['episode_index']];state=ep['state'][row['start']]
   normalized.append(value[0]);predicted.append(decode_prediction(dataset,value[0],state,row['phase']))
  save_checkpoint_report(dest,dataset,np.asarray(predicted),np.asarray(normalized),binding)
 event('resume' if args.resume else 'restore_exact_old4000_new_schedule');save_receipt();last_saved=step
 try:
  if not args.resume:checkpoint=save_checkpoint();receipt['latest_checkpoint']=str(checkpoint);diagnose(checkpoint)
  elif step in schedule['candidate_steps']:diagnose(Path(args.resume))
  limit=schedule['total_steps'] if args.stop_at is None else args.stop_at
  need(type(limit) is int and step<=limit<=schedule['total_steps'],'Invalid bounded stop-at')
  while step<limit and not stop[0]:
   indices=schedule_api.batch_indices(schedule,step);obs,actions=make_batch(dataset,'train',indices,runtime['tokenizer']);rng,key=jax.random.split(rng)
   loss,grads=gradient(runtime['trainable'],runtime['frozen'],model_api.Observation(**jax.device_put(obs)),jax.device_put(actions),key)
   need(bool(jax.device_get(jnp.all(jnp.stack([jnp.isfinite(g).all() for g in jax.tree.leaves(grads)])))) and np.isfinite(float(loss)),'Nonfinite training update')
   runtime['trainable'],runtime['opt_state']=update(runtime['trainable'],runtime['opt_state'],grads);jax.block_until_ready((runtime['trainable'],runtime['opt_state']));step+=1
   metrics=dict(local_step=step,optimizer_step=4000+step,train_loss=float(loss),schedule_cursor=4*step,train_indices=indices.tolist(),segment=str(segment))
   with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
   if step in schedule['candidate_steps'] or step==limit or stop[0]:
    checkpoint=save_checkpoint();last_saved=step;receipt['latest_checkpoint']=str(checkpoint)
    if step in schedule['candidate_steps']:diagnose(checkpoint)
   if step%25==0:print(json.dumps(metrics),flush=True)
   save_receipt()
  if step>last_saved:receipt['latest_checkpoint']=str(save_checkpoint())
  receipt.update(training_complete=step==schedule['total_steps'],interrupted=stop[0],requested_segment_complete=step==limit)
 except BaseException as exc:
  receipt['error']=type(exc).__name__+': '+str(exc);raise
 finally:
  receipt['finished_utc']=utc();save_receipt();event('segment_finished')
 return receipt

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--bootstrap',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--resume',type=Path);p.add_argument('--stop-at',type=int);p.add_argument('--verify-only',action='store_true');args=p.parse_args()
 if args.verify_only:print(json.dumps(run(args),indent=2));return
 with (parent.ROOT/'locks/isaac_gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU workload is active')
  print(json.dumps(run(args),indent=2))
if __name__=='__main__':main()
