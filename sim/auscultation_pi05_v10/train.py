"""Declared4000-step corrective fine-tune of exact B7300 pi05 parameters.

Fresh Adam/key61, immutable fixed-B normalization and a qualified16 TRAIN
schedule. Actual step0 parity must finish before the first update. No simulator.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import uuid
import numpy as np
from sim.auscultation_pi05_v7.coverage_pilot_schedule import file_sha,digest,json_bytes
from sim.auscultation_pi05_v8 import train as ancestor
from . import training_data as D, schedule as S

VERSION='wam_pi05_v10_recovery_training_v1'
INITIALIZATION_SCHEMA='wam_pi05_v10_recovery_warm_initialization_v1'
ROOT=Path('/home/william/wam-vla-v2')
DESIGN=ROOT/'diagnostics/pi05_recovery_data_training_design_v1/machine_spec_v2.json'
DESIGN_SHA='d4ab529ce6982433c447531b78b9d1f28de2a5726e22d001d5e704bc211f1f04'
LOCK_NAMES=('V10_SELECTION_LOCKED.json','selection','FINAL_TEST_STARTED.json')
STEPS=tuple(range(0,4001,500))
need=ancestor.need
read=ancestor.read
freeze=ancestor.freeze
utc=ancestor.utc
parent_module=ancestor.parent_module


def clock(step):
    need(type(step) is int and 0<=step<=4000,'Invalid local optimizer clock')
    return dict(local_step=step,optimizer_step=step,schedule_cursor=4*step,cumulative_trainable_update_age=7300+step)


def source_hashes():
    sources=D.core_sources();root=Path(__file__).resolve().parents[2]
    for name in ('schedule.py','train.py','evaluation.py','parity.py','execution.py'):
        sources['sim/auscultation_pi05_v10/'+name]=file_sha(Path(__file__).with_name(name))
    # This inventory is from the original accepted geometry; labels never set limits.
    inventory=read(Path(__file__).with_name('posture_sources.json'))
    sources['sim/auscultation_pi05_v10/posture_sources.json']=file_sha(Path(__file__).with_name('posture_sources.json'))
    for name,h in inventory.items():need(file_sha(root/name)==h,'Original posture source differs');sources[name]=h
    return sources


def design_binding():
    need(DESIGN.is_file() and not DESIGN.is_symlink() and file_sha(DESIGN)==DESIGN_SHA,'Accepted recovery design differs')
    return dict(path=str(DESIGN),sha256=DESIGN_SHA)


def parent_binding():
    design_binding();b=read(DESIGN)['parent_checkpoint_binding'];p=Path(b['checkpoint_path'])
    need(p.is_absolute() and p==p.resolve() and not p.is_symlink(),'Canonical B7300 checkpoint required')
    for n,k in (('metadata.json','metadata_sha256'),('arrays.npz','arrays_sha256'),('COMPLETE.json','complete_sha256')):
        need((p/n).is_file() and not (p/n).is_symlink() and file_sha(p/n)==b[k],'B7300 '+n+' bytes changed')
    m=read(p/'metadata.json');need(m['step']==7300 and m['data_hash']==b['data_hash'] and digest(m['config'])==b['config_sha256'] and digest(m['normalization'])==b['normalization_sha256'],'B7300 metadata differs')
    parent_module()._verified_checkpoint_metadata(p,data_hash=b['data_hash'],config=m['config'],normalization=m['normalization'])
    init=p.parents[1]/'initialization.json'
    need(file_sha(init)==m['initialization_sha256'],'Original B frozen initialization differs')
    return b,m,read(init)


def load_inputs(manifest,norm):
    source_hashes();db=design_binding()
    dataset=D.load_dataset(Path(manifest),scopes=('train','validation'),normalization_path=Path(norm),expected_normalization_sha256=D.NORMALIZATION_SHA256)
    schedule=S.build_schedule(dataset,design_binding=db);S.verify_schedule(schedule,dataset,design_binding=db)
    _,_,assets=parent_module()._verify_public_assets(ROOT)
    b,_,_=parent_binding()
    return dataset,schedule,assets,b


def make_config(output,dataset,schedule_binding,assets,parent):
    return dict(training_schema=VERSION,training_run_path=str(output),source_sha256=source_hashes(),design_binding=design_binding(),
        dataset_binding=D.dataset_binding(dataset),schedule_binding=schedule_binding,parent_checkpoint_binding=parent,assets=assets,
        seed=61,learning_rate=5e-5,batch_size=4,total_steps=4000,candidate_steps=list(STEPS),action_dim=7,action_horizon=10,state_width=41,
        openpi_commit=parent_module().OPENPI_COMMIT,initialization='Exact B7300 trainable leaves, original official frozen backbone; fresh Adam0/key61',
        action_encoding_binding=dataset.normalization['action_encoding_binding'],normalization_fit_episode_ids=dataset.normalization['residual_actions']['fit_episode_ids'],
        observation_contract=dataset.normalization['encoding_binding']['observation_contract'],
        dynamics_contract_sha256=dataset.dynamics['contract_sha256'],dynamics_recipe_sha256=dataset.dynamics['recipe_sha256'],
        execution_contract_sha256=dataset.dynamics['execution_contract_sha256'],model_architecture='unmodified official pi05 seven-joint10-horizon LoRA/projection partition')


def verify_configuration(saved,dataset,schedule,assets,parent):
    c=saved['config'];run=Path(c['training_run_path']);need(str(run)==str(run.resolve()),'Original run path required')
    sb=c['schedule_binding'];need(sb['path']==str(run/'training_schedule.json') and file_sha(sb['path'])==sb['sha256'] and read(sb['path'])==schedule,'Frozen full schedule differs')
    need(c==make_config(run,dataset,sb,assets,parent),'Training configuration changed')
    need(saved['data_hash']==dataset.data_hash and saved['normalization']==dataset.normalization,'Saved data or normalization differs')
    return c


def verify_initialization(config):
    run=Path(config['training_run_path']);p=run/'initialization.json';v=read(p)
    need(v.get('schema')==INITIALIZATION_SCHEMA and v['config_sha256']==digest(config) and v['parent_checkpoint_binding']==config['parent_checkpoint_binding'],'Warm initialization lineage differs')
    ancestor._validate_numerical(v['numerical_initialization'])
    need(v['integrity_sha256']==digest({k:x for k,x in v.items() if k!='integrity_sha256'}),'Initialization integrity differs')
    b,metadata,original=parent_binding()
    need(v['numerical_initialization']['frozen']==original['numerical_initialization']['frozen'],'Original official frozen weights differ')
    with np.load(Path(b['checkpoint_path'])/'arrays.npz',allow_pickle=False) as z:
        pairs=((leaf['path'][3:],z['leaf_%05d'%i]) for i,leaf in enumerate(metadata['leaf_contract']) if leaf['path'].startswith('[0]'))
        need(ancestor.fingerprint_leaves(pairs)==v['numerical_initialization']['trainable'],'Restored trainable weights differ from exact B7300')
    return v,file_sha(p)


def checkpoint_binding(path):
    path=Path(path);need(path==path.resolve() and not path.is_symlink(),'Original canonical checkpoint required')
    m=read(path/'metadata.json');c=m['config'];run=Path(c['training_run_path']);saved=read(run/'configuration.json')
    need(path.parent.name in ('model','recovery') and path.parent.parent==run and re.fullmatch('step-[0-9]{12}',path.name),'Checkpoint path/clock differs')
    need(c['training_schema']==VERSION and c['source_sha256']==source_hashes() and saved['config']==c and saved['data_hash']==m['data_hash'] and saved['normalization']==m['normalization'],'Checkpoint configuration differs')
    m=parent_module()._verified_checkpoint_metadata(path,data_hash=saved['data_hash'],config=c,normalization=saved['normalization'])
    step=m['step'];expected=clock(step);need(step==int(path.name[5:]) and all(m.get(k)==v for k,v in expected.items()),'Checkpoint age/cursor differs')
    need(m.get('training_schema')==VERSION and m['sampler_rng_state']==dict(cursor=4*step,schedule_sha256=c['schedule_binding']['sha256']),'Checkpoint schedule differs')
    init,ih=verify_initialization(c);need(m['initialization_sha256']==ih,'Checkpoint initialization differs')
    with np.load(path/'arrays.npz',allow_pickle=False) as z:
        ancestor.verify_optimizer_age(z,m['leaf_contract'],step)
        if step==0:ancestor.verify_step_zero_arrays(z,m['leaf_contract'],init['numerical_initialization'])
    return dict(schema='wam_pi05_v10_recovery_checkpoint_binding_v1',checkpoint_path=str(path),**expected,
        metadata_sha256=file_sha(path/'metadata.json'),arrays_sha256=m['arrays_sha256'],complete_sha256=file_sha(path/'COMPLETE.json'),
        data_hash=m['data_hash'],normalization_sha256=digest(m['normalization']),config_sha256=digest(c),schedule_sha256=c['schedule_binding']['sha256'],source_sha256=c['source_sha256'])


def verify_terminal(run):
    run=Path(run);p=run/'training_run.json';raw=p.read_bytes();r=json.loads(raw);seg=Path(r['segment'])
    need(seg.parent==run/'training_segments' and not seg.is_symlink() and seg.read_bytes()==raw,'Current terminal segment must equal main receipt')
    cancel=seg.with_suffix('.cancelled.json');need(not cancel.exists() and not cancel.is_symlink(),'Current segment cancelled')
    need(r.get('schema')==VERSION and r.get('training_run_path')==str(run) and r.get('training_complete') is True and r.get('interrupted') is False and r.get('error') is None,'Actual complete4000 terminal receipt required')
    need(all(r.get(k)==v for k,v in clock(4000).items()),'Incomplete terminal budget/cursor')
    need(p.read_bytes()==raw and seg.read_bytes()==raw and not cancel.exists() and not cancel.is_symlink(),'Terminal evidence changed')
    return r


def verified_run_context(run):
    from . import evaluation as E,parity
    run=Path(run).resolve(strict=True)
    artifacts={str(run/n):file_sha(run/n) for n in ('configuration.json','training_run.json','training_schedule.json','diagnostic_schedule.json','initialization.json','step0_parity.json')}
    for step in STEPS:
        checkpoint=run/'model'/('step-%012d'%step);report=run/'diagnostics'/('local_%06d'%step)
        for p in [checkpoint/n for n in ('metadata.json','arrays.npz','COMPLETE.json')]+[report/n for n in ('report.json','train.npz','validation.npz')]:
            artifacts[str(p)]=file_sha(p)
    saved=read(run/'configuration.json');dataset,schedule,assets,parent=load_inputs(saved['manifest'],saved['normalization_path'])
    config=verify_configuration(saved,dataset,schedule,assets,parent);verify_initialization(config);terminal=verify_terminal(run)
    need(terminal['source_sha256']==config['source_sha256'],'Terminal source differs')
    parity_proof=parity.verify_receipt(run/'step0_parity.json',dataset,checkpoint_binding(run/'model/step-000000000000'))
    for p,h in parity_proof['artifact_sha256'].items():
        need(p not in artifacts or artifacts[p]==h,'Conflicting early parity evidence binding');artifacts[p]=h
    rows={s:E.build_report_schedule(dataset,s) for s in ('train','validation')}
    need(read(run/'diagnostic_schedule.json')==dict(data_hash=dataset.data_hash,rows=rows,probes=E.build_probe_schedule(dataset)),'Frozen report/probe rows differ')
    for step in STEPS:
        checkpoint=run/'model'/('step-%012d'%step)
        E.verify_checkpoint_report(run/'diagnostics'/('local_%06d'%step),dataset,checkpoint_binding(checkpoint))
    for p,h in artifacts.items():need(file_sha(p)==h,'Consumed run context changed')
    need(verify_terminal(run)==terminal,'Current segment changed or cancelled during context verification')
    artifacts[terminal['segment']]=artifacts[str(run/'training_run.json')]
    return dict(dataset=dataset,schedule=schedule,config=config,completed=True,terminal=terminal,artifact_sha256=artifacts)


def latest_resume(run,resume,saved):
    need(not any((run/n).exists() or (run/n).is_symlink() for n in LOCK_NAMES),'Selected/test-consumed run cannot resume')
    allpaths=[]
    for folder in ('model','recovery'):
        if (run/folder).exists():
            for p in (run/folder).iterdir():
                if re.fullmatch('step-[0-9]{12}',p.name):allpaths.append(p)
    need(allpaths,'No complete checkpoint');latest=max(allpaths,key=lambda p:int(p.name[5:]))
    need(Path(resume)==latest and sum(p.name==latest.name for p in allpaths)==1,'Only unique latest complete same-run checkpoint may resume')
    b=checkpoint_binding(latest);need(b['data_hash']==saved['data_hash'],'Resume data differs');return b


def finish_training(receipt,stop,handlers,save_receipt,write_cancellation):
    """A final write or handler restoration must not hide cancellation."""
    errors=[];receipt['finished_utc']=utc()
    if stop[0]:receipt.update(training_complete=False,requested_segment_complete=False,interrupted=True)
    try:save_receipt()
    except BaseException as exc:errors.append(exc)
    finally:
        for sig,handler in handlers.items():
            try:signal.signal(sig,handler)
            except BaseException as exc:errors.append(exc)
    if stop[0]:
        receipt.update(training_complete=False,requested_segment_complete=False,interrupted=True,
            cancelled_utc=utc(),error=receipt.get('error') or 'InterruptedError: training cancelled during execution or finalization')
        cancellation=dict(schema='wam_pi05_v10_recovery_training_cancellation_v1',training_complete=False,interrupted=True,
            local_step=receipt.get('local_step'),cancelled_utc=receipt['cancelled_utc'],error=receipt['error'])
        try:write_cancellation(cancellation)
        except BaseException as exc:errors.append(exc)
        try:save_receipt()
        except BaseException as exc:errors.append(exc)
        raise InterruptedError('Training cancelled; successful completion invalid') from (errors[0] if errors else None)
    if errors:raise errors[0]


def run(args):
    from . import evaluation as E,parity
    output=Path(args.output).resolve();manifest=Path(args.manifest).resolve();norm=Path(args.normalization).resolve()
    ancestor.separate(output,[Path(__file__).resolve().parents[2],manifest.parent,norm.parent,ROOT/'public-assets',DESIGN.parent])
    need(not any((output/n).exists() or (output/n).is_symlink() for n in LOCK_NAMES),'Run locked')
    need(not output.exists() or args.resume is not None,'Fresh output or explicit same-run resume required')
    data,schedule,assets,parent=load_inputs(manifest,norm)
    _,parent_metadata,_=parent_binding()
    protected=[Path(parent['checkpoint_path']).parents[1]]
    paired=parent_metadata['config'].get('paired_initialization')
    if paired is not None:protected.append(Path(paired['path']).parent)
    protected.extend(Path(p).parent for p in data._identity_artifacts)
    protected.extend(Path(p).parent for p in data._scoped_artifacts)
    ancestor.separate(output,protected)
    if args.verify_only:
        if args.resume:
            saved=read(output/'configuration.json');verify_configuration(saved,data,schedule,assets,parent)
            need(saved['manifest']==str(manifest) and saved['normalization_path']==str(norm),'Resume input paths differ')
            latest_resume(output,Path(args.resume).resolve(),saved)
        return dict(verified=True,gpu_initialized=False,dataset_binding=D.dataset_binding(data),schedule_sha256=digest(schedule),qualified_scopes=list(data.loaded_scopes))
    output.mkdir(parents=True,exist_ok=True);sp=output/'training_schedule.json'
    if args.resume:
        saved=read(output/'configuration.json');config=verify_configuration(saved,data,schedule,assets,parent)
        need(saved['manifest']==str(manifest) and saved['normalization_path']==str(norm),'Resume input paths differ')
        step=latest_resume(output,Path(args.resume).resolve(),saved)['local_step']
    else:
        freeze(sp,schedule);config=make_config(output,data,dict(path=str(sp),sha256=file_sha(sp)),assets,parent);step=0
        freeze(output/'configuration.json',dict(config=config,data_hash=data.data_hash,normalization=data.normalization,manifest=str(manifest),normalization_path=str(norm)))
    reports={s:E.build_report_schedule(data,s) for s in ('train','validation')};freeze(output/'diagnostic_schedule.json',dict(data_hash=data.data_hash,rows=reports,probes=E.build_probe_schedule(data)))
    limit=4000 if args.stop_at is None else args.stop_at;need(type(limit) is int and step<=limit<=4000,'Bounded segment limit invalid')
    stop=[False];old={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for sig in old:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    folder=output/'training_segments';folder.mkdir(exist_ok=True);segment=folder/(utc().replace(':','-')+'-'+uuid.uuid4().hex+'.json')
    receipt=dict(schema=VERSION,training_run_path=str(output),segment=str(segment),pid=os.getpid(),started_utc=utc(),source_sha256=config['source_sha256'],training_complete=False,interrupted=False,error=None,**clock(step))
    parent_api=parent_module()
    if (output/'training_run.json').exists():
        with segment.with_suffix('.previous.json').open('xb') as f:f.write((output/'training_run.json').read_bytes())
    def save_receipt():
        receipt.update(**clock(step),interrupted=stop[0]);parent_api._atomic_json(segment,receipt);parent_api._atomic_json(output/'training_run.json',receipt)
    save_receipt()
    try:
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        import optax
        from openpi.models import model as model_api
        from sim.auscultation_pi05_v7 import model as implementation
        runtime=implementation.initialize(61,5e-5);need(runtime['assets']==assets,'Official assets changed')
        _,pm,_=parent_binding();implementation.restore(runtime,parent['checkpoint_path'],parent['data_hash'],pm['config'],pm['normalization'])
        runtime['opt_state']=runtime['optimizer'].init(runtime['trainable']);rng=jax.random.key(61)
        numerical=ancestor._numeric_initialization(runtime,rng,jax);ancestor._validate_numerical(numerical)
        init=dict(schema=INITIALIZATION_SCHEMA,config_sha256=digest(config),parent_checkpoint_binding=parent,numerical_initialization=numerical)
        init['integrity_sha256']=digest(init)
        if args.resume:
            previous,ih=verify_initialization(config);need(previous==init,'Exact warm initialization no longer reproducible')
            _,key=implementation.restore(runtime,args.resume,data.data_hash,config,data.normalization);rng=jax.random.wrap_key_data(key)
        else:freeze(output/'initialization.json',init);_,ih=verify_initialization(config)
        sample=implementation.sampler(runtime,steps=10)
        def predict(obs):
            return np.asarray(sample(runtime['trainable'],runtime['frozen'],model_api.Observation(**jax.device_put(obs)),jax.random.key(917)))
        def checkpoint():
            need(source_hashes()==config['source_sha256'] and file_sha(sp)==config['schedule_binding']['sha256'],'Sources or schedule changed')
            leaves=jax.tree.leaves((runtime['trainable'],runtime['opt_state']));arrays={'leaf_%05d'%i:np.asarray(v) for i,v in enumerate(jax.device_get(leaves))};arrays['jax_rng']=np.asarray(jax.random.key_data(rng))
            ancestor.verify_optimizer_age(arrays,runtime['leaf_contract'],step)
            metadata=dict(training_schema=VERSION,data_hash=data.data_hash,config=config,normalization=data.normalization,leaf_contract=runtime['leaf_contract'],**clock(step),
                sampler_rng_state=dict(cursor=4*step,schedule_sha256=config['schedule_binding']['sha256']),initialization_sha256=ih,best_step=None,best_validation_loss=None,heldout_axes=['single_R5_episode_same_body'])
            return parent_api.save_checkpoint(output/('model' if step in STEPS else 'recovery'),step=step,arrays=arrays,metadata=metadata)
        def diagnose(cp):
            dest=output/'diagnostics'/('local_%06d'%step);binding=checkpoint_binding(cp)
            if dest.exists():E.verify_checkpoint_report(dest,data,binding);return
            values={}
            for split,rows in reports.items():
                predictions=[]
                for row in rows:
                    if stop[0]:raise InterruptedError('Cancelled during candidate report')
                    index=row['train_index'] if split=='train' else row['split_index']
                    obs,_=D.make_batch(data,split,[index],runtime['tokenizer']);z=predict(obs)
                    need(z.shape==(1,10,7) and z.dtype.kind=='f','Malformed actual sampler output')
                    predictions.append(z[0])
                values[split]=np.asarray(predictions)
            E.save_checkpoint_report(dest,data,values,binding)
        if not args.resume:
            cp=checkpoint();receipt['latest_checkpoint']=str(cp)
            parity.run_step0(output/'step0_parity.json',data,checkpoint_binding(cp),runtime['tokenizer'],predict,cancelled=lambda:stop[0])
        else:
            parity.verify_receipt(output/'step0_parity.json',data,checkpoint_binding(output/'model/step-000000000000'))
            cp=Path(args.resume)
        if step in STEPS and not stop[0]:diagnose(cp)
        def objective(parameters,frozen,obs,actions,key):
            model=nnx.merge(runtime['graph'],parameters,frozen);model.train();return model.compute_loss(key,obs,actions,train=True).mean()
        gradient=jax.jit(jax.value_and_grad(objective,argnums=0))
        @jax.jit
        def update(parameters,state,grads):
            changes,state=runtime['optimizer'].update(grads,state,parameters);return optax.apply_updates(parameters,changes),state
        last_saved=step
        while step<limit and not stop[0]:
            indices=S.batch_indices(schedule,step);obs,actions=D.make_batch(data,'train',indices,runtime['tokenizer']);rng,key=jax.random.split(rng)
            loss,grads=gradient(runtime['trainable'],runtime['frozen'],model_api.Observation(**jax.device_put(obs)),jax.device_put(actions),key)
            need(bool(jax.device_get(jnp.all(jnp.stack([jnp.isfinite(g).all() for g in jax.tree.leaves(grads)])))) and np.isfinite(float(loss)),'Nonfinite training update')
            runtime['trainable'],runtime['opt_state']=update(runtime['trainable'],runtime['opt_state'],grads);jax.block_until_ready((runtime['trainable'],runtime['opt_state']));step+=1
            m=dict(**clock(step),train_loss=float(loss),train_indices=np.asarray(indices).tolist(),segment=str(segment))
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(m,allow_nan=False)+'\n')
            if step in STEPS or step==limit or stop[0]:
                cp=checkpoint();last_saved=step;receipt['latest_checkpoint']=str(cp)
                if step in STEPS and not stop[0]:diagnose(cp)
            if step%25==0:print(json.dumps(m),flush=True)
            save_receipt()
        if step>last_saved:receipt['latest_checkpoint']=str(checkpoint())
        receipt.update(training_complete=step==4000 and not stop[0],requested_segment_complete=step==limit)
    except BaseException as exc:
        receipt['error']=type(exc).__name__+': '+str(exc);raise
    finally:
        finish_training(receipt,stop,old,save_receipt,lambda v:freeze(segment.with_suffix('.cancelled.json'),v))
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('manifest','normalization','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--stop-at',type=int);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:print(json.dumps(run(a),indent=2));return
    with (ROOT/'locks/isaac_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU workload is active')
        result=run(a);print(json.dumps(result),flush=True)
        if result['interrupted'] or result['error']:raise SystemExit(1)


if __name__=='__main__':main()
