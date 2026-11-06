"""Early ACT fit diagnosis on available qualified live TRAIN/VALIDATION episodes.

This separate entry implements the six-hour plan's early pipeline fit without
changing the formal150episode/20identity/95percent population gate. Every
source is freshly verified by m2_act.prepare_dataset(qualification=None).
No TEST, CPU-fixture datasets, prepared-array injection or deployment hook.

CLI: python -m sim.ausc_policy.m2_act_pilot --config CONFIG --output NEW_DIR.
Config mode='pilot_train', manifest + manifest_sha256, lerobot_root, explicit
cpu/cuda device, seed, steps, batch_size, eval_every; optional lr, weight_decay,
q_noise_std, q_dropout, cpu_threads(1..4), max_wall_seconds(1..21600), and base
(required for CUDA's unchanged shared isaac_gpu.lock). CPU requires empty
CUDA_VISIBLE_DEVICES and threadpoolctl; CPU takes no GPU lease. CUDA requires
owned Linux. There is no launch or scheduler in this module.

best_pilot.pt uses DISTINCT m2_act_pilot_diagnostic_v1, production_load_allowed
False, cpu_fixture_only False. The unchanged production checkpoint validator
rejects it even with allow_cpu_fixture=True. TRAIN and VALIDATION diagnostics
include all H20 horizons and HOLD/constant-velocity baselines. Only VALIDATION
whole-chunk MAE selects checkpoints. No closed-loop/physics acceptance claim.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from . import m2_act as act

SCHEMA='m2_act_pilot_diagnostic_history2_v2'
FLAGS=dict(pilot_only=True,production_load_allowed=False,formal_population_qualified=False,
           training_eligible=False,training_dataset_eligible=False,physical_acceptance=False,
           closed_loop_evaluated=False,real_hardware_validated=False,cpu_fixture_only=False)


def need(ok,message):
    if not ok:raise ValueError(message)


def configuration(config):
    need(isinstance(config,dict) and config.get('mode')=='pilot_train','Explicit diagnostic pilot_train mode required')
    allowed={'mode','manifest','manifest_sha256','lerobot_root','device','seed','steps','batch_size','eval_every',
             'lr','weight_decay','q_noise_std','q_dropout','cpu_threads','max_wall_seconds','base'}
    need(set(config)<=allowed,'Unsupported pilot configuration (qualification/deployment/fixtures forbidden)')
    result=deepcopy(config)
    for key,limit in [('steps',1_000_000),('batch_size',4096),('eval_every',1_000_000)]:
        need(type(result.get(key)) is int and 1<=result[key]<=limit,'Finite positive '+key+' required')
    need(type(result.get('seed')) is int and 0<=result['seed']<2**32,'Explicit fixed integer seed required')
    need(result.get('device') in ('cpu','cuda'),'Explicit CPU/CUDA device required')
    for key,default,low,high in [('lr',1e-4,0.,1.),('weight_decay',1e-4,0.,1.),
        ('q_noise_std',.002,0.,1.),('q_dropout',.05,0.,1.),('max_wall_seconds',3600.,1.,21600.)]:
        result.setdefault(key,default);value=result[key]
        need(type(value) in (int,float) and math.isfinite(value) and low<=value<=high and
             (key!='lr' or value>0),'Invalid finite '+key)
    result.setdefault('cpu_threads',4)
    need(type(result['cpu_threads']) is int and 1<=result['cpu_threads']<=4,'CPU thread cap must be1..4')
    for key in ('manifest','lerobot_root'):
        need(isinstance(result.get(key),str) and bool(result[key]),'Explicit '+key+' required')
        result[key]=str(Path(result[key]).resolve())
    need(isinstance(result.get('manifest_sha256'),str) and len(result['manifest_sha256'])==64,'Manifest SHA256 required')
    if result['device']=='cpu':
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU pilot requires CUDA_VISIBLE_DEVICES empty')
    else:
        need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1','CUDA pilot requires owned Linux execution')
        need(isinstance(result.get('base'),str) and bool(result['base']),'CUDA shared-lock base required')
        result['base']=str(Path(result['base']).resolve())
    return result


@contextmanager
def resources(config):
    """CPU numerical thread caps; CUDA uses exactly the existing shared lease."""
    if config['device']=='cpu':
        try:from threadpoolctl import threadpool_limits
        except ImportError as error:raise RuntimeError('CPU pilot requires threadpoolctl in the official ACT venv') from error
        with threadpool_limits(limits=config['cpu_threads']):
            import torch
            torch.set_num_threads(config['cpu_threads'])
            if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
            need(torch.get_num_threads()<=4 and torch.get_num_interop_threads()==1,'CPU thread cap not applied')
            yield
    else:
        import fcntl
        with (Path(config['base'])/'locks/isaac_gpu.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            yield


def evaluate(torch,policy,samples,stats,batch_size,device,deadline):
    """Deterministic entire split, with no augmentation or action inputs."""
    policy.eval();predictions=[]
    with torch.no_grad():
        for start in range(0,len(samples['states']),batch_size):
            need(time.monotonic()<deadline,'Pilot wall deadline exceeded during evaluation')
            ids=np.arange(start,min(start+batch_size,len(samples['states'])))
            batch=act._tensors(torch,act.pack_batch(samples,stats,ids,training=False),device)
            observations={key:value for key,value in batch.items() if key.startswith('observation.')}
            result=policy.predict_action_chunk(observations).detach().cpu().numpy()
            predictions.append(result*np.asarray(stats['action_std'])+np.asarray(stats['action_mean']))
    return act.evaluate_predictions(np.concatenate(predictions),samples)


def source_bindings(prepared):
    """Preserve original expected hashes, never adopt replacements at finalization."""
    path=Path(prepared['manifest_path']);data=path.read_bytes()
    need(hashlib.sha256(data).hexdigest()==prepared['manifest_sha256'],'Dataset manifest changed after preparation')
    manifest=json.loads(data)
    bindings={str(path):prepared['manifest_sha256'],str(act.contract.IDENTITY_SPLIT_PATH):act.contract.IDENTITY_SPLIT_SHA256}
    def bind(path,digest):
        name=str(Path(path).resolve())
        need(name not in bindings or bindings[name]==digest,'Conflicting source digest')
        bindings[name]=digest
    for row in manifest['episodes']:
        act.require_membership(row['split'],row['subject_id'])
        root=(path.parent/row['capture_root']).resolve()
        for name,expected in row['artifacts'].items():bind(root/name,expected['sha256'])
        bind(path.parent/row['run_result']['path'],row['run_result']['sha256'])
    from . import training_capture, m2_stage
    from sim.auscultation_expert_layout_v1 import kinematics
    for module in (sys.modules[__name__],act,act.contract,training_capture,m2_stage,kinematics):
        bind(module.__file__,act.sha256(module.__file__))
    return bindings,data


def verify_bindings(bindings):
    for path,digest in bindings.items():need(act.sha256(path)==digest,'Pilot source changed: '+path)


def fit_pilot(config,output):
    config=configuration(config);started=time.monotonic();deadline=started+config['max_wall_seconds']
    output=Path(output).resolve()
    need(not output.exists(),'Pilot output must be new')
    with resources(config):
        # This is the only data admission route; no prepared array or truthy flag
        # can replace native capture/physics, label, split and inventory checks.
        prepared=act.prepare_dataset(config['manifest'],config['manifest_sha256'],qualification=None)
        need(prepared.get('training_eligible') is False and prepared.get('qualification_reference') is None,
             'Pilot preparation must remain formally ineligible')
        bindings,manifest_bytes=source_bindings(prepared)
        verify_bindings(bindings)
        need(time.monotonic()<deadline,'Pilot deadline exceeded during preparation')
        torch,policy,official=act.make_native(config)
        official_bindings={str(Path(config['lerobot_root'])/name):digest for name,digest in official.items()}
        verify_bindings(official_bindings)
        runtime=dict(python_version=platform.python_version(),python_executable=sys.executable,
            python_prefix=sys.prefix,platform=sys.platform,torch_version=str(torch.__version__),
            numpy_version=np.__version__,device=config['device'],seed=config['seed'],augmentation_seed=config['seed']+1,
            cpu_threads=config['cpu_threads'] if config['device']=='cpu' else None,
            torch_intraop_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
            gpu_lease_acquired=config['device']=='cuda',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            upstream_commit=act.UPSTREAM_COMMIT,official_source_hashes=official,source_hashes=bindings,
            formal_cut_thresholds_unchanged=dict(minimum_episodes=150,minimum_identities=20,minimum_success_rate=.95))
        stats=prepared['normalization'];train=prepared['train'];validation=prepared['validation']
        output.mkdir(parents=True,exist_ok=False)
        act._save(output/'configuration.json',config);act._save(output/'normalization.json',stats)
        act._save(output/'runtime.json',runtime);(output/'source_manifest.json').write_bytes(manifest_bytes)
        act._save(output/'preparation.json',{key:value for key,value in prepared.items() if key not in ('train','validation')})
        optimizer=torch.optim.AdamW(policy.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        rng=np.random.default_rng(config['seed']);augmentation=np.random.default_rng(config['seed']+1)
        best=float('inf');best_step=None;selected_metrics=None
        try:
            with (output/'metrics.jsonl').open('x') as log:
                for step in range(1,config['steps']+1):
                    need(time.monotonic()<deadline,'Pilot finite wall deadline exceeded')
                    policy.train();ids=rng.integers(len(train['states']),size=config['batch_size'])
                    batch=act._tensors(torch,act.pack_batch(train,stats,ids,training=True,rng=augmentation,
                        q_noise_std=config['q_noise_std'],q_dropout=config['q_dropout']),config['device'])
                    optimizer.zero_grad(set_to_none=True);loss,_=policy(batch)
                    need(bool(torch.isfinite(loss)),'Nonfinite pilot loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    log.write(json.dumps(dict(step=step,train_loss=float(loss.detach().cpu())))+'\n')
                    if step%config['eval_every'] and step!=config['steps']:continue
                    metrics={split:evaluate(torch,policy,samples,stats,config['batch_size'],config['device'],deadline)
                             for split,samples in [('train',train),('validation',validation)]}
                    score=metrics['validation']['act']['full_chunk_mae_rad']
                    log.write(json.dumps(dict(step=step,metrics=metrics,selection_split='validation'),allow_nan=False)+'\n');log.flush()
                    if score<best:
                        best=score;best_step=step;selected_metrics=metrics
                        checkpoint=dict(schema=SCHEMA,state_dict=policy.state_dict(),step=step,model=act.native_spec(),
                            normalization=stats,config=config,metadata=prepared['metadata'],runtime=runtime,
                            observation_contract=act.contract.observation_contract(),
                            upstream_commit=act.UPSTREAM_COMMIT,manifest_sha256=prepared['manifest_sha256'],
                            source_hashes=official,selection_split='validation',metrics=metrics,
                            state_only=True,action_representation='block_relative_joint',**FLAGS)
                        # Never call production loader or give this file a production schema.
                        torch.save(checkpoint,output/'best_pilot.pt')
            verify_bindings(bindings);verify_bindings(official_bindings)
            need(time.monotonic()<deadline,'Pilot deadline exceeded during final source audit')
            artifacts={name:dict(bytes=(output/name).stat().st_size,sha256=act.sha256(output/name)) for name in
                ('best_pilot.pt','configuration.json','normalization.json','runtime.json','source_manifest.json','preparation.json','metrics.jsonl')}
            result=dict(schema=SCHEMA,completed=True,steps=config['steps'],best_step=best_step,
                selection_split='validation',selection_metric='validation.act.full_chunk_mae_rad',metrics=selected_metrics,
                train_samples=len(train['states']),validation_samples=len(validation['states']),
                manifest_sha256=prepared['manifest_sha256'],artifacts=artifacts,wall_seconds=time.monotonic()-started,
                split_access_scope=prepared['split_access_scope'],test_payloads_opened=False,
                scope='Early offline fit diagnosis; does not qualify population or policy for production/Isaac deployment',**FLAGS)
            act._save(output/'COMPLETE.json',result)
            return result
        except BaseException as error:
            act._save(output/'FAILED.json',dict(schema=SCHEMA,completed=False,error=repr(error),
                wall_seconds=time.monotonic()-started,**FLAGS))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();fit_pilot(json.loads(args.config.read_text()),args.output)


if __name__=='__main__':main()
