"""Explicit VAL-only diagnostic ACT rollout admission; never production qualification.

Prepare ONCE in the pinned ACT venv with CUDA_VISIBLE_DEVICES='': hash-bound
pilot COMPLETE + frozen attempt plan -> new diagnostic receipt. Every attempt
contains episode_id/subject_id/split='validation'/target_id/runtime_config_sha256.
The config digest excludes only m2_act_deployment, avoiding a receipt hash cycle.
All other config fields (patient, noisy estimate, plans, physics) are frozen.

Use the existing m2_act_deployment key with schema m2_act_pilot_history2_deployment_v2
and its six ordinary fields plus diagnostic_receipt={file,sha256}. Production
load_checkpoint remains unchanged and rejects best_pilot.pt. Isaac delegates
inference to a non-detached CPU child in the official ACT venv; no GPU lease,
no queued/replayed actions, no expert L2 fallback, no dataset reevaluation in
workers. The existing M2ACTController retains all native240Hz safety gates.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time
import numpy as np
from . import m2_act as act,m2_act_pilot as pilot,m2_contract as contract

SCHEMA='m2_act_pilot_history2_closed_loop_admission_v2'
DEPLOYMENT_SCHEMA='m2_act_pilot_history2_deployment_v2'
PLAN_SCHEMA='m2_act_pilot_validation_attempt_plan_v1'
FLAGS=dict(diagnostic_only=True,pilot_only=True,production_load_allowed=False,
    formal_population_qualified=False,training_eligible=False,training_dataset_eligible=False,
    expert_demonstration=False,learned_policy=True)


def need(ok,message):
    if not ok:raise ValueError(message)


def read_bound(ref):
    need(isinstance(ref,dict) and set(ref)=={'file','sha256'} and
         isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']), 'Explicit file/SHA256 reference required')
    path=Path(ref['file']).resolve();data=path.read_bytes()
    need(hashlib.sha256(data).hexdigest()==ref['sha256'],'Bound source hash changed: '+str(path))
    return json.loads(data),dict(file=str(path),sha256=ref['sha256'])


def runtime_config_sha256(config):
    data={key:value for key,value in config.items() if key!='m2_act_deployment'}
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def metric_gate(metrics):
    need(isinstance(metrics,dict) and set(metrics)=={'train','validation'},'Both TRAIN and VALIDATION metrics required')
    checks=[]
    for split in ('train','validation'):
        row=metrics[split]
        need(type(row.get('sample_count')) is int and row['sample_count']>0 and row.get('horizon')==20,'Complete nonempty H20 metrics required')
        for method in ('act','hold','constant_velocity'):
            value=row[method];full=value['full_chunk_mae_rad'];h=np.asarray(value['per_horizon_mae_rad'])
            need(type(full) in (int,float) and np.isfinite(full) and full>=0 and h.shape==(20,) and
                 np.isfinite(h).all() and (h>=0).all(),'Finite nonnegative all-horizon MAE required')
        for baseline in ('hold','constant_velocity'):
            a=float(row['act']['full_chunk_mae_rad']);b=float(row[baseline]['full_chunk_mae_rad'])
            checks.append(dict(split=split,baseline=baseline,act_mae_rad=a,baseline_mae_rad=b,
                threshold_mae_rad=.8*b,passed=a<=.8*b,required_for_admission=split=='train',ratio=a/b if b>0 else None))
    return dict(passed=all(row['passed'] for row in checks if row['required_for_admission']),maximum_ratio=.8,checks=checks,
        scope='Entire deterministic TRAIN H20 must beat both baselines; entire finite VALIDATION H20 comparisons reported without an extra gate')


def _plan(doc):
    need(doc.get('schema')==PLAN_SCHEMA and doc.get('frozen') is True and bool(doc.get('attempts')),'Frozen diagnostic attempt plan required')
    seen=set()
    for row in doc['attempts']:
        need(set(row)=={'episode_id','subject_id','split','target_id','runtime_config_sha256'},'Exact attempt identity/config binding required')
        need(row['split']=='validation','Diagnostic rollouts are VALIDATION only')
        act.require_membership(row['split'],row['subject_id'])
        need(isinstance(row['episode_id'],str) and 0<len(row['episode_id'])<=128 and row['episode_id'] not in seen,'Duplicate/missing rollout episode')
        seen.add(row['episode_id'])
        need(row['target_id'] in {side+str(n) for side in ('L','R') for n in range(1,6)} and
             re.fullmatch('[0-9a-f]{64}',row['runtime_config_sha256']) is not None,'Invalid target/config digest')
    return doc['attempts']


def _checkpoint(data):
    need(data.get('schema')==pilot.SCHEMA and data.get('upstream_commit')==act.UPSTREAM_COMMIT and
         data.get('model')==act.native_spec() and data.get('selection_split')=='validation' and
         data.get('state_only') is True and data.get('action_representation')=='block_relative_joint', 'Invalid diagnostic ACT checkpoint')
    for key,value in pilot.FLAGS.items():need(data.get(key) is value,'Pilot checkpoint declaration differs: '+key)
    contract.require_observation_contract(data['metadata'])
    need(data.get('observation_contract')==contract.observation_contract(),'Pilot observation contract mismatch')
    need(data['metadata'].get('source_kind')=='isaac_live_physics' and data['metadata'].get('replay') is False,
         'Diagnostic checkpoint needs actual live non-replay sources')
    need(data['normalization'].get('fitted_split')=='train' and
         data['normalization'].get('observation_contract')==contract.observation_contract(),
         'TRAIN-only v2 history normalization required')


def _load_weights(reference,lerobot_root):
    # Hash and deserialize the SAME bytes, not a reopened mutable checkpoint.
    import torch
    path=Path(reference['file']).resolve();data=path.read_bytes()
    need(hashlib.sha256(data).hexdigest()==reference['sha256'],'Checkpoint hash changed')
    checkpoint=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True);_checkpoint(checkpoint)
    need(Path(checkpoint['config']['lerobot_root']).resolve()==Path(lerobot_root).resolve(),'Original official ACT environment changed')
    torch,policy,sources=act.make_native(dict(checkpoint['config'],device='cpu',lerobot_root=str(lerobot_root)))
    need(sources==checkpoint['source_hashes'],'Official ACT source hashes changed')
    policy.load_state_dict(checkpoint['state_dict'],strict=True);policy.eval()
    return torch,policy,checkpoint


def source_metadata(reference):
    """Authenticate TRAIN/VAL metadata before any archived payload hashing."""
    manifest,ref=read_bound(reference);base=Path(ref['file']).parent
    need(manifest.get('schema')=='m2_act_dataset_v1' and bool(manifest.get('episodes')),'Explicit dataset manifest required')
    for row in manifest['episodes']:act.require_membership(row.get('split'),row.get('subject_id'))
    metadata=[]
    for row in manifest['episodes']:
        path=base/row['capture_root']/'compact/manifest.json'
        doc,_=read_bound(dict(file=str(path),sha256=row['artifacts']['compact/manifest.json']['sha256']))
        meta=doc['metadata'];act.require_membership(meta.get('split'),meta.get('subject_id'))
        need(all(meta.get(key)==row[key] for key in ('split','subject_id','episode_id')),'Source split/identity differs')
        need(meta.get('source_kind')=='isaac_live_physics' and meta.get('replay') is False,'Actual live source required')
        metadata.append(meta)
    return metadata


def prepare_diagnostic(pilot_complete,attempt_plan,*,lerobot_root,output,batch_size=64,cpu_threads=2):
    """One CPU re-admission/evaluation; output may truthfully have allowed=False."""
    need(not Path(output).exists(),'Diagnostic receipt must be new')
    need(type(batch_size) is int and 1<=batch_size<=4096 and type(cpu_threads) is int and 1<=cpu_threads<=4,'Bounded CPU settings required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Diagnostic preparation is CPU-only')
    plan,plan_ref=read_bound(attempt_plan);attempts=_plan(plan)  # identities before any dataset payload
    complete,complete_ref=read_bound(pilot_complete)
    need(complete.get('schema')==pilot.SCHEMA and complete.get('completed') is True,'Successful original pilot COMPLETE required')
    for key,value in pilot.FLAGS.items():need(complete.get(key) is value,'Original pilot declaration differs: '+key)
    root=Path(complete_ref['file']).parent
    names={'best_pilot.pt','configuration.json','normalization.json','runtime.json','source_manifest.json','preparation.json','metrics.jsonl'}
    need(set(complete.get('artifacts',{}))==names,'Exact original pilot artifact inventory required')
    bindings={complete_ref['file']:complete_ref['sha256'],plan_ref['file']:plan_ref['sha256']}
    for name in sorted(names):
        entry=complete['artifacts'][name];path=(root/name).resolve()
        need(path.stat().st_size==entry['bytes'] and act.sha256(path)==entry['sha256'],'Pilot artifact hash/size changed: '+name)
        bindings[str(path)]=entry['sha256']
    runtime,_=read_bound(dict(file=str(root/'runtime.json'),sha256=bindings[str((root/'runtime.json').resolve())]))
    configuration,_=read_bound(dict(file=str(root/'configuration.json'),sha256=bindings[str((root/'configuration.json').resolve())]))
    need(runtime.get('source_hashes') and configuration['manifest_sha256']==complete['manifest_sha256'],'Original source bindings required')
    admitted_metadata=source_metadata(dict(file=configuration['manifest'],sha256=complete['manifest_sha256']))
    for path,digest in runtime['source_hashes'].items():
        need(path not in bindings or bindings[path]==digest,'Conflicting original source digest')
        bindings[path]=digest
    pilot.verify_bindings(bindings)
    # prepare_dataset enforces native physical success, entire inventories,
    # TRAIN/VAL metadata preflight, quarantines, label/phase boundaries and H20.
    prepared=act.prepare_dataset(configuration['manifest'],complete['manifest_sha256'],qualification=None)
    current,_=pilot.source_bindings(prepared)
    need(current==runtime['source_hashes'],'Original training source bindings differ')
    eligible={(meta['subject_id'],meta['target_id']) for meta in admitted_metadata if meta['split']=='validation'}
    need(all((row['subject_id'],row['target_id']) in eligible for row in attempts),
         'Every diagnostic target must have a physically successful VALIDATION source')
    checkpoint_ref=dict(file=str((root/'best_pilot.pt').resolve()),sha256=complete['artifacts']['best_pilot.pt']['sha256'])
    with pilot.resources(dict(device='cpu',cpu_threads=cpu_threads)):
        torch,policy,checkpoint=_load_weights(checkpoint_ref,lerobot_root)
        need(checkpoint['manifest_sha256']==complete['manifest_sha256'] and checkpoint['runtime']==runtime and
             checkpoint['config']==configuration and checkpoint['normalization']==prepared['normalization'] and
             checkpoint['step']==complete['best_step'] and checkpoint['metrics']==complete['metrics'], 'Pilot checkpoint/COMPLETE/source metadata disagree')
        deadline=time.monotonic()+3600.
        metrics={split:pilot.evaluate(torch,policy,prepared[split],prepared['normalization'],batch_size,'cpu',deadline)
                 for split in ('train','validation')}
    gate=metric_gate(metrics)
    for name,digest in checkpoint['source_hashes'].items():bindings[str((Path(lerobot_root)/name).resolve())]=digest
    from . import m2_act_controller,fresh_expert,m2_estimate_provider,geometry_qualification
    for module in (sys.modules[__name__],m2_act_controller,fresh_expert,m2_estimate_provider,geometry_qualification):
        bindings[str(Path(module.__file__).resolve())]=act.sha256(module.__file__)
    pilot.verify_bindings(bindings)
    result=dict(schema=SCHEMA,completed=True,diagnostic_allowed=gate['passed'],gate=gate,metrics=metrics,
        pilot_complete=complete_ref,attempt_plan=plan_ref,attempts=attempts,checkpoint=checkpoint_ref,
        lerobot_root=str(Path(lerobot_root).resolve()),source_hashes=bindings,cpu_threads=cpu_threads,
        manifest_sha256=complete['manifest_sha256'],upstream_commit=act.UPSTREAM_COMMIT,
        checkpoint_selection='original validation-selected best_pilot.pt; no reselection',
        validation_reuse='Validation selects checkpoint and is reused for diagnostic rollout; not held-out evidence',
        physical_acceptance=False,closed_loop_evaluated=False,**FLAGS)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with Path(output).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


def verify_receipt(ref,*,check_sources=True):
    receipt,canonical=read_bound(ref)
    need(receipt.get('schema')==SCHEMA and receipt.get('completed') is True and receipt.get('diagnostic_allowed') is True,
         'No diagnostic admission permit')
    for key,value in FLAGS.items():need(receipt.get(key) is value,'Diagnostic declaration differs: '+key)
    need(receipt.get('upstream_commit')==act.UPSTREAM_COMMIT and receipt.get('gate')==metric_gate(receipt['metrics']) and
         receipt['gate']['passed'],'Offline diagnostic gate failed')
    plan,_=read_bound(receipt['attempt_plan']);need(_plan(plan)==receipt['attempts'],'Frozen rollout plan changed')
    need(receipt.get('source_hashes') and 1<=receipt['cpu_threads']<=4,'Missing diagnostic source/runtime bindings')
    if check_sources:pilot.verify_bindings(receipt['source_hashes'])
    return receipt


def preflight(config):
    """Run before scene/patient payload reads; hash sources, never decode samples."""
    dep=config.get('m2_act_deployment',{});metadata=dep.get('metadata',{})
    need(metadata.get('split')=='validation','Diagnostic rollouts are VALIDATION only')
    act.require_membership('validation',metadata.get('subject_id'))
    need(isinstance(config.get('patient_asset'),dict) and config['patient_asset'].get('expected_split')=='validation',
         'Diagnostic patient specification must explicitly select VALIDATION')
    fields={'schema','checkpoint','checkpoint_sha256','lerobot_root','metadata','episode_id','diagnostic_receipt'}
    need(set(dep)==fields and dep['schema']==DEPLOYMENT_SCHEMA,'Explicit diagnostic deployment fields required')
    contract.require_observation_contract(metadata)
    need(not config.get('fresh_expert',{}).get('raw_multimodal_capture'),'Learned diagnostic cannot use expert capture')
    receipt=verify_receipt(dep['diagnostic_receipt'])
    need(Path(dep['checkpoint']).resolve()==Path(receipt['checkpoint']['file']) and
         dep['checkpoint_sha256']==receipt['checkpoint']['sha256'] and
         Path(dep['lerobot_root']).resolve()==Path(receipt['lerobot_root']), 'Diagnostic checkpoint/environment changed')
    selected=[row for row in receipt['attempts'] if row['episode_id']==dep['episode_id']]
    need(len(selected)==1,'Episode is outside frozen diagnostic attempt plan')
    row=selected[0]
    need(all(metadata.get(key)==row[key] for key in ('subject_id','split','target_id')),'Diagnostic identity differs from frozen attempt')
    need(row['runtime_config_sha256']==runtime_config_sha256(config),'Frozen runtime configuration changed')
    return receipt


def decode_response(response,*,request_id,measured_q,metadata,inference_context=None):
    need(response.get('request_id')==request_id and 'error' not in response,'Diagnostic inference response failed/out of order')
    if inference_context is not None:
        need(response.get('inference_context')==inference_context,'Inference provider context/clock changed')
    need(response.get('metadata')==dict(metadata,checkpoint_policy=pilot.SCHEMA),'Inference metadata/context changed')
    origin=np.asarray(response.get('origin'),float)
    need(origin.shape==(7,) and np.array_equal(origin,measured_q),'Inference changed fixed block origin')
    return contract.JointActionBlock('block_relative_joint',np.asarray(response.get('relative'),float),origin,response['metadata'])


class CPUBridge:
    """Owned-session child, bounded messages/deadlines; EOF/exception fail closed."""
    def __init__(self,deployment,*,startup_timeout=120.,request_timeout=10.):
        self.dep=deepcopy(deployment);self.process=None;self.sequence=0;self.buffer=b'';self.request_timeout=request_timeout;self.context=None
        python=Path(deployment['lerobot_root'])/'.venv/bin/python'
        need(python.is_file(),'Pinned official ACT Python is missing')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                 PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])
        ref=deployment['diagnostic_receipt']
        command=[str(python),'-m',__name__,'--serve','--receipt',ref['file'],'--receipt-sha256',ref['sha256'],
                 '--episode-id',deployment['episode_id']]
        try:
            self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,env=env,
                start_new_session=False,close_fds=True,bufsize=0)
            ready=self._read(startup_timeout)
            need(ready.get('ready') is True and ready.get('checkpoint_sha256')==deployment['checkpoint_sha256'],
                 'Diagnostic CPU child failed initialization')
            self.runtime=ready
        except BaseException:self.close();raise

    def _read(self,timeout):
        deadline=time.monotonic()+timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout,selectors.EVENT_READ)
            while b'\n' not in self.buffer:
                remaining=deadline-time.monotonic()
                need(remaining>0 and bool(selector.select(max(0.,remaining))),'Diagnostic inference timed out')
                chunk=os.read(self.process.stdout.fileno(),65536)
                need(bool(chunk),'Diagnostic inference child exited')
                self.buffer+=chunk;need(len(self.buffer)<=131072,'Oversized diagnostic inference response')
        line,self.buffer=self.buffer.split(b'\n',1)
        return json.loads(line)

    def __call__(self,measured_q,estimated_target_base,confidence,*,metadata,history,history_times):
        self.sequence+=1
        need(isinstance(self.context,dict),'Explicit same-time provider context required')
        request=dict(request_id=self.sequence,measured_q=np.asarray(measured_q).tolist(),
            estimated_target_base=np.asarray(estimated_target_base).tolist(),confidence=float(confidence),metadata=metadata,
            inference_context=deepcopy(self.context),history=np.asarray(history).tolist(),
            history_times=np.asarray(history_times).tolist(),observation_contract=contract.observation_contract())
        try:
            payload=json.dumps(request,allow_nan=False,separators=(',',':')).encode()+b'\n'
            need(len(payload)<=32768,'Oversized diagnostic inference request')
            # One small request is outstanding; nonblocking writes share the
            # request deadline, so a wedged child cannot block pipe writes.
            descriptor=self.process.stdin.fileno();os.set_blocking(descriptor,False)
            deadline=time.monotonic()+self.request_timeout;offset=0
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor,selectors.EVENT_WRITE)
                while offset<len(payload):
                    remaining=deadline-time.monotonic()
                    need(remaining>0 and bool(selector.select(max(0.,remaining))),'Diagnostic request write timed out')
                    try:offset+=os.write(descriptor,payload[offset:])
                    except BlockingIOError:continue
            return decode_response(self._read(max(0.,deadline-time.monotonic())),request_id=self.sequence,
                measured_q=measured_q,metadata=metadata,inference_context=request['inference_context'])
        except BaseException:self.close();raise

    def close(self):
        child=self.process
        if child is None:return
        self.process=None
        if child.stdin:
            try:child.stdin.close()
            except OSError:pass
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=2.)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2.)
        if child.stdout:child.stdout.close()


def make_controller(*args,deployment,estimate_provider,**kwargs):
    """Caller has run config preflight before creating the scene/controller."""
    from .m2_act_controller import M2ACTController
    receipt=verify_receipt(deployment['diagnostic_receipt'])
    meta=deployment['metadata'];act.require_membership(meta.get('split'),meta.get('subject_id'))
    need(meta.get('split')=='validation','Diagnostic rollout is VALIDATION only')
    selected=[row for row in receipt['attempts'] if row['episode_id']==deployment['episode_id']]
    need(len(selected)==1 and all(meta.get(key)==selected[0][key] for key in ('split','subject_id','target_id')),'Frozen diagnostic identity changed')
    predictor=CPUBridge(deployment)
    policy_receipt=dict(schema=DEPLOYMENT_SCHEMA,diagnostic_receipt=deployment['diagnostic_receipt'],
        checkpoint_sha256=deployment['checkpoint_sha256'],metadata=deepcopy(meta),child_runtime=predictor.runtime,
        state_only=True,observation_contract=contract.observation_contract(),
        horizon=20,execution_prefix=10,control_hz=20,physics_hz=240,device='cpu',**FLAGS)
    class DiagnosticController(M2ACTController):
        def _query(self,observation):
            self.predictor.context=dict(observed_at_s=float(observation['observed_at_s']),
                context=list(observation['context']),source_id=observation['source_id'])
            try:return super()._query(observation)
            finally:self.predictor.context=None
        def run(self,points):
            try:return super().run(points)
            finally:self.predictor.close()
    try:
        return DiagnosticController(*args,predictor=predictor,metadata=meta,episode_id=deployment['episode_id'],
            estimate_provider=estimate_provider,policy_receipt=policy_receipt,**kwargs)
    except BaseException:predictor.close();raise


def serve(receipt_ref,episode_id):
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU bridge requires no visible GPU')
    receipt=verify_receipt(receipt_ref,check_sources=False)
    attempts=[row for row in receipt['attempts'] if row['episode_id']==episode_id]
    need(len(attempts)==1,'Child episode is outside frozen plan');attempt=attempts[0]
    with pilot.resources(dict(device='cpu',cpu_threads=receipt['cpu_threads'])):
        with redirect_stdout(sys.stderr):torch,policy,checkpoint=_load_weights(receipt['checkpoint'],receipt['lerobot_root'])
        predictor=act.ChunkPredictor(torch,policy,checkpoint['normalization'],checkpoint['metadata'],'cpu')
        print(json.dumps(dict(ready=True,pid=os.getpid(),python=sys.executable,python_prefix=sys.prefix,
            torch_version=str(torch.__version__),cpu_threads=torch.get_num_threads(),cuda_visible_devices='',
            checkpoint_sha256=receipt['checkpoint']['sha256'],observation_contract=contract.observation_contract())),flush=True)
        sequence=0;last_stamp=None;last_context=None
        while True:
            line=sys.stdin.buffer.readline(32769)
            if not line:return
            need(len(line)<=32768 and line.endswith(b'\n'),'Invalid diagnostic request bound')
            request=json.loads(line);sequence+=1
            need(request.get('request_id')==sequence,'Nonconsecutive diagnostic request')
            metadata=request['metadata'];contract.require_observation_contract(metadata)
            need(metadata.get('episode_id')==episode_id and all(metadata.get(key)==attempt[key] for key in
                ('subject_id','split','target_id')),'Inference context differs from frozen diagnostic attempt')
            envelope=request['inference_context'];context=envelope['context'];stamp=envelope['observed_at_s']
            need(set(envelope)=={'observed_at_s','context','source_id'} and len(context)==4 and
                 context[:2]==[episode_id,attempt['target_id']] and type(context[2]) is int and context[2]>=0 and
                 context[3] in contract.L2_PHASES and type(stamp) in (int,float) and np.isfinite(stamp) and stamp>=0 and
                 isinstance(envelope['source_id'],str) and bool(envelope['source_id']), 'Invalid same-time provider context')
            need(last_stamp is None or stamp>last_stamp,'Nonadvancing diagnostic inference clock')
            need(last_context is None or not(last_context[3]=='APPROACH' and context[3]=='FREE'),'Diagnostic phase regressed')
            last_stamp=stamp;last_context=context
            history=act._vector(request.get('history'),(2,17),'IPC causal history')
            history_times=act._vector(request.get('history_times'),(2,),'IPC history times')
            need(request.get('observation_contract')==contract.observation_contract() and
                 abs(history_times[-1]-stamp)<=1e-8 and abs(history_times[-1]-history_times[0]-.05)<=1e-8,
                 'IPC observation history contract/clock mismatch')
            block=predictor(request['measured_q'],request['estimated_target_base'],request['confidence'],metadata=metadata,
                history=history,history_times=history_times)
            print(json.dumps(dict(request_id=sequence,relative=block.values.tolist(),origin=block.observed_q.tolist(),
                metadata=dict(metadata,checkpoint_policy=pilot.SCHEMA),inference_context=envelope),allow_nan=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--serve',action='store_true')
    p.add_argument('--receipt');p.add_argument('--receipt-sha256');p.add_argument('--episode-id')
    p.add_argument('--pilot-complete');p.add_argument('--pilot-complete-sha256');p.add_argument('--attempt-plan')
    p.add_argument('--attempt-plan-sha256');p.add_argument('--lerobot-root');p.add_argument('--output')
    a=p.parse_args()
    if a.serve:serve(dict(file=a.receipt,sha256=a.receipt_sha256),a.episode_id)
    else:prepare_diagnostic(dict(file=a.pilot_complete,sha256=a.pilot_complete_sha256),
        dict(file=a.attempt_plan,sha256=a.attempt_plan_sha256),lerobot_root=a.lerobot_root,output=a.output)


if __name__=='__main__':main()
