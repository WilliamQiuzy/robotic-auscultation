"""Eight-case TRAIN/VAL ACT diagnostic, with an owned official-CPU pipe child.

This route does not issue a formal permit. Native GivenContactACT retains all
physical gates and consumes actual H20 predictions through its E10 executor.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from . import m2_given_contact_runtime as runtime
from . import m2_given_contact_pilot_train as pilot
from .m2_given_contact_deployment import FramedPipe, wire_decode

need=runtime.need
MODULE='sim.ausc_policy.m2_given_contact_pilot_deployment'
ROUTE_KEY='m2_given_contact_pilot_deployment'
DEPLOYMENT_SCHEMA='m2_given_contact_pilot_simulation_deployment_v1'
PROTOCOL_SCHEMA='m2_given_contact_pilot_observation_pipe_v1'
PREPARED_SCHEMA='m2_given_contact_pilot_native_preparation_v1'
MANIFEST_SHA256='5df6e25e6e9c24c9e85cd143d4b1befebab8e2a8c721f2bd2988c4c48fe2403b'
NATIVE_PYTHON='/opt/wam-runtime/envs/isaacsim60/bin/python'
DEFAULT_RUN_ID='m2-given-eight-act-native-pilot-20261029-v1'
DEFAULT_ATTEMPTS=(('p000449','R3'),('p001005','L3'))
FLAGS=dict(diagnostic_only=True,formal_acceptance=False,training_eligible=False,
    training_accepted=False,training_dataset_eligible=False,test_payloads_opened=False,
    simulation_only=True,learned_policy=True,expert_demonstration=False,expert_l2_fallback=False)


def validate_complete(value):
    need(value.get('schema')==pilot.PILOT_SCHEMA and value.get('complete') is True and
        value.get('completed_steps')==5000 and value.get('task_contract')==runtime.task_contract() and
        value.get('model')==runtime.shared.native_spec(), 'Complete 5000-step eight-case pilot required')
    need(all(value.get(k) is v for k,v in pilot.FLAGS.items()), 'Pilot declaration/TEST boundary changed')
    rows=value.get('evaluations',[])
    need([r.get('step') for r in rows]==list(range(500,5001,500)) and
        value.get('selected_step')==pilot.selected_step(rows), 'Pilot selected step differs from earliest VAL argmin')
    pilot.full_reference(value.get('best'))
    scope=value.get('scope',{})
    need(set(scope)=={'train','validation'} and all(scope[s].get('subject_ids') and scope[s].get('episode_ids')
        for s in scope), 'Explicit TRAIN/VAL-only checkpoint scope required')
    return value


def model_binding(ref):
    complete=validate_complete(runtime.read_bound(ref))
    # Hashing weights does not deserialize Torch or touch patient/scene payloads.
    runtime.bound_path(complete['best'])
    return dict(checkpoint=deepcopy(complete['best']),complete=deepcopy(ref)),complete


def sources():
    root=Path(__file__).resolve().parents[2]
    modules=('m2_given_contact_pilot_deployment.py','m2_given_contact_pilot_train.py',
        'm2_given_contact_pilot_native_audit.py','m2_given_contact_evaluation.py',
        'm2_given_contact_runtime.py','m2_given_contact_controller.py','m2_given_contact_act.py',
        'm2_given_contact_pose.py','m2_given_contact_intent.py','m2_given_contact_handoff.py','m2_stop_residual_trajectory.py',
        'm2_act.py','m2_act_controller.py','fresh_expert.py','geometry_qualification.py','scene_store.py')
    files={name:runtime.reference(root/'sim/ausc_policy'/name) for name in modules}
    commit_file=root/'CODE_COMMIT'
    commit=commit_file.read_text().strip() if commit_file.exists() else subprocess.check_output(
        ['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    return dict(root=str(root),code_commit=commit,files=files,
        source_manifest=runtime.reference(root/'SOURCE_MANIFEST.json') if (root/'SOURCE_MANIFEST.json').exists() else None)


def config_digest(config):
    return runtime.digest({k:v for k,v in config.items() if k!=ROUTE_KEY})


def derive_config(original,*,source_config,model,run_id,source_inventory,scope,stop_residual_trajectory=None):
    original=deepcopy(original);route=original.pop('m2_given_contact_control')
    meta=deepcopy(route['metadata']);split=meta.get('split');subject=meta.get('subject_id')
    need(split in ('train','validation') and subject in scope.get(split,{}).get('subject_ids',[]) and
        route['episode_id'] in scope[split].get('episode_ids',[]), 'Diagnostic case outside checkpoint TRAIN/VAL scope')
    source_episode=route['episode_id'];job=subject+'-'+meta['target_id'].lower()
    episode=run_id+'-'+job
    meta.update(episode_id=episode,**FLAGS)
    if stop_residual_trajectory is not None:
        from . import m2_stop_residual_trajectory as residual
        option={residual.CONFIG_KEY:stop_residual_trajectory}
        original=residual.configuration_projection(original,option,metadata=meta)
    original['fresh_expert'].pop('raw_multimodal_capture',None)
    if 'compact_scene_storage' in original:original['compact_scene_storage']['batch_id']=run_id
    original[ROUTE_KEY]=dict(schema=DEPLOYMENT_SCHEMA,model=deepcopy(model),metadata=meta,
        episode_id=episode,source_episode_id=source_episode,source_config=deepcopy(source_config),
        source_inventory=deepcopy(source_inventory),runtime_config_sha256=config_digest(original),**FLAGS)
    return original


def preflight(config,*,output=None):
    dep=config.get(ROUTE_KEY)
    need(isinstance(dep,dict) and dep.get('schema')==DEPLOYMENT_SCHEMA and
        all(dep.get(k) is v for k,v in FLAGS.items()), 'Explicit diagnostic pilot deployment required')
    need(not any(k in config for k in ('m2_given_contact_control','m2_given_contact_deployment',
        'm2_given_contact_evaluation','m2_act_deployment')) and
        'raw_multimodal_capture' not in config.get('fresh_expert',{}), 'Pilot cannot mix teacher/formal/TEST capture routes')
    model,complete=model_binding(dep['model']['complete'])
    need(dep['model']==model and dep['source_inventory']==sources() and
        dep['runtime_config_sha256']==config_digest(config), 'Pilot model/source/config binding changed')
    meta=dep['metadata'];split=meta.get('split');subject=meta.get('subject_id')
    need(split in ('train','validation') and subject in complete['scope'][split]['subject_ids'] and
        dep['source_episode_id'] in complete['scope'][split]['episode_ids'] and
        meta.get('episode_id')==dep.get('episode_id'), 'Only exact pilot TRAIN/VAL identities may execute')
    runtime.shared.require_membership(split,subject)
    ref=runtime.pose.require_route(meta)
    need(meta.get('handoff_policy_id')==runtime.handoff.POLICY_ID, 'Original physical handoff contract required')
    original=runtime.read_bound(dep['source_config']);old=original.get('m2_given_contact_control',{})
    expected_meta=deepcopy(old.get('metadata',{}));expected_meta.update(episode_id=dep['episode_id'],**FLAGS)
    need(old.get('episode_id')==dep['source_episode_id'] and meta==expected_meta,
        'Original qualified case/reference changed')
    projected=deepcopy(original);projected.pop('m2_given_contact_control')
    projected['fresh_expert'].pop('raw_multimodal_capture',None)
    if 'compact_scene_storage' in projected:
        projected['compact_scene_storage']['batch_id']=config['compact_scene_storage']['batch_id']
    if 'stop_residual_trajectory' in config:
        from . import m2_stop_residual_trajectory as residual
        projected=residual.configuration_projection(projected,config,metadata=meta)
    need(projected=={k:v for k,v in config.items() if k!=ROUTE_KEY},
        'Pilot altered original geometry/camera/physics/force/target configuration')
    plans=config.get('fresh_expert',{}).get('plans',[])
    need(len(plans)==1 and {k:plans[0].get(k) for k in ('file','sha256')}==ref.snapshot()['sources']['fresh_plan'] and
        config.get('isaac_python')==NATIVE_PYTHON, 'Exact qualified plan and Native60 interpreter required')
    if output is not None:
        need(Path(output).name==subject+'-'+meta['target_id'].lower(), 'Explicit diagnostic job output identity required')
    return dict(deployment=deepcopy(dep),complete=complete,reference=ref)


def prepare(complete_file,manifest_file,output,*,attempts=None,run_id=DEFAULT_RUN_ID):
    output=Path(output).resolve();need(not output.exists(),'Fresh diagnostic configuration directory required')
    model,complete=model_binding(runtime.reference(complete_file))
    manifest_ref=runtime.reference(manifest_file)
    need(manifest_ref['sha256']==MANIFEST_SHA256,'Only frozen physically qualified eight-case manifest permitted')
    manifest=runtime.read_bound(manifest_ref);rows=manifest.get('episodes',[])
    need(len(rows)==8 and all(r.get('split') in ('train','validation') for r in rows), 'Eight TRAIN/VAL source cases required')
    selected=list(DEFAULT_ATTEMPTS if attempts is None else attempts)
    need(selected and len(selected)==len(set(selected)),'Finite unique diagnostic attempts required')
    inventory=sources();jobs=[];configs=[]
    # Authenticate all selected metadata before opening any selected configuration.
    for subject,target in selected:
        matches=[r for r in rows if (r['subject_id'],r['target_id'])==(subject,target)]
        need(len(matches)==1,'Requested diagnostic case missing or ambiguous')
        row=matches[0];split=row['split']
        need(subject in complete['scope'][split]['subject_ids'] and row['episode_id'] in complete['scope'][split]['episode_ids'],
            'Requested diagnostic case outside trained pilot scope')
        original=runtime.read_bound(row['config'])
        config=derive_config(original,source_config=row['config'],model=model,run_id=run_id,
            source_inventory=inventory,scope=complete['scope'])
        preflight(config);configs.append((subject+'-'+target.lower(),config))
    output.mkdir(parents=True,exist_ok=False)
    for job,config in configs:
        ref=runtime.write_new(output/(job+'.json'),config);jobs.append(dict(id=job,config=ref['path'],sha256=ref['sha256']))
    first=configs[0][1]
    batch=dict(experiment='isaac_batch',base=first['base'],parallelism=1,
        time_limit_seconds=sum(c['time_limit_seconds'] for _,c in configs)+300,jobs=jobs,**FLAGS)
    batch_ref=runtime.write_new(output/'batch.json',batch)
    return runtime.write_new(output/'PREPARED.json',dict(schema=PREPARED_SCHEMA,model=model,manifest=manifest_ref,
        source_inventory=inventory,attempts=[list(x) for x in selected],batch=batch_ref,run_id=run_id,**FLAGS))


def make_request(dep,sequence,q,target,confidence,*,metadata,history,history_times,context,protocol_schema=PROTOCOL_SCHEMA,binding_extra=None):
    binding=dict(checkpoint_sha256=dep['model']['checkpoint']['sha256'],episode_id=dep['episode_id'],
            source_episode_id=dep['source_episode_id'],source_config_sha256=dep['source_config']['sha256'],
            target_id=metadata['target_id'],reference_sha256=metadata['reference_sha256'])
    if binding_extra is not None:
        need(not set(binding)&set(binding_extra),'Additional protocol binding cannot replace original case/model fields')
        binding.update(binding_extra)
    return dict(schema=protocol_schema,request_id=sequence,binding=binding,
        metadata=deepcopy(metadata),measured_q=np.asarray(q).tolist(),estimated_target_base=np.asarray(target).tolist(),
        confidence=float(confidence),history=np.asarray(history).tolist(),history_times=np.asarray(history_times).tolist(),
        context=deepcopy(context))


def validate_request(request,dep,*,sequence,previous=None,protocol_schema=PROTOCOL_SCHEMA,binding_extra=None):
    need(isinstance(request,dict) and request.get('schema')==protocol_schema and request.get('request_id')==sequence and
        request.get('metadata')==dep['metadata'], 'Pilot protocol/sequence/case differs')
    expected=make_request(dep,sequence,request['measured_q'],request['estimated_target_base'],request['confidence'],
        metadata=request['metadata'],history=request['history'],history_times=request['history_times'],context=request['context'],
        protocol_schema=protocol_schema,binding_extra=binding_extra)
    need(request==expected,'Pilot checkpoint/case/target/reference binding differs')
    meta=request['metadata'];ref=runtime.pose.require_route(meta)
    q=runtime.shared._vector(request['measured_q'],(7,),'measured origin')
    history=np.asarray(request['history'],dtype=float);times=np.asarray(request['history_times'],dtype=float)
    state=runtime.shared.contract.encode_state(q,request['estimated_target_base'],request['confidence'],metadata=meta)
    context=request['context']
    need(history.shape==(2,17) and times.shape==(2,) and np.isfinite(history).all() and np.isfinite(times).all() and
        times[0]>=0 and abs(times[1]-times[0]-.05)<=1e-8 and np.array_equal(history[1],state) and
        request['confidence']==1. and np.array_equal(request['estimated_target_base'],ref.pose_base) and
        isinstance(context,dict) and abs(context['observed_at_s']-times[1])<=1e-8 and
        context['context'][:3]==[dep['episode_id'],meta['target_id'],0] and context['context'][3] in ('FREE','APPROACH') and
        type(context['physics_row']) is int and context['physics_row']>=12 and
        context['source_id']=='given-contact-'+ref.sha256, 'Two causal frames and same-time given observation required')
    if previous is not None:
        need(times[1]>previous['context']['observed_at_s'] and context['physics_row']>previous['context']['physics_row'],
            'Pilot inference clock/native rows did not advance')
    return request


def response_for(request,relative,*,protocol_schema=PROTOCOL_SCHEMA):
    relative=np.asarray(relative,dtype=float)
    need(relative.shape==(20,7) and np.isfinite(relative).all(),'Finite actual H20 prediction required')
    return dict(schema=protocol_schema,request_id=request['request_id'],request_sha256=runtime.digest(request),
        binding=deepcopy(request['binding']),context=deepcopy(request['context']),origin=deepcopy(request['measured_q']),
        metadata=deepcopy(request['metadata']),relative=relative.tolist())


def decode_response(value,request,*,protocol_schema=PROTOCOL_SCHEMA,checkpoint_schema=pilot.PILOT_SCHEMA):
    expected=response_for(request,value.get('relative'),protocol_schema=protocol_schema)
    need(value==expected,'Pilot response checkpoint/request/case/reference/origin differs')
    return runtime.shared.contract.JointActionBlock('block_relative_joint',value['relative'],value['origin'],
        dict(value['metadata'],checkpoint_policy=checkpoint_schema,**FLAGS))


def load_model(dep):
    model,complete=model_binding(dep['model']['complete'])
    need(model==dep['model'],'Pilot selected checkpoint binding changed')
    torch,policy,software=runtime.make_model({'lerobot_root':runtime.VENDOR_ROOT},device='cpu')
    saved_software=runtime.read_bound(complete['inputs']['software'])
    # Source roots differ between the frozen trainer and diagnostic runtime;
    # the common numerical modules must retain the exact trained source bytes.
    for key in ('vendor','dependencies','python','prefix','python_version'):
        need(saved_software[key]==software[key],'Official pilot CPU environment changed: '+key)
    need({Path(k).name:v for k,v in saved_software['source_bindings'].items()}==
        {Path(k).name:v for k,v in software['source_bindings'].items()},'Trained numerical source bytes changed')
    raw=runtime.bound_path(model['checkpoint']).read_bytes()
    need(runtime.hashlib.sha256(raw).hexdigest()==model['checkpoint']['sha256'],'Checkpoint changed while reading')
    checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    need(checkpoint.get('schema')==pilot.PILOT_SCHEMA and checkpoint.get('step')==complete['selected_step'] and
        checkpoint.get('inputs')==complete['inputs'] and checkpoint.get('scope')==complete['scope'] and
        checkpoint.get('task_contract')==runtime.task_contract() and checkpoint.get('model')==runtime.shared.native_spec() and
        all(checkpoint.get(k) is v for k,v in pilot.FLAGS.items()),'Exact selected pilot checkpoint required')
    runtime.validate_normalization(checkpoint['normalization'])
    policy.load_state_dict(checkpoint['state_dict'],strict=True);policy.eval()
    return torch,policy,checkpoint


class CPUBridge:
    def __init__(self,deployment,*,output,adapter=None):
        from . import process_guard as guard
        need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1','Owned native Linux parent required')
        self.adapter=sys.modules[__name__] if adapter is None else adapter
        self.dep=deepcopy(deployment);self.output=Path(output);self.process=None;self.stderr=None;self.trace=None
        self.sequence=0;self.previous=None;self.context=None;self.exit_receipt=None
        self.parent=guard.process_identity(os.getpid());self.identity=None
        ref=runtime.write_new(self.output/'pilot_cpu_deployment.json',self.dep)
        root=Path(__file__).resolve().parents[2]
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',
            PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
        for key in ('PYTHONHOME','PYTHONUSERBASE','HF_TOKEN'):environment.pop(key,None)
        python=str(Path(runtime.VENDOR_ROOT)/'.venv/bin/python')
        command=[python,'-u','-B','-m',self.adapter.MODULE,'--serve','--deployment',ref['path'],'--deployment-sha256',ref['sha256']]
        try:
            self.stderr=(self.output/'pilot_cpu_stderr.log').open('xb')
            self.trace=(self.output/'pilot_cpu_ipc.jsonl').open('x',buffering=1)
            self.process=subprocess.Popen(command,cwd=root,env=environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=self.stderr,start_new_session=False,close_fds=True,bufsize=0)
            self.identity=guard.process_identity(self.process.pid)
            need(self.identity is not None and self.parent is not None and all(self.identity[k]==self.parent[k]
                for k in ('pgid','sid')),'Pilot CPU child left original native session')
            self.pipe=FramedPipe(self.process.stdout,self.process.stdin)
            ready=self.pipe.read(timeout=180.)
            need(ready.get('ready') is True and ready.get('schema')==self.adapter.PROTOCOL_SCHEMA and
                ready.get('deployment_sha256')==ref['sha256'] and ready.get('checkpoint')==deployment['model']['checkpoint'] and
                ready.get('pid')==self.process.pid and ready.get('parent_pid')==os.getpid() and
                ready.get('source_inventory')==deployment['source_inventory'] and self.process.poll() is None,
                'Pilot CPU startup/model/source/identity handshake failed')
            runtime.validate_execution_settings(ready['execution'],device='cpu')
            self.runtime=ready;self.trace.write(json.dumps(dict(event='ready',value=ready),allow_nan=False)+'\n')
        except BaseException:self.close();raise

    def __call__(self,q,target,confidence,*,metadata,history,history_times):
        need(self.process is not None and self.process.poll() is None and isinstance(self.context,dict),'Live pilot child/context required')
        self.sequence+=1;row=dict(event='inference',request_id=self.sequence,accepted=False);started=time.monotonic()
        try:
            request=self.adapter.make_request(self.dep,self.sequence,q,target,confidence,metadata=metadata,history=history,
                history_times=history_times,context=self.context)
            self.adapter.validate_request(request,self.dep,sequence=self.sequence,previous=self.previous);row['request']=request
            response=self.pipe.exchange(request,timeout=10.);row['response']=response
            block=self.adapter.decode_response(response,request);row['accepted']=True;self.previous=request
            return block
        except BaseException as error:row['error']=repr(error);raise
        finally:
            row['wall_latency_s']=time.monotonic()-started
            if self.trace is not None:self.trace.write(json.dumps(row,allow_nan=False)+'\n')
            if not row['accepted']:self.close()

    def close(self):
        if self.exit_receipt is not None:return self.exit_receipt
        child=self.process;signals=[]
        try:
            if child is not None:
                if child.stdin:
                    try:child.stdin.close()
                    except OSError:pass
                try:child.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    child.terminate();signals.append('SIGTERM')
                    try:child.wait(timeout=2.)
                    except subprocess.TimeoutExpired:child.kill();signals.append('SIGKILL');child.wait(timeout=2.)
        finally:
            if child is not None and child.stdout:child.stdout.close()
            for stream in (self.trace,self.stderr):
                if stream is not None:stream.close()
            self.trace=None;self.stderr=None
        receipt=dict(schema=getattr(self.adapter,'CPU_EXIT_SCHEMA','m2_given_contact_pilot_cpu_exit_v1'),identity=self.identity,parent_identity=self.parent,
            started=child is not None,reaped=child is None or child.poll() is not None,
            exit_code=None if child is None else child.returncode,normal_exit=child is not None and child.returncode==0,
            signals=signals,request_count=self.sequence,**FLAGS)
        if child is not None:
            receipt.update(stderr=runtime.reference(self.output/'pilot_cpu_stderr.log'),trace=runtime.reference(self.output/'pilot_cpu_ipc.jsonl'))
        runtime.write_new(self.output/'pilot_cpu_exit.json',receipt);self.exit_receipt=receipt
        return receipt


def serve(deployment_ref):
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and
        os.environ.get('CUDA_VISIBLE_DEVICES')=='','Owned official CPU child must hide CUDA')
    dep=runtime.read_bound(deployment_ref)
    need(dep.get('schema')==DEPLOYMENT_SCHEMA and dep.get('source_inventory')==sources(),'Pilot child source binding changed')
    with redirect_stdout(sys.stderr):torch,policy,checkpoint=load_model(dep)
    def send(value):print(json.dumps(value,allow_nan=False,separators=(',',':')),flush=True)
    send(dict(schema=PROTOCOL_SCHEMA,ready=True,deployment_sha256=deployment_ref['sha256'],checkpoint=dep['model']['checkpoint'],
        pid=os.getpid(),parent_pid=os.getppid(),source_inventory=sources(),execution=runtime.execution_settings(torch,policy,device='cpu')))
    sequence=0;previous=None
    while True:
        raw=sys.stdin.buffer.readline(32769)
        if not raw:return
        need(len(raw)<=32768 and raw.endswith(b'\n'),'Bounded complete pilot request required')
        request=wire_decode(raw);sequence+=1;validate_request(request,dep,sequence=sequence,previous=previous)
        history=np.asarray(request['history'],dtype=float)
        with redirect_stdout(sys.stderr):
            relative=runtime.infer(torch,policy,dict(states=history[None,1],histories=history[None]),
                checkpoint['normalization'],device='cpu')[0]
        send(response_for(request,relative));previous=request


from .m2_given_contact_controller import GivenContactACT


class PilotGivenACT(GivenContactACT):
    def _tick(self,command,subphase,active=False,recovery=False):
        start=len(self.env.rows)
        try:return super()._tick(command,subphase,active=active,recovery=recovery)
        finally:
            for row in self.env.rows[start:]:
                row.update(FLAGS)
                if self.control_owner=='ACT_L2':row['command_source_id']='given_contact_pilot_act_l2_interpolated_v1'
                elif self.control_owner=='L2_STOP':row['command_source_id']='given_contact_pilot_bounded_parking_v1'
                intent=getattr(self,'_current_intent_audit',None)
                row['pilot_act_attribution']=dict(checkpoint_sha256=self.policy_receipt['model']['checkpoint']['sha256'],
                    episode_id=self.episode_id,reference_sha256=self.reference.sha256,control_owner=self.control_owner,
                    query_index=len(self.l2_queries)-1,action_index=len(self.l2_actions)-1,
                    raw_target_q=None if intent is None else deepcopy(intent['live_target_q']))
    def _query(self,observation):
        self.predictor.context=dict(observed_at_s=float(observation['observed_at_s']),physics_row=len(self.env.rows),
            context=list(observation['context']),source_id=observation['source_id'])
        try:return super()._query(observation)
        finally:self.predictor.context=None
    def close_bridge(self):
        receipt=self.predictor.close();self.policy_receipt['cpu_child_exit']=receipt;return receipt
    def run(self,points):
        try:return super().run(points)
        finally:self.close_bridge()


def make_controller(*args,deployment,estimate_provider,output,_adapter=None,**kwargs):
    adapter=sys.modules[__name__] if _adapter is None else _adapter
    env=args[0];before=env.state();count=len(env.rows);q=np.asarray(before['q']).copy();stamp=before['time_s']
    bridge=CPUBridge(deployment,output=output,adapter=adapter)
    try:
        after=env.state()
        need(len(env.rows)==count and after['time_s']==stamp and np.array_equal(after['q'],q),'Physics advanced during pilot CPU startup')
        receipt=dict(schema=adapter.DEPLOYMENT_SCHEMA,model=deployment['model'],metadata=deployment['metadata'],
            source_inventory=deployment['source_inventory'],child_runtime=bridge.runtime,state_only=True,
            action_semantics_id=runtime.given.ACTION_SEMANTICS_ID,task_contract=runtime.task_contract(),**FLAGS)
        return adapter.PilotGivenACT(*args,predictor=bridge,metadata=deployment['metadata'],episode_id=deployment['episode_id'],
            estimate_provider=estimate_provider,policy_receipt=receipt,**kwargs)
    except BaseException:bridge.close();raise


def tag_result(result,deployment,controller=None):
    from .m2_act_controller import tag_learned_result
    receipt=getattr(controller,'policy_receipt',dict(schema=DEPLOYMENT_SCHEMA,configured_deployment=deployment,controller_constructed=False,**FLAGS))
    handoffs=getattr(controller,'l2_handoffs',[]) or [dict(l2_handoff_success=False,expert_l2_fallback=False)]
    tag_learned_result(result,receipt,handoffs)
    for row in [result,*result.get('points',[])]:row.update(FLAGS)
    result.update(executed_controller_class=MODULE+'.PilotGivenACT',
        controller_source='eight-case pilot ACT L2 + original analytic L3',
        scope='Reused eight-case TRAIN/VAL diagnostic; no formal population or held-out qualification',
        diagnostic_closed_loop_evaluated=any(a.get('physics_row_end_exclusive',a['physics_row_start'])>a['physics_row_start']
            for a in getattr(controller,'l2_actions',[])),
        l2_queries=deepcopy(getattr(controller,'l2_queries',[])),l2_actions=deepcopy(getattr(controller,'l2_actions',[])),
        raw_capture_complete=False,compact_capture_complete=False)
    for row in result['l2_queries']+result['l2_actions']+result['l2_handoffs']:row.update(FLAGS)
    if controller is not None:
        result['given_reference']=controller.estimate_provider.snapshot();exit_receipt=controller.close_bridge()
        result['cpu_child_exit']=exit_receipt
        if not exit_receipt['normal_exit']:
            result.update(success=False,physical_validation=False,failure_reason='Pilot CPU child did not exit normally')
    return result


def run_native_probe(app,stage,output,config,mass):
    from . import fresh_expert
    lifecycle={};deployment=config.get(ROUTE_KEY,{})
    try:
        preflight(config,output=output)
        return fresh_expert._run_probe(app,stage,output,config,mass,_learned_lifecycle=lifecycle)
    except BaseException as error:
        result=dict(success=False,physical_validation=False,diagnostic_completed=False,points=[],
            physics_steps=len(lifecycle['env'].rows) if lifecycle.get('env') is not None else 0,
            failure_reason=str(error),errors=[dict(stage='pilot_native',error=repr(error))],
            runtime_sources_sha256=lifecycle.get('source_hashes',{}))
        tag_result(result,deployment,lifecycle.get('controller'))
        path=Path(output)/'fresh_expert_result.json'
        if not path.exists():runtime.write_new(path,result)
        if not isinstance(error,Exception):raise
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--serve',action='store_true')
    parser.add_argument('--complete');parser.add_argument('--manifest');parser.add_argument('--output')
    parser.add_argument('--attempts',help='JSON list of [subject_id,target_id] pairs; defaults to p000449/R3,p001005/L3')
    parser.add_argument('--run-id',default=DEFAULT_RUN_ID);parser.add_argument('--deployment');parser.add_argument('--deployment-sha256')
    args=parser.parse_args()
    if args.serve:
        need(args.deployment and args.deployment_sha256,'Bound CPU deployment reference required')
        serve(dict(path=str(Path(args.deployment).resolve()),sha256=args.deployment_sha256))
    else:
        need(args.complete and args.manifest and args.output,'Complete/manifest/new output directory required')
        attempts=None if not args.attempts else [tuple(row) for row in json.loads(Path(args.attempts).read_text())]
        ref=prepare(args.complete,args.manifest,args.output,attempts=attempts,run_id=args.run_id)
        print(json.dumps(dict(prepared=ref,**FLAGS)),flush=True)


if __name__=='__main__':main()
