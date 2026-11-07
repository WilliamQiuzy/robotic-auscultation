"""Expanded TRAIN/VAL ACT diagnostic using the original guarded native hook.

Each configuration binds the actual completed expanded training run, selected
weights and exact expert source case. No expert L2 or formal/TEST route.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import m2_given_contact_runtime as runtime
from . import m2_given_contact_expanded_pilot_train as pilot
from . import m2_given_contact_pilot_deployment as original

need=runtime.need
MODULE='sim.ausc_policy.m2_given_contact_expanded_pilot_deployment'
ROUTE_KEY='m2_given_contact_expanded_pilot_deployment'
DEPLOYMENT_SCHEMA='m2_given_contact_expanded_pilot_simulation_deployment_v1'
PROTOCOL_SCHEMA='m2_given_contact_expanded_pilot_observation_pipe_v1'
PREPARED_SCHEMA='m2_given_contact_expanded_pilot_native_preparation_v1'
CPU_EXIT_SCHEMA='m2_given_contact_expanded_pilot_cpu_exit_v1'
NATIVE_PYTHON=original.NATIVE_PYTHON
DEFAULT_RUN_ID='m2-given-diverse-act-native-pilot-20261029-v1'
FLAGS=deepcopy(original.FLAGS)


def validate_complete(value):
    if value.get('schema') in ('m2_act_auxiliary_act_pilot_v1','m2_act_auxiliary_act_pilot_v2','m2_act_auxiliary_act_pilot_v3'):
        from . import m2_act_auxiliary_train
        return m2_act_auxiliary_train.validate_complete(value)
    need(value.get('schema')==pilot.PILOT_SCHEMA and value.get('complete') is True and
        value.get('task_contract')==runtime.task_contract() and value.get('model')==runtime.shared.native_spec() and
        all(value.get(k) is v for k,v in pilot.FLAGS.items()),'Complete expanded diagnostic ACT run required')
    config=pilot.validate_config(value.get('schedule',{}))
    need(value.get('completed_steps')==config['steps'] and value.get('protocol')==pilot.training_protocol(config),
        'Completed expanded schedule/protocol differs')
    rows=value.get('evaluations',[]);scope=value.get('scope',{})
    need([row.get('step') for row in rows]==pilot.evaluation_steps(config) and
        value.get('selected_step')==pilot.selected_step(rows),'Expanded selected step differs from earliest full-VAL argmin')
    need(set(scope)=={'train','validation'} and all(scope[s].get('subject_ids') and scope[s].get('episode_ids') and
        type(scope[s].get('window_count')) is int and scope[s]['window_count']>0 for s in scope),
        'Explicit positive expanded TRAIN/VAL scope required')
    need(not set(scope['train']['subject_ids'])&set(scope['validation']['subject_ids']),'Expanded TRAIN/VAL identities overlap')
    for split in scope:
        for subject in scope[split]['subject_ids']:runtime.shared.require_membership(split,subject)
        for row in rows:
            metrics=runtime.validate_metrics(row.get(split,{}))
            need(metrics['sample_count']==scope[split]['window_count'],'Complete expanded evaluation population differs')
        selected=runtime.validate_metrics(value.get('selected_metrics',{}).get(split,{}))
        need(selected==next(row[split] for row in rows if row['step']==value['selected_step']),
            'Expanded selected metrics differ from selected full evaluation')
    pilot.full_reference(value.get('best'));pilot.full_reference(value.get('cpu_equivalence'))
    need(value.get('inputs',{}).get('dataset')==config['dataset'],'Expanded completed dataset binding differs')
    return value


def model_binding(ref):
    if runtime.read_bound(ref).get('schema') in ('m2_act_auxiliary_act_pilot_v1','m2_act_auxiliary_act_pilot_v2','m2_act_auxiliary_act_pilot_v3'):
        from . import m2_act_auxiliary_train
        return m2_act_auxiliary_train.model_binding(ref)
    pilot.full_reference(ref);complete=validate_complete(runtime.read_bound(ref))
    runtime.bound_path(complete['best'])
    cpu=runtime.read_bound(complete['cpu_equivalence'])
    need(cpu.get('schema')==pilot.CPU_SCHEMA and cpu.get('passed') is True and
        cpu.get('checkpoint')==complete['best'] and cpu.get('selected_step')==complete['selected_step'] and
        cpu.get('scope')==complete['scope'] and all(cpu.get(k) is v for k,v in pilot.FLAGS.items()) and
        set(cpu.get('comparisons',{}))=={'train','validation'} and
        all(row.get('passed') is True for row in cpu['comparisons'].values()),
        'Actual selected expanded CPU load/inference equality receipt required')
    selection=runtime.read_bound(cpu['selection'])
    need(selection.get('inputs')==complete['inputs'] and selection.get('best')==complete['best'] and
        selection.get('protocol')==complete['protocol'] and selection.get('selected_step')==complete['selected_step'] and
        selection.get('window_order')==complete['window_order'], 'Expanded CPU receipt selection binding differs')
    pilot.validate_code(complete['inputs']['code'])
    return dict(checkpoint=deepcopy(complete['best']),complete=deepcopy(ref)),complete


def sources():
    result=original.sources();root=Path(__file__).resolve().parents[2]
    for name in (Path(__file__).name,Path(pilot.__file__).name):result['files'][name]=runtime.reference(root/'sim/ausc_policy'/name)
    from . import m2_act_auxiliary_train
    result['files'].update(m2_act_auxiliary_train.code_binding(version=3)['files'])
    return result


def config_digest(config):return runtime.digest({k:v for k,v in config.items() if k!=ROUTE_KEY})


def derive_config(original_config,*,source_config,model,run_id,source_inventory,scope,stop_residual_trajectory=None):
    config=deepcopy(original_config);route=config.pop('m2_given_contact_control')
    meta=deepcopy(route['metadata']);split=meta.get('split');subject=meta.get('subject_id')
    need(split in ('train','validation') and subject in scope.get(split,{}).get('subject_ids',[]) and
        route['episode_id'] in scope[split].get('episode_ids',[]),'Expanded diagnostic case outside checkpoint TRAIN/VAL scope')
    episode=run_id+'-'+subject+'-'+meta['target_id'].lower();meta.update(episode_id=episode,**FLAGS)
    if stop_residual_trajectory is not None:
        from . import m2_stop_residual_trajectory as residual
        option={residual.CONFIG_KEY:stop_residual_trajectory}
        config=residual.configuration_projection(config,option,metadata=meta)
    config['fresh_expert'].pop('raw_multimodal_capture',None)
    if 'compact_scene_storage' in config:config['compact_scene_storage']['batch_id']=run_id
    config[ROUTE_KEY]=dict(schema=DEPLOYMENT_SCHEMA,model=deepcopy(model),metadata=meta,episode_id=episode,
        source_episode_id=route['episode_id'],source_config=deepcopy(source_config),source_inventory=deepcopy(source_inventory),
        runtime_config_sha256=config_digest(config),**FLAGS)
    return config


def preflight(config,*,output=None):
    dep=config.get(ROUTE_KEY)
    need(isinstance(dep,dict) and dep.get('schema')==DEPLOYMENT_SCHEMA and all(dep.get(k) is v for k,v in FLAGS.items()),
        'Explicit expanded diagnostic deployment required')
    need(not any(k in config for k in ('m2_given_contact_control','m2_given_contact_deployment','m2_given_contact_evaluation',
        'm2_act_deployment',original.ROUTE_KEY)) and 'raw_multimodal_capture' not in config.get('fresh_expert',{}),
        'Expanded pilot cannot mix expert/formal/TEST/old pilot routes')
    model,complete=model_binding(dep['model']['complete'])
    need(dep['model']==model and dep['source_inventory']==sources() and dep['runtime_config_sha256']==config_digest(config),
        'Expanded actual model/source/config binding changed')
    from .m2_act_posture_head import validate_runtime_head
    validate_runtime_head(complete,config)
    meta=dep['metadata'];split=meta.get('split');subject=meta.get('subject_id')
    need(split in ('train','validation') and subject in complete['scope'][split]['subject_ids'] and
        dep['source_episode_id'] in complete['scope'][split]['episode_ids'] and meta.get('episode_id')==dep.get('episode_id'),
        'Only exact expanded TRAIN/VAL identities may execute')
    runtime.shared.require_membership(split,subject);ref=runtime.pose.require_route(meta)
    need(meta.get('handoff_policy_id')==runtime.handoff.POLICY_ID,'Original physical handoff required')
    preparation=runtime.read_bound(complete['inputs']['dataset'])
    matches=[r for r in preparation.get('episodes',[]) if r.get('episode_id')==dep['source_episode_id']]
    need(preparation.get('schema') in pilot.PREPARATION_SCHEMAS and len(matches)==1 and
        matches[0].get('config')==dep['source_config'] and matches[0].get('subject_id')==subject and
        matches[0].get('split')==split and matches[0].get('target_id')==meta.get('target_id'),
        'Expanded actual trained successful expert case/config differs')
    original_config=runtime.read_bound(dep['source_config']);old=original_config.get('m2_given_contact_control',{})
    expected=deepcopy(old.get('metadata',{}));expected.update(episode_id=dep['episode_id'],**FLAGS)
    need(old.get('episode_id')==dep['source_episode_id'] and meta==expected,'Original qualified expanded case/reference changed')
    projected=deepcopy(original_config);projected.pop('m2_given_contact_control');projected['fresh_expert'].pop('raw_multimodal_capture',None)
    if 'compact_scene_storage' in projected:projected['compact_scene_storage']['batch_id']=config['compact_scene_storage']['batch_id']
    if 'stop_residual_trajectory' in config:
        from . import m2_stop_residual_trajectory as residual
        projected=residual.configuration_projection(projected,config,metadata=meta)
    need(projected=={k:v for k,v in config.items() if k!=ROUTE_KEY},'Expanded pilot altered original geometry/camera/physics/force/target configuration')
    plans=config.get('fresh_expert',{}).get('plans',[])
    need(len(plans)==1 and {k:plans[0].get(k) for k in ('file','sha256')}==ref.snapshot()['sources']['fresh_plan'] and
        config.get('isaac_python')==NATIVE_PYTHON,'Exact qualified plan and Native60 interpreter required')
    if output is not None:need(Path(output).name==subject+'-'+meta['target_id'].lower(),'Explicit expanded job output identity required')
    return dict(deployment=deepcopy(dep),complete=complete,reference=ref)


def prepare(complete_file,manifest_file,output,*,attempts=None,run_id=DEFAULT_RUN_ID):
    output=Path(output).resolve();need(not output.exists(),'Fresh expanded diagnostic configurations required')
    model,complete=model_binding(runtime.reference(complete_file));manifest_ref=runtime.reference(manifest_file)
    need(manifest_ref==complete['inputs']['dataset'],'Expanded preparation manifest must be the actual trained dataset COMPLETE')
    manifest=runtime.read_bound(manifest_ref);rows=manifest.get('episodes',[])
    need(manifest.get('schema') in pilot.PREPARATION_SCHEMAS and rows and
        all(row.get('split') in ('train','validation') for row in rows),'Expanded successful TRAIN/VAL cases required')
    selected=[(row['subject_id'],row['target_id']) for row in rows] if attempts is None else list(attempts)
    need(selected and len(selected)==len(set(selected)),'Finite unique expanded attempts required')
    inventory=sources();configs=[];jobs=[]
    for subject,target in selected:
        matches=[row for row in rows if (row['subject_id'],row['target_id'])==(subject,target)]
        need(len(matches)==1,'Requested expanded case missing or ambiguous')
        row=matches[0];pilot.full_reference(row.get('config'))
        config=derive_config(runtime.read_bound(row['config']),source_config=row['config'],model=model,
            run_id=run_id,source_inventory=inventory,scope=complete['scope'])
        preflight(config);configs.append((subject+'-'+target.lower(),config))
    output.mkdir(parents=True,exist_ok=False)
    for job,config in configs:
        ref=runtime.write_new(output/(job+'.json'),config);jobs.append(dict(id=job,config=ref['path'],sha256=ref['sha256']))
    batch=dict(experiment='isaac_batch',base=configs[0][1]['base'],parallelism=1,
        time_limit_seconds=sum(c['time_limit_seconds'] for _,c in configs)+300,jobs=jobs,**FLAGS)
    return runtime.write_new(output/'PREPARED.json',dict(schema=PREPARED_SCHEMA,model=model,manifest=manifest_ref,
        source_inventory=inventory,attempts=[list(x) for x in selected],batch=runtime.write_new(output/'batch.json',batch),run_id=run_id,**FLAGS))


def _checkpoint_binding(dep):
    schema=dep.get('model',{}).get('checkpoint_schema')
    need(schema in (None,'m2_act_auxiliary_act_pilot_v1','m2_act_auxiliary_act_pilot_v2','m2_act_auxiliary_act_pilot_v3'),'Explicit supported checkpoint schema required')
    return None if schema is None else dict(checkpoint_schema=schema)


def make_request(dep,*args,**kwargs):
    return original.make_request(dep,*args,protocol_schema=PROTOCOL_SCHEMA,binding_extra=_checkpoint_binding(dep),**kwargs)


def validate_request(request,dep,**kwargs):
    return original.validate_request(request,dep,protocol_schema=PROTOCOL_SCHEMA,binding_extra=_checkpoint_binding(dep),**kwargs)


def response_for(*args,**kwargs):return original.response_for(*args,protocol_schema=PROTOCOL_SCHEMA,**kwargs)
def decode_response(value,request,**kwargs):
    schema=request.get('binding',{}).get('checkpoint_schema',pilot.PILOT_SCHEMA)
    need(schema in (pilot.PILOT_SCHEMA,'m2_act_auxiliary_act_pilot_v1','m2_act_auxiliary_act_pilot_v2','m2_act_auxiliary_act_pilot_v3'),'Explicit supported checkpoint schema required')
    return original.decode_response(value,request,protocol_schema=PROTOCOL_SCHEMA,checkpoint_schema=schema,**kwargs)


def load_model(dep):
    model,complete=model_binding(dep['model']['complete']);need(model==dep['model'],'Expanded selected model changed')
    return pilot.load_selected_policy(complete)


def serve(deployment_ref):
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
        'Owned official expanded CPU child must hide CUDA')
    dep=runtime.read_bound(deployment_ref)
    need(dep.get('schema')==DEPLOYMENT_SCHEMA and dep.get('source_inventory')==sources(),'Expanded CPU source binding changed')
    with redirect_stdout(sys.stderr):torch,policy,checkpoint=load_model(dep)
    def send(value):print(json.dumps(value,allow_nan=False,separators=(',',':')),flush=True)
    send(dict(schema=PROTOCOL_SCHEMA,ready=True,deployment_sha256=deployment_ref['sha256'],checkpoint=dep['model']['checkpoint'],
        pid=os.getpid(),parent_pid=os.getppid(),source_inventory=sources(),execution=runtime.execution_settings(torch,policy,device='cpu')))
    sequence=0;previous=None
    while True:
        raw=sys.stdin.buffer.readline(32769)
        if not raw:return
        need(len(raw)<=32768 and raw.endswith(b'\n'),'Bounded complete expanded request required')
        request=original.wire_decode(raw);sequence+=1;validate_request(request,dep,sequence=sequence,previous=previous)
        history=np.asarray(request['history'],dtype=float)
        with redirect_stdout(sys.stderr):
            relative=runtime.infer(torch,policy,dict(states=history[None,1],histories=history[None]),checkpoint['normalization'],device='cpu')[0]
        send(response_for(request,relative));previous=request


class ExpandedGivenACT(original.PilotGivenACT):
    """Reuse the qualified hook, force, collision, handoff and parking logic."""
    def _tick(self,*args,**kwargs):
        start=len(self.env.rows)
        try:return super()._tick(*args,**kwargs)
        finally:
            for row in self.env.rows[start:]:
                if self.control_owner=='ACT_L2':row['command_source_id']='given_contact_expanded_pilot_act_l2_interpolated_v1'
                elif self.control_owner=='L2_STOP':row['command_source_id']='given_contact_expanded_pilot_bounded_parking_v1'


PilotGivenACT=ExpandedGivenACT


def make_controller(*args,**kwargs):return original.make_controller(*args,_adapter=sys.modules[__name__],**kwargs)


def tag_result(result,deployment,controller=None):
    original.tag_result(result,deployment,controller)
    result.update(executed_controller_class=MODULE+'.ExpandedGivenACT',
        controller_source='expanded pilot ACT L2 + original analytic L3',
        scope='Reused expanded TRAIN/VAL diagnostic; no formal population or held-out qualification')
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
            failure_reason=str(error),errors=[dict(stage='expanded_pilot_native',error=repr(error))],
            runtime_sources_sha256=lifecycle.get('source_hashes',{}))
        tag_result(result,deployment,lifecycle.get('controller'));path=Path(output)/'fresh_expert_result.json'
        if not path.exists():runtime.write_new(path,result)
        if not isinstance(error,Exception):raise
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--serve',action='store_true')
    for name in ('complete','manifest','output','attempts','deployment','deployment-sha256'):parser.add_argument('--'+name)
    parser.add_argument('--run-id',default=DEFAULT_RUN_ID);args=parser.parse_args()
    if args.serve:
        need(args.deployment and args.deployment_sha256,'Bound expanded CPU deployment required')
        serve(dict(path=str(Path(args.deployment).resolve()),sha256=args.deployment_sha256))
    else:
        need(args.complete and args.manifest and args.output,'Actual COMPLETE, dataset manifest and new output required')
        attempts=None if args.attempts is None else [tuple(row) for row in json.loads(Path(args.attempts).read_text())]
        print(json.dumps(prepare(args.complete,args.manifest,args.output,attempts=attempts,run_id=args.run_id)),flush=True)


if __name__=='__main__':main()
