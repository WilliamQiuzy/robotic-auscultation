"""Thirty paired held-out VALIDATION scenarios for the expanded diagnostic ACT.

All identities are disjoint from optimizer TRAIN. Twenty-four scenarios use
checkpoint-selection identities; six use two predeclared fresh identities.
This is not sealed TEST or broad generalization evidence. No formal permit.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

from . import m2_given_contact_runtime as runtime
from . import m2_given_contact_expanded_pilot_train as training
from . import m2_given_contact_expanded_pilot_deployment as expanded
from . import m2_given_contact_pilot_deployment as bridge
from . import approved_camera
from .geometry_epoch import require_current

need=runtime.need
MODULE='sim.ausc_policy.m2_given_contact_expanded_val_scenarios'
ROUTE_KEY='m2_given_contact_expanded_val_scenarios'
INPUT_SCHEMA='m2_given_val_scenario_expert_configs_v1'
FROZEN_SCHEMA='m2_given_expanded_val_scenario_protocol_v1'
DEPLOYMENT_SCHEMA='m2_given_expanded_val_scenario_deployment_v1'
PROTOCOL_SCHEMA='m2_given_expanded_val_scenario_observation_pipe_v1'
PREPARED_SCHEMA='m2_given_expanded_val_scenario_preparation_v1'
BATCH_SCHEMA='m2_given_expanded_val_scenario_batch_v1'
CPU_EXIT_SCHEMA='m2_given_expanded_val_scenario_cpu_exit_v1'
RESOURCE_PROFILE='wam_l40s_diagnostic_eight_v1'
NATIVE_PYTHON=expanded.NATIVE_PYTHON
DEFAULT_RUN_ID='m2-given-val-scenarios-act-20261030-v1'
COUNT=30
INDEPENDENT_IDS=frozenset(('p000486','p001139'))
TARGETS={side+str(i) for side in 'LR' for i in range(1,6)}
FLAGS=dict(expanded.FLAGS,optimizer_holdout_validation=True,selection_validation_used=True,
    independent_test_claim=False,independent_sealed_test=False)


def _name(value):
    return isinstance(value,str) and bool(value) and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in value)


def validate_cases(value):
    need(value.get('schema')==INPUT_SCHEMA and value.get('complete') is True and value.get('fixed_denominator')==COUNT and
        value.get('diagnostic_only') is True and value.get('formal_acceptance') is False and
        value.get('test_payloads_opened') is False and value.get('model_outcomes_consulted') is False,
        'Explicit fixed thirty-case diagnostic VALIDATION input without model-outcome selection required')
    cases=value.get('cases',[])
    need(isinstance(cases,list) and len(cases)==COUNT and all(type(c.get('index')) is int for c in cases) and
        [c.get('index') for c in cases]==list(range(1,COUNT+1)),
        'Exactly thirty ordered validation scenarios required; no replacements')
    pairs=set();ids=set();episodes=set()
    for row in cases:
        need(row.get('split')=='validation','Only original VALIDATION scenarios permitted; no TEST')
        runtime.shared.require_membership('validation',row.get('subject_id'))
        need(row.get('target') in TARGETS and _name(row.get('scenario_id')) and _name(row.get('episode_id')) and
            type(row.get('seed')) is int and row['seed']==0,'Frozen scenario/target/seed0 required')
        pair=(row['subject_id'],row['target'])
        need(pair not in pairs and row['scenario_id'] not in ids and row['episode_id'] not in episodes,
            'Unique validation identity/target/scenario/episode required')
        pairs.add(pair);ids.add(row['scenario_id']);episodes.add(row['episode_id']);training.full_reference(row.get('config'))
        fresh=row['subject_id'] in INDEPENDENT_IDS
        expected='independent_subject_holdout' if fresh else 'checkpoint_selection_validation'
        need(row.get('validation_stratum')==expected and row.get('independent_subject_holdout') is fresh and
            row.get('model_outcomes_consulted',False) is False and (not fresh or row.get('tuning_permitted') is False),
            'Predeclared independent/selection stratum and no-outcome-selection/no-fresh-tuning boundaries required')
    subjects={row['subject_id'] for row in cases}
    need(len(subjects)==10 and all(sum(row['subject_id']==subject for row in cases)==3 for subject in subjects),
        'Exactly three scenarios on each of the ten frozen original VALIDATION identities required')
    return cases


def strata(cases,train_ids,selection_ids):
    subjects={row['subject_id'] for row in cases}
    need(not subjects&set(train_ids),'Validation identities overlap optimizer TRAIN')
    need(INDEPENDENT_IDS<=subjects and not INDEPENDENT_IDS&(set(train_ids)|set(selection_ids)),
        'Predeclared fresh identities must be absent from optimizer and checkpoint selection')
    need(subjects-INDEPENDENT_IDS<=set(selection_ids),'Selection-validation stratum differs from actual checkpoint selection scope')
    value={}
    for name,ids in [('independent_subject_holdout',INDEPENDENT_IDS),
                     ('checkpoint_selection_validation',subjects-INDEPENDENT_IDS)]:
        rows=[row for row in cases if row['subject_id'] in ids]
        value[name]=dict(subject_ids=sorted(ids),scenario_ids=[row['scenario_id'] for row in rows],
            scenario_count=len(rows),identity_count=len(ids))
    need(value['independent_subject_holdout']['scenario_count']==6 and
        value['checkpoint_selection_validation']['scenario_count']==24,'Fixed six-fresh and twenty-four selection-validation denominators required')
    value['independent_subject_holdout'].update(tuning_permitted=False,
        independent_from_optimizer_and_checkpoint_selection=True,
        limitation='Two predeclared bodies and six scenarios only; any tuning on these outcomes invalidates independence.')
    return value


def model_binding(ref):
    model,complete=expanded.model_binding(ref)
    need(complete.get('completed_steps')==20000,'Actual completed selected 20000-step expanded model required')
    return model,complete


def sources():
    value=expanded.sources();root=Path(__file__).resolve().parents[2]
    for name in (Path(__file__).name,'isaac_batch.py','approved_camera.py'):
        value['files'][name]=runtime.reference(root/'sim/ausc_policy'/name)
    value['approved_camera']=runtime.reference(approved_camera.CAMERA_FILE)
    need(value['approved_camera']['sha256']==approved_camera.CAMERA_SHA256,'Approved camera source bytes changed')
    return value


def validate_expert_config(config,case,train_ids,*,verify_sources=True):
    """Authenticate VALIDATION identity before opening its JSON plan sources."""
    need(case['subject_id'] not in train_ids,'Validation identity appeared in optimizer TRAIN')
    runtime.shared.require_membership('validation',case['subject_id']);require_current(config)
    need(config.get('experiment')=='geometry_qualification' and config.get('isaac_python')==NATIVE_PYTHON and
        config.get('physics')==dict(position_iterations=64,velocity_iterations=1,armature_kg_m2=.02) and
        Path(str(config.get('base',''))).is_absolute() and type(config.get('time_limit_seconds')) is int and
        60<config['time_limit_seconds']<=7200,'Exact current Native60/64-1-.02 bounded physics configuration required')
    forbidden=('m2_given_contact_evaluation','m2_given_contact_deployment','m2_act_deployment',
        bridge.ROUTE_KEY,expanded.ROUTE_KEY,ROUTE_KEY)
    need(not any(key in config for key in forbidden),'Expert source cannot mix learned/formal/TEST routes')
    route=config.get('m2_given_contact_control',{});meta=route.get('metadata',{})
    need(meta.get('split')=='validation' and meta.get('subject_id')==case['subject_id'] and
        meta.get('target_id')==case['target'] and meta.get('episode_id')==route.get('episode_id')==case['episode_id'],
        'Frozen expert scenario/reference identity differs')
    runtime.shared.require_membership(meta['split'],meta['subject_id']);reference=runtime.pose.require_route(meta)
    need(meta.get('handoff_policy_id')==runtime.handoff.POLICY_ID,'Original physical handoff required')
    refs=reference.snapshot()['sources'];plans=config.get('fresh_expert',{}).get('plans',[])
    need(len(plans)==1 and {k:plans[0].get(k) for k in ('file','sha256')}==refs['fresh_plan'],
        'Exact frozen fresh plan required')
    if verify_sources:
        manifest=runtime.pose.bound_json(refs['task_manifest'])
        patient=runtime.pose.bound_json(refs['patient_receipt'])
        need(patient.get('identity')==case['subject_id'] and patient.get('split')=='validation',
            'Actual source patient must be the declared original VALIDATION identity')
        targets=[row for row in manifest.get('targets',[]) if row.get('name')==case['target']]
        need(len(targets)==1,'Unique authentic prescribed target required')
        rebuilt=runtime.pose.build_given_reference(dict(targets[0],manifest_ref=refs['task_manifest']),
            refs['fresh_plan'],refs['patient_receipt'],reference.snapshot()['world_from_base'],reference.snapshot()['available_at_s'])
        need(rebuilt.sha256==reference.sha256,'Actual expert sources and full given reference differ')
    return reference


def load_protocol(ref):
    training.full_reference(ref);value=runtime.read_bound(ref)
    need(value.get('schema')==FROZEN_SCHEMA and value.get('fixed_denominator')==COUNT and
        all(value.get(k) is v for k,v in FLAGS.items()),'Frozen held-out VALIDATION protocol required')
    source=runtime.read_bound(value['expert_configs']);cases=validate_cases(source)
    model,complete=model_binding(value['model']['complete'])
    need(value.get('model')==model and value.get('cases')==cases and value.get('source_inventory')==sources() and
        value.get('train_subject_ids')==sorted(complete['scope']['train']['subject_ids']) and
        value.get('selection_validation_subject_ids')==sorted(complete['scope']['validation']['subject_ids']),
        'Frozen actual model/source/configuration list or optimizer identity scope changed')
    need(not set(value['train_subject_ids'])&{row['subject_id'] for row in cases},'Validation scenario overlaps optimizer TRAIN')
    need(value.get('strata')==strata(cases,value['train_subject_ids'],value['selection_validation_subject_ids']) and
        value.get('target_counts')=={target:sum(row['target']==target for row in cases) for target in sorted(TARGETS)} and
        value.get('missing_targets')==sorted(TARGETS-{row['target'] for row in cases}),
        'Fixed independent strata or actual target coverage changed')
    return value,complete


def config_digest(config):return runtime.digest({k:v for k,v in config.items() if k!=ROUTE_KEY})


def derive_config(config,case,*,protocol_ref,protocol,run_id):
    config=deepcopy(config);old=config.pop('m2_given_contact_control');meta=deepcopy(old['metadata'])
    episode=run_id+'-'+case['scenario_id'];meta.update(episode_id=episode,**FLAGS)
    config['fresh_expert'].pop('raw_multimodal_capture',None)
    if 'compact_scene_storage' in config:config['compact_scene_storage']['batch_id']=run_id
    config[ROUTE_KEY]=dict(schema=DEPLOYMENT_SCHEMA,model=deepcopy(protocol['model']),protocol=deepcopy(protocol_ref),
        scenario_id=case['scenario_id'],source_episode_id=case['episode_id'],episode_id=episode,metadata=meta,
        source_config=deepcopy(case['config']),source_inventory=deepcopy(protocol['source_inventory']),
        runtime_config_sha256=config_digest(config),**FLAGS)
    return config


def preflight(config,*,output=None):
    dep=config.get(ROUTE_KEY)
    need(isinstance(dep,dict) and dep.get('schema')==DEPLOYMENT_SCHEMA and all(dep.get(k) is v for k,v in FLAGS.items()),
        'Explicit held-out VALIDATION scenario deployment required')
    forbidden=('m2_given_contact_control','m2_given_contact_evaluation','m2_given_contact_deployment',
        'm2_act_deployment',bridge.ROUTE_KEY,expanded.ROUTE_KEY)
    need(not any(k in config for k in forbidden) and 'raw_multimodal_capture' not in config.get('fresh_expert',{}),
        'Validation scenario cannot mix expert/formal/TEST/old learned routes')
    meta=dep.get('metadata',{})
    need(meta.get('split')=='validation','Only original VALIDATION metadata permitted')
    runtime.shared.require_membership('validation',meta.get('subject_id'))
    protocol,complete=load_protocol(dep['protocol'])
    rows=[row for row in protocol['cases'] if row['scenario_id']==dep.get('scenario_id')]
    need(len(rows)==1,'Scenario outside frozen thirty-case protocol')
    case=rows[0]
    need(dep.get('model')==protocol['model'] and dep.get('source_inventory')==protocol['source_inventory'] and
        dep.get('source_config')==case['config'] and dep.get('source_episode_id')==case['episode_id'] and
        dep.get('runtime_config_sha256')==config_digest(config),'Actual validation model/source/scenario/configuration binding changed')
    original=runtime.read_bound(case['config']);reference=validate_expert_config(original,case,protocol['train_subject_ids'])
    expected=deepcopy(original['m2_given_contact_control']['metadata']);expected.update(episode_id=dep['episode_id'],**FLAGS)
    need(meta==expected,'Original validation given reference/metadata changed')
    projected=deepcopy(original);projected.pop('m2_given_contact_control');projected['fresh_expert'].pop('raw_multimodal_capture',None)
    if 'compact_scene_storage' in projected:projected['compact_scene_storage']['batch_id']=config['compact_scene_storage']['batch_id']
    need(projected=={k:v for k,v in config.items() if k!=ROUTE_KEY},
        'Validation learned configuration altered original geometry/camera/physics/force/parking/collision controls')
    need(meta['subject_id']==case['subject_id'] and meta['target_id']==case['target'] and meta['episode_id']==dep['episode_id'],
        'Validation case/episode differs')
    if output is not None:need(Path(output).name==case['scenario_id'],'Frozen validation output scenario identity required')
    return dict(deployment=deepcopy(dep),complete=complete,reference=reference,case=deepcopy(case),protocol=protocol)


def _batch_flags(arm):return dict(FLAGS,learned_policy=arm=='learned',expert_demonstration=arm=='expert')


def prepare(complete_file,expert_configs_file,output,*,parallelism=4,run_id=DEFAULT_RUN_ID):
    need(type(parallelism) is int and 1<=parallelism<=8,'Validation resource parallelism must be 1 through 8')
    need(_name(run_id),'Explicit canonical validation run identity required')
    output=Path(output).resolve();need(not output.exists(),'Fresh validation protocol/configuration output required')
    expert_ref=runtime.reference(expert_configs_file);cases=validate_cases(runtime.read_bound(expert_ref))
    model,complete=model_binding(runtime.reference(complete_file));train_ids=sorted(complete['scope']['train']['subject_ids'])
    need(not set(train_ids)&{row['subject_id'] for row in cases},'Validation identities must be disjoint from all optimizer TRAIN')
    selection_ids=sorted(complete['scope']['validation']['subject_ids']);declared_strata=strata(cases,train_ids,selection_ids)
    originals=[]
    for case in cases:
        config=runtime.read_bound(case['config']);validate_expert_config(config,case,train_ids);originals.append(config)
    need(len({config['base'] for config in originals})==1,'Paired validation cases require one native base')
    protocol=dict(schema=FROZEN_SCHEMA,fixed_denominator=COUNT,expert_configs=expert_ref,cases=deepcopy(cases),model=model,
        train_subject_ids=train_ids,selection_validation_subject_ids=selection_ids,strata=declared_strata,
        target_counts={target:sum(row['target']==target for row in cases) for target in sorted(TARGETS)},
        missing_targets=sorted(TARGETS-{row['target'] for row in cases}),
        source_inventory=sources(),run_id=run_id,comparison='paired actual ACT L2 versus analytic expert on unchanged source configurations',
        limitation='All thirty exclude optimizer TRAIN. Twenty-four scenarios use checkpoint-selection identities; six use two predeclared unseen identities. No sealed TEST claim; no tuning on the fresh two bodies.',**FLAGS)
    output.mkdir(parents=True,exist_ok=False);protocol_ref=runtime.write_new(output/'PROTOCOL.json',protocol)
    learned=[];expert=[]
    for case,original in zip(cases,originals):
        config=derive_config(original,case,protocol_ref=protocol_ref,protocol=protocol,run_id=run_id)
        ref=runtime.write_new(output/(case['scenario_id']+'.json'),config)
        learned.append(dict(id=case['scenario_id'],config=ref['path'],sha256=ref['sha256']))
        expert.append(dict(id=case['scenario_id'],config=case['config']['path'],sha256=case['config']['sha256']))
    batches={}
    for arm,jobs in [('learned',learned),('expert',expert)]:
        batch=dict(experiment='isaac_batch',base=originals[0]['base'],parallelism=parallelism,
            time_limit_seconds=sum(config['time_limit_seconds'] for config in originals)+300,jobs=jobs,
            validation_scenarios=dict(schema=BATCH_SCHEMA,protocol=protocol_ref,arm=arm,fixed_denominator=COUNT),**_batch_flags(arm))
        if parallelism>4:batch['resource_profile']=RESOURCE_PROFILE
        batches[arm+'_batch']=runtime.write_new(output/(arm+'-batch.json'),batch)
    return runtime.write_new(output/'PREPARED.json',dict(schema=PREPARED_SCHEMA,protocol=protocol_ref,model=model,
        fixed_denominator=COUNT,parallelism=parallelism,run_id=run_id,**batches,**FLAGS))


def validate_batch(config,jobs):
    descriptor=config.get('validation_scenarios',{});arm=descriptor.get('arm')
    need(descriptor.get('schema')==BATCH_SCHEMA and arm in ('learned','expert') and descriptor.get('fixed_denominator')==COUNT and
        all(config.get(k) is v for k,v in _batch_flags(arm).items()),'Explicit paired validation batch declaration required')
    protocol,_=load_protocol(descriptor['protocol'])
    need(len(jobs)==COUNT and [j['id'] for j in jobs]==[c['scenario_id'] for c in protocol['cases']],
        'All thirty frozen validation outcomes required; no replacements or filtering')
    for job,case in zip(jobs,protocol['cases']):
        if arm=='learned':
            binding=preflight(job['settings'])
            need(binding['case']==case and binding['deployment']['protocol']==descriptor['protocol'],
                'Learned validation worker differs from frozen protocol')
        else:
            need(job['config']==case['config']['path'] and job['sha256']==case['config']['sha256'] and
                job['settings']==runtime.read_bound(case['config']),'Paired expert must use exact unchanged source configuration')
            validate_expert_config(job['settings'],case,protocol['train_subject_ids'])
        job['validation_scenario']=dict(protocol=descriptor['protocol'],arm=arm,scenario_id=case['scenario_id'],seed=case['seed'])
    return jobs


def seed_worker(config,*,output):
    """Validate the batch-owned scenario declaration before native scene reads."""
    path=os.environ.get('AUSC_VAL_SCENARIO_PROTOCOL')
    if path is None:return None
    ref=dict(path=path,bytes=int(os.environ['AUSC_VAL_SCENARIO_PROTOCOL_BYTES']),sha256=os.environ['AUSC_VAL_SCENARIO_PROTOCOL_SHA256'])
    protocol,_=load_protocol(ref);arm=os.environ['AUSC_VAL_SCENARIO_ARM'];scenario=os.environ['AUSC_VAL_SCENARIO_ID']
    cases=[row for row in protocol['cases'] if row['scenario_id']==scenario]
    need(len(cases)==1 and arm in ('learned','expert'),'Batch-owned frozen validation worker identity required')
    case=cases[0]
    need(os.environ.get('AUSC_OWNED_RUN')=='1' and os.environ.get('AUSC_VAL_SCENARIO_SEED')==str(case['seed']) and
        Path(output).name==scenario,'Exact owned scenario seed/output required')
    if arm=='learned':need(preflight(config,output=output)['deployment']['protocol']==ref,'Worker learned protocol differs')
    else:
        need(config==runtime.read_bound(case['config']),'Worker expert configuration differs from frozen pair')
        validate_expert_config(config,case,protocol['train_subject_ids'])
    random.seed(case['seed']);np.random.seed(case['seed'])
    return dict(protocol=ref,scenario_id=scenario,arm=arm,seed=case['seed'])


def _binding(dep):return dict(scenario_protocol_sha256=dep['protocol']['sha256'],scenario_id=dep['scenario_id'])
def make_request(dep,*args,**kwargs):return bridge.make_request(dep,*args,protocol_schema=PROTOCOL_SCHEMA,binding_extra=_binding(dep),**kwargs)
def validate_request(request,dep,**kwargs):return bridge.validate_request(request,dep,protocol_schema=PROTOCOL_SCHEMA,binding_extra=_binding(dep),**kwargs)
def response_for(*args,**kwargs):return bridge.response_for(*args,protocol_schema=PROTOCOL_SCHEMA,**kwargs)
def decode_response(*args,**kwargs):return bridge.decode_response(*args,protocol_schema=PROTOCOL_SCHEMA,checkpoint_schema=training.PILOT_SCHEMA,**kwargs)


def load_model(dep):
    protocol,complete=load_protocol(dep['protocol']);meta=dep.get('metadata',{})
    need(meta.get('split')=='validation','CPU child only accepts original VALIDATION metadata')
    runtime.shared.require_membership('validation',meta.get('subject_id'))
    rows=[row for row in protocol['cases'] if row['scenario_id']==dep.get('scenario_id')]
    need(dep['model']==protocol['model'] and len(rows)==1,'Actual selected validation model or frozen scenario differs')
    case=rows[0];original=runtime.read_bound(case['config'])
    validate_expert_config(original,case,protocol['train_subject_ids'])
    expected=deepcopy(original['m2_given_contact_control']['metadata']);expected.update(episode_id=dep['episode_id'],**FLAGS)
    need(meta==expected and dep.get('source_config')==case['config'] and dep.get('source_episode_id')==case['episode_id'],
        'CPU child frozen validation case/source/reference differs')
    return training.load_selected_policy(complete)


def serve(deployment_ref):
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
        'Owned official validation CPU child must hide CUDA')
    dep=runtime.read_bound(deployment_ref)
    need(dep.get('schema')==DEPLOYMENT_SCHEMA and dep.get('source_inventory')==sources(),'Validation CPU source binding changed')
    with redirect_stdout(sys.stderr):torch,policy,checkpoint=load_model(dep)
    def send(value):print(json.dumps(value,allow_nan=False,separators=(',',':')),flush=True)
    send(dict(schema=PROTOCOL_SCHEMA,ready=True,deployment_sha256=deployment_ref['sha256'],checkpoint=dep['model']['checkpoint'],
        pid=os.getpid(),parent_pid=os.getppid(),source_inventory=sources(),execution=runtime.execution_settings(torch,policy,device='cpu')))
    sequence=0;previous=None
    while True:
        raw=sys.stdin.buffer.readline(32769)
        if not raw:return
        need(len(raw)<=32768 and raw.endswith(b'\n'),'Bounded complete validation request required')
        request=bridge.wire_decode(raw);sequence+=1;validate_request(request,dep,sequence=sequence,previous=previous)
        history=np.asarray(request['history'],dtype=float)
        with redirect_stdout(sys.stderr):
            relative=runtime.infer(torch,policy,dict(states=history[None,1],histories=history[None]),checkpoint['normalization'],device='cpu')[0]
        send(response_for(request,relative));previous=request


class ValidationScenarioACT(expanded.ExpandedGivenACT):
    def _tick(self,*args,**kwargs):
        start=len(self.env.rows)
        try:return super()._tick(*args,**kwargs)
        finally:
            for row in self.env.rows[start:]:row.update(FLAGS)


PilotGivenACT=ValidationScenarioACT


def make_controller(*args,**kwargs):
    controller=bridge.make_controller(*args,_adapter=sys.modules[__name__],**kwargs)
    controller.policy_receipt.update(FLAGS,validation_protocol=kwargs['deployment']['protocol'],
        scenario_id=kwargs['deployment']['scenario_id'])
    return controller


def tag_result(result,deployment,controller=None):
    bridge.tag_result(result,deployment,controller)
    for row in [result,*result.get('points',[]),*result.get('l2_queries',[]),*result.get('l2_actions',[]),*result.get('l2_handoffs',[])]:row.update(FLAGS)
    result.update(validation_protocol=deepcopy(deployment.get('protocol')),scenario_id=deployment.get('scenario_id'),
        executed_controller_class=MODULE+'.ValidationScenarioACT',controller_source='held-out VALIDATION ACT L2 + original analytic L3',
        scope='Optimizer-disjoint original VALIDATION; twenty-four selection scenarios plus six predeclared independent scenarios on two bodies. No sealed TEST claim.')
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
            failure_reason=str(error),errors=[dict(stage='heldout_validation_native',error=repr(error))],
            runtime_sources_sha256=lifecycle.get('source_hashes',{}))
        tag_result(result,deployment,lifecycle.get('controller'));path=Path(output)/'fresh_expert_result.json'
        if not path.exists():runtime.write_new(path,result)
        if not isinstance(error,Exception):raise
        return result


def summarize(prepared_file,learned_output,expert_output,output):
    prepared=runtime.read_bound(runtime.reference(prepared_file))
    need(prepared.get('schema')==PREPARED_SCHEMA and prepared.get('fixed_denominator')==COUNT,'Actual validation preparation required')
    protocol,_=load_protocol(prepared['protocol']);outcomes=[]
    batches={arm:runtime.read_bound(prepared[arm+'_batch']) for arm in ('learned','expert')}
    job_hashes={arm:{row['id']:row['sha256'] for row in batch['jobs']} for arm,batch in batches.items()}
    for case in protocol['cases']:
        row=dict(index=case['index'],scenario_id=case['scenario_id'],subject_id=case['subject_id'],target=case['target'],seed=case['seed'],
            validation_stratum=case['validation_stratum'],independent_subject_holdout=case['independent_subject_holdout'])
        for arm,folder in [('learned',learned_output),('expert',expert_output)]:
            root=Path(folder);report=root/case['scenario_id']/'fresh_expert_result.json';child=root/(case['scenario_id']+'.result.json')
            result=json.loads(report.read_bytes()) if report.exists() else {};worker=json.loads(child.read_bytes()) if child.exists() else {}
            attributed=(worker.get('config_sha256')==job_hashes[arm].get(case['scenario_id']) and
                (arm=='expert' or (result.get('learned_policy') is True and result.get('validation_protocol')==prepared['protocol'] and
                result.get('scenario_id')==case['scenario_id'] and result.get('expert_l2_fallback') is False))
            )
            executed=bool(attributed and result.get('physics_steps',0)>0 and
                (arm=='expert' or result.get('diagnostic_closed_loop_evaluated') is True))
            row[arm]=dict(worker_recorded=bool(worker),exit_code=worker.get('exit_code'),physics_steps=result.get('physics_steps',0),
                actual_episode=executed,evidence_matches_frozen_worker=attributed,
                physical_success=bool(executed and worker.get('exit_code')==0 and result.get('success') is True and
                    result.get('physical_validation') is True and not result.get('errors')),
                failure_reason=result.get('failure_reason','Missing native result' if not result else None),
                report=runtime.reference(report) if report.exists() else None,
                worker=runtime.reference(child) if child.exists() else None)
        outcomes.append(row)
    stratified={}
    for name,definition in protocol['strata'].items():
        rows=[row for row in outcomes if row['validation_stratum']==name]
        stratified[name]=dict(definition,**{arm:{'actual_episodes':sum(row[arm]['actual_episode'] for row in rows),
            'physical_successes':sum(row[arm]['physical_success'] for row in rows)} for arm in ('learned','expert')})
    return runtime.write_new(output,dict(schema='m2_given_expanded_val_scenario_outcomes_v1',protocol=prepared['protocol'],
        strata=stratified,target_counts=protocol['target_counts'],missing_targets=protocol['missing_targets'],
        fixed_denominator=COUNT,outcomes=outcomes,actual_learned_episodes=sum(row['learned']['actual_episode'] for row in outcomes),
        actual_expert_episodes=sum(row['expert']['actual_episode'] for row in outcomes),
        learned_physical_successes=sum(row['learned']['physical_success'] for row in outcomes),
        expert_physical_successes=sum(row['expert']['physical_success'] for row in outcomes),**FLAGS))


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','serve','summarize'):modes.add_argument('--'+mode,action='store_true')
    for name in ('complete','expert-configs','output','deployment','deployment-sha256','prepared','learned-output','expert-output'):
        parser.add_argument('--'+name)
    parser.add_argument('--parallelism',type=int,default=4);parser.add_argument('--run-id',default=DEFAULT_RUN_ID);args=parser.parse_args()
    if args.serve:
        need(args.deployment and args.deployment_sha256,'Bound validation CPU deployment required')
        serve(dict(path=str(Path(args.deployment).resolve()),sha256=args.deployment_sha256))
    elif args.summarize:
        need(args.prepared and args.learned_output and args.expert_output and args.output,'Both native output folders and preparation required')
        print(json.dumps(summarize(args.prepared,args.learned_output,args.expert_output,args.output)),flush=True)
    else:
        need(args.complete and args.expert_configs and args.output,'Actual selected model, thirty expert configs and fresh output required')
        print(json.dumps(prepare(args.complete,args.expert_configs,args.output,parallelism=args.parallelism,run_id=args.run_id)),flush=True)


if __name__=='__main__':main()
