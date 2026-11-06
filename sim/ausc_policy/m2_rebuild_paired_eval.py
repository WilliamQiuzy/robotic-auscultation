"""Opt-in rebuilt matched-scene ACT/expert/Astra diagnostic, never a native owner.

Freeze explicit source configs before outcomes; keep all planned failures. This
module prepares ordinary isaac_batch jobs and reuses original local execution,
CPU admission and complete 240 Hz physical evidence. No model call in prepare.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
from types import MethodType

import numpy as np

from . import m2_given_contact_runtime as runtime
from . import m2_given_contact_expanded_pilot_deployment as expanded
from . import m2_given_contact_expanded_pilot_train as training
from . import m2_given_contact_pilot_deployment as bridge
from . import m2_given_contact_pilot_native_audit as audit
from . import m2_given_contact_evaluation as neutral
from .geometry_epoch import require_current

need=runtime.need
MODULE='sim.ausc_policy.m2_rebuild_paired_eval'
ROUTE_KEY='m2_rebuild_paired_eval'
DESCRIPTOR='matched_scene_eval'
INPUT_SCHEMA='m2_rebuild_paired_eval_cases_v1'
PROTOCOL_SCHEMA='m2_rebuild_paired_eval_observation_pipe_v1'
FROZEN_SCHEMA='m2_rebuild_paired_eval_protocol_v1'
DEPLOYMENT_SCHEMA='m2_rebuild_paired_eval_act_deployment_v1'
PREPARED_SCHEMA='m2_rebuild_paired_eval_preparation_v1'
BATCH_SCHEMA='m2_rebuild_paired_eval_batch_v1'
QUALIFICATION_SCHEMA='m2_rebuild_paired_eval_native_qualification_v1'
CPU_EXIT_SCHEMA='m2_rebuild_paired_eval_cpu_exit_v1'
NATIVE_PYTHON=expanded.NATIVE_PYTHON
MAX_CASE_SECONDS=9000  # Preserve the original prepared source budget in every arm.
NATIVE_TRACE_FILENAME='given_control_native.jsonl.xz'  # Exact full JSON, inside the unchanged 100 MB job cap.
ARMS=('expert','act','astra')
COHORTS=dict(train=('p001120','p000758','p001031','p000833'),selection_validation=('p000923',),
    frozen_evaluation=('p001030','p000956'))
POSES=('p000474_a001230_f0000','p000476_a001231_f0023')
TARGETS={side+str(i) for side in 'LR' for i in range(1,6)}
FLAGS=dict({k:v for k,v in expanded.FLAGS.items() if k not in ('learned_policy','expert_demonstration')},
    training_eligible=False,training_dataset_eligible=False,
    independent_test_claim=False,independent_sealed_test=False)
DEFAULT_ASTRA=dict(schema='ausc_astra_given_contact_supervisor_v1',model='gpt-6-astra',
    endpoint='http://127.0.0.1:8123/decision',token_env='AUSC_ASTRA_RUN_TOKEN',timeout_s=150.)


def _name(value):
    return isinstance(value,str) and 0<len(value)<=80 and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in value)


def validate_cases(value):
    need(value.get('schema')==INPUT_SCHEMA and value.get('complete') is True and
        value.get('model_outcomes_consulted') is False,'Frozen explicit new cases without outcome selection required')
    targets=value.get('target_ids',[]);cohorts=value.get('eval_cohorts',['frozen_evaluation'])
    need(isinstance(targets,list) and 2<=len(targets)<=4 and len(set(targets))==len(targets) and
        set(targets)<=TARGETS,'Explicit root-chosen two through four targets required')
    need(value.get('pose_ids')==list(POSES) and isinstance(cohorts,list) and cohorts and
        len(set(cohorts))==len(cohorts) and set(cohorts)<=set(COHORTS),'Exact two observed poses and explicit evaluation cohorts required')
    expected={(subject,pose,target) for cohort in cohorts for subject in COHORTS[cohort] for pose in POSES for target in targets}
    cases=value.get('cases',[])
    need(isinstance(cases,list) and type(value.get('fixed_denominator')) is int and
        value['fixed_denominator']==len(expected)==len(cases),'Every planned evaluation case required; no skip or replacement')
    keys=set();names=set();episodes=set()
    for index,case in enumerate(cases,1):
        cohort=case.get('cohort');subject=case.get('subject_id');split='train' if cohort=='train' else 'validation'
        need(cohort in cohorts and subject in COHORTS[cohort] and case.get('split')==split,
            'Actual declared cohort and original identity split required')
        runtime.shared.require_membership(split,subject)
        key=(subject,case.get('pose_id'),case.get('target_id'))
        need(type(case.get('index')) is int and case['index']==index and key in expected and key not in keys and
            _name(case.get('scenario_id')) and case['scenario_id'] not in names and _name(case.get('episode_id')) and
            case['episode_id'] not in episodes and type(case.get('seed')) is int and case['seed']==0,
            'Unique ordered explicit scenario/pose/target/source episode and seed zero required')
        keys.add(key);names.add(case['scenario_id']);episodes.add(case['episode_id'])
        if case.get('preparation_status')=='prepared':training.full_reference(case.get('config'))
        else:need(case.get('config') is None and case.get('preparation_status') in ('failed','unavailable') and
            isinstance(case.get('reason'),str) and bool(case['reason']),'Unavailable planned case needs explicit retained reason')
    need(keys==expected,'Planned cohort/pose/target Cartesian population differs')
    return cases


def model_binding(ref):
    model,complete=expanded.model_binding(ref)  # Original selected weights + actual CPU equivalence and source admission.
    scope=complete['scope']
    need(set(scope['train']['subject_ids'])==set(COHORTS['train']) and
        set(scope['validation']['subject_ids'])==set(COHORTS['selection_validation']),
        'New checkpoint TRAIN4 and selection-VAL1 scope required; frozen evaluation cannot enter selection')
    need(not set(COHORTS['frozen_evaluation'])&(set(scope['train']['subject_ids'])|set(scope['validation']['subject_ids'])),
        'Frozen evaluation identities entered optimizer or checkpoint selection')
    return model,complete


def sources():
    value=expanded.sources()
    from . import astra_native,astra_supervisor,astra_sensors,approved_camera
    for module in (sys.modules[__name__],astra_native,astra_supervisor,astra_sensors,approved_camera):
        path=Path(module.__file__);value['files'][path.name]=runtime.reference(path)
    value['approved_camera']=runtime.reference(approved_camera.CAMERA_FILE)
    need(value['approved_camera']['sha256']==approved_camera.CAMERA_SHA256,'Original approved camera bytes changed')
    return value


def validate_expert_config(config,case):
    """Authenticate exact source identity, observed pose and prescribed full pose."""
    require_current(config);split=case['split'];subject=case['subject_id']
    runtime.shared.require_membership(split,subject)
    forbidden=('astra_supervisor',DESCRIPTOR,ROUTE_KEY,'m2_given_contact_evaluation','m2_given_contact_deployment',
        'm2_act_deployment',bridge.ROUTE_KEY,expanded.ROUTE_KEY,'m2_given_contact_expanded_val_scenarios')
    need(not any(key in config for key in forbidden),'Expert source cannot contain another deployment or supervisor')
    need(config.get('experiment')=='geometry_qualification' and config.get('isaac_python')==NATIVE_PYTHON and
        config.get('physics')==dict(position_iterations=64,velocity_iterations=1,armature_kg_m2=.02) and
        Path(str(config.get('base',''))).is_absolute() and type(config.get('time_limit_seconds')) is int and
        60<config['time_limit_seconds']<=MAX_CASE_SECONDS,'Original bounded Native60 64/1/.02 physics configuration required')
    route=config.get('m2_given_contact_control',{});meta=route.get('metadata',{})
    need(meta.get('subject_id')==subject and meta.get('split')==split and meta.get('target_id')==case['target_id'] and
        route.get('episode_id')==meta.get('episode_id')==case['episode_id'] and
        route.get('handoff_policy_id')==meta.get('handoff_policy_id')==runtime.handoff.POLICY_ID,
        'Explicit source case/reference/strict parking identity differs')
    reference=runtime.pose.require_route(meta);reference.verify_sources();refs=reference.snapshot()['sources']
    need(config.get('fresh_expert',{}).get('plans')==[refs['fresh_plan']],'Exact prescribed checked fresh plan required')
    patient=runtime.pose.bound_json(refs['patient_receipt']);manifest=runtime.pose.bound_json(refs['task_manifest'])
    need(patient.get('identity')==subject and patient.get('split')==split and
        patient.get('proposal')==subject+'_'+case['pose_id'],'Bound observed patient body/pose differs from scenario')
    targets=[row for row in manifest.get('targets',[]) if row.get('name')==case['target_id']]
    need(len(targets)==1,'Unique source prescribed target required')
    rebuilt=runtime.pose.build_given_reference(dict(targets[0],manifest_ref=refs['task_manifest']),
        refs['fresh_plan'],refs['patient_receipt'],reference.snapshot()['world_from_base'],reference.snapshot()['available_at_s'])
    need(rebuilt.sha256==reference.sha256,'Full prescribed reference changed')
    return reference


def _strata(cases):
    return {cohort:dict(subject_ids=list(subjects),scenario_count=sum(c['cohort']==cohort for c in cases),
        tuning_permitted=cohort!='frozen_evaluation',independent_from_optimizer_and_checkpoint_selection=cohort=='frozen_evaluation')
        for cohort,subjects in COHORTS.items()}


def load_protocol(ref):
    training.full_reference(ref);value=runtime.read_bound(ref)
    need(value.get('schema')==FROZEN_SCHEMA and all(value.get(k) is v for k,v in FLAGS.items()),'New frozen matched protocol required')
    source=runtime.read_bound(value['expert_configs']);cases=validate_cases(source)
    model,complete=model_binding(value['model']['complete'])
    need(value.get('cases')==cases and value.get('model')==model and value.get('checkpoint_scope')==complete['scope'] and
        value.get('cohorts')=={k:list(v) for k,v in COHORTS.items()} and value.get('source_inventory')==sources() and
        value.get('fixed_denominator')==len(cases) and value.get('planned_arm_denominator')==len(cases)*3 and
        value.get('strata')==_strata(cases) and value.get('eval_cohorts')==source.get('eval_cohorts',['frozen_evaluation']) and
        value.get('target_ids')==source['target_ids'] and value.get('pose_ids')==list(POSES),
        'Frozen cases/model/source/actual checkpoint scope changed')
    limits=value.get('arm_parallelism',{})
    need(set(limits)==set(ARMS) and all(type(n) is int and 1<=n<=4 for n in limits.values()),
        'Explicit frozen per-arm parallelism required')
    return value,complete


def _episode(protocol,case,arm):return protocol['run_id']+'-'+case['scenario_id']+'-'+arm


def derive_config(original,case,protocol_ref,protocol,arm):
    config=deepcopy(original);old=config.pop('m2_given_contact_control');meta=deepcopy(old['metadata'])
    episode=_episode(protocol,case,arm);need(len(episode)<=128,'Bounded separate arm episode id required')
    meta.update(episode_id=episode,learned_policy=arm=='act',expert_demonstration=False,**FLAGS);config['fresh_expert'].pop('raw_multimodal_capture',None)
    descriptor=dict(protocol=deepcopy(protocol_ref),scenario_id=case['scenario_id'],arm=arm,seed=case['seed'],
        learned_policy=arm=='act',expert_demonstration=False,**FLAGS)
    config[DESCRIPTOR]=descriptor
    if arm=='act':
        config[ROUTE_KEY]=dict(schema=DEPLOYMENT_SCHEMA,model=deepcopy(protocol['model']),protocol=deepcopy(protocol_ref),
            scenario_id=case['scenario_id'],source_episode_id=case['episode_id'],episode_id=episode,metadata=meta,
            source_config=deepcopy(case['config']),source_inventory=deepcopy(protocol['source_inventory']),
            learned_policy=True,expert_demonstration=False,**FLAGS)
        config[ROUTE_KEY]['runtime_config_sha256']=runtime.digest({k:v for k,v in config.items() if k!=ROUTE_KEY})
    else:
        config['m2_given_contact_control']=dict(old,metadata=meta,episode_id=episode)
        if arm=='astra':config['astra_supervisor']=deepcopy(protocol['astra_supervisor'])
    return config


def preflight_arm(config,*,output=None):
    descriptor=config.get(DESCRIPTOR,{})
    need(isinstance(descriptor,dict) and descriptor.get('arm') in ARMS and
        all(descriptor.get(k) is v for k,v in FLAGS.items()),'Explicit matched-scene arm declaration required')
    protocol,complete=load_protocol(descriptor['protocol']);arm=descriptor['arm']
    cases=[c for c in protocol['cases'] if c['scenario_id']==descriptor.get('scenario_id')]
    need(len(cases)==1 and cases[0]['preparation_status']=='prepared','Explicit prepared frozen scenario required')
    case=cases[0];original=runtime.read_bound(case['config']);reference=validate_expert_config(original,case)
    need(config==derive_config(original,case,descriptor['protocol'],protocol,arm),
        'Matched arm changed source scene/reference/home/physics/camera/control or actual model binding')
    if output is not None:need(Path(output).name==case['scenario_id'],'Exact frozen scenario output required')
    dep=config.get(ROUTE_KEY)
    return dict(arm=arm,case=deepcopy(case),protocol=protocol,complete=complete,reference=reference,
        deployment=deepcopy(dep),metadata=deepcopy(config[ROUTE_KEY if arm=='act' else 'm2_given_contact_control']['metadata']))


def preflight(config,*,output=None):
    binding=preflight_arm(config,output=output);need(binding['arm']=='act','Only actual ACT may use learned adapter')
    return binding


def prepare(complete_file,expert_configs_file,output,*,astra_settings,parallelism=1,astra_parallelism=1,run_id='m2-rebuild-paired-20261105-v1'):
    need(type(parallelism) is int and 1<=parallelism<=4,'Bounded one through four worker parallelism required')
    need(type(astra_parallelism) is int and 1<=astra_parallelism<=4,'Bounded one through four Astra worker parallelism required')
    arm_parallelism=dict(expert=parallelism,act=parallelism,astra=astra_parallelism)
    need(_name(run_id),'Explicit canonical run id required');output=Path(output).resolve()
    need(not output.exists(),'Fresh matched protocol output required')
    expert_ref=runtime.reference(expert_configs_file);source=runtime.read_bound(expert_ref);cases=validate_cases(source)
    model,complete=model_binding(runtime.reference(complete_file));originals={}
    need(astra_settings==dict(DEFAULT_ASTRA,**astra_settings) and set(astra_settings)==set(DEFAULT_ASTRA) and
        astra_settings['model']=='gpt-6-astra' and astra_settings['schema']==DEFAULT_ASTRA['schema'],
        'Explicit pinned Astra supervisor settings required')
    from .astra_supervisor import HTTPClient
    need(isinstance(astra_settings['token_env'],str) and astra_settings['token_env'].startswith('AUSC_ASTRA_'),
        'Explicit Astra token environment name required')
    HTTPClient(astra_settings['endpoint'],'prepare-validation-only',astra_settings['timeout_s'])
    for case in cases:
        if case['preparation_status']!='prepared':continue
        config=runtime.read_bound(case['config']);validate_expert_config(config,case);originals[case['scenario_id']]=config
    need(originals and len({c['base'] for c in originals.values()})==1,'At least one explicitly prepared case and one native base required')
    protocol=dict(schema=FROZEN_SCHEMA,expert_configs=expert_ref,cases=cases,model=model,checkpoint_scope=deepcopy(complete['scope']),
        cohorts={k:list(v) for k,v in COHORTS.items()},eval_cohorts=source.get('eval_cohorts',['frozen_evaluation']),
        target_ids=source['target_ids'],pose_ids=list(POSES),fixed_denominator=len(cases),planned_arm_denominator=3*len(cases),
        strata=_strata(cases),arm_parallelism=arm_parallelism,source_inventory=sources(),astra_supervisor=deepcopy(astra_settings),run_id=run_id,
        claim='Bounded simulation diagnostic; frozen identities untouched by optimization or checkpoint selection. No sealed TEST, learned Astra motor, clinical or training-data claim.',**FLAGS)
    output.mkdir(parents=True,exist_ok=False);protocol_ref=runtime.write_new(output/'PROTOCOL.json',protocol);batches={}
    for arm in ARMS:
        jobs=[]
        for case in cases:
            if case['preparation_status']!='prepared':continue
            config=derive_config(originals[case['scenario_id']],case,protocol_ref,protocol,arm)
            ref=runtime.write_new(output/(case['scenario_id']+'-'+arm+'.json'),config)
            jobs.append(dict(id=case['scenario_id'],config=ref['path'],sha256=ref['sha256']))
        batch=dict(experiment='isaac_batch',base=next(iter(originals.values()))['base'],parallelism=arm_parallelism[arm],
            time_limit_seconds=sum(c['time_limit_seconds'] for c in originals.values())+300,jobs=jobs,
            matched_scene_evaluation=dict(schema=BATCH_SCHEMA,protocol=protocol_ref,arm=arm,fixed_denominator=len(cases)),**FLAGS)
        batches[arm]=runtime.write_new(output/(arm+'-batch.json'),batch)
    return runtime.write_new(output/'PREPARED.json',dict(schema=PREPARED_SCHEMA,protocol=protocol_ref,batches=batches,
        fixed_denominator=len(cases),planned_arm_denominator=3*len(cases),model=model,arm_parallelism=arm_parallelism,**FLAGS))


def validate_batch(config,jobs):
    descriptor=config['matched_scene_evaluation'];protocol,_=load_protocol(descriptor['protocol']);arm=descriptor.get('arm')
    need(descriptor.get('schema')==BATCH_SCHEMA and arm in ARMS and descriptor.get('fixed_denominator')==protocol['fixed_denominator'] and
        all(config.get(k) is v for k,v in FLAGS.items()),'Explicit guarded matched batch required')
    need(config.get('parallelism')==protocol['arm_parallelism'][arm],'Batch parallelism differs from frozen arm resource binding')
    cases=[c for c in protocol['cases'] if c['preparation_status']=='prepared']
    need([job['id'] for job in jobs]==[c['scenario_id'] for c in cases],'Every available planned case required; no filtering or replacement')
    for job,case in zip(jobs,cases):
        binding=preflight_arm(job['settings']);need(binding['case']==case and binding['arm']==arm and
            job['settings'][DESCRIPTOR]['protocol']==descriptor['protocol'],'Batch/arm/frozen protocol differs')
    return jobs


def seed_worker(config,*,output):
    binding=preflight_arm(config,output=output)
    need(os.environ.get('AUSC_OWNED_RUN')=='1','Original guard-owned native worker required')
    random.seed(binding['case']['seed']);np.random.seed(binding['case']['seed'])
    return dict(protocol=config[DESCRIPTOR]['protocol'],scenario_id=binding['case']['scenario_id'],arm=binding['arm'],seed=0)


def _binding(dep):return dict(scenario_protocol_sha256=dep['protocol']['sha256'],scenario_id=dep['scenario_id'])
def make_request(dep,*args,**kwargs):return bridge.make_request(dep,*args,protocol_schema=PROTOCOL_SCHEMA,binding_extra=_binding(dep),**kwargs)
def validate_request(request,dep,**kwargs):return bridge.validate_request(request,dep,protocol_schema=PROTOCOL_SCHEMA,binding_extra=_binding(dep),**kwargs)
def response_for(*args,**kwargs):return bridge.response_for(*args,protocol_schema=PROTOCOL_SCHEMA,**kwargs)
def decode_response(*args,**kwargs):return bridge.decode_response(*args,protocol_schema=PROTOCOL_SCHEMA,checkpoint_schema=training.PILOT_SCHEMA,**kwargs)


def load_model(dep):
    protocol,complete=load_protocol(dep['protocol'])
    cases=[c for c in protocol['cases'] if c['scenario_id']==dep.get('scenario_id')]
    need(len(cases)==1 and dep['model']==protocol['model'],'Actual selected policy/frozen scenario differs')
    case=cases[0];original=runtime.read_bound(case['config']);validate_expert_config(original,case)
    expected=derive_config(original,case,dep['protocol'],protocol,'act')[ROUTE_KEY]
    need(dep==expected,'CPU child exact scenario/source/reference/model differs')
    return training.load_selected_policy(complete)


def serve(deployment_ref):
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
        'Owned official CPU child must hide CUDA')
    dep=runtime.read_bound(deployment_ref);need(dep.get('source_inventory')==sources(),'CPU source binding changed')
    with redirect_stdout(sys.stderr):torch,policy,checkpoint=load_model(dep)
    def send(value):print(json.dumps(value,allow_nan=False,separators=(',',':')),flush=True)
    send(dict(schema=PROTOCOL_SCHEMA,ready=True,deployment_sha256=deployment_ref['sha256'],checkpoint=dep['model']['checkpoint'],
        pid=os.getpid(),parent_pid=os.getppid(),source_inventory=sources(),execution=runtime.execution_settings(torch,policy,device='cpu')))
    sequence=0;previous=None
    while True:
        raw=sys.stdin.buffer.readline(32769)
        if not raw:return
        need(len(raw)<=32768 and raw.endswith(b'\n'),'Bounded complete ACT request required')
        request=bridge.wire_decode(raw);sequence+=1;validate_request(request,dep,sequence=sequence,previous=previous)
        history=np.asarray(request['history'],dtype=float)
        with redirect_stdout(sys.stderr):
            relative=runtime.infer(torch,policy,dict(states=history[None,1],histories=history[None]),checkpoint['normalization'],device='cpu')[0]
        send(response_for(request,relative));previous=request


PilotGivenACT=expanded.ExpandedGivenACT


def make_controller(*args,**kwargs):return bridge.make_controller(*args,_adapter=sys.modules[__name__],**kwargs)


def tag_result(result,deployment,controller=None):
    bridge.tag_result(result,deployment,controller)
    result.update(matched_scene_protocol=deepcopy(deployment.get('protocol')),scenario_id=deployment.get('scenario_id'),
        executed_controller_class=(None if controller is None else type(controller).__module__+'.'+type(controller).__name__),
        controller_source='Actual newly selected ACT L2 + original analytic local contact control',
        scope='Bounded rebuilt matched-scene diagnostic on the explicit frozen protocol; no sealed TEST or generalization claim.',**FLAGS)
    return result


def run_native_probe(app,stage,output,config,mass):
    from . import fresh_expert
    lifecycle={};dep=config.get(ROUTE_KEY,{})
    try:
        preflight(config,output=output)
        return fresh_expert._run_probe(app,stage,output,config,mass,_learned_lifecycle=lifecycle)
    except BaseException as error:
        result=dict(success=False,physical_validation=False,diagnostic_completed=False,points=[],
            physics_steps=len(lifecycle['env'].rows) if lifecycle.get('env') is not None else 0,
            failure_reason=str(error),errors=[dict(stage='matched_act_native',error=repr(error))])
        tag_result(result,dep,lifecycle.get('controller'));path=Path(output)/'fresh_expert_result.json'
        if not path.exists():runtime.write_new(path,result)
        if not isinstance(error,Exception):raise
        return result


class MatchedAnalyticAudit(audit.BufferedNativeAudit):
    """Original neutral tick observer with honest expert/Astra controller binding."""
    def __init__(self,path,controller,config,binding):
        self.path=Path(path);self.stream=audit.BufferedGzipText(path);self.metadata=deepcopy(controller.metadata)
        self.rows=0;self.errors=[];self.receipt=None;self.controller=controller;self.adapter=sys.modules[__name__]
        self.sources=sources();self.original_class=type(controller)
        self.binding=dict(configuration_sha256=runtime.digest(config),model=None,arm=binding['arm'],
            supervisor_model='gpt-6-astra' if binding['arm']=='astra' else None,local_motor_assistance=binding['arm']=='astra',
            episode_id=controller.episode_id,source_config=deepcopy(binding['case']['config']),
            controller_class=type(controller).__module__+'.'+type(controller).__name__,
            controller_source=runtime.reference(inspect.getfile(type(controller))),
            task_contract=runtime.task_contract(),flush_rows=audit.FLUSH_ROWS,full_native_row=True,**audit.FLAGS)

    def _identity(self,controller):
        need(controller is self.controller and type(controller) is self.original_class and controller.metadata==self.metadata and
            controller.episode_id==self.binding['episode_id'] and controller.reference.sha256==self.metadata['reference_sha256'],
            'Matched analytic observer actual controller/metadata/reference changed')


def attach_analytic(controller,config,output):
    from .m2_given_contact_controller import GivenContactExpert
    from .astra_native import AstraGivenContactExpert
    binding=preflight_arm(config,output=output);arm=binding['arm']
    expected=AstraGivenContactExpert if arm=='astra' else GivenContactExpert
    need(arm in ('expert','astra') and type(controller) is expected and controller.metadata==binding['metadata'] and
        not hasattr(controller,'_neutral_native_audit'),'Exact honest analytic controller and one neutral observer required')
    observer=MatchedAnalyticAudit(Path(output)/NATIVE_TRACE_FILENAME,controller,config,binding);original=controller._tick
    def tick(self,command,subphase,*args,**kwargs):
        return observer.tick(self,command,subphase,lambda:original(command,subphase,*args,**kwargs))
    controller._neutral_native_audit=observer;controller._pilot_light_native_audit=observer
    controller._neutral_original_tick=original;controller._tick=MethodType(tick,controller)
    return observer


def _astra_evidence(output,run,events,meta):
    from . import astra_supervisor as supervisor,astra_sensors
    rows=run.get('astra_decisions',[])
    need(run.get('learned_policy') is False and run.get('supervisor_model')=='gpt-6-astra' and
        run.get('motor_actor')=='analytic_GivenContactExpert' and run.get('supervisor_inference') is True and
        run.get('local_motor_assistance') is True and run.get('executed_controller_class')==
        'sim.ausc_policy.astra_native.AstraGivenContactExpert' and len(rows)==3,
        'Actual Astra inference with explicit analytic local motor assistance required')
    sensor_dir=Path(output)/'astra/sensors';teardown=runtime.read_bound(runtime.reference(sensor_dir/'teardown.json'))
    manifest_ref=runtime.reference(sensor_dir/'manifest.json');manifest=runtime.read_bound(manifest_ref)
    need(teardown.get('completed') is True and teardown.get('failures')==[] and teardown.get('frames')==3 and
        manifest.get('schema')==astra_sensors.SCHEMA and manifest.get('hall_source')=='simulated_adc' and
        manifest.get('force_label_used') is False,'Actual RGB-D/Hall observer and completed teardown required')
    for path,sha in manifest['source_sha256'].items():need(runtime.reference(path)['sha256']==sha,'Astra sensor source changed')
    need(runtime.reference(sensor_dir/'calibration.json')['sha256']==astra_sensors.CALIBRATION_SHA256 and
        runtime.reference(sensor_dir/'approved_camera.json')['sha256']==astra_sensors.APPROVED_CAMERA_SHA256,
        'Actual pinned camera/calibration bytes changed')
    ranges=[]
    for index,(row,boundary,skill) in enumerate(zip(rows,('HOME','PARKED','RELEASED'),
            ('approach_target','continue_local_control','retract'))):
        folder=Path(output)/'astra'/f'{index:04d}'
        record=runtime.read_bound(runtime.reference(folder/'record.json'))
        observation_ref=runtime.reference(folder/'observation.json');obs=runtime.read_bound(observation_ref)
        request_ref=runtime.reference(folder/'request.json');request=runtime.read_bound(request_ref)
        proposal=runtime.read_bound(runtime.reference(folder/'proposal.json'))
        execution=runtime.read_bound(runtime.reference(folder/'execution.json'))
        rgb=(folder/'rgb.png').read_bytes();supervisor.validate_observation(obs)
        need(record==row and row.get('model')=='gpt-6-astra' and row.get('proposal_admitted') is True and
            row.get('request_sha256')==request_ref['sha256'] and row.get('observation_sha256')==observation_ref['sha256'] and
            row.get('rgb_sha256')==hashlib.sha256(rgb).hexdigest() and request==supervisor.request_payload(obs,rgb) and
            row.get('proposal')==proposal and proposal.get('model')=='gpt-6-astra' and proposal.get('backend')=='codex_exec' and
            proposal.get('service_source_sha256')==runtime.reference(Path(__file__).with_name('astra_service.py'))['sha256'] and
            proposal.get('protocol_source_sha256')==runtime.reference(supervisor.__file__)['sha256'],
            'Actual Astra request/image/generated proposal/source binding differs')
        decision=supervisor.validate_decision(proposal['decision'],obs)
        need(decision['skill']==skill and row['boundary']==obs['boundary']==boundary and
            obs['episode_id']==meta['episode_id'] and obs['reference_sha256']==meta['reference_sha256'] and
            obs['sensor_bundle']['sensor_manifest_sha256']==manifest_ref['sha256'] and
            execution==row['execution'] and execution.get('completed') is True and execution.get('status')=='completed' and
            execution.get('physical_success') is None and row.get('physics_paused_during_inference') is True and
            row.get('inference_simulation_start_s')==row.get('inference_simulation_end_s')==obs['time_s'],
            'Actual admitted skill/paused native inference/execution binding differs')
        start=execution['physics_row_start'];end=execution['physics_row_end_exclusive']
        need(type(start) is int and type(end) is int and 0<=start<end<=len(events) and obs['physics_row']==start and
            obs['time_s']==events[start]['observed_before']['time_s'] and obs['q_rad']==events[start]['observed_before']['q'],
            'Astra observation/execution must bind to actual same-time native state')
        for event in events[start:end]:
            native=event['native_row']
            need(native.get('astra_decision_id')==index and native.get('supervisor_model')=='gpt-6-astra' and
                native.get('motor_actor')=='analytic_GivenContactExpert' and native.get('local_motor_assistance') is True and
                native.get('learned_policy') is False,'Astra executed native rows lost honest analytic motor attribution')
        raw=obs['sensor_bundle']['raw_rgbd'];raw_path=sensor_dir/raw['file']
        need(raw_path.parent==sensor_dir and runtime.reference(raw_path)['sha256']==raw['sha256'],'Actual exact native RGB-D artifact changed')
        ranges.append([start,end])
    need(ranges[0][0]==0 and all(left[1]==right[0] for left,right in zip(ranges,ranges[1:])) and
        ranges[-1][1]==len(events),'Astra skills must account for every actual native tick')
    return dict(model='gpt-6-astra',backend='codex_exec',executed_skills=3,native_row_ranges=ranges,
        local_motor_assistance=True,learned_motor_policy=False,sensor_manifest=manifest_ref)


def qualification_value(output,config):
    """Read-only check of intact original native evidence; no policy/model call."""
    output=Path(output);inputs={};checks={};passed=False;error=None;arm=None;protocol_ref=None;scenario=None
    try:
        if not isinstance(config,dict):
            inputs['configuration']=runtime.reference(config);config=runtime.read_bound(inputs['configuration'])
        binding=preflight_arm(config,output=output);arm=binding['arm'];meta=binding['metadata']
        protocol_ref=config[DESCRIPTOR]['protocol'];scenario=binding['case']['scenario_id']
        inputs['run']=runtime.reference(output/'fresh_expert_result.json');run=runtime.read_bound(inputs['run'])
        inputs['complete']=runtime.reference(output/'COMPLETE.json');complete=runtime.read_bound(inputs['complete'])
        need(run==complete and run.get('success') is True and run.get('physical_validation') is True and run.get('errors')==[],
            'Actual closed successful native result required; owner completion alone cannot qualify')
        receipt=run.get('neutral_native_audit',{});bound=receipt.get('pilot_binding',{})
        need(receipt.get('complete') is True and receipt.get('errors')==[] and receipt.get('metadata')==meta and
            receipt.get('source_inventory')==sources() and bound.get('configuration_sha256')==runtime.digest(config) and
            bound.get('episode_id')==meta['episode_id'] and bound.get('source_config')==binding['case']['config'],
            'Actual complete neutral audit/configuration/scene/reference binding differs')
        events=neutral.read_native_trace(receipt,meta)
        need(len(events)==run.get('physics_steps') and run.get('trace_file')==receipt['trace']['path'] and
            run.get('trace_sha256')==receipt['trace']['sha256'],'Actual full 240 Hz trace/row count differs')
        for path,sha in run.get('runtime_sources_sha256',{}).items():
            need(runtime.reference(path)['sha256']==sha,'Executed native runtime source changed')
        need(bool(run.get('runtime_sources_sha256')),'Actual runtime source inventory required')
        if arm=='act':
            dep=binding['deployment'];learned=run.get('learned_policy_receipt',{})
            need(run.get('learned_policy') is True and run.get('expert_l2_fallback') is False and
                bound.get('model')==dep['model'] and learned.get('model')==dep['model'] and
                learned.get('metadata')==meta and learned.get('source_inventory')==dep['source_inventory'],
                'Actual newly selected ACT required; expert L2 fallback cannot qualify as ACT')
            checks['ipc']=audit._ipc(output,run,dep,sys.modules[__name__],events)
            need(any(e['native_row'].get('command_source_id')=='given_contact_expanded_pilot_act_l2_interpolated_v1' for e in events) and
                not any(e['actual_control_owner']=='TEACHER_L2' for e in events),'Actual ACT native L2 action provenance required')
        else:
            need(run.get('learned_policy') is False and bound.get('model') is None and bound.get('arm')==arm and
                not any(e['actual_control_owner']=='ACT_L2' for e in events),'Expert/Astra must retain actual analytic motor provenance')
            if arm=='astra':checks['astra']=_astra_evidence(output,run,events,meta)
            else:need('astra_schema' not in run and bound.get('supervisor_model') is None,'Expert cannot count an Astra/learned arm')
        # This is exactly the original physical gate: 3D plane, parking/pose,
        # force, HOLD, release, collision, home, 240 Hz continuity and budgets.
        runtime.dataset._physical(run,events,meta)
        checks['original_native_physical_recheck']=neutral._physical_native(run,events,meta)
        checks.update(unchanged_physical_gate_passed=True,physics_steps=len(events),native_hz=240,
            final_measured_q=events[-1]['observed_after']['q'],final_subphase=events[-1]['controller_after']['subphase'])
        passed=True
    except Exception as failure:error=type(failure).__name__+': '+str(failure)
    return dict(schema=QUALIFICATION_SCHEMA,passed=passed,arm=arm,protocol=protocol_ref,scenario_id=scenario,
        configuration_sha256=runtime.digest(config) if isinstance(config,dict) else None,inputs=inputs,checks=checks,error=error,
        software_completion_is_physical_success=False,**FLAGS)


def qualify(output,config,*,receipt_path=None):
    return runtime.write_new(Path(output)/'MATCHED_NATIVE_QUALIFICATION.json' if receipt_path is None else receipt_path,
        qualification_value(output,config))


def _read_optional(path):
    if not Path(path).is_file():return None,None
    try:
        ref=runtime.reference(path);return runtime.read_bound(ref),ref
    except Exception as failure:return None,dict(path=str(path),error=type(failure).__name__+': '+str(failure))


def _outcome(case,arm,root,job,protocol_ref):
    row=dict(status='planned_unlaunched',qualified_physical_success=False,actual_native_episode=False,physics_steps=0,
        infrastructure_outcome=None,model_outcome=None,physical_outcome=None,evidence_errors=[],failure_reason=None)
    if case['preparation_status']!='prepared':
        row.update(status='preparation_failure',failure_reason=case['reason'],infrastructure_outcome='source_unavailable');return row
    folder=Path(root)/case['scenario_id'];result,report_ref=_read_optional(folder/'fresh_expert_result.json')
    worker,worker_ref=_read_optional(Path(root)/(case['scenario_id']+'.result.json'))
    launch,launch_ref=_read_optional(Path(root)/(case['scenario_id']+'.launch.json'))
    receipt,receipt_ref=_read_optional(folder/'MATCHED_NATIVE_QUALIFICATION.json')
    row.update(result=report_ref,worker=worker_ref,launch=launch_ref,qualification=receipt_ref)
    for ref in (report_ref,worker_ref,launch_ref,receipt_ref):
        if ref and 'error' in ref:row['evidence_errors'].append(ref['error'])
    if row['evidence_errors']:row['status']='evidence_error';return row
    if result is None:
        if worker is not None and worker.get('exit_code')!=0:row.update(status='infrastructure_abort',infrastructure_outcome='worker_failed_without_native_result')
        elif worker is not None or launch is not None:row['status']='missing_result'
        return row
    if worker is None or worker.get('config_sha256')!=job['sha256'] or worker.get('result_sha256')!=report_ref['sha256']:
        row.update(status='evidence_error',failure_reason='Original worker config/result binding absent or changed');return row
    steps=result.get('physics_steps');row['reported_physics_steps']=steps
    native_receipt=result.get('neutral_native_audit')
    if native_receipt is not None:
        try:
            config=runtime.read_bound(dict(path=job['config'],sha256=job['sha256']))
            meta=config[ROUTE_KEY if arm=='act' else 'm2_given_contact_control']['metadata']
            bound=native_receipt.get('pilot_binding',{})
            need(native_receipt.get('source_inventory')==sources() and
                bound.get('configuration_sha256')==runtime.digest(config) and
                bound.get('episode_id')==meta['episode_id'] and bound.get('source_config')==case['config'],
                'Reported native audit does not bind to original planned configuration')
            events=neutral._read_native_events(native_receipt,meta)
            need(type(steps) is int and len(events)==steps,'Reported native tick count differs from intact actual trace')
            row['physics_steps']=len(events);row['actual_native_episode']=bool(events)
        except Exception as failure:
            row['native_evidence_error']=type(failure).__name__+': '+str(failure)
    row['failure_reason']=result.get('failure_reason');row['reported_errors']=deepcopy(result.get('errors',[]))
    row['infrastructure_outcome']='worker_exit_0' if worker.get('exit_code')==0 else 'worker_nonzero_exit'
    if arm=='astra':
        decisions=result.get('astra_decisions',[])
        row['model_outcome']='abort' if any(d.get('execution',{}).get('status')=='aborted' for d in decisions) else (
            'inference_or_admission_failure' if any(d.get('inference_attempted') and not d.get('proposal_admitted') for d in decisions) else
            'generated_skills_executed' if result.get('supervisor_inference') else 'no_admitted_model_execution')
    elif arm=='act':row['model_outcome']='actual_policy_reported' if result.get('learned_policy') and not result.get('expert_l2_fallback') else 'no_attributed_act'
    else:row['model_outcome']='analytic_expert'
    if receipt is not None:
        expected=qualification_value(folder,job['config'])
        if receipt!=expected:
            row.update(status='evidence_error',failure_reason='Qualification changed or does not match current actual evidence');return row
        if receipt.get('passed') is True and receipt.get('protocol')==protocol_ref and receipt.get('arm')==arm:
            row.update(status='qualified_physical_success',qualified_physical_success=True,physical_outcome='original_native_gate_passed');return row
        row['qualification_error']=receipt.get('error')
    if result.get('success') is True and result.get('physical_validation') is True:
        row.update(status='unqualified_native_result',physical_outcome='reported_success_without_current_qualification')
    elif row['model_outcome']=='abort':row.update(status='model_abort',physical_outcome='not_qualified')
    elif type(steps) is int and steps>0 and not row['actual_native_episode']:
        row.update(status='unqualified_native_result',physical_outcome='reported_native_failure_without_intact_trace')
    elif not row['actual_native_episode']:row.update(status='infrastructure_abort',physical_outcome='no_native_episode')
    else:row.update(status='physical_or_execution_failure',physical_outcome='not_qualified')
    return row


def summarize(prepared_file,roots,output):
    prepared_ref=runtime.reference(prepared_file);prepared=runtime.read_bound(prepared_ref)
    need(prepared.get('schema')==PREPARED_SCHEMA and set(roots)==set(ARMS),'New preparation and all three explicit arm roots required')
    protocol,_=load_protocol(prepared['protocol']);outcomes=[];jobs={}
    for arm in ARMS:
        batch=runtime.read_bound(prepared['batches'][arm])
        from .isaac_batch import validate_jobs
        validate_jobs(batch);jobs[arm]={job['id']:job for job in batch['jobs']}
    for case in protocol['cases']:
        row={key:deepcopy(case[key]) for key in ('index','scenario_id','subject_id','split','cohort','pose_id','target_id','seed')}
        row['arms']={arm:_outcome(case,arm,roots[arm],jobs[arm].get(case['scenario_id']),prepared['protocol']) for arm in ARMS}
        outcomes.append(row)
    def counts(rows,arm):
        values=[r['arms'][arm] for r in rows];statuses=sorted({v['status'] for v in values})
        return dict(planned_cases=len(rows),actual_native_episodes=sum(v['actual_native_episode'] for v in values),
            qualified_physical_successes=sum(v['qualified_physical_success'] for v in values),
            qualified_physical_success_rate=(sum(v['qualified_physical_success'] for v in values)/len(rows) if rows else None),
            status_counts={status:sum(v['status']==status for v in values) for status in statuses})
    arms={arm:counts(outcomes,arm) for arm in ARMS}
    strata={cohort:dict(protocol['strata'][cohort],arms={arm:counts([r for r in outcomes if r['cohort']==cohort],arm) for arm in ARMS}) for cohort in COHORTS}
    return runtime.write_new(output,dict(schema='m2_rebuild_paired_eval_summary_v1',prepared=prepared_ref,protocol=prepared['protocol'],
        fixed_denominator=len(outcomes),planned_arm_denominator=len(outcomes)*3,outcomes=outcomes,arms=arms,strata=strata,
        qualified_physical_result_exists=any(v['qualified_physical_successes']>0 for v in arms.values()),
        qualification_required=True,software_completion_is_physical_success=False,**FLAGS))


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','serve','qualify','summarize'):modes.add_argument('--'+mode,action='store_true')
    for name in ('complete','cases','output','astra-settings','deployment','deployment-sha256','config','receipt','prepared',
            'expert-output','act-output','astra-output'):parser.add_argument('--'+name)
    parser.add_argument('--astra-parallelism',type=int,default=1)
    parser.add_argument('--parallelism',type=int,default=1);parser.add_argument('--run-id',default='m2-rebuild-paired-20261105-v1')
    args=parser.parse_args()
    if args.serve:
        need(args.deployment and args.deployment_sha256,'Bound actual CPU deployment required')
        serve(dict(path=str(Path(args.deployment).resolve()),sha256=args.deployment_sha256));return 0
    if args.prepare:
        need(args.complete and args.cases and args.output and args.astra_settings,'Actual new ACT COMPLETE/cases/Astra settings/new output required')
        settings=json.loads(Path(args.astra_settings).read_bytes());settings=settings.get('astra_supervisor',settings)
        ref=prepare(args.complete,args.cases,args.output,astra_settings=settings,parallelism=args.parallelism,astra_parallelism=args.astra_parallelism,run_id=args.run_id)
    elif args.qualify:
        need(args.output and args.config,'Explicit actual config/output required')
        ref=qualify(args.output,args.config,receipt_path=args.receipt);value=runtime.read_bound(ref)
        print(json.dumps(dict(receipt=ref,passed=value['passed'],error=value['error'])),flush=True);return 0 if value['passed'] else 1
    else:
        need(args.prepared and args.expert_output and args.act_output and args.astra_output and args.output,'Preparation and all actual arm roots required')
        ref=summarize(args.prepared,dict(expert=args.expert_output,act=args.act_output,astra=args.astra_output),args.output)
    print(json.dumps(ref),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
