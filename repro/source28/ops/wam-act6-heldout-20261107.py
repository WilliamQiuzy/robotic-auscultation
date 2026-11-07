#!/usr/bin/env python3
"""External finite ACT6 source28 expert/ACT heldout32 simulation operator.

The actual SHA-pinned four-case ACT6 DEV receipt is mandatory before any fresh
payload or builder access. Only original guard-owned continuation launches jobs.
No retry, resume, fitting, reset, expert ACT substitution, or physical-gate change.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.util
from importlib.machinery import ModuleSpec
import json
import os
from pathlib import Path
import stat
import sys
import traceback
from types import ModuleType, SimpleNamespace

CODE=Path('/opt/wam-runtime/code/wam-act-recovery-20261106-v28')
COMMIT='d932993b13030da7873d04a4a9471df99bb836ec'
MANIFEST_SHA256='f05a78822bcc6febbaa7f4a7a8de2cabd9aa33daed3d648b90053cb336edb89c'
FILE_COUNT=1673
BASE=Path('/opt/wam-runtime/fast-wam')
NATIVE='/opt/wam-runtime/envs/isaacsim60/bin/python'
CONTROL=BASE/'diagnostics/act6-terminal-heldout-20261107-v1'
PREPARE_RUN='act6-terminal-heldout-scenes-20261107-v1'
COMPARE_RUN='act6-terminal-heldout-20261107-v1'
TRAIN_RUN='act6-terminal-train-20261107-v1'
SMOKE_RUN='act6-terminal-smoke-20261107-v1'
MODEL_SCHEMA='m2_act_auxiliary_act_pilot_v4'
DEVELOPMENT_OPERATOR_SHA256='ba7cbe84bffe7c4fcc016653de9b20dcfddb99296746d95b3146bbb1a87684f8'
DEVELOPMENT_OPERATOR_BYTES=46964
POSES=('p000474_a001230_f0000','p000476_a001231_f0023')
FRESH8_IDS=('p000519','p000843','p000893','p000455','p000864','p000540','p000570','p000891')
CASE_SPECS=(('p001120','train','R3'),('p001120','train','L3'),('p000923','validation','R3'),('p000923','validation','L3'))
STOP_CONFIG=dict(schema='m2_stop_residual_native_cache_replan_v1',enabled=True,max_cache_recalculations=8)
TRAIN16_SCOPE=dict(path='/opt/wam-runtime/fast-wam/diagnostics/act-auxiliary-cohort-20261106-v2/COHORT_FREEZE.json',
    bytes=1521,sha256='adf6b0bd95669719989f6795c177f0a8419921dd79a257ca0c725f99b3220e71')
FRESH_COMPLETE=dict(path='/opt/wam-runtime/fast-wam/datasets/act-recovery-fresh-heldout-20261106-v1/COMPLETE.json',
    bytes=1319,sha256='22bbdad3769fed1c873f927aa7ef6753cb00a5815b417c551f8c1dee88df2eb2')
RECONSTRUCTION_SHA256='48be7e280dfbebd43ff42cd7b304378f149255ed66dbd1df0cda9b6a10d3ba61'
ARMS=('expert','act')


def need(condition,message):
    if not condition:raise ValueError(message)


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    path=Path(path);body=path.read_bytes()
    return dict(path=str(path),bytes=len(body),sha256=hashlib.sha256(body).hexdigest())


def require_host():
    need(sys.platform=='linux' and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
        sys.executable==NATIVE and sys.flags.isolated==1 and not sys.flags.optimize,
        'Unoptimized Native60 Linux -I -B CPU authoring operator with hidden CUDA required')
    need(os.environ.get('AUSC_OWNED_RUN') is None,'External authoring operator must not inherit GPU ownership')


def source_closure():
    """Full regular-file and closed-inventory authentication, never a filename-only check."""
    need(CODE.resolve()==CODE,'Exact resolved source28 root required')
    need((CODE/'CODE_COMMIT').read_text().strip()==COMMIT,'Exact source28 commit required')
    manifest=CODE/'SOURCE_MANIFEST.json'
    need(sha256(manifest)==MANIFEST_SHA256,'Exact source28 full manifest required')
    rows=json.loads(manifest.read_text());need(isinstance(rows,list) and len(rows)==FILE_COUNT,'Exact source28 file count required')
    names=set()
    for row in rows:
        need(isinstance(row,dict) and set(row)=={'file','bytes','sha256'},'Exact source inventory row required')
        name=row['file']
        need(isinstance(name,str) and bool(name) and not Path(name).is_absolute() and '..' not in Path(name).parts and
            Path(name).as_posix()==name and name not in names and name not in ('CODE_COMMIT','SOURCE_MANIFEST.json') and
            type(row['bytes']) is int and row['bytes']>=0,'Safe unique source inventory path required')
        names.add(name)
    actual=set()
    for path in CODE.rglob('*'):
        info=path.lstat();need(not stat.S_ISLNK(info.st_mode),'Symlink in immutable source28')
        if stat.S_ISDIR(info.st_mode):continue
        need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'Nonregular or hardlinked source28 file')
        actual.add(path.relative_to(CODE).as_posix())
    need(actual==names|{'CODE_COMMIT','SOURCE_MANIFEST.json'},'Unknown or omitted source inventory file')
    for row in rows:
        path=CODE/row['file']
        need(path.resolve()==path and reference(path)==dict(path=str(path),bytes=row['bytes'],sha256=row['sha256']),
            'Frozen source28 file changed: '+row['file'])
    return dict(root=str(CODE),code_commit=COMMIT,file_count=FILE_COUNT,
        source_manifest=reference(manifest),code_commit_file=reference(CODE/'CODE_COMMIT'))


def project_name(name):return name in ('sim','scripts','auscultation') or name.startswith(('sim.','scripts.','auscultation.'))


def module_origins(inventory=None):
    found={}
    for name,module in tuple(sys.modules.items()):
        if not project_name(name):continue
        path=getattr(module,'__file__',None);spec=getattr(module,'__spec__',None)
        if path is None:
            locations=list(getattr(module,'__path__',[]))
            need(locations and all(Path(p).resolve().is_relative_to(CODE) for p in locations),
                'Project namespace outside exact source28: '+name)
            continue
        need(isinstance(path,str) and spec is not None and spec.origin==path and Path(path).resolve()==Path(path) and
            Path(path).is_relative_to(CODE),'Project module origin outside actual source28: '+name)
        found[name]=dict(path=path,origin=spec.origin)
    if inventory is not None:
        for name,ref in inventory['files'].items():
            full='sim.ausc_policy.'+name[:-3]
            need(found.get(full)==dict(path=ref['path'],origin=ref['path']),
                'All22 actual V4 module origins required: '+full)
    return found


def pin_project_namespaces():
    # The official SDK may have installed a benign root `sim` namespace only.
    # It has executed no initializer; foreign project files/subnamespaces fail.
    for name,module in tuple(sys.modules.items()):
        if not project_name(name):continue
        path=getattr(module,'__file__',None)
        if path is not None:
            need(Path(path).resolve().is_relative_to(CODE),'Preimported project source outside source28: '+name)
        elif name not in ('sim','scripts'):
            locations=list(getattr(module,'__path__',[]))
            need(locations and all(Path(p).resolve().is_relative_to(CODE) for p in locations),
                'Preimported project namespace outside source28: '+name)
    for name in ('sim','scripts'):
        path=CODE/name;need(path.is_dir() and path.resolve()==path,'Exact source28 namespace root required')
        module=sys.modules.get(name)
        if module is None or getattr(module,'__file__',None) is None:
            module=ModuleType(name);module.__package__=name;module.__loader__=None
            module.__spec__=ModuleSpec(name,loader=None,is_package=True)
        module.__path__=[str(path)];module.__spec__.submodule_search_locations=module.__path__;sys.modules[name]=module


def current_inventory(runtime):
    value=runtime.aux.code_binding(version=4)
    need(value.get('code_commit')==COMMIT and len(value.get('files',{}))==22,'Exact current22 V4 numerical inventory required')
    for name,ref in value['files'].items():
        need(ref['path']==str(CODE/'sim/ausc_policy'/name) and reference(ref['path'])==ref,'Actual source28 numerical reference required')
        importlib.import_module('sim.ausc_policy.'+name[:-3])
    module_origins(value)
    return value


def bind_runtime():
    require_host();source_closure();sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.chdir(CODE);sys.path.insert(0,str(CODE));pin_project_namespaces()
    names=dict(development='scripts.act_student_recovery_round_20261106',rt='sim.ausc_policy.m2_given_contact_runtime',
        aux='sim.ausc_policy.m2_act_auxiliary_train',warm='sim.ausc_policy.m2_act_warm_start',head='sim.ausc_policy.m2_act_posture_head',
        scene_store='sim.ausc_policy.scene_store',residual='sim.ausc_policy.m2_stop_residual_trajectory',
        audit='sim.ausc_policy.m2_given_contact_pilot_native_audit',
        deployment='sim.ausc_policy.m2_given_contact_expanded_pilot_deployment',process_guard='sim.ausc_policy.process_guard')
    runtime=SimpleNamespace(**{key:importlib.import_module(name) for key,name in names.items()})
    need(runtime.development.CODE==CODE and runtime.development.BASE==BASE and runtime.development.NATIVE==NATIVE,
        'Original source28 owner helper/native authoring binding changed')
    current_inventory(runtime);module_origins();source_closure();runtime.load_heldout=load_heldout;return runtime


def load_heldout():
    # No heldout builder/reconstruction is imported before the complete DEV gate.
    names=dict(paired='sim.ausc_policy.m2_rebuild_paired_eval',builder='sim.ausc_policy.m2_pilot_config',
        rebuild='sim.ausc_policy.m2_rebuild_capture_dataset',scope='sim.ausc_policy.m2_act_recovery_scope',
        residual='sim.ausc_policy.m2_stop_residual_trajectory',isaac_batch='sim.ausc_policy.isaac_batch')
    value=SimpleNamespace(**{key:importlib.import_module(name) for key,name in names.items()})
    module_origins();source_closure();return value


def normal_owner(value):
    stage=value.get('stage',{})
    return (value.get('completed') is True and value.get('gpu_released') is True and
        stage.get('passed') is True and type(stage.get('exit_code')) is int and stage['exit_code']==0 and
        stage.get('process_group_drained') is True and all(stage.get(k)==[] for k in ('errors','signals','deferred_signals')))


def owner_binding(runtime,value,run_id,config_sha256,source):
    guard=dict(path=source['guard']['path'],sha256=source['guard']['sha256'],
        gpu_exit_grace_s=runtime.process_guard.EXIT_GPU_GRACE_S)
    return (normal_owner(value) and value.get('run_id')==run_id and value.get('code_commit')==COMMIT and
        value.get('config_sha256')==config_sha256 and value.get('resource_guard')==guard)


def load_development_operator(rt,ref):
    """Definition-only reuse, with the complete reviewed dependency authenticated."""
    need(ref.get('sha256')==DEVELOPMENT_OPERATOR_SHA256 and ref.get('bytes')==DEVELOPMENT_OPERATOR_BYTES,
        'Exact reviewed ACT6 development operator SHA/size required')
    need(rt.reference(ref['path'])==ref and not Path(ref['path']).resolve().is_relative_to(CODE),
        'External reviewed ACT6 development operator changed')
    spec=importlib.util.spec_from_file_location('wam_act6_closed_development_validator',ref['path'])
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    # PILOT_SCHEMA is the dependency's public constant (not MODEL_SCHEMA).
    need(value.CODE==CODE and value.COMMIT==COMMIT and value.PILOT_SCHEMA==MODEL_SCHEMA and
        value.MANIFEST_SHA256==MANIFEST_SHA256 and value.FILE_COUNT==FILE_COUNT and rt.reference(ref['path'])==ref,
        'Reviewed ACT6 validator constants or source changed')
    return value


def validate_act6_integration(runtime,developed,source,inventory,closure):
    helper=load_development_operator(runtime.rt,developed['operator'])
    audit=helper.FiniteRound(runtime);audit.operator=developed['operator'];audit.source=source
    audit.source28=closure;audit.inventory=inventory
    config,rows=audit.bind_integration(developed['integration'])
    need(audit.training_config==developed['training_config'] and developed.get('initialization')==helper.INIT_REF and
        developed.get('data_export_control')==audit.data_export_control and
        developed.get('data_preparation')==audit.data_preparation,
        'DEV exact actual integration/config/initialization/data references differ')
    return config,rows,audit.inputs


def validate_owned_training_artifacts(rt,ref,complete):
    """The new owner's own selection and CPU receipt cannot come from ACT5."""
    output=BASE/'runs/ausc_policy'/TRAIN_RUN/'output'
    need(ref.get('path')==str(output/'COMPLETE.json'),
        'Exact ACT6 training request/terminal output path required')
    refs=[]
    for key,filename in (('best','best_auxiliary_act.pt'),('cpu_equivalence','CPU_EQUIVALENCE.json')):
        item=complete.get(key,{})
        need(item.get('path')==str(output/filename) and rt.reference(item['path'])==item,
            'Selected ACT6 model and original own CPU-equivalence artifacts required')
        refs.append(item)
    cpu=rt.read_bound(complete['cpu_equivalence']);selection=cpu.get('selection',{})
    need(selection.get('path')==str(output/'SELECTED.json') and rt.reference(selection['path'])==selection,
        'Original ACT6 own selected provenance required')
    return refs+[selection]


def validate_development(runtime,developed_ref,source):
    """Fail closed before fresh payloads, builder preparation, or outcomes."""
    rt=runtime.rt;developed=rt.read_bound(developed_ref)
    need(developed.get('schema')=='act6_finite_development_v1' and
        all(type(developed.get(k)) is int and developed[k]==4 for k in
            ('physical_successes','fixed_denominator','declared_cases','native_requested_cases','executed_cases','qualification_attempted_cases')) and
        all(developed.get(k) is True for k in ('normal_training_owner','normal_smoke_owner','qualified_dev_only',
            'model_qualified_dev_only','stop_adapter_tested','combined_full_treatment_tested','fresh_unexposed')) and
        all(developed.get(k) is False for k in ('heldout_executed','fresh_identities_exposed','physics_gates_changed',
            'expert_l2_fallback','automatic_retry','automatic_resume')) and
        developed.get('protocol_errors')==[] and not developed.get('error') and developed.get('errors',[])==[] and
        developed.get('stop_configuration')==STOP_CONFIG,
        'All four original actual ACT6 DEV cases must qualify without protocol/errors')
    expected=[(s+'-'+t.lower(),s,split,t,POSES[0]) for s,split,t in CASE_SPECS]
    outcomes=developed.get('outcomes',[])
    actual=[tuple(c.get(k) for k in ('id','subject_id','split','target_id','pose_id')) for c in outcomes]
    need(actual==expected and all(c.get('passed') is True and c.get('qualification_attempted') is True and
        c.get('physical_gate_passed') is True and c.get('executed') is True and c.get('infrastructure_error') is False and
        c.get('preflight_passed') is True and not c.get('error') and c.get('qualification') for c in outcomes),
        'Exact original four DEV identities, actual physics and independent physical qualifications required')
    closure=source_closure();inventory=current_inventory(runtime)
    need(developed.get('source')==source and developed.get('source28')==developed.get('source28_end')==closure and
        developed.get('code_inventory22')==inventory and source.get('code_commit')==rt.reference(CODE/'CODE_COMMIT') and
        source.get('source_manifest')==rt.reference(CODE/'SOURCE_MANIFEST.json') and
        developed.get('continuation_snapshot_catalog')=='original unchanged helper legacy V3',
        'DEV actual source28 closure/current22/truthful continuation snapshot differs')
    runtime.development.unchanged(source)
    dev_operator=developed.get('operator',{})
    need(dev_operator.get('path') and rt.reference(dev_operator['path'])==dev_operator and
        not Path(dev_operator['path']).resolve().is_relative_to(CODE),
        'DEV external operator reference changed')
    train_ref=developed['training_config'];train_config=rt.read_bound(train_ref)
    need(rt.reference(train_ref['path'])==train_ref and train_config.get('schema')=='m2_act_auxiliary_training_config_v4' and
        type(train_config.get('steps')) is int and train_config['steps']==20000,
        'Exact new20000-step V4 DEV training request required')
    smoke_batch=rt.reference(BASE/'diagnostics/act6-terminal-development-20261107-v1/smokes/batch.json')
    for key,run_id in (('training_owner',TRAIN_RUN),('smoke_owner',SMOKE_RUN)):
        ref=developed[key];need(Path(ref['path'])==BASE/'runs/ausc_policy'/run_id/'result.json','Exact ACT6 DEV owner terminal required')
        owner=rt.read_bound(ref)
        config_sha=train_ref['sha256'] if key=='training_owner' else smoke_batch['sha256']
        need(owner_binding(runtime,owner,run_id,config_sha,source),'DEV original source28 owner must close, drain and release normally')
        if key=='training_owner':
            need(owner.get('config')==train_config,'DEV actual training owner request differs')
        if key=='smoke_owner':
            batch=owner.get('config',{});jobs=batch.get('jobs',[])
            need(batch==rt.read_bound(smoke_batch) and batch.get('parallelism')==2 and
                [j.get('id') for j in jobs]==[c['id'] for c in outcomes] and
                all(j.get('config')==c['config']['path'] and j.get('sha256')==c['config']['sha256'] for j,c in zip(jobs,outcomes)),
                'DEV smoke owner must execute all four exact bound configs with two workers')
    config,rows,frozen=validate_act6_integration(runtime,developed,source,inventory,closure)
    need(config==train_config,'Actual CPU integration and DEV training request differ')
    complete=rt.read_bound(developed['training'])
    owned=validate_owned_training_artifacts(rt,developed['training'],complete)
    need(complete.get('schema')==MODEL_SCHEMA and complete.get('complete') is True and
        type(complete.get('completed_steps')) is int and complete['completed_steps']==20000 and
        complete.get('schedule')==train_config and train_config.get('evaluation_cohort')==TRAIN16_SCOPE and
        [r.get('step') for r in complete.get('evaluations',[])]==list(range(500,20001,500)) and
        complete.get('inputs',{}).get('config')==train_ref and complete['inputs'].get('code')==inventory,
        'Actual new20k/40full-native-evaluation request/completed V4 and numerical22 required')
    # Original admission authenticates earliest-min full native VAL selection,
    # all train/VAL populations, initialization and original own CPU equivalence.
    model,bound=runtime.deployment.model_binding(developed['training'])
    need(bound==complete and model==developed.get('model') and model.get('checkpoint_schema')==MODEL_SCHEMA and
        model.get('complete')==developed['training'],'Exact actual DEV-selected V4 model binding changed')
    native_sources=runtime.deployment.model_sources(model)
    need(developed.get('source_inventory')==native_sources and
        all(native_sources.get('files',{}).get(k)==v for k,v in inventory['files'].items()),
        'Actual model-specific V4 native inventory must include all current22 modules')
    native_ref=complete['schedule']['dataset'];need(complete['inputs'].get('dataset')==native_ref,
        'DEV original nominal bank differs from actual new training input')
    native=rt.read_bound(native_ref)
    for case in outcomes:
        matches=[r for r in native.get('episodes',[]) if r.get('origin')=='qualified_actual_nominal' and
            all(r.get(k)==case[k] for k in ('subject_id','split','target_id','pose_id'))]
        need(len(matches)==1,'Unique actual original nominal DEV identity/split/site/pose474 required')
        row=matches[0];cfg=rt.read_bound(case['config']);dep=cfg.get('m2_given_contact_expanded_pilot_deployment',{})
        meta=dep.get('metadata',{});episode=SMOKE_RUN+'-'+case['id'];original=rt.read_bound(row['config'])
        old=original.get('m2_given_contact_control',{});oldmeta=old.get('metadata',{})
        need(dep.get('model')==model and dep.get('source_inventory')==native_sources and
            dep.get('source_episode_id')==case.get('source_episode_id')==row['episode_id'] and dep.get('source_config')==row['config'] and
            all(meta.get(k)==case[k] and oldmeta.get(k)==case[k] for k in ('subject_id','split','target_id')) and
            meta.get('episode_id')==dep.get('episode_id')==episode and meta.get('pose_id',POSES[0])==POSES[0] and
            old.get('episode_id')==oldmeta.get('episode_id')==row['episode_id'] and
            cfg.get('stop_residual_trajectory')==STOP_CONFIG and type(cfg['stop_residual_trajectory']['enabled']) is bool and
            type(cfg['stop_residual_trajectory']['max_cache_recalculations']) is int,
            'DEV exact original known-pose config/model/source/STOP attribution differs')
        runtime.deployment.preflight(cfg,output=BASE/'runs/ausc_policy'/SMOKE_RUN/'output'/case['id'])
        runtime.scene_store.validate_config(cfg)
        qualification=rt.read_bound(case['qualification'])
        need(qualification.get('schema')=='m2_given_contact_diagnostic_native_qualification_v1' and
            qualification.get('passed') is True and not qualification.get('error') and
            qualification.get('inputs',{}).get('configuration')==case['config'] and
            qualification.get('configuration_sha256')==hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest() and
            qualification.get('checks',{}).get('unchanged_physical_gate_passed') is True,
            'Actual independent DEV audit/configuration/unchanged physical gate binding failed')
        reports=[]
        for key,filename in (('run','fresh_expert_result.json'),('complete','COMPLETE.json')):
            ref=qualification['inputs'].get(key,{})
            need(ref.get('path')==str(BASE/'runs/ausc_policy'/SMOKE_RUN/'output'/case['id']/filename),
                'DEV audit must bind original actual closed native reports')
            report=rt.read_bound(ref)
            need(report.get('success') is True and report.get('physical_validation') is True and report.get('errors')==[] and
                report.get('learned_policy') is True and report.get('expert_l2_fallback') is False and
                type(report.get('physics_steps')) is int and report['physics_steps']>0,
                'Actual error-free learned DEV physics is required')
            reports.append(report)
        need(reports[0]==reports[1],'Original closed native DEV reports differ')
    runtime.act6_gate_inputs=deepcopy(frozen)+[smoke_batch]+deepcopy(owned)
    return developed,model,complete


def planned_cases(cohort):
    need(cohort.get('pose_ids') == list(POSES) and cohort.get('target_ids') == ['R3', 'L3'],
         'Original known poses and R3/L3 required')
    subjects = cohort['cohorts']['frozen_evaluation']
    need(subjects == list(FRESH8_IDS) and len(set(subjects))==8, 'Exact ordered original fresh8 identities required')
    training=set(cohort['cohorts'].get('train',[]))
    selection=set(cohort['cohorts'].get('selection_validation',[]))
    need(not set(subjects)&(training|selection) and not training&selection, 'Fresh identities must be disjoint from optimizer and selection')
    cases = []
    for subject in subjects:
        for pose in POSES:
            for target in ('R3', 'L3'):
                scenario = subject + '-' + pose + '-' + target.lower()
                episode = COMPARE_RUN + '-' + scenario
                need(len(episode) <= 80 and all(ch in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for ch in episode),
                     'Original builder canonical episode constraint exceeded')
                cases.append(dict(index=len(cases) + 1, scenario_id=scenario, episode_id=episode,
                    subject_id=subject, split='validation', cohort='frozen_evaluation', pose_id=pose, target_id=target,
                    seed=0, preparation_status='unavailable', config=None, reason='CPU preparation has not completed'))
    return cases


def build_cases(runtime, manifest, cohort_ref, output):
    """Original CPU construction with this external run id, never monkeypatching frozen modules."""
    rt = runtime.rt
    post = runtime.heldout
    frozen = rt.read_bound(cohort_ref)
    inventory = post.scope.validate(cohort_ref)
    need(manifest['purpose'] == 'frozen_evaluation' and manifest['requested_target_slots'] == 32 and
         set(manifest['subject_ids']) == set(inventory['frozen_evaluation']),
         'CPU preparation changed frozen identities or planned denominator')
    retained = rt.read_bound(rt.reference(BASE / 'datasets/rebuilt-expert-capture-20261105-v2-prepared/REQUEST.json'))
    store = deepcopy(retained['scene_store'])
    store.update(root=str(output / 'scenes'), batch_id=COMPARE_RUN)
    cases = []
    for scene in manifest['scenes']:
        for target in frozen['target_ids']:
            scenario = scene['subject_id'] + '-' + scene['pose_id'] + '-' + target.lower()
            case = dict(index=len(cases) + 1, scenario_id=scenario, episode_id=COMPARE_RUN + '-' + scenario,
                subject_id=scene['subject_id'], split='validation', cohort='frozen_evaluation', pose_id=scene['pose_id'],
                target_id=target, seed=0, preparation_status='unavailable', config=None,
                reason=scene['plans'][target].get('reason', 'CPU preparation unavailable'))
            cases.append(case)
            if 'preparation_error' in scene or 'plan' not in scene['plans'][target]:
                continue
            try:
                assembled = post.builder.build_given_contact_expert(
                    **{key: post.rebuild.pilot_ref(scene[key]) for key in ('preparation_config', 'prepared_complete', 'patient_receipt')},
                    fresh_plan=post.rebuild.pilot_ref(scene['plans'][target]['plan']), episode_id=case['episode_id'],
                    max_frames=5000, collision_profile='full',
                    retainer_witness_contract=deepcopy(retained['retainer_witness_contract']), scene_store=store)
                post.paired.validate_expert_config(assembled['config'], case)
                case.update(config=rt.write_new(output / 'configs' / (scenario + '.json'), assembled['config']),
                    preparation_status='prepared', reason=None)
                rt.write_new(output / 'configs' / (scenario + '.receipt.json'), assembled['receipt'])
            except Exception as error:
                case['reason'] = type(error).__name__ + ': ' + str(error)
    value = dict(schema=post.paired.INPUT_SCHEMA, complete=True, model_outcomes_consulted=False, identity_cohort=cohort_ref,
        fixed_denominator=32, eval_cohorts=['frozen_evaluation'], target_ids=frozen['target_ids'], pose_ids=frozen['pose_ids'], cases=cases)
    post.paired.validate_cases(value)
    return rt.write_new(output / 'CASES.json', value)


def project_cases(runtime, original_ref, output):
    """Bind new STOP-only configs separately, preserving CPU originals and all failed cases."""
    rt = runtime.rt
    post = runtime.heldout
    need(post.residual.CONFIG == STOP_CONFIG and post.residual.CONFIG_KEY == 'stop_residual_trajectory',
         'Source28 exact STOP adapter required')
    original = rt.read_bound(original_ref)
    post.paired.validate_cases(original)
    projected = deepcopy(original)
    for case in projected['cases']:
        if case['preparation_status'] != 'prepared':
            continue
        config = rt.read_bound(case['config'])
        metadata = config['m2_given_contact_control']['metadata']
        actual = post.residual.configuration_projection(config, {post.residual.CONFIG_KEY: deepcopy(STOP_CONFIG)}, metadata=metadata)
        need(actual.get(post.residual.CONFIG_KEY) == STOP_CONFIG and
             {key: value for key, value in actual.items() if key != post.residual.CONFIG_KEY} ==
             {key: value for key, value in config.items() if key != post.residual.CONFIG_KEY}, 'STOP projection changed original physical/scene settings')
        post.paired.validate_expert_config(actual, case)
        case['config'] = rt.write_new(output / 'configs' / (case['scenario_id'] + '.json'), actual)
    post.paired.validate_cases(projected)
    need(rt.reference(original_ref['path']) == original_ref, 'Original CPU cases changed')
    return rt.write_new(output / 'CASES.json', projected)


def select_media(protocol):
    return {target: next((case['scenario_id'] for case in protocol['cases'] if
        case['preparation_status'] == 'prepared' and case['target_id'] == target), None) for target in ('R3', 'L3')}


def verify_prepared(runtime, prepared_ref, model, cases_ref):
    rt = runtime.rt
    post = runtime.heldout
    prepared = rt.read_bound(prepared_ref)
    protocol = rt.read_bound(prepared['protocol'])
    cases = rt.read_bound(cases_ref)
    need(protocol.get('arms') == list(ARMS) and set(prepared.get('batches', {})) == set(ARMS) and
         protocol.get('cases') == cases['cases'] and protocol.get('expert_configs') == cases_ref and
         protocol.get('model') == prepared.get('model') == model and protocol.get('run_id') == COMPARE_RUN and
         protocol.get('arm_parallelism') == {'expert': 2, 'act': 2}, 'Exact finite expert/ACT frozen protocol differs')
    prepared_cases = [case for case in protocol['cases'] if case['preparation_status'] == 'prepared']
    for arm in ARMS:
        batch = rt.read_bound(prepared['batches'][arm])
        need(batch.get('parallelism') == 2 and [job['id'] for job in batch['jobs']] ==
             [case['scenario_id'] for case in prepared_cases], 'Both native arms require every CPU-prepared case')
        post.isaac_batch.validate_jobs(batch)
        for job, case in zip(batch['jobs'], prepared_cases):
            config_ref = rt.reference(job['config'])
            need(config_ref['sha256'] == job['sha256'], 'Frozen arm configuration changed')
            config = rt.read_bound(config_ref)
            option = config.get('stop_residual_trajectory')
            need(option == STOP_CONFIG and type(option.get('enabled')) is bool and
                 type(option.get('max_cache_recalculations')) is int, 'Identical exact top-level STOP option required in both arms')
            original = rt.read_bound(case['config'])
            expected = post.paired.derive_config(original, case, prepared['protocol'], protocol, arm,
                stop_residual_trajectory=deepcopy(STOP_CONFIG))
            need(config == expected, 'Arm differs from exact STOP-bound original source configuration')
            runtime.scene_store.validate_config(config)
            post.paired.preflight_arm(config, output=BASE / 'runs/ausc_policy' / (COMPARE_RUN + '-' + arm) / 'output' / case['scenario_id'])
    return prepared, protocol


def acceptance(protocol, summary, *, owners_normal=True):
    """Original32 acceptance plus row-level denominator/execution consistency."""
    cases = protocol['cases']
    need(protocol['fixed_denominator'] == 32 == len(cases) and len({case['scenario_id'] for case in cases}) == 32,
         'All unique original32 planned cases required')
    planned = {case['scenario_id']: case for case in cases}
    rows = summary['outcomes']
    need(len(rows) == 32 and {row['scenario_id'] for row in rows} == set(planned), 'Every planned32 outcome required')
    for row in rows:
        case = planned[row['scenario_id']]
        need(all(row.get(key) == case.get(key) for key in ('subject_id', 'pose_id', 'target_id')),
             'Outcome identity/known pose/target changed')
        for arm in ARMS:
            value = row['arms'][arm]
            need(type(value['actual_native_episode']) is bool and type(value['qualified_physical_success']) is bool and
                 (not value['qualified_physical_success'] or value['actual_native_episode']) and
                 (case['preparation_status'] == 'prepared' or not value['actual_native_episode']),
                 'Unavailable or unexecuted case cannot become an actual physical success')
    prepared = [case for case in cases if case['preparation_status'] == 'prepared']
    ids = {case['scenario_id'] for case in prepared}
    executed = {}
    for arm in ARMS:
        actual = sum(row['arms'][arm]['actual_native_episode'] for row in rows)
        successes = sum(row['arms'][arm]['qualified_physical_success'] for row in rows)
        need(summary['arms'][arm]['actual_native_episodes'] == actual and
             summary['arms'][arm]['qualified_physical_successes'] == successes, 'Arm counts disagree with actual case ledger')
        executed[arm] = actual
    positives = [row for row in rows if row['scenario_id'] in ids and row['arms']['act']['qualified_physical_success'] is True]
    n = len(prepared)
    successes = len(positives)
    rate = successes / n if n else 0.
    subjects = {row['subject_id'] for row in positives}
    poses = {row['pose_id'] for row in positives}
    targets = {row['target_id'] for row in positives}
    passed = (owners_normal and n >= 24 and all(value == n for value in executed.values()) and
        rate >= .95 and len(subjects) >= 6 and poses == set(POSES) and targets == {'R3', 'L3'})
    return dict(passed=passed, planned_cases=32, cpu_prepared_cases=n, cpu_unavailable_cases=32-n,
        actual_native_episodes=executed, qualified_act_successes=successes, prepared_act_success_rate=rate,
        planned_act_success_rate=successes/32, successful_subjects=len(subjects), successful_poses=sorted(poses),
        successful_targets=sorted(targets), minimum_prepared_cases=24, minimum_prepared_success_rate=.95,
        minimum_successful_subjects=6, owners_normal=owners_normal, physical_gate_changed=False,
        scope='Prescribed known 3D R3/L3, frozen8 identities, two known seated poses; state-only H2/state17 H20/E10 ACT L2 plus analytic L3. No VLA, neural RGB-D, unseen-pose or hardware acceptance.')


def unavailable_summary(protocol, reason):
    rows = []
    for case in protocol['cases']:
        row = deepcopy(case)
        row['arms'] = {arm: dict(status='preparation_failure' if case['preparation_status'] != 'prepared' else 'planned_unlaunched',
            qualified_physical_success=False, actual_native_episode=False,
            failure_reason=case.get('reason') or reason) for arm in ARMS}
        rows.append(row)
    return dict(fixed_denominator=32, planned_arm_denominator=64, outcomes=rows,
        arms={arm: dict(planned_cases=32, actual_native_episodes=0, qualified_physical_successes=0) for arm in ARMS},
        qualification_required=True, software_completion_is_physical_success=False, summary_error=reason)


class HeldoutRound:
    def __init__(self, runtime):
        self.runtime = runtime
        self.rt = runtime.rt
        self.source28 = None
        self.inventory = None
        self.source_inventory = None
        self.source = None
        self.operator = None
        self.inputs = []
        self.developed = None
        self.developed_ref = None
        self.model = None
        self.cohort_ref = None
        self.manifest_ref = None
        self.original_cases_ref = None
        self.cases_ref = None
        self.prepared_ref = None
        self.prepared = None
        self.protocol = None
        self.summary_ref = None
        self.rejected_summary = None
        self.preparation_owner = None
        self.owners = {}
        self.owner_requested = []
        self.audit_errors = []
        self.development_physical_rechecks = []
        self.phase = 'development_gate'
        self.roots = {arm: str(BASE / 'runs/ausc_policy' / (COMPARE_RUN + '-' + arm) / 'output') for arm in ARMS}

    def unchanged(self):
        need(source_closure()==self.source28, 'Actual full source28 closure changed')
        need(current_inventory(self.runtime)==self.inventory, 'Actual current22 numerical inventory changed')
        if self.model is not None:
            need(self.runtime.deployment.model_sources(self.model)==self.source_inventory, 'Model-specific V4 native inventory changed')
        self.runtime.development.unchanged(self.source)
        need(self.rt.reference(__file__) == self.operator, 'External held-out operator changed')
        for ref in self.inputs:
            need(self.rt.reference(ref['path']) == ref, 'Frozen DEV/model/cohort/configuration reference changed')

    def check_unique(self):
        need(not CONTROL.exists(), 'Unique finite control required; no retry or resume')
        for run_id in (PREPARE_RUN, COMPARE_RUN + '-expert', COMPARE_RUN + '-act'):
            need(not (BASE / 'runs/ausc_policy' / run_id).exists(), 'Owner run already exists; no retry or resume')

    def owner_terminals(self):
        result = {}
        for run_id in (PREPARE_RUN, COMPARE_RUN + '-expert', COMPARE_RUN + '-act'):
            path = BASE / 'runs/ausc_policy' / run_id / 'result.json'
            if path.is_file():
                result[run_id] = self.rt.reference(path)
        return result

    def launch(self, config, run_id):
        self.unchanged()
        need(run_id not in self.owner_requested and not (BASE / 'runs/ausc_policy' / run_id).exists(),
             'No duplicate, retry, or resumed owner launch')
        self.owner_requested.append(run_id)
        ref, value = self.runtime.development.run_owner(config, run_id, self.runtime.development.NATIVE, self.source, control=CONTROL)
        need(owner_binding(self.runtime,value,run_id,config['sha256'],self.source),
             'Original owner terminal/source/request/guard did not close normally')
        return ref

    def qualify_all(self):
        for arm in ARMS:
            batch = self.rt.read_bound(self.prepared['batches'][arm])
            for job in batch['jobs']:
                try:
                    self.runtime.heldout.paired.qualify(Path(self.roots[arm]) / job['id'], job['config'])
                except Exception as error:
                    self.audit_errors.append(dict(arm=arm, scenario_id=job['id'], error=repr(error)))
        self.unchanged()

    def recheck_development_physics(self):
        """Original auditor, new receipts, all four cases, before fresh access."""
        for case in self.developed['outcomes']:
            row=dict(id=case['id'],passed=False,receipt=None,error=None)
            try:
                ref=self.runtime.audit.qualify(BASE/'runs/ausc_policy'/SMOKE_RUN/'output'/case['id'],
                    case['config']['path'],receipt_path=CONTROL/'development-rechecks'/(case['id']+'.json'))
                self.inputs.append(ref);row['receipt']=ref
                value=self.rt.read_bound(ref)
                need(value==self.rt.read_bound(case['qualification']) and value.get('passed') is True and
                    not value.get('error'),'Original DEV physical audit changed or failed')
                row['passed']=True
            except Exception as error:
                row['error']=repr(error)
            self.development_physical_rechecks.append(row)
        need(len(self.development_physical_rechecks)==4 and all(r['passed'] for r in self.development_physical_rechecks),
            'All four original DEV physical gates must still pass before fresh payload access')
        self.unchanged()

    def summarize(self, reason=None):
        if self.summary_ref is not None:
            return self.rt.read_bound(self.summary_ref)
        if self.prepared_ref is not None:
            candidate = None
            try:
                candidate = self.runtime.heldout.paired.summarize(self.prepared_ref['path'], self.roots, CONTROL / 'SUMMARY.json')
                value = self.rt.read_bound(candidate)
                acceptance(self.protocol, value, owners_normal=False)
                self.summary_ref = candidate
                return value
            except Exception as error:
                self.rejected_summary = candidate
                reason = str(reason or '') + '; original paired summary failed: ' + repr(error)
                value = unavailable_summary(self.protocol, reason)
                self.summary_ref = self.rt.write_new(CONTROL / 'SUMMARY_FALLBACK.json', value)
                raise
        value = unavailable_summary(self.protocol, reason or 'No CPU-prepared cases')
        self.summary_ref = self.rt.write_new(CONTROL / 'SUMMARY.json', value)
        return value

    def actual_native_artifacts(self):
        """Retain raw partial native evidence even if independent summary fails."""
        rows = []
        for case in self.protocol['cases']:
            row = dict(scenario_id=case['scenario_id'], arms={})
            for arm in ARMS:
                root = Path(self.roots[arm])
                folder = root / case['scenario_id']
                refs = {}
                paths = dict(native_result=folder / 'fresh_expert_result.json', native_complete=folder / 'COMPLETE.json',
                    worker=root / (case['scenario_id'] + '.result.json'), launch=root / (case['scenario_id'] + '.launch.json'))
                for name, path in paths.items():
                    if path.is_file():
                        try:
                            refs[name] = self.rt.reference(path)
                        except Exception as error:
                            refs[name] = dict(path=str(path), evidence_error=repr(error))
                row['arms'][arm] = refs
            rows.append(row)
        return rows

    def result(self, summary, *, owners_normal):
        verdict = acceptance(self.protocol, summary, owners_normal=owners_normal and not self.audit_errors)
        return dict(development=self.developed_ref, summary=self.summary_ref, acceptance=verdict,
            owners=deepcopy(self.owners), preparation_owner=self.preparation_owner, rejected_summary=self.rejected_summary,
            owner_terminals=self.owner_terminals(), owner_requested=list(self.owner_requested),
            cpu_scene_manifest=self.manifest_ref, original_cpu_cases=self.original_cases_ref, stop_bound_cases=self.cases_ref,
            model=self.model, cohort=self.cohort_ref, source=self.source, source28=self.source28,
            code_inventory22=self.inventory, source_inventory=self.source_inventory, operator=self.operator, frozen_input_refs=deepcopy(self.inputs),
            audit_errors=deepcopy(self.audit_errors), actual_native_artifacts=self.actual_native_artifacts(), fixed_denominator=32, planned_arm_denominator=64,
            development_physical_rechecks=deepcopy(self.development_physical_rechecks),
            physics_gates_changed=False, automatic_retry=False, automatic_resume=False,
            state_only=True, expert_l2_substitution=False, fitting_or_tuning=False)

    def run(self, development, expected_sha256):
        require_host()
        self.source28=source_closure()
        self.inventory=current_inventory(self.runtime)
        need(BASE.resolve() == BASE and self.runtime.development.CODE.resolve() == CODE and
             self.runtime.development.BASE == BASE, 'Exact resolved source28/runtime base required')
        self.check_unique()
        self.developed_ref = self.rt.reference(Path(development).resolve())
        need(self.developed_ref['sha256'] == expected_sha256, 'DEV request SHA differs from frozen input')
        self.source = self.runtime.development.snapshot()
        self.operator = self.rt.reference(__file__)
        need(not Path(self.operator['path']).is_relative_to(CODE), 'External operator must remain outside immutable runtime')
        self.developed, self.model, complete = validate_development(self.runtime, self.developed_ref, self.source)
        self.source_inventory=self.runtime.deployment.model_sources(self.model)
        # Fresh builder APIs, DOF preparation and held-out payload/outcome
        # access require the complete gate above. Importing definitions alone
        # does not expose fresh data.
        self.runtime.heldout = self.runtime.load_heldout()
        self.cohort_ref = complete['schedule']['evaluation_cohort']
        cohorts = self.runtime.heldout.scope.validate(self.cohort_ref)
        cohort = self.rt.read_bound(self.cohort_ref)
        need(cohorts['frozen_evaluation'] == cohort['cohorts']['frozen_evaluation'], 'Fresh8 scope changed')
        self.protocol = dict(fixed_denominator=32, cases=planned_cases(cohort), arms=list(ARMS))
        self.inputs = [self.developed_ref, self.developed['training_config'], self.developed['training'],
            self.model['checkpoint'], self.cohort_ref, self.developed['operator'], self.developed['training_owner'], self.developed['smoke_owner']]
        self.inputs.extend(getattr(self.runtime,'act6_gate_inputs',[]))
        self.inputs.extend(case['qualification'] for case in self.developed['outcomes'])
        self.inputs.extend(case['config'] for case in self.developed['outcomes'])
        self.inputs.append(complete['schedule']['dataset'])
        self.inputs.extend(self.rt.read_bound(case['config'])['m2_given_contact_expanded_pilot_deployment']['source_config']
            for case in self.developed['outcomes'])
        for case in self.developed['outcomes']:
            self.inputs.extend(self.rt.read_bound(case['qualification'])['inputs'].values())
        if 'cpu_equivalence' in complete:
            self.inputs.append(complete['cpu_equivalence'])
            self.inputs.append(self.rt.read_bound(complete['cpu_equivalence'])['selection'])
        self.check_unique()
        CONTROL.mkdir(parents=True, exist_ok=False)
        self.rt.write_new(CONTROL / 'STARTED.json', dict(development=self.developed_ref, model=self.model,
            cohort=self.cohort_ref, source=self.source, operator=self.operator, pid=os.getpid(),
            started_utc=self.runtime.process_guard.now(), automatic_retry=False, automatic_resume=False,
            fixed_denominator=32, declared_cases=deepcopy(self.protocol['cases']),
            source28=self.source28, code_inventory22=self.inventory, source_inventory=self.source_inventory, arms=list(ARMS), frozen_input_refs=deepcopy(self.inputs)))
        try:
            self.phase='development_physical_recheck'
            self.recheck_development_physics()
            self.phase = 'cpu_prepare'
            self.unchanged()
            cfg_ref = self.rt.reference(BASE / 'configs/rebuild-scene-frozen-20261105-v3.json')
            cfg = self.rt.read_bound(cfg_ref)
            need(self.rt.reference(FRESH_COMPLETE['path']) == FRESH_COMPLETE, 'Frozen fresh reconstruction COMPLETE changed')
            fresh = self.rt.read_bound(FRESH_COMPLETE)
            need(fresh.get('reconstruction_source', {}).get('sha256') == RECONSTRUCTION_SHA256,
                 'Frozen fresh reconstruction source binding changed')
            self.inputs.extend([cfg_ref, deepcopy(FRESH_COMPLETE)])
            cfg.update(reconstruction_complete=self.runtime.heldout.rebuild.pilot_ref(FRESH_COMPLETE),
                purpose='frozen_evaluation', cohorts=['frozen_evaluation'], targets=['R3', 'L3'],
                parallelism=2, time_limit_seconds=10800)
            config = self.rt.write_new(CONTROL / 'scene-config.json', cfg)
            self.inputs.append(config)
            self.preparation_owner = self.launch(config, PREPARE_RUN)
            self.manifest_ref = self.rt.reference(BASE / 'runs/ausc_policy' / PREPARE_RUN / 'output/scene_manifest.json')
            self.inputs.append(self.manifest_ref)
            self.original_cases_ref = build_cases(self.runtime, self.rt.read_bound(self.manifest_ref), self.cohort_ref, CONTROL / 'original-cases')
            original_cases = self.rt.read_bound(self.original_cases_ref)
            self.protocol = dict(original_cases, arms=list(ARMS))
            for case in original_cases['cases']:
                if case['preparation_status'] == 'prepared':
                    self.inputs.extend([case['config'], self.rt.reference(CONTROL / 'original-cases/configs' / (case['scenario_id'] + '.receipt.json'))])
            self.cases_ref = project_cases(self.runtime, self.original_cases_ref, CONTROL / 'cases')
            self.inputs.extend([self.original_cases_ref, self.cases_ref])
            source_cases = self.rt.read_bound(self.cases_ref)
            self.protocol = dict(source_cases, arms=list(ARMS))
            self.inputs.extend(case['config'] for case in source_cases['cases'] if case['preparation_status'] == 'prepared')
            self.phase = 'freeze_protocol'
            self.unchanged()
            if any(case['preparation_status'] == 'prepared' for case in self.protocol['cases']):
                post = self.runtime.heldout
                self.prepared_ref = post.paired.prepare(self.developed['training']['path'], self.cases_ref['path'], CONTROL / 'paired',
                    astra_settings=deepcopy(post.paired.DEFAULT_ASTRA), parallelism=2, run_id=COMPARE_RUN, arms=ARMS)
                self.prepared, self.protocol = verify_prepared(self.runtime, self.prepared_ref, self.model, self.cases_ref)
                self.inputs.extend([self.prepared_ref, self.prepared['protocol']])
                for arm in ARMS:
                    self.inputs.append(self.prepared['batches'][arm])
                    batch = self.rt.read_bound(self.prepared['batches'][arm])
                    self.inputs.extend(self.rt.reference(job['config']) for job in batch['jobs'])
            media_ref = self.rt.write_new(CONTROL / 'MEDIA_SELECTION.json', dict(selected_before_native_outcomes=True,
                scenarios=select_media(self.protocol), arms=list(ARMS),
                policy='First CPU-prepared scenario per target; retain actual failures too'))
            self.inputs.append(media_ref)
            if self.prepared is not None:
                native_error = None
                for arm in ARMS:
                    self.phase = arm + '_native'
                    try:
                        self.owners[arm] = self.launch(self.prepared['batches'][arm], COMPARE_RUN + '-' + arm)
                    except Exception as error:
                        native_error = error
                        break
                self.phase = 'independent_qualification'
                self.qualify_all()
                summary = self.summarize(native_error)
                if native_error is not None:
                    raise native_error
            else:
                summary = self.summarize()
            self.unchanged()
            result = self.result(summary, owners_normal=self.prepared is not None and set(self.owners) == set(ARMS))
            self.rt.write_new(CONTROL / 'RESULTS.json', result)
            self.rt.write_new(CONTROL / ('COMPLETE.json' if result['acceptance']['passed'] else 'NEEDS_REPAIR.json'), result)
            return result
        except BaseException as error:
            (CONTROL / 'traceback.txt').write_text(traceback.format_exc())
            secondary_summary_error = None
            try:
                summary = self.summarize(repr(error))
            except BaseException as summary_error:
                # summarize persists its diagnostic fallback before re-raising.
                # Preserve the initiating error while completing the terminal
                # 32-case failure record from that fallback, never as a pass.
                secondary_summary_error = dict(error=repr(summary_error), traceback=traceback.format_exc(),
                    saved_fallback=self.summary_ref)
                if self.summary_ref is None:
                    self.summary_ref = self.rt.write_new(CONTROL / 'SUMMARY_FALLBACK.json',
                        unavailable_summary(self.protocol, 'Secondary summary failure: ' + repr(summary_error)))
                    secondary_summary_error['saved_fallback'] = self.summary_ref
                summary = self.rt.read_bound(self.summary_ref)
            result = self.result(summary, owners_normal=False)
            self.rt.write_new(CONTROL / 'FAILED.json', dict(result, phase=self.phase, error=repr(error),
                secondary_summary_error=secondary_summary_error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    HeldoutRound(bind_runtime()).run(args.development, args.sha256)


if __name__ == '__main__':
    main()
