"""Versioned state-only ACT optimization with separate computational sources.

Native demonstration admission and normalization remain unchanged. Planned
commands and genuine student first-cell oracle proposals are explicitly admitted
only by this optimization protocol; they never become physical demonstrations.
"""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from . import m2_given_contact_runtime as runtime
from . import m2_given_contact_expanded_pilot_train as expanded
from . import m2_act_auxiliary_data as data
from . import m2_act_recovery_scope as scope
from . import rebuild_gpu_binding as binding
from . import m2_act_chunk_data as chunk_data
from . import m2_act_posture_head as posture_head
from . import m2_act_conditional_data as conditional_data
from . import m2_act_terminal_data as terminal_data
from . import m2_act_phase_data as phase_data
from . import m2_act_corridor_sampling as corridor_sampling
from . import m2_act_measured_sampling as measured_sampling
from . import m2_measured_recovery_data as measured_data
from . import m2_act_diverse_sampling as diverse_sampling
from . import m2_act_refinement_sampling as refinement_sampling
from . import m2_act_task_selection as task_selection

need=runtime.need
CONFIG_SCHEMA='m2_act_auxiliary_training_config_v1'
PILOT_SCHEMA='m2_act_auxiliary_act_pilot_v1'
SELECTION_SCHEMA='m2_act_auxiliary_act_selection_v1'
CPU_SCHEMA='m2_act_auxiliary_act_cpu_equivalence_v1'
PROTOCOL_SCHEMA='m2_act_auxiliary_act_protocol_v1'
CONFIG_SCHEMA_V2='m2_act_auxiliary_training_config_v2'
PILOT_SCHEMA_V2='m2_act_auxiliary_act_pilot_v2'
SELECTION_SCHEMA_V2='m2_act_auxiliary_act_selection_v2'
CPU_SCHEMA_V2='m2_act_auxiliary_act_cpu_equivalence_v2'
PROTOCOL_SCHEMA_V2='m2_act_auxiliary_act_protocol_v2'
CONFIG_SCHEMA_V3='m2_act_auxiliary_training_config_v3'
PILOT_SCHEMA_V3='m2_act_auxiliary_act_pilot_v3'
SELECTION_SCHEMA_V3='m2_act_auxiliary_act_selection_v3'
CPU_SCHEMA_V3='m2_act_auxiliary_act_cpu_equivalence_v3'
PROTOCOL_SCHEMA_V3='m2_act_auxiliary_act_protocol_v3'
CONFIG_SCHEMA_V4='m2_act_auxiliary_training_config_v4'
PILOT_SCHEMA_V4='m2_act_auxiliary_act_pilot_v4'
SELECTION_SCHEMA_V4='m2_act_auxiliary_act_selection_v4'
CPU_SCHEMA_V4='m2_act_auxiliary_act_cpu_equivalence_v4'
PROTOCOL_SCHEMA_V4='m2_act_auxiliary_act_protocol_v4'
CONFIG_SCHEMA_V5='m2_act_auxiliary_training_config_v5'
PILOT_SCHEMA_V5='m2_act_auxiliary_act_pilot_v5'
SELECTION_SCHEMA_V5='m2_act_auxiliary_act_selection_v5'
CPU_SCHEMA_V5='m2_act_auxiliary_act_cpu_equivalence_v5'
PROTOCOL_SCHEMA_V5='m2_act_auxiliary_act_protocol_v5'
CONFIG_SCHEMA_V6='m2_act_auxiliary_training_config_v6'
PILOT_SCHEMA_V6='m2_act_auxiliary_act_pilot_v6'
SELECTION_SCHEMA_V6='m2_act_auxiliary_act_selection_v6'
CPU_SCHEMA_V6='m2_act_auxiliary_act_cpu_equivalence_v6'
PROTOCOL_SCHEMA_V6='m2_act_auxiliary_act_protocol_v6'
CONFIG_SCHEMA_V7='m2_act_auxiliary_training_config_v7'
PILOT_SCHEMA_V7='m2_act_auxiliary_act_pilot_v7'
SELECTION_SCHEMA_V7='m2_act_auxiliary_act_selection_v7'
CPU_SCHEMA_V7='m2_act_auxiliary_act_cpu_equivalence_v7'
PROTOCOL_SCHEMA_V7='m2_act_auxiliary_act_protocol_v7'
CONFIG_SCHEMA_V8='m2_act_auxiliary_training_config_v8'
PILOT_SCHEMA_V8='m2_act_auxiliary_act_pilot_v8'
SELECTION_SCHEMA_V8='m2_act_auxiliary_act_selection_v8'
CPU_SCHEMA_V8='m2_act_auxiliary_act_cpu_equivalence_v8'
PROTOCOL_SCHEMA_V8='m2_act_auxiliary_act_protocol_v8'
CONFIG_SCHEMA_V9='m2_act_auxiliary_training_config_v9'
PILOT_SCHEMA_V9='m2_act_auxiliary_act_pilot_v9'
SELECTION_SCHEMA_V9='m2_act_auxiliary_act_selection_v9'
CPU_SCHEMA_V9='m2_act_auxiliary_act_cpu_equivalence_v9'
PROTOCOL_SCHEMA_V9='m2_act_auxiliary_act_protocol_v9'
INITIALIZATION_V6=dict(path='/opt/wam-runtime/fast-wam/diagnostics/act7-initialization-20261107-v1/INITIALIZATION.json',
    bytes=47630,sha256='0718b63313a596e4504df95daea09c05988ab605ff5e525b319d850c712c9bc0')
INITIALIZATION_V7=deepcopy(INITIALIZATION_V6)
TRAIN16_SCOPE=dict(path='/opt/wam-runtime/fast-wam/diagnostics/act-auxiliary-cohort-20261106-v2/COHORT_FREEZE.json',
    bytes=1521,sha256='adf6b0bd95669719989f6795c177f0a8419921dd79a257ca0c725f99b3220e71')
SCHEDULE=deepcopy(expanded.SCHEDULE)
SCHEDULE_V5=dict(SCHEDULE,steps=1000)
FINAL_SELECTION_V5=dict(method="fixed_final_budget",selected_step=1000,validation_used_for_selection=False,diagnostic_steps=[500],new_optimizer_updates=1000)
SCHEDULE_V6=deepcopy(SCHEDULE_V5)
FINAL_SELECTION_V6=deepcopy(FINAL_SELECTION_V5)
SCHEDULE_V7=deepcopy(SCHEDULE_V6)
FINAL_SELECTION_V7=deepcopy(FINAL_SELECTION_V6)
SCHEDULE_V8=deepcopy(SCHEDULE)
SCHEDULE_V9=deepcopy(SCHEDULE)
SELECTION_POLICY_V9=dict(method='earliest_exact_minimum_observed_validation_TCP_task_score',
    candidate_steps=list(task_selection.CANDIDATE_STEPS),new_optimizer_updates=20000,
    metric=deepcopy(task_selection.METRIC),subset_rule=task_selection.RULE,
    raw_joint_mae='diagnostic_only',known4_used=False,fresh_used=False,physical_outcomes_used=False)
SELECTION_POLICY_V8=dict(method='earliest_minimum_full_native_validation_masked_raw_H20_MAE',
    validation_used_for_selection=True,new_optimizer_updates=20000,evaluation_steps=list(range(500,20001,500)),
    evaluated_populations=dict(train=8373,validation=2345),all_candidates_retained=True)
FLAGS=deepcopy(expanded.FLAGS)
NATIVE_DATASET=dict(path='/opt/wam-runtime/fast-wam/datasets/act-student-recovery-masked-20261106-v1/COMPLETE.json',
    bytes=164297,sha256='265495e293b4ba95ca779cdc9693762716a3ea0010c6430ad2fde3b456f30cee')
NATIVE_COUNTS=dict(train=8373,validation=2345)
OFFICIAL_FORWARD_SHA256='83d954f79ccd3eaa6774dbea92524939c8ac08359bb355ba7cd37b8601e1e3af'
SOURCE20_MANIFEST=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v20/SOURCE_MANIFEST.json',
    bytes=282448,sha256='34a722d283db0dfa78718e037b2473739eaf066fc5c455343b8740b7b929c844')
SOURCE20_GENERATOR=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v20/sim/ausc_policy/m2_student_oracle_proposals.py',
    bytes=11637,sha256='b507ed21c339d2ca371ccb9b08472aeca4a1de2abb89080b45cf9ac1fb594e5e')
SOURCE20_REQUEST_SHA256='2ffcf68f41d1ffd634d4e97568b831976c71653394bf2eb61c9104eae5bd470d'
GROUPS=dict(native=dict(size=32,coefficient=.50),planned=dict(size=16,coefficient=.25),oracle=dict(size=16,coefficient=.25))
GROUPS_V3=dict(native=dict(size=32,coefficient=.50),planned=dict(size=8,coefficient=.125),
    conditional=dict(size=8,coefficient=.125),oracle=dict(size=16,coefficient=.25))


def _version(value):
    schema=value.get('schema')
    for version,schemas in (
        (9,(CONFIG_SCHEMA_V9,PILOT_SCHEMA_V9,SELECTION_SCHEMA_V9,CPU_SCHEMA_V9,PROTOCOL_SCHEMA_V9)),
        (8,(CONFIG_SCHEMA_V8,PILOT_SCHEMA_V8,SELECTION_SCHEMA_V8,CPU_SCHEMA_V8,PROTOCOL_SCHEMA_V8)),
        (7,(CONFIG_SCHEMA_V7,PILOT_SCHEMA_V7,SELECTION_SCHEMA_V7,CPU_SCHEMA_V7,PROTOCOL_SCHEMA_V7)),
        (6,(CONFIG_SCHEMA_V6,PILOT_SCHEMA_V6,SELECTION_SCHEMA_V6,CPU_SCHEMA_V6,PROTOCOL_SCHEMA_V6)),
        (5,(CONFIG_SCHEMA_V5,PILOT_SCHEMA_V5,SELECTION_SCHEMA_V5,CPU_SCHEMA_V5,PROTOCOL_SCHEMA_V5,phase_data.SCHEMA)),
        (4,(CONFIG_SCHEMA_V4,PILOT_SCHEMA_V4,SELECTION_SCHEMA_V4,CPU_SCHEMA_V4,PROTOCOL_SCHEMA_V4,terminal_data.SCHEMA)),
        (3,(CONFIG_SCHEMA_V3,PILOT_SCHEMA_V3,SELECTION_SCHEMA_V3,CPU_SCHEMA_V3,PROTOCOL_SCHEMA_V3,conditional_data.SCHEMA)),
        (2,(CONFIG_SCHEMA_V2,PILOT_SCHEMA_V2,SELECTION_SCHEMA_V2,CPU_SCHEMA_V2,PROTOCOL_SCHEMA_V2,chunk_data.SCHEMA)),
        (1,(CONFIG_SCHEMA,PILOT_SCHEMA,SELECTION_SCHEMA,CPU_SCHEMA,PROTOCOL_SCHEMA,data.SCHEMA))):
        if schema in schemas:return version
    need(schema is None,'Explicit supported auxiliary schema/version required')
    return 1  # legacy untyped internal arrays/protocol callers only


def selected_step(rows,*,version):
    _groups(version)
    if version==9:
        for row in rows:
            for split,count in NATIVE_COUNTS.items():need(runtime.validate_metrics(row.get(split,{}))['sample_count']==count,'V9 requires full unchanged native TRAIN/VAL diagnostics')
        return task_selection.select_candidate(rows)['step']
    if version==8:
        need([row.get('step') for row in rows]==list(range(500,20001,500)), 'V8 requires all40 ordered full native evaluations at500 through20000')
        for row in rows:
            for split,count in NATIVE_COUNTS.items():need(runtime.validate_metrics(row.get(split,{}))['sample_count']==count,'V8 selection requires every full original native population')
        return expanded.selected_step(rows)
    if version<5:return expanded.selected_step(rows)
    need([row.get('step') for row in rows]==[500,1000],'V5 requires exactly two new evaluations at500/1000')
    return 1000


def _warm_start(version):
    _groups(version)
    if version==9:
        from . import m2_act_refinement_warm_start as module
    elif version==8:
        from . import m2_act_diverse_warm_start as module
    elif version>=5:
        from . import m2_act_phase_warm_start as module
    else:
        from . import m2_act_warm_start as module
    return module


def _api(version):
    need(type(version) is int and version in (1,2,3,4,5,6,7,8,9),'Explicit supported auxiliary data version required')
    return {1:data,2:chunk_data,3:conditional_data,4:terminal_data,5:phase_data,6:phase_data,7:phase_data,8:phase_data,9:phase_data}[version]


def _groups(version):
    need(type(version) is int and version in (1,2,3,4,5,6,7,8,9),'Explicit supported optimizer version required')
    return GROUPS_V3 if version>=3 else GROUPS


def _kinds(version):
    return conditional_data.KINDS if version>=3 else data.KINDS


def _v2(value):
    return value.get('schema') in (CONFIG_SCHEMA_V2,PILOT_SCHEMA_V2,SELECTION_SCHEMA_V2,chunk_data.SCHEMA)


def _schemas(value):
    if _version(value)==9:return PILOT_SCHEMA_V9,SELECTION_SCHEMA_V9,CPU_SCHEMA_V9,PROTOCOL_SCHEMA_V9
    if _version(value)==8:return PILOT_SCHEMA_V8,SELECTION_SCHEMA_V8,CPU_SCHEMA_V8,PROTOCOL_SCHEMA_V8
    if _version(value)==7:return PILOT_SCHEMA_V7,SELECTION_SCHEMA_V7,CPU_SCHEMA_V7,PROTOCOL_SCHEMA_V7
    if _version(value)==6:return PILOT_SCHEMA_V6,SELECTION_SCHEMA_V6,CPU_SCHEMA_V6,PROTOCOL_SCHEMA_V6
    if _version(value)>=5:return PILOT_SCHEMA_V5,SELECTION_SCHEMA_V5,CPU_SCHEMA_V5,PROTOCOL_SCHEMA_V5
    if _version(value)>=4:return PILOT_SCHEMA_V4,SELECTION_SCHEMA_V4,CPU_SCHEMA_V4,PROTOCOL_SCHEMA_V4
    if _version(value)==3:return PILOT_SCHEMA_V3,SELECTION_SCHEMA_V3,CPU_SCHEMA_V3,PROTOCOL_SCHEMA_V3
    return (PILOT_SCHEMA_V2,SELECTION_SCHEMA_V2,CPU_SCHEMA_V2,PROTOCOL_SCHEMA_V2) if _v2(value) else (PILOT_SCHEMA,SELECTION_SCHEMA,CPU_SCHEMA,PROTOCOL_SCHEMA)


def make_model(config,*,device,training=False):
    _require_head_version(config)
    torch,policy,software=runtime.make_model({'lerobot_root':config['lerobot_root']},device=device)
    if _version(config)>=2:
        posture_head.install(policy,config['posture_head'])
        if training and _version(config)<4:posture_head.initialize_training(policy,config['posture_head'])
    return torch,policy,software


def _require_head_version(config):
    version=_version(config)
    if version>=2:
        spec=config.get('posture_head',{});posture_head.validate_spec(spec)
        need(spec['schema']==(posture_head.SCHEMA_V2 if version>=3 else posture_head.SCHEMA),
            'Model/config head version mismatch; legacy weights cannot be relabeled')


def auxiliary_metadata(ref,*,version=1):
    """Reject identity/profile/eligibility faults before any numerical loader."""
    api=_api(version)
    if version>=5:
        _,metadata=phase_data.metadata(ref,scope.TRAIN16_IDS)
        need(metadata.get('input_profile')=='production_closed_inputs','V5 production trainer rejects local fixture profiles')
        need(metadata.get('optimization_admission')==phase_data._admission(True),'Complete four-origin V5 optimization admission required')
        need(all(s.get('pose_id') in runtime_scope_poses() and s.get('target_id') in ('R3','L3') for s in metadata['sources']) and
            {s['kind'] for s in metadata['sources']}==set(_kinds(version)), 'Known-pose TRAIN computational groups required')
        return metadata
    if version==4:
        _,metadata=terminal_data.metadata(ref,scope.TRAIN16_IDS)
        need(all(s.get('pose_id') in runtime_scope_poses() and s.get('target_id') in ('R3','L3') for s in metadata['sources']) and
            {s['kind'] for s in metadata['sources']}==set(_kinds(version)),
            'All original separate known-pose TRAIN computational kinds required')
        return metadata
    if version==3:
        _,metadata=conditional_data.metadata(ref,scope.TRAIN16_IDS)
        need(all(s.get('pose_id') in runtime_scope_poses() and s.get('target_id') in ('R3','L3') for s in metadata['sources']),
             'Explicit known-pose R3/L3 computational sources required')
        need({s['kind'] for s in metadata['sources']}==set(conditional_data.KINDS),
             'All three separate computational source kinds required')
        current={s['target_id'] for s in metadata['sources'] if s['schema']==conditional_data.SOURCE_SCHEMA and s['kind']=='student_oracle_unexecuted'}
        need(current=={'R3','L3'},'New genuine source24 TRAIN oracle queries required for both R3 and L3')
        return metadata
    expanded.full_reference(ref);complete=runtime.read_bound(ref)
    need(complete.get('schema')==api.SCHEMA and complete.get('complete') is True and
         all(complete.get(k) is v for k,v in data.FLAGS.items()),'Separate unexecuted auxiliary preparation required')
    metadata=runtime.read_bound(complete['metadata'])
    binding.validate_auxiliary_dataset(metadata,schema=api.SOURCE_SCHEMA,train_pool_ids=scope.TRAIN16_IDS)
    for source in metadata['sources']:
        kind=source['kind']
        need(kind in data.KINDS and source.get('pose_id') in runtime_scope_poses() and
             source.get('target_id') in ('R3','L3'),'Explicit known-pose R3/L3 computational source required')
        if kind=='planned_ideal_tracking':
            need(source.get('verification_profile')=='original_full_cpu_preview' and source.get('injected_factories')==[],
                 'Production original_full_cpu_preview required; injected CPU fixture profiles rejected')
        else:
            need(source.get('observation_origin')==data.ORIGINS[kind] and
                 source.get('source_episode_verdict_unchanged') is True and source.get('original_source_success') is False and
                 {'generator','native','owner','child','proposal','outcome','request','diagnostic','diagnostic_root',
                  'frozen_request','frozen_manifest','original_arrays','source','geometry'}<=set(source.get('bindings',{})),
                 'Genuine frozen authenticated negative TRAIN student oracle source required')
            if version==2:
                chunk_data.require_frozen_archive(source['bindings']['frozen_manifest'],source['bindings']['generator'])
            else:_require_source20(source['bindings'])
    need({s['kind'] for s in metadata['sources']}==set(data.KINDS),'Both planned and student oracle sources are required')
    if version==2:
        need(complete.get('optimization_admission')==chunk_data.ADMISSION and metadata.get('optimization_admission')==chunk_data.ADMISSION,
             'Explicit full computed prefix optimization admission required')
        chunk_data.require_current_targets(metadata['sources'])
    return metadata


def runtime_scope_poses():
    return ('p000474_a001230_f0000','p000476_a001231_f0023')


def validate_config(config):
    version=_version(config) if isinstance(config,dict) else 1
    v2=version>=2
    optional={'replacement_pod_gpu_binding'}&set(config) if isinstance(config,dict) else set()
    if v2:optional=optional|{'posture_head'}
    schedule=SCHEDULE_V9 if version==9 else SCHEDULE_V8 if version==8 else SCHEDULE_V7 if version==7 else SCHEDULE_V6 if version==6 else SCHEDULE_V5 if version==5 else SCHEDULE
    required={'schema','base','dataset','auxiliary','evaluation_cohort'}|({'initialization'} if version>=4 else set())|({'measured_bank'} if version==7 else {'measured_banks'} if version in (8,9) else set())|({'task_subset'} if version==9 else set())
    need(isinstance(config,dict) and set(config)==set(schedule)|required|optional and
         config.get('schema') in (CONFIG_SCHEMA,CONFIG_SCHEMA_V2,CONFIG_SCHEMA_V3,CONFIG_SCHEMA_V4,CONFIG_SCHEMA_V5,CONFIG_SCHEMA_V6,CONFIG_SCHEMA_V7,CONFIG_SCHEMA_V8,CONFIG_SCHEMA_V9) and all(config.get(k)==v and type(config[k]) is type(v) for k,v in schedule.items()) and
         Path(str(config.get('base',''))).is_absolute(),'Exact finite versioned 20k or V5 1000/64/500 auxiliary optimization schedule required; no resume')
    for key in ('dataset','auxiliary','evaluation_cohort')+ (('measured_bank',) if version==7 else ()):expanded.full_reference(config[key])
    need(config['dataset']==NATIVE_DATASET,'Exact immutable native recovery29 bank required')
    # Read metadata only. No native or auxiliary numerical arrays before overlap checks.
    cohorts=scope.validate(config['evaluation_cohort']);frozen=runtime.read_bound(config['evaluation_cohort'])
    need(frozen.get('schema')==scope.TRAIN16_SCHEMA,'Explicit new TRAIN16 scope required; legacy TRAIN4 admission is separate')
    if v2:
        need(config['evaluation_cohort']==TRAIN16_SCOPE,'Pinned accepted TRAIN16 pretraining scope required')
        if version>=3:posture_head.validate_bound_spec(config['posture_head'],native_normalization(config),version=2)
        else:posture_head.validate_bound_spec(config['posture_head'],native_normalization(config))
    if version>=4:
        if version==7:need(config['initialization']==INITIALIZATION_V7,'V7 requires pinned ACT6 selected18000 initialization; no ACT7/ACT8 restart')
        if version==6:need(config['initialization']==INITIALIZATION_V6,'V6 requires pinned ACT6 selected18000 CPU initialization; ACT7 weights cannot initialize ACT8')
        warm_start=_warm_start(version)
        expanded.full_reference(config['initialization'])
        warm_start.validate_receipt(config['initialization'],normalization=native_normalization(config),posture_head=config['posture_head'])
    auxiliary=auxiliary_metadata(config['auxiliary'],version=version) if v2 else auxiliary_metadata(config['auxiliary'])
    native=runtime.read_bound(config['dataset'])
    binding.validate_dataset(native,schema=expanded.MASKED_PREPARATION_SCHEMA,task_contract=runtime.task_contract())
    need(native.get('counts')==NATIVE_COUNTS,'Unchanged native TRAIN8373/selection-VAL2345 population required')
    if version>=6:corridor_sampling.native_metadata(native,allowed_train_ids=scope.TRAIN16_IDS)
    if version in (8,9):
        sampler=refinement_sampling if version==9 else diverse_sampling
        admitted=sampler.admitted_metadata(config['measured_banks'],dataset=config['dataset'],normalization=native['normalization'])
    if version==9:_validate_task_subset(config)
    if version==7:
        _,measured=measured_sampling.admitted_metadata(config['measured_bank'],dataset=config['dataset'],normalization=native['normalization'])
    actual=set(native['split_subject_ids']['train'])|{s['subject_id'] for s in auxiliary['sources']}
    if version==7:actual|={s['metadata']['subject_id'] for s in measured['sources']}
    if version in (8,9):actual|={s['metadata']['subject_id'] for bank in admitted.values() for s in bank['metadata']['sources']}
    scope.validate_train16_metadata(frozen,actual_train_ids=actual)
    need(not actual&set(cohorts['selection_validation']+cohorts['frozen_evaluation']),
         'Auxiliary/native optimizer overlaps selection or fresh evaluation identities')
    if 'replacement_pod_gpu_binding' in optional:binding.validate(config['replacement_pod_gpu_binding'],base=config['base'])
    return config


def _validate_task_subset(config,*,samples=None):
    """Pin the independent pre-optimizer subset, including teacher/kernel bytes."""
    ref=config.get('task_subset');expanded.full_reference(ref)
    path=Path(ref['path']);need(path==path.resolve(),'Canonical frozen task subset path required')
    refinement_sampling.data._regular_unaliased(path)
    subset=runtime.read_bound(ref);native=runtime.read_bound(config['dataset'])
    need(subset.get('schema')==task_selection.SUBSET_SCHEMA and subset.get('dataset')==config['dataset'] and
        subset.get('index')==native['index'] and subset.get('normalization')==native['normalization'] and
        subset.get('validation_arrays')==native['arrays']['validation'] and subset.get('metric')==task_selection.METRIC and
        subset.get('selection_rule')==task_selection.RULE and subset.get('selector_source')==runtime.reference(task_selection.__file__) and
        subset.get('geometry_sources')==task_selection._geometry_sources(), 'Exact pre-optimizer observed subset/data/FK source binding required')
    runtime.bound_path(subset['teacher_cache'])
    if samples is not None:task_selection._bound_subset(ref,samples)
    return subset


def training_protocol(config):
    _require_head_version(config)
    result=dict(schema=_schemas(config)[3],schedule=deepcopy(config),model=runtime.shared.native_spec(),parameter_count=2658823,
        state_only=True,observation_dimensions={'state':7,'environment':27,'total':34},raw_state_dimensions=17,
        history_frames=2,action_horizon=20,execution_prefix=10,model_dropout=.1,task_contract=runtime.task_contract(),
        evaluation_steps=expanded.evaluation_steps(config),selection='earliest minimum full native selection-VAL2345 raw masked MAE',
        label_protocol=runtime.shared.MASKED_LABEL_PROTOCOL,normalization='pinned real native TRAIN only; no auxiliary refit',
        objective={name:dict(**value,loss='official forward masked valid-cell/joint mean') for name,value in _groups(_version(config)).items()},
        native_sampling='half real near60mm origins, half uniform real origins',
        auxiliary_sampling='separate kind then uniform body -> pose -> target -> window',
        auxiliary_admission=dict(schema=binding.AUXILIARY_ADMISSION_SCHEMA,preparation_schema=data.SCHEMA,
            computational_optimization_eligible=True,**data.FLAGS),
        augmentation={'q_noise_std':0.,'q_dropout':0.},gradient_clip_norm=1.,gradient_clip_error_if_nonfinite=True,
        action_clipping=False,
        selection_auxiliary_used=False,selection_fresh_used=False,official_forward_sha256=OFFICIAL_FORWARD_SHA256,**FLAGS)
    if _version(config)>=2:
        result.update(posture_head=deepcopy(config['posture_head']),
            posture_initialization=posture_head.initialization_spec(config['posture_head']),head_active_in_loss_and_metrics=True,
            model_definition='official ACT backbone with deterministic smooth absolute posture-box output parameterization',
            oracle_supervision='full computed prefix up to20; genuine current H2 only; no future measured histories')
        result['auxiliary_admission']=dict(chunk_data.ADMISSION,preparation_schema=chunk_data.SCHEMA)
    if _version(config)>=3:
        result['auxiliary_admission']=dict(conditional_data.ADMISSION,preparation_schema=conditional_data.SCHEMA)
        result['auxiliary_sampling']='uniform body -> pose -> target -> anchor-or-window -> perturbation; groups separately sampled'
        result['conditional_supervision']='generated static ideal H2/q/v=a=0; canonical L2 continuation; no actual/future measurements'
        result['treatment']='combined conditional/genuine coverage, STOP residual adapter and head v2 softness; not a causal ablation'
    if _version(config)==4:
        from . import m2_act_warm_start as warm_start
        result['posture_initialization']=warm_start.protocol(config['initialization'])
        result['auxiliary_admission']=dict(terminal_data.ADMISSION,preparation_schema=terminal_data.SCHEMA)
        result['treatment']='fine-tuned ACT6: ACT5 selected19000 initialization plus 20000 new optimizer steps and terminal-query data; not a causal data-only ablation'
    if _version(config)>=5:
        result.update(selection='fixed final1000 new updates regardless of validation;500 diagnostic only',
            selection_policy=deepcopy(FINAL_SELECTION_V5),posture_initialization=_warm_start(_version(config)).protocol(config['initialization']),
            auxiliary_admission=dict(phase_data._admission(True),preparation_schema=phase_data.SCHEMA),
            conditional_sampling=dict(terminal_draws=4,full_tree_draws=4,threshold_m=.006,
                coordinate_error='norm([px,py,pz+source_canonical_gap])',balance='body -> pose -> target -> anchor -> window',
                missing_strata='record available inventory; fail only empty conditional terminal pool'),
            treatment='fine-tuned ACT7: ACT6 selected18000 initialization plus exactly1000 new optimizer updates and phase-query data')
    if _version(config) in (6,7):
        initial=result['posture_initialization']
        initial['method']=initial['method'].replace('ACT7','ACT8')
        initial['original_evaluations']=initial['original_evaluations'].replace('ACT7','ACT8')
        initial.update(historical_helper='m2_act_phase_warm_start',
            historical_equivalence_schema=_warm_start(6).EQUIVALENCE_SCHEMA,historical_private_attribute='_act7_initialization')
        result.update(data_version=5,model_optimizer_version=6,data_schema=phase_data.SCHEMA,
            native_sampling='first16 uniform eligible actual-source PREPOSE error<=12mm; other16 original uniform all TRAIN',
            planned_sampling=dict(corridor_draws=4,full_tree_draws=4,threshold_m=.012,balance='body -> pose -> target -> window'),
            treatment='ACT8 sampler-only: ACT6 selected18000 initialization; exactly1000 new updates; native/planned 12mm corridor sampling')
    if _version(config)==7:
        initial=result['posture_initialization']
        for key in ('method','original_evaluations'):initial[key]=initial[key].replace('ACT8','ACT9')
        measured_complete=runtime.read_bound(config['measured_bank'])
        need(measured_complete.get('complete') is True and measured_complete.get('schema') in
            (measured_data.SCHEMA,measured_data.SCHEMA_V2),'Validated measured bank COMPLETE schema required')
        result.update(model_optimizer_version=7,measured_bank=deepcopy(config['measured_bank']),
            measured_data_schema=measured_complete['schema'],
            native_sampling='8 Base29 actual PREPOSE<=12mm;8 full Base29;8 measured actual PREPOSE<=12mm;8 full measured',
            measured_sampling='target -> frozen slot -> window; fail empty target pool; record missing corridor slots',
            native_gradient_weighting='one original native32 valid-cell mean; equal examples are not equal gradient shares',
            treatment='ACT9 measured recovery sampler: ACT6 selected18000 initialization; exactly1000 new updates; separate all4 measured TRAIN supplement')
    if _version(config)==8:
        result.update(data_version=5,model_optimizer_version=8,data_schema=phase_data.SCHEMA,
            measured_banks=deepcopy(config['measured_banks']),selection='earliest minimum full native selection-VAL2345 H20 masked raw target MAE',
            selection_policy=deepcopy(SELECTION_POLICY_V8),
            native_sampling='8 Base29 actual PREPOSE<=12mm;8 full Base29;8 measured actual PREPOSE<=12mm;8 full measured',
            measured_sampling='target -> body -> available pose -> source slot -> window; both admitted immutable banks',
            planned_sampling=dict(corridor_draws=4,full_tree_draws=4,threshold_m=.012,balance='body -> pose -> target -> window'),
            native_gradient_weighting='one original native32 valid-cell mean; equal examples are not equal gradient shares',
            treatment='combined measured diversity, balanced measured sampling, ACT9 warm start and20000 new updates; not a causal ablation')
    if _version(config)==9:
        subset=_validate_task_subset(config)
        result.update(data_version=5,model_optimizer_version=9,data_schema=phase_data.SCHEMA,
            measured_banks=deepcopy(config['measured_banks']),
            measured_bank_schemas={name:dict(schema=profile[0],profile=profile[1]) for name,profile in refinement_sampling.PROFILES.items()},
            task_subset=deepcopy(config['task_subset']),teacher_cache=deepcopy(subset['teacher_cache']),
            scoring_kernel=deepcopy(subset['geometry_sources']),selector_source=deepcopy(subset['selector_source']),
            selection='earliest exact minimum task score from all40 saved canonical Official validation predictions',
            selection_policy=deepcopy(SELECTION_POLICY_V9),
            native_sampling='8 Base29 actual PREPOSE<=12mm;8 full Base29;8 measured actual PREPOSE<=12mm;8 full measured',
            measured_sampling='target -> body -> available pose -> bank/source slot -> window; three distinct admitted TRAIN banks',
            planned_sampling=dict(corridor_draws=4,full_tree_draws=4,threshold_m=.012,balance='body -> pose -> target -> window'),
            native_gradient_weighting='one original native32 valid-cell mean; equal examples are not equal gradient shares',
            optimizer=dict(name='AdamW',initial_step=0,lr=1e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8),
            attention_heads=4,raw_joint_mae='full original TRAIN/VAL diagnostic only',
            treatment='ACT11 V9: exact ACT10 Source41 selected17000 initialization;20000 fresh updates; three-bank TRAIN refinement and frozen observed task selection')
    return result


def code_binding(*,version=1):
    _groups(version)
    value=expanded.code_binding();root=Path(__file__).resolve().parent
    for name in ('m2_act_auxiliary_train.py','m2_act_auxiliary_data.py','m2_act_planned_supervision.py',
                 'm2_given_contact_recovery.py','m2_student_oracle_proposals.py','m2_student_recovery_prefix.py'):
        value['files'][name]=runtime.reference(root/name)
    if version>=2:
        for name in ('m2_act_chunk_data.py','m2_act_posture_head.py'):
            value['files'][name]=runtime.reference(root/name)
    if version>=3:
        for name in ('m2_act_conditional_data.py','m2_act_conditional_supervision.py'):
            value['files'][name]=runtime.reference(root/name)
    if version>=4:
        for name in ('m2_act_terminal_data.py','m2_act_frozen26_data.py','m2_act_warm_start.py'):
            value['files'][name]=runtime.reference(root/name)
    if version>=5:
        for name in ('m2_act_phase_data.py','m2_act_phase_warm_start.py'):
            value['files'][name]=runtime.reference(root/name)
    if version>=6:value['files']['m2_act_corridor_sampling.py']=runtime.reference(root/'m2_act_corridor_sampling.py')
    if version==7:
        for name,ref in measured_data.code_binding().items():value['files'][Path(ref['path']).name]=ref
        value['files']['m2_act_measured_sampling.py']=runtime.reference(root/'m2_act_measured_sampling.py')
    if version==8:
        for api in (measured_data,diverse_sampling.dart):
            for name,ref in api.code_binding().items():value['files'][Path(ref['path']).name]=ref
        for name in ('m2_act_diverse_sampling.py','m2_act_diverse_warm_start.py','m2_act_measured_sampling.py'):
            value['files'][name]=runtime.reference(root/name)
    if version==9:
        for api in (measured_data,diverse_sampling.dart,refinement_sampling.data):
            for name,ref in api.code_binding().items():value['files'][Path(ref['path']).name]=ref
        for name in ('m2_act_refinement_sampling.py','m2_act_refinement_warm_start.py','m2_act_task_selection.py',
                     'm2_act_measured_sampling.py','m2_given_contact_pilot_native_audit.py','m2_given_contact_recovery_dataset.py',
                     'expert_capture.py','training_capture.py'):
            value['files'][name]=runtime.reference(root/name)
        for ref in task_selection._geometry_sources().values():value['files'][Path(ref['path']).name]=ref
    return value


def validate_code(value,*,version=1):
    current=code_binding(version=version)
    if version>=4:
        need(value==current,'Current V4 numerical source origins/commit/bytes changed')
        return
    need(set(value.get('files',{}))==set(current['files']) and all(
         {k:ref[k] for k in ('bytes','sha256')}=={k:current['files'][name][k] for k in ('bytes','sha256')}
         for name,ref in value['files'].items()),'Mixed-source trained numerical/protocol source bytes changed')


def validate_official_forward(software):
    ref=software.get('vendor',{}).get('files',{}).get('src/lerobot/policies/act/modeling_act.py',{})
    need(ref.get('sha256')==OFFICIAL_FORWARD_SHA256,'Pinned official masked valid-cell mean forward bytes required')


def _require_source20(refs):
    need(refs.get('frozen_manifest')==SOURCE20_MANIFEST and refs.get('generator')==SOURCE20_GENERATOR and
         refs.get('frozen_request',{}).get('sha256')==SOURCE20_REQUEST_SHA256,
         'Pinned actual frozen source20 manifest/generator/request required; arbitrary archives cannot self-authenticate')


def native_normalization(config):
    native=runtime.read_bound(config['dataset'])
    return runtime.validate_normalization(runtime.read_bound(native['normalization']))


def validate_oracle_sample(samples,source_index,proposal,authenticated_query):
    """Resealing an export cannot substitute genuine H2 or first-cell labels."""
    source=samples['sources'][source_index];row=source['frozen_physics_row']
    need(proposal.get('query')==authenticated_query,'Oracle query differs from genuine authenticated student input')
    history,times,relative,_=data._oracle_query(proposal,source['metadata'],row)
    ids=np.flatnonzero(samples['source_indices']==source_index)
    need(len(ids)==1,'Each genuine oracle source must retain its single actual query')
    i=int(ids[0]);mask=samples['setter_is_pad'][i]
    need(np.array_equal(samples['source_histories'][i],history) and np.array_equal(samples['source_times'][i],times) and
         np.array_equal(samples['relative_setters'][i,0],relative[0]) and not mask[0] and np.all(mask[1:]) and
         np.all(samples['relative_setters'][i,1:]==0),
         'Oracle exported arrays must equal genuine H2 and the bound first raw setter only')


def authenticate_oracles(auxiliary):
    """Authenticate genuine queries once in their original frozen source20 code.

    Original model admission remains entirely in that immutable code archive.
    Only actual pre-query H2/current origin is compared; no future measurement
    or physical-success label is invented by the computational oracle.
    """
    version=_version(auxiliary.get('samples',{}));v2=version>=2
    if version>=5:
        provenance=auxiliary['provenance']
        need(provenance.get('input_profile')=='production_closed_inputs' and
            provenance.get('parent_admission',{}).get('complete')==phase_data.PRODUCTION_INPUTS.parent_complete and
            provenance.get('producer_admission',{}).get('producer')==phase_data.PRODUCTION_INPUTS.producer_complete,
            'Inherited sealed V4/source28 phase-query admission required')
        return [dict(actual_history_authenticated=True,future_measurements_claimed=False,
            parent_admission=deepcopy(provenance['parent_admission']),producer_admission=deepcopy(provenance['producer_admission']))]
    if version==4:
        # load_preparation has just rerun ALL original parent admission in26.
        # Its historical source20/23/24 paths must never be read by a28 V3 API.
        provenance=auxiliary['provenance'];proof=provenance['parent_archive_validation']
        need(provenance.get('parent_complete')==terminal_data.PARENT_COMPLETE and
            proof.get('manifest')==terminal_data.frozen26.SOURCE26_MANIFEST and proof.get('array_proofs'),
            'Full original26 parent authentication required before new V4 queries')
        parent=dict(parent_complete=terminal_data.PARENT_COMPLETE,manifest=terminal_data.frozen26.SOURCE26_MANIFEST,
            parent_original26_validation=deepcopy(proof),actual_history_authenticated=True,future_measurements_claimed=False)
        return [parent]+terminal_data.authenticate_new_genuine_groups(auxiliary,scope.TRAIN16_IDS)
    groups={}
    for source_index,source in enumerate(auxiliary['sources']):
        if source['kind']!='student_oracle_unexecuted':continue
        refs=source['bindings']
        if version==3:conditional_data.require_oracle_archive(source)
        elif v2:chunk_data.require_frozen_archive(refs['frozen_manifest'],refs['generator'])
        else:_require_source20(refs)
        request=runtime.read_bound(refs['request'])
        key=runtime.digest(dict(source=request['source'],manifest=refs['frozen_manifest']))
        groups.setdefault(key,dict(refs=refs,request=request,sources=[]))['sources'].append((source_index,source))
    program='''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from sim.ausc_policy import m2_student_oracle_proposals as original
assert Path(original.__file__).resolve() == Path(sys.argv[1])/'sim/ausc_policy/m2_student_oracle_proposals.py'
payload=json.load(sys.stdin)
binding=original.authenticate_source(payload['source'])
queried=[original.query_at(binding,row) for row in payload['rows']]
queries=[dict(query=q,snapshot=snapshot,commands=commands.tolist()) for q,snapshot,commands in queried] if payload.get('full_prefix') else [q for q,_,_ in queried]
print(json.dumps(queries,allow_nan=False))
'''
    receipts=[]
    for group in groups.values():
        refs=group['refs'];manifest_ref=refs['frozen_manifest'];root=Path(manifest_ref['path']).parent.resolve()
        manifest=runtime.read_bound(manifest_ref)
        need(isinstance(manifest,list) and manifest,'Complete frozen source20 inventory required')
        names=set()
        for item in manifest:
            relative=item.get('file');need(isinstance(relative,str),'Frozen manifest file path required')
            path=root/relative
            need(not path.is_symlink() and path.resolve().is_relative_to(root) and relative not in names,
                 'Unique regular files inside frozen oracle source archive required')
            names.add(relative);actual=runtime.reference(path)
            need(all(actual[k]==item.get(k) for k in ('bytes','sha256')),'Frozen oracle source archive bytes changed')
        generator=root/'sim/ausc_policy/m2_student_oracle_proposals.py'
        need('sim/ausc_policy/m2_student_oracle_proposals.py' in names and runtime.reference(generator)==refs['generator'],
             'Oracle generator must be its original fully bound frozen source20 module')
        rows=sorted({s['frozen_physics_row'] for _,s in group['sources']})
        payload=dict(source=group['request']['source'],rows=rows,full_prefix=v2)
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        result=subprocess.run([sys.executable,'-I','-B','-c',program,str(root)],input=json.dumps(payload),
            text=True,capture_output=True,check=False,timeout=1800,env=environment,cwd=root)
        need(result.returncode==0,'Frozen source20 actual student authentication failed: '+result.stderr[-8000:])
        queries=json.loads(result.stdout);need(len(queries)==len(rows),'Complete authenticated source query set required')
        by_row=dict(zip(rows,queries))
        for source_index,source in group['sources']:
            proposal=runtime.read_bound(source['bindings']['proposal'])
            authenticated=by_row[source['frozen_physics_row']]
            if v2:
                chunk_data.validate_oracle_sample(auxiliary['samples'],source_index,proposal,authenticated['query'],snapshot=authenticated['snapshot'])
            else:validate_oracle_sample(auxiliary['samples'],source_index,proposal,authenticated)
        receipts.append(dict(source=group['request']['source'],manifest=manifest_ref,physics_rows=rows,
            actual_history_authenticated=True,future_measurements_claimed=False))
    need(receipts,'At least one authenticated genuine student oracle source required')
    return receipts


def _audit_posture_inputs(config,prepared,auxiliary):
    """Audit every retained target and original source profile without clipping."""
    spec=config['posture_head'];profile=spec['physical_posture_profile'];refs=[]
    for episode in prepared['preparation']['episodes']:refs.append(episode['config'])
    for source in auxiliary['sources']:
        bindings=source['bindings'];refs.append(bindings['source'])
        if _version(config)>=5 and source.get('schema')==phase_data.SOURCE_SCHEMA:
            # V5 binds its original geometry config through the sealed request.
            # It does not duplicate the inherited sources' geometry binding.
            frozen=runtime.read_bound(bindings['request'])
            need(frozen.get('schema')==phase_data.PROBE_SCHEMA and frozen.get('source')==dict(
                config=bindings['source'],owner=bindings['owner'],child=bindings['child'],native=bindings['native']) and
                frozen.get('source_metadata')==source['metadata'],
                'V5 phase posture audit requires the original bound source/request metadata')
            expanded.full_reference(frozen.get('plan_config'));refs.append(frozen['plan_config'])
        else:refs.append(bindings['geometry'])
    seen=set()
    for ref in refs:
        key=runtime.digest(ref)
        if key in seen:continue
        seen.add(key);value=runtime.read_bound(ref)
        need(value.get('posture_limits')==profile,'Native/planned/oracle source must retain the sole original full posture profile')
    audits={}
    for split in ('train','validation'):
        samples=prepared[split]
        audits[split]=posture_head.audit_targets(samples['actions'],samples['action_is_pad'],samples['histories'],spec)
    samples=auxiliary['samples']
    audits['auxiliary']=posture_head.audit_targets(samples['relative_setters'],samples['setter_is_pad'],samples['source_histories'],spec)
    prepared['posture_target_audit']=audits


def load_inputs(config):
    validate_config(config)
    # Full native validation and normalization are exactly the original loader.
    native_metadata=(corridor_sampling.native_metadata(runtime.read_bound(config['dataset']),allowed_train_ids=scope.TRAIN16_IDS)
        if _version(config)>=6 else None)
    prepared=expanded.load_dataset(config['dataset'])
    if native_metadata is not None:prepared['corridor_native_metadata']=native_metadata
    auxiliary=_api(_version(config)).load_preparation(config['auxiliary'],scope.TRAIN16_IDS)
    if _version(config)>=6:auxiliary['array_reference']=deepcopy(runtime.read_bound(config['auxiliary'])['arrays'])
    if _version(config)>=2:
        if _version(config)==2:chunk_data.validate_lineage(auxiliary,scope.TRAIN16_IDS)
        _audit_posture_inputs(config,prepared,auxiliary)
    authentication=authenticate_oracles(auxiliary)
    frozen=runtime.read_bound(config['evaluation_cohort'])
    native_scope=deepcopy(prepared['scope'])
    prepared['scope']['train'].update(subject_ids=frozen['optimized_train_ids'],
        native_subject_ids=native_scope['train']['subject_ids'],native_episode_count=native_scope['train']['episode_count'],
        native_window_count=native_scope['train']['window_count'])
    prepared.update(auxiliary=auxiliary,optimization_scope=frozen,native_scope=native_scope,oracle_authentication=authentication)
    if _version(config)==7:
        complete,meta=measured_sampling.admitted_metadata(config['measured_bank'],dataset=config['dataset'],normalization=prepared['preparation']['normalization'])
        samples,loaded_meta=measured_data.load_arrays(config['measured_bank'])
        need(meta==loaded_meta,'Measured metadata changed during numerical loading')
        prepared['measured']=dict(reference=deepcopy(config['measured_bank']),complete=complete,metadata=meta,samples=samples)
        prepared['posture_target_audit']['measured']=posture_head.audit_targets(samples['actions'],samples['action_is_pad'],samples['histories'],config['posture_head'])
    if _version(config) in (8,9):
        sampler=refinement_sampling if _version(config)==9 else diverse_sampling
        prepared['measured_banks']=sampler.load_banks(config['measured_banks'],dataset=config['dataset'],normalization=prepared['preparation']['normalization'])
        for name,bank in prepared['measured_banks'].items():
            samples=bank['samples']
            prepared['posture_target_audit'][name]=posture_head.audit_targets(samples['actions'],samples['action_is_pad'],samples['histories'],config['posture_head'])
    if _version(config)==9:
        subset=_validate_task_subset(config,samples=prepared['validation'])
        need(subset['original_window_order_sha256']==prepared['window_order']['validation'],'Frozen subset original validation order differs')
        prepared['task_subset']=deepcopy(config['task_subset'])
    return prepared


def _tree(sources,source_indices,kind,*,anchor_balanced=False):
    tree={}
    for i,source in enumerate(sources):
        if source['kind']!=kind:continue
        targets=tree.setdefault(source['subject_id'],{}).setdefault(source['pose_id'],{})
        rows=np.flatnonzero(source_indices==i).tolist()
        if anchor_balanced:targets.setdefault(source['target_id'],{}).setdefault(source['anchor'],[]).extend(rows)
        else:targets.setdefault(source['target_id'],[]).extend(rows)
    return tree


def build_cache(prepared,auxiliary,*,torch=None,device='cpu',optimizer_version=None):
    """Pack validated arrays and sampling groups once, before optimization."""
    data_version=_version(auxiliary['samples']);version=data_version if optimizer_version is None else optimizer_version
    groups=_groups(version)
    need(data_version==(5 if version>=6 else version),'Optimizer/data version mismatch')
    need({s['kind'] for s in auxiliary['sources']}==set(_kinds(version)),'Both planned and oracle plus declared conditional auxiliary kinds required')
    native=prepared['train'];samples=auxiliary['samples']
    packed=dict(native=runtime.given.pack_batch(native,prepared['normalization'],np.arange(len(native['states'])),training=False),
        auxiliary=_api(data_version).pack_batch(samples,prepared['normalization'],np.arange(len(samples['source_indices']))))
    near=np.flatnonzero(np.linalg.norm(native['states'][:,7:10],axis=1)<=.060)
    if version not in (7,8,9):need(len(near)>0,'Real near60mm TRAIN origins required')
    native_subjects=np.empty(len(native['states']),dtype=object);native_subjects[:]=None
    native_sources=np.empty(len(native['states']),dtype=object);native_sources[:]=None
    for episode in prepared['preparation']['episodes']:
        if episode['split']=='train':
            start,end=episode['split_sample_range'];native_subjects[start:end]=episode['subject_id']
            native_sources[start:end]='native:'+episode['episode_id']
    need(all(isinstance(s,str) for s in native_subjects),'Every real origin needs actual native subject identity')
    trees={name:_tree(auxiliary['sources'],samples['source_indices'],kind,anchor_balanced=version>=3 and name=='conditional')
        for name,kind in dict(planned='planned_ideal_tracking',oracle='student_oracle_unexecuted',
            **({'conditional':conditional_data.CONDITIONAL_KIND} if version>=3 else {})).items()}
    need(all(trees.values()),'Both auxiliary kinds need nonempty balanced sampling groups')
    if torch is not None:
        packed={group:{key:torch.from_numpy(value).to(device) for key,value in batch.items()} for group,batch in packed.items()}
    result=dict(packed=packed,near=near,native_count=len(native['states']),trees=trees,version=version,groups=groups,
        source_ids=dict(native=native_sources,auxiliary=np.asarray([auxiliary['sources'][i].get('source_id','fixture-unqualified')
            for i in samples['source_indices']],dtype=object)),
        kinds=dict(native=np.full(len(native_subjects),'native',dtype=object),auxiliary=np.asarray([
            auxiliary['sources'][i]['kind'] for i in samples['source_indices']],dtype=object)),
        subjects=dict(native=native_subjects,auxiliary=np.asarray([auxiliary['sources'][i]['subject_id']
            for i in samples['source_indices']],dtype=object)))
    if version>=5:
        inventory=phase_data.terminal_pool_inventory(auxiliary,allowed_train_ids=scope.TRAIN16_IDS)
        eligible={r['row_index'] for r in inventory['rows'] if auxiliary['sources'][r['source_index']]['kind']==conditional_data.CONDITIONAL_KIND}
        tree=_tree(auxiliary['sources'],samples['source_indices'],conditional_data.CONDITIONAL_KIND,anchor_balanced=True)
        filtered={}
        for body,poses in tree.items():
            for pose,targets in poses.items():
                for target,anchors in targets.items():
                    for anchor,rows in anchors.items():
                        rows=[row for row in rows if row in eligible]
                        if rows:filtered.setdefault(body,{}).setdefault(pose,{}).setdefault(target,{})[anchor]=rows
        need(filtered,'Nonempty eligible conditional terminal prepose pool required')
        result.update(terminal_tree=filtered,terminal_pool_inventory=inventory,
            terminal_labels={int(row):dict(source=auxiliary['sources'][int(samples['source_indices'][row])]['source_id'],
                body=auxiliary['sources'][int(samples['source_indices'][row])]['subject_id'],
                pose=auxiliary['sources'][int(samples['source_indices'][row])]['pose_id'],
                target=auxiliary['sources'][int(samples['source_indices'][row])]['target_id'],
                anchor=auxiliary['sources'][int(samples['source_indices'][row])]['anchor']) for row in eligible})
    if version==6:
        inventory=corridor_sampling.build_inventory(prepared,auxiliary)
        result.update(data_version=data_version,corridor_inventory=inventory,
            near=np.asarray(inventory['pools']['native']['eligible_row_indices'],dtype=np.int64),
            corridor_labels={name:{row['row_index']:row for row in pool['rows']} for name,pool in inventory['pools'].items()},
            planned_corridor_tree=corridor_sampling.filtered_tree(trees['planned'],set(inventory['pools']['planned']['eligible_row_indices'])))
    if version==7:measured_sampling.configure_cache(result,prepared,auxiliary,torch=torch,device=device)
    if version==8:diverse_sampling.configure_cache(result,prepared,auxiliary,torch=torch,device=device)
    if version==9:refinement_sampling.configure_cache(result,prepared,auxiliary,torch=torch,device=device)
    return result


def sample_cached_indices(rng,cache):
    near=cache['near']
    if cache.get('version')==9:ids=dict(native=refinement_sampling.native_indices(rng,cache))
    elif cache.get('version')==8:ids=dict(native=diverse_sampling.native_indices(rng,cache))
    elif cache.get('version')==7:ids=dict(native=measured_sampling.native_indices(rng,cache))
    else:ids=dict(native=np.r_[near[rng.integers(len(near),size=16)],rng.integers(cache['native_count'],size=16)])
    def choose(values):return values[int(rng.integers(len(values)))]
    for kind,tree in cache['trees'].items():
        selected=[]
        for draw in range(cache.get('groups',GROUPS)[kind]['size']):
            active=cache['terminal_tree'] if cache.get('version',1)>=5 and kind=='conditional' and draw<4 else tree
            if cache.get('version') in (6,7,8,9) and kind=='planned' and draw<4:active=cache['planned_corridor_tree']
            poses=active[choose(sorted(active))];targets=poses[choose(sorted(poses))]
            windows=targets[choose(sorted(targets))]
            if isinstance(windows,dict):windows=windows[choose(sorted(windows))]
            selected.append(choose(windows))
        ids[kind]=np.asarray(selected,dtype=np.int64)
    return ids


def cached_batches(cache,indices):
    return {name:{key:value[indices[name]] for key,value in cache['packed']['native' if name=='native' else 'auxiliary'].items()}
        for name in cache.get('groups',GROUPS)}


def mixed_loss(policy,batches,*,version=1):
    """One official valid-cell mean per declared group, weighted exactly once."""
    groups=_groups(version)
    need(set(batches)==set(groups),'Exactly declared native/planned/conditional/oracle loss groups required')
    total=None;logged={}
    for name,contract in groups.items():
        batch=batches[name];valid=int((~batch['action_is_pad']).sum().item())
        need(valid>0 and len(batch['action'])==contract['size'],'Nonempty fixed declared supervision group size required')
        loss,_=policy(batch);need(loss.ndim==0,'Official ACT scalar loss required')
        weighted=contract['coefficient']*loss;total=weighted if total is None else total+weighted
        scalar=float(loss.detach().cpu()) if hasattr(loss,'detach') else float(loss)
        logged[name]=dict(loss=scalar,coefficient=contract['coefficient'],valid_cells=valid,valid_joints=valid*7,
            examples=len(batch['action']))
    return total,logged


def record_draws(value,cache,indices):
    """Call only after a successful optimizer step; count actual selected examples."""
    seen=set(value['actual_optimized_train_ids'])
    for name,index in indices.items():
        group='native' if name=='native' else 'auxiliary'
        value['gradient_examples'][name]+=len(index)
        for subject in cache['subjects'][group][index]:
            counts=value['gradient_subject_examples'][name];counts[subject]=counts.get(subject,0)+1;seen.add(subject)
        if cache.get('version',1)>=3:
            for key,field in (('gradient_source_examples','source_ids'),('gradient_kind_examples','kinds')):
                for label in cache[field][group][index]:
                    counts=value[key][name];counts[label]=counts.get(label,0)+1
    value['actual_optimized_train_ids']=sorted(seen)
    if cache.get('version',1)>=5:
        counts=value['conditional_terminal_draws'];counts['denominator']+=len(indices['conditional'])
        for row in indices['conditional'][:4]:
            labels=cache['terminal_labels'][int(row)];counts['draws']+=1
            for key,label in labels.items():counts[key][label]=counts[key].get(label,0)+1
    if cache.get('version')==8:diverse_sampling.record_draws(value,cache,indices)
    if cache.get('version')==9:refinement_sampling.record_draws(value,cache,indices)
    if cache.get('version')==7:measured_sampling.record_draws(value,cache,indices)
    if cache.get('version')==6:corridor_sampling.record_draws(value,indices,cache['corridor_inventory'],lookups=cache['corridor_labels'])


def new_optimization(*,version):
    groups=_groups(version)
    value=dict(gradient_examples={k:0 for k in groups},gradient_subject_examples={k:{} for k in groups},actual_optimized_train_ids=[])
    if version>=3:value.update(gradient_source_examples={k:{} for k in groups},gradient_kind_examples={k:{} for k in groups})
    if version>=5:value['conditional_terminal_draws']=dict(draws=0,denominator=0,**{key:{} for key in ('source','body','pose','target','anchor')})
    if version==6:value['corridor_draws']=corridor_sampling.new_draws()
    if version==7:value['measured_draws']=measured_sampling.new_draws()
    if version==8:value['diverse_draws']=diverse_sampling.new_draws()
    if version==9:value['refinement_draws']=refinement_sampling.new_draws()
    return value


def contribution_inventory(config):
    """Bind reported source counts to actual immutable metadata, without arrays."""
    native=runtime.read_bound(config['dataset']);auxiliary=auxiliary_metadata(config['auxiliary'],version=_version(config))
    inventory={k:{} for k in GROUPS_V3}
    for episode in native['episodes']:
        if episode['split']=='train':inventory['native']['native:'+episode['episode_id']]=dict(subject_id=episode['subject_id'],kind='native')
    mapping={'planned_ideal_tracking':'planned','student_oracle_unexecuted':'oracle',conditional_data.CONDITIONAL_KIND:'conditional'}
    for source in auxiliary['sources']:
        inventory[mapping[source['kind']]][source['source_id']]=dict(subject_id=source['subject_id'],kind=source['kind'])
        if _version(config)>=5:
            inventory[mapping[source['kind']]][source['source_id']].update(pose_id=source['pose_id'],target_id=source['target_id'],anchor=source.get('anchor'))
    if _version(config)==7:
        _,meta=measured_sampling.admitted_metadata(config['measured_bank'],dataset=config['dataset'],normalization=native['normalization'])
        for source in measured_sampling.source_records(meta):
            inventory['native'][source['source']]=dict(subject_id=source['body'],kind='measured_recovery',pose_id=source['pose'],target_id=source['target'],slot=source['slot'])
    if _version(config) in (8,9):
        sampler=refinement_sampling if _version(config)==9 else diverse_sampling
        banks=sampler.admitted_metadata(config['measured_banks'],dataset=config['dataset'],normalization=native['normalization'])
        for name,bank in banks.items():
            for source in sampler.source_records(bank['metadata'],name):
                inventory['native'][source['source']]=dict(subject_id=source['body'],kind='measured_recovery',pose_id=source['pose'],target_id=source['target'],slot=source['slot'],bank=name)
    return inventory


def _corridor_inventory(value,config):
    """Read the training-bound sampler proof, never historical factories/models."""
    need(value.get('data_version')==5 and value.get('model_optimizer_version')==6,
        'V6 optimizer/model and unchanged V5 data must be explicitly distinguished')
    expanded.full_reference(value.get('corridor_inventory'))
    inventory=runtime.read_bound(value['corridor_inventory'])
    complete,metadata=phase_data.metadata(config['auxiliary'],scope.TRAIN16_IDS)
    corridor_sampling.validate_binding(inventory,native=runtime.read_bound(config['dataset']),
        auxiliary_metadata=metadata,auxiliary_arrays=complete['arrays'],frozen_scope=runtime.read_bound(config['evaluation_cohort']))
    return inventory


def _measured_inventory(value,config):
    need(value.get('data_version')==5 and value.get('model_optimizer_version')==7 and
        value.get('inputs',{}).get('measured_bank')==config.get('measured_bank'),
        'V7/data5 and separate measured bank input required')
    expanded.full_reference(value.get('measured_inventory'))
    inventory=runtime.read_bound(value['measured_inventory'])
    complete,metadata=phase_data.metadata(config['auxiliary'],scope.TRAIN16_IDS)
    measured_sampling.validate_binding(inventory,config=config,native=runtime.read_bound(config['dataset']),
        auxiliary_metadata=metadata,auxiliary_arrays=complete['arrays'],frozen_scope=runtime.read_bound(config['evaluation_cohort']))
    need(value.get('measured_counts')==dict(episodes=4,train_windows=inventory['measured_windows'],
        base_counts=NATIVE_COUNTS,normalization_refitted=False), 'Separate measured/Base29 count provenance required')
    return inventory


def measured_counts(inventory):
    return dict(banks={name:dict(episodes=len(bank['sources']),train_windows=bank['windows'],admission=deepcopy(bank['optimization_admission']))
        for name,bank in inventory['measured_banks'].items()},train_windows=inventory['measured_windows'],base_counts=deepcopy(NATIVE_COUNTS),
        physical_measured_subject_ids=deepcopy(inventory['physical_measured_subject_ids']),computed_auxiliary_subject_ids=deepcopy(inventory['computed_auxiliary_subject_ids']),normalization_refitted=False)


def _diverse_inventory(value,config):
    need(value.get('data_version')==5 and value.get('model_optimizer_version')==8 and
        value.get('inputs',{}).get('measured_banks')==config.get('measured_banks'),'V8/data5 and both separate measured bank inputs required')
    expanded.full_reference(value.get('diverse_inventory'));inventory=runtime.read_bound(value['diverse_inventory'])
    complete,metadata=phase_data.metadata(config['auxiliary'],scope.TRAIN16_IDS)
    diverse_sampling.validate_binding(inventory,config=config,native=runtime.read_bound(config['dataset']),auxiliary_metadata=metadata,
        auxiliary_arrays=complete['arrays'],frozen_scope=runtime.read_bound(config['evaluation_cohort']))
    need(value.get('measured_counts')==measured_counts(inventory),'V8 source/request/prepared/qualified/physical/computed counts differ')
    return inventory


def _refinement_inventory(value,config):
    need(value.get('data_version')==5 and value.get('model_optimizer_version')==9 and
        value.get('inputs',{}).get('measured_banks')==config.get('measured_banks') and
        value.get('task_subset')==value.get('inputs',{}).get('task_subset')==config.get('task_subset'),
        'V9/data5, all three bank inputs and independent frozen subset required')
    expanded.full_reference(value.get('refinement_inventory'));inventory=runtime.read_bound(value['refinement_inventory'])
    complete,metadata=phase_data.metadata(config['auxiliary'],scope.TRAIN16_IDS)
    refinement_sampling.validate_binding(inventory,config=config,native=runtime.read_bound(config['dataset']),auxiliary_metadata=metadata,
        auxiliary_arrays=complete['arrays'],frozen_scope=runtime.read_bound(config['evaluation_cohort']))
    need(value.get('measured_counts')==measured_counts(inventory),'V9 source/physical/computed count provenance differs')
    return inventory


def checkpoint(policy,prepared,inputs,step,metrics,protocol,optimization):
    value=expanded.original.checkpoint(policy,prepared,inputs,step,metrics)
    value.update(schema={PROTOCOL_SCHEMA:PILOT_SCHEMA,PROTOCOL_SCHEMA_V2:PILOT_SCHEMA_V2,PROTOCOL_SCHEMA_V3:PILOT_SCHEMA_V3,PROTOCOL_SCHEMA_V4:PILOT_SCHEMA_V4,PROTOCOL_SCHEMA_V5:PILOT_SCHEMA_V5,PROTOCOL_SCHEMA_V6:PILOT_SCHEMA_V6,PROTOCOL_SCHEMA_V7:PILOT_SCHEMA_V7,PROTOCOL_SCHEMA_V8:PILOT_SCHEMA_V8,PROTOCOL_SCHEMA_V9:PILOT_SCHEMA_V9}[protocol['schema']],protocol=protocol,optimization_scope=prepared['optimization_scope'],
        native_scope=prepared['native_scope'],auxiliary_counts=prepared['auxiliary']['counts'],
        oracle_authentication=prepared['oracle_authentication'],optimization=deepcopy(optimization))
    if protocol['schema'] in (PROTOCOL_SCHEMA_V2,PROTOCOL_SCHEMA_V3,PROTOCOL_SCHEMA_V4,PROTOCOL_SCHEMA_V5,PROTOCOL_SCHEMA_V6,PROTOCOL_SCHEMA_V7,PROTOCOL_SCHEMA_V8,PROTOCOL_SCHEMA_V9):
        value['posture_head_diagnostics']=posture_head.diagnostics(policy)
        value['posture_target_audit']=deepcopy(prepared['posture_target_audit'])
    if protocol['schema'] in (PROTOCOL_SCHEMA_V4,PROTOCOL_SCHEMA_V5,PROTOCOL_SCHEMA_V6,PROTOCOL_SCHEMA_V7,PROTOCOL_SCHEMA_V8,PROTOCOL_SCHEMA_V9):value['initialization']=deepcopy(prepared.get('initialization'))
    if protocol['schema'] in (PROTOCOL_SCHEMA_V5,PROTOCOL_SCHEMA_V6,PROTOCOL_SCHEMA_V7):
        value.update(selection_split='fixed_final_budget',selection_policy=deepcopy(FINAL_SELECTION_V5),terminal_pool_inventory=deepcopy(prepared.get('terminal_pool_inventory')))
    if protocol['schema']==PROTOCOL_SCHEMA_V6:
        # Early failure packets precede cache creation; completed admission still requires the bound inventory.
        value.update(data_version=5,model_optimizer_version=6,corridor_inventory=deepcopy(prepared.get('corridor_inventory')))
    if protocol['schema']==PROTOCOL_SCHEMA_V7:
        value.update(data_version=5,model_optimizer_version=7,measured_inventory=deepcopy(prepared.get('measured_inventory')),
            measured_counts=deepcopy(prepared.get('measured_counts')))
    if protocol['schema']==PROTOCOL_SCHEMA_V8:
        value.update(data_version=5,model_optimizer_version=8,diverse_inventory=deepcopy(prepared.get('diverse_inventory')),
            measured_counts=deepcopy(prepared.get('measured_counts')),selection_split='validation',selection_policy=deepcopy(SELECTION_POLICY_V8),
            terminal_pool_inventory=deepcopy(prepared.get('terminal_pool_inventory')))
    if protocol['schema']==PROTOCOL_SCHEMA_V9:
        value.update(data_version=5,model_optimizer_version=9,refinement_inventory=deepcopy(prepared.get('refinement_inventory')),
            task_subset=deepcopy(prepared.get('task_subset')),measured_counts=deepcopy(prepared.get('measured_counts')),
            selection_split='frozen_observed_validation',selection_policy=deepcopy(SELECTION_POLICY_V9),
            terminal_pool_inventory=deepcopy(prepared.get('terminal_pool_inventory')))
    return value


def _validate_initialization(config,inputs,value):
    """Validate pinned provenance and the actual step0 check, without old loading."""
    warm_start=_warm_start(_version(config))
    receipt=warm_start.validate_receipt(config['initialization'],normalization=native_normalization(config),posture_head=config['posture_head'])
    need(inputs.get('initialization')==config['initialization'] and isinstance(value,dict) and
        value.get('receipt')==config['initialization'] and value.get('equivalence')==inputs.get('initialization_equivalence'),
        'Exact V4 initialization receipt/equivalence input references required')
    expanded.full_reference(value['equivalence']);equivalence=runtime.read_bound(value['equivalence'])
    difference=equivalence.get('prediction_max_abs_rad')
    need(equivalence.get('schema')==warm_start.EQUIVALENCE_SCHEMA and equivalence.get('passed') is True and
        equivalence.get('initialization')==config['initialization'] and equivalence.get('initial_tensor_proofs')==receipt['tensor_proofs'] and
        equivalence.get('optimizer_step')==0 and equivalence.get('optimizer_imported') is False and
        equivalence.get('scheduler_imported') is False and equivalence.get('final_linear_reset') is False and
        equivalence.get('probe_split')=='train' and equivalence.get('probe_indices')==warm_start.PROBE_TRAIN_INDICES and
        equivalence.get('prediction_rtol')==1e-5 and equivalence.get('prediction_atol')==1e-6 and
        type(difference) in (int,float) and np.isfinite(difference) and difference>=0,
        'Actual ACT6 initial tensor/prediction equivalence and optimizer step0 required')
    need(value.get('optimizer')==dict(name='AdamW',initial_step=0,initial_state_entries=0,lr=config['lr'],
        weight_decay=config['weight_decay'],betas=[.9,.999],eps=1e-8,optimizer_imported=False,scheduler_imported=False),
        'ACT6 must use fresh AdamW step0; historical optimizer/scheduler cannot resume')


def load_selected_policy(selection):
    pilot_schema,selection_schema,_,_=_schemas(selection)
    need(selection.get('schema') in (pilot_schema,selection_schema) and
         all(selection.get(k) is v for k,v in FLAGS.items()),'Explicit mixed-source ACT selected weights required')
    config=validate_config(runtime.read_bound(selection['inputs']['config']))
    need(_version(selection)==_version(config),'Selected model/config version mismatch; head cannot relabel legacy weights')
    need(selection.get('protocol')==training_protocol(config) and selection['inputs']['dataset']==config['dataset'] and
         selection['inputs'].get('auxiliary')==config['auxiliary'] and
         selection.get('optimization_scope')==runtime.read_bound(config['evaluation_cohort']),
         'Exact native/auxiliary/scope/protocol binding differs')
    candidate_verification=_validate_candidates(selection) if _version(selection)>=5 else None
    if _version(selection)>=4:_validate_initialization(config,selection['inputs'],selection.get('initialization'))
    scope.validate_train16_metadata(selection['optimization_scope'],actual_train_ids=selection['scope']['train']['subject_ids'])
    _validate_draws(selection.get('selected_optimization'),selection['selected_step'],selection['optimization_scope'],
        selection['native_scope']['train']['subject_ids'],version=_version(selection),
        inventory=contribution_inventory(config) if _version(selection)>=3 else None,
        corridor_inventory=_corridor_inventory(selection,config) if _version(selection)==6 else None,
        measured_inventory=_measured_inventory(selection,config) if _version(selection)==7 else None,
        diverse_inventory=_diverse_inventory(selection,config) if _version(selection)==8 else None,
        refinement_inventory=_refinement_inventory(selection,config) if _version(selection)==9 else None)
    validate_code(selection['inputs']['code'],version=_version(selection))
    torch,policy,software=make_model(config,device='cpu')
    validate_official_forward(software);saved=runtime.read_bound(selection['inputs']['software'])
    for key in ('vendor','dependencies','python','prefix','python_version','source_bindings'):
        need(saved[key]==software[key],'Official mixed-source CPU environment changed: '+key)
    packet=torch.load(io.BytesIO(runtime.bound_path(selection['best']).read_bytes()),map_location='cpu',weights_only=True)
    need(packet.get('schema')==pilot_schema and packet.get('step')==selection['selected_step'] and
         packet.get('task_contract')==runtime.task_contract() and packet.get('model')==runtime.shared.native_spec() and
         all(packet.get(k) is v for k,v in FLAGS.items()) and
         all(packet.get(k)==selection.get(k) for k in ('inputs','scope','protocol','optimization_scope','native_scope',
              'auxiliary_counts','oracle_authentication')) and packet.get('optimization')==selection.get('selected_optimization'),
         'Exact selected mixed-source checkpoint required')
    if _version(selection)>=5:
        need(all(packet.get(k)==selection.get(k) for k in ('selection_policy','terminal_pool_inventory')), 'V5 checkpoint selection/pool differs')
    if _version(selection)==6:need(packet.get('corridor_inventory')==selection.get('corridor_inventory') and
        packet.get('data_version')==selection.get('data_version')==5 and packet.get('model_optimizer_version')==selection.get('model_optimizer_version')==6,'V6 model/data/inventory binding differs')
    if _version(selection)==7:need(all(packet.get(k)==selection.get(k) for k in
        ('measured_inventory','measured_counts','data_version','model_optimizer_version')), 'V7 checkpoint measured provenance differs')
    if _version(selection)==8:need(all(packet.get(k)==selection.get(k) for k in ('diverse_inventory','measured_counts','data_version','model_optimizer_version')), 'V8 checkpoint composite provenance differs')
    if _version(selection)==9:need(all(packet.get(k)==selection.get(k) for k in V9_PACKET_BINDINGS), 'V9 checkpoint initialization/subset/composite provenance differs')
    if _version(selection)>=4:need(packet.get('initialization')==selection['initialization'],'Selected V4 initialization binding differs')
    need(runtime.validate_normalization(packet['normalization'])==native_normalization(config),
         'Selected normalization must equal the pinned real native TRAIN normalization; no auxiliary refit')
    policy.load_state_dict(packet['state_dict'],strict=True);policy.eval()
    if _version(selection)==9:packet['_task_candidate_verification']=candidate_verification
    return torch,policy,packet


def _validate_draws(value,steps,frozen,native_subjects,*,version=1,inventory=None,corridor_inventory=None,measured_inventory=None,diverse_inventory=None,refinement_inventory=None):
    groups=_groups(version)
    need(isinstance(value,dict) and set(value.get('gradient_examples',{}))==set(groups) and
         set(value.get('gradient_subject_examples',{}))==set(groups),'Complete actual gradient-contributing example counts required')
    subjects=set()
    for name,contract in groups.items():
        counts=value['gradient_subject_examples'][name]
        expected=steps*contract['size']
        need(value['gradient_examples'][name]==expected and isinstance(counts,dict) and counts and
             all(type(n) is int and n>0 for n in counts.values()) and sum(counts.values())==expected and
             set(counts)<=set(frozen['optimized_train_ids']) and (name!='native' or set(counts)<=set(native_subjects)|({entry['subject_id'] for entry in inventory['native'].values()} if version in (8,9) else set())),
             'Actual gradient-contributing identity/example counts differ: '+name)
        subjects.update(counts)
    need(value.get('actual_optimized_train_ids')==sorted(subjects),'Actual optimized identities differ from contributing records')
    scope.validate_train16_metadata(frozen,actual_train_ids=subjects)
    if version==6:corridor_sampling.validate_draws(value,steps=steps,inventory=corridor_inventory)
    if version==7:measured_sampling.validate_draws(value,steps=steps,inventory=measured_inventory)
    if version==8:diverse_sampling.validate_draws(value,steps=steps,inventory=diverse_inventory)
    if version==9:refinement_sampling.validate_draws(value,steps=steps,inventory=refinement_inventory)
    if version>=5:
        counts=value.get('conditional_terminal_draws',{})
        need(counts.get('draws')==steps*4 and counts.get('denominator')==steps*8,
            'Exact realized four-of-eight terminal draw denominator required')
        for key in ('source','body','pose','target','anchor'):
            items=counts.get(key,{})
            need(isinstance(items,dict) and items and all(type(n) is int and n>0 for n in items.values()) and
                sum(items.values())==steps*4,'Exact realized conditional terminal counts required: '+key)
        need(set(counts['source'])<=set(inventory['conditional']) and set(counts['body'])<=set(frozen['optimized_train_ids']),
            'Terminal contributions outside eligible TRAIN sources')
        derived={key:{} for key in ('body','pose','target','anchor')}
        for source_id,count in counts['source'].items():
            need(count<=value['gradient_source_examples']['conditional'].get(source_id,0),'Terminal source exceeds all conditional draws')
            entry=inventory['conditional'][source_id]
            for key,field in (('body','subject_id'),('pose','pose_id'),('target','target_id'),('anchor','anchor')):
                label=entry.get(field);need(isinstance(label,str),'Terminal source stratum required')
                derived[key][label]=derived[key].get(label,0)+count
        need(all(counts[key]==expected for key,expected in derived.items()),'Terminal source/body/pose/target/anchor counts must reconcile')
    if version>=3:
        need(isinstance(inventory,dict) and set(inventory)==set(groups) and
            set(value.get('gradient_source_examples',{}))==set(groups) and
            set(value.get('gradient_kind_examples',{}))==set(groups),'V3 actual source and kind contribution inventory required')
        for name in groups:
            counts=value['gradient_source_examples'][name];by_subject={};by_kind={}
            need(isinstance(counts,dict) and counts and set(counts)<=set(inventory[name]) and
                all(type(n) is int and n>0 for n in counts.values()) and sum(counts.values())==value['gradient_examples'][name],
                'Actual V3 source contribution counts differ: '+name)
            for source_id,count in counts.items():
                entry=inventory[name][source_id];subject=entry['subject_id'];kind=entry['kind']
                by_subject[subject]=by_subject.get(subject,0)+count;by_kind[kind]=by_kind.get(kind,0)+count
            need(by_subject==value['gradient_subject_examples'][name] and by_kind==value['gradient_kind_examples'][name],
                'Actual V3 source/kind/subject contribution sums differ: '+name)


def save_task_candidate(path,prediction,prepared):
    """Only retained canonical Official arrays may determine V9 task scores."""
    binding=runtime.save_predictions(path,prediction,window_order_sha256=prepared['window_order']['validation'])
    saved=runtime.load_predictions(binding,window_order_sha256=prepared['window_order']['validation'])
    return binding,task_selection.score(saved,prepared['validation'],prepared['task_subset'])


def _task_selection(value):
    return dict(schema=task_selection.SELECTION_SCHEMA,dataset=deepcopy(value['inputs']['dataset']),
        subset=deepcopy(value['task_subset']),selected_step=value['selected_step'],
        candidates=[dict(step=r['step'],predictions=deepcopy(r['predictions']['validation']),score=deepcopy(r['score']))
            for r in value['evaluation_candidates']])


def verify_task_candidates(value):
    """Exact all40 replay, separate from selected-policy CPU inference."""
    config=runtime.read_bound(value['inputs']['config'])
    need(_version(value)==_version(config)==9 and value['inputs']['dataset']==config['dataset'] and
        value.get('task_subset')==value['inputs'].get('task_subset')==config.get('task_subset') and
        value.get('task_selection')==_task_selection(value),'V9 retained task ranking/config/subset provenance differs')
    _validate_task_subset(config)
    proof=task_selection.verify_candidates(value['task_selection'])
    prepared=expanded.load_dataset(config['dataset'])
    need(prepared['window_order']==value['window_order'],'V9 complete original TRAIN/VAL sample ordering changed')
    for row in value['evaluation_candidates']:
        for split in NATIVE_COUNTS:
            prediction=runtime.load_predictions(row['predictions'][split],window_order_sha256=value['window_order'][split])
            need(prediction.shape==(NATIVE_COUNTS[split],20,7) and
                runtime.metrics(prediction,prepared[split])==row[split], 'V9 full saved canonical predictions/diagnostic metrics differ')
    return proof


V9_PACKET_BINDINGS=('inputs','protocol','task_subset','refinement_inventory','measured_counts','initialization',
    'terminal_pool_inventory','data_version','model_optimizer_version','selection_policy')


def _validate_v9_candidate_states(value):
    import torch
    config=runtime.read_bound(value['inputs']['config']);warm=_warm_start(9)
    template=warm.validate_receipt(config['initialization'])['tensor_proofs']
    for row in value['evaluation_candidates']:
        expanded.full_reference(row.get('checkpoint'))
        packet=torch.load(io.BytesIO(runtime.bound_path(row['checkpoint']).read_bytes()),map_location='cpu',weights_only=True)
        need(packet.get('schema')==PILOT_SCHEMA_V9 and packet.get('step')==row['step'] and
            packet.get('metrics')=={k:row[k] for k in ('step','train','validation','score')} and
            all(packet.get(k)==value.get(k) for k in V9_PACKET_BINDINGS+('scope','native_scope','optimization_scope','auxiliary_counts','oracle_authentication')) and
            packet.get('task_contract')==runtime.task_contract() and packet.get('model')==runtime.shared.native_spec() and
            all(packet.get(k) is v for k,v in FLAGS.items()), 'V9 retained candidate state/step/metrics/provenance differs')
        state=packet.get('state_dict',{});proof=warm.tensor_proofs(state)
        need(len(state)==54 and set(proof)==set(template) and
            all(all(proof[k][field]==template[k][field] for field in ('shape','dtype','bytes')) for k in proof) and
            row.get('state_proofs')==proof,'All54 finite candidate tensor states must match unchanged initialized architecture and saved proofs')


def _validate_candidates(value):
    candidates=value.get('evaluation_candidates',[])
    if _version(value)==9:
        need(value.get('selection_policy')==SELECTION_POLICY_V9 and value.get('completed_steps')==20000 and
            value.get('selected_step')==selected_step(candidates,version=9),'V9 completed20000/all40/exact earliest task minimum required')
        for row in candidates:
            need(set(row.get('predictions',{}))==set(NATIVE_COUNTS),'Both complete original V9 prediction populations required')
        need(len({r['checkpoint']['path'] for r in candidates})==40,'All40 distinct immutable V9 candidate states required')
        chosen=next(r for r in candidates if r['step']==value['selected_step'])
        need(value.get('best')==chosen['checkpoint'] and value.get('selected_predictions')==chosen['predictions'] and
            value.get('selected_metrics')=={s:chosen[s] for s in NATIVE_COUNTS},'V9 selected state/predictions/metrics differ from exact task minimum')
        if 'evaluations' in value:need(value['evaluations']==[{k:r[k] for k in ('step','train','validation','score')} for r in candidates], 'V9 evaluation rows differ from retained candidate evidence')
        proof=verify_task_candidates(value);_validate_v9_candidate_states(value)
        return proof
    if _version(value)==8:
        need(value.get('selection_policy')==SELECTION_POLICY_V8 and value.get('completed_steps')==20000 and
            value.get('selected_step')==selected_step(candidates,version=8),'V8 completed20000/all40/earliest full native VAL minimum required')
        for row in candidates:
            expanded.full_reference(row.get('checkpoint'));runtime.bound_path(row['checkpoint'])
            need(set(row.get('predictions',{}))=={'train','validation'}, 'Every V8 full evaluation needs both prediction bindings')
            for split,prediction in row['predictions'].items():
                ref=prediction.get('reference');expanded.full_reference(ref);runtime.bound_path(ref)
                array=prediction.get('array',{})
                need(prediction.get('window_order_sha256')==value.get('window_order',{}).get(split) and
                    array.get('shape')==[NATIVE_COUNTS[split],20,7] and array.get('dtype')=='<f8' and array.get('order')=='C' and
                    isinstance(array.get('sha256'),str) and len(array['sha256'])==64,'V8 full ordered canonical prediction bindings required')
        need(len({row['checkpoint']['path'] for row in candidates})==40,'All40 distinct immutable V8 candidate states required')
        selected=next(row for row in candidates if row['step']==value['selected_step'])
        need(all(selected['checkpoint'][k]==value['best'][k] for k in ('bytes','sha256')) and value.get('selected_predictions')==selected['predictions'] and
            value.get('selected_metrics')=={s:selected.get(s,{}) for s in ('train','validation')},'V8 selected checkpoint/predictions/full metrics must match the earliest native VAL minimum')
        if 'evaluations' in value:need(value['evaluations']==[{k:v for k,v in row.items() if k not in ('checkpoint','predictions')} for row in candidates], 'V8 immutable candidate/evaluation rows differ')
        return
    need(value.get('selection_policy')==FINAL_SELECTION_V5 and value.get('selected_step')==1000 and
        [row.get('step') for row in candidates]==[500,1000], 'V5 fixed final1000 and both immutable candidates required')
    for row in candidates:
        expanded.full_reference(row.get('checkpoint'));runtime.bound_path(row['checkpoint'])
    need(len({row['checkpoint']['path'] for row in candidates})==2,'Distinct immutable V5 candidate states required')
    need(all(candidates[1]['checkpoint'][k]==value['best'][k] for k in ('bytes','sha256')),
        'V5 selected checkpoint must equal immutable final1000 candidate')


def validate_complete(value):
    pilot_schema,_,_,_=_schemas(value)
    need(value.get('schema')==pilot_schema and value.get('complete') is True and
         value.get('task_contract')==runtime.task_contract() and value.get('model')==runtime.shared.native_spec() and
         all(value.get(k) is v for k,v in FLAGS.items()),'Complete explicit mixed-source ACT diagnostic run required')
    config=validate_config(value.get('schedule',{}));inputs=value.get('inputs',{})
    need(_version(value)==_version(config),'Completed model/config version mismatch; head cannot relabel legacy weights')
    need(runtime.read_bound(inputs['config'])==config and inputs.get('dataset')==config['dataset'] and
         inputs.get('auxiliary')==config['auxiliary'] and value.get('protocol')==training_protocol(config) and
         value.get('completed_steps')==config['steps'],'Completed exact mixed-source inputs/schedule/protocol required')
    if _version(value)>=4:_validate_initialization(config,inputs,value.get('initialization'))
    frozen=runtime.read_bound(config['evaluation_cohort']);selected=value.get('selected_step')
    need(value.get('optimization_scope')==frozen,'Actual new scope differs from pretraining freeze')
    native=value.get('native_scope',{});actual=value.get('scope',{})
    need(set(native)==set(actual)=={'train','validation'} and
         set(native['train'].get('subject_ids',[]))==set(binding.TRAIN_IDS) and
         native['validation'].get('subject_ids')==['p000923'] and
         actual['train'].get('subject_ids')==frozen['optimized_train_ids'] and actual['validation']==native['validation'] and
         actual['train'].get('native_subject_ids')==native['train']['subject_ids'] and
         actual['train'].get('native_episode_count')==native['train']['episode_count'] and
         actual['train'].get('native_window_count')==NATIVE_COUNTS['train'] and
         all(actual[s].get('window_count')==native[s].get('window_count')==NATIVE_COUNTS[s] and
             actual[s].get('episode_ids')==native[s].get('episode_ids') and native[s].get('episode_ids') for s in native),
         'Actual optimization identities and unchanged separately typed native evaluation population required')
    rows=value.get('evaluations',[])
    need([row.get('step') for row in rows]==expanded.evaluation_steps(config) and selected==selected_step(rows,version=_version(value)),
         'Earliest minimum complete native raw masked selection-VAL MAE required')
    if _version(value)>=5:
        need(value.get('selection_policy')==(SELECTION_POLICY_V9 if _version(value)==9 else SELECTION_POLICY_V8 if _version(value)==8 else FINAL_SELECTION_V5),'Explicit version-specific complete selection policy required')
        _validate_candidates(value)
    for split,count in NATIVE_COUNTS.items():
        for row in rows:need(runtime.validate_metrics(row[split])['sample_count']==count,'Complete native evaluation population differs')
        need(runtime.validate_metrics(value['selected_metrics'][split])==next(row[split] for row in rows if row['step']==selected),
             'Selected native metrics differ from full selected evaluation')
    inventory=contribution_inventory(config) if _version(value)>=3 else None
    _validate_draws(value.get('optimization'),config['steps'],frozen,native['train']['subject_ids'],version=_version(value),inventory=inventory,corridor_inventory=_corridor_inventory(value,config) if _version(value)==6 else None,measured_inventory=_measured_inventory(value,config) if _version(value)==7 else None,diverse_inventory=_diverse_inventory(value,config) if _version(value)==8 else None,refinement_inventory=_refinement_inventory(value,config) if _version(value)==9 else None)
    _validate_draws(value.get('selected_optimization'),selected,frozen,native['train']['subject_ids'],version=_version(value),inventory=inventory,corridor_inventory=_corridor_inventory(value,config) if _version(value)==6 else None,measured_inventory=_measured_inventory(value,config) if _version(value)==7 else None,diverse_inventory=_diverse_inventory(value,config) if _version(value)==8 else None,refinement_inventory=_refinement_inventory(value,config) if _version(value)==9 else None)
    counts=value.get('auxiliary_counts',{})
    if _version(value)>=4:
        need(counts==auxiliary_metadata(config['auxiliary'],version=_version(value))['counts'], 'Actual V4 parent/new contribution counts differ')
        computational=([counts.get('groups',{}).get(name,{}) for name in ('planned','conditional','oracle')] if _version(value)==4 else
            [dict(samples=len(inventory[name]),accepted_sources=len(inventory[name])) for name in ('planned','conditional','oracle')])
    else:
        need(set(counts)==set(_kinds(_version(value))),'Separate computational kinds required')
        computational=list(counts.values())
    need(all(row.get('samples',0)>0 and row.get('accepted_sources',0)>0 for row in computational) and
         value.get('oracle_authentication') and all(r.get('actual_history_authenticated') is True and
         r.get('future_measurements_claimed') is False for r in value['oracle_authentication']),
         'Separate computational counts and actual pre-query history authentication required')
    for key in ('best','cpu_equivalence'):expanded.full_reference(value.get(key))
    return value


def model_binding(ref):
    expanded.full_reference(ref);complete=validate_complete(runtime.read_bound(ref));runtime.bound_path(complete['best'])
    pilot_schema,selection_schema,cpu_schema,_=_schemas(complete)
    cpu=runtime.read_bound(complete['cpu_equivalence'])
    need(cpu.get('schema')==cpu_schema and cpu.get('passed') is True and cpu.get('checkpoint')==complete['best'] and
         cpu.get('selected_step')==complete['selected_step'] and cpu.get('scope')==complete['scope'] and
         cpu.get('optimization_scope')==complete['optimization_scope'] and all(cpu.get(k) is v for k,v in FLAGS.items()) and
         set(cpu.get('comparisons',{}))=={'train','validation'} and all(row.get('passed') is True and
             row.get('prediction_rtol')==1e-5 and row.get('prediction_atol')==1e-6 and
             row.get('metric_rtol')==1e-5 and row.get('metric_atol')==1e-7 for row in cpu['comparisons'].values()),
         'Actual selected mixed-source CPU equivalence under original tolerances required')
    selection=runtime.read_bound(cpu['selection'])
    need(selection.get('schema')==selection_schema and all(selection.get(k)==complete.get(k) for k in
         ('inputs','best','selected_step','scope','native_scope','optimization_scope','protocol','window_order',
          'selected_metrics','selected_predictions','selected_optimization','optimization','auxiliary_counts','oracle_authentication')),
         'Mixed-source CPU receipt must bind the exact selected provenance and population')
    if _version(complete)>=5:
        need(all(selection.get(k)==complete.get(k) for k in ('selection_policy','evaluation_candidates','terminal_pool_inventory')),
            'V5 fixed final selection/inventory bindings differ')
    if _version(complete)==6:need(all(selection.get(k)==complete.get(k)==cpu.get(k) for k in
        ('corridor_inventory','data_version','model_optimizer_version')),'V6 CPU/selection/inventory binding differs')
    if _version(complete)==7:need(all(selection.get(k)==complete.get(k)==cpu.get(k) for k in
        ('measured_inventory','measured_counts','data_version','model_optimizer_version')), 'V7 CPU/selection/measured provenance differs')
    if _version(complete)==8:need(all(selection.get(k)==complete.get(k)==cpu.get(k) for k in ('diverse_inventory','measured_counts','data_version','model_optimizer_version')), 'V8 CPU/selection/composite provenance differs')
    if _version(complete)==9:
        need(all(selection.get(k)==complete.get(k)==cpu.get(k) for k in
            ('refinement_inventory','task_subset','task_selection','measured_counts','data_version','model_optimizer_version')),
            'V9 CPU/selection/banks/subset/retained task ranking provenance differs')
        # validate_complete has just independently replayed every retained score.
        # Compare that exact selected result without another FK pass or ranking.
        chosen=next(r for r in complete['evaluation_candidates'] if r['step']==complete['selected_step'])
        expected=dict(selected_step=complete['selected_step'],task_score=chosen['score']['task_score'],verified_candidates=40)
        need(cpu.get('task_candidate_verification')==expected, 'V9 own CPU receipt requires exact all40 retained task replay')
    if _version(complete)>=4:
        need(selection.get('initialization')==complete['initialization']==cpu.get('initialization'),
            'V4 CPU/selected/completed initialization bindings differ')
    validate_code(complete['inputs']['code'],version=_version(complete))
    return dict(checkpoint=deepcopy(complete['best']),complete=deepcopy(ref),checkpoint_schema=pilot_schema),complete


def verify_selected(selection_ref,receipt_path):
    selection=runtime.read_bound(selection_ref);prepared=load_inputs(runtime.read_bound(selection['inputs']['config']))
    need(all(prepared[k]==selection[k] for k in ('scope','native_scope','optimization_scope','window_order')),
         'Selected mixed-source native scope/window ordering changed')
    torch,policy,packet=load_selected_policy(selection)
    execution=runtime.execution_settings(torch,policy,device='cpu');comparisons={};metrics={}
    for split in ('train','validation'):
        cpu=runtime.infer(torch,policy,prepared[split],packet['normalization'],device='cpu')
        cuda=runtime.load_predictions(selection['selected_predictions'][split],window_order_sha256=prepared['window_order'][split])
        metrics[split]=runtime.metrics(cpu,prepared[split])
        comparisons[split]=runtime.compare_devices(cpu,cuda,metrics[split],selection['selected_metrics'][split])
    value=dict(schema=_schemas(selection)[2],passed=True,selection=selection_ref,
        checkpoint=selection['best'],selected_step=selection['selected_step'],scope=selection['scope'],
        optimization_scope=selection['optimization_scope'],comparisons=comparisons,metrics=metrics,execution=execution,**FLAGS)
    if _version(selection)>=4:value['initialization']=deepcopy(selection['initialization'])
    if _version(selection)==6:
        value.update(data_version=5,model_optimizer_version=6,corridor_inventory=deepcopy(selection['corridor_inventory']))
    if _version(selection)==7:
        value.update(data_version=5,model_optimizer_version=7,measured_inventory=deepcopy(selection['measured_inventory']),
            measured_counts=deepcopy(selection['measured_counts']))
    if _version(selection)==8:
        value.update(data_version=5,model_optimizer_version=8,diverse_inventory=deepcopy(selection['diverse_inventory']),measured_counts=deepcopy(selection['measured_counts']))
    if _version(selection)==9:
        value.update({k:deepcopy(selection[k]) for k in ('data_version','model_optimizer_version','refinement_inventory','measured_counts','task_subset','task_selection')})
        value['task_candidate_verification']=packet['_task_candidate_verification']
    return runtime.write_new(receipt_path,value)


def fit(config_ref,output):
    started=time.monotonic();config=validate_config(runtime.read_bound(config_ref));prepared=load_inputs(config)
    pilot_schema,selection_schema,_,_=_schemas(config)
    output=Path(output).resolve();need(not output.exists(),'Fresh mixed-source output required; no resume');output.mkdir(parents=True)
    torch=None;policy=None;optimizer=None;completed=0;inputs=dict(config=config_ref,dataset=config['dataset'],
        auxiliary=config['auxiliary'],code=code_binding(version=_version(config)));protocol=training_protocol(config)
    if _version(config)==7:inputs['measured_bank']=deepcopy(config['measured_bank'])
    if _version(config) in (8,9):inputs['measured_banks']=deepcopy(config['measured_banks'])
    if _version(config)==9:inputs['task_subset']=deepcopy(config['task_subset'])
    groups=_groups(_version(config))
    optimization=new_optimization(version=_version(config))
    try:
        lease=runtime.owned_gpu(config['replacement_pod_gpu_binding']) if 'replacement_pod_gpu_binding' in config else runtime.owned_gpu()
        with lease as ownership:
            identity=runtime._identity();need(identity['pid']==identity['pgid']==identity['sid'],'Original guard-owned mixed-source session required')
            torch,policy,software=make_model(config,device='cuda',training=True)
            validate_official_forward(software);execution=runtime.execution_settings(torch,policy,device='cuda')
            need(sum(p.numel() for p in policy.parameters())==2658823,'Approved official ACT parameter count differs')
            if _version(config)>=4:
                warm_start=_warm_start(_version(config))
                inputs['initialization']=deepcopy(config['initialization'])
                initial=warm_start.initialize_policy(torch,policy,config['initialization'],normalization=prepared['normalization'],posture_head=config['posture_head'])
                inputs['initialization_equivalence']=runtime.write_new(output/'INITIALIZATION_EQUIVALENCE.json',initial)
                optimizer,optimizer_initialization=warm_start.fresh_optimizer(torch,policy,lr=config['lr'],weight_decay=config['weight_decay'])
                prepared['initialization']=dict(receipt=deepcopy(config['initialization']),equivalence=inputs['initialization_equivalence'],optimizer=optimizer_initialization)
            cache=build_cache(prepared,prepared['auxiliary'],torch=torch,device='cuda',optimizer_version=_version(config))
            if _version(config)>=5:prepared['terminal_pool_inventory']=deepcopy(cache['terminal_pool_inventory'])
            if _version(config)==6:
                prepared['corridor_inventory']=runtime.write_new(output/'CORRIDOR_INVENTORY.json',cache['corridor_inventory'])
            if _version(config)==7:
                prepared['measured_inventory']=runtime.write_new(output/'MEASURED_INVENTORY.json',cache['measured_inventory'])
                prepared['measured_counts']=dict(episodes=4,train_windows=cache['measured_count'],base_counts=deepcopy(NATIVE_COUNTS),normalization_refitted=False)
            if _version(config)==8:
                prepared['diverse_inventory']=runtime.write_new(output/'DIVERSE_INVENTORY.json',cache['diverse_inventory'])
                prepared['measured_counts']=measured_counts(cache['diverse_inventory'])
            if _version(config)==9:
                prepared['refinement_inventory']=runtime.write_new(output/'REFINEMENT_INVENTORY.json',cache['refinement_inventory'])
                prepared['measured_counts']=measured_counts(cache['refinement_inventory'])
            inputs['software']=runtime.write_new(output/'software.json',software)
            runtime.write_new(output/'STARTED.json',dict(schema=pilot_schema,inputs=inputs,scope=prepared['scope'],
                optimization_scope=prepared['optimization_scope'],protocol=protocol,execution=execution,
                ownership=ownership,identity=identity,**FLAGS))
            rng=np.random.Generator(np.random.PCG64(config['seed']))
            if _version(config)<4:optimizer=torch.optim.AdamW(policy.parameters(),lr=config['lr'],weight_decay=config['weight_decay'],betas=(.9,.999),eps=1e-8)
            evaluations=[];candidates=[];best_score=float('inf');last_metrics=None;seen=set()
            with (output/'metrics.jsonl').open('x',buffering=1) as log:
                for step in range(1,config['steps']+1):
                    need(time.monotonic()-started<config['time_limit_seconds'],'Finite mixed-source training deadline reached')
                    policy.train();ids=sample_cached_indices(rng,cache);batches=cached_batches(cache,ids)
                    optimizer.zero_grad(set_to_none=True);loss,losses=mixed_loss(policy,batches,version=_version(config))
                    need(bool(torch.isfinite(loss)),'Finite official mixed-source scalar ACT loss required')
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();completed=step
                    record_draws(optimization,cache,ids);seen=set(optimization['actual_optimized_train_ids'])
                    if step==1 or step%50==0:
                        row=dict(step=step,train_loss=float(loss.detach().cpu()),groups=losses,grad_norm=float(norm),
                            elapsed_wall_s=time.monotonic()-started)
                        if _version(config)>=2:row['posture_head_diagnostics']=posture_head.diagnostics(policy)
                        log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
                    if step%config['eval_every']:continue
                    row={'step':step};candidate_predictions={}
                    for split in ('train','validation'):
                        prediction=runtime.infer(torch,policy,prepared[split],prepared['normalization'],device='cuda')
                        row[split]=runtime.metrics(prediction,prepared[split])
                        if _version(config)==8:
                            candidate_predictions[split]=runtime.save_predictions(output/('candidate-'+str(step)+'-'+split+'.npy'),prediction,window_order_sha256=prepared['window_order'][split])
                        if _version(config)==9:
                            path=output/('candidate-'+str(step)+'-'+split+'.npy')
                            if split=='validation':candidate_predictions[split],row['score']=save_task_candidate(path,prediction,prepared)
                            else:candidate_predictions[split]=runtime.save_predictions(path,prediction,window_order_sha256=prepared['window_order'][split])
                            saved=runtime.load_predictions(candidate_predictions[split],window_order_sha256=prepared['window_order'][split])
                            row[split]=runtime.metrics(saved,prepared[split])
                    evaluations.append(row);last_metrics=row
                    log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
                    score=row['score']['task_score'] if _version(config)==9 else row['validation']['act']['full_chunk_mae_rad']
                    if _version(config)>=5:
                        path=output/('candidate-'+str(step)+'.pt')
                        need(not path.exists(),'Immutable new V5 evaluation candidate required')
                        ref=expanded.save_checkpoint(torch,path,checkpoint(policy,prepared,inputs,step,row,protocol,optimization))
                        if _version(config)==9:
                            candidates.append(dict(row,checkpoint=ref,predictions=candidate_predictions,state_proofs=_warm_start(9).tensor_proofs(policy.state_dict())))
                        else:candidates.append(dict(row,checkpoint=ref,predictions=candidate_predictions) if _version(config)==8 else dict(step=step,checkpoint=ref))
                    if (5<=_version(config)<=7 and step==1000) or ((_version(config)<5 or _version(config) in (8,9)) and score<best_score):
                        best_score=score;expanded.save_checkpoint(torch,output/'best_auxiliary_act.pt',
                            checkpoint(policy,prepared,inputs,step,row,protocol,optimization))
                scope.validate_train16_metadata(prepared['optimization_scope'],actual_train_ids=seen)
                last_packet=checkpoint(policy,prepared,inputs,completed,last_metrics,protocol,optimization)
                last_packet['optimizer_state_dict']=optimizer.state_dict()
                last=expanded.save_checkpoint(torch,output/'last_auxiliary_act.pt',last_packet)
            selected=selected_step(evaluations,version=_version(config))
            best=(deepcopy(next(row['checkpoint'] for row in candidates if row['step']==selected)) if _version(config)==9
                else runtime.reference(output/'best_auxiliary_act.pt'))
            packet=torch.load(runtime.bound_path(best),map_location='cpu',weights_only=True)
            need(packet['step']==selected,'Earliest minimum full raw masked VAL selected checkpoint differs')
            scope.validate_train16_metadata(prepared['optimization_scope'],actual_train_ids=packet['optimization']['actual_optimized_train_ids'])
            policy.load_state_dict(packet['state_dict'],strict=True);predictions={};selected_metrics={}
            if _version(config)!=9:
                for split in ('train','validation'):
                    value=runtime.infer(torch,policy,prepared[split],prepared['normalization'],device='cuda')
                    predictions[split]=runtime.save_predictions(output/('selected-'+split+'.npy'),value,window_order_sha256=prepared['window_order'][split])
                    selected_metrics[split]=runtime.metrics(value,prepared[split])
            selection={k:deepcopy(packet[k]) for k in ('scope','native_scope','optimization_scope','auxiliary_counts','oracle_authentication')}
            if _version(config)>=4:selection['initialization']=deepcopy(packet['initialization'])
            if _version(config)>=5:
                selection.update(selection_policy=deepcopy(SELECTION_POLICY_V9 if _version(config)==9 else SELECTION_POLICY_V8 if _version(config)==8 else FINAL_SELECTION_V5),evaluation_candidates=candidates,
                    terminal_pool_inventory=deepcopy(packet['terminal_pool_inventory']))
            if _version(config)==6:
                selection.update(data_version=5,model_optimizer_version=6,corridor_inventory=deepcopy(packet['corridor_inventory']))
            if _version(config)==7:
                selection.update(data_version=5,model_optimizer_version=7,measured_inventory=deepcopy(packet['measured_inventory']),measured_counts=deepcopy(packet['measured_counts']))
            if _version(config)==8:
                selection.update(data_version=5,model_optimizer_version=8,diverse_inventory=deepcopy(packet['diverse_inventory']),measured_counts=deepcopy(packet['measured_counts']),completed_steps=completed)
                selected_candidate=next(row for row in candidates if row['step']==selected)
                need(all(selected_metrics[s]==selected_candidate[s] for s in ('train','validation')), 'V8 selected full evaluation metrics changed')
                predictions=deepcopy(selected_candidate['predictions'])
            if _version(config)==9:
                selection.update({k:deepcopy(packet[k]) for k in ('data_version','model_optimizer_version','refinement_inventory','measured_counts','task_subset')})
                selection['completed_steps']=completed
                selected_candidate=next(row for row in candidates if row['step']==selected)
                predictions=deepcopy(selected_candidate['predictions']);selected_metrics={s:deepcopy(selected_candidate[s]) for s in NATIVE_COUNTS}
            selection.update(schema=selection_schema,selected_step=selected,best=best,inputs=inputs,
                window_order=prepared['window_order'],selected_predictions=predictions,selected_metrics=selected_metrics,
                selected_optimization=packet['optimization'],optimization=optimization,protocol=protocol,**FLAGS)
            if _version(config)==9:selection['task_selection']=_task_selection(selection)
            if _version(config)>=2:
                selection.update(posture_target_audit=packet['posture_target_audit'],
                    posture_head_diagnostics=posture_head.diagnostics(policy))
            selected_ref=runtime.write_new(output/'SELECTED.json',selection)
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
            subprocess.run([sys.executable,'-B','-m',expanded.MODULE,'--verify-selected',selected_ref['path'],
                '--receipt',str(output/'CPU_EQUIVALENCE.json')],check=True,env=environment,timeout=1800)
            cpu=runtime.reference(output/'CPU_EQUIVALENCE.json');need(runtime.read_bound(cpu).get('passed') is True,'Actual CPU equivalence required')
            complete=dict(selection,schema=pilot_schema,complete=True,completed_steps=completed,last=last,
                task_contract=runtime.task_contract(),model=runtime.shared.native_spec(),evaluations=evaluations,cpu_equivalence=cpu,
                execution=execution,ownership=ownership,identity=identity,schedule=deepcopy(config),
                elapsed_wall_s=time.monotonic()-started,training_log=runtime.reference(output/'metrics.jsonl'))
            runtime.write_new(output/'COMPLETE.json',complete);return complete
    except BaseException as error:
        partial=None
        if policy is not None:
            packet=checkpoint(policy,prepared,inputs,completed,None,protocol,optimization);packet.update(complete=False,error=(type(error).__name__+': '+str(error))[:2000])
            if optimizer is not None:packet['optimizer_state_dict']=optimizer.state_dict()
            partial=expanded.save_checkpoint(torch,output/'failed_auxiliary_act.pt',packet)
        (output/'traceback.txt').write_text(traceback.format_exc())
        runtime.write_new(output/'FAILED.json',dict(schema=pilot_schema,complete=False,completed_steps=completed,
            inputs=inputs,partial_checkpoint=partial,error=(type(error).__name__+': '+str(error))[:2000],traceback=runtime.reference(output/'traceback.txt'),**FLAGS))
        raise
