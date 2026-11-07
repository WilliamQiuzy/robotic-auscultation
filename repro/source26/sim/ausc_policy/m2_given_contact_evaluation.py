"""Finite once-only paired given-pose simulation evaluation.

Identity/pose/seed/all-ten priority is frozen from metadata before TEST payload
access. Planning and native capabilities are separate authenticated stages;
neither a caller flag nor a synthetic fixture authorizes heldout access.
"""
from copy import deepcopy
from contextvars import ContextVar
from functools import wraps
import argparse
import hashlib
import gzip
import json
import lzma
import os
from pathlib import Path
import re
import subprocess
import sys
import stat
from types import MethodType

import numpy as np

from . import m2_given_contact_deployment as deployment

runtime=deployment.runtime
need=runtime.need
MODULE='sim.ausc_policy.m2_given_contact_evaluation'
BODY_PLAN_SCHEMA='m2_given_contact_heldout_body_plan_v1'
PREPARATION_SCHEMA='m2_given_contact_sealed_test_preparation_v1'
PAIR_PLAN_SCHEMA='m2_given_contact_heldout_pair_plan_v1'
CAPABILITY_SCHEMA='m2_given_contact_sealed_test_capability_v1'
PREP_CAPABILITY_SCHEMA='m2_given_contact_sealed_test_preparation_capability_v1'
PREP_PLAN_SCHEMA='m2_given_contact_frozen_preparation_plan_v1'
PREPARED_SCHEMA='m2_given_contact_sealed_prepared_inventory_v1'
SMOKE_SCHEMA='m2_given_contact_actual_bridge_smoke_qualification_v1'
ROUTE_KEY='m2_given_contact_evaluation'
PREP_ROUTE_SCHEMA='m2_given_contact_sealed_test_preparation_route_v1'
NATIVE_ROUTE_SCHEMA='m2_given_contact_sealed_test_native_route_v1'
PLANNING_KEY='m2_given_contact_heldout_planning'
PLANNING_SCHEMA='m2_given_contact_heldout_bounded_planning_route_v1'
PLANNING_RESULT_SCHEMA='m2_given_contact_actual_bounded_planning_v1'
PREPARATION_RESULT_SCHEMA='m2_given_contact_actual_cpu_preparation_v1'
AGGREGATE_KEY='m2_given_contact_heldout_planning_aggregate'
AGGREGATE_SCHEMA='m2_given_contact_sealed_planning_aggregate_v1'
FROZEN_PAIR_SCHEMA='m2_given_contact_frozen_heldout_pairs_v1'
SUMMARY_SCHEMA='m2_given_contact_paired_heldout_summary_v1'
CURRENT_NATIVE_COMMON_SCHEMA='m2_given_contact_current_raw_to_dense_sdk_native_common_v1'
TARGETS=tuple(s+str(i) for i in range(1,6) for s in 'LR')
ARMS=('expert','act')
_VERIFICATION=ContextVar('heldout_verification_call',default=None)


def _verification_scope(operation):
    """Reuse completed metadata authentication only within this call stack."""
    @wraps(operation)
    def verified(*args,**kwargs):
        if _VERIFICATION.get() is not None:return operation(*args,**kwargs)
        token=_VERIFICATION.set({})
        try:return operation(*args,**kwargs)
        finally:_VERIFICATION.reset(token)
    return verified


def configuration_projection(config):
    """Only named authorization refs are excluded, never controls or sources."""
    value=deepcopy(config);route=value.get(ROUTE_KEY)
    need(isinstance(route,dict) and route.get('schema') in (PREP_ROUTE_SCHEMA,NATIVE_ROUTE_SCHEMA) and
         'capability' in route, 'Exact formal preparation/native configuration required')
    ref=route.pop('capability')
    if ref is not None:runtime._reference_shape(ref)
    if 'm2_given_contact_deployment' in value:
        dep=value['m2_given_contact_deployment']
        need(route['schema']==NATIVE_ROUTE_SCHEMA and route.get('arm')=='act' and
             isinstance(dep,dict) and dep.get('permit')==ref,
             'ACT deployment and evaluation must bind the exact same native capability')
        dep.pop('permit')
    return value


def _file(ref):
    need(isinstance(ref,dict) and set(ref) in ({'file','sha256'},{'file','sha256','bytes'},
        {'path','sha256'},{'path','sha256','bytes'}), 'Exact immutable file reference required')
    value=dict(path=ref.get('path',ref.get('file')),sha256=ref['sha256'])
    if 'bytes' in ref:value['bytes']=ref['bytes']
    runtime._reference_shape(value);return value


def _identity_metadata():
    contract=runtime.shared.contract
    value=runtime.read_bound(dict(path=str(contract.IDENTITY_SPLIT_PATH),sha256=contract.IDENTITY_SPLIT_SHA256))
    splits=value['splits'];ids=splits['test']
    need(len(ids)==34 and len(set(ids))==34 and all(identity not in contract.QUARANTINED_IDENTITIES and
         [name for name,members in splits.items() if identity in members]==['test'] for identity in ids),
         'Exactly the original 34 nonquarantined TEST identities required')
    return sorted(ids)


def _sources(processing):
    from . import m2_pilot_config,patient_scene,geometry_plan,tool_geometry_review
    inventory=deployment._sources(processing)
    inventory.update({module.__name__:runtime.reference(module.__file__) for module in
        (m2_pilot_config,patient_scene,geometry_plan,tool_geometry_review)})
    inventory[MODULE]=runtime.reference(__file__)
    descriptor=runtime.read_bound(processing['processing_freeze'])
    for ref in inventory.values():
        need(ref in descriptor['files'].values(), 'Every heldout helper must belong to the final complete processing freeze')
    return inventory


def _prerequisites(admission,smokes):
    runtime.bound_path(admission);runtime.bound_path(smokes)
    context=_VERIFICATION.get();key=runtime.digest([admission,smokes])
    if context is None:return _verified_prerequisites(admission,smokes)
    if key not in context:context[key]=deepcopy(_verified_prerequisites(admission,smokes))
    return deepcopy(context[key])


def _verified_prerequisites(admission,smokes):
    bundle=deployment._admission_metadata(admission);qualified=verify_bridge_smokes(smokes)
    need(qualified['admission']==admission and qualified['model']==bundle['admission']['model'] and
         qualified['processing']==bundle['config']['processing'], 'Four actual bridge smokes must use the admitted model/root')
    permit=deployment.verify_permit(qualified['permit']);mapping,_=deployment._allocation_mapping(
        permit['allocation_mapping'],permit['processing'])
    counts={target:sum(row.get('target')==target and row.get('fresh_plan') is not None for row in
        mapping['ordered_allocations']['train']) for target in TARGETS}
    templates={};pose_sources={}
    for pose in sorted(deployment.POSE_TEMPLATES):
        row=next(row for row in permit['attempts'] if row['split']=='train' and row['pose_template']==pose)
        templates[pose]=runtime.read_bound(_file(row['pose_binding']['preparation_config']))
        spec=templates[pose]['patient_asset']
        manifest=runtime.read_bound(dict(path=str(Path(spec['directory'])/'asset_manifest.json'),sha256=spec['manifest_sha256']))
        records=[record for record in manifest if record.get('file')==spec['file']]
        need(len(records)==1 and records[0].get('identity')==row['subject_id'] and records[0].get('split')=='train' and
             isinstance(records[0].get('source_pose'),dict) and records[0]['source_pose'], 'Original authenticated TRAIN pose metadata required')
        pose_sources[pose]=deepcopy(records[0]['source_pose'])
    common=_qualified_native_common([runtime.read_bound(arm['config']) for arm in qualified['arms']],
        permit,mapping,bundle['config']['processing'])
    return dict(admission=deepcopy(admission),model=deepcopy(bundle['admission']['model']),native_common=common,
        processing=deepcopy(bundle['config']['processing']),source_inventory=_sources(bundle['config']['processing']),
        train_counts=counts,templates=templates,pose_sources=pose_sources)


def _patient_metadata(spec,identity,pose_source,template):
    """Authenticate JSON roles only; the body file is deliberately not opened."""
    need(isinstance(spec,dict) and spec.get('expected_split')=='test' and
         spec.get('identity_split')==template.get('identity_split'),
         'Exact frozen TEST metadata role required')
    variable={'directory','file','expected_split','manifest_sha256','summary_sha256'}
    need({k:v for k,v in spec.items() if k not in variable}=={k:v for k,v in template.items() if k not in variable},
         'Patient anchors/layout/shared source roles must match the authenticated known-pose template')
    root=Path(spec['directory'])
    need(root.is_absolute() and '..' not in root.parts and str(root)==spec['directory'] and
         Path(spec['file']).name==spec['file'] and spec['file'].endswith('.npz'), 'Canonical declared body path required')
    refs={key:dict(path=str(root/name),sha256=spec[key+'_sha256']) for key,name in
        (('manifest','asset_manifest.json'),('summary','summary.json'))}
    split=dict(path=spec['identity_split'],sha256=runtime.shared.contract.IDENTITY_SPLIT_SHA256)
    runtime.read_bound(split)
    refs['identity_split']=runtime.reference(split['path'])
    need(refs['identity_split']['sha256']==split['sha256'], 'Authenticated historical identity split bytes changed')
    manifest=runtime.read_bound(refs['manifest']);summary=runtime.read_bound(refs['summary'])
    rows=[r for r in manifest if r.get('file')==spec['file']]
    need(summary.get('schema')=='m2_observed_seated_retarget_v1' and
         summary.get('identity_split_sha256')==runtime.shared.contract.IDENTITY_SPLIT_SHA256 and len(rows)==1,
         'Original observed retarget metadata/unique body required')
    row=rows[0]
    need(row.get('identity')==identity and row.get('split')=='test' and row.get('source_pose')==pose_source and
         row.get('mesh_scale')==1. and row.get('numeric_candidate_screen_passed') is True,
         'Heldout identity/known pose/numeric provenance differs before body access')
    payload=dict(path=str(root/spec['file']),sha256=row['sha256']);runtime._reference_shape(payload)
    return dict(metadata=refs,body=payload,record=deepcopy(row))


def freeze_body_plan(admission,smokes,*,seed,patient_specs,root,output):
    authority=_prerequisites(admission,smokes);ids=_identity_metadata()
    need(isinstance(patient_specs,dict) and set(patient_specs)==set(ids), 'All 34 original TEST metadata specifications required')
    rows=allocate_bodies(ids,seed=seed,train_feasibility=authority['train_counts'])
    layout=preparation_layout(rows,root=root,planning_base=authority['native_common']['base']);require_future_outputs_absent(layout)
    patients={row['subject_id']:_patient_metadata(patient_specs[row['subject_id']],row['subject_id'],
        authority['pose_sources'][row['pose_template']],authority['templates'][row['pose_template']]['patient_asset']) for row in rows}
    value=dict(schema=BODY_PLAN_SCHEMA,protocol=evaluation_protocol(),admission=deepcopy(admission),
        bridge_smokes=deepcopy(smokes),seed=seed,patient_specs=deepcopy(patient_specs),patients=patients,
        layout=layout,**{k:deepcopy(authority[k]) for k in ('model','processing','source_inventory','train_counts','templates','native_common','pose_sources')})
    return runtime.write_new(output,value)


@_verification_scope
def verify_body_plan(ref):
    value=runtime.read_bound(ref)
    need(set(value)=={'schema','protocol','admission','bridge_smokes','seed','patient_specs','patients','layout',
        'model','processing','source_inventory','train_counts','templates','native_common','pose_sources'} and value['schema']==BODY_PLAN_SCHEMA and
        value['protocol']==evaluation_protocol(), 'Exact frozen heldout metadata body plan required')
    authority=_prerequisites(value['admission'],value['bridge_smokes'])
    need(all(value[k]==authority[k] for k in ('model','processing','source_inventory','train_counts','templates','native_common','pose_sources')),
         'Admitted model/actual smokes/final source freeze changed')
    rows=allocate_bodies(_identity_metadata(),seed=value['seed'],train_feasibility=authority['train_counts'])
    need(value['layout']==preparation_layout(rows,root=value['layout']['root'],planning_base=authority['native_common']['base']) and
         set(value['patient_specs'])==set(value['patients'])=={row['subject_id'] for row in rows},
         'Immutable original-34 metadata allocation changed')
    for row in rows:
        identity=row['subject_id']
        need(value['patients'][identity]==_patient_metadata(value['patient_specs'][identity],identity,
             authority['pose_sources'][row['pose_template']],authority['templates'][row['pose_template']]['patient_asset']),
             'Patient metadata roles changed before payload access')
    return value


def _preparation_configuration(body,row):
    config=deepcopy(body['templates'][row['pose_template']])
    need(ROUTE_KEY not in config and 'm2_given_contact_deployment' not in config and config.get('prepare_only') is True,
         'Only authenticated original prepare-only template is permitted')
    config['patient_asset']=deepcopy(body['patient_specs'][row['subject_id']])
    config[ROUTE_KEY]=dict(schema=PREP_ROUTE_SCHEMA,subject_id=row['subject_id'],pose_template=row['pose_template'],seed=row['seed'])
    return config


def freeze_preparation_plan(body_ref,configs,*,output):
    body=verify_body_plan(body_ref);layout=body['layout'];require_future_outputs_absent(layout)
    need(isinstance(configs,list) and len(configs)==34, 'Exactly 34 ordered frozen preparation configurations required')
    configurations=[]
    for row,config in zip(layout['bodies'],configs):
        projected=configuration_projection(config)
        need(config[ROUTE_KEY]['capability'] is None and projected==_preparation_configuration(body,row),
             'Preparation controls/known pose/identity/seed/source fields differ from authenticated template')
        path=Path(row['preparation_config'])
        need(not path.exists() and not path.is_symlink(), 'A future immutable preparation configuration already exists')
        configurations.append(dict(subject_id=row['subject_id'],config=str(path),output=row['prepared_output'],
            configuration=projected,configuration_sha256=runtime.digest(projected)))
    return runtime.write_new(output,dict(schema=PREP_PLAN_SCHEMA,body_plan=deepcopy(body_ref),configurations=configurations))


@_verification_scope
def verify_preparation_plan(ref):
    value=runtime.read_bound(ref)
    need(set(value)=={'schema','body_plan','configurations'} and value['schema']==PREP_PLAN_SCHEMA,
         'Exact immutable 34-configuration preparation plan required')
    body=verify_body_plan(value['body_plan']);rows=value['configurations']
    need(isinstance(rows,list) and len(rows)==34, 'All original 34 preparation configurations required')
    for declared,row in zip(rows,body['layout']['bodies']):
        projected=_preparation_configuration(body,row)
        need(declared==dict(subject_id=row['subject_id'],config=row['preparation_config'],output=row['prepared_output'],
            configuration=projected,configuration_sha256=runtime.digest(projected)), 'Frozen preparation configuration changed')
    return value


def _capability(body_ref,prep_ref,stage):
    body=verify_body_plan(body_ref)
    return dict(schema=PREP_CAPABILITY_SCHEMA if stage=='preparation' else CAPABILITY_SCHEMA,stage=stage,
        body_plan=deepcopy(body_ref),preparation_plan=deepcopy(prep_ref),protocol=deployment.protocol(),
        task_contract=runtime.task_contract(),production_ready=True,simulation_only=True,scope='sealed_test_once_only',
        physical_acceptance=False,real_hardware_validated=False,closed_loop_evaluated=False,
        **{key:deepcopy(body[key]) for key in ('admission','model','processing','source_inventory')})


def issue_preparation_capability(preparation_plan,*,output):
    plan=verify_preparation_plan(preparation_plan);body=verify_body_plan(plan['body_plan'])
    require_future_outputs_absent(body['layout'])
    for row in plan['configurations']:
        path=Path(row['config']);need(not path.exists() and not path.is_symlink(), 'Immutable configuration already exists')
    value=_capability(plan['body_plan'],preparation_plan,'preparation');value['configurations']=deepcopy(plan['configurations'])
    cap=runtime.write_new(output,value);configs=[]
    for row in plan['configurations']:
        config=deepcopy(row['configuration']);config[ROUTE_KEY]['capability']=cap
        configs.append(dict(subject_id=row['subject_id'],config=runtime.write_new(row['config'],config),output=row['output']))
    aggregate=_preparation_aggregate_configuration(cap,body,configs)
    owner_config=runtime.write_new(body['layout']['preparation_owner']['config'],aggregate)
    return dict(capability=cap,configurations=configs,owner_configuration=owner_config)


def _tree(root,*,payloads=True):
    """Closed, link-free prepared output; preserve directory names as well."""
    root=Path(root);need(root.is_dir() and not root.is_symlink(), 'Missing real prepared output directory')
    files={};directories=[]
    for path in sorted(root.rglob('*')):
        mode=path.lstat();name=path.relative_to(root).as_posix()
        need(not path.is_symlink(), 'Links are forbidden in the prepared output inventory')
        if stat.S_ISDIR(mode.st_mode):directories.append(name)
        else:
            need(stat.S_ISREG(mode.st_mode) and mode.st_nlink==1, 'One regular prepared artifact per path required')
            files[name]=runtime.reference(path) if payloads else str(path)
    return dict(root=str(root),files=files,directories=directories)


def _prepared_header(cap,body,declaration):
    identity=declaration['subject_id'];root=Path(declaration['output'])
    config_ref=runtime.reference(declaration['config']);config=runtime.read_bound(config_ref)
    need(config.get(ROUTE_KEY,{}).get('capability')==cap and
         configuration_projection(config)==declaration['configuration'], 'Actual immutable preparation config changed')
    complete_ref=runtime.reference(root/'COMPLETE.json');patient_ref=runtime.reference(root/'patient_receipt.json')
    complete=runtime.read_bound(complete_ref);patient=runtime.read_bound(patient_ref)
    need(complete.get('completed') is True and complete.get('patient')==patient and
         complete.get('geometry_id')==runtime.geometry.GEOMETRY_ID and
         complete.get('source_sha256')==runtime.geometry.GEOMETRY_SOURCE_SHA256 and
         complete.get('preparation_backend')=='cpu_usd_authoring_no_simulation_app' and
         all(complete.get(key) is False for key in ('physics_executed','physical_validation','training_eligible')) and
         patient.get('identity')==identity and patient.get('split')=='test' and
         patient.get('asset_sha256')==body['patients'][identity]['body']['sha256'] and
         patient.get('identity_split_sha256')==runtime.shared.contract.IDENTITY_SPLIT_SHA256 and
         patient.get('non_patient_scene_unchanged') is True, 'Prepared TEST patient identity/metadata differs')
    runtime.shared.contract._pose(complete['world_from_left_base'],'actual prepared robot base')
    return dict(subject_id=identity,output=str(root),preparation_config=config_ref,prepared_complete=complete_ref,
        patient_receipt=patient_ref)


def _prepared_payloads(row,files):
    complete=runtime.read_bound(row['prepared_complete']);geometry=complete['geometry_artifacts']
    required={'candidate_physics.usdc':complete['effective_scene_sha256'],
        'geometry/collision_geometry.json':geometry['collision_geometry_sha256'],
        'geometry/collision_geometry.npz':geometry['collision_arrays_sha256']}
    need(all(name in files and files[name]['sha256']==digest for name,digest in required.items()),
         'Prepared scene/collision artifact SHA differs from its authenticated COMPLETE')


def freeze_prepared_inventory(preparation_capability,*,output):
    cap=verify_capability(preparation_capability,stage='preparation');body=verify_body_plan(cap['body_plan'])
    owner=_preparation_owner(preparation_capability,cap,body)
    # Authenticate ALL declared body headers before hashing any USD/NPZ payload.
    rows=[_prepared_header(preparation_capability,body,row) for row in cap['configurations']]
    for row in rows:
        row['inventory']=_tree(row['output']);_prepared_payloads(row,row['inventory']['files'])
    need(rows==[dict(_prepared_header(preparation_capability,body,declaration),inventory=row['inventory'])
         for declaration,row in zip(cap['configurations'],rows)], 'Prepared metadata changed during payload inventory')
    return runtime.write_new(output,dict(schema=PREPARED_SCHEMA,preparation_capability=deepcopy(preparation_capability),
        preparation_owner=owner,bodies=rows))


@_verification_scope
def verify_prepared_inventory(ref,*,payloads=True):
    value=runtime.read_bound(ref)
    need(set(value)=={'schema','preparation_capability','preparation_owner','bodies'} and value['schema']==PREPARED_SCHEMA,
         'Exact post-preparation closed inventory required')
    cap=verify_capability(value['preparation_capability'],stage='preparation');body=verify_body_plan(cap['body_plan'])
    need(value['preparation_owner']==_preparation_owner(value['preparation_capability'],cap,body), 'Original CPU preparation owner changed')
    rows=value['bodies'];need(isinstance(rows,list) and len(rows)==34, 'All 34 prepared outputs required')
    for row,declaration in zip(rows,cap['configurations']):
        expected=_prepared_header(value['preparation_capability'],body,declaration)
        need(set(row)==set(expected)|{'inventory'} and all(row[k]==v for k,v in expected.items()),
             'Prepared identity/path/metadata role substitution')
        inventory=row['inventory'];need(inventory['root']==row['output'] and set(inventory)=={'root','files','directories'},
             'Exact prepared tree role required')
        # Named metadata refs must be members of this closed tree, not aliases.
        for key,name in (('prepared_complete','COMPLETE.json'),('patient_receipt','patient_receipt.json')):
            need(inventory['files'].get(name)==row[key], 'Prepared metadata not bound by the closed inventory')
        _prepared_payloads(row,inventory['files'])
    for row in rows:
        actual=_tree(row['output'],payloads=payloads);expected=row['inventory']
        need(actual['directories']==expected['directories'] and set(actual['files'])==set(expected['files']),
             'Prepared closed tree added/deleted/renamed artifacts')
        if payloads:need(actual==expected, 'Prepared output artifact bytes changed')
    return value


def authorize_prepared(context,preparation_config,prepared_complete,patient_receipt):
    need(isinstance(context,dict) and set(context)=={'capability','prepared_inventory','subject_id'},
         'Exact authenticated preparation capability/inventory context required')
    inventory=verify_prepared_inventory(context['prepared_inventory'],payloads=False)
    need(inventory['preparation_capability']==context['capability'], 'Prepared inventory and capability differ')
    matches=[row for row in inventory['bodies'] if row['subject_id']==context['subject_id']]
    need(len(matches)==1, 'Subject outside exact prepared capability')
    row=matches[0]
    for key,ref in (('preparation_config',preparation_config),('prepared_complete',prepared_complete),('patient_receipt',patient_receipt)):
        supplied=_file(ref);expected=row[key]
        need(all(supplied[k]==expected[k] for k in supplied), 'Prepared metadata role alias/substitution')
    return row


def _planning_configuration(cap,inventory_ref,row,body_row,target):
    """Original geometry builder's pure assembly; no payload hashing here."""
    from . import m2_pilot_config as builder,geometry_plan
    prep=runtime.read_bound(row['preparation_config']);complete=runtime.read_bound(row['prepared_complete'])
    patient=runtime.read_bound(row['patient_receipt']);point=builder._target(patient,target)
    config={key:deepcopy(prep[key]) for key in
        ('geometry_id','geometry_source','geometry_source_sha256','freshly_generated','posture_limits')}
    config.update(collision_directory=str(Path(row['output'])/'geometry'),
        world_from_left_base=complete['world_from_left_base'],**complete['geometry_artifacts'],
        patient_receipt={k:row['patient_receipt']['path' if k=='file' else k] for k in ('file','sha256')},
        targets=[point],loaded_compression_m=.0025,fresh_pre_gap_search_m=[.025,.03],
        seeds_per_candidate=12,per_gap_seconds=180.,plane_tilt_limit_deg=3.,collision_profile='full')
    config[PLANNING_KEY]=dict(schema=PLANNING_SCHEMA,capability=deepcopy(cap),prepared_inventory=deepcopy(inventory_ref),
        subject_id=row['subject_id'],target_id=target,output=body_row['planning'][target]['output'],
        entry_module=MODULE,delegate_module=geometry_plan.__name__,delegate_source=runtime.reference(geometry_plan.__file__))
    return config


def planning_sources():
    from . import geometry_plan,geometry_qualification,tool_geometry_review
    return {MODULE:runtime.reference(__file__),**{module.__name__:runtime.reference(module.__file__) for module in
        (geometry_plan,geometry_qualification,tool_geometry_review)}}


def _aggregate_owner(ref,config_ref,receipt,processing,stage_name):
    from . import process_guard as guard
    terminal=runtime.read_bound(ref);root=runtime.bound_path(ref).parent;run=runtime.read_bound(runtime.reference(root/'run.json'))
    stage=terminal.get('stage',{});source=runtime.reference(guard.__file__);identity=receipt.get('identity',{})
    group=stage.get('process_identity',{});worker=runtime.read_bound(runtime.reference(root/'output/native_worker_exit.json'))
    config=runtime.read_bound(config_ref);route=config[AGGREGATE_KEY];owner=route['owner']
    need(route.get('stage')==stage_name, 'Wrong original CPU aggregate execution stage')
    need(str(root/'output')==owner['output'] and root.name==owner['run_id'] and config_ref['path']==owner['config'],
         'Planning output/owner/config differs from the metadata-only predeclaration')
    planning_aggregate_preflight(config,output=owner['review']);complete_ref=runtime.reference(Path(owner['review'])/'COMPLETE.json')
    complete=runtime.read_bound(complete_ref)
    need(runtime.bound_path(ref)==root/'result.json' and terminal.get('completed') is True and terminal.get('gpu_released') is True and
         all(terminal.get(k)==run.get(k) for k in ('config','config_sha256','command','code_commit','resource_guard','run_id','started_utc')) and
         terminal.get('code_commit')==runtime.read_bound(processing['processing_freeze'])['reviewed_commit'] and
         terminal.get('config_sha256')==config_ref['sha256'] and terminal.get('config')==runtime.read_bound(config_ref) and
         terminal['config'].get('experiment')=='m2_geometry_plan' and terminal.get('run_id')==owner['run_id'] and
         terminal.get('command',[])[1:]==['-u','-B','-m','sim.ausc_policy.tool_geometry_review','--config',config_ref['path'],
             '--output',owner['output']], 'Original frozen CPU planning owner/config/source binding required')
    need(source['sha256']==runtime.GUARD_SHA256 and terminal.get('resource_guard')==dict(path=source['path'],sha256=source['sha256'],
         gpu_exit_grace_s=guard.EXIT_GPU_GRACE_S) and stage.get('passed') is True and stage.get('exit_code')==0 and
         stage.get('process_group_drained') is True and all(stage.get(k)==[] for k in ('errors','signals','deferred_signals')) and
         worker.get('worker')==MODULE and worker.get('pid')==identity.get('pid') and
         deployment._same_live_process(worker.get('identity'),identity) and
         worker.get('returncode')==0 and worker.get('clean_exit') is True and worker.get('timed_out') is False and
         worker.get('complete_file_present') is True and worker.get('configuration')==config_ref and
         worker.get('command')==[config['isaac_python'],'-u','-B','-m',MODULE,'--config',config_ref['path'],'--output',owner['review']] and
         worker.get('output')==owner['review'] and worker.get('source_inventory')==planning_sources() and worker.get('complete')==complete_ref and
         identity.get('pgid')==identity.get('sid')==group.get('pgid')==group.get('sid') and
         group.get('members',{}).get(str(identity.get('pid')))==identity.get('start_ticks'),
         'Bare planner, abnormal CPU owner or missing exact wrapper child identity refused')
    need(set(complete)=={'schema','stage','configuration','prepared_inventory','source_inventory','identity','slots','completed',
         'physics_executed','training_eligible','wall_s'} and complete['schema']==AGGREGATE_SCHEMA and
         complete['stage']==stage_name and
         complete['configuration']==config_ref and complete['prepared_inventory']==route['prepared_inventory'] and
         complete['source_inventory']==planning_sources() and deployment._same_live_process(complete['identity'],identity) and
         complete['completed'] is True and complete['physics_executed'] is False and complete['training_eligible'] is False and
         runtime.read_bound(runtime.reference(root/'output/COMPLETE.json'))==dict(complete,isaac_exit_code=0),
         'Actual aggregate completion/source/child identity differs')
    if stage_name=='planning':_materialized_configurations(route['slots'],owner['review'])
    count=340 if stage_name=='planning' else 34;keys=('subject_id','target_id') if stage_name=='planning' else ('subject_id',)
    name='heldout_planning.json' if stage_name=='planning' else 'heldout_preparation.json'
    need(len(complete['slots'])==count and all(set(r)==set(keys)|{'receipt'} and
         all(r[key]==slot[key] for key in keys) and r['receipt']['path']==str(Path(slot['output'])/name)
         for r,slot in zip(complete['slots'],route['slots'])), 'Original aggregate must preserve every ordered actual slot receipt')
    for item in complete['slots']:runtime.bound_path(item['receipt'])
    return complete


def _planning_owner(ref,config_ref,receipt,processing):
    return _aggregate_owner(ref,config_ref,receipt,processing,'planning')


def _preparation_owner(cap_ref,cap,body):
    owner=body['layout']['preparation_owner'];ref=runtime.reference(Path(owner['output']).parent/'result.json')
    config_ref=runtime.reference(owner['config']);config=runtime.read_bound(config_ref)
    need(config==_preparation_aggregate_configuration(cap_ref,body,_preparation_slots(cap_ref,cap)),
         'Original preparation aggregate/config/order changed')
    first=runtime.read_bound(runtime.reference(Path(owner['review'])/'prepared'/body['layout']['bodies'][0]['subject_id']/'heldout_preparation.json'))
    complete=_aggregate_owner(ref,config_ref,first,body['processing'],'preparation')
    for item,slot in zip(complete['slots'],config[AGGREGATE_KEY]['slots']):
        receipt=runtime.read_bound(item['receipt'])
        need(set(receipt)=={'schema','configuration','capability','output','complete','source_inventory','identity','command','completed'} and
             receipt['schema']==PREPARATION_RESULT_SCHEMA and receipt['configuration']==slot['config'] and receipt['capability']==cap_ref and
             receipt['output']==slot['output'] and receipt['source_inventory']==planning_sources() and receipt['completed'] is True and
             deployment._same_live_process(receipt['identity'],first['identity']) and receipt['command']==
             ['sim.ausc_policy.geometry_qualification','--cpu-prepare-only','--config',slot['config']['path'],'--output',slot['output']] and
             receipt['complete']==runtime.reference(Path(slot['output'])/'COMPLETE.json'), 'Only actual explicit CPU-only preparation wrapper receipts accepted')
    return ref


def _planning_proofs(prepared_ref,records):
    inventory=verify_prepared_inventory(prepared_ref,payloads=False);cap=verify_capability(inventory['preparation_capability'],stage='preparation')
    body=verify_body_plan(cap['body_plan']);expected=[(row,b,target) for row,b in zip(inventory['bodies'],body['layout']['bodies'])
        for target in b['target_priority']]
    need(isinstance(records,list) and len(records)==340, 'Every original 340 planning execution outcomes required')
    # Check every role and path before opening even the first outcome leaf.
    for record,(row,b,target) in zip(records,expected):
        need(isinstance(record,dict) and set(record)=={'subject_id','target_id','receipt','owner_terminal'} and
             (record['subject_id'],record['target_id'])==(row['subject_id'],target), 'Planning execution identity/order differs')
        _declared_json(record['receipt'],str(Path(b['planning'][target]['output'])/'heldout_planning.json'))
        runtime._reference_shape(record['owner_terminal'])
        need(record['owner_terminal']['path']==str(Path(body['layout']['planning_owner']['output']).parent/'result.json') and
             record['owner_terminal']==records[0]['owner_terminal'], 'All340 slots require the one predeclared aggregate owner terminal')
    first=runtime.read_bound(records[0]['receipt'])
    _planning_owner(records[0]['owner_terminal'],runtime.reference(body['layout']['planning_owner']['config']),first,body['processing'])
    outcomes=[]
    for record,(row,b,target) in zip(records,expected):
        role=b['planning'][target];receipt=runtime.read_bound(record['receipt'])
        config_ref=runtime.reference(role['config']);config=runtime.read_bound(config_ref)
        need(config==_planning_configuration(inventory['preparation_capability'],prepared_ref,row,b,target) and
             set(receipt)=={'schema','configuration','prepared_inventory','output','complete','source_inventory','identity','completed'} and
             receipt.get('schema')==PLANNING_RESULT_SCHEMA and receipt.get('configuration')==config_ref and
             receipt.get('prepared_inventory')==prepared_ref and receipt.get('output')==role['output'] and
             receipt.get('source_inventory')==planning_sources() and receipt.get('completed') is True and
             deployment._same_live_process(receipt.get('identity'),first.get('identity')),
             'Only exact source/config-bound wrapper completion can prove a planning result')
        _declared_json(receipt['complete'],role['complete'])
        complete=runtime.read_bound(receipt['complete']);targets=complete.get('targets',[])
        need(complete.get('completed') is True and complete.get('geometry_id')==runtime.geometry.GEOMETRY_ID and
             complete.get('geometry_artifacts')==runtime.read_bound(row['prepared_complete'])['geometry_artifacts'] and
             complete.get('physics_executed') is False and complete.get('training_eligible') is False and
             isinstance(targets,list) and len(targets)==1 and targets[0].get('name')==target and
             type(targets[0].get('geometry_path_found')) is bool, 'Actual bounded planner result changed')
        found=targets[0]['geometry_path_found'];path=Path(role['fresh_plan'])
        need(found or not path.exists() and not path.is_symlink(), 'Unfound planning result carries a hidden fresh plan')
        outcomes.append(dict(subject_id=row['subject_id'],target_id=target,status='found' if found else 'bounded_planning_unfound',
            planner_result=receipt['complete'],fresh_plan=runtime.reference(path) if found else None))
    return body,cap,inventory,planning_inventory(body['layout'],outcomes)


def _native_reference_roles(row):
    need(row.get('split')=='test' and row.get('subject_id') in _identity_metadata() and row.get('arm') in ARMS,
         'Native episode outside original sealed TEST membership/arm')
    ref=runtime.pose.require_route(row['metadata']);patient=_file(row['pose_binding']['patient_receipt']);fresh=_file(row['fresh_plan'])
    def plain(value):return dict(file=value['path'],sha256=value['sha256'])
    expected=dict(task_manifest=plain(patient),patient_receipt=plain(patient),fresh_plan=plain(fresh))
    need(ref.snapshot()['sources']==expected and ref.snapshot()['available_at_s']==0. and
         row['metadata'].get('handoff_policy_id')==runtime.handoff.POLICY_ID and
         all(row['metadata'].get(k)==row[k] for k in ('subject_id','split','target_id','episode_id')),
         'Native reference leaf roles/identity/revision/availability differ before source access')
    return ref


def _native_configuration(body,prepared,attempt,*,metadata=None):
    from . import m2_pilot_config as builder
    from .fresh_expert import contact_plane_contract
    prep=runtime.read_bound(prepared['preparation_config']);patient=runtime.read_bound(prepared['patient_receipt'])
    complete=runtime.read_bound(prepared['prepared_complete']);common=body['native_common']
    if 'heldout_provenance' in common:
        _dense_current_common(common)
        need(_exact(common['physics'],prep.get('physics')) and
             _exact(common['geometry_contract'],{key:prep.get(key) for key in common['geometry_contract']}),
             'Dense heldout preparation changed the qualified physical/geometry conditions')
    target=builder._target(patient,attempt['target_id']);patient_ref=dict(file=prepared['patient_receipt']['path'],sha256=prepared['patient_receipt']['sha256'])
    fresh_ref=dict(file=attempt['fresh_plan']['path'],sha256=attempt['fresh_plan']['sha256'])
    if metadata is None:
        plan=runtime.read_bound(attempt['fresh_plan']);plane=contact_plane_contract(plan)
        need(plane is not None and plane['plane_tilt_limit_deg']==3. and plane['tilt_relaxed'] is False and
             plane['contact_plane_radius_m']==.023 and plan.get('loaded_compression_m')==.0025 and
             plan.get('free_approach_start_gap_m') in (.025,.03) and plan.get('posture_limits')==prep['posture_limits'] and
             plan.get('geometry_artifacts')==complete['geometry_artifacts'], 'Selected fresh plan changed original nonrelaxed physical geometry')
        ref=runtime.pose.build_given_reference(dict(target,manifest_ref=patient_ref),fresh_ref,patient_ref,complete['world_from_left_base'],0.)
        metadata=dict(geometry_id=runtime.geometry.GEOMETRY_ID,geometry_source_sha256=runtime.geometry.GEOMETRY_SOURCE_SHA256,
            freshly_generated=True,**complete['geometry_artifacts'],state_dim=17,horizon=20,execution_prefix=10,control_hz=20,
            joint_unit='rad',position_unit='m',target_pose_frame='openarm_left_link0',episode_id=attempt['job_id'],
            target_id=attempt['target_id'],subject_id=attempt['subject_id'],split='test',
            identity_split_sha256=runtime.shared.contract.IDENTITY_SPLIT_SHA256,patient_asset_sha256=patient['asset_sha256'],
            installed_patient_receipt_sha256=patient_ref['sha256'],source_kind='isaac_live_physics',replay=False,
            physical_qualification=False,artifact_hashes_verified=False,**builder.FLAGS)
        metadata=runtime.pose.route_metadata(metadata,ref);metadata['handoff_policy_id']=runtime.handoff.POLICY_ID
        metadata['native_retainer_audit']=deepcopy(common['fresh']['native_retainer_audit'])
    config=deepcopy(prep)
    for key in ('prepare_only',ROUTE_KEY,'patient_preview','m2_sensor_probe','m2_render_settling_probe','contact_probe'):config.pop(key,None)
    original={k:v for k,v in prep['fresh_expert'].items() if k not in ('plans','raw_multimodal_capture','collision_profile','native_retainer_audit')}
    need(all(common['fresh'].get(k)==v for k,v in original.items()) and
         set(common['fresh'])<=set(original)|{'collision_profile','native_retainer_audit'} and common['fresh'].get('collision_profile')=='full',
         'Original teacher/student physical controls differ from authenticated preparation template')
    config.update({key:common[key] for key in ('experiment','base','isaac_python')})
    config.update(targets=[target],raw_target_provenance=dict(source='installed patient surface anchor and 23 mm local plane',
        patient_receipt=patient_ref,identity=patient['identity'],asset_sha256=patient['asset_sha256'],target_id=attempt['target_id'],nominal_template_targets_used=False),
        fresh_expert=dict(deepcopy(common['fresh']),plans=[fresh_ref]),
        compact_scene_storage=_binding_from_inventory(body,prepared,attempt['owner_run_id']))
    episode=attempt['job_id'];config[ROUTE_KEY]=dict(schema=NATIVE_ROUTE_SCHEMA,capability=None,
        episode_id=episode,arm=attempt['arm'],metadata=deepcopy(metadata),seed=attempt['seed'])
    if attempt['arm']=='expert':
        config['m2_given_contact_control']=dict(schema=runtime.pose.TASK_ID,metadata=deepcopy(metadata),episode_id=episode,
            handoff_policy_id=runtime.handoff.POLICY_ID)
    else:config['m2_given_contact_deployment']=dict(schema=runtime.given.DEPLOYMENT_SCHEMA,production_ready=True,simulation_only=True,
        permit=None,episode_id=episode,metadata=deepcopy(metadata))
    return config,metadata


def _paired_attempts(body,prepared,schedule,*,stored=None):
    index={row['subject_id']:row for row in prepared['bodies']};attempts=[]
    need(stored is None or len(stored)==len(schedule['attempts']), 'Exact complete paired configuration inventory required')
    for i,attempt in enumerate(schedule['attempts']):
        row=index[attempt['subject_id']]
        config,metadata=_native_configuration(body,row,attempt,metadata=None if stored is None else stored[i]['metadata'])
        projected=configuration_projection(config);digest=runtime.digest(projected)
        value=dict(deepcopy(attempt),episode_id=attempt['job_id'],attempt_id=attempt['job_id'],metadata=metadata,
            pose_binding={key:row[key] for key in ('preparation_config','prepared_complete','patient_receipt')},
            configuration=projected,configuration_sha256=digest,runtime_config_sha256=digest)
        _native_reference_roles(value);attempts.append(value)
    return attempts


def freeze_pair_plan(prepared_inventory,planning_results,*,owner_run_id,output):
    body,cap,prepared,inventory=_planning_proofs(prepared_inventory,planning_results)
    schedule=paired_schedule(body['layout'],inventory,owner_run_id=owner_run_id)
    attempts=_paired_attempts(body,prepared,schedule)
    for row in attempts:
        need(not Path(row['config']).exists() and not Path(row['config']).is_symlink() and
             not Path(row['output']).parent.parent.exists(), 'Native paired version/config/owner already exists')
    return runtime.write_new(output,dict(schema=FROZEN_PAIR_SCHEMA,protocol=evaluation_protocol(),body_plan=cap['body_plan'],
        preparation_plan=cap['preparation_plan'],prepared_inventory=deepcopy(prepared_inventory),planning_results=deepcopy(planning_results),
        planning_inventory=inventory,schedule=schedule,attempts=attempts))


@_verification_scope
def verify_pair_plan(ref,*,payloads=True):
    value=runtime.read_bound(ref)
    need(set(value)=={'schema','protocol','body_plan','preparation_plan','prepared_inventory','planning_results','planning_inventory','schedule','attempts'} and
         value['schema']==FROZEN_PAIR_SCHEMA and value['protocol']==evaluation_protocol(), 'Exact immutable paired heldout plan required')
    body,cap,prepared,inventory=_planning_proofs(value['prepared_inventory'],value['planning_results'])
    need(value['body_plan']==cap['body_plan'] and value['preparation_plan']==cap['preparation_plan'] and value['planning_inventory']==inventory,
         'Original complete planning inventory/body/preparation chain changed')
    owner=value['schedule']['owner_run_id']
    # Even zero available bodies retain the predeclared native owner identity.
    schedule=paired_schedule(body['layout'],inventory,owner_run_id=owner)
    need(value['schedule']==schedule, 'Original first-feasible/fixed-34 paired schedule changed')
    expected=_paired_attempts(body,prepared,schedule,stored=None if payloads else value['attempts'])
    need(value['attempts']==expected, 'Frozen paired config/full reference/arm identity changed')
    if payloads:verify_prepared_inventory(value['prepared_inventory'])
    return value


def _child_pair_plan(ref,body,preparation_plan):
    """The parent proves planning; inference rebinds the sealed metadata only.

    No planner receipt/COMPLETE/fresh-plan trajectory leaf is opened here. The
    child's only model inputs remain the actual two-frame request observation.
    """
    value=runtime.read_bound(ref)
    need(set(value)=={'schema','protocol','body_plan','preparation_plan','prepared_inventory','planning_results','planning_inventory','schedule','attempts'} and
         value['schema']==FROZEN_PAIR_SCHEMA and value['protocol']==evaluation_protocol() and
         value['preparation_plan']==preparation_plan, 'Exact frozen paired inference metadata required')
    prepared=verify_prepared_inventory(value['prepared_inventory'],payloads=False)
    prep=runtime.read_bound(prepared['preparation_capability'])
    need(value['body_plan']==prep['body_plan'] and value['preparation_plan']==prep['preparation_plan'], 'Sealed child source roles changed')
    inventory=planning_inventory(body['layout'],value['planning_inventory']['outcomes'])
    need(value['planning_inventory']==inventory, 'Child cannot change all340 planning declarations or first-feasible selection')
    records=value['planning_results'];expected=[(row,target) for row in body['layout']['bodies'] for target in row['target_priority']]
    need(isinstance(records,list) and len(records)==340, 'All340 sealed planning receipt roles required')
    terminal=str(Path(body['layout']['planning_owner']['output']).parent/'result.json')
    for record,(row,target) in zip(records,expected):
        need(set(record)=={'subject_id','target_id','receipt','owner_terminal'} and
             (record['subject_id'],record['target_id'])==(row['subject_id'],target), 'Sealed child planning order/identity changed')
        _declared_json(record['receipt'],str(Path(row['planning'][target]['output'])/'heldout_planning.json'))
        runtime._reference_shape(record['owner_terminal'])
        need(record['owner_terminal']['path']==terminal and record['owner_terminal']==records[0]['owner_terminal'],
             'Child planning owner role changed')
    schedule=paired_schedule(body['layout'],inventory,owner_run_id=value['schedule']['owner_run_id'])
    need(value['schedule']==schedule and value['attempts']==_paired_attempts(body,prepared,schedule,stored=value['attempts']),
         'Sealed native observation/configuration/arm/reference metadata changed')
    return value


def issue_native_capability(pair_plan,*,output):
    pair=verify_pair_plan(pair_plan);value=_capability(pair['body_plan'],pair['preparation_plan'],'native')
    value.update(pair_plan=deepcopy(pair_plan),attempts=deepcopy(pair['attempts']),prepared_inventory=pair['prepared_inventory'])
    for row in pair['attempts']:
        need(not Path(row['config']).exists() and not Path(row['config']).is_symlink() and
             not Path(row['output']).parent.parent.exists(), 'Native paired version already launched/configured')
    cap=runtime.write_new(output,value);configs=[]
    for row in pair['attempts']:
        config=deepcopy(row['configuration']);config[ROUTE_KEY]['capability']=cap
        if row['arm']=='act':config['m2_given_contact_deployment']['permit']=cap
        configs.append(dict(subject_id=row['subject_id'],arm=row['arm'],config=runtime.write_new(row['config'],config),output=row['output']))
    return dict(capability=cap,configurations=configs)


def freeze_planning_configurations(prepared_inventory):
    inventory=verify_prepared_inventory(prepared_inventory);cap_ref=inventory['preparation_capability']
    cap=verify_capability(cap_ref,stage='preparation');body=verify_body_plan(cap['body_plan']);configs=[]
    declarations=[];owner=body['layout']['planning_owner']
    need(not Path(owner['output']).exists() and not Path(owner['output']).is_symlink() and
         not Path(owner['config']).exists() and not Path(owner['config']).is_symlink(), 'Frozen planning owner/config already exists')
    for row,body_row in zip(inventory['bodies'],body['layout']['bodies']):
        for target in body_row['target_priority']:
            role=body_row['planning'][target]
            need(not Path(role['config']).exists() and not Path(role['config']).is_symlink() and
                 not Path(role['output']).exists() and not Path(role['output']).is_symlink(), 'Planning version/config/output already exists')
            config=_planning_configuration(cap_ref,prepared_inventory,row,body_row,target)
            declarations.append(dict(subject_id=row['subject_id'],target_id=target,
                config=_future_configuration(role['config'],config),configuration=config,output=role['output']))
    aggregate=_aggregate_configuration(cap_ref,prepared_inventory,inventory,body,declarations)
    return runtime.write_new(owner['config'],aggregate)


def _configuration_bytes(value):
    return (json.dumps(value,allow_nan=False,sort_keys=True,indent=2)+'\n').encode()


def _future_configuration(path,value):
    content=_configuration_bytes(value)
    return dict(path=path,bytes=len(content),sha256=hashlib.sha256(content).hexdigest())


def _aggregate_configuration(cap_ref,inventory_ref,inventory,body,slots):
    owner=body['layout']['planning_owner'];template=body['templates'][body['layout']['bodies'][0]['pose_template']]
    value={key:deepcopy(template[key]) for key in ('isaac_python','asset_root','ld_library_path')}
    value.update(experiment='m2_geometry_plan',base=owner['base'],time_limit_seconds=340*2*180+600)
    value[AGGREGATE_KEY]=dict(schema=AGGREGATE_SCHEMA,stage='planning',capability=deepcopy(cap_ref),prepared_inventory=deepcopy(inventory_ref),
        owner=deepcopy(owner),source_inventory=planning_sources(),slots=slots,
        slot_configurations='generated_future_materialization_of_frozen_bytes')
    return value


@_verification_scope
def planning_aggregate_preflight(config,*,output):
    route=config.get(AGGREGATE_KEY)
    need(isinstance(route,dict) and route.get('schema')==AGGREGATE_SCHEMA, 'Exact sealed aggregate planning schema required')
    if route.get('stage')=='preparation':return _preparation_aggregate_preflight(config,output=output)
    need(route.get('stage')=='planning', 'Unknown/cross-stage aggregate refused')
    cap=verify_capability(route.get('capability'),stage='preparation')
    inventory=verify_prepared_inventory(route['prepared_inventory'],payloads=False)
    need(inventory['preparation_capability']==route['capability'], 'Aggregate planning capability/inventory differs')
    body=verify_body_plan(cap['body_plan']);owner=body['layout']['planning_owner'];slots=[]
    need(str(Path(output))==owner['review'], 'Original predeclared owner/output/review subtree required')
    for row,b in zip(inventory['bodies'],body['layout']['bodies']):
        for target in b['target_priority']:
            role=b['planning'][target];value=_planning_configuration(route['capability'],route['prepared_inventory'],row,b,target)
            slots.append(dict(subject_id=row['subject_id'],target_id=target,config=_future_configuration(role['config'],value),
                configuration=value,output=role['output']))
    need(config==_aggregate_configuration(route['capability'],route['prepared_inventory'],inventory,body,slots),
         'Frozen aggregate/all340 config bytes/order/roles/source/deadline changed before payload access')
    return dict(stage='planning',layout=body['layout'],inventory=inventory,capability=cap,slots=slots)


def _preparation_slots(cap_ref,cap):
    slots=[]
    for row in cap['configurations']:
        ref=runtime.reference(row['config']);config=runtime.read_bound(ref)
        need(config.get(ROUTE_KEY,{}).get('capability')==cap_ref and configuration_projection(config)==row['configuration'],
             'Exact final preparation capability/config bytes changed')
        slots.append(dict(subject_id=row['subject_id'],config=ref,output=row['output']))
    return slots


def _preparation_aggregate_configuration(cap_ref,body,slots):
    owner=body['layout']['preparation_owner'];template=body['templates'][body['layout']['bodies'][0]['pose_template']]
    value={key:deepcopy(template[key]) for key in ('isaac_python','asset_root','ld_library_path')}
    value.update(experiment='m2_geometry_plan',base=owner['base'],time_limit_seconds=sum(
        body['templates'][row['pose_template']]['time_limit_seconds'] for row in body['layout']['bodies'])+60)
    value[AGGREGATE_KEY]=dict(schema=AGGREGATE_SCHEMA,stage='preparation',capability=deepcopy(cap_ref),
        prepared_inventory=None,owner=deepcopy(owner),source_inventory=planning_sources(),slots=deepcopy(slots))
    return value


def _preparation_aggregate_preflight(config,*,output):
    route=config[AGGREGATE_KEY];cap=verify_capability(route.get('capability'),stage='preparation');body=verify_body_plan(cap['body_plan'])
    slots=_preparation_slots(route['capability'],cap)
    need(str(Path(output))==body['layout']['preparation_owner']['review'] and
         config==_preparation_aggregate_configuration(route['capability'],body,slots),
         'All34 CPU preparation configs/order/output/source/stage must reconstruct before the first payload')
    return dict(stage='preparation',layout=body['layout'],body=body,capability=cap,slots=slots)


def planning_preflight(config,*,output):
    route=config.get(PLANNING_KEY)
    need(isinstance(route,dict) and route.get('schema')==PLANNING_SCHEMA, 'Only the fixed heldout planning wrapper is authorized')
    cap=verify_capability(route.get('capability'),stage='preparation')
    inventory=verify_prepared_inventory(route['prepared_inventory'],payloads=False)
    need(inventory['preparation_capability']==route['capability'], 'Planning preparation capability/inventory differs')
    body=verify_body_plan(cap['body_plan']);identity=route.get('subject_id');target=route.get('target_id')
    rows=[(row,b) for row,b in zip(inventory['bodies'],body['layout']['bodies']) if row['subject_id']==identity]
    need(len(rows)==1 and target in TARGETS, 'Planning subject/target outside immutable all-340 allocation')
    row,b=rows[0]
    expected=_planning_configuration(route['capability'],route['prepared_inventory'],row,b,target)
    need(config==expected and str(Path(output))==b['planning'][target]['output'],
         'Planning config/algorithm/budget/source/target/output differs before payload access')
    return dict(subject_id=identity,target_id=target,prepared=row,role=b['planning'][target],capability=cap)


def run_planning(config_ref,output):
    """The sole authorized heldout CPU delegate, with no altered algorithm."""
    from . import geometry_plan,process_guard
    config=runtime.read_bound(config_ref);binding=planning_preflight(config,output=output)
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
         'Original owned Linux CPU planning stage required')
    path=Path(output);need(not path.exists() and not path.is_symlink(), 'Planning output must be the absent declared path')
    prepared=binding['prepared'];expected=prepared['inventory'];sources=planning_sources()
    need(_tree(prepared['output'])==expected, 'Selected prepared inputs changed before the original planner')
    geometry=dict(path=config['geometry_source'],sha256=config['geometry_source_sha256']);runtime.bound_path(geometry)
    identity=process_guard.process_identity(os.getpid());need(identity is not None, 'Actual CPU worker process identity missing')
    before=sys.argv
    try:
        sys.argv=[geometry_plan.__name__,'--config',config_ref['path'],'--output',str(path)]
        geometry_plan.main()
    finally:sys.argv=before
    runtime.bound_path(config_ref);runtime.bound_path(geometry)
    need(planning_sources()==sources and _tree(prepared['output'])==expected,
         'Planner/source/prepared bytes changed during the original bounded algorithm')
    return runtime.write_new(path/'heldout_planning.json',dict(schema=PLANNING_RESULT_SCHEMA,configuration=deepcopy(config_ref),
        prepared_inventory=config[PLANNING_KEY]['prepared_inventory'],output=str(path),complete=runtime.reference(path/'COMPLETE.json'),
        source_inventory=sources,identity=identity,completed=True))


def _materialized_configurations(slots,review):
    root=Path(review)/'configs';actual=_tree(root)
    expected={str(Path(row['config']['path']).relative_to(root)):row['config'] for row in slots}
    directories=sorted({str(Path(name).parent) for name in expected})
    need(actual['files']==expected and actual['directories']==directories,
         'All340 materialized config bytes and exact closed path set must match the frozen aggregate')


def _run_preparation_aggregate(config_ref,config,binding,output):
    from . import geometry_qualification,process_guard
    import time
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
         'Original owned CPU-only preparation stage required')
    path=runtime.dataset._path(str(output),Path.cwd());path.mkdir(exist_ok=False)
    sources=planning_sources();identity=process_guard.process_identity(os.getpid());started=time.monotonic()
    need(identity is not None, 'Original CPU preparation child identity missing')
    outcomes=[dict(subject_id=row['subject_id'],config=deepcopy(row['config']),output=row['output'],status='planned_unlaunched')
        for row in binding['slots']]
    receipts=[];failure=None;current=None
    try:
        for slot,current in zip(binding['slots'],outcomes):
            settings=runtime.read_bound(slot['config']);geometry=dict(path=settings['geometry_source'],sha256=settings['geometry_source_sha256'])
            body=binding['body']['patients'][slot['subject_id']]['body']
            runtime.bound_path(geometry);runtime.bound_path(body)
            runtime.write_new(path/('preparation-'+slot['subject_id']+'.start.json'),dict(subject_id=slot['subject_id'],
                config=slot['config'],output=slot['output'],identity=identity,source_inventory=sources))
            current['status']='delegate_started';before=sys.argv
            command=[geometry_qualification.__name__,'--cpu-prepare-only','--config',slot['config']['path'],'--output',slot['output']]
            try:
                sys.argv=command;geometry_qualification.main()
            finally:sys.argv=before
            runtime.bound_path(slot['config']);runtime.bound_path(geometry);runtime.bound_path(body)
            need(planning_sources()==sources, 'CPU preparation sources changed during original authoring')
            complete_ref=runtime.reference(Path(slot['output'])/'COMPLETE.json');complete=runtime.read_bound(complete_ref)
            need(complete.get('completed') is True and complete.get('preparation_backend')=='cpu_usd_authoring_no_simulation_app' and
                 all(complete.get(key) is False for key in ('physics_executed','physical_validation','training_eligible')),
                 'CPU preparation cannot contain SimulationApp/physics or synthetic acceptance')
            receipt=runtime.write_new(Path(slot['output'])/'heldout_preparation.json',dict(schema=PREPARATION_RESULT_SCHEMA,
                configuration=slot['config'],capability=config[AGGREGATE_KEY]['capability'],output=slot['output'],complete=complete_ref,
                source_inventory=sources,identity=identity,command=command,completed=True))
            current.update(status='completed',receipt=receipt);receipts.append(dict(subject_id=slot['subject_id'],receipt=receipt))
        runtime.bound_path(config_ref);planning_aggregate_preflight(config,output=output)
    except BaseException as error:
        failure=error
        if current is not None and current['status']=='delegate_started':current.update(status='failed',error=repr(error))
        raise
    finally:
        try:runtime.write_new(path/'preparation_execution.json',dict(stage='preparation',configuration=config_ref,
            slots=outcomes,completed=failure is None,error=None if failure is None else repr(failure),physics_executed=False))
        except BaseException:
            if failure is None:raise
    return runtime.write_new(path/'COMPLETE.json',dict(schema=AGGREGATE_SCHEMA,stage='preparation',configuration=config_ref,
        prepared_inventory=None,source_inventory=sources,identity=identity,slots=receipts,completed=True,
        physics_executed=False,training_eligible=False,wall_s=time.monotonic()-started))


def run_planning_aggregate(config_ref,output):
    from . import process_guard
    import time
    config=runtime.read_bound(config_ref);binding=planning_aggregate_preflight(config,output=output)
    if binding['stage']=='preparation':return _run_preparation_aggregate(config_ref,config,binding,output)
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
         'Original owned Linux CPU aggregate stage required')
    path=runtime.dataset._path(str(output),Path.cwd());path.mkdir(exist_ok=False)
    slots=binding['slots'];sources=planning_sources();identity=process_guard.process_identity(os.getpid());started=time.monotonic()
    need(identity is not None, 'Actual original CPU planning child identity required')
    for row in binding['inventory']['bodies']:
        need(_tree(row['output'])==row['inventory'], 'Prepared payloads changed before aggregate planning')
    # Future refs were frozen from these exact bytes, not from newly read inputs.
    for row in slots:
        target=runtime.dataset._path(row['config']['path'],Path.cwd());target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(_configuration_bytes(row['configuration']));stream.flush();os.fsync(stream.fileno())
    _materialized_configurations(slots,path);receipts=[]
    for row in slots:
        runtime.bound_path(row['config'])
        receipt=run_planning(row['config'],row['output'])
        receipts.append(dict(subject_id=row['subject_id'],target_id=row['target_id'],receipt=receipt))
    _materialized_configurations(slots,path);runtime.bound_path(config_ref)
    need(planning_sources()==sources, 'Aggregate planning sources changed')
    for row in binding['inventory']['bodies']:
        need(_tree(row['output'])==row['inventory'], 'Prepared payloads changed during aggregate planning')
    return runtime.write_new(path/'COMPLETE.json',dict(schema=AGGREGATE_SCHEMA,stage='planning',configuration=deepcopy(config_ref),
        prepared_inventory=config[AGGREGATE_KEY]['prepared_inventory'],source_inventory=sources,identity=identity,
        slots=receipts,completed=True,physics_executed=False,training_eligible=False,wall_s=time.monotonic()-started))


@_verification_scope
def preflight(config,*,output=None):
    """Authenticate exact stage/config/identity before any TEST leaf payload."""
    route=config.get(ROUTE_KEY)
    need(isinstance(route,dict) and route.get('schema') in (PREP_ROUTE_SCHEMA,NATIVE_ROUTE_SCHEMA),
         'Explicit sealed heldout stage required')
    stage='preparation' if route['schema']==PREP_ROUTE_SCHEMA else 'native'
    # First authenticate the immutable metadata/capability chain. The caller's
    # exact config, output and identity must match before full payload rehash.
    cap=verify_capability(route.get('capability'),stage=stage,child_only=stage=='native')
    rows=cap['configurations'] if stage=='preparation' else cap['attempts']
    identity=route.get('subject_id') if stage=='preparation' else route.get('episode_id')
    key='subject_id' if stage=='preparation' else 'episode_id'
    matches=[row for row in rows if row[key]==identity]
    need(len(matches)==1, 'Identity outside exact sealed capability')
    row=matches[0]
    need(configuration_projection(config)==row['configuration'] and
         runtime.digest(row['configuration'])==row['configuration_sha256'],
         'Frozen heldout configuration projection changed')
    if output is not None:need(str(Path(output))==row['output'], 'Exact predeclared stage output required')
    if stage=='native':
        need(route.get('arm')==row['arm'] and route.get('metadata')==row['metadata'],
             'Exact arm/full reference metadata required')
        if row['arm']=='act':need(config['m2_given_contact_deployment']['permit']==route['capability'],
             'ACT child permit differs from native capability')
        _native_reference_roles(row)
        need(verify_capability(route['capability'],stage='native')==cap,
             'Native capability changed after exact metadata authorization')
    return dict(capability=cap,reference=deepcopy(route['capability']),stage=stage,row=row)


def authorize_patient(spec,config):
    binding=preflight(config)
    need(spec==config.get('patient_asset') and spec.get('expected_split')=='test',
         'Patient specification differs from the exact stage capability')
    row=binding['row'];body=verify_body_plan(binding['capability']['body_plan'])
    need(spec==body['patient_specs'][row['subject_id']], 'Patient source role differs from frozen original body plan')
    return row


def require_permitted_request(metadata,dep,permit,row):
    """Protocol membership only; the bridge/child authenticate the full cap."""
    need(permit.get('schema')==CAPABILITY_SCHEMA and permit.get('stage')=='native' and permit.get('scope')=='sealed_test_once_only' and
         permit.get('production_ready') is True and permit.get('simulation_only') is True and row.get('arm')=='act' and
         metadata==row.get('metadata')==dep.get('metadata') and row.get('split')==metadata.get('split')=='test' and
         metadata.get('subject_id')==row.get('subject_id') in _identity_metadata() and
         runtime.read_bound(dep['permit'])==permit, 'Request outside exact immutable admitted TEST ACT episode/reference')
    _native_reference_roles(row)


SCENE_AUTH_SCHEMA='m2_given_contact_sealed_test_scene_authorization_v1'


def scene_authorization(config):
    binding=preflight(config)
    need(binding['stage']=='native', 'Only native capability authorizes stored scene use')
    return dict(schema=SCENE_AUTH_SCHEMA,capability=binding['reference'],episode_id=binding['row']['episode_id'],
        configuration_sha256=binding['row']['configuration_sha256'])


def verify_scene_authorization(binding,authorization):
    need(isinstance(authorization,dict) and set(authorization)=={'schema','capability','episode_id','configuration_sha256'} and
         authorization['schema']==SCENE_AUTH_SCHEMA, 'Exact schema-tagged stored native scene authorization required')
    cap=verify_capability(authorization['capability'],stage='native',child_only=True)
    rows=[row for row in cap['attempts'] if row['episode_id']==authorization['episode_id']]
    need(len(rows)==1 and rows[0]['configuration_sha256']==authorization['configuration_sha256'] and
         rows[0]['configuration']['compact_scene_storage']==binding, 'Stored scene differs from exact native capability/config projection')
    need(verify_capability(authorization['capability'],stage='native')==cap,
         'Stored scene capability changed after exact metadata authorization')
    return rows[0]


def _binding_from_inventory(body,row,batch_id):
    from . import scene_store
    common=body['native_common'];sources=deepcopy(common['shared_sources']);files=row['inventory']['files']
    if 'heldout_provenance' in common:
        _dense_current_common(common)
        root=Path(row['prepared_complete']['path']).parent
        need(set(scene_store.FILES.values())<=set(files) and all(
            files[name]['path']==str(root/name) for name in scene_store.FILES.values()),
            'Every dense TEST scene role must come from its exact prepared inventory')
    else:
        need(common['dependency_policy']!=scene_store.SDK_LAYER_CURRENT_RAW_POLICY,
             'Current RAW heldout requires an explicit authenticated dense SDK projection')
    for name,filename in {**scene_store.FILES,**scene_store.OPTIONAL}.items():
        if filename in files:sources[name]=dict(file=files[filename]['path'],sha256=files[filename]['sha256'],bytes=files[filename]['bytes'])
    for key in ('preparation_config','prepared_complete'):
        ref=row[key];sources[key]=dict(file=ref['path'],sha256=ref['sha256'],bytes=ref['bytes'])
    binding=dict(schema=common['schema'],dependency_policy=common['dependency_policy'],store_root=common['store_root'],
        batch_id=batch_id,sources=sources)
    scene_store._binding_shape(binding);return binding


def verify_prepared_binding(binding,context):
    inventory=verify_prepared_inventory(context['prepared_inventory'],payloads=False)
    cap=verify_capability(context['capability'],stage='preparation');body=verify_body_plan(cap['body_plan'])
    need(inventory['preparation_capability']==context['capability'], 'Prepared scene capability changed')
    rows=[row for row in inventory['bodies'] if row['subject_id']==context['subject_id']]
    need(len(rows)==1 and binding==_binding_from_inventory(body,rows[0],binding.get('batch_id')),
         'Prepared scene/SDK/native source role differs before payload access')
    return rows[0]


@_verification_scope
def verify_capability(ref,*,stage,child_only=False):
    value=runtime.read_bound(ref)
    fields={'schema','stage','body_plan','preparation_plan','protocol','task_contract','production_ready','simulation_only','scope',
        'physical_acceptance','real_hardware_validated','closed_loop_evaluated','admission','model','processing','source_inventory'}
    need(set(value)==fields|({'configurations'} if stage=='preparation' else {'pair_plan','attempts','prepared_inventory'}),
         'Unknown/excess heldout capability fields refused')
    expected=PREP_CAPABILITY_SCHEMA if stage=='preparation' else CAPABILITY_SCHEMA
    need(stage in ('preparation','native') and value.get('schema')==expected and value.get('stage')==stage,
         'Cross-stage, unissued or unknown heldout capability refused')
    need(value.get('production_ready') is True and value.get('simulation_only') is True and
         value.get('task_contract')==runtime.task_contract() and value.get('protocol')==deployment.protocol() and
         value.get('scope')=='sealed_test_once_only' and value.get('physical_acceptance') is False and
         value.get('real_hardware_validated') is False and value.get('closed_loop_evaluated') is False,
         'Only the exact once-only formal simulation capability is recognized')
    body=verify_body_plan(value['body_plan'])
    need(value['model']==body['model'] and value['admission']==body['admission'] and
         value['processing']==body['processing'] and value['source_inventory']==body['source_inventory'],
         'Heldout capability changed admitted model/source/prerequisites')
    prep=verify_preparation_plan(value['preparation_plan'])
    need(prep['body_plan']==value['body_plan'], 'Preparation and body plan differ')
    if stage=='preparation':
        need(value.get('configurations')==prep['configurations'], 'Exact 34 frozen preparation configs required')
    else:
        pair=(_child_pair_plan(value['pair_plan'],body,value['preparation_plan']) if child_only else verify_pair_plan(value['pair_plan']))
        need(pair['body_plan']==value['body_plan'] and pair['preparation_plan']==value['preparation_plan'] and
             value.get('attempts')==pair['attempts'] and value.get('prepared_inventory')==pair['prepared_inventory'],
             'Exact complete paired native plan/prepared source inventory required')
    return value


def evaluation_protocol():
    return dict(schema='m2_given_contact_once_only_paired_protocol_v1',task_contract=runtime.task_contract(),
        bodies=34,known_pose_counts=[17,17],targets=list(TARGETS),planning_attempts=340,
        selection='first bounded-planner-found target in immutable per-body priority',
        native_arms_max=68,executed_pairs_milestone=30,arm_order_counts=[17,17],
        retries=False,replacement=False,seed_search=False,expert_l2_substitution=False,
        independent_native_resets=True,video_hz=10,video_size=[640,360],
        review_video_selection='first two original body IDs, expert then ACT, irrespective of outcome',
        confidence=.95,interval='two-sided exact Clopper-Pearson',
        scope='34 frozen bodies; two known poses; conditional bounded first-feasible prescribed full-pose control')


def _rank(seed,domain,value):
    return hashlib.sha256(('%s:%s:%s'%(seed,domain,value)).encode()).digest()


def allocate_bodies(identities,*,seed,train_feasibility):
    """Pure metadata choice; this function reads no file or asset payload."""
    need(isinstance(identities,list) and len(identities)==34 and len(set(identities))==34 and
         all(isinstance(i,str) and re.fullmatch('p[0-9]{6}',i) for i in identities),
         'Exactly 34 distinct frozen TEST identity names required')
    need(type(seed) is int and 0<=seed<2**32 and isinstance(train_feasibility,dict) and
         set(train_feasibility)==set(TARGETS) and all(type(n) is int and n>=0 for n in train_feasibility.values()),
         'Predeclared uint32 seed and complete nonnegative TRAIN planning counts required')
    poses=sorted(deployment.POSE_TEMPLATES)
    pose_rank=sorted(identities,key=lambda i:_rank(seed,'pose',i))
    order_rank=sorted(identities,key=lambda i:_rank(seed,'arm-order',i))
    target_rank=sorted(identities,key=lambda i:_rank(seed,'target-order',i))
    targets=sorted(TARGETS,key=lambda t:(-train_feasibility[t],t))
    seeds={i:int.from_bytes(_rank(seed,'native-seed',i)[:4],'big') for i in identities}
    need(len(set(seeds.values()))==34, 'Native seed collision; freeze a new metadata version before access')
    rows=[]
    for identity in sorted(identities):
        offset=target_rank.index(identity)%10
        rows.append(dict(subject_id=identity,split='test',pose_template=poses[pose_rank.index(identity)%2],
            seed=seeds[identity],target_priority=targets[offset:]+targets[:offset],
            arm_order=list(ARMS if order_rank.index(identity)%2==0 else reversed(ARMS))))
    return rows


def select_first_feasible(priority,outcomes):
    need(isinstance(priority,list) and len(priority)==10 and set(priority)==set(TARGETS),
         'Complete immutable all-ten target priority required')
    need(isinstance(outcomes,list) and len(outcomes)==10 and all(isinstance(r,dict) and
         set(r)=={'target_id','status'} and r['status'] in ('found','bounded_planning_unfound') for r in outcomes) and
         {r['target_id'] for r in outcomes}==set(TARGETS),
         'Exactly ten distinct bounded planning outcomes required; native outcomes are forbidden')
    found={r['target_id'] for r in outcomes if r['status']=='found'}
    return next((target for target in priority if target in found),None)


def _body_rows(rows):
    need(isinstance(rows,list) and len(rows)==34 and all(isinstance(row,dict) and set(row)=={
        'subject_id','split','pose_template','seed','target_priority','arm_order'} for row in rows),
        'Complete 34-body metadata allocation required')
    ids=[row['subject_id'] for row in rows]
    need(ids==sorted(ids) and len(set(ids))==34 and all(isinstance(i,str) and
         re.fullmatch('p[0-9]{6}',i) for i in ids), 'Original sorted distinct body identities required')
    need(len({row['seed'] for row in rows})==34 and all(type(row['seed']) is int and
         0<=row['seed']<2**32 and row['split']=='test' for row in rows), 'Fixed TEST metadata and distinct uint32 seeds required')
    for pose in deployment.POSE_TEMPLATES:
        need(sum(row['pose_template']==pose for row in rows)==17, 'Exactly 17 bodies per original known pose required')
    for order in (list(ARMS),list(reversed(ARMS))):
        need(sum(row['arm_order']==order for row in rows)==17, 'Exactly 17 original bodies per arm order required')
    for row in rows:
        select_first_feasible(row['target_priority'],[dict(target_id=t,status='bounded_planning_unfound') for t in TARGETS])
    return rows


def preparation_layout(rows,*,root,planning_base=None):
    """Declare future names only. This is never a TEST-access capability.

    Actual prerequisite authentication, preparation configuration freezing and
    detached capability issuance are separate gates. No body, scene, collision
    payload or future output is read, opened or hashed here.
    """
    _body_rows(rows);path=Path(root)
    need(path.is_absolute() and '..' not in path.parts and str(path)==root,
         'One canonical absolute preparation root required')
    base=str(path.parent) if planning_base is None else planning_base;owner_id=path.name+'-cpu-planning'
    need(isinstance(base,str) and Path(base).is_absolute() and '..' not in Path(base).parts and str(Path(base))==base and
         re.fullmatch('[a-z0-9][a-z0-9_-]{0,127}',owner_id), 'Canonical fixed planning base/run ID required')
    output=Path(base)/'runs/ausc_policy'/owner_id/'output';review=output/'review'
    owner=dict(base=base,run_id=owner_id,config=str(path/'planning-owner.json'),output=str(output),review=str(review))
    prep_id=path.name+'-cpu-preparation';prep_output=Path(base)/'runs/ausc_policy'/prep_id/'output'
    prep_owner=dict(base=base,run_id=prep_id,config=str(path/'preparation-owner.json'),output=str(prep_output),review=str(prep_output/'review'))
    bodies=[]
    for row in rows:
        identity=row['subject_id']
        bodies.append(dict(deepcopy(row),preparation_config=str(path/'configs'/(identity+'.json')),
            prepared_output=str(Path(prep_owner['review'])/'prepared'/identity),planning={target:dict(
                config=str(review/'configs'/identity/(target+'.json')),
                output=str(review/'slots'/identity/target),
                complete=str(review/'slots'/identity/target/'COMPLETE.json'),
                fresh_plan=str(review/'slots'/identity/target/(target+'_fresh_plan.json')))
                for target in row['target_priority']}))
    return dict(schema=PREPARATION_SCHEMA,root=root,protocol=evaluation_protocol(),preparation_owner=prep_owner,planning_owner=owner,bodies=bodies,
        metadata_only=True,test_payload_access_authorized=False,native_access_authorized=False)


def _layout(value):
    need(isinstance(value,dict) and set(value)=={'schema','root','protocol','preparation_owner','planning_owner','bodies','metadata_only',
        'test_payload_access_authorized','native_access_authorized'}, 'Exact metadata-only preparation declarations required')
    rows=[{key:row[key] for key in ('subject_id','split','pose_template','seed','target_priority','arm_order')}
          for row in value['bodies']]
    need(value==preparation_layout(rows,root=value['root'],planning_base=value['planning_owner']['base']), 'Preparation paths/protocol/body allocation changed')
    return rows


def require_future_outputs_absent(layout):
    """Only inspect future names; never hash or decode TEST assets/outputs."""
    _layout(layout)
    paths=[Path(row['prepared_output']) for row in layout['bodies']]
    paths += [Path(layout['planning_owner'][key]) for key in ('config','output')]
    paths += [Path(layout['preparation_owner'][key]) for key in ('config','output')]
    paths += [Path(plan['output']) for row in layout['bodies'] for plan in row['planning'].values()]
    need(all(not path.exists() and not path.is_symlink() for path in paths),
         'A predeclared future preparation/planning output already exists')


def _declared_json(ref,path):
    need(isinstance(ref,dict) and set(ref)=={'path','bytes','sha256'} and ref['path']==path and
         type(ref['bytes']) is int and ref['bytes']>0 and isinstance(ref['sha256'],str) and
         re.fullmatch('[0-9a-f]{64}',ref['sha256']), 'Exact predeclared JSON metadata role required')


def planning_inventory(layout,outcomes):
    """Check the complete declared result inventory, without reading its files.

    These syntactic checks cannot verify a planner result or grant TEST/native
    access. The later sealed-capability verifier must authenticate all records.
    """
    _layout(layout)
    expected=[(row,target) for row in layout['bodies'] for target in row['target_priority']]
    need(isinstance(outcomes,list) and len(outcomes)==340, 'Exactly all 340 fixed planning slots required')
    for outcome,(row,target) in zip(outcomes,expected):
        need(isinstance(outcome,dict) and set(outcome)=={'subject_id','target_id','status','planner_result','fresh_plan'} and
             (outcome['subject_id'],outcome['target_id'])==(row['subject_id'],target) and
             outcome['status'] in ('found','bounded_planning_unfound'), 'Fixed planning identity/order/status required; native outcomes forbidden')
        role=row['planning'][target]
        _declared_json(outcome['planner_result'],role['complete'])
        if outcome['status']=='found':_declared_json(outcome['fresh_plan'],role['fresh_plan'])
        else:need(outcome['fresh_plan'] is None, 'Planning-unfound cannot carry a selected fresh plan')
    selected=[]
    for index,row in enumerate(layout['bodies']):
        values=outcomes[index*10:(index+1)*10]
        target=select_first_feasible(row['target_priority'],[{key:r[key] for key in ('target_id','status')} for r in values])
        fresh=next((r['fresh_plan'] for r in values if r['target_id']==target),None)
        selected.append(dict(subject_id=row['subject_id'],target_id=target,fresh_plan=deepcopy(fresh)))
    return dict(schema='m2_given_contact_declared_planning_inventory_v1',preparation_sha256=runtime.digest(layout),
        outcomes=deepcopy(outcomes),selected=selected,body_denominator=34,planning_denominator=340,
        available_bodies=sum(row['target_id'] is not None for row in selected),
        payloads_verified=False,native_access_authorized=False)


def paired_schedule(layout,inventory,*,owner_run_id):
    """Declare at most 68 ordered arms; never authorize or launch them."""
    _layout(layout)
    need(inventory==planning_inventory(layout,inventory['outcomes']), 'Original full planning inventory/first-feasible selection changed')
    need(isinstance(owner_run_id,str) and re.fullmatch('[a-z0-9][a-z0-9_-]{0,127}',owner_run_id),
         'One explicit finite native owner identity required')
    attempts=[];pairs=[]
    for row,selected in zip(layout['bodies'],inventory['selected']):
        pair=dict(subject_id=row['subject_id'],split='test',pose_template=row['pose_template'],seed=row['seed'],
            arm_order=deepcopy(row['arm_order']),selected_target=selected['target_id'],attempts=[])
        if selected['target_id'] is not None:
            for arm in row['arm_order']:
                job=row['subject_id']+'-'+arm
                attempt=dict(subject_id=row['subject_id'],split='test',arm=arm,seed=row['seed'],
                    pose_template=row['pose_template'],target_id=selected['target_id'],
                    fresh_plan=deepcopy(selected['fresh_plan']),owner_run_id=owner_run_id,job_id=job,
                    config=str(Path(layout['root'])/'native-configs'/(job+'.json')),
                    output='/workspace/wam/runs/ausc_policy/'+owner_run_id+'/output/'+job)
                pair['attempts'].append(attempt);attempts.append(deepcopy(attempt))
        pairs.append(pair)
    return dict(schema=PAIR_PLAN_SCHEMA,protocol=evaluation_protocol(),preparation_sha256=runtime.digest(layout),
        planning_sha256=runtime.digest(inventory),body_denominator=34,owner_run_id=owner_run_id,pairs=pairs,attempts=attempts,
        payloads_verified=False,native_access_authorized=False)


def binary_rate(successes,denominator):
    need(type(successes) is int and type(denominator) is int and 0<=successes<=denominator,
         'Explicit valid binary numerator and denominator required')
    interval=None
    if denominator:
        from scipy.stats import beta
        interval=[0. if not successes else float(beta.ppf(.025,successes,denominator-successes+1)),
                  1. if successes==denominator else float(beta.ppf(.975,successes+1,denominator-successes))]
    return dict(successes=successes,denominator=denominator,rate=successes/denominator if denominator else None,
                confidence=.95,interval=interval)


def select_review_videos(arms,*,body_ids):
    """Fixed first two original IDs; missing/failed video remains visible."""
    need(isinstance(body_ids,list) and len(body_ids)==len(set(body_ids))==34 and body_ids==sorted(body_ids),
         'All 34 original ordered body identities required for video selection')
    index={(r['subject_id'],r['arm']):r for r in arms}
    need(len(index)==len(arms) and all(identity in body_ids and arm in ARMS for identity,arm in index),
         'Duplicate or undeclared native arm in video inventory')
    return [deepcopy(index.get((identity,arm),dict(subject_id=identity,arm=arm,video=None,
        availability='native_arm_absent_or_unreported'))) for identity in body_ids[:2] for arm in ARMS]


def reported_pair_counts(schedule,outcomes):
    """Metadata arithmetic only; real native evidence must be audited separately.

    Explicit missing/failed/unlaunched arms preserve the fixed denominator. A
    caller's statuses can never turn this helper into a qualification receipt.
    """
    need(schedule.get('schema')==PAIR_PLAN_SCHEMA and schedule.get('protocol')==evaluation_protocol() and
         schedule.get('body_denominator')==34 and len(schedule.get('pairs',[]))==34 and
         schedule.get('payloads_verified') is False and schedule.get('native_access_authorized') is False,
         'Original metadata-only fixed 34-body schedule required')
    attempts=schedule['attempts']
    need(isinstance(outcomes,list) and len(outcomes)==len(attempts)<=68 and
         [r for pair in schedule['pairs'] for r in pair['attempts']]==attempts,
         'Complete fixed native arm outcome slots required')
    statuses=('passed','physical_failure','infrastructure_abort','missing_result','planned_unlaunched')
    for row,attempt in zip(outcomes,attempts):
        need(isinstance(row,dict) and set(row)=={'subject_id','arm','job_id','launched','physics_steps','status'} and
             all(row[k]==attempt[k] for k in ('subject_id','arm','job_id')) and row['status'] in statuses and
             type(row['launched']) is bool and (row['physics_steps'] is None or
                 type(row['physics_steps']) is int and row['physics_steps']>=0),
             'Original arm identity/order and explicit launch/physics/status required')
        if row['status'] in ('passed','physical_failure'):
            need(row['launched'] and row['physics_steps'] is not None and row['physics_steps']>0,
                 'Native success/safety failure requires actual launched physics')
        if row['status']=='missing_result':need(row['launched'], 'Missing native result requires a launch')
        if row['status']=='planned_unlaunched':need(not row['launched'] and row['physics_steps']==0,
             'Unlaunched arm cannot invent physics')
        if not row['launched']:need(row['physics_steps'] in (0,None), 'Unlaunched arm cannot invent physics')
    index={(r['subject_id'],r['arm']):r for r in outcomes}
    need(len(index)==len(outcomes), 'Duplicate body/arm outcomes')
    paired=dict(both=0,only_expert=0,only_act=0,neither=0);executed=0;expert=0;act=0
    for pair in schedule['pairs']:
        values=[index.get((pair['subject_id'],arm)) for arm in ARMS]
        if not all(value is not None and value['launched'] and value['physics_steps'] is not None and
                   value['physics_steps']>0 for value in values):continue
        executed+=1;e,a=[value['status']=='passed' for value in values];expert+=e;act+=a
        paired['both' if e and a else 'only_expert' if e else 'only_act' if a else 'neither']+=1
    available=sum(pair['selected_target'] is not None for pair in schedule['pairs'])
    return dict(schema='m2_given_contact_reported_pair_counts_v1',body_denominator=34,available_bodies=available,
        availability=binary_rate(available,34),executed_pairs=executed,reported_milestone_reached=executed>=30,
        expert_success=binary_rate(expert,executed),act_success=binary_rate(act,executed),paired=paired,
        status_counts={status:sum(row['status']==status for row in outcomes) for status in statuses},
        native_evidence_verified=False,heldout_evaluation_complete=False)


def _observed(env):
    state=env.state();stamp=runtime.shared.contract._scalar(state['time_s'],'actual native time')
    q=runtime.shared._vector(state['q'],(7,),'actual native measured q')
    base=runtime.shared.contract._pose(env.base,'actual native world_from_base')
    pose=np.eye(4);pose[:3,:3]=base[:3,:3].T@runtime.shared._vector(state['rotation_world'],(3,3),'actual native full R')
    pose[:3,3]=runtime.shared._vector(state['tcp_base'],(3,),'actual native TCP')
    runtime.shared.contract._pose(pose,'actual native full TCP pose')
    return dict(physics_row=len(env.rows),time_s=stamp,q=q.tolist(),pose_base=pose.tolist())


def _interpolator(controller):
    trajectory=getattr(controller,'trajectory',None)
    if trajectory is None:return None
    handoff=runtime.handoff;profile='l2';transition_sha=None
    if hasattr(handoff,'STOP_PLANNING_LIMITS'):
        if list(trajectory.input.max_jerk)==[handoff.STOP_PLANNING_LIMITS['jerk']]*7:
            profile='stop';receipts=getattr(controller,'l2_handoffs',[])
            need(controller.control_owner not in ('TEACHER_L2','ACT_L2') and receipts,
                 'Persisted STOP profile requires an explicit completed L2 profile transition')
            transition=receipts[-1].get('parking_stop_profile_transition')
            need(isinstance(transition,dict) and set(transition)=={'schema','handoff_policy_id','from_profile','to_profile',
                 'changed_fields','before','before_sha256','after','after_sha256'} and
                 transition['schema']=='m2_given_l2_to_stop_profile_transition_v3' and
                 transition['handoff_policy_id']==handoff.POLICY_ID and transition['from_profile']==handoff.PLANNING_LIMITS and
                 transition['to_profile']==handoff.STOP_PLANNING_LIMITS and transition['changed_fields']==['max_jerk'] and
                 transition['before_sha256']==handoff.digest(transition['before']) and
                 transition['after_sha256']==handoff.digest(transition['after']), 'Actual explicit STOP transition binding differs')
            expected=deepcopy(transition['before'])
            need(expected['fields']['max_jerk']==[handoff.PLANNING_LIMITS['jerk']]*7, 'Original L2 jerk profile differs')
            expected['fields']['max_jerk']=[handoff.STOP_PLANNING_LIMITS['jerk']]*7
            need(expected==transition['after'], 'STOP transition changed more than the single jerk bound')
            transition_sha=handoff.digest(transition)
        snapshot=handoff.input_snapshot(trajectory,profile=profile)
    else:snapshot=handoff.input_snapshot(trajectory)
    return dict(input=snapshot,output=handoff.output_snapshot(trajectory),planning_profile=profile,
        stop_profile_transition_sha256=transition_sha,
        objects={name:id(getattr(trajectory,name)) for name in ('input','output','generator')},trajectory_object=id(trajectory))


def _labels(controller,subphase,*,phase=None):
    labels=dict(phase=controller.manager.state.phase.value if phase is None else phase,subphase=subphase,
        actual_control_owner=controller.control_owner,actor=dict(phase=controller.actor_phase),
        raw_intent=deepcopy(getattr(controller,'_current_intent_audit',None)),
        previous_command_q=(None if not hasattr(controller,'q') else np.asarray(controller.q).tolist()),
        contact_active=bool(getattr(controller,'_contact_active',False)),
        parking_interpolator_hold=deepcopy(getattr(controller,'_parking_tail_audit',None)))
    residual=getattr(controller,'_current_stop_residual_audit',None)
    if residual is not None:labels['stop_residual_adapter_tick']=deepcopy(residual)
    return labels


class NativeAudit:
    """Neutral observer around one original tick; no capture/physics/setter API."""
    def __init__(self,path,metadata):
        self.path=Path(path)
        self.stream=gzip.open(self.path,'xt') if self.path.suffix=='.gz' else self.path.open('x')
        self.metadata=deepcopy(metadata)
        self.rows=0;self.errors=[];self.receipt=None

    def tick(self,controller,command,subphase,operation):
        need(self.receipt is None, 'Native audit already closed')
        command=runtime.shared._vector(command,(7,),'actual issued native command')
        before=_observed(controller.env);labels=_labels(controller,subphase);trajectory=_interpolator(controller)
        error=None
        try:return operation()
        except BaseException as failure:
            error=repr(failure);self.errors.append(dict(stage='original_tick',physics_row=before['physics_row'],error=error));raise
        finally:
            try:self._append(controller,command,subphase,before,labels,trajectory,error)
            except BaseException as failure:
                self.errors.append(dict(stage='audit',physics_row=before['physics_row'],error=repr(failure)))
                if error is None:raise

    def _append(self,controller,command,subphase,before,labels,trajectory,error):
        env=controller.env;count=before['physics_row']
        if len(env.rows)==count:return
        need(len(env.rows)==count+1 and count==self.rows, 'Neutral audit requires exactly one actual physics tick')
        row=env.rows[-1];after=_observed(env)
        need(after['time_s']==row['time_s'] and abs(after['time_s']-before['time_s']-1/240.)<=1e-8 and
             np.array_equal(after['q'],row['q']) and np.array_equal(command,row['q_command']),
             'Neutral audit actual state/command/native clock readback differs')
        end=_labels(controller,subphase,phase=row['phase'])
        need(labels['actual_control_owner']==end['actual_control_owner'] and labels['raw_intent']==end['raw_intent'],
             'Actual owner/held raw target changed inside one native tick')
        need(labels.get('stop_residual_adapter_tick')==end.get('stop_residual_adapter_tick'),
             'Actual STOP residual sidecar changed inside one native tick')
        need(trajectory==_interpolator(controller), 'Native physics tick changed persistent interpolator state or objects')
        raw=labels['raw_intent']
        if raw is not None:
            need(trajectory is not None and np.array_equal(raw['live_target_q'],trajectory['input']['fields']['target_position']) and
                 np.array_equal(raw['live_target_after_step_q'],trajectory['input']['fields']['target_position']),
                 'Held raw target differs from actual live interpolator readback')
        measurements={key:deepcopy(value) for key,value in row.items() if key in
            ('compression_m','tcp_speed_m_s','gap_m','tangent_m','angle_deg','axis_world','hold_elapsed_s','audit_hold_s') or
            (key.endswith('_N') and np.isscalar(value))}
        event=dict(schema='m2_given_contact_neutral_native_tick_v1',physics_row=count,time_s=after['time_s'],
            observed_before=before,observed_after=after,controller_before=labels,controller_after=end,
            actual_control_owner=labels['actual_control_owner'],raw_intent=raw,q_command=command.tolist(),
            interpolator=trajectory,measurements=measurements,state_audit='completed' if 'physical_violations' in row else 'not_completed',
            physical_violations=deepcopy(row.get('physical_violations')),tick_error=error,
            native_row=deepcopy(row),
            native_retainer_audit=deepcopy(row.get('native_retainer_audit')),
            retainer_front_audit=deepcopy(row.get('retainer_front_audit')))
        if 'stop_residual_adapter_tick' in labels:
            event['stop_residual_adapter_tick']=deepcopy(labels['stop_residual_adapter_tick'])
        if subphase=='given_l2_park':
            def parking(sample,speed):
                value=dict(schema=runtime.handoff.POSE_SCHEMA,source='actual_native_env_state',
                    physics_row=sample['physics_row'],time_s=sample['time_s'],pose_base=sample['pose_base'],
                    measured_q=sample['q'],tcp_speed_m_s=speed)
                runtime.handoff.validate_sample(value);return value
            # This evidence is sampled from actual native state on both sides.
            event['parking_pose_before']=parking(before,env.rows[-2]['tcp_speed_m_s'] if count else None)
            event['parking_pose_after']=parking(after,row.get('tcp_speed_m_s'))
        self.stream.write(json.dumps(event,allow_nan=False,sort_keys=True)+'\n');self.stream.flush();self.rows+=1

    def close(self):
        if self.receipt is not None:return self.receipt
        self.stream.close()
        value=dict(schema='m2_given_contact_neutral_native_audit_v1',metadata=self.metadata,
            rows=self.rows,trace=runtime.reference(self.path),complete=not self.errors,errors=deepcopy(self.errors),
            source_inventory=deepcopy(getattr(self,'sources',{})))
        index=runtime.write_new(self.path.with_suffix('.index.json'),value)
        self.receipt=dict(value,index=index);return self.receipt


def attach_native_audit(controller,config,output):
    """Observe the approved formal routes; the original tick is called once."""
    if ROUTE_KEY in config:
        binding=preflight(config,output=output)
        need(binding['stage']=='native', 'Preparation capability cannot attach a native observer')
        metadata=binding['row']['metadata'];processing=binding['capability']['processing']
    elif 'm2_given_contact_deployment' in config:
        binding=deployment.preflight(config,output=output)
        metadata=binding['attempt']['metadata'];processing=binding['permit']['processing']
    else:return None
    need(not hasattr(controller,'_neutral_native_audit'), 'Duplicate actual-step audit attachment')
    sources=_sources(processing);original=controller._tick
    audit=NativeAudit(Path(output)/'given_control_native.jsonl.gz',metadata)
    audit.sources=sources
    def tick(self,command,subphase,*args,**kwargs):
        return audit.tick(self,command,subphase,lambda:original(command,subphase,*args,**kwargs))
    controller._neutral_native_audit=audit;controller._neutral_original_tick=original
    controller._tick=MethodType(tick,controller)
    return audit


def close_native_audit(controller):
    audit=getattr(controller,'_neutral_native_audit',None)
    if audit is None:return None
    try:
        receipt=audit.close();controller.neutral_native_audit=receipt
        return receipt
    finally:controller._tick=controller._neutral_original_tick


def _audit_act_execution(events,ipc,run,dep,permit,predictor):
    """Reconstruct observations/H20 and attribute only actually executed E10.

    The real qualifier creates its official CPU predictor internally. This pure
    numerical checker alone is not a capability or a model qualification.
    """
    from .m2_given_contact_intent import validate_native_intents
    from .m2_stage import CausalStage
    metadata=dep['metadata'];ref=runtime.pose.require_route(metadata)
    validate_native_intents(events,metadata)
    need(events and isinstance(ipc,list), 'Actual native events and complete pipe trace required')
    samples=[event['observed_before'] for event in events]+[events[-1]['observed_after']]
    stage=CausalStage();encoded=[];phases=[]
    for index,sample in enumerate(samples):
        need(sample['physics_row']==index, 'Actual inference observation rows are discontinuous')
        state=runtime.shared.contract.encode_state(sample['q'],ref.pose_base,1.,metadata=metadata)
        encoded.append(state);phases.append(stage.update(float(np.linalg.norm(state[7:10])),sample['time_s']))
    inferences=[record for record in ipc if record.get('event')=='inference']
    queries=run.get('l2_queries');actions=run.get('l2_actions')
    need(isinstance(queries,list) and isinstance(actions,list) and len(queries)==len(inferences)>0,
         'Every original ACT query must have one real pipe exchange')
    previous=None;predictions=[];requests=[];accepted=[];maximum=0.;failed=run.get('success') is False
    for index,(record,query) in enumerate(zip(inferences,queries)):
        need(type(record.get('accepted')) is bool and type(query.get('accepted')) is bool and
             record.get('simulated_time_pause_s')==0. and query.get('index')==index and
             (record['accepted'] and query['accepted'] or failed and index==len(inferences)-1 and query.get('failure_reason')),
             'Failed/unpaired inference cannot qualify a bridge smoke')
        request=record['request'];response=record.get('response')
        previous=deployment.validate_request(request,dep,permit,previous=previous)
        row=request['context']['physics_row'];need(type(row) is int and 12<=row<len(samples), 'Inference outside actual native rows')
        need(query.get('physics_row')==row and query.get('time_s')==samples[row]['time_s'] and
             request['context']['observed_at_s']==samples[row]['time_s'] and
             np.array_equal(request['measured_q'],samples[row]['q']), 'Inference q/clock differs from actual same-time native state')
        times=request['history_times'];history=request['history'];start=row-12
        need(np.array_equal(history,[encoded[start],encoded[row]]) and
             np.array_equal(times,[samples[start]['time_s'],samples[row]['time_s']]) and
             phases[start]==phases[row]==request['context']['context'][-1] and
             query.get('history')==history and query.get('history_times')==times,
             'Inference history is not the two real same-context 20 Hz observations')
        need(all(e['actual_control_owner']=='ACT_L2' and e['controller_before']['subphase'] in
             ('given_l2_history_wait','given_l2_motion') for e in events[start:row]),
             'History crosses STOP/ALIGN/contact/home or a different action owner')
        requests.append(request);accepted.append(query['accepted'])
        if response is None:
            need(failed and not record['accepted'] and not query['accepted'] and record.get('error'),
                 'Missing returned chunk requires the original explicit failed IPC exchange')
            predictions.append(None);continue
        block=deployment.decode_response(response,request)
        prediction=predictor(request['measured_q'],request['estimated_target_base'],request['confidence'],metadata=metadata,
            history=np.asarray(history),history_times=np.asarray(times))
        if isinstance(prediction,runtime.shared.contract.JointActionBlock):prediction=prediction.values
        prediction=runtime.shared._vector(prediction,(20,7),'independent official CPU H20 prediction')
        difference=float(np.max(np.abs(prediction-block.values)));maximum=max(maximum,difference)
        need(np.array_equal(prediction,block.values), 'Actual CPU child H20 cannot reconstruct byte-identically')
        absolute=block.observed_q+prediction
        if query['accepted']:
            need(record['accepted'] and np.array_equal(query.get('absolute_h20'),absolute),
                 'Original native query chunk differs from IPC/CPU reconstruction')
        else:need('absolute_h20' not in query, 'Rejected query cannot declare an accepted native prefix')
        predictions.append(absolute)
    covered=set();ranges=[];last=-1;prefixes={index:[] for index in range(len(requests))}
    for action in actions:
        query=action.get('query_index');prefix=action.get('prefix_index');start=action.get('physics_row_start')
        need(type(query) is int and query in prefixes and accepted[query] and type(prefix) is int and 0<=prefix<10 and
             type(start) is int and last<start<=len(events) and start%12==0,
             'Native ACT action query/order/E10 prefix/grid differs')
        need(prefix==len(prefixes[query]) and start==requests[query]['context']['physics_row']+12*prefix and
             action.get('time_s')==samples[start]['time_s'], 'Native action skipped/reused its fixed E10 prefix or clock')
        proposal=predictions[query][prefix]
        need(np.array_equal(action.get('proposal'),proposal), 'Actual action differs from selected H20 fixed-origin prefix')
        end=start
        while end<min(start+12,len(events)) and events[end]['controller_before']['subphase']=='given_l2_motion':end+=1
        need(type(action.get('completed')) is bool and
             (end-start==12 if action['completed'] else bool(action.get('interrupted') or action.get('failure_reason'))) and
             (end>start or failed and action.get('failure_reason')) and
             action.get('physics_row_end_exclusive',end)==end,
             'Missing actual action interval or invented completed 20 Hz cell')
        for row in range(start,end):
            event=events[row];raw=event.get('raw_intent')
            need(row not in covered and event['actual_control_owner']=='ACT_L2' and raw is not None and
                 raw['setter']['physics_row_start']==start and raw['setter']['control_owner']=='ACT_L2' and
                 np.array_equal(raw['setter']['target_q'],proposal) and np.array_equal(raw['live_target_q'],proposal) and
                 np.array_equal(raw['live_target_after_step_q'],proposal), 'Actual native raw setter/target/owner differs from selected E10 action')
            covered.add(row)
        prefixes[query].append(prefix);ranges.append(dict(query_index=query,prefix_index=prefix,native_rows=[start,end]));last=start
    expected={index for index,event in enumerate(events) if event['controller_before']['subphase']=='given_l2_motion'}
    need(covered==expected and all(prefixes[i] or failed and i==len(requests)-1 for i in range(len(requests))),
         'Missing/extra native action attribution or unused accepted inference')
    return dict(schema='m2_given_contact_actual_act_reconstruction_v1',inferences=len(requests),executed_actions=len(actions),
        returned_chunks=sum(value is not None for value in predictions),failed_queries=sum(not value for value in accepted),
        executed_native_ticks=len(covered),prediction_max_abs_diff=maximum,prediction_byte_identical=True,
        action_ranges=ranges,observation_only=True,expert_l2_substitution=False)


def _read_native_events(receipt,metadata):
    """Read intact actual ticks, including a terminal failed tick, without acceptance."""
    need(isinstance(receipt,dict) and 'index' in receipt, 'Actual closed native audit receipt required')
    index=runtime.read_bound(receipt['index'])
    need(index=={k:v for k,v in receipt.items() if k!='index'} and index.get('schema')=='m2_given_contact_neutral_native_audit_v1' and
         index.get('metadata')==metadata and type(index.get('complete')) is bool and isinstance(index.get('errors'),list) and
         index['complete']==(not index['errors']), 'Actual closed native audit/index binding differs')
    path=runtime.bound_path(index['trace'])
    need(path.suffix in ('.gz','.xz','.jsonl'),'Explicit native trace transport required')
    opener=(lambda path,mode:lzma.open(path,mode,format=lzma.FORMAT_XZ)) if path.suffix=='.xz' else (gzip.open if path.suffix=='.gz' else open)
    with opener(path,'rt') as stream:events=[deployment.wire_decode(line) for line in stream]
    need(type(index.get('rows')) is int and len(events)==index['rows'], 'Missing/extra actual native audit rows')
    previous=None
    for i,event in enumerate(events):
        before=event['observed_before'];after=event['observed_after'];row=event['native_row']
        need(event.get('schema')=='m2_given_contact_neutral_native_tick_v1' and event.get('physics_row')==i and
             before.get('physics_row')==i and after.get('physics_row')==i+1 and
             (previous is None or before==previous) and abs(after['time_s']-before['time_s']-1/240.)<=1e-8 and
             event.get('time_s')==after['time_s']==row.get('time_s') and
             event.get('state_audit')==('completed' if 'physical_violations' in row else 'not_completed') and
             event.get('physical_violations')==row.get('physical_violations'),
             'Native before/after row/clock/physical evidence is discontinuous')
        if event.get('tick_error') is not None or event.get('physical_violations')!=[] or event.get('state_audit')!='completed':
            need(i==len(events)-1 and index['errors'] and index['complete'] is False,
                 'Failed actual native tick must remain the terminal failed tick')
        for sample in (before,after):
            runtime.shared.contract._pose(sample['pose_base'],'actual measured TCP full rotation')
            runtime.shared._vector(sample['q'],(7,),'actual native measured q')
        need(np.array_equal(after['q'],row.get('q')) and np.array_equal(event['q_command'],row.get('q_command')),
             'Native actual q/command mirror differs from its original physics row')
        previous=after
    need(not events or events[0]['observed_before']['time_s']==0. and np.max(np.abs(events[0]['observed_before']['q']))<=.03,
         'Each paired arm must start at its own original native home/reset')
    return events


def read_native_trace(receipt,metadata):
    events=_read_native_events(receipt,metadata)
    need(events and receipt['complete'] is True and receipt['errors']==[] and all(
         event['tick_error'] is None and event['state_audit']=='completed' and event['physical_violations']==[] for event in events),
         'Actual full native audit did not close without errors')
    return events


def _exact(left,right):
    return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)


def _current_scene_roles():
    from . import scene_store
    return {'preparation_config','prepared_complete',*scene_store.FILES,*scene_store.OPTIONAL,
        'scene_base','scene_authoring_receipt','layered_runtime_equivalence'}


def _dense_current_common(common):
    """The versioned heldout choice is dense SDKv2, never a runtime fallback."""
    from . import scene_store
    from .sdk_mdl import APPROVED_FILE_COUNT
    need(set(common)=={'schema','store_root','dependency_policy','shared_sources','fresh','experiment','base',
        'isaac_python','physics','geometry_contract','heldout_provenance'}, 'Closed current RAW native common projection required')
    provenance=common['heldout_provenance']
    need(set(provenance)=={'schema','source_schema','source_dependency_policy','target_schema','target_dependency_policy',
        'scene_owned_roles','recipe','qualified_scenes'} and provenance['schema']==CURRENT_NATIVE_COMMON_SCHEMA and
        provenance['source_schema']==scene_store.SDK_LAYER_SCHEMA and
        provenance['source_dependency_policy']==scene_store.SDK_LAYER_CURRENT_RAW_POLICY and
        provenance['target_schema']==common['schema']==scene_store.SDK_SCHEMA and
        provenance['target_dependency_policy']==common['dependency_policy']==scene_store.SDK_DEPENDENCY_POLICY and
        provenance['scene_owned_roles']==sorted(_current_scene_roles()) and len(provenance['qualified_scenes'])==4,
        'Exact predeclared current RAW to dense SDK heldout representation required')
    shared=common['shared_sources'];sdk={'sdk_mdl_manifest',*('sdk_mdl_'+str(i).zfill(4) for i in range(APPROVED_FILE_COUNT))}
    need(sdk<=set(shared)<=sdk|scene_store.RUNTIME_SOURCES|scene_store.OPTIONAL_RUNTIME_SOURCES and
        not set(shared)&_current_scene_roles(), 'Dense heldout cannot inherit scene-owned layered provenance')
    for name,ref in shared.items():scene_store._source_shape(ref,allow_empty=scene_store._sdk_file_name(name))
    cases=provenance['qualified_scenes'];identities=set();indices=set();sidecars=set()
    from . import layered_scene_authoring as layers,m2_given_contact_lossless_mapping as lossless
    for case in cases:
        lossless._keys(case,'subject_id split pose_template scene_index historical_prepared_complete layered_runtime_equivalence '
            'prepared_complete scene_authoring_receipt sidecar scene_sources')
        runtime.shared.require_membership(case['split'],case['subject_id'])
        need(case['pose_template'] in deployment.POSE_TEMPLATES and type(case['scene_index']) is int and case['scene_index']>=0,
             'Exact qualified current RAW scene index/pose required')
        identities.add((case['split'],case['subject_id']));indices.add(case['scene_index'])
        refs=case['scene_sources'];required=_current_scene_roles()-set(scene_store.OPTIONAL)
        need(required<=set(refs)<=required|set(scene_store.OPTIONAL), 'Complete known scene-owned current RAW roles required')
        for ref in refs.values():scene_store._source_shape(ref)
        side=case['sidecar'];lossless._keys(side,'schema dependency_policy historical_prepared_complete authoring_receipt '
            'historical_prepared_sources scene_index dependency_inventory original_selected original_dense original_iteration_proof '
            'historical_authored_semantics_sha256 historical_digest_scope physics_executed physical_validation training_eligible USD_written')
        sidecars.add(lossless._reference(case['layered_runtime_equivalence'])['path'])
        need(side['schema']==layers.CURRENT_RAW_SIDECAR_SCHEMA and side['dependency_policy']==scene_store.SDK_LAYER_CURRENT_RAW_POLICY and
            type(side['scene_index']) is int and side['scene_index']==case['scene_index'] and
            side['historical_digest_scope']=='original_authoring_cpu_sdf_registry_only' and
            all(side[name] is False for name in ('physics_executed','physical_validation','training_eligible','USD_written')) and
            _exact(side['historical_prepared_complete'],case['historical_prepared_complete']) and
            _exact(side['authoring_receipt'],case['scene_authoring_receipt']) and
            _exact(refs['layered_runtime_equivalence'],case['layered_runtime_equivalence']) and
            _exact(refs['scene_authoring_receipt'],case['scene_authoring_receipt']) and
            _exact(lossless._file(refs['prepared_complete']),lossless._file(case['prepared_complete'])),
            'Closed current RAW qualified provenance differs')
        need(_exact(side['dependency_inventory'],dict(schema=layers.INVENTORY_SCHEMA,primary=refs['scene'],base=refs['scene_base'])) and
            _exact(side['historical_prepared_sources'],{k:v for k,v in refs.items() if k in {'preparation_config',*scene_store.FILES,*scene_store.OPTIONAL}}),
            'Qualified historical scene/primary/base source roles differ')
        lossless._current_proof(side['original_iteration_proof'])
    need(len(identities)==len(indices)==len(sidecars)==4 and all(
        len([case for case in cases if case['split']==split])==2 and
        {case['pose_template'] for case in cases if case['split']==split}==deployment.POSE_TEMPLATES
        for split in ('train','validation')), 'Exactly four distinct two-pose TRAIN/VAL provenance records required')
    return common


def _qualified_native_common(configs,permit,mapping,processing):
    """Classify only after all four registered current-scene headers authenticate.

    The caller first verifies the actual admission/smokes and authenticates the
    entire original mapping through deployment._allocation_mapping. This adds
    exact per-case source/sidecar rebinding before choosing the dense TEST route.
    """
    from . import scene_store,layered_scene_authoring as layers,m2_given_contact_lossless_mapping as lossless
    current=[c['compact_scene_storage']['dependency_policy']==scene_store.SDK_LAYER_CURRENT_RAW_POLICY for c in configs]
    # Configs cannot downgrade an already authenticated current-family authority.
    family=any(current) or any('layered_runtime_equivalence' in c['compact_scene_storage']['sources'] for c in configs) or any(
        {'current_raw','raw_preparation_config'}&set(scene) for scene in mapping.get('scenes',[]))
    authorities=[processing]
    if not _exact(permit.get('processing',{}),processing):authorities.append(permit.get('processing',{}))
    for authority in authorities:
        if 'plan' not in authority:continue
        header=runtime.read_bound(authority['plan']);probe=lossless._Metadata(header)
        report_header=probe.read(header['derivation']) if 'derivation' in header else {}
        recipe_header=probe.read(header['source_recipe']) if 'source_recipe' in header else {}
        family=family or 'current_raw_equivalence' in header or 'current_raw_prepared' in report_header or \
            bool({'current_raw_equivalence','current_raw_output_root','raw_selected','baseline_provenance'}&
                set(report_header.get('authoring_request',{}))) or \
            'current_raw_equivalence' in report_header.get('cpu_execution',{}) or \
            bool({'current_raw_equivalence','scene_dependency_policy'}&set(recipe_header)) or \
            recipe_header.get('collector_conditions',{}).get('scene_dependencies',{}).get('dependency_policy')==scene_store.SDK_LAYER_CURRENT_RAW_POLICY or \
            'fixed:layered_runtime_equivalence' in recipe_header.get('inventory',{}) or \
            any({'current_raw','raw_preparation_config'}&set(entry.get('derived',{})) for entry in report_header.get('scenes',[]))
    if not family:
        common=[_native_common(config) for config in configs]
        need(len(common)==4 and all(row==common[0] for row in common),
             'All four actual bridge smokes must share original native conditions')
        return common[0]
    need(len(configs)==len(permit['attempts'])==4 and all(current), 'Four unmixed authenticated current RAW smoke scenes required')
    need('plan' in processing and _exact(permit.get('processing'),processing), 'Exact current RAW processing authority required')
    plan=runtime.read_bound(processing['plan']);reader=lossless._CurrentMetadata(plan)
    need(plan.get('current_raw_equivalence') is True, 'Explicit current RAW processing plan required')
    report=reader.read(plan['derivation']);recipe=reader.read(plan['source_recipe'])
    need(plan.get('current_raw_equivalence') is True and recipe.get('current_raw_equivalence') is True and
        report.get('authoring_request',{}).get('current_raw_equivalence') is True and
        report.get('cpu_execution',{}).get('current_raw_equivalence') is True and
        report.get('cpu_execution',{}).get('dependency_policy')==scene_store.SDK_LAYER_CURRENT_RAW_POLICY and
        isinstance(report.get('current_raw_prepared'),list) and bool(report['current_raw_prepared']) and
        recipe.get('scene_dependency_policy')==scene_store.SDK_LAYER_CURRENT_RAW_POLICY and
        recipe.get('collector_conditions',{}).get('scene_dependencies',{}).get('dependency_policy')==scene_store.SDK_LAYER_CURRENT_RAW_POLICY,
        'Explicit authenticated current RAW source recipe required')
    conditions=recipe['collector_conditions']
    headers=[]
    # All source-role/identity headers precede the first sidecar/COMPLETE leaf.
    for config,row in zip(configs,permit['attempts']):
        binding=config['compact_scene_storage'];scene_store._binding_shape(binding);sources=binding['sources']
        for ref in sources.values():reader.bind(ref)
        matches=[(i,entry['derived']) for i,entry in enumerate(report['scenes']) if
            entry['derived']['subject_id']==row['subject_id'] and entry['derived']['split']==row['split'] and
            _exact(entry['derived']['patient_receipt'],row['pose_binding']['patient_receipt'])]
        need(len(matches)==1, 'Exact current RAW smoke scene identity required')
        index,scene=matches[0]
        need(scene['template']==row['pose_template'] and sum(_exact(scene,candidate) for candidate in mapping['scenes'])==1 and
            _exact(sources,report['layered_authoring']['runtime_equivalence'][index]['derived_sources']),
            'Smoke source roles differ from the authenticated current RAW mapping')
        record=scene['current_raw'];result=scene['derivation']
        lossless._keys(record,'historical_prepared_complete layered_runtime_equivalence prepared_complete')
        for ref in [*record.values(),result['scene_authoring_receipt'],*result['dependency_inventory'].values()]:
            if isinstance(ref,dict):reader.bind(ref)
        fresh={k:v for k,v in config['fresh_expert'].items() if k!='plans'}
        sdk={k:v for k,v in sources.items() if k=='sdk_mdl_manifest' or scene_store._sdk_file_name(k)}
        need('m2_given_contact_control' not in config and 'raw_multimodal_capture' not in config['fresh_expert'] and
            _exact(config.get('physics'),layers.ITERATIONS) and _exact(config['physics'],conditions['physics']) and
            _exact(fresh,conditions['fresh']) and config['isaac_python']==conditions['isaac_python'] and
            _exact(sdk,conditions['scene_dependencies']['sources']),
            'Current RAW smoke changed the full qualified native/shared SDK conditions')
        headers.append((config,row,index,scene))
    cases=[];verified=[];projected=[]
    geometry_keys=('geometry_id','geometry_source','geometry_source_sha256','posture_limits','freshly_generated')
    for config,row,index,scene in headers:
        sources=config['compact_scene_storage']['sources'];record=scene['current_raw'];result=scene['derivation']
        sidecar=reader.read(record['layered_runtime_equivalence']);receipt=reader.read(result['scene_authoring_receipt'])
        need(type(sidecar.get('scene_index')) is int and sidecar['scene_index']==index and
            receipt.get('schema')==layers.RECEIPT_SCHEMA and
            _exact(receipt['scenes'][index]['dependency_inventory'],result['dependency_inventory']),
            'Exact current RAW authoring receipt/index/inventory required')
        historical=sidecar['historical_prepared_sources'];expected_roles={'preparation_config',*scene_store.FILES}
        need(expected_roles<=set(historical)<=expected_roles|set(scene_store.OPTIONAL), 'Closed historical prepared source inventory required')
        root=Path(record['historical_prepared_complete']['file']).parent
        for role,ref in historical.items():
            bound=reader.bind(ref);name='prepare.json' if role=='preparation_config' else {**scene_store.FILES,**scene_store.OPTIONAL}[role]
            need(bound['path']==str(root/name) and _exact(ref,sources[role]), 'Exact historical current RAW scene source role required')
        expected=layers.current_raw_sidecar_record(record['historical_prepared_complete'],result['scene_authoring_receipt'],
            historical,receipt,receipt['scenes'][index],index)
        need(_exact(sidecar,expected), 'Full authenticated current RAW sidecar provenance required')
        old=reader.read(record['historical_prepared_complete']);new=reader.read(record['prepared_complete'])
        need(not {'scene_dependency_policy','layered_runtime_equivalence'}&set(old) and
            _exact(new,dict(old,scene_dependency_policy=scene_store.SDK_LAYER_CURRENT_RAW_POLICY,
                layered_runtime_equivalence=record['layered_runtime_equivalence'])), 'Current COMPLETE must retain the exact historical content')
        prep=reader.read(sources['preparation_config']);geometry={key:config.get(key) for key in geometry_keys}
        need(_exact(geometry,{key:prep.get(key) for key in geometry_keys}), 'Qualified smoke geometry/native controls differ from prepared metadata')
        cases.append(dict(subject_id=row['subject_id'],split=row['split'],pose_template=row['pose_template'],scene_index=index,
            **{key:deepcopy(record[key]) for key in record},scene_authoring_receipt=deepcopy(result['scene_authoring_receipt']),
            sidecar=sidecar,scene_sources={k:deepcopy(v) for k,v in sources.items() if k in _current_scene_roles()}))
        verified.append((config,geometry))
    # Every case has completed provenance authentication before classification.
    for config,geometry in verified:
        sources=config['compact_scene_storage']['sources']
        dense=deepcopy(config);dense['compact_scene_storage'].update(schema=scene_store.SDK_SCHEMA,
            dependency_policy=scene_store.SDK_DEPENDENCY_POLICY,sources={k:deepcopy(v) for k,v in sources.items()
                if k not in {'scene_base','scene_authoring_receipt','layered_runtime_equivalence'}})
        common=_native_common(dense);common.update(physics=deepcopy(config['physics']),geometry_contract=geometry)
        projected.append(common)
    need(all(_exact(row,projected[0]) for row in projected), 'All four current RAW smokes must share exact physical/camera/SDK conditions')
    common=projected[0];common['heldout_provenance']=dict(schema=CURRENT_NATIVE_COMMON_SCHEMA,
        source_schema=scene_store.SDK_LAYER_SCHEMA,source_dependency_policy=scene_store.SDK_LAYER_CURRENT_RAW_POLICY,
        target_schema=scene_store.SDK_SCHEMA,target_dependency_policy=scene_store.SDK_DEPENDENCY_POLICY,
        scene_owned_roles=sorted(_current_scene_roles()),recipe=reader.bind(plan['source_recipe']),qualified_scenes=cases)
    return _dense_current_common(common)


def _native_common(config):
    from . import scene_store
    binding=config['compact_scene_storage'];scene_store._binding_shape(binding)
    need(binding['dependency_policy']!=scene_store.SDK_LAYER_CURRENT_RAW_POLICY,
         'An authenticated current RAW projection is required before classifying scene/shared roles')
    prepared={'preparation_config','prepared_complete',*scene_store.FILES,*scene_store.OPTIONAL}
    return dict(schema=binding['schema'],store_root=binding['store_root'],dependency_policy=binding['dependency_policy'],
        shared_sources={k:deepcopy(v) for k,v in binding['sources'].items() if k not in prepared},
        fresh={k:deepcopy(v) for k,v in config['fresh_expert'].items() if k not in ('plans','raw_multimodal_capture')},
        experiment=config['experiment'],base=config['base'],isaac_python=config['isaac_python'])


def _native_owner(row,terminal_ref,permit_ref):
    """Original guard/batch -> launched native child -> normal terminal chain."""
    from . import process_guard as guard
    root=Path(row['output']).parent.parent;terminal=runtime.read_bound(terminal_ref);run=runtime.read_bound(runtime.reference(root/'run.json'))
    need(_file(terminal_ref)['path']==str(root/'result.json') and terminal.get('run_id')==root.name==row['owner_run_id'] and
         terminal.get('completed') is True and terminal.get('gpu_released') is True and
         all(terminal.get(k)==run.get(k) for k in ('run_id','config','config_sha256','code_commit','command','resource_guard','started_utc')),
         'Exact original native owner/result/run binding required')
    source=runtime.reference(guard.__file__);stage=terminal.get('stage',{});batch=terminal.get('config',{})
    need(source['sha256']==runtime.GUARD_SHA256 and terminal.get('resource_guard')==dict(path=source['path'],
         sha256=source['sha256'],gpu_exit_grace_s=guard.EXIT_GPU_GRACE_S) and
         stage.get('passed') is True and stage.get('exit_code')==0 and stage.get('process_group_drained') is True and
         all(stage.get(key)==[] for key in ('errors','signals','deferred_signals')) and
         batch.get('experiment')=='isaac_batch' and type(batch.get('parallelism')) is int and 1<=batch['parallelism']<=4,
         'Original normal native stage exit/drain/GPU release/guard required')
    jobs=[job for job in batch.get('jobs',[]) if job.get('id')==row['job_id']]
    need(len(jobs)==1, 'Native arm outside original frozen batch job inventory')
    job=jobs[0];config_ref=dict(path=job['config'],sha256=job['sha256']);config=runtime.read_bound(config_ref)
    if ROUTE_KEY in config:
        binding=preflight(config,output=row['output']);need(binding['reference']==permit_ref and binding['row']['episode_id']==row['episode_id'],
             'Native arm changed its exact sealed capability/config')
        book=binding['capability']
    else:
        binding=deployment.preflight(config,output=row['output'])
        need(config['m2_given_contact_deployment']['permit']==permit_ref and binding['attempt']==row,
             'Actual smoke changed its original permit/config')
        book=binding['permit']
    command=terminal.get('command');descriptor=runtime.read_bound(book['processing']['processing_freeze'])
    need(terminal.get('code_commit')==descriptor['reviewed_commit'] and isinstance(command,list) and len(command)==9 and
         command[1:6]==['-u','-B','-m','sim.ausc_policy.isaac_batch','--config'] and command[7:]==['--output',str(root/'output')] and
         runtime.read_bound(dict(path=command[6],sha256=terminal['config_sha256']))==batch,
         'Original reviewed native owner source/command/configuration changed')
    need(batch['jobs']==[dict(id=attempt['job_id'],config=attempt['config'],sha256=runtime.reference(attempt['config'])['sha256'])
         for attempt in book['attempts'] if attempt['owner_run_id']==row['owner_run_id']],
         'Native owner jobs must preserve every exact predeclared arm in order')
    output=Path(row['output']);launch_ref=runtime.reference(output.parent/(row['job_id']+'.launch.json'))
    child_ref=runtime.reference(output.parent/(row['job_id']+'.result.json'));complete_ref=runtime.reference(output/'COMPLETE.json')
    launch=runtime.read_bound(launch_ref);child=runtime.read_bound(child_ref);complete=runtime.read_bound(complete_ref)
    identity=child.get('identity',{});group=stage.get('process_identity',{})
    need(launch.get('identity')==identity and launch.get('pid')==child.get('pid')==identity.get('pid') and
         identity.get('pgid')==identity.get('sid')==group.get('pgid')==group.get('sid') and
         group.get('members',{}).get(str(identity.get('pid')))==identity.get('start_ticks') and
         child.get('id')==row['job_id'] and child.get('output')==row['output'] and
         child.get('config_sha256')==config_ref['sha256'] and child.get('result_sha256')==complete_ref['sha256'] and
         child.get('exit_code')==0 and child.get('worker_complete') is True,
         'Actual launched native child identity/config/report/normal completion differs')
    shutdown=complete.get('runtime_shutdown',{})
    need(shutdown.get('app_close_returned') is True and shutdown.get('fast_shutdown') is False and
         shutdown.get('interpreter_exit_policy')=='success_only_os_exit_after_native_close_and_fsynced_receipt',
         'Normal native close receipt required')
    return dict(owner_terminal=deepcopy(terminal_ref),config=runtime.reference(runtime.bound_path(config_ref)),
        complete=complete_ref,child_result=child_ref,launch=launch_ref,identity=identity)


def _execution_inventory(cap,terminal_ref,cap_ref):
    """Authenticate the entire terminal attempt header before reading traces.

    A launch receipt is authoritative even when a child never writes its result.
    A failed, fully drained owner remains a failed owner; it is not promoted by
    successful children. This private helper accepts only the caller's already
    verified native capability, not caller-reported outcomes.
    """
    from . import process_guard as guard
    pair=verify_pair_plan(cap['pair_plan'],payloads=False);attempts=cap['attempts']
    if not attempts:
        need(terminal_ref is None, 'Geometry-unfound inventory cannot invent a native owner')
        return dict(owner_terminal=None,owner_normal=False,attempts=[],root=None)
    roots={str(Path(row['output']).parent.parent) for row in attempts}
    need(len(roots)==1, 'One predeclared finite paired native owner required')
    root=Path(next(iter(roots)));out=root/'output';terminal=runtime.read_bound(terminal_ref)
    run_ref=runtime.reference(root/'run.json');run=runtime.read_bound(run_ref)
    need(_file(terminal_ref)['path']==str(root/'result.json') and
         terminal.get('run_id')==run.get('run_id')==root.name==pair['schedule']['owner_run_id'] and
         all(terminal.get(k)==run.get(k) for k in ('run_id','config','config_sha256','code_commit','command','resource_guard','started_utc')) and
         isinstance(terminal.get('ended_utc'),str) and type(terminal.get('completed')) is bool and
         terminal.get('gpu_released') is True, 'Exact closed paired owner/run/result required')
    source=runtime.reference(guard.__file__);stage=terminal.get('stage',{});batch=terminal.get('config',{})
    descriptor=runtime.read_bound(cap['processing']['processing_freeze'])
    need(source['sha256']==runtime.GUARD_SHA256 and terminal.get('resource_guard')==dict(path=source['path'],
         sha256=source['sha256'],gpu_exit_grace_s=guard.EXIT_GPU_GRACE_S) and
         terminal.get('code_commit')==descriptor['reviewed_commit'] and stage.get('process_group_drained') is True and
         type(stage.get('exit_code')) is int and type(stage.get('passed')) is bool and
         all(isinstance(stage.get(k),list) for k in ('errors','signals','deferred_signals')) and
         batch.get('experiment')=='isaac_batch' and type(batch.get('parallelism')) is int and 1<=batch['parallelism']<=4,
         'Original guard/source/terminal drain/release and finite batch required')
    command=terminal.get('command');need(isinstance(command,list) and len(command)==9 and command[1:6]==
         ['-u','-B','-m','sim.ausc_policy.isaac_batch','--config'] and command[7:]==['--output',str(out)],
         'Original native batch command/output binding required')
    cfg_ref=dict(path=command[6],sha256=terminal['config_sha256'])
    need(runtime.read_bound(cfg_ref)==batch, 'Owner configuration bytes changed')
    expected=[];configs=[]
    for row in attempts:
        need(row['owner_run_id']==root.name and Path(row['output'])==out/row['job_id'], 'Changed paired owner/output')
        ref=runtime.reference(row['config']);config=runtime.read_bound(ref);binding=preflight(config,output=row['output'])
        need(binding['row']==row and binding['reference']==cap_ref, 'Native arm changed its exact issued configuration/capability')
        expected.append(dict(id=row['job_id'],config=row['config'],sha256=ref['sha256']));configs.append(ref)
    need(batch.get('jobs')==expected, 'Owner jobs must exactly preserve every predeclared paired arm in order')
    ids={row['job_id'] for row in attempts}
    need(out.is_dir() and all(path.name.removesuffix('.launch.json') in ids for path in out.glob('*.launch.json')) and
         all(path.name.removesuffix('.result.json') in ids for path in out.glob('*.result.json')),
         'Unexpected native launch/result outside fixed paired inventory')
    group=stage.get('process_identity',{});members=group.get('members',{});identities=set();rows=[]
    for row,config_ref in zip(attempts,configs):
        output=Path(row['output']);launch_path=out/(row['job_id']+'.launch.json');result_path=out/(row['job_id']+'.result.json')
        complete_path=output/'COMPLETE.json';launch_ref=runtime.reference(launch_path) if launch_path.exists() else None
        need(launch_ref is not None or not (output.exists() or result_path.exists()), 'Native output/result exists without its original launch')
        child_ref=runtime.reference(result_path) if result_path.exists() else None
        complete_ref=runtime.reference(complete_path) if complete_path.exists() else None
        identity=None
        if launch_ref is not None:
            launch=runtime.read_bound(launch_ref);identity=launch.get('identity',{});key=(identity.get('pid'),identity.get('start_ticks'))
            need(launch.get('id')==row['job_id'] and launch.get('output')==row['output'] and
                 launch.get('config_sha256')==config_ref['sha256'] and launch.get('pid')==identity.get('pid') and
                 type(identity.get('pid')) is int and isinstance(identity.get('start_ticks'),str) and key not in identities and
                 identity.get('pgid')==identity.get('sid')==group.get('pgid')==group.get('sid') and
                 members.get(str(identity['pid']))==identity['start_ticks'], 'Native launch identity/config/session is unbound or reused')
            identities.add(key)
            if child_ref is not None:
                child=runtime.read_bound(child_ref)
                need(all(child.get(k)==v for k,v in launch.items()) and type(child.get('exit_code')) is int and
                     type(child.get('worker_complete')) is bool and type(child.get('physical_pass')) is bool and
                     child.get('result_sha256')==(None if complete_ref is None else complete_ref['sha256']),
                     'Native partial/complete result changed its original launch/report binding')
        rows.append(dict(attempt=deepcopy(row),config=config_ref,launched=launch_ref is not None,
            launch=launch_ref,child_result=child_ref,complete=complete_ref,identity=identity))
    normal=terminal['completed'] and stage['passed'] and stage['exit_code']==0 and all(stage[k]==[] for k in ('errors','signals','deferred_signals'))
    return dict(owner_terminal=deepcopy(terminal_ref),owner_run=run_ref,owner_config=runtime.reference(runtime.bound_path(cfg_ref)),
        owner_normal=bool(normal),attempts=rows,root=str(root),stage=deepcopy(stage))


def verify_bridge_smokes(ref):
    value=runtime.read_bound(ref)
    need(set(value)=={'schema','passed','cpu_fixture_only','test_payloads_opened','protocol','admission','model','processing',
         'source_inventory','permit','arms','owner_inventories'} and value.get('schema')==SMOKE_SCHEMA and
         value.get('passed') is True and value.get('cpu_fixture_only') is False and value.get('test_payloads_opened') is False and
         value.get('protocol')==evaluation_protocol(), 'Four completed actual bridge smokes required; CPU/pilot/self-reported receipts refused')
    permit=deployment.verify_permit(value['permit']);bundle=deployment._admission_metadata(value['admission'])
    need(value['admission']==permit['admission'] and value['model']==permit['model']==bundle['admission']['model'] and
         value['processing']==permit['processing'] and value['source_inventory']==_sources(permit['processing']) and
         isinstance(value['arms'],list) and len(value['arms'])==4, 'Actual bridge qualification model/source/finite inventory changed')
    roots=set()
    for evidence,row in zip(value['arms'],permit['attempts']):
        owner=_native_owner(row,evidence['owner_terminal'],value['permit'])
        need(all(evidence.get(k)==v for k,v in owner.items()) and evidence.get('physical_audit',{}).get('passed') is True and
             evidence.get('act_reconstruction',{}).get('prediction_byte_identical') is True and
             evidence['act_reconstruction'].get('prediction_max_abs_diff')==0. and
             type(evidence['act_reconstruction'].get('inferences')) is int and evidence['act_reconstruction']['inferences']>0,
             'Actual smoke owner/physical/inference evidence changed')
        complete=runtime.read_bound(owner['complete'])
        need(complete.get('success') is True and complete.get('physical_validation') is True and
             complete.get('controller_motion_complete') is True and complete.get('errors')==[], 'Actual successful full physical smoke required')
        roots.add(str(Path(row['output']).parent.parent))
    inventories=value['owner_inventories']
    need(isinstance(inventories,list) and len(inventories)==len(roots) and {row['root'] for row in inventories}==roots,
         'All actual smoke owner trees must remain closed and bound')
    for inventory in inventories:need(_tree(inventory['root'])==inventory, 'Actual smoke owner/input bytes changed')
    return value


def _physical_scalars(run, audit, meta, *, native_config=None):
    """Recheck scalar native contact evidence and preserve all original gates."""
    need(bool(audit), 'Native physics audit is empty')
    # The caller has authenticated every source role using the native permit.
    # Preserve the original scalar gates below without changing TRAIN/VAL-only
    # dataset readers or relabeling a heldout identity.
    from . import fresh_expert as geometry
    shared=runtime.shared;pose=runtime.pose
    _interior_parking=runtime.dataset._interior_parking
    reference=pose.require_route(meta);reference.verify_sources();sources=reference.snapshot()['sources']
    plan=pose.bound_json(sources['fresh_plan']);patient=pose.bound_json(sources['patient_receipt'])
    from .fresh_expert import contact_plane_contract, contact_plane_observation
    plane=contact_plane_contract(plan)
    need(isinstance(plane,dict) and plane['contact_plane_radius_m']==.023 and
         plane['plane_tilt_limit_deg']==3. and plane['tilt_relaxed'] is False,
         'Original 23 mm, 3 degree, nonrelaxed local contact plane required')
    targets=[t for t in patient.get('targets',[]) if t.get('name')==meta['target_id']]
    need(len(targets)==1, 'Unique original patient local contact plane required')
    patient_normal=shared.contract._array(targets[0].get('contact_plane_normal_world'),(3,),'patient local plane normal')
    need(targets[0].get('contact_plane_radius_m')==.023 and
         np.allclose(patient_normal,plane['contact_plane_normal_world'],atol=1e-10,rtol=0) and
         run.get('contact_plane_contracts',{}).get(meta['target_id'])==plane and
         run['contact_plane_contracts'][meta['target_id']].get('tilt_relaxed') is False and
         contact_plane_observation(plane,np.asarray(plan['command_rotation_world'])[:,2])['passed'],
         'Original patient/plan/run local contact plane differs')
    gap=plan.get('free_approach_start_gap_m')
    need(type(gap) in (int,float) and np.isfinite(gap) and .025<=gap<=.06,
         'Authenticated checked fresh-plan free approach gap required')
    points=run.get('points',[])
    need(len(points)==1 and points[0].get('target')==meta['target_id'] and points[0].get('success') is True and
         points[0].get('final_phase')=='DONE' and points[0].get('force_audit',{}).get('passed') is True and
         points[0].get('contact_plane_audit',{}).get('passed') is True,
         'Full actual HOLD, release and DONE point evidence required')
    for i,event in enumerate(audit):
        need(type(event.get('physics_row')) is int and event['physics_row']==i and
             abs(event.get('time_s',-1)-(i+1)/240.)<=1e-8 and event.get('state_audit')=='completed' and
             event.get('physical_violations')==[] and 'tick_error' in event and event['tick_error'] is None,
             'Native physical audit missing, discontinuous or failed')
    for key in ('command_collision_samples','observed_collision_samples'):
        need(run.get(key)==len(audit), 'Original collision checks must cover every native tick')
    need(run.get('posture_command_checks')=='every commanded physics sample' and
         run.get('posture_observation_checks')=='every observed physics sample' and
         run.get('phase_failure_diagnostics')==[] and run.get('collision_failure_diagnostics')==[],
         'Original posture/phase/collision gates are missing or failed')
    world=run.get('static_world_verification',{})
    need(world.get('checked_physics_steps')==len(audit) and world.get('last_report',{}).get('clear') is True and
         world.get('last_report',{}).get('violations')==[] and
         world.get('containment_gate',{}).get('source_npz_sha256')==meta.get('collision_arrays_sha256'),
         'Original static-world/containment audit missing or changed')
    handoffs=run.get('l2_handoffs',[])
    need(len(handoffs)==1 and handoffs[0].get('target')==meta['target_id'], 'Exactly one given L2 handoff required')
    h=handoffs[0]
    need(all(h.get(key) is True for key in ('l2_coarse_reached','l2_handoff_success','alignment_success','l3_started','l3_completed')) and
         h.get('expert_l2_fallback') is False and h.get('coarse_entry',{}).get('passed') is True and
         h.get('parked_entry',{}).get('passed') is True and h.get('alignment_admission',{}).get('passed') is True and
         h.get('numerical_curve_entry',{}).get('passed') is True,
         'Measured coarse entry, persistent parking, bounded ALIGN and original L3 must all pass')
    alignment=h.get('alignment_plan',{})
    from .m2_given_contact_controller import BOUNDS
    need(alignment.get('schema')=='m2_given_contact_l3_align_v1' and alignment.get('bounds')==BOUNDS and
         alignment.get('reference_sha256')==meta['reference_sha256'] and alignment.get('reference_revision')==0 and
         alignment.get('all_240hz_commands_verified') is True and alignment.get('final_stationary_stop_included') is True and
         type(alignment.get('duration_s')) in (int,float) and 0<=alignment['duration_s']<=BOUNDS['simulation_s'] and
         type(alignment.get('planning_cpu_s')) in (int,float) and 0<=alignment['planning_cpu_s']<=BOUNDS['planning_cpu_s'],
         'Actual ALIGN reference/bounds/command verification mismatch')
    align_ticks=[e for e in audit if e['controller_after']['subphase']=='L3_ALIGN']
    need(align_ticks and abs(len(align_ticks)/240.-alignment['duration_s'])<=1e-8 and
         all(e['controller_after']['phase']=='APPROACH' for e in align_ticks), 'Actual bounded ALIGN tick duration differs')
    for kind in ('command','observed'):
        channel=h.get('alignment_actual_audit',{}).get(kind,{})
        need(type(channel.get('samples')) is int and channel['samples']==len(align_ticks)+1,
             'Actual ALIGN command/observed samples missing')
        for key,limit in [('max_translation_m',BOUNDS['translation_m']),('path_length_m',BOUNDS['translation_m']),
                          ('max_full_rotation_rad',BOUNDS['rotation_rad']),('rotation_path_rad',BOUNDS['rotation_rad'])]:
            need(type(channel.get(key)) in (int,float) and np.isfinite(channel[key]) and 0<=channel[key]<=limit+1e-12,
                 'Actual ALIGN displacement/path/rotation budget failed')
        need(type(channel.get('max_local_plane_deg')) in (int,float) and
             np.isfinite(channel['max_local_plane_deg']) and 0<=channel['max_local_plane_deg']<=3.+1e-9,
             'Actual ALIGN local-plane audit exceeds original 3 degree limit')
    # The native audit always retains observed axes; raw q_command, when
    # retained, independently supplies the command channel through original FK.
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    kin=None;base=np.asarray(reference.snapshot()['world_from_base'])
    first_align=align_ticks[0]['physics_row']
    for event in ([audit[first_align-1]] if first_align else [])+align_ticks:
        need(contact_plane_observation(plane,event['measurements'].get('axis_world'))['passed'],
             'Actual observed ALIGN axis exceeds original local-plane limit')
        if 'q_command' in event:
            if kin is None:kin=OpenArmKinematics(contact_frame=geometry.contact_frame)
            axis=base[:3,:3]@kin.tcp(event['q_command'])[:3,2]
            need(contact_plane_observation(plane,axis)['passed'],
                 'Actual commanded ALIGN axis exceeds original local-plane limit')
    expected_gate=dict(precontact_gap_m=.04,precontact_tolerance_m=.002,approach_tangential_tolerance_m=.003,
        angle_tolerance_deg=5.,approach_speed_limit_m_s=.02)
    need(all(run.get('phase_config',{}).get(k)==v for k,v in expected_gate.items()), 'Original measured coarse gate changed')
    expected_gate['precontact_gap_m']=min(.04,gap)
    need(points[0].get('checked_prepose_gap_m')==gap and
         all(points[0].get('phase_config',{}).get(k)==v for k,v in expected_gate.items()),
         'Original per-point coarse gate differs from authenticated checked plan gap')
    from types import SimpleNamespace
    from .m2_act_controller import approach_gate
    coarse=[e['physics_row']+1 for e in audit if e['controller_after']['subphase'] in ('given_l2_motion','given_l2_history_wait') and
        e['controller_after'].get('actor',{}).get('phase')=='APPROACH' and
        approach_gate(e.get('measurements',{}),SimpleNamespace(**expected_gate))['passed']]
    first=h['coarse_entry'];parked=h['parked_entry'];trigger=h.get('parking_trigger',{})
    need(coarse and first.get('physics_row')==min(coarse) and
         abs(first.get('time_s',-1)-min(coarse)/240.)<=1e-8 and type(parked.get('physics_row')) is int and
         type(trigger.get('physics_row')) is int and first['physics_row']<=trigger['physics_row']<parked['physics_row']<=len(audit) and
         parked['physics_row']-trigger['physics_row']<=5*240 and
         abs(parked.get('time_s',-1)-parked['physics_row']/240.)<=1e-8 and
         all(e['controller_after']['subphase']=='given_l2_park' and
             approach_gate(e['measurements'],SimpleNamespace(**expected_gate))['passed']
             for e in audit[trigger['physics_row']:parked['physics_row']]),
         'First actual measured coarse entry and persistent bounded parking mismatch')
    if native_config is not None and 'stop_residual_trajectory' in native_config:
        from . import m2_stop_residual_trajectory as residual
        residual.verify_native_parking(run,h,audit,meta,reference,SimpleNamespace(**expected_gate),config=native_config)
    else:_interior_parking(run,h,audit,meta,reference,SimpleNamespace(**expected_gate))
    rows=[dict(event.get('measurements',{}),phase=event['controller_after']['phase'],
        subphase=event['controller_after']['subphase'],target=meta['target_id'],
        native_retainer_audit=event.get('native_retainer_audit'),retainer_front_audit=event.get('retainer_front_audit')) for event in audit]
    independent=geometry.qualify_contact(rows,retainer_witness_contract=meta['native_retainer_audit'])
    need(independent.get('passed') is True, 'Independent native v3 contact/force/compression gate failed')
    hold=longest=0
    for row in rows:
        contact=row.get('diaphragm_N',0)+row.get('retainer_N',0)
        good=(row['phase']=='HOLD' and 3.5<=contact<=4.5 and
              contact_plane_observation(plane,row.get('axis_world'))['passed'])
        hold=hold+1 if good else 0; longest=max(longest,hold)
    need(longest>=720, 'Actual continuous 3 s native HOLD and local-plane gate required')
    release=longest_release=0
    last_hold=max(i for i,row in enumerate(rows) if row['phase']=='HOLD')
    for row in rows[last_hold+1:]:
        if row['subphase']=='home_settle':break
        good=(row['subphase']=='normal_release_confirm' and row['phase'] in ('RETRACT','DONE') and
              row.get('hall_N',float('inf'))<.2 and row.get('compression_m',float('inf'))<=.00015 and
              sum(row.get(key+'_N',float('inf')) for key in ('diaphragm','retainer'))<.2)
        release=release+1 if good else 0;longest_release=max(longest_release,release)
    need(longest_release>=48, 'Actual continuous 0.2 s native normal release confirmation required')
    need(rows[-1]['subphase']=='home_settle' and
         all(sum(row.get(key+'_N',float('inf')) for key in ('diaphragm','retainer'))<.2
             for row in rows[-36:]) and len(rows)>=36,
         'Actual release confirmation and final home settle required')


def _alignment_actual(events,reference,plane,base,*,kin=None):
    from .m2_given_contact_controller import AlignmentAudit
    if kin is None:
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        kin=OpenArmKinematics(contact_frame=runtime.geometry.contact_frame)
    rows=[event for event in events if event['controller_after']['subphase']=='L3_ALIGN']
    need(bool(rows), 'Actual ALIGN full-pose samples missing')
    first=rows[0];start=events.index(first)
    before_q=events[start-1]['q_command'] if start else first['q_command']
    audit=AlignmentAudit(reference,kin.tcp(before_q),first['observed_before']['pose_base'],plane,base)
    previous=first['observed_before']
    for event in rows:
        need(event['observed_before']==previous, 'Actual ALIGN measured full poses are discontinuous')
        audit.observe(kin.tcp(event['q_command']),'command')
        audit.observe(event['observed_after']['pose_base'],'observed')
        previous=event['observed_after']
    return audit.report()


def _physical_native(run,events,metadata,*,native_config=None):
    if native_config is None:_physical_scalars(run,events,metadata)
    else:_physical_scalars(run,events,metadata,native_config=native_config)
    command_gates=_command_gates(events)
    from .fresh_expert import contact_plane_contract
    ref=runtime.pose.require_route(metadata);plan=runtime.pose.bound_json(ref.snapshot()['sources']['fresh_plan'])
    alignment=_alignment_actual(events,ref,contact_plane_contract(plan),np.asarray(ref.snapshot()['world_from_base']))
    for event in events:
        row=event['native_row'];actual=event['observed_after'];base=np.asarray(ref.snapshot()['world_from_base'])
        need(np.array_equal(actual['q'],row['q']) and np.allclose(base[:3,:3]@np.asarray(actual['pose_base'])[:3,2],
             row['axis_world'],atol=1e-9,rtol=0), 'Actual full pose differs from original measured axis/q audit')
    need(np.max(np.abs(events[-1]['observed_after']['q']))<=.03 and
         np.max(np.abs(events[-1]['q_command']))<=.03, 'Original measured and commanded final home required')
    return dict(passed=True,physics_steps=len(events),alignment_actual=alignment,original_command_gates=command_gates,
        original_contact_hold_release_home_gates=True,original_parking_reconstructed=True)


def _probe_video(path):
    value=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,r_frame_rate,nb_frames','-of','json',str(path)],text=True))
    need(len(value.get('streams',[]))==1, 'One real video stream required')
    return value['streams'][0]


def _video_evidence(output,run,events):
    output=Path(output);video=run.get('video',{});path=output/'actual_fresh_expert.mp4'
    need(video.get('file')==str(path) and all(video.get(k)==v for k,v in
         dict(fps=10,width=640,height=360,camera='/World/RobotCam').items()), 'Original real 10 Hz fixed-camera video required')
    frame_ref=runtime.reference(output/'video_frames.json');frames=runtime.read_bound(frame_ref)
    expected=[dict(video_frame=i,physics_row=row,time_s=events[row]['time_s']) for i,row in enumerate(range(23,len(events),24))]
    need(frames==expected and video.get('frames')==len(frames)>0, 'Actual video/native frame indices or timestamps differ')
    probe=_probe_video(path)
    need(probe.get('width')==640 and probe.get('height')==360 and probe.get('r_frame_rate')=='10/1' and
         int(probe.get('nb_frames',-1))==len(frames), 'Actual encoded video dimensions/rate/frame count differ')
    return dict(video=runtime.reference(path),frame_index=frame_ref,probe=probe)


def _cpu_exchange_records(output,run,dep,permit,native_identity,bundle):
    output=Path(output);exit_ref=runtime.reference(output/'given_act_cpu_exit.json');exit=runtime.read_bound(exit_ref)
    need(exit==run.get('cpu_child_exit') and exit.get('schema')=='m2_given_contact_cpu_child_exit_v1' and
         exit.get('started') is True and type(exit.get('reaped')) is bool and type(exit.get('normal_exit')) is bool and
         isinstance(exit.get('signals'),list) and isinstance(exit.get('errors'),list) and exit.get('shared_native_group_signaled') is False and
         deployment._same_live_process(exit.get('parent_identity'),native_identity), 'Actual learned CPU child exit/parent identity binding required')
    child=exit['identity'];need(child['pid']!=native_identity['pid'] and
         child['pgid']==child['sid']==native_identity['pgid']==native_identity['sid'], 'CPU model child left its original native session/group')
    trace=runtime.reference(output/'given_act_ipc.jsonl');stderr=runtime.reference(output/'given_act_cpu_stderr.log')
    need(exit.get('trace')==trace and exit.get('stderr')==stderr, 'Exact actual CPU IPC/stderr files required')
    with runtime.bound_path(trace).open() as stream:records=[deployment.wire_decode(line) for line in stream]
    need(records and records[0].get('event')=='ready' and all(row.get('event')=='inference' for row in records[1:]),
         'One ready handshake followed by actual consecutive inference exchanges required')
    ready=records[0]['value'];expected=deployment._ready_binding(dep,permit,bundle['admission']['source_inventory'],bundle['software'])
    need(set(ready)==set(expected)|{'identity','parent_identity','execution'} and all(ready.get(k)==v for k,v in expected.items()) and
         deployment._same_live_process(ready.get('identity'),child) and
         deployment._same_live_process(ready.get('parent_identity'),native_identity), 'Actual CPU ready/model/source/identity binding differs')
    runtime.validate_execution_settings(ready['execution'],device='cpu')
    need(exit.get('request_count')==len(records)-1, 'Missing/extra actual CPU inference exchange')
    policy=run.get('learned_policy_receipt',{})
    need(policy.get('model')==permit['model'] and policy.get('admission')==permit['admission'] and policy.get('permit')==dep['permit'] and
         policy.get('metadata')==dep['metadata'] and policy.get('child_runtime')==ready and policy.get('expert_l2_fallback') is False,
         'Original learned controller/model/runtime receipt changed')
    normal=exit['reaped'] and exit['normal_exit'] and exit.get('exit_code')==0 and exit['signals']==exit['errors']==[]
    return records,dict(exit=exit_ref,trace=trace,stderr=stderr,identity=child,normal_exit=bool(normal))


def _cpu_evidence(output,run,dep,permit,native_identity,bundle):
    records,evidence=_cpu_exchange_records(output,run,dep,permit,native_identity,bundle)
    need(evidence['normal_exit'], 'Actual learned CPU child normal exit/reap required')
    return records,evidence


def _native_scene_evidence(output,config,run):
    from . import scene_store
    from .storage_policy import episode_size
    output=Path(output);binding,_=scene_store.validate_config(config)
    ref=runtime.reference(output/'scene_ref.json');scene=runtime.read_bound(ref);scene_store.resolve_scene(scene)
    need(scene.get('binding')==binding, 'Actual scene reference differs from frozen paired configuration')
    ledger=run.get('scene_storage',{});measured=episode_size(output)
    need(ledger.get('schema')==scene_store.LEDGER_SCHEMA and ledger.get('cas_unchanged') is True and
         ledger.get('source_bindings_unchanged') is True and ledger.get('within_budget') is True and ledger.get('limit_bytes')==100_000_000 and
         ledger.get('job_directory_bytes')==measured['total_bytes'] and measured['within_budget'] is True,
         'Original exact whole-job/CAS/source 100 MB ledger required')
    return dict(scene_ref=ref,whole_job_bytes=measured['total_bytes'])


def qualify_bridge_smokes(permit_ref,owner_terminals,*,output):
    """Actual four-episode CPU reconstruction, never a synthetic qualification."""
    permit=deployment.verify_permit(permit_ref);bundle=deployment._admission_metadata(permit['admission'])
    need(permit['schema']==deployment.PERMIT_SCHEMA, 'TRAIN/VAL smoke permit required; TEST native capability cannot qualify itself')
    need(isinstance(owner_terminals,list) and len(owner_terminals)==len({r['owner_run_id'] for r in permit['attempts']}),
         'Exact completed original smoke owner terminals required')
    owners={runtime.read_bound(ref)['run_id']:ref for ref in owner_terminals}
    need(set(owners)=={r['owner_run_id'] for r in permit['attempts']}, 'Missing/extra actual smoke owner')
    # All original owner/config identities authenticate before opening traces.
    arms=[_native_owner(row,owners[row['owner_run_id']],permit_ref) for row in permit['attempts']]
    need(len({(r['identity']['pid'],r['identity']['start_ticks']) for r in arms})==4, 'Actual smoke resets must use four distinct native children')
    sources=_sources(permit['processing']);torch,policy,checkpoint,complete=runtime.load_checkpoint(permit['model'],device='cpu')
    need(checkpoint['normalization']==bundle['normalization'] and complete==bundle['complete'], 'Independent official CPU model differs from admission')
    runtime.validate_execution_settings(runtime.execution_settings(torch,policy,device='cpu'),device='cpu')
    for row,evidence in zip(permit['attempts'],arms):
        run=runtime.read_bound(evidence['complete']);config=runtime.read_bound(evidence['config']);meta=row['metadata'];folder=Path(row['output'])
        need(run.get('success') is True and run.get('physical_validation') is True and run.get('controller_motion_complete') is True and
             run.get('errors')==[] and run.get('learned_policy') is True and run.get('expert_demonstration') is False and
             run.get('executed_controller_class')==deployment.MODULE+'.FormalGivenACT' and
             run.get('handoff_policy_id')==runtime.handoff.POLICY_ID and run.get('action_semantics_id')==runtime.given.ACTION_SEMANTICS_ID,
             'Every actual smoke must complete learned L2 and original full physical gates')
        need(all(run.get('runtime_sources_sha256',{}).get(ref['path'])==ref['sha256'] for ref in sources.values()),
             'Native runtime source inventory must explicitly bind every formal/evaluation helper')
        receipt=run.get('neutral_native_audit');need(receipt and receipt.get('source_inventory')==sources,
             'Actual neutral native audit source inventory missing or changed')
        events=read_native_trace(receipt,meta);need(len(events)==run.get('physics_steps'), 'Actual original native tick count differs')
        evidence['physical_audit']=_physical_native(run,events,meta,native_config=config)
        evidence['scene']=_native_scene_evidence(folder,config,run)
        ipc,evidence['cpu_child']=_cpu_evidence(folder,run,config['m2_given_contact_deployment'],permit,evidence['identity'],bundle)
        predictor=runtime.ChunkPredictor(torch,policy,checkpoint,meta,device='cpu')
        evidence['act_reconstruction']=_audit_act_execution(events,ipc,run,config['m2_given_contact_deployment'],permit,predictor)
        evidence['video']=_video_evidence(folder,run,events)
    inventories=[_tree(root) for root in sorted({str(Path(row['output']).parent.parent) for row in permit['attempts']})]
    value=dict(schema=SMOKE_SCHEMA,passed=True,cpu_fixture_only=False,test_payloads_opened=False,protocol=evaluation_protocol(),
        admission=permit['admission'],model=permit['model'],processing=permit['processing'],source_inventory=sources,
        permit=deepcopy(permit_ref),arms=arms,owner_inventories=inventories)
    return runtime.write_new(output,value)


def _arm_status(proof,*,owner_normal):
    physical_failure=proof['actual_physics_steps'] not in (None,0) and not proof['physical_passed']
    lifecycle_failure=proof['launched'] and not (owner_normal and proof['child_normal'])
    verified=all(proof[key] for key in ('source_verified','trace_verified','act_verified','video_verified')) and not proof['audit_errors']
    if not proof['launched']:status='planned_unlaunched'
    elif proof['complete'] is None:status='missing_result'
    elif lifecycle_failure or not verified:status='infrastructure_abort'
    elif proof['physical_passed']:status='passed'
    elif physical_failure:status='physical_failure'
    else:status='infrastructure_abort'
    return dict(status=status,reason_flags=dict(physical_failure=physical_failure,lifecycle_failure=lifecycle_failure,
        evidence_failure=not verified,missing_result=proof['launched'] and proof['complete'] is None))


def _measured_metrics(run,events):
    """Unfiltered actual rows; reported handoff flags never replace hard audits."""
    def derivatives(values):
        value=np.asarray(values,dtype=float);result={}
        for label in ('velocity_rad_s','acceleration_rad_s2','jerk_rad_s3'):
            value=np.diff(value,axis=0)*240.
            result['max_'+label]=float(np.max(np.abs(value))) if len(value) else None
        return result
    phases={};owners={};extrema={};hold=release=longest_hold=longest_release=0
    for event in events:
        labels=event['controller_after'];phase=labels['phase'];owner=event['actual_control_owner'];measurements=event['measurements']
        phases[phase]=phases.get(phase,0)+1;owners[owner]=owners.get(owner,0)+1
        for key,value in measurements.items():
            if type(value) not in (float,int) or not np.isfinite(value):continue
            row=extrema.setdefault(key,dict(min=float(value),max=float(value)))
            row['min']=min(row['min'],float(value));row['max']=max(row['max'],float(value))
        contact=measurements.get('diaphragm_N',float('inf'))+measurements.get('retainer_N',float('inf'))
        hold=hold+1 if phase=='HOLD' and 3.5<=contact<=4.5 else 0;longest_hold=max(longest_hold,hold)
        good=labels['subphase']=='normal_release_confirm' and contact<.2 and measurements.get('hall_N',float('inf'))<.2
        release=release+1 if good else 0;longest_release=max(longest_release,release)
    result=dict(actual_physics_steps=len(events),actual_simulation_s=len(events)/240.,actual_phase_ticks=phases,
        actual_owner_ticks=owners,measurement_extrema=extrema,longest_force_band_hold_s=longest_hold/240.,
        longest_below_force_normal_release_s=longest_release/240.,reported_l2_handoffs=deepcopy(run.get('l2_handoffs',[])),
        reported_runtime_cost=deepcopy(run.get('runtime_cost')),raw_target_setters=len(run.get('raw_target_setters',[])),
        original_errors=deepcopy(run.get('errors',[])),failure_reason=run.get('failure_reason'))
    if events:
        result['command_fd']=derivatives([events[0]['observed_before']['q']]+[e['q_command'] for e in events])
        result['observed_fd']=derivatives([events[0]['observed_before']['q']]+[e['observed_after']['q'] for e in events])
    return result


def _command_gates(events):
    """Replay the existing _block and SkillExpert whole-stream checks exactly."""
    from sim.auscultation_skill_v1.executor import command_derivatives
    commands=[];peaks=dict(velocity=0.,acceleration=0.,jerk=0.)
    for event in events:
        labels=event['controller_before'];previous=runtime.shared._vector(labels.get('previous_command_q'),(7,),'prior actual issued command')
        command=runtime.shared._vector(event['q_command'],(7,),'actual original command')
        need(type(labels.get('contact_active')) is bool and (not commands or np.array_equal(previous,commands[-1])) and
             np.max(np.abs(command-previous))<=.175/240.+1e-9, 'Original _block command history/speed gate failed')
        if len(commands)>=3:
            values=command_derivatives(commands[-3:],command)
            limits=dict(velocity=.175,acceleration=.5 if labels['contact_active'] else 1.,jerk=8.)
            need(all(values[key]<=limits[key]+1e-8 for key in limits), 'Original whole-stream derivative gate failed')
            for key,value in values.items():peaks[key]=max(peaks[key],value)
        commands.append(command)
    return dict(checked_ticks=len(events),whole_stream_peaks=peaks,
        original_free_acceleration_rad_s2=1.,original_contact_acceleration_rad_s2=.5,original_whole_stream_jerk_rad_s3=8.)


def _references(value):
    if isinstance(value,dict):
        if ('path' in value or 'file' in value) and 'sha256' in value:
            yield _file({key:v for key,v in value.items() if key in ('path','file','bytes','sha256')})
        else:
            for item in value.values():yield from _references(item)
    elif isinstance(value,list):
        for item in value:yield from _references(item)


def _storage_union(trees,refs,*,observed_jobs,provisioned_paths=()):
    """Exact path charging; inode uniqueness is a separate physical report."""
    need(type(observed_jobs) is int and observed_jobs>=0, 'Actual observed native job denominator required')
    files={};physical={};provisioned=set(provisioned_paths)
    for inventory in trees:
        need(_tree(inventory['root'])==inventory, 'Accounted closed tree changed')
        refs=[*refs,*inventory['files'].values()]
    for ref in refs:
        path=runtime.bound_path(ref);mode=path.stat();key=str(path)
        need(stat.S_ISREG(mode.st_mode) and (mode.st_nlink==1 or key in provisioned),
             'Accounting input must be one regular file or an authenticated provisioned runtime file')
        current=dict(path=key,bytes=mode.st_size,sha256=ref['sha256'])
        need(key not in files or files[key]==current, 'Conflicting path accounting bindings')
        files[key]=current;physical[(mode.st_dev,mode.st_ino)]=mode.st_size
    total=sum(row['bytes'] for row in files.values())
    return dict(charged_path_bytes=total,charged_files=len(files),physical_inode_bytes=sum(physical.values()),
        observed_jobs=observed_jobs,bytes_per_observed_job=total/observed_jobs if observed_jobs else None,
        inventory=[files[key] for key in sorted(files)],planned_pairs_are_not_cost_denominator=True)


def _summary_storage(cap_ref,cap,body,pair,execution,bundle,*,output):
    from . import scene_store
    from .storage_policy import episode_size
    prepared=runtime.read_bound(cap['prepared_inventory']);smokes=runtime.read_bound(body['bridge_smokes'])
    roots=[body['layout']['root'],str(Path(body['layout']['planning_owner']['output']).parent),
        str(Path(body['layout']['preparation_owner']['output']).parent)]
    if execution['root'] is not None:roots.append(execution['root'])
    roots.extend(row['root'] for row in smokes['owner_inventories'])
    roots=sorted(set(roots));destination=runtime.dataset._path(str(output),Path.cwd())
    need(all(Path(root)!=destination and Path(root) not in destination.parents for root in roots),
         'Summary output cannot mutate any frozen campaign/prerequisite input tree')
    trees=[_tree(root) for root in roots];descriptor=runtime.read_bound(cap['processing']['processing_freeze'])
    refs=[cap_ref,*_references(cap),*_references(body),*_references(pair),*_references(prepared),*_references(descriptor),
        *_references(bundle['admission']),*_references(bundle['complete'])]
    provisioned=set();ledger=None;ledger_error=None;smoke_ledgers=[]
    native_roots=[Path(row['attempt']['output']) for row in execution['attempts'] if row['launched'] and Path(row['attempt']['output']).is_dir()]
    if native_roots:
        try:ledger=scene_store.batch_ledger(native_roots,batch_root=execution['root'])
        except (ValueError,FileNotFoundError) as error:ledger_error=str(error)
    permit=deployment.verify_permit(smokes['permit'])
    for inventory in smokes['owner_inventories']:
        # Original smoke roots are authenticated above; exact actual job paths
        # come from the original permit, never a favorable directory scan.
        jobs=[Path(row['output']) for row in permit['attempts'] if str(Path(row['output']).parent.parent)==inventory['root']]
        report=scene_store.batch_ledger(jobs,batch_root=inventory['root']);smoke_ledgers.append(report)
        refs.extend(_references(report['shared_inputs']))
        provisioned.update(row['file'] for row in report['shared_inputs'] if any(scene_store._sdk_file_name(k) for k in row['kinds']))
    if ledger is not None:refs.extend(_references(ledger['shared_inputs']))
    for header in execution['attempts']:
        config=runtime.read_bound(header['config']);binding=config['compact_scene_storage']
        scene_store._sdk_bundle(binding)
        for name,ref in binding['sources'].items():
            scene_store.read_source(ref,allow_empty=scene_store._sdk_file_name(name))
            normalized=_file(ref);refs.append(normalized)
            if scene_store._sdk_file_name(name):provisioned.add(normalized['path'])
        path=Path(header['attempt']['output'])/'scene_ref.json'
        if path.exists():
            scene=runtime.read_bound(runtime.reference(path));cas=scene_store.resolve_scene(scene)
            refs.append(dict(path=str(cas),sha256=scene['scene']['sha256'],bytes=scene['scene']['bytes']))
    union=_storage_union(trees,refs,observed_jobs=len(native_roots),provisioned_paths=provisioned)
    software_refs=list(_references(bundle['software']))
    software=_storage_union([],software_refs,observed_jobs=0,provisioned_paths=[ref['path'] for ref in software_refs])
    whole_jobs=[dict(job_id=row['attempt']['job_id'],**episode_size(row['attempt']['output'])) for row in execution['attempts']
        if row['launched'] and Path(row['attempt']['output']).is_dir()]
    launched=sum(row['launched'] for row in execution['attempts'])
    native_complete=(not launched or (len(native_roots)==launched and ledger is not None and
        ledger.get('total_accounted_bytes_complete') is True))
    scope_complete=native_complete and all(report.get('total_accounted_bytes_complete') is True for report in smoke_ledgers)
    return dict(scope='All heldout preparation/planning/native trees and directly required frozen model/source/smoke artifacts; upstream dataset accounting separately referenced',
        input_cost_scope_complete=scope_complete,**union,whole_jobs=whole_jobs,original_native_batch_ledger=ledger,
        original_native_batch_ledger_error=ledger_error,original_smoke_batch_ledgers=smoke_ledgers,
        all_whole_jobs_within_100MB=all(row['within_budget'] for row in whole_jobs),
        original_native_shared_budget_passed=ledger is not None and ledger['all_accounted_jobs_within_budget'],
        required_input_trees=trees,provisioned_runtime_software=software,
        upstream_population=deepcopy(bundle['config']['dataset']),upstream_training_owner=deepcopy(bundle['owner']),
        whole_project_within_100MB_per_episode_claim=False)


def _inspect_arm(header,cap,cap_ref,bundle,model):
    row=header['attempt'];folder=Path(row['output']);proof=dict(launched=header['launched'],complete=header['complete'],
        child_normal=False,actual_physics_steps=0 if not header['launched'] else None,physical_passed=False,
        source_verified=False,trace_verified=False,act_verified=row['arm']=='expert',video_verified=False,audit_errors=[])
    result=dict(subject_id=row['subject_id'],arm=row['arm'],job_id=row['job_id'],launched=header['launched'],
        report=None,video=None,metrics=None,physical_audit=None,act_reconstruction=None,headers=header)
    if not header['launched']:return result,proof
    report_ref=header['complete'];partial=folder/'fresh_expert_result.json'
    if report_ref is None and partial.is_file():report_ref=runtime.reference(partial)
    if report_ref is None:return result,proof
    run=runtime.read_bound(report_ref);config=runtime.read_bound(header['config']);result['report']=report_ref
    def check(stage,operation):
        try:return operation()
        except Exception as error:proof['audit_errors'].append(dict(stage=stage,error=type(error).__name__+': '+str(error)));return None
    def sources():
        tag=run.get('heldout_evaluation',{})
        need(tag.get('capability')==cap_ref and tag.get('episode_id')==row['episode_id'] and tag.get('subject_id')==row['subject_id'] and
             tag.get('arm')==row['arm'] and tag.get('capability_authenticated') is True and tag.get('model')==cap['model'] and
             tag.get('protocol')==evaluation_protocol() and tag.get('simulation_only') is True and
             all(run.get('runtime_sources_sha256',{}).get(ref['path'])==ref['sha256'] for ref in cap['source_inventory'].values()),
             'Actual arm/capability/model/source evidence differs')
        expected=deployment.MODULE+'.FormalGivenACT' if row['arm']=='act' else 'sim.ausc_policy.m2_given_contact_controller.GivenContactExpert'
        need(run.get('executed_controller_class')==expected and run.get('learned_policy') is (row['arm']=='act') and
             run.get('handoff_policy_id')==runtime.handoff.POLICY_ID and run.get('action_semantics_id')==runtime.given.ACTION_SEMANTICS_ID and
             _reference_availability(run.get('given_reference'),row['metadata']), 'Genuine expert/ACT identity or original prescribed reference differs')
        need(runtime.read_bound(runtime.reference(folder/'fresh_expert_inputs/given_reference_before_motion.json'))==run['given_reference'],
             'Actual before-motion provider envelope differs from the final immutable reference')
        return True
    proof['source_verified']=check('native_sources_identity',sources) is True
    receipt=run.get('neutral_native_audit');events=check('actual_native_trace',lambda:_read_native_events(receipt,row['metadata']))
    if events is not None:
        proof['trace_verified']=bool(receipt.get('source_inventory')==cap['source_inventory'] and len(events)==run.get('physics_steps'))
        if not proof['trace_verified']:proof['audit_errors'].append(dict(stage='actual_native_trace',error='Original tick count/source differs'))
        else:proof['actual_physics_steps']=len(events)
        result['metrics']=_measured_metrics(run,events)
        if run.get('success') is True and run.get('physical_validation') is True:
            result['physical_audit']=check('full_original_physical_gates',lambda:_physical_native(run,events,row['metadata'],native_config=config))
            proof['physical_passed']=result['physical_audit'] is not None
            result['scene']=check('original_whole_job_scene_storage',lambda:_native_scene_evidence(folder,config,run))
        else:result['physical_audit']=dict(passed=False,original_errors=deepcopy(run.get('errors',[])),failure_reason=run.get('failure_reason'))
        video=check('original_actual_video',lambda:_video_evidence(folder,run,events)) if events else None
        result['video']=video;proof['video_verified']=video is not None
        if row['arm']=='act' and events:
            def act():
                ipc,cpu=_cpu_exchange_records(folder,run,config['m2_given_contact_deployment'],cap,header['identity'],bundle)
                proof['cpu_normal']=cpu['normal_exit']
                if not model:
                    torch,policy,checkpoint,complete=runtime.load_checkpoint(cap['model'],device='cpu')
                    need(complete==bundle['complete'] and checkpoint['normalization']==bundle['normalization'], 'Selected independent CPU model changed')
                    runtime.validate_execution_settings(runtime.execution_settings(torch,policy,device='cpu'),device='cpu')
                    model.update(torch=torch,policy=policy,checkpoint=checkpoint)
                predictor=runtime.ChunkPredictor(model['torch'],model['policy'],model['checkpoint'],row['metadata'],device='cpu')
                return dict(_audit_act_execution(events,ipc,run,config['m2_given_contact_deployment'],cap,predictor),cpu=cpu)
            result['act_reconstruction']=check('every_actual_act_inference',act)
            proof['act_verified']=result['act_reconstruction'] is not None
    child=runtime.read_bound(header['child_result']) if header['child_result'] is not None else {}
    shutdown=run.get('runtime_shutdown',{})
    proof['child_normal']=(child.get('exit_code')==0 and child.get('worker_complete') is True and shutdown.get('app_close_returned') is True and
        shutdown.get('fast_shutdown') is False and shutdown.get('interpreter_exit_policy')=='success_only_os_exit_after_native_close_and_fsynced_receipt' and
        (row['arm']=='expert' or proof.get('cpu_normal') is True))
    return result,proof


def _reference_availability(value,metadata):
    ref=runtime.pose.require_route(metadata)
    expected=dict(reference=ref.snapshot(),reference_sha256=ref.sha256,
        availability_binding=dict(bound_at_s=0.,physics_row=0,reference_sha256=ref.sha256),
        task_id=runtime.pose.TASK_ID,state_only=True,visual_or_sensor_robustness_claim=False)
    need(ref.snapshot()['available_at_s']==0. and value==expected,
         'Original same full reference must be bound before the first actual motion')
    return True


def summarize(capability,owner_terminal,*,output):
    """One complete CPU audit; failed and absent arms retain their fixed slots."""
    cap=verify_capability(capability,stage='native');pair=verify_pair_plan(cap['pair_plan']);body=verify_body_plan(cap['body_plan'])
    bundle=deployment._admission_metadata(cap['admission']);runtime.validate_software(bundle['software'],runtime.software_inventory(runtime.VENDOR_ROOT))
    execution=_execution_inventory(cap,owner_terminal,capability)
    before=_summary_storage(capability,cap,body,pair,execution,bundle,output=output)
    model={};outcomes=[];details=[];videos=[]
    for header in execution['attempts']:
        detail,proof=_inspect_arm(header,cap,capability,bundle,model);classification=_arm_status(proof,owner_normal=execution['owner_normal'])
        outcomes.append(dict(subject_id=detail['subject_id'],arm=detail['arm'],job_id=detail['job_id'],launched=proof['launched'],
            physics_steps=proof['actual_physics_steps'],status=classification['status']))
        detail.update(classification,proof=proof);details.append(detail)
        videos.append(dict(subject_id=detail['subject_id'],arm=detail['arm'],video=detail['video'],status=classification['status']))
    counts=reported_pair_counts(pair['schedule'],outcomes)
    verified=all(not detail['proof']['audit_errors'] for detail in details) and before.get('input_cost_scope_complete') is True
    counts['native_evidence_verified']=verified
    milestone=counts['executed_pairs']>=30 and verified
    after=_summary_storage(capability,cap,body,pair,execution,bundle,output=output)
    need(before==after and verify_capability(capability,stage='native')==cap and
         deployment._admission_metadata(cap['admission'])==bundle, 'Frozen campaign/model/source/input bytes changed during final CPU audit')
    complete=(milestone and execution['owner_normal'] and after.get('all_whole_jobs_within_100MB') is True and
        after.get('original_native_shared_budget_passed') is True and
        all(row['status'] not in ('missing_result','planned_unlaunched','infrastructure_abort') for row in outcomes))
    counts['heldout_evaluation_complete']=complete
    value=dict(schema=SUMMARY_SCHEMA,capability=deepcopy(capability),pair_plan=cap['pair_plan'],owner_terminal=deepcopy(owner_terminal),
        protocol=evaluation_protocol(),model=cap['model'],source_inventory=cap['source_inventory'],outcomes=outcomes,arms=details,counts=counts,
        planning_coverage=pair['planning_inventory'],pose_coverage={pose:sum(row['pose_template']==pose for row in body['layout']['bodies']) for pose in deployment.POSE_TEMPLATES},
        review_videos=select_review_videos(videos,body_ids=[row['subject_id'] for row in body['layout']['bodies']]),
        storage=after,milestone_reached=milestone,heldout_evaluation_complete=complete,simulation_only=True,training_eligible=False,
        real_hardware_validated=False,test_is_no_longer_untouched=any(row['launched'] for row in outcomes),
        scope='Original34 identities; two known poses; bounded all-ten planning; one prescribed first-feasible target and at most68 arms; no universal/full-ten physical generalization claim')
    return runtime.write_new(output,value)


def tag_arm_result(result,config,binding):
    route=config[ROUTE_KEY];row=None if binding is None else binding['row']
    result['heldout_evaluation']=dict(schema='m2_given_contact_heldout_arm_result_v1',
        capability=deepcopy(route.get('capability')),episode_id=route.get('episode_id'),arm=route.get('arm'),
        capability_authenticated=binding is not None,model=None if binding is None else binding['capability']['model'],
        subject_id=None if row is None else row['subject_id'],simulation_only=True,protocol=evaluation_protocol())
    result.update(training_eligible=False,training_accepted=False,raw_capture_complete=False,compact_capture_complete=False,
        raw_multimodal_training_capture=False,simulation_only=True,real_hardware_validated=False,
        scope='Fixed 34 heldout identities; two known poses; one prescribed first-feasible target per body; paired expert/ACT simulation')
    return result


def run_native_probe(app,stage,output,config,mass):
    from . import fresh_expert
    binding=None;lifecycle={}
    try:
        binding=preflight(config,output=output)
        need(binding['stage']=='native', 'Preparation capability cannot execute a native arm')
        if binding['row']['arm']=='act':result=deployment.run_native_probe(app,stage,output,config,mass)
        else:result=fresh_expert._run_probe(app,stage,output,config,mass,_learned_lifecycle=lifecycle)
        return tag_arm_result(result,config,binding)
    except BaseException as error:
        env=lifecycle.get('env');result=dict(success=False,physical_validation=False,points=[],
            physics_steps=0 if env is None else len(env.rows),video=lifecycle.get('video'),
            failure_reason=str(error),errors=[dict(stage='heldout_native_preflight_or_finalization',error=repr(error))])
        tag_arm_result(result,config,binding)
        try:runtime.write_new(Path(output)/'fresh_expert_result.json',result)
        except Exception as saving:result['errors'].append(dict(stage='failure_result_save',error=repr(saving)))
        if not isinstance(error,Exception):raise
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    operations=('body-plan','preparation-plan','issue-preparation','prepared-inventory','planning-configurations',
        'pair-plan','issue-native','qualify-smokes','summary')
    parser.add_argument('--operation',choices=operations)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();ref=runtime.reference(args.config);request=runtime.read_bound(ref)
    if args.operation is None:return run_planning_aggregate(ref,args.output)
    fields={'body-plan':{'admission','smokes','seed','patient_specs','root'},
        'preparation-plan':{'body_plan','configurations'},'issue-preparation':{'preparation_plan'},
        'prepared-inventory':{'preparation_capability'},'planning-configurations':{'prepared_inventory'},
        'pair-plan':{'prepared_inventory','planning_results','owner_run_id'},'issue-native':{'pair_plan'},
        'qualify-smokes':{'permit','owner_terminals'},'summary':{'capability','owner_terminal'}}
    need(isinstance(request,dict) and set(request)==fields[args.operation], 'Exact finite operation request fields required')
    if args.operation=='body-plan':return freeze_body_plan(**request,output=args.output)
    if args.operation=='preparation-plan':return freeze_preparation_plan(request['body_plan'],request['configurations'],output=args.output)
    if args.operation=='issue-preparation':return issue_preparation_capability(request['preparation_plan'],output=args.output)
    if args.operation=='prepared-inventory':return freeze_prepared_inventory(request['preparation_capability'],output=args.output)
    if args.operation=='planning-configurations':
        inventory=verify_prepared_inventory(request['prepared_inventory'],payloads=False)
        cap=verify_capability(inventory['preparation_capability'],stage='preparation');body=verify_body_plan(cap['body_plan'])
        need(str(args.output)==body['layout']['planning_owner']['config'], 'Only the predeclared planning owner configuration may be written')
        return freeze_planning_configurations(request['prepared_inventory'])
    if args.operation=='pair-plan':return freeze_pair_plan(**request,output=args.output)
    if args.operation=='issue-native':return issue_native_capability(request['pair_plan'],output=args.output)
    if args.operation=='qualify-smokes':return qualify_bridge_smokes(request['permit'],request['owner_terminals'],output=args.output)
    return summarize(request['capability'],request['owner_terminal'],output=args.output)


if __name__=='__main__':main()
