"""Fail-closed Task5A reconstruction for the prescribed-contact raw-intent task.

Only the declared ``isaac_batch`` layout is accepted. Rows bind ``run_id``,
``job_id``, globally unique ``reset_id``, ``config``, ``owner_terminal``,
``child_result``, ``run_result`` (the child's COMPLETE), ``source_freeze``,
``capture_root`` and an exhaustive ``artifacts`` inventory. JSON references
use {path,sha256}; archived source entries also carry bytes. All declared
membership and compact metadata are checked before ANY capture payload read.

Source freezes bind CODE_COMMIT, SOURCE_MANIFEST and the exact union of run
and capture sources to immutable archived files. They are distinct from the
current reconstruction code bindings: adding a future checkpoint loader must
not silently redefine capture semantics or invalidate an archived source file.
No schema auto-detection, old-data promotion, TEST access, simulator or fitting.
CPU fixtures and hashes alone do not establish actual native qualification.

Population receipts bind ``manifest={path,sha256}``, ``manifest_sha256``, the
new task/action IDs, fixed minimum cuts, and separate hash-bound attempt_plan,
execution_inventory and attempt_ledger documents. The plan declares frozen_utc,
run_root, reviewed allocation_proposal/allocation_mapping references, and all
ordered attempts (attempt/episode/reset/run/job/slot/target/subject/split IDs,
config and source_freeze references, preclassified_reachable=true). The finalized
inventory covers every planned ID; the ledger additionally preserves those
identities, native_success and the independently checked outcome. Launch files
are authoritative even when a child result is missing. Bind partial_result to
the original fresh_expert_result.json when COMPLETE is absent. Primary outcomes
are mutually exclusive; lifecycle/safety reason counts explicitly overlap.

The prelaunch accounting_scope declares owner_roots, extra_input_refs entries
{category,reference={path,bytes,sha256}}, tier_b_attempt_ids and storage_roots
{category,path}. Extra categories are source/config/sdk/prepared/other_runtime;
separate storage categories are complete_cpu_bank, backup_archive and
provisioned_runtime. Per-owner ledgers and one unique population union use the
existing scene_store ledger unchanged. Statistics-only storage inventories do
not open assets or make a whole-project per-episode storage claim.

CLI: ``python -m sim.ausc_policy.m2_given_contact_dataset --manifest FILE
--sha256 SHA --output NEW_DIRECTORY [--qualification FILE
--qualification-sha256 SHA]``. It reconstructs twice and emits a new five-file
bundle referencing the original immutable manifest. Arrays are regenerated;
the CLI never edits episode results or copies raw archives/checkpoints.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import tarfile

import numpy as np

from . import m2_act as shared, m2_given_contact_act as given
from . import m2_given_contact_pose as pose, retainer_witness as witness
from . import geometry_epoch as geometry
from . import m2_given_contact_handoff as handoff
from .training_capture import GIVEN_ADAPTIVE_SCHEMA, iter_training_records

MANIFEST_SCHEMA = 'm2_given_contact_act_manifest_v2'
INDEX_SCHEMA = 'm2_given_contact_causal_dataset_index_v2'
SOURCE_FREEZE_SCHEMA = 'm2_given_contact_source_freeze_v1'
POPULATION_SCHEMA = 'm2_given_contact_expert_population_qualification_v2'
PLAN_SCHEMA = 'm2_given_contact_expert_attempt_plan_v2'
EXECUTION_SCHEMA = 'm2_given_contact_expert_execution_inventory_v2'
LEDGER_SCHEMA = 'm2_given_contact_expert_attempt_ledger_v2'
ACCOUNTING_SCHEMA = 'm2_given_operational_metadata_envelope_v1'
METADATA_ENVELOPE_BYTES = 2147483648
SEAL_ENVELOPE_BYTES = 1048576
PROCESSING_ENVELOPE_BYTES = 67108864
FINAL_PROCESSING_ROOT = '/opt/wam-runtime/code/m2-given-formal-processing-v1'
OFFICIAL_ACT_PREFIX = '/opt/wam-runtime/vendor/lerobot-eb7a54c/.venv'
FINAL_SEAL_SCHEMA = 'm2_given_formal_collection_final_seal_v1'
ALLOCATION_PROPOSAL_SHA256 = '00cb65e507cfae5892db762146c7908a63a874cea96afaa9c573ac6aa96a6c86'
ALLOCATION_MAPPING_SHA256 = '9162e5a546c97a4dceeba6d7d36ce5c50a74f7c9efabe863364643aa5ceb3156'
need = shared.need
sha256 = shared.sha256


def _path(value, base):
    need(isinstance(value, str) and value and '..' not in Path(value).parts, 'Explicit path without traversal required')
    path = Path(base)/value
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink in evidence path')
    return path.resolve()


def _ref_path(ref, base):
    need(isinstance(ref, dict) and set(ref) <= {'path','sha256','bytes'} and
         {'path','sha256'} <= set(ref) and isinstance(ref['sha256'],str) and
         re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None, 'Explicit hash-bound artifact reference required')
    return _path(ref.get('path'),base)


def _bound(ref, base):
    path = _ref_path(ref,base)
    need(path.is_file() and ('bytes' not in ref or
         type(ref['bytes']) is int and path.stat().st_size == ref['bytes']) and sha256(path)==ref['sha256'],
         'Bound artifact hash/bytes changed: '+str(path))
    return shared._json(path),path


def _declarations(rows, base):
    need(isinstance(rows,list) and 0<len(rows)<=240, 'Explicit finite episode declarations (at most 240) required')
    seen = {k:set() for k in ('episode_id','reset_id','capture_root','run_result','job')}
    for row in rows:
        need(isinstance(row,dict), 'Episode declaration must be an object')
        shared.require_membership(row.get('split'),row.get('subject_id'))
        for key in ('episode_id','reset_id','run_id','job_id'):
            need(isinstance(row.get(key),str) and bool(row[key]), 'Missing explicit '+key)
        tokens = dict(episode_id=row['episode_id'],reset_id=row['reset_id'],
            capture_root=str(_path(row.get('capture_root'),base)),
            run_result=str(_ref_path(row.get('run_result'),base)),
            job=(row['run_id'],row['job_id']))
        for key,value in tokens.items():
            need(value not in seen[key], 'Duplicate episode/reset/run receipt/source identity: '+key)
            seen[key].add(value)
        # Reusing the same bytes with a new pathname is still source reuse.
        digest=row['run_result']['sha256']
        need(digest not in seen.setdefault('run_hash',set()), 'Duplicate run receipt hash')
        seen['run_hash'].add(digest)


def _metadata(row, base):
    shared.require_membership(row.get('split'),row.get('subject_id'))
    root = _path(row.get('capture_root'),base)
    expected = row.get('artifacts',{}).get('compact/manifest.json')
    need(isinstance(expected,dict), 'Hash-bound compact metadata required')
    manifest,path = _bound(dict(expected,path=str(root/'compact/manifest.json')),base)
    meta=manifest.get('metadata',{})
    shared.require_membership(meta.get('split'),meta.get('subject_id'))
    for key in ('episode_id','subject_id','split'):
        need(meta.get(key)==row.get(key), 'Explicit compact episode declaration differs: '+key)
    ref=pose.require_route(meta)
    need(meta.get('handoff_policy_id')==handoff.POLICY_ID, 'Formal given dataset requires the interior handoff policy')
    shared.contract.require_metadata(meta)
    need(meta.get('source_kind')=='isaac_live_physics' and meta.get('replay') is False,
         'Live Isaac acquisition required; CPU fixtures/replays cannot qualify')
    need(manifest.get('schema')==GIVEN_ADAPTIVE_SCHEMA and
         manifest.get('action_semantics_id')==given.ACTION_SEMANTICS_ID and
         manifest.get('observation_contract')==pose.observation_contract(),
         'Explicit new given raw-intent compact schema/observation contract required')
    need(ref.snapshot()['available_at_s'] <= 0., 'Given reference unavailable before first native motion')
    config,_=_bound(row.get('config'),base)
    configured=config.get('m2_given_contact_control',{}).get('metadata',{})
    shared.require_membership(configured.get('split'),configured.get('subject_id'))
    configured_ref=pose.require_route(configured)
    need(configured.get('handoff_policy_id')==handoff.POLICY_ID, 'Frozen config requires the interior handoff policy')
    need(all(configured.get(k)==meta.get(k) for k in ('episode_id','subject_id','split','target_id')) and
         configured_ref.sha256==ref.sha256, 'Frozen config/compact reference or identity differs')
    _reference_metadata(meta)
    return root,manifest,meta


def _source_freeze(ref, base, expected=None):
    freeze,path=_bound(ref,base)
    need(freeze.get('schema')==SOURCE_FREEZE_SCHEMA and
         isinstance(freeze.get('code_commit'),str) and re.fullmatch('[0-9a-f]{40}',freeze['code_commit']) is not None,
         'Trusted given-task source freeze required')
    commit_path=_ref_path(freeze.get('code_commit_file'),path.parent)
    need(sha256(commit_path)==freeze['code_commit_file']['sha256'] and
         commit_path.name=='CODE_COMMIT' and commit_path.read_text().strip()==freeze['code_commit'],
         'Frozen CODE_COMMIT changed')
    sources,source_path=_bound(freeze.get('source_manifest'),path.parent)
    need(source_path.name=='SOURCE_MANIFEST.json' and source_path.parent==commit_path.parent and
         isinstance(sources,list) and bool(sources), 'Trusted SOURCE_MANIFEST required')
    code_root=source_path.parent
    names=set()
    for row in sources:
        name=row.get('file')
        need(isinstance(name,str) and not Path(name).is_absolute() and name not in names,
             'Duplicate/invalid frozen source manifest row')
        names.add(name)
        source=_path(name,code_root)
        need(code_root in source.parents and source.is_file() and sha256(source)==row.get('sha256') and
             ('bytes' not in row or source.stat().st_size==row['bytes']), 'Frozen SOURCE_MANIFEST row hash changed: '+name)
    files=freeze.get('files')
    need(isinstance(files,dict) and bool(files), 'Exact archived source inventory required')
    if expected is not None:
        need(set(files)==set(expected), 'Frozen source inventory differs from run/capture union')
    for original,binding in files.items():
        need(isinstance(original,str) and Path(original).is_absolute() and
             type(binding.get('bytes')) is int and binding['bytes']>=0, 'Explicit archived source byte binding required')
        source=_ref_path(binding,path.parent)
        need(source.is_file() and source.stat().st_size==binding['bytes'] and sha256(source)==binding['sha256'],
             'Frozen archived source changed: '+original)
        if expected is not None:
            need(binding['sha256']==expected[original], 'Archived runtime source digest differs: '+original)
        original_path=Path(original)
        if code_root in original_path.parents:
            relative=original_path.relative_to(code_root).as_posix()
            need(relative in names and next(x['sha256'] for x in sources if x['file']==relative)==binding['sha256'],
                 'Runtime code source missing from trusted SOURCE_MANIFEST')
    return freeze


def _reference_metadata(meta):
    ref=pose.require_route(meta); ref.verify_sources()
    snapshot=ref.snapshot();sources=snapshot['sources']
    patient=pose.bound_json(sources['patient_receipt'])
    shared.require_membership(patient.get('split'),patient.get('identity'))
    need(patient.get('identity')==meta['subject_id'] and patient.get('split')==meta['split'] and
         patient.get('asset_sha256')==meta.get('patient_asset_sha256') and
         sources['patient_receipt']['sha256']==meta.get('installed_patient_receipt_sha256'),
         'Installed reference patient identity/split/asset differs from compact declaration')
    task=pose.bound_json(sources['task_manifest'])
    targets=[t for t in task.get('targets',[]) if t.get('name')==meta['target_id']]
    need(len(targets)==1, 'Original frozen task surface point/normal must be unique')
    rebuilt=pose.build_given_reference(dict(targets[0],manifest_ref=sources['task_manifest']),sources['fresh_plan'],
        sources['patient_receipt'],snapshot['world_from_base'],snapshot['available_at_s'])
    need(rebuilt.sha256==ref.sha256, 'Reference cannot reconstruct from frozen task/plan/patient sources')
    return ref


def _reference_receipt(value, meta):
    ref=_reference_metadata(meta)
    need(isinstance(value,dict) and value.get('task_id')==pose.TASK_ID and
         value.get('reference_sha256')==ref.sha256 and value.get('reference')==ref.snapshot(),
         'Native given reference receipt mismatch')
    binding=value.get('availability_binding',{})
    need(binding.get('reference_sha256')==ref.sha256 and type(binding.get('physics_row')) is int and
         binding['physics_row']==0 and type(binding.get('bound_at_s')) in (int,float) and
         np.isfinite(binding['bound_at_s']) and ref.snapshot()['available_at_s']<=binding['bound_at_s']<=0.,
         'Given reference must be actually bound before first native motion')


def _native_evidence(row, base, meta, *, physical=True):
    """One exact batch owner → job child → native COMPLETE/config chain."""
    owner,owner_path=_bound(row.get('owner_terminal'),base)
    child,child_path=_bound(row.get('child_result'),base)
    run,run_path=_bound(row.get('run_result'),base)
    config,config_path=_bound(row.get('config'),base)
    stage=owner.get('stage',{}); identity=child.get('identity',{}); group=stage.get('process_identity',{})
    need(owner.get('run_id')==row.get('run_id') and owner.get('completed') is True and
         owner.get('gpu_released') is True and isinstance(owner.get('ended_utc'),str) and
         stage.get('passed') is True and type(stage.get('exit_code')) is int and stage['exit_code']==0 and
         stage.get('process_group_drained') is True and stage.get('errors')==[] and stage.get('signals')==[],
         'Native owner must finish normally, exit 0, drain and release GPU')
    batch=owner.get('config',{}); jobs=batch.get('jobs',[])
    need(batch.get('experiment')=='isaac_batch' and type(batch.get('parallelism')) is int and
         1<=batch['parallelism']<=4 and isinstance(jobs,list), 'Explicit finite isaac_batch owner required')
    matched=[job for job in jobs if job.get('id')==row.get('job_id')]
    need(len(matched)==1 and _path(matched[0].get('config'),base)==config_path and
         matched[0].get('sha256')==row['config']['sha256'], 'Native batch job/config binding mismatch')
    output=owner_path.parent/'output'/row['job_id']
    need(child_path==output.parent/(row['job_id']+'.result.json') and run_path==output/'COMPLETE.json' and
         _path(child.get('output'),base)==output and child.get('id')==row['job_id'] and
         child.get('config_sha256')==row['config']['sha256'] and
         child.get('result_sha256')==row['run_result']['sha256'], 'Exact isaac_batch child/report layout mismatch')
    need(type(child.get('exit_code')) is int and child['exit_code']==0 and child.get('worker_complete') is True and
         (not physical or child.get('physical_pass') is True), 'Native child must complete normally with exit 0')
    need(type(child.get('pid')) is int and child['pid']==identity.get('pid') and
         type(group.get('pgid')) is int and identity.get('pgid')==identity.get('sid')==group.get('pgid')==group.get('sid') and
         group.get('members',{}).get(str(child['pid']))==identity.get('start_ticks'),
         'Native child is not bound to the owning guard process group')
    need(config.get('experiment')=='geometry_qualification' and config.get('base')==batch.get('base') and
         isinstance(config.get('isaac_python'),str) and '/bin/python' in config['isaac_python'] and
         config.get('m2_given_contact_control',{}).get('schema')==pose.TASK_ID and
         config['m2_given_contact_control'].get('metadata')==meta, 'Native given teacher configuration mismatch')
    need(run.get('task_id')==pose.TASK_ID and run.get('target_source')==pose.TARGET_SOURCE and
         run.get('action_semantics_id')==given.ACTION_SEMANTICS_ID and
         run.get('observation_contract')==pose.observation_contract() and run.get('learned_policy') is False,
         'Native report is not the new given-contact expert task')
    need(run.get('handoff_policy_id')==handoff.POLICY_ID and
         config['m2_given_contact_control'].get('handoff_policy_id')==handoff.POLICY_ID,
         'Native/config interior handoff policy missing or changed')
    need(sum(Path(name).name=='m2_given_contact_handoff.py' for name in run.get('runtime_sources_sha256',{}))==1,
         'Actual native runtime source inventory must explicitly bind interior handoff helper')
    from .m2_given_contact_controller import GIVEN_TRAJECTORY_LIMITS,GIVEN_VELOCITY_NUMERICAL_RESERVE_RAD_S,GIVEN_JERK_NUMERICAL_RESERVE_RAD_S3
    need(run.get('raw_target_interpolation_limits')==GIVEN_TRAJECTORY_LIMITS and
         run.get('planning_velocity_numerical_reserve_rad_s')==GIVEN_VELOCITY_NUMERICAL_RESERVE_RAD_S and
         run.get('planning_jerk_numerical_reserve_rad_s3')==GIVEN_JERK_NUMERICAL_RESERVE_RAD_S3,
         'Reviewed raw-intent interpolation planning reserve/limits required')
    for key in ('geometry_id','geometry_source_sha256'):
        need(run.get(key)==meta.get(key), 'Native geometry differs from compact metadata')
    if physical:
        shared._require_run_identity(run,row)
        need(run.get('success') is True and run.get('physical_validation') is True and run.get('errors')==[] and
             run.get('controller_motion_complete') is True and run.get('compact_capture_complete') is True and
             run.get('executed_controller_class')=='sim.ausc_policy.m2_given_contact_controller.GivenContactExpert',
             'Full native given expert physical/capture completion required')
    shutdown=run.get('runtime_shutdown',{})
    need(shutdown.get('app_close_returned') is True and shutdown.get('fast_shutdown') is False and
         shutdown.get('interpreter_exit_policy')=='success_only_os_exit_after_native_close_and_fsynced_receipt',
         'Normal native close and finalization receipt required')
    _reference_receipt(run.get('given_reference'),meta)
    freeze=_source_freeze(row.get('source_freeze'),base)
    need(owner.get('code_commit')==freeze['code_commit'], 'Native owner code commit differs from frozen archive')
    native_contract=meta.get('native_retainer_audit')
    need(run.get('native_retainer_audit')==native_contract and native_contract is not None,
         'Qualified native v3 contact contract required in run and capture')
    qualification=witness.load_contract(native_contract)
    need(qualification.get('cpu_diagnostic_only') is not True, 'CPU witness fixture cannot qualify native data')
    return run,config,freeze


def _decode_capture(row, base, context, run, config, freeze, *, allow_truncated=False):
    need(type(allow_truncated) is bool,'Explicit label-only truncation switch required')
    root,manifest,meta=context
    _scene_integrity(row,base,run,config,meta)
    total=shared._verify_inventory(root,row.get('artifacts'))
    outer=shared._json(root/'receipt.json'); compact=shared._json(root/'compact/receipt.json')
    capture=shared._json(root/'manifest.json')
    need(outer==run.get('raw_multimodal_capture') and outer.get('schema')==pose.CAPTURE_SCHEMA and
         capture.get('schema')==pose.CAPTURE_SCHEMA and capture.get('capture_profile')=='compact' and
         outer.get('capture_profile')=='compact' and outer.get('episode_id')==row['episode_id'] and
         outer.get('compact_capture_complete') is True and outer.get('errors')==[] and
         outer.get('total_bytes_complete') is True and type(outer.get('total_bytes')) is int and
         outer['total_bytes']==total and total<=outer.get('limit_bytes',0)<=100_000_000,
         'Exact complete given compact capture/100 MB receipt required')
    for value in (outer,capture,compact,manifest):
        need(value.get('action_semantics_id')==given.ACTION_SEMANTICS_ID, 'Capture raw target action semantics differ')
    for value in (outer,capture,manifest):
        need(value.get('observation_contract')==pose.observation_contract(), 'Capture observation contract differs')
    need(outer.get('reference_sha256')==meta['reference_sha256'] and outer.get('reference_revision')==0 and
         shared._json(root/'given_reference.json')==meta['given_reference'], 'Capture reference receipt mismatch')
    need(capture.get('binding',{}).get('compact',{}).get('metadata')==meta and
         capture.get('physics_hz')==240 and capture.get('capture_hz')==20 and
         outer.get('receipts',{}).get('compact')==compact and compact.get('schema')==GIVEN_ADAPTIVE_SCHEMA,
         'Capture binding/native clocks/inner receipt mismatch')
    _numeric_container_binding(config,capture,compact,manifest)
    sources={}
    for mapping in (run.get('runtime_sources_sha256'),capture.get('runtime_sources_sha256')):
        need(isinstance(mapping,dict) and bool(mapping), 'Run and capture runtime source inventories required')
        for path,digest in mapping.items():
            need(path not in sources or sources[path]==digest, 'Conflicting run/capture source digest')
            sources[path]=digest
        helper=[digest for path,digest in mapping.items() if Path(path).name=='m2_given_contact_handoff.py']
        need(len(helper)==1 and helper[0]==manifest.get('code_bindings',{}).get('interior_handoff'),
             'Run/capture/compact source inventory must bind the same interior handoff helper')
    _source_freeze(row['source_freeze'],base,sources)
    camera=outer.get('receipts',{}).get('camera',{})
    need(camera.get('completed') is True and camera.get('failures')==[] and
         camera.get('frames')==compact.get('frames') and camera.get('native_image_size')==capture.get('native_image_size'),
         'Actual native RGB-D frame receipt required')
    path=root/'native_audit.jsonl.gz'
    need(sha256(path)==outer.get('native_audit_sha256'), 'Native audit hash mismatch')
    with gzip.open(path,'rt') as stream: audit=[json.loads(line) for line in stream]
    need(len(audit)==run.get('physics_steps')==outer.get('physics_steps')==compact.get('final_native_step_count') and
         outer.get('expected_frames')==compact.get('frames')==len(audit)//12,
         'Native 240 Hz audit and real 20 Hz frame counts differ')
    _physical(run,audit,meta)
    records=[]
    for value in iter_training_records(root/'compact'):
        # Exhaust the fidelity reader, retaining no image/Hall encoder inputs.
        records.append({key:item for key,item in value.items() if key not in ('rgb','depth_q','depth_code','depth_valid')})
    need(len(records)==compact['frames'], 'Decoded real observation frame count differs')
    data=(given.assemble_samples(records,audit,meta,allow_truncated=True) if allow_truncated else
          given.assemble_samples(records,audit,meta))
    need(records and np.max(np.abs(records[-1]['measured_q']))<=.03 and
         np.max(np.abs(audit[-1]['q_command']))<=.03, 'Final measured/commanded native home missing')
    data['derived_index'].update(source_observation_count=len(records),source_row_range=[0,len(records)])
    data['derived_index']['source_bindings']=dict(native_audit_sha256=outer['native_audit_sha256'],
        compact_metadata_sha256=row['artifacts']['compact/manifest.json']['sha256'],
        run_result=row['run_result'],owner_terminal=row['owner_terminal'],child_result=row['child_result'],
        config=row['config'],capture_inventory=row['artifacts'],source_freeze=row['source_freeze'],
        reference_sha256=meta['reference_sha256'],action_semantics_id=given.ACTION_SEMANTICS_ID)
    receipt=dict(**{key:deepcopy(row[key]) for key in ('episode_id','subject_id','split','reset_id','run_id','job_id',
        'run_result','owner_terminal','child_result','config','source_freeze')},windows=len(data['states']),
        capture_bytes=total,reference_sha256=meta['reference_sha256'])
    child,_=_bound(row['child_result'],base)
    receipt['native_reset_identity']={k:child['identity'][k] for k in ('pid','pgid','sid','start_ticks')}
    return data,receipt,meta


def _numeric_container_binding(config,capture,receipt,manifest):
    """A decoded container must be the one selected by the frozen collector."""
    settings=config.get('fresh_expert',{}).get('raw_multimodal_capture',{}).get('compact',{})
    expected=settings.get('numeric_container_codec')
    need(expected in (None,'framed_npz_gzip_single_member_v1'), 'Unknown numeric container codec')
    need(capture.get('binding',{}).get('compact',{}).get('numeric_container_codec')==expected and
         manifest.get('numeric_container_codec')==expected and
         receipt.get('numeric_container',{}).get('codec')==expected,
         'Native numeric container differs from frozen collector configuration')


def _scene_integrity(row, base, run, config, meta):
    from . import scene_store
    from .storage_policy import episode_size
    result_path=_ref_path(row['run_result'],base);job=result_path.parent
    need(_path(row['capture_root'],base)==job/'compact_multimodal', 'Capture root differs from exact native job layout')
    binding,_=scene_store.validate_config(config)
    ref=shared._json(job/'scene_ref.json')
    need(ref.get('binding')==binding, 'Actual native scene reference differs from frozen configuration')
    scene_store.resolve_scene(ref)
    storage=run.get('scene_storage',{})
    need(storage.get('schema')==scene_store.LEDGER_SCHEMA and storage.get('cas_unchanged') is True and
         storage.get('source_bindings_unchanged') is True and storage.get('within_budget') is True and
         storage.get('limit_bytes')==100_000_000 and storage.get('scene')==ref['scene'],
         'Native whole-job scene/CAS/source budget ledger required')
    measured=episode_size(job)
    need(measured['within_budget'] is True and measured['total_bytes']==storage.get('job_directory_bytes'),
         'Native entire-job bytes changed or exceed 100 MB')
    need(all(run.get('geometry_artifacts',{}).get(k)==meta.get(k) for k in
             ('collision_geometry_sha256','collision_arrays_sha256')), 'Actual scene collision artifact hashes differ')


def _reset_preflight(rows, base):
    seen=set()
    for row in rows:
        if 'child_result' not in row:continue  # Missing lifecycle evidence still refuses in exact reader.
        child,_=_bound(row['child_result'],base)
        identity=child.get('identity',{})
        need(type(identity.get('pid')) is int and type(identity.get('sid')) is int and
             isinstance(identity.get('start_ticks'),str), 'Native reset process identity missing')
        key=(identity['pid'],identity['sid'],identity['start_ticks'])
        need(key not in seen, 'Duplicate native process reused across episode resets')
        seen.add(key)


def _numeric_preview_rejections(receipt,audit,reference,start,end):
    """Reconstruct typed rejected clones and audit the stricter shared clock."""
    rows=receipt.get('parking_numeric_rejections',[])
    if not rows:return  # Existing successful archives retain their exact contract.
    from sim.auscultation_pi05_v6.execution import FinishedStationaryRoundoffError
    from sim.auscultation_skill_v1.executor import command_derivatives
    need(isinstance(rows,list) and 0<len(rows)<100,'Finite numerical preview rejection ledger required')
    first=rows[0].get('physics_row');previous=-1;cpu=0.
    need(type(first) is int and receipt.get('parking_budget_start_physics_row')==first and
         receipt.get('parking_budget_steps')==end-first<=1200,
         'Rejected hypotheses and actual parking must share one original 5 second budget')
    for row in rows:
        i=row.get('physics_row');cost=row.get('planning_cpu_s');inp=row.get('input_state')
        before=row.get('live_state_before');after=row.get('live_state_after')
        need(row.get('schema')=='m2_given_stop_preview_numeric_rejection_v1' and
             row.get('exception_type')=='FinishedStationaryRoundoffError' and
             type(i) is int and 3<=i<start and i>previous and i%12==0 and row.get('time_s')==audit[i-1]['time_s'] and
             row.get('stop_setter_issued') is False and row.get('physics_advanced') is False and
             row.get('live_state_unchanged') is True and before==after and isinstance(before,dict) and
             before.get('input')==inp and before.get('rows')==i and before.get('reference_sha256')==reference.sha256 and
             row.get('input_sha256')==handoff.digest(inp), 'Exact read-only rejected numerical preview binding required')
        history=np.asarray([e['q_command'] for e in audit[i-3:i]],dtype=float)
        need(np.array_equal(history,row.get('frozen_command_history')) and
             np.array_equal(history,before.get('command_history')) and np.array_equal(history[-1],inp['fields']['current_position']),
             'Rejected numerical preview command/input seam differs')
        need(type(cost) in (int,float) and np.isfinite(cost) and cost>=0,'Finite original preview CPU cost required')
        cpu+=cost;previous=i
        clone=handoff.restore_trajectory(inp);handoff.apply_stop_profile(clone,expected_input=inp);clone.set_target(history[-1])
        failed=False
        for _ in range(handoff.MAX_STOP_STEPS):
            try:q=clone.step()
            except FinishedStationaryRoundoffError as error:
                need(getattr(error,'details',{})==row.get('details'),'Rejected native numerical details differ on replay')
                failed=True;break
            values=command_derivatives(history,q)
            need(values['velocity']<=.175 and values['acceleration']<=.5+1e-8 and values['jerk']<=1.5+1e-8,
                 'Rejected hypothesis concealed an earlier original derivative failure')
            history=np.vstack([history[1:],q])
            if np.array_equal(history,np.tile(q,(3,1))) and np.max(abs(clone.velocity))<=1e-10 and np.max(abs(clone.acceleration))<=1e-10:break
        need(failed,'Claimed rejected numerical preview did not reproduce its narrow native failure')
    need(cpu<30 and receipt['parking_preview']['planning_cpu_s']>=cpu,'Cumulative original preview CPU bound changed')
    for event in audit[first:start]:
        need(event.get('actual_control_owner') in ('TEACHER_L2','ACT_L2') and
             event.get('raw_intent',{}).get('setter',{}).get('offgrid_stop') is False,
             'Numerical preview wait must remain ordinary gated L2 execution')


def _interior_parking(run, receipt, audit, meta, reference, phase_config, *, kin=None):
    """Reconstruct the new trigger, exact stop and actual full-pose settling."""
    need(run.get('handoff_policy_id')==receipt.get('handoff_policy_id')==meta.get('handoff_policy_id')==handoff.POLICY_ID,
         'Original interior handoff policy missing or changed')
    from .m2_act_controller import approach_gate
    first=receipt.get('coarse_entry',{});first_row=first.get('physics_row')
    need(type(first_row) is int and 1<=first_row<=len(audit), 'Original first measured coarse row required')
    measured=audit[first_row-1].get('measurements',{})
    expected_first=approach_gate({key:measured.get(key) for key in ('gap_m','tangent_m','angle_deg','tcp_speed_m_s')},phase_config)
    need(expected_first['passed'] and first.get('measured')==expected_first and
         first.get('reference_sha256')==reference.sha256 and type(first.get('reference_revision')) is int and
         first['reference_revision']==0 and first.get('context')==[meta['episode_id'],meta['target_id'],0,'APPROACH'] and
         np.array_equal(first.get('command_q'),audit[first_row-1]['q_command']),
         'Immutable first original coarse measurements/reference/context changed')
    preview=receipt.get('parking_preview',{});trigger=receipt.get('parking_trigger',{})
    start=receipt.get('parking_start_physics_row');end=receipt.get('parking_end_physics_row')
    need(type(start) is int and type(end) is int and 3<=start<end<=len(audit) and
         end-start==receipt.get('parking_steps') and end-start<=handoff.MAX_PARK_STEPS and
         trigger.get('physics_row')==start and receipt.get('parked_entry',{}).get('physics_row')==end,
         'Complete parking interval must use one original 5 second clock')
    _numeric_preview_rejections(receipt,audit,reference,start,end)
    need(preview.get('schema')==handoff.PREVIEW_SCHEMA and preview.get('handoff_policy_id')==handoff.POLICY_ID and
         preview.get('l2_planning_limits')==handoff.PLANNING_LIMITS and
         preview.get('planning_limits')==handoff.STOP_PLANNING_LIMITS and preview.get('reserved_settle_intervals')==24 and
         preview.get('max_parking_steps')==1200 and preview.get('all_commands_verified') is True and
         preview.get('physics_advanced') is False and preview.get('live_state_unchanged') is True and
         type(preview.get('planning_cpu_s')) in (int,float) and 0<=preview['planning_cpu_s']<30.,
         'Frozen current-position stop preview bounds or verification changed')
    source=preview.get('source_sha256',{});runtime=run.get('runtime_sources_sha256',{})
    need(isinstance(source,dict) and sum(Path(p).name=='m2_given_contact_handoff.py' for p in source)==1 and
         all(runtime.get(p)==digest for p,digest in source.items()), 'Preview source inventory differs from actual runtime')
    history=np.asarray([e['q_command'] for e in audit[start-3:start]],dtype=np.float64)
    commands=np.asarray(preview.get('commands'),dtype=np.float64);steps=preview.get('stop_steps')
    need(type(steps) is int and 1<=steps<=1176 and commands.shape==(steps,7) and np.isfinite(commands).all() and
         steps+24<=end-start and receipt.get('parking_stop_complete_physics_row')==start+steps and
         np.array_equal(history,np.asarray(preview.get('frozen_command_history'))) and
         hashlib.sha256(commands.tobytes()).hexdigest()==preview.get('commands_sha256'),
         'Stop preview commands/history/reserved measured-settle interval changed')
    before=preview.get('live_state_before');after=preview.get('live_state_after');inp=preview.get('input_state')
    need(isinstance(before,dict) and before==after and handoff.digest(before)==preview.get('live_before_sha256')==
         preview.get('live_after_sha256') and before.get('rows')==start and before.get('reference_sha256')==reference.sha256 and
         before.get('input')==inp and handoff.digest(inp)==preview.get('input_sha256') and
         np.array_equal(before.get('command_q'),history[-1]) and np.array_equal(before.get('command_history'),history) and
         before.get('native_commands_sha256')==hashlib.sha256(np.asarray([e['q_command'] for e in audit[:start]],dtype=np.float64).tobytes()).hexdigest(),
         'Actual live input/history/physics/reference preview state binding changed')
    clone=handoff.restore_trajectory(inp)
    need(np.array_equal(inp['fields']['current_position'],history[-1]) and
         np.array_equal(preview.get('target_q'),history[-1]), 'Preview must target actual current commanded position')
    transition=handoff.apply_stop_profile(clone,expected_input=inp)
    need(transition==preview.get('stop_profile_transition')==receipt.get('parking_stop_profile_transition'),
         'Independent L2 to STOP single-field limit transition differs')
    clone.set_target(history[-1]);stop_input=handoff.input_snapshot(clone,profile='stop')
    need(stop_input==preview.get('stop_input_state')==receipt.get('parking_stop_input_state') and
         handoff.digest(stop_input)==preview.get('stop_input_sha256')==receipt.get('parking_stop_input_sha256'),
         'Complete STOP profile/target/current-state input cannot independently reconstruct')
    seam=history.copy();peaks={k:0. for k in ('velocity','acceleration','jerk')}
    from sim.auscultation_skill_v1.executor import command_derivatives
    for q in commands:
        need(np.array_equal(clone.step(),q), 'Frozen actual Ruckig input cannot reconstruct preview bit-exactly')
        values=command_derivatives(seam,q)
        need(values['velocity']<=.175 and values['acceleration']<=.5+1e-8 and values['jerk']<=1.5+1e-8,
             'Original parking derivative seam/limits failed')
        for key in peaks:peaks[key]=max(peaks[key],values[key])
        seam=np.vstack([seam[1:],q])
    need(peaks==preview.get('command_derivative_maxima') and np.max(abs(clone.velocity))<=1e-10 and
         np.max(abs(clone.acceleration))<=1e-10 and np.array_equal(seam,np.tile(commands[-1],(3,1))),
         'Complete preview command stationarity or derivative audit changed')
    # Replay ends here: the tail explicitly repeats the verified command
    # without stepping or resetting the completed live interpolator. Object
    # IDs belong to the source process; every captured tail tick must bind the
    # same four distinct objects and the independently reconstructed q/v/a.
    tail=receipt.get('parking_tail');binding=handoff.constant_tail_binding(tail)
    objects=tail.get('object_ids',{})
    need(isinstance(objects,dict) and set(objects)=={'trajectory','input','output','generator'} and
         all(type(value) is int and value>0 for value in objects.values()) and len(set(objects.values()))==4 and
         before.get('interpolator_object_ids')==objects,
         'Original constant-tail object identities missing or aliased')
    expected_tail=handoff.constant_tail_snapshot(clone,commands[-1],start+steps)
    expected_tail['object_ids']=objects.copy()
    original_output=before.get('interpolator_output',{})
    need(all(np.array_equal(original_output.get(output),inp['fields'][current]) for output,current in
         [('new_position','current_position'),('new_velocity','current_velocity'),('new_acceleration','current_acceleration')]) and
         type(original_output.get('time')) in (int,float) and np.isfinite(original_output['time']) and original_output['time']>=0,
         'Actual pre-stop output differs from frozen live input')
    # Even an unchanged target has a changed full input: max_jerk is 1.49.
    # Ruckig therefore replans and resets its auxiliary output clock. Use the
    # actual independent clone's clock, never the v2 same-target shortcut.
    need(tail==expected_tail, 'Completed stop input/output/command differs from frozen constant tail')
    if kin is None:
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        kin=OpenArmKinematics(contact_frame=geometry.contact_frame)
    first_pose=audit[start].get('parking_pose_before');handoff.validate_sample(first_pose)
    need(before.get('measured_pose')==first_pose, 'Trigger observed pose differs from actual pre-stop native state')
    expected_trigger=handoff.interior_trigger(reference,command_pose=kin.tcp(history[-1]),observed_pose=first_pose['pose_base'],
        measured_speed_m_s=first_pose['tcp_speed_m_s'],phase_config=phase_config,physics_row=start,time_s=first_pose['time_s'],
        context=(meta['episode_id'],meta['target_id'],0,'APPROACH'),reference_sha256=reference.sha256)
    need(expected_trigger['passed'] and trigger==expected_trigger, 'Fresh command/observed interior trigger cannot reconstruct')
    samples=[first_pose]
    from .m2_act_controller import approach_gate
    for i,event in enumerate(audit[start:end]):
        q=np.asarray(event['q_command']);expected=commands[min(i,steps-1)]
        need(np.array_equal(q,expected) and event['controller_after']['subphase']=='given_l2_park' and
             event['controller_after']['phase']=='APPROACH', 'Actual stop/tail command or physical phase differs from preview')
        for channel in ('controller_before','controller_after'):
            labels=event.get(channel,{})
            need('parking_interpolator_hold' in labels and
                 labels['parking_interpolator_hold']==(None if i<steps else binding),
                 'Early/missing constant tail or changed live object/input/output/command binding')
        prior=event.get('parking_pose_before');current=event.get('parking_pose_after')
        handoff.validate_sample(prior);pose_value,speed=handoff.validate_sample(current)
        need(prior==samples[-1] and prior['physics_row']==start+i and current['physics_row']==start+i+1 and
             current['tcp_speed_m_s']==event['measurements'].get('tcp_speed_m_s') and
             approach_gate(dict(handoff.pose_metrics(pose_value,reference),tcp_speed_m_s=speed),phase_config)['passed'] and
             approach_gate(dict(handoff.pose_metrics(kin.tcp(q),reference),tcp_speed_m_s=0.),phase_config)['passed'],
             'Actual consecutive measured/command parking poses left original coarse tube')
        samples.append(current)
    settle=handoff.settle_window(samples[-25:])
    need(samples[-25]['physics_row']>=start+steps and settle['passed'] and settle==receipt.get('parking_settle'),
         'Actual 24 interval full-pose measured settling cannot reconstruct')


def _physical(run, audit, meta):
    """Recheck scalar native contact evidence and preserve all original gates."""
    need(bool(audit), 'Native physics audit is empty')
    reference=_reference_metadata(meta);sources=reference.snapshot()['sources']
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
    _interior_parking(run,h,audit,meta,reference,SimpleNamespace(**expected_gate))
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


def _read_payload(row, base, context, *, allow_truncated=False):
    run,config,freeze=_native_evidence(row,base,context[2])
    return _decode_capture(row,base,context,run,config,freeze,allow_truncated=allow_truncated)


def read_episode(row, base, *, allow_truncated=False):
    """Reconstruct one independently qualified native episode; never pad."""
    need(type(allow_truncated) is bool,'Explicit label-only truncation switch required')
    base=Path(base).resolve()
    _declarations([row],base)
    context=_metadata(row,base)
    if allow_truncated:return _read_payload(row,base,context,allow_truncated=True)
    return _read_payload(row,base,context)


def prepare_dataset(manifest, expected_sha256, qualification=None):
    """Rebuild every array and index from native archives, TRAIN-only stats."""
    manifest_path=Path(manifest).resolve()
    value,manifest_path=_bound(dict(path=str(manifest_path),sha256=expected_sha256),manifest_path.parent)
    need(value.get('schema')==MANIFEST_SCHEMA and value.get('task_id')==pose.TASK_ID and
         value.get('action_semantics_id')==given.ACTION_SEMANTICS_ID and value.get('handoff_policy_id')==handoff.POLICY_ID,
         'Explicit new given dataset manifest and interior handoff policy required')
    rows=value.get('episodes'); base=manifest_path.parent
    _declarations(rows,base)
    contexts=[_metadata(row,base) for row in rows]  # ALL metadata before any payload or source array.
    _reset_preflight(rows,base)
    if qualification is not None:
        _population_preflight(qualification,base,expected_sha256)
    grouped={'train':[],'validation':[]}; receipts=[]; indices=[]; references=[]; sample_rows=[]
    offsets={'train':0,'validation':0}
    for order,(row,context) in enumerate(zip(rows,contexts)):
        samples,receipt,meta=_read_payload(row,base,context)
        given.require_samples(samples)
        split=row['split']; start=offsets[split]; count=len(samples['states']); offsets[split]+=count
        grouped[split].append(samples)
        receipt.update(manifest_sha256=expected_sha256,manifest_row=order,split_sample_range=[start,start+count],
            handoff_policy_id=handoff.POLICY_ID)
        receipts.append(receipt)
        references.append(dict(episode_id=row['episode_id'],reference_sha256=meta['reference_sha256'],
            reference_revision=0,reference=deepcopy(meta['given_reference'])))
        index=deepcopy(samples['derived_index'])
        index.update(manifest_row=order,split=split,
                     split_sample_range=[start,start+count],reference_sha256=meta['reference_sha256'])
        # Explicit source record ranges are derived from the actual window indices.
        windows=index.get('windows',[])
        used=[min((min(w.get('observation_indices',[0])) for w in windows),default=0),
            max((max(w['action_record_indices'])+2 if w.get('action_record_indices') else
                 max(w.get('observation_indices',[0]))+1 for w in windows),default=0)]
        index['used_source_row_range']=used
        index.setdefault('source_row_range',used)
        indices.append(index)
        for local,window in enumerate(windows):
            sample_rows.append(dict(episode_id=row['episode_id'],manifest_row=order,split=split,
                sample_index=start+local,episode_sample_index=local,reference_sha256=meta['reference_sha256'],
                window=deepcopy(window)))
    output={}
    for split,parts in grouped.items():
        need(parts and offsets[split]>0, 'Both TRAIN and VALIDATION require complete real H20 windows')
        output[split]={key:np.concatenate([part[key] for part in parts],axis=0) for key in shared.ARRAY_KEYS}
        output[split].update(schema=given.DATASET_SCHEMA,task_id=pose.TASK_ID,
            observation_contract=pose.observation_contract(),action_semantics_id=given.ACTION_SEMANTICS_ID,
            handoff_policy_id=handoff.POLICY_ID)
    index=dict(schema=INDEX_SCHEMA,manifest_sha256=expected_sha256,task_id=pose.TASK_ID,
        action_semantics_id=given.ACTION_SEMANTICS_ID,episodes=indices,samples=sample_rows)
    output.update(schema=given.DATASET_SCHEMA,task_id=pose.TASK_ID,action_semantics_id=given.ACTION_SEMANTICS_ID,
        observation_contract=pose.observation_contract(),manifest_path=str(manifest_path),manifest_sha256=expected_sha256,
        task_contract=dict(task_id=pose.TASK_ID,target_source=pose.TARGET_SOURCE,contract_id=pose.CONTRACT_ID,
            geometry_id=geometry.GEOMETRY_ID,geometry_source_sha256=geometry.GEOMETRY_SOURCE_SHA256,
            identity_split_sha256=shared.contract.IDENTITY_SPLIT_SHA256,action_semantics_id=given.ACTION_SEMANTICS_ID,
            handoff_policy_id=handoff.POLICY_ID,
            action_representation='block_relative_joint',observation_contract=pose.observation_contract(),
            state_dim=17,history_steps=2,horizon=20,execution_prefix=10,physics_hz=240,control_hz=20),
        episodes=receipts,reference_inventory=references,derived_index=index,test_payloads_opened=False,
        training_eligible=False,normalization=given.fit_normalization(output['train'],split='train'),
        reconstruction_sources={str(Path(module.__file__).resolve()):sha256(module.__file__) for module in
            (shared,given,pose,witness,handoff)},preparer_sha256=sha256(__file__),qualification_reference=qualification)
    if qualification is not None:
        ref=deepcopy(qualification);ref['path']=str(_ref_path(ref,base))
        output['qualification']=_verify_population(ref,receipts)
        output['training_eligible']=True
    return output


def _time(value):
    need(isinstance(value,str), 'Explicit native/freeze timestamp required')
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    need(result.tzinfo is not None, 'Timezone-bound native/freeze timestamp required')
    return result


def _allocation(plan, base):
    """Use the reviewed metadata-only proposal as allocation, never qualification."""
    proposal,p=_bound(plan.get('allocation_proposal'),base)
    mapping,_=_bound(plan.get('allocation_mapping'),base)
    need(plan['allocation_proposal']['sha256']==ALLOCATION_PROPOSAL_SHA256 and
         plan['allocation_mapping']['sha256']==ALLOCATION_MAPPING_SHA256 and
         proposal.get('mapping',{}).get('sha256')==ALLOCATION_MAPPING_SHA256 and
         proposal.get('allocation_feasible') is True and
         proposal.get('train_original_slots_retained')==800 and proposal.get('validation_original_slots_retained')==200,
         'Reviewed finite allocation proposal/mapping binding required')
    slots=mapping.get('all_original_slots',[])
    need(len(slots)==1000 and len({s.get('slot_id') for s in slots})==1000,
         'All 800 TRAIN + 200 VAL original geometry slots must be retained exactly once')
    for slot in slots:
        shared.require_membership(slot.get('split'),slot.get('subject_id'))
        need(type(slot.get('geometry_path_found')) is bool, 'Every geometry outcome must remain explicit')
    for split,n,seed in [('train',800,2026102701),('validation',200,2026102702)]:
        need(sum(s['split']==split for s in slots)==n and mapping.get('summaries',{}).get(split,{}).get('seed')==seed,
             'Original geometry denominator or frozen allocation seed changed')
    allocations=mapping.get('ordered_allocations',{})
    need(len(allocations.get('train',[]))==200 and len(allocations.get('validation',[]))==40,
         'Exactly the reviewed 200 TRAIN/40 VAL finite allocation is required')
    return mapping


def _population_native(attempt, entry, base):
    """Failures remain in the ledger even without a native COMPLETE file."""
    config,config_path=_bound(attempt.get('config'),base)
    metadata=config.get('m2_given_contact_control',{}).get('metadata',{})
    pose.require_route(metadata)
    shared.require_membership(metadata.get('split'),metadata.get('subject_id'))
    need(metadata.get('source_kind')=='isaac_live_physics' and metadata.get('replay') is False,
         'Population attempts must be live native acquisition, never CPU/replay')
    for key in ('episode_id','subject_id','split','target_id'):
        need(metadata.get(key)==attempt.get(key), 'Frozen native config differs from planned identity: '+key)
    owner,owner_path=_bound(entry.get('owner_terminal'),base)
    launch,launch_path=_bound(entry.get('launch_receipt'),base)
    output=owner_path.parent/'output'/attempt['job_id']
    jobs=owner.get('config',{}).get('jobs',[])
    need(owner.get('run_id')==attempt['run_id'] and owner.get('config',{}).get('experiment')=='isaac_batch' and
         any(j.get('id')==attempt['job_id'] and _path(j.get('config'),base)==config_path and
             j.get('sha256')==attempt['config']['sha256'] for j in jobs) and
         launch_path==output.parent/(attempt['job_id']+'.launch.json') and launch.get('id')==attempt['job_id'] and
         launch.get('config_sha256')==attempt['config']['sha256'] and _path(launch.get('output'),base)==output,
         'Launched outcome differs from frozen batch/job/config identity')
    result=dict(success=False,physical_validation=False,episode_id=attempt['episode_id'])
    end=owner.get('ended_utc');flags=[];failure='resource_abort';child=None
    child_file=output.parent/(attempt['job_id']+'.result.json');complete_file=output/'COMPLETE.json'
    partial_file=output/'fresh_expert_result.json'
    need(('child_result' in entry)==child_file.is_file(), 'Finalized child result inventory differs from actual batch files')
    need(('run_result' in entry)==complete_file.is_file(), 'Native COMPLETE result omitted or invented in finalized inventory')
    need(('partial_result' in entry)==(not complete_file.is_file() and partial_file.is_file()),
         'Existing partial native result must be bound when COMPLETE is absent')
    if child_file.is_file():
        child,child_path=_bound(entry['child_result'],base)
        need(child_path==output.parent/(attempt['job_id']+'.result.json') and
             all(child.get(k)==launch.get(k) for k in ('id','pid','identity','config_sha256','output','started_utc')),
             'Child outcome differs from original launch receipt')
        end=child.get('ended_utc')
    else:flags.extend(('lifecycle_failure','launched_aborted_missing_result'))
    if complete_file.is_file():
        result,result_path=_bound(entry['run_result'],base)
        need(result_path==complete_file and (child is None or child.get('result_sha256')==entry['run_result']['sha256']),
             'Outcome native COMPLETE binding differs from child exit receipt')
        reported=result.get('success') is True and result.get('physical_validation') is True and result.get('errors')==[]
        if reported:
            shared._require_run_identity(result,attempt)
        stage=owner.get('stage',{})
        normal=(child is not None and type(child.get('exit_code')) is int and child['exit_code']==0 and
            child.get('worker_complete') is True and owner.get('completed') is True and owner.get('gpu_released') is True and
            stage.get('passed') is True and stage.get('exit_code')==0 and stage.get('process_group_drained') is True)
        if child is not None:
            need(child.get('physical_pass') is (reported and child.get('exit_code')==0),
                 'Child physical verdict differs from native COMPLETE/exit status')
        if normal:
            failure='native_failure'
            if reported:
                combined=dict(attempt,**{key:entry[key] for key in ('run_result','child_result','owner_terminal')})
                _native_evidence(combined,base,metadata,physical=False)
                need(result.get('controller_motion_complete') is True and
                     result.get('executed_controller_class')=='sim.ausc_policy.m2_given_contact_controller.GivenContactExpert',
                     'Population success is not a completed given native expert')
        else:
            flags.append('lifecycle_failure')
            result=dict(result,success=False,physical_validation=False,reported_physical_validation=reported)
        if not reported:flags.append('native_safety_failure')
    else:
        if child is not None:
            need(child.get('physical_pass') is False and child.get('worker_complete') is False,
                 'Missing native result cannot claim completion or success')
            flags.append('lifecycle_failure')
        if partial_file.is_file():
            partial,p=_bound(entry['partial_result'],base)
            need(p==partial_file, 'Partial native result must be the original job report')
            if partial.get('physical_validation') is False or partial.get('success') is False:
                flags.append('native_safety_failure')
    result=dict(result,_started_utc=launch.get('started_utc'),_ended_utc=end,
                _primary_failure_kind=failure,_reason_flags=sorted(set(flags)))
    return result


def verify_population(ref, episode_rows):
    """Public audit always reconstructs captures; serialized counts are untrusted."""
    receipt,path=_bound(ref,Path.cwd())
    manifest_path=_population_manifest(receipt,path.parent)
    manifest_ref=receipt['manifest']
    _population_preflight(ref,Path.cwd(),manifest_ref['sha256'])
    rebuilt=prepare_dataset(manifest_path,manifest_ref['sha256'])
    need(episode_rows==rebuilt['episodes'], 'Caller episode counts/indices do not match fresh reconstruction')
    return _verify_population(ref,rebuilt['episodes'])


def _plan_metadata(rows, base):
    """Every planned config and reference, including failures, before payloads."""
    for row in rows:
        shared.require_membership(row.get('split'),row.get('subject_id'))
    for row in rows:
        config,_=_bound(row.get('config'),base)
        meta=config.get('m2_given_contact_control',{}).get('metadata',{})
        shared.require_membership(meta.get('split'),meta.get('subject_id'))
        pose.require_route(meta)
        need(meta.get('handoff_policy_id')==handoff.POLICY_ID and
             config.get('m2_given_contact_control',{}).get('handoff_policy_id')==handoff.POLICY_ID,
             'Every frozen attempt requires the new interior handoff policy')
        need(meta.get('source_kind')=='isaac_live_physics' and meta.get('replay') is False and
             all(meta.get(k)==row.get(k) for k in ('episode_id','subject_id','split','target_id')),
             'Every planned config must bind live given-task identity, never CPU/replay/TEST')
        _reference_metadata(meta)


def _population_manifest(receipt, base):
    """The private preparer and public reconstruction share one exact binding."""
    ref=receipt.get('manifest')
    need(isinstance(ref,dict) and set(ref)=={'path','sha256'} and
         ref['sha256']==receipt.get('manifest_sha256'),
         'Population qualification requires exact original immutable manifest reference')
    value,path=_bound(ref,base)
    need(value.get('schema')==MANIFEST_SCHEMA and value.get('task_id')==pose.TASK_ID and
         value.get('action_semantics_id')==given.ACTION_SEMANTICS_ID and value.get('handoff_policy_id')==handoff.POLICY_ID,
         'Population original manifest must declare the new given task/action')
    return path


def _population_preflight(ref, base, manifest_sha256):
    receipt,path=_bound(ref,base)
    need(receipt.get('handoff_policy_id')==handoff.POLICY_ID, 'Population interior handoff policy required before payload access')
    _population_manifest(receipt,path.parent)
    need(receipt.get('manifest_sha256')==manifest_sha256, 'Population manifest binding differs before payload access')
    plan,plan_path=_bound(receipt.get('attempt_plan'),path.parent)
    need(plan.get('handoff_policy_id')==handoff.POLICY_ID, 'Frozen attempt plan interior handoff policy required before payload access')
    rows=plan.get('attempts')
    need(isinstance(rows,list) and bool(rows), 'Complete planned attempt metadata required before payload access')
    _plan_metadata(rows,plan_path.parent)
    if 'processing_sources' in plan.get('accounting_scope',{}):_processing_preflight(plan,receipt,plan_path.parent)


def _launch_state(plan, attempt, entry, base):
    root=_path(plan.get('run_root'),base)
    need(all(isinstance(attempt.get(k),str) and re.fullmatch('[a-z0-9_-]+',attempt[k]) for k in ('run_id','job_id')),
         'Frozen native run/job names required')
    path=root/attempt['run_id']/'output'/(attempt['job_id']+'.launch.json')
    actual=path.is_file()
    need(type(entry.get('launched')) is bool and entry['launched'] is actual,
         'Launched status differs from authoritative native launch file')
    if actual:
        _,bound=_bound(entry.get('launch_receipt'),base)
        need(bound==path, 'Authoritative native launch receipt differs from fixed run/job path')
    return actual


def _verify_population(ref, episode_rows):
    """Private path receives only this preparer's freshly reconstructed rows."""
    """Qualify the fixed finite TRAIN denominator; no positive-only ledger."""
    receipt,path=_bound(ref,Path.cwd());base=path.parent
    _population_manifest(receipt,base)
    need(receipt.get('schema')==POPULATION_SCHEMA and receipt.get('task_id')==pose.TASK_ID and
         receipt.get('action_semantics_id')==given.ACTION_SEMANTICS_ID and receipt.get('passed') is True and
         receipt.get('handoff_policy_id')==handoff.POLICY_ID,
         'Explicit new-task population qualification receipt required')
    need(type(receipt.get('minimum_episodes')) is int and receipt['minimum_episodes']>=150 and
         type(receipt.get('minimum_identities')) is int and receipt['minimum_identities']>=20 and
         type(receipt.get('minimum_success_rate')) in (int,float) and .95<=receipt['minimum_success_rate']<=1,
         'Population qualification thresholds weakened')
    need(isinstance(episode_rows,list) and bool(episode_rows) and
         all(r.get('manifest_sha256')==receipt.get('manifest_sha256') for r in episode_rows),
         'Population receipt/dataset manifest binding mismatch')
    plan,plan_path=_bound(receipt.get('attempt_plan'),base)
    need(plan.get('schema')==PLAN_SCHEMA and plan.get('task_id')==pose.TASK_ID and
         plan.get('action_semantics_id')==given.ACTION_SEMANTICS_ID and plan.get('frozen_before_launch') is True and
         plan.get('handoff_policy_id')==handoff.POLICY_ID,
         'New immutable prelaunch given-task plan required')
    frozen=_time(plan.get('frozen_utc'));mapping=_allocation(plan,plan_path.parent)
    rows=plan.get('attempts');planned=shared._attempt_index(rows,'Frozen given plan')
    _plan_metadata(rows,plan_path.parent)
    if 'processing_sources' in plan.get('accounting_scope',{}):_processing_preflight(plan,receipt,plan_path.parent)
    seen={key:set() for key in ('episode_id','reset_id','job','config','slot_id')}
    allocated=[dict(a,split=split) for split in ('train','validation') for a in mapping['ordered_allocations'][split]]
    need(len(rows)==len(allocated), 'Fixed plan must cover exactly the ordered finite allocation')
    for row,allocation in zip(rows,allocated):
        shared.require_membership(row.get('split'),row.get('subject_id'))
        need(all(row.get(k)==allocation.get(k) for k in ('slot_id','subject_id','split')) and
             row.get('target_id')==allocation.get('target') and row.get('preclassified_reachable') is True,
             'Frozen attempt order/identity/reachability differs from reviewed allocation')
        for key,value in dict(episode_id=row.get('episode_id'),reset_id=row.get('reset_id'),
            job=(row.get('run_id'),row.get('job_id')),config=tuple(sorted(row.get('config',{}).items())),slot_id=row['slot_id']).items():
            need(value and value not in seen[key], 'Duplicate planned reset/episode/config/job/slot identity')
            seen[key].add(value)
    execution,_=_bound(receipt.get('execution_inventory'),base)
    ledger,_=_bound(receipt.get('attempt_ledger'),base)
    need(execution.get('schema')==EXECUTION_SCHEMA and execution.get('finalized') is True and
         execution.get('attempt_plan_sha256')==receipt['attempt_plan']['sha256'] and ledger.get('schema')==LEDGER_SCHEMA,
         'Finalized new-task execution inventory/ledger binding required')
    inventory=shared._attempt_index(execution.get('attempts'),'Execution inventory')
    outcomes=shared._attempt_index(ledger.get('attempts'),'ALL-outcome ledger')
    need(set(inventory)==set(planned)==set(outcomes), 'Inventory/ledger must retain ALL planned outcomes including unlaunched')
    captured={}
    for row in episode_rows:
        key=row.get('episode_id')
        need(key not in captured, 'Duplicate reconstructed episode identity')
        captured[key]=row
    hashes=set();paths=set();successes={};native_times={};categories={};reasons={};unlaunched=0
    for aid,attempt in planned.items():
        entry=inventory[aid];outcome=outcomes[aid]
        _launch_state(plan,attempt,entry,base)
        need(type(entry.get('launched')) is bool and outcome.get('launched') is entry['launched'] and
             all(outcome.get(k)==attempt.get(k) for k in ('episode_id','subject_id','split','reset_id','run_id',
                 'job_id','slot_id','target_id','preclassified_reachable','config')),
             'Outcome identity/reachability differs from prelaunch plan')
        for key in ('run_result','owner_terminal','child_result','launch_receipt','partial_result'):
            need(outcome.get(key)==entry.get(key), 'Outcome receipt differs from finalized execution inventory')
        if not entry['launched']:
            need(all(key not in entry for key in ('run_result','owner_terminal','child_result','launch_receipt','partial_result')),
                 'Unlaunched attempt carries native receipts')
            unlaunched+=1;success=False;category='planned_unlaunched'
        else:
            result=_population_native(attempt,entry,base)
            for reason in result.get('_reason_flags',[]):reasons[reason]=reasons.get(reason,0)+1
            success=result.get('success') is True and result.get('physical_validation') is True and result.get('errors')==[]
            if success:
                need(result.get('learned_policy') is False and result.get('task_id')==pose.TASK_ID and
                     result.get('geometry_id')==geometry.GEOMETRY_ID and
                     result.get('geometry_source_sha256')==geometry.GEOMETRY_SOURCE_SHA256,
                     'Native success is not the frozen given expert/current geometry')
            if 'run_result' in entry:
                digest=entry['run_result']['sha256'];p=str(_ref_path(entry['run_result'],base))
                need(digest not in hashes and p not in paths, 'Duplicate native result receipt identity')
                hashes.add(digest);paths.add(p)
            native_times[aid]=(_time(result.get('_started_utc')),_time(result.get('_ended_utc')))
            need(native_times[aid][0]>=frozen and native_times[aid][1]>=native_times[aid][0],
                 'Native launch/outcome predates prelaunch freeze or has reversed time')
            capture=captured.get(attempt['episode_id'])
            if capture is not None:
                need(all(capture.get(k)==attempt.get(k) for k in ('subject_id','split','reset_id','run_id','job_id')) and
                     capture.get('run_result')==entry.get('run_result') and success,
                     'Reconstructed capture absent from successful frozen native attempt')
            category=(result.get('_primary_failure_kind','native_failure') if not success else 'capture_failure' if capture is None else
                      'success_zero_windows' if capture.get('windows',0)==0 else 'eligible')
        need(outcome.get('native_success') is success and outcome.get('outcome')==category,
             'Ledger native verdict/capture outcome differs from independently bound evidence')
        successes[aid]=success;categories[category]=categories.get(category,0)+1
    need(set(captured)<=set(a['episode_id'] for a in rows), 'Dataset has an unplanned episode')
    train=[a for a in rows if a['split']=='train'];first=train[:100]
    gate=execution.get('first100_gate',{});first_success=sum(successes[a['attempt_id']] for a in first)
    need(len(first)==100 and gate.get('attempt_ids')==[a['attempt_id'] for a in first] and
         gate.get('native_successes')==first_success and gate.get('passed') is True and first_success>=95,
         'First 100 fixed TRAIN attempts failed the 95-success bulk gate')
    decided=_time(gate.get('decided_utc'))
    need(all(a['attempt_id'] in native_times and native_times[a['attempt_id']][1]<=decided for a in first) and
         all(native_times[a['attempt_id']][0]>=decided for a in rows if a not in first and a['attempt_id'] in native_times),
         'Fixed remainder was launched before completed first100 gate')
    need(unlaunched==0, 'Finite collection incomplete: planned/unlaunched attempts retained')
    nsuccess=sum(successes[a['attempt_id']] for a in train);denominator=len(train)
    qualified=[r for r in episode_rows if r['split']=='train' and r.get('windows',0)>0]
    need(len(qualified)>=receipt['minimum_episodes'] and len({r['subject_id'] for r in qualified})>=receipt['minimum_identities'],
         'Eligible TRAIN episodes with real windows/identities below required 150/20 gate')
    need(nsuccess/denominator>=receipt['minimum_success_rate'], 'Native conditional-reachable success rate below .95 gate')
    accounting=_accounting(plan,rows,inventory,base)
    need(accounting.get('all_accounted_jobs_within_budget') is True,
         'Whole-job/global population amortization exceeds unchanged 100 MB gate')
    return dict(schema=POPULATION_SCHEMA,passed=True,receipt_sha256=ref['sha256'],
        manifest_sha256=receipt['manifest_sha256'],attempt_plan_sha256=receipt['attempt_plan']['sha256'],
        handoff_policy_id=handoff.POLICY_ID,
        execution_inventory_sha256=receipt['execution_inventory']['sha256'],
        reachable_denominator=denominator,reachable_successes=nsuccess,success_rate=nsuccess/denominator,
        first100_native_successes=first_success,planned_attempts=len(rows),launched_attempts=len(rows)-unlaunched,
        unlaunched_attempts=unlaunched,original_train_slots=800,original_validation_slots=200,
        planning_unfound=sum(not s['geometry_path_found'] for s in mapping['all_original_slots']),
        qualified_train_episodes=len(qualified),qualified_train_identities=len({r['subject_id'] for r in qualified}),
        outcome_counts=categories,overlapping_reason_counts=reasons,accounting=accounting,
        empirical_scope='Declared conditional-reachable population; not confidence lower bound or all-target coverage')


def _storage_statistics(root):
    """Separate physical-storage report: lstat only, never follow any link.

    Logical bytes count each regular path and the links' own bytes. Allocated
    physical bytes include directories and count each device/inode only once.
    This report supplies no episode budget or native qualification decision.
    """
    need(stat.S_ISDIR(root.lstat().st_mode), 'Real storage inventory directory required')
    stack=[root];inodes=set();regular=links=files=symlinks=directories=physical=0
    while stack:
        path=stack.pop();entry=path.lstat();key=(entry.st_dev,entry.st_ino)
        if key not in inodes:
            inodes.add(key);physical+=entry.st_blocks*512
        if stat.S_ISLNK(entry.st_mode):
            symlinks+=1;links+=entry.st_size
        elif stat.S_ISREG(entry.st_mode):
            files+=1;regular+=entry.st_size
        elif stat.S_ISDIR(entry.st_mode):
            directories+=1;stack.extend(path.iterdir())
        else:
            need(False, 'Unsupported special file in separate storage inventory')
    return dict(total_bytes=regular+links,regular_file_bytes=regular,files=files,
        symlink_bytes=links,symlinks=symlinks,directories=directories,
        inode_unique_physical_bytes=physical,unique_inodes=len(inodes),training_eligible=False)


def metadata_accounting(scope, base, *, snapshot_phase, excluded_paths=()):
    """Measure a finite metadata tree; charge its frozen future envelopes once.

    Missing declared files are allowed while collection is in progress. Every
    existing path is checked with lstat; no links, hard links or unknown files
    can hide bytes. This accounting decision never supplies native eligibility.
    """
    need(scope.get('version')==ACCOUNTING_SCHEMA and snapshot_phase in
         ('prelaunch','reconcile','pre_export','post_export','training_preflight'), 'Explicit metadata accounting version/phase required')
    meta=scope.get('operational_metadata',{});seal=scope.get('final_seal',{})
    need(meta.get('limit_bytes')==meta.get('charged_bytes')==METADATA_ENVELOPE_BYTES and
         seal.get('limit_bytes')==seal.get('charged_bytes')==SEAL_ENVELOPE_BYTES,
         'Frozen metadata/seal envelope caps or charges changed')
    need(Path(meta.get('root','')).is_absolute() and Path(seal.get('path','')).is_absolute(),
         'Absolute finite operational metadata/seal paths required')
    root=_path(meta['root'],base);seal_path=_path(seal['path'],base)
    def overlap(a,b):return a==b or a in b.parents or b in a.parents
    excluded=[_path(p,base) for p in scope.get('owner_roots',[])]
    excluded.extend(_path(e['reference']['path'],base) for e in scope.get('extra_input_refs',[]))
    excluded.extend(_path(p,base) for p in excluded_paths)
    need(not overlap(root,seal_path) and all(not overlap(root,p) and not overlap(seal_path,p) for p in excluded),
         'Metadata/seal roots overlap an owner or explicitly charged input')
    allowed=meta.get('allowed_files')
    need(isinstance(allowed,list) and bool(allowed) and len(set(allowed))==len(allowed) and all(
         isinstance(p,str) and p and not Path(p).is_absolute() and '..' not in Path(p).parts and
         Path(p).as_posix()==p and Path(p).suffix in ('.json','.log') for p in allowed),
         'Exact finite relative metadata file allowlist required')
    directories={str(p) for name in allowed for p in Path(name).parents if str(p)!='.'}
    stack=[root] if root.exists() else [];inodes=set();physical=total=0;files={}
    while stack:
        path=stack.pop();entry=path.lstat();key=(entry.st_dev,entry.st_ino)
        need(key not in inodes and not stat.S_ISLNK(entry.st_mode), 'Duplicate inode or symlink in operational metadata')
        inodes.add(key);physical+=entry.st_blocks*512
        relative=path.relative_to(root).as_posix()
        if stat.S_ISDIR(entry.st_mode):
            need(path==root or relative in directories, 'Unknown operational metadata directory')
            stack.extend(path.iterdir())
        else:
            need(stat.S_ISREG(entry.st_mode) and entry.st_nlink==1 and relative in allowed,
                 'Unknown/hard-linked/special operational metadata file')
            total+=entry.st_size
            need(total<=METADATA_ENVELOPE_BYTES, 'Operational metadata envelope overrun')
            files[relative]=dict(bytes=entry.st_size,sha256=sha256(path))
    seal_bytes=0
    if seal_path.exists():
        info=seal_path.lstat()
        need(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and (info.st_dev,info.st_ino) not in inodes,
             'External final seal must be a unique regular file')
        seal_bytes=info.st_size
        need(seal_bytes<=SEAL_ENVELOPE_BYTES, 'External final seal envelope overrun')
    return dict(schema=ACCOUNTING_SCHEMA,charged_bytes=METADATA_ENVELOPE_BYTES+SEAL_ENVELOPE_BYTES,
        observed_bytes=total,seal_observed_bytes=seal_bytes,files=dict(sorted(files.items())),
        observed_at=datetime.now(timezone.utc).isoformat(),snapshot_phase=snapshot_phase,
        inode_unique_physical_bytes=physical,unique_inodes=len(inodes),training_eligible=False)


def _source_tree(root,allowed):
    """Closed lstat inventory; code and its physical archive never share inodes."""
    need(isinstance(allowed,list) and allowed and len(set(allowed))==len(allowed) and all(isinstance(p,str) and p and
         not Path(p).is_absolute() and '..' not in Path(p).parts and '*' not in p and '?' not in p and
         Path(p).as_posix()==p for p in allowed), 'Finite exact source allowlist required')
    parents={str(p) for n in allowed for p in Path(n).parents if str(p)!='.'}
    pending=[root] if root.exists() else [];files={};seen=set();total=physical=0
    while pending:
        path=pending.pop();info=path.lstat();inode=(info.st_dev,info.st_ino)
        need(inode not in seen and not stat.S_ISLNK(info.st_mode), 'Duplicate inode or link in source scope')
        seen.add(inode);physical+=info.st_blocks*512;relative=path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            need(path==root or relative in parents, 'Unknown source directory');pending.extend(path.iterdir())
        else:
            need(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and relative in allowed, 'Unknown, linked or special source file')
            total+=info.st_size;need(total<=PROCESSING_ENVELOPE_BYTES, 'Processing source envelope overrun')
            files[relative]=dict(path=str(path),bytes=info.st_size,sha256=sha256(path))
    return files,total,physical,seen


def _historical_execution(frozen,base):
    """Reopen the original gate at its original root; never substitute new code."""
    value,path=_bound(frozen.get('reference'),base);root=_path(frozen.get('root'),base)
    need(value.get('schema')=='m2_given_collection_execution_freeze_v1' and value.get('root')==str(root) and root not in path.parents,
         'Original execution freeze/root binding changed')
    files,_,_,_=_source_tree(root,list(frozen.get('files',{})))
    need(files==value.get('files')==frozen.get('files') and value.get('environment')==frozen.get('environment'),
         'Original executed tree/environment changed')
    need(value.get('code_commit')==files.get('CODE_COMMIT') and value.get('source_manifest')==files.get('SOURCE_MANIFEST.json') and
         value.get('environment',{}).get('ruckig_version')=='0.15.3', 'Original execution commit/dependency role differs')
    for ref in (value.get('code_commit'),value.get('source_manifest'),value.get('environment',{}).get('python'),value.get('environment',{}).get('ruckig')):
        need(isinstance(ref,dict), 'Original execution dependency reference required')
        target=_ref_path({key:ref[key] for key in ('path','bytes','sha256') if key in ref},base)
        need(target.is_file() and target.stat().st_size==ref.get('bytes') and sha256(target)==ref.get('sha256'),
             'Original execution source/dependency changed')
        if 'invoked_path' in ref:
            need(Path(ref['invoked_path']).is_absolute() and Path(ref['invoked_path']).resolve()==target,
                 'Original invoked interpreter now resolves to another binary')
    need(value.get('reviewed_commit')==(root/'CODE_COMMIT').read_text().strip() and
         re.fullmatch('[0-9a-f]{40}',str(value.get('reviewed_commit'))), 'Original execution reviewed commit changed')
    return frozen


def processing_environment():
    """Exact official CPU environment used for final export and later fitting."""
    need(str(Path(__file__).resolve().parents[2])==FINAL_PROCESSING_ROOT and sys.prefix==OFFICIAL_ACT_PREFIX,
         'Final processing requires its frozen root and official environment')
    try:import ruckig
    except ImportError as error:raise ValueError('Official processing requires actual Ruckig 0.15.3') from error
    need(ruckig.__version__=='0.15.3', 'Official processing requires actual Ruckig 0.15.3')
    def binding(path):
        path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path))
    return dict(prefix=sys.prefix,python=binding(sys.executable),numpy_version=np.__version__,
        ruckig_version=ruckig.__version__,ruckig=binding(ruckig.__file__))


def _processing_preflight(plan,receipt,base):
    scope=plan['accounting_scope'];source=scope['processing_sources']
    metadata_accounting(scope,base,snapshot_phase='pre_export')
    excluded=[*scope['owner_roots'],scope['operational_metadata']['root'],scope['final_seal']['path'],
        *(e['reference']['path'] for e in scope['extra_input_refs'])]
    report=processing_source_accounting(source,base,snapshot_phase='pre_export',excluded_paths=excluded)
    need(receipt.get('processing_freeze')==report.get('descriptor'), 'Qualification must forward-bind the exact final processing freeze')
    descriptor,_=_bound(report['descriptor'],base)
    need(descriptor.get('official_environment')==processing_environment() and descriptor.get('preparer_sha256')==sha256(__file__),
         'Final official environment/preparer differs from its frozen descriptor')
    root=Path(source['live_root']);files=descriptor['files']
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and (name=='sim' or name.startswith('sim.')):
            path=Path(path).resolve()
            need(root in path.parents and files.get(path.relative_to(root).as_posix(),{}).get('sha256')==sha256(path),
                 'Imported processing source lies outside its exact final freeze')
    return report


def processing_source_accounting(scope,base,*,snapshot_phase,excluded_paths=()):
    """Precharge the final source tree and exact archive, then close every file."""
    need(scope.get('scope_id')=='final_processing_sources_v1' and
         scope.get('limit_bytes')==scope.get('charged_bytes')==PROCESSING_ENVELOPE_BYTES and
         scope.get('live_root')==FINAL_PROCESSING_ROOT, 'Fixed final processing source scope/root/cap required')
    live=_path(scope['live_root'],base);archive_root=_path(scope.get('archive_root'),base)
    descriptor=_path(scope.get('descriptor_path'),base);name=scope.get('archive_file')
    need(isinstance(name,str) and Path(name).name==name and name.endswith('.tar.gz') and
         Path(scope['archive_root']).is_absolute() and Path(scope['descriptor_path']).is_absolute(), 'Unique absolute processing archive/descriptor paths required')
    def overlap(a,b):return a==b or a in b.parents or b in a.parents
    need(not overlap(live,archive_root) and not overlap(live,descriptor) and not overlap(archive_root,descriptor) and
         all(not overlap(root,_path(p,base)) for root in (live,archive_root) for p in excluded_paths),
         'Processing source roots overlap another charged path')
    files,live_bytes,physical,inodes=_source_tree(live,scope.get('allowed_files'))
    archives,archive_bytes,archive_physical,archive_inodes=_source_tree(archive_root,[name])
    need(not inodes&archive_inodes and live_bytes+archive_bytes<=PROCESSING_ENVELOPE_BYTES, 'Duplicate source inode or total source cap overrun')
    result=dict(scope_id=scope['scope_id'],charged_bytes=PROCESSING_ENVELOPE_BYTES,observed_bytes=live_bytes+archive_bytes,
        live_bytes=live_bytes,archive_bytes=archive_bytes,inode_unique_physical_bytes=physical+archive_physical,
        observed_at=datetime.now(timezone.utc).isoformat(),snapshot_phase=snapshot_phase,pending=not descriptor.exists())
    if not descriptor.exists():
        need(snapshot_phase in ('prelaunch','reconcile'), 'Final processing descriptor required')
        return result
    desc_ref=dict(path=str(descriptor),bytes=descriptor.stat().st_size,sha256=sha256(descriptor));value,_=_bound(desc_ref,base)
    need(value.get('schema')=='m2_given_final_processing_freeze_v1' and value.get('scope_id')==scope['scope_id'] and
         value.get('live_root')==str(live) and value.get('files')==files and
         value.get('required_unchanged_gate_sources')==scope.get('required_unchanged_gate_sources') and bool(scope.get('required_unchanged_gate_sources')),
         'Final descriptor must close every live file and retain all prior executed gate bindings')
    need(value.get('original_execution_freeze')==scope.get('original_execution_freeze') and
         isinstance(scope.get('original_execution_freeze'),dict), 'Final descriptor must forward-bind the original execution freeze')
    _historical_execution(scope['original_execution_freeze'],base)
    for relative,digest in scope['required_unchanged_gate_sources'].items():
        need(files.get(relative,{}).get('sha256')==digest, 'Final processing replaced an already executed gate')
    need(value.get('code_commit')==files.get('CODE_COMMIT') and value.get('source_manifest')==files.get('SOURCE_MANIFEST.json') and
         re.fullmatch('[0-9a-f]{40}',str(value.get('reviewed_commit'))) and
         (live/'CODE_COMMIT').read_text().strip()==value['reviewed_commit'], 'Reviewed processing commit/source manifest changed')
    manifest,_=_bound(value['source_manifest'],base);names=set()
    for row in manifest:
        key=row.get('file');need(key in files and key not in names and row.get('sha256')==files[key]['sha256'] and
            ('bytes' not in row or row['bytes']==files[key]['bytes']), 'Processing SOURCE_MANIFEST row differs')
        names.add(key)
    need(names==set(files)-{'CODE_COMMIT','SOURCE_MANIFEST.json'}, 'Processing manifest omits live/importable files')
    need(value.get('archive')==archives.get(name) and value.get('preparer_sha256')==
         files.get('sim/ausc_policy/m2_given_contact_dataset.py',{}).get('sha256'), 'Processing archive/preparer binding changed')
    archived=set();parents={str(p) for n in files for p in Path(n).parents if str(p)!='.'}
    with tarfile.open(archive_root/name,'r:gz') as stream:
        members=set()
        for member in stream:
            key=member.name
            need(key and key not in members and not Path(key).is_absolute() and '..' not in Path(key).parts and
                 Path(key).as_posix()==key, 'Duplicate/absolute/traversing processing archive entry')
            members.add(key)
            if member.isdir():need(key in parents, 'Unknown processing archive directory');continue
            need(member.type in (tarfile.REGTYPE,tarfile.AREGTYPE) and key in files and member.size==files[key]['bytes'],
                 'Extra/linked/special processing archive entry')
            reader=stream.extractfile(member);digest=hashlib.sha256()
            for chunk in iter(lambda:reader.read(1024*1024),b''):digest.update(chunk)
            need(digest.hexdigest()==files[key]['sha256'], 'Processing archive and live bytes differ');archived.add(key)
    need(archived==set(files), 'Processing archive omits live files')
    result.update(descriptor=desc_ref,reviewed_commit=value['reviewed_commit'],official_environment=value.get('official_environment'))
    return result


def _failed_native_accounting(scope,base,job_roots,required_refs,*,snapshot_phase):
    """Retained cost only when the original observed-job ledger is unavailable.

    A launch can die before creating a job directory. No fake directory or
    denominator is supplied to scene_store.batch_ledger in that case.
    """
    owner_bytes=0;owners=[];seen=set();terminals=[];failed=False
    for name in scope['owner_roots']:
        root=_path(name,base)
        if not root.exists():continue
        terminal=root/'result.json'
        need(terminal.is_file(), 'Unlaunched owner lacks terminal failure evidence')
        binding=dict(path=str(terminal),bytes=terminal.stat().st_size,sha256=sha256(terminal));value,_=_bound(binding,base)
        stage=value.get('stage',{})
        need(value.get('gpu_released') is True and stage.get('process_group_drained') is True and value.get('ended_utc'),
             'Unlaunched owner must have actual drained terminal evidence')
        failed=failed or not (stage.get('passed') is True and stage.get('exit_code')==0)
        _time(value['ended_utc']);terminals.append(binding)
        files=[]
        for path in root.rglob('*'):
            info=path.lstat();inode=(info.st_dev,info.st_ino)
            need(not stat.S_ISLNK(info.st_mode) and inode not in seen, 'Linked/duplicate native failure path')
            seen.add(inode)
            if stat.S_ISDIR(info.st_mode):continue
            need(stat.S_ISREG(info.st_mode) and info.st_nlink==1, 'Special/hard-linked native failure file')
            files.append(dict(path=str(path),bytes=info.st_size,sha256=sha256(path)));owner_bytes+=info.st_size
        owners.append(dict(root=str(root),files=files))
    need(bool(terminals) and failed, 'No terminal failure exists for diagnostic cost accounting')
    extras={}
    for entry in scope['extra_input_refs']:
        ref=entry['reference'];path=_ref_path(ref,base);info=path.lstat()
        need(entry.get('category') in ('source','config','sdk','prepared','other_runtime') and stat.S_ISREG(info.st_mode) and
             (info.st_nlink==1 or entry.get('category') in ('other_runtime','sdk')) and
             info.st_size==ref.get('bytes') and sha256(path)==ref.get('sha256'), 'Changed failure-accounting input')
        if not any(_path(root,base) in path.parents for root in scope['owner_roots']):extras[str(path)]=ref
    need(all(path in extras or any(_path(root,base) in Path(path).parents for root in scope['owner_roots'])
         for path in required_refs), 'Required input omitted from diagnostic frozen scope')
    metadata=metadata_accounting(scope,base,snapshot_phase=snapshot_phase,excluded_paths=list(extras))
    source=None
    if 'processing_sources' in scope:
        source=processing_source_accounting(scope['processing_sources'],base,snapshot_phase=snapshot_phase,
            excluded_paths=[*scope['owner_roots'],*extras,scope['operational_metadata']['root'],scope['final_seal']['path']])
    extra_bytes=sum(ref['bytes'] for ref in extras.values());observed=sum(root.is_dir() for root in job_roots)
    return dict(qualification=False,training_eligible=False,complete_shared_union_available=False,
        observed_jobs=observed,accounted_bytes_per_observed_job=None,owner_operational_bytes=owner_bytes,
        owner_files=owners,owner_terminal_failures=terminals,additional_inputs=list(extras.values()),additional_input_bytes=extra_bytes,
        metadata_envelope=metadata,processing_source_envelope=source,
        total_accounted_bytes_known=owner_bytes+extra_bytes+metadata['charged_bytes']+(0 if source is None else source['charged_bytes']),
        all_accounted_jobs_within_budget=False,storage_reports=[dict(category=r['category'],root=r['path'],
            **_storage_statistics(_path(r['path'],base))) for r in scope['storage_roots']],
        scope='Diagnostic terminal failure only; complete shared/CAS union unavailable; no episode amortization or qualification')


def _accounting(plan, rows, inventory, base, *, snapshot_phase='pre_export'):
    """Existing scene ledger, per owner and one unique union; no partial scope."""
    from . import scene_store
    from .storage_policy import EPISODE_LIMIT_BYTES
    scope=plan.get('accounting_scope',{})
    need(isinstance(scope,dict) and isinstance(scope.get('owner_roots'),list) and
         isinstance(scope.get('extra_input_refs'),list), 'Predeclared whole-population accounting scope required')
    owners={};required={}
    for row in rows:
        aid=row['attempt_id'];entry=inventory[aid]
        cfg=_ref_path(row['config'],base);required[str(cfg)]=row['config']
        if row.get('source_freeze') is not None:
            freeze,freeze_path=_bound(row['source_freeze'],base)
            required[str(freeze_path)]=row['source_freeze']
            for reference in [freeze.get('code_commit_file'),freeze.get('source_manifest'),*freeze.get('files',{}).values()]:
                required[str(_ref_path(reference,freeze_path.parent))]=reference
        if not entry['launched']:continue
        _,owner_path=_bound(entry.get('owner_terminal'),base)
        root=owner_path.parent;owners.setdefault(root,[]).append(root/'output'/row['job_id'])
    declared=[_path(value,base) for value in scope['owner_roots']]
    subset=scope.get('version')==ACCOUNTING_SCHEMA
    need(len(declared)==len(set(declared)) and (set(owners)<=set(declared) if subset else set(declared)==set(owners)),
         'Accounting owner scope must retain every launched owner and failure')
    roots=[root for owner in declared for root in owners.get(owner,[])]
    if subset and (not roots or any(not root.is_dir() for root in roots) or any(
        owner.exists() for owner in set(declared)-set(owners))):
        return _failed_native_accounting(scope,base,roots,required,snapshot_phase=snapshot_phase)
    for owner in set(declared)-set(owners):need(not owner.exists(), 'Unlaunched owner has an unaccounted directory')
    need(bool(roots), 'No observed native job exists for the original scene ledger')
    try:
        per_owner=[scene_store.batch_ledger(owners[owner],batch_root=owner) for owner in declared if owner in owners]
        union=scene_store.batch_ledger(roots)
    except (OSError,ValueError) as error:
        if not subset:raise
        diagnostic=_failed_native_accounting(scope,base,roots,required,snapshot_phase=snapshot_phase)
        diagnostic['original_scene_ledger_error']=str(error)
        return diagnostic
    need(union.get('total_accounted_bytes_complete') is True and all(
        ledger.get('total_accounted_bytes_complete') is True for ledger in per_owner),
        'Existing scene ledger cannot completely account every launched attempt')
    charged={ref['file']:ref for ref in union['shared_inputs']}
    extras={};by_category={}
    for entry in scope['extra_input_refs']:
        category=entry.get('category');binding=entry.get('reference')
        need(category in ('source','config','sdk','prepared','other_runtime'), 'Explicit extra-input accounting category required')
        path=_ref_path(binding,base)
        need(type(binding.get('bytes')) is int and binding['bytes']>=0 and path.is_file() and
             path.stat().st_size==binding['bytes'] and sha256(path)==binding['sha256'],
             'Extra retained input changed or lacks exact bytes/hash')
        key=str(path)
        if key in charged:
            need(charged[key]['sha256']==binding['sha256'] and charged[key]['bytes']==binding['bytes'],
                 'Existing scene input and extra input have conflicting bindings')
        elif not any(owner in path.parents for owner in declared):
            need(key not in extras or extras[key]['sha256']==binding['sha256'], 'Conflicting retained extra input')
            extras[key]=dict(file=key,bytes=binding['bytes'],sha256=binding['sha256'])
            by_category.setdefault(category,set()).add(key)
    for key,binding in required.items():
        path=Path(key)
        need(key in charged or key in extras or any(owner in path.parents for owner in declared),
             'Required source-freeze/config input omitted from frozen accounting scope')
    overhead=sum(ledger['batch_overhead_bytes'] for ledger in per_owner)
    additional=sum(ref['bytes'] for ref in extras.values())
    envelope=None
    if 'version' in scope:
        envelope=metadata_accounting(scope,base,snapshot_phase=snapshot_phase,
            excluded_paths=[*charged,*required,*extras])
    source_envelope=None
    if 'processing_sources' in scope:
        source_envelope=processing_source_accounting(scope['processing_sources'],base,snapshot_phase=snapshot_phase,
            excluded_paths=[*charged,*required,*extras,*scope['owner_roots'],scope['operational_metadata']['root'],scope['final_seal']['path']])
    shared=union['shared_input_bytes']+overhead+additional+(0 if envelope is None else envelope['charged_bytes'])+(
        0 if source_envelope is None else source_envelope['charged_bytes'])
    total=union['job_directory_bytes']+shared
    job_rows=[dict(row,accounted_bytes_with_population_amortization=row['total_bytes']+shared/len(roots),
        within_population_budget=row['total_bytes']+shared/len(roots)<=EPISODE_LIMIT_BYTES) for row in union['jobs']]
    tier_b=scope.get('tier_b_attempt_ids')
    need(isinstance(tier_b,list) and len(set(tier_b))==len(tier_b) and set(tier_b)<=set(inventory) and
         len(tier_b)<=int(.02*len(rows)), 'Predeclared separate TierB population exceeds 2 percent')
    tier_roots={str(_ref_path(inventory[a]['owner_terminal'],base).parent/'output'/
        next(row['job_id'] for row in rows if row['attempt_id']==a)) for a in tier_b if inventory[a]['launched']}
    storage=scope.get('storage_roots')
    need(isinstance(storage,list) and {r.get('category') for r in storage}==
         {'complete_cpu_bank','backup_archive','provisioned_runtime'},
         'CPU-bank, backup/archive and provisioned-runtime storage scopes must be reported separately')
    reports=[]
    for row in storage:
        root=_path(row.get('path'),base)
        reports.append(dict(category=row['category'],root=str(root),**_storage_statistics(root),
            access='filesystem sizes only; no payload decoding or qualification',scope='separate physical storage; not amortized episode acceptance'))
    return dict(per_owner=per_owner,unique_union=union,planned_owner_count=len(declared),launched_owner_count=len(owners),
        owner_operational_bytes=overhead,metadata_envelope=envelope,processing_source_envelope=source_envelope,
        additional_inputs=list(extras.values()),additional_input_bytes=additional,
        additional_input_bytes_by_category={k:sum(extras[p]['bytes'] for p in paths) for k,paths in by_category.items()},
        total_accounted_bytes=total,accounted_bytes_per_observed_job=total/len(roots),jobs=job_rows,
        all_accounted_jobs_within_budget=union['all_job_directories_within_budget'] and
            all(row['within_population_budget'] for row in job_rows),
        tier_b_attempts=tier_b,tier_b_job_bytes=sum(row['total_bytes'] for row in union['jobs'] if row['root'] in tier_roots),
        storage_reports=reports,scope='All fixed TRAIN first100 + remainder + VAL; unique external paths charged once; no whole-project per-episode claim')


def _fingerprint(prepared):
    arrays={}
    for split in ('train','validation'):
        arrays[split]={}
        for key in shared.ARRAY_KEYS:
            value=np.ascontiguousarray(prepared[split][key],dtype='<f8')
            arrays[split][key]=dict(shape=list(value.shape),dtype=value.dtype.str,
                sha256=hashlib.sha256(value.tobytes()).hexdigest())
    return dict(arrays=arrays,index_sha256=hashlib.sha256(pose.canonical(prepared['derived_index'])).hexdigest(),
        normalization_sha256=hashlib.sha256(pose.canonical(prepared['normalization'])).hexdigest(),
        episodes=prepared['episodes'],reference_inventory=prepared['reference_inventory'],
        preparer_sha256=prepared['preparer_sha256'],reconstruction_sources=prepared['reconstruction_sources'])


def export_dataset(manifest, expected_sha256, output, qualification=None):
    """Write a new immutable index/manifest/qualification, never episode results."""
    output=Path(output).resolve()
    need(not output.exists(), 'Dataset export requires a new output directory')
    first=prepare_dataset(manifest,expected_sha256,qualification)
    second=prepare_dataset(manifest,expected_sha256,qualification)
    fingerprint=_fingerprint(first)
    need(fingerprint==_fingerprint(second), 'Fresh reconstruction arrays/indices/counts/statistics differ')
    output.mkdir(parents=True,exist_ok=False)
    written={}
    for name,value in [('index.json',first['derived_index']),('normalization.json',first['normalization']),
        ('qualification.json',first.get('qualification',dict(passed=False,training_eligible=False,
            reason='No population qualification supplied; preparation is diagnostic only'))),
        ('reconstruction.json',fingerprint)]:
        shared._save(output/name,value)
        written[name]=dict(path=str(output/name),bytes=(output/name).stat().st_size,sha256=sha256(output/name))
    result=dict(schema='m2_given_contact_dataset_bundle_v2',task_id=pose.TASK_ID,
        action_semantics_id=given.ACTION_SEMANTICS_ID,handoff_policy_id=handoff.POLICY_ID,training_eligible=first['training_eligible'],
        source_manifest=dict(path=str(Path(manifest).resolve()),sha256=expected_sha256),artifacts=written,
        reconstruction_identical=True,raw_episode_results_modified=False,test_payloads_opened=False)
    shared._save(output/'manifest.json',result)
    return result


def verify_final_seal(plan_ref,seal_ref):
    """Fresh training preflight; a serialized eligible flag never admits data."""
    plan,plan_path=_bound(plan_ref,Path.cwd());scope=plan.get('accounting_scope',{})
    need(scope.get('version')==ACCOUNTING_SCHEMA and 'processing_sources' in scope, 'Final seal requires the finite accounting/source contract')
    measured=metadata_accounting(scope,plan_path.parent,snapshot_phase='training_preflight')
    seal,path=_bound(seal_ref,plan_path.parent)
    need(path==_path(scope['final_seal']['path'],plan_path.parent) and seal.get('schema')==FINAL_SEAL_SCHEMA and
         seal.get('plan')==plan_ref and seal.get('task_id')==pose.TASK_ID and
         seal.get('action_semantics_id')==given.ACTION_SEMANTICS_ID and seal.get('handoff_policy_id')==handoff.POLICY_ID,
         'Original external final seal/plan/task binding required')
    _plan_metadata(plan.get('attempts'),plan_path.parent)
    _processing_preflight(plan,seal,plan_path.parent)
    need(seal.get('metadata_inventory')==measured['files'], 'Sealed operational metadata changed after export')
    manifest=seal.get('manifest');qualification=seal.get('qualification');bundle_ref=seal.get('bundle')
    bundle,bundle_path=_bound(bundle_ref,plan_path.parent)
    need(bundle_path==plan_path.parent/'export/manifest.json' and bundle.get('training_eligible') is True and
         bundle.get('source_manifest')==manifest and set(bundle.get('artifacts',{}))==
         {'index.json','normalization.json','qualification.json','reconstruction.json'}, 'Sealed bundle inventory/manifest differs')
    for name,binding in bundle['artifacts'].items():
        _,artifact=_bound(binding,plan_path.parent)
        need(artifact==bundle_path.parent/name and binding.get('bytes')==artifact.stat().st_size,
             'Sealed bundle artifact path/size differs')
    need(seal.get('fingerprint')==bundle['artifacts']['reconstruction.json'], 'Seal must bind the full reconstruction fingerprint')
    population,population_path=_bound(qualification,plan_path.parent)
    need(population_path==plan_path.parent/'population_qualification.json' and population.get('attempt_plan')==plan_ref and
         population.get('processing_freeze')==seal['processing_freeze'], 'Final qualification uses a different original plan/freeze')
    prepared=prepare_dataset(manifest['path'],manifest['sha256'],qualification)
    frozen_fingerprint,_=_bound(seal['fingerprint'],plan_path.parent)
    need(_fingerprint(prepared)==frozen_fingerprint, 'Fresh final arrays/indices/counts/statistics/source fingerprint differs')
    saved_qualification=shared._json(bundle_path.parent/'qualification.json')
    need({k:v for k,v in prepared['qualification'].items() if k!='accounting'}==
         {k:v for k,v in saved_qualification.items() if k!='accounting'}, 'Bundle qualification differs from fresh population audit')
    # Both caps are reopened after reconstruction, with one fixed charge each.
    after=metadata_accounting(scope,plan_path.parent,snapshot_phase='training_preflight')
    need(after['files']==measured['files'], 'Operational metadata changed during training preflight')
    source=_processing_preflight(plan,seal,plan_path.parent)
    return dict(passed=True,training_eligible=True,seal_sha256=seal_ref['sha256'],
        processing_freeze=seal['processing_freeze'],fingerprint=seal['fingerprint'],
        metadata_accounting=after,processing_source_accounting=source,qualification=prepared['qualification'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--qualification-sha256')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    need((args.qualification is None)==(args.qualification_sha256 is None), 'Qualification requires exact path and SHA256 together')
    qualification=(None if args.qualification is None else dict(path=str(args.qualification.resolve()),sha256=args.qualification_sha256))
    result=export_dataset(args.manifest,args.sha256,args.output,qualification)
    print(json.dumps(dict(output=str(args.output.resolve()),training_eligible=result['training_eligible'],reconstruction_identical=True)))


if __name__=='__main__':
    main()
