"""Finite lossless given-contact successor; prepare/launch/reconcile/export.

The operator never trains, retries, replaces a slot or edits native results.
Qualification requires the fixed eight STOP-v3/64-1 native episodes and an
exact first-five compatibility manifest; all eight are publicly decoded. A recipe is the exact normalized
run/capture source union, not a claim that a smoke or CPU fixture passed.

Run every command with ``CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
<native60-python> -B -m scripts.experiments.wam_runpod_20261027.given_formal_collection_lossless_v1``.
``prepare --settings FILE --settings-sha256 SHA --output NEW_CONFIG_ROOT``
requires allocation_proposal/allocation_mapping, qualification (first five),
qualification_all_eight, qualification_postflight, qualification_cost_ledger,
current_raw_equivalence=true (explicit current RAW provenance opt-in),
source_recipe (lossless-v1 including both exact helpers and selected-profile refs),
collector_root, base, collection_id/batch_id, run_ids (three phases),
owner_python/cpu_config_python, previous_owner, metadata_root/seal_path,
scene_store_root, processing_archive_root/processing_archive_file,
storage_roots and a prior execution_freeze. Optional extra_input_refs are
exact category/reference pairs frozen and charged before launch. prepare also
retains every legacy baseline config, one shared static base and a private
primary per selected original scene, all240 provenance-only plan rebindings and a complete
DERIVATION_MAP/COST_FORECAST. Forecast mean/max are estimates; only unavoidable
fixed costs can mathematically block a plan. Actual100MB accounting is unchanged.

The external execution descriptor is m2_given_collection_execution_freeze_v1:
root, reviewed_commit, code_commit, source_manifest, files (relative names to
absolute path/bytes/SHA), environment (python invoked_path and binary binding,
prefix, numpy_version, ruckig_version and binary binding). Its closed tree
contains CODE_COMMIT and SOURCE_MANIFEST.json; the descriptor lives outside it.

``launch_phase`` and ``reconcile`` require --plan/--plan-sha256/--phase.
``export`` additionally takes --processing-freeze/--processing-freeze-sha256;
without them it reports blocked. Final export runs with the official ACT Python
in the unique final processing root. The fixed final descriptor schema
m2_given_final_processing_freeze_v1 binds scope_id/live_root/reviewed_commit,
code_commit/source_manifest/files/archive/preparer_sha256/official_environment,
original_execution_freeze and required_unchanged_gate_sources. Archive entries
are exactly the live relative file names. The original execution root remains
immutable; only operator/dataset gate files must also match in the final root.
``verify_seal --plan ... --plan-sha256 ... --seal ... --seal-sha256 ...``
is a fresh official-environment reconstruction, suitable for training preflight.
"""
import argparse
from copy import deepcopy
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

from sim.ausc_policy import m2_given_contact_dataset as data
from sim.ausc_policy import m2_pilot_config as builder, process_guard as guard, scene_store, isaac_batch

PHASES=('first100','remainder','validation')
RECIPE_SCHEMA='m2_given_qualified_source_recipe_lossless_v1'
DERIVATION_SCHEMA='m2_given_lossless_derivation_map_v1'
SEAL_SCHEMA='m2_given_formal_collection_final_seal_v1'
BASE=Path('/workspace/wam')
LOCK_ID=(47,288098408)
CPU_CONFIG_PYTHON='/opt/wam-runtime/envs/isaacsim51/bin/python'
GATE_PYTHON='/opt/wam-runtime/envs/isaacsim60/bin/python'
COLLECTOR_ROOT='/opt/wam-runtime/code/m2-lossless-current-raw-core-v4'
# Fill only from the real reviewed closed collector/processing freezes.
# None is deliberately non-executable; no draft or synthetic source pins.
COLLECTOR_COMMIT='67b74cfd1103db78d1d63a951ea4fff16b962617'
COLLECTOR_MANIFEST_SHA256='3f6bfecfc90f6037fc3a8d5d20969844093d6ed32c8274b3c96f15a6bcc12ff0'
NUMERIC_CODEC='framed_npz_gzip_single_member_v1'
RUNTIME_EQUIVALENCE_POLICY='m2_current_context_approved_sdk_dense_equivalence_v1'
AUTHOR_PROVIDER_RELATIVE='sim/ausc_policy/layered_scene_authoring.py'
QUALIFICATION_HELPER_MODULE='scripts.experiments.wam_runpod_20261027.lossless_eight_v5'
QUALIFICATION_HELPER_RELATIVE=QUALIFICATION_HELPER_MODULE.replace('.','/')+'.py'
QUALIFICATION_HELPER_SHA256='6c1c03a9bff30c717eab642219562241c3bfc85569a2b0b398f822033f75cfe1'
POLICY_ID='m2_given_interior_parking_handoff_v3'
FAILED_PLAN=dict(path='/workspace/wam/datasets/m2-given-formal-20261027-v1/metadata/plan.json',
    sha256='4e19404b666b4f6a9b3d3d2040524c099c15457a8224c2178b292f5d75aa4a51')
HELPER_MODULE='scripts.experiments.wam_runpod_20261027.wam_given_stop3_solver_eight_operator_20261027_v3'
HELPER_SHA256='725b98a0d4865f5ce2918b737e612163123920ea6abe509e7de51bddee31ec4a'
HELPER_RELATIVE=HELPER_MODULE.replace('.','/')+'.py'
HISTORICAL_COLLECTOR_ROOT='/opt/wam-runtime/code/m2-stop-margin-5861806'
HISTORICAL_COLLECTOR_COMMIT='5861806c084dc57db389b3bf5d712c76a36f8ffb'
HISTORICAL_COLLECTOR_MANIFEST_SHA256='74887416b4fbdfc278f0f9096b10013611695870f5c825d9a2ae3a640ca67391'
HISTORICAL_QUALIFICATION_OPERATOR=dict(path='/workspace/wam/bootstrap/wam_given_stop3_solver_eight_operator_20261027_v3.py',sha256=HELPER_SHA256)
QUALIFICATION_OPERATOR=dict(path='/workspace/wam/bootstrap/lossless_eight_v5.py',sha256='6c1c03a9bff30c717eab642219562241c3bfc85569a2b0b398f822033f75cfe1')
QUALIFICATION_NAME='m2-given-lossless-eight-20261028-v5'
SELECTED_PHYSICS_PROFILE=dict(position_iterations=64,velocity_iterations=1,armature_kg_m2=.02)
QUALIFICATION_CASES=(('p001005','L3','validation'),('p000449','R3','train'),
    ('p000465','R3','train'),('p000547','L4','train'),('p000945','L5','train'),
    ('p000829','R1','train'),('p001197','L3','train'),('p001191','L4','train'))
FUTURE_SOURCE_FILES=[
    'scripts/experiments/wam_runpod_20261027/completed_authoring_checkpoint_v1.py',
    'scripts/experiments/wam_runpod_20261027/lossless_formal_support_v4.py',
    'scripts/experiments/wam_runpod_20261027/lossless_formal_support_v3.py',
    'tests/test_ausc_completed_authoring_checkpoint_v1.py',
    'tests/test_ausc_lossless_formal_support_v4.py',
    HELPER_MODULE.replace('.','/')+'.py',
    'scripts/experiments/wam_runpod_20261027/given_formal_collection_lossless_v1.py',
    QUALIFICATION_HELPER_RELATIVE,AUTHOR_PROVIDER_RELATIVE,
    'tests/test_ausc_given_formal_collection_lossless_v1.py','tests/test_ausc_given_contact_accounting_envelope.py',
    'tests/test_ausc_given_processing_source_envelope.py',
    'sim/ausc_policy/m2_given_contact_lossless_mapping.py',
    'tests/test_ausc_given_contact_lossless_mapping.py',
    *['sim/ausc_policy/m2_given_contact_'+name+'.py' for name in ('runtime','train','admission','deployment','evaluation')],
    *['tests/test_ausc_given_contact_'+name+'.py' for name in ('runtime','train','admission','deployment','evaluation')],
    'configs/ausc_policy/m2_given_contact_act_formal_v2.json']
GUARD_SHA='35305b9151a1da5046023c4462acbfb50aadc2d997b073f4acabc356bd16dc5d'
need=data.need
read=data._bound


def _require_pins():
    need(isinstance(COLLECTOR_COMMIT,str) and re.fullmatch('[0-9a-f]{40}',COLLECTOR_COMMIT) is not None and
         isinstance(COLLECTOR_MANIFEST_SHA256,str) and re.fullmatch('[0-9a-f]{64}',COLLECTOR_MANIFEST_SHA256) is not None and
         isinstance(QUALIFICATION_HELPER_SHA256,str) and re.fullmatch('[0-9a-f]{64}',QUALIFICATION_HELPER_SHA256) is not None and
         QUALIFICATION_OPERATOR==dict(path='/workspace/wam/bootstrap/lossless_eight_v5.py',sha256=QUALIFICATION_HELPER_SHA256),
         'Actual reviewed lossless source freeze and qualification pins are required')


def reference(path):
    path=data._path(str(path),Path.cwd())
    need(path.is_file(), 'Required immutable input missing: '+str(path))
    return dict(path=str(path),bytes=path.stat().st_size,sha256=data.sha256(path))


def _write(path,value):
    path=data._path(str(path),Path.cwd());path.parent.mkdir(parents=True,exist_ok=True)
    guard.write_new(path,value)
    return reference(path)


def _safe(value):
    need(isinstance(value,str) and re.fullmatch('[a-z0-9_-]{1,100}',value), 'Bounded unique run/job identity required')
    return value


def owner_deadline(n):
    need(type(n) is int and 1<=n<=100, 'Finite phase size required')
    return math.ceil(n/4)*2400+600


def _attempts(rows,mapping):
    need(isinstance(rows,list) and len(rows)==240, 'Exactly 200 TRAIN then 40 VAL attempts required')
    for row in rows:data.shared.require_membership(row.get('split'),row.get('subject_id'))
    allocations=[dict(a,split=s) for s in ('train','validation') for a in mapping['ordered_allocations'][s]]
    need(len(allocations)==240, 'Reviewed allocation size changed')
    seen={k:set() for k in ('attempt_id','episode_id','reset_id','job_id','slot_id','config','source_freeze')}
    for row,allocation in zip(rows,allocations):
        need(all(row.get(k)==allocation.get(k) for k in ('slot_id','subject_id','split')) and
             row.get('target_id')==allocation.get('target') and row.get('preclassified_reachable') is True,
             'Fixed allocation order/identity/reachability changed')
        for key in seen:
            value=row.get(key)
            if key in ('config','source_freeze'):
                need(isinstance(value,dict) and {'path','sha256'}<=set(value), 'Frozen config/source reference required')
                value=value['path']
            elif key=='slot_id':need(isinstance(value,str) and re.fullmatch('[A-Za-z0-9_.-]{1,200}',value), 'Original slot ID required')
            else:_safe(value)
            need(value not in seen[key], 'Duplicate predeclared '+key);seen[key].add(value)
        _safe(row.get('run_id'))
    groups=dict(zip(PHASES,(rows[:100],rows[100:200],rows[200:])))
    owners=[]
    for name,group in groups.items():
        ids={r['run_id'] for r in group};need(len(ids)==1, 'Each fixed phase must have one distinct owner')
        owners.extend(ids)
    need(len(set(owners))==3 and len({r['subject_id'] for r in rows[:200]})==38 and
         len({r['subject_id'] for r in rows[200:]})==10, 'Distinct owners and reviewed 38/10 identity allocation required')
    return groups


def _selected_scenes(mapping):
    """Pure metadata: preserve the original first-use scene order, never count 48."""
    scenes={};roles={};ordered=[];seen=set()
    for scene in mapping.get('scenes',[]):
        key=scene.get('proposal_id')
        need(isinstance(key,str) and key not in scenes, 'Duplicate original scene proposal')
        data.shared.require_membership(scene.get('split'),scene.get('subject_id'));scenes[key]=scene
    for split in ('train','validation'):
        for allocation in mapping['ordered_allocations'][split]:
            data.shared.require_membership(split,allocation.get('subject_id'))
            key=allocation.get('proposal_id');scene=scenes.get(key,{})
            need(scene.get('subject_id')==allocation['subject_id'] and scene.get('split')==split,
                 'Selected allocation differs from original scene identity/split')
            if key in seen:continue
            for name in ('preparation_config','prepared_complete','patient_receipt'):
                ref=scene.get(name,{});path=ref.get('file',ref.get('path'))
                need(isinstance(path,str) and Path(path).is_absolute(), 'Exact absolute original prepared role required')
                data._ref_path(dict(path=path,sha256=ref.get('sha256')),Path.cwd())
                need(path not in roles, 'Original prepared scene/receipt role reused across scene identities')
                roles[path]=(key,name)
            ordered.append(deepcopy(scene));seen.add(key)
    return ordered


def _cost_forecast(extra_inputs,future_cas,smoke_roots):
    """A conservative projection, separately labelled from actual qualification.

    The full fixed path union and all three envelopes are charged. New native
    job cost is estimated from the largest actual required qualification job;
    planned240 is a forecast denominator and never an observed-job denominator.
    """
    import stat
    from sim.ausc_policy.storage_policy import episode_size,EPISODE_LIMIT_BYTES
    fixed={};physical={}
    for entry in extra_inputs:
        need(entry.get('category') in ('source','config','sdk','prepared','other_runtime'), 'Exact retained input category required')
        ref=entry['reference'];current=reference(ref['path']);mode=Path(ref['path']).stat()
        need(current==ref and stat.S_ISREG(mode.st_mode) and
             (mode.st_nlink==1 or entry['category'] in ('sdk','other_runtime')), 'Cost forecast input changed or linked')
        need(ref['path'] not in fixed or fixed[ref['path']]==ref, 'Conflicting cost input path')
        fixed[ref['path']]=deepcopy(ref);physical[(mode.st_dev,mode.st_ino)]=mode.st_size
    reserved={}
    for ref in future_cas:
        path=data._ref_path(ref,Path.cwd())
        need(type(ref.get('bytes')) is int and ref['bytes']>0 and path.is_absolute(), 'Exact future CAS byte reservation required')
        if path.exists():
            current=reference(path);need(current==ref, 'Present future CAS differs from the immutable derived scene')
            need(ref['path'] not in fixed or fixed[ref['path']]==ref, 'CAS input binding conflict')
            fixed[ref['path']]=current;mode=path.stat();physical[(mode.st_dev,mode.st_ino)]=mode.st_size
        else:
            need(ref['path'] not in reserved or reserved[ref['path']]==ref, 'Conflicting future CAS role')
            reserved[ref['path']]=deepcopy(ref)
    jobs=[dict(root=str(root),**episode_size(root)) for root in smoke_roots]
    need(jobs and len({row['root'] for row in jobs})==len(jobs), 'Actual unique qualification job measurements required')
    sizes=sorted(row['total_bytes'] for row in jobs);maximum=sizes[-1];mean=sum(sizes)/len(sizes)
    caps=dict(operational_metadata=data.METADATA_ENVELOPE_BYTES,final_seal=data.SEAL_ENVELOPE_BYTES,
        final_processing_sources=data.PROCESSING_ENVELOPE_BYTES)
    actual=sum(ref['bytes'] for ref in fixed.values());future=sum(ref['bytes'] for ref in reserved.values());envelopes=sum(caps.values())
    projected=(actual+future+envelopes+240*maximum)/240
    return dict(schema='m2_given_successor_prelaunch_cost_forecast_v1',fixed_input_path_bytes=actual,future_cas_bytes=future,
        fixed_input_inode_unique_bytes=sum(physical.values()),charged_envelopes_bytes=envelopes,envelopes=caps,
        fixed_inputs=[fixed[key] for key in sorted(fixed)],future_cas=[reserved[key] for key in sorted(reserved)],
        observed_qualification_jobs=len(jobs),smoke_whole_job_max_bytes=maximum,qualification_jobs=jobs,
        smoke_whole_job_mean_bytes=mean,smoke_whole_job_sorted_bytes=sizes,
        planned_forecast_jobs=240,formal_observed_jobs=0,projected_bytes_per_planned_job=projected,
        mean_projected_bytes_per_planned_job=(actual+future+envelopes+240*mean)/240,
        fixed_bytes_per_planned_job=(actual+future+envelopes)/240,projected_within_100MB=projected<=EPISODE_LIMIT_BYTES,
        fixed_cost_mathematically_exceeds_100MB=(actual+future+envelopes)/240>EPISODE_LIMIT_BYTES,
        qualification=False,training_eligible=False,
        forecast_artifact_cost='This forecast is charged by exact path/SHA in the final actual ledger; its own bytes are excluded only from this conservative prelaunch lower bound',
        limitation='Largest measured qualification job is a forecast assumption, never a guarantee for unobserved native jobs; original actual ledger gates remain mandatory')


def _negative_roles(settings,output):
    """Read only this historic JSON; never traverse its artifact references."""
    old,_=read(FAILED_PLAN,Path.cwd());scope=old['accounting_scope']
    old_paths=[data._ref_path(old['config_inventory'],Path.cwd()).parent,
        data._path(scope['operational_metadata']['root'],Path.cwd()),data._path(scope['final_seal']['path'],Path.cwd()),
        data._path(scope['processing_sources']['archive_root'],Path.cwd())]
    new_paths=[output,*[data._path(settings[key],Path.cwd()) for key in
        ('metadata_root','seal_path','processing_archive_root')]]
    need(all(a!=b and a not in b.parents and b not in a.parents for a in new_paths for b in old_paths),
         'Successor cannot reuse failed population output/metadata/seal/archive roles')
    old_runs={row['run_id'] for row in old['attempts']}
    need(not old_runs.intersection(settings['run_ids'].values()), 'Successor cannot reuse failed population owner IDs')
    collection=_safe(settings.get('collection_id'));batch=_safe(settings.get('batch_id'))
    need(batch!=old.get('batch_id'), 'Successor cannot reuse failed population batch ID')
    for key,suffix in (('attempt_id',''),('job_id',''),('episode_id','-episode'),('reset_id','-reset')):
        historic={row.get(key) for row in old['attempts']}
        current={collection+'-'+str(i+1).zfill(3)+suffix for i in range(240)}
        need(not historic.intersection(current), 'Successor cannot reuse failed population '+key)
    return deepcopy(FAILED_PLAN)


def _runtime_rebinding(original,derived,source_pairs,*,capture_paths=None):
    """A complete config comparison with named provenance-only substitutions."""
    old=deepcopy(original);new=deepcopy(derived)
    if capture_paths is not None:
        need(set(capture_paths)=={'camera','calibration'}, 'Only two checkpoint capture substitutions permitted')
        for name,pair in capture_paths.items():
            a=old['fresh_expert']['raw_multimodal_capture'];b=new['fresh_expert']['raw_multimodal_capture']
            need(set(pair)=={'default','qualified'} and a[name+'_file']==pair['default']['path'] and
                 b[name+'_file']==pair['qualified']['path'] and pair['default']['sha256']==pair['qualified']['sha256'] and
                 a[name+'_sha256']==b[name+'_sha256']==pair['default']['sha256'] and
                 pair['default']['bytes']==pair['qualified']['bytes'], 'Checkpoint capture path substitution differs')
            a[name+'_file']=b[name+'_file']
    binding=_reference_rebinding(old['m2_given_contact_control']['metadata']['given_reference'],
        new['m2_given_contact_control']['metadata']['given_reference'])
    need(old.get('physics')==dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02) and
         new.get('physics')==SELECTED_PHYSICS_PROFILE, 'Only baseline32/1 to selected64/1 physics transition permitted')
    a=old['compact_scene_storage'];b=new['compact_scene_storage']
    new_roles={'scene_base','scene_authoring_receipt','layered_runtime_equivalence'}
    store_keys={'schema','dependency_policy','store_root','batch_id','sources'}
    need(set(a)==set(b)==store_keys and a['schema']==scene_store.SDK_SCHEMA and
         a['dependency_policy']==scene_store.SDK_DEPENDENCY_POLICY and
         b['schema']==scene_store.SDK_LAYER_SCHEMA and b['dependency_policy']==scene_store.SDK_LAYER_CURRENT_RAW_POLICY,
         'Only exact SDK dense-to-layered schema/policy transition permitted')
    need(not (set(a['sources'])&new_roles) and set(b['sources'])==set(a['sources'])|new_roles and
         set(source_pairs)==set(b['sources']) and
         all(source_pairs[key]==dict(original=a['sources'].get(key),derived=b['sources'][key]) for key in source_pairs),
         'Runtime source role/path/SHA/bytes differs from authenticated derivation')
    need(all(a[key]==b[key] for key in ('store_root','batch_id')),
         'Runtime CAS root/cohort/schema/dependency policy changed')
    old_compact=old['fresh_expert']['raw_multimodal_capture']['compact']
    new_compact=new['fresh_expert']['raw_multimodal_capture']['compact']
    need('numeric_container_codec' not in old_compact and new_compact.pop('numeric_container_codec',None)==NUMERIC_CODEC,
         'Only explicit reviewed lossless numeric codec opt-in from legacy baseline permitted')
    replacements={};objects=[]
    for old_value,new_value in [(pair['original'],pair['derived']) for key,pair in source_pairs.items() if key not in new_roles]+[
            (binding['original_sources'][key],binding['derived_sources'][key]) for key in binding['original_sources']]:
        objects.append((new_value,old_value))
        for key in ('file','sha256'):
            before=old_value[key];after=new_value[key]
            need(after not in replacements or replacements[after]==before, 'Ambiguous source alias/digest substitution')
            replacements[after]=before
    replacements[binding['derived_reference_sha256']]=binding['original_reference_sha256']
    replacements['given-contact-'+binding['derived_reference_sha256']]='given-contact-'+binding['original_reference_sha256']
    def revert(value):
        for after,before in objects:
            if value==after:return deepcopy(before)
        if isinstance(value,dict):return {key:revert(item) for key,item in value.items()}
        if isinstance(value,list):return [revert(item) for item in value]
        return replacements.get(value,value) if isinstance(value,str) else value
    new['physics']=old['physics']
    for key in new_roles:b['sources'].pop(key)
    b['schema']=a['schema'];b['dependency_policy']=a['dependency_policy']
    need(revert(new)==old, 'Unlisted control/geometry/camera/identity/target/reference/runtime change')
    return binding


def _derive_current_raw_metadata(authored,output_root,*,layers,store):
    """Issue opt-in JSON metadata; retain every historical prepared file role."""
    def require(condition,message):
        if not condition:raise ValueError(message)
    root=Path(output_root);headers=[]
    require(not root.exists(),'Exclusive current RAW metadata root required')
    # Entire population headers precede the issuer's first RAW read or output.
    for item in authored['prepared']:
        historical=store.source_ref(item['prepared_complete']['file'])
        require({k:historical[k] for k in ('file','sha256')}==item['prepared_complete'],
            'Historical factored COMPLETE changed')
        layers._single_file(historical,immutable=True)
        complete=store._strict_json(store.read_source(historical))
        require(not ({'scene_dependency_policy','layered_runtime_equivalence'} & set(complete)),
            'Historical COMPLETE already carries current RAW provenance')
        receipt,scene,index,metadata=store._layered_metadata(complete)
        layers.historical_prepared_sources(historical,complete,receipt,scene,index,metadata)
        headers.append((historical,complete))
    records=[]
    for index,(historical,complete) in enumerate(headers):
        directory=root/str(index).zfill(3)
        directory.mkdir(parents=True,exist_ok=False)
        sidecar=layers.issue_current_raw_equivalence(prepared_complete=historical,
            output_file=str(directory/'layered_runtime_equivalence.json'))
        require(sidecar==store.source_ref(directory/'layered_runtime_equivalence.json') and
            set(sidecar)=={'file','bytes','sha256'},'Exact current RAW issuer sidecar required')
        derived=deepcopy(complete)
        derived.update(scene_dependency_policy=store.SDK_LAYER_CURRENT_RAW_POLICY,layered_runtime_equivalence=sidecar)
        written=layers._write_new(directory/'COMPLETE.json',derived)
        records.append(dict(historical_prepared_complete=historical,layered_runtime_equivalence=sidecar,
            prepared_complete={k:written[k] for k in ('file','sha256')}))
    return records


def _verify_current_raw_metadata(authored,records,output_root):
    """Read-only reconstruction of every new header before any RAW proof."""
    from sim.ausc_policy import layered_scene_authoring as layers
    root=data._path(str(output_root),Path.cwd());expected_files=set()
    need(isinstance(records,list) and len(records)==len(authored['prepared']), 'Complete current RAW metadata population required')
    for index,(item,record) in enumerate(zip(authored['prepared'],records)):
        directory=root/str(index).zfill(3)
        need(set(record)=={'historical_prepared_complete','layered_runtime_equivalence','prepared_complete'} and
             record['historical_prepared_complete']==scene_store.source_ref(item['prepared_complete']['file']) and
             _file_ref(record['historical_prepared_complete'])==item['prepared_complete'] and
             record['layered_runtime_equivalence']==scene_store.source_ref(directory/'layered_runtime_equivalence.json') and
             record['prepared_complete']==_file_ref(scene_store.source_ref(directory/'COMPLETE.json')),
             'Exact historical/issuer/current COMPLETE metadata roles required')
        for ref in (record['historical_prepared_complete'],record['layered_runtime_equivalence'],scene_store.source_ref(directory/'COMPLETE.json')):
            layers._single_file(ref,immutable=True)
        old=scene_store._strict_json(scene_store.read_source(record['historical_prepared_complete']))
        new=scene_store._strict_json(scene_store.read_source(scene_store.source_ref(directory/'COMPLETE.json')))
        expected=deepcopy(old);expected.update(scene_dependency_policy=scene_store.SDK_LAYER_CURRENT_RAW_POLICY,
            layered_runtime_equivalence=record['layered_runtime_equivalence'])
        need(not ({'scene_dependency_policy','layered_runtime_equivalence'} & set(old)) and scene_store._exact_json(new,expected),
             'Current RAW COMPLETE must equal historical COMPLETE plus exactly two provenance keys')
        scene_store._layered_metadata(new)
        need(scene_store._prepared_files_root(new,directory)==Path(record['historical_prepared_complete']['file']).parent,
             'Current RAW prepared FILES root must remain the authenticated historical parent')
        expected_files.update((str(directory/'COMPLETE.json'),str(directory/'layered_runtime_equivalence.json')))
    actual=set()
    for path in root.rglob('*'):
        need(not path.is_symlink() and (path.is_file() or path.is_dir()), 'Invalid current RAW metadata output')
        if path.is_file():actual.add(str(path))
    need(actual==expected_files, 'Unknown or omitted current RAW metadata output')
    return records


def _authoring_request(raw_selected,baselines,provenance,output):
    """Use each authenticated baseline builder's exact normalized preparation."""
    need(len(raw_selected)==len(baselines)==len(provenance), 'Complete baseline preparation lineage required')
    selected=[]
    for raw,baseline_ref,provenance_ref in zip(raw_selected,baselines,provenance):
        need(reference(baseline_ref['path'])==baseline_ref and reference(provenance_ref['path'])==provenance_ref,
             'Exact baseline config/provenance path, bytes and SHA required')
        baseline,_=read(baseline_ref,Path.cwd());receipt,_=read(provenance_ref,Path.cwd())
        need(all(receipt.get(key)==raw[key] for key in ('preparation_config','prepared_complete','patient_receipt')),
             'Baseline builder provenance differs from exact raw selected preparation')
        runtime=receipt.get('runtime_preparation',{})
        need(set(runtime)=={'path','bytes','sha256'} and reference(runtime['path'])==runtime and
             baseline['compact_scene_storage']['sources']['preparation_config']==dict(file=runtime['path'],bytes=runtime['bytes'],sha256=runtime['sha256']) and
             baseline.get('time_limit_seconds')==2400,
             'Exact authenticated baseline runtime preparation at 2400 seconds required')
        original,_=read(dict(path=raw['preparation_config']['file'],sha256=raw['preparation_config']['sha256']),Path.cwd())
        normalized,_=read(runtime,Path.cwd());expected=deepcopy(original)
        expected.update(base=baseline['base'],time_limit_seconds=2400)
        need(scene_store._exact_json(normalized,expected), 'Preparation changed beyond declared base/deadline normalization')
        selected.append(dict(raw,preparation_config=_file_ref(runtime)))
    return dict(selected=selected,raw_selected=deepcopy(raw_selected),baseline_provenance=deepcopy(provenance),
        output_root=str(output/'derived'),current_raw_equivalence=True,current_raw_output_root=str(output/'current_raw'),
        iteration_spec=deepcopy(SELECTED_PHYSICS_PROFILE),baseline_configs=deepcopy(baselines))


def _derive_population(mapping,allocated,settings,output):
    """Keep baseline configs and exact original numerical plans for every slot."""
    _require_pins()
    need(settings.get('current_raw_equivalence') is True, 'Explicit current RAW equivalence opt-in required')
    if settings.get('completed_authoring_checkpoint') is not None:
        return _recover_population(mapping,allocated,settings,output)
    selected=_selected_scenes(mapping);original={scene['proposal_id']:scene for scene in selected}
    baselines=[];provenance=[]
    for i,a in enumerate(allocated):
        episode=_safe(settings['collection_id']+'-'+str(i+1).zfill(3))+'-episode'
        built=_build_episode(original[a['proposal_id']],a,settings,output,episode)
        ref=_write(output/'baseline'/(str(i+1).zfill(3)+'.json'),built['config'])
        provenance.append(_write(output/'baseline'/(str(i+1).zfill(3)+'.provenance.json'),built['receipt']))
        baselines.append(ref)
    # One full-population operation is necessary to identify shared static CAD.
    # Provider authenticates ALL selected metadata before any USD/body payload.
    authoring_root=data._path(str(output/'derived'),Path.cwd())
    need(not authoring_root.exists(), 'Exclusive new batch authoring root required')
    first=[next(j for j,a in enumerate(allocated) if a['proposal_id']==scene['proposal_id']) for scene in selected]
    request=_authoring_request([{key:_file_ref(scene[key]) for key in
        ('preparation_config','prepared_complete','patient_receipt')} for scene in selected],
        [baselines[j] for j in first],[provenance[j] for j in first],output)
    evidence=_frozen_helper('author_population',request);authored=evidence['result']
    need(isinstance(authored.get('prepared'),list) and len(authored['prepared'])==len(selected),
         'Batch author must return the complete ordered selected population')
    parity_ref=_write(output/'SDK_RUNTIME_EQUIVALENCE.json',dict(schema='m2_given_lossless_sdk_cpu_receipt_v1',
        authoring_request=request,cpu_execution=evidence['cpu_execution'],
        scene_authoring_receipt=authored.get('receipt'),proofs=authored.get('runtime_equivalence')))
    current=evidence['current_raw_prepared']
    need(len(current)==len(selected), 'Full current RAW metadata population required')
    derived={};lineages=[]
    for scene,selection,result,record in zip(selected,request['selected'],authored['prepared'],current):
        need(result.get('original')==selection, 'CPU batch author changed selected scene order or original roles')
        new=deepcopy(scene)
        for key in ('preparation_config','prepared_complete','patient_receipt'):new[key]=_file_ref(result[key])
        # Keep original identity/template metadata; bind every concrete output/input.
        new['prepared_complete']=deepcopy(record['prepared_complete'])
        new['derivation']=result;new['current_raw']=deepcopy(record)
        new['raw_preparation_config']=reference(_file_ref(scene['preparation_config'])['file'])
        derived[scene['proposal_id']]=new
        lineages.append(dict(proposal_id=scene['proposal_id'],original=scene,derived=new))
    class BoundPlanContext:
        def read(self,ref):return read(dict(path=ref['file'],sha256=ref['sha256']),Path.cwd())[0]
        def write(self,path,value):return _write(path,value)
    plans=[];plan_lineages=[];helper=_helper()
    for i,(a,baseline) in enumerate(zip(allocated,baselines)):
        scene=original[a['proposal_id']];new=derived[a['proposal_id']]
        plan=helper.rebind_geometric_plan(_file_ref(a['fresh_plan']),old_patient_receipt=_file_ref(scene['patient_receipt']),
            new_patient_receipt=_file_ref(new['patient_receipt']),output=output/'plans'/(str(i+1).zfill(3)+'.json'),context=BoundPlanContext())
        plans.append(dict(a,fresh_plan=_file_ref(plan),baseline_config=baseline,baseline_provenance=provenance[i]))
        plan_lineages.append(dict(slot_id=a['slot_id'],original=_file_ref(a['fresh_plan']),derived=_file_ref(plan),
            proposal_id=a['proposal_id'],baseline_config=baseline,baseline_provenance=provenance[i]))
    report=dict(schema=DERIVATION_SCHEMA,author_helper=deepcopy(settings['author_helper']),
        selected_profile=deepcopy(helper.SELECTED_PROFILE),profile=deepcopy(SELECTED_PHYSICS_PROFILE),
        unique_original_scenes=len(selected),scenes=lineages,plans=plan_lineages,numerical_plans_unchanged=True,
        authoring_request=request,layered_authoring=authored,current_raw_prepared=current,cpu_execution=evidence['cpu_execution'],
        runtime_equivalence_receipt=parity_ref,
        physics_executed=False,training_eligible=False)
    return derived,plans,report


def _completed_authoring(settings,mapping,output):
    from scripts.experiments.wam_runpod_20261027 import completed_authoring_checkpoint_v1 as checkpoint
    binding=settings.get('completed_authoring_checkpoint')
    need(isinstance(binding,dict) and set(binding)=={'path','bytes','sha256'} and reference(binding['path'])==binding,
         'Bound completed-authoring checkpoint required')
    value,_=read(binding,Path.cwd())
    allocated=[dict(a,split=s) for s in ('train','validation') for a in mapping['ordered_allocations'][s]]
    result=checkpoint.reconstruct(value,_selected_scenes(mapping),allocated,output)
    need(reference(binding['path'])==binding and settings.get('source_recipe')==value['bindings']['original_recipe'],
         'Checkpoint or original qualified recipe binding changed')
    result.update(checkpoint=value,reference=binding)
    return result


def _checkpoint_metadata(value,selected,allocated,result):
    """All saved metadata identities and plans precede opaque payload hashing."""
    scenes={s['proposal_id']:s for s in selected};request=result['authoring_request'];rows=value['plans'];root=Path(value['source_root'])
    originals=[{k:_file_ref(scenes[a['proposal_id']][k]) for k in ('preparation_config','prepared_complete','patient_receipt')} for a in allocated]
    # Authenticate every baseline/provenance, including slots not used as the
    # first-use authoring input. This invokes only bound JSON reads.
    _authoring_request(originals,[r['baseline_config'] for r in rows],[r['baseline_provenance'] for r in rows],root)
    byid={s['proposal_id']:prepared for s,prepared in zip(selected,result['authored']['prepared'])}
    original_settings,_=read(value['bindings']['original_settings'],Path.cwd())
    for i,(row,a) in enumerate(zip(rows,allocated)):
        old,_=read(dict(path=row['original']['file'],sha256=row['original']['sha256']),Path.cwd())
        new,_=read(row['derived'],Path.cwd());restored=deepcopy(new)
        restored['patient_provenance']['binding']=old['patient_provenance']['binding']
        need(restored==old and old['patient_provenance']['binding']==_file_ref(scenes[a['proposal_id']]['patient_receipt']) and
             new['patient_provenance']['binding']==byid[a['proposal_id']]['patient_receipt'], 'Saved rebound plan numerical/provenance changed')
        baseline,_=read(row['baseline_config'],Path.cwd());provenance,_=read(row['baseline_provenance'],Path.cwd())
        meta=baseline['m2_given_contact_control']['metadata']
        need(all(meta[k]==a[k] for k in ('subject_id','split')) and meta['target_id']==a['target'] and
             meta['episode_id']==original_settings['collection_id']+'-'+str(i+1).zfill(3)+'-episode' and
             provenance['fresh_plan']==row['original'] and provenance['subject_id']==a['subject_id'] and provenance['split']==a['split'],
             'Saved baseline/provenance fixed identity changed')
    first=[next(r for r in rows if r['proposal_id']==s['proposal_id']) for s in selected]
    expected=_authoring_request([{k:_file_ref(s[k]) for k in ('preparation_config','prepared_complete','patient_receipt')} for s in selected],
        [r['baseline_config'] for r in first],[r['baseline_provenance'] for r in first],root)
    need(request==expected and result['cpu_execution']==_authoring_cpu_binding(expected), 'Original authoring request/script/provider/executable changed')
    _verify_current_raw_metadata(result['authored'],result['current_raw_prepared'],root/'current_raw')
    _checkpoint_source(value,original_settings)


def _checkpoint_source(value,original_settings):
    """Reopen actual E4 evidence without representing E5 as the old author."""
    from scripts.experiments.wam_runpod_20261027 import completed_authoring_checkpoint_v1 as checkpoint
    historical=_frozen_execution(value['bindings']['original_execution_freeze'])
    descriptor,_=read(value['bindings']['original_execution_freeze'],Path.cwd())
    archive,_=read(value['bindings']['historical_archive_receipt'],Path.cwd())
    need(historical['root']==checkpoint.E4_ROOT and descriptor['reviewed_commit']==checkpoint.E4_COMMIT and
         original_settings['execution_freeze']==value['bindings']['original_execution_freeze'] and
         original_settings['source_recipe']==value['bindings']['original_recipe'] and
         archive.get('schema')=='wam_committed_runtime_closed_freeze_v3' and archive.get('commit')==checkpoint.E4_COMMIT and
         archive.get('code')==checkpoint.E4_ROOT and archive.get('source_manifest_sha256')==value['bindings']['historical_source_manifest']['sha256'] and
         archive.get('full_source_archive')==value['bindings']['historical_archive']['path'] and
         archive.get('full_source_archive_sha256')==value['bindings']['historical_archive']['sha256'] and
         archive.get('full_source_archive_bytes')==value['bindings']['historical_archive']['bytes'] and
         archive.get('all_selected_git_blobs_verified') is True and archive.get('closed_root_inventory') is True,
         'Original E4 source/archive/actual receipt provenance changed')


def _checkpoint_settings(settings,context):
    """No symbolic namespace normalization or unrelated settings override."""
    original,_=read(context['checkpoint']['bindings']['original_settings'],Path.cwd())
    restored=deepcopy(settings);restored.pop('completed_authoring_checkpoint',None)
    for key in ('execution_freeze','cpu_config_python','run_ids','metadata_root','seal_path','processing_archive_root','processing_archive_file','extra_input_refs'):
        need(key in original and key in restored, 'Complete original and recovered settings required')
        restored[key]=deepcopy(original[key])
    need(scene_store._exact_json(restored,original), 'Checkpoint settings changed outside the named CPU/assembly/cost fields')


def _checkpoint_author(context,executed):
    """Historical E4 provenance and current E5 packaged code are distinct roles."""
    from scripts.experiments.wam_runpod_20261027 import completed_authoring_checkpoint_v1 as checkpoint
    _historical_author(executed)  # close the actual current E5 implementation
    binding=context['checkpoint']['bindings']['original_execution_freeze']
    original=_frozen_execution(binding)
    descriptor,_=read(binding,Path.cwd())
    need(original['root']==checkpoint.E4_ROOT and descriptor['reviewed_commit']==checkpoint.E4_COMMIT,
         'Checkpoint historical E4 execution source changed')
    helper=original['files'][HELPER_RELATIVE]
    need(helper==reference(Path(checkpoint.E4_ROOT)/HELPER_RELATIVE) and helper['sha256']==HELPER_SHA256 and
         reference(_helper().__file__)['sha256']==HELPER_SHA256, 'Exact historical/current packaged author helper required')
    context['historical_execution']=original
    return deepcopy(helper)


def _recover_population(mapping,allocated,settings,output):
    """Reuse actual completed outputs; no provider/sidecar issuer is invoked."""
    context=settings['_completed_authoring_context'];saved=context['checkpoint']
    selected=_selected_scenes(mapping);authored=context['authored'];current=context['current_raw_prepared']
    derived={};lineages=[]
    for scene,result,record in zip(selected,authored['prepared'],current):
        new=deepcopy(scene)
        for key in ('preparation_config','prepared_complete','patient_receipt'):new[key]=_file_ref(result[key])
        new.update(prepared_complete=deepcopy(record['prepared_complete']),derivation=deepcopy(result),
            current_raw=deepcopy(record),raw_preparation_config=reference(_file_ref(scene['preparation_config'])['file']))
        derived[scene['proposal_id']]=new;lineages.append(dict(proposal_id=scene['proposal_id'],original=scene,derived=new))
    plans=[];lineage=[]
    for i,(a,entry) in enumerate(zip(allocated,saved['plans'])):
        source=entry['derived'];target=output/'plans'/(str(i+1).zfill(3)+'.json');target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(Path(source['path']).read_bytes());stream.flush();os.fsync(stream.fileno())
        copied=reference(target)
        need((copied['bytes'],copied['sha256'])==(source['bytes'],source['sha256']), 'Recovered plan must be byte-identical')
        plans.append(dict(a,fresh_plan=_file_ref(copied),baseline_config=entry['baseline_config'],baseline_provenance=entry['baseline_provenance']))
        lineage.append(dict(slot_id=a['slot_id'],proposal_id=a['proposal_id'],original=_file_ref(a['fresh_plan']),derived=_file_ref(copied),
            baseline_config=entry['baseline_config'],baseline_provenance=entry['baseline_provenance']))
    report=dict(schema=DERIVATION_SCHEMA,author_helper=deepcopy(settings['author_helper']),selected_profile=deepcopy(_helper().SELECTED_PROFILE),
        profile=deepcopy(SELECTED_PHYSICS_PROFILE),unique_original_scenes=len(selected),scenes=lineages,plans=lineage,numerical_plans_unchanged=True,
        authoring_request=deepcopy(context['authoring_request']),layered_authoring=deepcopy(authored),current_raw_prepared=deepcopy(current),
        cpu_execution=deepcopy(context['cpu_execution']),runtime_equivalence_receipt=deepcopy(context['runtime_equivalence_receipt']),
        completed_authoring_checkpoint=deepcopy(context['reference']),physics_executed=False,training_eligible=False)
    return derived,plans,report


def _source_pairs(baseline,config,scene,built):
    """Close every builder source role against the reviewed author output."""
    before=baseline['compact_scene_storage']['sources'];after=config['compact_scene_storage']['sources'];result=scene['derivation']
    need(set(result['original'])=={'preparation_config','prepared_complete','patient_receipt'} and
         all(_file_ref(before[key])==result['original'][key] for key in ('preparation_config','prepared_complete','patient_receipt')) and
         before['scene']['sha256']==result['baseline_scene_sha256'],
         'Baseline prepared roles differ from CPU author input')
    original_prep,_=read(scene['raw_preparation_config'],Path.cwd())
    need(reference(scene['raw_preparation_config']['path'])==scene['raw_preparation_config'], 'Original raw preparation lineage changed')
    baseline_prep,_=read(dict(path=before['preparation_config']['file'],sha256=before['preparation_config']['sha256']),Path.cwd())
    expected_baseline=deepcopy(original_prep);expected_baseline.update(base=baseline['base'],time_limit_seconds=baseline['time_limit_seconds'])
    need(baseline_prep==expected_baseline, 'Baseline runtime preparation changed beyond original base/deadline')
    expected=deepcopy(before)
    for key in ('patient_receipt','physics_profile'):
        expected[key]=dict(file=result[key]['file'],**{k:v for k,v in reference(result[key]['file']).items() if k!='path'})
    record=scene['current_raw']
    need(record['historical_prepared_complete']==scene_store.source_ref(result['prepared_complete']['file']) and
         scene['prepared_complete']==record['prepared_complete'], 'Exact historical/current COMPLETE lineage required')
    expected['prepared_complete']=scene_store.source_ref(record['prepared_complete']['file'])
    expected['layered_runtime_equivalence']=deepcopy(record['layered_runtime_equivalence'])
    runtime_prep=built['receipt']['runtime_preparation']
    a,_=read(runtime_prep,Path.cwd());b,_=read(dict(path=result['preparation_config']['file'],sha256=result['preparation_config']['sha256']),Path.cwd())
    need(a==b and runtime_prep==reference(result['preparation_config']['file']),
         'Current RAW runtime must retain exact historical preparation without base/deadline rewrite')
    expected['preparation_config']=dict(file=runtime_prep['path'],bytes=runtime_prep['bytes'],sha256=runtime_prep['sha256'])
    expected['scene']=deepcopy(result['dependency_inventory']['primary'])
    expected['scene_base']=deepcopy(result['dependency_inventory']['base'])
    expected['scene_authoring_receipt']=deepcopy(result['scene_authoring_receipt'])
    root=Path(result['prepared_complete']['file']).parent
    for key,filename in {**scene_store.FILES,**scene_store.OPTIONAL}.items():
        if key in expected and key not in ('scene','physics_profile','patient_receipt'):
            ref=reference(root/filename)
            need(ref['sha256']==before[key]['sha256'] and ref['bytes']==before[key]['bytes'],
                 'Copied prepared source bytes differ from original')
            expected[key]=dict(file=ref['path'],bytes=ref['bytes'],sha256=ref['sha256'])
    need(after==expected, 'Final config source roles differ from exact original-to-derived author map')
    return {key:dict(original=before.get(key),derived=expected[key]) for key in expected}


def _fixed_sources(meta,config):
    capture=config['fresh_expert']['raw_multimodal_capture']
    result={name:dict(file=capture[name+'_file'],sha256=capture[name+'_sha256']) for name in ('camera','calibration')}
    result['native_qualification']=meta['native_retainer_audit']['qualification']
    result.update({name:_file_ref(ref) for name,ref in config['compact_scene_storage']['sources'].items()
        if name=='sdk_mdl_manifest' or scene_store._sdk_file_name(name) or name=='layered_runtime_equivalence'})
    return result


def _recipe_inventory(union,code,roles,fixed):
    """Normalize only code, exact given semantic roles and known fixed inputs."""
    code=Path(code);result={}
    for original,digest in union.items():
        path=Path(original)
        if code in path.parents:
            relative=path.relative_to(code).as_posix();key='code:'+relative
            item=dict(kind='code',relative_path=relative,sha256=digest)
        else:
            names=sorted(name for name,ref in roles.items() if ref['file']==original)
            if names:
                need(all(roles[n]['sha256']==digest for n in names), 'Given semantic role digest differs from source union')
                key='given:'+','.join(names);item=dict(kind='given',roles=names)
            else:
                names=sorted(name for name,ref in fixed.items() if ref['file']==original and ref['sha256']==digest)
                need(bool(names), 'Unknown non-code source key: '+original)
                key='fixed:'+','.join(names)
                item=(dict(kind='prepared',roles=names) if names==['layered_runtime_equivalence'] else
                    dict(kind='fixed',roles=names,reference=reference(original)))
        need(key not in result, 'Duplicate normalized source identity');result[key]=item
    need(set(n for item in result.values() if item['kind']=='given' for n in item['roles'])==set(roles),
         'Exact given source union and alias relationships required')
    return result


def _conditions(config):
    need(config.get('experiment')=='geometry_qualification' and 'm2_act_deployment' not in config and
         config.get('prepare_only') is not True, 'Only original live given expert collector conditions supported')
    need(config.get('physics')==SELECTED_PHYSICS_PROFILE, 'Exact first-qualified iteration-only physics profile required')
    fresh=deepcopy(config.get('fresh_expert',{}));capture=fresh.pop('raw_multimodal_capture',{})
    fresh.pop('plans',None)
    need(fresh.get('collision_profile')=='full' and capture.get('enabled') is True and capture.get('profile')=='compact',
         'Original full collision and compact native capture required')
    compact=capture.get('compact',{})
    need(compact.get('native_image_size')==[256,144] and compact.get('max_frames')==5000 and
         compact.get('limit_bytes')==100000000 and compact.get('numeric_container_codec')==NUMERIC_CODEC,
         'Original real RGB-D size/frame/byte condition and exact lossless numeric codec required')
    compact=deepcopy(compact)
    for key in ('metadata','estimated_target_base','target_estimate_source_id'):compact.pop(key,None)
    compact.get('code_bindings',{}).pop('body',None)
    capture=deepcopy(capture);capture['compact']=compact;capture.pop('episode_id',None)
    store=config.get('compact_scene_storage',{})
    need(set(store)=={'schema','dependency_policy','store_root','batch_id','sources'}, 'Closed layered scene store required')
    sdk={name:ref for name,ref in store.get('sources',{}).items() if name=='sdk_mdl_manifest' or scene_store._sdk_file_name(name)}
    need(store.get('schema')==scene_store.SDK_LAYER_SCHEMA and store.get('dependency_policy')==scene_store.SDK_LAYER_CURRENT_RAW_POLICY and
         'sdk_mdl_manifest' in sdk and len(sdk)>1 and {'scene_base','scene_authoring_receipt','layered_runtime_equivalence'}<=set(store.get('sources',{})),
         'Qualified layered SDK schema/policy/manifest/base/authoring sources required')
    return dict(top_keys=sorted(config),physics=deepcopy(config['physics']),fresh=fresh,capture=capture,isaac_python=config.get('isaac_python'),
        scene_dependencies=dict(schema=store['schema'],dependency_policy=store['dependency_policy'],sources=sdk))


def _qualification_headers(five_ref,eight_ref):
    """All fixed identities and the exact public subset precede any leaf read."""
    five,five_path=read(five_ref,Path.cwd());eight,eight_path=read(eight_ref,Path.cwd())
    for value,count in ((five,5),(eight,8)):
        need(value.get('schema')==data.MANIFEST_SCHEMA and value.get('task_id')==data.pose.TASK_ID and
             value.get('action_semantics_id')==data.given.ACTION_SEMANTICS_ID and
             value.get('handoff_policy_id')==POLICY_ID==data.handoff.POLICY_ID and
             value.get('expert_smoke_only') is True and value.get('cpu_diagnostic_only') is not True and
             value.get('training_eligible') is False and isinstance(value.get('episodes'),list) and
             len(value['episodes'])==count, 'Actual STOP-v3 fixed eight qualification and public five subset required')
    rows=eight['episodes']
    need(five_path.parent==eight_path.parent and five['episodes']==rows[:5] and
         [(row.get('subject_id'),row.get('target_id'),row.get('split')) for row in rows]==list(QUALIFICATION_CASES),
         'Original ordered eight identities/references and exact public first-five subset required')
    return rows,eight_path.parent


def _helper():
    helper=importlib.import_module(HELPER_MODULE)
    need(reference(helper.__file__)['sha256']==HELPER_SHA256 and str(helper.CODE)==HISTORICAL_COLLECTOR_ROOT and
         helper.COMMIT==HISTORICAL_COLLECTOR_COMMIT and helper.MANIFEST_SHA==HISTORICAL_COLLECTOR_MANIFEST_SHA256,
         'Reviewed immutable plan rebinding/selected-profile helper bytes changed')
    return helper


def _historical_author(executed):
    """Private: consume this call's authenticated original execution freeze.

    The recorded author stays at its original absolute path. A final-root
    implementation can verify it only with identical reviewed helper bytes in
    its own closed source tree; equal-content aliases never replace either role.
    """
    root=data._path(executed.get('root'),Path.cwd());path=root/HELPER_RELATIVE
    historical=executed.get('files',{}).get(HELPER_RELATIVE)
    need(historical==reference(path) and historical['sha256']==HELPER_SHA256 and
         str(path)!=HISTORICAL_QUALIFICATION_OPERATOR['path'], 'Exact historical author-helper role missing or changed')
    helper=_helper();current_root=Path(__file__).resolve().parents[3]
    current=reference(helper.__file__)
    need(current['path']==str(current_root/HELPER_RELATIVE) and current['sha256']==HELPER_SHA256,
         'Current helper must occupy the reviewed packaged source role')
    if current_root!=root:
        need(current_root==data._path(data.FINAL_PROCESSING_ROOT,Path.cwd()), 'Only original or final processing implementation root allowed')
        manifest,_=read(reference(current_root/'SOURCE_MANIFEST.json'),current_root)
        need(isinstance(manifest,list) and manifest, 'Closed final implementation source manifest required')
        names=[row['file'] for row in manifest]
        files,_,_,_=data._source_tree(current_root,names+['CODE_COMMIT','SOURCE_MANIFEST.json'])
        need(set(files)==set(names)|{'CODE_COMMIT','SOURCE_MANIFEST.json'} and
             all(files[row['file']]['sha256']==row['sha256'] for row in manifest), 'Final implementation source inventory changed')
    return deepcopy(historical)


def _qualification_helper(executed):
    """Keep the packaged implementation and external bootstrap as distinct roles."""
    _require_pins()
    need(isinstance(executed,dict), 'Authenticated original execution freeze required')
    root=data._path(executed.get('root'),Path.cwd());historical=root/QUALIFICATION_HELPER_RELATIVE
    binding=executed.get('files',{}).get(QUALIFICATION_HELPER_RELATIVE)
    need(binding==reference(historical) and binding['sha256']==QUALIFICATION_HELPER_SHA256,
         'Exact historical packaged lossless qualification helper required')
    current_root=Path(__file__).resolve().parents[3]
    need(current_root in (root,Path(data.FINAL_PROCESSING_ROOT)) and
         reference(current_root/QUALIFICATION_HELPER_RELATIVE)['sha256']==QUALIFICATION_HELPER_SHA256,
         'Current packaged qualification source must authenticate before import')
    # Close a final implementation tree before executing its helper module.
    _historical_author(executed)
    helper=importlib.import_module(QUALIFICATION_HELPER_MODULE)
    current=reference(helper.__file__)
    need(current['path']==str(current_root/QUALIFICATION_HELPER_RELATIVE) and current['sha256']==QUALIFICATION_HELPER_SHA256 and
         current_root in (root,Path(data.FINAL_PROCESSING_ROOT)) and
         str(helper.CODE)==COLLECTOR_ROOT and helper.COMMIT==COLLECTOR_COMMIT and helper.MANIFEST_SHA==COLLECTOR_MANIFEST_SHA256 and
         helper.NAME==QUALIFICATION_NAME and helper.OUT==BASE/'configs'/QUALIFICATION_NAME and
         helper.RUN==BASE/'runs/ausc_policy'/QUALIFICATION_NAME and
         helper.POST==BASE/'diagnostics'/(QUALIFICATION_NAME+'-postflight') and
         helper.NUMERIC_CODEC==NUMERIC_CODEC and helper.CASES==_helper().CASES,
         'Current packaged lossless qualification helper role/bytes/frozen protocol changed')
    return helper


def _reference_rebinding(original,derived):
    before=data.pose.GivenReference.from_snapshot(original);after=data.pose.GivenReference.from_snapshot(derived)
    old=before.snapshot();new=after.snapshot();keys=tuple(old['sources'])
    for sources in (old['sources'],new['sources']):
        for ref in sources.values():
            need(set(ref)=={'file','sha256'} and Path(ref['file']).is_absolute(), 'Exact prescribed JSON source role required')
            data._ref_path(dict(path=ref['file'],sha256=ref['sha256']),Path.cwd())
    need(all((old['sources'][a]==old['sources'][b])==(new['sources'][a]==new['sources'][b]) for a in keys for b in keys),
         'Original given reference source alias relation changed')
    new['sources']=old['sources']
    need(new==old, 'Numerical prescribed pose/base/availability/target or reference semantics changed')
    return dict(original_reference_sha256=before.sha256,derived_reference_sha256=after.sha256,
        original_sources=old['sources'],derived_sources=deepcopy(derived['sources']))


def _qualification_origins(rows,contexts,configs,executed=None):
    helper=_qualification_helper(executed)
    need(len(rows)==len(contexts)==len(configs)==len(helper.CASES)==8, 'Complete original eight projections required')
    for row,context,config,case in zip(rows,contexts,configs,helper.CASES):
        need(row.get('job_id')==case['job_id'] and row.get('run_id')==helper.NAME and
             row.get('config',{}).get('path')==str(helper.OUT/(case['job_id']+'.json')),
             'All-eight case/config/owner differs from reviewed fixed qualification')
        old,_=read(dict(path=case['original_config']['file'],sha256=case['original_config']['sha256']),Path.cwd())
        meta=old['m2_given_contact_control']['metadata']
        need(meta['reference_sha256']==case['reference_sha256'], 'Original fixed prescribed reference changed')
        binding=_reference_rebinding(meta['given_reference'],context[2]['given_reference'])
        need(binding['original_reference_sha256']==case['reference_sha256'] and
             binding['derived_reference_sha256']==context[2]['reference_sha256'], 'All-eight original reference lineage differs')
        replacements={}
        for key,source in binding['derived_sources'].items():
            original=binding['original_sources'][key]
            for part in ('file','sha256'):
                need(source[part] not in replacements or replacements[source[part]]==original[part],
                     'All-eight source alias substitution differs')
                replacements[source[part]]=original[part]
        replacements[binding['derived_reference_sha256']]=binding['original_reference_sha256']
        helper.verify_runtime_projection(old,config,new_episode=row['episode_id'],selected=(64,1),replacements=replacements)


def _qualification_contexts(five_ref,eight_ref):
    """Authenticate every fixed qualification identity before source bytes."""
    rows,base=_qualification_headers(five_ref,eight_ref)
    data._declarations(rows,base)
    contexts=[data._metadata(row,base) for row in rows]
    return rows,base,contexts


def _qualification_evidence(five_ref,eight_ref,executed):
    rows,base,contexts=_qualification_contexts(five_ref,eight_ref)
    evidence=[data._native_evidence(row,base,context[2]) for row,context in zip(rows,contexts)]
    data._reset_preflight(rows,base)
    _qualification_origins(rows,contexts,[item[1] for item in evidence],executed)
    return rows,base,contexts,evidence


def _qualified_eight_metadata(five_ref,eight_ref,executed=None):
    """Reconstruct native JSON/source unions without episode or USD decoding."""
    rows,base,contexts,evidence=_qualification_evidence(five_ref,eight_ref,executed)
    checked=[]
    for row,context,(run,config,freeze) in zip(rows,contexts,evidence):
        capture=data.shared._json(context[0]/'manifest.json');union=dict(run['runtime_sources_sha256'])
        for key,digest in capture['runtime_sources_sha256'].items():
            need(key not in union or union[key]==digest, 'Conflicting actual source union');union[key]=digest
        data._source_freeze(row['source_freeze'],base,union)
        checked.append(dict(row=row,meta=context[2],config=config,freeze=freeze,union=union))
    return checked


def _qualified_eight(five_ref,eight_ref,executed=None):
    rows,base,contexts,evidence=_qualification_evidence(five_ref,eight_ref,executed)
    checked=[]
    for row,context,(run,config,freeze) in zip(rows,contexts,evidence):
        # The exact reader checks native physics, contact, source union, actual
        # stop/settling and complete capture; no serialized passed flag is used.
        samples,receipt,meta=data.read_episode(row,base)
        del samples
        need(type(receipt.get('windows')) is int and receipt['windows']>0,
             'Each freshly decoded qualification episode requires positive integer windows')
        capture=data.shared._json(context[0]/'manifest.json');union=dict(run['runtime_sources_sha256'])
        for key,digest in capture['runtime_sources_sha256'].items():
            need(key not in union or union[key]==digest, 'Conflicting actual source union');union[key]=digest
        data._source_freeze(row['source_freeze'],base,union)
        checked.append(dict(row=row,meta=meta,config=config,freeze=freeze,union=union,receipt=receipt))
    return checked


def _collector(code):
    _require_pins()
    code=data._path(str(code),Path.cwd())
    need(str(code)==COLLECTOR_ROOT, 'Exact lossless collector absolute root required')
    commit=reference(code/'CODE_COMMIT');manifest=reference(code/'SOURCE_MANIFEST.json')
    value,_=read(manifest,code);text=(code/'CODE_COMMIT').read_text().strip()
    need(text==COLLECTOR_COMMIT and manifest['sha256']==COLLECTOR_MANIFEST_SHA256,
         'Exact reviewed lossless collector commit and source manifest required')
    need(re.fullmatch('[0-9a-f]{40}',text) and isinstance(value,list) and bool(value), 'Frozen collector commit/manifest required')
    seen=set()
    for row in value:
        name=row.get('file');need(isinstance(name,str) and not Path(name).is_absolute() and name not in seen,
            'Invalid collector source manifest identity');seen.add(name)
        path=data._path(name,code)
        need(code in path.parents and path.is_file() and data.sha256(path)==row.get('sha256') and
             ('bytes' not in row or row['bytes']==path.stat().st_size), 'Frozen collector bytes changed')
    actual,_,_,_=data._source_tree(code,sorted(seen|{'CODE_COMMIT','SOURCE_MANIFEST.json'}))
    need(set(actual)==seen|{'CODE_COMMIT','SOURCE_MANIFEST.json'}, 'Collector source tree must close the exact frozen manifest')
    need(data.sha256(code/'sim/ausc_policy/process_guard.py')==GUARD_SHA, 'Original guard bytes changed')
    return dict(root=str(code),commit=text,code_commit_file=commit,source_manifest=manifest)


def _recipe_headers(qualification,all_eight,recipe,executed,*,checkpoint=None):
    author=_historical_author(executed) if checkpoint is None else _checkpoint_author(checkpoint,executed)
    declared,_=read(recipe,Path.cwd());helper=_helper();_qualification_helper(executed)
    need(declared.get('schema')==RECIPE_SCHEMA and declared.get('qualification')==qualification and
         declared.get('qualification_all_eight')==all_eight and declared.get('author_helper')==author and
         declared.get('selected_profile')==helper.SELECTED_PROFILE and
         declared.get('qualification_helper')==(executed if checkpoint is None else checkpoint['historical_execution'])['files'][QUALIFICATION_HELPER_RELATIVE] and
         declared.get('numeric_container_codec')==NUMERIC_CODEC and declared.get('current_raw_equivalence') is True and
         declared.get('scene_dependency_policy')==scene_store.SDK_LAYER_CURRENT_RAW_POLICY,
         'Successor recipe must bind full fixed eight, exact public five subset and reviewed selected-profile author')
    return declared


def _recipe_source_unions(declared,smokes,code):
    expected=None;condition=None
    for smoke in smokes:
        root=Path(smoke['freeze']['source_manifest']['path']).parent
        need(root==Path(code), 'Native execution must retain the exact qualified collector root')
        manifest,_=read(smoke['freeze']['source_manifest'],root)
        for source in manifest:
            name=source['file']
            if name.startswith('sim/') and name.endswith('.py') and name!='sim/ausc_policy/m2_given_contact_dataset.py':
                need(data.sha256(data._path(name,code))==source['sha256'],
                     'Collector module differs from qualified smoke; only reviewed dataset processing is separate')
        inventory=_recipe_inventory(smoke['union'],root,smoke['meta']['given_reference']['sources'],
            _fixed_sources(smoke['meta'],smoke['config']))
        current=_conditions(smoke['config'])
        if expected is None:expected=inventory;condition=current
        need(inventory==expected and current==condition, 'All eight actual source unions/collector conditions must agree')
    need(declared.get('inventory')==expected and declared.get('collector_conditions')==condition,
         'Recipe is not the exact actual eight-case source union/conditions')
    for item in expected.values():
        if item['kind']=='code':
            need(data.sha256(data._path(item['relative_path'],code))==item['sha256'], 'Formal collector differs from qualified smoke bytes')
    need(any(i.get('relative_path')=='sim/ausc_policy/m2_given_contact_handoff.py' for i in expected.values()),
         'Qualified source recipe must explicitly include the interior handoff helper')
    return declared,smokes


def _verified_recipe_metadata(qualification,all_eight,recipe,code,executed,*,checkpoint=None):
    declared=_recipe_headers(qualification,all_eight,recipe,executed,checkpoint=checkpoint)
    return _recipe_source_unions(declared,_qualified_eight_metadata(qualification,all_eight,executed),code)


def _verified_recipe(qualification,all_eight,recipe,code,executed,*,checkpoint=None):
    declared=_recipe_headers(qualification,all_eight,recipe,executed,checkpoint=checkpoint)
    return _recipe_source_unions(declared,_qualified_eight(qualification,all_eight,executed),code)


def _execution_entry(plan,row):
    owner=data._path(plan['run_root'],Path.cwd())/row['run_id'];out=owner/'output';job=out/row['job_id']
    launch=out/(row['job_id']+'.launch.json')
    entry=dict(attempt_id=row['attempt_id'],launched=launch.is_file())
    if not entry['launched']:
        need(not job.exists() and not (out/(row['job_id']+'.result.json')).exists(),
             'Unexplained job/output exists without authoritative launch receipt')
        return entry
    entry.update(launch_receipt=reference(launch),owner_terminal=reference(owner/'result.json'))
    for key,path in [('child_result',out/(row['job_id']+'.result.json')),('run_result',job/'COMPLETE.json')]:
        if path.is_file():entry[key]=reference(path)
    if 'run_result' not in entry and (job/'fresh_expert_result.json').is_file():
        entry['partial_result']=reference(job/'fresh_expert_result.json')
    return entry


def _closed_owner(ref,base):
    owner,_=read(ref,base);stage=owner.get('stage',{})
    need(owner.get('completed') is True and owner.get('gpu_released') is True and stage.get('passed') is True and
         stage.get('exit_code')==0 and stage.get('process_group_drained') is True and stage.get('errors')==[] and
         stage.get('signals')==[] and owner.get('resource_guard',{}).get('sha256')==GUARD_SHA,
         'Previous owner must finish normally, exit0, drain and release under original guard')
    data._time(owner.get('ended_utc'))
    return owner


def _first100_gate(plan,inventory,base,decided):
    rows=plan['attempts'][:100];success=0;times=[]
    for row in rows:
        entry=inventory[row['attempt_id']]
        need(entry.get('launched') is True, 'First100 gate requires all original physical outcomes')
        result=data._population_native(row,entry,base)
        started=data._time(result['_started_utc']);ended=data._time(result['_ended_utc'])
        need(data._time(plan['frozen_utc'])<=started<=ended, 'First100 native outcome predates freeze or has reversed clock')
        success+=int(result.get('success') is True and result.get('physical_validation') is True and result.get('errors')==[])
        times.append(ended)
    need(all(t<=data._time(decided) for t in times), 'First100 gate predates completed native outcomes')
    return dict(attempt_ids=[r['attempt_id'] for r in rows],native_successes=success,passed=success>=95,
        decided_utc=decided,denominator=100,gate_scope='Original full native physical outcomes; capture/windows/storage are separate')


def _import_collector(code):
    need(all(Path(m.__file__).resolve().parents[2]==Path(code) for m in (builder,guard,scene_store,isaac_batch)),
         'Run preparation from the frozen collector; imported collector modules differ')


def _file_ref(ref):return dict(file=ref.get('file',ref.get('path')),sha256=ref['sha256'])


def _executable(path):
    need(isinstance(path,str) and Path(path).is_absolute() and Path(path).is_file(), 'Explicit installed owner executable required')
    # Provisioned Python environments can have a symlinked executable. Bind
    # both the invoked name and actual binary, without traversing runtime trees.
    return dict(invoked_path=path,**reference(Path(path).resolve()))


def _authoring_script():
    import inspect
    helper_source=inspect.getsource(_derive_current_raw_metadata)
    return """import sys
if sys.flags.optimize != 0 or not __debug__:
 raise RuntimeError('Frozen CPU author requires optimization disabled')
import hashlib,json,os,stat
from copy import deepcopy
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['cpu_execution']['collector_root']);a=p['payload']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def closure():
 rows=json.loads((root/'SOURCE_MANIFEST.json').read_text());names=[r['file'] for r in rows]
 assert len(names)==len(set(names)) and rows
 assert all(isinstance(n,str) and not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
 assert sha(root/'SOURCE_MANIFEST.json')==p['collector']['source_manifest']['sha256']
 assert (root/'CODE_COMMIT').read_text().strip()==p['collector']['commit']
 actual=[]
 for path in root.rglob('*'):
  mode=path.lstat();assert not path.is_symlink()
  if path.is_dir():continue
  assert stat.S_ISREG(mode.st_mode) and mode.st_nlink==1;actual.append(path.relative_to(root).as_posix())
 assert set(actual)==set(names)|{'CODE_COMMIT','SOURCE_MANIFEST.json'}
 for row in rows:assert sha(root/row['file'])==row['sha256'] and ('bytes' not in row or (root/row['file']).stat().st_size==row['bytes'])
closure()
from sim.ausc_policy import layered_scene_authoring as layers,scene_store as store
from pxr import Ar
assert all(Path(m.__file__).resolve().parents[2]==root for m in (layers,store))
assert sha(layers.__file__)==p['cpu_execution']['provider']['sha256']
# Every selected scene and retained baseline header precedes provider payload.
raw_metadata=layers._population_metadata(a['raw_selected'])
metadata=layers._population_metadata(a['selected']);originals=[]
assert len(a['baseline_configs'])==len(a['baseline_provenance'])==len(a['raw_selected'])==len(a['selected'])==len(metadata)
for original_ref,provenance_ref,raw_selection,selection,raw_meta,meta in zip(a['baseline_configs'],a['baseline_provenance'],a['raw_selected'],a['selected'],raw_metadata,metadata):
 assert sha(original_ref['path'])==original_ref['sha256'] and Path(original_ref['path']).stat().st_size==original_ref['bytes']
 assert sha(provenance_ref['path'])==provenance_ref['sha256'] and Path(provenance_ref['path']).stat().st_size==provenance_ref['bytes']
 original=json.loads(Path(original_ref['path']).read_text());patient=meta[2];header=original['m2_given_contact_control']['metadata']
 provenance=json.loads(Path(provenance_ref['path']).read_text());runtime=provenance['runtime_preparation']
 assert all(provenance[key]==raw_selection[key] for key in ('preparation_config','prepared_complete','patient_receipt'))
 assert set(runtime)=={'path','bytes','sha256'} and sha(runtime['path'])==runtime['sha256'] and Path(runtime['path']).stat().st_size==runtime['bytes']
 assert selection==dict(raw_selection,preparation_config=dict(file=runtime['path'],sha256=runtime['sha256']))
 assert original['compact_scene_storage']['sources']['preparation_config']==dict(file=runtime['path'],bytes=runtime['bytes'],sha256=runtime['sha256'])
 expected=deepcopy(raw_meta[0]);expected.update(base=original['base'],time_limit_seconds=2400)
 assert original['time_limit_seconds']==2400 and store._exact_json(meta[0],expected)
 assert header['subject_id']==patient['identity'] and header['split']==patient['split']
 assert original['physics']==dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02)
 assert original['compact_scene_storage']['schema']==store.SDK_SCHEMA
 assert all({k:original['compact_scene_storage']['sources'][role][k] for k in ('file','sha256')}==selection[role]
  for role in ('prepared_complete','patient_receipt'))
 originals.append(original)
# The provider independently repeats complete selected metadata admission.
assert a['current_raw_equivalence'] is True
result=layers.author_population(selected=a['selected'],output_root=a['output_root'],iteration_spec=a['iteration_spec'])
"""+helper_source+"""
current_raw_prepared=_derive_current_raw_metadata(result,a['current_raw_output_root'],layers=layers,store=store)
assert len(result['prepared'])==len(a['selected'])==len(a['baseline_configs'])
equivalence=[]
for original,derived,record in zip(originals,result['prepared'],current_raw_prepared):
 old_binding=original['compact_scene_storage']
 binding=store.build_binding(preparation_config=derived['preparation_config'],prepared_complete=record['prepared_complete'],
  patient_receipt=derived['patient_receipt'],store_root=old_binding['store_root'],batch_id=old_binding['batch_id'],
  retainer_witness_contract=original['m2_given_contact_control']['metadata']['native_retainer_audit'],
  sdk_mdl_manifest={k:old_binding['sources']['sdk_mdl_manifest'][k] for k in ('file','sha256')})
 context=Ar.GetResolver().GetCurrentContext()
 with Ar.ResolverContextBinder(context),Ar.ResolverScopedCache():
  old=store.prepared_runtime_stage(old_binding);selected,proof=layers.apply_qualified_iterations(old)
  new=store.prepared_runtime_stage(binding)
  assert layers.scene_snapshot(selected)==layers.scene_snapshot(new), 'Complete SDK-selected private runtime differs'
  assert all(layer.anonymous for stage in (old,selected,new) for layer in stage.GetUsedLayers())
  assert not store.runtime_stage_leases()
  equivalence.append(dict(schema='m2_lossless_sdk_private_runtime_equivalence_v1',original_sources=old_binding['sources'],
   derived_sources=binding['sources'],selected_iteration_proof=proof,composed_semantics_sha256=layers.digest(layers.scene_snapshot(new)),
   SDK_selected_runtime_readback_exact=True,all_layers_private_anonymous=True,dense_copy_created=False,
   physics_executed=False,training_eligible=False))
result['runtime_equivalence']=equivalence
for original_ref in a['baseline_configs']:
 assert sha(original_ref['path'])==original_ref['sha256'] and Path(original_ref['path']).stat().st_size==original_ref['bytes']
closure()
print(json.dumps(dict(result=result,current_raw_prepared=current_raw_prepared),allow_nan=False))
"""


def _authoring_cpu_binding(payload):
    collector=_collector(COLLECTOR_ROOT);provider=reference(Path(COLLECTOR_ROOT)/AUTHOR_PROVIDER_RELATIVE)
    script=(_qualification_validation_script() if set(payload)=={'qualification_operator','prepared','qualified_probe'} else _authoring_script())
    return dict(executable=_executable(GATE_PYTHON),collector_root=COLLECTOR_ROOT,collector=collector,provider=provider,
        script_sha256=__import__('hashlib').sha256(script.encode()).hexdigest(),
        request_sha256=data.handoff.digest(payload),current_raw_equivalence=True,
        dependency_policy=scene_store.SDK_LAYER_CURRENT_RAW_POLICY,gpu_work=False,physics_advanced=False)


def _qualification_validation_script():
    # Reuse the exact closed-core authentication bootstrap, before imports.
    return _authoring_script().split('from sim.ausc_policy import layered_scene_authoring as layers,scene_store as store',1)[0]+"""import importlib.util
operator=a['qualification_operator']
assert set(operator)=={'file','bytes','sha256'}
assert Path(operator['file']).is_absolute() and not Path(operator['file']).is_symlink()
assert sha(operator['file'])==operator['sha256'] and Path(operator['file']).stat().st_size==operator['bytes']
spec=importlib.util.spec_from_file_location('wam_qualified_current_raw_eight',operator['file'])
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
assert str(helper.CODE)==str(root) and helper.COMMIT==p['collector']['commit']
assert helper.MANIFEST_SHA==p['collector']['source_manifest']['sha256']
ctx=helper.Context()
assert a['prepared']==helper.sized_ref(helper.OUT/'PREPARED.json')
prepared,batch,expected=helper.load_prepared(ctx,a['prepared']['sha256'],(64,1))
qualified=helper.authenticate_qualified_probe(ctx,prepared,a['prepared']['sha256'],a['qualified_probe']['sha256'])
assert helper.exact(qualified,a['qualified_probe'])
ctx.recheck();closure()
assert sha(operator['file'])==operator['sha256'] and Path(operator['file']).stat().st_size==operator['bytes']
print(json.dumps(dict(prepared=a['prepared'],qualified_software_probe=qualified),allow_nan=False))
"""


def _author_population_timeout(count):
    need(type(count) is int and 2<=count<=48, 'Finite CPU authoring population of 2..48 scenes required')
    return max(2400,600*count)


def _frozen_helper(operation,payload):
    """Frozen native60 CPU authoring or read-only qualification reconstruction."""
    validation=operation=='validate_qualification'
    need((validation and isinstance(payload,dict) and set(payload)=={'qualification_operator','prepared','qualified_probe'}) or
         (operation=='author_population' and isinstance(payload,dict) and
         set(payload)=={'selected','raw_selected','baseline_provenance','output_root','iteration_spec','baseline_configs','current_raw_equivalence','current_raw_output_root'} and
         payload.get('current_raw_equivalence') is True and
         all(type(payload[key]) is list for key in ('selected','raw_selected','baseline_provenance','baseline_configs')) and
         all(len(payload[key])==len(payload['selected']) for key in ('raw_selected','baseline_provenance','baseline_configs')) and
         type(payload.get('iteration_spec')) is dict and payload['iteration_spec']==SELECTED_PHYSICS_PROFILE and
         all(type(payload['iteration_spec'][key]) is type(value) for key,value in SELECTED_PHYSICS_PROFILE.items())),
         'Finite lossless batch CPU authoring operation required')
    timeout=7200 if validation else _author_population_timeout(len(payload['selected']))
    binding=_authoring_cpu_binding(payload);script=_qualification_validation_script() if validation else _authoring_script()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=COLLECTOR_ROOT,PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[binding['executable']['invoked_path'],'-B','-c',script]
    # _authoring_cpu_binding has already authenticated the actual closed core.
    request=dict(operation=operation,payload=payload,cpu_execution=binding,collector=binding['collector'])
    result=subprocess.run(command,cwd=COLLECTOR_ROOT,env=env,input=json.dumps(request,allow_nan=False),
        text=True,capture_output=True,timeout=timeout)
    need(result.returncode==0 and len(result.stdout.encode())<=16000000,
         'Frozen CPU lossless population author failed: '+result.stderr[-4096:])
    need(_authoring_cpu_binding(payload)==binding, 'Author source/executable/request changed across isolated CPU operation')
    value=json.loads(result.stdout)
    if validation:
        need(value==dict(prepared=payload['prepared'],qualified_software_probe=payload['qualified_probe']),
             'Exact current qualification/probe CPU reconstruction required')
        return dict(result=value,cpu_execution=binding)
    need(set(value)=={'result','current_raw_prepared'}, 'Exact historical author/current RAW output roles required')
    return dict(value,cpu_execution=binding)


def _frozen_execution(ref):
    value,path=read(ref,Path.cwd());root=data._path(value.get('root'),path.parent)
    need(value.get('schema')=='m2_given_collection_execution_freeze_v1' and root not in path.parents,
         'Execution gate freeze must exist outside its source tree before first execution')
    files,total,physical,inodes=data._source_tree(root,list(value.get('files',{})))
    need(files==value['files'] and reference(root/'CODE_COMMIT')==value.get('code_commit') and
         reference(root/'SOURCE_MANIFEST.json')==value.get('source_manifest') and
         (root/'CODE_COMMIT').read_text().strip()==value.get('reviewed_commit') and
         re.fullmatch('[0-9a-f]{40}',str(value.get('reviewed_commit'))), 'Executed source freeze/commit changed')
    manifest,_=read(value['source_manifest'],root);names=set()
    for row in manifest:
        name=row.get('file');need(name in files and name not in names and row.get('sha256')==files[name]['sha256'],
            'Executed source manifest row changed');names.add(name)
    need(names==set(files)-{'CODE_COMMIT','SOURCE_MANIFEST.json'}, 'Executed source manifest must close all files')
    env=value.get('environment',{})
    need(_executable(env.get('python',{}).get('invoked_path'))==env.get('python') and
         reference(env.get('ruckig',{}).get('path'))==env.get('ruckig') and env.get('ruckig_version')=='0.15.3',
         'Frozen native CPU gate interpreter/Ruckig binding changed')
    gates=['scripts/experiments/wam_runpod_20261027/given_formal_collection_lossless_v1.py','sim/ausc_policy/m2_given_contact_dataset.py']
    checkpoint_helper='scripts/experiments/wam_runpod_20261027/completed_authoring_checkpoint_v1.py'
    if checkpoint_helper in files:gates.append(checkpoint_helper)
    return dict(reference=ref,root=str(root),files=files,environment=env,
        required_unchanged_gate_sources={name:files[name]['sha256'] for name in gates})


def _execution_freeze(ref):
    frozen=_frozen_execution(ref);root=Path(frozen['root']);env=frozen['environment']
    need(Path(data.__file__).resolve().parents[2]==root and Path(__file__).resolve().parents[3]==root and
         sys.executable==GATE_PYTHON==env['python']['invoked_path'] and sys.prefix==env.get('prefix') and
         os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Run collection gate in frozen native60 CPU root with CUDA hidden')
    import ruckig
    need(ruckig.__version__=='0.15.3' and reference(Path(ruckig.__file__).resolve())==env['ruckig'] and
         data.np.__version__==env.get('numpy_version'), 'Current native CPU gate dependency differs from prior freeze')
    for name,module in list(sys.modules.items()):
        source=getattr(module,'__file__',None)
        if source and (name.startswith('sim.') or name==__name__):
            path=Path(source).resolve();need(root in path.parents and path.relative_to(root).as_posix() in frozen['files'] and
                data.sha256(path)==frozen['files'][path.relative_to(root).as_posix()]['sha256'], 'Imported gate code is outside its prior freeze')
    return frozen


def _run_builder(command,payload,code,env,output,episode,*,retain=False):
    started=guard.now();timed_out=False;result=None
    try:
        result=subprocess.run(command,cwd=code,env=env,input=json.dumps(payload),text=True,capture_output=True,timeout=300)
        stdout=result.stdout;stderr=result.stderr;returncode=result.returncode
    except subprocess.TimeoutExpired as error:
        if not retain:raise
        timed_out=True;returncode=None;stdout=error.stdout or '';stderr=error.stderr or ''
    if retain:
        directory=output/'cpu_builds';directory.mkdir(parents=True,exist_ok=True)
        refs={}
        for name,value in (('stdout',stdout),('stderr',stderr)):
            path=directory/(episode+'.'+name)
            with path.open('xb') as stream:
                stream.write(value if isinstance(value,bytes) else value.encode());stream.flush();os.fsync(stream.fileno())
            refs[name]=reference(path)
        _write(directory/(episode+'.terminal.json'),dict(schema='m2_given_checkpoint_cpu_builder_terminal_v1',
            invoked_path=command[0],collector_root=code,script_sha256=__import__('hashlib').sha256(command[3].encode()).hexdigest(),
            request_sha256=data.handoff.digest(payload),started_utc=started,ended_utc=guard.now(),timeout_seconds=300,
            timed_out=timed_out,returncode=returncode,gpu_work=False,physics_advanced=False,**refs))
    need(not timed_out and returncode==0 and len(stdout.encode())<=16000000, 'Frozen collector CPU configuration builder failed; durable evidence retained' if retain else
         'Frozen collector CPU configuration builder failed')
    return result


def _build_episode(scene,allocation,settings,output,episode):
    _require_pins()
    args={key:_file_ref(scene[key]) for key in ('preparation_config','prepared_complete','patient_receipt')}
    prep=data.shared._json(args['preparation_config']['file'])
    need(data.sha256(args['preparation_config']['file'])==args['preparation_config']['sha256'], 'Original preparation changed')
    current=scene.get('current_raw')
    if current is None:prep.update(base=settings['base'],time_limit_seconds=2400)
    digest=data.handoff.digest(prep);path=output/'preparations'/(digest+'.json')
    if current is not None:path=Path(args['preparation_config']['file'])
    elif not path.exists():_write(path,prep)
    else:need(data.shared._json(path)==prep, 'Immutable preparation content collision')
    # The builder writes its own module paths/hashes into metadata. It must
    # execute in the immutable qualified native collector, never this separate
    # processing root. This subprocess is CPU-only and advances no physics.
    script="""import json,sys
from pathlib import Path
from sim.ausc_policy import m2_pilot_config as b, scene_store as s
p=json.load(sys.stdin);root=Path(p['collector_root'])
assert all(Path(m.__file__).resolve().parents[2]==root for m in (b,s)), 'Wrong collector import root'
built=b.build_given_contact_expert(**p['builder_args'],fresh_plan=p['fresh_plan'],episode_id=p['episode'],max_frames=5000,
 collision_profile='full',retainer_witness_contract=p['native_contract'],scene_store=p['store'])
c=built['config'];c.update(base=p['base'],time_limit_seconds=2400)
c['fresh_expert']['raw_multimodal_capture']['compact']['native_image_size']=[256,144]
c['compact_scene_storage']=s.build_binding(preparation_config=p['runtime_preparation'],
 prepared_complete=p['args']['prepared_complete'],patient_receipt=p['args']['patient_receipt'],
 store_root=p['store']['root'],batch_id=p['store']['batch_id'],retainer_witness_contract=p['native_contract'],
 sdk_mdl_manifest=p['store']['sdk_mdl_manifest'])
if c['compact_scene_storage']['schema']==s.SDK_LAYER_SCHEMA:
 c['fresh_expert']['raw_multimodal_capture']['compact']['numeric_container_codec']='framed_npz_gzip_single_member_v1'
built['receipt']['compact_scene_storage']=c['compact_scene_storage']
if p['current_raw'] is not None:built['receipt']['current_raw_prepared_complete']=p['args']['prepared_complete']
s.validate_config(c)
print(json.dumps(built,allow_nan=False))
"""
    code=settings['collector_root'];executable=_executable(settings['cpu_config_python'])
    builder_args=deepcopy(args)
    if current is not None:
        need(current['prepared_complete']==args['prepared_complete'] and
             _file_ref(current['historical_prepared_complete'])==scene['derivation']['prepared_complete'],
             'Exact historical builder COMPLETE required for current RAW metadata')
        builder_args['prepared_complete']=_file_ref(current['historical_prepared_complete'])
    context=settings.get('_completed_authoring_context')
    if settings.get('completed_authoring_checkpoint') is not None:
        from scripts.experiments.wam_runpod_20261027 import completed_authoring_checkpoint_v1 as checkpoint
        need(context is not None and context['reference']==settings['completed_authoring_checkpoint'] and current is not None and
             settings['cpu_config_python']==GATE_PYTHON and settings['owner_python']==CPU_CONFIG_PYTHON,
             'Checkpoint builder requires explicit Native60 CPU and original Native51 owner')
        builder_args.update(checkpoint.capture_paths(context['checkpoint']['capture_paths'],pinned=True))
    payload=dict(args=args,builder_args=builder_args,current_raw=current,fresh_plan=_file_ref(allocation['fresh_plan']),episode=episode,base=settings['base'],
        collector_root=code,native_contract=settings['native_contract'],runtime_preparation=_file_ref(reference(path)),
        store=dict(root=settings['scene_store_root'],batch_id=settings['batch_id'],sdk_mdl_manifest=settings['sdk_mdl_manifest']))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=code,PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[executable['invoked_path'],'-B','-c',script]
    result=_run_builder(command,payload,code,env,output,episode,retain=context is not None)
    built=json.loads(result.stdout)
    if context is not None:
        capture=built['config']['fresh_expert']['raw_multimodal_capture']
        need(built['config'].get('isaac_python')==CPU_CONFIG_PYTHON, 'CPU recovery cannot change native job interpreter')
        for name,role in (('camera','camera'),('calibration','hall_calibration')):
            qualified=context['checkpoint']['capture_paths'][name]['qualified']
            need(capture[name+'_file']==qualified['path'] and capture[name+'_sha256']==qualified['sha256'] and
                 built['receipt'][role]==_file_ref(qualified), 'CPU builder camera/Hall receipt must match actual qualified capture')
    built['receipt']['collector_cpu_builder']=dict(collector_root=code,executable=executable,
        script_sha256=__import__('hashlib').sha256(script.encode()).hexdigest(),gpu_work=False,physics_advanced=False)
    built['receipt']['runtime_preparation']=reference(path)
    return built


def _freeze_sources(inventory,code,roles,fixed,output):
    files={}
    for item in inventory.values():
        if item['kind']=='code':original=Path(code)/item['relative_path'];digest=item['sha256']
        else:
            refs=roles if item['kind']=='given' else fixed
            selected=[refs[name] for name in item['roles']]
            need(all(ref==selected[0] for ref in selected), 'Qualified semantic source alias relationship changed')
            original=Path(selected[0]['file']);digest=selected[0]['sha256']
            if item['kind']=='fixed':need(reference(original)==item['reference'], 'Known fixed qualification/calibration binding changed')
        ref=reference(original);need(ref['sha256']==digest, 'Source changed before immutable archive')
        archived=output/'archive'/digest
        if not archived.exists():
            archived.parent.mkdir(parents=True,exist_ok=True)
            with archived.open('xb') as stream:stream.write(original.read_bytes())
        binding=reference(archived)
        need(binding['sha256']==digest and binding['bytes']==ref['bytes'], 'Archived source content changed')
        need(str(original) not in files, 'Duplicate original source key')
        files[str(original)]=binding
    return files


def _allowed_metadata():
    return ['plan.json','first100_gate.json','execution_inventory.json','attempt_ledger.json','episode_manifest.json',
        'population_qualification.json','population_qualification.candidate.json','reconciliation.json','final_processing_freeze.json',
        *['export/'+n for n in ('index.json','normalization.json','qualification.json','reconstruction.json','manifest.json')],
        *['phases/'+p+suffix for p in PHASES for suffix in ('.intent.json','.launch.json','.log')],
        *['operator/'+p+'.log' for p in ('prepare','reconcile-first100','reconcile-all','export',*['launch-'+p for p in PHASES])]]


def _qualification_shared_inputs(smokes):
    """Retain exactly the qualified jobs' external CAS/SDK/prepared inputs.

    These are fixed prerequisites, never additional formal observed jobs or
    a separate five-case amortization gate. The native reader ran beforehand.
    """
    roots=[data._ref_path(smoke['row']['run_result'],Path.cwd()).parent for smoke in smokes if 'row' in smoke]
    if not roots:return []  # explicit CPU boundary fixtures have no native rows
    ledger=scene_store.batch_ledger(roots)
    need(ledger.get('total_accounted_bytes_complete') is True, 'Qualified smoke shared-input ledger must be complete')
    return ledger['shared_inputs']


def _qualification_cost_metadata(settings,executed):
    """Authenticate retained JSON and complete byte/cost closure; no oracle."""
    helper=_qualification_helper(executed);summary,sp=read(settings.get('qualification_postflight'),Path.cwd())
    ledger,lp=read(settings.get('qualification_cost_ledger'),Path.cwd())
    need(sp==helper.POST/'summary.json' and lp==helper.POST/'cost_ledger.json' and
         summary.get('schema')=='m2_lossless_solver_eight_postflight_v1' and summary.get('run_id')==helper.NAME and
         summary.get('all_eight_qualified') is True and summary.get('terminal_payload_hashes_verified') is True and
         summary.get('planned_denominator')==8 and
         summary.get('eight_manifest')==_file_ref(settings['qualification_all_eight']) and
         summary.get('five_manifest')==_file_ref(settings['qualification']), 'Exact qualified eight postflight/cost evidence required')
    prepared=summary.get('prepared',{});need(prepared.get('file')==str(helper.OUT/'PREPARED.json'), 'Original eight preparation binding required')
    need(isinstance(ledger.get('files'),list), 'Complete eight prerequisite path ledger required')
    refs=[reference(sp),reference(lp),_historical_author(executed),deepcopy(executed['files'][QUALIFICATION_HELPER_RELATIVE])]
    prepared_value,_=read(dict(path=prepared['file'],sha256=prepared['sha256']),Path.cwd())
    operator=reference(data._ref_path(QUALIFICATION_OPERATOR,Path.cwd()))
    need(operator['sha256']==QUALIFICATION_HELPER_SHA256, 'New bootstrap qualification operator bytes changed')
    refs.append(operator)
    need(prepared_value.get('schema')=='m2_lossless_solver_eight_prepared_v4' and
         prepared_value.get('run_id')==helper.NAME and prepared_value.get('code')==COLLECTOR_ROOT and
         prepared_value.get('code_commit')==COLLECTOR_COMMIT and prepared_value.get('operator')==_file_ref(operator) and
         prepared_value.get('selected_profile')==helper.SELECTED_PROFILE and prepared_value.get('selected_counts')==[64,1] and
         prepared_value.get('original_cases')==list(helper.CASES) and prepared_value.get('planned_denominator')==8 and
         prepared_value.get('numeric_container_codec')==NUMERIC_CODEC and
         prepared_value.get('prior_qualification_summary')==helper.PRIOR_EIGHT_SUMMARY and
         prepared_value.get('prior_qualification_ledger')==helper.PRIOR_EIGHT_LEDGER and
         isinstance(prepared_value.get('fixed_bindings'),dict) and bool(prepared_value['fixed_bindings']),
         'Actual fixed eight author preparation/input closure required')
    qualified_probe=summary.get('qualified_software_probe')
    need(isinstance(qualified_probe,dict) and set(qualified_probe)=={'file','bytes','sha256'} and
         qualified_probe.get('file')==str(helper.OUT/'PROBE_QUALIFIED.json') and
         reference(qualified_probe['file'])==dict(path=qualified_probe['file'],bytes=qualified_probe['bytes'],sha256=qualified_probe['sha256']) and
         type(summary.get('software_probe_denominator_units')) is int and summary['software_probe_denominator_units']==0 and
         summary.get('software_probe_shared_cost_paths')==len(helper.probe_output_paths())+len(helper.PROBE_LAUNCH_FILES)+1,
         'Exact software probe prerequisite with zero physical denominator units required')
    validation=dict(qualification_operator=scene_store.source_ref(operator['path']),
        prepared=scene_store.source_ref(prepared['file']),qualified_probe=qualified_probe)
    seen={};total=0
    for item in ledger.get('files',[]):
        ref=dict(path=item['file'],bytes=item['bytes'],sha256=item['sha256'])
        need(ref['path'] not in seen and reference(ref['path'])==ref, 'Eight cost input omitted, duplicated or changed')
        seen[ref['path']]=ref;total+=ref['bytes'];refs.append(ref)
    need(seen and total==summary.get('full_input_union_unique_path_bytes'), 'Incomplete eight prerequisite cost union')
    required=dict(prepared_value['fixed_bindings'])
    for name in helper.probe_output_paths()|{str(helper.OUT/n) for n in (*helper.PROBE_LAUNCH_FILES,'PROBE_QUALIFIED.json')}:
        required[name]=data.sha256(name)
    prior,prior_path=read(dict(path=helper.PRIOR_EIGHT_SUMMARY['file'],sha256=helper.PRIOR_EIGHT_SUMMARY['sha256']),Path.cwd())
    old_ledger,old_path=read(dict(path=helper.PRIOR_EIGHT_LEDGER['file'],sha256=helper.PRIOR_EIGHT_LEDGER['sha256']),Path.cwd())
    old_files=old_ledger.get('files',[])
    need(prior.get('all_eight_qualified') is True and prior.get('terminal_payload_hashes_verified') is True and old_files and
         len({item['file'] for item in old_files})==len(old_files) and
         sum(item['bytes'] for item in old_files)==prior.get('full_input_union_unique_path_bytes'),
         'Complete immutable original eight prerequisite cost union required')
    for item in old_files:
        need(item['file'] not in required or required[item['file']]==item['sha256'], 'Conflicting original qualification cost binding')
        required[item['file']]=item['sha256']
    required[str(prior_path)]=helper.PRIOR_EIGHT_SUMMARY['sha256'];required[str(old_path)]=helper.PRIOR_EIGHT_LEDGER['sha256']
    for path in helper.OUT.rglob('*'):
        need(not path.is_symlink(), 'Link in eight author output scope')
        if path.is_file():required[str(path)]=data.sha256(path)
        else:need(path.is_dir(), 'Special eight author output')
    need(helper.RUN.is_dir(), 'Actual new eight owner output tree required')
    for path in helper.RUN.rglob('*'):
        need(not path.is_symlink(), 'Link in actual eight owner output scope')
        if path.is_file():required[str(path)]=data.sha256(path)
        else:need(path.is_dir(), 'Special actual eight owner output')
    need(all(name in seen and seen[name]['sha256']==digest for name,digest in required.items()),
         'Eight prerequisite ledger omits an authenticated preparation/source/output path')
    refs.append(reference(prepared['file']))
    # Review JSONs themselves are additional real copies outside the measured union.
    for path in helper.POST.rglob('*'):
        if path.is_file():refs.append(reference(path))
    return refs,validation


def _qualification_cost_inputs(settings,executed):
    return _qualification_cost_metadata(settings,executed)[0]


def _qualification_costs(settings,executed):
    """Full fresh qualification subprocess/SDK checks plus complete costs."""
    refs,validation=_qualification_cost_metadata(settings,executed)
    _frozen_helper('validate_qualification',validation)
    after,after_validation=_qualification_cost_metadata(settings,executed)
    need(after_validation==validation and {r['path']:r for r in after}=={r['path']:r for r in refs},
         'Qualification cost/input closure changed during fresh validation')
    return refs


def _original_headers(mapping,allocated):
    """Authenticate ALL selected identities/manifest headers before first body."""
    refs={};scenes={scene['proposal_id']:scene for scene in _selected_scenes(mapping)}
    def bound(ref):
        binding=_file_ref(ref);value,path=read(dict(path=binding['file'],sha256=binding['sha256']),Path.cwd())
        refs[str(path)]=reference(path);return value
    payloads={}
    from sim.ausc_policy import m2_contract as contract
    for scene in scenes.values():
        prep=bound(scene['preparation_config']);complete=bound(scene['prepared_complete']);patient=bound(scene['patient_receipt'])
        spec=prep.get('patient_asset',{});split=scene['split'];subject=scene['subject_id']
        builder.require_current(prep);builder.require_current(spec)
        need(patient.get('identity')==subject and patient.get('split')==split and complete.get('patient')==patient and
             prep.get('prepare_only') is True and complete.get('completed') is True and
             complete.get('geometry_id')==builder.GEOMETRY_ID and complete.get('source_sha256')==builder.GEOMETRY_SOURCE_SHA256 and
             patient.get('non_patient_scene_unchanged') is True and spec.get('expected_split','train')==split,
             'Original prepared identity/header differs from fixed allocation')
        split_data=bound(dict(file=spec['identity_split'],sha256=contract.IDENTITY_SPLIT_SHA256))
        need([name for name,ids in split_data['splits'].items() if subject in ids]==[split] and
             patient.get('identity_split_sha256')==contract.IDENTITY_SPLIT_SHA256, 'Original frozen split/header differs')
        directory=data._path(spec['directory'],Path.cwd())
        manifest=bound(dict(file=str(directory/'asset_manifest.json'),sha256=spec['manifest_sha256']))
        summary=bound(dict(file=str(directory/'summary.json'),sha256=spec['summary_sha256']))
        bodies=[row for row in manifest if row.get('file')==spec['file']]
        need(len(bodies)==1 and bodies[0].get('identity')==subject and bodies[0].get('split')==split and
             bodies[0].get('sha256')==patient.get('asset_sha256') and bodies[0].get('numeric_candidate_screen_passed') is True and
             bodies[0].get('mesh_scale')==1. and
             summary.get('identity_split_sha256')==contract.IDENTITY_SPLIT_SHA256 and
             summary.get('schema')=='m2_observed_seated_retarget_v1', 'Original body manifest/summary identity differs')
        body=bodies[0];body_path=data._path(spec['file'],directory)
        from sim.ausc_policy.patient_scene import layout_translation
        need(body_path.parent==directory and
             Path(_file_ref(scene['patient_receipt'])['file'])==Path(_file_ref(scene['prepared_complete'])['file']).parent/'patient_receipt.json' and
             data.np.allclose(patient.get('layout_translation_world_m'),layout_translation(spec),rtol=0,atol=1e-12),
             'Original body/receipt/layout differs from authenticated preparation')
        payloads[str(body_path)]=dict(path=str(body_path),sha256=body['sha256'])
        if body.get('provenance_file'):
            bound(dict(file=str(data._path(body['provenance_file'],directory)),sha256=body['provenance_sha256']))
        # Only the authenticated approved geometry alias resolves to a real path.
        source=Path(prep['geometry_source']).resolve()
        need(prep.get('geometry_source_sha256',builder.GEOMETRY_SOURCE_SHA256)==builder.GEOMETRY_SOURCE_SHA256,
             'Approved geometry source identity changed')
        payloads[str(source)]=dict(path=str(source),sha256=builder.GEOMETRY_SOURCE_SHA256)
    for a in allocated:
        plan=bound(a['fresh_plan'])
        builder.require_current(plan)
        need(plan.get('patient_provenance',{}).get('binding')==_file_ref(scenes[a['proposal_id']]['patient_receipt']) and
             plan.get('name')==a['target'] and plan.get('success') is True,
             'Original numerical plan patient/target/success differs from fixed slot')
    return dict(headers=list(refs.values()),payloads=list(payloads.values()))


def _iteration_proof(proof):
    changes=proof.get('only_iteration_changes',[]);roles=proof.get('rigid_articulation_roles',[])
    need(proof.get('schema')=='m2_solver_iteration_only_authoring_v2' and proof.get('profile')==SELECTED_PHYSICS_PROFILE and
         len(changes)==len(roles)==24 and len({tuple(role) for role in roles})==24 and
         len({(c.get('path'),c.get('attribute')) for c in changes})==24 and
         all(type(c.get('before')) is int and c['before']==32 and type(c.get('after')) is int and c['after']==64 and
             (c.get('path'),c.get('attribute'))==(role[0],role[1]+':solverPositionIterationCount') for c,role in zip(changes,roles)) and
         isinstance(proof.get('float32_armature_attributes'),list) and bool(proof['float32_armature_attributes']) and
         proof.get('armature_authoring_changed') is False and proof.get('geometry_camera_drives_mass_contact_unchanged') is True and
         proof.get('exhaustive_metadata_API_relationship_connection_time_sample_check') is True and
         all(proof.get(k) is False for k in ('physics_executed','physical_validation','training_eligible')) and
         all(re.fullmatch('[0-9a-f]{64}',str(proof.get(k))) for k in
             ('before_stage_semantics_sha256','after_stage_semantics_sha256','before_authored_sha256','after_authored_sha256')),
         'Exact 24 existing int32-to64 defaults, unchanged velocity1/f32 armature and exhaustive iteration proof required')


def _verify_authored_population(authored,selected,output_root):
    """Rebuild full authored raw proof; preserve offline composed-context evidence."""
    from sim.ausc_policy import layered_scene_authoring as layers
    from pxr import Sdf
    metadata=layers._population_metadata(selected)
    root=data._path(str(output_root),Path.cwd())
    receipt_ref=authored.get('receipt',{});receipt,_=read(dict(path=receipt_ref.get('file'),sha256=receipt_ref.get('sha256')),Path.cwd())
    need(authored.get('schema')==receipt.get('schema')==layers.RECEIPT_SCHEMA and
         receipt_ref==scene_store.source_ref(root/'scene_authoring_receipt.json') and
         receipt.get('original_selected')==selected and receipt.get('representation_change_only') is True and
         receipt.get('runtime_equivalence_policy')==RUNTIME_EQUIVALENCE_POLICY==layers.RUNTIME_EQUIVALENCE_POLICY and
         all(receipt.get(k) is False for k in ('physics_executed','physical_validation','training_eligible')) and
         authored.get('base')==receipt.get('base')==scene_store.source_ref(root/'static_base.usdc') and
         authored.get('original_bytes')==receipt.get('original_bytes') and authored.get('saved_bytes')==receipt.get('saved_bytes') and
         authored.get('static_subtrees')==receipt.get('static_subtrees') and bool(receipt.get('static_subtrees')) and
         len(authored.get('prepared',[]))==len(receipt.get('scenes',[]))==len(receipt.get('iteration_proofs',[]))==len(selected),
         'Exact complete batch authoring receipt/output inventory required')
    original_inputs=layers.authoring_input_inventory(receipt,metadata)
    for binding in original_inputs:
        need(reference(binding['file'])==dict(path=binding['file'],bytes=binding['bytes'],sha256=binding['sha256']),
             'Original population authoring input changed')
    expected_files={str(root/'scene_authoring_receipt.json'),str(root/'static_base.usdc')}
    original_scenes={};primaries={}
    for selection,meta,result,item,proof,inputs in zip(selected,metadata,authored['prepared'],receipt['scenes'],
            receipt['iteration_proofs'],receipt['input_bindings']):
        prep,complete,patient,oldroot=meta;inventory=result.get('dependency_inventory',{})
        need(result.get('original')==selection and result.get('input_bindings')==inputs and
             result.get('scene_authoring_receipt')==receipt_ref and inventory==item.get('dependency_inventory') and
             inventory.get('base')==authored['base'] and result.get('iteration_proof')==proof and
             result.get('baseline_scene_sha256')==item.get('original',{}).get('sha256') and
             result.get('composed_semantics_sha256')==item.get('composed_semantics_sha256') and
             item.get('saved_USD_readback_exact') is True and item.get('private_runtime_readback_exact') is True,
             'Per-scene order/input/dependency/readback proof differs from batch receipt')
        _iteration_proof(proof)
        directory=Path(inventory['primary']['file']).parent
        need(directory.parent==root and inventory['primary']['file']==str(directory/'candidate_physics.usdc'),
             'Exact per-body primary output role required')
        for key,name in (('preparation_config','prepare.json'),('prepared_complete','COMPLETE.json'),
                         ('patient_receipt','patient_receipt.json'),('physics_profile','physics_profile.json')):
            need(result.get(key)==_file_ref(scene_store.source_ref(directory/name)), 'Exact per-scene metadata output role required')
            expected_files.add(str(directory/name))
        newprep,_=read(dict(path=result['preparation_config']['file'],sha256=result['preparation_config']['sha256']),Path.cwd())
        selected_prep=deepcopy(prep);selected_prep['physics']=deepcopy(SELECTED_PHYSICS_PROFILE)
        need(prep.get('physics')==dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02) and newprep==selected_prep,
             'Preparation changed beyond selected32-to64 physics')
        profile,_=read(dict(path=result['physics_profile']['file'],sha256=result['physics_profile']['sha256']),Path.cwd())
        expected_profile=dict(proof,original_scene=item['original'],inherited_profile=scene_store.source_ref(oldroot/'physics_profile.json'),
            new_scene=inventory['primary'],representation_lineage=receipt_ref,saved_USD_readback_exact=True)
        need(profile==expected_profile, 'Saved iteration profile differs from exact original/profile/authoring proof')
        newcomplete,_=read(dict(path=result['prepared_complete']['file'],sha256=result['prepared_complete']['sha256']),Path.cwd())
        expected_complete=deepcopy(complete);expected_complete.update(effective_scene_sha256=inventory['primary']['sha256'],
            scene_dependency_inventory=inventory,scene_authoring_receipt=receipt_ref,solver_isolation_lineage=result['physics_profile'],
            preparation_backend='cpu_lossless_static_layer_and_exact_existing_iteration_authoring')
        need(newcomplete==expected_complete, 'Complete receipt changed beyond explicit layered authoring fields')
        need(scene_store.validate_prepared_dependency_inventory(newcomplete)==inventory,
             'Header-authenticated exact prepared dependency inventory changed')
        for key,name in {**scene_store.FILES,**scene_store.OPTIONAL}.items():
            if key in ('scene','physics_profile') or not (oldroot/name).exists():continue
            before=reference(oldroot/name);after=reference(directory/name)
            need(before['bytes']==after['bytes'] and before['sha256']==after['sha256'], 'Copied original sidecar changed')
            expected_files.add(str(directory/name))
        source=layers._private_source(item['original']);qualified,actual_proof=layers.apply_qualified_iterations(source)
        # The mandatory policy explicitly retains these two stage digests as
        # immutable offline authoring evidence. Resolver-sensitive stage values
        # are compared in full by the fresh SDK parity below, in one context.
        offline_stage_digests={'before_stage_semantics_sha256','after_stage_semantics_sha256'}
        need(set(actual_proof)==set(proof) and all(actual_proof[key]==proof[key] for key in proof if key not in offline_stage_digests),
             'Rebuilt complete raw authored iteration delta/typed invariants differ')
        need(layers.digest(layers._layered_authored_snapshot(inventory,receipt['static_subtrees']))==
             layers.digest(layers._layer_snapshot(qualified.GetRootLayer()))==item['authored_semantics_sha256'] and
             re.fullmatch('[0-9a-f]{64}',str(item.get('composed_semantics_sha256'))) is not None,
             'Saved complete authored scene differs from qualified original or offline composed evidence is missing')
        # Independently compare each actual saved authored layer against its
        # intended specifications; composition cannot hide a masked opinion.
        intended=qualified.GetRootLayer();base=Sdf.Layer.CreateAnonymous();ancestors=set()
        for path in receipt['static_subtrees']:
            parent=Sdf.Path(path).GetParentPath()
            while parent!=Sdf.Path.absoluteRootPath:ancestors.add(str(parent));parent=parent.GetParentPath()
        def skeleton(spec):
            Sdf.CreatePrimInLayer(base,spec.path)
            if str(spec.path) in ancestors:
                for child in spec.nameChildren:skeleton(child)
        for spec in intended.rootPrims:skeleton(spec)
        primary=Sdf.Layer.CreateAnonymous();primary.TransferContent(intended)
        for path in receipt['static_subtrees']:
            need(bool(intended.GetPrimAtPath(path)) and Sdf.CopySpec(intended,path,base,path), 'Missing declared static subtree')
            shell=Sdf.Layer.CreateAnonymous();Sdf.CreatePrimInLayer(shell,path)
            need(Sdf.CopySpec(shell,path,primary,path), 'Cannot reconstruct intended primary shell')
        primary.subLayerPaths=[authored['base']['file']]
        saved_primary=Sdf.Layer.OpenAsAnonymous(inventory['primary']['file']);saved_base=Sdf.Layer.OpenAsAnonymous(authored['base']['file'])
        need(list(saved_primary.subLayerPaths)==list(primary.subLayerPaths) and
             [(x.offset,x.scale) for x in saved_primary.subLayerOffsets]==[(x.offset,x.scale) for x in primary.subLayerOffsets],
             'Saved primary sublayer paths/offsets changed')
        for layer in (saved_primary,primary):
            layer.pseudoRoot.ClearInfo('subLayers');layer.pseudoRoot.ClearInfo('subLayerOffsets')
        need(layers._layer_snapshot(saved_primary)==layers._layer_snapshot(primary) and
             layers._layer_snapshot(saved_base)==layers._layer_snapshot(base), 'Saved primary/base authored USD specifications differ')
        expected_files.add(inventory['primary']['file']);original_scenes[item['original']['file']]=item['original'];primaries[inventory['primary']['file']]=inventory['primary']
    need(sum(r['bytes'] for r in original_scenes.values())==authored['original_bytes'] and
         authored['base']['bytes']+sum(r['bytes'] for r in primaries.values())==authored['saved_bytes'] and
         authored['saved_bytes']<authored['original_bytes']*.9, 'Actual original/shared/primary byte savings differ')
    actual=set()
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'Link in immutable author output')
        if path.is_file():actual.add(str(path));layers._single_file(scene_store.source_ref(path),immutable=True)
        else:need(path.is_dir(), 'Special author output')
    need(actual==expected_files, 'Unknown or omitted immutable batch author output')
    return receipt


def _runtime_equivalence(original,derived):
    """Fresh actual SDK-selected private routes, without publishing any CAS."""
    from sim.ausc_policy import layered_scene_authoring as layers
    from pxr import Ar
    old_binding=original['compact_scene_storage']
    binding=scene_store.build_binding(**{key:derived[key] for key in
        ('preparation_config','prepared_complete','patient_receipt')},store_root=old_binding['store_root'],batch_id=old_binding['batch_id'],
        retainer_witness_contract=original['m2_given_contact_control']['metadata']['native_retainer_audit'],
        sdk_mdl_manifest=_file_ref(old_binding['sources']['sdk_mdl_manifest']))
    need(old_binding['schema']==scene_store.SDK_SCHEMA and binding['schema']==scene_store.SDK_LAYER_SCHEMA and
         all(binding['sources'][key]==ref for key,ref in old_binding['sources'].items() if
             key.startswith('sdk_mdl_') or key in scene_store.RUNTIME_SOURCES|scene_store.OPTIONAL_RUNTIME_SOURCES),
         'Actual private equivalence must retain exact original SDK/native-retainer source roles')
    context=Ar.GetResolver().GetCurrentContext()
    with Ar.ResolverContextBinder(context),Ar.ResolverScopedCache():
        old=scene_store.prepared_runtime_stage(old_binding);selected,proof=layers.apply_qualified_iterations(old)
        new=scene_store.prepared_runtime_stage(binding)
        need(layers.scene_snapshot(selected)==layers.scene_snapshot(new), 'Complete SDK-selected private runtime differs')
        need(all(layer.anonymous for stage in (old,selected,new) for layer in stage.GetUsedLayers()) and
             not scene_store.runtime_stage_leases(), 'CPU equivalence must remain anonymous and acquire no owned native stage lease')
        return dict(schema='m2_lossless_sdk_private_runtime_equivalence_v1',original_sources=deepcopy(old_binding['sources']),
            derived_sources=deepcopy(binding['sources']),selected_iteration_proof=proof,composed_semantics_sha256=layers.digest(layers.scene_snapshot(new)),
            SDK_selected_runtime_readback_exact=True,all_layers_private_anonymous=True,dense_copy_created=False,
            physics_executed=False,training_eligible=False)


def _verify_runtime_equivalence_receipt(original,derived,stored):
    """Keep sealed offline hashes and freshly compare ALL current SDK fields."""
    _iteration_proof(stored.get('selected_iteration_proof',{}))
    need(re.fullmatch('[0-9a-f]{64}',str(stored.get('composed_semantics_sha256'))) is not None,
         'Complete original offline SDK parity digest required')
    fresh=_runtime_equivalence(original,derived)
    need(set(stored)==set(fresh) and all(stored[key]==fresh[key] for key in fresh if key not in
         ('selected_iteration_proof','composed_semantics_sha256')), 'Original offline SDK parity source/role/claim changed')
    old_proof=stored['selected_iteration_proof'];new_proof=fresh['selected_iteration_proof']
    # All four hashes in this SDK-selected offline proof include the original
    # SDK resolver diagnostic metadata. The approved policy retains them as
    # bound offline evidence, while typed physical invariants remain exact and
    # _runtime_equivalence compares every current snapshot field without masks.
    offline_sdk_digests={'before_stage_semantics_sha256','after_stage_semantics_sha256','before_authored_sha256','after_authored_sha256'}
    need(set(old_proof)==set(new_proof) and all(old_proof[key]==new_proof[key] for key in old_proof if key not in offline_sdk_digests),
         'Exact current SDK typed iteration invariants differ from the original offline proof')
    return fresh


def _checkpoint_plan_headers(binding,mapping,report,output):
    from scripts.experiments.wam_runpod_20261027 import completed_authoring_checkpoint_v1 as recovery
    need(report.get('completed_authoring_checkpoint')==binding, 'Plan/derivation checkpoint capability changed')
    value,_=read(binding,Path.cwd())
    allocated=[dict(a,split=s) for s in ('train','validation') for a in mapping['ordered_allocations'][s]]
    recovery.validate_headers(value,_selected_scenes(mapping),allocated,output)
    recovery.validate_plan_headers(value,report.get('plans'),output)
    recovery.validate_saved_plans(value,allocated)
    recovery.validate_copied_plans(value,report['plans'])
    return value


def _verify_derivation(ref,mapping,attempts,executed,*,checkpoint=None):
    report,path=read(ref,Path.cwd());helper=_helper();selected=_selected_scenes(mapping)
    if checkpoint is not None or report.get('completed_authoring_checkpoint') is not None:
        binding=report.get('completed_authoring_checkpoint')
        need(checkpoint is None or binding==checkpoint['reference'], 'Exact completed-stage checkpoint capability required')
        value=_checkpoint_plan_headers(binding,mapping,report,path.parent)
        if checkpoint is None:
            checkpoint=_completed_authoring(dict(completed_authoring_checkpoint=binding,source_recipe=value['bindings']['original_recipe']),mapping,path.parent)
    source_root=path.parent if checkpoint is None else Path(checkpoint['checkpoint']['source_root'])
    author=_historical_author(executed) if checkpoint is None else _checkpoint_author(checkpoint,executed)
    if checkpoint is not None:
        need(report.get('completed_authoring_checkpoint')==checkpoint['reference'] and
             report.get('layered_authoring')==checkpoint['authored'] and report.get('current_raw_prepared')==checkpoint['current_raw_prepared'] and
             report.get('runtime_equivalence_receipt')==checkpoint['runtime_equivalence_receipt'], 'Exact completed-stage checkpoint derivation required')
    need(report.get('schema')==DERIVATION_SCHEMA and report.get('author_helper')==author and
         report.get('selected_profile')==helper.SELECTED_PROFILE and report.get('profile')==SELECTED_PHYSICS_PROFILE and
         report.get('unique_original_scenes')==len(selected) and report.get('numerical_plans_unchanged') is True and
         report.get('physics_executed') is False and report.get('training_eligible') is False and
         len(report.get('plans',[]))==len(attempts), 'Exact exhaustive lossless scene/all-plan derivation map required')
    need([row.get('original') for row in report.get('scenes',[])]==selected, 'Derived scene order/original roles changed')
    raw_selection=[{key:_file_ref(scene[key]) for key in ('preparation_config','prepared_complete','patient_receipt')} for scene in selected]
    first=[next(entry for entry in report['plans'] if entry['proposal_id']==scene['proposal_id']) for scene in selected]
    expected_request=_authoring_request(raw_selection,[entry['baseline_config'] for entry in first],
        [entry['baseline_provenance'] for entry in first],source_root)
    selection=expected_request['selected']
    need(report.get('authoring_request')==expected_request and report.get('cpu_execution')==_authoring_cpu_binding(expected_request),
         'Exact frozen native60 CPU provider/script/executable/input request receipt required')
    authored=report.get('layered_authoring',{})
    current=report.get('current_raw_prepared');_verify_current_raw_metadata(authored,current,source_root/'current_raw')
    _verify_authored_population(authored,selection,source_root/'derived')
    parity=authored.get('runtime_equivalence',[])
    parity_ref=report.get('runtime_equivalence_receipt',{});parity_receipt,parity_path=read(parity_ref,Path.cwd())
    need(parity_path==source_root/'SDK_RUNTIME_EQUIVALENCE.json' and parity_receipt==dict(
         schema='m2_given_lossless_sdk_cpu_receipt_v1',authoring_request=expected_request,cpu_execution=report['cpu_execution'],
         scene_authoring_receipt=authored['receipt'],proofs=parity), 'Original immutable SDK CPU parity receipt or offline evidence changed')
    need(len(parity)==len(selected), 'One actual private SDK runtime equivalence proof per selected original required')
    for baseline_ref,result,record,proof in zip(expected_request['baseline_configs'],authored['prepared'],current,parity):
        baseline,_=read(baseline_ref,Path.cwd())
        _verify_runtime_equivalence_receipt(baseline,dict(result,prepared_complete=record['prepared_complete']),proof)
    scenes={row['proposal_id']:row for row in report['scenes']}
    need(len(scenes)==len(selected), 'Duplicate derived proposal identity')
    for row,result,record in zip(report['scenes'],authored['prepared'],current):
        derived=row['derived'];original=row['original'];restored=deepcopy(derived);restored.pop('derivation');restored.pop('current_raw');restored.pop('raw_preparation_config')
        for key in ('preparation_config','prepared_complete','patient_receipt'):restored[key]=original[key]
        need(row['proposal_id']==original['proposal_id'] and restored==original and derived['derivation']==result and
             derived['current_raw']==record and derived['prepared_complete']==record['prepared_complete'] and
             derived['raw_preparation_config']==reference(_file_ref(original['preparation_config'])['file']) and
             all(derived[key]==result[key] for key in ('preparation_config','patient_receipt')),
             'Exact original identity/template metadata or provider output role changed')
    allocated=[dict(a,split=s) for s in ('train','validation') for a in mapping['ordered_allocations'][s]]
    for entry,a,attempt in zip(report['plans'],allocated,attempts):
        scene=scenes[a['proposal_id']]
        need(entry['slot_id']==a['slot_id'] and entry['proposal_id']==a['proposal_id'] and entry['original']==_file_ref(a['fresh_plan']),
             'Derived plan belongs to another original allocation')
        need(entry.get('baseline_provenance')==reference(Path(entry['baseline_config']['path']).with_suffix('.provenance.json')),
             'Exact original baseline builder provenance role required')
        old,_=read(dict(path=entry['original']['file'],sha256=entry['original']['sha256']),Path.cwd())
        new,_=read(dict(path=entry['derived']['file'],sha256=entry['derived']['sha256']),Path.cwd())
        restored=deepcopy(new);restored['patient_provenance']['binding']=old['patient_provenance']['binding']
        need(restored==old and old['patient_provenance']['binding']==_file_ref(scene['original']['patient_receipt']) and
             new['patient_provenance']['binding']==_file_ref(scene['derived']['patient_receipt']), 'Numerical plan or exact patient provenance changed')
        need(entry.get('runtime')==attempt['config'], 'Derived runtime belongs to another fixed attempt')
        baseline,_=read(entry['baseline_config'],Path.cwd());config,_=read(attempt['config'],Path.cwd())
        receipt,_=read(entry['builder_receipt'],Path.cwd())
        if checkpoint is not None:
            prior,_=read(entry['baseline_provenance'],Path.cwd())
            need(receipt['collector_cpu_builder']['executable']==_executable(GATE_PYTHON) and
                 receipt['collector_cpu_builder']['collector_root']==prior['collector_cpu_builder']['collector_root']==COLLECTOR_ROOT and
                 receipt['collector_cpu_builder']['script_sha256']==prior['collector_cpu_builder']['script_sha256'] and
                 prior['collector_cpu_builder']['executable']['invoked_path']==CPU_CONFIG_PYTHON and config.get('isaac_python')==CPU_CONFIG_PYTHON,
                 'Checkpoint requires unchanged child script/collector and exact Native51 owner/Native60 CPU provenance')
        pairs=_source_pairs(baseline,config,scene['derived'],dict(receipt=receipt))
        need(_runtime_rebinding(baseline,config,pairs,**({} if checkpoint is None else
             dict(capture_paths=checkpoint['checkpoint']['capture_paths'])))==entry['reference_binding'], 'Full runtime/reference projection changed')
    return report


def prepare(settings,output):
    """Freeze every config and source union before publishing the attempt plan."""
    _require_pins()
    need(settings.get('current_raw_equivalence') is True, 'Explicit current RAW equivalence opt-in required')
    need(isinstance(settings.get('qualification_all_eight'),dict), 'Mandatory full eight-case qualification required; five alone cannot admit collection')
    output=data._path(str(output),Path.cwd());need(not output.exists(), 'Preparation requires a new immutable config directory')
    executed=_execution_freeze(settings.get('execution_freeze'))
    mapping=data._allocation(settings,Path.cwd())
    allocated=[dict(a,split=s) for s in ('train','validation') for a in mapping['ordered_allocations'][s]]
    # Full metadata pass, including all original rejected geometry slots, before
    # the qualification reader opens a capture or a builder hashes any payload.
    for row in allocated:data.shared.require_membership(row.get('split'),row.get('subject_id'))
    need(len(allocated)==240, 'Fixed 240-allocation required')
    headers=_original_headers(mapping,allocated)
    checkpoint=None
    if settings.get('completed_authoring_checkpoint') is not None:
        checkpoint=_completed_authoring(settings,mapping,output)
        _checkpoint_settings(settings,checkpoint)
    final_live=data._path(data.FINAL_PROCESSING_ROOT,Path.cwd())
    need(not final_live.exists(), 'API-fixed final processing root must still be never materialized before successor prepare')
    negative_role=_negative_roles(settings,output)
    code=data._path(settings.get('collector_root'),Path.cwd());need(str(code)==COLLECTOR_ROOT, 'Exact original qualified native collector required')
    collector=_collector(code)
    author=_historical_author(executed) if checkpoint is None else _checkpoint_author(checkpoint,executed)
    qualification_inputs=_qualification_costs(settings,executed)
    recipe,smokes=_verified_recipe(settings.get('qualification'),settings.get('qualification_all_eight'),settings.get('source_recipe'),code,executed,
        **({} if checkpoint is None else dict(checkpoint=checkpoint)))
    base=data._path(settings.get('base'),Path.cwd());need(base==BASE, 'Original WAM base required')
    collection=_safe(settings.get('collection_id'));cohort=_safe(settings.get('batch_id'))
    runs=settings.get('run_ids',{});need(set(runs)==set(PHASES) and len(set(runs.values()))==3, 'Three distinct predeclared owner IDs required')
    for run in runs.values():_safe(run);need(not (base/'runs/ausc_policy'/run).exists(), 'Pre-existing formal run directory')
    lock=base/'locks/isaac_gpu.lock';lock_stat=lock.stat()
    need((lock_stat.st_dev,lock_stat.st_ino)==LOCK_ID, 'Original GPU lock identity changed')
    _closed_owner(settings.get('previous_owner'),Path.cwd())
    executable=_executable(settings.get('owner_python'))
    need(settings.get('owner_python')==CPU_CONFIG_PYTHON and
         settings.get('cpu_config_python')==(CPU_CONFIG_PYTHON if checkpoint is None else GATE_PYTHON),
         'Original51 owner and explicitly checkpoint-bound Native60 CPU builder required')
    config_executable=_executable(settings.get('cpu_config_python'))
    metadata=data._path(settings.get('metadata_root'),Path.cwd());seal=data._path(settings.get('seal_path'),Path.cwd())
    need(not metadata.exists() and not seal.exists(), 'New metadata root and external seal required')
    archive=data._path(settings.get('processing_archive_root'),Path.cwd())
    need(not archive.exists(), 'New exclusive processing archive root required')
    roots=[output,metadata,seal,code,archive,final_live,Path(executed['root'])]
    need(all(a!=b and a not in b.parents and b not in a.parents for i,a in enumerate(roots) for b in roots[i+1:]),
         'Config/archive, metadata, seal and frozen collector roots must be disjoint')
    scenes={s['proposal_id']:s for s in mapping.get('scenes',[])}
    need(len(scenes)==len(mapping.get('scenes',[])), 'Duplicate original scene identity')
    for a in allocated:
        scene=scenes.get(a.get('proposal_id'),{})
        need(scene.get('subject_id')==a['subject_id'] and scene.get('split')==a['split'], 'Allocation differs from original scene membership')
    runtime=deepcopy(settings);runtime['native_contract']=smokes[0]['meta']['native_retainer_audit']
    runtime.pop('_completed_authoring_context',None)
    if checkpoint is not None:runtime['_completed_authoring_context']=checkpoint
    runtime['author_helper']=author
    runtime['sdk_mdl_manifest']=smokes[0]['config']['compact_scene_storage']['sources']['sdk_mdl_manifest']
    output.mkdir(parents=True,exist_ok=False);rows=[]
    scenes,derived_allocated,derivation=_derive_population(mapping,allocated,runtime,output)
    for i,a in enumerate(derived_allocated):
        phase='first100' if i<100 else 'remainder' if i<200 else 'validation'
        aid=_safe(collection+'-'+str(i+1).zfill(3));episode=aid+'-episode'
        built=_build_episode(scenes[a['proposal_id']],a,runtime,output,episode);config=built['config']
        need(_conditions(config)==recipe['collector_conditions'] and config.get('base')==str(base) and
             config.get('time_limit_seconds')==2400 and config.get('compact_scene_storage',{}).get('batch_id')==cohort,
             'Unsupported collector conditions or changed cohort/child deadline')
        meta=config['m2_given_contact_control']['metadata']
        need(all(meta.get(k)==v for k,v in dict(episode_id=episode,subject_id=a['subject_id'],split=a['split'],target_id=a['target']).items()),
             'Rebuilt given expert identity differs from fixed slot')
        baseline,_=read(a['baseline_config'],Path.cwd())
        pairs=_source_pairs(baseline,config,scenes[a['proposal_id']],built)
        reference_binding=_runtime_rebinding(baseline,config,pairs,**({} if checkpoint is None else
            dict(capture_paths=checkpoint['checkpoint']['capture_paths'])))
        cfg=_write(output/'episodes'/(aid+'.json'),config)
        provenance=_write(output/'episodes'/(aid+'.provenance.json'),built['receipt'])
        derivation['plans'][i].update(runtime=cfg,builder_receipt=provenance,reference_binding=reference_binding)
        roles=meta['given_reference']['sources']
        fixed=_fixed_sources(meta,config) if any(s['kind']=='fixed' for s in recipe['inventory'].values()) else {}
        files=_freeze_sources(recipe['inventory'],code,roles,fixed,output)
        freeze=_write(output/'episodes'/(aid+'.source-freeze.json'),dict(schema=data.SOURCE_FREEZE_SCHEMA,
            code_commit=collector['commit'],code_commit_file=collector['code_commit_file'],
            source_manifest=collector['source_manifest'],files=files))
        rows.append(dict(attempt_id=aid,episode_id=episode,reset_id=aid+'-reset',run_id=runs[phase],job_id=aid,
            slot_id=a['slot_id'],subject_id=a['subject_id'],split=a['split'],target_id=a['target'],preclassified_reachable=True,
            config=cfg,source_freeze=freeze))
    derivation_ref=_write(output/'DERIVATION_MAP.json',derivation)
    _verify_derivation(derivation_ref,mapping,rows,executed,**({} if checkpoint is None else dict(checkpoint=checkpoint)))
    groups=_attempts(rows,mapping);batches={}
    for phase,group in groups.items():
        batch=dict(experiment='isaac_batch',base=str(base),parallelism=4,time_limit_seconds=owner_deadline(len(group)),
            jobs=[dict(id=r['job_id'],config=r['config']['path'],sha256=r['config']['sha256']) for r in group])
        isaac_batch.validate_jobs(batch);batches[phase]=_write(output/(phase+'.batch.json'),batch)
    inventory=_write(output/'CONFIG_INVENTORY.json',dict(collector=collector,phase_batches=batches,
        derivation=derivation_ref,configs=[dict(config=r['config'],source_freeze=r['source_freeze']) for r in rows]))
    extras={}
    sdk_paths={source['file'] for smoke in smokes for name,source in smoke['config']['compact_scene_storage']['sources'].items()
        if name=='sdk_mdl_manifest' or scene_store._sdk_file_name(name)}
    runtime_paths={executable['path'],config_executable['path'],executed['environment'].get('ruckig',{}).get('path'),
        executed['environment'].get('python',{}).get('path')}
    def add(path,category):
        ref=reference(path)
        if ref['path'] in sdk_paths:category='sdk'
        elif ref['path'] in runtime_paths:category='other_runtime'
        prior=extras.get(ref['path']);need(prior is None or prior['reference']==ref, 'Conflicting retained input path')
        extras[ref['path']]=dict(category=category,reference=ref)
    for path in output.rglob('*'):
        if path.is_file():add(path,'source' if path.parent.name=='archive' else 'config')
    for key in ('allocation_proposal','allocation_mapping','qualification','qualification_all_eight','qualification_postflight',
                'qualification_cost_ledger','source_recipe','previous_owner'):
        add(data._ref_path(settings[key],Path.cwd()),'prepared')
    add(data._ref_path(negative_role,Path.cwd()),'prepared')
    for ref in qualification_inputs+headers['headers']+headers['payloads']:
        current=reference(ref['path']);need(current['sha256']==ref['sha256'] and ('bytes' not in ref or current==ref), 'Original prepared/header input changed')
        add(ref['path'],'prepared')
    for scene in scenes.values():
        for binding in scene['derivation']['input_bindings']:add(binding['file'],'prepared')
    for entry in settings.get('extra_input_refs',[]):
        need(entry.get('category') in ('source','config','sdk','prepared','other_runtime'), 'Explicit retained-input category required')
        binding=entry.get('reference');need(reference(data._ref_path(binding,Path.cwd()))==binding, 'Extra prelaunch input changed')
        add(binding['path'],entry['category'])
    if checkpoint is not None:
        add(checkpoint['reference']['path'],'prepared')
        for entry in checkpoint['checkpoint']['retained_inputs']:add(entry['reference']['path'],entry['category'])
        for pair in checkpoint['checkpoint']['capture_paths'].values():
            for binding in pair.values():add(binding['path'],'prepared')
    for path in (code/'CODE_COMMIT',code/'SOURCE_MANIFEST.json'):add(path,'source')
    for path in (Path(__file__).resolve(),Path(data.__file__).resolve()):add(path,'source')
    add(data._ref_path(settings['execution_freeze'],Path.cwd()),'source')
    for binding in executed['files'].values():add(binding['path'],'source')
    for binding in (executed['environment'].get('python'),executed['environment'].get('ruckig')):
        if binding:add(binding['path'],'other_runtime')
    add(executable['path'],'other_runtime');add(config_executable['path'],'other_runtime')
    for item in data.shared._json(code/'SOURCE_MANIFEST.json'):add(code/item['file'],'source')
    original_scenes={row['proposal_id']:row for row in _selected_scenes(mapping)}
    for a in allocated:
        for ref in [a['fresh_plan'],*[original_scenes[a['proposal_id']][k] for k in ('preparation_config','prepared_complete','patient_receipt')]]:
            add(_file_ref(ref)['file'],'prepared')
    for smoke in smokes:
        for key in ('config','source_freeze','run_result','owner_terminal','child_result'):
            if key in smoke.get('row',{}):add(data._ref_path(smoke['row'][key],Path.cwd()),'prepared')
        for name in smoke.get('row',{}).get('artifacts',{}):add(Path(smoke['row']['capture_root'])/name,'prepared')
        if 'freeze' in smoke:
            for ref in [smoke['freeze']['code_commit_file'],smoke['freeze']['source_manifest'],*smoke['freeze']['files'].values()]:
                add(data._ref_path(ref,Path.cwd()),'source')
            for original,archived in smoke['freeze']['files'].items():
                need(data.sha256(original)==archived['sha256'], 'Original qualified source differs from its retained archive')
                add(original,'source')
        if 'row' in smoke:
            owner_root=data._ref_path(smoke['row']['owner_terminal'],Path.cwd()).parent
            for path in owner_root.rglob('*'):
                if path.is_file():add(path,'prepared')
    for source in _qualification_shared_inputs(smokes):
        need(reference(source['file'])==dict(path=source['file'],bytes=source['bytes'],sha256=source['sha256']),
             'Qualified shared input changed during preparation')
        add(source['file'],'sdk' if any(kind.startswith('sdk_mdl_') for kind in source.get('kinds',[])) else 'prepared')
    # Building and validating every frozen config already reads these inputs,
    # including bodies whose later phase may never launch. Retain them all.
    for binding in [*[row['config'] for row in rows],*[row['baseline_config'] for row in derived_allocated]]:
        config,_=read(binding,Path.cwd())
        for name,source in config['compact_scene_storage']['sources'].items():
            binding=dict(path=source['file'],bytes=source['bytes'],sha256=source['sha256'])
            prior=extras.get(binding['path'])
            need((prior is not None and prior['reference']==binding) or (prior is None and reference(binding['path'])==binding),
                 'Built config input differs from its frozen retained reference')
            if prior is None:
                extras[binding['path']]=dict(category='sdk' if name.startswith('sdk_mdl_') else 'prepared',reference=binding)
    future_cas=[]
    for row in rows:
        config,_=read(row['config'],Path.cwd());binding=config['compact_scene_storage'];scene=binding['sources']['scene']
        cas=Path(binding['store_root'])/'objects'/scene['sha256'][:2]/(scene['sha256']+'.usdc')
        reservation=dict(path=str(cas),bytes=scene['bytes'],sha256=scene['sha256'])
        future_cas.append(reservation)
        if cas.exists():
            need(reference(cas)==reservation and cas.stat().st_mode & 0o222==0, 'Existing selected CAS object is not exact/immutable')
            add(cas,'prepared')
    smoke_roots=[data._ref_path(smoke['row']['run_result'],Path.cwd()).parent for smoke in smokes]
    forecast=_cost_forecast(list(extras.values()),future_cas,smoke_roots)
    forecast_ref=_write(output/'COST_FORECAST.json',forecast);add(forecast_ref['path'],'config')
    # No max/mean forecast cancels slots. Only unavoidable fixed costs can
    # mathematically preclude the unchanged final actual accounting gate.
    need(not forecast['fixed_cost_mathematically_exceeds_100MB'],
         'Complete immutable fixed union alone exceeds100MB per planned240; forecast retained, no collection plan issued')
    scope=dict(version=data.ACCOUNTING_SCHEMA,owner_roots=[str(base/'runs/ausc_policy'/runs[p]) for p in PHASES],
        extra_input_refs=list(extras.values()),tier_b_attempt_ids=[],storage_roots=settings['storage_roots'],
        operational_metadata=dict(root=str(metadata),allowed_files=_allowed_metadata(),
            limit_bytes=data.METADATA_ENVELOPE_BYTES,charged_bytes=data.METADATA_ENVELOPE_BYTES),
        final_seal=dict(path=str(seal),limit_bytes=data.SEAL_ENVELOPE_BYTES,charged_bytes=data.SEAL_ENVELOPE_BYTES))
    allowed=sorted({row['file'] for row in data.shared._json(code/'SOURCE_MANIFEST.json')}|
        set(FUTURE_SOURCE_FILES)|{'CODE_COMMIT','SOURCE_MANIFEST.json'})
    scope['processing_sources']=dict(scope_id='final_processing_sources_v1',live_root=data.FINAL_PROCESSING_ROOT,
        archive_root=settings['processing_archive_root'],archive_file=settings['processing_archive_file'],allowed_files=allowed,
        descriptor_path=str(metadata/'final_processing_freeze.json'),limit_bytes=data.PROCESSING_ENVELOPE_BYTES,
        charged_bytes=data.PROCESSING_ENVELOPE_BYTES,required_unchanged_gate_sources=executed['required_unchanged_gate_sources'],
        original_execution_freeze=executed)
    data.metadata_accounting(scope,Path.cwd(),snapshot_phase='prelaunch')
    data.processing_source_accounting(scope['processing_sources'],Path.cwd(),snapshot_phase='prelaunch',
        excluded_paths=[str(output),str(metadata),str(seal),str(code),*scope['owner_roots'],*extras])
    for entry in extras.values():need(reference(entry['reference']['path'])==entry['reference'], 'Prelaunch input changed during prepare')
    plan=dict(schema=data.PLAN_SCHEMA,task_id=data.pose.TASK_ID,action_semantics_id=data.given.ACTION_SEMANTICS_ID,
        handoff_policy_id=data.handoff.POLICY_ID,frozen_before_launch=True,frozen_utc=guard.now(),attempts=rows,
        allocation_proposal=settings['allocation_proposal'],allocation_mapping=settings['allocation_mapping'],
        collector=collector,source_recipe=settings['source_recipe'],qualification=settings['qualification'],current_raw_equivalence=True,
        qualification_all_eight=settings['qualification_all_eight'],qualification_postflight=settings['qualification_postflight'],
        qualification_cost_ledger=settings['qualification_cost_ledger'],negative_role_plan=negative_role,
        derivation=derivation_ref,cost_forecast=forecast_ref,config_inventory=inventory,
        run_root=str(base/'runs/ausc_policy'),phase_batches=batches,batch_id=cohort,collection_id=collection,owner_executable=executable,
        config_executable=config_executable,
        execution_freeze=executed,
        previous_owner=settings['previous_owner'],lock=dict(path=str(lock),device=lock_stat.st_dev,inode=lock_stat.st_ino),
        accounting_scope=scope,processing_sources={str(Path(__file__).resolve()):data.sha256(__file__),
            str(Path(data.__file__).resolve()):data.sha256(data.__file__)})
    if checkpoint is not None:plan['completed_authoring_checkpoint']=checkpoint['reference']
    metadata.mkdir(parents=True,exist_ok=False)
    return _write(metadata/'plan.json',plan)


def _declared_input(plan,binding):
    """Bind an exact role before permitting reads outside the plan itself."""
    from sim.ausc_policy import m2_given_contact_lossless_mapping as mapping
    return deepcopy(mapping._CurrentMetadata(plan).bind(binding))


def _input_identity(path,*,category=None):
    path=data._path(str(path),Path.cwd());before=path.lstat()
    # Preserve the existing accounting exception for frozen SDK/runtime files
    # mirrored by the installed environments. Other retained roles stay unique.
    need(stat.S_ISREG(before.st_mode) and (before.st_nlink==1 or category in ('sdk','other_runtime')),
         'Unique regular immutable input required; unapproved linked input')
    binding=reference(path);after=path.lstat()
    def identity(value):
        return (value.st_dev,value.st_ino,value.st_mode,value.st_nlink,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    need(identity(before)==identity(after), 'Immutable input changed while hashing')
    return binding,identity(after)


def _fixed_input_snapshot(plan,path):
    """Complete frozen bytes/stat identities and exact config namespace."""
    snapshot={};declared={};inodes=set()
    for entry in plan['accounting_scope']['extra_input_refs']:
        binding=entry['reference'];canonical=data._ref_path(binding,Path.cwd())
        need(entry.get('category') in ('source','config','sdk','prepared','other_runtime'), 'Exact retained input category required')
        need(str(canonical)==binding['path'] and binding['path'] not in declared,
             'Unique canonical absolute retained input required')
        actual,identity=_input_identity(canonical,category=entry['category'])
        need(actual==binding, 'Predeclared input bytes changed')
        need(identity[:2] not in inodes, 'Aliased immutable input inode')
        inodes.add(identity[:2]);declared[binding['path']]=binding;snapshot[binding['path']]=(actual,identity)
    actual,identity=_input_identity(path);snapshot[str(path)]=(actual,identity)
    root=data._ref_path(plan['config_inventory'],path.parent).parent
    files={name for name in declared if root in Path(name).parents}
    pending=[root];seen=set();actual_files=set()
    while pending:
        item=pending.pop();info=item.lstat();inode=(info.st_dev,info.st_ino)
        need(inode not in seen and not stat.S_ISLNK(info.st_mode), 'Link or duplicate inode in immutable config scope')
        seen.add(inode)
        if stat.S_ISDIR(info.st_mode):
            snapshot[str(item)]=(info.st_dev,info.st_ino,info.st_mode,info.st_mtime_ns,info.st_ctime_ns)
            pending.extend(item.iterdir())
        else:
            need(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and str(item) in files,
                 'Unknown/omitted immutable config/archive file')
            actual_files.add(str(item))
    need(actual_files==files, 'Unknown/omitted immutable config/archive file')
    return snapshot


def _verify_sdk_inputs(plan,base):
    """Recheck the existing exact SDK byte inventory without constructing USD."""
    from sim.ausc_policy import m2_given_contact_lossless_mapping as mapping
    reader=mapping._CurrentMetadata(plan);bindings=[];libraries={}
    for row in plan['attempts']:
        config,_=read(row['config'],base);binding=config['compact_scene_storage']
        sources={name:ref for name,ref in binding['sources'].items() if name=='sdk_mdl_manifest' or scene_store._sdk_file_name(name)}
        for ref in sources.values():reader.bind(ref)
        manifest=reader.bind(sources['sdk_mdl_manifest']);key=(manifest['path'],manifest['bytes'],manifest['sha256'])
        bindings.append((key,sources));libraries.setdefault(key,binding)
    # Only metadata/byte inventories are shared within this one preflight.
    # The next preflight validates every library again; USD is never cached.
    expected={}
    for key,binding in libraries.items():
        bundle=scene_store._sdk_bundle(binding)
        need(bundle is not None, 'Exact approved SDK manifest/library required')
        expected[key]=scene_store._sdk_sources(bundle)
    for key,sources in bindings:
        need(sources==expected[key], 'SDK MDL bound input inventory differs from exact manifest')


def _verify_plan_metadata_and_closure(ref):
    """Private exhaustive registered JSON and closed-byte preflight, no USD."""
    _require_pins()
    plan,path=read(ref,Path.cwd());base=path.parent
    need(plan.get('current_raw_equivalence') is True, 'Explicit current RAW equivalence plan required')
    need(plan.get('schema')==data.PLAN_SCHEMA and plan.get('task_id')==data.pose.TASK_ID and
         plan.get('action_semantics_id')==data.given.ACTION_SEMANTICS_ID and
         plan.get('handoff_policy_id')==data.handoff.POLICY_ID and plan.get('frozen_before_launch') is True,
         'Original given formal prelaunch plan required')
    data._time(plan.get('frozen_utc'));mapping=data._allocation(plan,base)
    groups=_attempts(plan.get('attempts'),mapping)
    # This pure registered reader authenticates ALL240 original/current numeric
    # references, ALL scene metadata and saved/copied checkpoint lineages before
    # a collector, executable, scene, body, array or archived byte is opened.
    from sim.ausc_policy import m2_given_contact_lossless_mapping as lossless_mapping
    lossless_mapping.authenticated_view(mapping,plan)
    _declared_input(plan,plan['previous_owner'])
    data._plan_metadata(plan['attempts'],base)  # ALL memberships/configs before qualification payloads.
    _qualification_contexts(plan['qualification'],plan['qualification_all_eight'])
    collector=_collector(plan['collector']['root']);need(collector==plan['collector'], 'Collector freeze changed')
    executed=_frozen_execution(plan['execution_freeze']['reference'])
    need(collector['root']==COLLECTOR_ROOT and executed==plan['execution_freeze'],
         'Native collector or already executed gate freeze changed')
    # Preparation provenance remains at its original immutable paths. A later
    # reviewed processing root can reconstruct these captures; its own final
    # freeze/fingerprint is an independent prerequisite for export and training.
    for source,digest in plan.get('processing_sources',{}).items():
        need(data.sha256(source)==digest, 'Separately frozen operator/dataset processing bytes changed')
    need(len(plan.get('processing_sources',{}))==2 and {Path(p).name for p in plan['processing_sources']}==
         {Path(__file__).name,Path(data.__file__).name}, 'Exact original preparation processing provenance required')
    qualification_inputs=_qualification_cost_inputs(plan,executed)
    checkpoint=None
    if plan.get('completed_authoring_checkpoint') is not None:
        checkpoint=dict(checkpoint=read(plan['completed_authoring_checkpoint'],base)[0])
    recipe,smokes=_verified_recipe_metadata(plan['qualification'],plan['qualification_all_eight'],plan['source_recipe'],Path(collector['root']),executed,
        **({} if checkpoint is None else dict(checkpoint=checkpoint)))
    need(plan.get('negative_role_plan')==FAILED_PLAN, 'Original negative role isolation binding changed')
    _negative_roles(dict(collection_id=plan['collection_id'],batch_id=plan['batch_id'],
        run_ids={phase:rows[0]['run_id'] for phase,rows in groups.items()},metadata_root=str(base),
        seal_path=plan['accounting_scope']['final_seal']['path'],
        processing_archive_root=plan['accounting_scope']['processing_sources']['archive_root']),
        data._ref_path(plan['config_inventory'],base).parent)
    forecast,_=read(plan['cost_forecast'],base)
    need(forecast.get('schema')=='m2_given_successor_prelaunch_cost_forecast_v1' and
         forecast.get('fixed_cost_mathematically_exceeds_100MB') is False and forecast.get('formal_observed_jobs')==0,
         'Original complete fixed-union forecast required')
    need(_executable(plan['owner_executable']['invoked_path'])==plan['owner_executable'], 'Installed owner executable changed')
    if checkpoint is not None:
        need(plan['owner_executable']['invoked_path']==CPU_CONFIG_PYTHON and
             plan['config_executable']==_executable(GATE_PYTHON), 'Checkpoint Native51 owner/Native60 CPU executable binding changed')
    inventory,_=read(plan['config_inventory'],base)
    need(inventory==dict(collector=collector,phase_batches=plan['phase_batches'],derivation=plan['derivation'],
         configs=[dict(config=r['config'],source_freeze=r['source_freeze']) for r in plan['attempts']]),
         'Original complete configuration inventory changed')
    for phase,rows in groups.items():
        batch,_=read(plan['phase_batches'][phase],base)
        expected=dict(experiment='isaac_batch',base=str(BASE),parallelism=4,time_limit_seconds=owner_deadline(len(rows)),
            jobs=[dict(id=r['job_id'],config=r['config']['path'],sha256=r['config']['sha256']) for r in rows])
        need(batch==expected, 'Fixed phase batch/parallelism/deadline/job inventory changed')
        for row in rows:
            config,_=read(row['config'],base)
            need(_conditions(config)==recipe['collector_conditions'] and config.get('base')==str(BASE) and
                 config.get('time_limit_seconds')==2400 and config.get('compact_scene_storage',{}).get('batch_id')==plan['batch_id'],
                 'Collector condition/cohort/child deadline differs from prelaunch freeze')
            freeze=data._source_freeze(row['source_freeze'],base)
            need(freeze['code_commit']==collector['commit'] and freeze['code_commit_file']==collector['code_commit_file'] and
                 freeze['source_manifest']==collector['source_manifest'], 'Attempt source freeze belongs to another collector')
            meta=config['m2_given_contact_control']['metadata'];roles=meta['given_reference']['sources']
            fixed=_fixed_sources(meta,config) if any(s['kind']=='fixed' for s in recipe['inventory'].values()) else {}
            normalized=_recipe_inventory({k:v['sha256'] for k,v in freeze['files'].items()},Path(collector['root']),roles,fixed)
            need(normalized==recipe['inventory'], 'Prelaunch source freeze differs from exact qualified union/aliases')
    _verify_sdk_inputs(plan,base)
    scope=plan['accounting_scope'];expected_owners=[str(BASE/'runs/ausc_policy'/groups[p][0]['run_id']) for p in PHASES]
    need(scope['version']==data.ACCOUNTING_SCHEMA and scope['owner_roots']==expected_owners and
         scope['operational_metadata']['root']==str(base) and scope['operational_metadata']['allowed_files']==_allowed_metadata() and
         path==base/'plan.json' and plan['run_root']==str(BASE/'runs/ausc_policy'), 'Frozen metadata/owner scope changed')
    declared=set();declared_bindings={}
    for entry in scope['extra_input_refs']:
        binding=entry['reference'];need(reference(binding['path'])==binding, 'Predeclared input bytes changed')
        need(binding['path'] not in declared, 'Duplicate predeclared input path');declared.add(binding['path'])
        declared_bindings[binding['path']]=binding
    need(all(declared_bindings.get(binding['path'])==binding for binding in qualification_inputs),
         'Unknown, changed or omitted retained qualification input/cost path')
    config_root=data._ref_path(plan['config_inventory'],base).parent
    actual={str(p) for p in config_root.rglob('*') if p.is_file()}
    need(actual=={p for p in declared if config_root in Path(p).parents}, 'Unknown/omitted immutable config/archive file')
    data.metadata_accounting(scope,base,snapshot_phase='prelaunch')
    expected=sorted({row['file'] for row in data.shared._json(Path(collector['root'])/'SOURCE_MANIFEST.json')}|
        set(FUTURE_SOURCE_FILES)|{'CODE_COMMIT','SOURCE_MANIFEST.json'})
    need(scope['processing_sources']['allowed_files']==expected and
         scope['processing_sources']['required_unchanged_gate_sources']==plan['execution_freeze']['required_unchanged_gate_sources'] and
         scope['processing_sources']['original_execution_freeze']==plan['execution_freeze'] and
         scope['processing_sources']['descriptor_path']==str(base/'final_processing_freeze.json'), 'Prelaunch source whitelist/executed gate scope changed')
    data.processing_source_accounting(scope['processing_sources'],base,snapshot_phase='reconcile',
        excluded_paths=[str(config_root),str(base),scope['final_seal']['path'],collector['root'],*declared,*expected_owners])
    return plan,path,_fixed_input_snapshot(plan,path)


def verify_plan(ref):
    """Always run fresh complete semantics between fixed closed-input checks."""
    frozen_ref=deepcopy(ref)
    plan,path,before=_verify_plan_metadata_and_closure(frozen_ref)
    base=path.parent;mapping=data._allocation(plan,base);executed=plan['execution_freeze']
    qualification_inputs=_qualification_costs(plan,executed)
    declared_inputs={entry['reference']['path']:entry['reference'] for entry in plan['accounting_scope']['extra_input_refs']}
    need(all(declared_inputs.get(binding['path'])==binding for binding in qualification_inputs),
         'Unknown, changed or omitted retained qualification input/cost path')
    checkpoint=None
    if plan.get('completed_authoring_checkpoint') is not None:
        checkpoint=_completed_authoring(plan,mapping,data._ref_path(plan['config_inventory'],base).parent)
    _verified_recipe(plan['qualification'],plan['qualification_all_eight'],plan['source_recipe'],Path(plan['collector']['root']),executed,
        **({} if checkpoint is None else dict(checkpoint=checkpoint)))
    _verify_derivation(plan['derivation'],mapping,plan['attempts'],executed,**({} if checkpoint is None else dict(checkpoint=checkpoint)))
    after_plan,after_path,after=_verify_plan_metadata_and_closure(frozen_ref)
    need(ref==frozen_ref and after_plan==plan and after_path==path and after==before,
         'Fixed prelaunch inputs or closed namespace changed during semantic verification')
    return plan,path


def _active_owners():
    owners=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:
            if b'sim.ausc_policy.run' in (proc/'cmdline').read_bytes().split(b'\0'):owners.append(int(proc.name))
        except (FileNotFoundError,ProcessLookupError):pass
    return owners


def _require_bulk_gate(plan,base):
    gate,_=read(reference(base/'first100_gate.json'),base)
    inventory={row['attempt_id']:_execution_entry(plan,row) for row in plan['attempts'][:100]}
    fresh=_first100_gate(plan,inventory,base,gate['decided_utc'])
    need(gate==fresh and fresh['passed'] is True, 'First100 original native gate failed or changed')
    for binding in {e['owner_terminal']['path']:e['owner_terminal'] for e in inventory.values()}.values():
        owner=_closed_owner(binding,base)
        need(data._time(owner['ended_utc'])<=data._time(gate['decided_utc']), 'First100 owner ended after its gate decision')
    return gate


def launch_phase(ref,phase):
    """One explicit phase, one original guarded owner, no retries or signals."""
    need(phase in PHASES, 'Unknown finite collection phase')
    frozen_ref=deepcopy(ref)
    plan,path,inputs=_verify_plan_metadata_and_closure(frozen_ref);base=path.parent;index=PHASES.index(phase)
    need(_execution_freeze(plan['execution_freeze']['reference'])==plan['execution_freeze'], 'Active gate differs from prelaunch execution freeze')
    run_id=plan['attempts'][0 if index==0 else 100 if index==1 else 200]['run_id']
    run=Path(plan['run_root'])/run_id;launch=base/'phases'/(phase+'.launch.json');intent=base/'phases'/(phase+'.intent.json')
    need(not run.exists() and not launch.exists() and not intent.exists(), 'Phase is single use; existing run/launch/intent')
    gate=None
    if index:
        gate=_require_bulk_gate(plan,base)
        gate_identity=_input_identity(base/'first100_gate.json')
        previous_run=plan['attempts'][0 if index==1 else 100]['run_id']
        previous=reference(Path(plan['run_root'])/previous_run/'result.json')
    else:previous=_declared_input(plan,plan['previous_owner'])
    previous_identity=_input_identity(data._ref_path(previous,base))
    need(previous_identity[0]==previous, 'Previous owner differs from the authenticated input role')
    owner=_closed_owner(previous,base)
    if index:
        previous_batch,_=read(plan['phase_batches'][PHASES[index-1]],base)
        need(owner.get('run_id')==previous_run and owner.get('config')==previous_batch,
             'Previous owner/config differs from the frozen preceding phase')
    need(sys.platform=='linux', 'Native launch is Linux only')
    need(not guard.group_members(owner['stage']['process_identity']), 'Previous owned process group is not drained')
    lock=BASE/'locks/isaac_gpu.lock';st=lock.stat()
    need(plan['lock']==dict(path=str(lock),device=st.st_dev,inode=st.st_ino) and
         (st.st_dev,st.st_ino)==LOCK_ID and data.sha256(guard.__file__)==GUARD_SHA, 'Original guard/lock identity changed')
    checks=[]
    with lock.open('r+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locked=os.fstat(stream.fileno());need((locked.st_dev,locked.st_ino)==LOCK_ID, 'Locked original inode changed')
        for i in range(2):
            gpu=guard.gpu_pids();owners=_active_owners()
            need(not gpu and not owners, 'GPU/owner occupied; finite phase refused')
            checks.append(dict(utc=guard.now(),gpu_compute_pids=sorted(gpu),active_owners=owners,
                lock_device=locked.st_dev,lock_inode=locked.st_ino,lock_exclusively_held=True))
            if i==0:time.sleep(2)
        final_plan,final_path=verify_plan(frozen_ref)
        need(ref==frozen_ref and final_plan==plan and final_path==path,
             'Fixed launch plan changed between preflights')
        need(_input_identity(data._ref_path(previous,base))==previous_identity,
             'Previous owner changed during final semantic verification')
        final_owner=_closed_owner(previous,base)
        need(final_owner==owner and not guard.group_members(final_owner['stage']['process_identity']),
             'Previous owner or its drained group changed before reservation')
        if index:
            need(_input_identity(base/'first100_gate.json')==gate_identity and _require_bulk_gate(plan,base)==gate,
                 'First100 original bulk gate changed before reservation')
            need(_input_identity(base/'first100_gate.json')==gate_identity, 'First100 gate changed during its final check')
        # Close bytes/namespaces again after the owner/bulk callbacks. This is
        # metadata-only; every fresh RAW/SDK semantic oracle ran once above.
        closed_plan,closed_path,closed_inputs=_verify_plan_metadata_and_closure(frozen_ref)
        need(ref==frozen_ref and closed_plan==plan and closed_path==path and closed_inputs==inputs,
             'Fixed launch inputs or namespace changed before reservation')
        need(_input_identity(data._ref_path(previous,base))==previous_identity and
             not guard.group_members(final_owner['stage']['process_identity']),
             'Previous owner or its drained group changed during final closure')
        if index:need(_input_identity(base/'first100_gate.json')==gate_identity, 'First100 gate changed during final closure')
        current_lock=lock.stat()
        need((current_lock.st_dev,current_lock.st_ino)==LOCK_ID and data.sha256(guard.__file__)==GUARD_SHA,
             'Original guard/lock identity changed before reservation')
        need(not run.exists() and not launch.exists() and not intent.exists() and not _active_owners(),
             'Owner/run/launch/intent appeared before reservation')
        # Every prelaunch check is serializable before ownership is started.
        json.dumps(checks,allow_nan=False)
        _write(intent,dict(status='single_use_launch_reserved',plan=ref,phase=phase,run_id=run_id,utc=guard.now(),prelaunch_idle_checks=checks))
    command=[plan['owner_executable']['invoked_path'],'-u','-B','-m','sim.ausc_policy.run',
        '--config',plan['phase_batches'][phase]['path'],'--run-id',run_id]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PYTHONPATH']=plan['collector']['root']
    with (base/'phases'/(phase+'.log')).open('xb') as log:
        process=subprocess.Popen(command,cwd=plan['collector']['root'],env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    identity=guard.process_identity(process.pid)
    result=dict(plan=ref,phase=phase,run_id=run_id,command=command,executable=plan['owner_executable'],
        pid=process.pid,identity=identity,collector=plan['collector'],config=plan['phase_batches'][phase],
        started_utc=guard.now(),prelaunch_idle_checks=checks,lock=plan['lock'],previous_owner=previous,
        finite_wall_seconds=owner_deadline(100 if index<2 else 40),training_eligible=False)
    return _write(launch,result)


def _capture_row(plan,row,entry):
    root=Path(plan['run_root'])/row['run_id']/'output'/row['job_id']/'compact_multimodal'
    need(root.is_dir(), 'Successful native case lacks capture directory')
    artifacts={}
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'Symlink in original compact capture')
        if path.is_file():
            ref=reference(path);artifacts[path.relative_to(root).as_posix()]=dict(bytes=ref['bytes'],sha256=ref['sha256'])
        else:need(path.is_dir(), 'Special file in original compact capture')
    return dict(row,**{k:entry[k] for k in ('run_result','owner_terminal','child_result')},
        capture_root=str(root),artifacts=artifacts)


def reconcile(ref,*,phase='all'):
    """Retain every fixed attempt, independently deriving original outcomes."""
    need(phase in ('first100','all'), 'Unknown reconciliation phase')
    plan,path=verify_plan(ref);base=path.parent
    need(_execution_freeze(plan['execution_freeze']['reference'])==plan['execution_freeze'], 'Active reconciliation gate differs from prelaunch freeze')
    rows=plan['attempts'];inventory={r['attempt_id']:_execution_entry(plan,r) for r in rows}
    # Every attempted native declaration, including failures/partial reports,
    # is checked before ANY capture payload is opened.
    native={}
    for row in rows:
        entry=inventory[row['attempt_id']];data._launch_state(plan,row,entry,base)
        if entry['launched']:native[row['attempt_id']]=data._population_native(row,entry,base)
    if phase=='first100':
        if (base/'first100_gate.json').exists():raise FileExistsError('First100 decision is immutable')
        decision=guard.now()
        try:
            gate=_first100_gate(plan,inventory,base,decision)
            owners={entry['owner_terminal']['path']:entry['owner_terminal'] for aid,entry in inventory.items()
                    if aid in {r['attempt_id'] for r in rows[:100]} and entry['launched']}
            for binding in owners.values():
                owner=_closed_owner(binding,base)
                need(data._time(owner['ended_utc'])<=data._time(decision), 'First100 owner ended after its gate decision')
        except ValueError as error:
            gate=dict(attempt_ids=[r['attempt_id'] for r in rows[:100]],native_successes=sum(
                n.get('success') is True and n.get('physical_validation') is True and n.get('errors')==[]
                for aid,n in native.items() if aid in {r['attempt_id'] for r in rows[:100]}),
                passed=False,decided_utc=decision,denominator=100,decision_error=str(error))
        _write(base/'first100_gate.json',gate)
        return gate
    gate,_=read(reference(base/'first100_gate.json'),base)
    unlaunched=sum(not e['launched'] for e in inventory.values())
    terminal_failure=False
    for phase_name in PHASES:
        phase_rows=rows[:100] if phase_name=='first100' else rows[100:200] if phase_name=='remainder' else rows[200:]
        terminal=Path(plan['run_root'])/phase_rows[0]['run_id']/'result.json'
        if terminal.exists():
            owner,_=read(reference(terminal),base);stage=owner.get('stage',{})
            need(owner.get('gpu_released') is True and stage.get('process_group_drained') is True and
                 owner.get('resource_guard',{}).get('sha256')==GUARD_SHA and owner.get('ended_utc'),
                 'A terminal phase must be drained and release the original guard')
            data._time(owner['ended_utc'])
            terminal_failure=terminal_failure or not (owner.get('completed') is True and stage.get('passed') is True and stage.get('exit_code')==0)
    need(not (gate.get('passed') is True and unlaunched and not terminal_failure),
         'Successful prefix is still awaiting the fixed remaining attempts')
    candidates=[];contexts=[];excluded={}
    for row in rows:
        aid=row['attempt_id'];result=native.get(aid,{})
        success=result.get('success') is True and result.get('physical_validation') is True and result.get('errors')==[]
        if success:
            try:
                need(result.get('compact_capture_complete') is True, 'Native capture completion failed')
                candidate=_capture_row(plan,row,inventory[aid]);context=data._metadata(candidate,base)
                candidates.append(candidate);contexts.append(context)
            except ValueError as error:excluded[aid]=str(error)
    if candidates:
        data._declarations(candidates,base);data._reset_preflight(candidates,base)
    captured={};episodes=[]
    for row,context in zip(candidates,contexts):
        try:
            samples,receipt,meta=data._read_payload(row,base,context);del samples
            captured[row['episode_id']]=receipt;episodes.append(row)
        except ValueError as error:excluded[row['attempt_id']]=str(error)
    outcomes=[];counts={};reasons={};chronology=True
    for i,row in enumerate(rows):
        aid=row['attempt_id'];entry=inventory[aid];result=native.get(aid,{})
        success=result.get('success') is True and result.get('physical_validation') is True and result.get('errors')==[]
        capture=captured.get(row['episode_id']);flags=result.get('_reason_flags',[])
        category=('planned_unlaunched' if not entry['launched'] else result.get('_primary_failure_kind','native_failure') if not success
            else 'capture_failure' if capture is None else 'success_zero_windows' if capture['windows']==0 else 'eligible')
        counts[category]=counts.get(category,0)+1
        for flag in flags:reasons[flag]=reasons.get(flag,0)+1
        if i>=100 and entry['launched']:
            chronology=chronology and gate.get('passed') is True and data._time(result['_started_utc'])>=data._time(gate['decided_utc'])
        outcomes.append(dict(row,**entry,native_success=success,outcome=category,reason_flags=flags,
            capture_error=excluded.get(aid)))
    execution=_write(base/'execution_inventory.json',dict(schema=data.EXECUTION_SCHEMA,finalized=True,
        attempt_plan_sha256=ref['sha256'],attempts=list(inventory.values()),first100_gate=gate))
    ledger=_write(base/'attempt_ledger.json',dict(schema=data.LEDGER_SCHEMA,attempts=outcomes))
    manifest=_write(base/'episode_manifest.json',dict(schema=data.MANIFEST_SCHEMA,task_id=data.pose.TASK_ID,
        action_semantics_id=data.given.ACTION_SEMANTICS_ID,handoff_policy_id=data.handoff.POLICY_ID,episodes=episodes))
    try:accounting=data._accounting(plan,rows,inventory,base,snapshot_phase='reconcile')
    except ValueError as error:accounting=dict(all_accounted_jobs_within_budget=False,error=str(error),qualification=False)
    summary=dict(status='blocked_final_processing_freeze',plan=ref,execution_inventory=execution,attempt_ledger=ledger,manifest=manifest,
        qualification=None,training_eligible=False,planned_attempts=240,launched_attempts=240-unlaunched,
        unlaunched_attempts=unlaunched,outcome_counts=counts,overlapping_reason_counts=reasons,
        first100_gate=gate,chronology_passed=chronology,accounting=accounting,raw_episode_results_modified=False)
    if not gate.get('passed') or not chronology or terminal_failure:summary['status']='failed_collection_version'
    _write(base/'reconciliation.json',summary)
    data.metadata_accounting(plan['accounting_scope'],base,snapshot_phase='pre_export')
    return summary


def export(ref,*,processing_freeze=None):
    """Qualify and seal once in the final official environment, after all phases."""
    if processing_freeze is None:
        return dict(status='blocked_final_processing_freeze',training_eligible=False,
            reason='Final processing/training root and its full fingerprint must be frozen before export')
    plan,path=verify_plan(ref);base=path.parent;scope=plan['accounting_scope']
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Final dataset export is CPU-only with CUDA hidden')
    need(data._ref_path(processing_freeze,base)==Path(scope['processing_sources']['descriptor_path']),
         'Exact predeclared final processing freeze required')
    data._processing_preflight(plan,dict(processing_freeze=processing_freeze),base)
    need(not Path(scope['final_seal']['path']).exists() and not (base/'export').exists(), 'Final export/seal is single use')
    summary,_=read(reference(base/'reconciliation.json'),base)
    need(summary.get('plan')==ref and summary.get('status')=='blocked_final_processing_freeze',
         'Final export requires the original completed reconciliation')
    bindings={name:reference(base/file) for name,file in [('manifest','episode_manifest.json'),
        ('execution_inventory','execution_inventory.json'),('attempt_ledger','attempt_ledger.json')]}
    need(all(summary.get(k)==v for k,v in bindings.items()), 'Original reconciliation artifact binding changed')
    population=dict(schema=data.POPULATION_SCHEMA,task_id=data.pose.TASK_ID,action_semantics_id=data.given.ACTION_SEMANTICS_ID,
        handoff_policy_id=data.handoff.POLICY_ID,passed=True,minimum_episodes=150,minimum_identities=20,minimum_success_rate=.95,
        attempt_plan=ref,execution_inventory=bindings['execution_inventory'],attempt_ledger=bindings['attempt_ledger'],
        manifest=dict(path=bindings['manifest']['path'],sha256=bindings['manifest']['sha256']),
        manifest_sha256=bindings['manifest']['sha256'],processing_freeze=processing_freeze)
    candidate=_write(base/'population_qualification.candidate.json',population)
    prepared=data.prepare_dataset(bindings['manifest']['path'],bindings['manifest']['sha256'])
    data._verify_population(candidate,prepared['episodes'])  # only these freshly rebuilt rows enter the private path
    qualification=_write(base/'population_qualification.json',population)
    data.export_dataset(bindings['manifest']['path'],bindings['manifest']['sha256'],base/'export',qualification)
    bundle=reference(base/'export/manifest.json');bundle_value,_=read(bundle,base)
    for binding in bundle_value['artifacts'].values():
        need(reference(binding['path'])==binding, 'Exported artifact changed before final seal')
    measured=data.metadata_accounting(scope,base,snapshot_phase='post_export')
    data._processing_preflight(plan,population,base)
    seal=dict(schema=SEAL_SCHEMA,task_id=data.pose.TASK_ID,action_semantics_id=data.given.ACTION_SEMANTICS_ID,
        handoff_policy_id=data.handoff.POLICY_ID,plan=ref,manifest=population['manifest'],qualification=qualification,
        processing_freeze=processing_freeze,bundle=bundle,fingerprint=bundle_value['artifacts']['reconstruction.json'],
        metadata_inventory=measured['files'],sealed_utc=guard.now(),raw_episode_results_modified=False)
    need(len((json.dumps(seal,allow_nan=False,indent=2)+'\n').encode())<=data.SEAL_ENVELOPE_BYTES, 'Final seal envelope overrun')
    sealed=_write(scope['final_seal']['path'],seal)
    verified=data.verify_final_seal(ref,sealed)
    return dict(status='sealed',plan=ref,seal=sealed,bundle=bundle,qualification=qualification,
        processing_freeze=processing_freeze,training_eligible=verified['training_eligible'])


def verify_final_seal(plan_ref,seal_ref):
    _require_pins()
    return data.verify_final_seal(plan_ref,seal_ref)


def main():
    _require_pins()
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prepare_command=commands.add_parser('prepare');prepare_command.add_argument('--settings',required=True,type=Path)
    prepare_command.add_argument('--settings-sha256',required=True);prepare_command.add_argument('--output',required=True,type=Path)
    for name in ('launch_phase','reconcile','export','verify_seal'):
        command=commands.add_parser(name);command.add_argument('--plan',required=True,type=Path);command.add_argument('--plan-sha256',required=True)
        if name=='launch_phase':command.add_argument('--phase',required=True,choices=PHASES)
        elif name=='reconcile':command.add_argument('--phase',required=True,choices=('first100','all'))
        elif name=='export':
            command.add_argument('--processing-freeze',type=Path);command.add_argument('--processing-freeze-sha256')
        else:command.add_argument('--seal',required=True,type=Path);command.add_argument('--seal-sha256',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        settings,path=read(dict(path=str(args.settings),sha256=args.settings_sha256),Path.cwd())
        settings=deepcopy(settings);extras=settings.setdefault('extra_input_refs',[])
        need(isinstance(extras,list), 'Explicit settings retained-input list required')
        extras.append(dict(category='config',reference=reference(path)))
        result=prepare(settings,args.output)
    else:
        plan_ref=dict(path=str(args.plan.resolve()),sha256=args.plan_sha256)
        if args.command=='launch_phase':result=launch_phase(plan_ref,args.phase)
        elif args.command=='reconcile':result=reconcile(plan_ref,phase=args.phase)
        elif args.command=='verify_seal':result=verify_final_seal(plan_ref,dict(path=str(args.seal.resolve()),sha256=args.seal_sha256))
        else:
            need(bool(args.processing_freeze)==bool(args.processing_freeze_sha256), 'Both processing freeze path and SHA are required')
            freeze=None if args.processing_freeze is None else dict(path=str(args.processing_freeze.resolve()),
                bytes=args.processing_freeze.stat().st_size,sha256=args.processing_freeze_sha256)
            result=export(plan_ref,processing_freeze=freeze)
    print(json.dumps(result,allow_nan=False,sort_keys=True,indent=2))


if __name__=='__main__':main()
