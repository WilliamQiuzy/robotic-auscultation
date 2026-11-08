"""New M2 state-only official LeRobot ACT baseline. No physics or capture code.

Dataset manifest ``m2_act_dataset_v1`` has ``episodes`` with explicit split,
subject_id, episode_id, capture_root, run_result={path,sha256}, and ``artifacts``
(an EXACT capture-relative inventory mapping paths to {bytes,sha256}). Paths
are relative to the dataset manifest. Inventory includes all receipts, native
240 Hz audit and archived RGB-D/Hall; preparation uses iter_training_records.
Only TRAIN and VALIDATION payloads are accepted. Hash-bound compact metadata
is inspected first, before payload inventory hashing. A mislabeled path can
therefore cause a metadata read; zero access to TEST paths is not claimed.

Preparation alone is diagnostic/ineligible. Training additionally requires a
separately hash-bound ``m2_expert_population_qualification_v1`` receipt with
manifest_sha256, passed=true, minimum_episodes>=150, minimum_identities>=20,
minimum_success_rate>=.95, and three independently bound JSON references:
attempt_plan, execution_inventory, and attempt_ledger, each {path,sha256}.
The frozen plan has schema=m2_expert_attempt_plan_v1, frozen_before_launch=true,
and attempts with attempt_id, episode_id, subject_id, split and the immutable
preclassified_reachable boolean. The execution inventory has schema=
m2_expert_execution_inventory_v1, finalized=true, attempt_plan_sha256, and one
entry per planned attempt: {attempt_id,launched,run_result} (run_result is absent
when not launched). The outcome ledger must cover EXACTLY the launched IDs,
including failures. Result paths in both are relative to the receipt directory.
The ledger JSON is
{attempts:[{attempt_id,split,subject_id,episode_id,preclassified_reachable,
success,run_result:{path,sha256}}]}. Every reachable failure stays in the
TRAIN denominator; unreachable and validation attempts are reported separately.
Native result identity is raw_multimodal_capture.episode_id, or an explicit
top-level episode_id on a launcher failure receipt. Conflicting identities,
duplicate episode IDs, repeated result hashes and repeated result paths fail.
The caller must obtain that classification and physics verdict externally.
Hashes bind evidence, they do not authenticate or create physics acceptance.

CLI: python -m sim.ausc_policy.m2_act --config CONFIG --output NEW_DIRECTORY
Config mode is prepare or train, with manifest/path SHA, optional qualification
path/SHA, fixed seed, finite steps, batch_size, eval_every, lr, weight_decay,
q_noise_std, q_dropout, lerobot_root, device. Native CPU smoke is available via
make_native; actual training requires AUSC_OWNED_RUN=1 and the shared GPU lock.
"""
import argparse
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

from . import m2_contract as contract, m2_stage as stage

UPSTREAM_COMMIT = 'eb7a54c6ff4983fc86075442bf4fa61503e7b5d1'
SCHEMA = 'm2_act_state17_history2_h20_e10_v2'
ARRAY_KEYS = ('states', 'histories', 'actions', 'origins', 'velocities', 'observation_times')
MASKED_LABEL_PROTOCOL = 'real_teacher_prefix_h20_masked_suffix_v1'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def _bound_json(reference, base):
    need(isinstance(reference, dict) and isinstance(reference.get('path'), str), 'Hash-bound JSON reference required')
    path = (Path(base)/reference['path']).resolve()
    need(isinstance(reference.get('sha256'), str) and sha256(path) == reference['sha256'], 'JSON artifact hash mismatch')
    return _json(path), path


def _scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _vector(value, shape, label):
    array = np.asarray(value)
    need(array.shape == shape and array.dtype.kind in 'fiu' and np.isfinite(array).all(), 'Invalid '+label)
    return array.astype(np.float64, copy=True)


def require_membership(split, subject_id):
    need(split in ('train', 'validation'), 'Only explicit TRAIN or VALIDATION membership permitted; TEST is sealed')
    need(sha256(contract.IDENTITY_SPLIT_PATH) == contract.IDENTITY_SPLIT_SHA256, 'Frozen identity split changed')
    splits = _json(contract.IDENTITY_SPLIT_PATH)['splits']
    need(subject_id not in contract.QUARANTINED_IDENTITIES and
         [name for name, ids in splits.items() if subject_id in ids] == [split], 'Frozen split identity mismatch or quarantined subject')


def _context_matches(event, record, meta, actor, execution_role):
    if event.get('control_owner') != _scalar(record['control_owner']):
        return False
    for side in ('controller_before', 'controller_after'):
        context = event[side]
        if meta.get('target_source') == 'given_contact_pose' and (
                context['actor'].get('reference_sha256') != actor['reference_sha256'] or
                context['actor'].get('reference_revision') != actor['reference_revision'] or
                context['actor'].get('role') != 'L2'):
            return False
        if (context.get('phase') != _scalar(record['phase']) or context.get('target') != meta['target_id'] or
                context.get('episode_id', meta['episode_id']) != meta['episode_id'] or
                context.get('control_owner', event['control_owner']) != 'L2' or
                context.get('subphase')!=execution_role or context.get('active') is not False or
                context.get('recovery') is not False or
                context['actor']['phase']!=actor['phase'] or context['actor']['owner']!='L2'):
            return False
    return True


def assemble_samples(records, native_audit, metadata, *, phase_source=stage.NATIVE_SOURCE, route=None,
                     allow_truncated=False):
    """Numerical assembly, never qualification. Labels are actual future commands.

    Every H20 sample uses 241 inclusive native ticks and both before/after contexts.
    Context transitions exclude complete chunks; corrupt/missing audits abort.
    ACT and CV require the same two real causal observations; starts are never padded.
    """
    contract.require_metadata(metadata)
    given = metadata.get('target_source') == 'given_contact_pose'
    need(type(allow_truncated) is bool and (not allow_truncated or given),
         'Truncated futures require explicit given-contact route')
    if given:
        from .m2_given_contact_pose import TASK_ID, NATIVE_SOURCE, L2_ROLES, require_route
        require_route(metadata)
        need(route == TASK_ID and phase_source == NATIVE_SOURCE, 'Explicit new-task native assembly route required')
        roles = L2_ROLES
    else:
        need(route is None, 'Given-contact route cannot consume noisy captures')
        roles = stage.L2_EXECUTION_ROLES
    require_membership(metadata['split'], metadata['subject_id'])
    need(metadata['identity_split_sha256'] == contract.IDENTITY_SPLIT_SHA256, 'Frozen split binding mismatch')
    records = list(records)
    historical=phase_source=='historical_cpu_reconstruction'
    need(historical or phase_source==stage.NATIVE_SOURCE or given,'Unknown phase source')
    if historical:
        native_audit=stage.historical_contexts(records,native_audit,metadata)
    source=stage.HISTORICAL_SOURCE if historical else phase_source
    need(len(native_audit) >= len(records)*12 and len(native_audit) < (len(records)+1)*12,
         'Native audit/frame count mismatch')
    for i, event in enumerate(native_audit):
        need(type(event.get('physics_row')) is int and event['physics_row'] == i,
             'Missing or nonconsecutive native audit tick')
        need(abs(event.get('time_s', -1)-(i+1)/240.) <= 1e-9, 'Native clock mismatch')
        need(event.get('state_audit') == 'completed' and event.get('physical_violations') == [] and
             'tick_error' in event and event['tick_error'] is None, 'Native physical audit or tick error')
        need(all(isinstance(event.get(side), dict) and 'phase' in event[side] and 'target' in event[side]
                 for side in ('controller_before', 'controller_after')), 'Missing native before/after context')
        for side,t in [('controller_before',i/240.),('controller_after',(i+1)/240.)]:
            context=event[side]
            need(isinstance(context.get('subphase'),str) and type(context.get('active')) is bool and
                 type(context.get('recovery')) is bool,'Missing native execution role/owner context')
            stage.validate_actor(context.get('actor'),t,source=source,
                                 reference_sha256=metadata.get('reference_sha256'))
    states, commands = [], []
    for i, row in enumerate(records):
        ideal_time = (i+1)*.05
        t = _scalar(row.get('observed_at_s', -1))
        need(_scalar(row.get('index')) == i and abs(t-ideal_time) <= 1e-9,
             '20 Hz observation clock mismatch')
        # Causality is relative to the actual validated observation clock.
        # Summing 240 Hz ticks and multiplying the 20 Hz frame index can
        # differ by a few ULPs; an estimate stamped exactly at capture time
        # must not be rejected for being later than the ideal grid.
        for key in ('episode_id', 'target_id', 'target_source'):
            need(_scalar(row.get(key)) == metadata[key], 'Observation metadata mismatch: '+key)
        need(0 <= _scalar(row.get('target_estimate_time_s', -1)) <= t and
             t-_scalar(row['target_estimate_time_s']) <= .05+1e-9, 'Noncausal target estimate')
        state = contract.encode_state(row['measured_q'], row['estimated_target_base'],
                                      _scalar(row['target_confidence']), metadata=metadata)
        need(np.allclose(_vector(row['state17'], (17,), 'state17'), state, rtol=0, atol=1e-10),
             'State17 disagrees with measured q and estimated target')
        states.append(state)
        if i == len(records)-1 and _scalar(row.get('record_kind')) == 'terminal_observation':
            need('q_command' not in row, 'Terminal observation cannot invent action')
            commands.append(None)
        else:
            need('q_command' in row, 'Actual q_command required; measured q is not a label')
            command = _vector(row['q_command'], (7,), 'actual command')
            endpoint = _scalar(row.get('action_endpoint_at_s', -1))
            issue = _scalar(row.get('action_issued_at_s', -1))
            need(abs(endpoint-t-.05) <= 1e-9 and t <= issue <= endpoint and
                 abs(_scalar(row.get('action_readback_at_s', -1))-endpoint) <= 1e-9,
                 'Actual command endpoint/readback mismatch')
            need(isinstance(_scalar(row.get('action_source_id')), str) and bool(_scalar(row['action_source_id'])),
                 'Explicit actual command source required')
            commands.append(command)
    output = {key: [] for key in ARRAY_KEYS}
    masks = []
    index=dict(schema='m2_causal_window_index_v2',observation_contract=contract.observation_contract(metadata),
               phase_source=source,recorded_actor_phase=not historical,training_eligible=False,
               raw_data_modified=False,episode_id=metadata['episode_id'],windows=[])
    for start, row in enumerate(records):
        if start<1 or start+(1 if allow_truncated else 20) >= len(records):
            continue  # Two real observations and twenty real future endpoints; no padding.
        if _scalar(row['phase']) not in contract.L2_PHASES or _scalar(row['control_owner']) != 'L2':
            continue
        origin_tick = 12*(start+1)-1
        origin=native_audit[origin_tick]['controller_after']
        actor=origin['actor']; role=origin['subphase']
        if actor['owner']!='L2' or role not in roles:
            continue
        horizon=min(20,len(records)-start-1) if allow_truncated else 20
        # The preceding 13 native ticks prove both history frames share context.
        history_span=native_audit[origin_tick-12:origin_tick+1]
        if (len(history_span)!=13 or
                not all(_context_matches(event,row,metadata,actor,role) for event in history_span) or
                any(_scalar(value['phase'])!=_scalar(row['phase']) or _scalar(value['control_owner'])!='L2'
                    for value in records[start-1:start+1])):
            continue
        valid=0
        for k in range(horizon):
            span=native_audit[origin_tick+12*k+(1 if k else 0):origin_tick+12*(k+1)+1]
            future=records[start+k:start+k+2]
            if (len(span)!=(13 if k==0 else 12) or
                    not all(_context_matches(event,row,metadata,actor,role) for event in span) or
                    any(_scalar(value['phase'])!=_scalar(row['phase']) or _scalar(value['control_owner'])!='L2'
                        for value in future)):
                break
            valid+=1
        if valid==0 or (not allow_truncated and valid!=20):continue
        horizon=valid
        labels=np.repeat(states[start][:7][None,:],20,axis=0)
        labels[:horizon]=np.asarray(commands[start:start+horizon])
        block = contract.encode_joint_chunk(labels, states[start][:7],
                                            arm='block_relative_joint', metadata=metadata)
        history=np.asarray(states[start-1:start+1])
        velocity=(history[1,:7]-history[0,:7])/.05
        for key,value in [('states',states[start]),('histories',history),('actions',block.values),
                          ('origins',states[start][:7]),('velocities',velocity),
                          ('observation_times',_scalar(row['observed_at_s']))]:
            output[key].append(value)
        index['windows'].append(dict(observation_indices=[start-1,start],
            action_record_indices=list(range(start,start+horizon)),
            inclusive_native_ticks=[origin_tick,origin_tick+12*horizon],history_native_ticks=[origin_tick-12,origin_tick],
            phase=actor['phase'],execution_role=role,phase_source=source))
        if allow_truncated:
            masks.append(np.arange(20)>=horizon)
            index['windows'][-1].update(valid_future_cells=horizon,label_protocol=MASKED_LABEL_PROTOCOL)
    for key,shape in [('states',(-1,17)),('histories',(-1,2,17)),('actions',(-1,20,7)),('origins',(-1,7)),
                       ('velocities',(-1,7)),('observation_times',(-1,))]:
        output[key] = np.asarray(output[key], dtype=np.float64).reshape(shape)
    output.update(training_eligible=False, physical_qualification=False, action_representation='block_relative_joint',
                  derived_index=index)
    if allow_truncated:output.update(action_is_pad=np.asarray(masks,dtype=bool).reshape(-1,20),
        label_protocol=MASKED_LABEL_PROTOCOL)
    return output


def _verify_inventory(root, inventory):
    need(isinstance(inventory, dict) and bool(inventory), 'Exact capture artifact inventory required')
    files = list(root.rglob('*'))
    need(not any(path.is_symlink() for path in files), 'Symlink in capture inventory')
    actual = {path.relative_to(root).as_posix():path for path in files if path.is_file()}
    need(set(actual) == set(inventory), 'Capture artifact inventory is not exhaustive')
    for name, path in actual.items():
        expected = inventory[name]
        need(path.stat().st_size == expected.get('bytes') and sha256(path) == expected.get('sha256'),
             'Capture artifact bytes/hash mismatch: '+name)
    return sum(path.stat().st_size for path in actual.values())


def _read_episode(row, base, *, phase_source=stage.NATIVE_SOURCE):
    from .training_capture import iter_training_records, ADAPTIVE_SCHEMA
    require_membership(row.get('split'),row.get('subject_id'))
    root = (base/row['capture_root']).resolve()
    need(root.is_dir(), 'Missing compact capture directory')
    # Read only this independently hash-bound metadata first. Hashing the whole
    # inventory here would read held-out payload bytes before validating split.
    expected = row.get('artifacts',{}).get('compact/manifest.json',{})
    metadata_path = root/'compact/manifest.json'
    need(not metadata_path.is_symlink() and metadata_path.is_file() and
         metadata_path.stat().st_size == expected.get('bytes') and
         sha256(metadata_path) == expected.get('sha256'), 'Compact metadata hash/size binding mismatch')
    manifest = _json(root/'compact/manifest.json')
    metadata = manifest['metadata']
    historical=phase_source=='historical_cpu_reconstruction'
    if not historical:
        contract.require_observation_contract(metadata)
    contract.require_metadata(metadata)
    require_membership(metadata.get('split'), metadata.get('subject_id'))
    for key in ('episode_id','subject_id','split'):
        need(metadata.get(key) == row[key], 'Explicit episode declaration differs: '+key)
    need(metadata.get('source_kind') == 'isaac_live_physics' and metadata.get('replay') is False,
         'Production preparation requires live Isaac physics, replay=false; CPU fixtures cannot promote')
    need(manifest.get('schema') == ADAPTIVE_SCHEMA, 'Only native adaptive compact v4 sources are accepted')
    total = _verify_inventory(root, row.get('artifacts'))
    outer = _json(root/'receipt.json')
    compact = _json(root/'compact/receipt.json')
    if not historical:
        need(outer.get('observation_contract')==contract.observation_contract() and
             manifest.get('observation_contract')==contract.observation_contract(),
             'Capture metadata/receipt observation contracts differ')
    run, _ = _bound_json(row.get('run_result'), base)
    _require_run_identity(run,row)
    need(run.get('physical_validation') is True and run.get('success') is True and run.get('errors') == [] and
         run.get('learned_policy') is False and run.get('compact_capture_complete') is True,
         'Hash-bound run result lacks expert physical qualification or complete capture')
    for key in ('geometry_id','geometry_source_sha256'):
        need(run.get(key) == metadata.get(key), 'Run result geometry mismatch')
    need(run.get('raw_multimodal_capture') == outer, 'Run result does not bind exact capture receipt')
    capture_schema='m2_compact_expert_capture_v1' if historical else 'm2_compact_expert_capture_v2'
    need(outer.get('schema') == capture_schema and outer.get('capture_profile') == 'compact' and
         outer.get('compact_capture_complete') is True and outer.get('errors') == [] and
         outer.get('total_bytes_complete') is True and outer.get('total_bytes') == total and
         total <= outer.get('limit_bytes',0) <= 100_000_000, 'Invalid compact capture completion/bytes receipt')
    need(outer.get('episode_id') == metadata['episode_id'] and outer.get('receipts',{}).get('compact') == compact,
         'Outer and inner receipt mismatch')
    need(sha256(root/'native_audit.jsonl.gz') == outer.get('native_audit_sha256'), 'Native audit hash mismatch')
    with gzip.open(root/'native_audit.jsonl.gz', 'rt') as stream:
        audit = [json.loads(line) for line in stream]
    need(len(audit) == outer.get('physics_steps') == compact.get('final_native_step_count') and
         outer.get('expected_frames') == compact.get('frames') == len(audit)//12, 'Receipt native/frame counts mismatch')
    # Exhaust fidelity reader and discard pixel arrays promptly: images are archived,
    # but this ACT model is state-only. No multimodal training claim is made.
    records = []
    for value in iter_training_records(root/'compact'):
        records.append({key:item for key,item in value.items() if key not in
                        ('rgb','depth_q','depth_code','depth_valid')})
    need(len(records) == compact['frames'], 'Decoded frame count mismatch')
    samples = assemble_samples(records, audit, metadata,phase_source=phase_source)
    samples['derived_index']['source_bindings']=dict(native_audit_sha256=outer['native_audit_sha256'],
        compact_metadata_sha256=expected['sha256'],run_result=row['run_result'],
        capture_inventory=row['artifacts'])
    return samples, dict(episode_id=row['episode_id'],subject_id=row['subject_id'],split=row['split'],
                        windows=len(samples['states']),capture_bytes=total,run_result=row['run_result']), metadata


def _require_run_identity(result, expected):
    identities = [value for value in (result.get('episode_id'),
        result.get('raw_multimodal_capture',{}).get('episode_id')) if value is not None]
    need(bool(identities) and all(value == expected.get('episode_id') for value in identities),
         'Hash-bound native run episode identity mismatch or missing identity')
    for key in ('subject_id','split','attempt_id'):
        if key in result:
            need(result[key] == expected.get(key), 'Hash-bound run identity mismatch: '+key)


def _attempt_index(rows, label):
    need(isinstance(rows,list) and bool(rows), 'Nonempty '+label+' attempts required')
    indexed = {}
    for row in rows:
        aid = row.get('attempt_id')
        need(isinstance(aid,str) and aid and aid not in indexed, label+' duplicate/missing attempt identity')
        indexed[aid] = row
    return indexed


def _qualification(reference, base, manifest_sha256, episode_rows):
    receipt, receipt_path = _bound_json(reference, base)
    need(receipt.get('schema') == 'm2_expert_population_qualification_v1' and receipt.get('passed') is True and
         receipt.get('manifest_sha256') == manifest_sha256, 'Population qualification receipt mismatch')
    minimum_n = receipt.get('minimum_episodes',0)
    minimum_ids = receipt.get('minimum_identities',0)
    minimum_rate = receipt.get('minimum_success_rate',0)
    need(type(minimum_n) is int and minimum_n >= 150 and type(minimum_ids) is int and minimum_ids >= 20 and
         isinstance(minimum_rate,(float,int)) and .95 <= minimum_rate <= 1, 'Qualification cut thresholds weakened')
    plan, _ = _bound_json(receipt.get('attempt_plan'),receipt_path.parent)
    need(plan.get('schema') == 'm2_expert_attempt_plan_v1' and plan.get('frozen_before_launch') is True,
         'Separate frozen prelaunch attempt plan required')
    planned = _attempt_index(plan.get('attempts'),'Frozen plan')
    episode_ids = set()
    for attempt in planned.values():
        require_membership(attempt.get('split'),attempt.get('subject_id'))
        eid = attempt.get('episode_id')
        need(isinstance(eid,str) and eid and eid not in episode_ids, 'Frozen plan duplicate episode identity')
        episode_ids.add(eid)
        need(type(attempt.get('preclassified_reachable')) is bool, 'Frozen plan reachability required')
    execution, _ = _bound_json(receipt.get('execution_inventory'),receipt_path.parent)
    need(execution.get('schema') == 'm2_expert_execution_inventory_v1' and execution.get('finalized') is True and
         execution.get('attempt_plan_sha256') == receipt['attempt_plan']['sha256'],
         'Separate finalized execution inventory bound to frozen plan required')
    inventory = _attempt_index(execution.get('attempts'),'Execution inventory')
    need(set(inventory) == set(planned), 'Execution inventory coverage differs from frozen plan')
    for entry in inventory.values():
        need(type(entry.get('launched')) is bool, 'Execution inventory requires explicit launched status')
        if not entry['launched']:
            need('run_result' not in entry, 'Unlaunched attempt cannot carry run result')
    launched = {aid:entry for aid,entry in inventory.items() if entry['launched']}
    ledger, _ = _bound_json(receipt.get('attempt_ledger'), receipt_path.parent)
    attempts = _attempt_index(ledger.get('attempts'),'Outcome ledger')
    need(set(attempts) == set(launched), 'Outcome ledger coverage must include exactly ALL launched attempts and failures')
    seen_episodes, seen_hashes, seen_paths, good = set(), set(), set(), set()
    denominator = successes = unreachable = validation = 0
    for aid,attempt in attempts.items():
        require_membership(attempt.get('split'), attempt.get('subject_id'))
        need(all(attempt.get(key) == planned[aid].get(key) for key in
                 ('episode_id','subject_id','split','preclassified_reachable')),
             'Outcome identity/reachability differs from frozen plan')
        need(type(attempt.get('preclassified_reachable')) is bool and type(attempt.get('success')) is bool,
             'Attempt reachability and outcome must be explicit')
        need(attempt.get('run_result') == launched[aid].get('run_result'),
             'Outcome run result differs from execution inventory binding')
        result, result_path = _bound_json(attempt.get('run_result'), receipt_path.parent)
        _require_run_identity(result,attempt)
        digest = attempt['run_result']['sha256']
        need(attempt['episode_id'] not in seen_episodes and digest not in seen_hashes and result_path not in seen_paths,
             'Duplicate episode or run receipt identity across distinct attempts')
        seen_episodes.add(attempt['episode_id']);seen_hashes.add(digest);seen_paths.add(result_path)
        success = result.get('success') is True and result.get('physical_validation') is True and result.get('errors') == []
        if success:
            from .geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
            need(result.get('learned_policy') is False and
                 result.get('geometry_id') == GEOMETRY_ID and
                 result.get('geometry_source_sha256') == GEOMETRY_SOURCE_SHA256,
                 'Population success must be an expert on the current geometry')
        need(success == attempt['success'], 'Attempt outcome differs from hash-bound run result')
        if attempt['split'] != 'train':
            validation += 1
            continue
        if not attempt['preclassified_reachable']:
            unreachable += 1
            continue
        denominator += 1
        successes += int(success)
        if success:
            good.add((attempt['episode_id'],attempt['subject_id'],attempt['run_result']['sha256']))
    train = [row for row in episode_rows if row['split'] == 'train' and row.get('windows',0) > 0]
    need(len(train) >= minimum_n and len({row['subject_id'] for row in train}) >= minimum_ids,
         'Qualified TRAIN episode/identity counts below cut threshold')
    need(all((row['episode_id'],row['subject_id'],row['run_result']['sha256']) in good for row in train),
         'TRAIN episode missing from successful reachable attempt ledger')
    need(denominator > 0 and successes/denominator >= minimum_rate, 'Reachable denominator success rate below qualification threshold')
    return dict(passed=True, receipt_sha256=reference['sha256'], reachable_denominator=denominator,
                reachable_successes=successes, success_rate=successes/denominator,
                attempt_plan_sha256=receipt['attempt_plan']['sha256'],
                execution_inventory_sha256=receipt['execution_inventory']['sha256'],
                planned_attempts=len(planned),launched_attempts=len(launched),
                unlaunched_attempts=len(planned)-len(launched),
                unreachable_attempts=unreachable, validation_attempts=validation,
                qualified_train_episodes=len(train),qualified_train_identities=len({r['subject_id'] for r in train}))


def prepare_dataset(manifest_path, expected_sha256, *, qualification=None, phase_source=stage.NATIVE_SOURCE):
    """Validate bound split metadata before payload reads; reject TEST payloads."""
    manifest_path = Path(manifest_path).resolve()
    need(sha256(manifest_path) == expected_sha256, 'Dataset manifest hash mismatch')
    manifest = _json(manifest_path)
    need(phase_source in (stage.NATIVE_SOURCE,'historical_cpu_reconstruction'),'Unknown phase source')
    need(qualification is None or phase_source==stage.NATIVE_SOURCE,
         'Historical CPU stage reconstruction is diagnostic only; cannot qualify training')
    need(manifest.get('schema') == 'm2_act_dataset_v1' and bool(manifest.get('episodes')), 'Explicit dataset manifest required')
    rows = manifest['episodes']
    seen = set()
    # Preflight every declared split before opening ANY episode path.
    for row in rows:
        require_membership(row.get('split'), row.get('subject_id'))
        eid = row.get('episode_id')
        need(isinstance(eid,str) and eid and eid not in seen, 'Duplicate/missing episode ID')
        seen.add(eid)
    grouped = {'train':[], 'validation':[]}
    receipts, metadata_rows = [], []
    for row in rows:
        sample, receipt, metadata = _read_episode(row, manifest_path.parent,phase_source=phase_source)
        grouped[row['split']].append(sample)
        receipts.append(receipt)
        metadata_rows.append(metadata)
    output = {}
    for split, data in grouped.items():
        need(bool(data), 'Both TRAIN and VALIDATION episodes are required')
        output[split] = {key:np.concatenate([sample[key] for sample in data],axis=0) for key in ARRAY_KEYS}
        need(len(output[split]['states']) > 0, 'No complete boundary-safe H20 chunks in '+split)
    output.update(schema=SCHEMA, training_eligible=False, test_payloads_opened=False,
                  split_access_scope='hash-bound compact metadata preflight; no claim of zero TEST metadata path access',
                  manifest_sha256=expected_sha256, manifest_path=str(manifest_path), episodes=receipts,
                  metadata=dict(metadata_rows[0],observation_contract=contract.observation_contract()),
                  derived_index=dict(schema='m2_causal_dataset_index_v2',manifest_sha256=expected_sha256,
                    observation_contract=contract.observation_contract(),phase_source=phase_source,
                    recorded_actor_phase=phase_source==stage.NATIVE_SOURCE,
                    episodes=[sample['derived_index'] for group in grouped.values() for sample in group]),
                  qualification_reference=qualification,
                  normalization=fit_normalization(output['train'],split='train'))
    if qualification is not None:
        output['qualification'] = _qualification(qualification,manifest_path.parent,expected_sha256,receipts)
        output['training_eligible'] = True
    return output


def valid_action_mask(samples):
    """Explicit real-prefix contract; absent future values are never supervision."""
    if 'action_is_pad' not in samples:
        need('label_protocol' not in samples,'Masked protocol requires actual mask')
        return np.ones(np.asarray(samples['actions']).shape[:2],bool)
    mask=np.asarray(samples['action_is_pad'])
    actions=np.asarray(samples['actions'])
    need(samples.get('label_protocol')==MASKED_LABEL_PROTOCOL and mask.dtype==np.dtype(bool) and
         mask.shape==actions.shape[:2] and mask.ndim==2 and mask.shape[1]==20 and
         np.all(~mask[:,0]) and np.all(np.diff(mask.astype(int),axis=1)>=0) and
         np.all(actions[mask]==0),'Explicit finite nonempty real-prefix mask required')
    return ~mask


def fit_normalization(samples, *, split, observation_schema=None):
    need(split == 'train', 'Normalization is TRAIN-only')
    state = np.asarray(samples['states'],float)
    actions = np.asarray(samples['actions'],float)
    need(state.ndim == 2 and state.shape[1] == 17 and len(state) and actions.shape == (len(state),20,7)
         and np.isfinite(state).all() and np.isfinite(actions).all(), 'Finite nonempty H20 TRAIN samples required')
    history=_vector(samples['histories'],(len(state),2,17),'causal history')
    need(np.array_equal(history[:,1],state),'History current state mismatch')
    valid=valid_action_mask(samples)
    action_values=actions[valid] if 'action_is_pad' in samples else actions
    return dict(fitted_split='train',observation_contract=observation_schema or contract.observation_contract(),
                state_mean=history.mean((0,1)).tolist(),state_std=np.maximum(history.std((0,1)),1e-6).tolist(),
                action_mean=action_values.mean(0 if 'action_is_pad' in samples else (0,1)).tolist(),
                action_std=np.maximum(action_values.std(0 if 'action_is_pad' in samples else (0,1)),1e-6).tolist())


def pack_batch(samples, stats, indices, *, training, rng=None, q_noise_std=0., q_dropout=0., observation_schema=None):
    need(stats.get('fitted_split') == 'train', 'Normalization must be fitted on TRAIN')
    need(np.isfinite(q_noise_std) and q_noise_std >= 0 and 0 <= q_dropout <= 1, 'Invalid q augmentation')
    need(stats.get('observation_contract')==(observation_schema or contract.observation_contract()),'Normalization observation contract mismatch')
    history=np.array(samples['histories'][indices],copy=True)
    need(np.array_equal(history[:,1],samples['states'][indices]),'History/current state mismatch')
    actions = np.asarray(samples['actions'][indices])
    valid=valid_action_mask(samples)
    if training:
        need(rng is not None, 'Seeded augmentation RNG required')
        history[:,:,:7] += rng.normal(0,q_noise_std,size=history[:,:,:7].shape)
    history = (history-np.asarray(stats['state_mean']))/np.asarray(stats['state_std'])
    state=history[:,1]
    if training:
        history[:,:,:7][rng.random(history[:,:,:7].shape) < q_dropout] = 0
    action = (actions-np.asarray(stats['action_mean']))/np.asarray(stats['action_std'])
    return {'observation.state':state[:,:7].astype(np.float32),
            'observation.environment_state':np.concatenate([state[:,7:],history[:,0]],axis=1).astype(np.float32),
            'action':action.astype(np.float32),'action_is_pad':(~valid[indices]).copy()}


def evaluate_predictions(predicted_relative, samples):
    truth = np.asarray(samples['actions'])
    predicted = _vector(predicted_relative,truth.shape,'predicted H20 actions')
    need(len(truth)>0 and truth.shape[1:] == (20,7), 'All twenty horizons required')
    baseline = np.asarray(samples['velocities'])[:,None,:]*(np.arange(1,21)*.05)[None,:,None]
    valid=valid_action_mask(samples);counts=valid.sum(0)
    need(np.all(counts>0),'Every evaluated horizon requires real labels')
    result = {}
    for name,value in [('act',predicted),('hold',np.zeros_like(truth)),('constant_velocity',baseline)]:
        error = abs(value-truth)
        result[name] = (dict(full_chunk_mae_rad=float(error[valid].mean()),
            per_horizon_mae_rad=((error*valid[:,:,None]).sum((0,2))/(counts*7)).tolist()) if 'action_is_pad' in samples else
            dict(full_chunk_mae_rad=float(error.mean()),per_horizon_mae_rad=error.mean((0,2)).tolist()))
    result.update(sample_count=len(truth),horizon=20,selection_metric='act.full_chunk_mae_rad',
                  evaluated_in='absolute radians (fixed block origin cancels in error)')
    if 'action_is_pad' in samples:result.update(label_protocol=MASKED_LABEL_PROTOCOL,
        valid_future_cells=int(valid.sum()),per_horizon_valid_cells=counts.tolist())
    return result


def native_spec():
    return dict(n_obs_steps=1,chunk_size=20,n_action_steps=10,temporal_ensemble_coeff=None,
                dim_model=256,n_heads=4,n_encoder_layers=2,n_decoder_layers=1,
                dim_feedforward=1024,use_vae=False,dropout=.1,
                pretrained_backbone_weights=None,vision_backbone='resnet18')


def make_native(config):
    """Construct pinned official ACT; pure CPU smoke may run without experiment ownership."""
    vendor = Path(config['lerobot_root']).resolve()
    device = config.get('device','cpu')
    need(device in ('cpu','cuda'), 'Explicit CPU/CUDA device required')
    if device == 'cuda':
        need(sys.platform == 'linux' and os.environ.get('AUSC_OWNED_RUN') == '1', 'CUDA requires owned Linux execution')
    else:
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU smoke requires CUDA_VISIBLE_DEVICES empty')
    commit = subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
    need(commit == UPSTREAM_COMMIT, 'Official ACT commit differs')
    names = ['src/lerobot/policies/act/'+name for name in ('configuration_act.py','modeling_act.py','processor_act.py')]
    names += ['src/lerobot/configs/types.py','src/lerobot/configs/policies.py']
    subprocess.run(['git','-C',str(vendor),'diff','--exit-code','HEAD','--',*names],check=True,stdout=subprocess.DEVNULL)
    need(Path(sys.prefix).resolve() == (vendor/'.venv').resolve(), 'Use existing isolated official ACT environment')
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    need(Path(inspect.getfile(ACTPolicy)).resolve() == vendor/'src/lerobot/policies/act/modeling_act.py', 'Wrong ACT import provenance')
    seed = config.get('seed',20261026)
    need(type(seed) is int and 0 <= seed < 2**32, 'Fixed integer seed required')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == 'cuda':
        need(torch.cuda.is_available(), 'CUDA unavailable')
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    cfg = ACTConfig(input_features={'observation.state':PolicyFeature(FeatureType.STATE,(7,)),
        'observation.environment_state':PolicyFeature(FeatureType.ENV,(27,))},
        output_features={'action':PolicyFeature(FeatureType.ACTION,(7,))},
        normalization_mapping={key:NormalizationMode.IDENTITY for key in ('STATE','ENV','ACTION')},
        **native_spec(),device=device,use_amp=False,push_to_hub=False,
        optimizer_lr=config.get('lr',1e-4),optimizer_weight_decay=config.get('weight_decay',1e-4))
    need(cfg.device == device, 'Unexpected automatic device substitution')
    policy = ACTPolicy(cfg).to(device)
    need(not hasattr(policy.model,'backbone') and not hasattr(policy.model,'vae_encoder'), 'Unexpected image/VAE encoder')
    return torch,policy,{name:sha256(vendor/name) for name in names}


def _tensors(torch, batch, device):
    return {key:torch.from_numpy(value).to(device) for key,value in batch.items()}


def fit(prepared, config, output):
    """Finite training; only full-chunk VALIDATION MAE selects checkpoint."""
    need(prepared.get('training_eligible') is True and prepared.get('qualification',{}).get('passed') is True,
         'External population qualification required before fitting')
    # Rebind original sources at fit time. A caller-set boolean or stale cache is insufficient.
    prepared = prepare_dataset(prepared['manifest_path'],prepared['manifest_sha256'],
                               qualification=prepared['qualification_reference'])
    need(sys.platform == 'linux' and os.environ.get('AUSC_OWNED_RUN') == '1', 'Training requires owned Linux execution')
    for key in ('steps','batch_size','eval_every'):
        need(type(config.get(key)) is int and 1 <= config[key] <= 1_000_000, 'Finite positive '+key+' required')
    import fcntl
    lock = (Path(config['base'])/'locks/isaac_gpu.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        torch,policy,sources = make_native(config)
        device=config.get('device','cpu'); seed=config.get('seed',20261026)
        rng=np.random.default_rng(seed); augmentation=np.random.default_rng(seed+1)
        optimizer=torch.optim.AdamW(policy.parameters(),lr=config.get('lr',1e-4),weight_decay=config.get('weight_decay',1e-4))
        output=Path(output); output.mkdir(parents=True,exist_ok=False)
        stats=prepared['normalization']; train=prepared['train']; validation=prepared['validation']
        _save(output/'configuration.json',config); _save(output/'normalization.json',stats)
        best=float('inf'); best_step=None
        with (output/'metrics.jsonl').open('x') as log:
            for step in range(1,config['steps']+1):
                policy.train()
                ids=rng.integers(len(train['states']),size=config['batch_size'])
                batch=_tensors(torch,pack_batch(train,stats,ids,training=True,rng=augmentation,
                    q_noise_std=config.get('q_noise_std',.002),q_dropout=config.get('q_dropout',.05)),device)
                optimizer.zero_grad(set_to_none=True)
                loss,_=policy(batch)
                need(bool(torch.isfinite(loss)), 'Nonfinite ACT loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                log.write(json.dumps(dict(step=step,train_loss=float(loss.detach().cpu())))+'\n')
                if step%config['eval_every'] and step != config['steps']:
                    continue
                policy.eval(); predictions=[]
                with torch.no_grad():
                    for start in range(0,len(validation['states']),config['batch_size']):
                        ids=np.arange(start,min(start+config['batch_size'],len(validation['states'])))
                        batch=_tensors(torch,pack_batch(validation,stats,ids,training=False),device)
                        batch={key:value for key,value in batch.items() if key.startswith('observation.')}
                        result=policy.predict_action_chunk(batch).cpu().numpy()
                        predictions.append(result*np.asarray(stats['action_std'])+np.asarray(stats['action_mean']))
                metrics=evaluate_predictions(np.concatenate(predictions),validation)
                score=metrics['act']['full_chunk_mae_rad']
                log.write(json.dumps(dict(step=step,split='validation',metrics=metrics),allow_nan=False)+'\n');log.flush()
                if score < best:
                    best=score;best_step=step
                    checkpoint=dict(schema=SCHEMA,state_dict=policy.state_dict(),step=step,
                        model=native_spec(),normalization=stats,config=config,metadata=prepared['metadata'],
                        observation_contract=contract.observation_contract(),
                        source_hashes=sources,upstream_commit=UPSTREAM_COMMIT,manifest_sha256=prepared['manifest_sha256'],
                        qualification=prepared['qualification'],selection_split='validation',metrics=metrics,
                        state_only=True,cpu_fixture_only=False,
                        action_representation='block_relative_joint',closed_loop_evaluated=False)
                    torch.save(checkpoint,output/'best.pt')
        _save(output/'COMPLETE.json',dict(completed=True,steps=config['steps'],best_step=best_step,
            selection_split='validation',validation_h20_mae_rad=best,test_payloads_opened=False,
            split_access_scope=prepared['split_access_scope'],
            closed_loop_evaluated=False,physical_acceptance=False,state_only=True))
    finally:
        lock.close()


class ChunkPredictor:
    """Inference hook: complete immutable H20 block; executor owns E10 and safety."""
    def __init__(self, torch, policy, normalization, metadata, device='cpu', *, observation_schema=None, checkpoint_schema=SCHEMA):
        contract.require_observation_contract(metadata)
        self.observation_schema=observation_schema or contract.observation_contract()
        self.checkpoint_schema=checkpoint_schema
        need(metadata['observation_contract']==self.observation_schema and
             normalization.get('observation_contract')==self.observation_schema,
             'Checkpoint normalization observation contract mismatch')
        self.torch,self.policy,self.stats,self.metadata,self.device=torch,policy,normalization,dict(metadata),device

    def __call__(self, measured_q, estimated_target_base, confidence, *, metadata, history, history_times):
        contract.require_observation_contract(metadata)
        for key in ('contract_id','geometry_id','geometry_source_sha256','target_source'):
            need(metadata[key] == self.metadata[key], 'Inference/checkpoint metadata mismatch: '+key)
        state=contract.encode_state(measured_q,estimated_target_base,confidence,metadata=metadata)
        history=_vector(history,(2,17),'causal inference history')
        times=_vector(history_times,(2,),'causal inference history times')
        need(times[0]>=0 and abs(times[1]-times[0]-.05)<=1e-8 and
             np.array_equal(history[1],state),'Inference history/current state or clock mismatch')
        normalized=(history-np.asarray(self.stats['state_mean']))/np.asarray(self.stats['state_std'])
        batch={'observation.state':normalized[None,1,:7].astype(np.float32),
               'observation.environment_state':np.r_[normalized[1,7:],normalized[0]][None].astype(np.float32)}
        self.policy.eval()
        with self.torch.no_grad():
            relative=self.policy.predict_action_chunk(_tensors(self.torch,batch,self.device)).cpu().numpy()[0]
        relative=relative*np.asarray(self.stats['action_std'])+np.asarray(self.stats['action_mean'])
        # Never call select_action (queue) and never reanchor per predicted step.
        return contract.JointActionBlock('block_relative_joint',relative,state[:7],
            dict(metadata,training_eligible=False,physical_qualification=False,checkpoint_policy=self.checkpoint_schema))


def validate_checkpoint(checkpoint, *, allow_cpu_fixture=False, expected_schema=SCHEMA, observation_schema=None):
    need(checkpoint.get('schema') == expected_schema and checkpoint.get('upstream_commit') == UPSTREAM_COMMIT and
         checkpoint.get('model') == native_spec() and checkpoint.get('selection_split') == 'validation' and
         checkpoint.get('action_representation') == 'block_relative_joint' and checkpoint.get('state_only') is True,
         'Checkpoint architecture/action/provenance mismatch')
    contract.require_metadata(checkpoint.get('metadata',{}))
    contract.require_observation_contract(checkpoint['metadata'])
    need(checkpoint.get('observation_contract')==(observation_schema or contract.observation_contract()) and
         checkpoint['metadata']['observation_contract']==checkpoint.get('observation_contract'),
         'Checkpoint observation contract mismatch')
    if checkpoint.get('cpu_fixture_only') is True:
        need(allow_cpu_fixture, 'CPU fixture checkpoint requires explicitly isolated fixture route')
    else:
        need(checkpoint.get('qualification',{}).get('passed') is True and
             isinstance(checkpoint.get('manifest_sha256'),str) and len(checkpoint['manifest_sha256']) == 64 and
             checkpoint['metadata'].get('source_kind') == 'isaac_live_physics' and
             checkpoint['metadata'].get('replay') is False,
             'Production checkpoint requires qualified live source and population qualification')


def load_checkpoint(path, expected_sha256, *, lerobot_root, device='cpu', allow_cpu_fixture=False):
    need(sha256(path) == expected_sha256, 'Checkpoint hash mismatch')
    import torch
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    validate_checkpoint(checkpoint,allow_cpu_fixture=allow_cpu_fixture)
    need(not checkpoint.get('cpu_fixture_only') or device == 'cpu', 'Fixture checkpoint is CPU-only')
    config=dict(checkpoint['config'],lerobot_root=str(lerobot_root),device=device)
    torch,policy,sources=make_native(config)
    need(sources == checkpoint['source_hashes'], 'Official ACT source hashes changed')
    policy.load_state_dict(checkpoint['state_dict'],strict=True)
    return ChunkPredictor(torch,policy,checkpoint['normalization'],checkpoint['metadata'],device)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();config=_json(args.config)
    need(config.get('mode') in ('prepare','train'), 'Explicit prepare/train mode required')
    phase_source=config.get('phase_source',stage.NATIVE_SOURCE)
    need(config['mode']=='prepare' or phase_source==stage.NATIVE_SOURCE,
         'Historical CPU stage reconstruction is prepare/diagnostic only')
    prepared=prepare_dataset(config['manifest'],config['manifest_sha256'],qualification=config.get('qualification'),
                             phase_source=phase_source)
    if config['mode']=='train':
        fit(prepared,config,args.output)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        for split in ('train','validation'):
            np.savez_compressed(args.output/(split+'.npz'),**{key:prepared[split][key] for key in ARRAY_KEYS})
        _save(args.output/'derived_window_index.json',prepared['derived_index'])
        _save(args.output/'preparation.json',{key:value for key,value in prepared.items() if key not in ('train','validation')})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
