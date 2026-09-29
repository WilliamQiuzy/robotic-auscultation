"""Finite official ACT baseline. Imports only NumPy until an owned training run.

Source NPZs are read-only. Only six non-image members are streamed to new NPY
caches, then opened with mmap_mode='r'. Missing raw L2 boundaries stop preparation.
No Isaac, remote execution, automatic installation, or resumed training exists.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

UPSTREAM_COMMIT = 'eb7a54c6ff4983fc86075442bf4fa61503e7b5d1'
TRAIN16 = (
    'R1_observed_armature02_v1', 'L1_observed_armature02_v1',
    'R2_observed_armature02_v1', 'L2_observed_armature02_v1',
    'R3_observed_armature02_v1', 'L3_observed_armature02_v1',
    'R4_observed_armature02_v1', 'L4_observed_armature02_v1',
    'R1_train_yaw_positive_v1_001', 'L1_train_yaw_positive_v1_001',
    'R2_train_yaw_positive_v1_001', 'L2_train_yaw_positive_v1_001',
    'R4_train_yaw_positive_v1_001', 'R4_train_B7300_prefix20_v1_001',
    'R4_train_B7300_approach_entry_v1_001', 'R4_train_B7300_outward_onset_v1_001',
)
TARGETS = tuple(side + str(level) for level in range(1, 6) for side in ('R', 'L'))
MEMBERS = ('state', 'actions', 'times', 'phases', 'targets', 'target_sources')
L2_SUBPHASES = frozenset(('transfer', 'inter_point_transfer', 'lower_approach', 'align'))
ABSOLUTE = 'absolute_joint_radians'
BLOCK_RELATIVE = 'block_relative_joint_radians'


def need(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError('Nonfinite JSON: ' + value)))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def pinned_json(path, digest):
    need(isinstance(digest, str) and len(digest) == 64 and sha256(path) == digest,
         'Source JSON hash differs: ' + str(path))
    return read_json(path)


def relative_path(root, value):
    root = Path(root).resolve()
    need(isinstance(value, str) and value and not Path(value).is_absolute()
         and '..' not in Path(value).parts, 'Source member must be relative')
    result = (root / value).resolve()
    need(result.is_relative_to(root), 'Source path escaped manifest directory')
    return result


def validate_config(config):
    need(config['schema'] == 'ausc_m0_act_v1' and config['experiment'] == 'act_baseline', 'Wrong ACT config')
    need(tuple(config['train_episode_ids']) == TRAIN16, 'Exact original TRAIN16 required')
    need(config['upstream_commit'] == UPSTREAM_COMMIT, 'Official ACT commit differs')
    need(config['mode'] in ('prepare', 'native_smoke', 'train32', 'train'), 'Unknown finite mode')
    need(config['horizon'] == 10 and config['execute_steps'] == 5 and config['period_s'] == .05,
         'M0 requires H10/E5 and 50ms actions')
    need(config['action_representation'] in (ABSOLUTE, BLOCK_RELATIVE) and config['state_dim'] == 19,
         'M0 state/action representation differs')
    if config['action_representation'] == BLOCK_RELATIVE:
        need(config['mode'] in ('train32', 'native_smoke') and config.get('prepared_manifest'),
             'Block relative control requires pinned source preparation and finite train32/native_smoke mode')
    need(config['device'] in ('cpu', 'cuda') and config['seed'] == 61, 'Explicit device/seed required')
    if config['mode'] == 'native_smoke':
        need(config['device'] == 'cpu', 'Native shape/backward smoke must use CPU')
    need(config['modes']['train32'] == dict(steps=512, batch_size=32, q_dropout=0.0)
         and config['modes']['train'] == dict(steps=2000, batch_size=32, q_dropout=0.1), 'Finite budgets differ')
    need(config['model'] == dict(dim_model=256, n_heads=4, dim_feedforward=1024,
         n_encoder_layers=2, n_decoder_layers=1, use_vae=False, dropout=0.0), 'ACT architecture differs')
    need(config['optimizer'] == dict(lr=1e-4, betas=[.9, .999], weight_decay=1e-4, grad_clip_norm=10.),
         'Finite optimizer recipe differs')


def resolve_train_records(manifest_path, expected_sha256, *, expected_ids=TRAIN16):
    """Read source metadata only; never resolve/open a heldout array or report."""
    manifest_path = Path(manifest_path).resolve()
    composite = pinned_json(manifest_path, expected_sha256)
    need(composite.get('schema') == 'wam_pi05_v10_recovery_composite_manifest_v1'
         and composite.get('sealed') is True, 'Sealed composite manifest required')
    sources, records = {}, {}
    for source in composite['sources']:
        sid = source['id']; binding = source['manifest']; path = Path(binding['path'])
        need(sid not in sources and path.is_absolute(), 'Duplicate/noncanonical source manifest')
        data = pinned_json(path, binding['sha256'])
        need(data.get('schema') == 'wam_vla_training_manifest_v1' and data.get('sealed') is True
             and data.get('horizon') == 10 and data.get('action_period_s') == .05, 'Original raw source contract differs')
        sources[sid] = path
        for record in data['episodes']:
            eid = record['episode_id']
            need(eid not in records, 'Duplicate source episode')
            records[eid] = (sid, record)
    mappings = composite['episodes']
    need(len({x['episode_id'] for x in mappings}) == len(mappings), 'Duplicate composite episode')
    need({x['episode_id'] for x in mappings} == set(records), 'Incomplete composite mapping')
    train = {}
    for item in mappings:
        sid, record = records[item['episode_id']]
        need(sid == item['source_id'], 'Composite source remapped')
        if record['split'] != 'train':
            continue
        eid = record['episode_id']
        need(eid in expected_ids and record['target'] in TARGETS[:8], 'Unexpected/heldout TRAIN identity')
        own = dict(record, source_manifest=str(sources[sid]), source_id=sid)
        own['array_path'] = str(relative_path(sources[sid].parent, record['path']))
        own['report_path'] = str(relative_path(sources[sid].parent, record['qualification_path']))
        train[eid] = own
    need(set(train) == set(expected_ids), 'Missing or changed exact TRAIN episode set')
    need(len({r['sha256'] for r in train.values()}) == len(train), 'Duplicate TRAIN content')
    for _, record in records.values():
        if record['split'] != 'train':
            need(not any(record.get('motion_family') and record['motion_family'] == r.get('motion_family')
                         for r in train.values()), 'TRAIN/heldout motion ancestry overlaps')
    return [train[eid] for eid in expected_ids]


def open_episode_mmaps(path, expected_sha256, cache):
    """NPZ mmap_mode is ineffective: stream selected NPY members before mmap."""
    path, cache = Path(path), Path(cache)
    need(sha256(path) == expected_sha256, 'Source NPZ hash differs: ' + str(path))
    cache.mkdir(parents=True, exist_ok=False)
    arrays, bindings = {}, {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)), 'Duplicate NPZ members')
        for key in MEMBERS:
            member = key + '.npy'
            need(member in names, 'Missing source array: ' + member)
            destination = cache / member
            with archive.open(member) as source, destination.open('xb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            arrays[key] = np.load(destination, mmap_mode='r', allow_pickle=False)
            need(isinstance(arrays[key], np.memmap) and not arrays[key].flags.writeable,
                 'Read-only mmap required')
            bindings[key] = dict(path=str(destination.resolve()), sha256=sha256(destination))
    return arrays, bindings


def bound_jsonl(path, expected_sha256):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for line in stream:
            digest.update(line)
            yield json.loads(line)
    need(digest.hexdigest() == expected_sha256, 'Raw trace hash differs: ' + str(path))


def l2_mask(arrays, pairs, frames, physics):
    """Verify actual next-command labels and pre-contact raw subphase boundary."""
    n = len(arrays['times'])
    need(len(pairs) == n and n > 0, 'Missing original temporal label pairs')
    by_frame = {f['index']: f for f in frames}
    need(len(by_frame) == len(frames), 'Duplicate acquisition frame index')
    starts, ends = defaultdict(list), defaultdict(list)
    valid = np.ones(n, bool); seen = np.zeros(n, bool)
    for i, pair in enumerate(pairs):
        first, last = pair['frame_index'], pair['following_frame_index']
        start, stop = pair['physics_interval_inclusive']
        need(type(first) is int and last == first + 1 and first in by_frame and last in by_frame,
             'Missing/nonadjacent acquired frame')
        f, g = by_frame[first], by_frame[last]
        need(stop-start == 12 and pair['command_physics_row'] == stop
             and f['physics_row'] == start and g['physics_row'] == stop, 'Original command/physics mapping differs')
        need(abs(float(f['time_s'])-float(arrays['times'][i])) < 1e-8
             and abs(float(g['time_s'])-float(f['time_s'])-.05) < 1e-7, 'Original 50ms acquired clock differs')
        valid[i] = (f.get('eligible') is True and g.get('eligible') is True
                    and f.get('target') == g.get('target') == str(arrays['targets'][i]))
        starts[start].append(i); ends[stop].append(i)
    active = set(); previous = deque(maxlen=3); contact = None; origin_phase = {}
    for row_index, row in enumerate(physics):
        active.update(starts[row_index])
        command = np.asarray(row.get('q_command'), float)
        need(command.shape == (7,) and np.isfinite(command).all(), 'Missing/nonfinite raw q_command')
        if row.get('subphase') == 'contact_cycle' and contact is None:
            need(len(previous) == 3 and all(r[0] == 'align' for r in previous)
                 and np.max(np.abs(np.diff(np.stack([r[1] for r in previous]), axis=0))) < 1e-10,
                 'Cannot prove stationary raw align -> contact_cycle handoff')
            contact = row_index
        for i in tuple(active):
            target = str(arrays['targets'][i])
            if row_index == pairs[i]['physics_interval_inclusive'][0]:
                origin_phase[i] = row.get('phase')
                q = np.asarray(row.get('q'), np.float32)
                need(q.shape == (7,) and np.array_equal(q, arrays['state'][i, :7]), 'Raw measured q binding differs')
                need(abs(float(row['time_s'])-float(arrays['times'][i])) < 1e-8, 'Raw observation time differs')
            valid[i] &= (row.get('subphase') in L2_SUBPHASES and row.get('recovery') is False
                         and row.get('target') == target and contact is None
                         and origin_phase[i] is not None and row.get('phase') == origin_phase[i])
        for i in ends[row_index]:
            need(i in active, 'Label endpoint appeared before origin')
            need(np.array_equal(command.astype(np.float32), arrays['actions'][i].astype(np.float32)),
                 'Raw following command differs; no second label shift allowed')
            need(abs(float(row['time_s'])-(float(arrays['times'][i])+.05)) < 1e-7,
                 'Raw following command timestamp differs')
            seen[i] = True; active.remove(i)
        previous.append((row.get('subphase'), command))
    need(contact is not None, 'Missing raw align -> contact_cycle boundary; L2 cutoff unproven')
    need(seen.all() and not active, 'Raw trace missing label endpoints')
    return valid, dict(first_contact_physics_row=contact, verified_label_pairs=n,
                       l2_label_pairs=int(valid.sum()), stationary_handoff_proved=True,
                       boundary_source='raw align to contact_cycle, before all contact actions')


def window_starts(arrays, pairs, l2_valid, cross_boundary, horizon):
    times, phases, targets = (arrays[k] for k in ('times', 'phases', 'targets'))
    need(times.dtype == np.float64 and len(pairs) == len(times) == len(l2_valid)
         and np.isfinite(times).all() and (np.diff(times) > 0).all(), 'Invalid source clock/length')
    need(isinstance(cross_boundary, list) and all(type(i) is int and 0 <= i < len(times)
         for i in cross_boundary), 'Missing/invalid cross-boundary label indices')
    valid = np.array(l2_valid, dtype=bool, copy=True)
    valid[cross_boundary] = False
    result = []
    for start in range(len(times)-horizon+1):
        stop = start+horizon
        if not valid[start:stop].all() or str(phases[start]) not in ('FREE', 'APPROACH'):
            continue
        if (np.all(phases[start:stop] == phases[start]) and np.all(targets[start:stop] == targets[start])
                and np.allclose(np.diff(times[start:stop]), .05, rtol=0, atol=1e-7)
                and all(pairs[i]['following_frame_index'] == pairs[i+1]['frame_index']
                        for i in range(start, stop-1))):
            result.append(start)
    return result


def small_state(raw, target, phase):
    raw = np.asarray(raw, dtype=np.float64)
    need(raw.shape == (41,) and np.isfinite(raw).all(), 'Finite physical state41 required')
    need(target in TARGETS and phase in ('FREE', 'APPROACH'), 'Unknown L2 target/phase')
    need(0 <= raw[22] <= 1, 'Invalid acquired confidence')
    columns = raw[35:41].reshape(2, 3).T
    rotation = np.column_stack((columns, np.cross(columns[:, 0], columns[:, 1])))
    need(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
         and abs(np.linalg.det(rotation)-1) < 1e-5, 'Invalid acquired task rotation')
    inverse = rotation.T
    return np.r_[raw[:7], -inverse @ raw[32:35], inverse[:, :2].T.reshape(-1),
                 raw[22], 2*TARGETS.index(target)/9-1, -1 if phase == 'FREE' else 1].astype(np.float32)


def fit_stats(states, actions):
    states, actions = np.asarray(states), np.asarray(actions)
    need(states.ndim == 2 and states.shape[1] == 19 and actions.ndim == 3 and actions.shape[2] == 7
         and len(states) == len(actions) and np.isfinite(states).all() and np.isfinite(actions).all(),
         'Invalid TRAIN-only normalization input')
    result = {}
    for key, value, minimum in (('state', states, np.r_[np.full(7, .002), np.full(3, .005), np.full(9, .05)]),
                                ('action', actions.reshape(-1, 7), np.full(7, .002))):
        q01, q99 = np.quantile(value, [.01, .99], axis=0)
        center = (q01+q99)/2; width = np.maximum(q99-q01, minimum)
        low, high = center-width/2, center+width/2
        if key == 'state':
            low[10:16], high[10:16] = -1, 1
            low[16], high[16] = 0, 1
            low[17:19], high[17:19] = -1, 1
        z = 2*(value-low)/(high-low)-1
        hist = [np.bincount(np.rint((np.clip(z[:, j], -1, 1)+1)*127.5).astype(int), minlength=256).tolist()
                for j in range(value.shape[1])]
        result[key] = dict(low=low.tolist(), high=high.tolist(), q01=q01.tolist(), q99=q99.tolist(),
                           span_floor_used=((q99-q01)<minimum).tolist(),
                           potential_clamp_fraction=(np.abs(z)>1).mean(axis=0).tolist(),
                           bucket_histogram_256=hist, actual_clamp=False)
    return result


def scale(value, stats):
    low, high = np.asarray(stats['low']), np.asarray(stats['high'])
    need(np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low), 'Invalid frozen range')
    return (2*(np.asarray(value)-low)/(high-low)-1).astype(np.float32)


def unscale(value, stats):
    low, high = np.asarray(stats['low']), np.asarray(stats['high'])
    return ((np.asarray(value)+1)*.5*(high-low)+low).astype(np.float32)


def _action_and_anchor(actions, measured_q, representation):
    need(representation in (ABSOLUTE, BLOCK_RELATIVE), 'Unknown action representation')
    actions = np.asarray(actions, dtype=np.float32)
    need(actions.ndim in (2, 3) and actions.shape[-1] == 7 and np.isfinite(actions).all(),
         'Finite Hx7 or BxHx7 actions required')
    anchor = np.asarray(measured_q, dtype=np.float32)
    need(anchor.shape == actions.shape[:-2]+(7,) and np.isfinite(anchor).all(),
         'One finite measured-q anchor per block required')
    return actions, anchor[..., None, :]


def encode_actions(actions, measured_q, representation):
    """Subtract the acquired block origin once, never adjacent commands or dropped q."""
    actions, anchor = _action_and_anchor(actions, measured_q, representation)
    return actions.copy() if representation == ABSOLUTE else actions-anchor


def decode_actions(actions, measured_q, representation):
    """Restore absolute commands with the same fixed block origin at every horizon."""
    actions, anchor = _action_and_anchor(actions, measured_q, representation)
    return actions.copy() if representation == ABSOLUTE else actions+anchor


def targets_for_rows(rows, representation):
    actions = np.stack([r['actions'] for r in rows])
    if representation == ABSOLUTE:
        return actions
    return encode_actions(actions, np.stack([r['measured_q'] for r in rows]), representation)


def pack_batch(rows, stats, *, training, rng=None, q_dropout=0., action_representation=ABSOLUTE):
    need(action_representation in (ABSOLUTE, BLOCK_RELATIVE), 'Unknown action representation')
    state = scale(np.stack([r['state'] for r in rows]), stats['state'])
    if training and q_dropout:
        need(rng is not None and 0 <= q_dropout <= 1, 'Explicit dropout RNG required')
        state[rng.random(len(state)) < q_dropout, :7] = 0
    result = {'observation.state': state[:, :7].copy(), 'observation.environment_state': state[:, 7:].copy()}
    if training:
        labels = scale(targets_for_rows(rows, action_representation), stats['action'])
        result.update(action=labels, action_is_pad=np.zeros(labels.shape[:2], dtype=bool))
    return result


def row_from_episode(episode, start, horizon):
    arrays = episode['arrays']
    return dict(state=small_state(arrays['state'][start], str(arrays['targets'][start]), str(arrays['phases'][start])),
                actions=np.array(arrays['actions'][start:start+horizon], np.float32, copy=True),
                measured_q=np.array(arrays['state'][start, :7], copy=True),
                measured_dq=np.array(arrays['state'][start, 7:14], copy=True),
                episode_id=episode['episode_id'], start=start)


def prepare(config, output):
    records = resolve_train_records(config['manifest'], config['manifest_sha256'])
    episodes, receipt_rows = [], []
    for record in records:
        eid = record['episode_id']
        report = pinned_json(record['report_path'], record['qualification_sha256'])
        need(report.get('qualified') is True and report.get('episode_id') == eid
             and report.get('arrays_sha256') == record['sha256'], 'Qualified raw episode differs: ' + eid)
        evidence = report['evidence']; derivation = evidence['joint_action_derivation']; pairs = evidence['label_pairs']
        need(derivation.get('action_contract_version') == 'absolute_joint_command_v1'
             and derivation.get('split') == 'train' and derivation.get('period_s') == .05
             and derivation.get('physics_steps') == 12 and derivation.get('no_clipping') is True,
             'Raw absolute action derivation missing: ' + eid)
        pair_hash = hashlib.sha256((json.dumps(pairs, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()).hexdigest()
        need(pair_hash == derivation['label_pairs_sha256'], 'Temporal pair hash differs: ' + eid)
        arrays, cache = open_episode_mmaps(record['array_path'], record['sha256'], output/'cache'/eid)
        n = len(arrays['times'])
        need(arrays['state'].shape == (n, 41) and arrays['actions'].shape == (n, 7)
             and all(len(value) == n for value in arrays.values()), 'Array shapes differ: ' + eid)
        need(np.isfinite(arrays['state']).all() and np.isfinite(arrays['actions']).all()
             and np.all(arrays['target_sources'] == 'rgbd_surface_tracking'), 'Invalid/non-observed source: ' + eid)
        trace_root = Path(derivation['trace_root']); hashes = derivation['trace_sha256']
        need(trace_root.is_absolute(), 'Absolute bound trace root required')
        try:
            # Keep only small frame fields; never load raw sensor frame NPZs.
            wanted = {p[k] for p in pairs for k in ('frame_index', 'following_frame_index')}
            frames = [{k: row.get(k) for k in ('index', 'physics_row', 'time_s', 'eligible', 'target')}
                      for row in bound_jsonl(trace_root/'sensors/frames.jsonl', hashes['sensors/frames.jsonl'])
                      if row['index'] in wanted]
            valid, boundary = l2_mask(arrays, pairs, frames,
                bound_jsonl(trace_root/'expert_trace.jsonl', hashes['expert_trace.jsonl']))
        except (KeyError, FileNotFoundError, ValueError) as error:
            raise ValueError(f'{eid}: L2 evidence unavailable/unproved: {error}') from error
        starts = window_starts(arrays, pairs, valid,
            derivation['phase_derivation']['cross_boundary_label_indices'], config['horizon'])
        need(len(starts) >= 2, 'At least two proven L2 windows per TRAIN episode required: ' + eid)
        episodes.append(dict(episode_id=eid, arrays=arrays, starts=starts))
        receipt_rows.append(dict(episode_id=eid, split='train', source_record=record, cache=cache,
                                 starts=starts, boundary=boundary))
    rows = [row_from_episode(e, s, config['horizon']) for e in episodes for s in e['starts']]
    stats = fit_stats(np.stack([r['state'] for r in rows]), np.stack([r['actions'] for r in rows]))
    save_json(output/'normalization.json', stats)
    receipt = dict(schema='ausc_m0_act_preparation_v1', episodes=receipt_rows, train_windows=len(rows),
        train_episode_ids=list(TRAIN16), heldout_arrays_opened=False, image_arrays_opened=False,
        horizon=10, execute_steps=5, period_s=.05, action_representation='absolute_joint_radians',
        state_definition='q7 + inverse_registered_task9 + acquired_confidence + target_id + precontact_phase',
        manifest=dict(path=config['manifest'], sha256=config['manifest_sha256']),
        normalization=dict(path=str((output/'normalization.json').resolve()), sha256=sha256(output/'normalization.json')),
        normalization_denominator='each eligible TRAIN L2 origin once; actions each origin/horizon once')
    save_json(output/'prepared.json', receipt)
    return episodes, stats, receipt


def load_prepared(path, config):
    """Only the explicitly generated, TRAIN-only read-only caches are consumed."""
    receipt = pinned_json(path, config.get('prepared_manifest_sha256'))
    need(receipt['schema'] == 'ausc_m0_act_preparation_v1' and receipt['train_episode_ids'] == list(TRAIN16)
         and receipt['heldout_arrays_opened'] is False and receipt['image_arrays_opened'] is False
         and receipt['manifest'] == dict(path=config['manifest'], sha256=config['manifest_sha256'])
         and (receipt['horizon'], receipt['execute_steps'], receipt['period_s']) == (10, 5, .05),
         'Prepared scope/contract differs')
    need(receipt['action_representation'] == ABSOLUTE, 'Source cache must contain original absolute commands')
    stats = pinned_json(receipt['normalization']['path'], receipt['normalization']['sha256'])
    need([e['episode_id'] for e in receipt['episodes']] == list(TRAIN16), 'Prepared TRAIN order differs')
    episodes = []
    for item in receipt['episodes']:
        need(item['split'] == 'train' and item['boundary']['stationary_handoff_proved'] is True,
             'Prepared split/handoff differs')
        arrays = {}
        need(set(item['cache']) == set(MEMBERS), 'Unexpected prepared members')
        for key, binding in item['cache'].items():
            need(sha256(binding['path']) == binding['sha256'], 'Prepared cache changed: ' + key)
            arrays[key] = np.load(binding['path'], mmap_mode='r', allow_pickle=False)
            need(isinstance(arrays[key], np.memmap) and not arrays[key].flags.writeable, 'Prepared mmap required')
        episodes.append(dict(episode_id=item['episode_id'], starts=item['starts'], arrays=arrays))
    if config['action_representation'] == BLOCK_RELATIVE:
        # This receipt remains the immutable absolute source-cache binding. Fit
        # the new action range over every eligible TRAIN origin, not just 32 probes.
        rows = [row_from_episode(e, s, 10) for e in episodes for s in e['starts']]
        need(len(rows) == receipt['train_windows'], 'Relative TRAIN normalization denominator differs')
        relative_stats = fit_stats(np.stack([r['state'] for r in rows]), targets_for_rows(rows, BLOCK_RELATIVE))
        need(relative_stats['state'] == stats['state'], 'Relative control changed frozen state statistics')
        stats = relative_stats
    return episodes, stats, receipt


def save_action_contract(output, config, receipt):
    relative = config['action_representation'] == BLOCK_RELATIVE
    save_json(output/'action_contract.json', dict(
        action_representation=config['action_representation'], stored_source_actions=ABSOLUTE,
        prediction_npz_raw=ABSOLUTE, anchor='acquired measured_q at block origin' if relative else None,
        encoding='a[t+(k+1)*0.05] - measured_q[t]' if relative else 'a[t+(k+1)*0.05]',
        decoding='add the same acquired measured_q[t] once to each horizon' if relative else 'identity',
        action_statistics='recomputed from eligible TRAIN relative blocks' if relative else 'frozen TRAIN absolute blocks',
        normalization=dict(path=str((output/'normalization.json').resolve()),
                           sha256=sha256(output/'normalization.json')),
        normalization_denominator=receipt['normalization_denominator'], train_windows=receipt['train_windows'],
        prepared_json_role='immutable original absolute source-cache receipt',
        source_prepared_manifest=config.get('prepared_manifest'),
        source_prepared_sha256=config.get('prepared_manifest_sha256'),
        source_absolute_normalization=receipt['normalization'], heldout_arrays_opened=False))


def make_native(config):
    """Fresh official ACT, never imported by CPU adapter tests/preparation."""
    need(sys.platform == 'linux' and os.environ.get('AUSC_OWNED_RUN') == '1',
         'Native ACT construction requires the Linux experiment owner')
    vendor = Path(config['lerobot_root']).resolve()
    commit = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    need(commit == UPSTREAM_COMMIT, 'Installed official ACT revision differs')
    source_names = ['src/lerobot/policies/act/'+name for name in
                    ('configuration_act.py', 'modeling_act.py', 'processor_act.py')]
    source_names += ['src/lerobot/configs/types.py', 'src/lerobot/configs/policies.py']
    subprocess.run(['git', '-C', str(vendor), 'diff', '--exit-code', 'HEAD', '--', *source_names],
                   stdout=subprocess.DEVNULL, check=True)
    need(Path(sys.prefix).resolve() == (vendor/'.venv').resolve(), 'Use existing isolated native Python environment')
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    import inspect
    need(Path(inspect.getfile(ACTPolicy)).resolve() == vendor/'src/lerobot/policies/act/modeling_act.py',
         'ACT imported from a different installation')
    torch.manual_seed(config['seed'])
    if config['device'] == 'cuda':
        need(torch.cuda.is_available(), 'Requested CUDA unavailable')
        torch.cuda.manual_seed_all(config['seed'])
    cfg = ACTConfig(input_features={
        'observation.state': PolicyFeature(FeatureType.STATE, (7,)),
        'observation.environment_state': PolicyFeature(FeatureType.ENV, (12,))},
        output_features={'action': PolicyFeature(FeatureType.ACTION, (7,))},
        normalization_mapping={k: NormalizationMode.IDENTITY for k in ('STATE', 'ENV', 'ACTION')},
        n_obs_steps=1, chunk_size=10, n_action_steps=5, temporal_ensemble_coeff=None,
        pretrained_backbone_weights=None, vision_backbone='resnet18', **config['model'],
        optimizer_lr=config['optimizer']['lr'], optimizer_weight_decay=config['optimizer']['weight_decay'],
        device=config['device'], use_amp=False, push_to_hub=False)
    need(cfg.device == config['device'], 'Unexpected automatic device substitution')
    policy = ACTPolicy(cfg).to(config['device'])
    need(not hasattr(policy.model, 'backbone') and not hasattr(policy.model, 'vae_encoder'), 'Unexpected image/VAE model')
    return torch, policy, {name: sha256(vendor/name) for name in source_names}


def error_summary(error):
    error = abs(np.asarray(error, dtype=np.float64))
    need(error.ndim == 3 and error.size and np.isfinite(error).all(), 'Finite full-chunk error required')
    return dict(full_chunk_mae_rad=float(error.mean()), h0_mae_rad=float(error[:, 0].mean()),
                median_absolute_rad=float(np.median(error)),
                median_sample_mae_rad=float(np.median(error.mean(axis=(1, 2)))),
                h0_median_absolute_rad=float(np.median(error[:, 0])),
                h0_median_sample_mae_rad=float(np.median(error[:, 0].mean(axis=1))))


def native_smoke(config, output, episodes, stats):
    """Owner-dispatched CPU forward/backward only; no optimizer update."""
    torch, policy, sources = make_native(config)
    rows = [row_from_episode(episodes[i], episodes[i]['starts'][0], 10) for i in range(2)]
    batch = {k: torch.from_numpy(v) for k, v in pack_batch(rows, stats, training=True,
             action_representation=config['action_representation']).items()}
    loss, _ = policy(batch)
    need(bool(torch.isfinite(loss)), 'Native CPU smoke loss nonfinite')
    loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    need(gradients and all(bool(torch.isfinite(g).all()) for g in gradients), 'Native CPU smoke gradient nonfinite')
    prediction_batch = {k: torch.from_numpy(v) for k, v in pack_batch(rows, stats, training=False).items()}
    with torch.inference_mode():
        prediction = policy.predict_action_chunk(prediction_batch)
    need(tuple(prediction.shape) == (2, 10, 7) and bool(torch.isfinite(prediction).all()), 'Native CPU smoke shape differs')
    save_json(output/'COMPLETE.json', dict(completed=True, mode='native_smoke', device='cpu',
        source_sha256=sources, prediction_shape=list(prediction.shape), gradient_tensors=len(gradients),
        input_keys=sorted(prediction_batch), images=False, vae=False, optimizer_updates=0,
        action_representation=config['action_representation'],
        heldout_arrays_opened=False, m0_passed=False, closed_loop_evaluated=False))


def fit(config, output, episodes, stats, receipt):
    mode = config['mode']; settings = config['modes'][mode]
    representation = config['action_representation']
    torch, policy, sources = make_native(config)
    draw_rng = np.random.default_rng(config['seed'])
    dropout_rng = np.random.default_rng(config['seed']+1)
    probe_rng = np.random.default_rng(config['seed']+2)
    probe = [(i, int(s)) for i, e in enumerate(episodes) for s in probe_rng.choice(e['starts'], 2, replace=False)]
    probe_rows = [row_from_episode(episodes[i], s, 10) for i, s in probe]
    opt = config['optimizer']
    optimizer = torch.optim.AdamW(policy.parameters(), lr=opt['lr'], betas=tuple(opt['betas']), weight_decay=opt['weight_decay'])
    checkpoints = {0, 128, 256, 512} if mode == 'train32' else {0, 250, 500, 1000, 2000}
    save_json(output/'native.json', dict(upstream_commit=UPSTREAM_COMMIT, source_sha256=sources,
        device=config['device'], torch_version=torch.__version__, fresh_initialization=True, resumed=False,
        action_representation=representation))
    save_json(output/'probe_rows.json', [dict(episode_id=episodes[i]['episode_id'], start=s) for i, s in probe])
    def tensors(batch):
        return {k: torch.from_numpy(v).to(config['device']) for k, v in batch.items()}
    def evaluate(step):
        policy.eval()
        with torch.inference_mode():
            prediction = policy.predict_action_chunk(tensors(pack_batch(probe_rows, stats, training=False)))
        model_action = unscale(prediction.detach().cpu().numpy(), stats['action'])
        reference = np.stack([r['actions'] for r in probe_rows])
        q = np.stack([r['measured_q'] for r in probe_rows]); dq = np.stack([r['measured_dq'] for r in probe_rows])
        raw = decode_actions(model_action, q, representation)
        need(raw.shape == reference.shape and np.isfinite(raw).all(), 'Nonfinite/malformed ACT evaluation')
        hold = np.repeat(q[:, None], 10, axis=1)
        cv = q[:, None]+dq[:, None]*np.arange(1, 11)[None, :, None]*.05
        metrics = {}
        for name, candidate in (('act', raw), ('hold', hold), ('constant_velocity', cv)):
            metrics[name] = error_summary(candidate-reference)
        directory = output/f'checkpoint_{step:04d}'; directory.mkdir()
        policy.save_pretrained(directory)
        torch.save(dict(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                        draw_rng=draw_rng.bit_generator.state, dropout_rng=dropout_rng.bit_generator.state,
                        step=step, resumable_by_this_entrypoint=False), directory/'training_state.pt')
        save_json(directory/'metrics.json', dict(step=step, probe_scope='32 TRAIN L2 observations', metrics=metrics,
                                                action_representation=representation, evaluated_representation=ABSOLUTE,
                                                closed_loop_evaluated=False, m0_passed=False))
        extra = dict(model_action_physical=model_action) if representation == BLOCK_RELATIVE else {}
        np.savez(directory/'predictions.npz', raw=raw, reference=reference, measured_q=q, measured_dq=dq, **extra)
        policy.train()
    evaluate(0)
    seen = set(); started = time.monotonic()
    with (output/'loss.jsonl').open('x') as log:
        for step in range(1, settings['steps']+1):
            if mode == 'train32':
                selected = [probe[int(i)] for i in draw_rng.permutation(32)]
            else:
                selected = []
                for _ in range(settings['batch_size']):
                    i = int(draw_rng.integers(len(episodes)))
                    selected.append((i, int(draw_rng.choice(episodes[i]['starts']))))
            seen.update(selected)
            rows = [row_from_episode(episodes[i], s, 10) for i, s in selected]
            batch = tensors(pack_batch(rows, stats, training=True, rng=dropout_rng, q_dropout=settings['q_dropout'],
                                       action_representation=representation))
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy(batch)
            need(bool(torch.isfinite(loss)), 'Nonfinite ACT loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), opt['grad_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            record = dict(step=step, loss=float(loss.detach().cpu()), grad_norm=float(gradient.detach().cpu()),
                          draws=step*settings['batch_size'], unique_windows=len(seen), elapsed_s=time.monotonic()-started)
            log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
            if step in checkpoints:
                evaluate(step)
            if step % 25 == 0:
                save_json(output/'progress.json', record)
    save_json(output/'COMPLETE.json', dict(completed=True, mode=mode, steps=settings['steps'],
        train_episode_ids=list(TRAIN16), train_windows=receipt['train_windows'], draws=settings['steps']*32,
        unique_windows=len(seen), checkpoint_step=settings['steps'], m0_passed=False,
        action_representation=representation,
        closed_loop_evaluated=False, heldout_arrays_opened=False, scope='finite state-only ACT fit diagnostic'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = read_json(args.config); validate_config(config)
    # run.py supplies owner/process-group management; use the shared GPU lock
    # before importing torch or constructing any model, even for a CPU fit.
    if config['mode'] != 'prepare':
        need(sys.platform == 'linux' and os.environ.get('AUSC_OWNED_RUN') == '1',
             'Training requires the Linux experiment owner; no direct/local fitting')
        import fcntl
        lock = (Path(config['base'])/'locks/isaac_gpu.lock').open('a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        lock = None
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'configuration.json', config)
    try:
        if config.get('prepared_manifest'):
            episodes, stats, receipt = load_prepared(config['prepared_manifest'], config)
            save_json(args.output/'normalization.json', stats)
            save_json(args.output/'prepared.json', receipt)
        else:
            episodes, stats, receipt = prepare(config, args.output)
        save_action_contract(args.output, config, receipt)
        if config['mode'] == 'prepare':
            save_json(args.output/'COMPLETE.json', dict(completed=True, mode='prepare',
                train_windows=receipt['train_windows'], heldout_arrays_opened=False,
                training_executed=False, m0_passed=False))
        elif config['mode'] == 'native_smoke':
            native_smoke(config, args.output, episodes, stats)
        else:
            fit(config, args.output, episodes, stats, receipt)
    except Exception as error:
        save_json(args.output/'FAILED.json', dict(error=type(error).__name__, reason=str(error),
            mode=config['mode'], m0_passed=False, completed=False))
        raise
    finally:
        if lock is not None:
            lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
