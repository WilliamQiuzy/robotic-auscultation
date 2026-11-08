"""Derive joint-command labels from immutable, previously qualified rollouts.

This does not recollect sensors or certify a learned rollout. The parent
qualification remains verbatim evidence of acquisition; a distinct derivation
record binds new actions and observed pre-contact phases to its exact selected
frames and executed commands. Other acquired arrays remain byte-for-byte intact.
"""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from . import action_contract, phase_contract
from sim.auscultation_vla_v2 import multimodal


ARRAY_KEYS = {'state', 'actions', 'rgb', 'times', 'phases', 'targets', 'prompts', 'target_sources'}
CHECKS = ('rgbd_tracking_valid', 'measured_state_valid', 'hall_valid', 'synchronized',
          'actions_bounded', 'episode_complete', 'no_oracle_target', 'v1_audits_valid',
          'source_binding_valid', 'camera_valid')
EXCLUDED = {'settle', 'initial_home', 'return_bridge', 'return_home', 'home_settle'}
FREE_SUBPHASES = {'transfer', 'inter_point_transfer', 'align', 'settle', 'initial_home'}
OBSERVED_AUDIT_ARTIFACTS = {
    'outcome.json','expert_trace.jsonl','sensors/frames.jsonl','sensors/body_poses.jsonl',
    'supervisor_decisions.jsonl','supervisor_decisions_receipt.json',
    'task_frame_registration.json','input_snapshot/manifest.json',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def observed_supervisor_binding():
    """Read its literal version/hash without importing simulator dependencies."""
    path=Path(__file__).with_name('observed_supervisor.py')
    tree=ast.parse(path.read_text())
    versions=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
              and any(isinstance(target,ast.Name) and target.id=='VERSION' for target in node.targets)]
    if len(versions)!=1 or not isinstance(versions[0],str):
        raise ValueError('Observed supervisor source has no unique literal version')
    return dict(version=versions[0],sha256=sha256(path),path=path)


def verify_contact_supervision(outcome,evidence,raw_root,outcome_sha256):
    """Only promote exact source-bound independently audited new observations.

    Historical acquisitions remain explicitly unverified. A producer claiming
    the new supervisor must supply the complete bound audit or export fails.
    """
    raw_root=Path(raw_root)
    legacy=dict(contact_supervisor_equivalence_verified=False,
                contact_phase_source='privileged acquired supervisor labels; no verified ADC replay')
    if not any(key in outcome for key in ('observed_supervisor_version','observed_supervisor_sha256','observed_supervisor_ledger')):
        return legacy,{}
    binding=observed_supervisor_binding()
    if (outcome.get('observed_supervisor_version')!=binding['version'] or
            outcome.get('observed_supervisor_sha256')!=binding['sha256']):
        raise ValueError('Observed supervisor outcome/source mismatch')
    recorded_sources={value for key,value in evidence.get('source_sha256',{}).items()
                      if Path(key).name=='observed_supervisor.py'}
    if recorded_sources!={binding['sha256']}:
        raise ValueError('Qualified observed supervisor source mismatch')
    name='observed_supervisor_audit.json'
    digest=evidence.get('artifact_sha256',{}).get(name)
    if not isinstance(digest,str):raise ValueError('Independent observed supervisor audit is not qualification-bound')
    path=bound_path(raw_root,name,digest)
    audit=read_json(path)
    if (audit.get('schema')!='wam_observed_supervisor_replay_audit_v1' or audit.get('passed') is not True or
            audit.get('source_supervisor_version')!=binding['version'] or
            audit.get('source_supervisor_sha256')!=binding['sha256'] or
            audit.get('audit_source_sha256')!=sha256(Path(__file__).with_name('audit_observed_supervisor.py'))):
        raise ValueError('Independent observed supervisor audit failed or source differs')
    artifacts=audit.get('artifact_sha256',{})
    if not isinstance(artifacts,dict) or not OBSERVED_AUDIT_ARTIFACTS<=artifacts.keys():
        raise ValueError('Independent observed supervisor audit misses required artifacts')
    if artifacts['outcome.json']!=outcome_sha256:
        raise ValueError('Observed audit belongs to a different qualified outcome')
    for key,value in artifacts.items():
        if key in evidence.get('artifact_sha256',{}) and evidence['artifact_sha256'][key]!=value:
            raise ValueError('Observed audit and qualified artifact hashes differ')
    bindings={bound_path(raw_root,key,value):value for key,value in artifacts.items()}
    receipt=read_json(raw_root/'supervisor_decisions_receipt.json')
    if (receipt!=outcome.get('observed_supervisor_ledger') or
            receipt.get('observed_supervisor_sha256')!=binding['sha256'] or
            receipt.get('observed_supervisor_version')!=binding['version'] or
            receipt.get('ledger_sha256')!=artifacts['supervisor_decisions.jsonl']):
        raise ValueError('Observed supervisor ledger receipt/outcome mismatch')
    auditor_path=Path(__file__).with_name('audit_observed_supervisor.py')
    bindings.update({path:digest,binding['path']:binding['sha256'],auditor_path:audit['audit_source_sha256']})
    return dict(contact_supervisor_equivalence_verified=True,
                contact_phase_source='shared ADC/FK/RGB-D supervisor with independently replayed actual decision ledger',
                observed_supervisor_version=binding['version'],observed_supervisor_sha256=binding['sha256'],
                independent_supervisor_audit=dict(path=name,sha256=digest,
                    schema=audit['schema'],audit_source_sha256=audit['audit_source_sha256'],
                    artifact_sha256=artifacts)),bindings


def bound_path(root, relative, digest):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Expected relative bound path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
        raise ValueError('Source path/hash mismatch: ' + relative)
    return path


def read_bound_rows(path, expected_sha256, keys=None):
    """Hash the exact bytes being parsed; optionally retain only needed fields."""
    digest, rows = hashlib.sha256(), []
    with Path(path).open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            rows.append(row if keys is None else {key: row.get(key) for key in keys})
    if digest.hexdigest() != expected_sha256:
        raise ValueError('Source trace hash mismatch: ' + str(path))
    return rows


def frame_contract(row):
    subphase = row.get('subphase')
    phase = ('FREE' if subphase in FREE_SUBPHASES else 'RETRACT'
             if isinstance(subphase, str) and subphase.startswith('normal_') else row.get('phase'))
    eligible = subphase not in EXCLUDED and not row.get('recovery')
    return (phase, eligible) if phase in action_contract.PHASES else ('FREE', False)


def eligible_pair(current, following, physics):
    start, end = current['physics_row'], following['physics_row']
    if not current.get('eligible') or not following.get('eligible'):
        return False
    if (current['phase'] != following['phase'] or current['target'] != following['target']
            or not np.isclose(following['time_s'] - current['time_s'], .05, atol=1e-7, rtol=0)
            or end - start != 12):
        return False
    interval = physics[start:end + 1]
    return (len(interval) == 13 and len({row['phase'] for row in interval}) == 1
            and all(row['target'] == current['target'] and row['recovery'] is False
                    and frame_contract(row) == (current['phase'], True) for row in interval))


def derive_joint_actions(arrays, frames, physics, pairs):
    """Use the original ordered label pairs; never regenerate a new selection."""
    n = len(arrays['state'])
    if len(pairs) != n or not n or [row['index'] for row in frames] != list(range(len(frames))):
        raise ValueError('Invalid label pair count or frame indices')
    if arrays['state'].shape != (n, 32) or not np.isfinite(arrays['state']).all():
        raise ValueError('Invalid measured state array')
    result = []
    for index, pair in enumerate(pairs):
        first, last = pair['frame_index'], pair['following_frame_index']
        if (type(first) is not int or type(last) is not int or first < 0
                or last != first + 1 or last >= len(frames)):
            raise ValueError('Invalid adjacent frame pair')
        current, following = frames[first], frames[last]
        start, end = current['physics_row'], following['physics_row']
        if (type(start) is not int or type(end) is not int or start < 0 or end >= len(physics)
                or pair['physics_interval_inclusive'] != [start, end]
                or pair['command_physics_row'] != end or end - start != 12):
            raise ValueError('Invalid command/physics pair mapping')
        if not np.isclose(following['time_s'] - current['time_s'], .05, atol=1e-7, rtol=0):
            raise ValueError('Invalid pair time interval')
        if not eligible_pair(current, following, physics):
            raise ValueError('Label interval crosses phase, target, recovery or eligibility')
        measured = np.asarray(current['q'], dtype=np.float64)
        command = np.asarray(physics[end]['q_command'], dtype=np.float64)
        if (measured.shape != (7,) or command.shape != (7,) or not np.isfinite([measured, command]).all()
                or not np.array_equal(measured, np.asarray(physics[start]['q'], dtype=np.float64))
                or not np.array_equal(measured.astype(np.float32), arrays['state'][index, :7])):
            raise ValueError('Current measured q is not bound to frame, physics and frozen state')
        if (arrays['phases'][index] != current['phase'] or arrays['targets'][index] != current['target']
                or not np.isclose(arrays['times'][index], current['time_s'], atol=1e-8, rtol=0)
                or not np.isclose(physics[start]['time_s'], current['time_s'], atol=1e-8, rtol=0)
                or not np.isclose(physics[end]['time_s'], following['time_s'], atol=1e-8, rtol=0)):
            raise ValueError('Frozen phase/target/time pairing mismatch')
        action = command.copy()
        if not action_contract.bounded(action[None], current['phase']):
            raise ValueError('Raw absolute joint target exceeds URDF bound; no clipping')
        result.append(action)
    reproduced = []
    for current, following in zip(frames[:-1], frames[1:]):
        if eligible_pair(current, following, physics):
            reproduced.append(dict(frame_index=current['index'], following_frame_index=following['index'],
                                   physics_interval_inclusive=[current['physics_row'], following['physics_row']],
                                   command_physics_row=following['physics_row']))
    if reproduced != pairs:
        raise ValueError('Stored label pairs differ from exact qualified selection')
    actions = np.asarray(result, dtype=np.float32)
    for phase in set(arrays['phases']):
        if not action_contract.bounded(actions[arrays['phases'] == phase], phase):
            raise ValueError('Serialized absolute joint target exceeds URDF bound')
    return actions


def replace_actions(source, destination, actions, *, phases=None, prompts=None):
    """Only explicit action/phase/prompt derivations change; retain other bytes."""
    replacements = {'actions.npy': actions}
    if phases is not None: replacements['phases.npy'] = phases
    if prompts is not None: replacements['prompts.npy'] = prompts
    payloads = {}
    for name, value in replacements.items():
        payload = io.BytesIO()
        np.save(payload, value, allow_pickle=False)
        payloads[name] = payload.getvalue()
    evidence = {}
    with zipfile.ZipFile(source) as parent, zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as child:
        expected = {key + '.npy' for key in ARRAY_KEYS}
        if set(parent.namelist()) != expected or len(parent.namelist()) != len(expected):
            raise ValueError('Parent NPZ array contract mismatch')
        for name in parent.namelist():
            before = parent.read(name)
            after = payloads.get(name, before)
            child.writestr(name, after)
            evidence[name] = dict(parent_sha256=hashlib.sha256(before).hexdigest(),
                                  derived_sha256=hashlib.sha256(after).hexdigest(), unchanged=before == after,
                                  explicitly_derived=name in replacements)
    with zipfile.ZipFile(destination) as child:
        for name, binding in evidence.items():
            if hashlib.sha256(child.read(name)).hexdigest() != binding['derived_sha256']:
                raise ValueError('Derived NPZ member verification failed')
    return evidence


def joint_names(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = {joint.find('child').get('link'): joint for joint in root.findall('joint')}
    current, reverse = 'openarm_left_link7', []
    while current != 'openarm_left_link0':
        if current not in joints or len(reverse) >= 7:
            raise ValueError('Invalid seven-joint URDF chain')
        joint = joints[current]
        if joint.get('type') != 'revolute':
            raise ValueError('Expected seven revolute joints')
        reverse.append(joint.get('name'))
        current = joint.find('parent').get('link')
    if len(reverse) != 7 or len(set(reverse)) != 7:
        raise ValueError('Invalid joint order')
    return reverse[::-1]


def phase_diagnostics(actions, phases):
    return {phase: dict(label_count=int(np.sum(phases == phase)),
                       max_absolute_per_joint_rad=np.max(np.abs(actions[phases == phase]), axis=0).tolist(),
                       max_absolute_rad=float(np.max(np.abs(actions[phases == phase]))))
            for phase in sorted(set(phases))}


def export_dataset(parent_manifest, trace_root, output, urdf_path):
    if action_contract.VERSION != 'absolute_joint_command_v1':
        raise ValueError('Joint action contract is not ready')
    parent_manifest, trace_root, output, urdf_path = map(Path, (parent_manifest, trace_root, output, urdf_path))
    parent_manifest, output = parent_manifest.resolve(), output.resolve()
    parent = parent_manifest.parent
    if output.exists() or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('Derived output must be a new directory outside the parent dataset')
    manifest = read_json(parent_manifest)
    if (manifest.get('schema') != 'wam_vla_training_manifest_v1' or manifest.get('sealed') is not True
            or manifest.get('horizon') != 10 or manifest.get('action_period_s') != .05):
        raise ValueError('Expected sealed 20 Hz ten-step parent manifest')
    records = manifest['episodes']
    ids = [record['episode_id'] for record in records]
    if len(ids) != len(set(ids)) or any(Path(name).name != name for name in ids):
        raise ValueError('Invalid or duplicate episode identity')
    parent_hash = sha256(parent_manifest)
    protocol = bound_path(parent, 'protocol.json', manifest['protocol_sha256'])
    contract_hash = sha256(Path(action_contract.__file__))
    phase_hash = sha256(Path(phase_contract.__file__))
    instructions_hash = sha256(Path(multimodal.__file__))
    exporter_hash = sha256(Path(__file__))
    urdf_hash = sha256(urdf_path)
    names = joint_names(urdf_path)
    all_bindings = {parent_manifest: parent_hash, protocol: manifest['protocol_sha256'],
                    Path(action_contract.__file__): contract_hash, Path(phase_contract.__file__): phase_hash,
                    Path(multimodal.__file__): instructions_hash,
                    Path(__file__): exporter_hash, urdf_path: urdf_hash}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + output.name + '-derivation-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        stage.mkdir()
        provenance = stage / 'parent'
        provenance.mkdir()
        shutil.copyfile(parent_manifest, provenance / 'manifest.json')
        shutil.copyfile(protocol, provenance / 'protocol.json')
        if (parent / 'seal_receipt.json').is_file():
            all_bindings[parent / 'seal_receipt.json'] = sha256(parent / 'seal_receipt.json')
            shutil.copyfile(parent / 'seal_receipt.json', provenance / 'seal_receipt.json')
        (stage / 'derivation_sources').mkdir()
        for source, name in [(Path(__file__), 'export_joint_dataset.py'),
                             (Path(phase_contract.__file__), 'phase_contract.py'),
                             (Path(multimodal.__file__), 'multimodal.py'),
                             (Path(action_contract.__file__), 'action_contract.py'), (urdf_path, urdf_path.name)]:
            shutil.copyfile(source, stage / 'derivation_sources' / name)
        derived_records, summaries = [], []
        for record in records:
            episode_id = record['episode_id']
            source = bound_path(parent, record['path'], record['sha256'])
            qualification_path = bound_path(parent, record['qualification_path'], record['qualification_sha256'])
            outcome_path = bound_path(parent, record['outcome_path'], record['outcome_sha256'])
            all_bindings.update({source: record['sha256'], qualification_path: record['qualification_sha256'],
                                 outcome_path: record['outcome_sha256']})
            report = read_json(qualification_path)
            if (report.get('schema') != 'wam_vla_episode_qualification_v1' or report.get('episode_id') != episode_id
                    or report.get('qualified') is not True or report.get('data_kind') != 'closed_loop_rollout'
                    or report.get('arrays_sha256') != record['sha256']
                    or any(report.get('checks', {}).get(key) is not True for key in CHECKS)
                    or report.get('modalities') != manifest.get('modalities')):
                raise ValueError('Qualified parent acquisition required: ' + episode_id)
            evidence = report['evidence']
            source_urdf_hashes = [value for key, value in evidence['source_sha256'].items()
                                  if Path(key).name == urdf_path.name]
            if source_urdf_hashes != [urdf_hash]:
                raise ValueError('Joint order URDF differs from qualified source')
            raw_root = trace_root / ('expert_rgbd_' + episode_id)
            trace_hashes = {name: evidence['artifact_sha256'][name]
                            for name in ('expert_trace.jsonl', 'sensors/frames.jsonl')}
            frames = read_bound_rows(raw_root / 'sensors/frames.jsonl', trace_hashes['sensors/frames.jsonl'])
            physics = read_bound_rows(raw_root / 'expert_trace.jsonl', trace_hashes['expert_trace.jsonl'],
                                      keys=('time_s', 'phase', 'target', 'recovery', 'subphase', 'q', 'q_command'))
            all_bindings.update({raw_root / name: value for name, value in trace_hashes.items()})
            with np.load(source, allow_pickle=False) as archive:
                if set(archive.files) != ARRAY_KEYS:
                    raise ValueError('Parent array names differ from contract')
                arrays = {key: archive[key] for key in ('state', 'actions', 'times', 'phases', 'targets')}
            pairs = evidence['label_pairs']
            actions = derive_joint_actions(arrays, frames, physics, pairs)
            phases, phase_derivation = phase_contract.derive_phases(arrays, frames, pairs)
            phase_derivation['phase_contract_sha256'] = phase_hash
            phase_derivation['canonical_phase_instructions'] = dict(multimodal.INSTRUCTIONS)
            phase_derivation['instruction_source_sha256'] = instructions_hash
            contact_evidence,contact_bindings=verify_contact_supervision(
                read_json(outcome_path),evidence,raw_root,record['outcome_sha256'])
            phase_derivation.update(contact_evidence)
            all_bindings.update(contact_bindings)
            if contact_evidence['contact_supervisor_equivalence_verified']:
                audit_name=episode_id+'.observed_supervisor_audit.json'
                shutil.copyfile(raw_root/'observed_supervisor_audit.json',provenance/audit_name)
                phase_derivation['independent_supervisor_audit']['path']='parent/'+audit_name
                supervisor_copy=stage/'derivation_sources/observed_supervisor.py'
                if not supervisor_copy.exists():
                    shutil.copyfile(Path(__file__).with_name('observed_supervisor.py'),supervisor_copy)
                    shutil.copyfile(Path(__file__).with_name('audit_observed_supervisor.py'),
                                    stage/'derivation_sources/audit_observed_supervisor.py')
            prompts = np.asarray([multimodal.INSTRUCTIONS[phase] for phase in phases])
            array_binding = replace_actions(source, stage / (episode_id + '.npz'), actions,
                                           phases=phases, prompts=prompts)
            shutil.copyfile(qualification_path, provenance / qualification_path.name)
            shutil.copyfile(outcome_path, stage / outcome_path.name)
            derived_hash = sha256(stage / (episode_id + '.npz'))
            derivation = dict(schema='wam_joint_action_derivation_v1',
                parent_manifest=dict(path='parent/manifest.json', sha256=parent_hash),
                parent_episode=dict(path=str(source), sha256=record['sha256']),
                parent_qualification=dict(path='parent/' + qualification_path.name, sha256=record['qualification_sha256']),
                trace_root=str(raw_root.resolve()), trace_sha256=trace_hashes,
                label_pairs_sha256=hashlib.sha256(json_bytes(pairs)).hexdigest(),
                label_pair_digest_encoding='sorted compact JSON plus newline, UTF-8',
                exact_parent_pair_selection=True, exact_float32_measured_q_binding=True,
                action_formula='expert_trace[label_pair.command_physics_row].q_command',
                period_s=.05, physics_steps=12, origin='fixed URDF joint-zero convention', command='executed command at following frame',
                joint_names=names, units='radians', action_contract_version=action_contract.VERSION,
                action_contract_sha256=contract_hash, exporter_sha256=exporter_hash, urdf_sha256=urdf_hash,
                phase_contract_version=phase_contract.VERSION, phase_contract_sha256=phase_hash,
                phase_derivation=phase_derivation,
                array_members=array_binding, split=record['split'], modalities=manifest['modalities'],
                no_clipping=True, no_sensor_reconstruction=True,
                raw_sensor_provenance='inherited verbatim from parent qualification; archived raw files unchanged',
                acquired_prompts='original prompt member hash and parent episode retained; derived prompts match canonical phase instructions; model adds target identifier',
                scope='absolute commands and observed pre-contact phases derived from qualified acquisition; contact-mode verification is explicitly declared in phase_derivation; no learned-policy acceptance')
            (stage / (episode_id + '.derivation.json')).write_bytes(json_bytes(derivation))
            child_report = copy.deepcopy(report)
            child_report.update(arrays_sha256=derived_hash, action_contract_version=action_contract.VERSION,
                                action_contract_sha256=contract_hash,
                                phase_contract_version=phase_contract.VERSION, phase_contract_sha256=phase_hash,
                                scope=derivation['scope'], label_diagnostics=dict(label_count=len(actions),
                                actions_bounded=True, phase_maxima=phase_diagnostics(actions, phases)))
            child_report['checks']['joint_derivation_valid'] = True
            child_report['checks']['precontact_phase_derivation_valid'] = True
            child_report['checks']['observed_contact_supervision_verified'] = contact_evidence['contact_supervisor_equivalence_verified']
            child_report['evidence']['joint_action_derivation'] = derivation
            child_report['evidence']['qualification_origin'] = 'sensor/mechanics checks inherited from bound parent; joint labels independently derived'
            child_report['evidence']['parent_label_diagnostics'] = report.get('label_diagnostics')
            child_report['checks']['actions_bounded'] = True
            child_qualification = stage / (episode_id + '.qualification.json')
            child_qualification.write_bytes(json_bytes(child_report))
            child_record = dict(record, path=episode_id + '.npz', sha256=derived_hash,
                                qualification_path=child_qualification.name, qualification_sha256=sha256(child_qualification),
                                derivation_path=episode_id + '.derivation.json',
                                derivation_sha256=sha256(stage / (episode_id + '.derivation.json')))
            derived_records.append(child_record)
            summaries.append(dict(episode_id=episode_id, split=record['split'], rows=len(actions),
                                  sha256=derived_hash, phase_maxima=phase_diagnostics(actions, phases),
                                  changed_phase_labels=phase_derivation['changed_label_count']))
        protocol_value = dict(schema='wam_joint_action_derivation_protocol_v1', parent_protocol_sha256=manifest['protocol_sha256'],
                              parent_manifest_sha256=parent_hash, action_contract_version=action_contract.VERSION,
                              action_contract_sha256=contract_hash, exporter_sha256=exporter_hash,
                              phase_contract_version=phase_contract.VERSION, phase_contract_sha256=phase_hash,
                              joint_names=names, units='radians', preserved_split_counts=dict(Counter(r['split'] for r in records)),
                              proposal_limits={phase: action_contract.limits(phase) for phase in sorted(action_contract.PHASES)},
                              acceptance='No learned-policy acceptance inferred from successful label export')
        (stage / 'protocol.json').write_bytes(json_bytes(protocol_value))
        child_manifest = dict(manifest, episodes=derived_records, protocol_sha256=sha256(stage / 'protocol.json'),
                              parent_manifest_sha256=parent_hash, action_contract_version=action_contract.VERSION,
                              phase_contract_version=phase_contract.VERSION, phase_contract_sha256=phase_hash,
                              scope='Same acquired modalities, pairs and splits; derived absolute commands and pre-contact phases; per-episode audit declares observed contact-mode verification; no new generalization evidence')
        (stage / 'manifest.json').write_bytes(json_bytes(child_manifest))
        receipt = dict(schema='wam_joint_action_derivation_receipt_v1', manifest_sha256=sha256(stage / 'manifest.json'),
                       parent_manifest_sha256=parent_hash, episode_count=len(summaries),
                       rows=sum(item['rows'] for item in summaries), episodes=summaries,
                       split_row_counts={split: sum(item['rows'] for item in summaries if item['split'] == split)
                                         for split in sorted({item['split'] for item in summaries})},
                       parent_sources_unchanged=True, source_binding_count=len(all_bindings))
        (stage / 'seal_receipt.json').write_bytes(json_bytes(receipt))
        for path, expected in all_bindings.items():
            if sha256(path) != expected:
                raise ValueError('Source changed during export: ' + str(path))
        if output.exists():
            raise ValueError('Output appeared during export; refusing replacement')
        os.rename(stage, output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-manifest', required=True)
    parser.add_argument('--trace-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--urdf', required=True)
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.parent_manifest, args.trace_root, args.output, args.urdf), sort_keys=True))


if __name__ == '__main__':
    main()
