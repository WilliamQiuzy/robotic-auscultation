"""Regressions for deriving joint commands without changing acquired inputs."""
import importlib
import importlib.util
import json

import numpy as np
import pytest


def exporter():
    name = 'sim.auscultation_pi05_v6.export_joint_dataset'
    assert importlib.util.find_spec(name) is not None, 'joint dataset exporter required'
    return importlib.import_module(name)


def fixture_rows():
    physics = []
    for index in range(49):
        phase = 'FREE' if index <= 12 else 'HOLD'
        physics.append(dict(time_s=1 + index / 240, phase=phase, target='R1',
                            subphase='transfer' if phase == 'FREE' else 'hold',
                            recovery=False, q=[.1] * 7, q_command=[.12] * 7))
    frames = []
    for index in range(5):
        phase = 'FREE' if index <= 1 else 'HOLD'
        q = [float(x) for x in np.full(7, .1 + index * .01, np.float32)]
        physics[12 * index]['q'] = q
        frames.append(dict(index=index, physics_row=12 * index, time_s=1 + index * .05,
                           phase=phase, target='R1', eligible=index != 2, q=q,
                           subphase='transfer' if phase=='FREE' else 'hold',
                           state=q+[0.]*7+[.085,0.,0.]+[0.]*15))
    pairs = [dict(frame_index=0, following_frame_index=1,
                  physics_interval_inclusive=[0, 12], command_physics_row=12),
             dict(frame_index=3, following_frame_index=4,
                  physics_interval_inclusive=[36, 48], command_physics_row=48)]
    expected = np.array([np.arange(1, 8) * .001, -np.arange(1, 8) * .0005], np.float32)
    for pair, delta in zip(pairs, expected):
        physics[pair['command_physics_row']]['q_command'] = (
            np.asarray(frames[pair['frame_index']]['q']) + delta).tolist()
    state = np.zeros((2, 32), np.float32)
    state[:] = [frames[p['frame_index']]['state'] for p in pairs]
    arrays = dict(state=state, actions=np.zeros((2, 7), np.float32),
                  times=np.array([1., 1.15]), phases=np.array(['FREE', 'HOLD']),
                  targets=np.array(['R1', 'R1']))
    expected=np.asarray([physics[p['command_physics_row']]['q_command'] for p in pairs],np.float32)
    return arrays, frames, physics, pairs, expected


def test_uses_absolute_next_command_independent_of_servo_lag():
    arrays, frames, physics, pairs, expected = fixture_rows()
    actions = exporter().derive_joint_actions(arrays, frames, physics, pairs)
    np.testing.assert_array_equal(actions, expected)
    assert actions[0, 6] != 0 and actions[1, 6] != 4
    # The next measured state and present command deliberately differ from
    # both the correct origin and label: neither may enter the subtraction.
    assert not np.allclose(actions[0], np.array(physics[12]['q_command']) - frames[1]['q'])
    assert not np.allclose(actions[0], np.array(physics[12]['q_command']) - physics[0]['q_command'])


@pytest.mark.parametrize('mutation,match', [
    ('command_row', 'pair'), ('cross_phase', 'interval'), ('state_origin', 'measured'),
    ('future_time', 'time'), ('out_of_bound', 'bound'), ('missing_pair', 'pair'),
])
def test_rejects_misaligned_or_unsafe_derivation(mutation, match):
    arrays, frames, physics, pairs, _ = fixture_rows()
    if mutation == 'command_row': pairs[0]['command_physics_row'] = 11
    elif mutation == 'cross_phase': physics[6]['phase'] = 'CONTACT'
    elif mutation == 'state_origin': arrays['state'][0, 0] += .001
    elif mutation == 'future_time': frames[1]['time_s'] += .001
    elif mutation == 'out_of_bound': physics[12]['q_command'][0] += 5.
    elif mutation == 'missing_pair': pairs.pop()
    with pytest.raises(ValueError, match=match):
        exporter().derive_joint_actions(arrays, frames, physics, pairs)


def test_rewrites_only_actions_preserving_npy_members(tmp_path):
    import zipfile
    arrays, *_ = fixture_rows()
    arrays.update(rgb=np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3),
                  prompts=np.array(['free', 'hold']),
                  target_sources=np.array(['rgbd_surface_tracking'] * 2))
    source, destination = tmp_path / 'parent.npz', tmp_path / 'child.npz'
    np.savez_compressed(source, **arrays)
    before = source.read_bytes()
    replacement = np.full((2, 7), .003, np.float32)
    hashes = exporter().replace_actions(source, destination, replacement)
    assert source.read_bytes() == before
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination) as new:
        assert old.namelist() == new.namelist()
        for name in old.namelist():
            if name != 'actions.npy':
                assert old.read(name) == new.read(name)
                assert hashes[name]['parent_sha256'] == hashes[name]['derived_sha256']
    with np.load(destination, allow_pickle=False) as derived:
        np.testing.assert_array_equal(derived['actions'], replacement)


def test_rewrites_explicit_phases_and_prompts_preserving_every_other_member(tmp_path):
    import zipfile
    arrays, *_ = fixture_rows()
    arrays.update(rgb=np.zeros((2,2,2,3),np.uint8), prompts=np.array(['free','hold']),
                  target_sources=np.array(['rgbd_surface_tracking']*2))
    source,destination=tmp_path/'parent.npz',tmp_path/'derived.npz'
    np.savez_compressed(source,**arrays)
    phases=np.array(['APPROACH','HOLD'])
    prompts=np.array(['approach','hold'])
    evidence=exporter().replace_actions(source,destination,arrays['actions'],phases=phases,prompts=prompts)
    with zipfile.ZipFile(source) as parent,zipfile.ZipFile(destination) as derived:
        for name in parent.namelist():
            if name not in {'actions.npy','phases.npy','prompts.npy'}:
                assert parent.read(name)==derived.read(name)
                assert evidence[name]['unchanged']
    assert not evidence['phases.npy']['unchanged']
    with np.load(destination) as derived:
        np.testing.assert_array_equal(derived['phases'],phases)
        np.testing.assert_array_equal(derived['prompts'],prompts)


def test_bound_read_rejects_tampered_source(tmp_path):
    source = tmp_path / 'expert_trace.jsonl'
    source.write_text('{}\n')
    with pytest.raises(ValueError, match='hash'):
        exporter().read_bound_rows(source, '0' * 64)


def observed_audit_fixture(tmp_path):
    module=exporter()
    binding=module.observed_supervisor_binding()
    outcome=dict(observed_supervisor_version=binding['version'],observed_supervisor_sha256=binding['sha256'])
    for name in module.OBSERVED_AUDIT_ARTIFACTS:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(outcome) if name=='outcome.json' else '{}\n')
    ledger_hash=module.sha256(tmp_path/'supervisor_decisions.jsonl')
    outcome['observed_supervisor_ledger']=dict(observed_supervisor_version=binding['version'],
        observed_supervisor_sha256=binding['sha256'],ledger_sha256=ledger_hash)
    (tmp_path/'supervisor_decisions_receipt.json').write_text(json.dumps(outcome['observed_supervisor_ledger']))
    (tmp_path/'outcome.json').write_text(json.dumps(outcome))
    audit=dict(schema='wam_observed_supervisor_replay_audit_v1',passed=True,
        source_supervisor_version=binding['version'],source_supervisor_sha256=binding['sha256'],
        audit_source_sha256=module.sha256(module.Path(module.__file__).with_name('audit_observed_supervisor.py')),
        artifact_sha256={name:module.sha256(tmp_path/name) for name in module.OBSERVED_AUDIT_ARTIFACTS})
    path=tmp_path/'observed_supervisor_audit.json';path.write_text(json.dumps(audit))
    qualification=dict(artifact_sha256={'observed_supervisor_audit.json':module.sha256(path)},
                       source_sha256={str(binding['path']):binding['sha256']})
    return outcome,audit,qualification


def test_only_bound_independent_observed_audit_promotes_contact_equivalence(tmp_path):
    module=exporter()
    outcome,audit,evidence=observed_audit_fixture(tmp_path)
    result,bindings=module.verify_contact_supervision(outcome,evidence,tmp_path,module.sha256(tmp_path/'outcome.json'))
    assert result['contact_supervisor_equivalence_verified'] is True
    assert tmp_path/'supervisor_decisions.jsonl' in bindings
    legacy,_=module.verify_contact_supervision({}, {},tmp_path,'a'*64)
    assert legacy['contact_supervisor_equivalence_verified'] is False


@pytest.mark.parametrize('mutation',['missing_audit','unbound_audit','failed','wrong_source','wrong_auditor','wrong_schema','tampered_ledger','missing_artifact','different_outcome'])
def test_observed_export_rejects_missing_or_mismatched_replay_audit(tmp_path,mutation):
    module=exporter()
    outcome,audit,evidence=observed_audit_fixture(tmp_path)
    path=tmp_path/'observed_supervisor_audit.json'
    outcome_hash=module.sha256(tmp_path/'outcome.json')
    if mutation=='missing_audit':path.unlink()
    elif mutation=='unbound_audit':evidence['artifact_sha256'].clear()
    elif mutation=='failed':audit['passed']=False
    elif mutation=='wrong_source':audit['source_supervisor_sha256']='a'*64
    elif mutation=='wrong_auditor':audit['audit_source_sha256']='a'*64
    elif mutation=='wrong_schema':audit['schema']='unverified_claim'
    elif mutation=='tampered_ledger':(tmp_path/'supervisor_decisions.jsonl').write_text('changed')
    elif mutation=='missing_artifact':audit['artifact_sha256'].pop('sensors/frames.jsonl')
    elif mutation=='different_outcome':outcome_hash='a'*64
    if mutation not in {'missing_audit','unbound_audit'}:
        path.write_text(json.dumps(audit));evidence['artifact_sha256'][path.name]=module.sha256(path)
    with pytest.raises(ValueError):module.verify_contact_supervision(outcome,evidence,tmp_path,outcome_hash)


def test_export_preserves_parent_protocol_and_split_provenance(tmp_path):
    module = exporter()
    parent, raw, output = (tmp_path / name for name in ('parent', 'raw', 'derived'))
    parent.mkdir()
    raw.mkdir()
    urdf = tmp_path / 'arm.urdf'
    urdf.write_text('<robot>' + ''.join(
        f'<joint name="joint{i}" type="revolute"><parent link="openarm_left_link{i-1}"/>'
        f'<child link="openarm_left_link{i}"/></joint>' for i in range(1, 8)) + '</robot>')
    protocol = parent / 'protocol.json'
    protocol.write_text('{"units":"parent pose meters and radians plus force"}\n')
    modalities = dict(rgb='camera', depth='camera_metric_optical_z', hall='adc_voltage',
                      robot='measured_q_dq_fk', target='rgbd_surface_tracking', audio='absent')
    records = []
    for episode_id, split in [('A', 'train'), ('B', 'validation'), ('C', 'test')]:
        arrays, frames, physics, pairs, _ = fixture_rows()
        arrays.update(rgb=np.full((2, 2, 2, 3), ord(episode_id), np.uint8),
                      prompts=np.array(['free', 'hold']),
                      target_sources=np.array(['rgbd_surface_tracking'] * 2))
        episode = raw / ('expert_rgbd_' + episode_id)
        (episode / 'sensors').mkdir(parents=True)
        for name, rows in [('expert_trace.jsonl', physics), ('sensors/frames.jsonl', frames)]:
            (episode / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        npz = parent / (episode_id + '.npz')
        np.savez_compressed(npz, **arrays)
        outcome = parent / (episode_id + '.outcome.json')
        outcome.write_text('{"success":true}\n')
        qualification = parent / (episode_id + '.qualification.json')
        report = dict(schema='wam_vla_episode_qualification_v1', episode_id=episode_id,
                      qualified=True, data_kind='closed_loop_rollout', arrays_sha256=module.sha256(npz),
                      checks={key: True for key in module.CHECKS}, modalities=modalities,
                      action_contract_version='parent_pose_contract', action_contract_sha256='a' * 64,
                      label_diagnostics={'old_pose_labels': True},
                      evidence=dict(label_pairs=pairs, artifact_sha256={
                          name: module.sha256(episode / name)
                          for name in ('expert_trace.jsonl', 'sensors/frames.jsonl')},
                          source_sha256={str(urdf): module.sha256(urdf)}))
        qualification.write_text(json.dumps(report))
        records.append(dict(episode_id=episode_id, split=split, path=npz.name, sha256=module.sha256(npz),
                            qualification_path=qualification.name, qualification_sha256=module.sha256(qualification),
                            outcome_path=outcome.name, outcome_sha256=module.sha256(outcome)))
    manifest_path = parent / 'manifest.json'
    manifest_path.write_text(json.dumps(dict(schema='wam_vla_training_manifest_v1', sealed=True,
        horizon=10, action_period_s=.05, modalities=modalities, episodes=records,
        protocol_sha256=module.sha256(protocol))))
    originals = {path: path.read_bytes() for path in parent.iterdir()}
    receipt = module.export_dataset(manifest_path, raw, output, urdf)
    assert receipt['episode_count'] == 3 and receipt['rows'] == 6
    assert receipt['split_row_counts'] == {'train': 2, 'validation': 2, 'test': 2}
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    assert (output / 'parent/protocol.json').read_bytes() == protocol.read_bytes()
    child_protocol = json.loads((output / 'protocol.json').read_text())
    assert child_protocol['units'] == 'radians'
    assert child_protocol['joint_names'] == ['joint' + str(i) for i in range(1, 8)]
    assert child_protocol['parent_protocol_sha256'] == module.sha256(protocol)
    child_manifest = json.loads((output / 'manifest.json').read_text())
    from sim.auscultation_pi05_v6 import phase_contract
    assert child_manifest['phase_contract_version']==phase_contract.VERSION
    assert child_manifest['phase_contract_sha256']==module.sha256(phase_contract.__file__)
    assert [r['split'] for r in child_manifest['episodes']] == ['train', 'validation', 'test']
    for record in records:
        name = record['qualification_path']
        assert (output / 'parent' / name).read_bytes() == (parent / name).read_bytes()
        report = json.loads((output / name).read_text())
        assert report['action_contract_version'] == 'absolute_joint_command_v1'
        assert report['checks']['actions_bounded'] is True
        assert report['evidence']['parent_label_diagnostics'] == {'old_pose_labels': True}
        derivation=report['evidence']['joint_action_derivation']
        assert derivation['phase_derivation']['changed_label_count']==1
        assert derivation['phase_derivation']['contact_supervisor_equivalence_verified'] is False
        assert report['evidence']['label_pairs']==pairs
        with np.load(output/record['path']) as derived:
            assert derived['phases'].tolist()==['APPROACH','HOLD']
            assert derived['prompts'].tolist()==[module.multimodal.INSTRUCTIONS[p] for p in ['APPROACH','HOLD']]
    with pytest.raises(ValueError, match='new directory'):
        module.export_dataset(manifest_path, raw, output, urdf)
    with (raw / 'expert_rgbd_B/expert_trace.jsonl').open('a') as stream:
        stream.write('{}\n')
    failed_output = tmp_path / 'tampered-output'
    with pytest.raises(ValueError, match='hash'):
        module.export_dataset(manifest_path, raw, failed_output, urdf)
    assert not failed_output.exists()
