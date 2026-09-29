"""Tiny real NPZ/JSONL wiring fixtures, not physics or sensor qualification.

All arrays and observations are generated test data. The supplied qualification
is a synthetic external attestation; these tests cannot certify an acquisition.
"""
import copy
import json

import numpy as np
import pytest

from sim.auscultation_pi05_v7 import audit_start as a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(a.canonical(value))


def ledger(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b''.join(a.canonical(row) for row in rows))


class TinyRaw:
    def __init__(self, root, *, legacy=False):
        self.root = root
        self.episode = root/'episode'
        self.episode.mkdir()
        self.family = a.NOMINAL_FAMILY if legacy else 'synthetic_new_start'
        self.identity = 'SYNTHETIC_raw_fixture'
        self.vector = [0.]*7 if legacy else [.12, -.06, .06, .12, -.04, .04, 0.]
        self.physics = []
        for index in range(72):
            subphase = 'initial_home' if index < 12 else 'settle' if index < 24 else 'transfer'
            self.physics.append(dict(time_s=(index+1)/240., q=self.vector.copy(), dq=[.001]*7,
                q_command=self.vector.copy(), subphase=subphase, target='R1',
                phase='FREE' if index < 48 else 'APPROACH', recovery=False,
                preparation_role=('eligible_start_route' if index >= 36 else
                    'start_observation' if index >= 24 else 'test_preparation')))
        self.frames = []
        for index, row_index in enumerate(range(11, len(self.physics), 12)):
            row = self.physics[row_index]
            raw = self.episode/'sensors'/('raw_'+str(index)+'.npz')
            raw.parent.mkdir(parents=True, exist_ok=True)
            np.savez(raw, rgb=np.full((2, 2, 3), index, np.uint8),
                depth_m=np.full((2, 2), .5, np.float32), depth_valid=np.ones((2, 2), bool))
            self.frames.append(dict(index=index, physics_row=row_index, time_s=row['time_s'],
                q=row['q'].copy(), dq=row['dq'].copy(), hall={'synthetic_adc': 100+index},
                eligible=index >= 2, phase=row['phase'], subphase=row['subphase'], target='R1',
                raw_file=raw.name, raw_sha256=a.sha(raw), prompt='SYNTHETIC test only'))
        plan = a.canonical({'synthetic_plan': True})
        geometry = a.canonical({'synthetic_geometry': True})
        plan_hash, geometry_hash = (a.hashlib.sha256(value).hexdigest() for value in (plan, geometry))
        snapshot_root = self.episode/'input_snapshot'
        objects = snapshot_root/'objects'
        objects.mkdir(parents=True)
        (objects/plan_hash).write_bytes(plan)
        (objects/geometry_hash).write_bytes(geometry)
        write(snapshot_root/'manifest.json', {'files':{'R1/plan':{'sha256':plan_hash},
            'source/v7/start_manifest':{'sha256':geometry_hash}}})
        self.protocol = dict(schema='wam_v7_frozen_start_family_protocol_v1',
            start_auditor_sha256=a.sha(a.__file__), actual_start_tolerance_rad=.03,
            families=[{'family_id':self.family, 'requested_q_rad':self.vector}],
            cases={self.family+'::R1':{'source_plan_sha256':plan_hash,
                'start_manifest_sha256':geometry_hash}})
        self.protocol_path = root/'protocol.json'
        write(self.protocol_path, self.protocol)
        self.receipt_path = self.episode/'measured_start_receipt.json'
        first = self.frames[2]
        self.receipt = dict(schema='wam_v7_measured_start_receipt_v1',episode_id=self.identity,
            target='R1', start_family=self.family, experiment_protocol_sha256=a.sha(self.protocol_path),
            source_plan_sha256=plan_hash, start_manifest_sha256=geometry_hash,
            preparation_physics_range=[0,12], settle_physics_range=[12,24],
            actual_start_route={'planning_accepted':True,'motion_complete':True,
                'planning_physics_row':36,'end_physics_row_exclusive':72},
            first_eligible=dict(frame_index=2, physics_row=35, time_s=first['time_s'],
                measured_q_rad=first['q'], measured_dq_rad_s=first['dq'],
                retained_command_q_rad=self.physics[35]['q_command']), synthetic_fixture=True)
        self.outcome = dict(requested_points=['R1'], input_snapshot='input_snapshot/manifest.json',
            input_snapshot_manifest_sha256=a.sha(snapshot_root/'manifest.json'))
        self.report = dict(schema='wam_vla_episode_qualification_v1', qualified=True,
            data_kind='closed_loop_rollout',episode_id=self.identity,synthetic_fixture=True,
            evidence={'label_pairs':[dict(frame_index=2,following_frame_index=3,
                physics_interval_inclusive=[35,47],command_physics_row=47)]})
        self.report_path = root/'synthetic.qualification.json'
        self.legacy = legacy
        self.flush()

    def flush(self):
        ledger(self.episode/'sensors/frames.jsonl', self.frames)
        ledger(self.episode/'expert_trace.jsonl', self.physics)
        if not self.legacy:
            self.receipt['sensor_frames_sha256'] = a.sha(self.episode/'sensors/frames.jsonl')
            write(self.receipt_path, self.receipt)
            self.outcome['measured_start_receipt'] = dict(path=self.receipt_path.name,
                sha256=a.sha(self.receipt_path))
        write(self.episode/'outcome.json', self.outcome)
        self.report['evidence']['artifact_sha256'] = {name:a.sha(self.episode/name) for name in
            ('sensors/frames.jsonl','expert_trace.jsonl','outcome.json')}
        write(self.report_path, self.report)

    def audit(self):
        return a.audit(self.episode, self.report_path, self.protocol_path,
            self.family, self.receipt_path, legacy=self.legacy)


def test_actual_raw_start_receipt_binds_exact_first_observation(tmp_path):
    raw = TinyRaw(tmp_path)
    before = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report = raw.audit()
    assert report['passed'] and report['first_eligible'] == raw.receipt['first_eligible']
    assert report['qualification_sha256'] == a.sha(raw.report_path)
    assert all(p.read_bytes()==value for p,value in before.items())


def test_changed_raw_artifact_rejected(tmp_path):
    raw = TinyRaw(tmp_path)
    (raw.episode/'expert_trace.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='changed'):
        raw.audit()


def test_unnatural_camera_cadence_rejected_with_rebound_hashes(tmp_path):
    raw = TinyRaw(tmp_path)
    raw.frames[2]['physics_row'] = 34
    raw.flush()
    with pytest.raises(ValueError, match='cadence'):
        raw.audit()


def test_measured_start_must_match_same_physics_q_and_dq(tmp_path):
    raw = TinyRaw(tmp_path)
    raw.physics[35]['dq'][3] = .002
    raw.flush()
    with pytest.raises(ValueError, match='state not bound'):
        raw.audit()


def test_receipt_must_match_first_qualified_frame(tmp_path):
    raw = TinyRaw(tmp_path)
    raw.receipt['first_eligible']['frame_index'] = 3
    raw.flush()
    with pytest.raises(ValueError, match='measurement mismatch'):
        raw.audit()


def test_first_pair_must_bind_both_endpoint_rows(tmp_path):
    raw = TinyRaw(tmp_path)
    raw.report['evidence']['label_pairs'][0]['physics_interval_inclusive'][1] = 46
    raw.flush()
    with pytest.raises(ValueError, match='pair|interval|endpoint'):
        raw.audit()


def test_no_home_washout_within_initial_transfer(tmp_path):
    raw = TinyRaw(tmp_path)
    raw.physics[40]['q_command'] = [0.]*7
    raw.flush()
    with pytest.raises(ValueError, match='home'):
        raw.audit()


def test_no_home_washout_after_early_phase_transition_before_route_end(tmp_path):
    raw = TinyRaw(tmp_path)
    raw.physics[65]['q_command'] = [0.]*7
    raw.flush()
    with pytest.raises(ValueError, match='home'):
        raw.audit()


def test_route_receipt_cannot_truncate_raw_role_interval(tmp_path):
    raw = TinyRaw(tmp_path)
    raw.physics[65]['q_command'] = [0.]*7
    raw.receipt['actual_start_route']['end_physics_row_exclusive'] = 60
    raw.flush()
    with pytest.raises(ValueError, match='route|role|home'):
        raw.audit()


def test_start_observation_before_first_camera_frame_also_cannot_return_home(tmp_path):
    raw = TinyRaw(tmp_path)
    raw.physics[30]['q_command'] = [0.]*7
    raw.flush()
    with pytest.raises(ValueError, match='home'):
        raw.audit()


def test_preparation_cannot_enter_any_training_pair(tmp_path):
    raw = TinyRaw(tmp_path)
    raw.physics[40]['subphase'] = 'settle'
    raw.flush()
    with pytest.raises(ValueError, match='Preparation entered'):
        raw.audit()


def test_failed_external_qualification_cannot_be_promoted(tmp_path):
    raw = TinyRaw(tmp_path)
    raw.report['qualified'] = False
    raw.flush()
    with pytest.raises(ValueError, match='qualified'):
        raw.audit()


def test_nominal_receipt_is_explicitly_retrospective_and_never_overwritten(tmp_path):
    raw = TinyRaw(tmp_path, legacy=True)
    report = raw.audit()
    receipt = a.read(raw.receipt_path)
    assert report['legacy_retrospective'] is True
    assert receipt['schema'] == 'wam_v7_legacy_start_replay_receipt_v1'
    assert 'Retrospective' in receipt['provenance']
    with pytest.raises(ValueError, match='New retrospective'):
        raw.audit()


def test_raw_content_fingerprint_ignores_target_prompt_and_file_names(tmp_path):
    raw = TinyRaw(tmp_path)
    before = a.numeric_fingerprint(raw.episode, raw.frames)
    changed = copy.deepcopy(raw.frames)
    for frame in changed:
        old = raw.episode/'sensors'/frame['raw_file']
        frame['target'] = 'retagged_target'
        frame['prompt'] = 'retagged_prompt'
        frame['raw_file'] = 'renamed_'+frame['raw_file']
        (old.parent/frame['raw_file']).write_bytes(old.read_bytes())
    assert a.numeric_fingerprint(raw.episode, changed) == before
    changed[0]['hall']['synthetic_adc'] += 1
    assert a.numeric_fingerprint(raw.episode, changed) != before
