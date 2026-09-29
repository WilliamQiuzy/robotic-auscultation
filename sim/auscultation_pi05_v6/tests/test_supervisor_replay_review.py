"""Independent review regressions for ledger/label/history consistency.

The existing synthetic physical trajectory is reused as an input generator;
these mutations exercise omissions not covered by its original replay tests.
No acquisition source is modified by this review.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sim.auscultation_pi05_v6.audit_observed_supervisor import replay
from sim.auscultation_pi05_v6.tests.test_supervisor_replay import fixture


def acquisition_fixture():
    data=fixture()
    decisions={row['physics_row']:row for row in data['records']}
    for frame in data['frames']:
        index=frame['physics_row']
        if index in decisions:
            row=decisions[index]
            phase=row['post_state']['phase']
            frame.update(phase='RETRACT' if phase=='DONE' else phase,eligible=True,
                         hall=copy.deepcopy(row['raw_hall_packet']))
        else:
            frame.update(phase='APPROACH',eligible=False,hall=dict(raw=4000,
                sample_time_s=frame['time_s'],received_time_s=frame['time_s'],
                calibration_sha256='a'*64,source='simulated_adc'))
    return data


def test_recorded_camera_phase_cannot_disagree_with_observed_hold():
    data=acquisition_fixture()
    frame=next(frame for frame in data['frames'] if frame['phase']=='HOLD')
    frame['phase']='RETRACT'
    with pytest.raises(ValueError):replay(**data)


def test_legitimate_repeated_zero_time_handoff_has_no_hold_credit():
    data=acquisition_fixture()
    entry=copy.deepcopy(data['records'][0])
    entry['prior_state']=copy.deepcopy(entry['post_state'])
    data['records'].insert(1,entry)
    for index,row in enumerate(data['records']):row['index']=index
    result=replay(**data)
    assert result['max_continuous_hold_s']=={'R4':3.}
    assert result['final_states']['R4']['success'] is True


def test_camera_adc_packet_must_agree_with_same_tick_ledger_packet():
    data=acquisition_fixture()
    frame=next(frame for frame in data['frames'] if frame['phase']=='HOLD')
    frame['hall']['raw']=3000
    with pytest.raises(ValueError):replay(**data)


def test_speed_reference_cannot_fork_preceding_adc_history():
    data=acquisition_fixture()
    # The fixture FK intentionally ignores compression, so a consistency check
    # must bind packet identity, not just happen to detect a changed position.
    data['records'][200]['speed_reference']['raw_hall_packet']['raw']=3000
    with pytest.raises(ValueError):replay(**data)


def test_contact_suffix_cannot_disappear_with_its_own_annotations():
    data=acquisition_fixture()
    # All real physics rows remain. Only the decision producer's annotations
    # and ledger suffix are removed, exposing self-referential coverage checks.
    first_done=next(index for index,row in enumerate(data['records']) if row['post_state']['phase']=='DONE')
    last_row=data['records'][first_done]['physics_row']
    data['records']=data['records'][:first_done+1]
    for row in data['physics'][last_row+1:]:
        row.pop('observed_supervisor_phase',None)
        row.pop('hold_elapsed_s',None)
    with pytest.raises(ValueError):replay(**data)


def test_training_gate_rejects_audit_from_unapproved_validation_source(tmp_path):
    from sim.auscultation_pi05_v6.train_openpi import verify_observed_audit
    from sim.auscultation_pi05_v6 import observed_supervisor, audit_observed_supervisor
    source_hash=hashlib.sha256(Path(observed_supervisor.__file__).read_bytes()).hexdigest()
    artifacts={'outcome.json':'a'*64}
    audit=dict(schema='wam_observed_supervisor_replay_audit_v1',passed=True,
        source_supervisor_version=observed_supervisor.VERSION,source_supervisor_sha256=source_hash,
        audit_source_sha256=hashlib.sha256(Path(audit_observed_supervisor.__file__).read_bytes()).hexdigest(),
        artifact_sha256=artifacts,replay=dict(every_observation_recomputed=True,
            every_manager_update_replayed=True,all_active_ticks_covered=True))
    derivation=dict(observed_supervisor_version=observed_supervisor.VERSION,
        observed_supervisor_sha256=source_hash,independent_supervisor_audit=dict(
            path='audit.json',artifact_sha256=artifacts))
    path=tmp_path/'audit.json'
    def seal():
        path.write_text(json.dumps(audit))
        derivation['independent_supervisor_audit']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    seal()
    verify_observed_audit(tmp_path,derivation,dict(outcome_sha256='a'*64))
    audit['audit_source_sha256']='f'*64
    seal() # A self-consistent file hash does not authorize an obsolete auditor.
    with pytest.raises(ValueError):
        verify_observed_audit(tmp_path,derivation,dict(outcome_sha256='a'*64))
