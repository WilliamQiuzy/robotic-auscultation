import copy
from dataclasses import asdict

import pytest

from sim.auscultation_training.audit_episode import audit_rows, audit_sequence
from sim.auscultation_training.phase_manager import PhaseConfig


def episode():
    rows = []
    for phase, count in [('CONTACT', 96), ('HOLD', 720), ('RETRACT', 48)]:
        for _ in range(count):
            contact = phase != 'RETRACT'
            rows.append(dict(time_s=(len(rows)+1)/240, phase=phase,
                             hall_N=4. if contact else 0., compression_m=.002 if contact else 0.,
                             gap_m=.001 if contact else .05, tangent_m=0., angle_deg=0.,
                             tcp_speed_m_s=0., diaphragm_N=0., lip_N=4. if contact else 0.,
                             contact_rim_N=0., carriage_other_N=0., frame_N=0., forbidden_N=0.,
                             q=[0.]*7, q_command=[0.]*7))
    return rows, {'success': True, 'phase_config': asdict(PhaseConfig())}


def test_front_ring_contact_is_mechanical_only():
    result = audit_rows(*episode())
    assert result['accepted']
    assert result['continuous_front_contact_hold_s'] == 3.
    assert result['hold_membrane_force_max_N'] == 0.
    assert 'acoustic coupling' in result['scope']


def test_disconnected_holds_cannot_be_added_together():
    rows, outcome = episode()
    rows[450]['hall_N'] = 0.
    result = audit_rows(rows, outcome)
    assert not result['accepted']
    assert 'continuous_contact_hold_too_short' in result['reasons']


@pytest.mark.parametrize('key,value,reason', [
    ('forbidden_N', 1., 'forbidden_N_contact'),
    ('hall_N', float('nan'), 'nonfinite_trace'),
    ('time_s', .1, 'timestamp_gap'),
    ('frame_N', 1., 'frame_N_contact'),
])
def test_reject_unsafe_sample_outside_hold(key, value, reason):
    rows, outcome = episode()
    rows[0][key] = value
    result = audit_rows(rows, outcome)
    assert not result['accepted'] and reason in result['reasons']


def test_success_flag_cannot_substitute_for_physical_contact():
    rows, outcome = episode()
    for row in rows:
        row['lip_N'] = 0.
    result = audit_rows(rows, outcome)
    assert not result['accepted']
    assert 'continuous_contact_hold_too_short' in result['reasons']


def test_unreleased_contact_rejected_after_good_hold():
    rows, outcome = episode()
    rows[-1]['lip_N'] = .2
    assert 'release_not_verified' in audit_rows(rows, outcome)['reasons']


def test_each_point_requires_its_own_complete_hold():
    first, p1 = episode()
    second, p2 = episode()
    n = len(first)
    for row in first:
        row['target'] = 'R1'
    for row in second:
        row['target'] = 'L1'
        row['time_s'] += n/240
    p1.update(target='R1',training_accepted=True,start_physics_row=0,contact_end_physics_row=n,end_physics_row=n)
    p2.update(target='L1',training_accepted=True,start_physics_row=n,contact_end_physics_row=2*n,end_physics_row=2*n)
    outcome = dict(success=True,training_accepted=True,points=[p1,p2],requested_points=['R1','L1'])
    assert audit_sequence(first+second,outcome)['accepted']
    for row in second:
        row['lip_N'] = 0.
    report = audit_sequence(first+second,outcome)
    assert not report['accepted']
    assert report['points'][0]['accepted'] and not report['points'][1]['accepted']


def test_total_force_includes_small_forbidden_contact():
    rows, outcome = episode()
    rows[0].update(lip_N=5.9,forbidden_N=.2)
    assert 'physical_force_limit' in audit_rows(rows,outcome)['reasons']


def test_exact_hard_limit_rejected():
    rows, outcome = episode()
    rows[0]['hall_N'] = 5.5
    assert 'hall_force_limit' in audit_rows(rows,outcome)['reasons']
