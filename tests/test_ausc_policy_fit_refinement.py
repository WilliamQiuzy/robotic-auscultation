"""Pure injected distance tests; no native FCL or simulated physics."""
from copy import deepcopy
import json

import numpy as np
import pytest

from sim.ausc_policy.finite_head import FiniteHeadFit

CONTACT_SURFACE = "membrane_and_device_retainer_v2"


def orientation(tilt, azimuth=0., roll=0., normal=(0., 0., 1.)):
    normal = np.asarray(normal, float)
    x = np.cross(normal, [0., 0., 1.])
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(normal, [1., 0., 0.])
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    tilt, azimuth, roll = np.deg2rad([tilt, azimuth, roll])
    n = np.cos(tilt)*normal + np.sin(tilt)*(np.cos(azimuth)*x + np.sin(azimuth)*y)
    tangent = x - n*(x@n)
    tangent /= np.linalg.norm(tangent)
    a = np.cos(roll)*tangent + np.sin(roll)*np.cross(n, tangent)
    return np.column_stack((a, np.cross(-n, a), -n))


def distance_oracle(target=None):
    """FiniteHeadFit.first_contact with analytic distances injected below it."""
    fit = object.__new__(FiniteHeadFit)
    calls = []

    def check(center, rotation, compression):
        # The plane is through the unchanged anchor; all rejected orientations
        # fail the original fixed-component floor, never a custom pass flag.
        margin = -.0001 if target is None else .000005 - .001*np.linalg.norm(rotation-target)
        signed_gap = float(center[2])
        calls.append((center.copy(), rotation.copy(), compression))
        return dict(contact_surface=CONTACT_SURFACE,distances_m=dict(fixed=.003+margin, clamp=.01, shell=.001,
                                     contact_face=max(0., signed_gap)),
                    contact_face_collision=signed_gap <= 0.,
                    diagnostic_marker={'retained': [1, 2, 3]})

    fit.check = check
    return fit, calls


def test_coarse_enumeration_matches_every_original_orientation_in_order():
    from sim.ausc_policy.fit_refinement import FitRefinementConfig, refine_finite_head_fit
    position = np.array([.1, .2, .3])
    normal = np.array([.3, .4, np.sqrt(.75)])
    oracle, _ = distance_oracle()
    old_calls = []
    original = oracle.first_contact

    def observed(*args):
        old_calls.append(tuple(np.array(value, copy=True) for value in args))
        return original(*args)

    oracle.first_contact = observed
    old = oracle.candidates(position, normal, compression=.0025)
    records = []
    result = refine_finite_head_fit(oracle, position, normal, sink=records.append,
                                   config=FitRefinementConfig(top_k=0))
    assert len(records) == old['orientation_attempts'] == 1260
    for record, previous in zip(records, old_calls[:1260]):
        np.testing.assert_array_equal(record['position_world_m'], previous[0])
        np.testing.assert_array_equal(record['local_normal_world'], previous[1])
        np.testing.assert_allclose(record['rotation_world'], previous[2], atol=1e-14, rtol=0)
        assert record['compression_m'] == previous[3] == .0025
    assert result['orientation_attempts'] == 1260
    assert result['contact_surface'] == CONTACT_SURFACE
    assert result['search_definition']['contact_surface'] == CONTACT_SURFACE
    assert result['search_definition']['contact_face_gap_band_m'] == [1e-5,3e-5]
    assert all(record['contact_surface'] == CONTACT_SURFACE for record in records)
    assert result['stage_counts'] == {'coarse': 1260, 'refine': 0}
    assert result['outside_family_proposals_skipped'] == 0
    assert not result['candidates']


@pytest.mark.parametrize('tilt', [9., 11.])
def test_refinement_finds_off_grid_fit_on_both_sides_of_failed_coarse_tilt(tilt):
    from sim.ausc_policy.fit_refinement import FitRefinementConfig, refine_finite_head_fit
    target = orientation(tilt, 3.75, 5.)
    oracle, queries = distance_oracle(target)
    records = []
    result = refine_finite_head_fit(oracle, [0, 0, 0], [0, 0, 1], sink=records.append,
                                   config=FitRefinementConfig(top_k=1))
    assert not any(r['first_contact']['passed'] for r in records if r['stage'] == 'coarse')
    assert result['selected_seed_attempts'][0]['angles_deg']['tilt'] == 10.
    assert result['candidates']
    candidate = result['candidates'][0]
    assert candidate['stage'] == 'refine'
    np.testing.assert_allclose(candidate['rotation_world'], target, atol=1e-12)
    assert candidate['normal_deviation_from_surface_deg'] == tilt
    assert candidate['fit']['distances_m']['fixed'] >= .003
    assert 1e-5 <= candidate['fit']['distances_m']['contact_face'] <= 3e-5
    assert all(q[2] == .0025 for q in queries)
    assert all(r['first_contact']['first_contact_search_interval_m'] == [0., .005] for r in records)
    assert result['scope'].find('no CCD') >= 0
    assert result['globally_unreachable'] is None


def test_refinement_deduplicates_rotations_and_never_leaves_original_square():
    from sim.ausc_policy.fit_refinement import FitRefinementConfig, refine_finite_head_fit
    records = []
    normal = np.array([.3, .4, np.sqrt(.75)])
    oracle, _ = distance_oracle()
    result = refine_finite_head_fit(oracle, [0, 0, 0], normal, sink=records.append,
                                   config=FitRefinementConfig(top_k=8))
    keys = [tuple(np.round(np.asarray(r['rotation_world']).ravel(), 12)) for r in records]
    assert len(keys) == len(set(keys))
    assert result['duplicate_orientations_skipped'] > 0
    x = np.cross(normal, [0, 0, 1]); x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    for record in records:
        rotation = np.asarray(record['rotation_world'])
        n = -rotation[:, 2]
        uv = np.array([n@x, n@y])/(n@normal)
        assert np.max(abs(uv)) <= .6 + 1e-12
        np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-12)
    refined_tilts = {r['angles_deg']['tilt'] for r in records if r['stage'] == 'refine'}
    assert {-1., 1.} <= refined_tilts


def test_budget_is_hard_bounded_and_reports_incomplete_search_without_unreachability():
    from sim.ausc_policy.fit_refinement import FitRefinementConfig, refine_finite_head_fit
    oracle, _ = distance_oracle()
    records = []
    result = refine_finite_head_fit(oracle, [0, 0, 0], [0, 0, 1], sink=records.append,
                                   config=FitRefinementConfig(top_k=35, max_calls=1271))
    assert len(records) == result['orientation_attempts'] == 1271
    assert result['budget_exhausted']
    assert not result['search_complete']
    assert result['search_definition']['raw_orientation_upper_bound'] <= 10000
    assert result['globally_unreachable'] is None
    assert result['outcome'] == 'no_fit_found_in_sampled_search'


def test_failure_outputs_are_complete_json_records_with_deterministic_ranking():
    from sim.ausc_policy.fit_refinement import FitRefinementConfig, refine_finite_head_fit
    def run():
        records = []
        oracle, _ = distance_oracle()
        result = refine_finite_head_fit(oracle, [0, 0, 0], [0, 0, 1], sink=records.append,
                                       config=FitRefinementConfig(top_k=2))
        return result, records
    first, records = run()
    second, second_records = run()
    assert first == second and records == second_records
    assert [r['attempt_index'] for r in first['selected_seed_attempts']] == [0, 1]
    assert first['search_definition']['noncontact_clearance_floors_m'] == dict(fixed=.003, clamp=.003, shell=.0003)
    assert first['rejected_orientations'] == {'noncontact_obstruction': len(records)}
    assert records[0]['first_contact']['rejected_fit']['diagnostic_marker'] == {'retained': [1, 2, 3]}
    assert json.loads(json.dumps(records)) == records


def test_oracle_exception_streams_attempt_and_preserves_prior_records_before_propagation():
    from sim.ausc_policy.fit_refinement import refine_finite_head_fit
    oracle, _ = distance_oracle()
    original = oracle.first_contact
    records = []
    count = 0
    error = RuntimeError('injected query failure')
    def failing(*args):
        nonlocal count
        count += 1
        if count == 4:
            raise error
        return original(*args)
    oracle.first_contact = failing
    with pytest.raises(RuntimeError, match='injected query failure') as caught:
        refine_finite_head_fit(oracle, [0, 0, 0], [0, 0, 1], sink=records.append)
    assert caught.value is error
    assert len(records) == 4
    assert all(r['first_contact']['reason'] == 'noncontact_obstruction' for r in records[:3])
    assert records[-1]['first_contact'] is None
    assert records[-1]['exception'] == {'type': 'RuntimeError', 'message': str(error)}


def test_sink_failure_stops_search_and_does_not_consume_extra_oracle_calls():
    from sim.ausc_policy.fit_refinement import refine_finite_head_fit
    oracle, queries = distance_oracle()
    records = []
    def sink(record):
        records.append(record)
        if len(records) == 3:
            raise OSError('evidence disk unavailable')
    with pytest.raises(OSError, match='evidence disk unavailable'):
        refine_finite_head_fit(oracle, [0, 0, 0], [0, 0, 1], sink=sink)
    assert len(queries) == len(records) == 3


def test_sink_mutations_cannot_change_ranking_or_candidate_evidence():
    from sim.ausc_policy.fit_refinement import FitRefinementConfig, refine_finite_head_fit
    oracle, _ = distance_oracle(orientation(9., 3.75, 5.))
    def sink(record):
        record['first_contact']['passed'] = False
        record['rotation_world'][0][0] = 999.
    result = refine_finite_head_fit(oracle, [0, 0, 0], [0, 0, 1], sink=sink,
                                   config=FitRefinementConfig(top_k=1))
    assert result['candidates']
    assert result['candidates'][0]['rotation_world'][0][0] != 999.


@pytest.mark.parametrize('kwargs', [
    {'top_k': -1}, {'top_k': 36}, {'top_k': True}, {'top_k': 1.5},
    {'max_calls': 1259}, {'max_calls': 10001}, {'max_calls': True}, {'max_calls': float('inf')},
])
def test_invalid_config_rejected(kwargs):
    from sim.ausc_policy.fit_refinement import FitRefinementConfig
    with pytest.raises(ValueError):
        FitRefinementConfig(**kwargs)


@pytest.mark.parametrize('override', [
    {'position': [0, 0]}, {'position': [0, 0, np.nan]},
    {'local_normal': [0, 0, 0]}, {'local_normal': [0, 0, 2]},
    {'compression': .002}, {'compression': np.nan}, {'sink': None},
    {'oracle': object()}, {'config': {}},
])
def test_invalid_inputs_fail_before_any_attempt(override):
    from sim.ausc_policy.fit_refinement import refine_finite_head_fit
    oracle, queries = distance_oracle()
    records = []
    args = dict(oracle=oracle, position=[0, 0, 0], local_normal=[0, 0, 1], sink=records.append)
    args.update(override)
    with pytest.raises((ValueError, TypeError)):
        refine_finite_head_fit(**args)
    assert not records and not queries


def test_invalid_oracle_output_is_streamed_before_fail_closed_validation():
    from sim.ausc_policy.fit_refinement import refine_finite_head_fit
    oracle, _ = distance_oracle()
    invalid = {'passed': 'yes', 'fit_evaluations': 1, 'unmodified': {'a': [1]}}
    oracle.first_contact = lambda *args: deepcopy(invalid)
    records = []
    with pytest.raises(ValueError, match='first_contact'):
        refine_finite_head_fit(oracle, [0, 0, 0], [0, 0, 1], sink=records.append)
    assert len(records) == 1 and records[0]['first_contact'] == invalid


@pytest.mark.parametrize('part,floor', [('fixed', .003), ('clamp', .003), ('shell', .0003)])
def test_contradictory_oracle_cannot_use_ranking_to_pass_below_original_floor(part, floor):
    from sim.ausc_policy.fit_refinement import refine_finite_head_fit
    oracle, _ = distance_oracle()
    distances = dict(fixed=.01, clamp=.01, shell=.001, contact_face=.00002)
    distances[part] = floor - 1e-12
    invalid = dict(contact_surface=CONTACT_SURFACE,passed=True, fit_evaluations=1,
                   fit=dict(contact_surface=CONTACT_SURFACE,distances_m=distances, contact_face_collision=False))
    oracle.first_contact = lambda *args: deepcopy(invalid)
    records = []
    with pytest.raises(ValueError, match='accepted fit'):
        refine_finite_head_fit(oracle, [0, 0, 0], [0, 0, 1], sink=records.append)
    assert len(records) == 1 and records[0]['first_contact'] == invalid


def test_square_corner_refinement_rejects_outside_proposals_and_stays_bounded_at_max_config():
    from sim.ausc_policy.fit_refinement import FitRefinementConfig, refine_finite_head_fit
    oracle, _ = distance_oracle(orientation(40., 45., 0.))
    # The coarse 40-degree pose also fails; its distance margins remain best.
    original = oracle.check
    def blocked(*args):
        report = original(*args)
        report['distances_m']['fixed'] -= .00001
        return report
    oracle.check = blocked
    records = []
    result = refine_finite_head_fit(oracle, [0, 0, 0], [0, 0, 1], sink=records.append,
                                   config=FitRefinementConfig(top_k=35))
    assert result['selected_seed_attempts'][0]['angles_deg']['tilt'] == 40.
    assert result['outside_family_proposals_skipped'] > 0
    assert result['search_complete'] and not result['budget_exhausted']
    assert len(records) <= result['search_definition']['raw_orientation_upper_bound'] <= 10000
    for record in records:
        n = -np.asarray(record['rotation_world'])[:, 2]
        assert max(abs(n[0]/n[2]), abs(n[1]/n[2])) <= .6 + 1e-12
    assert not result['candidates'] and result['globally_unreachable'] is None


def test_failed_search_without_distance_report_keeps_evidence_and_sorts_after_measured_reports():
    from sim.ausc_policy.fit_refinement import FitRefinementConfig, refine_finite_head_fit
    oracle, _ = distance_oracle()
    original = oracle.first_contact
    count = 0
    def missing_report(*args):
        nonlocal count
        count += 1
        if count == 1:
            return dict(contact_surface=CONTACT_SURFACE,passed=False, reason='near_contact_gap_not_resolved', fit_evaluations=7)
        return original(*args)
    oracle.first_contact = missing_report
    records = []
    result = refine_finite_head_fit(oracle, [0, 0, 0], [0, 0, 1], sink=records.append,
                                   config=FitRefinementConfig(top_k=1))
    assert result['selected_seed_attempts'][0]['attempt_index'] == 1
    assert records[0]['first_contact'] == dict(contact_surface=CONTACT_SURFACE,passed=False, reason='near_contact_gap_not_resolved', fit_evaluations=7)
    assert result['fit_attempts'] == len(records) + 6


@pytest.mark.parametrize('layer', ['first_contact','fit'])
def test_unversioned_oracle_cannot_pass_refinement(layer):
    from sim.ausc_policy.fit_refinement import refine_finite_head_fit
    oracle,_=distance_oracle(orientation(0.))
    found=oracle.first_contact(np.zeros(3),[0,0,1],orientation(0.))
    assert found['passed']
    (found if layer=='first_contact' else found['fit']).pop('contact_surface',None)
    oracle.first_contact=lambda *args:deepcopy(found)
    records=[]
    with pytest.raises(ValueError,match='contact.surface'):
        refine_finite_head_fit(oracle,[0,0,0],[0,0,1],sink=records.append)
    assert records[0]['first_contact']==found


def test_unversioned_rejected_fit_cannot_guide_refinement():
    from sim.ausc_policy.fit_refinement import refine_finite_head_fit
    oracle,_=distance_oracle()
    found=oracle.first_contact(np.zeros(3),[0,0,1],orientation(0.))
    assert not found['passed']
    found['rejected_fit'].pop('contact_surface')
    oracle.first_contact=lambda *args:deepcopy(found)
    records=[]
    with pytest.raises(ValueError,match='contact.surface'):
        refine_finite_head_fit(oracle,[0,0,0],[0,0,1],sink=records.append)
    assert records[0]['first_contact']==found
