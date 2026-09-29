"""Frozen gates and fixed input contract adversarial checks."""
import copy
import importlib
import json
from dataclasses import FrozenInstanceError, asdict

import pytest


def module(name):
    try:
        return importlib.import_module('sim.auscultation_skill_v1.' + name)
    except ModuleNotFoundError as error:
        pytest.fail('V1 implementation missing: ' + str(error))


def test_executor_values_are_frozen_and_config_rejects_relaxed_limits():
    config = module('config')
    executor = config.ExecutorConfig()
    assert executor.normal_speed_limit_m_s == .004
    assert executor.normal_inward_speed_limit_m_s == .002
    assert executor.nominal_compression_reserve_m == .0005
    assert executor.joint_jerk_limit_rad_s3 == 8.
    with pytest.raises(FrozenInstanceError):
        executor.normal_speed_limit_m_s = .02
    frozen = config.frozen_configuration()
    config.validate_config(frozen)
    modified = copy.deepcopy(frozen)
    modified['engineering_gates']['loaded_drag_max_event_m'] = .001
    with pytest.raises(ValueError, match='frozen'):
        config.validate_config(modified)
    assert frozen['executor'] == asdict(executor)


def test_outward_revision_preserves_prior_acceptance_gates_and_other_executor_limits():
    config = module('config')
    previous = json.loads((config.ROOT / 'results/control_revision_v2/before/contract.json').read_text())
    current = config.frozen_configuration()
    assert current['engineering_gates'] == previous['configuration']['engineering_gates']
    expected = dict(previous['configuration']['executor'])
    expected.update(normal_speed_limit_m_s=.004, normal_inward_speed_limit_m_s=.002,
                    nominal_compression_reserve_m=.0005)
    assert current['executor'] == expected


def test_compression_domain_revision_preserves_all_prior_gates_and_other_limits():
    config = module('config')
    previous = json.loads((config.ROOT / 'results/control_revision_v3/before/contract.json').read_text())
    expected = copy.deepcopy(previous['configuration'])
    expected['executor']['nominal_compression_reserve_m'] = .0005
    expected['action_scaling'] = 'inward_scaled_outward_fixed_v1'
    assert config.frozen_configuration() == expected
    relaxed = copy.deepcopy(expected)
    relaxed['executor']['nominal_compression_reserve_m'] = .00075
    with pytest.raises(ValueError, match='frozen'):
        config.validate_config(relaxed)


def test_outward_scaling_revision_preserves_gates_and_rejects_old_action_meaning():
    config = module('config')
    previous = json.loads((config.ROOT / 'results/control_revision_v4/before/contract.json').read_text())
    expected = copy.deepcopy(previous['configuration'])
    expected['action_scaling'] = 'inward_scaled_outward_fixed_v1'
    assert config.frozen_configuration() == expected
    with pytest.raises(ValueError, match='frozen'):
        config.validate_config(previous['configuration'])


def test_contract_rejects_missing_pin_and_escaped_path(tmp_path):
    contract = module('contract')
    config = module('config')
    source = tmp_path / 'sim' / 'accepted.json'
    source.parent.mkdir()
    source.write_text('{}')
    record = {'schema': 'auscultation_skill_contract_v1',
              'target_order': list(config.TARGETS), 'configuration': config.frozen_configuration(),
              'fixed_files': {'sim/accepted.json': contract.sha(source)},
              'required_files': ['sim/accepted.json'],
              'selected_plan_sha256': {name: 'a'*64 for name in config.TARGETS},
              'baseline_phase_config': {'hold_duration_s': 3.}}
    contract.verify_contract(record, project=tmp_path)
    bad = copy.deepcopy(record)
    bad['fixed_files'] = {}
    with pytest.raises(ValueError, match='pin'):
        contract.verify_contract(bad, project=tmp_path)
    bad = copy.deepcopy(record)
    bad['required_files'] = ['../outside.json']
    bad['fixed_files'] = {'../outside.json': 'a'*64}
    with pytest.raises(ValueError, match='unsafe'):
        contract.verify_contract(bad, project=tmp_path)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed'):
        contract.verify_contract(record, project=tmp_path)


def test_outcome_contract_rejects_phase_change_and_partial_point_inventory():
    contract = module('contract')
    fixed = {'baseline_phase_config': {'hold_duration_s': 3.},
             'configuration': module('config').frozen_configuration()}
    outcome = {'success': True, 'phase_config': {'hold_duration_s': 3.},
               'requested_points': ['L4'], 'points': [{'target': 'L4', 'success': True}],
               'speed_scale': 1., 'executor_config': fixed['configuration']['executor'],
               'actual_contact_policy': {'action_scaling': 'inward_scaled_outward_fixed_v1',
                                         'desired_normal_velocity_bounds_m_s': [-.002, .004]}}
    contract.validate_outcome_contract(outcome, fixed, ['L4'], speed_scale=1.)
    altered = copy.deepcopy(outcome)
    altered['phase_config']['hold_duration_s'] = 2.
    with pytest.raises(ValueError, match='phase'):
        contract.validate_outcome_contract(altered, fixed, ['L4'], speed_scale=1.)
    altered = copy.deepcopy(outcome)
    altered['points'] = []
    with pytest.raises(ValueError, match='point'):
        contract.validate_outcome_contract(altered, fixed, ['L4'], speed_scale=1.)
    altered = copy.deepcopy(outcome)
    altered['executor_config']['joint_jerk_limit_rad_s3'] = 20.
    with pytest.raises(ValueError, match='executor'):
        contract.validate_outcome_contract(altered, fixed, ['L4'], speed_scale=1.)
    altered = copy.deepcopy(outcome)
    altered['actual_contact_policy']['action_scaling'] = 'both_directions_scaled'
    with pytest.raises(ValueError, match='scaling'):
        contract.validate_outcome_contract(altered, fixed, ['L4'], speed_scale=1.)


def test_repository_contract_pins_accepted_selected_plans():
    contract = module('contract')
    config = module('config')
    record = contract.read(config.ROOT / 'contract.json')
    contract.verify_contract(record)
    assert record['selected_plans'] == 'sim/upper_pair_posture_20260924/results/candidate_b_plans'
    assert record['target_order'] == list(config.TARGETS)
    assert len(record['fixed_files']) >= 30


def test_speed_variation_changes_only_four_travel_watchdogs():
    contract = module('contract')
    phase = {'hold_duration_s': 3., 'hard_force_limit_N': 5.5,
             'approach_timeout_s': 120., 'contact_timeout_s': 20.,
             'retract_timeout_s': 20., 'episode_timeout_s': 180., 'hold_timeout_s': 20.}
    fixed = {'baseline_phase_config': phase, 'reference_speed_scale': 1.,
             'configuration': module('config').frozen_configuration()}
    expected = dict(phase)
    for key in ('approach_timeout_s', 'contact_timeout_s', 'retract_timeout_s', 'episode_timeout_s'):
        expected[key] /= .75
    outcome = {'success': True, 'phase_config': expected, 'requested_points': ['L4'],
               'points': [{'target': 'L4', 'success': True}], 'speed_scale': .75,
               'executor_config': fixed['configuration']['executor'],
               'actual_contact_policy': {'action_scaling': 'inward_scaled_outward_fixed_v1',
                                         'desired_normal_velocity_bounds_m_s': [-.0015, .004]}}
    contract.validate_outcome_contract(outcome, fixed, ['L4'], speed_scale=.75)
    outcome['phase_config']['hard_force_limit_N'] = 6.
    with pytest.raises(ValueError, match='phase'):
        contract.validate_outcome_contract(outcome, fixed, ['L4'], speed_scale=.75)


def test_public_gate_mapping_cannot_redefine_frozen_defaults():
    config = module('config')
    with pytest.raises(TypeError):
        config.ENGINEERING_GATES['joint_jerk_rad_s3'] = 100.
