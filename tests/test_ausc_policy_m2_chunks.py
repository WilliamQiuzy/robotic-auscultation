"""Numerical H20 assembly tests; fixtures are not qualified physics episodes."""
from dataclasses import asdict, replace
import importlib

import numpy as np
import pytest

from sim.ausc_policy import m2_contract as contract
from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256


def module():
    return importlib.import_module('sim.ausc_policy.m2_chunks')


def metadata():
    return dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
        freshly_generated=True, contract_id=contract.CONTRACT_ID, state_dim=17,
        horizon=20, execution_prefix=10, control_hz=20, joint_unit='rad', position_unit='m',
        target_pose_frame='openarm_left_link0', target_source='rgbd_surface_tracking')


def inputs():
    m = module(); context = contract.ChunkContext('episode-1', 'R3', 'APPROACH', 'L2')
    observed = m.M2Observation(source_id='observations:83', physics_row=1000, time_s=7.3,
        measured_q=np.linspace(-.2, .2, 7), estimated_target_base=np.eye(4),
        target_confidence=.8, target_source='rgbd_surface_tracking', context=context)
    commands = [m.PhysicsCommand(source_id=f'commands:{1000+tick}', physics_row=1000+tick,
        time_s=7.3+tick/240., q_command=observed.measured_q+.2+tick*.00001, context=context)
        for tick in range(1, 241)]
    return observed, commands


@pytest.mark.parametrize('arm', ['block_relative_joint', 'absolute_joint'])
def test_future_endpoints_and_state_use_measured_start_not_current_command(arm):
    observation, commands = inputs()
    result = module().assemble_joint_sample(observation, commands, arm=arm, metadata=metadata())
    selected = commands[11::12]
    expected = np.asarray([row.q_command for row in selected])
    np.testing.assert_allclose(contract.decode_joint_chunk(result.action_block), expected)
    np.testing.assert_array_equal(result.state[:7], observation.measured_q)
    np.testing.assert_allclose(result.state, contract.encode_state(observation.measured_q,
        observation.estimated_target_base, .8, metadata=metadata()))
    assert [row.physics_row for row in result.action_sources] == list(range(1012, 1241, 12))
    assert [row.time_s for row in result.action_sources] == [row.time_s for row in selected]
    assert [row.source_id for row in result.command_sources] == [row.source_id for row in commands]
    assert len(result.command_sources) == 240
    assert result.observation_source.physics_row == 1000
    assert result.observation_source.time_s == 7.3
    assert result.observation_source.source_id == 'observations:83'
    assert not result.training_eligible and not result.physical_qualification
    assert not result.artifact_hashes_verified
    assert result.action_block.metadata['training_eligible'] is False
    assert result.action_block.observed_q.tolist() == observation.measured_q.tolist()
    if arm == 'block_relative_joint':
        np.testing.assert_allclose(result.action_block.values, expected-observation.measured_q)
        assert result.action_block.values[-1, 0] > .2


def test_explicit_dict_schema_matches_typed_input_and_output_is_copied():
    observation, commands = inputs()
    source_observation = asdict(observation); source_commands = [asdict(row) for row in commands]
    result = module().assemble_joint_sample(source_observation, source_commands,
        arm='block_relative_joint', metadata=metadata())
    before = contract.decode_joint_chunk(result.action_block)
    source_observation['measured_q'][:] = 9.
    source_observation['context']['phase'] = 'CONTACT'
    source_commands[11]['q_command'][:] = 8.
    np.testing.assert_array_equal(contract.decode_joint_chunk(result.action_block), before)
    np.testing.assert_array_equal(result.state[:7], observation.measured_q)
    assert result.observation_source.context.phase == 'APPROACH'
    with pytest.raises(ValueError): result.state.setflags(write=True)


@pytest.mark.parametrize('field,changed', [
    ('episode_id', 'episode-2'), ('target_id', 'L3'), ('phase', 'FREE'),
    ('phase', 'CONTACT'), ('control_owner', 'L3')])
def test_transient_out_and_back_boundary_between_endpoints_is_rejected(field, changed):
    observation, commands = inputs()
    # Tick 5 is not any 20 Hz endpoint; endpoint-only checking would miss it.
    commands[4] = replace(commands[4], context=replace(commands[4].context, **{field: changed}))
    with pytest.raises(ValueError, match='context|boundary'):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=metadata())


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'reordered', 'current_first', 'short', 'extra', 'endpoint_only'])
def test_incomplete_nonconsecutive_or_current_time_command_stream_is_rejected(mutation):
    observation, commands = inputs()
    if mutation == 'missing': commands.pop(5)
    elif mutation == 'duplicate': commands[5] = commands[4]
    elif mutation == 'reordered': commands[4], commands[5] = commands[5], commands[4]
    elif mutation == 'current_first':
        commands = [replace(row, physics_row=row.physics_row-1, time_s=row.time_s-1/240.) for row in commands]
    elif mutation == 'short': commands.pop()
    elif mutation == 'extra': commands.append(replace(commands[-1], physics_row=1241))
    else: commands = commands[11::12]
    with pytest.raises(ValueError, match='240|consecutive|future'):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=metadata())


@pytest.mark.parametrize('tick', [1, 5, 12, 240])
def test_any_intervening_clock_mismatch_is_rejected(tick):
    observation, commands = inputs()
    commands[tick-1] = replace(commands[tick-1], time_s=commands[tick-1].time_s+1e-5)
    with pytest.raises(ValueError, match='timestamp'):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=metadata())


@pytest.mark.parametrize('bad', [np.full(7, np.nan), np.full(7, np.inf), np.zeros(6), ['0']*7])
def test_nonfinite_or_malformed_intervening_command_rejected(bad):
    observation, commands = inputs(); commands[4] = replace(commands[4], q_command=bad)
    with pytest.raises(ValueError, match='q_command'):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=metadata())


@pytest.mark.parametrize('source', ['ground_truth', 'mesh_truth', 'gt', 'gt_plus_measured_noise'])
def test_truth_or_metadata_mismatched_target_sources_rejected(source):
    observation, commands = inputs(); observation = replace(observation, target_source=source)
    with pytest.raises(ValueError, match='source|Target'):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=metadata())


def test_existing_measured_noise_label_is_allowed_only_when_explicitly_matched():
    observation, commands = inputs(); observation = replace(observation, target_source='gt_plus_measured_noise')
    meta = metadata(); meta['target_source'] = 'gt_plus_measured_noise'
    result = module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=meta)
    assert result.action_block.metadata['target_source'] == 'gt_plus_measured_noise'
    assert not result.training_eligible and not result.artifact_hashes_verified


@pytest.mark.parametrize('location,key', [('observation', 'context'), ('observation', 'estimated_target_base'),
    ('observation', 'target_source'), ('observation', 'source_id'), ('command', 'context'), ('command', 'time_s')])
def test_missing_context_estimates_or_source_fields_are_never_inferred(location, key):
    observation, commands = inputs(); observation = asdict(observation); commands = [asdict(row) for row in commands]
    del (observation if location == 'observation' else commands[4])[key]
    with pytest.raises(ValueError, match='fields'):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=metadata())


@pytest.mark.parametrize('target', ['R0', 'R6', 'L0', 'r3', 'anything', ''])
def test_only_ten_explicit_target_ids_are_accepted(target):
    observation, commands = inputs(); context = replace(observation.context, target_id=target)
    observation = replace(observation, context=context)
    commands = [replace(row, context=context) for row in commands]
    with pytest.raises(ValueError, match='target'):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=metadata())


@pytest.mark.parametrize('bad', [True, 1001., -1, '1001'])
def test_row_ids_require_nonnegative_integers(bad):
    observation, commands = inputs(); commands[0] = replace(commands[0], physics_row=bad)
    with pytest.raises(ValueError, match='physics_row'):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=metadata())


def test_duplicate_source_ids_and_metadata_episode_mismatch_rejected():
    observation, commands = inputs(); original = commands[4]
    commands[4] = replace(original, source_id=commands[3].source_id)
    with pytest.raises(ValueError, match='source_id'):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=metadata())
    commands[4] = original
    with pytest.raises(ValueError, match='episode'):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint',
                                       metadata=dict(metadata(), episode_id='different-episode'))


def test_no_legacy_contract_or_pose_psi_fallback():
    observation, commands = inputs()
    with pytest.raises(ValueError, match='state_dim'):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=dict(metadata(), state_dim=41))
    with pytest.raises(ValueError, match='joint action arm'):
        module().assemble_joint_sample(observation, commands, arm='target_pose', metadata=metadata())


@pytest.mark.parametrize('field,bad', [('measured_q', np.full(7, np.nan)),
    ('estimated_target_base', np.diag([1., 1., -1., 1.])), ('target_confidence', -1.),
    ('time_s', np.nan), ('physics_row', True)])
def test_invalid_actual_observation_cannot_produce_a_sample(field, bad):
    observation, commands = inputs(); observation = replace(observation, **{field: bad})
    with pytest.raises(ValueError):
        module().assemble_joint_sample(observation, commands, arm='absolute_joint', metadata=metadata())


def test_caller_training_flag_cannot_promote_numerical_output():
    observation, commands = inputs()
    result = module().assemble_joint_sample(observation, commands, arm='absolute_joint',
        metadata=dict(metadata(), training_eligible=True,
                      physical_qualification=True, artifact_hashes_verified=True))
    assert not result.training_eligible and not result.action_block.metadata['training_eligible']
    assert not result.physical_qualification and not result.artifact_hashes_verified
    assert not result.action_block.metadata['physical_qualification']
    assert not result.action_block.metadata['artifact_hashes_verified']
