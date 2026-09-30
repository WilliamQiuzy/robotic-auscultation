import hashlib
import inspect

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.ausc_policy import m2_contract as m
from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics


def metadata():
    return dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
                freshly_generated=True, contract_id=m.CONTRACT_ID, state_dim=17,
                horizon=20, execution_prefix=10, control_hz=20, joint_unit='rad',
                position_unit='m', target_pose_frame='openarm_left_link0',
                target_source='rgbd_surface_tracking')


Q = np.array([.1, -.2, -.1, .3, .1, .1, -.2])


def pose(rotation=(0., 0., 0.), position=(0., 0., 0.)):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    out[:3, 3] = position
    return out


def test_state_has_exactly_q_and_estimated_target_relative_uncompressed_tcp():
    target = pose((.2, -.1, .3), (.05, .1, .8))
    kin = OpenArmKinematics()
    tcp = kin.fk(Q) @ pose(position=(0, 0, .132))
    relative = np.linalg.inv(target) @ tcp
    value = m.encode_state(Q, target, .6, metadata=metadata())
    assert value.shape == (17,)
    np.testing.assert_array_equal(value[:7], Q)
    np.testing.assert_allclose(value[7:10], relative[:3, 3], atol=1e-12)
    np.testing.assert_allclose(value[10:13], relative[:3, 0], atol=1e-12)
    np.testing.assert_allclose(value[13:16], relative[:3, 1], atol=1e-12)
    assert value[16] == .6
    assert not {'hall', 'compression', 'dq', 'force', 'task_id'} & set(inspect.signature(m.encode_state).parameters)
    with pytest.raises(TypeError):
        m.encode_state(Q, target, .6, metadata=metadata(), compression=.005)


def test_state_responds_to_estimate_and_measured_q():
    a = m.encode_state(Q, np.eye(4), .8, metadata=metadata())
    b = m.encode_state(Q, pose(position=(.01, -.02, .03)), .8, metadata=metadata())
    np.testing.assert_allclose(b[7:10], a[7:10] - [.01, -.02, .03])
    changed = Q.copy(); changed[2] += .01
    c = m.encode_state(changed, np.eye(4), .8, metadata=metadata())
    assert np.linalg.norm(c[7:10] - a[7:10]) > 1e-5


def test_relative_pose_is_invariant_to_common_rigid_frame_change():
    tcp = pose((.3, .2, -.4), (.1, .2, .7))
    target = pose((.1, -.5, .3), (-.1, .4, .9))
    world = pose((-.4, .2, .6), (3., -2., 1.))
    np.testing.assert_allclose(m.target_relative_pose(world @ tcp, world @ target),
                               m.target_relative_pose(tcp, target), atol=1e-12)


@pytest.mark.parametrize('change', [dict(state_dim=41), dict(horizon=10),
                                   dict(execution_prefix=1), dict(position_unit='mm'),
                                   dict(geometry_source_sha256='a' * 64),
                                   dict(freshly_generated=False), dict(target_source='ground_truth')])
def test_old_contract_and_privileged_targets_are_rejected(change):
    meta = metadata(); meta.update(change)
    with pytest.raises(ValueError):
        m.encode_state(Q, np.eye(4), .5, metadata=meta)


@pytest.mark.parametrize('bad', [np.diag([1., 1., -1., 1.]), np.zeros((4, 4)),
                                 np.eye(3), np.eye(4) * float('nan')])
def test_improper_target_pose_rejected(bad):
    with pytest.raises(ValueError):
        m.encode_state(Q, bad, .5, metadata=metadata())


@pytest.mark.parametrize('confidence', [-.01, 1.01, float('nan'), True])
def test_invalid_confidence_rejected(confidence):
    with pytest.raises(ValueError):
        m.encode_state(Q, np.eye(4), confidence, metadata=metadata())


def test_joint_block_relative_uses_single_copied_start_anchor():
    commands = Q + np.arange(1, 21)[:, None] * .001
    start = Q.copy()
    block = m.encode_joint_chunk(commands, start, arm='block_relative_joint', metadata=metadata())
    np.testing.assert_allclose(block.values, commands - start)
    start[:] = 2.  # changing subsequent measured q cannot move the block anchor
    np.testing.assert_allclose(m.decode_joint_chunk(block), commands)
    np.testing.assert_allclose(m.decode_joint_chunk(block), commands)
    with pytest.raises(TypeError):
        m.decode_joint_chunk(block, current_q=start)
    with pytest.raises(ValueError):
        block.values[0, 0] = 0.
    absolute = m.encode_joint_chunk(commands, Q, arm='absolute_joint', metadata=metadata())
    np.testing.assert_array_equal(m.decode_joint_chunk(absolute), commands)


def test_invalid_actions_and_undefined_elbow_contract_are_not_silently_encoded():
    for commands in (np.zeros((19, 7)), np.full((20, 7), np.nan), np.zeros((20, 41))):
        with pytest.raises(ValueError):
            m.encode_joint_chunk(commands, Q, arm='absolute_joint', metadata=metadata())
    with pytest.raises(NotImplementedError, match='psi'):
        m.encode_target_pose_chunk(np.zeros((20, 4, 4)), metadata=metadata())


@pytest.mark.parametrize('name', ['observed_q', 'values'])
def test_block_arrays_cannot_reenable_writes(name):
    commands = Q + np.arange(1, 21)[:, None] * .001
    block = m.encode_joint_chunk(commands, Q, arm='block_relative_joint', metadata=metadata())
    with pytest.raises(ValueError):
        getattr(block, name).setflags(write=True)
    np.testing.assert_allclose(m.decode_joint_chunk(block), commands)


def context(phase='FREE', target='R3', episode='e1'):
    return m.ChunkContext(episode, target, phase, 'L2')


def test_time_alignment_is_future_twenty_hz_and_h20_e10():
    assert (m.HORIZON, m.EXECUTION_PREFIX, m.PERIOD_S) == (20, 10, .05)
    times = 2. + np.arange(1, 21) * .05
    m.validate_chunk_timing(2., times, context(), [context()] * 20, metadata=metadata())
    for bad in (times[:-1], times - .05, times + .001):
        with pytest.raises(ValueError):
            m.validate_chunk_timing(2., bad, context(), [context()] * len(bad), metadata=metadata())


@pytest.mark.parametrize('changed', [context('CONTACT'), context('FREE', 'R4'),
                                     context('FREE', 'R3', 'e2'), m.ChunkContext('e1', 'R3', 'FREE', 'L3')])
def test_chunk_cannot_cross_l3_target_or_episode(changed):
    contexts = [context()] * 19 + [changed]
    with pytest.raises(ValueError):
        m.validate_chunk_timing(0., np.arange(1, 21)*.05, context(), contexts, metadata=metadata())


@pytest.mark.parametrize('start,end', [('FREE', 'APPROACH'), ('APPROACH', 'FREE')])
def test_chunk_cannot_cross_l2_phase_change(start, end):
    contexts = [context(start)] * 19 + [context(end)]
    with pytest.raises(ValueError, match='phase'):
        m.validate_chunk_timing(0., np.arange(1, 21)*.05, context(start), contexts,
                                metadata=metadata())


def episode(tmp_path):
    meta = metadata()
    meta.update(subject_id='p000439', split='train', episode_id='fresh-e1',
                identity_split_sha256=m.IDENTITY_SPLIT_SHA256, source_kind='isaac_live_physics',
                replay=False, artifacts={})
    for name in m.REQUIRED_ARTIFACTS:
        path = tmp_path / (name + '.bin'); path.write_bytes(('fresh-' + name).encode())
        meta['artifacts'][name] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return meta


def test_episode_provenance_binds_actual_bytes_and_frozen_subject_split(tmp_path):
    meta = episode(tmp_path)
    m.require_episode(meta, expected_split='train', artifact_root=tmp_path)
    (tmp_path / 'depth.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='depth'):
        m.require_episode(meta, expected_split='train', artifact_root=tmp_path)


@pytest.mark.parametrize('change', [dict(subject_id='p000438'), dict(subject_id='p000452'),
                                   dict(split='validation'), dict(replay=True),
                                   dict(source_kind='static_render'), dict(identity_split_sha256='a'*64)])
def test_old_or_wrong_split_or_replay_episode_rejected(tmp_path, change):
    meta = episode(tmp_path); meta.update(change)
    with pytest.raises(ValueError):
        m.require_episode(meta, expected_split='train', artifact_root=tmp_path)


def test_missing_multimodal_source_and_unmodeled_noise_rejected(tmp_path):
    meta = episode(tmp_path); del meta['artifacts']['hall_adc']
    with pytest.raises(ValueError, match='hall_adc'):
        m.require_episode(meta, expected_split='train', artifact_root=tmp_path)
    meta = episode(tmp_path); meta['target_source'] = 'gt_plus_measured_noise'
    with pytest.raises(ValueError, match='noise_model'):
        m.require_episode(meta, expected_split='train', artifact_root=tmp_path)
