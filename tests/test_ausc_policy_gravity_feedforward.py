"""CPU contract tests; native PhysX behavior still requires owned-runtime evidence."""
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy import fresh_expert as fresh


def module():
    return importlib.import_module('sim.ausc_policy.gravity_feedforward')


class Robot:
    def __init__(self):
        self.left = np.array([2, 4, 6, 8, 10, 12, 14])
        self.q = np.linspace(-.3, .3, 7)
        self.dq = np.linspace(-.02, .02, 7)
        self.gravity = np.array([[.2, -.4, .6, -.8, 1., -1.2, 1.4]])
        self.kp = np.arange(16, dtype=float)+10.
        self.kd = np.arange(16, dtype=float)*.1+.5
        self.caps = np.full((1, 7), 8.)
        self.applied = np.zeros(16)
        self.actions = []
        self._articulation_view = self

    def get_joint_positions(self, joint_indices):
        np.testing.assert_array_equal(joint_indices, self.left)
        return self.q.copy()

    def get_joint_velocities(self, joint_indices):
        np.testing.assert_array_equal(joint_indices, self.left)
        return self.dq.copy()

    def get_generalized_gravity_forces(self, joint_indices):
        np.testing.assert_array_equal(joint_indices, self.left)
        return self.gravity.copy()

    def get_max_efforts(self, joint_indices):
        np.testing.assert_array_equal(joint_indices, self.left)
        return self.caps.copy()

    def get_articulation_controller(self):
        return self

    def get_gains(self):
        return self.kp.copy(), self.kd.copy()

    def get_applied_joint_efforts(self, joint_indices):
        return self.applied[joint_indices].copy()

    def apply_action(self, action):
        self.actions.append(action)
        if hasattr(action, 'joint_efforts'):
            self.applied[action.joint_indices] = np.asarray(action.joint_efforts, np.float32)


def controller(robot=None, left=None, slider=15):
    robot = robot or Robot()
    return module().NativeGravityFeedforward(robot, robot.left if left is None else left, slider)


def test_explicit_configuration_is_strict_and_missing_remains_disabled():
    configure = module().gravity_feedforward_settings
    assert configure({}) is None
    assert configure({'gravity_feedforward': {'enabled': True}}) == {'enabled': True}
    assert configure({'gravity_feedforward': {'enabled': False}}) == {'enabled': False}
    for bad in (None, True, 1, {}, {'enabled': 1}, {'enabled': 'true'},
                {'enabled': True, 'scale': 2}, {'enabled': np.bool_(True)}):
        with pytest.raises(ValueError, match='gravity_feedforward'):
            configure({'gravity_feedforward': bad})


@pytest.mark.parametrize('indices,slider', [
    ([0, 1, 2, 3, 4, 5], 8), ([0, 1, 2, 3, 4, 5, 5], 8),
    ([0, 1, 2, 3, 4, 5, 6.], 8), ([0, 1, 2, 3, 4, 5, True], 8),
    ([-1, 1, 2, 3, 4, 5, 6], 8), ([[0, 1, 2, 3, 4, 5, 6]], 8),
    ([0, 1, 2, 3, 4, 5, 6], 6), ([0, 1, 2, 3, 4, 5, 6], 7.),
    ([0, 1, 2, 3, 4, 5, 6], True),
])
def test_invalid_or_overlapping_dof_indices_rejected(indices, slider):
    with pytest.raises(ValueError, match='indices|slider'):
        controller(left=indices, slider=slider)


def test_native_positive_gravity_and_existing_gains_produce_auditable_combined_effort():
    robot = Robot(); feedforward = controller(robot)
    target = robot.q+.001
    indices, efforts, receipt = feedforward.prepare(target)
    np.testing.assert_array_equal(indices, [*robot.left, 15])
    np.testing.assert_array_equal(efforts, [*robot.gravity[0], -1.5])
    pd = robot.kp[robot.left]*(target-robot.q)-robot.kd[robot.left]*robot.dq
    np.testing.assert_allclose(receipt['pd_effort_estimate_Nm'], pd)
    np.testing.assert_allclose(receipt['pd_plus_gravity_estimate_Nm'], pd+robot.gravity[0])
    assert receipt['q_target_rad'] == target.tolist()
    assert receipt['q_rad'] == robot.q.tolist() and receipt['dq_rad_s'] == robot.dq.tolist()
    assert receipt['kp'] == robot.kp[robot.left].tolist()
    assert receipt['kd'] == robot.kd[robot.left].tolist()
    assert receipt['effort_limits_Nm'] == robot.caps[0].tolist()
    assert receipt['gravity_efforts_Nm'] == robot.gravity[0].tolist()
    assert not receipt['estimate_is_measured_motor_torque']
    assert not robot.actions
    # Read current gains/limits on every physics command, never reset drives.
    robot.kp *= 2
    assert feedforward.prepare(target)[2]['kp'] == robot.kp[robot.left].tolist()
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize('attribute,bad', [
    ('q', np.zeros((1, 7))), ('q', np.full(7, np.nan)),
    ('dq', np.zeros(6)), ('dq', np.full(7, np.inf)),
    ('gravity', np.zeros(7)), ('gravity', np.full((1, 7), np.nan)),
    ('caps', np.full(7, 8.)), ('caps', np.zeros((1, 7))),
    ('caps', np.full((1, 7), np.inf)), ('kp', np.zeros((1, 16))),
    ('kd', np.zeros(15)), ('kp', np.full(16, -1.)),
    ('kp', np.full(16, np.nan)),
])
def test_malformed_native_data_rejects_before_effort_action(attribute, bad):
    robot = Robot(); setattr(robot, attribute, bad)
    with pytest.raises(ValueError):
        controller(robot).prepare(np.zeros(7))
    assert not robot.actions


@pytest.mark.parametrize('target', [np.zeros((1, 7)), np.full(7, np.nan), ['0']*7])
def test_malformed_position_command_rejected_before_effort_action(target):
    with pytest.raises(ValueError):
        controller().prepare(target)


@pytest.mark.parametrize('mode', ['gravity', 'pd_plus_gravity'])
def test_effort_bounds_reject_without_clipping_or_increasing_existing_limits(mode):
    robot = Robot(); robot.q[:] = 0; robot.dq[:] = 0
    target = np.zeros(7)
    if mode == 'gravity':
        robot.gravity[0, 0] = 8.001
    else:
        target[0] = (8.-robot.gravity[0, 0]+.001)/robot.kp[robot.left[0]]
    with pytest.raises(module().GravityFeedforwardRejected, match=mode) as failure:
        controller(robot).prepare(target)
    assert not robot.actions
    assert np.all(robot.caps == 8.)
    assert not failure.value.receipt['effort_bounds_passed']
    json.dumps(failure.value.receipt, allow_nan=False)


@pytest.mark.parametrize('overwritten_index',[2,15])
def test_applied_efforts_allow_float32_roundtrip_but_reject_overwrite_with_receipt(overwritten_index):
    robot = Robot(); feedforward = controller(robot)
    indices, efforts, receipt = feedforward.prepare(robot.q)
    robot.apply_action(SimpleNamespace(joint_indices=indices, joint_efforts=efforts))
    feedforward.verify_applied(receipt)
    assert receipt['applied_efforts_persisted']
    robot.applied[overwritten_index] = 0.
    with pytest.raises(module().GravityFeedforwardRejected, match='persist'):
        feedforward.verify_applied(receipt)
    assert not receipt['applied_efforts_persisted']
    assert receipt['post_step_applied_efforts'][list(indices).index(overwritten_index)] == 0.


def fake_env(monkeypatch, enabled):
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.types',
                        SimpleNamespace(ArticulationAction=SimpleNamespace))
    env = fresh.FreshContactEnv.__new__(fresh.FreshContactEnv)
    env.robot = Robot(); env.left = env.robot.left; env.slider = 15
    env.passive_preload_N = 1.5; env.video_recorder = None; env.rows = []
    if enabled:
        env.gravity_feedforward = controller(env.robot)
    return env


def test_missing_variant_keeps_original_effort_position_and_metadata_path(monkeypatch):
    env = fake_env(monkeypatch, False); target = np.zeros(7); supplied = {'subphase': 'home'}
    def parent(self, q_target, label, target_name, meta):
        assert len(self.robot.actions) == 1
        np.testing.assert_array_equal(self.robot.actions[0].joint_indices, [15])
        np.testing.assert_array_equal(self.robot.actions[0].joint_efforts, [-1.5])
        assert q_target is target
        assert meta == dict(subphase='home', explicit_passive_preload_N=1.5, training_eligible=False)
        return {'unmodified': True}
    monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent)
    assert env.step(target, 'HOME', 'R3', supplied) == {'unmodified': True}
    assert supplied == {'subphase': 'home'}


@pytest.mark.parametrize('fault', ['none', 'overwrite', 'native_read'])
def test_enabled_variant_preserves_row_and_metadata_on_poststep_effort_failure(monkeypatch, fault):
    env = fake_env(monkeypatch, True); target = env.robot.q.copy()
    def parent(self, q_target, label, target_name, meta):
        self.robot.apply_action(SimpleNamespace(joint_positions=q_target, joint_indices=self.left))
        self.rows.append(dict(q_command=q_target.tolist(), **meta))
        if fault == 'overwrite': self.robot.applied[self.left] = 0.
        if fault == 'native_read':
            self.robot.get_applied_joint_efforts = lambda **kwargs: np.full(8, np.nan)
        return {'unmodified': True}
    monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent)
    if fault == 'none':
        assert env.step(target, 'CONTACT', 'R3', {'subphase': 'approach'}) == {'unmodified': True}
    else:
        with pytest.raises(ValueError, match='persist|applied'):
            env.step(target, 'CONTACT', 'R3', {'subphase': 'approach'})
    assert len(env.rows) == 1 and env.rows[0]['subphase'] == 'approach'
    assert env.rows[0]['q_command'] == target.tolist()
    assert not env.rows[0]['training_eligible']
    receipt = env.rows[0]['gravity_feedforward']
    assert receipt['applied_efforts_persisted'] == (fault == 'none')
    assert receipt['gravity_efforts_Nm'] == env.robot.gravity[0].tolist()
    json.dumps(env.rows, allow_nan=False)


@pytest.mark.parametrize('fault', ['cap', 'native_read', 'malformed_gravity', 'malformed_command'])
def test_prestep_failures_preserve_partial_receipt_without_a_physics_row(monkeypatch, fault):
    env = fake_env(monkeypatch, True); target = env.robot.q.copy()
    if fault == 'cap':
        env.robot.gravity[0, 0] = 8.001
    elif fault == 'native_read':
        def fail(**kwargs): raise RuntimeError('native gravity unavailable')
        env.robot.get_generalized_gravity_forces = fail
    elif fault == 'malformed_gravity':
        env.robot.gravity[0, 0] = np.nan
    else:
        target = ['invalid']*7
    with pytest.raises(module().GravityFeedforwardRejected) as failure:
        env.step(target, 'CONTACT', 'R3', {'subphase': 'approach'})
    assert not env.rows and not env.robot.actions
    entry = env.gravity_feedforward_failure_receipts[-1]
    assert entry['stage'] == 'prepare' and entry['target'] == 'R3'
    assert entry['subphase'] == 'approach'
    assert entry['physics_rows_before_attempt'] == entry['physics_rows_after_attempt'] == 0
    assert entry['executed_physics_row_index'] is None
    assert not entry['effort_action_applied'] and not entry['physics_step_executed']
    assert entry['receipt'] == failure.value.receipt
    receipt = entry['receipt']
    assert not receipt['effort_bounds_passed']
    if fault != 'malformed_command':
        assert receipt['q_target_rad'] == target.tolist()
        assert receipt['q_rad'] == env.robot.q.tolist()
        assert receipt['dq_rad_s'] == env.robot.dq.tolist()
        assert receipt['pd_effort_estimate_Nm'] is not None
    if fault == 'cap':
        assert receipt['gravity_efforts_Nm'][0] == 8.001
        assert receipt['effort_limits_Nm'] == [8.]*7
        assert receipt['pd_plus_gravity_estimate_Nm'] is not None
        assert receipt['requested_efforts'][-1] == -1.5
    elif fault == 'malformed_gravity':
        assert receipt['raw_inputs']['gravity_efforts_Nm'][0][0] is None
        assert receipt['gravity_efforts_Nm'] is None
    elif fault == 'malformed_command':
        assert receipt['raw_inputs']['q_target_rad'] == ['invalid']*7
        assert receipt['q_rad'] is None
    else:
        assert receipt['gravity_efforts_Nm'] is None
        assert 'native gravity unavailable' in receipt['preparation_error']
    json.dumps(entry, allow_nan=False)


@pytest.mark.parametrize('fault', ['cap', 'native_read'])
def test_run_probe_saves_prestep_failure_independently_of_trace_and_in_final_result(tmp_path, monkeypatch, fault):
    from sim.ausc_policy import contact_collision
    from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
    env = fake_env(monkeypatch, True); env.robot.q[:] = 0.
    env.base = np.eye(4); env.out = tmp_path/'fresh_expert'; env.out.mkdir()
    env.state = lambda: dict(q=env.robot.q.copy())
    called = []
    env.world = SimpleNamespace(stop=lambda: called.append('stop'))
    def fail_save(name): raise OSError('trace unavailable')
    env.save = fail_save
    if fault == 'cap': env.robot.gravity[0, 0] = 8.001
    else:
        def fail(**kwargs): raise RuntimeError('native gravity unavailable')
        env.robot.get_generalized_gravity_forces = fail
    class Controller:
        def __init__(self, environment, *args, **kwargs):
            self.env = environment; self.control_rows = []
        def run(self, points):
            self.env.step(np.zeros(7), 'HOME', 'R3', {'subphase': 'home'})
    class Recorder:
        def __init__(self, *args): pass
        def close(self): return {'file': 'unchanged.mp4'}
    metadata = dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
        freshly_generated=True, geometry_artifacts={}, world_from_left_base=np.eye(4).tolist(), plans=[])
    monkeypatch.setattr(sys, 'platform', 'linux'); monkeypatch.setenv('AUSC_OWNED_RUN', '1')
    monkeypatch.setattr(fresh, 'read_plans', lambda *args: ([object()], metadata))
    monkeypatch.setattr(contact_collision, 'ContactCollisionChecker', lambda *args: SimpleNamespace(static_bundle={}))
    monkeypatch.setattr(fresh, 'FreshContactEnv', lambda *args, **kwargs: env)
    monkeypatch.setattr(fresh, 'FreshExpert', Controller)
    monkeypatch.setattr(fresh, '_VideoRecorder', Recorder)
    monkeypatch.setattr(fresh, 'require_geometry_artifacts', lambda *args: {})
    config = dict(geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
        fresh_expert=dict(passive_preload_N=1.5, gravity_feedforward=dict(enabled=True)))
    result = fresh.run_probe(None, None, tmp_path, config, dict(mass_kg=.067))
    assert called == ['stop'] and not env.rows and not env.robot.actions
    assert not result['success'] and not result['physical_validation']
    assert result['physics_steps'] == 0
    saved = json.loads((tmp_path/'fresh_expert_result.json').read_text())
    ledger = json.loads((tmp_path/'fresh_expert_gravity_feedforward_failures.json').read_text())
    assert saved['gravity_feedforward_failure_receipts'] == ledger == env.gravity_feedforward_failure_receipts
    assert result['gravity_feedforward_failure_receipts'] == ledger
    assert len(ledger) == 1 and not ledger[0]['physics_step_executed']
    assert ledger[0]['receipt']['q_rad'] == [0.]*7
    assert any(item['stage'] == 'trace_save' for item in result['errors'])
