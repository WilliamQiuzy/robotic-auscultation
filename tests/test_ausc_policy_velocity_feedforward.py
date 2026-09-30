"""CPU command contracts; these fakes establish no native physical acceptance."""
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy import fresh_expert as fresh


DT = 1/240


def module():
    return importlib.import_module('sim.ausc_policy.velocity_feedforward')


class Robot:
    def __init__(self):
        self.left = np.array([12, 2, 10, 4, 8, 6, 14])
        self.positions = np.zeros(16)
        self.velocities = np.full(16, .123)
        self.efforts = np.zeros(16)
        self.actions = []
        self.readback = None
    def get_dof_index(self, name):
        return int(self.left[int(name.removeprefix('openarm_left_joint'))-1])
    def get_joint_positions(self, joint_indices):
        return np.full(7, -.1)  # Must never become the derivative baseline.
    def get_joint_velocities(self, joint_indices):
        return np.full(7, .03)
    def apply_action(self, action):
        self.actions.append(action)
        for field, array in [('joint_positions', self.positions),
                             ('joint_velocities', self.velocities), ('joint_efforts', self.efforts)]:
            if getattr(action, field, None) is not None:
                array[action.joint_indices] = np.asarray(getattr(action, field), np.float32)
    def get_applied_action(self):
        if self.readback is not None: return self.readback()
        return SimpleNamespace(joint_positions=self.positions.copy(),
                               joint_velocities=self.velocities.copy(), joint_indices=None)
    def set_joint_velocities(self, *args, **kwargs):
        pytest.fail('Actual joint velocities must never be set')


def env(monkeypatch, enabled=True):
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.types',
                        SimpleNamespace(ArticulationAction=SimpleNamespace))
    result = fresh.FreshContactEnv.__new__(fresh.FreshContactEnv)
    result.robot = Robot(); result.left = result.robot.left; result.slider = 15
    result.rows = []; result.t = 0.; result.passive_preload_N = 1.5
    result.video_recorder = None
    result.world = SimpleNamespace(current_time=5., get_physics_dt=lambda: DT)
    if enabled:
        result.velocity_feedforward = module().NativeVelocityFeedforward(result.robot, result.left, physics_dt=DT)
    return result


def parent_step(environment, q_target, label, target, meta):
    # Same position action -> one physics step -> row order as native parent.
    environment.robot.apply_action(SimpleNamespace(joint_positions=q_target,
                                                   joint_indices=environment.left))
    environment.world.current_time += DT; environment.t += DT
    environment.rows.append(dict(q_command=np.asarray(q_target).tolist(), **meta))
    return dict(unmodified=True)


def test_settings_default_off_and_strict_bool():
    setting = module().velocity_feedforward_settings
    assert setting({}) is None
    for enabled in (True, False):
        assert setting(dict(velocity_feedforward=dict(enabled=enabled))) == dict(enabled=enabled)
    for bad in (None, True, 1, {}, {'enabled': 1}, {'enabled': np.bool_(True)},
                {'enabled': True, 'scale': 1}, {'enabled': 'true'}):
        with pytest.raises(ValueError, match='velocity_feedforward'):
            setting(dict(velocity_feedforward=bad))


def test_first_command_moving_then_stationary_and_same_positions(monkeypatch):
    e = env(monkeypatch); monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    commands = [np.full(7, .2), np.full(7, .2)+DT*.1, np.full(7, .2)+DT*.1]
    for q in commands:
        assert e.step(q, 'HOLD', 'R3', dict(subphase='contact_cycle')) == {'unmodified': True}
    receipts = [row['velocity_feedforward'] for row in e.rows]
    for receipt, expected in zip(receipts, [0., .1, 0.]):
        np.testing.assert_allclose(receipt['velocity_target_rad_s'], expected, atol=1e-14)
        assert receipt['velocity_targets_persisted']
        assert receipt['physics_step_executed']
    assert receipts[0]['velocity_source'] == 'first_command_explicit_stationary_zero'
    for action, q in zip(e.robot.actions[1::2], commands):
        assert action.joint_positions is q
    assert len(e.robot.actions) == 6 and e.t == pytest.approx(3*DT)
    assert e.robot.efforts[15] == -1.5
    assert np.all(e.robot.velocities[[0, 1, 3, 5, 7, 9, 11, 13, 15]] == .123)
    assert 'apply_action' not in e.robot.__dict__
    json.dumps(e.rows, allow_nan=False)


def test_off_is_original_action_path(monkeypatch):
    e = env(monkeypatch, False); monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    e.step(np.ones(7), 'HOME', 'R3')
    assert len(e.robot.actions) == 2
    assert not hasattr(e.robot.actions[-1], 'joint_velocities')
    assert 'velocity_feedforward' not in e.rows[0]


@pytest.mark.parametrize('bad', [np.full(7, np.nan), np.full(7, np.inf), np.zeros(6), ['0']*7])
def test_invalid_command_rejected_without_actions_or_physics(monkeypatch, bad):
    e = env(monkeypatch); monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    with pytest.raises(module().VelocityFeedforwardRejected): e.step(bad)
    assert not e.rows and not e.robot.actions and e.t == 0.
    assert not e.velocity_feedforward_failure_receipts[-1]['physics_step_executed']
    json.dumps(e.velocity_feedforward_failure_receipts, allow_nan=False)


def test_rate_limit_rejects_without_clipping_and_rejection_is_not_baseline(monkeypatch):
    e = env(monkeypatch); monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    e.step(np.zeros(7))
    with pytest.raises(module().VelocityFeedforwardRejected, match='limit'):
        e.step(np.full(7, DT*.17501))
    assert len(e.rows) == 1 and len(e.robot.actions) == 2
    e.step(np.full(7, DT*.174))
    np.testing.assert_allclose(e.rows[-1]['velocity_feedforward']['velocity_target_rad_s'], .174)


@pytest.mark.parametrize('kind', ['indexed', 'batch', 'wrong_indices', 'missing', 'nan', 'overwritten'])
def test_native_readback_selected_dofs_and_rejections(monkeypatch, kind):
    e = env(monkeypatch); monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    e.step(np.zeros(7))
    if kind == 'indexed':
        e.robot.readback = lambda: SimpleNamespace(joint_indices=e.left[::-1],
                                                   joint_velocities=e.robot.velocities[e.left[::-1]])
    elif kind == 'batch':
        e.robot.get_applied_action = None
        e.robot._articulation_view = SimpleNamespace(get_applied_actions=lambda:
            SimpleNamespace(joint_velocities=e.robot.velocities[None, :], joint_indices=None))
    elif kind == 'wrong_indices':
        e.robot.readback = lambda: SimpleNamespace(joint_indices=np.arange(7), joint_velocities=np.ones(7)*.1)
    elif kind == 'missing': e.robot.readback = lambda: SimpleNamespace(joint_velocities=None)
    elif kind == 'nan': e.robot.readback = lambda: SimpleNamespace(joint_velocities=np.full(16, np.nan))
    else: e.robot.readback = lambda: SimpleNamespace(joint_velocities=np.zeros(16))
    if kind in ('indexed', 'batch'): e.step(np.full(7, DT*.1))
    else:
        with pytest.raises(module().VelocityFeedforwardRejected): e.step(np.full(7, DT*.1))
        assert e.velocity_feedforward_failure_receipts[-1]['physics_step_executed']
    assert len(e.rows) == 2
    assert e.rows[-1]['velocity_feedforward']['velocity_targets_persisted'] == (kind in ('indexed', 'batch'))


@pytest.mark.parametrize('row_written', [False, True])
def test_parent_failure_keeps_original_exception_capture_and_advanced_physics_evidence(monkeypatch, row_written):
    e = env(monkeypatch); events = []
    def broken(environment, q, label, target, meta):
        parent_step(environment, q, label, target, meta)
        if not row_written: environment.rows.pop()
        raise RuntimeError('initiating observation failure')
    e.raw_capture = SimpleNamespace(after_physics=lambda environment, before: events.append(len(environment.rows)))
    e.robot.readback = lambda: SimpleNamespace(joint_velocities=np.full(16, np.nan))
    monkeypatch.setattr(fresh.IsaacContactEnv, 'step', broken)
    with pytest.raises(RuntimeError, match='initiating observation failure'):
        e.step(np.zeros(7))
    assert events == [int(row_written)]
    failure = e.velocity_feedforward_failure_receipts[-1]
    assert failure['physics_step_executed']
    assert failure['executed_physics_row_index'] == (0 if row_written else None)
    assert 'initiating observation failure' in failure['original_error']
    assert failure['receipt']['post_step_readback_error']
    assert 'apply_action' not in e.robot.__dict__
    json.dumps(failure, allow_nan=False)


def test_parent_rewrite_is_rejected_and_hook_restored(monkeypatch):
    e = env(monkeypatch)
    def changed(environment, q, *args):
        environment.robot.apply_action(SimpleNamespace(joint_positions=q+.001, joint_indices=e.left))
    monkeypatch.setattr(fresh.IsaacContactEnv, 'step', changed)
    with pytest.raises(module().VelocityFeedforwardRejected, match='position'):
        e.step(np.zeros(7))
    assert not e.rows and e.t == 0.
    assert 'apply_action' not in e.robot.__dict__


def test_finalize_zeroes_targets_without_physics_and_preserves_positions_efforts(monkeypatch, tmp_path):
    e = env(monkeypatch); monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    e.step(np.zeros(7)); e.step(np.full(7, .1*DT))
    e.save = lambda name: None; events = []; e.world.stop = lambda: events.append('stop')
    before = (e.t, e.world.current_time, e.robot.positions.copy(), e.robot.efforts.copy())
    _, errors = fresh.finalize_probe(e, None, None, tmp_path)
    assert not errors and events == ['stop']
    assert e.t == before[0] and e.world.current_time == before[1]
    np.testing.assert_array_equal(e.robot.positions, before[2])
    np.testing.assert_array_equal(e.robot.efforts, before[3])
    np.testing.assert_array_equal(e.robot.velocities[e.left], 0.)
    assert not hasattr(e.robot.actions[-1], 'joint_positions')
    assert e.velocity_feedforward_reset_receipt['velocity_targets_persisted']
    assert not e.velocity_feedforward_reset_receipt['physics_step_executed']
    assert (tmp_path/'fresh_expert_velocity_feedforward_reset.json').exists()


def test_actual_parent_steps_once_with_joint_targets_already_installed(monkeypatch):
    from collections import defaultdict
    e = env(monkeypatch); observed = []
    e.contact_impulses = defaultdict(float); e.contact_pairs = defaultdict(float); e.contact_points = []
    e.filtered_hall_force = 0.; e.previous_tcp = None; e.base = np.eye(4)
    e.poses = []; e.pose_row_indices = []; e.bodies = []
    e.state = lambda: dict(q=np.zeros(7), dq=np.zeros(7), hall_raw_N=0.,
        compression_m=0., tcp_world=np.zeros(3), tcp_base=np.zeros(3),
        axis_world=np.array([0., 0., 1.]), rotation_world=np.eye(3))
    def step(*, render):
        assert render is False
        observed.append(e.robot.velocities[e.left].copy())
        e.world.current_time += DT
    e.world.step = step
    e.step(np.zeros(7)); e.step(np.ones(7)*DT*.1)
    np.testing.assert_array_equal(observed[0], 0.)
    np.testing.assert_allclose(observed[1], .1)
    assert len(e.rows) == 2 and e.t == 2*DT
    assert e.rows[-1]['q_command'] == (np.ones(7)*DT*.1).tolist()


@pytest.mark.parametrize('stage', ['command', 'observation'])
def test_existing_collision_gates_remain_authoritative_with_velocity_enabled(monkeypatch, stage):
    e = env(monkeypatch); monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    state = dict(q=np.zeros(7), compression_m=.0004); e.state = lambda: state
    controller = fresh.FreshExpert.__new__(fresh.FreshExpert)
    controller.env = e; controller.target = SimpleNamespace(name='R3')
    controller.executor_config = dict(contact_clearance_m=.0003)
    controller._contact_active = True; controller.control_rows = []
    controller.command_collision_samples = controller.observed_collision_samples = 0
    queries = []; capture_events = []
    controller.manager = SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='CONTACT')))
    e.raw_capture = SimpleNamespace(after_physics=lambda env, before: capture_events.append('hall'),
        finish_tick=lambda *args, **kwargs: capture_events.append(kwargs.get('error')))
    def check(q, **kwargs):
        queries.append(q.copy())
        clear = stage == 'observation' and len(queries) == 1
        return dict(clear=clear, minimum_distance_m=.00031 if clear else .00029)
    controller.checker = SimpleNamespace(check=check)
    def inherited_tick(self, q, subphase, **kwargs):
        self.env.step(q, 'CONTACT', 'R3', dict(subphase=subphase))
        return state
    monkeypatch.setattr(fresh.SkillExpert.__mro__[1], '_tick', inherited_tick)
    with pytest.raises(fresh.EpisodeRejected, match='mesh collision rejected'):
        controller._tick(np.zeros(7), 'contact_cycle', active=True)
    assert controller.collision_failure_diagnostics[-1]['stage'] == stage
    assert len(e.rows) == (stage == 'observation')
    if stage == 'command': assert not e.robot.actions
    else:
        assert e.rows[-1]['velocity_feedforward']['velocity_targets_persisted']
        assert capture_events[0] == 'hall'
    assert 'mesh collision rejected' in capture_events[-1]


def test_native_gravity_and_raw_capture_coexist_and_diagnostic_uses_target(monkeypatch):
    from sim.ausc_policy.gravity_feedforward import NativeGravityFeedforward
    e = env(monkeypatch); robot = e.robot; events = []
    robot._articulation_view = SimpleNamespace(
        get_generalized_gravity_forces=lambda **kwargs: np.full((1, 7), .5),
        get_max_efforts=lambda **kwargs: np.full((1, 7), 8.))
    robot.get_articulation_controller = lambda: SimpleNamespace(get_gains=lambda: (np.full(16, 10.), np.ones(16)))
    robot.get_applied_joint_efforts = lambda joint_indices: robot.efforts[joint_indices]
    bindings = [dict(joint_index=int(index), joint_name=f'openarm_left_joint{i}',
                    joint_path=f'/Robot/joint{i}', drive_instance='angular', drive_type='acceleration')
                for i, index in enumerate(e.left, 1)]
    e.gravity_feedforward = NativeGravityFeedforward(robot, e.left, 15, drive_mode_provider=lambda: bindings)
    e.raw_capture = SimpleNamespace(after_physics=lambda env, before: events.append(len(env.rows)))
    monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    e.step(np.zeros(7)); e.step(np.full(7, DT*.1))
    assert events == [1, 2]
    np.testing.assert_array_equal(robot.efforts[e.left], .5)
    assert robot.efforts[15] == -1.5
    gravity = e.rows[-1]['gravity_feedforward']
    np.testing.assert_allclose(gravity['desired_velocity_rad_s'], .1)
    np.testing.assert_allclose(gravity['desired_velocity_pd_difference_rad_s2'], .1)
    assert gravity['applied_efforts_persisted']
    assert not gravity['total_actuator_torque_validated']


def test_parent_velocity_overwrite_after_combined_action_is_rejected(monkeypatch):
    e = env(monkeypatch)
    def overwritten(environment, *args):
        result = parent_step(environment, *args)
        environment.robot.velocities[e.left] = .03
        return result
    monkeypatch.setattr(fresh.IsaacContactEnv, 'step', overwritten)
    with pytest.raises(module().VelocityFeedforwardRejected, match='persist'):
        e.step(np.zeros(7))
    assert len(e.rows) == 1
    assert e.velocity_feedforward_failure_receipts[-1]['physics_step_executed']


def test_failed_reset_still_saves_receipt_trace_and_stops_world(monkeypatch, tmp_path):
    e = env(monkeypatch); monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    e.step(np.zeros(7)); e.step(np.full(7, DT*.1))
    events = []; e.save = lambda name: events.append('trace'); e.world.stop = lambda: events.append('stop')
    e.robot.readback = lambda: SimpleNamespace(joint_velocities=np.ones(16))
    _, errors = fresh.finalize_probe(e, None, None, tmp_path)
    assert events == ['trace', 'stop']
    assert any(item['stage'] == 'velocity_feedforward_reset' for item in errors)
    receipt = json.loads((tmp_path/'fresh_expert_velocity_feedforward_reset.json').read_text())
    assert not receipt['velocity_targets_persisted'] and not receipt['physics_step_executed']


@pytest.mark.parametrize('fault', ['dt', 'native_binding'])
def test_live_native_binding_and_physics_dt_cannot_drift(monkeypatch, fault):
    e = env(monkeypatch); monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    if fault == 'dt': e.world.get_physics_dt = lambda: DT*2
    else: e.robot.get_dof_index = lambda name: 1
    with pytest.raises(module().VelocityFeedforwardRejected): e.step(np.zeros(7))
    assert not e.rows and not e.robot.actions


@pytest.mark.parametrize('bad', [np.nan, 'bad', [DT]])
def test_malformed_native_dt_retains_raw_input_rejection_and_zero_execution(monkeypatch, bad):
    e = env(monkeypatch); monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    e.world.get_physics_dt = lambda: bad
    with pytest.raises(module().VelocityFeedforwardRejected) as error:
        e.step(np.zeros(7))
    receipt = error.value.receipt
    assert receipt['preparation_stage'] == 'native_physics_dt_s'
    assert receipt['raw_inputs']['native_physics_dt_s'] == (None if isinstance(bad, float) else bad)
    assert 'native physics dt' in receipt['preparation_error']
    assert isinstance(error.value.__cause__, ValueError)
    assert not e.robot.actions and not e.rows and e.t == 0.
    failure = e.velocity_feedforward_failure_receipts[-1]
    assert not failure['physics_step_executed'] and failure['receipt'] == receipt
    json.dumps(failure, allow_nan=False)


@pytest.mark.parametrize('clock', ['world', 'env', 'both'])
@pytest.mark.parametrize('original_error', [False, True])
def test_malformed_poststep_clock_retains_actual_row_and_attempts_readback(monkeypatch, clock, original_error):
    e = env(monkeypatch); reads = []
    native_read = e.robot.get_applied_action
    e.robot.get_applied_action = lambda: (reads.append('readback'), native_read())[1]
    def corrupted(environment, *args):
        result = parent_step(environment, *args)
        if clock in ('world', 'both'): environment.world.current_time = np.nan
        if clock in ('env', 'both'): environment.t = 'bad'
        if original_error: raise RuntimeError('original post-row failure')
        return result
    monkeypatch.setattr(fresh.IsaacContactEnv, 'step', corrupted)
    expected = RuntimeError if original_error else module().VelocityFeedforwardRejected
    with pytest.raises(expected) as error: e.step(np.zeros(7))
    if original_error: assert str(error.value) == 'original post-row failure'
    assert reads == ['readback'] and len(e.rows) == 1
    receipt = e.rows[-1]['velocity_feedforward']
    assert receipt['physics_step_executed'] and receipt['velocity_targets_persisted']
    assert receipt['after']['physics_rows'] == 1 and receipt['clock_read_errors']
    np.testing.assert_array_equal(e.velocity_feedforward.previous_executed, 0.)
    failure = e.velocity_feedforward_failure_receipts[-1]
    assert failure['physics_step_executed'] and failure['executed_physics_row_index'] == 0
    assert 'clock' in failure['receipt']['secondary_completion_error' if original_error else 'completion_error']
    json.dumps(failure, allow_nan=False)


def test_native_dt_accessor_error_retains_original_cause(monkeypatch):
    e = env(monkeypatch)
    def fail(): raise RuntimeError('native dt unavailable')
    e.world.get_physics_dt = fail
    with pytest.raises(module().VelocityFeedforwardRejected) as error: e.step(np.zeros(7))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == 'native dt unavailable'
    failure = e.velocity_feedforward_failure_receipts[-1]
    assert failure['receipt']['preparation_stage'] == 'native_physics_dt_s'
    assert 'native dt unavailable' in failure['receipt']['preparation_error']
    assert not failure['physics_step_executed'] and not e.robot.actions


@pytest.mark.parametrize('available_clock', ['world', 'env', 'neither'])
def test_partial_clock_evidence_without_row_does_not_invent_row_or_execution(monkeypatch, available_clock):
    e = env(monkeypatch); reads = []
    native_read = e.robot.get_applied_action
    e.robot.get_applied_action = lambda: (reads.append('readback'), native_read())[1]
    def corrupted(environment, *args):
        parent_step(environment, *args); environment.rows.pop()
        if available_clock != 'world': environment.world.current_time = np.nan
        if available_clock != 'env': environment.t = 'bad'
        raise RuntimeError('original pre-row state-read error')
    monkeypatch.setattr(fresh.IsaacContactEnv, 'step', corrupted)
    with pytest.raises(RuntimeError, match='original pre-row state-read error'): e.step(np.zeros(7))
    assert not e.rows and reads == ['readback']
    failure = e.velocity_feedforward_failure_receipts[-1]
    assert failure['executed_physics_row_index'] is None
    assert failure['physics_step_executed'] == (available_clock != 'neither')
    assert failure['receipt']['physics_step_execution_known'] == (available_clock != 'neither')
    assert failure['receipt']['velocity_targets_persisted']
    json.dumps(failure, allow_nan=False)


@pytest.mark.parametrize('clock', ['world', 'both'])
def test_finalize_zero_target_is_independent_of_broken_poststep_clocks(monkeypatch, tmp_path, clock):
    e = env(monkeypatch); monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    e.step(np.zeros(7))
    def corrupted(environment, *args):
        result = parent_step(environment, *args)
        environment.world.current_time = np.nan
        if clock == 'both': environment.t = 'bad'
        return result
    monkeypatch.setattr(fresh.IsaacContactEnv, 'step', corrupted)
    with pytest.raises(module().VelocityFeedforwardRejected): e.step(np.full(7, DT*.1))
    np.testing.assert_allclose(e.robot.velocities[e.left], .1)
    events = []; e.save = lambda name: events.append('trace')
    e.world.stop = lambda: events.append('stop')
    action_count = len(e.robot.actions); position = e.robot.positions.copy(); effort = e.robot.efforts.copy()
    _, errors = fresh.finalize_probe(e, None, None, tmp_path)
    assert len(e.robot.actions) == action_count+1
    np.testing.assert_array_equal(e.robot.velocities[e.left], 0.)
    np.testing.assert_array_equal(e.robot.positions, position)
    np.testing.assert_array_equal(e.robot.efforts, effort)
    assert events == ['trace', 'stop'] and len(e.rows) == 2
    assert any(item['stage'] == 'velocity_feedforward_reset' for item in errors)
    receipt = json.loads((tmp_path/'fresh_expert_velocity_feedforward_reset.json').read_text())
    assert receipt['target_action_applied'] and receipt['velocity_targets_persisted']
    assert not receipt['physics_step_executed'] and not receipt['physics_step_execution_known']
    assert not receipt['reset_no_physics_advance_verified']
    assert receipt['clock_read_errors']['before'] and receipt['clock_read_errors']['after']
    assert receipt['raw_clocks']['before']['world_time_s'] is None
    if clock == 'both': assert receipt['raw_clocks']['before']['env_time_s'] == 'bad'


@pytest.mark.parametrize('fault', ['reset_action', 'reset_readback'])
def test_failed_reset_releases_environment_and_robot_without_gc(monkeypatch, tmp_path, fault):
    import gc
    import weakref
    monkeypatch.setattr(fresh.IsaacContactEnv, 'step', parent_step)
    def scope():
        e = env(monkeypatch)
        e.step(np.zeros(7)); e.step(np.full(7, DT*.1))
        e.save = lambda name: None; e.world.stop = lambda: None
        def fail(*args, **kwargs): raise RuntimeError(fault+' failed')
        if fault == 'reset_action': e.robot.apply_action = fail
        else: e.robot.readback = fail
        references = weakref.ref(e), weakref.ref(e.robot), weakref.ref(e.velocity_feedforward)
        _, errors = fresh.finalize_probe(e, None, None, tmp_path)
        assert any(item['stage'] == 'velocity_feedforward_reset' for item in errors)
        assert fault+' failed' in next(item['error'] for item in errors
                                     if item['stage'] == 'velocity_feedforward_reset')
        return references
    was_enabled = gc.isenabled(); gc.disable()
    try:
        references = scope()
        assert all(reference() is None for reference in references)
    finally:
        if was_enabled: gc.enable()
        gc.collect()
