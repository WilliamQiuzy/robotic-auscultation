"""CPU protocol evidence only; no native Isaac, physics or training qualification."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

LEFT_NAMES = tuple(f'openarm_left_joint{i}' for i in range(1, 8))
NAMES = ('base', *LEFT_NAMES, 'right')
INDICES = tuple(range(1, 8))
REQUESTED = np.array([.1, -.2, .3, -.4, .5, -.6, .7], dtype=np.float64)


def module():
    assert importlib.util.find_spec('sim.ausc_policy.applied_position_targets') is not None, 'applied target evidence component is missing'
    return importlib.import_module('sim.ausc_policy.applied_position_targets')


class Clock:
    def __init__(self, offset=7.):
        self.rows = 0
        self.episode = 0.
        self.world = offset
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return module().PhysicsClock(self.rows, self.episode, self.world)

    def step(self, count=1):
        self.rows += count
        self.episode += count / 240.
        self.world += count / 240.


class Robot:
    def __init__(self, dtype=np.float32, names=NAMES):
        self.dof_names = list(names)
        targets = np.zeros(len(names), dtype=dtype)
        targets[list(INDICES)] = REQUESTED.astype(dtype)
        self.action = SimpleNamespace(joint_positions=targets, joint_indices=None)
        self.reads = 0
        self.mutations = 0
        self.hook = None

    def get_applied_action(self):
        self.reads += 1
        if self.hook:
            self.hook()
        return self.action

    def apply_action(self, *_args, **_kwargs):
        self.mutations += 1
        raise AssertionError('component must not apply commands')

    def set_joint_positions(self, *_args, **_kwargs):
        self.mutations += 1
        raise AssertionError('component must not mutate joints')

    def get_joint_positions(self, *_args, **_kwargs):
        raise AssertionError('measured q is not applied-target evidence')


def make(*, robot=None, clock=None, indices=INDICES, **kwargs):
    robot = robot or Robot()
    clock = clock or Clock()
    obj = module().AppliedPositionTargets(robot, indices, clock_provider=clock,
                                          world_time_offset_s=7., **kwargs)
    return obj, robot, clock


def begin(obj, clock, row=0, requested=REQUESTED, issue=None):
    return obj.begin_command(next_row_index=row, requested_q=requested,
                             issued_clock=issue or clock())


def complete(obj, clock, row=0):
    begin(obj, clock, row)
    clock.step()
    return obj.finish_step()


def assert_false_qualifications(receipt):
    assert receipt.training_eligible is False
    assert receipt.native_qualified is False
    assert receipt.physical_qualification is False


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_two_real_reads_and_backend_quantization(dtype):
    obj, robot, clock = make(robot=Robot(dtype=dtype))
    pre = begin(obj, clock)
    assert robot.reads == 1 and clock.calls == 3
    assert pre.status == 'pre_step_verified' and not pre.verified
    assert pre.command_readback_at_s is None
    assert pre.command_pre_readback_at_s == 0.
    assert pre.pre_read.clock.world_time_s == 7.
    assert pre.pre_read.full_targets.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(pre.requested_q_backend, REQUESTED.astype(dtype))
    np.testing.assert_array_equal(pre.pre_read.left_targets, REQUESTED.astype(dtype))
    assert pre.pre_read.dof_names == NAMES and pre.pre_read.left_indices == INDICES
    assert pre.pre_read.source_api == 'robot.get_applied_action().joint_positions'
    assert_false_qualifications(pre)
    clock.step()
    final = obj.finish_step()
    assert robot.reads == 2 and clock.calls == 5
    assert final.verified and final.status == 'post_step_verified'
    assert final.physics_row == 0 and final.executed_physics_row == 0
    assert final.command_issued_at_s == 0.
    assert final.command_pre_readback_at_s == 0.
    assert final.command_readback_at_s == clock.episode == 1 / 240.
    assert final.post_read.clock.world_time_s == clock.world
    assert final.post_read.clock.physics_row_count == 1
    assert_false_qualifications(final)
    assert robot.mutations == 0


def test_receipts_and_inputs_have_no_mutable_array_aliases():
    obj, robot, clock = make()
    request = REQUESTED.copy()
    pre = begin(obj, clock, requested=request)
    request[:] = 99
    clock.step()
    receipt = obj.finish_step()
    robot.action.joint_positions[:] = 22
    robot.dof_names[:] = ['corrupted'] * 9
    for array in (receipt.requested_q_backend, pre.pre_read.full_targets,
                  pre.pre_read.left_targets, receipt.post_read.full_targets,
                  receipt.post_read.left_targets):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    np.testing.assert_array_equal(receipt.requested_q_backend, REQUESTED.astype(np.float32))
    assert receipt.pre_read.dof_names == NAMES
    assert obj.current_receipt is receipt
    with pytest.raises((AttributeError, TypeError)):
        receipt.status = 'pretend'


def test_consecutive_rows_use_fresh_native_reads_and_fixed_offset():
    obj, robot, clock = make()
    first = complete(obj, clock)
    robot.action.joint_positions[list(INDICES)] = -REQUESTED.astype(np.float32)
    second_pre = begin(obj, clock, 1, -REQUESTED)
    assert obj.current_receipt is second_pre and not second_pre.verified
    clock.step()
    second = obj.finish_step()
    assert second.physics_row == 1 and second.executed_physics_row == 1
    assert first.command_readback_at_s == 1 / 240.
    assert second.command_readback_at_s == 2 / 240.
    assert robot.reads == 4 and not robot.mutations


@pytest.mark.parametrize('indices', [None, np.arange(9), tuple(range(9))])
def test_only_none_or_canonical_full_action_indices_are_accepted(indices):
    obj, robot, clock = make()
    robot.action.joint_indices = indices
    assert complete(obj, clock).verified


def test_arbitrary_full_dof_order_requires_exact_named_left_mapping():
    names = ('right', LEFT_NAMES[3], LEFT_NAMES[0], LEFT_NAMES[6], 'base',
             LEFT_NAMES[2], LEFT_NAMES[1], LEFT_NAMES[4], LEFT_NAMES[5])
    indices = (2, 6, 5, 1, 7, 8, 3)
    robot = Robot(names=names)
    robot.action.joint_positions[:] = 0
    robot.action.joint_positions[list(indices)] = REQUESTED
    obj, _, clock = make(robot=robot, indices=indices)
    assert complete(obj, clock).post_read.left_indices == indices


@pytest.mark.parametrize('names,indices', [
    ((NAMES[0], *reversed(LEFT_NAMES), NAMES[-1]), INDICES),
    ((NAMES[0], *LEFT_NAMES[:-1], LEFT_NAMES[0], NAMES[-1]), INDICES),
    ((NAMES[0], *LEFT_NAMES[:-1], 'missing_left7', NAMES[-1]), INDICES),
    (NAMES, (1, 2, 3, 4, 5, 6, 6)),
    (NAMES, (1, 2, 3, 4, 5, 6, 9)),
    (NAMES, (1, 2, 3, 4, 5, 6)),
    (NAMES, (1., 2, 3, 4, 5, 6, 7)),
    (NAMES, (True, 2, 3, 4, 5, 6, 7)),
    (('base', *LEFT_NAMES, 'base'), INDICES),
])
def test_constructor_rejects_invalid_identity_or_indices(names, indices):
    robot = Robot(names=names)
    with pytest.raises(ValueError):
        make(robot=robot, indices=indices)
    assert not robot.reads and not robot.mutations


def test_documented_names_provider_is_read_each_time():
    robot = Robot()
    names = list(NAMES)
    del robot.dof_names
    obj, _, clock = make(robot=robot, dof_names_provider=lambda: names)
    begin(obj, clock)
    names[0] = 'changed'
    clock.step()
    with pytest.raises(module().AppliedTargetReadbackError, match='DOF'):
        obj.finish_step()
    assert obj.current_receipt.executed_physics_row == 0


@pytest.mark.parametrize('bad', [
    'none_action', 'no_positions', 'none_positions', 'partial_shape', 'batch_shape',
    'int_dtype', 'float16_dtype', 'object_dtype', 'list_targets', 'nan_left', 'inf_other',
    'partial_indices', 'reordered_indices', 'duplicate_indices', 'float_indices',
    'batch_indices', 'bool_indices', 'unequal_request', 'nextafter_target',
])
def test_invalid_native_evidence_fails_before_step_and_latches(bad):
    obj, robot, clock = make()
    if bad == 'none_action': robot.action = None
    elif bad == 'no_positions': del robot.action.joint_positions
    elif bad == 'none_positions': robot.action.joint_positions = None
    elif bad == 'partial_shape': robot.action.joint_positions = np.zeros(7, np.float32)
    elif bad == 'batch_shape': robot.action.joint_positions = robot.action.joint_positions[None, :]
    elif bad == 'int_dtype': robot.action.joint_positions = np.zeros(9, np.int32)
    elif bad == 'float16_dtype': robot.action.joint_positions = np.zeros(9, np.float16)
    elif bad == 'object_dtype': robot.action.joint_positions = np.zeros(9, object)
    elif bad == 'list_targets': robot.action.joint_positions = robot.action.joint_positions.tolist()
    elif bad == 'nan_left': robot.action.joint_positions[1] = np.nan
    elif bad == 'inf_other': robot.action.joint_positions[0] = np.inf
    elif bad == 'partial_indices': robot.action.joint_indices = np.array(INDICES)
    elif bad == 'reordered_indices': robot.action.joint_indices = np.arange(9)[::-1]
    elif bad == 'duplicate_indices': robot.action.joint_indices = np.zeros(9, int)
    elif bad == 'float_indices': robot.action.joint_indices = np.arange(9, dtype=float)
    elif bad == 'batch_indices': robot.action.joint_indices = np.arange(9)[None, :]
    elif bad == 'bool_indices': robot.action.joint_indices = np.ones(9, bool)
    elif bad == 'unequal_request': robot.action.joint_positions[1] = .2
    elif bad == 'nextafter_target':
        robot.action.joint_positions[1] = np.nextafter(robot.action.joint_positions[1], np.float32(1.))
    with pytest.raises(module().AppliedTargetReadbackError):
        begin(obj, clock)
    failure = obj.current_receipt
    assert failure.status == 'failed_before_step' and not failure.verified
    assert failure.executed_physics_row is None and failure.command_readback_at_s is None
    assert obj.failed and failure.error and failure.failure_stage == 'pre_step'
    assert_false_qualifications(failure)
    reads = robot.reads
    with pytest.raises(module().AppliedTargetReadbackError, match='latched'):
        begin(obj, clock)
    with pytest.raises(module().AppliedTargetReadbackError, match='latched'):
        obj.finish_step()
    assert robot.reads == reads and obj.current_receipt is failure and not robot.mutations


@pytest.mark.parametrize('requested', [np.zeros(6), np.zeros((1, 7)),
                                       [np.nan] * 7, [np.inf] * 7,
                                       ['.1'] * 7, [complex(.1)] * 7,
                                       [1e300] * 7, [True] * 7])
def test_invalid_requested_command_fails_without_executed_row(requested):
    obj, robot, clock = make()
    with pytest.raises(module().AppliedTargetReadbackError):
        begin(obj, clock, requested=requested)
    assert obj.current_receipt.executed_physics_row is None


@pytest.mark.parametrize('change', ['left_target', 'dtype', 'dof_name', 'dof_order', 'indices', 'action_missing'])
def test_post_step_native_failure_retains_executed_unverified_row(change):
    obj, robot, clock = make()
    pre = begin(obj, clock)
    clock.step()
    if change == 'left_target': robot.action.joint_positions[1] += .01
    elif change == 'dtype': robot.action.joint_positions = robot.action.joint_positions.astype(np.float64)
    elif change == 'dof_name': robot.dof_names[0] = 'different_base'
    elif change == 'dof_order': robot.dof_names[:2] = robot.dof_names[1::-1]
    elif change == 'indices': robot.action.joint_indices = np.arange(9)[::-1]
    elif change == 'action_missing': robot.action = None
    with pytest.raises(module().AppliedTargetReadbackError):
        obj.finish_step()
    result = obj.current_receipt
    assert result is not pre and result.status == 'failed_after_step'
    assert result.pre_read is pre.pre_read and result.observed_clock.physics_row_count == 1
    assert result.executed_physics_row == 0 and result.failure_stage == 'post_step'
    assert not result.verified and result.command_readback_at_s is None


@pytest.mark.parametrize('rows', [0, 2, 3])
def test_missing_or_extra_actual_step_fails_with_actual_count(rows):
    obj, _, clock = make()
    begin(obj, clock)
    clock.step(rows)
    with pytest.raises(module().AppliedTargetReadbackError, match='row'):
        obj.finish_step()
    result = obj.current_receipt
    assert result.observed_clock.physics_row_count == rows
    assert result.executed_physics_row == (0 if rows else None)
    assert result.status == ('failed_after_step' if rows else 'failed_before_step')


@pytest.mark.parametrize('field,delta', [('episode', 2e-9), ('world', 2e-8)])
def test_clock_tolerances_reject_drift(field, delta):
    obj, _, clock = make()
    begin(obj, clock)
    clock.step()
    setattr(clock, field, getattr(clock, field) + delta)
    with pytest.raises(module().AppliedTargetReadbackError, match='clock'):
        obj.finish_step()
    result = obj.current_receipt
    assert result.executed_physics_row == 0 and result.observed_clock.physics_row_count == 1


def test_allowed_clock_noise_is_retained_without_retimestamping():
    obj, _, clock = make()
    begin(obj, clock)
    clock.step()
    clock.episode += 4e-10
    clock.world += 4e-9
    result = obj.finish_step()
    assert result.command_readback_at_s == clock.episode != 1 / 240.
    assert result.post_read.clock.world_time_s == clock.world


@pytest.mark.parametrize('stage', ['pre_step', 'post_step'])
@pytest.mark.parametrize('field', ['rows', 'episode', 'world'])
def test_native_read_is_clock_bracketed(stage, field):
    obj, robot, clock = make()
    if stage == 'post_step':
        begin(obj, clock)
        clock.step()
    def advance():
        setattr(clock, field, getattr(clock, field) + (1 if field == 'rows' else .001))
    robot.hook = advance
    with pytest.raises(module().AppliedTargetReadbackError, match='clock|row'):
        begin(obj, clock) if stage == 'pre_step' else obj.finish_step()
    assert obj.current_receipt.failure_stage == stage and not obj.current_receipt.verified


@pytest.mark.parametrize('field', ['episode', 'world'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_clocks_fail_closed(field, value):
    obj, _, clock = make()
    setattr(clock, field, value)
    with pytest.raises(module().AppliedTargetReadbackError):
        begin(obj, clock)
    assert not obj.current_receipt.verified and obj.failed


@pytest.mark.parametrize('row', [-1, 1, 2, True, 0.])
def test_initial_row_must_be_zero_integer(row):
    obj, _, clock = make()
    with pytest.raises(module().AppliedTargetReadbackError, match='row'):
        begin(obj, clock, row)
    assert obj.current_receipt.executed_physics_row is None


def test_initial_actual_row_cannot_be_nonzero_even_if_clocks_match():
    obj, _, clock = make()
    clock.step()
    with pytest.raises(module().AppliedTargetReadbackError, match='row'):
        begin(obj, clock)


def test_caller_issue_clock_is_checked_against_actual_pre_read():
    obj, _, clock = make()
    issue = module().PhysicsClock(0, 0., 7.)
    clock.world += .1
    with pytest.raises(module().AppliedTargetReadbackError, match='clock'):
        begin(obj, clock, issue=issue)


def test_stale_issue_clock_after_success_does_not_reuse_old_receipt():
    obj, _, clock = make()
    previous = complete(obj, clock)
    with pytest.raises(module().AppliedTargetReadbackError):
        begin(obj, clock, 1, issue=module().PhysicsClock(0, 0., 7.))
    assert obj.current_receipt is not previous and not obj.current_receipt.verified
    assert obj.current_receipt.physics_row == 1 and obj.current_receipt.pre_read is None


def test_overlapping_command_attempt_latches_original_outstanding_row():
    obj, _, clock = make()
    pre = begin(obj, clock)
    with pytest.raises(module().AppliedTargetReadbackError, match='outstanding'):
        begin(obj, clock)
    assert obj.failed and not obj.current_receipt.verified
    assert obj.current_receipt.pre_read is pre.pre_read


def test_finish_without_command_and_duplicate_finish_fail_closed():
    obj, _, _ = make()
    with pytest.raises(module().AppliedTargetReadbackError, match='outstanding'):
        obj.finish_step()
    assert obj.failed and obj.current_receipt is not None
    obj, _, clock = make()
    receipt = complete(obj, clock)
    with pytest.raises(module().AppliedTargetReadbackError, match='outstanding'):
        obj.finish_step()
    assert obj.failed and not obj.current_receipt.verified and obj.current_receipt is not receipt


def test_changed_identity_during_native_read_is_detected():
    obj, robot, clock = make()
    robot.hook = lambda: robot.dof_names.__setitem__(0, 'changed')
    with pytest.raises(module().AppliedTargetReadbackError, match='DOF'):
        begin(obj, clock)


def test_unprintable_native_exception_retains_bounded_safe_failure():
    class Unprintable(RuntimeError):
        def __str__(self): raise RuntimeError('unprintable')
        def __repr__(self): raise RuntimeError('also unprintable')
    obj, robot, clock = make()
    def fail(): raise Unprintable()
    robot.hook = fail
    with pytest.raises(module().AppliedTargetReadbackError, match='Unprintable'):
        begin(obj, clock)
    result = obj.current_receipt
    assert 'message unavailable' in result.error and len(result.error) <= 1024
    assert isinstance(result.error, str) and obj.failed
    assert result.executed_physics_row is None


def test_long_failure_message_is_bounded():
    obj, robot, clock = make()
    def fail(): raise RuntimeError('x' * 10000)
    robot.hook = fail
    with pytest.raises(module().AppliedTargetReadbackError):
        begin(obj, clock)
    assert len(obj.current_receipt.error) <= 1024


def test_failed_post_comparison_preserves_actual_counterexample_vector():
    obj, robot, clock = make()
    begin(obj, clock)
    clock.step()
    robot.action.joint_positions[1] = .9
    with pytest.raises(module().AppliedTargetReadbackError):
        obj.finish_step()
    receipt = obj.current_receipt
    assert receipt.post_read is not None
    assert receipt.post_read.left_targets[0] == np.float32(.9)
    assert receipt.pre_read.left_targets[0] == np.float32(.1)
    assert receipt.command_readback_at_s is None and not receipt.verified


def test_failed_pre_comparison_preserves_actual_native_vector_and_request():
    obj, robot, clock = make()
    robot.action.joint_positions[1] = .9
    with pytest.raises(module().AppliedTargetReadbackError):
        begin(obj, clock)
    receipt = obj.current_receipt
    assert receipt.pre_read is not None
    assert receipt.pre_read.left_targets[0] == np.float32(.9)
    assert receipt.requested_q_backend[0] == np.float32(.1)
    assert receipt.executed_physics_row is None


@pytest.mark.parametrize('field,start,finish', [('episode', 6e-10, -6e-10), ('world', 6e-9, -6e-9)])
def test_individually_valid_clock_offsets_cannot_hide_step_delta_drift(field, start, finish):
    obj, _, clock = make()
    setattr(clock, field, getattr(clock, field) + start)
    begin(obj, clock)
    clock.step()
    setattr(clock, field, getattr(clock, field) - start + finish)
    with pytest.raises(module().AppliedTargetReadbackError, match='advance'):
        obj.finish_step()
    assert obj.current_receipt.executed_physics_row == 0


def test_nonfinite_post_clock_keeps_the_observed_executed_count():
    obj, _, clock = make()
    begin(obj, clock)
    clock.step()
    clock.episode = np.nan
    with pytest.raises(module().AppliedTargetReadbackError):
        obj.finish_step()
    receipt = obj.current_receipt
    assert receipt.observed_physics_row_count == 1 and receipt.executed_physics_row == 0
    assert receipt.status == 'failed_after_step'


def test_original_native_error_survives_secondary_clock_failure():
    obj, robot, clock = make()
    begin(obj, clock)
    clock.step()
    def fail():
        clock.episode = np.nan
        raise RuntimeError('original native read failed')
    robot.hook = fail
    with pytest.raises(module().AppliedTargetReadbackError, match='original native read failed'):
        obj.finish_step()
    receipt = obj.current_receipt
    assert receipt.observed_physics_row_count == 1
    assert receipt.executed_physics_row == 0


def test_missing_action_indices_is_not_implicitly_assumed_canonical():
    obj, robot, clock = make()
    del robot.action.joint_indices
    with pytest.raises(module().AppliedTargetReadbackError):
        begin(obj, clock)
    assert obj.failed


def test_request_is_copied_before_native_api_can_reuse_caller_buffer():
    obj, robot, clock = make()
    request = REQUESTED.copy()
    robot.hook = lambda: request.fill(0)
    begin(obj, clock, requested=request)
    assert np.all(request == 0)
    np.testing.assert_array_equal(obj.current_receipt.requested_q_backend, REQUESTED.astype(np.float32))


def test_reentrant_begin_cannot_rehabilitate_failed_acquisition():
    obj, robot, clock = make()
    def reenter():
        try:
            begin(obj, clock)
        except module().AppliedTargetReadbackError:
            pass
    robot.hook = reenter
    with pytest.raises(module().AppliedTargetReadbackError, match='outstanding'):
        begin(obj, clock)
    assert obj.failed and not obj.current_receipt.verified
    assert robot.reads == 1


@pytest.mark.parametrize('row', [0, 1])
def test_post_read_clock_regression_cannot_erase_established_execution(row):
    obj, robot, clock = make()
    if row:
        complete(obj, clock)
    begin(obj, clock, row)
    clock.step()
    def regress():
        clock.rows = row
        clock.episode = row / 240.
        clock.world = 7. + clock.episode
    robot.hook = regress
    with pytest.raises(module().AppliedTargetReadbackError, match='row'):
        obj.finish_step()
    receipt = obj.current_receipt
    assert receipt.status == 'failed_after_step'
    assert receipt.executed_physics_row == row
    assert receipt.execution_observed_physics_row_count == row + 1
    assert receipt.observed_physics_row_count == row
    assert receipt.observed_clock.physics_row_count == row
    assert not receipt.verified and receipt.command_readback_at_s is None
    assert receipt.failure_stage == 'post_step'


@pytest.mark.parametrize('swallow_nested_error', [False, True])
def test_reentrant_begin_during_post_read_retains_executed_row(swallow_nested_error):
    obj, robot, clock = make()
    begin(obj, clock)
    clock.step()
    def reenter():
        if not swallow_nested_error:
            begin(obj, clock, row=1)
        else:
            with pytest.raises(module().AppliedTargetReadbackError, match='outstanding'):
                begin(obj, clock, row=1)
    robot.hook = reenter
    with pytest.raises(module().AppliedTargetReadbackError, match='outstanding'):
        obj.finish_step()
    receipt = obj.current_receipt
    assert receipt.status == 'failed_after_step'
    assert receipt.executed_physics_row == 0 and receipt.physics_row == 0
    assert receipt.execution_observed_physics_row_count == 1
    assert receipt.observed_physics_row_count == 1
    assert receipt.failure_stage == 'pre_step'  # Error operation and execution are independent.
    assert not receipt.verified and receipt.command_readback_at_s is None
    reads = robot.reads
    with pytest.raises(module().AppliedTargetReadbackError, match='latched'):
        obj.finish_step()
    assert obj.current_receipt is receipt and robot.reads == reads


def test_new_pre_step_failure_cannot_borrow_prior_execution_evidence():
    obj, robot, clock = make()
    complete(obj, clock)
    robot.action = None
    with pytest.raises(module().AppliedTargetReadbackError):
        begin(obj, clock, row=1)
    receipt = obj.current_receipt
    assert receipt.status == 'failed_before_step'
    assert receipt.executed_physics_row is None
    assert receipt.execution_observed_physics_row_count is None
