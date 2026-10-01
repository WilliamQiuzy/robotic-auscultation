"""Bounded, read-only position-target evidence; no controller integration.

The caller applies one command, calls ``begin_command`` with its explicit next
zero-based physics row and issue clock, executes exactly one original 1/240 s
physics step, then calls ``finish_step`` before applying another command. The
required clock provider must read the actual episode/world clocks and completed
native row count on each call, returning PhysicsClock. A supplied issue clock is
checked against these observations; it is never used as readback evidence.
The explicit fixed world offset permits setup steps without inventing a reset.
Episode checks use compact-v4's 1e-9 s tolerance; native world checks preserve
m2_capture's 1e-8 s tolerance. Raw clock values are never rounded or corrected.

The articulation protocol exposes ``dof_names`` and ``get_applied_action()``.
The optional dof_names_provider is a read-only callable for wrappers that expose
actual articulation names elsewhere; it must reread those names on each call.
Native targets must be a full NumPy float32/float64 vector; joint_indices must
be None or the exact full canonical range. The installed Isaac 5.1 source path
SingleArticulation -> ArticulationController -> view.get_applied_actions reads
physics-view DOF position targets with a valid handle. This component always
calls that public robot API; it never accepts a requested-command cache or
measured joint positions as a substitute. Native handle/API qualification must
still be established by a future adapter and native execution, not CPU fakes.

Both reads are bracketed by fresh clock samples and DOF-name checks. Receipts
retain full native vectors, backend dtype, named left mapping, request expressed
in backend dtype, and distinct issue/pre/post clocks. Arrays are immutable owned
copies. Only a verified post-step receipt exposes command_readback_at_s for the
existing compact writer. A readback proves stored targets, not achieved joint
positions, force, contact, safety or training eligibility.

Any failure latches permanently. Only the current receipt and bounded safe error
text are retained, without exception objects or an accumulating trace. A failed
pre-step attempt creates no executed-row evidence. A failed post-step attempt
retains the observed actual count and the attempted row as executed/unverified
only when that count establishes execution; missing/extra steps remain failures.
The first post-attempt count establishing execution is retained separately from
the latest observed count: a later counter regression or reentrant call cannot
erase executed-row evidence. Error stage identifies the failing operation, not
whether the outstanding row has already executed.
The caller must stop on a failure and preserve this receipt. Calls are synchronous;
reentrant/overlapping attempts fail. No native resources are owned or released.
"""
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Callable, Optional, Protocol, Sequence

import numpy as np


PHYSICS_DT_S = 1. / 240.
EPISODE_CLOCK_TOLERANCE_S = 1e-9
WORLD_CLOCK_TOLERANCE_S = 1e-8
LEFT_DOF_NAMES = tuple(f'openarm_left_joint{i}' for i in range(1, 8))
SOURCE_API = 'robot.get_applied_action().joint_positions'


class ReadOnlyArticulation(Protocol):
    @property
    def dof_names(self) -> Sequence[str]: ...

    def get_applied_action(self): ...


class AppliedTargetReadbackError(RuntimeError):
    """Readback failed; preserve current_receipt and stop acquisition."""


@dataclass(frozen=True)
class PhysicsClock:
    """One actual clock observation, with the number of completed physics rows."""
    physics_row_count: int
    episode_time_s: float
    world_time_s: float


@dataclass(frozen=True)
class PositionTargetRead:
    before_clock: PhysicsClock
    clock: PhysicsClock
    full_targets: np.ndarray
    left_targets: np.ndarray
    dof_names: tuple
    left_indices: tuple
    source_api: str = field(default=SOURCE_API, init=False)


@dataclass(frozen=True)
class AppliedPositionReceipt:
    physics_row: Optional[int]
    status: str
    issued_clock: Optional[PhysicsClock] = None
    requested_q_backend: Optional[np.ndarray] = None
    pre_read: Optional[PositionTargetRead] = None
    post_read: Optional[PositionTargetRead] = None
    observed_clock: Optional[PhysicsClock] = None
    observed_physics_row_count: Optional[int] = None
    execution_observed_physics_row_count: Optional[int] = None
    executed_physics_row: Optional[int] = None
    failure_stage: Optional[str] = None
    error: Optional[str] = None
    training_eligible: bool = field(default=False, init=False)
    native_qualified: bool = field(default=False, init=False)
    physical_qualification: bool = field(default=False, init=False)

    @property
    def verified(self):
        return self.status == 'post_step_verified'

    @property
    def command_issued_at_s(self):
        return None if self.issued_clock is None else self.issued_clock.episode_time_s

    @property
    def command_pre_readback_at_s(self):
        return None if self.pre_read is None else self.pre_read.clock.episode_time_s

    @property
    def command_readback_at_s(self):
        return self.post_read.clock.episode_time_s if self.verified else None


def _scalar(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(label + ' must be a finite real clock')
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(label + ' must be a finite real clock')
    return result


def _clock_copy(value):
    if not isinstance(value, PhysicsClock):
        raise ValueError('clock provider and command issue require PhysicsClock')
    if type(value.physics_row_count) is not int or value.physics_row_count < 0:
        raise ValueError('actual physics row count must be a nonnegative integer')
    return PhysicsClock(value.physics_row_count,
                        _scalar(value.episode_time_s, 'episode clock'),
                        _scalar(value.world_time_s, 'world clock'))


def _owned(array):
    # The immutable bytes owner prevents callers from re-enabling WRITEABLE.
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)


def _safe_error(error):
    try:
        message = str(error)
    except BaseException:
        message = '<exception message unavailable>'
    return (type(error).__name__ + ': ' + message)[:1024]


class AppliedPositionTargets:
    def __init__(self, robot: ReadOnlyArticulation, left_indices, *,
                 clock_provider: Callable[[], PhysicsClock], world_time_offset_s,
                 dof_names_provider: Optional[Callable[[], Sequence[str]]] = None):
        if not callable(clock_provider) or not callable(getattr(robot, 'get_applied_action', None)):
            raise ValueError('read-only articulation and actual clock provider required')
        if dof_names_provider is not None and not callable(dof_names_provider):
            raise ValueError('DOF names provider must be a read-only callable')
        self._robot = robot
        self._clock_provider = clock_provider
        self._names_provider = dof_names_provider or (lambda: robot.dof_names)
        self._world_offset = _scalar(world_time_offset_s, 'fixed world clock offset')
        self._dof_names = self._read_names()
        try:
            indices = tuple(left_indices)
        except TypeError:
            raise ValueError('seven unique named left DOF indices required') from None
        if (len(indices) != 7 or any(isinstance(i, (bool, np.bool_)) or
                not isinstance(i, (int, np.integer)) for i in indices)):
            raise ValueError('seven unique integer left DOF indices required')
        indices = tuple(int(i) for i in indices)
        if (len(set(indices)) != 7 or any(i < 0 or i >= len(self._dof_names) for i in indices) or
                tuple(self._dof_names[i] for i in indices) != LEFT_DOF_NAMES):
            raise ValueError('left DOF indices must map exactly to openarm_left_joint1..7 in order')
        self._left_indices = indices
        self._expected_row = 0
        self._pending = False
        self._busy = False
        self._failed = False
        self._current = None
        self._observed_clock = None
        self._observed_rows = None
        self._execution_observed_rows = None
        self._post_attempt = False

    @property
    def current_receipt(self):
        return self._current

    @property
    def failed(self):
        return self._failed

    def _read_names(self):
        values = self._names_provider()
        if isinstance(values, (str, bytes)):
            raise ValueError('full unique actual DOF names required')
        names = tuple(values)
        if (not names or any(type(name) is not str or not name for name in names) or
                len(set(names)) != len(names)):
            raise ValueError('full unique actual DOF names required')
        return names

    def _check_identity(self):
        if self._read_names() != self._dof_names:
            raise ValueError('actual DOF identity or order changed')

    def _observe(self):
        raw = self._clock_provider()
        # Preserve a valid actual count even if its associated clock is invalid.
        if isinstance(raw, PhysicsClock) and type(raw.physics_row_count) is int and raw.physics_row_count >= 0:
            self._observed_rows = raw.physics_row_count
            # Execution belongs to the outstanding post-step attempt, not the
            # latest sample or the operation that ultimately triggers failure.
            if (self._post_attempt and not self._failed and self._execution_observed_rows is None and
                    raw.physics_row_count > self._current.physics_row):
                self._execution_observed_rows = raw.physics_row_count
        observed = _clock_copy(raw)
        self._observed_clock = observed
        return observed

    def _check_clock(self, clock, expected_rows):
        if clock.physics_row_count != expected_rows:
            raise ValueError('actual physics row count must match the consecutive requested row')
        if abs(clock.episode_time_s - expected_rows * PHYSICS_DT_S) > EPISODE_CLOCK_TOLERANCE_S:
            raise ValueError('episode clock must match actual consecutive 240 Hz row time')
        if abs(clock.world_time_s - self._world_offset - clock.episode_time_s) > WORLD_CLOCK_TOLERANCE_S:
            raise ValueError('world clock must retain the explicit fixed episode offset')

    @staticmethod
    def _same_clock(before, after):
        if before.physics_row_count != after.physics_row_count:
            raise ValueError('native target read changed actual physics row count')
        if (abs(after.episode_time_s - before.episode_time_s) > EPISODE_CLOCK_TOLERANCE_S or
                abs(after.world_time_s - before.world_time_s) > WORLD_CLOCK_TOLERANCE_S):
            raise ValueError('native target read advanced episode/world clock')

    def _native_targets(self):
        self._check_identity()
        action = self._robot.get_applied_action()
        if action is None:
            raise ValueError('missing native applied action; valid physics handle required')
        targets = getattr(action, 'joint_positions', None)
        if (not isinstance(targets, np.ndarray) or targets.dtype not in (np.dtype('float32'), np.dtype('float64')) or
                targets.shape != (len(self._dof_names),) or not np.isfinite(targets).all()):
            raise ValueError('native targets must be finite full-DOF NumPy float32/float64 positions')
        indices = action.joint_indices
        if indices is not None:
            index_array = np.asarray(indices)
            if (index_array.shape != (len(self._dof_names),) or index_array.dtype.kind not in 'iu' or
                    not np.array_equal(index_array, np.arange(len(self._dof_names)))):
                raise ValueError('native action indices must be None or the exact canonical full DOF range')
        full = _owned(targets)
        left = _owned(full[list(self._left_indices)])
        self._check_identity()
        return full, left

    def _read(self, expected_rows, anchor=None):
        before = self._observe()
        self._check_clock(before, expected_rows)
        if anchor is not None:
            self._same_clock(anchor, before)
        original_error = None
        try:
            full, left = self._native_targets()
        except BaseException as error:
            original_error = _safe_error(error)
        # Read after the API even if it failed. A secondary clock failure cannot
        # replace the original native error or erase an observed executed count.
        try:
            after = self._observe()
            self._same_clock(before, after)
            self._check_clock(after, expected_rows)
            self._check_identity()
        except BaseException as error:
            if original_error is None:
                original_error = _safe_error(error)
        if original_error is not None:
            raise AppliedTargetReadbackError(original_error)
        if self._failed:
            raise AppliedTargetReadbackError('readback failure is latched')
        return PositionTargetRead(before, after, full, left, self._dof_names, self._left_indices)

    def _reject_latched(self):
        if self._failed:
            raise AppliedTargetReadbackError('readback failure is latched; preserve current_receipt')

    def _fail(self, stage, error):
        # An overlapping/reentrant failure wins; outer calls cannot rehabilitate
        # or overwrite it. Receipts never hold exception/traceback objects.
        if not self._failed:
            self._failed = True
            receipt = self._current or AppliedPositionReceipt(None, 'attempting')
            executed = receipt.physics_row if self._execution_observed_rows is not None else None
            self._current = replace(receipt,
                status='failed_after_step' if executed is not None else 'failed_before_step',
                observed_clock=self._observed_clock, observed_physics_row_count=self._observed_rows,
                execution_observed_physics_row_count=self._execution_observed_rows,
                executed_physics_row=executed, failure_stage=stage, error=_safe_error(error))
        self._pending = False
        raise AppliedTargetReadbackError(self._current.error) from None

    def begin_command(self, *, next_row_index, requested_q, issued_clock):
        """Read actual targets immediately after caller apply_action, before step."""
        self._reject_latched()
        if self._pending or self._busy:
            self._fail('pre_step', ValueError('one outstanding command is allowed'))
        self._observed_clock = None
        self._observed_rows = None
        self._execution_observed_rows = None
        row = next_row_index if type(next_row_index) is int else None
        self._current = AppliedPositionReceipt(row, 'attempting')
        self._busy = True
        try:
            if type(next_row_index) is not int or next_row_index != self._expected_row:
                raise ValueError('next physics row must start at zero and be consecutive')
            issue = _clock_copy(issued_clock)
            self._check_clock(issue, next_row_index)
            self._current = replace(self._current, issued_clock=issue)
            requested = np.asarray(requested_q)
            if requested.shape != (7,) or requested.dtype.kind not in 'fiu' or not np.isfinite(requested).all():
                raise ValueError('requested command must contain seven finite real positions')
            requested = requested.copy()
            pre = self._read(next_row_index, anchor=issue)
            self._current = replace(self._current, pre_read=pre)
            with np.errstate(over='ignore', invalid='ignore'):
                expressed = _owned(requested.astype(pre.full_targets.dtype))
            if not np.isfinite(expressed).all():
                raise ValueError('requested command is nonfinite in backend dtype')
            self._current = replace(self._current, requested_q_backend=expressed)
            if not np.array_equal(expressed, pre.left_targets):
                raise ValueError('native left targets do not exactly match requested command in backend dtype')
            self._current = replace(self._current, status='pre_step_verified', requested_q_backend=expressed,
                                    pre_read=pre, observed_clock=pre.clock,
                                    observed_physics_row_count=pre.clock.physics_row_count)
            self._pending = True
            return self._current
        except BaseException as error:
            self._fail('pre_step', error)
        finally:
            self._busy = False

    def finish_step(self):
        """Read again after exactly one actual physics step, before next command."""
        self._reject_latched()
        if not self._pending or self._busy:
            if not self._pending and not self._busy:
                self._current = AppliedPositionReceipt(self._expected_row, 'attempting')
                self._observed_clock = None
                self._observed_rows = None
                self._execution_observed_rows = None
            self._fail('post_step', ValueError('an outstanding pre-step command is required'))
        self._busy = True
        self._post_attempt = True
        try:
            pre = self._current.pre_read
            post = self._read(self._current.physics_row + 1)
            self._current = replace(self._current, post_read=post)
            if (abs(post.clock.episode_time_s - pre.clock.episode_time_s - PHYSICS_DT_S) > EPISODE_CLOCK_TOLERANCE_S or
                    abs(post.clock.world_time_s - pre.clock.world_time_s - PHYSICS_DT_S) > WORLD_CLOCK_TOLERANCE_S):
                raise ValueError('episode/world clock must advance by exactly one original physics step')
            if post.full_targets.dtype != pre.full_targets.dtype:
                raise ValueError('native target dtype changed between reads')
            if not np.array_equal(post.left_targets, pre.left_targets):
                raise ValueError('native left targets changed between pre-step and post-step reads')
            self._current = replace(self._current, status='post_step_verified', post_read=post,
                                    observed_clock=post.clock, observed_physics_row_count=post.clock.physics_row_count,
                                    execution_observed_physics_row_count=self._execution_observed_rows,
                                    executed_physics_row=self._current.physics_row)
            self._expected_row += 1
            self._pending = False
            return self._current
        except BaseException as error:
            self._fail('post_step', error)
        finally:
            self._busy = False
            self._post_attempt = False
