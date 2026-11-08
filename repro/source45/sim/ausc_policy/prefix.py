"""Pure NumPy H10/E5 absolute-joint proposal queue for M0 simulation.

At simulation time t, install one decoded absolute [10, 7] chunk. Inference
must leave simulation at t. At t+k*0.05, pop proposal k (k=0..4) whose label
belongs to t+(k+1)*0.05. The caller feeds that target into the SAME external
JointTrajectory and advances twelve 240 Hz ticks without resetting q/v/a.
This module neither constructs Ruckig nor shapes, clamps, or executes actions.
Relative predictions must already have had the block-origin anchor added once.

Age limits retain the original contracts: observation .15 s and Hall .05 s
from pi05_v6/task_observation.py; feedback .05 s from vla_v2/chunks.py. They
apply to NEW safety observations at every pop, independently of plan age.
Call sync_context on context events and clear on external fault/reset events;
pop also checks context and safety even before its next dispatch time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HORIZON = 10
EXECUTION_PREFIX = 5
PERIOD_S = .05
MAX_OBSERVATION_AGE_S = .15
MAX_FEEDBACK_AGE_S = .05
MAX_HALL_AGE_S = .05
TIME_TOLERANCE_S = 1e-7  # numerical roundoff, far below one 240 Hz tick


@dataclass(frozen=True)
class PrefixContext:
    episode_id: str
    target: str
    revision: int
    phase: str
    control_owner: str


@dataclass(frozen=True)
class PrefixResult:
    status: str
    reason: str
    proposal: object = None
    index: object = None
    execute_at_s: object = None


def _valid_time(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)) and np.isfinite(value) and value >= 0)


def _valid_context(context):
    return (isinstance(context, PrefixContext)
            and all(isinstance(getattr(context, name), str) and getattr(context, name).strip()
                    for name in ('episode_id', 'target', 'phase', 'control_owner'))
            and type(context.revision) is int and context.revision >= 0)


class ActionPrefix:
    """One proposal queue; explicit context changes are replans, not faults.

    A returned ``action`` is only a proposal. All existing collision, joint,
    force, tracking and continuous-dynamics gates remain in the caller. No
    status here authorizes physical robot motion or certifies closed-loop success.
    """
    def __init__(self):
        self._context = None
        self._actions = None
        self._index = 0
        self._observed_at = None
        self._last_installed_at = None
        self._last_label_time = None
        self._last_now = None
        self._idle_status = 'replan'
        self.last_clear_reason = 'no_prefix'

    @property
    def remaining(self):
        return 0 if self._actions is None else EXECUTION_PREFIX-self._index

    def _discard(self, status, reason):
        self._actions = None
        self._index = 0
        self._idle_status = status
        self.last_clear_reason = reason
        return PrefixResult(status, reason)

    def clear(self, reason):
        """Explicit external invalidation; preserves the supplied reason."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('A nonempty clear reason is required')
        return self._discard('replan', reason)

    def sync_context(self, context):
        """Call immediately on any context event, including between pop calls."""
        if not _valid_context(context):
            return self._discard('stop', 'invalid_context')
        previous = self._context
        self._context = context
        if previous is None:
            return PrefixResult('ready', 'context_bound')
        changed = next((name for name in ('episode_id', 'target', 'revision', 'control_owner', 'phase')
                        if getattr(previous, name) != getattr(context, name)), None)
        if changed is None:
            return PrefixResult('ready', 'context_unchanged')
        if context.episode_id != previous.episode_id:
            self._last_now = None  # simulation time may reset only at an episode boundary
        self._last_installed_at = None
        self._last_label_time = None
        reason = 'episode_changed' if changed == 'episode_id' else changed+'_changed'
        return self._discard('replan', reason)

    def _clock(self, now):
        if not _valid_time(now):
            return self._discard('stop', 'invalid_clock')
        if self._last_now is not None and now < self._last_now-TIME_TOLERANCE_S:
            return self._discard('stop', 'clock_went_backwards')
        self._last_now = max(float(now), self._last_now if self._last_now is not None else float(now))
        return None

    @staticmethod
    def _age_reason(name, stamp, now, maximum):
        if not _valid_time(stamp):
            return 'invalid_'+name+'_time'
        if stamp > now+TIME_TOLERANCE_S:
            return 'future_'+name
        if now-stamp > maximum+TIME_TOLERANCE_S:
            return 'stale_'+name
        return None

    def install(self, absolute_chunk, *, observed_at_s, now_s, context):
        """Copy a complete H10 absolute-radian chunk exactly once; no padding."""
        event = self.sync_context(context)
        if event.status == 'stop':
            return event
        clock = self._clock(now_s)
        if clock is not None:
            return clock
        if self._actions is not None:
            return self._discard('rejected', 'prefix_not_consumed')
        try:
            raw = np.asarray(absolute_chunk)
            valid = raw.shape == (HORIZON, 7) and raw.dtype.kind in 'fiu' and np.isfinite(raw).all()
            actions = raw.astype(np.float64) if valid else None
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid or not np.isfinite(actions).all():
            return self._discard('rejected', 'invalid_absolute_chunk')
        reason = self._age_reason('observation', observed_at_s, now_s, MAX_OBSERVATION_AGE_S)
        if reason:
            return self._discard('rejected', reason)
        if abs(now_s-observed_at_s) > TIME_TOLERANCE_S:
            return self._discard('rejected', 'plan_clock_advanced_during_inference')
        if self._last_installed_at is not None and observed_at_s <= self._last_installed_at+TIME_TOLERANCE_S:
            return self._discard('rejected', 'observation_already_used')
        if self._last_label_time is not None and now_s < self._last_label_time-TIME_TOLERANCE_S:
            return self._discard('rejected', 'previous_action_interval_active')
        if context.phase not in ('FREE', 'APPROACH'):
            return self._discard('rejected', 'phase_not_l2')
        self._actions = actions.copy()
        self._index = 0
        self._observed_at = float(observed_at_s)
        self._last_installed_at = float(observed_at_s)
        return PrefixResult('installed', 'absolute_H10_E5_prefix')

    def pop(self, *, now_s, context, observation_time_s, feedback_time_s, hall_time_s,
            target_visible, safety_ok):
        """Return the next proposal at its interval start, with its label time.

        The original plan observation is NOT reused for the current safety-age
        check. These three timestamps must describe current acquired feedback.
        Wall-clock inference duration is deliberately outside this sim clock API.
        """
        event = self.sync_context(context)
        if event.status == 'stop':
            return event
        clock = self._clock(now_s)
        if clock is not None:
            return clock
        if event.status == 'replan':
            return event
        for name, stamp, maximum in (
                ('observation', observation_time_s, MAX_OBSERVATION_AGE_S),
                ('feedback', feedback_time_s, MAX_FEEDBACK_AGE_S),
                ('hall', hall_time_s, MAX_HALL_AGE_S)):
            reason = self._age_reason(name, stamp, now_s, maximum)
            if reason:
                return self._discard('stop', reason)
        if type(target_visible) is not bool or not target_visible:
            return self._discard('stop', 'target_lost')
        if type(safety_ok) is not bool or not safety_ok:
            return self._discard('stop', 'safety_unsafe')
        if self._actions is None:
            return PrefixResult(self._idle_status, self.last_clear_reason)
        dispatch_at = self._observed_at+self._index*PERIOD_S
        if now_s < dispatch_at-TIME_TOLERANCE_S:
            return PrefixResult('wait', 'not_next_command_time')
        if now_s > dispatch_at+TIME_TOLERANCE_S:
            return self._discard('stop', 'missed_command_deadline')
        index = self._index
        proposal = self._actions[index].copy()
        execute_at = self._observed_at+(index+1)*PERIOD_S
        self._last_label_time = execute_at
        self._index += 1
        if self._index == EXECUTION_PREFIX:
            self._discard('replan', 'prefix_exhausted')
        return PrefixResult('action', 'proposal_requires_executor_checks', proposal, index, execute_at)
