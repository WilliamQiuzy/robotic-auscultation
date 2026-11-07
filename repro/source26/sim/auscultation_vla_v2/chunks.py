"""Per-command proposal gate. A stop disposition is NOT physical unloading.

Action: [dx,dy,dz] meters in robot-base axes, [rx,ry,rz] base-axis
rotation-vector radians (left composition), desired normal contact force N.
The downstream executor composes increments with fresh measured TCP pose and
must independently check IK, collision, tracking and continuous dynamics.
No native 32-dimensional pretrained output is executable through this gate.
"""
from dataclasses import dataclass
import copy
import math
import numpy as np
from .action_contract import PHASES,bounded

ACTION_FEATURES=('dx_base_m','dy_base_m','dz_base_m','rx_base_rad','ry_base_rad','rz_base_rad','normal_force_reference_N')


@dataclass
class Chunk:
    sequence: int
    observed_at_s: float
    ready_at_s: float
    target_revision: int
    phase: str
    action_period_s: float
    actions: np.ndarray
    execution_prefix: int


@dataclass(frozen=True)
class Disposition:
    status: str
    reason: str
    action: object = None


class ChunkGate:
    def __init__(self, *, max_observation_age_s=.15, max_prefix_s=.1, max_feedback_age_s=.05):
        values=[max_observation_age_s,max_prefix_s,max_feedback_age_s]
        if not np.isfinite(values).all() or min(values)<=0: raise ValueError('invalid gate limits')
        self.max_age,self.max_prefix,self.max_feedback_age=values
        self._chunk=None; self._sequence=-1; self._last_observed=-math.inf
        self._last_time=-math.inf; self._start=0.; self._index=-1

    def _clear(self,status,reason):
        self._chunk=None
        return Disposition(status,reason)

    def _clock(self,now):
        if not np.isfinite(now) or now<0 or now<self._last_time-1e-9: return False
        self._last_time=now
        return True

    def accept(self, chunk, *, now_s, target_revision, phase):
        if not self._clock(now_s): return self._clear('rejected','invalid_clock')
        if not isinstance(chunk,Chunk): return self._clear('rejected','invalid_chunk')
        if type(chunk.sequence) is not int or chunk.sequence<0: return self._clear('rejected','invalid_sequence')
        if chunk.sequence<=self._sequence: return self._clear('rejected','old_sequence')
        self._sequence=chunk.sequence
        try:
            times=np.asarray([chunk.observed_at_s,chunk.ready_at_s,chunk.action_period_s],float)
            a=np.asarray(chunk.actions,float)
            valid=(np.isfinite(times).all() and 0<=chunk.observed_at_s<=chunk.ready_at_s<=now_s+1e-9
                and abs(chunk.action_period_s-.05)<1e-9 and type(chunk.execution_prefix) is int
                and a.ndim==2 and a.shape[1]==7 and 0<chunk.execution_prefix<=len(a)
                and np.isfinite(a).all() and chunk.execution_prefix*chunk.action_period_s<=self.max_prefix+1e-9
                and type(chunk.target_revision) is int and chunk.target_revision>=0
                and chunk.phase in PHASES and phase in PHASES
                and type(target_revision) is int and target_revision>=0)
        except (TypeError,ValueError): valid=False
        if not valid:return self._clear('rejected','invalid_chunk')
        if chunk.target_revision!=target_revision:return self._clear('rejected','target_changed')
        if chunk.phase!=phase:return self._clear('rejected','phase_changed')
        if now_s-chunk.observed_at_s>self.max_age+1e-9:return self._clear('rejected','stale_observation')
        if chunk.observed_at_s<self._last_observed-1e-9:return self._clear('rejected','older_observation')
        # Requested pose offsets include servo lag; they are not velocities.
        if not bounded(a,phase):
            return self._clear('rejected','action_bounds')
        self._last_observed=chunk.observed_at_s; self._chunk=copy.deepcopy(chunk)
        self._chunk.actions=a.copy(); self._start=now_s; self._index=-1
        return Disposition('accepted','bounded_proposal')

    def next(self, *, now_s, target_revision, phase, feedback_time_s, force_safe=None, visible=None):
        if not self._clock(now_s):return self._clear('stop','invalid_clock')
        if not np.isfinite(feedback_time_s) or feedback_time_s<0 or feedback_time_s>now_s+1e-9 or now_s-feedback_time_s>self.max_feedback_age+1e-9:
            return self._clear('stop','stale_feedback')
        if type(force_safe) is not bool or not force_safe:return self._clear('stop','force_unsafe')
        if type(visible) is not bool or not visible:return self._clear('stop','target_lost')
        chunk=self._chunk
        if chunk is None:return Disposition('replan','no_chunk')
        if type(target_revision) is not int or target_revision!=chunk.target_revision:return self._clear('stop','target_changed')
        if phase!=chunk.phase:return self._clear('stop','phase_changed')
        if now_s-chunk.observed_at_s>self.max_age+1e-9:return self._clear('stop','stale_observation')
        index=int(math.floor((now_s-self._start+1e-9)/chunk.action_period_s))
        if index>=chunk.execution_prefix:return self._clear('replan','prefix_exhausted')
        if index>self._index+1:return self._clear('stop','missed_command_deadline')
        if index<=self._index:return Disposition('wait','not_next_command_time')
        self._index=index
        return Disposition('action','proposal_requires_executor_checks',chunk.actions[index].copy())
