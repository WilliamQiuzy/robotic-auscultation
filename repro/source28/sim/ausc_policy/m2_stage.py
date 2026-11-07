"""Causal actor supervision, separate from the unchanged analytic L3 manager.

Execution role controls eligibility; it never assigns FREE from a route name.
The shared stage rule uses only measured q and the already acquired estimated
target (state17 translation). A target-local latch switches below 9 cm.
"""
from copy import deepcopy
import numpy as np

RULE_ID = 'm2_estimated_tcp_distance_latch_9cm_v1'
NATIVE_SOURCE = 'native_measured_q_estimated_target_v1'
HISTORICAL_SOURCE = 'historical_cpu_causal_20hz_zoh_v1'
APPROACH_DISTANCE_M = .09
L2_EXECUTION_ROLES = frozenset(('transfer', 'lower_approach'))
ACTOR_FIELDS = {'phase','owner','sampled_at_s','rule_id','source','estimated_tcp_distance_m'}


class CausalStage:
    def __init__(self):
        self.phase='FREE'
        self.last_time=None

    def update(self, distance_m, sampled_at_s):
        values=np.asarray([distance_m,sampled_at_s])
        if (values.dtype.kind not in 'fiu' or not np.isfinite(values).all() or
                distance_m<0 or sampled_at_s<0 or
                (self.last_time is not None and sampled_at_s<self.last_time-1e-9)):
            raise ValueError('Invalid causal stage distance/clock')
        if self.phase=='FREE' and distance_m<APPROACH_DISTANCE_M:
            self.phase='APPROACH'
        self.last_time=float(sampled_at_s)
        return self.phase


def validate_actor(value, sampled_at_s, *, source=NATIVE_SOURCE, reference_sha256=None):
    from .m2_given_contact_pose import NATIVE_SOURCE as GIVEN_SOURCE, STAGE_RULE
    given = source == GIVEN_SOURCE
    fields = ACTOR_FIELDS | {'reference_sha256','reference_revision','role'} if given else ACTOR_FIELDS
    if not isinstance(value,dict) or set(value)!=fields:
        raise ValueError('Missing explicit native actor phase/owner event')
    distance=value['estimated_tcp_distance_m']
    stamp=value['sampled_at_s']
    if (value['rule_id']!=(STAGE_RULE if given else RULE_ID) or value['source']!=source or
            value['owner'] not in ('L2','L3','PREPARATION') or
            value['phase'] not in ('FREE','APPROACH',None) or
            (value['owner']=='L2' and value['phase'] is None) or
            not isinstance(stamp,(int,float)) or not np.isfinite(stamp) or
            abs(stamp-sampled_at_s)>1e-9 or
            not isinstance(distance,(int,float)) or not np.isfinite(distance) or distance<0):
        raise ValueError('Invalid native actor stage provenance/clock')
    if given and (value['reference_sha256'] != reference_sha256 or value['reference_revision'] != 0 or
                  value['role'] not in ('L2','L2_PARK','L3_ALIGN','L3','PREPARATION') or
                  (value['owner']=='L2' and value['role']!='L2') or
                  (value['role']=='L3_ALIGN' and value['owner']!='L3')):
        raise ValueError('Invalid given-reference actor revision/owner')
    return value


class CausalHistory:
    """Two real state17 observations on one 20 Hz grid; never pad a reset."""
    def __init__(self, anchor_s):
        self.anchor=float(anchor_s)
        self.context=None
        self.frames=[]
        self.last_time=None

    def observe(self, state17, stamp, context):
        state=np.asarray(state17,dtype=float)
        if state.shape!=(17,) or not np.isfinite(state).all() or not np.isfinite(stamp):
            raise ValueError('Finite state17 history required')
        if self.last_time is not None and stamp<self.last_time-1e-8:
            raise ValueError('History clock moved backwards')
        self.last_time=float(stamp)
        if context!=self.context:
            self.frames=[]
            self.context=deepcopy(context)
        grid=(stamp-self.anchor)/.05
        if grid< -1e-8 or abs(grid-round(grid))>1e-7:
            return
        if self.frames and abs(stamp-self.frames[-1][0])<1e-8:
            if not np.array_equal(self.frames[-1][1],state):
                raise ValueError('History state changed at unchanged clock')
            return
        if self.frames and abs(stamp-self.frames[-1][0]-.05)>1e-8:
            self.frames=[]
        self.frames.append((float(stamp),state.copy()))
        self.frames=self.frames[-2:]

    def snapshot(self):
        if len(self.frames)!=2:
            raise ValueError('Two real same-context history observations required; no padding')
        return np.stack([row[1] for row in self.frames]), np.array([row[0] for row in self.frames])


def historical_contexts(records, native_audit, metadata):
    """Diagnostic only: old data has no native q/actor phase.

    Reconstruct the named rule at recorded 20 Hz observation times and hold
    the latest *past* phase between them. The before side at an observation
    tick still uses the preceding observation. This is no native actor record.
    """
    result=deepcopy(native_audit)
    tracker=CausalStage()
    phases=[]
    for row in records:
        t=float(row['observed_at_s'])
        from .m2_contract import encode_state
        state=encode_state(row['measured_q'],row['estimated_target_base'],
                           row['target_confidence'],metadata=metadata)
        d=float(np.linalg.norm(state[7:10]))
        phases.append((t,tracker.update(d,t),d))
    observation_times=np.asarray([row[0] for row in phases])
    for event in result:
        for side,stamp in [('controller_before',event['time_s']-1/240),
                           ('controller_after',event['time_s'])]:
            past=int(np.searchsorted(observation_times,stamp+1e-10,side='right'))-1
            context=event[side]
            eligible=(past>=0 and context.get('subphase') in L2_EXECUTION_ROLES and
                      context.get('phase')=='APPROACH' and event.get('control_owner')=='L2' and
                      context.get('active') is False and context.get('recovery') is False)
            phase=phases[past][1] if past>=0 else None
            context['actor']=dict(phase=phase if eligible else None, owner='L2' if eligible else 'PREPARATION',
                sampled_at_s=stamp,rule_id=RULE_ID,source=HISTORICAL_SOURCE,
                estimated_tcp_distance_m=phases[past][2] if past>=0 else 0.)
    return result
