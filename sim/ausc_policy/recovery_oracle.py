"""Bounded stationary-state expert query; no moving-state or contact oracle claim.

A query replans from measured q, then times and independently checks every new
240 Hz command. The first H10 labels are block-relative to that measured q.
The full trajectory is an explicit expert recovery proposal, not a policy
rollout. Unsupported, loaded, moving or infeasible states return Abstain.
"""
from __future__ import annotations

import hashlib
import json
import time
import numpy as np

from sim.curobo_expert_v1.timing import smooth_commands


def measured_stationarity(state):
    """Bind 13 consecutive measured positions, without replacing reported dq.

    PhysX can report a nonzero solver velocity at an unchanged held position.
    As in the existing measured TCP supervisor, derive observed movement from
    acquired positions and clocks. Keep raw dq and its original speed bound.
    """
    rows = state['recent_measurements']
    if not isinstance(rows,list) or len(rows) != 13:
        raise ValueError('Stationarity requires 13 actual consecutive measurements')
    q = np.asarray([r['q'] for r in rows],float)
    commands = np.asarray([r['q_command'] for r in rows],float)
    times = np.asarray([r['time_s'] for r in rows],float)
    indices = [r['physics_row'] for r in rows]
    if (q.shape != (13,7) or commands.shape != (13,7) or times.shape != (13,)
            or not np.isfinite(np.r_[q.ravel(),commands.ravel(),times]).all()
            or any(type(i) is not int for i in indices)
            or indices != list(range(state['physics_row']-12,state['physics_row']+1))
            or indices[0] < 0 or not np.allclose(np.diff(times),1/240,atol=1e-8,rtol=0)
            or times[-1] != state['time_s'] or not np.array_equal(q[-1],state['q'])
            or not np.array_equal(commands[-3:],state['command_history'])):
        raise ValueError('Stationarity history clock, positions or commands do not bind current query')
    velocity = float(np.max(abs(np.diff(q,axis=0)/np.diff(times)[:,None])))
    reported = float(np.max(abs(np.asarray(state['dq']))))
    stationary_commands = bool(np.max(abs(np.diff(commands,axis=0))) <= 1e-12)
    return dict(passed=bool(velocity <= .005 and reported <= .175 and stationary_commands),
        pose_difference_velocity_peak_rad_s=velocity,reported_dq_peak_rad_s=reported,
        stationary_commands=stationary_commands,window_s=float(times[-1]-times[0]),
        definition='13 acquired q/time samples; raw reported dq retained and bounded separately',
        pose_velocity_limit_rad_s=.005,reported_velocity_limit_rad_s=.175)


class StationaryRecoveryOracle:
    def __init__(self, checker, planner, *, lower, upper):
        self.checker, self.planner = checker, planner
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        if (self.lower.shape != (7,) or self.upper.shape != (7,)
                or not np.isfinite(np.r_[self.lower, self.upper]).all()
                or np.any(self.lower >= self.upper)):
            raise ValueError('Invalid oracle joint bounds')

    def query(self, state, context):
        started = time.monotonic()
        result = dict(kind='abstain', training_dataset_eligible=False,
            physical_qualification_passed=False, expert_not_learned=True,
            scope='R4 released stationary-state recovery proposal only; no moving/contact oracle or generalization claim')
        try:
            q, dq = np.asarray(state['q'], float), np.asarray(state['dq'], float)
            history = np.asarray(state['command_history'], float)
            goal = np.asarray(context['q_goal'], float)
            scalars = [state[k] for k in ('hall_load_N','physical_contact_N','compression_m','time_s')]
            if (q.shape != (7,) or dq.shape != (7,) or history.shape != (3,7)
                    or goal.shape != (7,) or not np.isfinite(np.r_[q,dq,history.ravel(),goal,scalars]).all()):
                raise ValueError('Malformed or nonfinite measured query')
            if (context['target'] != 'R4' or context['phase'] not in ('FREE','APPROACH')
                    or type(context['revision']) is not int or context['revision'] < 0
                    or type(state['physics_row']) is not int or state['physics_row'] < 0
                    or state['time_s'] < 0):
                raise ValueError('Unsupported query target, phase, revision or clock')
            encoded = json.dumps(dict(state=state,context=context),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            result['query_sha256'] = hashlib.sha256(encoded).hexdigest()
            result['measured_query'] = json.loads(encoded)
            result['stationarity'] = measured_stationarity(state)
            if not result['stationarity']['passed'] or np.max(abs(history[-1]-q)) > .03:
                raise ValueError('Query requires stationary measured and retained-command state')
            if (state['hall_load_N'] >= .2 or state['physical_contact_N'] >= .2
                    or state['physical_contact_N'] < 0 or not 0 <= state['compression_m'] <= .00015):
                raise ValueError('Query requires actual released contact state')
            vectors = np.vstack([q,history,goal])
            if np.any(vectors < self.lower) or np.any(vectors > self.upper):
                raise ValueError('Query exceeds joint bounds; no clipping')
            deadline = started+210.

            def check(command):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Query geometry budget exhausted')
                if np.any(command < self.lower) or np.any(command > self.upper):
                    raise ValueError('Timed command exceeds joint bounds')
                collision = self.checker.check(command,clearance=.003,allow_patient_contact=False,compression=0.)
                distance = float(collision.get('minimum_distance_m',float('nan')))
                if collision.get('clear') is not True or not np.isfinite(distance) or distance < .003-1e-9:
                    raise ValueError('Query clearance or posture refused: '+str(collision))
                return distance

            distances = [check(x) for x in (q,history[-1],goal)]
            bridge, search = self.planner(q.copy(),goal.copy(),np.random.default_rng(65),started+30.)
            result['search'] = search
            if bridge is None:
                raise ValueError('No checked current-state route')
            bridge = np.asarray(bridge,float)
            if (bridge.ndim != 2 or bridge.shape[1:] != (7,) or len(bridge)<2
                    or not np.isfinite(bridge).all() or not np.allclose(bridge[0],q,atol=1e-9,rtol=0)
                    or not np.allclose(bridge[-1],goal,atol=1e-9,rtol=0)):
                raise ValueError('Planner did not bind current measured origin and exact goal')
            route = np.vstack([history[-1],bridge])
            commands, timing = smooth_commands(route,speed_limit=.175,acceleration_limit=.5,
                jerk_limit=1.5,minimum_seconds=1/.7,maximum_seconds=120.,return_report=True)
            # These are the exact samples returned and later executed, not merely
            # an edge/spline surrogate. Measured dynamics still require Isaac.
            distances.extend(check(command) for command in commands)
            result.update(kind='trajectory',reason=None,waypoints=route.tolist(),timing=timing,
                commands_240hz=commands.tolist(),minimum_sampled_clearance_m=min(distances),
                checked_samples=len(distances),action_chunk_relative_rad=(commands[11:120:12]-q).tolist(),
                horizon=10,executed_prefix_steps=5,period_s=.05,
                continuation='Full expert proposal executed only by diagnostic teacher; further queries must supply fresh supported state')
        except (ValueError,TypeError,KeyError,TimeoutError,RuntimeError) as error:
            result.update(kind='abstain',reason=type(error).__name__+': '+str(error))
        result['wall_time_s'] = time.monotonic()-started
        return result
