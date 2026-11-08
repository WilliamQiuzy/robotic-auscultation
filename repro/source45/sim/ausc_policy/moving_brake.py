"""A checked braking proposal from retained native command state, never a reset.

This is the first stage of a moving-state recovery expert. It does not predict
the measured stopping pose or call a stationary oracle with fabricated history.
After actual braking, fresh acquired state must qualify for the next query.
"""
from __future__ import annotations

import hashlib
import json
import time
import numpy as np

DT=1/240
LIMITS=dict(velocity=.175,acceleration=.5,jerk=7.99999)


def vector(value):
    v=np.asarray(value,float)
    if v.shape!=(7,) or not np.isfinite(v).all(): raise ValueError('Invalid seven-vector')
    return v.copy()


def validate_origin(state,lower,upper):
    q,dq=vector(state['q']),vector(state['dq'])
    current={k:vector(state['command_state'][k]) for k in ('position','velocity','acceleration')}
    rows=state['recent_measurements']
    if not isinstance(rows,list) or len(rows)!=13: raise ValueError('13 actual measurements required')
    positions=np.stack([vector(r['q']) for r in rows])
    commands=np.stack([vector(r['q_command']) for r in rows])
    stamps=np.asarray([r['time_s'] for r in rows],float)
    indices=[r['physics_row'] for r in rows]
    if (type(state['physics_row']) is not int or state['physics_row']<12
            or not np.isfinite(stamps).all() or stamps[0]<0
            or indices!=list(range(state['physics_row']-12,state['physics_row']+1))
            or not np.allclose(np.diff(stamps),DT,atol=1e-8,rtol=0)
            or stamps[-1]!=state['time_s'] or not np.array_equal(positions[-1],q)
            or not np.array_equal(commands[-1],current['position'])):
        raise ValueError('Actual history does not bind current measured/command state')
    scalars=np.asarray([state[k] for k in ('hall_load_N','physical_contact_N','compression_m')],float)
    if (not np.isfinite(scalars).all() or scalars[0]>=.2 or not 0<=scalars[1]<.2
            or not 0<=scalars[2]<=.00015 or state['target']!='R4'
            or state['phase'] not in ('FREE','APPROACH')):
        raise ValueError('Only released R4 FREE/APPROACH braking is supported')
    if (np.max(abs(dq))>LIMITS['velocity']
            or np.max(abs(np.diff(positions,axis=0)/DT))>LIMITS['velocity']
            or np.max(abs(current['velocity']))>LIMITS['velocity']
            or np.max(abs(current['acceleration']))>LIMITS['acceleration']
            or np.max(abs(q-current['position']))>.03):
        raise ValueError('Actual or retained dynamics outside original command limits')
    allq=np.vstack([positions,commands])
    if np.any(allq<lower) or np.any(allq>upper): raise ValueError('Actual history outside joint bounds')
    return q,current,commands


def plan_moving_brake(state,inspect_command,*,lower,upper,native_module=None):
    started=time.monotonic()
    report=dict(kind='abstain',physics_executed=False,training_dataset_eligible=False,
        m0_passed=False,expert_not_learned=True,
        scope='Sampled native command braking only; actual stopping, contact and recovery not certified')
    try:
        lower,upper=vector(lower),vector(upper)
        if np.any(lower>=upper) or not callable(inspect_command): raise ValueError('Invalid bounds or checker')
        q,current,history=validate_origin(state,lower,upper)
        encoded=json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        report.update(measured_query=json.loads(encoded),query_sha256=hashlib.sha256(encoded).hexdigest())
        if native_module is None:
            import ruckig as native_module
        r=native_module;generator=r.Ruckig(7,DT);inp=r.InputParameter(7);out=r.OutputParameter(7)
        inp.control_interface=r.ControlInterface.Velocity
        for name,value in current.items(): setattr(inp,'current_'+name,value.tolist())
        for name,limit in LIMITS.items(): setattr(inp,'max_'+name,[limit]*7)
        inp.target_velocity=[0.]*7;inp.target_acceleration=[0.]*7
        if not generator.validate_input(inp,check_current_state_within_limits=True,check_target_state_within_limits=True):
            raise ValueError('Native braking input refused')
        commands=[]
        for tick in range(480):
            status=generator.update(inp,out)
            if status not in (r.Result.Working,r.Result.Finished): raise ValueError('Native braking failed')
            command=vector(out.new_position);velocity=vector(out.new_velocity);acceleration=vector(out.new_acceleration)
            if (np.max(abs(velocity))>LIMITS['velocity']+1e-9
                    or np.max(abs(acceleration))>LIMITS['acceleration']+1e-9):
                raise ValueError('Native braking exceeds retained dynamics')
            commands.append(command);out.pass_to_input(inp)
            if status==r.Result.Finished:
                if np.max(abs(np.r_[velocity,acceleration]))>1e-12:
                    raise ValueError('Native endpoint is not stationary')
                break
        else: raise TimeoutError('Braking exceeds declared two-second budget')
        native_steps=len(commands)
        # Padding is a proposed command hold, never measured stationarity.
        length=max(120,int(np.ceil((native_steps+13)/12))*12)
        commands=np.vstack([commands,[commands[-1]]*(length-native_steps)])
        sequence=np.vstack([history[-3:],commands])
        peaks={name:float(np.max(abs(np.diff(sequence,n=order,axis=0)/DT**order)))
            for order,name in enumerate(('velocity','acceleration','jerk'),start=1)}
        if any(peaks[k]>LIMITS[k]+1e-6 for k in LIMITS):
            raise ValueError('Executed command derivative seam exceeds original limits: '+str(peaks))
        distances=[]
        for command in np.vstack([q,current['position'],commands]):
            if time.monotonic()-started>120: raise TimeoutError('Braking geometry budget exhausted')
            if np.any(command<lower) or np.any(command>upper): raise ValueError('Braking exceeds joint bounds')
            result=inspect_command(command.copy());distance=float(result.get('minimum_distance_m',float('nan')))
            if result.get('clear') is not True or not np.isfinite(distance) or distance<.003-1e-9:
                raise ValueError('Braking clearance or posture refused: '+str(result))
            distances.append(distance)
        report.update(kind='braking_proposal',reason=None,commands_240hz=commands.tolist(),
            native_stop_steps=native_steps,proposed_duration_s=len(commands)*DT,
            retained_limits=LIMITS.copy(),command_derivative_peaks=peaks,
            minimum_sampled_clearance_m=min(distances),checked_samples=len(distances),
            action_chunk_relative_rad=(commands[11:120:12]-q).tolist(),horizon=10,executed_prefix_steps=5,period_s=.05,
            continuation='Execute checked braking, then acquire fresh stationary query; do not synthesize the stopped state')
    except (ValueError,TypeError,KeyError,RuntimeError,TimeoutError) as error:
        report.update(kind='abstain',reason=type(error).__name__+': '+str(error))
    report['wall_time_s']=time.monotonic()-started
    return report


def check_prefix_chunk(expected,actual,command):
    if len(expected)!=len(actual) or not expected: raise ValueError('Missing actual replay prefix')
    q=np.asarray([r['q'] for r in actual]);reference=np.asarray([r['q'] for r in expected])
    error=float(np.max(abs(q-reference)))
    if (not np.isfinite(q).all() or error>1e-4
            or not np.array_equal([r['q_command'] for r in actual],[r['q_command'] for r in expected])
            or not np.allclose([r['time_s'] for r in actual],[r['time_s'] for r in expected],atol=1e-8,rtol=0)
            or not np.array_equal(command,expected[-1]['q_command'])):
        raise ValueError('Actual prefix differs from retained source; no state reset permitted')
    return error
