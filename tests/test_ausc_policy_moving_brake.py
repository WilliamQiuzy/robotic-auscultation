"""A moving-state proposal must preserve the command generator's real state."""
from copy import deepcopy
import types
import numpy as np
import pytest

from sim.ausc_policy.moving_brake import plan_moving_brake


class Native:
    ControlInterface=types.SimpleNamespace(Velocity='velocity')
    Result=types.SimpleNamespace(Working=0,Finished=1)
    received=None
    class InputParameter:
        def __init__(self,n): pass
    class OutputParameter:
        def __init__(self,n): pass
        def pass_to_input(self,i):
            for key in ('position','velocity','acceleration'):
                setattr(i,'current_'+key,getattr(self,'new_'+key))
    class Ruckig:
        def __init__(self,n,dt): pass
        def validate_input(self,i,**kwargs):
            Native.received=deepcopy(vars(i));return True
        def update(self,i,o):
            # Valid stationary special case; native moving behavior is exercised
            # separately on pinned Linux Ruckig, not claimed by this fake.
            o.new_position=i.current_position;o.new_velocity=[0.]*7;o.new_acceleration=[0.]*7
            return Native.Result.Finished


def state():
    return dict(q=[.1]*7,dq=[.008]*7,time_s=1.,physics_row=239,
        target='R4',phase='APPROACH',hall_load_N=0.,physical_contact_N=0.,compression_m=0.,
        command_state=dict(position=[.1]*7,velocity=[0.]*7,acceleration=[0.]*7),
        recent_measurements=[dict(q=[.1]*7,q_command=[.1]*7,
            time_s=1-(12-i)/240,physics_row=227+i) for i in range(13)])


def query(s,inspect=lambda q:dict(clear=True,minimum_distance_m=.004)):
    return plan_moving_brake(s,inspect,lower=[-1.]*7,upper=[1.]*7,native_module=Native)


def test_stationary_special_case_is_not_permission_to_rewrite_solver_velocity():
    s=state();before=deepcopy(s);r=query(s)
    assert r['kind']=='braking_proposal' and r['training_dataset_eligible'] is False
    assert len(r['commands_240hz'])==120
    assert Native.received['current_velocity']==[0.]*7
    assert r['measured_query']['dq']==[.008]*7 and s==before
    assert r['physics_executed'] is False


@pytest.mark.parametrize('change',['clock','origin','command','loaded','limits','nan','api_velocity'])
def test_stale_unbound_or_loaded_state_is_refused_before_native(change):
    s=state();Native.received=None
    if change=='clock': s['recent_measurements'][0]['time_s']-=.01
    if change=='origin': s['recent_measurements'][-1]['q'][0]+=.01
    if change=='command': s['command_state']['position'][0]+=.01
    if change=='loaded': s['hall_load_N']=.2
    if change=='limits': s['command_state']['velocity'][0]=.176
    if change=='nan': s['command_state']['acceleration'][0]=float('nan')
    if change=='api_velocity': s['dq'][0]=.176
    r=query(s)
    assert r['kind']=='abstain' and Native.received is None
    assert 'action_chunk_relative_rad' not in r


def test_collision_refusal_discards_proposal_including_final_stationary_padding():
    calls=[]
    def inspect(q):
        calls.append(q);good=len(calls)!=70
        return dict(clear=good,minimum_distance_m=.004 if good else .002)
    r=query(state(),inspect)
    assert r['kind']=='abstain' and 'commands_240hz' not in r


def test_generator_cannot_reset_actual_velocity_at_the_seam():
    s=state();s['command_state']['velocity']=[.03]*7
    for i,row in enumerate(s['recent_measurements']):
        row['q_command']=[.1+.03*(i-12)/240]*7
    r=query(s)
    assert Native.received['current_velocity']==[.03]*7
    # The deliberately false fake snaps to rest; actual command differences
    # must reject it even when the native engine reports Finished.
    assert r['kind']=='abstain' and 'derivative' in r['reason']
