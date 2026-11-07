"""Historical student targets are data collection inputs, never teacher labels."""
from copy import deepcopy
import importlib
import numpy as np
import pytest


def module():
    return importlib.import_module('sim.ausc_policy.m2_student_recovery_prefix')


def fixture():
    absolute=np.arange(140,dtype=float).reshape(20,7)/1000
    events=[]
    for i in range(48):
        raw=None
        if i>=12:
            cell=(i-12)//12
            raw=dict(native_tick_in_cell=i%12,live_target_q=absolute[cell].tolist(),
                live_target_after_step_q=absolute[cell].tolist(),setter=dict(physics_row_start=12+cell*12,
                control_owner='ACT_L2',execution_role='given_l2_motion',offgrid_stop=False,target_q=absolute[cell].tolist()))
        events.append(dict(physics_row=i,actual_control_owner='ACT_L2' if raw else 'PREPARATION',
            raw_intent=raw,tick_error=None,state_audit='completed',physical_violations=[]))
    return events,{12:absolute},absolute


def test_only_completed_source_e10_raw_targets_are_reexecuted():
    m=module();events,blocks,absolute=fixture()
    got=m.extract_cells(events,blocks,cut_row=48)
    np.testing.assert_array_equal(got['raw_targets'],absolute[:3])
    assert got['start_row']==12 and got['end_row_exclusive']==48
    assert got['source_kind']=='retained_actual_student_raw_targets'
    assert got['positive_teacher_labels'] is False


def test_actual_context_requery_replaces_the_unexecuted_old_chunk_suffix():
    m=module();events,blocks,absolute=fixture();new=absolute+1.;blocks[24]=new
    for event in events[24:]:
        i=(event['physics_row']-24)//12
        event['raw_intent']['setter']['target_q']=new[i].tolist()
        event['raw_intent']['live_target_q']=new[i].tolist()
        event['raw_intent']['live_target_after_step_q']=new[i].tolist()
    got=m.extract_cells(events,blocks,cut_row=48)
    np.testing.assert_array_equal(got['raw_targets'],np.vstack([absolute[:1],new[:2]]))


def test_initial_history_wait_is_omitted_and_internal_native_wait_is_truthfully_retained():
    m=module();absolute=np.arange(140,dtype=float).reshape(20,7)/1000;new=absolute+1.;events=[]
    for i in range(288):
        raw=None;owner='PREPARATION'
        if i>=240:
            owner='ACT_L2';start=(i//12)*12
            wait=start in (240,264);role='given_l2_history_wait' if wait else 'given_l2_motion'
            target=([0.]*7 if start==240 else absolute[0].tolist() if start<276 else new[0].tolist())
            raw=dict(native_tick_in_cell=i%12,live_target_q=target,live_target_after_step_q=target,
                setter=dict(physics_row_start=start,control_owner='ACT_L2',execution_role=role,
                    offgrid_stop=False,target_q=target))
        events.append(dict(physics_row=i,actual_control_owner=owner,raw_intent=raw,
            tick_error=None,state_audit='completed',physical_violations=[]))
    got=m.extract_cells(events,{252:absolute,276:new},cut_row=288)
    np.testing.assert_array_equal(got['raw_targets'],[absolute[0],absolute[0],new[0]])
    assert got['start_row']==252 and got['source_history_wait_cells']==1 and got['source_policy_motion_cells']==2
    assert got['source_roles']==['given_l2_motion','given_l2_history_wait','given_l2_motion']
    events[264]['raw_intent']['setter']['target_q']=[0.]*7
    with pytest.raises(ValueError):m.extract_cells(events,{252:absolute,276:new},cut_row=288)


@pytest.mark.parametrize('fault',['changed_target','changed_live_target','stop_cell','missing_tick','failed_tick','wrong_owner','partial_cell','future_chunk'])
def test_bad_or_unexecuted_source_cells_cannot_become_a_recovery_prefix(fault):
    m=module();events,blocks,_=fixture();cut=48
    if fault=='changed_target':events[25]['raw_intent']['setter']['target_q'][0]=1.
    elif fault=='changed_live_target':events[25]['raw_intent']['live_target_after_step_q'][0]=1.
    elif fault=='stop_cell':events[25]['raw_intent']['setter']['offgrid_stop']=True
    elif fault=='missing_tick':events.pop(25)
    elif fault=='failed_tick':events[25]['tick_error']='original physical guard'
    elif fault=='wrong_owner':events[25]['actual_control_owner']='TEACHER_L2'
    elif fault=='partial_cell':cut=47
    else:blocks={}
    with pytest.raises(ValueError):m.extract_cells(events,blocks,cut_row=cut)


def test_cut_is_the_first_predeclared_completed_query_and_not_a_later_better_state():
    m=module();requests=[]
    for i,d in enumerate((.2,.081,.079,.061,.059)):
        state=np.zeros(17);state[9]=d
        requests.append(dict(context=dict(physics_row=12+i*120),history=[state.tolist(),state.tolist()]))
    assert m.select_cut(requests,.080)==252
    assert m.select_cut(requests,.060)==492
    with pytest.raises(ValueError):m.select_cut(requests,.01)


def test_validation_source_is_rejected_before_native_payload_is_opened(monkeypatch):
    m=module();opened=[]
    cfg=dict(m2_given_contact_expanded_pilot_deployment=dict(metadata=dict(split='validation',subject_id='p000923')))
    def read(ref):opened.append(ref);return deepcopy(cfg)
    monkeypatch.setattr(m.rt,'read_bound',read)
    with pytest.raises(ValueError,match='TRAIN'):m.authenticate(dict(path='config'),dict(path='owner'),dict(path='child'),dict(path='native'))
    assert opened==[dict(path='config')]
