"""Genuine observations with explicitly unexecuted CPU teacher proposals."""
from copy import deepcopy
import importlib
import numpy as np
import pytest

from test_ausc_given_contact_capture import new_inputs


def module():
    return importlib.import_module('sim.ausc_policy.m2_student_oracle_proposals')


def fixture(tmp_path):
    from sim.ausc_policy import m2_given_contact_runtime as rt
    ref,_,_,meta=new_inputs(tmp_path)
    events=[];row=24
    for i in range(row+1):
        q=np.full(7,.018 if i==12 else .02)
        command=np.full(7,.01)
        native=dict(q=[.02]*7,dq=[.03]*7,hall_N=0.,compression_m=-1e-8,
            **{k+'_N':0. for k in ('diaphragm','retainer','lip','contact_rim','carriage_other','frame','forbidden')})
        events.append(dict(physics_row=i,time_s=(i+1)/240.,state_audit='completed',
            physical_violations=[],tick_error=None,actual_control_owner='ACT_L2',
            controller_before=dict(subphase='given_l2_motion',actor=dict(phase='APPROACH')),q_command=command.tolist(),
            observed_before=dict(physics_row=i,time_s=i/240.,q=q.tolist()),
            native_row=native,interpolator=dict(input=dict(fields=dict(current_position=command.tolist(),
                current_velocity=[.01]*7,current_acceleration=[.02]*7)))))
    history=[rt.shared.contract.encode_state(events[i]['observed_before']['q'],ref.pose_base,1.,metadata=meta).tolist()
        for i in (12,24)]
    request=dict(measured_q=[.02]*7,history=history,history_times=[.05,.1],
        context=dict(physics_row=row,observed_at_s=.1,context=[meta['episode_id'],'L1',0,'APPROACH']))
    binding=dict(metadata=meta,requests=[request],events=events)
    return binding,request


def test_restore_previous_command_state_not_measured_or_post_query_step(tmp_path):
    m=module();binding,request=fixture(tmp_path)
    binding['events'][24]['interpolator']['input']['fields']['current_position']=[9.]*7
    binding['events'][24]['native_row'].update(hall_N=9.,dq=[9.]*7)
    query,snapshot,commands=m.query_at(binding,24)
    assert query['measured_q']==[.02]*7 and query['measured_dq']==[.03]*7
    assert snapshot['fields']['current_position']==[.01]*7
    assert snapshot['fields']['current_velocity']==[.01]*7
    assert snapshot['fields']['current_acceleration']==[.02]*7
    np.testing.assert_array_equal(commands,np.full((3,7),.01))
    assert query['hall_N']==0. and query['compression_m']==-1e-8


@pytest.mark.parametrize('fault',['history','clock','previous_failure','previous_incomplete','seam','free','not_requested','past_owner','past_phase'])
def test_unobserved_or_failed_query_is_rejected(tmp_path,fault):
    m=module();binding,request=fixture(tmp_path)
    if fault=='history':request['history'][0][0]+=.001
    elif fault=='clock':request['history_times'][0]+=.001
    elif fault=='previous_failure':binding['events'][23]['physical_violations']=['clearance']
    elif fault=='previous_incomplete':binding['events'][23]['state_audit']='not_completed'
    elif fault=='seam':binding['events'][23]['q_command'][0]+=.001
    elif fault=='free':request['context']['context'][-1]='FREE'
    elif fault=='not_requested':binding['requests']=[]
    elif fault=='past_owner':binding['events'][12]['actual_control_owner']='ANALYTIC_L3'
    else:binding['events'][12]['controller_before']['actor']['phase']='FREE'
    with pytest.raises(ValueError):m.query_at(binding,24)


def test_preserves_actual_floating_native_clocks_with_original_tolerance(tmp_path):
    m=module();binding,request=fixture(tmp_path)
    for j,row in enumerate((12,24)):
        stamp=row/240.+3e-12
        binding['events'][row]['observed_before']['time_s']=stamp
        request['history_times'][j]=stamp
    request['context']['observed_at_s']=request['history_times'][1]
    binding['events'][23]['time_s']=request['history_times'][1]
    query,_,_=m.query_at(binding,24)
    assert query['time_s']==request['history_times'][1]


def test_proposed_raw_setters_use_measured_fixed_origin_and_cannot_claim_execution():
    m=module();query=dict(measured_q=[.2]*7)
    proposal=dict(kind='recovery_proposal',preview=dict(cells=[[.7]*7,[.8]*7],
        all_240hz_commands_verified=True,physics_executed=False))
    got=m.proposed_block(proposal,query)
    np.testing.assert_allclose(got['proposed_relative_setters'][:2],[[.5]*7,[.6]*7])
    assert got['proposed_is_pad'].tolist()==[False,False]+[True]*18
    assert got['actions_executed'] is False and got['physics_qualified'] is False and got['training_eligible'] is False
    assert 'actions' not in got and 'physical_violations' not in got
    bad=deepcopy(proposal);bad['preview']['physics_executed']=True
    with pytest.raises(ValueError):m.proposed_block(bad,query)


def test_validation_is_rejected_before_source_native_authentication(tmp_path,monkeypatch):
    m=module();from sim.ausc_policy import m2_given_contact_runtime as rt
    ref=rt.write_new(tmp_path/'config.json',dict(m2_given_contact_expanded_pilot_deployment=dict(metadata=dict(split='validation'))))
    monkeypatch.setattr(m.student,'authenticate',lambda *args:pytest.fail('native authentication must not start'))
    with pytest.raises(ValueError,match='TRAIN'):
        m.authenticate_source(dict(config=ref,owner={},child={},native={}))
