from copy import deepcopy

import numpy as np
import pytest

from sim.ausc_policy.recovery_labels import audit_query_labels, audit_brake_labels


def fixture():
    physics=[]
    for i in range(144):
        command=np.full(7,i*.00001)
        physics.append(dict(q=(command-.000001).tolist(),q_command=command.tolist(),
            time_s=(i+1)/240,oracle_role='PREPARATION' if i<24 else 'EXPERT_RECOVERY'))
    current=physics[23]
    query=dict(kind='trajectory',measured_query=dict(state=dict(q=current['q'],time_s=current['time_s'],
        physics_row=23)),commands_240hz=[r['q_command'] for r in physics[24:]],
        horizon=10,executed_prefix_steps=5,period_s=.05)
    query['action_chunk_relative_rad']=(np.asarray(query['commands_240hz'])[11:120:12]-current['q']).tolist()
    return query,physics


def test_first_label_is_t_plus_50ms_and_original_measured_anchor():
    query,physics=fixture();old=deepcopy(physics)
    result=audit_query_labels(query,physics)
    assert result['h10_actual_command_match'] is True
    assert result['label_physics_rows']==list(range(35,144,12))
    assert result['training_dataset_eligible'] is False
    assert physics==old


@pytest.mark.parametrize('change',['double_shift','anchor','executed','role','clock'])
def test_mispaired_query_or_execution_is_rejected(change):
    query,physics=fixture()
    if change=='double_shift': query['action_chunk_relative_rad'][0]=query['action_chunk_relative_rad'][1]
    if change=='anchor': query['measured_query']['state']['q']=[0.]*7
    if change=='executed': physics[35]['q_command'][0]+=.01
    if change=='role': physics[35]['oracle_role']='PREPARATION'
    if change=='clock': physics[35]['time_s']+=.01
    with pytest.raises(ValueError): audit_query_labels(query,physics)


def brake_fixture():
    query,physics=fixture()
    for row in physics: row['oracle_role']='EXPERT_BRAKE'
    query['kind']='braking_proposal'
    state=query['measured_query']['state'];query['measured_query']=state
    state['command_state']={'position':physics[23]['q_command']}
    state['recent_measurements']=[dict(physics[i],physics_row=i) for i in range(11,24)]
    return query,physics


def test_brake_labels_bind_actual_origin_and_full_physics_commands():
    query,physics=brake_fixture()
    report=audit_brake_labels(query,physics)
    assert report['proposal_physics_commands_matched']==120
    assert report['query_physics_row']==23
    assert report['training_dataset_eligible'] is False


@pytest.mark.parametrize('change',['history','origin_command','role','future','anchor'])
def test_brake_labels_refuse_fabricated_origin_or_execution(change):
    query,physics=brake_fixture()
    if change=='history': query['measured_query']['recent_measurements'][0]['physics_row']+=1
    if change=='origin_command': query['measured_query']['command_state']['position']=[.4]*7
    if change=='role': physics[35]['oracle_role']='EXPERT_RECOVERY'
    if change=='future': query['commands_240hz'][0]=[.5]*7
    if change=='anchor': query['measured_query']['q']=[.8]*7
    with pytest.raises(ValueError): audit_brake_labels(query,physics)


def capture_fixture():
    physics=[dict(q=[0.]*7,dq=[0.]*7,q_command=[0.]*7,time_s=(i+1)/240,
                  oracle_role='PREPARATION' if i<24 else 'MEASURED_ANALYTIC_L3') for i in range(36)]
    entry=dict(q=[0.]*7,dq=[0.]*7,physics_row=23,time_s=.1,command_history=[[0.]*7]*3,
               recent_measurements=[dict(physics[i],physics_row=i) for i in range(11,24)])
    receipt=dict(preparation_completed=True,local_alignment_used=False,physical_joint_resets=0,
                 requested_command=[0.]*7,entry=entry)
    return receipt,physics


def test_capture_entry_binds_actual_stationary_command_and_direct_l3():
    from sim.ausc_policy.recovery_labels import audit_capture_entry
    receipt,physics=capture_fixture()
    result=audit_capture_entry(receipt,physics)
    assert result['entry_physics_row']==23
    assert result['stationarity']['passed'] is True
    assert result['training_dataset_eligible'] is False


@pytest.mark.parametrize('defect',['history','role','q','requested','alignment','reset'])
def test_capture_entry_rejects_modified_preparation_or_hidden_correction(defect):
    from sim.ausc_policy.recovery_labels import audit_capture_entry
    receipt,physics=capture_fixture()
    if defect=='history': receipt['entry']['recent_measurements'][0]['q']=[1.]*7
    if defect=='role': physics[24]['oracle_role']='EXPERT_RECOVERY'
    if defect=='q': receipt['entry']['q']=[.1]*7
    if defect=='requested': receipt['requested_command']=[.2]*7
    if defect=='alignment': receipt['local_alignment_used']=True
    if defect=='reset': receipt['physical_joint_resets']=1
    with pytest.raises(ValueError): audit_capture_entry(receipt,physics)
