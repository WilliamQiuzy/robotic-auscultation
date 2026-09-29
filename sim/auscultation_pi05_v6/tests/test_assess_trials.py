import pytest
from sim.auscultation_pi05_v6.assess_trials import check_documents,assess,POINTS
from sim.auscultation_pi05_v6.training_data import MODALITIES
from sim.auscultation_pi05_v6.task_observation import observation_binding
from sim.auscultation_pi05_v6.action_encoding import action_encoding_binding


def documents():
    outcome=dict(controller_source='pi05_absolute_joint_policy',expert_demonstration=False,expert_fallbacks=0,
        learned_policy_binding=dict(checkpoint_metadata_sha256='a'*64,
            training_recipe='dynamics_bound_absolute_joint_actions_v6_task_pose_logit_v1',modalities=MODALITIES,audio='absent',observation_contract=observation_binding(),action_encoding_binding=action_encoding_binding()),
        requested_points=['R4'],task_scope='full',learned_policy_queries=100,success=True,
        points=[dict(target='R4',success=True,observed_hold_s=3.,verified_front_contact_hold_s=3.,final_phase='DONE')])
    collection=dict(passed=True,smoothness_target_met=True,audit={'accepted':True},teardown=dict(close_returned=True,errors=[]))
    actions=dict(schema='wam_learned_absolute_joint_action_evidence_audit_v6',expected_checkpoint_metadata_sha256='a'*64,
        passed=True,evidence_consistent=True,command_binding_verified=True,sensor_payload_binding_verified=True,policy_task_observation_verified=True,
        request_receipt_binding_verified=True,server_request_log_verified=True,verified_executed_steps=1200,
        failed_queries=0,unverified_requests=0,artifact_sha256={'/acquired/episode/outcome.json':'b'*64})
    supervisor=dict(schema='wam_observed_supervisor_replay_audit_v1',passed=True,artifact_sha256={'outcome.json':'b'*64},
        replay=dict(every_observation_recomputed=True,every_manager_update_replayed=True,all_active_ticks_covered=True))
    return outcome,collection,actions,supervisor


def check(values):return check_documents(*values,checkpoint='a'*64,points=['R4'],scope='full',outcome_sha='b'*64)


def test_consistent_learned_task_documents_pass():
    assert not check(documents())


@pytest.mark.parametrize('change',['teacher','fallback','component','short_hold','wrong_model','missing_server','physical_failure'])
def test_expert_component_or_partial_success_cannot_pass(change):
    values=documents();outcome,collection,actions,_=values
    if change=='teacher':outcome['controller_source']='analytic_expert_shared_measured_supervisor'
    elif change=='fallback':outcome['expert_fallbacks']=1
    elif change=='component':outcome['task_scope']='contact'
    elif change=='short_hold':outcome['points'][0]['observed_hold_s']=2.25
    elif change=='wrong_model':outcome['learned_policy_binding']['checkpoint_metadata_sha256']='c'*64
    elif change=='missing_server':actions['server_request_log_verified']=False
    elif change=='physical_failure':collection['passed']=False
    assert check(values)


def test_ten_independent_points_do_not_establish_continuous_sequence():
    plan=dict(checkpoint_metadata_sha256='a'*64,cases=[dict(case_id=p,points=[p],scope='full',
        role='nominal_single_point',directory='/missing/'+p) for p in POINTS])
    with pytest.raises(ValueError,match='continuous sequence'):assess(plan)


def test_continuous_sequence_must_use_the_preserved_approved_point_order():
    records=[dict(case_id=p,points=[p],scope='full',role='nominal_single_point',directory='/missing/'+p) for p in POINTS]
    records.append(dict(case_id='full10',points=list(reversed(POINTS)),scope='full',role='nominal_continuous_sequence',directory='/missing/full10'))
    with pytest.raises(ValueError,match='continuous sequence'):
        assess(dict(checkpoint_metadata_sha256='a'*64,cases=records))
