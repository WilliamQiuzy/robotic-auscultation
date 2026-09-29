"""Completed task evidence must satisfy both V7 identity and unchanged physical gates."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from sim.auscultation_pi05_v7.tests.test_runtime_binding import binding,module


def documents():
    outcome=dict(controller_source='pi05_absolute_joint_policy',expert_demonstration=False,expert_fallbacks=0,
        learned_policy_binding=binding(),requested_points=['R4'],task_scope='full',learned_policy_queries=100,
        success=True,points=[dict(target='R4',success=True,observed_hold_s=3.,
                                 verified_front_contact_hold_s=3.,final_phase='DONE')])
    collection=dict(passed=True,smoothness_target_met=True,audit={'accepted':True},
                    teardown=dict(close_returned=True,errors=[]))
    actions=dict(schema='wam_learned_absolute_joint_action_evidence_audit_v7',
        expected_checkpoint_metadata_sha256='a'*64,passed=True,evidence_consistent=True,
        command_binding_verified=True,sensor_payload_binding_verified=True,policy_task_observation_verified=True,
        request_receipt_binding_verified=True,server_request_log_verified=True,inference_source_snapshot_verified=True,
        verified_executed_steps=1200,failed_queries=0,unverified_requests=0,
        artifact_sha256={'/acquired/episode/outcome.json':'b'*64})
    supervisor=dict(schema='wam_observed_supervisor_replay_audit_v1',passed=True,
        artifact_sha256={'outcome.json':'b'*64},replay=dict(every_observation_recomputed=True,
        every_manager_update_replayed=True,all_active_ticks_covered=True))
    return outcome,collection,actions,supervisor


def check(values):
    return module('assess_trials').check_documents(*values,checkpoint='a'*64,
        points=['R4'],scope='full',outcome_sha='b'*64)


def test_complete_v7_evidence_and_frozen_v6_supervisor_contract():
    assert check(documents())==[]


@pytest.mark.parametrize('case',['v6_recipe','v6_audit','missing_source','missing_model_semantics',
    'short_hold','fallback','failed_jerk','partial_query','no_snapshot','no_teardown'])
def test_semantic_and_physical_failures_cannot_be_accepted(case):
    data=documents();outcome,collection,actions,supervisor=data
    if case=='v6_recipe':outcome['learned_policy_binding']['training_recipe']='dynamics_bound_absolute_joint_actions_v6_task_pose_logit_v1'
    elif case=='v6_audit':actions['schema']='wam_learned_absolute_joint_action_evidence_audit_v6'
    elif case=='missing_source':outcome['learned_policy_binding']['inference_source_sha256'].pop('v6/execution.py')
    elif case=='missing_model_semantics':outcome['learned_policy_binding'].pop('model_action_semantics')
    elif case=='short_hold':outcome['points'][0]['observed_hold_s']=2.99
    elif case=='fallback':outcome['expert_fallbacks']=1
    elif case=='failed_jerk':collection['smoothness_target_met']=False
    elif case=='partial_query':actions['failed_queries']=1
    elif case=='no_snapshot':actions.pop('inference_source_snapshot_verified')
    elif case=='no_teardown':collection['teardown']['close_returned']=False
    assert check(data)


def test_ten_single_runs_do_not_prove_interpoint_execution():
    assessor=module('assess_trials')
    cases=[dict(case_id=p,points=[p],scope='full',role='nominal_single_point',directory='/absent/'+p)
           for p in assessor.POINTS]
    with pytest.raises(ValueError,match='continuous sequence'):
        assessor.assess(dict(cases=cases,checkpoint_metadata_sha256='a'*64))


def test_runner_rejects_mixed_sequence_geometry_and_base():
    runner=module('run_isaac')
    a=SimpleNamespace(target=SimpleNamespace(name='R4'),plan=dict(world_from_left_base=np.eye(4).tolist(),
        revalidation={'collision_regeneration_sha256':'a'*64}),
        inputs={'collision_metadata':(None,b'json'),'collision_arrays':(None,b'npz')})
    b=copy.deepcopy(a);b.target.name='L1'
    runner.validate_sequence_bundles([a,b],['R4','L1'],'a'*64)
    b.plan['world_from_left_base'][0][3]=.001
    with pytest.raises(ValueError,match='bases differ'):runner.validate_sequence_bundles([a,b],['R4','L1'],'a'*64)
    b=copy.deepcopy(a);b.target.name='L1';b.inputs['collision_arrays']=(None,b'changed')
    with pytest.raises(ValueError,match='geometry differs'):runner.validate_sequence_bundles([a,b],['R4','L1'],'a'*64)
    with pytest.raises(ValueError):runner.validate_sequence_bundles([],[],'a'*64)
