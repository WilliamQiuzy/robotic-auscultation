"""CPU audit rejection tests; no synthetic trial is qualified or persisted."""
import copy
from types import SimpleNamespace as S
from unittest.mock import patch
import numpy as np
import pytest
from sim.auscultation_pi05_v7 import audit_numerical_learned_actions as audit
from sim.auscultation_pi05_v7.numerical_command_gate import compare_command
from sim.auscultation_pi05_v7.tests.test_numerical_command_gate import actual_fixture


def fixture():
    # Independent supplied cubic/commands, replayed through a deterministic test
    # trajectory. Full real-Ruckig + recorded-data coverage is a separate test.
    q,c=actual_fixture();prefix=q[0]*0
    physics=[dict(q_command=prefix.tolist(),time_s=i/240,subphase='settle') for i in range(3)]
    for i,command in enumerate(q):physics.append(dict(q_command=command.tolist(),time_s=(i+3)/240,subphase='learned_free'))
    c['samples']=[dict(physics_row=3+i,calculation_id=2,trajectory_time_s=(i+1)/240,command=q[i].tolist()) for i in range(4)]
    comparison=compare_command(q[:3],q[3],velocity=.175,acceleration=1.,certificate=c)
    event=dict(index=0,physics_row=6,time_s=physics[5]['time_s'],subphase='learned_free',active=False,recovery=False,contact_active=False,
        history=q[:3].tolist(),command=q[3].tolist(),certificate=c,comparison=comparison,physics_executed=True)
    learned=[dict(accepted=True,physics_row_start=3,physics_row_end_exclusive=7,target='R4',phase='FREE',raw_actions=[q[-1].tolist()]*10)]
    receipt=dict(raw_rejected_ticks=1,numerical_interventions=1,compared_ticks=4)
    return learned,physics,[event],receipt


def test_duplicate_events_fail_before_any_replay():
    learned,physics,events,receipt=fixture();events.append(dict(events[0],index=1))
    with pytest.raises(ValueError,match='Duplicate numerical event'):audit.replay_gate_records(learned,physics,events,receipt)


def test_rejected_numerical_controller_is_not_old_identity():
    from sim.auscultation_pi05_v7.assess_numerical_trials import check_documents
    with patch('sim.auscultation_pi05_v7.assess_numerical_trials.verify_policy_binding'):
        errors=check_documents(dict(controller_source='pi05_absolute_joint_policy'),{}, {}, {},
            checkpoint='a'*64,points=['R4'],scope='full',outcome_sha='b'*64)
    assert 'wrong numerical controller identity' in errors
    assert 'independent numerical evidence incomplete' in errors


def test_runner_does_not_accept_changed_original_inventory(tmp_path):
    import json,hashlib
    from sim.auscultation_pi05_v7.run_numerical_isaac import verify_numerical_inputs
    from sim.auscultation_pi05_v7.numerical_command_gate import numerical_gate_contract
    gate=tmp_path/'gate.json';gate.write_text(json.dumps(numerical_gate_contract()))
    inventory=tmp_path/'inventory.json';inventory.write_text('{}')
    args=S(task_scope='full',speed_scale=.7,armature=.02,position_iterations=32,numerical_gate_contract=gate,
        numerical_gate_contract_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),parent_runtime_inventory=inventory,
        parent_runtime_inventory_sha256='a'*64,source_project=tmp_path)
    with pytest.raises(ValueError,match='immutable313'):verify_numerical_inputs(args)


def test_runner_rejects_new_motion_recipe_before_startup(tmp_path):
    from sim.auscultation_pi05_v7.run_numerical_isaac import verify_numerical_inputs
    with pytest.raises(ValueError,match='full original'):
        verify_numerical_inputs(S(task_scope='full',speed_scale=.7,armature=.01,position_iterations=32))


def test_runner_snapshots_all_new_code_and_exposes_new_class():
    from sim.auscultation_pi05_v7 import run_numerical_isaac as runner
    source=runner.runtime_source_paths()
    assert {'pi05_v7/'+name for name in audit.MODULES}.issubset(source)
    klass,trajectory=runner.physical_classes()
    assert klass.__name__=='NumericalLearnedController'
    from sim.auscultation_pi05_v6.execution import JointTrajectory
    assert trajectory is JointTrajectory
