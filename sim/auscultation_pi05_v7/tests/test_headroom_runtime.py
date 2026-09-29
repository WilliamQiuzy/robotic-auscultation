"""CPU tests using real Ruckig and the actual inherited physical call chain."""
import ast
import copy
import inspect
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import numpy as np
import pytest
from sim.auscultation_pi05_v7 import headroom_execution as execution
from sim.auscultation_pi05_v7 import headroom_learned_controller as controller
from sim.auscultation_pi05_v7 import audit_headroom_learned_actions as audit
from sim.auscultation_pi05_v7.tests import _headroom_fixture as raw_fixture
from sim.auscultation_pi05_v7.tests.test_runtime_binding import binding
from sim.auscultation_pi05_v6.learned_controller import LearnedController
from sim.auscultation_pi05_v6.dynamics_contract import execution_contract,sha256_json
from sim.auscultation_skill_v1.controller import ExpertController as SafetyController
from sim.curobo_expert_v1.controller import ExpertController as PhysicalParent


def test_contract_changes_only_internal_jerk_and_keeps_trained_binding():
    old=execution_contract();new=execution.runtime_execution_contract()
    assert [k for k in old if old[k]!=new[k]]==['jerk_rad_s3']
    assert old['jerk_rad_s3']==[8.]*7 and new['jerk_rad_s3']==[7.99999]*7
    assert sha256_json(old)=='9738053f355012d3b86d51f7685c6a79f0c85cda53849942cf1a0387ca20c3a8'
    assert sha256_json(new)=='1eee855014cad972156c13cee84360c9a29aa720a3595e11dacf4788fb4a821c'
    descriptor=execution.runtime_adaptation_contract()
    outcome=dict(runtime_adaptation_contract=descriptor,runtime_adaptation_contract_sha256=sha256_json(descriptor),
        trained_execution_sha256=sha256_json(old),runtime_execution_sha256=sha256_json(new),checkpoint_retrained_for_runtime=False)
    execution.verify_contract_bindings(outcome,{'execution_contract_sha256':sha256_json(old)})
    with pytest.raises(ValueError,match='retain original'):
        execution.verify_contract_bindings(outcome,{'execution_contract_sha256':sha256_json(new)})


@pytest.mark.parametrize('key,value',[('jerk',8.),('velocity',.25),('acceleration',1.),('dt',.01)])
def test_no_runtime_parameter_override(key,value):
    with pytest.raises(ValueError):execution.HeadroomJointTrajectory(np.zeros(7),**{key:value})


def test_three_copied_methods_have_only_factory_identifier_change():
    class Factory(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='HeadroomJointTrajectory':node.id='JointTrajectory'
            return node
    for name in ('_infer_step','_move_to_prepose','_contact_cycle'):
        original=ast.parse(textwrap.dedent(inspect.getsource(getattr(LearnedController,name))))
        adapted=ast.parse(textwrap.dedent(inspect.getsource(getattr(controller.HeadroomLearnedController,name))))
        assert ast.dump(original)==ast.dump(Factory().visit(adapted)),name
    assert controller.HeadroomLearnedController._tick is LearnedController._tick
    assert controller.HeadroomLearnedController._return_home is LearnedController._return_home
    assert '_tick' not in controller.HeadroomLearnedController.__dict__


@dataclass
class Packet:sample_time_s:float=0.


def physical_fixture(tmp_path):
    obj=controller.HeadroomLearnedController.__new__(controller.HeadroomLearnedController);calls=[]
    obj.joint_speed_limit=.175;obj._contact_active=False;obj.command_collision_samples=0;obj.observed_collision_samples=0
    obj._last_tcp_measurement=None;obj._last_observed_hall_packet=Packet()
    state=dict(q=np.zeros(7),dq=np.zeros(7),compression_m=0.,time_s=3/240)
    obj.env=S(rows=[dict(q_command=[0.]*7) for _ in range(3)],state=lambda:state,base=np.eye(4),out=tmp_path)
    obj.sensor=S(rows=[],tick=lambda s:calls.append('sensor'))
    obj.kin=S(tcp=lambda q,compression:np.eye(4))
    obj._read_observed_hall=lambda s:(calls.append('Hall ADC') or S(displacement_estimate_m=0.))
    obj._remember_observed_tcp=lambda *a:calls.append('measured TCP')
    obj._capture_observed_tick=lambda *a:calls.append('capture')
    obj._check_collision=lambda *a:calls.append('collision')
    def physical(self,q,subphase,active=False,recovery=False):
        calls.append('physics');assert active is False
        obj.env.rows.append(dict(q_command=np.asarray(q).tolist()));state['time_s']+=1/240
        return state
    return obj,calls,physical


def test_actual_mro_executes_gate_both_collisions_physics_adc_once(tmp_path):
    obj,calls,physical=physical_fixture(tmp_path)
    assert SafetyController in type(obj).__mro__
    with patch.object(PhysicalParent,'_tick',physical):obj._tick(np.zeros(7),'learned_free')
    assert calls==['collision','physics','collision','sensor','Hall ADC','measured TCP','capture']
    assert obj.command_collision_samples==1 and obj.observed_collision_samples==1


def test_original_jerk_gate_is_not_overridden(tmp_path):
    obj,calls,physical=physical_fixture(tmp_path)
    # Exact old failure: retained q representation has jerk 8.000000011179509.
    values=[1.171404415658409,1.1716842574510222,1.171962995227978,1.1722412076929807]
    for row,q in zip(obj.env.rows,values):row['q_command'][3]=q
    command=np.zeros(7);command[3]=values[-1]
    with patch.object(PhysicalParent,'_tick',physical),pytest.raises(controller.EpisodeRejected,match='derivative gate'):
        obj._tick(command,'learned_free')
    assert calls==[]


def test_measured_collision_rejection_is_never_retried(tmp_path):
    obj,calls,physical=physical_fixture(tmp_path)
    def collision(*args):
        calls.append('collision')
        if calls.count('collision')==2:raise RuntimeError('physical collision')
    obj._check_collision=collision
    with patch.object(PhysicalParent,'_tick',physical),pytest.raises(RuntimeError,match='physical collision'):
        obj._tick(np.zeros(7),'learned_free')
    assert calls==['collision','physics','collision']


def evidence(tmp_path,**kwargs):
    data=raw_fixture.fixture(tmp_path,**kwargs);data['binding']=binding()
    for row in data['learned']:row['inference'].update(data['binding'])
    return data


def test_independent_raw_ruckig_headroom_replay_and_analytic_tick(tmp_path):
    data=evidence(tmp_path)
    expected=raw_fixture.SEED+np.array([1.]*6+[-1.])*7.99999*(1/240)**3/6
    np.testing.assert_allclose(data['physics'][3]['q_command'],expected,atol=1e-14,rtol=0)
    report=audit.audit_records(**data)
    assert report['passed'] and report['verified_executed_steps']==24,report
    assert report['replay']['joint_jerk_rad_s3']==7.99999
    assert audit.audit_original_gates(data['physics'])['original_gate_checked_ticks']==24
    assert data['learned'][0]['goal_q']==data['learned'][0]['raw_actions'][0]


@pytest.mark.parametrize('case',['double_add','reset','old_executor','tamper','wrong_service'])
def test_rejects_wrong_semantics_and_execution_even_with_consistent_receipts(tmp_path,case):
    data=evidence(tmp_path,add_measured_origin=case=='double_add',reset_between=case=='reset')
    if case=='old_executor':
        from sim.auscultation_pi05_v6.execution import JointTrajectory
        t=JointTrajectory(raw_fixture.SEED,velocity=.175)
        for query in data['learned']:
            t.set_target(query['goal_q'])
            for row in data['physics'][query['physics_row_start']:query['physics_row_end_exclusive']]:row['q_command']=t.step().tolist()
    if case=='tamper':data['physics'][3]['q_command'][6]+=1e-12
    if case=='wrong_service':
        data['binding']['training_recipe']='other'
        for row in data['learned']:row['inference']['training_recipe']='other'
    report=audit.audit_records(**data)
    assert not report['passed'],report


def test_partial_rejected_prefix_is_evidence_not_success(tmp_path):
    data=evidence(tmp_path);last=data['learned'][-1];end=last['physics_row_start']+5
    data['physics']=data['physics'][:end]
    last.update(physics_row_end_exclusive=end,failure_reason='independent safety rejection')
    report=audit.audit_records(**data)
    assert report['passed'] and report['failed_queries']==1 and report['verified_executed_steps']==17
    assert report['policy_acceptance']=='not_evaluated'


def test_next_target_requires_real_retained_state_release_and_stationary_reset(tmp_path):
    data=evidence(tmp_path,new_target=True)
    report=audit.audit_records(**data)
    assert report['passed'],report
    assert report['trajectory_initializations']==2 and report['verified_release_stop_steps']>0
    gates=audit.audit_original_gates(data['physics'])
    assert gates['original_gate_checked_ticks']==24+report['verified_release_stop_steps']


def test_original_gates_reject_actual_excess():
    physics=[dict(q_command=[0.]*7,subphase='settle') for _ in range(3)]
    physics.append(dict(q_command=[1e-5]+[0.]*6,subphase='learned_free'))
    with pytest.raises(ValueError,match='whole-stream'):audit.audit_original_gates(physics)


def test_auditor_uses_original_gate_not_internal_headroom_limit():
    dt=1/240
    for jerk,passed in [(7.99999002288132,True),(8.000001,False)]:
        physics=[]
        for index in range(4):
            q=[0.]*7;q[0]=jerk*(index*dt)**3/6
            physics.append(dict(q_command=q,subphase='learned_free' if index==3 else 'settle'))
        if passed:assert audit.audit_original_gates(physics)['original_whole_stream_gate_verified']
        else:
            with pytest.raises(ValueError,match='whole-stream'):audit.audit_original_gates(physics)


def test_runner_identity_and_new_sources():
    from sim.auscultation_pi05_v7 import run_headroom_isaac as runner
    assert runner.physical_classes()==(controller.HeadroomLearnedController,execution.HeadroomJointTrajectory)
    assert {'pi05_v7/'+name for name in execution.MODULES}.issubset(runner.runtime_source_paths())
    with pytest.raises(ValueError,match='full original'):
        runner.verify_headroom_inputs(S(task_scope='contact',speed_scale=.7,armature=.02,position_iterations=32))


def test_assessment_rejects_old_identity_and_missing_adaptation():
    from sim.auscultation_pi05_v7.assess_headroom_trials import check_documents
    with patch('sim.auscultation_pi05_v7.assess_headroom_trials.verify_policy_binding'):
        errors=check_documents(dict(controller_source='pi05_absolute_joint_policy'),{},{},{},
            checkpoint='a'*64,points=['R4'],scope='full',outcome_sha='b'*64)
    assert 'wrong headroom controller class' in errors
    assert 'runtime adaptation/unchanged gate audit incomplete' in errors
