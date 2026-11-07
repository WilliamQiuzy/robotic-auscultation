"""CPU contract tests only; no model, native physics or qualification claims."""
from types import SimpleNamespace
import numpy as np
import pytest
from test_ausc_policy_m2_contract import metadata
from sim.ausc_policy import m2_contract as contract


def module():
    from sim.ausc_policy import m2_act_controller
    return m2_act_controller


def test_learned_interpolator_preserves_approved_expert_free_motion_limits():
    assert module().LIMITS==dict(velocity=.175,acceleration=.5,jerk=1.5)


def estimate(q=None, **changes):
    row=dict(episode_id='episode',target_id='L1',revision=0,phase='FREE',observed_at_s=0.,
             measured_q=np.zeros(7) if q is None else q,estimated_target_base=np.eye(4),
             confidence=.9,source_id='rgbd-frame-0')
    row.update(changes);return row


def test_prefix_decodes_fixed_measured_origin_once_and_executes_ten():
    m=module();q=np.arange(7)*.01;relative=np.arange(1,21)[:,None]*np.ones((20,7))*.001
    block=contract.JointActionBlock('block_relative_joint',relative,q,metadata())
    prefix=m.FixedOriginPrefix(block,observed_q=q,context=('episode','L1',0,'FREE'),stamp=0.)
    for i in range(10):
        command=prefix.pop(i*.05,('episode','L1',0,'FREE'))
        np.testing.assert_allclose(command,q+relative[i])
    assert prefix.remaining==0
    with pytest.raises(ValueError,match='exhausted'):prefix.pop(.5,('episode','L1',0,'FREE'))


@pytest.mark.parametrize('context,stamp',[(('episode','L1',1,'FREE'),0.),
    (('episode','L2',0,'FREE'),0.),(('other','L1',0,'FREE'),0.),
    (('episode','L1',0,'APPROACH'),0.),(('episode','L1',0,'FREE'),.001)])
def test_prefix_context_or_clock_change_never_reuses_tail(context,stamp):
    m=module();b=contract.JointActionBlock('block_relative_joint',np.zeros((20,7)),np.zeros(7),metadata())
    p=m.FixedOriginPrefix(b,observed_q=np.zeros(7),context=('episode','L1',0,'FREE'),stamp=0.)
    with pytest.raises(ValueError):p.pop(stamp,context)
    assert p.remaining==0


def test_prefix_rejects_wrong_origin_or_absolute_contract():
    m=module()
    for arm,origin in [('absolute_joint',np.zeros(7)),('block_relative_joint',np.ones(7))]:
        b=contract.JointActionBlock(arm,np.zeros((20,7)),origin,metadata())
        with pytest.raises(ValueError):m.FixedOriginPrefix(b,observed_q=np.zeros(7),context=('e','L1',0,'FREE'),stamp=0.)


@pytest.mark.parametrize('changes',[dict(phase='CONTACT'),dict(revision=-1),dict(target_id='R1'),
    dict(episode_id='wrong'),dict(observed_at_s=-.05),dict(measured_q=np.ones(7)),
    dict(confidence=.79),dict(source_id=''),dict(estimated_target_base=np.zeros((4,4)))])
def test_estimate_is_same_time_identity_bound_without_truth_fallback(changes):
    with pytest.raises(ValueError):module().validate_estimate(estimate(**changes),
        dict(q=np.zeros(7),time_s=0.),episode_id='episode',target_id='L1',metadata=metadata())


def test_model_state_is_contract_state17_only():
    row=module().validate_estimate(estimate(),dict(q=np.zeros(7),time_s=0.),
        episode_id='episode',target_id='L1',metadata=metadata())
    assert row['state17'].shape==(17,)
    assert row['context']==('episode','L1',0,'FREE')


def test_frozen_analytic_handoff_does_not_relax_boundary():
    from sim.auscultation_expert_layout_v1.phase_manager import PhaseConfig
    c=PhaseConfig();m=module()
    metrics=dict(gap_m=.04,tangent_m=.003,angle_deg=5.,tcp_speed_m_s=.019)
    assert m.approach_gate(metrics,c)['passed']
    for changes in [dict(gap_m=.043),dict(tangent_m=.00301),dict(angle_deg=5.01),dict(tcp_speed_m_s=.02)]:
        assert not m.approach_gate(dict(metrics,**changes),c)['passed']


def test_result_tagging_cannot_leave_expert_labels_or_claim_qualification():
    m=module();row=dict(training_accepted=True,learned_policy=False,expert_demonstration=True)
    result=dict(success=True,training_accepted=True,learned_policy=False,points=[row])
    m.tag_learned_result(result,dict(checkpoint_sha256='a'*64),[dict(l2_handoff_success=True)])
    for record in (result,row):
        assert record['learned_policy'] and not record['expert_demonstration'] and not record['training_accepted']
    assert result['l2_handoff_success_count']==1 and result['l2_handoff_attempt_count']==1
    assert result['l2_handoff_success_rate']==1 and not result['training_eligible']


def test_factory_forbids_fixture_route_and_pins_cpu(monkeypatch):
    m=module();calls=[]
    def load(*args,**kwargs):calls.append((args,kwargs));return object()
    monkeypatch.setattr(m.act,'load_checkpoint',load)
    monkeypatch.setattr(m,'M2ACTController',lambda *args,**kwargs:kwargs)
    deployment=dict(schema=m.DEPLOYMENT_SCHEMA,checkpoint='best.pt',checkpoint_sha256='a'*64,
        lerobot_root='/official',metadata=metadata(),episode_id='episode')
    result=m.make_controller(None,None,None,deployment=deployment,estimate_provider=lambda *_:None)
    assert calls[0][1]['device']=='cpu' and calls[0][1]['allow_cpu_fixture'] is False
    assert result['policy_receipt']['learned_policy'] is True
    with pytest.raises(ValueError):m.make_controller(None,None,None,
        deployment=dict(deployment,allow_cpu_fixture=True),estimate_provider=lambda *_:None)


def bare_controller(monkeypatch):
    m=module();c=m.M2ACTController.__new__(m.M2ACTController)
    state=dict(q=np.zeros(7),time_s=0.)
    c.env=SimpleNamespace(rows=[],state=lambda:state)
    c.q=np.zeros(7);c.control_rows=[];c.l2_actions=[];c.l2_queries=[]
    c.prefix=None;c._l2_started_at=0.;c.control_owner='ACT_L2';c.actor_phase='FREE'
    c.target=SimpleNamespace(name='L1');c.metadata=metadata();c.episode_id='episode'
    c.kin=SimpleNamespace(lower_limits=np.full(7,-3.),upper_limits=np.full(7,3.))
    c.trajectory=SimpleNamespace(set_target=lambda q:None,step=lambda:np.zeros(7))
    c._observation=lambda:dict(context=('episode','L1',0,'FREE'))
    c._handoff_measurements=lambda:dict(passed=False)
    def checked_block(commands,*args,**kwargs):
        for q in commands:
            c.q=np.array(q);state['time_s']+=1/240
            c.env.rows.append(dict(q_command=q.tolist()))
    c._block=checked_block
    block=contract.JointActionBlock('block_relative_joint',np.zeros((20,7)),np.zeros(7),metadata())
    c.prefix=m.FixedOriginPrefix(block,observed_q=np.zeros(7),context=('episode','L1',0,'FREE'),stamp=0.)
    return c,state


def test_ten_rows_execute_through_120_separate_native_safety_blocks(monkeypatch):
    c,state=bare_controller(monkeypatch)
    for _ in range(10):assert c._execute_action(('episode','L1',0,'FREE'))=='continue'
    assert len(c.env.rows)==120 and state['time_s']==pytest.approx(.5)
    assert c.prefix.remaining==0 and len(c.l2_actions)==10


def test_handoff_or_context_change_interrupts_before_second_native_tick(monkeypatch):
    for event in ('handoff','context'):
        c,state=bare_controller(monkeypatch)
        if event=='handoff':c._handoff_measurements=lambda:dict(passed=True)
        else:c._observation=lambda:dict(context=('episode','L1',1,'FREE'))
        assert c._execute_action(('episode','L1',0,'FREE'))==event
        assert len(c.env.rows)==1 and c.prefix is None


def test_native_safety_exception_is_not_hidden_or_retried(monkeypatch):
    c,state=bare_controller(monkeypatch)
    def reject(*args,**kwargs):raise RuntimeError('full mesh rejected')
    c._block=reject
    with pytest.raises(RuntimeError,match='full mesh'):c._execute_action(('episode','L1',0,'FREE'))
    assert len(c.env.rows)==0 and not c.l2_actions[0]['completed']


def query_history(c,state,observation):
    from sim.ausc_policy.m2_stage import CausalHistory
    c._history=CausalHistory(0.)
    c._history.observe(observation['state17'],0.,observation['context'])
    state['time_s']=observation['observed_at_s']=.05
    c._history.observe(observation['state17'],.05,observation['context'])


def test_query_passes_only_measured_q_estimated_pose_confidence(monkeypatch):
    c,state=bare_controller(monkeypatch);calls=[]
    state.update(hall_N=99.,dq=np.ones(7),compression_m=.02)
    def predict(q,pose,confidence,*,metadata,history,history_times):
        calls.append((q.copy(),pose,confidence,metadata))
        return contract.JointActionBlock('block_relative_joint',np.zeros((20,7)),q,metadata)
    c.predictor=predict
    observation=module().validate_estimate(estimate(),state,episode_id='episode',target_id='L1',metadata=metadata())
    query_history(c,state,observation)
    c._query(observation)
    assert len(calls)==1 and c.l2_queries[-1]['accepted']
    assert len(c.l2_queries[-1]['state17'])==17
    np.testing.assert_array_equal(c.prefix.origin,np.zeros(7))


@pytest.mark.parametrize('fault',['clock','feedback','limits'])
def test_query_fails_closed_on_inference_mutation_or_raw_joint_limit(monkeypatch,fault):
    c,state=bare_controller(monkeypatch)
    observation=module().validate_estimate(estimate(),state,episode_id='episode',target_id='L1',metadata=metadata())
    def predict(q,pose,confidence,*,metadata,history,history_times):
        if fault=='clock':state['time_s']+=1/240
        if fault=='feedback':state['q'][0]=.001
        return contract.JointActionBlock('block_relative_joint',np.full((20,7),4. if fault=='limits' else 0.),q,metadata)
    c.predictor=predict
    query_history(c,state,observation)
    with pytest.raises(ValueError):c._query(observation)
    assert c.prefix is None and not c.l2_queries[-1]['accepted']


def test_analytic_l3_calls_original_unchanged_contact_cycle_and_never_model(monkeypatch):
    m=module();c,state=bare_controller(monkeypatch)
    c.env.rows=[dict(q_command=np.zeros(7).tolist()) for _ in range(3)]
    c.prefix=None;c.control_owner='L3_ENTRY';calls=[]
    c.predictor=lambda *a,**k:pytest.fail('Model invoked after handoff')
    c._handoff_measurements=lambda:dict(passed=True)
    c.l2_handoffs=[dict(l2_handoff_success=True,entry=dict(physics_row=3,time_s=0.,command_q=np.zeros(7).tolist()),
                       query_count_at_handoff=0,l3_completed=False)]
    monkeypatch.setattr(m.FreshExpert,'_contact_cycle',lambda self:calls.append(self))
    c._contact_cycle()
    assert calls==[c] and c.l2_handoffs[0]['l3_completed'] and c.control_owner=='ANALYTIC_L3'
    with pytest.raises(ValueError,match='after L3'):c._query({})


def test_native_tick_keeps_full_super_chain_and_labels_even_when_it_fails(monkeypatch):
    m=module();c,state=bare_controller(monkeypatch);calls=[]
    def original_tick(self,command,subphase,**kwargs):
        calls.append((command,subphase,kwargs))
        self.env.rows.append(dict(q_command=command.tolist(),physical_violations=['force']))
        raise RuntimeError('Original physical gate')
    monkeypatch.setattr(m.FreshExpert,'_tick',original_tick)
    with pytest.raises(RuntimeError,match='physical gate'):c._tick(np.zeros(7),'m2_act_l2_free',active=False)
    assert len(calls)==1 and c.env.rows[0]['physical_violations']==['force']
    assert c.env.rows[0]['learned_policy'] and not c.env.rows[0]['expert_demonstration']
