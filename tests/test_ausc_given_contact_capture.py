from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_ausc_given_contact_pose import inputs, module as pose_module


def given_ref(tmp_path):
    point, plan, patient = inputs(tmp_path)
    # Use a real declared TRAIN identity; synthetic payload remains unqualified.
    value = json.loads(open(patient['file']).read()); value['identity'] = 'p000439'
    from test_ausc_given_contact_pose import bound
    patient = bound(tmp_path, 'patient.json', value)
    value = json.loads(open(plan['file']).read())
    value['patient_provenance']['identity'] = 'p000439'; value['patient_provenance']['binding'] = patient
    plan = bound(tmp_path, 'plan.json', value); point['manifest_ref'] = patient
    return pose_module().build_given_reference(point, plan, patient, np.eye(4), 0.)


def new_inputs(tmp_path, n=22):
    from test_ausc_policy_m2_act import inputs as old_inputs
    from sim.ausc_policy import m2_contract
    ref = given_ref(tmp_path); rows, audit, meta = old_inputs(n)
    meta['target_id'] = 'L1'; meta = pose_module().route_metadata(meta, ref)
    for row in rows:
        row.update(target_id='L1', target_source='given_contact_pose', target_confidence=1.,
            target_estimate_source_id='given-contact-'+ref.sha256, estimated_target_base=ref.pose_base)
        row['state17'] = m2_contract.encode_state(row['measured_q'], ref.pose_base, 1., metadata=meta)
    for event in audit:
        event['actual_control_owner']='TEACHER_L2'
        for side in ('controller_before', 'controller_after'):
            value = event[side]; value.update(target='L1', subphase='given_l2_motion')
            value['actual_control_owner']='TEACHER_L2'
            value['actor'].update(rule_id=pose_module().STAGE_RULE, source=pose_module().NATIVE_SOURCE,
                reference_sha256=ref.sha256, reference_revision=0, role='L2')
    from sim.ausc_policy.m2_given_contact_intent import IntentDispatcher
    trajectory = SimpleNamespace(input=SimpleNamespace(target_position=[0.]*7))
    trajectory.set_target = lambda q: setattr(trajectory.input,'target_position',q.tolist())
    dispatcher = IntentDispatcher(trajectory,reference_sha256=ref.sha256)
    for i,event in enumerate(audit):
        if i%12 == 0:
            dispatcher.issue(np.full(7,.7+(i//12)*.002),time_s=i/240.,physics_row=i,
                context=(meta['episode_id'],'L1',0,'APPROACH'),owner='TEACHER_L2',role='given_l2_motion')
        evidence = dispatcher.snapshot(time_s=i/240.,physics_row=i)
        evidence['live_target_after_step_q'] = evidence['live_target_q'].copy()
        event.update(raw_intent=evidence,q_command=[.4+(i//12-1)*.003]*7)
    return ref, rows, audit, meta


def test_causal_windows_use_new_native_reference_and_all_241_plus_13_ticks(tmp_path):
    from sim.ausc_policy import m2_given_contact_act as act, m2_act as old
    ref, rows, audit, meta = new_inputs(tmp_path)
    samples = act.assemble_samples(rows, audit, meta)
    assert len(samples['states']) == 1
    assert samples['derived_index']['windows'][0]['inclusive_native_ticks'] == [23, 263]
    assert samples['derived_index']['windows'][0]['history_native_ticks'] == [11, 23]
    assert samples['schema'] == act.DATASET_SCHEMA
    with pytest.raises(ValueError): old.assemble_samples(rows, audit, meta)


@pytest.mark.parametrize('fault', ['owner', 'role', 'reference', 'revision', 'source', 'clock', 'padding'])
def test_cross_owner_reference_or_clock_never_becomes_l2_window(tmp_path, fault):
    from sim.ausc_policy import m2_given_contact_act as act
    ref, rows, audit, meta = new_inputs(tmp_path)
    actor = audit[251]['controller_before']['actor']
    if fault == 'owner': actor.update(owner='L3', role='L3_ALIGN', phase=None)
    if fault == 'role': audit[251]['controller_before']['subphase'] = 'given_l2_park'
    if fault == 'reference': actor['reference_sha256'] = '0'*64
    if fault == 'revision': actor['reference_revision'] = 1
    if fault == 'source': actor['source'] = 'native_measured_q_estimated_target_v1'
    if fault == 'clock': actor['sampled_at_s'] += .001
    if fault == 'padding': rows.pop()
    if fault == 'owner':
        assert len(act.assemble_samples(rows, audit, meta)['states']) == 0
    else:
        with pytest.raises(ValueError): act.assemble_samples(rows, audit, meta)


def test_actual_align_owner_is_l3_before_first_command_even_contact_false(tmp_path):
    from sim.ausc_policy import expert_capture
    from test_ausc_policy_compact_expert_capture import compact_binding
    from test_ausc_policy_expert_capture import GEOMETRY, Env
    ref = given_ref(tmp_path); binding = compact_binding()
    meta = binding['compact']['metadata']; meta['target_id'] = 'L1'
    meta = pose_module().route_metadata(meta, ref)
    binding['compact'].update(metadata=meta, estimated_target_base=ref.pose_base.tolist(),
        target_confidence=1., target_estimate_source_id='given-contact-'+ref.sha256)
    capture = expert_capture.create_expert_capture(tmp_path/'capture', binding=binding, geometry_metadata=GEOMETRY)
    env = Env(); env.raw_capture = capture
    controller = SimpleNamespace(env=env, target=SimpleNamespace(name='L1'), control_owner='ANALYTIC_L3_ALIGN',
        task_id=pose_module().TASK_ID, _contact_active=False,
        manager=SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='APPROACH'))))
    value = expert_capture.controller_labels(controller, 'L3_ALIGN', False, False)
    assert value['actor']['owner'] == 'L3' and value['actor']['role'] == 'L3_ALIGN'
    assert value['actor']['reference_sha256'] == ref.sha256
    assert (capture.output/'given_reference.json').read_bytes().strip() == ref.payload
    controller.control_owner = 'L2_STOP'
    assert expert_capture.controller_labels(controller, 'given_l2_park', False, False)['actor']['owner'] == 'PREPARATION'
    capture.close()


def test_old_controller_cannot_relabel_precise_route_as_new_native_capture(tmp_path):
    from sim.ausc_policy import expert_capture
    from test_ausc_policy_compact_expert_capture import compact_binding
    from test_ausc_policy_expert_capture import GEOMETRY, Env
    ref = given_ref(tmp_path); binding = compact_binding(); settings = binding['compact']
    settings['metadata']['target_id'] = 'L1'
    settings.update(metadata=pose_module().route_metadata(settings['metadata'], ref),
        estimated_target_base=ref.pose_base.tolist(), target_confidence=1.,
        target_estimate_source_id='given-contact-'+ref.sha256)
    capture = expert_capture.create_expert_capture(tmp_path/'capture', binding=binding, geometry_metadata=GEOMETRY)
    env = Env(); env.raw_capture = capture
    old = SimpleNamespace(env=env, target=SimpleNamespace(name='L1'),
        manager=SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='APPROACH'))))
    with pytest.raises(ValueError, match='new controller'):
        expert_capture.controller_labels(old, 'transfer', False, False)
    capture.close()


def test_given_config_builder_preserves_original_noise_route_and_uses_new_teacher(tmp_path, monkeypatch):
    from test_ausc_policy_m2_pilot_config import fixture
    from sim.ausc_policy import m2_pilot_config as config
    refs, plan, save = fixture(tmp_path, monkeypatch)
    row = json.loads(open(plan['file']).read())
    row.update(command_rotation_world=np.diag([1., -1., -1.]).tolist(), command_normal_world=[0., 0., 1.])
    plan = save('L1_fresh_plan.json', row)
    args = dict(**refs, fresh_plan=plan, episode_id='given-test')
    old = config.build_compact_expert(**args, noise_seed=2)
    new = config.build_given_contact_expert(**args)
    from sim.ausc_policy import m2_given_contact_handoff as handoff
    route=new['config']['m2_given_contact_control']
    assert route['handoff_policy_id']==route['metadata']['handoff_policy_id']==handoff.POLICY_ID
    assert new['receipt']['handoff_policy_id']==handoff.POLICY_ID
    assert new['config']['fresh_expert']['raw_multimodal_capture']['compact']['code_bindings']['interior_handoff']==config.hash_file(handoff.__file__)
    assert 'm2_given_contact_control' not in old['config']
    route = new['config']['m2_given_contact_control']
    assert route['metadata']['target_source'] == 'given_contact_pose'
    assert new['receipt']['task_id'] == pose_module().TASK_ID
    assert old['config']['fresh_expert']['raw_multimodal_capture']['compact']['metadata']['target_source'] == 'gt_plus_measured_noise'
    assert new['config']['posture_limits'] == old['config']['posture_limits']


def test_native_fresh_tick_records_actual_align_owner_and_roundtrips_reference(tmp_path, monkeypatch):
    from sim.ausc_policy import fresh_expert, expert_capture, training_capture
    from test_ausc_policy_compact_expert_capture import compact_binding
    from test_ausc_policy_expert_capture import GEOMETRY, Env, Backend
    ref = given_ref(tmp_path); binding = compact_binding(); settings = binding['compact']
    settings['metadata']['target_id'] = 'L1'
    settings.update(metadata=pose_module().route_metadata(settings['metadata'], ref),
        estimated_target_base=ref.pose_base.tolist(), target_confidence=1.,
        target_estimate_source_id='given-contact-'+ref.sha256)
    capture = expert_capture.create_expert_capture(tmp_path/'capture', binding=binding, geometry_metadata=GEOMETRY)
    env = Env(); env.raw_capture = capture; capture.start(env, backend_factory=Backend)
    controller = fresh_expert.FreshExpert.__new__(fresh_expert.FreshExpert)
    controller.env=env;controller.target=SimpleNamespace(name='L1');controller.task_id=pose_module().TASK_ID
    controller.control_owner='ANALYTIC_L3_ALIGN';controller._contact_active=False
    controller.manager=SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='APPROACH')))
    def original(self, command, subphase, **kwargs):
        before = len(self.env.rows); self.env.advance()
        self.env.rows[-1].update(q_command=command.tolist(), target='L1', subphase=subphase)
        capture.after_physics(env, before)
        return env.state()
    monkeypatch.setattr(fresh_expert.SkillExpert, '_tick', original)
    for _ in range(24): fresh_expert.FreshExpert._tick(controller, np.zeros(7), 'L3_ALIGN', active=False)
    capture.close()
    result = capture.result(24, execution_completed=True)
    assert result['compact_capture_complete']
    assert result['schema'] == 'm2_given_contact_compact_expert_capture_v2'
    assert result['reference_sha256'] == ref.sha256
    records = list(training_capture.iter_training_records(capture.output/'compact'))
    assert len(records) == 2 and records[0]['control_owner'] == 'L3'
    assert records[0]['target_source'] == 'given_contact_pose'
    import gzip
    with gzip.open(capture.output/'native_audit.jsonl.gz', 'rt') as stream: event = json.loads(next(stream))
    assert event['q_command'] == [0.]*7
    assert event['raw_intent'] is None
    manifest = json.loads((capture.output/'compact/manifest.json').read_text())
    from sim.ausc_policy.m2_given_contact_intent import ACTION_SEMANTICS_ID
    assert manifest['schema'] == 'm2_given_compact_raw_target_audit_v5'
    assert manifest['action_semantics_id'] == ACTION_SEMANTICS_ID
    assert event['control_owner'] == 'L3'
    assert event['controller_before']['actor']['role'] == 'L3_ALIGN'
    assert event['controller_after']['actor']['role'] == 'L3_ALIGN'
    assert event['actual_control_owner']=='ANALYTIC_L3_ALIGN'
    assert event['controller_before']['actual_control_owner']=='ANALYTIC_L3_ALIGN'
    assert event['controller_after']['actual_control_owner']=='ANALYTIC_L3_ALIGN'


@pytest.mark.parametrize('offgrid_stop,interior_policy,stale_pose',[
    (False,False,False),(True,False,False),(False,True,False),(True,True,False),(True,True,True)])
def test_actual_given_dispatcher_evidence_survives_fresh_tick_and_compact_writer(tmp_path,monkeypatch,offgrid_stop,interior_policy,stale_pose):
    from sim.ausc_policy import fresh_expert,expert_capture,training_capture,m2_given_contact_controller as controller
    from sim.ausc_policy.m2_given_contact_intent import validate_native_intents
    from test_ausc_policy_compact_expert_capture import compact_binding
    from test_ausc_policy_expert_capture import GEOMETRY,Env,Backend
    ref=given_ref(tmp_path); binding=compact_binding(); settings=binding['compact']
    settings['metadata']['target_id']='L1'
    settings.update(metadata=pose_module().route_metadata(settings['metadata'],ref),
        estimated_target_base=ref.pose_base.tolist(),target_confidence=1.,target_estimate_source_id='given-contact-'+ref.sha256)
    from sim.ausc_policy import m2_given_contact_handoff as handoff
    if interior_policy:settings['metadata']['handoff_policy_id']=handoff.POLICY_ID
    capture=expert_capture.create_expert_capture(tmp_path/'capture',binding=binding,geometry_metadata=GEOMETRY)
    env=Env();env.base=np.eye(4)
    env.value.update(tcp_base=np.array([.2,.3,.4]),rotation_world=__import__('scipy').spatial.transform.Rotation.from_euler('z',.3).as_matrix())
    env.raw_capture=capture; capture.start(env,backend_factory=Backend)
    c=controller.GivenContactExpert.__new__(controller.GivenContactExpert)
    c.env=env; c.target=SimpleNamespace(name='L1'); c.reference=ref; c.metadata=settings['metadata']
    c.control_owner='TEACHER_L2'; c._contact_active=False; c.q=np.zeros(7)
    c.manager=SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='APPROACH')))
    c._deadline=lambda: None
    c.trajectory=SimpleNamespace(input=SimpleNamespace(target_position=[0.]*7))
    c.trajectory.set_target=lambda q:setattr(c.trajectory.input,'target_position',q.tolist())
    if interior_policy:
        # This transport-only fixture still needs a complete typed input for
        # the explicit v3 STOP profile gate; it is not real Ruckig evidence.
        from test_ausc_given_contact_handoff import FixtureTrajectory
        c.trajectory=FixtureTrajectory(c.q,dt=handoff.DT,**handoff.PLANNING_LIMITS)
    c.trajectory.step=lambda:np.full(7,(len(env.rows)+1)*.001)
    def original(self,command,subphase,**kwargs):
        before=len(env.rows); env.advance()
        env.rows[-1].update(q_command=command.tolist(),target='L1',subphase=subphase,tcp_speed_m_s=0.)
        capture.after_physics(env,before)
        return env.state()
    monkeypatch.setattr(fresh_expert.SkillExpert,'_tick',original)
    c._block=lambda commands,role,**kwargs:[c._tick(q,role) for q in commands]
    if stale_pose:
        original_labels=fresh_expert.controller_labels
        def stale_labels(controller,subphase,*args):
            labels=original_labels(controller,subphase,*args)
            if subphase=='given_l2_park' and len(controller.env.rows)==14:
                labels['parking_observed_pose']['physics_row']-=1
                labels['parking_observed_pose']['time_s']-=1/240.
            return labels
        monkeypatch.setattr(fresh_expert,'controller_labels',stale_labels)
    context=(settings['metadata']['episode_id'],'L1',0,'FREE')
    for i in range(24):
        if i%12==0: c._issue_intent(np.full(7,.08),context,'given_l2_motion')
        if offgrid_stop and i==13:
            if interior_policy:
                handoff.apply_stop_profile(c.trajectory,expected_input=handoff.input_snapshot(c.trajectory))
            c.control_owner='L2_STOP'
            c._issue_intent(np.zeros(7),context,'given_l2_park',stop=True)
        if stale_pose and i==13:
            with pytest.raises(ValueError,match='Actual full parking observation differs'):
                c._advance_tick(c._dispatcher().current['execution_role'])
            break
        c._advance_tick(c._dispatcher().current['execution_role'])
    capture.close();result=capture.result(len(env.rows),execution_completed=not stale_pose)
    if stale_pose:
        assert result['compact_capture_complete'] is False
        assert any('Actual full parking observation differs' in str(error) for error in result['errors'])
        return
    assert result['compact_capture_complete']
    import gzip
    with gzip.open(capture.output/'native_audit.jsonl.gz','rt') as stream: audit=[json.loads(x) for x in stream]
    validate_native_intents(audit,settings['metadata'])
    records=list(training_capture.iter_training_records(capture.output/'compact'))
    np.testing.assert_array_equal(records[0]['q_command'],np.full(7,.024))
    assert audit[12]['raw_intent']['setter']['target_q']==[.08]*7
    assert audit[12]['raw_intent']['setter']['issued_at_s']==.05
    assert audit[12]['actual_control_owner']=='TEACHER_L2'
    assert audit[12]['controller_before']['actual_control_owner']=='TEACHER_L2'
    if interior_policy:
        helper=str(__import__('pathlib').Path(handoff.__file__).resolve())
        assert helper in capture.sources
        if offgrid_stop:
            assert audit[13]['parking_pose_before']['physics_row']==13
            assert audit[13]['parking_pose_after']['physics_row']==14
            np.testing.assert_array_equal(np.asarray(audit[13]['parking_pose_after']['pose_base'])[:3,:3],env.value['rotation_world'])
            assert capture._last_parking_pose==audit[-1]['parking_pose_after']==c._actual_parking_pose()
    assert audit[12]['controller_after']['actual_control_owner']=='TEACHER_L2'
    if offgrid_stop:
        assert audit[12]['raw_intent']['setter']['offgrid_stop'] is False
        assert audit[12]['q_command']==[13*.001]*7
        assert audit[13]['raw_intent']['setter']['offgrid_stop'] is True
        assert audit[13]['raw_intent']['setter']['target_q']==[0.]*7
        assert audit[13]['raw_intent']['native_tick_in_cell']==0
        assert audit[13]['actual_control_owner']=='L2_STOP'
        assert audit[13]['controller_before']['actual_control_owner']=='L2_STOP'
        assert audit[13]['controller_after']['actual_control_owner']=='L2_STOP'
        assert audit[13]['q_command']==[14*.001]*7
    else:
        assert audit[23]['raw_intent']['native_tick_in_cell']==11


def test_excluded_history_wait_does_not_reset_the_causal_stage_latch(tmp_path,monkeypatch):
    from sim.ausc_policy import expert_capture,m2_contract
    from test_ausc_policy_compact_expert_capture import compact_binding
    from test_ausc_policy_expert_capture import GEOMETRY,Env
    ref=given_ref(tmp_path); binding=compact_binding(); settings=binding['compact']
    settings['metadata']['target_id']='L1'
    settings.update(metadata=pose_module().route_metadata(settings['metadata'],ref),
        estimated_target_base=ref.pose_base.tolist(),target_confidence=1.,target_estimate_source_id='given-contact-'+ref.sha256)
    capture=expert_capture.create_expert_capture(tmp_path/'capture',binding=binding,geometry_metadata=GEOMETRY)
    env=Env(); env.raw_capture=capture
    c=SimpleNamespace(env=env,target=SimpleNamespace(name='L1'),task_id=pose_module().TASK_ID,
        control_owner='TEACHER_L2',_contact_active=False,
        manager=SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='APPROACH'))))
    distance=[.08]
    monkeypatch.setattr(m2_contract,'encode_state',lambda *a,**k:np.r_[np.zeros(7),distance[0],np.zeros(9)])
    assert expert_capture.controller_labels(c,'given_l2_history_wait',False,False)['actor']['phase'] is None
    distance[0]=.1
    assert expert_capture.controller_labels(c,'given_l2_motion',False,False)['actor']['phase']=='APPROACH'
    capture.close()
