"""Raw target intent fidelity, not physical or learned-policy qualification."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest


def test_setter_owns_exactly_twelve_subsequent_ticks_and_live_readback():
    from sim.ausc_policy.m2_given_contact_intent import IntentDispatcher
    calls = []
    trajectory = SimpleNamespace(input=SimpleNamespace(target_position=[0.]*7))
    def setter(q):
        calls.append(q.copy()); trajectory.input.target_position = q.tolist()
    trajectory.set_target = setter
    d = IntentDispatcher(trajectory, reference_sha256='a'*64)
    d.issue(np.ones(7), time_s=1., physics_row=240,
            context=('e','L1',0,'FREE'), owner='TEACHER_L2', role='given_l2_motion')
    for tick in range(12):
        evidence = d.snapshot(time_s=1.+tick/240., physics_row=240+tick)
        assert evidence['native_tick_in_cell'] == tick
        assert evidence['setter']['issued_at_s'] == 1.
        assert evidence['live_target_q'] == [1.]*7
    assert len(calls) == 1
    with pytest.raises(ValueError, match='expired'):
        d.snapshot(time_s=1.05, physics_row=252)
    with pytest.raises(ValueError, match='grid'):
        d.issue(np.ones(7), time_s=243/240., physics_row=243,
                context=('e','L1',0,'FREE'), owner='TEACHER_L2', role='given_l2_motion')
    trajectory.input.target_position[0] = 9.
    with pytest.raises(ValueError, match='live'):
        d.snapshot(time_s=1., physics_row=240)


def test_raw_intent_versions_are_bound_to_given_metadata(tmp_path):
    from test_ausc_given_contact_pose import reference, metadata
    from sim.ausc_policy import m2_given_contact_pose as pose, m2_given_contact_act as act
    ref = reference(tmp_path); meta = metadata(ref)
    assert meta['action_semantics_id'] == act.ACTION_SEMANTICS_ID
    assert pose.CAPTURE_SCHEMA.endswith('_v2')
    assert act.DATASET_SCHEMA.endswith('_v2')
    bad = deepcopy(meta); bad.pop('action_semantics_id')
    with pytest.raises(ValueError, match='semantics'): pose.require_route(bad)


def test_dispatcher_capture_labels_are_raw_targets_and_applied_commands_remain_separate(tmp_path):
    from test_ausc_given_contact_capture import new_inputs
    from sim.ausc_policy import m2_given_contact_act as act
    ref, rows, audit, meta = new_inputs(tmp_path)
    samples = act.assemble_samples(rows, audit, meta)
    np.testing.assert_allclose(samples['actions'][0,:,0], .7+np.arange(2,22)*.002-.101)
    assert rows[1]['q_command'][0] == pytest.approx(.403)
    assert audit[24]['q_command'][0] != audit[24]['raw_intent']['setter']['target_q'][0]


@pytest.mark.parametrize('fault', ['missing','live','late','id','rawq','after','tick'])
def test_corrupt_setter_evidence_never_qualifies(tmp_path, fault):
    from test_ausc_given_contact_capture import new_inputs
    from sim.ausc_policy import m2_given_contact_act as act
    ref, rows, audit, meta = new_inputs(tmp_path)
    if fault == 'missing': audit[24].pop('raw_intent')
    elif fault == 'live': audit[24]['raw_intent']['live_target_q'][0] += .01
    elif fault == 'after': audit[24]['raw_intent']['live_target_after_step_q'][0] += .01
    elif fault == 'tick': audit[24]['raw_intent']['native_tick_in_cell'] = 1
    else:
        setter = audit[24]['raw_intent']['setter']
        setter[dict(late='issued_at_s',id='intent_id',rawq='target_q')[fault]] = 4
    with pytest.raises(ValueError): act.assemble_samples(rows,audit,meta)


def test_compact_applied_endpoint_must_match_native_command_audit(tmp_path):
    from test_ausc_given_contact_capture import new_inputs
    from sim.ausc_policy import m2_given_contact_act as act
    ref,rows,audit,meta=new_inputs(tmp_path)
    audit[35]['q_command'][0] += .001
    with pytest.raises(ValueError,match='endpoint'): act.assemble_samples(rows,audit,meta)


def test_setter_stage_is_actual_at_issue_but_immutable_during_natural_flip(tmp_path):
    from test_ausc_given_contact_capture import new_inputs
    from sim.ausc_policy import m2_given_contact_act as act
    import hashlib,json
    ref,rows,audit,meta=new_inputs(tmp_path)
    for i in range(24,36):
        setter=audit[i]['raw_intent']['setter']
        setter['context'][-1]='FREE'
        setter.pop('setter_sha256')
        setter['setter_sha256']=hashlib.sha256(json.dumps(setter,sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError,match='stage'): act.assemble_samples(rows,audit,meta)


@pytest.mark.parametrize('start,end',[(120,132),(24,264),(0,12)])
def test_valid_actual_owner_change_excludes_full_history_and_future_window(tmp_path,start,end):
    from test_ausc_given_contact_capture import new_inputs
    from sim.ausc_policy import m2_given_contact_act as act
    import hashlib,json
    ref,rows,audit,meta=new_inputs(tmp_path)
    for event in audit[start:end]:
        setter=event['raw_intent']['setter']
        setter['control_owner']='ACT_L2'; setter.pop('setter_sha256')
        setter['setter_sha256']=hashlib.sha256(json.dumps(setter,sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()
        for side in ('controller_before','controller_after'):
            event[side]['actual_control_owner']='ACT_L2'
        event['actual_control_owner']='ACT_L2'
    assert len(act.assemble_samples(rows,audit,meta)['states'])==0


@pytest.mark.parametrize('where',['controller_before','controller_after','event'])
def test_actual_owner_annotation_cannot_disagree_with_held_setter(tmp_path,where):
    from test_ausc_given_contact_capture import new_inputs
    from sim.ausc_policy import m2_given_contact_act as act
    ref,rows,audit,meta=new_inputs(tmp_path)
    selected=audit[120] if where=='event' else audit[120][where]
    selected['actual_control_owner']='ACT_L2'
    with pytest.raises(ValueError,match='actual owner'): act.assemble_samples(rows,audit,meta)


def test_natural_stage_transition_is_exclusion_not_corruption(tmp_path):
    from test_ausc_given_contact_capture import new_inputs
    from sim.ausc_policy import m2_given_contact_act as act
    ref, rows, audit, meta = new_inputs(tmp_path)
    audit[100]['controller_before']['actor']['phase'] = 'FREE'
    assert len(act.assemble_samples(rows,audit,meta)['states']) == 0


def test_wait_preserves_live_target_between_grid_boundaries(tmp_path):
    from test_ausc_given_contact_controller import bare_boundary
    from sim.ausc_policy.m2_given_contact_intent import IntentDispatcher
    c = bare_boundary(tmp_path); c.control_owner = 'ACT_L2'
    c.env.rows = [{}]*240; stamp = [1.]
    c.env.state = lambda: dict(time_s=stamp[0])
    c.trajectory = SimpleNamespace(input=SimpleNamespace(target_position=[.1]*7))
    calls=[]
    def setter(q): calls.append(q.copy()); c.trajectory.input.target_position=q.tolist()
    c.trajectory.set_target=setter
    c._intent_dispatcher=IntentDispatcher(c.trajectory,reference_sha256=c.reference.sha256)
    c._reference_observation=lambda: dict(context=('e','L1',0,'FREE'))
    c._handoff_measurements=lambda: dict(passed=False)
    def advance(role):
        c.env.rows.append({}); stamp[0]=len(c.env.rows)/240.
    c._advance_tick=advance
    c._wait_history_cell(c._reference_observation())
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [.1]*7)
    assert len(c.env.rows) == 252
