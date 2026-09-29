import importlib
import numpy as np
import pytest


@pytest.fixture
def api():
    try: module=importlib.import_module('sim.auscultation_vla_v2.chunks')
    except ModuleNotFoundError: module=None
    assert module is not None, 'interruptible chunk gate is not implemented'
    return module


def proposal(api, seq=1, observed=10., ready=10.08, revision=4, phase='FREE', n=10, prefix=2):
    return api.Chunk(sequence=seq, observed_at_s=observed, ready_at_s=ready,
        target_revision=revision, phase=phase, action_period_s=.05,
        actions=np.zeros((n,7)), execution_prefix=prefix)


def test_only_bounded_prefix_can_be_executed(api):
    gate=api.ChunkGate(max_observation_age_s=.2, max_prefix_s=.1)
    assert gate.accept(proposal(api),now_s=10.08,target_revision=4,phase='FREE').status=='accepted'
    assert gate.next(force_safe=True,visible=True,now_s=10.08,feedback_time_s=10.08,target_revision=4,phase='FREE').status=='action'
    assert gate.next(force_safe=True,visible=True,now_s=10.13,feedback_time_s=10.13,target_revision=4,phase='FREE').status=='action'
    assert gate.next(force_safe=True,visible=True,now_s=10.18,feedback_time_s=10.18,target_revision=4,phase='FREE').status=='replan'


def test_delayed_tick_does_not_replay_missed_actions(api):
    gate=api.ChunkGate(max_observation_age_s=.5, max_prefix_s=.1)
    gate.accept(proposal(api),now_s=10.08,target_revision=4,phase='FREE')
    assert gate.next(force_safe=True,visible=True,now_s=10.19,feedback_time_s=10.19,target_revision=4,phase='FREE').status=='replan'


@pytest.mark.parametrize('kwargs,reason', [
    ({'target_revision':5,'phase':'FREE'},'target_changed'),
    ({'target_revision':4,'phase':'CONTACT'},'phase_changed'),
    ({'target_revision':4,'phase':'FREE','force_safe':False},'force_unsafe'),
    ({'target_revision':4,'phase':'FREE','visible':False},'target_lost'),
])
def test_fresh_feedback_invalidates_queue_and_cannot_resume_old_chunk(api,kwargs,reason):
    gate=api.ChunkGate(max_observation_age_s=.2,max_prefix_s=.1)
    gate.accept(proposal(api),now_s=10.08,target_revision=4,phase='FREE')
    result=gate.next(now_s=10.09,feedback_time_s=10.09,**dict({"force_safe":True,"visible":True},**kwargs))
    assert result.status=='stop' and result.reason==reason and result.action is None
    assert gate.next(force_safe=True,visible=True,now_s=10.10,feedback_time_s=10.10,target_revision=4,phase='FREE').action is None


def test_new_target_during_inference_rejects_delayed_response(api):
    gate=api.ChunkGate(max_observation_age_s=.2,max_prefix_s=.1)
    result=gate.accept(proposal(api),now_s=10.08,target_revision=5,phase='FREE')
    assert result.status=='rejected' and result.reason=='target_changed'


def test_stale_response_does_not_keep_old_queue_running(api):
    gate=api.ChunkGate(max_observation_age_s=.2,max_prefix_s=.1)
    gate.accept(proposal(api),now_s=10.08,target_revision=4,phase='FREE')
    bad=proposal(api,seq=2,observed=9.8)
    assert gate.accept(bad,now_s=10.09,target_revision=4,phase='FREE').reason=='stale_observation'
    assert gate.next(force_safe=True,visible=True,now_s=10.10,feedback_time_s=10.10,target_revision=4,phase='FREE').action is None


def test_out_of_order_responses_and_reused_sequence_rejected(api):
    gate=api.ChunkGate(max_observation_age_s=.2,max_prefix_s=.1)
    gate.accept(proposal(api,seq=2),now_s=10.08,target_revision=4,phase='FREE')
    assert gate.accept(proposal(api,seq=1),now_s=10.09,target_revision=4,phase='FREE').reason=='old_sequence'
    assert gate.accept(proposal(api,seq=2),now_s=10.09,target_revision=4,phase='FREE').reason=='old_sequence'


@pytest.mark.parametrize('change', [
    lambda c:setattr(c,'actions',np.full((10,7),np.nan)),
    lambda c:setattr(c,'actions',np.zeros((10,1))),
    lambda c:setattr(c,'execution_prefix',10),
    lambda c:setattr(c,'execution_prefix',True),
    lambda c:setattr(c,'action_period_s',0.),
    lambda c:setattr(c,'action_period_s',.04),
    lambda c:setattr(c,'ready_at_s',9.),
    lambda c:setattr(c,'ready_at_s',11.),
    lambda c:setattr(c,'observed_at_s',float('nan')),
])
def test_malformed_chunk_never_yields_action(api,change):
    gate=api.ChunkGate(max_observation_age_s=.2,max_prefix_s=.1)
    chunk=proposal(api);change(chunk)
    result=gate.accept(chunk,now_s=10.08,target_revision=4,phase='FREE')
    assert result.status=='rejected'
    assert gate.next(force_safe=True,visible=True,now_s=10.09,feedback_time_s=10.09,target_revision=4,phase='FREE').action is None


def test_same_tick_cannot_consume_two_actions(api):
    gate=api.ChunkGate(max_observation_age_s=.2,max_prefix_s=.1)
    gate.accept(proposal(api),now_s=10.08,target_revision=4,phase='FREE')
    assert gate.next(force_safe=True,visible=True,now_s=10.08,feedback_time_s=10.08,target_revision=4,phase='FREE').status=='action'
    assert gate.next(force_safe=True,visible=True,now_s=10.09,feedback_time_s=10.09,target_revision=4,phase='FREE').status=='wait'


def test_rejected_target_revision_cannot_be_reintroduced_with_older_response(api):
    gate=api.ChunkGate(max_observation_age_s=.2,max_prefix_s=.1)
    gate.accept(proposal(api),now_s=10.08,target_revision=4,phase='FREE')
    gate.next(force_safe=True,visible=True,now_s=10.09,feedback_time_s=10.09,target_revision=5,phase='FREE')
    assert gate.accept(proposal(api,seq=2),now_s=10.10,target_revision=5,phase='FREE').status=='rejected'


def test_original_sensor_timestamp_expires_during_prefix(api):
    gate=api.ChunkGate(max_observation_age_s=.1)
    gate.accept(proposal(api),now_s=10.08,target_revision=4,phase='FREE')
    assert gate.next(force_safe=True,visible=True,now_s=10.08,feedback_time_s=10.08,target_revision=4,phase='FREE').status=='action'
    assert gate.next(force_safe=True,visible=True,now_s=10.13,feedback_time_s=10.13,target_revision=4,phase='FREE').reason=='stale_observation'


def test_stale_force_feedback_clears_queue(api):
    gate=api.ChunkGate()
    gate.accept(proposal(api),now_s=10.08,target_revision=4,phase='FREE')
    assert gate.next(force_safe=True,visible=True,now_s=10.09,feedback_time_s=10.,target_revision=4,phase='FREE').reason=='stale_feedback'
    assert gate.next(force_safe=True,visible=True,now_s=10.10,feedback_time_s=10.10,target_revision=4,phase='FREE').action is None


def test_missing_an_in_prefix_deadline_stops(api):
    gate=api.ChunkGate(max_observation_age_s=.5,max_prefix_s=.2)
    gate.accept(proposal(api,prefix=4),now_s=10.08,target_revision=4,phase='FREE')
    assert gate.next(force_safe=True,visible=True,now_s=10.18,feedback_time_s=10.18,target_revision=4,phase='FREE').reason=='missed_command_deadline'


@pytest.mark.parametrize('phase,channel,value', [('FREE',0,.021),('FREE',3,.101),('CONTACT',0,.0021),('CONTACT',3,.011),('FREE',6,1.),('HOLD',6,4.1)])
def test_action_schema_and_phase_bounds(api,phase,channel,value):
    gate=api.ChunkGate(); chunk=proposal(api,phase=phase)
    chunk.actions[0,channel]=value
    assert gate.accept(chunk,now_s=10.08,target_revision=4,phase=phase).reason=='action_bounds'


def test_measured_servo_offset_is_a_proposal_not_an_execution_certificate(api):
    gate=api.ChunkGate();chunk=proposal(api,phase='HOLD')
    chunk.actions[:,0]=.0004;chunk.actions[:,3]=.0022;chunk.actions[:,6]=4.
    assert gate.accept(chunk,now_s=10.08,target_revision=4,phase='HOLD').status=='accepted'
    result=gate.next(now_s=10.08,feedback_time_s=10.08,target_revision=4,phase='HOLD',force_safe=True,visible=True)
    assert result.reason=='proposal_requires_executor_checks'
    np.testing.assert_array_equal(result.action,chunk.actions[0])


def test_clock_regression_clears_queue(api):
    gate=api.ChunkGate()
    gate.accept(proposal(api),now_s=10.08,target_revision=4,phase='FREE')
    assert gate.next(force_safe=True,visible=True,now_s=10.07,feedback_time_s=10.07,target_revision=4,phase='FREE').reason=='invalid_clock'


def test_new_sequence_cannot_reintroduce_older_sensor_observation(api):
    gate=api.ChunkGate(max_observation_age_s=.5)
    gate.accept(proposal(api),now_s=10.08,target_revision=4,phase='FREE')
    assert gate.accept(proposal(api,seq=2,observed=9.99),now_s=10.09,target_revision=4,phase='FREE').reason=='older_observation'


def test_omitted_current_safety_flags_never_authorize_motion(api):
    gate=api.ChunkGate()
    gate.accept(proposal(api),now_s=10.08,target_revision=4,phase='FREE')
    result=gate.next(now_s=10.08,feedback_time_s=10.08,target_revision=4,phase='FREE')
    assert result.status=='stop' and result.action is None
