"""A numerical preview rejection never authorizes a changed physical command."""
from copy import deepcopy

import numpy as np
import pytest

from test_ausc_given_contact_handoff import boundary,module as handoff,append
from sim.ausc_policy import m2_given_contact_controller as controller
from sim.auscultation_pi05_v6 import execution


def numerical_error():
    cls=getattr(execution,'FinishedStationaryRoundoffError',None)
    assert cls is not None,'Narrow native Finished numerical exception missing'
    return cls('Finished stationary output exceeds numerical roundoff bound')


def test_only_typed_clone_failure_defers_without_stop_setter_state_reset_or_same_row_retry(tmp_path,monkeypatch):
    c=boundary(tmp_path);assert c._handoff_ready(c._reference_observation())
    before=c._parking_live_snapshot();original=handoff().preview_stop
    monkeypatch.setattr(handoff(),'preview_stop',lambda *a,**kw:(_ for _ in ()).throw(numerical_error()))
    assert c._stop_for_handoff(('e','L1',0,'APPROACH')) is None
    assert c._parking_live_snapshot()==before and c.control_owner=='TEACHER_L2'
    assert not getattr(c,'_intent_dispatcher',None)
    rejected=c.l2_handoffs[-1]['parking_numeric_rejections'];assert len(rejected)==1
    assert rejected[0]['physics_row']==240 and rejected[0]['stop_setter_issued'] is False
    assert c._handoff_ready(c._reference_observation()) is False
    for _ in range(12):append(c,c.q)
    assert c._handoff_ready(c._reference_observation()) is True
    monkeypatch.setattr(handoff(),'preview_stop',original)
    assert c._stop_for_handoff(('e','L1',0,'APPROACH'))['passed']
    receipt=c.l2_handoffs[-1]
    assert receipt['parking_budget_start_physics_row']==240
    assert receipt['parking_budget_steps']==len(c.env.rows)-240<=1200


@pytest.mark.parametrize('error',[ValueError('geometry gate'),RuntimeError('native error')])
def test_geometric_derivative_and_untyped_native_errors_remain_fatal(tmp_path,monkeypatch,error):
    c=boundary(tmp_path);assert c._handoff_ready(c._reference_observation())
    before=c._parking_live_snapshot()
    monkeypatch.setattr(handoff(),'preview_stop',lambda *a,**kw:(_ for _ in ()).throw(error))
    with pytest.raises(type(error),match=str(error)):c._stop_for_handoff(('e','L1',0,'APPROACH'))
    assert c._parking_live_snapshot()==before and not c.l2_handoffs[-1].get('parking_numeric_rejections')


def test_numeric_deferral_cannot_extend_single_parking_budget(tmp_path,monkeypatch):
    c=boundary(tmp_path);assert c._handoff_ready(c._reference_observation())
    monkeypatch.setattr(handoff(),'preview_stop',lambda *a,**kw:(_ for _ in ()).throw(numerical_error()))
    assert c._stop_for_handoff(('e','L1',0,'APPROACH')) is None
    for _ in range(1200):append(c,c.q)
    with pytest.raises(Exception,match='single 5 second'):c._handoff_ready(c._reference_observation())


def test_numeric_deferral_regates_next_live_interior_pose(tmp_path,monkeypatch):
    c=boundary(tmp_path);assert c._handoff_ready(c._reference_observation())
    monkeypatch.setattr(handoff(),'preview_stop',lambda *a,**kw:(_ for _ in ()).throw(numerical_error()))
    assert c._stop_for_handoff(('e','L1',0,'APPROACH')) is None
    for _ in range(12):append(c,c.q)
    origin=c.env.state
    def outside():
        s=origin();s['tcp_base']=np.asarray(s['tcp_base']).copy()+np.array([.02,0,0]);return s
    c.env.state=outside
    assert c._handoff_ready(c._reference_observation()) is False


def test_same_typed_error_from_live_execution_is_fatal(tmp_path,monkeypatch):
    c=boundary(tmp_path);c._issue_intent(c.q.copy(),('e','L1',0,'APPROACH'),'given_l2_motion')
    monkeypatch.setattr(c.trajectory,'step',lambda:(_ for _ in ()).throw(numerical_error()))
    before=len(c.env.rows)
    with pytest.raises(execution.FinishedStationaryRoundoffError):c._advance_tick('given_l2_motion')
    assert len(c.env.rows)==before and not c.l2_handoffs[-1].get('parking_numeric_rejections')


def test_real_0153_counterexample_remains_rejected_with_same64ulp_bound():
    pytest.importorskip('ruckig');assert execution.FINISHED_ROUNDOFF_ULPS==64
    # Scalar J7 synchronized STOP counterexample independently isolated from
    # the actual Linux replay; robot captures remain on Linux.
    import ruckig
    r=ruckig.Ruckig(1,1/240);i=ruckig.InputParameter(1);o=ruckig.OutputParameter(1)
    i.current_position=[1.0024632797748667];i.current_velocity=[-.00015361940021556424];i.current_acceleration=[0.]
    i.target_position=i.current_position;i.target_velocity=[0.];i.target_acceleration=[0.]
    i.max_velocity=[.174999];i.max_acceleration=[.5];i.max_jerk=[1.49];i.minimum_duration=.1264993177084053
    assert r.update(i,o)==ruckig.Result.Working
    p=o.trajectory.profiles[0][0]
    assert p.p[-1]-p.pf==pytest.approx(3.196354292356318e-11,rel=1e-5)
    assert abs(p.p[-1]-p.pf)>64*np.finfo(float).eps*max(1.,abs(p.pf))


def test_reader_replays_rejection_and_refuses_forged_numeric_failure(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_dataset as data
    c=boundary(tmp_path);c.trajectory=execution.JointTrajectory(c.q,**handoff().PLANNING_LIMITS);c.trajectory.step()
    assert c._handoff_ready(c._reference_observation())
    original=handoff().preview_stop
    monkeypatch.setattr(handoff(),'preview_stop',lambda *a,**kw:(_ for _ in ()).throw(numerical_error()))
    assert c._stop_for_handoff(('e','L1',0,'APPROACH')) is None
    for _ in range(12):append(c,c.q)
    monkeypatch.setattr(handoff(),'preview_stop',original)
    assert c._handoff_ready(c._reference_observation())
    start=len(c.env.rows);assert c._stop_for_handoff(('e','L1',0,'APPROACH'))['passed']
    for index,event in enumerate(c.env.rows):event['time_s']=(index+1)/240.
    with pytest.raises(ValueError,match='reproduce'):
        data._numeric_preview_rejections(c.l2_handoffs[-1],c.env.rows,c.reference,start,len(c.env.rows))
