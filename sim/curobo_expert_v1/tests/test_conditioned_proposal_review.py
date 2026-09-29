"""Independent review regressions for conditioning and explicit release bridges."""
from types import SimpleNamespace

import numpy as np
import pytest

from sim.auscultation_expert_layout_v1.controller import EpisodeRejected
from sim.curobo_expert_v1 import controller as module


def actor(tmp_path, monkeypatch):
    c=module.ExpertController.__new__(module.ExpertController)
    c.generator_policy='curobo';c.sequence_index=1
    c.q=np.full(7,.001);observed=np.full(7,.0011);goal=np.full(7,.03)
    c.target=SimpleNamespace(name='L1');c.checker=object()
    c.env=SimpleNamespace(out=tmp_path,rows=[{} for _ in range(4)],last_contacts={},
        state=lambda:{'q':observed.copy(),'time_s':1.,'hall_N':0.,'compression_m':0.})
    c.released_prepose={'target':'R1','physics_row_end_exclusive':4,'time_s':1.}
    c.bridge_dir=tmp_path;c.launcher='unused';c.planner_calls=[];c.inter_point_routes=[]
    c.speed_scale=1.;c.joint_speed_limit=.25;c.free_collision_samples=0;c.free_motion_timing=[]
    c._hold=lambda *args:None
    native=np.linspace(observed+.0002,goal-.0002,5000)
    monkeypatch.setattr(module,'request_plan',lambda *args:(native.copy(),{'metadata':{'test_only':True}}, {'wall_time_s':.1}))
    canonical=np.vstack([np.zeros(7),np.full(7,.015),goal])
    row={'success':True,'transfer_q':canonical,'lower_staging_q':canonical[1]}
    return c,row,observed,goal,native


@pytest.mark.parametrize('failed_connector',[0,1,2])
def test_rdp_removed_endpoint_connectors_still_reject_before_execution(tmp_path,monkeypatch,failed_connector):
    c,row,observed,goal,native=actor(tmp_path,monkeypatch)
    expected=[(c.q.copy(),observed),(observed,native[0]),(native[-1],goal)]
    calls=[]
    def edge(a,b,*args):
        index=len(calls);calls.append((a.copy(),b.copy()))
        return {'clear':index!=failed_connector,'minimum_distance_m':0. if index==failed_connector else .003}
    monkeypatch.setattr(module,'checked_edge',edge)
    def cannot_move(*args):
        raise AssertionError('execution was reached after rejected endpoint connector')
    c._continuous_path=cannot_move
    with pytest.raises(EpisodeRejected,match='proposal failed'):
        c._move_to_prepose(row)
    assert len(c.env.rows)==4 and c.inter_point_routes==[]
    assert c.planner_calls[0]['proposal_accepted'] is False
    for actual,wanted in zip(calls,expected):
        np.testing.assert_array_equal(actual[0],wanted[0]);np.testing.assert_array_equal(actual[1],wanted[1])


def test_conditioned_edge_success_never_skips_final_command_gate_or_raw_evidence(tmp_path,monkeypatch):
    c,row,observed,goal,native=actor(tmp_path,monkeypatch)
    checked=[]
    def edge(a,b,*args):
        checked.append((a.copy(),b.copy()))
        return {'clear':True,'minimum_distance_m':.003}
    monkeypatch.setattr(module,'checked_edge',edge)
    class Checker:
        def __init__(self):self.calls=0
        def check(self,q,**kwargs):
            self.calls+=1
            return {'clear':self.calls<3,'minimum_distance_m':.003 if self.calls<3 else 0.}
    c.checker=Checker()
    def cannot_move(*args):
        raise AssertionError('any block moved before all actual commands passed')
    c._block=cannot_move
    with pytest.raises(EpisodeRejected,match='free command failed'):
        c._move_to_prepose(row)
    assert len(checked)==4  # three explicit connectors plus one conditioned chord
    assert c.checker.calls==3 and len(c.env.rows)==4 and c.free_collision_samples==0
    record=c.inter_point_routes[0]
    np.testing.assert_array_equal(record['joint_waypoints_q'],np.vstack([c.q,observed,native,goal]))
    assert record['proposal_conditioning']['original_knots']==5003
    assert record['proposal_conditioning']['retained_knots']==2
    assert 'conditioned geometric proposal' in record['edge_checks_scope']
    assert c.planner_calls[0]['original_proposal_metrics']['waypoint_count']==5000
    assert record['motion_complete'] is False and c.planner_calls[0]['motion_complete'] is False
