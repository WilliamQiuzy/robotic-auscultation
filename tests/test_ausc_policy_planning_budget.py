"""A completed search must not consume the subsequent safety-check budget."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

from sim.ausc_policy import fresh_expert as fresh
from sim.expert_quality_20260924.candidate_controller import CandidateController

MODE = {'mode': 'separate_search_validation_v1'}


def fixture(monkeypatch, setting, *, distance=.003, check_seconds=.4):
    clock=[0.]; monkeypatch.setattr(fresh.time,'monotonic',lambda:clock[0])
    value=object.__new__(fresh.FreshExpert)
    value.inter_point_budget=setting
    value.policy='checked_shortcut';value.sequence_index=1
    value.q=np.full(7,.02);value.target=NS(name='L3')
    value.inter_point_routes=[];moves=[];checked=[]
    state=dict(q=value.q.copy(),time_s=238.,hall_N=0.)
    value.env=NS(state=lambda:state,last_contacts={},rows=[])
    value.released_prepose=dict(target='R2',time_s=238.,physics_row_end_exclusive=0)
    def check(q,**kwargs):
        assert kwargs == dict(clearance=.003,allow_patient_contact=False,compression=0.)
        clock[0]+=check_seconds;checked.append(q.copy())
        return dict(clear=True,minimum_distance_m=distance)
    value.checker=NS(check=check)
    stage=np.full(7,.01);pre=np.full(7,.025)
    row=dict(success=True,transfer_q=np.array([np.zeros(7),stage,pre]),lower_staging_q=stage)
    def search(start,goal,deadline):
        assert deadline==30.
        clock[0]=27.
        return np.array([start,goal]),dict(reason='fixture',seconds=27.)
    value._plan_inter_point_bridge=search
    value._joint_segment=lambda *args:moves.append(args)
    value._execute_staged=lambda *args,**kwargs:moves.append(args)
    value._hold=lambda *args:None
    return value,row,moves,checked,clock


def test_late_valid_search_has_its_own_complete_validation_budget(monkeypatch):
    value,row,moves,checked,clock=fixture(monkeypatch,MODE)
    CandidateController._move_to_prepose(value,row)
    assert clock[0]>30. and moves and len(checked)>=8
    record=value.inter_point_routes[0]
    assert record['planning_accepted'] and record['motion_complete']
    assert record['search_wall_budget_s']==30.
    assert record['validation_wall_budget_s']==30.
    assert record['planning_simulation_time_s']==238.


@pytest.mark.parametrize('setting,distance,seconds',[(None,.003,.4),(MODE,.0029,.4),(MODE,.003,31.)])
def test_legacy_timeout_collision_and_validation_timeout_still_reject(monkeypatch,setting,distance,seconds):
    value,row,moves,checked,clock=fixture(monkeypatch,setting,distance=distance,check_seconds=seconds)
    with pytest.raises((RuntimeError,ValueError,TimeoutError)):
        CandidateController._move_to_prepose(value,row)
    assert not moves and not value.env.rows
    assert value.inter_point_routes[0]['planning_accepted'] is False


@pytest.mark.parametrize('bad',[{},True,{'mode':'unknown'},{'mode':'separate_search_validation_v1','validation_s':10000}])
def test_budget_contract_rejects_unbounded_or_unknown_settings(bad):
    assert hasattr(fresh,'inter_point_budget_settings')
    with pytest.raises(ValueError):fresh.inter_point_budget_settings(bad)
