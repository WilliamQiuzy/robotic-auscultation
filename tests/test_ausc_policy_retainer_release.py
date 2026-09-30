"""CPU release gates over the real Fresh MRO; injected motion and sensor state."""
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy.fresh_expert import FreshExpert
from sim.auscultation_expert_layout_v1.controller import (
    EpisodeRejected, FRONT_GROUPS, FORBIDDEN_GROUPS,
)
from sim.auscultation_expert_layout_v1.low_elbow_controller import LowElbowController
from sim.curobo_expert_v1 import controller as curobo_controller
from sim.curobo_expert_v1.withdrawal import NormalWithdrawalMixin


GATES=('low_elbow_handoff','low_elbow_receipt','candidate_handoff','curobo_handoff',
       'normal_unload','normal_receipt')


class PlanningReached(RuntimeError):
    """Stop after a handoff gate to avoid running an external planner."""


def actor_fixture(tmp_path,monkeypatch,contacts,*,fresh=True):
    actor_class=FreshExpert if fresh else curobo_controller.ExpertController
    actor=actor_class.__new__(actor_class)
    actor.q=np.zeros(7);actor.normal_base=np.array([0.,0.,1.]);actor.rotation_base=np.eye(3)
    actor.pre_gap=.025;actor.speed_scale=1.;actor.sequence_index=1;actor.policy='checked_shortcut'
    actor.target=SimpleNamespace(name='R3',position_base=np.zeros(3))
    state=dict(q=np.zeros(7),time_s=1.,hall_N=0.,compression_m=0.)
    actor.env=SimpleNamespace(rows=[{}],last_contacts=dict(contacts),state=lambda:state,out=tmp_path)
    actor.released_prepose=dict(target='R2',time_s=1.,physics_row_end_exclusive=1)
    actor.inter_point_routes=[];actor.planner_calls=[];actor.checker=object()
    actor.bridge_dir=tmp_path;actor.launcher='not-executed'
    actor._normal_release_target='R3'
    actor.config=SimpleNamespace(release_force_limit_N=.2)
    actor.manager=SimpleNamespace(state=SimpleNamespace(success=True))
    actor.metrics=lambda observed:dict(gap_m=.025)
    events=[]
    actor._hold=lambda seconds,kind:events.append(kind)
    actor._pose_block=lambda *args,**kwargs:events.append('pose_block')
    actor._normal_segment=lambda gap,speed,kind:events.append(kind)
    actor._block=lambda commands,kind,**kwargs:events.append(kind)
    actor.kin_position_from_command=lambda:actor.normal_base*actor.pre_gap
    def planner(*args,**kwargs):
        events.append('planning')
        raise PlanningReached
    actor._plan_inter_point_bridge=planner
    monkeypatch.setattr(curobo_controller,'request_plan',planner)
    canonical=np.array([np.zeros(7),np.full(7,.01),np.full(7,.02)])
    row=dict(success=True,transfer_q=canonical,lower_staging_q=canonical[1])
    return actor,row,events


def invoke_gate(actor,row,gate):
    if gate=='low_elbow_handoff':
        return LowElbowController._move_to_prepose(actor,row)
    if gate=='low_elbow_receipt':
        actor.released_prepose=None
        return LowElbowController._withdraw_to_prepose(actor,row['transfer_q'])
    if gate in ('candidate_handoff','curobo_handoff'):
        actor.generator_policy='revised_shortcut' if gate=='candidate_handoff' else 'curobo'
        # The inherited Skill -> cuRobo -> Candidate path is intentionally used.
        return actor._move_to_prepose(row)
    if gate=='normal_unload':
        actor._normal_release_target=None
        actor.kin_position_from_command=lambda:np.zeros(3)
        return actor._smooth_normal_release()
    if gate=='normal_receipt':
        return actor._withdraw_to_prepose(row['transfer_q'])
    raise AssertionError(gate)


def assert_gate_allows(actor,row,events,gate):
    if gate.endswith('handoff'):
        with pytest.raises(PlanningReached):invoke_gate(actor,row,gate)
        assert events==['planning']
    else:
        invoke_gate(actor,row,gate)
        if gate=='normal_unload':
            assert actor._normal_release_target=='R3'
            assert events[:2]==['unload','released_retreat']
        else:
            assert actor.released_prepose['target']=='R3'


def test_fresh_resolves_real_normal_release_methods_and_seven_force_groups():
    assert FreshExpert._withdraw_to_prepose is NormalWithdrawalMixin._withdraw_to_prepose
    assert FreshExpert._smooth_normal_release is NormalWithdrawalMixin._smooth_normal_release
    assert 'retainer' in FreshExpert.total_contact_groups
    assert len(FreshExpert.total_contact_groups)==7
    assert curobo_controller.ExpertController.total_contact_groups==FRONT_GROUPS+FORBIDDEN_GROUPS
    assert len(curobo_controller.ExpertController.total_contact_groups)==6


@pytest.mark.parametrize('gate',GATES)
@pytest.mark.parametrize('contacts',[
    {'retainer':.2},{'retainer':4.},{'diaphragm':.11,'retainer':.11},
    {'retainer':float('nan')},
])
def test_each_fresh_release_or_handoff_gate_rejects_retainer_load(tmp_path,monkeypatch,gate,contacts):
    actor,row,events=actor_fixture(tmp_path,monkeypatch,contacts)
    with pytest.raises(EpisodeRejected,match='release|prepose|withdrawal'):
        invoke_gate(actor,row,gate)
    assert 'planning' not in events
    assert 'released_retreat' not in events
    assert 'normal_release_confirm' not in events
    if gate.endswith('receipt'):assert actor.released_prepose is None
    if gate=='normal_unload':assert actor._normal_release_target is None


@pytest.mark.parametrize('gate',GATES)
def test_fresh_keeps_original_point_two_newton_release_threshold(tmp_path,monkeypatch,gate):
    actor,row,events=actor_fixture(tmp_path,monkeypatch,{'retainer':.199999})
    assert_gate_allows(actor,row,events,gate)


@pytest.mark.parametrize('gate',GATES)
def test_legacy_default_still_uses_original_six_channels(tmp_path,monkeypatch,gate):
    actor,row,events=actor_fixture(tmp_path,monkeypatch,{'retainer':4.},fresh=False)
    assert 'total_contact_groups' not in vars(actor)
    assert_gate_allows(actor,row,events,gate)


@pytest.mark.parametrize('gate',GATES)
def test_legacy_original_contact_force_still_blocks_release(tmp_path,monkeypatch,gate):
    actor,row,events=actor_fixture(tmp_path,monkeypatch,{'diaphragm':.2},fresh=False)
    with pytest.raises(EpisodeRejected,match='release|prepose|withdrawal'):
        invoke_gate(actor,row,gate)
    assert 'planning' not in events
    assert 'released_retreat' not in events
