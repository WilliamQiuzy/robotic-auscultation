"""Actual inherited MRO with one mocked physical parent; no Isaac launch."""
from dataclasses import dataclass
from types import SimpleNamespace as S
from unittest.mock import patch
import numpy as np
import pytest
from sim.auscultation_pi05_v7 import numerical_learned_controller as m
from sim.auscultation_pi05_v7.numerical_command_gate import CertificateTracker
from sim.auscultation_pi05_v6.observed_supervisor import ObservedContactSupervisorMixin
from sim.curobo_expert_v1.controller import ExpertController as PhysicalParent
from sim.auscultation_pi05_v7.tests.test_numerical_command_gate import actual_fixture

@dataclass
class Packet:sample_time_s:float=0.


def fixture(tmp_path,actual=False):
    obj=m.NumericalLearnedController.__new__(m.NumericalLearnedController);calls=[]
    obj._numerical_tracker=CertificateTracker();obj._numerical_buffer=[];obj._numerical_event_count=0
    obj._numerical_interventions=0;obj._numerical_compared=0;obj._numerical_ledger_initialized=False
    obj.joint_speed_limit=.175;obj._contact_active=False;obj.command_collision_samples=0;obj.observed_collision_samples=0
    obj.trajectory=None;obj._last_tcp_measurement=None;obj._last_observed_hall_packet=Packet()
    history=np.zeros((3,7));command=np.zeros(7)
    if actual:
        q,certificate=actual_fixture();history=q[:3];command=q[3]
        profiles=[]
        for axis in range(7):
            if axis==3:
                p=certificate['axis_sections']['3'];profiles.append(S(p=p['p'],v=p['v'],a=p['a'],j=p['j'],t_sum=[p['end_s']]))
            else:profiles.append(S(p=[0.],v=[0.],a=[0.],j=[0.],t_sum=[1.]))
        obj.trajectory=S(output=S(new_position=command.tolist(),time=4/240,new_calculation=False,trajectory=S(profiles=[profiles])))
        obj._numerical_tracker.trajectory=obj.trajectory;obj._numerical_tracker.calculation_id=379
        obj._numerical_tracker.samples=certificate['samples'][:3]
    rows=[dict(q_command=x.tolist()) for x in history]
    if actual:rows=[dict(q_command=[0.]*7) for _ in range(4788)]+rows
    state=dict(q=np.zeros(7),dq=np.zeros(7),compression_m=0.,time_s=len(rows)/240)
    obj.env=S(rows=rows,out=tmp_path,state=lambda:state,base=np.eye(4))
    obj.sensor=S(rows=[],tick=lambda s:calls.append('sensor_tick'))
    obj.kin=S(tcp=lambda q,compression:np.eye(4))
    obj._read_observed_hall=lambda s:(calls.append('decoded_hall') or S(displacement_estimate_m=0.))
    obj._remember_observed_tcp=lambda t,p:calls.append('remember_tcp')
    obj._capture_observed_tick=lambda *a:calls.append('capture_after_physics')
    obj._check_collision=lambda *a:calls.append('collision')
    def physical(self,q,subphase,active=False,recovery=False):
        calls.append('physical_tick');assert active is False
        state['time_s']+=1/240;obj.env.rows.append(dict(q_command=np.asarray(q).tolist()))
        return state
    return obj,command,calls,physical


def test_mro_preserves_observed_and_physical_chain_exactly_once(tmp_path):
    obj,command,calls,physical=fixture(tmp_path)
    order=m.NumericalLearnedController.__mro__
    assert order.index(ObservedContactSupervisorMixin)<order.index(m.NumericalSafetyController)<order.index(m.SafetyController)<order.index(PhysicalParent)
    with patch.object(PhysicalParent,'_tick',physical):obj._tick(command,'learned_free')
    assert calls==['collision','physical_tick','collision','sensor_tick','decoded_hall','remember_tcp','capture_after_physics']
    assert obj.command_collision_samples==1 and obj.observed_collision_samples==1


def test_actual_query379_certificate_passes_and_tick_runs_once(tmp_path):
    obj,command,calls,physical=fixture(tmp_path,True)
    with patch.object(PhysicalParent,'_tick',physical):obj._tick(command,'learned_free')
    assert calls.count('physical_tick')==1 and calls.count('collision')==2
    event=obj._numerical_buffer[0]
    assert event['comparison']['raw_passed'] is False and event['comparison']['passed'] is True
    assert event['physics_executed'] is True and obj._numerical_interventions==1
    receipt=obj.numerical_receipt();assert receipt['raw_rejected_ticks']==1
    assert (tmp_path/m.LEDGER).read_text().count('\n')==1


def test_unproved_excess_stops_before_collision_and_physics(tmp_path):
    obj,command,calls,physical=fixture(tmp_path);command[2]=.01
    with patch.object(PhysicalParent,'_tick',physical),pytest.raises(m.EpisodeRejected):obj._tick(command,'learned_free')
    assert calls==[] and obj._numerical_buffer[0]['physics_executed'] is False


def test_observed_collision_failure_is_not_swallowed_or_retried(tmp_path):
    obj,command,calls,physical=fixture(tmp_path,True)
    def collision(*args):
        calls.append('collision')
        if calls.count('collision')==2:raise RuntimeError('real measured collision')
    obj._check_collision=collision
    with patch.object(PhysicalParent,'_tick',physical),pytest.raises(RuntimeError,match='measured collision'):obj._tick(command,'learned_free')
    assert calls==['collision','physical_tick','collision']
    assert obj._numerical_buffer[0]['physics_executed'] is True


def test_unchanged_learned_and_contact_implementation():
    for method in ('_infer_step','_contact_cycle','_move_to_prepose','_return_home'):
        assert getattr(m.NumericalLearnedController,method) is getattr(m.LearnedController,method)
