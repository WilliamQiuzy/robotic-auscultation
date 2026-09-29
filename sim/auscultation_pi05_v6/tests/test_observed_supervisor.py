"""Teacher and policy share one causal ADC supervisor without a truth fallback."""
from types import SimpleNamespace as S

import numpy as np
import pytest

from sim.auscultation_pi05_v6.observed_supervisor import ObservedContactSupervisorMixin
from sim.auscultation_expert_layout_v1.phase_manager import PhaseManager, PhaseConfig, Phase
from sim.auscultation_expert_layout_v1.controller import DT


class Parent:
    def __init__(self, *, env):
        self.env=env

    def _tick(self, command, subphase, active=False, recovery=False):
        assert active is False, 'ideal-Hall supervisor must remain disabled'
        self.env.rows.append(dict())
        self.env.time_s+=DT
        return dict(time_s=self.env.time_s,q=np.zeros(7),dq=np.zeros(7))


class Observer(ObservedContactSupervisorMixin, Parent):
    pass


def actor_fixture(tmp_path):
    events=[]
    hall=S(displacement_estimate_m=.003,spring_force_estimate_n=4.,
           hall_voltage_v=1.3,sample_time_s=1.)
    def tick(state):
        events.append('adc');hall.sample_time_s=state['time_s']
    from sim.auscultation_vla_v2.hall import HallPacket
    sensor=S(rows=[dict(index=7,time_s=.95,target='R4',phase='CONTACT',raw_sha256='a'*64,
                        body_reference_to_current_base=np.eye(4).tolist())],
             tick=tick,
             hall=S(read=lambda stamp:HallPacket(27000,stamp,stamp,'a'*64,'simulated_adc')),
             cal=S(decode=lambda *a,**kw:hall))
    def capture(state,phase,target,physics_row,**kw):
        events.append(('capture',phase))
        sensor.rows.append(dict(index=sensor.rows[-1]['index']+1,time_s=state['time_s'],phase=phase,target=target,
                                raw_sha256='a'*64,body_reference_to_current_base=np.eye(4).tolist()))
    sensor.capture=capture
    actor=Observer(sensor=sensor,env=S(rows=[],out=tmp_path,base=np.eye(4),time_s=1.-DT))
    actor.target=S(name='R4');actor.task_frames={'R4':{}}
    actor.manager=PhaseManager(PhaseConfig(contact_normal_tolerance_m=.006))
    actor.manager._phase=Phase.CONTACT
    actor.manager._contact_confirmed=True
    actor.manager._settle_elapsed=.3-DT
    actor.actor_phase='CONTACT'
    actor.audit_hold=actor.best_audit_hold=0.
    actor.face_filtered=actor.total_filtered=4.
    pose=np.eye(4);pose[2,3]=.003
    actor.kin=S(tcp=lambda *a:pose.copy())
    def tracked(state):
        events.append(('geometry',sensor.rows[-1]['index']))
        return dict(gap_m=.003,tangent_m=0.,angle_deg=0.),hall,pose.copy()
    actor._tracked_metrics=tracked
    actor._remember_observed_tcp(1.-DT,pose[:3,3])
    actor._last_tcp_measurement=dict(time_s=1.-DT,physics_row=-1,measured_q=[0.]*7,
        raw_hall_packet=dict(raw=27000,sample_time_s=1.-DT,received_time_s=1.-DT,
                            calibration_sha256='a'*64,source='simulated_adc'),
        tcp_position_m=pose[:3,3].tolist())
    return actor,events,hall


def test_capture_follows_shared_update_and_records_geometry_provenance(tmp_path):
    actor,events,_=actor_fixture(tmp_path)
    actor._tick(np.zeros(7),'contact_cycle',active=True)
    assert actor.manager.state.phase==Phase.HOLD
    assert actor.manager.state.hold_elapsed==0.
    assert events==['adc',('geometry',7),('capture','HOLD')]
    assert actor.sensor.rows[-1]['phase']=='HOLD'
    assert actor.env.rows[-1]['observed_supervisor_geometry_frame']==7
    assert actor.env.rows[-1]['observed_supervisor_geometry_time_s']==.95
    assert actor.env.rows[-1]['observed_supervisor_phase']=='HOLD'
    assert actor.audit_hold==0.
    decision=actor._decision_buffer[0]
    assert decision['dt_s']==DT
    assert decision['raw_hall_packet']['raw']==27000
    assert decision['geometry']['frame_index']==7
    assert decision['post_state']['phase']=='HOLD'
    assert decision['prior_state']['phase']=='CONTACT'
    assert decision['speed_reference']['time_s']==1.-DT
    assert decision['speed_reference']['raw_hall_packet']['raw']==27000


def test_measured_hold_credit_cannot_replace_independent_front_contact_hold(tmp_path):
    actor,_,hall=actor_fixture(tmp_path)
    actor._tick(np.zeros(7),'contact_cycle',active=True) # enter HOLD with zero credit
    for _ in range(12):actor._tick(np.zeros(7),'contact_cycle',active=True)
    assert actor.audit_hold==pytest.approx(.05)
    hall.spring_force_estimate_n=3. # Invalid measured load breaks continuous hold.
    actor._tick(np.zeros(7),'contact_cycle',active=True)
    assert actor.manager.state.hold_elapsed==0.
    assert actor.audit_hold==0.
    hall.spring_force_estimate_n=4.
    actor.face_filtered=0. # Hall remains valid, actual front contact does not.
    for _ in range(84):actor._tick(np.zeros(7),'contact_cycle',active=True)
    assert actor.manager.state.hold_elapsed==pytest.approx(.05)
    assert actor.audit_hold==0.
    assert actor.best_audit_hold==pytest.approx(.05)


def test_missing_geometry_fails_before_active_observed_motion_update(tmp_path):
    sensor=S(rows=[])
    actor=Observer(sensor=sensor,env=S(out=tmp_path))
    with pytest.raises(RuntimeError,match='tracker'):
        actor._tracked_metrics(dict(time_s=1.))


def test_bounded_ledger_flush_and_partial_run_failure(tmp_path):
    import json
    actor,_,_=actor_fixture(tmp_path)
    actor.DECISION_BUFFER_LIMIT=2
    actor._tick(np.zeros(7),'contact_cycle',active=True)
    actor._tick(np.zeros(7),'contact_cycle',active=True)
    assert len(actor._decision_buffer)==0
    assert len((tmp_path/'supervisor_decisions.jsonl').read_text().splitlines())==2
    actor._tick(np.zeros(7),'contact_cycle',active=True)
    class ParentFailure(Parent):
        def run(self,points):raise RuntimeError('Deliberate partial failure')
    class FailingObserver(ObservedContactSupervisorMixin,ParentFailure):pass
    actor.__class__=FailingObserver
    actor._prepare_task_frames=lambda points:None
    actor._verify_observed_base=lambda points:None
    with pytest.raises(RuntimeError,match='partial failure'):
        actor.run([object()])
    rows=[json.loads(line) for line in (tmp_path/'supervisor_decisions.jsonl').read_text().splitlines()]
    assert [row['index'] for row in rows]==[0,1,2]
    receipt=json.loads((tmp_path/'supervisor_decisions_receipt.json').read_text())
    assert receipt['decision_count']==3
    assert receipt['flushed_on_run_exit'] is True
    assert receipt['world_from_left_base']==np.eye(4).tolist()


def test_actual_base_is_checked_against_bound_plan_without_substitution(tmp_path):
    import hashlib,json
    actor,_,_=actor_fixture(tmp_path)
    snapshot=tmp_path/'input_snapshot';(snapshot/'objects').mkdir(parents=True)
    approved=np.eye(4);raw=json.dumps(dict(world_from_left_base=approved.tolist())).encode()
    digest=hashlib.sha256(raw).hexdigest();(snapshot/'objects'/digest).write_bytes(raw)
    (snapshot/'manifest.json').write_text(json.dumps(dict(files={'R4/plan':dict(sha256=digest)})))
    actor.env.base[0,3]=1e-8
    actor._verify_observed_base([(actor.target,{})])
    assert actor.env.base[0,3]==1e-8
    actor.env.base[0,3]=1e-6
    with pytest.raises(RuntimeError,match='differs'):
        actor._verify_observed_base([(actor.target,{})])
