"""Replay catches dwell inflation, sensor substitutions, and missing ticks."""
import copy
from dataclasses import asdict
from types import SimpleNamespace as S
import numpy as np
import pytest
from sim.auscultation_expert_layout_v1.phase_manager import PhaseManager,PhaseConfig,ContactObservation
from sim.auscultation_pi05_v6.audit_observed_supervisor import replay,DT


class Kin:
    def tcp(self,q,compression):
        pose=np.eye(4);pose[2,3]=-q[0];return pose


class Cal:
    def decode(self,packet,now_s):
        assert packet.sample_time_s==now_s
        return S(displacement_estimate_m=.003 if packet.raw==4000 else 0.,
                 spring_force_estimate_n=packet.raw/1000,sample_time_s=packet.sample_time_s)


def fixture():
    config=asdict(PhaseConfig(precontact_gap_m=.025,retract_clearance_m=.02,
                              contact_normal_tolerance_m=.006))
    manager=PhaseManager(PhaseConfig(**config));kin=Kin();cal=Cal()
    def sample(i,gap):return dict(time_s=i*DT,target='R4',q=[gap]+[0.]*6,dq=[0.]*7)
    def packet(i,raw):return dict(raw=raw,sample_time_s=i*DT,received_time_s=i*DT,
                                  calibration_sha256='a'*64,source='simulated_adc')
    physics=[sample(0,.025)];frames=[];body=[];records=[]
    def frame(i,raw=4000):
        phase='RETRACT' if manager.state.phase=='DONE' else manager.state.phase.value
        frames.append(dict(index=len(frames),time_s=i*DT,physics_row=i,target='R4',raw_sha256=str(i),
                           phase=phase,hall=packet(i,raw)))
        body.append(dict(frame_index=len(body),time_s=i*DT,body_reference_to_current_base=np.eye(4).tolist()))
    frame(0)
    registration=dict(source_frame=0,source_time_s=0.,initial_body_reference_to_current_base=np.eye(4).tolist(),
        initial_task_position=[0.,0.,0.],initial_task_rotation=np.eye(3).tolist(),task_pose_in_body_reference=np.eye(4).tolist())
    for i in range(1,850):
        released=manager.state.phase in ('RETRACT','DONE');gap=.025 if released or i==1 else .002
        raw=1000 if released else 4000;old_raw=records[-1]['raw_hall_packet']['raw'] if records else 4000
        physics.append(sample(i,gap));dt=0. if i==1 else DT
        pose=kin.tcp(physics[-1]['q'],0);previous=kin.tcp(physics[-2]['q'],0)
        observation=ContactObservation(0. if released else 4.,0. if released else .003,gap,0.,0.,
            float(np.linalg.norm(pose[:3,3]-previous[:3,3])/DT),observation_age_s=i*DT-frames[-1]['time_s'],
            tcp_position_m=tuple(pose[:3,3]))
        before=manager.state;after=manager.update(observation,dt)
        records.append(dict(schema='wam_observed_supervisor_decision_v1',index=len(records),target='R4',
            physics_row=i,time_s=i*DT,dt_s=dt,decision_kind='initialization' if i==1 else 'physics_tick',
            measured_q=physics[-1]['q'],measured_dq=physics[-1]['dq'],raw_hall_packet=packet(i,raw),
            speed_reference=dict(time_s=(i-1)*DT,physics_row=i-1,measured_q=physics[-2]['q'],
                raw_hall_packet=packet(i-1,old_raw),tcp_position_m=previous[:3,3].tolist()),
            observation=asdict(observation),phase_config=config,prior_state=asdict(before),post_state=asdict(after),
            geometry=dict(frame_index=frames[-1]['index'],time_s=frames[-1]['time_s'],
                raw_sensor_sha256=frames[-1]['raw_sha256'],body_reference_to_current_base=np.eye(4).tolist())))
        if dt:physics[-1].update(observed_supervisor_phase=after.phase.value,hold_elapsed_s=after.hold_elapsed,
                                subphase='contact_cycle')
        if i%12==0:frame(i,raw)
    return dict(records=records,physics=physics,frames=frames,body_poses=body,
                registrations={'R4':registration},configs={'R4':config},kin=kin,cal=cal,base=np.eye(4))


def test_replay_complete_contact_accounts_for_three_seconds():
    result=replay(**fixture())
    assert result['max_continuous_hold_s']=={'R4':3.}
    assert result['final_states']['R4']['success'] is True
    assert [x['current'] for x in result['transitions']]==['CONTACT','HOLD','RETRACT','DONE']


@pytest.mark.parametrize('mutation', ['hold','adc','dq','speed','dt','omission','geometry'])
def test_replay_rejects_substitutions(mutation):
    data=fixture();row=data['records'][200]
    if mutation=='hold':row['post_state']['hold_elapsed']+=.75
    elif mutation=='adc':row['raw_hall_packet']['raw']=3000
    elif mutation=='dq':row['measured_dq']=copy.deepcopy(row['measured_dq']);row['measured_dq'][0]=1.
    elif mutation=='speed':row['speed_reference']['tcp_position_m'][0]+=.01
    elif mutation=='dt':row['dt_s']*=2
    elif mutation=='omission':data['records'].pop(200)
    elif mutation=='geometry':row['geometry']['frame_index']-=1
    with pytest.raises(ValueError):replay(**data)
