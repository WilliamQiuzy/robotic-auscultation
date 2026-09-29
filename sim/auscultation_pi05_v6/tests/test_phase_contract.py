"""Observed phases must agree at export and at the first policy query."""
from types import SimpleNamespace as S

import numpy as np
import pytest

from sim.auscultation_pi05_v6 import phase_contract as contract


def state(distance):
    result = np.zeros(32, np.float32)
    result[14] = distance
    return result


def test_precontact_is_measured_monotonic_and_target_local():
    phase = 'FREE'
    actual = []
    for distance in [.6, .1, .089, .2, .025]:
        phase = contract.precontact_phase(phase, state(distance))
        actual.append(phase)
    assert actual == ['FREE', 'FREE', 'APPROACH', 'APPROACH', 'APPROACH']
    assert contract.precontact_phase('FREE', state(.1)) == 'FREE'
    for invalid in [np.zeros(31), state(float('nan'))]:
        with pytest.raises(ValueError):
            contract.precontact_phase('FREE', invalid)


def test_derived_phases_preserve_pairs_and_original_provenance():
    phases = ['FREE', 'FREE', 'APPROACH', 'FREE', 'CONTACT', 'HOLD', 'RETRACT']
    subphases = ['transfer', 'transfer', 'lower_approach', 'align', 'contact_cycle',
                 'contact_cycle', 'normal_unload']
    frames = [dict(index=i, target='R4', phase=phase, subphase=subphase,
                   state=state(distance).tolist(), eligible=True)
              for i, (phase, subphase, distance) in enumerate(zip(phases, subphases,
                  [.12, .085, .06, .025, .025, .0036, .004]))]
    pairs = [dict(frame_index=i, following_frame_index=i+1) for i in range(6)]
    arrays = dict(state=np.array([f['state'] for f in frames[:-1]], np.float32),
                  phases=np.array(phases[:-1]))
    import copy
    before = copy.deepcopy((frames, pairs))
    derived, receipt = contract.derive_phases(arrays, frames, pairs)
    assert derived.tolist() == ['FREE', 'APPROACH', 'APPROACH', 'APPROACH', 'CONTACT', 'HOLD']
    assert (frames, pairs) == before
    assert receipt['changed_label_count'] == 2
    assert receipt['cross_boundary_label_indices'] == [0, 3, 4, 5]
    assert receipt['selected_frames'][3]['acquired_subphase'] == 'align'
    assert receipt['selected_frames'][3]['acquired_phase'] == 'FREE'
    assert receipt['selected_frames'][3]['derived_phase'] == 'APPROACH'
    arrays['state'][0, 14] = .04
    with pytest.raises(ValueError, match='state'):
        contract.derive_phases(arrays, frames, pairs)


def test_new_phase_boundaries_prevent_crossing_model_chunks():
    from sim.auscultation_pi05_v6.training_data import full_chunks
    observed = [state(.2)] * 11 + [state(.05)] * 11
    phases = []
    current = 'FREE'
    for observation in observed:
        current = contract.precontact_phase(current, observation)
        phases.append(current)
    chunks = full_chunks(np.arange(22, dtype=np.float64)*.05, phases, ['R4']*22,
                         np.zeros((22, 7)), horizon=10)
    assert [item['start'] for item in chunks] == [0, 1, 11, 12]


def actor_fixture(tmp_path, speed=.001, gap=.025, force=1.5):
    from sim.auscultation_pi05_v6.learned_controller import LearnedController
    from sim.auscultation_expert_layout_v1.phase_manager import PhaseManager, PhaseConfig
    actor = LearnedController.__new__(LearnedController)
    actor.manager = PhaseManager(PhaseConfig(precontact_gap_m=.025))
    actor.previous_observed_tcp = None
    actor.previous_observed_time = None
    actor.observed_tcp_speed = None
    actor.env = S(base=np.eye(4), state=lambda: dict(time_s=1., q=np.zeros(7),dq=np.zeros(7)), out=tmp_path,rows=[])
    actor.sensor = S(rows=[dict(index=0,time_s=1., target='R4',raw_sha256='a'*64,
                    body_reference_to_current_base=np.eye(4).tolist())])
    from sim.auscultation_vla_v2.hall import HallPacket
    actor._last_observed_hall_packet=HallPacket(31591,1.,1.,'a'*64,'simulated_adc')
    actor._decision_buffer=[];actor._decision_count=0;actor._decision_file_initialized=False
    actor._tcp_speed_reference=dict(time_s=1.-1/240,physics_row=-1,measured_q=[0.]*7,
        raw_hall_packet=dict(raw=31591,sample_time_s=1.-1/240,received_time_s=1.-1/240,
                             calibration_sha256='a'*64,source='simulated_adc'),
        tcp_position_m=[-speed/240,0.,gap])
    actor.target = S(name='R4')
    hall = S(displacement_estimate_m=0., spring_force_estimate_n=force,
             hall_voltage_v=1.59, sample_time_s=1.)
    pose = np.eye(4)
    pose[2, 3] = gap
    actor._tracked_metrics = lambda _: (dict(gap_m=gap, tangent_m=0., angle_deg=0.), hall, pose)
    actor._remember_observed_tcp(1.-1/240, np.array([-speed/240, 0., gap]))
    actor._remember_observed_tcp(1., pose[:3, 3])
    return actor


def test_zero_time_handoff_uses_real_speed_and_has_no_hold_credit(tmp_path):
    from sim.auscultation_expert_layout_v1.phase_manager import Phase
    actor = actor_fixture(tmp_path)
    actor._initialize_contact_supervisor()
    assert actor.manager.state.phase == Phase.CONTACT
    assert actor.manager.state.phase_elapsed == 0
    assert actor.manager.state.hold_elapsed == 0
    assert actor.manager._episode_elapsed == 0
    actor._initialize_contact_supervisor()
    assert actor.manager.state.hold_elapsed == 0
    assert actor.manager._settle_elapsed == 0
    assert len(actor._decision_buffer)==2
    assert all(row['dt_s']==0. for row in actor._decision_buffer)
    assert actor._decision_buffer[0]['raw_hall_packet']['raw']==31591


@pytest.mark.parametrize('speed,gap', [(.03, .025), (.001, .03)])
def test_handoff_does_not_skip_speed_or_geometry_gates(tmp_path, speed, gap):
    from sim.auscultation_expert_layout_v1.phase_manager import Phase
    actor = actor_fixture(tmp_path, speed=speed, gap=gap)
    actor._initialize_contact_supervisor()
    assert actor.manager.state.phase == Phase.APPROACH


def test_unknown_speed_and_changed_same_time_pose_fail_closed(tmp_path):
    actor = actor_fixture(tmp_path)
    with pytest.raises(RuntimeError, match='changed'):
        actor._remember_observed_tcp(1., np.ones(3))
    actor.observed_tcp_speed = None
    with pytest.raises(RuntimeError, match='speed'):
        actor._initialize_contact_supervisor()


def test_handoff_cannot_hide_loaded_hall_limit_or_stale_observation(tmp_path):
    actor=actor_fixture(tmp_path)
    original=actor._tracked_metrics
    metric,hall,pose=original(None)
    hall.displacement_estimate_m=.003
    hall.spring_force_estimate_n=5.5
    with pytest.raises(RuntimeError,match='overforce'):
        actor._initialize_contact_supervisor()
    actor=actor_fixture(tmp_path)
    actor.sensor.rows[-1]['time_s']=.7
    with pytest.raises(RuntimeError,match='stale'):
        actor._initialize_contact_supervisor()


def test_contact_cycle_first_query_uses_current_supervisor_phase(tmp_path):
    actor = actor_fixture(tmp_path)
    actor.trajectory = object()
    queried = []
    class StopProbe(Exception): pass
    def infer(phase, *, active):
        queried.append((phase, active))
        raise StopProbe
    actor._infer_step = infer
    with pytest.raises(StopProbe):
        actor._contact_cycle()
    assert queried == [('CONTACT', True)]


@pytest.mark.parametrize('acquired,requested', [('FREE','APPROACH'),('APPROACH','CONTACT')])
def test_zero_time_mode_change_reuses_one_physical_frame_and_one_request(tmp_path,acquired,requested):
    actor=actor_fixture(tmp_path)
    actor.env.rows=[];actor.learned_rows=[]
    actor.sensor.out=tmp_path
    np.savez(tmp_path/'frame.npz',rgb=np.zeros((2,2,3),np.uint8))
    frame=actor.sensor.rows[-1]
    from dataclasses import asdict
    from sim.auscultation_pi05_v6.tests.test_task_observation import fixture
    from sim.auscultation_pi05_v6.task_observation import observation_binding
    raw,parameters=fixture()
    actor.kin=parameters['kinematics'];actor.sensor.cal=parameters['hall_calibration']
    actor.task_frames={'R4':dict(task_pose_in_body_reference=parameters['task_pose_in_body_reference'].tolist())}
    actor.env.state=lambda:dict(time_s=1.,q=parameters['measured_q'],dq=np.zeros(7))
    frame.update(index=0,phase=acquired,state=raw.tolist(),q=parameters['measured_q'].tolist(),
                 hall=asdict(parameters['hall_packet']),raw_file='frame.npz',raw_sha256='a'*64)
    def forbidden_capture(*args,**kwargs):
        pytest.fail('Mode change must not append another same-time physical frame')
    actor.sensor.capture=forbidden_capture
    queried=[]
    class StopProbe(Exception):pass
    def predict(rgb,state,phase,target):
        queried.append((phase,target,np.asarray(state).copy()));raise StopProbe
    actor.client=S(predict=predict,binding=dict(observation_contract=observation_binding()))
    with pytest.raises(StopProbe):actor._infer_step(requested,active=requested=='CONTACT')
    assert len(actor.sensor.rows)==1 and frame['phase']==acquired
    assert len(queried)==1 and queried[0][:2]==(requested,'R4')
    assert queried[0][2].shape==(41,)
    np.testing.assert_array_equal(queried[0][2][:32],raw)
    assert frame['state']==raw.tolist()
    np.testing.assert_allclose(queried[0][2][32:35],[.012,-.004,.023],atol=1e-9,rtol=0)
    assert actor.learned_rows[0]['acquired_observation_phase']==acquired
    assert actor.learned_rows[0]['requested_policy_phase']==requested
    assert actor.learned_rows[0]['observation_contract']==observation_binding()
    assert actor.learned_rows[0]['policy_state']==queried[0][2].tolist()
