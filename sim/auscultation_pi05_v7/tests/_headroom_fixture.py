"""Independent raw-Ruckig evidence fixture for explicit runtime headroom, not the runtime executor."""
import base64

import hashlib

import importlib

import importlib.util

import io

import json

from pathlib import Path

import numpy as np

from PIL import Image

import pytest

import ruckig

from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics

from sim.auscultation_vla_v2.hall import HallPacket, default_calibration

from sim.auscultation_pi05_v6.task_observation import observation_binding

from sim.auscultation_pi05_v6.action_encoding import action_encoding_binding

DT = 1 / 240

CHECKPOINT = 'a' * 64

SEED = np.array([.2, -.4, .3, 1.2, -.2, .1, .4])

def f32_digest(value):
    return hashlib.sha256(np.asarray(value, dtype='<f4').tobytes()).hexdigest()

def generator(q):
    otg, inp, out = ruckig.Ruckig(7, DT), ruckig.InputParameter(7), ruckig.OutputParameter(7)
    inp.current_position = list(q)
    inp.current_velocity = inp.current_acceleration = [0.] * 7
    inp.target_velocity = inp.target_acceleration = [0.] * 7
    inp.max_velocity, inp.max_acceleration, inp.max_jerk = [.175] * 7, [.5] * 7, [7.99999] * 7
    return otg, inp, out

def fixture(tmp_path, *, new_target=False, unexplained_gap=False, reset_between=False, add_measured_origin=False):
    sensor_root = tmp_path / 'sensors'
    sensor_root.mkdir()
    binding = dict(checkpoint_metadata_sha256=CHECKPOINT, data_hash='b' * 64,
                   inference_source_sha256={'actor.py': 'c' * 64}, sampling_steps=10,
                   sampling_seed=917, sampling_policy='fixed', audio='absent', modalities={'rgb': 'camera'},
                   dynamics_recipe_sha256='d'*64, dynamics_contract_sha256='e'*64,
                   execution_contract_sha256='f'*64,
                   action_semantics='absolute_joint_command_v1',
                   training_recipe='dynamics_bound_absolute_joint_actions_v6_task_pose_logit_v1',
                   observation_contract=observation_binding(),action_encoding_binding=action_encoding_binding())
    physics, frames, learned, requests = [], [], [], []
    body_poses=[];kin=OpenArmKinematics();cal=default_calibration()
    registrations={}
    for target,x in [('R1',.1),('R2',.2)]:
        task=np.eye(4);task[:3,3]=[x,.2,.4]
        registrations[target]=dict(initial_task_position=task[:3,3].tolist(),
            initial_task_rotation=task[:3,:3].tolist(),task_pose_in_body_reference=task.tolist(),
            initial_body_reference_to_current_base=np.eye(4).tolist(),source_frame=0,
            source_time_s=3*DT,registration_observed_target='R1')

    def append(command, target='R1', subphase='settle'):
        measured = np.asarray(np.asarray(command) + .001, dtype=np.float32)
        physics.append(dict(time_s=(len(physics) + 1) * DT, q_command=list(command),
                            q=measured.tolist(), target=target, phase='APPROACH', subphase=subphase, recovery=False))

    for _ in range(3):
        append(SEED)
    otg, inp, out = generator(SEED)
    for number in range(2):
        target = 'R2' if number and new_target else 'R1'
        if number and new_target:
            inp.target_position = physics[-1]['q_command']
            for _ in range(120):
                assert otg.update(inp, out) in (ruckig.Result.Working, ruckig.Result.Finished)
                append(out.new_position, subphase='learned_release_stop')
                out.pass_to_input(inp)
                if max(abs(x) for x in out.new_velocity + out.new_acceleration) < 1e-10:
                    break
            else:
                raise AssertionError('fixture release did not stop')
            for _ in range(3):
                append(physics[-1]['q_command'], subphase='withdraw_settle')
            otg, inp, out = generator(physics[-1]['q_command'])
        elif number and unexplained_gap:
            append(physics[-1]['q_command'], subphase='unlogged_route')
        if number and reset_between:
            otg, inp, out = generator(physics[-1]['q_command'])
        start = len(physics)
        measured = np.asarray(physics[-1]['q'])
        phase = 'RETRACT' if new_target and number == 0 else 'FREE'
        state = np.zeros(32, dtype=np.float32)
        state[:7] = measured
        packet=HallPacket(27000,physics[-1]['time_s'],physics[-1]['time_s'],cal.sha256,'simulated_adc')
        decoded=cal.decode(packet,now_s=physics[-1]['time_s'])
        state[20:22]=[decoded.hall_voltage_v,decoded.spring_force_estimate_n]
        body=np.eye(4);body[0,3]=number*.001
        task=body@np.asarray(registrations[target]['task_pose_in_body_reference'])
        tcp=kin.tcp(measured,decoded.displacement_estimate_m)
        relative=tcp[:3,:3].T@task[:3,:3]
        policy_state=np.r_[state,tcp[:3,:3].T@(task[:3,3]-tcp[:3,3]),relative[:,:2].T.reshape(-1)].astype(np.float32)
        rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3) + number
        raw_path = sensor_root / ('%05d.npz' % number)
        np.savez_compressed(raw_path, rgb=rgb, depth_m=np.ones((3, 4)), depth_valid=np.ones((3, 4), bool))
        raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        frame = dict(index=number, physics_row=start-1, time_s=physics[-1]['time_s'], phase=phase,
                     target=target, q=measured.tolist(), state=state.tolist(), raw_file=raw_path.name,
                     raw_sha256=raw_sha, target_estimate={'revision': 0},hall=vars(packet))
        frames.append(frame)
        body_poses.append(dict(frame_index=number,time_s=frame['time_s'],
            body_reference_to_current_base=body.tolist(),source='measured RGB-D rigid registration'))
        raw = np.zeros((10, 7))
        if number == 0:
            distance = .005 if new_target else .01
            raw[0] = SEED + np.array([distance] * 6 + [-distance])
        else:
            raw[0] = SEED + np.array([-.003] * 6 + [.003])
        request_id = 'request-' + str(number)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format='PNG')
        request_bytes = json.dumps(dict(request_id=request_id, phase=phase, target=target,
                                       state=policy_state.tolist(), rgb_png=base64.b64encode(buf.getvalue()).decode())).encode()
        hashes = dict(request_payload_sha256=hashlib.sha256(request_bytes).hexdigest(),
                      rgb_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), rgb_shape=list(rgb.shape),
                      state_f32_sha256=f32_digest(policy_state), raw_action_f32_sha256=f32_digest(raw))
        receipt = dict(binding, request_id=request_id, raw_actions=raw.tolist(), **hashes)
        requests.append(dict(request_id=request_id, phase=phase, target=target,
                             checkpoint_metadata_sha256=CHECKPOINT, **hashes))
        goal = raw[0] + measured if add_measured_origin else raw[0].copy()
        inp.target_position = goal.tolist()
        for _ in range(12):
            assert otg.update(inp, out) in (ruckig.Result.Working, ruckig.Result.Finished)
            append(out.new_position, target=target, subphase='learned_' + phase.lower())
            out.pass_to_input(inp)
        learned.append(dict(index=number, phase=phase, target=target, time_s=frame['time_s'],
                            observation_frame=number, raw_sensor_sha256=raw_sha,
                            policy_state=policy_state.tolist(),observation_contract=observation_binding(),
                            physics_row_start=start, physics_row_end_exclusive=len(physics),
                            accepted=True, expert_fallback=False, executed_chunk_prefix=1,
                            raw_actions=raw.tolist(), measured_q=measured.tolist(), goal_q=goal.tolist(), inference=receipt))
    return dict(learned=learned, frames=frames, physics=physics, binding=binding,
                expected_checkpoint_metadata_sha256=CHECKPOINT, raw_sensor_root=sensor_root,
                server_requests=requests, velocity=.175, task_scope='full',body_poses=body_poses,
                registrations=registrations,kinematics=kin,hall_calibration=cal)
