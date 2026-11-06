"""CPU numerical-boundary regression through the unchanged native strict gate."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest

from test_ausc_given_contact_ruckig_parity import harness
from sim.ausc_policy.velocity_feedforward import NativeVelocityFeedforward, VelocityFeedforwardRejected


class Robot:
    def get_dof_index(self,name): return (int(name.removeprefix('openarm_left_joint'))-1)*2
    def get_joint_positions(self,**kwargs): return np.zeros(7)
    def get_joint_velocities(self,**kwargs): return np.zeros(7)


def strict_command_audit(commands):
    gate=NativeVelocityFeedforward(Robot(),np.arange(7)*2,physics_dt=1/240.)
    maximum=0.
    for command in commands:
        receipt=gate.prepare(command)
        maximum=max(maximum,float(np.max(abs(np.asarray(receipt['velocity_target_rad_s'])))))
        # CPU command-sequence fixture: the next derivative baseline is the
        # preceding accepted command, as native _finish commits after a step.
        gate.previous_executed=np.asarray(command).copy()
    return maximum


@pytest.mark.parametrize('actor',['teacher','student'])
def test_actual_given_ruckig_stream_passes_unmodified_strict_velocity_gate(tmp_path,actor):
    pytest.importorskip('ruckig')
    direction=np.array([-.8,-.05,-.75,1.,-.55,.45,-.35]); route=[direction*.02,direction*.4]
    teacher=harness(tmp_path/'teacher',False,route,flip=True,stop_tick=901)
    teacher._move_to_prepose({})
    selected=teacher
    if actor=='student':
        cells={e['physics_row_start']//12:np.asarray(e['target_q'])
               for e in teacher._intent_dispatcher.events if not e['offgrid_stop']}
        selected=harness(tmp_path/'student',True,route,flip=True,stop_tick=901,
                         target_cells=cells,shared_reference=teacher.reference)
        selected._move_to_prepose({})
    commands=np.asarray(selected.trace)[:,:7]
    peak=strict_command_audit(commands)
    peaks=np.max(abs(np.diff(commands,axis=0)/(1/240.)),axis=0)
    assert .174 < peak <= .175  # exact strict bound, with real cruising plateau
    assert .174 < peaks[3] <= .175  # joint 4, the actual native failure channel
    from sim.ausc_policy.m2_act_controller import LIMITS
    assert LIMITS==dict(velocity=.175,acceleration=.5,jerk=1.5)
    output=os.environ.get('AUSC_VELOCITY_RESERVE_OUTPUT')
    if output:
        from sim.ausc_policy import velocity_feedforward,m2_given_contact_controller as controller
        import ruckig
        root=Path(output);root.mkdir(parents=True,exist_ok=True)
        result=dict(schema='given_strict_velocity_numerical_reserve_cpu_v1',actor=actor,
            ruckig_version=ruckig.__version__,commands=len(selected.trace),
            maximum_exact_adjacent_command_velocity_rad_s=peak,
            maximum_exact_adjacent_command_velocity_by_joint_rad_s=peaks.tolist(),
            hard_velocity_gate_rad_s=velocity_feedforward.JOINT_VELOCITY_LIMIT_RAD_S,
            planning_limits=controller.GIVEN_TRAJECTORY_LIMITS,
            original_shared_limits=LIMITS,clipping=False,hard_gate_epsilon_added=False,
            gpu_work=False,native_physics=False,physical_qualification=False,training_eligible=False,
            strict_gate_source_sha256=hashlib.sha256(Path(velocity_feedforward.__file__).read_bytes()).hexdigest(),
            controller_source_sha256=hashlib.sha256(Path(controller.__file__).read_bytes()).hexdigest())
        (root/(actor+'.json')).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')


def test_original_native_captured_tuple_still_rejects_without_any_physics():
    # Exact scalar excerpt from the original 416-row R3 before-step receipt:
    # /workspace/wam/runs/ausc_policy/m2-given-contact-native-smoke-20261027-v1/
    # output/given-p000449-r3/fresh_expert_velocity_feedforward_failures.json
    # SHA256 86b380b3d1aecec3734625598e972e408559222ddc242780de99c8797f5f3358
    previous=np.array([-.04719230488268759,-.003740923561899376,-.04616339413242737,
        .05979324961445529,-.035937039977574525,.030960867387609725,-.025506860527876506])
    target=np.array([-.0476784994471924,-.0037660703944891517,-.04663364137995261,
        .06052241628112196,-.03626723462340102,.031232588926154443,-.02571998408894384])
    gate=NativeVelocityFeedforward(Robot(),np.arange(7)*2,physics_dt=1/240.)
    gate.previous_executed=previous.copy()
    with pytest.raises(VelocityFeedforwardRejected) as failure:gate.prepare(target)
    receipt=failure.value.receipt
    assert receipt['velocity_target_rad_s'][3]==.1750000000000007
    assert receipt['command_delta_rad'][3]==.0007291666666666696
    assert receipt['joint_velocity_limit_rad_s']==.175
    assert not receipt['physics_step_executed'] and not receipt['target_action_applied']
    np.testing.assert_array_equal(gate.previous_executed,previous)
    output=os.environ.get('AUSC_VELOCITY_RESERVE_OUTPUT')
    if output:
        root=Path(output);root.mkdir(parents=True,exist_ok=True)
        result=dict(schema='original_native_velocity_boundary_negative_cpu_v1',
            source_native_failure_sha256='86b380b3d1aecec3734625598e972e408559222ddc242780de99c8797f5f3358',
            previous_q=previous.tolist(),target_q=target.tolist(),
            rejected=True,velocity_target_rad_s=receipt['velocity_target_rad_s'],
            hard_velocity_gate_rad_s=receipt['joint_velocity_limit_rad_s'],
            physics_step_executed=False,target_action_applied=False,
            gpu_work=False,native_physics=False,physical_qualification=False)
        (root/'captured_negative.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
