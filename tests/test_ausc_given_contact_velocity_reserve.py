"""CPU numerical-boundary regression through the unchanged native strict gate."""
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest

from test_ausc_given_contact_ruckig_parity import harness
from test_ausc_given_contact_controller import pose_at
from sim.ausc_policy import m2_given_contact_handoff as h
from sim.auscultation_expert_layout_v1.phase_manager import PhaseConfig
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



def current_handoff_fixture(c):
    """Current v3 contract with synthetic CPU feedback, not physical FK."""
    c.metadata['handoff_policy_id']=h.POLICY_ID
    c.manager.config=replace(PhaseConfig(),precontact_gap_m=.03)
    # Explicit geometry fixture. The original test also uses a synthetic coarse
    # row threshold, not physical FK/contact feedback. Preserve that threshold;
    # keep the pose outside the interior tube until the declared coarse row.
    def pose():return pose_at(c.reference,tangent=.01 if len(c.env.rows)<901 else .001,roll=0.)
    c.kin.tcp=lambda q:pose()
    original_state=c.env.state
    def state():
        p=pose()
        return dict(original_state(),tcp_base=p[:3,3],rotation_world=p[:3,:3],compression_m=0.)
    c.env.state=state
    c.posture_limits=SimpleNamespace(evaluate=lambda *a,**k:dict(clear=True))
    c.checker=SimpleNamespace(check=lambda *a,**k:dict(clear=True,minimum_distance_m=.01))
    c.static_world=SimpleNamespace(check=lambda:None)
    original_block,original_hold=c._block,c._hold
    def enrich(start):
        for row in c.env.rows[start:]:row['tcp_speed_m_s']=0.
    def block(*a,**k):
        start=len(c.env.rows);value=original_block(*a,**k);enrich(start);return value
    def hold(*a,**k):
        start=len(c.env.rows);value=original_hold(*a,**k);enrich(start);return value
    c._block=block;c._hold=hold
    return c


@pytest.mark.parametrize('actor',['teacher','student'])
def test_current_given_ruckig_stream_passes_unmodified_strict_velocity_gate(tmp_path,actor):
    pytest.importorskip('ruckig')
    direction=np.array([-.8,-.05,-.75,1.,-.55,.45,-.35]); route=[direction*.02,direction*.4]
    teacher=current_handoff_fixture(harness(tmp_path/'teacher',False,route,flip=True,stop_tick=901))
    teacher._move_to_prepose({})
    selected=teacher
    if actor=='student':
        cells={e['physics_row_start']//12:np.asarray(e['target_q'])
               for e in teacher._intent_dispatcher.events if not e['offgrid_stop']}
        selected=current_handoff_fixture(harness(tmp_path/'student',True,route,flip=True,stop_tick=901,
                         target_cells=cells,shared_reference=teacher.reference))
        selected._move_to_prepose({})
    assert selected.metadata['handoff_policy_id']==h.POLICY_ID
    assert selected._interior_policy() is True
    assert selected.l2_handoffs[-1]['parking_budget_steps']<=h.MAX_PARK_STEPS
    commands=np.asarray(selected.trace)[:,:7]
    if actor=='student':np.testing.assert_array_equal(commands,np.asarray(teacher.trace)[:,:7])
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
            handoff_policy_id=selected.metadata['handoff_policy_id'],synthetic_cpu_geometry_feedback=True,
            strict_gate_source_sha256=hashlib.sha256(Path(velocity_feedforward.__file__).read_bytes()).hexdigest(),
            controller_source_sha256=hashlib.sha256(Path(controller.__file__).read_bytes()).hexdigest())
        (root/(actor+'.json')).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')


def test_archival_handoff_keeps_exact_stationary_roundoff_rejection(tmp_path):
    pytest.importorskip('ruckig')
    from sim.auscultation_pi05_v6 import execution
    direction=np.array([-.8,-.05,-.75,1.,-.55,.45,-.35]);route=[direction*.02,direction*.4]
    # Exact old synthetic fixture: no current handoff ID selects the archival
    # parking branch. It is a guard-negative, not a supported positive route.
    teacher=harness(tmp_path/'legacy',False,route,flip=True,stop_tick=901)
    assert teacher.metadata.get('handoff_policy_id') is None and not teacher._interior_policy()
    with pytest.raises(execution.FinishedStationaryRoundoffError) as failure:
        teacher._move_to_prepose({})
    details=failure.value.details
    assert execution.FINISHED_ROUNDOFF_ULPS==details['roundoff_ulps']==64
    target=np.asarray(details['target_position']);raw=np.asarray(details['raw_position'])
    bound=64*np.finfo(np.float64).eps
    assert abs(raw[1]-target[1])>bound*max(1.,abs(target[1]))
    assert np.max(abs(np.asarray(details['raw_velocity'])))<=bound
    assert np.max(abs(np.asarray(details['raw_acceleration'])))<=bound
    assert teacher.control_owner=='STOPPED'
    assert len(teacher.trace)==len(teacher.env.rows)==1247  # no further action after refusal


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
