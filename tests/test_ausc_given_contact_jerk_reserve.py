"""Tighter requested trajectories; original FD acceptance remains unchanged."""
import numpy as np
import pytest

from sim.ausc_policy import m2_given_contact_controller as controller,m2_given_contact_handoff as handoff
from sim.ausc_policy.m2_act_controller import LIMITS
from sim.auscultation_skill_v1.executor import command_derivatives


def test_l2_requested_jerk_is_consistent_and_stricter_with_original_stop_and_physical_gates():
    assert controller.GIVEN_JERK_NUMERICAL_RESERVE_RAD_S3==1e-4
    assert controller.GIVEN_TRAJECTORY_LIMITS==handoff.PLANNING_LIMITS==dict(velocity=.174999,acceleration=.5,jerk=1.4999)
    assert handoff.STOP_PLANNING_LIMITS==dict(velocity=.174999,acceleration=.5,jerk=1.49)
    assert LIMITS==dict(velocity=.175,acceleration=.5,jerk=1.5)
    boundary=controller.GivenContactBoundary();boundary.metadata={}
    assert boundary._trajectory_limits()['jerk']==1.5
    boundary.metadata={'handoff_policy_id':handoff.POLICY_ID}
    assert boundary._trajectory_limits()==handoff.PLANNING_LIMITS


def test_actual_l3_j4_four_command_counterexample_still_exceeds_original_fd_gate():
    # Linux counterfactual continuation of the retained pose2/L3 failure,
    # original requested jerk1.5, row7446. Four f64 scalars, not a capture.
    values=[2.109328622332521,2.1093349904499203,2.1093402376660033,2.1093444724877157]
    history=np.zeros((3,7));history[:,3]=values[:3];q=np.zeros(7);q[3]=values[-1]
    assert command_derivatives(history,q)['jerk']==1.5000000166764949
    assert command_derivatives(history,q)['jerk']>LIMITS['jerk']+1e-8


@pytest.mark.parametrize('owner',['TEACHER_L2','ACT_L2'])
def test_real_ruckig_shared_teacher_student_stream_passes_original_fd_gates(owner):
    pytest.importorskip('ruckig')
    from sim.auscultation_pi05_v6.execution import JointTrajectory
    from sim.ausc_policy.m2_given_contact_intent import IntentDispatcher
    direction=np.array([-.8,-.05,-.75,1.,-.55,.45,-.35]);route=[direction*.02,direction*.4]
    trajectory=JointTrajectory(np.zeros(7),**controller.GIVEN_TRAJECTORY_LIMITS)
    dispatcher=IntentDispatcher(trajectory,reference_sha256='a'*64);commands=[np.zeros(7)]*3
    for i in range(1800):
        if i%12==0:dispatcher.issue(route[0 if i<12 else 1],time_s=i/240.,physics_row=i,
            context=('e','L1',0,'FREE'),owner=owner,role='given_l2_motion')
        commands.append(trajectory.step())
    q=np.asarray(commands)
    for order,limit in [(1,.175),(2,.5+1e-8),(3,1.5+1e-8)]:
        assert np.max(abs(np.diff(q,n=order,axis=0)))/(1/240.)**order<=limit
