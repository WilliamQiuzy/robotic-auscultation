"""Independent numerical checks for the constrained pose IK differential."""
from types import SimpleNamespace
import time
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sim.auscultation_training.kinematics import OpenArmKinematics
from sim.auscultation_training.pose_ik import PoseConstraint,solve_posture_ik,IKBudgetExpired


def fixture():
    kin=OpenArmKinematics()
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('xyz',[.3,-.7,.1]).as_matrix()
    base[:3,3]=[.1,.2,1.]
    q=np.array([-.6,-.8,.3,1.4,-.2,.1,.5])
    tcp=kin.tcp(q)
    limits=SimpleNamespace(max_elbow_z_m=2.,joint_lower_rad=tuple(kin.lower_limits),
        joint_upper_rad=tuple(kin.upper_limits),min_contact_elbow_below_wrist_m=0.,
        evaluate=lambda *args:{'clear':True})
    return kin,base,q,tcp,limits


def numerical(function,q):
    return np.column_stack([(np.atleast_1d(function(q+np.eye(7)[i]*1e-6))-
                             np.atleast_1d(function(q-np.eye(7)[i]*1e-6)))/2e-6 for i in range(7)])


def test_analytical_pose_elbow_and_wrist_gap_differentials():
    kin,base,q,tcp,_=fixture()
    rotation=Rotation.from_euler('xyz',[.5,-.3,.4]).as_matrix()@tcp[:3,:3]
    pose=PoseConstraint(kin,base,tcp[:3,3]+[.01,.02,-.01],rotation,compression=.002)
    analytic=pose.derivative(q).copy()
    np.testing.assert_allclose(analytic,numerical(pose.residual,q),atol=2e-7,rtol=1e-5)
    np.testing.assert_allclose(pose.elbow_derivative(q),numerical(pose.elbow_height,q)[0],atol=2e-7)
    np.testing.assert_allclose(pose.wrist_gap_derivative(q),numerical(pose.wrist_gap,q)[0],atol=2e-7)


def test_closest_seed_full_pose_solution_preserves_local_branch():
    kin,base,q,tcp,limits=fixture();seed=q+np.array([.01,-.01,.005,.006,-.004,.003,.005])
    solved,accepted,pe,re=solve_posture_ik(kin,base,limits,tcp[:3,3],tcp[:3,:3],seed)
    assert accepted and pe<=2e-5 and re<=2e-4
    assert np.max(np.abs(solved-q))<.025


def test_expired_budget_stops_before_optimization():
    kin,base,q,tcp,limits=fixture()
    with pytest.raises(IKBudgetExpired):
        solve_posture_ik(kin,base,limits,tcp[:3,3],tcp[:3,:3],q,deadline=time.monotonic()-1.)


def test_infeasible_height_constraint_never_reports_pose_success():
    kin,base,q,tcp,limits=fixture();limits.max_elbow_z_m=-10.
    _,accepted,_,_=solve_posture_ik(kin,base,limits,tcp[:3,3],tcp[:3,:3],q,maxiter=10)
    assert not accepted


@pytest.mark.parametrize('bad_rotation',[np.eye(3)*2.,np.diag([1.,1.,-1.]),np.full((3,3),np.nan)])
def test_rejects_improper_or_nonfinite_desired_rotation(bad_rotation):
    kin,base,q,tcp,limits=fixture()
    with pytest.raises(ValueError,match='proper'):
        solve_posture_ik(kin,base,limits,tcp[:3,3],bad_rotation,q)


@pytest.mark.parametrize('part',['position','base','deadline'])
def test_rejects_nonfinite_pose_inputs(part):
    kin,base,q,tcp,limits=fixture();position=tcp[:3,3].copy();deadline=None
    if part=='position':position[0]=np.inf
    if part=='base':base[0,0]=np.nan
    if part=='deadline':deadline=np.inf
    with pytest.raises(ValueError):
        solve_posture_ik(kin,base,limits,position,tcp[:3,:3],q,deadline=deadline)


def test_explicit_zero_contact_gap_still_excludes_elbow_above_wrist():
    kin,_,_,_,limits=fixture()
    base=np.eye(4);base[:3,:3]=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
    base[2,3]=1.376;home=np.zeros(7);tcp=kin.tcp(home)
    _,free,_,_=solve_posture_ik(kin,base,limits,tcp[:3,3],tcp[:3,:3],home,maxiter=10)
    _,contact,_,_=solve_posture_ik(kin,base,limits,tcp[:3,3],tcp[:3,:3],home,maxiter=10,contact=True)
    assert free and not contact
