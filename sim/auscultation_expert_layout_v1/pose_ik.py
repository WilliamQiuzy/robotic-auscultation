"""Analytic full-pose equality IK with explicit elbow and joint constraints.

This module has no Isaac, FCL, or hardware imports. Feasible IK does not establish
collision-free motion; callers must check every proposed motion segment.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from .tool_frames import wrist_from_diaphragm
from .targets import _base_transform, _vector


class IKBudgetExpired(Exception):
    pass


def _crossmat(x):
    return np.array([[0.,-x[2],x[1]],[x[2],0.,-x[0]],[-x[1],x[0],0.]])


class PoseConstraint:
    """Cached pose residual and analytical spatial SO(3) differential."""
    def __init__(self, kin, base, position, rotation, compression=0., deadline=None):
        self.kin, self.base = kin, _base_transform(base).copy()
        self.position, self.rotation = _vector(position,'position_base').copy(), np.asarray(rotation,float).copy()
        if (self.rotation.shape!=(3,3) or not np.isfinite(self.rotation).all()
                or not np.allclose(self.rotation.T@self.rotation,np.eye(3),atol=1e-6,rtol=0)
                or not np.isclose(np.linalg.det(self.rotation),1.,atol=1e-6,rtol=0)):
            raise ValueError('rotation_base must be a finite proper 3 x 3 rotation')
        if deadline is not None and (not np.isscalar(deadline) or not np.isfinite(deadline)):
            raise ValueError('IK deadline must be a finite absolute monotonic time')
        self.offset = kin.contact_frame(compression)
        self.deadline, self.last = deadline, None

    def calc(self,q):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise IKBudgetExpired
        q=np.asarray(q,float)
        if q.shape!=(7,) or not np.isfinite(q).all(): raise ValueError('invalid IK pose configuration')
        if self.last is not None and np.array_equal(q,self.last): return
        self.last=q.copy()
        links,origins,axes=self.kin._chain(q)
        tcp=links[self.kin.wrist_link]@self.offset
        r=Rotation.from_matrix(tcp[:3,:3]@self.rotation.T).as_rotvec()
        theta=np.linalg.norm(r); K=_crossmat(r)
        factor=1./12. if theta<1e-5 else (1.-.5*theta/np.tan(theta/2.))/(theta*theta)
        inverse=np.eye(3)-.5*K+factor*(K@K)
        self.error=np.r_[tcp[:3,3]-self.position,.1*r]
        self.jac=np.vstack([np.cross(axes,tcp[:3,3]-origins).T,.1*inverse@axes.T])
        elbow=links['openarm_left_link4'][:3,3]
        elbow_jac=np.zeros((3,7));elbow_jac[:,:4]=np.cross(axes[:4],elbow-origins[:4]).T
        wrist=links[self.kin.wrist_link][:3,3]
        wrist_jac=np.cross(axes,wrist-origins).T
        self.height=float((self.base@links['openarm_left_link4'])[2,3])
        self.height_jac=self.base[2,:3]@elbow_jac
        self.wrist_height=float((self.base@links[self.kin.wrist_link])[2,3])
        self.gap_jac=self.base[2,:3]@(wrist_jac-elbow_jac)

    def residual(self,q): self.calc(q);return self.error
    def derivative(self,q): self.calc(q);return self.jac
    def elbow_height(self,q): self.calc(q);return self.height
    def elbow_derivative(self,q): self.calc(q);return self.height_jac
    def wrist_gap(self,q): self.calc(q);return self.wrist_height-self.height
    def wrist_gap_derivative(self,q): self.calc(q);return self.gap_jac


def solve_posture_ik(kin,base,limits,position_base,rotation_base,seed,*,compression=0.,
                     maxiter=100,deadline=None,minimize_elbow=False,planning_margin_m=.005,
                     joint_margin_rad=np.deg2rad(.2),contact=False):
    """Return ``q, accepted, position_error_m, rotation_error_rad``.

    Default objective stays close to the supplied seed. The elbow objective is
    intended for candidate discovery only, never continuous contact tracking.
    Published posture limits are retained; internal margins tighten the solve.
    """
    if (isinstance(maxiter,bool) or not isinstance(maxiter,int) or maxiter<1 or
            not np.isfinite([planning_margin_m,joint_margin_rad]).all() or
            planning_margin_m<0 or joint_margin_rad<0):
        raise ValueError('invalid constrained IK budgets or margins')
    lower=np.asarray(limits.joint_lower_rad,float).copy()
    upper=np.asarray(limits.joint_upper_rad,float).copy()
    contact_gap=float(getattr(limits,'min_contact_elbow_below_wrist_m',0.))
    if (lower.shape!=(7,) or upper.shape!=(7,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower>=upper)
            or np.any(lower<kin.lower_limits) or np.any(upper>kin.upper_limits)
            or not np.isfinite(limits.max_elbow_z_m) or not np.isfinite(contact_gap) or contact_gap<0):
        raise ValueError('invalid IK posture limits')
    lower+=joint_margin_rad
    upper-=joint_margin_rad
    if np.any(lower>=upper):raise ValueError('IK internal margin removes joint range')
    seed=np.asarray(seed,float)
    if seed.shape!=(7,) or not np.isfinite(seed).all():raise ValueError('invalid IK seed')
    seed=np.clip(seed,lower,upper)
    pose=PoseConstraint(kin,base,position_base,rotation_base,compression,deadline)
    ceiling=limits.max_elbow_z_m-planning_margin_m
    constraints=[{'type':'eq','fun':pose.residual,'jac':pose.derivative},
                 {'type':'ineq','fun':lambda q:ceiling-pose.elbow_height(q),
                  'jac':lambda q:-pose.elbow_derivative(q)}]
    if contact:
        constraints.append({'type':'ineq','fun':lambda q:pose.wrist_gap(q)-contact_gap-planning_margin_m,
                            'jac':pose.wrist_gap_derivative})
    objective=pose.elbow_height if minimize_elbow else lambda q:float(np.sum((q-seed)**2))
    derivative=pose.elbow_derivative if minimize_elbow else lambda q:2*(q-seed)
    result=minimize(objective,seed,jac=derivative,method='SLSQP',bounds=list(zip(lower,upper)),
                    constraints=constraints,options={'maxiter':maxiter,'ftol':1e-11,'disp':False})
    q=np.clip(result.x,lower,upper);pose.calc(q)
    pe=float(np.linalg.norm(pose.error[:3]));re=float(np.linalg.norm(pose.error[3:])*10.)
    accepted=(pe<=2e-5 and re<=2e-4 and pose.height<=ceiling+1e-8
              and (not contact or pose.wrist_gap(q)>=contact_gap+planning_margin_m-1e-8)
              and limits.evaluate(kin,base,q)['clear'])
    return q,accepted,pe,re
