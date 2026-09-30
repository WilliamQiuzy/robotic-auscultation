"""Five-dimensional contact IK with free tool roll inside the actual safety box."""
import time
import numpy as np
from scipy.optimize import minimize
from sim.auscultation_expert_layout_v1.pose_ik import PoseConstraint, IKBudgetExpired


def solve_axis(kin,base,limits,position,axis,seed,*,contact=True,deadline=None,axis_tolerance_deg=0.):
    if not np.isfinite(axis_tolerance_deg) or not 0<=axis_tolerance_deg<=5.:
        raise ValueError('Original five degree contact angle limit must be retained')
    axis=np.array(axis,dtype=float,copy=True);axis/=np.linalg.norm(axis)
    reference=np.eye(3)[int(np.argmin(abs(axis)))];u=np.cross(axis,reference);u/=np.linalg.norm(u)
    v=np.cross(axis,u);rotation=np.column_stack([u,v,axis])
    pose=PoseConstraint(kin,base,position,rotation,deadline=deadline)
    lower=np.array(limits.joint_lower_rad);upper=np.array(limits.joint_upper_rad)
    lower+=np.where(abs(lower)>1e-10,np.deg2rad(.2),0.);upper-=np.deg2rad(.2)
    seed=np.clip(np.asarray(seed,float),lower,upper)
    cached={}
    def calc(q):
        if deadline is not None and time.monotonic()>=deadline:raise IKBudgetExpired
        if 'q' in cached and np.array_equal(q,cached['q']):return
        pose.calc(q);tcp=kin.tcp(q);jac=kin.jacobian(q);a=tcp[:3,2]
        cached.update(q=q.copy(),axis=a,
            error=np.r_[tcp[:3,3]-position,.1*(a@u),.1*(a@v)],
            derivative=np.vstack([jac[:3],.1*np.cross(a,u)@jac[3:],.1*np.cross(a,v)@jac[3:]]),
            axis_dot=float(a@axis))
    def equality(q):calc(q);return cached['error']
    def derivative(q):calc(q);return cached['derivative']
    def hemisphere(q):calc(q);return cached['axis_dot']-.99
    constraints=[dict(type='eq',fun=equality,jac=derivative),dict(type='ineq',fun=hemisphere),
        dict(type='ineq',fun=lambda q:limits.max_elbow_z_m-.005-pose.elbow_height(q),
             jac=lambda q:-pose.elbow_derivative(q))]
    if axis_tolerance_deg:
        def cone(q):calc(q);return cached['axis_dot']-np.cos(np.deg2rad(axis_tolerance_deg))
        constraints[0]=dict(type='eq',fun=lambda q:equality(q)[:3],jac=lambda q:derivative(q)[:3])
        constraints[1]=dict(type='ineq',fun=cone)
    if contact:
        constraints.append(dict(type='ineq',fun=lambda q:pose.wrist_gap(q)-limits.min_contact_elbow_below_wrist_m-.005,
            jac=pose.wrist_gap_derivative))
    result=minimize(lambda q:float(np.sum((q-seed)**2)),seed,jac=lambda q:2*(q-seed),
        method='SLSQP',bounds=list(zip(lower,upper)),constraints=constraints,
        options=dict(maxiter=180,ftol=1e-11,disp=False))
    q=np.asarray(result.x);calc(q)
    pe=float(np.linalg.norm(cached['error'][:3]));angle=float(np.arccos(np.clip(cached['axis_dot'],-1,1)))
    angle_limit=np.deg2rad(axis_tolerance_deg)+1e-8 if axis_tolerance_deg else 2e-4
    good=(pe<=2e-5 and angle<=angle_limit and limits.evaluate(kin,base,q,contact=contact)['clear']
        and pose.elbow_height(q)<=limits.max_elbow_z_m-.005+1e-8
        and (not contact or pose.wrist_gap(q)>=limits.min_contact_elbow_below_wrist_m+.005-1e-8))
    return dict(q=q,success=bool(good),position_error_m=pe,orientation_error_rad=angle)
