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


def solve_contact_region(kin, base, limits, position, axis, seed, *, contact=True,
                         tangent_limit_m=.002, angle_limit_deg=4.5, deadline=None):
    """Propose a pose inside the existing 3 mm / 5 degree target region.

    Target coordinates are never changed. The nominal surface plane is retained;
    the actual residual is reported separately for collision and physics checks.
    Defaults reserve tracking margin inside the original acceptance tolerances.
    This does not certify finite-head fit or permit motion.
    """
    position, axis, seed = [np.array(x, dtype=float, copy=True) for x in (position, axis, seed)]
    if (position.shape != (3,) or axis.shape != (3,) or seed.shape != (7,)
            or not np.isfinite(np.r_[position, axis, seed]).all()
            or not .99 <= np.linalg.norm(axis) <= 1.01
            or not np.isfinite([tangent_limit_m, angle_limit_deg]).all()
            or not 0 < tangent_limit_m <= .003 or not 0 < angle_limit_deg <= 5.):
        raise ValueError('Contact-region proposal must retain original target tolerances')
    axis /= np.linalg.norm(axis)
    u = np.cross(axis, np.eye(3)[np.argmin(abs(axis))]); u /= np.linalg.norm(u)
    rotation = np.column_stack([u, np.cross(axis, u), axis])
    pose = PoseConstraint(kin, base, position, rotation, deadline=deadline)
    lower = np.array(limits.joint_lower_rad, dtype=float)
    upper = np.array(limits.joint_upper_rad, dtype=float)
    if (lower.shape != (7,) or upper.shape != (7,)
            or not np.isfinite(np.r_[lower, upper, limits.max_elbow_z_m,
                                    limits.min_contact_elbow_below_wrist_m]).all()
            or np.any(lower >= upper) or np.any(lower < kin.lower_limits)
            or np.any(upper > kin.upper_limits) or limits.min_contact_elbow_below_wrist_m < 0):
        raise ValueError('Invalid contact-region posture bounds')
    lower += np.deg2rad(.2)
    upper -= np.deg2rad(.2)
    if np.any(lower >= upper):
        raise ValueError('Contact-region planning margin removes joint range')
    seed = np.clip(seed, lower, upper)
    tangent_projection = np.eye(3) - np.outer(axis, axis)
    cached = {}

    def calc(q):
        if deadline is not None and time.monotonic() >= deadline:
            raise IKBudgetExpired
        if 'q' in cached and np.array_equal(q, cached['q']): return
        pose.calc(q)
        tcp = kin.tcp(q); jac = kin.jacobian(q)
        residual = tcp[:3,3] - position
        cached.update(q=q.copy(), error=residual, jac=jac, tangent=tangent_projection@residual,
                      axis=tcp[:3,2], dot=float(tcp[:3,2]@axis))

    def objective(q):
        calc(q)
        return float(cached['error']@cached['error']/tangent_limit_m**2 + .001*np.sum((q-seed)**2))

    def derivative(q):
        calc(q)
        return 2*cached['error']@cached['jac'][:3]/tangent_limit_m**2 + .002*(q-seed)

    def normal(q): calc(q); return cached['error']@axis*1000.
    def normal_jac(q): calc(q); return axis@cached['jac'][:3]*1000.
    def radius(q): calc(q); return 1.-cached['tangent']@cached['tangent']/tangent_limit_m**2
    def radius_jac(q): calc(q); return -2*cached['tangent']@cached['jac'][:3]/tangent_limit_m**2
    def cone(q): calc(q); return (cached['dot']-np.cos(np.deg2rad(angle_limit_deg)))*100.
    def cone_jac(q): calc(q); return np.cross(cached['axis'],axis)@cached['jac'][3:]*100.
    constraints = [dict(type='eq',fun=normal,jac=normal_jac),
        dict(type='ineq',fun=radius,jac=radius_jac),dict(type='ineq',fun=cone,jac=cone_jac),
        dict(type='ineq',fun=lambda q:limits.max_elbow_z_m-.005-pose.elbow_height(q),
             jac=lambda q:-pose.elbow_derivative(q))]
    if contact:
        constraints.append(dict(type='ineq',fun=lambda q:pose.wrist_gap(q)-limits.min_contact_elbow_below_wrist_m-.005,
                                jac=pose.wrist_gap_derivative))
    result = minimize(objective, seed, jac=derivative, method='SLSQP',
        bounds=list(zip(lower, upper)), constraints=constraints,
        options=dict(maxiter=180,ftol=1e-11,disp=False))
    q=np.asarray(result.x);calc(q)
    gap=float(cached['error']@axis);tangent=float(np.linalg.norm(cached['tangent']))
    angle=float(np.arccos(np.clip(cached['dot'],-1,1)))
    good=(abs(gap)<=2e-5 and tangent<=tangent_limit_m+1e-8
          and angle<=np.deg2rad(angle_limit_deg)+1e-8
          and np.all(q>=lower-1e-10) and np.all(q<=upper+1e-10)
          and limits.evaluate(kin,base,q,contact=contact)['clear']
          and pose.elbow_height(q)<=limits.max_elbow_z_m-.005+1e-8
          and (not contact or pose.wrist_gap(q)>=limits.min_contact_elbow_below_wrist_m+.005-1e-8))
    return dict(q=q,success=bool(good),position_residual_base_m=cached['error'].copy(),
                tangent_error_m=tangent,normal_error_m=gap,orientation_error_rad=angle,
                collision_checked=False,physics_executed=False)
