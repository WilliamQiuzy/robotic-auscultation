"""Seven measured-relative joint command targets, radians in pinned URDF order.

q_command(t + 50 ms) - q_measured(t) includes servo lag; it is not a velocity.
Engineering proposal limits contain the retained qualified demonstrations.
They do not replace independent command, posture, collision or force audits.
This contract is incompatible with every prior pose/force checkpoint.
"""
import numpy as np
VERSION='measured_joint_command_offset_v1'
COMPATIBLE_QUALIFICATIONS=frozenset()
PHASES={'FREE','APPROACH','CONTACT','HOLD','RETRACT'}


def limits(phase):
    if phase not in PHASES:raise ValueError('Unknown action phase')
    return dict(joint_offset_rad=.007 if phase in {'CONTACT','HOLD','RETRACT'} else .035)


def bounded(actions,phase):
    a=np.asarray(actions,float);limit=limits(phase)['joint_offset_rad']
    return bool(a.ndim==2 and a.shape[1]==7 and len(a)>0 and np.isfinite(a).all()
        and np.all(np.abs(a)<=limit+1e-12))


def compose_joints(measured_q,action,phase,lower,upper):
    q=np.asarray(measured_q,float);a=np.asarray(action,float)
    lo=np.asarray(lower,float);hi=np.asarray(upper,float)
    if any(x.shape!=(7,) for x in (q,a,lo,hi)) or not np.isfinite([q,a,lo,hi]).all():
        raise ValueError('Invalid measured joint/action/limits')
    if not bounded(a[None],phase):raise ValueError('Joint proposal rejected; no clipping')
    goal=q+a
    if np.any(lo>=hi) or np.any(goal<lo) or np.any(goal>hi):raise ValueError('Joint position limit')
    return goal
