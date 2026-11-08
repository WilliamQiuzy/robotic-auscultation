"""Absolute commanded joint positions at t+50 ms in pinned URDF order.

The policy never adds historical servo-lag compensation to the current measured
position. No reinterpretation of V4 pose or V5 offset checkpoints is permitted.
Independent trajectory, collision, posture and force gates remain required.
"""
import numpy as np

VERSION='absolute_joint_command_v1'
COMPATIBLE_QUALIFICATIONS=frozenset()
PHASES={'FREE','APPROACH','CONTACT','HOLD','RETRACT'}
LOWER=(-3.490659,-3.3161253267948965,-1.570796,0.,-1.570796,-.785398,-1.570796)
UPPER=(1.3962629999999998,.17453267320510335,1.570796,2.443461,1.570796,.785398,1.570796)


def limits(phase):
    if phase not in PHASES:raise ValueError('Unknown action phase')
    return dict(lower_rad=list(LOWER),upper_rad=list(UPPER))


def bounded(actions,phase):
    limits(phase)
    a=np.asarray(actions,float)
    return bool(a.ndim==2 and a.shape[1]==7 and len(a)>0 and np.isfinite(a).all()
                and np.all(a>=np.asarray(LOWER)) and np.all(a<=np.asarray(UPPER)))


def compose_joints(measured_q,action,phase,lower,upper):
    q,a,lo,hi=(np.asarray(x,float) for x in (measured_q,action,lower,upper))
    if any(x.shape!=(7,) for x in (q,a,lo,hi)) or not np.isfinite([q,a,lo,hi]).all():
        raise ValueError('Invalid measured joint/action/limits')
    if not np.allclose(lo,LOWER,atol=1e-12,rtol=0) or not np.allclose(hi,UPPER,atol=1e-12,rtol=0):
        raise ValueError('Joint limits differ from the trained URDF convention')
    if not bounded(a[None],phase):raise ValueError('Absolute joint target outside URDF limits; no clipping')
    return a.copy()
