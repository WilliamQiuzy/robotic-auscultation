"""V2 measured-pose-relative command targets, distinct from executed motion.

A position servo can lag its target. TCP(command_next)-TCP(measured_now) is
therefore NOT velocity multiplied by 50 ms. Limits below bound a requested
pose offset; downstream trajectories still require independent 240 Hz joint,
collision, force and posture checks. This module authorizes no execution.

The initial 0.1 mm contact bound incorrectly conflated this command offset
with a motion increment. R1_v5 records ~0.41 mm loaded servo error despite a
nearly stationary HOLD, and ~0.73 mm offset during a measured 4.08 mm/s retreat.
The pose envelope is distinct from the existing 2 mm contact position-error
and 5 degree alignment audits. Requested loaded offsets are bounded to 2 mm
and 0.01 rad; actual contact motion remains independently audited at 240 Hz.
In particular, L2's stationary HOLD servo offset is 5.24 mrad, so the earlier
5 mrad proposal bound rejected an otherwise audited stationary controller.
Free-space orientation proposals allow 0.1 rad, including retained servo error;
the executed seven-joint stream still independently obeys its 0.25 rad/s per
joint limit. A Cartesian rotation-vector norm is not a single joint increment.
The explicit proposal envelope is fixed before task optimization; no label
is clipped or rescaled to fit it. It is an engineering envelope, not a clinical
limit. Changed scene/controller conditions require independent qualification.
"""
import numpy as np

VERSION='measured_tcp_command_offset_v4'
# Same units/origin/force semantics, stricter 1 mm / 5 mrad loaded limits.
# Preserve original attestations verbatim. The loader also rechecks all raw
# actions against the current envelope; this is not a blanket version bypass.
COMPATIBLE_QUALIFICATIONS=frozenset({('measured_tcp_command_offset_v2',
    '294c498966a0da5b69b440ec28bc8c19894570412c747e2c83b7f8eaba5d6d1a'),
    ('measured_tcp_command_offset_v3','0de458b2006ba840142afc8325d882ceecd6d9ed96827401370fb374d5698487'),
    ('measured_tcp_command_offset_v4','7f5d47ea4dd7e06c021e0d9235ac25dd94bf6f5d014ac17aa722c05edbab9322')})
PHASES={'FREE','APPROACH','CONTACT','HOLD','RETRACT'}


def supervised_force_reference(actions,phase):
    """The data's constant 0/4 N phase reference is supervisor-owned.

    This is not learned force regulation. Preserve the complete raw output in
    evaluation; never alter any learned pose channel or actual measured force.
    """
    if phase not in PHASES:raise ValueError('Unknown force-reference phase')
    a=np.array(actions,dtype=float,copy=True)
    if a.ndim!=2 or a.shape[1]!=7 or len(a)==0 or not np.isfinite(a).all():
        raise ValueError('Invalid raw proposal')
    a[:,6]=4. if phase in {'CONTACT','HOLD'} else 0.
    return a


def limits(phase):
    if phase not in PHASES:raise ValueError('unknown action phase')
    loaded=phase in {'CONTACT','HOLD','RETRACT'}
    return dict(translation_m=.002 if loaded else .02,
        rotation_rad=.01 if loaded else .1,force_min_N=0.,force_max_N=4.,
        inactive_force_zero=phase in {'FREE','APPROACH','RETRACT'})


def bounded(actions,phase):
    a=np.asarray(actions,float);bounds=limits(phase)
    if a.ndim!=2 or a.shape[1]!=7 or len(a)==0 or not np.isfinite(a).all():return False
    return bool(np.all(np.linalg.norm(a[:,:3],axis=1)<=bounds['translation_m']+1e-12)
        and np.all(np.linalg.norm(a[:,3:6],axis=1)<=bounds['rotation_rad']+1e-12)
        and np.all((a[:,6]>=0)&(a[:,6]<=4))
        and (not bounds['inactive_force_zero'] or np.all(a[:,6]==0)))
