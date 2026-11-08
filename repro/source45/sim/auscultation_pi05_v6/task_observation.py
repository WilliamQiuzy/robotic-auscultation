"""Explicit supplied task pose appended to the immutable measured 32D state.

The task is registered once against RGB-D body geometry by the caller. This
adapter only transports that registration using the acquired camera transform
and expresses it relative to calibrated ADC/measured-joint TCP geometry. It has
no scene, simulator, IK, action, or default-calibration access.
"""
import hashlib
from pathlib import Path

import numpy as np

VERSION = 'observed32_registered_task9_v1'
RAW_WIDTH = 32
STATE_WIDTH = STATE_DIM = 41
TASK_FEATURES = (tuple('supplied_task_relative_%s_m' % axis for axis in 'xyz')
                 + tuple('supplied_task_rotation_tool_r%dc%d' % (row,column)
                         for column in range(2) for row in range(3)))
MAX_OBSERVATION_AGE_S = .15
MAX_HALL_AGE_S = .05


def observation_binding():
    return dict(version=VERSION,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                state_width=STATE_WIDTH,raw_width=RAW_WIDTH)


def _finite(value,shape,name):
    result=np.asarray(value,dtype=float)
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError('Invalid '+name)
    return result


def _transform(value,name):
    result=_finite(value,(4,4),name+' transform')
    rotation=result[:3,:3]
    if (not np.allclose(result[3],[0.,0.,0.,1.],atol=1e-9,rtol=0)
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6,rtol=0)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6,rtol=0)):
        raise ValueError('Invalid rigid '+name+' transform')
    return result


def encode_task_state(raw_state, *, measured_q, hall_packet, observation_time_s, now_s,
                      body_reference_to_current_base, body_time_s, task_pose_in_body_reference,
                      kinematics, hall_calibration):
    """Return float32[41], with the acquired state preserved in its first 32 slots.

    ``hall_packet`` is the actual typed HallPacket saved with this observation;
    ``hall_calibration`` and ``kinematics`` must be explicitly source-bound by
    the caller. No measurements are refreshed here. Camera transform and raw
    state must come from the same acquisition, not a later tracker update.
    Task rotation uses its first two columns in column-major order, matching
    the original measured surface-rotation convention.
    """
    clock=_finite([observation_time_s,body_time_s,now_s],(3,),'observation clock')
    if np.any(clock<0):raise ValueError('Invalid observation clock')
    if abs(float(body_time_s)-float(observation_time_s))>1e-9:
        raise ValueError('Body transform and state must share the same acquired frame')
    if max(float(observation_time_s),float(body_time_s))>float(now_s)+1e-9:
        raise ValueError('Observation is from the future')
    if float(now_s)-min(float(observation_time_s),float(body_time_s))>MAX_OBSERVATION_AGE_S+1e-9:
        raise ValueError('Observation is stale')
    raw=_finite(raw_state,(RAW_WIDTH,),'raw state')
    with np.errstate(over='ignore',invalid='ignore'):
        raw32=raw.astype(np.float32)
    if not np.isfinite(raw32).all():raise ValueError('Invalid raw float32 state')
    q=_finite(measured_q,(7,),'measured joints')
    if not np.array_equal(q.astype(np.float32),raw32[:7]):
        raise ValueError('Measured joints disagree with raw state')
    # Verify at both clocks: a packet acquired after the camera is not that
    # frame's packet, and a once-valid packet may be stale at consumption.
    decoded=hall_calibration.decode(hall_packet,now_s=float(observation_time_s),max_age_s=MAX_HALL_AGE_S)
    hall_calibration.decode(hall_packet,now_s=float(now_s),max_age_s=MAX_HALL_AGE_S)
    if decoded.status in {'SCREENED_LIMIT','HARD_LIMIT'}:
        raise ValueError('Hall calibration force limit: independent unloading required')
    hall_features=np.asarray([decoded.hall_voltage_v,decoded.spring_force_estimate_n],np.float32)
    if not np.array_equal(hall_features,raw32[20:22]):
        raise ValueError('Decoded Hall features disagree with raw state')
    body=_transform(body_reference_to_current_base,'body')
    task_in_body=_transform(task_pose_in_body_reference,'registered task')
    tcp=_transform(kinematics.tcp(q,decoded.displacement_estimate_m),'measured TCP')
    task=body@task_in_body
    inverse=tcp[:3,:3].T
    relative_position=inverse@(task[:3,3]-tcp[:3,3])
    relative_rotation=inverse@task[:3,:3]
    with np.errstate(over='ignore',invalid='ignore'):
        result=np.concatenate((raw32,relative_position,relative_rotation[:,:2].T.reshape(-1))).astype(np.float32)
    if result.shape!=(STATE_WIDTH,) or not np.isfinite(result).all():raise ValueError('Invalid task observation')
    return result
