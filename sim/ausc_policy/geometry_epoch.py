"""Active geometry contract. Old evidence is retained, never silently relabeled."""
import re
import numpy as np
from .tool_protrusion import wrist_contact_offset

GEOMETRY_ID = 'stethoscope_protruding_5mm_v1'
RETIRED_EXPERIMENTS = frozenset(('act_baseline', 'act_runtime', 'history_fit',
    'history_runtime', 'isaac_run', 'recovery_run', 'history_prepare',
    'history_runtime_prepare'))


def plane_lower_clearance(boxes, normal, point):
    boxes=np.asarray(boxes,float);normal=np.asarray(normal,float);point=np.asarray(point,float)
    if (boxes.ndim!=3 or boxes.shape[1:]!=(2,3) or normal.shape!=(3,) or point.shape!=(3,)
            or not np.isfinite(boxes).all() or not np.isfinite(point).all()
            or not np.isclose(np.linalg.norm(normal),1.) or np.any(boxes[:,0]>boxes[:,1])):
        raise ValueError('Invalid infinite plane clearance query')
    corners=np.where(normal>=0,boxes[:,0],boxes[:,1])
    return float(np.min((corners-point)@normal))


def contact_frame(compression=0.):
    transform = np.eye(4)
    transform[2, 3] = wrist_contact_offset(compression)
    return transform


def require_current(metadata):
    if (metadata.get('geometry_id') != GEOMETRY_ID
            or metadata.get('freshly_generated') is not True
            or not re.fullmatch('[0-9a-f]{64}', metadata.get('geometry_source_sha256', ''))):
        raise ValueError('Retired or unbound geometry: regenerate paths, data and handoff measurements')


def qualify_contact(rows, dt=1/240):
    """Strict physical hold gate; successful setup alone is never a pass."""
    required = ('diaphragm_N', 'lip_N', 'contact_rim_N', 'carriage_other_N',
                'frame_N', 'forbidden_N', 'compression_m', 'hall_N')
    values = np.asarray([[r[k] for k in required] for r in rows], float)
    if not len(rows) or values.shape != (len(rows), 8) or not np.isfinite(values).all():
        return dict(passed=False, reason='missing or nonfinite physics')
    loaded = (values[:, 0] >= 3.5) & (values[:, 0] <= 4.5)
    longest = count = 0
    for good in loaded:
        count = count + 1 if good else 0
        longest = max(longest, count)
    maxima = dict(zip(required, values.max(axis=0).tolist()))
    return dict(passed=bool(longest*dt >= 3.-1e-9 and (values[:,1:6] == 0.).all()
        and (values[:,:6].sum(axis=1) <= 6.).all() and (values[:,6] <= .006).all()
        and (values[:,7] < 5.5).all()), continuous_membrane_hold_s=longest*dt,
        maxima=maxima, lip_must_be_exactly_zero=True, training_eligible=False)
