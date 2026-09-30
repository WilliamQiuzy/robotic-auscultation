"""Active geometry contract. Old evidence is retained, never silently relabeled."""
import hashlib
from pathlib import Path
import numpy as np
from .tool_protrusion import wrist_contact_offset

GEOMETRY_ID = 'stethoscope_retainer_contact_5mm_recess1mm_v2'
GEOMETRY_SOURCE_SHA256 = 'd4ef5726096055c5cdbbb409825d853d610ec865a7bd32e7135c39ec7653a571'
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
            or metadata.get('geometry_source_sha256') != GEOMETRY_SOURCE_SHA256):
        raise ValueError('Retired or unbound geometry: regenerate paths, data and handoff measurements')


def require_geometry_artifacts(directory, metadata):
    """Bind both exported collision files, not merely their caller's epoch label."""
    require_current(metadata)
    result={}
    for name,key in [('collision_geometry.json','collision_geometry_sha256'),
                     ('collision_geometry.npz','collision_arrays_sha256')]:
        expected=metadata.get(key)
        if not isinstance(expected,str) or len(expected)!=64:
            raise ValueError('Missing geometry artifact hash: '+name)
        actual=hashlib.sha256((Path(directory)/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('Geometry artifact changed: '+name)
        result[key]=actual
    return result


def qualify_contact(rows, dt=1/240):
    """Strict physical hold gate; successful setup alone is never a pass."""
    from .retainer_contact import ALL_FORCE_GROUPS, CONTACT_SURFACE
    required = tuple(k+'_N' for k in ALL_FORCE_GROUPS) + ('compression_m', 'hall_N')
    values = np.asarray([[r[k] for k in required] for r in rows], float)
    if (not len(rows) or values.shape != (len(rows), 9) or not np.isfinite(values).all()
            or (values[:,:7]<0).any() or not np.isfinite(dt) or dt<=0):
        return dict(passed=False, reason='missing or nonfinite physics')
    for row,retainer in zip(rows,values[:,1]):
        if retainer==0.:continue
        front=row.get('retainer_front_audit',{})
        if (front.get('passed') is not True or front.get('force_matches_ledger') is not True
                or not np.isclose(front.get('force_N',float('nan')),retainer,atol=1e-8,rtol=1e-8)
                or front.get('off_front_force_N')!=0.):
            return dict(passed=False,reason='unverified or non-front retainer contact')
    contact_force=values[:,:2].sum(axis=1)
    loaded = (contact_force >= 3.5) & (contact_force <= 4.5)
    longest = count = 0
    for good in loaded:
        count = count + 1 if good else 0
        longest = max(longest, count)
    maxima = dict(zip(required, values.max(axis=0).tolist()))
    return dict(passed=bool(longest*dt >= 3.-1e-9 and (values[:,2:7] == 0.).all()
        and (values[:,:7].sum(axis=1) <= 6.).all() and (values[:,7] <= .006).all()
        and (values[:,8] < 5.5).all()), continuous_contact_hold_s=longest*dt,
        contact_surface=CONTACT_SURFACE, contact_force_peak_N=float(contact_force.max()),
        maxima=maxima, lip_must_be_exactly_zero=True, training_eligible=False)
