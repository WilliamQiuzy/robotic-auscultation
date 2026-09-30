"""User-confirmed contact interface: recessed membrane plus device retainer.

This namespace is separate from the printed RetentionLip and ContactRim.
Contact permission is restricted to these exact current-geometry components
against the patient; it grants no table, arm or general shell exemption.
"""
import numpy as np

CONTACT_SURFACE = 'membrane_and_device_retainer_v2'
ROOT = '/World/Stethoscope/Carriage/Colliders/'
DIAPHRAGM_PATH = ROOT+'Diaphragm'
RETAINER_PATHS = frozenset(ROOT+'DeviceContactRing/sector_'+str(i).zfill(2)
                           for i in range(32))
CONTACT_PATHS = RETAINER_PATHS | {DIAPHRAGM_PATH}
CONTACT_FORCE_GROUPS = ('diaphragm', 'retainer')
FORBIDDEN_FORCE_GROUPS = ('lip', 'contact_rim', 'carriage_other', 'frame', 'forbidden')
ALL_FORCE_GROUPS = CONTACT_FORCE_GROUPS + FORBIDDEN_FORCE_GROUPS


def is_retainer_patient_pair(a, b):
    return ((a in RETAINER_PATHS and b.startswith('/World/Patient/'))
            or (b in RETAINER_PATHS and a.startswith('/World/Patient/')))


def audit_retainer_front(points, world_from_carriage, reported_force_N, dt=1/240):
    """Reconcile actual impulses and screen ring front vs side/back contacts.

    0.1 mm is an explicit numerical contact-location tolerance, not a gap,
    mesh offset, allowed penetration, or alteration of the user 1 mm recess.
    Native current carriage pose is required; commanded FK is not substituted.
    """
    matrix=np.asarray(world_from_carriage,float)
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3],[0,0,0,1],atol=1e-10,rtol=0)
            or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-7,rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3,:3]),1.,atol=1e-7,rtol=0)
            or not np.isfinite(reported_force_N) or reported_force_N<0
            or not np.isfinite(dt) or dt<=0):
        raise ValueError('Invalid native retainer contact audit input')
    tolerance=.0001
    total=bad=0.;locations=[]
    for point in points:
        pair=point['colliders']
        if len(pair)!=2 or not all(isinstance(p,str) for p in pair):
            raise ValueError('Invalid raw contact pair')
        if not is_retainer_patient_pair(*pair):continue
        impulse=np.asarray(point['impulse_Ns'],float)
        position=np.asarray(point['position_world'],float)
        if impulse.shape!=(3,) or position.shape!=(3,) or not np.isfinite([impulse,position]).all():
            raise ValueError('Invalid raw contact point')
        force=float(np.linalg.norm(impulse)/dt)
        local=matrix[:3,:3].T@(position-matrix[:3,3])
        radial=float(np.linalg.norm(local[:2]))
        on_front=bool(abs(local[2]+.047)<=tolerance
                      and .021*np.cos(np.pi/32)-tolerance<=radial<=.023+tolerance)
        total+=force
        if not on_front:bad+=force
        locations.append(dict(carriage_position_m=local.tolist(),force_N=force,on_front=on_front))
    matched=bool(np.isclose(total,reported_force_N,atol=1e-8,rtol=1e-8))
    return dict(passed=bool(matched and bad==0.),force_N=total,
                force_matches_ledger=matched,off_front_force_N=bad,
                location_tolerance_m=tolerance,points=locations)
