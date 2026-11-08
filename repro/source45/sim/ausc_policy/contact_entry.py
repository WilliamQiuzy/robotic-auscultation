"""Necessary retained-start geometry for FixedNormalCurve, without motion.

This is separate from the measured coarse candidate and from curve/FCL checks.
It neither projects the pose nor supplies an analytic correction to the policy.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def check_retained_pose(pose, target, normal, rotation, *,
                        position_tolerance_m, angle_tolerance_rad):
    pose, target, normal, rotation = [np.asarray(x, float) for x in (pose, target, normal, rotation)]
    if (pose.shape != (4, 4) or target.shape != (3,) or normal.shape != (3,)
            or rotation.shape != (3, 3) or not all(np.isfinite(x).all() for x in (pose, target, normal, rotation))
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or abs(np.linalg.norm(normal)-1) > 1e-9):
        raise ValueError('Invalid retained curve geometry')
    for matrix in (pose[:3, :3], rotation):
        if (not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7, rtol=0)
                or abs(np.linalg.det(matrix)-1) > 1e-7):
            raise ValueError('Invalid retained curve rotation')
    tolerances = np.asarray([position_tolerance_m, angle_tolerance_rad], float)
    if not np.isfinite(tolerances).all() or np.any(tolerances <= 0):
        raise ValueError('Invalid retained curve tolerance')
    gap = float((pose[:3, 3]-target) @ normal)
    error = float(np.linalg.norm(pose[:3, 3]-(target+normal*gap)))
    angle = float(Rotation.from_matrix(pose[:3, :3] @ rotation.T).magnitude())
    failures = []
    if error > position_tolerance_m:
        failures.append('retained_position')
    if angle > angle_tolerance_rad:
        failures.append('retained_full_rotation')
    return dict(passed=not failures, failures=failures, gap_m=gap,
                position_error_m=error, full_rotation_error_rad=angle,
                position_tolerance_m=float(position_tolerance_m),
                angle_tolerance_rad=float(angle_tolerance_rad),
                scope='command zero-compression pose in the existing static curve frame',
                collision_or_contact_certified=False)
