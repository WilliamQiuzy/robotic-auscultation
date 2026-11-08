"""Stethoscope contact frames measured from ``openarm_left_link7``.

All distances are metres. Positive compression retracts the moving head along
the wrist's -Z axis. The stage builder uses a 7.25 mm physical slider travel.
These are the membrane and lip surfaces, not the legacy gripper TCP.
"""

import numpy as np

DIAPHRAGM_OFFSET_M = 0.1262
OUTER_LIP_OFFSET_M = 0.1270
MAX_COMPRESSION_M = 0.00725


def _contact_frame(offset, compression):
    value = np.asarray(compression, dtype=float)
    if value.shape != () or not np.isfinite(value) or not 0 <= value <= MAX_COMPRESSION_M:
        raise ValueError(f"compression must be a finite scalar in [0, {MAX_COMPRESSION_M}] metres")
    transform = np.eye(4)
    transform[2, 3] = offset - float(value)
    return transform


def wrist_from_diaphragm(compression=0.0):
    """Return the membrane contact pose in the wrist frame (4 x 4)."""
    return _contact_frame(DIAPHRAGM_OFFSET_M, compression)


def wrist_from_outer_lip(compression=0.0):
    """Return the outer lip pose in the wrist frame (4 x 4)."""
    return _contact_frame(OUTER_LIP_OFFSET_M, compression)
