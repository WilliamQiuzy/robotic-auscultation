"""Versioned slow requests through the unchanged bounded normal executor.

Only demonstration timing changes. Contact acceptance, safety limits and
continuous HOLD remain the same. These are simulation engineering settings.
"""
import math

VERSION = 'measured_gentle_v1'
NEAR_CONTACT_SPEED_M_S = .0004
LOADED_WITHDRAWAL_SPEED_M_S = .0002


def measured_gentle_v1(phase, desired, gap, center):
    if phase not in {'APPROACH', 'CONTACT', 'HOLD', 'RETRACT', 'DONE'}:
        raise ValueError('Unknown contact phase')
    if not all(math.isfinite(v) for v in (desired, gap, center)):
        raise ValueError('Nonfinite contact request')
    if phase in {'RETRACT', 'DONE'}:
        if desired < 0:
            raise ValueError('Withdrawal cannot request inward movement')
        return min(desired, LOADED_WITHDRAWAL_SPEED_M_S) if gap <= center + .002 else desired
    if gap <= center + .003:
        return max(-NEAR_CONTACT_SPEED_M_S, min(desired, NEAR_CONTACT_SPEED_M_S))
    return desired
