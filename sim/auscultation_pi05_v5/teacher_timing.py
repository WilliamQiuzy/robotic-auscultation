"""Slower expert requests through the existing bounded normal executor.

This changes demonstration timing, never contact/safety acceptance thresholds.
Values are simulation engineering settings, not clinical pressure guidance.
"""
import math


def gentle_v2(phase, desired, gap, center):
    if phase not in {'APPROACH', 'CONTACT', 'HOLD', 'RETRACT', 'DONE'}:
        raise ValueError('Unknown contact phase')
    if not all(math.isfinite(v) for v in (desired, gap, center)):
        raise ValueError('Nonfinite contact request')
    if phase in {'RETRACT', 'DONE'}:
        if desired < 0:
            raise ValueError('Withdrawal cannot request inward movement')
        return min(desired, .00025) if gap <= center + .002 else desired
    return max(-.0005, min(desired, .0005)) if gap <= center + .003 else desired
