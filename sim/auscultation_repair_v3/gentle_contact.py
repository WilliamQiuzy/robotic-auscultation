"""Rate requests for a checked normal executor, never direct joint commands."""
import math


def limit_near_contact(phase,desired,gap_m,nominal_contact_gap_m):
    if phase not in {'CONTACT','HOLD','RETRACT','DONE','APPROACH'}:
        raise ValueError('Unknown expert contact phase')
    if not all(math.isfinite(x) for x in (desired,gap_m,nominal_contact_gap_m)):
        raise ValueError('Nonfinite contact command')
    if phase in {'RETRACT','DONE'}:
        if desired<0:raise ValueError('Withdrawal must not request inward motion')
        if gap_m<=nominal_contact_gap_m+.002:return min(desired,.0005)
    elif gap_m<=nominal_contact_gap_m+.003:
        return max(-.001,min(desired,.001))
    return desired
