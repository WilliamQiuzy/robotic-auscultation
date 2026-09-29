"""Fixed H10/E5 terminal-velocity estimates for 50 ms reference tracking.

These hints do not retime or filter the supplied positions. A bounded estimate
does not establish feasibility of the raw timed waypoints. Always retain raw
secant exceedances and actual command-to-target error in the execution record.
The caller owns chunk identity, freshness, boundary checks and persistent q/v/a.
"""
import numpy as np

PERIOD_S = .05
VELOCITY_LIMIT_RAD_S = .175

# Exact least-squares quadratic derivative weights, in units of samples^-1.
# For k=0,1, use q[0:5] and evaluate at k itself (not the window midpoint).
# k=2,3,4 use centered q[k-2:k+3]. Subtract q[k] before applying these
# zero-sum weights to remove the constant position offset numerically.
_WEIGHTS = (
    np.array([-54.,13.,40.,27.,-26.])/70,
    np.array([-34.,3.,20.,17.,-6.])/70,
    np.array([-2.,-1.,0.,1.,2.])/10,
)


def velocity_hints(chunk, method='quadratic5'):
    """Return fresh velocity/raw-secant arrays and explicit diagnostic maxima.

    For quadratic5, fit q[j] = b0 + b1*tau + b2*tau^2 to five points with
    tau=(j-k)*0.05, and use b1 at original target k, t0+(k+1)*0.05.
    No measured q, earlier chunk, clipping, padding or fitted position is used.
    The original terminal acceleration remains zero in the caller's executor.
    """
    if method not in ('quadratic5','adjacent','zero'):
        raise ValueError('Unknown fixed velocity estimator')
    raw = np.asarray(chunk)
    if raw.shape != (10,7) or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError('Motion hints require one finite numeric H10x7 chunk')
    q = raw.astype(np.float64,copy=True)
    with np.errstate(over='ignore',invalid='ignore'):
        secant = np.diff(q[:6],axis=0)/PERIOD_S
        if method == 'quadratic5':
            velocity = np.stack([_WEIGHTS[min(k,2)] @ (q[max(0,k-2):max(0,k-2)+5]-q[k])
                                 /PERIOD_S for k in range(5)])
        elif method == 'adjacent':
            velocity = secant.copy()
        else:
            velocity = np.zeros((5,7))
    if not np.isfinite(secant).all() or not np.isfinite(velocity).all():
        raise ValueError('Nonfinite derived velocity')
    peak = float(np.max(np.abs(velocity)))
    if peak > VELOCITY_LIMIT_RAD_S:
        raise ValueError(f'Estimated terminal velocity max {peak:.9g} exceeds 0.175 rad/s; refused without clipping')
    raw_peak = float(np.max(np.abs(secant)))
    exceeded = np.abs(secant) > VELOCITY_LIMIT_RAD_S
    return dict(method=method,velocity=velocity,raw_secant=secant,
        raw_secant_exceeds_limit=bool(exceeded.any()),summary=dict(
            estimated_velocity_max_abs_rad_s=peak,raw_secant_max_abs_rad_s=raw_peak,
            raw_secant_excess_max_rad_s=max(0.,raw_peak-VELOCITY_LIMIT_RAD_S),
            raw_secant_exceedance_count=int(exceeded.sum()),
            estimated_minus_raw_max_abs_rad_s=float(np.max(np.abs(velocity-secant)))))
