"""Unfiltered motion metrics. Different derivative estimators stay separate."""
import numpy as np


def motion_metrics(times, positions, velocities):
    t=np.asarray(times,dtype=np.float64)
    q=np.asarray(positions,dtype=np.float64);dq=np.asarray(velocities,dtype=np.float64)
    if (t.ndim!=1 or len(t)<9 or q.shape!=(len(t),7) or dq.shape!=q.shape
            or not all(np.isfinite(x).all() for x in (t,q,dq))):
        raise ValueError('Need finite paired 7-joint measurements')
    dt=np.diff(t)
    if np.any(dt<=0) or not np.allclose(dt,dt[0],atol=1e-8,rtol=1e-6):
        raise ValueError('Need strictly increasing uniform physical timestamps')
    def peak(x,stamp):
        i,j=np.unravel_index(np.argmax(np.abs(x)),x.shape)
        return dict(max=float(abs(x[i,j])),signed_value=float(x[i,j]),
                    joint_index=int(j),time_s=float(stamp[i]))
    forward={};centered={};x=q.copy();y=q.copy()
    for order,name in enumerate(('velocity_rad_s','acceleration_rad_s2','jerk_rad_s3'),1):
        x=np.diff(x,axis=0)/dt[0]
        y=np.gradient(y,t,axis=0,edge_order=2)
        forward[name]=peak(x,t[order:])
        # Remove all boundary stencils affected by one-sided estimates.
        centered[name]=peak(y[order:-order],t[order:-order])
    a=np.diff(dq,axis=0)/dt[0];j=np.diff(a,axis=0)/dt[0]
    return dict(samples=len(t),period_s=float(dt[0]),forward=forward,centered=centered,
        stored_velocity=dict(velocity_rad_s=peak(dq,t),acceleration_rad_s2=peak(a,t[1:]),
                             jerk_rad_s3=peak(j,t[2:])),
        filtering='none',centered_boundary='discard order samples at both ends')
