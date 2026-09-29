"""Stop at every retained waypoint, preserving original joint-space edges.

No curve fitting, shortcutting or near-collinear simplification. Every returned
command still needs the original mesh/posture check; this is not physical proof.
"""
import numpy as np
from sim.expert_quality_20260924.candidate_planner import discrete_dynamic_peaks


def checked_edge_commands(path,*,maximum_seconds=120.):
    path=np.asarray(path,float)
    if (path.ndim!=2 or path.shape[1:]!=(7,) or len(path)<2 or not np.isfinite(path).all()
            or not np.isfinite(maximum_seconds) or maximum_seconds<1/.7):
        raise ValueError('Finite seven-joint route and original duration budget required')
    original=path.copy()
    # Only exactly identical adjacent knots can be omitted without changing edges.
    path=path[np.r_[True,np.any(np.diff(path,axis=0)!=0,axis=1)]]
    dt=1/240;minimum=1/.7
    if len(path)==1:
        commands=np.repeat(path,int(np.ceil(minimum/.05))*12,axis=0)
        steps=np.array([len(commands)])
    else:
        delta=np.max(abs(np.diff(path,axis=0)),axis=1)
        # Quintic smoothstep maxima: 1.875, 10/sqrt(3), 60.
        seconds=np.maximum.reduce([np.full(len(delta),minimum/len(delta)),
            1.875*delta/.175,np.sqrt((10/np.sqrt(3))*delta/.5),np.cbrt(60*delta/1.5)])
        for attempt in range(12):
            steps=np.maximum(12,np.ceil(seconds/.05).astype(int)*12)
            if steps.sum()*dt>maximum_seconds+1e-9:raise ValueError('Checked-edge timing exceeds original duration budget')
            blocks=[]
            for start,end,n in zip(path[:-1],path[1:],steps):
                u=np.arange(1,n+1)/n;s=10*u**3-15*u**4+6*u**5
                block=start+s[:,None]*(end-start);block[-1]=end;blocks.append(block)
            commands=np.concatenate(blocks)
            peaks=discrete_dynamic_peaks(original,commands,dt)
            ratio=max(peaks['velocity_rad_s']/.175,np.sqrt(peaks['acceleration_rad_s2']/.5),
                np.cbrt(peaks['jerk_rad_s3']/1.5))
            if ratio<=1+1e-10:break
            seconds=steps*dt*max(1.05,ratio*1.01)
        else:raise ValueError('Checked-edge discrete derivative budget exhausted')
    peaks=discrete_dynamic_peaks(original,commands,dt)
    if (peaks['velocity_rad_s']>.175+1e-9 or peaks['acceleration_rad_s2']>.5+1e-9
            or peaks['jerk_rad_s3']>1.5+1e-9):raise ValueError('Checked-edge sampled dynamics refused')
    return commands,dict(method='checked_edges_v1',unique_knots=len(path),original_knots=len(original),
        duration_s=len(commands)*dt,segment_steps=steps.tolist(),peaks=peaks,
        stationary_at_each_retained_waypoint=True,velocity_limit_rad_s=.175,acceleration_limit_rad_s2=.5,
        jerk_limit_rad_s3=1.5,geometry='piecewise quintic time law on original joint-space edges; no spatial smoothing',
        scope='Proposed commands only; all sampled mesh/posture gates and actual physics remain required')
