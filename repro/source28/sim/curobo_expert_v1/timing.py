"""C2 spatial interpolation with sampled dynamic bounds; collision checked later."""
import time
import numpy as np
from scipy.interpolate import CubicSpline
from sim.expert_quality_20260924.candidate_planner import joint_path,discrete_dynamic_peaks


def condition_path(path,tolerance_rad=2e-5):
    """Deterministic RDP removes dense quantization knots, preserving endpoints.

    Every removed original sample is <= tolerance from its retained chord.
    The resulting spline is a new proposal: executed collision checks are still
    mandatory and do not inherit clearance from this numerical tolerance.
    """
    path=joint_path(path)
    if not np.isfinite(tolerance_rad) or tolerance_rad<=0:raise ValueError('invalid conditioning tolerance')
    keep={0,len(path)-1};pending=[(0,len(path)-1)]
    while pending:
        a,b=pending.pop()
        if b-a<2:continue
        delta=path[b]-path[a];square=float(delta@delta)
        points=path[a+1:b]
        u=np.clip((points-path[a])@delta/square,0.,1.) if square>1e-28 else np.zeros(len(points))
        error=np.linalg.norm(points-(path[a]+u[:,None]*delta),axis=1)
        index=int(np.argmax(error))
        if error[index]>tolerance_rad:
            split=a+1+index;keep.add(split);pending.extend([(a,split),(split,b)])
    indices=sorted(keep)
    return path[indices],{'method':'RDP Euclidean seven-joint chord error bound',
        'maximum_removed_sample_chord_error_rad':tolerance_rad,'original_knots':len(path),
        'retained_knots':len(indices),'retained_indices':indices,
        'scope':'conditioning proposal only; final spline samples require full mesh/posture verification'}


def smooth_commands(path, *, speed_limit=.25, minimum_seconds=1., dt=1/240,
                    block_size=12, acceleration_limit=1., jerk_limit=1.5,
                    maximum_seconds=300., maximum_compute_seconds=15., return_report=False):
    path=joint_path(path)
    if (not np.isfinite([speed_limit,minimum_seconds,dt,acceleration_limit,jerk_limit,maximum_seconds,maximum_compute_seconds]).all()
            or min(speed_limit,minimum_seconds,dt,acceleration_limit,jerk_limit,maximum_seconds,maximum_compute_seconds)<=0
            or type(block_size)!=int or block_size<1):raise ValueError('invalid smooth timing bounds')
    original=path.copy();started=time.monotonic();path,conditioning=condition_path(path)
    keep=np.r_[True,np.max(np.abs(np.diff(path,axis=0)),axis=1)>1e-12]
    path=path[keep]
    cumulative=np.r_[0.,np.cumsum(np.max(np.abs(np.diff(path,axis=0)),axis=1))]
    if len(path)>1:
        spline=CubicSpline(cumulative/cumulative[-1],path,axis=0,bc_type='natural')
    seconds=max(minimum_seconds,1.875*cumulative[-1]/speed_limit);initial=seconds;peaks=None
    for attempt in range(1,25):
        steps=int(np.ceil(seconds/(dt*block_size)))*block_size;duration=steps*dt
        if duration>maximum_seconds+1e-9 or time.monotonic()-started>maximum_compute_seconds:break
        u=np.arange(1,steps+1)/steps;s=10*u**3-15*u**4+6*u**5
        commands=spline(s) if len(path)>1 else np.repeat(path,steps,axis=0)
        commands[-1]=path[-1]
        peaks=discrete_dynamic_peaks(original,commands,dt)
        if (peaks['velocity_rad_s']<=speed_limit+1e-9 and peaks['acceleration_rad_s2']<=acceleration_limit+1e-9
                and peaks['jerk_rad_s3']<=jerk_limit+1e-9):
            report={'duration_s':duration,'attempts':attempt,'peaks':peaks,'velocity_limit_rad_s':speed_limit,
                'acceleration_limit_rad_s2':acceleration_limit,'jerk_limit_rad_s3':jerk_limit,
                'maximum_seconds':maximum_seconds,'maximum_iterations':24,'initial_speed_only_seconds':initial,
                'maximum_compute_seconds':maximum_compute_seconds,'compute_seconds':time.monotonic()-started,
                'endpoint_stationary_padding_samples':3,'physics_dt_s':dt,
                'differentiation':'forward finite differences through order 3; includes stationary boundaries',
                'geometry':'natural C2 cubic joint-space interpolation through conditioned proposal knots; all executed samples require separate collision and posture checks',
                'conditioning':conditioning,
                'scope':'sampled command bounds, not measured dynamics or continuous collision certification'}
            return (commands,report) if return_report else commands
        seconds=duration*1.05*max(1.15,peaks['velocity_rad_s']/speed_limit,
            np.sqrt(peaks['acceleration_rad_s2']/acceleration_limit),np.cbrt(peaks['jerk_rad_s3']/jerk_limit))
    raise RuntimeError('smooth trajectory timing budget exhausted: '+str(peaks))
