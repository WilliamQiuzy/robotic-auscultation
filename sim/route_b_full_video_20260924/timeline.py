"""Continuous 1x timeline; retains final fractional frame and never cuts point tails."""
import math
import numpy as np


def full_schedule(times, pose_rows, fps=15):
    times=np.asarray(times, dtype=float)
    pose_rows=np.asarray(pose_rows)
    if fps not in (10,15,30): raise ValueError('unsupported fps')
    if times.ndim!=1 or len(times)<16 or not np.isfinite(times).all(): raise ValueError('invalid clock')
    if not np.allclose(np.diff(times),1/240,atol=1e-7,rtol=0): raise ValueError('trace time gap')
    if pose_rows.ndim!=1 or len(pose_rows)<2 or not np.issubdtype(pose_rows.dtype,np.integer): raise ValueError('invalid pose indices')
    if np.any(np.diff(pose_rows)<=0) or pose_rows[0]!=7 or pose_rows[-1]>=len(times): raise ValueError('invalid pose bounds')
    if np.max(np.diff(pose_rows))>8 or len(times)-1-pose_rows[-1]>7: raise ValueError('pose time gap')
    duration=len(times)/240
    count=int(math.ceil(duration*fps-1e-7))
    # Frames sample the end of contiguous exposure intervals, exactly as prior clips.
    # The last exposure is shorter: retain its final source row and pad the video
    # by < 1/fps seconds. No simulated time is removed or rescaled.
    end_steps=np.minimum(np.arange(1,count+1)*(240//fps),len(times))
    rows=end_steps-1
    indices=np.searchsorted(pose_rows,rows,side='right')-1
    if np.any(indices<0) or np.max(rows-pose_rows[indices])>7: raise ValueError('unavailable pose')
    return rows,indices,(times[0]-1/240)+end_steps/240
