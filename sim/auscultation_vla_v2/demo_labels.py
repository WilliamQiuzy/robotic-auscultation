"""Seven-dimensional teacher command labels from synchronized recorded frames.

The next approved joint command supplies supervision. Current measured q and
current Hall-estimated compression define the command origin. Future measured
state, future Hall readings and simulator TCP truth are never substituted.
This conversion alone does not qualify collision/contact safety or a rollout.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .hall import HallPacket,default_calibration
from .chunks import PHASES


def label_pair(current,following,next_joint_command,*,kinematics=None,calibration=None):
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    if (current.get('eligible') is not True or following.get('eligible') is not True
        or current.get('phase') not in PHASES or current['phase']!=following.get('phase')
        or not current.get('target') or current['target']!=following.get('target')):
        raise ValueError('action label crosses an ineligible phase or target boundary')
    t=float(current['time_s']);next_time=float(following['time_s'])
    if not np.isfinite([t,next_time]).all() or not np.isclose(next_time-t,.05,atol=1e-7,rtol=0):
        raise ValueError('action label requires adjacent 20 Hz observations')
    calibration=default_calibration() if calibration is None else calibration
    hall=calibration.decode(HallPacket(**current['hall']),now_s=t)
    if hall.status in {'SCREENED_LIMIT','HARD_LIMIT'}:raise ValueError('unsafe current Hall feedback')
    kin=OpenArmKinematics() if kinematics is None else kinematics
    current_tcp=kin.tcp(np.asarray(current['q'],float),hall.displacement_estimate_m)
    commanded_tcp=kin.tcp(np.asarray(next_joint_command,float),hall.displacement_estimate_m)
    translation=commanded_tcp[:3,3]-current_tcp[:3,3]
    rotation=Rotation.from_matrix(commanded_tcp[:3,:3]@current_tcp[:3,:3].T).as_rotvec()
    force=4. if current['phase'] in {'CONTACT','HOLD'} else 0.
    return np.r_[translation,rotation,force]
