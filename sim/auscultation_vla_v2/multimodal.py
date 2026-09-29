"""Assemble deployable inputs from RGB-D, ADC and measured joints only."""
import hashlib
import numpy as np
from .hall import default_calibration
from .observations import Observation,RobotState,encode,finite
from .chunks import PHASES

INSTRUCTIONS={
    'FREE':'Move toward the designated back target through free space.',
    'APPROACH':'Approach the designated back target gently.',
    'CONTACT':'Establish gentle contact at the designated back target.',
    'HOLD':'Hold gentle contact at the designated back target.',
    'RETRACT':'Release contact and retract from the designated back target.',
}


class MultimodalAssembler:
    """An operator supplies a new pixel seed after target changes or track loss.

    Hold duration and stage progression belong to the explicit supervisor.
    This is an observation path, not an action executor or anatomy detector.
    """
    def __init__(self,intrinsics,camera_to_base,*,hall_calibration=None):
        from .tracking import SurfaceTracker
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        self.intrinsics=np.array(intrinsics,float,copy=True)
        self.camera_to_base=np.array(camera_to_base,float,copy=True)
        self.tracker=SurfaceTracker(self.intrinsics,self.camera_to_base)
        self.kinematics=OpenArmKinematics()
        self.calibration=default_calibration() if hall_calibration is None else hall_calibration
        self.target_key=None
        self._last_robot_time=-np.inf
        self._last_hall_time=-np.inf

    def observe(self,*,rgb,depth_m,depth_valid,rgb_time_s,depth_time_s,now_s,
                q,dq,robot_time_s,hall,phase,target_key,seed_pixel_xy=None):
        if phase not in PHASES:raise ValueError('Unknown requested phase')
        if not isinstance(target_key,str) or not target_key:raise ValueError('Missing target identity')
        q=finite(q,(7,),'measured q');dq=finite(dq,(7,),'measured dq')
        finite([now_s,robot_time_s],(2,),'robot timestamps')
        if not 0<=robot_time_s<=now_s or now_s-robot_time_s>.15:
            raise ValueError('Robot state is stale or future')
        hall_obs=self.calibration.decode(hall,now_s=now_s)
        # Repeated reads of a still-fresh sample are permitted, reordered data is not.
        if robot_time_s<self._last_robot_time or hall.sample_time_s<self._last_hall_time:
            raise ValueError('Robot or Hall sample time reversed')
        if hall_obs.status in {'SCREENED_LIMIT','HARD_LIMIT'}:
            raise ValueError('Hall force limit requires independent unloading')
        wrist=self.kinematics.fk(q)
        changed=target_key!=self.target_key
        if changed or self.tracker.lost:
            if seed_pixel_xy is None:raise ValueError('Explicit pixel seed required for target initialization')
        elif seed_pixel_xy is not None:
            raise ValueError('Continuous tracking must not be reseeded on every frame')
        times=dict(rgb_time_s=rgb_time_s,depth_time_s=depth_time_s,now_s=now_s)
        if changed or self.tracker.lost:
            target=self.tracker.initialize(rgb,depth_m,depth_valid,seed_pixel_xy,**times)
        else:
            target=self.tracker.update(rgb,depth_m,depth_valid,**times)
        robot=RobotState(q,dq,wrist[:3,3],wrist[:3,:3],hall,robot_time_s)
        obs=Observation(rgb,depth_m,depth_valid,self.intrinsics,self.camera_to_base,
                        rgb_time_s,depth_time_s,robot,target,INSTRUCTIONS[phase])
        encoded=encode(obs,now_s=now_s,hall_calibration=self.calibration)
        self.target_key=target_key
        self._last_robot_time=robot_time_s
        self._last_hall_time=hall.sample_time_s
        encoded['modality_receipt']=dict(
            schema='wam_measured_multimodal_input_v1',depth_used_for_geometry=True,
            target_source=target.source,wrist_source='fk_measured_joints',audio='absent',
            rgb_time_s=rgb_time_s,depth_time_s=depth_time_s,robot_time_s=robot_time_s,
            hall_sample_time_s=hall.sample_time_s,hall_raw=hall.raw,
            calibration_sha256=hall.calibration_sha256,
            target_revision=target.revision,phase=phase,target_key=target_key,
            rgb_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
            depth_sha256=hashlib.sha256(np.ascontiguousarray(depth_m).tobytes()).hexdigest())
        return encoded
