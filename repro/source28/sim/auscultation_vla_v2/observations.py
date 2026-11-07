"""Measured single-camera deployment interface; not a tracker or calibration proof."""
from dataclasses import dataclass
import numpy as np
from .hall import HallPacket, default_calibration

STATE_FEATURES = (
    tuple('q_%d_rad' % i for i in range(7))
    + tuple('dq_%d_rad_s' % i for i in range(7))
    + tuple('target_relative_%s_m' % a for a in 'xyz')
    + tuple('normal_tool_%s' % a for a in 'xyz')
    + ('hall_voltage_V', 'hall_spring_estimate_N', 'target_confidence')
    + tuple('target_velocity_tool_%s_m_s' % a for a in 'xyz')
    + tuple('target_rotation_tool_r%dc%d' % (r,c) for c in range(2) for r in range(3))
)


@dataclass
class RobotState:
    q: np.ndarray
    dq: np.ndarray
    wrist_position_base_m: np.ndarray  # FK from measured robot joints
    wrist_rotation_base: np.ndarray
    hall: HallPacket
    time_s: float


@dataclass
class TargetEstimate:
    position_base_m: np.ndarray
    rotation_base: np.ndarray  # z column is outward surface normal
    velocity_base_m_s: np.ndarray
    confidence: float
    time_s: float
    revision: int  # planning epoch, not a camera frame counter
    source: str


@dataclass
class Observation:
    rgb: np.ndarray
    depth_m: np.ndarray
    depth_valid: np.ndarray
    intrinsics: np.ndarray
    camera_to_base: np.ndarray
    rgb_time_s: float
    depth_time_s: float
    robot: RobotState
    target: TargetEstimate
    instruction: str


def finite(value, shape, name):
    arr=np.asarray(value, dtype=float)
    if arr.shape!=shape or not np.isfinite(arr).all():
        raise ValueError('invalid '+name)
    return arr


def rotation(value, name):
    arr=finite(value,(3,3),name)
    if not np.allclose(arr.T@arr,np.eye(3),atol=1e-5,rtol=0) or not np.isclose(np.linalg.det(arr),1.,atol=1e-5):
        raise ValueError('invalid rigid rotation '+name)
    return arr


def encode(obs, *, now_s, max_age_s=.15, max_rgbd_skew_s=.02, minimum_confidence=.8,
           hall_calibration=None, max_hall_age_s=.05, tool_tip_offset_m=.1262):
    """Return raw task state. Dataset-fitted normalization is a separate step.

    Target geometry must come from the actual RGB-D tracker. A source label is
    an interface constraint, not evidence that a tracker estimates it correctly.
    All times must already be synchronized to the consumer's monotonic clock.
    """
    for value,name in [(now_s,'now'),(max_age_s,'max_age'),(max_rgbd_skew_s,'max_skew'),(minimum_confidence,'confidence threshold')]:
        finite(value,(),name)
    if now_s<0 or max_age_s<=0 or max_rgbd_skew_s<0 or not 0<=minimum_confidence<=1:
        raise ValueError('invalid observation limits')
    if not isinstance(obs,Observation): raise ValueError('expected typed observation')
    if not isinstance(obs.rgb,np.ndarray) or obs.rgb.dtype!=np.uint8 or obs.rgb.ndim!=3 or obs.rgb.shape[2]!=3 or min(obs.rgb.shape[:2])<2:
        raise ValueError('RGB must be a nonempty uint8 HxWx3 frame')
    depth=np.asarray(obs.depth_m); valid=np.asarray(obs.depth_valid)
    if depth.shape!=obs.rgb.shape[:2] or not np.issubdtype(depth.dtype,np.floating) or valid.shape!=depth.shape or valid.dtype!=bool or not valid.any():
        raise ValueError('invalid metric depth or validity mask')
    if not np.isfinite(depth[valid]).all() or np.any(depth[valid]<=0):
        raise ValueError('invalid depth marked valid')
    k=finite(obs.intrinsics,(3,3),'intrinsics')
    if k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1],atol=1e-8):
        raise ValueError('invalid intrinsics')
    transform=finite(obs.camera_to_base,(4,4),'camera calibration')
    rotation(transform[:3,:3],'camera calibration')
    if not np.allclose(transform[3],[0,0,0,1],atol=1e-8): raise ValueError('invalid calibration bottom row')
    robot,target=obs.robot,obs.target
    if not isinstance(robot,RobotState) or not isinstance(target,TargetEstimate):
        raise ValueError('expected typed robot state and target')
    calibration=default_calibration() if hall_calibration is None else hall_calibration
    hall=calibration.decode(robot.hall,now_s=now_s,max_age_s=max_hall_age_s)
    if hall.status in {'SCREENED_LIMIT','HARD_LIMIT'}:
        raise ValueError('Hall calibration force limit: independent unloading required')
    for stamp in [obs.rgb_time_s,obs.depth_time_s,robot.time_s,target.time_s]:
        finite(stamp,(),'timestamp')
        if stamp<0 or stamp>now_s+1e-9 or now_s-stamp>max_age_s+1e-9:
            raise ValueError('stale or future observation')
    if abs(obs.rgb_time_s-obs.depth_time_s)>max_rgbd_skew_s+1e-9:
        raise ValueError('RGB/depth synchronization failed')
    if target.source!='rgbd_surface_tracking' or type(target.revision) is not int or target.revision<0:
        raise ValueError('invalid target provenance or planning revision')
    confidence=float(finite(target.confidence,(),'target confidence'))
    if not minimum_confidence<=confidence<=1: raise ValueError('target not observable with required confidence')
    if not isinstance(obs.instruction,str) or not obs.instruction.strip(): raise ValueError('missing task instruction')
    q=finite(robot.q,(7,),'q'); dq=finite(robot.dq,(7,),'dq')
    wrist=finite(robot.wrist_position_base_m,(3,),'measured-joint FK wrist position')
    tcp_rotation=rotation(robot.wrist_rotation_base,'measured-joint FK wrist rotation')
    finite(tool_tip_offset_m,(),'uncompressed tool offset')
    if tool_tip_offset_m<=calibration.travel_m:
        raise ValueError('invalid tool geometry')
    # 0.1262 m is the existing scene/CAD convention, not a new physical calibration.
    # Real and sim use Hall-estimated travel here; never accept a truth TCP that
    # already contains the simulator's perfectly observed slider position.
    tcp=wrist+tcp_rotation[:,2]*(tool_tip_offset_m-hall.displacement_estimate_m)
    target_position=finite(target.position_base_m,(3,),'target')
    target_rotation=rotation(target.rotation_base,'target')
    velocity=finite(target.velocity_base_m_s,(3,),'target velocity')
    inverse=tcp_rotation.T
    state=np.concatenate([q,dq,inverse@(target_position-tcp),inverse@target_rotation[:,2],
        [hall.hall_voltage_v,hall.spring_force_estimate_n,confidence],inverse@velocity,
        (inverse@target_rotation)[:,:2].T.reshape(-1)]).astype(np.float32)
    if state.shape!=(32,) or not np.isfinite(state).all(): raise ValueError('invalid encoded state')
    return dict(images={'base_0_rgb':obs.rgb.copy(),'left_wrist_0_rgb':np.zeros_like(obs.rgb),'right_wrist_0_rgb':np.zeros_like(obs.rgb)},
        image_masks={'base_0_rgb':True,'left_wrist_0_rgb':False,'right_wrist_0_rgb':False},state=state,prompt=obs.instruction)
