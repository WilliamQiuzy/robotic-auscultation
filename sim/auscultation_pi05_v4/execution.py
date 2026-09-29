"""Measured-pose composition and bounded continuous joint execution.

No planner, teacher path, force feedback displacement or hardware interface.
Ruckig targets are joint positions from independently validated pose IK.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .action_contract import bounded,supervised_force_reference


def compose_pose(measured_tcp,raw_action,phase):
    p=np.asarray(measured_tcp,float);a=np.asarray(raw_action,float)
    if p.shape!=(4,4) or a.shape!=(7,) or not np.isfinite(p).all():
        raise ValueError('Invalid measured pose/action')
    if (not np.allclose(p[3],[0,0,0,1],atol=1e-8) or
        not np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-6) or
        not np.isclose(np.linalg.det(p[:3,:3]),1.,atol=1e-6)):
        raise ValueError('Invalid measured rigid transform')
    semantic=supervised_force_reference(a[None],phase)
    if not bounded(semantic,phase):raise ValueError('Model pose proposal rejected; no pose clipping')
    goal=p.copy();goal[:3,3]+=a[:3]
    goal[:3,:3]=Rotation.from_rotvec(a[3:6]).as_matrix()@p[:3,:3]
    return goal


class JointTrajectory:
    """Continuous command state across new targets, including derivatives."""
    def __init__(self,q,*,dt=1/240,velocity=.25,acceleration=.5,jerk=8.):
        import ruckig
        self.module=ruckig;self.dt=float(dt)
        q=np.asarray(q,float)
        if q.shape!=(7,) or not np.isfinite(q).all() or not np.isfinite([dt,velocity,acceleration,jerk]).all() or min(dt,velocity,acceleration,jerk)<=0:
            raise ValueError('Invalid trajectory initialization')
        self.generator=ruckig.Ruckig(7,self.dt)
        self.input=ruckig.InputParameter(7);self.output=ruckig.OutputParameter(7)
        self.input.current_position=q.tolist();self.input.current_velocity=[0.]*7;self.input.current_acceleration=[0.]*7
        self.input.target_position=q.tolist();self.input.target_velocity=[0.]*7;self.input.target_acceleration=[0.]*7
        self.input.max_velocity=[velocity]*7;self.input.max_acceleration=[acceleration]*7;self.input.max_jerk=[jerk]*7
        self.velocity=np.zeros(7);self.acceleration=np.zeros(7)

    def set_target(self,q):
        q=np.asarray(q,float)
        if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('Invalid joint target')
        self.input.target_position=q.tolist()

    def step(self):
        result=self.generator.update(self.input,self.output)
        if result not in (self.module.Result.Working,self.module.Result.Finished):
            raise RuntimeError('Ruckig rejected continuous target: '+str(result))
        q=np.asarray(self.output.new_position,float)
        self.velocity=np.asarray(self.output.new_velocity,float)
        self.acceleration=np.asarray(self.output.new_acceleration,float)
        if not np.isfinite([q,self.velocity,self.acceleration]).all():raise RuntimeError('Nonfinite trajectory')
        self.output.pass_to_input(self.input)
        return q
