"""Continuous bounded joint execution; no pose inversion or teacher path."""
import numpy as np

FINISHED_ROUNDOFF_ULPS=64


class FinishedStationaryRoundoffError(RuntimeError):
    """The native Finished profile exceeds the unchanged normalization bound."""


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
        self.terminal_normalizations=[]

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
        if result==self.module.Result.Finished:
            self._normalize_finished_stationary(q)
            q=np.asarray(self.output.new_position,float)
            self.velocity=np.asarray(self.output.new_velocity,float)
            self.acceleration=np.asarray(self.output.new_acceleration,float)
        self.output.pass_to_input(self.input)
        return q

    def _normalize_finished_stationary(self,q):
        """Canonicalize only native Finished roundoff at a valid stopped target.

        Native OutputParameter q/v/a are read-only. Reevaluate the exact
        stationary endpoint with the same native objects, retaining output
        time, then pass that actual output to input. Never touch Working or
        nonzero endpoint derivatives; actor/URDF predicates remain strict.
        """
        if self.input.target_velocity!=[0.]*7 or self.input.target_acceleration!=[0.]*7:return
        target=np.asarray(self.input.target_position,float)
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        kin=OpenArmKinematics()
        if np.any(target<kin.lower_limits) or np.any(target>kin.upper_limits):
            raise RuntimeError('Finished stationary target outside strict joint limits')
        if np.array_equal(q,target) and np.array_equal(self.velocity,np.zeros(7)) and np.array_equal(self.acceleration,np.zeros(7)):return
        tolerance=FINISHED_ROUNDOFF_ULPS*np.finfo(np.float64).eps
        if (np.any(abs(q-target)>tolerance*np.maximum(1.,abs(target))) or
                np.any(abs(self.velocity)>tolerance) or np.any(abs(self.acceleration)>tolerance)):
            error=FinishedStationaryRoundoffError('Finished stationary output exceeds numerical roundoff bound')
            error.details=dict(raw_position=q.tolist(),raw_velocity=self.velocity.tolist(),
                raw_acceleration=self.acceleration.tolist(),target_position=target.tolist(),
                roundoff_ulps=FINISHED_ROUNDOFF_ULPS,output_time=self.output.time)
            raise error
        clock=self.output.time
        record=dict(schema='m2_finished_stationary_target_roundoff_v1',output_time=clock,
            raw_position=q.tolist(),raw_velocity=self.velocity.tolist(),raw_acceleration=self.acceleration.tolist(),
            target_position=target.tolist(),roundoff_ulps=FINISHED_ROUNDOFF_ULPS)
        self.input.current_position=target.tolist();self.input.current_velocity=[0.]*7;self.input.current_acceleration=[0.]*7
        status=self.generator.update(self.input,self.output)
        if (status!=self.module.Result.Finished or self.output.new_position!=target.tolist() or
                self.output.new_velocity!=[0.]*7 or self.output.new_acceleration!=[0.]*7):
            raise RuntimeError('Native exact stationary endpoint normalization failed')
        self.output.time=clock
        self.terminal_normalizations.append(record)
