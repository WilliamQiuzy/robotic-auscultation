"""Opt-in native model-based gravity feedforward, with unchanged position drives.

Isaac 5.1's native generalized gravity forces are the DOF efforts required to
counteract gravity: use their positive sign directly. This controller reads no
contact/Hall signals and never changes gains, effort limits, physics or poses.
PD plus feedforward is an estimate, not measured or reconstructed motor torque.
This module is importable without Isaac for CPU contract tests.
"""
import numpy as np


SOURCE = 'native generalized gravity forces; model-based controller feedforward'
PERSISTENCE_FLOAT32_EPS_MULTIPLIER = 4.


class GravityFeedforwardRejected(ValueError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def gravity_feedforward_settings(fresh_config):
    """Only absence is implicit off; an explicit setting must be a JSON bool."""
    if 'gravity_feedforward' not in fresh_config:
        return None
    value = fresh_config['gravity_feedforward']
    if (not isinstance(value, dict) or set(value) != {'enabled'}
            or type(value['enabled']) is not bool):
        raise ValueError('fresh_expert.gravity_feedforward requires only explicit enabled: bool')
    return dict(enabled=value['enabled'])


def _numeric(value, label, shape=None):
    result = np.asarray(value)
    if (result.dtype.kind not in 'iuf' or (shape is not None and result.shape != shape)
            or not np.isfinite(result).all()):
        raise ValueError('Invalid gravity feedforward '+label)
    return result.astype(float, copy=True)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _diagnostic_value(value):
    """Keep malformed native input inspectable without writing NaN JSON."""
    if isinstance(value, np.ndarray):
        return _diagnostic_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) and np.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    return repr(value)


class NativeGravityFeedforward:
    """Read current native dynamics and return bounded efforts plus a receipt."""
    def __init__(self, robot, left, slider):
        # Inspect elements before NumPy could coerce a bool/float into an int.
        values = np.asarray(left, dtype=object)
        if (values.shape != (7,) or not all(_integer(v) and v >= 0 for v in values)
                or len(set(values)) != 7):
            raise ValueError('Gravity feedforward requires seven distinct integer indices')
        if not _integer(slider) or slider < 0 or slider in values:
            raise ValueError('Gravity feedforward slider must be distinct integer index')
        self.robot = robot
        self.left = values.astype(int)
        self.slider = int(slider)
        self.indices = np.r_[self.left, self.slider]

    def prepare(self, q_target):
        """Read only; reject both native gravity and estimated total over caps."""
        receipt = dict(enabled=True, source=SOURCE, learned_policy=False,
            real_hardware_validated=False, training_eligible=False,
            left_joint_indices=self.left.tolist(), slider_joint_index=self.slider,
            q_target_rad=None, q_rad=None, dq_rad_s=None, kp=None, kd=None,
            effort_limits_Nm=None, gravity_efforts_Nm=None,
            pd_effort_estimate_Nm=None, pd_plus_gravity_estimate_Nm=None,
            estimate_is_measured_motor_torque=False,
            pd_estimate_assumption='kp*(q_target-q)-kd*dq; zero desired velocity',
            gravity_sign='positive native counter-gravity effort, no scaling',
            requested_joint_indices=self.indices.tolist(), requested_efforts=None,
            requested_effort_units=['N m']*7+['N'], effort_bounds_passed=False,
            applied_efforts_persisted=False,
            raw_inputs={}, raw_input_nonfinite_representation='null',
            persistence_float32_eps_multiplier=PERSISTENCE_FLOAT32_EPS_MULTIPLIER)
        def read(key, operation, shape=None):
            receipt['preparation_stage'] = key
            value = operation()
            receipt['raw_inputs'][key] = _diagnostic_value(value)
            result = _numeric(value, key, shape)
            receipt[key] = (result[0] if shape == (1, 7) else result).tolist()
            return result
        try:
            target = read('q_target_rad', lambda: q_target, (7,))
            q = read('q_rad', lambda: self.robot.get_joint_positions(joint_indices=self.left), (7,))
            dq = read('dq_rad_s', lambda: self.robot.get_joint_velocities(joint_indices=self.left), (7,))
            receipt['preparation_stage'] = 'existing_gains'
            gains = self.robot.get_articulation_controller().get_gains()
            receipt['raw_inputs']['existing_gains'] = _diagnostic_value(gains)
            kp_raw, kd_raw = gains
            kp_all = read('kp_full', lambda: kp_raw)
            kd_all = read('kd_full', lambda: kd_raw)
            if (kp_all.ndim != 1 or kd_all.shape != kp_all.shape
                    or len(kp_all) <= int(max(self.indices))
                    or np.any(kp_all < 0.) or np.any(kd_all < 0.)):
                raise ValueError('Invalid gravity feedforward full articulation gains')
            kp, kd = kp_all[self.left], kd_all[self.left]
            receipt.update(kp=kp.tolist(), kd=kd.tolist())
            with np.errstate(over='ignore', invalid='ignore'):
                pd = read('pd_effort_estimate_Nm', lambda: kp*(target-q)-kd*dq, (7,))
            view = self.robot._articulation_view
            gravity = read('gravity_efforts_Nm',
                lambda: view.get_generalized_gravity_forces(joint_indices=self.left), (1, 7))[0]
            efforts = np.r_[gravity, -1.5]
            receipt['requested_efforts'] = efforts.tolist()
            with np.errstate(over='ignore', invalid='ignore'):
                estimate = read('pd_plus_gravity_estimate_Nm', lambda: pd+gravity, (7,))
            caps = read('effort_limits_Nm',
                lambda: view.get_max_efforts(joint_indices=self.left), (1, 7))[0]
            if np.any(caps <= 0.):
                raise ValueError('Gravity feedforward requires positive existing effort limits')
            for name, values in [('gravity', gravity), ('pd_plus_gravity', estimate)]:
                receipt['preparation_stage'] = name+'_effort_limit'
                if np.any(np.abs(values) > caps):
                    raise GravityFeedforwardRejected(name+' exceeds existing effort limits', receipt)
        except Exception as error:
            receipt['preparation_error'] = repr(error)
            if isinstance(error, GravityFeedforwardRejected):
                raise
            raise GravityFeedforwardRejected('Gravity feedforward preparation rejected: '+str(error), receipt) from error
        receipt['preparation_stage'] = 'ready'
        receipt['effort_bounds_passed'] = True
        return self.indices.copy(), efforts, receipt

    def verify_applied(self, receipt):
        """Check requested effort persistence after the inherited position step.

        Applied command efforts are not measured joint/motor efforts. Mutate the
        row's receipt before raising so post-step rejection cannot hide evidence.
        """
        receipt['applied_efforts_persisted'] = False
        try:
            applied = _numeric(self.robot.get_applied_joint_efforts(joint_indices=self.indices),
                               'post-step applied efforts', (8,))
        except Exception as error:
            receipt['post_step_applied_efforts_error'] = repr(error)
            raise GravityFeedforwardRejected('Invalid post-step applied efforts: '+str(error), receipt) from error
        receipt['post_step_applied_efforts'] = applied.tolist()
        requested = np.asarray(receipt['requested_efforts'], float)
        tolerance = (PERSISTENCE_FLOAT32_EPS_MULTIPLIER*np.finfo(np.float32).eps
                     * np.maximum(1., np.abs(requested)))
        error = np.abs(applied-requested)
        receipt['applied_effort_absolute_error'] = error.tolist()
        receipt['applied_effort_tolerance'] = tolerance.tolist()
        if np.any(error > tolerance):
            raise GravityFeedforwardRejected('Requested gravity/slider efforts did not persist', receipt)
        receipt['applied_efforts_persisted'] = True
