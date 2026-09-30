"""Opt-in native model-based gravity feedforward, with unchanged position drives.

Isaac 5.1's native generalized gravity forces are the DOF efforts required to
counteract gravity: use their positive sign directly. This controller reads no
contact/Hall signals and never changes gains, effort limits, physics or poses.
This variant requires existing angular acceleration drives. The native-gain PD
expression is an acceleration estimate in rad/s^2, never a torque to add to G.
Implicit-drive maxForce remains unchanged; external gravity is separately
bounded. No total actuator torque bound or measured torque is established.
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
    if isinstance(value, dict):
        return {str(key): _diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) and np.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    return repr(value)


def read_native_angular_drive_modes(stage, robot, left):
    """Read existing USD drives through actual native DOF paths, never Apply.

    Isaac's get_effort_modes may author a missing DriveAPI. This read-only path
    instead rejects a missing binding/API and checks the live type every call.
    """
    from pxr import UsdPhysics
    indices = np.asarray(left, dtype=object)
    paths = np.asarray(robot._articulation_view._dof_paths, dtype=object)
    if (indices.shape != (7,) or not all(_integer(v) and v >= 0 for v in indices)
            or len(set(indices)) != 7 or paths.ndim != 2 or paths.shape[0] != 1
            or max(indices) >= paths.shape[1]):
        raise ValueError('Invalid native angular drive DOF binding')
    bindings = []
    for ordinal, index in enumerate(indices, 1):
        name = f'openarm_left_joint{ordinal}'
        mapped_index = robot.get_dof_index(name)
        path = paths[0, index]
        if (not _integer(mapped_index) or mapped_index != index
                or not isinstance(path, str) or not path.startswith('/')):
            raise ValueError('Native angular drive name/index binding differs')
        prim = stage.GetPrimAtPath(path)
        if (not prim.IsValid() or prim.GetName() != name
                or not prim.HasAPI(UsdPhysics.DriveAPI, 'angular')):
            raise ValueError('Missing existing angular DriveAPI or joint binding: '+path)
        drive = UsdPhysics.DriveAPI(prim, 'angular')
        bindings.append(dict(joint_index=int(index), joint_name=name, joint_path=path,
                             drive_instance='angular', drive_type=drive.GetTypeAttr().Get()))
    return bindings


class NativeGravityFeedforward:
    """Read current native dynamics and return bounded efforts plus a receipt."""
    def __init__(self, robot, left, slider, *, drive_mode_provider):
        # Inspect elements before NumPy could coerce a bool/float into an int.
        values = np.asarray(left, dtype=object)
        if (values.shape != (7,) or not all(_integer(v) and v >= 0 for v in values)
                or len(set(values)) != 7):
            raise ValueError('Gravity feedforward requires seven distinct integer indices')
        if not _integer(slider) or slider < 0 or slider in values:
            raise ValueError('Gravity feedforward slider must be distinct integer index')
        if not callable(drive_mode_provider):
            raise ValueError('Gravity feedforward requires a live USD drive mode provider')
        self.robot = robot
        self.drive_mode_provider = drive_mode_provider
        self.left = values.astype(int)
        self.slider = int(slider)
        self.indices = np.r_[self.left, self.slider]

    def prepare(self, q_target):
        """Require acceleration drives; bound native gravity independently."""
        receipt = dict(schema='native_acceleration_drive_gravity_v2',
            enabled=True, source=SOURCE, learned_policy=False,
            real_hardware_validated=False, training_eligible=False,
            left_joint_indices=self.left.tolist(), slider_joint_index=self.slider,
            q_target_rad=None, q_rad=None, dq_rad_s=None, kp=None, kd=None,
            effort_limits_Nm=None, gravity_efforts_Nm=None,
            drive_mode_bindings=None, required_drive_mode='acceleration',
            pd_acceleration_estimate_rad_s2=None, pd_estimate_units='rad/s^2',
            pd_estimate_is_measured_acceleration=False,
            pd_estimate_assumption='kp*(q_target-q)-kd*dq; zero desired velocity',
            total_actuator_torque_validated=False,
            implicit_drive_max_force_unchanged=True, external_gravity_effort_separately_bounded=True,
            torque_validation_scope='native implicit-drive maxForce unchanged; external G independently bounded; combined total not bounded by this diagnostic',
            historical_units_correction='v1 PD-as-Nm and PD-plus-G-as-Nm fields/gate were invalid for acceleration drives and are removed',
            gravity_sign='positive native counter-gravity effort, no scaling',
            requested_joint_indices=self.indices.tolist(), requested_efforts=None,
            requested_effort_units=['N m']*7+['N'], gravity_effort_bounds_passed=False,
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
            receipt['preparation_stage'] = 'drive_mode_bindings'
            bindings = self.drive_mode_provider()
            receipt['drive_mode_bindings'] = _diagnostic_value(bindings)
            if not isinstance(bindings, list) or len(bindings) != 7:
                raise ValueError('Seven native angular drive bindings required')
            seen_paths = set()
            for ordinal, (index, binding) in enumerate(zip(self.left, bindings), 1):
                if (not isinstance(binding, dict)
                        or not _integer(binding.get('joint_index'))
                        or binding['joint_index'] != index
                        or binding.get('joint_name') != f'openarm_left_joint{ordinal}'
                        or binding.get('drive_instance') != 'angular'
                        or not isinstance(binding.get('joint_path'), str)
                        or not binding['joint_path'].startswith('/')
                        or binding['joint_path'] in seen_paths):
                    raise ValueError('Native angular drive binding differs from selected DOFs')
                seen_paths.add(binding['joint_path'])
                if binding.get('drive_type') != 'acceleration':
                    raise ValueError('Gravity feedforward variant requires existing acceleration drives')
            with np.errstate(over='ignore', invalid='ignore'):
                read('pd_acceleration_estimate_rad_s2', lambda: kp*(target-q)-kd*dq, (7,))
            view = self.robot._articulation_view
            gravity = read('gravity_efforts_Nm',
                lambda: view.get_generalized_gravity_forces(joint_indices=self.left), (1, 7))[0]
            efforts = np.r_[gravity, -1.5]
            receipt['requested_efforts'] = efforts.tolist()
            caps = read('effort_limits_Nm',
                lambda: view.get_max_efforts(joint_indices=self.left), (1, 7))[0]
            if np.any(caps <= 0.):
                raise ValueError('Gravity feedforward requires positive existing effort limits')
            receipt['preparation_stage'] = 'gravity_effort_limit'
            if np.any(np.abs(gravity) > caps):
                raise GravityFeedforwardRejected('gravity exceeds existing effort limits', receipt)
        except Exception as error:
            receipt['preparation_error'] = repr(error)
            if isinstance(error, GravityFeedforwardRejected):
                raise
            raise GravityFeedforwardRejected('Gravity feedforward preparation rejected: '+str(error), receipt) from error
        receipt['preparation_stage'] = 'ready'
        receipt['gravity_effort_bounds_passed'] = True
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
