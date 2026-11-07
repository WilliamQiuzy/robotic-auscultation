"""Default-off joint velocity TARGETS from adjacent executed position commands.

No actual-state velocity setter, trajectory interpolation, clipping, gain/limit
changes, physics stepping or rendering. Stationary holds have zero targets;
moving force corrections retain their derivative even in supervisor HOLD.
"""
from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from .gravity_feedforward import _diagnostic_value, _integer


JOINT_VELOCITY_LIMIT_RAD_S = .175
SOURCE = 'adjacent executed joint position commands divided by unchanged physics dt'


class VelocityFeedforwardRejected(ValueError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def velocity_feedforward_settings(fresh_config):
    if 'velocity_feedforward' not in fresh_config:
        return None
    value = fresh_config['velocity_feedforward']
    if (not isinstance(value, dict) or set(value) != {'enabled'}
            or type(value['enabled']) is not bool):
        raise ValueError('fresh_expert.velocity_feedforward requires only explicit enabled: bool')
    return dict(enabled=value['enabled'])


def _numeric(value, label, shape=None):
    array = np.asarray(value)
    if (array.dtype.kind not in 'iuf' or not np.isfinite(array).all()
            or (shape is not None and array.shape != shape)):
        raise ValueError('Invalid velocity feedforward '+label)
    return array.astype(float, copy=True)


def _snapshot(env):
    return dict(physics_rows=len(env.rows),
                world_time_s=float(_numeric(env.world.current_time, 'world clock', ())),
                env_time_s=float(_numeric(env.t, 'environment clock', ())))


class NativeVelocityFeedforward:
    """Attach a target to the parent's exact position action and audit execution."""
    def __init__(self, robot, left, *, physics_dt):
        indices = np.asarray(left, dtype=object)
        if (indices.shape != (7,) or not all(_integer(v) and v >= 0 for v in indices)
                or len(set(indices)) != 7):
            raise ValueError('Velocity feedforward requires seven distinct integer indices')
        self.dt = float(_numeric(physics_dt, 'physics dt', ()))
        if self.dt <= 0.: raise ValueError('Velocity feedforward requires positive physics dt')
        self.robot = robot; self.left = indices.astype(int)
        self.previous_executed = None
        self._check_binding()

    def _check_binding(self):
        for ordinal, index in enumerate(self.left, 1):
            native = self.robot.get_dof_index(f'openarm_left_joint{ordinal}')
            if not _integer(native) or native != index:
                raise ValueError('Native velocity target name/index binding differs')

    def prepare(self, q_target):
        receipt = dict(schema='native_joint_velocity_target_v1', enabled=True,
            source=SOURCE, learned_policy=False, real_hardware_validated=False,
            training_eligible=False, velocity_is_target_not_measured_state=True,
            position_commands_unchanged=True, gains_effort_caps_gravity_unchanged=True,
            left_joint_indices=self.left.tolist(), physics_dt_s=self.dt,
            joint_velocity_limit_rad_s=JOINT_VELOCITY_LIMIT_RAD_S,
            q_target_rad=None, previous_executed_q_target_rad=None,
            command_delta_rad=None, velocity_target_rad_s=None,
            q_before_rad=None, dq_before_rad_s=None, position_error_before_rad=None,
            velocity_error_before_rad_s=None, velocity_source=None,
            velocity_targets_persisted=False, target_action_applied=False,
            physics_step_executed=False, raw_inputs={},
            raw_input_nonfinite_representation='null',
            readback_is_native_command_target_not_measured_velocity=True,
            causal_scope='command derivative and tracking errors only; damping-lag hypothesis is not physical validation')
        def read(key, value):
            receipt['raw_inputs'][key] = _diagnostic_value(value)
            array = _numeric(value, key, (7,)); receipt[key] = array.tolist()
            return array
        try:
            self._check_binding()
            target = read('q_target_rad', q_target)
            if self.previous_executed is None:
                velocity = np.zeros(7)
                receipt['velocity_source'] = 'first_command_explicit_stationary_zero'
            else:
                previous = read('previous_executed_q_target_rad', self.previous_executed)
                with np.errstate(over='ignore', invalid='ignore'):
                    delta = read('command_delta_rad', target-previous)
                    velocity = delta/self.dt
                receipt['velocity_source'] = SOURCE
            velocity = read('velocity_target_rad_s', velocity)
            if np.any(np.abs(velocity) > JOINT_VELOCITY_LIMIT_RAD_S):
                raise ValueError('Joint velocity target exceeds unchanged .175 rad/s limit')
            q = read('q_before_rad', self.robot.get_joint_positions(joint_indices=self.left))
            dq = read('dq_before_rad_s', self.robot.get_joint_velocities(joint_indices=self.left))
            with np.errstate(over='ignore', invalid='ignore'):
                read('position_error_before_rad', target-q)
                read('velocity_error_before_rad_s', velocity-dq)
            return receipt
        except Exception as error:
            receipt['preparation_error'] = repr(error)
            raise VelocityFeedforwardRejected(str(error), receipt) from error

    def verify_applied(self, receipt):
        """Read native position/velocity targets, with explicit DOF mapping."""
        receipt['velocity_targets_persisted'] = False
        try:
            self._check_binding()
            reader = getattr(self.robot, 'get_applied_action', None)
            source = 'SingleArticulation.get_applied_action'
            batched = False
            if not callable(reader):
                controller = getattr(self.robot, 'get_articulation_controller', lambda: None)()
                reader = getattr(controller, 'get_applied_action', None)
                source = 'ArticulationController.get_applied_action'
            if not callable(reader):
                reader = getattr(getattr(self.robot, '_articulation_view', None), 'get_applied_actions', None)
                source = 'Articulation.get_applied_actions'; batched = True
            if not callable(reader): raise ValueError('Native applied-action target readback unavailable')
            action = reader()
            field = lambda key: action.get(key) if isinstance(action, dict) else getattr(action, key, None)
            raw = field('joint_velocities'); raw_indices = field('joint_indices')
            receipt.update(readback_source=source,
                raw_applied_velocity_targets_rad_s=_diagnostic_value(raw),
                raw_applied_joint_indices=_diagnostic_value(raw_indices))
            values = _numeric(raw, 'native velocity target readback')
            if batched:
                if values.ndim != 2 or values.shape[0] != 1:
                    raise ValueError('Expected one native articulation target row')
                values = values[0]
            if values.ndim != 1: raise ValueError('Native velocity targets must be one-dimensional')
            if raw_indices is None:
                if len(values) <= max(self.left): raise ValueError('Incomplete full native DOF targets')
                selected = values[self.left]
            else:
                indices = np.asarray(raw_indices, dtype=object)
                if (indices.shape != values.shape or not all(_integer(v) and v >= 0 for v in indices)
                        or len(set(indices)) != len(indices) or not set(self.left).issubset(indices)):
                    raise ValueError('Native velocity target readback indices differ')
                selected = values[[list(indices).index(index) for index in self.left]]
            receipt['applied_velocity_targets_rad_s'] = selected.tolist()
            requested = _numeric(receipt['velocity_target_rad_s'], 'requested target', (7,))
            # Native targets round-trip through float32; zero must remain zero.
            tolerance = 4*np.finfo(np.float32).eps*np.abs(requested)
            error = np.abs(selected-requested)
            receipt.update(readback_absolute_error_rad_s=error.tolist(),
                           readback_tolerance_rad_s=tolerance.tolist())
            if np.any(error > tolerance): raise ValueError('Requested velocity targets did not persist')
            receipt['velocity_targets_persisted'] = True
        except Exception as error:
            receipt['post_step_readback_error'] = repr(error)
            raise VelocityFeedforwardRejected('Velocity target readback rejected: '+str(error), receipt) from error

    def _failure(self, env, receipt, before, *, label, target, meta, error, original=None):
        if not hasattr(env, 'velocity_feedforward_failure_receipts'):
            env.velocity_feedforward_failure_receipts = []
        rows_after = len(env.rows)
        env.velocity_feedforward_failure_receipts.append(dict(
            stage='post_step' if receipt['physics_step_executed'] else 'before_step',
            phase=label, target=target, subphase=(meta or {}).get('subphase'),
            physics_rows_before_attempt=before['physics_rows'], physics_rows_after_attempt=rows_after,
            executed_physics_row_index=(before['physics_rows'] if rows_after == before['physics_rows']+1 else None),
            physics_step_executed=receipt['physics_step_executed'],
            original_error=None if original is None else repr(original), error=repr(error),
            receipt=deepcopy(receipt)))

    def _finish(self, env, receipt, before):
        # Row accounting is independent of native clock availability. A broken
        # clock must not hide an actual appended row or skip native readback.
        after = dict(physics_rows=len(env.rows), world_time_s=None, env_time_s=None)
        receipt['after'] = after
        receipt['raw_after_clocks'] = {}; receipt['clock_read_errors'] = {}
        for key, read in [('world_time_s', lambda: env.world.current_time),
                          ('env_time_s', lambda: env.t)]:
            try:
                raw = read(); receipt['raw_after_clocks'][key] = _diagnostic_value(raw)
                after[key] = float(_numeric(raw, 'post-step '+key+' clock', ()))
            except Exception as error:
                receipt['clock_read_errors'][key] = repr(error)
        row_delta = after['physics_rows']-before['physics_rows']
        deltas = {key: None if after[key] is None else after[key]-before[key]
                  for key in ('world_time_s', 'env_time_s')}
        receipt['physics_step_executed'] = bool(row_delta > 0 or
            any(value is not None and value > 0 for value in deltas.values()))
        receipt['physics_step_execution_known'] = bool(receipt['physics_step_executed'] or
            (row_delta == 0 and all(value == 0 for value in deltas.values())))
        if receipt['physics_step_executed'] and receipt['target_action_applied']:
            self.previous_executed = np.asarray(receipt['q_target_rad'], float).copy()
        # Readback is attempted even when the parent advanced physics but failed
        # to append its row. The ledger records this without inventing a row.
        if receipt['target_action_applied']: self.verify_applied(receipt)
        if receipt['clock_read_errors']:
            raise VelocityFeedforwardRejected('Invalid post-step clock evidence: '+
                '; '.join(receipt['clock_read_errors'].values()), receipt)
        if (row_delta != 1 or not all(np.isclose(value, self.dt, atol=1e-8, rtol=0)
                                     for value in deltas.values())
                or not receipt['target_action_applied']):
            raise VelocityFeedforwardRejected('Expected one unchanged inherited physics step and row', receipt)

    @contextmanager
    def command(self, env, q_target, label='', target='', meta=None):
        before = _snapshot(env)
        try:
            receipt = self.prepare(q_target)
            receipt['before'] = before
            receipt['preparation_stage'] = 'native_physics_dt_s'
            try:
                raw_dt = env.world.get_physics_dt()
                receipt['raw_inputs']['native_physics_dt_s'] = _diagnostic_value(raw_dt)
                native_dt = float(_numeric(raw_dt, 'native physics dt', ()))
                if native_dt != self.dt: raise ValueError('Native physics dt changed')
            except Exception as error:
                receipt['preparation_error'] = repr(error)
                raise VelocityFeedforwardRejected('Native physics dt rejected: '+str(error), receipt) from error
            receipt['preparation_stage'] = 'ready'
        except VelocityFeedforwardRejected as error:
            self._failure(env, error.receipt, before, label=label, target=target, meta=meta, error=error)
            raise
        original_apply = self.robot.apply_action
        owned = 'apply_action' in self.robot.__dict__
        previous_attribute = self.robot.__dict__.get('apply_action')
        calls = 0
        def apply(action):
            nonlocal calls
            positions = getattr(action, 'joint_positions', None)
            if positions is None:
                # The existing passive preload/native G action is independent.
                if getattr(action, 'joint_velocities', None) is not None:
                    raise VelocityFeedforwardRejected('Unexpected inherited velocity action', receipt)
                return original_apply(action)
            calls += 1
            if (calls != 1 or not np.array_equal(np.asarray(action.joint_indices), self.left)
                    or not np.array_equal(_numeric(positions, 'inherited positions', (7,)), receipt['q_target_rad'])
                    or getattr(action, 'joint_velocities', None) is not None):
                raise VelocityFeedforwardRejected('Inherited position action differs from exact command', receipt)
            action.joint_velocities = np.asarray(receipt['velocity_target_rad_s'], float)
            original_apply(action)
            receipt['target_action_applied'] = True
        try:
            try:
                self.robot.apply_action = apply
                yield receipt
            finally:
                if owned: self.robot.apply_action = previous_attribute
                else: del self.robot.apply_action
        except BaseException as original:
            try: self._finish(env, receipt, before)
            except BaseException as secondary:
                receipt['secondary_completion_error'] = repr(secondary)
            self._failure(env, receipt, before, label=label, target=target, meta=meta,
                          error=original, original=original)
            raise
        else:
            try: self._finish(env, receipt, before)
            except BaseException as error:
                receipt['completion_error'] = repr(error)
                self._failure(env, receipt, before, label=label, target=target, meta=meta, error=error)
                raise

    def reset(self, env):
        """Clear targets at termination without advancing physics or changing q."""
        from isaacsim.core.utils.types import ArticulationAction
        receipt = dict(schema='native_joint_velocity_target_reset_v1',
            velocity_target_rad_s=[0.]*7, left_joint_indices=self.left.tolist(),
            velocity_targets_persisted=False, physics_step_executed=False,
            physics_step_execution_known=False, reset_no_physics_advance_verified=False,
            target_action_applied=False, raw_clocks={}, clock_read_errors={},
            source='explicit stationary zero at termination; no physics step', training_eligible=False)
        env.velocity_feedforward_reset_receipt = receipt
        def snapshot(stage):
            result = dict(physics_rows=len(env.rows), world_time_s=None, env_time_s=None)
            receipt[stage] = result
            receipt['raw_clocks'][stage] = {}; receipt['clock_read_errors'][stage] = {}
            for key, read in [('world_time_s', lambda: env.world.current_time),
                              ('env_time_s', lambda: env.t)]:
                try:
                    raw = read(); receipt['raw_clocks'][stage][key] = _diagnostic_value(raw)
                    result[key] = float(_numeric(raw, 'reset '+stage+' '+key+' clock', ()))
                except Exception as error:
                    receipt['clock_read_errors'][stage][key] = repr(error)
            return result
        before = snapshot('before'); failure = None
        try:
            self.robot.apply_action(ArticulationAction(joint_velocities=np.zeros(7), joint_indices=self.left))
            receipt['target_action_applied'] = True
        except Exception as error:
            receipt['reset_action_error'] = repr(error); failure = error
        # Native readback remains useful even if the action call or clocks fail.
        try: self.verify_applied(receipt)
        except Exception as error:
            if failure is None: failure = error
        after = snapshot('after')
        row_delta = after['physics_rows']-before['physics_rows']
        deltas = {key: None if before[key] is None or after[key] is None else after[key]-before[key]
                  for key in ('world_time_s', 'env_time_s')}
        receipt['physics_step_executed'] = bool(row_delta > 0 or
            any(value is not None and value > 0 for value in deltas.values()))
        unchanged = row_delta == 0 and all(value == 0 for value in deltas.values())
        receipt['physics_step_execution_known'] = bool(receipt['physics_step_executed'] or unchanged)
        receipt['reset_no_physics_advance_verified'] = unchanged
        if failure is not None:
            try:
                raise VelocityFeedforwardRejected('Velocity target reset rejected: '+str(failure), receipt) from failure
            finally:
                # A saved exception owns its traceback, including this frame
                # and env. Release the local reference before cleanup catches
                # the chained rejection so no native objects await cyclic GC.
                failure = None
        if not unchanged:
            raise VelocityFeedforwardRejected('Velocity target reset cannot verify unchanged physics clocks/rows', receipt)
