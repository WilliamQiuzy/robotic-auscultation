"""Stateful normal-motion integration; no simulator, force policy or IK imports."""
from dataclasses import asdict, is_dataclass
import numpy as np


DT = 1. / 240.
DEFAULTS = dict(normal_speed_limit_m_s=.004, normal_inward_speed_limit_m_s=.002,
                normal_acceleration_limit_m_s2=.02,
                normal_jerk_limit_m_s3=.2, joint_velocity_limit_rad_s=.25,
                joint_acceleration_limit_rad_s2=.5, joint_jerk_limit_rad_s3=8.)


def config_values(config=None):
    return {**DEFAULTS, **(asdict(config) if is_dataclass(config) else dict(config or {}))}


def command_derivatives(history, command, dt=DT):
    history, command = np.asarray(history, float), np.asarray(command, float)
    if history.shape != (3, 7) or command.shape != (7,) or not np.isfinite(history).all() or not np.isfinite(command).all():
        raise ValueError('three finite preceding commands are required')
    q = np.vstack([history, command])
    return {name: float(np.max(np.abs(np.diff(q, n=order, axis=0)[-1])) / dt**order)
            for name, order in (('velocity', 1), ('acceleration', 2), ('jerk', 3))}


class NormalExecutor:
    """Continuous scalar g/v/a with explicit rejection, never position clipping.

    The desired velocity is held by the policy at 20 Hz. A damped velocity servo
    produces bounded jerk at 240 Hz; its exact constant-jerk integration retains
    g/v/a across policy changes. Joint finite differences are the final gate.
    A separately verified quintic endpoint trajectory permits an exact rest.
    """
    dt = DT

    def __init__(self, curve, config=None, history=None, speed_scale=1.):
        self.curve, self.config = curve, config_values(config)
        if not np.isfinite(speed_scale) or not .7 <= speed_scale <= 1.:
            raise ValueError('invalid speed scale')
        if any(not np.isfinite(v) or v <= 0 for k, v in self.config.items() if k in DEFAULTS):
            raise ValueError('invalid executor limit')
        # Slow approach conditions must not prolong loaded withdrawal. The
        # shared action space fixes its outward ceiling while scaling inward.
        self.speed_limit = self.config['normal_speed_limit_m_s']
        self.inward_speed_limit = self.config['normal_inward_speed_limit_m_s'] * speed_scale
        self.gap = float(curve.maximum_gap)
        self.velocity = self.acceleration = 0.
        self.command = np.asarray(curve(self.gap), float).copy()
        self.history = (np.repeat(self.command[None], 3, axis=0) if history is None else np.asarray(history, float).copy())
        if self.history.shape != (3, 7) or not np.array_equal(self.history[-1], self.command):
            raise ValueError('curve does not start at retained command')
        if np.max(np.abs(np.diff(self.history, axis=0))) > 1e-10:
            raise ValueError('contact curve entry requires stationary retained commands')
        self.peaks = dict(velocity=0., acceleration=0., jerk=0.)
        self._stop = []
        self.mode = 'velocity_servo'

    @property
    def stop_pending(self):
        return bool(self._stop)

    def _check(self, g, v, a, history=None):
        if not np.isfinite([g, v, a]).all():
            raise RuntimeError('nonfinite normal state')
        if not self.curve.minimum_gap <= g <= self.curve.maximum_gap:
            raise RuntimeError('normal curve domain exhausted')
        if (not -self.inward_speed_limit - 1e-10 <= v <= self.speed_limit + 1e-10
                or abs(a) > self.config['normal_acceleration_limit_m_s2'] + 1e-10):
            raise RuntimeError('normal dynamic limit')
        command = np.asarray(self.curve(g), float)
        peaks = command_derivatives(self.history if history is None else history, command)
        for kind, field in (('velocity', 'joint_velocity_limit_rad_s'), ('acceleration', 'joint_acceleration_limit_rad_s2'), ('jerk', 'joint_jerk_limit_rad_s3')):
            if peaks[kind] > self.config[field] + 1e-8:
                raise RuntimeError('joint ' + kind + ' finite-difference limit')
        return command, peaks

    def step(self, desired_velocity):
        if isinstance(desired_velocity, bool) or not np.isscalar(desired_velocity) or not np.isfinite(desired_velocity):
            raise ValueError('normal velocity must be finite')
        if not -self.inward_speed_limit - 1e-12 <= desired_velocity <= self.speed_limit + 1e-12:
            raise ValueError('normal action outside declared bound')
        use_stop = bool(self._stop) and desired_velocity > 0.
        if use_stop:
            g, v, a = self._stop[0]
        else:
            # Continuous acceleration; saturation is applied to jerk, not q/g/v.
            jerk = np.clip(64. * (desired_velocity - self.velocity) - 16. * self.acceleration,
                           -self.config['normal_jerk_limit_m_s3'], self.config['normal_jerk_limit_m_s3'])
            next_a = np.clip(self.acceleration + jerk * self.dt,
                             -self.config['normal_acceleration_limit_m_s2'], self.config['normal_acceleration_limit_m_s2'])
            jerk = (next_a - self.acceleration) / self.dt
            g = self.gap + self.velocity*self.dt + .5*self.acceleration*self.dt**2 + jerk*self.dt**3/6.
            v = self.velocity + self.acceleration*self.dt + .5*jerk*self.dt**2
            a = next_a
        command, peaks = self._check(g, v, a)
        self.gap, self.velocity, self.acceleration = float(g), float(v), float(a)
        self.command = command.copy()
        self.history = np.vstack([self.history[1:], command])
        self.peaks = peaks
        self.mode = 'endpoint_stop' if use_stop else 'velocity_servo'
        if use_stop:
            self._stop.pop(0)
        else:
            # Cancel only after the replacement step passes every bound. The
            # velocity servo starts from the unchanged current g/v/a/history.
            self._stop.clear()
        return command.copy()

    def hold_stationary(self):
        """Record a genuine stationary command through the same derivative gate."""
        if self.stop_pending or self.velocity != 0. or self.acceleration != 0.:
            raise RuntimeError('stationary hold requires completed endpoint stop')
        if self.gap != self.curve.maximum_gap:
            raise RuntimeError('stationary hold requires retained endpoint')
        command, peaks = self._check(self.gap, 0., 0.)
        self.command = command.copy()
        self.history = np.vstack([self.history[1:], command])
        self.peaks = peaks
        self.mode = 'endpoint_stationary'
        return command.copy()

    def plan_stop(self, destination):
        """Preverify an exact rest at an outward endpoint including both joins."""
        if not np.isfinite(destination) or not self.gap < destination <= self.curve.maximum_gap:
            raise ValueError('stop destination must be outward inside curve')
        distance = destination - self.gap
        for duration in np.linspace(.5, 6., 56):
            steps = int(np.ceil(duration / self.dt)); duration = steps * self.dt
            c = np.zeros(6); c[:3] = self.gap, self.velocity, self.acceleration / 2.
            matrix = np.array([[duration**3, duration**4, duration**5], [3*duration**2, 4*duration**3, 5*duration**4], [6*duration, 12*duration**2, 20*duration**3]])
            c[3:] = np.linalg.solve(matrix, [distance-self.velocity*duration-.5*self.acceleration*duration**2,
                                           -self.velocity-self.acceleration*duration, -self.acceleration])
            t = np.arange(1, steps+1) * self.dt
            g = np.polynomial.polynomial.polyval(t, c)
            v = np.polynomial.polynomial.polyval(t, np.arange(1, 6)*c[1:])
            a = np.polynomial.polynomial.polyval(t, np.arange(1, 5)*np.arange(2, 6)*c[2:])
            jerk = np.polynomial.polynomial.polyval(t, np.array([6., 24., 60.])*c[3:])
            if (abs(jerk).max() > self.config['normal_jerk_limit_m_s3'] or np.min(np.diff(np.r_[self.gap, g])) < -1e-12):
                continue
            g[-1], v[-1], a[-1] = destination, 0., 0.  # exact polynomial boundary values
            history = self.history.copy(); states = list(zip(g, v, a))
            try:
                for state in states + [(destination, 0., 0.)] * 3:
                    q, _ = self._check(*state, history=history)
                    history = np.vstack([history[1:], q])
            except (RuntimeError, ValueError):
                continue
            self._stop = states
            return
        raise RuntimeError('bounded normal endpoint stop unavailable')
