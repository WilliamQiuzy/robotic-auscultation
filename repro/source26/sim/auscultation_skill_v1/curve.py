"""A retained-branch, fixed-frame contact curve with no extrapolation."""
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


def force_derived_domain(loaded_gap, loaded_compression, normal_world, target_force_N, reserve_m,
                         *, moving_mass_kg=.067155):
    """Declare an operating domain from the frozen static Hall model.

    The reserve is fixed before geometry verification. It is not a guarantee
    about dynamic force, sensor perturbations, tracking error or actor actions.
    """
    normal = np.asarray(normal_world, float)
    values = np.asarray([loaded_gap, loaded_compression, target_force_N, reserve_m,moving_mass_kg], float)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or abs(np.linalg.norm(normal)-1.) > 1e-6 or not np.isfinite(values).all()
            or loaded_compression < 0. or target_force_N <= 0. or reserve_m <= 0. or moving_mass_kg<=0):
        raise ValueError('invalid force-derived curve domain inputs')
    preload, stiffness, mass, gravity = 1.5, 1000., moving_mass_kg, 9.81
    static_compression = (target_force_N-preload-mass*gravity*normal[2])/stiffness
    if static_compression <= 0.:
        raise ValueError('force target has no positive static compression')
    center = float(loaded_gap+loaded_compression)
    maximum = float(static_compression+reserve_m)
    return dict(method='fixed static Hall target compression plus declared reserve',
                minimum_gap_m=center-maximum, nominal_contact_center_gap_m=center,
                static_target_compression_m=float(static_compression), maximum_nominal_compression_m=maximum,
                nominal_compression_reserve_m=float(reserve_m), nominal_upper_force_N=float(target_force_N+stiffness*reserve_m),
                target_force_N=float(target_force_N), normal_world=normal.tolist(),
                static_hall_model=dict(preload_N=preload, stiffness_N_m=stiffness, moving_mass_kg=mass,
                                       gravity_m_s2=gravity, slider_velocity_m_s=0.),
                perturbation_envelope_certified=False,
                scope='sampled nominal geometry domain; actual force, stroke, collision and domain-exit gates remain active')


class CurveCollisionRejected(RuntimeError):
    def __init__(self, gap, command, collision):
        self.evidence = dict(gap_m=float(gap), q_command=np.asarray(command, float).tolist(), collision=dict(collision))
        super().__init__('normal curve nominal compression collision rejected: '+str(self.evidence))


class FixedNormalCurve:
    def __init__(self, gaps, commands):
        gaps, commands = np.asarray(gaps, float), np.asarray(commands, float)
        if (gaps.ndim != 1 or len(gaps) < 3 or commands.shape != (len(gaps), 7)
                or not np.isfinite(gaps).all() or not np.isfinite(commands).all() or np.any(np.diff(gaps) <= 0)):
            raise ValueError('invalid normal curve knots')
        self.gaps, self.commands = gaps.copy(), commands.copy()
        self.minimum_gap, self.maximum_gap = float(gaps[0]), float(gaps[-1])
        self.spline = CubicSpline(gaps, commands, axis=0, extrapolate=False)
        self.report = {}

    def __call__(self, gap):
        if not np.isscalar(gap) or not np.isfinite(gap) or not self.minimum_gap <= gap <= self.maximum_gap:
            raise ValueError('gap outside checked curve domain')
        if gap == self.maximum_gap:
            return self.commands[-1].copy()
        if gap == self.minimum_gap:
            return self.commands[0].copy()
        return np.asarray(self.spline(gap), float)

    @classmethod
    def build(cls, kin, base, limits, retained_q, target_position_base, normal_base,
              rotation_base, minimum_gap, *, spacing=.00025, position_tolerance=2e-5,
              angle_tolerance=2e-4, collision_check=None, wall_budget_s=120.):
        from sim.auscultation_expert_layout_v1.pose_ik import solve_posture_ik
        q = np.asarray(retained_q, float).copy()
        target, normal, rotation = map(lambda x: np.asarray(x, float), (target_position_base, normal_base, rotation_base))
        start = kin.tcp(q)
        maximum_gap = float((start[:3, 3] - target) @ normal)
        if not np.isfinite(minimum_gap) or minimum_gap >= maximum_gap or spacing <= 0:
            raise ValueError('invalid curve domain')
        deadline = time.monotonic() + wall_budget_s
        descending = np.linspace(maximum_gap, minimum_gap, max(3, int(np.ceil((maximum_gap-minimum_gap)/spacing))+1))
        knots = [q]
        for gap in descending[1:]:
            q, good, _, _ = solve_posture_ik(kin, base, limits, target+normal*gap, rotation, knots[-1],
                maxiter=80, deadline=deadline, planning_margin_m=.002, joint_margin_rad=0., contact=True)
            if not good:
                raise RuntimeError('fixed normal curve posture IK failed')
            knots.append(q)
        curve = cls(descending[::-1], np.asarray(knots)[::-1])
        position_max = angle_max = 0.; checked = 0
        for gap in np.linspace(minimum_gap, maximum_gap, (len(knots)-1)*4+1):
            if time.monotonic() >= deadline:
                raise RuntimeError('normal curve verification deadline')
            q = curve(gap); pose = kin.tcp(q)
            pe = float(np.linalg.norm(pose[:3, 3] - (target+normal*gap)))
            ae = float(Rotation.from_matrix(pose[:3, :3] @ rotation.T).magnitude())
            if pe > position_tolerance or ae > angle_tolerance:
                raise RuntimeError('normal curve FK interpolation error')
            if not limits.evaluate(kin, base, q, contact=True)['clear']:
                raise RuntimeError('normal curve posture rejected')
            if collision_check is not None:
                collision = collision_check(q, float(gap))
                if collision.get('clear') is not True:
                    raise CurveCollisionRejected(gap, q, collision)
            position_max, angle_max = max(position_max, pe), max(angle_max, ae)
            checked += 1
        curve.report = dict(minimum_gap_m=minimum_gap, maximum_gap_m=maximum_gap, knots=len(knots),
                            verification_samples=checked, maximum_position_error_m=position_max,
                            maximum_angle_error_rad=angle_max, retained_endpoint_exact=True,
                            collision_scope='sampled nominal compression; actual commands and observations checked separately')
        return curve
