import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def executor_class():
    assert importlib.util.find_spec('sim.auscultation_skill_v1.executor') is not None, 'shared executor is missing'
    from sim.auscultation_skill_v1.executor import NormalExecutor
    return NormalExecutor


class LinearCurve:
    minimum_gap = -.01
    maximum_gap = .025

    def __call__(self, gap):
        return np.array([gap * 5., gap * -3., gap, 0., gap * 6., 0., 0.])


def test_zero_action_cannot_advance_contact():
    cls = executor_class()
    actor = cls(LinearCurve())
    initial = actor.command.copy()
    for _ in range(1200):
        actor.step(0.)
    np.testing.assert_array_equal(actor.command, initial)
    assert actor.gap == actor.curve.maximum_gap
    assert actor.velocity == actor.acceleration == 0.


def test_activation_reversal_and_stop_have_bounded_raw_joint_derivatives():
    actor = executor_class()(LinearCurve())
    commands = [actor.command.copy()] * 3
    for action in [-.002] * 500 + [0.] * 240 + [.002] * 400 + [0.] * 480:
        commands.append(actor.step(action).copy())
    commands = np.array(commands)
    for order, limit in ((1, .25), (2, .5), (3, 8.)):
        assert np.max(np.abs(np.diff(commands, n=order, axis=0))) / actor.dt ** order <= limit + 1e-8
    assert actor.velocity < 1e-6


def test_planned_endpoint_stop_is_continuous_and_exact():
    actor = executor_class()(LinearCurve())
    for _ in range(400):
        actor.step(-.002)
    for _ in range(200):
        actor.step(.002)
    actor.plan_stop(actor.curve.maximum_gap)
    commands = [*actor.history.copy()]
    while actor.stop_pending:
        commands.append(actor.step(.002).copy())
    assert actor.gap == actor.curve.maximum_gap
    assert actor.velocity == actor.acceleration == 0.
    np.testing.assert_array_equal(actor.command, actor.curve(actor.curve.maximum_gap))
    for order, limit in ((1, .25), (2, .5), (3, 8.)):
        assert abs(np.diff(np.array(commands), n=order, axis=0)).max() / actor.dt ** order <= limit + 1e-8


@pytest.mark.parametrize('changed_action', [0., -.002])
def test_endpoint_stop_cancels_when_actor_requests_zero_or_inward(changed_action):
    actor = executor_class()(LinearCurve())
    for _ in range(600):
        actor.step(-.002)
    for _ in range(240):
        actor.step(.002)
    actor.plan_stop(actor.curve.maximum_gap)
    for _ in range(20):
        actor.step(.002)
    history = actor.history.copy()
    actor.step(changed_action)
    assert not actor.stop_pending
    assert actor.mode == 'velocity_servo'
    assert actor.velocity != 0. and actor.gap != actor.curve.maximum_gap
    np.testing.assert_array_equal(actor.history[:2], history[1:])
    assert actor.peaks['jerk'] <= 8.


def test_stationary_endpoint_updates_history_and_reports_actual_derivatives():
    actor = executor_class()(LinearCurve())
    for _ in range(400):
        actor.step(-.002)
    for _ in range(200):
        actor.step(.002)
    actor.plan_stop(actor.curve.maximum_gap)
    while actor.stop_pending:
        actor.step(.002)
    previous = actor.history.copy()
    assert hasattr(actor, 'hold_stationary'), 'stationary endpoint path must update executor state'
    command = actor.hold_stationary()
    assert actor.mode == 'endpoint_stationary'
    np.testing.assert_array_equal(actor.history[:2], previous[1:])
    for order, kind in ((1, 'velocity'), (2, 'acceleration'), (3, 'jerk')):
        expected = abs(np.diff(np.vstack([previous, command]), n=order, axis=0)[-1]).max()/actor.dt**order
        assert actor.peaks[kind] == pytest.approx(expected)


def test_rejected_command_does_not_advance_scalar_or_command_history():
    class SharpCurve(LinearCurve):
        def __call__(self, gap):
            q = super().__call__(gap)
            q[0] += 1. if gap < .025 else 0.
            return q
    actor = executor_class()(SharpCurve())
    before = (actor.gap, actor.velocity, actor.acceleration, actor.history.copy())
    with pytest.raises(RuntimeError, match='joint'):
        actor.step(-.002)
    assert (actor.gap, actor.velocity, actor.acceleration) == before[:3]
    np.testing.assert_array_equal(actor.history, before[3])


def test_nonfinite_or_out_of_domain_commands_fail_without_clipping():
    actor = executor_class()(LinearCurve())
    with pytest.raises(ValueError):
        actor.step(float('nan'))
    with pytest.raises(ValueError):
        actor.step(.005)
    with pytest.raises(RuntimeError, match='domain'):
        actor.step(.002)


def test_asymmetric_velocity_bounds_apply_to_actor_and_actual_state():
    actor = executor_class()(LinearCurve(), speed_scale=.75)
    assert actor.speed_limit == .004
    assert actor.inward_speed_limit == .0015
    for _ in range(600):
        actor.step(-.0015)
    for _ in range(200):
        actor.step(.004)
    assert -.0015 <= actor.velocity <= .004
    for invalid in (-.0015001, .0040001):
        with pytest.raises(ValueError, match='bound'):
            actor.step(invalid)
    with pytest.raises(RuntimeError, match='dynamic'):
        actor._check(actor.gap, -.0015001, 0.)
    with pytest.raises(RuntimeError, match='dynamic'):
        actor._check(actor.gap, .0040001, 0.)


def test_curve_rejects_extrapolation_and_preserves_exact_retained_endpoint():
    assert importlib.util.find_spec('sim.auscultation_skill_v1.curve') is not None, 'normal curve is missing'
    from sim.auscultation_skill_v1.curve import FixedNormalCurve
    gaps = np.array([-.005, .01, .025])
    knots = np.array([LinearCurve()(g) for g in gaps])
    curve = FixedNormalCurve(gaps, knots)
    np.testing.assert_array_equal(curve(.025), knots[-1])
    with pytest.raises(ValueError, match='domain'):
        curve(.02501)


def test_force_derived_domain_has_fixed_upper_nominal_force_band():
    from sim.auscultation_skill_v1.curve import force_derived_domain
    normal = np.array([0., np.sqrt(1.-.43582029513643267**2), .43582029513643267])
    domain = force_derived_domain(.0027673129462866095, .002, normal, 4., .0005)
    assert domain['nominal_contact_center_gap_m'] == pytest.approx(.0047673129462866095)
    assert domain['static_target_compression_m'] == pytest.approx(.002212885708065907)
    assert domain['maximum_nominal_compression_m'] == pytest.approx(.002712885708065907)
    assert domain['minimum_gap_m'] == pytest.approx(.002054427238220702)
    assert domain['nominal_upper_force_N'] == pytest.approx(4.5)
    assert domain['perturbation_envelope_certified'] is False


@pytest.mark.parametrize('normal,reserve', [([0., 0., 2.], .0005), ([0., 0., np.nan], .0005),
                                            ([0., 0., 1.], -.001)])
def test_force_domain_rejects_invalid_inputs(normal, reserve):
    from sim.auscultation_skill_v1.curve import force_derived_domain
    with pytest.raises(ValueError):
        force_derived_domain(.003, .002, normal, 4., reserve)


def test_curve_collision_failure_preserves_requested_domain_and_exact_pair(monkeypatch):
    from sim.auscultation_skill_v1.curve import FixedNormalCurve, CurveCollisionRejected
    from sim.auscultation_expert_layout_v1 import pose_ik
    class Kin:
        def tcp(self, q):
            result = np.eye(4); result[0, 3] = q[0]
            return result
    class Limits:
        def evaluate(self, *args, **kwargs):
            return {'clear': True}
    def solve(kin, base, limits, position, rotation, seed, **kwargs):
        q = np.zeros(7); q[0] = position[0]
        return q, True, 0., 0.
    monkeypatch.setattr(pose_ik, 'solve_posture_ik', solve)
    checked = []
    def collision(q, gap):
        checked.append(gap)
        return {'clear': False, 'pair': ['fixed_post', 'shirt'], 'minimum_distance_m': .0002}
    with pytest.raises(CurveCollisionRejected) as caught:
        FixedNormalCurve.build(Kin(), np.eye(4), Limits(), [.025, 0., 0., 0., 0., 0., 0.],
            [0., 0., 0.], [1., 0., 0.], np.eye(3), .002, collision_check=collision)
    assert checked == [.002]
    assert caught.value.evidence['gap_m'] == .002
    assert caught.value.evidence['collision']['pair'] == ['fixed_post', 'shirt']
    assert 'fixed_post' in str(caught.value)


def test_selected_l4_fixed_frame_curve_and_unload_commands():
    from sim.auscultation_skill_v1.curve import FixedNormalCurve, force_derived_domain
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    path = Path(__file__).parents[2] / 'upper_pair_posture_20260924/results/candidate_b_plans/planned_paths_L4.json'
    plan = json.loads(path.read_text()); row = plan['points'][0]
    kin = OpenArmKinematics(); base = np.array(plan['world_from_left_base'])
    normal = base[:3, :3].T @ row['command_normal_world']
    rotation = base[:3, :3].T @ row['command_rotation_world']
    target = (np.linalg.inv(base) @ np.r_[row['position_world'], 1.])[:3]
    loaded_gap = (kin.tcp(row['loaded_q'])[:3, 3]-target) @ normal
    domain = force_derived_domain(loaded_gap, row['loaded_compression_m'], row['command_normal_world'], 4., .0005)
    curve = FixedNormalCurve.build(kin, base, PostureLimits.from_dict(row['posture_limits'], kin),
        row['transfer_q'][-1], target, normal, rotation, domain['minimum_gap_m'])
    np.testing.assert_array_equal(curve(curve.maximum_gap), row['transfer_q'][-1])
    assert curve.report['maximum_position_error_m'] <= 2e-5
    actor = executor_class()(curve)
    for _ in range(600):
        actor.step(-.002)
    for _ in range(240):
        actor.step(0.)
    for _ in range(350):
        actor.step(.002)
    actor.plan_stop(curve.maximum_gap)
    while actor.stop_pending:
        actor.step(.002)
    np.testing.assert_array_equal(actor.command, row['transfer_q'][-1])
