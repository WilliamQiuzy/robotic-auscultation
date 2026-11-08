from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_ausc_given_contact_pose import reference


def module():
    from sim.ausc_policy import m2_given_contact_controller
    return m2_given_contact_controller


def pose_at(ref, tangent=.001, roll=.02):
    pose = ref.pose_base.copy()
    pose[:3, 3] += ref.normal_base*.03 + pose[:3, 0]*tangent
    pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec([0, 0, roll]).as_matrix()
    return pose


def gates(ref, pose):
    from sim.auscultation_expert_layout_v1.phase_manager import PhaseConfig
    from dataclasses import replace
    return dict(phase_config=replace(PhaseConfig(), precontact_gap_m=.03),
                plane=dict(contact_plane_normal_world=[0., 0., -1.], plane_tilt_limit_deg=3.,
                           contact_plane_radius_m=.023, tilt_relaxed=False),
                world_from_base=np.eye(4), measured_pose_base=pose.copy(), measured_speed_m_s=0.,
                patient_contacts=dict(diaphragm=0., retainer=0.), original_violations=[],
                reference_sha256=ref.sha256)


@pytest.mark.parametrize('fault', ['coarse', 'retained_coarse', 'roll', 'plane', 'force', 'revision', 'violation'])
def test_admission_never_repairs_failed_l2_or_relaxes_original_gates(tmp_path, fault):
    ref = reference(tmp_path); pose = pose_at(ref); g = gates(ref, pose)
    command = pose.copy()
    if fault == 'coarse': g['measured_pose_base'][0, 3] += .01
    if fault == 'retained_coarse': command[0, 3] += .01
    if fault == 'roll': command[:3, :3] = command[:3, :3] @ Rotation.from_euler('z', .1).as_matrix()
    if fault == 'plane': g['plane']['contact_plane_normal_world'] = [0, 1, 0]
    if fault == 'force': g['patient_contacts']['retainer'] = .00001
    if fault == 'revision': g['reference_sha256'] = '0'*64
    if fault == 'violation': g['original_violations'] = ['stroke']
    with pytest.raises(ValueError): module().alignment_admission(command, ref, g)


def test_admission_and_audit_distinguish_command_observation_bounds(tmp_path):
    ref = reference(tmp_path); command = pose_at(ref)
    g = gates(ref, command)
    assert module().alignment_admission(command, ref, g)['passed']
    audit = module().AlignmentAudit(ref, command, command, g['plane'], np.eye(4))
    next_pose = command.copy(); next_pose[0, 3] += .001
    audit.observe(next_pose, 'command')
    audit.observe(command, 'observed')
    assert audit.report()['command']['path_length_m'] == pytest.approx(.001)
    assert audit.report()['observed']['path_length_m'] == 0
    next_pose[0, 3] += .00201
    with pytest.raises(ValueError, match='translation'): audit.observe(next_pose, 'command')


def test_observed_plane_is_checked_with_contact_inactive(tmp_path):
    ref = reference(tmp_path); p = pose_at(ref); g = gates(ref, p)
    audit = module().AlignmentAudit(ref, p, p, g['plane'], np.eye(4))
    bad = p.copy(); bad[:3, :3] = Rotation.from_euler('x', .06).as_matrix() @ bad[:3, :3]
    with pytest.raises(ValueError, match='plane'): audit.observe(bad, 'observed')


def bare_boundary(tmp_path):
    m = module(); c = m.GivenContactExpert.__new__(m.GivenContactExpert)
    ref = reference(tmp_path); c.reference = ref
    c.control_owner = 'TEACHER_L2'; c.prefix = object(); c._history = object()
    c.q = np.zeros(7); c.actor_phase = 'APPROACH'; c.l2_handoffs = []
    c._l2_started_at = 0.; c._reference_observation = lambda: dict(context=('e', 'L1', 0, 'APPROACH'))
    c.env = SimpleNamespace(rows=[dict(q_command=[0.]*7)]*240, state=lambda: dict(time_s=1.))
    c.trajectory = SimpleNamespace(velocity=np.zeros(7), acceleration=np.zeros(7),
        input=SimpleNamespace(target_position=[0.]*7))
    c.trajectory.set_target = lambda q: setattr(c.trajectory.input,'target_position',q.tolist())
    c._advance_tick = lambda label: None
    c._handoff_measurements = lambda: dict(passed=True, physics_row=3, time_s=0., command_q=[0.]*7)
    return c


def test_parking_uses_existing_trajectory_and_rechecks_entry(tmp_path):
    c = bare_boundary(tmp_path); trajectory = c.trajectory
    assert c._stop_for_handoff(('e', 'L1', 0, 'APPROACH'))['passed']
    assert c.trajectory is trajectory and c.prefix is None and c._history is None
    assert c.control_owner == 'L3_ENTRY'
    c = bare_boundary(tmp_path); c._handoff_measurements = lambda: dict(passed=False)
    with pytest.raises(ValueError, match='lost'): c._stop_for_handoff(('e', 'L1', 0, 'APPROACH'))


def test_transient_entry_loss_during_parking_fails_before_stationary(tmp_path):
    c = bare_boundary(tmp_path)
    c.trajectory.velocity[:] = .01
    calls = []
    def advance(label):
        calls.append(label)
        c._handoff_measurements = lambda: dict(passed=False)
        if len(calls) == 2:
            c.trajectory.velocity[:] = 0
            c._handoff_measurements = lambda: dict(passed=True)
    c._advance_tick = advance
    with pytest.raises(ValueError, match='lost'): c._stop_for_handoff(('e', 'L1', 0, 'APPROACH'))
    assert len(calls) == 1


def test_actual_alignment_report_survives_unchanged_contact_cycle(tmp_path, monkeypatch):
    m = module(); c = bare_boundary(tmp_path); ref = c.reference
    initial = pose_at(ref, tangent=0., roll=0.)
    c.kin = CartesianKin(initial); c._contact_active = False
    c.control_owner = 'L3_ENTRY'; c.manager = SimpleNamespace(state=SimpleNamespace(phase=m.Phase.APPROACH))
    c.metadata = __import__('test_ausc_given_contact_pose').metadata(ref)
    c.l2_handoffs = [dict(l2_handoff_success=True)]
    c.env.base = np.eye(4); c.env.state = lambda: dict(time_s=0., q=np.zeros(7))
    c._alignment_gates = lambda: gates(ref, initial)
    c.static_world = SimpleNamespace(check=lambda: None)
    c.executor_config = dict(curve_position_tolerance_m=2e-5, curve_angle_tolerance_rad=2e-4)
    c._block = lambda command, *args, **kwargs: None
    monkeypatch.setattr(m, 'propose_l3_alignment', lambda *args: m.AlignmentProposal(np.zeros((3, 7)), {}))
    calls = []
    monkeypatch.setattr(m.FreshExpert, '_contact_cycle', lambda self: calls.append(self.q.copy()))
    c._contact_cycle()
    assert len(calls) == 1 and c.l2_handoffs[-1]['numerical_curve_entry']['passed']
    assert 'alignment_actual_audit' in c.l2_handoffs[-1]
    assert c.l2_handoffs[-1]['alignment_actual_audit']['observed']['samples'] >= 1


def test_first_measured_coarse_tick_interrupts_teacher_before_next_route_command(tmp_path):
    c = bare_boundary(tmp_path); steps = []
    c._deadline = lambda: None
    c._advance_tick = lambda label: steps.append(label)
    c._reference_observation = lambda: dict(context=('e','L1',0,'APPROACH'))
    c._history = module().stage.CausalHistory(1.)
    c._history.frames = [(1.,np.zeros(17)),(1.05,np.zeros(17))]
    assert c._teacher_waypoint(np.ones(7)) is True
    assert steps == ['given_l2_motion']


def test_l2_failure_cannot_start_alignment_and_no_recovery_fallback(tmp_path):
    c = bare_boundary(tmp_path)
    c.l2_handoffs = [dict(l2_handoff_success=False)]
    with pytest.raises(ValueError, match='parked'): c._contact_cycle()
    assert c._recover()['attempted'] is False


class CartesianKin:
    lower_limits = np.full(7, -10.)
    upper_limits = np.full(7, 10.)
    def __init__(self, origin): self.origin = origin
    def tcp(self, q):
        out = self.origin.copy(); out[:3, 3] += q[:3]
        out[:3, :3] = self.origin[:3, :3] @ Rotation.from_rotvec(q[3:6].copy()).as_matrix()
        return out


def test_full_240hz_proposal_preserves_normal_coordinate_and_limits(tmp_path):
    ref = reference(tmp_path); initial = pose_at(ref); kin = CartesianKin(initial)
    g = gates(ref, initial)
    g.update(kin=kin, base=np.eye(4), posture_limits=SimpleNamespace(evaluate=lambda *a, **k: dict(clear=True)),
             checker=SimpleNamespace(check=lambda *a, **k: dict(clear=True, minimum_distance_m=.01)),
             command_history=np.zeros((3, 7)), compression_m=0.)
    def solve(kin, base, limits, p, r, seed, **kwargs):
        q = np.zeros(7); q[:3] = p-initial[:3, 3]
        q[3:6] = Rotation.from_matrix(initial[:3, :3].T @ r).as_rotvec()
        return q, True, 0., 0.
    result = module().propose_l3_alignment(np.zeros(7), np.zeros(7), ref, g, solver=solve)
    assert 0 < len(result.commands) <= 2400
    assert result.report['all_240hz_commands_verified']
    assert result.report['duration_s'] <= 10
    for q in result.commands:
        assert (kin.tcp(q)[:3, 3]-ref.pose_base[:3, 3]) @ ref.normal_base == pytest.approx(.03, abs=2e-5)
    np.testing.assert_array_equal(result.commands[-1], result.commands[-3])
    for fault in ('collision', 'posture', 'history', 'timeout'):
        bad = dict(g)
        if fault == 'collision': bad['checker'] = SimpleNamespace(check=lambda *a, **k: dict(clear=False))
        if fault == 'posture': bad['posture_limits'] = SimpleNamespace(evaluate=lambda *a, **k: dict(clear=False))
        if fault == 'history': bad['command_history'] = np.zeros((2, 7))
        if fault == 'timeout': bad['clock'] = lambda: 100.
        with pytest.raises(ValueError):
            module().propose_l3_alignment(np.zeros(7), np.zeros(7), ref, bad, solver=solve,
                                          started_at_s=0. if fault == 'timeout' else None)


def soft_joint_box_alignment(tmp_path, monkeypatch, expected_margin=None):
    from sim.auscultation_expert_layout_v1.posture import PostureLimits

    ref = reference(tmp_path); initial = pose_at(ref); kin = CartesianKin(initial)
    # Invented redundant-joint fixture: Cartesian FK ignores q[6].
    kin._chain = lambda q: (dict(openarm_left_link4=np.eye(4)), None, None)
    lower = np.full(7, -1.); lower[6] = -.001
    limits = PostureLimits(1., tuple(lower), tuple(np.ones(7)))
    original_evaluate = limits.evaluate
    posture_commands = []; checked_commands = []; checks = []; solves = []; knots = [np.zeros(7)]

    def evaluate(self, kin, base, q, **kwargs):
        posture_commands.append(np.asarray(q).copy())
        return original_evaluate(kin, base, q, **kwargs)

    def check(q, **kwargs):
        checked_commands.append(q.copy()); checks.append(dict(kwargs))
        return dict(clear=True, minimum_distance_m=.01)

    def solve(kin, base, posture, p, r, seed, **kwargs):
        solves.append(dict(position=p.copy(), rotation=r.copy(), seed=seed.copy(), kwargs=dict(kwargs)))
        if expected_margin is not None:
            assert kwargs['joint_margin_rad'] == expected_margin
        q = np.zeros(7); q[:3] = p-initial[:3, 3]
        q[3:6] = Rotation.from_matrix(initial[:3, :3].T @ r).as_rotvec()
        margin = kwargs['joint_margin_rad']
        q[6] = max(-min(len(solves), 10)*.0001, posture.joint_lower_rad[6]+margin)
        assert np.all(q >= np.asarray(posture.joint_lower_rad)+margin)
        assert np.all(q <= np.asarray(posture.joint_upper_rad)-margin)
        assert original_evaluate(kin, base, q, contact=False)['clear']
        knots.append(q.copy())
        return q, True, 0., 0.

    monkeypatch.setattr(PostureLimits, 'evaluate', evaluate)
    g = gates(ref, initial)
    g.update(kin=kin, base=np.eye(4), posture_limits=limits, clock=lambda: 0.,
             checker=SimpleNamespace(check=check), command_history=np.zeros((3, 7)), compression_m=0.)
    return SimpleNamespace(ref=ref, initial=initial, kin=kin, gates=g, solve=solve, solves=solves,
        knots=knots, evaluate=original_evaluate, posture_commands=posture_commands,
        checked_commands=checked_commands, checks=checks)


def test_zero_margin_valid_knots_can_undershoot_original_soft_joint_box(tmp_path, monkeypatch):
    case = soft_joint_box_alignment(tmp_path, monkeypatch)

    def zero_margin_solve(*args, **kwargs):
        kwargs['joint_margin_rad'] = 0.
        return case.solve(*args, **kwargs)

    with pytest.raises(ValueError, match='ALIGN posture rejected'):
        module().propose_l3_alignment(np.zeros(7), np.zeros(7), case.ref, case.gates, solver=zero_margin_solve)
    assert len(case.solves) == 12 and len(case.knots) == 13
    assert all(case.evaluate(case.kin, case.gates['base'], q, contact=False)['clear'] for q in case.knots)
    lower = case.gates['posture_limits'].joint_lower_rad[6]
    np.testing.assert_array_equal(np.asarray(case.knots)[-3:, 6], np.full(3, lower))
    assert case.posture_commands[-1][6] < lower-1e-9
    assert case.evaluate(case.kin, case.gates['base'], case.posture_commands[-1], contact=False)[
        'posture_violations'] == ['posture_joint_envelope']


def test_default_alignment_reserves_soft_joint_box_without_changing_cartesian_path(tmp_path, monkeypatch):
    case = soft_joint_box_alignment(tmp_path, monkeypatch, expected_margin=1e-4)
    m = module()
    result = m.propose_l3_alignment(np.zeros(7), np.zeros(7), case.ref, case.gates, solver=case.solve)
    assert len(case.solves) == 12 and len(case.knots) == 13
    gap = (case.initial[:3, 3]-case.ref.pose_base[:3, 3]) @ case.ref.normal_base
    goal = case.ref.pose_base[:3, 3]+case.ref.normal_base*gap
    rotvec = Rotation.from_matrix(case.ref.pose_base[:3, :3] @ case.initial[:3, :3].T).as_rotvec()
    for index, solve in enumerate(case.solves):
        u = (index+1)/12
        assert solve['kwargs'] == dict(deadline=30., planning_margin_m=.002,
            joint_margin_rad=1e-4, contact=False, maxiter=80)
        np.testing.assert_array_equal(solve['seed'], case.knots[index])
        np.testing.assert_allclose(solve['position'], case.initial[:3, 3]+u*(goal-case.initial[:3, 3]), rtol=0, atol=1e-12)
        expected_r = Rotation.from_rotvec(u*rotvec).as_matrix() @ case.initial[:3, :3]
        np.testing.assert_allclose(solve['rotation'], expected_r, rtol=0, atol=1e-12)
    margins = [case.evaluate(case.kin, case.gates['base'], q, contact=False)['posture_joint_margin_rad']
               for q in result.commands]
    assert min(margins) > 0.
    np.testing.assert_array_equal(case.posture_commands, result.commands)
    np.testing.assert_array_equal(case.checked_commands, result.commands)
    assert all(check == dict(clearance=.003, allow_patient_contact=False, compression=0.) for check in case.checks)
    count = len(result.commands)-3
    t = np.arange(1, count+1)/count
    progress = np.r_[10*t**3-15*t**4+6*t**5, np.ones(3)]
    for q, u in zip(result.commands, progress):
        pose = case.kin.tcp(q)
        np.testing.assert_allclose(pose[:3, 3], case.initial[:3, 3]+u*(goal-case.initial[:3, 3]), rtol=0, atol=1e-12)
        expected_r = Rotation.from_rotvec(u*rotvec).as_matrix() @ case.initial[:3, :3]
        assert Rotation.from_matrix(pose[:3, :3] @ expected_r.T).magnitude() < 1e-12
        assert (pose[:3, 3]-case.ref.pose_base[:3, 3]) @ case.ref.normal_base == pytest.approx(gap, abs=1e-12)
    assert result.report['all_240hz_commands_verified']
    assert result.report['final_stationary_stop_included']
    assert result.report['duration_s'] <= m.BOUNDS['simulation_s']
    np.testing.assert_array_equal(result.commands[-3:], np.tile(result.commands[-1], (3, 1)))
