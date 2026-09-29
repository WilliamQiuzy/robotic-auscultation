import hashlib
import json

import numpy as np
import pytest

from sim.auscultation_training.controller import CONTROL_STEPS, FORBIDDEN_GROUPS, EpisodeRejected, ExpertController
from sim.auscultation_training.low_elbow_controller import LowElbowController
from sim.auscultation_training.phase_manager import PhaseManager, Phase
from sim.auscultation_training.posture import PostureLimits
from sim.auscultation_training.run_low_elbow import verify_saved
from sim.auscultation_training.tests.test_controller import FakeEnv, FakeKin, target, row


class PostureKin(FakeKin):
    def _chain(self, q):
        self.last_fk_q = np.array(q, copy=True)
        elbow, shoulder, wrist = np.eye(4), np.eye(4), np.eye(4)
        elbow[2, 3] = .5 + q[0]
        shoulder[2, 3] = 1.
        wrist[2, 3] = .55
        return {"openarm_left_link4": elbow, "openarm_left_link1": shoulder,
                "openarm_left_link7": wrist}, None, None

    def link_transforms(self, q):
        return self._chain(q)[0]


class PostureCollisionChecker:
    def __init__(self, kin):
        self.kinematics = kin

    def check(self, q, **kwargs):
        return {"clear": True, "minimum_distance_m": .01}


def profile(height=1.):
    return {"schema": "low_elbow_posture_v1", "max_elbow_z_m": height,
            "joint_lower_rad": [-1.] * 7, "joint_upper_rad": [1.] * 7,
            "observed_joint_tolerance_rad": .005}


def controller(env=None):
    kin, env = PostureKin(), env or FakeEnv()
    value = LowElbowController(env, kin, PostureCollisionChecker(kin))
    value.posture_limits = PostureLimits.from_dict(profile(), kin)
    value.target = target("R1")
    value.normal = value.normal_base = np.array([0., 0., 1.])
    value.rotation_base = np.diag([1., -1., -1.])
    value.manager = PhaseManager(value.config)
    value.face_filtered = value.total_filtered = value.audit_hold = value.best_audit_hold = 0.
    value.recovery_baseline = {name: 0. for name in FORBIDDEN_GROUPS}, 0., 0.
    return value


def test_commands_are_strict_but_measured_joint_tolerance_is_explicit_and_not_clipped():
    kin = PostureKin()
    data = profile(100.)
    data["joint_upper_rad"] = [3.] * 7
    limits = PostureLimits.from_dict(data, kin)
    q = np.zeros(7)
    q[0] = 3.004
    assert not limits.evaluate(kin, np.eye(4), q)["clear"]
    result = limits.evaluate(kin, np.eye(4), q, observed=True)
    assert result["clear"]
    assert kin.last_fk_q[0] == q[0]
    assert result["elbow_z_m"] == pytest.approx(3.504)
    q[0] = 3.006
    assert not limits.evaluate(kin, np.eye(4), q, observed=True)["clear"]


@pytest.mark.parametrize("tolerance", [.0051, -.001, float("nan")])
def test_posture_rejects_unbounded_observation_tolerance(tolerance):
    data = profile()
    data["observed_joint_tolerance_rad"] = tolerance
    with pytest.raises(ValueError, match="invalid posture bounds"):
        PostureLimits.from_dict(data, PostureKin())


def test_high_ceiling_rejected_before_any_physics():
    value = controller()
    plan = row("R1")
    plan["posture_limits"] = profile(1.001)
    with pytest.raises(ValueError, match="ceiling must not exceed"):
        value.run([(target("R1"), plan)])
    assert not value.env.rows


def test_explicit_shoulder_allowance_is_published_and_never_implicit(monkeypatch):
    value = controller()
    plan = row("R1")
    plan["posture_limits"] = dict(profile(1.075), max_elbow_above_shoulder_m=.075,
                                  min_contact_elbow_below_wrist_m=.03)
    monkeypatch.setattr(ExpertController, "run", lambda *args: {"success": True})
    result = value.run([(target("R1"), plan)])
    assert result["posture_limits"]["max_elbow_above_shoulder_m"] == .075
    assert result["posture_limits"]["min_contact_elbow_below_wrist_m"] == .03
    plan["posture_limits"]["max_elbow_z_m"] = 1.07501
    with pytest.raises(ValueError, match="declared allowance"):
        controller().run([(target("R1"), plan)])
    old = PostureLimits.from_dict(profile(), PostureKin())
    assert old.max_elbow_above_shoulder_m == old.min_contact_elbow_below_wrist_m == 0.


@pytest.mark.parametrize("field,value", [
    ("max_elbow_above_shoulder_m", .07501), ("max_elbow_above_shoulder_m", -.001),
    ("max_elbow_above_shoulder_m", float("nan")), ("min_contact_elbow_below_wrist_m", -.001),
    ("min_contact_elbow_below_wrist_m", float("inf"))])
def test_invalid_explicit_posture_profile_rejected(field, value):
    with pytest.raises(ValueError, match="invalid posture bounds"):
        PostureLimits.from_dict(dict(profile(), **{field: value}), PostureKin())


def test_contact_gap_is_strict_despite_joint_solver_tolerance():
    limits = PostureLimits.from_dict(dict(profile(), min_contact_elbow_below_wrist_m=.03), PostureKin())
    q = np.zeros(7)
    q[0] = .0201
    assert limits.evaluate(PostureKin(), np.eye(4), q)["clear"]
    result = limits.evaluate(PostureKin(), np.eye(4), q, observed=True, contact=True)
    assert result["posture_violations"] == ["contact_elbow_below_wrist"]
    assert result["elbow_below_wrist_m"] == pytest.approx(.0299)


@pytest.mark.parametrize("phase", [Phase.CONTACT, Phase.HOLD, Phase.RETRACT])
def test_contact_phase_commands_must_keep_elbow_below_wrist(phase):
    value = controller()
    value.posture_limits = PostureLimits.from_dict(dict(profile(), min_contact_elbow_below_wrist_m=.06), value.kin)
    value.manager._transition(phase)
    with pytest.raises(EpisodeRejected, match="contact_elbow_below_wrist"):
        value._block(np.zeros((CONTROL_STEPS, 7)), "contact_cycle")
    assert not value.env.rows


def test_contact_phase_transition_inside_block_checks_the_next_command(monkeypatch):
    value = controller()
    value.posture_limits = PostureLimits.from_dict(dict(profile(), min_contact_elbow_below_wrist_m=.06), value.kin)
    executed = []

    def transition(self, command, subphase, **kwargs):
        executed.append(command.copy())
        self.manager._transition(Phase.CONTACT)
        return self.env.state()

    monkeypatch.setattr(ExpertController, "_tick", transition)
    with pytest.raises(EpisodeRejected, match="contact_elbow_below_wrist"):
        value._block(np.zeros((CONTROL_STEPS, 7)), "contact_cycle", active=True)
    assert len(executed) == 1


def test_actual_contact_gap_violation_latches_after_first_frame():
    class GapDriftEnv(FakeEnv):
        def step(self, *args, **kwargs):
            super().step(*args, **kwargs)
            self.q[0] = .03
            return self.state()
    value = controller(GapDriftEnv())
    value.posture_limits = PostureLimits.from_dict(dict(profile(), min_contact_elbow_below_wrist_m=.03), value.kin)
    value.manager._transition(Phase.CONTACT)
    with pytest.raises(EpisodeRejected, match="contact_elbow_below_wrist"):
        value._block(np.zeros((CONTROL_STEPS, 7)), "contact_cycle")
    assert len(value.env.rows) == 1
    assert value.env.rows[-1]["elbow_below_wrist_m"] == pytest.approx(.02)
    with pytest.raises(EpisodeRejected, match="posture failure latched"):
        value._block(np.zeros((CONTROL_STEPS, 7)), "fault_retract", recovery=True)
    assert len(value.env.rows) == 1


def test_pose_tracking_uses_only_bounded_closest_seed_solver(monkeypatch):
    from sim.auscultation_training import pose_ik
    value, calls = controller(), []
    value.manager._transition(Phase.CONTACT)

    def constrained(kin, base, limits, position, rotation, seed, **kwargs):
        calls.append((seed.copy(), kwargs.copy()))
        q = seed.copy()
        q[2] += .001
        return q, True, 0., 0.

    def forbidden(*args, **kwargs):
        raise AssertionError("unconstrained IK must never be called")

    monkeypatch.setattr(pose_ik, "solve_posture_ik", constrained)
    monkeypatch.setattr(value.kin, "solve_ik", forbidden)
    value._pose_block(np.array([0., 0., .299]), value.rotation_base, "contact_cycle")
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.zeros(7))
    assert calls[0][1] == {"maxiter": 80, "planning_margin_m": .002,
                          "joint_margin_rad": 0., "contact": True}
    assert len(value.env.rows) == CONTROL_STEPS
    assert value.q[2] == pytest.approx(.001)


@pytest.mark.parametrize("failure", ["infeasible", "branch_jump"])
def test_pose_tracking_rejects_without_unconstrained_fallback_or_motion(monkeypatch, failure):
    from sim.auscultation_training import pose_ik
    value, calls = controller(), []

    def constrained(*args, **kwargs):
        calls.append(True)
        return np.array([0., 0., .2, 0., 0., 0., 0.]), failure != "infeasible", 0., 0.

    def forbidden(*args, **kwargs):
        raise AssertionError("unconstrained IK must never be called")

    monkeypatch.setattr(pose_ik, "solve_posture_ik", constrained)
    monkeypatch.setattr(value.kin, "solve_ik", forbidden)
    with pytest.raises(EpisodeRejected, match="bounded posture IK failed|command speed limit"):
        value._pose_block(np.array([0., 0., .299]), value.rotation_base, "contact_cycle")
    assert len(calls) == 1
    assert not value.env.rows


def test_every_command_checked_before_any_of_block_is_executed():
    value = controller()
    commands = np.zeros((CONTROL_STEPS, 7))
    commands[5, 0] = .6
    with pytest.raises(EpisodeRejected, match="command posture rejected: elbow_height"):
        value._block(commands, "transfer")
    assert not value.env.rows


@pytest.mark.parametrize("recovery", [False, True])
def test_actual_high_elbow_frame_stops_immediately_even_during_recovery(recovery):
    class DriftEnv(FakeEnv):
        def step(self, *args, **kwargs):
            super().step(*args, **kwargs)
            self.q[0] = .6
            return self.state()
    value = controller(DriftEnv())
    with pytest.raises(EpisodeRejected, match="elbow_height"):
        value._block(np.zeros((CONTROL_STEPS, 7)), "fault_retract" if recovery else "transfer",
                     recovery=recovery)
    assert len(value.env.rows) == 1
    assert value.env.rows[-1]["posture_violations"] == ["elbow_height"]
    assert value.control_rows[-1]["executed_physics_steps"] == 1
    assert not value.control_rows[-1]["command_block_complete"]
    with pytest.raises(EpisodeRejected, match="posture failure latched"):
        value._block(np.zeros((CONTROL_STEPS, 7)), "fault_retract", recovery=True)
    assert len(value.env.rows) == 1


def verified_fixture(path):
    outcome = {"success": True, "training_accepted": True,
               "controller_type": "algorithmic_expert_low_elbow",
               "posture_limits": profile(), "requested_points": ["R1"],
               "points": [{"target": "R1", "success": True, "training_accepted": True}]}
    artifacts = {"outcome.json": json.dumps(outcome).encode(),
                 "control_commands.json": b"[]", "expert_trace.jsonl": b"{}\n{}\n",
                 "expert_replay.npz": b"fixture bytes, independently audited in child"}
    for name, data in artifacts.items():
        (path / name).write_bytes(data)
    required = {"mechanics": ("outcome.json", "control_commands.json", "expert_trace.jsonl"),
                "posture": ("outcome.json", "expert_trace.jsonl", "expert_replay.npz")}
    report = {"accepted": True, "audits": {
        kind: {"accepted": True, "artifact_sha256": {
            name: hashlib.sha256(artifacts[name]).hexdigest() for name in names}}
        for kind, names in required.items()}}
    sources = {"kinematics_urdf": path / "fixture.urdf",
               "world_from_left_base": path / "world_from_left_base.json"}
    report["audits"]["posture"]["source_sha256"] = {}
    for name, source in sources.items():
        source.write_bytes(b"independent source fixture")
        report["audits"]["posture"]["source_sha256"][name] = {
            "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (path / "verification_summary.json").write_text(json.dumps(report))
    return report, outcome


def test_outer_verifier_accepts_complete_nested_evidence(tmp_path):
    verified_fixture(tmp_path)
    assert verify_saved(tmp_path, 0)["accepted"]


@pytest.mark.parametrize("kind", ["mechanics", "posture"])
def test_outer_verifier_rejects_empty_evidence_despite_child_zero_and_accepted_flags(tmp_path, kind):
    report, _ = verified_fixture(tmp_path)
    report["audits"][kind]["artifact_sha256"] = {}
    (tmp_path / "verification_summary.json").write_text(json.dumps(report))
    assert not verify_saved(tmp_path, 0)["accepted"]


@pytest.mark.parametrize("kind", ["mechanics", "posture"])
def test_outer_verifier_requires_both_actual_acceptances(tmp_path, kind):
    report, _ = verified_fixture(tmp_path)
    report["audits"][kind]["accepted"] = False
    (tmp_path / "verification_summary.json").write_text(json.dumps(report))
    assert not verify_saved(tmp_path, 0)["accepted"]


@pytest.mark.parametrize("mutation", ["point_failed", "not_training_accepted", "wrong_controller"])
def test_outer_verifier_cannot_pass_incomplete_or_wrong_controller_outcome(tmp_path, mutation):
    report, outcome = verified_fixture(tmp_path)
    if mutation == "point_failed":
        outcome["points"][0]["success"] = False
    elif mutation == "not_training_accepted":
        outcome["training_accepted"] = False
    else:
        outcome["controller_type"] = "algorithmic_expert"
    data = json.dumps(outcome).encode()
    (tmp_path / "outcome.json").write_bytes(data)
    for audit in report["audits"].values():
        audit["artifact_sha256"]["outcome.json"] = hashlib.sha256(data).hexdigest()
    (tmp_path / "verification_summary.json").write_text(json.dumps(report))
    assert not verify_saved(tmp_path, 0)["accepted"]


@pytest.mark.parametrize("report", [[], {"accepted": True, "audits": []},
                                  {"accepted": True, "audits": {"mechanics": [], "posture": {}}}])
def test_outer_verifier_rejects_malformed_evidence_without_crashing(tmp_path, report):
    verified_fixture(tmp_path)
    (tmp_path / "verification_summary.json").write_text(json.dumps(report))
    assert not verify_saved(tmp_path, 0)["accepted"]


def test_native_zero_exit_cannot_override_missing_or_rejected_saved_evidence(tmp_path):
    assert not verify_saved(tmp_path, 0)["accepted"]
    verified_fixture(tmp_path)
    (tmp_path / "expert_replay.npz").unlink()
    assert not verify_saved(tmp_path, 0)["accepted"]


def test_outer_verifier_rejects_artifact_changes_and_child_failure(tmp_path):
    verified_fixture(tmp_path)
    assert not verify_saved(tmp_path, 1)["accepted"]
    (tmp_path / "expert_trace.jsonl").write_bytes(b"changed")
    assert not verify_saved(tmp_path, 0)["accepted"]


def test_outer_verifier_rechecks_base_dependency_after_posture_audit(tmp_path):
    verified_fixture(tmp_path)
    (tmp_path / "world_from_left_base.json").write_bytes(b"changed base")
    assert not verify_saved(tmp_path, 0)["accepted"]
