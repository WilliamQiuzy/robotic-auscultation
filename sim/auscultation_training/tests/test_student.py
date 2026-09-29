import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sim.auscultation_training.dataset import ACTION_LIMIT_RAD, FRAME_FEATURES, HISTORY_STEPS, encode_frame, validate_control_trace
from sim.auscultation_training.controller import EpisodeRejected
from sim.auscultation_training.controller import DT, CONTROL_DT
from sim.auscultation_training.phase_manager import ContactObservation, PhaseManager
from dataclasses import replace
from sim.auscultation_training.phase_manager import Phase
from sim.auscultation_training.policy import BoundedJointPolicy
from sim.auscultation_training.student import StudentController, load_actor
from sim.auscultation_training.tests.test_controller import FakeEnv, FakeKin, FakeChecker, target, row


def sha(data):
    return hashlib.sha256(data).hexdigest()


def artifact(tmp_path):
    policy = BoundedJointPolicy(HISTORY_STEPS * len(FRAME_FEATURES), hidden=(8,))
    checkpoint = {"schema": "auscultation_bc_policy_v1", "state_dict": policy.state_dict(),
        "input_dim": policy.input_dim, "hidden": [8], "history_steps": HISTORY_STEPS,
        "frame_features": list(FRAME_FEATURES), "action_limit_rad": ACTION_LIMIT_RAD,
        "dataset_manifest_sha256": "a" * 64}
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    (tmp_path / "policy.pt").write_bytes(buffer.getvalue())
    root = Path(__file__).parents[1]
    manifest = {"schema": "auscultation_bc_training_v1", "checkpoint_sha256": sha(buffer.getvalue()),
        "dataset_manifest_sha256": "a" * 64,
        "architecture": {"type": "shared_goal_conditioned_MLP", "input_dim": policy.input_dim,
                         "hidden": [8], "output_dim": 7, "output": "tanh times 0.0125 rad"},
        "scaler": {"mean": policy.input_mean.tolist(), "std": policy.input_std.tolist()},
        "code_sha256": {name: sha((root / name).read_bytes()) for name in ("dataset.py", "policy.py", "train_bc.py")}}
    data = json.dumps(manifest).encode()
    (tmp_path / "run_manifest.json").write_bytes(data)
    return sha(data)


def test_checkpoint_loaded_with_pinned_manifest_and_frozen_strict_architecture(tmp_path):
    actor = load_actor(tmp_path, artifact(tmp_path))
    action = actor.predict(np.zeros((HISTORY_STEPS, len(FRAME_FEATURES)), np.float32))
    assert action.shape == (7,)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) <= ACTION_LIMIT_RAD
    assert not actor.policy.training
    assert not any(p.requires_grad for p in actor.policy.parameters())
    assert actor.metadata["checkpoint_sha256"] == sha((tmp_path / "policy.pt").read_bytes())


def test_unpinned_or_changed_model_artifacts_are_rejected(tmp_path):
    manifest_sha = artifact(tmp_path)
    with pytest.raises(ValueError, match="manifest fingerprint"):
        load_actor(tmp_path, "0" * 64)
    with (tmp_path / "policy.pt").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        load_actor(tmp_path, manifest_sha)


class ZeroActor:
    metadata = {"checkpoint_sha256": "test", "scope": "fake test actor"}

    def __init__(self):
        self.histories = []

    def predict(self, history):
        self.histories.append(history.copy())
        return np.zeros(7)


def test_stalled_actor_times_out_without_expert_transfer_fallback():
    env, actor = FakeEnv(), ZeroActor()
    controller = StudentController(env, FakeKin(), FakeChecker(), actor=actor,
                                   waypoint_timeout_s=.3, contact_mode="pure_bc")
    outcome = controller.run([(target("R1"), row("R1"))])
    assert not outcome["success"]
    assert "waypoint timeout" in outcome["points"][0]["failure_reason"]
    assert outcome["controller_type"] == "learned_bc"
    assert outcome["dataset_role"] == "student_evaluation"
    assert outcome["expert_fallbacks"] == 0
    assert actor.histories
    assert np.max(np.abs(env.q)) == 0.
    assert all(np.max(np.abs(r["action_joint_delta_rad"])) == 0. for r in controller.control_rows)
    learned_rows = [r for r in controller.control_rows if r["controller_source"] == "learned_bc"]
    assert learned_rows
    assert np.array_equal(actor.histories[0][-1], encode_frame(learned_rows[0]))


def test_hybrid_replaces_normal_component_once_preserving_tangent_rotation_and_nullspace():
    class JacobianKin(FakeKin):
        def jacobian(self, q):
            return np.hstack([np.eye(6), np.zeros((6, 1))])
    controller = StudentController(FakeEnv(), JacobianKin(), FakeChecker(), actor=ZeroActor(),
                                   contact_mode="hybrid_normal")
    controller.normal_base = np.array([0., 0., 1.])
    controller.manager = SimpleNamespace(state=SimpleNamespace(phase=Phase.CONTACT))
    raw = np.array([.002, -.001, -.0012, .0003, .0002, -.0002, .0004])
    applied, metadata = controller.compose_action(raw, {"hall_N": 2., "compression_m": .0005})
    assert applied[2] == pytest.approx(-.0001)
    assert np.allclose(applied[[0, 1, 3, 4, 5, 6]], raw[[0, 1, 3, 4, 5, 6]])
    assert metadata["normal_feedback_applied"]
    assert metadata["learned_normal_step_m"] == pytest.approx(-.0012)
    assert metadata["requested_normal_step_m"] == pytest.approx(-.0001)


def test_pure_bc_does_not_use_hall_to_replace_its_action():
    controller = StudentController(FakeEnv(), FakeKin(), FakeChecker(), actor=ZeroActor(), contact_mode="pure_bc")
    raw = np.array([.001] * 7)
    applied, metadata = controller.compose_action(raw, {"hall_N": 4., "compression_m": .002})
    assert np.array_equal(applied, raw)
    assert not metadata["normal_feedback_applied"]


def test_invalid_actor_action_fails_before_motion():
    class InvalidActor(ZeroActor):
        def predict(self, history):
            return np.full(7, np.nan)
    env = FakeEnv()
    result = StudentController(env, FakeKin(), FakeChecker(), actor=InvalidActor()).run([(target("R1"), row("R1"))])
    assert not result["success"]
    assert "invalid learned action" in result["points"][0]["failure_reason"]
    assert np.max(np.abs(env.q)) == 0.


def test_observed_waypoint_advances_and_all_applied_actions_match_trace():
    class PoseKin(FakeKin):
        def tcp(self, q):
            transform = super().tcp(q)
            transform[:3, :3] = np.diag([1., -1., -1.])
            return transform

    class RecordedEnv(FakeEnv):
        def step(self, q, phase, target_name, meta):
            state = super().step(q, phase, target_name, meta)
            self.rows[-1].update(q=state["q"].tolist(), dq=state["dq"].tolist(),
                hall_N=state["hall_N"], compression_m=state["compression_m"],
                tcp_base_m=state["tcp_base"].tolist(), tcp_rotation_base=state["rotation_world"].tolist())
            return state

    class GoalActor(ZeroActor):
        def predict(self, history):
            self.histories.append(history.copy())
            action = np.zeros(7)
            # This test fixture reacts to observed q; no frame/time counter.
            action[2] = np.clip(.24 - float(history[-1, 2]), -.01, .01)
            return action

    class StopAfterTransfer(StudentController):
        def _contact_cycle(self):
            raise EpisodeRejected("intentional test stop after learned transfer")

    env, actor = RecordedEnv(), GoalActor()
    controller = StopAfterTransfer(env, PoseKin(), FakeChecker(), actor=actor,
                                  waypoint_timeout_s=3., contact_mode="pure_bc")
    outcome = controller.run([(target("R1"), row("R1"))])
    assert not outcome["success"]
    assert outcome["waypoint_events"][0]["reached"]
    assert "intentional test stop" in outcome["points"][0]["failure_reason"]
    learned = [control for control in controller.control_rows if control["controller_source"] == "learned_bc"]
    assert len(learned) == 24
    assert all(control["action_joint_delta_rad"][2] > 0 for control in learned)
    binding = validate_control_trace(controller.control_rows, env.rows)
    assert binding["physics_rows_covered"] == len(env.rows)
    assert all(control["phase"] == "RECOVERY" for control in controller.control_rows if control["recovery"])


def test_collision_guard_rejects_learned_motion_without_substituting_a_path():
    class RejectingChecker(FakeChecker):
        def edge(self, start, end):
            return {"clear": False}
    actor, env = ZeroActor(), FakeEnv()
    outcome = StudentController(env, FakeKin(), RejectingChecker(), actor=actor).run([(target("R1"), row("R1"))])
    assert not outcome["success"]
    assert "collision preflight" in outcome["points"][0]["failure_reason"]
    assert outcome["expert_fallbacks"] == 0
    assert np.max(np.abs(env.q)) == 0.


def test_eval_cli_requires_explicit_model_manifest_pin():
    from sim.auscultation_training.evaluate_bc import main
    with pytest.raises(SystemExit) as error:
        main(["--model-dir", "/absent/model"])
    assert error.value.code == 2


@pytest.mark.parametrize("controller_success", [True, False])
def test_evaluation_artifacts_and_audit_are_persisted_before_simulator_teardown(tmp_path, controller_success):
    from sim.auscultation_training.evaluate_bc import finalize_evaluation
    events = []
    outcome = {"success": controller_success, "contact_mode": "pure_bc", "policy": {"checkpoint_sha256": "a" * 64}}
    controller = SimpleNamespace(control_rows=[{"applied": "fixture"}])

    class Env:
        def save(self, name):
            events.append("save")
            (tmp_path / "expert_trace.jsonl").write_text('{"trace":"fixture"}\n')
            (tmp_path / "training_scene.usdc").write_bytes(b"scene fixture")

    class App:
        def close(self):
            events.append("close")
            assert (tmp_path / "evaluation_summary.json").is_file()
            assert (tmp_path / "independent_audit.json").is_file()

    def audit(path):
        events.append("audit")
        assert events == ["save", "audit"]
        data = json.loads((path / "outcome.json").read_text())
        assert data["training_scene_sha256"] == sha(b"scene fixture")
        hashes = {name: sha((path / name).read_bytes()) for name in
                  ("outcome.json", "expert_trace.jsonl", "control_commands.json")}
        report = {"accepted": True, "reasons": [], "artifact_sha256": hashes}
        (path / "independent_audit.json").write_text(json.dumps(report))
        return report

    assert finalize_evaluation(tmp_path, App(), Env(), controller, outcome, audit_function=audit) == (0 if controller_success else 1)
    assert events == ["save", "audit", "close"]
    saved_audit = json.loads((tmp_path / "independent_audit.json").read_text())
    assert all(sha((tmp_path / name).read_bytes()) == digest for name, digest in saved_audit["artifact_sha256"].items())


def test_evaluation_teardown_still_runs_if_saving_fails(tmp_path):
    from sim.auscultation_training.evaluate_bc import finalize_evaluation
    closed = []

    class Env:
        def save(self, name):
            raise OSError("fixture disk error")

    app = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(OSError, match="disk error"):
        finalize_evaluation(tmp_path, app, Env(), None, {"success": False})
    assert closed == [True]


def test_audit_exception_produces_rejected_summary_before_teardown(tmp_path):
    from sim.auscultation_training.evaluate_bc import finalize_evaluation
    closed = []

    class Env:
        def save(self, name):
            (tmp_path / "expert_trace.jsonl").write_text("{}\n")
            (tmp_path / "training_scene.usdc").write_bytes(b"scene")

    def broken_audit(path):
        raise RuntimeError("fixture audit error")

    def close():
        summary = json.loads((tmp_path / "evaluation_summary.json").read_text())
        assert not summary["closed_loop_accepted"]
        assert summary["audit_reasons"] == ["audit_exception:fixture audit error"]
        closed.append(True)

    outcome = {"success": True, "contact_mode": "pure_bc", "policy": {"checkpoint_sha256": "a" * 64}}
    result = finalize_evaluation(tmp_path, SimpleNamespace(close=close), Env(),
                                SimpleNamespace(control_rows=[]), outcome, audit_function=broken_audit)
    assert result == 1
    assert closed == [True]
    report = json.loads((tmp_path / "independent_audit.json").read_text())
    assert report["mechanical_accepted"] is None
    assert all(sha((tmp_path / name).read_bytes()) == digest for name, digest in report["artifact_sha256"].items())


def terminal_controller(success):
    """A real PhaseManager reaches its terminal state on the first next tick."""
    class MovingActor(ZeroActor):
        def predict(self, history):
            self.histories.append(history.copy())
            return np.array([.006, 0., 0., 0., 0., 0., 0.])
    env, actor = FakeEnv(), MovingActor()
    controller = StudentController(env, FakeKin(), FakeChecker(), actor=actor, contact_mode="pure_bc")
    controller.target = target("R1")
    controller.normal = controller.normal_base = np.array([0., 0., 1.])
    controller.rotation_base = np.diag([1., -1., -1.])
    controller.pre_gap = .06
    controller.face_filtered = controller.total_filtered = controller.audit_hold = 0.
    controller.best_audit_hold = 3.
    config = replace(controller.config, release_confirm_duration_s=DT,
                     retract_timeout_s=20. if success else DT)
    controller.manager = PhaseManager(config)
    controller.manager.update(ContactObservation(0., 0., .04, 0., 0., 0.), DT)
    contact = ContactObservation(4., .002, 0., 0., 0., 0.)
    for _ in range(1000):
        controller.manager.update(contact, DT)
        if controller.manager.state.phase == Phase.RETRACT:
            break
    assert controller.manager.state.phase == Phase.RETRACT
    assert controller.manager.state.failure_reason is None
    return controller, env, actor


def test_latched_retract_failure_blocks_actor_before_any_further_tick():
    controller, env, actor = terminal_controller(False)
    controller.manager.update(ContactObservation(0., 0., .04, 0., 0., 0., target_valid=False), 0.)
    assert controller.manager.state.phase == Phase.RETRACT
    assert controller.manager.state.failure_reason == "target_lost"
    with pytest.raises(EpisodeRejected, match="supervisor.*RETRACT"):
        controller._learned_step("contact_cycle", active=True)
    assert not actor.histories
    assert not env.rows
    assert not controller.control_rows


def test_new_supervisor_failure_stops_at_retract_entry_without_waiting_for_abort():
    controller, env, actor = terminal_controller(False)
    controller.manager = PhaseManager(replace(controller.config, approach_timeout_s=DT))
    with pytest.raises(EpisodeRejected, match="supervisor.*RETRACT"):
        controller._learned_step("contact_cycle", active=True)
    assert controller.manager.state.failure_reason == "approach_timeout"
    assert len(env.rows) == 1
    assert len(actor.histories) == 1
    assert not controller.control_rows[-1]["command_block_complete"]
    assert controller.control_rows[-1]["action_duration_s"] == pytest.approx(DT)


def test_supervisor_abort_stops_on_first_tick_and_records_actual_partial_action():
    controller, env, actor = terminal_controller(False)
    with pytest.raises(EpisodeRejected, match="supervisor.*ABORT"):
        controller._learned_step("contact_cycle", active=True)
    assert controller.manager.state.phase == Phase.ABORT
    assert len(env.rows) == 1
    assert env.q[0] == pytest.approx(.006 / 12)
    control = controller.control_rows[-1]
    assert control["executed_physics_steps"] == 1
    assert not control["command_block_complete"]
    assert not control["training_accepted"]
    assert control["action_duration_s"] == pytest.approx(DT)
    assert control["planned_action_duration_s"] == pytest.approx(CONTROL_DT)
    assert np.allclose(control["q_command"], env.rows[-1]["q_command"])
    assert np.allclose(control["action_joint_delta_rad"], env.q)
    assert control["supervisor_terminal_phase"] == "ABORT"
    with pytest.raises(ValueError, match="invalid bounds"):
        validate_control_trace(controller.control_rows, env.rows)
    calls = len(actor.histories)
    for subphase in ("contact_cycle", "return_home"):
        with pytest.raises(EpisodeRejected, match="supervisor.*ABORT"):
            controller._learned_step(subphase, active=subphase == "contact_cycle")
    assert len(actor.histories) == calls
    assert len(env.rows) == 1


def test_supervisor_done_holds_remainder_with_bound_labels_and_no_more_contact_inference():
    controller, env, actor = terminal_controller(True)
    controller._learned_step("contact_cycle", active=True)
    assert controller.manager.state.phase == Phase.DONE
    assert len(env.rows) == 12
    assert all(row["q_command"][0] == pytest.approx(.006 / 12) for row in env.rows)
    control = controller.control_rows[-1]
    assert control["command_block_complete"]
    assert control["action_duration_s"] == pytest.approx(CONTROL_DT)
    assert control["supervisor_terminal_phase"] == "DONE"
    assert control["terminal_stationary_ticks"] == 11
    assert control["action_joint_delta_rad"][0] == pytest.approx(.006 / 12)
    assert np.allclose(control["q_command"], env.rows[-1]["q_command"])
    assert np.allclose(control["applied_action_rad"], control["action_joint_delta_rad"])
    assert validate_control_trace(controller.control_rows, env.rows)["physics_rows_covered"] == 12
    calls, ticks = len(actor.histories), len(env.rows)
    controller._contact_cycle()
    assert len(actor.histories) == calls
    assert len(env.rows) == ticks
    # DONE closes contact_cycle, while the explicit later return stage stays learned.
    controller._learned_step("return_home")
    assert len(actor.histories) == calls + 1
    assert env.q[0] > .006 / 12


@pytest.mark.parametrize("controller_success", [True, False])
def test_teardown_system_exit_cannot_override_evaluation_exit_status(tmp_path, controller_success):
    from sim.auscultation_training.evaluate_bc import finalize_evaluation

    class Env:
        def save(self, name):
            (tmp_path / "expert_trace.jsonl").write_text("{}\n")
            (tmp_path / "training_scene.usdc").write_bytes(b"scene")

    def close():
        assert (tmp_path / "evaluation_summary.json").is_file()
        raise SystemExit(0)

    outcome = {"success": controller_success, "contact_mode": "pure_bc", "policy": {"checkpoint_sha256": "a" * 64}}
    code = finalize_evaluation(tmp_path, SimpleNamespace(close=close), Env(),
        SimpleNamespace(control_rows=[]), outcome,
        audit_function=lambda path: {"accepted": True, "reasons": []})
    assert code == (0 if controller_success else 1)
    assert json.loads((tmp_path / "teardown_status.json").read_text())["requested_exit_code"] == 0


def test_non_system_exit_teardown_error_remains_visible_with_saved_evidence(tmp_path):
    from sim.auscultation_training.evaluate_bc import finalize_evaluation

    def close():
        raise RuntimeError("fixture teardown failure")

    outcome = {"success": False, "contact_mode": "pure_bc", "policy": {"checkpoint_sha256": "a" * 64}}
    with pytest.raises(RuntimeError, match="teardown failure"):
        finalize_evaluation(tmp_path, SimpleNamespace(close=close), None, None, outcome)
    assert (tmp_path / "outcome.json").is_file()
    assert not json.loads((tmp_path / "evaluation_summary.json").read_text())["closed_loop_accepted"]
