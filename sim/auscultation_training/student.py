"""Frozen BC inference with shared simulation supervision and rejection gates.

Normal motion is learned. Optional contact feedback replaces, rather than adds
to, the learned normal component while loaded in CONTACT/HOLD. Stationary holds
and failed-episode recovery are explicit supervisor actions, never expert data.
"""
from dataclasses import dataclass
import io
import json
from pathlib import Path

import numpy as np
import torch

from .controller import CONTROL_DT, CONTROL_STEPS, DT, EpisodeRejected, ExpertController, digest
from .dataset import ACTION_LIMIT_RAD, FRAME_FEATURES, HISTORY_STEPS, ObservationHistory, encode_frame
from .phase_manager import Phase
from .policy import BoundedJointPolicy

# The online history is exactly the dataset implementation, not a second encoder.
ActorHistory = ObservationHistory


@dataclass
class FrozenActor:
    policy: BoundedJointPolicy
    metadata: dict
    artifacts: dict

    def predict(self, history):
        history = np.asarray(history, np.float32)
        if history.shape != (HISTORY_STEPS, len(FRAME_FEATURES)) or not np.isfinite(history).all():
            raise ValueError("invalid actor history")
        with torch.inference_mode():
            return self.policy(torch.from_numpy(history[None])).cpu().numpy()[0].copy()


def load_actor(directory, expected_manifest_sha256):
    """Load only a pinned local tensor checkpoint; never deserialize Python code."""
    directory = Path(directory)
    artifacts = {name: (directory / name).read_bytes() for name in ("run_manifest.json", "policy.pt")}
    if digest(artifacts["run_manifest.json"]) != expected_manifest_sha256:
        raise ValueError("training run manifest fingerprint mismatch")
    manifest = json.loads(artifacts["run_manifest.json"])
    checkpoint_sha = digest(artifacts["policy.pt"])
    if checkpoint_sha != manifest.get("checkpoint_sha256"):
        raise ValueError("checkpoint fingerprint mismatch")
    if manifest.get("schema") != "auscultation_bc_training_v1":
        raise ValueError("unsupported training manifest schema")
    for name in ("dataset.py", "policy.py", "train_bc.py"):
        if digest(Path(__file__).with_name(name).read_bytes()) != manifest.get("code_sha256", {}).get(name):
            raise ValueError("model code fingerprint differs: " + name)
    checkpoint = torch.load(io.BytesIO(artifacts["policy.pt"]), map_location="cpu", weights_only=True)
    input_dim = HISTORY_STEPS * len(FRAME_FEATURES)
    if (not isinstance(checkpoint, dict) or checkpoint.get("schema") != "auscultation_bc_policy_v1"
            or checkpoint.get("input_dim") != input_dim
            or checkpoint.get("history_steps") != HISTORY_STEPS
            or checkpoint.get("frame_features") != list(FRAME_FEATURES)
            or checkpoint.get("action_limit_rad") != ACTION_LIMIT_RAD
            or checkpoint.get("dataset_manifest_sha256") != manifest.get("dataset_manifest_sha256")):
        raise ValueError("checkpoint observation/action schema mismatch")
    architecture = {"type": "shared_goal_conditioned_MLP", "input_dim": input_dim,
                    "hidden": checkpoint.get("hidden"), "output_dim": 7,
                    "output": "tanh times 0.0125 rad"}
    if manifest.get("architecture") != architecture:
        raise ValueError("checkpoint architecture differs from pinned manifest")
    policy = BoundedJointPolicy(input_dim, hidden=checkpoint["hidden"])
    state = checkpoint.get("state_dict", {})
    if not state or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("invalid model state tensors")
    policy.load_state_dict(state, strict=True)
    if torch.any(policy.input_std <= 0):
        raise ValueError("invalid model scaler")
    for key, tensor in (("mean", policy.input_mean), ("std", policy.input_std)):
        expected = np.asarray(manifest.get("scaler", {}).get(key), dtype=float)
        if expected.shape != (input_dim,) or not np.allclose(tensor.numpy(), expected, atol=1e-7, rtol=0):
            raise ValueError("model scaler differs from pinned manifest")
    policy.eval().requires_grad_(False)
    return FrozenActor(policy, {"checkpoint_sha256": checkpoint_sha,
        "run_manifest_sha256": expected_manifest_sha256, "architecture": architecture,
        "dataset_manifest_sha256": checkpoint["dataset_manifest_sha256"],
        "history_steps": HISTORY_STEPS, "frame_features": list(FRAME_FEATURES),
        "device": "cpu", "weight_updates": False}, artifacts)


class StudentController(ExpertController):
    """Learned increments with observed-waypoint progression; no motion fallback."""
    def __init__(self, env, kinematics, checker, *, actor, contact_mode="hybrid_normal",
                 waypoint_timeout_s=None, **kwargs):
        if contact_mode not in ("pure_bc", "hybrid_normal"):
            raise ValueError("unsupported contact mode")
        if waypoint_timeout_s is not None and (not np.isfinite(waypoint_timeout_s) or not 0 < waypoint_timeout_s <= 180):
            raise ValueError("waypoint timeout must be positive and no greater than 180 seconds")
        super().__init__(env, kinematics, checker, **kwargs)
        self.actor, self.contact_mode = actor, contact_mode
        self.waypoint_timeout_s = waypoint_timeout_s
        self.history = ActorHistory()
        self.pending_prediction = None
        self.waypoint_events = []
        self.block_terminal_event = None

    def _reject_supervisor_failure(self):
        state = self.manager.state
        if state.phase == Phase.ABORT or state.failure_reason is not None:
            raise EpisodeRejected("supervisor failure in " + state.phase.value + ": " + str(state.failure_reason))

    def _tick(self, command, subphase, active=False, recovery=False):
        if not recovery:
            self._reject_supervisor_failure()
        # Once contact release succeeds, finish this fixed-duration block with
        # the last applied command. Explicit later return stages are separate.
        if active and self.manager.state.phase == Phase.DONE:
            command = self.q.copy()
            if self.block_terminal_event is not None:
                self.block_terminal_event["terminal_stationary_ticks"] += 1
        state = super()._tick(command, subphase, active=active, recovery=recovery)
        if active:
            supervisor = self.manager.state
            failed = supervisor.phase == Phase.ABORT or supervisor.failure_reason is not None
            if failed or supervisor.phase == Phase.DONE:
                if self.block_terminal_event is None:
                    self.block_terminal_event = {
                        "supervisor_terminal_phase": supervisor.phase.value,
                        "supervisor_failure_reason": supervisor.failure_reason,
                        "terminal_event_physics_row": len(self.env.rows) - 1,
                        "terminal_stationary_ticks": 0,
                    }
                # Raising here prevents the next command in the twelve-tick
                # block from executing. Only labeled fault recovery may follow.
                if failed:
                    self._reject_supervisor_failure()
        return state

    def _observed_row(self, subphase):
        state = self.env.state()
        return {"target": self.target.name, "phase": self.manager.state.phase.value, "subphase": subphase,
            "previous_q_command": self.q.tolist(), "observation": {
                "q": state["q"].tolist(), "dq": state["dq"].tolist(),
                "tcp_base_m": state["tcp_base"].tolist(),
                "goal_relative_base_m": (self.target.position_base - state["tcp_base"]).tolist(),
                "normal_base": self.normal_base.tolist(), "hall_N": float(state["hall_N"]),
                "compression_m": float(state["compression_m"]),
                "tcp_rotation_base": (self.env.base[:3, :3].T @ state["rotation_world"]).tolist(),
                "target_rotation_base": self.rotation_base.tolist()}}

    def _block(self, commands, subphase, active=False, recovery=False, gap=None):
        if not recovery:
            self._reject_supervisor_failure()
        self.block_terminal_event = None
        pending = self.pending_prediction
        if pending is None and not recovery:
            self.history.push(self._observed_row(subphase))
        before = len(self.control_rows)
        try:
            return super()._block(commands, subphase, active=active, recovery=recovery, gap=gap)
        finally:
            if len(self.control_rows) > before:
                row = self.control_rows[-1]
                row.update(pending or {"controller_source": "analytic_fault_recovery" if recovery else "supervisor_stationary_hold"})
                if recovery:
                    row["phase"] = "RECOVERY"
                row["expert_demonstration"] = False
                row["action_interpolation"] = "linear joint-command increments at 240 Hz; exact applied endpoints recorded"
                # The actor proposal is not the executed action when a terminal
                # event interrupts movement. Bind labels to the actual trace.
                executed = row["executed_physics_steps"]
                row["planned_action_duration_s"] = row["action_duration_s"]
                row["requested_q_command"] = row["q_command"]
                prior = np.asarray(row["previous_q_command"], float)
                endpoint = (np.asarray(self.env.rows[row["physics_row_end_exclusive"] - 1]["q_command"], float)
                            if executed else prior)
                row["q_command"] = endpoint.tolist()
                row["action_joint_delta_rad"] = (endpoint - prior).tolist()
                row["action_duration_s"] = executed * DT
                if pending is not None:
                    row["requested_applied_action_rad"] = pending["applied_action_rad"]
                    row["applied_action_rad"] = row["action_joint_delta_rad"]
                    if pending["normal_feedback_applied"]:
                        row["requested_applied_normal_step_m"] = pending["applied_normal_step_m"]
                        row["applied_normal_step_m"] = float(self.normal_base @
                            (self.kin.jacobian(prior) @ (endpoint - prior))[:3])
                if not row["command_block_complete"]:
                    row["training_accepted"] = False
                    row["action_interpolation"] = "interrupted command block; actual partial endpoint and duration recorded; rejected"
                if self.block_terminal_event is not None:
                    row.update(self.block_terminal_event)
                    if row["command_block_complete"]:
                        row["action_interpolation"] = "linear commands until contact DONE, then stationary command remainder; physics trace is authoritative"
            self.pending_prediction = None
            self.block_terminal_event = None

    def compose_action(self, raw, state):
        raw = np.asarray(raw, float)
        if raw.shape != (7,) or not np.isfinite(raw).all() or np.max(np.abs(raw)) > ACTION_LIMIT_RAD + 1e-8:
            raise EpisodeRejected("invalid learned action")
        action = raw.copy()
        metadata = {"normal_feedback_applied": False, "raw_policy_action_rad": raw.tolist()}
        if (self.contact_mode == "hybrid_normal" and self.manager.state.phase in (Phase.CONTACT, Phase.HOLD)
                and (state["hall_N"] > .3 or state["compression_m"] > .00015)):
            jacobian = np.asarray(self.kin.jacobian(self.q), float)
            if jacobian.shape != (6, 7) or not np.isfinite(jacobian).all():
                raise EpisodeRejected("invalid feedback Jacobian")
            learned_twist = jacobian @ raw
            learned_normal = float(self.normal_base @ learned_twist[:3])
            requested_normal = float(np.clip((state["hall_N"] - self.config.target_force_N) * .001, -.002, .002)) * CONTROL_DT
            correction_twist = np.r_[self.normal_base * (requested_normal - learned_normal), np.zeros(3)]
            correction = np.linalg.pinv(jacobian, rcond=1e-5) @ correction_twist
            if np.linalg.norm(jacobian @ correction - correction_twist) > 1e-6:
                raise EpisodeRejected("contact Jacobian cannot preserve tangent/rotation feedback constraint")
            action += correction
            metadata.update(normal_feedback_applied=True, learned_normal_step_m=learned_normal,
                            requested_normal_step_m=requested_normal, normal_correction_joint_rad=correction.tolist(),
                            projection_model="first-order FK Jacobian at prior command; joint safety scaling applied afterward")
        # Uniform safety scaling bounds the actually applied joint increment;
        # the resulting normal step is reported, never substituted in the label.
        limit = self.joint_speed_limit * CONTROL_DT
        scale = min(1., limit / max(float(np.max(np.abs(action))), 1e-15))
        action *= scale
        metadata.update(joint_bound_scale=scale, applied_action_rad=action.tolist())
        if metadata["normal_feedback_applied"]:
            metadata["applied_normal_step_m"] = float(self.normal_base @ (jacobian @ action)[:3])
        return action, metadata

    def _preflight(self, destination, subphase):
        state = self.env.state()
        if subphase != "contact_cycle" or self.metrics(state)["gap_m"] > .015:
            check = self.checker.edge(self.q, destination)
            if not check["clear"]:
                raise EpisodeRejected("learned free-motion collision preflight rejected action")
        else:
            # Near contact only the already designated front surfaces may touch.
            for fraction in np.linspace(0., 1., CONTROL_STEPS + 1):
                check = self.checker.check(self.q + fraction * (destination - self.q), clearance=.0003,
                    allow_patient_contact=True, compression=max(0., state["compression_m"]))
                if not check["clear"]:
                    raise EpisodeRejected("learned contact collision preflight rejected action")

    def _learned_step(self, subphase, active=False):
        self._reject_supervisor_failure()
        if subphase == "contact_cycle" and self.manager.state.phase == Phase.DONE:
            return self.env.state()
        observed = self._observed_row(subphase)
        history = self.history.push(observed)
        raw = self.actor.predict(history)
        action, metadata = self.compose_action(raw, self.env.state())
        destination = self.q + action
        if np.any(destination < self.kin.lower_limits) or np.any(destination > self.kin.upper_limits):
            raise EpisodeRejected("learned action exceeds joint limits")
        self._preflight(destination, subphase)
        self.pending_prediction = {"controller_source": "learned_bc_with_normal_feedback" if metadata["normal_feedback_applied"] else "learned_bc",
            "policy_input_sha256": digest(history.tobytes()), "policy_checkpoint_sha256": self.actor.metadata.get("checkpoint_sha256"), **metadata}
        commands = self.q + np.arange(1, CONTROL_STEPS + 1)[:, None] / CONTROL_STEPS * action
        return self._block(commands, subphase, active=active)

    def _waypoint_reached(self, destination):
        state = self.env.state()
        goal = self.kin.tcp(destination)
        rotation = self.env.base[:3, :3].T @ state["rotation_world"]
        angle = float(np.arccos(np.clip((np.trace(goal[:3, :3].T @ rotation) - 1) / 2, -1, 1)))
        return (np.max(np.abs(state["q"] - destination)) <= .025
                and np.max(np.abs(self.q - destination)) <= .025
                and np.linalg.norm(state["tcp_base"] - goal[:3, 3]) <= .003
                and angle <= np.deg2rad(3.) and np.max(np.abs(state["dq"])) <= .06)

    def _joint_segment(self, destination, subphase):
        destination = np.asarray(destination, float)
        distance = float(np.max(np.abs(self.q - destination)))
        timeout = self.waypoint_timeout_s or min(120., max(15., 6. * distance / self.joint_speed_limit))
        event = {"target": self.target.name, "subphase": subphase, "destination_q": destination.tolist(),
                 "start_time_s": self.env.state()["time_s"], "timeout_s": timeout, "reached": False}
        self.waypoint_events.append(event)
        for _ in range(int(np.ceil(timeout / CONTROL_DT))):
            if self._waypoint_reached(destination):
                event.update(reached=True, end_time_s=self.env.state()["time_s"])
                return
            self._learned_step(subphase)
        event["end_time_s"] = self.env.state()["time_s"]
        raise EpisodeRejected("learned waypoint timeout in " + subphase)

    def _contact_cycle(self):
        for _ in range(int(np.ceil(80 / self.speed_scale / CONTROL_DT))):
            self._learned_step("contact_cycle", active=True)
            if self.metrics(self.env.state())["gap_m"] < -.01:
                raise EpisodeRejected("learned contact displacement budget exhausted")
            if self.manager.state.completed:
                break
        if not self.manager.state.success:
            raise EpisodeRejected("learned contact state machine failed: " + str(self.manager.state.failure_reason))
        if self.best_audit_hold < self.config.hold_duration_s - 2 * DT:
            raise EpisodeRejected("learned observable hold lacked independent front contact")

    def _return_home(self, transfer):
        confirmed = 0.
        for _ in range(int(np.ceil(20 / self.speed_scale / CONTROL_DT))):
            state = self.env.state()
            metric = self.metrics(state)
            reached = (abs(metric["gap_m"] - self.pre_gap) <= .002 and metric["tangent_m"] <= .003
                       and metric["angle_deg"] <= 5. and state["hall_N"] < .2
                       and self.env.rows[-1]["tcp_speed_m_s"] < .02)
            confirmed = confirmed + CONTROL_DT if reached else 0.
            if confirmed >= self.config.release_confirm_duration_s:
                break
            self._learned_step("withdraw_to_prepose")
        else:
            raise EpisodeRejected("learned withdrawal to checked prepose timed out")
        self._hold(.2, "withdraw_settle")
        check = self.checker.edge(self.q, transfer[-1])
        self.bridge_checks.append(check)
        if not check["clear"]:
            raise EpisodeRejected("learned return bridge has no sampled clearance")
        self._joint_segment(transfer[-1], "return_bridge")
        for destination in reversed(transfer[:-1]):
            self._joint_segment(destination, "return_home")
        self._hold(.5, "home_settle")
        if np.max(np.abs(self.env.state()["q"])) > .03:
            raise EpisodeRejected("learned dynamic home tracking error")

    def _pose_block(self, position_base, rotation_base, subphase, active=False, recovery=False, gap=None):
        if not recovery:
            raise EpisodeRejected("analytic pose motion is disabled for student evaluation")
        # Failed-episode recovery is the only analytic moving action permitted.
        return super()._pose_block(position_base, rotation_base, subphase, active, recovery, gap)

    def run(self, points):
        self.history.reset()
        outcome = super().run(points)
        outcome.update(controller_type="learned_bc", dataset_role="student_evaluation", expert_demonstration=False,
            expert_fallbacks=0, contact_mode=self.contact_mode, policy=self.actor.metadata,
            privileged_safety_guard="sampled collision geometry checks reject actions; geometry/PhysX truth never enters actor features",
            waypoint_events=self.waypoint_events, evaluation_scope=
            "Learned transfers, approach, retraction and returns; observed-waypoint supervisor; stationary holds; "
            + ("single normal-component Hall feedback during loaded CONTACT/HOLD; " if self.contact_mode == "hybrid_normal" else "pure BC contact actions; ")
            + "analytic recovery only after rejection. No online training, acoustic result, or clinical validation.")
        counts = {}
        for control in self.control_rows:
            source = control.get("controller_source", "unlabeled")
            counts[source] = counts.get(source, 0) + 1
        outcome["action_source_counts"] = counts
        return outcome
