"""Frozen waypoint-conditioned BC; same physical supervisor, explicit safety projection."""
from dataclasses import dataclass
import io
import json
from pathlib import Path

import numpy as np
import torch

from .controller import CONTROL_DT, EpisodeRejected, digest
from .dataset_v2 import ACTION_LIMIT_RAD, FRAME_FEATURES, HISTORY_STEPS, ObservationHistory, JOINT_SUBPHASES
from .policy import BoundedJointPolicy
from .student import StudentController

MODEL_CODE_FILES = ("dataset.py", "dataset_v2.py", "policy.py", "train_bc.py", "train_bc_v2.py")


@dataclass
class FrozenActorV2:
    policy: BoundedJointPolicy
    metadata: dict
    artifacts: dict

    def predict(self, history):
        history = np.asarray(history, np.float32)
        if history.shape != (HISTORY_STEPS, len(FRAME_FEATURES)) or not np.isfinite(history).all():
            raise ValueError("invalid v2 actor history")
        with torch.inference_mode():
            return self.policy(torch.from_numpy(history[None])).cpu().numpy()[0].copy()


def load_actor(directory, expected_manifest_sha256):
    directory = Path(directory)
    artifacts = {name:(directory/name).read_bytes() for name in ("run_manifest.json", "policy.pt")}
    if digest(artifacts["run_manifest.json"]) != expected_manifest_sha256:
        raise ValueError("training run manifest fingerprint mismatch")
    manifest = json.loads(artifacts["run_manifest.json"])
    checkpoint_sha = digest(artifacts["policy.pt"])
    if checkpoint_sha != manifest.get("checkpoint_sha256"):
        raise ValueError("checkpoint fingerprint mismatch")
    if manifest.get("schema") != "auscultation_bc_training_v2":
        raise ValueError("unsupported v2 training manifest schema")
    for name in MODEL_CODE_FILES:
        if digest(Path(__file__).with_name(name).read_bytes()) != manifest.get("code_sha256", {}).get(name):
            raise ValueError("model code fingerprint differs: "+name)
    checkpoint = torch.load(io.BytesIO(artifacts["policy.pt"]), map_location="cpu", weights_only=True)
    input_dim = HISTORY_STEPS*len(FRAME_FEATURES)
    if (not isinstance(checkpoint, dict) or checkpoint.get("schema") != "auscultation_bc_policy_v2"
            or checkpoint.get("input_dim") != input_dim or checkpoint.get("history_steps") != HISTORY_STEPS
            or checkpoint.get("frame_features") != list(FRAME_FEATURES)
            or checkpoint.get("action_limit_rad") != ACTION_LIMIT_RAD
            or checkpoint.get("dataset_manifest_sha256") != manifest.get("dataset_manifest_sha256")):
        raise ValueError("checkpoint observation/action schema mismatch")
    architecture = {"type":"shared_waypoint_conditioned_MLP", "input_dim":input_dim,
        "hidden":checkpoint.get("hidden"), "output_dim":7, "output":"tanh times 0.0125 rad"}
    if manifest.get("architecture") != architecture:
        raise ValueError("checkpoint architecture differs from pinned manifest")
    policy = BoundedJointPolicy(input_dim, hidden=checkpoint["hidden"])
    state = checkpoint.get("state_dict", {})
    if not state or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("invalid model state tensors")
    policy.load_state_dict(state, strict=True)
    if torch.any(policy.input_std <= 0):
        raise ValueError("invalid model scaler")
    for key, tensor in (("mean",policy.input_mean),("std",policy.input_std)):
        expected = np.asarray(manifest.get("scaler", {}).get(key), float)
        if expected.shape != (input_dim,) or not np.allclose(tensor.numpy(), expected, atol=1e-7, rtol=0):
            raise ValueError("model scaler differs from pinned manifest")
    speeds = manifest.get("training_speed_scales")
    if not isinstance(speeds, list) or not speeds or any(type(s) not in (int,float) or not .7 <= s <= 1 for s in speeds):
        raise ValueError("missing supported training speed scales")
    policy.eval().requires_grad_(False)
    return FrozenActorV2(policy, {"checkpoint_sha256":checkpoint_sha,
        "run_manifest_sha256":expected_manifest_sha256,"architecture":architecture,
        "dataset_manifest_sha256":checkpoint["dataset_manifest_sha256"], "history_steps":HISTORY_STEPS,
        "frame_features":list(FRAME_FEATURES), "device":"cpu", "weight_updates":False,
        "training_speed_scales":speeds, "waypoint_source":"live checked planner waypoint; no demonstration progress"}, artifacts)


class StudentControllerV2(StudentController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = ObservationHistory()
        self.current_waypoint = None

    def _condition(self, subphase):
        valid = subphase in JOINT_SUBPHASES
        if valid and self.current_waypoint is None:
            raise EpisodeRejected("missing live checked waypoint")
        return {"local_waypoint_valid":valid,
            "local_waypoint_q":self.current_waypoint.tolist() if valid else None,
            "speed_scale":self.speed_scale}

    def _observed_row(self, subphase):
        return dict(super()._observed_row(subphase), **self._condition(subphase))

    def _joint_segment(self, destination, subphase):
        self.current_waypoint = np.asarray(destination, float).copy()
        try:
            return super()._joint_segment(destination, subphase)
        finally:
            self.current_waypoint = None

    def _block(self, commands, subphase, active=False, recovery=False, gap=None):
        condition = self._condition(subphase)
        before = len(self.control_rows)
        try:
            return super()._block(commands, subphase, active=active, recovery=recovery, gap=gap)
        finally:
            if len(self.control_rows) > before:
                self.control_rows[-1].update(condition, observation_schema="auscultation_bc_observation_v2")

    def compose_action(self, raw, state):
        action, metadata = super().compose_action(raw, state)
        lower, upper = np.asarray(self.kin.lower_limits), np.asarray(self.kin.upper_limits)
        if np.any(self.q < lower) or np.any(self.q > upper):
            raise EpisodeRejected("prior learned command outside joint limits")
        requested = action.copy()
        # Euclidean projection onto the intersection of velocity and position
        # boxes cannot introduce movement toward a waypoint or an expert action.
        limit = self.joint_speed_limit*CONTROL_DT
        destination = np.clip(self.q+action, np.maximum(lower,self.q-limit), np.minimum(upper,self.q+limit))
        action = destination-self.q
        correction = action-requested
        metadata.update(joint_position_projection_applied=bool(np.any(np.abs(correction)>1e-15)),
            pre_position_projection_action_rad=requested.tolist(),
            joint_position_projection_delta_rad=correction.tolist(),
            applied_action_rad=action.tolist(),
            position_projection_scope="joint box projection only; collision/force/dwell/watchdog gates unchanged")
        if metadata["normal_feedback_applied"]:
            jacobian = np.asarray(self.kin.jacobian(self.q), float)
            metadata["applied_normal_step_m"] = float(self.normal_base @ (jacobian @ action)[:3])
            metadata["joint_position_projection_twist_delta"] = (jacobian @ correction).tolist()
            metadata["projection_model"] += "; position-box projection may alter tangent/rotation; actual twist delta reported"
        return action, metadata

    def run(self, points):
        outcome = super().run(points)
        outcome.update(observation_schema="auscultation_bc_observation_v2",
            waypoint_conditioning="live checked local joint waypoint error plus validity and requested speed",
            joint_limit_supervision="logged command box projection; no nominal analytic action rescue")
        return outcome
