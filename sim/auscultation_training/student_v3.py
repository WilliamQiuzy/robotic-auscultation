"""Frozen learned progression on planner routes; learned contact/withdraw head.

The planner supplies route direction. The network supplies only its progress
fraction there. Shared normal-force assistance, supervision, and rejected-only
fault recovery remain explicit; no nominal analytic rescue is introduced.
"""
from dataclasses import dataclass
import io
import json
from pathlib import Path

import numpy as np
import torch

from .controller import CONTROL_DT,CONTROL_STEPS,EpisodeRejected,digest
from .dataset_v2 import ACTION_LIMIT_RAD,FRAME_FEATURES,HISTORY_STEPS,JOINT_SUBPHASES
from .phase_manager import Phase
from .policy_v3 import RouteProgressPolicy,architecture,MOTION_PARAMETERIZATION,ROUTE_DIRECTION_SOURCE
from .student_v2 import StudentControllerV2
from .train_bc_v3 import MODEL_CODE_FILES


def route_metadata():
    return {"policy_version":"v3","motion_parameterization":MOTION_PARAMETERIZATION,
        "route_planning_learned":False,"route_direction_source":ROUTE_DIRECTION_SOURCE,
        "route_progress_min_fraction":0.}


@dataclass
class FrozenActorV3:
    policy:RouteProgressPolicy
    metadata:dict
    artifacts:dict

    def predict_components(self,history):
        values=np.asarray(history,np.float32)
        if values.shape!=(HISTORY_STEPS,len(FRAME_FEATURES)) or not np.isfinite(values).all():
            raise ValueError("invalid v3 actor history")
        with torch.inference_mode():
            parts=self.policy.components(torch.from_numpy(values[None]))
            return {"progress_fraction":float(parts["progress_fraction"][0]),
                "joint_action_rad":parts["joint_action_rad"][0].cpu().numpy().copy()}


def load_actor(directory,expected_manifest_sha256):
    directory=Path(directory)
    artifacts={name:(directory/name).read_bytes() for name in ("run_manifest.json","policy.pt")}
    if digest(artifacts["run_manifest.json"])!=expected_manifest_sha256:
        raise ValueError("training run manifest fingerprint mismatch")
    manifest=json.loads(artifacts["run_manifest.json"]);checkpoint_sha=digest(artifacts["policy.pt"])
    if checkpoint_sha!=manifest.get("checkpoint_sha256"):raise ValueError("checkpoint fingerprint mismatch")
    if manifest.get("schema")!="auscultation_bc_training_v3" or manifest.get("motion_parameterization")!=MOTION_PARAMETERIZATION:
        raise ValueError("unsupported v3 training manifest schema")
    for name in MODEL_CODE_FILES:
        if digest(Path(__file__).with_name(name).read_bytes())!=manifest.get("code_sha256",{}).get(name):
            raise ValueError("model code fingerprint differs: "+name)
    checkpoint=torch.load(io.BytesIO(artifacts["policy.pt"]),map_location="cpu",weights_only=True)
    if (not isinstance(checkpoint,dict) or checkpoint.get("schema")!="auscultation_bc_policy_v3"
            or checkpoint.get("input_dim")!=HISTORY_STEPS*len(FRAME_FEATURES)
            or checkpoint.get("history_steps")!=HISTORY_STEPS or checkpoint.get("frame_features")!=list(FRAME_FEATURES)
            or checkpoint.get("action_limit_rad")!=ACTION_LIMIT_RAD or checkpoint.get("motion_parameterization")!=MOTION_PARAMETERIZATION
            or checkpoint.get("dataset_manifest_sha256")!=manifest.get("dataset_manifest_sha256")):
        raise ValueError("checkpoint observation/action schema mismatch")
    model_architecture=architecture(checkpoint.get("hidden",[]))
    if manifest.get("architecture")!=model_architecture:raise ValueError("checkpoint architecture mismatch")
    policy=RouteProgressPolicy(hidden=checkpoint["hidden"])
    state=checkpoint.get("state_dict",{})
    if not state or any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("invalid model state tensors")
    policy.load_state_dict(state,strict=True)
    if torch.any(policy.input_std<=0):raise ValueError("invalid model scaler")
    for key,tensor in (("mean",policy.input_mean),("std",policy.input_std)):
        expected=np.asarray(manifest.get("scaler",{}).get(key),float)
        if expected.shape!=(policy.input_dim,) or not np.allclose(tensor.numpy(),expected,atol=1e-7,rtol=0):
            raise ValueError("model scaler differs from pinned manifest")
    speeds=manifest.get("training_speed_scales")
    if not isinstance(speeds,list) or not speeds or any(type(s) not in (int,float) or not .7<=s<=1 for s in speeds):
        raise ValueError("missing supported training speed scales")
    policy.eval().requires_grad_(False)
    metadata={"checkpoint_sha256":checkpoint_sha,"run_manifest_sha256":expected_manifest_sha256,
        "architecture":model_architecture,"dataset_manifest_sha256":checkpoint["dataset_manifest_sha256"],
        "history_steps":HISTORY_STEPS,"frame_features":list(FRAME_FEATURES),"device":"cpu","weight_updates":False,
        "training_speed_scales":speeds,**route_metadata()}
    return FrozenActorV3(policy,metadata,artifacts)


class StudentControllerV3(StudentControllerV2):
    def _learned_step(self,subphase,active=False):
        self._reject_supervisor_failure()
        if subphase=="contact_cycle" and self.manager.state.phase==Phase.DONE:return self.env.state()
        observed=self._observed_row(subphase);history=self.history.push(observed)
        prediction=self.actor.predict_components(history)
        fraction=prediction.get("progress_fraction")
        raw_joint=np.asarray(prediction.get("joint_action_rad"),float)
        if type(fraction) not in (int,float) or not np.isfinite(fraction) or not 0<=fraction<=1:
            raise EpisodeRejected("invalid learned progress fraction")
        if raw_joint.shape!=(7,) or not np.isfinite(raw_joint).all() or np.max(np.abs(raw_joint))>ACTION_LIMIT_RAD+1e-8:
            raise EpisodeRejected("invalid learned joint head")
        constrained=subphase in JOINT_SUBPHASES
        metadata={"route_constraint_applied":constrained,"unconstrained_joint_head_action_rad":raw_joint.tolist()}
        if constrained:
            if active or self.manager.state.phase not in (Phase.APPROACH,Phase.DONE):
                raise EpisodeRejected("route progression used outside supervised free-motion stage")
            goal=np.asarray(self.current_waypoint,float)
            if goal.shape!=(7,) or not np.isfinite(goal).all() or np.any(goal<self.kin.lower_limits) or np.any(goal>self.kin.upper_limits):
                raise EpisodeRejected("invalid checked route waypoint")
            remaining=goal-self.q
            base=remaining*min(1.,self.joint_speed_limit*CONTROL_DT/max(float(np.max(np.abs(remaining))),1e-300))
            raw=base*fraction
            metadata.update(predicted_route_fraction=float(fraction),route_base_step_rad=base.tolist(),
                route_remaining_rad=remaining.tolist(),route_direction_source=ROUTE_DIRECTION_SOURCE,
                route_progress_min_fraction=0.,route_planning_learned=False)
        else:
            raw=raw_joint
        action,supervision=self.compose_action(raw,self.env.state())
        if constrained and not np.allclose(action,raw,atol=1e-12,rtol=0):
            raise EpisodeRejected("safety correction would alter checked route direction")
        destination=self.q+action
        if np.any(destination<self.kin.lower_limits) or np.any(destination>self.kin.upper_limits):
            raise EpisodeRejected("learned action exceeds joint limits")
        self._preflight(destination,subphase)
        source=("learned_progress_on_checked_route" if constrained else
            "learned_bc_with_normal_feedback" if supervision["normal_feedback_applied"] else "learned_bc")
        self.pending_prediction={"controller_source":source,"policy_input_sha256":digest(history.tobytes()),
            "policy_checkpoint_sha256":self.actor.metadata.get("checkpoint_sha256"),**supervision,**metadata}
        commands=self.q+np.arange(1,CONTROL_STEPS+1)[:,None]/CONTROL_STEPS*action
        return self._block(commands,subphase,active=active)

    def _block(self,commands,subphase,active=False,recovery=False,gap=None):
        before=len(self.control_rows)
        try:
            return super()._block(commands,subphase,active=active,recovery=recovery,gap=gap)
        finally:
            if len(self.control_rows)>before:
                row=self.control_rows[-1];row.setdefault("route_constraint_applied",False)
                if row["route_constraint_applied"]:
                    base=np.asarray(row["route_base_step_rad"],float);applied=np.asarray(row["action_joint_delta_rad"],float)
                    denominator=float(base@base)
                    actual=float(applied@base/denominator) if denominator else 0.
                    row.update(applied_route_fraction=actual,
                        applied_route_collinearity_error_rad=float(np.max(np.abs(applied-base*actual))))

    def run(self,points):
        outcome=super().run(points)
        outcome.update(**route_metadata(),evaluation_scope=
            "Planner provides checked joint-route direction; network learns progress fraction without a positive floor. "
            "Other contact/withdraw stages use a learned seven-joint head; stationary holds are supervised. "
            +("Analytic normal-force assistance during loaded CONTACT/HOLD. " if self.contact_mode=="hybrid_normal" else "Pure BC contact action. ")
            +"Only rejected fault recovery may use analytic motion. No learned route planning, online training, audio, or clinical result.")
        return outcome
