"""Frozen scalar policy over the common executor; no teacher motion fallback."""
import io
import json
from pathlib import Path

import numpy as np
import torch

from .controller import ExpertController
from .dataset import sha, SAMPLE_SEMANTICS, SAMPLE_WEIGHTING
from .observations import ACTION_LIMIT_M_S, ACTION_INWARD_LIMIT_M_S, ACTION_SCALING, FRAME_FEATURES, HISTORY_STEPS, History
from .policy import CODE_FILES, FrozenActor, NormalVelocityPolicy
from .provenance import STUDENT_CLASS
from .train import resolve_device


class NegativeActor(FrozenActor):
    def predict(self, history):
        # The zero control deliberately never evaluates a neural output.
        if self.metadata["mode"]=="zero":
            values=np.asarray(history,np.float32)
            if values.shape!=(HISTORY_STEPS,len(FRAME_FEATURES)) or not np.isfinite(values).all():
                raise ValueError("invalid zero actor history")
            return 0.
        return super().predict(history)


def prepare_negative_control(output, mode, contract_sha256, seed=0):
    """Save the actual random weights even when the zero baseline ignores them."""
    if mode not in ("zero","untrained") or len(contract_sha256)!=64 or type(seed) is not int:
        raise ValueError("invalid negative control specification")
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); policy=NormalVelocityPolicy()
    checkpoint={"schema":"auscultation_scalar_negative_policy_v1","state_dict":policy.state_dict(),
                "hidden":list(policy.hidden),"history_steps":HISTORY_STEPS,"frame_features":list(FRAME_FEATURES),
                "action_scaling":ACTION_SCALING,
                "action_limit_m_s":ACTION_LIMIT_M_S,"action_inward_limit_m_s":ACTION_INWARD_LIMIT_M_S,"mode":mode}
    torch.save(checkpoint,output/"policy.pt")
    manifest={"schema":"auscultation_scalar_negative_control_v1","mode":mode,"seed":seed,
              "weight_updates":False,"trained":False,"contract_sha256":contract_sha256,
              "action_limit_m_s":ACTION_LIMIT_M_S,"action_inward_limit_m_s":ACTION_INWARD_LIMIT_M_S,
              "action_scaling":ACTION_SCALING,
              "checkpoint_sha256":sha((output/"policy.pt").read_bytes()),
              "code_sha256":{name:sha(Path(__file__).with_name(name).read_bytes()) for name in CODE_FILES+("student.py",)},
              "scope":"zero output ignores saved network" if mode=="zero" else "frozen random neural weights; no optimization"}
    (output/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def load_negative_control(directory, expected_manifest_sha256, expected_contract_sha256, mode=None, device="cpu"):
    directory=Path(directory); data=(directory/"run_manifest.json").read_bytes()
    if sha(data)!=expected_manifest_sha256: raise ValueError("negative manifest hash mismatch")
    manifest=json.loads(data); checkpoint=(directory/"policy.pt").read_bytes()
    mode=manifest.get("mode") if mode is None else mode
    if (manifest.get("schema")!="auscultation_scalar_negative_control_v1" or manifest.get("mode")!=mode
            or manifest.get("contract_sha256")!=expected_contract_sha256
            or manifest.get("checkpoint_sha256")!=sha(checkpoint) or manifest.get("trained") is not False):
        raise ValueError("negative control mode or executor contract mismatch")
    if (manifest.get("action_limit_m_s")!=ACTION_LIMIT_M_S
            or manifest.get("action_inward_limit_m_s")!=ACTION_INWARD_LIMIT_M_S
            or manifest.get("action_scaling")!=ACTION_SCALING):
        raise ValueError("negative control action scaling or bounds mismatch")
    for name in CODE_FILES+("student.py",):
        if sha(Path(__file__).with_name(name).read_bytes())!=manifest.get("code_sha256",{}).get(name):
            raise ValueError("negative control code changed")
    saved=torch.load(io.BytesIO(checkpoint),map_location="cpu",weights_only=True)
    if (saved.get("schema")!="auscultation_scalar_negative_policy_v1" or saved.get("mode")!=mode
            or saved.get("frame_features")!=list(FRAME_FEATURES) or saved.get("history_steps")!=HISTORY_STEPS
            or saved.get("action_limit_m_s")!=ACTION_LIMIT_M_S
            or saved.get("action_scaling")!=ACTION_SCALING
            or saved.get("action_inward_limit_m_s")!=ACTION_INWARD_LIMIT_M_S):
        raise ValueError("negative checkpoint interface mismatch")
    model=NormalVelocityPolicy(hidden=saved["hidden"]); model.load_state_dict(saved["state_dict"],strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("invalid negative model tensors")
    selected=resolve_device(device); model.to(selected).eval().requires_grad_(False)
    return NegativeActor(model,manifest,selected)


class StudentController(ExpertController):
    def __init__(self,*args,mode="policy",actor=None,**kwargs):
        if mode not in ("policy","zero","untrained") or actor is None:
            raise ValueError("student requires explicit mode and frozen actor")
        expected=("auscultation_scalar_training_v1" if mode=="policy" else "auscultation_scalar_negative_control_v1")
        if actor.metadata.get("schema")!=expected or (mode!="policy" and actor.metadata.get("mode")!=mode):
            raise ValueError("actor does not match requested student mode")
        if (actor.metadata.get("action_scaling")!=ACTION_SCALING
                or actor.metadata.get("action_limit_m_s")!=ACTION_LIMIT_M_S
                or actor.metadata.get("action_inward_limit_m_s")!=ACTION_INWARD_LIMIT_M_S):
            raise ValueError("actor action scaling or bounds mismatch")
        if mode=="policy" and (actor.metadata.get("sample_semantics")!=SAMPLE_SEMANTICS
                               or actor.metadata.get("sample_weighting")!=SAMPLE_WEIGHTING):
            raise ValueError("actor issued-decision sample semantics mismatch")
        if "action_provider" in kwargs:
            raise ValueError("student action provider cannot be replaced")
        self.actor=actor; self.mode=mode; self.actor_history=History()
        self.policy_record=None; self.policy_decisions=0
        super().__init__(*args,action_provider=self._policy_action,**kwargs)

    def _policy_action(self,controller,conditioned_state):
        # condition processing and current_skill_observation are owned by the
        # common controller; read exactly that snapshot once at each decision.
        history=self.actor_history.push(self.current_skill_observation,self.target.name)
        prediction=float(self.actor.predict(history))
        if (not np.isfinite(prediction) or prediction<-ACTION_INWARD_LIMIT_M_S*self.speed_scale-1e-9
                or prediction>ACTION_LIMIT_M_S+1e-9):
            raise ValueError("invalid learned desired normal velocity")
        self.policy_decisions+=1
        self.policy_record={"policy_mode":self.mode,"policy_decision_index":self.policy_decisions,
                            "policy_input_sha256":sha(history.tobytes()),
                            "policy_checkpoint_sha256":self.actor.metadata["checkpoint_sha256"],
                            "policy_action_scaling":ACTION_SCALING,
                            "network_prediction_m_s":None if self.mode=="zero" else prediction,
                            "policy_prediction_m_s":prediction,"expert_normal_rescue_used":False}
        return prediction

    def _block(self,*args,**kwargs):
        before=len(self.control_rows)
        try:
            return super()._block(*args,**kwargs)
        finally:
            if self._contact_active and len(self.control_rows)>before:
                if self.policy_record is None:
                    raise ValueError("student contact block lacks a policy decision")
                self.control_rows[-1].update(self.policy_record)

    def run(self,points):
        self.actor_history.reset(); self.policy_record=None; self.policy_decisions=0
        outcome=super().run(points)
        recovery=any(row.get("recovery") for row in self.control_rows)
        outcome.update(executed_controller_class=STUDENT_CLASS,candidate_method=STUDENT_CLASS,
                       dataset_role="student_evaluation",expert_demonstration=False,
                       controller_source="learned_normal_policy",policy_mode=self.mode,
                       policy=self.actor.metadata,policy_decisions=self.policy_decisions,
                       policy_checkpoint_sha256=self.actor.metadata["checkpoint_sha256"],
                       expert_fallback_used=False,expert_normal_rescue_used=False,
                       recovery_used=recovery,failure_latched=not outcome.get("success",False),
                       learned_task_success=False,assisted_system_success=False,
                       success_authority="independent simulation quality plus frozen-policy action attribution",
                       learning_scope="single-point normal speed on checked curve; planner owns transfers; state inputs; no RGB-D encoder")
        return outcome
