"""Learned desired normal speed; the checked curve/executor is explicit."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .observations import ACTION_LIMIT_M_S, ACTION_INWARD_LIMIT_M_S, ACTION_SCALING, FRAME_FEATURES, HISTORY_STEPS
from .dataset import EXECUTOR_CODE_FILES, SAMPLE_SEMANTICS, SAMPLE_WEIGHTING, verify_executor_inventory

CODE_FILES=tuple(dict.fromkeys(("observations.py","dataset.py","policy.py","train.py","conditions.json")+EXECUTOR_CODE_FILES))


class NormalVelocityPolicy(nn.Module):
    def __init__(self, hidden=(64,64), mean=None, std=None):
        super().__init__()
        self.input_dim=HISTORY_STEPS*len(FRAME_FEATURES); self.hidden=tuple(hidden)
        if not self.hidden or any(type(width) is not int or width<=0 for width in self.hidden):
            raise ValueError("invalid network dimensions")
        mean=np.zeros(self.input_dim,np.float32) if mean is None else np.asarray(mean,np.float32)
        std=np.ones(self.input_dim,np.float32) if std is None else np.asarray(std,np.float32)
        if mean.shape!=(self.input_dim,) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
            raise ValueError("invalid training scaler")
        self.register_buffer("input_mean",torch.from_numpy(mean.copy()))
        self.register_buffer("input_std",torch.from_numpy(std.copy()))
        layers=[]; previous=self.input_dim
        for width in self.hidden:
            layers.extend((nn.Linear(previous,width),nn.Tanh())); previous=width
        layers.append(nn.Linear(previous,1)); self.network=nn.Sequential(*layers)

    def forward(self, history):
        values=history.reshape(history.shape[0],-1)
        if values.shape[1]!=self.input_dim or not torch.isfinite(values).all():
            raise ValueError("invalid policy input")
        speed=values[:,-len(FRAME_FEATURES)+FRAME_FEATURES.index("speed_scale")]
        if torch.any((speed<.7)|(speed>1.)):
            raise ValueError("invalid policy speed condition")
        unit=torch.tanh(self.network((values-self.input_mean)/self.input_std))
        # Continuous at zero; the network parameterization itself expresses the
        # asymmetric action space, without post-prediction hard clipping.
        action=torch.where(unit<0,ACTION_INWARD_LIMIT_M_S*speed[:,None]*unit,ACTION_LIMIT_M_S*unit)
        if not torch.isfinite(action).all():
            raise ValueError("nonfinite policy output")
        return action


@dataclass
class FrozenActor:
    policy: NormalVelocityPolicy
    metadata: dict
    device: object

    def predict(self, history):
        values=np.asarray(history,np.float32)
        if values.shape!=(HISTORY_STEPS,len(FRAME_FEATURES)) or not np.isfinite(values).all():
            raise ValueError("invalid frozen actor history")
        with torch.inference_mode():
            return float(self.policy(torch.as_tensor(values[None],device=self.device))[0,0].cpu())


def load_policy(directory, expected_manifest_sha256, expected_contract_sha256, device="cpu"):
    """Verify source, model and executor identity before frozen online inference."""
    directory=Path(directory); data=(directory/"run_manifest.json").read_bytes()
    digest=lambda value:hashlib.sha256(value).hexdigest()
    if digest(data)!=expected_manifest_sha256:
        raise ValueError("model manifest hash mismatch")
    manifest=json.loads(data); checkpoint=(directory/"policy.pt").read_bytes()
    if (manifest.get("schema")!="auscultation_scalar_training_v1"
            or manifest.get("contract_sha256")!=expected_contract_sha256
            or manifest.get("checkpoint_sha256")!=digest(checkpoint)):
        raise ValueError("model/executor contract mismatch")
    if (manifest.get("action_scaling")!=ACTION_SCALING
            or manifest.get("action_limit_m_s")!=ACTION_LIMIT_M_S
            or manifest.get("action_inward_limit_m_s")!=ACTION_INWARD_LIMIT_M_S):
        raise ValueError("model action scaling or bounds mismatch")
    if manifest.get("sample_semantics")!=SAMPLE_SEMANTICS or manifest.get("sample_weighting")!=SAMPLE_WEIGHTING:
        raise ValueError("model issued-decision sample semantics mismatch")
    for name in CODE_FILES:
        if digest(Path(__file__).with_name(name).read_bytes())!=manifest.get("code_sha256",{}).get(name):
            raise ValueError("policy source hash changed: "+name)
    inventory=verify_executor_inventory(manifest.get("executor_code_sha256"))
    if any(manifest["code_sha256"].get(name)!=digest for name,digest in inventory.items()):
        raise ValueError("model source differs from qualified executor inventory")
    saved=torch.load(io.BytesIO(checkpoint),map_location="cpu",weights_only=True)
    if (saved.get("schema")!="auscultation_scalar_policy_v1" or saved.get("frame_features")!=list(FRAME_FEATURES)
            or saved.get("history_steps")!=HISTORY_STEPS or saved.get("action_limit_m_s")!=ACTION_LIMIT_M_S
            or saved.get("action_inward_limit_m_s")!=ACTION_INWARD_LIMIT_M_S
            or saved.get("action_scaling")!=ACTION_SCALING
            or saved.get("sample_semantics")!=SAMPLE_SEMANTICS or saved.get("sample_weighting")!=SAMPLE_WEIGHTING
            or saved.get("dataset_manifest_sha256")!=manifest.get("dataset_manifest_sha256")):
        raise ValueError("checkpoint observation/action mismatch")
    if saved.get("executor_code_sha256")!=inventory:
        raise ValueError("checkpoint qualified executor source mismatch")
    model=NormalVelocityPolicy(hidden=saved["hidden"])
    state=saved.get("state_dict",{})
    if not state or any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("invalid checkpoint tensors")
    model.load_state_dict(state,strict=True)
    if torch.any(model.input_std<=0):
        raise ValueError("invalid checkpoint scaler")
    for key,tensor in (("mean",model.input_mean),("std",model.input_std)):
        if not np.allclose(tensor.numpy(),manifest["scaler"][key],atol=1e-7,rtol=0):
            raise ValueError("checkpoint scaler mismatch")
    from .train import resolve_device
    selected=resolve_device(device); model.to(selected).eval().requires_grad_(False)
    return FrozenActor(model,manifest,selected)
