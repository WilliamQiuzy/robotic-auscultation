"""Planner-provided route direction with learned progression, plus a joint head.

The network observes exactly the v2 four-frame history. Route direction is
separate, explicit planner context for the action mapping; it is not learned.
No clock, trajectory index, positive progress floor, or analytic rescue exists.
"""
import torch
from torch import nn

from .dataset_v2 import ACTION_LIMIT_RAD, FRAME_FEATURES, HISTORY_STEPS, JOINT_SUBPHASES
from .policy import BoundedJointPolicy

MOTION_PARAMETERIZATION = "checked_route_learned_progress_v3"
ROUTE_DIRECTION_SOURCE = "checked planner waypoint minus previous command"


def architecture(hidden=(128,128)):
    return {"type":"planner_route_progress_and_joint_MLP", "input_dim":HISTORY_STEPS*len(FRAME_FEATURES),
        "hidden":list(hidden), "output_dim":8,
        "heads":{"route_progress":"sigmoid scalar, no positive floor", "joint":"7 tanh times 0.0125 rad"},
        "action_mapping":MOTION_PARAMETERIZATION}


def route_base_step(remaining, speed):
    if remaining.ndim != 2 or remaining.shape[1] != 7 or speed.shape != (len(remaining),):
        raise ValueError("invalid route context shape")
    if not torch.isfinite(remaining).all() or not torch.isfinite(speed).all() or torch.any((speed<.7)|(speed>1.)):
        raise ValueError("invalid route context or speed")
    maximum=torch.amax(torch.abs(remaining),dim=1)
    scale=torch.minimum(torch.ones_like(maximum),(.25*.05*speed)/torch.clamp(maximum,min=1e-30))
    return remaining*scale[:,None]


class RouteProgressPolicy(BoundedJointPolicy):
    def __init__(self,hidden=(128,128),mean=None,std=None):
        super().__init__(HISTORY_STEPS*len(FRAME_FEATURES),hidden=hidden,mean=mean,std=std)
        self.network[-1]=nn.Linear(hidden[-1],8)

    def components(self,observations):
        values=observations.reshape(observations.shape[0],-1)
        if values.shape[1] != self.input_dim or not torch.isfinite(values).all():
            raise ValueError("invalid v3 actor observation")
        latest=values[:,-len(FRAME_FEATURES):]
        validity=latest[:,FRAME_FEATURES.index("local_waypoint_valid")]
        stages=latest[:,[FRAME_FEATURES.index("subphase_"+name) for name in JOINT_SUBPHASES]]
        speed=latest[:,-1]
        if (torch.any((validity!=0)&(validity!=1)) or torch.any((stages!=0)&(stages!=1))
                or torch.any(stages.sum(1)!=validity) or torch.any((speed<.7)|(speed>1.))):
            raise ValueError("invalid route validity, supervised stage, or speed")
        logits=self.network((values-self.input_mean)/self.input_std)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite v3 actor prediction")
        return {"progress_fraction":torch.sigmoid(logits[:,0]),
            "joint_action_rad":ACTION_LIMIT_RAD*torch.tanh(logits[:,1:]),
            "route_valid":validity.bool(), "speed_scale":speed}

    def forward(self,observations,route_remaining):
        parts=self.components(observations)
        remaining=torch.as_tensor(route_remaining,dtype=observations.dtype,device=observations.device)
        base=route_base_step(remaining,parts["speed_scale"])
        if torch.any(remaining[~parts["route_valid"]]!=0):
            raise ValueError("non-route stages require zero route context")
        return torch.where(parts["route_valid"][:,None],base*parts["progress_fraction"][:,None],parts["joint_action_rad"])
