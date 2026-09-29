"""One goal-conditioned bounded joint-increment actor for every target.

The observable phase supervisor and safety/recovery checks remain external.
No contact-ground-truth labels, acoustic features, or rewards enter this actor.
"""
import numpy as np
import torch
from torch import nn

from .dataset import ACTION_LIMIT_RAD


class BoundedJointPolicy(nn.Module):
    def __init__(self, input_dim, hidden=(128, 128), mean=None, std=None):
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0 or not hidden or any(type(n) is not int or n <= 0 for n in hidden):
            raise ValueError("positive input and hidden dimensions required")
        mean = np.zeros(input_dim, np.float32) if mean is None else np.asarray(mean, np.float32)
        std = np.ones(input_dim, np.float32) if std is None else np.asarray(std, np.float32)
        if mean.shape != (input_dim,) or std.shape != (input_dim,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("invalid actor scaler")
        self.input_dim, self.hidden = input_dim, tuple(hidden)
        self.register_buffer("input_mean", torch.from_numpy(mean.copy()))
        self.register_buffer("input_std", torch.from_numpy(std.copy()))
        layers = []
        previous = input_dim
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, 7))
        self.network = nn.Sequential(*layers)

    def forward(self, observations):
        # Accept N x H x F offline histories or N x (H*F) flattened histories.
        values = observations.reshape(observations.shape[0], -1)
        if values.shape[1] != self.input_dim:
            raise ValueError("actor observation dimension mismatch")
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite actor observation")
        action = ACTION_LIMIT_RAD * torch.tanh(self.network((values - self.input_mean) / self.input_std))
        if not torch.isfinite(action).all():
            raise ValueError('nonfinite actor prediction')
        return action
