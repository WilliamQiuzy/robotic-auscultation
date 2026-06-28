"""Per-point constant-velocity Kalman filter in the canonical frame.

Each auscultation point gets one ``PointEKF``. State is 3D position + velocity in the
**canonical patient frame** (where the anatomical target is near-static, so the filter is
essentially a robust multi-measurement fuser with a slow-drift allowance).

Measurements arrive from up to four independent channels (detector / flow / ICP / IUV); each
is a 3D position + isotropic covariance. ``update`` applies **Mahalanobis gating** so a wrong
detection or a bad ICP step is rejected instead of corrupting the track.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# chi-square 0.997 quantile, 3 dof (gate ~3σ) — measurements beyond this are rejected.
_CHI2_GATE_3DOF = 14.16


@dataclass
class EKFConfig:
    q_pos: float = 1e-4      # position process noise [m^2] per step (allows slow drift)
    q_vel: float = 1e-3      # velocity process noise [(m/s)^2] per step
    init_pos_var: float = 1e-2   # initial position variance [m^2]
    init_vel_var: float = 1e-1   # initial velocity variance
    gate: float = _CHI2_GATE_3DOF
    max_speed: float = 0.5   # m/s clamp on estimated velocity (sanity)


class PointEKF:
    """Constant-velocity EKF for a single 3D point. Units: metres, seconds."""

    def __init__(self, p0: np.ndarray, cfg: Optional[EKFConfig] = None) -> None:
        self.cfg = cfg or EKFConfig()
        self.x = np.zeros(6)
        self.x[:3] = np.asarray(p0, dtype=np.float64).reshape(3)
        self.P = np.diag([self.cfg.init_pos_var] * 3 + [self.cfg.init_vel_var] * 3)
        self.n_updates = 0
        self.n_rejected = 0
        self.misses = 0  # consecutive predict-only steps (no accepted measurement)

    # -- prediction -------------------------------------------------------
    def predict(self, dt: float) -> None:
        dt = float(dt)
        if not np.isfinite(dt):
            dt = 1e-3                              # NaN/inf timestamp (dropped/dup frame) → minimal step
        dt = min(max(dt, 1e-3), 1.0)              # clamp to a sane band so a huge dt can't blow up F@x/P
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        self.x = F @ self.x
        # clamp velocity to a sane band
        sp = np.linalg.norm(self.x[3:])
        if sp > self.cfg.max_speed:
            self.x[3:] *= self.cfg.max_speed / sp
        Q = np.diag([self.cfg.q_pos] * 3 + [self.cfg.q_vel] * 3)
        self.P = F @ self.P @ F.T + Q

    # -- measurement update (gated) --------------------------------------
    def update(self, z: np.ndarray, var: float) -> bool:
        """Fuse a 3D position measurement with isotropic variance ``var`` [m^2].

        Returns True if accepted (passed the Mahalanobis gate), False if rejected.
        """
        z = np.asarray(z, dtype=np.float64).reshape(3)
        if not np.isfinite(z).all() or not np.isfinite(var) or var <= 0:
            return False
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.eye(3) * var
        y = z - H @ self.x                      # innovation
        S = H @ self.P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        md2 = float(y @ Sinv @ y)               # squared Mahalanobis distance
        if md2 > self.cfg.gate:
            self.n_rejected += 1
            return False
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.n_updates += 1
        self.misses = 0
        return True

    def mark_miss(self) -> None:
        self.misses += 1

    # -- readout ----------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    @property
    def position_std(self) -> float:
        """Scalar position uncertainty [m] (trace of the position covariance block)."""
        return float(np.sqrt(np.trace(self.P[:3, :3]) / 3.0))
