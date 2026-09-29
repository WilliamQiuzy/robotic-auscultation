"""Explicit low-elbow posture constraints, in the existing world frame.

These geometric gates supplement collision and force checks. They do not
certify motor torque, cable routing, or real hardware reliability.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PostureLimits:
    max_elbow_z_m: float
    joint_lower_rad: tuple
    joint_upper_rad: tuple
    elbow_link: str = "openarm_left_link4"
    observed_joint_tolerance_rad: float = .005
    max_elbow_above_shoulder_m: float = 0.
    min_contact_elbow_below_wrist_m: float = 0.

    @classmethod
    def from_dict(cls, data, kin):
        if data.get("schema") != "low_elbow_posture_v1":
            raise ValueError("unsupported posture constraint schema")
        lower = np.asarray(data["joint_lower_rad"], float)
        upper = np.asarray(data["joint_upper_rad"], float)
        height = float(data["max_elbow_z_m"])
        tolerance = float(data.get("observed_joint_tolerance_rad", .005))
        shoulder_allowance = float(data.get("max_elbow_above_shoulder_m", 0.))
        contact_gap = float(data.get("min_contact_elbow_below_wrist_m", 0.))
        if (lower.shape != (7,) or upper.shape != (7,)
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or not np.isfinite(height) or not np.isfinite(tolerance)
                or not np.isfinite(shoulder_allowance) or not 0 <= shoulder_allowance <= .075
                or not np.isfinite(contact_gap) or contact_gap < 0
                or not 0 <= tolerance <= .005 or np.any(lower >= upper)
                or np.any(lower < kin.lower_limits) or np.any(upper > kin.upper_limits)):
            raise ValueError("invalid posture bounds")
        link = data.get("elbow_link", "openarm_left_link4")
        if link != "openarm_left_link4":
            raise ValueError("unsupported elbow landmark")
        return cls(height, tuple(lower), tuple(upper), link, tolerance, shoulder_allowance, contact_gap)

    def to_dict(self):
        return {"schema": "low_elbow_posture_v1", "elbow_link": self.elbow_link,
                "max_elbow_z_m": self.max_elbow_z_m,
                "joint_lower_rad": list(self.joint_lower_rad),
                "joint_upper_rad": list(self.joint_upper_rad),
                "observed_joint_tolerance_rad": self.observed_joint_tolerance_rad,
                "max_elbow_above_shoulder_m": self.max_elbow_above_shoulder_m,
                "min_contact_elbow_below_wrist_m": self.min_contact_elbow_below_wrist_m}

    def evaluate(self, kin, base, q, *, observed=False, contact=False):
        q, base = np.asarray(q, float), np.asarray(base, float)
        if (q.shape != (7,) or not np.isfinite(q).all() or base.shape != (4, 4)
                or not np.isfinite(base).all()):
            raise ValueError("invalid posture observation")
        reasons = []
        wrist_height = below_wrist = None
        margin = float(np.min(np.r_[q - self.joint_lower_rad, self.joint_upper_rad - q]))
        tolerance = self.observed_joint_tolerance_rad if observed else 0.
        if margin < -tolerance - 1e-9:
            reasons.append("posture_joint_envelope")
        if np.any(q < kin.lower_limits-tolerance-1e-9) or np.any(q > kin.upper_limits+tolerance+1e-9):
            height = None
            reasons.append("urdf_joint_limits")
        else:
            # Measured joints can slightly cross the ideal joint stop in the
            # physics solver. Preserve the actual angles in FK; never clip them.
            transforms = kin._chain(q)[0]
            elbow = (base @ transforms[self.elbow_link])[:3, 3]
            height = float(elbow[2])
            if height > self.max_elbow_z_m + 1e-9:
                reasons.append("elbow_height")
            if contact:
                wrist_height = float((base @ transforms["openarm_left_link7"])[2, 3])
                below_wrist = wrist_height - height
                if below_wrist < self.min_contact_elbow_below_wrist_m - 1e-9:
                    reasons.append("contact_elbow_below_wrist")
        return {"clear": not reasons, "posture_violations": reasons,
                "elbow_z_m": height, "posture_joint_margin_rad": margin,
                "wrist_z_m": wrist_height, "elbow_below_wrist_m": below_wrist}


class PostureChecker:
    """Sample both collisions and posture on every route extension."""

    def __init__(self, checker, base, limits):
        self.checker, self.base, self.limits = checker, np.asarray(base, float), limits
        self.kinematics = checker.kinematics

    def check(self, q, **kwargs):
        posture = self.limits.evaluate(self.kinematics, self.base, q)
        if not posture["clear"]:
            return {**posture, "minimum_distance_m": 0.0, "reason": "posture_constraint"}
        result = self.checker.check(q, **kwargs)
        return {**result, **{k: v for k, v in posture.items() if k != "clear"}}

    def edge(self, start, end, **kwargs):
        start, end = np.asarray(start, float), np.asarray(end, float)
        count = max(1, int(np.ceil(np.max(np.abs(end-start)) / .015)))
        minimum = float("inf")
        kwargs.setdefault("clearance", .003)
        for fraction in np.linspace(0., 1., count+1):
            result = self.check(start + fraction*(end-start), **kwargs)
            minimum = min(minimum, result["minimum_distance_m"])
            if not result["clear"]:
                return {**result, "fraction": float(fraction), "samples": count+1}
        return {"clear": True, "minimum_distance_m": minimum, "samples": count+1}
