"""NumPy/SciPy kinematics of the checked-in OpenArm left-arm URDF.

Every pose and Jacobian is expressed in ``openarm_left_link0``. The stage's
world-from-base transform must be supplied separately; this module makes no
assumption about the physical robot or Isaac stage placement. Joint positions
are radians, distances are metres, and geometric Jacobians are [linear; angular].
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tool_frames import wrist_from_diaphragm


DEFAULT_URDF_PATH = Path(__file__).with_name("openarm_v10_follower_stetho.urdf")


def _vector(value, length, name):
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {length} finite numbers") from error
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must have shape ({length},) and contain finite numbers")
    return vector


@dataclass(frozen=True)
class _Joint:
    name: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class IKResult:
    """Bounded numerical IK outcome, not a collision or motion authorization.

    ``success`` requires both physical error tolerances, regardless of the
    optimizer's termination flag. Failed searches still return their best
    bounded q; they are not proofs that a pose is mathematically unreachable.
    """

    q: np.ndarray
    success: bool
    position_error_m: float
    orientation_error_rad: float
    attempts: int
    nfev: int
    message: str


class OpenArmKinematics:
    """Seven-joint left arm from ``left_link0`` to wrist ``left_link7``.

    ``fk(q)`` and ``tcp(q, compression)`` return 4 x 4 poses. ``tcp`` uses the
    diaphragm surface. ``link_transforms(q)`` includes the base and all seven
    moving links. Forward methods reject joint values outside the URDF limits.
    """

    base_link = "openarm_left_link0"
    wrist_link = "openarm_left_link7"

    def __init__(self, urdf_path=DEFAULT_URDF_PATH):
        self.urdf_path = Path(urdf_path)
        try:
            root = ET.parse(self.urdf_path).getroot()
        except ET.ParseError as error:
            raise ValueError(f"Invalid URDF XML: {error}") from error
        by_child = {}
        for element in root.findall("joint"):
            child = element.find("child")
            if child is not None:
                child_name = child.get("link")
                if child_name in by_child:
                    raise ValueError(f"URDF has duplicate parent joints for {child_name}")
                by_child[child_name] = element
        reverse_chain, visited = [], set()
        child_name = self.wrist_link
        while child_name != self.base_link:
            if child_name in visited or child_name not in by_child:
                raise ValueError("URDF has no valid left_link0 to left_link7 chain")
            visited.add(child_name)
            element = by_child[child_name]
            if element.get("type") != "revolute":
                raise ValueError("The left arm chain must contain seven revolute joints")
            origin_element = element.find("origin")
            xyz = _vector((origin_element.get("xyz", "0 0 0") if origin_element is not None else "0 0 0").split(), 3, "URDF origin xyz")
            rpy = _vector((origin_element.get("rpy", "0 0 0") if origin_element is not None else "0 0 0").split(), 3, "URDF origin rpy")
            axis_element = element.find("axis")
            axis = _vector((axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0").split(), 3, "URDF joint axis")
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                raise ValueError("URDF joint axis must be nonzero")
            limit = element.find("limit")
            try:
                lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError("URDF revolute joint requires finite lower/upper limits") from error
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError("URDF joint limits must be finite with lower < upper")
            origin = np.eye(4)
            origin[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            origin[:3, 3] = xyz
            reverse_chain.append(_Joint(element.get("name"), child_name, origin, axis / norm, lower, upper))
            parent = element.find("parent")
            child_name = parent.get("link") if parent is not None else None
        if len(reverse_chain) != 7:
            raise ValueError("The left arm chain must contain exactly seven revolute joints")
        self._joints = tuple(reversed(reverse_chain))
        self.joint_names = tuple(joint.name for joint in self._joints)
        self.lower_limits = np.array([joint.lower for joint in self._joints])
        self.upper_limits = np.array([joint.upper for joint in self._joints])
        self.lower_limits.setflags(write=False)
        self.upper_limits.setflags(write=False)

    def _q(self, q):
        q = _vector(q, 7, "q")
        if np.any(q < self.lower_limits - 1e-10) or np.any(q > self.upper_limits + 1e-10):
            raise ValueError("q exceeds the URDF joint limits")
        return q

    def _chain(self, q):
        transform = np.eye(4)
        links = {self.base_link: transform.copy()}
        origins, axes = [], []
        for joint, angle in zip(self._joints, q):
            transform = transform @ joint.origin
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, :3] @ joint.axis)
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(joint.axis * angle).as_matrix()
            transform = transform @ motion
            links[joint.child] = transform.copy()
        return links, np.asarray(origins), np.asarray(axes)

    def link_transforms(self, q):
        """Return base-from-link poses keyed by URDF link names, in chain order."""
        return self._chain(self._q(q))[0]

    def fk(self, q):
        """Return the wrist pose in the left base frame."""
        return self.link_transforms(q)[self.wrist_link]

    def tcp(self, q, compression=0.0):
        """Return the diaphragm pose, with positive compression retracting it."""
        return self.fk(q) @ wrist_from_diaphragm(compression)

    def jacobian(self, q, compression=0.0):
        """Return the diaphragm geometric Jacobian (6 x 7) in the base frame."""
        links, origins, axes = self._chain(self._q(q))
        position = (links[self.wrist_link] @ wrist_from_diaphragm(compression))[:3, 3]
        return np.vstack((np.cross(axes, position - origins).T, axes.T))

    def solve_ik(
        self, position_base, approach_axis_base=None, seed=None, *,
        orientation_base=None, compression=0.0, position_tolerance=1e-4,
        orientation_tolerance=1e-3, max_starts=12, max_nfev=300, posture_weight=0.,
    ):
        """Solve bounded, deterministic multiseed diaphragm pose IK.

        ``position_base`` is the diaphragm position in metres. The default
        orientation constraint aligns tool +Z with ``approach_axis_base`` and
        leaves roll free; use the *negative* outward surface normal for contact.
        Alternatively supply ``orientation_base`` as a proper 3 x 3 tool rotation
        for full pose IK. If both are supplied, they must agree on tool +Z.

        Each start is bounded by ``max_nfev``. Seed, midpoint, and deterministic
        random starts are attempted until both tolerances pass or ``max_starts``
        is exhausted. Tolerances are metres and radians respectively. No
        collision, joint velocity, or contact-force constraints are solved here.
        """
        position = _vector(position_base, 3, "position_base")
        if not np.isscalar(posture_weight) or not np.isfinite(posture_weight) or posture_weight < 0:
            raise ValueError('posture_weight must be finite and nonnegative')
        tcp_offset = wrist_from_diaphragm(compression)
        for value, name in ((position_tolerance, "position_tolerance"), (orientation_tolerance, "orientation_tolerance")):
            if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive scalar")
        for value, name in ((max_starts, "max_starts"), (max_nfev, "max_nfev")):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        axis = None
        if approach_axis_base is not None:
            axis = _vector(approach_axis_base, 3, "approach_axis_base")
            norm = np.linalg.norm(axis)
            if not np.isfinite(norm) or norm < 1e-12:
                raise ValueError("approach_axis_base must be nonzero")
            axis = axis / norm
        desired_rotation = None
        if orientation_base is not None:
            desired_rotation = np.asarray(orientation_base, dtype=float)
            if (desired_rotation.shape != (3, 3) or not np.isfinite(desired_rotation).all()
                    or not np.allclose(desired_rotation.T @ desired_rotation, np.eye(3), atol=1e-6, rtol=0)
                    or not np.isclose(np.linalg.det(desired_rotation), 1.0, atol=1e-6, rtol=0)):
                raise ValueError("orientation_base must be a proper 3 x 3 rotation")
            if axis is not None and not np.allclose(axis, desired_rotation[:, 2], atol=1e-6, rtol=0):
                raise ValueError("approach_axis_base must agree with orientation_base +Z")
        elif axis is None:
            raise ValueError("Provide approach_axis_base or orientation_base")

        midpoint = (self.lower_limits + self.upper_limits) / 2
        starts = [self._q(seed).copy()] if seed is not None else [midpoint]
        if seed is not None:
            starts.append(midpoint)
        random = np.random.default_rng(0)
        while len(starts) < max_starts:
            starts.append(random.uniform(self.lower_limits, self.upper_limits))

        # A 0.1 m orientation weight gives comparable scales for arm geometry.
        # Vector difference (not cross product alone) distinguishes opposite axes.
        orientation_weight = 0.1
        cached_q, cached_pose, cached_origins, cached_axes = None, None, None, None

        def evaluate(q):
            nonlocal cached_q, cached_pose, cached_origins, cached_axes
            if cached_q is None or not np.array_equal(q, cached_q):
                links, cached_origins, cached_axes = self._chain(q)
                cached_pose = links[self.wrist_link] @ tcp_offset
                cached_q = q.copy()
            return cached_pose, cached_origins, cached_axes

        def residual(q):
            pose, _, _ = evaluate(q)
            if desired_rotation is None:
                orientation_residual = pose[:3, 2] - axis
            else:
                orientation_residual = Rotation.from_matrix(pose[:3, :3] @ desired_rotation.T).as_rotvec()
            result = np.concatenate((pose[:3, 3] - position, orientation_weight * orientation_residual))
            return np.r_[result, posture_weight*(q-posture_reference)] if posture_weight else result

        def axis_jacobian(q):
            pose, origins, axes = evaluate(q)
            result = np.vstack((
                np.cross(axes, pose[:3, 3] - origins).T,
                orientation_weight * np.cross(axes, pose[:3, 2]).T,
            ))
            return np.vstack((result,posture_weight*np.eye(7))) if posture_weight else result

        best = None
        total_evaluations = 0
        for attempt, initial in enumerate(starts[:max_starts], start=1):
            posture_reference = initial.copy()
            solution = least_squares(
                residual, initial, jac=axis_jacobian if desired_rotation is None else "3-point",
                bounds=(self.lower_limits, self.upper_limits), max_nfev=max_nfev,
                ftol=1e-11, xtol=1e-11, gtol=1e-11,
            )
            total_evaluations += solution.nfev
            pose, _, _ = evaluate(solution.x)
            position_error = float(np.linalg.norm(pose[:3, 3] - position))
            if desired_rotation is None:
                orientation_error = float(np.arctan2(
                    np.linalg.norm(np.cross(pose[:3, 2], axis)),
                    np.clip(np.dot(pose[:3, 2], axis), -1.0, 1.0),
                ))
            else:
                orientation_error = float(Rotation.from_matrix(pose[:3, :3] @ desired_rotation.T).magnitude())
            score = max(position_error / position_tolerance, orientation_error / orientation_tolerance)
            if best is None or score < best[0]:
                best = (score, solution.x.copy(), position_error, orientation_error)
            if position_error <= position_tolerance and orientation_error <= orientation_tolerance:
                return IKResult(solution.x.copy(), True, position_error, orientation_error,
                                attempt, total_evaluations, "Pose tolerances satisfied")
        return IKResult(best[1], False, best[2], best[3], attempt, total_evaluations,
                        "No candidate met pose tolerances within the configured search")
