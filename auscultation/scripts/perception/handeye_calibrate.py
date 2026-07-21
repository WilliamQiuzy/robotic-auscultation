"""Fail-closed Lingbox eye-to-hand commissioning for a fixed Orbbec and OpenArm.

The only commissioned protocol is manual tip-touch against a rigid non-human 3-D
fixture: 10 spatial positions, two wrist orientations per position, then three
independent held-out positions. Every accepted click persists RGB, depth, joints,
FK and timing evidence. A passing run creates a candidate only; it never activates
production calibration.

The companion gravity-comp program sends motor torque and first drives the selected
arm to its home pose. Clear the workspace, support the arm and keep the E-stop ready
before starting that program. This script itself never sends robot commands.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time
from xml.etree import ElementTree

from runtime_paths import activate_runtime
from handeye_provenance import collect_handeye_provenance
from handeye_orbbec_preflight import canonical_json_bytes as preflight_canonical_json_bytes

RUNTIME_ROLE = "linbox_calibration"

import cv2
import numpy as np


def _read_stable_direct_file(path, *, label: str) -> bytes:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("%s must be a direct regular file" % label)
    descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise RuntimeError("%s changed while being read" % label)
    return b"".join(chunks)


def load_orbbec_preflight_bundle(report_path):
    path = Path(report_path)
    body = _read_stable_direct_file(path, label="Orbbec preflight report")
    marker_path = path.with_name(path.name + ".PASS")
    marker_body = _read_stable_direct_file(
        marker_path, label="Orbbec preflight PASS marker",
    )
    try:
        report = json.loads(body.decode("utf-8"))
        marker = json.loads(marker_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Orbbec preflight bundle is not valid JSON") from exc
    if (
        not isinstance(report, dict)
        or body != preflight_canonical_json_bytes(report)
        or not isinstance(marker, dict)
        or marker_body != preflight_canonical_json_bytes(marker)
    ):
        raise RuntimeError("Orbbec preflight bundle is not canonical JSON")
    digest = hashlib.sha256(body).hexdigest()
    if marker != {
        "report": path.name,
        "report_sha256": digest,
        "status": "PASS",
    }:
        raise RuntimeError("Orbbec preflight PASS marker hash is invalid")
    if (
        report.get("schema") != "wam_handeye_orbbec_preflight_v1"
        or report.get("status") != "PASS"
        or report.get("serial") != "CL8326300A0"
        or not isinstance(report.get("intrinsics"), dict)
        or not isinstance(report.get("distortion"), dict)
        or not isinstance(report.get("modes"), dict)
    ):
        raise RuntimeError("Orbbec preflight report is not an accepted exact-camera result")
    return {
        "pass_marker_sha256": hashlib.sha256(marker_body).hexdigest(),
        "report": report,
        "report_sha256": digest,
    }


@dataclass(frozen=True)
class CalibrationIntrinsics:
    """Pinhole model copied from the exact active colour stream profile."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def unproject(self, uv: np.ndarray, z_m: np.ndarray) -> np.ndarray:
        pixels = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        depth = np.asarray(z_m, dtype=np.float64).reshape(-1)
        if pixels.shape != (depth.size, 2):
            raise ValueError("pixel/depth arrays do not match")
        x = (pixels[:, 0] - self.cx) * depth / self.fx
        y = (pixels[:, 1] - self.cy) * depth / self.fy
        return np.stack((x, y, depth), axis=1)


def calibration_intrinsics_from_stream(cam, frame=None) -> CalibrationIntrinsics:
    """Bind unprojection to CameraStream's enabled profile, never the SDK default."""
    raw = getattr(cam, "color_intrinsics", None)
    mode = getattr(cam, "negotiated_mode", None)
    if not isinstance(raw, dict) or not isinstance(mode, dict):
        raise RuntimeError("active color profile intrinsics are unavailable")
    try:
        intrinsics = CalibrationIntrinsics(
            fx=float(raw["fx"]), fy=float(raw["fy"]),
            cx=float(raw["cx"]), cy=float(raw["cy"]),
            width=int(raw["width"]), height=int(raw["height"]),
        )
        mode_width = int(mode["width"])
        mode_height = int(mode["height"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("active color profile intrinsics are invalid") from exc
    numeric = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    if (
        not all(np.isfinite(value) for value in numeric)
        or intrinsics.fx <= 0.0
        or intrinsics.fy <= 0.0
        or intrinsics.width <= 0
        or intrinsics.height <= 0
        or not 0.0 <= intrinsics.cx < intrinsics.width
        or not 0.0 <= intrinsics.cy < intrinsics.height
        or (intrinsics.width, intrinsics.height) != (mode_width, mode_height)
    ):
        raise RuntimeError("active color profile intrinsics do not match the negotiated mode")
    if frame is not None:
        rgb_shape = getattr(getattr(frame, "frame", None), "shape", ())
        depth_shape = getattr(getattr(frame, "depth", None), "shape", ())
        expected = (intrinsics.height, intrinsics.width)
        if tuple(rgb_shape[:2]) != expected or tuple(depth_shape[:2]) != expected:
            raise RuntimeError("active color profile does not match aligned RGB-D frames")
    return intrinsics


def calibration_frame_ready(cam, frame, *, max_age_s: float = 0.1) -> bool:
    """Require a fresh aligned frame and a healthy live capture thread."""
    if frame is None or getattr(frame, "frame", None) is None or getattr(frame, "depth", None) is None:
        return False
    try:
        age_s = float(frame.age_s)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    if (
        getattr(frame, "status", None) != "OK"
        or not np.isfinite(age_s)
        or age_s < 0.0
        or age_s > float(max_age_s)
    ):
        return False
    health_reader = getattr(cam, "delivery_health", None)
    if not callable(health_reader):
        return False
    try:
        health = health_reader()
    except Exception:  # noqa: BLE001 - hardware health calls must fail closed
        return False
    return bool(
        isinstance(health, dict)
        and health.get("status") == "OK"
        and health.get("thread_alive") is True
        and health.get("thread_error") is None
        and health.get("frame_fresh") is True
    )


def openarm_joint_limits(urdf_path, *, arm: str):
    if arm not in ("left", "right"):
        raise ValueError("OpenArm side must be left or right")
    root = ElementTree.parse(os.path.expanduser(os.fspath(urdf_path))).getroot()
    by_name = {joint.get("name"): joint for joint in root.findall("joint")}
    lower, upper = [], []
    for index in range(1, 8):
        name = "openarm_%s_joint%d" % (arm, index)
        joint = by_name.get(name)
        limit = None if joint is None else joint.find("limit")
        try:
            lo = float(limit.attrib["lower"])
            hi = float(limit.attrib["upper"])
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("URDF is missing finite limits for %s" % name) from exc
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise RuntimeError("URDF has invalid limits for %s" % name)
        lower.append(lo)
        upper.append(hi)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def validate_joint_vector(q7, lower, upper) -> np.ndarray:
    joints = np.asarray(q7, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if joints.shape != (7,) or low.shape != (7,) or high.shape != (7,):
        raise ValueError("OpenArm joint vector/limit shape is invalid")
    if not np.all(np.isfinite(joints)) or np.any(joints < low) or np.any(joints > high):
        raise ValueError("OpenArm joint vector is outside the exact URDF limits")
    return joints


# ── OpenArm forward kinematics (joints → TCP pose in arm base) ────────────────
class ArmFK:
    def __init__(self, urdf_path: str, arm: str = "left"):
        from yourdfpy import URDF
        self.robot = URDF.load(os.path.expanduser(urdf_path), load_meshes=False,
                               build_collision_scene_graph=False)
        self.joints = [f"openarm_{arm}_joint{i}" for i in range(1, 8)]
        self.base = f"openarm_{arm}_link0"
        self.tcp = f"openarm_{arm}_hand_tcp"
        self.lower, self.upper = openarm_joint_limits(urdf_path, arm=arm)
        assert all(j in self.robot.actuated_joint_names for j in self.joints), "joint names mismatch"

    def fk(self, q7: np.ndarray) -> np.ndarray:
        """7 joint angles → 4x4 T_tcp_base (TCP pose in the arm base frame)."""
        q7 = validate_joint_vector(q7, self.lower, self.upper)
        cfg = {j: float(a) for j, a in zip(self.joints, q7)}
        self.robot.update_cfg(cfg)
        return np.asarray(self.robot.get_transform(self.tcp, self.base), dtype=np.float64)


def read_latest_joints(
    csv_path: str,
    settle_rows: int = 30,
    vel_eps: float = 2e-3,
    *,
    max_file_age_s: float = 0.5,
    now_s: float | None = None,
):
    """Return (q7, is_still) from the tail of the gravity-comp CSV (t,a1..a7,grip). is_still =
    the arm has been ~stationary over the last `settle_rows` samples (so a clean pose to capture)."""
    if settle_rows < 2 or not np.isfinite(vel_eps) or vel_eps <= 0.0:
        raise ValueError("joint settle window and threshold must be positive")
    expanded = os.path.expanduser(csv_path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(expanded, flags)
    except OSError:
        return None, False
    try:
        stat = os.fstat(descriptor)
        now = time.time() if now_s is None else float(now_s)
        if (
            not np.isfinite(now)
            or now - stat.st_mtime > max_file_age_s
            or stat.st_mtime - now > 1.0
        ):
            return None, False
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    body = b"".join(chunks)
    # The writer can be appending the final record. Ignore only that incomplete
    # suffix; every complete row used below must satisfy the exact contract.
    if not body.endswith(b"\n"):
        body = body.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in body else b""
    try:
        lines = body.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        return None, False
    expected_header = "t,a1,a2,a3,a4,a5,a6,a7,grip"
    if not lines or lines[0] != expected_header or len(lines) - 1 < settle_rows:
        return None, False
    rows = []
    try:
        for line in lines[-settle_rows:]:
            fields = line.split(",")
            if len(fields) != 9:
                return None, False
            row = [float(value) for value in fields]
            if not np.all(np.isfinite(row)):
                return None, False
            rows.append(row)
    except (TypeError, ValueError, OverflowError):
        return None, False
    arr = np.asarray(rows, dtype=np.float64)
    if not np.all(np.diff(arr[:, 0]) > 0.0):
        return None, False
    q_window = arr[:, 1:8]
    q7 = q_window[-1].copy()
    still = float(np.abs(q_window - q7).max()) < vel_eps
    return q7, still


def _direct_existing_path(value, *, label: str) -> Path:
    path = Path(os.path.expanduser(os.fspath(value)))
    if path.is_symlink() or not path.exists():
        raise RuntimeError("%s must be an existing direct path" % label)
    return path.resolve(strict=True)


def verify_gravity_comp_process(
    *,
    arm: str,
    can_interface: str,
    gravity_comp_bin,
    joint_csv,
    urdf,
    proc_root="/proc",
):
    """Prove that exactly one expected gravity-comp writer owns the joint CSV."""
    expected_binary = _direct_existing_path(gravity_comp_bin, label="gravity-comp binary")
    expected_csv = _direct_existing_path(joint_csv, label="joint CSV")
    expected_urdf = _direct_existing_path(urdf, label="OpenArm URDF")
    if not expected_binary.is_file() or not os.access(expected_binary, os.X_OK):
        raise RuntimeError("gravity-comp binary is not executable")
    if arm not in ("left", "right"):
        raise ValueError("gravity-comp arm must be left or right")
    arm_argument = arm + "_arm"
    root = Path(proc_root)
    matches = []
    for process in root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.geteuid():
                continue
            exe_link = process / "exe"
            if exe_link.resolve(strict=True) != expected_binary:
                continue
            argv = [
                os.fsdecode(value)
                for value in (process / "cmdline").read_bytes().split(b"\0")
                if value
            ]
            if len(argv) != 4 or argv[1:3] != [arm_argument, can_interface]:
                continue
            if Path(os.path.expanduser(argv[3])).resolve(strict=True) != expected_urdf:
                continue
            environment = {}
            for value in (process / "environ").read_bytes().split(b"\0"):
                if b"=" in value:
                    key, raw = value.split(b"=", 1)
                    environment[os.fsdecode(key)] = os.fsdecode(raw)
            rec_csv = environment.get("REC_CSV")
            if not rec_csv or Path(os.path.expanduser(rec_csv)).resolve(strict=True) != expected_csv:
                continue
            opened = False
            for descriptor in (process / "fd").iterdir():
                try:
                    if descriptor.resolve(strict=True) == expected_csv:
                        opened = True
                        break
                except OSError:
                    continue
            if not opened:
                continue
            binary_sha = _sha256(exe_link)
            if binary_sha != _sha256(expected_binary):
                continue
            start_ticks = None
            stat_path = process / "stat"
            if stat_path.is_file():
                fields = stat_path.read_text(encoding="utf-8").split()
                if len(fields) >= 22:
                    start_ticks = int(fields[21])
            matches.append({
                "arm_argument": arm_argument,
                "binary_path": str(expected_binary),
                "binary_sha256": binary_sha,
                "can_interface": can_interface,
                "cmdline": argv,
                "joint_csv": str(expected_csv),
                "joint_csv_open_fd": True,
                "pid": int(process.name),
                "proc_start_ticks": start_ticks,
                "urdf_path": str(expected_urdf),
                "urdf_sha256": _sha256(expected_urdf),
            })
        except (OSError, UnicodeError, ValueError):
            # Processes can exit while /proc is inspected. They cannot count as
            # a stable calibration writer.
            continue
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one matching gravity-comp process; found %d" % len(matches)
        )
    return matches[0]


def clicked_depth_metres(frame, u: int, v: int):
    """Return aligned depth in metres without assuming one raw unit is one millimetre."""
    if frame is None or frame.depth is None:
        return None
    if v < 0 or u < 0 or v >= frame.depth.shape[0] or u >= frame.depth.shape[1]:
        return None
    raw = float(frame.depth[v, u])
    scale_mm = float(frame.depth_scale)
    if not np.isfinite(raw) or not np.isfinite(scale_mm) or raw <= 0 or scale_mm <= 0:
        return None
    value = raw * scale_mm / 1000.0
    return value if 0.1 < value < 3.0 else None


def clicked_depth_patch(
    frame,
    u: int,
    v: int,
    *,
    radius: int = 2,
    min_valid: int = 16,
    max_span_mm: float = 5.0,
):
    """Return a robust local depth only on a well-supported single surface."""
    if frame is None or getattr(frame, "depth", None) is None or radius < 1:
        return None
    depth = np.asarray(frame.depth)
    if depth.ndim != 2 or not (radius <= u < depth.shape[1] - radius):
        return None
    if not radius <= v < depth.shape[0] - radius:
        return None
    try:
        scale_mm = float(frame.depth_scale)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(scale_mm) or scale_mm <= 0.0:
        return None
    patch = np.asarray(
        depth[v - radius:v + radius + 1, u - radius:u + radius + 1],
        dtype=np.float64,
    )
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size < min_valid:
        return None
    low, high = np.percentile(valid, (5.0, 95.0))
    span_mm = float((high - low) * scale_mm)
    raw_median = float(np.median(valid))
    depth_m = raw_median * scale_mm / 1000.0
    if span_mm > max_span_mm or not 0.1 < depth_m < 3.0:
        return None
    return {
        "depth_m": depth_m,
        "raw_median": raw_median,
        "span_mm": span_mm,
        "valid_count": int(valid.size),
        "window_size": int(2 * radius + 1),
    }


def tip_touch_sample_plan(*, pair_count: int, heldout_count: int):
    if pair_count <= 0 or heldout_count <= 0:
        raise ValueError("tip-touch plan counts must be positive")
    plan = []
    for position in range(1, pair_count + 1):
        for orientation in ("A", "B"):
            plan.append({
                "orientation": orientation,
                "phase": "fit",
                "position_index": position,
            })
    for position in range(1, heldout_count + 1):
        plan.append({
            "orientation": None,
            "phase": "heldout",
            "position_index": position,
        })
    return plan


def build_tip_touch_sample(
    *, frame, intrinsics, pixel, q7, T_tcp_base, sample_id: int, plan_entry,
):
    u, v = int(pixel[0]), int(pixel[1])
    depth_measurement = clicked_depth_patch(frame, u, v)
    if depth_measurement is None:
        raise ValueError("clicked depth patch is invalid or crosses a depth edge")
    depth_m = float(depth_measurement["depth_m"])
    q7 = np.asarray(q7, dtype=np.float64)
    transform = np.asarray(T_tcp_base, dtype=np.float64)
    if q7.shape != (7,) or transform.shape != (4, 4):
        raise ValueError("tip-touch FK sample has invalid shape")
    tip_camera = intrinsics.unproject(
        np.asarray([[u, v]], dtype=np.float64),
        np.asarray([depth_m], dtype=np.float64),
    )[0]
    return {
        "T_tcp_base": transform.tolist(),
        "camera_age_s": float(frame.age_s),
        "camera_device_t_ms": (
            None if frame.device_t_ms is None else float(frame.device_t_ms)
        ),
        "camera_frame_index": int(frame.index),
        "camera_host_monotonic_s": float(frame.t),
        "camera_time_source": str(frame.time_source),
        "depth_m": float(depth_m),
        "depth_patch_span_mm": float(depth_measurement["span_mm"]),
        "depth_patch_valid_count": int(depth_measurement["valid_count"]),
        "depth_patch_window_size": int(depth_measurement["window_size"]),
        "depth_raw": float(depth_measurement["raw_median"]),
        "depth_scale_mm": float(frame.depth_scale),
        "orientation": plan_entry["orientation"],
        "phase": str(plan_entry["phase"]),
        "pixel": [u, v],
        "position_index": int(plan_entry["position_index"]),
        "q7_rad": q7.tolist(),
        "sample_id": int(sample_id),
        "tip_cam_m": np.asarray(tip_camera, dtype=np.float64).tolist(),
    }


# ── ChArUco board pose in camera frame ───────────────────────────────────────
class CharucoPoseEstimator:
    def __init__(self, sx, sy, square_len, marker_len, dict_name, K, dist):
        d = getattr(cv2.aruco, f"DICT_{dict_name}")
        self.adict = cv2.aruco.getPredefinedDictionary(d)
        self.board = cv2.aruco.CharucoBoard((sx, sy), square_len, marker_len, self.adict)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.K, self.dist = K, dist

    def pose(self, rgb):
        """Return (T_target_cam 4x4, n_corners) or (None, 0)."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cc, cids, _, _ = self.detector.detectBoard(gray)
        if cids is None or len(cids) < 6:
            return None, 0
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            cc, cids, self.board, self.K, self.dist, None, None)
        if not ok:
            return None, 0
        T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rvec)[0]; T[:3, 3] = tvec.ravel()
        return T, len(cids)


def _split(T):
    return T[:3, :3], T[:3, 3]


def solve_eye_to_hand(T_tcp_base_list, T_target_cam_list):
    """Camera fixed, target on TCP. Returns T_cam_base (4x4): camera pose in the arm base frame.
    Eye-to-hand uses base→gripper (inverse of FK) with cv2.calibrateHandEye."""
    R_b2g, t_b2g, R_t2c, t_t2c = [], [], [], []
    for Ttb, Ttc in zip(T_tcp_base_list, T_target_cam_list):
        Tbg = np.linalg.inv(Ttb)                       # base→gripper(TCP)
        R, t = _split(Tbg); R_b2g.append(R); t_b2g.append(t)
        R, t = _split(Ttc); R_t2c.append(R); t_t2c.append(t)
    R_cb, t_cb = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c,
                                      method=cv2.CALIB_HAND_EYE_PARK)
    T_cam_base = np.eye(4); T_cam_base[:3, :3] = R_cb; T_cam_base[:3, 3] = t_cb.ravel()
    return T_cam_base


def consistency_residual(T_cam_base, T_tcp_base_list, T_target_cam_list):
    """T_base_target = T_base_cam · T_cam_target must be a CONSTANT rigid offset (target↔TCP).
    Report the spread of the implied TCP→target offset across poses = calibration quality (mm)."""
    offs = []
    for Ttb, Ttc in zip(T_tcp_base_list, T_target_cam_list):
        T_base_target = T_cam_base @ Ttc               # target in base (T_cam_base maps cam→base)
        T_tcp_target = np.linalg.inv(Ttb) @ T_base_target  # TCP→target offset; constant if calib is right
        offs.append(T_tcp_target[:3, 3])
    offs = np.array(offs)
    return float(np.linalg.norm(offs - offs.mean(0), axis=1).mean() * 1000.0)


# ── tip-touch calibration (no board): touch the tool tip to N points the camera sees ──────────
def _solve_tip_touch_diagnostics(tip_cam, R_tcp, t_tcp):
    """Solve T_cam_base (cam→base) AND the unknown tip offset p_tip (in TCP frame) from N touches.
    Per touch i: T_cam_base · tip_cam_i  ==  R_tcp_i · p_tip + t_tcp_i  (same physical tip point).
    Unknowns: rotvec(3) + t(3) of T_cam_base, plus p_tip(3). Nonlinear LS over 9 params."""
    from scipy.optimize import least_squares
    tip_cam = np.asarray(tip_cam); R_tcp = np.asarray(R_tcp); t_tcp = np.asarray(t_tcp)

    def resid(x):
        R = cv2.Rodrigues(x[:3])[0]; t = x[3:6]; p = x[6:9]
        lhs = (R @ tip_cam.T).T + t                       # tip in base via camera
        rhs = np.einsum("ijk,k->ij", R_tcp, p) + t_tcp    # tip in base via FK + offset
        return (lhs - rhs).ravel()

    x0 = np.zeros(9); x0[3:6] = t_tcp.mean(0) - tip_cam.mean(0)
    sol = least_squares(resid, x0, method="lm")
    if not sol.success or not np.all(np.isfinite(sol.x)):
        raise ValueError("tip-touch optimizer did not produce a finite solution")
    R = cv2.Rodrigues(sol.x[:3])[0]
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = sol.x[3:6]
    residual = resid(sol.x).reshape(-1, 3)
    errors_mm = np.linalg.norm(residual, axis=1) * 1000.0
    # The gate is a 3-D point error, not a per-coordinate RMS that silently
    # permits sqrt(3) more Euclidean error.
    rms = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))) * 1000.0)
    singular = np.linalg.svd(np.asarray(sol.jac, dtype=np.float64), compute_uv=False)
    rank = int(np.linalg.matrix_rank(np.asarray(sol.jac, dtype=np.float64)))
    condition = float("inf") if singular[-1] <= 0 else float(singular[0] / singular[-1])
    if not all(
        np.all(np.isfinite(value))
        for value in (T, sol.x[6:9], errors_mm, singular)
    ) or not np.isfinite(rms):
        raise ValueError("tip-touch optimizer diagnostics are not finite")
    return T, sol.x[6:9], rms, rank, condition, errors_mm


def solve_tip_touch(tip_cam, R_tcp, t_tcp):
    """Compatibility wrapper that refuses to bypass the commissioning gate."""
    candidate = evaluate_tip_touch_candidate(
        tip_cam, R_tcp, t_tcp,
    )
    if not candidate["accepted"]:
        raise ValueError(
            "tip-touch candidate rejected: %s"
            % ",".join(candidate["failure_reasons"])
        )
    return (
        np.asarray(candidate["T_cam_base"], dtype=np.float64),
        np.asarray(candidate["tip_offset_tcp_m"], dtype=np.float64),
        float(candidate["training_rms_mm"]),
    )


def _rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def evaluate_tip_touch_candidate(
    tip_cam,
    R_tcp,
    t_tcp,
    *,
    fit_count: int = 20,
    pair_count: int = 10,
    heldout_count: int = 3,
    max_training_rms_mm: float = 3.0,
    max_training_error_mm: float = 5.0,
    max_heldout_error_mm: float = 5.0,
    min_pair_rotation_deg: float = 30.0,
    max_pair_position_delta_mm: float = 5.0,
    min_fit_pca_std_mm: float = 30.0,
    min_fit_diameter_mm: float = 250.0,
    min_fit_separation_mm: float = 50.0,
    min_heldout_separation_mm: float = 50.0,
    min_heldout_to_fit_mm: float = 50.0,
):
    """Solve on paired captures and independently score the reserved held-out touches."""
    tip_cam = np.asarray(tip_cam, dtype=np.float64)
    R_tcp = np.asarray(R_tcp, dtype=np.float64)
    t_tcp = np.asarray(t_tcp, dtype=np.float64)
    expected = fit_count + heldout_count
    if (
        fit_count != 20
        or pair_count != 10
        or heldout_count != 3
        or tip_cam.shape != (expected, 3)
    ):
        raise ValueError("tip-touch sample count must equal paired fit captures plus held-out")
    if R_tcp.shape != (expected, 3, 3) or t_tcp.shape != (expected, 3):
        raise ValueError("tip-touch FK arrays have invalid shapes")
    if not (
        np.all(np.isfinite(tip_cam))
        and np.all(np.isfinite(R_tcp))
        and np.all(np.isfinite(t_tcp))
    ):
        raise ValueError("tip-touch inputs must all be finite")
    identity = np.eye(3, dtype=np.float64)
    for rotation in R_tcp:
        if (
            np.linalg.norm(rotation.T @ rotation - identity, ord="fro") > 1e-6
            or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6
        ):
            raise ValueError("tip-touch FK rotations must be proper finite rotations")

    T, tip_offset, rms, rank, condition, training_errors = _solve_tip_touch_diagnostics(
        tip_cam[:fit_count], R_tcp[:fit_count], t_tcp[:fit_count],
    )
    pair_rotation = []
    pair_position_delta = []
    for index in range(pair_count):
        first, second = 2 * index, 2 * index + 1
        pair_rotation.append(_rotation_angle_deg(R_tcp[first], R_tcp[second]))
        pair_position_delta.append(
            float(np.linalg.norm(tip_cam[first] - tip_cam[second]) * 1000.0)
        )

    wrist_matrix = np.vstack([
        R_tcp[2 * index] - R_tcp[2 * index + 1]
        for index in range(pair_count)
    ])
    wrist_singular = np.linalg.svd(wrist_matrix, compute_uv=False)
    wrist_rank = int(np.linalg.matrix_rank(wrist_matrix))
    wrist_condition = (
        float("inf") if wrist_singular[-1] <= 0.0
        else float(wrist_singular[0] / wrist_singular[-1])
    )
    wrist_normalized_smin = float(wrist_singular[-1] / math.sqrt(pair_count))

    fit_centroids = tip_cam[:fit_count].reshape(pair_count, 2, 3).mean(axis=1)
    centered = fit_centroids - fit_centroids.mean(axis=0)
    fit_singular = np.linalg.svd(centered, compute_uv=False)
    fit_pca_std_mm = (fit_singular / math.sqrt(pair_count) * 1000.0).tolist()
    fit_distances_mm = np.linalg.norm(
        fit_centroids[:, None, :] - fit_centroids[None, :, :], axis=2,
    ) * 1000.0
    upper_fit = fit_distances_mm[np.triu_indices(pair_count, 1)]
    fit_min_separation_mm = float(np.min(upper_fit))
    fit_diameter_mm = float(np.max(upper_fit))

    heldout_errors = []
    for index in range(fit_count, expected):
        camera_estimate = T[:3, :3] @ tip_cam[index] + T[:3, 3]
        fk_estimate = R_tcp[index] @ tip_offset + t_tcp[index]
        heldout_errors.append(float(np.linalg.norm(camera_estimate - fk_estimate) * 1000.0))
    heldout = tip_cam[fit_count:expected]
    heldout_to_fit_mm = np.min(
        np.linalg.norm(heldout[:, None, :] - fit_centroids[None, :, :], axis=2),
        axis=1,
    ) * 1000.0
    heldout_distances_mm = np.linalg.norm(
        heldout[:, None, :] - heldout[None, :, :], axis=2,
    ) * 1000.0
    upper_heldout = heldout_distances_mm[np.triu_indices(heldout_count, 1)]
    heldout_min_separation_mm = float(np.min(upper_heldout))

    failures = []
    if min(pair_rotation) < min_pair_rotation_deg:
        failures.append("insufficient_pair_rotation")
    if max(pair_position_delta) > max_pair_position_delta_mm:
        failures.append("paired_position_mismatch")
    if (
        wrist_rank < 3
        or not np.isfinite(wrist_condition)
        or wrist_condition > 5.0
        or wrist_normalized_smin < 0.30
    ):
        failures.append("insufficient_wrist_axis_excitation")
    if (
        min(fit_pca_std_mm) < min_fit_pca_std_mm
        or fit_diameter_mm < min_fit_diameter_mm
        or fit_min_separation_mm < min_fit_separation_mm
    ):
        failures.append("insufficient_fit_spatial_coverage")
    if (
        heldout_min_separation_mm < min_heldout_separation_mm
        or float(np.min(heldout_to_fit_mm)) < min_heldout_to_fit_mm
    ):
        failures.append("heldout_points_not_independent")
    if rank < 9:
        failures.append("solver_rank_deficient")
    if not np.isfinite(condition) or condition > 1e8:
        failures.append("solver_ill_conditioned")
    if rms > max_training_rms_mm:
        failures.append("training_rms_above_3mm")
    if float(np.max(training_errors)) > max_training_error_mm:
        failures.append("training_max_error_above_5mm")
    if max(heldout_errors) > max_heldout_error_mm:
        failures.append("heldout_error_above_5mm")

    return {
        "T_cam_base": T.tolist(),
        "accepted": not failures,
        "failure_reasons": failures,
        "fit_count": fit_count,
        "heldout_count": heldout_count,
        "heldout_errors_mm": heldout_errors,
        "heldout_max_error_mm": max(heldout_errors),
        "pair_position_delta_mm": pair_position_delta,
        "pair_rotation_deg": pair_rotation,
        "fit_diameter_mm": fit_diameter_mm,
        "fit_min_separation_mm": fit_min_separation_mm,
        "fit_pca_std_mm": fit_pca_std_mm,
        "heldout_min_separation_mm": heldout_min_separation_mm,
        "heldout_to_fit_mm": heldout_to_fit_mm.tolist(),
        "solver_condition_number": condition,
        "solver_rank": rank,
        "tip_offset_tcp_m": tip_offset.tolist(),
        "training_errors_mm": training_errors.tolist(),
        "training_max_error_mm": float(np.max(training_errors)),
        "training_rms_mm": rms,
        "wrist_axis_condition_number": wrist_condition,
        "wrist_axis_normalized_smin": wrist_normalized_smin,
        "wrist_axis_rank": wrist_rank,
    }


_RUN_ID = re.compile(r"^handeye_(?:left|right)_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")


def _canonical_json(value) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_all(descriptor: int, body: bytes) -> None:
    view = memoryview(body)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("short write while persisting hand-eye evidence")
        offset += written


def _write_new(path: Path, body: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        _write_all(descriptor, body)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_commissioning_run(root, run_id: str, metadata) -> Path:
    root = Path(root)
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid hand-eye run id")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or root.resolve(strict=True) != root.absolute():
        raise RuntimeError("hand-eye run root must not contain a symlink")
    run = root / run_id
    run.mkdir(mode=0o700)
    (run / "captures").mkdir(mode=0o700)
    _write_new(run / "RUN_IN_PROGRESS.json", _canonical_json(metadata), 0o600)
    _fsync_directory(run)
    _fsync_directory(root)
    _fsync_directory(root.parent)
    return run


def persist_tip_touch_sample(run_dir, sample, rgb: np.ndarray, depth: np.ndarray):
    run = Path(run_dir)
    sample_id = int(sample["sample_id"])
    if sample_id <= 0:
        raise ValueError("sample id must be positive")
    prefix = "%04d" % sample_id
    rgb_path = run / "captures" / (prefix + "_rgb.png")
    depth_path = run / "captures" / (prefix + "_depth.npy")
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("RGB capture could not be encoded")
    depth_buffer = io.BytesIO()
    np.save(depth_buffer, np.asarray(depth), allow_pickle=False)
    rgb_body = encoded.tobytes()
    depth_body = depth_buffer.getvalue()
    _write_new(rgb_path, rgb_body)
    _write_new(depth_path, depth_body)
    persisted = dict(sample)
    persisted.update({
        "depth_path": depth_path.relative_to(run).as_posix(),
        "depth_sha256": hashlib.sha256(depth_body).hexdigest(),
        "rgb_path": rgb_path.relative_to(run).as_posix(),
        "rgb_sha256": hashlib.sha256(rgb_body).hexdigest(),
    })
    descriptor = os.open(
        run / "samples.jsonl",
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _write_all(descriptor, _canonical_json(persisted))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(run / "captures")
    _fsync_directory(run)
    return persisted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_file(run_dir, source, relative: str):
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != relative
        or any(part in ("", ".", "..") for part in parsed.parts)
    ):
        raise ValueError("snapshot destination is unsafe")
    source_path = Path(source)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_path, flags)
    try:
        before = os.fstat(descriptor)
        remaining = before.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("joint stream was truncated during snapshot")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError("joint stream identity changed during snapshot")
    finally:
        os.close(descriptor)
    body = b"".join(chunks)
    destination = Path(run_dir) / Path(*parsed.parts)
    _write_new(destination, body, 0o600)
    _fsync_directory(destination.parent)
    return {
        "path": relative,
        "sha256": hashlib.sha256(body).hexdigest(),
        "size_bytes": len(body),
        "source_path": str(source_path),
    }


def finalize_commissioning_run(run_dir, candidate, provenance):
    run = Path(run_dir)
    in_progress = run / "RUN_IN_PROGRESS.json"
    if not in_progress.is_file() or in_progress.is_symlink():
        raise RuntimeError("hand-eye run is not in progress")
    if not candidate.get("accepted"):
        result = {
            "failure_reasons": list(candidate.get("failure_reasons", [])),
            "provenance": provenance,
            "schema": "wam_handeye_failure_v1",
            "status": "REJECTED",
        }
        _write_new(run / "FAILED.json", _canonical_json(result))
        in_progress.unlink()
        _fsync_directory(run)
        return result

    result = {
        "candidate": candidate,
        "production_activation": False,
        "provenance": provenance,
        "schema": "wam_handeye_candidate_v1",
        "status": "CANDIDATE_COMPLETE_NOT_ACTIVATED",
    }
    _write_new(run / "cam2base_candidate.json", _canonical_json(result))
    payload = []
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("hand-eye run contains a symlink")
        if path.is_file() and path.name not in {
            "RUN_IN_PROGRESS.json", "SHA256SUMS.txt", "COMPLETE.json",
        }:
            relative = path.relative_to(run).as_posix()
            payload.append((relative, _sha256(path)))
    sums = "".join("%s  %s\n" % (digest, relative) for relative, digest in payload).encode()
    _write_new(run / "SHA256SUMS.txt", sums)
    complete = {
        "file_count": len(payload),
        "payload_inventory_sha256": hashlib.sha256(sums).hexdigest(),
        "production_activation": False,
        "schema": "wam_handeye_completion_v1",
        "status": "CANDIDATE_COMPLETE_NOT_ACTIVATED",
    }
    _write_new(run / "COMPLETE.json", _canonical_json(complete))
    in_progress.unlink()
    _fsync_directory(run)
    return result


def capture_tiptouch(
    cam,
    joint_csv,
    fk,
    intr,
    plan,
    run_dir,
    *,
    display_scale: float = 0.5,
):
    """Capture and persist one auditable RGB-D/FK record for every accepted click."""
    click = {"uv": None}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            click["uv"] = (x, y)
    cv2.namedWindow("tip-touch calib"); cv2.setMouseCallback("tip-touch calib", on_mouse)
    samples = []
    aborted = False
    try:
        while len(samples) < len(plan):
            frame = cam.latest_frame()
            if not calibration_frame_ready(cam, frame):
                # A click belongs only to the frame on which it was made. Never carry it
                # across a stale/frozen interval and combine it with newer FK.
                click["uv"] = None
                time.sleep(0.05)
                continue
            vis = cv2.cvtColor(frame.frame, cv2.COLOR_RGB2BGR)
            q7, still = read_latest_joints(joint_csv)
            entry = plan[len(samples)]
            if click["uv"] is not None:
                display_u, display_v = click["uv"]
                click["uv"] = None
                if q7 is None or not still:
                    print("  rejected: gravity-comp joint stream is stale or arm is moving")
                else:
                    u = int(display_u / display_scale)
                    v = int(display_v / display_scale)
                    transform = fk.fk(q7)
                    try:
                        sample = build_tip_touch_sample(
                            frame=frame,
                            intrinsics=intr,
                            pixel=(u, v),
                            q7=q7,
                            T_tcp_base=transform,
                            sample_id=len(samples) + 1,
                            plan_entry=entry,
                        )
                    except ValueError as exc:
                        print("  rejected: %s" % exc)
                    else:
                        if entry["phase"] == "fit" and entry["orientation"] == "B":
                            previous = samples[-1]
                            delta = np.linalg.norm(
                                np.asarray(previous["tip_cam_m"]) - np.asarray(sample["tip_cam_m"])
                            ) * 1000.0
                            angle = _rotation_angle_deg(
                                np.asarray(previous["T_tcp_base"])[:3, :3],
                                np.asarray(sample["T_tcp_base"])[:3, :3],
                            )
                            if delta > 5.0:
                                print("  rejected: touch the same rigid position (delta %.1f mm)" % delta)
                                sample = None
                            elif angle < 30.0:
                                print("  rejected: change wrist orientation by at least 30 deg (%.1f)" % angle)
                                sample = None
                        if sample is not None:
                            persisted = persist_tip_touch_sample(
                                run_dir, sample, frame.frame, frame.depth,
                            )
                            samples.append(persisted)
                            print(
                                "  captured %d/%d: %s P%02d %s"
                                % (
                                    len(samples), len(plan), entry["phase"],
                                    entry["position_index"], entry["orientation"] or "",
                                )
                            )
            label = "%s P%02d %s" % (
                entry["phase"], entry["position_index"], entry["orientation"] or "",
            )
            message = "%d/%d %s still=%s | click contact | q=abort" % (
                len(samples), len(plan), label, still,
            )
            cv2.putText(
                vis, message, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2,
            )
            cv2.imshow(
                "tip-touch calib",
                cv2.resize(vis, None, fx=display_scale, fy=display_scale),
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                aborted = True
                break
    finally:
        cv2.destroyAllWindows()
    return samples, aborted


def parse_arguments(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        choices=("left", "right"),
        default="left",
        help="OpenArm side (production default: left; follower CAN: can3)",
    )
    ap.add_argument("--can-interface")
    ap.add_argument("--joint-csv", required=True)
    ap.add_argument("--urdf", default="~/Desktop/Openarm_project/openarm_v10.urdf")
    ap.add_argument(
        "--gravity-comp-bin",
        default="~/Desktop/Openarm_project/openarm_teleop/build/gravity_comp",
    )
    ap.add_argument("--camera-id", default="CL8326300A0")
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-len", type=float, default=0.030)   # metres
    ap.add_argument("--marker-len", type=float, default=0.022)
    ap.add_argument("--dict", default="5X5_50")
    ap.add_argument("--n-poses", type=int, default=18)
    ap.add_argument("--mode", choices=["tiptouch", "charuco"], default="tiptouch")
    ap.add_argument("--pair-count", type=int, default=10)
    ap.add_argument("--n-points", type=int, default=20)          # paired fit captures
    ap.add_argument("--heldout-points", type=int, default=3)
    ap.add_argument("--run-id")
    ap.add_argument("--run-root")
    ap.add_argument("--fixture-id", required=True)
    ap.add_argument("--environment-lock-sha256")
    ap.add_argument("--environment-clone-marker-sha256")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out")  # legacy Charuco-only path
    return ap.parse_args(argv)


def validate_commissioning_arguments(args) -> None:
    if args.mode != "tiptouch":
        raise ValueError("Charuco capture is not yet covered by the commissioning contract")
    if args.pair_count != 10 or args.n_points != 20:
        raise ValueError("tip-touch requires two wrist orientations for every rigid position")
    if args.heldout_points != 3:
        raise ValueError("tip-touch requires exactly three independent held-out touches")
    if args.headless:
        raise ValueError("tip-touch commissioning requires the Lingbox graphical desktop")
    if not args.camera_id:
        raise ValueError("an exact Orbbec camera serial is required")
    if not args.fixture_id.strip():
        raise ValueError("a stable rigid fixture identifier is required")
    if not re.fullmatch(r"[0-9a-f]{64}", args.environment_lock_sha256 or ""):
        raise ValueError("the exact perception environment lock SHA-256 is required")
    if not re.fullmatch(r"[0-9a-f]{64}", args.environment_clone_marker_sha256 or ""):
        raise ValueError("the exact perception environment clone marker SHA-256 is required")
    expected_can = "can2" if args.arm == "right" else "can3"
    if args.can_interface is None:
        args.can_interface = expected_can
    if args.can_interface != expected_can:
        raise ValueError("%s follower must use %s" % (args.arm, expected_can))


def main(argv=None):
    paths = activate_runtime(RUNTIME_ROLE)
    args = parse_arguments(argv)
    validate_commissioning_arguments(args)
    software_provenance = collect_handeye_provenance(
        environment_lock_sha256=args.environment_lock_sha256,
        environment_report_sha256=args.environment_clone_marker_sha256,
    )

    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [os.path.join(HERE, "..", "..", "ausc", "perception")]
    from camera_stream import CameraStream

    joint_csv = Path(os.path.expanduser(args.joint_csv))
    urdf = Path(os.path.expanduser(args.urdf))
    gravity_comp_bin = Path(os.path.expanduser(args.gravity_comp_bin))
    if joint_csv.is_symlink() or not joint_csv.is_file():
        raise RuntimeError("fresh gravity-comp joint CSV is required before opening the camera")
    joints, _still = read_latest_joints(str(joint_csv))
    if joints is None:
        raise RuntimeError("gravity-comp joint CSV is stale or invalid")
    if urdf.is_symlink() or not urdf.is_file():
        raise RuntimeError("exact OpenArm URDF is missing or unsafe")
    gravity_process = verify_gravity_comp_process(
        arm=args.arm,
        can_interface=args.can_interface,
        gravity_comp_bin=gravity_comp_bin,
        joint_csv=joint_csv,
        urdf=urdf,
    )

    run_id = args.run_id or "handeye_%s_%s_%s" % (
        args.arm,
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        os.urandom(4).hex(),
    )
    run_root = Path(args.run_root) if args.run_root else (
        paths.output_root / "calibration" / "runs"
    )
    plan = tip_touch_sample_plan(
        pair_count=args.pair_count, heldout_count=args.heldout_points,
    )
    fk = ArmFK(str(urdf), args.arm)
    fk.fk(joints)  # Preflight exact joint ordering and URDF limits before camera ownership.
    cam = CameraStream(
        backend="orbbec",
        camera_id=args.camera_id,
        align_depth_to_color=True,
        width=1920,
        height=1080,
        fps=30,
        color_format="MJPG",
        depth_mode={"width": 640, "height": 576, "fps": 30, "format": "Y16"},
    )
    run = None
    try:
        cam.start()
        deadline = time.monotonic() + 8.0
        frame = None
        while time.monotonic() < deadline:
            candidate_frame = cam.latest_frame()
            if calibration_frame_ready(cam, candidate_frame):
                frame = candidate_frame
                break
            time.sleep(0.05)
        if frame is None:
            raise RuntimeError("Orbbec did not produce aligned RGB-D within 8 seconds")
        intr = calibration_intrinsics_from_stream(cam, frame)
        metadata = {
            "arm": args.arm,
            "camera": {
                "device_identity": cam.device_identity,
                "negotiated_mode": cam.negotiated_mode,
                "requested_serial": args.camera_id,
                "rgb_intrinsics": {
                    "cx": float(intr.cx), "cy": float(intr.cy),
                    "fx": float(intr.fx), "fy": float(intr.fy),
                    "height": int(intr.height), "width": int(intr.width),
                },
                "rgb_distortion": cam.color_distortion,
            },
            "can_interface": args.can_interface,
            "capture_contract": {
                "fit_count": args.n_points,
                "heldout_count": args.heldout_points,
                "pair_count": args.pair_count,
                "rigid_nonhuman_fixture_required": True,
                "two_wrist_orientations_per_position": True,
                "fit_3d_rms_max_mm": 3.0,
                "fit_each_point_max_mm": 5.0,
                "heldout_each_point_max_mm": 5.0,
                "minimum_pair_rotation_deg": 30.0,
                "maximum_pair_contact_delta_mm": 5.0,
            },
            "code": {
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": _sha256(Path(__file__).resolve()),
            },
            "joint_csv_source": str(joint_csv),
            "fixture": {
                "fixture_id": args.fixture_id,
                "rigid_nonhuman": True,
            },
            "mode": "tiptouch",
            "perception_model_used": False,
            "production_activation": False,
            "robot_writer": gravity_process,
            "software_provenance": software_provenance,
            "python_executable": str(Path(sys.executable).resolve()),
            "run_id": run_id,
            "schema": "wam_handeye_run_start_v1",
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "urdf": {
                "path": str(urdf), "sha256": _sha256(urdf),
                "size_bytes": urdf.stat().st_size,
            },
        }
        run = create_commissioning_run(run_root, run_id, metadata)
        print("[handeye] run=%s" % run, flush=True)
        print(
            "[handeye] use a rigid non-human 3D fixture: 10 positions x 2 wrist orientations, "
            "then 3 independent held-out touches. q aborts the run.",
            flush=True,
        )
        samples, aborted = capture_tiptouch(
            cam, str(joint_csv), fk, intr, plan, run,
        )
        joint_snapshot = snapshot_file(
            run, joint_csv, "joint_stream_snapshot.csv",
        )
        provenance = {
            **metadata,
            "joint_stream_snapshot": joint_snapshot,
            "sample_count": len(samples),
            "robot_writer_end": verify_gravity_comp_process(
                arm=args.arm,
                can_interface=args.can_interface,
                gravity_comp_bin=gravity_comp_bin,
                joint_csv=joint_csv,
                urdf=urdf,
            ),
        }
        if provenance["robot_writer_end"]["pid"] != gravity_process["pid"]:
            raise RuntimeError("gravity-comp writer changed during calibration")
        if aborted or len(samples) != len(plan):
            result = finalize_commissioning_run(
                run,
                {
                    "accepted": False,
                    "failure_reasons": [
                        "operator_abort" if aborted else "incomplete_capture",
                    ],
                },
                provenance,
            )
            print("[handeye] REJECTED: %s" % result["failure_reasons"], flush=True)
            return 2

        tip_camera = [np.asarray(sample["tip_cam_m"], dtype=np.float64) for sample in samples]
        transforms = [np.asarray(sample["T_tcp_base"], dtype=np.float64) for sample in samples]
        candidate = evaluate_tip_touch_candidate(
            tip_camera,
            [transform[:3, :3] for transform in transforms],
            [transform[:3, 3] for transform in transforms],
            fit_count=args.n_points,
            pair_count=args.pair_count,
            heldout_count=args.heldout_points,
        )
        candidate.update({
            "arm": args.arm,
            "convention": "p_base = T_cam_base @ p_camera",
            "mode": "tiptouch",
            "run_id": run_id,
        })
        result = finalize_commissioning_run(run, candidate, provenance)
        if not candidate["accepted"]:
            print("[handeye] REJECTED: %s" % candidate["failure_reasons"], flush=True)
            return 3
        print(
            "[handeye] CANDIDATE COMPLETE (not activated): train RMS %.2f mm; "
            "held-out max %.2f mm; run=%s"
            % (candidate["training_rms_mm"], candidate["heldout_max_error_mm"], run),
            flush=True,
        )
        return 0
    except Exception as exc:
        if run is not None and (run / "RUN_IN_PROGRESS.json").is_file():
            provenance = {
                "arm": args.arm,
                "exception_type": type(exc).__name__,
                "production_activation": False,
                "run_id": run_id,
            }
            finalize_commissioning_run(
                run,
                {"accepted": False, "failure_reasons": ["runtime_exception"]},
                provenance,
            )
        raise
    finally:
        if hasattr(cam, "stop"):
            cam.stop()


if __name__ == "__main__":
    raise SystemExit(main())
