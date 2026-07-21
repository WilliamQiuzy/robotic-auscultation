#!/usr/bin/env python3
"""Run the isolated official-asset OpenArm occlusion experiment on Lingbox.

The script never opens the Orbbec, imports a production collector module, or sends a robot
command.  It verifies a frozen asset bundle, imports the complete official OpenArm v1 dual-arm
URDF at metre scale, references the pinned NVIDIA medical human, and obtains robot/human
visibility from Isaac RGB/depth/semantic products.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, is_dataclass
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import sys
import traceback
from typing import Callable, Iterator, Mapping, Optional, Sequence

import numpy as np


# Exact source closure for the isolated simulation_code artifact.  In particular,
# the production tracking package initializer is excluded because it imports the
# live detector/tracker stack; the tracking directory is an implicit namespace in
# this deliberately minimal release.
ISOLATED_CODE_RELEASE_PATHS = (
    "auscultation/ausc/__init__.py",
    "auscultation/ausc/perception/tracking/experimental_visibility_geometry.py",
    "auscultation/ausc/perception/tracking/official_asset_contract.py",
    "auscultation/ausc/perception/tracking/official_openarm_experiment.py",
    "auscultation/scripts/perception/isaac_official_openarm_occlusion.py",
)

_POINT_POLICY_SCHEMA = "wam_per_point_occlusion_policy_v3"
_POINT_POLICY_STATES = (
    "DIRECT_GEOMETRY_UPDATE",
    "BOUNDED_PATIENT_FRAME_FREEZE",
    "LOST",
)
_POINT_POLICY_CSV_FIELDS = (
    "frame_index",
    "joint_time_s",
    "point_name",
    "point_index",
    "trajectory_target_name",
    "trajectory_target_index",
    "trajectory_group_index",
    "wrist_index",
    "motion_phase",
    "policy_state",
    "policy_reason",
    "predicted_error_p95_m",
    "last_valid_age_s",
    "history_available",
    "estimate_valid",
    "estimate_source",
    "estimate_camera_x_m",
    "estimate_camera_y_m",
    "estimate_camera_z_m",
    "target_rendered_observation_available",
    "target_geometry_inlier",
    "direct_update_accepted",
    "direct_update_failure_reasons",
    "geometry_surface_accepted",
    "geometry_failure_reasons",
    "geometry_target_error_p95_m",
    "target_visible_fraction",
    "target_robot_front_fraction",
    "target_valid_depth_fraction",
    "robot_front_fraction_sum",
    "max_single_point_robot_front_fraction",
    "fixed_visible_point_count_rule_used",
    "robot_motion_policy",
    "direct_estimate_source",
)
_POINT_POLICY_INT_FIELDS = frozenset(
    {
        "frame_index",
        "point_index",
        "trajectory_target_index",
        "trajectory_group_index",
        "wrist_index",
    }
)
_POINT_POLICY_REQUIRED_FLOAT_FIELDS = frozenset(
    {
        "joint_time_s",
        "predicted_error_p95_m",
        "target_visible_fraction",
        "target_robot_front_fraction",
        "target_valid_depth_fraction",
        "robot_front_fraction_sum",
        "max_single_point_robot_front_fraction",
    }
)
_POINT_POLICY_OPTIONAL_FLOAT_FIELDS = frozenset(
    {
        "last_valid_age_s",
        "estimate_camera_x_m",
        "estimate_camera_y_m",
        "estimate_camera_z_m",
        "geometry_target_error_p95_m",
    }
)
_POINT_POLICY_BOOL_FIELDS = frozenset(
    {
        "history_available",
        "estimate_valid",
        "target_rendered_observation_available",
        "target_geometry_inlier",
        "direct_update_accepted",
        "geometry_surface_accepted",
        "fixed_visible_point_count_rule_used",
    }
)


def _python_root() -> Path:
    return Path(__file__).resolve().parents[2]


if str(_python_root()) not in sys.path:
    sys.path.insert(0, str(_python_root()))

from ausc.perception.tracking import official_openarm_experiment as experiment  # noqa: E402
from ausc.perception.tracking import experimental_visibility_geometry as geometry  # noqa: E402


def _arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-bundle", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--ffmpeg-path")
    parser.add_argument("--ffmpeg-sha256")
    parser.add_argument("--ffprobe-path")
    parser.add_argument("--ffprobe-sha256")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--rt-subframes", type=int, default=4)
    args = parser.parse_args(argv)
    if args.mode == "full":
        required_media_options = {
            "--ffmpeg-path": args.ffmpeg_path,
            "--ffmpeg-sha256": args.ffmpeg_sha256,
            "--ffprobe-path": args.ffprobe_path,
            "--ffprobe-sha256": args.ffprobe_sha256,
        }
        missing = [name for name, value in required_media_options.items() if not value]
        if missing:
            parser.error(
                "full mode requires explicit media-tool bindings: " + ", ".join(missing)
            )
    return args


def _verified_runtime_provenance(
    *,
    version_lookup: Callable[[str], str] = importlib.metadata.version,
    executable: Optional[str] = None,
    python_version: Optional[Sequence[int]] = None,
) -> dict:
    """Fail closed unless the executing runtime is the pinned Isaac/Python pair."""

    raw_python_version = tuple(
        int(part)
        for part in (
            tuple(python_version) if python_version is not None else sys.version_info[:3]
        )
    )
    if len(raw_python_version) < 3 or raw_python_version[:2] != (3, 11):
        raise RuntimeError("official OpenArm experiment requires Python 3.11")

    executable_text = str(executable if executable is not None else sys.executable)
    try:
        executable_realpath = Path(executable_text).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("Python executable provenance is unavailable") from exc
    if not executable_realpath.is_file():
        raise RuntimeError("Python executable provenance is unavailable")

    try:
        isaacsim_version = str(version_lookup("isaacsim"))
        kernel_version = str(version_lookup("isaacsim-kernel"))
    except Exception as exc:
        raise RuntimeError("Isaac 5.1 runtime distributions are unavailable") from exc
    if isaacsim_version != "5.1.0.0" or kernel_version != "5.1.0.0":
        raise RuntimeError("official OpenArm experiment requires the exact Isaac 5.1 runtime")

    return {
        "python_executable": executable_text,
        "python_executable_realpath": str(executable_realpath),
        "python_version": ".".join(str(part) for part in raw_python_version[:3]),
        "isaacsim_version": isaacsim_version,
        "isaacsim_kernel_version": kernel_version,
        "expected_isaacsim_version": "5.1.0.0",
    }


def _media_identity(info: os.stat_result) -> dict:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size_bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _hash_media_descriptor(
    descriptor: int,
    *,
    expected_identity: Mapping[str, object],
    tool_name: str,
) -> str:
    before = os.fstat(descriptor)
    if _media_identity(before) != dict(expected_identity):
        raise RuntimeError(f"{tool_name} executable identity changed since startup")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    after = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if _media_identity(after) != dict(expected_identity):
        raise RuntimeError(f"{tool_name} executable changed while hashing")
    return digest.hexdigest()


def _media_fd_execution_path(
    descriptor: int,
    *,
    expected_identity: Mapping[str, object],
    tool_name: str,
    allow_test_fallback: bool = False,
) -> str:
    if sys.platform.startswith("linux"):
        root = "/proc/self/fd"
    elif allow_test_fallback and sys.platform == "darwin":
        # Unit tests mock the exec on macOS; the commissioned runtime is Linux and
        # must use /proc. macOS cannot exec a Mach-O image through /dev/fd.
        return f"/dev/fd/{descriptor}"
    else:
        raise RuntimeError(
            f"{tool_name} FD-pinned execution requires Linux /proc/self/fd"
        )
    execution_path = f"{root}/{descriptor}"
    try:
        linked_info = os.stat(execution_path)
    except OSError as exc:
        raise RuntimeError(f"{tool_name} FD-pinned execution path is unavailable") from exc
    identity = dict(expected_identity)
    if (linked_info.st_dev, linked_info.st_ino) != (
        identity.get("device"),
        identity.get("inode"),
    ):
        raise RuntimeError(f"{tool_name} FD-pinned execution path changed")
    return execution_path


def _verified_media_tool(
    supplied_path: str | Path,
    expected_sha256: str,
    *,
    tool_name: str,
    expected_identity: Optional[Mapping[str, object]] = None,
    _allow_test_fd_fallback: bool = False,
) -> dict:
    """Bind one explicit executable to its immutable bytes and version output."""

    if not isinstance(expected_sha256, str) or (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(
            f"{tool_name} SHA-256 must be canonical lowercase 64-character hex"
        )
    supplied_text = os.fspath(supplied_path)
    if not isinstance(supplied_text, str) or not supplied_text:
        raise RuntimeError(f"{tool_name} executable path is unavailable")
    requested = Path(supplied_text)
    if not requested.is_absolute():
        raise RuntimeError(f"{tool_name} executable path is unsafe or unavailable")

    try:
        requested_info = requested.lstat()
    except OSError as exc:
        raise RuntimeError(f"{tool_name} executable path is unavailable") from exc
    if stat.S_ISLNK(requested_info.st_mode):
        raise RuntimeError(f"{tool_name} executable path must not be a symlink")
    if (
        not stat.S_ISREG(requested_info.st_mode)
        or requested_info.st_mode & 0o111 == 0
        or not os.access(supplied_text, os.X_OK)
    ):
        raise RuntimeError(f"{tool_name} path is not an executable regular file")

    try:
        resolved = requested.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"{tool_name} executable path is unavailable") from exc

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(supplied_text, flags)
    except OSError as exc:
        raise RuntimeError(f"{tool_name} executable path is unsafe or unavailable") from exc
    failures: list[tuple[str, BaseException, object]] = []
    try:
        opened_info = os.fstat(descriptor)
        opened_identity = _media_identity(opened_info)
        if (
            not stat.S_ISREG(opened_info.st_mode)
            or opened_info.st_mode & 0o111 == 0
            or (opened_info.st_dev, opened_info.st_ino)
            != (requested_info.st_dev, requested_info.st_ino)
        ):
            raise RuntimeError(f"{tool_name} path is not an executable regular file")
        if expected_identity is not None and opened_identity != dict(expected_identity):
            raise RuntimeError(f"{tool_name} executable identity changed since startup")
        actual_sha256 = _hash_media_descriptor(
            descriptor,
            expected_identity=opened_identity,
            tool_name=tool_name,
        )
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{tool_name} SHA-256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        execution_path = _media_fd_execution_path(
            descriptor,
            expected_identity=opened_identity,
            tool_name=tool_name,
            allow_test_fallback=_allow_test_fd_fallback,
        )
        try:
            version = subprocess.run(
                [execution_path, "-version"],
                pass_fds=(descriptor,),
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"{tool_name} version is unavailable") from exc
        version_lines = version.stdout.splitlines() or version.stderr.splitlines()
        if not version_lines or not version_lines[0]:
            raise RuntimeError(f"{tool_name} version is unavailable")
        if _hash_media_descriptor(
            descriptor,
            expected_identity=opened_identity,
            tool_name=tool_name,
        ) != expected_sha256:
            raise RuntimeError(f"{tool_name} executable changed during verification")
        try:
            final_info = requested.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"{tool_name} executable changed during verification"
            ) from exc
        if _media_identity(final_info) != opened_identity:
            raise RuntimeError(f"{tool_name} executable changed during verification")
    except BaseException as exc:
        failures.append(
            (f"{tool_name} startup verification failed", exc, exc.__traceback__)
        )
    try:
        os.close(descriptor)
    except BaseException as exc:
        failures.append(
            (f"{tool_name} descriptor close failed", exc, exc.__traceback__)
        )
    _raise_first_failure(failures)
    return {
        "supplied_path": supplied_text,
        "resolved_realpath": str(resolved),
        "size_bytes": int(opened_info.st_size),
        "sha256": actual_sha256,
        "version_line": version_lines[0],
        "identity": opened_identity,
    }


def _open_pinned_media_tool(
    provenance: Mapping[str, object], *, tool_name: str, allow_test_fallback: bool
) -> tuple[int, str]:
    supplied_path = provenance.get("supplied_path")
    expected_sha256 = provenance.get("sha256")
    expected_identity = provenance.get("identity")
    expected_realpath = provenance.get("resolved_realpath")
    expected_size = provenance.get("size_bytes")
    if (
        not isinstance(supplied_path, str)
        or not isinstance(expected_sha256, str)
        or not isinstance(expected_identity, Mapping)
        or not isinstance(expected_realpath, str)
        or not isinstance(expected_size, int)
    ):
        raise RuntimeError(
            f"{tool_name} executable changed since startup: provenance is incomplete"
        )
    requested = Path(supplied_path)
    if not requested.is_absolute():
        raise RuntimeError(f"{tool_name} executable changed since startup")
    try:
        path_info = requested.lstat()
        resolved = requested.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"{tool_name} executable changed since startup") from exc
    if (
        stat.S_ISLNK(path_info.st_mode)
        or not stat.S_ISREG(path_info.st_mode)
        or path_info.st_mode & 0o111 == 0
        or not os.access(supplied_path, os.X_OK)
        or _media_identity(path_info) != dict(expected_identity)
        or str(resolved) != expected_realpath
        or path_info.st_size != expected_size
    ):
        raise RuntimeError(f"{tool_name} executable changed since startup")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(supplied_path, flags)
    except OSError as exc:
        raise RuntimeError(f"{tool_name} executable changed since startup") from exc
    try:
        opened_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_info.st_mode)
            or opened_info.st_mode & 0o111 == 0
            or _media_identity(opened_info) != dict(expected_identity)
            or _hash_media_descriptor(
                descriptor,
                expected_identity=expected_identity,
                tool_name=tool_name,
            )
            != expected_sha256
        ):
            raise RuntimeError(f"{tool_name} executable changed since startup")
        execution_path = _media_fd_execution_path(
            descriptor,
            expected_identity=expected_identity,
            tool_name=tool_name,
            allow_test_fallback=allow_test_fallback,
        )
        return descriptor, execution_path
    except BaseException as exc:
        failures = [
            (f"{tool_name} reuse verification failed", exc, exc.__traceback__)
        ]
        try:
            os.close(descriptor)
        except BaseException as close_error:
            failures.append(
                (
                    f"{tool_name} descriptor close failed",
                    close_error,
                    close_error.__traceback__,
                )
            )
        _raise_first_failure(failures)
        raise RuntimeError("unreachable media-tool cleanup path")


def _verify_pinned_media_tool(
    descriptor: int,
    provenance: Mapping[str, object],
    *,
    tool_name: str,
) -> None:
    expected_identity = provenance.get("identity")
    expected_sha256 = provenance.get("sha256")
    supplied_path = provenance.get("supplied_path")
    expected_realpath = provenance.get("resolved_realpath")
    if (
        not isinstance(expected_identity, Mapping)
        or not isinstance(expected_sha256, str)
        or not isinstance(supplied_path, str)
        or not isinstance(expected_realpath, str)
    ):
        raise RuntimeError(f"{tool_name} executable changed since startup")
    if _hash_media_descriptor(
        descriptor,
        expected_identity=expected_identity,
        tool_name=tool_name,
    ) != expected_sha256:
        raise RuntimeError(f"{tool_name} executable changed since startup")
    requested = Path(supplied_path)
    try:
        path_info = requested.lstat()
        resolved = requested.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"{tool_name} executable changed since startup") from exc
    if (
        stat.S_ISLNK(path_info.st_mode)
        or not stat.S_ISREG(path_info.st_mode)
        or path_info.st_mode & 0o111 == 0
        or _media_identity(path_info) != dict(expected_identity)
        or str(resolved) != expected_realpath
    ):
        raise RuntimeError(f"{tool_name} executable changed since startup")


@contextmanager
def _pinned_media_tool(
    provenance: Mapping[str, object], *, tool_name: str, allow_test_fallback: bool
) -> Iterator[tuple[str, int]]:
    descriptor, execution_path = _open_pinned_media_tool(
        provenance,
        tool_name=tool_name,
        allow_test_fallback=allow_test_fallback,
    )
    failures: list[tuple[str, BaseException, object]] = []
    try:
        try:
            yield execution_path, descriptor
        except BaseException as exc:
            failures.append((f"{tool_name} execution failed", exc, exc.__traceback__))
        try:
            _verify_pinned_media_tool(descriptor, provenance, tool_name=tool_name)
        except BaseException as exc:
            failures.append(
                (f"{tool_name} post-use verification failed", exc, exc.__traceback__)
            )
        try:
            os.close(descriptor)
        except BaseException as exc:
            failures.append((f"{tool_name} descriptor close failed", exc, exc.__traceback__))
    except BaseException as exc:
        failures.append((f"{tool_name} cleanup orchestration failed", exc, exc.__traceback__))
    _raise_first_failure(failures)


def _verified_media_tools(
    *,
    ffmpeg_path: str | Path,
    ffmpeg_sha256: str,
    ffprobe_path: str | Path,
    ffprobe_sha256: str,
    _allow_test_fd_fallback: bool = False,
) -> dict:
    return {
        "ffmpeg": _verified_media_tool(
            ffmpeg_path,
            ffmpeg_sha256,
            tool_name="ffmpeg",
            _allow_test_fd_fallback=_allow_test_fd_fallback,
        ),
        "ffprobe": _verified_media_tool(
            ffprobe_path,
            ffprobe_sha256,
            tool_name="ffprobe",
            _allow_test_fd_fallback=_allow_test_fd_fallback,
        ),
    }


def _runtime_provenance_with_media_tools(
    runtime_provenance: Mapping[str, object],
    media_tools: Mapping[str, Mapping[str, object]],
) -> dict:
    combined = dict(runtime_provenance)
    combined["media_tools"] = {
        tool_name: dict(media_tools[tool_name]) for tool_name in ("ffmpeg", "ffprobe")
    }
    return combined


def _validated_stage_inventory(
    raw: Mapping[str, object], runtime_provenance: Mapping[str, object]
) -> dict:
    """Validate the exact official-scene core before attaching run metadata."""

    inventory = experiment.validate_stage_inventory(raw)
    inventory["runtime_provenance"] = dict(runtime_provenance)
    return inventory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_asset_contract(path: Path):
    spec = importlib.util.spec_from_file_location("official_asset_contract", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the official asset contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _semantic_mask(packet: Mapping[str, object], label: str) -> np.ndarray:
    data = np.asarray(packet["data"])
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 2:
        raise RuntimeError(f"semantic annotation has unexpected shape {data.shape}")
    info = packet.get("info", {})
    mapping = info.get("idToLabels", {}) if isinstance(info, Mapping) else {}
    selected = []
    for raw_identifier, labels in mapping.items():
        if label in json.dumps(labels, sort_keys=True):
            selected.append(int(raw_identifier))
    if not selected:
        raise RuntimeError(f"semantic label did not render: {label}")
    return np.isin(data, selected)


def _instance_mask(packet: Mapping[str, object], scene_root: str) -> np.ndarray:
    """Select instance IDs whose authored prim paths are below one exact scene root."""

    data = np.asarray(packet["data"])
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 2:
        raise RuntimeError(f"instance annotation has unexpected shape {data.shape}")
    if not isinstance(scene_root, str) or not scene_root.startswith("/World/") or scene_root.endswith("/"):
        raise RuntimeError("instance scene root is invalid")
    info = packet.get("info", {})
    mapping = info.get("idToLabels", {}) if isinstance(info, Mapping) else {}
    selected = []
    for raw_identifier, labels in mapping.items():
        values = []
        if isinstance(labels, str):
            values.append(labels)
        elif isinstance(labels, Mapping):
            values.extend(value for value in labels.values() if isinstance(value, str))
        elif isinstance(labels, (list, tuple)):
            values.extend(value for value in labels if isinstance(value, str))
        if any(value == scene_root or value.startswith(scene_root + "/") for value in values):
            selected.append(int(raw_identifier))
    if not selected:
        raise RuntimeError(f"instance root did not render: {scene_root}")
    return np.isin(data, selected)


def _json_safe(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _articulation_root_path(openarm_prim) -> str:
    from pxr import Usd, UsdPhysics

    roots = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(openarm_prim, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(roots) != 1:
        raise RuntimeError(f"expected exactly one official OpenArm articulation root, found {roots}")
    return roots[0]


def _world_to_robot_root(points_world_m: np.ndarray) -> np.ndarray:
    """Invert the frozen unit-scale root pose used by this isolated scene."""

    points = np.asarray(points_world_m, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
        squeeze = True
    else:
        squeeze = False
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RuntimeError("world points must be finite XYZ values")
    translation = np.array([0.15, 1.15, 0.0], dtype=np.float64)
    rotation = np.diag([-1.0, -1.0, 1.0])
    local = (points - translation) @ rotation
    return local[0] if squeeze else local


def _ik_seeds(joint_limits: np.ndarray) -> np.ndarray:
    """Frozen deterministic seed table; randomness never enters the primary masks."""

    limits = np.asarray(joint_limits, dtype=np.float64)
    if limits.shape != (7, 2):
        raise RuntimeError("official OpenArm limits must have shape (7, 2)")
    lower, upper = limits[:, 0], limits[:, 1]
    margin = np.minimum(0.08, 0.08 * (upper - lower))
    rng = np.random.default_rng(20260715)
    seeds = rng.uniform(lower + margin, upper - margin, size=(36, 7))
    home = np.array([-1.0, -1.2, 0.0, 1.15, 0.0, 0.0, -0.30])
    return np.vstack((home, seeds))


def _capture_products(annotators: Mapping[str, object], *, rt_subframes: int):
    import omni.replicator.core as rep

    rep.orchestrator.step(
        rt_subframes=max(rt_subframes, 1),
        pause_timeline=True,
        delta_time=0.0,
    )
    rgb = np.asarray(annotators["rgb"].get_data())
    depth = np.asarray(annotators["distance_to_image_plane"].get_data())
    instance = annotators["instance_segmentation"].get_data()
    semantic = annotators["semantic_segmentation"].get_data()
    _rgb_gate(rgb)
    robot_mask = _instance_mask(instance, "/World/OpenArm")
    human_mask = _instance_mask(instance, "/World/Human")
    semantic_robot = _semantic_mask(semantic, "openarm_robot")
    semantic_human = _semantic_mask(semantic, "human")
    if not np.any(robot_mask) or not np.any(human_mask):
        raise RuntimeError("official robot/human semantic pixels disappeared from render")
    for name, instance_value, semantic_value in (
        ("robot", robot_mask, semantic_robot),
        ("human", human_mask, semantic_human),
    ):
        union = int(np.count_nonzero(instance_value | semantic_value))
        disagreement = int(np.count_nonzero(instance_value ^ semantic_value))
        if union == 0 or disagreement / union > 0.002:
            raise RuntimeError(f"{name} instance/semantic masks disagree")
    return rgb, depth, robot_mask, human_mask, instance, semantic


def _configure_replicator_explicit_capture(rep, settings) -> dict:
    setter = getattr(rep.orchestrator, "set_capture_on_play", None)
    if not callable(setter):
        raise RuntimeError("Replicator capture-on-play control is unavailable")
    setter(False)
    if settings.get("/omni/replicator/captureOnPlay") is not False:
        raise RuntimeError("Replicator capture-on-play could not be disabled")
    return {
        "capture_on_play": False,
        "capture_trigger": "explicit_zero_delta_orchestrator_step_only",
    }


def _capture_products_zero_time(
    world,
    timeline,
    annotators: Mapping[str, object],
    *,
    rt_subframes: int,
    label: str,
):
    return _run_zero_time_operation(
        world,
        timeline,
        lambda: _capture_products(annotators, rt_subframes=rt_subframes),
        label=label,
    )


def _decision_surface() -> geometry.DecisionSurface:
    return geometry.DecisionSurface(
        min_relative_geometry_weight=0.05,
        max_depth_consistency_sigma=3.0,
        min_second_spatial_eigenvalue_m2=2.0e-4,
        min_hull_area_m2=1.0e-3,
        min_spread_m=8.0e-2,
        max_condition_number=40.0,
        max_residual_m=4.0e-3,
        max_target_leverage=4.0,
        max_target_error_p95_m=1.0e-2,
    )


def _rendered_anchor_observations(
    *,
    projected_pixels: np.ndarray,
    depth_m: np.ndarray,
    human_mask: np.ndarray,
    robot_mask: np.ndarray,
    intrinsics: Mapping[str, float],
    support_radius_px: int,
    robot_mask_dilation_px: int,
) -> dict:
    """Back-project nearest unoccluded human-instance depth for each anchor."""

    projected = np.asarray(projected_pixels, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    human = np.asarray(human_mask, dtype=bool)
    robot = np.asarray(robot_mask, dtype=bool)
    if (
        projected.ndim != 2
        or projected.shape[1:] != (2,)
        or not np.isfinite(projected).all()
        or depth.ndim != 2
        or human.shape != depth.shape
        or robot.shape != depth.shape
        or type(support_radius_px) is not int
        or support_radius_px <= 0
        or type(robot_mask_dilation_px) is not int
        or robot_mask_dilation_px < 0
        or not isinstance(intrinsics, Mapping)
    ):
        raise RuntimeError("rendered anchor observation input is invalid")
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("rendered anchor observation intrinsics are invalid") from exc
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0.0 or fy <= 0.0:
        raise RuntimeError("rendered anchor observation intrinsics are invalid")

    dilated_robot = robot.copy()
    radius = robot_mask_dilation_px
    height, width = depth.shape
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            if delta_x * delta_x + delta_y * delta_y > radius * radius:
                continue
            source_y0 = max(0, -delta_y)
            source_y1 = min(height, height - delta_y)
            source_x0 = max(0, -delta_x)
            source_x1 = min(width, width - delta_x)
            target_y0 = source_y0 + delta_y
            target_y1 = source_y1 + delta_y
            target_x0 = source_x0 + delta_x
            target_x1 = source_x1 + delta_x
            dilated_robot[target_y0:target_y1, target_x0:target_x1] |= robot[
                source_y0:source_y1, source_x0:source_x1
            ]

    count = len(projected)
    camera_points = np.full((count, 3), np.nan, dtype=np.float64)
    sample_pixels = np.full((count, 2), np.nan, dtype=np.float64)
    sample_distance = np.full(count, np.nan, dtype=np.float64)
    sample_depth = np.full(count, np.nan, dtype=np.float64)
    available = np.zeros(count, dtype=bool)
    for point_index, (projected_x, projected_y) in enumerate(projected):
        x0 = max(0, int(math.floor(projected_x - support_radius_px)))
        x1 = min(width, int(math.ceil(projected_x + support_radius_px)) + 1)
        y0 = max(0, int(math.floor(projected_y - support_radius_px)))
        y1 = min(height, int(math.ceil(projected_y + support_radius_px)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
        distance_squared = np.square(grid_x - projected_x) + np.square(
            grid_y - projected_y
        )
        candidate = (
            (distance_squared <= float(support_radius_px * support_radius_px))
            & human[y0:y1, x0:x1]
            & ~dilated_robot[y0:y1, x0:x1]
            & np.isfinite(depth[y0:y1, x0:x1])
            & (depth[y0:y1, x0:x1] > 0.0)
        )
        candidate_y, candidate_x = np.nonzero(candidate)
        if len(candidate_y) == 0:
            continue
        global_y = candidate_y + y0
        global_x = candidate_x + x0
        distances = distance_squared[candidate_y, candidate_x]
        order = np.lexsort((global_x, global_y, distances))
        selected = int(order[0])
        pixel_y = int(global_y[selected])
        pixel_x = int(global_x[selected])
        observed_depth = float(depth[pixel_y, pixel_x])
        camera_points[point_index] = [
            (float(pixel_x) - cx) * observed_depth / fx,
            (float(pixel_y) - cy) * observed_depth / fy,
            observed_depth,
        ]
        sample_pixels[point_index] = [float(pixel_x), float(pixel_y)]
        sample_distance[point_index] = math.sqrt(float(distances[selected]))
        sample_depth[point_index] = observed_depth
        available[point_index] = True
    return {
        "source": "nearest_unoccluded_human_instance_depth_sample",
        "camera_points_m": camera_points,
        "available": available,
        "sample_pixels": sample_pixels,
        "sample_distance_px": sample_distance,
        "sample_depth_m": sample_depth,
        "support_radius_px": support_radius_px,
        "robot_mask_dilation_px": robot_mask_dilation_px,
    }


def _evaluate_geometry(
    camera_points: np.ndarray,
    visibility: Sequence[experiment.RenderedVisibility],
    target_index: int,
    *,
    observed_camera_points: np.ndarray,
    observed_valid: np.ndarray,
):
    points = np.asarray(camera_points, dtype=np.float64)
    observed = np.asarray(observed_camera_points, dtype=np.float64)
    valid = np.asarray(observed_valid)
    if (
        observed.shape != points.shape
        or valid.shape != (len(points),)
        or valid.dtype != np.bool_
        or np.any(valid & ~np.isfinite(observed).all(axis=1))
        or np.any(valid & (observed[:, 2] <= 0.0))
    ):
        raise RuntimeError("rendered geometry observations are invalid")
    safe_observed = points.copy()
    safe_observed[valid] = observed[valid]
    confidence = np.asarray(
        [row.visible_fraction for row in visibility], dtype=np.float64
    ) * valid.astype(np.float64)
    lateral = 0.0015 + 0.0015 * (1.0 - confidence)
    axial = 0.0030 + 0.0030 * (1.0 - confidence)
    return geometry.evaluate_visibility_geometry(
        points,
        safe_observed,
        points[target_index],
        depth_m=safe_observed[:, 2].copy(),
        mask_confidence=confidence,
        lateral_noise_std_m=lateral,
        axial_noise_std_m=axial,
        robust_delta_m=0.004,
        decision_surface=_decision_surface(),
    )


def _rendered_geometry_inlier_selection(
    reference_camera_points: np.ndarray,
    visibility: Sequence[experiment.RenderedVisibility],
    observed_camera_points: np.ndarray,
    observed_valid: np.ndarray,
) -> dict:
    """Trim gross rendered-depth correspondences after one robust SE(3) fit.

    The 6 mm inclusive threshold is 1.5 times the existing 4 mm Huber delta.
    It is a correspondence residual gate, never a fixed point-count rule.  The
    selected points must still pass the complete, unchanged decision surface in
    a separate refit before any target can receive a DIRECT update.
    """

    reference = np.asarray(reference_camera_points, dtype=np.float64)
    observed = np.asarray(observed_camera_points, dtype=np.float64)
    valid = np.asarray(observed_valid)
    point_count = len(experiment.POINT_NAMES)
    if (
        reference.shape != (point_count, 3)
        or observed.shape != reference.shape
        or valid.shape != (point_count,)
        or valid.dtype != np.bool_
        or len(visibility) != point_count
        or not np.isfinite(reference).all()
        or np.any(valid & ~np.isfinite(observed).all(axis=1))
        or np.any(valid & (observed[:, 2] <= 0.0))
    ):
        raise RuntimeError("rendered geometry inlier input is invalid")
    confidence = np.asarray(
        [row.visible_fraction for row in visibility], dtype=np.float64
    )
    if (
        confidence.shape != (point_count,)
        or not np.isfinite(confidence).all()
        or np.any(confidence < 0.0)
        or np.any(confidence > 1.0)
    ):
        raise RuntimeError("rendered geometry inlier visibility is invalid")

    initial_fit = _evaluate_geometry(
        reference,
        visibility,
        0,
        observed_camera_points=observed,
        observed_valid=valid,
    )
    transform = np.asarray(getattr(initial_fit, "transform", None), dtype=np.float64)
    rotation = transform[:3, :3] if transform.shape == (4, 4) else np.empty((0, 0))
    allowed_initial_failures = {
        "residual_above_surface",
        "target_uncertainty_above_surface",
    }
    raw_failures = getattr(initial_fit, "failure_reasons", ())
    fit_recoverable = (
        transform.shape == (4, 4)
        and np.isfinite(transform).all()
        and np.allclose(
            transform[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1.0e-12
        )
        and rotation.shape == (3, 3)
        and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-8)
        and math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-8)
        and set(raw_failures).issubset(allowed_initial_failures)
        and (
            not hasattr(initial_fit, "effective_support")
            or float(initial_fit.effective_support) >= 3.0
        )
    )
    threshold_m = 0.006
    residual_m = np.full(point_count, np.nan, dtype=np.float64)
    inlier_mask = np.zeros(point_count, dtype=bool)
    eligible = valid & (confidence > 0.0)
    if fit_recoverable and int(np.count_nonzero(eligible)) >= 3:
        fitted = (rotation @ reference.T).T + transform[:3, 3]
        residual_m[eligible] = np.linalg.norm(
            observed[eligible] - fitted[eligible], axis=1
        )
        inlier_mask = (
            eligible
            & np.isfinite(residual_m)
            & (residual_m <= threshold_m)
        )
    return {
        "initial_fit": initial_fit,
        "residual_m": residual_m,
        "inlier_mask": inlier_mask,
        "threshold_m": threshold_m,
    }


def _target_direct_update_gate(
    *,
    geometry_surface_accepted: bool,
    target_rendered_observation_available: bool,
    target_geometry_inlier: bool,
    target_visible_fraction: float,
    target_robot_front_fraction: float,
    target_valid_depth_fraction: float,
) -> dict:
    """Authorize a fresh estimate for one point, never from other points' support."""

    if (
        type(geometry_surface_accepted) is not bool
        or type(target_rendered_observation_available) is not bool
        or type(target_geometry_inlier) is not bool
        or (
            target_geometry_inlier
            and not target_rendered_observation_available
        )
    ):
        raise RuntimeError("target direct-update geometry evidence must be a bool")
    values = np.asarray(
        [
            target_visible_fraction,
            target_robot_front_fraction,
            target_valid_depth_fraction,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError("target direct-update fractions must be finite in [0, 1]")
    minimum_visible_fraction = 0.35
    maximum_robot_front_fraction = 0.25
    minimum_valid_depth_fraction = 0.35
    reasons = []
    if not geometry_surface_accepted:
        reasons.append("geometry_surface_rejected")
    if not target_rendered_observation_available:
        reasons.append("target_rendered_depth_observation_unavailable")
    elif not target_geometry_inlier:
        reasons.append("target_rendered_geometry_outlier")
    if target_visible_fraction < minimum_visible_fraction:
        reasons.append("target_visible_support_below_threshold")
    if target_robot_front_fraction >= maximum_robot_front_fraction:
        reasons.append("target_robot_front_occlusion")
    if target_valid_depth_fraction < minimum_valid_depth_fraction:
        reasons.append("target_valid_depth_below_threshold")
    return {
        "accepted": not reasons,
        "geometry_surface_accepted": geometry_surface_accepted,
        "target_rendered_observation_available": (
            target_rendered_observation_available
        ),
        "target_geometry_inlier": target_geometry_inlier,
        "target_visible_fraction": float(target_visible_fraction),
        "target_robot_front_fraction": float(target_robot_front_fraction),
        "target_valid_depth_fraction": float(target_valid_depth_fraction),
        "minimum_visible_fraction": minimum_visible_fraction,
        "maximum_robot_front_fraction_exclusive": maximum_robot_front_fraction,
        "minimum_valid_depth_fraction": minimum_valid_depth_fraction,
        "failure_reasons": reasons,
        "fixed_visible_point_count_rule_used": False,
    }


def _geometry_target_estimate_camera_m(
    reference_camera_point_m: np.ndarray,
    geometry_result,
) -> np.ndarray:
    """Apply the fitted SE(3) result instead of returning the oracle anchor."""

    point = np.asarray(reference_camera_point_m, dtype=np.float64)
    transform = np.asarray(getattr(geometry_result, "transform", None), dtype=np.float64)
    if (
        point.shape != (3,)
        or not np.isfinite(point).all()
        or transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(
            transform[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1.0e-12
        )
    ):
        raise RuntimeError("geometry target estimate transform is invalid")
    estimate = transform[:3, :3] @ point + transform[:3, 3]
    if estimate.shape != (3,) or not np.isfinite(estimate).all():
        raise RuntimeError("geometry target estimate is invalid")
    return estimate


def _advance_point_lock_policy(
    *,
    time_s: float,
    current_camera_point_m: np.ndarray,
    current_target_error_p95_m: float,
    direct_gate: Mapping[str, object],
    last_valid_time_s: Optional[float],
    last_valid_error_p95_m: Optional[float],
    last_valid_camera_point_m: Optional[np.ndarray],
) -> dict:
    """Advance one point without coupling visibility state to robot motion."""

    current_point = np.asarray(current_camera_point_m, dtype=np.float64)
    if (
        current_point.shape != (3,)
        or not math.isfinite(time_s)
        or time_s < 0.0
        or not isinstance(direct_gate, Mapping)
        or type(direct_gate.get("accepted")) is not bool
    ):
        raise RuntimeError("per-point lock policy input is invalid")
    history = (
        last_valid_time_s,
        last_valid_error_p95_m,
        last_valid_camera_point_m,
    )
    history_present = [item is not None for item in history]
    if any(history_present) and not all(history_present):
        raise RuntimeError("per-point lock policy history is incomplete")

    patient_motion_bound_m_s = 0.010
    max_freeze_age_s = 0.40
    max_freeze_error_p95_m = 0.010
    validated_last_time: Optional[float] = None
    validated_last_error: Optional[float] = None
    validated_last_point: Optional[np.ndarray] = None
    if all(history_present):
        assert last_valid_time_s is not None
        assert last_valid_error_p95_m is not None
        assert last_valid_camera_point_m is not None
        validated_last_point = np.asarray(last_valid_camera_point_m, dtype=np.float64)
        history_values = np.asarray(
            [last_valid_time_s, last_valid_error_p95_m], dtype=np.float64
        )
        if (
            validated_last_point.shape != (3,)
            or not np.isfinite(validated_last_point).all()
            or not np.isfinite(history_values).all()
            or np.any(history_values < 0.0)
            or time_s < last_valid_time_s
        ):
            raise RuntimeError("per-point lock policy history or time is invalid")
        validated_last_time = float(last_valid_time_s)
        validated_last_error = float(last_valid_error_p95_m)

    if direct_gate["accepted"]:
        if (
            not np.isfinite(current_point).all()
            or not math.isfinite(current_target_error_p95_m)
            or current_target_error_p95_m < 0.0
            or current_target_error_p95_m > max_freeze_error_p95_m
        ):
            raise RuntimeError("per-point direct estimate is invalid")
        age = 0.0
        reported_age: Optional[float] = age
        history_available = True
        next_time = float(time_s)
        next_error = float(current_target_error_p95_m)
        next_point = current_point.copy()
        decision = experiment.decide_occlusion_policy(
            geometry_accepted=True,
            last_valid_age_s=age,
            last_target_error_p95_m=next_error,
            patient_motion_bound_m_s=patient_motion_bound_m_s,
            max_freeze_age_s=max_freeze_age_s,
            max_freeze_error_p95_m=max_freeze_error_p95_m,
        )
    elif not all(history_present):
        age = max_freeze_age_s + 0.01
        reported_age = None
        history_available = False
        next_time = None
        next_error = None
        next_point = None
        decision = experiment.decide_occlusion_policy(
            geometry_accepted=False,
            last_valid_age_s=age,
            last_target_error_p95_m=max_freeze_error_p95_m,
            patient_motion_bound_m_s=patient_motion_bound_m_s,
            max_freeze_age_s=max_freeze_age_s,
            max_freeze_error_p95_m=max_freeze_error_p95_m,
        )
    else:
        assert validated_last_time is not None
        assert validated_last_error is not None
        assert validated_last_point is not None
        age = float(time_s - validated_last_time)
        reported_age = age
        history_available = True
        next_time = validated_last_time
        next_error = validated_last_error
        next_point = validated_last_point.copy()
        decision = experiment.decide_occlusion_policy(
            geometry_accepted=False,
            last_valid_age_s=age,
            last_target_error_p95_m=next_error,
            patient_motion_bound_m_s=patient_motion_bound_m_s,
            max_freeze_age_s=max_freeze_age_s,
            max_freeze_error_p95_m=max_freeze_error_p95_m,
        )

    if decision.state == "DIRECT_GEOMETRY_UPDATE":
        estimate = current_point.copy()
        estimate_valid = True
        estimate_source = "current_rendered_geometry"
    elif decision.state == "BOUNDED_PATIENT_FRAME_FREEZE":
        if next_point is None:
            raise RuntimeError("per-point freeze has no last-valid 3D estimate")
        estimate = next_point.copy()
        estimate_valid = True
        estimate_source = "last_valid_patient_frame_lock"
    elif decision.state == "LOST":
        estimate = np.full(3, np.nan, dtype=np.float64)
        estimate_valid = False
        estimate_source = "none_fail_closed"
    else:
        raise RuntimeError("per-point lock policy returned an unknown state")
    return {
        "policy_state": decision.state,
        "predicted_error_p95_m": float(decision.predicted_error_p95_m),
        "policy_reason": (
            "no_last_valid_estimate" if not history_available else decision.reason
        ),
        "last_valid_age_s": reported_age,
        "history_available": history_available,
        "estimate_camera_m": estimate,
        "estimate_valid": estimate_valid,
        "estimate_source": estimate_source,
        "next_last_valid_time_s": next_time,
        "next_last_valid_error_p95_m": next_error,
        "next_last_valid_camera_point_m": next_point,
        "patient_motion_bound_m_s": patient_motion_bound_m_s,
        "max_freeze_age_s": max_freeze_age_s,
        "max_freeze_error_p95_m": max_freeze_error_p95_m,
        "fixed_visible_point_count_rule_used": False,
        "robot_motion_policy": "unchanged_by_point_visibility_policy",
    }


def _holdout_parameter_trials(
    camera_points: np.ndarray,
    visibility: Sequence[experiment.RenderedVisibility],
    *,
    target_index: int,
    seed: int,
    trials_per_stratum: int,
) -> dict:
    """Frozen independent perturbation trials; these are not independent trajectories."""

    points = np.asarray(camera_points, dtype=np.float64)
    if points.shape != (10, 3) or not np.isfinite(points).all():
        raise RuntimeError("holdout sensitivity requires ten finite camera-frame points")
    if type(target_index) is not int or target_index < 0 or target_index >= len(points):
        raise RuntimeError("holdout sensitivity target index is invalid")
    if type(seed) is not int or seed < 0 or type(trials_per_stratum) is not int or trials_per_stratum < 2:
        raise RuntimeError("holdout sensitivity seed/trial count is invalid")
    confidence = np.asarray([row.visible_fraction for row in visibility], dtype=np.float64)
    if confidence.shape != (10,) or not np.isfinite(confidence).all():
        raise RuntimeError("holdout sensitivity visibility is invalid")
    base_lateral = 0.0015 + 0.0015 * (1.0 - confidence)
    base_axial = 0.0030 + 0.0030 * (1.0 - confidence)
    strata = (
        ("nominal_declared_noise", 1.0, 1.0, np.zeros(3)),
        ("noise_scale_x2", 2.0, 2.0, np.zeros(3)),
        ("structured_lateral_bias_plus3mm", 1.0, 1.0, np.array([0.003, 0.0, 0.0])),
        ("structured_depth_bias_plus5mm", 1.0, 1.0, np.array([0.0, 0.0, 0.005])),
    )
    rng = np.random.default_rng(seed)
    rows = []
    for name, lateral_scale, axial_scale, bias in strata:
        lateral = base_lateral * lateral_scale
        axial = base_axial * axial_scale
        errors = []
        accepted = []
        reported = []
        for _ in range(trials_per_stratum):
            noise = np.column_stack(
                (
                    rng.normal(0.0, lateral),
                    rng.normal(0.0, lateral),
                    rng.normal(0.0, axial),
                )
            )
            observed = points + noise + bias
            result = geometry.evaluate_visibility_geometry(
                points,
                observed,
                points[target_index],
                depth_m=observed[:, 2].copy(),
                mask_confidence=confidence,
                lateral_noise_std_m=lateral,
                axial_noise_std_m=axial,
                robust_delta_m=0.004 * max(lateral_scale, axial_scale),
                decision_surface=_decision_surface(),
            )
            predicted = (
                result.transform[:3, :3] @ points[target_index]
                + result.transform[:3, 3]
            )
            errors.append(float(np.linalg.norm(predicted - points[target_index])))
            accepted.append(bool(result.accepted))
            reported.append(float(result.target_error_p95_m))
        error_array = np.asarray(errors, dtype=np.float64)
        bootstrap = []
        for _ in range(300):
            sample = rng.choice(error_array, size=len(error_array), replace=True)
            bootstrap.append(float(np.percentile(sample, 95.0)))
        false_accept = [
            decision and error > 0.010 for decision, error in zip(accepted, errors)
        ]
        rows.append(
            {
                "name": name,
                "trial_count": trials_per_stratum,
                "error_p50_m": float(np.percentile(error_array, 50.0)),
                "error_p90_m": float(np.percentile(error_array, 90.0)),
                "error_p95_m": float(np.percentile(error_array, 95.0)),
                "error_p99_m": float(np.percentile(error_array, 99.0)),
                "bootstrap_error_p95_95ci_m": [
                    float(np.percentile(bootstrap, 2.5)),
                    float(np.percentile(bootstrap, 97.5)),
                ],
                "accepted_fraction_on_declared_surface": float(np.mean(accepted)),
                "false_accept_fraction_against_10mm_reference_only": float(
                    np.mean(false_accept)
                ),
                "median_reported_target_p95_m": (
                    float(np.median([value for value in reported if math.isfinite(value)]))
                    if any(math.isfinite(value) for value in reported)
                    else None
                ),
                "bias_m": bias.tolist(),
                "lateral_noise_scale": lateral_scale,
                "axial_noise_scale": axial_scale,
            }
        )
    return {
        "scope": "parameter_sensitivity_only",
        "physical_accuracy_pass": False,
        "independent_trajectory_claim": False,
        "trials_per_stratum": trials_per_stratum,
        "seed": seed,
        "strata": rows,
    }


def _draw_frame(
    rgb: np.ndarray,
    projected: np.ndarray,
    visibility: Sequence[experiment.RenderedVisibility],
    *,
    title: str,
    subtitle: str,
    oblique_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    import cv2

    bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (bgr.shape[1], 72), (12, 16, 22), -1)
    cv2.addWeighted(overlay, 0.78, bgr, 0.22, 0.0, bgr)
    cv2.putText(bgr, title, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    cv2.putText(bgr, subtitle, (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (215, 225, 235), 1)
    for name, pixel, row in zip(experiment.POINT_NAMES, projected, visibility):
        color = (60, 220, 60)
        if row.robot_front_fraction >= 0.25:
            color = (40, 40, 235)
        elif row.visible_fraction < 0.35:
            color = (40, 210, 235)
        center = tuple(int(round(value)) for value in pixel)
        cv2.circle(bgr, center, 7, (12, 12, 12), 3)
        cv2.circle(bgr, center, 5, color, -1)
        cv2.putText(
            bgr,
            name,
            (center[0] + 7, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
        )
    if oblique_rgb is not None:
        oblique = np.asarray(oblique_rgb)
        _rgb_gate(oblique)
        inset_width = max(160, bgr.shape[1] // 3)
        inset_height = max(90, bgr.shape[0] // 3)
        inset = cv2.resize(
            cv2.cvtColor(oblique[..., :3], cv2.COLOR_RGB2BGR),
            (inset_width, inset_height),
            interpolation=cv2.INTER_AREA,
        )
        x0 = bgr.shape[1] - inset_width - 10
        y0 = bgr.shape[0] - inset_height - 10
        bgr[y0 : y0 + inset_height, x0 : x0 + inset_width] = inset
        cv2.rectangle(
            bgr,
            (x0 - 2, y0 - 22),
            (x0 + inset_width + 2, y0 + inset_height + 2),
            (255, 255, 255),
            2,
        )
        cv2.putText(
            bgr,
            "OBLIQUE VIEW (reference only)",
            (x0 + 4, y0 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
        )
    return bgr


def _validate_smoke_anchor_alignment(
    projected_pixels: np.ndarray,
    visibility: Sequence[experiment.RenderedVisibility],
    *,
    width: int,
    height: int,
) -> dict:
    pixels = np.asarray(projected_pixels, dtype=np.float64)
    if pixels.shape != (10, 2) or len(visibility) != 10 or not np.isfinite(pixels).all():
        raise RuntimeError("smoke anchor products are incomplete")
    inside = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    coverage = np.asarray(
        [row.visible_fraction + row.robot_front_fraction for row in visibility],
        dtype=np.float64,
    )
    aligned = inside & np.isfinite(coverage) & (coverage >= 0.25)
    count = int(np.count_nonzero(aligned))
    if count < 8:
        raise RuntimeError(
            f"only {count}/10 anchors align with the rendered official back"
        )
    return {
        "aligned_anchor_count": count,
        "inside_image_count": int(np.count_nonzero(inside)),
        "minimum_required": 8,
        "coverage_threshold": 0.25,
    }


def _add_exception_note(primary: BaseException, *, label: str, error: BaseException) -> None:
    text = f"{label}: {type(error).__name__}: {error}"
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(text)
        return
    try:
        fallback = list(getattr(primary, "_wam_secondary_failures", []))
        fallback.append(text)
        setattr(primary, "_wam_secondary_failures", fallback)
    except BaseException:
        pass


def _raise_first_failure(
    failures: Sequence[tuple[str, BaseException, object]],
) -> None:
    if not failures:
        return
    _label, primary, traceback = failures[0]
    for label, error, _traceback in failures[1:]:
        _add_exception_note(primary, label=label, error=error)
    raise primary.with_traceback(traceback)  # type: ignore[arg-type]


def _runtime_timing_snapshot(world, timeline) -> dict:
    snapshot = {
        "world_time_s": float(world.current_time),
        "physics_step_index": int(world.current_time_step_index),
        "timeline_time_s": float(timeline.get_current_time()),
        "physics_dt_s": float(world.get_physics_dt()),
        "rendering_dt_s": float(world.get_rendering_dt()),
        "world_playing": bool(world.is_playing()),
        "world_stopped": bool(world.is_stopped()),
        "timeline_playing": bool(timeline.is_playing()),
        "timeline_stopped": bool(timeline.is_stopped()),
        "timeline_auto_updating": bool(timeline.is_auto_updating()),
    }
    finite_values = (
        snapshot["world_time_s"],
        snapshot["timeline_time_s"],
        snapshot["physics_dt_s"],
        snapshot["rendering_dt_s"],
    )
    if (
        not all(math.isfinite(float(value)) for value in finite_values)
        or snapshot["physics_dt_s"] <= 0.0
        or snapshot["rendering_dt_s"] <= 0.0
        or snapshot["physics_step_index"] < 0
    ):
        raise RuntimeError("Isaac runtime timing state is invalid")
    return snapshot


def _require_paused_running_timeline(snapshot: Mapping[str, object], *, label: str) -> None:
    if snapshot["world_playing"] or snapshot["timeline_playing"]:
        raise RuntimeError(f"{label} requires a paused timeline")
    if snapshot["world_stopped"] or snapshot["timeline_stopped"]:
        raise RuntimeError(f"{label} requires a paused, non-stopped timeline")


def _run_zero_time_operation(
    world,
    timeline,
    operation: Callable[[], object],
    *,
    label: str,
):
    """Run a render/capture operation and prove it did not advance either clock."""

    before = _runtime_timing_snapshot(world, timeline)
    _require_paused_running_timeline(before, label=label)
    failures: list[tuple[str, BaseException, object]] = []
    result = None
    try:
        result = operation()
    except BaseException as exc:
        failures.append((f"{label} failed", exc, exc.__traceback__))
    after = None
    try:
        after = _runtime_timing_snapshot(world, timeline)
        _require_paused_running_timeline(after, label=f"{label} postcondition")
    except BaseException as exc:
        failures.append((f"{label} timing audit failed", exc, exc.__traceback__))
    _raise_first_failure(failures)
    assert after is not None

    world_time_delta = float(after["world_time_s"]) - float(before["world_time_s"])
    timeline_time_delta = float(after["timeline_time_s"]) - float(
        before["timeline_time_s"]
    )
    step_index_delta = int(after["physics_step_index"]) - int(
        before["physics_step_index"]
    )
    if not math.isclose(world_time_delta, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"{label} advanced Isaac physics time")
    if step_index_delta != 0:
        raise RuntimeError(f"{label} advanced the physics step index")
    if not math.isclose(timeline_time_delta, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"{label} advanced the USD timeline")
    physics_dt_unchanged = math.isclose(
        float(after["physics_dt_s"]),
        float(before["physics_dt_s"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    rendering_dt_unchanged = math.isclose(
        float(after["rendering_dt_s"]),
        float(before["rendering_dt_s"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    if not physics_dt_unchanged or not rendering_dt_unchanged:
        raise RuntimeError(f"{label} changed the simulation timestep contract")
    timeline_auto_update_unchanged = (
        after["timeline_auto_updating"] == before["timeline_auto_updating"]
    )
    if not timeline_auto_update_unchanged:
        raise RuntimeError(f"{label} changed timeline auto-update state")
    return result, {
        "label": label,
        "world_time_delta_s": world_time_delta,
        "physics_step_index_delta": step_index_delta,
        "timeline_time_delta_s": timeline_time_delta,
        "physics_dt_unchanged": physics_dt_unchanged,
        "rendering_dt_unchanged": rendering_dt_unchanged,
        "timeline_auto_update_unchanged": timeline_auto_update_unchanged,
        "timeline_paused_after": True,
        "timeline_stopped_after": False,
    }


def _synchronize_articulation_render_state(world, *, timeline) -> dict:
    """Flush an authored articulation pose to Fabric without physics integration."""

    before = _runtime_timing_snapshot(world, timeline)
    _require_paused_running_timeline(before, label="render synchronization")
    original_physics_dt = float(before["physics_dt_s"])
    original_rendering_dt = float(before["rendering_dt_s"])
    original_timeline_auto_update = bool(before["timeline_auto_updating"])
    if not original_timeline_auto_update:
        raise RuntimeError("render synchronization requires timeline auto-update enabled at entry")
    failures: list[tuple[str, BaseException, object]] = []
    try:
        timeline.set_auto_update(False)
        timeline.commit_silently()
        if timeline.is_auto_updating():
            raise RuntimeError("Isaac timeline auto-update remained enabled")
        world.set_simulation_dt(physics_dt=0.0)
        if float(world.get_physics_dt()) != 0.0:
            raise RuntimeError("Isaac did not accept the zero-dt synchronization gate")
        world.play()
        if not world.is_playing() or world.is_stopped():
            raise RuntimeError("Isaac timeline did not enter synchronization mode")
        world.step(render=False, step_sim=True, update_fabric=True)
    except BaseException as exc:
        failures.append(("render synchronization failed", exc, exc.__traceback__))

    try:
        world.pause()
    except BaseException as exc:
        failures.append(("could not pause after render synchronization", exc, exc.__traceback__))
    try:
        world.set_simulation_dt(physics_dt=original_physics_dt)
    except BaseException as exc:
        failures.append(("could not restore physics dt", exc, exc.__traceback__))
    try:
        timeline.set_auto_update(original_timeline_auto_update)
    except BaseException as exc:
        failures.append(
            ("could not restore timeline auto-update", exc, exc.__traceback__)
        )
    try:
        timeline.commit_silently()
    except BaseException as exc:
        failures.append(
            ("could not commit timeline auto-update restoration", exc, exc.__traceback__)
        )
    after_flush = None
    step_delta = None
    try:
        after_flush = _runtime_timing_snapshot(world, timeline)
        _require_paused_running_timeline(
            after_flush, label="render synchronization cleanup audit"
        )
        step_delta = int(after_flush["physics_step_index"]) - int(
            before["physics_step_index"]
        )
        cleanup_errors = []
        if step_delta != 0:
            cleanup_errors.append("physics step index advanced")
        if not math.isclose(
            float(after_flush["world_time_s"]),
            float(before["world_time_s"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            cleanup_errors.append("physics time advanced")
        if not math.isclose(
            float(after_flush["timeline_time_s"]),
            float(before["timeline_time_s"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            cleanup_errors.append("USD timeline advanced")
        if not math.isclose(
            float(after_flush["physics_dt_s"]),
            original_physics_dt,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            cleanup_errors.append("physics dt was not restored")
        if not math.isclose(
            float(after_flush["rendering_dt_s"]),
            original_rendering_dt,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            cleanup_errors.append("rendering dt changed")
        if after_flush["timeline_auto_updating"] is not True:
            cleanup_errors.append("timeline auto-update was not restored")
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))
    except BaseException as exc:
        failures.append(
            ("render synchronization cleanup audit failed", exc, exc.__traceback__)
        )
    _raise_first_failure(failures)
    assert after_flush is not None and step_delta == 0

    _unused, render_gate = _run_zero_time_operation(
        world,
        timeline,
        world.render,
        label="world_render_after_fabric_flush",
    )
    after = _runtime_timing_snapshot(world, timeline)
    return {
        "method": "zero_physics_dt_step_update_fabric_then_render",
        "original_physics_dt_s": original_physics_dt,
        "restored_physics_dt_s": float(after["physics_dt_s"]),
        "original_rendering_dt_s": original_rendering_dt,
        "restored_rendering_dt_s": float(after["rendering_dt_s"]),
        "simulation_time_before_s": float(before["world_time_s"]),
        "simulation_time_after_s": float(after["world_time_s"]),
        "physics_step_index_before": int(before["physics_step_index"]),
        "physics_step_index_after": int(after["physics_step_index"]),
        "physics_step_index_delta": step_delta,
        "fabric_flush_call_count": 1,
        "timeline_time_before_s": float(before["timeline_time_s"]),
        "timeline_time_after_s": float(after["timeline_time_s"]),
        "timeline_auto_update_before_sync": original_timeline_auto_update,
        "timeline_auto_update_after_sync": bool(after["timeline_auto_updating"]),
        "timeline_auto_update_disabled_during_flush": True,
        "timeline_paused_after_sync": True,
        "timeline_stopped_after_sync": False,
        "physics_integration_used": False,
        "claim_scope": "visual_pose_and_fabric_flush_only",
        "render_gate": render_gate,
    }


def _validated_render_sync_evidence(
    *,
    initial_robot_mask: np.ndarray,
    moved_robot_mask: np.ndarray,
    initial_human_mask: np.ndarray,
    moved_human_mask: np.ndarray,
    restored_robot_mask: np.ndarray,
    restored_human_mask: np.ndarray,
    initial_joint_positions: np.ndarray,
    commanded_joint_positions: np.ndarray,
    observed_joint_positions: np.ndarray,
    sync_evidence: Mapping[str, object],
) -> dict:
    initial_robot = np.asarray(initial_robot_mask, dtype=bool)
    moved_robot = np.asarray(moved_robot_mask, dtype=bool)
    initial_human = np.asarray(initial_human_mask, dtype=bool)
    moved_human = np.asarray(moved_human_mask, dtype=bool)
    restored_robot = np.asarray(restored_robot_mask, dtype=bool)
    restored_human = np.asarray(restored_human_mask, dtype=bool)
    if (
        initial_robot.ndim != 2
        or initial_robot.shape != moved_robot.shape
        or initial_robot.shape != restored_robot.shape
        or initial_robot.shape != initial_human.shape
        or initial_robot.shape != moved_human.shape
        or initial_robot.shape != restored_human.shape
        or not np.any(initial_robot)
        or not np.any(moved_robot)
        or not np.any(restored_robot)
        or not np.any(initial_human)
        or not np.any(moved_human)
        or not np.any(restored_human)
    ):
        raise RuntimeError("render synchronization masks are incomplete")
    initial = np.asarray(initial_joint_positions, dtype=np.float64)
    commanded = np.asarray(commanded_joint_positions, dtype=np.float64)
    observed = np.asarray(observed_joint_positions, dtype=np.float64)
    if (
        initial.shape != (7,)
        or commanded.shape != (7,)
        or observed.shape != (7,)
        or not np.isfinite(initial).all()
        or not np.isfinite(commanded).all()
        or not np.isfinite(observed).all()
    ):
        raise RuntimeError("render synchronization joint evidence is incomplete")
    joint_motion = float(np.max(np.abs(commanded - initial)))
    if joint_motion < 0.25:
        raise RuntimeError("render synchronization joint motion is too small")
    readback_error = float(np.max(np.abs(observed - commanded)))
    if readback_error > 1.0e-6:
        raise RuntimeError("render synchronization joint readback is inaccurate")
    changed_pixels = int(np.count_nonzero(initial_robot != moved_robot))
    motion_union_pixels = int(np.count_nonzero(initial_robot | moved_robot))
    minimum_changed_pixels = max(64, int(math.ceil(0.01 * motion_union_pixels)))
    if changed_pixels < minimum_changed_pixels:
        raise RuntimeError("robot mask did not follow the commanded articulation")
    restored_robot_changed_pixels = int(
        np.count_nonzero(initial_robot != restored_robot)
    )
    restored_robot_union_pixels = int(
        np.count_nonzero(initial_robot | restored_robot)
    )
    restored_robot_tolerance_pixels = max(
        8, int(math.ceil(0.001 * restored_robot_union_pixels))
    )
    if restored_robot_changed_pixels > restored_robot_tolerance_pixels:
        raise RuntimeError("robot mask did not return to the initial pose")
    restored_human_changed_pixels = int(
        np.count_nonzero(initial_human != restored_human)
    )
    restored_human_union_pixels = int(
        np.count_nonzero(initial_human | restored_human)
    )
    restored_human_tolerance_pixels = max(
        8, int(math.ceil(0.001 * restored_human_union_pixels))
    )
    if restored_human_changed_pixels > restored_human_tolerance_pixels:
        raise RuntimeError("human mask did not return after articulation restoration")
    try:
        original_dt = float(sync_evidence["original_physics_dt_s"])
        restored_dt = float(sync_evidence["restored_physics_dt_s"])
        original_rendering_dt = float(sync_evidence["original_rendering_dt_s"])
        restored_rendering_dt = float(sync_evidence["restored_rendering_dt_s"])
        time_before = float(sync_evidence["simulation_time_before_s"])
        time_after = float(sync_evidence["simulation_time_after_s"])
        timeline_time_before = float(sync_evidence["timeline_time_before_s"])
        timeline_time_after = float(sync_evidence["timeline_time_after_s"])
        physics_step_index_delta = sync_evidence["physics_step_index_delta"]
        fabric_flush_call_count = sync_evidence["fabric_flush_call_count"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("render synchronization timing evidence is incomplete") from exc
    if (
        sync_evidence.get("method")
        != "zero_physics_dt_step_update_fabric_then_render"
        or sync_evidence.get("timeline_paused_after_sync") is not True
        or sync_evidence.get("timeline_stopped_after_sync") is not False
        or sync_evidence.get("physics_integration_used") is not False
        or type(physics_step_index_delta) is not int
        or physics_step_index_delta != 0
        or type(fabric_flush_call_count) is not int
        or fabric_flush_call_count != 1
        or sync_evidence.get("timeline_auto_update_before_sync") is not True
        or sync_evidence.get("timeline_auto_update_after_sync") is not True
        or sync_evidence.get("timeline_auto_update_disabled_during_flush") is not True
        or sync_evidence.get("claim_scope")
        != "visual_pose_and_fabric_flush_only"
        or not math.isclose(original_dt, restored_dt, rel_tol=0.0, abs_tol=1.0e-15)
        or not math.isclose(
            original_rendering_dt,
            restored_rendering_dt,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(time_before, time_after, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(
            timeline_time_before,
            timeline_time_after,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise RuntimeError("render synchronization timing contract failed")
    return {
        **dict(sync_evidence),
        "joint_motion_max_abs_rad": joint_motion,
        "readback_max_abs_error_rad": readback_error,
        "robot_mask_changed_pixels": changed_pixels,
        "robot_mask_motion_union_pixels": motion_union_pixels,
        "human_mask_changed_pixels": int(
            np.count_nonzero(initial_human != moved_human)
        ),
        "minimum_robot_mask_changed_pixels": minimum_changed_pixels,
        "restored_robot_mask_changed_pixels": restored_robot_changed_pixels,
        "restored_robot_mask_tolerance_pixels": restored_robot_tolerance_pixels,
        "restored_human_mask_changed_pixels": restored_human_changed_pixels,
        "restored_human_mask_tolerance_pixels": restored_human_tolerance_pixels,
    }


def _validated_trajectory_mask_keyframes(
    samples: Sequence[Mapping[str, object]],
) -> dict:
    """Prove segment-endpoint silhouettes follow motion and return to home."""

    if len(samples) < 5:
        raise RuntimeError("trajectory mask keyframe evidence is incomplete")
    normalized = []
    previous_frame_index = -1
    reference_shape = None
    for sample in samples:
        try:
            frame_index = int(sample["frame_index"])
            joints = np.asarray(sample["joint_positions_rad"], dtype=np.float64)
            mask = np.asarray(sample["robot_mask"], dtype=bool)
            phase = str(sample["motion_phase"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("trajectory mask keyframe evidence is incomplete") from exc
        if (
            frame_index <= previous_frame_index
            or joints.shape != (7,)
            or not np.isfinite(joints).all()
            or mask.ndim != 2
            or not np.any(mask)
            or not phase
            or (reference_shape is not None and mask.shape != reference_shape)
        ):
            raise RuntimeError("trajectory mask keyframe evidence is invalid")
        previous_frame_index = frame_index
        reference_shape = mask.shape
        normalized.append((frame_index, joints, mask, phase))

    home_frame_index, home_joints, home_mask, _home_phase = normalized[0]
    if home_frame_index != 0:
        raise RuntimeError("trajectory mask keyframes must begin at home frame zero")
    motion_rows = []
    return_rows = []
    for frame_index, joints, mask, phase in normalized[1:]:
        joint_motion = float(np.max(np.abs(joints - home_joints)))
        if joint_motion >= 0.25:
            changed_pixels = int(np.count_nonzero(home_mask != mask))
            union_pixels = int(np.count_nonzero(home_mask | mask))
            minimum_changed_pixels = max(
                64, int(math.ceil(0.01 * union_pixels))
            )
            if changed_pixels < minimum_changed_pixels:
                raise RuntimeError(
                    f"trajectory keyframe mask did not follow frame {frame_index} motion"
                )
            motion_rows.append(
                {
                    "frame_index": frame_index,
                    "motion_phase": phase,
                    "joint_motion_from_home_max_abs_rad": joint_motion,
                    "robot_mask_changed_pixels_from_home": changed_pixels,
                    "minimum_robot_mask_changed_pixels": minimum_changed_pixels,
                    "robot_mask_union_pixels": union_pixels,
                }
            )
        if phase == "return_home":
            if joint_motion > 1.0e-6:
                raise RuntimeError("return-home keyframe joint command did not restore")
            changed_pixels = int(np.count_nonzero(home_mask != mask))
            union_pixels = int(np.count_nonzero(home_mask | mask))
            tolerance_pixels = max(8, int(math.ceil(0.001 * union_pixels)))
            if changed_pixels > tolerance_pixels:
                raise RuntimeError("return-home mask did not restore the initial silhouette")
            return_rows.append(
                {
                    "frame_index": frame_index,
                    "robot_mask_changed_pixels_from_home": changed_pixels,
                    "robot_mask_restore_tolerance_pixels": tolerance_pixels,
                    "joint_motion_from_home_max_abs_rad": joint_motion,
                }
            )
    if len(motion_rows) < 3:
        raise RuntimeError("trajectory has too few meaningful mask-motion keyframes")
    if not return_rows:
        raise RuntimeError("trajectory has no validated return-home mask keyframe")
    return {
        "scope": "segment_endpoint_keyframes_against_initial_home",
        "sampled_keyframe_count": len(normalized),
        "meaningful_motion_keyframe_count": len(motion_rows),
        "return_home_keyframe_count": len(return_rows),
        "all_meaningful_motion_keyframes_passed": True,
        "all_return_home_keyframes_passed": True,
        "motion_threshold": "max(64,ceil(0.01*union_pixels))",
        "return_tolerance": "max(8,ceil(0.001*union_pixels))",
        "motion_rows": motion_rows,
        "return_home_rows": return_rows,
    }


def _validate_joint_readback(
    articulation,
    *,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    commanded_left: np.ndarray,
    frozen_right: np.ndarray,
) -> dict:
    commanded = np.asarray(commanded_left, dtype=np.float64)
    frozen = np.asarray(frozen_right, dtype=np.float64)
    observed_left = np.asarray(
        articulation.get_joint_positions(joint_indices=left_indices), dtype=np.float64
    )
    observed_right = np.asarray(
        articulation.get_joint_positions(joint_indices=right_indices), dtype=np.float64
    )
    if observed_left.shape != (7,) or not np.isfinite(observed_left).all() or not np.allclose(
        observed_left, commanded, rtol=0.0, atol=1.0e-6
    ):
        raise RuntimeError("left joint readback differs from the rendered command")
    if observed_right.shape != (7,) or not np.isfinite(observed_right).all() or not np.allclose(
        observed_right, frozen, rtol=0.0, atol=1.0e-7
    ):
        raise RuntimeError("right reference chain drifted during the rendered trajectory")
    return {
        "observed_left": observed_left,
        "observed_right": observed_right,
        "left_max_abs_error_rad": float(np.max(np.abs(observed_left - commanded))),
        "right_max_abs_drift_rad": float(np.max(np.abs(observed_right - frozen))),
    }


def _targeted_occlusion_status(
    *,
    max_robot_front_fraction_sum: float,
    max_single_point_robot_front_fraction: float,
) -> dict:
    values = np.asarray(
        [max_robot_front_fraction_sum, max_single_point_robot_front_fraction],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise RuntimeError("targeted occlusion evidence is invalid")
    threshold = 0.25
    observed = bool(
        max_robot_front_fraction_sum >= threshold
        and max_single_point_robot_front_fraction >= threshold
    )
    return {
        "status": (
            "TARGETED_OCCLUSION_OBSERVED"
            if observed
            else "NO_TARGETED_OCCLUSION_EVIDENCE"
        ),
        "observed": observed,
        "minimum_fraction": threshold,
        "max_robot_front_fraction_sum": float(max_robot_front_fraction_sum),
        "max_single_point_robot_front_fraction": float(
            max_single_point_robot_front_fraction
        ),
    }


def _ordered_policy_trace(states: Sequence[str]) -> bool:
    """Require one direct -> bounded-freeze -> lost episode without reacquisition."""

    direct = "DIRECT_GEOMETRY_UPDATE"
    freeze = "BOUNDED_PATIENT_FRAME_FREEZE"
    lost = "LOST"
    phase = 0
    for state in states:
        if state not in {direct, freeze, lost}:
            raise RuntimeError(f"unknown policy state in trace: {state!r}")
        if phase == 0:
            if state == direct:
                phase = 1
        elif phase == 1:
            if state == freeze:
                phase = 2
            elif state == lost:
                phase = 0
        else:
            if state == lost:
                return True
            if state == direct:
                phase = 1
    return False


def _targeted_policy_episode(rows: Sequence[Mapping[str, object]]) -> dict:
    """Bind one direct -> freeze -> lost episode to one occluded target/wrist.

    A qualifying episode is contiguous in target/wrist identity, contains no
    direct reacquisition after entering the freeze phase, and contains at least
    one frame where both robot-front occlusion measures meet the threshold.
    """

    direct = "DIRECT_GEOMETRY_UPDATE"
    freeze = "BOUNDED_PATIENT_FRAME_FREEZE"
    lost = "LOST"
    valid_states = {direct, freeze, lost}
    threshold = 0.25

    phase = 0
    episode_key: tuple[str, int] | None = None
    episode_indices: list[int] = []
    episode_has_occlusion = False

    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError("policy evidence row must be a mapping")
        try:
            target_name = str(row["target_name"])
            wrist_index = int(row["wrist_index"])
            state = str(row["policy_state"])
            front_sum = float(row["robot_front_fraction_sum"])
            max_single = float(row["max_single_point_robot_front_fraction"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("policy evidence row is invalid") from exc
        values = np.asarray([front_sum, max_single], dtype=np.float64)
        if (
            not target_name
            or wrist_index < 0
            or state not in valid_states
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
        ):
            raise RuntimeError("policy evidence row is invalid")

        key = (target_name, wrist_index)
        frame_index = int(row.get("frame_index", position))
        if frame_index < 0:
            raise RuntimeError("policy evidence frame index is invalid")
        frame_is_occluded = bool(front_sum >= threshold and max_single >= threshold)

        if episode_key != key:
            phase = 0
            episode_key = key
            episode_indices = []
            episode_has_occlusion = False

        if phase == 0:
            if state == direct:
                phase = 1
                episode_indices = [frame_index]
                episode_has_occlusion = frame_is_occluded
        elif phase == 1:
            episode_indices.append(frame_index)
            episode_has_occlusion = episode_has_occlusion or frame_is_occluded
            if state == freeze:
                phase = 2
            elif state == lost:
                phase = 0
                episode_indices = []
                episode_has_occlusion = False
        else:
            if state == direct:
                # Direct geometry is a reacquisition, so it starts a new episode.
                phase = 1
                episode_indices = [frame_index]
                episode_has_occlusion = frame_is_occluded
            else:
                episode_indices.append(frame_index)
                episode_has_occlusion = episode_has_occlusion or frame_is_occluded
                if state == lost:
                    if episode_has_occlusion:
                        return {
                            "observed": True,
                            "minimum_fraction": threshold,
                            "target_name": target_name,
                            "wrist_index": wrist_index,
                            "frame_indices": episode_indices,
                        }
                    phase = 0
                    episode_indices = []
                    episode_has_occlusion = False

    return {
        "observed": False,
        "minimum_fraction": threshold,
        "target_name": None,
        "wrist_index": None,
        "frame_indices": [],
    }


def _targeted_point_policy_episode(rows: Sequence[Mapping[str, object]]) -> dict:
    """Find a point-specific direct/freeze/lost trace within one motion context."""

    groups: dict[tuple[str, str, int, int], list[dict]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError("point policy evidence row must be a mapping")
        try:
            point_name = str(row["point_name"])
            point_index = int(row["point_index"])
            trajectory_target = str(row["trajectory_target_name"])
            wrist_index = int(row["wrist_index"])
            trajectory_group_index = int(row["trajectory_group_index"])
            frame_index = int(row.get("frame_index", position))
            target_front = float(row["target_robot_front_fraction"])
            state = str(row["policy_state"])
            policy_reason = str(row["policy_reason"])
            predicted_error = float(row["predicted_error_p95_m"])
            last_valid_age = row["last_valid_age_s"]
            direct_accepted = row["direct_update_accepted"]
            rendered_observation_available = row[
                "target_rendered_observation_available"
            ]
            target_geometry_inlier = row["target_geometry_inlier"]
            history_available = row["history_available"]
            estimate_valid = row["estimate_valid"]
            estimate_source = str(row["estimate_source"])
            estimate_values = (
                row["estimate_camera_x_m"],
                row["estimate_camera_y_m"],
                row["estimate_camera_z_m"],
            )
            robot_motion_policy = str(row["robot_motion_policy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("point policy evidence row is invalid") from exc
        if (
            point_name not in experiment.POINT_NAMES
            or point_index != experiment.POINT_NAMES.index(point_name)
            or trajectory_target not in experiment.POINT_NAMES
            or wrist_index < 0
            or trajectory_group_index < 0
            or frame_index < 0
            or not math.isfinite(target_front)
            or target_front < 0.0
            or target_front > 1.0
            or state not in _POINT_POLICY_STATES
            or not math.isfinite(predicted_error)
            or predicted_error < 0.0
            or type(direct_accepted) is not bool
            or type(rendered_observation_available) is not bool
            or type(target_geometry_inlier) is not bool
            or type(history_available) is not bool
            or type(estimate_valid) is not bool
            or robot_motion_policy != "unchanged_by_point_visibility_policy"
        ):
            raise RuntimeError("point policy evidence row is invalid")
        coordinates_are_none = all(value is None for value in estimate_values)
        coordinates_are_finite = all(
            value is not None and math.isfinite(float(value))
            for value in estimate_values
        )
        if state == "DIRECT_GEOMETRY_UPDATE":
            semantic_valid = (
                direct_accepted
                and rendered_observation_available
                and target_geometry_inlier
                and history_available
                and estimate_valid
                and estimate_source == "current_rendered_geometry"
                and coordinates_are_finite
                and last_valid_age == 0.0
                and predicted_error <= 0.010
                and policy_reason
                == "current_rendered_support_passes_declared_geometry_surface"
            )
        elif state == "BOUNDED_PATIENT_FRAME_FREEZE":
            semantic_valid = (
                not direct_accepted
                and history_available
                and estimate_valid
                and estimate_source == "last_valid_patient_frame_lock"
                and coordinates_are_finite
                and last_valid_age is not None
                and math.isfinite(float(last_valid_age))
                and 0.0 <= float(last_valid_age) <= 0.40
                and predicted_error <= 0.010
                and policy_reason
                == "short_hold_with_declared_motion_uncertainty_bound"
            )
        else:
            semantic_valid = (
                not direct_accepted
                and not estimate_valid
                and estimate_source == "none_fail_closed"
                and coordinates_are_none
                and (
                    (
                        not history_available
                        and last_valid_age is None
                        and policy_reason == "no_last_valid_estimate"
                    )
                    or (
                        history_available
                        and last_valid_age is not None
                        and math.isfinite(float(last_valid_age))
                        and float(last_valid_age) >= 0.0
                        and (
                            float(last_valid_age) > 0.40
                            or predicted_error > 0.010
                        )
                    )
                )
            )
        if not semantic_valid:
            raise RuntimeError("point policy evidence state semantics are invalid")
        normalized = dict(row)
        normalized.update(
            {
                "target_name": point_name,
                "frame_index": frame_index,
                # Reuse the strict episode state machine, but bind its occlusion
                # predicate to this point rather than any other occluded point.
                "robot_front_fraction_sum": target_front,
                "max_single_point_robot_front_fraction": target_front,
            }
        )
        groups.setdefault(
            (
                point_name,
                trajectory_target,
                wrist_index,
                trajectory_group_index,
            ),
            [],
        ).append(normalized)

    for (
        point_name,
        trajectory_target,
        wrist_index,
        trajectory_group_index,
    ), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["frame_index"]))
        frame_indices = [int(row["frame_index"]) for row in ordered]
        if len(set(frame_indices)) != len(frame_indices):
            raise RuntimeError("point policy evidence has duplicate frame indices")
        consecutive_runs: list[list[dict]] = []
        for row in ordered:
            if (
                not consecutive_runs
                or int(row["frame_index"])
                != int(consecutive_runs[-1][-1]["frame_index"]) + 1
            ):
                consecutive_runs.append([row])
            else:
                consecutive_runs[-1].append(row)
        for run in consecutive_runs:
            evidence = _targeted_policy_episode(run)
            if evidence["observed"]:
                selected_by_frame = {
                    int(row["frame_index"]): row for row in run
                }
                selected_rows = [
                    selected_by_frame[frame_index]
                    for frame_index in evidence["frame_indices"]
                ]
                return {
                    "observed": True,
                    "minimum_fraction": evidence["minimum_fraction"],
                    "point_name": point_name,
                    "trajectory_target_name": trajectory_target,
                    "wrist_index": wrist_index,
                    "trajectory_group_index": trajectory_group_index,
                    "frame_indices": evidence["frame_indices"],
                    "state_trace": [
                        str(row["policy_state"]) for row in selected_rows
                    ],
                    "target_robot_front_fraction": [
                        float(row["target_robot_front_fraction"])
                        for row in selected_rows
                    ],
                }
    return {
        "observed": False,
        "minimum_fraction": 0.25,
        "point_name": None,
        "trajectory_target_name": None,
        "wrist_index": None,
        "trajectory_group_index": None,
        "frame_indices": [],
        "state_trace": [],
        "target_robot_front_fraction": [],
    }


def _normalized_per_point_policy_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_point_names: Sequence[str],
) -> tuple[tuple[dict, ...], dict]:
    """Validate the typed frame-major policy table before it can be evidence."""

    point_names = tuple(str(name) for name in expected_point_names)
    if (
        not point_names
        or len(set(point_names)) != len(point_names)
        or any(not name for name in point_names)
    ):
        raise RuntimeError("per-point policy expected names are invalid")
    normalized_rows: list[dict] = []
    expected_keys = set(_POINT_POLICY_CSV_FIELDS)
    for raw_row in rows:
        if not isinstance(raw_row, Mapping) or set(raw_row) != expected_keys:
            raise RuntimeError("per-point policy CSV field set is invalid")
        row: dict[str, object] = {}
        for field in _POINT_POLICY_CSV_FIELDS:
            value = raw_row[field]
            if field in _POINT_POLICY_INT_FIELDS:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)
                ):
                    raise RuntimeError(f"per-point policy {field} is not an integer")
                row[field] = int(value)
            elif field in _POINT_POLICY_REQUIRED_FLOAT_FIELDS:
                if isinstance(value, (bool, np.bool_)):
                    raise RuntimeError(f"per-point policy {field} is not a float")
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"per-point policy {field} is not a float"
                    ) from exc
                if not math.isfinite(numeric):
                    raise RuntimeError(f"per-point policy {field} is not finite")
                row[field] = numeric
            elif field in _POINT_POLICY_OPTIONAL_FLOAT_FIELDS:
                if value is None:
                    row[field] = None
                else:
                    if isinstance(value, (bool, np.bool_)):
                        raise RuntimeError(
                            f"per-point policy {field} is not an optional float"
                        )
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"per-point policy {field} is not an optional float"
                        ) from exc
                    if not math.isfinite(numeric):
                        raise RuntimeError(
                            f"per-point policy {field} is not finite"
                        )
                    row[field] = numeric
            elif field in _POINT_POLICY_BOOL_FIELDS:
                if type(value) is not bool:
                    raise RuntimeError(f"per-point policy {field} is not a bool")
                row[field] = value
            else:
                if type(value) is not str:
                    raise RuntimeError(f"per-point policy {field} is not a string")
                row[field] = value
        normalized_rows.append(row)

    point_count = len(point_names)
    if not normalized_rows or len(normalized_rows) % point_count:
        raise RuntimeError("per-point policy rows do not form complete frames")
    frame_count = len(normalized_rows) // point_count
    last_direct: list[Optional[tuple[float, float, tuple[float, float, float]]]] = [
        None
    ] * point_count
    per_frame_state_counts: list[dict[str, int]] = []
    previous_time: Optional[float] = None
    for frame_index in range(frame_count):
        frame_rows = normalized_rows[
            frame_index * point_count : (frame_index + 1) * point_count
        ]
        first = frame_rows[0]
        frame_time = float(first["joint_time_s"])
        if previous_time is not None and frame_time <= previous_time:
            raise RuntimeError("per-point policy frame time is not strictly increasing")
        previous_time = frame_time
        common_fields = (
            "joint_time_s",
            "trajectory_target_name",
            "trajectory_target_index",
            "trajectory_group_index",
            "wrist_index",
            "motion_phase",
            "robot_front_fraction_sum",
            "max_single_point_robot_front_fraction",
        )
        for point_index, row in enumerate(frame_rows):
            if (
                row["frame_index"] != frame_index
                or row["point_index"] != point_index
                or row["point_name"] != point_names[point_index]
                or any(row[field] != first[field] for field in common_fields)
            ):
                raise RuntimeError("per-point policy frame-point ordering is invalid")
            target_name = str(row["trajectory_target_name"])
            target_index = int(row["trajectory_target_index"])
            if (
                target_name not in point_names
                or target_index != point_names.index(target_name)
                or int(row["trajectory_group_index"]) < 0
                or int(row["wrist_index"]) < 0
                or not row["motion_phase"]
            ):
                raise RuntimeError("per-point policy motion context is invalid")
            fractions = (
                float(row["target_visible_fraction"]),
                float(row["target_robot_front_fraction"]),
                float(row["target_valid_depth_fraction"]),
            )
            if any(value < 0.0 or value > 1.0 for value in fractions):
                raise RuntimeError("per-point policy target fraction is outside [0, 1]")
            gate = _target_direct_update_gate(
                geometry_surface_accepted=bool(row["geometry_surface_accepted"]),
                target_rendered_observation_available=bool(
                    row["target_rendered_observation_available"]
                ),
                target_geometry_inlier=bool(row["target_geometry_inlier"]),
                target_visible_fraction=fractions[0],
                target_robot_front_fraction=fractions[1],
                target_valid_depth_fraction=fractions[2],
            )
            if (
                row["direct_update_accepted"] != gate["accepted"]
                or row["direct_update_failure_reasons"]
                != "+".join(gate["failure_reasons"])
                or row["fixed_visible_point_count_rule_used"]
                != gate["fixed_visible_point_count_rule_used"]
            ):
                raise RuntimeError("per-point policy direct gate evidence is inconsistent")
            if bool(row["geometry_surface_accepted"]) != (
                row["geometry_failure_reasons"] == ""
            ):
                raise RuntimeError("per-point policy geometry evidence is inconsistent")
            if row["robot_motion_policy"] != "unchanged_by_point_visibility_policy":
                raise RuntimeError("per-point policy changed the robot motion policy")
            if (
                row["direct_estimate_source"]
                != "rendered_depth_se3_transform_of_canonical_reference_point"
            ):
                raise RuntimeError("per-point policy direct estimate source is invalid")

            state = str(row["policy_state"])
            if state not in _POINT_POLICY_STATES:
                raise RuntimeError("per-point policy state is invalid")
            coordinates = tuple(
                row[field]
                for field in (
                    "estimate_camera_x_m",
                    "estimate_camera_y_m",
                    "estimate_camera_z_m",
                )
            )
            all_none = all(value is None for value in coordinates)
            all_finite = all(
                value is not None and math.isfinite(float(value))
                for value in coordinates
            )
            if bool(row["estimate_valid"]):
                if not all_finite or state == "LOST":
                    raise RuntimeError("per-point policy valid estimate mask is inconsistent")
                finite_coordinates = tuple(float(value) for value in coordinates)
            else:
                if not all_none or state != "LOST":
                    raise RuntimeError("per-point policy invalid estimate mask is inconsistent")
                finite_coordinates = None
            history = last_direct[point_index]
            age = row["last_valid_age_s"]
            predicted = float(row["predicted_error_p95_m"])
            if predicted < 0.0:
                raise RuntimeError("per-point policy predicted error is negative")
            if state == "DIRECT_GEOMETRY_UPDATE":
                geometry_error = row["geometry_target_error_p95_m"]
                expected_decision = experiment.decide_occlusion_policy(
                    geometry_accepted=True,
                    last_valid_age_s=0.0,
                    last_target_error_p95_m=(
                        float(geometry_error) if geometry_error is not None else 0.010
                    ),
                    patient_motion_bound_m_s=0.010,
                    max_freeze_age_s=0.40,
                    max_freeze_error_p95_m=0.010,
                )
                if (
                    not row["direct_update_accepted"]
                    or not row["target_rendered_observation_available"]
                    or not row["target_geometry_inlier"]
                    or not row["history_available"]
                    or not row["estimate_valid"]
                    or row["estimate_source"] != "current_rendered_geometry"
                    or age != 0.0
                    or geometry_error is None
                    or float(geometry_error) > 0.010
                    or predicted != float(geometry_error)
                    or state != expected_decision.state
                    or predicted != expected_decision.predicted_error_p95_m
                    or row["policy_reason"] != expected_decision.reason
                    or finite_coordinates is None
                ):
                    raise RuntimeError("per-point DIRECT state semantics are invalid")
                last_direct[point_index] = (
                    frame_time,
                    float(geometry_error),
                    finite_coordinates,
                )
            elif history is None:
                no_history_age = 0.40 + 0.01
                expected_no_history_predicted = 0.010 + 0.010 * no_history_age
                if (
                    state != "LOST"
                    or row["direct_update_accepted"]
                    or row["history_available"]
                    or row["estimate_valid"]
                    or row["estimate_source"] != "none_fail_closed"
                    or row["policy_reason"] != "no_last_valid_estimate"
                    or age is not None
                    or predicted != expected_no_history_predicted
                ):
                    raise RuntimeError("per-point no-history LOST semantics are invalid")
            else:
                last_time, last_error, last_coordinates = history
                expected_age = frame_time - last_time
                expected_predicted = last_error + 0.010 * expected_age
                expected_decision = experiment.decide_occlusion_policy(
                    geometry_accepted=False,
                    last_valid_age_s=expected_age,
                    last_target_error_p95_m=last_error,
                    patient_motion_bound_m_s=0.010,
                    max_freeze_age_s=0.40,
                    max_freeze_error_p95_m=0.010,
                )
                if (
                    row["direct_update_accepted"]
                    or not row["history_available"]
                    or age != expected_age
                    or predicted != expected_predicted
                    or state != expected_decision.state
                    or row["policy_reason"] != expected_decision.reason
                ):
                    raise RuntimeError("per-point history evidence is inconsistent")
                if state == "BOUNDED_PATIENT_FRAME_FREEZE":
                    if (
                        not row["estimate_valid"]
                        or row["estimate_source"]
                        != "last_valid_patient_frame_lock"
                        or finite_coordinates != last_coordinates
                        or expected_age > 0.40
                        or expected_predicted > 0.010
                    ):
                        raise RuntimeError("per-point FREEZE state semantics are invalid")
                elif state == "LOST":
                    if (
                        row["estimate_valid"]
                        or row["estimate_source"] != "none_fail_closed"
                        or not all_none
                        or not (
                            expected_age > 0.40 or expected_predicted > 0.010
                        )
                    ):
                        raise RuntimeError("per-point LOST state semantics are invalid")
                else:
                    raise RuntimeError("per-point history state is invalid")

        target_front_values = [
            float(row["target_robot_front_fraction"]) for row in frame_rows
        ]
        expected_sum = float(sum(target_front_values))
        expected_max = float(max(target_front_values))
        if any(
            float(row["robot_front_fraction_sum"]) != expected_sum
            or float(row["max_single_point_robot_front_fraction"]) != expected_max
            for row in frame_rows
        ):
            raise RuntimeError("per-point policy frame occlusion aggregates are invalid")
        counts = {
            state: sum(row["policy_state"] == state for row in frame_rows)
            for state in _POINT_POLICY_STATES
        }
        if sum(counts.values()) != point_count:
            raise RuntimeError("per-point policy frame state counts are invalid")
        per_frame_state_counts.append(counts)

    point_frame_state_counts = {
        state: sum(row["policy_state"] == state for row in normalized_rows)
        for state in _POINT_POLICY_STATES
    }
    if sum(point_frame_state_counts.values()) != len(normalized_rows):
        raise RuntimeError("per-point policy aggregate state counts are invalid")
    frames_with_point_state = {
        state: sum(counts[state] > 0 for counts in per_frame_state_counts)
        for state in _POINT_POLICY_STATES
    }
    return tuple(normalized_rows), {
        "frame_count": frame_count,
        "point_count": point_count,
        "row_count": len(normalized_rows),
        "per_frame_state_counts": per_frame_state_counts,
        "point_frame_state_counts": point_frame_state_counts,
        "frames_with_point_state": frames_with_point_state,
    }


def _canonical_per_point_policy_csv_bytes(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_point_names: Sequence[str],
) -> bytes:
    normalized, _summary = _normalized_per_point_policy_rows(
        rows, expected_point_names=expected_point_names
    )
    stream = io.StringIO(newline="")
    writer = csv.writer(
        stream,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    writer.writerow(_POINT_POLICY_CSV_FIELDS)
    for row in normalized:
        encoded = []
        for field in _POINT_POLICY_CSV_FIELDS:
            value = row[field]
            if field in _POINT_POLICY_BOOL_FIELDS:
                encoded.append("true" if value else "false")
            elif field in _POINT_POLICY_INT_FIELDS:
                encoded.append(str(value))
            elif field in _POINT_POLICY_REQUIRED_FLOAT_FIELDS:
                encoded.append(format(float(value), ".17g"))
            elif field in _POINT_POLICY_OPTIONAL_FLOAT_FIELDS:
                encoded.append("" if value is None else format(float(value), ".17g"))
            else:
                encoded.append(str(value))
        writer.writerow(encoded)
    return stream.getvalue().encode("utf-8")


def _parse_per_point_policy_csv(
    path: Path,
    *,
    expected_point_names: Sequence[str],
) -> tuple[tuple[dict, ...], dict]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("per-point policy CSV could not be read") from exc
    parsed = list(csv.reader(io.StringIO(text, newline="")))
    if not parsed or tuple(parsed[0]) != _POINT_POLICY_CSV_FIELDS:
        raise RuntimeError("per-point policy CSV header is not exact")
    if any(len(record) != len(_POINT_POLICY_CSV_FIELDS) for record in parsed[1:]):
        raise RuntimeError("per-point policy CSV row width is invalid")
    rows: list[dict] = []
    for record in parsed[1:]:
        row: dict[str, object] = {}
        for field, token in zip(_POINT_POLICY_CSV_FIELDS, record):
            if field in _POINT_POLICY_BOOL_FIELDS:
                if token not in {"true", "false"}:
                    raise RuntimeError("per-point policy CSV bool is not canonical")
                value: object = token == "true"
            elif field in _POINT_POLICY_INT_FIELDS:
                try:
                    value = int(token)
                except ValueError as exc:
                    raise RuntimeError("per-point policy CSV integer is invalid") from exc
                if token != str(value):
                    raise RuntimeError("per-point policy CSV integer is not canonical")
            elif field in _POINT_POLICY_REQUIRED_FLOAT_FIELDS:
                try:
                    value = float(token)
                except ValueError as exc:
                    raise RuntimeError("per-point policy CSV float is invalid") from exc
                if not math.isfinite(value) or token != format(value, ".17g"):
                    raise RuntimeError("per-point policy CSV float is not canonical")
            elif field in _POINT_POLICY_OPTIONAL_FLOAT_FIELDS:
                if token == "":
                    value = None
                else:
                    try:
                        value = float(token)
                    except ValueError as exc:
                        raise RuntimeError(
                            "per-point policy CSV optional float is invalid"
                        ) from exc
                    if not math.isfinite(value) or token != format(value, ".17g"):
                        raise RuntimeError(
                            "per-point policy CSV optional float is not canonical"
                        )
            else:
                value = token
            row[field] = value
        rows.append(row)
    normalized, summary = _normalized_per_point_policy_rows(
        rows, expected_point_names=expected_point_names
    )
    if raw != _canonical_per_point_policy_csv_bytes(
        normalized, expected_point_names=expected_point_names
    ):
        raise RuntimeError("per-point policy CSV bytes are not canonical")
    return normalized, summary


def _build_per_point_policy_arrays(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_point_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict]:
    normalized, summary = _normalized_per_point_policy_rows(
        rows, expected_point_names=expected_point_names
    )
    frame_count = int(summary["frame_count"])
    point_count = int(summary["point_count"])
    by_frame = [
        normalized[index * point_count : (index + 1) * point_count]
        for index in range(frame_count)
    ]
    estimate = np.asarray(
        [
            [
                [
                    row["estimate_camera_x_m"],
                    row["estimate_camera_y_m"],
                    row["estimate_camera_z_m"],
                ]
                if row["estimate_valid"]
                else [np.nan, np.nan, np.nan]
                for row in frame_rows
            ]
            for frame_rows in by_frame
        ],
        dtype="<f8",
    )
    arrays = {
        "schema": np.asarray(_POINT_POLICY_SCHEMA, dtype="<U33"),
        "point_names": np.asarray(tuple(expected_point_names), dtype="<U2"),
        "frame_index": np.arange(frame_count, dtype="<i8"),
        "joint_time_s": np.asarray(
            [frame_rows[0]["joint_time_s"] for frame_rows in by_frame], dtype="<f8"
        ),
        "trajectory_target_name": np.asarray(
            [frame_rows[0]["trajectory_target_name"] for frame_rows in by_frame],
            dtype="<U2",
        ),
        "wrist_index": np.asarray(
            [frame_rows[0]["wrist_index"] for frame_rows in by_frame], dtype="<i8"
        ),
        "trajectory_group_index": np.asarray(
            [frame_rows[0]["trajectory_group_index"] for frame_rows in by_frame],
            dtype="<i8",
        ),
        "policy_state": np.asarray(
            [[row["policy_state"] for row in frame_rows] for frame_rows in by_frame],
            dtype="<U28",
        ),
        "estimate_valid": np.asarray(
            [[row["estimate_valid"] for row in frame_rows] for frame_rows in by_frame],
            dtype=bool,
        ),
        "history_valid": np.asarray(
            [
                [row["history_available"] for row in frame_rows]
                for frame_rows in by_frame
            ],
            dtype=bool,
        ),
        "estimate_camera_m": estimate,
        "predicted_error_p95_m": np.asarray(
            [
                [row["predicted_error_p95_m"] for row in frame_rows]
                for frame_rows in by_frame
            ],
            dtype="<f8",
        ),
        "last_valid_age_s": np.asarray(
            [
                [
                    np.nan
                    if row["last_valid_age_s"] is None
                    else row["last_valid_age_s"]
                    for row in frame_rows
                ]
                for frame_rows in by_frame
            ],
            dtype="<f8",
        ),
        "target_rendered_observation_available": np.asarray(
            [
                [
                    row["target_rendered_observation_available"]
                    for row in frame_rows
                ]
                for frame_rows in by_frame
            ],
            dtype=bool,
        ),
        "target_geometry_inlier": np.asarray(
            [
                [row["target_geometry_inlier"] for row in frame_rows]
                for frame_rows in by_frame
            ],
            dtype=bool,
        ),
        "direct_update_accepted": np.asarray(
            [
                [row["direct_update_accepted"] for row in frame_rows]
                for frame_rows in by_frame
            ],
            dtype=bool,
        ),
        "target_robot_front_fraction": np.asarray(
            [
                [row["target_robot_front_fraction"] for row in frame_rows]
                for frame_rows in by_frame
            ],
            dtype="<f8",
        ),
        "robot_motion_unchanged": np.ones(frame_count, dtype=bool),
    }
    return arrays, summary


def _load_validated_per_point_policy_npz(
    path: Path,
    *,
    frame_count: int,
    expected_point_names: Sequence[str],
) -> dict[str, np.ndarray]:
    point_count = len(tuple(expected_point_names))
    shapes = {
        "schema": (),
        "point_names": (point_count,),
        "frame_index": (frame_count,),
        "joint_time_s": (frame_count,),
        "trajectory_target_name": (frame_count,),
        "wrist_index": (frame_count,),
        "trajectory_group_index": (frame_count,),
        "policy_state": (frame_count, point_count),
        "estimate_valid": (frame_count, point_count),
        "history_valid": (frame_count, point_count),
        "estimate_camera_m": (frame_count, point_count, 3),
        "predicted_error_p95_m": (frame_count, point_count),
        "last_valid_age_s": (frame_count, point_count),
        "target_rendered_observation_available": (frame_count, point_count),
        "target_geometry_inlier": (frame_count, point_count),
        "direct_update_accepted": (frame_count, point_count),
        "target_robot_front_fraction": (frame_count, point_count),
        "robot_motion_unchanged": (frame_count,),
    }
    dtypes = {
        "schema": np.dtype("<U33"),
        "point_names": np.dtype("<U2"),
        "frame_index": np.dtype("<i8"),
        "joint_time_s": np.dtype("<f8"),
        "trajectory_target_name": np.dtype("<U2"),
        "wrist_index": np.dtype("<i8"),
        "trajectory_group_index": np.dtype("<i8"),
        "policy_state": np.dtype("<U28"),
        "estimate_valid": np.dtype(bool),
        "history_valid": np.dtype(bool),
        "estimate_camera_m": np.dtype("<f8"),
        "predicted_error_p95_m": np.dtype("<f8"),
        "last_valid_age_s": np.dtype("<f8"),
        "target_rendered_observation_available": np.dtype(bool),
        "target_geometry_inlier": np.dtype(bool),
        "direct_update_accepted": np.dtype(bool),
        "target_robot_front_fraction": np.dtype("<f8"),
        "robot_motion_unchanged": np.dtype(bool),
    }
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != set(shapes):
                raise RuntimeError("per-point policy NPZ key set is not exact")
            arrays = {name: np.array(archive[name], copy=True) for name in shapes}
    except RuntimeError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError("per-point policy NPZ could not be read") from exc
    for name, array in arrays.items():
        if array.shape != shapes[name] or array.dtype != dtypes[name]:
            raise RuntimeError(f"per-point policy NPZ {name} contract is invalid")
    if (
        str(arrays["schema"].item()) != _POINT_POLICY_SCHEMA
        or tuple(arrays["point_names"].tolist()) != tuple(expected_point_names)
        or not np.array_equal(arrays["frame_index"], np.arange(frame_count))
    ):
        raise RuntimeError("per-point policy NPZ identity is invalid")
    return arrays


def _verify_persisted_per_point_policy(
    *,
    csv_path: Path,
    npz_path: Path,
    expected_point_names: Sequence[str],
) -> dict:
    rows, summary = _parse_per_point_policy_csv(
        csv_path, expected_point_names=expected_point_names
    )
    expected_arrays, expected_summary = _build_per_point_policy_arrays(
        rows, expected_point_names=expected_point_names
    )
    if summary != expected_summary:
        raise RuntimeError("per-point policy CSV validation summaries differ")
    arrays = _load_validated_per_point_policy_npz(
        npz_path,
        frame_count=int(summary["frame_count"]),
        expected_point_names=expected_point_names,
    )
    for name, expected in expected_arrays.items():
        actual = arrays[name]
        if expected.dtype.kind == "f":
            equal = np.array_equal(actual, expected, equal_nan=True)
        else:
            equal = np.array_equal(actual, expected)
        if not equal:
            raise RuntimeError(f"per-point policy CSV/NPZ {name} alignment failed")
    estimate_valid = arrays["estimate_valid"]
    estimate = arrays["estimate_camera_m"]
    if (
        not np.isfinite(estimate[estimate_valid]).all()
        or not np.isnan(estimate[~estimate_valid]).all()
        or not np.array_equal(
            np.isfinite(arrays["last_valid_age_s"]), arrays["history_valid"]
        )
        or not np.all(arrays["robot_motion_unchanged"])
    ):
        raise RuntimeError("per-point policy mask or motion contract failed")
    selected_episode = _targeted_point_policy_episode(rows)
    return {
        "schema": _POINT_POLICY_SCHEMA,
        "frame_count": int(summary["frame_count"]),
        "point_count": int(summary["point_count"]),
        "row_count": int(summary["row_count"]),
        "checks": {
            "exact_csv_schema": True,
            "exact_npz_schema": True,
            "frame_point_cartesian_complete": True,
            "csv_npz_alignment": True,
            "mask_nan_contract": True,
            "history_state_contract": True,
            "direct_gate_recomputed": True,
            "robot_motion_unchanged": True,
            "state_counts_complete": True,
        },
        "point_frame_state_counts": dict(summary["point_frame_state_counts"]),
        "frames_with_point_state": dict(summary["frames_with_point_state"]),
        "per_frame_state_counts": list(summary["per_frame_state_counts"]),
        "selected_episode": selected_episode,
    }


def _persist_and_verify_per_point_policy(
    rows: Sequence[Mapping[str, object]],
    *,
    csv_path: Path,
    npz_path: Path,
    expected_point_names: Sequence[str],
) -> dict:
    csv_destination = Path(csv_path)
    npz_destination = Path(npz_path)
    if csv_destination.parent != npz_destination.parent:
        raise RuntimeError("per-point policy artifacts must share one directory")
    parent = csv_destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError("per-point policy artifact directory is unavailable")
    csv_bytes = _canonical_per_point_policy_csv_bytes(
        rows, expected_point_names=expected_point_names
    )
    arrays, _summary = _build_per_point_policy_arrays(
        rows, expected_point_names=expected_point_names
    )
    csv_temporary = csv_destination.with_name(
        f".{csv_destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    npz_temporary = npz_destination.with_name(
        f".{npz_destination.stem}.tmp-{os.getpid()}-{secrets.token_hex(8)}.npz"
    )
    try:
        csv_temporary.write_bytes(csv_bytes)
        np.savez_compressed(npz_temporary, **arrays)
        os.replace(csv_temporary, csv_destination)
        os.replace(npz_temporary, npz_destination)
    except BaseException:
        for temporary in (csv_temporary, npz_temporary):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return _verify_persisted_per_point_policy(
        csv_path=csv_destination,
        npz_path=npz_destination,
        expected_point_names=expected_point_names,
    )


def _verify_rendered_geometry_inlier_frame_products(
    frame_product_root: Path,
    per_point_policy: Sequence[Mapping[str, object]],
    *,
    frame_count: int,
) -> dict:
    """Reopen every frame product and bind raw observations to v3 policy rows."""

    root = Path(frame_product_root)
    point_count = len(experiment.POINT_NAMES)
    if (
        not root.is_dir()
        or root.is_symlink()
        or type(frame_count) is not int
        or frame_count <= 0
        or len(per_point_policy) != frame_count * point_count
    ):
        raise RuntimeError("rendered geometry inlier frame-product input is invalid")
    inlier_counts = []
    required = {
        "rendered_observation_available",
        "rendered_geometry_residual_m",
        "rendered_geometry_inlier_mask",
        "rendered_geometry_inlier_threshold_m",
        "rendered_geometry_initial_transform",
        "rendered_geometry_initial_fit_recoverable",
        "visible_fraction",
        "point_geometry_transform",
        "point_geometry_target_error_p95_m",
        "point_geometry_surface_accepted",
        "point_geometry_failure_reasons",
        "point_target_geometry_inlier",
        "point_direct_update_accepted",
        "frame_index",
    }
    for frame_index in range(frame_count):
        path = root / f"frame_{frame_index:05d}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                if not required.issubset(archive.files):
                    raise RuntimeError(
                        "rendered geometry inlier frame-product keys are incomplete"
                    )
                values = {name: np.array(archive[name], copy=True) for name in required}
        except RuntimeError:
            raise
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(
                "rendered geometry inlier frame product could not be read"
            ) from exc
        raw_valid = values["rendered_observation_available"]
        residual = values["rendered_geometry_residual_m"]
        inlier = values["rendered_geometry_inlier_mask"]
        threshold = values["rendered_geometry_inlier_threshold_m"]
        initial_transform = values["rendered_geometry_initial_transform"]
        recoverable = values["rendered_geometry_initial_fit_recoverable"]
        visible = values["visible_fraction"]
        point_transform = values["point_geometry_transform"]
        point_p95 = values["point_geometry_target_error_p95_m"]
        point_accepted = values["point_geometry_surface_accepted"]
        point_failures = values["point_geometry_failure_reasons"]
        point_inlier = values["point_target_geometry_inlier"]
        point_direct = values["point_direct_update_accepted"]
        if (
            raw_valid.shape != (point_count,)
            or raw_valid.dtype != np.dtype(bool)
            or residual.shape != (point_count,)
            or residual.dtype != np.dtype("<f8")
            or inlier.shape != (point_count,)
            or inlier.dtype != np.dtype(bool)
            or threshold.shape != ()
            or threshold.dtype != np.dtype("<f8")
            or initial_transform.shape != (4, 4)
            or initial_transform.dtype != np.dtype("<f8")
            or recoverable.shape != ()
            or recoverable.dtype != np.dtype(bool)
            or visible.shape != (point_count,)
            or visible.dtype != np.dtype("<f8")
            or point_transform.shape != (point_count, 4, 4)
            or point_transform.dtype != np.dtype("<f8")
            or point_p95.shape != (point_count,)
            or point_p95.dtype != np.dtype("<f8")
            or point_accepted.shape != (point_count,)
            or point_accepted.dtype != np.dtype(bool)
            or point_failures.shape != (point_count,)
            or point_failures.dtype != np.dtype("<U160")
            or point_inlier.shape != (point_count,)
            or point_inlier.dtype != np.dtype(bool)
            or point_direct.shape != (point_count,)
            or point_direct.dtype != np.dtype(bool)
            or values["frame_index"].shape != ()
            or values["frame_index"].dtype != np.dtype("<i8")
            or int(values["frame_index"]) != frame_index
        ):
            raise RuntimeError("rendered geometry inlier frame-product schema is invalid")
        expected_inlier = (
            raw_valid
            & (visible > 0.0)
            & np.isfinite(residual)
            & (residual <= 0.006)
        )
        if (
            float(threshold) != 0.006
            or not np.array_equal(inlier, expected_inlier)
            or not np.array_equal(point_inlier, inlier)
            or np.any(~raw_valid & ~np.isnan(residual))
            or bool(recoverable) != bool(np.isfinite(residual).any())
            or np.any(point_direct & ~point_inlier)
            or not np.isfinite(initial_transform).all()
            or not np.isfinite(point_transform).all()
        ):
            raise RuntimeError("rendered geometry inlier frame-product masks are invalid")
        frame_rows = per_point_policy[
            frame_index * point_count : (frame_index + 1) * point_count
        ]
        for point_index, row in enumerate(frame_rows):
            persisted_p95 = float(point_p95[point_index])
            expected_p95 = row["geometry_target_error_p95_m"]
            p95_matches = (
                not math.isfinite(persisted_p95)
                if expected_p95 is None
                else persisted_p95 == float(expected_p95)
            )
            if (
                bool(point_inlier[point_index])
                != bool(row["target_geometry_inlier"])
                or bool(point_direct[point_index])
                != bool(row["direct_update_accepted"])
                or bool(point_accepted[point_index])
                != bool(row["geometry_surface_accepted"])
                or str(point_failures[point_index])
                != str(row["geometry_failure_reasons"])
                or not p95_matches
            ):
                raise RuntimeError(
                    "rendered geometry inlier frame product differs from policy rows"
                )
        inlier_counts.append(int(np.count_nonzero(inlier)))
    return {
        "schema": "wam_rendered_geometry_inlier_frame_products_v1",
        "verified_frame_count": frame_count,
        "verified_point_frame_count": frame_count * point_count,
        "threshold_m": 0.006,
        "comparison": "less_than_or_equal",
        "inlier_count_min": min(inlier_counts),
        "inlier_count_median": float(np.median(inlier_counts)),
        "inlier_count_max": max(inlier_counts),
        "raw_observation_mask_bound": True,
        "policy_rows_bound": True,
        "robot_motion_input": False,
    }


def _encode_h264(
    frame_directory: Path,
    destination: Path,
    *,
    fps: int,
    expected_width: int,
    expected_height: int,
    expected_frames: int,
    ffmpeg_provenance: Mapping[str, object],
    ffprobe_provenance: Mapping[str, object],
    _allow_test_fd_fallback: bool = False,
) -> dict:
    with _pinned_media_tool(
        ffmpeg_provenance,
        tool_name="ffmpeg",
        allow_test_fallback=_allow_test_fd_fallback,
    ) as (ffmpeg, ffmpeg_descriptor):
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_directory / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        subprocess.run(
            command,
            pass_fds=(ffmpeg_descriptor,),
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    with _pinned_media_tool(
        ffprobe_provenance,
        tool_name="ffprobe",
        allow_test_fallback=_allow_test_fd_fallback,
    ) as (ffprobe, ffprobe_descriptor):
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,nb_frames,r_frame_rate",
                "-of",
                "json",
                str(destination),
            ],
            pass_fds=(ffprobe_descriptor,),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    value = json.loads(probe.stdout)
    if len(list(frame_directory.glob("frame_*.png"))) != expected_frames:
        raise RuntimeError("video frame set changed before encoding verification")
    _validate_video_probe(
        value,
        expected_width=expected_width,
        expected_height=expected_height,
        expected_fps=fps,
        expected_frames=expected_frames,
    )
    return value


def _validate_video_probe(
    probe: Mapping[str, object],
    *,
    expected_width: Optional[int],
    expected_height: Optional[int],
    expected_fps: int,
    expected_frames: int,
) -> Mapping[str, object]:
    streams = probe.get("streams", []) if isinstance(probe, Mapping) else []
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], Mapping):
        raise RuntimeError("video stream contract mismatch: expected one stream")
    stream = streams[0]
    try:
        rate = Fraction(str(stream.get("r_frame_rate")))
        frame_count = int(stream.get("nb_frames"))
        width = int(stream.get("width"))
        height = int(stream.get("height"))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError("video stream contract metadata is invalid") from exc
    valid = (
        stream.get("codec_name") == "h264"
        and stream.get("pix_fmt") == "yuv420p"
        and rate == Fraction(expected_fps, 1)
        and frame_count == expected_frames
        and (expected_width is None or width == expected_width)
        and (expected_height is None or height == expected_height)
    )
    if not valid:
        raise RuntimeError("video stream contract mismatch")
    return stream


def _output_payload_paths(root: Path) -> list[tuple[str, Path]]:
    root = Path(root)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise RuntimeError("output integrity root is unavailable") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeError("output integrity root must be a regular directory")

    destination = root / "SHA256SUMS.txt"
    paths: list[tuple[str, Path]] = []
    for current_text, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            directory = current / name
            try:
                info = directory.lstat()
            except OSError as exc:
                raise RuntimeError(
                    "output integrity inventory changed while enumerating directories"
                ) from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError(
                    f"output integrity payload contains a symlink or non-directory: {directory}"
                )
        for name in file_names:
            path = current / name
            try:
                info = path.lstat()
            except OSError as exc:
                raise RuntimeError(
                    "output integrity inventory changed while enumerating files"
                ) from exc
            if path == destination:
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    raise RuntimeError(
                        "output integrity manifest is a symlink or non-regular file"
                    )
                continue
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise RuntimeError(
                    f"output integrity payload contains a symlink or non-regular file: {path}"
                )
            relative = path.relative_to(root).as_posix()
            components = relative.split("/")
            if (
                not relative
                or relative.startswith("/")
                or relative != relative.strip()
                or "\\" in relative
                or any(part in ("", ".", "..") for part in components)
                or any(ord(character) < 0x20 or ord(character) == 0x7F for character in relative)
            ):
                raise RuntimeError("output integrity payload path is not canonical and safe")
            paths.append((relative, path))
    paths.sort(key=lambda item: item[0])
    if len({relative for relative, _path in paths}) != len(paths):
        raise RuntimeError("output integrity inventory contains duplicate paths")
    return paths


def _sha256_regular_payload_file(path: Path) -> str:
    try:
        expected = path.lstat()
    except OSError as exc:
        raise RuntimeError("output integrity payload file is unavailable") from exc
    if stat.S_ISLNK(expected.st_mode) or not stat.S_ISREG(expected.st_mode):
        raise RuntimeError("output integrity payload must contain regular files only")
    expected_identity = _media_identity(expected)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("output integrity payload file could not be pinned") from exc
    failures: list[tuple[str, BaseException, object]] = []
    digest: Optional[str] = None
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _media_identity(opened) != expected_identity
        ):
            raise RuntimeError("output integrity payload changed before hashing")
        hasher = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
        after = os.fstat(descriptor)
        if _media_identity(after) != expected_identity:
            raise RuntimeError("output integrity payload changed while hashing")
        digest = hasher.hexdigest()
    except BaseException as exc:
        failures.append(("output integrity payload hashing failed", exc, exc.__traceback__))
    try:
        os.close(descriptor)
    except BaseException as exc:
        failures.append(
            ("output integrity payload descriptor close failed", exc, exc.__traceback__)
        )
    _raise_first_failure(failures)
    if digest is None:
        raise RuntimeError("output integrity payload hashing produced no digest")
    return digest


def _read_pinned_regular_bytes(
    path: Path, *, label: str, maximum_bytes: int
) -> tuple[bytes, dict]:
    try:
        expected = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(expected.st_mode) or not stat.S_ISREG(expected.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    expected_identity = _media_identity(expected)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label} could not be pinned") from exc
    failures: list[tuple[str, BaseException, object]] = []
    raw: Optional[bytes] = None
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _media_identity(opened) != expected_identity
        ):
            raise RuntimeError(f"{label} changed before reading")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise RuntimeError(f"{label} exceeds its size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        final_path = path.lstat()
        if (
            _media_identity(after) != expected_identity
            or stat.S_ISLNK(final_path.st_mode)
            or not stat.S_ISREG(final_path.st_mode)
            or _media_identity(final_path) != expected_identity
        ):
            raise RuntimeError(f"{label} changed while reading")
        raw = b"".join(chunks)
    except BaseException as exc:
        failures.append((f"{label} read failed", exc, exc.__traceback__))
    try:
        os.close(descriptor)
    except BaseException as exc:
        failures.append((f"{label} descriptor close failed", exc, exc.__traceback__))
    _raise_first_failure(failures)
    if raw is None:
        raise RuntimeError(f"{label} produced no bytes")
    return raw, expected_identity


def _output_integrity_summary(raw_manifest: bytes, *, file_count: int) -> dict:
    digest = hashlib.sha256(raw_manifest).hexdigest()
    return {
        "schema": "wam_output_integrity_v1",
        "file_count": int(file_count),
        "sha256s_sha256": digest,
        "payload_inventory_sha256": digest,
    }


def _write_sha256s(root: Path) -> dict:
    root = Path(root)
    destination = root / "SHA256SUMS.txt"
    if os.path.lexists(destination):
        raise RuntimeError("output integrity manifest already exists")
    inventory = _output_payload_paths(root)
    rows = [
        f"{_sha256_regular_payload_file(path)}  {relative}"
        for relative, path in inventory
    ]
    if [relative for relative, _path in _output_payload_paths(root)] != [
        relative for relative, _path in inventory
    ]:
        raise RuntimeError("output integrity file set changed while hashing")
    raw = ("\n".join(rows) + "\n").encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(destination, flags, 0o644)
    except OSError as exc:
        raise RuntimeError("output integrity manifest could not be created exclusively") from exc
    failures: list[tuple[str, BaseException, object]] = []
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("output integrity manifest write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException as exc:
        failures.append(("output integrity manifest write failed", exc, exc.__traceback__))
    try:
        os.close(descriptor)
    except BaseException as exc:
        failures.append(
            ("output integrity manifest descriptor close failed", exc, exc.__traceback__)
        )
    if failures:
        try:
            destination.unlink(missing_ok=True)
        except BaseException as exc:
            failures.append(
                ("output integrity manifest cleanup failed", exc, exc.__traceback__)
            )
        _raise_first_failure(failures)
    expected = _output_integrity_summary(raw, file_count=len(inventory))
    verified = _verify_output_integrity(root)
    if verified != expected:
        raise RuntimeError("output integrity manifest did not verify after writing")
    return verified


def _verify_output_integrity(root: Path) -> dict:
    """Verify a quiescent runner-owned tree, not a concurrent-write transaction.

    The supervisor must call this only after the isolated runner has exited, then
    immediately exact-copy the tree with ``artifact_contract`` and reverify the
    immutable destination.  Repeated path scans cannot serialize a malicious
    same-UID writer; that stronger guarantee belongs to the publication layer.
    """

    root = Path(root)
    destination = root / "SHA256SUMS.txt"
    try:
        raw, initial_manifest_identity = _read_pinned_regular_bytes(
            destination,
            label="output integrity manifest",
            maximum_bytes=16 * 1024 * 1024,
        )
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise RuntimeError("output integrity manifest is unreadable") from exc
    if not text or not text.endswith("\n") or "\r" in text:
        raise RuntimeError("output integrity manifest is not canonical")

    expected_rows: list[tuple[str, str]] = []
    for row in text.splitlines():
        if (
            len(row) < 67
            or row[64:66] != "  "
            or any(character not in "0123456789abcdef" for character in row[:64])
        ):
            raise RuntimeError("output integrity manifest row is not canonical")
        digest = row[:64]
        relative = row[66:]
        components = relative.split("/")
        if (
            not relative
            or relative.startswith("/")
            or relative != relative.strip()
            or "\\" in relative
            or any(part in ("", ".", "..") for part in components)
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in relative)
        ):
            raise RuntimeError("output integrity manifest path is unsafe")
        expected_rows.append((relative, digest))
    if expected_rows != sorted(expected_rows) or len({row[0] for row in expected_rows}) != len(
        expected_rows
    ):
        raise RuntimeError("output integrity manifest inventory is not sorted and unique")

    actual_inventory = _output_payload_paths(root)
    actual_relatives = [relative for relative, _path in actual_inventory]
    expected_relatives = [relative for relative, _digest in expected_rows]
    if actual_relatives != expected_relatives:
        raise RuntimeError("output integrity file set does not match the manifest inventory")
    for (relative, path), (_expected_relative, expected_digest) in zip(
        actual_inventory, expected_rows
    ):
        actual_digest = _sha256_regular_payload_file(path)
        if actual_digest != expected_digest:
            raise RuntimeError(f"output integrity SHA-256 mismatch for {relative}")
    if [relative for relative, _path in _output_payload_paths(root)] != actual_relatives:
        raise RuntimeError("output integrity file set changed during verification")

    canonical = "".join(
        f"{digest}  {relative}\n" for relative, digest in expected_rows
    ).encode("utf-8")
    if canonical != raw:
        raise RuntimeError("output integrity manifest bytes are not canonical")
    final_raw, final_manifest_identity = _read_pinned_regular_bytes(
        destination,
        label="output integrity manifest",
        maximum_bytes=16 * 1024 * 1024,
    )
    if final_raw != raw or final_manifest_identity != initial_manifest_identity:
        raise RuntimeError("output integrity manifest changed during verification")
    return _output_integrity_summary(raw, file_count=len(expected_rows))


def _set_root_transform(prim, *, translation, yaw_degrees: float) -> None:
    from pxr import Gf, Usd, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*map(float, translation))
    )
    if yaw_degrees:
        xform.AddRotateZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            float(yaw_degrees)
        )


def _import_openarm(urdf_path: Path, destination_usd: Path) -> str:
    import omni.kit.commands
    from isaacsim.asset.importer.urdf import _urdf

    config = _urdf.ImportConfig()
    config.merge_fixed_joints = False
    config.fix_base = True
    config.make_default_prim = True
    config.create_physics_scene = False
    config.self_collision = False
    config.distance_scale = 1.0
    config.density = 0.0
    status, model = omni.kit.commands.execute(
        "URDFParseFile",
        urdf_path=str(urdf_path),
        import_config=config,
    )
    if not status:
        raise RuntimeError("official OpenArm URDF parse failed")
    status, imported = omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_path=str(urdf_path),
        urdf_robot=model,
        import_config=config,
        dest_path=str(destination_usd),
    )
    if not status or not destination_usd.is_file():
        raise RuntimeError(f"official OpenArm URDF import failed: {imported!r}")
    return str(imported)


def _count_meshes(root_prim) -> int:
    from pxr import Usd, UsdGeom

    return sum(
        prim.IsA(UsdGeom.Mesh)
        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())
    )


def _rgb_gate(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.dtype != np.uint8:
        raise RuntimeError(f"RGB annotation has invalid shape/dtype: {rgb.shape}/{rgb.dtype}")
    luminance = rgb[..., :3].astype(np.float32).mean(axis=2)
    if np.mean(luminance > 8.0) < 0.01 or np.percentile(luminance, 99) <= 20.0:
        raise RuntimeError("RGB render is effectively black")


def _triangulate_mesh(mesh_prim):
    from pxr import Gf, Usd, UsdGeom

    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(mesh.GetPointsAttr().Get(Usd.TimeCode.Default()), dtype=np.float64)
    counts = np.asarray(
        mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default()), dtype=np.int64
    )
    indices = np.asarray(
        mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default()), dtype=np.int64
    )
    if points.ndim != 2 or points.shape[1] != 3 or counts.ndim != 1 or indices.ndim != 1:
        raise RuntimeError("official human mesh topology is invalid")
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(mesh_prim)
    world_points = np.asarray(
        [transform.Transform(Gf.Vec3d(*map(float, point))) for point in points],
        dtype=np.float64,
    )
    triangles = []
    cursor = 0
    for count in counts:
        polygon = indices[cursor : cursor + int(count)]
        cursor += int(count)
        for offset in range(1, len(polygon) - 1):
            triangles.append((int(polygon[0]), int(polygon[offset]), int(polygon[offset + 1])))
    if cursor != len(indices) or len(triangles) < 10:
        raise RuntimeError("official human mesh could not be triangulated")
    return world_points, np.asarray(triangles, dtype=np.int64)


def _human_back_anchors(human_prim):
    from pxr import Usd, UsdGeom

    candidates = []
    for prim in Usd.PrimRange(human_prim, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh) and "labcoat" in prim.GetName().lower():
            candidates.append(prim)
    if len(candidates) != 1:
        paths = [str(prim.GetPath()) for prim in candidates]
        raise RuntimeError(f"expected one official labcoat mesh, found {paths}")
    vertices, triangles = _triangulate_mesh(candidates[0])
    anchors = experiment.anchor_ten_points_to_back_mesh(
        vertices,
        triangles,
        back_axis=1,
        back_sign=1,
    )
    return anchors, str(candidates[0].GetPath()), len(vertices), len(triangles)


def _world_to_camera(points_world_m: np.ndarray, position, look_at) -> np.ndarray:
    points = np.asarray(points_world_m, dtype=np.float64)
    origin = np.asarray(position, dtype=np.float64)
    forward = np.asarray(look_at, dtype=np.float64) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    delta = points - origin
    return np.column_stack((delta @ right, delta @ (-up), delta @ forward))


def _scaled_intrinsics(width: int, height: int) -> dict:
    source = experiment.ORBBEC_RGB_INTRINSICS
    sx = float(width) / float(source["width"])
    sy = float(height) / float(source["height"])
    return {
        "width": width,
        "height": height,
        "fx": source["fx"] * sx,
        "fy": source["fy"] * sy,
        "cx": source["cx"] * sx,
        "cy": source["cy"] * sy,
    }


def _usd_camera_parameters(
    intrinsics: Mapping[str, float], *, horizontal_aperture_mm: float
) -> dict:
    values = {name: float(intrinsics[name]) for name in ("width", "height", "fx", "fy", "cx", "cy")}
    if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
        raise RuntimeError("camera intrinsics must be finite and positive")
    if not math.isfinite(horizontal_aperture_mm) or horizontal_aperture_mm <= 0.0:
        raise RuntimeError("horizontal camera aperture must be finite and positive")
    focal = horizontal_aperture_mm * values["fx"] / values["width"]
    vertical_aperture = focal * values["height"] / values["fy"]
    horizontal_offset = (
        values["width"] / 2.0 - values["cx"]
    ) * horizontal_aperture_mm / values["width"]
    vertical_offset = (
        values["cy"] - values["height"] / 2.0
    ) * vertical_aperture / values["height"]
    return {
        "focal_length_mm": focal,
        "horizontal_aperture_mm": horizontal_aperture_mm,
        "vertical_aperture_mm": vertical_aperture,
        "horizontal_aperture_offset_mm": horizontal_offset,
        "vertical_aperture_offset_mm": vertical_offset,
    }


def _unique_new_prim_path(before: Sequence[str], after: Sequence[str], *, label: str) -> str:
    previous = {str(path) for path in before}
    current = {str(path) for path in after}
    created = sorted(current - previous)
    if len(created) != 1:
        raise RuntimeError(f"expected one new {label} prim, found {created}")
    return created[0]


def _run_full_experiment(
    *,
    args: argparse.Namespace,
    world,
    timeline,
    articulation,
    annotators: Mapping[str, object],
    oblique_rgb_annotator,
    anchors: Sequence[experiment.MeshAnchor],
    projected: np.ndarray,
    camera_points: np.ndarray,
    urdf_path: Path,
    manifest_path: Path,
    out: Path,
    runtime_provenance: Mapping[str, object],
) -> dict:
    """Render real OpenArm joint motion and audited mask/depth evidence."""

    import cv2

    if world.is_playing():
        raise RuntimeError("full visibility capture requires a paused physics timeline")

    raw_root = out / "raw"
    scenario_root = raw_root / "scenarios"
    frame_root = raw_root / "video_frames"
    frame_product_root = raw_root / "frame_products"
    for directory in (raw_root, scenario_root, frame_root, frame_product_root):
        directory.mkdir(parents=True, exist_ok=True)
    render_sync_count = 0
    render_sync_max_time_advance_s = 0.0
    zero_time_capture_count = 0
    zero_time_capture_max_world_delta_s = 0.0
    zero_time_capture_max_timeline_delta_s = 0.0

    urdf_raw = urdf_path.read_bytes()
    joint_limits = experiment.left_joint_limits(urdf_raw)
    velocity_limits = experiment.left_joint_velocity_limits(urdf_raw)
    seed_table = _ik_seeds(joint_limits)
    left_indices = np.asarray(
        [articulation.get_dof_index(name) for name in experiment.LEFT_FOLLOWER_JOINTS],
        dtype=np.int64,
    )
    right_indices = np.asarray(
        [articulation.get_dof_index(name) for name in experiment.RIGHT_REFERENCE_JOINTS],
        dtype=np.int64,
    )
    if np.any(left_indices < 0) or np.any(right_indices < 0):
        raise RuntimeError("Isaac articulation omitted an official OpenArm revolute DOF")
    if len(set(left_indices.tolist() + right_indices.tolist())) != 14:
        raise RuntimeError("Isaac articulation did not expose 14 distinct official revolute DOFs")
    right_fixed = np.asarray(
        articulation.get_joint_positions(joint_indices=right_indices), dtype=np.float64
    ).copy()
    if right_fixed.shape != (7,) or not np.isfinite(right_fixed).all():
        raise RuntimeError("could not freeze the official right reference chain")

    anchor_positions = np.asarray([anchor.position_m for anchor in anchors])
    anchor_normals = np.asarray([anchor.surface_normal for anchor in anchors])
    contact_offset_m = 0.10
    approach_offset_m = 0.18
    ik_rows = []
    reachable = []
    for target_index, anchor in enumerate(anchors):
        contact_world = anchor_positions[target_index] + contact_offset_m * anchor_normals[target_index]
        contact_local = _world_to_robot_root(contact_world)
        approach_world = anchor_positions[target_index] + approach_offset_m * anchor_normals[target_index]
        approach_local = _world_to_robot_root(approach_world)
        candidates = experiment.solve_left_position_ik_candidates(
            urdf_raw,
            contact_local,
            initial_seeds=seed_table,
            tolerance_m=5.0e-4,
            max_iterations=320,
            max_solutions=2,
            min_orientation_separation_rad=math.radians(20.0),
        )
        solution_rows = []
        complete_phase_count = 0
        for wrist_index, candidate in enumerate(candidates):
            try:
                approach_q = experiment.solve_left_position_ik(
                    urdf_raw,
                    approach_local,
                    initial_q=candidate.joint_positions,
                    tolerance_m=5.0e-4,
                    max_iterations=320,
                )
                phase_status = "kinematic_phases_complete_collision_uncommissioned"
                phase_failure = None
                complete_phase_count += 1
                reachable.append((target_index, wrist_index, candidate, approach_q))
            except experiment.ExperimentContractError as exc:
                approach_q = None
                phase_status = "failed_stratum"
                phase_failure = str(exc)
            solution_rows.append(
                {
                    **_json_safe(candidate),
                    "approach_joint_positions_rad": (
                        None if approach_q is None else approach_q.tolist()
                    ),
                    "contact_joint_positions_rad": candidate.joint_positions.tolist(),
                    "depart_joint_positions_rad": (
                        None if approach_q is None else approach_q.tolist()
                    ),
                    "phase_status": phase_status,
                    "phase_failure": phase_failure,
                }
            )
        stratum_complete = len(candidates) == 2 and complete_phase_count == 2
        row = {
            "target_name": anchor.name,
            "target_index": target_index,
            "surface_anchor_world_m": anchor_positions[target_index].tolist(),
            "surface_normal_world": anchor_normals[target_index].tolist(),
            "nominal_tool_tip_contact_offset_m": contact_offset_m,
            "approach_depart_offset_m": approach_offset_m,
            "contact_target_world_m": contact_world.tolist(),
            "contact_target_robot_root_m": contact_local.tolist(),
            "approach_target_world_m": approach_world.tolist(),
            "approach_target_robot_root_m": approach_local.tolist(),
            "required_wrist_attitudes": 2,
            "solved_wrist_attitudes": len(candidates),
            "status": "kinematic_phases_complete" if stratum_complete else "failed_stratum",
            "failure_reason": (
                None
                if stratum_complete
                else "missing_two_distinct_contact_ik_or_approach_depart_solution"
            ),
            "collision_contract_status": "uncommissioned_fail_closed",
            "solutions": solution_rows,
        }
        ik_rows.append(row)
    _write_json(
        raw_root / "ik_plan.json",
        {
            "source": "official_openarm_v1_urdf_real_left_seven_joint_chain",
            "root_pose_animated": False,
            "position_only_ik": True,
            "wrist_attitude_source": "measured_fk_rotation_of_distinct_position_ik_branches",
            "seed": 20260715,
            "seed_count": len(seed_table),
            "rows": ik_rows,
        },
    )
    if not reachable:
        raise RuntimeError("official OpenArm IK could not reach any official-human back target")

    support_radius = max(4, int(round(args.width / 80.0)))
    dilation = max(1, int(round(args.width / 320.0)))
    render_intrinsics = _scaled_intrinsics(args.width, args.height)
    static_rows = []
    holdout_rows = []
    for static_capture_index, (target_index, wrist_index, candidate, approach_q) in enumerate(reachable):
        target_name = experiment.POINT_NAMES[target_index]
        scenario = experiment.scenario_name(
            target_name, wrist_index, "cam_nominal_uncommissioned", 0
        )
        articulation.set_joint_positions(
            candidate.joint_positions, joint_indices=left_indices
        )
        articulation.set_joint_velocities(np.zeros(7), joint_indices=left_indices)
        render_sync = _synchronize_articulation_render_state(
            world, timeline=timeline
        )
        render_sync_count += 1
        render_sync_max_time_advance_s = max(
            render_sync_max_time_advance_s,
            abs(
                float(render_sync["simulation_time_after_s"])
                - float(render_sync["simulation_time_before_s"])
            ),
        )
        capture_products, capture_gate = _capture_products_zero_time(
            world,
            timeline,
            annotators,
            rt_subframes=args.rt_subframes,
            label=f"static_capture_{static_capture_index:03d}",
        )
        rgb, depth, robot_mask, human_mask, _instance, _semantic = capture_products
        zero_time_capture_count += 1
        zero_time_capture_max_world_delta_s = max(
            zero_time_capture_max_world_delta_s,
            abs(float(capture_gate["world_time_delta_s"])),
        )
        zero_time_capture_max_timeline_delta_s = max(
            zero_time_capture_max_timeline_delta_s,
            abs(float(capture_gate["timeline_time_delta_s"])),
        )
        readback = _validate_joint_readback(
            articulation,
            left_indices=left_indices,
            right_indices=right_indices,
            commanded_left=candidate.joint_positions,
            frozen_right=right_fixed,
        )
        oblique_rgb = np.asarray(oblique_rgb_annotator.get_data())
        visibility = experiment.measure_rendered_visibility(
            point_names=experiment.POINT_NAMES,
            projected_pixels=projected,
            point_depth_m=camera_points[:, 2],
            robot_mask=robot_mask,
            human_mask=human_mask,
            depth_m=depth,
            support_radius_px=support_radius,
            robot_mask_dilation_px=dilation,
            front_depth_margin_m=0.003,
        )
        rendered_observations = _rendered_anchor_observations(
            projected_pixels=projected,
            depth_m=depth,
            human_mask=human_mask,
            robot_mask=robot_mask,
            intrinsics=render_intrinsics,
            support_radius_px=support_radius,
            robot_mask_dilation_px=dilation,
        )
        geometry_inliers = _rendered_geometry_inlier_selection(
            camera_points,
            visibility,
            rendered_observations["camera_points_m"],
            rendered_observations["available"],
        )
        geometry_result = _evaluate_geometry(
            camera_points,
            visibility,
            target_index,
            observed_camera_points=rendered_observations["camera_points_m"],
            observed_valid=geometry_inliers["inlier_mask"],
        )
        holdout = _holdout_parameter_trials(
            camera_points,
            visibility,
            target_index=target_index,
            seed=2026071500 + target_index * 10 + wrist_index,
            trials_per_stratum=30,
        )
        holdout_rows.append({"scenario": scenario, **holdout})
        frame = _draw_frame(
            rgb,
            projected,
            visibility,
            title=f"OpenArm v1 846a82d8baf | {target_name} wrist {wrist_index}",
            subtitle=(
                f"geometry={geometry_result.decision} | scope=parameter sensitivity only"
            ),
            oblique_rgb=oblique_rgb,
        )
        if not cv2.imwrite(str(scenario_root / f"{scenario}.png"), frame):
            raise RuntimeError("could not persist static scenario frame")
        np.savez_compressed(
            scenario_root / f"{scenario}.npz",
            depth_m=np.asarray(depth, dtype=np.float32),
            robot_mask=np.asarray(robot_mask, dtype=np.uint8),
            human_mask=np.asarray(human_mask, dtype=np.uint8),
            projected_points_px=np.asarray(projected, dtype=np.float64),
            point_camera_depth_m=np.asarray(camera_points[:, 2], dtype=np.float64),
            rendered_observation_camera_m=np.asarray(
                rendered_observations["camera_points_m"], dtype=np.float64
            ),
            rendered_observation_available=np.asarray(
                rendered_observations["available"], dtype=bool
            ),
            rendered_observation_sample_pixels=np.asarray(
                rendered_observations["sample_pixels"], dtype=np.float64
            ),
            rendered_observation_sample_distance_px=np.asarray(
                rendered_observations["sample_distance_px"], dtype=np.float64
            ),
            rendered_observation_sample_depth_m=np.asarray(
                rendered_observations["sample_depth_m"], dtype=np.float64
            ),
            rendered_observation_source=np.asarray(
                rendered_observations["source"]
            ),
            rendered_geometry_residual_m=np.asarray(
                geometry_inliers["residual_m"], dtype=np.float64
            ),
            rendered_geometry_inlier_mask=np.asarray(
                geometry_inliers["inlier_mask"], dtype=bool
            ),
            rendered_geometry_inlier_threshold_m=np.asarray(
                geometry_inliers["threshold_m"], dtype=np.float64
            ),
            rendered_geometry_initial_transform=np.asarray(
                geometry_inliers["initial_fit"].transform, dtype=np.float64
            ),
            rendered_geometry_initial_fit_recoverable=np.asarray(
                bool(np.isfinite(geometry_inliers["residual_m"]).any()),
                dtype=bool,
            ),
            visible_fraction=np.asarray(
                [item.visible_fraction for item in visibility], dtype=np.float64
            ),
            robot_front_fraction=np.asarray(
                [item.robot_front_fraction for item in visibility], dtype=np.float64
            ),
            valid_depth_fraction=np.asarray(
                [item.valid_depth_fraction for item in visibility], dtype=np.float64
            ),
            support_radius_px=np.asarray(support_radius, dtype=np.int64),
            capture_index=np.asarray(static_capture_index, dtype=np.int64),
            scene_identity=np.asarray(scenario),
        )
        static_rows.append(
            {
                "scenario": scenario,
                "target_name": target_name,
                "target_index": target_index,
                "wrist_index": wrist_index,
                "joint_positions_rad": candidate.joint_positions.tolist(),
                "observed_joint_positions_rad": readback["observed_left"].tolist(),
                "left_readback_max_abs_error_rad": readback[
                    "left_max_abs_error_rad"
                ],
                "right_max_abs_drift_rad": readback["right_max_abs_drift_rad"],
                "render_sync": render_sync,
                "tip_position_error_m": candidate.position_error_m,
                "orientation_separation_from_first_deg": math.degrees(
                    candidate.orientation_separation_from_first_rad
                ),
                "robot_pixel_count": int(np.count_nonzero(robot_mask)),
                "human_pixel_count": int(np.count_nonzero(human_mask)),
                "visible_point_count_report_only": int(
                    sum(item.visible_fraction >= 0.35 for item in visibility)
                ),
                "visibility": [_json_safe(item) for item in visibility],
                "rendered_observations": _json_safe(rendered_observations),
                "rendered_geometry_inlier_selection": {
                    "initial_fit": _json_safe(geometry_inliers["initial_fit"]),
                    "residual_m": _json_safe(geometry_inliers["residual_m"]),
                    "inlier_mask": _json_safe(geometry_inliers["inlier_mask"]),
                    "inlier_count": int(
                        np.count_nonzero(geometry_inliers["inlier_mask"])
                    ),
                    "threshold_m": geometry_inliers["threshold_m"],
                    "comparison": "less_than_or_equal",
                    "post_hoc_candidate": True,
                },
                "geometry": _json_safe(geometry_result),
                "holdout_parameter_sensitivity_summary": {
                    "trials_per_stratum": holdout["trials_per_stratum"],
                    "strata": holdout["strata"],
                    "independent_trajectory_claim": False,
                },
                "scope": "parameter_sensitivity_only",
            }
        )
    _write_json(
        raw_root / "holdout_parameter_sensitivity.json",
        {
            "scope": "parameter_sensitivity_only",
            "physical_accuracy_pass": False,
            "independent_trajectory_claim": False,
            "rows": holdout_rows,
        },
    )

    home = np.array([-1.0, -1.2, 0.0, 1.15, 0.0, 0.0, -0.30])
    waypoints = [home]
    segment_targets = []
    segment_wrists = []
    segment_phases = []
    for target_index, wrist_index, candidate, approach_q in reachable:
        for waypoint, phase in (
            (approach_q, "approach"),
            (candidate.joint_positions, "nominal_contact"),
            (approach_q, "depart"),
            (home, "return_home"),
        ):
            waypoints.append(waypoint)
            segment_targets.append(target_index)
            segment_wrists.append(wrist_index)
            segment_phases.append(phase)
    trajectory = experiment.build_minimum_jerk_trajectory(
        np.asarray(waypoints), frames_per_segment=9, seconds_per_segment=2.8
    )
    trajectory_gate = experiment.validate_left_joint_trajectory(
        trajectory.joint_positions,
        trajectory.time_s,
        trajectory.root_poses,
        joint_limits=joint_limits,
        velocity_limits=velocity_limits,
        acceleration_limits=np.full(7, 8.0),
    )
    _write_json(raw_root / "trajectory_contract.json", trajectory_gate)
    np.savez_compressed(
        raw_root / "representative_joint_trajectory.npz",
        joint_positions_rad=trajectory.joint_positions,
        time_s=trajectory.time_s,
        root_poses=trajectory.root_poses,
        left_joint_names=np.asarray(experiment.LEFT_FOLLOWER_JOINTS),
        segment_target_indices=np.asarray(segment_targets, dtype=np.int64),
        segment_wrist_indices=np.asarray(segment_wrists, dtype=np.int64),
        segment_phases=np.asarray(segment_phases),
    )

    per_frame = []
    per_point_policy = []
    frame_hashes = []
    robot_mask_hashes = []
    trajectory_mask_keyframes = []
    last_valid_time_by_point: list[Optional[float]] = [None] * len(
        experiment.POINT_NAMES
    )
    last_valid_p95_by_point: list[Optional[float]] = [None] * len(
        experiment.POINT_NAMES
    )
    last_valid_camera_point_by_point: list[Optional[np.ndarray]] = [None] * len(
        experiment.POINT_NAMES
    )
    best_occlusion = (-1.0, None)
    max_single_point_occlusion = 0.0
    frames_per_segment_minus_endpoint = 8
    for frame_index, (time_s, joints) in enumerate(
        zip(trajectory.time_s, trajectory.joint_positions)
    ):
        segment = min(
            max(frame_index - 1, 0) // frames_per_segment_minus_endpoint,
            len(segment_targets) - 1,
        )
        target_index = segment_targets[segment]
        wrist_index = segment_wrists[segment]
        motion_phase = segment_phases[segment]
        trajectory_group_index = segment // 4
        articulation.set_joint_positions(joints, joint_indices=left_indices)
        articulation.set_joint_velocities(np.zeros(7), joint_indices=left_indices)
        render_sync = _synchronize_articulation_render_state(
            world, timeline=timeline
        )
        render_sync_count += 1
        render_sync_max_time_advance_s = max(
            render_sync_max_time_advance_s,
            abs(
                float(render_sync["simulation_time_after_s"])
                - float(render_sync["simulation_time_before_s"])
            ),
        )
        capture_products, capture_gate = _capture_products_zero_time(
            world,
            timeline,
            annotators,
            rt_subframes=args.rt_subframes,
            label=f"trajectory_capture_{frame_index:05d}",
        )
        rgb, depth, robot_mask, human_mask, _instance, _semantic = capture_products
        zero_time_capture_count += 1
        zero_time_capture_max_world_delta_s = max(
            zero_time_capture_max_world_delta_s,
            abs(float(capture_gate["world_time_delta_s"])),
        )
        zero_time_capture_max_timeline_delta_s = max(
            zero_time_capture_max_timeline_delta_s,
            abs(float(capture_gate["timeline_time_delta_s"])),
        )
        readback = _validate_joint_readback(
            articulation,
            left_indices=left_indices,
            right_indices=right_indices,
            commanded_left=joints,
            frozen_right=right_fixed,
        )
        oblique_rgb = np.asarray(oblique_rgb_annotator.get_data())
        visibility = experiment.measure_rendered_visibility(
            point_names=experiment.POINT_NAMES,
            projected_pixels=projected,
            point_depth_m=camera_points[:, 2],
            robot_mask=robot_mask,
            human_mask=human_mask,
            depth_m=depth,
            support_radius_px=support_radius,
            robot_mask_dilation_px=dilation,
            front_depth_margin_m=0.003,
        )
        rendered_observations = _rendered_anchor_observations(
            projected_pixels=projected,
            depth_m=depth,
            human_mask=human_mask,
            robot_mask=robot_mask,
            intrinsics=render_intrinsics,
            support_radius_px=support_radius,
            robot_mask_dilation_px=dilation,
        )
        geometry_inliers = _rendered_geometry_inlier_selection(
            camera_points,
            visibility,
            rendered_observations["camera_points_m"],
            rendered_observations["available"],
        )
        robot_front_sum = float(sum(item.robot_front_fraction for item in visibility))
        frame_max_single_point_occlusion = float(
            max(item.robot_front_fraction for item in visibility)
        )
        point_geometry_results = [
            _evaluate_geometry(
                camera_points,
                visibility,
                point_index,
                observed_camera_points=rendered_observations["camera_points_m"],
                observed_valid=geometry_inliers["inlier_mask"],
            )
            for point_index in range(len(experiment.POINT_NAMES))
        ]
        geometry_result = point_geometry_results[target_index]
        point_policy_frame = []
        for point_index, (point_name, point_visibility, point_geometry) in enumerate(
            zip(experiment.POINT_NAMES, visibility, point_geometry_results)
        ):
            direct_gate = _target_direct_update_gate(
                geometry_surface_accepted=bool(point_geometry.accepted),
                target_rendered_observation_available=bool(
                    rendered_observations["available"][point_index]
                ),
                target_geometry_inlier=bool(
                    geometry_inliers["inlier_mask"][point_index]
                ),
                target_visible_fraction=float(point_visibility.visible_fraction),
                target_robot_front_fraction=float(
                    point_visibility.robot_front_fraction
                ),
                target_valid_depth_fraction=float(
                    point_visibility.valid_depth_fraction
                ),
            )
            current_estimate = (
                _geometry_target_estimate_camera_m(
                    camera_points[point_index], point_geometry
                )
                if direct_gate["accepted"]
                else np.full(3, np.nan, dtype=np.float64)
            )
            point_policy = _advance_point_lock_policy(
                time_s=float(time_s),
                current_camera_point_m=current_estimate,
                current_target_error_p95_m=float(
                    point_geometry.target_error_p95_m
                ),
                direct_gate=direct_gate,
                last_valid_time_s=last_valid_time_by_point[point_index],
                last_valid_error_p95_m=last_valid_p95_by_point[point_index],
                last_valid_camera_point_m=last_valid_camera_point_by_point[
                    point_index
                ],
            )
            last_valid_time_by_point[point_index] = point_policy[
                "next_last_valid_time_s"
            ]
            last_valid_p95_by_point[point_index] = point_policy[
                "next_last_valid_error_p95_m"
            ]
            last_valid_camera_point_by_point[point_index] = point_policy[
                "next_last_valid_camera_point_m"
            ]
            estimate = np.asarray(
                point_policy["estimate_camera_m"], dtype=np.float64
            )
            policy_row = {
                "frame_index": frame_index,
                "joint_time_s": float(time_s),
                "point_name": point_name,
                "point_index": point_index,
                "trajectory_target_name": experiment.POINT_NAMES[target_index],
                "trajectory_target_index": target_index,
                "trajectory_group_index": trajectory_group_index,
                "wrist_index": wrist_index,
                "motion_phase": motion_phase,
                "policy_state": point_policy["policy_state"],
                "policy_reason": point_policy["policy_reason"],
                "predicted_error_p95_m": point_policy[
                    "predicted_error_p95_m"
                ],
                "last_valid_age_s": point_policy["last_valid_age_s"],
                "history_available": point_policy["history_available"],
                "estimate_valid": point_policy["estimate_valid"],
                "estimate_source": point_policy["estimate_source"],
                "estimate_camera_x_m": (
                    float(estimate[0]) if point_policy["estimate_valid"] else None
                ),
                "estimate_camera_y_m": (
                    float(estimate[1]) if point_policy["estimate_valid"] else None
                ),
                "estimate_camera_z_m": (
                    float(estimate[2]) if point_policy["estimate_valid"] else None
                ),
                "target_rendered_observation_available": direct_gate[
                    "target_rendered_observation_available"
                ],
                "target_geometry_inlier": direct_gate[
                    "target_geometry_inlier"
                ],
                "direct_update_accepted": direct_gate["accepted"],
                "direct_update_failure_reasons": "+".join(
                    direct_gate["failure_reasons"]
                ),
                "geometry_surface_accepted": point_geometry.accepted,
                "geometry_failure_reasons": "+".join(
                    point_geometry.failure_reasons
                ),
                "geometry_target_error_p95_m": (
                    float(point_geometry.target_error_p95_m)
                    if math.isfinite(point_geometry.target_error_p95_m)
                    else None
                ),
                "target_visible_fraction": float(
                    point_visibility.visible_fraction
                ),
                "target_robot_front_fraction": float(
                    point_visibility.robot_front_fraction
                ),
                "target_valid_depth_fraction": float(
                    point_visibility.valid_depth_fraction
                ),
                "robot_front_fraction_sum": robot_front_sum,
                "max_single_point_robot_front_fraction": (
                    frame_max_single_point_occlusion
                ),
                "fixed_visible_point_count_rule_used": False,
                "robot_motion_policy": point_policy["robot_motion_policy"],
                "direct_estimate_source": (
                    "rendered_depth_se3_transform_of_canonical_reference_point"
                ),
            }
            point_policy_frame.append(policy_row)
            per_point_policy.append(policy_row)
        point_state_counts = {
            state: sum(row["policy_state"] == state for row in point_policy_frame)
            for state in (
                "DIRECT_GEOMETRY_UPDATE",
                "BOUNDED_PATIENT_FRAME_FREEZE",
                "LOST",
            )
        }
        state_severity = {
            "DIRECT_GEOMETRY_UPDATE": 0,
            "BOUNDED_PATIENT_FRAME_FREEZE": 1,
            "LOST": 2,
        }
        selected_point_policy = max(
            point_policy_frame,
            key=lambda row: (
                state_severity[str(row["policy_state"])],
                float(row["target_robot_front_fraction"]),
            ),
        )
        visible_report_only = int(
            sum(item.visible_fraction >= 0.35 for item in visibility)
        )
        state_label = {
            "DIRECT_GEOMETRY_UPDATE": "DIRECT",
            "BOUNDED_PATIENT_FRAME_FREEZE": "FREEZE",
            "LOST": "LOST",
        }[str(selected_point_policy["policy_state"])]
        frame = _draw_frame(
            rgb,
            projected,
            visibility,
            title=(
                f"OpenArm v1 846a82d8baf | {experiment.POINT_NAMES[target_index]} "
                f"wrist {wrist_index} {motion_phase}"
            ),
            subtitle=(
                "D/F/L "
                f"{point_state_counts['DIRECT_GEOMETRY_UPDATE']}/"
                f"{point_state_counts['BOUNDED_PATIENT_FRAME_FREEZE']}/"
                f"{point_state_counts['LOST']} | "
                f"{selected_point_policy['point_name']} {state_label} | arm keeps moving"
            ),
            oblique_rgb=oblique_rgb,
        )
        frame_path = frame_root / f"frame_{frame_index:05d}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError("could not persist official OpenArm video frame")
        frame_hashes.append(_sha256(frame_path))
        robot_mask_sha256 = hashlib.sha256(
            np.ascontiguousarray(robot_mask, dtype=np.uint8).tobytes()
        ).hexdigest()
        robot_mask_hashes.append(robot_mask_sha256)
        if frame_index % frames_per_segment_minus_endpoint == 0:
            trajectory_mask_keyframes.append(
                {
                    "frame_index": frame_index,
                    "joint_positions_rad": np.asarray(joints, dtype=np.float64).copy(),
                    "robot_mask": np.asarray(robot_mask, dtype=bool).copy(),
                    "motion_phase": (
                        "initial_home" if frame_index == 0 else motion_phase
                    ),
                }
            )
        np.savez_compressed(
            frame_product_root / f"frame_{frame_index:05d}.npz",
            depth_m=np.asarray(depth, dtype=np.float32),
            robot_mask=np.asarray(robot_mask, dtype=np.uint8),
            human_mask=np.asarray(human_mask, dtype=np.uint8),
            joint_positions_rad=np.asarray(joints, dtype=np.float64),
            observed_joint_positions_rad=np.asarray(
                readback["observed_left"], dtype=np.float64
            ),
            projected_points_px=np.asarray(projected, dtype=np.float64),
            point_camera_depth_m=np.asarray(camera_points[:, 2], dtype=np.float64),
            rendered_observation_camera_m=np.asarray(
                rendered_observations["camera_points_m"], dtype=np.float64
            ),
            rendered_observation_available=np.asarray(
                rendered_observations["available"], dtype=bool
            ),
            rendered_observation_sample_pixels=np.asarray(
                rendered_observations["sample_pixels"], dtype=np.float64
            ),
            rendered_observation_sample_distance_px=np.asarray(
                rendered_observations["sample_distance_px"], dtype=np.float64
            ),
            rendered_observation_sample_depth_m=np.asarray(
                rendered_observations["sample_depth_m"], dtype=np.float64
            ),
            rendered_observation_source=np.asarray(
                rendered_observations["source"]
            ),
            rendered_geometry_residual_m=np.asarray(
                geometry_inliers["residual_m"], dtype=np.float64
            ),
            rendered_geometry_inlier_mask=np.asarray(
                geometry_inliers["inlier_mask"], dtype=bool
            ),
            rendered_geometry_inlier_threshold_m=np.asarray(
                geometry_inliers["threshold_m"], dtype=np.float64
            ),
            rendered_geometry_initial_transform=np.asarray(
                geometry_inliers["initial_fit"].transform, dtype=np.float64
            ),
            rendered_geometry_initial_fit_recoverable=np.asarray(
                bool(np.isfinite(geometry_inliers["residual_m"]).any()),
                dtype=bool,
            ),
            visible_fraction=np.asarray(
                [item.visible_fraction for item in visibility], dtype=np.float64
            ),
            robot_front_fraction=np.asarray(
                [item.robot_front_fraction for item in visibility], dtype=np.float64
            ),
            valid_depth_fraction=np.asarray(
                [item.valid_depth_fraction for item in visibility], dtype=np.float64
            ),
            point_names=np.asarray(experiment.POINT_NAMES),
            point_geometry_transform=np.asarray(
                [row.transform for row in point_geometry_results], dtype=np.float64
            ),
            point_geometry_target_error_p95_m=np.asarray(
                [row.target_error_p95_m for row in point_geometry_results],
                dtype=np.float64,
            ),
            point_geometry_surface_accepted=np.asarray(
                [row.accepted for row in point_geometry_results], dtype=bool
            ),
            point_geometry_failure_reasons=np.asarray(
                ["+".join(row.failure_reasons) for row in point_geometry_results],
                dtype="<U160",
            ),
            point_policy_state=np.asarray(
                [row["policy_state"] for row in point_policy_frame]
            ),
            point_estimate_valid=np.asarray(
                [row["estimate_valid"] for row in point_policy_frame], dtype=bool
            ),
            point_history_valid=np.asarray(
                [row["history_available"] for row in point_policy_frame], dtype=bool
            ),
            point_estimate_camera_m=np.asarray(
                [
                    [
                        row["estimate_camera_x_m"],
                        row["estimate_camera_y_m"],
                        row["estimate_camera_z_m"],
                    ]
                    if row["estimate_valid"]
                    else [np.nan, np.nan, np.nan]
                    for row in point_policy_frame
                ],
                dtype=np.float64,
            ),
            point_predicted_error_p95_m=np.asarray(
                [row["predicted_error_p95_m"] for row in point_policy_frame],
                dtype=np.float64,
            ),
            point_last_valid_age_s=np.asarray(
                [
                    (
                        row["last_valid_age_s"]
                        if row["last_valid_age_s"] is not None
                        else np.nan
                    )
                    for row in point_policy_frame
                ],
                dtype=np.float64,
            ),
            point_direct_update_accepted=np.asarray(
                [row["direct_update_accepted"] for row in point_policy_frame],
                dtype=bool,
            ),
            point_target_geometry_inlier=np.asarray(
                [row["target_geometry_inlier"] for row in point_policy_frame],
                dtype=bool,
            ),
            point_policy_robot_motion_unchanged=np.ones(
                len(experiment.POINT_NAMES), dtype=bool
            ),
            support_radius_px=np.asarray(support_radius, dtype=np.int64),
            joint_time_s=np.asarray(time_s, dtype=np.float64),
            frame_index=np.asarray(frame_index, dtype=np.int64),
            scene_identity=np.asarray(
                f"trajectory__{experiment.POINT_NAMES[target_index]}__wrist{wrist_index:02d}"
                f"__{motion_phase}__frame{frame_index:05d}"
            ),
        )
        max_single_point_occlusion = max(
            max_single_point_occlusion,
            frame_max_single_point_occlusion,
        )
        if robot_front_sum > best_occlusion[0]:
            best_occlusion = (robot_front_sum, frame_path)
        per_frame.append(
            {
                "frame_index": frame_index,
                "joint_time_s": float(time_s),
                "target_name": experiment.POINT_NAMES[target_index],
                "wrist_index": wrist_index,
                "trajectory_group_index": trajectory_group_index,
                "motion_phase": motion_phase,
                "policy_state": selected_point_policy["policy_state"],
                "policy_state_semantics": "worst_of_10_report_only",
                "policy_point_name": selected_point_policy["point_name"],
                "predicted_error_p95_m": selected_point_policy[
                    "predicted_error_p95_m"
                ],
                "direct_point_count": point_state_counts[
                    "DIRECT_GEOMETRY_UPDATE"
                ],
                "freeze_point_count": point_state_counts[
                    "BOUNDED_PATIENT_FRAME_FREEZE"
                ],
                "lost_point_count": point_state_counts["LOST"],
                "robot_motion_policy": "unchanged_by_point_visibility_policy",
                "geometry_accepted": geometry_result.accepted,
                "nominal_target_direct_update_accepted": point_policy_frame[
                    target_index
                ]["direct_update_accepted"],
                "geometry_decision": geometry_result.decision,
                "geometry_failure_reasons": "+".join(geometry_result.failure_reasons),
                "geometry_effective_support": geometry_result.effective_support,
                "geometry_target_error_p95_m": (
                    float(geometry_result.target_error_p95_m)
                    if math.isfinite(geometry_result.target_error_p95_m)
                    else None
                ),
                "visible_point_count_report_only": visible_report_only,
                "rendered_geometry_inlier_count": int(
                    np.count_nonzero(geometry_inliers["inlier_mask"])
                ),
                "rendered_geometry_inlier_threshold_m": geometry_inliers[
                    "threshold_m"
                ],
                "rendered_geometry_initial_failure_reasons": "+".join(
                    geometry_inliers["initial_fit"].failure_reasons
                ),
                "robot_front_fraction_sum": robot_front_sum,
                "max_single_point_robot_front_fraction": (
                    frame_max_single_point_occlusion
                ),
                "robot_pixels": int(np.count_nonzero(robot_mask)),
                "human_pixels": int(np.count_nonzero(human_mask)),
                "robot_mask_sha256": robot_mask_sha256,
                "left_readback_max_abs_error_rad": readback[
                    "left_max_abs_error_rad"
                ],
                "right_max_abs_drift_rad": readback["right_max_abs_drift_rad"],
                "render_sync_method": render_sync["method"],
                "render_sync_time_advance_s": (
                    float(render_sync["simulation_time_after_s"])
                    - float(render_sync["simulation_time_before_s"])
                ),
            }
        )

    if len(set(frame_hashes)) < 2:
        raise RuntimeError("official OpenArm evidence frames are not temporally changing")
    trajectory_mask_gate = _validated_trajectory_mask_keyframes(
        trajectory_mask_keyframes
    )
    if len(set(robot_mask_hashes)) < 3:
        raise RuntimeError("official OpenArm rendered mask did not follow the joint trajectory")
    if best_occlusion[1] is None:
        raise RuntimeError("could not select an occlusion report frame")
    targeted_occlusion = _targeted_occlusion_status(
        max_robot_front_fraction_sum=best_occlusion[0],
        max_single_point_robot_front_fraction=max_single_point_occlusion,
    )
    media_tools = runtime_provenance.get("media_tools")
    if not isinstance(media_tools, Mapping):
        raise RuntimeError("full experiment requires verified media-tool provenance")
    ffmpeg_provenance = media_tools.get("ffmpeg")
    ffprobe_provenance = media_tools.get("ffprobe")
    if not isinstance(ffmpeg_provenance, Mapping) or not isinstance(
        ffprobe_provenance, Mapping
    ):
        raise RuntimeError("full experiment requires verified media-tool provenance")
    shutil.copyfile(best_occlusion[1], out / "report_frame.png")
    video_path = out / "official_openarm_occlusion_h264.mp4"
    video_probe = _encode_h264(
        frame_root,
        video_path,
        fps=12,
        expected_width=args.width,
        expected_height=args.height,
        expected_frames=len(per_frame),
        ffmpeg_provenance=ffmpeg_provenance,
        ffprobe_provenance=ffprobe_provenance,
    )

    csv_fields = list(per_frame[0])
    with (raw_root / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(per_frame)
    point_policy_validation = _persist_and_verify_per_point_policy(
        per_point_policy,
        csv_path=raw_root / "per_point_policy.csv",
        npz_path=raw_root / "per_point_policy.npz",
        expected_point_names=experiment.POINT_NAMES,
    )
    if (
        point_policy_validation["frame_count"] != len(per_frame)
        or point_policy_validation["row_count"]
        != len(per_frame) * len(experiment.POINT_NAMES)
    ):
        raise RuntimeError("persisted per-point policy frame count is inconsistent")
    geometry_inlier_product_validation = (
        _verify_rendered_geometry_inlier_frame_products(
            frame_product_root,
            per_point_policy,
            frame_count=len(per_frame),
        )
    )
    shutil.copyfile(manifest_path, raw_root / "asset_manifest.json")
    completed_strata = sum(
        row["status"] == "kinematic_phases_complete" for row in ik_rows
    )
    aggregate_policy_counts = {
        state: sum(row["policy_state"] == state for row in per_frame)
        for state in (
            "DIRECT_GEOMETRY_UPDATE",
            "BOUNDED_PATIENT_FRAME_FREEZE",
            "LOST",
        )
    }
    point_policy_counts = point_policy_validation["point_frame_state_counts"]
    frames_with_point_state = point_policy_validation["frames_with_point_state"]
    for frame_row, state_counts in zip(
        per_frame, point_policy_validation["per_frame_state_counts"]
    ):
        if (
            frame_row["direct_point_count"]
            != state_counts["DIRECT_GEOMETRY_UPDATE"]
            or frame_row["freeze_point_count"]
            != state_counts["BOUNDED_PATIENT_FRAME_FREEZE"]
            or frame_row["lost_point_count"] != state_counts["LOST"]
        ):
            raise RuntimeError("per-frame and persisted point-state counts differ")
    selected_policy_episode = point_policy_validation["selected_episode"]
    policy_trace_observed = selected_policy_episode["observed"]
    if not targeted_occlusion["observed"]:
        result_status = "NO_TARGETED_OCCLUSION_EVIDENCE"
    elif not policy_trace_observed:
        result_status = "NO_DIRECT_FREEZE_LOST_TRACE"
    elif completed_strata == len(experiment.POINT_NAMES):
        result_status = "PRELIMINARY_OFFICIAL_MODEL_RESULT_INCOMPLETE"
    else:
        result_status = "INCOMPLETE_IK_STRATA"
    metrics = {
        "schema": "wam_official_openarm_occlusion_result_v3",
        "status": result_status,
        "scope": "parameter_sensitivity_only",
        "production_activation": False,
        "physical_accuracy_pass": False,
        "runtime_provenance": dict(runtime_provenance),
        "official_robot": {
            "source": "enactic/openarm_description",
            "commit": "846a82d8baf64efdbb7f55c7f64b9b004939061b",
            "revision": "OpenArm v1.0 full dual arm",
            "root_scale": 1.0,
            "primitive_proxy_used": False,
        },
        "official_human": {
            "source": "NVIDIA Isaac 5.1 male_adult_medical_01",
            "pose": "static_official_T_pose",
            "animation_retargeted": False,
            "limitation": "official idle skeleton is incompatible without a separately validated retarget",
        },
        "camera": {
            "source_intrinsics": experiment.ORBBEC_RGB_INTRINSICS,
            "render_intrinsics": _scaled_intrinsics(args.width, args.height),
            "pose_status": "installation_pose_uncommissioned",
            "distortion_available": False,
            "camera_to_base_covariance_available": False,
        },
        "strata": {
            "target_count": len(experiment.POINT_NAMES),
            "required_wrist_attitudes_per_target": 2,
            "targets_with_two_solutions": completed_strata,
            "static_render_count": len(static_rows),
            "rows": static_rows,
        },
        "trajectory": {
            **trajectory_gate,
            "video_frame_count": len(per_frame),
            "video_fps": 12,
            "video_time_is_compressed": True,
            "rendered_mask_keyframe_gate": trajectory_mask_gate,
        },
        "render_sync": {
            "method": "zero_physics_dt_step_update_fabric_then_render",
            "per_command_timing_and_joint_readback_fail_closed": True,
            "mask_motion_gate_scope": "segment_endpoint_keyframes",
            "sync_count": render_sync_count,
            "max_simulation_time_advance_s": render_sync_max_time_advance_s,
            "physics_dt_restored_after_each_sync": True,
            "physics_integration_used": False,
            "claim_scope": "visual_pose_and_fabric_flush_only",
            "zero_time_capture_count": zero_time_capture_count,
            "zero_time_capture_max_world_delta_s": (
                zero_time_capture_max_world_delta_s
            ),
            "zero_time_capture_max_timeline_delta_s": (
                zero_time_capture_max_timeline_delta_s
            ),
            "rt_subframes_role": "same_timeline_instant_rtx_accumulation",
        },
        "policy_comparison": {
            "selected_experimental_mechanism": (
                "direct_geometry_then_bounded_patient_frame_freeze_then_lost"
                if policy_trace_observed
                else None
            ),
            "candidate_mechanism": "direct_geometry_then_bounded_patient_frame_freeze_then_lost",
            "patient_motion_bound_m_s": 0.010,
            "max_freeze_age_s": 0.40,
            "max_freeze_error_p95_m": 0.010,
            "policy_scope": "independent_state_per_auscultation_point",
            "aggregate_frame_state_semantics": "worst_of_10_report_only",
            "aggregate_frame_state_counts": aggregate_policy_counts,
            "point_frame_state_counts": point_policy_counts,
            "frames_with_point_state": frames_with_point_state,
            "per_point_evidence_rows": len(per_point_policy),
            "per_point_evidence_csv": "raw/per_point_policy.csv",
            "per_point_evidence_npz": "raw/per_point_policy.npz",
            "per_point_artifact_contract": point_policy_validation,
            "direct_estimate_source": (
                "rendered_depth_se3_transform_of_canonical_reference_point"
            ),
            "rendered_observation_source": (
                "nearest_unoccluded_human_instance_depth_sample"
            ),
            "rendered_geometry_inlier_selection": {
                "method": "one_robust_se3_then_single_pass_residual_trim_then_full_refit",
                "initial_fit_robust_delta_m": 0.004,
                "residual_definition": "euclidean_observed_xyz_minus_initial_fitted_reference_xyz",
                "threshold_m": geometry_inlier_product_validation["threshold_m"],
                "comparison": geometry_inlier_product_validation["comparison"],
                "raw_observation_availability_preserved": (
                    geometry_inlier_product_validation[
                        "raw_observation_mask_bound"
                    ]
                ),
                "target_direct_update_requires_target_inlier": True,
                "frame_level_common_inlier_mask": True,
                "fixed_inlier_count_rule_used": False,
                "final_refit_uses_complete_unchanged_decision_surface": _json_safe(
                    _decision_surface()
                ),
                "frame_product_validation": geometry_inlier_product_validation,
                "threshold_selection_scope": "post_hoc_candidate_from_predecessor_same_trajectory_sweep",
                "threshold_is_held_out": False,
                "predecessor_sweep_thresholds_m": [
                    0.004,
                    0.006,
                    0.008,
                    0.010,
                    0.012,
                    0.015,
                    0.020,
                ],
                "physical_accuracy_claim": False,
            },
            "projected_anchor_source": "official_mesh_oracle_2d_center",
            "oracle_2d_conditioned": True,
            "freeze_estimate_source": "last_valid_patient_frame_lock",
            "lost_estimate_valid": False,
            "history_validity_mask_persisted": point_policy_validation[
                "checks"
            ]["mask_nan_contract"],
            "estimate_validity_mask_persisted": point_policy_validation[
                "checks"
            ]["mask_nan_contract"],
            "robot_motion_policy": "unchanged_by_point_visibility_policy",
            "direct_freeze_lost_trace_observed": policy_trace_observed,
            "selected_episode": selected_policy_episode,
            "fixed_visible_point_count_rule_used": False,
            "visible_count_is_report_only": True,
            "patient_frame_motion_model": "static_official_human_with_declared_motion_bound_only",
            "physical_detector_in_loop": False,
        },
        "targeted_occlusion_evidence": targeted_occlusion,
        "collision_status": {
            "physical_contact_safety_evaluated": False,
            "reason": "official_visual_human_has_no_commissioned_collision_or_tissue_contract",
            "production_contact_motion_allowed": False,
        },
        "completion_gates": {
            "official_openarm_geometry_rendered": True,
            "instance_mask_depth_products_persisted": True,
            "targeted_occlusion_observed": targeted_occlusion["observed"],
            "direct_freeze_lost_trace_observed": policy_trace_observed,
            "per_point_policy_rows_complete": point_policy_validation[
                "checks"
            ]["frame_point_cartesian_complete"],
            "per_point_validity_masks_persisted": point_policy_validation[
                "checks"
            ]["mask_nan_contract"],
            "per_point_artifact_contract_verified": all(
                point_policy_validation["checks"].values()
            ),
            "rendered_geometry_inlier_products_verified": (
                geometry_inlier_product_validation["verified_frame_count"]
                == len(per_frame)
                and geometry_inlier_product_validation[
                    "verified_point_frame_count"
                ]
                == len(per_point_policy)
                and geometry_inlier_product_validation[
                    "raw_observation_mask_bound"
                ]
                is True
                and geometry_inlier_product_validation["policy_rows_bound"]
                is True
            ),
            "robot_motion_unchanged_by_visibility_policy": point_policy_validation[
                "checks"
            ]["robot_motion_unchanged"],
            "all_targets_two_wrist_solutions": completed_strata == len(experiment.POINT_NAMES),
            "approach_contact_depart_all_strata": completed_strata == len(experiment.POINT_NAMES),
            "commissioned_collision_penetration_contract": False,
            "camera_base_pose_grid_complete": False,
            "thirty_holdout_trajectories_per_target_orientation": False,
            "bootstrap_confidence_intervals_complete": False,
            "noise_bias_dilation_delay_patient_motion_sweeps_complete": False,
            "all_1024_landmark_masks_enumerated": False,
            "physical_detector_in_loop_replay_complete": False,
            "result_complete": False,
        },
        "video_probe": video_probe,
    }
    _write_json(raw_root / "metrics.json", _json_safe(metrics))
    _write_json(
        out / "run_status.json",
        {
            "status": metrics["status"],
            "scope": metrics["scope"],
            "production_activation": False,
            "video_sha256": _sha256(video_path),
            "report_frame_sha256": _sha256(out / "report_frame.png"),
            "metrics_sha256": _sha256(raw_root / "metrics.json"),
        },
    )
    return metrics


def _run(
    args: argparse.Namespace,
    *,
    runtime_provenance: Mapping[str, object],
) -> dict:
    import carb
    import cv2
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.semantics import add_labels
    from pxr import Gf, UsdGeom

    replicator_capture_contract = _configure_replicator_explicit_capture(
        rep,
        carb.settings.get_settings(),
    )

    bundle = Path(args.asset_bundle).resolve(strict=True)
    manifest_path = Path(args.asset_manifest).resolve(strict=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract_path = _python_root() / "ausc/perception/tracking/official_asset_contract.py"
    asset_contract = _load_asset_contract(contract_path)
    manifest = asset_contract.verify_manifest(manifest_path.read_bytes(), bundle)
    if manifest_path.parent != bundle:
        raise RuntimeError("asset manifest must be stored at the frozen asset-bundle root")

    urdf_path = bundle / manifest["robot_source"]["urdf_path"]
    human_usd = bundle / manifest["human_source"]["usd_path"]
    urdf_inventory = experiment.inspect_openarm_urdf(urdf_path.read_bytes())
    imported_usd = out / "openarm_v1_import.usd"
    imported_prim_path = _import_openarm(urdf_path, imported_usd)

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world_prim)
    openarm = stage.DefinePrim("/World/OpenArm", "Xform")
    if not openarm.GetReferences().AddReference(str(imported_usd)):
        raise RuntimeError("could not reference imported official OpenArm USD")
    human = stage.DefinePrim("/World/Human", "Xform")
    if not human.GetReferences().AddReference(str(human_usd)):
        raise RuntimeError("could not reference official NVIDIA human USD")
    _set_root_transform(openarm, translation=(0.15, 1.15, 0.0), yaw_degrees=180.0)
    _set_root_transform(human, translation=(0.0, 0.50, -0.30), yaw_degrees=0.0)
    add_labels(openarm, ["openarm_robot"], "class")
    add_labels(human, ["human"], "class")

    rep.create.light(light_type="dome", intensity=700.0, temperature=5600, name="Fill")
    rep.create.light(
        light_type="distant",
        rotation=(315.0, 20.0, 15.0),
        intensity=2600.0,
        temperature=5200,
        name="Key",
    )
    render_intrinsics = _scaled_intrinsics(args.width, args.height)
    camera_parameters = _usd_camera_parameters(
        render_intrinsics, horizontal_aperture_mm=36.0
    )
    camera_position = (0.0, 2.35, 1.18)
    camera_look_at = (0.0, 0.58, 0.80)
    camera_paths_before = {
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)
    }
    camera = rep.create.camera(
        position=camera_position,
        look_at=camera_look_at,
        focal_length=camera_parameters["focal_length_mm"],
        horizontal_aperture=camera_parameters["horizontal_aperture_mm"],
        horizontal_aperture_offset=camera_parameters["horizontal_aperture_offset_mm"],
        vertical_aperture_offset=camera_parameters["vertical_aperture_offset_mm"],
        clipping_range=(0.05, 10.0),
    )
    camera_paths_after = {
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)
    }
    primary_camera_path = _unique_new_prim_path(
        camera_paths_before, camera_paths_after, label="camera"
    )
    usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(primary_camera_path))
    usd_camera.GetVerticalApertureAttr().Set(
        camera_parameters["vertical_aperture_mm"]
    )
    camera_attributes = {
        "focal_length_mm": float(usd_camera.GetFocalLengthAttr().Get()),
        "horizontal_aperture_mm": float(usd_camera.GetHorizontalApertureAttr().Get()),
        "vertical_aperture_mm": float(usd_camera.GetVerticalApertureAttr().Get()),
        "horizontal_aperture_offset_mm": float(
            usd_camera.GetHorizontalApertureOffsetAttr().Get()
        ),
        "vertical_aperture_offset_mm": float(
            usd_camera.GetVerticalApertureOffsetAttr().Get()
        ),
    }
    if not all(
        math.isclose(camera_attributes[key], value, rel_tol=0.0, abs_tol=1.0e-5)
        for key, value in camera_parameters.items()
    ):
        raise RuntimeError("authored USD camera does not reproduce the frozen fx/fy/cx/cy contract")
    product = rep.create.render_product(camera, (args.width, args.height))
    annotators = {
        "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
        "distance_to_image_plane": rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        ),
        "instance_segmentation": rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation", init_params={"colorize": False}
        ),
        "semantic_segmentation": rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False, "semanticTypes": ["class"]},
        ),
    }
    for annotator in annotators.values():
        annotator.attach(product)
    oblique_camera = rep.create.camera(
        position=(1.45, 1.65, 1.30),
        look_at=(0.0, 0.60, 0.82),
        focal_length=28.0,
        horizontal_aperture=36.0,
        clipping_range=(0.05, 10.0),
    )
    oblique_product = rep.create.render_product(
        oblique_camera,
        (max(160, args.width // 2), max(90, args.height // 2)),
    )
    oblique_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    oblique_rgb_annotator.attach(oblique_product)

    articulation_root_path = _articulation_root_path(openarm)
    world = World(stage_units_in_meters=1.0)
    from isaacsim.core.prims import SingleArticulation

    articulation = world.scene.add(
        SingleArticulation(
            prim_path=articulation_root_path,
            name="official_openarm_v1_articulation",
        )
    )
    world.reset()
    world.pause()
    if world.is_playing():
        raise RuntimeError("Isaac physics timeline could not be paused for deterministic capture")
    timeline = omni.timeline.get_timeline_interface()
    _require_paused_running_timeline(
        _runtime_timing_snapshot(world, timeline),
        label="initial deterministic capture",
    )
    actual_left_indices = [
        articulation.get_dof_index(name) for name in experiment.LEFT_FOLLOWER_JOINTS
    ]
    actual_right_indices = [
        articulation.get_dof_index(name) for name in experiment.RIGHT_REFERENCE_JOINTS
    ]
    if (
        any(index < 0 for index in actual_left_indices + actual_right_indices)
        or len(set(actual_left_indices + actual_right_indices)) != 14
    ):
        raise RuntimeError("imported articulation does not expose both official seven-joint arms")
    warmup_gates = []
    for warmup_index in range(8):
        _unused, warmup_gate = _run_zero_time_operation(
            world,
            timeline,
            lambda: rep.orchestrator.step(
                rt_subframes=max(args.rt_subframes, 1),
                pause_timeline=True,
                delta_time=0.0,
            ),
            label=f"replicator_warmup_{warmup_index:02d}",
        )
        warmup_gates.append(warmup_gate)
    initial_products, initial_capture_gate = _capture_products_zero_time(
        world,
        timeline,
        annotators,
        rt_subframes=args.rt_subframes,
        label="smoke_initial_capture",
    )
    rgb, depth, robot_mask, human_mask, instance_packet, semantic_packet = (
        initial_products
    )
    smoke_oblique_rgb = np.asarray(oblique_rgb_annotator.get_data())
    smoke_left_indices = np.asarray(actual_left_indices, dtype=np.int64)
    smoke_right_indices = np.asarray(actual_right_indices, dtype=np.int64)
    smoke_initial_left = np.asarray(
        articulation.get_joint_positions(joint_indices=smoke_left_indices),
        dtype=np.float64,
    )
    smoke_initial_right = np.asarray(
        articulation.get_joint_positions(joint_indices=smoke_right_indices),
        dtype=np.float64,
    )
    smoke_commanded_left = smoke_initial_left.copy()
    first_joint_limits = experiment.left_joint_limits(urdf_path.read_bytes())[0]
    for delta in (0.45, -0.45):
        candidate = float(smoke_initial_left[0] + delta)
        if first_joint_limits[0] <= candidate <= first_joint_limits[1]:
            smoke_commanded_left[0] = candidate
            break
    if abs(float(smoke_commanded_left[0] - smoke_initial_left[0])) < 0.25:
        raise RuntimeError("could not construct a safe render-synchronization smoke motion")
    articulation.set_joint_positions(
        smoke_commanded_left, joint_indices=smoke_left_indices
    )
    articulation.set_joint_velocities(
        np.zeros(7), joint_indices=smoke_left_indices
    )
    smoke_sync = _synchronize_articulation_render_state(
        world, timeline=timeline
    )
    moved_products, moved_capture_gate = _capture_products_zero_time(
        world,
        timeline,
        annotators,
        rt_subframes=args.rt_subframes,
        label="smoke_moved_capture",
    )
    second_rgb, second_depth, second_robot_mask, second_human_mask, _, _ = (
        moved_products
    )
    second_oblique_rgb = np.asarray(oblique_rgb_annotator.get_data())
    smoke_moved_readback = _validate_joint_readback(
        articulation,
        left_indices=smoke_left_indices,
        right_indices=smoke_right_indices,
        commanded_left=smoke_commanded_left,
        frozen_right=smoke_initial_right,
    )
    articulation.set_joint_positions(
        smoke_initial_left, joint_indices=smoke_left_indices
    )
    articulation.set_joint_velocities(
        np.zeros(7), joint_indices=smoke_left_indices
    )
    smoke_restore_sync = _synchronize_articulation_render_state(
        world, timeline=timeline
    )
    restore_products, restore_capture_gate = _capture_products_zero_time(
        world,
        timeline,
        annotators,
        rt_subframes=args.rt_subframes,
        label="smoke_restored_capture",
    )
    (
        restore_rgb,
        restore_depth,
        restore_robot_mask,
        restore_human_mask,
        _,
        _,
    ) = restore_products
    smoke_restore_readback = _validate_joint_readback(
        articulation,
        left_indices=smoke_left_indices,
        right_indices=smoke_right_indices,
        commanded_left=smoke_initial_left,
        frozen_right=smoke_initial_right,
    )
    smoke_sync_evidence = _validated_render_sync_evidence(
        initial_robot_mask=robot_mask,
        moved_robot_mask=second_robot_mask,
        initial_human_mask=human_mask,
        moved_human_mask=second_human_mask,
        restored_robot_mask=restore_robot_mask,
        restored_human_mask=restore_human_mask,
        initial_joint_positions=smoke_initial_left,
        commanded_joint_positions=smoke_commanded_left,
        observed_joint_positions=smoke_moved_readback["observed_left"],
        sync_evidence=smoke_sync,
    )
    smoke_sync_evidence.update(
        {
            "restore_sync": smoke_restore_sync,
            "restore_readback_max_abs_error_rad": smoke_restore_readback[
                "left_max_abs_error_rad"
            ],
            "zero_time_capture_gates": {
                "warmup": warmup_gates,
                "initial": initial_capture_gate,
                "moved": moved_capture_gate,
                "restored": restore_capture_gate,
            },
        }
    )
    positive_depth = np.isfinite(depth) & (depth > 0.0)
    if not np.any(positive_depth):
        raise RuntimeError("rendered depth has no finite positive foreground")

    anchors, anchor_mesh_path, anchor_vertex_count, anchor_triangle_count = (
        _human_back_anchors(human)
    )
    anchor_positions = np.asarray([anchor.position_m for anchor in anchors])
    camera_points = _world_to_camera(
        anchor_positions, camera_position, camera_look_at
    )
    projected = experiment.project_camera_points(
        camera_points, render_intrinsics
    )
    visibility = experiment.measure_rendered_visibility(
        point_names=experiment.POINT_NAMES,
        projected_pixels=projected,
        point_depth_m=camera_points[:, 2],
        robot_mask=robot_mask,
        human_mask=human_mask,
        depth_m=depth,
        support_radius_px=max(4, int(round(args.width / 80.0))),
        robot_mask_dilation_px=max(1, int(round(args.width / 320.0))),
        front_depth_margin_m=0.003,
    )
    anchor_alignment = _validate_smoke_anchor_alignment(
        projected,
        visibility,
        width=args.width,
        height=args.height,
    )

    robot_mesh_count = _count_meshes(openarm)
    if robot_mesh_count != 42:
        raise RuntimeError(f"imported OpenArm rendered {robot_mesh_count} meshes, expected 42")
    inventory = {
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "import_root_scale": 1.0,
        "link_count": urdf_inventory.link_count,
        "joint_count": urdf_inventory.joint_count,
        "revolute_joint_names": list(urdf_inventory.revolute_joint_names),
        "animated_joint_names": list(experiment.LEFT_FOLLOWER_JOINTS),
        "robot_pixel_count": int(np.count_nonzero(robot_mask)),
        "human_pixel_count": int(np.count_nonzero(human_mask)),
        "human_asset_sha256": _sha256(human_usd),
        "robot_urdf_sha256": _sha256(urdf_path),
    }
    inventory = _validated_stage_inventory(inventory, runtime_provenance)
    inventory.update(
        {
            "asset_manifest_sha256": _sha256(manifest_path),
            "imported_usd_sha256": _sha256(imported_usd),
            "import_command_result": imported_prim_path,
            "articulation_root_path": articulation_root_path,
            "actual_left_revolute_dof_indices": actual_left_indices,
            "actual_right_revolute_dof_indices": actual_right_indices,
            "actual_articulation_dof_names": list(articulation.dof_names),
            "physics_timeline_paused_for_capture": True,
            "render_sync_contract": smoke_sync_evidence,
            "replicator_capture_contract": replicator_capture_contract,
            "camera_render_intrinsics": render_intrinsics,
            "camera_usd_attributes": camera_attributes,
            "primary_camera_prim_path": primary_camera_path,
            "oblique_camera_role": "reference_visualization_only",
            "imported_mesh_count_with_instance_proxies": robot_mesh_count,
            "primary_contract": experiment.primary_experiment_contract(),
            "rgb_shape": list(rgb.shape),
            "rgb_dtype": str(rgb.dtype),
            "depth_shape": list(depth.shape),
            "depth_dtype": str(depth.dtype),
            "finite_positive_depth_pixels": int(np.count_nonzero(positive_depth)),
            "depth_min_m": float(np.min(depth[positive_depth])),
            "depth_max_m": float(np.max(depth[positive_depth])),
            "anchor_mesh_path": anchor_mesh_path,
            "anchor_vertex_count": anchor_vertex_count,
            "anchor_triangle_count": anchor_triangle_count,
            "smoke_anchor_alignment": anchor_alignment,
        }
    )
    _write_json(out / "stage_inventory.json", inventory)
    _write_json(
        out / "annotation_mappings.json",
        {
            "instance_info": _json_safe(instance_packet.get("info", {})),
            "semantic_info": _json_safe(semantic_packet.get("info", {})),
        },
    )
    np.save(out / "smoke_depth.npy", depth.astype(np.float32))
    np.save(out / "smoke_robot_mask.npy", robot_mask)
    np.save(out / "smoke_human_mask.npy", human_mask)
    if not cv2.imwrite(
        str(out / "smoke_oblique_frame.png"),
        cv2.cvtColor(smoke_oblique_rgb[..., :3], cv2.COLOR_RGB2BGR),
    ):
        raise RuntimeError("could not write oblique smoke frame")
    np.save(out / "smoke_frame_001_depth.npy", second_depth.astype(np.float32))
    np.save(out / "smoke_frame_001_robot_mask.npy", second_robot_mask)
    np.save(out / "smoke_frame_001_human_mask.npy", second_human_mask)
    np.save(out / "smoke_restore_depth.npy", restore_depth.astype(np.float32))
    np.save(out / "smoke_restore_robot_mask.npy", restore_robot_mask)
    np.save(out / "smoke_restore_human_mask.npy", restore_human_mask)
    if not cv2.imwrite(
        str(out / "smoke_restore_frame.png"),
        cv2.cvtColor(restore_rgb[..., :3], cv2.COLOR_RGB2BGR),
    ):
        raise RuntimeError("could not write restored smoke frame")
    bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
    cv2.putText(
        bgr,
        "OpenArm v1 commit 846a82d8baf + NVIDIA medical human",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )
    anchor_rows = []
    for anchor, pixel, row in zip(anchors, projected, visibility):
        color = (60, 220, 60)
        if row.robot_front_fraction >= 0.25:
            color = (40, 40, 235)
        elif row.visible_fraction < 0.35:
            color = (40, 210, 235)
        center = tuple(int(round(value)) for value in pixel)
        cv2.circle(bgr, center, 7, (15, 15, 15), 3)
        cv2.circle(bgr, center, 5, color, -1)
        cv2.putText(
            bgr,
            anchor.name,
            (center[0] + 7, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
        anchor_rows.append(
            {
                "name": anchor.name,
                "triangle_id": anchor.triangle_id,
                "barycentric": anchor.barycentric.tolist(),
                "position_m": anchor.position_m.tolist(),
                "surface_normal": anchor.surface_normal.tolist(),
                "projected_pixel": pixel.tolist(),
                "camera_depth_m": float(camera_points[len(anchor_rows), 2]),
                "visible_fraction": row.visible_fraction,
                "robot_front_fraction": row.robot_front_fraction,
                "valid_depth_fraction": row.valid_depth_fraction,
                "mask_edge_distance_px": row.mask_edge_distance_px,
                "visibility_source": row.visibility_source,
            }
        )
    _write_json(
        out / "ten_point_anchors.json",
        {
            "identity_order": list(experiment.POINT_NAMES),
            "mesh_path": anchor_mesh_path,
            "source": "normalized_wam_identity_transferred_to_official_nvidia_mesh",
            "anatomical_ground_truth_claim": False,
            "anchors": anchor_rows,
        },
    )
    if not cv2.imwrite(str(out / "smoke_frame.png"), bgr):
        raise RuntimeError("could not write smoke frame")
    second_visibility = experiment.measure_rendered_visibility(
        point_names=experiment.POINT_NAMES,
        projected_pixels=projected,
        point_depth_m=camera_points[:, 2],
        robot_mask=second_robot_mask,
        human_mask=second_human_mask,
        depth_m=second_depth,
        support_radius_px=max(4, int(round(args.width / 80.0))),
        robot_mask_dilation_px=max(1, int(round(args.width / 320.0))),
        front_depth_margin_m=0.003,
    )
    second_bgr = _draw_frame(
        second_rgb,
        projected,
        second_visibility,
        title="OpenArm v1 commit 846a82d8baf | smoke frame 2/2",
        subtitle="zero-dt Fabric sync + instance mask | no proxy geometry",
        oblique_rgb=second_oblique_rgb,
    )
    if not cv2.imwrite(str(out / "smoke_frame_001.png"), second_bgr):
        raise RuntimeError("could not write second smoke frame")
    if args.mode == "full":
        metrics = _run_full_experiment(
            args=args,
            world=world,
            timeline=timeline,
            articulation=articulation,
            annotators=annotators,
            oblique_rgb_annotator=oblique_rgb_annotator,
            anchors=anchors,
            projected=projected,
            camera_points=camera_points,
            urdf_path=urdf_path,
            manifest_path=manifest_path,
            out=out,
            runtime_provenance=runtime_provenance,
        )
        status = metrics["status"]
    else:
        status = "SMOKE_PASS"
    summary = {
        "status": status,
        "mode": args.mode,
        "out": str(out),
        "robot_pixels": inventory["robot_pixel_count"],
        "human_pixels": inventory["human_pixel_count"],
        "mesh_count": robot_mesh_count,
    }
    # This is intentionally the final write inside the payload tree.  The
    # process-terminal marker is a sibling and binds this exact inventory.
    summary["output_integrity"] = _write_sha256s(out)
    return summary


def _simulation_app_config(args: argparse.Namespace) -> dict:
    return {
        "headless": True,
        "hide_ui": True,
        "multi_gpu": False,
        "active_gpu": 0,
        "physics_gpu": 0,
        "anti_aliasing": 0,
        "samples_per_pixel_per_frame": 1,
        "disable_viewport_updates": False,
        # The isolated process persists its terminal result before invoking Isaac's
        # immediate shutdown. Failure paths bypass unsafe extension teardown and exit 1.
        "fast_shutdown": True,
        "width": args.width,
        "height": args.height,
    }


def _process_terminal_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    if not output.name or output.name in (".", ".."):
        raise RuntimeError("simulation output path cannot bind a terminal marker")
    return output.parent / f".{output.name}.process-terminal.json"


def _validated_supervisor_completion(
    terminal_path: Path,
    systemd_properties: Mapping[str, object],
    output_root: Path,
) -> dict:
    """Bind payload bytes, the terminal marker, and the observed process result."""

    output_root = Path(output_root)
    terminal_path = Path(terminal_path)
    if not output_root.is_absolute() or not terminal_path.is_absolute():
        raise RuntimeError("supervisor completion paths must be absolute")
    if terminal_path != _process_terminal_path(output_root):
        raise RuntimeError("supervisor completion terminal path is not bound to output")
    try:
        terminal_info = terminal_path.lstat()
    except OSError as exc:
        raise RuntimeError("supervisor completion terminal marker is unavailable") from exc
    if stat.S_ISLNK(terminal_info.st_mode) or not stat.S_ISREG(terminal_info.st_mode):
        raise RuntimeError("supervisor completion terminal marker must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(terminal_path, flags)
    except OSError as exc:
        raise RuntimeError("supervisor completion terminal marker could not be pinned") from exc
    terminal_failures: list[tuple[str, BaseException, object]] = []
    terminal_raw: Optional[bytes] = None
    try:
        opened_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_info.st_mode)
            or _media_identity(opened_info) != _media_identity(terminal_info)
        ):
            raise RuntimeError("supervisor completion terminal marker changed before reading")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > 1024 * 1024:
                raise RuntimeError("supervisor completion terminal marker is oversized")
            chunks.append(chunk)
        after_info = os.fstat(descriptor)
        if _media_identity(after_info) != _media_identity(terminal_info):
            raise RuntimeError("supervisor completion terminal marker changed while reading")
        terminal_raw = b"".join(chunks)
    except BaseException as exc:
        terminal_failures.append(
            ("supervisor completion terminal read failed", exc, exc.__traceback__)
        )
    try:
        os.close(descriptor)
    except BaseException as exc:
        terminal_failures.append(
            ("supervisor completion terminal close failed", exc, exc.__traceback__)
        )
    _raise_first_failure(terminal_failures)
    if terminal_raw is None:
        raise RuntimeError("supervisor completion terminal marker is empty")
    try:
        terminal = json.loads(terminal_raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("supervisor completion terminal marker is invalid JSON") from exc
    if not isinstance(terminal, Mapping):
        raise RuntimeError("supervisor completion terminal marker must be an object")

    expected_out = str(output_root)
    mode = terminal.get("mode")
    summary = terminal.get("summary")
    shutdown = terminal.get("shutdown")
    if (
        terminal.get("schema") != "wam_isolated_simulation_process_terminal_v1"
        or terminal.get("terminal_state")
        != "RUN_BODY_PASS_AWAITING_SUPERVISOR_EXIT"
        or type(terminal.get("expected_process_exit_code")) is not int
        or terminal.get("expected_process_exit_code") != 0
        or terminal.get("production_activation") is not False
        or mode not in ("smoke", "full")
        or terminal.get("out") != expected_out
        or not isinstance(summary, Mapping)
    ):
        raise RuntimeError("supervisor completion terminal contract failed")
    expected_shutdown = {
        "method": "SimulationApp.close",
        "wait_for_replicator": False,
        "skip_cleanup": True,
        "fast_shutdown": True,
        "supervisor_exit_status_required": True,
    }
    if not isinstance(shutdown, Mapping) or dict(shutdown) != expected_shutdown:
        raise RuntimeError("supervisor completion terminal shutdown contract failed")
    if (
        summary.get("mode") != mode
        or summary.get("out") != expected_out
        or not isinstance(summary.get("status"), str)
        or not summary.get("status")
        or not isinstance(summary.get("output_integrity"), Mapping)
    ):
        raise RuntimeError("supervisor completion terminal summary binding failed")
    if mode == "smoke" and summary.get("status") != "SMOKE_PASS":
        raise RuntimeError("supervisor completion smoke status is not a pass")

    required_properties = {
        # The audited launcher uses RemainAfterExit=yes so systemd retains the
        # real waitid result instead of garbage-collecting the transient unit
        # and returning a misleading all-zero/not-found property set.
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "exited",
        "Result": "success",
    }
    for name, expected_value in required_properties.items():
        if systemd_properties.get(name) != expected_value:
            raise RuntimeError("supervisor systemd process result is not successful")

    def is_exact_zero(value: object) -> bool:
        return (type(value) is int and value == 0) or value == "0"

    def is_cld_exited(value: object) -> bool:
        # Raw `systemctl show` exposes waitid(2) CLD_EXITED as numeric 1.
        return (type(value) is int and value == 1) or value == "1"

    if not is_cld_exited(systemd_properties.get("ExecMainCode")) or not is_exact_zero(
        systemd_properties.get("ExecMainStatus")
    ):
        raise RuntimeError("supervisor systemd process did not exit normally with status zero")

    output_integrity = _verify_output_integrity(output_root)
    if dict(summary["output_integrity"]) != output_integrity:
        raise RuntimeError("supervisor completion output integrity binding failed")
    return {
        "schema": "wam_validated_supervisor_completion_v1",
        "terminal": dict(terminal),
        "systemd": dict(systemd_properties),
        "output_integrity": output_integrity,
    }


def _write_process_terminal(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError("process terminal parent is unavailable")
    if os.path.lexists(destination):
        raise RuntimeError("process terminal marker already exists")
    raw = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(
        f"{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    failures: list[tuple[str, BaseException, object]] = []
    descriptor: Optional[int] = None
    directory_descriptor: Optional[int] = None
    temporary_exists = False
    committed = False
    try:
        descriptor = os.open(temporary, flags, 0o644)
        temporary_exists = True
    except BaseException as exc:
        failures.append(("process terminal temporary open failed", exc, exc.__traceback__))

    if descriptor is not None:
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RuntimeError("process terminal write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException as exc:
            failures.append(("process terminal write failed", exc, exc.__traceback__))
        try:
            os.close(descriptor)
        except BaseException as exc:
            failures.append(
                ("process terminal descriptor close failed", exc, exc.__traceback__)
            )

    if not failures:
        try:
            # A same-filesystem hard link is an atomic no-clobber commit: unlike
            # os.replace(), it can never overwrite a racing terminal marker.
            os.link(temporary, destination, follow_symlinks=False)
            committed = True
        except FileExistsError as exc:
            error = RuntimeError(
                "process terminal marker already exists during atomic commit"
            )
            error.__cause__ = exc
            failures.append(("process terminal atomic commit failed", error, error.__traceback__))
        except BaseException as exc:
            error = RuntimeError("process terminal atomic commit failed")
            error.__cause__ = exc
            failures.append(("process terminal atomic commit failed", error, error.__traceback__))

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        directory_descriptor = os.open(parent, directory_flags)
    except BaseException as exc:
        failures.append(
            ("process terminal directory open failed", exc, exc.__traceback__)
        )

    if committed and directory_descriptor is not None:
        try:
            os.fsync(directory_descriptor)
        except BaseException as exc:
            failures.append(
                ("process terminal commit directory fsync failed", exc, exc.__traceback__)
            )

    if temporary_exists:
        try:
            temporary.unlink()
            temporary_exists = False
        except BaseException as exc:
            failures.append(
                ("process terminal temporary cleanup failed", exc, exc.__traceback__)
            )

    if directory_descriptor is not None:
        try:
            os.fsync(directory_descriptor)
        except BaseException as exc:
            failures.append(
                ("process terminal cleanup directory fsync failed", exc, exc.__traceback__)
            )
        try:
            os.close(directory_descriptor)
        except BaseException as exc:
            failures.append(
                ("process terminal directory close failed", exc, exc.__traceback__)
            )

    if committed:
        try:
            info = destination.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or destination.read_bytes() != raw
            ):
                raise RuntimeError("process terminal marker verification failed")
        except BaseException as exc:
            failures.append(
                ("process terminal marker verification failed", exc, exc.__traceback__)
            )

    _raise_first_failure(failures)


def _run_isolated_simulation_process(
    args: argparse.Namespace,
    simulation_app,
    runtime_provenance: Mapping[str, object],
    *,
    exit_process: Callable[[int], object] = os._exit,
    stdout=None,
    stderr=None,
) -> None:
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    terminal_path = _process_terminal_path(args.out)

    def hard_exit(code: int, diagnostic_error: Optional[BaseException] = None) -> None:
        if diagnostic_error is not None:
            try:
                traceback.print_exception(
                    type(diagnostic_error),
                    diagnostic_error,
                    diagnostic_error.__traceback__,
                    file=stderr,
                )
            except BaseException:
                pass
        for stream in (stdout, stderr):
            try:
                stream.flush()
            except BaseException:
                pass
        exit_process(code)
        raise RuntimeError(f"process exit {code} unexpectedly returned")

    def exit_failure(error: BaseException, *, state: str) -> None:
        payload = {
            "schema": "wam_isolated_simulation_process_terminal_v1",
            "terminal_state": state,
            "mode": args.mode,
            "out": str(args.out),
            "expected_process_exit_code": 1,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "production_activation": False,
        }
        try:
            _write_process_terminal(terminal_path, payload)
        except BaseException as marker_error:
            _add_exception_note(
                error,
                label="could not persist process terminal failure",
                error=marker_error,
            )
        hard_exit(1, error)

    try:
        summary = _run(args, runtime_provenance=runtime_provenance)
    except BaseException as exc:
        exit_failure(exc, state="RUN_BODY_FAILED")
        return
    if not isinstance(summary, dict):
        exit_failure(
            RuntimeError("official OpenArm experiment returned no completion summary"),
            state="RUN_BODY_FAILED",
        )
        return

    success_payload = {
        "schema": "wam_isolated_simulation_process_terminal_v1",
        "terminal_state": "RUN_BODY_PASS_AWAITING_SUPERVISOR_EXIT",
        "mode": args.mode,
        "out": str(args.out),
        "expected_process_exit_code": 0,
        "summary": summary,
        "shutdown": {
            "method": "SimulationApp.close",
            "wait_for_replicator": False,
            "skip_cleanup": True,
            "fast_shutdown": True,
            "supervisor_exit_status_required": True,
        },
        "production_activation": False,
    }
    try:
        _write_process_terminal(terminal_path, success_payload)
    except BaseException as exc:
        exit_failure(exc, state="TERMINAL_SUCCESS_WRITE_FAILED")
        return
    try:
        print(json.dumps(summary, sort_keys=True), file=stdout, flush=True)
        stderr.flush()
    except BaseException as exc:
        hard_exit(1, exc)
        return
    try:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    except BaseException as exc:
        hard_exit(1, exc)
        return
    hard_exit(0)


def main() -> None:
    args = _arguments()
    if args.width <= 0 or args.height <= 0 or args.rt_subframes <= 0:
        raise ValueError("render dimensions/subframes must be positive")
    runtime_provenance = _verified_runtime_provenance()
    if args.mode == "full":
        media_tools = _verified_media_tools(
            ffmpeg_path=args.ffmpeg_path,
            ffmpeg_sha256=args.ffmpeg_sha256,
            ffprobe_path=args.ffprobe_path,
            ffprobe_sha256=args.ffprobe_sha256,
        )
        runtime_provenance = _runtime_provenance_with_media_tools(
            runtime_provenance, media_tools
        )
    from isaacsim import SimulationApp

    terminal_path = _process_terminal_path(args.out)
    if os.path.lexists(terminal_path):
        raise RuntimeError("process terminal marker already exists")
    simulation_app = SimulationApp(_simulation_app_config(args))
    _run_isolated_simulation_process(
        args,
        simulation_app,
        runtime_provenance,
    )


if __name__ == "__main__":
    main()
