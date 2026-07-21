"""Offline, fail-closed contract for the official OpenArm occlusion assets.

The builder accepts only an asset bundle root.  It derives cleanliness from local Git
commands, validates fixed official bytes and dependency evidence, and never opens the camera
or accesses the network.
"""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Dict, List, Mapping, Sequence, Tuple
import xml.etree.ElementTree as ET


SCHEMA = "wam_official_asset_manifest_v1"
ENACTIC_REPOSITORY_URL = "https://github.com/enactic/openarm_description.git"
ENACTIC_COMMIT = "846a82d8baf64efdbb7f55c7f64b9b004939061b"
CHECKOUT_PATH = "openarm_description"
OFFICIAL_URDF_PATH = (
    "openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf"
)
OFFICIAL_URDF_SHA256 = (
    "63caf1116a844c7eca25e91f0acf025c2798cc4795cf727b23433eacde80ca12"
)
OFFICIAL_URDF_CANONICAL_XML_SHA256 = (
    "66a9ce4ca5153624d3a785c2d5d4d4efd0361db4abf6f82c64fe216a62d98909"
)
DEPLOYED_URDF_EVIDENCE_SHA256 = (
    "092ac325b834e69b3cca703c9711d0166f6c8cad54356926968bca562e13f0d5"
)
OPENARM_PACKAGE = "openarm_description"
OPENARM_VARIANT = "openarm_v1_dual_arm"
ANIMATED_CHAIN = "left_follower"

ISAAC_VERSION = "5.1.0.0"
NVIDIA_MEDICAL_HUMAN_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/"
    "5.1/Isaac/People/Characters/original_male_adult_medical_01/"
    "male_adult_medical_01.usd"
)
HUMAN_USD_PATH = "human/male_adult_medical_01.usd"
HUMAN_USD_SHA256 = (
    "2c65e15587a311c5d14cdbd6a1408e0c3d8f3bec69f483150f04ca4c8f4e3ccc"
)
HUMAN_USD_SIZE = 598342
NVIDIA_TEXTURE_SHA256 = {
    "human/textures/male_adult_medical_01_BaseColor.png": (
        "ee433d8a6ef8de8bb172e3b733b4a9cc3e226da5ce2884fa06be6ae37b15f585"
    ),
    "human/textures/male_adult_medical_01_Normal.png": (
        "f76f5442a1a8e96f6547c358245589212cfdf257c8b232ee263dfb432e0e65bd"
    ),
    "human/textures/male_adult_medical_01_ORM.png": (
        "96e15659f384477922ddd6e92489515605d12c508de18645bb412e9f13fbb0b0"
    ),
}
NVIDIA_TEXTURE_SIZE = {
    "human/textures/male_adult_medical_01_BaseColor.png": 5667369,
    "human/textures/male_adult_medical_01_Normal.png": 13786667,
    "human/textures/male_adult_medical_01_ORM.png": 12140285,
}
NVIDIA_TEXTURE_AUTHORED_URI = {
    "human/textures/male_adult_medical_01_BaseColor.png": (
        "./textures/male_adult_medical_01_BaseColor.png"
    ),
    "human/textures/male_adult_medical_01_Normal.png": (
        "./textures/male_adult_medical_01_Normal.png"
    ),
    "human/textures/male_adult_medical_01_ORM.png": (
        "./textures/male_adult_medical_01_ORM.png"
    ),
}
OMNIPBR_AUTHORED_URI = "OmniPBR.mdl"
OMNIPBR_RUNTIME_RELATIVE_PATH = "isaacsim/kit/mdl/core/Base/OmniPBR.mdl"
OMNIPBR_SHA256 = "685a82c18b381f1cd29831490e4c356e61dbeac2d86803babc115cd6a4a845db"
OMNIPBR_SIZE = 15591
OMNIPBR_RUNTIME_PROVENANCE = {
    "distribution": "isaacsim-kernel",
    "record_hash": "sha256=aFqCwYs4HxzSmDFJDkw1bmHb6sLYaAO6vBFc1qSoRds",
    "record_size": 15591,
    "version": "5.1.0.0",
}
DEPENDENCY_REPORT_PATH = "human/compute_all_dependencies.json"
DEPENDENCY_REPORT_SCHEMA = "isaac_compute_all_dependencies_v2"
LICENSE_PAGE_PATH = "human/licenses/isaac_sim_additional_license.html"
LICENSE_EVIDENCE_PATH = "human/licenses/license_evidence.json"
LICENSE_EVIDENCE_SCHEMA = "nvidia_isaac_additional_license_evidence_v1"
NVIDIA_ADDITIONAL_LICENSE_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/latest/common/"
    "license-isaac-sim-additional.html"
)
NVIDIA_LICENSE_PAGE_SHA256 = (
    "addbfb2377a3984c7c4d530df41da4e7166f59ea485ef5e59bda0da268db02ec"
)
NVIDIA_LICENSE_PAGE_SIZE = 116699

CAMERA_SERIAL = "CL8326300A0"
INTRINSICS_FILE_SHA256 = (
    "b05ce66978015518bcbe4dc39e09a5dbef6606153ccc04eab0eac0fc4f3875b3"
)
UNCOMMISSIONED_POSE_STATUS = "installation_pose_uncommissioned"
GIT_COMMAND_TIMEOUT_SECONDS = 10.0
COMMISSIONED_POSE_STATUS = "installation_pose_commissioned"
CAMERA_CONTRACT = {
    "camera_to_base_covariance": "unavailable",
    "depth_bias": "unavailable",
    "distortion": "unavailable",
    "installation_specific_pass": False,
    "negotiated_modes": {
        "align_mode": "software_depth_to_color",
        "depth": {"format": "Y16", "fps": 30.0, "height": 576, "width": 640},
        "frame_aggregate_mode": "full_frame_require",
        "rgb": {"format": "MJPG", "fps": 30.0, "height": 1080, "width": 1920},
    },
    "pose": {"pose_status": UNCOMMISSIONED_POSE_STATUS},
    "rgb_intrinsics": {
        "cx": 955.7200927734375,
        "cy": 533.1757202148438,
        "fx": 1123.173095703125,
        "fy": 1122.9962158203125,
        "height": 1080,
        "intrinsics_file_sha256": INTRINSICS_FILE_SHA256,
        "width": 1920,
    },
    "serial": CAMERA_SERIAL,
}

_TEXTURE_PATHS = tuple(NVIDIA_TEXTURE_SHA256)
_ROBOT_STRUCTURE = {
    "joint_count": 23,
    "link_count": 24,
    "mesh_reference_count": 42,
    "revolute_joint_count": 14,
    "unique_mesh_count": 20,
}
_LEFT_JOINTS = tuple(f"openarm_left_joint{index}" for index in range(1, 8))
_RIGHT_JOINTS = tuple(f"openarm_right_joint{index}" for index in range(1, 8))

_TOP_KEYS = {
    "camera", "clean_source", "dependencies", "files", "human_source",
    "isaac_version", "primitive_fallback", "robot_source", "root_scale", "schema",
}
_ROBOT_SOURCE_KEYS = {
    "animated_chain", "canonical_xml_sha256", "checkout_path", "commit",
    "deployed_evidence_urdf_sha256", "git_evidence", "package", "repository_url",
    "structure", "urdf_path", "urdf_sha256", "variant",
}
_HUMAN_SOURCE_KEYS = {
    "asset_url", "dependency_report_path", "license_evidence_path", "license_page_path",
    "usd_path", "usd_sha256",
}
_GIT_EVIDENCE_KEYS = {
    "detached_head", "head", "remote_url", "status_porcelain", "tracked_paths",
}
_DEPENDENCY_KEYS = {"human_usd", "robot_mesh"}
_CAMERA_KEYS = {
    "camera_to_base_covariance", "depth_bias", "distortion",
    "installation_specific_pass", "negotiated_modes", "pose", "rgb_intrinsics", "serial",
}
_MODE_KEYS = {"align_mode", "depth", "frame_aggregate_mode", "rgb"}
_STREAM_MODE_KEYS = {"format", "fps", "height", "width"}
_INTRINSICS_KEYS = {
    "cx", "cy", "fx", "fy", "height", "intrinsics_file_sha256", "width",
}
_FILE_BASE_KEYS = {"path", "role", "sha256", "size"}
_ROLES = {
    "dependency_report", "human_usd", "human_usd_dependency", "license_evidence",
    "license_page", "robot_mesh", "robot_urdf",
}
_REPORT_KEYS = {
    "asset_count", "asset_url", "assets", "isaac_version", "layer_count", "resolver",
    "root_layer", "schema", "unresolved", "unresolved_count",
}
_LICENSE_KEYS = {
    "additional_license_url", "asset_url", "page_path", "page_sha256", "page_size",
    "retrieved_at", "schema",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID_RE = re.compile(r"[0-9a-f]{40}\Z")
_RFC3339_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_PRIMITIVE_TAGS = {"box", "capsule", "cylinder", "plane", "sphere"}


class AssetContractError(ValueError):
    """The asset bundle or manifest failed a fail-closed contract check."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AssetContractError("value cannot be encoded as canonical JSON") from exc


def _require_keys(value: object, expected: set, label: str) -> Dict[str, object]:
    if type(value) is not dict:
        raise AssetContractError(f"{label} must be an object with exact keys")
    if set(value) != expected:
        raise AssetContractError(
            f"{label} has wrong keys: expected {sorted(expected)!r}, got {sorted(value)!r}"
        )
    return value


def _safe_relative(value: object, label: str = "path") -> str:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise AssetContractError(f"{label} must be a safe relative POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or str(relative) != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise AssetContractError(f"{label} must be a safe relative POSIX path")
    return value


def _asset_root(value: object) -> Tuple[Path, Path]:
    try:
        root = Path(os.fspath(value))
    except TypeError as exc:
        raise AssetContractError("asset root must be path-like") from exc
    if root.is_symlink():
        raise AssetContractError("asset root must not be a symlink")
    if not root.is_dir():
        raise AssetContractError("asset root must be an existing directory")
    try:
        return root, root.resolve(strict=True)
    except OSError as exc:
        raise AssetContractError("asset root cannot be resolved") from exc


def _safe_asset_path(
    root: Path, resolved_root: Path, relative: str, *, directory: bool
) -> Path:
    relative = _safe_relative(relative)
    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise AssetContractError(f"asset path contains a symlink: {relative}")
    try:
        candidate.resolve(strict=True).relative_to(resolved_root)
        mode = candidate.lstat().st_mode
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise AssetContractError(f"asset path is missing or escapes asset root: {relative}") from exc
    expected = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not expected:
        kind = "directory" if directory else "regular file"
        raise AssetContractError(f"asset path is not a {kind}: {relative}")
    return candidate


def _safe_asset_file(root: Path, resolved_root: Path, relative: str) -> Path:
    return _safe_asset_path(root, resolved_root, relative, directory=False)


def _sha256_and_size(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if size <= 0:
        raise AssetContractError(f"asset file must have positive size: {path}")
    return digest.hexdigest(), size


def _validate_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise AssetContractError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _decode_canonical(raw: object, label: str) -> Dict[str, object]:
    if type(raw) is not bytes:
        raise AssetContractError(f"{label} must be supplied as canonical JSON bytes")

    def reject_constant(value: str) -> object:
        raise AssetContractError(f"{label} contains non-finite JSON: {value}")

    def reject_duplicates(pairs: List[Tuple[str, object]]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AssetContractError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except AssetContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssetContractError(f"{label} is not valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise AssetContractError(f"{label} JSON must be an object")
    if canonical_json_bytes(value) != raw:
        raise AssetContractError(f"{label} JSON is not canonical")
    return value


def _validated_git_layout(checkout: Path) -> Tuple[Path, Path]:
    """Require a standalone Git directory whose metadata and objects stay in the bundle."""

    if checkout.is_symlink() or not checkout.is_dir():
        raise AssetContractError("official git checkout must be a real directory")
    try:
        resolved_checkout = checkout.resolve(strict=True)
        git_dir_entry = resolved_checkout / ".git"
        if git_dir_entry.is_symlink() or not git_dir_entry.is_dir():
            raise AssetContractError("official checkout .git must be a real directory")
        git_dir = git_dir_entry.resolve(strict=True)
        git_dir.relative_to(resolved_checkout)
    except AssetContractError:
        raise
    except (OSError, ValueError) as exc:
        raise AssetContractError("official checkout git directory is unavailable") from exc

    try:
        pending = [git_dir]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        relative = Path(entry.path).relative_to(resolved_checkout)
                        raise AssetContractError(
                            f"git metadata contains a symlink: {relative.as_posix()}"
                        )
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
        for name in ("HEAD", "config", "index"):
            if not stat.S_ISREG((git_dir / name).lstat().st_mode):
                raise AssetContractError(f"git metadata {name} must be a regular file")
        objects = git_dir / "objects"
        if not stat.S_ISDIR(objects.lstat().st_mode):
            raise AssetContractError("git object storage must be a real directory")
    except AssetContractError:
        raise
    except OSError as exc:
        raise AssetContractError("git metadata cannot be fully inspected") from exc

    forbidden_metadata = {
        "commondir": "git metadata commondir is forbidden",
        "objects/info/alternates": "git object storage alternates are forbidden",
        "objects/info/http-alternates": "git object storage HTTP alternates are forbidden",
        "worktrees": "linked-worktree git metadata is forbidden",
    }
    for relative, message in forbidden_metadata.items():
        if os.path.lexists(os.fspath(git_dir / relative)):
            raise AssetContractError(message)
    return resolved_checkout, git_dir


def _git_invocation(
    checkout: Path,
    args: Sequence[str],
    *,
    layout: Tuple[Path, Path] | None = None,
) -> Tuple[Path, List[str]]:
    """Bind a read-only Git command to validated in-bundle metadata and work tree."""

    resolved_checkout, git_dir = layout or _validated_git_layout(checkout)

    command = [
        "git",
        "--no-pager",
        f"--git-dir={git_dir}",
        f"--work-tree={resolved_checkout}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.pager=cat",
        "-c",
        "pager.status=false",
        "-c",
        "diff.external=",
        *args,
    ]
    return resolved_checkout, command


def _invoke_git(
    checkout: Path,
    args: Sequence[str],
    *,
    text: bool,
    layout: Tuple[Path, Path] | None = None,
):
    resolved_checkout, command = _git_invocation(checkout, args, layout=layout)
    # Start from a minimal environment so caller-controlled Git config, object-store,
    # executable-path, pager, diff, and dynamic-loader variables cannot redirect a probe.
    environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PAGER": "cat",
        "PATH": os.environ.get("PATH", os.defpath),
    }
    try:
        return subprocess.run(
            command,
            cwd=str(resolved_checkout),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            check=False,
            timeout=GIT_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssetContractError(
            f"git checkout probe timed out after {GIT_COMMAND_TIMEOUT_SECONDS:g} seconds"
        ) from exc
    except OSError as exc:
        raise AssetContractError("local git executable is unavailable") from exc


def _run_git(
    checkout: Path,
    args: Sequence[str],
    allowed: Sequence[int] = (0,),
    *,
    layout: Tuple[Path, Path] | None = None,
):
    result = _invoke_git(checkout, args, text=True, layout=layout)
    if result.returncode not in allowed:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise AssetContractError(f"git checkout probe failed: {detail}")
    return result


def _run_git_bytes(
    checkout: Path,
    args: Sequence[str],
    *,
    layout: Tuple[Path, Path] | None = None,
):
    result = _invoke_git(checkout, args, text=False, layout=layout)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AssetContractError(
            f"git checkout probe failed: {detail or 'git command failed'}"
        )
    return result


def _git_path(raw: bytes, label: str) -> str:
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssetContractError(f"{label} is not a UTF-8 path") from exc
    return _safe_relative(value, label)


def _git_tree_entries(
    checkout: Path, head: str, layout: Tuple[Path, Path]
) -> Dict[str, Tuple[str, str]]:
    raw = _run_git_bytes(
        checkout,
        ("ls-tree", "-r", "-z", "--full-tree", head),
        layout=layout,
    ).stdout
    entries: Dict[str, Tuple[str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split(b" ")
        if separator != b"\t" or len(fields) != 3:
            raise AssetContractError("git tree contains a malformed entry")
        try:
            mode, object_type, object_id = (
                field.decode("ascii") for field in fields
            )
        except UnicodeDecodeError as exc:
            raise AssetContractError("git tree metadata is not ASCII") from exc
        if object_type != "blob" or mode not in {"100644", "100755", "120000"}:
            raise AssetContractError("git tree contains unsupported non-blob content")
        if _GIT_OID_RE.fullmatch(object_id) is None:
            raise AssetContractError("git tree contains an invalid object id")
        path = _git_path(raw_path, "git tree path")
        if path in entries:
            raise AssetContractError("git tree contains a duplicate path")
        entries[path] = (mode, object_id)
    if not entries:
        raise AssetContractError("git tree must contain tracked files")
    return entries


def _git_index_entries(
    checkout: Path, layout: Tuple[Path, Path]
) -> Dict[str, Tuple[str, str]]:
    raw = _run_git_bytes(
        checkout,
        ("ls-files", "--cached", "--stage", "-z", "--full-name"),
        layout=layout,
    ).stdout
    entries: Dict[str, Tuple[str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split(b" ")
        if separator != b"\t" or len(fields) != 3:
            raise AssetContractError("git index contains a malformed entry")
        try:
            mode, object_id, stage = (field.decode("ascii") for field in fields)
        except UnicodeDecodeError as exc:
            raise AssetContractError("git index metadata is not ASCII") from exc
        if stage != "0" or mode not in {"100644", "100755", "120000"}:
            raise AssetContractError("git index contains unmerged or unsupported content")
        if _GIT_OID_RE.fullmatch(object_id) is None:
            raise AssetContractError("git index contains an invalid object id")
        path = _git_path(raw_path, "git index path")
        if path in entries:
            raise AssetContractError("git index contains a duplicate path")
        entries[path] = (mode, object_id)
    return entries


def _validate_git_index_flags(
    checkout: Path,
    expected_paths: Sequence[str],
    layout: Tuple[Path, Path],
) -> None:
    raw = _run_git_bytes(
        checkout,
        ("ls-files", "--cached", "-v", "-z", "--full-name"),
        layout=layout,
    ).stdout
    flagged: Dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        tag, separator, raw_path = record.partition(b" ")
        if separator != b" " or len(tag) != 1:
            raise AssetContractError("git index flag output is malformed")
        try:
            decoded_tag = tag.decode("ascii")
        except UnicodeDecodeError as exc:
            raise AssetContractError("git index flag is not ASCII") from exc
        path = _git_path(raw_path, "git index flag path")
        if path in flagged:
            raise AssetContractError("git index flags contain a duplicate path")
        flagged[path] = decoded_tag
    if set(flagged) != set(expected_paths) or any(
        tag != "H" for tag in flagged.values()
    ):
        raise AssetContractError(
            "git index flag rejects assume-unchanged/skip-worktree content"
        )


def _worktree_non_directories(checkout: Path) -> set[str]:
    paths: set[str] = set()
    pending: List[Tuple[Path, str]] = [(checkout, "")]
    try:
        while pending:
            directory, prefix = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not prefix and entry.name == ".git":
                        continue
                    relative = f"{prefix}/{entry.name}" if prefix else entry.name
                    relative = _safe_relative(relative, "working-tree path")
                    if entry.is_symlink():
                        paths.add(relative)
                    elif entry.is_dir(follow_symlinks=False):
                        pending.append((Path(entry.path), relative))
                    else:
                        paths.add(relative)
    except AssetContractError:
        raise
    except OSError as exc:
        raise AssetContractError("working tree cannot be fully inspected") from exc
    return paths


def _working_tree_bytes(checkout: Path, relative: str, expected_mode: str) -> bytes:
    candidate = checkout
    parts = PurePosixPath(relative).parts
    try:
        for part in parts[:-1]:
            candidate = candidate / part
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise AssetContractError(
                    f"tracked working-tree parent is not a real directory: {relative}"
                )
        candidate = candidate / parts[-1]
        mode = candidate.lstat().st_mode
        if expected_mode == "120000":
            if not stat.S_ISLNK(mode):
                raise AssetContractError(
                    f"tracked working-tree path is not the committed symlink: {relative}"
                )
            return os.fsencode(os.readlink(candidate))
        if not stat.S_ISREG(mode):
            raise AssetContractError(
                f"tracked working-tree path is not a regular file: {relative}"
            )
        expected_executable = expected_mode == "100755"
        if bool(mode & 0o111) != expected_executable:
            raise AssetContractError(
                f"working-tree executable mode differs at HEAD: {relative}"
            )
        return candidate.read_bytes()
    except AssetContractError:
        raise
    except OSError as exc:
        raise AssetContractError(
            f"tracked working-tree file is unavailable: {relative}"
        ) from exc


def _git_blob(
    checkout: Path,
    object_id: str,
    relative: str,
    layout: Tuple[Path, Path],
) -> bytes:
    if _GIT_OID_RE.fullmatch(object_id) is None:
        raise AssetContractError(f"committed blob id is invalid for {relative}")
    result = _run_git_bytes(
        checkout, ("cat-file", "blob", object_id), layout=layout
    )
    header = f"blob {len(result.stdout)}\0".encode("ascii")
    if hashlib.sha1(header + result.stdout).hexdigest() != object_id:
        raise AssetContractError(f"committed blob object is corrupt: {relative}")
    return result.stdout


def _probe_git_checkout(
    checkout: Path, required_paths: Sequence[str]
) -> Dict[str, object]:
    """Compare raw tree, index, object, and work-tree bytes without content filters."""

    layout = _validated_git_layout(checkout)
    checkout = layout[0]
    required = sorted(
        {_safe_relative(path, "tracked git path") for path in required_paths}
    )
    if not required or len(required) != len(required_paths):
        raise AssetContractError("tracked git paths must be nonempty and unique")

    top = Path(
        _run_git(
            checkout, ("rev-parse", "--show-toplevel"), layout=layout
        ).stdout.strip()
    )
    try:
        top = top.resolve(strict=True)
    except OSError as exc:
        raise AssetContractError("git top-level path cannot be resolved") from exc
    if top != checkout:
        raise AssetContractError("official checkout is not the git top-level directory")
    remote = _run_git(
        checkout,
        ("config", "--local", "--no-includes", "--get", "remote.origin.url"),
        layout=layout,
    ).stdout.strip()
    if remote != ENACTIC_REPOSITORY_URL:
        raise AssetContractError("git remote URL is not the official Enactic repository")
    head = _run_git(checkout, ("rev-parse", "HEAD"), layout=layout).stdout.strip()
    if head != ENACTIC_COMMIT:
        raise AssetContractError("git HEAD is not the pinned Enactic commit")
    symbolic = _run_git(
        checkout,
        ("symbolic-ref", "-q", "HEAD"),
        allowed=(0, 1),
        layout=layout,
    )
    if symbolic.returncode != 1 or symbolic.stdout.strip():
        raise AssetContractError("git HEAD must be detached")

    tree_entries = _git_tree_entries(checkout, head, layout)
    index_entries = _git_index_entries(checkout, layout)
    if index_entries != tree_entries:
        raise AssetContractError("git index differs from the pinned HEAD tree")
    _validate_git_index_flags(checkout, sorted(tree_entries), layout)

    actual_paths = _worktree_non_directories(checkout)
    untracked = sorted(actual_paths - set(tree_entries))
    if untracked:
        raise AssetContractError(
            f"git checkout contains untracked files: {untracked[0]}"
        )

    for relative, (mode, object_id) in tree_entries.items():
        working_bytes = _working_tree_bytes(checkout, relative, mode)
        committed_bytes = _git_blob(checkout, object_id, relative, layout)
        if (
            working_bytes != committed_bytes
            or hashlib.sha256(working_bytes).digest()
            != hashlib.sha256(committed_bytes).digest()
        ):
            raise AssetContractError(
                f"working-tree file differs from committed blob at HEAD: {relative}"
            )

    tracked: List[str] = []
    for relative in required:
        entry = tree_entries.get(relative)
        if entry is None:
            raise AssetContractError(f"required URDF/mesh is not tracked by git: {relative}")
        if entry[0] not in {"100644", "100755"}:
            raise AssetContractError(
                f"required URDF/mesh is not a regular committed blob: {relative}"
            )
        tracked.append(relative)
    return {
        "detached_head": True,
        "head": head,
        "remote_url": remote,
        "status_porcelain": "",
        "tracked_paths": tracked,
    }


def _validate_git_evidence(value: object, required_paths: Sequence[str]) -> Dict[str, object]:
    evidence = _require_keys(value, _GIT_EVIDENCE_KEYS, "git evidence")
    expected = {
        "detached_head": True,
        "head": ENACTIC_COMMIT,
        "remote_url": ENACTIC_REPOSITORY_URL,
        "status_porcelain": "",
        "tracked_paths": sorted(required_paths),
    }
    if canonical_json_bytes(evidence) != canonical_json_bytes(expected):
        raise AssetContractError("git evidence is not clean, detached, pinned, and fully tracked")
    return evidence


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _scale_vector(value: object, label: str) -> Tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise AssetContractError(f"{label} must be an exact three-number scale vector")
    output: List[float] = []
    for component in value:
        if type(component) not in {int, float}:
            raise AssetContractError(f"{label} must be an exact three-number scale vector")
        number = float(component)
        if not math.isfinite(number) or number == 0.0:
            raise AssetContractError(f"{label} has an invalid scale component")
        output.append(number)
    return output[0], output[1], output[2]


def _validate_official_urdf(data: bytes) -> Dict[str, object]:
    raw_sha = hashlib.sha256(data).hexdigest()
    if raw_sha != OFFICIAL_URDF_SHA256:
        raise AssetContractError("official URDF sha256 does not match clean v1 example bytes")
    try:
        document = ET.fromstring(data)
    except ET.ParseError as exc:
        raise AssetContractError("official URDF is not well-formed XML") from exc
    canonical_sha = hashlib.sha256(ET.tostring(document, encoding="utf-8")).hexdigest()
    if canonical_sha != OFFICIAL_URDF_CANONICAL_XML_SHA256:
        raise AssetContractError("official URDF canonical XML sha256 does not match")
    if _local_name(document.tag) != "robot" or document.get("name") != "openarm":
        raise AssetContractError("official URDF robot name must be exactly openarm")

    children = list(document)
    links = [item for item in children if _local_name(item.tag) == "link"]
    joints = [item for item in children if _local_name(item.tag) == "joint"]
    revolute = [item for item in joints if item.get("type") == "revolute"]
    if len(links) != 24 or len(joints) != 23 or len(revolute) != 14:
        raise AssetContractError(
            "official URDF must contain exactly 24 links, 23 joints, and 14 revolute joints"
        )
    link_names = [item.get("name") for item in links]
    joint_names = [item.get("name") for item in joints]
    if None in link_names or len(set(link_names)) != 24:
        raise AssetContractError("official URDF must have 24 uniquely named links")
    if None in joint_names or len(set(joint_names)) != 23:
        raise AssetContractError("official URDF must have 23 uniquely named joints")
    revolute_by_name = {item.get("name"): item for item in revolute}
    required_revolute = set(_LEFT_JOINTS + _RIGHT_JOINTS)
    if set(revolute_by_name) != required_revolute:
        raise AssetContractError("official URDF must contain complete left/right seven-joint chains")

    for element in document.iter():
        if _local_name(element.tag) in _PRIMITIVE_TAGS:
            raise AssetContractError("official URDF contains forbidden primitive fallback geometry")
    prefix = f"package://{OPENARM_PACKAGE}/"
    references: List[Tuple[str, Tuple[float, float, float]]] = []
    for mesh in document.iter():
        if _local_name(mesh.tag) != "mesh":
            continue
        uri = mesh.get("filename")
        if not isinstance(uri, str) or not uri.startswith(prefix):
            raise AssetContractError("robot mesh is outside the official OpenArm package")
        relative = _safe_relative(
            f"{CHECKOUT_PATH}/{uri[len(prefix):]}", "robot mesh path"
        )
        scale_text = mesh.get("scale")
        if not isinstance(scale_text, str):
            raise AssetContractError("robot mesh is missing its explicit pinned scale vector")
        try:
            parts = [float(part) for part in scale_text.split()]
        except ValueError as exc:
            raise AssetContractError("robot mesh has an invalid pinned scale vector") from exc
        scale = _scale_vector(parts, "robot mesh scale")
        references.append((relative, scale))
    unique_paths = sorted({path for path, _scale in references})
    if len(references) != 42 or len(unique_paths) != 20:
        raise AssetContractError("official URDF must contain 42 mesh refs and 20 unique meshes")
    return {
        "canonical_sha256": canonical_sha,
        "references": references,
        "unique_paths": unique_paths,
    }


def _validate_dependency_report(data: bytes) -> Dict[str, object]:
    report = _decode_canonical(data, "dependency report")
    if report.get("schema") != DEPENDENCY_REPORT_SCHEMA:
        raise AssetContractError(
            "dependency report schema v2 must separate the root layer from OmniPBR assets"
        )
    _require_keys(report, _REPORT_KEYS, "dependency report")
    if report["unresolved_count"] != 0 or report["unresolved"] != []:
        raise AssetContractError("dependency report must have unresolved=0")
    assets = report["assets"]
    if (
        report["layer_count"] != 1
        or report["asset_count"] != 4
        or type(assets) is not list
        or len(assets) != 4
    ):
        raise AssetContractError(
            "dependency report must have one root layer and assets=4 including OmniPBR"
        )
    expected_root = {
        "resolved_path": HUMAN_USD_PATH,
        "sha256": HUMAN_USD_SHA256,
        "size_bytes": HUMAN_USD_SIZE,
    }
    if canonical_json_bytes(report["root_layer"]) != canonical_json_bytes(expected_root):
        raise AssetContractError("dependency report root layer record does not match official USD")
    expected_omnipbr = {
        "authored_uri": OMNIPBR_AUTHORED_URI,
        "resolved_path": "",
        "resolution_kind": "kit_mdl_search_path",
        "runtime_provenance": dict(OMNIPBR_RUNTIME_PROVENANCE),
        "runtime_relative_path": OMNIPBR_RUNTIME_RELATIVE_PATH,
        "sha256": OMNIPBR_SHA256,
        "size_bytes": OMNIPBR_SIZE,
    }
    if canonical_json_bytes(assets[0]) != canonical_json_bytes(expected_omnipbr):
        raise AssetContractError("OmniPBR asset record/runtime provenance does not match Isaac")
    expected_textures = [
        {
            "authored_uri": NVIDIA_TEXTURE_AUTHORED_URI[path],
            "resolution_kind": "usd_relative",
            "resolved_path": path,
            "sha256": NVIDIA_TEXTURE_SHA256[path],
            "size_bytes": NVIDIA_TEXTURE_SIZE[path],
        }
        for path in _TEXTURE_PATHS
    ]
    if canonical_json_bytes(assets[1:]) != canonical_json_bytes(expected_textures):
        raise AssetContractError("dependency report texture asset records are not the exact set")
    expected = {
        "asset_count": 4,
        "asset_url": NVIDIA_MEDICAL_HUMAN_URL,
        "assets": [expected_omnipbr, *expected_textures],
        "isaac_version": ISAAC_VERSION,
        "layer_count": 1,
        "resolver": "UsdUtils.ComputeAllDependencies",
        "root_layer": expected_root,
        "schema": DEPENDENCY_REPORT_SCHEMA,
        "unresolved": [],
        "unresolved_count": 0,
    }
    if canonical_json_bytes(report) != canonical_json_bytes(expected):
        raise AssetContractError("dependency report does not match the pinned Isaac preflight")
    return report


def _validate_license_evidence(
    data: bytes, root: Path, resolved_root: Path
) -> Dict[str, object]:
    evidence = _decode_canonical(data, "license evidence")
    _require_keys(evidence, _LICENSE_KEYS, "license evidence")
    if evidence["additional_license_url"] != NVIDIA_ADDITIONAL_LICENSE_URL:
        raise AssetContractError("license evidence has the wrong official license URL")
    if evidence["asset_url"] != NVIDIA_MEDICAL_HUMAN_URL:
        raise AssetContractError("license evidence has the wrong NVIDIA asset URL")
    if evidence["schema"] != LICENSE_EVIDENCE_SCHEMA:
        raise AssetContractError("license evidence schema is not supported")
    if evidence["page_path"] != LICENSE_PAGE_PATH:
        raise AssetContractError("license evidence page_path is not pinned")
    retrieved = evidence["retrieved_at"]
    if type(retrieved) is not str or _RFC3339_UTC_RE.fullmatch(retrieved) is None:
        raise AssetContractError("license evidence retrieved_at must be RFC3339 UTC")
    try:
        datetime.strptime(retrieved, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise AssetContractError("license evidence retrieved_at is invalid") from exc
    page = _safe_asset_file(root, resolved_root, LICENSE_PAGE_PATH)
    page_sha, page_size = _sha256_and_size(page)
    if page_sha != NVIDIA_LICENSE_PAGE_SHA256 or page_size != NVIDIA_LICENSE_PAGE_SIZE:
        raise AssetContractError("license snapshot sha256/size does not match official bytes")
    if evidence["page_sha256"] != page_sha or evidence["page_size"] != page_size:
        raise AssetContractError("license evidence page hash/size does not match downloaded bytes")
    return evidence


def _file_entry(
    root: Path,
    resolved_root: Path,
    relative: str,
    role: str,
    cache: Dict[str, Tuple[str, int]],
    scale: Tuple[float, float, float] = None,
) -> Dict[str, object]:
    path = _safe_asset_file(root, resolved_root, relative)
    if relative not in cache:
        cache[relative] = _sha256_and_size(path)
    digest, size = cache[relative]
    result: Dict[str, object] = {
        "path": relative, "role": role, "sha256": digest, "size": size,
    }
    if scale is not None:
        result["scale"] = list(scale)
    return result


def _entry_sort_key(entry: Mapping[str, object]) -> Tuple[str, str, bytes]:
    return (
        str(entry.get("path")),
        str(entry.get("role")),
        canonical_json_bytes(entry.get("scale")),
    )


def _checkout_relative(asset_path: str) -> str:
    prefix = f"{CHECKOUT_PATH}/"
    if not asset_path.startswith(prefix):
        raise AssetContractError("tracked robot path is outside official checkout")
    return _safe_relative(asset_path[len(prefix):], "tracked git path")


def _validate_camera(value: object) -> None:
    camera = _require_keys(value, _CAMERA_KEYS, "camera")
    modes = _require_keys(camera["negotiated_modes"], _MODE_KEYS, "camera.negotiated_modes")
    _require_keys(modes["rgb"], _STREAM_MODE_KEYS, "camera.negotiated_modes.rgb")
    _require_keys(modes["depth"], _STREAM_MODE_KEYS, "camera.negotiated_modes.depth")
    _require_keys(camera["rgb_intrinsics"], _INTRINSICS_KEYS, "camera.rgb_intrinsics")
    pose = camera["pose"]
    if type(pose) is not dict:
        raise AssetContractError("camera.pose must be an object with exact keys")
    if pose.get("pose_status") == COMMISSIONED_POSE_STATUS:
        if "transform" not in pose or "provenance" not in pose:
            raise AssetContractError(
                "commissioned camera pose requires both transform and provenance evidence"
            )
        raise AssetContractError("commissioned camera pose is outside this uncommissioned contract")
    _require_keys(pose, {"pose_status"}, "camera.pose")
    if canonical_json_bytes(camera) != canonical_json_bytes(CAMERA_CONTRACT):
        raise AssetContractError("camera contract does not match frozen evidence/limitations")


def _validate_fixed_human_files(root: Path, resolved_root: Path) -> None:
    usd = _safe_asset_file(root, resolved_root, HUMAN_USD_PATH)
    usd_sha, usd_size = _sha256_and_size(usd)
    if usd_sha != HUMAN_USD_SHA256 or usd_size != HUMAN_USD_SIZE:
        raise AssetContractError("official human USD sha256/size does not match")
    if tuple(NVIDIA_TEXTURE_SHA256) != _TEXTURE_PATHS:
        raise AssetContractError("official texture path set was modified")
    if set(NVIDIA_TEXTURE_SIZE) != set(_TEXTURE_PATHS):
        raise AssetContractError("official texture size path set was modified")
    if set(NVIDIA_TEXTURE_AUTHORED_URI) != set(_TEXTURE_PATHS):
        raise AssetContractError("official texture authored-URI path set was modified")
    for path, expected_sha in NVIDIA_TEXTURE_SHA256.items():
        texture = _safe_asset_file(root, resolved_root, path)
        actual_sha, actual_size = _sha256_and_size(texture)
        if actual_sha != expected_sha or actual_size != NVIDIA_TEXTURE_SIZE[path]:
            raise AssetContractError(f"official texture sha256/size does not match: {path}")


def build_manifest(asset_root: object) -> Dict[str, object]:
    """Build and self-verify the one allowed official asset manifest."""

    root, resolved_root = _asset_root(asset_root)
    checkout = _safe_asset_path(
        root, resolved_root, CHECKOUT_PATH, directory=True
    )
    urdf_file = _safe_asset_file(root, resolved_root, OFFICIAL_URDF_PATH)
    urdf = _validate_official_urdf(urdf_file.read_bytes())
    tracked_paths = sorted(
        _checkout_relative(path)
        for path in [OFFICIAL_URDF_PATH, *urdf["unique_paths"]]
    )
    git_evidence = _probe_git_checkout(checkout, tracked_paths)
    _validate_git_evidence(git_evidence, tracked_paths)

    _validate_fixed_human_files(root, resolved_root)
    report_file = _safe_asset_file(root, resolved_root, DEPENDENCY_REPORT_PATH)
    _validate_dependency_report(report_file.read_bytes())
    license_file = _safe_asset_file(root, resolved_root, LICENSE_EVIDENCE_PATH)
    _validate_license_evidence(license_file.read_bytes(), root, resolved_root)

    cache: Dict[str, Tuple[str, int]] = {}
    files = [_file_entry(root, resolved_root, OFFICIAL_URDF_PATH, "robot_urdf", cache)]
    files.extend(
        _file_entry(root, resolved_root, path, "robot_mesh", cache, scale)
        for path, scale in sorted(set(urdf["references"]))
    )
    files.append(_file_entry(root, resolved_root, HUMAN_USD_PATH, "human_usd", cache))
    files.extend(
        _file_entry(root, resolved_root, path, "human_usd_dependency", cache)
        for path in _TEXTURE_PATHS
    )
    files.append(
        _file_entry(root, resolved_root, DEPENDENCY_REPORT_PATH, "dependency_report", cache)
    )
    files.append(_file_entry(root, resolved_root, LICENSE_PAGE_PATH, "license_page", cache))
    files.append(
        _file_entry(root, resolved_root, LICENSE_EVIDENCE_PATH, "license_evidence", cache)
    )
    files.sort(key=_entry_sort_key)

    manifest: Dict[str, object] = {
        "camera": copy.deepcopy(CAMERA_CONTRACT),
        "clean_source": True,
        "dependencies": {
            "human_usd": list(_TEXTURE_PATHS),
            "robot_mesh": list(urdf["unique_paths"]),
        },
        "files": files,
        "human_source": {
            "asset_url": NVIDIA_MEDICAL_HUMAN_URL,
            "dependency_report_path": DEPENDENCY_REPORT_PATH,
            "license_evidence_path": LICENSE_EVIDENCE_PATH,
            "license_page_path": LICENSE_PAGE_PATH,
            "usd_path": HUMAN_USD_PATH,
            "usd_sha256": HUMAN_USD_SHA256,
        },
        "isaac_version": ISAAC_VERSION,
        "primitive_fallback": False,
        "robot_source": {
            "animated_chain": ANIMATED_CHAIN,
            "canonical_xml_sha256": OFFICIAL_URDF_CANONICAL_XML_SHA256,
            "checkout_path": CHECKOUT_PATH,
            "commit": ENACTIC_COMMIT,
            "deployed_evidence_urdf_sha256": DEPLOYED_URDF_EVIDENCE_SHA256,
            "git_evidence": git_evidence,
            "package": OPENARM_PACKAGE,
            "repository_url": ENACTIC_REPOSITORY_URL,
            "structure": dict(_ROBOT_STRUCTURE),
            "urdf_path": OFFICIAL_URDF_PATH,
            "urdf_sha256": OFFICIAL_URDF_SHA256,
            "variant": OPENARM_VARIANT,
        },
        "root_scale": 1,
        "schema": SCHEMA,
    }
    verify_manifest(canonical_json_bytes(manifest), root)
    return manifest


def _validate_dependency_list(value: object, label: str) -> List[str]:
    if type(value) is not list or not value:
        raise AssetContractError(f"{label} must be nonempty")
    paths = [_safe_relative(item, label) for item in value]
    if paths != sorted(set(paths)):
        raise AssetContractError(f"{label} must be sorted and contain no duplicates")
    return paths


def verify_manifest(raw: bytes, asset_root: object) -> Dict[str, object]:
    """Verify canonical manifest bytes and re-probe all local evidence."""

    manifest = _decode_canonical(raw, "manifest")
    _require_keys(manifest, _TOP_KEYS, "manifest")
    if manifest["schema"] != SCHEMA:
        raise AssetContractError("manifest schema is not supported")
    if manifest["clean_source"] is not True:
        raise AssetContractError("clean_source must be exactly true")
    if type(manifest["root_scale"]) is not int or manifest["root_scale"] != 1:
        raise AssetContractError("root_scale must be exactly integer 1")
    if manifest["primitive_fallback"] is not False:
        raise AssetContractError("primitive fallback geometry is forbidden")
    if manifest["isaac_version"] != ISAAC_VERSION:
        raise AssetContractError(f"isaac_version must be exactly {ISAAC_VERSION}")

    robot = _require_keys(manifest["robot_source"], _ROBOT_SOURCE_KEYS, "robot_source")
    if robot["repository_url"] != ENACTIC_REPOSITORY_URL or robot["commit"] != ENACTIC_COMMIT:
        raise AssetContractError("robot source/commit is not the pinned official checkout")
    if robot["checkout_path"] != CHECKOUT_PATH or robot["urdf_path"] != OFFICIAL_URDF_PATH:
        raise AssetContractError("robot checkout/URDF path is not the official v1 example")
    if robot["urdf_sha256"] != OFFICIAL_URDF_SHA256:
        raise AssetContractError("official URDF sha256 claim does not match the fixed digest")
    if robot["canonical_xml_sha256"] != OFFICIAL_URDF_CANONICAL_XML_SHA256:
        raise AssetContractError("official URDF canonical XML sha256 claim does not match")
    if robot["deployed_evidence_urdf_sha256"] != DEPLOYED_URDF_EVIDENCE_SHA256:
        raise AssetContractError("deployed URDF evidence sha256 claim does not match")
    if robot["package"] != OPENARM_PACKAGE or robot["variant"] != OPENARM_VARIANT:
        raise AssetContractError("robot is not the official OpenArm v1 dual-arm variant")
    if robot["animated_chain"] != ANIMATED_CHAIN:
        raise AssetContractError("animated chain must be the left follower")
    if canonical_json_bytes(robot["structure"]) != canonical_json_bytes(_ROBOT_STRUCTURE):
        raise AssetContractError("robot structure claim does not match official OpenArm v1")

    human = _require_keys(manifest["human_source"], _HUMAN_SOURCE_KEYS, "human_source")
    expected_human = {
        "asset_url": NVIDIA_MEDICAL_HUMAN_URL,
        "dependency_report_path": DEPENDENCY_REPORT_PATH,
        "license_evidence_path": LICENSE_EVIDENCE_PATH,
        "license_page_path": LICENSE_PAGE_PATH,
        "usd_path": HUMAN_USD_PATH,
        "usd_sha256": HUMAN_USD_SHA256,
    }
    if human["usd_sha256"] != HUMAN_USD_SHA256:
        raise AssetContractError("official human USD sha256 claim does not match")
    if canonical_json_bytes(human) != canonical_json_bytes(expected_human):
        raise AssetContractError("human source does not match the fixed official NVIDIA asset")
    _validate_camera(manifest["camera"])

    dependencies = _require_keys(manifest["dependencies"], _DEPENDENCY_KEYS, "dependencies")
    robot_dependencies = _validate_dependency_list(
        dependencies["robot_mesh"], "dependencies.robot_mesh"
    )
    human_dependencies = _validate_dependency_list(
        dependencies["human_usd"], "dependencies.human_usd"
    )
    if human_dependencies != list(_TEXTURE_PATHS):
        raise AssetContractError("human USD dependency set is not exactly the three textures")

    files = manifest["files"]
    if type(files) is not list or not files:
        raise AssetContractError("files must be a nonempty list")
    root, resolved_root = _asset_root(asset_root)
    role_rows: Dict[str, List[Dict[str, object]]] = {role: [] for role in _ROLES}
    identities = set()
    role_by_path: Dict[str, str] = {}
    actual_cache: Dict[str, Tuple[str, int]] = {}
    validated_rows: List[Dict[str, object]] = []
    for index, raw_entry in enumerate(files):
        if type(raw_entry) is not dict:
            raise AssetContractError(f"files[{index}] must be an object with exact keys")
        role = raw_entry.get("role")
        if type(role) is not str:
            raise AssetContractError(f"files[{index}].role must be exactly a string")
        if role not in _ROLES:
            raise AssetContractError(f"files[{index}].role is not allowed")
        keys = _FILE_BASE_KEYS | ({"scale"} if role == "robot_mesh" else set())
        entry = _require_keys(raw_entry, keys, f"files[{index}]")
        path_value = _safe_relative(entry["path"], f"files[{index}].path")
        digest = _validate_sha256(entry["sha256"], f"files[{index}].sha256")
        size = entry["size"]
        if type(size) is not int or size <= 0:
            raise AssetContractError(f"files[{index}].size must be a positive integer")
        scale = _scale_vector(entry["scale"], f"files[{index}].scale") if role == "robot_mesh" else None
        identity = path_value, role, scale
        if identity in identities:
            raise AssetContractError("files contains a duplicate path/role/scale entry")
        identities.add(identity)
        previous_role = role_by_path.setdefault(path_value, role)
        if previous_role != role:
            raise AssetContractError("one asset path cannot have conflicting roles")
        candidate = _safe_asset_file(root, resolved_root, path_value)
        if path_value not in actual_cache:
            actual_cache[path_value] = _sha256_and_size(candidate)
        actual_sha, actual_size = actual_cache[path_value]
        if size != actual_size:
            raise AssetContractError(f"asset size mismatch for {path_value}")
        if digest != actual_sha:
            raise AssetContractError(f"asset sha256 mismatch for {path_value}")
        role_rows[role].append(entry)
        validated_rows.append(entry)
    if validated_rows != sorted(validated_rows, key=_entry_sort_key):
        raise AssetContractError("files entries must be in canonical path/role/scale order")

    expected_singletons = {
        "robot_urdf": OFFICIAL_URDF_PATH,
        "human_usd": HUMAN_USD_PATH,
        "dependency_report": DEPENDENCY_REPORT_PATH,
        "license_page": LICENSE_PAGE_PATH,
        "license_evidence": LICENSE_EVIDENCE_PATH,
    }
    for role, expected_path in expected_singletons.items():
        rows = role_rows[role]
        if len(rows) != 1 or rows[0]["path"] != expected_path:
            raise AssetContractError(f"files must contain exactly one pinned {role}")
    if role_rows["robot_urdf"][0]["sha256"] != OFFICIAL_URDF_SHA256:
        raise AssetContractError("official URDF sha256 file entry does not match")
    if role_rows["human_usd"][0]["sha256"] != HUMAN_USD_SHA256:
        raise AssetContractError("official human USD sha256 file entry does not match")
    texture_rows = {row["path"]: row for row in role_rows["human_usd_dependency"]}
    if set(texture_rows) != set(_TEXTURE_PATHS):
        raise AssetContractError("files must contain exactly the three official textures")
    for path, expected_sha in NVIDIA_TEXTURE_SHA256.items():
        if texture_rows[path]["sha256"] != expected_sha:
            raise AssetContractError(f"official texture sha256 file entry does not match: {path}")
    if not role_rows["robot_mesh"]:
        raise AssetContractError("files must contain nonempty robot_mesh dependencies")

    _validate_fixed_human_files(root, resolved_root)
    urdf_file = _safe_asset_file(root, resolved_root, OFFICIAL_URDF_PATH)
    urdf = _validate_official_urdf(urdf_file.read_bytes())
    expected_mesh_rows = sorted(set(urdf["references"]))
    manifest_mesh_rows = sorted(
        (row["path"], _scale_vector(row["scale"], "robot mesh scale"))
        for row in role_rows["robot_mesh"]
    )
    if manifest_mesh_rows != expected_mesh_rows:
        raise AssetContractError("robot mesh path/scale entries do not match pinned URDF scales")
    if robot_dependencies != list(urdf["unique_paths"]):
        raise AssetContractError("dependencies.robot_mesh does not match the official URDF closure")

    tracked_paths = sorted(
        _checkout_relative(path)
        for path in [OFFICIAL_URDF_PATH, *urdf["unique_paths"]]
    )
    manifest_git = _validate_git_evidence(robot["git_evidence"], tracked_paths)
    checkout = _safe_asset_path(root, resolved_root, CHECKOUT_PATH, directory=True)
    live_git = _probe_git_checkout(checkout, tracked_paths)
    _validate_git_evidence(live_git, tracked_paths)
    if canonical_json_bytes(manifest_git) != canonical_json_bytes(live_git):
        raise AssetContractError("manifest git evidence does not match live checkout probe")

    report = _safe_asset_file(root, resolved_root, DEPENDENCY_REPORT_PATH)
    _validate_dependency_report(report.read_bytes())
    license_evidence = _safe_asset_file(root, resolved_root, LICENSE_EVIDENCE_PATH)
    _validate_license_evidence(license_evidence.read_bytes(), root, resolved_root)
    return manifest


__all__ = [
    "ANIMATED_CHAIN", "AssetContractError", "CAMERA_CONTRACT", "CHECKOUT_PATH",
    "DEPENDENCY_REPORT_PATH", "DEPLOYED_URDF_EVIDENCE_SHA256", "ENACTIC_COMMIT",
    "ENACTIC_REPOSITORY_URL", "HUMAN_USD_PATH", "HUMAN_USD_SHA256", "HUMAN_USD_SIZE",
    "ISAAC_VERSION",
    "LICENSE_EVIDENCE_PATH", "LICENSE_PAGE_PATH", "NVIDIA_ADDITIONAL_LICENSE_URL",
    "NVIDIA_LICENSE_PAGE_SHA256", "NVIDIA_LICENSE_PAGE_SIZE",
    "NVIDIA_MEDICAL_HUMAN_URL", "NVIDIA_TEXTURE_AUTHORED_URI", "NVIDIA_TEXTURE_SHA256",
    "NVIDIA_TEXTURE_SIZE", "OMNIPBR_AUTHORED_URI", "OMNIPBR_RUNTIME_RELATIVE_PATH",
    "OMNIPBR_RUNTIME_PROVENANCE", "OMNIPBR_SHA256", "OMNIPBR_SIZE",
    "OFFICIAL_URDF_CANONICAL_XML_SHA256", "OFFICIAL_URDF_PATH", "OFFICIAL_URDF_SHA256",
    "SCHEMA", "build_manifest", "canonical_json_bytes", "verify_manifest",
]
