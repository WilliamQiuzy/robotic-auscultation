"""Fail-closed runtime roles and paths for perception jobs.

Mac is development-only.  H200 owns data generation, training, and offline
evaluation; linbox owns camera/robot calibration and real-time/HIL execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import pwd
from pathlib import PurePosixPath
import re
import socket
import sys
from typing import Mapping, MutableSequence, Optional, Union


_H200_REPO = Path("/home/ssh-ziyue/wam_ausc/current/auscultation")
_H200_DATA = Path("/home/ssh-ziyue/wam_ausc/data")
_H200_OUTPUT = _H200_DATA / "outputs" / "perception"
_LINBOX_REPO = Path("/home/robot/Desktop/stethoscope/.wam-current/auscultation")
_LINBOX_DATA = Path("/home/robot/wam_data/sessions")
_LINBOX_OUTPUT = Path("/home/robot/wam_data/outputs/perception")
_H200_HOSTNAME = "computeinstance-e00saxxvybxg7qvj0s"
_LINBOX_HOSTNAME = "robot-G914"

_SCRIPT_HOST = {
    "h200_training": "h200",
    "h200_evaluation": "h200",
    "linbox_realtime": "linbox",
    "linbox_calibration": "linbox",
}


class RuntimePathError(RuntimeError):
    """A perception job was started on the wrong host or with an unsafe path."""


@dataclass(frozen=True)
class RuntimePaths:
    script_role: str
    host_role: str
    repo_root: Path
    data_root: Path
    output_root: Path


def detect_host_role(
    home: Optional[Union[str, Path]] = None,
    hostname: Optional[str] = None,
) -> str:
    """Bind a role to both the OS account home and the versioned machine identity."""
    actual = (
        Path(home)
        if home is not None
        else Path(pwd.getpwuid(os.geteuid()).pw_dir)
    )
    actual_hostname = socket.gethostname() if hostname is None else hostname
    text = actual.as_posix()
    if text == "/home/ssh-ziyue" and actual_hostname == _H200_HOSTNAME:
        return "h200"
    if text == "/home/robot" and actual_hostname == _LINBOX_HOSTNAME:
        return "linbox"
    return "mac_dev"


def _absolute_override(
    environ: Mapping[str, str], key: str, default: Path,
) -> Path:
    raw = environ.get(key)
    if raw is None or raw == "":
        return default
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise RuntimePathError("%s must be an absolute path" % key)
    if ".." in candidate.parts:
        raise RuntimePathError("%s must not contain parent traversal" % key)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        try:
            if current.is_symlink():
                raise RuntimePathError("%s must not contain a symlink" % key)
        except OSError as exc:
            raise RuntimePathError("%s path cannot be inspected safely" % key) from exc
    return candidate


def resolve_runtime_paths(
    script_role: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
    home: Optional[Union[str, Path]] = None,
    hostname: Optional[str] = None,
    require_exists: bool = True,
) -> RuntimePaths:
    if script_role not in _SCRIPT_HOST:
        raise RuntimePathError("unknown perception runtime role: %s" % script_role)
    env = os.environ if environ is None else environ
    expected_host = _SCRIPT_HOST[script_role]
    detected_host = detect_host_role(home, hostname)
    if detected_host != expected_host:
        test_override = (
            env.get("WAM_TEST_ROLE_OVERRIDE") == "1"
            and env.get("WAM_TESTING") == "1"
        )
        if not test_override:
            label = "H200" if expected_host == "h200" else "linbox"
            if env.get("WAM_TEST_ROLE_OVERRIDE") == "1":
                raise RuntimePathError(
                    "%s role override is test-only and requires WAM_TESTING=1" % label
                )
            raise RuntimePathError(
                "%s requires %s; detected %s" % (script_role, label, detected_host)
            )
        host_role = "test_override"
    else:
        host_role = detected_host

    if expected_host == "h200":
        repo_default, data_default, output_default = (
            _H200_REPO, _H200_DATA, _H200_OUTPUT,
        )
    else:
        repo_default, data_default, output_default = (
            _LINBOX_REPO, _LINBOX_DATA, _LINBOX_OUTPUT,
        )
    repo = _absolute_override(env, "AUSC_REPO", repo_default)
    data = _absolute_override(env, "AUSC_DATA", data_default)
    output = _absolute_override(env, "AUSC_OUTPUT", output_default)
    if expected_host == "h200":
        protected_roots = (
            _H200_DATA / "raw_sessions" / "releases",
            _H200_DATA / "lerobot" / "releases",
            _H200_DATA / "annotation_pools",
        )
        if any(output == root or root in output.parents for root in protected_roots):
            raise RuntimePathError("AUSC_OUTPUT must not target an immutable release tree")
    if require_exists:
        if not repo.is_dir():
            raise RuntimePathError("perception repo root is missing: %s" % repo)
        if not data.is_dir():
            raise RuntimePathError("perception data root is missing: %s" % data)
    return RuntimePaths(
        script_role=script_role,
        host_role=host_role,
        repo_root=repo,
        data_root=data,
        output_root=output,
    )


def activate_runtime(
    script_role: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
    home: Optional[Union[str, Path]] = None,
    hostname: Optional[str] = None,
    module_path: Optional[MutableSequence[str]] = None,
) -> RuntimePaths:
    """Validate the role and persistent roots before changing process state."""
    paths = resolve_runtime_paths(
        script_role, environ=environ, home=home, hostname=hostname,
        require_exists=True,
    )
    os.chdir(str(paths.repo_root))
    search = sys.path if module_path is None else module_path
    prefixes = (
        paths.repo_root / "ausc" / "perception",
        paths.repo_root / "ausc",
        paths.repo_root / "ausc" / "models",
        paths.repo_root / "ausc" / "training",
        paths.repo_root / "scripts",
    )
    for prefix in reversed(prefixes):
        text = str(prefix)
        if text not in search:
            search.insert(0, text)
    return paths


def require_h200_raw_release(
    environment_key: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
    allowed_root: Optional[Union[str, Path]] = None,
    required_files: tuple[str, ...] = (),
) -> Path:
    """Resolve and fully rehash one completed H200 raw-session release."""
    env = os.environ if environ is None else environ
    raw = env.get(environment_key)
    if not raw:
        raise RuntimePathError(
            "%s must name a completed H200 raw release" % environment_key
        )
    candidate = Path(raw)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimePathError("%s must be an absolute release path" % environment_key)
    releases = (
        Path(allowed_root).absolute()
        if allowed_root is not None
        else _H200_DATA / "raw_sessions" / "releases"
    )
    if (
        candidate.is_symlink()
        or candidate.parent.as_posix() != releases.as_posix()
        or not re.fullmatch(r"raw_session-[0-9a-f]{20}", candidate.name)
    ):
        raise RuntimePathError(
            "%s is not a direct immutable H200 raw release" % environment_key
        )
    deployment = Path(__file__).resolve().parents[1] / "deployment"
    deployment_text = str(deployment)
    if deployment_text not in sys.path:
        sys.path.insert(0, deployment_text)
    try:
        import remote_agent

        registry = env.get("WAM_TRAINING_FINALIZATION_REGISTRY")
        snapshots = env.get("WAM_TRAINING_SNAPSHOT_ROOT")
        finalization_sha256 = env.get("WAM_TRAINING_FINALIZATION_SHA256")
        if not registry or not snapshots or not finalization_sha256:
            raise RuntimePathError(
                "trusted H200 finalization registry, snapshot root, and finalization "
                "identity are required"
            )
        verified = remote_agent.validate_raw_session(
            candidate,
            allowed_sessions_root=releases,
            trusted_finalization_registry=registry,
            trusted_snapshot_root=snapshots,
            expected_finalization_sha256=finalization_sha256,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimePathError(
            "%s does not reference a completed raw release" % environment_key
        ) from exc
    verified_root = Path(str(verified["verified_snapshot_path"]))
    for relative in required_files:
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != relative:
            raise RuntimePathError("required perception input path is unsafe")
        path = verified_root / Path(*parsed.parts)
        if path.is_symlink() or not path.is_file():
            raise RuntimePathError(
                "completed raw release lacks required file: %s" % relative
            )
    return verified_root
