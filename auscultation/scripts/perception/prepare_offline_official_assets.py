#!/usr/bin/env python3
"""Freeze the pinned official OpenArm and NVIDIA assets into a local release.

Every network transfer first lands below ``.staging``.  A release is published only
after the existing offline asset contract and the release completion marker both
verify from disk.  This module never creates or changes a ``current`` pointer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
import subprocess
import sys
from typing import Callable, Iterator, Sequence
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
AUSCULTATION_ROOT = WORKSPACE_ROOT / "auscultation"
if os.fspath(AUSCULTATION_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(AUSCULTATION_ROOT))

from ausc.perception.tracking import official_asset_contract as contract  # noqa: E402


BUILD_ROOT = WORKSPACE_ROOT / ".wam-build/offline-official-openarm-assets"
STAGING_ROOT = BUILD_ROOT / ".staging"
RELEASES_ROOT = BUILD_ROOT / "releases"
EVIDENCE_ROOT = WORKSPACE_ROOT / ".wam-build/official-openarm-asset-evidence"
RESULT_ROOT = (
    WORKSPACE_ROOT
    / "docs/runs/2026-07-15-offline-official-mesh-occlusion/raw"
)
RESULT_MANIFEST = RESULT_ROOT / "asset_manifest.json"
RESULT_COMPLETION = RESULT_ROOT / "asset_completion.json"

EXPECTED_MANIFEST_SHA256 = (
    "05e66c94cd8b36ff79bef7bfcaa73d176e621120ac6398e808ad5df938a0cf35"
)
EXPECTED_MANIFEST_SIZE = 14362
COMPLETION_SCHEMA = "wam_official_asset_completion_v1"
RELEASE_ID = f"official_assets-{EXPECTED_MANIFEST_SHA256[:20]}"
RELEASES_ROOT = BUILD_ROOT / "releases"
RELEASE_PATH = RELEASES_ROOT / RELEASE_ID
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 90
GIT_TIMEOUT_SECONDS = 300
GIT_EXECUTABLE = "/usr/bin/git"

_COMPLETION_KEYS = {
    "asset_manifest_sha256",
    "inventory_sha256",
    "release_id",
    "schema",
    "status",
}
_EXPECTED_TOP_LEVEL = {
    contract.CHECKOUT_PATH,
    "human",
    "asset_manifest.json",
    "ASSET_COMPLETE.json",
}
_HEX64 = frozenset("0123456789abcdef")

if (
    len(EXPECTED_MANIFEST_SHA256) != 64
    or not set(EXPECTED_MANIFEST_SHA256) <= _HEX64
):
    raise RuntimeError("official canonical manifest identity must be exactly 64 hex digits")


class PreparationError(RuntimeError):
    """The official release could not be built without weakening the contract."""


@dataclass(frozen=True)
class DownloadSpec:
    url: str
    path: str
    sha256: str
    size: int


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _download_specs() -> tuple[DownloadSpec, ...]:
    """Derive all mutable-looking asset facts from the frozen contract module."""

    root_url = contract.NVIDIA_MEDICAL_HUMAN_URL
    base_url = root_url.rsplit("/", 1)[0]
    return (
        DownloadSpec(
            url=root_url,
            path=contract.HUMAN_USD_PATH,
            sha256=contract.HUMAN_USD_SHA256,
            size=contract.HUMAN_USD_SIZE,
        ),
        *(
            DownloadSpec(
                url=f"{base_url}/textures/{PurePosixPath(path).name}",
                path=path,
                sha256=contract.NVIDIA_TEXTURE_SHA256[path],
                size=contract.NVIDIA_TEXTURE_SIZE[path],
            )
            for path in contract.NVIDIA_TEXTURE_SHA256
        ),
    )


def _allowed_download_hosts() -> frozenset[str]:
    host = urlsplit(contract.NVIDIA_MEDICAL_HUMAN_URL).hostname
    if not host:
        raise PreparationError("the contract NVIDIA URL has no host")
    return frozenset({host.lower()})


def _validate_https_url(url: object, allowed_hosts: frozenset[str]) -> str:
    if type(url) is not str:
        raise PreparationError("download/redirect URL must be a string")
    parsed = urlsplit(url)
    host = parsed.hostname.lower() if parsed.hostname else ""
    if (
        parsed.scheme != "https"
        or host not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or not parsed.path.startswith("/")
        or parsed.fragment
    ):
        raise PreparationError("download/redirect URL must use HTTPS on a pinned host")
    return url


class _PinnedRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _validate_https_url(newurl, _allowed_download_hosts())
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_pinned_url(url: str):
    _validate_https_url(url, _allowed_download_hosts())
    opener = build_opener(ProxyHandler({}), _PinnedRedirectHandler())
    request = Request(url, headers={"User-Agent": "wam-official-asset-freezer/1"})
    return opener.open(request, timeout=NETWORK_TIMEOUT_SECONDS)


def _safe_relative(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise PreparationError("asset path must be a safe relative POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or str(relative) != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PreparationError("asset path must be a safe relative POSIX path")
    return value


def _require_regular_file(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise PreparationError(f"{label} is unavailable: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise PreparationError(f"{label} must be a regular file: {path}")
    return path


def _mkdir_real(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        if path.is_symlink() or not path.is_dir():
            raise PreparationError(f"directory must be real, not a symlink: {path}")
    except OSError as exc:
        raise PreparationError(f"directory is unavailable: {path}") from exc


def _fsync_directory(path: Path) -> None:
    """Persist one directory entry set or fail closed."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise PreparationError(f"could not fsync directory: {path}") from exc


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` only when ``destination`` does not exist.

    macOS ``rename()`` replaces an existing file and can even replace an empty
    directory, so a preceding ``lexists`` check is not a publication contract.
    ``RENAME_EXCL`` is the kernel-enforced no-replace primitive on Darwin.  A
    Linux ``renameat2(RENAME_NOREPLACE)`` path is kept for read-only CI review;
    unknown platforms fail closed instead of falling back to racy rename.
    """

    source = Path(source)
    destination = Path(destination)
    if source.is_symlink() or not os.path.lexists(os.fspath(source)):
        raise PreparationError(f"publication source is missing or a symlink: {source}")
    _mkdir_real(destination.parent)
    libc = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source)
    destination_raw = os.fsencode(destination)
    result: int
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = int(renamex_np(source_raw, destination_raw, 0x00000004))
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = int(renameat2(-100, source_raw, -100, destination_raw, 0x00000001))
    else:
        raise PreparationError(
            "atomic no-replace publication is unavailable on this platform"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise PreparationError(
                f"publication conflict: destination already exists: {destination}"
            )
        raise PreparationError(
            "atomic no-replace publication failed: "
            f"{os.strerror(error_number) if error_number else 'unknown error'}"
        )
    _fsync_directory(destination.parent)
    if source.parent != destination.parent:
        _fsync_directory(source.parent)


def _write_exclusive(path: Path, raw: bytes) -> None:
    if type(raw) is not bytes:
        raise PreparationError("only immutable bytes may be written")
    _mkdir_real(path.parent)
    if os.path.lexists(os.fspath(path)):
        raise PreparationError(f"refusing to overwrite existing path: {path}")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except OSError as exc:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise PreparationError(f"cannot write immutable file: {path}") from exc


def _download_verified(
    url: str,
    target: Path,
    expected_sha256: str,
    expected_size: int,
    *,
    opener: Callable[[str], object] = _open_pinned_url,
) -> None:
    """Stream one pinned URL to an exclusive temp file and atomically install it."""

    _validate_https_url(url, _allowed_download_hosts())
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or not set(expected_sha256) <= _HEX64
        or type(expected_size) is not int
        or expected_size <= 0
    ):
        raise PreparationError("download digest/size contract is invalid")
    _mkdir_real(target.parent)
    if os.path.lexists(os.fspath(target)):
        raise PreparationError(f"download target already exists: {target}")
    temporary = target.parent / f".{target.name}.partial-{secrets.token_hex(12)}"
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with opener(url) as response, os.fdopen(descriptor, "wb") as handle:
                final_url = response.geturl()
                _validate_https_url(final_url, _allowed_download_hosts())
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    if type(chunk) is not bytes:
                        raise PreparationError("download response yielded non-bytes data")
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    if size > expected_size:
                        raise PreparationError("download exceeded the pinned byte size")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise PreparationError("download SHA-256/size does not match the pinned contract")
        _require_regular_file(temporary, "verified temporary download")
        _rename_no_replace(temporary, target)
        actual_sha, actual_size = _sha256_file(_require_regular_file(target, "download"))
        if (actual_sha, actual_size) != (expected_sha256, expected_size):
            raise PreparationError("installed download changed after verification")
    except PreparationError:
        raise
    except Exception as exc:
        raise PreparationError(f"pinned download failed: {url}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            raise PreparationError(f"could not remove partial download: {temporary}") from exc


def _remove_owned_staging(path: Path) -> None:
    if not os.path.lexists(os.fspath(path)):
        return
    try:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            path.unlink()
        else:
            shutil.rmtree(path)
    except OSError as exc:
        raise PreparationError(f"could not clean owned staging directory: {path}") from exc


@contextmanager
def _staging_bundle(build_root: Path = BUILD_ROOT) -> Iterator[Path]:
    staging_root = build_root / ".staging"
    _mkdir_real(staging_root)
    staging = staging_root / f"build-{secrets.token_hex(16)}"
    try:
        staging.mkdir(mode=0o700)
    except OSError as exc:
        raise PreparationError("could not create exclusive staging directory") from exc
    try:
        yield staging
    finally:
        _remove_owned_staging(staging)


def _git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/usr/bin/false",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _run_git(args: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            [GIT_EXECUTABLE, *args],
            cwd=os.fspath(cwd) if cwd is not None else None,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreparationError("official Git operation failed or timed out") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git failed"
        raise PreparationError(f"official Git operation failed: {detail}")
    return result


def _clone_official_checkout(bundle: Path) -> Path:
    checkout = bundle / contract.CHECKOUT_PATH
    if os.path.lexists(os.fspath(checkout)):
        raise PreparationError("official checkout staging path already exists")
    _run_git(
        (
            "-c",
            "http.followRedirects=false",
            "-c",
            "protocol.file.allow=never",
            "-c",
            "protocol.ext.allow=never",
            "-c",
            "core.hooksPath=/dev/null",
            "clone",
            contract.ENACTIC_REPOSITORY_URL,
            os.fspath(checkout),
        )
    )
    _run_git(("checkout", "--detach", contract.ENACTIC_COMMIT), cwd=checkout)
    _verify_checkout_transport_state(bundle)
    return checkout


def _verify_checkout_transport_state(bundle: Path) -> None:
    """Reject transport metadata that could make a release externally mutable."""

    checkout = bundle / contract.CHECKOUT_PATH
    if checkout.is_symlink() or not checkout.is_dir():
        raise PreparationError("official Git checkout must be a real directory")
    git_directory = checkout / ".git"
    if git_directory.is_symlink() or not git_directory.is_dir():
        raise PreparationError("official checkout .git must be a real directory")
    for relative, label in (
        ("shallow", "shallow Git checkout is forbidden"),
        ("objects/info/alternates", "Git object alternates are forbidden"),
        ("objects/info/http-alternates", "Git HTTP object alternates are forbidden"),
    ):
        if os.path.lexists(os.fspath(git_directory / relative)):
            raise PreparationError(label)
    pack_directory = git_directory / "objects/pack"
    try:
        if pack_directory.is_symlink() or not pack_directory.is_dir():
            raise PreparationError("Git pack directory must be a real directory")
        with os.scandir(pack_directory) as entries:
            if any(entry.name.endswith(".promisor") for entry in entries):
                raise PreparationError(
                    "partial-clone/promisor Git pack metadata is forbidden"
                )
    except PreparationError:
        raise
    except OSError as exc:
        raise PreparationError("Git pack metadata could not be inspected") from exc
    shallow = _run_git(
        ("rev-parse", "--is-shallow-repository"), cwd=checkout
    ).stdout.strip()
    if shallow != "false":
        raise PreparationError("official Git checkout is not a full clone")
    config = _run_git(("config", "--local", "--null", "--list"), cwd=checkout)
    for record in config.stdout.split("\0"):
        if not record:
            continue
        key = record.partition("\n")[0].strip().lower()
        if key == "extensions.partialclone" or (
            key.startswith("remote.")
            and key.endswith((".promisor", ".partialclonefilter"))
        ):
            raise PreparationError("partial-clone/promisor Git metadata is forbidden")


def _source_evidence_file(relative: str) -> Path:
    relative = _safe_relative(relative)
    if EVIDENCE_ROOT.is_symlink() or not EVIDENCE_ROOT.is_dir():
        raise PreparationError("frozen evidence root must be a real directory")
    candidate = EVIDENCE_ROOT
    resolved_root = EVIDENCE_ROOT.resolve(strict=True)
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise PreparationError("frozen evidence path contains a symlink")
    try:
        candidate.resolve(strict=True).relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise PreparationError("frozen evidence path escapes its root") from exc
    return _require_regular_file(candidate, "frozen evidence")


def _copy_frozen_evidence(bundle: Path) -> None:
    for relative in (
        contract.DEPENDENCY_REPORT_PATH,
        contract.LICENSE_PAGE_PATH,
        contract.LICENSE_EVIDENCE_PATH,
    ):
        source = _source_evidence_file(relative)
        raw = source.read_bytes()
        if not raw:
            raise PreparationError(f"frozen evidence is empty: {relative}")
        target = bundle.joinpath(*PurePosixPath(relative).parts)
        _write_exclusive(target, raw)
        if target.read_bytes() != raw:
            raise PreparationError(f"frozen evidence bytes changed during copy: {relative}")


def _assert_tree_no_symlinks(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise PreparationError(f"asset root must be a real directory: {root}")
    pending = [root]
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        raise PreparationError(f"asset release contains symlink: {entry.path}")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif not entry.is_file(follow_symlinks=False):
                        raise PreparationError(
                            f"asset release contains non-regular entry: {entry.path}"
                        )
    except PreparationError:
        raise
    except OSError as exc:
        raise PreparationError("asset release tree could not be fully inspected") from exc


def _fsync_tree(root: Path) -> None:
    """Flush every staged regular file and directory before publication."""

    _assert_tree_no_symlinks(root)
    directories: list[Path] = []
    try:
        for directory, names, files in os.walk(root, topdown=True, followlinks=False):
            directory_path = Path(directory)
            directories.append(directory_path)
            names.sort()
            files.sort()
            for name in files:
                path = directory_path / name
                before = path.lstat()
                if not stat.S_ISREG(before.st_mode):
                    raise PreparationError(f"staged entry is not a regular file: {path}")
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(path, flags)
                try:
                    opened = os.fstat(descriptor)
                    if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                        raise PreparationError(f"staged file changed while flushing: {path}")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        for directory in reversed(directories):
            _fsync_directory(directory)
    except PreparationError:
        raise
    except OSError as exc:
        raise PreparationError("could not durably flush the staged release") from exc
    _assert_tree_no_symlinks(root)


def _completion_bytes(manifest: object, manifest_sha256: str) -> bytes:
    if (
        type(manifest) is not dict
        or type(manifest.get("files")) is not list
        or manifest_sha256 != EXPECTED_MANIFEST_SHA256
    ):
        raise PreparationError("completion marker requires the exact canonical manifest")
    inventory_raw = contract.canonical_json_bytes(manifest["files"])
    marker = {
        "asset_manifest_sha256": manifest_sha256,
        "inventory_sha256": _sha256(inventory_raw),
        "release_id": f"official_assets-{manifest_sha256[:20]}",
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
    }
    return contract.canonical_json_bytes(marker)


def _verify_completion(raw: bytes, manifest: dict[str, object]) -> dict[str, object]:
    try:
        marker = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError("completion marker is not valid UTF-8 JSON") from exc
    if type(marker) is not dict or set(marker) != _COMPLETION_KEYS:
        raise PreparationError("completion marker has the wrong exact schema keys")
    expected = _completion_bytes(manifest, EXPECTED_MANIFEST_SHA256)
    if raw != expected or contract.canonical_json_bytes(marker) != raw:
        raise PreparationError("completion marker is noncanonical or not bound to the release")
    return marker


def _verify_bundle_contents(root: Path) -> dict[str, object]:
    _assert_tree_no_symlinks(root)
    try:
        top_level = {entry.name for entry in os.scandir(root)}
    except OSError as exc:
        raise PreparationError("release top level cannot be inspected") from exc
    if top_level != _EXPECTED_TOP_LEVEL:
        raise PreparationError("release has unexpected or missing top-level content")
    manifest_path = _require_regular_file(root / "asset_manifest.json", "asset manifest")
    manifest_raw = manifest_path.read_bytes()
    if (
        len(manifest_raw) != EXPECTED_MANIFEST_SIZE
        or _sha256(manifest_raw) != EXPECTED_MANIFEST_SHA256
    ):
        raise PreparationError(
            "canonical asset manifest SHA-256/size is not the frozen identity"
        )
    try:
        manifest = contract.verify_manifest(manifest_raw, root)
    except contract.AssetContractError as exc:
        raise PreparationError(f"official asset contract failed: {exc}") from exc
    marker_path = _require_regular_file(root / "ASSET_COMPLETE.json", "completion marker")
    _verify_completion(marker_path.read_bytes(), manifest)
    return manifest


def verify_release(release: Path = RELEASE_PATH) -> dict[str, object]:
    release = Path(release)
    if release.name != RELEASE_ID:
        raise PreparationError("release directory name does not match its exact identity")
    _verify_checkout_transport_state(release)
    return _verify_bundle_contents(release)


def _write_result_copy(path: Path, raw: bytes) -> None:
    _mkdir_real(path.parent)
    if os.path.lexists(os.fspath(path)):
        existing = _require_regular_file(path, "result evidence").read_bytes()
        if existing != raw:
            raise PreparationError(f"existing result evidence conflicts: {path}")
        return
    temporary = path.parent / f".{path.name}.partial-{secrets.token_hex(12)}"
    try:
        _write_exclusive(temporary, raw)
        _rename_no_replace(temporary, path)
        if _require_regular_file(path, "result evidence").read_bytes() != raw:
            raise PreparationError(f"published result evidence changed: {path}")
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            raise PreparationError(f"could not clean result partial: {temporary}") from exc


def _emit_results(release: Path) -> None:
    _write_result_copy(RESULT_MANIFEST, (release / "asset_manifest.json").read_bytes())
    _write_result_copy(RESULT_COMPLETION, (release / "ASSET_COMPLETE.json").read_bytes())


def _refuse_current_pointer() -> None:
    if os.path.lexists(os.fspath(BUILD_ROOT / "current")):
        raise PreparationError("a current pointer is forbidden for this offline release")


def _build_staged_release(staging: Path) -> None:
    _clone_official_checkout(staging)
    for spec in _download_specs():
        target = staging.joinpath(*PurePosixPath(_safe_relative(spec.path)).parts)
        _download_verified(spec.url, target, spec.sha256, spec.size)
    _copy_frozen_evidence(staging)
    _assert_tree_no_symlinks(staging)
    try:
        manifest = contract.build_manifest(staging)
    except contract.AssetContractError as exc:
        raise PreparationError(f"official asset manifest build failed: {exc}") from exc
    manifest_raw = contract.canonical_json_bytes(manifest)
    if (
        len(manifest_raw) != EXPECTED_MANIFEST_SIZE
        or _sha256(manifest_raw) != EXPECTED_MANIFEST_SHA256
    ):
        raise PreparationError(
            "built manifest does not match the frozen canonical SHA-256/size"
        )
    _write_exclusive(staging / "asset_manifest.json", manifest_raw)
    disk_raw = (staging / "asset_manifest.json").read_bytes()
    try:
        disk_manifest = contract.verify_manifest(disk_raw, staging)
    except contract.AssetContractError as exc:
        raise PreparationError(f"on-disk asset manifest verification failed: {exc}") from exc
    _write_exclusive(
        staging / "ASSET_COMPLETE.json",
        _completion_bytes(disk_manifest, _sha256(disk_raw)),
    )
    _verify_bundle_contents(staging)


def _release_directory_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PreparationError(f"release directory is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PreparationError(f"release must be a real directory: {path}")
    return int(metadata.st_dev), int(metadata.st_ino)


def _rollback_owned_release(path: Path, identity: tuple[int, int]) -> None:
    """Remove only the exact directory inode published by this process."""

    if _release_directory_identity(path) != identity:
        raise PreparationError(
            "post-publication verification failed and release identity changed; "
            "refusing unsafe rollback"
        )
    try:
        shutil.rmtree(path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise PreparationError(
            "post-publication verification failed and owned release rollback failed"
        ) from exc


def _publish_staged_release(
    staging: Path, *, destination: Path = RELEASE_PATH
) -> None:
    """Reverify, durably publish, and rollback an owned invalid publication."""

    staging = Path(staging)
    destination = Path(destination)
    _fsync_tree(staging)
    _verify_checkout_transport_state(staging)
    _verify_bundle_contents(staging)
    _refuse_current_pointer()
    if os.path.lexists(os.fspath(destination)):
        raise PreparationError("release identity appeared concurrently; refusing conflict")
    identity = _release_directory_identity(staging)
    _rename_no_replace(staging, destination)
    if _release_directory_identity(destination) != identity:
        raise PreparationError("published release directory identity changed")
    try:
        verify_release(destination)
    except Exception:
        _rollback_owned_release(destination, identity)
        raise


def prepare_offline_official_assets() -> Path:
    """Build or fully reverify the immutable-style release and return its path."""

    _mkdir_real(BUILD_ROOT)
    _mkdir_real(STAGING_ROOT)
    _mkdir_real(RELEASES_ROOT)
    _refuse_current_pointer()
    if os.path.lexists(os.fspath(RELEASE_PATH)):
        verify_release(RELEASE_PATH)
        _emit_results(RELEASE_PATH)
        _refuse_current_pointer()
        return RELEASE_PATH

    with _staging_bundle(BUILD_ROOT) as staging:
        _build_staged_release(staging)
        _publish_staged_release(staging)

    verify_release(RELEASE_PATH)
    _emit_results(RELEASE_PATH)
    _refuse_current_pointer()
    return RELEASE_PATH


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the exact final release without downloading or publishing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify_only:
            manifest = verify_release(RELEASE_PATH)
            release = RELEASE_PATH
        else:
            release = prepare_offline_official_assets()
            manifest = verify_release(release)
        summary = {
            "asset_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "asset_manifest_size": EXPECTED_MANIFEST_SIZE,
            "release_id": RELEASE_ID,
            "release_path": os.fspath(release),
            "status": "complete",
            "tracked_asset_rows": len(manifest["files"]),
        }
        print(contract.canonical_json_bytes(summary).decode("utf-8"))
        return 0
    except PreparationError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
