"""Relocating, selective, hardened canonical re-finalizer for a production activation package.

The commissioning activation package was written non-canonically (no trailing
newline; two files non-compact) while both boot gates require the exact canonical
encoding + one trailing newline (``artifact_contract.canonical_json_bytes``).  The
production loader (``ausc/perception/production_runtime_activation.py``) reads only
a specific set of files as canonical -- the activation manifest, the top-level
``COMPLETE`` marker, and every artifact ``COMPLETE`` marker (plus the ``handeye``
payload, already canonical in its read-only external release) -- and reads every
artifact PAYLOAD + the OpenArm PAIR with raw-SHA integrity only.  It also
cross-checks raw payload SHAs against the pair identity, so a payload can never be
canonicalized without breaking the manifest reference and the pair.

This builder produces a new self-contained, relocatable release:

  * canonicalizes only the canonical-enforced files, rebinding the
    ``complete_sha256`` chain and the top marker's ``activation_sha256``;
  * copies every artifact payload and the pair sub-package byte-for-byte from
    already-verified bytes (the source is opened exactly once, TOCTOU-safe);
  * relocates into a new release root, rewriting the manifest's absolute paths,
    leaving ONLY ``handeye`` referenced in place at its read-only external release
    (the other 11 artifacts MUST enter the release);
  * enforces the loader's exact contract on the source (12-item artifact allowlist,
    manifest/entry/pair/bounds key sets, schemas, strict JSON with no duplicate
    keys / no non-finite numbers);
  * writes a new immutable release with resolve-under-root output paths, O_EXCL +
    O_NOFOLLOW + full short-write loop + fsync of file/dir/parent, files 0444 /
    dirs 0555, COMPLETE written last;
  * runs a MANDATORY full consumer verification of the staged release before it is
    ever published (atomic rename), so no consumable release can exist without
    passing the real loader;
  * is idempotent only after a full re-verification of an existing release
    (inventory exactness, every payload/marker/pair byte, provenance, no extra
    files); a same-id/different-content release is refused;
  * binds provenance to the source activation SHA, the re-finalizer code SHA +
    release, owner authorization, source run, the final activation SHA, and the
    complete in-release inventory.

It never mutates the source or any external input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Callable, Mapping

# --- Contract constants (must match the production loader exactly) ----------- #
ACTIVATION_FILE = "activation.json"
COMPLETE_MARKER = "COMPLETE"
STAGING_MARKER = "STAGING"
REFINALIZE_PROVENANCE_FILE = "REFINALIZED.json"
REFINALIZE_PROVENANCE_SCHEMA = "wam_activation_refinalization_provenance_v1"

ACTIVATION_SCHEMA = "wam_production_runtime_activation_v1"
ACTIVATION_COMPLETION_SCHEMA = "wam_production_runtime_activation_completion_v1"
ARTIFACT_COMPLETION_SCHEMA = "wam_production_runtime_artifact_completion_v1"

_MAX_JSON_BYTES = 64 * 1024 * 1024
_HANDEYE = "handeye"  # the ONLY artifact permitted to remain external

_ARTIFACT_NAMES = frozenset({
    "alignment", "back_mask", "camera_latency", "camera_profile",
    "collision_geometry", "depth_noise", "handeye", "occlusion_identity",
    "occlusion_runtime", "patient_pose", "robot_mask_uncertainty", "tool_geometry",
})
_MANIFEST_KEYS = frozenset({
    "artifacts", "follower_can_interface", "leader_can_interface",
    "openarm_pair", "robot_state_bounds", "schema", "side", "status",
})
_ENTRY_KEYS = frozenset({"complete_path", "complete_sha256", "path", "sha256"})
_PAIR_KEYS = frozenset({"contract_sha256", "release_directory"})
_COMPLETE_KEYS = frozenset({"activation_sha256", "schema", "status"})
_BOUNDS_KEYS = frozenset({
    "buffer_duration_ns", "gripper_motor_position_limits_rad",
    "joint_position_limits_rad", "max_abs_joint_velocity_rad_s",
    "max_exposure_interval_ns", "max_gap_ns", "max_packets", "max_staleness_ns",
})
_DYNAMIC_COMPONENTS = frozenset({"active", "current", "latest", ".wam-current", "wam-current"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ActivationRefinalizerError(RuntimeError):
    """The activation package cannot be re-finalised into a trustworthy release."""


def canonical_json_bytes(value: object) -> bytes:
    """The one accepted encoding: compact, sorted, non-ASCII-preserving, + newline."""
    try:
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ActivationRefinalizerError("value is not canonical-JSON encodable") from exc
    return (text + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ActivationRefinalizerError(f"{label} is not a lowercase SHA-256")
    return value


# --- Strict JSON (reject duplicate keys + non-finite numbers) ---------------- #

def _reject_duplicate_keys(pairs):
    seen: dict = {}
    for key, val in pairs:
        if key in seen:
            raise ActivationRefinalizerError(f"duplicate JSON key: {key!r}")
        seen[key] = val
    return seen


def _reject_constant(token: str):
    raise ActivationRefinalizerError(f"non-finite JSON constant: {token}")


def _strict_json_loads(raw: bytes, *, label: str) -> object:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ActivationRefinalizerError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise ActivationRefinalizerError(f"{label} is not valid strict JSON") from exc


# --- Path safety ------------------------------------------------------------- #

def _validate_source_path(value: object, *, label: str, directory: bool) -> Path:
    """Absolute, no current/latest pointer, no symlink component (mirrors the loader)."""
    try:
        path = Path(os.fspath(value))
    except TypeError as exc:
        raise ActivationRefinalizerError(f"{label} is not path-like") from exc
    if not path.is_absolute():
        raise ActivationRefinalizerError(f"{label} must be absolute")
    if any(part.lower() in _DYNAMIC_COMPONENTS for part in PurePosixPath(str(path)).parts):
        raise ActivationRefinalizerError(f"{label} must not use a current/latest/active pointer")
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ActivationRefinalizerError(f"{label} is missing") from exc
    if stat.S_ISLNK(info.st_mode) or resolved != path:
        raise ActivationRefinalizerError(f"{label} or a parent component is a symlink")
    ok = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not ok:
        raise ActivationRefinalizerError(f"{label} is not a {'directory' if directory else 'file'}")
    return path


def _safe_child(root: Path, rel: str) -> Path:
    """Join ``rel`` under ``root`` rejecting '..', absolute, and dynamic components."""
    rel_path = PurePosixPath(rel)
    if rel_path.is_absolute() or any(
        part in ("", "..", ".") or part.lower() in _DYNAMIC_COMPONENTS for part in rel_path.parts
    ):
        raise ActivationRefinalizerError(f"unsafe release-relative path: {rel!r}")
    child = (root / rel_path).resolve()
    if root.resolve() != child and root.resolve() not in child.parents:
        raise ActivationRefinalizerError(f"release path escapes the root: {rel!r}")
    return root / rel_path


# --- TOCTOU-safe single-FD source read --------------------------------------- #

def _read_source_bytes(path: Path, *, label: str) -> bytes:
    """Open once (O_NOFOLLOW), fstat/read/hash on the same FD, TOCTOU identity check."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise ActivationRefinalizerError(f"{label} cannot be opened") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_JSON_BYTES:
            raise ActivationRefinalizerError(f"{label} has an invalid size")
        chunks, remaining = [], before.st_size
        while remaining:
            block = os.read(fd, min(remaining, 1024 * 1024))
            if not block:
                raise ActivationRefinalizerError(f"{label} was truncated while reading")
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise ActivationRefinalizerError(f"{label} changed while reading")
    finally:
        os.close(fd)
    return b"".join(chunks)


def _read_source_json(path: Path, *, label: str) -> "tuple[bytes, object]":
    _validate_source_path(path, label=label, directory=False)
    raw = _read_source_bytes(path, label=label)
    return raw, _strict_json_loads(raw, label=label)


# --- Atomic, fsynced, short-write-safe output -------------------------------- #

def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    total = 0
    while total < len(view):
        total += os.write(fd, view[total:])


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.parent / (path.name + ".inprogress")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(tmp), flags, 0o444)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))
    _fsync_dir(path.parent)


def _fsync_dir(path: Path) -> None:
    dir_fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _finalize_modes(root: Path) -> None:
    for p in sorted(root.rglob("*"), reverse=True):
        if p.is_symlink():
            raise ActivationRefinalizerError(f"release must not contain symlinks: {p}")
        os.chmod(p, 0o555 if p.is_dir() else 0o444)
    os.chmod(root, 0o555)


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


# --- Planning ---------------------------------------------------------------- #

@dataclass(frozen=True)
class _Plan:
    name: str
    action: str            # "keep" | "relocate"
    entry: dict            # NEW manifest entry
    payload_bytes: "bytes | None" = None
    payload_rel: "str | None" = None
    marker_bytes: "bytes | None" = None
    marker_rel: "str | None" = None


def _plan_artifact(name: str, entry_value: object, *, new_root: Path) -> _Plan:
    entry = entry_value
    if not isinstance(entry, dict) or frozenset(entry) != _ENTRY_KEYS:
        raise ActivationRefinalizerError(f"{name} artifact entry keys are not exactly {sorted(_ENTRY_KEYS)}")
    payload_sha = _require_sha256(entry["sha256"], label=f"{name} payload SHA")
    marker_sha = _require_sha256(entry["complete_sha256"], label=f"{name} marker SHA")
    payload_path = _validate_source_path(entry["path"], label=f"{name} payload path", directory=False)
    marker_path = _validate_source_path(entry["complete_path"], label=f"{name} marker path", directory=False)

    payload_raw = _read_source_bytes(payload_path, label=f"{name} payload")
    marker_raw = _read_source_bytes(marker_path, label=f"{name} marker")
    if _sha256(payload_raw) != payload_sha:
        raise ActivationRefinalizerError(f"{name} payload SHA does not match the manifest")
    if _sha256(marker_raw) != marker_sha:
        raise ActivationRefinalizerError(f"{name} marker SHA does not match the manifest")
    marker_obj = _strict_json_loads(marker_raw, label=f"{name} marker")

    if name == _HANDEYE:
        # ONLY handeye may stay external, and only if it is already canonical.
        payload_obj = _strict_json_loads(payload_raw, label=f"{name} payload")
        if payload_raw != canonical_json_bytes(payload_obj) or marker_raw != canonical_json_bytes(marker_obj):
            raise ActivationRefinalizerError("handeye external release must already be canonical")
        return _Plan(name=name, action="keep", entry=dict(entry))

    # All other 11 artifacts MUST enter the release.  Canonicalize the marker and
    # require the standard binding shape; copy the payload byte-for-byte.
    expected_marker = {"artifact_sha256": payload_sha, "schema": ARTIFACT_COMPLETION_SCHEMA, "status": "COMPLETE"}
    if not isinstance(marker_obj, dict) or marker_obj != expected_marker:
        raise ActivationRefinalizerError(f"{name} COMPLETE marker does not bind the payload")
    marker_bytes = canonical_json_bytes(marker_obj)
    payload_rel = f"artifacts/{name}.json"
    marker_rel = f"markers/{name}.COMPLETE"
    new_entry = {
        "path": str(new_root / payload_rel),
        "complete_path": str(new_root / marker_rel),
        "sha256": payload_sha,                       # payload copied byte-for-byte
        "complete_sha256": _sha256(marker_bytes),    # marker canonicalized -> rebound
    }
    return _Plan(
        name=name, action="relocate", entry=new_entry,
        payload_bytes=payload_raw, payload_rel=payload_rel,
        marker_bytes=marker_bytes, marker_rel=marker_rel,
    )


def _read_pair_tree(pair_src: Path) -> "list[tuple[str, bytes]]":
    """Capture every pair file's verified bytes once; reject symlinks."""
    out: list = []
    for child in sorted(pair_src.rglob("*")):
        if child.is_symlink():
            raise ActivationRefinalizerError(f"pair sub-package contains a symlink: {child}")
        if child.is_dir():
            continue
        rel = child.relative_to(pair_src).as_posix()
        out.append((rel, _read_source_bytes(child, label=f"pair file {rel}")))
    if not out:
        raise ActivationRefinalizerError("pair sub-package is empty")
    return out


# --- Consumer verification (pluggable; default imports the real loader) ------- #

def _loader_consumer_verify(release_dir: Path, final_activation_sha: str) -> None:
    """Full consumer verification via the real production loader (box-side)."""
    import collection_service_unit as csu  # noqa: F401
    from ausc.perception import production_runtime_activation as pra

    act = Path(release_dir) / ACTIVATION_FILE
    csu.verify_production_activation(act, final_activation_sha)
    _p, manifest, _s = pra._load_activation_manifest(act, expected_sha256=final_activation_sha)
    for artifact_name in sorted(pra._ARTIFACT_NAMES):
        pra._load_artifact(artifact_name, manifest["artifacts"][artifact_name])
    pair = manifest["openarm_pair"]
    pra.load_activated_openarm_pair(pair["release_directory"], expected_contract_sha256=pair["contract_sha256"])


ConsumerVerifier = Callable[[Path, str], None]


# --- Build ------------------------------------------------------------------- #

def refinalize_activation_release(
    source_dir: "str | Path",
    releases_root: "str | Path",
    *,
    owner_authorization: str,
    source_run: str,
    refinalizer_code_sha256: str,
    refinalizer_code_release: str,
    consumer_verifier: "ConsumerVerifier | None" = None,
) -> Path:
    """Build a new immutable, self-contained, canonical activation release.

    ``consumer_verifier`` is MANDATORY-in-effect: the staged release is verified by
    it before publish (defaults to the real production loader).  Returns the release
    directory.  Idempotent only after a full re-verification of an existing release.
    """

    for label, value in (
        ("owner authorization", owner_authorization), ("source-run identity", source_run),
        ("refinalizer code release", refinalizer_code_release),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ActivationRefinalizerError(f"{label} is required")
    _require_sha256(refinalizer_code_sha256, label="refinalizer code SHA")
    verify: ConsumerVerifier = consumer_verifier or _loader_consumer_verify

    source = _validate_source_path(source_dir, label="activation source", directory=True)
    releases = Path(releases_root)
    if not releases.is_absolute():
        raise ActivationRefinalizerError("releases root must be absolute")

    # --- Read + validate the source (single-FD, strict) ---------------------- #
    manifest_raw, manifest = _read_source_json(source / ACTIVATION_FILE, label="activation manifest")
    if not isinstance(manifest, dict) or frozenset(manifest) != _MANIFEST_KEYS:
        raise ActivationRefinalizerError("activation manifest keys are not exact")
    if (manifest["schema"] != ACTIVATION_SCHEMA or manifest["status"] != "COMPLETE"
            or manifest["side"] != "left" or manifest["leader_can_interface"] != "can1"
            or manifest["follower_can_interface"] != "can3"):
        raise ActivationRefinalizerError("activation must be COMPLETE left can1 leader -> can3 follower")
    bounds = manifest["robot_state_bounds"]
    if not isinstance(bounds, dict) or frozenset(bounds) != _BOUNDS_KEYS:
        raise ActivationRefinalizerError("robot_state_bounds keys are not exact")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or frozenset(artifacts) != _ARTIFACT_NAMES:
        raise ActivationRefinalizerError("activation artifacts are not exactly the 12-item allowlist")
    source_activation_sha = _sha256(manifest_raw)

    complete_raw, complete = _read_source_json(source / COMPLETE_MARKER, label="top COMPLETE marker")
    if not isinstance(complete, dict) or frozenset(complete) != _COMPLETE_KEYS:
        raise ActivationRefinalizerError("top COMPLETE marker keys are not exact")
    if (complete["activation_sha256"] != source_activation_sha
            or complete["schema"] != ACTIVATION_COMPLETION_SCHEMA or complete["status"] != "COMPLETE"):
        raise ActivationRefinalizerError("top COMPLETE marker does not bind the source manifest")

    # Release identity keyed off (source sha, absolute releases root): same id can
    # never map to two different final activation SHAs.
    release_id = "activation-" + _sha256(
        (source_activation_sha + "|" + str(releases.resolve())).encode("utf-8")
    )[:20]
    dest = releases / release_id

    plans = [_plan_artifact(n, artifacts[n], new_root=dest) for n in sorted(artifacts)]

    pair_entry = manifest["openarm_pair"]
    if not isinstance(pair_entry, dict) or frozenset(pair_entry) != _PAIR_KEYS:
        raise ActivationRefinalizerError("openarm_pair keys are not exact")
    _require_sha256(pair_entry["contract_sha256"], label="pair contract SHA")
    pair_src = _validate_source_path(pair_entry["release_directory"], label="openarm pair release", directory=True)
    pair_rel = f"pairs/{pair_src.name}"
    pair_files = _read_pair_tree(pair_src)
    new_pair_entry = {"contract_sha256": pair_entry["contract_sha256"], "release_directory": str(dest / pair_rel)}

    # --- Build new manifest + COMPLETE + provenance -------------------------- #
    new_manifest = dict(manifest)
    new_manifest["artifacts"] = {p.name: p.entry for p in plans}
    new_manifest["openarm_pair"] = new_pair_entry
    new_manifest_bytes = canonical_json_bytes(new_manifest)
    final_activation_sha = _sha256(new_manifest_bytes)
    new_complete_bytes = canonical_json_bytes(
        {"activation_sha256": final_activation_sha, "schema": ACTIVATION_COMPLETION_SCHEMA, "status": "COMPLETE"}
    )
    _assert_semantic_equivalence(manifest, new_manifest, plans)

    inventory = {ACTIVATION_FILE: final_activation_sha, COMPLETE_MARKER: _sha256(new_complete_bytes)}
    for p in plans:
        if p.action == "relocate":
            inventory[p.payload_rel] = p.entry["sha256"]
            inventory[p.marker_rel] = p.entry["complete_sha256"]
    for rel, data in pair_files:
        inventory[f"{pair_rel}/{rel}"] = _sha256(data)

    provenance = {
        "schema": REFINALIZE_PROVENANCE_SCHEMA,
        "source_activation_sha256": source_activation_sha,
        "final_activation_sha256": final_activation_sha,
        "refinalizer_code_sha256": refinalizer_code_sha256,
        "refinalizer_code_release": refinalizer_code_release,
        "owner_authorization": owner_authorization,
        "source_run": source_run,
        "release_id": release_id,
        "release_directory": str(dest),
        "kept_external_artifacts": sorted(p.name for p in plans if p.action == "keep"),
        "relocated_artifacts": sorted(p.name for p in plans if p.action == "relocate"),
        "inventory": inventory,
    }
    prov_bytes = canonical_json_bytes(provenance)

    all_files = {
        ACTIVATION_FILE: new_manifest_bytes, COMPLETE_MARKER: new_complete_bytes,
        REFINALIZE_PROVENANCE_FILE: prov_bytes,
    }
    for p in plans:
        if p.action == "relocate":
            all_files[p.payload_rel] = p.payload_bytes
            all_files[p.marker_rel] = p.marker_bytes
    for rel, data in pair_files:
        all_files[f"{pair_rel}/{rel}"] = data

    if dest.exists():
        return _reuse_or_refuse(dest, all_files, provenance, final_activation_sha, verify)

    _materialize(releases, dest, all_files, final_activation_sha, verify)
    return dest


def _assert_semantic_equivalence(source: Mapping, new: Mapping, plans: "list[_Plan]") -> None:
    for key in _MANIFEST_KEYS - {"artifacts", "openarm_pair"}:
        if source[key] != new[key]:
            raise ActivationRefinalizerError(f"manifest field {key} changed unexpectedly")
    for p in plans:
        s, n = source["artifacts"][p.name], new["artifacts"][p.name]
        if s["sha256"] != n["sha256"]:
            raise ActivationRefinalizerError(f"{p.name} payload SHA must not change")
        if p.action == "keep" and s != n:
            raise ActivationRefinalizerError(f"kept artifact {p.name} entry must be identical")
    if source["openarm_pair"]["contract_sha256"] != new["openarm_pair"]["contract_sha256"]:
        raise ActivationRefinalizerError("pair contract identity must not change")


def _materialize(releases, dest, all_files, final_sha, verify) -> None:
    """Build directly at the final path (manifest paths are absolute-final).

    A ``STAGING`` marker is present until verification passes, so the release is
    never loader-consumable while unverified (the loader rejects a STAGING-marked
    release); the marker is removed last, as the atomic commit.  On any failure the
    whole directory is destroyed, and since ``dest`` did not pre-exist, nothing that
    could be trusted is left behind.
    """

    if dest.exists():
        raise ActivationRefinalizerError("release path already exists; refusing to build over it")
    releases.mkdir(parents=True, exist_ok=True)
    dest.mkdir(mode=0o755)
    staging_marker = dest / STAGING_MARKER
    try:
        _atomic_write(staging_marker, b"STAGING\n")  # not consumable until removed
        for rel, data in all_files.items():
            if rel == COMPLETE_MARKER:
                continue
            out = _safe_child(dest, rel)
            out.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(out, data)
        _atomic_write(_safe_child(dest, COMPLETE_MARKER), all_files[COMPLETE_MARKER])
        # Commit: drop the STAGING marker (the loader rejects a STAGING-marked
        # release, so it was non-consumable up to this point), then run the MANDATORY
        # full consumer verification.  Any failure destroys the whole directory, and
        # nothing references this fresh path until it is explicitly SHA-pinned.
        os.chmod(staging_marker, 0o644)
        os.unlink(staging_marker)
        verify(dest, final_sha)
        _finalize_modes(dest)
        _fsync_dir(dest.parent)
    except BaseException:
        try:
            os.chmod(dest, 0o700)
            for p in sorted(dest.rglob("*"), reverse=True):
                os.chmod(p, 0o700 if p.is_dir() else 0o600)
        except OSError:
            pass
        _rmtree(dest)
        raise


def _reuse_or_refuse(dest, all_files, provenance, final_sha, verify) -> Path:
    dest = Path(dest)
    if dest.is_symlink() or not dest.is_dir():
        raise ActivationRefinalizerError("existing activation release path is not a directory")
    # Exact inventory: the on-disk files must be EXACTLY the expected set.
    expected = set(all_files) | {REFINALIZE_PROVENANCE_FILE}
    on_disk = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    if on_disk != expected:
        raise ActivationRefinalizerError(
            f"existing release {dest.name} inventory differs (missing/extra files)"
        )
    for rel, data in all_files.items():
        f = dest / rel
        if f.is_symlink() or not f.is_file() or f.read_bytes() != data:
            raise ActivationRefinalizerError(f"existing release {dest.name} differs at {rel}")
    if (dest / REFINALIZE_PROVENANCE_FILE).read_bytes() != canonical_json_bytes(provenance):
        raise ActivationRefinalizerError(f"existing release {dest.name} provenance differs")
    verify(dest, final_sha)  # re-run full consumer verification on reuse
    return dest


def verify_relocated_release(release_dir: "str | Path") -> dict:
    """Standard-library preflight mirroring the loader's canonical + binding contract."""
    root = Path(release_dir)
    manifest_raw, manifest = _read_source_json(root / ACTIVATION_FILE, label="manifest")
    if manifest_raw != canonical_json_bytes(manifest) or frozenset(manifest) != _MANIFEST_KEYS:
        raise ActivationRefinalizerError("manifest is not canonical / exact")
    activation_sha = _sha256(manifest_raw)
    complete_raw, complete = _read_source_json(root / COMPLETE_MARKER, label="top COMPLETE")
    if complete_raw != canonical_json_bytes(complete) or complete != {
        "activation_sha256": activation_sha, "schema": ACTIVATION_COMPLETION_SCHEMA, "status": "COMPLETE"
    }:
        raise ActivationRefinalizerError("top COMPLETE is not canonical / does not bind")
    if frozenset(manifest["artifacts"]) != _ARTIFACT_NAMES:
        raise ActivationRefinalizerError("artifacts are not the 12-item allowlist")
    for name, entry in manifest["artifacts"].items():
        if frozenset(entry) != _ENTRY_KEYS:
            raise ActivationRefinalizerError(f"{name} entry keys not exact")
        payload_raw, _ = _read_source_json(Path(entry["path"]), label=f"{name} payload")
        if _sha256(payload_raw) != entry["sha256"]:
            raise ActivationRefinalizerError(f"{name} payload SHA mismatch")
        marker_raw, marker = _read_source_json(Path(entry["complete_path"]), label=f"{name} marker")
        if _sha256(marker_raw) != entry["complete_sha256"]:
            raise ActivationRefinalizerError(f"{name} marker SHA mismatch")
        if Path(entry["complete_path"]).is_relative_to(root):
            if marker_raw != canonical_json_bytes(marker) or marker != {
                "artifact_sha256": entry["sha256"], "schema": ARTIFACT_COMPLETION_SCHEMA, "status": "COMPLETE"
            }:
                raise ActivationRefinalizerError(f"{name} marker not canonical / does not bind")
    return {"status": "VERIFIED", "activation_sha256": activation_sha, "release": str(root)}


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Hardened relocating canonical activation re-finalizer")
    parser.add_argument("--source", required=True)
    parser.add_argument("--releases", required=True)
    parser.add_argument("--owner-authorization", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--refinalizer-code-sha256", required=True)
    parser.add_argument("--refinalizer-code-release", required=True)
    parser.add_argument("--stdlib-verify-only", action="store_true",
                        help="verify with the stdlib preflight instead of importing the loader (for hosts without it)")
    args = parser.parse_args(argv)

    def stdlib_verify(release_dir: Path, _sha: str) -> None:
        verify_relocated_release(release_dir)

    try:
        dest = refinalize_activation_release(
            args.source, args.releases,
            owner_authorization=args.owner_authorization, source_run=args.source_run,
            refinalizer_code_sha256=args.refinalizer_code_sha256,
            refinalizer_code_release=args.refinalizer_code_release,
            consumer_verifier=stdlib_verify if args.stdlib_verify_only else None,
        )
    except ActivationRefinalizerError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    print(json.dumps({
        "ok": True, "release_dir": str(dest), "release_id": dest.name,
        "activation_sha256": _sha256((dest / ACTIVATION_FILE).read_bytes()),
    }))
    return 0


__all__ = [
    "ACTIVATION_FILE", "COMPLETE_MARKER", "ActivationRefinalizerError",
    "canonical_json_bytes", "main", "refinalize_activation_release", "verify_relocated_release",
]


if __name__ == "__main__":
    raise SystemExit(main())
