"""Owner-authorized single-field amendment of a production activation package.

The activation's ``robot_state_bounds.max_staleness_ns`` was commissioned at
50 ms against a fast-inference performance envelope.  The deployed box brackets
recorded frames through a Python evidence pipeline whose measured floor
(2026-08-12: bracket 1 ms + bit-identical-optimized mask bundle ~60 ms) sits
above that bound, so every armed session died on its first frames.  The bound
is a bracketing-timeliness parameter: raising it creates no telemetry gap and
changes no recorded value — brackets are computed from recorded envelopes with
exact timestamps regardless of when the query runs.

This tool reissues the activation with exactly ONE amended field, following the
hardened pattern of :mod:`ausc.perception.activation_refinalizer` (whose
machinery it reuses):

  * the source release is read once and fully verified (strict JSON, exact key
    sets, marker chain, per-artifact SHA binding);
  * every artifact payload and the pair sub-package are copied byte-for-byte;
    ONLY ``robot_state_bounds.max_staleness_ns`` changes, and only from the
    expected previous value to the requested one within [50 ms, 500 ms];
  * the canonical chain (activation manifest, top COMPLETE marker, artifact
    markers) is rebound; the release id is keyed off the source SHA, the
    amended value, and the releases root;
  * a MANDATORY consumer verification runs the real production loader against
    the staged release before it can be published;
  * provenance binds the source SHA, the final SHA, the amended field with its
    previous and new values, the amendment reason, the owner authorization,
    and the full release inventory.

It never mutates the source or any external input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .activation_refinalizer import (
    ACTIVATION_FILE,
    ACTIVATION_COMPLETION_SCHEMA,
    ACTIVATION_SCHEMA,
    COMPLETE_MARKER,
    ActivationRefinalizerError,
    ConsumerVerifier,
    _ARTIFACT_NAMES,
    _BOUNDS_KEYS,
    _COMPLETE_KEYS,
    _MANIFEST_KEYS,
    _PAIR_KEYS,
    _loader_consumer_verify,
    _materialize,
    _plan_artifact,
    _read_pair_tree,
    _read_source_json,
    _require_sha256,
    _sha256,
    _validate_source_path,
    canonical_json_bytes,
)

AMENDMENT_PROVENANCE_FILE = "BOUNDS_AMENDED.json"
AMENDMENT_PROVENANCE_SCHEMA = "wam_activation_bounds_amendment_provenance_v1"

_AMENDABLE_FIELD = "max_staleness_ns"
_MIN_STALENESS_NS = 50_000_000
_MAX_STALENESS_NS = 500_000_000


class ActivationBoundsAmendmentError(ActivationRefinalizerError):
    """The activation bounds amendment cannot produce a trustworthy release."""


def amend_activation_staleness_bound(
    source_dir: "str | Path",
    releases_root: "str | Path",
    *,
    expected_previous_ns: int,
    amended_ns: int,
    amendment_reason: str,
    owner_authorization: str,
    source_run: str,
    amender_code_sha256: str,
    amender_code_release: str,
    consumer_verifier: "ConsumerVerifier | None" = None,
) -> Path:
    """Reissue the activation with ``max_staleness_ns`` amended.

    Returns the new immutable release directory.  Idempotent only after a full
    re-verification of an existing identical release; a same-id release with
    different content is refused.
    """

    for label, value in (
        ("amendment reason", amendment_reason),
        ("owner authorization", owner_authorization),
        ("source-run identity", source_run),
        ("amender code release", amender_code_release),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ActivationBoundsAmendmentError(f"{label} is required")
    _require_sha256(amender_code_sha256, label="amender code SHA")
    for label, value in (
        ("expected previous staleness", expected_previous_ns),
        ("amended staleness", amended_ns),
    ):
        if type(value) is not int:
            raise ActivationBoundsAmendmentError(f"{label} must be an int")
    if not _MIN_STALENESS_NS <= amended_ns <= _MAX_STALENESS_NS:
        raise ActivationBoundsAmendmentError(
            "amended staleness must stay within [50 ms, 500 ms]"
        )
    if amended_ns == expected_previous_ns:
        raise ActivationBoundsAmendmentError(
            "amended staleness equals the previous value; nothing to amend"
        )
    verify: ConsumerVerifier = consumer_verifier or _loader_consumer_verify

    source = _validate_source_path(
        source_dir, label="activation source", directory=True
    )
    releases = Path(releases_root)
    if not releases.is_absolute():
        raise ActivationBoundsAmendmentError("releases root must be absolute")

    # --- Read + validate the source (single-FD, strict; refinalizer rules) --- #
    manifest_raw, manifest = _read_source_json(
        source / ACTIVATION_FILE, label="activation manifest"
    )
    if not isinstance(manifest, dict) or frozenset(manifest) != _MANIFEST_KEYS:
        raise ActivationBoundsAmendmentError("activation manifest keys are not exact")
    if (
        manifest["schema"] != ACTIVATION_SCHEMA
        or manifest["status"] != "COMPLETE"
        or manifest["side"] != "left"
        or manifest["leader_can_interface"] != "can1"
        or manifest["follower_can_interface"] != "can3"
    ):
        raise ActivationBoundsAmendmentError(
            "activation must be COMPLETE left can1 leader -> can3 follower"
        )
    bounds = manifest["robot_state_bounds"]
    if not isinstance(bounds, dict) or frozenset(bounds) != _BOUNDS_KEYS:
        raise ActivationBoundsAmendmentError("robot_state_bounds keys are not exact")
    if bounds[_AMENDABLE_FIELD] != expected_previous_ns:
        raise ActivationBoundsAmendmentError(
            "source staleness bound does not match the expected previous value"
        )
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or frozenset(artifacts) != _ARTIFACT_NAMES:
        raise ActivationBoundsAmendmentError(
            "activation artifacts are not exactly the 12-item allowlist"
        )
    source_activation_sha = _sha256(manifest_raw)

    complete_raw, complete = _read_source_json(
        source / COMPLETE_MARKER, label="top COMPLETE marker"
    )
    if not isinstance(complete, dict) or frozenset(complete) != _COMPLETE_KEYS:
        raise ActivationBoundsAmendmentError("top COMPLETE marker keys are not exact")
    if (
        complete["activation_sha256"] != source_activation_sha
        or complete["schema"] != ACTIVATION_COMPLETION_SCHEMA
        or complete["status"] != "COMPLETE"
    ):
        raise ActivationBoundsAmendmentError(
            "top COMPLETE marker does not bind the source manifest"
        )

    # Release identity keyed off (source sha, amended value, releases root):
    # the same id can never map to two different final activation SHAs.
    release_id = "activation-" + _sha256(
        (
            source_activation_sha
            + "|" + _AMENDABLE_FIELD + "=" + str(amended_ns)
            + "|" + str(releases.resolve())
        ).encode("utf-8")
    )[:20]
    dest = releases / release_id

    plans = [
        _plan_artifact(name, artifacts[name], new_root=dest)
        for name in sorted(artifacts)
    ]

    pair_entry = manifest["openarm_pair"]
    if not isinstance(pair_entry, dict) or frozenset(pair_entry) != _PAIR_KEYS:
        raise ActivationBoundsAmendmentError("openarm_pair keys are not exact")
    _require_sha256(pair_entry["contract_sha256"], label="pair contract SHA")
    pair_src = _validate_source_path(
        pair_entry["release_directory"], label="openarm pair release", directory=True,
    )
    pair_rel = f"pairs/{pair_src.name}"
    pair_files = _read_pair_tree(pair_src)
    new_pair_entry = {
        "contract_sha256": pair_entry["contract_sha256"],
        "release_directory": str(dest / pair_rel),
    }

    # --- Build amended manifest + COMPLETE + provenance ---------------------- #
    new_bounds = dict(bounds)
    new_bounds[_AMENDABLE_FIELD] = amended_ns
    new_manifest = dict(manifest)
    new_manifest["robot_state_bounds"] = new_bounds
    new_manifest["artifacts"] = {p.name: p.entry for p in plans}
    new_manifest["openarm_pair"] = new_pair_entry
    new_manifest_bytes = canonical_json_bytes(new_manifest)
    final_activation_sha = _sha256(new_manifest_bytes)
    new_complete_bytes = canonical_json_bytes(
        {
            "activation_sha256": final_activation_sha,
            "schema": ACTIVATION_COMPLETION_SCHEMA,
            "status": "COMPLETE",
        }
    )
    _assert_amended_equivalence(manifest, new_manifest, plans)

    inventory = {
        ACTIVATION_FILE: final_activation_sha,
        COMPLETE_MARKER: _sha256(new_complete_bytes),
    }
    for p in plans:
        if p.action == "relocate":
            inventory[p.payload_rel] = p.entry["sha256"]
            inventory[p.marker_rel] = p.entry["complete_sha256"]
    for rel, data in pair_files:
        inventory[f"{pair_rel}/{rel}"] = _sha256(data)

    provenance = {
        "schema": AMENDMENT_PROVENANCE_SCHEMA,
        "source_activation_sha256": source_activation_sha,
        "final_activation_sha256": final_activation_sha,
        "amended_field": "robot_state_bounds." + _AMENDABLE_FIELD,
        "previous_value_ns": expected_previous_ns,
        "amended_value_ns": amended_ns,
        "amendment_reason": amendment_reason,
        "amender_code_sha256": amender_code_sha256,
        "amender_code_release": amender_code_release,
        "owner_authorization": owner_authorization,
        "source_run": source_run,
        "release_id": release_id,
        "release_directory": str(dest),
        "kept_external_artifacts": sorted(
            p.name for p in plans if p.action == "keep"
        ),
        "relocated_artifacts": sorted(
            p.name for p in plans if p.action == "relocate"
        ),
        "inventory": inventory,
    }
    prov_bytes = canonical_json_bytes(provenance)

    all_files = {
        ACTIVATION_FILE: new_manifest_bytes,
        COMPLETE_MARKER: new_complete_bytes,
        AMENDMENT_PROVENANCE_FILE: prov_bytes,
    }
    for p in plans:
        if p.action == "relocate":
            all_files[p.payload_rel] = p.payload_bytes
            all_files[p.marker_rel] = p.marker_bytes
    for rel, data in pair_files:
        all_files[f"{pair_rel}/{rel}"] = data

    if dest.exists():
        return _reuse_or_refuse_amended(
            dest, all_files, provenance, final_activation_sha, verify,
        )

    _materialize(releases, dest, all_files, final_activation_sha, verify)
    return dest


def _reuse_or_refuse_amended(dest, all_files, provenance, final_sha, verify) -> Path:
    """Idempotent reuse mirror of the refinalizer's, keyed to THIS tool's
    provenance file (which is already part of ``all_files``)."""

    dest = Path(dest)
    if dest.is_symlink() or not dest.is_dir():
        raise ActivationBoundsAmendmentError(
            "existing activation release path is not a directory"
        )
    expected = set(all_files)
    on_disk = {
        p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()
    }
    if on_disk != expected:
        raise ActivationBoundsAmendmentError(
            f"existing release {dest.name} inventory differs (missing/extra files)"
        )
    for rel, data in all_files.items():
        f = dest / rel
        if f.is_symlink() or not f.is_file() or f.read_bytes() != data:
            raise ActivationBoundsAmendmentError(
                f"existing release {dest.name} differs at {rel}"
            )
    if (
        dest / AMENDMENT_PROVENANCE_FILE
    ).read_bytes() != canonical_json_bytes(provenance):
        raise ActivationBoundsAmendmentError(
            f"existing release {dest.name} provenance differs"
        )
    verify(dest, final_sha)
    return dest


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    import json as _json

    from .activation_refinalizer import verify_relocated_release

    parser = argparse.ArgumentParser(
        description=(
            "Owner-authorized single-field amendment of "
            "robot_state_bounds.max_staleness_ns"
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--releases", required=True)
    parser.add_argument("--expected-previous-ns", required=True, type=int)
    parser.add_argument("--amended-ns", required=True, type=int)
    parser.add_argument("--amendment-reason", required=True)
    parser.add_argument("--owner-authorization", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--amender-code-sha256", required=True)
    parser.add_argument("--amender-code-release", required=True)
    parser.add_argument(
        "--stdlib-verify-only", action="store_true",
        help="verify with the stdlib preflight instead of importing the loader",
    )
    args = parser.parse_args(argv)

    def stdlib_verify(release_dir: Path, _sha: str) -> None:
        verify_relocated_release(release_dir)

    try:
        dest = amend_activation_staleness_bound(
            args.source, args.releases,
            expected_previous_ns=args.expected_previous_ns,
            amended_ns=args.amended_ns,
            amendment_reason=args.amendment_reason,
            owner_authorization=args.owner_authorization,
            source_run=args.source_run,
            amender_code_sha256=args.amender_code_sha256,
            amender_code_release=args.amender_code_release,
            consumer_verifier=stdlib_verify if args.stdlib_verify_only else None,
        )
    except ActivationRefinalizerError as exc:
        print(_json.dumps({"ok": False, "error": str(exc)}))
        return 2
    print(_json.dumps({
        "ok": True, "release_dir": str(dest), "release_id": dest.name,
        "activation_sha256": _sha256((dest / ACTIVATION_FILE).read_bytes()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _assert_amended_equivalence(
    source: Mapping, new: Mapping, plans,
) -> None:
    """Everything except the single amended bound must be semantically equal."""

    for key in _MANIFEST_KEYS - {"artifacts", "openarm_pair", "robot_state_bounds"}:
        if source[key] != new[key]:
            raise ActivationBoundsAmendmentError(
                f"manifest field {key} changed unexpectedly"
            )
    source_bounds = source["robot_state_bounds"]
    new_bounds = new["robot_state_bounds"]
    for key in _BOUNDS_KEYS - {_AMENDABLE_FIELD}:
        if source_bounds[key] != new_bounds[key]:
            raise ActivationBoundsAmendmentError(
                f"bounds field {key} changed unexpectedly"
            )
    if new_bounds[_AMENDABLE_FIELD] == source_bounds[_AMENDABLE_FIELD]:
        raise ActivationBoundsAmendmentError(
            "amended bound did not change"
        )
    for p in plans:
        s, n = source["artifacts"][p.name], new["artifacts"][p.name]
        if s["sha256"] != n["sha256"]:
            raise ActivationBoundsAmendmentError(
                f"{p.name} payload SHA must not change"
            )
        if p.action == "keep" and s != n:
            raise ActivationBoundsAmendmentError(
                f"kept artifact {p.name} entry must be identical"
            )
    if source["openarm_pair"]["contract_sha256"] != new["openarm_pair"]["contract_sha256"]:
        raise ActivationBoundsAmendmentError(
            "pair contract identity must not change"
        )
