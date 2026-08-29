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
    ARTIFACT_COMPLETION_SCHEMA,
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
    _strict_json_loads,
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


def _reuse_or_refuse_amended(
    dest, all_files, provenance, final_sha, verify,
    provenance_file: str = AMENDMENT_PROVENANCE_FILE,
) -> Path:
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
        dest / provenance_file
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

    import sys as _sys

    if argv is None:
        argv = _sys.argv[1:]
    box_mode = bool(argv) and argv[0] == "box-budget"
    handeye_mode = bool(argv) and argv[0] == "handeye-swap"
    if box_mode or handeye_mode:
        argv = argv[1:]
    parser = argparse.ArgumentParser(
        description=(
            "Owner-authorized single-field amendment of "
            "back_mask.config.maximum_box_pixels"
            if box_mode
            else "Owner-authorized single-field amendment of "
            "robot_state_bounds.max_staleness_ns"
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--releases", required=True)
    if box_mode:
        parser.add_argument("--expected-previous-px", required=True, type=int)
        parser.add_argument("--amended-px", required=True, type=int)
    elif handeye_mode:
        parser.add_argument("--new-handeye-release", required=True)
        parser.add_argument("--new-handeye-artifact-sha256", required=True)
        parser.add_argument("--new-handeye-complete-sha256", required=True)
    else:
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

    common = dict(
        amendment_reason=args.amendment_reason,
        owner_authorization=args.owner_authorization,
        source_run=args.source_run,
        amender_code_sha256=args.amender_code_sha256,
        amender_code_release=args.amender_code_release,
        consumer_verifier=stdlib_verify if args.stdlib_verify_only else None,
    )
    try:
        if box_mode:
            dest = amend_back_mask_box_budget(
                args.source, args.releases,
                expected_previous_px=args.expected_previous_px,
                amended_px=args.amended_px,
                **common,
            )
        elif handeye_mode:
            dest = amend_handeye_swap(
                args.source, args.releases,
                new_release_dir=args.new_handeye_release,
                new_artifact_sha256=args.new_handeye_artifact_sha256,
                new_complete_sha256=args.new_handeye_complete_sha256,
                **common,
            )
        else:
            dest = amend_activation_staleness_bound(
                args.source, args.releases,
                expected_previous_ns=args.expected_previous_ns,
                amended_ns=args.amended_ns,
                **common,
            )
    except ActivationRefinalizerError as exc:
        print(_json.dumps({"ok": False, "error": str(exc)}))
        return 2
    print(_json.dumps({
        "ok": True, "release_dir": str(dest), "release_id": dest.name,
        "activation_sha256": _sha256((dest / ACTIVATION_FILE).read_bytes()),
    }))
    return 0



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


# --------------------------------------------------------------------------- #
# back_mask work-budget amendment (2026-08-14, owner-authorized)               #
#                                                                              #
# The commissioned back_mask config capped the person-box work budget at       #
# 1024^2 px while the frame budget admits full 1080p.  A patient seated close  #
# to the camera (the REAL auscultation pose) produces a person box above the   #
# cap, so the provider refused every frame (sealed forensics: 550/581 rows     #
# 'person box pixel count exceeds the fixed work budget') and no point could   #
# publish.  A box can never exceed the frame, so raising the box budget up to  #
# the frame budget admits real poses without changing any safety semantics.    #
# --------------------------------------------------------------------------- #

BOX_AMENDMENT_PROVENANCE_FILE = "BOX_BUDGET_AMENDED.json"
BOX_AMENDMENT_PROVENANCE_SCHEMA = (
    "wam_activation_box_budget_amendment_provenance_v1"
)

_BACK_MASK_ARTIFACT = "back_mask"
_BACK_MASK_SCHEMA = "wam_posterior_back_mask_activation_v1"
_BOX_FIELD = "maximum_box_pixels"
_FRAME_FIELD = "maximum_frame_pixels"
_IDENTITY_ARTIFACT = "occlusion_identity"
_IDENTITY_SCHEMAS = (
    "wam_activated_occlusion_identity_v2",
    "wam_activated_occlusion_identity_v1",
)


def amend_back_mask_box_budget(
    source_dir: "str | Path",
    releases_root: "str | Path",
    *,
    expected_previous_px: int,
    amended_px: int,
    amendment_reason: str,
    owner_authorization: str,
    source_run: str,
    amender_code_sha256: str,
    amender_code_release: str,
    consumer_verifier: "ConsumerVerifier | None" = None,
) -> Path:
    """Reissue the activation with ``back_mask.config.maximum_box_pixels``
    amended.

    Mirrors :func:`amend_activation_staleness_bound` exactly, except the
    single amended value lives in the ``back_mask`` artifact payload, so that
    payload is re-canonicalized and its SHA chain (payload, marker, manifest,
    top COMPLETE) is rebound.  Every other artifact is copied byte-for-byte
    and ``robot_state_bounds`` is untouched.  The mandatory consumer
    verification runs the real production loader, which rebuilds the provider
    config and enforces ``1 <= maximum_box_pixels <= maximum_frame_pixels``.
    """

    import dataclasses

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
        ("expected previous box budget", expected_previous_px),
        ("amended box budget", amended_px),
    ):
        if type(value) is not int:
            raise ActivationBoundsAmendmentError(f"{label} must be an int")
    if amended_px == expected_previous_px:
        raise ActivationBoundsAmendmentError(
            "amended box budget equals the previous value; nothing to amend"
        )
    verify: ConsumerVerifier = consumer_verifier or _loader_consumer_verify

    source = _validate_source_path(
        source_dir, label="activation source", directory=True
    )
    releases = Path(releases_root)
    if not releases.is_absolute():
        raise ActivationBoundsAmendmentError("releases root must be absolute")

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

    release_id = "activation-" + _sha256(
        (
            source_activation_sha
            + "|" + _BACK_MASK_ARTIFACT + "." + _BOX_FIELD
            + "=" + str(amended_px)
            + "|" + str(releases.resolve())
        ).encode("utf-8")
    )[:20]
    dest = releases / release_id

    plans = [
        _plan_artifact(name, artifacts[name], new_root=dest)
        for name in sorted(artifacts)
    ]

    # --- Amend the back_mask budget AND the occlusion_identity cross-pin ---- #
    # The loader re-derives the back_mask config's content-addressed identity
    # (semantic sha + release id) and asserts the occlusion_identity artifact
    # records the same values, so both artifacts must move together — the
    # first deployment of this amendment changed only back_mask and the
    # collector refused to start ('activated occlusion identity mismatches
    # back_mask_release_id', 2026-08-14). Identities are derived through the
    # REAL provider config class, never re-implemented here.
    from .tracking.posterior_back_mask_provider import PosteriorBackMaskConfig

    amended_plans = []
    source_back_mask_payload = None
    amended_back_mask_payload = None
    old_config_identity = None
    new_config_identity = None
    identity_plan_index = None
    for plan in plans:
        if plan.name == _IDENTITY_ARTIFACT:
            identity_plan_index = len(amended_plans)
            amended_plans.append(plan)
            continue
        if plan.name != _BACK_MASK_ARTIFACT:
            amended_plans.append(plan)
            continue
        if plan.action != "relocate":
            raise ActivationBoundsAmendmentError(
                "back_mask artifact must be a relocated in-release payload"
            )
        payload = _strict_json_loads(
            plan.payload_bytes, label="back_mask payload"
        )
        if (
            not isinstance(payload, dict)
            or frozenset(payload) != frozenset({"config", "schema"})
            or payload["schema"] != _BACK_MASK_SCHEMA
            or not isinstance(payload["config"], dict)
        ):
            raise ActivationBoundsAmendmentError(
                "back_mask payload shape is not the activation schema"
            )
        config = payload["config"]
        if (
            type(config.get(_BOX_FIELD)) is not int
            or type(config.get(_FRAME_FIELD)) is not int
        ):
            raise ActivationBoundsAmendmentError(
                "back_mask budgets are missing or not ints"
            )
        if config[_BOX_FIELD] != expected_previous_px:
            raise ActivationBoundsAmendmentError(
                "source box budget does not match the expected previous value"
            )
        if not 1 <= amended_px <= config[_FRAME_FIELD]:
            raise ActivationBoundsAmendmentError(
                "amended box budget must stay within [1, maximum_frame_pixels]"
            )
        new_config = dict(config)
        new_config[_BOX_FIELD] = amended_px
        try:
            old_config_identity = PosteriorBackMaskConfig.create(**config)
            new_config_identity = PosteriorBackMaskConfig.create(**new_config)
        except Exception as exc:
            raise ActivationBoundsAmendmentError(
                f"back_mask config does not build through the real provider: {exc}"
            ) from exc
        new_payload = {"config": new_config, "schema": _BACK_MASK_SCHEMA}
        new_payload_bytes = canonical_json_bytes(new_payload)
        new_payload_sha = _sha256(new_payload_bytes)
        new_marker_bytes = canonical_json_bytes({
            "artifact_sha256": new_payload_sha,
            "schema": ARTIFACT_COMPLETION_SCHEMA,
            "status": "COMPLETE",
        })
        new_entry = dict(plan.entry)
        new_entry["sha256"] = new_payload_sha
        new_entry["complete_sha256"] = _sha256(new_marker_bytes)
        amended_plans.append(dataclasses.replace(
            plan,
            entry=new_entry,
            payload_bytes=new_payload_bytes,
            marker_bytes=new_marker_bytes,
        ))
        source_back_mask_payload = payload
        amended_back_mask_payload = new_payload
    if amended_back_mask_payload is None:
        raise ActivationBoundsAmendmentError(
            "back_mask artifact is missing from the activation"
        )
    if identity_plan_index is None:
        raise ActivationBoundsAmendmentError(
            "occlusion_identity artifact is missing from the activation"
        )

    identity_plan = amended_plans[identity_plan_index]
    if identity_plan.action != "relocate":
        raise ActivationBoundsAmendmentError(
            "occlusion_identity artifact must be a relocated in-release payload"
        )
    identity_payload = _strict_json_loads(
        identity_plan.payload_bytes, label="occlusion_identity payload"
    )
    if (
        not isinstance(identity_payload, dict)
        or identity_payload.get("schema") not in _IDENTITY_SCHEMAS
    ):
        raise ActivationBoundsAmendmentError(
            "occlusion_identity payload schema is unsupported"
        )
    if (
        identity_payload.get("back_mask_release_id")
        != old_config_identity.release_id
        or identity_payload.get("back_mask_sha256")
        != old_config_identity.semantic_sha256
    ):
        raise ActivationBoundsAmendmentError(
            "occlusion_identity does not record the source back_mask identity"
        )
    new_identity_payload = dict(identity_payload)
    new_identity_payload["back_mask_release_id"] = (
        new_config_identity.release_id
    )
    new_identity_payload["back_mask_sha256"] = (
        new_config_identity.semantic_sha256
    )
    new_identity_bytes = canonical_json_bytes(new_identity_payload)
    new_identity_sha = _sha256(new_identity_bytes)
    new_identity_marker = canonical_json_bytes({
        "artifact_sha256": new_identity_sha,
        "schema": ARTIFACT_COMPLETION_SCHEMA,
        "status": "COMPLETE",
    })
    new_identity_entry = dict(identity_plan.entry)
    new_identity_entry["sha256"] = new_identity_sha
    new_identity_entry["complete_sha256"] = _sha256(new_identity_marker)
    amended_plans[identity_plan_index] = dataclasses.replace(
        identity_plan,
        entry=new_identity_entry,
        payload_bytes=new_identity_bytes,
        marker_bytes=new_identity_marker,
    )
    source_identity_payload = identity_payload
    amended_identity_payload = new_identity_payload
    plans = amended_plans

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

    new_manifest = dict(manifest)
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
    _assert_box_amended_equivalence(
        manifest,
        new_manifest,
        plans,
        source_back_mask_payload,
        amended_back_mask_payload,
        source_identity_payload,
        amended_identity_payload,
    )

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
        "schema": BOX_AMENDMENT_PROVENANCE_SCHEMA,
        "source_activation_sha256": source_activation_sha,
        "final_activation_sha256": final_activation_sha,
        "amended_field": (
            "artifacts." + _BACK_MASK_ARTIFACT + ".config." + _BOX_FIELD
        ),
        "previous_value_px": expected_previous_px,
        "amended_value_px": amended_px,
        "cross_pin_updates": {
            "artifacts.occlusion_identity.back_mask_release_id": {
                "previous": old_config_identity.release_id,
                "amended": new_config_identity.release_id,
            },
            "artifacts.occlusion_identity.back_mask_sha256": {
                "previous": old_config_identity.semantic_sha256,
                "amended": new_config_identity.semantic_sha256,
            },
        },
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
        BOX_AMENDMENT_PROVENANCE_FILE: prov_bytes,
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
            provenance_file=BOX_AMENDMENT_PROVENANCE_FILE,
        )

    _materialize(releases, dest, all_files, final_activation_sha, verify)
    return dest


def _assert_box_amended_equivalence(
    source: Mapping,
    new: Mapping,
    plans,
    source_payload: Mapping,
    new_payload: Mapping,
    source_identity: Mapping,
    new_identity: Mapping,
) -> None:
    """Everything except the amended budget and its cross-pin must be equal."""

    for key in _MANIFEST_KEYS - {"artifacts", "openarm_pair"}:
        if source[key] != new[key]:
            raise ActivationBoundsAmendmentError(
                f"manifest field {key} changed unexpectedly"
            )
    for p in plans:
        s, n = source["artifacts"][p.name], new["artifacts"][p.name]
        if p.name in (_BACK_MASK_ARTIFACT, _IDENTITY_ARTIFACT):
            if s["sha256"] == n["sha256"]:
                raise ActivationBoundsAmendmentError(
                    f"{p.name} payload SHA did not change"
                )
            continue
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
    source_config = dict(source_payload["config"])
    new_config = dict(new_payload["config"])
    if source_config.pop(_BOX_FIELD) == new_config.pop(_BOX_FIELD):
        raise ActivationBoundsAmendmentError("amended budget did not change")
    if source_config != new_config:
        raise ActivationBoundsAmendmentError(
            "back_mask config changed beyond the amended budget"
        )
    src_ident = dict(source_identity)
    new_ident = dict(new_identity)
    for field in ("back_mask_release_id", "back_mask_sha256"):
        if src_ident.pop(field) == new_ident.pop(field):
            raise ActivationBoundsAmendmentError(
                f"occlusion_identity {field} did not change"
            )
    if src_ident != new_ident:
        raise ActivationBoundsAmendmentError(
            "occlusion_identity changed beyond the back_mask cross-pin"
        )




# --------------------------------------------------------------------------- #
# hand-eye swap amendment (graduation day, prepared 2026-08-29)                #
#                                                                              #
# Graduating the R4 provisional hand-eye to the validated tier produces a NEW  #
# external hand-eye release. The activation pins it twice: the manifest's      #
# external ``handeye`` artifact entry, and FIVE derived fields inside the      #
# ``occlusion_identity`` artifact (release id, artifact sha, capability        #
# schema, calibration tier, policy sha). Both must move together — the same    #
# cross-pin class that crash-looped the first box-budget deployment.  Every    #
# derived value comes from the REAL loader (load_completed_handeye_release),   #
# never re-implemented here.                                                   #
# --------------------------------------------------------------------------- #

HANDEYE_AMENDMENT_PROVENANCE_FILE = "HANDEYE_SWAPPED.json"
HANDEYE_AMENDMENT_PROVENANCE_SCHEMA = (
    "wam_activation_handeye_swap_provenance_v1"
)

_HANDEYE_ARTIFACT = "handeye"
_IDENTITY_HANDEYE_FIELDS = (
    "handeye_release_id",
    "handeye_sha256",
    "handeye_capability_schema",
    "handeye_calibration_tier",
    "handeye_policy_sha256",
)


def _derive_handeye_identity(release_dir, artifact_sha256, complete_sha256,
                             camera_id):
    """Derive the five identity pins through the real production loader."""
    from .production_runtime_activation import (
        load_completed_handeye_release,
    )

    handeye = load_completed_handeye_release(
        release_dir,
        expected_artifact_sha256=artifact_sha256,
        expected_complete_sha256=complete_sha256,
        expected_camera_id=camera_id,
    )
    return {
        "handeye_release_id": handeye.handeye_release_id,
        "handeye_sha256": handeye.handeye_sha256,
        "handeye_capability_schema": handeye.capability_schema,
        "handeye_calibration_tier": handeye.calibration_tier,
        "handeye_policy_sha256": handeye.policy_sha256,
    }


def amend_handeye_swap(
    source_dir: "str | Path",
    releases_root: "str | Path",
    *,
    new_release_dir: "str | Path",
    new_artifact_sha256: str,
    new_complete_sha256: str,
    amendment_reason: str,
    owner_authorization: str,
    source_run: str,
    amender_code_sha256: str,
    amender_code_release: str,
    consumer_verifier: "ConsumerVerifier | None" = None,
) -> Path:
    """Reissue the activation with the external hand-eye release swapped."""

    for label, value in (
        ("amendment reason", amendment_reason),
        ("owner authorization", owner_authorization),
        ("source-run identity", source_run),
        ("amender code release", amender_code_release),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ActivationBoundsAmendmentError(f"{label} is required")
    _require_sha256(amender_code_sha256, label="amender code SHA")
    new_artifact_sha256 = _require_sha256(
        new_artifact_sha256, label="new hand-eye artifact SHA"
    )
    new_complete_sha256 = _require_sha256(
        new_complete_sha256, label="new hand-eye COMPLETE SHA"
    )
    new_release = _validate_source_path(
        new_release_dir, label="new hand-eye release", directory=True,
    )
    verify: ConsumerVerifier = consumer_verifier or _loader_consumer_verify

    source = _validate_source_path(
        source_dir, label="activation source", directory=True
    )
    releases = Path(releases_root)
    if not releases.is_absolute():
        raise ActivationBoundsAmendmentError("releases root must be absolute")

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

    release_id = "activation-" + _sha256(
        (
            source_activation_sha
            + "|handeye=" + new_artifact_sha256
            + "|" + str(releases.resolve())
        ).encode("utf-8")
    )[:20]
    dest = releases / release_id

    plans = [
        _plan_artifact(name, artifacts[name], new_root=dest)
        for name in sorted(artifacts)
    ]

    # camera identity for the loader derivation comes from the identity
    # artifact itself (it is the authority the loader compares against).
    identity_plan_index = None
    handeye_plan_index = None
    for index, plan in enumerate(plans):
        if plan.name == _IDENTITY_ARTIFACT:
            identity_plan_index = index
        elif plan.name == _HANDEYE_ARTIFACT:
            handeye_plan_index = index
    if identity_plan_index is None or handeye_plan_index is None:
        raise ActivationBoundsAmendmentError(
            "activation is missing the hand-eye or occlusion_identity artifact"
        )
    identity_plan = plans[identity_plan_index]
    if identity_plan.action != "relocate":
        raise ActivationBoundsAmendmentError(
            "occlusion_identity artifact must be a relocated in-release payload"
        )
    identity_payload = _strict_json_loads(
        identity_plan.payload_bytes, label="occlusion_identity payload"
    )
    if (
        not isinstance(identity_payload, dict)
        or identity_payload.get("schema") not in _IDENTITY_SCHEMAS
    ):
        raise ActivationBoundsAmendmentError(
            "occlusion_identity payload schema is unsupported"
        )
    camera_id = identity_payload.get("camera_id")
    if not isinstance(camera_id, str) or not camera_id:
        raise ActivationBoundsAmendmentError(
            "occlusion_identity camera identity is missing"
        )

    import dataclasses

    old_entry = dict(plans[handeye_plan_index].entry)
    old_derived = _derive_handeye_identity(
        Path(old_entry["path"]).parent,
        old_entry["sha256"],
        old_entry["complete_sha256"],
        camera_id,
    )
    new_derived = _derive_handeye_identity(
        new_release, new_artifact_sha256, new_complete_sha256, camera_id,
    )
    if new_derived["handeye_sha256"] == old_derived["handeye_sha256"]:
        raise ActivationBoundsAmendmentError(
            "new hand-eye release is identical to the source; nothing to swap"
        )
    for field in _IDENTITY_HANDEYE_FIELDS:
        if identity_payload.get(field) != old_derived[field]:
            raise ActivationBoundsAmendmentError(
                "occlusion_identity does not record the source hand-eye "
                f"identity ({field})"
            )

    new_handeye_entry = {
        "path": str(new_release / "handeye.json"),
        "complete_path": str(new_release / "handeye.COMPLETE"),
        "sha256": new_artifact_sha256,
        "complete_sha256": new_complete_sha256,
    }
    plans[handeye_plan_index] = dataclasses.replace(
        plans[handeye_plan_index], entry=new_handeye_entry,
    )

    new_identity_payload = dict(identity_payload)
    for field in _IDENTITY_HANDEYE_FIELDS:
        new_identity_payload[field] = new_derived[field]
    new_identity_bytes = canonical_json_bytes(new_identity_payload)
    new_identity_sha = _sha256(new_identity_bytes)
    new_identity_marker = canonical_json_bytes({
        "artifact_sha256": new_identity_sha,
        "schema": ARTIFACT_COMPLETION_SCHEMA,
        "status": "COMPLETE",
    })
    new_identity_entry = dict(identity_plan.entry)
    new_identity_entry["sha256"] = new_identity_sha
    new_identity_entry["complete_sha256"] = _sha256(new_identity_marker)
    plans[identity_plan_index] = dataclasses.replace(
        identity_plan,
        entry=new_identity_entry,
        payload_bytes=new_identity_bytes,
        marker_bytes=new_identity_marker,
    )

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

    new_manifest = dict(manifest)
    new_manifest["artifacts"] = {p.name: p.entry for p in plans}
    new_manifest["openarm_pair"] = new_pair_entry
    new_manifest_bytes = canonical_json_bytes(new_manifest)
    final_activation_sha = _sha256(new_manifest_bytes)
    new_complete_bytes = canonical_json_bytes({
        "activation_sha256": final_activation_sha,
        "schema": ACTIVATION_COMPLETION_SCHEMA,
        "status": "COMPLETE",
    })

    for key in _MANIFEST_KEYS - {"artifacts", "openarm_pair"}:
        if manifest[key] != new_manifest[key]:
            raise ActivationBoundsAmendmentError(
                f"manifest field {key} changed unexpectedly"
            )
    for p in plans:
        if p.name in (_HANDEYE_ARTIFACT, _IDENTITY_ARTIFACT):
            continue
        if manifest["artifacts"][p.name]["sha256"] != new_manifest["artifacts"][p.name]["sha256"]:
            raise ActivationBoundsAmendmentError(
                f"{p.name} payload SHA must not change"
            )

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
        "schema": HANDEYE_AMENDMENT_PROVENANCE_SCHEMA,
        "source_activation_sha256": source_activation_sha,
        "final_activation_sha256": final_activation_sha,
        "amended_field": "artifacts.handeye (external release swap)",
        "previous_entry": old_entry,
        "amended_entry": new_handeye_entry,
        "cross_pin_updates": {
            f"artifacts.occlusion_identity.{field}": {
                "previous": old_derived[field],
                "amended": new_derived[field],
            }
            for field in _IDENTITY_HANDEYE_FIELDS
        },
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
        HANDEYE_AMENDMENT_PROVENANCE_FILE: prov_bytes,
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
            provenance_file=HANDEYE_AMENDMENT_PROVENANCE_FILE,
        )
    _materialize(releases, dest, all_files, final_activation_sha, verify)
    return dest


if __name__ == "__main__":
    raise SystemExit(main())
