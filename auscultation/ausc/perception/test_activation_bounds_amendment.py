"""Tests for the owner-authorized staleness-bound activation amendment.

Reuses the refinalizer test fixture (full 12-artifact package, handeye
external-canonical) and its stdlib consumer verifier; the on-box run uses the
real production loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ausc.perception.activation_bounds_amendment import (
    AMENDMENT_PROVENANCE_FILE,
    AMENDMENT_PROVENANCE_SCHEMA,
    ActivationBoundsAmendmentError,
    amend_activation_staleness_bound,
)
from ausc.perception.activation_refinalizer import (
    canonical_json_bytes,
    verify_relocated_release,
)
from ausc.perception.test_activation_refinalizer import _build_source

_AUTH = dict(
    amendment_reason=(
        "recorded-frame bracketing floor ~60 ms exceeds the commissioned "
        "50 ms envelope on the deployed box"
    ),
    owner_authorization="op:qz",
    source_run="run-2026-08-12",
    amender_code_sha256="a" * 64,
    amender_code_release="code_release-" + "b" * 20,
)

# The shared fixture commissions max_staleness_ns at 200 ms.
_FIXTURE_PREVIOUS_NS = 200_000_000
_AMENDED_NS = 300_000_000


def _verify(release_dir, _sha256):
    verify_relocated_release(release_dir)


def _amend(src, releases, **over):
    kw = dict(
        _AUTH,
        expected_previous_ns=_FIXTURE_PREVIOUS_NS,
        amended_ns=_AMENDED_NS,
        consumer_verifier=_verify,
    )
    kw.update(over)
    return amend_activation_staleness_bound(src, releases, **kw)


def test_amends_exactly_one_bounds_field(tmp_path):
    src = _build_source(tmp_path)
    dest = _amend(src, tmp_path / "releases")
    verify_relocated_release(dest)

    source_manifest = json.loads((src / "activation.json").read_text())
    amended_manifest = json.loads((dest / "activation.json").read_text())
    assert (
        amended_manifest["robot_state_bounds"]["max_staleness_ns"] == _AMENDED_NS
    )
    for key, value in source_manifest["robot_state_bounds"].items():
        if key == "max_staleness_ns":
            continue
        assert amended_manifest["robot_state_bounds"][key] == value
    for key in ("schema", "status", "side", "leader_can_interface",
                "follower_can_interface"):
        assert amended_manifest[key] == source_manifest[key]
    # Artifact payload identities never change.
    for name, entry in source_manifest["artifacts"].items():
        assert amended_manifest["artifacts"][name]["sha256"] == entry["sha256"]


def test_artifact_payload_bytes_are_copied_verbatim(tmp_path):
    src = _build_source(tmp_path)
    dest = _amend(src, tmp_path / "releases")
    source_manifest = json.loads((src / "activation.json").read_text())
    amended_manifest = json.loads((dest / "activation.json").read_text())
    for name, entry in amended_manifest["artifacts"].items():
        if name == "handeye":
            # stays external, entry identical
            assert entry == source_manifest["artifacts"][name]
            continue
        source_bytes = Path(source_manifest["artifacts"][name]["path"]).read_bytes()
        assert Path(entry["path"]).read_bytes() == source_bytes


def test_provenance_binds_amendment_identity(tmp_path):
    src = _build_source(tmp_path)
    dest = _amend(src, tmp_path / "releases")
    provenance = json.loads((dest / AMENDMENT_PROVENANCE_FILE).read_text())
    assert provenance["schema"] == AMENDMENT_PROVENANCE_SCHEMA
    assert provenance["amended_field"] == "robot_state_bounds.max_staleness_ns"
    assert provenance["previous_value_ns"] == _FIXTURE_PREVIOUS_NS
    assert provenance["amended_value_ns"] == _AMENDED_NS
    assert provenance["owner_authorization"] == "op:qz"
    manifest_raw = (dest / "activation.json").read_bytes()
    import hashlib
    assert provenance["final_activation_sha256"] == hashlib.sha256(
        manifest_raw
    ).hexdigest()
    # Canonical provenance encoding.
    assert (dest / AMENDMENT_PROVENANCE_FILE).read_bytes() == (
        canonical_json_bytes(provenance)
    )


def test_idempotent_reuse_and_conflicting_content_refused(tmp_path):
    src = _build_source(tmp_path)
    releases = tmp_path / "releases"
    first = _amend(src, releases)
    again = _amend(src, releases)
    assert first == again
    # A different amended value maps to a DIFFERENT release id, never reuse.
    other = _amend(src, releases, amended_ns=250_000_000)
    assert other != first


def test_wrong_expected_previous_value_refused(tmp_path):
    src = _build_source(tmp_path)
    with pytest.raises(
        ActivationBoundsAmendmentError, match="expected previous value",
    ):
        _amend(src, tmp_path / "releases", expected_previous_ns=50_000_000)


def test_out_of_range_amendment_refused(tmp_path):
    src = _build_source(tmp_path)
    with pytest.raises(ActivationBoundsAmendmentError, match="within"):
        _amend(src, tmp_path / "releases", amended_ns=600_000_000)
    with pytest.raises(ActivationBoundsAmendmentError, match="within"):
        _amend(src, tmp_path / "releases", amended_ns=10_000_000)


def test_noop_amendment_refused(tmp_path):
    src = _build_source(tmp_path)
    with pytest.raises(ActivationBoundsAmendmentError, match="nothing to amend"):
        _amend(src, tmp_path / "releases", amended_ns=_FIXTURE_PREVIOUS_NS)


def test_missing_authorization_refused(tmp_path):
    src = _build_source(tmp_path)
    with pytest.raises(
        ActivationBoundsAmendmentError, match="owner authorization",
    ):
        _amend(src, tmp_path / "releases", owner_authorization="  ")


def test_failed_consumer_verification_destroys_release(tmp_path):
    src = _build_source(tmp_path)

    def failing_verify(release_dir, sha):
        raise RuntimeError("loader refused")

    with pytest.raises(RuntimeError, match="loader refused"):
        _amend(src, tmp_path / "releases", consumer_verifier=failing_verify)
    remaining = list((tmp_path / "releases").glob("activation-*"))
    assert remaining == []


def test_source_never_mutated(tmp_path):
    src = _build_source(tmp_path)
    before = {
        p.relative_to(src).as_posix(): p.read_bytes()
        for p in src.rglob("*") if p.is_file()
    }
    _amend(src, tmp_path / "releases")
    after = {
        p.relative_to(src).as_posix(): p.read_bytes()
        for p in src.rglob("*") if p.is_file()
    }
    assert before == after


# --------------------------------------------------------------------------- #
# back_mask box-budget amendment                                               #
# --------------------------------------------------------------------------- #

from ausc.perception.activation_bounds_amendment import (  # noqa: E402
    BOX_AMENDMENT_PROVENANCE_FILE,
    BOX_AMENDMENT_PROVENANCE_SCHEMA,
    amend_back_mask_box_budget,
)

_REAL_BACK_MASK_PAYLOAD = {
    "config": {
        "central_seed_fraction": 0.4,
        "depth_band_m": 0.08,
        "mad_scale": 3.0,
        "maximum_adaptive_depth_band_m": 0.18,
        "maximum_box_pixels": 1048576,
        "maximum_connected_components": 1024,
        "maximum_depth_m": 3.0,
        "maximum_frame_pixels": 2097152,
        "minimum_component_pixels": 512,
        "minimum_depth_m": 0.25,
        "minimum_valid_depth_fraction": 0.15,
        "minimum_valid_depth_pixels": 256,
        "morphology_radius_px": 2,
    },
    "schema": "wam_posterior_back_mask_activation_v1",
}

_BOX_AUTH = dict(
    amendment_reason=(
        "seated-close person box exceeds the 1024^2 work budget; a box can "
        "never exceed the frame budget"
    ),
    owner_authorization="op:qz 2026-08-14",
    source_run="run-2026-08-14",
    amender_code_sha256="a" * 64,
    amender_code_release="code_release-" + "b" * 20,
)


def _hash(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _build_box_source(tmp_path):
    """The refinalizer fixture with a REAL-shaped back_mask payload."""
    src = _build_source(tmp_path)
    payload_bytes = canonical_json_bytes(_REAL_BACK_MASK_PAYLOAD)
    ppath = src / "artifacts" / "back_mask.json"
    ppath.write_bytes(payload_bytes)
    marker = {
        "artifact_sha256": _hash(payload_bytes),
        "schema": "wam_production_runtime_artifact_completion_v1",
        "status": "COMPLETE",
    }
    marker_bytes = canonical_json_bytes(marker)
    (src / "markers" / "back_mask.COMPLETE").write_bytes(marker_bytes)
    manifest = json.loads((src / "activation.json").read_text())
    entry = manifest["artifacts"]["back_mask"]
    entry["sha256"] = _hash(payload_bytes)
    entry["complete_sha256"] = _hash(marker_bytes)
    manifest_bytes = canonical_json_bytes(manifest)
    (src / "activation.json").write_bytes(manifest_bytes)
    (src / "COMPLETE").write_bytes(canonical_json_bytes({
        "activation_sha256": _hash(manifest_bytes),
        "schema": "wam_production_runtime_activation_completion_v1",
        "status": "COMPLETE",
    }))
    return src


def _amend_box(src, releases, **over):
    kw = dict(
        _BOX_AUTH,
        expected_previous_px=1048576,
        amended_px=2097152,
        consumer_verifier=_verify,
    )
    kw.update(over)
    return amend_back_mask_box_budget(src, releases, **kw)


def test_box_amendment_changes_exactly_one_config_field(tmp_path):
    src = _build_box_source(tmp_path)
    dest = _amend_box(src, tmp_path / "releases")
    payload = json.loads((dest / "artifacts" / "back_mask.json").read_text())
    assert payload["config"]["maximum_box_pixels"] == 2097152
    reference = {
        k: v for k, v in _REAL_BACK_MASK_PAYLOAD["config"].items()
        if k != "maximum_box_pixels"
    }
    observed = {
        k: v for k, v in payload["config"].items()
        if k != "maximum_box_pixels"
    }
    assert observed == reference
    verify_relocated_release(dest)
    manifest = json.loads((dest / "activation.json").read_text())
    src_manifest = json.loads((src / "activation.json").read_text())
    assert manifest["robot_state_bounds"] == src_manifest["robot_state_bounds"]


def test_box_amendment_rebinds_sha_chain(tmp_path):
    src = _build_box_source(tmp_path)
    dest = _amend_box(src, tmp_path / "releases")
    manifest = json.loads((dest / "activation.json").read_text())
    entry = manifest["artifacts"]["back_mask"]
    payload_bytes = (dest / "artifacts" / "back_mask.json").read_bytes()
    marker_bytes = (dest / "markers" / "back_mask.COMPLETE").read_bytes()
    assert entry["sha256"] == _hash(payload_bytes)
    assert entry["complete_sha256"] == _hash(marker_bytes)
    marker = json.loads(marker_bytes)
    assert marker["artifact_sha256"] == entry["sha256"]
    src_entry = json.loads((src / "activation.json").read_text())[
        "artifacts"]["back_mask"]
    assert entry["sha256"] != src_entry["sha256"]


def test_box_amendment_other_artifacts_verbatim(tmp_path):
    src = _build_box_source(tmp_path)
    dest = _amend_box(src, tmp_path / "releases")
    src_manifest = json.loads((src / "activation.json").read_text())
    for name, src_entry in src_manifest["artifacts"].items():
        if name in ("back_mask", "handeye"):
            continue
        assert (dest / "artifacts" / f"{name}.json").read_bytes() == Path(
            src_entry["path"]).read_bytes()


def test_box_amendment_provenance(tmp_path):
    src = _build_box_source(tmp_path)
    dest = _amend_box(src, tmp_path / "releases")
    prov = json.loads((dest / BOX_AMENDMENT_PROVENANCE_FILE).read_text())
    assert prov["schema"] == BOX_AMENDMENT_PROVENANCE_SCHEMA
    assert prov["amended_field"] == "artifacts.back_mask.config.maximum_box_pixels"
    assert prov["previous_value_px"] == 1048576
    assert prov["amended_value_px"] == 2097152
    assert prov["owner_authorization"] == _BOX_AUTH["owner_authorization"]
    assert prov["release_id"] == dest.name


def test_box_amendment_idempotent_reuse(tmp_path):
    src = _build_box_source(tmp_path)
    a = _amend_box(src, tmp_path / "releases")
    b = _amend_box(src, tmp_path / "releases")
    assert a == b


def test_box_amendment_above_frame_budget_refused(tmp_path):
    src = _build_box_source(tmp_path)
    with pytest.raises(Exception, match="maximum_frame_pixels"):
        _amend_box(src, tmp_path / "releases", amended_px=2097153)


def test_box_amendment_wrong_previous_refused(tmp_path):
    src = _build_box_source(tmp_path)
    with pytest.raises(Exception, match="expected previous"):
        _amend_box(src, tmp_path / "releases", expected_previous_px=999)


def test_box_amendment_noop_refused(tmp_path):
    src = _build_box_source(tmp_path)
    with pytest.raises(Exception, match="nothing to amend"):
        _amend_box(src, tmp_path / "releases", amended_px=1048576)


def test_box_amendment_source_never_mutated(tmp_path):
    src = _build_box_source(tmp_path)
    before = {
        p.relative_to(src).as_posix(): p.read_bytes()
        for p in sorted(src.rglob("*")) if p.is_file()
    }
    _amend_box(src, tmp_path / "releases")
    after = {
        p.relative_to(src).as_posix(): p.read_bytes()
        for p in sorted(src.rglob("*")) if p.is_file()
    }
    assert before == after
