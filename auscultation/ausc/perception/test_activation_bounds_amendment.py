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


def _derived_config_identity(config):
    from ausc.perception.tracking.posterior_back_mask_provider import (
        PosteriorBackMaskConfig,
    )

    return PosteriorBackMaskConfig.create(**config)


def _build_box_source(tmp_path):
    """The refinalizer fixture with REAL-shaped back_mask and
    occlusion_identity payloads whose cross-pin matches the derived config
    identity (the loader asserts this equality at activation time)."""
    src = _build_source(tmp_path)
    derived = _derived_config_identity(_REAL_BACK_MASK_PAYLOAD["config"])
    replacements = {
        "back_mask": _REAL_BACK_MASK_PAYLOAD,
        "occlusion_identity": {
            "schema": "wam_activated_occlusion_identity_v2",
            "camera_id": "CL-TEST",
            "back_mask_release_id": derived.release_id,
            "back_mask_sha256": derived.semantic_sha256,
            "other_pin": "untouched-value",
        },
    }
    manifest = json.loads((src / "activation.json").read_text())
    for name, payload in replacements.items():
        payload_bytes = canonical_json_bytes(payload)
        (src / "artifacts" / f"{name}.json").write_bytes(payload_bytes)
        marker_bytes = canonical_json_bytes({
            "artifact_sha256": _hash(payload_bytes),
            "schema": "wam_production_runtime_artifact_completion_v1",
            "status": "COMPLETE",
        })
        (src / "markers" / f"{name}.COMPLETE").write_bytes(marker_bytes)
        entry = manifest["artifacts"][name]
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
        if name in ("back_mask", "handeye", "occlusion_identity"):
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


def test_box_amendment_rewrites_the_occlusion_identity_cross_pin(tmp_path):
    """The loader re-derives the back_mask identity from config content and
    asserts occlusion_identity records it; the amendment must move both
    (first deployment without this refused to start: 'activated occlusion
    identity mismatches back_mask_release_id')."""
    src = _build_box_source(tmp_path)
    dest = _amend_box(src, tmp_path / "releases")
    identity = json.loads(
        (dest / "artifacts" / "occlusion_identity.json").read_text()
    )
    new_payload = json.loads(
        (dest / "artifacts" / "back_mask.json").read_text()
    )
    derived = _derived_config_identity(new_payload["config"])
    assert identity["back_mask_release_id"] == derived.release_id
    assert identity["back_mask_sha256"] == derived.semantic_sha256
    assert identity["other_pin"] == "untouched-value"
    manifest = json.loads((dest / "activation.json").read_text())
    entry = manifest["artifacts"]["occlusion_identity"]
    ident_bytes = (dest / "artifacts" / "occlusion_identity.json").read_bytes()
    assert entry["sha256"] == _hash(ident_bytes)
    prov = json.loads((dest / BOX_AMENDMENT_PROVENANCE_FILE).read_text())
    pins = prov["cross_pin_updates"]
    assert pins["artifacts.occlusion_identity.back_mask_release_id"][
        "amended"] == derived.release_id


def test_box_amendment_refuses_a_stale_identity_cross_pin(tmp_path):
    """If occlusion_identity does not record the SOURCE config's derived
    identity, the package is already inconsistent — refuse, never guess."""
    src = _build_box_source(tmp_path)
    ident_path = src / "artifacts" / "occlusion_identity.json"
    payload = json.loads(ident_path.read_text())
    payload["back_mask_release_id"] = "back_mask-" + "0" * 20
    payload_bytes = canonical_json_bytes(payload)
    ident_path.write_bytes(payload_bytes)
    marker_bytes = canonical_json_bytes({
        "artifact_sha256": _hash(payload_bytes),
        "schema": "wam_production_runtime_artifact_completion_v1",
        "status": "COMPLETE",
    })
    (src / "markers" / "occlusion_identity.COMPLETE").write_bytes(marker_bytes)
    manifest = json.loads((src / "activation.json").read_text())
    entry = manifest["artifacts"]["occlusion_identity"]
    entry["sha256"] = _hash(payload_bytes)
    entry["complete_sha256"] = _hash(marker_bytes)
    manifest_bytes = canonical_json_bytes(manifest)
    (src / "activation.json").write_bytes(manifest_bytes)
    (src / "COMPLETE").write_bytes(canonical_json_bytes({
        "activation_sha256": _hash(manifest_bytes),
        "schema": "wam_production_runtime_activation_completion_v1",
        "status": "COMPLETE",
    }))
    with pytest.raises(Exception, match="source back_mask identity"):
        _amend_box(src, tmp_path / "releases")


# --------------------------------------------------------------------------- #
# hand-eye swap amendment                                                      #
# --------------------------------------------------------------------------- #

import ausc.perception.activation_bounds_amendment as _abam
from ausc.perception.activation_bounds_amendment import (  # noqa: E402
    HANDEYE_AMENDMENT_PROVENANCE_FILE,
    HANDEYE_AMENDMENT_PROVENANCE_SCHEMA,
    amend_handeye_swap,
)

_OLD_DERIVED = {
    "handeye_release_id": "handeye-" + "a" * 20,
    "handeye_sha256": None,  # filled by _build_handeye_source from the fixture
    "handeye_capability_schema": "wam_provisional_robot_mask_handeye_capability_v1",
    "handeye_calibration_tier": "PROVISIONAL_USER_ACCEPTED_5MM",
    "handeye_policy_sha256": "b" * 64,
}
_NEW_DERIVED = {
    "handeye_release_id": "handeye-" + "c" * 20,
    "handeye_sha256": None,  # filled by _swap from the fake release content
    "handeye_capability_schema": "wam_provisional_robot_mask_handeye_capability_v1",
    "handeye_calibration_tier": "VALIDATED_3MM",
    "handeye_policy_sha256": "d" * 64,
}

_HE_AUTH = dict(
    amendment_reason="hand-eye graduated to the validated tier on the fixture",
    owner_authorization="op:qz graduation session",
    source_run="handeye_left_20260901T000000Z_deadbeef",
    amender_code_sha256="e" * 64,
    amender_code_release="code_release-" + "f" * 20,
)


def _build_handeye_source(tmp_path):
    """Box-source fixture whose occlusion_identity ALSO carries the five
    hand-eye pins matching the (fake) old derivation, and whose external
    hand-eye artifact sha matches _OLD_DERIVED."""
    src = _build_box_source(tmp_path)
    manifest = json.loads((src / "activation.json").read_text())
    # the old derivation binds to the fixture's REAL external handeye sha
    _OLD_DERIVED["handeye_sha256"] = manifest["artifacts"]["handeye"]["sha256"]
    ident_path = src / "artifacts" / "occlusion_identity.json"
    ident = json.loads(ident_path.read_text())
    ident.update(_OLD_DERIVED)
    ident_bytes = canonical_json_bytes(ident)
    ident_path.write_bytes(ident_bytes)
    marker_bytes = canonical_json_bytes({
        "artifact_sha256": _hash(ident_bytes),
        "schema": "wam_production_runtime_artifact_completion_v1",
        "status": "COMPLETE",
    })
    (src / "markers" / "occlusion_identity.COMPLETE").write_bytes(marker_bytes)
    entry = manifest["artifacts"]["occlusion_identity"]
    entry["sha256"] = _hash(ident_bytes)
    entry["complete_sha256"] = _hash(marker_bytes)
    manifest_bytes = canonical_json_bytes(manifest)
    (src / "activation.json").write_bytes(manifest_bytes)
    (src / "COMPLETE").write_bytes(canonical_json_bytes({
        "activation_sha256": _hash(manifest_bytes),
        "schema": "wam_production_runtime_activation_completion_v1",
        "status": "COMPLETE",
    }))
    return src


def _fake_derive(release_dir, artifact_sha256, complete_sha256, camera_id):
    if artifact_sha256 == _OLD_DERIVED["handeye_sha256"]:
        return dict(_OLD_DERIVED)
    return dict(_NEW_DERIVED)


def _swap(src, releases, tmp_path, monkeypatch, **over):
    monkeypatch.setattr(_abam, "_derive_handeye_identity", _fake_derive)
    new_rel = tmp_path / "handeye_new"
    new_rel.mkdir(exist_ok=True)
    payload_bytes = canonical_json_bytes({"T": [4.0, 5.0, 6.0]})
    (new_rel / "handeye.json").write_bytes(payload_bytes)
    marker_bytes = canonical_json_bytes({"status": "COMPLETE"})
    (new_rel / "handeye.COMPLETE").write_bytes(marker_bytes)
    _NEW_DERIVED["handeye_sha256"] = _hash(payload_bytes)
    kw = dict(
        _HE_AUTH,
        new_release_dir=new_rel,
        new_artifact_sha256=_hash(payload_bytes),
        new_complete_sha256=_hash(marker_bytes),
        consumer_verifier=_verify,
    )
    kw.update(over)
    return amend_handeye_swap(src, releases, **kw)


def test_handeye_swap_moves_entry_and_all_five_pins(tmp_path, monkeypatch):
    src = _build_handeye_source(tmp_path)
    dest = _swap(src, tmp_path / "releases", tmp_path, monkeypatch)
    manifest = json.loads((dest / "activation.json").read_text())
    he = manifest["artifacts"]["handeye"]
    assert he["sha256"] == _NEW_DERIVED["handeye_sha256"]
    assert he["path"].endswith("handeye_new/handeye.json")
    ident = json.loads((dest / "artifacts" / "occlusion_identity.json").read_text())
    for field, value in _NEW_DERIVED.items():
        assert ident[field] == value, field
    assert ident["other_pin"] == "untouched-value"
    verify_relocated_release(dest)
    prov = json.loads((dest / HANDEYE_AMENDMENT_PROVENANCE_FILE).read_text())
    assert prov["schema"] == HANDEYE_AMENDMENT_PROVENANCE_SCHEMA
    assert prov["cross_pin_updates"][
        "artifacts.occlusion_identity.handeye_calibration_tier"]["amended"] == "VALIDATED_3MM"


def test_handeye_swap_refuses_stale_identity_pins(tmp_path, monkeypatch):
    src = _build_handeye_source(tmp_path)
    ident_path = src / "artifacts" / "occlusion_identity.json"
    ident = json.loads(ident_path.read_text())
    ident["handeye_policy_sha256"] = "0" * 64
    ident_bytes = canonical_json_bytes(ident)
    ident_path.write_bytes(ident_bytes)
    marker_bytes = canonical_json_bytes({
        "artifact_sha256": _hash(ident_bytes),
        "schema": "wam_production_runtime_artifact_completion_v1",
        "status": "COMPLETE",
    })
    (src / "markers" / "occlusion_identity.COMPLETE").write_bytes(marker_bytes)
    manifest = json.loads((src / "activation.json").read_text())
    entry = manifest["artifacts"]["occlusion_identity"]
    entry["sha256"] = _hash(ident_bytes)
    entry["complete_sha256"] = _hash(marker_bytes)
    manifest_bytes = canonical_json_bytes(manifest)
    (src / "activation.json").write_bytes(manifest_bytes)
    (src / "COMPLETE").write_bytes(canonical_json_bytes({
        "activation_sha256": _hash(manifest_bytes),
        "schema": "wam_production_runtime_activation_completion_v1",
        "status": "COMPLETE",
    }))
    with pytest.raises(Exception, match="source hand-eye"):
        _swap(src, tmp_path / "releases", tmp_path, monkeypatch)


def test_handeye_swap_refuses_identical_release(tmp_path, monkeypatch):
    src = _build_handeye_source(tmp_path)
    with pytest.raises(Exception, match="nothing to swap"):
        _swap(src, tmp_path / "releases", tmp_path, monkeypatch,
              new_artifact_sha256=_OLD_DERIVED["handeye_sha256"])


def test_handeye_swap_idempotent(tmp_path, monkeypatch):
    src = _build_handeye_source(tmp_path)
    a = _swap(src, tmp_path / "releases", tmp_path, monkeypatch)
    b = _swap(src, tmp_path / "releases", tmp_path, monkeypatch)
    assert a == b
