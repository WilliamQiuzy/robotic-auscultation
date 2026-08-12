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
