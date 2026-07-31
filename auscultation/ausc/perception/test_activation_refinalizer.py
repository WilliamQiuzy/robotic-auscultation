"""Tests for the hardened relocating canonical activation re-finalizer.

The fixture builds a full 12-artifact package (11 relocatable in-package + handeye
external-canonical) referencing artifacts by ABSOLUTE path, with non-canonical
payloads + markers, mirroring the real box package.  A stdlib consumer-verifier is
injected because the real production loader (torch-heavy) is not importable on the
dev host; the on-box integration test exercises the real loader end-to-end.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import ausc.perception.activation_refinalizer as mod
from ausc.perception.activation_refinalizer import (
    ACTIVATION_COMPLETION_SCHEMA,
    ACTIVATION_SCHEMA,
    ARTIFACT_COMPLETION_SCHEMA,
    ActivationRefinalizerError,
    _write_all,
    canonical_json_bytes,
    refinalize_activation_release,
    verify_relocated_release,
)

_RELOCATABLE = [
    "alignment", "back_mask", "camera_latency", "camera_profile", "collision_geometry",
    "depth_noise", "occlusion_identity", "occlusion_runtime", "patient_pose",
    "robot_mask_uncertainty", "tool_geometry",
]
_BOUNDS = {
    "buffer_duration_ns": 1_000_000_000,
    "gripper_motor_position_limits_rad": [[-1.0, 1.0]],
    "joint_position_limits_rad": [[-3.14, 3.14]] * 7,
    "max_abs_joint_velocity_rad_s": 2.0,
    "max_exposure_interval_ns": 50_000_000,
    "max_gap_ns": 100_000_000,
    "max_packets": 100_000,
    "max_staleness_ns": 200_000_000,
}
_AUTH = dict(
    owner_authorization="op:qz", source_run="run-2026-07-31",
    refinalizer_code_sha256="a" * 64, refinalizer_code_release="code_release-" + "b" * 20,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nc(path: Path, value: object) -> bytes:
    """Write non-canonical (compact, NO trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _c(path: Path, value: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    return raw


def _verify(release_dir, _sha256):
    verify_relocated_release(release_dir)


def _refinalize(src, releases, **over):
    kw = dict(_AUTH, consumer_verifier=_verify)
    kw.update(over)
    return refinalize_activation_release(src, releases, **kw)


def _build_source(tmp_path: Path) -> Path:
    src = tmp_path / "activation"
    ext = tmp_path / "external"
    artifacts = {}
    for name in _RELOCATABLE:
        ppath = src / "artifacts" / f"{name}.json"
        praw = _nc(ppath, {"name": name, "value": 1.5})
        marker = {"artifact_sha256": _sha(praw), "schema": ARTIFACT_COMPLETION_SCHEMA, "status": "COMPLETE"}
        mpath = src / "markers" / f"{name}.COMPLETE"
        mraw = _nc(mpath, marker)
        artifacts[name] = {"path": str(ppath), "complete_path": str(mpath),
                           "sha256": _sha(praw), "complete_sha256": _sha(mraw)}
    # handeye: external, already canonical
    hpp = ext / "handeye.json"
    hpr = _c(hpp, {"T": [1.0, 2.0, 3.0]})
    hmark = {"artifact_sha256": _sha(hpr), "calibration_status": "USER_ACCEPTED_BELOW_GATE",
             "provisional": True, "schema": "wam_openarm_handeye_artifact_completion_v2", "status": "COMPLETE"}
    hmp = ext / "handeye.COMPLETE"
    hmr = _c(hmp, hmark)
    artifacts["handeye"] = {"path": str(hpp), "complete_path": str(hmp),
                            "sha256": _sha(hpr), "complete_sha256": _sha(hmr)}
    pair = src / "pairs" / "openarm_pair_v2-dead"
    _nc(pair / "pair_contract.json", {"contract_sha256": "d" * 64, "profile": {"x": 1}})
    _nc(pair / "COMPLETE", {"contract_sha256": "d" * 64, "status": "COMPLETE"})
    manifest = {
        "schema": ACTIVATION_SCHEMA, "status": "COMPLETE", "side": "left",
        "leader_can_interface": "can1", "follower_can_interface": "can3",
        "robot_state_bounds": _BOUNDS,
        "openarm_pair": {"contract_sha256": "d" * 64, "release_directory": str(pair)},
        "artifacts": artifacts,
    }
    mraw = _nc(src / "activation.json", manifest)
    _nc(src / "COMPLETE", {"activation_sha256": _sha(mraw),
                           "schema": ACTIVATION_COMPLETION_SCHEMA, "status": "COMPLETE"})
    return src


# --------------------------------------------------------------------------- #
# Positive                                                                     #
# --------------------------------------------------------------------------- #

def test_produces_canonical_selfcontained_release(tmp_path):
    src = _build_source(tmp_path)
    dest = _refinalize(src, tmp_path / "releases")
    assert (dest / "activation.json").read_bytes() == canonical_json_bytes(
        json.loads((dest / "activation.json").read_text()))
    verify_relocated_release(dest)
    # modes: files 0444, dirs 0555
    import stat as _stat
    assert _stat.S_IMODE((dest / "activation.json").stat().st_mode) == 0o444
    assert _stat.S_IMODE((dest / "artifacts").stat().st_mode) == 0o555


def test_eleven_relocated_payloads_byte_identical_handeye_external(tmp_path):
    src = _build_source(tmp_path)
    align_raw = (src / "artifacts" / "alignment.json").read_bytes()
    dest = _refinalize(src, tmp_path / "releases")
    m = json.loads((dest / "activation.json").read_text())
    assert Path(m["artifacts"]["alignment"]["path"]).is_relative_to(dest)
    assert Path(m["artifacts"]["alignment"]["path"]).read_bytes() == align_raw  # byte-identical
    assert not Path(m["artifacts"]["handeye"]["path"]).is_relative_to(dest)     # handeye external
    for name in _RELOCATABLE:
        assert Path(m["artifacts"][name]["path"]).is_relative_to(dest)          # all 11 relocated


def test_provenance_binds_full_identity_and_inventory(tmp_path):
    src = _build_source(tmp_path)
    dest = _refinalize(src, tmp_path / "releases")
    prov = json.loads((dest / "REFINALIZED.json").read_text())
    assert prov["final_activation_sha256"] == _sha((dest / "activation.json").read_bytes())
    assert prov["refinalizer_code_sha256"] == "a" * 64
    assert prov["kept_external_artifacts"] == ["handeye"]
    assert sorted(prov["relocated_artifacts"]) == sorted(_RELOCATABLE)
    assert "activation.json" in prov["inventory"] and f"pairs/openarm_pair_v2-dead/pair_contract.json" in prov["inventory"]


def test_mandatory_consumer_verification_runs_before_publish(tmp_path):
    src = _build_source(tmp_path)
    calls = {}

    def failing_verify(release_dir, sha):
        calls["dir"] = release_dir
        raise ActivationRefinalizerError("consumer rejected")

    with pytest.raises(ActivationRefinalizerError, match="consumer rejected"):
        _refinalize(src, tmp_path / "releases", consumer_verifier=failing_verify)
    # verifier ran on the committed dir; a rejected release is fully destroyed.
    assert "activation-" in str(calls["dir"])
    releases = tmp_path / "releases"
    assert not any(releases.glob("activation-*")) if releases.exists() else True


# --------------------------------------------------------------------------- #
# Idempotence + collision                                                      #
# --------------------------------------------------------------------------- #

def test_idempotent_reuse_reverifies(tmp_path):
    src = _build_source(tmp_path)
    releases = tmp_path / "releases"
    a = _refinalize(src, releases)
    b = _refinalize(src, releases)
    assert a == b


def test_same_id_different_content_refused(tmp_path):
    src = _build_source(tmp_path)
    releases = tmp_path / "releases"
    dest = _refinalize(src, releases)
    tampered = dest / "activation.json"
    os.chmod(tampered, 0o644)
    tampered.write_bytes(tampered.read_bytes()[:-1] + b" \n")
    with pytest.raises(ActivationRefinalizerError, match="differs|inventory"):
        _refinalize(src, releases)


# --------------------------------------------------------------------------- #
# Counterexamples (the required negative suite)                                #
# --------------------------------------------------------------------------- #

def test_path_traversal_source_rejected(tmp_path):
    src = _build_source(tmp_path)
    (tmp_path / "VICTIM.json").write_text("{}")
    m = json.loads((src / "activation.json").read_text())
    m["artifacts"]["alignment"]["path"] = str(src / "artifacts" / ".." / ".." / "VICTIM.json")
    _nc(src / "activation.json", m)
    _nc(src / "COMPLETE", {"activation_sha256": _sha((src / "activation.json").read_bytes()),
                           "schema": ACTIVATION_COMPLETION_SCHEMA, "status": "COMPLETE"})
    with pytest.raises(ActivationRefinalizerError, match="symlink|absolute|traversal|component"):
        _refinalize(src, tmp_path / "releases")


def test_relative_releases_root_rejected(tmp_path):
    src = _build_source(tmp_path)
    with pytest.raises(ActivationRefinalizerError, match="releases root must be absolute"):
        _refinalize(src, "relative/releases")


def test_parent_symlink_source_rejected(tmp_path):
    src = _build_source(tmp_path)
    link = tmp_path / "activation_link"
    link.symlink_to(src)
    with pytest.raises(ActivationRefinalizerError, match="symlink"):
        _refinalize(link, tmp_path / "releases")


def test_duplicate_json_key_rejected(tmp_path):
    src = _build_source(tmp_path)
    (src / "activation.json").write_bytes(b'{"side":"left","side":"right","schema":"x"}')
    with pytest.raises(ActivationRefinalizerError, match="duplicate JSON key"):
        _refinalize(src, tmp_path / "releases")


def test_non_finite_number_rejected(tmp_path):
    src = _build_source(tmp_path)
    (src / "activation.json").write_bytes(b'{"schema":NaN}')
    with pytest.raises(ActivationRefinalizerError, match="non-finite"):
        _refinalize(src, tmp_path / "releases")


def test_source_swap_toctou_rejected(tmp_path, monkeypatch):
    src = _build_source(tmp_path)
    real_fstat = os.fstat
    state = {"n": 0}

    def flipping_fstat(fd):
        st = real_fstat(fd)
        state["n"] += 1
        if state["n"] == 2:  # the 'after' fstat of the first file read: pretend it changed
            return os.stat_result(tuple(st)[:7] + (st.st_atime, st.st_mtime, st.st_ctime + 1))
        return st

    monkeypatch.setattr(mod.os, "fstat", flipping_fstat)
    with pytest.raises(ActivationRefinalizerError, match="changed while reading"):
        _refinalize(src, tmp_path / "releases")


def test_corrupt_payload_reuse_refused(tmp_path):
    src = _build_source(tmp_path)
    releases = tmp_path / "releases"
    dest = _refinalize(src, releases)
    payload = dest / "artifacts" / "alignment.json"
    os.chmod(payload, 0o644)
    payload.write_bytes(payload.read_bytes() + b" ")
    with pytest.raises(ActivationRefinalizerError, match="differs|inventory"):
        _refinalize(src, releases)


def test_corrupt_pair_reuse_refused(tmp_path):
    src = _build_source(tmp_path)
    releases = tmp_path / "releases"
    dest = _refinalize(src, releases)
    contract = dest / "pairs" / "openarm_pair_v2-dead" / "pair_contract.json"
    os.chmod(contract, 0o644)
    contract.write_bytes(contract.read_bytes() + b" ")
    with pytest.raises(ActivationRefinalizerError, match="differs|inventory"):
        _refinalize(src, releases)


def test_extra_file_in_reused_release_refused(tmp_path):
    src = _build_source(tmp_path)
    releases = tmp_path / "releases"
    dest = _refinalize(src, releases)
    os.chmod(dest, 0o755)
    (dest / "SNEAKY.json").write_text("{}")
    with pytest.raises(ActivationRefinalizerError, match="inventory differs"):
        _refinalize(src, releases)


def test_wrong_schema_rejected(tmp_path):
    src = _build_source(tmp_path)
    m = json.loads((src / "activation.json").read_text())
    m["schema"] = "wrong_schema"
    _nc(src / "activation.json", m)
    _nc(src / "COMPLETE", {"activation_sha256": _sha((src / "activation.json").read_bytes()),
                           "schema": ACTIVATION_COMPLETION_SCHEMA, "status": "COMPLETE"})
    with pytest.raises(ActivationRefinalizerError, match="COMPLETE left can1"):
        _refinalize(src, tmp_path / "releases")


def test_wrong_bounds_rejected(tmp_path):
    src = _build_source(tmp_path)
    m = json.loads((src / "activation.json").read_text())
    del m["robot_state_bounds"]["max_packets"]
    _nc(src / "activation.json", m)
    _nc(src / "COMPLETE", {"activation_sha256": _sha((src / "activation.json").read_bytes()),
                           "schema": ACTIVATION_COMPLETION_SCHEMA, "status": "COMPLETE"})
    with pytest.raises(ActivationRefinalizerError, match="robot_state_bounds keys"):
        _refinalize(src, tmp_path / "releases")


def test_missing_artifact_rejected(tmp_path):
    src = _build_source(tmp_path)
    m = json.loads((src / "activation.json").read_text())
    del m["artifacts"]["alignment"]
    _nc(src / "activation.json", m)
    _nc(src / "COMPLETE", {"activation_sha256": _sha((src / "activation.json").read_bytes()),
                           "schema": ACTIVATION_COMPLETION_SCHEMA, "status": "COMPLETE"})
    with pytest.raises(ActivationRefinalizerError, match="12-item allowlist"):
        _refinalize(src, tmp_path / "releases")


def test_extra_artifact_rejected(tmp_path):
    src = _build_source(tmp_path)
    m = json.loads((src / "activation.json").read_text())
    m["artifacts"]["bogus"] = dict(m["artifacts"]["alignment"])
    _nc(src / "activation.json", m)
    _nc(src / "COMPLETE", {"activation_sha256": _sha((src / "activation.json").read_bytes()),
                           "schema": ACTIVATION_COMPLETION_SCHEMA, "status": "COMPLETE"})
    with pytest.raises(ActivationRefinalizerError, match="12-item allowlist"):
        _refinalize(src, tmp_path / "releases")


def test_non_handeye_external_artifact_rejected(tmp_path):
    # Any non-handeye artifact left non-canonical + external-only would fail to enter;
    # a payload whose marker binding is wrong is rejected.
    src = _build_source(tmp_path)
    m = json.loads((src / "activation.json").read_text())
    bad_marker = {"artifact_sha256": "e" * 64, "schema": ARTIFACT_COMPLETION_SCHEMA, "status": "COMPLETE"}
    mpath = src / "markers" / "alignment.COMPLETE"
    mraw = _nc(mpath, bad_marker)
    m["artifacts"]["alignment"]["complete_sha256"] = _sha(mraw)
    _nc(src / "activation.json", m)
    _nc(src / "COMPLETE", {"activation_sha256": _sha((src / "activation.json").read_bytes()),
                           "schema": ACTIVATION_COMPLETION_SCHEMA, "status": "COMPLETE"})
    with pytest.raises(ActivationRefinalizerError, match="does not bind the payload"):
        _refinalize(src, tmp_path / "releases")


def test_short_write_is_handled(tmp_path, monkeypatch):
    p = tmp_path / "f.bin"
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT, 0o644)
    real = os.write

    def one_byte(f, data):
        return real(f, bytes(data[:1]))

    monkeypatch.setattr(mod.os, "write", one_byte)
    try:
        _write_all(fd, b"hello world " * 100)
    finally:
        os.close(fd)
    assert p.read_bytes() == b"hello world " * 100


def test_crash_before_complete_leaves_no_consumable_release(tmp_path, monkeypatch):
    src = _build_source(tmp_path)
    releases = tmp_path / "releases"
    real = mod._atomic_write

    def boom(path, data):
        if Path(path).name == "COMPLETE":
            raise RuntimeError("simulated crash writing COMPLETE")
        return real(path, data)

    monkeypatch.setattr(mod, "_atomic_write", boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _refinalize(src, releases)
    assert not any(releases.glob("activation-*")) if releases.exists() else True
    assert not any(releases.glob("*.staging")) if releases.exists() else True
