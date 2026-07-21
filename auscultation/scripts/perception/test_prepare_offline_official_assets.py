"""Focused fail-closed tests for freezing the official offline asset release."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(REPOSITORY_ROOT / "auscultation"))

from ausc.perception.tracking import official_asset_contract as contract  # noqa: E402
from scripts.perception import prepare_offline_official_assets as builder  # noqa: E402


EXPECTED_MANIFEST_SHA256 = (
    "05e66c94cd8b36ff79bef7bfcaa73d176e621120ac6398e808ad5df938a0cf35"
)
EXPECTED_RELEASE_ID = f"official_assets-{EXPECTED_MANIFEST_SHA256[:20]}"
LEGACY_TRUNCATED_MANIFEST_SHA256 = (
    "f35f7e512c7546bb25392d836ff544cbfe7609ea3c29d225ac10d1a82b52870"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_frozen_contract_and_download_specs_are_exact() -> None:
    assert contract.ENACTIC_REPOSITORY_URL == (
        "https://github.com/enactic/openarm_description.git"
    )
    assert contract.ENACTIC_COMMIT == "846a82d8baf64efdbb7f55c7f64b9b004939061b"
    assert contract.OFFICIAL_URDF_SHA256 == (
        "63caf1116a844c7eca25e91f0acf025c2798cc4795cf727b23433eacde80ca12"
    )
    expected_textures = {
        "human/textures/male_adult_medical_01_BaseColor.png": (
            "ee433d8a6ef8de8bb172e3b733b4a9cc3e226da5ce2884fa06be6ae37b15f585",
            5667369,
        ),
        "human/textures/male_adult_medical_01_Normal.png": (
            "f76f5442a1a8e96f6547c358245589212cfdf257c8b232ee263dfb432e0e65bd",
            13786667,
        ),
        "human/textures/male_adult_medical_01_ORM.png": (
            "96e15659f384477922ddd6e92489515605d12c508de18645bb412e9f13fbb0b0",
            12140285,
        ),
    }
    assert contract.HUMAN_USD_SHA256 == (
        "2c65e15587a311c5d14cdbd6a1408e0c3d8f3bec69f483150f04ca4c8f4e3ccc"
    )
    assert contract.HUMAN_USD_SIZE == 598342
    assert {
        path: (contract.NVIDIA_TEXTURE_SHA256[path], contract.NVIDIA_TEXTURE_SIZE[path])
        for path in contract.NVIDIA_TEXTURE_SHA256
    } == expected_textures

    specs = builder._download_specs()
    assert [(row.path, row.sha256, row.size) for row in specs] == [
        (contract.HUMAN_USD_PATH, contract.HUMAN_USD_SHA256, contract.HUMAN_USD_SIZE),
        *[
            (path, contract.NVIDIA_TEXTURE_SHA256[path], contract.NVIDIA_TEXTURE_SIZE[path])
            for path in contract.NVIDIA_TEXTURE_SHA256
        ],
    ]
    base_url = contract.NVIDIA_MEDICAL_HUMAN_URL.rsplit("/", 1)[0]
    assert [row.url for row in specs] == [
        contract.NVIDIA_MEDICAL_HUMAN_URL,
        *[f"{base_url}/textures/{Path(path).name}" for path in contract.NVIDIA_TEXTURE_SHA256],
    ]


def test_manifest_identity_is_64_hex_rejects_legacy_truncation_and_binds_size() -> None:
    assert len(EXPECTED_MANIFEST_SHA256) == 64
    assert int(EXPECTED_MANIFEST_SHA256, 16) >= 0
    assert len(LEGACY_TRUNCATED_MANIFEST_SHA256) == 63
    assert builder.EXPECTED_MANIFEST_SHA256 == EXPECTED_MANIFEST_SHA256
    assert builder.EXPECTED_MANIFEST_SIZE == 14362
    with pytest.raises(builder.PreparationError, match="exact canonical manifest"):
        builder._completion_bytes(
            {"files": []}, LEGACY_TRUNCATED_MANIFEST_SHA256
        )


def test_empty_bundle_fails_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(builder.PreparationError):
        builder.verify_release(empty)


def test_staging_is_removed_after_failure(tmp_path: Path) -> None:
    build_root = tmp_path / "build"
    with pytest.raises(RuntimeError, match="injected"):
        with builder._staging_bundle(build_root) as staging:
            (staging / "download.partial").write_bytes(b"partial")
            raise RuntimeError("injected")
    assert (build_root / ".staging").is_dir()
    assert list((build_root / ".staging").iterdir()) == []


def _init_local_git_checkout(release: Path) -> Path:
    checkout = release / contract.CHECKOUT_PATH
    checkout.mkdir(parents=True)
    subprocess.run(
        ["/usr/bin/git", "init", "--quiet", os.fspath(checkout)],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    return checkout


def test_every_release_verification_rejects_shallow_and_partial_git_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / builder.RELEASE_ID
    checkout = _init_local_git_checkout(release)
    monkeypatch.setattr(builder, "_verify_bundle_contents", lambda _root: {"files": []})

    shallow = checkout / ".git/shallow"
    shallow.write_text("0" * 40 + "\n", encoding="ascii")
    with pytest.raises(builder.PreparationError, match="shallow"):
        builder.verify_release(release)
    shallow.unlink()

    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            os.fspath(checkout),
            "config",
            "--local",
            "remote.origin.promisor",
            "true",
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    with pytest.raises(builder.PreparationError, match="partial|promisor"):
        builder.verify_release(release)

    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            os.fspath(checkout),
            "config",
            "--local",
            "--unset-all",
            "remote.origin.promisor",
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    promisor_marker = checkout / ".git/objects/pack/pack-deadbeef.promisor"
    promisor_marker.write_bytes(b"")
    with pytest.raises(builder.PreparationError, match="promisor|partial"):
        builder.verify_release(release)


def test_atomic_no_replace_never_overwrites_file_or_directory(tmp_path: Path) -> None:
    source_file = tmp_path / "source.bin"
    destination_file = tmp_path / "destination.bin"
    source_file.write_bytes(b"candidate")
    destination_file.write_bytes(b"concurrent")
    with pytest.raises(builder.PreparationError, match="exists|conflict|publish"):
        builder._rename_no_replace(source_file, destination_file)
    assert source_file.read_bytes() == b"candidate"
    assert destination_file.read_bytes() == b"concurrent"

    source_directory = tmp_path / "source-release"
    destination_directory = tmp_path / "destination-release"
    source_directory.mkdir()
    (source_directory / "candidate.txt").write_text("candidate", encoding="utf-8")
    destination_directory.mkdir()
    with pytest.raises(builder.PreparationError, match="exists|conflict|publish"):
        builder._rename_no_replace(source_directory, destination_directory)
    assert (source_directory / "candidate.txt").read_text(encoding="utf-8") == "candidate"
    assert list(destination_directory.iterdir()) == []


def test_release_publication_reverifies_and_rolls_back_owned_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / builder.RELEASE_ID
    staging.mkdir()
    (staging / "owned.txt").write_text("owned", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(
        builder,
        "_fsync_tree",
        lambda path: calls.append(f"fsync:{Path(path).name}"),
        raising=False,
    )
    monkeypatch.setattr(
        builder,
        "_verify_checkout_transport_state",
        lambda path: calls.append(f"transport:{Path(path).name}"),
    )
    monkeypatch.setattr(
        builder,
        "_verify_bundle_contents",
        lambda path: calls.append(f"bundle:{Path(path).name}") or {"files": []},
    )

    def fail_post_publish(_path: Path) -> dict[str, object]:
        calls.append("post-publish")
        raise builder.PreparationError("injected post-publication verification failure")

    monkeypatch.setattr(builder, "verify_release", fail_post_publish)
    with pytest.raises(builder.PreparationError, match="post-publication"):
        builder._publish_staged_release(staging, destination=destination)
    assert calls == [
        "fsync:staging",
        "transport:staging",
        "bundle:staging",
        "post-publish",
    ]
    assert not staging.exists()
    assert not destination.exists()

    rejected = tmp_path / "rejected-staging"
    rejected.mkdir()
    (rejected / "owned.txt").write_text("owned", encoding="utf-8")

    def reject_transport(_path: Path) -> None:
        raise builder.PreparationError("injected transport corruption")

    monkeypatch.setattr(builder, "_verify_checkout_transport_state", reject_transport)
    with pytest.raises(builder.PreparationError, match="transport corruption"):
        builder._publish_staged_release(rejected, destination=destination)
    assert rejected.is_dir()
    assert not destination.exists()


class _Response(io.BytesIO):
    def __init__(self, data: bytes, url: str):
        super().__init__(data)
        self._url = url
        self.read_sizes: list[int] = []

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


def test_download_is_streamed_verified_atomic_and_rejects_bad_final_url(
    tmp_path: Path,
) -> None:
    raw = b"official bytes" * 100
    url = contract.NVIDIA_MEDICAL_HUMAN_URL
    response = _Response(raw, url)
    target = tmp_path / "human" / "asset.usd"
    builder._download_verified(
        url, target, _sha(raw), len(raw), opener=lambda _url: response
    )
    assert target.read_bytes() == raw
    assert all(size == builder.DOWNLOAD_CHUNK_SIZE for size in response.read_sizes)
    assert not list(tmp_path.rglob("*.partial-*"))

    redirected = tmp_path / "bad.usd"
    with pytest.raises(builder.PreparationError, match="HTTPS|host|redirect"):
        builder._download_verified(
            url,
            redirected,
            _sha(raw),
            len(raw),
            opener=lambda _url: _Response(raw, "http://example.invalid/asset.usd"),
        )
    assert not redirected.exists()
    assert not list(tmp_path.rglob("*.partial-*"))


def test_completion_marker_has_exact_canonical_schema_and_bindings() -> None:
    manifest = {"files": [{"path": "x", "sha256": "1" * 64, "size": 1}]}
    raw = builder._completion_bytes(manifest, EXPECTED_MANIFEST_SHA256)
    marker = json.loads(raw)
    assert set(marker) == {
        "asset_manifest_sha256",
        "inventory_sha256",
        "release_id",
        "schema",
        "status",
    }
    assert marker == {
        "asset_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "inventory_sha256": _sha(contract.canonical_json_bytes(manifest["files"])),
        "release_id": EXPECTED_RELEASE_ID,
        "schema": "wam_official_asset_completion_v1",
        "status": "complete",
    }
    assert raw == contract.canonical_json_bytes(marker)


def test_real_final_release_is_exact_complete_and_idempotent() -> None:
    release = builder.prepare_offline_official_assets()
    assert release == builder.RELEASES_ROOT / EXPECTED_RELEASE_ID
    assert release.is_dir() and not release.is_symlink()
    assert not os.path.lexists(builder.BUILD_ROOT / "current")
    assert list(builder.STAGING_ROOT.iterdir()) == []

    manifest_raw = (release / "asset_manifest.json").read_bytes()
    assert _sha(manifest_raw) == EXPECTED_MANIFEST_SHA256
    assert len(manifest_raw) == builder.EXPECTED_MANIFEST_SIZE == 14362
    manifest = contract.verify_manifest(manifest_raw, release)
    assert manifest["robot_source"]["structure"] == {
        "joint_count": 23,
        "link_count": 24,
        "mesh_reference_count": 42,
        "revolute_joint_count": 14,
        "unique_mesh_count": 20,
    }
    assert manifest["primitive_fallback"] is False
    assert manifest["root_scale"] == 1
    assert manifest["clean_source"] is True

    checkout = release / contract.CHECKOUT_PATH
    assert subprocess.run(
        ["git", "-C", os.fspath(checkout), "symbolic-ref", "-q", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).returncode == 1
    assert subprocess.run(
        ["git", "-C", os.fspath(checkout), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout == ""
    assert not (checkout / ".git" / "shallow").exists()
    assert not (checkout / ".git" / "objects" / "info" / "alternates").exists()

    for path in release.rglob("*"):
        assert not path.is_symlink(), path
        assert path.is_dir() or path.is_file(), path

    marker_raw = (release / "ASSET_COMPLETE.json").read_bytes()
    assert marker_raw == builder._completion_bytes(manifest, EXPECTED_MANIFEST_SHA256)
    assert builder.verify_release(release) == manifest
    assert builder.prepare_offline_official_assets() == release

    assert builder.RESULT_MANIFEST.read_bytes() == manifest_raw
    assert builder.RESULT_COMPLETION.read_bytes() == marker_raw
    for relative in (
        contract.DEPENDENCY_REPORT_PATH,
        contract.LICENSE_PAGE_PATH,
        contract.LICENSE_EVIDENCE_PATH,
    ):
        assert (release / relative).read_bytes() == (
            builder.EVIDENCE_ROOT / relative
        ).read_bytes()
