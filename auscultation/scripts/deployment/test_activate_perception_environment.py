from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import activate_perception_environment as activation  # noqa: E402
import artifact_contract as contract  # noqa: E402
import verify_perception_environment as verifier  # noqa: E402
from test_perception_environment_lock_v3 import _v3_lock  # noqa: E402


AUSC_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = (
    AUSC_ROOT / "configs" / "perception_runtime" / "linbox-ausc-live.lock.json"
)


@pytest.fixture(autouse=True)
def _synthetic_strong_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    global LOCK_PATH
    value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    value["distribution_inventory"]["algorithm"] = (
        verifier.DISTRIBUTION_INVENTORY_ALGORITHM
    )
    value["modules"] = {
        name: {
            "algorithm": verifier.MODULE_TREE_ALGORITHM,
            "count": 1,
            "sha256": identity["sha256"],
        }
        for name, identity in value["modules"].items()
    }
    value["provisioning"]["source_environment_fingerprint_algorithm"] = (
        verifier.publisher.SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
    )
    value["environment_id"] = "linbox-perception-" + (
        verifier.publisher.environment_identity_sha256(value)[:20]
    )
    path = tmp_path / "synthetic-strong.lock.json"
    path.write_bytes(contract.canonical_json_bytes(value))
    monkeypatch.setattr(sys.modules[__name__], "LOCK_PATH", path)


def _lock() -> tuple[dict, str]:
    raw = LOCK_PATH.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _layout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "perception_envs"
    staging = root / ".staging"
    releases = root / "releases"
    current = root / "current"
    staging.mkdir(parents=True)
    releases.mkdir()
    return root, staging, releases, current


def _write_staged_environment(staging: Path) -> tuple[Path, str, str]:
    lock, lock_sha256 = _lock()
    environment_id = lock["environment_id"]
    candidate = staging / "gate-env-test-candidate"
    (candidate / "bin").mkdir(parents=True)
    python = candidate / "bin" / "python"
    python.write_bytes(b"test interpreter placeholder\n")
    python.chmod(0o755)
    (candidate / "proof").mkdir()
    (candidate / "proof" / "evidence.txt").write_text("preserve me\n", encoding="utf-8")
    (candidate / "proof" / "evidence.txt").chmod(0o666)
    marker = {
        "complete": True,
        "environment_id": environment_id,
        "environment_root": str(candidate),
        "lock_sha256": lock_sha256,
        "observed_distribution_inventory": dict(lock["distribution_inventory"]),
        "python_executable_relative": "bin/python",
        "schema": verifier.CLONE_MARKER_SCHEMA,
        "source_environment_fingerprint_sha256": lock["provisioning"][
            "source_environment_fingerprint_sha256"
        ],
    }
    (candidate / verifier.CLONE_MARKER_NAME).write_bytes(
        contract.canonical_json_bytes(marker)
    )
    return candidate, environment_id, lock_sha256


def _report(root: Path, environment_id: str, lock_sha256: str) -> dict:
    return {
        "environment_id": environment_id,
        "environment_root": str(root),
        "lock_sha256": lock_sha256,
        "schema": verifier.VERIFICATION_SCHEMA,
        "schema_gaps": [],
        "status": "VERIFIED",
    }


def _activate(
    candidate: Path,
    releases: Path,
    current: Path,
    environment_id: str,
    lock_sha256: str,
    *,
    execute: bool,
    expected_current: str | None,
    run_verifier,
    reconcile: bool = False,
) -> dict:
    return activation._activate_at_paths(
        candidate,
        releases_root=releases,
        current_link=current,
        lock_path=LOCK_PATH,
        environment_id=environment_id,
        expected_current=expected_current,
        execute=execute,
        confirm_environment_id=environment_id if execute else None,
        confirm_lock_sha256=lock_sha256 if execute else None,
        run_verifier=run_verifier,
        reconcile=reconcile,
    )


def test_execute_moves_rewrites_freezes_postverifies_then_cas_activates(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    previous_id = "linbox-perception-" + "1" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    calls: list[tuple[Path, bool, str]] = []

    def run_verifier(root: Path, lock_path: Path) -> dict:
        marker = json.loads((root / verifier.CLONE_MARKER_NAME).read_text())
        calls.append(
            (
                root,
                bool(stat.S_IMODE(root.stat().st_mode) & 0o222),
                marker["environment_root"],
            )
        )
        assert lock_path == LOCK_PATH
        return _report(root, environment_id, lock_sha256)

    evidence = _activate(
        candidate,
        releases,
        current,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=previous_id,
        run_verifier=run_verifier,
    )

    final = releases / environment_id
    assert not candidate.exists()
    assert final.is_dir() and not final.is_symlink()
    assert os.readlink(current) == "releases/%s" % environment_id
    marker_raw = (final / verifier.CLONE_MARKER_NAME).read_bytes()
    marker = json.loads(marker_raw.decode("utf-8"))
    assert marker_raw == contract.canonical_json_bytes(marker)
    assert marker["environment_root"] == str(final)
    assert calls == [
        (candidate, True, str(candidate)),
        (final, True, str(final)),
        (final, False, str(final)),
    ]
    for directory, dirnames, filenames in os.walk(final, followlinks=False):
        for name in [".", *dirnames, *filenames]:
            path = Path(directory) if name == "." else Path(directory) / name
            if not path.is_symlink():
                assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0
    assert (final / "proof" / "evidence.txt").read_text() == "preserve me\n"
    activation_evidence_path = (
        staging / "activation_evidence" / (environment_id + ".json")
    )
    persisted_raw = activation_evidence_path.read_bytes()
    persisted = json.loads(persisted_raw.decode("utf-8"))
    assert persisted_raw == contract.canonical_json_bytes(persisted)
    assert evidence == {
        "active_pointer": evidence["active_pointer"],
        "activation_evidence_path": str(activation_evidence_path),
        "environment_id": environment_id,
        "environment_root": str(final),
        "hardlink_policy": {
            "regular_entries": 3,
            "status": "VERIFIED_NO_EXTERNAL_HARDLINKS",
            "unique_inodes": 3,
        },
        "immutability": {
            "non_symlink_entries": 6,
            "status": "VERIFIED_A_W_REMOVED",
        },
        "lock_sha256": lock_sha256,
        "marker_sha256": hashlib.sha256(marker_raw).hexdigest(),
        "mode": "execute",
        "pointer_target": "releases/%s" % environment_id,
        "previous_environment_id": previous_id,
        "previous_pointer": evidence["previous_pointer"],
        "schema": activation.ACTIVATION_EVIDENCE_SCHEMA,
        "status": "ACTIVE",
        "symlink_policy": {
            "internal_relative_symlinks": 0,
            "status": "VERIFIED_NO_DANGLING_OR_ESCAPE",
        },
        "verification": {
            "post_immutable_sha256": activation._canonical_sha256(
                _report(final, environment_id, lock_sha256)
            ),
            "post_move_sha256": activation._canonical_sha256(
                _report(final, environment_id, lock_sha256)
            ),
            "staged_sha256": activation._canonical_sha256(
                _report(candidate, environment_id, lock_sha256)
            ),
        },
    }
    assert evidence["active_pointer"]["environment_id"] == environment_id
    assert evidence["active_pointer"]["target"] == (
        "releases/%s" % environment_id
    )
    assert evidence["previous_pointer"]["environment_id"] == previous_id
    assert evidence["previous_pointer"]["target"] == (
        "releases/%s" % previous_id
    )
    assert persisted == evidence


def test_dry_run_is_read_only_and_does_not_require_execute_confirmations(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    original_marker = (candidate / verifier.CLONE_MARKER_NAME).read_bytes()

    evidence = _activate(
        candidate,
        releases,
        current,
        environment_id,
        lock_sha256,
        execute=False,
        expected_current=None,
        run_verifier=lambda root, _lock_path: _report(
            root, environment_id, lock_sha256
        ),
    )

    assert candidate.is_dir()
    assert not (releases / environment_id).exists()
    assert not os.path.lexists(current)
    assert (candidate / verifier.CLONE_MARKER_NAME).read_bytes() == original_marker
    assert evidence["mode"] == "dry-run"
    assert evidence["status"] == "VERIFIED_DRY_RUN"
    assert evidence["pointer_target"] == "releases/%s" % environment_id
    assert evidence["hardlink_policy"] == {
        "regular_entries": 3,
        "status": "VERIFIED_NO_EXTERNAL_HARDLINKS",
        "unique_inodes": 3,
    }


def test_execute_rejects_confirmation_mismatch_before_mutation(tmp_path: Path) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)

    with pytest.raises(activation.ActivationPolicyError, match="double confirmation"):
        activation._activate_at_paths(
            candidate,
            releases_root=releases,
            current_link=current,
            lock_path=LOCK_PATH,
            environment_id=environment_id,
            expected_current=None,
            execute=True,
            confirm_environment_id=environment_id,
            confirm_lock_sha256="0" * 64,
            run_verifier=lambda root, _lock_path: _report(
                root, environment_id, lock_sha256
            ),
        )

    assert candidate.is_dir()
    assert not (releases / environment_id).exists()
    assert not os.path.lexists(current)


def test_cas_mismatch_keeps_staged_candidate_and_pointer_unchanged(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    actual_id = "linbox-perception-" + "2" * 20
    (releases / actual_id).mkdir()
    current.symlink_to("releases/%s" % actual_id)

    with pytest.raises(activation.ActivationConflictError, match="compare-and-swap"):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=True,
            expected_current="linbox-perception-" + "3" * 20,
            run_verifier=lambda root, _lock_path: _report(
                root, environment_id, lock_sha256
            ),
        )

    assert candidate.is_dir()
    assert not (releases / environment_id).exists()
    assert os.readlink(current) == "releases/%s" % actual_id


def test_post_move_verifier_failure_preserves_release_evidence_and_pointer(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    previous_id = "linbox-perception-" + "4" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    calls = 0

    def fail_after_move(root: Path, _lock_path: Path) -> dict:
        nonlocal calls
        calls += 1
        report = _report(root, environment_id, lock_sha256)
        if calls == 2:
            report["status"] = "VERIFIED_WITH_SCHEMA_GAPS"
            report["schema_gaps"] = ["legacy gap"]
        return report

    with pytest.raises(activation.ActivationVerificationError, match="VERIFIED"):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=True,
            expected_current=previous_id,
            run_verifier=fail_after_move,
        )

    final = releases / environment_id
    assert final.is_dir()
    assert (final / "proof" / "evidence.txt").read_text() == "preserve me\n"
    assert os.readlink(current) == "releases/%s" % previous_id


def test_rejects_symlink_candidate_and_noncanonical_managed_topology(
    tmp_path: Path,
) -> None:
    root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    alias = staging / "candidate-alias"
    alias.symlink_to(candidate.name)

    with pytest.raises(activation.ActivationPolicyError, match="real directory"):
        _activate(
            alias,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=False,
            expected_current=None,
            run_verifier=lambda path, _lock_path: _report(
                path, environment_id, lock_sha256
            ),
        )

    with pytest.raises(activation.ActivationPolicyError, match="canonical topology"):
        activation._activate_at_paths(
            candidate,
            releases_root=root / "other-releases",
            current_link=current,
            lock_path=LOCK_PATH,
            environment_id=environment_id,
            expected_current=None,
            execute=False,
            confirm_environment_id=None,
            confirm_lock_sha256=None,
            run_verifier=lambda path, _lock_path: _report(
                path, environment_id, lock_sha256
            ),
        )


def test_public_entrypoint_is_fixed_to_linbox_perception_namespace() -> None:
    assert activation.LINBOX_PERCEPTION_ROOT == Path(
        "/home/robot/wam_data/perception_envs"
    )
    assert activation.LINBOX_STAGING_ROOT == Path(
        "/home/robot/wam_data/perception_envs/.staging"
    )
    assert activation.LINBOX_RELEASES_ROOT == Path(
        "/home/robot/wam_data/perception_envs/releases"
    )
    assert activation.LINBOX_CURRENT_LINK == Path(
        "/home/robot/wam_data/perception_envs/current"
    )
    assert activation.LINBOX_ACTIVATION_EVIDENCE_ROOT == Path(
        "/home/robot/wam_data/perception_envs/.staging/activation_evidence"
    )


def test_public_v3_execute_requires_a_preverified_rollback_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _v3_lock()
    monkeypatch.setattr(
        activation,
        "_load_exact_lock",
        lambda *_args, **_kwargs: (lock, "a" * 64, Path("/tmp/v3.lock")),
    )
    with pytest.raises(
        activation.ActivationPolicyError, match="rollback lock"
    ):
        activation.activate_perception_environment(
            None,
            lock_path="/tmp/v3.lock",
            environment_id=lock["environment_id"],
            expected_current="linbox-perception-" + "f" * 20,
            execute=True,
            confirm_environment_id=lock["environment_id"],
            confirm_lock_sha256="a" * 64,
        )


def test_activation_rejects_checked_in_legacy_v2_lock() -> None:
    legacy_lock_path = (
        AUSC_ROOT
        / "configs"
        / "perception_runtime"
        / "linbox-ausc-live.lock.json"
    )
    value = json.loads(legacy_lock_path.read_text(encoding="utf-8"))

    with pytest.raises(
        activation.ActivationVerificationError,
        match="strong installed-file byte lock",
    ):
        activation._load_exact_lock(
            legacy_lock_path, value["environment_id"],
        )


def test_activation_rejects_superseded_unshared_byte_lock(
    tmp_path: Path,
) -> None:
    value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    value["distribution_inventory"]["algorithm"] = (
        verifier.LEGACY_BYTE_DISTRIBUTION_INVENTORY_ALGORITHM
    )
    value["provisioning"]["source_environment_fingerprint_algorithm"] = (
        verifier.publisher.LEGACY_BYTE_SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
    )
    value["environment_id"] = "linbox-perception-" + (
        verifier.publisher.environment_identity_sha256(value)[:20]
    )
    path = tmp_path / "superseded-byte.lock.json"
    path.write_bytes(contract.canonical_json_bytes(value))

    with pytest.raises(
        activation.ActivationVerificationError,
        match="strong installed-file byte lock",
    ):
        activation._load_exact_lock(path, value["environment_id"])


def test_verification_report_must_match_exact_v2_identity() -> None:
    lock, lock_sha256 = _lock()
    root = Path("/tmp/example")
    environment_id = lock["environment_id"]

    for mutation in (
        lambda report: report.update(status="VERIFIED_WITH_SCHEMA_GAPS"),
        lambda report: report.update(schema_gaps=["gap"]),
        lambda report: report.update(environment_id="linbox-perception-" + "0" * 20),
        lambda report: report.update(environment_root="/tmp/other"),
        lambda report: report.update(lock_sha256="0" * 64),
    ):
        report = _report(root, environment_id, lock_sha256)
        mutation(report)
        with pytest.raises(activation.ActivationVerificationError):
            activation._validate_verification_report(
                report,
                environment_root=root,
                environment_id=environment_id,
                lock_sha256=lock_sha256,
            )


def test_internal_python_symlink_is_allowed_but_escape_is_rejected(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    python = candidate / "bin" / "python"
    python.unlink()
    (candidate / "bin" / "python3.11").write_bytes(b"interpreter\n")
    python.symlink_to("python3.11")

    evidence = _activate(
        candidate,
        releases,
        current,
        environment_id,
        lock_sha256,
        execute=False,
        expected_current=None,
        run_verifier=lambda root, _lock_path: _report(
            root, environment_id, lock_sha256
        ),
    )
    assert evidence["status"] == "VERIFIED_DRY_RUN"

    python.unlink()
    outside = tmp_path / "outside-python"
    outside.write_bytes(b"outside\n")
    python.symlink_to(outside)
    with pytest.raises(activation.ActivationVerificationError, match="escapes"):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=False,
            expected_current=None,
            run_verifier=lambda root, _lock_path: _report(
                root, environment_id, lock_sha256
            ),
        )


def test_candidate_inode_swap_after_staged_verification_is_rejected(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    previous_id = "linbox-perception-" + "5" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    swapped = False

    def swap_after_report(root: Path, _lock_path: Path) -> dict:
        nonlocal swapped
        report = _report(root, environment_id, lock_sha256)
        if not swapped:
            original = staging / "forensic-original"
            candidate.rename(original)
            shutil.copytree(original, candidate, symlinks=True)
            swapped = True
        return report

    with pytest.raises(activation.ActivationConflictError, match="inode changed"):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=True,
            expected_current=previous_id,
            run_verifier=swap_after_report,
        )

    assert candidate.is_dir()
    assert not (releases / environment_id).exists()
    assert os.readlink(current) == "releases/%s" % previous_id


@pytest.mark.parametrize("target_kind", ["outside", "dangling", "absolute"])
def test_full_tree_symlink_policy_rejects_unsafe_targets(
    tmp_path: Path, target_kind: str,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    link = candidate / "proof" / "unsafe-link"
    outside = tmp_path / "outside-data"
    outside.write_text("mutable\n", encoding="utf-8")
    if target_kind == "outside":
        link.symlink_to(os.path.relpath(outside, link.parent))
    elif target_kind == "absolute":
        link.symlink_to(candidate / "proof" / "evidence.txt")
    else:
        link.symlink_to("missing-target")

    with pytest.raises(activation.ActivationVerificationError, match="symlink"):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=False,
            expected_current=None,
            run_verifier=lambda root, _lock_path: _report(
                root, environment_id, lock_sha256
            ),
        )

    assert candidate.is_dir()
    assert not (releases / environment_id).exists()
    assert not os.path.lexists(current)


def test_full_tree_symlink_policy_allows_internal_relative_chain(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    (candidate / "share").mkdir()
    (candidate / "share" / "target.bin").write_bytes(b"pinned\n")
    (candidate / "proof" / "first-link").symlink_to("../share/target.bin")
    (candidate / "proof" / "second-link").symlink_to("first-link")

    evidence = _activate(
        candidate,
        releases,
        current,
        environment_id,
        lock_sha256,
        execute=False,
        expected_current=None,
        run_verifier=lambda root, _lock_path: _report(
            root, environment_id, lock_sha256
        ),
    )

    assert evidence["symlink_policy"] == {
        "internal_relative_symlinks": 2,
        "status": "VERIFIED_NO_DANGLING_OR_ESCAPE",
    }


def test_symlink_escape_injected_after_move_fails_before_pointer_cas(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    previous_id = "linbox-perception-" + "6" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    calls = 0

    def inject_after_move(root: Path, _lock_path: Path) -> dict:
        nonlocal calls
        calls += 1
        report = _report(root, environment_id, lock_sha256)
        if calls == 2:
            (root / "proof" / "late-escape").symlink_to(tmp_path / "outside")
        return report

    with pytest.raises(activation.ActivationVerificationError, match="symlink"):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=True,
            expected_current=previous_id,
            run_verifier=inject_after_move,
        )

    assert (releases / environment_id / "proof" / "evidence.txt").is_file()
    assert os.readlink(current) == "releases/%s" % previous_id


def test_successful_activation_is_idempotent_after_candidate_was_moved(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    previous_id = "linbox-perception-" + "7" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    runner = lambda root, _lock_path: _report(root, environment_id, lock_sha256)

    first = _activate(
        candidate,
        releases,
        current,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=previous_id,
        run_verifier=runner,
    )
    assert not candidate.exists()

    retried = _activate(
        candidate,
        releases,
        current,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=previous_id,
        run_verifier=runner,
    )

    assert retried == first
    assert os.readlink(current) == "releases/%s" % environment_id


def test_idempotent_retry_rejects_stale_active_evidence_without_hardlink_audit(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    previous_id = "linbox-perception-" + "c" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    runner = lambda root, _lock_path: _report(root, environment_id, lock_sha256)
    active = _activate(
        candidate,
        releases,
        current,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=previous_id,
        run_verifier=runner,
    )
    evidence_path = Path(active["activation_evidence_path"])
    stale = dict(active)
    stale.pop("hardlink_policy")
    activation._write_activation_evidence(evidence_path, stale)

    with pytest.raises(
        activation.ActivationVerificationError, match="hardlink audit",
    ):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=True,
            expected_current=previous_id,
            run_verifier=runner,
        )


def test_existing_verified_final_requires_explicit_reconcile_before_cas(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    previous_id = "linbox-perception-" + "8" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    final = releases / environment_id
    candidate.rename(final)
    marker = json.loads((final / verifier.CLONE_MARKER_NAME).read_text())
    marker["environment_root"] = str(final)
    (final / verifier.CLONE_MARKER_NAME).write_bytes(
        contract.canonical_json_bytes(marker)
    )
    activation._make_tree_immutable(final)
    runner = lambda root, _lock_path: _report(root, environment_id, lock_sha256)

    with pytest.raises(activation.ActivationConflictError, match="reconcile"):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=True,
            expected_current=previous_id,
            run_verifier=runner,
        )
    assert os.readlink(current) == "releases/%s" % previous_id

    evidence = _activate(
        candidate,
        releases,
        current,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=previous_id,
        run_verifier=runner,
        reconcile=True,
    )
    assert evidence["mode"] == "reconcile"
    assert evidence["status"] == "ACTIVE"
    assert os.readlink(current) == "releases/%s" % environment_id


def test_active_evidence_fsync_failure_rolls_pointer_back_and_keeps_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    previous_id = "linbox-perception-" + "9" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    real_write = activation._write_activation_evidence

    def fail_active(path: Path, value: dict) -> None:
        if value.get("status") == "ACTIVE":
            raise OSError("injected evidence fsync failure")
        real_write(path, value)

    monkeypatch.setattr(activation, "_write_activation_evidence", fail_active)
    with pytest.raises(OSError, match="injected evidence"):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=True,
            expected_current=previous_id,
            run_verifier=lambda root, _lock_path: _report(
                root, environment_id, lock_sha256
            ),
        )

    assert (releases / environment_id / "proof" / "evidence.txt").is_file()
    assert os.readlink(current) == "releases/%s" % previous_id
    journal_path = staging / "activation_evidence" / (environment_id + ".json")
    journal = json.loads(journal_path.read_text())
    assert journal["status"] in {"PREPARED", "ROLLED_BACK"}


def test_retry_repairs_prepared_evidence_after_pointer_already_switched(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    previous_id = "linbox-perception-" + "a" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    runner = lambda root, _lock_path: _report(root, environment_id, lock_sha256)
    active = _activate(
        candidate,
        releases,
        current,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=previous_id,
        run_verifier=runner,
    )
    evidence_path = Path(active["activation_evidence_path"])
    prepared = activation._prepared_evidence(active)
    activation._write_activation_evidence(evidence_path, prepared)

    repaired = _activate(
        candidate,
        releases,
        current,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=previous_id,
        run_verifier=runner,
    )

    assert repaired["mode"] == "idempotent"
    assert repaired["status"] == "ACTIVE"
    assert json.loads(evidence_path.read_text()) == repaired
    assert os.readlink(current) == "releases/%s" % environment_id


def test_external_hardlink_is_rejected_before_publication(tmp_path: Path) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    os.link(candidate / "proof/evidence.txt", tmp_path / "external-hardlink")

    with pytest.raises(
        activation.ActivationVerificationError, match="external hardlink"
    ):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=False,
            expected_current=None,
            run_verifier=lambda root, _lock_path: _report(
                root, environment_id, lock_sha256
            ),
        )
    assert candidate.is_dir()
    assert not (releases / environment_id).exists()


def test_internal_hardlinks_are_rejected_by_strict_single_link_policy(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    os.link(
        candidate / "proof/evidence.txt",
        candidate / "proof/internal-hardlink.txt",
    )

    with pytest.raises(activation.ActivationVerificationError, match="hardlink"):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=False,
            expected_current=None,
            run_verifier=lambda root, _lock_path: _report(
                root, environment_id, lock_sha256
            ),
        )


def test_external_hardlink_injected_after_move_fails_before_cas(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    candidate, environment_id, lock_sha256 = _write_staged_environment(staging)
    previous_id = "linbox-perception-" + "b" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    calls = 0

    def inject(root: Path, _lock_path: Path) -> dict:
        nonlocal calls
        calls += 1
        report = _report(root, environment_id, lock_sha256)
        if calls == 2:
            os.link(root / "proof/evidence.txt", tmp_path / "late-hardlink")
        return report

    with pytest.raises(
        activation.ActivationVerificationError, match="external hardlink"
    ):
        _activate(
            candidate,
            releases,
            current,
            environment_id,
            lock_sha256,
            execute=True,
            expected_current=previous_id,
            run_verifier=inject,
        )
    assert (releases / environment_id / "proof/evidence.txt").is_file()
    assert os.readlink(current) == "releases/%s" % previous_id


def _write_final_v3_environment(
    releases: Path, tmp_path: Path,
) -> tuple[Path, Path, str, str]:
    lock = _v3_lock()
    environment_id = lock["environment_id"]
    lock_path = tmp_path / "v3.lock.json"
    lock_path.write_bytes(contract.canonical_json_bytes(lock))
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    final = releases / environment_id
    (final / "bin").mkdir(parents=True)
    python = final / "bin/python"
    python.write_bytes(b"final-path interpreter\n")
    python.chmod(0o555)
    marker = {
        "complete": True,
        "environment_id": environment_id,
        "environment_root": str(final),
        "lock_sha256": lock_sha256,
        "observed_environment_tree": verifier.environment_tree_summary(
            lock["environment_tree"]
        ),
        "observed_record_observation": verifier.record_observation_summary(
            lock["record_observation"]
        ),
        "python_executable_relative": "bin/python",
        "required_service_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
        "schema": verifier.RELOCATION_AWARE_CLONE_MARKER_SCHEMA,
        "source_environment_fingerprint_sha256": lock["provisioning"][
            "source_environment_fingerprint_sha256"
        ],
    }
    (final / verifier.CLONE_MARKER_NAME).write_bytes(
        contract.canonical_json_bytes(marker)
    )
    for directory, dirnames, filenames in os.walk(final, topdown=False):
        for name in filenames:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        for name in dirnames:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    final.chmod(stat.S_IMODE(final.stat().st_mode) & ~0o222)
    return final, lock_path, environment_id, lock_sha256


def _activate_v3(
    final: Path,
    releases: Path,
    current: Path,
    lock_path: Path,
    environment_id: str,
    lock_sha256: str,
    *,
    execute: bool,
    expected_current: str | None,
    run_verifier,
    rollback_lock_path: Path | None = None,
    confirm_rollback_environment_id: str | None = None,
    confirm_rollback_lock_sha256: str | None = None,
) -> dict:
    return activation._activate_at_paths(
        final,
        releases_root=releases,
        current_link=current,
        lock_path=lock_path,
        environment_id=environment_id,
        expected_current=expected_current,
        execute=execute,
        confirm_environment_id=environment_id if execute else None,
        confirm_lock_sha256=lock_sha256 if execute else None,
        run_verifier=run_verifier,
        rollback_lock_path=rollback_lock_path,
        confirm_rollback_environment_id=confirm_rollback_environment_id,
        confirm_rollback_lock_sha256=confirm_rollback_lock_sha256,
    )


def _write_final_v2_environment(
    releases: Path,
) -> tuple[Path, str, str]:
    lock, lock_sha256 = _lock()
    environment_id = lock["environment_id"]
    final = releases / environment_id
    (final / "bin").mkdir(parents=True)
    python = final / "bin/python"
    python.write_bytes(b"rollback interpreter\n")
    python.chmod(0o555)
    marker = {
        "complete": True,
        "environment_id": environment_id,
        "environment_root": str(final),
        "lock_sha256": lock_sha256,
        "observed_distribution_inventory": dict(lock["distribution_inventory"]),
        "python_executable_relative": "bin/python",
        "schema": verifier.CLONE_MARKER_SCHEMA,
        "source_environment_fingerprint_sha256": lock["provisioning"][
            "source_environment_fingerprint_sha256"
        ],
    }
    (final / verifier.CLONE_MARKER_NAME).write_bytes(
        contract.canonical_json_bytes(marker)
    )
    for path in sorted(
        [final, *final.rglob("*")], key=lambda item: len(item.parts), reverse=True
    ):
        if not path.is_symlink():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    return final, environment_id, lock_sha256


def test_v3_execute_verifies_final_release_then_only_cas_switches_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, _staging, releases, current = _layout(tmp_path)
    final, lock_path, environment_id, lock_sha256 = _write_final_v3_environment(
        releases, tmp_path
    )
    previous_id = "linbox-perception-" + "c" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    before_inode = final.stat().st_ino
    before_marker = (final / verifier.CLONE_MARKER_NAME).read_bytes()
    before_modes = {
        path.relative_to(final).as_posix(): stat.S_IMODE(path.lstat().st_mode)
        for path in [final, *final.rglob("*")]
        if not path.is_symlink()
    }
    calls: list[Path] = []

    def run_verifier(root: Path, exact_lock: Path) -> dict:
        assert os.readlink(current) == "releases/%s" % previous_id
        assert root == final
        assert exact_lock == lock_path
        calls.append(root)
        return _report(root, environment_id, lock_sha256)

    monkeypatch.setattr(
        activation,
        "_rename_noreplace_between",
        lambda *_args, **_kwargs: pytest.fail("v3 activation must not rename"),
    )
    monkeypatch.setattr(
        activation,
        "_rewrite_marker_root",
        lambda *_args, **_kwargs: pytest.fail("v3 activation must not rewrite marker"),
    )
    monkeypatch.setattr(
        activation,
        "_make_tree_immutable",
        lambda *_args, **_kwargs: pytest.fail("v3 activation must not freeze"),
    )

    evidence = _activate_v3(
        final,
        releases,
        current,
        lock_path,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=previous_id,
        run_verifier=run_verifier,
    )

    assert calls == [final]
    assert os.readlink(current) == "releases/%s" % environment_id
    assert final.stat().st_ino == before_inode
    assert (final / verifier.CLONE_MARKER_NAME).read_bytes() == before_marker
    assert {
        path.relative_to(final).as_posix(): stat.S_IMODE(path.lstat().st_mode)
        for path in [final, *final.rglob("*")]
        if not path.is_symlink()
    } == before_modes
    assert evidence["mode"] == "execute-v3-final-path"
    assert evidence["status"] == "ACTIVE"
    assert set(evidence["verification"]) == {"final_sha256"}
    assert evidence["previous_pointer"]["environment_id"] == previous_id
    assert evidence["previous_pointer"]["target"] == (
        "releases/%s" % previous_id
    )
    assert set(evidence["previous_pointer"]["identity"]) == {
        "device", "inode", "mode", "size",
    }
    assert evidence["active_pointer"]["environment_id"] == environment_id
    assert evidence["active_pointer"]["target"] == (
        "releases/%s" % environment_id
    )
    assert set(evidence["active_pointer"]["identity"]) == {
        "device", "inode", "mode", "size",
    }


def test_post_activation_rollback_requires_active_evidence_and_strong_previous(
    tmp_path: Path,
) -> None:
    root, staging, releases, current = _layout(tmp_path)
    previous, previous_id, previous_lock_sha256 = _write_final_v2_environment(
        releases
    )
    current.symlink_to("releases/%s" % previous_id)
    final, lock_path, environment_id, lock_sha256 = _write_final_v3_environment(
        releases, tmp_path
    )

    _activate_v3(
        final,
        releases,
        current,
        lock_path,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=previous_id,
        run_verifier=lambda root, _path: _report(
            root, environment_id, lock_sha256
        ),
    )
    evidence_path = staging / "activation_evidence" / (environment_id + ".json")
    active_raw = evidence_path.read_bytes()
    active_sha256 = hashlib.sha256(active_raw).hexdigest()

    dry_run = activation._rollback_at_paths(
        releases_root=releases,
        current_link=current,
        activated_environment_id=environment_id,
        previous_lock_path=LOCK_PATH,
        execute=False,
        confirm_activation_evidence_sha256=None,
        confirm_previous_environment_id=None,
        confirm_previous_lock_sha256=None,
        run_verifier=lambda root, exact_lock: (
            _report(root, previous_id, previous_lock_sha256)
            if root == previous and exact_lock == LOCK_PATH
            else pytest.fail("rollback verifier input drift")
        ),
    )
    assert dry_run["status"] == "VERIFIED_ROLLBACK_DRY_RUN"
    assert os.readlink(current) == "releases/%s" % environment_id
    assert evidence_path.read_bytes() == active_raw

    rolled_back = activation._rollback_at_paths(
        releases_root=releases,
        current_link=current,
        activated_environment_id=environment_id,
        previous_lock_path=LOCK_PATH,
        execute=True,
        confirm_activation_evidence_sha256=active_sha256,
        confirm_previous_environment_id=previous_id,
        confirm_previous_lock_sha256=previous_lock_sha256,
        run_verifier=lambda root, exact_lock: (
            _report(root, previous_id, previous_lock_sha256)
            if root == previous and exact_lock == LOCK_PATH
            else pytest.fail("rollback verifier input drift")
        ),
    )

    assert os.readlink(current) == "releases/%s" % previous_id
    assert rolled_back["schema"] == activation.ROLLBACK_EVIDENCE_SCHEMA
    assert rolled_back["status"] == "ROLLED_BACK"
    assert rolled_back["activation_evidence_sha256"] == active_sha256
    assert rolled_back["activation"] == json.loads(active_raw)
    assert rolled_back["restored_pointer"]["environment_id"] == previous_id
    assert rolled_back["restored_pointer"]["target"] == (
        "releases/%s" % previous_id
    )
    assert json.loads(evidence_path.read_bytes()) == rolled_back


def test_v3_activation_can_bind_a_preverified_strong_rollback_clone(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    rollback_clone, rollback_id, rollback_lock_sha256 = (
        _write_final_v2_environment(releases)
    )
    weak_previous_id = "linbox-perception-" + "e" * 20
    (releases / weak_previous_id).mkdir()
    current.symlink_to("releases/%s" % weak_previous_id)
    final, lock_path, environment_id, lock_sha256 = _write_final_v3_environment(
        releases, tmp_path
    )

    def run_verifier(root: Path, exact_lock: Path) -> dict:
        if root == final and exact_lock == lock_path:
            return _report(root, environment_id, lock_sha256)
        if root == rollback_clone and exact_lock == LOCK_PATH:
            return _report(root, rollback_id, rollback_lock_sha256)
        return pytest.fail("unexpected activation verifier input")

    active = _activate_v3(
        final,
        releases,
        current,
        lock_path,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=weak_previous_id,
        run_verifier=run_verifier,
        rollback_lock_path=LOCK_PATH,
        confirm_rollback_environment_id=rollback_id,
        confirm_rollback_lock_sha256=rollback_lock_sha256,
    )
    assert active["previous_pointer"]["environment_id"] == weak_previous_id
    assert active["rollback_target"]["environment_id"] == rollback_id
    assert active["rollback_target"]["lock_sha256"] == rollback_lock_sha256
    assert os.readlink(current) == "releases/%s" % environment_id

    evidence_path = staging / "activation_evidence" / (environment_id + ".json")
    rolled_back = activation._rollback_at_paths(
        releases_root=releases,
        current_link=current,
        activated_environment_id=environment_id,
        previous_lock_path=LOCK_PATH,
        execute=True,
        confirm_activation_evidence_sha256=hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest(),
        confirm_previous_environment_id=rollback_id,
        confirm_previous_lock_sha256=rollback_lock_sha256,
        run_verifier=run_verifier,
    )
    assert os.readlink(current) == "releases/%s" % rollback_id
    assert rolled_back["previous_environment_id"] == rollback_id
    assert rolled_back["activation"]["previous_environment_id"] == (
        weak_previous_id
    )


def test_v3_activation_rejects_rollback_clone_verifier_failure_before_cas(
    tmp_path: Path,
) -> None:
    _root, _staging, releases, current = _layout(tmp_path)
    rollback_clone, rollback_id, rollback_lock_sha256 = (
        _write_final_v2_environment(releases)
    )
    weak_previous_id = "linbox-perception-" + "e" * 20
    (releases / weak_previous_id).mkdir()
    current.symlink_to("releases/%s" % weak_previous_id)
    final, lock_path, environment_id, lock_sha256 = _write_final_v3_environment(
        releases, tmp_path
    )

    def run_verifier(root: Path, exact_lock: Path) -> dict:
        if root == final and exact_lock == lock_path:
            return _report(root, environment_id, lock_sha256)
        assert root == rollback_clone and exact_lock == LOCK_PATH
        failed = _report(root, rollback_id, rollback_lock_sha256)
        failed["status"] = "VERIFIED_WITH_SCHEMA_GAPS"
        failed["schema_gaps"] = ["rollback gap"]
        return failed

    with pytest.raises(
        activation.ActivationVerificationError, match="VERIFIED"
    ):
        _activate_v3(
            final,
            releases,
            current,
            lock_path,
            environment_id,
            lock_sha256,
            execute=True,
            expected_current=weak_previous_id,
            run_verifier=run_verifier,
            rollback_lock_path=LOCK_PATH,
            confirm_rollback_environment_id=rollback_id,
            confirm_rollback_lock_sha256=rollback_lock_sha256,
        )
    assert os.readlink(current) == "releases/%s" % weak_previous_id


def test_post_activation_rollback_rejects_same_target_with_replaced_pointer_inode(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    _previous, previous_id, previous_lock_sha256 = _write_final_v2_environment(
        releases
    )
    current.symlink_to("releases/%s" % previous_id)
    final, lock_path, environment_id, lock_sha256 = _write_final_v3_environment(
        releases, tmp_path
    )
    _activate_v3(
        final,
        releases,
        current,
        lock_path,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=previous_id,
        run_verifier=lambda root, _path: _report(
            root, environment_id, lock_sha256
        ),
    )
    evidence_path = staging / "activation_evidence" / (environment_id + ".json")
    evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    current.unlink()
    current.symlink_to("releases/%s" % environment_id)

    with pytest.raises(
        activation.ActivationConflictError, match="active pointer identity"
    ):
        activation._rollback_at_paths(
            releases_root=releases,
            current_link=current,
            activated_environment_id=environment_id,
            previous_lock_path=LOCK_PATH,
            execute=True,
            confirm_activation_evidence_sha256=evidence_sha256,
            confirm_previous_environment_id=previous_id,
            confirm_previous_lock_sha256=previous_lock_sha256,
            run_verifier=lambda *_args: pytest.fail(
                "pointer drift must fail before previous verification"
            ),
        )
    assert os.readlink(current) == "releases/%s" % environment_id


def test_post_activation_rollback_rejects_v1_previous_lock_without_mutation(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    legacy_lock_path = (
        AUSC_ROOT / "configs/perception_runtime/linbox-ausc-live.lock.json"
    )
    legacy_lock = json.loads(legacy_lock_path.read_text(encoding="utf-8"))
    previous_id = legacy_lock["environment_id"]
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    final, lock_path, environment_id, lock_sha256 = _write_final_v3_environment(
        releases, tmp_path
    )
    _activate_v3(
        final,
        releases,
        current,
        lock_path,
        environment_id,
        lock_sha256,
        execute=True,
        expected_current=previous_id,
        run_verifier=lambda root, _path: _report(
            root, environment_id, lock_sha256
        ),
    )
    evidence_path = staging / "activation_evidence" / (environment_id + ".json")

    with pytest.raises(
        activation.ActivationVerificationError,
        match="strong v2 or v3 lock|strong installed-file byte lock",
    ):
        activation._rollback_at_paths(
            releases_root=releases,
            current_link=current,
            activated_environment_id=environment_id,
            previous_lock_path=legacy_lock_path,
            execute=True,
            confirm_activation_evidence_sha256=hashlib.sha256(
                evidence_path.read_bytes()
            ).hexdigest(),
            confirm_previous_environment_id=previous_id,
            confirm_previous_lock_sha256=hashlib.sha256(
                legacy_lock_path.read_bytes()
            ).hexdigest(),
            run_verifier=lambda *_args: pytest.fail(
                "v1 must fail before verifier"
            ),
        )
    assert os.readlink(current) == "releases/%s" % environment_id
    assert json.loads(evidence_path.read_bytes())["status"] == "ACTIVE"


def test_v3_verifier_failure_leaves_pointer_and_final_release_untouched(
    tmp_path: Path,
) -> None:
    _root, _staging, releases, current = _layout(tmp_path)
    final, lock_path, environment_id, lock_sha256 = _write_final_v3_environment(
        releases, tmp_path
    )
    previous_id = "linbox-perception-" + "d" * 20
    (releases / previous_id).mkdir()
    current.symlink_to("releases/%s" % previous_id)
    before = (final / verifier.CLONE_MARKER_NAME).read_bytes()

    with pytest.raises(RuntimeError, match="live verification failed"):
        _activate_v3(
            final,
            releases,
            current,
            lock_path,
            environment_id,
            lock_sha256,
            execute=True,
            expected_current=previous_id,
            run_verifier=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("live verification failed")
            ),
        )

    assert os.readlink(current) == "releases/%s" % previous_id
    assert (final / verifier.CLONE_MARKER_NAME).read_bytes() == before


def test_v3_rejects_staging_candidate_instead_of_using_legacy_move_path(
    tmp_path: Path,
) -> None:
    _root, staging, releases, current = _layout(tmp_path)
    final, lock_path, environment_id, lock_sha256 = _write_final_v3_environment(
        releases, tmp_path
    )
    staged = staging / "must-not-move"
    shutil.copytree(final, staged, symlinks=True)

    with pytest.raises(
        activation.ActivationPolicyError, match="permanent final release path"
    ):
        _activate_v3(
            staged,
            releases,
            current,
            lock_path,
            environment_id,
            lock_sha256,
            execute=False,
            expected_current=None,
            run_verifier=lambda *_args: pytest.fail("must reject before verifier"),
        )
    assert staged.is_dir()
    assert not os.path.lexists(current)


def test_v3_requires_completion_marker_before_running_verifier(
    tmp_path: Path,
) -> None:
    _root, _staging, releases, current = _layout(tmp_path)
    final, lock_path, environment_id, lock_sha256 = _write_final_v3_environment(
        releases, tmp_path
    )
    final.chmod(0o755)
    marker = final / verifier.CLONE_MARKER_NAME
    marker.chmod(0o644)
    marker.unlink()
    final.chmod(0o555)

    with pytest.raises(
        activation.ActivationVerificationError, match="completion marker"
    ):
        _activate_v3(
            final,
            releases,
            current,
            lock_path,
            environment_id,
            lock_sha256,
            execute=False,
            expected_current=None,
            run_verifier=lambda *_args: pytest.fail("must reject before verifier"),
        )
