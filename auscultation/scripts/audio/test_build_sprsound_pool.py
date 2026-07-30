from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


HERE = Path(__file__).resolve().parent
DEPLOYMENT = HERE.parent / "deployment"
sys.path[:0] = [str(HERE), str(DEPLOYMENT)]

import build_sprsound_pool as pool  # noqa: E402
import preference_annotator as annotator  # noqa: E402


SMALL_QUOTAS = {
    "Poor Quality": 2,
    "Normal": 1,
    "CAS": 1,
    "DAS": 1,
    "CAS & DAS": 1,
}


def _source_identity(pair_count: int) -> dict:
    return {
        "commit": "874eeb8736ddb78937c2fb5332fc7e7293d0f0ca",
        "corpus": "BioCAS2022",
        "pair_count": pair_count,
        "repository": "https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git",
        "schema": "wam_sprsound_source_identity_v1",
    }


def _record(root: Path, directory: str, label: str, body: bytes) -> None:
    target = root / directory / "repeated"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_suffix(".wav").write_bytes(body)
    target.with_suffix(".json").write_text(
        json.dumps({"record_annotation": label}), encoding="utf-8"
    )


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "BioCAS2022"
    rows = [
        ("poor/a", "Poor Quality", b"poor-a"),
        ("poor/b", "poor quality", b"poor-b"),
        ("normal/a", "Normal", b"normal"),
        ("cas/a", "CAS", b"cas"),
        ("das/a", "DAS", b"das"),
        ("both/a", "CAS&DAS", b"both"),
    ]
    for row in rows:
        _record(root, *row)
    return root


def _production_release(home: Path, *, seed: int = 0) -> Path:
    source = {
        "commit": pool.SPRSOUND_COMMIT,
        "corpus": "BioCAS2022",
        "pair_count": pool.SPRSOUND_PAIR_COUNT,
        "repository": pool.SPRSOUND_REPOSITORY,
        "schema": "wam_sprsound_git_source_identity_v2",
        "tree": pool.SPRSOUND_TREE,
    }
    rows = []
    bodies = {}
    index = 0
    for stratum, count in pool.DEFAULT_QUOTAS.items():
        for _ in range(count):
            body = ("production-clip-%03d" % index).encode("ascii")
            content_sha256 = hashlib.sha256(body).hexdigest()
            source_relative = "BioCAS2022/train2022_wav/%03d.wav" % index
            metadata_relative = "BioCAS2022/train2022_json/%03d.json" % index
            source_id = hashlib.sha256(source_relative.encode("utf-8")).hexdigest()
            slug = stratum.lower().replace(" & ", "-").replace(" ", "-")
            destination = "%s-%s-%s.wav" % (
                slug, source_id[:16], content_sha256[:16],
            )
            rows.append({
                "content_sha256": content_sha256,
                "destination": destination,
                "metadata_relative": metadata_relative,
                "metadata_sha256": hashlib.sha256(
                    ("production-meta-%03d" % index).encode("ascii")
                ).hexdigest(),
                "size": len(body),
                "source_id": source_id,
                "source_relative": source_relative,
                "stratum": stratum,
            })
            bodies[destination] = body
            index += 1
    projection = pool._identity_projection(rows, source, pool.DEFAULT_QUOTAS, seed)
    release_id = "sprsound_pool-" + hashlib.sha256(
        pool.contract.canonical_json_bytes(projection)
    ).hexdigest()[:20]
    release = (
        home / "wam_ausc" / "data" / "annotation_pools" / "sprsound"
        / "releases" / release_id
    )
    release.mkdir(parents=True)
    for destination, body in bodies.items():
        (release / destination).write_bytes(body)
    manifest = b"".join(pool.contract.canonical_json_bytes(row) for row in rows)
    (release / pool.MANIFEST_NAME).write_bytes(manifest)
    (release / pool.SOURCE_NAME).write_bytes(
        pool.contract.canonical_json_bytes(source)
    )
    marker = {
        "counts": dict(pool.DEFAULT_QUOTAS),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "quotas": dict(pool.DEFAULT_QUOTAS),
        "release_id": release_id,
        "schema": "wam_sprsound_pool_complete_v1",
        "seed": seed,
        "source_commit": pool.SPRSOUND_COMMIT,
        "total": 400,
    }
    (release / pool.COMPLETE_NAME).write_bytes(
        pool.contract.canonical_json_bytes(marker)
    )
    return release


def _activate_production_for_test(release_id: str, **arguments):
    home = arguments.pop("test_home")
    assert arguments.pop("allow_test_home") is True
    return pool._activate_pool_production_at_home(
        release_id, production_home=home, **arguments,
    )


def _reconcile_production_for_test(**arguments):
    home = arguments.pop("test_home")
    assert arguments.pop("allow_test_home") is True
    return pool._reconcile_pool_production_at_home(production_home=home)


def test_internal_production_paths_keep_the_canonical_layout(tmp_path: Path) -> None:
    paths = pool._production_paths(tmp_path)

    assert paths.root == (
        tmp_path / "wam_ausc" / "data" / "annotation_pools" / "sprsound"
    )
    assert paths.releases == paths.root / "releases"
    assert paths.current == paths.root / "current"
    assert paths.legacy == tmp_path / "wam_ausc" / "data" / "annotation_pool_sprsound"


def test_public_production_apis_expose_no_test_home_or_path_override() -> None:
    activate = inspect.signature(pool.activate_pool_production).parameters
    reconcile = inspect.signature(pool.reconcile_pool_production).parameters

    assert "test_home" not in activate
    assert "allow_test_home" not in activate
    assert "test_home" not in reconcile
    assert "allow_test_home" not in reconcile


def test_public_production_api_ignores_forged_home_and_pytest_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_home = tmp_path / "fixed-production-home"
    attacker = tmp_path / "attacker-home"
    release = _production_release(production_home)
    monkeypatch.setattr(pool, "ANNOTATION_PRODUCTION_HOME", production_home)
    monkeypatch.setenv("HOME", str(attacker))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "forged runtime gate")

    pool.activate_pool_production(
        release.name,
        confirm_release_id=release.name,
        expected_current=None,
    )

    assert (release.parents[1] / "current").resolve() == release.resolve()
    assert not attacker.exists()


def test_production_reconcile_rejects_journal_path_escape_without_mutation(
    tmp_path: Path,
) -> None:
    paths = pool._production_paths(tmp_path)
    paths.root.mkdir(parents=True)
    sentinel = tmp_path / "outside-sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    release_id = "sprsound_pool-" + "a" * 20
    journal = {
        "backup_name": "../../outside-sentinel",
        "legacy_device": 1,
        "legacy_inode": 1,
        "legacy_kind": "directory",
        "legacy_sha256": "b" * 64,
        "previous_current": None,
        "release_id": release_id,
        "schema": "wam_sprsound_production_transaction_v1",
        "stage": "prepared",
    }
    paths.journal.write_bytes(pool.contract.canonical_json_bytes(journal))

    with pytest.raises(pool.PoolContractError, match="backup journal"):
        pool._reconcile_pool_production_at_home(production_home=tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert paths.journal.is_file()


def test_production_activation_requires_explicit_matching_release_id(
    tmp_path: Path,
) -> None:
    release = _production_release(tmp_path)
    legacy = tmp_path / "wam_ausc" / "data" / "annotation_pool_sprsound"
    legacy.mkdir()
    (legacy / "keep.wav").write_bytes(b"legacy")

    with pytest.raises(pool.PoolContractError, match="explicit release confirmation"):
        _activate_production_for_test(
            release.name,
            confirm_release_id="sprsound_pool-" + "0" * 20,
            expected_current=None,
            test_home=tmp_path,
            allow_test_home=True,
        )

    assert legacy.is_dir() and not legacy.is_symlink()
    assert not os.path.lexists(str(release.parents[1] / "current"))


def test_production_verifier_requires_exact_official_v2_400_release(
    tmp_path: Path,
) -> None:
    release = _production_release(tmp_path)

    verified = pool.verify_production_pool_release(
        release, expected_release_id=release.name,
    )

    assert verified["release_id"] == release.name
    assert verified["counts"] == pool.DEFAULT_QUOTAS
    assert verified["total"] == 400

    source_path = release / pool.SOURCE_NAME
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.pop("tree")
    source["schema"] = "wam_sprsound_source_identity_v1"
    source_path.write_bytes(pool.contract.canonical_json_bytes(source))
    with pytest.raises(pool.PoolContractError, match="Git-object|official|production"):
        pool.verify_production_pool_release(
            release, expected_release_id=release.name,
        )


def test_production_activation_migrates_legacy_and_publishes_both_pointers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _production_release(tmp_path)
    data = tmp_path / "wam_ausc" / "data"
    legacy = data / "annotation_pool_sprsound"
    (legacy / "nested").mkdir(parents=True)
    (legacy / "old.wav").write_bytes(b"legacy-audio")
    (legacy / "nested" / "note.json").write_bytes(b'{"legacy":true}\n')
    monkeypatch.setattr(
        pool, "_production_timestamp", lambda: "20260712T120000Z",
    )

    result = _activate_production_for_test(
        release.name,
        confirm_release_id=release.name,
        expected_current=None,
        test_home=tmp_path,
        allow_test_home=True,
    )

    backup = Path(result["legacy_backup"])
    assert backup.name == (
        "annotation_pool_sprsound.legacy-20260712T120000Z-"
        + result["legacy_sha256"]
    )
    assert (backup / "old.wav").read_bytes() == b"legacy-audio"
    assert (backup / "nested" / "note.json").read_bytes() == b'{"legacy":true}\n'
    current = release.parents[1] / "current"
    assert os.readlink(current) == "releases/%s" % release.name
    assert current.resolve() == release.resolve()
    assert legacy.is_symlink()
    assert os.readlink(legacy) == "annotation_pools/sprsound/current"
    assert legacy.resolve() == release.resolve()
    assert result["previous_current"] is None
    assert result["release_id"] == release.name
    assert result["schema"] == "wam_sprsound_production_activation_v1"
    pool.verify_production_pool_release(
        current.resolve(), expected_release_id=release.name,
    )


def test_production_legacy_backup_is_content_bound_and_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _production_release(tmp_path)
    data = tmp_path / "wam_ausc" / "data"
    legacy = data / "annotation_pool_sprsound"
    legacy.mkdir()
    (legacy / "keep.wav").write_bytes(b"legacy")
    descriptor = os.open(legacy, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        digest = pool._legacy_inventory_digest(descriptor)
    finally:
        os.close(descriptor)
    monkeypatch.setattr(
        pool, "_production_timestamp", lambda: "20260712T120001Z",
    )
    collision = data / (
        "annotation_pool_sprsound.legacy-20260712T120001Z-%s" % digest
    )
    collision.mkdir()
    (collision / "sentinel").write_text("do-not-overwrite", encoding="utf-8")

    with pytest.raises(Exception, match="exists|overwrite|publication"):
        _activate_production_for_test(
            release.name,
            confirm_release_id=release.name,
            expected_current=None,
            test_home=tmp_path,
            allow_test_home=True,
        )

    assert (legacy / "keep.wav").read_bytes() == b"legacy"
    assert (collision / "sentinel").read_text(encoding="utf-8") == "do-not-overwrite"
    assert not os.path.lexists(str(release.parents[1] / "current"))


@pytest.mark.parametrize("unsafe_kind", ["symlink", "fifo"])
def test_production_legacy_inventory_rejects_links_and_special_entries(
    tmp_path: Path, unsafe_kind: str,
) -> None:
    release = _production_release(tmp_path)
    legacy = tmp_path / "wam_ausc" / "data" / "annotation_pool_sprsound"
    legacy.mkdir()
    if unsafe_kind == "symlink":
        (legacy / "unsafe").symlink_to("missing")
    else:
        os.mkfifo(legacy / "unsafe")

    with pytest.raises(pool.PoolContractError, match="symlink|special"):
        _activate_production_for_test(
            release.name,
            confirm_release_id=release.name,
            expected_current=None,
            test_home=tmp_path,
            allow_test_home=True,
        )

    assert legacy.is_dir() and not legacy.is_symlink()
    assert not os.path.lexists(str(release.parents[1] / "current"))


def test_production_activation_rejects_a_symlinked_home_ancestor(tmp_path: Path) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    fake_home = tmp_path / "fake-home"
    fake_home.symlink_to(real_home, target_is_directory=True)
    release = _production_release(fake_home)

    with pytest.raises(pool.PoolContractError, match="symlink|real directory"):
        _activate_production_for_test(
            release.name,
            confirm_release_id=release.name,
            expected_current=None,
            test_home=fake_home,
            allow_test_home=True,
        )

    assert not os.path.lexists(str(release.parents[1] / "current"))


@pytest.mark.parametrize(
    "fault_stage",
    [
        "after_legacy_migration",
        "after_current_switch",
        "after_compat_switch",
        "after_postverify",
    ],
)
def test_production_failure_restores_legacy_directory_and_previous_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_stage: str,
) -> None:
    previous = _production_release(tmp_path, seed=1)
    candidate = _production_release(tmp_path, seed=2)
    root = candidate.parents[1]
    current = root / "current"
    current.symlink_to("releases/%s" % previous.name)
    legacy = tmp_path / "wam_ausc" / "data" / "annotation_pool_sprsound"
    legacy.mkdir()
    (legacy / "keep.wav").write_bytes(b"legacy-before-activation")
    monkeypatch.setattr(
        pool, "_production_timestamp", lambda: "20260712T120002Z",
    )

    def fail_at(stage: str) -> None:
        if stage == fault_stage:
            raise RuntimeError("injected production failure: %s" % stage)

    monkeypatch.setattr(pool, "_production_checkpoint", fail_at)

    with pytest.raises(RuntimeError, match="injected production failure"):
        _activate_production_for_test(
            candidate.name,
            confirm_release_id=candidate.name,
            expected_current=previous.name,
            test_home=tmp_path,
            allow_test_home=True,
        )

    assert legacy.is_dir() and not legacy.is_symlink()
    assert (legacy / "keep.wav").read_bytes() == b"legacy-before-activation"
    assert os.readlink(current) == "releases/%s" % previous.name
    assert current.resolve() == previous.resolve()
    assert not list(legacy.parent.glob("annotation_pool_sprsound.legacy-*"))
    assert not (root / ".production-activation.json").exists()


def test_production_failure_preserves_previous_compatibility_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = _production_release(tmp_path, seed=3)
    candidate = _production_release(tmp_path, seed=4)
    legacy = tmp_path / "wam_ausc" / "data" / "annotation_pool_sprsound"
    legacy.mkdir()
    (legacy / "keep.wav").write_bytes(b"legacy")
    monkeypatch.setattr(
        pool, "_production_timestamp", lambda: "20260712T120003Z",
    )
    _activate_production_for_test(
        previous.name,
        confirm_release_id=previous.name,
        expected_current=None,
        test_home=tmp_path,
        allow_test_home=True,
    )
    backup = next(legacy.parent.glob("annotation_pool_sprsound.legacy-*"))

    def fail_after_current(stage: str) -> None:
        if stage == "after_current_switch":
            raise RuntimeError("injected second activation failure")

    monkeypatch.setattr(pool, "_production_checkpoint", fail_after_current)
    with pytest.raises(RuntimeError, match="second activation"):
        _activate_production_for_test(
            candidate.name,
            confirm_release_id=candidate.name,
            expected_current=previous.name,
            test_home=tmp_path,
            allow_test_home=True,
        )

    current = candidate.parents[1] / "current"
    assert os.readlink(current) == "releases/%s" % previous.name
    assert current.resolve() == previous.resolve()
    assert legacy.is_symlink()
    assert os.readlink(legacy) == "annotation_pools/sprsound/current"
    assert legacy.resolve() == previous.resolve()
    assert backup.is_dir()


def test_production_reconcile_rolls_back_an_interrupted_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("requires fork to simulate an uncatchable interruption")
    previous = _production_release(tmp_path, seed=5)
    candidate = _production_release(tmp_path, seed=6)
    root = candidate.parents[1]
    current = root / "current"
    current.symlink_to("releases/%s" % previous.name)
    legacy = tmp_path / "wam_ausc" / "data" / "annotation_pool_sprsound"
    legacy.mkdir()
    (legacy / "keep.wav").write_bytes(b"legacy-before-crash")
    monkeypatch.setattr(
        pool, "_production_timestamp", lambda: "20260712T120004Z",
    )

    def crash_after_current(stage: str) -> None:
        if stage == "after_current_switch":
            os._exit(92)

    monkeypatch.setattr(pool, "_production_checkpoint", crash_after_current)
    child = os.fork()
    if child == 0:
        _activate_production_for_test(
            candidate.name,
            confirm_release_id=candidate.name,
            expected_current=previous.name,
            test_home=tmp_path,
            allow_test_home=True,
        )
        os._exit(0)
    _pid, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 92
    assert (root / ".production-activation.json").is_file()
    assert not os.path.lexists(str(legacy))
    assert current.resolve() == candidate.resolve()

    reconciled = _reconcile_production_for_test(
        test_home=tmp_path, allow_test_home=True,
    )

    assert reconciled["status"] == "rolled_back"
    assert legacy.is_dir() and not legacy.is_symlink()
    assert (legacy / "keep.wav").read_bytes() == b"legacy-before-crash"
    assert current.resolve() == previous.resolve()
    assert not (root / ".production-activation.json").exists()


def test_production_reconcile_finishes_a_durably_committed_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("requires fork to simulate an uncatchable interruption")
    candidate = _production_release(tmp_path, seed=7)
    root = candidate.parents[1]
    legacy = tmp_path / "wam_ausc" / "data" / "annotation_pool_sprsound"
    legacy.mkdir()
    (legacy / "keep.wav").write_bytes(b"legacy-before-commit")
    monkeypatch.setattr(
        pool, "_production_timestamp", lambda: "20260712T120005Z",
    )

    def crash_after_commit_record(stage: str) -> None:
        if stage == "after_commit_record":
            os._exit(93)

    monkeypatch.setattr(pool, "_production_checkpoint", crash_after_commit_record)
    child = os.fork()
    if child == 0:
        _activate_production_for_test(
            candidate.name,
            confirm_release_id=candidate.name,
            expected_current=None,
            test_home=tmp_path,
            allow_test_home=True,
        )
        os._exit(0)
    _pid, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 93
    assert (root / ".production-activation.json").is_file()
    assert (root / "current").resolve() == candidate.resolve()
    assert legacy.is_symlink() and legacy.resolve() == candidate.resolve()

    reconciled = _reconcile_production_for_test(
        test_home=tmp_path, allow_test_home=True,
    )

    assert reconciled["status"] == "committed"
    assert (root / "current").resolve() == candidate.resolve()
    assert legacy.is_symlink() and legacy.resolve() == candidate.resolve()
    assert next(legacy.parent.glob("annotation_pool_sprsound.legacy-*")).is_dir()
    assert not (root / ".production-activation.json").exists()


def test_production_activation_fails_closed_when_another_writer_holds_lock(
    tmp_path: Path,
) -> None:
    release = _production_release(tmp_path, seed=8)
    paths = pool._production_paths(tmp_path)

    with pool._production_lock(paths):
        with pytest.raises(pool.PoolContractError, match="already held"):
            _activate_production_for_test(
                release.name,
                confirm_release_id=release.name,
                expected_current=None,
                test_home=tmp_path,
                allow_test_home=True,
            )

    assert not os.path.lexists(str(paths.current))
    assert not os.path.lexists(str(paths.legacy))


def test_production_current_same_target_replacement_inode_is_not_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = _production_release(tmp_path, seed=9)
    candidate = _production_release(tmp_path, seed=10)
    paths = pool._production_paths(tmp_path)
    paths.current.symlink_to("releases/%s" % previous.name)
    original_write = pool._write_production_journal
    replacement_identity = {}

    def write_then_replace_current(journal_paths, journal):
        original_write(journal_paths, journal)
        if not replacement_identity:
            target = os.readlink(paths.current)
            paths.current.unlink()
            paths.current.symlink_to(target)
            info = paths.current.lstat()
            replacement_identity["value"] = (
                info.st_dev,
                info.st_ino,
                stat.S_IFMT(info.st_mode),
                info.st_mtime_ns,
                info.st_ctime_ns,
            )

    monkeypatch.setattr(pool, "_write_production_journal", write_then_replace_current)
    with pytest.raises(pool.PoolContractError, match="compare-and-swap|inode|identity"):
        _activate_production_for_test(
            candidate.name,
            confirm_release_id=candidate.name,
            expected_current=previous.name,
            test_home=tmp_path,
            allow_test_home=True,
        )

    info = paths.current.lstat()
    assert (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_mtime_ns,
        info.st_ctime_ns,
    ) == replacement_identity["value"]
    assert os.readlink(paths.current) == "releases/%s" % previous.name


def test_exchange_precheck_rejects_same_target_inode_without_syscall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "exchange-precheck"
    parent.mkdir()
    prepared = parent / "prepared"
    current = parent / "current"
    prepared.symlink_to("candidate-target")
    current.symlink_to("previous-target")
    descriptor = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    syscall_calls = []
    try:
        prepared_identity = pool._path_identity_at(descriptor, prepared.name)
        expected_current_identity = pool._path_identity_at(descriptor, current.name)
        target = os.readlink(current)
        current.unlink()
        current.symlink_to(target)
        replacement = current.lstat()
        replacement_identity = (
            replacement.st_dev,
            replacement.st_ino,
            stat.S_IFMT(replacement.st_mode),
            replacement.st_size,
            replacement.st_mtime_ns,
            replacement.st_ctime_ns,
        )

        def forbid_exchange_syscall(*_args, **_kwargs):
            syscall_calls.append(True)
            raise AssertionError("exchange syscall must not run after identity mismatch")

        monkeypatch.setattr(pool.ctypes, "CDLL", forbid_exchange_syscall)
        with pytest.raises(pool.PoolContractError, match="identity|inode|changed"):
            pool._exchange_at(
                descriptor,
                prepared.name,
                current.name,
                expected_first_identity=prepared_identity,
                expected_second_identity=expected_current_identity,
            )
    finally:
        os.close(descriptor)

    after = current.lstat()
    assert syscall_calls == []
    assert (
        after.st_dev,
        after.st_ino,
        stat.S_IFMT(after.st_mode),
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) == replacement_identity
    assert os.readlink(current) == "previous-target"


def test_production_legacy_name_swap_does_not_move_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _production_release(tmp_path, seed=11)
    paths = pool._production_paths(tmp_path)
    paths.legacy.mkdir()
    (paths.legacy / "original.wav").write_bytes(b"held-original")
    held_original = paths.data / "attacker-held-original"
    original_write = pool._write_production_journal
    swapped = {"value": False}

    def write_then_swap_legacy(journal_paths, journal):
        original_write(journal_paths, journal)
        value = journal.get("payload", {}).get("request", journal)
        legacy = value.get("legacy", value)
        if legacy.get("legacy_kind") == "directory" and not swapped["value"]:
            paths.legacy.rename(held_original)
            paths.legacy.mkdir()
            (paths.legacy / "replacement.wav").write_bytes(b"third-party")
            swapped["value"] = True

    monkeypatch.setattr(pool, "_write_production_journal", write_then_swap_legacy)
    with pytest.raises(pool.PoolContractError, match="legacy|inode|changed|rollback"):
        _activate_production_for_test(
            candidate.name,
            confirm_release_id=candidate.name,
            expected_current=None,
            test_home=tmp_path,
            allow_test_home=True,
        )

    assert (held_original / "original.wav").read_bytes() == b"held-original"
    assert (paths.legacy / "replacement.wav").read_bytes() == b"third-party"
    assert not list(paths.data.glob("annotation_pool_sprsound.legacy-*"))


def test_production_reconcile_rejects_tampered_request_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("requires fork to create an interrupted transaction")
    previous = _production_release(tmp_path, seed=12)
    candidate = _production_release(tmp_path, seed=13)
    forged = _production_release(tmp_path, seed=14)
    paths = pool._production_paths(tmp_path)
    paths.current.symlink_to("releases/%s" % previous.name)

    def crash_after_current(stage: str) -> None:
        if stage == "after_current_switch":
            os._exit(94)

    monkeypatch.setattr(pool, "_production_checkpoint", crash_after_current)
    child = os.fork()
    if child == 0:
        _activate_production_for_test(
            candidate.name,
            confirm_release_id=candidate.name,
            expected_current=previous.name,
            test_home=tmp_path,
            allow_test_home=True,
        )
        os._exit(0)
    _pid, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 94
    value = json.loads(paths.journal.read_text(encoding="utf-8"))
    if "payload" in value:
        value["payload"]["request"]["previous_current"] = forged.name
    else:
        value["previous_current"] = forged.name
    paths.journal.write_bytes(pool.contract.canonical_json_bytes(value))
    before = paths.current.lstat()
    before_target = os.readlink(paths.current)

    with pytest.raises(pool.PoolContractError, match="integrity|journal|hash"):
        _reconcile_production_for_test(
            test_home=tmp_path, allow_test_home=True,
        )

    after = paths.current.lstat()
    assert (after.st_dev, after.st_ino, after.st_ctime_ns) == (
        before.st_dev, before.st_ino, before.st_ctime_ns,
    )
    assert os.readlink(paths.current) == before_target
    assert paths.journal.is_file()


@pytest.mark.parametrize("partial", [b"", b'{"payload":'])
def test_production_journal_sigkill_pending_is_owned_and_reconcile_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, partial: bytes,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("requires fork to simulate an uncatchable journal write")
    candidate = _production_release(tmp_path, seed=15)
    paths = pool._production_paths(tmp_path)
    sentinel = paths.root / "third-party-sentinel"
    sentinel.write_bytes(b"preserve")

    child = os.fork()
    if child == 0:
        original_fdopen = pool.os.fdopen
        armed = {"value": False}
        original_write = pool._write_production_journal

        def arm_write(*args, **kwargs):
            armed["value"] = True
            return original_write(*args, **kwargs)

        def kill_pending(descriptor, *args, **kwargs):
            if armed["value"]:
                if partial:
                    os.write(descriptor, partial)
                    os.fsync(descriptor)
                os.kill(os.getpid(), 9)
            return original_fdopen(descriptor, *args, **kwargs)

        pool._write_production_journal = arm_write
        pool.os.fdopen = kill_pending
        _activate_production_for_test(
            candidate.name,
            confirm_release_id=candidate.name,
            expected_current=None,
            test_home=tmp_path,
            allow_test_home=True,
        )
        os._exit(0)
    _pid, status = os.waitpid(child, 0)
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == 9
    assert not os.path.lexists(str(paths.journal))

    result = _reconcile_production_for_test(
        test_home=tmp_path, allow_test_home=True,
    )

    assert result["status"] == "clean"
    assert sentinel.read_bytes() == b"preserve"
    leftovers = [
        name for name in os.listdir(paths.root)
        if "production-activation" in name and name != paths.lock.name
    ]
    assert leftovers == []


def test_official_separate_wav_json_layout_pairs_by_unique_basename(tmp_path: Path) -> None:
    root = tmp_path / "BioCAS2022"
    _record(root / "train2022_wav", "nested", "ignored", b"audio")
    wav = root / "train2022_wav" / "nested" / "repeated.wav"
    metadata = root / "train2022_json" / "nested" / "repeated.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"record_annotation": "Normal"}))
    # Remove the helper's same-directory JSON: the official corpus separates modalities.
    wav.with_suffix(".json").unlink()

    records = pool.discover_records(root)
    assert len(records) == 1
    assert records[0].audio_path == wav
    assert records[0].metadata_path == metadata


def test_production_quota_contract_is_exactly_400() -> None:
    assert pool.DEFAULT_QUOTAS == {
        "Poor Quality": 120,
        "Normal": 70,
        "CAS": 70,
        "DAS": 70,
        "CAS & DAS": 70,
    }
    assert sum(pool.DEFAULT_QUOTAS.values()) == 400


def test_release_verifier_rejects_any_nondefault_400_clip_quota_mix() -> None:
    quotas = dict(pool.DEFAULT_QUOTAS)
    quotas["Poor Quality"] -= 1
    quotas["Normal"] += 1

    with pytest.raises(pool.PoolContractError, match="exact 400-clip"):
        pool._validate_release_quota_contract(_source_identity(2683), quotas)


def test_repeated_basenames_publish_unique_content_addressed_names(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    root = tmp_path / "annotation_pools" / "sprsound"
    release = pool.build_pool(
        corpus, root, source_identity=_source_identity(6),
        quotas=SMALL_QUOTAS, seed=0,
    )
    verified = pool.verify_pool_release(release, expected_quotas=SMALL_QUOTAS)

    assert verified["total"] == 6
    assert verified["counts"] == SMALL_QUOTAS
    rows = [json.loads(line) for line in (release / "pool_manifest.jsonl").read_text().splitlines()]
    names = [row["destination"] for row in rows]
    hashes = [row["content_sha256"] for row in rows]
    assert len(names) == len(set(names)) == 6
    assert len(hashes) == len(set(hashes)) == 6
    assert len(list(release.glob("*.wav"))) == 6
    assert all("repeated.wav" != name for name in names)


def test_build_is_deterministic_idempotent_and_does_not_activate_production(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    root = tmp_path / "annotation_pools" / "sprsound"
    first = pool.build_pool(
        corpus, root, source_identity=_source_identity(6),
        quotas=SMALL_QUOTAS, seed=11,
    )
    second = pool.build_pool(
        corpus, root, source_identity=_source_identity(6),
        quotas=SMALL_QUOTAS, seed=11,
    )

    assert first == second
    assert first.parent == root / "releases"
    assert not os.path.lexists(str(root / "current"))


def test_pool_root_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(real, target_is_directory=True)

    with pytest.raises(pool.PoolContractError, match="symlink|real directory"):
        pool.build_pool(
            corpus, redirected / "sprsound",
            source_identity=_source_identity(6), quotas=SMALL_QUOTAS, seed=0,
        )


def test_duplicate_content_is_deduplicated_before_sampling(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    _record(corpus, "poor/duplicate", "Poor Quality", b"poor-a")
    records = pool.discover_records(corpus)
    unique = pool.deduplicate_records(records)

    assert len(records) == 7
    assert len(unique) == 6
    with pytest.raises(pool.PoolContractError, match="available"):
        pool.select_records(unique, {**SMALL_QUOTAS, "Poor Quality": 3}, seed=0)


def test_same_content_in_multiple_strata_fails_closed(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    _record(corpus, "normal/conflict", "Normal", b"poor-a")

    with pytest.raises(pool.PoolContractError, match="multiple strata"):
        pool.deduplicate_records(pool.discover_records(corpus))


@pytest.mark.parametrize("fault", ["missing", "extra", "changed"])
def test_final_count_and_exact_file_validation_rejects_drift(
    tmp_path: Path, fault: str,
) -> None:
    corpus = _corpus(tmp_path)
    release = pool.build_pool(
        corpus, tmp_path / "pool", source_identity=_source_identity(6),
        quotas=SMALL_QUOTAS, seed=0,
    )
    clip = next(release.glob("*.wav"))
    if fault == "missing":
        clip.unlink()
    elif fault == "extra":
        (release / "extra.wav").write_bytes(b"extra")
    else:
        clip.write_bytes(b"changed")

    with pytest.raises(pool.PoolContractError):
        pool.verify_pool_release(release, expected_quotas=SMALL_QUOTAS)


def test_atomic_pointer_activation_and_cas_rollback(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    root = tmp_path / "pool"
    release = pool.build_pool(
        corpus, root, source_identity=_source_identity(6),
        quotas=SMALL_QUOTAS, seed=0,
    )
    current = root / ".pool-canary" / "current"
    previous = pool.activate_pool_canary(
        release, current_link=current, expected_current=None,
    )
    target = os.readlink(current)
    assert previous is None and current.resolve() == release.resolve()
    with pytest.raises(pool.PoolContractError, match="compare-and-swap"):
        pool.rollback_pool_canary(
            current_link=current, expected_current="wrong", previous=None,
        )
    pool.rollback_pool_canary(
        current_link=current, expected_current=target, previous=None,
    )
    assert not os.path.lexists(str(current))


def test_preferences_are_outside_release_and_bind_pool_and_clip_hashes(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    release = pool.build_pool(
        corpus, tmp_path / "pool", source_identity=_source_identity(6),
        quotas=SMALL_QUOTAS, seed=0,
    )
    identity = annotator.load_pool_identity(release)
    preferences = annotator.resolve_preferences_path(
        release, preferences_root=tmp_path / "preferences",
    )
    clips = sorted(release.glob("*.wav"))
    row = annotator.preference_record(
        pool_identity=identity,
        clip_a=clips[0], clip_b=clips[1], preferred="a",
        annotator="doctor-1", pair_id="p0001",
    )

    assert preferences == tmp_path / "preferences" / release.name / "preferences.jsonl"
    assert not preferences.is_relative_to(release)
    assert row["pool_release_id"] == release.name
    assert row["segment_a_sha256"] == hashlib.sha256(clips[0].read_bytes()).hexdigest()
    assert row["segment_b_sha256"] == hashlib.sha256(clips[1].read_bytes()).hexdigest()
    with pytest.raises(annotator.PreferenceContractError, match="inside"):
        annotator.resolve_preferences_path(
            release, explicit_path=release / "preferences.jsonl",
        )


def test_verifier_rejects_malformed_manifest_field_types_as_contract_error(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    release = pool.build_pool(
        corpus, tmp_path / "pool", source_identity=_source_identity(6),
        quotas=SMALL_QUOTAS, seed=0,
    )
    manifest = release / "pool_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["destination"] = ["not", "a", "path"]
    manifest.write_bytes(b"".join(pool.contract.canonical_json_bytes(row) for row in rows))

    with pytest.raises(pool.PoolContractError, match="manifest row"):
        pool.verify_pool_release(release, expected_quotas=SMALL_QUOTAS)


def test_release_verifier_rejects_noncanonical_directory_name(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    release = pool.build_pool(
        corpus, tmp_path / "pool", source_identity=_source_identity(6),
        quotas=SMALL_QUOTAS, seed=0,
    )
    copied = tmp_path / "mutable-copy"
    __import__("shutil").copytree(release, copied)
    with pytest.raises(pool.PoolContractError, match="directory"):
        pool.verify_pool_release(copied, expected_quotas=SMALL_QUOTAS)


def test_git_object_snapshot_ignores_mutated_worktree(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    wav = checkout / "BioCAS2022/train2022_wav/sample.wav"
    metadata = checkout / "BioCAS2022/train2022_json/sample.json"
    wav.parent.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    wav.write_bytes(b"PINNED")
    metadata.write_text(json.dumps({"record_annotation": "Normal"}))
    subprocess.run(["git", "-C", str(checkout), "add", "BioCAS2022"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"], text=True,
    ).strip()
    wav.write_bytes(b"MUTATED_AFTER_VERIFY")

    snapshot = tmp_path / "snapshot"
    evidence = pool.materialize_git_snapshot(checkout, snapshot, commit=commit, tree=tree)
    assert (evidence.root / "BioCAS2022/train2022_wav/sample.wav").read_bytes() == b"PINNED"


def test_production_quota_rejects_caller_asserted_source_without_git_evidence(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    fake_production_quotas = dict(pool.DEFAULT_QUOTAS)
    source = {
        **_source_identity(6),
        "schema": "wam_sprsound_git_source_identity_v2",
        "tree": pool.SPRSOUND_TREE,
        "pair_count": pool.SPRSOUND_PAIR_COUNT,
    }
    with pytest.raises(pool.PoolContractError, match="Git-object snapshot"):
        pool.build_pool(
            corpus, tmp_path / "pool", source_identity=source,
            quotas=fake_production_quotas,
        )


def test_any_other_400_clip_quota_mix_is_rejected(tmp_path: Path) -> None:
    quotas = dict(pool.DEFAULT_QUOTAS)
    quotas["Poor Quality"] -= 1
    quotas["Normal"] += 1
    with pytest.raises(pool.PoolContractError, match="exact 400-clip"):
        pool.build_pool(
            _corpus(tmp_path), tmp_path / "pool",
            source_identity=_source_identity(6), quotas=quotas,
        )


def test_preference_append_rejects_symlink_parent_and_duplicate_pair(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(real, target_is_directory=True)
    row = {
        "annotator": "doctor-1", "pair_id": "pair-1", "pool_release_id": "pool-1",
        "preferred": "a", "schema_version": "wam_audio_preference_v2",
        "segment_a": "a.wav", "segment_a_sha256": "a" * 64,
        "segment_b": "b.wav", "segment_b_sha256": "b" * 64, "ts_wall": "",
    }
    with pytest.raises(annotator.PreferenceContractError, match="symlink|directory"):
        annotator.append_preference(redirected / "preferences.jsonl", row)

    path = tmp_path / "preferences" / "preferences.jsonl"
    annotator.append_preference(path, row)
    with pytest.raises(annotator.PreferenceContractError, match="duplicate"):
        annotator.append_preference(path, row)


def test_http_choice_must_match_current_server_pair() -> None:
    expected = {"a": 3, "b": 7, "pair_id": "pair-bound", "preferred": "a"}
    assert annotator.validate_choice(expected, a=3, b=7, pair_id="pair-bound") == "a"
    with pytest.raises(annotator.PreferenceContractError, match="current pair"):
        annotator.validate_choice(
            {"a": 2, "b": 7, "pair_id": "forged", "preferred": "b"},
            a=3, b=7, pair_id="pair-bound",
        )


def test_annotation_restart_resumes_after_completed_pairs() -> None:
    identity = {
        "release_id": "sprsound_pool-" + "a" * 20,
        "clips": {"a.wav": "1" * 64, "b.wav": "2" * 64, "c.wav": "3" * 64},
        "clip_paths": ["/pool/a.wav", "/pool/b.wav", "/pool/c.wav"],
    }
    queue = [(0, 1), (1, 2)]
    first = annotator._server_pair_id(identity, "doctor-1", 0, 0, 1)
    assert annotator._resume_queue_index(queue, identity, "doctor-1", {first}) == 1
