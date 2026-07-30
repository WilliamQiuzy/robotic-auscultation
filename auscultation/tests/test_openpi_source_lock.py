from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = ROOT / "auscultation" / "scripts" / "deployment"
sys.path.insert(0, str(DEPLOYMENT))

import sync_openpi_source as source_sync  # noqa: E402


LOCK_PATH = ROOT / "auscultation" / "configs" / "policy" / "openpi.source.lock.json"
LOCAL_OPENPI = ROOT / "VLA-baselines" / "openpi"


def _initialize_fixture_repository(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "WAM Test"],
        check=True,
    )


def _fixture_source_lock(tmp_path: Path, *, releases_root: Path = None):
    upstream = tmp_path / "upstream"
    _initialize_fixture_repository(upstream)
    (upstream / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    (upstream / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (upstream / ".gitmodules").write_text("", encoding="utf-8")
    (upstream / "policy.py").write_text("VALUE = 'PINNED'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(upstream), "add", "."], check=True)
    subprocess.run(["git", "-C", str(upstream), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    rows = [
        {"path": name, "sha256": source_sync.sha256_file(upstream / name)}
        for name in (".gitmodules", "pyproject.toml", "uv.lock")
    ]
    return {
        "commit": commit,
        "environment_boundary": "separate_openpi_uv_lock",
        "files": rows,
        "name": "openpi",
        "policy_adapter_status": "blocked_pending_tracked_wam_adapter",
        "releases_root": (releases_root or (tmp_path / "releases")).as_posix(),
        "repository": upstream.as_uri(),
        "schema": "wam_external_source_lock_v1",
        "submodules": [],
        "target_role": "h200",
        "tree": tree,
    }


def _fixture_initialized_submodule_checkout(tmp_path: Path):
    child = tmp_path / "child"
    _initialize_fixture_repository(child)
    (child / "child.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(child), "add", "."], check=True)
    subprocess.run(["git", "-C", str(child), "commit", "-qm", "child"], check=True)
    child_commit = subprocess.check_output(
        ["git", "-C", str(child), "rev-parse", "HEAD"], text=True
    ).strip()

    parent = tmp_path / "parent"
    _initialize_fixture_repository(parent)
    (parent / "pyproject.toml").write_text(
        "[project]\nname='parent'\n", encoding="utf-8"
    )
    (parent / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (parent / ".gitmodules").write_text(
        '[submodule "third_party/child"]\n'
        "\tpath = third_party/child\n"
        "\turl = %s\n" % child.as_uri(),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "git", "-C", str(parent), "update-index", "--add", "--cacheinfo",
            "160000,%s,third_party/child" % child_commit,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(parent), "add", ".gitmodules", "pyproject.toml", "uv.lock"],
        check=True,
    )
    subprocess.run(["git", "-C", str(parent), "commit", "-qm", "parent"], check=True)

    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "-q", parent.as_uri(), str(checkout)], check=True)
    subprocess.run(
        [
            "git", "-C", str(checkout), "-c", "protocol.file.allow=always",
            "submodule", "update", "--init", "--recursive",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    rows = [
        {"path": name, "sha256": source_sync.sha256_file(checkout / name)}
        for name in (".gitmodules", "pyproject.toml", "uv.lock")
    ]
    lock = {
        "commit": commit,
        "environment_boundary": "separate_openpi_uv_lock",
        "files": rows,
        "name": "openpi",
        "policy_adapter_status": "blocked_pending_tracked_wam_adapter",
        "releases_root": (tmp_path / "releases").as_posix(),
        "repository": parent.as_uri(),
        "schema": "wam_external_source_lock_v1",
        "submodules": [{
            "commit": child_commit,
            "path": "third_party/child",
            "repository": child.as_uri(),
        }],
        "target_role": "h200",
        "tree": tree,
    }
    return checkout, lock


def test_tracked_openpi_lock_is_canonical_and_exact() -> None:
    lock = source_sync.load_source_lock(LOCK_PATH)

    assert LOCK_PATH.read_bytes() == source_sync.canonical_json_bytes(lock)
    assert lock["repository"] == "https://github.com/Physical-Intelligence/openpi.git"
    assert lock["commit"] == "c23745b5ad24e98f66967ea795a07b2588ed6c79"
    assert lock["tree"] == "bc1af9de9f53acf776ddcfe7be54c6219d1d5faf"
    assert lock["target_role"] == "h200"
    assert lock["environment_boundary"] == "separate_openpi_uv_lock"
    assert {row["path"] for row in lock["files"]} == {
        ".gitmodules", "pyproject.toml", "uv.lock",
    }
    assert {row["path"]: row["commit"] for row in lock["submodules"]} == {
        "third_party/aloha": "d1dc83afd89ded4379851257fe5d85632d31d5ec",
        "third_party/libero": "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(extra=True), "fields"),
        (lambda value: value.update(commit="main"), "commit"),
        (lambda value: value.update(repository="git@github.com:evil/openpi.git"), "repository"),
        (lambda value: value["files"].append(dict(value["files"][0])), "duplicate"),
        (lambda value: value["submodules"][0].update(path="../escape"), "path"),
    ],
)
def test_source_lock_rejects_ambiguous_or_unsafe_metadata(
    tmp_path: Path, mutation, message: str,
) -> None:
    value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    mutation(value)
    changed = tmp_path / "changed.json"
    changed.write_bytes(source_sync.canonical_json_bytes(value))

    with pytest.raises(source_sync.SourceLockError, match=message):
        source_sync.load_source_lock(changed)


def test_current_checkout_main_identity_is_pinned_but_incomplete() -> None:
    lock = source_sync.load_source_lock(LOCK_PATH)
    identity = source_sync.verify_checkout(LOCAL_OPENPI, lock, require_submodules=False)

    assert identity["commit"] == lock["commit"]
    assert identity["tree"] == lock["tree"]
    assert identity["missing_submodules"] == [
        "third_party/aloha", "third_party/libero",
    ]
    with pytest.raises(source_sync.SourceLockError, match="submodule"):
        source_sync.verify_checkout(LOCAL_OPENPI, lock, require_submodules=True)


def test_checkout_rejects_top_level_origin_mismatch() -> None:
    lock = copy.deepcopy(source_sync.load_source_lock(LOCK_PATH))
    lock["repository"] = "https://github.com/example/not-openpi.git"

    with pytest.raises(source_sync.SourceLockError, match="origin"):
        source_sync.verify_checkout(LOCAL_OPENPI, lock, require_submodules=False)


def test_checkout_rejects_gitmodules_repository_mismatch() -> None:
    lock = copy.deepcopy(source_sync.load_source_lock(LOCK_PATH))
    lock["submodules"][0]["repository"] = (
        "https://github.com/example/not-aloha.git"
    )

    with pytest.raises(source_sync.SourceLockError, match="submodule repository"):
        source_sync.verify_checkout(LOCAL_OPENPI, lock, require_submodules=False)


def test_checkout_rejects_initialized_submodule_origin_mismatch(tmp_path: Path) -> None:
    checkout, lock = _fixture_initialized_submodule_checkout(tmp_path)
    subprocess.run(
        [
            "git", "-C", str(checkout / "third_party" / "child"),
            "remote", "set-url", "origin", "file:///wrong-child",
        ],
        check=True,
    )

    with pytest.raises(source_sync.SourceLockError, match="submodule origin"):
        source_sync.verify_checkout(checkout, lock, require_submodules=True)


def test_install_source_candidate_is_content_addressed_and_idempotent(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    lock = _fixture_source_lock(tmp_path)
    first = source_sync.install_source_candidate(lock)
    second = source_sync.install_source_candidate(lock)

    assert first == second
    envelope = source_sync.verify_source_release(first, lock)
    assert envelope["release_id"] == first.name
    assert (first / "source" / "pyproject.toml").is_file()
    assert not any(path.name == ".git" for path in first.rglob(".git"))


def test_install_materializes_pinned_git_objects_after_worktree_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _fixture_source_lock(tmp_path)
    original_verify = source_sync.verify_checkout
    mutation_observed = []

    def verify_then_mutate(checkout, value, *, require_submodules=True):
        identity = original_verify(
            checkout, value, require_submodules=require_submodules
        )
        (Path(checkout) / "policy.py").write_text(
            "VALUE = 'MUTATED_AFTER_VERIFY'\n", encoding="utf-8"
        )
        mutation_observed.append(True)
        return identity

    monkeypatch.setattr(source_sync, "verify_checkout", verify_then_mutate)

    release = source_sync.install_source_candidate(lock)

    assert mutation_observed == [True]
    assert (release / "source" / "policy.py").read_text(encoding="utf-8") == (
        "VALUE = 'PINNED'\n"
    )
    source_sync.verify_source_release(release, lock)


def test_install_rejects_snapshot_mutation_after_git_object_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _fixture_source_lock(tmp_path)
    original_materialize = source_sync.materialize_pinned_source

    def materialize_then_mutate(checkout, destination, value):
        inventory = original_materialize(checkout, destination, value)
        (Path(destination) / "source" / "policy.py").write_text(
            "VALUE = 'MUTATED_SNAPSHOT'\n", encoding="utf-8"
        )
        return inventory

    monkeypatch.setattr(
        source_sync, "materialize_pinned_source", materialize_then_mutate
    )

    with pytest.raises(source_sync.SourceLockError, match="inventory"):
        source_sync.install_source_candidate(lock)


@pytest.mark.parametrize("change", ["tracked", "untracked"])
def test_checkout_rejects_dirty_or_untracked_initialized_submodule(
    tmp_path: Path, change: str,
) -> None:
    checkout, lock = _fixture_initialized_submodule_checkout(tmp_path)
    child = checkout / "third_party" / "child"
    if change == "tracked":
        (child / "child.py").write_text("VALUE = 2\n", encoding="utf-8")
    else:
        (child / "untracked.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(source_sync.SourceLockError, match="submodule.*dirty"):
        source_sync.verify_checkout(checkout, lock, require_submodules=True)


def test_source_provenance_binds_complete_file_mode_and_hash_inventory(
    tmp_path: Path,
) -> None:
    lock = _fixture_source_lock(tmp_path)
    release = source_sync.install_source_candidate(lock)

    provenance = json.loads(
        (release / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8")
    )
    inventory = provenance["source_inventory"]
    rows = inventory["files"]
    assert inventory["schema"] == "wam_external_source_inventory_v1"
    assert inventory["sha256"] == source_sync.hash_source_inventory(rows)
    assert {row["path"] for row in rows} == set(
        source_sync.snapshot_paths(release / "source")
    )
    assert all(set(row) == {"mode", "path", "sha256", "size"} for row in rows)
    source_sync.verify_source_release(release, lock)


def test_source_install_rejects_symlinked_release_root_ancestor(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(actual, target_is_directory=True)
    lock = _fixture_source_lock(
        tmp_path / "fixture", releases_root=redirected / "openpi" / "releases"
    )

    with pytest.raises(source_sync.SourceLockError, match="symlink"):
        source_sync.install_source_candidate(lock)
    assert not (actual / "openpi").exists()


def test_concurrent_source_install_converges_to_one_verified_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _fixture_source_lock(tmp_path)
    original_write = source_sync.contract.write_release
    first_write_entered = threading.Event()
    allow_first_write = threading.Event()
    write_calls = []
    write_calls_lock = threading.Lock()

    def delayed_first_write(*args, **kwargs):
        with write_calls_lock:
            write_calls.append(args[2])
            call_number = len(write_calls)
        if call_number == 1:
            first_write_entered.set()
            assert allow_first_write.wait(10)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(source_sync.contract, "write_release", delayed_first_write)
    results = []
    errors = []

    def install() -> None:
        try:
            results.append(source_sync.install_source_candidate(lock))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=install)
    second = threading.Thread(target=install)
    first.start()
    assert first_write_entered.wait(10)
    second.start()
    time.sleep(1.0)
    allow_first_write.set()
    first.join(15)
    second.join(15)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert len(results) == 2 and results[0] == results[1]
    assert len(write_calls) == 1
    source_sync.verify_source_release(results[0], lock)


def test_source_install_rejects_symlinked_payload(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "real.py").write_text("ok\n", encoding="utf-8")
    (checkout / "linked.py").symlink_to(checkout / "real.py")

    with pytest.raises(source_sync.SourceLockError, match="symlink"):
        source_sync.snapshot_paths(checkout)
