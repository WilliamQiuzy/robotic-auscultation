from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
import threading
import time

import pytest


DEPLOYMENT_DIR = Path(__file__).resolve().parents[1] / "deployment"
sys.path.insert(0, str(DEPLOYMENT_DIR))

import artifact_contract as contract  # noqa: E402
import remote_agent  # noqa: E402
from test_artifact_contract import (  # noqa: E402
    _training_session,
    _trusted_finalization_record,
)


def _manager():
    return importlib.import_module("linbox_converter_env")


def _trusted_training_config(session: Path, root: Path) -> dict:
    registry = root / "TRAINING_FINALIZATIONS.json"
    registry.write_bytes(contract.canonical_json_bytes({
        "records": [_trusted_finalization_record(session)],
        "schema_version": "wam_trusted_training_finalization_registry_v1",
    }))
    return {
        "trusted_finalization_registry": registry,
        "trusted_snapshot_root": root / "trusted-training-snapshots",
    }


def test_live_smoke_uses_an_encoder_safe_video_frame_size() -> None:
    manager = _manager()
    assert '"shape": [64, 64, 3]' in manager._SMOKE_SCRIPT
    assert "np.full((64, 64, 3)" in manager._SMOKE_SCRIPT
    assert "16, 16" not in manager._SMOKE_SCRIPT
    assert manager._SMOKE_SCRIPT.index('os.environ["HF_HOME"]') < manager._SMOKE_SCRIPT.index(
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset"
    )


def test_standard_venv_lib64_symlink_is_removed_before_sealing(tmp_path: Path) -> None:
    manager = _manager()
    candidate = tmp_path / "candidate"
    (candidate / "lib").mkdir(parents=True)
    (candidate / "lib64").symlink_to("lib", target_is_directory=True)
    manager._remove_standard_venv_symlink(candidate)
    assert not os.path.lexists(str(candidate / "lib64"))

    (candidate / "lib64").symlink_to("/tmp", target_is_directory=True)
    with pytest.raises(manager.ConverterPolicyError, match="lib64"):
        manager._remove_standard_venv_symlink(candidate)


def _complete_lock_and_wheelhouse(tmp_path: Path) -> tuple[Path, Path]:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheels = {
        "lerobot-0.3.3-py3-none-any.whl": b"lerobot-wheel",
        "av-15.1.0-cp310-cp310-manylinux_2_17_x86_64.whl": b"av-wheel",
    }
    hashes = {}
    for name, body in wheels.items():
        path = wheelhouse / name
        path.write_bytes(body)
        hashes[name] = hashlib.sha256(body).hexdigest()
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text(
        "lerobot==0.3.3 --hash=sha256:%s\n"
        "av==15.1.0 --hash=sha256:%s\n"
        % (hashes["lerobot-0.3.3-py3-none-any.whl"], hashes[
            "av-15.1.0-cp310-cp310-manylinux_2_17_x86_64.whl"
        ]),
        encoding="utf-8",
    )
    return lock, wheelhouse


def _sealed_session(sessions_root: Path, session_id: str = "session-001") -> Path:
    return _training_session(sessions_root, session_id)


def _complete_candidate(home: Path) -> tuple[Path, dict]:
    manager = _manager()
    descriptor = {
        "lock_sha256": "b" * 64,
        "media_runtime": {"external_binaries": [], "pyav_version": "15.1.0"},
        "packages": {"av": "15.1.0", "lerobot": "0.3.3"},
        "runtime": {
            "abi": "cp310", "implementation": "CPython", "machine": "x86_64",
            "python_version": "3.10.14", "system": "Linux",
        },
        "schema": "wam_linbox_converter_environment_descriptor_v1",
        "wheels": [
            {
                "name": "av-15.1.0-cp310-cp310-manylinux_2_17_x86_64.whl",
                "package": "av", "sha256": "c" * 64, "size": 1,
                "version": "15.1.0",
            },
            {
                "name": "lerobot-0.3.3-py3-none-any.whl",
                "package": "lerobot", "sha256": "d" * 64, "size": 1,
                "version": "0.3.3",
            },
        ],
    }
    descriptor_sha256 = hashlib.sha256(
        manager.canonical_json_bytes(descriptor)
    ).hexdigest()
    environment_id = "lerobot-" + descriptor_sha256[:20]
    candidate = home / "envs" / "releases" / environment_id
    (candidate / "bin").mkdir(parents=True)
    (candidate / "bin" / "python").write_text("python", encoding="utf-8")
    smoke = {
        "api": {
            "add_frame_task": True, "create": True, "reopen": True,
            "save_episode": True,
        },
        "lerobot_version": "0.3.3",
        "pyav_libraries": {"libavcodec": "61.19.100"},
        "pyav_version": "15.1.0",
        "schema": "wam_lerobot_033_smoke_v1",
        "tiny_dataset": {"episodes": 1, "frames": 2, "media_files": 1},
    }
    marker = {
        "descriptor": descriptor,
        "descriptor_sha256": descriptor_sha256,
        "environment_id": environment_id,
        "installed_files": list(manager._installed_tree_inventory(candidate)),
        "installed_tree_sha256": manager._installed_tree_sha256(
            manager._installed_tree_inventory(candidate)
        ),
        "schema": "wam_linbox_converter_environment_complete_v1",
        "smoke": smoke,
    }
    (candidate / manager.ENVIRONMENT_COMPLETE).write_bytes(
        manager.canonical_json_bytes(marker)
    )
    return candidate, marker


def _active_code_release(home: Path) -> tuple[Path, str]:
    source = home / "code-source"
    relative = "auscultation/scripts/session_to_lerobot_v2.py"
    script = source / relative
    script.parent.mkdir(parents=True)
    audited = Path(__file__).resolve().parents[1] / "session_to_lerobot_v2.py"
    script.write_bytes(audited.read_bytes())
    manifest = contract.build_manifest(
        source, [relative], artifact_kind="code_release", source_role="mac"
    )
    releases = home / "Desktop" / "stethoscope" / ".wam-releases"
    release = releases / manifest["release_id"]
    releases.mkdir(parents=True)
    contract.write_release(source, manifest, release, target_role="linbox")
    current = home / "Desktop" / "stethoscope" / ".wam-current"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.symlink_to(Path(".wam-releases") / release.name)
    compatibility = current.parent / "auscultation"
    compatibility.symlink_to(".wam-current/auscultation")
    return release / relative, release.name


def _production_code_confirmation(home: Path) -> tuple[str, str]:
    _converter, release_id = _active_code_release(home)
    root = home / "Desktop" / "stethoscope"
    envelope = contract.verify_release(root / ".wam-releases" / release_id)
    return release_id, envelope["payload_sha256"]


def test_converter_public_production_api_has_fixed_live_paths_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    monkeypatch.setenv("HOME", "/tmp/forged-home")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "forged runtime gate")
    parameters = inspect.signature(manager.activate_production_environment).parameters

    assert manager.LINBOX_PRODUCTION_RELEASES_ROOT == Path("/home/robot/envs/releases")
    assert manager.LINBOX_PRODUCTION_CURRENT == Path("/home/robot/envs/lerobot-current")
    assert manager.LINBOX_PRODUCTION_CODE_ROOT == Path(
        "/home/robot/Desktop/stethoscope"
    )
    assert not {"home", "releases_root", "current_link", "code_root"}.intersection(
        parameters
    )


def test_converter_production_activation_is_code_bound_and_canonical(
    tmp_path: Path,
) -> None:
    manager = _manager()
    legacy_payload_sha256 = "a" * 64
    legacy_release_id = "code-" + legacy_payload_sha256[:20]
    with pytest.raises(manager.ConverterPolicyError, match="identity"):
        manager._confirmed_code_identity(
            legacy_release_id,
            legacy_payload_sha256,
            legacy_release_id,
            legacy_payload_sha256,
        )

    entrypoint_acceptances = []
    for compatibility_state in ("missing", "ordinary-directory", "wrong-target"):
        compatibility_home = tmp_path / ("compatibility-" + compatibility_state)
        compatibility_candidate, _compatibility_marker = _complete_candidate(
            compatibility_home
        )
        compatibility_release_id, compatibility_payload_sha256 = (
            _production_code_confirmation(compatibility_home)
        )
        code_root = compatibility_home / "Desktop" / "stethoscope"
        compatibility = code_root / "auscultation"
        compatibility.unlink()
        if compatibility_state == "ordinary-directory":
            compatibility.mkdir()
        elif compatibility_state == "wrong-target":
            compatibility.symlink_to(".wam-current")

        try:
            manager._active_converter_entrypoint(
                compatibility_home, compatibility_release_id
            )
        except manager.ConverterPolicyError as exc:
            assert "compatibility" in str(exc)
        else:
            entrypoint_acceptances.append(compatibility_state)

        compatibility_current = compatibility_home / "envs" / "lerobot-current"
        with pytest.raises(manager.ConverterPolicyError, match="compatibility"):
            manager._activate_production_environment_at_paths(
                compatibility_candidate,
                releases_root=compatibility_home / "envs" / "releases",
                current_link=compatibility_current,
                code_root=code_root,
                expected_current=None,
                frozen_code_release_id=compatibility_release_id,
                frozen_code_payload_sha256=compatibility_payload_sha256,
                confirm_code_release_id=compatibility_release_id,
                confirm_code_payload_sha256=compatibility_payload_sha256,
            )
        assert not os.path.lexists(str(compatibility_current))
    assert entrypoint_acceptances == []

    home = tmp_path / "linbox-home"
    candidate, marker = _complete_candidate(home)
    code_release_id, code_payload_sha256 = _production_code_confirmation(home)
    legacy = home / "miniconda3" / "envs" / "lerobot"
    legacy.mkdir(parents=True)
    (legacy / "sentinel").write_text("never-touch", encoding="utf-8")

    evidence = manager._activate_production_environment_at_paths(
        candidate,
        releases_root=home / "envs" / "releases",
        current_link=home / "envs" / "lerobot-current",
        code_root=home / "Desktop" / "stethoscope",
        expected_current=None,
        frozen_code_release_id=code_release_id,
        frozen_code_payload_sha256=code_payload_sha256,
        confirm_code_release_id=code_release_id,
        confirm_code_payload_sha256=code_payload_sha256,
    )

    current = home / "envs" / "lerobot-current"
    assert os.readlink(current) == "releases/%s" % candidate.name
    assert current.resolve() == candidate.resolve()
    assert evidence["schema"] == "wam_linbox_converter_production_activation_v1"
    assert evidence["environment_id"] == candidate.name
    assert evidence["descriptor_sha256"] == marker["descriptor_sha256"]
    assert evidence["frozen_code"] == {
        "payload_sha256": code_payload_sha256,
        "release_id": code_release_id,
    }
    assert (legacy / "sentinel").read_text(encoding="utf-8") == "never-touch"


def test_converter_production_failure_rolls_back_and_uses_code_then_env_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    home = tmp_path / "linbox-home"
    candidate, _marker = _complete_candidate(home)
    code_release_id, code_payload_sha256 = _production_code_confirmation(home)
    order = []
    original_code_lock = manager._locked_production_code
    original_env_lock = manager._environment_deploy_lock

    @manager.contextmanager
    def code_lock(*args, **kwargs):
        order.append("code-enter")
        try:
            with original_code_lock(*args, **kwargs) as locked:
                yield locked
        finally:
            order.append("code-exit")

    @manager.contextmanager
    def env_lock(*args, **kwargs):
        order.append("env-enter")
        try:
            with original_env_lock(*args, **kwargs):
                yield
        finally:
            order.append("env-exit")

    def fail_after_switch(stage: str) -> None:
        if stage == "after_switch":
            raise RuntimeError("injected converter production failure")

    monkeypatch.setattr(manager, "_locked_production_code", code_lock)
    monkeypatch.setattr(manager, "_environment_deploy_lock", env_lock)
    monkeypatch.setattr(manager, "_production_checkpoint", fail_after_switch)

    with pytest.raises(RuntimeError, match="injected converter"):
        manager._activate_production_environment_at_paths(
            candidate,
            releases_root=home / "envs" / "releases",
            current_link=home / "envs" / "lerobot-current",
            code_root=home / "Desktop" / "stethoscope",
            expected_current=None,
            frozen_code_release_id=code_release_id,
            frozen_code_payload_sha256=code_payload_sha256,
            confirm_code_release_id=code_release_id,
            confirm_code_payload_sha256=code_payload_sha256,
        )

    assert order == ["code-enter", "env-enter", "env-exit", "code-exit"]
    assert not os.path.lexists(str(home / "envs" / "lerobot-current"))


def test_converter_production_reconcile_rolls_back_uncatchable_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("requires fork to simulate an uncatchable interruption")
    manager = _manager()
    home = tmp_path / "linbox-home"
    candidate, _marker = _complete_candidate(home)
    code_release_id, code_payload_sha256 = _production_code_confirmation(home)

    def crash_after_switch(stage: str) -> None:
        if stage == "after_switch":
            os._exit(94)

    monkeypatch.setattr(manager, "_production_checkpoint", crash_after_switch)
    child = os.fork()
    if child == 0:
        manager._activate_production_environment_at_paths(
            candidate,
            releases_root=home / "envs" / "releases",
            current_link=home / "envs" / "lerobot-current",
            code_root=home / "Desktop" / "stethoscope",
            expected_current=None,
            frozen_code_release_id=code_release_id,
            frozen_code_payload_sha256=code_payload_sha256,
            confirm_code_release_id=code_release_id,
            confirm_code_payload_sha256=code_payload_sha256,
        )
        os._exit(0)
    _pid, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 94
    journal = home / "envs" / ".lerobot-production-activation.json"
    assert journal.is_file()
    assert (home / "envs" / "lerobot-current").resolve() == candidate.resolve()

    result = manager._reconcile_production_environment_at_paths(
        releases_root=home / "envs" / "releases",
        current_link=home / "envs" / "lerobot-current",
        code_root=home / "Desktop" / "stethoscope",
        frozen_code_release_id=code_release_id,
        frozen_code_payload_sha256=code_payload_sha256,
        confirm_code_release_id=code_release_id,
        confirm_code_payload_sha256=code_payload_sha256,
    )

    assert result["status"] == "rolled_back"
    assert not os.path.lexists(str(home / "envs" / "lerobot-current"))
    assert not journal.exists()


def test_converter_production_rejects_releases_root_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    home = tmp_path / "linbox-home"
    candidate, _marker = _complete_candidate(home)
    code_release_id, code_payload_sha256 = _production_code_confirmation(home)
    releases = home / "envs" / "releases"
    moved = home / "envs" / "moved-releases"
    original = manager._switch_environment_at

    def switch_then_swap(envs_fd: int, current_name: str, environment_id: str) -> None:
        original(envs_fd, current_name, environment_id)
        releases.rename(moved)
        releases.symlink_to(moved, target_is_directory=True)

    monkeypatch.setattr(manager, "_switch_environment_at", switch_then_swap)
    with pytest.raises(manager.ConverterPolicyError, match="root|symlink|changed"):
        manager._activate_production_environment_at_paths(
            candidate,
            releases_root=releases,
            current_link=home / "envs" / "lerobot-current",
            code_root=home / "Desktop" / "stethoscope",
            expected_current=None,
            frozen_code_release_id=code_release_id,
            frozen_code_payload_sha256=code_payload_sha256,
            confirm_code_release_id=code_release_id,
            confirm_code_payload_sha256=code_payload_sha256,
        )

    assert not os.path.lexists(str(home / "envs" / "lerobot-current"))


def test_converter_production_reconcile_preserves_committed_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("requires fork to simulate an uncatchable interruption")
    manager = _manager()
    home = tmp_path / "linbox-home"
    candidate, _marker = _complete_candidate(home)
    code_release_id, code_payload_sha256 = _production_code_confirmation(home)

    def crash_after_commit(stage: str) -> None:
        if stage == "after_commit_record":
            os._exit(95)

    monkeypatch.setattr(manager, "_production_checkpoint", crash_after_commit)
    child = os.fork()
    if child == 0:
        manager._activate_production_environment_at_paths(
            candidate,
            releases_root=home / "envs" / "releases",
            current_link=home / "envs" / "lerobot-current",
            code_root=home / "Desktop" / "stethoscope",
            expected_current=None,
            frozen_code_release_id=code_release_id,
            frozen_code_payload_sha256=code_payload_sha256,
            confirm_code_release_id=code_release_id,
            confirm_code_payload_sha256=code_payload_sha256,
        )
        os._exit(0)
    _pid, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 95

    result = manager._reconcile_production_environment_at_paths(
        releases_root=home / "envs" / "releases",
        current_link=home / "envs" / "lerobot-current",
        code_root=home / "Desktop" / "stethoscope",
        frozen_code_release_id=code_release_id,
        frozen_code_payload_sha256=code_payload_sha256,
        confirm_code_release_id=code_release_id,
        confirm_code_payload_sha256=code_payload_sha256,
    )

    assert result["status"] == "committed"
    assert (home / "envs" / "lerobot-current").resolve() == candidate.resolve()


def test_converter_production_rejects_unverified_prior_pointer(tmp_path: Path) -> None:
    manager = _manager()
    home = tmp_path / "linbox-home"
    candidate, _marker = _complete_candidate(home)
    code_release_id, code_payload_sha256 = _production_code_confirmation(home)
    missing_id = "lerobot-" + "a" * 20
    current = home / "envs" / "lerobot-current"
    current.symlink_to("releases/%s" % missing_id)

    with pytest.raises(manager.ConverterPolicyError, match="candidate|release|missing"):
        manager._activate_production_environment_at_paths(
            candidate,
            releases_root=home / "envs" / "releases",
            current_link=current,
            code_root=home / "Desktop" / "stethoscope",
            expected_current=missing_id,
            frozen_code_release_id=code_release_id,
            frozen_code_payload_sha256=code_payload_sha256,
            confirm_code_release_id=code_release_id,
            confirm_code_payload_sha256=code_payload_sha256,
        )

    assert os.readlink(current) == "releases/%s" % missing_id


def test_converter_reconcile_rejects_releases_root_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("requires fork to create a durable interrupted transaction")
    manager = _manager()
    home = tmp_path / "linbox-home"
    candidate, _marker = _complete_candidate(home)
    code_release_id, code_payload_sha256 = _production_code_confirmation(home)

    def crash_after_switch(stage: str) -> None:
        if stage == "after_switch":
            os._exit(96)

    monkeypatch.setattr(manager, "_production_checkpoint", crash_after_switch)
    child = os.fork()
    if child == 0:
        manager._activate_production_environment_at_paths(
            candidate,
            releases_root=home / "envs" / "releases",
            current_link=home / "envs" / "lerobot-current",
            code_root=home / "Desktop" / "stethoscope",
            expected_current=None,
            frozen_code_release_id=code_release_id,
            frozen_code_payload_sha256=code_payload_sha256,
            confirm_code_release_id=code_release_id,
            confirm_code_payload_sha256=code_payload_sha256,
        )
        os._exit(0)
    _pid, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 96
    releases = home / "envs" / "releases"
    moved = home / "envs" / "moved-releases"
    releases.rename(moved)
    releases.symlink_to(moved, target_is_directory=True)

    with pytest.raises(manager.ConverterPolicyError, match="symlink|root|canonical"):
        manager._reconcile_production_environment_at_paths(
            releases_root=releases,
            current_link=home / "envs" / "lerobot-current",
            code_root=home / "Desktop" / "stethoscope",
            frozen_code_release_id=code_release_id,
            frozen_code_payload_sha256=code_payload_sha256,
            confirm_code_release_id=code_release_id,
            confirm_code_payload_sha256=code_payload_sha256,
        )

    assert (home / "envs" / ".lerobot-production-activation.json").is_file()


def test_converter_reconcile_rejects_foreign_code_journal_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("requires fork to create an interrupted transaction")
    manager = _manager()
    home = tmp_path / "linbox-home"
    candidate, marker = _complete_candidate(home)
    code_release_id, code_payload_sha256 = _production_code_confirmation(home)

    def crash_after_switch(stage: str) -> None:
        if stage == "after_switch":
            os._exit(97)

    monkeypatch.setattr(manager, "_production_checkpoint", crash_after_switch)
    child = os.fork()
    if child == 0:
        manager._activate_production_environment_at_paths(
            candidate,
            releases_root=home / "envs" / "releases",
            current_link=home / "envs" / "lerobot-current",
            code_root=home / "Desktop" / "stethoscope",
            expected_current=None,
            frozen_code_release_id=code_release_id,
            frozen_code_payload_sha256=code_payload_sha256,
            confirm_code_release_id=code_release_id,
            confirm_code_payload_sha256=code_payload_sha256,
        )
        os._exit(0)
    _pid, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 97
    current = home / "envs" / "lerobot-current"
    journal = home / "envs" / ".lerobot-production-activation.json"
    value = json.loads(journal.read_text(encoding="utf-8"))
    request = value["payload"]["request"]
    prepared = request["prepared_pointer"]
    previous = request["previous_pointer"]
    foreign_payload_sha256 = "f" * 64
    foreign_release_id = "code_release-" + foreign_payload_sha256[:20]
    value = manager._production_journal(
        desired_id=request["environment"]["environment_id"],
        previous_id=request["previous_environment_id"],
        previous_target=None if previous is None else previous["target"],
        previous_identity=(
            None if previous is None else manager._pointer_identity_tuple(
                previous["identity"], "previous production pointer identity",
            )
        ),
        prepared_name=prepared["name"],
        prepared_target=prepared["target"],
        prepared_identity=manager._pointer_identity_tuple(
            prepared["identity"], "prepared production pointer identity",
        ),
        marker=marker,
        code_release_id=foreign_release_id,
        code_payload_sha256=foreign_payload_sha256,
        stage=value["payload"]["stage"],
    )
    assert manager._validate_production_journal(value) == value
    journal.write_bytes(manager.canonical_json_bytes(value))
    journal_before = journal.read_bytes()
    before = current.lstat()
    target_before = os.readlink(current)

    with pytest.raises(
        manager.ConverterPolicyError,
        match="journal frozen code identity does not match confirmation",
    ):
        manager._reconcile_production_environment_at_paths(
            releases_root=home / "envs" / "releases",
            current_link=current,
            code_root=home / "Desktop" / "stethoscope",
            frozen_code_release_id=code_release_id,
            frozen_code_payload_sha256=code_payload_sha256,
            confirm_code_release_id=code_release_id,
            confirm_code_payload_sha256=code_payload_sha256,
        )

    after = current.lstat()
    assert (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    assert os.readlink(current) == target_before
    assert current.resolve() == candidate.resolve()
    assert journal.read_bytes() == journal_before


def test_converter_pointer_same_target_inode_race_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    home = tmp_path / "linbox-home"
    candidate, _marker = _complete_candidate(home)
    code_release_id, code_payload_sha256 = _production_code_confirmation(home)
    current = home / "envs" / "lerobot-current"
    current.symlink_to("releases/%s" % candidate.name)
    original_write = manager._write_json_at
    replacement_identity = {}

    def write_then_replace(directory_fd, name, value):
        original_write(directory_fd, name, value)
        if name == manager._PRODUCTION_JOURNAL and not replacement_identity:
            target = os.readlink(current)
            current.unlink()
            current.symlink_to(target)
            info = current.lstat()
            replacement_identity["value"] = (info.st_dev, info.st_ino, info.st_ctime_ns)

    monkeypatch.setattr(manager, "_write_json_at", write_then_replace)
    with pytest.raises(manager.ConverterPolicyError, match="compare-and-swap|inode"):
        manager._activate_production_environment_at_paths(
            candidate,
            releases_root=home / "envs" / "releases",
            current_link=current,
            code_root=home / "Desktop" / "stethoscope",
            expected_current=candidate.name,
            frozen_code_release_id=code_release_id,
            frozen_code_payload_sha256=code_payload_sha256,
            confirm_code_release_id=code_release_id,
            confirm_code_payload_sha256=code_payload_sha256,
        )

    info = current.lstat()
    assert (info.st_dev, info.st_ino, info.st_ctime_ns) == replacement_identity["value"]
    assert current.resolve() == candidate.resolve()


def test_lock_audit_requires_exact_lerobot_and_hash_for_every_pin(tmp_path: Path) -> None:
    manager = _manager()
    wheel = tmp_path / "lerobot-0.3.3-py3-none-any.whl"
    wheel.write_bytes(b"lerobot-wheel")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    av_digest = hashlib.sha256(b"av-wheel").hexdigest()
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text(
        "lerobot==0.3.3 --hash=sha256:%s\n"
        "av==15.1.0 --hash=sha256:%s\n" % (digest, av_digest),
        encoding="utf-8",
    )

    audit = manager.audit_lock(lock)
    assert audit.lock_sha256 == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert audit.packages == {"av": "15.1.0", "lerobot": "0.3.3"}
    assert audit.allowed_hashes == frozenset({digest, av_digest})

    lock.write_text("lerobot==0.3.3\n", encoding="utf-8")
    with pytest.raises(manager.ConverterPolicyError, match="hash"):
        manager.audit_lock(lock)
    lock.write_text(
        "lerobot==0.5.2 --hash=sha256:%s\n"
        "av==15.1.0 --hash=sha256:%s\n" % (digest, av_digest),
        encoding="utf-8",
    )
    with pytest.raises(manager.ConverterPolicyError, match="0.3.3"):
        manager.audit_lock(lock)


def test_lock_audit_requires_exact_pyav_pin(tmp_path: Path) -> None:
    manager = _manager()
    digest = "a" * 64
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text(
        "lerobot==0.3.3 --hash=sha256:%s\n" % digest,
        encoding="utf-8",
    )
    with pytest.raises(manager.ConverterPolicyError, match="av=="):
        manager.audit_lock(lock)


def test_lock_rejects_requirement_options_markers_and_invalid_wheel_names(
    tmp_path: Path,
) -> None:
    manager = _manager()
    digest = "a" * 64
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text(
        "lerobot==0.3.3 --hash=sha256:%s --find-links https://example.invalid\n"
        "av==15.1.0 --hash=sha256:%s\n" % (digest, digest),
        encoding="utf-8",
    )
    with pytest.raises(manager.ConverterPolicyError, match="only exact"):
        manager.audit_lock(lock)

    fixture = tmp_path / "wheel-fixture"
    fixture.mkdir()
    lock, wheelhouse = _complete_lock_and_wheelhouse(fixture)
    original = wheelhouse / "lerobot-0.3.3-py3-none-any.whl"
    invalid = wheelhouse / "lerobot-0.3.3-py3-none-any-extra.whl"
    invalid.write_bytes(original.read_bytes())
    original.unlink()
    with pytest.raises(manager.ConverterPolicyError, match="filename"):
        manager.inventory_wheelhouse(wheelhouse, manager.audit_lock(lock))


def test_candidate_plan_is_linbox_python310_linux_x86_64_and_content_addressed(
    tmp_path: Path,
) -> None:
    manager = _manager()
    lock, wheelhouse = _complete_lock_and_wheelhouse(tmp_path)
    runtime = {
        "abi": "cp310",
        "implementation": "CPython",
        "machine": "x86_64",
        "python_version": "3.10.14",
        "system": "Linux",
    }
    releases_root = tmp_path / "envs" / "releases"
    plan = manager.plan_candidate(
        host_role="linbox",
        lock_path=lock,
        releases_root=releases_root,
        runtime=runtime,
        wheelhouse=wheelhouse,
        external_binaries=[],
    )

    assert plan.candidate.parent == releases_root
    assert plan.candidate.name == plan.environment_id
    assert plan.environment_id.startswith("lerobot-")
    assert len(plan.environment_id) == len("lerobot-") + 20
    assert plan.descriptor["lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert [row["name"] for row in plan.descriptor["wheels"]] == sorted(
        path.name for path in wheelhouse.iterdir()
    )
    assert plan.descriptor["media_runtime"] == {
        "external_binaries": [], "pyav_version": "15.1.0"
    }

    with pytest.raises(manager.ConverterPolicyError, match="linbox"):
        manager.plan_candidate(
            host_role="h200", lock_path=lock, releases_root=releases_root,
            runtime=runtime, wheelhouse=wheelhouse, external_binaries=[],
        )
    for field, value in (
        ("python_version", "3.12.1"), ("system", "Darwin"), ("machine", "arm64")
    ):
        changed = dict(runtime)
        changed[field] = value
        with pytest.raises(manager.ConverterPolicyError, match="Python 3.10 Linux x86_64"):
            manager.plan_candidate(
                host_role="linbox", lock_path=lock, releases_root=releases_root,
                runtime=changed, wheelhouse=wheelhouse, external_binaries=[],
            )


def test_wheelhouse_rejects_unlocked_or_missing_wheels(tmp_path: Path) -> None:
    manager = _manager()
    lock, wheelhouse = _complete_lock_and_wheelhouse(tmp_path)
    audit = manager.audit_lock(lock)
    (wheelhouse / "unexpected-1.0-py3-none-any.whl").write_bytes(b"unexpected")
    with pytest.raises(manager.ConverterPolicyError, match="wheel"):
        manager.inventory_wheelhouse(wheelhouse, audit)
    (wheelhouse / "unexpected-1.0-py3-none-any.whl").unlink()
    next(wheelhouse.glob("av-*.whl")).unlink()
    with pytest.raises(manager.ConverterPolicyError, match="wheel"):
        manager.inventory_wheelhouse(wheelhouse, audit)


def test_build_candidate_is_direct_immutable_hash_install_and_smoke_gated(
    tmp_path: Path,
) -> None:
    manager = _manager()
    lock, wheelhouse = _complete_lock_and_wheelhouse(tmp_path)
    runtime = {
        "abi": "cp310", "implementation": "CPython", "machine": "x86_64",
        "python_version": "3.10.14", "system": "Linux",
    }
    plan = manager.plan_candidate(
        host_role="linbox", lock_path=lock,
        releases_root=tmp_path / "envs" / "releases", runtime=runtime,
        wheelhouse=wheelhouse, external_binaries=[],
    )
    base_python = tmp_path / "python3.10"
    base_python.write_text("python", encoding="utf-8")
    calls = []
    pip_snapshot_identity = {}
    smoke = {
        "api": {
            "add_frame_task": True, "create": True, "reopen": True,
            "save_episode": True,
        },
        "lerobot_version": "0.3.3",
        "pyav_libraries": {"libavcodec": "61.19.100"},
        "pyav_version": "15.1.0",
        "schema": "wam_lerobot_033_smoke_v1",
        "tiny_dataset": {"episodes": 1, "frames": 2, "media_files": 1},
    }

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[:3] == [str(base_python), "-m", "venv"]:
            destination = Path(argv[-1])
            (destination / "bin").mkdir(parents=True)
            (destination / "bin" / "python").write_text("python", encoding="utf-8")
            (destination / "bin" / "lerobot-tool").write_text(
                "#!%s/bin/python\n" % destination, encoding="utf-8"
            )
            (destination / "pyvenv.cfg").write_text(
                "command = %s\n" % destination, encoding="utf-8"
            )
            bytecode = (
                destination
                / "lib"
                / "python3.10"
                / "site-packages"
                / "example"
                / "__pycache__"
                / "module.cpython-310.pyc"
            )
            bytecode.parent.mkdir(parents=True)
            bytecode.write_bytes(b"\x00compiled-from=" + os.fsencode(destination))
            (destination / "lib64").symlink_to("lib", target_is_directory=True)
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if argv[0] == str(base_python):
            stdout = manager.canonical_json_bytes(runtime)
        elif argv[1:4] == ["-m", "pip", "install"]:
            snapshot_wheelhouse = Path(argv[argv.index("--find-links") + 1])
            snapshot_lock = Path(argv[argv.index("-r") + 1])
            assert snapshot_wheelhouse != wheelhouse
            assert snapshot_lock != lock
            assert snapshot_lock.read_bytes() == lock.read_bytes()
            assert sorted(path.name for path in snapshot_wheelhouse.iterdir()) == sorted(
                path.name for path in wheelhouse.iterdir()
            )
            pip_snapshot_identity["lock_sha256"] = hashlib.sha256(
                snapshot_lock.read_bytes()
            ).hexdigest()
            pip_snapshot_identity["wheels"] = [
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size": path.stat().st_size,
                }
                for path in sorted(snapshot_wheelhouse.iterdir())
            ]
            stdout = b""
        else:
            assert kwargs["env"]["PATH"] == str(Path(argv[0]).parent)
            stdout = manager.canonical_json_bytes(smoke)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    candidate = manager.build_candidate(
        plan, base_python=base_python, host_role="linbox", runner=runner
    )
    marker = manager.verify_candidate(candidate)
    assert marker["environment_id"] == plan.environment_id
    assert marker["smoke"] == smoke
    assert marker["descriptor"]["lock_sha256"] == pip_snapshot_identity["lock_sha256"]
    assert [
        {key: row[key] for key in ("name", "sha256", "size")}
        for row in marker["descriptor"]["wheels"]
    ] == pip_snapshot_identity["wheels"]
    pip_call = next(call for call in calls if call[1:4] == ["-m", "pip", "install"])
    assert "--no-index" in pip_call
    assert "--require-hashes" in pip_call
    assert Path(pip_call[pip_call.index("--find-links") + 1]) != wheelhouse
    assert Path(pip_call[pip_call.index("-r") + 1]) != lock
    assert not (tmp_path / "envs" / "lerobot-current").exists()
    assert str(candidate) in (candidate / "bin" / "lerobot-tool").read_text(
        encoding="utf-8"
    )
    assert str(candidate) in (candidate / "pyvenv.cfg").read_text(encoding="utf-8")
    assert ".staging" not in (candidate / "bin" / "lerobot-tool").read_text(
        encoding="utf-8"
    )
    assert not list(candidate.rglob("*.pyc"))
    assert not list(candidate.rglob("__pycache__"))

    assert manager.build_candidate(
        plan, base_python=base_python, host_role="linbox", runner=runner
    ) == candidate


def test_build_candidate_installs_from_a_snapshot_and_rechecks_original_inputs(
    tmp_path: Path,
) -> None:
    manager = _manager()
    lock, wheelhouse = _complete_lock_and_wheelhouse(tmp_path)
    runtime = {
        "abi": "cp310", "implementation": "CPython", "machine": "x86_64",
        "python_version": "3.10.14", "system": "Linux",
    }
    plan = manager.plan_candidate(
        host_role="linbox", lock_path=lock,
        releases_root=tmp_path / "envs" / "releases", runtime=runtime,
        wheelhouse=wheelhouse, external_binaries=[],
    )
    base_python = tmp_path / "python3.10"
    base_python.write_text("python", encoding="utf-8")
    pip_inputs = []
    smoke = {
        "api": {
            "add_frame_task": True, "create": True, "reopen": True,
            "save_episode": True,
        },
        "lerobot_version": "0.3.3",
        "pyav_libraries": {"libavcodec": "61.19.100"},
        "pyav_version": "15.1.0",
        "schema": "wam_lerobot_033_smoke_v1",
        "tiny_dataset": {"episodes": 1, "frames": 2, "media_files": 1},
    }

    def runner(argv, **kwargs):
        if argv[:3] == [str(base_python), "-m", "venv"]:
            destination = Path(argv[-1])
            (destination / "bin").mkdir(parents=True)
            (destination / "bin" / "python").write_text("python", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if argv[0] == str(base_python):
            return SimpleNamespace(
                returncode=0, stdout=manager.canonical_json_bytes(runtime), stderr=b""
            )
        if argv[1:4] == ["-m", "pip", "install"]:
            snapshot_wheelhouse = Path(argv[argv.index("--find-links") + 1])
            snapshot_lock = Path(argv[argv.index("-r") + 1])
            pip_inputs.append((snapshot_lock, snapshot_wheelhouse))
            next(wheelhouse.glob("av-*.whl")).write_bytes(b"changed after pip opened inputs")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(
            returncode=0, stdout=manager.canonical_json_bytes(smoke), stderr=b""
        )

    with pytest.raises(manager.ConverterPolicyError, match="build inputs changed"):
        manager.build_candidate(
            plan, base_python=base_python, host_role="linbox", runner=runner
        )

    assert len(pip_inputs) == 1
    snapshot_lock, snapshot_wheelhouse = pip_inputs[0]
    assert snapshot_lock != lock
    assert snapshot_wheelhouse != wheelhouse
    assert not plan.candidate.exists()


def test_completed_environment_detects_installed_tree_mutation(tmp_path: Path) -> None:
    manager = _manager()
    candidate, _marker = _complete_candidate(tmp_path / "home")
    manager.verify_candidate(candidate)
    (candidate / "bin" / "python").write_text("mutated", encoding="utf-8")

    with pytest.raises(manager.ConverterPolicyError, match="tree"):
        manager.verify_candidate(candidate)


def test_candidate_marker_rejects_self_consistent_unknown_descriptor_fields(
    tmp_path: Path,
) -> None:
    manager = _manager()
    candidate, marker = _complete_candidate(tmp_path / "home")
    marker["descriptor"]["unexpected"] = True
    digest = hashlib.sha256(
        manager.canonical_json_bytes(marker["descriptor"])
    ).hexdigest()
    replacement = candidate.parent / ("lerobot-" + digest[:20])
    candidate.rename(replacement)
    marker["descriptor_sha256"] = digest
    marker["environment_id"] = replacement.name
    (replacement / manager.ENVIRONMENT_COMPLETE).write_bytes(
        manager.canonical_json_bytes(marker)
    )

    with pytest.raises(manager.ConverterPolicyError, match="descriptor"):
        manager.verify_candidate(replacement)


def test_candidate_environment_drops_loader_python_and_ffmpeg_pollution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    for key in (
        "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME",
        "FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE",
    ):
        monkeypatch.setenv(key, "/attacker/value")
    candidate = tmp_path / "envs" / "releases" / ("lerobot-" + "a" * 20)
    value = manager._isolated_candidate_environment(candidate, [])

    assert not set(value).intersection({
        "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME",
        "FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE",
    })
    assert value["PATH"] == str(candidate / "bin")


def test_failed_smoke_removes_only_incomplete_candidate(tmp_path: Path) -> None:
    manager = _manager()
    lock, wheelhouse = _complete_lock_and_wheelhouse(tmp_path)
    runtime = {
        "abi": "cp310", "implementation": "CPython", "machine": "x86_64",
        "python_version": "3.10.14", "system": "Linux",
    }
    plan = manager.plan_candidate(
        host_role="linbox", lock_path=lock,
        releases_root=tmp_path / "envs" / "releases", runtime=runtime,
        wheelhouse=wheelhouse, external_binaries=[],
    )
    base_python = tmp_path / "python3.10"
    base_python.write_text("python", encoding="utf-8")

    def runner(argv, **kwargs):
        if argv[:3] == [str(base_python), "-m", "venv"]:
            destination = Path(argv[-1])
            (destination / "bin").mkdir(parents=True)
            (destination / "bin" / "python").write_text("python", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if argv[0] == str(base_python):
            return SimpleNamespace(
                returncode=0, stdout=manager.canonical_json_bytes(runtime), stderr=b""
            )
        if argv[1:4] == ["-m", "pip", "install"]:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=70, stdout=b"", stderr=b"smoke failed")

    with pytest.raises(manager.ConverterExecutionError, match="smoke"):
        manager.build_candidate(
            plan, base_python=base_python, host_role="linbox", runner=runner
        )
    assert not plan.candidate.exists()


def test_canary_activation_and_rollback_use_compare_and_swap(tmp_path: Path) -> None:
    manager = _manager()
    candidate = tmp_path / "envs" / "releases" / ("lerobot-" + "a" * 20)
    (candidate / "bin").mkdir(parents=True)
    (candidate / "bin" / "python").write_text("python", encoding="utf-8")
    descriptor = {
        "lock_sha256": "b" * 64,
        "media_runtime": {"external_binaries": [], "pyav_version": "15.1.0"},
        "packages": {"av": "15.1.0", "lerobot": "0.3.3"},
        "runtime": {
            "abi": "cp310", "implementation": "CPython", "machine": "x86_64",
            "python_version": "3.10.14", "system": "Linux",
        },
        "schema": "wam_linbox_converter_environment_descriptor_v1",
        "wheels": [
            {
                "name": "av-15.1.0-cp310-cp310-manylinux_2_17_x86_64.whl",
                "package": "av", "sha256": "c" * 64, "size": 1,
                "version": "15.1.0",
            },
            {
                "name": "lerobot-0.3.3-py3-none-any.whl",
                "package": "lerobot", "sha256": "d" * 64, "size": 1,
                "version": "0.3.3",
            },
        ],
    }
    environment_id = "lerobot-" + hashlib.sha256(
        manager.canonical_json_bytes(descriptor)
    ).hexdigest()[:20]
    content_addressed_candidate = candidate.parent / environment_id
    candidate.rename(content_addressed_candidate)
    candidate = content_addressed_candidate
    marker = {
        "descriptor": descriptor,
        "descriptor_sha256": hashlib.sha256(
            manager.canonical_json_bytes(descriptor)
        ).hexdigest(),
        "environment_id": candidate.name,
        "installed_files": list(manager._installed_tree_inventory(candidate)),
        "installed_tree_sha256": manager._installed_tree_sha256(
            manager._installed_tree_inventory(candidate)
        ),
        "schema": "wam_linbox_converter_environment_complete_v1",
        "smoke": {
            "api": {"add_frame_task": True, "create": True, "reopen": True, "save_episode": True},
            "lerobot_version": "0.3.3", "pyav_version": "15.1.0",
            "pyav_libraries": {"libavcodec": "61.19.100"},
            "schema": "wam_lerobot_033_smoke_v1",
            "tiny_dataset": {"episodes": 1, "frames": 2, "media_files": 1},
        },
    }
    (candidate / manager.ENVIRONMENT_COMPLETE).write_bytes(
        manager.canonical_json_bytes(marker)
    )
    canary_root = tmp_path / "envs" / ".lerobot-canary"
    current = canary_root / "current"
    previous = manager.activate_canary(
        candidate, current_link=current, canary_root=canary_root,
        expected_current=None,
    )
    assert previous is None and current.resolve() == candidate.resolve()
    with pytest.raises(manager.ConverterPolicyError, match="compare-and-swap"):
        manager.rollback_canary(
            current_link=current, canary_root=canary_root,
            expected_current="wrong-target", previous=None,
        )
    assert current.resolve() == candidate.resolve()
    expected = os.readlink(current)
    manager.rollback_canary(
        current_link=current, canary_root=canary_root,
        expected_current=expected, previous=None,
    )
    assert not current.exists() and not current.is_symlink()


def test_canary_compare_and_swap_is_serialized_across_concurrent_activations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    candidate, _marker = _complete_candidate(tmp_path)
    canary_root = tmp_path / "envs" / ".lerobot-canary"
    current = canary_root / "current"
    original_switch = manager._switch_pointer
    first_entered = threading.Event()
    release_first = threading.Event()
    calls = []
    call_lock = threading.Lock()

    def delayed_switch(path, target):
        with call_lock:
            calls.append(target)
            number = len(calls)
        if number == 1:
            first_entered.set()
            assert release_first.wait(10)
        return original_switch(path, target)

    monkeypatch.setattr(manager, "_switch_pointer", delayed_switch)
    successes = []
    errors = []

    def activate() -> None:
        try:
            successes.append(manager.activate_canary(
                candidate, current_link=current, canary_root=canary_root,
                expected_current=None,
            ))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=activate)
    second = threading.Thread(target=activate)
    first.start()
    assert first_entered.wait(10)
    second.start()
    time.sleep(0.5)
    release_first.set()
    first.join(10)
    second.join(10)

    assert len(successes) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], manager.ConverterPolicyError)
    assert "compare-and-swap" in str(errors[0])


def test_conversion_session_gate_is_linbox_only_direct_child_and_full_rehash(
    tmp_path: Path,
) -> None:
    manager = _manager()
    sessions_root = tmp_path / "home" / "wam_data" / "sessions"
    session = _sealed_session(sessions_root)
    trusted = _trusted_training_config(session, tmp_path)
    identity = manager.verify_sealed_session(
        session, sessions_root=sessions_root, host_role="linbox", **trusted,
    )
    assert identity["raw_release_id"] == (
        "raw_session-" + identity["raw_manifest_sha256"][:20]
    )
    assert len(identity["raw_payload_sha256"]) == 64
    with pytest.raises(manager.ConverterPolicyError, match="linbox"):
        manager.verify_sealed_session(
            session, sessions_root=sessions_root, host_role="h200", **trusted,
        )
    (session / "robot_exact_control.csv").write_bytes(b"tampered")
    with pytest.raises(manager.ConverterPolicyError, match="full artifact rehash"):
        manager.verify_sealed_session(
            session, sessions_root=sessions_root, host_role="linbox", **trusted,
        )
    assert Path(identity["verified_snapshot_path"]) != session


def test_convert_reopens_then_publishes_task2_compatible_dataset_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    home = tmp_path / "home"
    session = _sealed_session(home / "wam_data" / "sessions")
    trusted = _trusted_training_config(session, tmp_path)
    candidate, marker = _complete_candidate(home)
    converter, code_release_id = _active_code_release(home)
    calls = []
    reopen = {
        "episodes": 1,
        "frames": 2,
        "media_files": 1,
        "repo_id": "ausc/session-001",
        "schema": "wam_lerobot_dataset_reopen_v1",
    }
    sealed_rehashes = []
    original_verify_session = manager.verify_sealed_session

    def counted_verify_session(*args, **kwargs):
        sealed_rehashes.append(Path(args[0]))
        return original_verify_session(*args, **kwargs)

    monkeypatch.setattr(manager, "verify_sealed_session", counted_verify_session)

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if len(argv) > 2 and argv[1].endswith("session_to_lerobot_v2.py"):
            assert Path(argv[1]) != converter
            assert Path(argv[1]).read_bytes() == converter.read_bytes()
            assert kwargs["env"]["PATH"] == str(candidate / "bin")
            output = Path(argv[argv.index("--root") + 1])
            conversion_number = sum(
                len(call) > 2 and call[1].endswith("session_to_lerobot_v2.py")
                for call in calls
            )
            for relative, body in {
                "meta/info.json": b"{}\n",
                "data/chunk-000/file-000.parquet": b"parquet",
                "videos/observation.images.top/episode_000000.mp4": (
                    "video-%s" % conversion_number
                ).encode("ascii"),
            }.items():
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
            return SimpleNamespace(returncode=0, stdout=b"converted\n", stderr=b"")
        if "platform.python_version" in argv[-1]:
            stdout = manager.canonical_json_bytes(marker["descriptor"]["runtime"])
        elif argv[-1] == manager._SMOKE_SCRIPT:
            stdout = manager.canonical_json_bytes(marker["smoke"])
        else:
            assert kwargs["env"]["PATH"] == str(candidate / "bin")
            stdout = manager.canonical_json_bytes(reopen)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    result = manager.convert_and_publish(
        session_path=session,
        repo_id="ausc/session-001",
        candidate_path=candidate,
        converter_code_release_id=code_release_id,
        home=home,
        converter_script=converter,
        host_role="linbox",
        **trusted,
        runner=runner,
    )
    repeated = manager.convert_and_publish(
        session_path=session,
        repo_id="ausc/session-001",
        candidate_path=candidate,
        converter_code_release_id=code_release_id,
        home=home,
        converter_script=converter,
        host_role="linbox",
        **trusted,
        runner=runner,
    )

    release = Path(result["release_path"])
    assert release.parent == home / "wam_data" / "lerobot" / "releases"
    envelope = contract.verify_release(release)
    assert envelope["payload"]["artifact_kind"] == "lerobot_dataset"
    identity = remote_agent.validate_converted_dataset(
        release, allowed_releases_root=release.parent
    )
    assert identity["converter_environment_id"] == candidate.name
    assert identity["converter_lock_sha256"] == marker["descriptor"]["lock_sha256"]
    assert identity["lerobot_version"] == "0.3.3"
    converter_calls = [
        call for call in calls
        if len(call) > 2 and call[1].endswith("session_to_lerobot_v2.py")
    ]
    assert converter_calls
    assert all(Path(call[1]) != converter for call in converter_calls)
    assert all(Path(call[2]) != session for call in converter_calls)
    assert all(
        Path(call[2]).parent == trusted["trusted_snapshot_root"]
        for call in converter_calls
    )
    assert all("--trusted-finalization-registry" in call for call in converter_calls)
    assert all("--trusted-snapshot-root" in call for call in converter_calls)
    assert all("--trusted-session-id" in call for call in converter_calls)
    assert sum(call[-1] == manager._SMOKE_SCRIPT for call in calls) >= 2
    assert len(sealed_rehashes) >= 6
    assert repeated["release_id"] == result["release_id"]
    assert not (home / "envs" / "lerobot-current").exists()


def test_conversion_executes_a_code_snapshot_and_rechecks_the_same_release_inode(
    tmp_path: Path,
) -> None:
    manager = _manager()
    home = tmp_path / "home"
    session = _sealed_session(home / "wam_data" / "sessions")
    trusted = _trusted_training_config(session, tmp_path)
    candidate, marker = _complete_candidate(home)
    converter, code_release_id = _active_code_release(home)
    original_bytes = converter.read_bytes()
    executed_scripts = []
    reopen = {
        "episodes": 1, "frames": 2, "media_files": 1,
        "repo_id": "ausc/session-001",
        "schema": "wam_lerobot_dataset_reopen_v1",
    }

    def runner(argv, **kwargs):
        if len(argv) > 2 and argv[1].endswith("session_to_lerobot_v2.py"):
            executed = Path(argv[1])
            executed_scripts.append(executed)
            assert executed.read_bytes() == original_bytes
            converter.write_bytes(b"mutated after the converter started")
            output = Path(argv[argv.index("--root") + 1])
            for relative, body in {
                "meta/info.json": b"{}\n",
                "data/chunk-000/file-000.parquet": b"parquet",
                "videos/observation.images.top/episode_000000.mp4": b"video",
            }.items():
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
            return SimpleNamespace(returncode=0, stdout=b"converted\n", stderr=b"")
        if "platform.python_version" in argv[-1]:
            stdout = manager.canonical_json_bytes(marker["descriptor"]["runtime"])
        elif argv[-1] == manager._SMOKE_SCRIPT:
            stdout = manager.canonical_json_bytes(marker["smoke"])
        else:
            stdout = manager.canonical_json_bytes(reopen)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    with pytest.raises(manager.ConverterPolicyError, match="code release changed"):
        manager.convert_and_publish(
            session_path=session,
            repo_id="ausc/session-001",
            candidate_path=candidate,
            converter_code_release_id=code_release_id,
            home=home,
            converter_script=converter,
            host_role="linbox",
            **trusted,
            runner=runner,
        )

    assert len(executed_scripts) == 1
    assert executed_scripts[0] != converter
    assert ".conversion" in executed_scripts[0].parts
    releases = home / "wam_data" / "lerobot" / "releases"
    assert not releases.exists() or not any(releases.iterdir())


def test_conversion_reconciles_a_published_release_after_operation_marker_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    home = tmp_path / "home"
    session = _sealed_session(home / "wam_data" / "sessions")
    trusted = _trusted_training_config(session, tmp_path)
    candidate, marker = _complete_candidate(home)
    converter, code_release_id = _active_code_release(home)
    reopen = {
        "episodes": 1, "frames": 2, "media_files": 1,
        "repo_id": "ausc/session-001",
        "schema": "wam_lerobot_dataset_reopen_v1",
    }
    conversion_calls = []

    def runner(argv, **kwargs):
        if len(argv) > 2 and argv[1].endswith("session_to_lerobot_v2.py"):
            conversion_calls.append(list(argv))
            output = Path(argv[argv.index("--root") + 1])
            for relative, body in {
                "meta/info.json": b"{}\n",
                "data/chunk-000/file-000.parquet": b"parquet",
                "videos/observation.images.top/episode_000000.mp4": (
                    "video-%s" % len(conversion_calls)
                ).encode("ascii"),
            }.items():
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
            return SimpleNamespace(returncode=0, stdout=b"converted\n", stderr=b"")
        if "platform.python_version" in argv[-1]:
            stdout = manager.canonical_json_bytes(marker["descriptor"]["runtime"])
        elif argv[-1] == manager._SMOKE_SCRIPT:
            stdout = manager.canonical_json_bytes(marker["smoke"])
        else:
            stdout = manager.canonical_json_bytes(reopen)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    original_atomic_json = manager._atomic_json

    def crash_before_operation_marker(path, value):
        if Path(path).parent.name == ".operations":
            raise RuntimeError("simulated crash before operation marker")
        return original_atomic_json(path, value)

    monkeypatch.setattr(manager, "_atomic_json", crash_before_operation_marker)
    with pytest.raises(RuntimeError, match="simulated crash"):
        manager.convert_and_publish(
            session_path=session,
            repo_id="ausc/session-001",
            candidate_path=candidate,
            converter_code_release_id=code_release_id,
            home=home,
            converter_script=converter,
            host_role="linbox",
            **trusted,
            runner=runner,
        )

    monkeypatch.setattr(manager, "_atomic_json", original_atomic_json)
    result = manager.convert_and_publish(
        session_path=session,
        repo_id="ausc/session-001",
        candidate_path=candidate,
        converter_code_release_id=code_release_id,
        home=home,
        converter_script=converter,
        host_role="linbox",
        **trusted,
        runner=runner,
    )

    releases = home / "wam_data" / "lerobot" / "releases"
    assert len(conversion_calls) == 1
    assert [path.name for path in releases.iterdir()] == [result["release_id"]]


def test_environment_and_data_roots_reject_symlink_ancestors(tmp_path: Path) -> None:
    manager = _manager()
    actual = tmp_path / "actual"
    actual.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(actual, target_is_directory=True)

    with pytest.raises(manager.ConverterPolicyError, match="symlink"):
        with manager._environment_deploy_lock(
            redirected / "envs" / "releases"
        ):
            pass


def test_manager_rejects_spoofed_linbox_role_without_actual_host_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    captured = {}

    def fake_convert(**kwargs):
        captured.update(kwargs)
        return {"schema": "wam_linbox_conversion_result_v1"}

    monkeypatch.setattr(manager, "convert_and_publish", fake_convert)
    with pytest.raises(manager.ConverterPolicyError, match="host identity"):
        manager.main(
            [
                "convert", "--session", "/tmp/session", "--repo-id", "ausc/test",
                "--environment", "/tmp/env", "--code-release-id", "code_release-" + "1" * 20,
            ],
            environ={"WAM_HOST_ROLE": "linbox"},
            home=tmp_path / "not-robot-home",
        )
    assert captured == {}


def test_manager_does_not_trust_pytest_environment_as_a_host_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    captured = {}

    def fake_convert(**kwargs):
        captured.update(kwargs)
        return {"schema": "wam_linbox_conversion_result_v1"}

    monkeypatch.setattr(manager, "convert_and_publish", fake_convert)
    with pytest.raises(manager.ConverterPolicyError, match="host identity"):
        manager.main(
            [
                "convert", "--session", "/tmp/session", "--repo-id", "ausc/test",
                "--environment", "/tmp/env", "--code-release-id", "code_release-" + "1" * 20,
            ],
            environ={
                "WAM_HOST_ROLE": "linbox", "WAM_TESTING": "1",
                "PYTEST_CURRENT_TEST": "attacker supplied this",
            },
            home=tmp_path / "not-robot-home",
        )
    assert captured == {}


def test_shell_wrapper_rejects_non_linbox_before_conversion() -> None:
    wrapper = Path(__file__).resolve().parents[1] / "convert_session_on_linbox.sh"
    assert os.access(wrapper, os.X_OK)
    completed = subprocess.run(
        ["bash", str(wrapper), "/tmp/session", "ausc/test", "/tmp/env", "code_release-" + "1" * 20],
        env={**os.environ, "WAM_HOST_ROLE": "h200"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert b"linbox" in completed.stderr
    assert "lerobot-current" not in wrapper.read_text(encoding="utf-8")


def test_shell_wrapper_does_not_trust_pytest_environment_as_a_host_override() -> None:
    wrapper = Path(__file__).resolve().parents[1] / "convert_session_on_linbox.sh"
    completed = subprocess.run(
        ["bash", str(wrapper), "/tmp/session", "ausc/test", "/tmp/env", "code_release-" + "1" * 20],
        env={
            **os.environ,
            "WAM_HOST_ROLE": "linbox",
            "WAM_TESTING": "1",
            "PYTEST_CURRENT_TEST": "attacker supplied this",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert b"audited linbox host identity" in completed.stderr


def test_manager_convert_cli_routes_only_fixed_linbox_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    manager = _manager()
    home = tmp_path / "home"
    environment = home / "envs" / "releases" / ("lerobot-" + "a" * 20)
    session = home / "wam_data" / "sessions" / "session-001"
    converter, code_release_id = _active_code_release(home)
    captured = {}

    def fake_convert(**kwargs):
        captured.update(kwargs)
        return {"release_id": "lerobot_dataset-" + "b" * 20,
                "schema": "wam_linbox_conversion_result_v1"}

    monkeypatch.setattr(manager, "convert_and_publish", fake_convert)
    code = manager.main(
        [
            "convert", "--session", str(session), "--repo-id", "ausc/session-001",
            "--environment", str(environment),
            "--code-release-id", code_release_id,
        ],
        environ={
            "WAM_HOST_ROLE": "linbox", "WAM_TESTING": "1",
            "PYTEST_CURRENT_TEST": "task4",
            "WAM_TRAINING_FINALIZATION_REGISTRY": (
                tmp_path / "TRAINING_FINALIZATIONS.json"
            ).as_posix(),
            "WAM_TRAINING_SNAPSHOT_ROOT": (
                tmp_path / "trusted-training-snapshots"
            ).as_posix(),
        },
        home=home,
        host_identity_probe=lambda: (Path("/home/robot"), "robot-G914"),
    )
    assert code == 0
    assert captured["home"] == home
    assert captured["host_role"] == "linbox"
    assert captured["candidate_path"] == environment
    assert captured["converter_script"] == converter
    assert captured["trusted_finalization_registry"] == (
        tmp_path / "TRAINING_FINALIZATIONS.json"
    ).as_posix()
    assert captured["trusted_snapshot_root"] == (
        tmp_path / "trusted-training-snapshots"
    ).as_posix()
    assert json.loads(capsys.readouterr().out)["schema"] == "wam_linbox_conversion_result_v1"


def test_checked_in_requirement_input_and_complete_lock_are_auditable(
    tmp_path: Path,
) -> None:
    manager = _manager()
    auscultation = Path(__file__).resolve().parents[2]
    requirement_input = (
        auscultation / "requirements-linbox-lerobot.in"
    ).read_text(encoding="utf-8")
    pins = [line for line in requirement_input.splitlines()
            if line and not line.startswith("#")]
    assert pins == [
        "lerobot==0.3.3",
        "av==15.1.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
    ]
    audit = manager.audit_lock(
        auscultation / "requirements-linbox-lerobot.lock.txt"
    )
    assert len(audit.packages) == 94
    assert audit.packages["lerobot"] == "0.3.3"
    assert audit.packages["av"] == "15.1.0"

    unresolved = tmp_path / "requirements-unresolved.lock.txt"
    unresolved.write_text(
        "# unresolved fixture; checked-in lock must never be replaced by this\n",
        encoding="utf-8",
    )
    with pytest.raises(manager.ConverterPolicyError, match="unresolved"):
        manager.audit_lock(unresolved)


def test_manager_build_cli_targets_only_inactive_environment_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    manager = _manager()
    home = tmp_path / "home"
    lock, wheelhouse = _complete_lock_and_wheelhouse(tmp_path)
    runtime = {
        "abi": "cp310", "implementation": "CPython", "machine": "x86_64",
        "python_version": "3.10.14", "system": "Linux",
    }
    captured = {}

    monkeypatch.setattr(manager, "probe_runtime", lambda *args, **kwargs: runtime)

    def fake_build(plan, **kwargs):
        captured["plan"] = plan
        captured.update(kwargs)
        return plan.candidate

    monkeypatch.setattr(manager, "build_candidate", fake_build)
    code = manager.main(
        [
            "build", "--base-python", "/opt/python3.10/bin/python",
            "--lock", str(lock), "--wheelhouse", str(wheelhouse),
        ],
        environ={
            "WAM_HOST_ROLE": "linbox", "WAM_TESTING": "1",
            "PYTEST_CURRENT_TEST": "task4",
        },
        home=home,
        host_identity_probe=lambda: (Path("/home/robot"), "robot-G914"),
    )
    assert code == 0
    assert captured["plan"].candidate.parent == home / "envs" / "releases"
    assert captured["host_role"] == "linbox"
    assert not (home / "envs" / "lerobot-current").exists()
    assert json.loads(capsys.readouterr().out)["environment_id"].startswith("lerobot-")
