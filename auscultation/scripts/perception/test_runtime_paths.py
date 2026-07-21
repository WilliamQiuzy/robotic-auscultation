from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
DEPLOYMENT = HERE.parent / "deployment"
sys.path.insert(0, str(DEPLOYMENT))

import runtime_paths
import artifact_contract as contract
from test_artifact_contract import (  # noqa: E402
    _training_session,
    _trusted_finalization_record,
)


SCRIPT_ROLES = {
    "geom_export.py": "h200_training",
    "stage0_train_normals.py": "h200_training",
    "stage1_cotrain.py": "h200_training",
    "stage2_drape_train.py": "h200_training",
    "teacher_smpl_fit.py": "h200_training",
    "eval_detection.py": "h200_evaluation",
    "eval_model_effect.py": "h200_evaluation",
    "latency_bench.py": "h200_evaluation",
    "track_final.py": "linbox_realtime",
    "run_closed_loop.py": "linbox_realtime",
    "run_compare.py": "linbox_realtime",
    "clean_depth_noise.py": "linbox_realtime",
    "handeye_calibrate.py": "linbox_calibration",
}


def test_role_defaults_encode_the_canonical_three_host_topology() -> None:
    training = runtime_paths.resolve_runtime_paths(
        "h200_training", home=Path("/home/ssh-ziyue"),
        hostname="computeinstance-e00saxxvybxg7qvj0s", require_exists=False,
    )
    realtime = runtime_paths.resolve_runtime_paths(
        "linbox_realtime", home=Path("/home/robot"), hostname="robot-G914",
        require_exists=False,
    )

    assert training.host_role == "h200"
    assert training.repo_root == Path("/home/ssh-ziyue/wam_ausc/current/auscultation")
    assert training.data_root == Path("/home/ssh-ziyue/wam_ausc/data")
    assert training.output_root == Path("/home/ssh-ziyue/wam_ausc/data/outputs/perception")
    assert realtime.host_role == "linbox"
    assert realtime.repo_root == Path(
        "/home/robot/Desktop/stethoscope/.wam-current/auscultation"
    )
    assert realtime.data_root == Path("/home/robot/wam_data/sessions")
    assert realtime.output_root == Path("/home/robot/wam_data/outputs/perception")


def test_wrong_host_role_fails_closed() -> None:
    with pytest.raises(runtime_paths.RuntimePathError, match="requires H200"):
        runtime_paths.resolve_runtime_paths(
            "h200_training", home=Path("/home/robot"), require_exists=False,
        )
    with pytest.raises(runtime_paths.RuntimePathError, match="requires linbox"):
        runtime_paths.resolve_runtime_paths(
            "linbox_realtime", home=Path("/home/ssh-ziyue"), require_exists=False,
        )


def test_home_environment_spoof_does_not_change_actual_host_role(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/home/ssh-ziyue")
    assert runtime_paths.detect_host_role() == "mac_dev"


@pytest.mark.parametrize(
    ("home", "hostname"),
    [
        (Path("/home/ssh-ziyue"), "robot-G914"),
        (Path("/home/robot"), "computeinstance-e00saxxvybxg7qvj0s"),
    ],
)
def test_home_and_machine_identity_must_both_match(home: Path, hostname: str) -> None:
    assert runtime_paths.detect_host_role(home=home, hostname=hostname) == "mac_dev"


def test_test_override_requires_an_explicit_test_context(tmp_path: Path) -> None:
    env = {
        "AUSC_REPO": str(tmp_path),
        "AUSC_DATA": str(tmp_path),
        "AUSC_OUTPUT": str(tmp_path / "out"),
        "WAM_TEST_ROLE_OVERRIDE": "1",
    }
    with pytest.raises(runtime_paths.RuntimePathError, match="test-only"):
        runtime_paths.resolve_runtime_paths(
            "h200_training", environ=env, home=tmp_path, require_exists=True,
        )
    env["WAM_TESTING"] = "1"
    resolved = runtime_paths.resolve_runtime_paths(
        "h200_training", environ=env, home=tmp_path, require_exists=True,
    )
    assert resolved.host_role == "test_override"


@pytest.mark.parametrize("field", ["AUSC_REPO", "AUSC_DATA", "AUSC_OUTPUT"])
def test_path_overrides_must_be_absolute(field: str, tmp_path: Path) -> None:
    env = {
        "AUSC_REPO": str(tmp_path),
        "AUSC_DATA": str(tmp_path),
        "AUSC_OUTPUT": str(tmp_path / "out"),
        "WAM_TESTING": "1",
        "WAM_TEST_ROLE_OVERRIDE": "1",
    }
    env[field] = "relative/path"
    with pytest.raises(runtime_paths.RuntimePathError, match="absolute"):
        runtime_paths.resolve_runtime_paths(
            "h200_evaluation", environ=env, home=tmp_path, require_exists=False,
        )


def test_activate_validates_before_chdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.cwd()
    env = {
        "AUSC_REPO": str(tmp_path / "missing"),
        "AUSC_DATA": str(tmp_path),
        "WAM_TESTING": "1",
        "WAM_TEST_ROLE_OVERRIDE": "1",
    }
    with pytest.raises(runtime_paths.RuntimePathError, match="repo root"):
        runtime_paths.activate_runtime(
            "h200_training", environ=env, home=tmp_path,
        )
    assert Path.cwd() == original


def test_all_perception_jobs_declare_one_role_and_drop_legacy_defaults() -> None:
    forbidden = (
        "~/Desktop/wam-vla-medical/auscultation",
        "~/ausc_test",
        'os.environ.get("AUSC_REPO"',
    )
    for name, role in SCRIPT_ROLES.items():
        source = (HERE / name).read_text(encoding="utf-8")
        assert 'RUNTIME_ROLE = "%s"' % role in source
        assert "activate_runtime(RUNTIME_ROLE)" in source
        assert not any(value in source for value in forbidden), name


def test_readme_assigns_training_to_h200_and_realtime_to_linbox() -> None:
    text = (HERE / "README.md").read_text(encoding="utf-8")
    assert "H200" in text and "training" in text.lower()
    assert "linbox" in text and "real-time" in text.lower()
    assert "~/ausc_test" not in text
    assert "GPU box (RTX 5090" not in text


def test_handeye_commissioning_disables_legacy_charuco_and_uses_fk_method() -> None:
    source = (HERE / "handeye_calibrate.py").read_text(encoding="utf-8")
    assert 'args.mode != "tiptouch"' in source
    assert "Charuco capture is not yet covered by the commissioning contract" in source
    assert "transform = fk.fk(q7)" in source


def test_h200_real_input_requires_a_completed_exact_raw_release(tmp_path: Path) -> None:
    releases = tmp_path / "raw_sessions" / "releases"
    staged = _training_session(releases, "session-001")
    trusted_record = _trusted_finalization_record(staged)
    manifest_sha256 = trusted_record["artifact_sha256"]["manifest"]
    release = releases / ("raw_session-" + manifest_sha256[:20])
    staged.rename(release)
    registry = tmp_path / "TRAINING_FINALIZATIONS.json"
    registry.write_bytes(contract.canonical_json_bytes({
        "records": [trusted_record],
        "schema_version": "wam_trusted_training_finalization_registry_v1",
    }))
    env = {
        "AUSC_PERCEPTION_REAL_RELEASE": str(release),
        "WAM_TRAINING_FINALIZATION_REGISTRY": str(registry),
        "WAM_TRAINING_SNAPSHOT_ROOT": str(tmp_path / "trusted-snapshots"),
        "WAM_TRAINING_FINALIZATION_SHA256": trusted_record[
            "artifact_sha256"
        ]["finalization"],
    }
    verified = runtime_paths.require_h200_raw_release(
        "AUSC_PERCEPTION_REAL_RELEASE", environ=env, allowed_root=releases,
        required_files=("audio.wav",),
    )
    assert verified != release
    assert verified.parent == tmp_path / "trusted-snapshots"

    (release / "audio.wav").write_bytes(b"drift")
    with pytest.raises(runtime_paths.RuntimePathError, match="completed raw release"):
        runtime_paths.require_h200_raw_release(
            "AUSC_PERCEPTION_REAL_RELEASE", environ=env, allowed_root=releases,
            required_files=("audio.wav",),
        )


def test_h200_real_data_scripts_do_not_default_to_mutable_data_root() -> None:
    for name in (
        "stage1_cotrain.py", "stage2_drape_train.py", "teacher_smpl_fit.py",
        "eval_detection.py", "eval_model_effect.py", "latency_bench.py",
    ):
        source = (HERE / name).read_text(encoding="utf-8")
        assert "require_h200_raw_release" in source, name


def test_h200_output_override_cannot_target_an_immutable_release(tmp_path: Path) -> None:
    protected = Path("/home/ssh-ziyue/wam_ausc/data/raw_sessions/releases/raw_session-" + "a" * 20)
    env = {
        "AUSC_REPO": str(tmp_path), "AUSC_DATA": str(tmp_path),
        "AUSC_OUTPUT": str(protected),
    }
    with pytest.raises(runtime_paths.RuntimePathError, match="immutable|symlink"):
        runtime_paths.resolve_runtime_paths(
            "h200_training", environ=env, home=Path("/home/ssh-ziyue"),
            hostname="computeinstance-e00saxxvybxg7qvj0s", require_exists=False,
        )


def test_output_override_rejects_symlink_alias_before_any_write(tmp_path: Path) -> None:
    real = tmp_path / "real-output"
    real.mkdir()
    alias = tmp_path / "output-alias"
    alias.symlink_to(real, target_is_directory=True)
    env = {
        "AUSC_REPO": str(tmp_path), "AUSC_DATA": str(tmp_path),
        "AUSC_OUTPUT": str(alias),
    }
    with pytest.raises(runtime_paths.RuntimePathError, match="symlink"):
        runtime_paths.resolve_runtime_paths(
            "h200_training", environ=env, home=Path("/home/ssh-ziyue"),
            hostname="computeinstance-e00saxxvybxg7qvj0s", require_exists=False,
        )


def test_teacher_labels_cannot_escape_the_completed_release() -> None:
    source = (HERE / "teacher_smpl_fit.py").read_text(encoding="utf-8")
    assert "--labels" not in source
    assert "args.labels" not in source
