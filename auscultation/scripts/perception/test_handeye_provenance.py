from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import types

import pytest


HERE = Path(__file__).resolve().parent
AUSC_ROOT = HERE.parents[1]
MODULE_PATH = HERE / "handeye_provenance.py"
DEPLOYMENT = AUSC_ROOT / "scripts" / "deployment"
if str(DEPLOYMENT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT))

import artifact_contract


SOURCE_PATHS = {
    "handeye": "auscultation/scripts/perception/handeye_calibrate.py",
    "runtime_paths": "auscultation/scripts/perception/runtime_paths.py",
    "camera_stream": "auscultation/ausc/perception/camera_stream.py",
    "orbbec_preflight": "auscultation/scripts/perception/handeye_orbbec_preflight.py",
    "provenance": "auscultation/scripts/perception/handeye_provenance.py",
}
ARTIFACT_CONTRACT = "auscultation/scripts/deployment/artifact_contract.py"
PACKAGES = {
    "numpy": "1.26.4",
    "opencv-python": "4.11.0.86",
    "scipy": "1.17.1",
    "yourdfpy": "0.0.60",
    "pyorbbecsdk2": "2.1.1",
}


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_code_release(tmp_path: Path, *, target_role: str = "linbox") -> Path:
    assert MODULE_PATH.is_file(), "handeye_provenance implementation is missing"
    source = tmp_path / "code-source"
    for label, relative in SOURCE_PATHS.items():
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if label == "provenance":
            shutil.copyfile(MODULE_PATH, destination)
        else:
            destination.write_text("# %s source\n" % label, encoding="utf-8")
    verifier = source / ARTIFACT_CONTRACT
    verifier.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEPLOYMENT / "artifact_contract.py", verifier)
    relative_paths = tuple(SOURCE_PATHS.values()) + (ARTIFACT_CONTRACT,)
    manifest = artifact_contract.build_manifest(
        source,
        relative_paths,
        artifact_kind="handeye_calibration_code",
        source_role="mac",
    )
    releases = tmp_path / "handeye_code" / "releases"
    destination = releases / manifest["release_id"]
    artifact_contract.write_release(
        source, manifest, destination, target_role=target_role,
    )
    return destination


def _load_release_module(release: Path):
    path = release / SOURCE_PATHS["provenance"]
    name = "handeye_provenance_test_%s" % release.name.replace("-", "_")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def _build_environment(tmp_path: Path):
    environment_id = "linbox-perception-" + "a" * 20
    root = tmp_path / "perception_envs" / "releases" / environment_id
    executable = root / "bin" / "python3.11"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fake direct CPython executable\n")
    executable.chmod(0o755)
    executable_sha256 = _sha256(executable.read_bytes())
    lock = {
        "environment_tree": {"sha256": "e" * 64},
        "environment_id": environment_id,
        "packages": [
            {"name": name, "record_sha256": _sha256(name.encode()), "version": version}
            for name, version in sorted(PACKAGES.items())
        ],
        "python": {
            "executable_sha256": executable_sha256,
            "implementation": "CPython",
            "version": "3.11.15",
        },
        "record_observation": {"sha256": "d" * 64},
        "schema": "wam_perception_environment_lock_v3",
    }
    lock_path = root / "PERCEPTION_ENVIRONMENT_STRONG_LOCK.json"
    lock_bytes = _canonical_json(lock)
    lock_path.write_bytes(lock_bytes)
    lock_sha256 = _sha256(lock_bytes)
    report = {
        "complete": True,
        "environment_id": environment_id,
        "environment_root": str(root),
        "lock_sha256": lock_sha256,
        "observed_environment_tree": lock["environment_tree"] if "environment_tree" in lock else {},
        "observed_record_observation": lock["record_observation"] if "record_observation" in lock else {},
        "python_executable_relative": "bin/python",
        "required_service_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
        "schema": "wam_perception_environment_clone_marker_v2",
        "source_environment_fingerprint_sha256": "f" * 64,
    }
    report_path = root / "PERCEPTION_ENVIRONMENT_CLONE_COMPLETE.json"
    report_bytes = _canonical_json(report)
    report_path.write_bytes(report_bytes)
    return {
        "root": root,
        "executable": executable,
        "lock_path": lock_path,
        "lock_sha256": lock_sha256,
        "report_path": report_path,
        "report_sha256": _sha256(report_bytes),
    }


def _patch_runtime(monkeypatch, module, environment) -> None:
    monkeypatch.setattr(module.sys, "executable", str(environment["executable"]))
    monkeypatch.setattr(module.platform, "python_version", lambda: "3.11.15")
    monkeypatch.setattr(module.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(
        module.metadata,
        "version",
        lambda name: PACKAGES[name],
    )


def _collect(module, environment):
    return module.collect_handeye_provenance(
        environment_lock_sha256=environment["lock_sha256"],
        environment_report_sha256=environment["report_sha256"],
    )


def test_collects_only_hash_bound_non_secret_audit_fields(tmp_path, monkeypatch) -> None:
    release = _build_code_release(tmp_path)
    environment = _build_environment(tmp_path)
    module = _load_release_module(release)
    _patch_runtime(monkeypatch, module, environment)

    result = _collect(module, environment)

    assert result["schema"] == "wam_handeye_provenance_v1"
    assert result["status"] == "VERIFIED"
    assert result["code_release"]["release_id"] == release.name
    assert result["code_release"]["artifact_kind"] == "handeye_calibration_code"
    assert result["code_release"]["source_role"] == "mac"
    assert result["code_release"]["target_role"] == "linbox"
    assert set(result["code_release"]["sources"]) == set(SOURCE_PATHS)
    for row in result["code_release"]["sources"].values():
        assert set(row) == {"line_count", "line_sha256", "path", "sha256"}
        assert len(row["sha256"]) == len(row["line_sha256"]) == 64
    assert result["environment"] == {
        "environment_id": environment["root"].name,
        "lock_sha256": environment["lock_sha256"],
        "package_versions": {
            "cv2": PACKAGES["opencv-python"],
            "numpy": PACKAGES["numpy"],
            "pyorbbecsdk2": PACKAGES["pyorbbecsdk2"],
            "scipy": PACKAGES["scipy"],
            "yourdfpy": PACKAGES["yourdfpy"],
        },
        "python": {
            "executable_sha256": _sha256(environment["executable"].read_bytes()),
            "implementation": "CPython",
            "version": "3.11.15",
        },
        "clone_marker_sha256": environment["report_sha256"],
    }
    serialized = json.dumps(result, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "SECRET" not in serialized and "TOKEN" not in serialized


def test_rejects_symlinked_environment_evidence(tmp_path, monkeypatch) -> None:
    release = _build_code_release(tmp_path)
    environment = _build_environment(tmp_path)
    module = _load_release_module(release)
    _patch_runtime(monkeypatch, module, environment)
    copied = tmp_path / "copied-report.json"
    copied.write_bytes(environment["report_path"].read_bytes())
    environment["report_path"].unlink()
    environment["report_path"].symlink_to(copied)

    with pytest.raises(module.HandeyeProvenanceError, match="symlink|direct"):
        _collect(module, environment)


def test_rejects_code_release_completed_for_wrong_target(tmp_path, monkeypatch) -> None:
    release = _build_code_release(tmp_path, target_role="h200")
    environment = _build_environment(tmp_path)
    module = _load_release_module(release)
    _patch_runtime(monkeypatch, module, environment)

    with pytest.raises(module.HandeyeProvenanceError, match="target.*linbox"):
        _collect(module, environment)


def test_rejects_python_outside_direct_environment_release(tmp_path, monkeypatch) -> None:
    release = _build_code_release(tmp_path)
    environment = _build_environment(tmp_path)
    module = _load_release_module(release)
    _patch_runtime(monkeypatch, module, environment)
    other = tmp_path / "python"
    other.write_bytes(environment["executable"].read_bytes())
    other.chmod(0o755)
    monkeypatch.setattr(module.sys, "executable", str(other))

    with pytest.raises(module.HandeyeProvenanceError, match="direct perception environment"):
        _collect(module, environment)


@pytest.mark.parametrize("tamper", ["lock", "report_hash", "python_hash"])
def test_rejects_missing_or_mismatched_environment_hashes(
    tmp_path, monkeypatch, tamper: str,
) -> None:
    release = _build_code_release(tmp_path)
    environment = _build_environment(tmp_path)
    module = _load_release_module(release)
    _patch_runtime(monkeypatch, module, environment)
    if tamper == "lock":
        environment["lock_path"].unlink()
    elif tamper == "report_hash":
        environment["report_sha256"] = "0" * 64
    else:
        lock = json.loads(environment["lock_path"].read_text(encoding="utf-8"))
        lock["python"]["executable_sha256"] = "0" * 64
        data = _canonical_json(lock)
        environment["lock_path"].write_bytes(data)
        environment["lock_sha256"] = _sha256(data)

    with pytest.raises(module.HandeyeProvenanceError, match="missing|hash|Python"):
        _collect(module, environment)
