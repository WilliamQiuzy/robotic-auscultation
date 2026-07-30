from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import artifact_contract as contract  # noqa: E402
import publish_perception_runtime as publisher  # noqa: E402
import verify_perception_environment as verifier  # noqa: E402


AUSC_ROOT = Path(__file__).resolve().parents[2]
CURRENT_LOCK_PATH = (
    AUSC_ROOT / "configs" / "perception_runtime" / "linbox-ausc-live.lock.json"
)
LOCK_PATH = (
    AUSC_ROOT / "configs" / "perception_runtime" / "linbox-ausc-live-v1.lock.json"
)


def _lock() -> tuple[dict, str]:
    raw = LOCK_PATH.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _observation(root: Path, lock=None) -> dict:
    if lock is None:
        lock, _lock_sha256 = _lock()
    inventory = (
        copy.deepcopy(lock["distribution_inventory"])
        if lock["schema"] == publisher.ENVIRONMENT_LOCK_SCHEMA
        else {
            "algorithm": verifier.LEGACY_DISTRIBUTION_INVENTORY_ALGORITHM,
            "count": lock["distribution_inventory"]["count"],
            "sha256": "9" * 64,
        }
    )
    return {
        "environment_root": str(root),
        "python": {
            **lock["python"],
            "executable": str(root / "bin" / "python"),
        },
        "distribution_inventory": inventory,
        "packages": copy.deepcopy(lock["packages"]),
        "modules": {
            name: {
                "algorithm": identity.get(
                    "algorithm", verifier.MODULE_TREE_ALGORITHM,
                ),
                "count": identity.get("count", 1),
                "entrypoint_sha256": identity["sha256"],
                "path": str(root / "lib" / "python3.11" / "site-packages" / name / "__init__.py"),
                "sha256": identity["sha256"],
            }
            for name, identity in lock["modules"].items()
        },
        "cuda": {
            **lock["cuda"],
            "runtime_available": True,
            "device_capabilities": ["sm_120"],
        },
    }


def _marker(
    root: Path,
    observation: dict,
    lock=None,
    lock_sha256=None,
) -> dict:
    if lock is None:
        lock, default_lock_sha256 = _lock()
        lock_sha256 = default_lock_sha256 if lock_sha256 is None else lock_sha256
    if lock_sha256 is None:
        lock_sha256 = hashlib.sha256(contract.canonical_json_bytes(lock)).hexdigest()
    return {
        "schema": verifier.CLONE_MARKER_SCHEMA,
        "complete": True,
        "environment_id": lock["environment_id"],
        "environment_root": str(root),
        "python_executable_relative": "bin/python",
        "lock_sha256": lock_sha256,
        "source_environment_fingerprint_sha256": lock["provisioning"][
            "source_environment_fingerprint_sha256"
        ],
        "observed_distribution_inventory": copy.deepcopy(
            observation["distribution_inventory"]
        ),
    }


def _gate_lock() -> tuple[dict, str]:
    lock, _legacy_sha256 = _lock()
    lock["schema"] = publisher.ENVIRONMENT_LOCK_SCHEMA
    by_name = {row["name"]: row for row in lock["packages"]}
    for index, (name, version) in enumerate(sorted(publisher.SUPPORTED_PACKAGES.items())):
        by_name.setdefault(name, {
            "name": name,
            "record_sha256": format(index + 1, "x") * 64,
            "version": version,
        })
    lock["packages"] = [
        by_name[name] for name in sorted(publisher.SUPPORTED_PACKAGES)
    ]
    lock["modules"] = {
        name: {
            "algorithm": verifier.MODULE_TREE_ALGORITHM,
            "count": index + 1,
            "sha256": (
                "a" * 64 if name == "ultralytics"
                else lock["modules"][name]["sha256"]
            ),
        }
        for index, name in enumerate(sorted(publisher.SUPPORTED_MODULES))
    }
    lock["distribution_inventory"]["algorithm"] = (
        publisher.DISTRIBUTION_INVENTORY_ALGORITHM
    )
    lock["provisioning"]["source_environment_fingerprint_algorithm"] = (
        publisher.SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
    )
    lock["environment_id"] = (
        "linbox-perception-" + publisher.environment_identity_sha256(lock)[:20]
    )
    raw = contract.canonical_json_bytes(lock)
    return lock, hashlib.sha256(raw).hexdigest()


def test_verifies_reconstructable_identity_and_marks_opaque_lock_fields(
    tmp_path: Path,
) -> None:
    lock, lock_sha256 = verifier.load_environment_lock(LOCK_PATH)
    root = tmp_path / "ausc-live-clone"
    observation = _observation(root)
    marker = _marker(root, observation)

    report = verifier.verify_observation(
        lock=lock,
        lock_sha256=lock_sha256,
        marker=marker,
        marker_sha256="8" * 64,
        marker_path=root / verifier.CLONE_MARKER_NAME,
        observation=observation,
    )

    assert report["status"] == "VERIFIED_WITH_SCHEMA_GAPS"
    assert report["environment_id"] == lock["environment_id"]
    assert report["checks"]["python"] == "VERIFIED"
    assert report["checks"]["critical_distributions"] == "VERIFIED"
    assert report["checks"]["module_files"] == "VERIFIED"
    assert report["checks"]["cuda"] == "VERIFIED"
    assert report["checks"]["clone_marker"] == "VERIFIED"
    assert report["checks"]["lock_distribution_inventory_sha256"] == (
        "OPAQUE_BOUND_BY_CLONE_MARKER"
    )
    assert report["checks"]["source_environment_fingerprint_sha256"] == (
        "OPAQUE_BOUND_BY_CLONE_MARKER"
    )
    assert report["schema_gaps"] == [
        "distribution_inventory.sha256 lacks an algorithm and row projection in wam_perception_environment_lock_v1",
        "provisioning.source_environment_fingerprint_sha256 lacks an algorithm in wam_perception_environment_lock_v1",
    ]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value["python"].update(version="3.11.14"), "python identity"),
        (
            lambda value: value["packages"][0].update(record_sha256="0" * 64),
            "critical distributions",
        ),
        (
            lambda value: value["modules"]["torch"].update(
                entrypoint_sha256="0" * 64,
            ),
            "module files",
        ),
        (
            lambda value: value["cuda"].update(architectures=["sm_90"]),
            "CUDA build/architectures",
        ),
        (
            lambda value: value["cuda"].update(device_capabilities=["sm_90"]),
            "sm_120 runtime device",
        ),
    ],
)
def test_fails_closed_on_live_identity_mismatch(
    tmp_path: Path, mutate, match: str,
) -> None:
    lock, lock_sha256 = verifier.load_environment_lock(LOCK_PATH)
    root = tmp_path / "ausc-live-clone"
    observation = _observation(root)
    marker = _marker(root, observation)
    mutate(observation)

    with pytest.raises(verifier.EnvironmentVerificationError, match=match):
        verifier.verify_observation(
            lock=lock,
            lock_sha256=lock_sha256,
            marker=marker,
            marker_sha256="8" * 64,
            marker_path=root / verifier.CLONE_MARKER_NAME,
            observation=observation,
        )


def test_fails_closed_when_clone_marker_does_not_bind_observed_inventory(
    tmp_path: Path,
) -> None:
    lock, lock_sha256 = verifier.load_environment_lock(LOCK_PATH)
    root = tmp_path / "ausc-live-clone"
    observation = _observation(root)
    marker = _marker(root, observation)
    marker["observed_distribution_inventory"]["sha256"] = "0" * 64

    with pytest.raises(verifier.EnvironmentVerificationError, match="observed inventory"):
        verifier.verify_observation(
            lock=lock,
            lock_sha256=lock_sha256,
            marker=marker,
            marker_sha256="8" * 64,
            marker_path=root / verifier.CLONE_MARKER_NAME,
            observation=observation,
        )


def test_fails_closed_when_marker_is_not_inside_the_environment_root(
    tmp_path: Path,
) -> None:
    lock, lock_sha256 = verifier.load_environment_lock(LOCK_PATH)
    root = tmp_path / "ausc-live-clone"
    observation = _observation(root)
    marker = _marker(root, observation)

    with pytest.raises(verifier.EnvironmentVerificationError, match="marker path"):
        verifier.verify_observation(
            lock=lock,
            lock_sha256=lock_sha256,
            marker=marker,
            marker_sha256="8" * 64,
            marker_path=tmp_path / verifier.CLONE_MARKER_NAME,
            observation=observation,
        )


def test_distribution_inventory_algorithm_is_canonical_and_name_normalized() -> None:
    assert (
        verifier.DISTRIBUTION_INVENTORY_ALGORITHM
        == publisher.DISTRIBUTION_INVENTORY_ALGORITHM
    )
    assert verifier.MODULE_TREE_ALGORITHM == publisher.MODULE_TREE_ALGORITHM
    first = [
        {
            "installed_paths": ["lib/shared.py", "lib/z.py"],
            "name": "Z_pkg",
            "record_entries_count": 2,
            "version": "2",
            "record_sha256": "2" * 64,
        },
        {
            "installed_paths": ["lib/shared.py"],
            "name": "a.pkg",
            "record_entries_count": 1,
            "version": "1",
            "record_sha256": "1" * 64,
        },
    ]
    second = list(reversed(first))
    installed_paths = {
        "lib/shared.py": {
            "device": 1,
            "inode": 2,
            "owners": {"z-pkg", "a-pkg"},
            "path": "lib/shared.py",
            "record_path": "lib/shared.py",
            "sha256": "3" * 64,
            "size": 7,
        },
        "lib/z.py": {
            "device": 1,
            "inode": 3,
            "owners": {"z-pkg"},
            "path": "lib/z.py",
            "record_path": "lib/z.py",
            "sha256": "4" * 64,
            "size": 9,
        },
    }

    identity = verifier.canonical_distribution_inventory(first, installed_paths)

    assert identity == verifier.canonical_distribution_inventory(
        second, installed_paths,
    )
    assert identity["algorithm"] == verifier.DISTRIBUTION_INVENTORY_ALGORITHM
    assert identity["count"] == 2
    expected_rows = [
        {
            "name": "a-pkg",
            "record_sha256": "1" * 64,
            "record_entries_count": 1,
            "installed_paths_count": 1,
            "version": "1",
        },
        {
            "name": "z-pkg",
            "record_sha256": "2" * 64,
            "record_entries_count": 2,
            "installed_paths_count": 2,
            "version": "2",
        },
    ]
    assert identity["sha256"] == hashlib.sha256(
        contract.canonical_json_bytes({
            "distributions": expected_rows,
            "installed_paths": [
                {
                    "owners": ["a-pkg", "z-pkg"],
                    "path": "lib/shared.py",
                    "record_path": "lib/shared.py",
                    "sha256": "3" * 64,
                    "size": 7,
                },
                {
                    "owners": ["z-pkg"],
                    "path": "lib/z.py",
                    "record_path": "lib/z.py",
                    "sha256": "4" * 64,
                    "size": 9,
                },
            ],
        })
    ).hexdigest()


class _FixtureDistribution:
    def __init__(self, site_packages: Path, name: str = "Demo_Pkg") -> None:
        self.metadata = {"Name": name}
        self.version = "1.2.3"
        self._site_packages = site_packages
        self._path = site_packages / "demo_pkg-1.2.3.dist-info"

    def locate_file(self, relative: str) -> Path:
        return self._site_packages / relative


def _fixture_distribution(root: Path) -> _FixtureDistribution:
    site = root / "lib" / "python3.11" / "site-packages"
    dist = _FixtureDistribution(site)
    dist._path.mkdir(parents=True)
    (site / "demo_pkg").mkdir()
    (site / "demo_pkg" / "__init__.py").write_bytes(b"VERSION = 1\n")
    (site / "demo_pkg" / "native.so").write_bytes(b"native-v1")
    (dist._path / "METADATA").write_bytes(b"Name: Demo-Pkg\nVersion: 1.2.3\n")
    (dist._path / "RECORD").write_text(
        "demo_pkg/__init__.py,,\n"
        "demo_pkg/native.so,,\n"
        "demo_pkg-1.2.3.dist-info/METADATA,,\n"
        "demo_pkg-1.2.3.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    return dist


def test_distribution_identity_hashes_actual_record_files_not_only_record_text(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    distribution = _fixture_distribution(root)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        first_claimed = {}
        first = verifier.collect_distribution_identity(
            distribution,
            environment_root=root,
            environment_fd=root_fd,
            claimed_paths=first_claimed,
        )
        first_identity = verifier.canonical_distribution_inventory(
            [first], first_claimed,
        )
        (root / "lib/python3.11/site-packages/demo_pkg/native.so").write_bytes(
            b"native-v2"
        )
        second_claimed = {}
        second = verifier.collect_distribution_identity(
            distribution,
            environment_root=root,
            environment_fd=root_fd,
            claimed_paths=second_claimed,
        )
        second_identity = verifier.canonical_distribution_inventory(
            [second], second_claimed,
        )
    finally:
        os.close(root_fd)

    assert first["record_sha256"] == second["record_sha256"]
    assert first["record_entries_count"] == 4
    assert first["installed_paths"] == sorted(first["installed_paths"])
    assert first["record_sha256"] == second["record_sha256"]
    assert first_identity["sha256"] != second_identity["sha256"]


@pytest.mark.parametrize("violation", ["missing", "escape", "spelling", "symlink"])
def test_distribution_record_paths_fail_closed(
    tmp_path: Path, violation: str,
) -> None:
    root = tmp_path / "env"
    distribution = _fixture_distribution(root)
    record = distribution._path / "RECORD"
    if violation == "missing":
        record.write_text("missing.bin,,\n", encoding="utf-8")
    elif violation == "escape":
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"outside")
        record.write_text("../../../../../outside.bin,,\n", encoding="utf-8")
    elif violation == "spelling":
        record.write_text(
            "demo_pkg/native.so,,\ndemo_pkg/../demo_pkg/native.so,,\n",
            encoding="utf-8",
        )
    else:
        native = root / "lib/python3.11/site-packages/demo_pkg/native.so"
        target = native.with_name("native-target.so")
        native.rename(target)
        native.symlink_to(target.name)
        record.write_text("demo_pkg/native.so,,\n", encoding="utf-8")
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(
            verifier.EnvironmentVerificationError,
            match="missing|escape|duplicate|spelling|symlink|regular",
        ):
            verifier.collect_distribution_identity(
                distribution,
                environment_root=root,
                environment_fd=root_fd,
                claimed_paths={},
            )
    finally:
        os.close(root_fd)


def test_legacy_record_text_inventory_allows_missing_repair_debt_only_without_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    distribution = _fixture_distribution(root)
    record = distribution._path / "RECORD"
    record.write_text("../../../bin/pip3.13,,\n", encoding="utf-8")
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        row = verifier.collect_distribution_identity(
            distribution,
            environment_root=root,
            environment_fd=root_fd,
            claimed_paths={},
            observe_installed_paths=False,
        )
    finally:
        os.close(root_fd)

    assert row["installed_paths"] == ["bin/pip3.13"]
    assert row["record_entries_count"] == 1


def test_legacy_record_text_inventory_still_rejects_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "env"
    distribution = _fixture_distribution(root)
    (distribution._path / "RECORD").write_text(
        "../../../../../../outside.bin,,\n", encoding="utf-8"
    )
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(
            verifier.EnvironmentVerificationError, match="escapes the environment root"
        ):
            verifier.collect_distribution_identity(
                distribution,
                environment_root=root,
                environment_fd=root_fd,
                claimed_paths={},
                observe_installed_paths=False,
            )
    finally:
        os.close(root_fd)


def test_shared_nvidia_namespace_path_hashes_bytes_once_with_sorted_owners(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    first = _fixture_distribution(root)
    owner_names = [
        "NVIDIA_Cublas_Cu12",
        "NVIDIA_Cuda_Cupti_Cu12",
        "NVIDIA_Cuda_Nvrtc_Cu12",
        "NVIDIA_Cuda_Runtime_Cu12",
        "NVIDIA_Cudnn_Cu12",
        "NVIDIA_Cufft_Cu12",
        "NVIDIA_Cufile_Cu12",
        "NVIDIA_Curand_Cu12",
        "NVIDIA_Cusolver_Cu12",
        "NVIDIA_Cusparse_Cu12",
        "NVIDIA_Nccl_Cu12",
    ]
    first.metadata = {"Name": owner_names[0]}
    shared = first._site_packages / "nvidia" / "__init__.py"
    shared.parent.mkdir()
    shared.write_bytes(b"shared namespace\n")
    (first._path / "RECORD").write_text(
        "nvidia/__init__.py,,\n",
        encoding="utf-8",
    )
    other_distributions = []
    for index, owner in enumerate(owner_names[1:], 1):
        distribution = _FixtureDistribution(first._site_packages, owner)
        distribution._path = (
            first._site_packages / ("nvidia_owner_%d.dist-info" % index)
        )
        distribution._path.mkdir()
        (distribution._path / "RECORD").write_text(
            "nvidia/__init__.py,,\n", encoding="utf-8",
        )
        other_distributions.append(distribution)
    claimed = {}
    rows = []
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        rows.append(verifier.collect_distribution_identity(
            first,
            environment_root=root,
            environment_fd=root_fd,
            claimed_paths=claimed,
        ))
        for distribution in other_distributions:
            rows.append(verifier.collect_distribution_identity(
                distribution,
                environment_root=root,
                environment_fd=root_fd,
                claimed_paths=claimed,
            ))
    finally:
        os.close(root_fd)

    relative = "lib/python3.11/site-packages/nvidia/__init__.py"
    expected_owners = {
        verifier._normalize_distribution_name(name) for name in owner_names
    }
    assert claimed[relative]["owners"] == expected_owners
    identity = verifier.canonical_distribution_inventory(
        list(reversed(rows)), claimed,
    )
    assert identity["count"] == 11


def test_shared_path_rejects_content_or_inode_change_between_owners(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    first = _fixture_distribution(root)
    shared = first._site_packages / "nvidia" / "__init__.py"
    shared.parent.mkdir()
    shared.write_bytes(b"first content\n")
    (first._path / "RECORD").write_text(
        "nvidia/__init__.py,,\n", encoding="utf-8",
    )
    second = _FixtureDistribution(first._site_packages, "NVIDIA_Cublas_Cu12")
    second._path = first._site_packages / "nvidia_cublas_cu12-1.dist-info"
    second._path.mkdir()
    (second._path / "RECORD").write_text(
        "nvidia/__init__.py,,\n", encoding="utf-8",
    )
    claimed = {}
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        verifier.collect_distribution_identity(
            first,
            environment_root=root,
            environment_fd=root_fd,
            claimed_paths=claimed,
        )
        replacement = shared.with_name("replacement.py")
        replacement.write_bytes(b"other content\n")
        replacement.replace(shared)
        with pytest.raises(
            verifier.EnvironmentVerificationError,
            match="shared installed path observation conflicts",
        ):
            verifier.collect_distribution_identity(
                second,
                environment_root=root,
                environment_fd=root_fd,
                claimed_paths=claimed,
            )
    finally:
        os.close(root_fd)


def test_same_owner_duplicate_record_rows_are_bound_not_silently_dropped(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    distribution = _fixture_distribution(root)
    pyc = root / "lib/python3.11/site-packages/numpy/distutils/__pycache__/conv_template.cpython-311.pyc"
    pyc.parent.mkdir(parents=True)
    pyc.write_bytes(b"compiled-template")
    record = distribution._path / "RECORD"
    record.write_text(
        "numpy/distutils/__pycache__/conv_template.cpython-311.pyc,,\n"
        "numpy/distutils/__pycache__/conv_template.cpython-311.pyc,sha256=abc,17\n",
        encoding="utf-8",
    )
    claimed = {}
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        first = verifier.collect_distribution_identity(
            distribution,
            environment_root=root,
            environment_fd=root_fd,
            claimed_paths=claimed,
        )
        first_identity = verifier.canonical_distribution_inventory(
            [first], claimed,
        )
        record.write_text(
            "numpy/distutils/__pycache__/conv_template.cpython-311.pyc,,\n"
            "numpy/distutils/__pycache__/conv_template.cpython-311.pyc,sha256=changed,17\n",
            encoding="utf-8",
        )
        second_claimed = {}
        second = verifier.collect_distribution_identity(
            distribution,
            environment_root=root,
            environment_fd=root_fd,
            claimed_paths=second_claimed,
        )
        second_identity = verifier.canonical_distribution_inventory(
            [second], second_claimed,
        )
    finally:
        os.close(root_fd)

    assert first["record_entries_count"] == 2
    assert len(first["installed_paths"]) == 1
    assert first["record_sha256"] != second["record_sha256"]
    assert first_identity["sha256"] != second_identity["sha256"]


def test_canonical_inventory_rejects_shared_owner_drift() -> None:
    rows = [{
        "installed_paths": ["lib/shared.py"],
        "name": "owner-a",
        "record_entries_count": 1,
        "record_sha256": "1" * 64,
        "version": "1",
    }]
    claimed = {
        "lib/shared.py": {
            "device": 1,
            "inode": 2,
            "owners": {"owner-a", "unexpected-owner"},
            "path": "lib/shared.py",
            "record_path": "lib/shared.py",
            "sha256": "2" * 64,
            "size": 1,
        },
    }

    with pytest.raises(
        verifier.EnvironmentVerificationError, match="ownership drift",
    ):
        verifier.canonical_distribution_inventory(rows, claimed)


def test_module_identity_hashes_complete_package_tree_including_native_code(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    package = root / "lib/python3.11/site-packages/demo_pkg"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    native = package / "native.so"
    init.write_bytes(b"top-level-stable\n")
    native.write_bytes(b"native-v1")
    module = SimpleNamespace(
        __file__=str(init),
        __spec__=SimpleNamespace(submodule_search_locations=[str(package)]),
    )
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        first = verifier.collect_module_tree_identity(
            "demo_pkg", module, environment_root=root, environment_fd=root_fd
        )
        native.write_bytes(b"native-v2")
        second = verifier.collect_module_tree_identity(
            "demo_pkg", module, environment_root=root, environment_fd=root_fd
        )
    finally:
        os.close(root_fd)

    assert first["algorithm"] == verifier.MODULE_TREE_ALGORITHM
    assert first["count"] == 2
    assert first["sha256"] != second["sha256"]


def test_source_fingerprint_is_path_independent_and_content_sensitive(tmp_path: Path) -> None:
    lock, _ = _lock()
    first = _observation(tmp_path / "first", lock)
    second = _observation(tmp_path / "second", lock)

    assert publisher.source_environment_fingerprint_sha256(first) == (
        publisher.source_environment_fingerprint_sha256(second)
    )
    second["modules"]["torch"]["sha256"] = "0" * 64
    assert publisher.source_environment_fingerprint_sha256(first) != (
        publisher.source_environment_fingerprint_sha256(second)
    )


def test_checked_in_lock_is_canonical_and_validated_by_publisher() -> None:
    lock, lock_sha256 = verifier.load_environment_lock(CURRENT_LOCK_PATH)

    assert publisher.validate_environment_lock(lock) == lock
    assert lock_sha256 == hashlib.sha256(CURRENT_LOCK_PATH.read_bytes()).hexdigest()


def test_checked_in_legacy_v2_lock_is_read_only_with_explicit_schema_gaps(
    tmp_path: Path,
) -> None:
    lock, lock_sha256 = verifier.load_environment_lock(CURRENT_LOCK_PATH)
    root = tmp_path / "legacy-v2-clone"
    observation = _observation(root, lock)
    marker = _marker(root, observation, lock, lock_sha256)

    report = verifier.verify_observation(
        lock=lock,
        lock_sha256=lock_sha256,
        marker=marker,
        marker_sha256="8" * 64,
        marker_path=root / verifier.CLONE_MARKER_NAME,
        observation=observation,
    )

    assert report["status"] == "VERIFIED_WITH_SCHEMA_GAPS"
    assert report["schema_gaps"] == [
        "distribution inventory locks RECORD text but not installed file bytes",
        "module identities lock only entrypoint files, not complete module trees",
    ]


def test_noncanonical_lock_is_rejected_before_runtime_probe(tmp_path: Path) -> None:
    lock, _lock_sha256 = _lock()
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    with pytest.raises(verifier.EnvironmentVerificationError, match="canonical JSON"):
        verifier.load_environment_lock(path)


def test_cli_emits_canonical_success_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "schema": verifier.VERIFICATION_SCHEMA,
        "status": "VERIFIED_WITH_SCHEMA_GAPS",
    }
    monkeypatch.setattr(verifier, "verify_current_environment", lambda **_kwargs: expected)

    assert verifier.main(["--lock", str(LOCK_PATH)]) == 0
    raw = capsys.readouterr().out.encode("utf-8")
    assert raw == contract.canonical_json_bytes(expected)


def test_cli_returns_nonzero_and_canonical_error_on_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(**_kwargs):
        raise verifier.EnvironmentVerificationError("torch module files mismatch")

    monkeypatch.setattr(verifier, "verify_current_environment", fail)

    assert verifier.main(["--lock", str(LOCK_PATH)]) == 1
    raw = capsys.readouterr().err.encode("utf-8")
    value = json.loads(raw)
    assert raw == contract.canonical_json_bytes(value)
    assert value == {
        "error": "torch module files mismatch",
        "schema": verifier.VERIFICATION_SCHEMA,
        "status": "FAILED",
    }


def test_native_module_import_noise_cannot_corrupt_canonical_stdout(
    capfd: pytest.CaptureFixture[str],
) -> None:
    sentinel = object()

    def noisy_import(_name: str):
        os.write(1, b"native extension noise\n")
        os.write(2, b"native extension warning\n")
        return sentinel

    assert verifier._import_modules_quietly(["native"], importer=noisy_import) == {
        "native": sentinel,
    }
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_yolo_gate_module_is_a_fail_closed_environment_dependency() -> None:
    assert verifier.CRITICAL_MODULES == (
        "cv2", "pyorbbecsdk", "torch", "ultralytics",
    )

    imported = []

    def recording_import(name: str):
        imported.append(name)
        return object()

    verifier._import_modules_quietly(
        verifier.CRITICAL_MODULES,
        importer=recording_import,
    )
    assert imported == list(verifier.CRITICAL_MODULES)


def test_gate_profile_hashes_ultralytics_and_rejects_module_drift(
    tmp_path: Path,
) -> None:
    lock, lock_sha256 = _gate_lock()
    root = tmp_path / "gate-env"
    observation = _observation(root, lock)
    marker = _marker(root, observation, lock, lock_sha256)

    report = verifier.verify_observation(
        lock=lock,
        lock_sha256=lock_sha256,
        marker=marker,
        marker_sha256="8" * 64,
        marker_path=root / verifier.CLONE_MARKER_NAME,
        observation=observation,
    )
    assert report["modules"]["ultralytics"]["sha256"] == "a" * 64
    assert report["status"] == "VERIFIED"
    assert report["schema_gaps"] == []
    assert report["checks"]["lock_distribution_inventory_sha256"] == "VERIFIED"

    observation["modules"]["ultralytics"]["sha256"] = "0" * 64
    with pytest.raises(verifier.EnvironmentVerificationError, match="ultralytics"):
        verifier.verify_observation(
            lock=lock,
            lock_sha256=lock_sha256,
            marker=marker,
            marker_sha256="8" * 64,
            marker_path=root / verifier.CLONE_MARKER_NAME,
            observation=observation,
        )


def test_gate_profile_rejects_complete_distribution_inventory_drift(
    tmp_path: Path,
) -> None:
    lock, lock_sha256 = _gate_lock()
    root = tmp_path / "gate-env"
    observation = _observation(root, lock)
    marker = _marker(root, observation, lock, lock_sha256)
    observation["distribution_inventory"]["sha256"] = "0" * 64
    marker["observed_distribution_inventory"] = copy.deepcopy(
        observation["distribution_inventory"]
    )

    with pytest.raises(
        verifier.EnvironmentVerificationError,
        match="distribution inventory identity",
    ):
        verifier.verify_observation(
            lock=lock,
            lock_sha256=lock_sha256,
            marker=marker,
            marker_sha256="8" * 64,
            marker_path=root / verifier.CLONE_MARKER_NAME,
            observation=observation,
        )
