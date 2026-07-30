from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import artifact_contract as contract  # noqa: E402
import perception_environment_manifest as manifest  # noqa: E402
import publish_perception_runtime as publisher  # noqa: E402
import verify_perception_environment as verifier  # noqa: E402


def _digest(value: object) -> str:
    return hashlib.sha256(contract.canonical_json_bytes(value)).hexdigest()


def test_v3_schema_is_canonical_and_documents_the_final_path_contract() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/perception_runtime/perception-environment-lock-v3.schema.json"
    )
    raw = path.read_bytes()
    schema = json.loads(raw.decode("utf-8"))

    assert raw == contract.canonical_json_bytes(schema)
    assert schema["$id"] == "wam://schemas/perception-environment-lock-v3"
    assert schema["properties"]["schema"]["const"] == (
        publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA
    )
    assert schema["$defs"]["build_recipe"]["properties"]["algorithm"][
        "const"
    ] == publisher.FINAL_PATH_BUILD_RECIPE_ALGORITHM
    assert schema["$defs"]["build_recipe"]["properties"]["source_environment"][
        "required"
    ] == [
        "environment_id",
        "lock_sha256",
        "marker_sha256",
        "path",
        "snapshot_sha256",
    ]
    assert schema["$defs"]["record_observation"]["properties"]["algorithm"][
        "const"
    ] == manifest.RECORD_OBSERVATION_ALGORITHM
    assert schema["$defs"]["environment_tree"]["properties"]["algorithm"][
        "const"
    ] == manifest.ENVIRONMENT_TREE_ALGORITHM
    assert schema["$defs"]["ffmpeg"]["properties"]["path"]["const"] == (
        publisher.BUNDLED_FFMPEG_RELATIVE_PATH
    )


def _pyc_evidence() -> dict:
    return manifest.generated_pyc_omission_evidence(
        path=manifest.GENERATED_PYC_OMISSION,
        source_path=(
            "lib/python3.11/site-packages/numpy/distutils/conv_template.py"
        ),
        source_sha256="a" * 64,
        inspection_sha256="b" * 64,
    )


def _record_observation() -> tuple[dict, list[dict]]:
    distributions = []
    tree_rows = [{"kind": "directory", "mode": "0555", "path": "."}]
    for index, (name, version) in enumerate(
        sorted(publisher.RELOCATION_AWARE_SUPPORTED_PACKAGES.items()), 1
    ):
        path = f"lib/python3.11/site-packages/{name.replace('-', '_')}/payload-{index}.bin"
        data_sha256 = format(index, "02x") * 32
        size = index
        claim = {
            "actual": {"sha256": data_sha256, "size": size},
            "expected": {
                "hash_field": "",
                "sha256": None,
                "size": None,
                "size_field": "",
            },
            "installed_path": path,
            "record_index": 1,
            "record_path": f"{name.replace('-', '_')}/payload-{index}.bin",
            "status": "record_unhashed",
        }
        claims = [claim]
        if name == "numpy":
            claims.append({
                "actual": {"sha256": None, "size": None},
                "expected": {
                    "hash_field": "sha256=YWJj",
                    "sha256": None,
                    "size": 10,
                    "size_field": "10",
                },
                "installed_path": manifest.GENERATED_PYC_OMISSION,
                "omission_evidence_sha256": _digest(_pyc_evidence()),
                "record_index": 2,
                "record_path": (
                    "numpy/distutils/__pycache__/conv_template.cpython-311.pyc"
                ),
                "status": "omitted_generated_pyc",
            })
        distributions.append({
            "claims": claims,
            "name": name,
            "record_entries_count": len(claims),
            "record_path": (
                f"lib/python3.11/site-packages/{name.replace('-', '_')}-{version}.dist-info/RECORD"
            ),
            "record_sha256": format(index + 20, "02x") * 32,
            "version": version,
        })
        tree_rows.append({
            "kind": "regular",
            "mode": "0444",
            "path": path,
            "sha256": data_sha256,
            "size": size,
        })
    status_counts = {status: 0 for status in manifest.RECORD_STATUSES}
    status_counts["record_unhashed"] = len(distributions)
    status_counts["omitted_generated_pyc"] = 1
    projection = {
        "distributions": distributions,
        "status_counts": status_counts,
    }
    return {
        "algorithm": manifest.RECORD_OBSERVATION_ALGORITHM,
        "claim_count": len(distributions) + 1,
        "distribution_count": len(distributions),
        "distributions": distributions,
        "sha256": _digest(projection),
        "status_counts": status_counts,
    }, tree_rows


def _tree(rows: list[dict]) -> dict:
    ffmpeg = publisher.BUNDLED_FFMPEG_RELATIVE_PATH
    ffmpeg_row = {
        "kind": "regular",
        "mode": "0555",
        "path": ffmpeg,
        "sha256": "e" * 64,
        "size": 123456,
    }
    rows = sorted([*rows, ffmpeg_row], key=lambda row: row["path"])
    exclusions = {
        "required_absent_generated_cache": [manifest.GENERATED_PYC_OMISSION],
        "self_referential": list(manifest.SELF_REFERENCE_EXCLUSIONS),
    }
    return {
        "algorithm": manifest.ENVIRONMENT_TREE_ALGORITHM,
        "directories": 1,
        "entries": rows,
        "entries_count": len(rows),
        "exclusions": exclusions,
        "regular_files": len(rows) - 1,
        "sha256": _digest({"entries": rows, "exclusions": exclusions}),
        "symlinks": 0,
    }


def _build_recipe(repair_manifest_sha256: str) -> dict:
    source_id = "linbox-perception-ff99f690fa05a46fb2f7"
    recipe = {
        "algorithm": publisher.FINAL_PATH_BUILD_RECIPE_ALGORITHM,
        "operations": list(publisher.FINAL_PATH_BUILD_OPERATIONS),
        "source_environment": {
            "environment_id": source_id,
            "lock_sha256": "1" * 64,
            "marker_sha256": "2" * 64,
            "path": (
                f"{publisher.LINBOX_PERCEPTION_RELEASES_ROOT}/{source_id}"
            ),
            "snapshot_sha256": "5" * 64,
        },
        "target": {
            "abi": "cp311",
            "cuda_build": "12.8",
            "platform": "linux-x86_64",
            "python_version": "3.11.15",
            "releases_root": publisher.LINBOX_PERCEPTION_RELEASES_ROOT,
            "strategy": "reserve-and-build-in-final-path-no-rename",
        },
        "wheelhouse_manifests": [
            {"purpose": "gate-additions", "sha256": "3" * 64},
            {"purpose": "opencv-repair", "sha256": "4" * 64},
            {"purpose": "runtime-repairs", "sha256": repair_manifest_sha256},
        ],
    }
    recipe["sha256"] = publisher.build_recipe_sha256(recipe)
    return recipe


def _v3_lock() -> dict:
    record, tree_rows = _record_observation()
    tree = _tree(tree_rows)
    packages = []
    for index, (name, version) in enumerate(
        sorted(publisher.RELOCATION_AWARE_SUPPORTED_PACKAGES.items()), 1
    ):
        packages.append({
            "name": name,
            "record_sha256": format(index + 20, "02x") * 32,
            "version": version,
        })
    pyc_evidence = _pyc_evidence()
    value = {
        "build_recipe": _build_recipe("c" * 64),
        "cuda": {
            "architectures": list(publisher.SUPPORTED_CUDA_ARCHITECTURES),
            "build": "12.8",
        },
        "environment_id": "linbox-perception-" + "0" * 20,
        "environment_tree": tree,
        "ffmpeg": {
            "distribution": "imageio-ffmpeg",
            "mode": "0555",
            "path": publisher.BUNDLED_FFMPEG_RELATIVE_PATH,
            "resolver": "imageio_ffmpeg.get_ffmpeg_exe",
            "sha256": "e" * 64,
            "size": 123456,
            "version": "0.6.0",
        },
        "modules": {
            name: {
                "algorithm": publisher.MODULE_TREE_ALGORITHM,
                "count": index,
                "sha256": format(index + 100, "02x") * 32,
            }
            for index, name in enumerate(
                sorted(publisher.RELOCATION_AWARE_SUPPORTED_MODULES), 1
            )
        },
        "packages": packages,
        "provisioning": {
            "generated_pyc_omission": pyc_evidence,
            "mode": "verified_clone_with_exact_repairs",
            "repair_wheelhouse": {
                "algorithm": publisher.REPAIR_WHEELHOUSE_ALGORITHM,
                "manifest_sha256": "c" * 64,
                "wheels": copy.deepcopy(publisher.EXACT_REPAIR_WHEELS),
            },
            "required_service_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
            "source_environment_fingerprint_algorithm": (
                publisher.SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
            ),
            "source_environment_fingerprint_sha256": "d" * 64,
            "wheelhouse_complete": False,
        },
        "python": {
            "abi": "cp311",
            "executable_sha256": "f" * 64,
            "implementation": "CPython",
            "platform": "linux-x86_64",
            "version": "3.11.15",
        },
        "record_observation": record,
        "schema": publisher.RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA,
    }
    value["environment_id"] = "linbox-perception-" + (
        value["build_recipe"]["sha256"][:20]
    )
    return value


def _refresh_record(value: dict) -> None:
    record = value["record_observation"]
    status_counts = {status: 0 for status in manifest.RECORD_STATUSES}
    for distribution in record["distributions"]:
        for claim in distribution["claims"]:
            status_counts[claim["status"]] += 1
    record["status_counts"] = status_counts
    record["sha256"] = _digest({
        "distributions": record["distributions"],
        "status_counts": status_counts,
    })


def _refresh_id(value: dict) -> None:
    value["environment_id"] = "linbox-perception-" + (
        value["build_recipe"]["sha256"][:20]
    )


def test_v3_lock_binds_complete_record_tree_repairs_and_ffmpeg() -> None:
    value = _v3_lock()

    assert publisher.validate_environment_lock(value) == value
    assert publisher.is_strong_environment_lock(value) is True
    assert value["provisioning"]["repair_wheelhouse"]["wheels"] == [
        {
            "filename": "imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl",
            "name": "imageio-ffmpeg",
            "sha256": "c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282",
            "size": 29498237,
            "url": "https://files.pythonhosted.org/packages/a0/2d/43c8522a2038e9d0e7dbdf3a61195ecc31ca576fb1527a528c877e87d973/imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl",
            "version": "0.6.0",
        },
        {
            "filename": "pip-26.1.2-py3-none-any.whl",
            "name": "pip",
            "sha256": "382ff9f685ee3bc25864f820aa50505825f10f5458ffff07e30a6d96e5715cab",
            "size": 1813144,
            "url": "https://files.pythonhosted.org/packages/5d/95/6b5cb3461ea5673ba0995989746db58eb18b91b54dbf331e72f569540946/pip-26.1.2-py3-none-any.whl",
            "version": "26.1.2",
        },
    ]


def test_v3_lock_accepts_duplicate_record_claims_for_one_audited_omission() -> None:
    value = _v3_lock()
    numpy_distribution = next(
        row for row in value["record_observation"]["distributions"]
        if row["name"] == "numpy"
    )
    omitted = next(
        claim for claim in numpy_distribution["claims"]
        if claim["status"] == "omitted_generated_pyc"
    )
    duplicate = copy.deepcopy(omitted)
    duplicate["record_index"] = len(numpy_distribution["claims"]) + 1
    duplicate["expected"] = {
        "hash_field": "",
        "sha256": None,
        "size": None,
        "size_field": "",
    }
    numpy_distribution["claims"].append(duplicate)
    numpy_distribution["record_entries_count"] += 1
    value["record_observation"]["claim_count"] += 1
    _refresh_record(value)

    assert value["record_observation"]["status_counts"][
        "omitted_generated_pyc"
    ] == 2
    assert publisher.validate_environment_lock(value) == value


def test_v3_lock_rejects_unresolved_record_claim_even_when_fingerprint_matches() -> None:
    value = _v3_lock()
    claim = value["record_observation"]["distributions"][0]["claims"][0]
    claim["status"] = "metadata_mismatch"
    _refresh_record(value)
    _refresh_id(value)

    with pytest.raises(publisher.PerceptionRuntimeError, match="unresolved RECORD"):
        publisher.validate_environment_lock(value)


@pytest.mark.parametrize("mutation", ["wheel-url", "ffmpeg-hash", "tree", "service-env"])
def test_v3_lock_rejects_any_repair_tree_or_runtime_policy_drift(
    mutation: str,
) -> None:
    value = _v3_lock()
    if mutation == "wheel-url":
        value["provisioning"]["repair_wheelhouse"]["wheels"][0]["url"] += "?mirror"
    elif mutation == "ffmpeg-hash":
        value["ffmpeg"]["sha256"] = "0" * 64
    elif mutation == "tree":
        value["environment_tree"]["entries"][-1]["mode"] = "0777"
    else:
        value["provisioning"]["required_service_environment"] = {}
    _refresh_id(value)

    with pytest.raises(publisher.PerceptionRuntimeError):
        publisher.validate_environment_lock(value)


def test_v3_environment_id_is_recipe_bound_not_tree_path_fixed_point() -> None:
    value = _v3_lock()
    first_id = value["environment_id"]
    ffmpeg_row = next(
        row for row in value["environment_tree"]["entries"]
        if row["path"] == publisher.BUNDLED_FFMPEG_RELATIVE_PATH
    )
    ffmpeg_row["sha256"] = "1" * 64
    value["environment_tree"]["sha256"] = _digest({
        "entries": value["environment_tree"]["entries"],
        "exclusions": value["environment_tree"]["exclusions"],
    })
    value["ffmpeg"]["sha256"] = "1" * 64
    _refresh_id(value)

    assert value["environment_id"] == first_id
    assert publisher.validate_environment_lock(value) == value


def test_v3_build_recipe_change_selects_a_new_final_release_path() -> None:
    value = _v3_lock()
    first_id = value["environment_id"]
    value["build_recipe"]["source_environment"]["marker_sha256"] = "5" * 64

    with pytest.raises(publisher.PerceptionRuntimeError, match="build recipe"):
        publisher.validate_environment_lock(value)

    value["build_recipe"]["sha256"] = publisher.build_recipe_sha256(
        value["build_recipe"]
    )
    _refresh_id(value)
    assert value["environment_id"] != first_id
    assert publisher.expected_final_environment_root(value) == Path(
        publisher.LINBOX_PERCEPTION_RELEASES_ROOT
    ) / value["environment_id"]
    assert publisher.validate_environment_lock(value) == value


def _v3_observation(root: Path, lock: dict) -> dict:
    return {
        "cuda": {
            **lock["cuda"],
            "device_capabilities": ["sm_120"],
            "runtime_available": True,
        },
        "environment_root": str(root),
        "environment_tree": copy.deepcopy(lock["environment_tree"]),
        "ffmpeg": {
            **copy.deepcopy(lock["ffmpeg"]),
            "absolute_path": str(root / lock["ffmpeg"]["path"]),
        },
        "modules": {
            name: {
                **copy.deepcopy(identity),
                "entrypoint_sha256": identity["sha256"],
                "path": str(root / "lib/python3.11/site-packages" / name),
            }
            for name, identity in lock["modules"].items()
        },
        "packages": copy.deepcopy(lock["packages"]),
        "python": {
            **copy.deepcopy(lock["python"]),
            "executable": str(root / "bin/python"),
        },
        "record_observation": copy.deepcopy(lock["record_observation"]),
        "runtime_policy": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "sys_dont_write_bytecode": True,
        },
    }


def test_live_collection_routes_v3_to_the_complete_final_path_collector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "env"
    executable = root / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python fixture")
    lock = _v3_lock()
    sentinel = {"complete": True}
    captured = {}

    def fake_complete_collector(*, lock: dict, root: Path, executable: Path):
        captured.update({"lock": lock, "root": root, "executable": executable})
        return sentinel

    monkeypatch.setattr(verifier.sys, "prefix", str(root))
    monkeypatch.setattr(verifier.sys, "executable", str(executable))
    monkeypatch.setattr(
        verifier,
        "_collect_relocation_aware_current_environment",
        fake_complete_collector,
    )

    assert verifier.collect_current_environment(
        lock=lock, environment_root=root,
    ) is sentinel
    assert captured == {"lock": lock, "root": root, "executable": executable}


def _v3_marker(root: Path, lock: dict, lock_sha256: str) -> dict:
    return {
        "complete": True,
        "environment_id": lock["environment_id"],
        "environment_root": str(root),
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


def test_v3_verification_reproduces_complete_record_and_tree_state(
    tmp_path: Path,
) -> None:
    lock = _v3_lock()
    lock_sha256 = _digest(lock)
    root = publisher.expected_final_environment_root(lock)
    observation = _v3_observation(root, lock)
    marker = _v3_marker(root, lock, lock_sha256)

    report = verifier.verify_observation(
        lock=lock,
        lock_sha256=lock_sha256,
        marker=marker,
        marker_sha256="9" * 64,
        marker_path=root / verifier.CLONE_MARKER_NAME,
        observation=observation,
    )

    assert report["status"] == "VERIFIED"
    assert report["checks"]["complete_record_observation"] == "VERIFIED"
    assert report["checks"]["complete_environment_tree"] == "VERIFIED"
    assert report["checks"]["bundled_ffmpeg"] == "VERIFIED"
    assert report["record_observation"] == verifier.record_observation_summary(
        lock["record_observation"]
    )
    assert report["environment_tree"] == verifier.environment_tree_summary(
        lock["environment_tree"]
    )


@pytest.mark.parametrize("mutation", ["record", "tree", "marker-tree", "runtime-policy"])
def test_v3_verification_rejects_any_observation_or_marker_drift(
    tmp_path: Path, mutation: str,
) -> None:
    lock = _v3_lock()
    lock_sha256 = _digest(lock)
    root = publisher.expected_final_environment_root(lock)
    observation = _v3_observation(root, lock)
    marker = _v3_marker(root, lock, lock_sha256)
    if mutation == "record":
        observation["record_observation"]["sha256"] = "0" * 64
    elif mutation == "tree":
        observation["environment_tree"]["sha256"] = "0" * 64
    elif mutation == "marker-tree":
        marker["observed_environment_tree"]["sha256"] = "0" * 64
    else:
        observation["runtime_policy"]["sys_dont_write_bytecode"] = False

    with pytest.raises(verifier.EnvironmentVerificationError):
        verifier.verify_observation(
            lock=lock,
            lock_sha256=lock_sha256,
            marker=marker,
            marker_sha256="9" * 64,
            marker_path=root / verifier.CLONE_MARKER_NAME,
            observation=observation,
        )


def test_v3_verification_rejects_a_clone_observed_outside_its_final_path(
    tmp_path: Path,
) -> None:
    lock = _v3_lock()
    lock_sha256 = _digest(lock)
    wrong_root = tmp_path / "staged-clone"

    with pytest.raises(
        verifier.EnvironmentVerificationError,
        match="final release path",
    ):
        verifier.verify_observation(
            lock=lock,
            lock_sha256=lock_sha256,
            marker=_v3_marker(wrong_root, lock, lock_sha256),
            marker_sha256="9" * 64,
            marker_path=wrong_root / verifier.CLONE_MARKER_NAME,
            observation=_v3_observation(wrong_root, lock),
        )


def test_bundled_ffmpeg_resolver_must_return_exact_regular_executable_in_env(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    ffmpeg = root / publisher.BUNDLED_FFMPEG_RELATIVE_PATH
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.write_bytes(b"ELF fixture")
    ffmpeg.chmod(0o555)

    class ImageioFfmpeg:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return str(ffmpeg)

    identity = verifier.collect_bundled_ffmpeg_identity(
        ImageioFfmpeg, environment_root=root
    )
    assert identity == {
        "absolute_path": str(ffmpeg),
        "distribution": "imageio-ffmpeg",
        "mode": "0555",
        "path": publisher.BUNDLED_FFMPEG_RELATIVE_PATH,
        "resolver": "imageio_ffmpeg.get_ffmpeg_exe",
        "sha256": hashlib.sha256(b"ELF fixture").hexdigest(),
        "size": len(b"ELF fixture"),
        "version": "0.6.0",
    }

    outside = tmp_path / "outside-ffmpeg"
    outside.write_bytes(b"outside")
    ImageioFfmpeg.get_ffmpeg_exe = staticmethod(lambda: str(outside))
    with pytest.raises(verifier.EnvironmentVerificationError, match="escapes"):
        verifier.collect_bundled_ffmpeg_identity(
            ImageioFfmpeg, environment_root=root
        )
