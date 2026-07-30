from __future__ import annotations

import hashlib
import io
import json
import marshal
import os
from pathlib import Path
import stat
import struct
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import artifact_contract as contract  # noqa: E402
import perception_environment_manifest as environment_manifest  # noqa: E402
import prepare_perception_environment_candidate as preparation  # noqa: E402
import publish_perception_runtime as publisher  # noqa: E402
import verify_perception_environment as verifier  # noqa: E402
from test_perception_environment_lock_v3 import _v3_lock  # noqa: E402


def _digest(value: object) -> str:
    return hashlib.sha256(contract.canonical_json_bytes(value)).hexdigest()


def _source() -> dict:
    environment_id = "linbox-perception-" + "1" * 20
    return {
        "environment_id": environment_id,
        "lock_sha256": "2" * 64,
        "marker_sha256": "3" * 64,
        "path": "%s/%s"
        % (publisher.LINBOX_PERCEPTION_RELEASES_ROOT, environment_id),
        "snapshot_sha256": "4" * 64,
    }


def _legacy_source_marker(source: dict, environment_root: str) -> dict:
    return {
        "complete": True,
        "environment_id": source["environment_id"],
        "environment_root": environment_root,
        "lock_sha256": source["lock_sha256"],
        "observed_distribution_inventory": {
            "algorithm": verifier.LEGACY_DISTRIBUTION_INVENTORY_ALGORITHM,
            "count": 1,
            "sha256": "8" * 64,
        },
        "python_executable_relative": "bin/python",
        "schema": verifier.CLONE_MARKER_SCHEMA,
        "source_environment_fingerprint_sha256": "9" * 64,
    }


def _wheelhouse(purpose: str, rows: list[dict] | None = None) -> dict:
    value = {
        "purpose": purpose,
        "schema": preparation.WHEELHOUSE_MANIFEST_SCHEMA,
        "wheels": [] if rows is None else rows,
    }
    value["sha256"] = preparation.wheelhouse_manifest_sha256(value)
    return value


def _recipe() -> dict:
    return preparation.build_recipe(
        source_environment=_source(),
        gate_wheelhouse_manifest=_wheelhouse("gate-additions"),
        opencv_wheelhouse_manifest=_wheelhouse("opencv-repair"),
        repair_wheelhouse_manifest=preparation.exact_repair_wheelhouse_manifest(),
    )


def test_recipe_is_selected_before_clone_and_binds_every_input_manifest() -> None:
    recipe = _recipe()
    publisher.validate_final_path_build_recipe(recipe)
    assert recipe["operations"] == list(publisher.FINAL_PATH_BUILD_OPERATIONS)
    assert preparation.environment_id_for_recipe(recipe) == (
        "linbox-perception-" + recipe["sha256"][:20]
    )
    assert recipe["wheelhouse_manifests"] == [
        {
            "purpose": "gate-additions",
            "sha256": _wheelhouse("gate-additions")["sha256"],
        },
        {
            "purpose": "opencv-repair",
            "sha256": _wheelhouse("opencv-repair")["sha256"],
        },
        {
            "purpose": "runtime-repairs",
            "sha256": preparation.exact_repair_wheelhouse_manifest()["sha256"],
        },
    ]

    changed = _wheelhouse(
        "gate-additions",
        [{"filename": "gate.whl", "sha256": "4" * 64, "size": 1}],
    )
    changed_recipe = preparation.build_recipe(
        source_environment=_source(),
        gate_wheelhouse_manifest=changed,
        opencv_wheelhouse_manifest=_wheelhouse("opencv-repair"),
        repair_wheelhouse_manifest=preparation.exact_repair_wheelhouse_manifest(),
    )
    assert changed_recipe["sha256"] != recipe["sha256"]
    assert preparation.environment_id_for_recipe(changed_recipe) != (
        preparation.environment_id_for_recipe(recipe)
    )

    changed_source = {**_source(), "snapshot_sha256": "5" * 64}
    changed_source_recipe = preparation.build_recipe(
        source_environment=changed_source,
        gate_wheelhouse_manifest=_wheelhouse("gate-additions"),
        opencv_wheelhouse_manifest=_wheelhouse("opencv-repair"),
        repair_wheelhouse_manifest=preparation.exact_repair_wheelhouse_manifest(),
    )
    assert changed_source_recipe["sha256"] != recipe["sha256"]


def test_recipe_rejects_source_without_complete_live_snapshot_binding() -> None:
    source = _source()
    source.pop("snapshot_sha256")
    with pytest.raises(
        preparation.PreparationPolicyError,
        match="source environment identity is incomplete",
    ):
        preparation.build_recipe(
            source_environment=source,
            gate_wheelhouse_manifest=_wheelhouse("gate-additions"),
            opencv_wheelhouse_manifest=_wheelhouse("opencv-repair"),
            repair_wheelhouse_manifest=preparation.exact_repair_wheelhouse_manifest(),
        )


def test_operational_input_paths_must_be_normalized_absolute(tmp_path: Path) -> None:
    assert preparation.normalized_absolute_path(tmp_path, "input") == tmp_path
    with pytest.raises(preparation.PreparationPolicyError, match="must be absolute"):
        preparation.normalized_absolute_path(Path("relative/path"), "input")
    with pytest.raises(preparation.PreparationPolicyError, match="must be normalized"):
        preparation.normalized_absolute_path(tmp_path / "child/..", "input")


def test_source_fingerprint_is_recomputed_from_verified_observation_not_marker() -> None:
    observation = {
        "cuda": {"architectures": ["sm_120"], "build": "12.8"},
        "distribution_inventory": {
            "algorithm": publisher.DISTRIBUTION_INVENTORY_ALGORITHM,
            "count": 1,
            "sha256": "4" * 64,
        },
        "modules": {
            "torch": {
                "algorithm": publisher.MODULE_TREE_ALGORITHM,
                "count": 1,
                "sha256": "5" * 64,
            }
        },
        "packages": [
            {"name": "torch", "record_sha256": "6" * 64, "version": "2.11.0"}
        ],
        "python": {
            "abi": "cp311",
            "executable_sha256": "7" * 64,
            "implementation": "CPython",
            "platform": "linux-x86_64",
            "version": "3.11.15",
        },
    }
    report = {
        **observation,
        "distribution_inventory": {
            "lock": {"algorithm": "legacy", "count": 1, "sha256": "8" * 64},
            "observed": observation["distribution_inventory"],
        },
        "status": "VERIFIED_WITH_SCHEMA_GAPS",
    }
    assert preparation.source_fingerprint_from_verification_report(report) == (
        publisher.source_environment_fingerprint_sha256(observation)
    )


def test_exact_official_runtime_repair_wheels_are_pinned() -> None:
    manifest = preparation.exact_repair_wheelhouse_manifest()
    assert manifest["algorithm"] == publisher.REPAIR_WHEELHOUSE_ALGORITHM
    assert manifest["wheels"] == publisher.EXACT_REPAIR_WHEELS
    assert [(row["filename"], row["sha256"], row["size"]) for row in manifest["wheels"]] == [
        (
            "imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl",
            "c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282",
            29498237,
        ),
        (
            "pip-26.1.2-py3-none-any.whl",
            "382ff9f685ee3bc25864f820aa50505825f10f5458ffff07e30a6d96e5715cab",
            1813144,
        ),
    ]
    assert all(row["url"].startswith("https://files.pythonhosted.org/") for row in manifest["wheels"])
    assert manifest["sha256"] == preparation.wheelhouse_manifest_sha256(manifest)


def test_official_wheel_download_is_size_bounded_and_cleans_partial(
    tmp_path: Path,
) -> None:
    row = {
        "filename": "demo.whl",
        "name": "demo",
        "sha256": hashlib.sha256(b"good").hexdigest(),
        "size": 4,
        "url": "https://files.pythonhosted.org/packages/demo.whl",
        "version": "1.0",
    }
    with pytest.raises(
        preparation.PreparationVerificationError, match="exceeded pinned size"
    ):
        preparation.download_verified_wheel(
            row,
            tmp_path,
            allow_download=True,
            opener=lambda *_args, **_kwargs: io.BytesIO(b"oversize"),
        )
    assert not list(tmp_path.iterdir())


def test_wheelhouse_inventory_binds_exact_bytes_and_rejects_extra_entries(
    tmp_path: Path,
) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "demo-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel payload")
    value = preparation.inventory_wheelhouse(
        wheelhouse, purpose="gate-additions"
    )
    assert value["wheels"] == [
        {
            "filename": wheel.name,
            "sha256": hashlib.sha256(b"wheel payload").hexdigest(),
            "size": len(b"wheel payload"),
        }
    ]
    assert value["sha256"] == preparation.wheelhouse_manifest_sha256(value)

    (wheelhouse / "notes.txt").write_text("unbound", encoding="utf-8")
    with pytest.raises(preparation.PreparationPolicyError, match="non-wheel"):
        preparation.inventory_wheelhouse(
            wheelhouse, purpose="gate-additions"
        )


def _write_managed_wheelhouse_metadata(
    wheelhouse: Path,
    wheel_rows: list[dict],
    *,
    include_payload_schema: bool,
    payload_sha256: str | None = None,
    manifest_schema: str | None = None,
) -> tuple[dict, dict]:
    payload = {
        "algorithm": preparation.MANAGED_WHEELHOUSE_PAYLOAD_ALGORITHM,
        "wheels": [
            {
                **row,
                "name": "demo",
                "requires_dist": [],
                "requires_python": ">=3.11",
                "version": "1.0",
            }
            for row in wheel_rows
        ],
    }
    if include_payload_schema:
        payload["schema"] = preparation.MANAGED_WHEELHOUSE_PAYLOAD_SCHEMA
    payload_digest = payload_sha256 or hashlib.sha256(
        contract.canonical_json_bytes(payload).rstrip(b"\n")
    ).hexdigest()
    release_id = "test-wheelhouse-" + payload_digest[:20]
    managed = {
        "payload": payload,
        "payload_sha256": payload_digest,
        "release_id": release_id,
        "schema": manifest_schema or preparation.MANAGED_WHEELHOUSE_MANIFEST_SCHEMA,
    }
    managed_raw = contract.canonical_json_bytes(managed)
    if not include_payload_schema:
        managed_raw = managed_raw.rstrip(b"\n")
    (wheelhouse / preparation.MANAGED_WHEELHOUSE_MANIFEST_NAME).write_bytes(
        managed_raw
    )
    managed_sha256 = hashlib.sha256(managed_raw).hexdigest()
    complete = {
        "manifest_sha256": managed_sha256,
        "payload_sha256": payload_digest,
        "release_id": release_id,
        "schema": preparation.MANAGED_WHEELHOUSE_COMPLETE_SCHEMA,
        "verified_wheels": len(wheel_rows),
    }
    complete_raw = contract.canonical_json_bytes(complete)
    if not include_payload_schema:
        complete_raw = complete_raw.rstrip(b"\n")
    (wheelhouse / preparation.MANAGED_WHEELHOUSE_COMPLETE_NAME).write_bytes(
        complete_raw
    )
    return managed, complete


@pytest.mark.parametrize("include_payload_schema", [False, True])
def test_wheelhouse_inventory_accepts_and_verifies_managed_release_metadata(
    tmp_path: Path, include_payload_schema: bool,
) -> None:
    payload_sha256 = "9" * 64 if include_payload_schema else None
    wheelhouse = tmp_path / (
        "test-wheelhouse-" + ((payload_sha256 or "placeholder")[:20])
    )
    wheelhouse.mkdir()
    wheel = wheelhouse / "demo-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel payload")
    wheel_rows = [
        {
            "filename": wheel.name,
            "sha256": hashlib.sha256(b"wheel payload").hexdigest(),
            "size": len(b"wheel payload"),
        }
    ]
    managed, _complete = _write_managed_wheelhouse_metadata(
        wheelhouse,
        wheel_rows,
        include_payload_schema=include_payload_schema,
        payload_sha256=payload_sha256,
    )
    expected_root = tmp_path / managed["release_id"]
    wheelhouse.rename(expected_root)

    observed = preparation.inventory_wheelhouse(
        expected_root, purpose="gate-additions"
    )

    assert observed["wheels"] == wheel_rows
    assert observed["sha256"] == preparation.wheelhouse_manifest_sha256(observed)


def test_wheelhouse_inventory_accepts_exact_legacy_gate_manifest_schema(
    tmp_path: Path,
) -> None:
    wheelhouse = tmp_path / "temporary"
    wheelhouse.mkdir()
    wheel = wheelhouse / "demo-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel payload")
    wheel_rows = [{
        "filename": wheel.name,
        "sha256": hashlib.sha256(b"wheel payload").hexdigest(),
        "size": len(b"wheel payload"),
    }]
    managed, _complete = _write_managed_wheelhouse_metadata(
        wheelhouse,
        wheel_rows,
        include_payload_schema=False,
        manifest_schema="wam_python_wheelhouse_v1",
    )
    expected_root = tmp_path / managed["release_id"]
    wheelhouse.rename(expected_root)

    observed = preparation.inventory_wheelhouse(
        expected_root, purpose="gate-additions"
    )

    assert observed["wheels"] == wheel_rows


def test_wheelhouse_inventory_rejects_legacy_schema_with_new_payload_shape(
    tmp_path: Path,
) -> None:
    wheelhouse = tmp_path / "temporary"
    wheelhouse.mkdir()
    wheel = wheelhouse / "demo-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel payload")
    wheel_rows = [{
        "filename": wheel.name,
        "sha256": hashlib.sha256(b"wheel payload").hexdigest(),
        "size": len(b"wheel payload"),
    }]
    managed, _complete = _write_managed_wheelhouse_metadata(
        wheelhouse,
        wheel_rows,
        include_payload_schema=True,
        payload_sha256="9" * 64,
        manifest_schema="wam_python_wheelhouse_v1",
    )
    expected_root = tmp_path / managed["release_id"]
    wheelhouse.rename(expected_root)

    with pytest.raises(
        preparation.PreparationVerificationError,
        match="manifest identity",
    ):
        preparation.inventory_wheelhouse(
            expected_root, purpose="gate-additions"
        )


def test_wheelhouse_inventory_rejects_managed_completion_or_payload_drift(
    tmp_path: Path,
) -> None:
    wheelhouse = tmp_path / "temporary"
    wheelhouse.mkdir()
    wheel = wheelhouse / "demo-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel payload")
    wheel_rows = [
        {
            "filename": wheel.name,
            "sha256": hashlib.sha256(b"wheel payload").hexdigest(),
            "size": len(b"wheel payload"),
        }
    ]
    managed, complete = _write_managed_wheelhouse_metadata(
        wheelhouse, wheel_rows, include_payload_schema=False
    )
    expected_root = tmp_path / managed["release_id"]
    wheelhouse.rename(expected_root)

    complete["verified_wheels"] = 2
    (expected_root / preparation.MANAGED_WHEELHOUSE_COMPLETE_NAME).write_bytes(
        contract.canonical_json_bytes(complete)
    )
    with pytest.raises(
        preparation.PreparationVerificationError,
        match="managed wheelhouse completion",
    ):
        preparation.inventory_wheelhouse(expected_root, purpose="gate-additions")

    complete["verified_wheels"] = 1
    (expected_root / preparation.MANAGED_WHEELHOUSE_COMPLETE_NAME).write_bytes(
        contract.canonical_json_bytes(complete)
    )
    managed["payload"]["wheels"][0]["size"] += 1
    (expected_root / preparation.MANAGED_WHEELHOUSE_MANIFEST_NAME).write_bytes(
        contract.canonical_json_bytes(managed)
    )
    with pytest.raises(
        preparation.PreparationVerificationError,
        match="managed wheelhouse completion|managed wheelhouse payload",
    ):
        preparation.inventory_wheelhouse(expected_root, purpose="gate-additions")


def test_reservation_is_no_replace_external_journal_and_resumable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "perception_envs"
    (root / ".staging/build_journal").mkdir(parents=True)
    (root / "releases").mkdir()
    plan = preparation.candidate_plan(root=root, recipe=_recipe())

    first = preparation.reserve_candidate(plan)
    assert plan.candidate.is_dir() and not plan.candidate.is_symlink()
    assert plan.journal_path.is_file()
    assert not (plan.candidate / verifier.CLONE_MARKER_NAME).exists()
    assert first["status"] == "INCOMPLETE"
    assert first["candidate"] == str(plan.candidate)
    assert first["recipe"] == plan.recipe
    assert plan.journal_path.parent.parent == root / ".staging/build_journal"

    inode = plan.candidate.stat().st_ino
    second = preparation.reserve_candidate(plan)
    assert second == first
    assert plan.candidate.stat().st_ino == inode

    plan.journal_path.unlink()
    with pytest.raises(
        preparation.PreparationConflictError, match="unowned incomplete"
    ):
        preparation.reserve_candidate(plan)


def test_first_reservation_safely_creates_external_journal_root(tmp_path: Path) -> None:
    root = tmp_path / "perception_envs"
    (root / ".staging").mkdir(parents=True)
    (root / "releases").mkdir()
    plan = preparation.candidate_plan(root=root, recipe=_recipe())
    preparation.reserve_candidate(plan)
    journal_root = root / ".staging/build_journal"
    assert journal_root.is_dir() and not journal_root.is_symlink()
    assert stat.S_IMODE(journal_root.stat().st_mode) == 0o700


def test_reservation_refuses_a_marker_without_complete_matching_journal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "perception_envs"
    (root / ".staging/build_journal").mkdir(parents=True)
    (root / "releases").mkdir()
    plan = preparation.candidate_plan(root=root, recipe=_recipe())
    preparation.reserve_candidate(plan)
    (plan.candidate / verifier.CLONE_MARKER_NAME).write_text(
        "{}", encoding="utf-8"
    )

    with pytest.raises(
        preparation.PreparationConflictError, match="completion marker"
    ):
        preparation.reserve_candidate(plan)


def test_reservation_allows_only_an_exact_inherited_marker_after_clone(
    tmp_path: Path,
) -> None:
    root = tmp_path / "perception_envs"
    (root / ".staging/build_journal").mkdir(parents=True)
    (root / "releases").mkdir()
    source = _source()
    source_marker = _legacy_source_marker(source, source["path"])
    source["marker_sha256"] = _digest(source_marker)
    recipe = preparation.build_recipe(
        source_environment=source,
        gate_wheelhouse_manifest=_wheelhouse("gate-additions"),
        opencv_wheelhouse_manifest=_wheelhouse("opencv-repair"),
        repair_wheelhouse_manifest=preparation.exact_repair_wheelhouse_manifest(),
    )
    plan = preparation.candidate_plan(root=root, recipe=recipe)
    preparation.reserve_candidate(plan)
    journal = preparation._load_journal(plan)
    journal["phases"]["clone"] = {"status": "COMPLETE"}
    preparation._write_journal(plan, journal)
    inherited = {
        **source_marker,
        "environment_root": str(plan.candidate),
    }
    contract._atomic_write(
        plan.candidate / verifier.CLONE_MARKER_NAME,
        contract.canonical_json_bytes(inherited),
    )

    resumed = preparation.reserve_candidate(plan)

    assert resumed["status"] == "INCOMPLETE"
    assert resumed["phases"]["clone"]["status"] == "COMPLETE"


def test_sanitize_inherited_metadata_removes_only_source_bound_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "perception_envs"
    (root / ".staging/build_journal").mkdir(parents=True)
    (root / "releases").mkdir()
    source = _source()
    source_marker = _legacy_source_marker(source, source["path"])
    source["marker_sha256"] = _digest(source_marker)
    recipe = preparation.build_recipe(
        source_environment=source,
        gate_wheelhouse_manifest=_wheelhouse("gate-additions"),
        opencv_wheelhouse_manifest=_wheelhouse("opencv-repair"),
        repair_wheelhouse_manifest=preparation.exact_repair_wheelhouse_manifest(),
    )
    plan = preparation.candidate_plan(root=root, recipe=recipe)
    preparation.reserve_candidate(plan)
    contract._atomic_write(
        plan.candidate / verifier.CLONE_MARKER_NAME,
        contract.canonical_json_bytes({
            **source_marker,
            "environment_root": str(plan.candidate),
        }),
    )
    verification_name = "PERCEPTION_ENVIRONMENT_VERIFICATION.json"
    verification_path = plan.candidate / verification_name
    verification_path.write_bytes(b"legacy verification evidence")
    sanitize = getattr(
        preparation, "sanitize_inherited_environment_metadata", None,
    )
    assert sanitize is not None

    evidence = sanitize(plan)

    assert [row["path"] for row in evidence["removed"]] == [
        verifier.CLONE_MARKER_NAME,
        verification_name,
    ]
    assert not (plan.candidate / verifier.CLONE_MARKER_NAME).exists()
    assert not verification_path.exists()


def _conda_rewritten_strong_lock_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[preparation.CandidatePlan, bytes]:
    root = tmp_path / "perception_envs"
    (root / ".staging/build_journal").mkdir(parents=True)
    (root / "releases").mkdir()
    source = _source()
    source_lock = {
        "environment_id": source["environment_id"],
        "relocation": {"new_prefix": source["path"]},
    }
    source_raw = contract.canonical_json_bytes(source_lock)
    source["lock_sha256"] = hashlib.sha256(source_raw).hexdigest()
    recipe = preparation.build_recipe(
        source_environment=source,
        gate_wheelhouse_manifest=_wheelhouse("gate-additions"),
        opencv_wheelhouse_manifest=_wheelhouse("opencv-repair"),
        repair_wheelhouse_manifest=preparation.exact_repair_wheelhouse_manifest(),
    )
    plan = preparation.candidate_plan(root=root, recipe=recipe)
    preparation.reserve_candidate(plan)
    source_lock_path = Path(source["path"]) / preparation.STRONG_LOCK_NAME
    original_read = preparation._read_regular_bytes

    def read_regular_bytes(path):
        if Path(path) == source_lock_path:
            return source_raw, SimpleNamespace(st_size=len(source_raw))
        return original_read(path)

    monkeypatch.setattr(preparation, "_read_regular_bytes", read_regular_bytes)
    monkeypatch.setattr(
        verifier,
        "load_environment_lock",
        lambda path: (
            (source_lock, source["lock_sha256"])
            if Path(path) == source_lock_path
            else (_ for _ in ()).throw(AssertionError("unexpected lock path"))
        ),
    )
    rewritten = source_raw.replace(
        source["path"].encode("utf-8"), str(plan.candidate).encode("utf-8")
    )
    return plan, rewritten


def test_sanitize_accepts_only_exact_conda_path_rewrite_of_source_strong_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, rewritten = _conda_rewritten_strong_lock_plan(tmp_path, monkeypatch)
    lock_path = plan.candidate / preparation.STRONG_LOCK_NAME
    lock_path.write_bytes(rewritten)

    evidence = preparation.sanitize_inherited_environment_metadata(plan)

    assert evidence["removed"] == [{
        "path": preparation.STRONG_LOCK_NAME,
        "sha256": hashlib.sha256(rewritten).hexdigest(),
        "size": len(rewritten),
    }]
    assert not lock_path.exists()


def test_sanitize_rejects_other_changes_to_conda_rewritten_strong_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, rewritten = _conda_rewritten_strong_lock_plan(tmp_path, monkeypatch)
    value = json.loads(rewritten.decode("utf-8"))
    value["tampered"] = True
    (plan.candidate / preparation.STRONG_LOCK_NAME).write_bytes(
        contract.canonical_json_bytes(value)
    )

    with pytest.raises(
        preparation.PreparationVerificationError,
        match="not the exact Conda path rewrite",
    ):
        preparation.sanitize_inherited_environment_metadata(plan)


def test_completed_phase_verifies_evidence_instead_of_repeating_action(
    tmp_path: Path,
) -> None:
    root = tmp_path / "perception_envs"
    (root / ".staging/build_journal").mkdir(parents=True)
    (root / "releases").mkdir()
    plan = preparation.candidate_plan(root=root, recipe=_recipe())
    preparation.reserve_candidate(plan)
    calls = {"action": 0, "verify": 0}

    def action() -> dict:
        calls["action"] += 1
        (plan.candidate / "phase-proof").write_bytes(b"proof")
        return {"sha256": hashlib.sha256(b"proof").hexdigest()}

    def verify(evidence: dict) -> None:
        calls["verify"] += 1
        assert evidence == {
            "sha256": hashlib.sha256(
                (plan.candidate / "phase-proof").read_bytes()
            ).hexdigest()
        }

    first = preparation.run_resumable_phase(
        plan, "clone", action=action, verify=verify
    )
    second = preparation.run_resumable_phase(
        plan, "clone", action=action, verify=verify
    )
    assert first == second
    assert calls == {"action": 1, "verify": 2}

    (plan.candidate / "phase-proof").write_bytes(b"drift")
    with pytest.raises(AssertionError):
        preparation.run_resumable_phase(
            plan, "clone", action=action, verify=verify
        )
    assert calls["action"] == 1


@pytest.mark.parametrize("resume_verify_source", [False, True])
def test_execute_rechecks_live_source_on_resume_and_immediately_before_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    resume_verify_source: bool,
) -> None:
    expected_snapshot = {
        "schema": preparation.SOURCE_SNAPSHOT_SCHEMA,
        "source_fingerprint_sha256": "a" * 64,
        "tree_generation": "expected",
    }
    drifted_snapshot = {**expected_snapshot, "tree_generation": "drifted"}
    source = {
        **_source(),
        "snapshot_sha256": preparation.source_snapshot_sha256(expected_snapshot),
    }
    gate_manifest = _wheelhouse("gate-additions")
    opencv_manifest = _wheelhouse("opencv-repair")
    recipe = preparation.build_recipe(
        source_environment=source,
        gate_wheelhouse_manifest=gate_manifest,
        opencv_wheelhouse_manifest=opencv_manifest,
        repair_wheelhouse_manifest=preparation.exact_repair_wheelhouse_manifest(),
    )
    plan = preparation.candidate_plan(root=tmp_path / "envs", recipe=recipe)
    state = {"drifted": resume_verify_source}

    monkeypatch.setattr(preparation, "reserve_candidate", lambda _plan: {})
    monkeypatch.setattr(
        preparation.activation, "_verify_tree_immutable", lambda _root: None,
    )
    monkeypatch.setattr(
        preparation,
        "collect_source_snapshot",
        lambda **_kwargs: (
            drifted_snapshot if state["drifted"] else expected_snapshot
        ),
    )
    monkeypatch.setattr(
        preparation,
        "_run_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("clone command ran after source drift")
        ),
    )

    def run_phase(_plan, name, *, action, verify):
        if name == "verify-source":
            evidence = {
                "fingerprint_sha256": expected_snapshot[
                    "source_fingerprint_sha256"
                ],
                "snapshot_sha256": preparation.source_snapshot_sha256(
                    expected_snapshot
                ),
            }
            if not resume_verify_source:
                evidence = action()
            verify(evidence)
            return evidence
        if name == "snapshot-gate-additions":
            return {"manifest": gate_manifest, "path": str(tmp_path / "gate")}
        if name == "snapshot-opencv-repair":
            return {
                "manifest": opencv_manifest,
                "path": str(tmp_path / "opencv"),
            }
        if name == "acquire-runtime-repairs":
            return {
                "manifest": preparation.exact_repair_wheelhouse_manifest(),
                "path": str(tmp_path / "repairs"),
            }
        if name == "clone":
            state["drifted"] = True
            return action()
        pytest.fail("unexpected phase before clone: %s" % name)

    monkeypatch.setattr(preparation, "run_resumable_phase", run_phase)

    with pytest.raises(
        preparation.PreparationVerificationError,
        match="source snapshot changed after recipe selection",
    ):
        preparation._execute_build(
            plan=plan,
            source_lock_path=tmp_path / "source.lock.json",
            source_lock={},
            source_marker={},
            source_snapshot=expected_snapshot,
            gate_source=tmp_path / "gate-source",
            gate_manifest=gate_manifest,
            opencv_source=tmp_path / "opencv-source",
            opencv_manifest=opencv_manifest,
            conda_executable=tmp_path / "conda",
            code_root=tmp_path / "code",
            v9_checkpoint=tmp_path / "v9.pth",
            posterior_gate_checkpoint=tmp_path / "posterior.pt",
            allow_repair_download=False,
        )


def test_completed_inactive_result_reports_the_bound_source_snapshot(
    tmp_path: Path,
) -> None:
    plan = preparation.candidate_plan(root=tmp_path, recipe=_recipe())
    build_result = getattr(preparation, "completed_inactive_result", None)
    assert build_result is not None

    result = build_result(
        plan=plan,
        final_evidence={"lock_sha256": "a" * 64},
        source_evidence={"snapshot_sha256": "b" * 64},
    )

    assert result["status"] == "COMPLETE_INACTIVE"
    assert result["source_snapshot_sha256"] == "b" * 64
    assert "source_verification_sha256" not in result


def test_internal_model_smoke_keeps_library_stdout_out_of_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd,
) -> None:
    import numpy as np

    class FakeYOLO:
        def __init__(self, _checkpoint) -> None:
            print("ultralytics-first-run-noise")

        def predict(self, *_args, **_kwargs) -> None:
            print("ultralytics-predict-noise")

    class FakeDetector:
        calls = 0

        def __init__(self, **_kwargs) -> None:
            pass

        def detect(self, *_args, **_kwargs):
            type(self).calls += 1
            return SimpleNamespace(
                back_present=type(self).calls == 1,
                xy_det=np.zeros((10, 2), dtype=np.float32),
            )

    class FakeIntrinsics:
        def __init__(self, **_kwargs) -> None:
            pass

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        get_arch_list=lambda: ["sm_120"],
        get_device_capability=lambda: (12, 0),
        is_available=lambda: True,
    )
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    ausc = ModuleType("ausc")
    ausc.__path__ = []
    perception = ModuleType("ausc.perception")
    perception.__path__ = []
    tracking = ModuleType("ausc.perception.tracking")
    tracking.__path__ = []
    detector = ModuleType("ausc.perception.tracking.detector")
    detector.NormalsDetector = FakeDetector
    geometry = ModuleType("ausc.perception.tracking.geometry")
    geometry.Intrinsics = FakeIntrinsics
    for name, module in (
        ("torch", torch),
        ("ultralytics", ultralytics),
        ("ausc", ausc),
        ("ausc.perception", perception),
        ("ausc.perception.tracking", tracking),
        ("ausc.perception.tracking.detector", detector),
        ("ausc.perception.tracking.geometry", geometry),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    v9 = tmp_path / "v9.pt"
    posterior = tmp_path / "posterior.pt"
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    monkeypatch.setattr(
        preparation,
        "_sha256_file",
        lambda path: (
            preparation.V9_CHECKPOINT_SHA256
            if Path(path) == v9
            else preparation.POSTERIOR_GATE_CHECKPOINT_SHA256,
            1,
        ),
    )

    result = preparation._internal_model_smoke(SimpleNamespace(
        code_root=tmp_path / "code",
        evidence_root=evidence_root,
        posterior_gate_checkpoint=posterior,
        v9_checkpoint=v9,
    ))
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert result == 0
    assert payload["status"] == "PASS"
    assert "ultralytics-first-run-noise" not in captured.out
    assert "ultralytics-first-run-noise" in captured.err


def _write_timestamp_pyc(root: Path, *, co_filename: str | None = None) -> tuple[Path, Path]:
    source = root / preparation.NUMPY_GENERATED_PYC_SOURCE
    pyc = root / environment_manifest.GENERATED_PYC_OMISSION
    source.parent.mkdir(parents=True)
    pyc.parent.mkdir(parents=True)
    source_bytes = b"VALUE = 7\n"
    source.write_bytes(source_bytes)
    code = compile(
        source_bytes,
        co_filename or preparation.NUMPY_GENERATED_PYC_SOURCE,
        "exec",
    )
    pyc.write_bytes(
        preparation.CPYTHON_311_MAGIC
        + struct.pack("<III", 0, 123456789, len(source_bytes))
        + marshal.dumps(code)
    )
    return source, pyc


def test_numpy_pyc_is_strictly_inspected_evidenced_then_removed(
    tmp_path: Path,
) -> None:
    source, pyc = _write_timestamp_pyc(tmp_path)
    result = preparation.inspect_and_remove_numpy_generated_pyc(tmp_path)
    assert not pyc.exists()
    assert result["inspection"]["magic_hex"] == preparation.CPYTHON_311_MAGIC.hex()
    assert result["inspection"]["flags"] == 0
    assert result["inspection"]["recorded_source_size"] == source.stat().st_size
    assert result["inspection"]["code_filename"].endswith(
        preparation.NUMPY_GENERATED_PYC_SOURCE
    )
    assert result["inspection"]["pyc_sha256"]
    assert result["inspection"]["header_sha256"]
    assert result["inspection"]["code_sha256"]
    assert result["inspection"]["source_sha256"]
    evidence = result["manifest_evidence"]
    assert evidence == environment_manifest.generated_pyc_omission_evidence(
        path=environment_manifest.GENERATED_PYC_OMISSION,
        source_path=preparation.NUMPY_GENERATED_PYC_SOURCE,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        inspection_sha256=_digest(result["inspection"]),
    )


def test_numpy_pyc_pipeline_persists_evidence_before_delete_and_resumes(
    tmp_path: Path,
) -> None:
    _source, pyc = _write_timestamp_pyc(tmp_path / "environment")
    evidence_path = tmp_path / "journal/PYC_OMISSION.json"
    evidence_path.parent.mkdir()

    first = preparation.persist_and_remove_numpy_generated_pyc(
        tmp_path / "environment", evidence_path=evidence_path
    )
    assert not pyc.exists()
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == first
    assert evidence_path.read_bytes() == contract.canonical_json_bytes(first)

    second = preparation.persist_and_remove_numpy_generated_pyc(
        tmp_path / "environment", evidence_path=evidence_path
    )
    assert second == first

    source = tmp_path / "environment" / preparation.NUMPY_GENERATED_PYC_SOURCE
    source.write_text("VALUE = 8\n", encoding="utf-8")
    with pytest.raises(
        preparation.PreparationVerificationError, match="source SHA-256 drift"
    ):
        preparation.persist_and_remove_numpy_generated_pyc(
            tmp_path / "environment", evidence_path=evidence_path
        )


def test_numpy_pyc_pipeline_inherits_only_explicit_verified_source_evidence(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_path, _source_pyc = _write_timestamp_pyc(source_root)
    inherited = preparation.inspect_numpy_generated_pyc(source_root)
    candidate_root = tmp_path / "candidate"
    candidate_source = candidate_root / preparation.NUMPY_GENERATED_PYC_SOURCE
    candidate_source.parent.mkdir(parents=True)
    candidate_source.write_bytes(source_path.read_bytes())
    evidence_path = tmp_path / "journal/PYC_OMISSION.json"
    evidence_path.parent.mkdir()

    result = preparation.persist_and_remove_numpy_generated_pyc(
        candidate_root,
        evidence_path=evidence_path,
        inherited_evidence=inherited,
    )

    assert result == inherited
    assert evidence_path.read_bytes() == contract.canonical_json_bytes(inherited)
    candidate_source.write_text("VALUE = 8\n", encoding="utf-8")
    evidence_path.unlink()
    with pytest.raises(
        preparation.PreparationVerificationError, match="source SHA-256 drift"
    ):
        preparation.persist_and_remove_numpy_generated_pyc(
            candidate_root,
            evidence_path=evidence_path,
            inherited_evidence=inherited,
        )


def test_source_pyc_omission_inheritance_is_bound_to_source_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "perception_envs"
    source = _source()
    source["path"] = str(root / "releases" / source["environment_id"])
    environment_id = "linbox-perception-" + "a" * 20
    plan = preparation.CandidatePlan(
        root=root,
        releases=root / "releases",
        journal_root=root / ".staging/build_journal" / environment_id,
        environment_id=environment_id,
        candidate=root / "releases" / environment_id,
        journal_path=(
            root / ".staging/build_journal" / environment_id
            / "BUILD_JOURNAL.json"
        ),
        recipe={"source_environment": source},
    )
    evidence_path = (
        root / ".staging/build_journal" / source["environment_id"]
        / "PYC_OMISSION_EVIDENCE.json"
    )
    evidence_path.parent.mkdir(parents=True)
    _write_timestamp_pyc(Path(source["path"]))
    inherited = preparation.persist_and_remove_numpy_generated_pyc(
        source["path"], evidence_path=evidence_path
    )
    source_lock = {
        "environment_id": source["environment_id"],
        "provisioning": {
            "generated_pyc_omission": inherited["manifest_evidence"],
        },
    }

    assert preparation.load_source_pyc_omission_evidence(
        plan, source_lock
    ) == inherited

    source_lock["provisioning"]["generated_pyc_omission"] = {
        **inherited["manifest_evidence"], "inspection_sha256": "f" * 64,
    }
    with pytest.raises(
        preparation.PreparationVerificationError,
        match="does not match the source strong lock",
    ):
        preparation.load_source_pyc_omission_evidence(plan, source_lock)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda raw: b"bad!" + raw[4:], "CPython 3.11 magic"),
        (
            lambda raw: raw[:4] + struct.pack("<I", 1) + raw[8:],
            "timestamp-based",
        ),
        (
            lambda raw: raw[:12] + struct.pack("<I", 999) + raw[16:],
            "source size",
        ),
    ],
)
def test_numpy_pyc_rejects_wrong_header_without_deleting(
    tmp_path: Path, mutate, message: str,
) -> None:
    _source_path, pyc = _write_timestamp_pyc(tmp_path)
    pyc.write_bytes(mutate(pyc.read_bytes()))
    with pytest.raises(preparation.PreparationVerificationError, match=message):
        preparation.inspect_and_remove_numpy_generated_pyc(tmp_path)
    assert pyc.exists()


def test_numpy_pyc_rejects_wrong_code_filename_without_deleting(
    tmp_path: Path,
) -> None:
    _source_path, pyc = _write_timestamp_pyc(
        tmp_path, co_filename="numpy/other.py"
    )
    with pytest.raises(
        preparation.PreparationVerificationError, match="co_filename"
    ):
        preparation.inspect_and_remove_numpy_generated_pyc(tmp_path)
    assert pyc.exists()


def test_offline_install_command_and_preflight_rehash_every_wheel(
    tmp_path: Path,
) -> None:
    python = tmp_path / "candidate/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    python.chmod(0o755)
    wheel = tmp_path / "wheels/demo.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"exact")
    manifest = _wheelhouse(
        "gate-additions",
        [
            {
                "filename": wheel.name,
                "sha256": hashlib.sha256(b"exact").hexdigest(),
                "size": len(b"exact"),
            }
        ],
    )
    command = preparation.offline_install_command(
        python=python,
        wheelhouse=wheel.parent,
        manifest=manifest,
    )
    assert command == [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--force-reinstall",
        "--no-compile",
        str(wheel),
    ]
    wheel.write_bytes(b"drift")
    with pytest.raises(
        preparation.PreparationVerificationError, match="wheel.*SHA-256"
    ):
        preparation.offline_install_command(
            python=python,
            wheelhouse=wheel.parent,
            manifest=manifest,
        )


def test_candidate_python_allows_only_internal_relative_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "candidate"
    (root / "bin").mkdir(parents=True)
    real = root / "bin/python3.11"
    real.write_bytes(b"python")
    real.chmod(0o755)
    python = root / "bin/python"
    python.symlink_to("python3.11")
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    wheel = wheelhouse / "demo.whl"
    wheel.write_bytes(b"wheel")
    manifest = _wheelhouse(
        "gate-additions",
        [{
            "filename": wheel.name,
            "sha256": hashlib.sha256(b"wheel").hexdigest(),
            "size": 5,
        }],
    )
    assert preparation.offline_install_command(
        python=python, wheelhouse=wheelhouse, manifest=manifest
    )[0] == str(python)

    python.unlink()
    python.symlink_to("/usr/bin/python3")
    with pytest.raises(
        preparation.PreparationVerificationError, match="escapes"
    ):
        preparation.offline_install_command(
            python=python, wheelhouse=wheelhouse, manifest=manifest
        )


def test_completed_or_crash_resumed_install_probes_versions_before_reinstall(
    tmp_path: Path,
) -> None:
    python = tmp_path / "candidate/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    python.chmod(0o755)
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    wheel = wheelhouse / "demo_pkg-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "demo_pkg-1.2.3.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: Demo_Pkg\nVersion: 1.2.3\n",
        )
    manifest = preparation.inventory_wheelhouse(
        wheelhouse, purpose="gate-additions"
    )
    expected = [{"name": "demo-pkg", "version": "1.2.3"}]
    assert preparation.wheel_distribution_profiles(
        wheelhouse=wheelhouse, manifest=manifest
    ) == expected
    runner_calls: list[list[str]] = []

    already = preparation.install_wheelhouse_resumable(
        python=python,
        wheelhouse=wheelhouse,
        manifest=manifest,
        profile_probe=lambda _python, profiles: list(profiles),
        runner=lambda command, **_kwargs: runner_calls.append(list(command)),
    )
    assert already == {
        "profiles": expected,
        "status": "ALREADY_SATISFIED",
    }
    assert runner_calls == []

    probes = iter(([], expected))

    def runner(command, **_kwargs):
        runner_calls.append(list(command))
        return {"returncode": 0, "stdout_sha256": "1" * 64}

    installed = preparation.install_wheelhouse_resumable(
        python=python,
        wheelhouse=wheelhouse,
        manifest=manifest,
        profile_probe=lambda _python, _profiles: next(probes),
        runner=runner,
    )
    assert installed["profiles"] == expected
    assert installed["status"] == "INSTALLED"
    assert installed["command"]["returncode"] == 0
    assert len(runner_calls) == 1

    forced_calls: list[list[str]] = []
    forced = preparation.install_wheelhouse_resumable(
        python=python,
        wheelhouse=wheelhouse,
        manifest=manifest,
        skip_if_satisfied=False,
        profile_probe=lambda _python, profiles: list(profiles),
        runner=lambda command, **_kwargs: (
            forced_calls.append(list(command))
            or {"returncode": 0}
        ),
    )
    assert forced["status"] == "INSTALLED"
    assert len(forced_calls) == 1


def test_crash_resumed_uninstall_probes_absence_before_mutating(tmp_path: Path) -> None:
    python = tmp_path / "bin/python"
    python.parent.mkdir()
    python.write_bytes(b"python")
    calls: list[list[str]] = []
    evidence = preparation.ensure_distribution_absent_resumable(
        python=python,
        distribution="opencv-python-headless",
        absence_probe=lambda _python, _name: True,
        runner=lambda command, **_kwargs: calls.append(list(command)),
    )
    assert evidence == {
        "distribution": "opencv-python-headless",
        "status": "ABSENT",
    }
    assert calls == []


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


def test_completion_marker_is_written_last_and_failure_remains_unpublishable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    (candidate / "bin/python").write_bytes(b"python")
    lock = _v3_lock()
    lock_sha256 = _digest(lock)
    marker = _v3_marker(candidate, lock, lock_sha256)
    writes: list[str] = []
    real_write = contract._atomic_write

    def observe(path: Path, data: bytes) -> None:
        writes.append(Path(path).name)
        real_write(path, data)

    monkeypatch.setattr(preparation.contract, "_atomic_write", observe)
    preparation.write_lock_and_completion_marker(
        candidate=candidate, lock=lock, marker=marker
    )
    assert writes == [
        preparation.STRONG_LOCK_NAME,
        verifier.CLONE_MARKER_NAME,
    ]
    assert (candidate / verifier.CLONE_MARKER_NAME).is_file()

    failed = tmp_path / "failed"
    (failed / "bin").mkdir(parents=True)
    (failed / "bin/python").write_bytes(b"python")
    failed_marker = _v3_marker(failed, lock, lock_sha256)

    def fail_on_marker(path: Path, data: bytes) -> None:
        if Path(path).name == verifier.CLONE_MARKER_NAME:
            raise OSError("simulated marker failure")
        real_write(path, data)

    monkeypatch.setattr(preparation.contract, "_atomic_write", fail_on_marker)
    with pytest.raises(OSError, match="simulated marker failure"):
        preparation.write_lock_and_completion_marker(
            candidate=failed, lock=lock, marker=failed_marker
        )
    assert (failed / preparation.STRONG_LOCK_NAME).is_file()
    assert not (failed / verifier.CLONE_MARKER_NAME).exists()

    persisted_lock = (failed / preparation.STRONG_LOCK_NAME).read_bytes()
    monkeypatch.setattr(preparation.contract, "_atomic_write", real_write)
    resumed = preparation.write_lock_and_completion_marker(
        candidate=failed, lock=lock, marker=failed_marker
    )
    assert (failed / preparation.STRONG_LOCK_NAME).read_bytes() == persisted_lock
    assert resumed == {
        "lock_sha256": lock_sha256,
        "marker_sha256": hashlib.sha256(
            contract.canonical_json_bytes(failed_marker)
        ).hexdigest(),
    }
    assert (failed / verifier.CLONE_MARKER_NAME).read_bytes() == (
        contract.canonical_json_bytes(failed_marker)
    )


def test_command_runner_passes_explicit_working_directory_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(preparation.subprocess, "run", run)
    preparation._run_command(["ffmpeg", "-version"], cwd=tmp_path)
    assert observed["command"] == ["ffmpeg", "-version"]
    assert observed["cwd"] == str(tmp_path)
    assert "shell" not in observed or observed["shell"] is False
