from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import stat
import threading
from types import SimpleNamespace
import sys

import pytest


DEPLOYMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOYMENT_DIR))

import deploy_remote  # noqa: E402
import artifact_contract as contract  # noqa: E402
import remote_agent  # noqa: E402
from test_artifact_contract import (  # noqa: E402
    _training_session,
    _trusted_finalization_record,
)


def _completed_release(tmp_path: Path, body: bytes = b"payload\n") -> Path:
    source = tmp_path / "source"
    payload = source / "auscultation" / "ausc" / "worker.py"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(body)
    manifest = contract.build_manifest(
        source,
        ["auscultation/ausc/worker.py"],
        artifact_kind="code",
        source_role="mac",
    )
    release = tmp_path / "release"
    contract.write_release(source, manifest, release, target_role="transfer")
    return release


def _legacy_sealed_session(
    sessions_root: Path, session_id: str = "session-001",
) -> Path:
    session = sessions_root / session_id
    artifacts = {
        "audio.wav": b"audio", "rgb.mp4": b"video", "robot.csv": b"robot",
        "force.csv": b"force", "rgb_times.csv": b"times", "obs.csv": b"obs",
        "targets.jsonl": b"target", "depth/depth_times.csv": b"depth-times",
        "depth/depth_000001.npy": b"depth-frame",
    }
    inventory = {}
    for relative, body in artifacts.items():
        path = session / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        inventory[relative] = {"sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}
    manifest = {
        "artifact_hashes": {
            "algorithm": "sha256", "files": inventory,
            "schema_version": "wam_session_artifact_hashes_v1",
        },
        "promotion_contract_version": "wam_promotion_v1",
        "schema_version": "wam_session_v2",
        "status": "complete",
        "training_eligible": True,
    }
    manifest_bytes = contract.canonical_json_bytes(manifest)
    (session / "manifest.json").write_bytes(manifest_bytes)
    (session / "FINALIZATION.json").write_bytes(contract.canonical_json_bytes({
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "promotion_contract_version": "wam_promotion_v1",
        "schema_version": "wam_session_finalization_v1",
    }))
    return session


def _sealed_session(sessions_root: Path, session_id: str = "session-001") -> Path:
    return _training_session(sessions_root, session_id)


def _trusted_training_config(session: Path, root: Path) -> dict:
    trust_root = root / "training-trust"
    trust_root.mkdir(parents=True, exist_ok=True)
    registry = trust_root / "TRAINING_FINALIZATIONS.json"
    registry.write_bytes(contract.canonical_json_bytes({
        "records": [_trusted_finalization_record(session)],
        "schema_version": "wam_trusted_training_finalization_registry_v1",
    }))
    return {
        "trusted_finalization_registry": registry,
        "trusted_snapshot_root": root / "trusted-training-snapshots",
    }


def _converted_dataset(
    releases_root: Path,
    *,
    lerobot_version: str = "0.3.3",
    dataset_schema_version: str = "lerobot_v2",
    raw_release_id: str | None = None,
    provenance_schema: str = "wam_converted_dataset_provenance_v2",
    raw_identity: dict | None = None,
) -> Path:
    source = releases_root.parent / "dataset-source"
    data = source / "dataset" / "data.bin"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"dataset")
    dataset_manifest = {
        "files": [{
            "path": "dataset/data.bin",
            "sha256": hashlib.sha256(b"dataset").hexdigest(),
            "size": 7,
        }],
        "schema": "wam_dataset_payload_manifest_v1",
    }
    dataset_manifest_bytes = contract.canonical_json_bytes(dataset_manifest)
    (source / "DATASET_MANIFEST.json").write_bytes(dataset_manifest_bytes)
    raw_manifest_sha256 = (
        str(raw_identity["manifest_sha256"]) if raw_identity else "5" * 64
    )
    provenance = {
        "converter_code_release_id": "code-" + "1" * 20,
        "converter_environment_id": "lerobot-" + "2" * 20,
        "converter_lock_sha256": "3" * 64,
        "dataset_manifest_sha256": hashlib.sha256(dataset_manifest_bytes).hexdigest(),
        "dataset_schema_version": dataset_schema_version,
        "lerobot_version": lerobot_version,
        "raw_finalization_sha256": (
            str(raw_identity["finalization_sha256"]) if raw_identity else "4" * 64
        ),
        "raw_training_authenticity_sha256": (
            str(raw_identity["training_authenticity_sha256"])
            if raw_identity else "7" * 64
        ),
        "raw_manifest_sha256": raw_manifest_sha256,
        "raw_payload_sha256": (
            str(raw_identity["raw_payload_sha256"]) if raw_identity else "6" * 64
        ),
        "raw_release_id": raw_release_id or (
            "raw_session-" + raw_manifest_sha256[:20]
        ),
        "raw_session_id": (
            str(raw_identity["session_id"]) if raw_identity else "session-001"
        ),
        "promotion_contract_version": "wam_promotion_v2",
        "schema": provenance_schema,
    }
    if provenance_schema == "wam_converted_dataset_provenance_v1":
        provenance.pop("raw_training_authenticity_sha256")
        provenance.pop("raw_session_id")
        provenance.pop("promotion_contract_version")
    (source / "DATASET_PROVENANCE.json").write_bytes(contract.canonical_json_bytes(provenance))
    paths = ["dataset/data.bin", "DATASET_MANIFEST.json", "DATASET_PROVENANCE.json"]
    manifest = contract.build_manifest(
        source, paths, artifact_kind="lerobot_dataset", source_role="linbox"
    )
    release = releases_root / manifest["release_id"]
    release.parent.mkdir(parents=True, exist_ok=True)
    contract.write_release(source, manifest, release, target_role="linbox")
    return release


PERCEPTION_ARTIFACT_POLICIES = {
    "perception_canary_evaluation": {
        "artifact_kind": "perception_canary_evaluation",
        "releases_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_canary_evaluation/releases"
        ),
        "remote_upload": False,
        "source_role": "h200",
        "staging_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_canary_evaluation/.staging"
        ),
    },
    "perception_geometry_source": {
        "artifact_kind": "perception_geometry_source",
        "releases_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_geometry_source/releases"
        ),
        "remote_upload": True,
        "source_role": "mac",
        "staging_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_geometry_source/.staging"
        ),
    },
    "perception_labels_holdout": {
        "artifact_kind": "perception_labels_holdout",
        "releases_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_labels_holdout/releases"
        ),
        "remote_upload": True,
        "source_role": "mac",
        "staging_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_labels_holdout/.staging"
        ),
    },
    "perception_labels_train": {
        "artifact_kind": "perception_labels_train",
        "releases_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_labels_train/releases"
        ),
        "remote_upload": True,
        "source_role": "mac",
        "staging_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_labels_train/.staging"
        ),
    },
    "perception_session2_train": {
        "artifact_kind": "perception_session2_train",
        "releases_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_session2_train/releases"
        ),
        "remote_upload": False,
        "source_role": "h200",
        "staging_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_session2_train/.staging"
        ),
    },
    "perception_session3_holdout": {
        "artifact_kind": "perception_session3_holdout",
        "releases_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_session3_holdout/releases"
        ),
        "remote_upload": False,
        "source_role": "h200",
        "staging_root": (
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_session3_holdout/.staging"
        ),
    },
    "perception_stage0": {
        "artifact_kind": "perception_stage0",
        "releases_root": (
            "/home/ssh-ziyue/wam_ausc/data/perception_stage0/releases"
        ),
        "remote_upload": False,
        "source_role": "h200",
        "staging_root": (
            "/home/ssh-ziyue/wam_ausc/data/perception_stage0/.staging"
        ),
    },
}


def _perception_release(
    releases_root: Path,
    *,
    artifact_kind: str,
    source_role: str,
    body: bytes = b"perception-payload\n",
) -> Path:
    source = releases_root.parent / (".%s-source" % artifact_kind)
    payload = source / "payload.bin"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(body)
    manifest = contract.build_manifest(
        source,
        ["payload.bin"],
        artifact_kind=artifact_kind,
        source_role=source_role,
    )
    release = releases_root / manifest["release_id"]
    release.parent.mkdir(parents=True, exist_ok=True)
    contract.write_release(source, manifest, release, target_role=source_role)
    return release


def _local_perception_agent_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    policies = {}
    for data_kind, expected in PERCEPTION_ARTIFACT_POLICIES.items():
        namespace = tmp_path / "h200-perception" / data_kind
        policies[data_kind] = {
            **expected,
            "releases_root": (namespace / "releases").as_posix(),
            "staging_root": (namespace / ".staging").as_posix(),
        }
    monkeypatch.setitem(
        remote_agent.HOST_POLICIES["h200"]["data"],
        "perception",
        policies,
    )
    monkeypatch.setitem(
        remote_agent.HOST_POLICIES["h200"]["data"],
        "perception_holdout_consumption_root",
        (tmp_path / "h200-perception" / "holdout-consumption").as_posix(),
    )
    return policies


def _staged_converted_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, Path, dict, dict]:
    root = tmp_path / "h200-data-lock"
    data_policy = {
        "training_trust": {},
        "raw_session": {
            "incoming_root": (root / "raw_sessions" / ".staging").as_posix(),
            "releases_root": (root / "raw_sessions" / "releases").as_posix(),
        },
        "converted_dataset": {
            "incoming_root": (root / "lerobot" / ".staging").as_posix(),
            "releases_root": (root / "lerobot" / "releases").as_posix(),
        },
    }
    monkeypatch.setitem(
        remote_agent.HOST_POLICIES, "h200",
        {"home": tmp_path.as_posix(), "canary": {}, "production": {}, "data": data_policy},
    )
    raw_staging_root = Path(data_policy["raw_session"]["incoming_root"])
    raw = _sealed_session(raw_staging_root, "transfer-raw-fixture")
    trusted = _trusted_training_config(raw, root)
    data_policy["training_trust"].update({
        "finalization_registry": trusted[
            "trusted_finalization_registry"
        ].as_posix(),
        "snapshot_root": trusted["trusted_snapshot_root"].as_posix(),
    })
    expected_finalization = _trusted_finalization_record(raw)["artifact_sha256"][
        "finalization"
    ]
    raw_identity = remote_agent.validate_raw_session(
        raw,
        allowed_sessions_root=raw_staging_root,
        expected_finalization_sha256=expected_finalization,
        **trusted,
    )
    remote_agent.archive_raw_session(
        raw,
        releases_root=data_policy["raw_session"]["releases_root"],
        expected_manifest_sha256=raw_identity["manifest_sha256"],
        expected_finalization_sha256=raw_identity["finalization_sha256"],
        **trusted,
    )
    source = _converted_dataset(
        tmp_path / "linbox-releases", raw_identity=raw_identity
    )
    release_id = source.name
    incoming = Path(data_policy["converted_dataset"]["incoming_root"]) / release_id
    reservation_id = "a" * 64
    remote_agent.prepare_data_incoming(
        "h200", data_operation="install_converted_dataset",
        incoming_path=incoming.as_posix(), reservation_id=reservation_id,
    )
    shutil.copytree(source, incoming, dirs_exist_ok=True)
    envelope = contract.verify_release(source)
    manifest_sha256 = hashlib.sha256(
        (source / contract.MANIFEST_NAME).read_bytes()
    ).hexdigest()
    promotion_request = {
        "agent_sha256": hashlib.sha256(
            (DEPLOYMENT_DIR / "remote_agent.py").read_bytes()
        ).hexdigest(),
        "contract_sha256": hashlib.sha256(
            (DEPLOYMENT_DIR / "artifact_contract.py").read_bytes()
        ).hexdigest(),
        "expected_manifest_sha256": manifest_sha256,
        "expected_payload_sha256": envelope["payload_sha256"],
        "expected_release_id": release_id,
        "incoming_path": incoming.as_posix(),
        "operation": "install_converted_dataset",
        "releases_root": data_policy["converted_dataset"]["releases_root"],
        "reservation_id": reservation_id,
        "schema": "wam_remote_data_request_v1",
        "target_role": "h200",
    }
    reconcile_request = dict(promotion_request)
    reconcile_request.pop("agent_sha256")
    reconcile_request.pop("contract_sha256")
    reconcile_request.pop("operation")
    reconcile_request.pop("schema")
    reconcile_request.pop("target_role")
    reconcile_request["data_operation"] = "install_converted_dataset"
    return data_policy, incoming, promotion_request, reconcile_request


def test_release_profile_is_strict_and_contains_canonical_three_host_paths() -> None:
    profile = deploy_remote.load_profile(DEPLOYMENT_DIR / "release_profile.json")

    assert profile["schema"] == "wam_release_profile_v1"
    assert profile["workspace_root"] == "/Users/williamqiu/Desktop/Harvard/WAM"
    assert profile["connections"] == [
        {"alias": "nebius-h200", "source_role": "mac", "target_role": "h200"},
        {"alias": "linbox", "source_role": "mac", "target_role": "linbox"},
        {"alias": "h200", "source_role": "linbox", "target_role": "h200"},
    ]
    assert profile["hosts"]["linbox"]["production"] == {
        "current_link": "/home/robot/Desktop/stethoscope/.wam-current",
        "incoming_root": "/home/robot/Desktop/stethoscope/.wam-incoming",
        "releases_root": "/home/robot/Desktop/stethoscope/.wam-releases",
    }
    assert profile["hosts"]["linbox"]["simulation"] == {
        "assets": {
            "artifact_kind": "simulation_asset",
            "current_link": "/home/robot/wam_data/simulation/assets/current",
            "releases_root": "/home/robot/wam_data/simulation/assets/releases",
            "staging_root": "/home/robot/wam_data/simulation/assets/.staging",
        },
        "code": {
            "artifact_kind": "simulation_code",
            "current_link": "/home/robot/wam_data/simulation/code/current",
            "releases_root": "/home/robot/wam_data/simulation/code/releases",
            "staging_root": "/home/robot/wam_data/simulation/code/.staging",
        },
        "results": {
            "artifact_kind": "simulation_result",
            "current_link": "/home/robot/wam_data/simulation/results/current",
            "releases_root": "/home/robot/wam_data/simulation/results/releases",
            "staging_root": "/home/robot/wam_data/simulation/results/.staging",
        },
    }
    assert profile["hosts"]["h200"]["production"] == {
        "current_link": "/home/ssh-ziyue/wam_ausc/current",
        "incoming_root": "/home/ssh-ziyue/wam_ausc/.incoming",
        "releases_root": "/home/ssh-ziyue/wam_ausc/.releases",
    }
    assert profile["hosts"]["linbox"]["python"] == "/home/robot/envs/ausc/bin/python"
    assert profile["hosts"]["h200"]["python"] == "/usr/bin/python3"
    assert profile["hosts"]["h200"]["helper_path"] == "/home/ssh-ziyue/.wam_remote_agent.py"
    assert profile["hosts"]["h200"]["contract_path"] == "/home/ssh-ziyue/.wam_artifact_contract.py"
    assert profile["hosts"]["h200"]["data"] == {
        "converted_dataset": {
            "incoming_root": "/home/ssh-ziyue/wam_ausc/data/lerobot/.staging",
            "releases_root": "/home/ssh-ziyue/wam_ausc/data/lerobot/releases",
        },
        "perception": PERCEPTION_ARTIFACT_POLICIES,
        "perception_holdout_consumption_root": (
            "/home/ssh-ziyue/wam_ausc/data/perception_holdout_consumption"
        ),
        "raw_session": {
            "incoming_root": "/home/ssh-ziyue/wam_ausc/data/raw_sessions/.staging",
            "releases_root": "/home/ssh-ziyue/wam_ausc/data/raw_sessions/releases",
        },
    }
    assert all(
        "current_link" not in policy
        for policy in profile["hosts"]["h200"]["data"]["perception"].values()
    )


def test_profile_rejects_unknown_fields(tmp_path: Path) -> None:
    raw = json.loads((DEPLOYMENT_DIR / "release_profile.json").read_text())
    raw["unexpected"] = True
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(deploy_remote.PolicyError, match="profile fields"):
        deploy_remote.load_profile(path)


def test_profile_rejects_substituted_simulation_root(tmp_path: Path) -> None:
    raw = json.loads((DEPLOYMENT_DIR / "release_profile.json").read_text())
    raw["hosts"]["linbox"]["simulation"]["results"]["releases_root"] = (
        "/home/robot/wam_data/simulation/results-substituted/releases"
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(deploy_remote.PolicyError, match="host policy"):
        deploy_remote.load_profile(path)


def test_ssh_plan_uses_fixed_remote_command_and_json_stdin(tmp_path: Path) -> None:
    release = _completed_release(tmp_path)
    plan = deploy_remote.build_ssh_plan(
        release,
        source_role="mac",
        target_role="h200",
        host_alias="nebius-h200",
        scope="canary",
        nonce="0" * 32,
    )

    assert plan.phases == (
        "verify_completed_source",
        "allocate_unique_incoming",
        "upload_content_hashed_agent",
        "rsync_no_inplace",
        "remote_exact_verify",
        "atomic_promotion",
        "capture_previous_pointer",
        "atomic_activation",
        "post_activation_verify_or_rollback",
    )
    assert plan.incoming_path.endswith("-" + "0" * 32)
    assert plan.remote_command == (
        "/usr/bin/python3 /home/ssh-ziyue/.wam_remote_agent.py"
    )
    assert [step.name for step in plan.steps] == [
        "upload_contract", "upload_agent", "bootstrap", "pre_inspect",
        "prepare_incoming", "stage_release", "install", "post_inspect",
    ]
    for upload in plan.steps[:2]:
        assert upload.argv[:3] == ("rsync", "-a", "-c")
        assert "--mkpath" not in upload.argv and "--protect-args" not in upload.argv
        assert upload.argv[3:5] == ("-e", "ssh -o BatchMode=yes -o ConnectTimeout=10")
        assert upload.argv[5] == "--"
    prepare = plan.steps[4]
    assert json.loads(prepare.stdin)["operation"] == "prepare_incoming"
    assert json.loads(prepare.stdin)["reservation_id"] == plan.reservation_id
    stage = plan.steps[5]
    assert stage.argv[0] == "rsync"
    assert stage.argv[:3] == ("rsync", "-a", "-c")
    assert stage.argv[3:5] == ("-e", "ssh -o BatchMode=yes -o ConnectTimeout=10")
    assert stage.argv[5] == "--"
    assert "--inplace" not in stage.argv
    assert "--partial" not in stage.argv
    assert stage.argv[-1] == "nebius-h200:" + plan.incoming_path + "/"
    bootstrap = plan.steps[2]
    assert bootstrap.argv[-1] == plan.remote_command
    assert json.loads(bootstrap.stdin)["operation"] == "bootstrap_canary"
    install = plan.steps[6]
    assert install.argv[-1] == plan.remote_command
    assert plan.incoming_path not in " ".join(install.argv)
    request = json.loads(install.stdin.decode("utf-8"))
    assert request["reservation_id"] == plan.reservation_id
    assert request["incoming_path"] == plan.incoming_path
    envelope = contract.verify_release(release)
    assert request["expected_release_id"] == envelope["release_id"]
    assert request["expected_payload_sha256"] == envelope["payload_sha256"]
    assert request["expected_manifest_sha256"] == hashlib.sha256(
        (release / contract.MANIFEST_NAME).read_bytes()
    ).hexdigest()
    assert request["agent_sha256"] == hashlib.sha256(
        (DEPLOYMENT_DIR / "remote_agent.py").read_bytes()
    ).hexdigest()
    assert request["contract_sha256"] == hashlib.sha256(
        (DEPLOYMENT_DIR / "artifact_contract.py").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    "bad",
    ["nebius h200", "nebius'h200", 'nebius"h200', "$(id)", "h200;id", "h200\nnext", "-oProxyCommand=x"],
)
def test_ssh_plan_rejects_injected_aliases(tmp_path: Path, bad: str) -> None:
    release = _completed_release(tmp_path)
    with pytest.raises(deploy_remote.PolicyError):
        deploy_remote.build_ssh_plan(
            release,
            source_role="mac",
            target_role="h200",
            host_alias=bad,
            scope="canary",
        )


def test_ssh_plan_rejects_wrong_alias_scope_and_nonce(tmp_path: Path) -> None:
    release = _completed_release(tmp_path)
    with pytest.raises(deploy_remote.PolicyError, match="connection"):
        deploy_remote.build_ssh_plan(
            release,
            source_role="mac",
            target_role="h200",
            host_alias="h200",
            scope="canary",
        )
    with pytest.raises(deploy_remote.PolicyError, match="nonce"):
        deploy_remote.build_ssh_plan(
            release,
            source_role="mac",
            target_role="h200",
            host_alias="nebius-h200",
            scope="canary",
            nonce="bad;nonce",
        )
    with pytest.raises(deploy_remote.PolicyError, match="canary-only"):
        deploy_remote.build_ssh_plan(
            release,
            source_role="mac",
            target_role="h200",
            host_alias="nebius-h200",
            scope="production",
        )


def test_remote_agent_enforces_role_and_exact_profile_paths(tmp_path: Path) -> None:
    plan = deploy_remote.build_ssh_plan(
        _completed_release(tmp_path),
        source_role="mac",
        target_role="h200",
        host_alias="nebius-h200",
        scope="canary",
        nonce="1" * 32,
    )
    request = json.loads(plan.steps[6].stdin)
    validated = remote_agent.validate_request(
        request, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
    )
    assert validated.incoming_path.as_posix() == plan.incoming_path
    assert validated.releases_root.as_posix().endswith("/.deploy-canary/releases")

    for field, bad in (
        ("current_link", request["current_link"] + ";id"),
        ("releases_root", request["releases_root"] + "/../escape"),
        ("incoming_path", "-" + request["incoming_path"]),
        ("target_role", "linbox"),
        ("expected_release_id", "../code"),
        ("expected_payload_sha256", "x" * 64),
        ("expected_manifest_sha256", "x" * 64),
    ):
        changed = dict(request)
        changed[field] = bad
        with pytest.raises(remote_agent.AgentPolicyError):
            remote_agent.validate_request(
                changed, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
            )


def test_remote_agent_rejects_unknown_request_fields_and_wrong_binary_hash(tmp_path: Path) -> None:
    plan = deploy_remote.build_ssh_plan(
        _completed_release(tmp_path),
        source_role="mac",
        target_role="linbox",
        host_alias="linbox",
        scope="canary",
        nonce="2" * 32,
    )
    request = json.loads(plan.steps[6].stdin)
    request["unexpected"] = True
    with pytest.raises(remote_agent.AgentPolicyError, match="fields"):
        remote_agent.validate_request(
            request, local_role="linbox", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )
    request.pop("unexpected")
    request["agent_sha256"] = "0" * 64
    with pytest.raises(remote_agent.AgentPolicyError, match="agent hash"):
        remote_agent.validate_request(
            request, local_role="linbox", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )


def test_execute_ssh_plan_runs_in_order_and_parses_canonical_responses(tmp_path: Path) -> None:
    plan = deploy_remote.build_ssh_plan(
        _completed_release(tmp_path),
        source_role="mac",
        target_role="h200",
        host_alias="nebius-h200",
        scope="canary",
        nonce="4" * 32,
    )
    calls = []
    rollback_request = None

    def runner(argv, **kwargs):
        nonlocal rollback_request
        calls.append((tuple(argv), kwargs.get("input", b"")))
        operation = json.loads(kwargs["input"])["operation"] if kwargs.get("input") else None
        if operation == "bootstrap_canary":
            stdout = contract.canonical_json_bytes({
                "schema": "wam_remote_bootstrap_response_v1", "target_role": "h200"
            })
        elif operation == "prepare_incoming":
            stdout = contract.canonical_json_bytes({
                "incoming_path": plan.incoming_path,
                "reservation_id": plan.reservation_id,
                "schema": "wam_remote_prepare_response_v1",
                "target_role": "h200",
            })
        elif operation == "inspect_canary":
            is_post = sum(1 for _argv, body in calls if body and json.loads(body).get("operation") == "inspect_canary") > 1
            stdout = contract.canonical_json_bytes({
                "current_target": (
                    "releases/" + plan.expected_release_id if is_post else None
                ),
                "release_ids": [plan.expected_release_id] if is_post else [],
                "schema": "wam_remote_canary_inspection_v1",
                "target_role": "h200",
            })
        elif operation == "install_release":
            stdout = contract.canonical_json_bytes({
                "current_link": "/home/ssh-ziyue/wam_ausc/.deploy-canary/current",
                "created_release": True,
                "final_release": "/home/ssh-ziyue/wam_ausc/.deploy-canary/releases/" + plan.expected_release_id,
                "release_id": contract.verify_release(tmp_path / "release")["release_id"],
                "schema": "wam_remote_install_response_v1",
                "target_role": "h200",
            })
        elif operation == "rollback_canary":
            rollback_request = json.loads(kwargs["input"])
            stdout = contract.canonical_json_bytes({
                "rolled_back_to": None,
                "schema": "wam_remote_canary_rollback_v1",
                "target_role": "h200",
            })
        elif operation == "cleanup_canary":
            stdout = contract.canonical_json_bytes({
                "removed_incoming": plan.incoming_path,
                "removed_release_id": plan.expected_release_id,
                "schema": "wam_remote_canary_cleanup_v1",
                "target_role": "h200",
            })
        else:
            stdout = b""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    response = deploy_remote.execute_ssh_plan(plan, runner=runner)
    assert len(calls) == 10
    assert response["schema"] == "wam_ssh_canary_evidence_v1"
    assert response["install"]["release_id"] == plan.expected_release_id
    assert rollback_request["expected_current_target"] == (
        "releases/" + plan.expected_release_id
    )
    operations = [json.loads(body)["operation"] for _argv, body in calls if body]
    assert operations == [
        "bootstrap_canary", "inspect_canary", "prepare_incoming", "install_release",
        "inspect_canary", "rollback_canary", "cleanup_canary",
    ]


def test_execute_ssh_plan_stops_at_first_failure(tmp_path: Path) -> None:
    plan = deploy_remote.build_ssh_plan(
        _completed_release(tmp_path), source_role="mac", target_role="linbox",
        host_alias="linbox", scope="canary", nonce="5" * 32,
    )
    calls = []
    operations = []

    def runner(argv, **kwargs):
        calls.append(tuple(argv))
        if kwargs.get("input"):
            operations.append(json.loads(kwargs["input"])["operation"])
        stdout = b""
        if len(calls) == 3:
            stdout = contract.canonical_json_bytes({
                "schema": "wam_remote_bootstrap_response_v1", "target_role": "linbox"
            })
        elif len(calls) == 4:
            stdout = contract.canonical_json_bytes({
                "current_target": None, "release_ids": [],
                "schema": "wam_remote_canary_inspection_v1", "target_role": "linbox"
            })
        elif len(calls) == 5:
            stdout = contract.canonical_json_bytes({
                "incoming_path": plan.incoming_path,
                "reservation_id": plan.reservation_id,
                "schema": "wam_remote_prepare_response_v1", "target_role": "linbox"
            })
        return SimpleNamespace(
            returncode=23 if len(calls) == 6 else 0, stdout=stdout, stderr=b"failed"
        )

    with pytest.raises(deploy_remote.DeploymentExecutionError, match="stage_release"):
        deploy_remote.execute_ssh_plan(plan, runner=runner)
    assert len(calls) >= 6
    assert "rollback_canary" not in operations


def test_failed_preinspection_never_reserves_incoming(tmp_path: Path) -> None:
    plan = deploy_remote.build_ssh_plan(
        _completed_release(tmp_path), source_role="mac", target_role="h200",
        host_alias="nebius-h200", scope="canary", nonce="a" * 32,
    )
    operations = []

    def runner(argv, **kwargs):
        operation = json.loads(kwargs["input"])["operation"] if kwargs.get("input") else None
        if operation:
            operations.append(operation)
        if operation == "bootstrap_canary":
            stdout = contract.canonical_json_bytes({
                "schema": "wam_remote_bootstrap_response_v1", "target_role": "h200"
            })
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        if operation == "inspect_canary":
            return SimpleNamespace(returncode=70, stdout=b"", stderr=b"inspection failed")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with pytest.raises(deploy_remote.DeploymentExecutionError, match="pre_inspect"):
        deploy_remote.execute_ssh_plan(plan, runner=runner)
    assert operations == ["bootstrap_canary", "inspect_canary"]
    assert "prepare_incoming" not in operations


def test_prepare_collision_executor_never_cleans_unowned_incoming(tmp_path: Path) -> None:
    plan = deploy_remote.build_ssh_plan(
        _completed_release(tmp_path), source_role="mac", target_role="h200",
        host_alias="nebius-h200", scope="canary", nonce="c" * 32,
    )
    operations = []

    def runner(argv, **kwargs):
        operation = json.loads(kwargs["input"])["operation"] if kwargs.get("input") else None
        if operation:
            operations.append(operation)
        if operation == "bootstrap_canary":
            stdout = contract.canonical_json_bytes({
                "schema": "wam_remote_bootstrap_response_v1", "target_role": "h200"
            })
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        if operation == "inspect_canary":
            stdout = contract.canonical_json_bytes({
                "current_target": None, "release_ids": [],
                "schema": "wam_remote_canary_inspection_v1", "target_role": "h200",
            })
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        if operation == "prepare_incoming":
            return SimpleNamespace(returncode=73, stdout=b"", stderr=b"already reserved")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with pytest.raises(deploy_remote.DeploymentExecutionError, match="prepare_incoming"):
        deploy_remote.execute_ssh_plan(plan, runner=runner)
    assert "cleanup_canary" not in operations
    assert "rollback_canary" not in operations


def test_remote_reservation_collision_preserves_first_owners_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = tmp_path / "canary"
    paths = {
        "current_link": (canary / "current").as_posix(),
        "incoming_root": (canary / "incoming").as_posix(),
        "releases_root": (canary / "releases").as_posix(),
    }
    monkeypatch.setitem(
        remote_agent.HOST_POLICIES, "h200",
        {"home": tmp_path.as_posix(), "canary": paths, "production": paths, "data": {}},
    )
    release_id = contract.verify_release(_completed_release(tmp_path / "artifact"))["release_id"]
    incoming = Path(paths["incoming_root"]) / (
        "incoming-%s-%s" % (release_id, "d" * 32)
    )
    remote_agent.bootstrap_canary("h200")
    first = remote_agent.prepare_incoming("h200", incoming.as_posix(), "1" * 64)
    owner_id = first["reservation_id"]
    (incoming / "owned").write_text("preserve")

    with pytest.raises(remote_agent.AgentPolicyError, match="already reserved"):
        remote_agent.prepare_incoming("h200", incoming.as_posix(), "2" * 64)
    with pytest.raises(remote_agent.AgentPolicyError, match="ownership"):
        remote_agent.cleanup_canary(
            "h200", incoming_path=incoming.as_posix(), reservation_id="0" * 64,
            created_release_id=None, pre_current_target=None, pre_release_ids=(),
        )
    assert (incoming / "owned").read_text() == "preserve"

    remote_agent.cleanup_canary(
        "h200", incoming_path=incoming.as_posix(), reservation_id=owner_id,
        created_release_id=None, pre_current_target=None, pre_release_ids=(),
    )
    assert not incoming.exists()


def test_canary_bootstrap_rejects_symlinked_canonical_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside-canary"
    outside.mkdir()
    substituted = tmp_path / "canary"
    substituted.symlink_to(outside, target_is_directory=True)
    paths = {
        "current_link": (substituted / "current").as_posix(),
        "incoming_root": (substituted / "incoming").as_posix(),
        "releases_root": (substituted / "releases").as_posix(),
    }
    monkeypatch.setitem(
        remote_agent.HOST_POLICIES, "h200",
        {"home": tmp_path.as_posix(), "canary": paths, "production": paths, "data": {}},
    )

    with pytest.raises(remote_agent.AgentPolicyError, match="symlink"):
        remote_agent.bootstrap_canary("h200")
    assert not (outside / "incoming").exists()
    assert not (outside / "releases").exists()


def test_data_prepare_rejects_symlinked_canonical_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside-data"
    outside.mkdir()
    substituted = tmp_path / "data"
    substituted.symlink_to(outside, target_is_directory=True)
    data_policy = {
        "raw_session": {
            "incoming_root": (substituted / ".staging").as_posix(),
            "releases_root": (substituted / "releases").as_posix(),
        },
        "converted_dataset": {
            "incoming_root": (substituted / "converted-staging").as_posix(),
            "releases_root": (substituted / "converted-releases").as_posix(),
        },
    }
    monkeypatch.setitem(
        remote_agent.HOST_POLICIES, "h200",
        {"home": tmp_path.as_posix(), "canary": {}, "production": {}, "data": data_policy},
    )
    incoming = data_policy["raw_session"]["incoming_root"] + "/transfer-raw-" + "1" * 32

    with pytest.raises(remote_agent.AgentPolicyError, match="symlink"):
        remote_agent.prepare_data_incoming(
            "h200", data_operation="archive_raw_session",
            incoming_path=incoming, reservation_id="2" * 64,
        )
    assert not (outside / ".staging").exists()


@pytest.mark.parametrize("lost_mode", ["nonzero", "exception"])
def test_lost_install_response_reinspects_then_rolls_back_and_cleans_created_release(
    tmp_path: Path, lost_mode: str,
) -> None:
    plan = deploy_remote.build_ssh_plan(
        _completed_release(tmp_path), source_role="mac", target_role="h200",
        host_alias="nebius-h200", scope="canary", nonce="b" * 32,
    )
    operations = []
    inspection_count = 0
    cleanup_request = None
    rollback_request = None

    def runner(argv, **kwargs):
        nonlocal inspection_count, cleanup_request, rollback_request
        request = json.loads(kwargs["input"]) if kwargs.get("input") else None
        operation = request["operation"] if request else None
        if operation:
            operations.append(operation)
        if operation == "bootstrap_canary":
            stdout = contract.canonical_json_bytes({
                "schema": "wam_remote_bootstrap_response_v1", "target_role": "h200"
            })
        elif operation == "inspect_canary":
            inspection_count += 1
            committed = inspection_count == 2
            stdout = contract.canonical_json_bytes({
                "current_target": (
                    "releases/" + plan.expected_release_id if committed else None
                ),
                "release_ids": [plan.expected_release_id] if committed else [],
                "schema": "wam_remote_canary_inspection_v1",
                "target_role": "h200",
            })
        elif operation == "prepare_incoming":
            stdout = contract.canonical_json_bytes({
                "incoming_path": plan.incoming_path,
                "reservation_id": plan.reservation_id,
                "schema": "wam_remote_prepare_response_v1", "target_role": "h200",
            })
        elif operation == "install_release":
            if lost_mode == "exception":
                raise TimeoutError("lost install response after commit")
            return SimpleNamespace(
                returncode=255, stdout=b"", stderr=b"lost install response after commit"
            )
        elif operation == "rollback_canary":
            rollback_request = request
            stdout = contract.canonical_json_bytes({
                "rolled_back_to": None,
                "schema": "wam_remote_canary_rollback_v1", "target_role": "h200",
            })
        elif operation == "cleanup_canary":
            cleanup_request = request
            stdout = contract.canonical_json_bytes({
                "removed_incoming": plan.incoming_path,
                "removed_release_id": plan.expected_release_id,
                "schema": "wam_remote_canary_cleanup_v1", "target_role": "h200",
            })
        else:
            stdout = b""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    with pytest.raises(deploy_remote.DeploymentExecutionError) as caught:
        deploy_remote.execute_ssh_plan(plan, runner=runner)
    assert caught.value.evidence["ambiguous_install_inspect"]["release_ids"] == [
        plan.expected_release_id
    ]
    assert cleanup_request["created_release_id"] == plan.expected_release_id
    assert rollback_request["expected_current_target"] == (
        "releases/" + plan.expected_release_id
    )
    assert operations == [
        "bootstrap_canary", "inspect_canary", "prepare_incoming", "install_release",
        "inspect_canary", "rollback_canary", "cleanup_canary",
    ]


def test_remote_agent_installs_verified_incoming_and_cleans_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_release = _completed_release(tmp_path / "source-release")
    release_id = contract.verify_release(source_release)["release_id"]
    target = tmp_path / "remote"
    paths = {
        "current_link": (target / "current").as_posix(),
        "incoming_root": (target / "incoming").as_posix(),
        "releases_root": (target / "releases").as_posix(),
    }
    incoming = Path(paths["incoming_root"]) / (
        "incoming-%s-%s" % (release_id, "3" * 32)
    )
    monkeypatch.setitem(
        remote_agent.HOST_POLICIES,
        "h200",
        {"home": tmp_path.as_posix(), "canary": paths, "production": paths},
    )
    reservation = remote_agent.prepare_incoming(
        "h200", incoming.as_posix(), "3" * 64
    )
    shutil.copytree(source_release, incoming, dirs_exist_ok=True)
    agent = DEPLOYMENT_DIR / "remote_agent.py"
    request = {
        "adopt_existing": False,
        "agent_sha256": hashlib.sha256(agent.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(
            (DEPLOYMENT_DIR / "artifact_contract.py").read_bytes()
        ).hexdigest(),
        "current_link": paths["current_link"],
        "expected_manifest_sha256": hashlib.sha256(
            (incoming / contract.MANIFEST_NAME).read_bytes()
        ).hexdigest(),
        "expected_payload_sha256": contract.verify_release(incoming)["payload_sha256"],
        "expected_release_id": release_id,
        "incoming_path": incoming.as_posix(),
        "operation": "install_release",
        "reservation_id": reservation["reservation_id"],
        "releases_root": paths["releases_root"],
        "schema": "wam_remote_install_request_v1",
        "scope": "canary",
        "target_role": "h200",
    }

    response = remote_agent.execute_request(
        request, local_role="h200", agent_path=agent
    )
    final = Path(response["final_release"])
    marker = json.loads((final / contract.COMPLETE_NAME).read_text())
    assert marker["target_role"] == "h200"
    assert Path(paths["current_link"]).resolve() == final.resolve()
    assert not incoming.exists()


def test_raw_session_validator_rehashes_full_sealed_inventory(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    session = _sealed_session(sessions)
    trusted = _trusted_training_config(session, tmp_path)
    expected_finalization = _trusted_finalization_record(session)["artifact_sha256"][
        "finalization"
    ]
    identity = remote_agent.validate_raw_session(
        session,
        allowed_sessions_root=sessions,
        expected_finalization_sha256=expected_finalization,
        **trusted,
    )
    assert identity["manifest_sha256"] == json.loads(
        (session / "FINALIZATION.json").read_text()
    )["manifest_sha256"]
    assert identity["verified_raw_files"] == 29
    assert identity["verified_components"] == [
        "audio", "force", "geometry", "quality", "robot",
    ]

    (session / "audio.wav").write_bytes(b"tampered")
    with pytest.raises(remote_agent.AgentPolicyError, match="raw training"):
        remote_agent.validate_raw_session(
            session,
            allowed_sessions_root=sessions,
            expected_finalization_sha256=expected_finalization,
            **trusted,
        )
    assert (Path(identity["verified_snapshot_path"]) / "audio.wav").read_bytes() != b"tampered"


def test_raw_session_validator_rejects_legacy_promotion_v1(
    tmp_path: Path,
) -> None:
    sessions = tmp_path / "sessions"
    legacy = _legacy_sealed_session(sessions)
    registry = tmp_path / "empty-trusted-finalizations.json"
    registry.write_bytes(contract.canonical_json_bytes({
        "records": [],
        "schema_version": "wam_trusted_training_finalization_registry_v1",
    }))

    with pytest.raises(remote_agent.AgentPolicyError, match="promotion"):
        remote_agent.validate_raw_session(
            legacy,
            allowed_sessions_root=sessions,
            trusted_finalization_registry=registry,
            trusted_snapshot_root=tmp_path / "trusted-snapshots",
            expected_finalization_sha256="0" * 64,
        )


def test_converted_dataset_validator_checks_all_provenance_bindings(tmp_path: Path) -> None:
    releases = tmp_path / "lerobot" / "releases"
    release = _converted_dataset(releases)
    identity = remote_agent.validate_converted_dataset(
        release, allowed_releases_root=releases
    )
    assert identity["lerobot_version"] == "0.3.3"
    assert identity["dataset_schema_version"] == "lerobot_v2"

    bad_releases = tmp_path / "bad" / "releases"
    bad = _converted_dataset(bad_releases, lerobot_version="0.5.2")
    with pytest.raises(remote_agent.AgentPolicyError, match="LeRobot"):
        remote_agent.validate_converted_dataset(bad, allowed_releases_root=bad_releases)

    bad_schema_root = tmp_path / "bad-schema" / "releases"
    bad_schema = _converted_dataset(
        bad_schema_root, dataset_schema_version="lerobot_v3"
    )
    with pytest.raises(remote_agent.AgentPolicyError, match="dataset schema"):
        remote_agent.validate_converted_dataset(
            bad_schema, allowed_releases_root=bad_schema_root
        )

    bad_raw_root = tmp_path / "bad-raw-id" / "releases"
    bad_raw = _converted_dataset(
        bad_raw_root, raw_release_id="raw_session-" + "9" * 20
    )
    with pytest.raises(remote_agent.AgentPolicyError, match="raw release"):
        remote_agent.validate_converted_dataset(
            bad_raw, allowed_releases_root=bad_raw_root
        )

    legacy_root = tmp_path / "legacy" / "releases"
    legacy = _converted_dataset(
        legacy_root, provenance_schema="wam_converted_dataset_provenance_v1"
    )
    with pytest.raises(remote_agent.AgentPolicyError, match="provenance"):
        remote_agent.validate_converted_dataset(
            legacy, allowed_releases_root=legacy_root
        )


def test_data_operation_policy_separates_h200_archive_and_training_roots() -> None:
    raw = remote_agent.operation_policy("h200", "archive_raw_session")
    converted = remote_agent.operation_policy("h200", "install_converted_dataset")
    assert raw["releases_root"] == "/home/ssh-ziyue/wam_ausc/data/raw_sessions/releases"
    assert converted["releases_root"] == "/home/ssh-ziyue/wam_ausc/data/lerobot/releases"
    assert raw["releases_root"] != converted["releases_root"]
    trust = remote_agent.trusted_training_policy("h200")
    assert trust == {
        "finalization_registry": (
            "/home/ssh-ziyue/wam_ausc/trust/TRAINING_FINALIZATIONS.json"
        ),
        "snapshot_root": (
            "/home/ssh-ziyue/wam_ausc/data/trusted_training_snapshots"
        ),
    }
    with pytest.raises(remote_agent.AgentPolicyError):
        remote_agent.operation_policy("linbox", "archive_raw_session")


def test_data_promotions_are_immutable_and_never_create_current(tmp_path: Path) -> None:
    raw_staging = tmp_path / "raw" / ".staging"
    raw_release_root = tmp_path / "raw" / "releases"
    raw = _sealed_session(raw_staging, "transfer-raw-001")
    trusted = _trusted_training_config(raw, tmp_path)
    expected_finalization = _trusted_finalization_record(raw)["artifact_sha256"][
        "finalization"
    ]
    raw_identity = remote_agent.validate_raw_session(
        raw,
        allowed_sessions_root=raw_staging,
        expected_finalization_sha256=expected_finalization,
        **trusted,
    )
    trusted_audio = (
        Path(raw_identity["verified_snapshot_path"]) / "audio.wav"
    ).read_bytes()
    (raw / "audio.wav").write_bytes(b"producer-mutated-before-archive")
    with pytest.raises(remote_agent.AgentPolicyError, match="raw training"):
        remote_agent.archive_raw_session(
            raw,
            releases_root=raw_release_root,
            expected_manifest_sha256=raw_identity["manifest_sha256"],
            expected_finalization_sha256=raw_identity["finalization_sha256"],
            **trusted,
        )

    # Restoring the exact trusted source allows the immutable archive path to
    # be exercised; a cached snapshot is never allowed to hide source drift.
    (raw / "audio.wav").write_bytes(trusted_audio)
    raw_result = remote_agent.archive_raw_session(
        raw,
        releases_root=raw_release_root,
        expected_manifest_sha256=raw_identity["manifest_sha256"],
        expected_finalization_sha256=raw_identity["finalization_sha256"],
        **trusted,
    )
    assert Path(raw_result["final_release"]).parent == raw_release_root
    assert (Path(raw_result["final_release"]) / "audio.wav").read_bytes() == trusted_audio
    assert not (tmp_path / "raw" / "current").exists()

    converted_staging = tmp_path / "lerobot" / ".staging"
    converted_releases = tmp_path / "lerobot" / "releases"
    converted = _converted_dataset(converted_staging, raw_identity=raw_identity)
    envelope = contract.verify_release(converted)
    converted_result = remote_agent.install_converted_dataset(
        converted,
        releases_root=converted_releases,
        expected_release_id=envelope["release_id"],
        expected_payload_sha256=envelope["payload_sha256"],
        expected_manifest_sha256=hashlib.sha256(
            (converted / contract.MANIFEST_NAME).read_bytes()
        ).hexdigest(),
        raw_releases_root=raw_release_root,
        **trusted,
    )
    assert Path(converted_result["final_release"]).parent == converted_releases
    assert not (tmp_path / "lerobot" / "current").exists()


def test_h200_converted_install_requires_the_bound_raw_v2_archive(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "lerobot" / ".staging"
    converted = _converted_dataset(staging)
    envelope = contract.verify_release(converted)
    missing_trust = {
        "trusted_finalization_registry": tmp_path / "missing-registry.json",
        "trusted_snapshot_root": tmp_path / "trusted-snapshots",
    }

    with pytest.raises(remote_agent.AgentPolicyError, match="raw archive"):
        remote_agent.install_converted_dataset(
            converted,
            releases_root=tmp_path / "lerobot" / "releases",
            expected_release_id=envelope["release_id"],
            expected_payload_sha256=envelope["payload_sha256"],
            expected_manifest_sha256=hashlib.sha256(
                (converted / contract.MANIFEST_NAME).read_bytes()
            ).hexdigest(),
            raw_releases_root=tmp_path / "raw" / "releases",
            **missing_trust,
        )


def test_client_data_plans_enforce_linbox_roots_and_bind_identities() -> None:
    raw = deploy_remote.build_raw_archive_plan(
        "/home/robot/wam_data/sessions/session-001",
        manifest_sha256="1" * 64,
        finalization_sha256="2" * 64,
        nonce="6" * 32,
    )
    assert [step.name for step in raw.steps][-3:] == [
        "prepare_data_incoming", "stage_data", "archive_raw_session"
    ]
    assert raw.steps[-2].argv[3:5] == (
        "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10"
    )
    raw_request = json.loads(raw.steps[-1].stdin)
    raw_prepare = json.loads(raw.steps[-3].stdin)
    assert raw_prepare["reservation_id"] == raw.reservation_id
    assert raw_request["reservation_id"] == raw.reservation_id
    assert raw_request["expected_manifest_sha256"] == "1" * 64
    assert raw_request["expected_finalization_sha256"] == "2" * 64

    converted = deploy_remote.build_converted_dataset_plan(
        "/home/robot/wam_data/lerobot/releases/lerobot_dataset-" + "3" * 20,
        release_id="lerobot_dataset-" + "3" * 20,
        payload_sha256="4" * 64,
        manifest_sha256="5" * 64,
    )
    converted_request = json.loads(converted.steps[-1].stdin)
    assert converted_request["operation"] == "install_converted_dataset"
    assert converted_request["expected_payload_sha256"] == "4" * 64

    with pytest.raises(deploy_remote.PolicyError, match="source root"):
        deploy_remote.build_raw_archive_plan(
            "/tmp/session-001", manifest_sha256="1" * 64,
            finalization_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    "data_kind",
    (
        "perception_geometry_source",
        "perception_labels_train",
        "perception_labels_holdout",
    ),
)
def test_mac_perception_plan_is_fresh_exact_and_pointer_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_kind: str,
) -> None:
    source_root = tmp_path / "mac-perception" / "releases"
    source = _perception_release(
        source_root,
        artifact_kind=data_kind,
        source_role="mac",
    )
    monkeypatch.setattr(deploy_remote, "_MAC_PERCEPTION_RELEASES_ROOT", source_root)
    envelope = contract.verify_release(source)
    plan = deploy_remote.build_perception_artifact_plan(
        source,
        data_kind=data_kind,
        source_role="mac",
        target_role="h200",
        host_alias="nebius-h200",
        nonce="a" * 32,
        reservation_id="b" * 64,
    )

    assert plan.data_kind == data_kind
    assert plan.operation == "install_perception_artifact"
    assert plan.incoming_path == (
        PERCEPTION_ARTIFACT_POLICIES[data_kind]["staging_root"]
        + "/incoming-"
        + envelope["release_id"]
        + "-"
        + "a" * 32
    )
    assert plan.expected_final_release == (
        PERCEPTION_ARTIFACT_POLICIES[data_kind]["releases_root"]
        + "/"
        + envelope["release_id"]
    )
    assert [step.name for step in plan.steps] == [
        "upload_contract",
        "upload_agent",
        "prepare_perception_incoming",
        "stage_perception_artifact",
        "install_perception_artifact",
    ]
    stage = plan.steps[-2]
    assert stage.argv[:3] == ("rsync", "-a", "-c")
    assert "--delete" in stage.argv and "--delay-updates" in stage.argv
    assert "--inplace" not in stage.argv
    prepare = json.loads(plan.steps[-3].stdin)
    install = json.loads(plan.steps[-1].stdin)
    assert set(prepare) == {
        "agent_sha256",
        "contract_sha256",
        "data_kind",
        "incoming_path",
        "operation",
        "reservation_id",
        "schema",
        "source_role",
        "target_role",
    }
    assert prepare["schema"] == "wam_remote_perception_prepare_request_v1"
    assert prepare["operation"] == "prepare_perception_incoming"
    assert set(install) == {
        "agent_sha256",
        "contract_sha256",
        "data_kind",
        "expected_manifest_sha256",
        "expected_payload_sha256",
        "expected_release_id",
        "incoming_path",
        "operation",
        "releases_root",
        "reservation_id",
        "schema",
        "source_role",
        "target_role",
    }
    assert install["schema"] == "wam_remote_perception_request_v1"
    assert install["source_role"] == "mac"
    assert install["data_kind"] == data_kind
    assert install["expected_release_id"] == envelope["release_id"]
    assert install["expected_payload_sha256"] == envelope["payload_sha256"]
    assert install["expected_manifest_sha256"] == hashlib.sha256(
        (source / contract.MANIFEST_NAME).read_bytes()
    ).hexdigest()
    assert not any("current" in key for key in prepare | install)
    assert "current" not in repr(plan)


@pytest.mark.parametrize(
    "data_kind",
    (
        "perception_stage0",
        "perception_session2_train",
        "perception_session3_holdout",
        "perception_canary_evaluation",
        "raw_session",
    ),
)
def test_mac_perception_plan_rejects_h200_local_or_fabricated_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_kind: str,
) -> None:
    source_root = tmp_path / "mac-perception" / "releases"
    source = _perception_release(
        source_root,
        artifact_kind=data_kind,
        source_role="mac",
    )
    monkeypatch.setattr(deploy_remote, "_MAC_PERCEPTION_RELEASES_ROOT", source_root)

    with pytest.raises(deploy_remote.PolicyError, match="remote upload"):
        deploy_remote.build_perception_artifact_plan(
            source,
            data_kind=data_kind,
            source_role="mac",
            target_role="h200",
            host_alias="nebius-h200",
        )


@pytest.mark.parametrize(
    ("source_role", "target_role", "host_alias"),
    (
        ("linbox", "h200", "h200"),
        ("mac", "linbox", "linbox"),
        ("mac", "h200", "h200"),
    ),
)
def test_perception_plan_rejects_any_linbox_or_wrong_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_role: str,
    target_role: str,
    host_alias: str,
) -> None:
    source_root = tmp_path / "mac-perception" / "releases"
    source = _perception_release(
        source_root,
        artifact_kind="perception_geometry_source",
        source_role="mac",
    )
    monkeypatch.setattr(deploy_remote, "_MAC_PERCEPTION_RELEASES_ROOT", source_root)

    with pytest.raises(deploy_remote.PolicyError):
        deploy_remote.build_perception_artifact_plan(
            source,
            data_kind="perception_geometry_source",
            source_role=source_role,
            target_role=target_role,
            host_alias=host_alias,
        )


def test_perception_plan_rejects_wrong_source_root_kind_and_declared_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_root = tmp_path / "canonical" / "releases"
    outside_root = tmp_path / "outside" / "releases"
    monkeypatch.setattr(deploy_remote, "_MAC_PERCEPTION_RELEASES_ROOT", canonical_root)
    outside = _perception_release(
        outside_root,
        artifact_kind="perception_geometry_source",
        source_role="mac",
    )
    with pytest.raises(deploy_remote.PolicyError, match="source root"):
        deploy_remote.build_perception_artifact_plan(
            outside,
            data_kind="perception_geometry_source",
            source_role="mac",
            target_role="h200",
            host_alias="nebius-h200",
        )

    wrong_kind = _perception_release(
        canonical_root,
        artifact_kind="perception_labels_train",
        source_role="mac",
    )
    with pytest.raises(deploy_remote.PolicyError, match="kind"):
        deploy_remote.build_perception_artifact_plan(
            wrong_kind,
            data_kind="perception_geometry_source",
            source_role="mac",
            target_role="h200",
            host_alias="nebius-h200",
        )

    wrong_role_root = tmp_path / "wrong-role" / "releases"
    wrong_role = _perception_release(
        wrong_role_root,
        artifact_kind="perception_geometry_source",
        source_role="h200",
    )
    monkeypatch.setattr(deploy_remote, "_MAC_PERCEPTION_RELEASES_ROOT", wrong_role_root)
    with pytest.raises(deploy_remote.PolicyError, match="source role"):
        deploy_remote.build_perception_artifact_plan(
            wrong_role,
            data_kind="perception_geometry_source",
            source_role="mac",
            target_role="h200",
            host_alias="nebius-h200",
        )


def _staged_perception_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    data_kind: str = "perception_geometry_source",
    reservation_id: str = "9" * 64,
    nonce: str = "8" * 32,
) -> tuple[Path, Path, dict, dict]:
    policies = _local_perception_agent_policy(tmp_path, monkeypatch)
    source = _perception_release(
        tmp_path / "mac-perception" / "releases",
        artifact_kind=data_kind,
        source_role="mac",
    )
    envelope = contract.verify_release(source)
    incoming = (
        Path(policies[data_kind]["staging_root"])
        / ("incoming-%s-%s" % (envelope["release_id"], nonce))
    )
    agent_sha256 = hashlib.sha256(
        (DEPLOYMENT_DIR / "remote_agent.py").read_bytes()
    ).hexdigest()
    contract_sha256 = hashlib.sha256(
        (DEPLOYMENT_DIR / "artifact_contract.py").read_bytes()
    ).hexdigest()
    prepare = {
        "agent_sha256": agent_sha256,
        "contract_sha256": contract_sha256,
        "data_kind": data_kind,
        "incoming_path": incoming.as_posix(),
        "operation": "prepare_perception_incoming",
        "reservation_id": reservation_id,
        "schema": "wam_remote_perception_prepare_request_v1",
        "source_role": "mac",
        "target_role": "h200",
    }
    remote_agent.dispatch_request(
        prepare,
        local_role="h200",
        agent_path=DEPLOYMENT_DIR / "remote_agent.py",
    )
    shutil.copytree(source, incoming, dirs_exist_ok=True)
    install = {
        "agent_sha256": agent_sha256,
        "contract_sha256": contract_sha256,
        "data_kind": data_kind,
        "expected_manifest_sha256": hashlib.sha256(
            (source / contract.MANIFEST_NAME).read_bytes()
        ).hexdigest(),
        "expected_payload_sha256": envelope["payload_sha256"],
        "expected_release_id": envelope["release_id"],
        "incoming_path": incoming.as_posix(),
        "operation": "install_perception_artifact",
        "releases_root": policies[data_kind]["releases_root"],
        "reservation_id": reservation_id,
        "schema": "wam_remote_perception_request_v1",
        "source_role": "mac",
        "target_role": "h200",
    }
    reconcile = dict(install)
    reconcile["operation"] = "reconcile_perception_transfer"
    reconcile["schema"] = "wam_remote_perception_reconcile_request_v1"
    return source, incoming, install, reconcile


def test_h200_perception_policy_contains_seven_exact_pointer_free_namespaces() -> None:
    assert remote_agent.HOST_POLICIES["h200"]["data"]["perception"] == (
        PERCEPTION_ARTIFACT_POLICIES
    )
    assert remote_agent.HOST_POLICIES["h200"]["data"][
        "perception_holdout_consumption_root"
    ] == "/home/ssh-ziyue/wam_ausc/data/perception_holdout_consumption"
    for data_kind, expected in PERCEPTION_ARTIFACT_POLICIES.items():
        assert remote_agent.perception_artifact_policy("h200", data_kind) == expected
        assert "current_link" not in expected
    with pytest.raises(remote_agent.AgentPolicyError, match="H200"):
        remote_agent.perception_artifact_policy(
            "linbox", "perception_geometry_source"
        )


@pytest.mark.parametrize(
    ("field", "poisoned_path"),
    (
        (
            "releases_root",
            "/home/ssh-ziyue/wam_ausc/data/current/"
            "perception_geometry_source/releases",
        ),
        (
            "staging_root",
            "/home/ssh-ziyue/wam_ausc/data/"
            "perception_geometry_source/current/.staging",
        ),
    ),
)
def test_perception_policy_rejects_current_as_normalized_root_component(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    poisoned_path: str,
) -> None:
    policy = remote_agent.HOST_POLICIES["h200"]["data"]["perception"][
        "perception_geometry_source"
    ]
    monkeypatch.setitem(policy, field, poisoned_path)

    with pytest.raises(remote_agent.AgentPolicyError, match="root is invalid"):
        remote_agent.perception_artifact_policy(
            "h200", "perception_geometry_source"
        )


@pytest.mark.parametrize("field", ("releases_root", "staging_root"))
def test_perception_policy_rejects_parent_traversal_in_root(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    policy = remote_agent.HOST_POLICIES["h200"]["data"]["perception"][
        "perception_geometry_source"
    ]
    monkeypatch.setitem(
        policy,
        field,
        "/home/ssh-ziyue/wam_ausc/data/../escape/" + field,
    )

    with pytest.raises(remote_agent.AgentPolicyError, match="root is invalid"):
        remote_agent.perception_artifact_policy(
            "h200", "perception_geometry_source"
        )


def test_perception_install_rehashes_rematerializes_h200_marker_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, incoming, request, _reconcile = _staged_perception_request(
        tmp_path,
        monkeypatch,
    )
    untouched_current = tmp_path / "h200-perception" / "current"
    untouched_current.parent.mkdir(parents=True, exist_ok=True)
    untouched_current.symlink_to("preexisting-target")

    first = remote_agent.dispatch_request(
        request,
        local_role="h200",
        agent_path=DEPLOYMENT_DIR / "remote_agent.py",
    )
    final = Path(first["final_release"])
    assert set(first) == {
        "artifact_kind",
        "created_release",
        "final_release",
        "manifest_sha256",
        "payload_sha256",
        "release_id",
        "schema",
        "source_role",
        "target_role",
    }
    assert first == {
        "artifact_kind": "perception_geometry_source",
        "created_release": True,
        "final_release": final.as_posix(),
        "manifest_sha256": request["expected_manifest_sha256"],
        "payload_sha256": request["expected_payload_sha256"],
        "release_id": request["expected_release_id"],
        "schema": "wam_perception_artifact_install_response_v1",
        "source_role": "mac",
        "target_role": "h200",
    }
    assert contract.verify_release(final)["release_id"] == source.name
    marker = json.loads((final / contract.COMPLETE_NAME).read_text())
    assert marker["target_role"] == "h200"
    assert marker["source_role"] == "mac"
    assert not incoming.exists()
    assert not remote_agent._reservation_marker_path(incoming).exists()
    assert os.readlink(untouched_current) == "preexisting-target"

    _source2, incoming2, request2, _reconcile2 = _staged_perception_request(
        tmp_path / "retry",
        monkeypatch,
        reservation_id="7" * 64,
        nonce="6" * 32,
    )
    # Point the retry policy at the already installed immutable namespace.
    request2["releases_root"] = final.parent.as_posix()
    remote_agent.HOST_POLICIES["h200"]["data"]["perception"][
        "perception_geometry_source"
    ]["releases_root"] = final.parent.as_posix()
    second = remote_agent.dispatch_request(
        request2,
        local_role="h200",
        agent_path=DEPLOYMENT_DIR / "remote_agent.py",
    )
    assert second["created_release"] is False
    assert second["final_release"] == final.as_posix()
    assert not incoming2.exists()
    assert os.readlink(untouched_current) == "preexisting-target"


def test_perception_remote_request_rejects_wrong_policy_and_raw_fabrication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, _incoming, request, _reconcile = _staged_perception_request(
        tmp_path,
        monkeypatch,
    )
    poisons = (
        ("data_kind", "perception_stage0"),
        ("data_kind", "raw_session"),
        ("source_role", "linbox"),
        ("target_role", "linbox"),
        ("releases_root", request["releases_root"] + "-substituted"),
        ("expected_release_id", "raw_session-" + "1" * 20),
    )
    for field, replacement in poisons:
        changed = dict(request)
        changed[field] = replacement
        with pytest.raises(remote_agent.AgentPolicyError):
            remote_agent.dispatch_request(
                changed,
                local_role="h200",
                agent_path=DEPLOYMENT_DIR / "remote_agent.py",
            )
    changed = dict(request)
    changed["unexpected"] = True
    with pytest.raises(remote_agent.AgentPolicyError, match="fields"):
        remote_agent.dispatch_request(
            changed,
            local_role="h200",
            agent_path=DEPLOYMENT_DIR / "remote_agent.py",
        )


@pytest.mark.parametrize("poison", ("extra", "symlink"))
def test_perception_post_rsync_exact_set_failure_reconciles_only_owned_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison: str,
) -> None:
    _source, incoming, request, reconcile = _staged_perception_request(
        tmp_path,
        monkeypatch,
    )
    if poison == "extra":
        (incoming / "extra.bin").write_bytes(b"extra")
    else:
        (incoming / "linked.bin").symlink_to(incoming / "payload.bin")
    sibling = incoming.parent / "unowned-sibling"
    sibling.mkdir()
    (sibling / "preserve").write_bytes(b"preserve")

    with pytest.raises(Exception, match="exact-set|symlink"):
        remote_agent.dispatch_request(
            request,
            local_role="h200",
            agent_path=DEPLOYMENT_DIR / "remote_agent.py",
        )
    recovered = remote_agent.dispatch_request(
        reconcile,
        local_role="h200",
        agent_path=DEPLOYMENT_DIR / "remote_agent.py",
    )
    assert recovered == {
        "artifact_kind": "perception_geometry_source",
        "final_release": None,
        "incoming_present": False,
        "marker_present": False,
        "schema": "wam_remote_perception_reconcile_response_v1",
        "status": "cleaned",
        "target_role": "h200",
    }
    assert (sibling / "preserve").read_bytes() == b"preserve"


def test_perception_conflict_preserves_existing_release_and_owned_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, incoming, request, _reconcile = _staged_perception_request(
        tmp_path,
        monkeypatch,
    )
    final = Path(request["releases_root"]) / request["expected_release_id"]
    final.mkdir(parents=True)
    sentinel = final / "preserve"
    sentinel.write_bytes(b"conflict")

    with pytest.raises(Exception, match="conflict|invalid|verification"):
        remote_agent.dispatch_request(
            request,
            local_role="h200",
            agent_path=DEPLOYMENT_DIR / "remote_agent.py",
        )
    assert sentinel.read_bytes() == b"conflict"
    assert incoming.exists()
    assert remote_agent._reservation_marker_path(incoming).exists()


@pytest.mark.parametrize("kind", ["raw", "converted"])
def test_execute_data_ssh_plan_validates_exact_remote_responses(kind: str) -> None:
    if kind == "raw":
        plan = deploy_remote.build_raw_archive_plan(
            "/home/robot/wam_data/sessions/session-001",
            manifest_sha256="1" * 64,
            finalization_sha256="2" * 64,
            nonce="8" * 32,
        )
        result_schema = "wam_raw_session_archive_response_v1"
    else:
        release_id = "lerobot_dataset-" + "3" * 20
        plan = deploy_remote.build_converted_dataset_plan(
            "/home/robot/wam_data/lerobot/releases/" + release_id,
            release_id=release_id,
            payload_sha256="4" * 64,
            manifest_sha256="5" * 64,
        )
        result_schema = "wam_converted_dataset_install_response_v1"
    calls = []

    def runner(argv, **kwargs):
        calls.append((tuple(argv), kwargs.get("input")))
        if not kwargs.get("input"):
            stdout = b""
        elif json.loads(kwargs["input"])["operation"] == "prepare_data_incoming":
            stdout = contract.canonical_json_bytes({
                "incoming_path": plan.incoming_path,
                "reservation_id": plan.reservation_id,
                "schema": "wam_remote_data_prepare_response_v1",
                "target_role": "h200",
            })
        else:
            assert json.loads(kwargs["input"])["reservation_id"] == plan.reservation_id
            stdout = contract.canonical_json_bytes({
                "final_release": plan.expected_final_release,
                "release_id": plan.expected_release_id,
                "schema": result_schema,
                "target_role": "h200",
            })
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    evidence = deploy_remote.execute_data_ssh_plan(plan, runner=runner)
    assert len(calls) == 5
    assert evidence["schema"] == "wam_ssh_data_evidence_v1"
    assert evidence["result"]["release_id"] == plan.expected_release_id


def test_execute_perception_plan_validates_exact_canonical_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "mac-perception" / "releases"
    source = _perception_release(
        source_root,
        artifact_kind="perception_geometry_source",
        source_role="mac",
    )
    monkeypatch.setattr(deploy_remote, "_MAC_PERCEPTION_RELEASES_ROOT", source_root)
    plan = deploy_remote.build_perception_artifact_plan(
        source,
        data_kind="perception_geometry_source",
        source_role="mac",
        target_role="h200",
        host_alias="nebius-h200",
        nonce="1" * 32,
        reservation_id="2" * 64,
    )
    calls = []

    def runner(argv, **kwargs):
        request = json.loads(kwargs["input"]) if kwargs.get("input") else None
        calls.append((tuple(argv), request))
        if request is None:
            stdout = b""
        elif request["operation"] == "prepare_perception_incoming":
            stdout = contract.canonical_json_bytes({
                "data_kind": plan.data_kind,
                "incoming_path": plan.incoming_path,
                "reservation_id": plan.reservation_id,
                "schema": "wam_remote_perception_prepare_response_v1",
                "source_role": "mac",
                "target_role": "h200",
            })
        else:
            stdout = contract.canonical_json_bytes({
                "artifact_kind": plan.data_kind,
                "created_release": True,
                "final_release": plan.expected_final_release,
                "manifest_sha256": plan.expected_manifest_sha256,
                "payload_sha256": plan.expected_payload_sha256,
                "release_id": plan.expected_release_id,
                "schema": "wam_perception_artifact_install_response_v1",
                "source_role": "mac",
                "target_role": "h200",
            })
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    evidence = deploy_remote.execute_data_ssh_plan(plan, runner=runner)
    assert len(calls) == 5
    assert evidence == {
        "operation": "install_perception_artifact",
        "prepare": {
            "data_kind": "perception_geometry_source",
            "incoming_path": plan.incoming_path,
            "reservation_id": plan.reservation_id,
            "schema": "wam_remote_perception_prepare_response_v1",
            "source_role": "mac",
            "target_role": "h200",
        },
        "result": {
            "artifact_kind": "perception_geometry_source",
            "created_release": True,
            "final_release": plan.expected_final_release,
            "manifest_sha256": plan.expected_manifest_sha256,
            "payload_sha256": plan.expected_payload_sha256,
            "release_id": plan.expected_release_id,
            "schema": "wam_perception_artifact_install_response_v1",
            "source_role": "mac",
            "target_role": "h200",
        },
        "schema": "wam_ssh_data_evidence_v1",
        "target_role": "h200",
    }


def test_lost_perception_install_response_reconciles_without_pointer_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "mac-perception" / "releases"
    source = _perception_release(
        source_root,
        artifact_kind="perception_labels_train",
        source_role="mac",
    )
    monkeypatch.setattr(deploy_remote, "_MAC_PERCEPTION_RELEASES_ROOT", source_root)
    plan = deploy_remote.build_perception_artifact_plan(
        source,
        data_kind="perception_labels_train",
        source_role="mac",
        target_role="h200",
        host_alias="nebius-h200",
        nonce="3" * 32,
        reservation_id="4" * 64,
    )
    operations = []

    def runner(argv, **kwargs):
        request = json.loads(kwargs["input"]) if kwargs.get("input") else None
        if request is None:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        operations.append(request["operation"])
        if request["operation"] == "prepare_perception_incoming":
            stdout = contract.canonical_json_bytes({
                "data_kind": plan.data_kind,
                "incoming_path": plan.incoming_path,
                "reservation_id": plan.reservation_id,
                "schema": "wam_remote_perception_prepare_response_v1",
                "source_role": "mac",
                "target_role": "h200",
            })
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        if request["operation"] == "install_perception_artifact":
            return SimpleNamespace(
                returncode=255,
                stdout=b"",
                stderr=b"response lost after install",
            )
        assert request["operation"] == "reconcile_perception_transfer"
        assert "current_link" not in request
        stdout = contract.canonical_json_bytes({
            "artifact_kind": plan.data_kind,
            "final_release": plan.expected_final_release,
            "incoming_present": False,
            "marker_present": False,
            "schema": "wam_remote_perception_reconcile_response_v1",
            "status": "promoted",
            "target_role": "h200",
        })
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    with pytest.raises(deploy_remote.DeploymentExecutionError) as caught:
        deploy_remote.execute_data_ssh_plan(plan, runner=runner)
    assert operations == [
        "prepare_perception_incoming",
        "install_perception_artifact",
        "reconcile_perception_transfer",
    ]
    assert caught.value.evidence["reconcile"]["status"] == "promoted"


def test_execute_data_ssh_plan_rejects_extra_response_fields() -> None:
    plan = deploy_remote.build_raw_archive_plan(
        "/home/robot/wam_data/sessions/session-001",
        manifest_sha256="1" * 64,
        finalization_sha256="2" * 64,
        nonce="9" * 32,
    )

    def runner(argv, **kwargs):
        if not kwargs.get("input"):
            stdout = b""
        elif json.loads(kwargs["input"])["operation"] == "prepare_data_incoming":
            stdout = contract.canonical_json_bytes({
                "incoming_path": plan.incoming_path,
                "reservation_id": plan.reservation_id,
                "schema": "wam_remote_data_prepare_response_v1",
                "target_role": "h200",
            })
        else:
            stdout = contract.canonical_json_bytes({
                "current_link": "/forbidden",
                "final_release": plan.expected_final_release,
                "release_id": plan.expected_release_id,
                "schema": "wam_raw_session_archive_response_v1",
                "target_role": "h200",
            })
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    with pytest.raises(deploy_remote.DeploymentExecutionError, match="outside policy"):
        deploy_remote.execute_data_ssh_plan(plan, runner=runner)


@pytest.mark.parametrize(
    ("failure_mode", "reconcile_status"),
    [("rsync", "cleaned"), ("validation", "cleaned"), ("lost_response", "promoted")],
)
def test_execute_data_plan_reconciles_owned_reservation_after_failure(
    failure_mode: str, reconcile_status: str,
) -> None:
    plan = deploy_remote.build_raw_archive_plan(
        "/home/robot/wam_data/sessions/session-001",
        manifest_sha256="1" * 64,
        finalization_sha256="2" * 64,
        nonce="f" * 32,
        reservation_id="a" * 64,
    )
    operations = []
    reconcile_request = None

    def runner(argv, **kwargs):
        nonlocal reconcile_request
        request = json.loads(kwargs["input"]) if kwargs.get("input") else None
        operation = request["operation"] if request else None
        if operation:
            operations.append(operation)
        if not request:
            is_stage = argv[-1] == "h200:%s/" % plan.incoming_path
            return SimpleNamespace(
                returncode=23 if failure_mode == "rsync" and is_stage else 0,
                stdout=b"", stderr=b"rsync failed",
            )
        if operation == "prepare_data_incoming":
            stdout = contract.canonical_json_bytes({
                "incoming_path": plan.incoming_path,
                "reservation_id": plan.reservation_id,
                "schema": "wam_remote_data_prepare_response_v1",
                "target_role": "h200",
            })
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        if operation == plan.operation:
            return SimpleNamespace(
                returncode=255 if failure_mode == "lost_response" else 65,
                stdout=b"", stderr=b"remote result unavailable",
            )
        if operation == "reconcile_data_transfer":
            reconcile_request = request
            stdout = contract.canonical_json_bytes({
                "final_release": (
                    plan.expected_final_release if reconcile_status == "promoted" else None
                ),
                "incoming_present": False,
                "marker_present": False,
                "schema": "wam_remote_data_reconcile_response_v1",
                "status": reconcile_status,
                "target_role": "h200",
            })
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        raise AssertionError("unexpected operation: %r" % operation)

    with pytest.raises(deploy_remote.DeploymentExecutionError) as caught:
        deploy_remote.execute_data_ssh_plan(plan, runner=runner)
    assert reconcile_request["reservation_id"] == plan.reservation_id
    assert reconcile_request["data_operation"] == plan.operation
    assert caught.value.evidence["reconcile"]["status"] == reconcile_status
    assert operations[-1] == "reconcile_data_transfer"


@pytest.mark.parametrize("prepare_failure", ["lost", "malformed", "nonzero"])
def test_execute_data_plan_reconciles_after_ambiguous_prepare_attempt(
    prepare_failure: str,
) -> None:
    release_id = "lerobot_dataset-" + "b" * 20
    plan = deploy_remote.build_converted_dataset_plan(
        "/home/robot/wam_data/lerobot/releases/" + release_id,
        release_id=release_id,
        payload_sha256="c" * 64,
        manifest_sha256="d" * 64,
        reservation_id="e" * 64,
    )
    operations = []

    def runner(argv, **kwargs):
        request = json.loads(kwargs["input"]) if kwargs.get("input") else None
        operation = request["operation"] if request else None
        if operation:
            operations.append(operation)
        if not request:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if operation == "prepare_data_incoming":
            if prepare_failure == "lost":
                raise TimeoutError("prepare response lost after reservation")
            if prepare_failure == "malformed":
                return SimpleNamespace(returncode=0, stdout=b"not-json", stderr=b"")
            return SimpleNamespace(
                returncode=70, stdout=b"", stderr=b"prepare result unavailable"
            )
        if operation == "reconcile_data_transfer":
            stdout = contract.canonical_json_bytes({
                "final_release": None,
                "incoming_present": False,
                "marker_present": False,
                "schema": "wam_remote_data_reconcile_response_v1",
                "status": "cleaned",
                "target_role": "h200",
            })
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        raise AssertionError("unexpected operation: %r" % operation)

    with pytest.raises(deploy_remote.DeploymentExecutionError) as caught:
        deploy_remote.execute_data_ssh_plan(plan, runner=runner)
    assert caught.value.evidence["reconcile"]["status"] == "cleaned"
    assert operations == ["prepare_data_incoming", "reconcile_data_transfer"]


def test_converted_reservation_reconcile_allows_deterministic_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "h200-data"
    data_policy = {
        "training_trust": {
            "finalization_registry": (root / "trust.json").as_posix(),
            "snapshot_root": (root / "trusted-snapshots").as_posix(),
        },
        "raw_session": {
            "incoming_root": (root / "raw-staging").as_posix(),
            "releases_root": (root / "raw-releases").as_posix(),
        },
        "converted_dataset": {
            "incoming_root": (root / "converted-staging").as_posix(),
            "releases_root": (root / "converted-releases").as_posix(),
        },
    }
    monkeypatch.setitem(
        remote_agent.HOST_POLICIES, "h200",
        {"home": tmp_path.as_posix(), "canary": {}, "production": {}, "data": data_policy},
    )
    release_id = "lerobot_dataset-" + "f" * 20
    incoming = Path(data_policy["converted_dataset"]["incoming_root"]) / release_id
    first_owner = "1" * 64
    remote_agent.prepare_data_incoming(
        "h200", data_operation="install_converted_dataset",
        incoming_path=incoming.as_posix(), reservation_id=first_owner,
    )
    request = {
        "data_operation": "install_converted_dataset",
        "expected_manifest_sha256": "2" * 64,
        "expected_payload_sha256": "3" * 64,
        "expected_release_id": release_id,
        "incoming_path": incoming.as_posix(),
        "releases_root": data_policy["converted_dataset"]["releases_root"],
        "reservation_id": first_owner,
    }
    reconciled = remote_agent.reconcile_data_transfer(request, local_role="h200")
    assert reconciled["status"] == "cleaned"

    retried = remote_agent.prepare_data_incoming(
        "h200", data_operation="install_converted_dataset",
        incoming_path=incoming.as_posix(), reservation_id="4" * 64,
    )
    assert retried["reservation_id"] == "4" * 64


@pytest.mark.parametrize(
    ("promotion_outcome", "expected_status"),
    [("success", "promoted"), ("failure", "cleaned")],
)
def test_data_reconcile_waits_for_same_namespace_promotion_and_converges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    promotion_outcome: str,
    expected_status: str,
) -> None:
    data_policy, incoming, promotion_request, reconcile_request = (
        _staged_converted_request(tmp_path, monkeypatch)
    )
    original_install = remote_agent.install_converted_dataset
    promotion_entered = threading.Event()
    allow_promotion = threading.Event()
    reconcile_attempted = threading.Event()
    reconcile_finished = threading.Event()
    results = {}
    errors = {}

    def controlled_install(*args, **kwargs):
        promotion_entered.set()
        if not allow_promotion.wait(5):
            raise AssertionError("test did not release the controlled promotion")
        if promotion_outcome == "failure":
            raise remote_agent.AgentPolicyError("controlled promotion failure")
        return original_install(*args, **kwargs)

    def promote() -> None:
        try:
            results["promotion"] = remote_agent.dispatch_request(
                promotion_request,
                local_role="h200",
                agent_path=DEPLOYMENT_DIR / "remote_agent.py",
            )
        except BaseException as exc:
            errors["promotion"] = exc

    def reconcile() -> None:
        reconcile_attempted.set()
        try:
            results["reconcile"] = remote_agent.reconcile_data_transfer(
                reconcile_request, local_role="h200"
            )
        except BaseException as exc:
            errors["reconcile"] = exc
        finally:
            reconcile_finished.set()

    monkeypatch.setattr(remote_agent, "install_converted_dataset", controlled_install)
    promotion_thread = threading.Thread(target=promote)
    reconcile_thread = threading.Thread(target=reconcile)
    promotion_thread.start()
    assert promotion_entered.wait(5)
    reconcile_thread.start()
    assert reconcile_attempted.wait(5)
    reconcile_was_blocked = not reconcile_finished.wait(0.2)
    allow_promotion.set()
    promotion_thread.join(5)
    reconcile_thread.join(5)

    assert reconcile_was_blocked is True
    assert not promotion_thread.is_alive() and not reconcile_thread.is_alive()
    assert "reconcile" not in errors
    assert results["reconcile"]["status"] == expected_status
    assert not incoming.exists()
    assert not remote_agent._reservation_marker_path(incoming).exists()
    final = Path(data_policy["converted_dataset"]["releases_root"]) / incoming.name
    if promotion_outcome == "success":
        assert "promotion" not in errors
        assert results["promotion"]["release_id"] == incoming.name
        assert contract.verify_release(final)["release_id"] == incoming.name
    else:
        assert isinstance(errors.get("promotion"), remote_agent.AgentPolicyError)
        assert not final.exists()


def test_data_namespace_lock_blocks_same_namespace_prepare_but_not_other_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_policy, _incoming, promotion_request, _reconcile_request = (
        _staged_converted_request(tmp_path, monkeypatch)
    )
    original_install = remote_agent.install_converted_dataset
    promotion_entered = threading.Event()
    allow_promotion = threading.Event()
    same_prepare_attempted = threading.Event()
    same_prepare_finished = threading.Event()
    promotion_errors = []
    same_prepare_errors = []

    def controlled_install(*args, **kwargs):
        promotion_entered.set()
        if not allow_promotion.wait(5):
            raise AssertionError("test did not release the controlled promotion")
        return original_install(*args, **kwargs)

    def promote() -> None:
        try:
            remote_agent.dispatch_request(
                promotion_request,
                local_role="h200",
                agent_path=DEPLOYMENT_DIR / "remote_agent.py",
            )
        except BaseException as exc:
            promotion_errors.append(exc)

    next_release = "lerobot_dataset-" + "b" * 20
    next_incoming = (
        Path(data_policy["converted_dataset"]["incoming_root"]) / next_release
    )

    def prepare_same_namespace() -> None:
        same_prepare_attempted.set()
        try:
            remote_agent.prepare_data_incoming(
                "h200",
                data_operation="install_converted_dataset",
                incoming_path=next_incoming.as_posix(),
                reservation_id="b" * 64,
            )
        except BaseException as exc:
            same_prepare_errors.append(exc)
        finally:
            same_prepare_finished.set()

    monkeypatch.setattr(remote_agent, "install_converted_dataset", controlled_install)
    promotion_thread = threading.Thread(target=promote)
    same_prepare_thread = threading.Thread(target=prepare_same_namespace)
    promotion_thread.start()
    assert promotion_entered.wait(5)
    same_prepare_thread.start()
    assert same_prepare_attempted.wait(5)
    same_prepare_was_blocked = not same_prepare_finished.wait(0.2)

    raw_incoming = (
        Path(data_policy["raw_session"]["incoming_root"])
        / ("transfer-raw-" + "c" * 32)
    )
    raw_prepared = remote_agent.prepare_data_incoming(
        "h200",
        data_operation="archive_raw_session",
        incoming_path=raw_incoming.as_posix(),
        reservation_id="c" * 64,
    )
    allow_promotion.set()
    promotion_thread.join(5)
    same_prepare_thread.join(5)

    assert same_prepare_was_blocked is True
    assert raw_prepared["incoming_path"] == raw_incoming.as_posix()
    assert not promotion_errors and not same_prepare_errors
    assert not promotion_thread.is_alive() and not same_prepare_thread.is_alive()


def test_remote_agent_rejects_production_code_install(tmp_path: Path) -> None:
    plan = deploy_remote.build_ssh_plan(
        _completed_release(tmp_path), source_role="mac", target_role="h200",
        host_alias="nebius-h200", scope="canary", nonce="7" * 32,
    )
    request = json.loads(plan.steps[6].stdin)
    request["scope"] = "production"
    with pytest.raises(remote_agent.AgentPolicyError, match="Task 8"):
        remote_agent.validate_request(
            request, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )


def _production_policy(tmp_path: Path, role: str) -> dict:
    root = tmp_path / role
    if role == "h200":
        production = {
            "current_link": (root / "current").as_posix(),
            "incoming_root": (root / ".incoming").as_posix(),
            "releases_root": (root / ".releases").as_posix(),
        }
    else:
        production = {
            "current_link": (root / ".wam-current").as_posix(),
            "incoming_root": (root / ".wam-incoming").as_posix(),
            "releases_root": (root / ".wam-releases").as_posix(),
        }
    return {
        "home": tmp_path.as_posix(),
        "helper_path": remote_agent.HOST_POLICIES[role]["helper_path"],
        "contract_path": remote_agent.HOST_POLICIES[role]["contract_path"],
        "canary": production,
        "production": production,
        "data": {},
    }


def _stage_production_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str,
    *, body: bytes = b"production\n",
) -> tuple[Path, dict, dict]:
    source = _completed_release(tmp_path / "source", body)
    envelope = contract.verify_release(source)
    policy = _production_policy(tmp_path / "remote", role)
    monkeypatch.setitem(remote_agent.HOST_POLICIES, role, policy)
    incoming = Path(policy["production"]["incoming_root"]) / (
        "incoming-%s-%s" % (envelope["release_id"], "a" * 32)
    )
    reservation = "b" * 64
    prepared = remote_agent.prepare_production_incoming(
        role,
        incoming_path=incoming.as_posix(),
        reservation_id=reservation,
        frozen_release_id=envelope["release_id"],
        frozen_payload_sha256=envelope["payload_sha256"],
    )
    assert prepared["reservation_id"] == reservation
    shutil.copytree(source, incoming, dirs_exist_ok=True)
    agent = DEPLOYMENT_DIR / "remote_agent.py"
    common = {
        "agent_sha256": hashlib.sha256(agent.read_bytes()).hexdigest(),
        "capability": "task8_frozen_release_activation_v1",
        "contract_sha256": hashlib.sha256(
            (DEPLOYMENT_DIR / "artifact_contract.py").read_bytes()
        ).hexdigest(),
        "expected_manifest_sha256": hashlib.sha256(
            (source / contract.MANIFEST_NAME).read_bytes()
        ).hexdigest(),
        "frozen_payload_sha256": envelope["payload_sha256"],
        "frozen_release_id": envelope["release_id"],
        "incoming_path": incoming.as_posix(),
        "reservation_id": reservation,
        "target_role": role,
    }
    request = dict(
        common,
        operation="activate_production_release",
        schema="wam_remote_production_activate_request_v1",
    )
    reconcile = dict(
        common,
        operation="reconcile_production_release",
        schema="wam_remote_production_reconcile_request_v1",
    )
    return source, request, reconcile


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="POSIX owner-mode test")
def test_production_code_release_directories_block_import_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200", body=b"VALUE = 1\n"
    )
    response = remote_agent.dispatch_request(
        request,
        local_role="h200",
        agent_path=DEPLOYMENT_DIR / "remote_agent.py",
    )
    final = Path(response["final_release"])
    payload = final / "auscultation" / "ausc" / "worker.py"
    before_mode = stat.S_IMODE(payload.stat().st_mode)
    process_env = dict(os.environ)
    process_env.pop("PYTHONDONTWRITEBYTECODE", None)
    process_env["PYTHONPATH"] = str(final)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util;"
                "s=importlib.util.spec_from_file_location('sealed_worker',%r);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)"
            ) % str(payload),
        ],
        check=False,
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr.decode()
    assert not list(final.rglob("__pycache__"))
    assert stat.S_IMODE(payload.stat().st_mode) == before_mode
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o555
        for path in [final] + [item for item in final.rglob("*") if item.is_dir()]
    )


def test_production_postverify_rejects_a_writable_release_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200", body=b"VALUE = 1\n"
    )
    response = remote_agent.dispatch_request(
        request,
        local_role="h200",
        agent_path=DEPLOYMENT_DIR / "remote_agent.py",
    )
    final = Path(response["final_release"])
    (final / "auscultation").chmod(0o755)

    with pytest.raises(remote_agent.AgentPolicyError, match="seal|writable|mode"):
        remote_agent._postverify_production(
            final=final,
            current=Path(response["current_link"]),
            local_role="h200",
            frozen_release_id=request["frozen_release_id"],
            frozen_payload_sha256=request["frozen_payload_sha256"],
            expected_manifest_sha256=request["expected_manifest_sha256"],
            compatibility=None,
            legacy=None,
        )


def test_existing_production_release_is_sealed_only_after_exact_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200", body=b"VALUE = 1\n"
    )
    paths = remote_agent.HOST_POLICIES["h200"]["production"]
    releases = Path(paths["releases_root"])
    releases.mkdir(parents=True, exist_ok=True)
    envelope = contract.verify_release(source)
    final = releases / envelope["release_id"]
    contract.write_release(source, envelope, final, target_role="h200")
    for directory in [final] + [item for item in final.rglob("*") if item.is_dir()]:
        directory.chmod(0o755)

    response = remote_agent.dispatch_request(
        request,
        local_role="h200",
        agent_path=DEPLOYMENT_DIR / "remote_agent.py",
    )
    assert response["created_release"] is False
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o555
        for path in [final] + [item for item in final.rglob("*") if item.is_dir()]
    )

    # A future idempotent attempt must fail exact verification, not hide or
    # delete an unexpected bytecode artifact before sealing.
    for directory in [final] + [item for item in final.rglob("*") if item.is_dir()]:
        directory.chmod(0o755)
    extra_parent = final / "auscultation" / "scripts" / "deployment" / "__pycache__"
    extra_parent.mkdir(parents=True)
    extra = extra_parent / "h200_env.cpython-312.pyc"
    extra.write_bytes(b"known unexpected bytecode")
    final.chmod(0o555)
    reservation2 = "c" * 64
    incoming2 = Path(paths["incoming_root"]) / (
        "incoming-%s-%s" % (envelope["release_id"], "d" * 32)
    )
    remote_agent.prepare_production_incoming(
        "h200",
        incoming_path=incoming2.as_posix(),
        reservation_id=reservation2,
        frozen_release_id=envelope["release_id"],
        frozen_payload_sha256=envelope["payload_sha256"],
    )
    shutil.copytree(source, incoming2, dirs_exist_ok=True)
    request2 = dict(
        request,
        incoming_path=incoming2.as_posix(),
        reservation_id=reservation2,
    )
    with pytest.raises((remote_agent.AgentPolicyError, contract.VerificationError)):
        remote_agent.dispatch_request(
            request2,
            local_role="h200",
            agent_path=DEPLOYMENT_DIR / "remote_agent.py",
        )
    assert extra.read_bytes() == b"known unexpected bytecode"


def test_upgrade_seals_exact_previous_current_before_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200", body=b"VALUE = 3\n"
    )
    paths = remote_agent.HOST_POLICIES["h200"]["production"]
    releases = Path(paths["releases_root"])
    current = Path(paths["current_link"])
    previous_source = _completed_release(
        tmp_path / "previous-exact", b"VALUE = 2\n"
    )
    previous_envelope = contract.verify_release(previous_source)
    previous = releases / previous_envelope["release_id"]
    contract.write_release(
        previous_source, previous_envelope, previous, target_role="h200"
    )
    current.symlink_to(os.path.relpath(previous, current.parent))
    assert any(
        stat.S_IMODE(path.stat().st_mode) & 0o222
        for path in [previous] + [item for item in previous.rglob("*") if item.is_dir()]
    )

    response = remote_agent.dispatch_request(
        request,
        local_role="h200",
        agent_path=DEPLOYMENT_DIR / "remote_agent.py",
    )

    final = Path(response["final_release"])
    assert current.resolve() == final.resolve()
    for release in (previous, final):
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o555
            for path in [release] + [item for item in release.rglob("*") if item.is_dir()]
        )


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="POSIX owner-mode test")
def test_owned_tree_cleanup_handles_a_sealed_release(tmp_path: Path) -> None:
    owned = tmp_path / "owned-sealed-release"
    nested = owned / "auscultation" / "ausc"
    nested.mkdir(parents=True)
    (nested / "worker.py").write_bytes(b"VALUE = 1\n")
    for directory in [nested, nested.parent, owned]:
        directory.chmod(0o555)

    remote_agent._remove_real_tree_path(owned)

    assert not owned.exists()


def test_production_directory_swap_to_symlink_is_rejected_during_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _completed_release(tmp_path / "source", b"VALUE = 1\n")
    envelope = contract.verify_release(source)
    releases = tmp_path / ".releases"
    final = releases / envelope["release_id"]
    contract.write_release(source, envelope, final, target_role="h200")
    original_seal = remote_agent._seal_release_directory_descriptor
    raced = False

    def swap_payload_then_seal(descriptor: int) -> None:
        nonlocal raced
        if not raced:
            raced = True
            os.rename(
                "auscultation", ".held-auscultation",
                src_dir_fd=descriptor, dst_dir_fd=descriptor,
            )
            os.symlink(
                ".held-auscultation", "auscultation", dir_fd=descriptor
            )
            os.fsync(descriptor)
        original_seal(descriptor)

    monkeypatch.setattr(
        remote_agent, "_seal_release_directory_descriptor", swap_payload_then_seal
    )
    with pytest.raises(remote_agent.AgentPolicyError, match="symlink|unsafe|changed"):
        remote_agent._seal_verified_production_release(final)

    assert raced is True
    assert (final / "auscultation").is_symlink()
    assert (
        final / ".held-auscultation" / "ausc" / "worker.py"
    ).read_bytes() == b"VALUE = 1\n"


def test_production_seal_rejects_an_unexpected_empty_directory_without_mutation(
    tmp_path: Path,
) -> None:
    source = _completed_release(tmp_path / "source", b"VALUE = 1\n")
    envelope = contract.verify_release(source)
    final = tmp_path / ".releases" / envelope["release_id"]
    contract.write_release(source, envelope, final, target_role="h200")
    unexpected = final / "auscultation" / "unexpected-empty"
    unexpected.mkdir()
    before_modes = {
        path.relative_to(final).as_posix(): stat.S_IMODE(path.stat().st_mode)
        for path in [final] + [item for item in final.rglob("*") if item.is_dir()]
    }

    with pytest.raises(remote_agent.AgentPolicyError, match="director.*exact|extra"):
        remote_agent._seal_verified_production_release(final)

    assert unexpected.is_dir()
    assert {
        path.relative_to(final).as_posix(): stat.S_IMODE(path.stat().st_mode)
        for path in [final] + [item for item in final.rglob("*") if item.is_dir()]
    } == before_modes


def test_production_plan_requires_frozen_double_confirmation_and_separate_capability(
    tmp_path: Path,
) -> None:
    release = _completed_release(tmp_path)
    envelope = contract.verify_release(release)
    with pytest.raises(deploy_remote.PolicyError, match="confirm-release"):
        deploy_remote.build_production_ssh_plan(
            release, target_role="h200", host_alias="nebius-h200",
            confirm_release=None,
            confirm_payload_sha256=envelope["payload_sha256"],
        )
    with pytest.raises(deploy_remote.PolicyError, match="confirm-payload"):
        deploy_remote.build_production_ssh_plan(
            release, target_role="h200", host_alias="nebius-h200",
            confirm_release=envelope["release_id"],
            confirm_payload_sha256="0" * 64,
        )

    plan = deploy_remote.build_production_ssh_plan(
        release, target_role="h200", host_alias="nebius-h200",
        confirm_release=envelope["release_id"],
        confirm_payload_sha256=envelope["payload_sha256"],
        nonce="1" * 32, reservation_id="2" * 64,
    )
    assert plan.host_alias == "nebius-h200"
    assert plan.target_role == "h200"
    assert [step.name for step in plan.steps] == [
        "upload_contract", "upload_agent", "prepare_production",
        "stage_release", "activate_production",
    ]
    prepare = json.loads(plan.steps[2].stdin)
    activate = json.loads(plan.steps[4].stdin)
    assert prepare["schema"] == "wam_remote_production_prepare_request_v1"
    assert activate["schema"] == "wam_remote_production_activate_request_v1"
    assert activate["capability"] == "task8_frozen_release_activation_v1"
    assert activate["frozen_release_id"] == envelope["release_id"]
    assert activate["frozen_payload_sha256"] == envelope["payload_sha256"]
    assert activate["operation"] != "install_release"

    with pytest.raises(deploy_remote.PolicyError, match="connection"):
        deploy_remote.build_production_ssh_plan(
            release, target_role="h200", host_alias="h200",
            confirm_release=envelope["release_id"],
            confirm_payload_sha256=envelope["payload_sha256"],
        )


def test_h200_production_install_activate_postverify_and_reconcile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200"
    )
    result = remote_agent.dispatch_request(
        request, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
    )
    assert result["schema"] == "wam_remote_production_activate_response_v1"
    assert result["release_id"] == request["frozen_release_id"]
    assert result["payload_sha256"] == request["frozen_payload_sha256"]
    current = Path(remote_agent.HOST_POLICIES["h200"]["production"]["current_link"])
    assert current.is_symlink()
    assert current.resolve() == Path(result["final_release"]).resolve()

    reconciled = remote_agent.dispatch_request(
        reconcile, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
    )
    assert reconciled["schema"] == "wam_remote_production_reconcile_response_v1"
    assert reconciled["status"] == "promoted"
    assert reconciled["current_target"] == os.readlink(current)


def test_h200_production_postverify_failure_cas_rolls_back_previous_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _production_policy(tmp_path / "remote", "h200")
    monkeypatch.setitem(remote_agent.HOST_POLICIES, "h200", policy)
    old = _completed_release(tmp_path / "old", b"old\n")
    old_envelope = contract.verify_release(old)
    old_final = Path(policy["production"]["releases_root"]) / old_envelope["release_id"]
    old_final.parent.mkdir(parents=True)
    contract.write_release(old, old_envelope, old_final, target_role="h200")
    current = Path(policy["production"]["current_link"])
    current.parent.mkdir(parents=True, exist_ok=True)
    current.symlink_to(os.path.relpath(old_final, current.parent))

    _source, request, _reconcile = _stage_production_request(
        tmp_path / "new", monkeypatch, "h200", body=b"new\n"
    )
    # Reuse the policy containing the old pointer after the staging helper patched it.
    monkeypatch.setitem(remote_agent.HOST_POLICIES, "h200", policy)
    incoming = Path(policy["production"]["incoming_root"]) / Path(request["incoming_path"]).name
    remote_agent.prepare_production_incoming(
        "h200", incoming_path=incoming.as_posix(),
        reservation_id=request["reservation_id"],
        frozen_release_id=request["frozen_release_id"],
        frozen_payload_sha256=request["frozen_payload_sha256"],
    )
    shutil.copytree(_source, incoming, dirs_exist_ok=True)
    request["incoming_path"] = incoming.as_posix()
    monkeypatch.setattr(
        remote_agent, "_postverify_production",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            remote_agent.AgentPolicyError("controlled postverify failure")
        ),
    )
    with pytest.raises(remote_agent.AgentPolicyError, match="controlled"):
        remote_agent.dispatch_request(
            request, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )
    assert current.resolve() == old_final.resolve()


def test_linbox_production_migrates_real_directory_without_deleting_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, request, reconcile = _stage_production_request(
        tmp_path, monkeypatch, "linbox"
    )
    base = Path(remote_agent.HOST_POLICIES["linbox"]["production"]["current_link"]).parent
    real = base / "auscultation"
    (real / "local").mkdir(parents=True)
    (real / "local" / "keep.txt").write_bytes(b"legacy-content")

    result = remote_agent.dispatch_request(
        request, local_role="linbox", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
    )
    legacy = Path(result["legacy_path"])
    assert legacy.name.startswith("auscultation.legacy-")
    assert legacy.is_dir() and not legacy.is_symlink()
    assert (legacy / "local" / "keep.txt").read_bytes() == b"legacy-content"
    assert real.is_symlink()
    assert os.readlink(real) == ".wam-current/auscultation"
    assert (real / "ausc" / "worker.py").read_bytes() == b"production\n"
    assert contract.verify_release(Path(result["final_release"]))["release_id"] == request[
        "frozen_release_id"
    ]
    assert remote_agent.dispatch_request(
        reconcile, local_role="linbox", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
    )["status"] == "promoted"
    assert source.exists()


def test_linbox_failed_postverify_restores_real_directory_and_never_overwrites_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "linbox"
    )
    base = Path(remote_agent.HOST_POLICIES["linbox"]["production"]["current_link"]).parent
    real = base / "auscultation"
    real.mkdir(parents=True)
    (real / "keep.txt").write_bytes(b"keep")
    occupied = base / (
        "auscultation.legacy-20000101T000000Z-" + "0" * 16
    )
    occupied.mkdir()
    (occupied / "do-not-touch").write_bytes(b"occupied")
    monkeypatch.setattr(
        remote_agent, "_postverify_production",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            remote_agent.AgentPolicyError("controlled postverify failure")
        ),
    )
    with pytest.raises(remote_agent.AgentPolicyError, match="controlled"):
        remote_agent.dispatch_request(
            request, local_role="linbox", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )
    assert real.is_dir() and not real.is_symlink()
    assert (real / "keep.txt").read_bytes() == b"keep"
    assert (occupied / "do-not-touch").read_bytes() == b"occupied"


def test_production_prepare_and_activate_reject_canary_schema_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200"
    )
    request["schema"] = "wam_remote_install_request_v1"
    request["operation"] = "install_release"
    with pytest.raises(remote_agent.AgentPolicyError):
        remote_agent.dispatch_request(
            request, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )


def test_execute_production_plan_reconciles_lost_activation_response(
    tmp_path: Path,
) -> None:
    release = _completed_release(tmp_path)
    envelope = contract.verify_release(release)
    plan = deploy_remote.build_production_ssh_plan(
        release, target_role="h200", host_alias="nebius-h200",
        confirm_release=envelope["release_id"],
        confirm_payload_sha256=envelope["payload_sha256"],
        nonce="3" * 32, reservation_id="4" * 64,
    )
    operations = []

    def runner(argv, **kwargs):
        request = json.loads(kwargs["input"]) if kwargs.get("input") else None
        operation = request["operation"] if request else None
        if operation:
            operations.append(operation)
        if operation == "prepare_production_incoming":
            stdout = contract.canonical_json_bytes({
                "frozen_payload_sha256": plan.expected_payload_sha256,
                "frozen_release_id": plan.expected_release_id,
                "incoming_path": plan.incoming_path,
                "reservation_id": plan.reservation_id,
                "schema": "wam_remote_production_prepare_response_v1",
                "target_role": "h200",
            })
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        if operation == "activate_production_release":
            return SimpleNamespace(
                returncode=255, stdout=b"", stderr=b"lost activation response",
            )
        if operation == "reconcile_production_release":
            stdout = contract.canonical_json_bytes({
                "compatibility_link": None,
                "current_target": ".releases/" + plan.expected_release_id,
                "final_release": plan.releases_root + "/" + plan.expected_release_id,
                "legacy_path": None,
                "schema": "wam_remote_production_reconcile_response_v1",
                "status": "promoted",
                "target_role": "h200",
            })
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    evidence = deploy_remote.execute_production_ssh_plan(plan, runner=runner)
    assert evidence["schema"] == "wam_ssh_production_evidence_v1"
    assert evidence["status"] == "reconciled_promoted"
    assert operations == [
        "prepare_production_incoming", "activate_production_release",
        "reconcile_production_release",
    ]


def test_linbox_legacy_inventory_rejects_symlink_without_moving_real_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "linbox"
    )
    base = Path(remote_agent.HOST_POLICIES["linbox"]["production"]["current_link"]).parent
    real = base / "auscultation"
    real.mkdir(parents=True)
    outside = base / "outside"
    outside.write_bytes(b"outside")
    (real / "unsafe").symlink_to(outside)

    with pytest.raises(remote_agent.AgentPolicyError, match="symlink|non-regular"):
        remote_agent.dispatch_request(
            request, local_role="linbox", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )
    assert real.is_dir() and not real.is_symlink()
    assert (real / "unsafe").is_symlink()
    assert outside.read_bytes() == b"outside"


def test_production_reconcile_waits_for_activation_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200"
    )
    entered = threading.Event()
    release = threading.Event()
    reconcile_started = threading.Event()
    reconcile_finished = threading.Event()
    original = remote_agent._postverify_production
    results = {}

    def controlled(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    def activate():
        results["activate"] = remote_agent.dispatch_request(
            request, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )

    def reconcile_worker():
        reconcile_started.set()
        results["reconcile"] = remote_agent.dispatch_request(
            reconcile, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )
        reconcile_finished.set()

    monkeypatch.setattr(remote_agent, "_postverify_production", controlled)
    a = threading.Thread(target=activate)
    b = threading.Thread(target=reconcile_worker)
    a.start()
    assert entered.wait(5)
    b.start()
    assert reconcile_started.wait(5)
    assert not reconcile_finished.wait(0.2)
    release.set()
    a.join(5); b.join(5)
    assert not a.is_alive() and not b.is_alive()
    assert results["reconcile"]["status"] == "promoted"


def test_sigkill_before_atomic_transaction_publish_leaves_no_blocking_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200"
    )
    paths = remote_agent.HOST_POLICIES["h200"]["production"]
    base = Path(paths["current_link"]).parent
    journal = remote_agent._production_transaction_path(
        base, request["reservation_id"]
    )

    child = os.fork()
    if child == 0:
        original_open = remote_agent.os.open
        original_publish = remote_agent._rename_noreplace_between_at

        def kill_legacy_seal(name, flags, *args, **kwargs):
            if str(name).endswith(".seal.json"):
                os.kill(os.getpid(), 9)
            return original_open(name, flags, *args, **kwargs)

        def kill_atomic_publish(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
            *,
            expected_identity,
        ):
            if destination_name == journal.name:
                os.kill(os.getpid(), 9)
            return original_publish(
                source_parent_descriptor,
                source_name,
                destination_parent_descriptor,
                destination_name,
                expected_identity=expected_identity,
            )

        remote_agent.os.open = kill_legacy_seal
        remote_agent._rename_noreplace_between_at = kill_atomic_publish
        remote_agent.dispatch_request(
            request,
            local_role="h200",
            agent_path=DEPLOYMENT_DIR / "remote_agent.py",
        )
        os._exit(0)

    _pid, status = os.waitpid(child, 0)
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == 9
    assert not os.path.lexists(str(journal))

    recovered = remote_agent.dispatch_request(
        reconcile,
        local_role="h200",
        agent_path=DEPLOYMENT_DIR / "remote_agent.py",
    )
    assert recovered["status"] == "cleaned"
    prepared_prefix = ".%s.activate-%s-" % (
        Path(paths["current_link"]).name, request["reservation_id"]
    )
    assert not any(name.startswith(prepared_prefix) for name in os.listdir(base))
    assert not any(
        name.startswith(journal.name + ".pending-")
        for name in os.listdir(base)
    )


@pytest.mark.parametrize("partial", [b"", b'{"schema":'])
def test_sigkill_during_pending_write_is_owned_and_reconcile_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial: bytes,
) -> None:
    _source, request, reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200"
    )
    paths = remote_agent.HOST_POLICIES["h200"]["production"]
    base = Path(paths["current_link"]).parent
    journal = remote_agent._production_transaction_path(
        base, request["reservation_id"]
    )
    sentinel = base / "third-party-sentinel"
    sentinel.write_bytes(b"preserve")

    child = os.fork()
    if child == 0:
        original_prepare = remote_agent._prepare_production_pointer
        original_fdopen = remote_agent.os.fdopen
        armed = {"value": False}

        def arm_after_pointer(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            armed["value"] = True
            return result

        def kill_during_pending_write(descriptor, *args, **kwargs):
            if armed["value"]:
                if partial:
                    os.write(descriptor, partial)
                    os.fsync(descriptor)
                os.kill(os.getpid(), 9)
            return original_fdopen(descriptor, *args, **kwargs)

        remote_agent._prepare_production_pointer = arm_after_pointer
        remote_agent.os.fdopen = kill_during_pending_write
        remote_agent.dispatch_request(
            request,
            local_role="h200",
            agent_path=DEPLOYMENT_DIR / "remote_agent.py",
        )
        os._exit(0)

    _pid, status = os.waitpid(child, 0)
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == 9
    assert not os.path.lexists(str(journal))

    recovered = remote_agent.dispatch_request(
        reconcile,
        local_role="h200",
        agent_path=DEPLOYMENT_DIR / "remote_agent.py",
    )
    assert recovered["status"] == "cleaned"
    assert sentinel.read_bytes() == b"preserve"
    assert not any(
        request["reservation_id"] in name
        and (
            ".pending-" in name
            or name.startswith(".%s.activate-" % Path(paths["current_link"]).name)
        )
        for name in os.listdir(base)
    )


def test_cli_production_mode_requires_named_confirmation_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    release = _completed_release(tmp_path)
    envelope = contract.verify_release(release)
    with pytest.raises(deploy_remote.PolicyError, match="confirm-release"):
        deploy_remote.main([
            "--source-release", str(release), "--source-role", "mac",
            "--target-role", "h200", "--host", "nebius-h200", "--production",
        ])
    assert deploy_remote.main([
        "--source-release", str(release), "--source-role", "mac",
        "--target-role", "h200", "--host", "nebius-h200", "--production",
        "--confirm-release", envelope["release_id"],
        "--confirm-payload-sha256", envelope["payload_sha256"],
    ]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["production"] is True
    assert printed["frozen_release_id"] == envelope["release_id"]
    assert printed["frozen_payload_sha256"] == envelope["payload_sha256"]


def test_cli_perception_data_kind_dry_run_prints_exact_pointer_free_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "mac-perception" / "releases"
    source = _perception_release(
        source_root,
        artifact_kind="perception_labels_holdout",
        source_role="mac",
    )
    monkeypatch.setattr(deploy_remote, "_MAC_PERCEPTION_RELEASES_ROOT", source_root)
    envelope = contract.verify_release(source)

    assert deploy_remote.main([
        "--source-release", str(source),
        "--source-role", "mac",
        "--target-role", "h200",
        "--host", "nebius-h200",
        "--data-kind", "perception_labels_holdout",
    ]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["data_kind"] == "perception_labels_holdout"
    assert printed["release_id"] == envelope["release_id"]
    assert printed["payload_sha256"] == envelope["payload_sha256"]
    assert printed["manifest_sha256"] == hashlib.sha256(
        (source / contract.MANIFEST_NAME).read_bytes()
    ).hexdigest()
    assert printed["steps"] == [
        "upload_contract",
        "upload_agent",
        "prepare_perception_incoming",
        "stage_perception_artifact",
        "install_perception_artifact",
    ]
    assert "current" not in json.dumps(printed)


def test_cli_perception_execute_requires_both_exact_confirmations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "mac-perception" / "releases"
    source = _perception_release(
        source_root,
        artifact_kind="perception_geometry_source",
        source_role="mac",
    )
    monkeypatch.setattr(deploy_remote, "_MAC_PERCEPTION_RELEASES_ROOT", source_root)
    envelope = contract.verify_release(source)
    base = [
        "--source-release", str(source),
        "--source-role", "mac",
        "--target-role", "h200",
        "--host", "nebius-h200",
        "--data-kind", "perception_geometry_source",
        "--execute",
    ]
    calls = []

    def execute(plan):
        calls.append(plan)
        return {"schema": "test_perception_cli_execution_v1"}

    monkeypatch.setattr(deploy_remote, "execute_data_ssh_plan", execute)
    with pytest.raises(deploy_remote.PolicyError, match="confirm-release"):
        deploy_remote.main(base)
    with pytest.raises(deploy_remote.PolicyError, match="confirm-payload-sha256"):
        deploy_remote.main(base + [
            "--confirm-release", envelope["release_id"],
            "--confirm-payload-sha256", "0" * 64,
        ])
    assert calls == []

    assert deploy_remote.main(base + [
        "--confirm-release", envelope["release_id"],
        "--confirm-payload-sha256", envelope["payload_sha256"],
    ]) == 0
    assert len(calls) == 1
    assert calls[0].data_kind == "perception_geometry_source"
    assert json.loads(capsys.readouterr().out) == {
        "schema": "test_perception_cli_execution_v1"
    }


def _transaction_pointer_fields(current: Path, reservation_id: str) -> dict:
    return {
        "activated_pointer_identity": {
            "device": 1,
            "inode": 2,
        },
        "prepared_pointer_name": ".%s.activate-%s-1-2" % (
            current.name, reservation_id
        ),
        "previous_pointer_identity": None,
    }


def test_linbox_reconcile_restores_legacy_after_interrupted_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, reconcile = _stage_production_request(
        tmp_path, monkeypatch, "linbox"
    )
    paths = remote_agent.HOST_POLICIES["linbox"]["production"]
    current = Path(paths["current_link"])
    base = current.parent
    real = base / "auscultation"
    real.mkdir(parents=True)
    (real / "keep.txt").write_bytes(b"survives-crash")
    inventory = remote_agent._inventory_real_tree(real)
    legacy_info = real.stat()
    legacy = base / (
        "auscultation.legacy-20000101T000000Z-%s" % inventory[:16]
    )
    activated = os.path.relpath(
        str(Path(paths["releases_root"]) / request["frozen_release_id"]), str(base)
    )
    transaction = {
        **_transaction_pointer_fields(
            Path(paths["current_link"]), request["reservation_id"]
        ),
        "activated_target": activated,
        "compatibility_preexisting": False,
        "frozen_payload_sha256": request["frozen_payload_sha256"],
        "frozen_release_id": request["frozen_release_id"],
        "incoming_path": request["incoming_path"],
        "expected_manifest_sha256": request["expected_manifest_sha256"],
        "legacy_device": legacy_info.st_dev,
        "legacy_inode": legacy_info.st_ino,
        "legacy_inventory_sha256": inventory,
        "legacy_path": legacy.as_posix(),
        "migrate_real_directory": True,
        "previous_current_target": None,
        "request_identity_sha256": remote_agent._production_request_identity(request),
        "reservation_id": request["reservation_id"],
        "schema": "wam_production_transaction_v2",
        "target_role": "linbox",
    }
    journal = remote_agent._production_transaction_path(
        base, request["reservation_id"]
    )
    state = base / ".owned-transaction-state"
    state.mkdir()
    state_descriptor = remote_agent._open_real_directory(state, create=False)
    try:
        remote_agent._create_production_transaction(
            journal, transaction, owned_state_descriptor=state_descriptor
        )
    finally:
        os.close(state_descriptor)
    contract._rename_noreplace(real, legacy)
    assert legacy.is_dir() and not real.exists()

    result = remote_agent.dispatch_request(
        reconcile, local_role="linbox", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
    )
    assert result["status"] == "cleaned"
    assert real.is_dir() and not real.is_symlink()
    assert (real / "keep.txt").read_bytes() == b"survives-crash"
    assert not legacy.exists() and not journal.exists()


def test_linbox_exact_legacy_name_collision_preserves_both_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "linbox"
    )
    paths = remote_agent.HOST_POLICIES["linbox"]["production"]
    base = Path(paths["current_link"]).parent
    real = base / "auscultation"
    real.mkdir(parents=True)
    (real / "keep.txt").write_bytes(b"original")
    inventory = remote_agent._inventory_real_tree(real)
    occupied = base / (
        "auscultation.legacy-20000101T000000Z-%s" % inventory[:16]
    )
    occupied.mkdir()
    (occupied / "occupied.txt").write_bytes(b"never-overwrite")

    class FixedDateTime:
        @classmethod
        def now(cls, _zone):
            from datetime import datetime as real_datetime
            return real_datetime(2000, 1, 1, tzinfo=remote_agent.timezone.utc)

    monkeypatch.setattr(remote_agent, "datetime", FixedDateTime)
    with pytest.raises(contract.ConflictError):
        remote_agent.dispatch_request(
            request, local_role="linbox", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )
    assert (real / "keep.txt").read_bytes() == b"original"
    assert (occupied / "occupied.txt").read_bytes() == b"never-overwrite"
    assert not remote_agent._production_transaction_path(
        base, request["reservation_id"]
    ).exists()
    assert not Path(request["incoming_path"]).exists()


def test_production_reconcile_removes_only_its_owned_staging_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200"
    )
    releases = Path(remote_agent.HOST_POLICIES["h200"]["production"]["releases_root"])
    staging = releases / (
        ".staging-%s-%s" % (request["frozen_release_id"], request["reservation_id"])
    )
    staging.mkdir(parents=True)
    (staging / "partial").write_bytes(b"partial")
    unrelated = releases / ".staging-unrelated"
    unrelated.mkdir()
    (unrelated / "preserve").write_bytes(b"preserve")

    result = remote_agent.dispatch_request(
        reconcile, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
    )
    assert result["status"] == "cleaned"
    assert not staging.exists()
    assert (unrelated / "preserve").read_bytes() == b"preserve"


def test_reconcile_rejects_forged_journal_previous_target_outside_managed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200"
    )
    paths = remote_agent.HOST_POLICIES["h200"]["production"]
    current = Path(paths["current_link"])
    current.parent.mkdir(parents=True, exist_ok=True)
    activated = os.path.relpath(
        str(Path(paths["releases_root"]) / request["frozen_release_id"]),
        str(current.parent),
    )
    current.symlink_to(activated)
    transaction = {
        **_transaction_pointer_fields(current, request["reservation_id"]),
        "activated_target": activated,
        "compatibility_preexisting": False,
        "frozen_payload_sha256": request["frozen_payload_sha256"],
        "frozen_release_id": request["frozen_release_id"],
        "incoming_path": request["incoming_path"],
        "expected_manifest_sha256": request["expected_manifest_sha256"],
        "legacy_device": None,
        "legacy_inode": None,
        "legacy_inventory_sha256": None,
        "legacy_path": None,
        "migrate_real_directory": False,
        "previous_current_target": "/tmp/outside-release",
        "request_identity_sha256": remote_agent._production_request_identity(request),
        "reservation_id": request["reservation_id"],
        "schema": "wam_production_transaction_v2",
        "target_role": "h200",
    }
    journal = remote_agent._production_transaction_path(
        current.parent, request["reservation_id"]
    )
    state = current.parent / ".owned-transaction-state"
    state.mkdir()
    state_descriptor = remote_agent._open_real_directory(state, create=False)
    try:
        remote_agent._create_production_transaction(
            journal, transaction, owned_state_descriptor=state_descriptor
        )
    finally:
        os.close(state_descriptor)

    with pytest.raises(remote_agent.AgentPolicyError, match="transaction|managed|outside"):
        remote_agent.dispatch_request(
            reconcile, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )
    assert current.is_symlink() and os.readlink(current) == activated


def test_activation_pointer_cas_does_not_overwrite_concurrent_third_party_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200"
    )
    paths = remote_agent.HOST_POLICIES["h200"]["production"]
    current = Path(paths["current_link"])
    original = remote_agent._set_pointer

    def race(target_path, target, *args, **kwargs):
        target_path.symlink_to("third-party-new-pointer")
        return original(target_path, target, *args, **kwargs)

    monkeypatch.setattr(remote_agent, "_set_pointer", race)
    with pytest.raises(remote_agent.AgentPolicyError, match="compare-and-swap|rollback"):
        remote_agent.dispatch_request(
            request, local_role="h200", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )
    assert current.is_symlink()
    assert os.readlink(current) == "third-party-new-pointer"


def test_rollback_cas_preserves_same_target_replacement_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200"
    )
    paths = remote_agent.HOST_POLICIES["h200"]["production"]
    current = Path(paths["current_link"])
    third_party_identity = {}

    def replace_with_same_target_then_fail(*, current: Path, **_kwargs) -> None:
        target = os.readlink(str(current))
        current.unlink()
        current.symlink_to(target)
        info = current.lstat()
        third_party_identity["value"] = (info.st_dev, info.st_ino, info.st_ctime_ns)
        raise remote_agent.AgentPolicyError("injected post-activation failure")

    monkeypatch.setattr(
        remote_agent, "_postverify_production", replace_with_same_target_then_fail
    )
    with pytest.raises(remote_agent.AgentPolicyError, match="rollback"):
        remote_agent.dispatch_request(
            request,
            local_role="h200",
            agent_path=DEPLOYMENT_DIR / "remote_agent.py",
        )

    info = current.lstat()
    assert current.is_symlink()
    assert (info.st_dev, info.st_ino, info.st_ctime_ns) == third_party_identity["value"]


def test_linbox_legacy_rename_rejects_inventory_name_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "linbox"
    )
    paths = remote_agent.HOST_POLICIES["linbox"]["production"]
    base = Path(paths["current_link"]).parent
    real = base / "auscultation"
    real.mkdir(parents=True)
    (real / "keep.txt").write_bytes(b"original-inventory")
    original_rename = remote_agent._rename_noreplace_at

    def swap_then_rename(
        parent_descriptor, source_name, destination_name, *, expected_identity,
    ):
        if source_name != "auscultation":
            return original_rename(
                parent_descriptor, source_name, destination_name,
                expected_identity=expected_identity,
            )
        os.rename(
            source_name, "auscultation.swapped-original",
            src_dir_fd=parent_descriptor, dst_dir_fd=parent_descriptor,
        )
        os.mkdir(source_name, dir_fd=parent_descriptor)
        descriptor = os.open(
            source_name + "/replacement.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600, dir_fd=parent_descriptor,
        )
        os.write(descriptor, b"replacement")
        os.close(descriptor)
        return original_rename(
            parent_descriptor, source_name, destination_name,
            expected_identity=expected_identity,
        )

    monkeypatch.setattr(remote_agent, "_rename_noreplace_at", swap_then_rename)
    with pytest.raises(remote_agent.AgentPolicyError, match="changed|inode|rollback"):
        remote_agent.dispatch_request(
            request, local_role="linbox", agent_path=DEPLOYMENT_DIR / "remote_agent.py"
        )
    assert (base / "auscultation.swapped-original" / "keep.txt").read_bytes() == (
        b"original-inventory"
    )
    assert (real / "replacement.txt").read_bytes() == b"replacement"


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("reservation_id", "c" * 64),
        ("frozen_release_id", "code-" + "d" * 20),
        ("incoming_path", "/tmp/outside-incoming"),
        ("activated_target", "../../outside-release"),
        ("request_identity_sha256", "e" * 64),
        ("previous_current_target", ".releases/code-" + "f" * 20),
    ],
)
def test_transaction_fields_are_strictly_bound_to_request_and_managed_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, bad: str,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200"
    )
    paths = remote_agent.HOST_POLICIES["h200"]["production"]
    base = Path(paths["current_link"]).parent
    transaction = {
        **_transaction_pointer_fields(
            Path(paths["current_link"]), request["reservation_id"]
        ),
        "activated_target": os.path.relpath(
            str(Path(paths["releases_root"]) / request["frozen_release_id"]), str(base)
        ),
        "compatibility_preexisting": False,
        "expected_manifest_sha256": request["expected_manifest_sha256"],
        "frozen_payload_sha256": request["frozen_payload_sha256"],
        "frozen_release_id": request["frozen_release_id"],
        "incoming_path": request["incoming_path"],
        "legacy_device": None,
        "legacy_inode": None,
        "legacy_inventory_sha256": None,
        "legacy_path": None,
        "migrate_real_directory": False,
        "previous_current_target": None,
        "request_identity_sha256": remote_agent._production_request_identity(request),
        "reservation_id": request["reservation_id"],
        "schema": "wam_production_transaction_v2",
        "target_role": "h200",
    }
    transaction[field] = bad
    with pytest.raises(remote_agent.AgentPolicyError):
        remote_agent._validate_production_transaction_for_request(
            transaction, request, local_role="h200"
        )


@pytest.mark.parametrize(
    "mutation",
    ["absolute_legacy", "hash_name_mismatch", "missing_inode", "preexisting_and_migrate"],
)
def test_linbox_transaction_legacy_fields_must_be_canonical_complete_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "linbox"
    )
    paths = remote_agent.HOST_POLICIES["linbox"]["production"]
    base = Path(paths["current_link"]).parent
    digest = "1" * 64
    transaction = {
        **_transaction_pointer_fields(
            Path(paths["current_link"]), request["reservation_id"]
        ),
        "activated_target": os.path.relpath(
            str(Path(paths["releases_root"]) / request["frozen_release_id"]), str(base)
        ),
        "compatibility_preexisting": False,
        "expected_manifest_sha256": request["expected_manifest_sha256"],
        "frozen_payload_sha256": request["frozen_payload_sha256"],
        "frozen_release_id": request["frozen_release_id"],
        "incoming_path": request["incoming_path"],
        "legacy_device": 1,
        "legacy_inode": 2,
        "legacy_inventory_sha256": digest,
        "legacy_path": (
            base / ("auscultation.legacy-20000101T000000Z-%s" % digest[:16])
        ).as_posix(),
        "migrate_real_directory": True,
        "previous_current_target": None,
        "request_identity_sha256": remote_agent._production_request_identity(request),
        "reservation_id": request["reservation_id"],
        "schema": "wam_production_transaction_v2",
        "target_role": "linbox",
    }
    if mutation == "absolute_legacy":
        transaction["legacy_path"] = "/tmp/auscultation.legacy-20000101T000000Z-" + digest[:16]
    elif mutation == "hash_name_mismatch":
        transaction["legacy_inventory_sha256"] = "2" * 64
    elif mutation == "missing_inode":
        transaction["legacy_inode"] = None
    else:
        transaction["compatibility_preexisting"] = True
    with pytest.raises(remote_agent.AgentPolicyError):
        remote_agent._validate_production_transaction_for_request(
            transaction, request, local_role="linbox"
        )


def test_existing_pointer_exchange_cas_restores_concurrent_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "pointer-cas"
    base.mkdir()
    current = base / "current"
    current.symlink_to("old-managed-target")
    parent = remote_agent._open_real_directory(base, create=False)
    old_target, old_identity = remote_agent._read_pointer_at(parent, current.name)
    original_exchange = remote_agent._exchange_at
    raced = {"done": False}

    def exchange_with_race(descriptor, first, second, *args, **kwargs):
        if not raced["done"]:
            raced["done"] = True
            os.unlink(second, dir_fd=descriptor)
            os.symlink("third-party-pointer", second, dir_fd=descriptor)
        return original_exchange(descriptor, first, second, *args, **kwargs)

    monkeypatch.setattr(remote_agent, "_exchange_at", exchange_with_race)
    try:
        with pytest.raises(remote_agent.AgentPolicyError, match="compare-and-swap"):
            remote_agent._set_pointer(
                current,
                "new-managed-target",
                expected_previous=old_target,
                expected_identity=old_identity,
                parent_descriptor=parent,
            )
    finally:
        os.close(parent)
    assert current.is_symlink()
    assert os.readlink(current) == "third-party-pointer"


def test_existing_pointer_preexchange_identity_race_never_touches_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, request, _reconcile = _stage_production_request(
        tmp_path, monkeypatch, "h200"
    )
    paths = remote_agent.HOST_POLICIES["h200"]["production"]
    releases = Path(paths["releases_root"])
    current = Path(paths["current_link"])
    previous_source = _completed_release(tmp_path / "previous-production", b"previous\n")
    previous_envelope = contract.verify_release(previous_source)
    previous = releases / previous_envelope["release_id"]
    contract.write_release(
        previous_source, previous_envelope, previous, target_role="h200"
    )
    current.symlink_to(os.path.relpath(previous, current.parent))

    original_exchange = remote_agent._exchange_at
    replacement = {}
    syscall_calls = {"count": 0}

    def replace_at_exchange_entry(descriptor, first, second, *args, **kwargs):
        if not replacement:
            prior_target = os.readlink(second, dir_fd=descriptor)
            os.unlink(second, dir_fd=descriptor)
            os.symlink(prior_target, second, dir_fd=descriptor)
            replacement["identity"] = remote_agent._path_identity_at(
                descriptor, second
            )
            replacement["target"] = prior_target
        return original_exchange(descriptor, first, second, *args, **kwargs)

    original_syscall = getattr(remote_agent, "_exchange_names_at", None)

    def count_syscall(*args, **kwargs):
        syscall_calls["count"] += 1
        if original_syscall is None:
            raise AssertionError("missing exchange syscall helper")
        return original_syscall(*args, **kwargs)

    monkeypatch.setattr(remote_agent, "_exchange_at", replace_at_exchange_entry)
    monkeypatch.setattr(
        remote_agent, "_exchange_names_at", count_syscall, raising=False
    )
    with pytest.raises(remote_agent.AgentPolicyError, match="rollback|compare-and-swap"):
        remote_agent.dispatch_request(
            request,
            local_role="h200",
            agent_path=DEPLOYMENT_DIR / "remote_agent.py",
        )

    parent = remote_agent._open_real_directory(current.parent, create=False)
    try:
        final_target, final_identity = remote_agent._read_pointer_at(
            parent, current.name
        )
    finally:
        os.close(parent)
    assert syscall_calls["count"] == 0
    assert final_target == replacement["target"]
    assert final_identity == replacement["identity"]


def test_canary_inspect_forced_rollback_and_cleanup_are_root_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = tmp_path / "canary"
    old = deploy_remote.deploy_local(
        _completed_release(tmp_path / "old-canary", b"old\n"), canary, target_role="h200"
    )
    new = deploy_remote.deploy_local(
        _completed_release(tmp_path / "new-canary", b"new\n"), canary, target_role="h200"
    )
    paths = {
        "current_link": (canary / "current").as_posix(),
        "incoming_root": (canary / "incoming").as_posix(),
        "releases_root": (canary / "releases").as_posix(),
    }
    monkeypatch.setitem(
        remote_agent.HOST_POLICIES, "h200",
        {"home": tmp_path.as_posix(), "canary": paths, "production": paths, "data": {}},
    )
    previous = Path(os.path.relpath(old.final_release, canary))
    run_target = Path(os.path.relpath(new.final_release, canary)).as_posix()
    result = remote_agent.force_canary_rollback(
        "h200", previous.as_posix(), expected_current_target=run_target
    )
    assert Path(paths["current_link"]).resolve() == old.final_release.resolve()
    assert result["rolled_back_to"] == previous.as_posix()
    inspected = remote_agent.inspect_canary("h200")
    assert inspected["current_target"] == previous.as_posix()
    assert len(inspected["release_ids"]) == 2

    outside = tmp_path / "outside"
    outside.write_text("preserve")
    reserved = canary / "incoming" / (
        "incoming-%s-%s" % (new.final_release.name, "e" * 32)
    )
    reservation = remote_agent.prepare_incoming(
        "h200", reserved.as_posix(), "4" * 64
    )
    cleaned = remote_agent.cleanup_canary(
        "h200",
        incoming_path=reserved.as_posix(),
        reservation_id=reservation["reservation_id"],
        created_release_id=new.final_release.name,
        pre_current_target=previous.as_posix(),
        pre_release_ids=(old.final_release.name,),
    )
    assert cleaned["removed_release_id"] == new.final_release.name
    assert canary.exists()
    assert old.final_release.exists() and not new.final_release.exists()
    assert not reserved.exists()
    assert outside.read_text() == "preserve"


def test_canary_rollback_cas_preserves_concurrent_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = tmp_path / "canary"
    old = deploy_remote.deploy_local(
        _completed_release(tmp_path / "cas-old", b"old\n"), canary, target_role="h200"
    )
    attempted = deploy_remote.deploy_local(
        _completed_release(tmp_path / "cas-attempted", b"attempted\n"),
        canary, target_role="h200",
    )
    concurrent = deploy_remote.deploy_local(
        _completed_release(tmp_path / "cas-concurrent", b"concurrent\n"),
        canary, target_role="h200",
    )
    paths = {
        "current_link": (canary / "current").as_posix(),
        "incoming_root": (canary / "incoming").as_posix(),
        "releases_root": (canary / "releases").as_posix(),
    }
    monkeypatch.setitem(
        remote_agent.HOST_POLICIES, "h200",
        {"home": tmp_path.as_posix(), "canary": paths, "production": paths, "data": {}},
    )
    old_target = Path(os.path.relpath(old.final_release, canary)).as_posix()
    attempted_target = Path(os.path.relpath(attempted.final_release, canary)).as_posix()

    with pytest.raises(remote_agent.AgentPolicyError, match="compare-and-swap"):
        remote_agent.force_canary_rollback(
            "h200", old_target, expected_current_target=attempted_target
        )
    assert Path(paths["current_link"]).resolve() == concurrent.final_release.resolve()


def test_cleanup_lock_prevents_concurrent_activation_of_release_being_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = tmp_path / "cleanup-lock-canary"
    old_source = _completed_release(tmp_path / "cleanup-lock-old", b"old\n")
    new_source = _completed_release(tmp_path / "cleanup-lock-new", b"new\n")
    old = deploy_remote.deploy_local(old_source, canary, target_role="h200")
    new = deploy_remote.deploy_local(new_source, canary, target_role="h200")
    paths = {
        "current_link": (canary / "current").as_posix(),
        "incoming_root": (canary / "incoming").as_posix(),
        "releases_root": (canary / "releases").as_posix(),
    }
    monkeypatch.setitem(
        remote_agent.HOST_POLICIES, "h200",
        {"home": tmp_path.as_posix(), "canary": paths, "production": paths, "data": {}},
    )
    old_target = Path(os.path.relpath(old.final_release, canary)).as_posix()
    new_target = Path(os.path.relpath(new.final_release, canary)).as_posix()
    remote_agent.force_canary_rollback(
        "h200", old_target, expected_current_target=new_target
    )
    incoming = Path(paths["incoming_root"]) / (
        "incoming-%s-%s" % (new.final_release.name, "5" * 32)
    )
    reservation = remote_agent.prepare_incoming(
        "h200", incoming.as_posix(), "6" * 64
    )
    (incoming / "staged").write_text("owned")
    original_rmtree = shutil.rmtree
    concurrent_errors = []
    attempted = False

    def rmtree_with_concurrent_install(path, *args, **kwargs):
        nonlocal attempted
        if Path(path) == incoming and not attempted:
            attempted = True
            try:
                contract.install_release(
                    new_source,
                    Path(paths["releases_root"]),
                    Path(paths["current_link"]),
                    target_role="h200",
                )
            except BaseException as exc:
                concurrent_errors.append(exc)
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(remote_agent.shutil, "rmtree", rmtree_with_concurrent_install)
    remote_agent.cleanup_canary(
        "h200",
        incoming_path=incoming.as_posix(),
        reservation_id=reservation["reservation_id"],
        created_release_id=new.final_release.name,
        pre_current_target=old_target,
        pre_release_ids=(old.final_release.name,),
    )

    assert attempted is True
    assert len(concurrent_errors) == 1
    assert isinstance(concurrent_errors[0], contract.ConflictError)
    current = Path(paths["current_link"])
    assert current.is_symlink() and current.exists()
    assert current.resolve() == old.final_release.resolve()
    assert not new.final_release.exists()


def test_audited_allowlist_preserves_top_level_and_required_runtime_files() -> None:
    workspace = Path("/Users/williamqiu/Desktop/Harvard/WAM")
    files = deploy_remote.build_source_allowlist(workspace)
    required = {
        "auscultation/README.md",
        "auscultation/ausc/health.py",
        "auscultation/ausc/perception/robot/openarm_v10.urdf",
        "auscultation/ausc/record_session.py",
        "auscultation/ausc/realtime_q.py",
        "auscultation/ausc/tests/test_recording_durability.py",
        "auscultation/configs/perception_runtime/perception-environment-lock-v3.schema.json",
        "auscultation/scripts/collection_supervisor.py",
        "auscultation/scripts/deployment/activate_perception_environment.py",
        "auscultation/scripts/deployment/prepare_perception_environment_candidate.py",
        "auscultation/scripts/deployment/test_prepare_perception_environment_candidate.py",
        "auscultation/scripts/posterior_gate.py",
        "auscultation/scripts/session_to_lerobot_v2.py",
        "auscultation/scripts/train_bc_smoke.py",
        "auscultation/mechanical_sensor/force_feedback/force_sensor.py",
        "auscultation/mechanical_sensor/force_feedback/firmware/main.py",
        "auscultation/mechanical_sensor/force_feedback/v8b_hall_reader/v8b_hall_reader.ino",
        "auscultation/mechanical_sensor/tests/test_force_sensor.py",
        "auscultation/mechanical_sensor/force_feedback/force_feedback_calibration_veo6_tared.json",
        "auscultation/requirements-perception.txt",
    }
    assert required <= set(files)
    assert all(path.startswith("auscultation/") for path in files)
    assert not any("/docs/runs/" in path or "/data/" in path for path in files)
    assert not any(
        path.endswith((".env", ".mp4", ".png", ".pt", ".pth", ".ckpt", ".safetensors"))
        for path in files
    )
    denied_names = {
        "results.jsonl", "opera_results.jsonl", "xdata_results.jsonl",
        "quality_meta.json", "perception_metrics.json",
    }
    assert not ({Path(path).name for path in files} & denied_names)
    mechanical = [path for path in files if path.startswith("auscultation/mechanical_sensor/")]
    assert mechanical
    assert all(
        path.startswith((
            "auscultation/mechanical_sensor/force_feedback/",
            "auscultation/mechanical_sensor/tests/",
        ))
        for path in mechanical
    )


def test_allowlist_fails_closed_on_symlink(tmp_path: Path) -> None:
    allowed = tmp_path / "auscultation" / "ausc"
    allowed.mkdir(parents=True)
    (allowed / "health.py").write_text("ok")
    (allowed / "linked.py").symlink_to(allowed / "health.py")
    with pytest.raises(deploy_remote.PolicyError, match="symlink"):
        deploy_remote.build_source_allowlist(tmp_path, required_paths=("auscultation/ausc/health.py",))


def test_local_transport_installs_idempotently_with_exact_target_marker(tmp_path: Path) -> None:
    release = _completed_release(tmp_path / "artifact")
    target = tmp_path / "target"
    first = deploy_remote.deploy_local(release, target, target_role="h200")
    second = deploy_remote.deploy_local(release, target, target_role="h200")

    assert first.final_release == second.final_release
    assert first.incoming_path != second.incoming_path
    assert not first.incoming_path.exists() and not second.incoming_path.exists()
    assert (target / "current").resolve() == first.final_release.resolve()
    marker = json.loads((first.final_release / contract.COMPLETE_NAME).read_text())
    assert marker["target_role"] == "h200"


@pytest.mark.parametrize("fault", ["interrupt", "corrupt"])
def test_local_transport_fails_closed_on_bad_staging(tmp_path: Path, fault: str) -> None:
    old = _completed_release(tmp_path / "old-artifact", b"old\n")
    target = tmp_path / "target"
    old_result = deploy_remote.deploy_local(old, target, target_role="h200")
    new = _completed_release(tmp_path / "new-artifact", b"new\n")

    def break_stage(path: Path) -> None:
        if fault == "interrupt":
            raise InterruptedError("transfer interrupted")
        (path / "auscultation" / "ausc" / "worker.py").write_bytes(b"corrupt")

    with pytest.raises((InterruptedError, contract.VerificationError)):
        deploy_remote.deploy_local(
            new, target, target_role="h200", staged_callback=break_stage
        )
    assert (target / "current").resolve() == old_result.final_release.resolve()
    assert not list((target / "incoming").glob("incoming-*"))


def test_local_transport_rolls_back_and_preserves_non_symlink_current(tmp_path: Path) -> None:
    release = _completed_release(tmp_path / "artifact")
    target = tmp_path / "target"
    current = target / "current"
    current.mkdir(parents=True)
    (current / "owned").write_text("preserve")
    with pytest.raises(contract.ConflictError, match="non-symlink"):
        deploy_remote.deploy_local(release, target, target_role="h200")
    assert (current / "owned").read_text() == "preserve"


def test_local_transport_restores_previous_pointer_after_post_check(tmp_path: Path) -> None:
    target = tmp_path / "target"
    old = deploy_remote.deploy_local(
        _completed_release(tmp_path / "old", b"old\n"), target, target_role="h200"
    )

    def fail(_path: Path) -> None:
        raise RuntimeError("post activation failed")

    with pytest.raises(RuntimeError, match="post activation"):
        deploy_remote.deploy_local(
            _completed_release(tmp_path / "new", b"new\n"),
            target,
            target_role="h200",
            post_activate=fail,
        )
    assert (target / "current").resolve() == old.final_release.resolve()
