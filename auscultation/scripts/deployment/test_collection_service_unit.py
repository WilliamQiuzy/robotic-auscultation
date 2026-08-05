from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Optional

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import artifact_contract as contract  # noqa: E402
import collection_service_unit as service  # noqa: E402
import publish_perception_runtime as publisher  # noqa: E402
from test_perception_environment_lock_v3 import _v3_lock  # noqa: E402


COLLECTOR_RELATIVE = "auscultation/scripts/collect_server.py"
GENERATOR_RELATIVE = "auscultation/scripts/deployment/collection_service_unit.py"
ENV_VERIFIER_RELATIVE = (
    "auscultation/scripts/deployment/verify_perception_environment.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contract.canonical_json_bytes(value))


def _code_release(root: Path) -> tuple[Path, dict]:
    source = root / "code-source"
    relatives = set(publisher.REQUIRED_HARDENED_CODE_FILES)
    relatives.update(
        {
            GENERATOR_RELATIVE,
            ENV_VERIFIER_RELATIVE,
            "auscultation/scripts/deployment/artifact_contract.py",
            "auscultation/scripts/deployment/perception_environment_manifest.py",
            "auscultation/scripts/deployment/publish_perception_runtime.py",
        }
    )
    for relative in sorted(relatives):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in publisher.REQUIRED_HARDENED_CODE_FILES:
            path.write_bytes((HERE.parents[2] / relative).read_bytes())
        else:
            path.write_text("FIXTURE = 'immutable'\n", encoding="utf-8")
    manifest = contract.build_manifest(
        source,
        sorted(relatives),
        artifact_kind="code_release",
        source_role="mac",
    )
    release = root / manifest["release_id"]
    contract.write_release(source, manifest, release, target_role="linbox")
    return release, manifest


def _hardened_hashes(manifest: dict) -> dict[str, str]:
    rows = {row["path"]: row["sha256"] for row in manifest["payload"]["files"]}
    return {
        relative: rows[relative]
        for relative in sorted(publisher.REQUIRED_HARDENED_CODE_FILES)
    }


def _environment_release(root: Path) -> tuple[Path, Path, dict]:
    lock = copy.deepcopy(_v3_lock())
    release = root / lock["environment_id"]
    python = release / "bin/python3.11"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fixture-cpython-3.11.15")
    python.chmod(0o555)
    (release / "bin/python").symlink_to("python3.11")
    lock["python"]["executable_sha256"] = _sha256(python)

    ffmpeg = release / lock["ffmpeg"]["path"]
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.write_bytes(b"fixture-ffmpeg-7")
    ffmpeg.chmod(0o555)
    lock["ffmpeg"]["sha256"] = _sha256(ffmpeg)
    lock["ffmpeg"]["size"] = ffmpeg.stat().st_size
    row = next(
        item
        for item in lock["environment_tree"]["entries"]
        if item["path"] == lock["ffmpeg"]["path"]
    )
    row.update(
        mode=lock["ffmpeg"]["mode"],
        sha256=lock["ffmpeg"]["sha256"],
        size=lock["ffmpeg"]["size"],
    )
    tree = lock["environment_tree"]
    tree["sha256"] = hashlib.sha256(
        contract.canonical_json_bytes(
            {"entries": tree["entries"], "exclusions": tree["exclusions"]}
        )
    ).hexdigest()
    publisher.validate_environment_lock(lock, require_posterior_gate=True)
    lock_path = root / "environment-lock.json"
    _write_canonical(lock_path, lock)
    return release, lock_path, lock


def _runtime_release(
    root: Path, code_release: Path, code_manifest: dict, lock_path: Path
) -> Path:
    inputs = root / "runtime-inputs"
    checkpoint = inputs / "perc_lmh_v9_occ" / "best.pt"
    gate = inputs / "yolo11x-pose.pt"
    dino = inputs / "dinov2_vitb14_pretrain.pth"
    repo = inputs / "facebookresearch_dinov2_main"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"exact-v9-checkpoint")
    gate.write_bytes(b"exact-posterior-gate")
    dino.write_bytes(b"exact-dino-checkpoint")
    (repo / "dinov2").mkdir(parents=True)
    (repo / "hubconf.py").write_text(
        "def dinov2_vitb14(): pass\n", encoding="utf-8"
    )
    (repo / "dinov2/__init__.py").write_text(
        "VERSION = 'fixture'\n", encoding="utf-8"
    )
    return publisher.build_perception_runtime_release(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=_sha256(checkpoint),
        expected_checkpoint_size=checkpoint.stat().st_size,
        posterior_gate_checkpoint=gate,
        expected_posterior_gate_checkpoint_sha256=_sha256(gate),
        expected_posterior_gate_checkpoint_size=gate.stat().st_size,
        dino_checkpoint=dino,
        expected_dino_checkpoint_sha256=_sha256(dino),
        expected_dino_checkpoint_size=dino.stat().st_size,
        dino_repo=repo,
        expected_dino_repo_sha256=publisher.hash_dino_source_tree(repo)["sha256"],
        code_release=code_release,
        expected_code_release_id=code_manifest["release_id"],
        expected_code_payload_sha256=code_manifest["payload_sha256"],
        expected_hardened_code_sha256=_hardened_hashes(code_manifest),
        environment_lock=lock_path,
        expected_environment_lock_sha256=_sha256(lock_path),
        releases_root=root / "runtime-releases",
    )


def _production_activation(root: Path) -> tuple[Path, str]:
    release = root / "production_activation-fixture"
    release.mkdir(parents=True)
    manifest = {
        "artifacts": {},
        "follower_can_interface": "can3",
        "leader_can_interface": "can1",
        "openarm_pair": {},
        "robot_state_bounds": {},
        "schema": "wam_production_runtime_activation_v1",
        "side": "left",
        "status": "COMPLETE",
    }
    raw = contract.canonical_json_bytes(manifest)
    path = release / "activation.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    _write_canonical(
        release / "COMPLETE",
        {
            "activation_sha256": digest,
            "schema": "wam_production_runtime_activation_completion_v1",
            "status": "COMPLETE",
        },
    )
    return path, digest


@pytest.fixture()
def deployment(tmp_path: Path) -> dict:
    code_release, code_manifest = _code_release(tmp_path / "code")
    environment_release, lock_path, lock = _environment_release(tmp_path / "env")
    runtime_release = _runtime_release(
        tmp_path, code_release, code_manifest, lock_path
    )
    runtime_envelope = publisher.verify_perception_runtime_release(runtime_release)
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    unit_path = tmp_path / "systemd/user/wam-collector.service"
    activation_path, activation_sha256 = _production_activation(
        tmp_path / "activation"
    )
    hall_calibration = tmp_path / "calibration-v1/hall.json"
    sync_calibration = tmp_path / "calibration-v1/audio-force-sync.json"
    _write_canonical(hall_calibration, {
        "calibrated": True,
        "schema_version": "wam_force_calibration_v1",
    })
    _write_canonical(sync_calibration, {
        "measurement_method": "physical_diaphragm_taps",
        "n_matched": 8,
        "offset_std_ms": 0.2,
        "schema_version": "wam_sync_v1",
        "unique_matches": True,
    })
    spec = service.ServiceSpec(
        code_release=code_release,
        code_release_id=code_manifest["release_id"],
        environment_release=environment_release,
        environment_id=lock["environment_id"],
        runtime_release=runtime_release,
        runtime_release_id=runtime_envelope["release_id"],
        camera_id="CL8326300A0",
        sessions_root=sessions_root,
        unit_path=unit_path,
        verifier_python=environment_release / "bin/python",
        production_activation=activation_path,
        production_activation_sha256=activation_sha256,
        hall_calibration=hall_calibration,
        hall_calibration_sha256=_sha256(hall_calibration),
        hall_port=(
            "/dev/serial/by-id/"
            "usb-MicroPython_Board_in_FS_mode_5303284738b7539c-if00"
        ),
        sync_calibration=sync_calibration,
        sync_calibration_sha256=_sha256(sync_calibration),
    )
    return {
        "spec": spec,
        "code_manifest": code_manifest,
        "environment_lock": lock,
        "runtime_envelope": runtime_envelope,
    }


def test_renders_reproducible_unit_from_real_cli_contract(deployment: dict) -> None:
    bound = service.bind_releases(deployment["spec"])
    first = service.render_unit(bound)
    second = service.render_unit(service.bind_releases(deployment["spec"]))

    assert first == second
    text = first.decode("utf-8")
    assert text.endswith("\n")
    assert "current" not in text.lower()
    assert "Environment=PYTHONDONTWRITEBYTECODE=1" in text
    assert "Environment=AUSC_POSTERIOR_GATE_CHECKPOINT=" in text
    assert text.count("ExecStartPre=") == 5
    assert "Restart=on-failure" in text
    assert "TimeoutStopSec=3600s" in text
    assert "--diagnostic-min-free-gib 50" in text
    assert "--max-tracking-age-ms 250" in text
    assert "--audio-device usb-serial:Y77BQVG17CD6E1" in text
    assert "--bind 127.0.0.1" in text

    collect_server = service.load_module_from_release_source(
        HERE.parent / "collect_server.py", "collector_cli_for_unit_test"
    )
    argv = service.collector_argv(bound)
    args = collect_server.build_parser().parse_args(argv[3:])
    assert (args.camera_width, args.camera_height, args.camera_fps) == (1920, 1080, 30)
    assert (args.depth_width, args.depth_height, args.depth_fps) == (640, 576, 30)
    assert args.camera_color_format == "MJPG"
    assert args.depth_format == "Y16"
    assert args.camera_id == "CL8326300A0"
    assert args.bind == "127.0.0.1"
    assert args.diagnostic_min_free_gib == 50.0
    assert args.max_tracking_age_ms == 250.0
    assert args.posterior_gate_checkpoint
    assert args.posterior_gate_checkpoint_sha256
    assert args.posterior_gate_release_id
    assert args.ffmpeg_path
    assert args.ffmpeg_sha256
    assert args.audio_device == "usb-serial:Y77BQVG17CD6E1"
    assert args.audio_sample_rate == 48_000
    assert args.audio_channels == 2
    assert args.audio_pick_channel == 1
    assert args.audio_expected_usb_vid == 0x1235
    assert args.audio_expected_usb_pid == 0x8211
    assert args.audio_expected_usb_serial == "Y77BQVG17CD6E1"
    assert args.audio_expected_usb_port_path is None
    assert args.require_audio is True
    assert args.require_force is True
    assert args.production_teleop is True
    assert args.commissioning_preview_only is False
    assert args.hall_calibration == str(bound.spec.hall_calibration)
    assert args.hall_calibration_sha256 == bound.spec.hall_calibration_sha256
    assert args.hall_port == (
        "/dev/serial/by-id/"
        "usb-MicroPython_Board_in_FS_mode_5303284738b7539c-if00"
    )
    assert args.hall_expected_usb_vid == 0x2E8A
    assert args.hall_expected_usb_pid == 0x0005
    assert args.hall_expected_usb_serial == "5303284738b7539c"
    assert args.sync_calibration == str(bound.spec.sync_calibration)
    assert args.sync_calibration_sha256 == bound.spec.sync_calibration_sha256
    assert args.production_activation == str(bound.spec.production_activation)
    assert args.production_activation_sha256 == (
        bound.spec.production_activation_sha256
    )


def test_v2_unit_refuses_a_self_consistent_legacy_runtime_contract(
    deployment: dict, monkeypatch,
) -> None:
    runtime_release = deployment["spec"].runtime_release
    semantic_path = runtime_release / publisher.RUNTIME_MANIFEST_NAME
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["schema"] = publisher.V4_RUNTIME_MANIFEST_SCHEMA
    semantic.pop("capabilities")
    semantic_path.chmod(0o644)
    semantic_path.write_bytes(contract.canonical_json_bytes(semantic))
    semantic_path.chmod(0o444)

    # Model a byte-consistent, independently verified v4 rollback release.  A
    # v2 service must still reject it instead of reinterpreting old runtime
    # semantics as the new dormant-arm/non-promotion capability boundary.
    monkeypatch.setattr(
        service.publisher,
        "verify_perception_runtime_release",
        lambda _path: deployment["runtime_envelope"],
    )

    with pytest.raises(service.UnitContractError, match="runtime schema v5"):
        service.bind_releases(deployment["spec"])


def test_v2_unit_runtime_boundary_is_not_reinterpreted_by_publisher_constants(
    deployment: dict, monkeypatch,
) -> None:
    monkeypatch.setattr(
        service.publisher,
        "verify_perception_runtime_release",
        lambda _path: deployment["runtime_envelope"],
    )
    monkeypatch.setattr(
        service.publisher,
        "RUNTIME_MANIFEST_SCHEMA",
        "wam_perception_runtime_v6",
    )
    monkeypatch.setattr(
        service.publisher,
        "RUNTIME_MANIFEST_NAME",
        "REINTERPRETED_RUNTIME.json",
    )
    monkeypatch.setattr(
        service.publisher,
        "RUNTIME_CAPABILITIES",
        {
            "arm_control_authority": True,
            "controller_launch_state": "ACTIVE",
            "training_promotion": "ACTIVATED",
        },
    )

    bound = service.bind_releases(deployment["spec"])

    assert bound.spec.runtime_release_id == deployment["spec"].runtime_release_id


def _verify_runtime_preflight(deployment: dict) -> dict:
    spec = deployment["spec"]
    return service.verify_runtime_release(
        runtime_release=spec.runtime_release,
        runtime_release_id=spec.runtime_release_id,
        code_release_id=spec.code_release_id,
        code_payload_sha256=deployment["code_manifest"]["payload_sha256"],
        environment_id=spec.environment_id,
        environment_lock_sha256=_sha256(
            spec.runtime_release / publisher.ENVIRONMENT_LOCK_PATH
        ),
    )


def test_runtime_boot_preflight_refuses_self_consistent_v4_rollback(
    deployment: dict, monkeypatch,
) -> None:
    runtime_release = deployment["spec"].runtime_release
    semantic_path = runtime_release / service.REQUIRED_RUNTIME_MANIFEST_V2
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["schema"] = publisher.V4_RUNTIME_MANIFEST_SCHEMA
    semantic.pop("capabilities")
    semantic_path.chmod(0o644)
    semantic_path.write_bytes(contract.canonical_json_bytes(semantic))
    semantic_path.chmod(0o444)

    # The generic runtime verifier intentionally supports older schemas.  The
    # unit's boot preflight must enforce its own v2->v5 capability boundary.
    monkeypatch.setattr(
        service.publisher,
        "verify_perception_runtime_release",
        lambda _path: deployment["runtime_envelope"],
    )

    with pytest.raises(service.UnitContractError, match="runtime schema v5"):
        _verify_runtime_preflight(deployment)


def test_runtime_boot_preflight_uses_local_v5_contract_not_publisher_constants(
    deployment: dict, monkeypatch,
) -> None:
    monkeypatch.setattr(
        service.publisher,
        "verify_perception_runtime_release",
        lambda _path: deployment["runtime_envelope"],
    )
    monkeypatch.setattr(
        service.publisher,
        "RUNTIME_MANIFEST_SCHEMA",
        "wam_perception_runtime_v6",
    )
    monkeypatch.setattr(
        service.publisher,
        "RUNTIME_MANIFEST_NAME",
        "REINTERPRETED_RUNTIME.json",
    )
    monkeypatch.setattr(
        service.publisher,
        "RUNTIME_CAPABILITIES",
        {
            "arm_control_authority": True,
            "controller_launch_state": "ACTIVE",
            "training_promotion": "ACTIVATED",
        },
    )

    report = _verify_runtime_preflight(deployment)

    assert report["status"] == "VERIFIED"


def _with_dormant_controller_launch(deployment: dict, tmp_path: Path):
    contract_sha256 = "d" * 64
    trusted_root = tmp_path / "controller-launch-releases"
    release = trusted_root / ("controller_launch_v1-" + contract_sha256)
    release.mkdir(parents=True)
    release.chmod(0o555)
    trusted_root.chmod(0o555)
    return service.replace_spec(
        deployment["spec"],
        controller_launch_release=release,
        controller_launch_trusted_root=trusted_root,
        controller_launch_contract_sha256=contract_sha256,
    )


def test_dormant_controller_launch_is_pinned_in_argv_but_never_executed(
    deployment: dict, tmp_path: Path,
) -> None:
    spec = _with_dormant_controller_launch(deployment, tmp_path)
    bound = service.bind_releases(spec)

    argv = service.collector_argv(bound)
    unit = service.render_unit(bound).decode("utf-8")

    assert argv[argv.index("--controller-launch-release") + 1] == str(
        spec.controller_launch_release
    )
    assert argv[argv.index("--controller-launch-trusted-root") + 1] == str(
        spec.controller_launch_trusted_root
    )
    assert argv[argv.index("--controller-launch-contract-sha256") + 1] == (
        spec.controller_launch_contract_sha256
    )
    assert service.UNIT_SCHEMA == "wam_linbox_collector_unit_v2"
    assert "GRIPPER_" not in unit
    assert "/controller/unilateral_control" not in unit
    assert "/tools/preflight" not in unit
    assert "stop_tool" not in unit
    assert "ExecStop=" not in unit
    assert "ExecStopPost=" not in unit
    assert "--teleop-launch-cmd" not in unit
    assert "--controller-disable-cmd" not in unit
    assert "--controller-preflight-cmd" not in unit


@pytest.mark.parametrize(
    "missing",
    (
        "controller_launch_release",
        "controller_launch_trusted_root",
        "controller_launch_contract_sha256",
    ),
)
def test_dormant_controller_launch_binding_is_all_or_none_and_exact(
    deployment: dict, tmp_path: Path, missing: str,
) -> None:
    spec = _with_dormant_controller_launch(deployment, tmp_path)
    values = {
        "controller_launch_release": spec.controller_launch_release,
        "controller_launch_trusted_root": spec.controller_launch_trusted_root,
        "controller_launch_contract_sha256": (
            spec.controller_launch_contract_sha256
        ),
    }
    values[missing] = None
    with pytest.raises(service.UnitContractError, match="controller launch"):
        service.bind_releases(service.replace_spec(spec, **values))

    wrong_release = spec.controller_launch_trusted_root / (
        "controller_launch_v1-" + "e" * 64
    )
    with pytest.raises(service.UnitContractError, match="controller launch"):
        service.bind_releases(
            service.replace_spec(spec, controller_launch_release=wrong_release)
        )


def test_preview_forbids_dormant_controller_launch_binding(
    deployment: dict, tmp_path: Path,
) -> None:
    spec = _with_dormant_controller_launch(deployment, tmp_path)
    preview = service.replace_spec(
        spec,
        commissioning_preview_only=True,
        production_activation=None,
        production_activation_sha256=None,
        hall_calibration=None,
        hall_calibration_sha256=None,
        hall_port=None,
        sync_calibration=None,
        sync_calibration_sha256=None,
    )

    with pytest.raises(service.UnitContractError, match="preview"):
        service.bind_releases(preview)


def test_commissioning_preview_unit_omits_activation_force_and_recording_argv(
    deployment: dict,
) -> None:
    preview_spec = service.replace_spec(
        deployment["spec"],
        commissioning_preview_only=True,
        production_activation=None,
        production_activation_sha256=None,
        hall_calibration=None,
        hall_calibration_sha256=None,
        hall_port=None,
        sync_calibration=None,
        sync_calibration_sha256=None,
    )

    bound = service.bind_releases(preview_spec)
    argv = service.collector_argv(bound)
    preflights = service.preflight_commands(bound)
    unit = service.render_unit(bound).decode("utf-8")

    assert "--commissioning-preview-only" in argv
    assert "--require-audio" in argv
    assert "usb-serial:Y77BQVG17CD6E1" in argv
    assert "--production-activation" not in argv
    assert "--production-activation-sha256" not in argv
    assert "--require-force" not in argv
    assert "--hall-calibration" not in argv
    assert "--sync-calibration" not in argv
    assert len(preflights) == 3
    assert all("verify-activation" not in command for command in preflights)
    assert all("verify-force" not in command for command in preflights)
    assert unit.count("ExecStartPre=") == 3
    assert "--commissioning-preview-only" in unit


def test_commissioning_preview_can_bind_unsynced_hall_monitor_without_training(
    deployment: dict,
) -> None:
    preview_spec = service.replace_spec(
        deployment["spec"],
        commissioning_preview_only=True,
        commissioning_hall_monitor_only=True,
        production_activation=None,
        production_activation_sha256=None,
        sync_calibration=None,
        sync_calibration_sha256=None,
    )

    bound = service.bind_releases(preview_spec)
    argv = service.collector_argv(bound)
    preflights = service.preflight_commands(bound)

    assert "--commissioning-hall-monitor-only" in argv
    assert "--hall-calibration" in argv
    assert "--hall-calibration-sha256" in argv
    assert "--hall-port" in argv
    assert "--require-force" not in argv
    assert "--sync-calibration" not in argv
    assert len(preflights) == 4
    assert any("verify-hall-monitor" in command for command in preflights)
    assert all("verify-force" not in command for command in preflights)


def test_commissioning_preview_deployment_cli_does_not_require_production_inputs():
    parser = service.build_deployment_parser()
    args = parser.parse_args([
        "--render",
        "--commissioning-preview-only",
        "--code-release", "/releases/code_release-" + "a" * 20,
        "--code-release-id", "code_release-" + "a" * 20,
        "--environment-release", "/releases/linbox-perception-" + "b" * 20,
        "--environment-id", "linbox-perception-" + "b" * 20,
        "--runtime-release", "/releases/perception_runtime-" + "c" * 20,
        "--runtime-release-id", "perception_runtime-" + "c" * 20,
        "--camera-id", "CL8326300A0",
        "--sessions-root", "/data/sessions",
        "--unit-path", "/home/robot/.config/systemd/user/wam.service",
        "--verifier-python", "/releases/env/bin/python",
    ])

    spec = service._service_spec_from_args(args)
    assert spec.commissioning_preview_only is True
    assert spec.production_activation is None
    assert spec.hall_calibration is None
    assert spec.sync_calibration is None


def test_deployment_cli_rejects_partial_or_preview_controller_launch_binding():
    common = [
        "--code-release", "/releases/code_release-" + "a" * 20,
        "--code-release-id", "code_release-" + "a" * 20,
        "--environment-release", "/releases/linbox-perception-" + "b" * 20,
        "--environment-id", "linbox-perception-" + "b" * 20,
        "--runtime-release", "/releases/perception_runtime-" + "c" * 20,
        "--runtime-release-id", "perception_runtime-" + "c" * 20,
        "--camera-id", "CL8326300A0",
        "--sessions-root", "/data/sessions",
        "--unit-path", "/home/robot/.config/systemd/user/wam.service",
        "--verifier-python", "/releases/env/bin/python",
    ]
    parser = service.build_deployment_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--render", "--commissioning-preview-only",
            "--controller-launch-release",
            "/releases/controller_launch_v1-" + "d" * 64,
            *common,
        ])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--render", "--commissioning-preview-only",
            "--controller-launch-release",
            "/releases/controller_launch_v1-" + "d" * 64,
            "--controller-launch-trusted-root", "/releases",
            "--controller-launch-contract-sha256", "d" * 64,
            *common,
        ])


def test_commissioning_preview_handoff_never_claims_production(monkeypatch):
    class Bindings:
        class Spec:
            commissioning_preview_only = True
        spec = Spec()

    monkeypatch.setattr(
        service, "check_unit", lambda *_args, **_kwargs: {"status": "VERIFIED"},
    )
    monkeypatch.setattr(
        service, "verify_user_linger",
        lambda *_args, **_kwargs: {"status": "VERIFIED"},
    )
    monkeypatch.setattr(
        service, "verify_systemd_handoff",
        lambda *_args, **_kwargs: {"status": "VERIFIED"},
    )

    report = service.production_handoff_check(
        Bindings(), linger_user="robot",
    )

    assert report["status"] == "COMMISSIONING_PREVIEW_HANDOFF_VERIFIED"
    assert "PRODUCTION" not in report["status"]


def test_default_audio_identity_matches_commissioned_scarlett_input_two() -> None:
    parser = service.build_deployment_parser()

    assert service.ServiceSpec.__dataclass_fields__["audio_device"].default == (
        "usb-serial:Y77BQVG17CD6E1"
    )
    assert service._AUDIO_DEVICE.fullmatch(
        "usb-serial:Y77BQVG17CD6E1"
    )
    assert parser.get_default("audio_device") == (
        "usb-serial:Y77BQVG17CD6E1"
    )
    assert parser.get_default("audio_expected_usb_vid") == 0x1235
    assert parser.get_default("audio_expected_usb_pid") == 0x8211
    assert parser.get_default("audio_expected_usb_serial") == "Y77BQVG17CD6E1"
    assert parser.get_default("audio_expected_usb_port_path") is None


def test_rejects_dynamic_or_cross_release_bindings(
    deployment: dict, tmp_path: Path
) -> None:
    spec = deployment["spec"]
    with pytest.raises(service.UnitContractError, match="code release id"):
        service.bind_releases(
            service.replace_spec(spec, code_release_id="code_release-" + "0" * 20)
        )

    current = tmp_path / "current"
    current.symlink_to(spec.code_release, target_is_directory=True)
    with pytest.raises(service.UnitContractError, match="immutable"):
        service.bind_releases(service.replace_spec(spec, code_release=current))

    with pytest.raises(service.UnitContractError, match="environment"):
        service.bind_releases(
            service.replace_spec(spec, environment_id="linbox-perception-" + "f" * 20)
        )


def test_verifier_python_must_be_real_and_belong_to_bound_environment(
    deployment: dict, tmp_path: Path
) -> None:
    spec = deployment["spec"]
    alias = tmp_path / "python3"
    alias.symlink_to(spec.verifier_python)

    with pytest.raises(service.UnitContractError, match="exact verified internal alias"):
        service.bind_releases(
            service.replace_spec(spec, verifier_python=alias)
        )
    with pytest.raises(service.UnitContractError, match="bound environment"):
        service.bind_releases(
            service.replace_spec(
                spec, verifier_python=Path(sys.executable).resolve(strict=True)
            )
        )


def test_environment_python_alias_must_be_the_internal_cpython311_target(
    deployment: dict,
) -> None:
    spec = deployment["spec"]
    alias = spec.environment_release / "bin/python"
    alias.unlink()
    alias.symlink_to("../outside-python")

    with pytest.raises(service.UnitContractError, match="environment Python"):
        service.bind_releases(spec)


def test_systemd_tokens_reject_dollar_expansion(deployment: dict) -> None:
    spec = deployment["spec"]
    unsafe_sessions = spec.sessions_root.parent / "$SESSIONS"
    unsafe_sessions.mkdir()

    with pytest.raises(service.UnitContractError, match="systemd"):
        service.bind_releases(
            service.replace_spec(spec, sessions_root=unsafe_sessions)
        )


def test_bind_normalizes_artifact_verification_failure(deployment: dict) -> None:
    spec = deployment["spec"]
    (spec.code_release / COLLECTOR_RELATIVE).write_text(
        "TAMPERED = True\n", encoding="utf-8"
    )

    with pytest.raises(service.UnitContractError, match="code release verification"):
        service.bind_releases(spec)


def test_five_preflights_include_force_hashes_pyserial_and_activation(deployment: dict) -> None:
    bound = service.bind_releases(deployment["spec"])
    code, runtime, activation, environment, force = service.preflight_commands(bound)

    code_args = service.build_parser().parse_args(code[3:])
    assert code_args.command == "verify-code"
    assert code_args.code_release_id == bound.spec.code_release_id
    runtime_args = service.build_parser().parse_args(runtime[3:])
    assert runtime_args.command == "verify-runtime"
    assert runtime_args.environment_id == bound.spec.environment_id
    activation_args = service.build_parser().parse_args(activation[3:])
    assert activation_args.command == "verify-activation"
    assert activation_args.production_activation == str(
        bound.spec.production_activation
    )
    assert activation_args.production_activation_sha256 == (
        bound.spec.production_activation_sha256
    )

    verifier = service.load_module_from_release_source(
        HERE / "verify_perception_environment.py", "environment_cli_for_unit_test"
    )
    environment_args = verifier._argument_parser().parse_args(environment[3:])
    assert environment_args.lock == str(bound.environment_lock_path)
    assert environment_args.environment_root == str(bound.spec.environment_release)
    force_args = service.build_parser().parse_args(force[3:])
    assert force_args.command == "verify-force"
    assert force_args.hall_calibration_sha256 == bound.spec.hall_calibration_sha256
    assert force_args.sync_calibration_sha256 == bound.spec.sync_calibration_sha256


def test_internal_preflights_fail_closed_on_any_identity_drift(
    deployment: dict,
) -> None:
    bound = service.bind_releases(deployment["spec"])
    code_report = service.verify_code_release(
        bound.spec.code_release,
        bound.spec.code_release_id,
        bound.code_payload_sha256,
    )
    assert code_report["status"] == "VERIFIED"
    runtime_report = service.verify_runtime_release(
        runtime_release=bound.spec.runtime_release,
        runtime_release_id=bound.spec.runtime_release_id,
        code_release_id=bound.spec.code_release_id,
        code_payload_sha256=bound.code_payload_sha256,
        environment_id=bound.spec.environment_id,
        environment_lock_sha256=bound.environment_lock_sha256,
    )
    assert runtime_report["status"] == "VERIFIED"

    with pytest.raises(service.UnitContractError, match="payload"):
        service.verify_code_release(
            bound.spec.code_release,
            bound.spec.code_release_id,
            "0" * 64,
        )
    with pytest.raises(service.UnitContractError, match="environment"):
        service.verify_runtime_release(
            runtime_release=bound.spec.runtime_release,
            runtime_release_id=bound.spec.runtime_release_id,
            code_release_id=bound.spec.code_release_id,
            code_payload_sha256=bound.code_payload_sha256,
            environment_id="linbox-perception-" + "f" * 20,
            environment_lock_sha256=bound.environment_lock_sha256,
        )


def test_force_preflight_imports_pyserial_and_verifies_both_pins(
    deployment: dict,
) -> None:
    spec = deployment["spec"]
    report = service.verify_force_configuration(
        hall_calibration=spec.hall_calibration,
        hall_calibration_sha256=spec.hall_calibration_sha256,
        sync_calibration=spec.sync_calibration,
        sync_calibration_sha256=spec.sync_calibration_sha256,
    )

    assert report["status"] == "VERIFIED"
    assert report["pyserial"] == "importable"


def test_force_bind_rejects_tamper_symlink_and_uncalibrated_seed(
    deployment: dict, tmp_path: Path,
) -> None:
    spec = deployment["spec"]
    spec.sync_calibration.write_bytes(spec.sync_calibration.read_bytes() + b" ")
    with pytest.raises(service.UnitContractError, match="SHA-256 mismatch"):
        service.bind_releases(spec)

    _write_canonical(spec.sync_calibration, {
        "measurement_method": "physical_diaphragm_taps",
        "n_matched": 8,
        "offset_std_ms": 0.2,
        "schema_version": "wam_sync_v1",
        "unique_matches": True,
    })
    uncalibrated = tmp_path / "immutable-calibration/uncalibrated.json"
    _write_canonical(uncalibrated, {"calibrated": False})
    with pytest.raises(service.UnitContractError, match="calibrated=true"):
        service.bind_releases(service.replace_spec(
            spec,
            hall_calibration=uncalibrated,
            hall_calibration_sha256=_sha256(uncalibrated),
            sync_calibration_sha256=_sha256(spec.sync_calibration),
        ))

    alias = tmp_path / "immutable-calibration/hall-alias.json"
    alias.symlink_to(spec.hall_calibration)
    with pytest.raises(service.UnitContractError, match="non-symlink"):
        service.bind_releases(service.replace_spec(
            spec,
            hall_calibration=alias,
            hall_calibration_sha256=spec.hall_calibration_sha256,
            sync_calibration_sha256=_sha256(spec.sync_calibration),
        ))


class _Runner:
    def __init__(self, *, fail_at: Optional[int] = None) -> None:
        self.fail_at = fail_at
        self.calls: list[tuple[tuple[str, ...], dict]] = []

    def __call__(self, command, **kwargs):
        index = len(self.calls)
        self.calls.append((tuple(command), kwargs))
        if index == self.fail_at:
            return subprocess.CompletedProcess(
                command, 9, stdout=b"", stderr=b"preflight failed"
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=contract.canonical_json_bytes(
                {"schema": "fixture", "status": "VERIFIED"}
            ),
            stderr=b"",
        )


class _HandoffRunner(_Runner):
    def __call__(self, command, **kwargs):
        self.calls.append((tuple(command), kwargs))
        if command[0] == "loginctl":
            return subprocess.CompletedProcess(
                command, 0, stdout=b"yes\n", stderr=b""
            )
        if command[0] == "systemctl":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    b"LoadState=loaded\n"
                    b"ActiveState=active\n"
                    b"SubState=running\n"
                    b"UnitFileState=enabled\n"
                    b"NeedDaemonReload=no\n"
                    b"MainPID=4242\n"
                    + ("FragmentPath=%s\n" % self.unit_path).encode("utf-8")
                ),
                stderr=b"",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=contract.canonical_json_bytes(
                {"schema": "fixture", "status": "VERIFIED"}
            ),
            stderr=b"",
        )

    def __init__(self, unit_path: Path) -> None:
        super().__init__()
        self.unit_path = unit_path


def test_install_runs_all_preflights_before_first_filesystem_mutation(
    deployment: dict,
) -> None:
    bound = service.bind_releases(deployment["spec"])
    parent = bound.spec.unit_path.parent
    assert not parent.exists()
    runner = _Runner(fail_at=1)

    with pytest.raises(service.UnitContractError, match="preflight"):
        service.install_unit(bound, expect_no_existing=True, runner=runner)

    assert len(runner.calls) == 2
    assert not parent.exists()


def test_fifth_force_preflight_failure_is_labeled_and_fail_closed(
    deployment: dict,
) -> None:
    bound = service.bind_releases(deployment["spec"])
    runner = _Runner(fail_at=4)

    with pytest.raises(service.UnitContractError, match="force preflight failed"):
        service.run_preflights(bound, runner=runner)

    assert len(runner.calls) == 5


def test_atomic_install_check_and_compare_and_swap(deployment: dict) -> None:
    first = service.bind_releases(deployment["spec"])
    runner = _Runner()
    report = service.install_unit(first, expect_no_existing=True, runner=runner)
    unit_path = first.spec.unit_path
    first_bytes = service.render_unit(first)
    first_sha = hashlib.sha256(first_bytes).hexdigest()

    assert report["status"] == "INSTALLED"
    assert report["unit_sha256"] == first_sha
    assert unit_path.read_bytes() == first_bytes
    assert stat.S_IMODE(unit_path.stat().st_mode) == 0o644
    assert len(runner.calls) == 5
    assert runner.calls[0][1]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not list(unit_path.parent.glob(".%s.*.tmp" % unit_path.name))
    assert not service.unit_lock_path(unit_path).exists()

    checked = service.check_unit(first, runner=_Runner())
    assert checked["status"] == "VERIFIED"
    assert checked["unit_sha256"] == first_sha

    second = service.bind_releases(
        service.replace_spec(first.spec, camera_id="CL8326300A1")
    )
    second_bytes = service.render_unit(second)
    service.install_unit(
        second, expected_unit_sha256=first_sha, runner=_Runner()
    )
    assert unit_path.read_bytes() == second_bytes

    with pytest.raises(service.UnitContractError, match="compare-and-swap"):
        service.install_unit(
            first, expected_unit_sha256=first_sha, runner=_Runner()
        )
    assert unit_path.read_bytes() == second_bytes


def test_check_rejects_unit_drift_and_symlink(deployment: dict) -> None:
    bound = service.bind_releases(deployment["spec"])
    service.install_unit(bound, expect_no_existing=True, runner=_Runner())
    unit_path = bound.spec.unit_path
    unit_path.write_bytes(unit_path.read_bytes() + b"# drift\n")
    with pytest.raises(service.UnitContractError, match="drift"):
        service.check_unit(bound, runner=_Runner())

    unit_path.unlink()
    target = unit_path.parent / "target.service"
    target.write_bytes(service.render_unit(bound))
    unit_path.symlink_to(target)
    with pytest.raises(service.UnitContractError, match="symlink"):
        service.check_unit(bound, runner=_Runner())


def test_replace_failure_preserves_old_unit_and_cleans_temporary_file(
    deployment: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = service.bind_releases(deployment["spec"])
    service.install_unit(first, expect_no_existing=True, runner=_Runner())
    old = first.spec.unit_path.read_bytes()
    old_sha = hashlib.sha256(old).hexdigest()
    second = service.bind_releases(
        service.replace_spec(first.spec, camera_id="CL8326300A1")
    )

    def fail_replace(_source, _destination):
        raise OSError("simulated interrupted replace")

    monkeypatch.setattr(service.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        service.install_unit(
            second, expected_unit_sha256=old_sha, runner=_Runner()
        )

    assert first.spec.unit_path.read_bytes() == old
    assert not list(first.spec.unit_path.parent.glob(".%s.*.tmp" % first.spec.unit_path.name))
    assert not service.unit_lock_path(first.spec.unit_path).exists()


def test_deployment_cli_exposes_check_and_fail_closed_install_expectation(
    deployment: dict,
) -> None:
    spec = deployment["spec"]
    common = [
        "--code-release", str(spec.code_release),
        "--code-release-id", spec.code_release_id,
        "--environment-release", str(spec.environment_release),
        "--environment-id", spec.environment_id,
        "--runtime-release", str(spec.runtime_release),
        "--runtime-release-id", spec.runtime_release_id,
        "--camera-id", spec.camera_id,
        "--sessions-root", str(spec.sessions_root),
        "--unit-path", str(spec.unit_path),
        "--verifier-python", str(spec.verifier_python),
        "--production-activation", str(spec.production_activation),
        "--production-activation-sha256", spec.production_activation_sha256,
        "--hall-calibration", str(spec.hall_calibration),
        "--hall-calibration-sha256", spec.hall_calibration_sha256,
        "--hall-port", spec.hall_port,
        "--sync-calibration", str(spec.sync_calibration),
        "--sync-calibration-sha256", spec.sync_calibration_sha256,
    ]
    checked = service.build_deployment_parser().parse_args(["--check", *common])
    assert checked.check is True
    install = service.build_deployment_parser().parse_args(
        ["--install", "--expect-no-existing", *common]
    )
    assert install.install is True
    assert install.expect_no_existing is True
    production = service.build_deployment_parser().parse_args(
        ["--production-check", "--linger-user", "robot", *common]
    )
    assert production.production_check is True
    assert production.linger_user == "robot"

    configured = service.build_deployment_parser().parse_args([
        "--check",
        "--audio-device", "usb-serial:Y77BQVG17CD6E1",
        "--audio-sample-rate", "48000",
        "--audio-channels", "2",
        "--audio-pick-channel", "1",
        "--audio-expected-usb-vid", "0x1235",
        "--audio-expected-usb-pid", "0x8211",
        "--audio-expected-usb-serial", "Y77BQVG17CD6E1",
        "--controller-launch-release",
        "/releases/controller_launch_v1-" + "d" * 64,
        "--controller-launch-trusted-root", "/releases",
        "--controller-launch-contract-sha256", "d" * 64,
        *common,
    ])
    configured_spec = service._service_spec_from_args(configured)
    assert configured_spec.audio_device == "usb-serial:Y77BQVG17CD6E1"
    assert configured_spec.audio_sample_rate == 48_000
    assert configured_spec.audio_channels == 2
    assert configured_spec.audio_pick_channel == 1
    assert configured_spec.audio_expected_usb_vid == 0x1235
    assert configured_spec.audio_expected_usb_pid == 0x8211
    assert configured_spec.audio_expected_usb_serial == "Y77BQVG17CD6E1"
    assert configured_spec.audio_expected_usb_port_path is None
    assert configured_spec.production_activation == spec.production_activation
    assert configured_spec.production_activation_sha256 == (
        spec.production_activation_sha256
    )
    assert configured_spec.controller_launch_release == Path(
        "/releases/controller_launch_v1-" + "d" * 64
    )
    assert configured_spec.controller_launch_trusted_root == Path("/releases")
    assert configured_spec.controller_launch_contract_sha256 == "d" * 64

    without_verifier = common[:-2]
    with pytest.raises(SystemExit):
        service.build_deployment_parser().parse_args(
            ["--check", *without_verifier]
        )


def test_rejects_audio_mode_that_does_not_match_commissioned_scarlett_input(
    deployment: dict,
) -> None:
    spec = deployment["spec"]
    with pytest.raises(service.UnitContractError, match="Scarlett audio mode"):
        service.bind_releases(service.replace_spec(spec, audio_sample_rate=44_100))
    with pytest.raises(service.UnitContractError, match="Scarlett audio mode"):
        service.bind_releases(service.replace_spec(spec, audio_channels=1))
    with pytest.raises(service.UnitContractError, match="Scarlett audio mode"):
        service.bind_releases(service.replace_spec(spec, audio_pick_channel=0))
    with pytest.raises(service.UnitContractError, match="Scarlett USB identity"):
        service.bind_releases(
            service.replace_spec(spec, audio_expected_usb_serial="wrong")
        )
    with pytest.raises(service.UnitContractError, match="Scarlett USB identity"):
        service.bind_releases(
            service.replace_spec(spec, audio_expected_usb_port_path="1-12")
        )
    with pytest.raises(service.UnitContractError, match="Scarlett audio mode"):
        service.bind_releases(
            service.replace_spec(spec, audio_device="Wrong USB Input")
        )


def test_activation_manifest_is_exactly_pinned_and_complete(
    deployment: dict,
) -> None:
    spec = deployment["spec"]
    spec.production_activation.write_bytes(
        spec.production_activation.read_bytes() + b"\n"
    )

    with pytest.raises(service.UnitContractError, match="activation.*SHA-256"):
        service.bind_releases(spec)


def test_read_only_linger_and_systemd_handoff_gate(
    deployment: dict, tmp_path: Path
) -> None:
    bound = service.bind_releases(deployment["spec"])
    service.install_unit(bound, expect_no_existing=True, runner=_Runner())
    runner = _HandoffRunner(bound.spec.unit_path)
    proc_root = tmp_path / "proc"
    cmdline = proc_root / "4242/cmdline"
    cmdline.parent.mkdir(parents=True)
    cmdline.write_bytes(
        b"\0".join(value.encode("utf-8") for value in service.collector_argv(bound))
        + b"\0"
    )

    report = service.production_handoff_check(
        bound, linger_user="robot", runner=runner, proc_root=proc_root
    )

    assert report["status"] == "PRODUCTION_HANDOFF_VERIFIED"
    assert [call[0][0] for call in runner.calls[-2:]] == ["loginctl", "systemctl"]
    assert runner.calls[-2][0] == (
        "loginctl", "show-user", "robot", "--property=Linger", "--value"
    )
    assert runner.calls[-1][0][:4] == (
        "systemctl", "--user", "show", bound.spec.unit_path.name
    )
    assert all("daemon-reload" not in call[0] for call in runner.calls)
    assert all("enable" not in call[0] for call in runner.calls)
    assert all("start" not in call[0] for call in runner.calls)
    assert report["systemd"]["main_pid"] == 4242


def test_production_handoff_rejects_old_collector_process(
    deployment: dict, tmp_path: Path
) -> None:
    bound = service.bind_releases(deployment["spec"])
    service.install_unit(bound, expect_no_existing=True, runner=_Runner())
    runner = _HandoffRunner(bound.spec.unit_path)
    proc_root = tmp_path / "proc"
    cmdline = proc_root / "4242/cmdline"
    cmdline.parent.mkdir(parents=True)
    cmdline.write_bytes(b"/old/env/bin/python\0/old/code/collect_server.py\0")

    with pytest.raises(service.UnitContractError, match="MainPID argv"):
        service.production_handoff_check(
            bound, linger_user="robot", runner=runner, proc_root=proc_root
        )


def test_linger_check_fails_closed_without_mutating_system_state() -> None:
    class DisabledLinger:
        def __call__(self, command, **kwargs):
            return subprocess.CompletedProcess(
                command, 0, stdout=b"no\n", stderr=b""
            )

    with pytest.raises(service.UnitContractError, match="linger"):
        service.verify_user_linger("robot", runner=DisabledLinger())
