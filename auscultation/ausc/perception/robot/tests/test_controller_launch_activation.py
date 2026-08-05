"""Fail-closed tests for the dormant OpenArm controller-launch sidecar."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from ausc.perception.robot.openarm_pair_activation import (
    LEFT_JOINT_NAMES,
    PAIR_SCHEMA,
    CanAdapterIdentity,
    ControllerSourceEntry,
    ExpectedMotorIds,
    OpenArmCanEndpoint,
    OpenArmPairActivation,
    OpenArmSdkEntry,
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pair(*, controller_sha: str, follower_config_sha: str, urdf_sha: str):
    motor_ids = ExpectedMotorIds(
        arm_send_can_ids=tuple(range(1, 8)),
        arm_receive_can_ids=tuple(range(17, 24)),
        gripper_send_can_id=8,
        gripper_receive_can_id=24,
    )
    leader = OpenArmCanEndpoint(
        role="leader",
        can_interface="can1",
        adapter=CanAdapterIdentity(
            id_path="pci-0000:00:14.0-usb-0:1.2:1.0",
            serial="OPENARM-LEADER-TEST",
            channel="0",
        ),
        expected_motor_ids=motor_ids,
    )
    follower = OpenArmCanEndpoint(
        role="follower",
        can_interface="can3",
        adapter=CanAdapterIdentity(
            id_path="pci-0000:00:14.0-usb-0:1.4:1.0",
            serial="OPENARM-FOLLOWER-TEST",
            channel="0",
        ),
        expected_motor_ids=motor_ids,
    )
    return OpenArmPairActivation(
        schema=PAIR_SCHEMA,
        release_id="openarm_pair_v2-" + "a" * 64,
        contract_sha256="a" * 64,
        side="left",
        controller_mode="unilateral",
        leader=leader,
        follower=follower,
        joint_names=LEFT_JOINT_NAMES,
        urdf_sha256=urdf_sha,
        joint_mapper_header_sha256="b" * 64,
        controller_source_manifest=(
            ControllerSourceEntry(
                path="config/follower.yaml",
                sha256=follower_config_sha,
            ),
        ),
        controller_source_manifest_sha256="c" * 64,
        follower_config_sha256=follower_config_sha,
        openarm_sdk_manifest=(
            OpenArmSdkEntry(
                kind="header",
                path="include/openarm/openarm.hpp",
                sha256="d" * 64,
            ),
            OpenArmSdkEntry(
                kind="library",
                path="lib/libopenarm.so",
                sha256="e" * 64,
            ),
        ),
        openarm_sdk_sha256="f" * 64,
        candidate_binary_sha256=controller_sha,
        telemetry_socket="/run/user/1000/wam/openarm-left-telemetry.sock",
        camera_profile_sha256="1" * 64,
        timing_artifact_sha256="2" * 64,
    )


def _fixture_bytes(module) -> dict[str, bytes]:
    return {
        module.CONTROLLER_PATH: b"\x7fELF\x02\x01controller-test-bytes",
        module.LEADER_CONFIG_PATH: b"leader: test\n",
        module.FOLLOWER_CONFIG_PATH: b"follower: test\n",
        module.LEADER_URDF_PATH: b"<robot name='leader'/>\n",
        module.FOLLOWER_URDF_PATH: b"<robot name='follower'/>\n",
        module.PREFLIGHT_PATH: b"\x7fELF\x02\x01preflight-test-bytes",
        module.STOP_TOOL_DESCRIPTOR_PATH: _canonical(
            {
                "schema": module.STOP_TOOL_DESCRIPTOR_SCHEMA,
                "status": "HIL_PENDING",
            }
        ),
    }


def _profile(module, pair: OpenArmPairActivation, blobs: dict[str, bytes]) -> dict:
    roles = {
        module.CONTROLLER_PATH: "controller_elf",
        module.LEADER_CONFIG_PATH: "leader_config",
        module.FOLLOWER_CONFIG_PATH: "follower_config",
        module.LEADER_URDF_PATH: "leader_urdf",
        module.FOLLOWER_URDF_PATH: "follower_urdf",
        module.PREFLIGHT_PATH: "preflight_elf",
        module.STOP_TOOL_DESCRIPTOR_PATH: "stop_tool_descriptor",
    }
    executable = {module.CONTROLLER_PATH, module.PREFLIGHT_PATH}
    return {
        "schema": module.PROFILE_SCHEMA,
        "state": "DORMANT_HIL_PENDING",
        "open_target_status": "HIL_PENDING",
        "openarm_pair": {
            "schema": pair.schema,
            "release_id": pair.release_id,
            "contract_sha256": pair.contract_sha256,
        },
        "artifacts": [
            {
                "role": roles[path],
                "path": path,
                "sha256": _sha(raw),
                "size": len(raw),
                "mode": "0555" if path in executable else "0444",
            }
            for path, raw in sorted(blobs.items())
        ],
        "cwd": ".",
        "argv": [
            module.CONTROLLER_PATH,
            module.LEADER_URDF_PATH,
            module.FOLLOWER_URDF_PATH,
            "left_arm",
            "can1",
            "can3",
        ],
        "environment": module.build_controller_environment(pair),
    }


def _write_release(tmp_path: Path, module, *, blobs: dict[str, bytes] | None = None):
    blobs = _fixture_bytes(module) if blobs is None else blobs
    pair = _pair(
        controller_sha=_sha(blobs[module.CONTROLLER_PATH]),
        follower_config_sha=_sha(blobs[module.FOLLOWER_CONFIG_PATH]),
        # Pair-v2 telemetry binds the leader URDF.  The stethoscope follower
        # URDF remains independently pinned by this launch sidecar.
        urdf_sha=_sha(blobs[module.LEADER_URDF_PATH]),
    )
    envelope = module.build_controller_launch_envelope(
        _profile(module, pair, blobs),
        openarm_pair=pair,
    )
    release = tmp_path / envelope["release_id"]
    release.mkdir(parents=True)
    for relative, raw in blobs.items():
        path = release / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o555 if relative in {module.CONTROLLER_PATH, module.PREFLIGHT_PATH} else 0o444)
    contract_raw = module.canonical_json_bytes(envelope)
    (release / module.CONTRACT_FILE).write_bytes(contract_raw)
    (release / module.COMPLETE_MARKER).write_bytes(
        module.canonical_json_bytes(
            module.build_completion_marker(envelope, contract_raw)
        )
    )
    (release / module.CONTRACT_FILE).chmod(0o444)
    (release / module.COMPLETE_MARKER).chmod(0o444)
    for directory in sorted(
        (path for path in tmp_path.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    tmp_path.chmod(0o555)
    return release, envelope, pair, blobs


def test_valid_sidecar_loads_as_dormant_immutable_direct_launch_contract(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "ausc.perception.robot.controller_launch_activation"
    )
    release, envelope, pair, _ = _write_release(tmp_path, module)

    loaded = module.load_controller_launch_activation(
        release,
        trusted_root=release.parent,
        expected_contract_sha256=envelope["contract_sha256"],
        openarm_pair=pair,
    )

    assert loaded.release_id == envelope["release_id"]
    assert loaded.state == "DORMANT_HIL_PENDING"
    assert loaded.open_target_status == "HIL_PENDING"
    assert loaded.cwd == release.resolve()
    assert loaded.argv == (
        str((release / module.CONTROLLER_PATH).resolve()),
        str((release / module.LEADER_URDF_PATH).resolve()),
        str((release / module.FOLLOWER_URDF_PATH).resolve()),
        "left_arm",
        "can1",
        "can3",
    )
    assert isinstance(loaded.environment, MappingProxyType)
    assert loaded.environment["GRIPPER_DECOUPLE"] == "1"
    assert loaded.environment["GRIPPER_HOLD_POS"] == "0.0"
    assert loaded.environment["GRIP_LEAD"] == "0.08"
    assert loaded.environment["WAM_OPENARM_TELEMETRY_SOCKET"] == pair.telemetry_socket
    assert loaded.environment["WAM_OPENARM_PAIR_CONTRACT_SHA256"] == pair.contract_sha256
    with pytest.raises(TypeError):
        loaded.environment["GRIPPER_HOLD_POS"] = "-1.0"


def test_contract_is_canonical_complete_and_never_staging(tmp_path: Path) -> None:
    module = importlib.import_module(
        "ausc.perception.robot.controller_launch_activation"
    )

    release, envelope, pair, _ = _write_release(tmp_path / "canonical", module)
    contract = release / module.CONTRACT_FILE
    contract.chmod(0o644)
    contract.write_bytes(json.dumps(envelope, indent=2, sort_keys=True).encode("utf-8"))
    contract.chmod(0o444)
    with pytest.raises(module.ControllerLaunchActivationError, match="canonical"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=pair,
        )

    release, envelope, pair, _ = _write_release(tmp_path / "staging", module)
    release.chmod(0o755)
    (release / module.STAGING_MARKER).write_text("in progress", encoding="utf-8")
    release.chmod(0o555)
    with pytest.raises(module.ControllerLaunchActivationError, match="STAGING"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=pair,
        )

    release, envelope, pair, _ = _write_release(tmp_path / "complete", module)
    complete = release / module.COMPLETE_MARKER
    complete.chmod(0o644)
    complete.write_bytes(_canonical({"status": "COMPLETE"}))
    complete.chmod(0o444)
    with pytest.raises(module.ControllerLaunchActivationError, match="keys|COMPLETE"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=pair,
        )


def test_exact_tree_modes_and_no_symlinks_are_fail_closed(tmp_path: Path) -> None:
    module = importlib.import_module(
        "ausc.perception.robot.controller_launch_activation"
    )

    release, envelope, pair, _ = _write_release(tmp_path / "extra", module)
    release.chmod(0o755)
    (release / "unapproved.txt").write_text("extra", encoding="utf-8")
    release.chmod(0o555)
    with pytest.raises(module.ControllerLaunchActivationError, match="allowlist"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=pair,
        )

    release, envelope, pair, _ = _write_release(tmp_path / "mode", module)
    (release / module.FOLLOWER_CONFIG_PATH).chmod(0o644)
    with pytest.raises(module.ControllerLaunchActivationError, match="mode"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=pair,
        )

    release, envelope, pair, _ = _write_release(tmp_path / "symlink", module)
    victim = release / module.FOLLOWER_CONFIG_PATH
    victim.parent.chmod(0o755)
    victim.chmod(0o644)
    victim.unlink()
    victim.symlink_to(release / module.LEADER_CONFIG_PATH)
    victim.parent.chmod(0o555)
    with pytest.raises(module.ControllerLaunchActivationError, match="symlink"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=pair,
        )

    release, _, pair, blobs = _write_release(tmp_path / "traversal", module)
    profile = _profile(module, pair, blobs)
    profile["artifacts"][0]["path"] = "../controller"
    with pytest.raises(module.ControllerLaunchActivationError, match="path"):
        module.build_controller_launch_envelope(profile, openarm_pair=pair)


def test_pair_v2_cross_binding_rejects_other_pair_or_bound_artifact(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "ausc.perception.robot.controller_launch_activation"
    )
    release, envelope, pair, blobs = _write_release(tmp_path, module)

    other_pair = replace(
        pair,
        release_id="openarm_pair_v2-" + "9" * 64,
        contract_sha256="9" * 64,
    )
    with pytest.raises(module.ControllerLaunchActivationError, match="cross-bind"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=other_pair,
        )

    wrong_binary_pair = replace(pair, candidate_binary_sha256="8" * 64)
    profile = _profile(module, wrong_binary_pair, blobs)
    # Preserve the artifact's real bytes while claiming another pair binary.
    for artifact in profile["artifacts"]:
        if artifact["path"] == module.CONTROLLER_PATH:
            artifact["sha256"] = _sha(blobs[module.CONTROLLER_PATH])
    with pytest.raises(module.ControllerLaunchActivationError, match="candidate binary"):
        module.build_controller_launch_envelope(
            profile,
            openarm_pair=wrong_binary_pair,
        )


def test_direct_elf_argv_rejects_shell_python_scripts_and_wrong_mapping(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "ausc.perception.robot.controller_launch_activation"
    )
    blobs = _fixture_bytes(module)
    pair = _pair(
        controller_sha=_sha(blobs[module.CONTROLLER_PATH]),
        follower_config_sha=_sha(blobs[module.FOLLOWER_CONFIG_PATH]),
        urdf_sha=_sha(blobs[module.LEADER_URDF_PATH]),
    )
    for first in ("bash", "sh", "python3", "controller/run.sh"):
        profile = _profile(module, pair, blobs)
        profile["argv"][0] = first
        with pytest.raises(module.ControllerLaunchActivationError, match="directly"):
            module.build_controller_launch_envelope(profile, openarm_pair=pair)
    for mapping in (
        ["right_arm", "can1", "can3"],
        ["left_arm", "can0", "can2"],
        ["left_arm", "can3", "can1"],
    ):
        profile = _profile(module, pair, blobs)
        profile["argv"][-3:] = mapping
        with pytest.raises(module.ControllerLaunchActivationError, match="left can1"):
            module.build_controller_launch_envelope(profile, openarm_pair=pair)

    script_blobs = dict(blobs)
    script_blobs[module.CONTROLLER_PATH] = b"#!/bin/sh\nexit 0\n"
    release, envelope, script_pair, _ = _write_release(
        tmp_path / "script",
        module,
        blobs=script_blobs,
    )
    with pytest.raises(module.ControllerLaunchActivationError, match="direct ELF"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=script_pair,
        )


def test_loaded_contract_uses_clean_frozen_env_despite_hostile_ambient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        "ausc.perception.robot.controller_launch_activation"
    )
    monkeypatch.setenv("GRIPPER_HOLD_POS", "-1.0")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/hostile.so")
    monkeypatch.setenv("PYTHONPATH", "/tmp/hostile")
    release, envelope, pair, _ = _write_release(tmp_path, module)
    loaded = module.load_controller_launch_activation(
        release,
        trusted_root=release.parent,
        expected_contract_sha256=envelope["contract_sha256"],
        openarm_pair=pair,
    )
    assert loaded.argv[0] == str(release / module.CONTROLLER_PATH)
    assert loaded.argv[-3:] == ("left_arm", "can1", "can3")
    assert "LD_PRELOAD" not in loaded.environment
    assert "PYTHONPATH" not in loaded.environment
    assert loaded.environment["GRIPPER_HOLD_POS"] == "0.0"
    assert "make_injected_controller_launcher" not in module.__all__
    assert not hasattr(module, "make_injected_controller_launcher")


def test_post_load_artifact_mutation_is_rejected_by_dormant_reverification(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "ausc.perception.robot.controller_launch_activation"
    )
    release, envelope, pair, _ = _write_release(tmp_path, module)
    loaded = module.load_controller_launch_activation(
        release,
        trusted_root=release.parent,
        expected_contract_sha256=envelope["contract_sha256"],
        openarm_pair=pair,
    )
    controller = release / module.CONTROLLER_PATH
    controller.chmod(0o755)
    controller.write_bytes(b"\x7fELFmutated-after-load")
    controller.chmod(0o555)
    with pytest.raises(module.ControllerLaunchActivationError, match="SHA-256|size"):
        module.verify_controller_launch_activation(loaded)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema", "forged-schema"),
        ("release_id", "controller_launch_v1-" + "f" * 64),
        ("state", "FORGED"),
        ("open_target_status", "FORGED"),
        ("openarm_pair_release_id", "openarm_pair_v2-" + "e" * 64),
        ("openarm_pair_contract_sha256", "e" * 64),
    ),
)
def test_dormant_reverification_rejects_forged_loaded_metadata(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    module = importlib.import_module(
        "ausc.perception.robot.controller_launch_activation"
    )
    release, envelope, pair, _ = _write_release(tmp_path, module)
    loaded = module.load_controller_launch_activation(
        release,
        trusted_root=release.parent,
        expected_contract_sha256=envelope["contract_sha256"],
        openarm_pair=pair,
    )

    forged = replace(loaded, **{field: replacement})
    with pytest.raises(
        module.ControllerLaunchActivationError,
        match="changed after it was loaded",
    ):
        module.verify_controller_launch_activation(forged)


def test_dormant_reverification_rejects_forged_trusted_root(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "ausc.perception.robot.controller_launch_activation"
    )
    release, envelope, pair, _ = _write_release(tmp_path / "trusted", module)
    loaded = module.load_controller_launch_activation(
        release,
        trusted_root=release.parent,
        expected_contract_sha256=envelope["contract_sha256"],
        openarm_pair=pair,
    )

    forged = replace(loaded, trusted_root=release)
    with pytest.raises(module.ControllerLaunchActivationError):
        module.verify_controller_launch_activation(forged)


def test_trusted_root_rejects_symlinked_or_writable_replacement_surfaces(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "ausc.perception.robot.controller_launch_activation"
    )
    release, envelope, pair, _ = _write_release(tmp_path / "trusted", module)

    alias = tmp_path / "trusted-alias"
    alias.symlink_to(release.parent, target_is_directory=True)
    with pytest.raises(module.ControllerLaunchActivationError, match="symlink"):
        module.load_controller_launch_activation(
            alias / release.name,
            trusted_root=alias,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=pair,
        )

    release.parent.chmod(0o755)
    with pytest.raises(module.ControllerLaunchActivationError, match="writable"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=pair,
        )
    release.parent.chmod(0o555)

    release.chmod(0o755)
    with pytest.raises(module.ControllerLaunchActivationError, match="writable"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=pair,
        )
    release.chmod(0o555)

    config_dir = release / "config"
    config_dir.chmod(0o755)
    with pytest.raises(module.ControllerLaunchActivationError, match="writable"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=pair,
        )


def test_extra_keys_in_envelope_profile_and_artifact_are_rejected(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "ausc.perception.robot.controller_launch_activation"
    )
    release, envelope, pair, blobs = _write_release(tmp_path / "envelope", module)
    contract = release / module.CONTRACT_FILE
    contract.chmod(0o644)
    envelope["comment"] = "unapproved"
    contract.write_bytes(module.canonical_json_bytes(envelope))
    contract.chmod(0o444)
    with pytest.raises(module.ControllerLaunchActivationError, match="wrong keys"):
        module.load_controller_launch_activation(
            release,
            trusted_root=release.parent,
            expected_contract_sha256=envelope["contract_sha256"],
            openarm_pair=pair,
        )

    profile = _profile(module, pair, blobs)
    profile["comment"] = "unapproved"
    with pytest.raises(module.ControllerLaunchActivationError, match="wrong keys"):
        module.build_controller_launch_envelope(profile, openarm_pair=pair)

    profile = _profile(module, pair, blobs)
    profile["artifacts"][0]["comment"] = "unapproved"
    with pytest.raises(module.ControllerLaunchActivationError, match="wrong keys"):
        module.build_controller_launch_envelope(profile, openarm_pair=pair)
