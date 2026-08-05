from __future__ import annotations

import hashlib
from pathlib import Path
import stat

import pytest

from ausc.perception.robot import controller_launch_activation as activation
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

import publish_controller_launch as publisher


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pair(*, controller_sha: str, follower_config_sha: str, urdf_sha: str):
    motor_ids = ExpectedMotorIds(
        arm_send_can_ids=tuple(range(1, 8)),
        arm_receive_can_ids=tuple(range(17, 24)),
        gripper_send_can_id=8,
        gripper_receive_can_id=24,
    )
    adapter = CanAdapterIdentity(
        id_path="pci-test", serial="PAIR-TEST", channel="0",
    )
    return OpenArmPairActivation(
        schema=PAIR_SCHEMA,
        release_id="openarm_pair_v2-" + "a" * 64,
        contract_sha256="a" * 64,
        side="left",
        controller_mode="unilateral",
        leader=OpenArmCanEndpoint(
            role="leader", can_interface="can1", adapter=adapter,
            expected_motor_ids=motor_ids,
        ),
        follower=OpenArmCanEndpoint(
            role="follower", can_interface="can3", adapter=adapter,
            expected_motor_ids=motor_ids,
        ),
        joint_names=LEFT_JOINT_NAMES,
        urdf_sha256=urdf_sha,
        joint_mapper_header_sha256="b" * 64,
        controller_source_manifest=(
            ControllerSourceEntry(
                path="config/follower.yaml", sha256=follower_config_sha,
            ),
        ),
        controller_source_manifest_sha256="c" * 64,
        follower_config_sha256=follower_config_sha,
        openarm_sdk_manifest=(
            OpenArmSdkEntry(
                kind="header", path="include/openarm.hpp", sha256="d" * 64,
            ),
            OpenArmSdkEntry(
                kind="library", path="lib/libopenarm.a", sha256="e" * 64,
            ),
        ),
        openarm_sdk_sha256="f" * 64,
        candidate_binary_sha256=controller_sha,
        telemetry_socket="/run/user/1000/wam/openarm-left-telemetry.sock",
        camera_profile_sha256="1" * 64,
        timing_artifact_sha256="2" * 64,
    )


def _inputs(tmp_path: Path):
    blobs = {
        activation.CONTROLLER_PATH: b"\x7fELF\x02\x01controller",
        activation.LEADER_CONFIG_PATH: b"leader: true\n",
        activation.FOLLOWER_CONFIG_PATH: b"follower: true\n",
        activation.LEADER_URDF_PATH: b"<robot name='leader'/>\n",
        activation.FOLLOWER_URDF_PATH: b"<robot name='follower'/>\n",
        activation.PREFLIGHT_PATH: b"\x7fELF\x02\x01preflight",
    }
    sources = {}
    for relative, raw in blobs.items():
        path = tmp_path / "inputs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        sources[relative] = path
    pair = _pair(
        controller_sha=_sha(blobs[activation.CONTROLLER_PATH]),
        follower_config_sha=_sha(blobs[activation.FOLLOWER_CONFIG_PATH]),
        urdf_sha=_sha(blobs[activation.LEADER_URDF_PATH]),
    )
    expected = {relative: _sha(raw) for relative, raw in blobs.items()}
    return pair, sources, expected


def test_builds_completed_content_addressed_dormant_release(tmp_path: Path):
    pair, sources, expected = _inputs(tmp_path)

    result = publisher.build_controller_launch_release(
        openarm_pair=pair,
        artifact_sources=sources,
        expected_artifact_sha256=expected,
        releases_root=tmp_path / "published",
    )

    assert result.release_directory.parent == result.trusted_root
    assert result.release_directory.name == result.release_id
    assert result.release_id == "controller_launch_v1-" + result.contract_sha256
    assert stat.S_IMODE(result.trusted_root.stat().st_mode) == 0o555
    assert stat.S_IMODE(result.release_directory.stat().st_mode) == 0o555
    assert not (result.release_directory / activation.STAGING_MARKER).exists()
    loaded = activation.load_controller_launch_activation(
        result.release_directory,
        trusted_root=result.trusted_root,
        expected_contract_sha256=result.contract_sha256,
        openarm_pair=pair,
    )
    assert loaded.state == "DORMANT_HIL_PENDING"
    assert loaded.open_target_status == "HIL_PENDING"
    assert loaded.environment["GRIPPER_DECOUPLE"] == "1"
    assert loaded.environment["GRIPPER_HOLD_POS"] == "0.0"

    repeated = publisher.build_controller_launch_release(
        openarm_pair=pair,
        artifact_sources=sources,
        expected_artifact_sha256=expected,
        releases_root=tmp_path / "published",
    )
    assert repeated == result


def test_rejects_unconfirmed_or_symlink_input_before_publish(tmp_path: Path):
    pair, sources, expected = _inputs(tmp_path)
    bad = dict(expected)
    bad[activation.LEADER_CONFIG_PATH] = "0" * 64

    with pytest.raises(publisher.ControllerLaunchPublishError, match="SHA-256"):
        publisher.build_controller_launch_release(
            openarm_pair=pair,
            artifact_sources=sources,
            expected_artifact_sha256=bad,
            releases_root=tmp_path / "bad-hash",
        )
    assert not (tmp_path / "bad-hash").exists()

    target = sources[activation.PREFLIGHT_PATH]
    alias = target.with_name("preflight-alias")
    alias.symlink_to(target)
    linked = dict(sources)
    linked[activation.PREFLIGHT_PATH] = alias
    with pytest.raises(publisher.ControllerLaunchPublishError, match="symlink"):
        publisher.build_controller_launch_release(
            openarm_pair=pair,
            artifact_sources=linked,
            expected_artifact_sha256=expected,
            releases_root=tmp_path / "bad-link",
        )
    assert not (tmp_path / "bad-link").exists()
