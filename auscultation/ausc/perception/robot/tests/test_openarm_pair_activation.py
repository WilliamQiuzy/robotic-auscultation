"""Fail-closed tests for the activated left OpenArm leader/follower pair."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ausc.perception.robot.openarm_pair_activation import (
    ACTIVATION_SCHEMA,
    COMPLETE_MARKER,
    COMPLETION_SCHEMA,
    CONTRACT_FILE,
    LEFT_JOINT_NAMES,
    PAIR_SCHEMA,
    STAGING_MARKER,
    OpenArmPairActivationError,
    build_activation_envelope,
    build_completion_marker,
    canonical_json_bytes,
    load_activated_openarm_pair,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_1 = "1" * 64
SHA_2 = "2" * 64
SHA_3 = "3" * 64
SHA_4 = "4" * 64
SHA_5 = "5" * 64


def _source_manifest() -> list[dict[str, str]]:
    return [
        {
            "path": "config/follower.yaml",
            "sha256": SHA_3,
        },
        {
            "path": "control/openarm_unilateral_control.cpp",
            "sha256": SHA_1,
        },
        {
            "path": "src/openarm_constants.hpp",
            "sha256": SHA_2,
        },
    ]


def _sdk_manifest() -> list[dict[str, str]]:
    return [
        {
            "kind": "header",
            "path": "include/openarm/openarm.hpp",
            "sha256": SHA_4,
        },
        {
            "kind": "library",
            "path": "lib/libopenarm.so",
            "sha256": SHA_5,
        },
    ]


def _profile() -> dict[str, object]:
    source_manifest = _source_manifest()
    sdk_manifest = _sdk_manifest()
    return {
        "schema": PAIR_SCHEMA,
        "side": "left",
        "controller_mode": "unilateral",
        "leader": {
            "role": "leader",
            "can_interface": "can1",
            "adapter": {
                "id_path": "pci-0000:00:14.0-usb-0:1.2:1.0",
                "serial": "OPENARM-LEADER-TEST",
                "channel": "0",
            },
            "expected_motor_ids": {
                "arm_send_can_ids": [1, 2, 3, 4, 5, 6, 7],
                "arm_receive_can_ids": [17, 18, 19, 20, 21, 22, 23],
                "gripper_send_can_id": 8,
                "gripper_receive_can_id": 24,
            },
        },
        "follower": {
            "role": "follower",
            "can_interface": "can3",
            "adapter": {
                "id_path": "pci-0000:00:14.0-usb-0:1.4:1.0",
                "serial": "OPENARM-FOLLOWER-TEST",
                "channel": "0",
            },
            "expected_motor_ids": {
                "arm_send_can_ids": [1, 2, 3, 4, 5, 6, 7],
                "arm_receive_can_ids": [17, 18, 19, 20, 21, 22, 23],
                "gripper_send_can_id": 8,
                "gripper_receive_can_id": 24,
            },
        },
        "joint_names": list(LEFT_JOINT_NAMES),
        "urdf_sha256": SHA_A,
        "joint_mapper_header_sha256": SHA_B,
        "controller_source_manifest": source_manifest,
        "controller_source_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(source_manifest)
        ).hexdigest(),
        "follower_config_sha256": SHA_3,
        "openarm_sdk_manifest": sdk_manifest,
        "openarm_sdk_sha256": hashlib.sha256(
            canonical_json_bytes(sdk_manifest)
        ).hexdigest(),
        "candidate_binary_sha256": SHA_C,
        "telemetry_socket": "/run/user/1000/wam-openarm-left-telemetry.sock",
        "camera_profile_sha256": SHA_D,
        "timing_artifact_sha256": SHA_E,
    }


def _write_release(
    root: Path,
    profile: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    envelope = build_activation_envelope(_profile() if profile is None else profile)
    release = root / str(envelope["release_id"])
    release.mkdir()
    contract_raw = canonical_json_bytes(envelope)
    (release / CONTRACT_FILE).write_bytes(contract_raw)
    (release / COMPLETE_MARKER).write_bytes(
        canonical_json_bytes(build_completion_marker(envelope, contract_raw))
    )
    return release, envelope


def _rewrite_complete(release: Path, envelope: dict[str, object]) -> None:
    raw = canonical_json_bytes(envelope)
    (release / CONTRACT_FILE).write_bytes(raw)
    marker = {
        "schema": COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "release_id": envelope["release_id"],
        "contract_sha256": envelope["contract_sha256"],
        "contract_file_sha256": hashlib.sha256(raw).hexdigest(),
    }
    (release / COMPLETE_MARKER).write_bytes(
        canonical_json_bytes(marker)
    )


def test_valid_content_addressed_left_pair_loads_as_immutable_value(tmp_path: Path) -> None:
    release, envelope = _write_release(tmp_path)

    loaded = load_activated_openarm_pair(
        release,
        expected_contract_sha256=str(envelope["contract_sha256"]),
    )

    assert loaded.schema == PAIR_SCHEMA
    assert loaded.release_id == envelope["release_id"]
    assert loaded.contract_sha256 == envelope["contract_sha256"]
    assert loaded.side == "left"
    assert loaded.controller_mode == "unilateral"
    assert loaded.leader.can_interface == "can1"
    assert loaded.follower.can_interface == "can3"
    assert loaded.leader.adapter.id_path.endswith("1.2:1.0")
    assert loaded.follower.adapter.serial == "OPENARM-FOLLOWER-TEST"
    assert loaded.leader.expected_motor_ids.arm_send_can_ids == tuple(range(1, 8))
    assert loaded.leader.expected_motor_ids.arm_receive_can_ids == tuple(range(17, 24))
    assert loaded.leader.expected_motor_ids.gripper_send_can_id == 8
    assert loaded.leader.expected_motor_ids.gripper_receive_can_id == 24
    assert loaded.joint_names == LEFT_JOINT_NAMES
    assert loaded.controller_source_manifest[1].path == (
        "control/openarm_unilateral_control.cpp"
    )
    assert loaded.follower_config_sha256 == SHA_3
    assert loaded.openarm_sdk_manifest[0].kind == "header"
    assert loaded.openarm_sdk_manifest[1].path == "lib/libopenarm.so"
    assert loaded.openarm_sdk_sha256 == hashlib.sha256(
        canonical_json_bytes(_sdk_manifest())
    ).hexdigest()
    assert loaded.joint_mapper_sha256 == loaded.joint_mapper_header_sha256
    assert loaded.pair_contract_sha256 == loaded.contract_sha256
    assert loaded.controller_binary_sha256 == loaded.candidate_binary_sha256
    with pytest.raises(AttributeError):
        loaded.side = "right"  # type: ignore[misc]


def test_canonical_json_and_release_id_are_deterministic() -> None:
    profile = _profile()
    shuffled = {key: profile[key] for key in reversed(profile)}

    first = build_activation_envelope(profile)
    second = build_activation_envelope(shuffled)

    assert first == second
    assert first["schema"] == ACTIVATION_SCHEMA
    assert PAIR_SCHEMA == "wam_openarm_pair_contract_v2"
    assert ACTIVATION_SCHEMA == "wam_openarm_pair_activation_v2"
    assert COMPLETION_SCHEMA == "wam_openarm_pair_completion_v2"
    assert first["release_id"] == (
        "openarm_pair_v2-" + str(first["contract_sha256"])
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    (
        (("side",), "right"),
        (("controller_mode",), "gravity_comp"),
        (("leader", "can_interface"), "can0"),
        (("leader", "can_interface"), "can2"),
        (("follower", "can_interface"), "can0"),
        (("follower", "can_interface"), "can2"),
        (("leader", "role"), "follower"),
        (("follower", "role"), "leader"),
        (("joint_names",), list(reversed(LEFT_JOINT_NAMES))),
        (("leader", "expected_motor_ids", "arm_send_can_ids"), [0, 1, 2, 3, 4, 5, 6]),
        (("follower", "expected_motor_ids", "arm_receive_can_ids"), [1, 2, 3, 4, 5, 6, 7]),
        (("leader", "expected_motor_ids", "gripper_send_can_id"), 9),
        (("follower", "expected_motor_ids", "gripper_receive_can_id"), 25),
    ),
)
def test_formal_profile_rejects_every_non_left_or_noncanonical_mapping(
    path: tuple[str, ...],
    bad_value: object,
) -> None:
    profile = _profile()
    target: dict[str, object] = profile
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = bad_value

    with pytest.raises(OpenArmPairActivationError):
        build_activation_envelope(profile)


@pytest.mark.parametrize(
    "field",
    (
        "urdf_sha256",
        "joint_mapper_header_sha256",
        "controller_source_manifest_sha256",
        "follower_config_sha256",
        "openarm_sdk_sha256",
        "candidate_binary_sha256",
        "camera_profile_sha256",
        "timing_artifact_sha256",
    ),
)
@pytest.mark.parametrize("bad_hash", ("0" * 63, "G" * 64, "A" * 64, 7, None))
def test_every_declared_top_level_hash_must_be_lowercase_sha256(
    field: str,
    bad_hash: object,
) -> None:
    profile = _profile()
    profile[field] = bad_hash

    with pytest.raises(OpenArmPairActivationError, match="SHA-256"):
        build_activation_envelope(profile)


def test_source_manifest_is_canonical_unique_and_hash_bound() -> None:
    profile = _profile()
    profile["controller_source_manifest_sha256"] = SHA_F
    with pytest.raises(OpenArmPairActivationError, match="source manifest"):
        build_activation_envelope(profile)

    profile = _profile()
    manifest = profile["controller_source_manifest"]
    assert isinstance(manifest, list)
    manifest.append(copy.deepcopy(manifest[0]))
    profile["controller_source_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    with pytest.raises(OpenArmPairActivationError, match="duplicate"):
        build_activation_envelope(profile)

    profile = _profile()
    manifest = profile["controller_source_manifest"]
    assert isinstance(manifest, list)
    manifest[0]["sha256"] = "Z" * 64
    profile["controller_source_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    with pytest.raises(OpenArmPairActivationError, match="SHA-256"):
        build_activation_envelope(profile)


def test_follower_config_pin_must_match_exact_source_manifest_entry() -> None:
    profile = _profile()
    profile["follower_config_sha256"] = SHA_F
    with pytest.raises(OpenArmPairActivationError, match="follower config"):
        build_activation_envelope(profile)

    profile = _profile()
    manifest = profile["controller_source_manifest"]
    assert isinstance(manifest, list)
    del manifest[0]
    profile["controller_source_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    with pytest.raises(OpenArmPairActivationError, match="config/follower.yaml"):
        build_activation_envelope(profile)

    profile = _profile()
    manifest = profile["controller_source_manifest"]
    assert isinstance(manifest, list)
    manifest[0]["path"] = "config/Follower.yaml"
    profile["controller_source_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    with pytest.raises(OpenArmPairActivationError, match="config/follower.yaml"):
        build_activation_envelope(profile)


def test_openarm_sdk_manifest_is_canonical_and_hash_bound() -> None:
    profile = _profile()
    profile["openarm_sdk_sha256"] = SHA_F
    with pytest.raises(OpenArmPairActivationError, match="SDK manifest SHA-256"):
        build_activation_envelope(profile)

    profile = _profile()
    manifest = profile["openarm_sdk_manifest"]
    assert isinstance(manifest, list)
    manifest.reverse()
    profile["openarm_sdk_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    with pytest.raises(OpenArmPairActivationError, match="sorted"):
        build_activation_envelope(profile)

    profile = _profile()
    manifest = profile["openarm_sdk_manifest"]
    assert isinstance(manifest, list)
    manifest[1]["kind"] = "header"
    profile["openarm_sdk_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    with pytest.raises(OpenArmPairActivationError, match="header and library"):
        build_activation_envelope(profile)

    profile = _profile()
    manifest = profile["openarm_sdk_manifest"]
    assert isinstance(manifest, list)
    manifest.append(copy.deepcopy(manifest[0]))
    profile["openarm_sdk_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    with pytest.raises(OpenArmPairActivationError, match="duplicate"):
        build_activation_envelope(profile)


def test_legacy_v1_profile_activation_release_and_completion_are_non_production(
    tmp_path: Path,
) -> None:
    profile = _profile()
    profile["schema"] = "wam_openarm_pair_contract_v1"
    with pytest.raises(OpenArmPairActivationError, match="legacy.*non-production"):
        build_activation_envelope(profile)

    envelope = build_activation_envelope(_profile())
    legacy_activation = copy.deepcopy(envelope)
    legacy_activation["schema"] = "wam_openarm_pair_activation_v1"
    with pytest.raises(OpenArmPairActivationError, match="legacy.*non-production"):
        build_completion_marker(
            legacy_activation,
            canonical_json_bytes(legacy_activation),
        )

    legacy_release = copy.deepcopy(envelope)
    legacy_release["release_id"] = (
        "openarm_pair-" + str(legacy_release["contract_sha256"])
    )
    with pytest.raises(OpenArmPairActivationError, match="v2"):
        build_completion_marker(
            legacy_release,
            canonical_json_bytes(legacy_release),
        )

    release, complete_envelope = _write_release(tmp_path)
    raw = (release / CONTRACT_FILE).read_bytes()
    marker = build_completion_marker(complete_envelope, raw)
    marker["schema"] = "wam_openarm_pair_completion_v1"
    (release / COMPLETE_MARKER).write_bytes(canonical_json_bytes(marker))
    with pytest.raises(OpenArmPairActivationError, match="legacy.*non-production"):
        load_activated_openarm_pair(
            release,
            expected_contract_sha256=str(complete_envelope["contract_sha256"]),
        )


def test_missing_extra_or_noncanonical_contract_fields_fail_closed(tmp_path: Path) -> None:
    for mutation in ("missing", "extra"):
        profile = _profile()
        if mutation == "missing":
            profile.pop("candidate_binary_sha256")
        else:
            profile["comment"] = "looks safe"
        with pytest.raises(OpenArmPairActivationError, match="keys"):
            build_activation_envelope(profile)

    release, envelope = _write_release(tmp_path)
    pretty = json.dumps(envelope, indent=2, sort_keys=True).encode("utf-8")
    (release / CONTRACT_FILE).write_bytes(pretty)
    with pytest.raises(OpenArmPairActivationError, match="canonical"):
        load_activated_openarm_pair(
            release,
            expected_contract_sha256=str(envelope["contract_sha256"]),
        )


def test_contract_content_drift_and_fully_rehashed_drift_both_fail_closed(
    tmp_path: Path,
) -> None:
    release, envelope = _write_release(tmp_path)
    pinned = str(envelope["contract_sha256"])
    tampered = copy.deepcopy(envelope)
    tampered["profile"]["candidate_binary_sha256"] = SHA_F
    _rewrite_complete(release, tampered)
    with pytest.raises(OpenArmPairActivationError, match="contract hash"):
        load_activated_openarm_pair(release, expected_contract_sha256=pinned)

    # An attacker/operator may recompute every self-attested field, but cannot
    # silently change the independently pinned activation identity.
    changed_profile = copy.deepcopy(_profile())
    changed_profile["candidate_binary_sha256"] = SHA_F
    changed = build_activation_envelope(changed_profile)
    changed_dir = tmp_path / str(changed["release_id"])
    changed_dir.mkdir()
    changed_raw = canonical_json_bytes(changed)
    (changed_dir / CONTRACT_FILE).write_bytes(changed_raw)
    (changed_dir / COMPLETE_MARKER).write_bytes(
        canonical_json_bytes(build_completion_marker(changed, changed_raw))
    )
    with pytest.raises(OpenArmPairActivationError, match="pinned"):
        load_activated_openarm_pair(changed_dir, expected_contract_sha256=pinned)


def test_staging_missing_or_drifted_completion_marker_fails_closed(tmp_path: Path) -> None:
    release, envelope = _write_release(tmp_path)
    pinned = str(envelope["contract_sha256"])

    (release / STAGING_MARKER).write_text("in progress", encoding="utf-8")
    with pytest.raises(OpenArmPairActivationError, match="STAGING"):
        load_activated_openarm_pair(release, expected_contract_sha256=pinned)
    (release / STAGING_MARKER).unlink()

    (release / COMPLETE_MARKER).unlink()
    with pytest.raises(OpenArmPairActivationError, match="COMPLETE"):
        load_activated_openarm_pair(release, expected_contract_sha256=pinned)

    contract_raw = (release / CONTRACT_FILE).read_bytes()
    marker = build_completion_marker(envelope, contract_raw)
    marker["contract_sha256"] = SHA_F
    (release / COMPLETE_MARKER).write_bytes(canonical_json_bytes(marker))
    with pytest.raises(OpenArmPairActivationError, match="completion"):
        load_activated_openarm_pair(release, expected_contract_sha256=pinned)


def test_release_name_adapter_identity_socket_and_exact_file_set_are_bound(
    tmp_path: Path,
) -> None:
    release, envelope = _write_release(tmp_path)
    pinned = str(envelope["contract_sha256"])
    wrong_name = tmp_path / "openarm_pair-wrong"
    release.rename(wrong_name)
    with pytest.raises(OpenArmPairActivationError, match="release"):
        load_activated_openarm_pair(wrong_name, expected_contract_sha256=pinned)

    for mutation in ("missing_id_path", "same_adapter", "relative_socket"):
        profile = _profile()
        if mutation == "missing_id_path":
            profile["leader"]["adapter"]["id_path"] = ""  # type: ignore[index]
        elif mutation == "same_adapter":
            profile["follower"]["adapter"] = copy.deepcopy(  # type: ignore[index]
                profile["leader"]["adapter"]  # type: ignore[index]
            )
        else:
            profile["telemetry_socket"] = "wam-openarm-left.sock"
        with pytest.raises(OpenArmPairActivationError):
            build_activation_envelope(profile)

    release2, envelope2 = _write_release(tmp_path)
    (release2 / "unapproved.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(OpenArmPairActivationError, match="files"):
        load_activated_openarm_pair(
            release2,
            expected_contract_sha256=str(envelope2["contract_sha256"]),
        )


def test_completion_marker_schema_and_contract_file_hash_are_exact(tmp_path: Path) -> None:
    release, envelope = _write_release(tmp_path)
    raw = (release / CONTRACT_FILE).read_bytes()
    marker = build_completion_marker(envelope, raw)

    assert marker == {
        "schema": COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "release_id": envelope["release_id"],
        "contract_sha256": envelope["contract_sha256"],
        "contract_file_sha256": hashlib.sha256(raw).hexdigest(),
    }

    marker["contract_file_sha256"] = SHA_F
    (release / COMPLETE_MARKER).write_bytes(canonical_json_bytes(marker))
    with pytest.raises(OpenArmPairActivationError, match="completion"):
        load_activated_openarm_pair(
            release,
            expected_contract_sha256=str(envelope["contract_sha256"]),
        )
