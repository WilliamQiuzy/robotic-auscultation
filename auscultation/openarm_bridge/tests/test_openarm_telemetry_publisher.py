from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from ausc.perception.robot.robot_state_stream import (
    RobotStateContract,
    parse_control_envelope_datagram,
)


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = BRIDGE_ROOT / "openarm_telemetry_publisher.cpp"
HEADER = BRIDGE_ROOT / "openarm_telemetry_publisher.hpp"
HARNESS = Path(__file__).with_name("publisher_harness.cpp")
APPLIER = BRIDGE_ROOT / "apply_openarm_telemetry_patch.py"
BASELINE = BRIDGE_ROOT / "lingbox_openarm_baseline.json"
PATCH = BRIDGE_ROOT / "openarm_unilateral_control.telemetry.patch"
SDK_STATUS_HEADER = BRIDGE_ROOT / "openarm_can_tx_enqueue_status.hpp"
SDK_PATCH = BRIDGE_ROOT / "openarm_can_1.2.2.tx_status.patch"
SDK_APPLIER = BRIDGE_ROOT / "apply_openarm_can_tx_status_patch.py"
SDK_HARNESS = Path(__file__).with_name("openarm_can_tx_status_harness.cpp")
FEEDBACK_HARNESS = Path(__file__).with_name(
    "openarm_can_feedback_liveness_harness.cpp"
)
FEEDBACK_HEADER = BRIDGE_ROOT / "openarm_can_feedback_liveness.hpp"
SDK_FEEDBACK_PATCH = BRIDGE_ROOT / "openarm_can_1.2.2.feedback_liveness.patch"
SDK_FEEDBACK_APPLIER = BRIDGE_ROOT / "apply_openarm_can_feedback_patch.py"
INIT_RESET_PATCH = BRIDGE_ROOT / "openarm_init.full_reset.patch"
INIT_RESET_APPLIER = BRIDGE_ROOT / "apply_openarm_init_reset_patch.py"
INIT_RESET_BASELINE = BRIDGE_ROOT / "lingbox_openarm_init_reset_baseline.json"

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_1 = "1" * 64
SHA_2 = "2" * 64
SHA_3 = "3" * 64
JOINT_NAMES = tuple(f"openarm_left_joint{i}" for i in range(1, 8))


def _contract() -> RobotStateContract:
    return RobotStateContract(
        side="left",
        controller_mode="unilateral",
        joint_names=JOINT_NAMES,
        joint_position_limits_rad=tuple((-2.0, 2.0) for _ in range(7)),
        leader_can_interface="can1",
        follower_can_interface="can3",
        urdf_sha256=SHA_A,
        joint_mapper_sha256=SHA_B,
        camera_profile_sha256=SHA_C,
        timing_artifact_sha256=SHA_D,
        pair_contract_sha256=SHA_E,
        controller_binary_sha256=SHA_F,
        max_abs_joint_velocity_rad_s=8.0,
        gripper_motor_position_limits_rad=(-0.20, 0.20),
        controller_source_manifest_sha256=SHA_1,
        follower_config_sha256=SHA_2,
        openarm_sdk_sha256=SHA_3,
    )


def _compile_harness(tmp_path: Path) -> Path:
    assert HEADER.is_file(), "versioned C++ publisher header is missing"
    assert SOURCE.is_file(), "versioned C++ publisher source is missing"
    binary = tmp_path / "publisher_harness"
    result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-pthread",
            "-I",
            str(BRIDGE_ROOT),
            str(SOURCE),
            str(HARNESS),
            "-o",
            str(binary),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return binary


def _compile_sdk_status_harness(tmp_path: Path) -> Path:
    binary = tmp_path / "openarm_can_tx_status_harness"
    result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-I",
            str(BRIDGE_ROOT),
            str(SDK_HARNESS),
            "-o",
            str(binary),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return binary


def _compile_feedback_harness(tmp_path: Path) -> Path:
    binary = tmp_path / "openarm_can_feedback_liveness_harness"
    result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-I",
            str(BRIDGE_ROOT),
            str(FEEDBACK_HARNESS),
            "-o",
            str(binary),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return binary


def _short_socket_path(tmp_path: Path, label: str) -> Path:
    digest = hashlib.sha256(str(tmp_path).encode("utf-8")).hexdigest()[:12]
    return Path("/tmp") / f"wam-{label}-{digest}.sock"


@contextmanager
def _bound_receiver(path: Path, *, receive_buffer: int | None = None):
    path.unlink(missing_ok=True)
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        if receive_buffer is not None:
            receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer)
        receiver.bind(str(path))
        receiver.settimeout(2.0)
        yield receiver
    finally:
        receiver.close()
        path.unlink(missing_ok=True)


def test_cpp_v4_packet_is_exactly_compatible_with_strict_python_contract(
    tmp_path: Path,
) -> None:
    binary = _compile_harness(tmp_path)
    result = subprocess.run(
        [str(binary), "serialize_single", "/tmp/not-used.sock"],
        check=True,
        timeout=2.0,
        stdout=subprocess.PIPE,
    )
    payload = result.stdout.strip()

    envelope = parse_control_envelope_datagram(payload, _contract())
    raw = json.loads(payload)
    assert raw["schema"] == "wam_openarm_control_envelope_v4"
    assert raw["publisher_nominal_hz"] == 100
    assert raw["control_nominal_hz"] == 1000
    assert raw["side"] == "left"
    assert raw["leader_can_interface"] == "can1"
    assert raw["follower_can_interface"] == "can3"
    assert raw["send_status_observable"] is True
    assert raw["send_status_reason"] == "openarm_can_1_2_2_full_frame_write_results"
    assert raw["send_status_semantics"] == (
        "linux_socketcan_full_frame_write_tx_enqueue_not_motor_ack"
    )
    assert raw["motor_ack_observable"] is False
    assert raw["feedback_liveness_observable"] is True
    assert raw["feedback_liveness_reason"] == (
        "openarm_can_1_2_2_matching_motor_id_valid_state_decode"
    )
    assert raw["feedback_liveness_semantics"] == (
        "counter_generation_increment_only_after_matching_motor_id_valid_state_decode"
    )
    assert raw["feedback_max_age_ns"] == 20_000_000
    assert raw["feedback_max_gap_ns"] == 20_000_000
    assert raw["ring_overflow_count"] == 0
    assert raw["publisher_rejected_record_count"] == 0
    assert raw["socket_send_failure_count"] == 0
    record = envelope.records[0]
    assert record.control_tick_seq == 100
    assert record.follower_q_rad == pytest.approx(
        tuple(index * 0.01 for index in range(7))
    )
    assert record.follower_qd_rad_s == pytest.approx((0.5,) * 7)
    assert record.action.arm_q == pytest.approx(
        tuple(index * 0.01 + 0.10 for index in range(7))
    )
    assert record.action.arm_dq == pytest.approx((0.4,) * 7)
    assert record.action.arm_tau == pytest.approx((0.0,) * 7)
    assert record.action.gripper_kp == pytest.approx(25.0)
    assert record.action.gripper_q == pytest.approx(-0.080)
    assert record.reference_generation == 7
    assert record.reference_snapshot_valid is True
    assert record.action_parameters_exact is True
    assert record.arm_tx_expected_frame_count == 7
    assert record.arm_tx_full_frame_write_count == 7
    assert record.arm_tx_failed_frame_write_count == 0
    assert record.gripper_tx_expected_frame_count == 1
    assert record.gripper_tx_full_frame_write_count == 1
    assert record.gripper_tx_failed_frame_write_count == 0
    assert record.action_send_status_verified is True
    assert record.feedback_liveness_observable is True
    assert record.follower_arm_state_rx_motor_ids == tuple(range(1, 8))
    assert record.follower_arm_state_rx_counters == tuple(range(101, 108))
    assert record.follower_arm_state_rx_generations == tuple(range(101, 108))
    assert record.follower_gripper_state_rx_motor_id == 8
    assert record.follower_gripper_state_rx_counter == 108
    assert record.feedback_snapshot_monotonic_ns == 1_000_000_000


def test_feedback_tracker_only_counts_matching_successful_decode(tmp_path: Path) -> None:
    assert FEEDBACK_HEADER.is_file(), "versioned feedback liveness header is missing"
    binary = _compile_feedback_harness(tmp_path)
    result = subprocess.run(
        [str(binary)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_publisher_batches_every_exact_1khz_record_without_downsampling(
    tmp_path: Path,
) -> None:
    binary = _compile_harness(tmp_path)
    result = subprocess.run(
        [str(binary), "serialize_batch", "/tmp/not-used.sock"],
        check=True,
        timeout=2.0,
        stdout=subprocess.PIPE,
    )
    envelopes = [
        parse_control_envelope_datagram(line, _contract())
        for line in result.stdout.splitlines()
    ]
    records = [
        record for envelope in envelopes for record in envelope.records
    ]

    assert [envelope.envelope_seq for envelope in envelopes] == list(
        range(len(envelopes))
    )
    assert [record.control_tick_seq for record in records] == list(range(100, 125))
    assert [record.observation_monotonic_ns for record in records] == [
        1_000_000_000 + index * 1_000_000 for index in range(25)
    ]
    assert all(len(envelope.records) <= 16 for envelope in envelopes)


def test_reference_epoch_rejects_empty_in_progress_and_changed_reads(
    tmp_path: Path,
) -> None:
    binary = _compile_harness(tmp_path)
    result = subprocess.run(
        [str(binary), "reference_epoch", "/tmp/not-used.sock"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=2.0,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_control_thread_entrypoint_is_pod_ring_push_only() -> None:
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    entrypoint = source.split(
        "void OpenArmTelemetryPublisher::try_enqueue", 1
    )[1].split("void OpenArmTelemetryPublisher::publisher_loop", 1)[0]

    assert "std::is_trivially_copyable_v<ControlTickRecord>" in header
    assert "SpscRing<ControlTickRecord" in header
    assert "ring_.try_push(record)" in entrypoint
    assert "sendto" not in entrypoint
    assert "JsonBuffer" not in entrypoint
    assert "std::thread" not in entrypoint
    assert "publisher_loop" in source
    assert "lower_publisher_thread_priority" in source


def test_openarm_can_status_aggregate_requires_every_expected_full_write(
    tmp_path: Path,
) -> None:
    binary = _compile_sdk_status_harness(tmp_path)
    result = subprocess.run(
        [str(binary)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_openarm_can_1_2_2_patch_propagates_underlying_bool_write_results() -> None:
    patch = SDK_PATCH.read_text(encoding="utf-8")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert baseline["openarm_can_version"] == "1.2.2"
    assert baseline["openarm_can_git_commit"] == (
        "3363f9879f8947f79db179e4e91f6dbf78f44a6f"
    )
    assert "bool DMDeviceCollection::send_command_to_device" in patch
    assert "return can_socket_.write_canfd_frame(frame);" in patch
    assert "return can_socket_.write_can_frame(frame);" in patch
    assert "MITControlSendStatus DMDeviceCollection::mit_control_all" in patch
    assert "status.observe_full_frame_write(mit_control_one" in patch


def test_openarm_can_1_2_2_feedback_patch_hooks_only_valid_matching_state_decode() -> None:
    patch = SDK_FEEDBACK_PATCH.read_text(encoding="utf-8")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert baseline["openarm_can_motor_header_source_sha256"] == (
        "cf0bd92628814b78286ff517c4c911444fd23610c0ba5850309bcb0524d3b4ef"
    )
    assert baseline["openarm_can_motor_source_sha256"] == (
        "2e3e99a9e2ef5a06081f686c1799e78fec63dd4fef210b5ea44e9640bd96c52a"
    )
    assert baseline["openarm_can_motor_device_source_sha256"] == (
        "44d37707bcdf33af54e78f1f7702180b4704bc00d9b5b7897a32a8851636092f"
    )
    assert baseline["openarm_can_cmake_source_sha256"] == (
        "d19af3513c27896dc3ce4f86e589ffbd63c8aae75bd372470cc0bcf563ed0551"
    )
    assert baseline["openarm_can_tx_status_header_output_sha256"] == (
        "c837f7d7aed0c3ff451c91f79b46a5cdfeadae67cce708a6ce45eda71864d596"
    )
    feedback_applier = SDK_FEEDBACK_APPLIER.read_text(encoding="utf-8")
    assert "TX_PREREQUISITES" in feedback_applier
    assert "openarm_can_tx_collection_header_output_sha256" in feedback_applier
    assert "openarm_can_tx_collection_source_output_sha256" in feedback_applier
    assert "--- a/CMakeLists.txt" in patch
    assert "include/openarm/damiao_motor/openarm_can_tx_enqueue_status.hpp" in patch
    assert "include/openarm/damiao_motor/openarm_can_feedback_liveness.hpp" in patch
    assert "CLOCK_MONOTONIC" in patch
    assert "frame.can_id == motor_.get_recv_can_id() && result.valid" in patch
    assert "if (frame.can_id != motor_.get_recv_can_id())" in patch
    assert patch.count("observe_successfully_decoded_matching_state") >= 3
    device_patch = patch.split(
        "--- a/src/openarm/damiao_motor/dm_motor_device.cpp", 1
    )[1]
    for hunk in device_patch.split("@@")[2::2]:
        if "observe_successfully_decoded_matching_state" in hunk:
            assert hunk.index("motor_.update_state") < hunk.index(
                "observe_successfully_decoded_matching_state"
            )


def test_openarm_can_feedback_applier_rejects_wrong_1_2_2_base_without_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "openarm_can"
    paths = (
        root / "CMakeLists.txt",
        root / "include/openarm/damiao_motor/dm_motor.hpp",
        root / "src/openarm/damiao_motor/dm_motor.cpp",
        root / "src/openarm/damiao_motor/dm_motor_device.cpp",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// wrong OpenArm CAN 1.2.2 source\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SDK_FEEDBACK_APPLIER),
            "--openarm-can-root",
            str(root),
            "--check",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
    assert all(path.read_text().startswith("// wrong") for path in paths)
    assert not (
        root
        / "include/openarm/damiao_motor/openarm_can_feedback_liveness.hpp"
    ).exists()


def test_lingbox_openarm_init_reset_patch_preserves_complete_verified_order() -> None:
    patch = INIT_RESET_PATCH.read_text(encoding="utf-8")
    baseline = json.loads(INIT_RESET_BASELINE.read_text(encoding="utf-8"))

    assert baseline["schema"] == "wam_openarm_init_reset_patch_baseline_v1"
    assert baseline["source_git_blob_prefix"] == "7212894"
    assert baseline["output_git_blob_prefix"] == "a6bd0b2"
    assert '#include <openarm/damiao_motor/dm_motor_constants.hpp>' in patch
    ordered = (
        "CallbackMode::IGNORE",
        "refresh_all()",
        "recv_all()",
        "CallbackMode::PARAM",
        "RID::MST_ID",
        "CallbackMode::STATE",
    )
    cursor = patch.index("CallbackMode::IGNORE")
    for item in ordered[1:]:
        cursor = patch.index(item, cursor + 1)
    applier = INIT_RESET_APPLIER.read_text(encoding="utf-8")
    assert '"enable_all()"' in applier
    assert "patch" in applier and '"-F0"' in applier


def test_lingbox_openarm_init_reset_applier_rejects_unknown_base_without_write(
    tmp_path: Path,
) -> None:
    root = tmp_path / "openarm_teleop"
    source = root / "src/openarm_port/openarm_init.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("// wrong init source\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(INIT_RESET_APPLIER),
            "--teleop-root",
            str(root),
            "--check",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "git blob prefix mismatch" in result.stderr
    assert source.read_text(encoding="utf-8") == "// wrong init source\n"


def test_openarm_can_applier_rejects_wrong_1_2_2_base_without_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "openarm_can"
    header = root / "include/openarm/damiao_motor/dm_motor_device_collection.hpp"
    source = root / "src/openarm/damiao_motor/dm_motor_device_collection.cpp"
    socket_header = root / "include/openarm/canbus/can_socket.hpp"
    socket_source = root / "src/openarm/canbus/can_socket.cpp"
    for path in (header, source, socket_header, socket_source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// wrong OpenArm CAN 1.2.2 source\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SDK_APPLIER),
            "--openarm-can-root",
            str(root),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
    assert header.read_text(encoding="utf-8").startswith("// wrong")
    assert not (
        root
        / "include/openarm/damiao_motor/openarm_can_tx_enqueue_status.hpp"
    ).exists()


def test_absent_receiver_never_blocks_or_fails_control_process(
    tmp_path: Path,
) -> None:
    binary = _compile_harness(tmp_path)
    absent_path = _short_socket_path(tmp_path, "absent")
    result = subprocess.run(
        [str(binary), "flood", str(absent_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3.0,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_full_receiver_queue_never_blocks_or_fails_control_process(
    tmp_path: Path,
) -> None:
    binary = _compile_harness(tmp_path)
    socket_path = _short_socket_path(tmp_path, "full")
    with _bound_receiver(socket_path, receive_buffer=1_024):
        result = subprocess.run(
            [str(binary), "flood", str(socket_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3.0,
            check=False,
        )
    assert result.returncode == 0, result.stderr


def test_guarded_applier_rejects_wrong_dirty_base_without_writes(
    tmp_path: Path,
) -> None:
    assert APPLIER.is_file(), "guarded patch applier is missing"
    teleop_root = tmp_path / "openarm_teleop"
    control_dir = teleop_root / "control"
    source_dir = teleop_root / "src" / "controller"
    config_dir = teleop_root / "config"
    control_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    source = control_dir / "openarm_unilateral_control.cpp"
    cmake = teleop_root / "CMakeLists.txt"
    source.write_text("// wrong source\n", encoding="utf-8")
    cmake.write_text("# wrong cmake\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(APPLIER),
            "--teleop-root",
            str(teleop_root),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
    assert source.read_text(encoding="utf-8") == "// wrong source\n"
    assert cmake.read_text(encoding="utf-8") == "# wrong cmake\n"
    assert not (teleop_root / "src" / "wam_openarm_telemetry").exists()


def test_versioned_patch_is_bound_to_v2_left_pair_and_full_control_closure() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    patch = PATCH.read_text(encoding="utf-8")

    assert baseline["schema"] == "wam_openarm_telemetry_patch_baseline_v2"
    assert baseline["teleop_git_commit"] == (
        "b8327e284de0f2edec322e6a1a4a064fd01874e5"
    )
    assert baseline["audited_side"] == "left"
    assert baseline["leader_can_interface"] == "can1"
    assert baseline["follower_can_interface"] == "can3"
    assert baseline["admin_frequency_hz"] == 1000
    assert baseline["control_frequency_hz"] == 1000
    assert baseline["publisher_envelope_frequency_hz"] == 100
    for field in (
        "unilateral_source_sha256",
        "cmake_sha256",
        "control_source_sha256",
        "control_header_sha256",
        "robot_state_header_sha256",
        "joint_converter_header_sha256",
        "follower_config_sha256",
        "openarm_mit_header_sha256",
        "openarm_collection_header_sha256",
        "openarm_static_library_sha256",
    ):
        assert len(baseline[field]) == 64

    assert "ControlTickRecord" in patch
    assert "follower_arm_mit" in patch
    assert "arm_call_return_monotonic_ns" in patch
    assert "FollowerReferenceEpoch" in patch
    prime = patch.index("+        follower_reference_epoch.begin_update();")
    attach = patch.index("+        control_follower->SetFollowerTelemetry(")
    assert prime < attach
    assert (
        "+        follower_reference_epoch.complete_update("
        in patch[prime:attach]
    )
    assert "try_enqueue" in patch
    assert patch.count("get_feedback_liveness_snapshot") == 2
    assert "follower_arm_state_rx_counters" in patch
    assert "follower_gripper_state_rx_counter" in patch
    assert 'arm_side == "left_arm"' in patch
    assert '"can1"' in patch
    assert '"can3"' in patch
    assert "follower_command_q_rad" not in patch
