from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import re
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


def _load_telemetry_applier_module():
    spec = importlib.util.spec_from_file_location(
        "wam_openarm_telemetry_patch_applier",
        APPLIER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exact_patch_base_inputs() -> dict[str, bytes]:
    """Reconstruct the complete pinned preimages carried by the patch.

    No untouched-line placeholders are permitted: every old-side byte must be
    present and hash to the baseline's byte-level preimage digest. Full-tree
    provenance is established separately by the sealed predecessor manifest.
    """

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    applier = _load_telemetry_applier_module()
    patch_lines = PATCH.read_text(encoding="utf-8").splitlines()
    files: dict[str, list[str]] = {}
    old_side_without_final_newline: set[str] = set()
    current_path: str | None = None
    current_line = 1
    previous_hunk_prefix: str | None = None
    index = 0
    while index < len(patch_lines):
        line = patch_lines[index]
        if line.startswith("--- a/"):
            current_path = line.removeprefix("--- a/")
            files[current_path] = []
            current_line = 1
            index += 2  # Skip the matching +++ path line.
            continue
        if current_path is not None and line.startswith("@@ "):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
            assert match is not None, line
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            assert old_start == current_line, (
                f"patch omits pinned preimage lines for {current_path}: "
                f"expected old line {current_line}, hunk starts at {old_start}"
            )
            consumed = 0
            index += 1
            while index < len(patch_lines):
                hunk_line = patch_lines[index]
                if hunk_line.startswith("@@ ") or hunk_line.startswith("--- a/"):
                    break
                if hunk_line.startswith(" ") or hunk_line.startswith("-"):
                    files[current_path].append(hunk_line[1:])
                    current_line += 1
                    consumed += 1
                    previous_hunk_prefix = hunk_line[0]
                elif hunk_line.startswith("+"):
                    previous_hunk_prefix = "+"
                elif hunk_line == r"\ No newline at end of file":
                    if previous_hunk_prefix in {" ", "-"}:
                        old_side_without_final_newline.add(current_path)
                else:
                    raise AssertionError(f"unexpected patch line: {hunk_line!r}")
                index += 1
            assert consumed == old_count, (
                current_path, consumed, old_count,
            )
            continue
        index += 1

    payloads = {
        relative: (
            "\n".join(lines)
            + ("" if relative in old_side_without_final_newline else "\n")
        ).encode("utf-8")
        for relative, lines in files.items()
    }
    assert set(payloads) == set(applier.PATCH_INPUTS)
    for relative, hash_field in applier.PATCH_INPUTS.items():
        assert hashlib.sha256(payloads[relative]).hexdigest() == baseline[hash_field]
    return payloads


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


def test_cli_refuses_missing_predecessor_without_candidate_writes(
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
            "--predecessor-manifest",
            str(tmp_path / "missing-predecessor.json"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "required regular file is missing" in result.stderr
    assert source.read_text(encoding="utf-8") == "// wrong source\n"
    assert cmake.read_text(encoding="utf-8") == "# wrong cmake\n"
    assert not (teleop_root / "src" / "wam_openarm_telemetry").exists()


def _write_exact_patch_inputs(root: Path) -> dict[str, bytes]:
    payloads = _exact_patch_base_inputs()
    for relative, payload in payloads.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return payloads


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_full_tree_predecessor_manifest(
    root: Path,
    destination: Path,
) -> tuple[dict, str]:
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "kind": "regular",
                "mode": f"{path.stat().st_mode & 0o777:04o}",
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "allowed_untracked_prefixes": [],
        "build_authorized": False,
        "coverage": "full_tracked_tree",
        "files": files,
        "purpose": "private_candidate_staging",
        "schema": "wam_openarm_teleop_tree_state_v1",
        "sealed": True,
        "state": "init_reset_applied",
        "transition": {
            "kind": "openarm_init_reset_v1",
            "predecessor_manifest_sha256": "0" * 64,
        },
    }
    payload = _canonical_json_bytes(manifest)
    destination.write_bytes(payload)
    return manifest, hashlib.sha256(payload).hexdigest()


def _configure_sealed_predecessor(
    applier,
    root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    manifest_path = tmp_path / "PREDECESSOR_TREE_STATE.json"
    _, manifest_sha256 = _write_full_tree_predecessor_manifest(root, manifest_path)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline.pop("teleop_git_commit", None)
    baseline.pop("teleop_git_tree", None)
    baseline.pop("tracked_tree_clean", None)
    baseline.pop("untracked_paths", None)
    baseline.pop("live_tree_was_dirty", None)
    baseline.update(
        {
            "schema": "wam_openarm_telemetry_patch_baseline_v5",
            "predecessor_manifest_schema": "wam_openarm_teleop_tree_state_v1",
            "predecessor_manifest_state": "init_reset_applied",
            "predecessor_full_tree_manifest_sha256": manifest_sha256,
            "predecessor_full_tree_manifest_status": "COMPLETE",
            "result_manifest_state": "gripper_patch_applied",
        }
    )
    baseline_path = tmp_path / "lingbox_openarm_baseline.v5.json"
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(applier, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(applier, "SUPPORT_FILES", {})
    monkeypatch.setattr(applier, "EXTERNAL_FILES", {})
    return manifest_path


def test_guarded_applier_rechecks_support_files_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    teleop_root = tmp_path / "openarm_teleop"
    base_payloads = _write_exact_patch_inputs(teleop_root)
    support = teleop_root / "support.cpp"
    support.write_bytes(SOURCE.read_bytes())
    predecessor_manifest = _configure_sealed_predecessor(
        applier, teleop_root, tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        applier,
        "SUPPORT_FILES",
        {"support.cpp": "telemetry_publisher_source_sha256"},
    )
    monkeypatch.setattr(applier, "EXTERNAL_FILES", {})
    original_stage = applier._apply_in_staging

    def stage_then_race(
        payloads: dict[str, bytes], baseline: dict | None = None
    ) -> dict[str, bytes]:
        outputs = original_stage(payloads, baseline)
        support.write_bytes(b"changed after staging\n")
        return outputs

    monkeypatch.setattr(applier, "_apply_in_staging", stage_then_race)

    with pytest.raises(applier.PatchApplicationError, match="SHA-256 mismatch"):
        applier.apply_patch(
            teleop_root,
            check_only=False,
            predecessor_manifest=predecessor_manifest,
        )

    for relative, payload in base_payloads.items():
        assert (teleop_root / relative).read_bytes() == payload


def test_guarded_applier_rechecks_external_files_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    teleop_root = tmp_path / "openarm_teleop"
    base_payloads = _write_exact_patch_inputs(teleop_root)
    external = tmp_path / "libopenarm_can.a"
    external_payload = b"pinned external OpenArm SDK\n"
    external.write_bytes(external_payload)
    predecessor_manifest = _configure_sealed_predecessor(
        applier, teleop_root, tmp_path, monkeypatch
    )
    baseline = json.loads(applier.BASELINE_PATH.read_text(encoding="utf-8"))
    baseline["openarm_static_library_path"] = str(external)
    baseline["openarm_static_library_sha256"] = hashlib.sha256(
        external_payload
    ).hexdigest()
    applier.BASELINE_PATH.write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(applier, "SUPPORT_FILES", {})
    monkeypatch.setattr(
        applier,
        "EXTERNAL_FILES",
        {"openarm_static_library_path": "openarm_static_library_sha256"},
    )
    original_stage = applier._apply_in_staging

    def stage_then_race(
        payloads: dict[str, bytes], baseline: dict | None = None
    ) -> dict[str, bytes]:
        outputs = original_stage(payloads, baseline)
        external.write_bytes(b"changed after staging\n")
        return outputs

    monkeypatch.setattr(applier, "_apply_in_staging", stage_then_race)

    with pytest.raises(applier.PatchApplicationError, match="SHA-256 mismatch"):
        applier.apply_patch(
            teleop_root,
            check_only=False,
            predecessor_manifest=predecessor_manifest,
        )

    for relative, payload in base_payloads.items():
        assert (teleop_root / relative).read_bytes() == payload


def test_guarded_applier_rechecks_patch_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    originals = _write_exact_patch_inputs(root)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )
    patch_copy = tmp_path / "candidate.patch"
    patch_copy.write_bytes(PATCH.read_bytes())
    monkeypatch.setattr(applier, "PATCH_PATH", patch_copy)
    original_stage = applier._apply_in_staging

    def stage_then_race(
        payloads: dict[str, bytes], baseline: dict | None = None
    ) -> dict[str, bytes]:
        outputs = original_stage(payloads, baseline)
        patch_copy.write_bytes(b"changed after staging\n")
        return outputs

    monkeypatch.setattr(applier, "_apply_in_staging", stage_then_race)
    with pytest.raises(applier.PatchApplicationError, match="SHA-256 mismatch"):
        applier.apply_patch(
            root,
            check_only=False,
            predecessor_manifest=manifest_path,
        )

    assert all((root / path).read_bytes() == payload for path, payload in originals.items())


def test_guarded_applier_rechecks_changed_source_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    originals = _write_exact_patch_inputs(root)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )
    raced = root / "src/controller/control.hpp"
    original_stage = applier._apply_in_staging

    def stage_then_race(
        payloads: dict[str, bytes], baseline: dict | None = None
    ) -> dict[str, bytes]:
        outputs = original_stage(payloads, baseline)
        raced.write_bytes(b"changed after staging\n")
        return outputs

    monkeypatch.setattr(applier, "_apply_in_staging", stage_then_race)
    with pytest.raises(
        applier.PatchApplicationError,
        match="full-tree manifest content mismatch",
    ):
        applier.apply_patch(
            root,
            check_only=False,
            predecessor_manifest=manifest_path,
        )

    assert raced.read_bytes() == b"changed after staging\n"
    for relative, payload in originals.items():
        if root / relative != raced:
            assert (root / relative).read_bytes() == payload


def test_check_mode_rechecks_patch_after_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    _write_exact_patch_inputs(root)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )
    patch_copy = tmp_path / "candidate.patch"
    patch_copy.write_bytes(PATCH.read_bytes())
    monkeypatch.setattr(applier, "PATCH_PATH", patch_copy)
    original_stage = applier._apply_in_staging

    def stage_then_race(
        payloads: dict[str, bytes], baseline: dict | None = None
    ) -> dict[str, bytes]:
        outputs = original_stage(payloads, baseline)
        patch_copy.write_bytes(b"changed during check\n")
        return outputs

    monkeypatch.setattr(applier, "_apply_in_staging", stage_then_race)
    with pytest.raises(applier.PatchApplicationError, match="SHA-256 mismatch"):
        applier.apply_patch(
            root,
            check_only=True,
            predecessor_manifest=manifest_path,
        )


def test_versioned_patch_pins_exact_preimages_and_changes_only_gripper_sources() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    applier = _load_telemetry_applier_module()
    patch = PATCH.read_text(encoding="utf-8")

    assert baseline["schema"] == "wam_openarm_telemetry_patch_baseline_v5"
    assert "teleop_git_commit" not in baseline
    assert "teleop_git_tree" not in baseline
    assert baseline["predecessor_full_tree_manifest_status"] == "COMPLETE"
    assert baseline["predecessor_full_tree_manifest_sha256"] == (
        "4dbdfcc3c9f83dfee4c84196fd2c4b5730968b5fe47601d6d799ad77591d70b0"
    )
    assert applier.EXPECTED_BASELINE_SCHEMA == baseline["schema"]
    assert applier.PATCH_INPUTS == {
        "control/openarm_unilateral_control.cpp": "unilateral_source_sha256",
        "src/controller/control.cpp": "control_source_sha256",
        "src/controller/control.hpp": "control_header_sha256",
    }
    assert baseline["unilateral_source_sha256"] == (
        "55ed43a7da944e5f86aa7bc42fe3b7d970a70ec64ff820f4ca2d4a34f741e80b"
    )
    assert baseline["control_source_sha256"] == (
        "3956527f3ba9e65e2213d12362562ec242863d543f401f7e2206cfafdd0ee3e9"
    )
    assert baseline["control_header_sha256"] == (
        "46fae89179ffeed4288eb432cfe37bbe6161e8bfebc027d8ea7f93540b65b794"
    )
    assert baseline["cmake_sha256"] == (
        "926ee98f5cc5682ce0f0fbe3cf6780c0d7d32e780d3e9b4fbdebdeab57eaa81d"
    )
    assert baseline["telemetry_publisher_source_sha256"] == (
        "4e1baa041b6f6ab95d8900618dff3f7f2635a6cb6105a936206fee0f0324efb8"
    )
    assert baseline["telemetry_publisher_header_sha256"] == (
        "3e5882c458b4bc30c87e7f6428a2920bddf3abba2e1d874d9932393317bbfbdb"
    )
    assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == baseline["patch_sha256"]
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
        "telemetry_publisher_source_sha256",
        "telemetry_publisher_header_sha256",
        "robot_state_header_sha256",
        "joint_converter_header_sha256",
        "follower_config_sha256",
        "openarm_mit_header_sha256",
        "openarm_collection_header_sha256",
        "openarm_static_library_sha256",
    ):
        assert len(baseline[field]) == 64

    assert patch.count("--- a/") == 3
    assert "--- a/control/openarm_unilateral_control.cpp" in patch
    assert "--- a/src/controller/control.cpp" in patch
    assert "--- a/src/controller/control.hpp" in patch
    assert "--- a/CMakeLists.txt" not in patch
    assert "src/wam_openarm_telemetry/openarm_telemetry_publisher" not in patch
    assert "follower_command_q_rad" not in patch
    _exact_patch_base_inputs()


def test_versioned_patch_declares_hil_pending_stethoscope_gripper_contract() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    applier = _load_telemetry_applier_module()

    assert baseline["schema"] == "wam_openarm_telemetry_patch_baseline_v5"
    assert applier.EXPECTED_BASELINE_SCHEMA == baseline["schema"]
    assert baseline["deployment_scope"] == "stethoscope_left_only"
    assert baseline["gripper_decouple_env"] == "GRIPPER_DECOUPLE"
    assert baseline["gripper_hold_position_env"] == "GRIPPER_HOLD_POS"
    assert baseline["gripper_hold_position_rad"] == 0.0
    assert baseline["gripper_hold_position_status"] == "HIL_PENDING"


def test_generated_cpp_gripper_hold_is_exact_pair_opt_in_and_covers_both_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    monkeypatch.setattr(applier, "PATCH_PATH", PATCH)
    base = _exact_patch_base_inputs()
    staged = applier._apply_in_staging(base)

    assert set(staged) == {
        "control/openarm_unilateral_control.cpp",
        "src/controller/control.cpp",
        "src/controller/control.hpp",
    }
    main = staged["control/openarm_unilateral_control.cpp"].decode("utf-8")
    control = staged["src/controller/control.cpp"].decode("utf-8")
    header = staged["src/controller/control.hpp"].decode("utf-8")

    exact_pair = main.index(
        'arm_side == "left_arm" && leader_can_interface == "can1" &&'
    )
    exact_follower = main.index(
        'follower_can_interface == "can3"', exact_pair
    )
    opt_in = main.index('std::getenv("GRIPPER_DECOUPLE")', exact_follower)
    hold_env = main.index('parse_bounded_env("GRIPPER_HOLD_POS"', opt_in)
    configure = main.index("control_follower->ConfigureFollowerGripperHold(", hold_env)
    homing = main.index("&Control::AdjustPosition", configure)
    assert exact_pair < exact_follower < opt_in < hold_env < configure < homing
    assert hold_env < main.index(
        "openarm_init::OpenArmInitializer::initialize_openarm"
    )
    assert 'std::string(gripper_decouple_env) == "1"' in main[exact_pair:configure]
    assert "stethoscope_left_pair &&" in main[exact_pair:configure]
    assert "#include <cerrno>" in main
    assert "#include <cmath>" in main
    assert "#include <cstdlib>" in main
    assert "#include <string>" in main

    assert "follower_gripper_hold_enabled_ =" in control
    assert 'enabled && role_ == ROLE_FOLLOWER && arm_type_ == "left_arm"' in control
    assert "follower_gripper_hold_position_rad_ = hold_position_rad;" in control
    assert "follower_gripper_hold_enabled_ = false;" in header
    assert "follower_gripper_hold_position_rad_ = 0.0;" in header
    assert "GRIP_RAMP_RATE" in header
    assert "follower_gripper_max_lead_rad_" in header

    steady = control.split("else if (role_ == ROLE_FOLLOWER)", 1)[1].split(
        "wam::openarm_bridge::ControlTickRecord", 1
    )[0]
    assert "const bool grip_decouple = follower_gripper_hold_enabled_;" in steady
    assert "if (grip_decouple)" in steady
    assert "follower_gripper_hold_position_rad_" in steady
    assert "GRIP_RAMP_RATE * Ts_" in steady
    assert "follower_gripper_max_lead_rad_" in steady
    assert "std::clamp" in steady
    assert "joint_hand_states_ref" in steady
    assert steady.index("if (grip_decouple)") < steady.index(
        "openarmgripperjointconverter_->joint_to_motor(joint_hand_states_ref)"
    )

    adjust = control.split("bool Control::AdjustPosition(void)", 1)[1].split(
        "bool Control::DetectVibration", 1
    )[0]
    assert "if (follower_gripper_hold_enabled_)" in adjust
    assert "follower_gripper_hold_position_rad_" in adjust
    assert "GRIP_RAMP_RATE" in adjust
    assert "joint_hand_goal" in adjust
    assert "joint_hand_command" in adjust
    assert "home_pose[NMOTORS - 1]" in adjust  # Preserved default/right path.
    assert 'std::getenv("GRIPPER_DECOUPLE")' not in control
    assert 'std::getenv("GRIPPER_HOLD_POS")' not in control


def test_rebased_gripper_patch_preserves_safety_and_default_follow_text() -> None:
    applier = _load_telemetry_applier_module()
    base = _exact_patch_base_inputs()
    staged = applier._apply_in_staging(base)
    before_main = base["control/openarm_unilateral_control.cpp"].decode("utf-8")
    after_main = staged["control/openarm_unilateral_control.cpp"].decode("utf-8")
    before_control = base["src/controller/control.cpp"].decode("utf-8")
    after_control = staged["src/controller/control.cpp"].decode("utf-8")

    telemetry_begin = "        // The control thread only pushes fixed-size records"
    assert before_main.split(telemetry_begin, 1)[1] == after_main.split(
        telemetry_begin, 1,
    )[1]

    default_follow_begin = "        } else {\n            for (size_t i = 0; i < joint_hand_states_ref.size(); ++i) {"
    default_follow_end = "\n        // Gravity feedforward from the follower URDF"
    before_default = before_control.split(default_follow_begin, 1)[1].split(
        default_follow_end, 1,
    )[0]
    after_default = after_control.split(default_follow_begin, 1)[1].split(
        default_follow_end, 1,
    )[0]
    assert after_default == before_default

    for marker in (
        "dynamics_f_->GetGravity(joint_arm_positions.data(), gravity.data());",
        "ref_lim_ready_ = dynamics_f_->GetJointLimits(ref_lim_lo_, ref_lim_hi_);",
        "q_ref = std::clamp(q_ref, ref_lim_lo_[i] + LIMIT_MARGIN,",
        "q_ref = std::clamp(q_ref, q_now - REF_GOVERNOR, q_now + REF_GOVERNOR);",
        "arm_motors[i].get_feedback_liveness_snapshot();",
        "gripper_motors[0].get_feedback_liveness_snapshot();",
        "const auto arm_tx_status = openarm_->get_arm().mit_control_all(arm_cmds);",
        "const auto gripper_tx_status =",
        "follower_telemetry_->try_enqueue(telemetry_record);",
        "openarm_->recv_all(200);",
    ):
        assert before_control.count(marker) == after_control.count(marker) >= 1

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert applier.SUPPORT_FILES["CMakeLists.txt"] == "cmake_sha256"
    assert (
        applier.SUPPORT_FILES[
            "src/wam_openarm_telemetry/openarm_telemetry_publisher.cpp"
        ]
        == "telemetry_publisher_source_sha256"
    )
    assert (
        applier.SUPPORT_FILES[
            "src/wam_openarm_telemetry/openarm_telemetry_publisher.hpp"
        ]
        == "telemetry_publisher_header_sha256"
    )
    assert baseline["predecessor_manifest_state"] == "init_reset_applied"
    assert baseline["result_manifest_state"] == "gripper_patch_applied"


def test_gripper_changed_cpp_files_are_pinned_in_patch_result_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    monkeypatch.setattr(applier, "PATCH_PATH", PATCH)
    staged = applier._apply_in_staging(_exact_patch_base_inputs())

    assert "control/openarm_unilateral_control.cpp" in applier.PATCH_INPUTS
    assert "src/controller/control.cpp" in applier.PATCH_INPUTS
    assert "src/controller/control.hpp" in applier.PATCH_INPUTS
    assert "CMakeLists.txt" not in applier.PATCH_INPUTS
    assert "control/openarm_unilateral_control.cpp" in staged
    assert "src/controller/control.cpp" in staged
    assert "src/controller/control.hpp" in staged


def test_gripper_source_comments_describe_exact_opt_in_contract() -> None:
    applier = _load_telemetry_applier_module()
    staged = applier._apply_in_staging(_exact_patch_base_inputs())
    header = staged["src/controller/control.hpp"].decode("utf-8")

    assert "Default: compliant-follow" in header
    assert "GRIPPER_DECOUPLE=1 enables" in header
    assert "FOLLOW_GRIPPER" not in header
    assert "Default: ramp to FULLY OPEN" not in header


def test_pinned_predecessor_manifest_is_still_required_explicitly(
    tmp_path: Path,
) -> None:
    applier = _load_telemetry_applier_module()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert baseline["schema"] == "wam_openarm_telemetry_patch_baseline_v5"
    assert "teleop_git_commit" not in baseline
    assert "teleop_git_tree" not in baseline
    assert baseline["predecessor_full_tree_manifest_status"] == "COMPLETE"
    assert baseline["predecessor_full_tree_manifest_sha256"] == (
        "4dbdfcc3c9f83dfee4c84196fd2c4b5730968b5fe47601d6d799ad77591d70b0"
    )

    root = tmp_path / "candidate"
    root.mkdir()
    _write_exact_patch_inputs(root)
    with pytest.raises(
        applier.PatchApplicationError,
        match="predecessor full-tree manifest is required",
    ):
        applier.apply_patch(root, check_only=True, predecessor_manifest=None)


def test_sealed_full_tree_manifest_is_required_and_result_never_claims_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    _write_exact_patch_inputs(root)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )

    result = applier.apply_patch(
        root,
        check_only=True,
        predecessor_manifest=manifest_path,
    )

    assert result["schema"] == "wam_openarm_telemetry_patch_result_v5"
    assert "teleop_git_commit" not in result
    assert "teleop_git_tree" not in result
    assert result["predecessor_full_tree_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    successor = result["result_full_tree_manifest"]
    assert successor["sealed"] is True
    assert successor["purpose"] == "private_candidate_staging"
    assert successor["coverage"] == "full_tracked_tree"
    assert successor["state"] == "gripper_patch_applied"
    assert successor["build_authorized"] is True
    assert successor["transition"] == {
        "kind": "openarm_gripper_patch_v1",
        "patch_sha256": result["patch_sha256"],
        "predecessor_manifest_sha256": result[
            "predecessor_full_tree_manifest_sha256"
        ],
    }
    assert result["result_full_tree_manifest_sha256"] == hashlib.sha256(
        _canonical_json_bytes(successor)
    ).hexdigest()


def test_successful_private_candidate_apply_matches_sealed_successor_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    _write_exact_patch_inputs(root)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )

    result = applier.apply_patch(
        root,
        check_only=False,
        predecessor_manifest=manifest_path,
        recovery_artifact=tmp_path / "RECOVERY.json",
    )

    assert result["mode"] == "applied"
    assert {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in sorted(result["outputs"])
    } == result["outputs"]
    applier._verify_tree_against_manifest(
        root,
        result["result_full_tree_manifest"],
    )
    assert not (tmp_path / "RECOVERY.json").exists()


def test_recovery_artifact_cannot_overwrite_predecessor_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    originals = _write_exact_patch_inputs(root)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(
        applier.PatchApplicationError,
        match="recovery artifact collides with protected input",
    ):
        applier.apply_patch(
            root,
            check_only=False,
            predecessor_manifest=manifest_path,
            recovery_artifact=manifest_path,
        )

    assert manifest_path.read_bytes() == manifest_before
    assert all((root / path).read_bytes() == payload for path, payload in originals.items())


def test_full_tree_manifest_refuses_unlisted_candidate_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    _write_exact_patch_inputs(root)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )
    (root / "unlisted.cpp").write_text("drift\n", encoding="utf-8")

    with pytest.raises(
        applier.PatchApplicationError,
        match="full-tree manifest path set mismatch",
    ):
        applier.apply_patch(
            root,
            check_only=True,
            predecessor_manifest=manifest_path,
        )


def test_private_candidate_manifest_refuses_stale_build_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    _write_exact_patch_inputs(root)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )
    stale = root / "build" / "unilateral_control"
    stale.parent.mkdir()
    stale.write_bytes(b"stale binary\n")

    with pytest.raises(
        applier.PatchApplicationError,
        match="full-tree manifest path set mismatch",
    ):
        applier.apply_patch(
            root,
            check_only=True,
            predecessor_manifest=manifest_path,
        )


@pytest.mark.parametrize("failed_replace", [1, 2, 3])
def test_replace_failure_rolls_back_every_changed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_replace: int,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    originals = _write_exact_patch_inputs(root)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )
    recovery = tmp_path / "RECOVERY.json"
    real_replace = applier._atomic_replace
    calls = 0
    failed = False

    def fail_selected(path: Path, payload: bytes, mode: int) -> None:
        nonlocal calls, failed
        calls += 1
        if not failed and calls == failed_replace:
            failed = True
            raise OSError(f"injected replace {failed_replace} failure")
        real_replace(path, payload, mode)

    monkeypatch.setattr(applier, "_atomic_replace", fail_selected)
    with pytest.raises(OSError, match=f"injected replace {failed_replace} failure"):
        applier.apply_patch(
            root,
            check_only=False,
            predecessor_manifest=manifest_path,
            recovery_artifact=recovery,
        )

    assert all((root / path).read_bytes() == payload for path, payload in originals.items())
    assert not recovery.exists()
    assert calls == (2 * failed_replace) - 1


def test_post_write_output_tamper_rolls_back_and_refuses_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    originals = _write_exact_patch_inputs(root)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )
    real_replace = applier._atomic_replace
    forward_writes = 0

    def tamper_after_last_write(path: Path, payload: bytes, mode: int) -> None:
        nonlocal forward_writes
        real_replace(path, payload, mode)
        if forward_writes < 3:
            forward_writes += 1
            if forward_writes == 3:
                (root / "src/controller/control.hpp").write_bytes(b"tampered\n")

    monkeypatch.setattr(applier, "_atomic_replace", tamper_after_last_write)
    with pytest.raises(
        applier.PatchApplicationError,
        match="post-write verification failed",
    ):
        applier.apply_patch(
            root,
            check_only=False,
            predecessor_manifest=manifest_path,
            recovery_artifact=tmp_path / "RECOVERY.json",
        )

    assert all((root / path).read_bytes() == payload for path, payload in originals.items())


@pytest.mark.parametrize("raced_input", ["patch", "baseline", "support", "external"])
def test_post_write_rechecks_every_static_input_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raced_input: str,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    originals = _write_exact_patch_inputs(root)
    support = root / "support.cpp"
    support.write_bytes(SOURCE.read_bytes())
    external = tmp_path / "libopenarm_can.a"
    external_payload = b"pinned external OpenArm SDK\n"
    external.write_bytes(external_payload)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )
    patch_copy = tmp_path / "candidate.patch"
    patch_copy.write_bytes(PATCH.read_bytes())
    monkeypatch.setattr(applier, "PATCH_PATH", patch_copy)
    monkeypatch.setattr(
        applier,
        "SUPPORT_FILES",
        {"support.cpp": "telemetry_publisher_source_sha256"},
    )
    monkeypatch.setattr(
        applier,
        "EXTERNAL_FILES",
        {"openarm_static_library_path": "openarm_static_library_sha256"},
    )
    baseline = json.loads(applier.BASELINE_PATH.read_text(encoding="utf-8"))
    baseline["openarm_static_library_path"] = str(external)
    baseline["openarm_static_library_sha256"] = hashlib.sha256(
        external_payload
    ).hexdigest()
    applier.BASELINE_PATH.write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
    )
    race_targets = {
        "patch": patch_copy,
        "baseline": applier.BASELINE_PATH,
        "support": support,
        "external": external,
    }
    real_replace = applier._atomic_replace
    forward_writes = 0

    def race_after_last_write(path: Path, payload: bytes, mode: int) -> None:
        nonlocal forward_writes
        real_replace(path, payload, mode)
        if forward_writes < 3:
            forward_writes += 1
            if forward_writes == 3:
                race_targets[raced_input].write_bytes(b"raced after final CAS\n")

    monkeypatch.setattr(applier, "_atomic_replace", race_after_last_write)
    with pytest.raises(
        applier.PatchApplicationError,
        match="post-write verification failed",
    ):
        applier.apply_patch(
            root,
            check_only=False,
            predecessor_manifest=manifest_path,
            recovery_artifact=tmp_path / "RECOVERY.json",
        )

    assert all((root / path).read_bytes() == payload for path, payload in originals.items())


def test_rollback_failure_writes_canonical_recovery_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applier = _load_telemetry_applier_module()
    root = tmp_path / "candidate"
    root.mkdir()
    _write_exact_patch_inputs(root)
    manifest_path = _configure_sealed_predecessor(
        applier, root, tmp_path, monkeypatch
    )
    recovery = tmp_path / "RECOVERY.json"
    real_replace = applier._atomic_replace
    calls = 0

    def fail_forward_and_rollback(path: Path, payload: bytes, mode: int) -> None:
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise OSError(f"injected atomic failure {calls}")
        real_replace(path, payload, mode)

    monkeypatch.setattr(applier, "_atomic_replace", fail_forward_and_rollback)
    with pytest.raises(
        applier.PatchApplicationError,
        match="rollback incomplete",
    ):
        applier.apply_patch(
            root,
            check_only=False,
            predecessor_manifest=manifest_path,
            recovery_artifact=recovery,
        )

    raw = recovery.read_bytes()
    decoded = json.loads(raw)
    assert raw == _canonical_json_bytes(decoded)
    assert decoded["schema"] == "wam_openarm_telemetry_recovery_v1"
    assert decoded["status"] == "ROLLBACK_INCOMPLETE"
    assert decoded["targets"]
