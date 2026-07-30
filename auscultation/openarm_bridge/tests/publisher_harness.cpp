#include "openarm_telemetry_publisher.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>

namespace {

wam::openarm_bridge::PublisherConfig make_config(const std::string& socket_path) {
    wam::openarm_bridge::PublisherConfig config;
    config.socket_path = socket_path;
    config.controller_start_token = "proc-start-991";
    config.urdf_sha256 = std::string(64, 'a');
    config.joint_mapper_sha256 = std::string(64, 'b');
    config.pair_contract_sha256 = std::string(64, 'e');
    config.controller_binary_sha256 = std::string(64, 'f');
    config.controller_source_manifest_sha256 = std::string(64, '1');
    config.follower_config_sha256 = std::string(64, '2');
    config.openarm_sdk_sha256 = std::string(64, '3');
    return config;
}

wam::openarm_bridge::ControlTickRecord make_record(
    std::uint64_t sequence,
    std::int64_t observation_ns) {
    wam::openarm_bridge::ControlTickRecord record;
    record.control_tick_seq = sequence;
    record.observation_monotonic_ns = observation_ns;
    record.reference_generation = 7;
    record.reference_source_monotonic_ns = observation_ns - 100'000;
    record.reference_snapshot_valid = true;
    record.control_call_begin_monotonic_ns = observation_ns + 50'000;
    record.arm_call_return_monotonic_ns = observation_ns + 100'000;
    record.gripper_call_begin_monotonic_ns = observation_ns + 110'000;
    record.gripper_call_return_monotonic_ns = observation_ns + 160'000;
    record.control_call_end_monotonic_ns = observation_ns + 170'000;
    record.arm_tx_expected_frame_count = 7;
    record.arm_tx_full_frame_write_count = 7;
    record.arm_tx_failed_frame_write_count = 0;
    record.gripper_tx_expected_frame_count = 1;
    record.gripper_tx_full_frame_write_count = 1;
    record.gripper_tx_failed_frame_write_count = 0;
    record.socketcan_tx_enqueue_success = true;
    for (std::size_t index = 0; index < record.follower_q_rad.size(); ++index) {
        const double base = static_cast<double>(index) * 0.01;
        record.follower_q_rad[index] = base;
        record.follower_qd_rad_s[index] = 0.5;
        record.follower_arm_mit[index] = {
            index < 4 ? 240.0 : 40.0,
            index < 4 ? 3.0 : 0.3,
            base + 0.10,
            0.4,
            0.0,
        };
        record.follower_arm_state_rx_motor_ids[index] =
            static_cast<std::uint32_t>(index + 1);
        record.follower_arm_state_rx_counters[index] = sequence + index + 1;
        record.follower_arm_state_rx_generations[index] = sequence + index + 1;
        record.follower_arm_state_rx_monotonic_ns[index] =
            observation_ns - 500'000 + static_cast<std::int64_t>(index);
        record.follower_arm_state_rx_last_gap_ns[index] = 1'500'000;
        record.follower_arm_state_rx_gap_violation_counts[index] = 0;
    }
    record.follower_gripper_q_rad = -0.070;
    record.follower_gripper_qd_rad_s = 0.010;
    record.follower_gripper_motor_q_rad = -0.075;
    record.follower_gripper_motor_qd_rad_s = 0.015;
    record.follower_gripper_mit = {25.0, 0.3, -0.080, 0.0, 0.0};
    record.feedback_snapshot_monotonic_ns = observation_ns;
    record.follower_gripper_state_rx_motor_id = 8;
    record.follower_gripper_state_rx_counter = sequence + 8;
    record.follower_gripper_state_rx_generation = sequence + 8;
    record.follower_gripper_state_rx_monotonic_ns = observation_ns - 400'000;
    record.follower_gripper_state_rx_last_gap_ns = 1'400'000;
    record.follower_gripper_state_rx_gap_violation_count = 0;
    return record;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        return 64;
    }
    const std::string mode(argv[1]);
    if (mode == "serialize_single" || mode == "serialize_batch") {
        const std::size_t total = mode == "serialize_single" ? 1 : 25;
        std::uint64_t envelope_sequence = 0;
        for (std::size_t offset = 0; offset < total;) {
            std::array<
                wam::openarm_bridge::ControlTickRecord,
                wam::openarm_bridge::kMaxRecordsPerEnvelope>
                records{};
            const std::size_t count = std::min(
                total - offset,
                wam::openarm_bridge::kMaxRecordsPerEnvelope);
            for (std::size_t index = 0; index < count; ++index) {
                const auto sequence =
                    static_cast<std::uint64_t>(100 + offset + index);
                records[index] = make_record(
                    sequence,
                    1'000'000'000 +
                        static_cast<std::int64_t>(offset + index) * 1'000'000);
            }
            wam::openarm_bridge::SerializedControlEnvelope serialized;
            if (!wam::openarm_bridge::serialize_control_envelope_json(
                    serialized,
                    make_config(argv[2]),
                    envelope_sequence++,
                    4242,
                    2'000'000'000,
                    0,
                    0,
                    0,
                    records,
                    count)) {
                return 68;
            }
            std::cout.write(
                serialized.bytes.data(),
                static_cast<std::streamsize>(serialized.size));
            std::cout.put('\n');
            offset += count;
        }
        return 0;
    }
    wam::openarm_bridge::OpenArmTelemetryPublisher publisher(
        make_config(argv[2]));
    if (mode != "reference_epoch" && !publisher.enabled()) {
        return 66;
    }

    if (mode == "single") {
        publisher.try_enqueue(make_record(100, 1'000'000'000));
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        if (publisher.rejected_record_count() != 0 ||
            publisher.send_attempt_count() != 1 ||
            publisher.last_send_return() <= 0) {
            std::cerr << "rejected=" << publisher.rejected_record_count()
                      << " attempts=" << publisher.send_attempt_count()
                      << " return=" << publisher.last_send_return()
                      << " errno=" << publisher.last_send_errno() << "\n";
            return 67;
        }
        return 0;
    }
    if (mode == "batch") {
        for (std::uint64_t index = 0; index < 25; ++index) {
            publisher.try_enqueue(
                make_record(100 + index, 1'000'000'000 + index * 1'000'000));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        return 0;
    }
    if (mode == "flood") {
        for (std::uint64_t index = 0; index < 100'000; ++index) {
            publisher.try_enqueue(
                make_record(index, 1'000'000'000 + index * 1'000'000));
        }
        return 0;
    }
    if (mode == "reference_epoch") {
        wam::openarm_bridge::FollowerReferenceEpoch epoch;
        const auto empty = epoch.begin_read();
        if (epoch.read_still_valid(empty)) {
            return 70;
        }
        epoch.begin_update();
        const auto during = epoch.begin_read();
        if (epoch.read_still_valid(during)) {
            return 71;
        }
        epoch.complete_update(999'000'000);
        const auto stable = epoch.begin_read();
        if (!epoch.read_still_valid(stable) || stable.generation != 1 ||
            stable.source_monotonic_ns != 999'000'000) {
            return 72;
        }
        epoch.begin_update();
        if (epoch.read_still_valid(stable)) {
            return 73;
        }
        epoch.complete_update(1'000'000'000);
        return 0;
    }
    return 65;
}
