#include "openarm_telemetry_publisher.hpp"

#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string_view>
#include <type_traits>
#include <unistd.h>

#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#endif
#include <sys/socket.h>

namespace wam::openarm_bridge {
namespace {

constexpr std::string_view kEnvelopeSchema =
    "wam_openarm_control_envelope_v4";
constexpr std::string_view kRecordSchema =
    "wam_openarm_exact_control_record_v4";
constexpr std::string_view kSendStatusReason =
    "openarm_can_1_2_2_full_frame_write_results";
constexpr std::string_view kSendStatusSemantics =
    "linux_socketcan_full_frame_write_tx_enqueue_not_motor_ack";
constexpr std::string_view kMotorAckReason =
    "socketcan_tx_enqueue_does_not_acknowledge_motor_receipt_or_execution";
constexpr std::string_view kFeedbackLivenessReason =
    "openarm_can_1_2_2_matching_motor_id_valid_state_decode";
constexpr std::string_view kFeedbackLivenessSemantics =
    "counter_generation_increment_only_after_matching_motor_id_valid_state_decode";
constexpr std::size_t kMaxJsonBytes = 131'072;

bool is_lower_sha256(std::string_view value) noexcept {
    if (value.size() != 64) {
        return false;
    }
    for (const char character : value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool is_start_token(std::string_view value) noexcept {
    if (value.empty() || value.size() > 128) {
        return false;
    }
    for (const char character : value) {
        const bool accepted =
            (character >= 'A' && character <= 'Z') ||
            (character >= 'a' && character <= 'z') ||
            (character >= '0' && character <= '9') || character == '_' ||
            character == '.' || character == ':' || character == '/' ||
            character == '-';
        if (!accepted) {
            return false;
        }
    }
    return true;
}

bool finite_mit(const MitParamPod& value) noexcept {
    return std::isfinite(value.kp) && std::isfinite(value.kd) &&
           std::isfinite(value.q) && std::isfinite(value.dq) &&
           std::isfinite(value.tau);
}

bool valid_record(const ControlTickRecord& record) noexcept {
    if (record.observation_monotonic_ns <= 0 ||
        record.reference_generation == 0 ||
        record.reference_source_monotonic_ns <= 0 ||
        record.control_call_begin_monotonic_ns <= 0 ||
        record.arm_call_return_monotonic_ns <= 0 ||
        record.gripper_call_begin_monotonic_ns <= 0 ||
        record.gripper_call_return_monotonic_ns <= 0 ||
        record.control_call_end_monotonic_ns <= 0 ||
        record.arm_tx_expected_frame_count != kOpenArmJointCount ||
        record.gripper_tx_expected_frame_count != 1 ||
        record.arm_tx_full_frame_write_count +
                record.arm_tx_failed_frame_write_count !=
            record.arm_tx_expected_frame_count ||
        record.gripper_tx_full_frame_write_count +
                record.gripper_tx_failed_frame_write_count !=
            record.gripper_tx_expected_frame_count ||
        record.socketcan_tx_enqueue_success !=
            (record.arm_tx_full_frame_write_count == kOpenArmJointCount &&
             record.arm_tx_failed_frame_write_count == 0 &&
             record.gripper_tx_full_frame_write_count == 1 &&
             record.gripper_tx_failed_frame_write_count == 0) ||
        record.feedback_snapshot_monotonic_ns !=
            record.observation_monotonic_ns ||
        record.follower_gripper_state_rx_motor_id == 0 ||
        record.follower_gripper_state_rx_counter == 0 ||
        record.follower_gripper_state_rx_generation !=
            record.follower_gripper_state_rx_counter ||
        record.follower_gripper_state_rx_monotonic_ns <= 0 ||
        record.follower_gripper_state_rx_monotonic_ns >
            record.feedback_snapshot_monotonic_ns ||
        record.follower_gripper_state_rx_last_gap_ns < 0 ||
        !(record.reference_source_monotonic_ns <=
              record.control_call_begin_monotonic_ns &&
          record.observation_monotonic_ns <=
              record.control_call_begin_monotonic_ns &&
          record.control_call_begin_monotonic_ns <=
              record.arm_call_return_monotonic_ns &&
          record.arm_call_return_monotonic_ns <=
              record.gripper_call_begin_monotonic_ns &&
          record.gripper_call_begin_monotonic_ns <=
              record.gripper_call_return_monotonic_ns &&
          record.gripper_call_return_monotonic_ns <=
              record.control_call_end_monotonic_ns)) {
        return false;
    }
    for (std::size_t index = 0; index < kOpenArmJointCount; ++index) {
        if (record.follower_arm_state_rx_motor_ids[index] == 0 ||
            record.follower_arm_state_rx_counters[index] == 0 ||
            record.follower_arm_state_rx_generations[index] !=
                record.follower_arm_state_rx_counters[index] ||
            record.follower_arm_state_rx_monotonic_ns[index] <= 0 ||
            record.follower_arm_state_rx_monotonic_ns[index] >
                record.feedback_snapshot_monotonic_ns ||
            record.follower_arm_state_rx_last_gap_ns[index] < 0 ||
            !std::isfinite(record.follower_q_rad[index]) ||
            !std::isfinite(record.follower_qd_rad_s[index]) ||
            !finite_mit(record.follower_arm_mit[index])) {
            return false;
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (record.follower_arm_state_rx_motor_ids[index] ==
                record.follower_arm_state_rx_motor_ids[previous]) {
                return false;
            }
        }
        if (record.follower_arm_state_rx_motor_ids[index] ==
            record.follower_gripper_state_rx_motor_id) {
            return false;
        }
    }
    return std::isfinite(record.follower_gripper_q_rad) &&
           std::isfinite(record.follower_gripper_qd_rad_s) &&
           std::isfinite(record.follower_gripper_motor_q_rad) &&
           std::isfinite(record.follower_gripper_motor_qd_rad_s) &&
           finite_mit(record.follower_gripper_mit);
}

class JsonBuffer {
public:
    bool append(std::string_view value) noexcept {
        if (!ok_ || value.size() > bytes_.size() - size_) {
            ok_ = false;
            return false;
        }
        std::memcpy(bytes_.data() + size_, value.data(), value.size());
        size_ += value.size();
        return true;
    }

    template <typename Integer>
    bool append_integer(Integer value) noexcept {
        static_assert(std::is_integral_v<Integer>);
        if (!ok_) {
            return false;
        }
        auto result = std::to_chars(
            bytes_.data() + size_, bytes_.data() + bytes_.size(), value);
        if (result.ec != std::errc()) {
            ok_ = false;
            return false;
        }
        size_ = static_cast<std::size_t>(result.ptr - bytes_.data());
        return true;
    }

    bool append_double(double value) noexcept {
        if (!ok_ || !std::isfinite(value)) {
            ok_ = false;
            return false;
        }
        auto result = std::to_chars(
            bytes_.data() + size_,
            bytes_.data() + bytes_.size(),
            value,
            std::chars_format::general,
            std::numeric_limits<double>::max_digits10);
        if (result.ec != std::errc()) {
            ok_ = false;
            return false;
        }
        size_ = static_cast<std::size_t>(result.ptr - bytes_.data());
        return true;
    }

    bool append_array(
        const std::array<double, kOpenArmJointCount>& values) noexcept {
        append("[");
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index != 0) {
                append(",");
            }
            append_double(values[index]);
        }
        return append("]");
    }

    template <typename Integer, std::size_t Count>
    bool append_integer_array(const std::array<Integer, Count>& values) noexcept {
        static_assert(std::is_integral_v<Integer>);
        append("[");
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index != 0) {
                append(",");
            }
            append_integer(values[index]);
        }
        return append("]");
    }

    template <typename Member>
    bool append_mit_array(
        const std::array<MitParamPod, kOpenArmJointCount>& values,
        Member member) noexcept {
        append("[");
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index != 0) {
                append(",");
            }
            append_double(values[index].*member);
        }
        return append("]");
    }

    [[nodiscard]] bool ok() const noexcept { return ok_; }
    [[nodiscard]] const char* data() const noexcept { return bytes_.data(); }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    bool export_to(SerializedControlEnvelope& output) const noexcept {
        if (!ok_) {
            return false;
        }
        std::memcpy(output.bytes.data(), bytes_.data(), size_);
        output.size = size_;
        return true;
    }

private:
    std::array<char, kMaxJsonBytes> bytes_{};
    std::size_t size_ = 0;
    bool ok_ = true;
};

bool append_record(JsonBuffer& output, const ControlTickRecord& record) noexcept {
    output.append("{\"schema\":\"");
    output.append(kRecordSchema);
    output.append("\",\"control_tick_seq\":");
    output.append_integer(record.control_tick_seq);
    output.append(",\"observation_monotonic_ns\":");
    output.append_integer(record.observation_monotonic_ns);
    output.append(",\"reference_generation\":");
    output.append_integer(record.reference_generation);
    output.append(",\"reference_source_monotonic_ns\":");
    output.append_integer(record.reference_source_monotonic_ns);
    output.append(",\"reference_snapshot_valid\":");
    output.append(record.reference_snapshot_valid ? "true" : "false");
    output.append(",\"control_call_begin_monotonic_ns\":");
    output.append_integer(record.control_call_begin_monotonic_ns);
    output.append(",\"arm_call_return_monotonic_ns\":");
    output.append_integer(record.arm_call_return_monotonic_ns);
    output.append(",\"gripper_call_begin_monotonic_ns\":");
    output.append_integer(record.gripper_call_begin_monotonic_ns);
    output.append(",\"gripper_call_return_monotonic_ns\":");
    output.append_integer(record.gripper_call_return_monotonic_ns);
    output.append(",\"control_call_end_monotonic_ns\":");
    output.append_integer(record.control_call_end_monotonic_ns);
    output.append(",\"arm_tx_expected_frame_count\":");
    output.append_integer(record.arm_tx_expected_frame_count);
    output.append(",\"arm_tx_full_frame_write_count\":");
    output.append_integer(record.arm_tx_full_frame_write_count);
    output.append(",\"arm_tx_failed_frame_write_count\":");
    output.append_integer(record.arm_tx_failed_frame_write_count);
    output.append(",\"gripper_tx_expected_frame_count\":");
    output.append_integer(record.gripper_tx_expected_frame_count);
    output.append(",\"gripper_tx_full_frame_write_count\":");
    output.append_integer(record.gripper_tx_full_frame_write_count);
    output.append(",\"gripper_tx_failed_frame_write_count\":");
    output.append_integer(record.gripper_tx_failed_frame_write_count);
    output.append(",\"socketcan_tx_enqueue_success\":");
    output.append(record.socketcan_tx_enqueue_success ? "true" : "false");
    output.append(",\"feedback_snapshot_monotonic_ns\":");
    output.append_integer(record.feedback_snapshot_monotonic_ns);
    output.append(",\"follower_arm_state_rx_motor_ids\":");
    output.append_integer_array(record.follower_arm_state_rx_motor_ids);
    output.append(",\"follower_arm_state_rx_counters\":");
    output.append_integer_array(record.follower_arm_state_rx_counters);
    output.append(",\"follower_arm_state_rx_generations\":");
    output.append_integer_array(record.follower_arm_state_rx_generations);
    output.append(",\"follower_arm_state_rx_monotonic_ns\":");
    output.append_integer_array(record.follower_arm_state_rx_monotonic_ns);
    output.append(",\"follower_arm_state_rx_last_gap_ns\":");
    output.append_integer_array(record.follower_arm_state_rx_last_gap_ns);
    output.append(",\"follower_arm_state_rx_gap_violation_counts\":");
    output.append_integer_array(
        record.follower_arm_state_rx_gap_violation_counts);
    output.append(",\"follower_gripper_state_rx_motor_id\":");
    output.append_integer(record.follower_gripper_state_rx_motor_id);
    output.append(",\"follower_gripper_state_rx_counter\":");
    output.append_integer(record.follower_gripper_state_rx_counter);
    output.append(",\"follower_gripper_state_rx_generation\":");
    output.append_integer(record.follower_gripper_state_rx_generation);
    output.append(",\"follower_gripper_state_rx_monotonic_ns\":");
    output.append_integer(record.follower_gripper_state_rx_monotonic_ns);
    output.append(",\"follower_gripper_state_rx_last_gap_ns\":");
    output.append_integer(record.follower_gripper_state_rx_last_gap_ns);
    output.append(",\"follower_gripper_state_rx_gap_violation_count\":");
    output.append_integer(
        record.follower_gripper_state_rx_gap_violation_count);
    output.append(",\"follower_q_rad\":");
    output.append_array(record.follower_q_rad);
    output.append(",\"follower_qd_rad_s\":");
    output.append_array(record.follower_qd_rad_s);
    output.append(",\"follower_gripper_q_rad\":");
    output.append_double(record.follower_gripper_q_rad);
    output.append(",\"follower_gripper_qd_rad_s\":");
    output.append_double(record.follower_gripper_qd_rad_s);
    output.append(",\"follower_gripper_motor_q_rad\":");
    output.append_double(record.follower_gripper_motor_q_rad);
    output.append(",\"follower_gripper_motor_qd_rad_s\":");
    output.append_double(record.follower_gripper_motor_qd_rad_s);
    output.append(",\"follower_arm_mit_kp\":");
    output.append_mit_array(record.follower_arm_mit, &MitParamPod::kp);
    output.append(",\"follower_arm_mit_kd\":");
    output.append_mit_array(record.follower_arm_mit, &MitParamPod::kd);
    output.append(",\"follower_arm_mit_q\":");
    output.append_mit_array(record.follower_arm_mit, &MitParamPod::q);
    output.append(",\"follower_arm_mit_dq\":");
    output.append_mit_array(record.follower_arm_mit, &MitParamPod::dq);
    output.append(",\"follower_arm_mit_tau\":");
    output.append_mit_array(record.follower_arm_mit, &MitParamPod::tau);
    output.append(",\"follower_gripper_mit_kp\":");
    output.append_double(record.follower_gripper_mit.kp);
    output.append(",\"follower_gripper_mit_kd\":");
    output.append_double(record.follower_gripper_mit.kd);
    output.append(",\"follower_gripper_mit_q\":");
    output.append_double(record.follower_gripper_mit.q);
    output.append(",\"follower_gripper_mit_dq\":");
    output.append_double(record.follower_gripper_mit.dq);
    output.append(",\"follower_gripper_mit_tau\":");
    output.append_double(record.follower_gripper_mit.tau);
    return output.append("}");
}

void lower_publisher_thread_priority() noexcept {
#if defined(__linux__) && defined(SYS_gettid)
    const auto tid = static_cast<id_t>(::syscall(SYS_gettid));
    (void)::setpriority(PRIO_PROCESS, tid, 10);
#endif
}

}  // namespace

std::int64_t monotonic_now_ns() noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

bool serialize_control_envelope_json(
    SerializedControlEnvelope& serialized,
    const PublisherConfig& config,
    std::uint64_t envelope_sequence,
    int controller_pid,
    std::int64_t publisher_monotonic_ns,
    std::uint64_t ring_overflow_count,
    std::uint64_t rejected_record_count,
    std::uint64_t socket_send_failure_count,
    const std::array<ControlTickRecord, kMaxRecordsPerEnvelope>& records,
    std::size_t count) noexcept {
    serialized.size = 0;
    if (count == 0 || count > records.size() || controller_pid <= 0 ||
        !is_start_token(config.controller_start_token) ||
        !is_lower_sha256(config.urdf_sha256) ||
        !is_lower_sha256(config.joint_mapper_sha256) ||
        !is_lower_sha256(config.pair_contract_sha256) ||
        !is_lower_sha256(config.controller_binary_sha256) ||
        !is_lower_sha256(config.controller_source_manifest_sha256) ||
        !is_lower_sha256(config.follower_config_sha256) ||
        !is_lower_sha256(config.openarm_sdk_sha256)) {
        return false;
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (!valid_record(records[index]) ||
            publisher_monotonic_ns <
                records[index].control_call_end_monotonic_ns) {
            return false;
        }
        if (index != 0 &&
            (records[index].control_tick_seq !=
                 records[index - 1].control_tick_seq + 1 ||
             records[index].observation_monotonic_ns <=
                 records[index - 1].observation_monotonic_ns)) {
            return false;
        }
        if (index != 0) {
            const auto& previous = records[index - 1];
            const auto& current = records[index];
            for (std::size_t motor = 0; motor < kOpenArmJointCount; ++motor) {
                if (current.follower_arm_state_rx_motor_ids[motor] !=
                        previous.follower_arm_state_rx_motor_ids[motor] ||
                    current.follower_arm_state_rx_counters[motor] <
                        previous.follower_arm_state_rx_counters[motor] ||
                    current.follower_arm_state_rx_monotonic_ns[motor] <
                        previous.follower_arm_state_rx_monotonic_ns[motor] ||
                    ((current.follower_arm_state_rx_counters[motor] ==
                          previous.follower_arm_state_rx_counters[motor]) !=
                     (current.follower_arm_state_rx_monotonic_ns[motor] ==
                          previous.follower_arm_state_rx_monotonic_ns[motor])) ||
                    current.follower_arm_state_rx_gap_violation_counts[motor] <
                        previous.follower_arm_state_rx_gap_violation_counts[motor]) {
                    return false;
                }
            }
            if (current.follower_gripper_state_rx_motor_id !=
                    previous.follower_gripper_state_rx_motor_id ||
                current.follower_gripper_state_rx_counter <
                    previous.follower_gripper_state_rx_counter ||
                current.follower_gripper_state_rx_monotonic_ns <
                    previous.follower_gripper_state_rx_monotonic_ns ||
                ((current.follower_gripper_state_rx_counter ==
                      previous.follower_gripper_state_rx_counter) !=
                 (current.follower_gripper_state_rx_monotonic_ns ==
                      previous.follower_gripper_state_rx_monotonic_ns)) ||
                current.follower_gripper_state_rx_gap_violation_count <
                    previous.follower_gripper_state_rx_gap_violation_count) {
                return false;
            }
        }
    }

    JsonBuffer output;
    output.append("{\"schema\":\"");
    output.append(kEnvelopeSchema);
    output.append("\",\"envelope_seq\":");
    output.append_integer(envelope_sequence);
    output.append(",\"clock_id\":\"CLOCK_MONOTONIC\","
                  "\"publisher_monotonic_ns\":");
    output.append_integer(publisher_monotonic_ns);
    output.append(",\"controller_pid\":");
    output.append_integer(controller_pid);
    output.append(",\"controller_start_token\":\"");
    output.append(config.controller_start_token);
    output.append(
        "\",\"side\":\"left\",\"controller_mode\":\"unilateral\","
        "\"leader_can_interface\":\"can1\","
        "\"follower_can_interface\":\"can3\","
        "\"joint_names\":["
        "\"openarm_left_joint1\",\"openarm_left_joint2\","
        "\"openarm_left_joint3\",\"openarm_left_joint4\","
        "\"openarm_left_joint5\",\"openarm_left_joint6\","
        "\"openarm_left_joint7\"],\"urdf_sha256\":\"");
    output.append(config.urdf_sha256);
    output.append("\",\"joint_mapper_sha256\":\"");
    output.append(config.joint_mapper_sha256);
    output.append("\",\"pair_contract_sha256\":\"");
    output.append(config.pair_contract_sha256);
    output.append("\",\"controller_binary_sha256\":\"");
    output.append(config.controller_binary_sha256);
    output.append("\",\"controller_source_manifest_sha256\":\"");
    output.append(config.controller_source_manifest_sha256);
    output.append("\",\"follower_config_sha256\":\"");
    output.append(config.follower_config_sha256);
    output.append("\",\"openarm_sdk_sha256\":\"");
    output.append(config.openarm_sdk_sha256);
    output.append(
        "\",\"publisher_nominal_hz\":100,\"control_nominal_hz\":1000,"
        "\"send_status_observable\":true,\"send_status_reason\":\"");
    output.append(kSendStatusReason);
    output.append("\",\"send_status_semantics\":\"");
    output.append(kSendStatusSemantics);
    output.append("\",\"motor_ack_observable\":false,\"motor_ack_reason\":\"");
    output.append(kMotorAckReason);
    output.append(
        "\",\"feedback_liveness_observable\":true,"
        "\"feedback_liveness_reason\":\"");
    output.append(kFeedbackLivenessReason);
    output.append("\",\"feedback_liveness_semantics\":\"");
    output.append(kFeedbackLivenessSemantics);
    output.append("\",\"feedback_max_age_ns\":");
    output.append_integer(kFeedbackMaxAgeNs);
    output.append(",\"feedback_max_gap_ns\":");
    output.append_integer(kFeedbackMaxGapNs);
    output.append(",\"ring_overflow_count\":");
    output.append_integer(ring_overflow_count);
    output.append(",\"publisher_rejected_record_count\":");
    output.append_integer(rejected_record_count);
    output.append(",\"socket_send_failure_count\":");
    output.append_integer(socket_send_failure_count);
    output.append(",\"records\":[");
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0) {
            output.append(",");
        }
        append_record(output, records[index]);
    }
    output.append("]}");
    return output.export_to(serialized);
}

void FollowerReferenceEpoch::begin_update() noexcept {
    version_.fetch_add(1, std::memory_order_acq_rel);
}

void FollowerReferenceEpoch::complete_update(
    std::int64_t source_monotonic_ns) noexcept {
    source_monotonic_ns_.store(source_monotonic_ns, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_relaxed);
    version_.fetch_add(1, std::memory_order_release);
}

ReferenceRead FollowerReferenceEpoch::begin_read() const noexcept {
    ReferenceRead result;
    result.version = version_.load(std::memory_order_acquire);
    result.generation = generation_.load(std::memory_order_relaxed);
    result.source_monotonic_ns =
        source_monotonic_ns_.load(std::memory_order_relaxed);
    return result;
}

bool FollowerReferenceEpoch::read_still_valid(
    const ReferenceRead& read) const noexcept {
    return (read.version % 2U) == 0U && read.generation > 0 &&
           read.source_monotonic_ns > 0 &&
           version_.load(std::memory_order_acquire) == read.version;
}

OpenArmTelemetryPublisher::OpenArmTelemetryPublisher(
    PublisherConfig config) noexcept
    : config_(std::move(config)) {
    if (config_.socket_path.empty() ||
        config_.socket_path.size() >= sizeof(destination_.sun_path) ||
        !is_start_token(config_.controller_start_token) ||
        !is_lower_sha256(config_.urdf_sha256) ||
        !is_lower_sha256(config_.joint_mapper_sha256) ||
        !is_lower_sha256(config_.pair_contract_sha256) ||
        !is_lower_sha256(config_.controller_binary_sha256) ||
        !is_lower_sha256(config_.controller_source_manifest_sha256) ||
        !is_lower_sha256(config_.follower_config_sha256) ||
        !is_lower_sha256(config_.openarm_sdk_sha256)) {
        return;
    }

    const int descriptor = ::socket(AF_UNIX, SOCK_DGRAM, 0);
    if (descriptor < 0) {
        return;
    }
    const int flags = ::fcntl(descriptor, F_GETFL, 0);
    if (flags < 0 || ::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) < 0) {
        ::close(descriptor);
        return;
    }

    std::memset(&destination_, 0, sizeof(destination_));
    destination_.sun_family = AF_UNIX;
    std::memcpy(
        destination_.sun_path,
        config_.socket_path.data(),
        config_.socket_path.size());
    destination_.sun_path[config_.socket_path.size()] = '\0';
    destination_length_ =
        offsetof(sockaddr_un, sun_path) + config_.socket_path.size() + 1;
#if defined(__APPLE__) || defined(__FreeBSD__)
    destination_.sun_len =
        static_cast<decltype(destination_.sun_len)>(destination_length_);
#endif
    socket_fd_ = descriptor;

    try {
        worker_ = std::thread([this]() noexcept { publisher_loop(); });
        worker_started_.store(true, std::memory_order_release);
    } catch (...) {
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
}

OpenArmTelemetryPublisher::~OpenArmTelemetryPublisher() {
    stop_requested_.store(true, std::memory_order_release);
    if (worker_.joinable()) {
        worker_.join();
    }
    worker_started_.store(false, std::memory_order_release);
    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
    }
}

void OpenArmTelemetryPublisher::try_enqueue(
    const ControlTickRecord& record) noexcept {
    if (!ring_.try_push(record)) {
        ring_overflow_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

void OpenArmTelemetryPublisher::publisher_loop() noexcept {
    lower_publisher_thread_priority();
    auto next_publish = std::chrono::steady_clock::now() +
                        std::chrono::nanoseconds(kEnvelopePeriodNs);
    std::array<ControlTickRecord, kMaxRecordsPerEnvelope> batch{};
    while (!stop_requested_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_until(next_publish);
        std::size_t count = 0;
        while (count < batch.size() && ring_.try_pop(batch[count])) {
            ++count;
        }
        if (count != 0) {
            publish_batch(batch, count);
        }
        next_publish += std::chrono::nanoseconds(kEnvelopePeriodNs);
        const auto now = std::chrono::steady_clock::now();
        if (next_publish < now) {
            next_publish = now + std::chrono::nanoseconds(kEnvelopePeriodNs);
        }
    }

    for (;;) {
        std::size_t count = 0;
        while (count < batch.size() && ring_.try_pop(batch[count])) {
            ++count;
        }
        if (count == 0) {
            break;
        }
        publish_batch(batch, count);
    }
}

void OpenArmTelemetryPublisher::publish_batch(
    const std::array<ControlTickRecord, kMaxRecordsPerEnvelope>& records,
    std::size_t count) noexcept {
    if (socket_fd_ < 0 || count == 0 || count > records.size()) {
        return;
    }
    const std::int64_t published_ns = monotonic_now_ns();
    SerializedControlEnvelope output;
    if (!serialize_control_envelope_json(
            output,
            config_,
            next_envelope_sequence_++,
            static_cast<int>(::getpid()),
            published_ns,
            ring_overflow_count_.load(std::memory_order_relaxed),
            publisher_rejected_record_count_.load(std::memory_order_relaxed),
            socket_send_failure_count_.load(std::memory_order_relaxed),
            records,
            count)) {
        publisher_rejected_record_count_.fetch_add(
            count, std::memory_order_relaxed);
        return;
    }

    int flags = 0;
#if defined(MSG_DONTWAIT)
    flags |= MSG_DONTWAIT;
#endif
#if defined(MSG_NOSIGNAL)
    flags |= MSG_NOSIGNAL;
#endif
    // Per-control-frame SocketCAN outcomes were captured synchronously by the
    // patched SDK before this record entered the ring.  The send below is only
    // telemetry delivery and cannot strengthen those outcomes into motor ACKs.
    const auto send_result = ::sendto(
        socket_fd_,
        output.bytes.data(),
        output.size,
        flags,
        reinterpret_cast<const sockaddr*>(&destination_),
        static_cast<socklen_t>(destination_length_));
    const int send_errno = send_result < 0 ? errno : 0;
    if (send_result < 0) {
        socket_send_failure_count_.fetch_add(1, std::memory_order_relaxed);
    }
    last_send_return_.store(
        static_cast<std::int64_t>(send_result), std::memory_order_release);
    last_send_errno_.store(send_errno, std::memory_order_release);
    send_attempt_count_.fetch_add(1, std::memory_order_release);
}

}  // namespace wam::openarm_bridge
