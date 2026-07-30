#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <type_traits>

#include <sys/un.h>

namespace wam::openarm_bridge {

constexpr std::size_t kOpenArmJointCount = 7;
constexpr std::size_t kControlRingCapacity = 4096;
constexpr std::size_t kMaxRecordsPerEnvelope = 16;
constexpr std::size_t kMaxSerializedEnvelopeBytes = 131072;
constexpr std::int64_t kEnvelopePeriodNs = 10'000'000;
constexpr std::int64_t kFeedbackMaxAgeNs = 20'000'000;
constexpr std::int64_t kFeedbackMaxGapNs = 20'000'000;

struct PublisherConfig {
    std::string socket_path;
    std::string controller_start_token;
    std::string urdf_sha256;
    std::string joint_mapper_sha256;
    std::string pair_contract_sha256;
    std::string controller_binary_sha256;
    std::string controller_source_manifest_sha256;
    std::string follower_config_sha256;
    std::string openarm_sdk_sha256;
};

// This POD mirrors openarm::damiao_motor::MITParam without depending on the
// SDK layout or retaining a reference to a temporary std::vector element.
struct MitParamPod {
    double kp = 0.0;
    double kd = 0.0;
    double q = 0.0;
    double dq = 0.0;
    double tau = 0.0;
};

// Exactly one follower unilateral_step.  The control thread constructs this
// fixed-size value immediately around both mit_control_all calls, then performs
// one non-blocking SPSC push.  It never serializes, allocates, logs, or uses a
// socket for telemetry.
struct ControlTickRecord {
    std::uint64_t control_tick_seq = 0;
    std::int64_t observation_monotonic_ns = 0;
    std::uint64_t reference_generation = 0;
    std::int64_t reference_source_monotonic_ns = 0;
    bool reference_snapshot_valid = false;
    std::int64_t control_call_begin_monotonic_ns = 0;
    std::int64_t arm_call_return_monotonic_ns = 0;
    std::int64_t gripper_call_begin_monotonic_ns = 0;
    std::int64_t gripper_call_return_monotonic_ns = 0;
    std::int64_t control_call_end_monotonic_ns = 0;
    std::uint32_t arm_tx_expected_frame_count = 0;
    std::uint32_t arm_tx_full_frame_write_count = 0;
    std::uint32_t arm_tx_failed_frame_write_count = 0;
    std::uint32_t gripper_tx_expected_frame_count = 0;
    std::uint32_t gripper_tx_full_frame_write_count = 0;
    std::uint32_t gripper_tx_failed_frame_write_count = 0;
    bool socketcan_tx_enqueue_success = false;
    std::int64_t feedback_snapshot_monotonic_ns = 0;
    std::array<std::uint32_t, kOpenArmJointCount>
        follower_arm_state_rx_motor_ids{};
    std::array<std::uint64_t, kOpenArmJointCount>
        follower_arm_state_rx_counters{};
    std::array<std::uint64_t, kOpenArmJointCount>
        follower_arm_state_rx_generations{};
    std::array<std::int64_t, kOpenArmJointCount>
        follower_arm_state_rx_monotonic_ns{};
    std::array<std::int64_t, kOpenArmJointCount>
        follower_arm_state_rx_last_gap_ns{};
    std::array<std::uint64_t, kOpenArmJointCount>
        follower_arm_state_rx_gap_violation_counts{};
    std::uint32_t follower_gripper_state_rx_motor_id = 0;
    std::uint64_t follower_gripper_state_rx_counter = 0;
    std::uint64_t follower_gripper_state_rx_generation = 0;
    std::int64_t follower_gripper_state_rx_monotonic_ns = 0;
    std::int64_t follower_gripper_state_rx_last_gap_ns = 0;
    std::uint64_t follower_gripper_state_rx_gap_violation_count = 0;
    std::array<double, kOpenArmJointCount> follower_q_rad{};
    std::array<double, kOpenArmJointCount> follower_qd_rad_s{};
    double follower_gripper_q_rad = 0.0;
    double follower_gripper_qd_rad_s = 0.0;
    double follower_gripper_motor_q_rad = 0.0;
    double follower_gripper_motor_qd_rad_s = 0.0;
    std::array<MitParamPod, kOpenArmJointCount> follower_arm_mit{};
    MitParamPod follower_gripper_mit{};
};

struct SerializedControlEnvelope {
    std::array<char, kMaxSerializedEnvelopeBytes> bytes{};
    std::size_t size = 0;
};

static_assert(std::is_trivially_copyable_v<MitParamPod>);
static_assert(std::is_trivially_copyable_v<ControlTickRecord>);
static_assert(std::is_standard_layout_v<ControlTickRecord>);

struct ReferenceRead {
    std::uint64_t version = 0;
    std::uint64_t generation = 0;
    std::int64_t source_monotonic_ns = 0;
};

// Single-writer seqlock binding the AdminThread reference copy to a generation
// and source time.  A follower read that overlaps either arm or hand update is
// marked invalid and rejected by the Python v2 parser.
class FollowerReferenceEpoch final {
public:
    void begin_update() noexcept;
    void complete_update(std::int64_t source_monotonic_ns) noexcept;
    [[nodiscard]] ReferenceRead begin_read() const noexcept;
    [[nodiscard]] bool read_still_valid(const ReferenceRead& read) const noexcept;

private:
    std::atomic<std::uint64_t> version_{0};
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<std::int64_t> source_monotonic_ns_{0};
};

template <typename Value, std::size_t Capacity>
class SpscRing final {
    static_assert(Capacity >= 2);
    static_assert(std::is_trivially_copyable_v<Value>);
    static_assert(std::atomic<std::size_t>::is_always_lock_free);

public:
    [[nodiscard]] bool try_push(const Value& value) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = increment(head);
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        values_[head] = value;
        head_.store(next, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool try_pop(Value& value) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;
        }
        value = values_[tail];
        tail_.store(increment(tail), std::memory_order_release);
        return true;
    }

private:
    static constexpr std::size_t increment(std::size_t index) noexcept {
        return (index + 1) % Capacity;
    }

    std::array<Value, Capacity> values_{};
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

class OpenArmTelemetryPublisher final {
public:
    explicit OpenArmTelemetryPublisher(PublisherConfig config) noexcept;
    ~OpenArmTelemetryPublisher();

    OpenArmTelemetryPublisher(const OpenArmTelemetryPublisher&) = delete;
    OpenArmTelemetryPublisher& operator=(const OpenArmTelemetryPublisher&) = delete;
    OpenArmTelemetryPublisher(OpenArmTelemetryPublisher&&) = delete;
    OpenArmTelemetryPublisher& operator=(OpenArmTelemetryPublisher&&) = delete;

    // The only telemetry operation called by the 1 kHz follower control thread.
    void try_enqueue(const ControlTickRecord& record) noexcept;

    [[nodiscard]] bool enabled() const noexcept {
        return worker_started_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t rejected_record_count() const noexcept {
        return publisher_rejected_record_count_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t send_attempt_count() const noexcept {
        return send_attempt_count_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::int64_t last_send_return() const noexcept {
        return last_send_return_.load(std::memory_order_acquire);
    }
    [[nodiscard]] int last_send_errno() const noexcept {
        return last_send_errno_.load(std::memory_order_acquire);
    }

private:
    void publisher_loop() noexcept;
    void publish_batch(
        const std::array<ControlTickRecord, kMaxRecordsPerEnvelope>& records,
        std::size_t count) noexcept;

    PublisherConfig config_;
    int socket_fd_ = -1;
    sockaddr_un destination_{};
    std::size_t destination_length_ = 0;
    SpscRing<ControlTickRecord, kControlRingCapacity> ring_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> worker_started_{false};
    std::atomic<std::uint64_t> ring_overflow_count_{0};
    std::atomic<std::uint64_t> publisher_rejected_record_count_{0};
    std::atomic<std::uint64_t> socket_send_failure_count_{0};
    std::atomic<std::uint64_t> send_attempt_count_{0};
    std::atomic<std::int64_t> last_send_return_{0};
    std::atomic<int> last_send_errno_{0};
    std::thread worker_;
    std::uint64_t next_envelope_sequence_ = 0;
};

[[nodiscard]] std::int64_t monotonic_now_ns() noexcept;

// Shared by the low-priority publisher and the compile/parser harness.  The
// 1 kHz control thread never calls this serializer.
[[nodiscard]] bool serialize_control_envelope_json(
    SerializedControlEnvelope& output,
    const PublisherConfig& config,
    std::uint64_t envelope_sequence,
    int controller_pid,
    std::int64_t publisher_monotonic_ns,
    std::uint64_t ring_overflow_count,
    std::uint64_t rejected_record_count,
    std::uint64_t socket_send_failure_count,
    const std::array<ControlTickRecord, kMaxRecordsPerEnvelope>& records,
    std::size_t count) noexcept;

}  // namespace wam::openarm_bridge
