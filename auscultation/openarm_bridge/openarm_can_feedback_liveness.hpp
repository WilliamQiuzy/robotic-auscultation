#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace wam::openarm_bridge {

struct FeedbackLivenessSnapshot {
    std::uint32_t motor_id = 0;
    std::uint64_t counter = 0;
    std::uint64_t generation = 0;
    std::int64_t monotonic_ns = 0;
    std::int64_t last_gap_ns = 0;
    std::uint64_t gap_violation_count = 0;
};

constexpr std::int64_t kFeedbackLivenessGapBoundNs = 20'000'000;

// Single-threaded OpenArm RX accounting.  The SDK callback invokes this only
// after a STATE frame has both matched the Motor receive ID and decoded
// successfully.  Joint values are deliberately absent: liveness never depends
// on motion or on a q-value change.
class FeedbackLivenessTracker final {
public:
    explicit constexpr FeedbackLivenessTracker(
        std::uint32_t expected_motor_id) noexcept
        : expected_motor_id_(expected_motor_id) {}

    [[nodiscard]] bool observe_successfully_decoded_matching_state(
        std::uint32_t decoded_motor_id,
        std::int64_t monotonic_ns) noexcept {
        if (expected_motor_id_ == 0 || decoded_motor_id != expected_motor_id_ ||
            monotonic_ns <= 0 || monotonic_ns <= monotonic_ns_ ||
            counter_ == std::numeric_limits<std::uint64_t>::max()) {
            return false;
        }
        if (monotonic_ns_ > 0) {
            last_gap_ns_ = monotonic_ns - monotonic_ns_;
            if (last_gap_ns_ > kFeedbackLivenessGapBoundNs &&
                gap_violation_count_ <
                    std::numeric_limits<std::uint64_t>::max()) {
                ++gap_violation_count_;
            }
        }
        monotonic_ns_ = monotonic_ns;
        ++counter_;
        generation_ = counter_;
        return true;
    }

    [[nodiscard]] constexpr FeedbackLivenessSnapshot snapshot() const noexcept {
        return {
            expected_motor_id_, counter_, generation_, monotonic_ns_,
            last_gap_ns_, gap_violation_count_};
    }

private:
    std::uint32_t expected_motor_id_ = 0;
    std::uint64_t counter_ = 0;
    std::uint64_t generation_ = 0;
    std::int64_t monotonic_ns_ = 0;
    std::int64_t last_gap_ns_ = 0;
    std::uint64_t gap_violation_count_ = 0;
};

static_assert(std::is_trivially_copyable_v<FeedbackLivenessSnapshot>);
static_assert(std::is_trivially_copyable_v<FeedbackLivenessTracker>);

}  // namespace wam::openarm_bridge
