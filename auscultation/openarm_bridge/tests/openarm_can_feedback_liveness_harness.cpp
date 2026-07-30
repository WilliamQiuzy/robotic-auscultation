#include "openarm_can_feedback_liveness.hpp"

#include <cstdint>

int main() {
    using wam::openarm_bridge::FeedbackLivenessTracker;

    FeedbackLivenessTracker tracker(7);
    if (tracker.snapshot().counter != 0 || tracker.snapshot().generation != 0) {
        return 1;
    }
    if (tracker.observe_successfully_decoded_matching_state(6, 1'000'000'000)) {
        return 2;
    }
    if (tracker.observe_successfully_decoded_matching_state(7, 0)) {
        return 3;
    }
    if (!tracker.observe_successfully_decoded_matching_state(7, 1'000'000'000)) {
        return 4;
    }
    const auto first = tracker.snapshot();
    if (first.motor_id != 7 || first.counter != 1 || first.generation != 1 ||
        first.monotonic_ns != 1'000'000'000 || first.last_gap_ns != 0 ||
        first.gap_violation_count != 0) {
        return 5;
    }
    if (tracker.observe_successfully_decoded_matching_state(7, 999'999'999)) {
        return 6;
    }
    if (!tracker.observe_successfully_decoded_matching_state(7, 1'004'000'000)) {
        return 7;
    }
    const auto second = tracker.snapshot();
    if (second.counter != 2 || second.generation != 2 ||
        second.monotonic_ns != 1'004'000'000 || second.last_gap_ns != 4'000'000 ||
        second.gap_violation_count != 0) {
        return 8;
    }
    if (!tracker.observe_successfully_decoded_matching_state(7, 1'030'000'000)) {
        return 9;
    }
    const auto violation = tracker.snapshot();
    if (violation.last_gap_ns != 26'000'000 ||
        violation.gap_violation_count != 1) {
        return 10;
    }
    return 0;
}
