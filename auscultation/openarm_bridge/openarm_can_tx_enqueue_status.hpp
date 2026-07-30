#pragma once

#include <cstddef>

namespace openarm::damiao_motor {

// Aggregate of the bool returned by CANSocket::write_can_frame() or
// write_canfd_frame().  A true observation means write(2) returned the full
// frame size, which proves Linux SocketCAN TX enqueue only.  It is not a motor
// receipt or execution acknowledgement.
struct MITControlSendStatus final {
    explicit constexpr MITControlSendStatus(
        std::size_t expected_frames = 0) noexcept
        : expected_frame_count(expected_frames) {}

    constexpr void observe_full_frame_write(bool full_frame_write) noexcept {
        ++observed_frame_count;
        if (full_frame_write) {
            ++full_frame_write_count;
        } else {
            ++failed_frame_write_count;
        }
    }

    [[nodiscard]] constexpr bool all_expected_frames_tx_enqueued() const
        noexcept {
        return expected_frame_count > 0 &&
               observed_frame_count == expected_frame_count &&
               full_frame_write_count == expected_frame_count &&
               failed_frame_write_count == 0;
    }

    std::size_t expected_frame_count = 0;
    std::size_t observed_frame_count = 0;
    std::size_t full_frame_write_count = 0;
    std::size_t failed_frame_write_count = 0;
};

}  // namespace openarm::damiao_motor
