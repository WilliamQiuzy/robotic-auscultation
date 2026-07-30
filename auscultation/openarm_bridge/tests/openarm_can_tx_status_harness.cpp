#include "openarm_can_tx_enqueue_status.hpp"

int main() {
    openarm::damiao_motor::MITControlSendStatus arm(7);
    for (int index = 0; index < 7; ++index) {
        arm.observe_full_frame_write(true);
    }
    if (!arm.all_expected_frames_tx_enqueued() ||
        arm.full_frame_write_count != 7 || arm.failed_frame_write_count != 0) {
        return 1;
    }

    openarm::damiao_motor::MITControlSendStatus partial(7);
    for (int index = 0; index < 6; ++index) {
        partial.observe_full_frame_write(true);
    }
    if (partial.all_expected_frames_tx_enqueued()) {
        return 2;
    }

    openarm::damiao_motor::MITControlSendStatus failed(7);
    for (int index = 0; index < 6; ++index) {
        failed.observe_full_frame_write(true);
    }
    failed.observe_full_frame_write(false);
    if (failed.all_expected_frames_tx_enqueued() ||
        failed.full_frame_write_count != 6 || failed.failed_frame_write_count != 1) {
        return 3;
    }
    return 0;
}
