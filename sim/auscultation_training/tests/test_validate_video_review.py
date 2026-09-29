"""CPU-only checks of the final video schedule and recorded-pose validation."""
from copy import deepcopy

import numpy as np
import pytest

from sim.auscultation_training.render_expert import playback_frames
from sim.auscultation_training.results.low_elbow_final_v5.validate_video import validate_frame_schedule


def recording(speed=2):
    rows = []
    for phase, subphase, count in [("APPROACH", "transfer", 480),
                                   ("HOLD", "contact_cycle", 720),
                                   ("DONE", "return_home", 480)]:
        for _ in range(count):
            rows.append({"target": "R5", "phase": phase, "subphase": subphase,
                         "time_s": (len(rows) + 1) / 240,
                         "lip_N": 4. if phase == "HOLD" else 0.,
                         "diaphragm_N": 0., "contact_rim_N": 0., "gap_m": .05})
    indices = np.arange(7, len(rows), 8)
    specs = ([{"measurement_row": int(indices[0]), "speed": 0}] * 30
             + playback_frames(rows, int(indices[0]), 15, max_motion_speed=speed)
             + [{"measurement_row": len(rows) - 1, "speed": 0}] * 30)
    frames = []
    for spec in specs:
        row = spec["measurement_row"]
        pose_index = int(np.searchsorted(indices, row, side="right") - 1)
        pose_row = int(indices[pose_index])
        frames.append({"measurement_row": row, "playback_speed": spec["speed"],
                       "pose_index": pose_index, "pose_row": pose_row,
                       "pose_age_s": rows[row]["time_s"] - rows[pose_row]["time_s"]})
    return rows, indices, frames


@pytest.mark.parametrize("speed", [2, 8])
def test_exact_recorded_schedule_and_pose_mapping_are_accepted(speed):
    rows, indices, frames = recording(speed)
    validate_frame_schedule(rows, indices, frames, 15, speed)


def test_repeated_hold_sample_cannot_claim_normal_speed_dwell():
    rows, indices, frames = recording()
    hold = [i for i, frame in enumerate(frames)
            if rows[frame["measurement_row"]]["phase"] == "HOLD"]
    assert len(hold) == 45
    for index in hold[1:]:
        frames[index] = deepcopy(frames[hold[0]])
    with pytest.raises(AssertionError, match="playback schedule"):
        validate_frame_schedule(rows, indices, frames, 15, 2)


def test_old_valid_pose_cannot_be_reused_for_later_measurement():
    rows, indices, frames = recording()
    frame = frames[-1]
    frame.update(pose_index=0, pose_row=int(indices[0]),
                 pose_age_s=rows[frame["measurement_row"]]["time_s"] - rows[int(indices[0])]["time_s"])
    with pytest.raises(AssertionError, match="nearest preceding"):
        validate_frame_schedule(rows, indices, frames, 15, 2)


def test_pose_age_must_equal_source_timestamps():
    rows, indices, frames = recording()
    frames[30]["pose_age_s"] += .001
    with pytest.raises(AssertionError, match="pose age"):
        validate_frame_schedule(rows, indices, frames, 15, 2)


def test_missing_freeze_frame_is_rejected():
    rows, indices, frames = recording()
    frames.pop()
    with pytest.raises(AssertionError, match="playback schedule"):
        validate_frame_schedule(rows, indices, frames, 15, 2)


def test_latest_available_pose_must_still_be_fresh():
    rows, indices, frames = recording()
    indices = indices[:-2]
    for frame in frames:
        row = frame["measurement_row"]
        k = int(np.searchsorted(indices, row, side="right") - 1)
        frame.update(pose_index=k, pose_row=int(indices[k]),
                     pose_age_s=rows[row]["time_s"] - rows[int(indices[k])]["time_s"])
    with pytest.raises(AssertionError, match="too far apart"):
        validate_frame_schedule(rows, indices, frames, 15, 2)
