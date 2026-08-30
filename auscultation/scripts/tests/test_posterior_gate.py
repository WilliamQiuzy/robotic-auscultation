from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from posterior_gate import (  # noqa: E402
    BackConfirmationError,
    OperatorConfirmedBackGate,
    UltralyticsPersonGate,
    select_single_person,
)


def _person(*, posterior=True, confidence=0.95, scale=1.0, face=0.05):
    """Minimal COCO-17 pose evidence in a 1000x1000 image."""
    box = np.array([150.0, 80.0, 850.0, 980.0])
    xy = np.zeros((17, 2), dtype=float)
    conf = np.full(17, 0.05, dtype=float)
    left_x, right_x = ((300.0, 700.0) if posterior else (700.0, 300.0))
    xy[5], xy[6] = [left_x, 330.0], [right_x, 330.0]
    xy[11], xy[12] = [left_x + 70.0, 750.0], [right_x - 70.0, 750.0]
    conf[[5, 6, 11, 12]] = 0.96
    conf[:3] = face
    if scale != 1.0:
        center = np.array([500.0, 500.0])
        box[:2] = center + (box[:2] - center) * scale
        box[2:] = center + (box[2:] - center) * scale
        xy = center + (xy - center) * scale
    return {
        "box": box,
        "box_confidence": confidence,
        "keypoints_xy": xy,
        "keypoints_confidence": conf,
    }


def _observed_back_person():
    """Pose evidence reproduced from the Lingbox back-view preview."""
    image_area = 640.0 * 576.0
    box_width = 214.5
    box_height = 0.0562 * image_area / box_width
    xy = np.zeros((17, 2), dtype=float)
    confidence = np.full(17, 0.01, dtype=float)
    xy[5], xy[6] = [106.6, 120.0], [284.4, 120.0]
    confidence[[5, 6]] = 0.0629
    confidence[:3] = 0.0672
    return {
        "box": np.array([90.0, 80.0, 90.0 + box_width, 80.0 + box_height]),
        "box_confidence": 0.6307,
        "keypoints_xy": xy,
        "keypoints_confidence": confidence,
    }


def _frame(index, t):
    return SimpleNamespace(index=index, t=t)


def test_person_presence_uses_the_lingbox_box_without_guessing_orientation():
    result = select_single_person(
        [_observed_back_person()], image_shape=(576, 640),
    )

    assert result["person_present"] is True
    assert result["reason"] == "ONE_PERSON"
    assert result["box_confidence"] == 0.6307
    assert result["box_area_fraction"] == pytest.approx(0.0562)


def test_person_presence_is_identical_for_front_and_back_pose_keypoints():
    posterior = select_single_person(
        [_person(posterior=True)], image_shape=(1000, 1000),
    )
    frontal = select_single_person(
        [_person(posterior=False)], image_shape=(1000, 1000),
    )

    assert posterior["person_present"] is frontal["person_present"] is True
    assert posterior["reason"] == frontal["reason"] == "ONE_PERSON"


def test_default_person_gate_accepts_a_partial_low_confidence_body():
    partial = {
        "box": np.array([100.0, 100.0, 300.0, 150.0]),
        "box_confidence": 0.25,
    }

    result = select_single_person([partial], image_shape=(1000, 1000))

    assert result["person_present"] is True
    assert result["reason"] == "ONE_PERSON"
    assert result["box_area_fraction"] == pytest.approx(0.01)


def test_ultralytics_gate_requests_low_confidence_person_candidates(tmp_path):
    checkpoint = tmp_path / "person.pt"
    checkpoint.write_bytes(b"test checkpoint")
    calls = []

    class Model:
        def predict(self, _image, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=None)]

    gate = UltralyticsPersonGate(
        str(checkpoint), model_factory=lambda _path: Model(),
    )
    frame = SimpleNamespace(
        frame=np.zeros((64, 64, 3), dtype=np.uint8), index=1,
    )

    result = gate.assess(frame)

    assert result["reason"] == "NO_PERSON"
    assert calls[0]["conf"] == pytest.approx(0.10)


def test_operator_confirmation_is_required_and_bound_to_the_current_person():
    now = [10.0]
    decisions = {
        1: select_single_person([_person()], image_shape=(1000, 1000)),
        2: select_single_person([_person()], image_shape=(1000, 1000)),
    }
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **decisions[frame.index], "frame_index": frame.index,
    })
    gate = OperatorConfirmedBackGate(raw_gate, clock=lambda: now[0])

    before = gate.assess(_frame(1, 10.0))
    assert before["posterior_present"] is False
    assert before["reason"] == "BACK_CONFIRMATION_REQUIRED"

    confirmed = gate.confirm_back(preview_frame_index=1)
    assert confirmed["ok"] is True
    now[0] += 0.05
    after = gate.assess(_frame(2, 10.05))
    assert after["posterior_present"] is True
    assert after["reason"] == "OPERATOR_CONFIRMED_BACK"
    assert after["confirmation_frame_index"] == 1


def test_default_back_confirmation_tracks_moderate_box_motion():
    now = [15.0]
    first = select_single_person([_person()], image_shape=(1000, 1000))
    moved = dict(first)
    moved["box"] = [370.0, 80.0, 1070.0, 980.0]
    decisions = iter((first, moved))
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **next(decisions), "frame_index": frame.index,
    })
    gate = OperatorConfirmedBackGate(raw_gate, clock=lambda: now[0])

    gate.assess(_frame(1, 15.0))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    now[0] += 0.05
    result = gate.assess(_frame(2, 15.05))

    assert result["posterior_present"] is True
    assert result["reason"] == "OPERATOR_CONFIRMED_BACK"


def test_default_back_confirmation_survives_four_transient_misses():
    now = [16.0]
    one = select_single_person([_person()], image_shape=(1000, 1000))
    none = select_single_person([], image_shape=(1000, 1000))
    decisions = iter((one, none, none, none, none))
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **next(decisions), "frame_index": frame.index,
    })
    gate = OperatorConfirmedBackGate(raw_gate, clock=lambda: now[0])

    gate.assess(_frame(1, 16.0))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    for index in range(2, 6):
        now[0] += 0.05
        result = gate.assess(_frame(index, 16.0 + index * 0.05))
        assert result["posterior_present"] is False
        assert gate.confirmation_snapshot()["active"] is True


def test_operator_confirmation_clears_immediately_for_multiple_or_changed_people():
    now = [20.0]
    one = select_single_person([_person()], image_shape=(1000, 1000))
    two = select_single_person(
        [_person(), _person()], image_shape=(1000, 1000),
    )
    moved = dict(one)
    moved["box"] = [0.0, 0.0, 180.0, 220.0]
    decisions = iter((one, two, one, moved))
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **next(decisions), "frame_index": frame.index,
    })
    gate = OperatorConfirmedBackGate(raw_gate, clock=lambda: now[0])

    gate.assess(_frame(1, 20.0))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    ambiguous = gate.assess(_frame(2, 20.01))
    assert ambiguous["posterior_present"] is False
    assert ambiguous["reason"] == "AMBIGUOUS_PEOPLE"
    assert gate.confirmation_snapshot()["active"] is False

    gate.assess(_frame(3, 20.02))
    assert gate.confirm_back(preview_frame_index=3)["ok"] is True
    changed = gate.assess(_frame(4, 20.03))
    assert changed["posterior_present"] is False
    assert changed["reason"] == "CONFIRMED_PERSON_CHANGED"
    assert gate.confirmation_snapshot()["active"] is False


def test_operator_confirmation_hides_immediately_then_expires_after_three_misses():
    now = [30.0]
    one = select_single_person([_person()], image_shape=(1000, 1000))
    none = select_single_person([], image_shape=(1000, 1000))
    decisions = iter((one, none, none, none, one))
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **next(decisions), "frame_index": frame.index,
    })
    gate = OperatorConfirmedBackGate(
        raw_gate, max_missed_frames=3, clock=lambda: now[0],
    )

    gate.assess(_frame(1, 30.0))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    for index in (2, 3):
        result = gate.assess(_frame(index, 30.0 + index / 100.0))
        assert result["posterior_present"] is False
        assert gate.confirmation_snapshot()["active"] is True
    expired = gate.assess(_frame(4, 30.04))
    assert expired["posterior_present"] is False
    assert expired["reason"] == "CONFIRMATION_TARGET_LOST"
    assert gate.confirmation_snapshot()["active"] is False
    reacquired = gate.assess(_frame(5, 30.05))
    assert reacquired["reason"] == "BACK_CONFIRMATION_REQUIRED"


def test_person_gate_exception_clears_an_existing_operator_confirmation():
    now = [40.0]
    one = select_single_person([_person()], image_shape=(1000, 1000))

    class RawGate:
        def __init__(self):
            self.fail = False

        def assess(self, frame):
            if self.fail:
                raise RuntimeError("detector failed")
            return {**one, "frame_index": frame.index}

    raw_gate = RawGate()
    gate = OperatorConfirmedBackGate(raw_gate, clock=lambda: now[0])
    gate.assess(_frame(1, 40.0))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    gate.assess(_frame(2, 40.0))
    assert gate.confirmation_snapshot()["confirmed"] is True

    raw_gate.fail = True
    with pytest.raises(RuntimeError, match="detector failed"):
        gate.assess(_frame(3, 40.0))
    snapshot = gate.confirmation_snapshot()
    assert snapshot["active"] is False
    assert snapshot["reason"] == "PERSON_GATE_ERROR"


def test_one_dominant_person_plus_one_malformed_detection_stays_present():
    # 2026-08-29 (operator-approved): a glitched detection row is recorded but
    # no longer voids a clean dominant patient; fail-closed survives below.
    malformed = {
        "box": [10.0, 10.0, 600.0, 900.0],
        "box_confidence": float("nan"),
    }

    result = select_single_person(
        [_person(), malformed], image_shape=(1000, 1000),
    )

    assert result["person_present"] is True
    assert result["reason"] == "ONE_PERSON"
    assert result["invalid_detections"] == 1
    assert result["bystander_people"] == 0


def test_malformed_detection_without_a_dominant_person_still_fails_closed():
    malformed = {
        "box": [10.0, 10.0, 600.0, 900.0],
        "box_confidence": float("nan"),
    }

    result = select_single_person([malformed], image_shape=(1000, 1000))

    assert result["person_present"] is False
    assert result["reason"] == "PERSON_DETECTION_INVALID"
    assert result["invalid_detections"] == 1


def test_distant_bystander_no_longer_voids_a_dominant_patient():
    # A colleague passing 4-5 m behind the patient projects to a small but
    # confidently detected box; the dominant patient must stay confirmable.
    bystander = {
        "box": [40.0, 40.0, 160.0, 160.0],  # 1.44% of the frame
        "box_confidence": 0.55,
    }

    result = select_single_person(
        [_person(), bystander], image_shape=(1000, 1000),
    )

    assert result["person_present"] is True
    assert result["reason"] == "ONE_PERSON"
    assert result["bystander_people"] == 1


def test_two_dominant_people_are_still_ambiguous():
    shifted = dict(_person())
    shifted["box"] = [500.0, 100.0, 900.0, 800.0]

    result = select_single_person(
        [_person(), shifted], image_shape=(1000, 1000),
    )

    assert result["person_present"] is False
    assert result["reason"] == "AMBIGUOUS_PEOPLE"


def test_confirmation_requires_the_displayed_preview_frame_and_a_newer_model_frame():
    now = [50.0]
    one = select_single_person([_person()], image_shape=(1000, 1000))
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **one, "frame_index": frame.index,
    })
    gate = OperatorConfirmedBackGate(raw_gate, clock=lambda: now[0])

    gate.assess(_frame(10, 50.0))
    assert gate.confirm_back(preview_frame_index=9)["ok"] is False
    assert gate.confirm_back(preview_frame_index=41)["ok"] is False
    assert gate.confirm_back(preview_frame_index=10)["ok"] is True

    replay = gate.assess(_frame(10, 50.0))
    assert replay["posterior_present"] is False
    assert replay["reason"] == "BACK_CONFIRMATION_PENDING_FRAME"
    newer = gate.assess(_frame(11, 50.01))
    assert newer["posterior_present"] is True


def test_active_confirmation_is_held_while_the_ingest_stream_is_stalled():
    """2026-08-30 soak forensics: the Start transition stalls the person-gate
    stream for >5 s and used to age the confirmation out on EVERY Start. A
    total stream stall now HOLDS the confirmation (active, not current)."""
    now = [60.0]
    one = select_single_person([_person()], image_shape=(1000, 1000))
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **one, "frame_index": frame.index,
    })
    gate = OperatorConfirmedBackGate(
        raw_gate, max_observation_age_s=0.75, clock=lambda: now[0],
    )
    gate.assess(_frame(1, 60.0))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    gate.assess(_frame(2, 60.05))
    assert gate.confirmation_snapshot()["confirmed"] is True

    now[0] = 60.81                       # no assess() since 60.05: stalled
    snapshot = gate.confirmation_snapshot()
    assert snapshot["active"] is True
    assert snapshot["confirmed"] is False        # held, not current
    assert snapshot["reason"] == "CONFIRMATION_HELD_STREAM_STALLED"

    now[0] = 66.0                        # still stalled, still held
    held = gate.confirmation_snapshot()
    assert held["active"] is True
    assert held["reason"] == "CONFIRMATION_HELD_STREAM_STALLED"

    result = gate.assess(_frame(3, 66.0))        # stream resumes, person back
    assert result["posterior_present"] is True
    assert result["reason"] == "OPERATOR_CONFIRMED_BACK"
    assert gate.confirmation_snapshot()["confirmed"] is True


def test_active_confirmation_expires_on_stale_backlog_with_live_stream():
    """The original staleness protection survives: decisions ARRIVING whose
    frames are old (stale backlog) still clear the confirmation."""
    now = [60.0]
    one = select_single_person([_person()], image_shape=(1000, 1000))
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **one, "frame_index": frame.index,
    })
    gate = OperatorConfirmedBackGate(
        raw_gate, max_observation_age_s=0.75, clock=lambda: now[0],
    )
    gate.assess(_frame(1, 60.0))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    gate.assess(_frame(2, 60.05))
    assert gate.confirmation_snapshot()["confirmed"] is True

    now[0] = 62.0
    gate.assess(_frame(3, 60.1))         # ingest alive, frame 1.9 s old
    snapshot = gate.confirmation_snapshot()
    assert snapshot["active"] is False
    # The clearing emits CONFIRMATION_OBSERVATION_STALE; a later snapshot of
    # the already-cleared gate may restate it as PERSON_OBSERVATION_STALE.
    assert snapshot["reason"] in (
        "CONFIRMATION_OBSERVATION_STALE", "PERSON_OBSERVATION_STALE",
    )


def test_held_confirmation_clears_at_the_stall_hold_cap():
    now = [60.0]
    one = select_single_person([_person()], image_shape=(1000, 1000))
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **one, "frame_index": frame.index,
    })
    gate = OperatorConfirmedBackGate(
        raw_gate, max_observation_age_s=0.75, clock=lambda: now[0],
    )
    gate.assess(_frame(1, 60.0))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    gate.assess(_frame(2, 60.05))

    now[0] = 60.05 + 31.0                # beyond _CONFIRMATION_STALL_HOLD_MAX_S
    snapshot = gate.confirmation_snapshot()
    assert snapshot["active"] is False
    assert snapshot["reason"] == "CONFIRMATION_OBSERVATION_STALE"


def test_stall_resume_without_person_walks_the_missed_frames_ladder():
    now = [60.0]
    one = select_single_person([_person()], image_shape=(1000, 1000))
    absent = {"person_present": False, "reason": "NO_PERSON"}
    mode = {"value": "person"}
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **(one if mode["value"] == "person" else absent),
        "frame_index": frame.index,
    })
    gate = OperatorConfirmedBackGate(
        raw_gate, max_missed_frames=3, max_observation_age_s=0.75,
        clock=lambda: now[0],
    )
    gate.assess(_frame(1, 60.0))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    gate.assess(_frame(2, 60.05))

    now[0] = 66.0                        # stall: held
    assert gate.confirmation_snapshot()["active"] is True

    mode["value"] = "absent"             # resume with the patient GONE
    for offset in range(3):
        gate.assess(_frame(3 + offset, 66.0 + offset * 0.05))
        now[0] = 66.0 + (offset + 1) * 0.05
    snapshot = gate.confirmation_snapshot()
    assert snapshot["active"] is False
    assert snapshot["reason"] == "CONFIRMATION_TARGET_LOST"


def test_start_claim_cannot_survive_a_loss_and_reacquisition_race():
    now = [70.0]
    one = select_single_person([_person()], image_shape=(1000, 1000))
    none = select_single_person([], image_shape=(1000, 1000))
    decisions = iter((one, one, none, one))
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **next(decisions), "frame_index": frame.index,
    })
    gate = OperatorConfirmedBackGate(raw_gate, clock=lambda: now[0])
    gate.assess(_frame(1, 70.0))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    gate.assess(_frame(2, 70.01))
    token, _snapshot = gate.begin_start_confirmation()

    gate.assess(_frame(3, 70.02))
    gate.assess(_frame(4, 70.03))
    with pytest.raises(BackConfirmationError, match="changed during Start"):
        gate.finish_start_confirmation(token)
