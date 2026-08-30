from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from perception_worker import (  # noqa: E402
    AssetIntegrityError,
    LatestFrameWorker,
    TrackerProcessor,
    tracker_result_to_dict,
    verify_file_sha256,
)
import perception_worker as perception_worker_module  # noqa: E402


_ROBOT_BUNDLE_HELPER_PATH = (
    Path(__file__).resolve().parents[2]
    / "ausc"
    / "perception"
    / "robot"
    / "tests"
    / "test_frame_evidence_bundle.py"
)
_ROBOT_BUNDLE_SPEC = importlib.util.spec_from_file_location(
    "_wam_perception_worker_bundle_helpers", _ROBOT_BUNDLE_HELPER_PATH,
)
assert _ROBOT_BUNDLE_SPEC is not None and _ROBOT_BUNDLE_SPEC.loader is not None
_ROBOT_BUNDLE_HELPERS = importlib.util.module_from_spec(_ROBOT_BUNDLE_SPEC)
_ROBOT_BUNDLE_SPEC.loader.exec_module(_ROBOT_BUNDLE_HELPERS)


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached")


def test_stop_reports_failure_when_worker_thread_is_stuck():
    entered = threading.Event()
    release = threading.Event()

    def processor(frame):
        entered.set()
        release.wait(5.0)  # hang until released (simulates a wedged GPU pass)
        return {"frame_index": frame.index, "points": []}

    worker = LatestFrameWorker(processor, name="stuck-worker").start()
    try:
        worker.submit(SimpleNamespace(index=1, t=1.0))
        assert entered.wait(1.0)
        # The thread is stuck inside the processor: stop() cannot join it and must
        # report failure so the caller refuses to spawn a replacement on top of it.
        assert worker.stop(timeout=0.2) is False
        assert worker.is_alive() is True
    finally:
        release.set()
    _wait_until(lambda: not worker.is_alive(), timeout=2.0)
    assert worker.stop(timeout=1.0) is True  # now it has exited


def test_stop_reports_success_on_clean_exit():
    worker = LatestFrameWorker(
        lambda frame: {"frame_index": frame.index, "points": []}, name="clean-worker"
    ).start()
    worker.submit(SimpleNamespace(index=1, t=1.0))
    assert worker.stop(timeout=1.0) is True
    assert worker.is_alive() is False


def test_stop_before_start_reports_success():
    worker = LatestFrameWorker(lambda frame: {"frame_index": frame.index, "points": []})
    assert worker.stop(timeout=0.1) is True
    assert worker.is_alive() is False


def test_asset_hash_mismatch_fails_before_model_load(tmp_path):
    asset = tmp_path / "best.pt"
    asset.write_bytes(b"checkpoint")
    with pytest.raises(AssetIntegrityError, match="SHA-256 mismatch"):
        verify_file_sha256(asset, "0" * 64, label="v9 checkpoint")


def test_latest_frame_worker_is_nonblocking_and_drops_intermediate_frames():
    entered = threading.Event()
    release = threading.Event()
    processed = []

    def processor(frame):
        processed.append(frame.index)
        if frame.index == 1:
            entered.set()
            assert release.wait(1.0)
        return {"frame_index": frame.index, "back_present": True, "points": []}

    worker = LatestFrameWorker(processor, name="test-worker").start()
    try:
        worker.submit(SimpleNamespace(index=1, t=1.0))
        assert entered.wait(1.0)
        started = time.monotonic()
        worker.submit(SimpleNamespace(index=2, t=2.0))
        worker.submit(SimpleNamespace(index=3, t=3.0))
        assert time.monotonic() - started < 0.05
        release.set()
        _wait_until(lambda: (worker.latest() or {}).get("frame_index") == 3)
        assert processed == [1, 3]
        assert worker.health()["dropped_before_inference"] >= 1
    finally:
        worker.stop()


def test_latest_frame_worker_drain_waits_for_the_last_accepted_item():
    entered = threading.Event()
    release = threading.Event()

    def processor(frame):
        entered.set()
        assert release.wait(1.0)
        return {"frame_index": frame.index, "back_present": False, "points": []}

    worker = LatestFrameWorker(processor, name="drain-worker").start()
    try:
        assert worker.submit(SimpleNamespace(index=11, t=1.0)) is True
        assert entered.wait(1.0)
        assert worker.drain(timeout=0.01) is False
        release.set()
        assert worker.drain(timeout=1.0) is True
        assert worker.health()["last_completed_index"] == 11
    finally:
        release.set()
        worker.stop()


def test_bounded_fifo_worker_never_overwrites_and_drains_exact_accepted_prefix():
    entered = threading.Event()
    release = threading.Event()
    processed = []

    def processor(frame):
        processed.append(frame.index)
        if frame.index == 1:
            entered.set()
            assert release.wait(1.0)
        return {"frame_index": frame.index}

    worker = perception_worker_module.BoundedFifoWorker(
        processor, capacity=2, name="capture-fifo",
    ).start()
    try:
        assert worker.begin_interval() is True
        assert worker.submit(SimpleNamespace(index=1, t=1.0)) is True
        assert entered.wait(1.0)
        assert worker.submit(SimpleNamespace(index=2, t=2.0)) is True
        assert worker.submit(SimpleNamespace(index=3, t=3.0)) is True
        started = time.monotonic()
        assert worker.submit(SimpleNamespace(index=4, t=4.0)) is False
        assert time.monotonic() - started < 0.05
        target = worker.close_interval()
        assert target == 3
        health = worker.health()
        assert health["overflow_rejections"] == 1
        assert health["queue_depth"] == 2
        assert health["queue_capacity"] == 2
        release.set()
        assert worker.drain(through_index=target, timeout=1.0) is True
        assert processed == [1, 2, 3]
        assert worker.health()["last_completed_index"] == 3
    finally:
        release.set()
        worker.stop()


def test_bounded_fifo_worker_rejects_frames_outside_an_explicit_interval():
    worker = perception_worker_module.BoundedFifoWorker(
        lambda frame: {"frame_index": frame.index}, capacity=2,
        name="capture-boundary",
    ).start()
    try:
        assert worker.submit(SimpleNamespace(index=5, t=5.0)) is False
        assert worker.begin_interval() is True
        assert worker.submit(SimpleNamespace(index=6, t=6.0)) is True
        target = worker.close_interval()
        assert target == 6
        assert worker.submit(SimpleNamespace(index=7, t=7.0)) is False
        assert worker.drain(through_index=target, timeout=1.0) is True
        # The pre-Start rejection belongs to the previous/no interval and is
        # intentionally reset. Completion proof is interval-local.
        assert worker.health()["outside_interval_rejections"] == 1
    finally:
        worker.stop()


def test_bounded_fifo_health_counters_are_interval_local():
    worker = perception_worker_module.BoundedFifoWorker(
        lambda frame: {"frame_index": frame.index}, capacity=1,
        name="capture-interval-counters",
    ).start()
    try:
        assert worker.submit(SimpleNamespace(index=1, t=1.0)) is False
        assert worker.begin_interval() is True
        assert worker.submit(SimpleNamespace(index=2, t=2.0)) is True
        target = worker.close_interval()
        assert worker.drain(through_index=target, timeout=1.0) is True
        assert worker.health()["outside_interval_rejections"] == 0

        assert worker.submit(SimpleNamespace(index=3, t=3.0)) is False
        assert worker.begin_interval() is True
        health = worker.health()
        assert health["accepted"] == 0
        assert health["processed"] == 0
        assert health["overflow_rejections"] == 0
        assert health["outside_interval_rejections"] == 0
        assert health["non_monotonic_rejections"] == 0
        assert health["max_queue_depth"] == 0
    finally:
        worker.stop()


def test_worker_error_is_visible_and_clears_stale_points():
    def processor(_frame):
        raise RuntimeError("cuda failed")

    worker = LatestFrameWorker(processor, name="test-worker").start()
    try:
        worker.submit(SimpleNamespace(index=9, t=9.0))
        _wait_until(lambda: worker.health()["status"] == "ERROR")
        assert worker.latest()["back_present"] is False
        assert worker.latest()["points"] == []
        assert "cuda failed" in worker.health()["error"]
    finally:
        worker.stop()


def test_worker_rejects_non_increasing_camera_frame_index():
    worker = LatestFrameWorker(
        lambda frame: {"frame_index": frame.index, "back_present": False, "points": []},
        name="test-worker",
    ).start()
    try:
        assert worker.submit(SimpleNamespace(index=4, t=4.0)) is True
        assert worker.submit(SimpleNamespace(index=4, t=4.1)) is False
        assert worker.submit(SimpleNamespace(index=3, t=4.2)) is False
    finally:
        worker.stop()


def _camera_frame(index=5):
    return SimpleNamespace(
        index=index,
        t=12.5,
        status="OK",
        frame=object(),
        depth=object(),
        depth_scale=1.0,
    )


def test_tracker_result_json_preserves_same_frame_occlusion_evidence():
    result = tracker_result_to_dict(SimpleNamespace(
        back_present=True,
        initialized=True,
        detector_confidence=0.9,
        reason="TRACKING",
        pose_measured_this_frame=True,
        pose_support_mask=(True, False),
        pose_geometry_accepted=True,
        pose_uncertainty_bound_m=0.006,
        sim3_rotation_camera_from_patient=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        sim3_scale=1.0,
        sim3_translation_camera_from_patient_m=(0.01, 0.0, 0.0),
        points=(SimpleNamespace(
            name="R1",
            pixel=(10.0, 20.0),
            xyz_camera=(0.1, 0.2, 0.9),
            xyz_patient=(0.09, 0.2, 0.9),
            status="TRACKING",
            measured=True,
            position_std=0.003,
            measurement_residual_m=0.001,
            robot_occluded=False,
            direct_geometry_accepted=True,
        ),),
    ))

    assert result["pose_measured_this_frame"] is True
    assert result["pose_support_mask"] == [True, False]
    assert result["pose_geometry_accepted"] is True
    assert result["pose_uncertainty_bound_m"] == pytest.approx(0.006)
    assert result["points"][0]["xyz_patient"] == [0.09, 0.2, 0.9]
    assert result["points"][0]["measurement_residual_m"] == pytest.approx(0.001)
    assert result["points"][0]["direct_geometry_accepted"] is True


def test_tracker_processor_hides_immediately_and_skips_landmark_model_without_person():
    class Tracker:
        def __init__(self):
            self.reset_reasons = []

        def reset(self, reason):
            self.reset_reasons.append(reason)

        def process(self, *_args, **_kwargs):
            raise AssertionError("landmark model must not run without a posterior person")

    tracker = Tracker()
    gate = SimpleNamespace(assess=lambda _frame: {
        "person_present": False,
        "posterior_present": False,
        "reason": "NO_POSTERIOR_PERSON",
        "qualified_people": 0,
    })
    processor = TrackerProcessor(tracker, lambda value: value, presence_gate=gate)

    result = processor(_camera_frame())

    assert result["back_present"] is False
    assert result["initialized"] is False
    assert result["points"] == []
    assert result["reason"] == "NO_POSTERIOR_PERSON"
    assert result["posterior_gate"]["qualified_people"] == 0
    # A single pose-gate miss must hide every point on that same result, but it
    # must not throw away a stable temporal track immediately.  Resetting on
    # one intermittent YOLO miss caused visible reacquisition flicker.
    assert tracker.reset_reasons == []


def test_tracker_processor_runs_private_landmark_probe_for_one_unconfirmed_person():
    class Tracker:
        def __init__(self):
            self.calls = 0

        def process(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                back_present=True,
                initialized=True,
                detector_confidence=0.94,
                reason="TRACKING",
                points=[SimpleNamespace(
                    name=f"P{index}",
                    pixel=(100.0 + index, 200.0 + index),
                    xyz_camera=(0.1, 0.2, 0.9),
                    status="TRACKING",
                    measured=True,
                    position_std=0.003,
                ) for index in range(10)],
            )

    tracker = Tracker()
    gate = SimpleNamespace(assess=lambda frame: {
        "frame_index": frame.index,
        "person_present": True,
        "posterior_present": False,
        "operator_confirmed": False,
        "reason": "BACK_CONFIRMATION_REQUIRED",
        "qualified_people": 1,
    })
    processor = TrackerProcessor(tracker, lambda value: value, presence_gate=gate)

    result = processor(_camera_frame(index=17))

    assert tracker.calls == 1
    assert result["back_present"] is False
    assert result["initialized"] is False
    assert result["points"] == []
    assert result["reason"] == "BACK_CONFIRMATION_REQUIRED"
    assert result["landmark_probe"] == {
        "executed": True,
        "frame_index": 17,
        "raw_initialized": True,
        "raw_point_count": 10,
        "raw_reason": "TRACKING",
    }


def test_tracker_processor_resets_private_probe_before_first_confirmed_publish():
    events = []

    class Tracker:
        def reset(self, reason):
            events.append(("reset", reason))

        def process(self, *_args, **_kwargs):
            events.append(("process",))
            return SimpleNamespace(
                back_present=True,
                initialized=True,
                detector_confidence=0.9,
                reason="TRACKING",
                points=[],
            )

    decisions = iter([
        {
            "person_present": True,
            "posterior_present": False,
            "operator_confirmed": False,
            "reason": "BACK_CONFIRMATION_REQUIRED",
            "qualified_people": 1,
        },
        {
            "person_present": True,
            "posterior_present": True,
            "operator_confirmed": True,
            "reason": "OPERATOR_CONFIRMED_BACK",
            "qualified_people": 1,
            "confirmation_frame_index": 17,
            "back_confirmation": {
                "confirmation_preview_frame_index": 18,
            },
        },
    ])
    gate = SimpleNamespace(assess=lambda frame: {
        "frame_index": frame.index,
        **next(decisions),
    })
    processor = TrackerProcessor(Tracker(), lambda value: value, presence_gate=gate)

    unconfirmed = processor(_camera_frame(index=18))
    confirmed = processor(_camera_frame(index=19))

    assert unconfirmed["points"] == []
    assert unconfirmed["landmark_probe"]["executed"] is True
    assert confirmed["posterior_gate"]["operator_confirmed"] is True
    assert events == [
        ("process",),
        ("reset", "BACK_CONFIRMATION_ACTIVATED"),
        ("process",),
    ]


def test_tracker_processor_does_not_reset_same_confirmation_after_one_gate_miss():
    events = []

    class Tracker:
        def reset(self, reason):
            events.append(("reset", reason))

        def process(self, *_args, **_kwargs):
            events.append(("process",))
            return SimpleNamespace(
                back_present=True,
                initialized=True,
                detector_confidence=0.9,
                reason="TRACKING",
                points=[],
            )

    confirmed = {
        "person_present": True,
        "posterior_present": True,
        "operator_confirmed": True,
        "reason": "OPERATOR_CONFIRMED_BACK",
        "qualified_people": 1,
        "confirmation_frame_index": 30,
        "back_confirmation": {"confirmation_preview_frame_index": 31},
    }
    miss = {
        "person_present": False,
        "posterior_present": False,
        "operator_confirmed": False,
        "confirmation_active": True,
        "reason": "NO_PERSON",
        "qualified_people": 0,
    }
    decisions = iter([confirmed, miss, confirmed])
    gate = SimpleNamespace(assess=lambda frame: {
        "frame_index": frame.index,
        **next(decisions),
    })
    processor = TrackerProcessor(Tracker(), lambda value: value, presence_gate=gate)

    processor(_camera_frame(index=32))
    hidden = processor(_camera_frame(index=33))
    processor(_camera_frame(index=34))

    assert hidden["points"] == []
    assert events == [
        ("reset", "BACK_CONFIRMATION_ACTIVATED"),
        ("process",),
        ("process",),
    ]


def test_tracker_processor_never_publishes_when_operator_confirmation_is_false():
    class Tracker:
        def __init__(self):
            self.reset_reasons = []

        def reset(self, reason):
            self.reset_reasons.append(reason)

        def process(self, *_args, **_kwargs):
            return SimpleNamespace(
                back_present=True,
                initialized=True,
                detector_confidence=0.9,
                reason="TRACKING",
                points=[],
            )

    tracker = Tracker()
    gate = SimpleNamespace(assess=lambda frame: {
        "frame_index": frame.index,
        "person_present": True,
        "posterior_present": True,
        "operator_confirmed": False,
        "reason": "BACK_CONFIRMATION_REQUIRED",
        "qualified_people": 1,
    })
    result = TrackerProcessor(
        tracker, lambda value: value, presence_gate=gate,
    )(_camera_frame(index=40))

    assert result["back_present"] is False
    assert result["initialized"] is False
    assert result["points"] == []
    assert result["landmark_probe"]["executed"] is True
    assert tracker.reset_reasons == []


def test_tracker_processor_resets_internal_track_only_after_consecutive_gate_misses():
    class Tracker:
        def __init__(self):
            self.reset_reasons = []
            self.calls = 0

        def reset(self, reason):
            self.reset_reasons.append(reason)

        def process(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                back_present=True,
                initialized=True,
                detector_confidence=0.9,
                reason="TRACKING",
                points=[],
            )

    decisions = iter([False, False, True, False, False, False, False])

    def assess(frame):
        accepted = next(decisions)
        return {
            "frame_index": frame.index,
            "person_present": accepted,
            "posterior_present": accepted,
            "operator_confirmed": accepted,
            "reason": "OPERATOR_CONFIRMED_BACK" if accepted else "NO_POSTERIOR_PERSON",
            "qualified_people": 1 if accepted else 0,
            "confirmation_frame_index": 2 if accepted else None,
            "back_confirmation": (
                {"confirmation_preview_frame_index": 2} if accepted else {}
            ),
        }

    gate = SimpleNamespace(assess=assess)
    tracker = Tracker()
    processor = TrackerProcessor(
        tracker,
        lambda value: value,
        presence_gate=gate,
        presence_reset_after_misses=3,
    )

    first = processor(_camera_frame(index=1))
    second = processor(_camera_frame(index=2))
    accepted = processor(_camera_frame(index=3))
    third = processor(_camera_frame(index=4))
    fourth = processor(_camera_frame(index=5))
    fifth = processor(_camera_frame(index=6))
    sixth = processor(_camera_frame(index=7))

    assert first["points"] == second["points"] == []
    assert first["back_present"] is second["back_present"] is False
    assert accepted["back_present"] is True
    assert third["points"] == fourth["points"] == fifth["points"] == sixth["points"] == []
    assert tracker.calls == 1
    assert tracker.reset_reasons == ["BACK_CONFIRMATION_ACTIVATED", "NO_BACK"]


def test_tracker_processor_runs_landmark_tracker_only_after_same_frame_gate_accepts():
    output = SimpleNamespace(
        back_present=True,
        initialized=False,
        detector_confidence=0.8,
        reason="INITIALIZING",
        points=[],
    )

    class Tracker:
        def __init__(self):
            self.calls = 0
            self.reset_reasons = []

        def reset(self, reason):
            self.reset_reasons.append(reason)

        def process(self, *_args, **_kwargs):
            self.calls += 1
            return output

    tracker = Tracker()
    gate = SimpleNamespace(assess=lambda frame: {
        "frame_index": frame.index,
        "person_present": True,
        "posterior_present": True,
        "operator_confirmed": True,
        "reason": "OPERATOR_CONFIRMED_BACK",
        "qualified_people": 1,
        "confirmation_frame_index": 16,
        "back_confirmation": {"confirmation_preview_frame_index": 16},
    })
    processor = TrackerProcessor(tracker, lambda value: value, presence_gate=gate)

    result = processor(_camera_frame(index=17))

    assert tracker.calls == 1
    assert tracker.reset_reasons == ["BACK_CONFIRMATION_ACTIVATED"]
    assert result["posterior_gate"]["frame_index"] == 17
    assert result["posterior_gate"]["posterior_present"] is True


def test_tracker_processor_passes_same_frame_robot_mask_bundle_to_tracker():
    calls = []

    class Tracker:
        def process(self, *args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(
                back_present=True,
                initialized=True,
                detector_confidence=0.9,
                reason="TRACKING",
                points=[],
            )

    camera = _camera_frame(index=31)
    mask = SimpleNamespace(
        mask=object(),
        frame_index=31,
        mask_sha256="a" * 64,
        geometry_release_id="robot_geometry-" + "1" * 20,
        tool_release_id="tool_geometry-" + "2" * 20,
        handeye_release_id="handeye_activation-" + "b" * 20,
        handeye_sha256="b" * 64,
        handeye_capability_schema=(
            "wam_provisional_robot_mask_handeye_capability_v1"
        ),
        handeye_calibration_tier="USER_ACCEPTED_BELOW_GATE_5MM",
        handeye_policy_sha256="c" * 64,
        handeye_calibration_status="USER_ACCEPTED_BELOW_GATE",
        handeye_provisional=True,
        handeye_quality_gate_passed=False,
        handeye_workspace_1sigma_max_m=0.00605,
        handeye_covariance_bound_transform=(
            "transform_follower_base_from_camera"
        ),
        handeye_covariance_scope="random_only",
        handeye_systematic_bias_status="NOT_CHARACTERIZED",
        handeye_systematic_bias_bound_m=None,
        handeye_autonomous_targeting_authorized=False,
        handeye_provisional_policy_allowance_m=0.0065,
    )
    bundle = SimpleNamespace(
        camera_frame=camera,
        robot_mask_packet=mask,
        bundle_id="frame_evidence-" + "4" * 20,
    )
    processor = TrackerProcessor(
        Tracker(),
        lambda value: value,
        # Legacy/offline callers may still consume an optional robot mask. A
        # production-required mask now also requires the independent pose and
        # occlusion evidence chain exercised below.
        require_robot_mask=False,
    )

    result = processor(bundle)

    assert calls[0][1]["robot_mask"] is mask.mask
    assert result["frame_evidence_bundle_id"] == bundle.bundle_id
    assert result["robot_mask"]["mask_sha256"] == "a" * 64
    assert result["robot_mask"]["handeye_sha256"] == "b" * 64
    assert result["robot_mask"]["handeye_capability_schema"] == (
        "wam_provisional_robot_mask_handeye_capability_v1"
    )
    assert result["robot_mask"]["handeye_calibration_tier"] == (
        "USER_ACCEPTED_BELOW_GATE_5MM"
    )
    assert result["robot_mask"]["handeye_policy_sha256"] == "c" * 64
    assert result["robot_mask"]["handeye_provisional"] is True
    assert result["robot_mask"]["handeye_quality_gate_passed"] is False
    assert result["robot_mask"]["handeye_autonomous_targeting_authorized"] is False


def test_tracker_processor_fails_closed_when_required_robot_mask_is_missing():
    class Tracker:
        def process(self, *_args, **_kwargs):
            raise AssertionError("tracker must not run without required robot evidence")

    with pytest.raises(ValueError, match="production occlusion evidence chain"):
        TrackerProcessor(
            Tracker(),
            lambda value: value,
            require_robot_mask=True,
        )


def _production_bundle():
    return _ROBOT_BUNDLE_HELPERS._bundle()


def _default_gate_decision(bundle):
    from ausc.perception.tracking.posterior_back_mask_provider import (
        OperatorConfirmedBackGate,
    )

    height, width = bundle.camera_frame.frame.shape[:2]
    return OperatorConfirmedBackGate.for_frame(
        camera_frame=bundle.camera_frame,
        posterior_present=True,
        operator_confirmed=True,
        person_box_xyxy=(0, 0, width, height),
    )


def _default_presence_gate():
    return SimpleNamespace(assess=lambda frame: {
        "frame_index": frame.index,
        "posterior_present": True,
        "operator_confirmed": True,
        "box": (
            0.0,
            0.0,
            float(frame.frame.shape[1]),
            float(frame.frame.shape[0]),
        ),
    })


def _back_packet(bundle, *, mask=None, operator_gate_decision_id=None):
    from ausc.perception.tracking.back_mask_packet import BackMaskPacket

    if mask is None:
        mask = np.ones(bundle.camera_frame.frame.shape[:2], dtype=np.bool_)
    if operator_gate_decision_id is None:
        operator_gate_decision_id = _default_gate_decision(bundle).decision_id
    return BackMaskPacket.build(
        camera_frame=bundle.camera_frame,
        mask=mask,
        back_mask_release_id="back_mask-" + "9" * 20,
        back_mask_model_sha256="8" * 64,
        operator_gate_decision_id=operator_gate_decision_id,
    )


class _BackMaskProvider:
    def __init__(self, packet):
        self.packet = packet
        self.frames = []

    def build(self, *, camera_frame, gate):
        self.frames.append(camera_frame)
        if self.packet.operator_gate_decision_id != gate.decision_id:
            raise ValueError("test packet gate mismatch")
        return self.packet


class _PatientPoseProvider:
    def __init__(self, packet):
        self.packet = packet
        self.calls = []
        self.reset_calls = 0

    def process_evidence(self, evidence, episode_id):
        self.calls.append((evidence, episode_id))
        return self.packet

    def reset(self):
        self.reset_calls += 1


class _OcclusionRuntime:
    point_names = ("R1",)

    def __init__(self, safe_frame):
        self.safe_frame = safe_frame
        self.calls = []
        self.reset_calls = 0

    def advance(self, observation, *, robot_mask_packet, patient_pose_packet):
        self.calls.append((observation, robot_mask_packet, patient_pose_packet))
        return self.safe_frame

    def reset(self):
        self.reset_calls += 1


def _landmark_runtime_frame(*, frame_index=12, robot_occluded=True):
    return SimpleNamespace(
        frame_index=frame_index,
        back_present=True,
        initialized=True,
        detector_confidence=0.91,
        reason="TRACKING",
        points=(SimpleNamespace(
            name="R1",
            pixel=(2.0, 1.0),
            xyz_camera=(0.01, 0.02, 0.9),
            xyz_patient=(0.01, 0.02, 0.9),
            status="TRACKING",
            measured=not robot_occluded,
            position_std=0.002,
            measurement_residual_m=0.001,
            robot_occluded=robot_occluded,
            direct_geometry_accepted=not robot_occluded,
        ),),
    )


def _safe_frame(*, source="independent_patient_pose", robot_occluded=True):
    from ausc.perception.tracking.occlusion_runtime import (
        PATIENT_FRAME_LOCKED,
        SafePoint,
        SafeTrackingFrame,
    )

    return SafeTrackingFrame(
        frame_index=12,
        camera_host_t_s=1.05,
        back_present=True,
        initialized=True,
        points=(SafePoint(
            name="R1",
            occlusion_state=(PATIENT_FRAME_LOCKED if robot_occluded else "DIRECT"),
            valid=True,
            measured=not robot_occluded,
            xyz_camera_m=np.array((0.02, 0.03, 0.91)),
            pixel=np.array((2.0, 1.0)),
            uncertainty_bound_m=0.006,
            robot_occluded=robot_occluded,
            source=source,
            reason="safe-test-point",
        ),),
        reason="TRACKING",
    )


def _production_processor(
    bundle,
    *,
    runtime=None,
    landmark_frame=None,
    patient_pose_packet=None,
    episode_id="patient-session-1",
    back_provider=None,
    presence_gate=None,
):
    patient_pose_packet = patient_pose_packet or SimpleNamespace(
        frame_index=bundle.index,
        camera_host_t_s=bundle.t,
        valid=True,
        reason="TRACKING",
        episode_id="patient-session-1/pose-epoch-test",
    )
    back_provider = back_provider or _BackMaskProvider(_back_packet(bundle))
    presence_gate = presence_gate or _default_presence_gate()
    patient_provider = _PatientPoseProvider(patient_pose_packet)
    occlusion_runtime = runtime or _OcclusionRuntime(_safe_frame())

    class Tracker:
        def __init__(self):
            self.calls = []
            self.reset_calls = []

        def process(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return SimpleNamespace(raw=True)

        def reset(self, reason="NO_BACK"):
            self.reset_calls.append(reason)

    tracker = Tracker()
    processor = TrackerProcessor(
        tracker,
        lambda _raw: landmark_frame or _landmark_runtime_frame(),
        presence_gate=presence_gate,
        require_robot_mask=True,
        back_mask_provider=back_provider,
        patient_pose_provider=patient_provider,
        occlusion_runtime=occlusion_runtime,
        episode_id=episode_id,
    )
    return processor, tracker, back_provider, patient_provider, occlusion_runtime


def test_production_tracker_binds_one_immutable_frame_through_pose_and_occlusion():
    from ausc.perception.tracking.patient_frame_evidence import PatientFrameEvidence

    bundle = _production_bundle()
    (
        processor,
        tracker,
        back_provider,
        patient_provider,
        occlusion_runtime,
    ) = _production_processor(bundle)

    result = processor(bundle)

    assert back_provider.frames == [bundle.camera_frame]
    assert tracker.calls[0][0][0] is bundle.camera_frame.frame
    assert tracker.calls[0][0][1] is bundle.camera_frame.depth
    assert tracker.calls[0][1]["robot_mask"] is bundle.robot_mask_packet.mask
    evidence, episode_id = patient_provider.calls[0]
    assert isinstance(evidence, PatientFrameEvidence)
    assert evidence.frame_bundle is bundle
    assert episode_id == "patient-session-1"
    observation, robot_packet, patient_packet = occlusion_runtime.calls[0]
    assert observation.frame_index == bundle.index
    assert observation.camera_id == bundle.camera_frame.camera_id
    assert robot_packet is bundle.robot_mask_packet
    assert patient_packet is patient_provider.packet
    assert result["frame_evidence_bundle_id"] == bundle.bundle_id
    assert result["patient_frame_evidence_id"] == evidence.evidence_id
    assert result["back_mask_packet_id"] == evidence.back_mask_packet.packet_id
    assert result["back_mask"]["depth_sha256"] == bundle.camera_frame.depth_sha256
    assert (
        result["back_mask"]["operator_gate_decision_id"]
        == result["operator_back_gate_id"]
    )
    assert result["points"][0]["source"] == "independent_patient_pose"
    assert result["points"][0]["robot_occluded"] is True


def test_production_tracker_hides_points_when_back_mask_is_for_another_frame():
    bundle = _production_bundle()
    processor, tracker, back_provider, patient_provider, runtime = (
        _production_processor(bundle)
    )
    forged = object.__new__(type(back_provider.packet))
    for name in back_provider.packet.__dataclass_fields__:
        object.__setattr__(
            forged,
            name,
            bundle.index + 1
            if name == "frame_index"
            else getattr(back_provider.packet, name),
        )
    back_provider.packet = forged

    result = processor(bundle)

    assert result["reason"] == "BACK_MASK_FRAME_MISMATCH"
    assert result["points"] == []
    assert tracker.calls == []
    assert patient_provider.calls == []
    assert runtime.calls == []


def test_production_tracker_requires_exact_frame_bundle_not_camera_only():
    bundle = _production_bundle()
    processor, tracker, _back, patient_provider, runtime = _production_processor(bundle)

    result = processor(bundle.camera_frame)

    assert result["reason"] == "FRAME_EVIDENCE_BUNDLE_REQUIRED"
    assert result["points"] == []
    assert tracker.calls == []
    assert patient_provider.calls == []
    assert runtime.calls == []


def test_production_tracker_rejects_occluded_point_not_maintained_by_patient_pose():
    bundle = _production_bundle()
    unsafe_runtime = _OcclusionRuntime(
        _safe_frame(source="last_valid_bounded_freeze", robot_occluded=True)
    )
    processor, tracker, _back, _pose, _runtime = _production_processor(
        bundle, runtime=unsafe_runtime,
    )

    result = processor(bundle)

    assert len(tracker.calls) == 1
    assert result["reason"] == "UNSAFE_OCCLUDED_POINT_SOURCE"
    assert result["points"] == []


def test_production_tracker_resets_all_state_when_episode_id_changes():
    bundle = _production_bundle()
    episode = {"value": "patient-session-1"}
    processor, tracker, _back, patient_provider, runtime = _production_processor(
        bundle, episode_id=lambda: episode["value"],
    )
    assert processor(bundle)["reason"] == "TRACKING"
    episode["value"] = "patient-session-2"

    result = processor(bundle)

    assert result["reason"] == "TRACKING"
    assert patient_provider.reset_calls == 1
    assert runtime.reset_calls == 1
    assert tracker.reset_calls == ["EPISODE_CHANGED"]
    assert patient_provider.calls[-1][1] == "patient-session-2"


@pytest.mark.parametrize(
    ("safe_frame", "expected_reason"),
    (
        (
            SimpleNamespace(
                frame_index="not-an-index",
                camera_host_t_s=1.05,
                back_present=True,
                initialized=True,
                reason="TRACKING",
                points=(),
            ),
            "OCCLUSION_RESULT_INVALID",
        ),
        (
            SimpleNamespace(
                frame_index=12,
                camera_host_t_s="not-a-time",
                back_present=True,
                initialized=True,
                reason="TRACKING",
                points=(),
            ),
            "OCCLUSION_RESULT_INVALID",
        ),
    ),
)
def test_production_tracker_hides_malformed_occlusion_result_with_explicit_reason(
    safe_frame, expected_reason,
):
    bundle = _production_bundle()
    processor, _tracker, _back, _pose, _runtime = _production_processor(
        bundle, runtime=_OcclusionRuntime(safe_frame),
    )

    result = processor(bundle)

    assert result["reason"] == expected_reason
    assert result["points"] == []


def test_production_tracker_hides_malformed_patient_pose_packet_with_explicit_reason():
    bundle = _production_bundle()
    processor, tracker, _back, _pose, runtime = _production_processor(
        bundle,
        patient_pose_packet=SimpleNamespace(
            frame_index="not-an-index",
            camera_host_t_s=bundle.t,
            valid=False,
            reason="INVALID",
        ),
    )

    result = processor(bundle)

    assert result["reason"] == "PATIENT_POSE_PACKET_INVALID"
    assert result["points"] == []
    assert tracker.calls == []
    assert runtime.calls == []


def test_production_tracker_hides_malformed_landmark_result_with_explicit_reason():
    bundle = _production_bundle()
    processor, _tracker, _back, _pose, runtime = _production_processor(
        bundle,
        landmark_frame=_landmark_runtime_frame(frame_index="not-an-index"),
    )

    result = processor(bundle)

    assert result["reason"] == "LANDMARK_RESULT_INVALID"
    assert result["points"] == []
    assert runtime.calls == []


def test_production_tracker_binds_operator_gate_for_independent_back_provider():
    from ausc.perception.tracking.posterior_back_mask_provider import (
        OperatorConfirmedBackGate,
    )

    bundle = _production_bundle()

    class GateAwareBackProvider:
        def __init__(self):
            self.calls = []

        def build(self, *, camera_frame, gate):
            self.calls.append((camera_frame, gate))
            return _back_packet(bundle)

    back_provider = GateAwareBackProvider()
    presence_gate = SimpleNamespace(assess=lambda frame: {
        "frame_index": frame.index,
        "posterior_present": True,
        "operator_confirmed": True,
        "box": (0.2, 0.3, 4.8, 3.9),
    })
    processor, _tracker, _back, _pose, _runtime = _production_processor(
        bundle,
        back_provider=back_provider,
        presence_gate=presence_gate,
    )

    result = processor(bundle)

    camera_frame, gate = back_provider.calls[0]
    assert camera_frame is bundle.camera_frame
    assert isinstance(gate, OperatorConfirmedBackGate)
    assert gate.person_box_xyxy == (0, 0, 5, 4)
    assert gate.frame_index == bundle.index
    assert result["operator_back_gate_id"] == gate.decision_id


def test_production_tracker_latches_failed_safety_reset_and_never_reuses_history():
    bundle = _production_bundle()

    class ResetFailingRuntime(_OcclusionRuntime):
        def reset(self):
            self.reset_calls += 1
            raise RuntimeError("cannot clear patient anchors")

    runtime = ResetFailingRuntime(_safe_frame())
    processor, tracker, _back, _pose, _runtime = _production_processor(
        bundle, runtime=runtime,
    )

    first = processor(bundle.camera_frame)
    second = processor(bundle)

    assert first["reason"] == "FRAME_EVIDENCE_BUNDLE_REQUIRED"
    assert "safety reset failed" in first["error"]
    assert second["reason"] == "SAFETY_RESET_FAILED"
    assert second["points"] == []
    assert tracker.calls == []
    assert runtime.calls == []


def test_production_unsafe_failure_carries_display_only_annex():
    """2026-08-30 soak forensics: an occlusion-gated failure record must still
    feed the operator display (amber-only), never the points channel."""
    bundle = _production_bundle()
    unsafe_runtime = _OcclusionRuntime(
        _safe_frame(source="last_valid_bounded_freeze", robot_occluded=True)
    )
    processor, tracker, _back, _pose, _runtime = _production_processor(
        bundle, runtime=unsafe_runtime,
    )

    result = processor(bundle)

    assert result["reason"] == "UNSAFE_OCCLUDED_POINT_SOURCE"
    assert result["points"] == []
    annex = result.get("display_only_points")
    assert isinstance(annex, list) and len(annex) == 1
    assert annex[0]["name"] == "R1"
    assert annex[0]["display_only"] is True
    assert annex[0]["reason"] == "UNPUBLISHED_TRACKER_POINT"
    assert annex[0]["pixel"] == [2.0, 1.0]


def test_production_occlusion_runtime_error_carries_display_only_annex():
    bundle = _production_bundle()

    class _ExplodingRuntime(_OcclusionRuntime):
        def advance(self, observation, *, robot_mask_packet, patient_pose_packet):
            raise ValueError("occlusion frame does not contain every point exactly once")

    processor, _tracker, _back, _pose, _runtime = _production_processor(
        bundle, runtime=_ExplodingRuntime(_safe_frame()),
    )
    result = processor(bundle)
    assert result["reason"] == "OCCLUSION_RUNTIME_ERROR"
    assert result["points"] == []
    annex = result.get("display_only_points")
    assert isinstance(annex, list) and annex and annex[0]["display_only"] is True


def test_production_pre_tracker_failure_has_no_display_annex():
    """Before an authorized back exists, nothing may be drawn at all: any
    failure BEFORE the tracker ran must carry no display annex."""
    bundle = _production_bundle()
    back_packet = _back_packet(bundle)
    invalid = SimpleNamespace(**{**vars(back_packet), "status": "NO_BACK"})
    processor, tracker, _back, _pose, _runtime = _production_processor(
        bundle, back_provider=_BackMaskProvider(invalid),
    )
    result = processor(bundle)
    assert result["reason"] in ("NO_BACK", "BACK_MASK_PACKET_INVALID")
    assert result["points"] == []
    assert not result.get("display_only_points")
    assert tracker.calls == []           # tracker never ran: nothing to show


def test_production_not_ready_occlusion_frame_carries_display_only_annex():
    """The not-ready branch (safe_frame.reason rides through, e.g.
    INCOMPLETE_OCCLUSION_SAFE_CONSTELLATION) was the second preview-starving
    exit found by the soak rig; it must carry the same amber-only annex."""
    from dataclasses import replace as dc_replace

    bundle = _production_bundle()
    not_ready = dc_replace(
        _safe_frame(), reason="INCOMPLETE_OCCLUSION_SAFE_CONSTELLATION",
    )
    processor, _tracker, _back, _pose, _runtime = _production_processor(
        bundle, runtime=_OcclusionRuntime(not_ready),
    )
    result = processor(bundle)
    assert result["reason"] == "INCOMPLETE_OCCLUSION_SAFE_CONSTELLATION"
    assert result["points"] == []
    annex = result.get("display_only_points")
    assert isinstance(annex, list) and len(annex) == 1
    assert annex[0]["name"] == "R1"
    assert annex[0]["display_only"] is True
