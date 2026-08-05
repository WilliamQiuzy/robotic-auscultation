"""Hardware-free contracts for the collection web control plane."""

from __future__ import annotations

import json
import sys
import threading
import time
from itertools import count
from pathlib import Path
from uuid import UUID

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from collection_platform import (  # noqa: E402
    DIAGNOSTIC_PERCEPTION_CONTRACT,
    PREVIEW_ONLY_CONTRACT,
    PRODUCTION_TELEOP_CONTRACT,
    CaptureContract,
    CaptureMode,
    CollectionPlatform,
    CollectionState,
    create_session_directory,
    make_session_id,
    reconcile_orphan_diagnostic_sessions,
)
from perception_capture import DiagnosticCaptureCancelled  # noqa: E402


class _Capture:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.started = 0
        self.finalized = 0
        self.aborted: list[str] = []
        self.events: list[dict] = []
        self.frames: list[tuple[object, dict]] = []

    def start(self) -> None:
        self.started += 1

    def record_event(self, payload: dict) -> None:
        self.events.append(dict(payload))

    def write_frame(self, frame, tracker_output: dict) -> None:  # noqa: ANN001
        self.frames.append((frame, dict(tracker_output)))

    def finalize(self) -> dict:
        self.finalized += 1
        return {"status": "diagnostic_complete", "training_eligible": False}

    def abort(self, reason: str) -> None:
        self.aborted.append(reason)

    def request_cancel(self) -> None:
        # Cooperative cancellation is now a mandatory part of the _Capture
        # protocol; the base fake supports it as a no-op unless a subclass wires it.
        pass


def _platform(tmp_path: Path, *, blockers=()) -> tuple[CollectionPlatform, list[_Capture]]:
    captures: list[_Capture] = []

    def factory(session_dir: Path) -> _Capture:
        capture = _Capture(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1", "R2", "R3"],
        capture_factory=factory,
        writer_types={DIAGNOSTIC_PERCEPTION_CONTRACT.writer_identity: _Capture},
        diagnostic_blockers=blockers,
        production_blockers=("robot-base targets missing",),
        wall_time_ns=lambda: 1_721_234_567_000_000_123,
        uuid_factory=lambda: UUID("12345678-1234-5678-1234-567812345678"),
    )
    return platform, captures


def _prod_platform(tmp_path: Path, capture_cls):
    captures = []

    def factory(session_dir: Path):
        capture = capture_cls(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        contract=PRODUCTION_TELEOP_CONTRACT,
        writer_types={PRODUCTION_TELEOP_CONTRACT.writer_identity: capture_cls},
    )
    return platform, captures


def test_production_capture_finalizes_as_raw_never_self_promoting(tmp_path):
    # #6 aligned: a production writer produces a RAW session (diagnostic_complete +
    # training_eligible=False). The writer NEVER self-promotes; the trusted finalizer is
    # the sole promoter. The platform completes it; capture_kind marks it promotable.
    class _RawProd(_Capture):
        def finalize(self):
            self.finalized += 1
            return {"status": "diagnostic_complete", "training_eligible": False}

    platform, _c = _prod_platform(tmp_path, _RawProd)
    assert platform.control("start")["ok"] is True
    assert platform.control("end")["state"] == "FINALIZING"
    terminal = platform.wait_for_terminal(timeout=2.0)
    assert terminal["state"] == "COMPLETE"
    assert terminal["training_eligible"] is False   # raw -- finalizer promotes later
    assert platform.state_snapshot()["capture_kind"] == "production_teleop_v4"


def test_writer_cannot_self_promote(tmp_path):
    # #6 aligned: the writer must NOT self-promote. A finalize claiming
    # training_eligible=True is rejected -- promotion is the finalizer's exclusive job.
    class _SelfPromoting(_Capture):
        def finalize(self):
            self.finalized += 1
            return {"status": "diagnostic_complete", "training_eligible": True}

    platform, _c = _prod_platform(tmp_path, _SelfPromoting)
    assert platform.control("start")["ok"] is True
    platform.control("end")
    terminal = platform.wait_for_terminal(timeout=2.0)
    assert terminal["state"] == "ERROR"
    assert "self-promote" in terminal["error"]


def test_reconciler_preserves_a_sealed_production_session(tmp_path):
    # #6 aligned + recovery: a production raw session uses the diagnostic-shape seal
    # (DIAGNOSTIC_FINALIZATION.json), so the restart reconciler recognises it as sealed
    # and NEVER rewrites it to diagnostic_aborted. (The old PRODUCTION_FINALIZATION.json
    # went unrecognised and corrupted the sealed session.)
    from collection_platform import reconcile_orphan_diagnostic_sessions

    root = tmp_path / "sessions"
    session = root / "diagnostic_perception_20260804T101112000000000Z_abcdef123456"
    session.mkdir(parents=True)
    manifest = {
        "schema_version": "wam_session_v2",
        "capture_kind": "production_teleop_v4",
        "promotion_contract_version": "wam_promotion_v2",
        "session_id": session.name,
        "rehearsal_overrides": [],
        "training_eligible": False,
        "status": "diagnostic_complete",
    }
    (session / "manifest.json").write_text(json.dumps(manifest))
    (session / "DIAGNOSTIC_FINALIZATION.json").write_text(json.dumps({"sealed": True}))

    recovered = reconcile_orphan_diagnostic_sessions(root)
    assert recovered == []                                   # sealed -> untouched
    after = json.loads((session / "manifest.json").read_text())
    assert after["status"] == "diagnostic_complete"          # NOT diagnostic_aborted
    assert after["capture_kind"] == "production_teleop_v4"
    assert after["training_eligible"] is False


def test_a_non_cancellable_writer_is_refused_before_recording(tmp_path):
    # A writer without request_cancel could race Abort against its own seal;
    # the platform must refuse it at Start (fail-closed), before any frame.
    class _NoCancelCapture:
        def __init__(self, session_dir):
            self.session_dir = session_dir

        def start(self): pass
        def record_event(self, payload): pass
        def write_frame(self, frame, tracker_output): pass
        def finalize(self): return {"status": "diagnostic_complete", "training_eligible": False}
        def abort(self, reason): pass
        # deliberately no request_cancel

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=lambda session_dir: _NoCancelCapture(session_dir),
        writer_types={DIAGNOSTIC_PERCEPTION_CONTRACT.writer_identity: _NoCancelCapture},
    )
    result = platform.control("start")
    assert result["ok"] is False
    assert platform.state is not CollectionState.RECORDING


def test_production_teleop_contract_is_the_only_promotable_recording_contract():
    # Production is the sole training-eligible, recording contract; diagnostic and
    # preview are structurally un-promotable.
    assert PRODUCTION_TELEOP_CONTRACT.mode is CaptureMode.PRODUCTION_CAPTURE
    assert PRODUCTION_TELEOP_CONTRACT.capture_kind == "production_teleop_v4"
    assert PRODUCTION_TELEOP_CONTRACT.training_intent is True
    assert PRODUCTION_TELEOP_CONTRACT.training_eligible is True
    assert PRODUCTION_TELEOP_CONTRACT.records is True
    assert PRODUCTION_TELEOP_CONTRACT.writer_identity == "ProductionTeleopCapture"
    assert DIAGNOSTIC_PERCEPTION_CONTRACT.training_eligible is False
    assert PREVIEW_ONLY_CONTRACT.training_eligible is False


def test_production_contract_cannot_be_fronted_by_a_diagnostic_writer():
    with pytest.raises(ValueError, match="diagnostic writer"):
        CaptureContract(
            mode=CaptureMode.PRODUCTION_CAPTURE,
            capture_kind="production_teleop_v4",
            training_intent=True,
            writer_identity="DiagnosticPerceptionCapture",
            manifest_schema="wam_session_v2",
            finalization_schema="wam_promotion_v2",
        )


@pytest.mark.parametrize(
    "capture_kind",
    ("production_future_v99", "production_teleop_v4_extra", "production_"),
)
def test_production_contract_accepts_only_the_exact_promotable_kind(capture_kind):
    with pytest.raises(ValueError, match="production_teleop_v4"):
        CaptureContract(
            mode=CaptureMode.PRODUCTION_CAPTURE,
            capture_kind=capture_kind,
            training_intent=True,
            writer_identity="ProductionTeleopCapture",
            manifest_schema="wam_session_v2",
            finalization_schema="wam_promotion_v2",
        )


def test_session_directory_is_contained_and_never_reuses_an_existing_path(tmp_path):
    root = tmp_path / "sessions"
    with pytest.raises(ValueError, match="session id"):
        create_session_directory(root, "../escape")

    session = create_session_directory(root, "diagnostic_perception_safe")
    marker = session / "preserve.me"
    marker.write_text("original\n")

    with pytest.raises(FileExistsError):
        create_session_directory(root, session.name)

    assert marker.read_text() == "original\n"
    assert not (tmp_path / "escape").exists()


def test_session_ids_remain_unique_inside_the_same_utc_nanosecond():
    now_ns = 1_721_234_567_000_000_123
    first = make_session_id(
        now_ns=now_ns, uuid_value=UUID("00000000-0000-0000-0000-000000000001")
    )
    second = make_session_id(
        now_ns=now_ns, uuid_value=UUID("00000000-0000-0000-0000-000000000002")
    )

    assert first != second
    assert first.startswith("diagnostic_perception_20240717T")
    assert first.endswith("_000000000001")


def test_restart_reconciles_only_unsealed_diagnostic_recording_sessions(tmp_path):
    root = tmp_path / "sessions"
    orphan = create_session_directory(
        root, "diagnostic_perception_20260713T120000000000000Z_000000000001",
    )
    (orphan / "manifest.json").write_text(json.dumps({
        "status": "recording",
        "training_eligible": False,
        "production_finalization_permitted": False,
    }))
    (orphan / "lifecycle.json").write_text(json.dumps({
        "state": "FINALIZING", "training_eligible": False,
    }))
    untouched = create_session_directory(
        root, "diagnostic_perception_20260713T120001000000000Z_000000000002",
    )
    (untouched / "manifest.json").write_text(json.dumps({
        "status": "diagnostic_complete", "training_eligible": False,
    }))
    (untouched / "DIAGNOSTIC_FINALIZATION.json").write_text("{}")

    reconciled = reconcile_orphan_diagnostic_sessions(
        root, wall_time_ns=lambda: 1_721_234_567_000_000_123,
    )

    assert reconciled == [str(orphan)]
    recovery = json.loads((orphan / "RECOVERY.json").read_text())
    assert recovery["status"] == "diagnostic_aborted_on_restart"
    assert recovery["training_eligible"] is False
    assert json.loads((orphan / "manifest.json").read_text())["status"] == "diagnostic_aborted"
    assert json.loads((orphan / "lifecycle.json").read_text())["state"] == "ABORTED"
    assert not (orphan / "DIAGNOSTIC_FINALIZATION.json").exists()
    assert not (untouched / "RECOVERY.json").exists()


def test_restart_reconciliation_recovers_sealed_video_without_deleting_partials(tmp_path):
    root = tmp_path / "sessions"
    orphan = create_session_directory(
        root, "diagnostic_perception_20260713T120000000000000Z_000000000003",
    )
    (orphan / "manifest.json").write_text(json.dumps({
        "status": "recording",
        "training_eligible": False,
        "production_finalization_permitted": False,
    }))
    (orphan / "lifecycle.json").write_text(json.dumps({
        "state": "RECORDING", "training_eligible": False,
    }))
    segment_dir = orphan / "rgb_segments"
    segment_dir.mkdir()
    sealed = segment_dir / "segment_000000.mp4"
    partial = segment_dir / "segment_000001.inprogress.mp4"
    sealed.write_bytes(b"sealed")
    partial.write_bytes(b"partial")

    def recover(session: Path) -> dict:
        assert session == orphan
        (session / "rgb.recovered.mp4").write_bytes(b"recovered")
        return {
            "status": "recovered_video",
            "recovered_frame_count": 12,
            "source_segments": ["rgb_segments/segment_000000.mp4"],
            "output": "rgb.recovered.mp4",
        }

    reconciled = reconcile_orphan_diagnostic_sessions(
        root,
        wall_time_ns=lambda: 1_721_234_567_000_000_123,
        video_recoverer=recover,
    )

    assert reconciled == [str(orphan)]
    recovery = json.loads((orphan / "RECOVERY.json").read_text())
    assert recovery["media_recovery"]["status"] == "recovered_video"
    assert recovery["training_eligible"] is False
    assert (orphan / "rgb.recovered.mp4").read_bytes() == b"recovered"
    assert sealed.read_bytes() == b"sealed"
    assert partial.read_bytes() == b"partial"
    assert not (orphan / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_preflight_failure_is_explicit_and_start_is_rejected(tmp_path):
    platform, captures = _platform(tmp_path, blockers=("camera stale",))

    assert platform.state is CollectionState.PREFLIGHT_FAILED
    result = platform.control("start")

    assert result["ok"] is False
    assert result["code"] == "invalid_state"
    assert "camera stale" in result["message"]
    assert captures == []


def test_runtime_preflight_can_arm_only_after_camera_and_model_are_ready(tmp_path):
    platform, _ = _platform(tmp_path, blockers=("camera/model starting",))
    assert platform.state_snapshot()["state"] == "PREFLIGHT_FAILED"

    platform.set_diagnostic_blockers([])
    assert platform.state_snapshot()["state"] == "READY"
    assert platform.state_snapshot()["diagnostic_blockers"] == []

    platform.set_diagnostic_blockers(["camera stale"])
    assert platform.state_snapshot()["state"] == "PREFLIGHT_FAILED"


def test_start_enters_recording_and_resets_point_progress(tmp_path):
    platform, captures = _platform(tmp_path)
    platform.completed[:] = [0, 1]
    platform.point_index = 2
    platform.attempt = 8

    result = platform.control("start")

    assert result["ok"] is True
    assert platform.state is CollectionState.RECORDING
    assert platform.completed == []
    assert platform.point_index == 0
    assert platform.attempt == 1
    assert captures[0].started == 1
    assert Path(result["session_dir"]).parent == (tmp_path / "sessions").resolve()


def test_redo_preserves_failed_attempt_stays_on_point_and_returns_ack(tmp_path):
    platform, captures = _platform(tmp_path)
    assert platform.control("start")["ok"]
    platform.ingest_frame(
        object(), {
            "back_present": True,
            "initialized": True,
            "target_valid": True,
            "frame_index": 41,
            "points": [{"name": "R1", "status": "TRACKING"}],
        },
    )

    result = platform.control("redo")

    assert result["ok"] is True
    assert result["ack"] == "Redo saved R1 attempt 1; attempt 2 is active"
    assert platform.point_index == 0
    assert platform.attempt == 2
    assert [event["event"] for event in captures[0].events[-2:]] == [
        "attempt_closed", "attempt_started",
    ]
    assert captures[0].events[-2]["outcome"] == "redo"
    journal = [
        json.loads(line)
        for line in (platform.session_dir / "control_journal.jsonl").read_text().splitlines()
    ]
    assert any(
        row["event"] == "attempt_closed"
        and row["attempt"] == 1
        and row["outcome"] == "redo"
        for row in journal
    )


def test_confirm_requires_tracking_but_redo_only_requires_recording(tmp_path):
    platform, _ = _platform(tmp_path)

    idle = platform.control("confirm")
    assert idle["ok"] is False
    assert idle["code"] == "not_recording"
    idle_redo = platform.control("redo")
    assert idle_redo["ok"] is False
    assert idle_redo["code"] == "not_recording"

    assert platform.control("start")["ok"]
    no_back = platform.control("redo")
    assert no_back["ok"] is True
    assert no_back["ack"] == "Redo saved R1 attempt 1; attempt 2 is active"
    assert platform.attempt == 2


def test_redo_reopens_a_previously_confirmed_point_until_it_is_confirmed_again(tmp_path):
    platform, _ = _platform(tmp_path)
    assert platform.control("start")["ok"]
    platform.completed[:] = [0, 2]
    platform.point_index = 0
    platform.attempt = 4

    result = platform.control("redo")

    assert result["ok"] is True
    assert platform.completed == [2]
    assert platform.point_index == 0
    assert platform.attempt == 5
    lifecycle = json.loads((platform.session_dir / "lifecycle.json").read_text())
    assert lifecycle["completed"] == [2]


def test_active_point_must_be_present_not_merely_some_other_landmark(tmp_path):
    platform, _ = _platform(tmp_path)
    assert platform.control("start")["ok"]
    platform.ingest_frame(object(), {
        "back_present": True,
        "initialized": True,
        "points": [{"name": "R2", "status": "TRACKING"}],
    })
    result = platform.control("confirm")
    assert result["ok"] is False
    assert result["code"] == "back_not_tracked"


def test_explicit_target_valid_true_cannot_override_missing_active_landmark(tmp_path):
    platform, _ = _platform(tmp_path)
    assert platform.control("start")["ok"]
    platform.ingest_frame(object(), {
        "back_present": True,
        "initialized": True,
        "target_valid": True,
        "points": [],
    })

    assert platform.target_valid is False
    assert platform.control("confirm")["code"] == "back_not_tracked"


def test_unknown_action_is_not_reported_as_success(tmp_path):
    platform, _ = _platform(tmp_path)

    result = platform.control("delete_everything")

    assert result == {
        "ok": False,
        "code": "invalid_action",
        "message": "unknown control action: delete_everything",
        "state": "READY",
    }


def test_end_is_durable_and_idempotent_after_completion(tmp_path):
    platform, captures = _platform(tmp_path)
    platform.control("start")

    first = platform.control("end")
    terminal = platform.wait_for_terminal(timeout=2.0)
    second = platform.control("end")

    assert first["ok"] is True and first["state"] == "FINALIZING"
    assert first["accepted"] is True
    assert terminal["state"] == "COMPLETE"
    assert second["ok"] is True and second["idempotent"] is True
    assert captures[0].finalized == 1
    closed = [event for event in captures[0].events if event["event"] == "attempt_closed"]
    assert closed[-1]["outcome"] == "ended_unconfirmed"
    lifecycle = json.loads((platform.session_dir / "lifecycle.json").read_text())
    assert lifecycle["state"] == "COMPLETE"
    assert lifecycle["training_eligible"] is False
    assert lifecycle["capture_mode"] == "diagnostic_perception"
    assert lifecycle["finalization_active"] is False


def test_ingest_uses_the_external_camera_frame_and_never_opens_a_camera(tmp_path):
    platform, captures = _platform(tmp_path)
    platform.control("start")
    frame = object()
    tracker = {
        "back_present": True,
        "initialized": True,
        "target_valid": True,
        "frame_index": 9,
        "points": [{"name": "R1", "status": "TRACKING"}],
    }

    platform.ingest_frame(frame, tracker)

    assert captures[0].frames == [(frame, tracker)]
    assert platform.back_present is True
    assert platform.target_valid is True


def test_a_completed_capture_can_start_a_new_unique_session_without_reusing_progress(tmp_path):
    captures: list[_Capture] = []
    times = count(1_721_234_567_000_000_123)
    uuids = iter((
        UUID("00000000-0000-0000-0000-000000000001"),
        UUID("00000000-0000-0000-0000-000000000002"),
    ))

    def factory(session_dir: Path) -> _Capture:
        capture = _Capture(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1", "R2"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
        wall_time_ns=lambda: next(times),
        uuid_factory=lambda: next(uuids),
    )
    first_start = platform.control("start")
    platform.completed.append(0)
    platform.control("end")
    assert platform.wait_for_terminal(timeout=2.0)["state"] == "COMPLETE"

    second_start = platform.control("start")

    assert second_start["ok"] is True
    assert second_start["session_dir"] != first_start["session_dir"]
    assert Path(first_start["session_dir"]).is_dir()
    assert Path(second_start["session_dir"]).is_dir()
    assert platform.completed == []
    assert len(captures) == 2


def test_completed_capture_cannot_restart_after_live_preflight_becomes_blocked(tmp_path):
    platform, captures = _platform(tmp_path)
    assert platform.control("start")["ok"]
    assert platform.control("end")["ok"]
    assert platform.wait_for_terminal(timeout=2.0)["state"] == "COMPLETE"

    platform.set_diagnostic_blockers(["perception model output is not current"])
    result = platform.control("start")

    assert result["ok"] is False
    assert result["code"] == "invalid_state"
    assert "not current" in result["message"]
    assert len(captures) == 1


def test_partial_capture_start_failure_is_aborted_and_never_reports_recording(tmp_path):
    captures: list[_Capture] = []

    class FailingCapture(_Capture):
        def start(self) -> None:
            super().start()
            raise RuntimeError("encoder unavailable")

    def factory(session_dir: Path) -> FailingCapture:
        capture = FailingCapture(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
    )

    result = platform.control("start")

    assert result["ok"] is False
    assert result["state"] == "ERROR"
    assert result["code"] == "action_failed"
    assert captures[0].aborted and "encoder unavailable" in captures[0].aborted[0]
    lifecycle = json.loads((platform.session_dir / "lifecycle.json").read_text())
    assert lifecycle["state"] == "ERROR"
    assert lifecycle["training_eligible"] is False
    failure = json.loads((platform.session_dir / "START_FAILURE.json").read_text())
    assert failure["training_eligible"] is False
    assert "encoder unavailable" in failure["error"]
    assert platform._journal is None


def test_capture_factory_failure_keeps_fail_closed_start_evidence_and_closes_journal(tmp_path):
    def factory(_session_dir: Path):
        raise RuntimeError("capture factory exploded")

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
    )

    result = platform.control("start")

    assert result["ok"] is False
    assert result["state"] == "ERROR"
    assert platform.session_dir is not None
    failure = json.loads((platform.session_dir / "START_FAILURE.json").read_text())
    assert failure["status"] == "diagnostic_start_failed"
    assert failure["training_eligible"] is False
    assert "capture factory exploded" in failure["error"]
    lifecycle = json.loads((platform.session_dir / "lifecycle.json").read_text())
    assert lifecycle["state"] == "ERROR"
    assert lifecycle["training_eligible"] is False
    journal = [
        json.loads(line)
        for line in (platform.session_dir / "control_journal.jsonl").read_text().splitlines()
    ]
    assert any(row["event"] == "start_failed" for row in journal)
    assert platform._journal is None


def test_terminal_start_error_can_retry_in_a_new_session_without_losing_evidence(tmp_path):
    calls = 0
    captures = []
    uuids = iter((
        UUID("00000000-0000-0000-0000-000000000101"),
        UUID("00000000-0000-0000-0000-000000000102"),
    ))

    def factory(session_dir: Path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient encoder failure")
        capture = _Capture(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
        uuid_factory=lambda: next(uuids),
    )
    first = platform.control("start")
    first_dir = platform.session_dir
    assert first["state"] == "ERROR"
    assert (first_dir / "START_FAILURE.json").is_file()
    assert platform._journal is None

    second = platform.control("start")

    assert second["ok"] is True
    assert platform.state is CollectionState.RECORDING
    assert platform.session_dir != first_dir
    assert (first_dir / "START_FAILURE.json").is_file()
    assert len(captures) == 1 and captures[0].started == 1
    platform.abort("test cleanup")


def test_end_returns_immediately_while_one_slow_finalizer_exposes_finalizing(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    captures: list[_Capture] = []

    class SlowCapture(_Capture):
        def finalize(self) -> dict:
            self.finalized += 1
            entered.set()
            if not release.wait(timeout=2.0):
                raise RuntimeError("test finalizer gate timed out")
            return {"status": "diagnostic_complete", "training_eligible": False}

    def factory(session_dir: Path) -> SlowCapture:
        capture = SlowCapture(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
    )
    assert platform.control("start")["ok"]

    before = time.monotonic()
    first = platform.control("end")
    elapsed = time.monotonic() - before

    assert elapsed < 0.1
    assert first["state"] == "FINALIZING" and first["accepted"] is True
    assert entered.wait(timeout=0.5)
    assert platform._finalize_thread.daemon is False
    assert platform.state_snapshot()["state"] == "FINALIZING"
    repeated = platform.control("stop")
    assert repeated["ok"] is True
    assert repeated["state"] == "FINALIZING"
    assert repeated["idempotent"] is True
    assert captures[0].finalized == 1

    release.set()
    terminal = platform.wait_for_terminal(timeout=2.0)
    assert terminal["state"] == "COMPLETE"
    assert captures[0].finalized == 1


def test_background_finalization_failure_transitions_to_error_and_end_stays_idempotent(tmp_path):
    captures: list[_Capture] = []

    class FailingFinalizer(_Capture):
        def finalize(self) -> dict:
            self.finalized += 1
            raise RuntimeError("decode verification failed")

    def factory(session_dir: Path) -> FailingFinalizer:
        capture = FailingFinalizer(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
    )
    platform.control("start")

    accepted = platform.control("end")
    terminal = platform.wait_for_terminal(timeout=2.0)
    repeated = platform.control("end")

    assert accepted["state"] == "FINALIZING"
    assert terminal["state"] == "ERROR"
    assert "decode verification failed" in terminal["error"]
    assert repeated["ok"] is True
    assert repeated["state"] == "ERROR"
    assert repeated["idempotent"] is True
    assert captures[0].finalized == 1


def test_finalize_mapping_cannot_report_complete_without_explicit_complete_status(tmp_path):
    captures: list[_Capture] = []

    class ForgedCapture(_Capture):
        def finalize(self) -> dict:
            self.finalized += 1
            return {"training_eligible": False}

    def factory(session_dir: Path) -> ForgedCapture:
        capture = ForgedCapture(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
    )
    assert platform.control("start")["ok"] is True
    assert platform.control("end")["state"] == "FINALIZING"

    terminal = platform.wait_for_terminal(timeout=2.0)

    assert terminal["state"] == "ERROR"
    assert "status is not 'diagnostic_complete'" in terminal["error"]
    lifecycle = json.loads((platform.session_dir / "lifecycle.json").read_text())
    assert lifecycle["state"] == "ERROR"
    assert lifecycle["finalization"] is None
    journal = [
        json.loads(line)
        for line in (platform.session_dir / "control_journal.jsonl").read_text().splitlines()
    ]
    assert not any(row["event"] == "capture_completed" for row in journal)
    assert captures[0].aborted


def test_completion_validator_failure_is_a_hard_gate_before_complete(tmp_path):
    captures: list[_Capture] = []
    validations = []

    def factory(session_dir: Path) -> _Capture:
        capture = _Capture(session_dir)
        captures.append(capture)
        return capture

    def reject_tampered_seal(result: dict, session_dir: Path) -> None:
        validations.append((dict(result), Path(session_dir)))
        raise RuntimeError("diagnostic seal hash mismatch")

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
        completion_validator=reject_tampered_seal,
    )
    assert platform.control("start")["ok"] is True
    assert platform.control("end")["state"] == "FINALIZING"

    terminal = platform.wait_for_terminal(timeout=2.0)

    assert terminal["state"] == "ERROR"
    assert "diagnostic seal hash mismatch" in terminal["error"]
    assert validations == [({
        "status": "diagnostic_complete",
        "training_eligible": False,
    }, platform.session_dir)]
    lifecycle = json.loads((platform.session_dir / "lifecycle.json").read_text())
    assert lifecycle["state"] == "ERROR"
    assert lifecycle["finalization"] is None
    assert captures[0].aborted


def test_abort_is_idempotent_after_capture_self_abort_and_preserves_first_error(tmp_path):
    captures = []

    class SelfAbortingCapture(_Capture):
        def __init__(self, session_dir: Path) -> None:
            super().__init__(session_dir)
            self.self_aborted = False

        def record_event(self, payload: dict) -> None:
            if self.self_aborted:
                raise AssertionError("must not mirror events after capture self-abort")
            super().record_event(payload)

        def write_frame(self, frame, tracker_output: dict) -> None:  # noqa: ANN001
            self.self_aborted = True
            raise OSError("injected depth fsync failure")

        def abort(self, reason: str) -> None:
            self.aborted.append(reason)
            raise RuntimeError("capture is already aborted")

    def factory(session_dir: Path) -> SelfAbortingCapture:
        capture = SelfAbortingCapture(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
    )
    assert platform.control("start")["ok"] is True
    with pytest.raises(RuntimeError, match="capture is already aborted"):
        platform.ingest_frame(object(), {"frame_index": 1})
    first_error = platform.last_error

    first = platform.abort("operator sent a second reason")
    second = platform.abort("operator sent a third reason")

    assert first["ok"] is True and first["state"] == "ABORTED"
    assert first_error in first["ack"]
    assert second["ok"] is True and second["idempotent"] is True
    assert platform.last_error == first_error
    lifecycle = json.loads((platform.session_dir / "lifecycle.json").read_text())
    assert lifecycle["state"] == "ABORTED"
    assert lifecycle["error"] == first_error
    assert platform._journal is None


# --- Explicit capture modes + single Start authority (#3.1) -------------------

_PRODUCTION_CONTRACT = CaptureContract(
    mode=CaptureMode.PRODUCTION_CAPTURE,
    capture_kind="production_teleop_v4",
    training_intent=True,
    writer_identity="OpenArmTeleopSessionWriter",
    manifest_schema="wam_session_v2",
    finalization_schema="wam_promotion_v2",
)


class _ProductionWriter(_Capture):
    """A distinct writer *class* a production contract may be bound to.

    Being a different class from ``_Capture`` is the whole point: the registry
    binds by class object, so a diagnostic ``_Capture`` can never satisfy a
    production identity even if it declared the matching string.
    """


def _make_platform(
    tmp_path: Path,
    *,
    contract=None,
    diagnostic_blockers=(),
    production_blockers=(),
    capture_class=_Capture,
    writer_types=None,
) -> tuple[CollectionPlatform, list[_Capture]]:
    captures: list[_Capture] = []
    effective_contract = contract or DIAGNOSTIC_PERCEPTION_CONTRACT
    if writer_types is None:
        writer_types = {effective_contract.writer_identity: capture_class}

    def factory(session_dir: Path) -> _Capture:
        capture = capture_class(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1", "R2", "R3"],
        capture_factory=factory,
        contract=contract,
        writer_types=writer_types,
        diagnostic_blockers=diagnostic_blockers,
        production_blockers=production_blockers,
        wall_time_ns=lambda: 1_721_234_567_000_000_123,
        uuid_factory=lambda: UUID("12345678-1234-5678-1234-567812345678"),
    )
    return platform, captures


def test_capture_contract_refuses_production_mode_with_a_diagnostic_writer():
    with pytest.raises(ValueError, match="diagnostic writer"):
        CaptureContract(
            mode=CaptureMode.PRODUCTION_CAPTURE,
            capture_kind="production_teleop_v4",
            training_intent=True,
            writer_identity="DiagnosticPerceptionCapture",
            manifest_schema="wam_session_v2",
            finalization_schema="wam_promotion_v2",
        )


def test_capture_contract_refuses_inconsistent_mode_intent_and_kind():
    with pytest.raises(ValueError, match="training intent"):
        CaptureContract(
            mode=CaptureMode.DIAGNOSTIC_CAPTURE,
            capture_kind="diagnostic_perception",
            training_intent=True,
            writer_identity="DiagnosticPerceptionCapture",
            manifest_schema="wam_session_v2",
            finalization_schema="wam_promotion_v2",
        )
    with pytest.raises(ValueError, match="production capture_kind"):
        CaptureContract(
            mode=CaptureMode.PRODUCTION_CAPTURE,
            capture_kind="diagnostic_perception",
            training_intent=True,
            writer_identity="OpenArmTeleopSessionWriter",
            manifest_schema="wam_session_v2",
            finalization_schema="wam_promotion_v2",
        )
    with pytest.raises(ValueError, match="training intent"):
        CaptureContract(
            mode=CaptureMode.PREVIEW_ONLY,
            capture_kind="diagnostic_perception_preview",
            training_intent=True,
            writer_identity="DiagnosticPerceptionCapture",
            manifest_schema="wam_session_v2",
            finalization_schema="wam_promotion_v2",
        )


def test_preview_only_never_authorizes_a_recording(tmp_path):
    platform, captures = _make_platform(tmp_path, contract=PREVIEW_ONLY_CONTRACT)

    assert platform.start_authorization() == "preview-only mode never records"
    result = platform.control("start")

    assert result["ok"] is False
    assert result["code"] == "invalid_state"
    assert "preview-only" in result["message"]
    assert captures == []
    snapshot = platform.state_snapshot()
    assert snapshot["training_eligible"] is False
    assert snapshot["operational_mode"] == "PREVIEW_ONLY"


def test_diagnostic_capture_records_but_ignores_production_blockers(tmp_path):
    # Diagnostic evidence is structurally un-promotable, so a diagnostic console
    # must still Start even when production preflight is dirty.
    platform, captures = _make_platform(
        tmp_path,
        contract=DIAGNOSTIC_PERCEPTION_CONTRACT,
        production_blockers=("motor-stop identity missing",),
    )

    assert platform.start_authorization() is None
    result = platform.control("start")

    assert result["ok"] is True
    assert platform.state is CollectionState.RECORDING
    assert captures[0].started == 1
    snapshot = platform.state_snapshot()
    assert snapshot["training_eligible"] is False
    assert snapshot["capture_kind"] == "diagnostic_perception"
    assert snapshot["operational_mode"] == "DIAGNOSTIC_CAPTURE"


def test_production_capture_start_is_blocked_until_production_preflight_is_clear(
    tmp_path,
):
    platform, captures = _make_platform(
        tmp_path,
        contract=_PRODUCTION_CONTRACT,
        production_blockers=("motor-stop identity missing",),
    )

    reason = platform.start_authorization()
    assert reason is not None and "production preflight failed" in reason
    result = platform.control("start")

    assert result["ok"] is False
    assert result["code"] == "invalid_state"
    assert captures == []


def test_production_capture_with_clear_preflight_and_real_writer_records(tmp_path):
    # The live writer IS the registered production writer class, so Start is
    # authorized.  But recording is not eligibility: training_eligible stays
    # False until a validated production finalize completes.
    platform, captures = _make_platform(
        tmp_path,
        contract=_PRODUCTION_CONTRACT,
        production_blockers=(),
        capture_class=_ProductionWriter,
    )

    assert platform.start_authorization() is None
    result = platform.control("start")

    assert result["ok"] is True
    assert captures[0].started == 1
    snapshot = platform.state_snapshot()
    assert snapshot["state"] == "RECORDING"
    assert snapshot["training_eligible"] is False  # realized-only, not declared
    assert snapshot["capture_kind"] == "production_teleop_v4"
    assert snapshot["operational_mode"] == "PRODUCTION_CAPTURE"


def test_production_contract_with_a_diagnostic_writer_is_refused(tmp_path):
    # The contract declares the production writer identity and the registry
    # binds it to the real production writer *class*.  The live factory here
    # produces a diagnostic writer instead: a different class object, so the
    # class-bound check refuses it fail-closed.  A self-declared identity string
    # could not help — the check never consults one.
    platform, captures = _make_platform(
        tmp_path,
        contract=_PRODUCTION_CONTRACT,
        production_blockers=(),
        capture_class=_Capture,
        writer_types={_PRODUCTION_CONTRACT.writer_identity: _ProductionWriter},
    )

    result = platform.control("start")

    assert result["ok"] is False
    assert platform.state is not CollectionState.RECORDING
    assert captures[0].started == 0  # writer verified BEFORE the first frame
    snapshot = platform.state_snapshot()
    assert snapshot["training_eligible"] is False


def test_start_authorized_is_false_while_recording(tmp_path):
    # The reviewer's exact inconsistency: start_authorized must not stay true
    # once a recording is in progress; it is the single, state-aware authority.
    platform, _ = _make_platform(tmp_path, contract=DIAGNOSTIC_PERCEPTION_CONTRACT)

    assert platform.state_snapshot()["start_authorized"] is True
    assert platform.control("start")["ok"] is True

    snapshot = platform.state_snapshot()
    assert snapshot["state"] == "RECORDING"
    assert snapshot["start_authorized"] is False
    assert "cannot start from RECORDING" in snapshot["start_rejection"]


def test_start_is_blocked_while_a_prior_finalizer_is_running(tmp_path):
    # Reproduce the cross-session contamination window: an in-flight finalizer
    # must fully exit before any new Start, so two sessions can never overlap
    # and an old finalizer can never commit onto a newer session.
    entered = threading.Event()
    release = threading.Event()

    class SlowCapture(_Capture):
        def finalize(self) -> dict:
            self.finalized += 1
            entered.set()
            if not release.wait(timeout=2.0):
                raise RuntimeError("test finalizer gate timed out")
            return {"status": "diagnostic_complete", "training_eligible": False}

    captures: list[_Capture] = []

    def factory(session_dir: Path) -> SlowCapture:
        capture = SlowCapture(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
    )
    assert platform.control("start")["ok"]
    first_generation = platform._generation
    assert platform.control("end")["state"] == "FINALIZING"
    assert entered.wait(timeout=1.0)

    # While the first finalizer is still running, Start is refused by the single
    # authority — the snapshot says so and the control is rejected.
    snapshot = platform.state_snapshot()
    assert snapshot["start_authorized"] is False
    assert "prior finalization is still running" in snapshot["start_rejection"]
    assert platform.control("start")["ok"] is False
    assert platform._generation == first_generation  # no new session was armed

    release.set()
    platform._finalize_thread.join(timeout=2.0)

    # The first session — and only it — completes; a fresh Start is now allowed.
    completed = platform.state_snapshot()
    assert completed["state"] == "COMPLETE"
    assert completed["start_authorized"] is True
    assert platform.control("start")["ok"] is True
    assert platform._generation == first_generation + 1


def test_production_refuses_a_subclass_of_the_registered_writer(tmp_path):
    # Exact-type binding: even a subclass of the registered production writer is
    # refused, so nothing can subclass its way into a production recording.
    class _SneakySubclass(_ProductionWriter):
        pass

    platform, captures = _make_platform(
        tmp_path,
        contract=_PRODUCTION_CONTRACT,
        production_blockers=(),
        capture_class=_SneakySubclass,
        writer_types={_PRODUCTION_CONTRACT.writer_identity: _ProductionWriter},
    )

    result = platform.control("start")

    assert result["ok"] is False
    assert platform.state is not CollectionState.RECORDING
    assert captures[0].started == 0


def test_production_without_a_registry_is_never_authorized(tmp_path):
    # No registry -> no non-forgeable identity -> production can never start,
    # and the single authority says so before any allocation.
    platform, captures = _make_platform(
        tmp_path,
        contract=_PRODUCTION_CONTRACT,
        production_blockers=(),
        capture_class=_ProductionWriter,
        writer_types={},  # explicitly empty: no registered production writer
    )

    reason = platform.start_authorization()
    assert reason is not None
    assert "registered production writer" in reason
    assert platform.control("start")["ok"] is False
    assert captures == []


def test_abort_during_finalizing_drains_finalizer_and_clears_active(tmp_path):
    # Abort of a FINALIZING session must clear the active flag, best-effort join
    # the finalizer, and its late return must never resurrect the aborted
    # session or leave a lingering non-daemon thread that blocks the next Start.
    entered = threading.Event()
    release = threading.Event()

    class SlowCapture(_Capture):
        def finalize(self) -> dict:
            self.finalized += 1
            entered.set()
            release.wait(timeout=2.0)
            return {"status": "diagnostic_complete", "training_eligible": False}

    captures: list[_Capture] = []

    def factory(session_dir: Path) -> SlowCapture:
        capture = SlowCapture(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
        finalize_join_timeout_s=0.2,
    )
    assert platform.control("start")["ok"]
    assert platform.control("end")["state"] == "FINALIZING"
    assert entered.wait(timeout=1.0)

    result = platform.abort("operator abort")  # bounded join times out here
    assert result["ok"] is True
    assert platform.state is CollectionState.ABORTED
    assert platform._finalization_active is False

    # Let the still-blocked finalizer finish; its late return must not resurrect.
    release.set()
    platform._finalize_thread.join(timeout=2.0)
    assert platform.state is CollectionState.ABORTED
    assert platform._finalization_active is False
    assert platform.state_snapshot()["finalization_active"] is False
    assert platform.control("start")["ok"] is True


def test_abort_cancels_the_finalizer_and_prevents_a_double_commit(tmp_path):
    # The reviewer's repro: an Abort during FINALIZING must cooperatively cancel
    # the finalizer so it bails BEFORE publishing a diagnostic_complete seal.  A
    # session must never carry both an ABORTED state and a completed seal.
    entered = threading.Event()
    sealed: list[bool] = []

    class CancellableCapture(_Capture):
        def __init__(self, session_dir: Path) -> None:
            super().__init__(session_dir)
            self._cancel = threading.Event()

        def request_cancel(self) -> None:
            self._cancel.set()

        def finalize(self) -> dict:
            self.finalized += 1
            entered.set()
            for _ in range(400):
                if self._cancel.is_set():
                    # Bail before the irreversible seal, like the real capture.
                    raise RuntimeError("finalization cancelled by abort")
                time.sleep(0.005)
            sealed.append(True)  # would publish the diagnostic_complete seal
            return {"status": "diagnostic_complete", "training_eligible": False}

    captures: list[_Capture] = []

    def factory(session_dir: Path) -> CancellableCapture:
        capture = CancellableCapture(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
        finalize_join_timeout_s=2.0,
    )
    assert platform.control("start")["ok"]
    assert platform.control("end")["state"] == "FINALIZING"
    assert entered.wait(timeout=1.0)

    result = platform.abort("operator abort")  # requests cancel, then joins
    assert result["ok"] is True
    assert platform.state is CollectionState.ABORTED
    assert platform._finalization_active is False

    platform._finalize_thread.join(timeout=2.0)
    # The finalizer was cancelled: it never sealed, and the session stays ABORTED.
    assert sealed == []
    assert platform.state is CollectionState.ABORTED
    assert platform.state_snapshot()["finalization_active"] is False


class _SealingCapture(_Capture):
    """Fake capture that mirrors the real terminal CAS.

    ``finalize`` seals under a terminal lock after the last cancel checkpoint;
    ``abort`` raises once that seal is committed.  This lets the platform race
    tests exercise the exact mutual-exclusion the production writer provides
    without any camera/encoder.
    """

    def __init__(self, session_dir: Path, *, seal_reached: threading.Event) -> None:
        super().__init__(session_dir)
        self._terminal_lock = threading.RLock()
        self._cancel = threading.Event()
        self._term_state = "recording"
        self._seal_reached = seal_reached

    def request_cancel(self) -> None:
        self._cancel.set()

    def finalize(self) -> dict:
        self.finalized += 1
        with self._terminal_lock:
            if self._cancel.is_set():
                self._term_state = "aborted"
                raise DiagnosticCaptureCancelled("cancelled before seal")
            self._term_state = "complete"
        self._seal_reached.set()
        return {"status": "diagnostic_complete", "training_eligible": False}

    def abort(self, reason: str) -> None:
        with self._terminal_lock:
            if self._term_state == "complete":
                raise RuntimeError("cannot abort a completed diagnostic capture")
            if self._term_state == "aborted":
                return
            self._term_state = "aborted"
            self.aborted.append(reason)


def _sealing_platform(tmp_path, *, completion_validator, finalize_join_timeout_s):
    seal_reached = threading.Event()
    captures: list[_SealingCapture] = []

    def factory(session_dir: Path) -> _SealingCapture:
        capture = _SealingCapture(session_dir, seal_reached=seal_reached)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
        completion_validator=completion_validator,
        finalize_join_timeout_s=finalize_join_timeout_s,
    )
    return platform, captures, seal_reached


def test_abort_loses_to_a_seal_committed_during_the_join(tmp_path):
    # Complete-wins: the capture seals diagnostic_complete before Abort's cancel
    # can bite, the finalizer commits COMPLETE while Abort is joining it, and
    # Abort must report "already completed" and NEVER write ABORTED over the seal.
    validator_gate = threading.Event()

    def completion_validator(result, session_dir):  # noqa: ANN001
        assert validator_gate.wait(timeout=2.0)

    platform, captures, seal_reached = _sealing_platform(
        tmp_path, completion_validator=completion_validator, finalize_join_timeout_s=2.0,
    )
    assert platform.control("start")["ok"]
    assert platform.control("end")["state"] == "FINALIZING"
    # The capture sealed; the finalizer is now parked in completion_validator with
    # the platform still FINALIZING.
    assert seal_reached.wait(timeout=1.0)

    abort_box: dict = {}

    def do_abort() -> None:
        abort_box["result"] = platform.abort("late abort")

    aborter = threading.Thread(target=do_abort)
    aborter.start()
    time.sleep(0.05)  # let Abort request-cancel (too late) and reach the join
    validator_gate.set()  # unpark the finalizer -> it commits COMPLETE -> join returns
    aborter.join(timeout=2.0)

    platform.wait_for_terminal(timeout=2.0)
    assert platform.state is CollectionState.COMPLETE
    result = abort_box["result"]
    assert result["ok"] is True
    assert result.get("accepted") is False
    assert "already completed" in result["ack"].lower()
    # Complete won cleanly: the capture was never aborted.
    assert captures[0].aborted == []


def test_abort_join_timeout_still_defers_to_a_sealed_capture(tmp_path):
    # Even when Abort's bounded join times out while the finalizer is stuck in a
    # slow completion_validator, the capture's durable seal is authoritative:
    # Abort probes the capture's terminal CAS, sees it sealed, and refuses to
    # write ABORTED.  The finalizer later commits COMPLETE.
    validator_gate = threading.Event()

    def completion_validator(result, session_dir):  # noqa: ANN001
        assert validator_gate.wait(timeout=3.0)

    platform, captures, seal_reached = _sealing_platform(
        tmp_path, completion_validator=completion_validator, finalize_join_timeout_s=0.2,
    )
    assert platform.control("start")["ok"]
    assert platform.control("end")["state"] == "FINALIZING"
    assert seal_reached.wait(timeout=1.0)

    result = platform.abort("timeout abort")  # join times out; probes the sealed capture
    assert result["ok"] is True
    assert result.get("accepted") is False
    assert "already completed" in result["ack"].lower()
    # Never wrote ABORTED over the seal; the probe hit the complete-refusal branch
    # before recording an abort.
    assert platform.state is not CollectionState.ABORTED
    assert captures[0].aborted == []

    validator_gate.set()
    assert platform.wait_for_terminal(timeout=2.0)["state"] == "COMPLETE"
    assert platform.state is CollectionState.COMPLETE


def test_clean_cancellation_drives_aborted_without_a_finalization_error(tmp_path):
    # Cancel-wins: the capture bails on its cancel checkpoint (raising the real
    # DiagnosticCaptureCancelled), so the platform drives the ABORTED terminal via
    # the cancellation path -- not the ERROR fallback -- and no seal is written.
    entered = threading.Event()
    sealed: list[bool] = []

    class CleanCancelCapture(_Capture):
        def __init__(self, session_dir: Path) -> None:
            super().__init__(session_dir)
            self._cancel = threading.Event()

        def request_cancel(self) -> None:
            self._cancel.set()

        def finalize(self) -> dict:
            self.finalized += 1
            entered.set()
            for _ in range(400):
                if self._cancel.is_set():
                    raise DiagnosticCaptureCancelled("finalization cancelled by abort")
                time.sleep(0.005)
            sealed.append(True)
            return {"status": "diagnostic_complete", "training_eligible": False}

    captures: list[CleanCancelCapture] = []

    def factory(session_dir: Path) -> CleanCancelCapture:
        capture = CleanCancelCapture(session_dir)
        captures.append(capture)
        return capture

    platform = CollectionPlatform(
        session_root=tmp_path / "sessions",
        point_names=["R1"],
        capture_factory=factory,
        writer_types={"DiagnosticPerceptionCapture": _Capture},
        finalize_join_timeout_s=2.0,
    )
    assert platform.control("start")["ok"]
    assert platform.control("end")["state"] == "FINALIZING"
    assert entered.wait(timeout=1.0)

    result = platform.abort("operator abort")
    assert result["ok"] is True

    assert platform.wait_for_terminal(timeout=2.0)["state"] == "ABORTED"
    assert platform.state is CollectionState.ABORTED
    assert sealed == []
    assert platform.state_snapshot()["finalization_active"] is False
    # Proves the clean cancellation path, not the ERROR->ABORTED fallback.
    assert "cancelled by abort" in (platform.last_error or "")
    assert "finalization failed" not in (platform.last_error or "")
