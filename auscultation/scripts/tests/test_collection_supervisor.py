"""Fail-closed collection supervisor tests without robot hardware."""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import collection_supervisor as sup  # noqa: E402
import session_to_lerobot_v2 as conv  # noqa: E402


def _release_provenance() -> dict:
    return {
        "schema_version": "wam_capture_release_provenance_v1",
        "code": {"release_id": "code_release-" + "a" * 20},
        "runtime": {
            "release_id": "perception_runtime-" + "b" * 20,
            "lock_sha256": "c" * 64,
        },
    }


def _capture_durability(
    frame_count: int = 1, *, first_frame_index: int = 2, obs_rows: int = 2,
) -> dict:
    last_frame_index = first_frame_index + frame_count - 1
    start_boundary_index = first_frame_index - 1
    return {
        "schema_version": "wam_streaming_capture_v1",
        "audio": {
            "streamed_to_disk": True, "n_samples": 32_000,
            "sample_rate": 16_000, "sha256": "1" * 64,
            "timeline": "audio_times.jsonl",
        },
        "video": {
            "path": "rgb.mp4", "frame_count": frame_count,
            "decoded_frame_count": frame_count, "segment_count": 1,
            "segment_frames_limit": 150, "bytes": 100,
            "sha256": "2" * 64,
        },
        "capture_fifo": {
            "capacity": 30, "accepted_frames": frame_count,
            "processed_frames": frame_count, "overflow_frames": 0,
            "missing_camera_frames": 0, "boundary_rejections": 0,
            "sequence_policy": "strict_contiguous_camera_index",
            "camera_backend": "orbbec",
            "start_boundary_frame_index": start_boundary_index,
            "start_boundary_frame_t": 0.99,
            "first_expected_frame_index": first_frame_index,
            "first_accepted_frame_index": first_frame_index,
            "start_boundary_verified": True, "max_queue_depth": 1,
            "queue_depth": 0, "available_queue_slots": 30,
            "last_submitted_frame_index": last_frame_index,
            "last_processed_frame_index": last_frame_index,
            "first_host_t": 1.0, "last_host_t": 2.0,
            "worker_error": None, "fatal_reason": None,
            "t_end_monotonic": 2.1, "drained": True,
        },
        "camera_delivery_fifo": {
            "status": "OK", "capacity": 32, "interval_id": 7,
            "interval_active": False, "accepting": False, "closing": False,
            "queue_depth": 0, "max_queue_depth": 1,
            "interval_enqueued_frames": frame_count,
            "interval_delivered_frames": frame_count,
            "first_enqueued_frame_index": first_frame_index,
            "last_enqueued_frame_index": last_frame_index,
            "last_delivered_frame_index": last_frame_index,
            "overflow_frames": 0, "interval_overflow_frames": 0,
            "first_overflow_frame_index": None,
            "rejected_after_overflow": 0, "discarded_frames": 0,
        },
        "camera_start_boundary": {
            "boundary_latched_atomically": True, "interval_id": 7,
            "start_boundary_frame_index": start_boundary_index,
            "start_boundary_host_t": 0.99,
            "inflight_pre_boundary_seen": False,
            "inflight_pre_boundary_completed": True,
            "inflight_pre_boundary_produced_frame": False,
            "capacity": 32, "overflow_baseline": 0,
        },
        "camera_end_boundary": {
            "boundary_latched_atomically": True,
            "inflight_pre_boundary_seen": True,
            "inflight_pre_boundary_completed": True,
            "inflight_pre_boundary_produced_frame": True,
            "post_boundary_overwrite_prevented": True,
            "snapshot_frame_index": last_frame_index,
            "snapshot_host_t": 2.0, "interval_id": 7,
            "delivery_capacity": 32, "delivery_queue_depth": 1,
            "delivery_enqueued_frames": frame_count,
            "delivery_delivered_frames": frame_count - 1,
            "delivery_first_frame_index": first_frame_index,
            "delivery_last_frame_index": last_frame_index,
            "delivery_overflow_frames": 0,
            "t_end_monotonic": 2.1, "tail_frame_submitted": False,
            "tail_frame_already_submitted": True,
            "tail_frame_precedes_start": False,
            "fifo_last_submitted_frame_index": last_frame_index,
            "fifo_last_processed_frame_index": last_frame_index,
        },
        "timelines": {
            "rgb_times.csv": {"rows": frame_count, "sha256": "3" * 64},
            "obs.csv": {"rows": obs_rows, "sha256": "4" * 64},
            "depth/depth_times.csv": {"rows": frame_count, "sha256": "5" * 64},
        },
        "crash_recovery_policy": "diagnostic_only_never_training_eligible",
    }


def test_artifact_hash_inventory_binds_raw_and_depth_but_not_manifest(tmp_path):
    for name in sup.REQUIRED_SESSION_ARTIFACTS:
        (tmp_path / name).write_bytes(f"raw:{name}".encode())
    (tmp_path / "manifest.json").write_text('{"status":"sensors_complete"}\n')
    depth = tmp_path / "depth"
    depth.mkdir()
    (depth / "depth_times.csv").write_text("frame_index,host_arrival_t_s\n0,1.0\n")
    (depth / "depth_000000.npy").write_bytes(b"depth-frame")

    inventory = sup._artifact_hash_inventory(tmp_path)

    assert inventory["schema_version"] == "wam_session_artifact_hashes_v1"
    assert "manifest.json" not in inventory["files"]
    assert set(sup.REQUIRED_SESSION_ARTIFACTS).issubset(inventory["files"])
    assert "depth/depth_times.csv" in inventory["files"]
    assert "depth/depth_000000.npy" in inventory["files"]
    assert inventory["files"]["audio.wav"]["sha256"] == hashlib.sha256(
        (tmp_path / "audio.wav").read_bytes()
    ).hexdigest()


def test_only_confirmed_normal_stop_outcomes_can_promote_training_data():
    for outcome in ("normal_stop_direct_disable", "normal_stop_fallback_disable"):
        sup._require_training_eligible_stop(outcome)
    for outcome in (
        "screened_limit_direct_disable", "hard_limit_emergency_disable",
        "unconfirmed_direct_disable", "normal_stop_acknowledged",
    ):
        with pytest.raises(ValueError, match="normal completion"):
            sup._require_training_eligible_stop(outcome)


def test_legacy_robot_csv_cannot_authorize_training_without_rx_liveness():
    with pytest.raises(
        ValueError,
        match="per-motor decoded CAN RX counters and update timestamps",
    ):
        sup._require_observable_robot_feedback_liveness()


def test_legacy_supervisor_contains_no_training_eligibility_publisher():
    source = (SCRIPTS / "collection_supervisor.py").read_text(encoding="utf-8")
    assert "training_eligible=True" not in source


def test_aborting_a_session_removes_any_stale_finalization_seal(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "FINALIZATION.json").write_text("{}")

    sup.mark_aborted(tmp_path, "late safety event")

    assert not (tmp_path / "FINALIZATION.json").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "aborted"


def test_legacy_finalization_publisher_rejects_training_eligibility(tmp_path):
    with pytest.raises(ValueError, match="legacy finalization publisher"):
        sup._publish_finalized_manifest(tmp_path, {
            "schema_version": "wam_session_v2",
            "status": "complete",
            "training_eligible": True,
            "promotion_contract_version": "wam_promotion_v1",
        })
    assert not (tmp_path / "FINALIZATION.json").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_retired_v1_raced_publication_cannot_enter_v2_conversion(
    tmp_path, monkeypatch,
):
    manifest = {
        "schema_version": "wam_session_v2",
        "promotion_contract_version": "wam_promotion_v1",
        "arm": {"side": "left"},
        "training_eligible": False,
    }
    original_atomic_bytes = sup._atomic_bytes

    def replace_manifest_after_publish(path, raw):
        original_atomic_bytes(path, raw)
        if Path(path).name == "manifest.json":
            Path(path).write_text(json.dumps({
                **manifest, "arm": {"side": "right"},
            }))

    monkeypatch.setattr(sup, "_atomic_bytes", replace_manifest_after_publish)
    sup._publish_finalized_manifest(tmp_path, manifest)

    # The legacy publisher intentionally cannot emit the independent v2
    # authenticity receipt.  The v2 converter must reject it before treating
    # either the legacy seal or its raced manifest as promotable evidence.
    with pytest.raises(FileNotFoundError, match="training authenticity receipt"):
        conv._verify_finalization_seal(tmp_path)


def test_retired_v1_crash_never_exposes_a_v2_convertible_session(
    tmp_path, monkeypatch,
):
    initial = {
        "schema_version": "wam_session_v2", "status": "sensors_complete",
        "training_eligible": False,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(initial))
    final = {
        **initial, "status": "complete", "training_eligible": False,
        "promotion_contract_version": "wam_promotion_v1",
    }
    original_atomic_bytes = sup._atomic_bytes
    events = []

    def fail_final_manifest(path, raw):
        events.append(Path(path).name)
        if Path(path).name == "manifest.json":
            raise OSError("simulated crash before final manifest commit")
        original_atomic_bytes(path, raw)

    monkeypatch.setattr(sup, "_atomic_bytes", fail_final_manifest)
    with pytest.raises(OSError, match="simulated crash"):
        sup._publish_finalized_manifest(tmp_path, final)

    assert events == ["FINALIZATION.json", "manifest.json"]
    assert json.loads((tmp_path / "manifest.json").read_text()) == initial
    with pytest.raises(FileNotFoundError, match="training authenticity receipt"):
        conv._verify_finalization_seal(tmp_path)


class _Process:
    def __init__(self, exits_on_signal):
        self.exits_on_signal = exits_on_signal; self.signals = []; self.returncode = None

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        if self.exits_on_signal:
            self.returncode = 0

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("teleop", timeout)
        return self.returncode

    def terminate(self):
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.returncode = -signal.SIGKILL


class _StubbornProcess(_Process):
    def __init__(self):
        super().__init__(exits_on_signal=False)
        self.terminate_calls = 0
        self.kill_calls = 0

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("teleop", timeout)
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1
        self.returncode = -signal.SIGKILL


class _AckOnWaitProcess(_Process):
    def __init__(self):
        super().__init__(exits_on_signal=False)

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class _ExitDuringSignalProcess(_Process):
    def __init__(self):
        super().__init__(exits_on_signal=False)

    def send_signal(self, sig):
        self.signals.append(sig)
        self.returncode = 0
        raise ProcessLookupError("process exited during signal delivery")


class _WaitTrackingProcess(_Process):
    def __init__(self):
        super().__init__(exits_on_signal=False)
        self.wait_called = False

    def wait(self, timeout=None):
        self.wait_called = True
        return super().wait(timeout=timeout)


class _SignalDeniedProcess(_Process):
    def __init__(self):
        super().__init__(exits_on_signal=False)

    def send_signal(self, sig):
        self.signals.append(sig)
        raise PermissionError("signal transport denied")


def test_screened_limit_starts_energy_isolation_without_waiting_for_teleop_ack():
    process = _WaitTrackingProcess()
    wait_state_at_disable = []

    sup.stop_teleop(
        process, severity="screened", stop_cmd=["disable-now"], timeout_s=0.001,
        run=lambda cmd, **kwargs: (
            wait_state_at_disable.append(process.wait_called)
            or subprocess.CompletedProcess(cmd, 0)
        ),
    )

    assert wait_state_at_disable == [False]


def test_signal_delivery_error_cannot_skip_independent_direct_disable():
    process = _SignalDeniedProcess()
    calls = []

    outcome = sup.stop_teleop(
        process, severity="screened", stop_cmd=["disable-now"], timeout_s=0.001,
        run=lambda cmd, **kwargs: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
    )

    assert calls == [["disable-now"]]
    assert outcome == "screened_limit_direct_disable"


def test_screened_limit_requests_shutdown_and_immediate_direct_disable():
    process = _Process(exits_on_signal=False); calls = []
    outcome = sup.stop_teleop(
        process, severity="screened", stop_cmd=["disable"],
        run=lambda cmd, **kwargs: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
        timeout_s=0.001,
    )
    assert process.signals == [signal.SIGINT]
    assert calls == [["disable"]]
    assert outcome == "screened_limit_direct_disable"


def test_hard_limit_invokes_direct_disable_without_waiting_for_ack():
    process = _Process(exits_on_signal=False); events = []
    outcome = sup.stop_teleop(
        process, severity="hard", stop_cmd=["disable-now"],
        run=lambda cmd, **kwargs: events.append("disable") or subprocess.CompletedProcess(cmd, 0),
        timeout_s=0.25,
    )
    assert events[0] == "disable"
    assert process.signals == [signal.SIGINT]
    assert outcome == "hard_limit_emergency_disable"


def test_failed_direct_disable_is_never_reported_as_confirmed():
    process = _Process(exits_on_signal=False)
    with pytest.raises(RuntimeError, match="direct motor-disable failed"):
        sup.stop_teleop(
            process, severity="hard", stop_cmd=["broken-disable"],
            run=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 7),
        )
    assert process.signals == [signal.SIGINT]
    assert process.returncode == -signal.SIGTERM


def test_failed_screened_fallback_still_terminates_teleop():
    process = _Process(exits_on_signal=False)
    with pytest.raises(RuntimeError, match="direct motor-disable failed"):
        sup.stop_teleop(
            process, severity="screened", stop_cmd=["broken-disable"], timeout_s=0.001,
            run=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 7),
        )
    assert process.signals == [signal.SIGINT]
    assert process.returncode == -signal.SIGTERM


def test_hanging_disable_still_kills_and_reaps_unresponsive_teleop():
    process = _StubbornProcess()

    def hanging_disable(cmd, **kwargs):
        assert kwargs["timeout"] == pytest.approx(0.001)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    with pytest.raises(RuntimeError, match="direct motor-disable timed out"):
        sup.stop_teleop(
            process, severity="hard", stop_cmd=["hanging-disable"], timeout_s=0.001,
            run=hanging_disable,
        )
    assert process.signals == [signal.SIGINT]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.poll() == -signal.SIGKILL


def test_hard_limit_runs_direct_disable_even_if_teleop_already_exited():
    process = _Process(exits_on_signal=False); process.returncode = 0; calls = []
    outcome = sup.stop_teleop(
        process, severity="hard", stop_cmd=["disable-now"],
        run=lambda cmd, **kwargs: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
    )
    assert calls == [["disable-now"]]
    assert process.signals == []
    assert outcome == "hard_limit_emergency_disable"


def test_hard_limit_still_runs_direct_disable_if_process_exits_during_signal():
    process = _ExitDuringSignalProcess(); calls = []
    outcome = sup.stop_teleop(
        process, severity="hard", stop_cmd=["disable-now"],
        run=lambda cmd, **kwargs: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
    )
    assert calls == [["disable-now"]]
    assert outcome == "hard_limit_emergency_disable"


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        ("screened", "screened_limit_direct_disable"),
        ("normal", "normal_stop_direct_disable"),
    ],
)
def test_nonhard_stop_runs_direct_disable_even_if_teleop_already_exited(severity, expected):
    process = _Process(exits_on_signal=False); process.returncode = 0; calls = []
    outcome = sup.stop_teleop(
        process, severity=severity, stop_cmd=["disable-now"],
        run=lambda cmd, **kwargs: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
    )
    assert calls == [["disable-now"]]
    assert outcome == expected


def test_screened_stop_runs_direct_disable_if_process_exits_during_signal():
    process = _ExitDuringSignalProcess(); calls = []
    outcome = sup.stop_teleop(
        process, severity="screened", stop_cmd=["disable-now"],
        run=lambda cmd, **kwargs: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
    )
    assert calls == [["disable-now"]]
    assert outcome == "screened_limit_direct_disable"


def test_normal_acknowledgement_is_followed_by_direct_disable():
    process = _AckOnWaitProcess(); calls = []
    outcome = sup.stop_teleop(
        process, severity="normal", stop_cmd=["disable-now"],
        run=lambda cmd, **kwargs: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
    )
    assert calls == [["disable-now"]]
    assert outcome == "normal_stop_direct_disable"


def test_normal_shutdown_is_preempted_by_a_screened_safety_event():
    process = _AckOnWaitProcess(); calls = []; probes = iter([None, "screened"])
    outcome = sup.stop_teleop(
        process, severity="normal", stop_cmd=["disable-now"], timeout_s=0.01,
        run=lambda cmd, **kwargs: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
        severity_probe=lambda: next(probes, "screened"),
    )
    assert calls == [["disable-now"]]
    assert outcome == "screened_limit_direct_disable"


def test_capture_ended_ack_requires_monotonic_actual_duration(tmp_path):
    ack = tmp_path / "CAPTURE_ENDED.json"
    ack.write_text(json.dumps({
        "schema_version": "wam_capture_ended_v1",
        "reason": "operator_end",
        "t0_monotonic": 10.0,
        "t_end_monotonic": 12.5,
        "actual_duration_s": 2.5,
    }))
    assert sup.read_capture_ended_ack(ack)["actual_duration_s"] == 2.5
    payload = json.loads(ack.read_text())
    payload["actual_duration_s"] = 99.0
    ack.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duration"):
        sup.read_capture_ended_ack(ack)


def test_operator_stop_disables_teleop_before_hung_recorder_ack_or_exit(
    tmp_path, monkeypatch,
):
    """STOP_CAPTURE must not depend on the recorder reaching its frame boundary."""
    session = tmp_path / "session"
    capture_ended = session / "CAPTURE_ENDED.json"
    events = []
    stop_called = False
    finalized = []

    class Recorder:
        returncode = None
        polls_without_stop = 0

        def poll(self):
            # Model a recorder stuck in frame/disk I/O.  Once the independently
            # supervised stop has disabled teleop, it eventually acknowledges
            # the frame boundary, flushes, and exits.  The no-stop escape keeps
            # the pre-fix test finite and exposes the unsafe ordering.
            if stop_called:
                if not capture_ended.exists():
                    events.append("recorder_ack")
                    capture_ended.write_text(json.dumps({
                        "schema_version": "wam_capture_ended_v1",
                        "reason": "operator_end",
                        "t0_monotonic": 10.0,
                        "t_end_monotonic": 12.0,
                        "actual_duration_s": 2.0,
                    }))
                    return None
                events.append("recorder_exit")
                self.returncode = 0
                return self.returncode
            self.polls_without_stop += 1
            if self.polls_without_stop >= 3:
                events.append("recorder_exit_without_supervised_stop")
                self.returncode = 0
            return self.returncode

        def send_signal(self, _signum):
            self.returncode = -signal.SIGTERM

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("recorder", timeout)
            return self.returncode

    class Teleop:
        returncode = None

        def poll(self):
            return self.returncode

    recorder = Recorder()
    teleop = Teleop()
    spawned = []

    def popen(_command, **_kwargs):
        if not spawned:
            (session / "SENSORS_READY.json").write_text(json.dumps({
                "ready": True, "force_calibrated": True,
            }))
            child = recorder
        else:
            (session / "STOP_CAPTURE.json").write_text(json.dumps({
                "reason": "operator_end",
            }))
            child = teleop
        spawned.append(child)
        return child

    def disable_teleop(_process, *, severity, **_kwargs):
        nonlocal stop_called
        assert severity == "normal"
        assert recorder.returncode is None
        assert not capture_ended.exists()
        events.append("teleop_disabled")
        stop_called = True
        teleop.returncode = 0
        return "normal_stop_direct_disable"

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    monkeypatch.setattr(sup, "stop_teleop", disable_teleop)
    monkeypatch.setattr(
        sup, "finalize_session",
        lambda *_args, **_kwargs: finalized.append(True) or {},
    )

    code = sup.supervise(
        session, ["recorder"], ["teleop"], ["disable"],
        readiness_timeout_s=1.0,
    )

    assert code == 0
    assert events == ["teleop_disabled", "recorder_ack", "recorder_exit"]
    assert finalized == [True]
    stopped = json.loads((session / "TELEOP_STOPPED.json").read_text())
    assert stopped["trigger"] == "operator_stop_marker"
    assert stopped["capture_ended"]["reason"] == "operator_end"


def test_operator_stop_without_recorder_ack_is_aborted_not_finalized(
    tmp_path, monkeypatch,
):
    session = tmp_path / "session"
    stop_called = False
    spawned = []

    class Recorder:
        returncode = None

        def poll(self):
            if stop_called:
                self.returncode = 0
            return self.returncode

    class Teleop:
        returncode = None

        def poll(self):
            return self.returncode

    recorder = Recorder()
    teleop = Teleop()

    def popen(_command, **_kwargs):
        if not spawned:
            (session / "SENSORS_READY.json").write_text(json.dumps({
                "ready": True, "force_calibrated": True,
            }))
            child = recorder
        else:
            (session / "STOP_CAPTURE.json").write_text("{}")
            child = teleop
        spawned.append(child)
        return child

    def stop(_process, *, severity, **_kwargs):
        nonlocal stop_called
        stop_called = True
        teleop.returncode = 0
        return (
            "normal_stop_direct_disable"
            if severity == "normal"
            else "hard_limit_emergency_disable"
        )

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    monkeypatch.setattr(sup, "stop_teleop", stop)
    monkeypatch.setattr(
        sup, "finalize_session",
        lambda *_args, **_kwargs: pytest.fail(
            "a recorder exit without CAPTURE_ENDED must not finalize"
        ),
    )

    code = sup.supervise(session, ["recorder"], ["teleop"], ["disable"])

    assert code == 5
    manifest = json.loads((session / "manifest.json").read_text())
    assert manifest["status"] == "aborted"
    assert "without CAPTURE_ENDED" in manifest["abort_reason"]
    assert not (session / "FINALIZATION.json").exists()


def test_production_supervisor_refuses_any_existing_session_directory(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    (session / "targets.jsonl").write_text("evidence\n")
    monkeypatch.setattr(
        sup.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("no child may start for a reused session"),
    )
    with pytest.raises(FileExistsError):
        sup.supervise(session, ["recorder"], ["teleop"], ["disable"])
    assert (session / "targets.jsonl").read_text() == "evidence\n"


def test_supervisor_rechecks_sentinel_after_normal_stop_before_promotion(
    tmp_path, monkeypatch,
):
    session = tmp_path / "session"
    ready = session / "SENSORS_READY.json"
    spawned = []

    class Child:
        def __init__(self, returncode):
            self.returncode = returncode

        def poll(self):
            return self.returncode

    def popen(_command, **_kwargs):
        if not spawned:
            session.mkdir(parents=True, exist_ok=True)
            ready.write_text(json.dumps({"ready": True, "force_calibrated": True}))
            child = Child(0)
        else:
            child = Child(None)
        spawned.append(child)
        return child

    def stop_then_trip(_process, *, severity, **_kwargs):
        assert severity == "normal"
        (session / "SAFETY_ABORT.json").write_text(json.dumps({
            "severity": "screened", "reason": "late screened event",
        }))
        return "screened_limit_direct_disable"

    finalized = []
    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    monkeypatch.setattr(sup, "stop_teleop", stop_then_trip)
    monkeypatch.setattr(
        sup, "finalize_session",
        lambda *_args, **_kwargs: finalized.append(True) or {},
    )

    code = sup.supervise(session, ["recorder"], ["teleop"], ["disable"])

    assert code == 2
    assert finalized == []
    assert json.loads((session / "manifest.json").read_text())["status"] == "aborted"


def test_screened_shutdown_records_hard_upgrade_during_immediate_disable():
    process = _Process(exits_on_signal=False); calls = []; probes = iter([None, "hard"])
    outcome = sup.stop_teleop(
        process, severity="screened", stop_cmd=["disable-now"], timeout_s=0.01,
        run=lambda cmd, **kwargs: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
        severity_probe=lambda: next(probes, "hard"),
    )
    assert calls == [["disable-now"]]
    assert process.signals == [signal.SIGINT]
    assert outcome == "hard_limit_emergency_disable"


def test_screened_disable_checks_once_more_for_hard_escalation_before_returning():
    process = _AckOnWaitProcess(); calls = []; probes = iter([None, "hard"])
    outcome = sup.stop_teleop(
        process, severity="screened", stop_cmd=["disable-now"], timeout_s=0.01,
        run=lambda cmd, **kwargs: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
        severity_probe=lambda: next(probes, "hard"),
    )
    assert calls == [["disable-now"]]
    assert outcome == "hard_limit_emergency_disable"


def test_robot_coverage_is_checked_at_every_camera_time():
    robot_t = np.array([1.0, 1.02, 1.04, 1.20, 1.22])
    camera_t = np.array([1.01, 1.03, 1.10, 1.21])
    with pytest.raises(ValueError, match="camera t=1.100000"):
        sup.validate_asof_coverage(robot_t, camera_t, max_age_s=0.05)


def test_internal_stream_gap_is_rejected_even_when_two_streams_gap_together():
    with pytest.raises(ValueError, match="robot internal gap"):
        sup.validate_internal_gaps(
            np.array([1.0, 1.02, 2.0, 2.02]), max_gap_s=0.05, stream="robot",
        )


def test_robot_metadata_rejects_non_numeric_joint_rows(tmp_path):
    path = tmp_path / "robot.csv"
    path.write_text("t,q1,a1\n1.0,not-a-number,0.0\n1.1,0.0,0.1\n")
    with pytest.raises(ValueError, match="non-numeric"):
        sup._robot_metadata(path)


def test_promotion_rejects_undecodable_audio_and_video(tmp_path):
    audio = tmp_path / "audio.wav"; audio.write_bytes(b"x")
    video = tmp_path / "rgb.mp4"; video.write_bytes(b"x")
    manifest = {
        "t0_monotonic": 1.0, "duration_s": 2.0,
        "audio": {"sr": 16_000, "t0_monotonic": 1.0},
    }
    with pytest.raises(ValueError, match="audio.wav"):
        sup._validate_audio_artifact(audio, manifest)
    with pytest.raises(ValueError, match="rgb.mp4"):
        sup._validate_video_artifact(video, expected_frames=20)


def test_promotion_rejects_malformed_observation_rows(tmp_path):
    path = tmp_path / "obs.csv"
    path.write_text(
        "t,force_n,force_status,audio_rms_db,audio_peak_db,cam_index,cam_t\n"
        "1.0,4.0,OK,-20,-10,0,1.0\n"
        "1.1,nan,OK,-20,-10,1,1.1\n"
    )
    with pytest.raises(ValueError, match="obs.csv.*finite"):
        sup._validate_obs_artifact(path, t0=1.0, duration_s=0.1, control_hz=10.0)


def test_promotion_rejects_observation_camera_reference_not_in_rgb_timeline(tmp_path):
    path = tmp_path / "obs.csv"
    path.write_text(
        "t,force_n,force_status,audio_rms_db,audio_peak_db,cam_index,cam_t\n"
        "1.0,4.0,OK,-20,-10,999,1.0\n"
        "1.1,4.0,OK,-20,-10,999,1.0\n"
    )
    with pytest.raises(ValueError, match="camera reference"):
        sup._validate_obs_artifact(
            path, t0=1.0, duration_s=0.1, control_hz=10.0,
            camera_frame_index=np.array([0, 1]), camera_host_t_s=np.array([1.0, 1.1]),
        )


def test_finalizer_rejects_rehearsal_or_uncalibrated_manifest(tmp_path):
    manifest = {
        "schema_version": "wam_session_v2", "status": "sensors_complete",
        "training_eligible": False, "rehearsal_overrides": [],
        "streams": ["force", "audio", "cam"], "force": {"calibrated": True},
        "sync": {
            "applied": False, "calibration_sha256": "a" * 64, "offset_std_ms": 1.0,
            "n_matched": 4, "unique_matches": True,
            "measurement_method": "physical_diaphragm_taps",
        },
        "camera": {
            "backend": "orbbec", "camera_id": "fixture-camera",
            "camera_id_source": "verified_orbbec_serial",
            "n_rgb_frames": 1, "n_depth_frames": 1, "has_depth": True,
            "timing": {
            "applied": False, "calibration_sha256": "b" * 64, "latency_ms": 12.0,
            "latency_std_ms": 1.0, "n_samples": 20,
            "measurement_method": "hardware_led_flash",
        }},
        "n_obs": 2,
        "health": {name: {"status": "OK"} for name in ("force", "audio", "camera")},
        "capture_durability": _capture_durability(),
        "release_provenance": _release_provenance(),
    }
    sup.validate_promotion_manifest(manifest)
    manifest["rehearsal_overrides"] = ["uncalibrated_force"]
    with pytest.raises(ValueError, match="rehearsal"):
        sup.validate_promotion_manifest(manifest)
    manifest["rehearsal_overrides"] = []
    manifest["health"]["audio"]["status"] = "STALE"
    with pytest.raises(ValueError, match="health"):
        sup.validate_promotion_manifest(manifest)
    manifest["health"]["audio"]["status"] = "OK"
    manifest["sync"]["offset_std_ms"] = 8.0
    with pytest.raises(ValueError, match="sync.*quality"):
        sup.validate_promotion_manifest(manifest)
    manifest["sync"]["offset_std_ms"] = 1.0
    manifest["camera"]["camera_id_source"] = "device_index"
    with pytest.raises(ValueError, match="serial-verifiable"):
        sup.validate_promotion_manifest(manifest)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("capture_fifo", "overflow_frames", 1),
        ("camera_delivery_fifo", "interval_delivered_frames", 0),
        ("camera_start_boundary", "interval_id", 8),
        ("camera_start_boundary", "inflight_pre_boundary_seen", None),
        ("camera_start_boundary", "start_boundary_host_t", float("nan")),
        ("camera_end_boundary", "post_boundary_overwrite_prevented", False),
        ("camera_end_boundary", "inflight_pre_boundary_seen", None),
        ("camera_end_boundary", "t_end_monotonic", float("nan")),
        ("video", "decoded_frame_count", 0),
    ],
)
def test_promotion_rejects_any_broken_two_fifo_or_start_end_proof(
    section, field, value,
):
    manifest = {
        "schema_version": "wam_session_v2", "status": "sensors_complete",
        "training_eligible": False, "rehearsal_overrides": [],
        "streams": ["force", "audio", "cam"], "force": {"calibrated": True},
        "sync": {
            "applied": False, "calibration_sha256": "a" * 64,
            "offset_std_ms": 1.0, "n_matched": 4, "unique_matches": True,
            "measurement_method": "physical_diaphragm_taps",
        },
        "camera": {
            "backend": "orbbec", "camera_id": "fixture-camera",
            "camera_id_source": "verified_orbbec_serial", "n_rgb_frames": 1,
            "n_depth_frames": 1, "has_depth": True,
            "timing": {
                "applied": False, "calibration_sha256": "b" * 64,
                "latency_ms": 12.0, "latency_std_ms": 1.0, "n_samples": 20,
                "measurement_method": "hardware_led_flash",
            },
        },
        "n_obs": 2,
        "health": {name: {"status": "OK"} for name in ("force", "audio", "camera")},
        "capture_durability": _capture_durability(),
        "release_provenance": _release_provenance(),
    }
    manifest["capture_durability"][section][field] = value

    with pytest.raises(ValueError, match="capture durability"):
        sup.validate_promotion_manifest(manifest)


def test_promotion_requires_versioned_code_and_runtime_provenance():
    manifest = {
        "schema_version": "wam_session_v2", "status": "sensors_complete",
        "training_eligible": False, "rehearsal_overrides": [],
        "streams": ["force", "audio", "cam"], "force": {"calibrated": True},
        "sync": {
            "applied": False, "calibration_sha256": "a" * 64,
            "offset_std_ms": 1.0, "n_matched": 4, "unique_matches": True,
            "measurement_method": "physical_diaphragm_taps",
        },
        "camera": {
            "backend": "orbbec", "camera_id": "fixture-camera",
            "camera_id_source": "verified_orbbec_serial", "n_rgb_frames": 1,
            "n_depth_frames": 1, "has_depth": True,
            "timing": {
                "applied": False, "calibration_sha256": "b" * 64,
                "latency_ms": 12.0, "latency_std_ms": 1.0, "n_samples": 20,
                "measurement_method": "hardware_led_flash",
            },
        },
        "n_obs": 2, "capture_durability": _capture_durability(),
        "health": {name: {"status": "OK"} for name in ("force", "audio", "camera")},
    }
    with pytest.raises(ValueError, match="release provenance"):
        sup.validate_promotion_manifest(manifest)
    manifest["release_provenance"] = _release_provenance()
    manifest["release_provenance"]["runtime"]["lock_sha256"] = "not-a-lock"
    with pytest.raises(ValueError, match="release provenance"):
        sup.validate_promotion_manifest(manifest)


def test_depth_validation_rejects_non_uint16_even_when_shape_and_count_match(tmp_path):
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    (depth_dir / "depth_times.csv").write_text(
        "frame_index,t,depth_scale\n2,1.000000,1.0\n"
    )
    np.save(depth_dir / "depth_000002.npy", np.ones((6, 8), dtype=np.float32))
    camera = type("Camera", (), {
        "frame_index": np.asarray([2], dtype=np.int64),
        "host_arrival_t_s": np.asarray([1.0], dtype=np.float64),
    })()
    camera_meta = {
        "has_depth": True, "n_depth_frames": 1,
        "timing": {"capture_mode": {
            "width": 8, "height": 6,
            "depth_mode": {"width": 8, "height": 6},
            "align_mode": "software_depth_to_color",
        }},
    }

    with pytest.raises(ValueError, match="uint16"):
        sup._validate_depth_artifacts(tmp_path, camera, camera_meta)


def test_depth_validation_rejects_all_zero_payload(tmp_path):
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    (depth_dir / "depth_times.csv").write_text(
        "frame_index,t,depth_scale\n2,1.000000,1.0\n"
    )
    np.save(depth_dir / "depth_000002.npy", np.zeros((6, 8), dtype=np.uint16))
    camera = type("Camera", (), {
        "frame_index": np.asarray([2], dtype=np.int64),
        "host_arrival_t_s": np.asarray([1.0], dtype=np.float64),
    })()
    camera_meta = {
        "has_depth": True, "n_depth_frames": 1,
        "timing": {"capture_mode": {
            "width": 8, "height": 6,
            "depth_mode": {"width": 8, "height": 6},
            "align_mode": "software_depth_to_color",
        }},
    }

    with pytest.raises(ValueError, match="no valid depth"):
        sup._validate_depth_artifacts(tmp_path, camera, camera_meta)


def test_finalizer_verifies_sync_files_instead_of_trusting_hash_shape(tmp_path):
    manifest = {
        "schema_version": "wam_session_v2", "status": "sensors_complete",
        "training_eligible": False, "rehearsal_overrides": [],
        "streams": ["force", "audio", "cam"], "force": {"calibrated": True},
        "sync": {
            "applied": False, "calibration_sha256": "a" * 64,
            "calibration_path": "missing_sync.json", "offset_definition": "force_t_minus_audio_t",
            "offset_units": "ms", "audio_to_force_offset_ms": 0.0, "offset_std_ms": 1.0,
            "n_matched": 4, "unique_matches": True,
            "measurement_method": "physical_diaphragm_taps",
        },
        "camera": {
            "backend": "orbbec", "camera_id": "fixture-camera",
            "camera_id_source": "verified_orbbec_serial",
            "n_rgb_frames": 1, "n_depth_frames": 1, "has_depth": True,
            "timing": {
            "applied": False, "calibration_sha256": "b" * 64, "latency_ms": 12.0,
            "latency_std_ms": 1.0, "n_samples": 20,
            "measurement_method": "hardware_led_flash",
        }},
        "n_obs": 2,
        "health": {name: {"status": "OK"} for name in ("force", "audio", "camera")},
        "arm": {"side": "left"},
        "capture_durability": _capture_durability(),
        "release_provenance": _release_provenance(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "robot.csv").write_text("t,q1,a1\n1.0,0,0\n1.1,0,0\n")
    (tmp_path / "rgb_times.csv").write_text("frame_index,t\n0,1.05\n")
    with pytest.raises(FileNotFoundError, match="sync calibration"):
        sup.finalize_session(tmp_path, stop_outcome="normal_stop_acknowledged")


def test_arm_supervisor_rejects_rehearsal_flags_before_spawn():
    with pytest.raises(ValueError, match="rehearsal"):
        sup.validate_recorder_command(["python", "record_session.py", "--allow-uncalibrated-force"])


def test_shell_launcher_is_thin_fail_closed_supervisor_wrapper():
    script = (SCRIPTS / "record_teleop_session.sh").read_text()
    assert "set -euo pipefail" in script
    assert "collection_supervisor.py" in script
    assert "|| true" not in script
    assert "sleep 1.5" not in script
    assert "TELEOP_STOP_CMD_JSON" in script
    assert 'CAMERA_ID="${CAMERA_ID:?' in script
    assert '--camera-id "$CAMERA_ID"' in script
    assert 'WAM_HALL_CALIBRATION:-${HALL_CAL:-$HOME/wam_data/config/calibration/' in script
    assert '--hall-cal "$HALL_CAL"' in script
    assert 'CODE_RELEASE_ID="${WAM_CODE_RELEASE_ID:?' in script
    assert 'RUNTIME_RELEASE_ID="${WAM_PERCEPTION_RUNTIME_RELEASE_ID:?' in script
    assert 'RUNTIME_LOCK_SHA256="${WAM_PERCEPTION_RUNTIME_LOCK_SHA256:?' in script
    assert '--code-release-id "$CODE_RELEASE_ID"' in script
    assert '--runtime-release-id "$RUNTIME_RELEASE_ID"' in script
    assert '--runtime-lock-sha256 "$RUNTIME_LOCK_SHA256"' in script


def test_mock_teleop_uses_the_same_start_barrier_as_recorder():
    supervisor = (SCRIPTS / "collection_supervisor.py").read_text()
    mock = (SCRIPTS / "mock_teleop.py").read_text()
    assert 'env["WAM_START_FILE"]' in supervisor
    assert "WAM_START_FILE" in mock
    assert "while not start_file.exists()" in mock


def test_supervisor_validates_coverage_but_refuses_unobservable_robot_feedback(tmp_path):
    session = tmp_path / "session"; session.mkdir()
    force_cal_file = session / "force_cal.json"
    force_cal_file.write_text(json.dumps({"calibrated": True, "source": "bench_scale"}))
    force_cal_hash = __import__("hashlib").sha256(force_cal_file.read_bytes()).hexdigest()
    sync_capture_mode = {
        "audio_channel": 1, "force_calibration_sha256": force_cal_hash,
        "audio": {
            "requested_device": "fixture-audio", "portaudio_index": 3,
            "hostapi_name": "ALSA", "device_name": "Fixture USB Audio",
            "max_input_channels": 2, "usb_serial": "AUDIO-SERIAL-001",
            "usb_vid": 0x1235, "usb_pid": 0x8211,
            "usb_identity_source": "linux_alsa_sysfs",
        },
        "hall": {
            "requested_port": "/dev/fixture-hall", "device": "/dev/fixture-hall",
            "usb_serial": "HALL-SERIAL-001", "usb_vid": 0x2E8A, "usb_pid": 0x000A,
            "manufacturer": "Raspberry Pi", "product": "Pico", "location": "1-2",
        },
    }
    runtime_sync_mode = json.loads(json.dumps(sync_capture_mode))
    runtime_sync_mode["audio"]["portaudio_index"] = 7
    runtime_sync_mode["hall"]["requested_port"] = "/dev/ttyACM0"
    runtime_sync_mode["hall"]["device"] = "/dev/ttyACM0"
    sync_identity_match = {
        "hall_usb": "matched", "audio_usb": "matched",
        "audio_portaudio_index": "remapped",
    }
    sync_file = session / "sync.json"
    sync_file.write_text(json.dumps({
        "schema_version": "wam_sync_v1", "offset_definition": "force_t_minus_audio_t",
        "audio_to_force_offset_ms": 0.0, "offset_std_ms": 1.0, "n_matched": 4,
        "measurement_method": "physical_diaphragm_taps", "unique_matches": True,
        "capture_mode": sync_capture_mode,
    }))
    sync_hash = __import__("hashlib").sha256(sync_file.read_bytes()).hexdigest()
    camera_capture_mode = {
        "backend": "orbbec", "camera_id": "fixture-camera",
        "width": 32, "height": 24, "fps": 10.0,
        "time_source": "orbbec_device_and_host_arrival",
        "color_format": "RGB",
        "depth_mode": {"width": 32, "height": 24, "fps": 10.0, "format": "Y16"},
        "align_mode": "software_depth_to_color",
        "frame_aggregate_mode": "full_frame_require",
    }
    camera_negotiated_mode = {
        field: camera_capture_mode[field]
        for field in (
            "width", "height", "fps", "color_format", "depth_mode", "align_mode",
            "frame_aggregate_mode",
        )
    }
    camera_file = session / "camera_latency.json"
    camera_file.write_text(json.dumps({
        "schema_version": "wam_camera_latency_v1",
        "latency_definition": "host_arrival_minus_exposure", "latency_ms": 12.0,
        "latency_std_ms": 1.0, "n_samples": 20,
        "measurement_method": "hardware_led_flash",
        "capture_mode": camera_capture_mode,
    }))
    camera_hash = __import__("hashlib").sha256(camera_file.read_bytes()).hexdigest()

    # Valid artifacts: promotion must prove these can actually be consumed, not
    # merely that filenames exist.
    sample_rate = 16_000
    samples = (2_000 * np.sin(2 * np.pi * 220 * np.arange(int(2.2 * sample_rate)) / sample_rate)) \
        .astype("<i2")
    with wave.open(str(session / "audio.wav"), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    import cv2
    writer = cv2.VideoWriter(
        str(session / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24),
    )
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((24, 32, 3), i * 10, dtype=np.uint8))
    writer.release()
    exposure_times = np.asarray([1.0 + i * 110 * 0.001001 for i in range(20)])
    host_arrival_times = exposure_times + 0.012
    camera_frame_indices = np.arange(2, 22, dtype=np.int64)
    depth_dir = session / "depth"; depth_dir.mkdir()
    depth_rows = ["frame_index,t,depth_scale"]
    for position, (frame_index, timestamp) in enumerate(zip(camera_frame_indices, host_arrival_times)):
        np.save(
            depth_dir / f"depth_{frame_index:06d}.npy",
            np.full((24, 32), 500 + position, np.uint16),
        )
        depth_rows.append(f"{frame_index},{timestamp:.6f},1.0")
    (depth_dir / "depth_times.csv").write_text("\n".join(depth_rows) + "\n")
    obs = ["t,force_n,force_status,audio_rms_db,audio_peak_db,cam_index,cam_t"]
    for i in range(104):
        t = 1.02 + i * 0.02
        cam_position = max(0, int(np.searchsorted(host_arrival_times, t, side="right") - 1))
        obs.append(
            f"{t:.6f},4.0,OK,-20.0,-10.0,"
            f"{camera_frame_indices[cam_position]},{host_arrival_times[cam_position]:.6f}"
        )
    (session / "obs.csv").write_text("\n".join(obs) + "\n")
    targets = []
    for i, (frame_index, t) in enumerate(zip(camera_frame_indices, exposure_times)):
        targets.append(json.dumps({
            "t": float(t), "camera_frame_index": int(frame_index),
            "name": "R1" if i < 10 else "L1", "uv": [0.4, 0.5],
            "xyz": [0.1, 0.2, 0.3], "xyz_frame": "robot_base",
        }))
    (session / "targets.jsonl").write_text("\n".join(targets) + "\n")

    rgb_rows = ["frame_index,host_arrival_t_s,device_t_ms,time_source"]
    for position, frame_index in enumerate(camera_frame_indices):
        device_t_ms = position * 110
        rgb_rows.append(
            f"{frame_index},{1.012 + device_t_ms * 0.001001:.9f},"
            f"{device_t_ms},orbbec_device_and_host_arrival"
        )
    rgb_timeline_bytes = ("\n".join(rgb_rows) + "\n").encode()
    capture_durability = _capture_durability(
        20, first_frame_index=2, obs_rows=104,
    )
    capture_durability["audio"]["sha256"] = hashlib.sha256(
        (session / "audio.wav").read_bytes()
    ).hexdigest()
    capture_durability["video"]["sha256"] = hashlib.sha256(
        (session / "rgb.mp4").read_bytes()
    ).hexdigest()
    capture_durability["timelines"]["rgb_times.csv"]["sha256"] = hashlib.sha256(
        rgb_timeline_bytes
    ).hexdigest()
    capture_durability["timelines"]["obs.csv"]["sha256"] = hashlib.sha256(
        (session / "obs.csv").read_bytes()
    ).hexdigest()
    capture_durability["timelines"]["depth/depth_times.csv"]["sha256"] = hashlib.sha256(
        (depth_dir / "depth_times.csv").read_bytes()
    ).hexdigest()

    recorder = tmp_path / "recorder.py"
    recorder.write_text(f"""
import json,pathlib,sys,time
s=pathlib.Path(sys.argv[1]); s.mkdir(parents=True,exist_ok=True)
(s/'SENSORS_READY.json').write_text(json.dumps({{'ready':True,'force_calibrated':True}}))
deadline=time.time()+2
while not (s/'START_CAPTURE').exists() and time.time()<deadline: time.sleep(.01)
m={{'schema_version':'wam_session_v2','status':'sensors_complete','training_eligible':False,
   't0_monotonic':1.0,'duration_s':2.09,'control_hz':50.0,
   'n_obs':104,
   'rehearsal_overrides':[],'streams':['force','audio','cam'],
   'capture_durability':{capture_durability!r},
   'release_provenance':{_release_provenance()!r},
   'force':{{'calibrated':True,'port':'/dev/ttyACM0','calibration_path':'force_cal.json',
             'calibration_sha256':'{force_cal_hash}',
             'hardware_descriptor':{runtime_sync_mode['hall']!r}}},
   'sync':{{'applied':False,'calibration_path':'sync.json','calibration_sha256':'{sync_hash}',
           'offset_definition':'force_t_minus_audio_t','offset_units':'ms','audio_to_force_offset_ms':0.0,
           'offset_std_ms':1.0,'n_matched':4,'unique_matches':True,
           'measurement_method':'physical_diaphragm_taps','capture_mode':{sync_capture_mode!r},
           'runtime_capture_mode':{runtime_sync_mode!r},
           'identity_match':{sync_identity_match!r}}},
   'camera':{{'backend':'orbbec','camera_id':'fixture-camera',
              'camera_id_source':'verified_orbbec_serial','timing':{{
              'applied':False,'calibration_path':'camera_latency.json',
              'calibration_sha256':'{camera_hash}','latency_definition':'host_arrival_minus_exposure',
              'latency_ms':12.0,'latency_std_ms':1.0,'n_samples':20,
              'measurement_method':'hardware_led_flash','capture_mode':{camera_capture_mode!r},
              'mode_validation':{{'width':'matched_frame_shape','height':'matched_frame_shape',
                 'fps':'matched_negotiated_mode','time_source':'matched_frame_metadata',
                 'device_timestamp':'present','camera_id':'matched_device_serial',
                 'pipeline':'matched_negotiated_pipeline','depth':'matched_pipeline_shape',
                 'negotiated_mode':{camera_negotiated_mode!r}}}}},
              'n_rgb_frames':20,'n_depth_frames':20,'has_depth':True}},
   'audio':{{'sr':16000,'t0_monotonic':1.0,'device':'fixture-audio','channel':1,
            'hardware_descriptor':{runtime_sync_mode['audio']!r}}},
   'health':{{'force':{{'status':'OK'}},'audio':{{'status':'OK'}},'camera':{{'status':'OK'}}}},
   'arm':{{'side':'left','joint_names':[f'joint_{{i}}' for i in range(1,8)],
          'state_unit':'rad','action_type':'position_target'}}}}
(s/'manifest.json').write_text(json.dumps(m))
force=['host_t_s,raw,hall_v,displacement_mm,force_n,status,device_t_ms']
for d in range(0,2220,20): force.append(f"{{1.0+d*.001001:.9f}},1,1,0,4.0,OK,{{d}}")
(s/'force.csv').write_text('\\n'.join(force)+'\\n')
cam=['frame_index,host_arrival_t_s,device_t_ms,time_source']
for i in range(20):
    d=i*110; cam.append(f"{{i+2}},{{1.012+d*.001001:.9f}},{{d}},orbbec_device_and_host_arrival")
(s/'rgb_times.csv').write_text('\\n'.join(cam)+'\\n')
time.sleep(.2)
""")
    teleop = tmp_path / "teleop.py"
    teleop.write_text(
        "import os,pathlib,signal,sys,time\n"
        "signal.signal(signal.SIGINT,lambda *a: sys.exit(0))\n"
        "rows=['t,'+','.join([f'q{i}' for i in range(1,8)]+[f'a{i}' for i in range(1,8)])]\n"
        "rows += [f'{1+i*.02:.6f},'+','.join(['0']*14) for i in range(112)]\n"
        "pathlib.Path(os.environ['REC_CSV']).write_text('\\n'.join(rows)+'\\n')\n"
        "while True: time.sleep(.05)\n"
    )
    stop = tmp_path / "stop.py"; stop.write_text("raise SystemExit(0)\n")

    with pytest.raises(
        ValueError,
        match="per-motor decoded CAN RX counters and update timestamps",
    ):
        sup.supervise(
            session, [sys.executable, str(recorder), str(session)],
            [sys.executable, str(teleop)], [sys.executable, str(stop)],
            readiness_timeout_s=2.0, allow_preallocated_session=True,
        )
    manifest = json.loads((session / "manifest.json").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["training_eligible"] is False
    assert "per-motor decoded CAN RX counters" in manifest["abort_reason"]
    assert not (session / "FINALIZATION.json").exists()
