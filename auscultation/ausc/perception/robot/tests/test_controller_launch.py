"""Tests for the subprocess controller launch + direct-disable wrappers."""

from __future__ import annotations

import subprocess

import pytest

import signal

from ausc.perception.robot import controller_launch as controller_launch_module
from ausc.perception.robot.controller_launch import (
    ControllerLaunchError,
    DisableResult,
    SubprocessLaunchedController,
    make_subprocess_direct_disable,
    make_subprocess_launcher,
    parse_command,
)


def test_terminate_escalates_to_sigkill_when_sigterm_ignored(monkeypatch):
    # A controller that ignores SIGTERM must NOT survive teardown -- it could keep driving
    # the arm. terminate() escalates SIGTERM -> wait -> SIGKILL -> wait.
    sent = []
    # Force the send_signal fallback so we can observe which signals were delivered.
    monkeypatch.setattr(controller_launch_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        controller_launch_module.os, "killpg",
        lambda pgid, sig: (_ for _ in ()).throw(ProcessLookupError()),
    )

    class _StubbornProc:
        pid = 4321

        def __init__(self):
            self._alive = True
            self._waits = 0

        def poll(self):
            return None if self._alive else 0

        def wait(self, timeout=None):
            self._waits += 1
            if self._waits == 1:
                raise subprocess.TimeoutExpired(cmd="controller", timeout=timeout)
            return 0

        def send_signal(self, sig):
            sent.append(sig)
            if sig == signal.SIGKILL:
                self._alive = False   # SIGKILL takes effect

    SubprocessLaunchedController(_StubbornProc()).terminate(sigterm_grace_s=0.01)
    assert sent == [signal.SIGTERM, signal.SIGKILL]   # escalated after SIGTERM ignored


def test_terminate_is_a_noop_for_an_already_exited_controller():
    class _DeadProc:
        pid = 4321
        def poll(self):
            return 0                      # already exited
        def send_signal(self, sig):      # must not be called
            raise AssertionError("signalled an already-exited process")

    SubprocessLaunchedController(_DeadProc()).terminate()   # no raise, no signal


def test_parse_command():
    assert parse_command("teleop.sh left", label="x") == ("teleop.sh", "left")
    assert parse_command(["a", "b"], label="x") == ("a", "b")
    with pytest.raises(ControllerLaunchError):
        parse_command("", label="x")
    with pytest.raises(ControllerLaunchError):
        parse_command([], label="x")
    with pytest.raises(ControllerLaunchError):
        parse_command(["a", ""], label="x")


def test_launcher_uses_new_session_and_reports_pid():
    seen = {}

    class _Proc:
        pid = 4321

        def poll(self):
            return None

    def fake_popen(argv, **kw):
        seen["argv"] = argv
        seen["kw"] = kw
        return _Proc()

    launch = make_subprocess_launcher(("teleop.sh", "left"), popen=fake_popen)
    controller = launch()
    assert controller.pid == 4321
    assert controller.poll() is None
    assert seen["argv"] == ["teleop.sh", "left"]
    assert seen["kw"]["start_new_session"] is True          # own process group
    assert seen["kw"]["stdin"] == subprocess.DEVNULL


def test_launcher_wraps_os_error():
    def boom(*a, **k):
        raise OSError("no such file")

    launch = make_subprocess_launcher(("x",), popen=boom)
    with pytest.raises(ControllerLaunchError, match="teleop launch failed"):
        launch()


def test_launcher_runs_preflight_validation_before_popen_and_stops_on_failure():
    events = []

    class _Proc:
        pid = 4321

        def poll(self):
            return None

    launch = make_subprocess_launcher(
        ("/immutable/controller",),
        popen=lambda _argv, **_kwargs: events.append("popen") or _Proc(),
        pre_launch_validate=lambda: events.append("validate"),
    )
    launch()
    assert events == ["validate", "popen"]

    def refuse():
        events.append("refuse")
        raise ControllerLaunchError("mutated")

    refused = make_subprocess_launcher(
        ("/immutable/controller",),
        popen=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Popen must not run after validation failure")
        ),
        pre_launch_validate=refuse,
    )
    with pytest.raises(ControllerLaunchError, match="mutated"):
        refused()
    assert events[-1] == "refuse"


def test_launcher_freezes_cwd_and_environment_at_construction():
    seen = {}
    source_env = {"ONLY_PINNED": "1"}

    class _Proc:
        pid = 4321

        def poll(self):
            return None

    launch = make_subprocess_launcher(
        ("/immutable/controller",),
        cwd="/immutable/release",
        env=source_env,
        popen=lambda argv, **kwargs: seen.update(argv=argv, kwargs=kwargs) or _Proc(),
    )
    source_env["ONLY_PINNED"] = "mutated"
    source_env["LD_PRELOAD"] = "/tmp/hostile.so"
    launch()

    assert seen["kwargs"]["cwd"] == "/immutable/release"
    assert seen["kwargs"]["env"] == {"ONLY_PINNED": "1"}


def test_direct_disable_success():
    class _Done:
        returncode = 0

    disable = make_subprocess_direct_disable(("disable.sh",), run=lambda argv, **k: _Done())
    result = disable("operator_estop")
    assert isinstance(result, DisableResult)
    # P0-2: command success is reported as command_succeeded, NOT energy_isolated
    # (which requires an independent motor-disable readback).
    assert result.outcome == "disable_command_succeeded" and result.returncode == 0


def test_direct_disable_nonzero_raises():
    class _Done:
        returncode = 3

    disable = make_subprocess_direct_disable(("disable.sh",), run=lambda argv, **k: _Done())
    with pytest.raises(ControllerLaunchError, match="exited 3"):
        disable("estop")


def test_preflight_runs_command_and_nonzero_raises():
    # P0-1: preflight runs the setup command; a nonzero exit refuses bring-up.
    from ausc.perception.robot.controller_launch import make_subprocess_preflight

    seen = {}

    class _Ok:
        returncode = 0

    def run_ok(argv, **_k):
        seen["argv"] = argv
        return _Ok()

    make_subprocess_preflight(("clear.sh", "can1", "can3"), run=run_ok)()
    assert seen["argv"] == ["clear.sh", "can1", "can3"]

    class _Bad:
        returncode = 2

    with pytest.raises(ControllerLaunchError):
        make_subprocess_preflight(("clear.sh",), run=lambda _a, **_k: _Bad())()


def test_build_launch_from_activation_binds_sealed_controller_and_candidate_disable():
    class _Act:
        argv = ("controller/unilateral_control", "urdf/leader.urdf",
                "urdf/follower.urdf", "left_arm", "can1", "can3")
        cwd = "/rel/dir"
        environment = {"WAM_OPENARM_X": "1"}
        controller_path = "/rel/dir/controller/unilateral_control"
        preflight_path = "/rel/dir/tools/preflight"

    seen = {}

    class _Proc:
        pid = 4321
        def poll(self):
            return None

    def fake_popen(argv, **kw):
        seen["launch_argv"] = argv
        seen["launch_kw"] = kw
        return _Proc()

    runs = []

    class _Done:
        returncode = 0

    def fake_run(argv, **_kw):
        runs.append(list(argv))
        return _Done()

    launcher, preflight, disable = controller_launch_module.build_launch_from_activation(
        _Act(),
        disable_command=("bash", "/home/robot/wam_canary/hil_disable.sh"),
        popen=fake_popen,
        run=fake_run,
    )
    controller = launcher()
    assert controller.pid == 4321
    # argv[0] is bound to the sealed ABSOLUTE controller ELF (attestation binds the exact
    # sealed binary); the rest of the sealed argv and the sealed cwd/env are preserved.
    assert seen["launch_argv"][0] == "/rel/dir/controller/unilateral_control"
    assert seen["launch_argv"][1:] == [
        "urdf/leader.urdf", "urdf/follower.urdf", "left_arm", "can1", "can3",
    ]
    assert seen["launch_kw"]["cwd"] == "/rel/dir"
    assert seen["launch_kw"]["env"] == {"WAM_OPENARM_X": "1"}
    # preflight defaults to the sealed preflight binary
    preflight()
    assert runs[0] == ["/rel/dir/tools/preflight"]
    # disable runs the CANDIDATE stop-tool (only validated later by operator witness)
    result = disable("safe_stop_test")
    assert runs[1] == ["bash", "/home/robot/wam_canary/hil_disable.sh"]
    assert result.outcome == "disable_command_succeeded"


def test_direct_disable_timeout_raises():
    def timeout_run(argv, **k):
        raise subprocess.TimeoutExpired("disable.sh", 5)

    disable = make_subprocess_direct_disable(("disable.sh",), run=timeout_run, timeout_s=5.0)
    with pytest.raises(ControllerLaunchError, match="timed out"):
        disable("estop")


def test_reap_kills_stale_binary_processes_and_spares_others(tmp_path):
    """Bring-up reap: processes running the sealed controller binary are
    terminated (the arm is operator-disabled at every bring-up, so they are
    inert socket-holders); unrelated processes are untouched. Linux-only —
    inert where /proc is absent."""
    import os
    import shutil
    import subprocess
    import sys
    import time as _time

    from ausc.perception.robot.controller_launch import (
        reap_stale_controller_processes,
    )

    if not os.path.isdir("/proc"):
        assert reap_stale_controller_processes("/nonexistent") == ()
        return

    sleep_bin = shutil.which("sleep")
    fake_controller = tmp_path / "unilateral_control"
    shutil.copy(sleep_bin, fake_controller)
    fake_controller.chmod(0o755)
    stale = subprocess.Popen([str(fake_controller), "60"], start_new_session=True)
    bystander = subprocess.Popen([sleep_bin, "60"], start_new_session=True)
    try:
        _time.sleep(0.2)
        reaped = reap_stale_controller_processes(fake_controller)
        assert stale.pid in reaped
        deadline = _time.monotonic() + 5
        while stale.poll() is None and _time.monotonic() < deadline:
            _time.sleep(0.05)
        assert stale.poll() is not None, "stale controller survived reap"
        assert bystander.poll() is None, "bystander was killed"
    finally:
        for proc in (stale, bystander):
            try:
                proc.kill()
            except OSError:
                pass


def test_apply_controller_scheduling_reports_not_raises():
    """Scheduling application never raises: a running non-RT controller is
    strictly better than no controller."""
    import subprocess
    import sys

    from ausc.perception.robot.controller_launch import (
        apply_controller_scheduling,
    )

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        outcome = apply_controller_scheduling(child.pid, rt_priority=50)
        assert set(outcome) == {"affinity", "rt"}
        # RT will fail without RLIMIT_RTPRIO (and on macOS entirely): the
        # failure must be a recorded string, never an exception.
        assert outcome["rt"] == 50 or str(outcome["rt"]).startswith("failed:")
    finally:
        child.kill()


def test_launcher_runs_reap_and_scheduling_hooks(monkeypatch):
    import types

    import ausc.perception.robot.controller_launch as cl

    calls = {"reap": [], "sched": []}
    monkeypatch.setattr(
        cl, "reap_stale_controller_processes",
        lambda path, **k: calls["reap"].append(str(path)) or (),
    )
    monkeypatch.setattr(
        cl, "apply_controller_scheduling",
        lambda pid, **k: calls["sched"].append(pid) or {"affinity": None, "rt": None},
    )

    class FakeProc:
        pid = 4242
        def poll(self): return None

    launch = cl.make_subprocess_launcher(
        ("/fake/unilateral_control", "arg"), popen=lambda *a, **k: FakeProc(),
    )
    launch()
    assert calls["reap"] == ["/fake/unilateral_control"]
    assert calls["sched"] == [4242]
