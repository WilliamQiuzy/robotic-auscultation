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


def test_direct_disable_timeout_raises():
    def timeout_run(argv, **k):
        raise subprocess.TimeoutExpired("disable.sh", 5)

    disable = make_subprocess_direct_disable(("disable.sh",), run=timeout_run, timeout_s=5.0)
    with pytest.raises(ControllerLaunchError, match="timed out"):
        disable("estop")
