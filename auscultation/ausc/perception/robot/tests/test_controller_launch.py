"""Tests for the subprocess controller launch + direct-disable wrappers."""

from __future__ import annotations

import subprocess

import pytest

from ausc.perception.robot.controller_launch import (
    ControllerLaunchError,
    DisableResult,
    make_subprocess_direct_disable,
    make_subprocess_launcher,
    parse_command,
)


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


def test_direct_disable_timeout_raises():
    def timeout_run(argv, **k):
        raise subprocess.TimeoutExpired("disable.sh", 5)

    disable = make_subprocess_direct_disable(("disable.sh",), run=timeout_run, timeout_s=5.0)
    with pytest.raises(ControllerLaunchError, match="timed out"):
        disable("estop")
