"""Subprocess wrappers for launching + energy-isolating the left OpenArm controller.

These adapt a configured teleop launch command and a confirmed direct-disable
command to the ``launch_controller`` / ``direct_disable`` dependencies of
``TrustedControllerCompanion``.  They are deliberately thin: the companion owns
the safety lifecycle (attestation, READY gate, fault -> disable-first ordering);
these only start the process group and run the disable command.

The launch command MUST be the pinned auscultation teleop (``stethoscope/scripts/
teleop.sh`` left, can1 leader -> can3 follower); never the shared right-arm
project.  The disable command MUST be an independent energy-isolation command.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
import signal
import subprocess
from typing import Callable, Optional, Sequence


class ControllerLaunchError(RuntimeError):
    """The controller launch or disable command is invalid or failed."""


def _validate_command(command: Sequence[str], *, label: str) -> tuple:
    if not isinstance(command, (list, tuple)) or not command:
        raise ControllerLaunchError(f"{label} must be a non-empty argv list")
    argv = tuple(str(part) for part in command)
    if any(not part for part in argv):
        raise ControllerLaunchError(f"{label} contains an empty argument")
    return argv


class SubprocessLaunchedController:
    """A launched controller process group exposing the companion's contract."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return int(self._process.pid)

    def poll(self) -> Optional[int]:
        return self._process.poll()

    def terminate(self) -> None:
        # Signal the whole process group; the companion has already run the
        # independent direct-disable before calling this.
        if self._process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self._process.terminate()
            except OSError:
                pass


def make_subprocess_launcher(
    command: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> Callable[[], SubprocessLaunchedController]:
    """Return a launch_controller() that starts ``command`` in a new session."""

    argv = _validate_command(command, label="teleop launch command")

    def launch() -> SubprocessLaunchedController:
        try:
            process = popen(
                list(argv),
                cwd=cwd,
                env=env,
                start_new_session=True,  # own process group -> group-signalable
                stdin=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise ControllerLaunchError(f"teleop launch failed: {exc}") from exc
        return SubprocessLaunchedController(process)

    return launch


@dataclass(frozen=True)
class DisableResult:
    outcome: str
    returncode: int


def make_subprocess_direct_disable(
    command: Sequence[str],
    *,
    timeout_s: float = 5.0,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Callable[[str], DisableResult]:
    """Return a direct_disable(reason) that runs the energy-isolation command."""

    argv = _validate_command(command, label="controller disable command")

    def direct_disable(reason: str) -> DisableResult:
        try:
            completed = run(
                list(argv),
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.TimeoutExpired as exc:
            raise ControllerLaunchError(
                f"direct-disable command timed out after {timeout_s:.1f}s"
            ) from exc
        except OSError as exc:
            raise ControllerLaunchError(f"direct-disable command failed: {exc}") from exc
        code = int(getattr(completed, "returncode", 1) or 0)
        if code != 0:
            raise ControllerLaunchError(f"direct-disable command exited {code}")
        # P0-2: the disable command running is NOT proof the motors are de-energized.
        # Report only that the command succeeded; energy isolation is claimed elsewhere
        # and only after an independent motor-disable readback.
        return DisableResult(outcome="disable_command_succeeded", returncode=code)

    return direct_disable


def parse_command(text: object, *, label: str) -> tuple:
    """Parse a shell-style command string into an argv tuple."""
    if isinstance(text, (list, tuple)):
        return _validate_command(text, label=label)
    if not isinstance(text, str) or not text.strip():
        raise ControllerLaunchError(f"{label} is empty")
    return _validate_command(shlex.split(text), label=label)


__all__ = [
    "ControllerLaunchError",
    "DisableResult",
    "SubprocessLaunchedController",
    "make_subprocess_direct_disable",
    "make_subprocess_launcher",
    "parse_command",
]
