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
import time
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

    def terminate(self, *, sigterm_grace_s: float = 2.0) -> None:
        # Robust kill: SIGTERM the process group, wait a bounded grace, then SIGKILL and
        # wait again. A controller that ignores SIGTERM must NEVER survive teardown -- a
        # surviving controller can keep driving the arm. Best-effort direct-disable is run
        # by the caller; this guarantees the commanding process is actually gone.
        if self._process.poll() is not None:
            return
        self._signal_group(signal.SIGTERM)
        try:
            self._process.wait(timeout=sigterm_grace_s)
            return
        except subprocess.TimeoutExpired:
            pass
        self._signal_group(signal.SIGKILL)
        try:
            self._process.wait(timeout=sigterm_grace_s)
        except subprocess.TimeoutExpired:
            pass

    def _signal_group(self, sig) -> None:
        try:
            os.killpg(os.getpgid(self._process.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self._process.send_signal(sig)
            except OSError:
                pass


CONTROLLER_RT_PRIORITY = 50
CONTROLLER_CPU_AFFINITY = (28, 29, 30, 31)


def reap_stale_controller_processes(
    binary_path, *, sigterm_grace_s: float = 2.0,
) -> tuple:
    """Kill lingering controller processes from previous collector instances.

    The collector unit runs KillMode=mixed with a one-hour stop timeout BY
    DESIGN: a service restart must never hard-kill a controller that may be
    holding a powered arm (gravity compensation would vanish and the arm
    would drop). The cost is orphan accumulation: 2026-08-29 four concurrent
    controllers fought over can1/can3 and latched the telemetry receiver
    with tick-gap/sequence faults.

    This reap runs ONLY on the bring-up launch path, whose precondition is
    an operator-disabled arm (motors torque-off), so any lingering
    controller is inert software holding sockets — never a live commander.
    Matching is by resolved /proc/<pid>/exe against the sealed controller
    binary, which can never touch the other arm's (differently released)
    controller.
    """
    target = os.path.realpath(os.fspath(binary_path))
    reaped = []
    if not os.path.isdir("/proc"):
        return tuple(reaped)  # Linux-only mechanism; inert elsewhere (tests)
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == os.getpid():
            continue
        try:
            exe = os.path.realpath("/proc/%d/exe" % pid)
        except OSError:
            continue
        if exe != target:
            continue
        reaped.append(pid)
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                continue
    if reaped:
        deadline = time.monotonic() + sigterm_grace_s
        while time.monotonic() < deadline:
            if not any(os.path.isdir("/proc/%d" % pid) for pid in reaped):
                break
            time.sleep(0.05)
        for pid in reaped:
            if os.path.isdir("/proc/%d" % pid):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
    return tuple(reaped)


def apply_controller_scheduling(
    pid: int,
    *,
    rt_priority: int = CONTROLLER_RT_PRIORITY,
    cpu_affinity: tuple = CONTROLLER_CPU_AFFINITY,
) -> dict:
    """Pin and RT-schedule a just-launched controller before it threads out.

    Called immediately after fork, before the exec'd binary spawns worker
    threads: CPU affinity and the SCHED_FIFO policy are both inherited by
    every later thread, so one early call replaces the racy external
    pin-after-birth sentinel (an unpinned controller wobbled its 1 kHz tick
    stream to 9.5 ms gaps and latched the receiver on 2026-08-29).

    SCHED_FIFO needs RLIMIT_RTPRIO >= rt_priority (unit: LimitRTPRIO); the
    failure is reported, never raised — a running non-RT controller is
    strictly better than no controller, and the external sentinel remains a
    fallback.
    """
    outcome = {"affinity": None, "rt": None}
    try:
        affinity = tuple(cpu for cpu in cpu_affinity if cpu < os.cpu_count())
        if affinity:
            os.sched_setaffinity(pid, affinity)
            outcome["affinity"] = affinity
    except (OSError, AttributeError) as exc:
        outcome["affinity"] = f"failed: {exc}"
    try:
        os.sched_setscheduler(
            pid, os.SCHED_FIFO, os.sched_param(int(rt_priority))
        )
        outcome["rt"] = int(rt_priority)
    except (OSError, AttributeError) as exc:
        outcome["rt"] = f"failed: {exc}"
    return outcome


def make_subprocess_launcher(
    command: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    pre_launch_validate: Optional[Callable[[], None]] = None,
    reap_stale: bool = True,
    scheduling: bool = True,
) -> Callable[[], SubprocessLaunchedController]:
    """Return a launch_controller() that starts ``command`` in a new session."""

    argv = _validate_command(command, label="teleop launch command")
    frozen_cwd = None if cwd is None else os.fspath(cwd)
    frozen_env = None if env is None else dict(env)

    def launch() -> SubprocessLaunchedController:
        if pre_launch_validate is not None:
            pre_launch_validate()
        if reap_stale:
            reaped = reap_stale_controller_processes(argv[0])
            if reaped:
                print(
                    "[controller-launch] reaped stale controller processes "
                    f"(operator-disabled arm): {list(reaped)}",
                    flush=True,
                )
        try:
            process = popen(
                list(argv),
                cwd=frozen_cwd,
                env=None if frozen_env is None else dict(frozen_env),
                start_new_session=True,  # own process group -> group-signalable
                stdin=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise ControllerLaunchError(f"teleop launch failed: {exc}") from exc
        if scheduling:
            outcome = apply_controller_scheduling(process.pid)
            print(
                f"[controller-launch] scheduling pid={process.pid} "
                f"affinity={outcome['affinity']} rt={outcome['rt']}",
                flush=True,
            )
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


def make_subprocess_preflight(
    command: Sequence[str],
    *,
    timeout_s: float = 30.0,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Callable[[], None]:
    """Return a preflight() that runs a synchronous setup command (e.g. clear_faults +
    telemetry-socket-dir creation) BEFORE the controller is launched.

    P0-1: doing clear_faults here rather than inside a shell wrapper lets the controller
    binary be launched directly, so its ``/proc/pid/exe`` attestation is sound. Any
    failure raises so bring-up refuses rather than launching onto an unprepared bus.
    """
    argv = _validate_command(command, label="controller preflight command")

    def preflight() -> None:
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
                f"controller preflight timed out after {timeout_s:.1f}s"
            ) from exc
        except OSError as exc:
            raise ControllerLaunchError(f"controller preflight failed: {exc}") from exc
        code = int(getattr(completed, "returncode", 1) or 0)
        if code != 0:
            raise ControllerLaunchError(f"controller preflight exited {code}")

    return preflight


def parse_command(text: object, *, label: str) -> tuple:
    """Parse a shell-style command string into an argv tuple."""
    if isinstance(text, (list, tuple)):
        return _validate_command(text, label=label)
    if not isinstance(text, str) or not text.strip():
        raise ControllerLaunchError(f"{label} is empty")
    return _validate_command(shlex.split(text), label=label)


def build_launch_from_activation(
    activation,
    *,
    disable_command,
    preflight_command=None,
    disable_timeout_s: float = 5.0,
    preflight_timeout_s: float = 30.0,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
):
    """Construct the trusted (launcher, preflight, disable) from a validated dormant
    ``ControllerLaunchActivation`` for a safe-stop HIL.

    The controller is the SEALED release binary: ``argv[0]`` (a release-relative path) is
    resolved to the release's attested absolute ELF, and it runs with the sealed cwd and
    environment -- so the companion's exe attestation binds the exact sealed controller.
    The preflight defaults to the sealed preflight binary. The stop-tool (disable) is held
    at ``HIL_PENDING`` in the release, so ``disable_command`` is the CANDIDATE the safe-stop
    HIL validates by operator witness (never trusted as energy isolation on its own).
    """
    argv = [str(part) for part in activation.argv]
    if not argv:
        raise ControllerLaunchError("controller launch activation has empty argv")
    # argv[0] is the release-relative controller path; bind it to the attested absolute ELF.
    argv[0] = os.fspath(activation.controller_path)
    launcher = make_subprocess_launcher(
        tuple(argv),
        cwd=os.fspath(activation.cwd),
        env=dict(activation.environment),
        popen=popen,
    )
    if preflight_command is None:
        preflight_command = (os.fspath(activation.preflight_path),)
    preflight = make_subprocess_preflight(
        tuple(str(part) for part in preflight_command),
        timeout_s=preflight_timeout_s,
        run=run,
    )
    disable = make_subprocess_direct_disable(
        tuple(str(part) for part in disable_command),
        timeout_s=disable_timeout_s,
        run=run,
    )
    return launcher, preflight, disable


__all__ = [
    "ControllerLaunchError",
    "DisableResult",
    "SubprocessLaunchedController",
    "build_launch_from_activation",
    "make_subprocess_direct_disable",
    "make_subprocess_launcher",
    "make_subprocess_preflight",
    "parse_command",
]
