"""Single-owner trusted launcher/supervisor for the left OpenArm controller.

The production robot-capture receiver (``ProductionRobotCaptureCoordinator`` +
``RobotSocketReceiver``) will only accept telemetry from a controller process
whose identity a *trusted launcher* pinned before control started.  This companion
is that launcher.  It owns exactly one controller lifecycle:

  1. prepare the read-only receiver socket (before any control exists);
  2. launch the pinned left controller (can1 leader -> can3 follower);
  3. independently capture the controller's process attestation (pid/uid/start
     time/executable identity) while holding a pidfd;
  4. attach the attestation and start the receiver;
  5. wait until the receiver reports READY with preflight_ready True -- i.e. every
     joint's per-motor RX/liveness, command-response and trajectory authenticity
     the receiver validates has passed;
  6. and on ANY failure -- bring-up, telemetry fault, force-unsafe, collector
     crash, or shutdown -- invoke the confirmed direct-disable, then stop the
     receiver and terminate the controller, energy-isolation first.

It never relaxes the robot gate to make Start green, and it is the single owner of
the telemetry socket, the controller process and the receiver: a second bring-up
is refused.  Every robot-facing dependency is injected, so the whole lifecycle is
exercised against a virtual-CAN / simulated-controller shadow with no real motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Callable, Optional, Protocol

# Ready only in this exact receiver state; every other state keeps waiting (during
# bring-up) or is a fault (after ready).
_READY_STATUS = "READY"
# Terminal-bad receiver states: never wait through these.
_TERMINAL_BAD_STATUS = frozenset({"FAULT", "STOPPED"})


class CompanionError(RuntimeError):
    """The trusted controller companion cannot proceed safely."""


class CompanionFault(CompanionError):
    """A safety fault occurred; the arm has been (or is being) direct-disabled."""


class LaunchedController(Protocol):
    @property
    def pid(self) -> int: ...
    def poll(self) -> Optional[int]: ...   # exit code, or None while alive
    def terminate(self) -> None: ...


class ReceiverCoordinator(Protocol):
    def prepare_receiver_socket(self) -> None: ...
    def attach_controller_process(self, attestation: object) -> None: ...
    def start_receiver(self) -> None: ...
    def stop_receiver(self) -> None: ...
    def receiver_health(self, *, now_monotonic_ns: Optional[int] = None) -> object: ...
    @property
    def faulted(self) -> bool: ...
    @property
    def fault_reason(self) -> str: ...


@dataclass(frozen=True)
class DisableOutcome:
    """The result of a fault-disable, split into an honest state ladder.

    ``command_succeeded`` means the disable *command* returned success; it is NOT
    proof the motors are de-energized.  ``motor_verified`` requires an independent
    readback that every motor is torque-off.  ``energy_isolated`` is derived and is
    claimed ONLY when ``motor_verified`` -- never from a mere command success -- so
    the safety surface can never falsely display "energy-isolated".
    """

    reason: str
    outcome: str
    command_succeeded: bool = False
    motor_verified: bool = False
    energy_isolated: bool = False


def _disable_command_succeeded(result: object) -> bool:
    """Whether a direct-disable result signals the command itself succeeded.

    Prefers an explicit ``returncode``/``command_succeeded``; falls back to treating a
    truthy legacy string outcome as success.  A command success is never, on its own,
    energy isolation -- that requires an independent motor readback.
    """
    if result is None:
        return False
    returncode = getattr(result, "returncode", None)
    if returncode is not None:
        return int(returncode) == 0
    command_succeeded = getattr(result, "command_succeeded", None)
    if command_succeeded is not None:
        return bool(command_succeeded)
    return bool(result)


def _default_attest_controller(pid: int, expected_uid: int) -> object:
    """Real attestation via a pidfd-guarded procfs inspection (Linux only)."""
    from ausc.perception.robot.robot_socket_receiver import (  # local import: Linux-only
        ControllerProcessAttestation,
        LinuxControllerProcessInspector,
    )

    return ControllerProcessAttestation.capture(
        pid=pid, expected_uid=expected_uid, inspector=LinuxControllerProcessInspector(),
    )


class TrustedControllerCompanion:
    """Own one controller lifecycle; fail-closed to direct-disable on any fault."""

    def __init__(
        self,
        *,
        coordinator: ReceiverCoordinator,
        launch_controller: Callable[[], LaunchedController],
        direct_disable: Callable[[str], object],
        preflight: Optional[Callable[[], object]] = None,
        expected_uid: Optional[int] = None,
        attest_controller: Optional[Callable[[int, int], object]] = None,
        verify_motor_disable: Optional[Callable[[], bool]] = None,
        readiness_timeout_s: float = 10.0,
        poll_interval_s: float = 0.05,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if readiness_timeout_s <= 0 or poll_interval_s <= 0:
            raise CompanionError("readiness timeout and poll interval must be positive")
        self._coordinator = coordinator
        self._launch_controller = launch_controller
        self._direct_disable = direct_disable
        # P0-1: synchronous preflight (clear_faults + socket dir) run BEFORE launch so
        # the launched process is the controller binary from t=0 (attestable exe).
        self._preflight = preflight
        self._expected_uid = os.getuid() if expected_uid is None else int(expected_uid)
        self._attest = attest_controller or _default_attest_controller
        # Optional independent readback: returns True only if every motor is confirmed
        # torque-off after the disable command.  Absent (None) -> energy isolation is
        # never claimed, only "disable command succeeded".
        self._verify_motor_disable = verify_motor_disable
        self._readiness_timeout_s = float(readiness_timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._monotonic = monotonic
        self._sleep = sleep
        self._state = "idle"           # idle -> ready -> faulted
        self._controller: Optional[LaunchedController] = None
        self._disabled = False
        self._disable_outcome: Optional[DisableOutcome] = None

    # -- introspection ------------------------------------------------------ #
    @property
    def state(self) -> str:
        return self._state

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def controller_pid(self) -> Optional[int]:
        return None if self._controller is None else int(self._controller.pid)

    @property
    def disable_outcome(self) -> Optional[DisableOutcome]:
        return self._disable_outcome

    # -- lifecycle ---------------------------------------------------------- #
    def bring_up(self) -> object:
        """Prepare socket, launch + attest controller, attach, and gate on READY.

        Returns the READY receiver health.  On ANY failure the arm is
        direct-disabled and a CompanionFault is raised.
        """
        if self._state != "idle":
            raise CompanionError(f"companion already {self._state}; single owner refuses re-entry")
        if self._preflight is not None:
            # Runs before anything is launched -- a failure means no controller exists,
            # so there is nothing to energy-isolate; fail fast without a disable.
            try:
                self._preflight()
            except BaseException as exc:  # noqa: BLE001 - surfaced as a bring-up refusal
                self._state = "faulted"
                raise CompanionError(
                    f"controller preflight failed: {type(exc).__name__}: {exc}"
                ) from exc
        # A concurrent fault (collector crash / E-stop) may have disabled this owner
        # during the preflight. NEVER launch a controller onto an already-disabled or
        # faulted owner -- that is the crash+preflight race that leaves a live controller.
        if self._disabled or self._state == "faulted":
            raise CompanionFault(
                "companion was disabled during preflight; refusing to launch a controller"
            )
        try:
            self._coordinator.prepare_receiver_socket()
            controller = self._launch_controller()
            self._controller = controller
            # If a fault raced in between launch and here, terminate the just-launched
            # controller and stay disabled rather than leaving it live under a dead owner.
            if self._disabled or self._state == "faulted":
                try:
                    controller.terminate()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
                raise CompanionFault(
                    "companion was disabled during launch; controller terminated"
                )
            pid = int(controller.pid)
            if pid <= 0:
                raise CompanionError("launched controller reported a non-positive pid")
            attestation = self._attest(pid, self._expected_uid)
            self._coordinator.attach_controller_process(attestation)
            self._coordinator.start_receiver()
            health = self._await_ready()
            self._state = "ready"
            return health
        except CompanionFault:
            raise
        except BaseException as exc:
            self._fault_disable(f"controller bring-up failed: {type(exc).__name__}: {exc}")
            raise CompanionFault(str(exc)) from exc

    def _await_ready(self) -> object:
        deadline = self._monotonic() + self._readiness_timeout_s
        while True:
            # A concurrent fault (collector crash / E-stop) that disabled this owner must
            # interrupt the wait immediately -- never keep waiting for READY on a
            # controller that is already being torn down.
            if self._disabled:
                raise CompanionFault("companion disabled while awaiting readiness")
            self._assert_controller_alive()
            health = self._coordinator.receiver_health()
            status = str(getattr(health, "status", "FAULT"))
            if self._coordinator.faulted or status in _TERMINAL_BAD_STATUS:
                raise CompanionError(
                    f"receiver fault before ready: status={status} "
                    f"reason={self._coordinator.fault_reason or getattr(health, 'error', '')}"
                )
            if status == _READY_STATUS and bool(getattr(health, "preflight_ready", False)):
                return health
            if self._monotonic() >= deadline:
                raise CompanionError(
                    f"controller did not reach per-motor READY within "
                    f"{self._readiness_timeout_s:.1f}s (last status={status})"
                )
            self._sleep(self._poll_interval_s)

    def poll(self) -> object:
        """Ongoing supervision tick after ready; fault-closed to direct-disable.

        Any regression from READY+preflight_ready, a receiver fault, or a
        controller exit direct-disables the arm and raises CompanionFault.
        """
        if self._state != "ready":
            raise CompanionError(f"cannot supervise from state {self._state}")
        try:
            self._assert_controller_alive()
        except CompanionError as exc:
            # P0-3: a controller exit during capture is a safety fault. Energy-isolate
            # first, then surface it as a CompanionFault (the same disable-first path as
            # a telemetry fault) -- never a bare CompanionError that would leave the arm
            # live because the collector only reacts to CompanionFault.
            self._fault_disable(str(exc))
            raise CompanionFault(str(exc)) from exc
        health = self._coordinator.receiver_health()
        status = str(getattr(health, "status", "FAULT"))
        if (
            self._coordinator.faulted
            or status != _READY_STATUS
            or not bool(getattr(health, "preflight_ready", False))
        ):
            reason = (
                f"robot telemetry fault during capture: status={status} "
                f"reason={self._coordinator.fault_reason or getattr(health, 'error', '')}"
            )
            self._fault_disable(reason)
            raise CompanionFault(reason)
        return health

    def report_external_fault(self, reason: str) -> None:
        """A force-unsafe / collector-crash signal from outside; disable now."""
        detail = f"external safety fault: {reason}"
        self._fault_disable(detail)
        raise CompanionFault(detail)

    def _assert_controller_alive(self) -> None:
        controller = self._controller
        if controller is None:
            raise CompanionError("no controller has been launched")
        code = controller.poll()
        if code is not None:
            raise CompanionError(f"controller process exited (code={code})")

    def shutdown(self, reason: str = "normal_shutdown") -> DisableOutcome:
        """Normal teardown: still energy-isolation first, then stop + terminate."""
        self._fault_disable(str(reason))
        assert self._disable_outcome is not None
        return self._disable_outcome

    def _fault_disable(self, reason: str) -> None:
        """Idempotent teardown: send the direct-disable command FIRST, then an
        independent motor-disable readback, then stop the receiver and terminate.

        Energy isolation (``energy_isolated``) is claimed ONLY when the readback
        confirms every motor is torque-off.  A mere command success is
        ``command_succeeded`` -- never ``energy_isolated`` -- so the safety surface can
        never falsely report the arm as de-energized.
        """
        if self._disabled:
            # Already SUCCESSFULLY disabled -> idempotent no-op (no double-disable).
            self._state = "faulted"
            return
        command_succeeded = False
        motor_verified = False
        outcome = "unknown"
        try:
            result = self._direct_disable(reason)
            command_succeeded = _disable_command_succeeded(result)
            outcome = str(getattr(result, "outcome", result) or "disable_requested")
        except BaseException as exc:  # noqa: BLE001 - disable must never mask itself
            outcome = f"direct_disable_error: {type(exc).__name__}: {exc}"
        else:
            # Independent readback while telemetry is still live.  Absent a verifier
            # (or if it raises/returns False), isolation is not claimed.
            if command_succeeded and self._verify_motor_disable is not None:
                try:
                    motor_verified = bool(self._verify_motor_disable())
                except BaseException:  # noqa: BLE001 - unverified, not isolated
                    motor_verified = False
        energy_isolated = bool(motor_verified)
        summary = (
            "energy_isolated" if energy_isolated
            else "disable_command_succeeded_motor_unverified"
            if command_succeeded else outcome
        )
        self._disable_outcome = DisableOutcome(
            reason=reason,
            outcome=summary,
            command_succeeded=command_succeeded,
            motor_verified=motor_verified,
            energy_isolated=energy_isolated,
        )
        # Mark disabled (block further retries) ONLY when the command succeeded, so a
        # FAILED disable stays retryable -- a first failure must never latch the arm as
        # "disabled" and silence subsequent E-stops.
        if command_succeeded:
            self._disabled = True
        try:
            self._coordinator.stop_receiver()
        except Exception:
            pass
        if self._controller is not None:
            try:
                self._controller.terminate()
            except Exception:
                pass
        self._state = "faulted"


__all__ = [
    "CompanionError",
    "CompanionFault",
    "DisableOutcome",
    "LaunchedController",
    "ReceiverCoordinator",
    "TrustedControllerCompanion",
]
