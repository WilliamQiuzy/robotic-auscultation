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
  6. and on any controller/runtime safety fault, collector crash, operator stop,
     or shutdown -- invoke the confirmed direct-disable, then stop the
     receiver and terminate the controller, energy-isolation first.

It never relaxes the robot gate to make Start green, and it is the single owner of
the telemetry socket, the controller process and the receiver: a second bring-up
is refused.  Every robot-facing dependency is injected, so the whole lifecycle is
exercised against a virtual-CAN / simulated-controller shadow with no real motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import Callable, Optional, Protocol

# Ready only in this exact receiver state; every other state keeps waiting (during
# bring-up) or is a fault (after ready).
_READY_STATUS = "READY"
# Terminal-bad receiver states: never wait through these.
_TERMINAL_BAD_STATUS = frozenset({"FAULT", "STOPPED"})
# Freshness (non-terminal) regressions must persist this many consecutive
# supervision polls before the arm is fault-disabled; one GIL-lagged health
# snapshot on a busy shared box is not a safety event.
_STALE_POLL_FAULT_LIMIT = 3


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


def execute_independent_stop(
    *,
    direct_disable: Callable[[str], object],
    verify_motor_disable: Optional[Callable[[], bool]],
    reason: str,
) -> DisableOutcome:
    """Issue a direct motor-disable and independently verify torque-off.

    This helper owns no controller process or receiver, so it is safe to use for
    an authorized idle-page stop where no ``TrustedControllerCompanion`` exists.
    A successful command is reported honestly as unverified unless the independent
    verifier returns the literal boolean ``True``.
    """

    reason = str(reason)
    command_succeeded = False
    motor_verified = False
    raw_outcome = "unknown"
    try:
        result = direct_disable(reason)
        command_succeeded = _disable_command_succeeded(result)
        raw_outcome = str(
            getattr(result, "outcome", result) or "disable_requested"
        )
    except BaseException as exc:  # noqa: BLE001 - stop must return honest evidence
        raw_outcome = f"direct_disable_error: {type(exc).__name__}: {exc}"
    else:
        if command_succeeded and verify_motor_disable is not None:
            try:
                motor_verified = verify_motor_disable() is True
            except BaseException:  # noqa: BLE001 - unverified, never isolated
                motor_verified = False

    energy_isolated = motor_verified is True
    summary = (
        "energy_isolated" if energy_isolated
        else "disable_command_succeeded_motor_unverified"
        if command_succeeded else raw_outcome
    )
    return DisableOutcome(
        reason=reason,
        outcome=summary,
        command_succeeded=command_succeeded,
        motor_verified=motor_verified,
        energy_isolated=energy_isolated,
    )


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
        self._fault_reason = ""        # first fault's reason, kept for evidence
        self._stale_polls = 0          # consecutive non-READY freshness polls
        self._controller: Optional[LaunchedController] = None
        self._disabled = False
        self._disable_outcome: Optional[DisableOutcome] = None
        # Cancellation token: set the INSTANT any disable/E-stop is requested, BEFORE the
        # (possibly slow or failing) disable I/O runs.  ``_disabled`` is only set once the
        # command CONFIRMS, so it cannot be used to interrupt a concurrent bring-up during
        # the disable window.  ``_stop_requested`` closes that race: a bring-up in flight
        # aborts the moment a stop is requested, so the arm can never reach READY after an
        # E-stop has begun.  Monotonic (never cleared) -- a companion is single-use.
        self._stop_requested = False
        # Serializes the disable critical section so a watchdog-poll fault and a console
        # E-stop cannot interleave their teardown of the SAME companion.
        self._disable_lock = threading.Lock()

    # -- introspection ------------------------------------------------------ #
    @property
    def state(self) -> str:
        return self._state

    @property
    def fault_reason(self) -> str:
        """The FIRST fault's reason, preserved through teardown for evidence."""
        return self._fault_reason

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

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
        # ``_stop_requested`` catches an E-stop whose disable I/O is still in flight
        # (``_disabled`` not yet set), which ``_disabled``/``faulted`` alone would miss.
        if self._stop_requested or self._disabled or self._state == "faulted":
            raise CompanionFault(
                "companion was disabled during preflight; refusing to launch a controller"
            )
        try:
            self._coordinator.prepare_receiver_socket()
            controller = self._launch_controller()
            self._controller = controller
            # If a fault raced in between launch and here, the just-launched controller may
            # already be driving the arm even though the earlier disable ran BEFORE this
            # controller existed. Kill it robustly FIRST (SIGTERM->SIGKILL->wait, so it can
            # no longer issue motor commands), THEN re-issue the direct-disable so the LAST
            # command the motors receive is torque-off. Never leave a live controller.
            if self._stop_requested or self._disabled or self._state == "faulted":
                try:
                    controller.terminate()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
                # A disable that raced *before* launch cannot prove this newly launched
                # process left the motors off.  Serialize with the ordinary stop path,
                # issue a fresh disable after terminating the late controller, and make
                # this final command carry its own independent torque-off evidence.  Do
                # not retain an older ``energy_isolated=True`` outcome for a command that
                # preceded the late launch.
                with self._disable_lock:
                    self._disable_outcome = execute_independent_stop(
                        direct_disable=self._direct_disable,
                        verify_motor_disable=self._verify_motor_disable,
                        reason="late controller terminated during E-stop",
                    )
                    verifier_configured = self._verify_motor_disable is not None
                    self._disabled = bool(
                        self._disable_outcome.command_succeeded
                        and (
                            self._disable_outcome.energy_isolated
                            or not verifier_configured
                        )
                    )
                self._state = "faulted"
                raise CompanionFault(
                    "companion was disabled during launch; controller terminated and re-disabled"
                )
            pid = int(controller.pid)
            if pid <= 0:
                raise CompanionError("launched controller reported a non-positive pid")
            attestation = self._attest(pid, self._expected_uid)
            self._coordinator.attach_controller_process(attestation)
            self._coordinator.start_receiver()
            health = self._await_ready()
            # Commit READY atomically with a FINAL stop-intent re-check, under the same
            # lock the disable path holds. Without this, a concurrent E-stop that set
            # _state="faulted" between _await_ready() returning and here would be
            # clobbered back to "ready" -- letting the arm report READY *after* an E-stop
            # began, defeating the _stop_requested cancellation token entirely.
            with self._disable_lock:
                if self._stop_requested or self._disabled:
                    raise CompanionFault(
                        "companion disabled before reaching ready"
                    )
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
            # controller that is already being torn down.  ``_stop_requested`` fires the
            # instant an E-stop begins, even before its (slow) disable command confirms,
            # so a bring-up can never win the race and return READY after an E-stop.
            if self._stop_requested or self._disabled:
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
        # INTEGRITY faults (a latched receiver fault or a terminal status) disable
        # immediately -- they carry a real reason and never self-heal.
        if self._coordinator.faulted or status in _TERMINAL_BAD_STATUS:
            reason = (
                f"robot telemetry fault during capture: status={status} "
                f"reason={self._coordinator.fault_reason or getattr(health, 'error', '')}"
            )
            self._fault_disable(reason)
            raise CompanionFault(reason)
        # FRESHNESS wobble (e.g. STALE: the last envelope is momentarily older than
        # the display bound) is an observation, not an integrity violation -- on a
        # busy shared box the GIL-bound receiver thread can lag for one poll while
        # the stream itself is intact (any real loss latches via the sequence/tick
        # checks above and takes the immediate path). Debounce: only a SUSTAINED
        # regression (consecutive polls) disables the arm.
        if status != _READY_STATUS or not bool(getattr(health, "preflight_ready", False)):
            self._stale_polls += 1
            if self._stale_polls >= _STALE_POLL_FAULT_LIMIT:
                reason = (
                    "robot telemetry fault during capture: status="
                    f"{status} sustained for {self._stale_polls} supervision polls "
                    f"reason={self._coordinator.fault_reason or getattr(health, 'error', '')}"
                )
                self._fault_disable(reason)
                raise CompanionFault(reason)
            return health
        self._stale_polls = 0
        return health

    def report_external_fault(self, reason: str) -> None:
        """A collector/runtime safety fault from outside; disable immediately.

        Physician-observed force is advisory and must not call this safety-fault
        path merely because it crosses a force-display threshold.
        """
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
        # Capture THIS call's outcome from inside the locked section, never by re-reading
        # the shared field afterwards -- a concurrent watchdog disable-retry could
        # overwrite self._disable_outcome between the disable and the read, making the
        # operator's E-stop response describe a different command's result.
        outcome = self._fault_disable(str(reason))
        assert outcome is not None
        return outcome

    def _fault_disable(self, reason: str) -> DisableOutcome:
        """Idempotent teardown: send the direct-disable command FIRST, then an
        independent motor-disable readback, then stop the receiver and terminate.

        Energy isolation (``energy_isolated``) is claimed ONLY when the readback
        confirms every motor is torque-off.  A mere command success is
        ``command_succeeded`` -- never ``energy_isolated`` -- so the safety surface can
        never falsely report the arm as de-energized.
        """
        # Publish stop intent BEFORE any (slow/failing) I/O so a concurrent bring-up
        # aborts immediately -- this is the cancellation token that closes the
        # "arm reaches READY while an E-stop disable is still running" race.
        self._stop_requested = True
        # Keep the FIRST fault's reason as evidence: post-READY auto-faults happen in
        # the watchdog with no operator action attached, so without this the reason
        # would be unrecoverable after teardown.
        if not self._fault_reason:
            self._fault_reason = str(reason)
        # Serialize teardown: a watchdog-poll fault and a console E-stop must not
        # interleave their disable of the SAME companion.  The outcome is captured and
        # returned from INSIDE the lock so the caller reports its own command's result.
        with self._disable_lock:
            return self._fault_disable_locked(reason)

    def _fault_disable_locked(self, reason: str) -> DisableOutcome:
        if self._disabled:
            # Already SUCCESSFULLY disabled -> idempotent no-op (no double-disable).
            self._state = "faulted"
            return self._disable_outcome
        self._disable_outcome = execute_independent_stop(
            direct_disable=self._direct_disable,
            verify_motor_disable=self._verify_motor_disable,
            reason=reason,
        )
        command_succeeded = self._disable_outcome.command_succeeded
        energy_isolated = self._disable_outcome.energy_isolated
        # Mark disabled (stop retries + let the arm be moved on from) ONLY when the arm is
        # confirmed SAFE -- not merely when the disable COMMAND was accepted:
        #   * a FAILED command stays retryable (never latch a live arm as "disabled");
        #   * a command that SUCCEEDED but whose independent readback says the motors are
        #     STILL energized (verifier present, energy_isolated False) ALSO stays
        #     unlatched, so the watchdog keeps re-issuing the disable and no
        #     replacement / second controller is ever allowed onto a still-live arm.
        # When NO verifier is configured, command success is the operating standard (the
        # UI shows energy isolation as UNVERIFIED, but the command is the best evidence).
        verifier_configured = self._verify_motor_disable is not None
        if command_succeeded and (energy_isolated or not verifier_configured):
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
        return self._disable_outcome


__all__ = [
    "CompanionError",
    "CompanionFault",
    "DisableOutcome",
    "LaunchedController",
    "ReceiverCoordinator",
    "TrustedControllerCompanion",
    "execute_independent_stop",
]
