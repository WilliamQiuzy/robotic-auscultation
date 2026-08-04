"""Shadow tests for the trusted OpenArm controller companion.

Every robot-facing dependency (controller launch, process attestation, receiver
coordinator, direct-disable) is a simulated double, so the full safety lifecycle
is exercised with no CAN bus and no real motion.  These stand in for the vcan
shadow: the receiver readiness is scripted the way a virtual controller would
drive it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pytest

from ausc.perception.robot.trusted_controller_companion import (
    CompanionError,
    CompanionFault,
    TrustedControllerCompanion,
)


@dataclass
class _Health:
    status: str
    preflight_ready: bool = False
    error: str = ""


class _FakeController:
    def __init__(self, pid=4321):
        self._pid = pid
        self._exit = None
        self.terminated = False

    @property
    def pid(self):
        return self._pid

    def poll(self):
        return self._exit

    def terminate(self):
        self.terminated = True

    def die(self, code=1):
        self._exit = code


class _FakeCoordinator:
    def __init__(self, health_script: List[_Health], faulted=False, fault_reason=""):
        self._script = list(health_script)
        self._faulted = faulted
        self._fault_reason = fault_reason
        self.events: List[str] = []

    def prepare_receiver_socket(self):
        self.events.append("prepare")

    def attach_controller_process(self, attestation):
        self.events.append(f"attach:{attestation}")

    def start_receiver(self):
        self.events.append("start")

    def stop_receiver(self):
        self.events.append("stop")

    def receiver_health(self, *, now_monotonic_ns=None):
        if len(self._script) > 1:
            return self._script.pop(0)
        return self._script[0]

    def set_faulted(self, reason):
        self._faulted = True
        self._fault_reason = reason

    @property
    def faulted(self):
        return self._faulted

    @property
    def fault_reason(self):
        return self._fault_reason


class _Disabler:
    def __init__(self):
        self.calls: List[str] = []
        self.event_log: Optional[List[str]] = None

    def bind_log(self, log):
        self.event_log = log

    def __call__(self, reason):
        self.calls.append(reason)
        if self.event_log is not None:
            self.event_log.append("disable")
        return "energy_isolated"


def _companion(coordinator, disabler=None, controller=None, **over):
    controller = controller or _FakeController()
    disabler = disabler or _Disabler()
    if getattr(disabler, "event_log", "unset") is None:
        disabler.bind_log(coordinator.events)
    kw = dict(
        coordinator=coordinator,
        launch_controller=lambda: controller,
        direct_disable=disabler,
        expected_uid=1000,
        attest_controller=lambda pid, uid: f"attest(pid={pid},uid={uid})",
        readiness_timeout_s=1.0,
        poll_interval_s=0.001,
        monotonic=_FakeClock(),
        sleep=lambda _s: None,
    )
    kw.update(over)
    return TrustedControllerCompanion(**kw), controller, disabler


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.01
        return self.t


# --------------------------------------------------------------------------- #

def test_bring_up_reaches_ready_without_disabling(tmp_path):
    coord = _FakeCoordinator([_Health("WARMING"), _Health("WARMING"), _Health("READY", True)])
    companion, controller, disabler = _companion(coord)
    health = companion.bring_up()
    assert health.status == "READY"
    assert companion.state == "ready"
    assert disabler.calls == []                      # never disabled on success
    assert coord.events[:3] == ["prepare", "attach:attest(pid=4321,uid=1000)", "start"]


def test_bring_up_fault_disables_and_raises():
    coord = _FakeCoordinator([_Health("WARMING"), _Health("FAULT", error="rx lost")])
    companion, controller, disabler = _companion(coord)
    with pytest.raises(CompanionFault):
        companion.bring_up()
    assert companion.disabled is True
    assert len(disabler.calls) == 1
    assert companion.state == "faulted"


def test_readiness_timeout_disables():
    coord = _FakeCoordinator([_Health("WARMING")])  # never becomes READY
    companion, controller, disabler = _companion(coord, readiness_timeout_s=0.05)
    with pytest.raises(CompanionFault):
        companion.bring_up()
    assert disabler.calls and companion.disabled is True


def test_controller_exit_during_bringup_disables():
    controller = _FakeController()
    coord = _FakeCoordinator([_Health("WARMING"), _Health("WARMING"), _Health("READY", True)])
    companion, _c, disabler = _companion(coord, controller=controller)
    controller.die(2)  # process exits before READY
    with pytest.raises(CompanionFault):
        companion.bring_up()
    assert disabler.calls and companion.disabled is True


def test_attestation_failure_disables():
    coord = _FakeCoordinator([_Health("READY", True)])

    def bad_attest(pid, uid):
        raise RuntimeError("pidfd exited during attestation")

    companion, _c, disabler = _companion(coord, attest_controller=bad_attest)
    with pytest.raises(CompanionFault):
        companion.bring_up()
    assert disabler.calls and "attach" not in "".join(coord.events)  # never attached a bad process


def test_energy_isolation_ordering_disable_before_stop():
    # direct-disable must be logged BEFORE stop_receiver on any teardown.
    coord = _FakeCoordinator([_Health("READY", True)])
    disabler = _Disabler()
    disabler.bind_log(coord.events)
    companion, _c, _d = _companion(coord, disabler=disabler)
    companion.bring_up()
    companion.shutdown("normal_shutdown")
    assert "disable" in coord.events and "stop" in coord.events
    assert coord.events.index("disable") < coord.events.index("stop")


def test_poll_regression_after_ready_disables():
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, _c, disabler = _companion(coord)
    companion.bring_up()
    assert companion.poll().status == "READY"        # still healthy
    coord._script = [_Health("STALE", False)]        # telemetry goes stale
    with pytest.raises(CompanionFault):
        companion.poll()
    assert disabler.calls and companion.disabled is True


def test_controller_exit_during_poll_disables():
    # P0-3: a controller crash AFTER ready (during supervision) must fault-close the
    # arm disable-first, not leak a bare CompanionError that leaves the arm live.
    controller = _FakeController()
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, _c, disabler = _companion(coord, controller=controller)
    companion.bring_up()
    assert companion.poll().status == "READY"        # healthy while alive
    controller.die(9)                                # controller crashes mid-capture
    with pytest.raises(CompanionFault):
        companion.poll()
    assert disabler.calls and companion.disabled is True
    assert companion.state == "faulted"


def test_external_fault_disables():
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, _c, disabler = _companion(coord)
    companion.bring_up()
    with pytest.raises(CompanionFault, match="force unsafe"):
        companion.report_external_fault("force unsafe")
    assert disabler.calls and companion.disabled is True


def test_shutdown_always_direct_disables():
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, controller, disabler = _companion(coord)
    companion.bring_up()
    outcome = companion.shutdown("operator_stop")
    assert disabler.calls == ["operator_stop"]
    assert outcome.command_succeeded is True     # the disable command ran
    assert controller.terminated is True


def test_disable_command_success_is_not_energy_isolated_without_readback():
    # P0-2: a successful disable COMMAND is not proof of energy isolation. With no
    # independent motor-disable readback, energy_isolated MUST stay False (the UI may
    # not claim "energy-isolated").
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, _c, _d = _companion(coord)  # default disabler = command success, no readback
    companion.bring_up()
    outcome = companion.shutdown("operator_stop")
    assert outcome.command_succeeded is True
    assert outcome.motor_verified is False
    assert outcome.energy_isolated is False


def test_disable_with_verified_motor_readback_is_energy_isolated():
    # P0-2: energy isolation is claimed ONLY when the independent readback confirms all
    # motors are torque-off.
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, _c, _d = _companion(coord, verify_motor_disable=lambda: True)
    companion.bring_up()
    outcome = companion.shutdown("operator_stop")
    assert outcome.command_succeeded is True
    assert outcome.motor_verified is True
    assert outcome.energy_isolated is True


def test_disable_command_failure_is_not_command_succeeded():
    # P0-2: if the disable command raises, command_succeeded and energy_isolated are
    # both False and the error is captured verbatim, never masked as isolated.
    coord = _FakeCoordinator([_Health("READY", True)])

    def raising_disable(reason):
        raise RuntimeError("CAN disable send failed")

    companion, _c, _d = _companion(coord, disabler=raising_disable)
    companion.bring_up()
    outcome = companion.shutdown("operator_stop")
    assert outcome.command_succeeded is False
    assert outcome.motor_verified is False
    assert outcome.energy_isolated is False
    assert "direct_disable_error" in outcome.outcome


def test_command_success_with_failing_readback_does_not_latch_disabled():
    # SAFETY: a disable command that SUCCEEDS but whose independent readback says the
    # motors are STILL energized must NOT latch `_disabled` -- otherwise the watchdog
    # stops retrying and a replacement/second controller is allowed onto a live arm.
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, _c, _d = _companion(coord, verify_motor_disable=lambda: False)
    companion.bring_up()
    outcome = companion.shutdown("estop")
    assert outcome.command_succeeded is True     # the command itself was accepted
    assert outcome.energy_isolated is False      # but motors are confirmed still on
    assert companion.disabled is False           # so NOT latched -> keep retrying/refuse


def test_verified_readback_latches_disabled():
    # The positive of the above: a readback confirming torque-off latches disabled.
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, _c, _d = _companion(coord, verify_motor_disable=lambda: True)
    companion.bring_up()
    companion.shutdown("estop")
    assert companion.disabled is True
    assert companion.disable_outcome.energy_isolated is True


def test_no_verifier_uses_command_success_as_operating_standard():
    # With NO readback configured, command success remains the operating standard (energy
    # isolation is displayed UNVERIFIED but the arm may be moved on from).
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, _c, _d = _companion(coord)          # no verify_motor_disable
    companion.bring_up()
    companion.shutdown("estop")
    assert companion.disabled is True
    assert companion.disable_outcome.energy_isolated is False


def test_late_controller_during_stop_is_terminated_and_re_disabled():
    # If a stop races in during launch, the just-launched controller is robustly
    # terminated AND the direct-disable is re-issued, so a late controller can never be
    # left driving the arm and the last motor command is torque-off.
    coord = _FakeCoordinator([_Health("READY", True)])
    controller = _FakeController()

    def racing_launch():
        companion._stop_requested = True   # E-stop begins exactly as the controller starts
        return controller

    companion, _c, disabler = _companion(
        coord, controller=controller, launch_controller=racing_launch,
    )
    with pytest.raises(CompanionFault, match="terminated and re-disabled"):
        companion.bring_up()
    assert controller.terminated is True                        # process killed
    assert any("late controller" in reason for reason in disabler.calls)  # re-disable issued


def test_preflight_runs_synchronously_before_launch():
    # P0-1: clear_faults (etc.) runs as a synchronous preflight BEFORE the controller
    # is launched, so the launched process is the controller binary from t=0 (its
    # /proc/pid/exe attestation is sound -- a shell wrapper that clear_faults-then-execs
    # would be attested as the shell).
    coord = _FakeCoordinator([_Health("READY", True)])
    order = []
    controller = _FakeController()
    companion, _c, _d = _companion(
        coord,
        preflight=lambda: order.append("preflight"),
        launch_controller=lambda: (order.append("launch"), controller)[1],
        controller=controller,
    )
    companion.bring_up()
    assert order == ["preflight", "launch"]


def test_preflight_failure_aborts_before_launch_without_disable():
    # P0-1: if the preflight fails, nothing is launched and no disable is issued
    # (there is no live controller to energy-isolate).
    coord = _FakeCoordinator([_Health("READY", True)])
    launched = []

    def failing_preflight():
        raise RuntimeError("clear_faults failed")

    companion, _c, disabler = _companion(
        coord,
        preflight=failing_preflight,
        launch_controller=lambda: launched.append("x") or _FakeController(),
    )
    with pytest.raises(CompanionError, match="preflight failed"):
        companion.bring_up()
    assert launched == []
    assert disabler.calls == []


def test_failed_disable_is_retryable_and_never_latches_disabled():
    # Safety: a first disable-command FAILURE must not latch the arm as "disabled" and
    # silence subsequent E-stops -- the disable stays retryable until it succeeds.
    coord = _FakeCoordinator([_Health("READY", True)])
    attempts = {"n": 0}

    def flaky_disable(reason):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("CAN disable send failed")
        return "energy_isolated"

    companion, _c, _d = _companion(coord, disabler=flaky_disable)
    companion.bring_up()
    companion.shutdown("estop-1")                    # command fails
    assert companion.disable_outcome.command_succeeded is False
    assert companion.disabled is False               # NOT latched
    companion.shutdown("estop-2")                    # retry succeeds
    assert companion.disable_outcome.command_succeeded is True
    assert companion.disabled is True
    assert attempts["n"] == 2


def test_bring_up_refuses_launch_if_disabled_during_preflight():
    # Safety (crash+preflight race): a concurrent fault (collector crash / E-stop) that
    # disables the owner during preflight must abort bring-up -- NEVER launch a controller
    # onto an already-disabled owner (which would leave a live controller behind).
    coord = _FakeCoordinator([_Health("READY", True)])
    launched = []

    def preflight():
        companion._disabled = True  # simulate a concurrent disable during preflight

    companion, _c, _d = _companion(
        coord,
        preflight=preflight,
        launch_controller=lambda: launched.append("x") or _FakeController(),
    )
    with pytest.raises(CompanionFault, match="disabled during preflight"):
        companion.bring_up()
    assert launched == []            # never launched onto a disabled owner


def test_await_ready_aborts_when_disabled_concurrently():
    # A concurrent E-stop during the readiness wait interrupts it immediately, rather
    # than blocking until the readiness timeout.
    coord = _FakeCoordinator([_Health("WARMING")])
    orig_health = coord.receiver_health
    calls = {"n": 0}

    def health(**k):
        calls["n"] += 1
        if calls["n"] >= 2:
            companion._disabled = True   # concurrent disable mid-wait
        return orig_health(**k)

    coord.receiver_health = health
    companion, _c, _d = _companion(coord, readiness_timeout_s=5.0)
    with pytest.raises(CompanionFault, match="disabled while awaiting"):
        companion.bring_up()


def test_single_owner_refuses_second_bring_up():
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, _c, _d = _companion(coord)
    companion.bring_up()
    with pytest.raises(CompanionError, match="single owner"):
        companion.bring_up()


def test_disable_is_idempotent():
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, _c, disabler = _companion(coord)
    companion.bring_up()
    companion.shutdown("first")
    companion.shutdown("second")  # already disabled
    assert disabler.calls == ["first"]  # direct-disable invoked exactly once


def test_await_ready_aborts_on_stop_requested_before_disable_confirms():
    # E-stop-vs-bring-up race: a SLOW/failing E-stop disable publishes stop intent
    # (_stop_requested) BEFORE its command confirms (_disabled still False). A bring-up
    # mid-readiness-wait must abort on the stop intent ALONE -- otherwise the arm could
    # reach READY while an E-stop's disable is still in flight.
    coord = _FakeCoordinator([_Health("WARMING")])
    orig_health = coord.receiver_health
    calls = {"n": 0}

    def health(**k):
        calls["n"] += 1
        if calls["n"] >= 2:
            companion._stop_requested = True   # E-stop begun; disable NOT yet confirmed
        return orig_health(**k)

    coord.receiver_health = health
    companion, _c, _d = _companion(coord, readiness_timeout_s=5.0)
    with pytest.raises(CompanionFault, match="disabled while awaiting"):
        companion.bring_up()
    assert companion.disabled is False   # command never confirmed, yet bring-up aborted


def test_bring_up_refuses_launch_on_stop_requested_during_preflight():
    # Same race at the preflight gate: an in-flight E-stop (stop intent published, disable
    # not yet confirmed) during preflight must still refuse to launch a controller.
    coord = _FakeCoordinator([_Health("READY", True)])
    launched = []

    def preflight():
        companion._stop_requested = True   # E-stop begun, command not confirmed

    companion, _c, _d = _companion(
        coord,
        preflight=preflight,
        launch_controller=lambda: launched.append("x") or _FakeController(),
    )
    with pytest.raises(CompanionFault, match="disabled during preflight"):
        companion.bring_up()
    assert launched == []                # never launched while an E-stop was in flight


def test_shutdown_publishes_stop_requested_even_when_command_fails():
    # The cancellation token must be published even when the disable command itself fails,
    # so a concurrent bring-up still aborts (and the failed disable stays retryable).
    coord = _FakeCoordinator([_Health("READY", True)])

    def failing(reason):
        raise RuntimeError("CAN disable send failed")

    companion, _c, _d = _companion(coord, disabler=failing)
    companion.bring_up()
    companion.shutdown("estop")
    assert companion.stop_requested is True   # intent published despite the failure
    assert companion.disabled is False        # command failed -> not latched, retryable


def test_bring_up_ready_commit_rechecks_stop_before_committing():
    # The FINAL READY commit must re-check the stop token under the disable lock: even if
    # readiness is observed, a stop requested in that instant (E-stop begun, disable not
    # yet confirmed) must abort the commit -- the arm can never report READY after an
    # E-stop began, so bring_up() must not clobber a concurrent "faulted" back to "ready".
    coord = _FakeCoordinator([_Health("READY", True)])
    orig_health = coord.receiver_health

    def health(**k):
        h = orig_health(**k)
        if h.status == "READY":
            companion._stop_requested = True   # E-stop begins exactly as READY is seen
        return h

    coord.receiver_health = health
    companion, _c, _d = _companion(coord)
    with pytest.raises(CompanionFault, match="before reaching ready"):
        companion.bring_up()
    assert companion.state != "ready"          # never committed READY after the stop


def test_shutdown_returns_its_own_disable_outcome():
    # shutdown() returns the outcome captured INSIDE the lock for THIS call, so a
    # concurrent watchdog retry cannot make it report a different command's result.
    coord = _FakeCoordinator([_Health("READY", True)])
    companion, _c, _d = _companion(coord)
    companion.bring_up()
    outcome = companion.shutdown("estop-A")
    assert outcome is not None
    assert outcome.reason == "estop-A"
    assert outcome.command_succeeded is True
