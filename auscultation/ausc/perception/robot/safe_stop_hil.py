"""Operator-witnessed safe-stop HIL gate for real-arm bring-up authorization.

The production controller-launch release ships DORMANT: the controller/preflight/urdf
are sealed and ready, but the stop-tool (disable) descriptor is held at ``HIL_PENDING``
and real-arm bring-up is NEVER authorized (``bringup_authorized = fake`` only). Before we
allow real motion for a recording, a human must PROVE, on this exact hardware and this
session, that the safe-stop actually de-energizes the arm.

This gate is that proof, as a strict state machine. Real bring-up becomes authorized ONLY
after the operator, in order, has:

  1. begin_test()        -- explicitly started a supervised safe-stop test (arm energizes);
  2. record_disable(...) -- the disable COMMAND was issued during the test and succeeded;
  3. witness(confirmed=True) -- the operator PHYSICALLY confirmed the arm went limp / lost
                                torque (the human-in-the-loop verification that stands in
                                for an automated per-motor torque-off readback we do not
                                yet have).

Any break in that sequence, a failed disable command, a "not de-energized" witness, or any
fault resets the gate and real bring-up stays refused. The pass is SESSION-BOUND: it lives
in memory and must be re-earned on every collector restart -- a stale safe-stop is never
trusted. This never authorizes autonomous or unattended motion; it only records that a
present operator validated the stop before a supervised recording.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional


class SafeStopHilError(RuntimeError):
    """A safe-stop HIL transition was requested out of order or with bad evidence."""


# States. Real bring-up is authorized ONLY in PASSED.
PENDING = "HIL_PENDING"                      # no validation this session
TESTING = "HIL_TESTING"                      # test controller energized, awaiting disable
AWAITING_WITNESS = "HIL_AWAITING_WITNESS"    # disable succeeded, awaiting operator witness
PASSED = "HIL_PASSED"                        # operator witnessed de-energize; bring-up ok
FAILED = "HIL_FAILED"                        # test failed; must restart the test

_RESTARTABLE = frozenset({PENDING, FAILED, PASSED})


@dataclass(frozen=True)
class SafeStopHilAttestation:
    """Immutable record that a present operator validated the safe-stop this session."""

    release_id: str
    contract_sha256: str
    session_id: str
    test_controller_pid: Optional[int]
    disable_outcome: str
    disable_command_succeeded: bool
    witness_note: str
    passed_utc_ns: int


class SafeStopHilGate:
    """Strict operator-witnessed safe-stop gate; bring-up authorized only in PASSED."""

    def __init__(
        self,
        *,
        release_id: str,
        contract_sha256: str,
        session_id: str,
        wall_time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._release_id = str(release_id)
        self._contract_sha256 = str(contract_sha256)
        self._session_id = str(session_id)
        self._wall_time_ns = wall_time_ns
        self._state = PENDING
        self._attestation: Optional[SafeStopHilAttestation] = None
        # Carried from record_disable() into the attestation minted at witness time.
        self._pending_pid: Optional[int] = None
        self._pending_outcome: str = ""

    # -- introspection ----------------------------------------------------- #
    @property
    def state(self) -> str:
        return self._state

    @property
    def bringup_authorized(self) -> bool:
        """The single authority bit: real bring-up ok iff the safe-stop passed."""
        return self._state == PASSED and self._attestation is not None

    @property
    def attestation(self) -> Optional[SafeStopHilAttestation]:
        return self._attestation

    def status_snapshot(self) -> dict:
        att = self._attestation
        return {
            "state": self._state,
            "bringup_authorized": self.bringup_authorized,
            "release_id": self._release_id,
            "passed_utc_ns": None if att is None else att.passed_utc_ns,
            "witness_note": None if att is None else att.witness_note,
        }

    # -- transitions ------------------------------------------------------- #
    def begin_test(self) -> None:
        """Operator starts a supervised safe-stop test (the arm is about to energize).

        Starting a test VOIDS any prior pass: authorization must be re-earned by
        witnessing this test's disable. Refused while a test is already in flight.
        """
        if self._state not in _RESTARTABLE:
            raise SafeStopHilError(
                f"cannot begin a safe-stop test from state {self._state}"
            )
        self._state = TESTING
        self._attestation = None          # a new test invalidates the old authorization
        self._pending_pid = None
        self._pending_outcome = ""

    def record_disable(
        self, *, command_succeeded: bool, outcome: str, controller_pid: Optional[int],
    ) -> None:
        """The disable command was issued during the test. Advance only if it succeeded.

        A failed disable command means the safe-stop is NOT proven -- go to FAILED, never
        toward a pass.
        """
        if self._state != TESTING:
            raise SafeStopHilError(
                f"record_disable requires a test in progress, not {self._state}"
            )
        self._pending_pid = None if controller_pid is None else int(controller_pid)
        self._pending_outcome = str(outcome)
        if command_succeeded:
            self._state = AWAITING_WITNESS
        else:
            self._state = FAILED

    def witness(self, *, confirmed: bool, note: str = "") -> bool:
        """The operator's PHYSICAL verdict on whether the arm de-energized.

        ``confirmed=True`` (arm went limp / lost torque) is the only path to PASSED and it
        mints the attestation. ``confirmed=False`` means the disable did NOT de-energize the
        arm -> FAILED, bring-up stays refused. Returns whether the gate is now PASSED.
        """
        if self._state != AWAITING_WITNESS:
            raise SafeStopHilError(
                f"witness requires an awaited disable, not {self._state}"
            )
        if not confirmed:
            self._state = FAILED
            self._attestation = None
            return False
        self._attestation = SafeStopHilAttestation(
            release_id=self._release_id,
            contract_sha256=self._contract_sha256,
            session_id=self._session_id,
            test_controller_pid=self._pending_pid,
            disable_outcome=self._pending_outcome,
            disable_command_succeeded=True,
            witness_note=str(note),
            passed_utc_ns=int(self._wall_time_ns()),
        )
        self._state = PASSED
        return True

    def fault_reset(self, reason: str) -> None:
        """A fault (companion fault, collector crash, shutdown) voids the gate.

        Any prior pass is revoked: a fault during or after validation means the safe-stop
        must be re-proven before real bring-up is authorized again.
        """
        self._state = PENDING
        self._attestation = None
        self._pending_pid = None
        self._pending_outcome = ""


__all__ = [
    "SafeStopHilError",
    "SafeStopHilAttestation",
    "SafeStopHilGate",
    "PENDING",
    "TESTING",
    "AWAITING_WITNESS",
    "PASSED",
    "FAILED",
]
