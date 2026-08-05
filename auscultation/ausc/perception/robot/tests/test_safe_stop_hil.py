"""Safety-invariant tests for the operator-witnessed safe-stop HIL gate."""
from __future__ import annotations

import pytest

from ausc.perception.robot.safe_stop_hil import (
    AWAITING_WITNESS,
    FAILED,
    PASSED,
    PENDING,
    TESTING,
    SafeStopHilError,
    SafeStopHilGate,
)


def _gate(clock=lambda: 111):
    return SafeStopHilGate(
        release_id="controller_launch_v1-" + "a" * 64,
        contract_sha256="b" * 64,
        session_id="sess-1",
        wall_time_ns=clock,
    )


def _pass(gate):
    gate.begin_test()
    gate.record_disable(command_succeeded=True, outcome="disable_command_succeeded", controller_pid=999)
    return gate.witness(confirmed=True, note="arm went limp")


def test_starts_pending_and_unauthorized():
    gate = _gate()
    assert gate.state == PENDING
    assert gate.bringup_authorized is False
    assert gate.attestation is None


def test_full_witnessed_sequence_authorizes_and_attests():
    gate = _gate(clock=lambda: 123456789)
    assert _pass(gate) is True
    assert gate.state == PASSED
    assert gate.bringup_authorized is True
    att = gate.attestation
    assert att is not None
    assert att.disable_command_succeeded is True
    assert att.witness_note == "arm went limp"
    assert att.test_controller_pid == 999
    assert att.passed_utc_ns == 123456789


def test_unauthorized_in_every_non_passed_state():
    gate = _gate()
    assert gate.bringup_authorized is False            # PENDING
    gate.begin_test()
    assert gate.state == TESTING and gate.bringup_authorized is False
    gate.record_disable(command_succeeded=True, outcome="ok", controller_pid=1)
    assert gate.state == AWAITING_WITNESS and gate.bringup_authorized is False


def test_failed_disable_command_never_reaches_witness():
    gate = _gate()
    gate.begin_test()
    gate.record_disable(command_succeeded=False, outcome="direct_disable_error", controller_pid=1)
    assert gate.state == FAILED
    assert gate.bringup_authorized is False
    # cannot witness a failed test
    with pytest.raises(SafeStopHilError):
        gate.witness(confirmed=True)


def test_operator_says_not_deenergized_fails_and_refuses():
    gate = _gate()
    gate.begin_test()
    gate.record_disable(command_succeeded=True, outcome="ok", controller_pid=1)
    assert gate.witness(confirmed=False, note="arm still holding") is False
    assert gate.state == FAILED
    assert gate.bringup_authorized is False
    assert gate.attestation is None


def test_out_of_order_transitions_are_refused():
    gate = _gate()
    with pytest.raises(SafeStopHilError):        # disable before a test
        gate.record_disable(command_succeeded=True, outcome="ok", controller_pid=1)
    with pytest.raises(SafeStopHilError):        # witness before a disable
        gate.witness(confirmed=True)
    gate.begin_test()
    with pytest.raises(SafeStopHilError):        # witness before disable, mid-test
        gate.witness(confirmed=True)
    with pytest.raises(SafeStopHilError):        # nested test start
        gate.begin_test()


def test_new_test_voids_a_prior_pass_until_rewitnessed():
    gate = _gate()
    _pass(gate)
    assert gate.bringup_authorized is True
    gate.begin_test()                            # start a fresh test
    assert gate.state == TESTING
    assert gate.bringup_authorized is False      # old pass no longer authorizes
    assert gate.attestation is None


def test_fault_reset_revokes_a_pass():
    gate = _gate()
    _pass(gate)
    assert gate.bringup_authorized is True
    gate.fault_reset("companion faulted during recording")
    assert gate.state == PENDING
    assert gate.bringup_authorized is False
    assert gate.attestation is None


def test_can_retry_after_a_failed_test():
    gate = _gate()
    gate.begin_test()
    gate.record_disable(command_succeeded=False, outcome="err", controller_pid=1)
    assert gate.state == FAILED
    # a fresh test is allowed from FAILED, and can then pass
    assert _pass(gate) is True
    assert gate.bringup_authorized is True
