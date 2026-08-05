#!/usr/bin/env bash
# WAM auscultation — safe-stop HIL disable CANDIDATE (design B, operator-witnessed).
#
# This is the disable command the operator-witnessed safe-stop HIL validates each
# session before real-arm bring-up is authorized (see ausc/perception/robot/safe_stop_hil.py).
# It SIGINTs the SEALED unilateral_control controller (launched from the sealed
# controller-launch release, argv[0] = ".../controller/unilateral_control") so its
# clean-shutdown handler disables the motors on can1/can3 (the LEFT auscultation arm).
#
# IMPORTANT: this is a TEST-GRADE stop, NOT a confirmed direct CAN-level energy
# isolation. The confirmed per-motor torque-off readback is the known HIL_PENDING gap
# (the sealed stop_tool descriptor is held at HIL_PENDING). Until that exists, the
# operator's physical witness that the arm de-energized, plus the hardware power
# switch / E-stop, are the real safety backstop. The safe-stop HIL gate refuses to
# authorize bring-up until a human confirms de-energization, every session.
set -uo pipefail

# Match ONLY the sealed controller path (".../controller/unilateral_control"), never
# the canary dev build (".../teleop_merge/build/unilateral_control").
pkill -INT -f "controller/unilateral_control"
rc=$?

# pkill exit: 0 = signalled a live controller, 1 = nothing matched (arm already not
# being driven). Both mean "no controller is driving the arm" -> report command
# success; whether the motors actually de-energized is decided by the operator
# witness, not by this script. Only a genuine pkill error (>=2) is a command failure.
echo "[safe-stop-disable] SIGINT -> sealed unilateral_control (pkill rc=${rc}); operator must witness de-energize"
if [ "${rc}" -le 1 ]; then
  exit 0
fi
exit "${rc}"
