#!/usr/bin/env bash
# WAM auscultation — safe-stop HIL disable CANDIDATE (design B, operator-witnessed).
#
# Validated each session by the operator-witnessed safe-stop HIL gate (see
# ausc/perception/robot/safe_stop_hil.py). SIGINTs the SEALED unilateral_control
# (argv[0] ".../controller/unilateral_control", LEFT arm can1/can3) so its clean
# shutdown handler disables the motors, then VERIFIES the process actually exited.
#
# HONESTY CONTRACT (P0 audit): success here proves ONLY "no sealed controller
# process is running". It is NOT an independent CAN-level energy isolation and has
# no per-motor torque-off readback (the sealed stop_tool remains HIL_PENDING).
# Whether the motors truly de-energized is decided by the OPERATOR WITNESS at the
# arm, with the hardware power switch / E-stop as the backstop.
set -uo pipefail

PATTERN="controller/unilateral_control"   # sealed path only; never the canary build

pkill -INT -f "${PATTERN}"
rc=$?
if [ "${rc}" -ge 2 ]; then
  echo "[safe-stop-disable] pkill ERROR rc=${rc} — controller NOT signalled" >&2
  exit "${rc}"
fi
if [ "${rc}" -eq 1 ]; then
  echo "[safe-stop-disable] no sealed controller matched — already stopped (nothing was signalled)"
  exit 0
fi

# Signalled a live controller: require it to actually exit (shutdown handler ran).
for _ in $(seq 1 40); do
  if ! pgrep -f "${PATTERN}" > /dev/null 2>&1; then
    echo "[safe-stop-disable] SIGINT delivered and sealed controller EXITED — operator must still witness de-energize"
    exit 0
  fi
  sleep 0.1
done
echo "[safe-stop-disable] controller STILL RUNNING 4s after SIGINT — treat as DISABLE FAILURE, use E-stop/power" >&2
exit 3
