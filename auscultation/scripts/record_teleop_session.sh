#!/usr/bin/env bash
# Combined multimodal teleop recording — the one launcher for real IL data collection.
#
# Runs the SENSOR recorder (ausc/record_session.py: audio + Hall force + camera) AND the leader-
# follower TELEOP (bin/unilateral_control, which logs robot.csv = follower obs + leader action) into
# ONE session dir, timestamped on ONE clock. Both use CLOCK_MONOTONIC (C++ steady_clock == Python
# time.monotonic() on Linux), so scripts/session_to_lerobot*.py aligns robot.csv with the sensor
# streams. Run on linbox (has the arm + CAN + sensors).
#
#   STETHO_DIR=~/Desktop/stethoscope scripts/record_teleop_session.sh <left|right> <seconds> <site> [name]
#   # dry-rehearse without the arm:  TELEOP_BIN=scripts/mock_teleop.py  (writes a fake robot.csv)
#
# Env: STETHO_DIR (teleop dir, default ~/Desktop/stethoscope), WAM_PY (python for record_session,
#      default ~/envs/ausc/bin/python), WAM_SESSIONS (session root, default ~/wam_data/sessions),
#      TELEOP_BIN (override the teleop binary — e.g. a mock for rehearsal).
set -u

STETHO="${STETHO_DIR:-$HOME/Desktop/stethoscope}"
AUSC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # auscultation/
source "$STETHO/scripts/_common.sh"                             # BIN, URDF, URDF_FOLLOWER, can map, clear_faults
PY="${WAM_PY:-$HOME/envs/ausc/bin/python}"
TELEOP_BIN="${TELEOP_BIN:-$BIN/unilateral_control}"

side="${1:?usage: record_teleop_session.sh <left|right> <seconds> <site> [name]}"
secs="${2:?seconds}"; site="${3:?site: LUL|LLL|RUL|RLL|posterior-lung}"
name="${4:-teleop_$(date +%Y%m%d_%H%M%S)}"
SDIR="${WAM_SESSIONS:-$HOME/wam_data/sessions}/$name"
mkdir -p "$SDIR"
case "$side" in
    left)  leader=$left_leader;  follower=$left_follower;  arm=left_arm ;;
    right) leader=$right_leader; follower=$right_follower; arm=right_arm ;;
    *) echo "side must be left|right"; exit 1 ;;
esac

echo "[combined] $SDIR  (side=$side site=$site ${secs}s)  teleop=$(basename "$TELEOP_BIN")"
clear_faults "$leader" "$follower" 2>/dev/null || true

# 1) TELEOP in background, logging robot.csv (absolute-monotonic obs+action) into the session dir.
if [[ "$TELEOP_BIN" == *.py ]]; then
    REC_CSV="$SDIR/robot.csv" "$PY" "$TELEOP_BIN" "$secs" > "$SDIR/teleop.log" 2>&1 &
else
    REC_CSV="$SDIR/robot.csv" "$TELEOP_BIN" "$URDF" "$URDF_FOLLOWER" "$arm" "$leader" "$follower" \
        > "$SDIR/teleop.log" 2>&1 &
fi
TPID=$!
sleep 1.5                                                       # let teleop connect + start logging first
if ! kill -0 "$TPID" 2>/dev/null; then
    echo "[combined] teleop failed to start — see $SDIR/teleop.log"; tail -3 "$SDIR/teleop.log"; exit 1
fi

# 2) SENSORS for N seconds into the SAME session dir (same monotonic clock).
"$PY" "$AUSC/ausc/record_session.py" --session-dir "$SDIR" --seconds "$secs" --site "$site" \
    --camera-backend orbbec || true

# 3) stop teleop cleanly (INT -> disable_all; then hard-kill as backstop).
kill -INT "$TPID" 2>/dev/null; for _ in 1 2 3 4 5; do kill -0 "$TPID" 2>/dev/null || break; sleep 1; done
kill -9 "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null || true

echo "[combined] done -> $SDIR : $(ls "$SDIR" | tr '\n' ' ')"
echo "[combined] convert (on linbox lerobot env):"
echo "  ~/miniconda3/envs/lerobot/bin/python $AUSC/scripts/session_to_lerobot_v2.py $SDIR --repo-id ausc/$name --root <out>"
