#!/usr/bin/env bash
# Sensor-first, fail-closed launcher for real imitation-learning collection.
#
# Required commissioning inputs:
#   SYNC_CAL=/path/wam_sync_v1.json
#   CAMERA_LATENCY_CAL=/path/wam_camera_latency_v1.json
#   CAMERA_ID=<stable camera serial/id used by the calibration>
#   AUDIO_DEVICE/AUDIO_CHANNEL may override the PortAudio query/channel used by sync calibration
#   TARGETS_FILE=/path/robot_base_targets.jsonl
#   TELEOP_STOP_CMD_JSON='["/path/direct_disable", "arg1", "arg2"]'
#     This command starts immediately on every unsafe force event. Collection has no automatic
#     retract; commission disable latency and phantom force relief before patient-adjacent use.
#
# Usage: record_teleop_session.sh <left|right> <seconds> <site> [session-name]
set -euo pipefail

STETHO="${STETHO_DIR:-$HOME/Desktop/stethoscope}"
AUSC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$STETHO/scripts/_common.sh"

PY="${WAM_PY:-$HOME/envs/ausc/bin/python}"
TELEOP_BIN="${TELEOP_BIN:-$BIN/unilateral_control}"
side="${1:?usage: record_teleop_session.sh <left|right> <seconds> <site> [name]}"
secs="${2:?seconds}"
site="${3:?site}"
name="${4:-teleop_$(date -u +%Y%m%dT%H%M%S)_$(cat /proc/sys/kernel/random/uuid)}"
[[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
    || { echo "session name must be a single safe path component" >&2; exit 2; }
[[ "$name" != *..* ]] || { echo "session name may not contain '..'" >&2; exit 2; }
SDIR="${WAM_SESSIONS:-$HOME/wam_data/sessions}/$name"

SYNC_CAL="${SYNC_CAL:?SYNC_CAL is required before arm-connected collection}"
CAMERA_LATENCY_CAL="${CAMERA_LATENCY_CAL:?CAMERA_LATENCY_CAL is required}"
CAMERA_ID="${CAMERA_ID:?stable CAMERA_ID used by the latency calibration is required}"
TARGETS_FILE="${TARGETS_FILE:?TARGETS_FILE with explicit robot_base xyz is required}"
TELEOP_STOP_CMD_JSON="${TELEOP_STOP_CMD_JSON:?direct motor-disable command JSON is required}"
CODE_RELEASE_ID="${WAM_CODE_RELEASE_ID:?immutable WAM_CODE_RELEASE_ID is required}"
RUNTIME_RELEASE_ID="${WAM_PERCEPTION_RUNTIME_RELEASE_ID:?immutable WAM_PERCEPTION_RUNTIME_RELEASE_ID is required}"
RUNTIME_LOCK_SHA256="${WAM_PERCEPTION_RUNTIME_LOCK_SHA256:?WAM_PERCEPTION_RUNTIME_LOCK_SHA256 is required}"
[[ "$CODE_RELEASE_ID" =~ ^code_release-[0-9a-f]{20}$ ]] \
    || { echo "invalid WAM_CODE_RELEASE_ID: $CODE_RELEASE_ID" >&2; exit 2; }
[[ "$RUNTIME_RELEASE_ID" =~ ^perception_runtime-[0-9a-f]{20}$ ]] \
    || { echo "invalid WAM_PERCEPTION_RUNTIME_RELEASE_ID: $RUNTIME_RELEASE_ID" >&2; exit 2; }
[[ "$RUNTIME_LOCK_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "invalid WAM_PERCEPTION_RUNTIME_LOCK_SHA256" >&2; exit 2; }
HALL_CAL="${WAM_HALL_CALIBRATION:-${HALL_CAL:-$HOME/wam_data/config/calibration/force_feedback_calibration_veo6_tared.json}}"

for required_file in "$SYNC_CAL" "$CAMERA_LATENCY_CAL" "$TARGETS_FILE" "$HALL_CAL"; do
    [[ -f "$required_file" ]] || { echo "missing required file: $required_file" >&2; exit 2; }
done

case "$side" in
    left)  leader=$left_leader;  follower=$left_follower;  arm=left_arm ;;
    right) leader=$right_leader; follower=$right_follower; arm=right_arm ;;
    *) echo "side must be left|right" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$SDIR")"
[[ ! -e "$SDIR" ]] || { echo "session directory already exists; refusing to overwrite: $SDIR" >&2; exit 2; }
ready_file="$SDIR/SENSORS_READY.json"
start_file="$SDIR/START_CAPTURE"
stop_file="$SDIR/STOP_CAPTURE.json"
capture_ended_file="$SDIR/CAPTURE_ENDED.json"

recorder_cmd=(
    "$PY" "$AUSC/ausc/record_session.py"
    --session-dir "$SDIR" --seconds "$secs" --site "$site"
    --audio-device "${AUDIO_DEVICE:-Scarlett Solo USB: Audio}" --audio-channel "${AUDIO_CHANNEL:-1}"
    --camera-backend "${CAMERA_BACKEND:-orbbec}" --camera-index "${CAMERA_INDEX:-0}"
    --camera-id "$CAMERA_ID" --arm-side "$side"
    --hall-cal "$HALL_CAL" --sync-cal "$SYNC_CAL"
    --camera-latency-cal "$CAMERA_LATENCY_CAL"
    --code-release-id "$CODE_RELEASE_ID"
    --runtime-release-id "$RUNTIME_RELEASE_ID"
    --runtime-lock-sha256 "$RUNTIME_LOCK_SHA256"
    --ready-file "$ready_file" --start-file "$start_file"
    --stop-file "$stop_file" --capture-ended-file "$capture_ended_file"
    --show-targets --targets "$TARGETS_FILE"
)
[[ -z "${HALL_PORT:-}" ]] || recorder_cmd+=(--hall-port "$HALL_PORT")

if [[ "$TELEOP_BIN" == *.py ]]; then
    teleop_cmd=("$PY" "$TELEOP_BIN" "$secs")
else
    teleop_cmd=("$TELEOP_BIN" "$URDF" "$URDF_FOLLOWER" "$arm" "$leader" "$follower")
fi

json_array() {
    "$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"
}

recorder_json="$(json_array "${recorder_cmd[@]}")"
teleop_json="$(json_array "${teleop_cmd[@]}")"

echo "[combined] sensor-first readiness -> teleop -> shared start: $SDIR"
"$PY" "$AUSC/scripts/collection_supervisor.py" \
    --session-dir "$SDIR" \
    --recorder-cmd-json "$recorder_json" \
    --teleop-cmd-json "$teleop_json" \
    --stop-cmd-json "$TELEOP_STOP_CMD_JSON"

echo "[combined] complete and coverage-certified -> $SDIR"
echo "[combined] convert in pinned lerobot==0.3.3 environment:"
echo "  python $AUSC/scripts/session_to_lerobot_v2.py $SDIR --repo-id ausc/$name --root <out>"
