#!/bin/bash
# Guarded 8801 restart — release-agnostic (supersedes t65_restart_8801.sh).
# Reads the ACTIVE code/runtime IDs from the installed unit file, restores
# both sealed sets (Orbbec SDK writes Log/ into the code release cwd; torch
# hub may write trusted_list into the runtime torch_home — either extra file
# fails the boot exact-set preflight into a restart loop), then restarts.
# Durable fix (SDK log redirect out of the sealed tree) remains queued.
set -e
if pgrep -f "controller/unilatera[l]" > /dev/null; then
  echo "CONTROLLER ALIVE - NOT RESTARTING (iron rule). Disable the arm first."
  exit 42
fi
UNIT=/home/robot/.config/systemd/user/wam-collector-8801.service
CODE_ID=$(grep -o "code=code_release-[a-z0-9]*" $UNIT | cut -d= -f2)
RT_ID=$(grep -o "runtime=perception_runtime-[a-z0-9]*" $UNIT | cut -d= -f2)
if [ -z "$CODE_ID" ] || [ -z "$RT_ID" ]; then
  echo "cannot read active release IDs from the unit header"; exit 1
fi
echo "active: $CODE_ID / $RT_ID"
CODE=/home/robot/Desktop/stethoscope/.wam-releases/$CODE_ID
RT=/home/robot/wam_data/perception_runtime/releases/$RT_ID
rm -f "$CODE/auscultation/Log/OrbbecSDK.log.txt" 2>/dev/null || true
rmdir "$CODE/auscultation/Log" 2>/dev/null || true
rm -f "$RT/torch_home/hub/trusted_list" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user restart wam-collector-8801.service
sleep 30
systemctl --user is-active wam-collector-8801.service
MAINPID=$(systemctl --user show wam-collector-8801.service -p MainPID --value)
tr "\0" " " < /proc/$MAINPID/cmdline | grep -o "code_release-[a-z0-9]*" | head -1
curl -s http://127.0.0.1:8801/state | python3 -c "import json,sys; d=json.load(sys.stdin); print('camera:', d['camera_status'], '| model:', d['model_status'])"
