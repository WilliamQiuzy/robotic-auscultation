#!/bin/bash
# Restart 8801 on train #65. MUST be used instead of a bare systemctl restart:
# the Orbbec SDK writes Log/OrbbecSDK.log.txt into the sealed code release
# (cwd) and torch.hub may write torch_home/hub/trusted_list into the sealed
# runtime release; either extra file fails the boot-time exact-set preflight.
# Proper fix (SDK log redirect out of the release) is queued for the next
# train; until then this script restores the sealed sets before starting.
set -e
if pgrep -f "controller/unilatera[l]" > /dev/null; then
  echo "CONTROLLER ALIVE - NOT RESTARTING (iron rule). Disable the arm first."
  exit 42
fi
CODE=/home/robot/Desktop/stethoscope/.wam-releases/code_release-932a031a7ed061b731ad
RT=/home/robot/wam_data/perception_runtime/releases/perception_runtime-8026d1846699aae5c71e
rm -f "$CODE/auscultation/Log/OrbbecSDK.log.txt"
rmdir "$CODE/auscultation/Log" 2>/dev/null || true
rm -f "$RT/torch_home/hub/trusted_list"
systemctl --user restart wam-collector-8801.service
sleep 30
systemctl --user is-active wam-collector-8801.service
curl -s http://127.0.0.1:8801/state | python3 -c "import json,sys; d=json.load(sys.stdin); print('state:', d['state'], '| camera:', d['camera_status'], '| model:', d['model_status'])"
