#!/bin/bash
# Train #68 unit swap: code_release-d780c2a61a2d6fb2e10c +
# perception_runtime-62c7dbdb937c12791b6e (gate stall-hold + display annex,
# includes #65 depth veto + #66 zmap floor). Absorbs every deployment lesson:
# pre-clean sealed-release contamination, CAS install, daemon-reload BEFORE
# restart, controller iron-rule check.
set -e
if pgrep -f "controller/unilatera[l]" > /dev/null; then
  echo "CONTROLLER ALIVE - NOT PROCEEDING (iron rule). Disable the arm first."
  exit 42
fi
BASE=/home/robot/Desktop/stethoscope/.wam-releases
NEW=$BASE/code_release-d780c2a61a2d6fb2e10c
RT=/home/robot/wam_data/perception_runtime/releases/perception_runtime-62c7dbdb937c12791b6e
OLDCODE=$BASE/code_release-af63448ef70a53ae5a05
OLDRT=/home/robot/wam_data/perception_runtime/releases/perception_runtime-904e6fbeac028a277b65
UNIT=/home/robot/.config/systemd/user/wam-collector-8801.service
PY=/home/robot/wam_data/perception_envs/releases/linbox-perception-8ea23197c6a4d699280e/bin/python

for C in "$NEW" "$OLDCODE"; do
  rm -f "$C/auscultation/Log/OrbbecSDK.log.txt" 2>/dev/null || true
  rmdir "$C/auscultation/Log" 2>/dev/null || true
done
for R in "$RT" "$OLDRT"; do
  rm -f "$R/torch_home/hub/trusted_list" 2>/dev/null || true
done

CUR_SHA=$(sha256sum $UNIT | cut -d" " -f1)
$PY -B $NEW/auscultation/scripts/deployment/collection_service_unit.py --install \
  --code-release $NEW \
  --code-release-id code_release-d780c2a61a2d6fb2e10c \
  --environment-release /home/robot/wam_data/perception_envs/releases/linbox-perception-8ea23197c6a4d699280e \
  --environment-id linbox-perception-8ea23197c6a4d699280e \
  --runtime-release $RT \
  --runtime-release-id perception_runtime-62c7dbdb937c12791b6e \
  --camera-id CL8326300A0 \
  --sessions-root /home/robot/wam_data/sessions \
  --unit-path $UNIT \
  --verifier-python $PY \
  --production-activation /home/robot/wam_canary/reissue_releases/activation-416a3f1f0d564c8b3a42/activation.json \
  --production-activation-sha256 87fa96b3480f08c79b1a8e1960dd26980572a6a8d30d4257317ebcf66bfe53ee \
  --hall-calibration /home/robot/wam_canary/commissioning_r5/force_cal_r5.json \
  --hall-calibration-sha256 93669c1a91c7909d32a486cb27ea5552f18d4f5691630b73ac7972ce2434c789 \
  --hall-port /dev/serial/by-id/usb-MicroPython_Board_in_FS_mode_5303284738b7539c-if00 \
  --hall-expected-usb-vid 11914 --hall-expected-usb-pid 5 \
  --hall-expected-usb-serial 5303284738b7539c \
  --sync-calibration /home/robot/wam_canary/commissioning_r5/sync_cal_r5.json \
  --sync-calibration-sha256 3ffff90d95f7fc979ecc1c6be10d2ac0f7d1d89169d3131c47f288ed599caa86 \
  --controller-launch-release /home/robot/wam_data/controller_launch/releases/controller_launch_root_v1-380ae525eb025a30e3193d965960e4196c0faeda5104695382f59c3d1d06da41/controller_launch_v1-380ae525eb025a30e3193d965960e4196c0faeda5104695382f59c3d1d06da41 \
  --controller-launch-trusted-root /home/robot/wam_data/controller_launch/releases/controller_launch_root_v1-380ae525eb025a30e3193d965960e4196c0faeda5104695382f59c3d1d06da41 \
  --controller-launch-contract-sha256 380ae525eb025a30e3193d965960e4196c0faeda5104695382f59c3d1d06da41 \
  --safe-stop-disable-cmd "bash $NEW/auscultation/scripts/safe_stop_disable.sh" \
  --port 8801 \
  --audio-device usb-serial:Y77BQVG17CD6E1 --audio-sample-rate 48000 \
  --audio-channels 2 --audio-pick-channel 1 \
  --audio-expected-usb-vid 4661 --audio-expected-usb-pid 33297 \
  --audio-expected-usb-serial Y77BQVG17CD6E1 \
  --expected-unit-sha256 $CUR_SHA
echo "=== unit installed:"
head -4 $UNIT
systemctl --user daemon-reload
if pgrep -f "controller/unilatera[l]" > /dev/null; then
  echo "CONTROLLER APPEARED - NOT RESTARTING (iron rule)."
  exit 42
fi
systemctl --user restart wam-collector-8801.service
sleep 30
systemctl --user is-active wam-collector-8801.service
MAINPID=$(systemctl --user show wam-collector-8801.service -p MainPID --value)
tr "\0" " " < /proc/$MAINPID/cmdline | grep -o "code_release-[a-z0-9]*" | head -1
echo "=== done: 8801 on train #68."
