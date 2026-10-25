# Auscultation rig — USB / hardware port map (linbox `robot-G914`)

Machine: **robot-G914** · RTX 5090 · `<lab-lan-ip>` (Tailscale `<tailnet-ip>`) · the box with the
working Femto Bolt + dual-arm CAN. This file records which USB device is what, where it was last seen,
and how to re-find it. Source of truth: `lsusb -t`, `dmesg`, and `journalctl -k` (persistent).

> ⚠️ **USB port paths (e.g. `9-2.2.1`) are NOT stable** — they change if you replug into a different
> physical port or hub. **Identify devices by VID:PID + serial number** (and by the node they create:
> `/dev/ttyACM*` for the sensor, an ALSA card named `Scarlett Solo USB` for the stethoscope). The paths
> below are "last known", useful for "which hub/port did I use".

## Project devices

| Role | Device | VID:PID | Serial | Appears as | Last-seen USB path | Status |
|---|---|---|---|---|---|---|
| **Stethoscope (audio)** | Focusrite **Scarlett Solo USB** (Gen 3) | `1235:8211` | `Y77BQVG17CD6E1` | ALSA USB-Audio card `"Scarlett Solo USB"` | `9-2.2.1` | ⛔ off (powered down) |
| **Magnetic force sensor** | RP2040 **MicroPython** (Pi Pico, SS49E Hall streamer) | `2e8a:0005` | `53032847188a2e1c` | `cdc_acm` → **`/dev/ttyACM0`** | `9-2.2.2` | ⛔ not connected |
| Depth camera (perception) | Orbbec **Femto Bolt** | `2bc5:066b` | `CL8326300A0` | UVC video (usbfs) | `10-2.3` | ✅ online |
| Arm CAN adapter A | PEAK **PCAN-USB Pro FD** | `0c72:0011` | — | `peak_usb` → **can0 + can1** | `5-1.3` | ✅ online |
| Arm CAN adapter B | PEAK **PCAN-USB Pro FD** | `0c72:0011` | — | `peak_usb` → **can2 + can3** | `5-1.4` | ✅ online |
| (secondary depth cam) | Intel **RealSense 405** | `8086:0b5b` | — | UVC video | bus 6 | present |
| (webcam) | ARC International Camera | `05a3:9230` | — | UVC video | `9-2.4` | present |
| (flash drive) | SanDisk 3.2Gen1 | `0781:5595` | — | usb-storage | `9-1` | present |

Arm CAN convention (see hand-eye memory): **left arm = leader can1 / follower can3 = AUSCULTATION**;
**right arm = leader can0 / follower can2 = DANGEROUS** (never energize casually). One PCAN adapter
(`5-1.3`) carries can0+can1, the other (`5-1.4`) carries can2+can3.

## Stethoscope + magnet sensor were on the same little hub

On **Jun 24 16:46** a **GenesysLogic USB2.1 Hub (`05e3:0610`)** was plugged at **`9-2.2`**, carrying:
`9-2.2.1` = Scarlett Solo (stethoscope), `9-2.2.2` = RP2040 sensor (`→ ttyACM0`), `9-2.2.4` = Femto Bolt
(camera has since moved to bus 10). **That sub-hub is currently disconnected** → both the stethoscope and
the magnet sensor are absent. To bring them back, reconnect that hub's **uplink cable into the 5090 box**
(and use a **data** USB cable, not charge-only).

## How to re-find each device quickly

```bash
# magnetic sensor (RP2040): is it enumerated?
ls -l /dev/serial/by-id/ ; lsusb | grep -i 2e8a        # expect a "MicroPython Board" + /dev/ttyACM0
# read the resting magnetic force (VEO6 calib is the default; rest ≈ 1.5 N):
cd ~/Desktop/wam-vla-medical/auscultation
~/miniconda3/envs/ausc/bin/python mechanical_sensor/force_feedback/force_sensor.py --port /dev/ttyACM0

# stethoscope (Scarlett Solo): is the audio interface present?
arecord -l | grep -i scarlett ; lsusb | grep -i 1235   # expect "Scarlett Solo USB" capture card
# code opens it by name "Scarlett Solo USB" (ausc/audio/audio_stream.py, record_session.py)
```

If a replug produces **no** `dmesg`/`lsusb` change at all, the data path isn't reaching linbox
(charge-only cable or the hub uplink) — that's physical, not software.
