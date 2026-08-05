# VEO6 contact-force Hall streamer -- MicroPython, RP2040.
#
# Flash MicroPython, then copy this file onto the board as main.py so it runs
# on boot. It streams one CSV line per sample over USB serial:
#
#     <t_ms>,<raw_adc>
#
# read by force_sensor.ForceSensor on the host. Wiring (validated 2026-06-22):
#     SS49E VCC -> 3V3 ,  GND -> GND ,  OUT -> GP28
#
# This firmware is direction-AGNOSTIC: it just streams the raw ADC. Whether the
# number RISES or FALLS as contact force increases depends on the magnet
# orientation; the host calibration (force_sensor.ForceCalibration) carries the
# sign. The as-built unit is INVERTED (force lowers raw): rest ~31577, hard stop
# ~16355. Healthy rest is steady to ~+/-30 counts -- a value flickering to ~0 is
# almost always a host-side partial-line read bug, NOT the wiring (use readline).
#
# To get the REPL back (e.g. to reflash), connect and send Ctrl-C to stop this
# loop, or replug while holding BOOTSEL.

from machine import ADC
from time import ticks_ms, sleep_ms

HALL = ADC(28)          # GP28 = ADC2, where the SS49E OUT is soldered
OVERSAMPLE = 8          # average N reads to cut RP2040 ADC noise
PERIOD_MS = 10          # ~100 Hz stream rate

while True:
    acc = 0
    for _ in range(OVERSAMPLE):
        acc += HALL.read_u16()
    raw = acc // OVERSAMPLE
    print("%d,%d" % (ticks_ms(), raw))
    sleep_ms(PERIOD_MS)
