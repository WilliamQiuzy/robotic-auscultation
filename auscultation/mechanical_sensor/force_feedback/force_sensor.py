"""Reusable contact-force sensor for the VEO6 ThinkLabs passive mount.

The fixed-frame SS49E Hall sensor sees the carriage magnet close the air gap as
contact force rises, so the Hall voltage changes MONOTONICALLY with force. The
SIGN of that change depends on the magnet orientation: the as-built unit (magnet
glued 2026-06-30) is INVERTED -- pressing LOWERS the reading (rest ~31577 raw /
1.59 V  ->  hard stop ~16355 raw / 0.82 V). Nothing in the reader assumes a
direction; the sign is carried entirely by the calibration file, whose knots are
stored volts-ascending with ``force_n`` DESCENDING. The matching file is
``force_feedback_calibration_veo6_tared.json``.

An RP2040 streams raw ADC over USB serial; this module converts raw -> volts ->
force with a calibration curve and exposes a NON-BLOCKING, thread-safe reader.
The serial stream is read STRICTLY line-by-line (``readline()``). Do NOT switch
to ``ser.read(in_waiting)`` + ``splitlines()`` to "grab the latest" -- a byte
chunk that ends mid-line parses the leading digit(s) of the raw field as a tiny
bogus value (e.g. "112608,3" -> raw=3) and manufactures phantom 0<->value
dropouts that look exactly like a flaky connection.

Two integration modes (the whole point of this module):

1. Imitation study (force becomes a LEARNED observation). In the record loop,
   call ``sensor.read()`` once per control step and store the value in the
   observation dict so the policy can learn force-aware behaviour::

       with ForceSensor("/dev/cu.usbmodem101", CAL) as force:
           while recording:
               obs = robot.get_observation()
               obs["observation.contact_force_n"] = force.read().force_n
               dataset.add_frame(obs, action)

2. Deployment (force is CONTINUOUSLY monitored for safety + control). The same
   reader runs a background thread; poll it every step and trip a hard stop if
   the contact force exceeds the screened limit::

       with ForceSensor(port, CAL, on_hard_limit=robot.estop) as force:
           while running:
               s = force.read()
               obs["observation.contact_force_n"] = s.force_n
               if s.status == "HARD_LIMIT":
                   robot.hold()

Firmware: flash ``firmware/main.py`` (MicroPython) which prints ``t_ms,raw`` at
~100 Hz on GP28. JSON lines (``{"t_ms":..,"raw":..}``) are also accepted.
"""

from __future__ import annotations

import json
import threading
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_HARD_LIMIT = "HARD_LIMIT"
STATUS_STALE = "STALE"
STATUS_NO_DATA = "NO_DATA"


@dataclass(frozen=True)
class ForceSample:
    """One immutable force reading. ``read()`` always returns one of these."""

    t: float            # host monotonic timestamp (s)
    raw: int            # raw ADC counts as sent by the firmware (0..adc_full_scale)
    hall_v: float       # Hall output voltage (V)
    displacement_mm: float
    force_n: float      # estimated contact force (N)
    status: str         # OK | WARN | HARD_LIMIT | STALE | NO_DATA
    age_s: float = 0.0  # how old this reading is at the moment read() returned it


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    """Monotonic 1-D linear interpolation with linear extrapolation past the
    last knot and clamping below the first knot. ``ys`` may ASCEND or DESCEND
    (the magnet face sets the sign), so this serves both an upright mount (force
    rises with volts) and the as-built INVERTED mount (force falls with volts).
    On the inverted face an over-force lands below xs[0] and clamps to ys[0] =
    the hard-limit force, so classify() still trips HARD_LIMIT safely."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        # extrapolate above the last point using the final segment slope so an
        # over-force reads as > the calibrated max (caller flags HARD_LIMIT)
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + slope * (x - xs[-1])
    i = bisect_left(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class ForceCalibration:
    """Maps Hall voltage -> contact force (and displacement) from a JSON file.
    ``calibration_points`` are sorted volts-ascending; ``force_n`` may ascend OR
    descend (inverted magnet face). The deployed unit uses the inverted
    ``force_feedback_calibration_veo6_tared.json`` (rest 1.5 N -> hard stop 5.5 N,
    anchored on the spring's mechanical hard-stop, no external weight = provisional
    absolute N; bench-check with a known weight to set ``calibrated: true``)."""

    supply_v: float
    adc_full_scale: int
    volts: list[float]           # calibration knots, ascending
    forces: list[float]
    displacements: list[float]
    target_force_n: float
    target_tolerance_n: float
    max_screened_force_n: float
    hard_limit_force_n: float
    stale_timeout_ms: float
    calibrated: bool

    @classmethod
    def from_json(cls, path: str | Path) -> "ForceCalibration":
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        pts = sorted(cfg["calibration_points"], key=lambda p: p["hall_v"])
        if len(pts) < 2:
            raise ValueError("calibration needs >= 2 calibration_points")
        return cls(
            supply_v=float(cfg.get("supply_v", 3.3)),
            adc_full_scale=int(cfg.get("adc_full_scale", 65535)),
            volts=[float(p["hall_v"]) for p in pts],
            forces=[float(p["force_n"]) for p in pts],
            displacements=[float(p.get("displacement_mm", 0.0)) for p in pts],
            target_force_n=float(cfg.get("target_force_n", 4.0)),
            target_tolerance_n=float(cfg.get("target_tolerance_n", 0.5)),
            max_screened_force_n=float(cfg.get("max_screened_force_n", 4.6)),
            hard_limit_force_n=float(cfg.get("hard_limit_force_n", 5.5)),
            stale_timeout_ms=float(cfg.get("stale_timeout_ms", 200.0)),
            calibrated=bool(cfg.get("calibrated", False)),
        )

    def raw_to_volts(self, raw: float) -> float:
        return raw / self.adc_full_scale * self.supply_v

    def volts_to_force(self, hall_v: float) -> float:
        return _interp(hall_v, self.volts, self.forces)

    def volts_to_displacement(self, hall_v: float) -> float:
        return _interp(hall_v, self.volts, self.displacements)

    def classify(self, force_n: float) -> str:
        if force_n >= self.hard_limit_force_n:
            return STATUS_HARD_LIMIT
        if abs(force_n - self.target_force_n) <= self.target_tolerance_n:
            return STATUS_OK
        return STATUS_WARN

    def sample_from_raw(self, raw: int, t: float) -> ForceSample:
        hall_v = self.raw_to_volts(raw)
        force = self.volts_to_force(hall_v)
        disp = self.volts_to_displacement(hall_v)
        return ForceSample(
            t=t,
            raw=int(raw),
            hall_v=round(hall_v, 4),
            displacement_mm=round(disp, 3),
            force_n=round(force, 3),
            status=self.classify(force),
        )


def parse_sensor_line(line: str) -> tuple[float | None, int] | None:
    """Parse one COMPLETE firmware line into (t_ms_or_None, raw_adc). Accepts
    ``raw`` / ``t_ms,raw`` CSV or ``{"t_ms":..,"raw":..}`` JSON. Returns None for
    blank/garbage lines. ``raw`` is the LAST CSV field, so only the very first
    read after connect (a possible fragment) can yield a wrong raw -- the reader
    drops that one (reset_input_buffer + first-line skip) and range-checks raw.
    This does no range check itself (it has no calibration); the caller does."""
    text = line.strip()
    if not text:
        return None
    try:
        if text.startswith("{"):
            payload = json.loads(text)
            raw = payload.get("raw", payload.get("adc"))
            if raw is None:
                return None
            return payload.get("t_ms"), int(raw)
        parts = text.split(",")
        if len(parts) == 1:
            return None, int(float(parts[0]))
        return float(parts[0]), int(float(parts[1]))
    except (ValueError, KeyError, TypeError):
        return None


class ForceSensor:
    """Background serial reader. Open with ``start()`` (or ``with``), then call
    the non-blocking ``read()`` from any loop. Optionally logs every sample to
    CSV and fires ``on_hard_limit`` once when the force first crosses the limit.
    """

    def __init__(
        self,
        port: str,
        calibration: ForceCalibration,
        baud: int = 115200,
        on_hard_limit: Callable[[ForceSample], None] | None = None,
        log_csv: str | Path | None = None,
    ) -> None:
        self.port = port
        self.cal = calibration
        self.baud = baud
        self.on_hard_limit = on_hard_limit
        self._log_path = Path(log_csv) if log_csv else None
        self._ser = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: ForceSample | None = None
        self._log_file = None
        self._in_hard_limit = False
        self._first_line = True   # first readline() after connect may be a fragment -> skip

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "ForceSensor":
        import serial  # lazy: only needed for live mode

        self._ser = serial.Serial(self.port, baudrate=self.baud, timeout=0.2)
        self._ser.reset_input_buffer()  # drop bytes buffered before we connected (likely a partial line)
        self._first_line = True
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("w", encoding="utf-8")
            self._log_file.write("host_t_s,raw,hall_v,displacement_mm,force_n,status\n")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="force-sensor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "ForceSensor":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- reader thread -----------------------------------------------------
    def _run(self) -> None:
        assert self._ser is not None
        # LINE-FRAMED read ONLY. readline() returns one complete "t_ms,raw\n"
        # record. Never replace this with ser.read(in_waiting)+splitlines() to
        # "get the freshest sample": a chunk ending mid-line parses the leading
        # digits of the raw field as a tiny bogus value and fabricates fake
        # 0<->value flicker that mimics a broken connection (learned the hard way).
        while not self._stop.is_set():
            try:
                raw_line = self._ser.readline().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - serial drop should not kill the loop
                continue
            parsed = parse_sensor_line(raw_line)
            if parsed is None:
                continue
            _t_ms, raw = parsed
            if self._first_line:  # first readline after connect can be a line fragment
                self._first_line = False
                continue
            if not (0 <= raw <= self.cal.adc_full_scale):  # reject corrupt/garbled USB lines
                continue
            sample = self.cal.sample_from_raw(raw, time.monotonic())
            with self._lock:
                self._latest = sample
            if self._log_file is not None:
                self._log_file.write(
                    f"{sample.t:.4f},{sample.raw},{sample.hall_v},"
                    f"{sample.displacement_mm},{sample.force_n},{sample.status}\n"
                )
            if sample.status == STATUS_HARD_LIMIT and not self._in_hard_limit:
                self._in_hard_limit = True
                if self.on_hard_limit is not None:
                    self.on_hard_limit(sample)
            elif sample.status != STATUS_HARD_LIMIT:
                self._in_hard_limit = False

    # -- non-blocking read -------------------------------------------------
    def read(self) -> ForceSample:
        """Latest force reading, never blocks. Marks the sample STALE if it is
        older than the calibration's stale_timeout, NO_DATA if none yet."""
        now = time.monotonic()
        with self._lock:
            latest = self._latest
        if latest is None:
            return ForceSample(now, 0, 0.0, 0.0, 0.0, STATUS_NO_DATA, age_s=0.0)
        age = now - latest.t
        if age * 1000.0 > self.cal.stale_timeout_ms:
            return ForceSample(
                latest.t, latest.raw, latest.hall_v, latest.displacement_mm,
                latest.force_n, STATUS_STALE, age_s=age,
            )
        return ForceSample(
            latest.t, latest.raw, latest.hall_v, latest.displacement_mm,
            latest.force_n, latest.status, age_s=age,
        )


def _live_monitor() -> int:
    """``python force_sensor.py --port /dev/cu.usbmodem101`` -> live force print.
    Doubles as the standalone deployment monitor."""
    import argparse

    ap = argparse.ArgumentParser(description="Live VEO6 contact-force monitor")
    ap.add_argument("--port", required=True)
    ap.add_argument("--calibration", default=str(Path(__file__).with_name("force_feedback_calibration_veo6.json")))
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--log-csv")
    ap.add_argument("--hz", type=float, default=20.0, help="print rate")
    args = ap.parse_args()

    cal = ForceCalibration.from_json(args.calibration)
    if not cal.calibrated:
        print("[warn] using DESIGN-NOMINAL calibration (calibrated=false) -- bench-calibrate before trusting absolute N")
    with ForceSensor(args.port, cal, baud=args.baud, log_csv=args.log_csv) as force:
        try:
            while True:
                s = force.read()
                print(f"force={s.force_n:5.2f}N  disp={s.displacement_mm:5.2f}mm  "
                      f"V={s.hall_v:5.3f}  raw={s.raw:5d}  [{s.status}]  age={s.age_s*1000:4.0f}ms",
                      end="\r", flush=True)
                time.sleep(1.0 / max(1.0, args.hz))
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_live_monitor())
