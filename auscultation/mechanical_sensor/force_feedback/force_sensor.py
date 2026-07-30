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
   reader runs a background thread; poll it every step and handle the screened
   and physical hard limits independently::

       with ForceSensor(port, CAL, on_hard_limit=handle_force_limit) as force:
           while running:
               s = force.read()
               obs["observation.contact_force_n"] = s.force_n
               if s.status in ("SCREENED_LIMIT", "HARD_LIMIT"):
                   robot.hold()

Firmware: flash ``firmware/main.py`` (MicroPython) which prints ``t_ms,raw`` at
~100 Hz on GP28. JSON lines (``{"t_ms":..,"raw":..}``) are also accepted.
"""

from __future__ import annotations

import json
import math
import threading
import time
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_SCREENED_LIMIT = "SCREENED_LIMIT"
STATUS_HARD_LIMIT = "HARD_LIMIT"
STATUS_STALE = "STALE"
STATUS_NO_DATA = "NO_DATA"
DEFAULT_CALIBRATION_PATH = Path(__file__).with_name("force_feedback_calibration_veo6_tared.json")


@dataclass(frozen=True)
class ForceSample:
    """One immutable force reading. ``read()`` always returns one of these."""

    t: float            # host monotonic timestamp (s)
    raw: int            # raw ADC counts as sent by the firmware (0..adc_full_scale)
    hall_v: float       # Hall output voltage (V)
    displacement_mm: float
    force_n: float      # estimated contact force (N)
    status: str         # OK | WARN | SCREENED_LIMIT | HARD_LIMIT | STALE | NO_DATA
    age_s: float = 0.0  # how old this reading is at the moment read() returned it
    device_t_ms: float | None = None  # RP2040 clock, preserved for drift fitting


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
    def from_mapping(cls, cfg: Mapping[str, object]) -> "ForceCalibration":
        """Validate and parse one calibration mapping without truthy/string coercions.

        Both upright and inverted Hall installations are supported, but force and
        displacement must each be monotonic in the same direction as Hall voltage.
        """
        if not isinstance(cfg, Mapping):
            raise ValueError("force calibration root must be a JSON object")

        def finite_number(name: str, default: float) -> float:
            value = cfg.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite JSON number")
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"{name} must be a finite JSON number")
            return result

        supply_v = finite_number("supply_v", 3.3)
        if supply_v <= 0:
            raise ValueError("supply_v must be > 0")
        adc_value = cfg.get("adc_full_scale", 65535)
        if type(adc_value) is not int or adc_value <= 0:
            raise ValueError("adc_full_scale must be a positive JSON integer")
        adc_full_scale = adc_value

        target_force_n = finite_number("target_force_n", 4.0)
        target_tolerance_n = finite_number("target_tolerance_n", 0.5)
        max_screened_force_n = finite_number("max_screened_force_n", 4.6)
        hard_limit_force_n = finite_number("hard_limit_force_n", 5.5)
        stale_timeout_ms = finite_number("stale_timeout_ms", 200.0)
        if target_force_n < 0:
            raise ValueError("target_force_n must be >= 0")
        if target_tolerance_n < 0:
            raise ValueError("target_tolerance_n must be >= 0")
        if max_screened_force_n <= target_force_n + target_tolerance_n:
            raise ValueError(
                "max_screened_force_n must exceed target_force_n + target_tolerance_n"
            )
        if hard_limit_force_n <= max_screened_force_n:
            raise ValueError("hard_limit_force_n must exceed max_screened_force_n")
        if stale_timeout_ms <= 0:
            raise ValueError("stale_timeout_ms must be > 0")

        calibrated = cfg.get("calibrated", False)
        if type(calibrated) is not bool:
            raise ValueError("calibrated must be a JSON boolean")

        raw_points = cfg.get("calibration_points")
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            raise ValueError("calibration needs >= 2 calibration_points")
        points: list[tuple[float, float, float]] = []
        for index, point in enumerate(raw_points):
            if not isinstance(point, Mapping):
                raise ValueError(f"calibration_points[{index}] must be a JSON object")

            def point_number(name: str, default: float | None = None) -> float:
                value = point.get(name, default)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"calibration_points[{index}].{name} must be a finite JSON number"
                    )
                result = float(value)
                if not math.isfinite(result):
                    raise ValueError(
                        f"calibration_points[{index}].{name} must be a finite JSON number"
                    )
                return result

            hall_v = point_number("hall_v")
            force_n = point_number("force_n")
            displacement_mm = point_number("displacement_mm", 0.0)
            if not 0 <= hall_v <= supply_v:
                raise ValueError(
                    f"calibration_points[{index}].hall_v must lie within [0, supply_v]"
                )
            if force_n < 0:
                raise ValueError(f"calibration_points[{index}].force_n must be >= 0")
            if displacement_mm < 0:
                raise ValueError(
                    f"calibration_points[{index}].displacement_mm must be >= 0"
                )
            points.append((hall_v, force_n, displacement_mm))

        points.sort(key=lambda point: point[0])
        volts = [point[0] for point in points]
        forces = [point[1] for point in points]
        displacements = [point[2] for point in points]
        if any(right <= left for left, right in zip(volts, volts[1:])):
            raise ValueError("calibration point hall_v knots must be strictly increasing")

        def monotonic_direction(values: list[float], name: str) -> int:
            deltas = [right - left for left, right in zip(values, values[1:])]
            if all(delta >= 0 for delta in deltas) and any(delta > 0 for delta in deltas):
                return 1
            if all(delta <= 0 for delta in deltas) and any(delta < 0 for delta in deltas):
                return -1
            raise ValueError(f"calibration point {name} curve must be monotonic and nonconstant")

        force_direction = monotonic_direction(forces, "force_n")
        displacement_direction = monotonic_direction(displacements, "displacement_mm")
        if force_direction != displacement_direction:
            raise ValueError(
                "force_n and displacement_mm curves must use the same monotonic direction"
            )
        force_min, force_max = min(forces), max(forces)
        if target_force_n - target_tolerance_n < force_min:
            raise ValueError("target_force_n - target_tolerance_n falls outside force_n knots")
        if hard_limit_force_n > force_max:
            raise ValueError("hard_limit_force_n falls outside force_n knots")

        return cls(
            supply_v=supply_v,
            adc_full_scale=adc_full_scale,
            volts=volts,
            forces=forces,
            displacements=displacements,
            target_force_n=target_force_n,
            target_tolerance_n=target_tolerance_n,
            max_screened_force_n=max_screened_force_n,
            hard_limit_force_n=hard_limit_force_n,
            stale_timeout_ms=stale_timeout_ms,
            calibrated=calibrated,
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ForceCalibration":
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("force calibration must be valid UTF-8 JSON") from exc
        return cls.from_mapping(payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "ForceCalibration":
        return cls.from_bytes(Path(path).read_bytes())

    def raw_to_volts(self, raw: float) -> float:
        return raw / self.adc_full_scale * self.supply_v

    def volts_to_force(self, hall_v: float) -> float:
        return _interp(hall_v, self.volts, self.forces)

    def volts_to_displacement(self, hall_v: float) -> float:
        return _interp(hall_v, self.volts, self.displacements)

    def classify(self, force_n: float) -> str:
        if force_n >= self.hard_limit_force_n:
            return STATUS_HARD_LIMIT
        if force_n >= self.max_screened_force_n:
            return STATUS_SCREENED_LIMIT
        if abs(force_n - self.target_force_n) <= self.target_tolerance_n:
            return STATUS_OK
        return STATUS_WARN

    def sample_from_raw(
        self, raw: int, t: float, *, device_t_ms: float | None = None,
    ) -> ForceSample:
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
            device_t_ms=device_t_ms,
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
    CSV and fires the legacy-named ``on_hard_limit`` callback on transitions
    into either ``SCREENED_LIMIT`` or ``HARD_LIMIT``.
    """

    def __init__(
        self,
        port: str,
        calibration: ForceCalibration,
        baud: int = 115200,
        on_hard_limit: Callable[[ForceSample], None] | None = None,
        log_csv: str | Path | None = None,
        on_sample: Callable[[ForceSample], None] | None = None,
        on_reader_error: Callable[[BaseException, float], None] | None = None,
    ) -> None:
        self.port = port
        self.cal = calibration
        self.baud = baud
        self.on_hard_limit = on_hard_limit
        self.on_sample = on_sample
        self.on_reader_error = on_reader_error
        self._log_path = Path(log_csv) if log_csv else None
        self._ser = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: ForceSample | None = None
        self._log_file = None
        self._last_safety_status: str | None = None
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
            self._log_file.write(
                "host_t_s,raw,hall_v,displacement_mm,force_n,status,device_t_ms\n"
            )
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
            except Exception as exc:  # noqa: BLE001 - keep safety monitoring alive
                if self.on_reader_error is not None:
                    try:
                        self.on_reader_error(exc, time.monotonic())
                    except Exception:
                        pass
                continue
            parsed = parse_sensor_line(raw_line)
            if parsed is None:
                continue
            device_t_ms, raw = parsed
            if self._first_line:  # first readline after connect can be a line fragment
                self._first_line = False
                continue
            if not (0 <= raw <= self.cal.adc_full_scale):  # reject corrupt/garbled USB lines
                continue
            sample = self.cal.sample_from_raw(
                raw, time.monotonic(), device_t_ms=device_t_ms,
            )
            with self._lock:
                self._latest = sample
            if self.on_sample is not None:
                try:
                    self.on_sample(sample)
                except Exception as exc:  # a recorder callback must fail closed, not kill safety
                    if self.on_reader_error is not None:
                        try:
                            self.on_reader_error(exc, time.monotonic())
                        except Exception:
                            pass
            if self._log_file is not None:
                self._log_file.write(
                    f"{sample.t:.4f},{sample.raw},{sample.hall_v},"
                    f"{sample.displacement_mm},{sample.force_n},{sample.status},"
                    f"{'' if sample.device_t_ms is None else sample.device_t_ms}\n"
                )
            if (sample.status in (STATUS_SCREENED_LIMIT, STATUS_HARD_LIMIT)
                    and sample.status != self._last_safety_status):
                self._last_safety_status = sample.status
                if self.on_hard_limit is not None:
                    try:
                        self.on_hard_limit(sample)
                    except Exception as exc:
                        # A supervisor callback failure is itself unsafe, but
                        # it must never kill the sole serial reader thread.
                        if self.on_reader_error is not None:
                            try:
                                self.on_reader_error(exc, time.monotonic())
                            except Exception:
                                pass
            elif sample.status not in (STATUS_SCREENED_LIMIT, STATUS_HARD_LIMIT):
                self._last_safety_status = None

    # -- non-blocking read -------------------------------------------------
    def read(self) -> ForceSample:
        """Latest force reading, never blocks. Marks the sample STALE if it is
        older than the calibration's stale_timeout, NO_DATA if none yet."""
        with self._lock:
            latest = self._latest
        now = time.monotonic()
        if latest is None:
            return ForceSample(now, 0, 0.0, 0.0, 0.0, STATUS_NO_DATA, age_s=0.0)
        age = max(0.0, now - latest.t)
        if age * 1000.0 > self.cal.stale_timeout_ms:
            return ForceSample(
                latest.t, latest.raw, latest.hall_v, latest.displacement_mm,
                latest.force_n, STATUS_STALE, age_s=age,
                device_t_ms=latest.device_t_ms,
            )
        return ForceSample(
            latest.t, latest.raw, latest.hall_v, latest.displacement_mm,
            latest.force_n, latest.status, age_s=age,
            device_t_ms=latest.device_t_ms,
        )


def _live_monitor() -> int:
    """``python force_sensor.py --port /dev/cu.usbmodem101`` -> live force print.
    Doubles as the standalone deployment monitor."""
    import argparse

    ap = argparse.ArgumentParser(description="Live VEO6 contact-force monitor")
    ap.add_argument("--port", required=True)
    ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
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
