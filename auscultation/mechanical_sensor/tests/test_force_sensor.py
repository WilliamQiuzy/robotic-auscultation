"""Unit tests for the VEO6 force sensor host module (no hardware needed)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

FF = Path(__file__).resolve().parents[1] / "force_feedback"
sys.path.insert(0, str(FF))

import force_sensor as fs  # noqa: E402

CAL_PATH = FF / "force_feedback_calibration_veo6.json"


@pytest.fixture()
def cal() -> fs.ForceCalibration:
    return fs.ForceCalibration.from_json(CAL_PATH)


# -- parsing ---------------------------------------------------------------
def test_parse_csv_raw_only():
    assert fs.parse_sensor_line("2300") == (None, 2300)


def test_parse_csv_t_and_raw():
    assert fs.parse_sensor_line("1234,2300") == (1234.0, 2300)


def test_parse_json():
    assert fs.parse_sensor_line('{"t_ms":1234,"raw":2300}') == (1234, 2300)


@pytest.mark.parametrize("line", ["", "   ", "\n", "garbage", "a,b"])
def test_parse_rejects_junk(line):
    assert fs.parse_sensor_line(line) is None


# -- calibration -----------------------------------------------------------
def test_calibration_loads(cal):
    assert len(cal.volts) >= 2
    assert cal.volts == sorted(cal.volts)  # ascending
    assert cal.hard_limit_force_n == 5.5


def test_force_at_knots(cal):
    # design-nominal knots: 1.7358V->1.5N, 1.853V->4.0N, 2.0371V->5.5N
    assert cal.volts_to_force(1.7358) == pytest.approx(1.5, abs=1e-6)
    assert cal.volts_to_force(1.8530) == pytest.approx(4.0, abs=1e-6)
    assert cal.volts_to_force(2.0371) == pytest.approx(5.5, abs=1e-6)


def test_force_is_monotonic(cal):
    vs = [1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.05]
    forces = [cal.volts_to_force(v) for v in vs]
    assert forces == sorted(forces)


def test_force_clamps_below_and_extrapolates_above(cal):
    # below the rest voltage -> clamp to the first knot (no negative force)
    assert cal.volts_to_force(1.50) == pytest.approx(cal.forces[0])
    # above the last knot -> extrapolate over the hard limit so we can flag it
    assert cal.volts_to_force(2.20) > cal.hard_limit_force_n


def test_classify(cal):
    assert cal.classify(4.0) == fs.STATUS_OK
    assert cal.classify(4.3) == fs.STATUS_OK          # within tolerance
    assert cal.classify(2.0) == fs.STATUS_WARN
    assert cal.classify(5.5) == fs.STATUS_HARD_LIMIT
    assert cal.classify(6.0) == fs.STATUS_HARD_LIMIT


def test_raw_to_volts(cal):
    # 16-bit full scale at 3.3V
    assert cal.raw_to_volts(0) == pytest.approx(0.0)
    assert cal.raw_to_volts(65535) == pytest.approx(3.3, abs=1e-6)


def test_sample_from_raw(cal):
    raw = int(1.853 / 3.3 * 65535)  # ~ target force voltage
    s = cal.sample_from_raw(raw, t=10.0)
    assert s.force_n == pytest.approx(4.0, abs=0.05)
    assert s.status == fs.STATUS_OK
    assert s.raw == raw


# -- ForceSensor.read() state machine (no serial) --------------------------
def test_read_no_data(cal):
    sensor = fs.ForceSensor("/dev/null", cal)
    assert sensor.read().status == fs.STATUS_NO_DATA


def test_read_fresh_and_stale(cal):
    sensor = fs.ForceSensor("/dev/null", cal)
    sensor._latest = cal.sample_from_raw(36800, t=time.monotonic())  # fresh
    assert sensor.read().status in (fs.STATUS_OK, fs.STATUS_WARN, fs.STATUS_HARD_LIMIT)
    sensor._latest = cal.sample_from_raw(36800, t=time.monotonic() - 1.0)  # 1s old
    assert sensor.read().status == fs.STATUS_STALE


# -- end-to-end with a fake serial port ------------------------------------
class _FakeSerial:
    """Endless live stream of one fixed raw value (always a fresh sample)."""

    def __init__(self, raw):
        self._raw = raw
        self._t = 0
    def readline(self):
        time.sleep(0.002)
        self._t += 10
        return f"{self._t},{self._raw}\n".encode()
    def close(self):
        pass


def test_background_reader_end_to_end(cal, monkeypatch):
    raw = int(2.10 / 3.3 * 65535)  # above the hard-stop voltage -> HARD_LIMIT
    tripped = []
    fake_mod = type(sys)("serial")
    fake_mod.Serial = lambda *a, **k: _FakeSerial(raw)
    monkeypatch.setitem(sys.modules, "serial", fake_mod)

    with fs.ForceSensor("/dev/fake", cal, on_hard_limit=lambda s: tripped.append(s)) as sensor:
        time.sleep(0.1)
        s = sensor.read()
    assert s.force_n >= cal.hard_limit_force_n - 0.01
    assert s.status == fs.STATUS_HARD_LIMIT
    assert s.age_s * 1000 < cal.stale_timeout_ms      # fresh, not stale
    assert tripped, "on_hard_limit should fire once when force crosses the limit"
