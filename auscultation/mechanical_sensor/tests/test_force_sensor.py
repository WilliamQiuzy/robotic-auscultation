"""Unit tests for the VEO6 force sensor host module (no hardware needed)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

FF = Path(__file__).resolve().parents[1] / "force_feedback"
sys.path.insert(0, str(FF))

import force_sensor as fs  # noqa: E402

CAL_PATH = FF / "force_feedback_calibration_veo6.json"


def test_deployed_default_matches_as_built_inverted_sensor():
    assert fs.DEFAULT_CALIBRATION_PATH.name == "force_feedback_calibration_veo6_tared.json"


@pytest.fixture()
def cal() -> fs.ForceCalibration:
    return fs.ForceCalibration.from_json(CAL_PATH)


def _calibration_payload() -> dict:
    return json.loads(CAL_PATH.read_text(encoding="utf-8"))


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


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_calibrated_flag_must_be_a_json_boolean(value):
    payload = _calibration_payload()
    payload["calibrated"] = value
    with pytest.raises(ValueError, match="calibrated.*boolean"):
        fs.ForceCalibration.from_mapping(payload)


def test_calibration_can_be_parsed_from_the_exact_bytes_that_are_hashed():
    raw = CAL_PATH.read_bytes()
    assert fs.ForceCalibration.from_bytes(raw) == fs.ForceCalibration.from_json(CAL_PATH)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("supply_v", float("nan")),
        ("supply_v", 0.0),
        ("adc_full_scale", 0),
        ("adc_full_scale", 65535.5),
        ("adc_full_scale", True),
        ("target_force_n", float("inf")),
        ("target_force_n", -0.1),
        ("target_tolerance_n", -0.1),
        ("max_screened_force_n", 4.5),
        ("hard_limit_force_n", 4.6),
        ("stale_timeout_ms", 0.0),
    ],
)
def test_calibration_rejects_nonfinite_or_logically_invalid_scalars(field, value):
    payload = _calibration_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        fs.ForceCalibration.from_mapping(payload)


@pytest.mark.parametrize("point_field", ["hall_v", "force_n", "displacement_mm"])
def test_calibration_rejects_nonfinite_knots(point_field):
    payload = _calibration_payload()
    payload["calibration_points"][1][point_field] = float("nan")
    with pytest.raises(ValueError, match=point_field):
        fs.ForceCalibration.from_mapping(payload)


def test_calibration_requires_strictly_increasing_hall_knots_after_sorting():
    payload = _calibration_payload()
    payload["calibration_points"][1]["hall_v"] = payload["calibration_points"][0]["hall_v"]
    with pytest.raises(ValueError, match="hall_v.*strictly increasing"):
        fs.ForceCalibration.from_mapping(payload)


def test_calibration_rejects_hall_knots_outside_adc_supply_range():
    payload = _calibration_payload()
    payload["calibration_points"][0]["hall_v"] = -0.01
    with pytest.raises(ValueError, match="hall_v.*supply_v"):
        fs.ForceCalibration.from_mapping(payload)


def test_force_and_displacement_curves_must_be_monotonic_in_one_shared_direction():
    payload = _calibration_payload()
    payload["calibration_points"] = [
        {"hall_v": 1.0, "force_n": 1.0, "displacement_mm": 0.0},
        {"hall_v": 2.0, "force_n": 3.0, "displacement_mm": 1.0},
        {"hall_v": 3.0, "force_n": 2.0, "displacement_mm": 2.0},
    ]
    with pytest.raises(ValueError, match="force_n.*monotonic"):
        fs.ForceCalibration.from_mapping(payload)

    payload["calibration_points"][2]["force_n"] = 4.0
    for point, displacement in zip(payload["calibration_points"], [2.0, 1.0, 0.0]):
        point["displacement_mm"] = displacement
    with pytest.raises(ValueError, match="same monotonic direction"):
        fs.ForceCalibration.from_mapping(payload)


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
    assert cal.classify(4.6) == fs.STATUS_SCREENED_LIMIT
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


def test_sample_preserves_firmware_timestamp(cal):
    raw = int(1.853 / 3.3 * 65535)
    s = cal.sample_from_raw(raw, t=10.0, device_t_ms=1234.5)
    assert s.t == 10.0
    assert s.device_t_ms == pytest.approx(1234.5)


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


def test_read_snapshots_before_sampling_now_so_concurrent_updates_cannot_make_negative_age(
    cal, monkeypatch,
):
    sensor = fs.ForceSensor("/dev/null", cal)
    clock = {"now": 10.0}

    class _UpdatingLock:
        def __enter__(self):
            sensor._latest = cal.sample_from_raw(36800, t=11.0)
            clock["now"] = 12.0

        def __exit__(self, *exc):
            return None

    sensor._lock = _UpdatingLock()
    monkeypatch.setattr(fs.time, "monotonic", lambda: clock["now"])

    sample = sensor.read()
    assert sample.age_s == pytest.approx(1.0)
    assert sample.age_s >= 0.0


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
    def reset_input_buffer(self):
        pass
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


def test_background_log_preserves_both_clocks(cal, monkeypatch, tmp_path):
    raw = int(1.853 / 3.3 * 65535)
    fake_mod = type(sys)("serial")
    fake_mod.Serial = lambda *a, **k: _FakeSerial(raw)
    monkeypatch.setitem(sys.modules, "serial", fake_mod)
    path = tmp_path / "force.csv"

    with fs.ForceSensor("/dev/fake", cal, log_csv=path):
        time.sleep(0.05)

    lines = path.read_text().splitlines()
    assert lines[0].split(",") == [
        "host_t_s", "raw", "hall_v", "displacement_mm", "force_n", "status", "device_t_ms"
    ]
    assert float(lines[1].split(",")[-1]) > 0.0
