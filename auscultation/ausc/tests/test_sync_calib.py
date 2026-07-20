"""Pure tests for persisted cross-device clock calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

AUSC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUSC))

import sync_calib as sync  # noqa: E402
from record_session import load_sync_calibration  # noqa: E402


FORCE_SHA256 = "a" * 64


def _capture_mode() -> dict:
    return {
        "audio_channel": 1,
        "force_calibration_sha256": FORCE_SHA256,
        "audio": {
            "requested_device": "Scarlett Solo USB",
            "portaudio_index": 3,
            "hostapi_name": "ALSA",
            "device_name": "Scarlett Solo USB Audio",
            "max_input_channels": 2,
            "usb_serial": "AUDIO-SERIAL-001",
            "usb_vid": 0x1235,
            "usb_pid": 0x8211,
            "usb_identity_source": "linux_alsa_sysfs",
        },
        "hall": {
            "requested_port": "/dev/cu.usbmodem101",
            "device": "/dev/cu.usbmodem101",
            "usb_serial": "HALL-SERIAL-001",
            "usb_vid": 0x2E8A,
            "usb_pid": 0x000A,
            "manufacturer": "Raspberry Pi",
            "product": "Pico",
            "location": "1-2",
        },
    }


def test_hall_descriptor_uses_pyserial_usb_identity_not_only_port_path():
    port = SimpleNamespace(
        device="/dev/ttyACM0", serial_number="HALL-SERIAL-001",
        vid=0x2E8A, pid=0x000A, manufacturer="Raspberry Pi",
        product="Pico", location="1-2",
    )
    descriptor = sync.resolve_hall_hardware_descriptor(
        "/dev/ttyACM0", comports=lambda: [port],
    )
    assert descriptor["requested_port"] == "/dev/ttyACM0"
    assert descriptor["usb_serial"] == "HALL-SERIAL-001"
    assert descriptor["usb_vid"] == 0x2E8A
    assert descriptor["usb_pid"] == 0x000A


def test_hall_descriptor_rejects_usb_device_without_stable_serial():
    port = SimpleNamespace(
        device="/dev/ttyACM0", serial_number=None, vid=0x2E8A, pid=0x000A,
        manufacturer=None, product=None, location=None,
    )
    with pytest.raises(ValueError, match="Hall.*USB serial"):
        sync.resolve_hall_hardware_descriptor("/dev/ttyACM0", comports=lambda: [port])


def test_audio_descriptor_resolves_selected_portaudio_device_and_usb_identity():
    devices = [
        {"name": "Built-in", "hostapi": 0, "max_input_channels": 2},
        {"name": "Scarlett Solo USB Audio", "hostapi": 1, "max_input_channels": 2},
    ]
    hostapis = [{"name": "Core Audio"}, {"name": "ALSA"}]

    def query_devices(device=None, kind=None):
        return devices if device is None else devices[1]

    descriptor = sync.resolve_audio_hardware_descriptor(
        "Scarlett Solo USB", query_devices=query_devices,
        query_hostapis=lambda index: hostapis[index],
        usb_resolver=lambda *_: {
            "usb_serial": "AUDIO-SERIAL-001", "usb_vid": 0x1235,
            "usb_pid": 0x8211, "usb_identity_source": "test_sysfs",
        },
    )
    assert descriptor == {
        "requested_device": "Scarlett Solo USB",
        "portaudio_index": 1,
        "hostapi_name": "ALSA",
        "device_name": "Scarlett Solo USB Audio",
        "max_input_channels": 2,
        "usb_serial": "AUDIO-SERIAL-001",
        "usb_vid": 0x1235,
        "usb_pid": 0x8211,
        "usb_identity_source": "test_sysfs",
    }


def test_audio_descriptor_selects_commissioned_usb_serial_before_stale_hw_name():
    devices = [
        {
            "name": "Scarlett Solo USB: Audio (hw:2,0)",
            "hostapi": 0,
            "max_input_channels": 2,
        },
        {
            "name": "Scarlett Solo USB: Audio (hw:7,0)",
            "hostapi": 0,
            "max_input_channels": 2,
        },
        {
            "name": "Scarlett Solo USB Mic 2",
            "hostapi": 1,
            "max_input_channels": 2,
        },
    ]
    identities = {
        0: {
            "usb_serial": "WRONG-DEVICE",
            "usb_vid": 0x1235,
            "usb_pid": 0x8211,
            "usb_identity_source": "test_sysfs",
        },
        1: {
            "usb_serial": "Y77BQVG17CD6E1",
            "usb_vid": 0x1235,
            "usb_pid": 0x8211,
            "usb_identity_source": "test_sysfs",
        },
    }

    descriptor = sync.resolve_audio_hardware_descriptor(
        "Scarlett Solo USB: Audio (hw:2,0)",
        expected_usb_vid=0x1235,
        expected_usb_pid=0x8211,
        expected_usb_serial="Y77BQVG17CD6E1",
        query_devices=lambda *args: devices,
        query_hostapis=lambda index: {"name": "ALSA" if index == 0 else "PipeWire"},
        usb_resolver=lambda index, *_args: identities.get(index),
    )

    assert descriptor["portaudio_index"] == 1
    assert descriptor["device_name"].endswith("(hw:7,0)")
    assert descriptor["usb_serial"] == "Y77BQVG17CD6E1"


def test_audio_descriptor_serial_selection_is_unique_and_never_name_fallback():
    devices = [
        {
            "name": "Scarlett Solo USB: Audio (hw:2,0)",
            "hostapi": 0,
            "max_input_channels": 2,
        },
        {
            "name": "Scarlett Solo USB: Audio (hw:4,0)",
            "hostapi": 0,
            "max_input_channels": 2,
        },
    ]
    hostapi = lambda _index: {"name": "ALSA"}

    with pytest.raises(ValueError, match="not found"):
        sync.resolve_audio_hardware_descriptor(
            "Scarlett Solo USB: Audio (hw:2,0)",
            expected_usb_vid=0x1235,
            expected_usb_pid=0x8211,
            expected_usb_serial="Y77BQVG17CD6E1",
            query_devices=lambda *args: devices,
            query_hostapis=hostapi,
            usb_resolver=lambda *_args: {
                "usb_serial": "WRONG",
                "usb_vid": 0x1235,
                "usb_pid": 0x8211,
                "usb_identity_source": "test_sysfs",
            },
        )

    with pytest.raises(ValueError, match="ambiguous"):
        sync.resolve_audio_hardware_descriptor(
            "ignored-stale-name",
            expected_usb_vid=0x1235,
            expected_usb_pid=0x8211,
            expected_usb_serial="Y77BQVG17CD6E1",
            query_devices=lambda *args: devices,
            query_hostapis=hostapi,
            usb_resolver=lambda *_args: {
                "usb_serial": "Y77BQVG17CD6E1",
                "usb_vid": 0x1235,
                "usb_pid": 0x8211,
                "usb_identity_source": "test_sysfs",
            },
        )


def test_capture_opens_resolved_endpoints_not_remappable_queries():
    audio_endpoint, hall_endpoint = sync.sync_capture_endpoints(_capture_mode())
    assert audio_endpoint == 3
    assert hall_endpoint == "/dev/cu.usbmodem101"


def test_audio_descriptor_rejects_ambiguous_portaudio_index_mapping():
    duplicate = {"name": "Scarlett Solo", "hostapi": 0, "max_input_channels": 2}

    def query_devices(device=None, kind=None):
        return [dict(duplicate), dict(duplicate)] if device is None else dict(duplicate)

    with pytest.raises(ValueError, match="PortAudio.*ambiguous"):
        sync.resolve_audio_hardware_descriptor(
            "Scarlett", query_devices=query_devices,
            query_hostapis=lambda index: {"name": "ALSA"},
            usb_resolver=lambda *_: None,
        )


def test_linux_alsa_sysfs_mapping_captures_usb_serial_vid_pid(tmp_path):
    usb = tmp_path / "devices" / "usb1" / "1-2"
    interface = usb / "1-2:1.0" / "sound" / "card2"
    interface.mkdir(parents=True)
    (usb / "serial").write_text("AUDIO-SERIAL-001\n")
    (usb / "idVendor").write_text("1235\n")
    (usb / "idProduct").write_text("8211\n")
    sound_root = tmp_path / "class" / "sound"
    sound_root.mkdir(parents=True)
    (sound_root / "card2").symlink_to(interface, target_is_directory=True)

    identity = sync.resolve_linux_alsa_usb_identity(
        "Scarlett USB (hw:2,0)", sound_root=sound_root,
    )
    assert identity["usb_serial"] == "AUDIO-SERIAL-001"
    assert identity["usb_vid"] == 0x1235
    assert identity["usb_pid"] == 0x8211
    assert identity["alsa_card"] == 2
    assert identity["usb_port_path"] == "1-2"


def test_formal_sync_calibration_accepts_serialless_fixed_usb_topology():
    hall = _capture_mode()["hall"]
    audio = _capture_mode()["audio"]
    audio.update(
        usb_serial=None,
        usb_port_path="1-12",
        usb_identity_source="linux_alsa_sysfs_topology",
    )
    mode = sync.build_calibration_capture_mode(
        audio_device="hw:3,0",
        audio_channel=1,
        hall_port="/dev/ttyACM0",
        force_calibration_sha256=FORCE_SHA256,
        hall_resolver=lambda _: hall,
        audio_resolver=lambda _: audio,
    )
    assert mode["audio"]["usb_serial"] is None
    assert mode["audio"]["usb_port_path"] == "1-12"
    assert sync.sync_identity_warning(mode) is None


def test_runtime_rejects_serialless_audio_moved_to_different_usb_topology(tmp_path):
    calibrated = _capture_mode()
    calibrated["audio"].update(
        usb_serial=None,
        usb_port_path="1-12",
        usb_identity_source="linux_alsa_sysfs_topology",
    )
    path = tmp_path / "sync.json"
    sync.persist_sync_result(
        path,
        {
            "faulted": False,
            "n_matched": 4,
            "offset_ms": 12.5,
            "offset_std_ms": 1.2,
            "unique_matches": True,
            "measurement_method": "physical_diaphragm_taps",
            "capture_mode": calibrated,
        },
    )
    runtime = json.loads(json.dumps(calibrated))
    runtime["audio"]["usb_port_path"] = "1-11"
    with pytest.raises(ValueError, match="audio usb_port_path"):
        load_sync_calibration(path, expected_capture_mode=runtime)


def test_missing_audio_usb_serial_has_explicit_arm_ineligibility_warning():
    mode = _capture_mode()
    mode["audio"].update(
        usb_serial=None, usb_vid=None, usb_pid=None,
        usb_identity_source="unavailable_non_linux_alsa",
    )
    warning = sync.sync_identity_warning(mode)
    assert "not arm-eligible" in warning
    assert "stable audio USB serial" in warning
    assert sync.sync_identity_warning(_capture_mode()) is None


def test_formal_sync_calibration_requires_audio_usb_serial_unless_explicit_rehearsal():
    hall = _capture_mode()["hall"]
    audio = _capture_mode()["audio"]
    audio.update(
        usb_serial=None, usb_vid=None, usb_pid=None,
        usb_identity_source="unavailable_non_linux_alsa",
    )
    kwargs = {
        "audio_device": "Scarlett Solo USB",
        "audio_channel": 1,
        "hall_port": "/dev/ttyACM0",
        "force_calibration_sha256": FORCE_SHA256,
        "hall_resolver": lambda _: hall,
        "audio_resolver": lambda _: audio,
    }
    with pytest.raises(ValueError, match="stable audio USB serial"):
        sync.build_calibration_capture_mode(**kwargs)
    rehearsal = sync.build_calibration_capture_mode(
        **kwargs, allow_unstable_audio_identity=True,
    )
    assert rehearsal["audio"]["usb_serial"] is None


def test_fit_device_clock_reports_drift_and_residual():
    device_ms = np.array([0.0, 1_000.0, 2_000.0, 3_000.0])
    host_s = 10.0 + device_ms * 0.001001
    fit = sync.fit_device_clock(host_s, device_ms)
    assert fit["host_s_per_device_ms"] == pytest.approx(0.001001)
    assert fit["host_intercept_s"] == pytest.approx(10.0)
    assert fit["drift_ppm"] == pytest.approx(1_000.0)
    assert fit["rmse_ms"] == pytest.approx(0.0, abs=1e-9)


def _synth_tap_capture(offset_s=0.030, n_taps=6, sr=16000, dur=15.0, t_start=1000.0):
    """Synthetic capture: quiet audio with sharp tap bursts + a large low-frequency stream-open
    transient, and a 100 Hz force log whose impulses lag the audio taps by ``offset_s``."""
    rng = np.random.default_rng(0)
    n = int(sr * dur)
    t = np.arange(n) / sr
    audio = 0.001 * rng.standard_normal(n)
    audio += 0.9 * np.exp(-t / 1.5) * np.sin(2 * np.pi * 5.0 * t)  # stream-open DC-ish transient
    tap_times = np.linspace(2.0, 2.0 + (n_taps - 1) * 2.0, n_taps)
    for tt in tap_times:
        i0 = int(tt * sr)
        k = np.arange(int(0.006 * sr))
        audio[i0:i0 + len(k)] += 0.5 * np.exp(-k / (0.0015 * sr)) * np.sin(2 * np.pi * 2000.0 * k / sr)
    fsr = 100
    ft = t_start + np.arange(int(fsr * dur)) / fsr
    ff = 1.5 + 0.02 * rng.standard_normal(ft.size)
    for tt in tap_times:
        c = tt + offset_s  # force lags audio
        dt = np.abs((ft - t_start) - c)
        bump = dt < 0.025
        ff[bump] += 5.5 * (1.0 - dt[bump] / 0.025)
    return audio.astype(np.float32), sr, t_start, ft, ff, offset_s


def test_compute_tap_sync_recovers_known_offset_despite_startup_transient():
    audio, sr, t_start, ft, ff, off = _synth_tap_capture(offset_s=0.030)
    det = sync.compute_tap_sync(audio, sr, t_start, ft, ff)
    assert det["n_matched"] >= 5
    assert det["offset_s"] == pytest.approx(off, abs=0.008)
    assert det["offset_std_s"] < 0.005  # tight enough for the <=5ms persist gate
    assert det["n_force_taps"] == 6


def test_compute_tap_sync_recovers_negative_offset():
    audio, sr, t_start, ft, ff, off = _synth_tap_capture(offset_s=-0.020)
    det = sync.compute_tap_sync(audio, sr, t_start, ft, ff)
    assert det["offset_s"] == pytest.approx(-0.020, abs=0.008)


def test_compute_tap_sync_returns_empty_when_streams_do_not_overlap():
    audio, sr, _t, ft, ff, _off = _synth_tap_capture(t_start=1000.0)
    det = sync.compute_tap_sync(audio, sr, 5000.0, ft, ff)  # audio window 4000s after the force log
    assert det["n_matched"] == 0
    assert det["offset_s"] is None


def test_persist_sync_result_rejects_unusable_measurement(tmp_path):
    with pytest.raises(ValueError, match="healthy.*matched taps"):
        sync.persist_sync_result(
            tmp_path / "sync.json",
            {"faulted": True, "n_matched": 4, "offset_ms": 12.0},
        )
    with pytest.raises(ValueError, match="healthy.*matched taps"):
        sync.persist_sync_result(
            tmp_path / "sync.json",
            {"faulted": False, "n_matched": 1, "offset_ms": 12.0},
        )
    with pytest.raises(ValueError, match="uncertainty|unique"):
        sync.persist_sync_result(
            tmp_path / "sync.json",
            {"faulted": False, "n_matched": 4, "offset_ms": 12.0,
             "offset_std_ms": 8.0, "unique_matches": True},
        )
    with pytest.raises(ValueError, match="uncertainty|unique"):
        sync.persist_sync_result(
            tmp_path / "sync.json",
            {"faulted": False, "n_matched": 4, "offset_ms": 12.0,
             "offset_std_ms": 1.0, "unique_matches": False},
        )


def test_tap_matching_is_one_to_one():
    pairs, deltas = sync.match_unique_taps(
        np.array([1.0, 2.0]), np.array([0.99, 1.01, 2.01]), max_delta_s=0.25,
    )
    assert len(pairs) == 2
    assert len({audio_i for audio_i, _ in pairs}) == 2
    assert len({force_i for _, force_i in pairs}) == 2
    assert deltas.shape == (2,)


def test_persist_sync_result_writes_versioned_contract(tmp_path):
    path = tmp_path / "sync.json"
    sync.persist_sync_result(
        path,
        {"faulted": False, "n_matched": 4, "offset_ms": 12.5, "offset_std_ms": 1.2,
         "unique_matches": True, "measurement_method": "physical_diaphragm_taps",
         "capture_mode": _capture_mode()},
    )
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "wam_sync_v1"
    assert payload["offset_definition"] == "force_t_minus_audio_t"
    assert payload["audio_to_force_offset_ms"] == pytest.approx(12.5)
    assert payload["unique_matches"] is True
    assert payload["measurement_method"] == "physical_diaphragm_taps"
    assert payload["capture_mode"] == _capture_mode()


def test_persist_sync_result_requires_concrete_capture_mode_identity(tmp_path):
    result = {
        "faulted": False,
        "n_matched": 4,
        "offset_ms": 12.5,
        "offset_std_ms": 1.2,
        "unique_matches": True,
        "measurement_method": "physical_diaphragm_taps",
        "capture_mode": _capture_mode(),
    }
    for field in _capture_mode():
        invalid = {**result, "capture_mode": _capture_mode()}
        invalid["capture_mode"].pop(field)
        with pytest.raises(ValueError, match=f"capture_mode.*{field}"):
            sync.persist_sync_result(tmp_path / f"missing_{field}.json", invalid)


@pytest.mark.parametrize(
    ("component", "field"),
    [
        ("hall", "usb_serial"),
        ("hall", "usb_vid"),
        ("hall", "usb_pid"),
        ("audio", "portaudio_index"),
        ("audio", "hostapi_name"),
        ("audio", "device_name"),
        ("audio", "max_input_channels"),
    ],
)
def test_persist_sync_result_requires_descriptor_fields(tmp_path, component, field):
    result = {
        "faulted": False,
        "n_matched": 4,
        "offset_ms": 12.5,
        "offset_std_ms": 1.2,
        "unique_matches": True,
        "measurement_method": "physical_diaphragm_taps",
        "capture_mode": _capture_mode(),
    }
    result["capture_mode"][component].pop(field)
    with pytest.raises(ValueError, match=f"capture_mode.*{component}.*{field}"):
        sync.persist_sync_result(tmp_path / f"missing_{component}_{field}.json", result)


def test_sync_audio_channel_must_exist_on_selected_portaudio_device(tmp_path):
    result = {
        "faulted": False,
        "n_matched": 4,
        "offset_ms": 12.5,
        "offset_std_ms": 1.2,
        "unique_matches": True,
        "measurement_method": "physical_diaphragm_taps",
        "capture_mode": _capture_mode(),
    }
    result["capture_mode"]["audio_channel"] = 2
    with pytest.raises(ValueError, match="audio_channel.*max_input_channels"):
        sync.persist_sync_result(tmp_path / "bad_channel.json", result)


def test_sync_cli_requires_explicit_device_and_calibration_identity(tmp_path):
    with pytest.raises(SystemExit):
        sync.main(["--seconds", "0", "--out", str(tmp_path / "sync.json")])


def _cli_args(tmp_path, *extra: str) -> list[str]:
    return [
        "--seconds", "0",
        "--audio-device", "Scarlett Solo USB",
        "--audio-channel", "1",
        "--hall-port", "/dev/ttyACM0",
        "--hall-cal", str(tmp_path / "force.json"),
        "--out", str(tmp_path / "sync.json"),
        *extra,
    ]


def _cli_result(*, faulted: bool = False, n_matched: int = 4) -> dict:
    return {
        "audio_health": {"status": "SILENT" if faulted else "OK"},
        "force_health": {"status": "OK"},
        "faulted": faulted,
        "n_matched": n_matched,
        "offset_ms": None if faulted or n_matched < 4 else 12.5,
        "offset_std_ms": None if faulted or n_matched < 4 else 1.2,
        "unique_matches": True,
        "measurement_method": "physical_diaphragm_taps",
        "capture_mode": _capture_mode(),
    }


def test_sync_cli_fails_before_capture_if_output_already_exists(tmp_path, monkeypatch):
    output = tmp_path / "sync.json"
    output.write_text('{"stale": true}\n')
    called = False

    def should_not_capture(**_kwargs):
        nonlocal called
        called = True
        return _cli_result()

    monkeypatch.setattr(sync, "calibrate_tap_sync", should_not_capture)
    assert sync.main(_cli_args(tmp_path)) != 0
    assert called is False
    assert json.loads(output.read_text()) == {"stale": True}


@pytest.mark.parametrize(
    "result",
    [_cli_result(faulted=True), _cli_result(n_matched=3)],
    ids=["device-fault", "too-few-taps"],
)
def test_sync_cli_returns_nonzero_and_writes_nothing_for_unusable_result(
    tmp_path, monkeypatch, result,
):
    monkeypatch.setattr(sync, "calibrate_tap_sync", lambda **_kwargs: result)
    assert sync.main(_cli_args(tmp_path)) != 0
    assert not (tmp_path / "sync.json").exists()


def test_sync_cli_applies_quality_gate_even_without_an_output_path(tmp_path, monkeypatch):
    result = _cli_result()
    result["offset_std_ms"] = 99.0
    monkeypatch.setattr(sync, "calibrate_tap_sync", lambda **_kwargs: result)

    assert sync.main(_cli_args(tmp_path)[:-2]) != 0


def test_sync_persistence_never_replaces_an_existing_calibration(tmp_path):
    output = tmp_path / "sync.json"
    output.write_text('{"original": true}\n')

    with pytest.raises(FileExistsError):
        sync.persist_sync_result(output, _cli_result())

    assert json.loads(output.read_text()) == {"original": True}
