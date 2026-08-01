"""Minimal imitation-learning data-collection loop: record the sensor streams to a
host-clock-aligned session container.

Opens the three acquisition primitives — ForceSensor (Hall), AudioStream (stethoscope),
CameraStream (RGB, or RGB+depth on Linux via the Orbbec backend) — and records, for a fixed
duration, a session directory that downstream IL / annotation reads. Everything is stamped on
one `time.monotonic()` clock (see docs/realtime_multimodal_pipeline.md §1, §3, §8).

Multi-rate by design (§3): audio is a continuous sidecar wav; force logs at its native ~100 Hz;
camera frames go to a video + a frame-time index; and a fused per-step `obs.csv` is sampled at
the control rate (default 50 Hz) — the LeRobot-frame precursor (robot state/action slot in when
the OpenArm teleop is wired in).

    python ausc/record_session.py --seconds 15 --camera-backend avfoundation --camera-index 1
    # on the 5090 (depth): --camera-backend orbbec   -> also writes depth/

Layout:
  data/sessions/session_<stamp>/
    manifest.json     t0 (monotonic+wall), durations, device config, stream metadata
    audio.wav         16 kHz mono, continuous
    force.csv         host_t_s, raw, hall_v, displacement_mm, force_n, status, device_t_ms
    rgb.mp4           color video
    rgb_times.csv     frame_index, host_arrival_t_s, device_t_ms, time_source
    depth/            depth_<index>.npy.zlib + depth_times.csv   (orbbec backend only)
    obs.csv           t, force_n, status, audio_rms_db, audio_peak_db, cam_index, cam_t  (50 Hz)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import math
import os
import queue
import re
import shutil
import sys
import threading
import time
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # auscultation/
sys.path[:0] = [HERE, os.path.join(HERE, "audio"), os.path.join(HERE, "perception"),
                os.path.join(ROOT, "mechanical_sensor", "force_feedback")]
from health import audio_status, force_status, warn  # noqa: E402
import depth_codec  # noqa: E402  # perception/ is on sys.path above

SESSION_SCHEMA_VERSION = "wam_session_v2"
CAPTURE_RELEASE_PROVENANCE_SCHEMA_VERSION = "wam_capture_release_provenance_v1"
CODE_RELEASE_ID_PATTERN = re.compile(r"^code_release-[0-9a-f]{20}$")
RUNTIME_RELEASE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*-[0-9a-f]{20}$")
CAMERA_TIME_SOURCE_BY_BACKEND = {
    "orbbec": "orbbec_device_and_host_arrival",
    "avfoundation": "opencv_host_arrival",
    "opencv": "opencv_host_arrival",
}
ORBBEC_COLOR_FORMATS = frozenset({"RGB", "BGR", "MJPG", "YUYV", "UYVY", "NV12"})
ORBBEC_ALIGN_MODES = frozenset({"software_depth_to_color", "disabled"})


class RecordingAborted(RuntimeError):
    """Raised when a collection cannot safely become training data."""


class SafetyStop:
    """Latch an unsafe force state and leave an atomic sentinel for the supervisor."""

    UNSAFE_FORCE_STATUSES = frozenset({"SCREENED_LIMIT", "HARD_LIMIT", "STALE", "NO_DATA"})

    def __init__(self, session_dir: str | Path) -> None:
        self._event = threading.Event()
        self._reason = "force hard limit tripped"
        self._severity: str | None = None
        self._force_status: str | None = None
        self._session_dir = Path(session_dir)
        self._lock = threading.Lock()

    def __call__(self, sample) -> None:  # noqa: ANN001
        self.observe(sample)

    def observe(self, sample) -> None:  # noqa: ANN001
        status = str(getattr(sample, "status", "NO_DATA"))
        if status not in self.UNSAFE_FORCE_STATUSES:
            return
        force_n = float(getattr(sample, "force_n", 0.0))
        reason = (f"force hard limit tripped at {force_n:.3f} N" if status == "HARD_LIMIT"
                  else f"unsafe force status {status} at {force_n:.3f} N")
        severity = "hard" if status == "HARD_LIMIT" else "screened"
        with self._lock:
            # The first unsafe event latches collection shutdown, but a later
            # physical hard limit must still replace a screened/stale event so
            # the supervisor takes the emergency-disable path.
            if self._event.is_set() and not (severity == "hard" and self._severity != "hard"):
                return
            previous_status = self._force_status
            self._reason = reason
            self._severity = severity
            self._force_status = status
            payload = {
                "reason": reason,
                "force_status": status,
                "severity": severity,
                "force_n": force_n,
                "t_monotonic": time.monotonic(),
            }
            if previous_status is not None:
                payload["escalated_from"] = previous_status
            self._session_dir.mkdir(parents=True, exist_ok=True)
            _write_manifest(self._session_dir / "SAFETY_ABORT.json", payload)
            self._event.set()

    @property
    def tripped(self) -> bool:
        return self._event.is_set()

    def raise_if_tripped(self) -> None:
        if self.tripped:
            raise RecordingAborted(self._reason)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_bytes(path: str | Path, raw: bytes) -> None:
    """Durably stage exact bytes beside the destination, then atomically replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
    )
    try:
        with tmp.open("xb") as fp:
            fp.write(raw)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _read_json_once(path: str | Path, *, label: str) -> tuple[bytes, dict, str]:
    """Read once so parsed content and its digest cannot refer to different revisions."""
    raw = Path(path).read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return raw, payload, _sha256_bytes(raw)


def load_force_calibration_snapshot(
    source_path: str | Path, session_dir: str | Path, calibration_cls,
) -> tuple[object, dict]:  # noqa: ANN001
    """Parse, hash, and snapshot one immutable read of the force calibration.

    The source path is never reopened during this operation, preventing an A->B->A
    replacement from making the parsed curve disagree with its recorded digest.
    """
    source = Path(source_path).resolve()
    raw = source.read_bytes()
    calibration = calibration_cls.from_bytes(raw)
    digest = _sha256_bytes(raw)
    snapshot = (Path(session_dir) / "calibrations" / "force_calibration.json").resolve()
    _atomic_write_bytes(snapshot, raw)
    return calibration, {
        "source_calibration_path": str(source),
        "calibration_path": str(snapshot),
        "calibration_sha256": digest,
    }


def _write_manifest(path: str | Path, manifest: dict) -> None:
    """Atomically replace the lifecycle manifest so a crash never leaves valid-looking JSON."""
    _atomic_write_bytes(path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))


def capture_should_stop(
    stop_file: str | Path | None, *, t0: float, max_duration_s: float, now: float | None = None,
) -> bool:
    """Cooperative frame-boundary stop gate used by the recorder loop."""
    current = time.monotonic() if now is None else float(now)
    if current - float(t0) >= float(max_duration_s):
        return True
    return stop_file is not None and Path(stop_file).is_file()


def publish_capture_ended(
    path: str | Path, *, t0: float, t_end: float, reason: str,
) -> dict:
    """Durably acknowledge that sampling ended so the supervisor can disable teleop now."""
    duration = max(0.0, float(t_end) - float(t0))
    payload = {
        "schema_version": "wam_capture_ended_v1",
        "reason": str(reason),
        "t0_monotonic": float(t0),
        "t_end_monotonic": float(t_end),
        "actual_duration_s": duration,
    }
    _atomic_write_bytes(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    return payload


def _validate_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def capture_release_provenance(
    *, code_release_id: str | None, runtime_release_id: str | None,
    runtime_lock_sha256: str | None,
) -> dict:
    """Bind the raw capture to one code release and one locked runtime.

    Rehearsal callers may omit the whole tuple, but a partial or malformed tuple
    is rejected before the session directory is claimed. Promotion independently
    requires all three identities.
    """
    values = (code_release_id, runtime_release_id, runtime_lock_sha256)
    if not any(value is not None for value in values):
        return {
            "schema_version": CAPTURE_RELEASE_PROVENANCE_SCHEMA_VERSION,
            "code": {"release_id": None},
            "runtime": {"release_id": None, "lock_sha256": None},
        }
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(
            "code release id, runtime release id, and runtime lock SHA-256 "
            "must be supplied together"
        )
    code = str(code_release_id).strip()
    runtime = str(runtime_release_id).strip()
    lock = str(runtime_lock_sha256).strip()
    if CODE_RELEASE_ID_PATTERN.fullmatch(code) is None:
        raise ValueError("code release id must be code_release- followed by 20 lowercase hex")
    if RUNTIME_RELEASE_ID_PATTERN.fullmatch(runtime) is None:
        raise ValueError("runtime release id must be a versioned 20-hex release identity")
    _validate_sha256(lock, label="runtime lock SHA-256")
    return {
        "schema_version": CAPTURE_RELEASE_PROVENANCE_SCHEMA_VERSION,
        "code": {"release_id": code},
        "runtime": {"release_id": runtime, "lock_sha256": lock},
    }


def _required_descriptor_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_descriptor_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string or null")
    return value.strip() or None


def _usb_id(value: object, *, label: str, required: bool) -> int | None:
    if value is None and not required:
        return None
    if type(value) is not int or not 0 < value <= 0xFFFF:
        qualifier = "" if required else " or null"
        raise ValueError(f"{label} must be a 16-bit USB integer{qualifier}")
    return value


def resolve_hall_hardware_descriptor(port: str, *, comports=None) -> dict:  # noqa: ANN001
    """Resolve a serial path to the RP2040's stable pyserial USB descriptor."""
    requested = _required_descriptor_text(port, label="Hall requested port")
    if comports is None:
        from serial.tools.list_ports import comports as list_comports
        comports = list_comports
    ports = list(comports())
    exact = [item for item in ports if str(getattr(item, "device", "")) == requested]
    requested_real = os.path.realpath(requested)
    matches = exact or [
        item for item in ports
        if os.path.realpath(str(getattr(item, "device", ""))) == requested_real
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Hall port {requested!r} was not uniquely found in pyserial list_ports"
        )
    item = matches[0]
    serial = _required_descriptor_text(
        getattr(item, "serial_number", None), label="Hall USB serial",
    )
    vid = _usb_id(getattr(item, "vid", None), label="Hall USB VID", required=True)
    pid = _usb_id(getattr(item, "pid", None), label="Hall USB PID", required=True)
    return {
        "requested_port": requested,
        "device": _required_descriptor_text(
            getattr(item, "device", None), label="Hall resolved device",
        ),
        "usb_serial": serial,
        "usb_vid": vid,
        "usb_pid": pid,
        "manufacturer": _optional_descriptor_text(
            getattr(item, "manufacturer", None), label="Hall manufacturer",
        ),
        "product": _optional_descriptor_text(
            getattr(item, "product", None), label="Hall product",
        ),
        "location": _optional_descriptor_text(
            getattr(item, "location", None), label="Hall USB location",
        ),
    }


_ALSA_CARD_RE = re.compile(r"(?:plughw|hw):(\d+),")
_USB_PORT_PATH_RE = re.compile(r"^\d+-\d+(?:\.\d+)*$")


def resolve_linux_alsa_usb_identity(
    device_name: str, *, sound_root: str | Path = "/sys/class/sound",
) -> dict | None:
    """Best-effort PortAudio ALSA name -> Linux USB parent descriptor mapping."""
    match = _ALSA_CARD_RE.search(str(device_name))
    if match is None:
        return None
    card = int(match.group(1))
    card_path = Path(sound_root) / f"card{card}"
    candidates = [card_path / "device", card_path]
    start = next((candidate for candidate in candidates if candidate.exists()), None)
    if start is None:
        return None
    try:
        resolved = start.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    for node in (resolved, *resolved.parents):
        vendor_path, product_path = node / "idVendor", node / "idProduct"
        if not vendor_path.is_file() or not product_path.is_file():
            continue
        try:
            vid = int(vendor_path.read_text(encoding="utf-8").strip(), 16)
            pid = int(product_path.read_text(encoding="utf-8").strip(), 16)
        except (OSError, ValueError):
            return None
        serial_path = node / "serial"
        try:
            serial = serial_path.read_text(encoding="utf-8").strip() if serial_path.is_file() else None
        except OSError:
            serial = None
        return {
            "usb_serial": serial or None,
            "usb_vid": vid,
            "usb_pid": pid,
            "usb_port_path": node.name,
            "usb_identity_source": (
                "linux_alsa_sysfs" if serial else "linux_alsa_sysfs_topology"
            ),
            "alsa_card": card,
        }
    return None


def resolve_audio_hardware_descriptor(
    device: str | int,
    *,
    expected_usb_vid: int | None = None,
    expected_usb_pid: int | None = None,
    expected_usb_serial: str | None = None,
    query_devices=None,
    query_hostapis=None,
    usb_resolver=None,
) -> dict:  # noqa: ANN001
    """Resolve one PortAudio input, selecting by USB serial when commissioned.

    A PortAudio/ALSA card number is enumeration state, not hardware identity.
    When a USB serial is supplied, every ALSA input is mapped to sysfs first and
    the unique VID/PID/serial match is selected before the stale display name is
    considered.  Zero or multiple matches fail closed and never fall back to a
    name or default input.
    """
    if query_devices is None or query_hostapis is None:
        import sounddevice as sd
        query_devices = query_devices or sd.query_devices
        query_hostapis = query_hostapis or sd.query_hostapis
    devices = [dict(info) for info in query_devices()]
    usb: dict = {}
    unavailable_source = "unavailable_injected_resolver"
    if expected_usb_serial is not None:
        serial = _required_descriptor_text(
            expected_usb_serial, label="commissioned audio USB serial",
        )
        vid = _usb_id(
            expected_usb_vid, label="commissioned audio USB VID", required=True,
        )
        pid = _usb_id(
            expected_usb_pid, label="commissioned audio USB PID", required=True,
        )
        matches: list[tuple[int, dict, str, dict]] = []
        for index, candidate in enumerate(devices):
            if type(candidate.get("max_input_channels")) is not int or int(
                candidate["max_input_channels"]
            ) <= 0:
                continue
            hostapi_index = candidate.get("hostapi")
            if type(hostapi_index) is not int or hostapi_index < 0:
                continue
            try:
                hostapi_name = _required_descriptor_text(
                    dict(query_hostapis(hostapi_index)).get("name"),
                    label="PortAudio hostapi name",
                )
            except (KeyError, TypeError, ValueError):
                continue
            if hostapi_name.casefold() != "alsa":
                continue
            try:
                resolved_usb = (
                    usb_resolver(index, candidate, hostapi_name)
                    if usb_resolver is not None
                    else resolve_linux_alsa_usb_identity(str(candidate.get("name", "")))
                )
                resolved_usb = dict(resolved_usb or {})
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            if (
                resolved_usb.get("usb_vid") == vid
                and resolved_usb.get("usb_pid") == pid
                and resolved_usb.get("usb_serial") == serial
            ):
                matches.append((index, candidate, hostapi_name, resolved_usb))
        if not matches:
            raise ValueError(
                "commissioned USB audio input was not found by VID/PID/serial"
            )
        if len(matches) != 1:
            raise ValueError(
                "commissioned USB audio input is ambiguous by VID/PID/serial"
            )
        selected_index, selected, hostapi_name, usb = matches[0]
    else:
        if type(device) is int:
            if not 0 <= device < len(devices):
                raise ValueError(f"PortAudio input index {device} is out of range")
            selected_index = device
            selected = devices[selected_index]
        else:
            selected = dict(query_devices(device, "input"))
            selected_index_value = selected.get("index")
            if type(selected_index_value) is int:
                candidates = [selected_index_value]
            else:
                signature = (
                    selected.get("name"), selected.get("hostapi"),
                    selected.get("max_input_channels"),
                )
                candidates = [
                    index for index, info in enumerate(devices)
                    if (
                        info.get("name"), info.get("hostapi"),
                        info.get("max_input_channels"),
                    ) == signature
                ]
            if len(candidates) != 1:
                raise ValueError(
                    f"PortAudio selected-device index is ambiguous for {device!r}: {candidates}"
                )
            selected_index = candidates[0]
            if not 0 <= selected_index < len(devices):
                raise ValueError(
                    f"PortAudio selected-device index {selected_index} is out of range"
                )
            selected = devices[selected_index]
        hostapi_index = selected.get("hostapi")
        if type(hostapi_index) is not int or hostapi_index < 0:
            raise ValueError("PortAudio hostapi index must be a non-negative integer")
        hostapi = dict(query_hostapis(hostapi_index))
        hostapi_name = _required_descriptor_text(
            hostapi.get("name"), label="PortAudio hostapi name",
        )
        if usb_resolver is None:
            if sys.platform.startswith("linux") and hostapi_name.casefold() == "alsa":
                usb = dict(resolve_linux_alsa_usb_identity(str(selected.get("name", ""))) or {})
                unavailable_source = "unavailable_linux_alsa_sysfs"
            else:
                unavailable_source = "unavailable_non_linux_alsa"
        else:
            usb = dict(usb_resolver(selected_index, selected, hostapi_name) or {})
    name = _required_descriptor_text(
        selected.get("name"), label="PortAudio device name",
    )
    max_inputs = selected.get("max_input_channels")
    if type(max_inputs) is not int or max_inputs <= 0:
        raise ValueError("PortAudio max_input_channels must be a positive integer")
    serial = _optional_descriptor_text(
        usb.get("usb_serial"), label="audio USB serial",
    )
    vid = _usb_id(usb.get("usb_vid"), label="audio USB VID", required=False)
    pid = _usb_id(usb.get("usb_pid"), label="audio USB PID", required=False)
    if serial is not None and (vid is None or pid is None):
        raise ValueError("audio USB serial requires USB VID and PID")
    port_path = _optional_descriptor_text(
        usb.get("usb_port_path"), label="audio USB port path",
    )
    if port_path is not None and _USB_PORT_PATH_RE.fullmatch(port_path) is None:
        raise ValueError("audio USB port path must be a Linux USB topology path")
    if port_path is not None and (vid is None or pid is None):
        raise ValueError("audio USB port path requires USB VID and PID")
    descriptor = {
        "requested_device": str(device),
        "portaudio_index": selected_index,
        "hostapi_name": hostapi_name,
        "device_name": name,
        "max_input_channels": max_inputs,
        "usb_serial": serial,
        "usb_vid": vid,
        "usb_pid": pid,
        "usb_identity_source": _required_descriptor_text(
            usb.get("usb_identity_source", unavailable_source),
            label="audio USB identity source",
        ),
    }
    if type(usb.get("alsa_card")) is int and usb["alsa_card"] >= 0:
        descriptor["alsa_card"] = usb["alsa_card"]
    if port_path is not None:
        descriptor["usb_port_path"] = port_path
    return descriptor


def validate_sync_capture_mode(value: object) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("sync capture_mode must be a JSON object")
    audio_channel = value.get("audio_channel")
    if type(audio_channel) is not int or audio_channel < 0:
        raise ValueError("sync capture_mode audio_channel must be a non-negative integer")
    force_sha = _validate_sha256(
        value.get("force_calibration_sha256"),
        label="sync capture_mode force_calibration_sha256",
    )
    audio = value.get("audio")
    if not isinstance(audio, Mapping):
        raise ValueError("sync capture_mode audio must be a JSON object")
    requested_device = _required_descriptor_text(
        audio.get("requested_device"), label="sync capture_mode audio requested_device",
    )
    portaudio_index = audio.get("portaudio_index")
    if type(portaudio_index) is not int or portaudio_index < 0:
        raise ValueError(
            "sync capture_mode audio portaudio_index must be a non-negative integer"
        )
    hostapi_name = _required_descriptor_text(
        audio.get("hostapi_name"), label="sync capture_mode audio hostapi_name",
    )
    device_name = _required_descriptor_text(
        audio.get("device_name"), label="sync capture_mode audio device_name",
    )
    max_inputs = audio.get("max_input_channels")
    if type(max_inputs) is not int or max_inputs <= 0:
        raise ValueError(
            "sync capture_mode audio max_input_channels must be a positive integer"
        )
    if audio_channel >= max_inputs:
        raise ValueError(
            "sync capture_mode audio_channel must be less than audio max_input_channels"
        )
    audio_serial = _optional_descriptor_text(
        audio.get("usb_serial"), label="sync capture_mode audio usb_serial",
    )
    audio_vid = _usb_id(
        audio.get("usb_vid"), label="sync capture_mode audio usb_vid", required=False,
    )
    audio_pid = _usb_id(
        audio.get("usb_pid"), label="sync capture_mode audio usb_pid", required=False,
    )
    if audio_serial is not None and (audio_vid is None or audio_pid is None):
        raise ValueError(
            "sync capture_mode audio usb_serial requires usb_vid and usb_pid"
        )
    audio_port_path = _optional_descriptor_text(
        audio.get("usb_port_path"),
        label="sync capture_mode audio usb_port_path",
    )
    if (audio_port_path is not None
            and _USB_PORT_PATH_RE.fullmatch(audio_port_path) is None):
        raise ValueError(
            "sync capture_mode audio usb_port_path must be a Linux USB topology path"
        )
    if audio_port_path is not None and (audio_vid is None or audio_pid is None):
        raise ValueError(
            "sync capture_mode audio usb_port_path requires usb_vid and usb_pid"
        )
    audio_result = {
        "requested_device": requested_device,
        "portaudio_index": portaudio_index,
        "hostapi_name": hostapi_name,
        "device_name": device_name,
        "max_input_channels": max_inputs,
        "usb_serial": audio_serial,
        "usb_vid": audio_vid,
        "usb_pid": audio_pid,
        "usb_identity_source": _required_descriptor_text(
            audio.get("usb_identity_source"),
            label="sync capture_mode audio usb_identity_source",
        ),
    }
    if "alsa_card" in audio:
        alsa_card = audio.get("alsa_card")
        if type(alsa_card) is not int or alsa_card < 0:
            raise ValueError("sync capture_mode audio alsa_card must be non-negative integer")
        audio_result["alsa_card"] = alsa_card
    if audio_port_path is not None:
        audio_result["usb_port_path"] = audio_port_path

    hall = value.get("hall")
    if not isinstance(hall, Mapping):
        raise ValueError("sync capture_mode hall must be a JSON object")
    hall_result = {
        "requested_port": _required_descriptor_text(
            hall.get("requested_port"), label="sync capture_mode hall requested_port",
        ),
        "device": _required_descriptor_text(
            hall.get("device"), label="sync capture_mode hall device",
        ),
        "usb_serial": _required_descriptor_text(
            hall.get("usb_serial"), label="sync capture_mode hall usb_serial",
        ),
        "usb_vid": _usb_id(
            hall.get("usb_vid"), label="sync capture_mode hall usb_vid", required=True,
        ),
        "usb_pid": _usb_id(
            hall.get("usb_pid"), label="sync capture_mode hall usb_pid", required=True,
        ),
        "manufacturer": _optional_descriptor_text(
            hall.get("manufacturer"), label="sync capture_mode hall manufacturer",
        ),
        "product": _optional_descriptor_text(
            hall.get("product"), label="sync capture_mode hall product",
        ),
        "location": _optional_descriptor_text(
            hall.get("location"), label="sync capture_mode hall location",
        ),
    }
    return {
        "audio_channel": audio_channel,
        "force_calibration_sha256": force_sha,
        "audio": audio_result,
        "hall": hall_result,
    }


def build_sync_capture_mode(
    *, audio_device: str | int, audio_channel: int, hall_port: str,
    force_calibration_sha256: str, hall_resolver=None, audio_resolver=None,
    require_audio_usb_serial: bool = False,
) -> dict:  # noqa: ANN001
    hall = (hall_resolver or resolve_hall_hardware_descriptor)(hall_port)
    audio = (audio_resolver or resolve_audio_hardware_descriptor)(audio_device)
    mode = validate_sync_capture_mode({
        "audio_channel": audio_channel,
        "force_calibration_sha256": force_calibration_sha256,
        "audio": audio,
        "hall": hall,
    })
    if (require_audio_usb_serial
            and mode["audio"]["usb_serial"] is None
            and mode["audio"].get("usb_port_path") is None):
        raise ValueError(
            "arm-connected collection requires a stable audio USB serial or "
            "fixed Linux USB topology; PortAudio name/index alone is not sufficient"
        )
    return mode


def sync_capture_endpoints(capture_mode: object) -> tuple[int, str]:
    """Return immutable endpoints selected while resolving the hardware descriptor."""
    mode = validate_sync_capture_mode(capture_mode)
    return mode["audio"]["portaudio_index"], mode["hall"]["device"]


def _match_sync_capture_mode(calibration: dict, runtime: dict) -> dict:
    def same(component: str, field: str, left: object, right: object) -> None:
        if left != right:
            raise ValueError(
                f"sync capture_mode {component} {field}={left!r} does not match "
                f"runtime {right!r}"
            )

    same("root", "audio_channel", calibration["audio_channel"], runtime["audio_channel"])
    same(
        "root", "force_calibration_sha256",
        calibration["force_calibration_sha256"], runtime["force_calibration_sha256"],
    )
    for field in ("usb_serial", "usb_vid", "usb_pid"):
        same("hall", field, calibration["hall"][field], runtime["hall"][field])
    for field in ("hostapi_name", "max_input_channels"):
        same("audio", field, calibration["audio"][field], runtime["audio"][field])

    cal_serial = calibration["audio"]["usb_serial"]
    runtime_serial = runtime["audio"]["usb_serial"]
    if cal_serial is not None or runtime_serial is not None:
        for field in ("usb_serial", "usb_vid", "usb_pid"):
            same("audio", field, calibration["audio"][field], runtime["audio"][field])
        audio_usb = "matched"
    else:
        cal_port = calibration["audio"].get("usb_port_path")
        runtime_port = runtime["audio"].get("usb_port_path")
        if cal_port is not None or runtime_port is not None:
            for field in ("usb_vid", "usb_pid", "usb_port_path"):
                same(
                    "audio", field,
                    calibration["audio"].get(field), runtime["audio"].get(field),
                )
            audio_usb = "matched_fixed_topology"
        else:
            for field in ("usb_vid", "usb_pid", "portaudio_index", "device_name"):
                same("audio", field, calibration["audio"][field], runtime["audio"][field])
            audio_usb = "unavailable_portaudio_fallback"
    return {
        "hall_usb": "matched",
        "audio_usb": audio_usb,
        "audio_portaudio_index": (
            "matched" if calibration["audio"]["portaudio_index"]
            == runtime["audio"]["portaudio_index"] else "remapped"
        ),
    }


def _validate_camera_capture_mode(value: object) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("camera latency capture_mode must be a JSON object")
    backend = value.get("backend")
    camera_id = value.get("camera_id")
    width = value.get("width")
    height = value.get("height")
    fps = value.get("fps")
    time_source = value.get("time_source")
    color_format = value.get("color_format")
    depth_mode = value.get("depth_mode")
    align_mode = value.get("align_mode")
    frame_aggregate_mode = value.get("frame_aggregate_mode")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("camera latency capture_mode backend must be a non-empty string")
    backend = backend.strip().lower()
    expected_time_source = CAMERA_TIME_SOURCE_BY_BACKEND.get(backend)
    if expected_time_source is None:
        raise ValueError(
            f"camera latency capture_mode backend {backend!r} is not a physical backend"
        )
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError("camera latency capture_mode camera_id must be a non-empty string")
    if type(width) is not int or width <= 0:
        raise ValueError("camera latency capture_mode width must be a positive integer")
    if type(height) is not int or height <= 0:
        raise ValueError("camera latency capture_mode height must be a positive integer")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ValueError("camera latency capture_mode fps must be a finite positive number")
    fps_value = float(fps)
    if not np.isfinite(fps_value) or fps_value <= 0:
        raise ValueError("camera latency capture_mode fps must be a finite positive number")
    if time_source != expected_time_source:
        raise ValueError(
            "camera latency capture_mode time_source must match backend semantics: "
            f"expected {expected_time_source!r}, got {time_source!r}"
        )
    if not isinstance(color_format, str) or not color_format.strip():
        raise ValueError("camera latency capture_mode color_format must be a non-empty string")
    color_format = color_format.strip()
    if backend == "orbbec":
        if color_format not in ORBBEC_COLOR_FORMATS:
            raise ValueError(
                f"camera latency capture_mode color_format {color_format!r} is unsupported"
            )
        if not isinstance(depth_mode, Mapping):
            raise ValueError("camera latency capture_mode depth_mode must be a JSON object")
        depth_width, depth_height = depth_mode.get("width"), depth_mode.get("height")
        depth_fps, depth_format = depth_mode.get("fps"), depth_mode.get("format")
        if type(depth_width) is not int or depth_width <= 0:
            raise ValueError("camera latency capture_mode depth_mode width must be positive integer")
        if type(depth_height) is not int or depth_height <= 0:
            raise ValueError("camera latency capture_mode depth_mode height must be positive integer")
        if isinstance(depth_fps, bool) or not isinstance(depth_fps, (int, float)):
            raise ValueError("camera latency capture_mode depth_mode fps must be finite and positive")
        depth_fps = float(depth_fps)
        if not np.isfinite(depth_fps) or depth_fps <= 0:
            raise ValueError("camera latency capture_mode depth_mode fps must be finite and positive")
        if not isinstance(depth_format, str) or not depth_format.strip():
            raise ValueError("camera latency capture_mode depth_mode format must be non-empty")
        depth_mode = {
            "width": depth_width, "height": depth_height, "fps": depth_fps,
            "format": depth_format.strip(),
        }
        if align_mode not in ORBBEC_ALIGN_MODES:
            raise ValueError(
                f"camera latency capture_mode align_mode must be one of "
                f"{sorted(ORBBEC_ALIGN_MODES)}"
            )
        if frame_aggregate_mode != "full_frame_require":
            raise ValueError(
                "camera latency capture_mode frame_aggregate_mode must be "
                "full_frame_require for Orbbec"
            )
    else:
        if color_format != "opencv_decoded_bgr":
            raise ValueError(
                "camera latency capture_mode color_format for OpenCV must be "
                "'opencv_decoded_bgr'"
            )
        if depth_mode is not None:
            raise ValueError("camera latency capture_mode depth_mode must be null for OpenCV")
        if align_mode != "disabled":
            raise ValueError("camera latency capture_mode align_mode must be disabled for OpenCV")
        if frame_aggregate_mode != "not_applicable":
            raise ValueError(
                "camera latency capture_mode frame_aggregate_mode must be not_applicable for OpenCV"
            )
    return {
        "backend": backend,
        "camera_id": camera_id.strip(),
        "width": width,
        "height": height,
        "fps": fps_value,
        "time_source": expected_time_source,
        "color_format": color_format,
        "depth_mode": depth_mode,
        "align_mode": align_mode,
        "frame_aggregate_mode": frame_aggregate_mode,
    }


def _require_matching_capture_mode(actual: dict, expected: Mapping[str, object] | None) -> None:
    for field, expected_value in (expected or {}).items():
        if expected_value is None:
            continue
        if field not in actual or actual[field] != expected_value:
            raise ValueError(
                f"capture_mode {field}={actual.get(field)!r} does not match "
                f"collection value {expected_value!r}"
            )


def load_sync_calibration(
    path: str | Path, *, expected_capture_mode: Mapping[str, object] | None = None,
) -> dict:
    """Validate the persisted tap-offset sign and return manifest-ready metadata."""
    path = Path(path)
    _, payload, digest = _read_json_once(path, label="sync calibration")
    if payload.get("schema_version") != "wam_sync_v1":
        raise ValueError("sync calibration schema_version must be wam_sync_v1")
    if payload.get("offset_definition") != "force_t_minus_audio_t":
        raise ValueError("sync calibration offset_definition must be force_t_minus_audio_t")
    try:
        offset_ms = float(payload["audio_to_force_offset_ms"])
        offset_std_ms = float(payload["offset_std_ms"])
        n_matched = int(payload["n_matched"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("sync calibration quality metadata is incomplete") from exc
    if (not np.isfinite(offset_ms) or abs(offset_ms) > 250.0
            or not np.isfinite(offset_std_ms) or not 0 <= offset_std_ms <= 5.0
            or n_matched < 4 or payload.get("unique_matches") is not True
            or payload.get("measurement_method") != "physical_diaphragm_taps"):
        raise ValueError("sync calibration quality gate failed")
    capture_mode = validate_sync_capture_mode(payload.get("capture_mode"))
    runtime_capture_mode = (
        validate_sync_capture_mode(expected_capture_mode)
        if expected_capture_mode is not None else None
    )
    identity_match = (
        _match_sync_capture_mode(capture_mode, runtime_capture_mode)
        if runtime_capture_mode is not None else None
    )
    metadata = {
        "calibration_path": str(path.resolve()),
        "calibration_sha256": digest,
        "offset_definition": "force_t_minus_audio_t",
        "offset_units": "ms",
        "audio_to_force_offset_ms": offset_ms,
        "offset_std_ms": offset_std_ms,
        "n_matched": n_matched,
        "unique_matches": True,
        "measurement_method": "physical_diaphragm_taps",
        "capture_mode": capture_mode,
        "applied": False,
    }
    if runtime_capture_mode is not None:
        metadata["runtime_capture_mode"] = runtime_capture_mode
        metadata["identity_match"] = identity_match
    return metadata


def load_camera_latency_calibration(
    path: str | Path, *, expected_capture_mode: Mapping[str, object] | None = None,
) -> dict:
    path = Path(path)
    _, payload, digest = _read_json_once(path, label="camera latency calibration")
    if (payload.get("schema_version") != "wam_camera_latency_v1"
            or payload.get("latency_definition") != "host_arrival_minus_exposure"):
        raise ValueError("camera latency calibration schema/definition is invalid")
    try:
        latency_ms = float(payload["latency_ms"])
        latency_std_ms = float(payload["latency_std_ms"])
        n_samples = int(payload["n_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("camera latency calibration quality metadata is incomplete") from exc
    if (not np.isfinite(latency_ms) or not 0 < latency_ms <= 500.0
            or not np.isfinite(latency_std_ms) or not 0 <= latency_std_ms <= 5.0
            or n_samples < 20
            or payload.get("measurement_method")
            not in ("hardware_led_flash", "device_timer_host_sync")):
        raise ValueError("camera latency calibration quality gate failed")
    capture_mode = _validate_camera_capture_mode(payload.get("capture_mode"))
    _require_matching_capture_mode(capture_mode, expected_capture_mode)
    return {
        "calibration_path": str(path.resolve()),
        "calibration_sha256": digest,
        "latency_definition": "host_arrival_minus_exposure",
        "latency_ms": latency_ms,
        "latency_std_ms": latency_std_ms,
        "n_samples": n_samples,
        "measurement_method": str(payload.get("measurement_method")),
        "capture_mode": capture_mode,
        "applied": False,
    }


class _ManifestLifecycle:
    """Persist `aborted` automatically unless the recorder explicitly completes."""

    def __init__(self, path: str | Path, manifest: dict) -> None:
        self.path = Path(path)
        self.manifest = manifest

    def __enter__(self) -> "_ManifestLifecycle":
        _write_manifest(self.path, self.manifest)
        return self

    def recording(self, t0: float) -> None:
        self.manifest.update(status="recording", t0_monotonic=t0)
        _write_manifest(self.path, self.manifest)

    def ready(self, readiness: dict) -> None:
        self.manifest.update(status="ready", readiness=readiness)
        _write_manifest(self.path, self.manifest)

    def sensors_complete(self) -> None:
        self.manifest.update(
            status="sensors_complete", training_eligible=False, abort_reason=None,
        )
        _write_manifest(self.path, self.manifest)

    def __exit__(self, exc_type, exc, traceback) -> bool:  # noqa: ANN001
        if exc is not None:
            self.manifest.update(
                status="aborted", training_eligible=False, abort_reason=str(exc),
            )
            _write_manifest(self.path, self.manifest)
        elif self.manifest.get("status") != "sensors_complete":
            self.manifest.update(
                status="aborted", training_eligible=False,
                abort_reason="recorder exited without completing finalization",
            )
            _write_manifest(self.path, self.manifest)
        return False


def bundled_hall_calibration_seed() -> Path:
    """Versioned seed only; commissioning must copy it into persistent linbox config."""
    return Path(ROOT) / "mechanical_sensor" / "force_feedback" / \
        "force_feedback_calibration_veo6_tared.json"


def default_hall_calibration() -> Path:
    """Select the commissioned linbox calibration, with an explicit env override first."""
    override = os.environ.get("WAM_HALL_CALIBRATION")
    if override:
        return Path(override).expanduser()
    home = Path(os.environ.get("HOME", str(Path.home()))).expanduser()
    return home / "wam_data" / "config" / "calibration" / \
        "force_feedback_calibration_veo6_tared.json"


def resolve_hall_calibration(selected: str | Path | None) -> Path:
    """Resolve one real, regular calibration file or fail before acquisition starts."""
    candidate = Path(selected).expanduser() if selected is not None else default_hall_calibration()
    if not candidate.exists():
        raise FileNotFoundError(f"Hall calibration does not exist: {candidate}")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"Hall calibration must be a real regular file: {candidate}")
    return candidate.resolve()


def validate_force_calibration(calibration, *, allow_uncalibrated: bool) -> None:  # noqa: ANN001
    """Refuse clinical/IL collection with a provisional force curve by default."""
    if not bool(getattr(calibration, "calibrated", False)) and not allow_uncalibrated:
        raise RuntimeError(
            "force calibration has calibrated=false; bench-calibrate it or pass "
            "--allow-uncalibrated-force for a non-training rehearsal"
        )


def required_stream_failures(streams: dict[str, object], required: set[str]) -> list[str]:
    """Return missing required acquisition streams in deterministic order."""
    return sorted(required.difference(streams))


def validate_collection_mode(
    arm_side: str | None, allow_uncalibrated_force: bool, allow_missing_streams: bool,
    *, camera_backend: str | None = None, camera_id: str | None = None,
) -> None:
    if arm_side is not None and (allow_uncalibrated_force or allow_missing_streams):
        raise ValueError("arm-connected collection forbids all rehearsal overrides")
    if arm_side is not None and camera_backend == "fake":
        raise ValueError("arm-connected collection forbids the synthetic camera backend")
    if arm_side is not None and camera_backend != "orbbec":
        raise ValueError("arm-connected collection requires the serial-verifiable Orbbec backend")
    if arm_side is not None and (not isinstance(camera_id, str) or not camera_id.strip()):
        raise ValueError("arm-connected collection requires an explicit stable camera_id")


def validate_capture_barriers(
    arm_side: str | None, ready_file: str | Path | None, start_file: str | Path | None,
) -> None:
    if arm_side is not None and (ready_file is None or start_file is None):
        raise ValueError("arm-connected collection requires both ready and start barrier files")


def validate_timing_inputs(
    arm_side: str | None, sync_cal: str | Path | None,
    camera_latency_cal: str | Path | None,
) -> None:
    if arm_side is not None and (sync_cal is None or camera_latency_cal is None):
        raise ValueError("arm-connected collection requires sync and camera latency calibrations")


def readiness_failures(force_samples: list, audio_level, camera_frame) -> list[str]:  # noqa: ANN001
    failures: list[str] = []
    if len(force_samples) < 5:
        failures.append("force:insufficient_samples")
    else:
        unsafe = next((str(sample.status) for sample in force_samples
                       if str(sample.status) not in ("OK", "WARN")), None)
        if unsafe:
            failures.append(f"force:{unsafe}")
        identities = {
            (getattr(sample, "device_t_ms", None), getattr(sample, "t", None))
            for sample in force_samples
        }
        if len(identities) < 5:
            failures.append("force:not_advancing")
    if audio_level is None or getattr(audio_level, "status", None) != "OK":
        failures.append(f"audio:{getattr(audio_level, 'status', 'NO_DATA')}")
    if (camera_frame is None or getattr(camera_frame, "status", None) != "OK"
            or getattr(camera_frame, "frame", None) is None):
        failures.append(f"camera:{getattr(camera_frame, 'status', 'NO_DATA')}")
    return failures


def runtime_stream_failures(
    audio_level, camera_frame, *, require_depth: bool = False,
) -> list[str]:  # noqa: ANN001
    """Detect a stream that was ready initially but stopped during capture."""
    failures: list[str] = []
    if audio_level is None or getattr(audio_level, "status", None) != "OK":
        failures.append(f"audio:{getattr(audio_level, 'status', 'NO_DATA')}")
    if (camera_frame is None or getattr(camera_frame, "status", None) != "OK"
            or getattr(camera_frame, "frame", None) is None):
        failures.append(f"camera:{getattr(camera_frame, 'status', 'NO_DATA')}")
    elif require_depth and getattr(camera_frame, "depth", None) is None:
        failures.append("camera:NO_DEPTH")
    return failures


def camera_time_row(frame) -> tuple[int, float, float | None, str]:  # noqa: ANN001
    """Preserve receipt and device clocks for offline exposure-time fitting."""
    return (
        int(frame.index), float(frame.t), getattr(frame, "device_t_ms", None),
        str(getattr(frame, "time_source", "host_arrival")),
    )


def validate_camera_frame_mode(frame, timing: dict, stream=None) -> dict:  # noqa: ANN001
    """Bind live frame/timestamp semantics and negotiated mode to the calibration."""
    image = getattr(frame, "frame", None)
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        raise RecordingAborted("camera frame mode cannot be validated without an image shape")
    actual_height, actual_width = int(shape[0]), int(shape[1])
    mode = timing["capture_mode"]
    expected_width, expected_height = int(mode["width"]), int(mode["height"])
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise RecordingAborted(
            "camera frame mode calibration expects "
            f"{expected_width}x{expected_height}, got {actual_width}x{actual_height}"
        )
    depth = getattr(frame, "depth", None)
    if mode["depth_mode"] is not None:
        if depth is None or getattr(depth, "ndim", 0) != 2:
            raise RecordingAborted("camera frame is missing required aligned depth")
        expected_depth_shape = (
            (expected_height, expected_width)
            if mode["align_mode"] == "software_depth_to_color"
            else (int(mode["depth_mode"]["height"]), int(mode["depth_mode"]["width"]))
        )
        if tuple(int(value) for value in depth.shape) != expected_depth_shape:
            raise RecordingAborted(
                f"camera depth shape {tuple(depth.shape)!r} does not match "
                f"pipeline {expected_depth_shape!r}"
            )
    actual_source = str(getattr(frame, "time_source", ""))
    expected_source = str(mode["time_source"])
    if actual_source != expected_source:
        raise RecordingAborted(
            f"camera time_source expects {expected_source!r}, got {actual_source!r}"
        )
    device_t_ms = getattr(frame, "device_t_ms", None)
    if mode["backend"] == "orbbec" and (
        device_t_ms is None or not np.isfinite(float(device_t_ms))
    ):
        raise RecordingAborted("orbbec camera frame is missing its device timestamp")
    validation = {
        "width": "matched_frame_shape",
        "height": "matched_frame_shape",
        "fps": "not_verified_by_recorder",
        "time_source": "matched_frame_metadata",
        "device_timestamp": "present" if device_t_ms is not None else "not_provided_by_backend",
        "depth": "matched_pipeline_shape" if depth is not None else "not_applicable",
    }
    if stream is not None:
        negotiated = getattr(stream, "negotiated_mode", None)
        if not isinstance(negotiated, Mapping):
            raise RecordingAborted("camera backend did not report its negotiated capture mode")
        if (int(negotiated.get("width", -1)), int(negotiated.get("height", -1))) != (
            expected_width, expected_height,
        ):
            raise RecordingAborted(
                "camera negotiated dimensions do not match latency calibration: "
                f"{negotiated.get('width')}x{negotiated.get('height')} vs "
                f"{expected_width}x{expected_height}"
            )
        try:
            negotiated_fps = float(negotiated["fps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordingAborted("camera backend did not report negotiated FPS") from exc
        expected_fps = float(mode["fps"])
        if (not np.isfinite(negotiated_fps)
                or abs(negotiated_fps - expected_fps) > max(0.5, expected_fps * 0.02)):
            raise RecordingAborted(
                f"camera negotiated FPS {negotiated_fps!r} does not match "
                f"latency calibration {expected_fps!r}"
            )
        validation["fps"] = "matched_negotiated_mode"
        for field in ("color_format", "depth_mode", "align_mode", "frame_aggregate_mode"):
            if negotiated.get(field) != mode[field]:
                raise RecordingAborted(
                    f"camera negotiated {field}={negotiated.get(field)!r} does not match "
                    f"latency calibration {mode[field]!r}"
                )
        validation["pipeline"] = "matched_negotiated_pipeline"
        validation["negotiated_mode"] = dict(negotiated)
        if mode["backend"] == "orbbec":
            actual_camera_id = getattr(stream, "device_identity", None)
            if actual_camera_id != mode["camera_id"]:
                raise RecordingAborted(
                    f"Orbbec serial {actual_camera_id!r} does not match latency calibration "
                    f"{mode['camera_id']!r}"
                )
            validation["camera_id"] = "matched_device_serial"
    return validation


def observation_timestamp(frame, *, clock=None) -> float:  # noqa: ANN001
    """Stamp a fused row after reading its camera reference, never before it."""
    now = float((clock or time.monotonic)())
    if frame is not None and getattr(frame, "frame", None) is not None:
        camera_t = float(getattr(frame, "t"))
        if not np.isfinite(camera_t) or camera_t > now + 1e-9:
            raise RecordingAborted(
                f"future camera reference {camera_t:.9f} exceeds observation time {now:.9f}"
            )
    return now


def video_writer_fps(camera_stream, camera_timing: dict | None, control_hz: float) -> float:  # noqa: ANN001
    """Use the negotiated camera cadence for the container, never the robot loop rate."""
    candidates = []
    if camera_timing:
        candidates.append((camera_timing.get("capture_mode") or {}).get("fps"))
    negotiated = getattr(camera_stream, "negotiated_mode", None)
    if isinstance(negotiated, Mapping):
        candidates.append(negotiated.get("fps"))
    candidates.append(control_hz)
    for value in candidates:
        try:
            fps = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fps) and fps > 0:
            return fps
    raise RecordingAborted("camera/video FPS is missing or invalid")


def open_video_writer(writer_factory, path, *, fourcc: int, fps: float, size: tuple[int, int]):
    """Open and immediately prove the encoder/container is writable."""
    writer = writer_factory(str(path), int(fourcc), float(fps), tuple(size))
    if writer is None or not writer.isOpened():
        try:
            writer.release()
        except (AttributeError, RuntimeError):
            pass
        raise RecordingAborted(
            f"video writer failed to open {path} at {size[0]}x{size[1]}@{fps:g}fps"
        )
    return writer


def _fsync_directory(path: str | Path) -> None:
    directory = Path(path)
    fd = os.open(str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: str | Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DurableCsv:
    """Append-only CSV with bounded memory and periodic stable-storage checkpoints."""

    def __init__(
        self, path: str | Path, header: str, *, fsync_interval_s: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("x", encoding="utf-8", newline="")
        self._interval = max(0.0, float(fsync_interval_s))
        self._last_fsync = float("-inf")
        self.count = 0
        self._closed = False
        self._fp.write(header)
        self.checkpoint(force=True)
        _fsync_directory(self.path.parent)

    def append(self, row: str) -> None:
        if self._closed:
            raise RuntimeError(f"timeline is closed: {self.path}")
        if not row.endswith("\n"):
            raise ValueError("durable CSV rows must end with a newline")
        self._fp.write(row)
        self.count += 1
        self.checkpoint()

    def checkpoint(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_fsync < self._interval:
            return
        self._fp.flush()
        os.fsync(self._fp.fileno())
        self._last_fsync = now

    def close(self) -> None:
        if self._closed:
            return
        self.checkpoint(force=True)
        self._fp.close()
        self._closed = True

    def sha256(self) -> str:
        if not self._closed:
            raise RuntimeError("timeline must be closed before hashing")
        return _sha256_file(self.path)

    def __enter__(self) -> "DurableCsv":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class BoundedCaptureWriter:
    """Never-overwrite FIFO between the one-slot camera and durable artifact I/O."""

    def __init__(self, processor, *, capacity: int = 30, name: str = "formal-capture") -> None:  # noqa: ANN001
        if int(capacity) <= 0:
            raise ValueError("capture FIFO capacity must be positive")
        self._processor = processor
        self._queue: queue.Queue = queue.Queue(maxsize=int(capacity))
        self._sentinel = object()
        self._name = str(name)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._accepting = False
        self._t0: float | None = None
        self._worker_error: BaseException | None = None
        self._fatal_reason: str | None = None
        self._accepted = 0
        self._processed = 0
        self._overflow = 0
        self._missing = 0
        self._boundary_rejections = 0
        self._max_queue_depth = 0
        self._last_submitted: int | None = None
        self._last_processed: int | None = None
        self._first_host_t: float | None = None
        self._last_host_t: float | None = None
        self._camera_backend: str | None = None
        self._camera_boundary_index: int | None = None
        self._camera_boundary_t: float | None = None
        self._first_expected_index: int | None = None
        self._first_accepted_index: int | None = None
        self._start_boundary_verified: bool | None = None

    def start(
        self, *, t0: float, camera_boundary_index: int | None = None,
        camera_boundary_t: float | None = None, camera_backend: str | None = None,
    ) -> "BoundedCaptureWriter":
        start = float(t0)
        if not np.isfinite(start):
            raise ValueError("capture FIFO Start boundary must be finite")
        boundary_index = (
            None if camera_boundary_index is None else int(camera_boundary_index)
        )
        boundary_t = None if camera_boundary_t is None else float(camera_boundary_t)
        if boundary_t is not None and not np.isfinite(boundary_t):
            raise ValueError("camera boundary timestamp must be finite")
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("capture FIFO has already been started")
            self._t0 = start
            self._camera_backend = None if camera_backend is None else str(camera_backend)
            self._camera_boundary_index = boundary_index
            self._camera_boundary_t = boundary_t
            self._first_expected_index = (
                None if boundary_index is None else boundary_index + 1
            )
            self._start_boundary_verified = None if boundary_index is None else False
            self._accepting = True
            self._thread = threading.Thread(
                target=self._run, name=self._name, daemon=True,
            )
            self._thread.start()
        return self

    def submit(self, frame) -> None:  # noqa: ANN001
        index = int(getattr(frame, "index"))
        host_t = float(getattr(frame, "t"))
        with self._lock:
            if not self._accepting:
                raise RecordingAborted(
                    self._fatal_reason or "capture writer is outside its recording interval"
                )
            if self._worker_error is not None:
                self._accepting = False
                self._fatal_reason = f"capture writer failed: {self._worker_error}"
                raise RecordingAborted(self._fatal_reason)
            if not np.isfinite(host_t) or self._t0 is None or host_t < self._t0:
                self._boundary_rejections += 1
                self._accepting = False
                self._fatal_reason = (
                    f"camera frame {index} at {host_t:.9f} predates Start boundary "
                    f"{self._t0:.9f}"
                )
                raise RecordingAborted(self._fatal_reason)
            if self._last_submitted is None and self._first_expected_index is not None:
                if index != self._first_expected_index:
                    missing = max(0, index - self._first_expected_index)
                    self._missing += missing
                    self._accepting = False
                    self._fatal_reason = (
                        f"camera Start boundary gap {self._camera_boundary_index}->{index}; "
                        f"missing {missing} frame(s), expected {self._first_expected_index}"
                    )
                    raise RecordingAborted(self._fatal_reason)
            if self._last_submitted is not None and index != self._last_submitted + 1:
                missing = max(0, index - self._last_submitted - 1)
                self._missing += missing
                self._accepting = False
                self._fatal_reason = (
                    f"camera frame index gap {self._last_submitted}->{index}; "
                    f"missing {missing} frame(s)"
                )
                raise RecordingAborted(self._fatal_reason)
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                self._overflow += 1
                self._accepting = False
                self._fatal_reason = (
                    f"capture writer FIFO overflow at camera frame {index}; "
                    "accepted frames were not overwritten"
                )
                raise RecordingAborted(self._fatal_reason) from None
            self._accepted += 1
            self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
            self._last_submitted = index
            if self._first_accepted_index is None:
                self._first_accepted_index = index
                if self._first_expected_index is not None:
                    self._start_boundary_verified = True
            if self._first_host_t is None:
                self._first_host_t = host_t
            self._last_host_t = host_t

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._sentinel:
                    return
                if self._worker_error is not None:
                    continue
                try:
                    self._processor(item)
                except BaseException as exc:
                    with self._lock:
                        self._worker_error = exc
                        self._accepting = False
                        self._fatal_reason = f"capture writer failed: {type(exc).__name__}: {exc}"
                    continue
                with self._lock:
                    self._processed += 1
                    self._last_processed = int(getattr(item, "index"))
            finally:
                self._queue.task_done()

    def _stop_thread(self, *, timeout_s: float) -> None:
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RecordingAborted("capture writer did not drain before its timeout")
            try:
                self._queue.put(self._sentinel, timeout=min(0.1, remaining))
                break
            except queue.Full:
                continue
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            raise RecordingAborted("capture writer did not stop after draining")

    def finish(self, *, t_end: float, timeout_s: float = 120.0) -> dict:
        end = float(t_end)
        with self._lock:
            self._accepting = False
            if (
                not np.isfinite(end) or self._t0 is None or end < self._t0
                or (self._last_host_t is not None and self._last_host_t > end + 1e-9)
            ):
                self._fatal_reason = "capture writer End boundary is invalid"
        self._stop_thread(timeout_s=timeout_s)
        status = self.status()
        if self._fatal_reason is not None:
            raise RecordingAborted(self._fatal_reason)
        if self._worker_error is not None:
            raise RecordingAborted(f"capture writer failed: {self._worker_error}")
        if status["processed_frames"] != status["accepted_frames"]:
            raise RecordingAborted(
                "capture writer drain was incomplete: "
                f"{status['processed_frames']} != {status['accepted_frames']}"
            )
        status["t_end_monotonic"] = end
        status["drained"] = True
        return status

    def abort(self) -> None:
        with self._lock:
            self._accepting = False
        try:
            self._stop_thread(timeout_s=5.0)
        except RecordingAborted:
            # The lifecycle manifest remains fail-closed. A daemon worker may be
            # reaped with the process rather than blocking emergency shutdown.
            pass

    def status(self) -> dict:
        with self._lock:
            return {
                "capacity": self._queue.maxsize,
                "accepted_frames": self._accepted,
                "processed_frames": self._processed,
                "overflow_frames": self._overflow,
                "missing_camera_frames": self._missing,
                "boundary_rejections": self._boundary_rejections,
                "sequence_policy": (
                    "strict_contiguous_camera_index"
                    if self._camera_boundary_index is not None
                    else "strict_contiguous_after_first_accepted_index"
                ),
                "camera_backend": self._camera_backend,
                "start_boundary_frame_index": self._camera_boundary_index,
                "start_boundary_frame_t": self._camera_boundary_t,
                "first_expected_frame_index": self._first_expected_index,
                "first_accepted_frame_index": self._first_accepted_index,
                "start_boundary_verified": self._start_boundary_verified,
                "max_queue_depth": self._max_queue_depth,
                "queue_depth": self._queue.qsize(),
                "available_queue_slots": max(
                    0, self._queue.maxsize - self._queue.qsize(),
                ),
                "last_submitted_frame_index": self._last_submitted,
                "last_processed_frame_index": self._last_processed,
                "first_host_t": self._first_host_t,
                "last_host_t": self._last_host_t,
                "worker_error": (
                    None if self._worker_error is None
                    else f"{type(self._worker_error).__name__}: {self._worker_error}"
                ),
                "fatal_reason": self._fatal_reason,
            }


class _DurableJsonl:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd = os.open(
            str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_APPEND, 0o640,
        )

    def append(self, payload: dict) -> None:
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        view = memoryview(raw)
        while view:
            written = os.write(self._fd, view)
            if written <= 0:
                raise OSError("video segment journal write made no progress")
            view = view[written:]
        os.fsync(self._fd)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _read_segment_journal(path: Path) -> list[dict]:
    if not path.is_file() or path.is_symlink():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordingAborted(
                    f"RGB segment journal is corrupt at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise RecordingAborted(
                    f"RGB segment journal line {line_number} is not an object"
                )
            records.append(record)
    return records


def _default_video_writer(path: Path, fps: float, size: tuple[int, int]):
    import cv2

    return cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), tuple(size),
    )


def _default_video_validator(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RecordingAborted(f"video is not decodable: {path}")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise RecordingAborted(f"video decoded an invalid frame: {path}")
            count += 1
    finally:
        capture.release()
    return count


def _default_video_stitcher(
    paths: list[Path], output: Path, fps: float, size: tuple[int, int],
) -> int:
    import cv2

    writer = _default_video_writer(output, fps, size)
    if writer is None or not writer.isOpened():
        try:
            writer.release()
        except (AttributeError, RuntimeError):
            pass
        raise RecordingAborted(f"final RGB video writer failed to open: {output}")
    count = 0
    try:
        for path in paths:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise RecordingAborted(f"sealed RGB segment is not decodable: {path}")
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame is None or tuple(frame.shape[:2][::-1]) != tuple(size):
                        raise RecordingAborted(f"sealed RGB segment changed shape: {path}")
                    writer.write(frame)
                    count += 1
            finally:
                capture.release()
    finally:
        writer.release()
    return count


class SegmentedVideoRecorder:
    """Recoverable MP4 segments with one compatible, exactly validated rgb.mp4 at End."""

    def __init__(
        self, session_dir: str | Path, *, fps: float, size: tuple[int, int],
        segment_frames: int | None = None, writer_factory=None,
        video_validator=None, video_stitcher=None,
    ) -> None:  # noqa: ANN001
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.fps = float(fps)
        self.size = (int(size[0]), int(size[1]))
        if (
            not np.isfinite(self.fps) or self.fps <= 0
            or min(self.size) <= 0
        ):
            raise ValueError("video mode must have positive finite FPS and dimensions")
        self.segment_frames = (
            max(1, int(round(self.fps * 5.0)))
            if segment_frames is None else int(segment_frames)
        )
        if self.segment_frames <= 0:
            raise ValueError("video segment frame count must be positive")
        self._writer_factory = writer_factory or _default_video_writer
        self._validator = video_validator or _default_video_validator
        self._stitcher = video_stitcher or _default_video_stitcher
        self._segment_dir = self.session_dir / "rgb_segments"
        self._journal_path = self.session_dir / "rgb_segments.jsonl"
        self._segment_dir.mkdir(mode=0o750, exist_ok=False)
        _fsync_directory(self.session_dir)
        self._journal = _DurableJsonl(self._journal_path)
        self._writer = None
        self._partial: Path | None = None
        self._segment_index = 0
        self._segment_count = 0
        self._segment_first_index: int | None = None
        self._segment_last_index: int | None = None
        self._last_frame_index: int | None = None
        self.frame_count = 0
        self._closed = False

    def _open_segment(self) -> None:
        partial = self._segment_dir / f"segment_{self._segment_index:06d}.inprogress.mp4"
        final = self._segment_dir / f"segment_{self._segment_index:06d}.mp4"
        if partial.exists() or final.exists():
            raise FileExistsError(f"RGB segment already exists: {partial}")
        writer = self._writer_factory(partial, self.fps, self.size)
        if writer is None or not bool(writer.isOpened()):
            try:
                writer.release()
            except (AttributeError, RuntimeError):
                pass
            raise RecordingAborted(f"RGB segment writer failed to open: {partial}")
        self._partial = partial
        self._writer = writer

    def write(self, bgr_frame: np.ndarray, *, frame_index: int) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a finalized RGB recorder")
        frame = np.asarray(bgr_frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise RecordingAborted("RGB video frame must be an HxWx3 uint8 array")
        if tuple(frame.shape[:2][::-1]) != self.size:
            raise RecordingAborted(
                f"RGB dimensions changed from {self.size} to {tuple(frame.shape[:2][::-1])}"
            )
        index = int(frame_index)
        if self._last_frame_index is not None and index != self._last_frame_index + 1:
            missing = max(0, index - self._last_frame_index - 1)
            raise RecordingAborted(
                f"camera frame index gap {self._last_frame_index}->{index}; "
                f"missing {missing} frame(s)"
            )
        if self._writer is None:
            self._open_segment()
        result = self._writer.write(np.ascontiguousarray(frame))
        if result is False:
            raise RecordingAborted(f"RGB encoder rejected camera frame {index}")
        if self._segment_count == 0:
            self._segment_first_index = index
        self._segment_last_index = index
        self._last_frame_index = index
        self._segment_count += 1
        self.frame_count += 1
        if self._segment_count >= self.segment_frames:
            self._seal_segment()

    def _seal_segment(self) -> Path:
        if self._writer is None or self._partial is None or self._segment_count <= 0:
            raise RuntimeError("no active RGB segment to seal")
        writer = self._writer
        partial = self._partial
        expected_count = self._segment_count
        writer.release()
        self._writer = None
        self._partial = None
        if not partial.is_file() or partial.stat().st_size <= 0:
            raise RecordingAborted("RGB segment encoder produced no bytes")
        # Do not decode the segment on the acquisition thread: that would stall
        # CameraStream and drop live frames every few seconds. The closed bytes,
        # expected count and hash are made durable now; End/recovery performs the
        # exact full decode before publishing any conventional output.
        _fsync_file(partial)
        final = self._segment_dir / f"segment_{self._segment_index:06d}.mp4"
        os.replace(partial, final)
        _fsync_directory(self._segment_dir)
        record = {
            "segment_index": self._segment_index,
            "path": final.relative_to(self.session_dir).as_posix(),
            "frame_count": expected_count,
            "first_frame_index": self._segment_first_index,
            "last_frame_index": self._segment_last_index,
            "fps": self.fps,
            "width": self.size[0],
            "height": self.size[1],
            "bytes": final.stat().st_size,
            "sha256": _sha256_file(final),
        }
        self._journal.append(record)
        self._segment_index += 1
        self._segment_count = 0
        self._segment_first_index = None
        self._segment_last_index = None
        return final

    def _validated_segments(self) -> list[Path]:
        records = _read_segment_journal(self._journal_path)
        paths: list[Path] = []
        total = 0
        for expected_index, record in enumerate(records):
            if int(record.get("segment_index", -1)) != expected_index:
                raise RecordingAborted("RGB segment journal is not contiguous")
            expected_rel = f"rgb_segments/segment_{expected_index:06d}.mp4"
            if record.get("path") != expected_rel:
                raise RecordingAborted("RGB segment journal path is invalid")
            path = self.session_dir / expected_rel
            if path.is_symlink() or not path.is_file():
                raise RecordingAborted(f"sealed RGB segment is missing: {path}")
            if (
                path.stat().st_size != int(record.get("bytes", -1))
                or _sha256_file(path) != str(record.get("sha256", ""))
            ):
                raise RecordingAborted(f"sealed RGB segment changed: {path}")
            count = int(record.get("frame_count", -1))
            if count <= 0 or int(self._validator(path)) != count:
                raise RecordingAborted(f"sealed RGB segment frame count changed: {path}")
            total += count
            paths.append(path)
        if total != self.frame_count:
            raise RecordingAborted(
                f"sealed RGB segments contain {total} frames but capture has {self.frame_count}"
            )
        return paths

    def finalize(self) -> dict:
        if self._closed:
            raise RuntimeError("RGB recorder is already finalized")
        if self.frame_count <= 0:
            raise RecordingAborted("capture has no RGB frames")
        if self._writer is not None:
            self._seal_segment()
        self._journal.close()
        paths = self._validated_segments()
        output = self.session_dir / "rgb.mp4"
        partial = self.session_dir / (
            f".rgb.{os.getpid()}.{time.time_ns()}.finalizing.mp4"
        )
        if output.exists():
            raise FileExistsError(f"final RGB video already exists: {output}")
        try:
            stitched = int(self._stitcher(paths, partial, self.fps, self.size))
            if stitched != self.frame_count:
                raise RecordingAborted(
                    f"final RGB stitch wrote {stitched} frames but capture has {self.frame_count}"
                )
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise RecordingAborted("final RGB stitch produced no bytes")
            _fsync_file(partial)
            os.replace(partial, output)
            _fsync_directory(self.session_dir)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        decoded = int(self._validator(output))
        if decoded != self.frame_count:
            raise RecordingAborted(
                f"final rgb.mp4 decoded {decoded} frames but capture has {self.frame_count}"
            )
        self._closed = True
        return {
            "path": output.name,
            "frame_count": self.frame_count,
            "decoded_frame_count": decoded,
            "segment_count": len(paths),
            "segment_frames_limit": self.segment_frames,
            "bytes": output.stat().st_size,
            "sha256": _sha256_file(output),
        }

    def abort(self) -> None:
        """Close the active encoder but keep its explicit in-progress path."""
        if self._closed:
            return
        if self._writer is not None:
            try:
                self._writer.release()
            finally:
                self._writer = None
        self._journal.close()

    def cleanup_segments(self) -> None:
        """Reclaim duplicate bytes only after the sensors-complete manifest is durable."""
        if not self._closed:
            return
        try:
            shutil.rmtree(self._segment_dir)
            self._journal_path.unlink(missing_ok=True)
            _fsync_directory(self.session_dir)
        except OSError:
            # Extra sealed recovery sources are harmless and must never turn a
            # valid sensor capture into an abort during best-effort cleanup.
            pass


def recover_interrupted_session(
    session_dir: str | Path, *, video_validator=None, video_stitcher=None,
) -> dict:  # noqa: ANN001
    """Recover only diagnostic sidecars from an interrupted formal capture.

    This deliberately never writes ``rgb.mp4``, ``audio.wav``, ``FINALIZATION.json``
    or a complete/training-eligible manifest. All source partials are preserved.
    """
    session = Path(session_dir).expanduser().resolve()
    if not session.is_dir() or session.is_symlink():
        raise ValueError(f"recovery target is not a real session directory: {session}")
    if (session / "FINALIZATION.json").exists():
        raise RecordingAborted("production-finalized sessions cannot enter diagnostic recovery")
    try:
        manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    if manifest.get("status") == "complete" or manifest.get("training_eligible") is True:
        raise RecordingAborted("complete/training-eligible sessions cannot enter recovery")

    validator = video_validator or _default_video_validator
    stitcher = video_stitcher or _default_video_stitcher
    records = _read_segment_journal(session / "rgb_segments.jsonl")
    paths: list[Path] = []
    recovered_frames = 0
    fps = None
    size = None
    for expected_index, record in enumerate(records):
        if int(record.get("segment_index", -1)) != expected_index:
            break
        expected_rel = f"rgb_segments/segment_{expected_index:06d}.mp4"
        if record.get("path") != expected_rel:
            raise RecordingAborted("recovery segment path is invalid")
        path = session / expected_rel
        if path.is_symlink() or not path.is_file():
            break
        if (
            path.stat().st_size != int(record.get("bytes", -1))
            or _sha256_file(path) != str(record.get("sha256", ""))
        ):
            raise RecordingAborted(f"recovery segment integrity mismatch: {path}")
        count = int(record.get("frame_count", -1))
        if count <= 0 or int(validator(path)) != count:
            raise RecordingAborted(f"recovery segment frame count mismatch: {path}")
        current_fps = float(record.get("fps", 0.0))
        current_size = (int(record.get("width", 0)), int(record.get("height", 0)))
        if fps is None:
            fps, size = current_fps, current_size
        elif fps != current_fps or size != current_size:
            raise RecordingAborted("recovery segments do not share one camera mode")
        recovered_frames += count
        paths.append(path)

    video_result = {
        "status": "no_recoverable_video",
        "recovered_frame_count": 0,
        "source_segments": [],
    }
    if paths:
        output = session / "rgb.recovered.mp4"
        if output.exists():
            if output.is_symlink() or not output.is_file():
                raise RecordingAborted(
                    f"diagnostic video recovery output is not a regular file: {output}"
                )
            if int(validator(output)) != recovered_frames:
                raise RecordingAborted(
                    "existing diagnostic recovered video failed exact decode validation"
                )
        else:
            partial = session / f".rgb.recovering.{os.getpid()}.{time.time_ns()}.mp4"
            try:
                stitched = int(stitcher(paths, partial, float(fps), tuple(size)))
                if stitched != recovered_frames:
                    raise RecordingAborted(
                        f"diagnostic recovery wrote {stitched}, expected "
                        f"{recovered_frames} frames"
                    )
                _fsync_file(partial)
                os.replace(partial, output)
                _fsync_directory(session)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        if int(validator(output)) != recovered_frames:
            raise RecordingAborted("diagnostic recovered video failed exact decode validation")
        recovered_timeline = None
        source_timeline = session / "rgb_times.csv"
        if source_timeline.is_file() and not source_timeline.is_symlink():
            recovered_timeline = session / "rgb_times.recovered.csv"
            expected_lines: list[str] = []
            copied = 0
            with source_timeline.open("r", encoding="utf-8", newline="") as source:
                header = source.readline()
                if header != "frame_index,host_arrival_t_s,device_t_ms,time_source\n":
                    raise RecordingAborted("RGB recovery timeline header is invalid")
                expected_lines.append(header)
                for line in source:
                    if copied >= recovered_frames:
                        break
                    if not line.strip():
                        continue
                    expected_lines.append(line)
                    copied += 1
            if copied != recovered_frames:
                raise RecordingAborted(
                    f"RGB recovery timeline has {copied}, expected {recovered_frames} rows"
                )
            expected_timeline = "".join(expected_lines).encode("utf-8")
            if recovered_timeline.exists():
                if recovered_timeline.is_symlink() or not recovered_timeline.is_file():
                    raise RecordingAborted(
                        "existing RGB recovery timeline is not a regular file"
                    )
                if recovered_timeline.read_bytes() != expected_timeline:
                    raise RecordingAborted(
                        "existing RGB recovery timeline does not match the sealed prefix"
                    )
            else:
                _atomic_write_bytes(recovered_timeline, expected_timeline)
        video_result = {
            "status": "diagnostic_recovered",
            "recovered_frame_count": recovered_frames,
            "source_segments": [path.relative_to(session).as_posix() for path in paths],
            "output": output.name,
            "sha256": _sha256_file(output),
            "timeline": ({
                "output": recovered_timeline.name,
                "rows": recovered_frames,
                "sha256": _sha256_file(recovered_timeline),
            } if recovered_timeline is not None else None),
        }

    audio_result = {"status": "no_recoverable_audio", "training_eligible": False}
    if (session / "audio_capture.json").is_file() and (session / "audio.partial.pcm").is_file():
        from audio_stream import recover_partial_recording

        audio_result = recover_partial_recording(session)
    partials = sorted(
        path.relative_to(session).as_posix()
        for path in session.rglob("*")
        if path.is_file() and ("inprogress" in path.name or ".partial" in path.name)
    )
    result = {
        "schema_version": "wam_interrupted_session_recovery_v1",
        "status": "diagnostic_recovered",
        "training_eligible": False,
        "production_finalization_permitted": False,
        "video": video_result,
        "audio": audio_result,
        "preserved_partials": partials,
        "recovered_utc_ns": time.time_ns(),
    }
    _write_manifest(session / "RECOVERY.json", result)
    return result


def _write_wav(path: str, samples: np.ndarray, sr: int) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _validate_depth_frame(
    values: np.ndarray, *, expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Reject anything that is not a well-formed depth frame before it is published."""
    if not isinstance(values, np.ndarray) or values.dtype != np.uint16:
        raise RecordingAborted("depth frame must be a uint16 ndarray")
    if values.ndim != 2:
        raise RecordingAborted("depth frame must have a two-dimensional shape")
    if expected_shape is not None and tuple(values.shape) != tuple(expected_shape):
        raise RecordingAborted(
            f"depth frame shape {tuple(values.shape)!r} does not match RGB "
            f"shape {tuple(expected_shape)!r}"
        )
    return np.ascontiguousarray(values)


def _atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Publish a blob only after its complete bytes reached stable storage."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.inprogress"
    )
    try:
        with tmp.open("xb") as fp:
            fp.write(payload)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_save_npy(
    path: str | Path, values: np.ndarray, *, expected_shape: tuple[int, int] | None = None,
) -> None:
    """Publish one depth frame as raw NPY only after it reached stable storage."""
    values = _validate_depth_frame(values, expected_shape=expected_shape)
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    _atomic_write_bytes(path, buffer.getvalue())


def _atomic_save_depth(
    path: str | Path, values: np.ndarray, *, expected_shape: tuple[int, int] | None = None,
) -> None:
    """Publish one depth frame in the lossless self-identifying codec container.

    Aligned depth is ~4 MiB raw per frame (~124 MB/s at 30 fps); the codec is
    byte-exact yet ~14x smaller, which is what makes multi-hour capture fit on
    disk.  The frame is only visible at its final path once its complete
    ``.npy.zlib`` bytes are durable.
    """
    values = _validate_depth_frame(values, expected_shape=expected_shape)
    _atomic_write_bytes(path, depth_codec.encode_depth(values))


# Single source of truth for the depth timeline contract shared by the recorder,
# collection_supervisor, and training_authenticity_finalizer.  Keeping these here
# (and reusing them in persist_camera_frame) prevents the writer and the verifiers
# from silently drifting apart.
DEPTH_TIMES_HEADER = "frame_index,host_arrival_t_s,device_t_ms,depth_scale,path\n"


def depth_frame_relative_path(index: int) -> str:
    """Session-relative path of one compressed depth frame."""
    return f"depth/depth_{int(index):06d}{depth_codec.DEPTH_SUFFIX}"


def format_depth_timeline_row(
    index: int, host_arrival_t_s: float, device_t_ms: float, depth_scale: float,
) -> str:
    """One ``depth_times.csv`` row in the exact production schema."""
    return (
        f"{int(index)},{float(host_arrival_t_s):.6f},{device_t_ms},"
        f"{depth_scale},{depth_frame_relative_path(index)}\n"
    )


def streaming_audio_status(summary) -> dict:  # noqa: ANN001
    """Health result from counters, without reloading a multi-hour WAV."""
    status = str(getattr(summary, "status", "NO_DATA"))
    sample_rate = int(getattr(summary, "sr", 0))
    n_samples = int(getattr(summary, "n_samples", 0))
    peak_db = float(getattr(summary, "peak_db", -120.0))
    if status != "OK":
        return {
            "status": status,
            "peak_db": round(peak_db, 1),
            "msg": f"audio stream ended with {status}",
        }
    if sample_rate <= 0 or n_samples < sample_rate // 4:
        return {
            "status": "NO_DATA", "peak_db": -120.0,
            "msg": "no audio captured (device not open?)",
        }
    if peak_db < -40.0:
        return {
            "status": "SILENT", "peak_db": round(peak_db, 1),
            "msg": "audio remained below the commissioned signal floor",
        }
    return {
        "status": "OK", "peak_db": round(peak_db, 1),
        "msg": "audio signal present",
    }


def streaming_force_status(*, count: int, minimum: float, maximum: float) -> dict:
    """Match the existing force health contract with O(1) retained state."""
    if count < 5 or not np.isfinite([minimum, maximum]).all():
        return {"status": "NO_DATA", "swing_n": 0.0, "msg": "no force samples (sensor/serial down?)"}
    swing = float(maximum - minimum)
    if swing < 0.05:
        return {
            "status": "FROZEN", "swing_n": round(swing, 3),
            "msg": f"force not moving (swing {swing:.2f} N) — Hall loose/off, or no compression",
        }
    return {"status": "OK", "swing_n": round(swing, 2), "msg": "force responding"}


def reserve_session_directory(
    out_root: str | Path, *, session_dir: str | Path | None,
    generated_name: str | None = None,
) -> Path:
    """Claim a write-once raw-session directory without deleting prior data.

    The combined supervisor intentionally creates the exact session directory
    immediately before spawning this recorder.  An explicitly supplied empty
    directory is therefore the only pre-existing path accepted.  Auto-named
    sessions and every directory containing even one artifact are exclusive.
    """
    root = Path(out_root)
    if session_dir is None:
        root.mkdir(parents=True, exist_ok=True)
        name = generated_name or f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        selected = root / name
        try:
            selected.mkdir(exist_ok=False)
        except FileExistsError:
            raise FileExistsError(
                f"auto-named session directory already exists: {selected}"
            ) from None
        return selected

    selected = Path(session_dir)
    try:
        selected.mkdir(parents=True, exist_ok=False)
        return selected
    except FileExistsError:
        if selected.is_symlink() or not selected.is_dir():
            raise FileExistsError(
                f"session path already exists and is not a real directory: {selected}"
            ) from None
        if any(selected.iterdir()):
            raise FileExistsError(
                f"session directory is non-empty; refusing to overwrite: {selected}"
            ) from None
        return selected


def _positive_number(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RecordingAborted(f"{label} is missing or invalid") from exc
    if not np.isfinite(number) or number <= 0:
        raise RecordingAborted(f"{label} is missing or invalid")
    return number


# Aligned depth is stored losslessly compressed (depth_codec, ~10x on real
# frames).  The pre-flight budget divides raw depth by a CONSERVATIVE floor well
# below the measured ratio; the runtime free-space watermark (which always keeps
# the full safety reserve) is the hard backstop, so an optimistic floor can only
# cause a clean mid-session abort, never disk-full corruption.
DEPTH_CONSERVATIVE_COMPRESSION_RATIO = 3.0


def estimate_capture_storage(
    negotiated_mode: Mapping[str, object], *, backend: str, seconds: float,
    control_hz: float,
    depth_compression_ratio: float = DEPTH_CONSERVATIVE_COMPRESSION_RATIO,
) -> dict:
    """Conservatively budget the artifacts produced by the current recorder.

    Depth frames are lossless-compressed on disk, so the depth estimate is the
    raw RGB-D geometry divided by ``depth_compression_ratio`` (a conservative
    floor, default 3x vs the ~10x measured).  MP4 size is content dependent; one
    encoded byte per RGB pixel is intentionally used as a conservative planning
    rate and the runtime watermark catches overruns.
    """
    duration = _positive_number(seconds, label="capture duration")
    obs_hz = _positive_number(control_hz, label="control_hz")
    width = int(_positive_number(negotiated_mode.get("width"), label="camera width"))
    height = int(_positive_number(negotiated_mode.get("height"), label="camera height"))
    fps = _positive_number(negotiated_mode.get("fps"), label="camera fps")

    depth_width = 0
    depth_height = 0
    depth_bytes_per_pixel = 0
    depth_rate = 0.0
    depth_mode = negotiated_mode.get("depth_mode")
    if backend == "orbbec":
        if not isinstance(depth_mode, Mapping):
            raise RecordingAborted("Orbbec negotiated mode is missing depth_mode")
        if negotiated_mode.get("align_mode") == "software_depth_to_color":
            depth_width, depth_height = width, height
        else:
            depth_width = int(_positive_number(
                depth_mode.get("width"), label="depth width",
            ))
            depth_height = int(_positive_number(
                depth_mode.get("height"), label="depth height",
            ))
        depth_fps = _positive_number(depth_mode.get("fps", fps), label="depth fps")
        depth_format = str(depth_mode.get("format", "")).upper()
        depth_bytes_per_pixel = 2 if depth_format in {"Y16", "Z16"} else 4
        ratio = _positive_number(
            depth_compression_ratio, label="depth compression ratio",
        )
        if ratio < 1.0:
            raise RecordingAborted("depth compression ratio must be >= 1.0")
        # Raw frame / conservative ratio, plus a generous per-frame container
        # overhead floor (magic + length header + zlib framing).
        raw_depth_frame_bytes = depth_width * depth_height * depth_bytes_per_pixel
        depth_rate = (
            int(np.ceil(raw_depth_frame_bytes / ratio)) + 256
        ) * depth_fps

    rgb_mp4_rate = width * height * fps
    force_csv_rate = 192 * 100
    target_log_rate = 320 * fps
    # Audio PCM and recoverable RGB segments now grow incrementally.  Normal End
    # temporarily needs a second conventional WAV/MP4 while the durable sources
    # remain available, so both sides of that peak are budgeted explicitly.
    audio_input_pcm_rate = 48_000 * 2
    streaming_bytes = int(np.ceil(
        (rgb_mp4_rate + depth_rate + force_csv_rate + target_log_rate
         + audio_input_pcm_rate) * duration
    )) + 1024 * 1024

    audio_wav_bytes = int(np.ceil(16_000 * 2 * duration)) + 44
    audio_model_staging_bytes = int(np.ceil(16_000 * 2 * duration))
    rgb_index_bytes = int(np.ceil(112 * fps * duration)) + 256
    depth_index_bytes = (
        int(np.ceil(80 * fps * duration)) + 256 if depth_rate else 0
    )
    obs_index_bytes = int(np.ceil(192 * obs_hz * duration)) + 256
    rgb_final_bytes = int(np.ceil(rgb_mp4_rate * duration))
    deferred_bytes = (
        audio_model_staging_bytes + audio_wav_bytes + rgb_final_bytes
        + rgb_index_bytes + depth_index_bytes + obs_index_bytes
        + 1024 * 1024
    )
    return {
        "schema_version": "wam_storage_budget_v1",
        "mode": {
            "width": width,
            "height": height,
            "fps": fps,
            "depth_width": depth_width,
            "depth_height": depth_height,
            "depth_bytes_per_pixel": depth_bytes_per_pixel,
        },
        "duration_s": duration,
        "control_hz": obs_hz,
        "streaming_bytes": streaming_bytes,
        "deferred_bytes": deferred_bytes,
        "estimated_capture_bytes": streaming_bytes + deferred_bytes,
        "rgb_mp4_planning_bytes_per_pixel": 1.0,
        "depth_compression_ratio": (
            float(depth_compression_ratio) if backend == "orbbec" else None
        ),
    }


class StorageGuard:
    """Fail before ENOSPC and leave a durable, non-promotable failure record."""

    def __init__(
        self, session_dir: str | Path, *, budget: Mapping[str, object],
        duration_s: float, safety_reserve_bytes: int,
        disk_usage=None, check_interval_s: float = 1.0,
        projection_warmup_s: float = 30.0,
    ) -> None:  # noqa: ANN001
        self.session_dir = Path(session_dir)
        self.duration_s = _positive_number(duration_s, label="capture duration")
        self.streaming_bytes = int(budget["streaming_bytes"])
        self.deferred_bytes = int(budget["deferred_bytes"])
        self.estimated_capture_bytes = int(budget["estimated_capture_bytes"])
        self.safety_reserve_bytes = int(safety_reserve_bytes)
        self._disk_usage = disk_usage or shutil.disk_usage
        self.check_interval_s = max(0.0, float(check_interval_s))
        self.projection_warmup_s = max(0.0, float(projection_warmup_s))
        self._last_check = float("-inf")
        self._initial_free: int | None = None
        if (self.streaming_bytes < 0 or self.deferred_bytes < 0
                or self.estimated_capture_bytes
                != self.streaming_bytes + self.deferred_bytes
                or self.safety_reserve_bytes < 0):
            raise ValueError("storage budget/reserve is invalid")

    def _available(self) -> int:
        return int(self._disk_usage(self.session_dir).free)

    def _abort(
        self, *, phase: str, required: int, available: int,
        elapsed_s: float | None, message: str,
    ) -> None:
        payload = {
            "schema_version": "wam_storage_abort_v1",
            "phase": phase,
            "reason": message,
            "required_free_bytes": int(required),
            "available_free_bytes": int(available),
            "safety_reserve_bytes": self.safety_reserve_bytes,
            "estimated_capture_bytes": self.estimated_capture_bytes,
            "elapsed_s": elapsed_s,
            "duration_s": self.duration_s,
            "t_monotonic": time.monotonic(),
        }
        _write_manifest(self.session_dir / "STORAGE_ABORT.json", payload)
        raise RecordingAborted(message)

    def check_start(self) -> dict:
        required = self.safety_reserve_bytes + self.estimated_capture_bytes
        available = self._available()
        if available < required:
            self._abort(
                phase="start", required=required, available=available, elapsed_s=0.0,
                message=(
                    f"storage budget requires {required} free bytes but only "
                    f"{available} are available"
                ),
            )
        self._initial_free = available
        return {"required_free_bytes": required, "available_free_bytes": available}

    def check_runtime(self, *, elapsed_s: float, force: bool = False) -> dict | None:
        now = time.monotonic()
        if not force and now - self._last_check < self.check_interval_s:
            return None
        self._last_check = now
        elapsed = min(max(0.0, float(elapsed_s)), self.duration_s)
        remaining_fraction = (self.duration_s - elapsed) / self.duration_s
        required = (
            self.safety_reserve_bytes + self.deferred_bytes
            + int(np.ceil(self.streaming_bytes * remaining_fraction))
        )
        available = self._available()
        if available < required:
            self._abort(
                phase="recording", required=required, available=available,
                elapsed_s=elapsed,
                message=(
                    f"runtime storage watermark requires {required} free bytes but only "
                    f"{available} are available"
                ),
            )
        # Predictive early abort: once a warmup of real bytes has landed,
        # extrapolate the ACTUAL streaming rate to the full duration.  If streaming
        # alone would eat into the safety reserve, the session cannot complete, so
        # abort now (losing seconds) instead of near the end (losing hours) -- this
        # is what makes an optimistic depth-compression estimate safe.  It projects
        # streaming only (never the fictionally large deferred MP4 budget), so a
        # normal session -- whose real bytes run well under budget -- never trips.
        if (
            self._initial_free is not None
            and elapsed >= self.projection_warmup_s
            and elapsed > 0.0
        ):
            used = max(0, self._initial_free - available)
            projected_streaming_used = used * (self.duration_s / elapsed)
            projected_free_at_end = self._initial_free - projected_streaming_used
            if projected_free_at_end < self.safety_reserve_bytes:
                self._abort(
                    phase="recording_projection",
                    required=int(projected_streaming_used + self.safety_reserve_bytes),
                    available=available, elapsed_s=elapsed,
                    message=(
                        "projected capture rate would exhaust storage before End: "
                        f"~{int(projected_streaming_used)} streaming bytes projected vs "
                        f"{int(self._initial_free - self.safety_reserve_bytes)} usable "
                        "(actual compression is worse than budgeted)"
                    ),
                )
        return {"required_free_bytes": required, "available_free_bytes": available}


@dataclass(frozen=True)
class AudioCaptureSnapshot:
    samples: np.ndarray
    sr: int
    t_start: float
    t_end: float
    status: str


def snapshot_audio_at_end(
    stream, *, t0: float, t_end: float, pre_roll_s: float = 3.0,
    edge_tolerance_s: float = 0.050,
) -> AudioCaptureSnapshot:  # noqa: ANN001
    """Freeze and timestamp capture-interval audio before End finalization.

    Reading a longer window and clipping by host time preserves the first
    samples.  A callback arriving after this function returns cannot shift the
    WAV window because the selected samples are copied here.
    """
    start = float(t0)
    end = float(t_end)
    if not np.isfinite([start, end]).all() or end <= start:
        raise RecordingAborted("audio capture interval is invalid")
    raw = stream.latest_window((end - start) + max(0.0, float(pre_roll_s)))
    sr = int(getattr(raw, "sr", 0))
    raw_start = float(getattr(raw, "t_start", np.nan))
    samples = np.asarray(getattr(raw, "samples", []), dtype=np.float32)
    if sr <= 0 or samples.ndim != 1 or not np.isfinite(raw_start):
        raise RecordingAborted("audio End snapshot is invalid")
    first = max(0, min(samples.size, int(np.floor((start - raw_start) * sr))))
    last = max(first, min(samples.size, int(np.ceil((end - raw_start) * sr))))
    selected = samples[first:last].copy()
    selected_start = raw_start + first / float(sr)
    selected_end = selected_start + selected.size / float(sr)
    source_status = str(getattr(raw, "status", "NO_DATA"))
    covers = (
        selected_start <= start + edge_tolerance_s
        and selected_end >= end - edge_tolerance_s
    )
    status = "STALE" if source_status == "STALE" else "OK" if covers else "NO_DATA"
    return AudioCaptureSnapshot(
        samples=selected, sr=sr, t_start=selected_start,
        t_end=selected_end, status=status,
    )


# Posterior-lung auscultation sites → natural-language instruction (the VLA "task"/language). A
# language-conditioned VLA trains on (image, instruction) → action; WAM reuses the same string as a
# task label. Record ONE site per session so every demo carries its instruction.
SITE_INSTRUCTIONS = {
    "LUL": "auscultate the left upper lobe on the posterior chest",
    "LLL": "auscultate the left lower lobe on the posterior chest",
    "RUL": "auscultate the right upper lobe on the posterior chest",
    "RLL": "auscultate the right lower lobe on the posterior chest",
    "posterior-lung": "auscultate the posterior lung",
}


def record_session(out_root: str, seconds: float, *, control_hz: float = 50.0,
                   audio_device: str = "Scarlett Solo USB: Audio", audio_channel: int | None = 1,
                   hall_port: str | None = None, hall_cal: str | None = None,
                   camera_backend: str = "avfoundation", camera_index: int = 0,
                   camera_id: str | None = None,
                   site: str = "posterior-lung", instruction: str | None = None,
                   session_dir: str | None = None,
                   show_targets: bool = False, targets_path: str | None = None,
                   allow_uncalibrated_force: bool = False,
                   allow_missing_streams: bool = False,
                   arm_side: str | None = None,
                   sync_cal: str | None = None,
                   camera_latency_cal: str | None = None,
                   ready_file: str | None = None,
                   start_file: str | None = None,
                   stop_file: str | None = None,
                   capture_ended_file: str | None = None,
                   start_timeout_s: float = 30.0,
                   storage_reserve_bytes: int = 10 * 1024**3,
                   disk_check_interval_s: float = 1.0,
                   code_release_id: str | None = None,
                   runtime_release_id: str | None = None,
                   runtime_lock_sha256: str | None = None) -> str:
    from contextlib import ExitStack

    validate_capture_barriers(arm_side, ready_file, start_file)
    validate_timing_inputs(arm_side, sync_cal, camera_latency_cal)
    release_provenance = capture_release_provenance(
        code_release_id=code_release_id,
        runtime_release_id=runtime_release_id,
        runtime_lock_sha256=runtime_lock_sha256,
    )
    # Claim the raw identity before resolving calibration or importing hardware
    # drivers.  Existing artifacts must never be opened or lifecycle-rewritten.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sdir = str(reserve_session_directory(
        out_root, session_dir=session_dir, generated_name=f"session_{stamp}",
    ))
    import cv2
    from audio_stream import AudioStream
    from camera_stream import CameraStream
    from force_sensor import ForceCalibration, ForceSensor

    # session_dir lets a combined launcher point the sensor recorder AND the teleop (robot.csv) at the
    # SAME dir so they land in one aligned session; else auto-create session_<stamp> under out_root.
    hall_cal = str(resolve_hall_calibration(hall_cal))
    hall_port = hall_port or (sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*")) or [None])[0]
    force_snapshot_path = (Path(sdir) / "calibrations" / "force_calibration.json").resolve()
    if arm_side not in (None, "left", "right"):
        raise ValueError("arm_side must be left, right, or None")
    explicit_camera_id = camera_id.strip() if isinstance(camera_id, str) else None
    validate_collection_mode(
        arm_side, allow_uncalibrated_force, allow_missing_streams,
        camera_backend=camera_backend, camera_id=explicit_camera_id,
    )
    camera_id = explicit_camera_id or str(camera_index)

    task = {
        "site": site,
        "instruction": instruction or SITE_INSTRUCTIONS.get(site, f"auscultate the {site}"),
    }
    manifest = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "status": "initializing",
        "training_eligible": False,
        "abort_reason": None,
        "t0_monotonic": None,
        "t0_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_s": seconds,
        "control_hz": control_hz,
        "streams": [],
        "stream_errors": {},
        "rehearsal_overrides": [],
        "release_provenance": release_provenance,
        "task": task,
        "arm": ({
            "side": arm_side,
            "joint_names": [f"joint_{i}" for i in range(1, 8)],
            "state_unit": "rad",
            "action_type": "position_target",
            "timestamp_clock": "host_monotonic",
        } if arm_side else None),
        "audio": None,
        "force": {
            "port": hall_port,
            "source_calibration_path": hall_cal,
            "calibration_path": str(force_snapshot_path),
            "calibration_sha256": None,
            "calibrated": None,
            "device_timestamp_column": "device_t_ms",
        },
        "camera": None,
        "sync": {
            "clock": "host_monotonic",
            "force_device_time_logged": True,
            "audio_time_source": "portaudio_adc_mapped_to_host_monotonic",
            "applied": False,
        },
        "health": {},
    }
    manifest_path = os.path.join(sdir, "manifest.json")
    safety_stop = SafetyStop(sdir)
    audio_open_device: str | int = audio_device
    hall_open_port = hall_port

    streams: dict[str, object] = {}
    with _ManifestLifecycle(manifest_path, manifest) as lifecycle, ExitStack() as stack:
        camera_timing = None
        runtime_sync_mode = None
        cal = None
        try:
            cal, calibration_metadata = load_force_calibration_snapshot(
                hall_cal, sdir, ForceCalibration,
            )
            manifest["force"].update(calibration_metadata)
            manifest["force"]["calibrated"] = cal.calibrated
            manifest["force"]["max_screened_force_n"] = cal.max_screened_force_n
            manifest["force"]["hard_limit_force_n"] = cal.hard_limit_force_n
            validate_force_calibration(cal, allow_uncalibrated=allow_uncalibrated_force)
            if not cal.calibrated:
                manifest["rehearsal_overrides"].append("uncalibrated_force")
        except Exception as exc:  # noqa: BLE001
            manifest["stream_errors"]["force"] = str(exc)
            print(f"[force ] calibration failed: {exc}")
        if sync_cal:
            try:
                if audio_channel is None:
                    raise ValueError("audio_channel must be explicit when applying sync calibration")
                runtime_sync_mode = build_sync_capture_mode(
                    audio_device=audio_device,
                    audio_channel=audio_channel,
                    hall_port=hall_port,
                    force_calibration_sha256=manifest["force"].get("calibration_sha256"),
                    require_audio_usb_serial=arm_side is not None,
                )
                audio_open_device, hall_open_port = sync_capture_endpoints(runtime_sync_mode)
                manifest["force"]["hardware_descriptor"] = runtime_sync_mode["hall"]
                manifest["sync"].update(load_sync_calibration(
                    sync_cal, expected_capture_mode=runtime_sync_mode,
                ))
            except Exception as exc:  # noqa: BLE001
                raise RecordingAborted(f"invalid sync calibration: {exc}") from exc
        if camera_latency_cal:
            try:
                camera_timing = load_camera_latency_calibration(
                    camera_latency_cal,
                    expected_capture_mode={
                        "backend": camera_backend,
                        "camera_id": camera_id,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                raise RecordingAborted(f"invalid camera latency calibration: {exc}") from exc
        if cal is not None and "force" not in manifest["stream_errors"]:
            try:
                streams["force"] = stack.enter_context(
                    ForceSensor(
                        hall_open_port, cal, log_csv=os.path.join(sdir, "force.csv"),
                        on_hard_limit=safety_stop,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                manifest["stream_errors"]["force"] = str(exc)
                print(f"[force ] open failed: {exc}")
        try:
            camera_mode = camera_timing["capture_mode"] if camera_timing else {}
            streams["cam"] = stack.enter_context(
                CameraStream(
                    backend=camera_backend, index=camera_index, rgb=True,
                    camera_id=(
                        explicit_camera_id if camera_backend == "orbbec" else None
                    ),
                    width=camera_mode.get("width"), height=camera_mode.get("height"),
                    fps=camera_mode.get("fps"), color_format=camera_mode.get("color_format"),
                    depth_mode=camera_mode.get("depth_mode"),
                    align_depth_to_color=(
                        camera_mode.get("align_mode", "software_depth_to_color")
                        == "software_depth_to_color"
                    ),
                )
            )
        except Exception as e:  # noqa: BLE001
            manifest["stream_errors"]["cam"] = str(e)
            print(f"[camera] open failed: {e}")
        # Force callbacks can fire while the other devices are still opening;
        # safety always outranks readiness and storage diagnostics.
        safety_stop.raise_if_tripped()
        essential_missing = required_stream_failures(streams, {"force", "cam"})
        if essential_missing and (
            not allow_missing_streams
            or ("force" in essential_missing and arm_side is not None)
        ):
            detail = "; ".join(
                f"{name}: {manifest['stream_errors'].get(name, 'not available')}"
                for name in essential_missing
            )
            raise RecordingAborted(f"required streams unavailable: {detail}")
        if "cam" in streams:
            negotiated_storage_mode = getattr(streams["cam"], "negotiated_mode", None)
            if not isinstance(negotiated_storage_mode, Mapping):
                raise RecordingAborted(
                    "camera backend did not report a negotiated mode for storage budgeting"
                )
            storage_backend = camera_backend
        else:
            # Partial rehearsals remain non-training; still reserve their audio,
            # force, and index output with a negligible synthetic camera rate.
            negotiated_storage_mode = {"width": 1, "height": 1, "fps": 1.0}
            storage_backend = "missing"
        storage_budget = estimate_capture_storage(
            negotiated_storage_mode, backend=storage_backend,
            seconds=seconds, control_hz=control_hz,
        )
        storage_budget["safety_reserve_bytes"] = int(storage_reserve_bytes)
        manifest["storage_budget"] = storage_budget
        storage_guard = StorageGuard(
            sdir, budget=storage_budget, duration_s=seconds,
            safety_reserve_bytes=storage_reserve_bytes,
            check_interval_s=disk_check_interval_s,
        )
        storage_budget["start_capacity"] = storage_guard.check_start()

        # The live inference ring stays fixed; the capture interval is streamed
        # to disk after the Start barrier and never retained in this ring.
        try:
            streams["audio"] = stack.enter_context(
                AudioStream(
                    audio_open_device, pick_channel=audio_channel,
                    buffer_seconds=12.0,
                )
            )
        except Exception as e:  # noqa: BLE001
            manifest["stream_errors"]["audio"] = str(e)
            print(f"[audio ] open failed: {e}")
        safety_stop.raise_if_tripped()
        missing = required_stream_failures(streams, {"force", "audio", "cam"})
        if missing and (not allow_missing_streams or ("force" in missing and arm_side is not None)):
            detail = "; ".join(
                f"{name}: {manifest['stream_errors'].get(name, 'not available')}" for name in missing
            )
            raise RecordingAborted(f"required streams unavailable: {detail}")
        if missing:
            manifest["rehearsal_overrides"].extend(f"missing_stream:{name}" for name in missing)
        manifest["streams"] = sorted(streams)
        print(f"[rec] {sorted(streams)} -> {sdir}  ({seconds}s @ {control_hz} Hz)")
        time.sleep(0.8)  # let buffers fill

        if not allow_missing_streams:
            deadline = time.monotonic() + 3.0
            force_samples: list = []
            last_identity = None
            failures = ["initializing"]
            while time.monotonic() < deadline:
                safety_stop.raise_if_tripped()
                sample = streams["force"].read()
                safety_stop.observe(sample)
                safety_stop.raise_if_tripped()
                identity = (getattr(sample, "device_t_ms", None), getattr(sample, "t", None))
                if identity != last_identity:
                    force_samples.append(sample)
                    force_samples = force_samples[-5:]
                    last_identity = identity
                level = streams["audio"].level_db()
                camera_ready = streams["cam"].latest_frame()
                failures = readiness_failures(force_samples, level, camera_ready)
                if not failures:
                    if camera_timing is not None:
                        camera_timing["mode_validation"] = validate_camera_frame_mode(
                            camera_ready, camera_timing, streams["cam"],
                        )
                    break
                time.sleep(0.01)
            if failures:
                raise RecordingAborted(f"sensor readiness failed: {', '.join(failures)}")
            readiness = {
                "ready": True,
                "force_calibrated": bool(manifest["force"]["calibrated"]),
                "force_samples": len(force_samples),
                "streams": sorted(streams),
            }
        else:
            readiness = {
                "ready": True, "force_calibrated": False,
                "streams": sorted(streams), "rehearsal": True,
            }
        lifecycle.ready(readiness)
        if ready_file:
            _write_manifest(ready_file, readiness)
        if start_file:
            deadline = time.monotonic() + start_timeout_s
            while not Path(start_file).is_file():
                if time.monotonic() >= deadline:
                    raise RecordingAborted("timed out waiting for supervisor start barrier")
                sample = streams["force"].read()
                safety_stop.observe(sample)
                safety_stop.raise_if_tripped()
                time.sleep(0.01)

        # point-conditioned display: show the doctor the current auscultation target overlaid on the
        # live camera (the recorded rgb.mp4 stays CLEAN — overlay only on the shown copy) and log the
        # active target to targets.jsonl (same monotonic clock → converter attaches it per frame).
        tlog = None; targets = None; active = 0
        if show_targets:
            from ausc_targets import TargetLog, load_targets, overlay
            targets = load_targets(targets_path); tlog = TargetLog(sdir)
            stack.callback(tlog.close)
            print(f"[targets] doctor view: {[t.name for t in targets]}  (SPACE/n=next  b=back)")

        encoded_video_fps = video_writer_fps(
            streams.get("cam"), camera_timing, control_hz,
        )
        expected_depth_frame_shape = None
        if camera_backend == "orbbec":
            align_mode = negotiated_storage_mode.get(
                "align_mode", "software_depth_to_color",
            )
            shape_mode = (
                negotiated_storage_mode
                if align_mode == "software_depth_to_color"
                else negotiated_storage_mode.get("depth_mode")
            )
            if not isinstance(shape_mode, Mapping):
                raise RecordingAborted("negotiated Orbbec depth shape is unavailable")
            try:
                expected_depth_frame_shape = (
                    int(shape_mode["height"]), int(shape_mode["width"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RecordingAborted("negotiated Orbbec depth shape is invalid") from exc
            if min(expected_depth_frame_shape) <= 0:
                raise RecordingAborted("negotiated Orbbec depth shape is invalid")
        depth_dir = Path(sdir) / "depth"
        rgb_timeline = DurableCsv(
            Path(sdir) / "rgb_times.csv",
            "frame_index,host_arrival_t_s,device_t_ms,time_source\n",
        )
        obs_timeline = DurableCsv(
            Path(sdir) / "obs.csv",
            "t,force_n,force_status,audio_rms_db,audio_peak_db,cam_index,cam_t\n",
        )
        depth_timeline = None
        if camera_backend == "orbbec":
            depth_dir.mkdir(mode=0o750, exist_ok=False)
            _fsync_directory(Path(sdir))
            depth_timeline = DurableCsv(
                depth_dir / "depth_times.csv", DEPTH_TIMES_HEADER,
            )
        stack.callback(rgb_timeline.close)
        stack.callback(obs_timeline.close)
        if depth_timeline is not None:
            stack.callback(depth_timeline.close)

        video_recorder = None
        previous_depth_device_t_ms: float | None = None
        capture_writer = None
        capture_pipeline_finalized = False
        camera_end_frame = None
        camera_end_boundary_proof: dict[str, object] | None = None
        runtime_faults: set[str] = set()
        force_count = 0
        force_minimum = float("inf")
        force_maximum = float("-inf")
        camera_start_boundary_proof: dict[str, object] | None = None
        camera_delivery_health: dict[str, object] = {}
        camera_delivery_interval_id: int | None = None
        camera_boundary_index = None
        camera_boundary_t = None
        last_cam = -1
        last_captured_frame = None
        t0 = None

        def persist_camera_frame(frame) -> None:  # noqa: ANN001
            nonlocal video_recorder, previous_depth_device_t_ms
            rgb = getattr(frame, "frame", None)
            if (
                not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.ndim != 3 or rgb.shape[2] != 3
            ):
                raise RecordingAborted(f"camera frame {frame.index} has invalid RGB bytes")
            if camera_backend == "orbbec" and getattr(frame, "depth", None) is None:
                raise RecordingAborted(f"Orbbec frame {frame.index} is missing depth")
            depth = getattr(frame, "depth", None)
            if depth_timeline is not None:
                if not isinstance(depth, np.ndarray) or depth.dtype != np.uint16:
                    raise RecordingAborted(
                        f"camera frame {frame.index} depth must be uint16"
                    )
                if (
                    depth.ndim != 2
                    or expected_depth_frame_shape is None
                    or tuple(depth.shape) != expected_depth_frame_shape
                ):
                    raise RecordingAborted(
                        f"camera frame {frame.index} depth shape does not match "
                        "the negotiated capture mode"
                    )
            if video_recorder is None:
                h, w = rgb.shape[:2]
                video_recorder = SegmentedVideoRecorder(
                    sdir, fps=encoded_video_fps, size=(w, h),
                )
            video_recorder.write(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), frame_index=frame.index,
            )
            index, host_t, device_t_ms, source = camera_time_row(frame)
            if depth_timeline is not None:
                # The finalizer requires a finite, monotonic device clock for every
                # depth frame; without one the session can never become compliant,
                # so refuse to write it rather than seal an un-finalizable capture.
                if device_t_ms is None or not math.isfinite(float(device_t_ms)):
                    raise RecordingAborted(
                        f"camera frame {frame.index} lacks a finite device timestamp; "
                        "a compliant depth session needs a monotonic device clock"
                    )
                # Fail on the FIRST frozen/regressing device clock (which the
                # finalizer would otherwise only catch after the whole session is
                # recorded), so a broken clock costs one frame, not hours.
                if (
                    previous_depth_device_t_ms is not None
                    and float(device_t_ms) <= previous_depth_device_t_ms
                ):
                    raise RecordingAborted(
                        f"camera frame {frame.index} device clock froze or regressed "
                        f"({device_t_ms} <= {previous_depth_device_t_ms})"
                    )
                previous_depth_device_t_ms = float(device_t_ms)
                depth_relative = depth_frame_relative_path(frame.index)
                _atomic_save_depth(
                    Path(sdir) / depth_relative, depth,
                    expected_shape=expected_depth_frame_shape,
                )
            rgb_timeline.append(
                f"{index},{host_t:.6f},"
                f"{'' if device_t_ms is None else device_t_ms},{source}\n"
            )
            if depth_timeline is not None:
                depth_timeline.append(
                    format_depth_timeline_row(
                        frame.index, host_t, device_t_ms, frame.depth_scale,
                    )
                )

        capture_writer = BoundedCaptureWriter(
            persist_camera_frame, capacity=30, name="formal-rgbd-writer",
        )
        cam = streams.get("cam")
        if cam is not None:
            begin_delivery = getattr(cam, "begin_delivery_interval", None)
            if not callable(begin_delivery):
                raise RecordingAborted(
                    "camera stream lacks the bounded lossless Start handoff"
                )
            try:
                camera_start_boundary = begin_delivery(timeout_s=2.0)
                t0 = float(getattr(camera_start_boundary, "t_start"))
            except Exception as exc:  # noqa: BLE001
                raise RecordingAborted(
                    f"camera Start boundary handoff failed: {exc}"
                ) from exc
            raw_start_proof = getattr(camera_start_boundary, "proof", None)
            if not isinstance(raw_start_proof, Mapping):
                raise RecordingAborted(
                    "camera Start boundary did not return durability proof"
                )
            camera_start_boundary_proof = dict(raw_start_proof)
            if camera_start_boundary_proof.get("boundary_latched_atomically") is not True:
                raise RecordingAborted("camera Start boundary atomicity proof failed")
            boundary_frame = getattr(camera_start_boundary, "frame", None)
            boundary_has_frame = (
                boundary_frame is not None
                and getattr(boundary_frame, "frame", None) is not None
                and int(getattr(boundary_frame, "index", 0)) > 0
            )
            if camera_backend == "orbbec" and not boundary_has_frame:
                raise RecordingAborted(
                    "Orbbec camera has no valid frame at the Start sequence boundary"
                )
            if boundary_has_frame:
                camera_boundary_index = int(boundary_frame.index)
                camera_boundary_t = float(boundary_frame.t)
                last_cam = camera_boundary_index
            raw_interval_id = getattr(camera_start_boundary, "interval_id", None)
            if raw_interval_id is None:
                raw_interval_id = camera_start_boundary_proof.get("interval_id")
            camera_delivery_interval_id = (
                None if raw_interval_id is None else int(raw_interval_id)
            )
            if camera_delivery_interval_id is None or camera_delivery_interval_id <= 0:
                raise RecordingAborted(
                    "camera Start boundary did not bind a positive delivery interval id"
                )
            if int(camera_start_boundary_proof.get("interval_id", -1)) != (
                camera_delivery_interval_id
            ):
                raise RecordingAborted(
                    "camera Start boundary proof delivery interval id mismatch"
                )
            health_getter = getattr(cam, "delivery_health", None)
            if not callable(health_getter):
                raise RecordingAborted("camera stream lacks delivery FIFO health")
            camera_delivery_health = dict(health_getter())
            if int(camera_delivery_health.get("interval_id", -1)) != (
                camera_delivery_interval_id
            ):
                raise RecordingAborted(
                    "camera delivery FIFO health interval id mismatch at Start"
                )
        else:
            t0 = time.monotonic()
        if not np.isfinite(t0):
            raise RecordingAborted("camera Start boundary timestamp is invalid")
        capture_writer.start(
            t0=t0, camera_boundary_index=camera_boundary_index,
            camera_boundary_t=camera_boundary_t, camera_backend=camera_backend,
        )
        if "audio" in streams:
            streams["audio"].begin_recording(sdir, t0=t0)

        def abort_capture_pipeline() -> None:
            if capture_pipeline_finalized:
                return
            abort_delivery = getattr(cam, "abort_delivery_interval", None)
            if callable(abort_delivery):
                try:
                    abort_delivery()
                except Exception:
                    pass
            capture_writer.abort()
            if video_recorder is not None:
                video_recorder.abort()
            capture_fifo_status = capture_writer.status()
            interrupted_durability = {
                "schema_version": "wam_streaming_capture_v1",
                "status": "interrupted",
                "training_eligible": False,
                "capture_fifo": capture_fifo_status,
                "camera_delivery_fifo": (
                    dict(cam.delivery_health())
                    if cam is not None and callable(getattr(cam, "delivery_health", None))
                    else camera_delivery_health
                ),
                "camera_start_boundary": camera_start_boundary_proof,
                "crash_recovery_policy": "diagnostic_only_never_training_eligible",
            }
            if camera_end_boundary_proof is not None:
                interrupted_end_proof = dict(camera_end_boundary_proof)
                interrupted_end_proof.update({
                    "fifo_last_submitted_frame_index": capture_fifo_status.get(
                        "last_submitted_frame_index"
                    ),
                    "fifo_last_processed_frame_index": capture_fifo_status.get(
                        "last_processed_frame_index"
                    ),
                })
                interrupted_durability["camera_end_boundary"] = interrupted_end_proof
            manifest["capture_durability"] = interrupted_durability

        stack.callback(abort_capture_pipeline)
        lifecycle.recording(t0)
        next_obs = t0
        dt = 1.0 / control_hz
        last_preview_cam = -1

        def drain_camera_delivery_to_writer(*, wait_for_space=False, timeout_s=5.0) -> int:
            """Ack one source frame only after one formal-writer slot is available."""

            nonlocal last_cam, last_captured_frame
            if cam is None:
                return 0
            drain_delivery = getattr(cam, "drain_delivery", None)
            health_getter = getattr(cam, "delivery_health", None)
            if not callable(drain_delivery) or not callable(health_getter):
                raise RecordingAborted("camera lossless delivery FIFO is unavailable")
            deadline = time.monotonic() + float(timeout_s)
            moved = 0
            while True:
                health = dict(health_getter())
                if (
                    health.get("status") == "OVERFLOW"
                    or int(health.get("interval_overflow_frames", 0) or 0) > 0
                ):
                    first = health.get("first_overflow_frame_index")
                    suffix = "" if first is None else f" at camera frame {first}"
                    raise RecordingAborted(
                        "camera delivery FIFO overflow" + suffix
                        + "; accepted frames were not overwritten"
                    )
                writer_status = capture_writer.status()
                available = int(writer_status.get("available_queue_slots", 0))
                if available <= 0:
                    if not wait_for_space:
                        return moved
                    if time.monotonic() >= deadline:
                        raise RecordingAborted(
                            "formal capture writer did not free a FIFO slot before timeout"
                        )
                    time.sleep(0.001)
                    continue
                batch = drain_delivery(
                    max_frames=1, interval_id=camera_delivery_interval_id,
                )
                if not batch:
                    return moved
                if len(batch) != 1:
                    raise RecordingAborted(
                        "camera delivery FIFO violated one-frame acknowledged drain"
                    )
                capture_writer.submit(batch[0])
                moved += 1
                # `last_cam` is the exact last frame accepted for durable output.
                last_cam = int(batch[0].index)
                last_captured_frame = batch[0]

        while not capture_should_stop(
            stop_file, t0=t0, max_duration_s=seconds,
        ):
            safety_stop.raise_if_tripped()
            now = time.monotonic()
            storage_guard.check_runtime(elapsed_s=now - t0)
            drain_camera_delivery_to_writer()
            f = cam.latest_frame() if cam else None
            if f is not None and f.frame is not None and f.index != last_preview_cam:
                last_preview_cam = int(f.index)
                if float(f.t) >= t0 and show_targets and targets:  # doctor-only overlay
                    tlog.log(
                        f.t, targets[active], f.frame.shape[1], f.frame.shape[0],
                        camera_frame_index=f.index,
                        camera_device_t_ms=getattr(f, "device_t_ms", None),
                    )
                    try:                                          # display is best-effort (headless-safe)
                        disp = cv2.cvtColor(f.frame, cv2.COLOR_RGB2BGR)
                        overlay(disp, targets, active)
                        cv2.imshow("auscultation target (doctor view)", disp)
                        kk = cv2.waitKey(1) & 0xFF
                        if kk in (ord(" "), ord("n")):
                            active = (active + 1) % len(targets)
                        elif kk == ord("b"):
                            active = (active - 1) % len(targets)
                    except Exception:  # noqa: BLE001
                        pass
            if now >= next_obs:
                fr = streams["force"].read() if "force" in streams else None
                if fr is not None:
                    safety_stop.observe(fr)
                    safety_stop.raise_if_tripped()
                    force_value = float(fr.force_n)
                    force_count += 1
                    force_minimum = min(force_minimum, force_value)
                    force_maximum = max(force_maximum, force_value)
                lv = streams["audio"].level_db() if "audio" in streams else None
                stream_failures = runtime_stream_failures(
                    lv, f, require_depth=camera_backend == "orbbec",
                )
                if stream_failures:
                    runtime_faults.update(stream_failures)
                    if not allow_missing_streams:
                        raise RecordingAborted(
                            f"stream failed during capture: {', '.join(stream_failures)}"
                        )
                obs_t = observation_timestamp(last_captured_frame)
                obs_t_value = round(obs_t, 4)
                row = (
                    obs_t_value,
                    fr.force_n if fr else "", fr.status if fr else "",
                    round(lv.rms_db, 1) if lv else "",
                    round(lv.peak_db, 1) if lv else "",
                    last_cam,
                    (
                        round(float(last_captured_frame.t), 4)
                        if last_captured_frame is not None else ""
                    ),
                )
                obs_timeline.append(",".join(map(str, row)) + "\n")
                next_obs += dt
            time.sleep(0.002)

        # Atomically establish End against camera-arrival registration.  A frame
        # whose host-arrival timestamp is already inside the interval may still
        # be in Orbbec alignment/decode; wait for that exact in-flight frame and
        # submit the frozen <= End snapshot before closing the never-overwrite
        # FIFO.  This is a condition handoff, not a polling sleep.
        cam = streams.get("cam")
        if cam is not None:
            latch_end_boundary = getattr(cam, "latch_end_boundary", None)
            if not callable(latch_end_boundary):
                raise RecordingAborted(
                    "camera stream lacks the atomic End boundary handoff"
                )
            try:
                camera_end_boundary = latch_end_boundary(
                    timeout_s=2.0, interval_id=camera_delivery_interval_id,
                )
            except Exception as exc:  # noqa: BLE001 - backend failure is fail-closed
                raise RecordingAborted(
                    f"camera End boundary handoff failed: {exc}"
                ) from exc
            try:
                t_end = float(getattr(camera_end_boundary, "t_end"))
            except (TypeError, ValueError) as exc:
                raise RecordingAborted(
                    "camera End boundary returned an invalid timestamp"
                ) from exc
            raw_end_proof = getattr(camera_end_boundary, "proof", None)
            if not isinstance(raw_end_proof, Mapping):
                raise RecordingAborted(
                    "camera End boundary did not return durability proof"
                )
            camera_end_boundary_proof = dict(raw_end_proof)
            if int(camera_end_boundary_proof.get("interval_id", -1)) != (
                camera_delivery_interval_id
            ):
                raise RecordingAborted(
                    "camera End boundary proof delivery interval id mismatch"
                )
            if (
                camera_end_boundary_proof.get("boundary_latched_atomically") is not True
                or camera_end_boundary_proof.get("post_boundary_overwrite_prevented") is not True
            ):
                raise RecordingAborted(
                    "camera End boundary atomicity/overwrite proof failed"
                )
            inflight_seen = (
                camera_end_boundary_proof.get("inflight_pre_boundary_seen") is True
            )
            if inflight_seen and (
                camera_end_boundary_proof.get("inflight_pre_boundary_completed") is not True
                or camera_end_boundary_proof.get(
                    "inflight_pre_boundary_produced_frame"
                ) is not True
            ):
                raise RecordingAborted(
                    "camera End boundary in-flight frame did not complete as a usable frame"
                )
            camera_end_frame = getattr(camera_end_boundary, "frame", None)
            # The source is now closed, so its fixed tail can be moved with
            # bounded backpressure until the camera FIFO is empty.
            drain_camera_delivery_to_writer(wait_for_space=True, timeout_s=10.0)
            camera_delivery_health = dict(cam.delivery_health())
            if int(camera_delivery_health.get("interval_id", -1)) != (
                camera_delivery_interval_id
            ):
                raise RecordingAborted(
                    "camera delivery FIFO health interval id mismatch at End"
                )
            if int(camera_delivery_health.get("queue_depth", 0) or 0) != 0:
                raise RecordingAborted(
                    "camera delivery FIFO remained non-empty after End handoff"
                )
        else:
            t_end = time.monotonic()

        if not np.isfinite(t_end) or t_end < t0:
            raise RecordingAborted("camera End boundary timestamp is outside the capture interval")

        tail_frame_submitted = False
        tail_frame_already_submitted = False
        tail_frame_precedes_start = False
        tail_frame_index = None
        tail_frame_t = None
        if camera_end_frame is not None:
            tail_frame_index = int(getattr(camera_end_frame, "index", 0))
            tail_frame_t = float(getattr(camera_end_frame, "t", float("nan")))
            if (
                tail_frame_index <= 0
                or not np.isfinite(tail_frame_t)
                or tail_frame_t > t_end + 1e-9
            ):
                raise RecordingAborted(
                    "camera End boundary snapshot is outside the sealed interval"
                )
            if tail_frame_index < last_cam:
                raise RecordingAborted(
                    "camera End boundary snapshot regressed behind the last sampled frame"
                )
            fifo_last_submitted = capture_writer.status().get(
                "last_submitted_frame_index"
            )
            if tail_frame_t < t0:
                tail_frame_precedes_start = True
            elif tail_frame_index == fifo_last_submitted:
                tail_frame_already_submitted = True
            else:
                capture_writer.submit(camera_end_frame)
                last_cam = tail_frame_index
                tail_frame_submitted = True
        elif cam is not None and camera_backend == "orbbec":
            raise RecordingAborted(
                "Orbbec camera End boundary has no deterministic RGB-D snapshot"
            )

        if camera_end_boundary_proof is not None:
            camera_end_boundary_proof.update({
                "t_end_monotonic": t_end,
                "snapshot_frame_index": tail_frame_index,
                "snapshot_host_t": tail_frame_t,
                "tail_frame_submitted": tail_frame_submitted,
                "tail_frame_already_submitted": tail_frame_already_submitted,
                "tail_frame_precedes_start": tail_frame_precedes_start,
            })

        # Sampling is over.  Publish this before potentially slow audio/timeline
        # flush work so the supervisor can independently disable and reap teleop.
        capture_duration_s = max(0.0, t_end - t0)
        stop_reason = (
            "operator_end"
            if stop_file is not None and Path(stop_file).is_file()
            else "max_duration"
        )
        manifest["duration_s"] = capture_duration_s
        manifest["capture_end_reason"] = stop_reason
        if capture_ended_file:
            publish_capture_ended(
                capture_ended_file, t0=t0, t_end=t_end, reason=stop_reason,
            )
        storage_guard.check_runtime(elapsed_s=capture_duration_s, force=True)
        capture_fifo_summary = capture_writer.finish(t_end=t_end)
        if camera_end_boundary_proof is not None:
            camera_end_boundary_proof.update({
                "fifo_last_submitted_frame_index": capture_fifo_summary.get(
                    "last_submitted_frame_index"
                ),
                "fifo_last_processed_frame_index": capture_fifo_summary.get(
                    "last_processed_frame_index"
                ),
            })
        audio_summary = (
            streams["audio"].end_recording(t_end=t_end) if "audio" in streams else None
        )
        rgb_timeline.close()
        obs_timeline.close()
        if depth_timeline is not None:
            depth_timeline.close()
        video_summary = video_recorder.finalize() if video_recorder is not None else None
        if video_summary is not None and (
            int(video_summary["frame_count"]) != rgb_timeline.count
            or int(video_summary["decoded_frame_count"]) != rgb_timeline.count
        ):
            raise RecordingAborted("RGB video and incremental timeline frame counts differ")
        if audio_summary is not None and (
            _sha256_file(Path(sdir) / "audio.wav") != str(audio_summary.sha256)
        ):
            raise RecordingAborted("audio.wav changed after streaming finalization")
        if video_summary is not None and (
            _sha256_file(Path(sdir) / "rgb.mp4") != str(video_summary["sha256"])
        ):
            raise RecordingAborted("rgb.mp4 changed after streaming finalization")

        # Finalize health and bind the exact streamed bytes into the sensor manifest.
        if tlog is not None:
            tlog.close()
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001  (headless)
                pass
        audio_t0 = getattr(audio_summary, "t_start", None)
        a_health = {"status": "NO_DATA"}
        if audio_summary is not None:
            a_health = streaming_audio_status(audio_summary)
        f_health = streaming_force_status(
            count=force_count, minimum=force_minimum, maximum=force_maximum,
        )
        final_camera = camera_end_frame if "cam" in streams else None
        final_camera_status = getattr(final_camera, "status", "NO_DATA")
        camera_status = (
            "NO_DATA" if rgb_timeline.count == 0 else
            "OK" if final_camera_status == "OK" and getattr(final_camera, "frame", None) is not None
            else final_camera_status
        )
        camera_health = {
            "status": camera_status,
            "n_rgb_frames": rgb_timeline.count,
            "msg": (
                "camera frames present and live"
                if rgb_timeline.count and final_camera_status == "OK"
                else f"camera ended {final_camera_status} with {rgb_timeline.count} frames"
            ),
        }
        capture_durability = {
            "schema_version": "wam_streaming_capture_v1",
            "audio": ({
                "streamed_to_disk": True,
                "n_samples": int(audio_summary.n_samples),
                "sample_rate": int(audio_summary.sr),
                "sha256": str(audio_summary.sha256),
                "timeline": "audio_times.jsonl",
            } if audio_summary is not None else None),
            "video": video_summary,
            "capture_fifo": capture_fifo_summary,
            "camera_delivery_fifo": camera_delivery_health,
            "camera_start_boundary": camera_start_boundary_proof,
            "camera_end_boundary": camera_end_boundary_proof,
            "timelines": {
                "rgb_times.csv": {
                    "rows": rgb_timeline.count, "sha256": rgb_timeline.sha256(),
                },
                "obs.csv": {
                    "rows": obs_timeline.count, "sha256": obs_timeline.sha256(),
                },
                "depth/depth_times.csv": ({
                    "rows": depth_timeline.count, "sha256": depth_timeline.sha256(),
                } if depth_timeline is not None else None),
            },
            "crash_recovery_policy": "diagnostic_only_never_training_eligible",
        }
        manifest.update({
            "streams": sorted(streams),
            "audio": ({
                "sr": int(audio_summary.sr) if audio_summary is not None else 16000,
                "t0_monotonic": audio_t0,
                "device": audio_device,
                "channel": audio_channel,
                "hardware_descriptor": (
                    runtime_sync_mode["audio"] if runtime_sync_mode is not None else None
                ),
                "time_source": "portaudio_adc_mapped_to_host_monotonic",
            } if "audio" in streams else None),
            "camera": ({
                "backend": camera_backend,
                "index": camera_index,
                "camera_id": camera_id,
                "camera_id_source": (
                    "verified_orbbec_serial"
                    if camera_backend == "orbbec" and explicit_camera_id
                    else "operator_declared" if explicit_camera_id else "device_index"
                ),
                "n_rgb_frames": rgb_timeline.count,
                "n_depth_frames": depth_timeline.count if depth_timeline is not None else 0,
                "has_depth": bool(depth_timeline is not None and depth_timeline.count),
                "encoded_video_fps": encoded_video_fps,
                "timing": camera_timing,
                "start_boundary": camera_start_boundary_proof,
                "delivery_fifo": camera_delivery_health,
                "end_boundary": camera_end_boundary_proof,
            } if "cam" in streams else None),
            "n_obs": obs_timeline.count,
            "runtime_faults": sorted(runtime_faults),
            "health": {"audio": a_health, "force": f_health, "camera": camera_health},
            "capture_durability": capture_durability,
        })
        bad = (warn("stethoscope/audio", a_health)
               | warn("force sensor", f_health)
               | warn("camera", camera_health))
        if bad and not allow_missing_streams:
            failed = ", ".join(
                f"{name}={status['status']}" for name, status in manifest["health"].items()
                if status.get("status") != "OK"
            )
            raise RecordingAborted(f"device health check failed: {failed}")
        if bad:
            manifest["rehearsal_overrides"].append("device_health_failure")
        # Sensor recording alone cannot prove final robot.csv coverage because the
        # teleop process is still writing.  The combined launcher promotes this to
        # true only after it validates the complete shared interval.
        manifest["training_eligible"] = False
        manifest["eligibility_pending"] = ["robot_coverage_validation"]
        lifecycle.sensors_complete()
        capture_pipeline_finalized = True
        if video_recorder is not None:
            video_recorder.cleanup_segments()
        cleanup_audio = getattr(streams.get("audio"), "cleanup_recording_staging", None)
        if callable(cleanup_audio):
            cleanup_audio()
    print(f"[rec] {'⚠️ recorded WITH a device fault (see above) ' if bad else ''}done: "
          f"{rgb_timeline.count} rgb frames, {obs_timeline.count} obs rows -> {sdir}")
    return sdir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sessions"))
    ap.add_argument("--audio-device", default="Scarlett Solo USB: Audio")
    ap.add_argument("--audio-channel", type=int, default=1)
    ap.add_argument("--hall-port", default=None)
    ap.add_argument("--hall-cal", default=None,
                    help="force calibration JSON (default: as-built inverted VEO6 tared curve)")
    ap.add_argument("--allow-uncalibrated-force", action="store_true",
                    help="REHEARSAL ONLY: collect with calibrated=false; never marks training-eligible")
    ap.add_argument("--allow-missing-streams", action="store_true",
                    help="REHEARSAL ONLY: keep partial/faulted sensor output")
    ap.add_argument("--camera-backend", default="avfoundation", choices=["avfoundation", "orbbec", "fake"])
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--camera-id", default=None,
                    help="stable camera serial/id; mandatory for arm-connected collection")
    ap.add_argument("--arm-side", choices=["left", "right"], default=None,
                    help="physical OpenArm side; required by the combined IL launcher")
    ap.add_argument("--sync-cal", default=None,
                    help="versioned tap-sync JSON; hashed now and applied once during conversion")
    ap.add_argument("--camera-latency-cal", default=None,
                    help="versioned exposure-to-host camera latency calibration JSON")
    ap.add_argument("--ready-file", default=None,
                    help="supervisor readiness acknowledgement written after safe live samples")
    ap.add_argument("--start-file", default=None,
                    help="supervisor start barrier; timed capture waits for this file")
    ap.add_argument("--stop-file", default=None,
                    help="cooperative stop request; recorder exits capture on a frame boundary")
    ap.add_argument("--capture-ended-file", default=None,
                    help="durable sampling-ended acknowledgement for immediate teleop disable")
    ap.add_argument("--code-release-id", default=None,
                    help="immutable code_release-<20 hex> identity for this raw capture")
    ap.add_argument("--runtime-release-id", default=None,
                    help="immutable Linbox runtime release identity for this raw capture")
    ap.add_argument("--runtime-lock-sha256", default=None,
                    help="SHA-256 of the exact Linbox runtime lock used for capture")
    ap.add_argument("--start-timeout-s", type=float, default=30.0)
    ap.add_argument("--site", default="posterior-lung",
                    help="auscultation site (VLA task/language): LUL/LLL/RUL/RLL/posterior-lung")
    ap.add_argument("--instruction", default=None, help="override the natural-language instruction")
    ap.add_argument("--session-dir", default=None,
                    help="exact session dir (a combined launcher points sensors + teleop robot.csv here)")
    ap.add_argument("--show-targets", action="store_true",
                    help="point-conditioned: overlay the current auscultation target on the live camera "
                         "for the doctor + log the active target to targets.jsonl")
    ap.add_argument("--targets", default=None, help="targets config JSON / perception landmarks JSONL")
    a = ap.parse_args()
    record_session(a.out, a.seconds, control_hz=a.control_hz, audio_device=a.audio_device,
                   audio_channel=a.audio_channel, hall_port=a.hall_port, hall_cal=a.hall_cal,
                   camera_backend=a.camera_backend, camera_index=a.camera_index,
                   camera_id=a.camera_id,
                   site=a.site, instruction=a.instruction, session_dir=a.session_dir,
                   show_targets=a.show_targets, targets_path=a.targets,
                   allow_uncalibrated_force=a.allow_uncalibrated_force,
                   allow_missing_streams=a.allow_missing_streams, arm_side=a.arm_side,
                   sync_cal=a.sync_cal, camera_latency_cal=a.camera_latency_cal,
                   ready_file=a.ready_file, start_file=a.start_file,
                   stop_file=a.stop_file, capture_ended_file=a.capture_ended_file,
                   start_timeout_s=a.start_timeout_s,
                   code_release_id=a.code_release_id,
                   runtime_release_id=a.runtime_release_id,
                   runtime_lock_sha256=a.runtime_lock_sha256)
