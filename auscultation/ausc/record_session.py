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
    force.csv         t, raw, hall_v, displacement_mm, force_n, status  (~100 Hz)
    rgb.mp4           color video
    rgb_times.csv     frame_index, t          (host monotonic, per frame)
    depth/            depth_<index>.npy + depth_times.csv   (orbbec backend only)
    obs.csv           t, force_n, status, audio_rms_db, audio_peak_db, cam_index, cam_t  (50 Hz)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # auscultation/
sys.path[:0] = [HERE, os.path.join(HERE, "audio"), os.path.join(HERE, "perception"),
                os.path.join(ROOT, "mechanical_sensor", "force_feedback")]
from health import audio_status, force_status, warn  # noqa: E402


def _write_wav(path: str, samples: np.ndarray, sr: int) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def record_session(out_root: str, seconds: float, *, control_hz: float = 50.0,
                   audio_device: str = "Scarlett Solo USB", audio_channel: int | None = 1,
                   hall_port: str | None = None, hall_cal: str | None = None,
                   camera_backend: str = "avfoundation", camera_index: int = 0) -> str:
    from contextlib import ExitStack

    import cv2
    from audio_stream import AudioStream
    from camera_stream import CameraStream
    from force_sensor import ForceCalibration, ForceSensor

    stamp = time.strftime("%Y%m%d_%H%M%S")
    sdir = os.path.join(out_root, f"session_{stamp}")
    os.makedirs(sdir, exist_ok=True)
    hall_port = hall_port or (sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*")) or [None])[0]
    hall_cal = hall_cal or os.path.join(ROOT, "mechanical_sensor", "force_feedback",
                                        "force_feedback_calibration_veo6.json")

    streams: dict[str, object] = {}
    with ExitStack() as stack:
        try:
            cal = ForceCalibration.from_json(hall_cal)
            streams["force"] = stack.enter_context(
                ForceSensor(hall_port, cal, log_csv=os.path.join(sdir, "force.csv")))
        except Exception as e:  # noqa: BLE001
            print(f"[force ] open failed: {e}")
        try:
            streams["audio"] = stack.enter_context(
                AudioStream(audio_device, pick_channel=audio_channel, buffer_seconds=seconds + 3.0))
        except Exception as e:  # noqa: BLE001
            print(f"[audio ] open failed: {e}")
        try:
            streams["cam"] = stack.enter_context(
                CameraStream(backend=camera_backend, index=camera_index, rgb=True))
        except Exception as e:  # noqa: BLE001
            print(f"[camera] open failed: {e}")
        print(f"[rec] {sorted(streams)} -> {sdir}  ({seconds}s @ {control_hz} Hz)")
        time.sleep(0.8)  # let buffers fill

        writer = None
        depth_dir = os.path.join(sdir, "depth")
        rgb_times, depth_times, obs_rows = [], [], []
        last_cam = -1
        t0 = time.monotonic()
        next_obs = t0
        dt = 1.0 / control_hz
        while time.monotonic() - t0 < seconds:
            now = time.monotonic()
            cam = streams.get("cam")
            f = cam.latest_frame() if cam else None
            if f is not None and f.frame is not None and f.index != last_cam:
                last_cam = f.index
                if writer is None:
                    h, w = f.frame.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(sdir, "rgb.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), control_hz, (w, h))
                writer.write(cv2.cvtColor(f.frame, cv2.COLOR_RGB2BGR))
                rgb_times.append((f.index, f.t))
                if f.depth is not None:
                    os.makedirs(depth_dir, exist_ok=True)
                    np.save(os.path.join(depth_dir, f"depth_{f.index:06d}.npy"), f.depth)
                    depth_times.append((f.index, f.t, f.depth_scale))
            if now >= next_obs:
                fr = streams["force"].read() if "force" in streams else None
                lv = streams["audio"].level_db() if "audio" in streams else None
                obs_rows.append((round(now, 4),
                                 fr.force_n if fr else "", fr.status if fr else "",
                                 round(lv.rms_db, 1) if lv else "", round(lv.peak_db, 1) if lv else "",
                                 last_cam, round(f.t, 4) if f and f.frame is not None else ""))
                next_obs += dt
            time.sleep(0.002)

        # finalize: audio dump + health + manifest
        audio_t0 = None
        a_health = {"status": "NO_DATA"}
        if "audio" in streams:
            w = streams["audio"].latest_window(seconds)
            _write_wav(os.path.join(sdir, "audio.wav"), w.samples, w.sr)
            audio_t0 = w.t_start
            a_health = audio_status(w.samples, w.sr)
        force_vals = [r[1] for r in obs_rows if r[1] != ""]
        f_health = force_status(force_vals) if force_vals else {"status": "NO_DATA"}
        if writer is not None:
            writer.release()
        with open(os.path.join(sdir, "rgb_times.csv"), "w") as fp:
            fp.write("frame_index,t\n"); fp.writelines(f"{i},{t:.4f}\n" for i, t in rgb_times)
        if depth_times:
            with open(os.path.join(depth_dir, "depth_times.csv"), "w") as fp:
                fp.write("frame_index,t,depth_scale\n")
                fp.writelines(f"{i},{t:.4f},{s}\n" for i, t, s in depth_times)
        with open(os.path.join(sdir, "obs.csv"), "w") as fp:
            fp.write("t,force_n,force_status,audio_rms_db,audio_peak_db,cam_index,cam_t\n")
            fp.writelines(",".join(map(str, r)) + "\n" for r in obs_rows)
        manifest = {
            "t0_monotonic": t0, "t0_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_s": seconds, "control_hz": control_hz,
            "streams": sorted(streams),
            "audio": {"sr": 16000, "t0_monotonic": audio_t0, "device": audio_device,
                      "channel": audio_channel} if "audio" in streams else None,
            "force": {"port": hall_port, "calibration": os.path.basename(hall_cal),
                      "n_rows_hint": "native ~100Hz in force.csv"} if "force" in streams else None,
            "camera": {"backend": camera_backend, "index": camera_index,
                       "n_rgb_frames": len(rgb_times), "n_depth_frames": len(depth_times),
                       "has_depth": bool(depth_times)} if "cam" in streams else None,
            "n_obs": len(obs_rows),
            "health": {"audio": a_health, "force": f_health},
        }
        with open(os.path.join(sdir, "manifest.json"), "w") as fp:
            json.dump(manifest, fp, indent=2)
    bad = warn("stethoscope/audio", a_health) | warn("force sensor", f_health)
    print(f"[rec] {'⚠️ recorded WITH a device fault (see above) ' if bad else ''}done: "
          f"{len(rgb_times)} rgb frames, {len(obs_rows)} obs rows -> {sdir}")
    return sdir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sessions"))
    ap.add_argument("--audio-device", default="Scarlett Solo USB")
    ap.add_argument("--audio-channel", type=int, default=1)
    ap.add_argument("--hall-port", default=None)
    ap.add_argument("--camera-backend", default="avfoundation", choices=["avfoundation", "orbbec", "fake"])
    ap.add_argument("--camera-index", type=int, default=0)
    a = ap.parse_args()
    record_session(a.out, a.seconds, control_hz=a.control_hz, audio_device=a.audio_device,
                   audio_channel=a.audio_channel, hall_port=a.hall_port,
                   camera_backend=a.camera_backend, camera_index=a.camera_index)
