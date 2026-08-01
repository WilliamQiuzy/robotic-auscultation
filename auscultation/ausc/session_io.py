"""Read side of the recording loop: load a recorded session and reconstruct host-clock
aligned multimodal samples for imitation learning / annotation.

`record_session.py` writes the session; this reads it back and, for any query time `t`,
returns the nearest force sample, the camera RGB+depth frame, and the audio window ending at
`t` — proving the on-disk format is consumable (and exercising the depth read path).

    s = Session("data/sessions/session_XXXX")
    snap = s.aligned_at(s.t0 + 3.0)     # {t, force_n, rgb, depth, audio}
"""

from __future__ import annotations

import csv
import json
import os
import wave
from bisect import bisect_left
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ausc.perception import depth_codec


@dataclass(frozen=True)
class Snapshot:
    t: float
    force_n: Optional[float]
    force_dt_ms: Optional[float]    # how far the nearest force sample is from t
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    cam_dt_ms: Optional[float]
    audio: Optional[np.ndarray]     # window ending at t
    audio_sr: int


def _read_csv(path: str) -> tuple[list[str], list[list[str]]]:
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


class Session:
    def __init__(self, sdir: str) -> None:
        self.dir = sdir
        self.manifest = json.load(open(os.path.join(sdir, "manifest.json")))
        self.t0 = self.manifest["t0_monotonic"]
        self.duration = self.manifest["duration_s"]
        # force: keep host_t + force_n, windowed to [t0, t0+duration] (drop close-time over-run)
        self._ft, self._fn = [], []
        fp = os.path.join(sdir, "force.csv")
        if os.path.exists(fp):
            _, rows = _read_csv(fp)
            for r in rows:
                try:
                    t = float(r[0])
                except (ValueError, IndexError):
                    continue
                if self.t0 - 0.5 <= t <= self.t0 + self.duration + 0.5:
                    self._ft.append(t); self._fn.append(float(r[4]))
        # camera frame index -> host time
        self._rgb_t = self._load_times(os.path.join(sdir, "rgb_times.csv"))
        self._depth = self.manifest.get("camera", {}).get("has_depth") if self.manifest.get("camera") else False
        self._depth_t = self._load_times(os.path.join(sdir, "depth", "depth_times.csv")) if self._depth else {}
        # audio
        self._audio_t0 = (self.manifest.get("audio") or {}).get("t0_monotonic")
        self.audio_sr = (self.manifest.get("audio") or {}).get("sr", 16000)
        self._wav = os.path.join(sdir, "audio.wav")
        self._mp4 = os.path.join(sdir, "rgb.mp4")

    @staticmethod
    def _load_times(path: str) -> dict[int, float]:
        if not os.path.exists(path):
            return {}
        _, rows = _read_csv(path)
        return {int(r[0]): float(r[1]) for r in rows if len(r) >= 2}

    # -- aligned access ----------------------------------------------------
    def force_at(self, t: float) -> tuple[Optional[float], Optional[float]]:
        if not self._ft:
            return None, None
        i = min(bisect_left(self._ft, t), len(self._ft) - 1)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self._ft)]
        j = min(cands, key=lambda k: abs(self._ft[k] - t))
        return self._fn[j], (self._ft[j] - t) * 1000.0

    def _nearest_index(self, times: dict[int, float], t: float) -> Optional[int]:
        if not times:
            return None
        return min(times, key=lambda i: abs(times[i] - t))

    def frame_at(self, t: float) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        idx = self._nearest_index(self._rgb_t, t)
        if idx is None:
            return None, None, None
        rgb = None
        if os.path.exists(self._mp4):
            import cv2
            cap = cv2.VideoCapture(self._mp4)
            # frames are written in index order starting at the first captured index
            order = sorted(self._rgb_t)
            pos = order.index(idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, bgr = cap.read(); cap.release()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok and bgr is not None else None
        depth = None
        if self._depth:
            di = self._nearest_index(self._depth_t, t) or idx
            for suffix in (depth_codec.DEPTH_SUFFIX, depth_codec.LEGACY_DEPTH_SUFFIX):
                dp = os.path.join(self.dir, "depth", f"depth_{di:06d}{suffix}")
                if os.path.exists(dp):
                    depth = depth_codec.load_depth(dp)   # new container or legacy raw .npy
                    break
        return rgb, depth, (self._rgb_t[idx] - t) * 1000.0

    def audio_window(self, t: float, seconds: float = 8.0) -> Optional[np.ndarray]:
        if self._audio_t0 is None or not os.path.exists(self._wav):
            return None
        w = wave.open(self._wav)
        n_total = w.getnframes()
        data = np.frombuffer(w.readframes(n_total), dtype="<i2").astype(np.float32) / 32768.0
        w.close()
        end = int(round((t - self._audio_t0) * self.audio_sr))
        start = max(0, end - int(seconds * self.audio_sr))
        end = min(max(end, 0), n_total)
        return data[start:end]

    def aligned_at(self, t: float, audio_seconds: float = 8.0) -> Snapshot:
        fn, fdt = self.force_at(t)
        rgb, depth, cdt = self.frame_at(t)
        au = self.audio_window(t, audio_seconds)
        return Snapshot(t=t, force_n=fn, force_dt_ms=fdt, rgb=rgb, depth=depth,
                        cam_dt_ms=cdt, audio=au, audio_sr=self.audio_sr)


if __name__ == "__main__":
    import sys

    sd = sys.argv[1] if len(sys.argv) > 1 else sorted(
        __import__("glob").glob("data/sessions/session_*"))[-1]
    s = Session(sd)
    print(f"session {sd}")
    print(f"  t0={s.t0:.2f} dur={s.duration}s force_rows={len(s._ft)} "
          f"rgb_frames={len(s._rgb_t)} depth={'yes' if s._depth else 'no'}")
    tq = s.t0 + s.duration / 2.0
    snap = s.aligned_at(tq)
    print(f"  aligned_at(midpoint t0+{s.duration/2:.1f}s):")
    print(f"    force_n={snap.force_n} (dt {snap.force_dt_ms:+.1f} ms)" if snap.force_n is not None else "    force: none")
    print(f"    rgb={None if snap.rgb is None else snap.rgb.shape} depth={None if snap.depth is None else snap.depth.shape} (cam dt {snap.cam_dt_ms:+.1f} ms)")
    print(f"    audio_window={None if snap.audio is None else snap.audio.shape} @ {snap.audio_sr}Hz")
    ok = (snap.force_n is not None and snap.rgb is not None and snap.audio is not None
          and abs(snap.force_dt_ms or 1e9) < 50 and abs(snap.cam_dt_ms or 1e9) < 100)
    print("  ALIGNED-CONSUMABLE:", "YES" if ok else "partial")
