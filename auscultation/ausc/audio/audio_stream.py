"""Continuous, non-blocking microphone/sound-card reader for the auscultation pipeline.

Mirrors `mechanical_sensor/force_feedback/force_sensor.ForceSensor`: a background
thread fills a ring buffer; any loop calls a **non-blocking** accessor. Where the force
sensor gives one scalar per call, the audio stream gives the **latest N-second window**
(resampled to the model rate, mono) plus a cheap input-level read for gating/monitoring.

The stethoscope path is analog: ThinkLabs One (3.5 mm) -> Scarlett Solo sound card -> USB.
The sound card shows up as a normal input device; pick it by name substring.

Two integration modes (same class):

1. **Imitation study** — pull the aligned audio window each control step and store a
   reference/feature so the policy + Q model see it::

       with AudioStream("Scarlett Solo") as audio:
           while recording:
               win = audio.latest_window(8.0)       # (samples_16k, t_start, t_end)
               obs["observation.q_usable"] = q_model(win.samples)

2. **Deployment / live monitoring** — poll the level for a contact/quality gate and run
   the sliding-window Q model::

       with AudioStream("Scarlett Solo") as audio:
           while running:
               if audio.level_db().peak_db > GATE_DB:        # something is coming through
                   win = audio.latest_window(8.0)
                   p_usable = q_model(win.samples)

Timestamps use `time.monotonic()` (same clock as `ForceSensor`), so the two streams
co-register; the fixed sound-card↔serial offset is calibrated once with a diaphragm tap
(cross-correlate the audio transient against the force impulse) — see
`docs/realtime_multimodal_pipeline.md` §1.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from math import gcd
from typing import Optional

import numpy as np

STATUS_OK = "OK"
STATUS_STALE = "STALE"      # stream alive but last block is older than stale_timeout
STATUS_NO_DATA = "NO_DATA"  # nothing captured yet (or fewer than requested samples)


@dataclass(frozen=True)
class AudioWindow:
    """One immutable window of audio at the model sample rate."""

    samples: np.ndarray   # float32 mono, length ~= seconds * sr
    sr: int               # sample rate of `samples` (model rate)
    t_start: float        # host monotonic time of the first sample (s)
    t_end: float          # host monotonic time of the last sample (s)
    n_in: int             # number of input-rate samples this window was built from
    status: str           # OK | STALE | NO_DATA


@dataclass(frozen=True)
class AudioLevel:
    """Cheap input level for gating/monitoring (computed on the most recent slice)."""

    rms_db: float
    peak_db: float
    t: float
    status: str


def _resample_poly(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample mono float32 (scipy if available, else linear fallback)."""
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32, copy=False)
    g = gcd(int(sr_in), int(sr_out))
    up, down = sr_out // g, sr_in // g
    try:
        from scipy.signal import resample_poly  # lazy: optional dependency
        return resample_poly(x, up, down).astype(np.float32)
    except Exception:  # noqa: BLE001 - fall back to linear interpolation
        n_out = int(round(x.size * sr_out / sr_in))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        xp = np.linspace(0.0, 1.0, x.size, endpoint=False)
        fp = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(fp, xp, x).astype(np.float32)


def _amp_to_db(amp: float) -> float:
    return 20.0 * np.log10(max(float(amp), 1e-9))


class _Ring:
    """Fixed-size mono ring buffer of input-rate samples, with a sample->host-time
    anchor so any slice can be timestamped. Thread-safety is provided by the caller."""

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(int(capacity), dtype=np.float32)
        self._cap = int(capacity)
        self._written = 0                 # total samples ever written (monotonic)
        self._anchor_idx = 0              # `written` at the last ingest
        self._anchor_t = 0.0             # host monotonic time at the last ingest

    def ingest(self, block: np.ndarray, host_t: float) -> None:
        n = block.size
        if n == 0:
            return
        if n >= self._cap:                # block bigger than ring: keep the tail
            block = block[-self._cap:]
            n = block.size
        start = self._written % self._cap
        end = start + n
        if end <= self._cap:
            self._buf[start:end] = block
        else:                             # wraps around
            first = self._cap - start
            self._buf[start:] = block[:first]
            self._buf[: n - first] = block[first:]
        self._written += n
        self._anchor_idx = self._written
        self._anchor_t = host_t

    def last(self, n: int) -> np.ndarray:
        n = min(int(n), self._written, self._cap)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        end = self._written % self._cap
        start = end - n
        if start >= 0:
            return self._buf[start:end].copy()
        return np.concatenate((self._buf[start:], self._buf[:end]))

    def end_time(self, sr_in: int) -> float:
        # the most recent sample's time ~= the last ingest time (block just arrived)
        return self._anchor_t + (self._written - self._anchor_idx) / sr_in

    @property
    def written(self) -> int:
        return self._written

    @property
    def anchor_t(self) -> float:
        return self._anchor_t


class AudioStream:
    """Background sound-card reader. Open with `start()` (or `with`), then call the
    non-blocking `latest_window()` / `level_db()` from any loop."""

    def __init__(
        self,
        device: str | int,
        sr_in: int = 48000,
        sr_model: int = 16000,
        channels: int | None = None,
        pick_channel: int | None = None,
        buffer_seconds: float = 12.0,
        block_ms: float = 50.0,
        stale_timeout_ms: float = 250.0,
    ) -> None:
        # channels: how many input channels to open (None = the device's max, so a
        #   multi-in interface like the Scarlett Solo exposes BOTH inputs).
        # pick_channel: which 0-based channel carries the signal (e.g. the Scarlett
        #   1/4" instrument input where the stethoscope sits is channel 1). None ->
        #   downmix all channels to mono (robust default; works for any wiring).
        self.device = device
        self.sr_in = int(sr_in)
        self.sr_model = int(sr_model)
        self.channels = None if channels is None else int(channels)
        self.pick_channel = pick_channel
        self.block_frames = max(1, int(self.sr_in * block_ms / 1000.0))
        self.stale_timeout_s = stale_timeout_ms / 1000.0
        self._ring = _Ring(int(buffer_seconds * self.sr_in))
        self._lock = threading.Lock()
        self._stream = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "AudioStream":
        import sounddevice as sd  # lazy: only needed for live capture

        if self.channels is None:  # open every input channel the device exposes
            self.channels = int(sd.query_devices(self.device, "input")["max_input_channels"])
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sr_in,
            channels=self.channels,
            blocksize=self.block_frames,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def __enter__(self) -> "AudioStream":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- capture callback (keep it light: only downmix + copy) -------------
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if indata.ndim == 1:
            block = indata
        elif self.pick_channel is not None and self.pick_channel < indata.shape[1]:
            block = indata[:, self.pick_channel]   # the channel carrying the signal
        else:
            block = indata.mean(axis=1)            # downmix all inputs to mono
        self._ingest(np.asarray(block, dtype=np.float32), time.monotonic())

    def _ingest(self, block: np.ndarray, host_t: float) -> None:
        """Push one mono block (testable without hardware)."""
        with self._lock:
            self._ring.ingest(block, host_t)

    # -- non-blocking accessors -------------------------------------------
    def latest_window(self, seconds: float) -> AudioWindow:
        """Most recent `seconds` of audio, resampled to the model rate, mono. Never
        blocks. Status is NO_DATA until the buffer has filled, STALE if capture stalled."""
        n_in = int(round(seconds * self.sr_in))
        now = time.monotonic()
        with self._lock:
            raw = self._ring.last(n_in)
            t_end = self._ring.end_time(self.sr_in)
            age = now - self._ring.anchor_t if self._ring.written else float("inf")
        if raw.size < n_in:
            status = STATUS_NO_DATA
        elif age > self.stale_timeout_s:
            status = STATUS_STALE
        else:
            status = STATUS_OK
        samples = _resample_poly(raw, self.sr_in, self.sr_model)
        return AudioWindow(
            samples=samples, sr=self.sr_model,
            t_start=t_end - raw.size / self.sr_in, t_end=t_end,
            n_in=int(raw.size), status=status,
        )

    def level_db(self, seconds: float = 0.2) -> AudioLevel:
        """RMS + peak level (dBFS) over the most recent `seconds`, for gating/monitoring."""
        n_in = int(round(seconds * self.sr_in))
        now = time.monotonic()
        with self._lock:
            raw = self._ring.last(n_in)
            age = now - self._ring.anchor_t if self._ring.written else float("inf")
        if raw.size == 0:
            return AudioLevel(rms_db=-120.0, peak_db=-120.0, t=now, status=STATUS_NO_DATA)
        status = STATUS_STALE if age > self.stale_timeout_s else STATUS_OK
        rms = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(raw)))
        return AudioLevel(rms_db=_amp_to_db(rms), peak_db=_amp_to_db(peak), t=now, status=status)


def list_input_devices() -> list[tuple[int, str]]:
    """(index, name) for every input-capable device — to find the Scarlett by name."""
    import sounddevice as sd
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]


if __name__ == "__main__":  # simple live level meter: python -m ausc.audio.audio_stream "Scarlett"
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name is None:
        for i, n in list_input_devices():
            print(f"  [{i}] {n}")
        sys.exit(0)
    with AudioStream(name) as a:
        while True:
            lv = a.level_db()
            print(f"peak {lv.peak_db:6.1f} dB | rms {lv.rms_db:6.1f} dB | {lv.status}", end="\r", flush=True)
            time.sleep(0.1)
