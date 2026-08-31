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

import hashlib
import json
import os
import queue
import threading
import time
import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np

STATUS_OK = "OK"
STATUS_STALE = "STALE"      # stream alive but last block is older than stale_timeout
STATUS_NO_DATA = "NO_DATA"  # nothing captured yet (or fewer than requested samples)
AUDIO_SIGNAL_AUTHENTICITY_SCHEMA_VERSION = "wam_audio_signal_authenticity_v1"
# v3 (2026-08-29, operator-approved): only fabrication-class evidence (all-zero /
# constant PCM) rejects a session. Degraded-but-real evidence (frozen dropouts,
# low AC energy, clipping, DC offset) is preserved as a ledger so downstream
# reward/Q layers can mask windows instead of voiding doctor-confirmed sessions.
AUDIO_SIGNAL_AUTHENTICITY_POLICY_VERSION = "audio_signal_degraded_real_v3"
_AUDIO_AUTHENTICITY_POLICY = {
    "distinct_value_count_cap": 1_024,
    "freeze_block_duration_ms": 50,
    "freeze_min_duration_s": 2.0,
    "freeze_min_blocks": 40,
    "clipping_abs_pcm": 32_760,
    "clipping_min_fraction": 0.05,
    "clipping_min_duration_s": 0.5,
    "dc_min_abs_mean_fraction": 0.25,
    "dc_min_mean_to_rms_ratio": 0.9,
    "dc_min_duration_s": 1.0,
    # Two PCM16 counts is deliberately far below the live Start threshold,
    # but rejects USB/digital dithering such as {-1, 0, 1} that previously
    # looked non-constant while carrying no usable acoustic information.
    "minimum_effective_ac_rms_pcm": 2.0,
    "degraded_region_ledger_cap": 64,
}


class AudioSignalAuthenticityError(RuntimeError):
    """PCM evidence is consistent with a dead or corrupted input channel."""


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


@dataclass(frozen=True)
class AudioRecordingSummary:
    """Bounded-memory recording result; waveform bytes stay on disk."""

    path: str
    sr: int
    t_start: float
    t_end: float
    n_samples: int
    status: str
    peak_db: float
    sha256: str
    signal_authenticity: dict[str, object]


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("atomic audio metadata write made no progress")
            view = view[written:]
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        tmp.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)
    try:
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_bytes(
        path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AudioSignalAuthenticityAccumulator:
    """Bounded-memory PCM16 dead-channel checks shared by producer and verifier."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("audio authenticity sample rate must be positive")
        block_ms = int(_AUDIO_AUTHENTICITY_POLICY["freeze_block_duration_ms"])
        self.block_samples = max(1, round(self.sample_rate * block_ms / 1_000.0))
        self.n_samples = 0
        self._all_zero = True
        self._constant = True
        self._first_value: int | None = None
        self._minimum: int | None = None
        self._maximum: int | None = None
        self._distinct: set[int] = set()
        self._sum = 0
        self._sum_squares = 0
        self._clipped = 0
        self._pending = np.zeros(0, dtype="<i2")
        self._previous_block: bytes | None = None
        self._identical_block_run = 0
        self._longest_identical_block_run = 0
        self._block_count = 0
        self._frozen_regions: list[dict[str, object]] = []
        self._frozen_regions_truncated = False

    def _close_frozen_run(self) -> None:
        run = self._identical_block_run
        if run <= 0:
            return
        duration_s = run * self.block_samples / float(self.sample_rate)
        if (
            run >= int(_AUDIO_AUTHENTICITY_POLICY["freeze_min_blocks"])
            and duration_s >= float(_AUDIO_AUTHENTICITY_POLICY["freeze_min_duration_s"])
        ):
            cap = int(_AUDIO_AUTHENTICITY_POLICY["degraded_region_ledger_cap"])
            if len(self._frozen_regions) < cap:
                start_sample = (self._block_count - run) * self.block_samples
                self._frozen_regions.append({
                    "start_sample": int(start_sample),
                    "start_s": round(start_sample / float(self.sample_rate), 9),
                    "n_blocks": int(run),
                    "duration_s": round(duration_s, 9),
                })
            else:
                self._frozen_regions_truncated = True

    def update(self, samples: np.ndarray) -> None:
        values = np.asarray(samples, dtype="<i2").reshape(-1)
        if values.size <= 0:
            return
        widened = values.astype(np.int64)
        self.n_samples += int(values.size)
        minimum = int(np.min(widened))
        maximum = int(np.max(widened))
        self._minimum = minimum if self._minimum is None else min(self._minimum, minimum)
        self._maximum = maximum if self._maximum is None else max(self._maximum, maximum)
        if self._first_value is None:
            self._first_value = int(values[0])
        if self._constant and bool(np.any(values != self._first_value)):
            self._constant = False
        if self._all_zero and bool(np.any(values != 0)):
            self._all_zero = False
        cap = int(_AUDIO_AUTHENTICITY_POLICY["distinct_value_count_cap"])
        if len(self._distinct) < cap:
            for value in np.unique(values):
                self._distinct.add(int(value))
                if len(self._distinct) >= cap:
                    break
        self._sum += int(np.sum(widened, dtype=np.int64))
        self._sum_squares += int(np.dot(widened, widened))
        clip_level = int(_AUDIO_AUTHENTICITY_POLICY["clipping_abs_pcm"])
        self._clipped += int(np.count_nonzero(np.abs(widened) >= clip_level))

        joined = (
            np.concatenate((self._pending, values))
            if self._pending.size else values
        )
        complete = (joined.size // self.block_samples) * self.block_samples
        for offset in range(0, complete, self.block_samples):
            raw = joined[offset:offset + self.block_samples].tobytes()
            if raw == self._previous_block:
                self._identical_block_run += 1
            else:
                self._close_frozen_run()
                self._previous_block = raw
                self._identical_block_run = 1
            self._block_count += 1
            self._longest_identical_block_run = max(
                self._longest_identical_block_run,
                self._identical_block_run,
            )
        self._pending = joined[complete:].copy()

    def finish(self) -> dict[str, object]:
        if self.n_samples <= 0:
            raise AudioSignalAuthenticityError(
                "audio signal authenticity failed: no PCM samples"
            )
        duration_s = self.n_samples / float(self.sample_rate)
        assert self._minimum is not None and self._maximum is not None
        mean_pcm = self._sum / float(self.n_samples)
        rms_pcm = float(np.sqrt(self._sum_squares / float(self.n_samples)))
        ac_variance = max(
            0.0,
            self._sum_squares / float(self.n_samples) - mean_pcm * mean_pcm,
        )
        ac_rms_pcm = float(np.sqrt(ac_variance))
        clipped_fraction = self._clipped / float(self.n_samples)
        mean_to_rms_ratio = abs(mean_pcm) / max(rms_pcm, 1e-12)
        repeated_duration_s = (
            self._longest_identical_block_run * self.block_samples
            / float(self.sample_rate)
        )

        self._close_frozen_run()

        # Fabrication-class evidence: the bytes cannot be a real capture. Hard fail.
        if self._all_zero:
            raise AudioSignalAuthenticityError(
                "audio signal authenticity failed: all-zero PCM"
            )
        if self._constant:
            raise AudioSignalAuthenticityError(
                "audio signal authenticity failed: constant PCM"
            )

        # Degraded-but-real evidence: preserved as a ledger, never voids the session.
        checks = {
            "all_zero": "PASS",
            "constant_pcm": "PASS",
            "minimum_effective_ac_energy": "PASS",
            "repeated_frozen_blocks": "PASS",
            "severe_clipping": "PASS",
            "severe_dc": "PASS",
        }
        degradations: list[str] = []
        if ac_rms_pcm < float(
            _AUDIO_AUTHENTICITY_POLICY["minimum_effective_ac_rms_pcm"]
        ):
            checks["minimum_effective_ac_energy"] = "DEGRADED"
            degradations.append("minimum_effective_ac_energy")
        if (
            duration_s >= float(_AUDIO_AUTHENTICITY_POLICY["clipping_min_duration_s"])
            and clipped_fraction
            >= float(_AUDIO_AUTHENTICITY_POLICY["clipping_min_fraction"])
        ):
            checks["severe_clipping"] = "DEGRADED"
            degradations.append("severe_clipping")
        if (
            duration_s >= float(_AUDIO_AUTHENTICITY_POLICY["dc_min_duration_s"])
            and abs(mean_pcm) / 32_767.0
            >= float(_AUDIO_AUTHENTICITY_POLICY["dc_min_abs_mean_fraction"])
            and mean_to_rms_ratio
            >= float(_AUDIO_AUTHENTICITY_POLICY["dc_min_mean_to_rms_ratio"])
        ):
            checks["severe_dc"] = "DEGRADED"
            degradations.append("severe_dc")
        if self._frozen_regions or self._frozen_regions_truncated:
            checks["repeated_frozen_blocks"] = "DEGRADED"
            degradations.append("repeated_frozen_blocks")
        return {
            "schema_version": AUDIO_SIGNAL_AUTHENTICITY_SCHEMA_VERSION,
            "policy_version": AUDIO_SIGNAL_AUTHENTICITY_POLICY_VERSION,
            "status": "PASS_DEGRADED" if degradations else "PASS",
            "degradations": degradations,
            "checks": checks,
            "policy": dict(_AUDIO_AUTHENTICITY_POLICY),
            "evidence": {
                "n_samples": self.n_samples,
                "duration_s": round(duration_s, 9),
                "minimum_pcm": self._minimum,
                "maximum_pcm": self._maximum,
                "distinct_pcm_values_capped": len(self._distinct),
                "clipped_samples": self._clipped,
                "clipped_fraction": round(clipped_fraction, 9),
                "mean_pcm": round(mean_pcm, 9),
                "rms_pcm": round(rms_pcm, 9),
                "ac_rms_pcm": round(ac_rms_pcm, 9),
                "mean_to_rms_ratio": round(mean_to_rms_ratio, 9),
                "longest_identical_block_run": self._longest_identical_block_run,
                "longest_identical_block_run_s": round(repeated_duration_s, 9),
                "frozen_regions": [dict(region) for region in self._frozen_regions],
                "frozen_regions_truncated": self._frozen_regions_truncated,
            },
        }


def analyze_audio_pcm16_file(
    path: str | Path, *, sample_rate: int, n_samples: int,
) -> dict[str, object]:
    """Recompute the versioned authenticity receipt from a raw PCM16 file."""

    pcm_path = Path(path)
    expected_samples = int(n_samples)
    if expected_samples <= 0 or pcm_path.stat().st_size != expected_samples * 2:
        raise AudioSignalAuthenticityError(
            "audio signal authenticity failed: PCM size does not match sample count"
        )
    analyzer = AudioSignalAuthenticityAccumulator(sample_rate)
    decoded = 0
    with pcm_path.open("rb") as source:
        while decoded < expected_samples:
            raw = source.read(min(65_536, expected_samples - decoded) * 2)
            if not raw or len(raw) % 2:
                raise AudioSignalAuthenticityError(
                    "audio signal authenticity failed: PCM stream is incomplete"
                )
            pcm = np.frombuffer(raw, dtype="<i2")
            analyzer.update(pcm)
            decoded += int(pcm.size)
    return analyzer.finish()


def _publish_wav_from_pcm(
    pcm_path: Path, output_path: Path, *, sample_rate: int, n_samples: int,
) -> None:
    """Stream PCM into a crash-safe WAV without loading the session into RAM."""
    tmp = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.{time.time_ns()}.finalizing.wav"
    )
    remaining = int(n_samples) * 2
    try:
        with pcm_path.open("rb") as source, wave.open(str(tmp), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(sample_rate))
            while remaining > 0:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    raise RuntimeError("partial audio PCM ended before the sealed sample count")
                wav.writeframesraw(block)
                remaining -= len(block)
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, output_path)
        _fsync_directory(output_path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class _StreamingResampler:
    """Small-state resampler for callback blocks.

    The deployed 48 kHz -> 16 kHz path uses a windowed-sinc anti-alias FIR before
    decimation. Other rational rates use continuous linear interpolation while
    retaining only the unconsumed tail between blocks.
    """

    def __init__(self, sr_in: int, sr_out: int) -> None:
        self.sr_in = int(sr_in)
        self.sr_out = int(sr_out)
        if self.sr_in <= 0 or self.sr_out <= 0:
            raise ValueError("audio sample rates must be positive")
        self._factor = self.sr_in // self.sr_out if self.sr_in % self.sr_out == 0 else None
        self._finished = False
        self._tail = np.zeros(0, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_base = 0
        self._received = 0
        self._output_index = 0
        self._fir = None
        self._fir_state = np.zeros(0, dtype=np.float32)
        self._fir_delay = 0
        self._fir_discard = 0
        if self._factor is not None and self._factor > 1:
            # Odd, delay-divisible tap count makes the compensated stream have
            # exactly ceil(n_in/factor) samples after flushing the FIR tail.
            taps = 20 * self._factor + 1
            self._fir_delay = (taps - 1) // 2
            indexes = np.arange(taps, dtype=np.float64) - self._fir_delay
            cutoff = 0.45 / float(self._factor)  # cycles per input sample
            fir = 2.0 * cutoff * np.sinc(2.0 * cutoff * indexes)
            fir *= np.hamming(taps)
            fir /= np.sum(fir)
            self._fir = fir.astype(np.float32)
            self._fir_state = np.zeros(taps - 1, dtype=np.float32)
            self._fir_discard = self._fir_delay // self._factor

    def push(self, block: np.ndarray) -> np.ndarray:
        values = np.asarray(block, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self.sr_in == self.sr_out:
            return values.copy()
        if self._factor is not None:
            return self._push_integer(values)

        if self._pending.size:
            self._pending = np.concatenate((self._pending, values))
        else:
            self._pending = values.copy()
            self._pending_base = self._received
        self._received += values.size
        last_index = self._received - 1
        output: list[float] = []
        while True:
            numerator = self._output_index * self.sr_in
            left = numerator // self.sr_out
            fraction = numerator % self.sr_out
            right = left + (1 if fraction else 0)
            if right > last_index:
                break
            offset = int(left - self._pending_base)
            a = float(self._pending[offset])
            if fraction:
                b = float(self._pending[offset + 1])
                a += (b - a) * (fraction / float(self.sr_out))
            output.append(a)
            self._output_index += 1
        next_left = (self._output_index * self.sr_in) // self.sr_out
        discard = max(0, int(next_left - self._pending_base))
        if discard:
            self._pending = self._pending[discard:].copy()
            self._pending_base += discard
        return np.asarray(output, dtype=np.float32)

    def _push_integer(self, values: np.ndarray) -> np.ndarray:
        if self._fir is None or self._factor is None:
            raise RuntimeError("integer audio resampler is not initialized")
        state_size = self._fir_state.size
        joined = np.concatenate((self._fir_state, values))
        convolved = np.convolve(joined, self._fir, mode="full")
        filtered = convolved[state_size:state_size + values.size]
        first = (-self._received) % self._factor
        output = filtered[first::self._factor].astype(np.float32, copy=False)
        self._received += values.size
        self._fir_state = joined[-state_size:].copy()
        if self._fir_discard:
            discard = min(self._fir_discard, output.size)
            output = output[discard:]
            self._fir_discard -= discard
        return output

    def finish(self) -> np.ndarray:
        """Flush fixed filter delay once; returned samples remain capture-aligned."""
        if self._finished:
            return np.zeros(0, dtype=np.float32)
        self._finished = True
        if self.sr_in == self.sr_out:
            return np.zeros(0, dtype=np.float32)
        if self._factor is not None:
            return self._push_integer(np.zeros(self._fir_delay, dtype=np.float32))

        # Linear fallback: emit any final interpolation point using edge hold.
        output: list[float] = []
        while self._pending.size:
            numerator = self._output_index * self.sr_in
            if numerator >= self._received * self.sr_out:
                break
            left = numerator // self.sr_out
            fraction = numerator % self.sr_out
            offset = int(left - self._pending_base)
            if not 0 <= offset < self._pending.size:
                break
            a = float(self._pending[offset])
            if fraction and offset + 1 < self._pending.size:
                b = float(self._pending[offset + 1])
                a += (b - a) * (fraction / float(self.sr_out))
            output.append(a)
            self._output_index += 1
        return np.asarray(output, dtype=np.float32)


class _StreamingAudioRecording:
    """Dedicated disk thread so PortAudio's callback never waits for storage."""

    SCHEMA_VERSION = "wam_audio_stream_capture_v1"

    def __init__(
        self, session_dir: Path, *, sr_in: int, sr_out: int, t0: float,
        queue_blocks: int = 128,
    ) -> None:
        self.session_dir = session_dir.resolve()
        self.sr_in = int(sr_in)
        self.sr_out = int(sr_out)
        self.t0 = float(t0)
        if not np.isfinite(self.t0):
            raise ValueError("audio capture start time must be finite")
        # The crash-recoverable spool deliberately stays at the sound-card's
        # native rate.  Only the exact [Start, End] prefix is resampled, after
        # End, so FIR look-ahead from a callback block beyond End can never
        # contaminate the production WAV or its health metrics.
        self.raw_path = self.session_dir / "audio.partial.pcm"
        self.model_pcm_path = self.session_dir / "audio.model.partial.pcm"
        self.input_timeline_partial = (
            self.session_dir / "audio_input_times.jsonl.inprogress"
        )
        self.input_timeline_final = self.session_dir / "audio_input_times.jsonl"
        self.timeline_final = self.session_dir / "audio_times.jsonl"
        self.output_path = self.session_dir / "audio.wav"
        self.metadata_path = self.session_dir / "audio_capture.json"
        self._queue: queue.Queue = queue.Queue(maxsize=max(2, int(queue_blocks)))
        self._sentinel = object()
        self._overflow = False
        self._callback_fault: str | None = None
        self._timeline_fault: str | None = None
        self._worker_error: BaseException | None = None
        self._raw_samples = 0
        self._first_t: float | None = None
        self._last_t: float | None = None
        self._last_source_end_t: float | None = None
        self._thread = threading.Thread(
            target=self._run, name="wam-audio-spool", daemon=True,
        )

    @property
    def queue_capacity(self) -> int:
        return self._queue.maxsize

    def start(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.raw_path, self.model_pcm_path, self.input_timeline_partial,
            self.input_timeline_final, self.timeline_final, self.output_path,
            self.metadata_path,
        ):
            if path.exists():
                raise FileExistsError(f"audio recording artifact already exists: {path}")
        _atomic_json(self.metadata_path, {
            "schema_version": self.SCHEMA_VERSION,
            "status": "recording",
            "training_eligible": False,
            "sample_rate": self.sr_out,
            "input_sample_rate": self.sr_in,
            "partial_sample_rate": self.sr_in,
            "partial_encoding": "signed_pcm_s16le_mono",
            "t0_monotonic": self.t0,
            "queue_blocks": self.queue_capacity,
        })
        self._thread.start()

    def submit(
        self, block: np.ndarray, host_t: float, *, callback_status: object = None,
    ) -> None:
        if callback_status:
            self._callback_fault = str(callback_status)
        if self._overflow:
            return
        try:
            self._queue.put_nowait((np.asarray(block, dtype=np.float32).copy(), float(host_t)))
        except queue.Full:
            self._overflow = True

    def _run(self) -> None:
        try:
            with self.raw_path.open("xb") as raw, self.input_timeline_partial.open(
                "x", encoding="utf-8"
            ) as timeline:
                _fsync_directory(self.session_dir)
                while True:
                    item = self._queue.get()
                    try:
                        if item is self._sentinel:
                            break
                        block, block_end_t = item
                        block_start_t = block_end_t - block.size / float(self.sr_in)
                        first = max(0, min(
                            block.size,
                            int(np.ceil((self.t0 - block_start_t) * self.sr_in - 1e-9)),
                        ))
                        selected = block[first:]
                        if selected.size == 0:
                            continue
                        selected_start = block_start_t + first / float(self.sr_in)
                        if self._first_t is None:
                            self._first_t = selected_start
                        if self._last_source_end_t is not None:
                            discontinuity_s = selected_start - self._last_source_end_t
                            tolerance_s = max(2.0 / float(self.sr_in), 0.010)
                            if abs(discontinuity_s) > tolerance_s:
                                self._timeline_fault = (
                                    f"source jumped by {discontinuity_s:.6f} s "
                                    f"(tolerance {tolerance_s:.6f} s)"
                                )
                        self._last_source_end_t = float(block_end_t)
                        pcm = (
                            np.clip(selected, -1.0, 1.0) * 32767.0
                        ).astype("<i2")
                        offset = self._raw_samples
                        raw.write(pcm.tobytes())
                        self._raw_samples += int(pcm.size)
                        row_start = self._first_t + offset / float(self.sr_in)
                        row_end = self._first_t + self._raw_samples / float(self.sr_in)
                        self._last_t = row_end
                        timeline.write(json.dumps({
                            "input_sample_offset": offset,
                            "input_sample_count": int(pcm.size),
                            "t_start": row_start,
                            "t_end": row_end,
                            "source_block_start_t": selected_start,
                            "source_block_end_t": block_end_t,
                        }, sort_keys=True, separators=(",", ":")) + "\n")
                        raw.flush()
                        timeline.flush()
                        # The disk thread, not PortAudio, pays the durability cost.
                        os.fsync(raw.fileno())
                        os.fsync(timeline.fileno())
                    finally:
                        self._queue.task_done()
        except BaseException as exc:  # keep the callback non-throwing; End fails closed
            self._worker_error = exc

    def finish(self, t_end: float) -> AudioRecordingSummary:
        end = float(t_end)
        if not np.isfinite(end) or end <= self.t0:
            raise ValueError("audio capture end time must follow its start")
        while self._thread.is_alive():
            try:
                self._queue.put(self._sentinel, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        if self._worker_error is not None:
            raise RuntimeError(f"audio spool failed: {self._worker_error}") from self._worker_error
        if self._overflow:
            raise RuntimeError("audio spool queue overflowed; capture is incomplete")
        if self._callback_fault is not None:
            raise RuntimeError(
                f"PortAudio callback reported a capture fault: {self._callback_fault}"
            )
        if self._timeline_fault is not None:
            raise RuntimeError(
                f"audio input timeline discontinuity: {self._timeline_fault}"
            )
        first_t = self._first_t if self._first_t is not None else self.t0
        desired_input = max(
            0, int(np.ceil((end - first_t) * self.sr_in - 1e-9)),
        )
        kept_input = min(self._raw_samples, desired_input)
        with self.raw_path.open("r+b") as raw:
            raw.truncate(kept_input * 2)
            raw.flush()
            os.fsync(raw.fileno())
        self._publish_clipped_input_timeline(kept_input)
        if kept_input <= 0:
            raise RuntimeError("audio recording captured no samples")
        kept, peak = self._resample_clipped_input(
            kept_input=kept_input, first_t=first_t, end=end,
        )
        if kept <= 0:
            raise RuntimeError("audio recording produced no model-rate samples")
        signal_authenticity = analyze_audio_pcm16_file(
            self.model_pcm_path,
            sample_rate=self.sr_out,
            n_samples=kept,
        )
        _publish_wav_from_pcm(
            self.model_pcm_path, self.output_path,
            sample_rate=self.sr_out, n_samples=kept,
        )
        actual_end = first_t + kept / float(self.sr_out)
        self._publish_model_timeline(
            n_samples=kept, input_samples=kept_input,
            first_t=first_t, actual_end=actual_end,
        )
        status = (
            STATUS_OK if first_t <= self.t0 + 0.050 and actual_end >= end - 0.050
            else STATUS_NO_DATA
        )
        summary = AudioRecordingSummary(
            path=str(self.output_path), sr=self.sr_out, t_start=first_t,
            t_end=actual_end, n_samples=kept, status=status,
            peak_db=_amp_to_db(peak), sha256=_sha256_file(self.output_path),
            signal_authenticity=signal_authenticity,
        )
        _atomic_json(self.metadata_path, {
            "schema_version": self.SCHEMA_VERSION,
            "status": "sensors_finalized",
            "training_eligible": False,
            "sample_rate": self.sr_out,
            "input_sample_rate": self.sr_in,
            "partial_sample_rate": self.sr_in,
            "partial_encoding": "signed_pcm_s16le_mono",
            "t0_monotonic": self.t0,
            "audio_t0_monotonic": summary.t_start,
            "audio_t_end_monotonic": summary.t_end,
            "partial_n_samples": kept_input,
            "n_samples": summary.n_samples,
            "sha256": summary.sha256,
            "signal_authenticity": signal_authenticity,
        })
        return summary

    def _publish_clipped_input_timeline(self, kept: int) -> None:
        tmp = self.input_timeline_final.with_name(
            f".{self.input_timeline_final.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        remaining = int(kept)
        with self.input_timeline_partial.open(
            "r", encoding="utf-8"
        ) as source, tmp.open("x", encoding="utf-8") as output:
            for line in source:
                if remaining <= 0:
                    break
                row = json.loads(line)
                count = min(int(row["input_sample_count"]), remaining)
                if count <= 0:
                    continue
                row["input_sample_count"] = count
                row["t_end"] = float(row["t_start"]) + count / float(self.sr_in)
                output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                remaining -= count
            if remaining:
                raise RuntimeError("audio timeline ended before retained PCM")
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp, self.input_timeline_final)
        _fsync_directory(self.session_dir)

    def _resample_clipped_input(
        self, *, kept_input: int, first_t: float, end: float,
    ) -> tuple[int, float]:
        resampler = _StreamingResampler(self.sr_in, self.sr_out)
        written = 0
        with self.raw_path.open("rb") as source, self.model_pcm_path.open("xb") as output:
            remaining = int(kept_input)

            def write_model(values: np.ndarray) -> None:
                nonlocal written
                if values.size <= 0:
                    return
                pcm = (np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
                output.write(pcm.tobytes())
                written += int(pcm.size)

            while remaining > 0:
                raw = source.read(min(65_536, remaining) * 2)
                if not raw or len(raw) % 2:
                    raise RuntimeError("partial audio PCM ended before its clipped sample count")
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
                samples /= 32767.0
                write_model(resampler.push(samples))
                remaining -= int(samples.size)
            write_model(resampler.finish())
            output.flush()
            os.fsync(output.fileno())
        _fsync_directory(self.session_dir)

        from_input = max(
            0, int(np.ceil(kept_input * self.sr_out / self.sr_in - 1e-9)),
        )
        from_end = max(
            0, int(np.ceil((end - first_t) * self.sr_out - 1e-9)),
        )
        kept = min(written, from_input, from_end)
        with self.model_pcm_path.open("r+b") as output:
            output.truncate(kept * 2)
            output.flush()
            os.fsync(output.fileno())

        peak = 0.0
        with self.model_pcm_path.open("rb") as output:
            remaining = kept
            while remaining > 0:
                raw = output.read(min(65_536, remaining) * 2)
                if not raw or len(raw) % 2:
                    raise RuntimeError("model-rate PCM ended before its sealed sample count")
                pcm = np.frombuffer(raw, dtype="<i2")
                if pcm.size:
                    peak = max(peak, float(np.max(np.abs(pcm.astype(np.int32)))) / 32767.0)
                remaining -= int(pcm.size)
        return kept, peak

    def _publish_model_timeline(
        self, *, n_samples: int, input_samples: int,
        first_t: float, actual_end: float,
    ) -> None:
        row = {
            "sample_offset": 0,
            "sample_count": int(n_samples),
            "t_start": float(first_t),
            "t_end": float(actual_end),
            "source_input_sample_count": int(input_samples),
            "input_sample_rate": self.sr_in,
            "sample_rate": self.sr_out,
        }
        _atomic_bytes(
            self.timeline_final,
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )

    def abort(self) -> None:
        if self._thread.is_alive():
            while self._thread.is_alive():
                try:
                    self._queue.put(self._sentinel, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._thread.join()
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {"schema_version": self.SCHEMA_VERSION}
        metadata.update(status="interrupted", training_eligible=False)
        _atomic_json(self.metadata_path, metadata)


def recover_partial_recording(session_dir: str | Path) -> dict:
    """Publish ``audio.recovered.wav`` from a durable prefix, never production audio.wav."""
    session = Path(session_dir).expanduser().resolve()
    if (session / "FINALIZATION.json").exists():
        raise RuntimeError("cannot run diagnostic audio recovery on a production-sealed session")
    metadata_path = session / "audio_capture.json"
    raw_path = session / "audio.partial.pcm"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != _StreamingAudioRecording.SCHEMA_VERSION
        or metadata.get("training_eligible") is not False
    ):
        raise RuntimeError("partial audio metadata is not a fail-closed capture contract")
    sample_rate = int(metadata.get("partial_sample_rate", metadata.get("sample_rate", 0)))
    if sample_rate <= 0 or raw_path.is_symlink() or not raw_path.is_file():
        raise RuntimeError("partial audio capture is missing or invalid")
    n_samples = raw_path.stat().st_size // 2
    if n_samples <= 0:
        return {
            "status": "no_recoverable_audio", "training_eligible": False,
            "n_samples": 0,
        }
    output = session / "audio.recovered.wav"
    if output.exists():
        raise FileExistsError(f"diagnostic audio recovery output already exists: {output}")
    _publish_wav_from_pcm(raw_path, output, sample_rate=sample_rate, n_samples=n_samples)
    return {
        "status": "diagnostic_recovered",
        "training_eligible": False,
        "output": output.name,
        "n_samples": n_samples,
        "sample_rate": sample_rate,
        "sha256": _sha256_file(output),
        "source_partial_preserved": raw_path.name,
    }


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


def _time_info_value(time_info, name: str) -> float | None:  # noqa: ANN001
    """Read a PortAudio callback-time field from object- or mapping-like wrappers."""
    try:
        value = getattr(time_info, name)
    except (AttributeError, TypeError):
        try:
            value = time_info[name]
        except (KeyError, TypeError):
            return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _block_end_host_time(time_info, frames: int, sr: int, host_now: float) -> float:  # noqa: ANN001
    """Map PortAudio's ADC start time onto the host monotonic clock.

    PortAudio's ``inputBufferAdcTime`` and ``currentTime`` share a device clock.
    Their difference is therefore a reliable block age even when that clock has
    a different epoch from ``time.monotonic()``.  Fall back to callback arrival
    only when a backend does not provide usable timing information.
    """
    adc_start = _time_info_value(time_info, "inputBufferAdcTime")
    pa_now = _time_info_value(time_info, "currentTime")
    if adc_start is None or pa_now is None or sr <= 0:
        return host_now
    return host_now + (adc_start - pa_now) + max(0, int(frames)) / float(sr)


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

    def since(
        self, cursor: int | None, *, max_samples: int,
    ) -> tuple[np.ndarray, int, int, bool]:
        """Return a bounded absolute-index slice for a live monitor client.

        ``cursor`` is the next sample the client expects. A new client receives
        only the newest ``max_samples``; old camera-room audio is never replayed.
        If a slow client falls behind this bounded ring, ``overrun`` makes the
        discontinuity explicit so its playback scheduler can reset.
        """

        limit = int(max_samples)
        if limit <= 0:
            raise ValueError("audio monitor max_samples must be positive")
        written = int(self._written)
        earliest = max(0, written - self._cap)
        if cursor is None:
            start = max(earliest, written - limit)
            overrun = False
        else:
            requested = int(cursor)
            start = min(max(requested, earliest), written)
            overrun = requested < earliest or requested > written
        count = min(limit, written - start)
        if count <= 0:
            return np.zeros(0, dtype=np.float32), start, start, overrun
        offset = start % self._cap
        end = offset + count
        if end <= self._cap:
            samples = self._buf[offset:end].copy()
        else:
            first = self._cap - offset
            samples = np.concatenate((self._buf[offset:], self._buf[: count - first]))
        return samples, start, start + count, overrun

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
        self._recording: _StreamingAudioRecording | None = None
        self._last_session_dir: Path | None = None
        # Ring continuity ledger (train #72): host end of the last ingested
        # block, and the earliest instant since which the ring tail is known
        # gap-free and fault-free (the only region backfill may seed from).
        self._ring_prev_end: float | None = None
        self._ring_clean_since: float | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "AudioStream":
        import sounddevice as sd  # lazy: only needed for live capture

        if self.channels is None:  # open every input channel the device exposes
            self.channels = int(sd.query_devices(self.device, "input")["max_input_channels"])
        candidate = sd.InputStream(
            device=self.device,
            samplerate=self.sr_in,
            channels=self.channels,
            blocksize=self.block_frames,
            dtype="float32",
            callback=self._callback,
        )
        try:
            candidate.start()
        except BaseException:
            try:
                candidate.close()
            except BaseException:
                pass
            raise
        self._stream = candidate
        return self

    def stop(self) -> None:
        with self._lock:
            recording = self._recording
            self._recording = None
        device = self._stream
        self._stream = None
        errors: list[BaseException] = []
        if recording is not None:
            try:
                recording.abort()
            except BaseException as exc:
                errors.append(exc)
        if device is not None:
            try:
                device.stop()
            except BaseException as exc:
                errors.append(exc)
            try:
                device.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def abort_recording(self) -> bool:
        """Abort only the active disk spool while keeping live monitoring open."""

        with self._lock:
            recording = self._recording
            self._recording = None
        if recording is None:
            return False
        recording.abort()
        return True

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
        host_now = time.monotonic()
        block_end_t = _block_end_host_time(time_info, frames, self.sr_in, host_now)
        self._ingest(
            np.asarray(block, dtype=np.float32), block_end_t,
            callback_status=status,
        )

    def _ingest(
        self, block: np.ndarray, host_t: float, *, callback_status: object = None,
    ) -> None:
        """Push one mono block (testable without hardware)."""
        with self._lock:
            # Ring continuity ledger (train #72): the begin_recording backfill
            # may only seed from a gap-free, fault-free tail. Mirror the
            # spool's own source-jump tolerance; a callback fault taints
            # through this block's END (its samples may be incomplete).
            block_start = host_t - block.size / float(self.sr_in)
            tolerance = max(2.0 / self.sr_in, 0.010)
            if callback_status:
                self._ring_clean_since = host_t
            elif (
                self._ring_prev_end is not None
                and abs(block_start - self._ring_prev_end) > tolerance
            ):
                self._ring_clean_since = block_start
            elif self._ring_clean_since is None:
                self._ring_clean_since = block_start
            self._ring_prev_end = host_t
            self._ring.ingest(block, host_t)
            recording = self._recording
            if recording is not None:
                recording.submit(block, host_t, callback_status=callback_status)

    # -- bounded-memory durable recording ---------------------------------
    def begin_recording(self, session_dir: str | Path, *, t0: float) -> None:
        """Begin an incremental disk spool at the capture barrier.

        This intentionally has no requested-duration argument: callback memory
        remains bounded for a minute or a multi-hour clinical collection alike.
        """
        recording = _StreamingAudioRecording(
            Path(session_dir), sr_in=self.sr_in, sr_out=self.sr_model, t0=t0,
        )
        recording.start()
        with self._lock:
            if self._recording is not None:
                recording.abort()
                raise RuntimeError("audio recording is already active")
            # Boundary backfill (train #72): the capture barrier t0 is the
            # camera first-frame time, which predates this call by the whole
            # Start machinery (robot/force begin, delivery wait) — routinely
            # more than the 50 ms head-coverage tolerance. The blocks covering
            # [t0, now] already streamed into the ring before this call, so
            # seed them into the new spool; its per-block trim clips the seed
            # to exactly t0, and the next live callback continues the source
            # timeline seamlessly (both run under this same lock).
            #
            # The seed is bounded to the ring's CLEAN tail: the ring has no
            # per-sample gap memory, so seeding across an input discontinuity
            # or callback fault would linearize older audio onto wrong
            # timestamps and seal a false-OK head (review train #72). If the
            # clean tail does not reach back to t0, the short seed leaves the
            # head uncovered and End honestly reports NO_DATA -> the declared
            # UNVERIFIED path — exactly the pre-backfill behavior.
            ring_end_t = self._ring.end_time(self.sr_in)
            backfill_s = ring_end_t - float(t0)
            if backfill_s > 0 and self._ring.written > 0:
                clean_since = self._ring_clean_since
                clean_span_s = (
                    ring_end_t - clean_since
                    if clean_since is not None else 0.0
                )
                n_backfill = min(
                    int(np.ceil(backfill_s * self.sr_in)) + self.sr_in // 10,
                    max(0, int(clean_span_s * self.sr_in)),
                    self._ring.written,
                )
                if n_backfill > 0:
                    seed = self._ring.last(n_backfill)
                    if seed.size:
                        recording.submit(seed, ring_end_t)
            self._recording = recording
            self._last_session_dir = Path(session_dir).expanduser().resolve()

    def end_recording(self, *, t_end: float) -> AudioRecordingSummary:
        """Detach the callback at End, drain accepted blocks, and seal audio.wav."""
        with self._lock:
            recording = self._recording
            self._recording = None
        if recording is None:
            raise RuntimeError("audio recording is not active")
        try:
            return recording.finish(t_end)
        except BaseException:
            recording.abort()
            raise

    @property
    def recording_queue_capacity(self) -> int:
        with self._lock:
            return 0 if self._recording is None else self._recording.queue_capacity

    def cleanup_recording_staging(self) -> None:
        """Remove raw spools after final WAV/timelines are durably published."""
        if self._last_session_dir is None:
            return
        for name in (
            "audio.partial.pcm", "audio.model.partial.pcm",
            "audio_input_times.jsonl.inprogress", "audio_times.jsonl.inprogress",
        ):
            try:
                (self._last_session_dir / name).unlink()
            except (FileNotFoundError, OSError):
                pass

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

    def monitor_chunk(
        self, cursor: int | None, *, max_samples: int = 4_800,
    ) -> dict[str, object]:
        """Return a sequential native-rate chunk for a local browser monitor.

        This accessor never opens a second device and never writes to disk. It
        reads the same bounded ring populated by the formal capture stream.
        """

        now = time.monotonic()
        with self._lock:
            samples, start, next_sample, overrun = self._ring.since(
                cursor, max_samples=max_samples,
            )
            age = (
                now - self._ring.anchor_t
                if self._ring.written else float("inf")
            )
        if samples.size == 0:
            status = STATUS_NO_DATA
        elif age > self.stale_timeout_s:
            status = STATUS_STALE
        else:
            status = STATUS_OK
        return {
            "samples": samples,
            "sample_rate": self.sr_in,
            "start_sample": start,
            "next_sample": next_sample,
            "overrun": overrun,
            "status": status,
        }


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
