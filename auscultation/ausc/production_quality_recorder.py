"""Session-level recorder for the production quality (Q) write-side.

Mirroring the geometry recorder, this is the single component the live
production capture drives for the quality channel.  It encapsulates the correct
production wiring that no live code performs today:

* a :class:`QualityPacketLogWriter` bound to the session (which persists the
  lossless f32 model-input windows and the staged packet log), and
* a *production*-mode :class:`StreamingQuality` fed that writer, so every window
  carries its ``raw_source`` provenance (Step 2b) and its persisted model input.

Per window it drives one :meth:`StreamingQuality.update`; :meth:`finalize`
issues the production authenticity receipt and publishes
``quality_inference_packets.jsonl`` + ``quality_inference_authenticity.json``.

The *scorer* is an injected seam: on the box it is the activated Torch/CUDA
quality model; in tests it is any ``(samples, sample_rate) -> float`` callable.
So the composition, ordering, and published artifacts are unit-testable without
Torch, exactly as the replay runner's ``model_run`` seam is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from ausc.realtime_q import QualityPacketLogWriter, StreamingQuality


class ProductionQualityRecorderError(RuntimeError):
    """The production quality recorder cannot record a trustworthy session."""


class ProductionQualitySessionRecorder:
    """Compose the packet-log writer with a production StreamingQuality."""

    def __init__(
        self,
        session_dir: "str | Path",
        *,
        scorer: Callable[..., float],
        model_release_id: str,
        model_sha256: str,
        policy: Mapping[str, object],
    ) -> None:
        if not callable(scorer):
            raise ProductionQualityRecorderError("quality scorer seam is not callable")
        self._session_dir = Path(session_dir).absolute()
        self._writer = QualityPacketLogWriter(self._session_dir)
        # Production mode with a packet-log writer is the only configuration that
        # produces training-eligible Q evidence; StreamingQuality itself refuses a
        # diagnostic writer and stays NOT_READY without one.
        self._quality = StreamingQuality(
            scorer=scorer,
            mode="production",
            model_release_id=model_release_id,
            model_sha256=model_sha256,
            policy=policy,
            packet_log_writer=self._writer,
        )
        self._finalized = False
        self._window_count = 0

    @property
    def window_count(self) -> int:
        return self._window_count

    @property
    def finalized(self) -> bool:
        return self._finalized

    def readiness(self) -> dict[str, object]:
        return self._quality.readiness()

    def record_window(
        self,
        *,
        audio_window,  # AudioWindow-compatible (samples, sr, t_start, t_end, status, raw_source)
        force_sample,  # ForceSample-compatible
        inference_sequence: int,
        inference_monotonic_s: float,
        observed_monotonic_s: float | None = None,
    ) -> dict[str, object]:
        """Score one window and durably record its packet + persisted input."""

        if self._finalized:
            raise ProductionQualityRecorderError("recorder is already finalized")
        result = self._quality.update(
            audio_window=audio_window,
            force_sample=force_sample,
            inference_sequence=inference_sequence,
            inference_monotonic_s=inference_monotonic_s,
            observed_monotonic_s=observed_monotonic_s,
        )
        self._window_count += 1
        return result

    def finalize(self, *, now_monotonic_s: float | None = None) -> dict[str, object]:
        """Issue the production authenticity receipt and publish the packet log."""

        if self._finalized:
            raise ProductionQualityRecorderError("recorder is already finalized")
        authenticity = self._quality.quality_authenticity(
            now_monotonic_s=now_monotonic_s
        )
        receipt = self._writer.finalize(authenticity, now_monotonic_s=now_monotonic_s)
        self._finalized = True
        return receipt


__all__ = [
    "ProductionQualityRecorderError",
    "ProductionQualitySessionRecorder",
]
