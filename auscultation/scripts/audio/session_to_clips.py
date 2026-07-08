"""Cut a recorded session's audio into per-segment annotation clips → the preference pool.

Closes the collection pipeline: record_session.py (record) → THIS (respiratory-cycle segmentation +
audio cut) → preference_annotator.py (doctors rank). Reuses session_to_lerobot's contact-aware
respiratory segmenter, so the clips are the same 2-3-cycle units the reward pipeline uses. Works on
any session — our ThinkLabs recordings now, robot teleop sessions later (audio from teleop needs
annotating too).

    python scripts/audio/session_to_clips.py data/sessions/session_XXXX --pool data/annotation_pool [--contact-only]

Each segment → <pool>/<session>__seg<NN>.wav (16 kHz). Point preference_annotator.py at <pool>.
"""
from __future__ import annotations

import argparse
import os
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # auscultation/
sys.path[:0] = [os.path.join(ROOT, "scripts"), os.path.join(ROOT, "ausc"), os.path.join(ROOT, "ausc", "audio")]
from session_io import Session                       # noqa: E402
from session_to_lerobot import _read_force, build_segments  # noqa: E402


def _write_wav(path: str, y: np.ndarray, sr: int) -> None:
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


def extract(session_dir: str, pool: str, contact_only: bool = False) -> list[str]:
    os.makedirs(pool, exist_ok=True)
    s = Session(session_dir)
    sid = os.path.basename(session_dir.rstrip("/"))
    ft, fn = _read_force(session_dir)
    lo, hi = s.t0 - 0.5, s.t0 + s.duration + 0.5
    win = [(t, v) for t, v in zip(ft, fn) if lo <= t <= hi]
    ft = [t for t, _ in win]; fn = [v for _, v in win]
    segs = build_segments(s, ft, fn, sid, q_fn=None)          # boundaries only (no Q needed to cut)
    written = []
    for i, seg in enumerate(segs):
        if contact_only and not seg.contact.gate:
            continue
        y = s.audio_window(seg.audio_t_end, seconds=seg.audio_t_end - seg.audio_t_start)
        if y is None or len(y) < s.audio_sr // 4:
            continue
        p = os.path.join(pool, f"{sid}__seg{i:02d}.wav")
        _write_wav(p, y, s.audio_sr); written.append(p)
    print(f"[clips] {sid}: {len(written)} clips -> {pool}"
          f"{' (contact-only)' if contact_only else ''}")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--pool", default=os.path.join(ROOT, "data", "annotation_pool"))
    ap.add_argument("--contact-only", action="store_true",
                    help="only export good-contact (gate=True) segments — the clips worth grading")
    a = ap.parse_args()
    extract(a.session_dir, a.pool, a.contact_only)
