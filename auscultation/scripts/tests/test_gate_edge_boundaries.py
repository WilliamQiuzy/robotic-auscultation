"""Boundary regression tests for the 2026-08-29 gate-loosening audit.

Ported from the 21-case edge harness that characterized every session-killing
cliff, updated to the operator-approved policies:
  - audio_signal_degraded_real_v3 (fabrication hard-fails, degraded-real ledgers)
  - distant-bystander / malformed-detection downgrade in select_single_person
  - max_missed_frames 10, min_area_ratio 0.25 back-confirmation defaults
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
AUSC_ROOT = SCRIPTS.parent
if str(AUSC_ROOT) not in sys.path:
    sys.path.insert(0, str(AUSC_ROOT))

from posterior_gate import (  # noqa: E402
    OperatorConfirmedBackGate,
    _boxes_match,
    select_single_person,
)
from ausc.audio.audio_stream import (  # noqa: E402
    AudioSignalAuthenticityAccumulator,
    AudioSignalAuthenticityError,
)

SHAPE = (720, 1280)
AREA = SHAPE[0] * SHAPE[1]
SR = 16_000


def _det(conf, area_frac, cx=640, cy=360):
    side = (area_frac * AREA) ** 0.5
    return {
        "box": [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2],
        "box_confidence": conf,
    }


# ---------- person-presence boundaries ----------

def test_confidence_floor_is_inclusive_at_0_20():
    assert select_single_person([_det(0.20, 0.06)], image_shape=SHAPE)["person_present"] is True
    r = select_single_person([_det(0.1999, 0.06)], image_shape=SHAPE)
    assert (r["person_present"], r["reason"]) == (False, "NO_PERSON")


def test_area_floor_is_inclusive_at_one_percent():
    assert select_single_person([_det(0.9, 0.0101)], image_shape=SHAPE)["person_present"] is True
    r = select_single_person([_det(0.9, 0.0099)], image_shape=SHAPE)
    assert (r["person_present"], r["reason"]) == (False, "NO_PERSON")


def test_dominance_floor_boundary_at_five_percent():
    # one dominant + one just-under-dominant bystander -> present
    r = select_single_person(
        [_det(0.9, 0.30), _det(0.6, 0.049, cx=150, cy=200)], image_shape=SHAPE,
    )
    assert (r["person_present"], r["bystander_people"]) == (True, 1)
    # the same second box grown to the dominance floor -> ambiguous again
    r = select_single_person(
        [_det(0.9, 0.30), _det(0.6, 0.051, cx=150, cy=200)], image_shape=SHAPE,
    )
    assert (r["person_present"], r["reason"]) == (False, "AMBIGUOUS_PEOPLE")


def test_single_small_person_is_still_selectable_without_dominance():
    r = select_single_person([_det(0.9, 0.02)], image_shape=SHAPE)
    assert (r["person_present"], r["bystander_people"]) == (True, 0)


# ---------- confirmed-box continuity boundaries ----------

_MATCH_KW = dict(
    frame_shape=SHAPE, min_iou=0.20, max_center_shift_fraction=0.20,
    min_area_ratio=0.25, max_area_ratio=3.0,
)
_REF = [500.0, 100.0, 900.0, 700.0]


def _shifted(dx=0.0, scale=1.0):
    w, h = 400 * scale ** 0.5, 600 * scale ** 0.5
    cx, cy = 700 + dx, 400
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_lean_forward_shrink_survives_down_to_the_quarter_area_floor():
    assert _boxes_match(_REF, _shifted(scale=0.26), **_MATCH_KW) is True
    assert _boxes_match(_REF, _shifted(scale=0.24), **_MATCH_KW) is False


def test_natural_sway_of_eight_percent_diagonal_survives():
    diag = (SHAPE[0] ** 2 + SHAPE[1] ** 2) ** 0.5
    assert _boxes_match(_REF, _shifted(dx=0.08 * diag), **_MATCH_KW) is True


# ---------- back-confirmation temporal boundaries ----------

def _make_confirmed_gate(now, decisions):
    it = iter(decisions)
    raw = SimpleNamespace(assess=lambda frame: {**next(it), "frame_index": frame.index})
    gate = OperatorConfirmedBackGate(raw, clock=lambda: now[0])
    gate.assess(SimpleNamespace(index=1, t=now[0]))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    return gate


def test_nine_transient_misses_survive_and_the_tenth_expires():
    one = select_single_person([_det(0.9, 0.30)], image_shape=SHAPE)
    none = select_single_person([], image_shape=SHAPE)
    now = [100.0]
    gate = _make_confirmed_gate(now, [one] + [none] * 10)
    for index in range(2, 11):  # nine misses
        now[0] += 0.05
        gate.assess(SimpleNamespace(index=index, t=now[0]))
        assert gate.confirmation_snapshot()["active"] is True
    now[0] += 0.05
    gate.assess(SimpleNamespace(index=11, t=now[0]))  # tenth miss
    snapshot = gate.confirmation_snapshot()
    assert snapshot["active"] is False
    assert snapshot["reason"] == "CONFIRMATION_TARGET_LOST"


# ---------- audio authenticity boundaries ----------

def _finish(samples):
    acc = AudioSignalAuthenticityAccumulator(SR)
    acc.update(np.asarray(samples, dtype=np.int16))
    return acc.finish()


def _noise(seconds, rms, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, rms, int(seconds * SR))
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def test_fabrication_class_pcm_still_hard_fails():
    with pytest.raises(AudioSignalAuthenticityError, match="all-zero"):
        _finish(np.zeros(SR, np.int16))
    with pytest.raises(AudioSignalAuthenticityError, match="constant"):
        _finish(np.full(SR, 55, np.int16))


def test_scarlett_dropout_beyond_two_seconds_degrades_with_exact_region():
    sig = np.concatenate([
        _noise(30, 200), np.zeros(int(2.05 * SR), np.int16), _noise(30, 200, seed=8),
    ])
    receipt = _finish(sig)
    assert receipt["status"] == "PASS_DEGRADED"
    assert receipt["degradations"] == ["repeated_frozen_blocks"]
    (region,) = receipt["evidence"]["frozen_regions"]
    assert region["start_s"] == pytest.approx(30.0, abs=0.06)
    assert region["duration_s"] == pytest.approx(2.05, abs=0.06)


def test_dropout_below_two_seconds_still_passes_clean():
    sig = np.concatenate([
        _noise(30, 200), np.zeros(int(1.90 * SR), np.int16), _noise(30, 200, seed=8),
    ])
    assert _finish(sig)["status"] == "PASS"


def test_low_gain_and_clipping_degrade_instead_of_voiding():
    low = _finish(np.tile(np.asarray([-1, 0, 1], np.int16), 20 * SR // 3))
    assert low["status"] == "PASS_DEGRADED"
    assert "minimum_effective_ac_energy" in low["degradations"]

    clipped = _finish(np.concatenate([
        np.full(int(30 * SR), 32760, np.int16), _noise(570, 300),
    ]))
    assert "severe_clipping" in clipped["degradations"]


def test_degraded_receipt_is_deterministic_for_producer_and_verifier():
    sig = np.concatenate([
        _noise(10, 200), np.zeros(int(2.5 * SR), np.int16), _noise(10, 200, seed=9),
    ])
    assert _finish(sig) == _finish(sig)
