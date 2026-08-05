"""Auscultation site semantics — the single source of truth that turns each opaque
point-conditioned target code (R1..L5) into its ANATOMICAL meaning, and turns that into
the natural-language instruction that fills the VLA "task"/"L" field.

WHY this file exists, and why BOTH the collection platform and the LeRobot converter import
it: the point codes (R1..L5) are opaque tokens a VLA cannot generalize from. Grounding each
code in an anatomical phrase ("right upper posterior lung field") lets the policy reuse a
language model's understanding of anatomy and follow described-but-unseen targets. The
mapping is defined ONCE here (clinician-reviewed), so scaling to any number of recordings
costs ZERO per-segment human annotation.

Design rules (match the collection intent):
  * SEGMENT level, not per frame. One instruction per auscultation target interval.
  * State the PURPOSE (which anatomical site to auscultate) — never coordinates, never
    per-recording specifics. Stating the goal without over-specifying keeps the policy from
    overfitting to numbers or phrasing. "How well it was done" is the reward's job (Q), not
    the instruction's.
  * Phrasing diversity (anti-overfit) comes from a FIXED template pool sampled
    deterministically per (session, site) -- written once, applied to all data, no human.

.. warning::
   LATERALITY — CLINICIAN REVIEW REQUIRED. The codes name a 5x2 posterior grid: the FIRST
   letter is the patient's ANATOMICAL side (``R`` = patient's right lung, ``L`` = patient's
   left), the DIGIT is the level top->bottom (``1`` = upper/apical ... ``5`` = lower/basal).
   The camera views the BACK, where patient-left appears on image-right (see
   ``ausc_targets.DEFAULT_TARGETS``), so IMAGE position is mirrored vs anatomy. This table is
   keyed on the anatomical side the code DECLARES, not image position. Before training on
   these labels a clinician must confirm each code's physical point actually sits on the
   declared side/level. Treat the strings below as a reviewable draft.
"""
from __future__ import annotations

import random
from typing import Iterable


#: code -> anatomical site phrase. Clinician-reviewed draft (5x2 posterior lung grid).
SITE_ANATOMY: dict[str, str] = {
    "R1": "right upper posterior lung field",
    "R2": "right upper-middle posterior lung field",
    "R3": "right middle posterior lung field",
    "R4": "right lower-middle posterior lung field",
    "R5": "right lower posterior lung field",
    "L1": "left upper posterior lung field",
    "L2": "left upper-middle posterior lung field",
    "L3": "left middle posterior lung field",
    "L4": "left lower-middle posterior lung field",
    "L5": "left lower posterior lung field",
}

#: Fixed paraphrase pool (anti-overfit). Written ONCE; applied to ALL recordings/segments
#: automatically. Each entry MUST contain ``{site}`` and stay at the level of purpose
#: (auscultate this site) -- no coordinates, no per-recording detail.
INSTRUCTION_TEMPLATES: tuple[str, ...] = (
    "auscultate the {site}",
    "listen to the {site}",
    "place the stethoscope on the {site}",
    "move the stethoscope to the {site}",
    "position the stethoscope over the {site} and take a reading",
    "record breath sounds at the {site}",
    "take an auscultation reading at the {site}",
)

#: Fallback site for a code not in the table (should not happen for a mapped point set).
DEFAULT_SITE = "posterior lung field"


def site_anatomy(name: object) -> str:
    """Anatomical phrase for a target code (falls back to a generic posterior-lung site)."""
    return SITE_ANATOMY.get(str(name), DEFAULT_SITE)


def site_instruction(name: object) -> str:
    """The CANONICAL, stable instruction for a site.

    Used by the UI and recorded into the raw-session manifest so the operator and the stored
    data both see the same semantic goal, independent of any training-time augmentation.
    """
    return "auscultate the " + site_anatomy(name)


def phrase_site_instruction(name: object, *, seed: object = None) -> str:
    """A phrasing-augmented instruction for the VLA ``task`` field.

    Deterministic given ``seed`` -- pass a stable key such as ``f"{session_id}:{name}"`` so a
    dataset is reproducible while phrasing still varies across sites and sessions. That is
    instruction augmentation with zero human cost; omit ``seed`` for a nondeterministic pick.
    """
    rng = random.Random(repr(seed)) if seed is not None else random.Random()
    return rng.choice(INSTRUCTION_TEMPLATES).format(site=site_anatomy(name))


def instruction_map(names: Iterable[object]) -> dict[str, str]:
    """``{code -> canonical instruction}`` for a point set.

    Recorded into the session manifest so every raw session is self-describing about what each
    point MEANS, and so the converter can bind to the exact semantics that were collected.
    """
    return {str(n): site_instruction(n) for n in names}


def unmapped_codes(names: Iterable[object]) -> list[str]:
    """Codes with no anatomical mapping -- lets collection warn/fail-closed on an unreviewed
    point set instead of silently emitting the generic fallback."""
    return [str(n) for n in names if str(n) not in SITE_ANATOMY]


__all__ = [
    "SITE_ANATOMY",
    "INSTRUCTION_TEMPLATES",
    "DEFAULT_SITE",
    "site_anatomy",
    "site_instruction",
    "phrase_site_instruction",
    "instruction_map",
    "unmapped_codes",
]
