"""Segment & reward schema for imitation-learning data.  Design: docs/audio_reward_design.md (LOCKED).

A SEGMENT = 2-3 respiratory cycles (~10-15 s), cycle-aligned to the Hall respiratory onsets. It is
the unit at which audio quality Q, the reward, and doctor labels live (audio is the task's success
*measurement* → it feeds the reward, not the v1 policy observation).

Key principle: store reward INGREDIENTS (recomputable) and keep `reward`/`weight` as DERIVED,
versioned fields, so a better reward model can relabel all history WITHOUT re-collecting and WITHOUT
rewriting the LeRobot frames — each frame carries only `segment_index` and joins to a Segment here.

    segs = load_segments("data/sessions/session_XXXX/segments.jsonl")
    seg  = attach_reward(segs[0], reward_qgate, version="qgate-1")   # returns a NEW Segment
    w    = filtered_bc_weight(seg)                                   # weight in [0,1] for weighted BC

`q_usable` is P(usable) ∈ [0,1] (higher = better), matching realtime_multimodal_pipeline.md §4's
`P_usable = gate · Q_audio`; the reward maximizes it.  This module is pure-stdlib (no torch): it is
the schema/contract, not the encoder.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Optional

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Contact:
    """Contact-force summary over the segment (the reward's gate term)."""
    force_mean_n: float
    force_in_band_frac: float          # fraction of samples within [F_min, F_max] (target ~4 N)
    force_stable: bool                 # low variance over the segment
    gate: bool                         # in-band AND stable → contact good enough to judge audio


@dataclass(frozen=True)
class RewardIngredients:
    """Everything needed to RECOMPUTE reward later; never freeze the reward itself into the data."""
    q_usable: float                    # P(usable) ∈ [0,1] at collection time (from the frozen Q head)
    heart_contamination: float         # low-band(20-150Hz)/total energy ratio; high → probe too anterior
    snr_db: Optional[float] = None     # coarse SNR proxy of the lung-band audio
    q_stream: list[float] = field(default_factory=list)  # dense ~1 Hz streaming Q over the segment


@dataclass(frozen=True)
class Segment:
    """One 2-3 cycle auscultation segment. Reward/weight are DERIVED (nullable, versioned)."""
    segment_id: str
    session_id: str
    t_start: float                     # host monotonic
    t_end: float
    cycle_onsets: list[float]          # inspiration onsets (Hall), host monotonic
    n_cycles: int
    audio_t_start: float               # pointer into the sidecar wav (host monotonic)
    audio_t_end: float
    contact: Contact
    ingredients: RewardIngredients
    doctor_label: Optional[str] = None       # "usable" | "unusable" | None (absolute label, if any)
    reward: Optional[float] = None           # DERIVED — recompute, don't trust as ground truth
    weight: Optional[float] = None           # DERIVED — filtered-BC weight in [0,1]
    reward_model_version: Optional[str] = None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class Preference:
    """A pairwise doctor ranking over two segments — trains the preference (Bradley-Terry) reward model."""
    pair_id: str
    segment_a: str
    segment_b: str
    preferred: str                     # "a" | "b" | "tie"
    annotator: str
    ts_wall: str                       # ISO time, passed in (no wall-clock read here)
    schema_version: str = SCHEMA_VERSION


# -- reward derivations (recomputable from ingredients; the anchor is gate · Q) --------------------

def reward_qgate(seg: Segment) -> float:
    """Anchor reward `R = gate · Q(usable)` (realtime_multimodal_pipeline §4). No contact → 0."""
    return (1.0 if seg.contact.gate else 0.0) * float(seg.ingredients.q_usable)


def filtered_bc_weight(seg: Segment, *, mode: str = "soft", threshold: float = 0.5) -> float:
    """Per-segment weight for reward-weighted / filtered BC.

    Uses `seg.reward` if a reward model already attached one, else falls back to the `gate · Q` anchor.
    - "soft": the reward value itself (reward-weighted regression) → weight ∈ [0,1].
    - "hard": 1.0 iff reward ≥ threshold else 0.0 (filtered BC). Default threshold tracks the Q
      operating point; tune per the deployed Q head (doctor Q v2 P(usable) op-point ≈ 0.64).
    """
    r = seg.reward if seg.reward is not None else reward_qgate(seg)
    if mode == "hard":
        return 1.0 if r >= threshold else 0.0
    if mode == "soft":
        return max(0.0, min(1.0, r))
    raise ValueError(f"unknown mode {mode!r} (use 'soft' or 'hard')")


def attach_reward(seg: Segment, fn: Callable[[Segment], float], *, version: str,
                  weight_mode: str = "soft") -> Segment:
    """Return a NEW Segment with reward/weight/version filled in (immutable — never mutate in place)."""
    r = float(fn(seg))
    tmp = replace(seg, reward=r, reward_model_version=version)
    return replace(tmp, weight=filtered_bc_weight(tmp, mode=weight_mode))


# -- validation & (de)serialization ----------------------------------------------------------------

def validate(seg: Segment) -> None:
    """Fail fast on a malformed segment (called at write time and after relabel)."""
    if seg.t_end <= seg.t_start:
        raise ValueError(f"{seg.segment_id}: t_end <= t_start")
    if not (seg.audio_t_start <= seg.audio_t_end):
        raise ValueError(f"{seg.segment_id}: audio pointer reversed")
    if not (0.0 <= seg.ingredients.q_usable <= 1.0):
        raise ValueError(f"{seg.segment_id}: q_usable out of [0,1]")
    if not (0.0 <= seg.contact.force_in_band_frac <= 1.0):
        raise ValueError(f"{seg.segment_id}: force_in_band_frac out of [0,1]")
    if seg.n_cycles < 1:
        raise ValueError(f"{seg.segment_id}: n_cycles < 1")
    if seg.reward is not None and seg.reward_model_version is None:
        raise ValueError(f"{seg.segment_id}: reward set but reward_model_version missing")
    if seg.doctor_label not in (None, "usable", "unusable"):
        raise ValueError(f"{seg.segment_id}: bad doctor_label {seg.doctor_label!r}")


def segment_to_dict(seg: Segment) -> dict:
    return asdict(seg)


def segment_from_dict(d: dict) -> Segment:
    d = dict(d)
    d["contact"] = Contact(**d["contact"])
    d["ingredients"] = RewardIngredients(**d["ingredients"])
    return Segment(**d)


def load_segments(path: str) -> list[Segment]:
    with open(path) as f:
        return [segment_from_dict(json.loads(line)) for line in f if line.strip()]


def save_segments(path: str, segs: list[Segment]) -> None:
    for s in segs:
        validate(s)
    with open(path, "w") as f:
        for s in segs:
            f.write(json.dumps(segment_to_dict(s)) + "\n")


if __name__ == "__main__":  # self-test: build → round-trip → derive reward/weight → validate
    seg = Segment(
        segment_id="s0", session_id="session_demo",
        t_start=0.0, t_end=12.0, cycle_onsets=[0.5, 4.2, 8.0], n_cycles=3,
        audio_t_start=0.0, audio_t_end=12.0,
        contact=Contact(force_mean_n=4.1, force_in_band_frac=0.92, force_stable=True, gate=True),
        ingredients=RewardIngredients(q_usable=0.82, heart_contamination=0.11, snr_db=18.0,
                                      q_stream=[0.7, 0.8, 0.85, 0.82]),
        doctor_label="usable",
    )
    validate(seg)
    seg2 = attach_reward(seg, reward_qgate, version="qgate-1")
    back = segment_from_dict(segment_to_dict(seg2))
    assert back == seg2, "round-trip mismatch"
    print(f"reward(gate·Q) = {seg2.reward:.3f}  weight(soft) = {seg2.weight:.3f}  "
          f"weight(hard@0.5) = {filtered_bc_weight(seg2, mode='hard'):.0f}  version = {seg2.reward_model_version}")
    # a no-contact segment must reward 0 regardless of Q
    ncg = replace(seg, contact=replace(seg.contact, gate=False))
    assert reward_qgate(ncg) == 0.0
    print("OK: schema round-trips, gate·Q reward, no-contact→0, validation passes")
