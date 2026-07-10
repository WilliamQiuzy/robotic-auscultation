"""Per-point auscultation completion feedback + auto-advance.

Answers "when/how do we know the doctor has auscultated this point, then switch to the next?" With
N points, we sweep them; each point runs a small state machine and auto-advances when done.

OCCLUSION-IMMUNE BY DESIGN: completion uses contact force (Hall) + audio quality Q + EE-to-target
distance (kinematics) + respiratory cycles — NONE of which need the camera to see the point. The arm
necessarily occludes the landmark at contact, but by then vision's job (pre-contact localization) is
over; the target is held in 3D and the display just projects it (dashed) while force/audio/kinematics
judge completion. So arm occlusion does not affect the feedback.

    sweep = PointSweep(points)                       # points: list with .xyz (3D on the back)
    idx, state, status = sweep.update(ee_xyz, force_n, q_usable, new_cycle)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(str, Enum):
    APPROACH = "APPROACH"       # EE not yet at the target
    CONTACT = "CONTACT"         # at the target but force not in the auscultation band
    LISTENING = "LISTENING"     # good contact + good audio → accumulating respiratory cycles
    DONE = "DONE"               # enough good cycles → this point is auscultated


@dataclass
class FeedbackConfig:
    reach_m: float = 0.02       # EE within this of the target 3D ⇒ "reached"
    f_min: float = 3.0          # contact-force band (N) — from the force calibration
    f_max: float = 5.0
    q_min: float = 0.6          # P(usable) threshold for "good audio"
    need_cycles: int = 2        # good respiratory cycles required to call a point done (2-3)


@dataclass
class PointFeedback:
    """State machine for ONE point. Feed it the sensors each tick; it counts good cycles → DONE."""
    cfg: FeedbackConfig = field(default_factory=FeedbackConfig)
    good_cycles: int = 0
    state: State = State.APPROACH

    def reset(self) -> None:
        self.good_cycles = 0
        self.state = State.APPROACH

    def update(self, ee_to_target_m: float, force_n: float, q_usable: float, new_cycle: bool) -> State:
        c = self.cfg
        reached = ee_to_target_m <= c.reach_m
        contact = c.f_min <= force_n <= c.f_max
        good = reached and contact and q_usable >= c.q_min
        if not reached:
            self.state = State.APPROACH
        elif not contact:
            self.state = State.CONTACT
        else:
            self.state = State.LISTENING
            if new_cycle and good:                                  # count a completed respiratory cycle
                self.good_cycles += 1
            if self.good_cycles >= c.need_cycles:
                self.state = State.DONE
        return self.state

    def status(self, ee_to_target_m: float, force_n: float, q_usable: float) -> str:
        c = self.cfg
        if self.state is State.APPROACH:
            return f"→ move to target  ({ee_to_target_m * 100:.0f} cm away)"
        if self.state is State.CONTACT:
            lo_hi = "press harder" if force_n < c.f_min else "ease off"
            return f"contact — {lo_hi}  (F={force_n:.1f} N, target {c.f_min:.0f}-{c.f_max:.0f} N)"
        if self.state is State.LISTENING:
            q = "clean" if q_usable >= c.q_min else "noisy — hold still"
            return f"listening… {self.good_cycles}/{c.need_cycles} cycles  ({q})"
        return "✓ done — next point"


@dataclass
class PointSweep:
    """Sweep a list of points; auto-advance to the next when the current one is DONE."""
    points: list
    cfg: FeedbackConfig = field(default_factory=FeedbackConfig)
    idx: int = 0
    fb: PointFeedback = field(default_factory=PointFeedback)
    completed: list = field(default_factory=list)

    def __post_init__(self):
        self.fb = PointFeedback(self.cfg)

    @staticmethod
    def _dist(a, b) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def update(self, ee_xyz, force_n: float, q_usable: float, new_cycle: bool):
        """Returns (active_index, state, status_text). Advances + resets when a point completes."""
        tp = self.points[self.idx]
        d = self._dist(ee_xyz, tp.xyz) if getattr(tp, "xyz", None) else 1e9
        st = self.fb.update(d, force_n, q_usable, new_cycle)
        status = self.fb.status(d, force_n, q_usable)
        if st is State.DONE:
            if self.idx not in self.completed:
                self.completed.append(self.idx)
            self.idx = (self.idx + 1) % len(self.points)           # auto-advance
            self.fb.reset()
        return self.idx, st, status

    @property
    def all_done(self) -> bool:
        return len(self.completed) >= len(self.points)


if __name__ == "__main__":  # self-test: one point should progress APPROACH→CONTACT→LISTENING→DONE→advance
    from types import SimpleNamespace
    pts = [SimpleNamespace(name="RUL", xyz=(0.0, 0.0, 0.5)),
           SimpleNamespace(name="LUL", xyz=(0.1, 0.0, 0.5))]
    sw = PointSweep(pts, FeedbackConfig(need_cycles=2))
    seq = []
    # approach (far), touch (bad force), good contact over 2 cycles → done → advance to LUL
    seq.append(sw.update((0.0, 0.0, 0.2), 0.0, 0.0, False)[1])         # APPROACH
    seq.append(sw.update((0.0, 0.0, 0.49), 1.5, 0.0, False)[1])        # CONTACT (force low)
    seq.append(sw.update((0.0, 0.0, 0.49), 4.0, 0.9, True)[1])         # LISTENING cycle1
    idx, st, status = sw.update((0.0, 0.0, 0.49), 4.0, 0.9, True)      # cycle2 → DONE → advance
    seq.append(st)
    print("states:", [s.value for s in seq], "| advanced to:", pts[idx].name, "| completed:", sw.completed)
    assert [s.value for s in seq] == ["APPROACH", "CONTACT", "LISTENING", "DONE"]
    assert idx == 1 and sw.completed == [0]
    print("OK: APPROACH→CONTACT→LISTENING→DONE, auto-advanced RUL→LUL after 2 good cycles")
