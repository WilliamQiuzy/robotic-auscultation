"""Pure supervision from deployable observations, with no simulator force labels.

The nominal 4 N target and three-second hold are engineering settings for this
simulation experiment, not clinically established pressure or recording duration.
This module requests phases; the caller must execute bounded motion and a safe
outward retraction. It neither moves a robot nor guarantees physical safety.
"""

from dataclasses import dataclass, fields
from enum import Enum
import math
from numbers import Real
from typing import Optional, Tuple


class Phase(str, Enum):
    APPROACH = "APPROACH"
    CONTACT = "CONTACT"
    HOLD = "HOLD"
    RETRACT = "RETRACT"
    DONE = "DONE"
    ABORT = "ABORT"


def _finite_real(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _elapsed(value, duration):
    # Avoid an extra sample caused solely by repeated floating-point addition.
    return value >= duration or math.isclose(value, duration, rel_tol=1e-12)


@dataclass(frozen=True)
class PhaseConfig:
    target_force_N: float = 4.0
    force_tolerance_N: float = 0.5
    hard_force_limit_N: float = 5.5
    min_contact_force_N: float = 0.2
    min_contact_compression_m: float = 0.0001
    max_compression_m: float = 0.00725
    precontact_gap_m: float = 0.04
    precontact_tolerance_m: float = 0.002
    approach_tangential_tolerance_m: float = 0.003
    contact_normal_tolerance_m: float = 0.002
    hold_tangential_tolerance_m: float = 0.002
    angle_tolerance_deg: float = 5.0
    approach_speed_limit_m_s: float = 0.02
    hold_speed_limit_m_s: float = 0.005
    settle_duration_s: float = 0.3
    hold_duration_s: float = 3.0
    retract_clearance_m: float = 0.04
    release_force_limit_N: float = 0.2
    release_confirm_duration_s: float = 0.15
    max_observation_age_s: float = 0.2
    max_update_gap_s: float = 0.2
    approach_timeout_s: float = 60.0
    contact_timeout_s: float = 20.0
    hold_timeout_s: float = 20.0
    retract_timeout_s: float = 20.0
    episode_timeout_s: float = 120.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not _finite_real(value) or value <= 0:
                raise ValueError(field.name + " must be finite and positive")
        if not (self.min_contact_force_N <= self.target_force_N - self.force_tolerance_N
                < self.target_force_N + self.force_tolerance_N < self.hard_force_limit_N):
            raise ValueError("force band must be above contact threshold and below hard force limit")
        if self.min_contact_compression_m >= self.max_compression_m:
            raise ValueError("contact compression must be below maximum compression")
        if self.precontact_tolerance_m >= self.precontact_gap_m:
            raise ValueError("precontact tolerance must keep approach outside the surface")
        if self.release_force_limit_N > self.min_contact_force_N:
            raise ValueError("release threshold must not exceed contact threshold")


@dataclass(frozen=True)
class ContactObservation:
    """Actor-observable estimates, all expressed in SI units except angle degrees.

    ``normal_gap_m`` is positive outside the estimated target surface; its frame
    and target must stay fixed through a retraction, even when target_valid drops.
    Hall force is the nonnegative calibrated estimate, not a PhysX contact label.
    TCP position is the finite FK/observed position used to derive these errors.
    ``observation_age_s`` is the oldest age of any constituent signal, not only
    Hall. The caller must not refresh an old sample's timestamp by reusing it.
    """

    hall_force_N: float
    compression_m: float
    normal_gap_m: float
    tangential_error_m: float
    angle_error_deg: float
    tcp_speed_m_s: float
    observation_age_s: float = 0.0
    target_valid: bool = True
    tcp_position_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PhaseState:
    phase: Phase
    hold_elapsed: float
    completed: bool
    success: bool
    failure_reason: Optional[str]
    phase_elapsed: float
    contact_confirmed: bool


class PhaseManager:
    """Bounded state machine; at most one phase transition per update.

    ``dt`` is elapsed wall/simulation time since the previous call. A finite,
    nonnegative interval is required (invalid dt raises ValueError before any
    mutation). A long but finite interval triggers the watchdog. Invalid, stale,
    missing, or lost observations request RETRACT and latch a failed outcome;
    they do not raise. DONE means both continuous hold and verified retraction.
    ABORT means failed termination, and may mean retraction could not be verified.

    Each current sample represents the preceding dt interval for confirmation
    timing. Phase entry never credits that same interval to the next phase.
    HOLD transients reset its timer and require a fresh settle period without
    changing phase, allowing the caller's force controller to recover smoothly.
    """

    def __init__(self, config: Optional[PhaseConfig] = None):
        self.config = config if config is not None else PhaseConfig()
        self._phase = Phase.APPROACH
        self._phase_elapsed = 0.0
        self._episode_elapsed = 0.0
        self._hold_elapsed = 0.0
        self._settle_elapsed = 0.0
        self._release_elapsed = 0.0
        self._hold_complete = False
        self._failure_reason = None
        self._contact_confirmed = False

    @property
    def state(self):
        return PhaseState(self._phase, self._hold_elapsed,
                          self._phase in (Phase.DONE, Phase.ABORT),
                          self._phase == Phase.DONE, self._failure_reason,
                          self._phase_elapsed, self._contact_confirmed)

    def _transition(self, phase):
        self._phase = phase
        self._phase_elapsed = 0.0

    @staticmethod
    def _valid_observation(observation):
        if not isinstance(observation, ContactObservation) or not isinstance(observation.target_valid, bool):
            return False
        for name in ("hall_force_N", "compression_m", "normal_gap_m", "tangential_error_m",
                     "angle_error_deg", "tcp_speed_m_s", "observation_age_s"):
            value = getattr(observation, name)
            if not _finite_real(value) or (name != "normal_gap_m" and value < 0):
                return False
        try:
            return len(observation.tcp_position_m) == 3 and all(
                _finite_real(value) for value in observation.tcp_position_m)
        except TypeError:
            return False

    def _fault(self, observation, dt, valid):
        config = self.config
        if not valid:
            return "invalid_observation"
        if dt > config.max_update_gap_s:
            return "watchdog_timeout"
        if observation.observation_age_s > config.max_observation_age_s:
            return "stale_observation"
        if observation.hall_force_N >= config.hard_force_limit_N:
            return "overforce"
        if observation.compression_m >= config.max_compression_m:
            return "overcompression"
        if not observation.target_valid:
            return "target_lost"
        return None

    def update(self, observation: ContactObservation, dt: float) -> PhaseState:
        if not _finite_real(dt) or dt < 0:
            raise ValueError("dt must be a finite nonnegative number")
        if self._phase in (Phase.DONE, Phase.ABORT):
            return self.state

        config = self.config
        self._phase_elapsed += dt
        self._episode_elapsed += dt
        valid = self._valid_observation(observation)
        fault = self._fault(observation, dt, valid)
        episode_fault = "episode_timeout" if _elapsed(self._episode_elapsed, config.episode_timeout_s) else None
        self._contact_confirmed = bool(valid and
            observation.hall_force_N >= config.min_contact_force_N and
            observation.compression_m >= config.min_contact_compression_m and
            abs(observation.normal_gap_m) <= config.contact_normal_tolerance_m)
        if self._failure_reason is None:
            self._failure_reason = fault or episode_fault

        if self._phase == Phase.RETRACT:
            # A disappeared target remains a failed outcome, but a retained
            # target frame can still confirm that outward recovery succeeded.
            # Episode expiry also latches failure without invalidating fresh
            # release observations or restarting the bounded retraction timer.
            release_valid = (valid and fault in (None, "target_lost") and
                             observation.normal_gap_m >= config.retract_clearance_m and
                             observation.hall_force_N < config.release_force_limit_N)
            self._release_elapsed = self._release_elapsed + dt if release_valid else 0.0
            if _elapsed(self._phase_elapsed, config.retract_timeout_s):
                if self._failure_reason:
                    self._failure_reason += "; retract_timeout"
                else:
                    self._failure_reason = "retract_timeout"
                self._transition(Phase.ABORT)
            elif _elapsed(self._release_elapsed, config.release_confirm_duration_s):
                self._transition(Phase.DONE if self._hold_complete and self._failure_reason is None else Phase.ABORT)
            return self.state

        fault = fault or episode_fault
        if fault is None and _elapsed(self._phase_elapsed, getattr(config, self._phase.value.lower() + "_timeout_s")):
            fault = self._phase.value.lower() + "_timeout"
        if fault:
            self._failure_reason = self._failure_reason or fault
            self._hold_elapsed = 0.0
            self._settle_elapsed = 0.0
            self._transition(Phase.RETRACT)
            return self.state

        if self._phase == Phase.APPROACH:
            gap_error = abs(observation.normal_gap_m - config.precontact_gap_m)
            at_precontact = (gap_error <= config.precontact_tolerance_m or
                             math.isclose(gap_error, config.precontact_tolerance_m, rel_tol=1e-12))
            aligned = (at_precontact and
                       observation.tangential_error_m <= config.approach_tangential_tolerance_m and
                       observation.angle_error_deg <= config.angle_tolerance_deg and
                       observation.tcp_speed_m_s < config.approach_speed_limit_m_s)
            if aligned:
                self._transition(Phase.CONTACT)
            return self.state

        hold_valid = (self._contact_confirmed and
                      abs(observation.hall_force_N - config.target_force_N) <= config.force_tolerance_N and
                      observation.tangential_error_m <= config.hold_tangential_tolerance_m and
                      observation.angle_error_deg <= config.angle_tolerance_deg and
                      observation.tcp_speed_m_s < config.hold_speed_limit_m_s)
        if not hold_valid:
            self._settle_elapsed = 0.0
            self._hold_elapsed = 0.0
        elif not _elapsed(self._settle_elapsed, config.settle_duration_s):
            self._settle_elapsed += dt
            if self._phase == Phase.CONTACT and _elapsed(self._settle_elapsed, config.settle_duration_s):
                self._transition(Phase.HOLD)
        elif self._phase == Phase.HOLD:
            self._hold_elapsed += dt
            if _elapsed(self._hold_elapsed, config.hold_duration_s):
                self._hold_elapsed = config.hold_duration_s
                self._hold_complete = True
                self._transition(Phase.RETRACT)
        return self.state
