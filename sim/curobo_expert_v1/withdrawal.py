"""Smooth normal unloading without changing the existing contact-force law.

The former controller switched directly from HOLD feedback to 10 mm/s at the
start of RETRACT. This mixin replaces that motion, including its command-block
boundaries; it does not filter measurements or weaken any acceptance threshold.
These command checks are not a substitute for measured PhysX/robot validation.
"""
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from sim.auscultation_expert_layout_v1.controller import (
    DT, CONTROL_DT, CONTROL_STEPS, FRONT_GROUPS, FORBIDDEN_GROUPS, EpisodeRejected,
)
from sim.auscultation_expert_layout_v1.phase_manager import Phase


def normal_path_commands(path, distance_m, normal_speed_m_s, previous_commands,
                         *, joint_speed_limit=.25, maximum_seconds=18.):
    """Retain incoming command derivatives and stop at the exact path endpoint.

    The spatial cubic interpolant is composed with a quintic time law. A compact
    quintic correction matches velocity and acceleration from the three actual
    preceding commands; it expires with zero position/velocity/acceleration.
    Every returned sample and both boundaries are checked with unfiltered
    forward differences. FK geometry is checked separately by the caller.
    """
    path = np.asarray(path, float)
    history = np.asarray(previous_commands, float)
    if (path.ndim != 2 or path.shape[1:] != (7,) or len(path) < 2
            or history.shape != (3, 7) or not np.isfinite(path).all()
            or not np.isfinite(history).all()
            or not np.isfinite([distance_m, normal_speed_m_s, joint_speed_limit, maximum_seconds]).all()
            or min(distance_m, normal_speed_m_s, joint_speed_limit, maximum_seconds) <= 0):
        raise ValueError("invalid normal withdrawal path or timing")
    if not np.array_equal(path[0], history[-1]):
        raise ValueError("normal path is disconnected from retained command")
    velocity = (history[-1]-history[-2])/DT
    acceleration = (history[-1]-2*history[-2]+history[-3])/DT**2
    interpolant = CubicSpline(np.linspace(0., 1., len(path)), path, axis=0)
    seconds = max(1., 1.875*distance_m/normal_speed_m_s)
    peaks = None
    for attempt in range(1, 25):
        steps = int(np.ceil(seconds/CONTROL_DT))*CONTROL_STEPS
        duration = steps*DT
        if duration > maximum_seconds+1e-10:
            break
        t = np.arange(1, steps+1)*DT
        u = t/duration
        fraction = 10*u**3-15*u**4+6*u**5
        commands = interpolant(fraction)
        correction_duration = min(1., duration/2.)
        v = np.minimum(t/correction_duration, 1.)
        velocity_basis = v-6*v**3+8*v**4-3*v**5
        acceleration_basis = .5*v**2-1.5*v**3+1.5*v**4-.5*v**5
        commands += (correction_duration*velocity_basis[:, None]*velocity
                     + correction_duration**2*acceleration_basis[:, None]*acceleration)
        commands[-1] = path[-1]
        padded = np.vstack([history, commands, np.repeat(path[-1:], 3, axis=0)])
        peaks = {name: float(np.max(np.abs(np.diff(padded, n=order, axis=0)))/DT**order)
                 for name, order in (("velocity_rad_s", 1), ("acceleration_rad_s2", 2), ("jerk_rad_s3", 3))}
        if (peaks["velocity_rad_s"] <= joint_speed_limit+1e-9
                and peaks["acceleration_rad_s2"] <= .5+1e-9
                and peaks["jerk_rad_s3"] <= 8.+1e-9):
            return commands, fraction, {
                "duration_s": duration, "attempts": attempt, "peaks": peaks,
                "boundary_history_samples": 3,
                "incoming_command_velocity_rad_s": velocity.tolist(),
                "incoming_command_acceleration_rad_s2": acceleration.tolist(),
                "normal_speed_limit_m_s": float(normal_speed_m_s),
                "joint_command_limits": {"velocity_rad_s": float(joint_speed_limit),
                                         "acceleration_rad_s2": .5, "jerk_rad_s3": 8.},
                "method": "spatial cubic interpolation, quintic time law, incoming derivative correction; raw finite-difference checks including endpoints",
            }
        seconds = duration*1.15
    raise EpisodeRejected("normal withdrawal timing budget exhausted; peaks="+str(peaks))


class NormalWithdrawalMixin:
    """Place before CandidateController in the MRO; no alternate force policy."""

    def run(self, points):
        self.normal_withdrawal_segments = []
        self._normal_release_target = None
        result = super().run(points)
        result.update(normal_withdrawal_segments=self.normal_withdrawal_segments,
                      normal_withdrawal_method="slow 8 mm normal unload, verified Hall release, smooth retreat to the same checked prepose",
                      normal_withdrawal_scope="command continuity and simulation release; measured dynamics audited independently; CONTACT/HOLD gains and timers unchanged")
        return result

    def _withdrawal_history(self):
        if len(self.env.rows) < 3:
            raise EpisodeRejected("normal withdrawal requires three preceding command samples")
        history = np.asarray([r["q_command"] for r in self.env.rows[-3:]], float)
        if history.shape != (3, 7) or not np.array_equal(history[-1], self.q):
            raise EpisodeRejected("normal withdrawal retained command history mismatch")
        return history

    def _normal_segment(self, goal_gap, speed, kind, *, active=True):
        from sim.auscultation_expert_layout_v1.pose_ik import solve_posture_ik

        history = self._withdrawal_history()
        start_q = self.q.copy()
        start_position = self.kin_position_from_command().copy()
        origin_gap = float((start_position-self.target.position_base)@self.normal_base)
        distance = float(goal_gap-origin_gap)
        if distance <= 1e-9:
            raise EpisodeRejected("normal withdrawal destination must be outward")
        end_position = self.target.position_base+self.normal_base*goal_gap
        tangent = start_position-(self.target.position_base+self.normal_base*origin_gap)
        if np.linalg.norm(tangent) > .0001:
            raise EpisodeRejected("normal withdrawal starts off the retained normal")
        count = max(17, int(np.ceil(distance/.0005))+1)
        knots = [start_q]
        deadline = time.monotonic()+30.
        for fraction in np.linspace(0., 1., count)[1:]:
            position = start_position+fraction*(end_position-start_position)
            q, good, _, _ = solve_posture_ik(self.kin, self.env.base, self.posture_limits,
                position, self.rotation_base, knots[-1], maxiter=80, deadline=deadline,
                planning_margin_m=.002, joint_margin_rad=0., contact=True)
            if not good:
                raise EpisodeRejected("normal withdrawal bounded posture IK failed")
            knots.append(q)
        remaining = (self.manager.config.retract_timeout_s-self.manager.state.phase_elapsed-.3
                     if self.manager.state.phase == Phase.RETRACT else 18.)
        commands, fractions, report = normal_path_commands(knots, distance, speed,
            history, joint_speed_limit=self.joint_speed_limit, maximum_seconds=remaining)
        max_position_error = max_angle_error = 0.
        normal_positions = []
        for q, fraction in zip(commands, fractions):
            pose = self.kin.tcp(q)
            expected = start_position+fraction*(end_position-start_position)
            max_position_error = max(max_position_error, float(np.linalg.norm(pose[:3, 3]-expected)))
            max_angle_error = max(max_angle_error,
                float(np.linalg.norm(Rotation.from_matrix(pose[:3, :3]@self.rotation_base.T).as_rotvec())))
            normal_positions.append(float((pose[:3, 3]-self.target.position_base)@self.normal_base))
        # Cubic joint interpolation is never assumed to preserve Cartesian
        # geometry. Reject an inaccurate spline before executing any of it.
        if max_position_error > .0001 or max_angle_error > .001:
            raise EpisodeRejected("normal withdrawal interpolation geometry error")
        inward = max(0., float(-np.min(np.diff(np.r_[origin_gap, normal_positions]))))
        if inward > 1e-6:
            raise EpisodeRejected("normal withdrawal profile has an inward command step")
        report.update(kind=kind, target=self.target.name, start_gap_m=origin_gap, end_gap_m=float(goal_gap),
                      start_physics_row=len(self.env.rows), motion_complete=False,
                      start_q_command=start_q.tolist(), end_q_command=commands[-1].tolist(),
                      maximum_cartesian_interpolation_error_m=max_position_error,
                      maximum_rotation_interpolation_error_rad=max_angle_error,
                      maximum_inward_command_step_m=inward)
        self.normal_withdrawal_segments.append(report)
        try:
            for offset in range(0, len(commands), CONTROL_STEPS):
                block = commands[offset:offset+CONTROL_STEPS]
                gap = normal_positions[offset+len(block)-1]
                self._block(block, "normal_"+kind, active=active, gap=gap)
                if self.manager.state.phase == Phase.ABORT:
                    raise EpisodeRejected("normal withdrawal state machine aborted: "+str(self.manager.state.failure_reason))
            report["motion_complete"] = True
        finally:
            report["end_physics_row_exclusive"] = len(self.env.rows)

    def _smooth_normal_release(self):
        gap = float((self.kin_position_from_command()-self.target.position_base)@self.normal_base)
        unload_goal = min(self.pre_gap, gap+.008)
        self._normal_segment(unload_goal, .002*self.speed_scale, "unload")
        # Hall is deployable controller input. PhysX forces remain an independent
        # release gate below and in the unchanged per-physics-sample watchdog.
        state = self.env.state()
        if not np.isfinite(state["hall_N"]) or state["hall_N"] >= self.config.release_force_limit_N:
            raise EpisodeRejected("normal unload did not establish Hall release")
        physical = sum(self.env.last_contacts.get(name, 0.) for name in self.total_contact_groups)
        if not np.isfinite(physical) or physical >= .2:
            raise EpisodeRejected("normal unload physical contact release not verified")
        if unload_goal < self.pre_gap-1e-9:
            self._normal_segment(self.pre_gap, .006*self.speed_scale, "released_retreat")
        for _ in range(int(np.ceil(.2/CONTROL_DT))):
            self._block(np.repeat(self.q[None, :], CONTROL_STEPS, axis=0),
                        "normal_release_confirm", active=True, gap=self.pre_gap)
        if not self.manager.state.success:
            raise EpisodeRejected("normal withdrawal release state machine failed: "+str(self.manager.state.failure_reason))
        self._normal_release_target = self.target.name

    def _contact_cycle(self):
        self._normal_release_target = None
        gap = self.pre_gap
        for _ in range(int(np.ceil(80/self.speed_scale/CONTROL_DT))):
            state = self.env.state()
            phase = self.manager.state.phase
            if phase == Phase.RETRACT:
                self._smooth_normal_release()
                break
            # Identical CONTACT/HOLD controller, including feedback gain, clamps,
            # displacement budget and active per-step phase-manager updates.
            if state["hall_N"] > .3 or state["compression_m"] > .00015:
                speed = float(np.clip((state["hall_N"]-self.config.target_force_N)*.001, -.002, .002))
            else:
                speed = (-.012 if gap > .020 else -.002)*self.speed_scale
            gap += speed*CONTROL_DT
            if gap < -.01:
                raise EpisodeRejected("contact search exceeded displacement budget")
            self._pose_block(self.target.position_base+self.normal_base*gap, self.rotation_base,
                             "contact_cycle", active=True, gap=gap)
            if self.manager.state.completed:
                break
        if not self.manager.state.success:
            raise EpisodeRejected("state machine failed: "+str(self.manager.state.failure_reason))
        if self.best_audit_hold < self.config.hold_duration_s-2*DT:
            raise EpisodeRejected("observable hold lacked independent front contact")

    def _withdraw_to_prepose(self, transfer):
        # RETRACT now already ends at this exact checked prepose with stationary
        # commands. Never restart a 10 mm/s block or silently snap to another IK.
        self.released_prepose = None
        if self._normal_release_target != self.target.name:
            raise EpisodeRejected("smooth normal withdrawal has not completed for this target")
        self._hold(.2, "withdraw_settle")
        state = self.env.state()
        gap = self.metrics(state)["gap_m"]
        total = sum(self.env.last_contacts.get(name, 0.) for name in self.total_contact_groups)
        if (not np.isfinite([gap, state["hall_N"], total]).all()
                or gap < self.pre_gap-.002 or state["hall_N"] >= .2):
            raise EpisodeRejected("prepose withdrawal not verified")
        if total >= .2:
            raise EpisodeRejected("physical contact release at prepose not verified")
        compression = float(state["compression_m"])
        if not np.isfinite(compression) or compression > .00015:
            raise EpisodeRejected("prepose release compression exceeds free-motion threshold")
        self.released_prepose = {"target": self.target.name, "time_s": float(state["time_s"]),
                                "physics_row_end_exclusive": len(self.env.rows),
                                "compression_m": compression,
                                "measured_q": np.asarray(state["q"]).tolist(), "command_q": self.q.tolist()}
