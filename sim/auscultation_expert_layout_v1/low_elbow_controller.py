"""Expert motion with an explicit posture envelope on commands and observations."""
import json
import time
import numpy as np

from .controller import ExpertController, EpisodeRejected, CONTROL_STEPS, FRONT_GROUPS, FORBIDDEN_GROUPS, digest
from .posture import PostureLimits, PostureChecker


class LowElbowController(ExpertController):
    def run(self, points):
        if not points:
            raise ValueError("low-elbow sequence requires points")
        configurations = [row.get("posture_limits") for _, row in points]
        if any(not isinstance(value, dict) for value in configurations):
            raise ValueError("every low-elbow plan requires posture limits")
        limits = [PostureLimits.from_dict(value, self.kin) for value in configurations]
        if len({json.dumps(value.to_dict(), sort_keys=True) for value in limits}) != 1:
            raise ValueError("sequence posture profiles disagree")
        self.posture_limits = limits[0]
        shoulder_z = (self.env.base @ self.kin.link_transforms(np.zeros(7))["openarm_left_link1"])[2, 3]
        if self.posture_limits.max_elbow_z_m > shoulder_z + self.posture_limits.max_elbow_above_shoulder_m + 1e-9:
            raise ValueError("low-elbow ceiling must not exceed shoulder plus declared allowance")
        self.checker = PostureChecker(self.checker, self.env.base, self.posture_limits)
        self.inter_point_routes = []
        self.inter_point_rng = np.random.default_rng(65)
        self.released_prepose = None
        result = super().run(points)
        result.update(controller_type="algorithmic_expert_low_elbow",
                      posture_limits=self.posture_limits.to_dict(),
                      posture_scope="command and observed joint envelope, declared elbow height cap and contact elbow below wrist; simulation only",
                      posture_command_checks="every commanded physics sample",
                      posture_observation_checks="every observed physics sample",
                      posture_ik="single closest-seed bounded exact-pose solve; no unconstrained fallback",
                      inter_point_routes=self.inter_point_routes,
                      inter_point_planning_wall_budget_s=30., inter_point_planner_seed=65,
                      home_policy="initial home and final canonical return only; released preposes between points",
                      route_selection="from below with constrained joint posture; normal final contact")
        return result

    def _plan_inter_point_bridge(self, start, goal, deadline):
        from .plan_low_elbow import checked_transfer
        return checked_transfer(self.checker, self.posture_limits, start, goal, self.kin,
                                self.env.base, self.inter_point_rng, deadline)

    def _check_inter_point_edge(self, start, end, deadline):
        """Verify free-space samples with a cooperative wall deadline."""
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / .015)))
        minimum = float("inf")
        for fraction in np.linspace(0., 1., count + 1):
            if time.monotonic() >= deadline:
                raise EpisodeRejected("inter-point planning deadline exceeded")
            result = self.checker.check(start + fraction * (end - start),
                                        clearance=.003, allow_patient_contact=False, compression=0.)
            if time.monotonic() >= deadline:
                raise EpisodeRejected("inter-point planning deadline exceeded")
            distance = float(result.get("minimum_distance_m", float("nan")))
            if result.get("clear") is not True or not np.isfinite(distance) or distance < .003 - 1e-9:
                return {**result, "clear": False, "fraction": float(fraction), "samples": count + 1}
            minimum = min(minimum, distance)
        return {"clear": True, "minimum_distance_m": minimum, "samples": count + 1}

    def _move_to_prepose(self, row):
        canonical = np.asarray(row["transfer_q"], float)
        if (not row.get("success") or canonical.ndim != 2 or canonical.shape[1] != 7
                or len(canonical) < 2 or not np.isfinite(canonical).all()
                or not np.allclose(canonical[0], 0., atol=1e-9, rtol=0)):
            raise EpisodeRejected("missing checked canonical home route")
        self.canonical_transfer = canonical.copy()
        if self.sequence_index == 0:
            return super()._move_to_prepose(row)
        state = self.env.state()
        release = self.released_prepose
        total = sum(self.env.last_contacts.get(name, 0.) for name in FRONT_GROUPS + FORBIDDEN_GROUPS)
        if (release is None or release["physics_row_end_exclusive"] != len(self.env.rows)
                or abs(float(state["time_s"]) - release["time_s"]) > 1e-9
                or not np.isfinite([state["hall_N"], total]).all()
                or state["hall_N"] >= .2 or total >= .2):
            raise EpisodeRejected("inter-point route requires the previous verified released prepose")
        stage = np.asarray(row.get("lower_staging_q"), float)
        if stage.shape != (7,) or not np.isfinite(stage).all():
            raise EpisodeRejected("next point has no checked lower staging pose")
        stage_indices = np.flatnonzero(np.all(np.isclose(canonical, stage, atol=1e-9, rtol=0), axis=1))
        if len(stage_indices) != 1 or not 0 < stage_indices[0] < len(canonical) - 1:
            raise EpisodeRejected("lower staging pose has no unique canonical approach tail")
        observed = np.asarray(state["q"], float).copy()
        record = {"from_target": release["target"], "to_target": self.target.name,
                  "start_physics_row": len(self.env.rows), "planning_simulation_time_s": float(state["time_s"]),
                  "measured_start_q": observed.tolist(), "retained_command_q": self.q.tolist(),
                  "lower_staging_q": stage.tolist(), "canonical_staging_index": int(stage_indices[0]),
                  "planning_accepted": False, "motion_complete": False, "edge_checks": [],
                  "free_clearance_m": .003, "joint_edge_sample_step_rad": .015,
                  "planner": "posture-constrained checked structured bridges then bounded RRT; no home fallback"}
        self.inter_point_routes.append(record)
        started = time.monotonic()
        deadline = started + 30.
        try:
            bridge, search = self._plan_inter_point_bridge(observed, stage, deadline)
            record["search"] = search
            if time.monotonic() >= deadline:
                raise EpisodeRejected("inter-point planning deadline exceeded")
            if bridge is None:
                raise EpisodeRejected("inter-point bridge unavailable: " + str(search.get("reason")))
            bridge = np.asarray(bridge, float)
            if (bridge.ndim != 2 or bridge.shape[1] != 7 or len(bridge) < 2
                    or not np.isfinite(bridge).all()
                    or not np.allclose(bridge[0], observed, atol=1e-9, rtol=0)
                    or not np.allclose(bridge[-1], stage, atol=1e-9, rtol=0)):
                raise EpisodeRejected("inter-point planner returned invalid route endpoints")
            # Preserve command continuity while planning from actual measured
            # joints. The small connector is checked just like every free edge.
            if np.max(np.abs(self.q - observed)) > 1e-9:
                bridge = np.vstack([self.q, bridge])
            staging_index = len(bridge) - 1
            route = np.vstack([bridge, canonical[int(stage_indices[0]) + 1:]])
            record.update(joint_waypoints_q=route.tolist(), lower_staging_waypoint_index=staging_index,
                          waypoints_sha256=digest(json.dumps(route.tolist(), separators=(",", ":"), allow_nan=False).encode()),
                          waypoint_digest_format="UTF-8 JSON joint_waypoints_q with separators comma and colon",
                          canonical_return_route_sha256=digest(json.dumps(canonical.tolist(), separators=(",", ":"), allow_nan=False).encode()))
            for start, end in zip(route[:-1], route[1:]):
                check = self._check_inter_point_edge(start, end, deadline)
                record["edge_checks"].append(check)
                if not check["clear"]:
                    raise EpisodeRejected("inter-point route lacks sampled free clearance or posture validity")
            if float(self.env.state()["time_s"]) != record["planning_simulation_time_s"] or len(self.env.rows) != record["start_physics_row"]:
                raise EpisodeRejected("simulation advanced during inter-point planning")
            record["planning_accepted"] = True
        except Exception as error:
            record["failure_reason"] = str(error) or type(error).__name__
            raise
        finally:
            record["planning_wall_time_s"] = time.monotonic() - started
        try:
            for index, destination in enumerate(route[1:], start=1):
                subphase = "inter_point_transfer" if index <= staging_index else "lower_approach"
                self._joint_segment(destination, subphase)
            self._hold(1., "align")
            record["motion_complete"] = True
        finally:
            record["end_physics_row_exclusive"] = len(self.env.rows)
        return route

    def _withdraw_to_prepose(self, transfer):
        super()._withdraw_to_prepose(transfer)
        state = self.env.state()
        total = sum(self.env.last_contacts.get(name, 0.) for name in FRONT_GROUPS + FORBIDDEN_GROUPS)
        if not np.isfinite(total) or total >= .2:
            raise EpisodeRejected("physical contact release at prepose not verified")
        self.released_prepose = {"target": self.target.name, "time_s": float(state["time_s"]),
                                "physics_row_end_exclusive": len(self.env.rows),
                                "measured_q": np.asarray(state["q"]).tolist(), "command_q": self.q.tolist()}

    def _return_home(self, transfer):
        if self.sequence_index + 1 < self.sequence_length:
            self._withdraw_to_prepose(transfer)
        else:
            # The final home route belongs to the final target; reversing the
            # inter-point bridge would unnecessarily revisit earlier points.
            super()._return_home(self.canonical_transfer)

    def _contact_posture_required(self):
        return self.manager.state.phase.value in ("CONTACT", "HOLD", "RETRACT")

    def _block(self, commands, subphase, active=False, recovery=False, gap=None):
        if getattr(self, "posture_failure", None):
            raise EpisodeRejected("observed posture failure latched: " + self.posture_failure)
        commands = np.asarray(commands, float)
        if commands.ndim != 2 or commands.shape[1] != 7 or not np.isfinite(commands).all():
            raise EpisodeRejected("invalid low-elbow command block")
        for command in commands:
            result = self.posture_limits.evaluate(self.kin, self.env.base, command,
                                                 contact=self._contact_posture_required())
            if not result["clear"]:
                raise EpisodeRejected("command posture rejected: " + ", ".join(result["posture_violations"]))
        return super()._block(commands, subphase, active=active, recovery=recovery, gap=gap)

    def _tick(self, command, subphase, active=False, recovery=False):
        # A phase transition may occur inside a block, so check the command
        # against the current contact posture before each individual step.
        result = self.posture_limits.evaluate(self.kin, self.env.base, command,
                                             contact=self._contact_posture_required())
        if not result["clear"]:
            raise EpisodeRejected("command posture rejected: " + ", ".join(result["posture_violations"]))
        state = super()._tick(command, subphase, active=active, recovery=recovery)
        # The base controller permits force-limited withdrawal after some
        # faults. A posture violation has no such exception, including in an
        # already-running recovery block: stop after this observed sample.
        if getattr(self, "posture_failure", None):
            raise EpisodeRejected("observed posture rejected: " + self.posture_failure)
        return state

    def _violations(self, state):
        violations, total = super()._violations(state)
        result = self.posture_limits.evaluate(self.kin, self.env.base, state["q"], observed=True,
                                             contact=self._contact_posture_required())
        violations += result["posture_violations"]
        self.env.rows[-1].update(elbow_z_m=result["elbow_z_m"],
                                posture_joint_margin_rad=result["posture_joint_margin_rad"],
                                wrist_z_m=result["wrist_z_m"],
                                elbow_below_wrist_m=result["elbow_below_wrist_m"],
                                posture_violations=result["posture_violations"])
        if result["posture_violations"]:
            self.posture_failure = ", ".join(result["posture_violations"])
        return violations, total

    def _pose_block(self, position_base, rotation_base, subphase, active=False, recovery=False, gap=None):
        from .pose_ik import solve_posture_ik
        q, good, _, _ = solve_posture_ik(self.kin, self.env.base, self.posture_limits,
            position_base, rotation_base, self.q, maxiter=80,
            planning_margin_m=.002, joint_margin_rad=0., contact=self._contact_posture_required())
        if not good:
            raise EpisodeRejected("bounded posture IK failed in " + subphase)
        commands = self.q + np.arange(1, CONTROL_STEPS + 1)[:, None] / CONTROL_STEPS * (q - self.q)
        return self._block(commands, subphase, active=active, recovery=recovery, gap=gap)
