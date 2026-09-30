"""Minimal controller alternatives over the immutable low-elbow controller."""
import json
import time

import numpy as np

from sim.auscultation_expert_layout_v1.controller import (
    CONTROL_STEPS, FRONT_GROUPS, FORBIDDEN_GROUPS, EpisodeRejected, digest,
)
from sim.auscultation_expert_layout_v1.low_elbow_controller import LowElbowController
from .candidate_planner import POLICIES, checked_edge, checked_shortcut, joint_path, path_commands, path_metrics


class CandidateController(LowElbowController):
    def __init__(self, *args, policy="staged_continuous", **kwargs):
        if policy not in POLICIES:
            raise ValueError("unknown candidate policy")
        self.policy = policy
        self.free_motion_timing = []
        super().__init__(*args, **kwargs)

    def run(self, points):
        outcome = super().run(points)
        outcome.update(candidate_policy=self.policy,
                       candidate_method="sim.expert_quality_20260924.candidate_controller.CandidateController",
                       candidate_geometry=("original staging routes" if self.policy == "staged_continuous" else
                           "checked direct released-prepose shortcuts; original staging fallback"),
                       candidate_timing="quintic polyline arclength retimed until padded 240 Hz forward differences satisfy velocity/acceleration/jerk limits; no C2 claim; bridge corners still stop",
                       free_motion_command_bounds={"joint_velocity_rad_s": self.joint_speed_limit,
                           "joint_acceleration_rad_s2": 2., "joint_jerk_rad_s3": 20.,
                           "scope": "sampled free-motion commands only; unchanged contact commands and measured dynamics require independent audit"},
                       free_motion_segments=self.free_motion_timing,
                       candidate_scope="same robot, contact poses, force controller and posture envelope; simulation comparison",
                       route_selection=("retained from-below staging and surface-normal contact" if self.policy == "staged_continuous" else
                           "may bypass lower staging after checked physical release; surface-normal final contact retained"))
        return outcome

    @staticmethod
    def _canonical(row):
        canonical = joint_path(row.get("transfer_q"))
        if not row.get("success") or not np.allclose(canonical[0], 0., atol=1e-9, rtol=0):
            raise EpisodeRejected("missing checked canonical home route")
        stage = np.asarray(row.get("lower_staging_q"), float)
        if stage.shape != (7,) or not np.isfinite(stage).all():
            raise EpisodeRejected("next point has no checked lower staging pose")
        matches = np.flatnonzero(np.all(np.isclose(canonical, stage, atol=1e-9, rtol=0), axis=1))
        if len(matches) != 1 or not 0 < matches[0] < len(canonical)-1:
            raise EpisodeRejected("lower staging pose has no unique canonical approach tail")
        return canonical, int(matches[0])

    def _continuous_path(self, route, subphase):
        route = joint_path(route)
        if not np.allclose(route[0], self.q, atol=1e-9, rtol=0):
            raise EpisodeRejected("continuous route disconnected from retained command")
        commands, report = path_commands(route, speed_limit=self.joint_speed_limit,
                                 minimum_seconds=1./self.speed_scale, return_report=True)
        report.update(subphase=subphase, target=self.target.name,
                      start_physics_row=len(self.env.rows), motion_complete=False)
        self.free_motion_timing.append(report)
        for offset in range(0, len(commands), CONTROL_STEPS):
            self._block(commands[offset:offset+CONTROL_STEPS], subphase)
        report.update(end_physics_row_exclusive=len(self.env.rows), motion_complete=True)

    def _joint_segment(self, destination, subphase):
        # Include short connectors and structured bridge legs in the same
        # discrete dynamic validation as dense IK tails.
        return self._continuous_path(np.vstack([self.q, destination]), subphase)

    def _execute_staged(self, route, stage_index, *, initial=False):
        # Preserve stops at structured bridge anchors and at the lower stage.
        for destination in route[1:stage_index+1]:
            self._joint_segment(destination, "transfer" if initial else "inter_point_transfer")
        self._continuous_path(route[stage_index:], "lower_approach")

    def _move_to_prepose(self, row):
        canonical, canonical_stage_index = self._canonical(row)
        self.canonical_transfer = canonical.copy()
        self.canonical_stage_index = canonical_stage_index
        if self.sequence_index == 0:
            if np.max(np.abs(self.env.state()["q"])) > .03 or np.max(np.abs(self.q)) > .03:
                raise EpisodeRejected("point must start at dynamically reached zero home")
            deadline = time.monotonic()+30.
            for start, end in zip(canonical, canonical[1:]):
                if not checked_edge(start, end, self.checker, deadline)["clear"]:
                    raise EpisodeRejected("canonical route lacks sampled clearance or posture validity")
            if np.max(np.abs(self.q)) > 1e-9:
                if not checked_edge(self.q, np.zeros(7), self.checker, deadline)["clear"]:
                    raise EpisodeRejected("initial home bridge has no sampled clearance")
                self._joint_segment(np.zeros(7), "initial_home")
            self._hold(1., "settle")
            self._execute_staged(canonical, canonical_stage_index, initial=True)
            self._hold(1., "align")
            return canonical

        state = self.env.state()
        release = self.released_prepose
        total = sum(self.env.last_contacts.get(name, 0.) for name in self.total_contact_groups)
        if (release is None or release["physics_row_end_exclusive"] != len(self.env.rows)
                or abs(float(state["time_s"])-release["time_s"]) > 1e-9
                or not np.isfinite([state["hall_N"], total]).all() or state["hall_N"] >= .2 or total >= .2):
            raise EpisodeRejected("inter-point route requires the previous verified released prepose")
        observed = np.asarray(state["q"], float).copy()
        stage = canonical[canonical_stage_index]
        record = {"from_target": release["target"], "to_target": self.target.name,
                  "start_physics_row": len(self.env.rows), "planning_simulation_time_s": float(state["time_s"]),
                  "measured_start_q": observed.tolist(), "retained_command_q": self.q.tolist(),
                  "lower_staging_q": stage.tolist(), "canonical_staging_index": canonical_stage_index,
                  "planning_accepted": False, "motion_complete": False, "edge_checks": [],
                  "free_clearance_m": .003, "joint_edge_sample_step_rad": .015,
                  "candidate_policy": self.policy, "planner": "original bounded bridge with optional fully checked direct shortcut"}
        self.inter_point_routes.append(record)
        started = time.monotonic()
        deadline = started+30.
        try:
            bridge, search = self._plan_inter_point_bridge(observed, stage, deadline)
            record["search"] = search
            if bridge is None:
                raise EpisodeRejected("inter-point bridge unavailable: "+str(search.get("reason")))
            bridge = joint_path(bridge)
            if (not np.allclose(bridge[0], observed, atol=1e-9, rtol=0)
                    or not np.allclose(bridge[-1], stage, atol=1e-9, rtol=0)):
                raise EpisodeRejected("inter-point planner returned invalid route endpoints")
            route = np.vstack([bridge, canonical[canonical_stage_index+1:]])
            stage_index = len(bridge)-1
            record["baseline_metrics"] = path_metrics(route)
            if self.policy == "checked_shortcut":
                route, record["shortcut"] = checked_shortcut(route, self.checker, deadline)
                if record["shortcut"]["shortcut_accepted"]:
                    stage_index = None
            # Keep both measured planning origin and retained command in the
            # verified path. The connector never disappears in shortcutting.
            if np.max(np.abs(self.q-observed)) > 1e-9:
                route = np.vstack([self.q, route])
                if stage_index is not None:
                    stage_index += 1
            for start, end in zip(route, route[1:]):
                result = checked_edge(start, end, self.checker, deadline)
                record["edge_checks"].append(result)
                if not result["clear"]:
                    raise EpisodeRejected("inter-point route lacks sampled clearance or posture validity")
            if float(self.env.state()["time_s"]) != record["planning_simulation_time_s"] or len(self.env.rows) != record["start_physics_row"]:
                raise EpisodeRejected("simulation advanced during inter-point planning")
            record.update(joint_waypoints_q=route.tolist(), lower_staging_waypoint_index=stage_index,
                          lower_staging_retained=stage_index is not None,
                          selected_metrics=path_metrics(route), planning_accepted=True,
                          waypoints_sha256=digest(json.dumps(route.tolist(), separators=(",", ":"), allow_nan=False).encode()),
                          waypoint_digest_format="UTF-8 JSON joint_waypoints_q with separators comma and colon",
                          canonical_return_route_sha256=digest(json.dumps(canonical.tolist(), separators=(",", ":"), allow_nan=False).encode()))
        except Exception as error:
            record["failure_reason"] = str(error) or type(error).__name__
            raise
        finally:
            record["planning_wall_time_s"] = time.monotonic()-started
        try:
            if stage_index is None:
                for destination in route[1:]:
                    self._joint_segment(destination, "inter_point_transfer")
            else:
                self._execute_staged(route, stage_index)
            self._hold(1., "align")
            record["motion_complete"] = True
        finally:
            record["end_physics_row_exclusive"] = len(self.env.rows)
        return route

    def _return_home(self, transfer):
        if self.sequence_index+1 < self.sequence_length:
            return self._withdraw_to_prepose(transfer)
        self._withdraw_to_prepose(transfer)
        check = checked_edge(self.q, self.canonical_transfer[-1], self.checker, time.monotonic()+30.)
        self.bridge_checks.append(check)
        if not check["clear"]:
            raise EpisodeRejected("return bridge has no sampled clearance")
        self._joint_segment(self.canonical_transfer[-1], "return_bridge")
        self._continuous_path(self.canonical_transfer[self.canonical_stage_index:][::-1], "return_home")
        for destination in self.canonical_transfer[:self.canonical_stage_index][::-1]:
            self._joint_segment(destination, "return_home")
        self._hold(.5, "home_settle")
        if np.max(np.abs(self.env.state()["q"])) > .03:
            raise EpisodeRejected("dynamic home tracking error")
