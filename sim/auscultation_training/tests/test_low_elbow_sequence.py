"""Continuous low-elbow sequencing without home cycling or simulated resets."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim.auscultation_training import low_elbow_controller, pose_ik
from sim.auscultation_training.controller import EpisodeRejected
from sim.auscultation_training.low_elbow_controller import LowElbowController
from sim.auscultation_training.posture import PostureChecker
from sim.auscultation_training.tests.test_controller import FakeEnv, target
from sim.auscultation_training.tests.test_low_elbow_runtime import (
    PostureKin, PostureCollisionChecker, controller, profile,
)


def plan(name, branch):
    stage = np.array([0., 0., .18, 0., branch, 0., 0.])
    pre = stage.copy()
    pre[2] = .24
    return {"name": name, "success": True, "command_normal_world": [0., 0., 1.],
            "command_rotation_world": np.diag([1., -1., -1.]).tolist(),
            "posture_limits": profile(), "lower_staging_q": stage.tolist(),
            "transfer_q": [np.zeros(7).tolist(), stage.tolist(), pre.tolist()]}


def fake_constrained_ik(kin, base, limits, position, rotation, seed, **kwargs):
    q = np.asarray(seed).copy()
    q[:3] = [position[0], position[1], .3 - position[2]]
    return q, limits.evaluate(kin, base, q, contact=kwargs.get("contact", False))["clear"], 0., 0.


def test_continuous_points_keep_released_prepose_and_only_final_canonical_home(monkeypatch):
    class TrackingOffsetEnv(FakeEnv):
        def state(self):
            state = super().state()
            state["q"][4] += .0001
            return state

    env, kin = TrackingOffsetEnv(), PostureKin()
    value = LowElbowController(env, kin, PostureCollisionChecker(kin))
    planner_calls = []

    def planner(start, goal, deadline):
        planner_calls.append((start.copy(), goal.copy(), env.t, len(env.rows)))
        return [start.copy(), (start + goal) / 2, goal.copy()], {"reason": "fixture checked bridge"}

    monkeypatch.setattr(value, "_plan_inter_point_bridge", planner)
    monkeypatch.setattr(pose_ik, "solve_posture_ik", fake_constrained_ik)
    plans = [plan("R1", .2), plan("L1", -.2)]
    outcome = value.run([(target(row["name"]), row) for row in plans])
    assert outcome["success"] and outcome["training_accepted"]
    assert outcome["resets_between_points"] == 0
    assert len(planner_calls) == len(outcome["inter_point_routes"]) == 1
    route = outcome["inter_point_routes"][0]
    assert route["from_target"] == "R1" and route["to_target"] == "L1"
    assert route["planning_accepted"] and route["motion_complete"]
    assert 0 <= route["planning_wall_time_s"] < 30.
    assert (route["planning_simulation_time_s"], route["start_physics_row"]) == planner_calls[0][2:]
    np.testing.assert_array_equal(route["measured_start_q"], planner_calls[0][0])
    np.testing.assert_array_equal(route["joint_waypoints_q"][0], route["retained_command_q"])
    np.testing.assert_array_equal(route["joint_waypoints_q"][1], route["measured_start_q"])
    stage_index = route["lower_staging_waypoint_index"]
    np.testing.assert_array_equal(route["joint_waypoints_q"][stage_index:], plans[1]["transfer_q"][1:])
    data = json.dumps(route["joint_waypoints_q"], separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(data).hexdigest() == route["waypoints_sha256"]
    assert len(route["edge_checks"]) == len(route["joint_waypoints_q"]) - 1
    assert all(check["clear"] and check["minimum_distance_m"] >= .003 for check in route["edge_checks"])
    first_end = outcome["points"][0]["end_physics_row"]
    assert env.rows[first_end - 1]["subphase"] == "withdraw_settle"
    assert np.max(np.abs(env.rows[first_end - 1]["q_command"])) > .1
    assert env.rows[first_end]["subphase"] == "inter_point_transfer"
    assert all(row["target"] == "L1" for row in env.rows if row["subphase"] in ("return_home", "home_settle"))
    # The last point's canonical branch has negative joint 5; reversing the
    # inter-point path would wrongly revisit the prior positive branch.
    assert all(row["q_command"][4] <= 1e-9 for row in env.rows if row["subphase"] == "return_home")
    assert all(point["verified_front_contact_hold_s"] >= 3 - 2 / 240 for point in outcome["points"])
    assert not any(row["subphase"] == "settle" and row["target"] == "L1" for row in env.rows)
    assert np.max(np.abs(env.q)) < 1e-9
    assert all(b["time_s"] > a["time_s"] for a, b in zip(env.rows, env.rows[1:]))


def released_controller():
    value = controller()
    value.env.q = np.array([0., 0., .24, 0., .2, 0., 0.])
    value.q = value.env.q.copy()
    value.env.rows.append({"time_s": value.env.t})
    value.checker = PostureChecker(value.checker, value.env.base, value.posture_limits)
    value.sequence_index, value.sequence_length = 1, 2
    value.target = target("L1")
    value.inter_point_routes = []
    value.inter_point_rng = np.random.default_rng(65)
    value.released_prepose = {"target": "R1", "time_s": value.env.t,
                             "physics_row_end_exclusive": len(value.env.rows)}
    return value


@pytest.mark.parametrize("failure", ["unavailable", "deadline", "wrong_endpoint", "no_clearance"])
def test_inter_point_failure_stops_at_released_pose_without_home_fallback(monkeypatch, failure):
    value = released_controller()
    before_q, before_rows = value.q.copy(), len(value.env.rows)
    clock = [0.]
    monkeypatch.setattr(low_elbow_controller, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def planner(start, goal, deadline):
        assert deadline == 30.
        if failure == "unavailable":
            return None, {"reason": "bounded_search_exhausted"}
        if failure == "deadline":
            clock[0] = 30.001
        if failure == "wrong_endpoint":
            goal = np.zeros(7)
        return [start.copy(), goal.copy()], {"reason": "fixture"}

    monkeypatch.setattr(value, "_plan_inter_point_bridge", planner)
    if failure == "no_clearance":
        monkeypatch.setattr(value.checker, "check", lambda *args, **kwargs: {
            "clear": True, "minimum_distance_m": 0.})
    with pytest.raises(EpisodeRejected, match="inter-point"):
        value._move_to_prepose(plan("L1", -.2))
    assert len(value.env.rows) == before_rows
    np.testing.assert_array_equal(value.q, before_q)
    np.testing.assert_array_equal(value.env.q, before_q)
    assert len(value.inter_point_routes) == 1
    assert not value.inter_point_routes[0]["planning_accepted"]
    assert not value.inter_point_routes[0]["motion_complete"]


def test_missing_release_or_canonical_lower_tail_rejects_before_planning(monkeypatch):
    value = released_controller()

    def forbidden(*args, **kwargs):
        raise AssertionError("planning must not start before release and canonical tail validation")

    monkeypatch.setattr(value, "_plan_inter_point_bridge", forbidden)
    row = plan("L1", -.2)
    del row["lower_staging_q"]
    with pytest.raises(EpisodeRejected, match="lower staging"):
        value._move_to_prepose(row)
    value.released_prepose = None
    with pytest.raises(EpisodeRejected, match="verified released prepose"):
        value._move_to_prepose(plan("L1", -.2))
    assert len(value.env.rows) == 1
