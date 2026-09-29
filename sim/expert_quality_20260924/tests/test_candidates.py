"""CPU tests for candidate geometry, timing, and fail-closed checks."""
import time
from types import SimpleNamespace

import numpy as np
import pytest


def planner():
    from sim.expert_quality_20260924 import candidate_planner
    return candidate_planner


class Checker:
    def __init__(self, reject=None, distance=.003):
        self.reject, self.distance, self.calls = reject, distance, []

    def check(self, q, **kwargs):
        self.calls.append((q.copy(), kwargs))
        return {"clear": not (self.reject and self.reject(q)),
                "minimum_distance_m": self.distance}


def route():
    return np.array([[0., 0., 0., 0., 0., 0., 0.],
                     [.3, .3, 0., 0., 0., 0., 0.],
                     [.6, 0., 0., 0., 0., 0., 0.]])


def test_shortcut_reduces_angle_without_changing_endpoints():
    module = planner()
    path, record = module.checked_shortcut(route(), Checker(), time.monotonic()+10)
    assert len(path) == 2 and record["shortcut_accepted"]
    np.testing.assert_array_equal(path[[0, -1]], route()[[0, -1]])
    assert module.path_metrics(path)["total_joint_travel_rad"] == pytest.approx(.6)
    assert record["baseline_metrics"]["total_joint_travel_rad"] == pytest.approx(1.2)


def test_collision_rejects_shortcut_and_preserves_checked_staging_route():
    module = planner()
    checker = Checker(reject=lambda q: .15 < q[0] < .45 and q[1] < .1)
    path, record = module.checked_shortcut(route(), checker, time.monotonic()+10)
    np.testing.assert_array_equal(path, route())
    assert not record["shortcut_accepted"]


@pytest.mark.parametrize("distance", [0., float("nan")])
def test_clear_flag_cannot_override_missing_clearance(distance):
    with pytest.raises(ValueError, match="baseline route"):
        planner().checked_shortcut(route(), Checker(distance=distance), time.monotonic()+10)


def test_deadline_fails_before_any_collision_check():
    checker = Checker()
    with pytest.raises(TimeoutError):
        planner().checked_shortcut(route(), checker, time.monotonic()-1)
    assert not checker.calls


def test_continuous_timing_preserves_route_and_speed_and_removes_per_waypoint_stops():
    module = planner()
    path = np.linspace(route()[0], route()[-1], 61)
    command = module.path_commands(path, speed_limit=.25)
    np.testing.assert_allclose(command[-1], path[-1], atol=1e-14)
    assert len(command) % 12 == 0
    dq = np.diff(np.vstack([path[0], command]), axis=0)
    assert np.max(np.abs(dq))*240 <= .25+1e-9
    assert module.path_metrics(command)["total_joint_travel_rad"] == pytest.approx(.6, abs=1e-6)
    assert len(command)/240 < 5
    assert np.count_nonzero(np.abs(dq[:, 0]) < 1e-10) == 0


def test_continuous_commands_follow_curved_polyline():
    path = route()
    path[1, 1] = .0001
    command = planner().path_commands(path, speed_limit=.25)
    # Each command lies exactly on one of the original checked line segments.
    expected_y = np.minimum(command[:, 0], .6-command[:, 0])*(.0001/.3)
    np.testing.assert_allclose(command[:, 1], expected_y, atol=1e-14)


def test_sharp_corner_rejected_within_bounded_duration():
    module = planner()
    with pytest.raises(RuntimeError, match="dynamic timing budget"):
        module.path_commands(route(), maximum_iterations=1)
    with pytest.raises(RuntimeError, match="dynamic timing budget"):
        module.path_commands(route())


def test_real_ik_tail_fails_unretimed_jerk_and_passes_bounded_retiming():
    import json
    module = planner()
    path = module.FROZEN/"results/expert_v2/plans/planned_paths_R1.json"
    row = json.loads(path.read_text())["points"][0]
    q = np.asarray(row["transfer_q"])
    index = np.flatnonzero(np.all(np.isclose(q, row["lower_staging_q"], atol=1e-9, rtol=0), axis=1))[0]
    tail = q[index:]
    with pytest.raises(RuntimeError, match="dynamic timing budget"):
        module.path_commands(tail, maximum_iterations=1)
    commands, report = module.path_commands(tail, return_report=True)
    assert report["attempts"] > 1
    assert report["geometry"] == "original piecewise-linear joint path; no C2 claim"
    assert report["peaks"]["velocity_rad_s"] <= .25+1e-9
    assert report["peaks"]["acceleration_rad_s2"] <= 2.+1e-9
    assert report["peaks"]["jerk_rad_s3"] <= 20.+1e-9
    assert report["duration_s"] <= 120.
    np.testing.assert_allclose(commands[-1], tail[-1])
    padded = np.vstack([np.repeat(tail[:1], 3, axis=0), commands,
                        np.repeat(tail[-1:], 3, axis=0)])
    assert np.max(np.abs(np.diff(padded, n=3, axis=0)))*240**3 <= 20.+1e-9


def test_dynamic_timing_rejects_duration_or_iteration_exhaustion():
    module = planner()
    for options in ({"maximum_seconds": 1.}, {"maximum_iterations": 1}, {"maximum_compute_seconds": 1e-12}):
        with pytest.raises(RuntimeError, match="dynamic timing budget"):
            module.path_commands(route(), **options)


def test_duplicate_knots_safe_and_invalid_paths_rejected():
    module = planner()
    p = np.vstack([route()[0], route()[0], route()[-1]])
    np.testing.assert_allclose(module.path_commands(p)[-1], p[-1])
    for path in [np.zeros((0, 7)), np.zeros((2, 6)), np.full((2, 7), np.nan)]:
        with pytest.raises(ValueError):
            module.path_commands(path)


def test_candidate_rejects_missing_release_before_motion():
    from sim.expert_quality_20260924.candidate_controller import CandidateController
    value = object.__new__(CandidateController)
    value.sequence_index = 1
    value.policy = "checked_shortcut"
    value.released_prepose = None
    value.env = SimpleNamespace(state=lambda: {"q": np.zeros(7), "time_s": 0., "hall_N": 0.},
                                last_contacts={}, rows=[])
    value.q = np.zeros(7)
    row = {"success": True, "transfer_q": route().tolist(), "lower_staging_q": route()[1].tolist()}
    with pytest.raises(RuntimeError, match="verified released prepose"):
        value._move_to_prepose(row)


def test_candidate_sources_included_in_frozen_runner_snapshot():
    from sim.expert_quality_20260924.run_candidate import candidate_source_paths
    paths = candidate_source_paths()
    assert {"candidate/candidate_controller.py", "candidate/candidate_planner.py", "candidate/run_candidate.py"} <= paths.keys()
    assert all(path.is_file() for path in paths.values())


@pytest.mark.parametrize("policy", ["staged_continuous", "checked_shortcut"])
@pytest.mark.parametrize("force_failure", [False, True])
def test_complete_candidate_retains_force_gate_hold_order_and_return(monkeypatch, policy, force_failure):
    from sim.expert_quality_20260924.candidate_controller import CandidateController
    from sim.auscultation_expert_layout_v1 import pose_ik
    from sim.auscultation_training.tests.test_controller import FakeEnv, target
    from sim.auscultation_training.tests.test_low_elbow_runtime import PostureKin, PostureCollisionChecker, profile
    from sim.auscultation_training.tests.test_low_elbow_sequence import fake_constrained_ik
    env, kin = FakeEnv(failure=force_failure), PostureKin()
    value = CandidateController(env, kin, PostureCollisionChecker(kin), policy=policy)
    monkeypatch.setattr(value, "_plan_inter_point_bridge", lambda start, goal, deadline:
                        ([start.copy(), goal.copy()], {"reason": "checked synthetic fixture"}))
    monkeypatch.setattr(pose_ik, "solve_posture_ik", fake_constrained_ik)
    plans = []
    for name, wrist in [("R1", .2), ("L1", -.2)]:
        stage = np.array([0., 0., .18, 0., wrist, 0., 0.])
        pre = stage.copy(); pre[2] = .24
        plans.append({"name": name, "success": True, "command_normal_world": [0., 0., 1.],
                      "command_rotation_world": np.diag([1., -1., -1.]).tolist(),
                      "posture_limits": profile(), "lower_staging_q": stage.tolist(),
                      "transfer_q": np.vstack([np.zeros(7), np.linspace(stage, pre, 31)]).tolist()})
    outcome = value.run([(target(row["name"]), row) for row in plans])
    assert outcome["candidate_policy"] == policy
    assert outcome["candidate_method"].endswith("CandidateController")
    assert outcome["training_accepted"] is not force_failure
    if force_failure:
        assert len(outcome["points"]) == 1
        assert not any(r.get("training_accepted") for r in env.rows)
        assert outcome["points"][0]["recovery"]["attempted"]
    else:
        assert [p["target"] for p in outcome["points"]] == ["R1", "L1"]
        assert all(p["verified_front_contact_hold_s"] >= 3-2/240 for p in outcome["points"])
        assert np.max(np.abs(env.q)) < 1e-9
        assert outcome["inter_point_routes"][0]["lower_staging_retained"] is (policy == "staged_continuous")
        assert all(np.max(np.abs(r["action_joint_delta_rad"])) <= .25*r["action_duration_s"]+1e-9
                   for r in value.control_rows)


def test_manifest_references_captured_bytes_and_outcome_hash(tmp_path):
    import hashlib
    import json
    from sim.expert_quality_20260924.run_candidate import candidate_source_paths, bind_candidate_manifest
    objects = tmp_path/"input_snapshot/objects"
    objects.mkdir(parents=True)
    files = {}
    for key, source in candidate_source_paths().items():
        data = source.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        (objects/digest).write_bytes(data)
        files["source/"+key] = {"sha256": digest}
    (objects.parent/"manifest.json").write_text(json.dumps({"files": files}))
    (tmp_path/"outcome.json").write_text(json.dumps({"success": False}))
    manifest = bind_candidate_manifest(tmp_path, "staged_continuous")
    outcome = json.loads((tmp_path/"outcome.json").read_text())
    data = (tmp_path/outcome["candidate_sources_manifest"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == outcome["candidate_sources_manifest_sha256"]
    for record in manifest["files"].values():
        assert hashlib.sha256((tmp_path/record["path"]).read_bytes()).hexdigest() == record["sha256"]


@pytest.mark.parametrize("failure", ["unavailable", "wrong_endpoint", "no_clearance", "time_advanced"])
def test_released_pose_kept_on_candidate_planning_failure(monkeypatch, failure):
    from sim.expert_quality_20260924.candidate_controller import CandidateController
    from sim.auscultation_training.tests.test_controller import FakeEnv, target
    from sim.auscultation_training.tests.test_low_elbow_runtime import PostureKin, PostureCollisionChecker
    env, kin = FakeEnv(), PostureKin()
    value = CandidateController(env, kin, PostureCollisionChecker(kin), policy="checked_shortcut")
    env.q = np.array([0., 0., .24, 0., .2, 0., 0.]); value.q = env.q.copy()
    value.sequence_index, value.sequence_length = 1, 2
    value.target = target("L1")
    value.released_prepose = {"target": "R1", "time_s": env.t, "physics_row_end_exclusive": 0}
    value.inter_point_routes = []
    stage = np.array([0., 0., .18, 0., -.2, 0., 0.]); pre = stage.copy(); pre[2] = .24
    row = {"success": True, "transfer_q": [np.zeros(7), stage, pre], "lower_staging_q": stage}

    def bridge(start, goal, deadline):
        if failure == "unavailable":
            return None, {"reason": "infeasible"}
        if failure == "wrong_endpoint":
            goal = np.zeros(7)
        if failure == "time_advanced":
            env.t += .01
        return [start, goal], {"reason": "synthetic fixture"}

    monkeypatch.setattr(value, "_plan_inter_point_bridge", bridge)
    if failure == "no_clearance":
        monkeypatch.setattr(value.checker, "check", lambda *a, **kw: {"clear": True, "minimum_distance_m": 0.})
    initial = value.q.copy()
    with pytest.raises((RuntimeError, ValueError)):
        value._move_to_prepose(row)
    assert not env.rows
    np.testing.assert_array_equal(value.q, initial)
    np.testing.assert_array_equal(env.q, initial)
    assert not value.inter_point_routes[0]["planning_accepted"]
    assert not value.inter_point_routes[0]["motion_complete"]
