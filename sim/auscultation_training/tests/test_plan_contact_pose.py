"""Safety semantics for geometry-only, bounded contact-pose planning."""

import importlib
import importlib.util
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def planner():
    name = "sim.auscultation_training.plan_contact_pose"
    assert importlib.util.find_spec(name) is not None, "bounded contact-pose planner is missing"
    return importlib.import_module(name)


def test_fit_center_uses_original_normal_and_keeps_target_unchanged():
    module = planner()
    original = np.array([0.1, 0.8, 1.3])
    target = SimpleNamespace(position_world=original.copy(), normal_world=np.array([0., -1., 0.]))
    rotation = Rotation.from_euler("xyz", [.9, .2, -.5]).as_matrix()
    normal = -rotation[:, 2]
    candidate = {"normal_world": normal.tolist(), "wrist_rotation_world": rotation.tolist(), "gap_m": .005}
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("z", .7).as_matrix()
    base[:3, 3] = [.1, .2, .3]
    geometry = module.candidate_geometry(target, candidate, base)
    np.testing.assert_allclose(geometry["center_world"], original + target.normal_world * .005)
    np.testing.assert_allclose(geometry["pre_world"], original + normal * .06)
    np.testing.assert_allclose(geometry["free_end_world"], original + normal * .015)
    np.testing.assert_allclose(base[:3, :3] @ geometry["rotation_base"], rotation)
    np.testing.assert_array_equal(target.position_world, original)


def test_fit_orientation_must_match_its_command_normal():
    target = SimpleNamespace(position_world=np.zeros(3), normal_world=np.array([0., 0., 1.]))
    candidate = {"normal_world": [0, 0, 1], "wrist_rotation_world": np.eye(3).tolist(), "gap_m": .005}
    with pytest.raises(ValueError, match="normal"):
        planner().candidate_geometry(target, candidate, np.eye(4))


def test_rrt_stops_before_collision_work_when_budget_expired():
    class Checker:
        kinematics = SimpleNamespace(lower_limits=np.full(7, -1.), upper_limits=np.full(7, 1.))

        def check(self, *args, **kwargs):
            pytest.fail("expired search must not perform more collision work")

    path, report = planner().bounded_rrt_path(
        Checker(), np.zeros(7), np.ones(7), np.random.default_rng(0),
        deadline=time.monotonic() - 1, max_iterations=2,
    )
    assert path is None
    assert report["reason"] == "time_budget_exhausted"


def test_rrt_rejects_final_collision_sample_overrunning_deadline(monkeypatch):
    module = planner()
    clock = [0.]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    class Checker:
        kinematics = SimpleNamespace(lower_limits=np.full(7, -1.), upper_limits=np.full(7, 1.))

        def check(self, *args, **kwargs):
            clock[0] += .6
            return {"clear": True, "minimum_distance_m": kwargs["clearance"]}

    path, report = module.bounded_rrt_path(
        Checker(), np.zeros(7), np.zeros(7), np.random.default_rng(0),
        deadline=1., max_iterations=2,
    )
    assert path is None
    assert report["reason"] == "time_budget_exhausted"


def test_rrt_iteration_bound_reports_failure_for_separating_obstacle():
    class Checker:
        kinematics = SimpleNamespace(lower_limits=np.full(7, -1.), upper_limits=np.full(7, 1.))

        def check(self, q, **kwargs):
            # A configuration-space slab separates start and goal.
            clear = not -.1 <= q[0] <= .1
            return {"clear": clear, "minimum_distance_m": .003 if clear else 0.}

    path, report = planner().bounded_rrt_path(
        Checker(), np.full(7, -.5), np.full(7, .5), np.random.default_rng(0),
        deadline=time.monotonic() + 5, max_iterations=2,
    )
    assert path is None
    assert report["reason"] == "iteration_budget_exhausted"
    assert report["edge_samples"] > 0


def test_endpoint_hypothesis_is_not_success_without_home_transfer(monkeypatch):
    module = planner()
    target = SimpleNamespace(name="R1", position_world=np.zeros(3), normal_world=np.array([0., 0., 1.]))
    rotation = np.diag([1., -1., -1.])
    candidate = {"normal_world": [0, 0, 1], "wrist_rotation_world": rotation.tolist(), "gap_m": .005,
                 "angle_change_deg": 0., "fixed_clearance_m": .002, "shell_clearance_m": .001}

    class Kinematics:
        lower_limits, upper_limits = np.full(7, -1.), np.full(7, 1.)

        def solve_ik(self, position, **kwargs):
            # A simple reachable pose manifold isolates the route acceptance gate.
            assert "orientation_base" in kwargs
            return SimpleNamespace(q=np.r_[position, np.zeros(4)], success=True,
                                   position_error_m=0., orientation_error_rad=0., nfev=1)

    class Checker:
        kinematics = Kinematics()
        calls = []

        def check(self, q, **kwargs):
            self.calls.append(kwargs)
            return {"clear": True, "minimum_distance_m": kwargs["clearance"]}

    checker = Checker()
    monkeypatch.setattr(module, "bounded_rrt_path", lambda *a, **k: (None, {"reason": "iteration_budget_exhausted"}))
    row = module.plan_target(target, {"name": "R1", "compression_m": .002, "candidates": [candidate]},
                             np.eye(4), checker.kinematics, checker, max_attempts=1)
    assert row["success"] is False
    assert "transfer_q" not in row
    assert row["failure_counts"] == {"transfer_iteration_budget_exhausted": 1}
    permitted = [call for call in checker.calls if call.get("allow_patient_contact")]
    assert len(permitted) == 2
    assert all(call["compression"] == .002 and call["clearance"] == .0003 for call in permitted)


def test_bidirectional_route_goes_around_obstacle_and_preserves_endpoints():
    module = planner()

    class Checker:
        kinematics = SimpleNamespace(lower_limits=np.full(7, -1.), upper_limits=np.full(7, 1.))

        def check(self, q, **kwargs):
            clear = np.linalg.norm(q[:2]) > .2
            return {'clear': clear, 'minimum_distance_m': .003 if clear else 0.}

    start, goal = np.zeros(7), np.zeros(7)
    start[0], goal[0] = -.5, .5
    checker = Checker()
    path, report = module.bounded_rrt_path(checker, start, goal, np.random.default_rng(9),
                                          deadline=time.monotonic()+5, max_iterations=100)
    assert report['reason'] == 'route_found'
    np.testing.assert_array_equal(path[0], start)
    np.testing.assert_array_equal(path[-1], goal)
    for left, right in zip(path, path[1:]):
        for q in np.linspace(left, right, 200):
            assert checker.check(q)['clear']


def test_closer_prepose_keeps_original_target_and_full_orientation():
    module = planner()
    target = SimpleNamespace(position_world=np.array([1.,2.,3.]), normal_world=np.array([0.,0.,1.]))
    candidate = {'normal_world': [0,0,1], 'wrist_rotation_world': np.diag([1.,-1.,-1.]).tolist(), 'gap_m': .003}
    geo = module.candidate_geometry(target, candidate, np.eye(4), pre_gap=.025)
    np.testing.assert_allclose(geo['pre_world'], [1,2,3.025])
    np.testing.assert_allclose(geo['center_world'], [1,2,3.003])
    with pytest.raises(ValueError, match='pre_gap'):
        module.candidate_geometry(target, candidate, np.eye(4), pre_gap=.005)
