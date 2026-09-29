"""Pure CPU regressions for new-layout source and sampled safety gates."""
import importlib
import unittest
from types import SimpleNamespace
import numpy as np

try:
    planner = importlib.import_module("sim.auscultation_expert_layout_v1.replan_layout")
except ModuleNotFoundError as error:
    if error.name != "sim.auscultation_expert_layout_v1.replan_layout":
        raise
    planner = None


class ReplanLayoutTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(planner, "new-layout planner must exist")

    def test_tightening_keeps_home_but_rejects_old_joint2_boundary(self):
        limits = SimpleNamespace(joint_lower_rad=np.array([-2.] * 3 + [0.] + [-2.] * 3),
                                 joint_upper_rad=np.array([2., .1745, 2., 2., 2., 2., 2.]))
        lower, upper = planner.tightened_bounds(limits, 3.)
        self.assertEqual(lower[3], 0.)
        self.assertTrue(np.all(lower <= 0.))
        self.assertGreater(limits.joint_upper_rad[1] - np.deg2rad(.2), upper[1])

    def test_urdf_inset_does_not_add_another_three_degrees_to_posture(self):
        kin = SimpleNamespace(lower_limits=np.array([-2.] * 3 + [0.] + [-2.] * 3),
                              upper_limits=np.array([2.] * 7))
        published_lower = kin.lower_limits + np.deg2rad(3.)
        published_lower[3] = 0.
        limits = SimpleNamespace(joint_lower_rad=published_lower,
                                 joint_upper_rad=kin.upper_limits-np.deg2rad(3.))
        lower, upper = planner.tightened_bounds(limits, 3., kin)
        np.testing.assert_array_equal(lower, published_lower)
        np.testing.assert_array_equal(upper, limits.joint_upper_rad)

    def test_local_ik_branch_jump_is_rejected_before_collision_check(self):
        from unittest.mock import patch
        checked = []
        checker = SimpleNamespace(edge=lambda *args, **kwargs: checked.append(args))
        first, last = np.eye(4), np.eye(4)
        last[0, 3] = .001
        with patch.object(planner, "solve", return_value=(np.ones(7), True, 0., 0.)):
            route, failure = planner.follow_poses(None, None, None, checker, np.zeros(7),
                                                  [first, last], float("inf"))
        self.assertIsNone(route)
        self.assertEqual(failure["reason"], "local_ik_or_branch_jump")
        self.assertFalse(checked)

    def test_coupled_contact_samples_compression_from_each_joint_sample(self):
        self.assertTrue(hasattr(planner, "coupled_contact_edge"), "coupled edge checker is required")
        calls = []
        class Kin:
            def tcp(self, q):
                pose = np.eye(4); pose[0, 3] = q[0]
                return pose
        def check(q, **kwargs):
            calls.append((q.copy(), kwargs))
            return {"clear": True, "minimum_distance_m": kwargs["clearance"]}
        start, end = np.zeros(7), np.zeros(7)
        start[0], end[0] = .006, .003
        model = {"center_base": [.005, 0., 0.], "normal_base": [1., 0., 0.], "compression_m": .002}
        result = planner.coupled_contact_edge(Kin(), SimpleNamespace(check=check), start, end, model)
        self.assertTrue(result["clear"])
        self.assertGreater(len(calls), 2)
        for q, kwargs in calls:
            self.assertAlmostEqual(kwargs["compression"], max(0., min(.002, .005-q[0])))
            self.assertEqual(kwargs["clearance"], .0003)
            self.assertTrue(kwargs["contact"] and kwargs["allow_patient_contact"])
        self.assertAlmostEqual(calls[0][1]["compression"], 0.)
        self.assertAlmostEqual(calls[-1][1]["compression"], .002)

    def test_coupled_contact_cannot_hide_interior_forbidden_collision(self):
        self.assertTrue(hasattr(planner, "coupled_contact_edge"), "coupled edge checker is required")
        class Kin:
            def tcp(self, q):
                pose = np.eye(4); pose[0, 3] = q[0]
                return pose
        def check(q, **kwargs):
            blocked = .0005 < kwargs["compression"] < .0015
            return {"clear": not blocked, "minimum_distance_m": 0. if blocked else .0003}
        start, end = np.zeros(7), np.zeros(7)
        start[0], end[0] = .005, .003
        model = {"center_base": [.005, 0., 0.], "normal_base": [1., 0., 0.], "compression_m": .002}
        result = planner.coupled_contact_edge(Kin(), SimpleNamespace(check=check), start, end, model)
        self.assertFalse(result["clear"])
        self.assertGreater(result["fraction"], 0.)
        self.assertLess(result["fraction"], 1.)

    def test_loaded_coupling_rejects_wrong_sign_and_unclamped_overshoot(self):
        self.assertTrue(hasattr(planner, "validate_coupled_endpoint"), "loaded coupling identity gate is required")
        class Kin:
            def tcp(self, q):
                pose = np.eye(4); pose[0, 3] = q[0]
                return pose
        q = np.zeros(7); q[0] = .003
        model = {"center_base": [.005, 0., 0.], "normal_base": [1., 0., 0.], "compression_m": .002}
        self.assertAlmostEqual(planner.validate_coupled_endpoint(Kin(), q, model)["compression_m"], .002)
        with self.assertRaisesRegex(ValueError, "endpoint"):
            planner.validate_coupled_endpoint(Kin(), q, {**model, "normal_base": [-1., 0., 0.]})
        q[0] = .002
        with self.assertRaisesRegex(ValueError, "endpoint"):
            planner.validate_coupled_endpoint(Kin(), q, model)

    def test_compression_law_rejects_nonfinite_input(self):
        self.assertTrue(hasattr(planner, "nominal_compression"), "finite compression law is required")
        with self.assertRaises(ValueError):
            planner.nominal_compression(None, None,
                {"center_base": [float("nan"), 0., 0.], "normal_base": [1., 0., 0.], "compression_m": .002})

    def test_wrong_layout_translation_is_rejected(self):
        old = {"position_world": [1., 2., 3.], "normal_world": [0., -1., 0.]}
        target = SimpleNamespace(position_world=np.array([1., 2.05, 2.98]),
                                 normal_world=np.array([0., -1., 0.]))
        planner.validate_translation(target, old)
        target.position_world[2] += .001
        with self.assertRaisesRegex(ValueError, "approved translation"):
            planner.validate_translation(target, old)

    def test_edge_checks_interior_and_rejects_collision(self):
        calls = []
        class Collision:
            kinematics = SimpleNamespace()
            def check(self, q, **kwargs):
                calls.append(kwargs)
                blocked = .02 < q[0] < .04
                return {"clear": not blocked, "minimum_distance_m": 0. if blocked else kwargs["clearance"]}
        class Limits:
            joint_lower_rad = [-2.] * 7
            joint_upper_rad = [2.] * 7
            max_elbow_z_m = 2.
            min_contact_elbow_below_wrist_m = 0.
            def evaluate(self, kin, base, q, contact=False):
                return {"clear": True, "elbow_z_m": 1., "elbow_below_wrist_m": .1,
                        "posture_violations": []}
        checker = planner.SafetyChecker(Collision(), np.eye(4), Limits(), 3.)
        check = checker.edge(np.zeros(7), np.array([.06, 0, 0, 0, 0, 0, 0]))
        self.assertFalse(check["clear"])
        self.assertGreater(check["fraction"], 0.)
        self.assertLess(check["fraction"], 1.)
        self.assertTrue(all(c["clearance"] == .003 and not c["allow_patient_contact"] for c in calls))

    def test_contact_corridor_requires_below_wrist_at_each_sample(self):
        class Collision:
            kinematics = SimpleNamespace()
            def check(self, q, **kwargs):
                return {"clear": True, "minimum_distance_m": kwargs["clearance"]}
        class Limits:
            joint_lower_rad = [-2.] * 7
            joint_upper_rad = [2.] * 7
            max_elbow_z_m = 2.
            min_contact_elbow_below_wrist_m = 0.
            def evaluate(self, kin, base, q, contact=False):
                return {"clear": True, "elbow_z_m": 1., "elbow_below_wrist_m": .004,
                        "posture_violations": []}
        checker = planner.SafetyChecker(Collision(), np.eye(4), Limits(), 3.)
        self.assertTrue(checker.check(np.zeros(7))["clear"])
        self.assertFalse(checker.check(np.zeros(7), contact=True)["clear"])

    def test_source_replacement_is_detected(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_bytes(b"before")
            pinned = planner.PinnedInputs()
            pinned.read(path)
            path.write_bytes(b"after")
            with self.assertRaisesRegex(ValueError, "changed"):
                pinned.verify()


if __name__ == "__main__":
    unittest.main()
