"""Native-FCL CPU equivalence on pinned WAM meshes, no GPU or pytest needed."""
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]/"auscultation_expert_layout_v1"


def poses():
    result = [np.zeros(7)]
    for name in ("R1", "L1", "R3", "L3", "R5", "L5"):
        row = json.loads((ROOT/f"results/expert_v2/plans/planned_paths_{name}.json").read_text())["points"][0]
        transfer = np.asarray(row["transfer_q"])
        result.extend(transfer[np.linspace(0, len(transfer)-1, 4).astype(int)])
        result.extend([np.asarray(row["approach_q"])[-1], np.asarray(row["loaded_q"])])
    return result


@unittest.skipUnless(importlib.util.find_spec("fcl"), "native FCL CPU environment required")
class TestFastCollision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sim.auscultation_expert_layout_v1.collision import CollisionChecker
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        cls.base = json.loads((ROOT/"results/expert_v2/physics_all10/world_from_left_base.json").read_text())
        cls.kin = OpenArmKinematics()
        cls.original = CollisionChecker(cls.kin, cls.base, ROOT/"results/expert_v2/plans")
        cls.fast = None

    def pair(self):
        name = "sim.curobo_expert_v1.fast_collision"
        self.assertIsNotNone(importlib.util.find_spec(name), "batched conservative broad phase is missing")
        module = importlib.import_module(name)
        if type(self).fast is None:
            type(self).fast = module.FastCollisionChecker(self.kin, self.base, ROOT/"results/expert_v2/plans")
        return self.original, self.fast, module

    def test_preserves_complete_original_pair_set_and_vectorized_bounds(self):
        original, fast, _ = self.pair()
        self.assertEqual(original.entries, fast.entries)
        self.assertEqual(original.dynamic, fast.dynamic)
        self.assertEqual(original.allowed, fast.allowed)
        np.testing.assert_array_equal(original.pair_indices, fast.pair_indices)
        for q in poses()[::3]:
            for compression in (0., .0025):
                original.update(q, compression)
                fast._update_bounds(q, compression)
                np.testing.assert_allclose(fast.bounds, original.bounds, atol=2e-15, rtol=0)

    def test_native_triangle_results_match_in_free_and_near_contact_poses(self):
        original, fast, _ = self.pair()
        for allow_contact, compression, details in [
                (False, 0., False), (False, 0., True), (False, .0025, True),
                (True, 0., False), (True, .0025, True)]:
            statuses = []
            for index, q in enumerate(poses()):
                with self.subTest(pose=index, allow_contact=allow_contact, compression=compression, details=details):
                    options = dict(clearance=.003, allow_patient_contact=allow_contact,
                                   compression=compression, details=details)
                    a, b = original.check(q, **options), fast.check(q, **options)
                    self.assertIs(a["clear"], b["clear"])
                    self.assertEqual(a["pair"], b["pair"])
                    self.assertAlmostEqual(a["minimum_distance_m"], b["minimum_distance_m"], delta=2e-12)
                    statuses.append(a["clear"])
            self.assertTrue(any(statuses), "suite must include free poses")
            self.assertFalse(all(statuses), "suite must include near-contact/colliding poses")

    def test_only_near_dynamic_objects_receive_one_transform_each(self):
        import fcl
        _, fast, module = self.pair()
        native_objects = fast.objects

        class Counted:
            def __init__(self, obj):
                self.obj, self.calls = obj, 0
            def setTransform(self, transform):
                self.calls += 1
                self.obj.setTransform(transform)

        counted = [Counted(obj) for obj in native_objects]
        fast.objects = counted
        facade = SimpleNamespace(Transform=fcl.Transform, DistanceRequest=fcl.DistanceRequest,
            DistanceResult=fcl.DistanceResult,
            distance=lambda a,b,*rest:fcl.distance(a.obj,b.obj,*rest))
        try:
            with patch.object(module, "fcl", facade):
                fast.check(np.zeros(7), details=True)
            touched = {i for i,o in enumerate(counted) if o.calls}
            self.assertTrue(touched <= set(fast.dynamic))
            self.assertTrue(all(o.calls <= 1 for o in counted))
            self.assertLess(len(touched), len(fast.dynamic))
            self.assertEqual(touched, set(fast.last_updated_dynamic_objects))
            self.assertEqual(fast.last_checked_aabb_pair_count, len(fast.pairs))
        finally:
            fast.objects = native_objects


if __name__ == "__main__":
    unittest.main()
