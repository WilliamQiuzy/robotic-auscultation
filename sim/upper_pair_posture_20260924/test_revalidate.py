"""Pure CPU contract tests; these do not certify mesh collision acceptance."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from sim.upper_pair_posture_20260924 import contract


class RecordingChecker:
    def __init__(self, reject=None):
        self.calls = []
        self.reject = reject

    def check(self, q, **kwargs):
        self.calls.append((np.asarray(q).copy(), dict(kwargs)))
        clear = not (self.reject and self.reject(q, kwargs))
        return {"clear": clear, "minimum_distance_m": kwargs["clearance"] if clear else 0.,
                "pair": None}


class RevalidationTests(unittest.TestCase):
    def module(self):
        name = "sim.upper_pair_posture_20260924.revalidate"
        self.assertIsNotNone(importlib.util.find_spec(name), "independent revalidator must exist")
        return importlib.import_module(name)

    def setUp(self):
        self.plan = contract.read(contract.PLANS / "planned_paths_R1.json")
        self.plan.pop("revalidation", None)
        self.kin = OpenArmKinematics()

    def test_checks_complete_routes_both_directions_including_home_and_staging(self):
        mod = self.module()
        checker = RecordingChecker()
        result = mod.check_plan_geometry(self.plan, self.kin, checker)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["independent_upper_pair_check"])
        self.assertEqual(set(result["routes"]), {"transfer", "approach", "contact_corridor",
                         "nominal_withdrawal", "return_approach", "return_transfer"})
        row = self.plan["points"][0]
        self.assertTrue(any(np.array_equal(q, np.zeros(7)) for q, _ in checker.calls))
        self.assertTrue(any(np.array_equal(q, row["lower_staging_q"]) for q, _ in checker.calls))
        self.assertTrue(any(kw["compression"] > 0 and kw["contact"] for _, kw in checker.calls))
        for label, route in result["routes"].items():
            self.assertGreater(route["checked_samples"], 1, label)
            self.assertLessEqual(route["max_sample_joint_step_rad"], .015 + 1e-12)

    def test_rejects_collision_in_interior_coupled_sample(self):
        mod = self.module()
        checker = RecordingChecker(lambda q, kw: 0.0002 < kw["compression"] < 0.0018)
        with self.assertRaisesRegex(ValueError, "contact_corridor.*rejected"):
            mod.check_plan_geometry(self.plan, self.kin, checker)

    def test_rejects_nonunique_staging_even_when_collision_clear(self):
        mod = self.module()
        row = self.plan["points"][0]
        row["transfer_q"].insert(1, row["lower_staging_q"])
        with self.assertRaisesRegex(ValueError, "unique interior lower-staging"):
            mod.check_plan_geometry(self.plan, self.kin, RecordingChecker())

    def test_rejects_disconnected_contact_corridor(self):
        mod = self.module()
        self.plan["points"][0]["contact_corridor_posture_q"][0][0] += .01
        with self.assertRaisesRegex(ValueError, "contact corridor.*disconnected"):
            mod.check_plan_geometry(self.plan, self.kin, RecordingChecker())

    def test_rejects_wrong_loaded_pose_even_when_collision_clear(self):
        mod = self.module()
        row = self.plan["points"][0]
        row["loaded_q"][0] += .01
        row["contact_corridor_posture_q"][-1] = row["loaded_q"].copy()
        with self.assertRaisesRegex(ValueError, "loaded.*pose"):
            mod.check_plan_geometry(self.plan, self.kin, RecordingChecker())

    def test_coupling_is_bound_to_audited_center_not_candidate_assertion(self):
        mod = self.module()
        self.plan["points"][0]["contact_assembly_model"]["center_base"][0] += .01
        with self.assertRaisesRegex(ValueError, "coupling.*center"):
            mod.check_plan_geometry(self.plan, self.kin, RecordingChecker())

    def test_published_receipt_preserves_exact_unvalidated_bytes_and_provenance(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planned_paths_R1.json"
            original = json.dumps(self.plan, separators=(",", ":")).encode()
            path.write_bytes(original)
            evidence = {"accepted": True, "independent_upper_pair_check": True,
                        "collider_receipt": {"collision_geometry_sha256": "abc"}}
            mod.write_verified_plan(path, original, evidence)
            before = path.with_suffix(".before_revalidation.json")
            result = json.loads(path.read_bytes())
            self.assertEqual(before.read_bytes(), original)
            self.assertEqual(result["provenance"], self.plan["provenance"])
            self.assertEqual(result["revalidation"]["input_plan_sha256"], contract.sha(before))
            result.pop("revalidation")
            self.assertEqual(result, self.plan)

    def test_changed_input_cannot_receive_old_evidence(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planned_paths_R1.json"
            original = json.dumps(self.plan).encode()
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "changed during revalidation"):
                mod.write_verified_plan(path, original, {"accepted": True})
            self.assertFalse(path.with_suffix(".before_revalidation.json").exists())

    def test_revalidation_cannot_inherit_existing_acceptance(self):
        mod = self.module()
        self.plan["revalidation"] = {"accepted": True}
        with self.assertRaisesRegex(ValueError, "inherited revalidation"):
            mod.check_plan_geometry(self.plan, self.kin, RecordingChecker())


if __name__ == "__main__":
    unittest.main()
