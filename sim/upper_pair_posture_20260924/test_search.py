"""CPU contract tests; the real FCL gate is deliberately not mocked as acceptance."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


def implementation():
    try:
        return importlib.import_module("sim.upper_pair_posture_20260924.search")
    except ImportError:
        return None


def fit(degrees=0., normal=None, gap=.005):
    rotation = np.diag([1., -1., -1.]) @ Rotation.from_euler("z", degrees, degrees=True).as_matrix()
    return {"normal_world": [0., 0., 1.] if normal is None else normal,
            "wrist_rotation_world": rotation.tolist(), "gap_m": gap,
            "fixed_clearance_m": .0011, "shell_clearance_m": .0004, "face_distance_m": .001}


class PoseKinematics:
    def tcp(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q)[:3]
        return pose


class SearchContracts(unittest.TestCase):
    def setUp(self):
        self.search = implementation()
        self.assertIsNotNone(self.search, "bounded posture search implementation is missing")

    def test_matching_rolls_keep_exact_normal_gap_and_original_indices(self):
        audit = {"candidates": [fit(180), fit(10), fit(0), fit(0, gap=.006),
                                fit(0, normal=[.01, 0., 1.]), fit(-40)]}
        found = self.search.matching_roll_candidates(audit, fit())
        self.assertEqual([x["audit_candidate_index"] for x in found], [2, 1, 5, 0])
        self.assertEqual([round(x["relative_roll_deg"]) for x in found], [0, 10, -40, 180])
        for row in found:
            self.assertEqual(row["fit"], audit["candidates"][row["audit_candidate_index"]])

    def test_matching_rejects_nonaxial_rotation_even_if_normal_field_unchanged(self):
        bad = fit()
        bad["wrist_rotation_world"] = Rotation.from_euler("x", .2).as_matrix().tolist()
        self.assertEqual(self.search.matching_roll_candidates({"candidates": [bad]}, fit()), [])

    def test_latest_tail_has_no_historical_translation(self):
        route = [[0.] * 7, [.1, .2, .3, 0, 0, 0, 0], [.15, .25, .35, 0, 0, 0, 0]]
        row = {"transfer_q": route, "lower_staging_q": route[1]}
        tail = self.search.current_staging_tail(row, PoseKinematics())
        np.testing.assert_allclose(tail[0][:3, 3], [.1, .2, .3], atol=0, rtol=0)
        np.testing.assert_allclose(tail[-1][:3, 3], [.15, .25, .35], atol=0, rtol=0)
        duplicate = {"transfer_q": route + [route[1]], "lower_staging_q": route[1]}
        with self.assertRaisesRegex(ValueError, "unique"):
            self.search.current_staging_tail(duplicate, PoseKinematics())

    def test_tail_roll_preserves_positions_and_tool_axis(self):
        tail = [np.eye(4), np.eye(4)]
        tail[1][:3, 3] = [.3, .2, .1]
        original = copy.deepcopy(tail)
        rolled = self.search.rolled_staging_tail(tail, np.deg2rad(30), "constant")
        for a, b in zip(original, rolled):
            np.testing.assert_array_equal(a[:3, 3], b[:3, 3])
            np.testing.assert_allclose(a[:3, 2], b[:3, 2], atol=1e-12)
        np.testing.assert_array_equal(tail[1], original[1])
        ramp = self.search.rolled_staging_tail(tail, np.deg2rad(30), "ramp")
        np.testing.assert_allclose(ramp[0][:3, :3], original[0][:3, :3], atol=1e-12)
        np.testing.assert_allclose(ramp[-1][:3, :3], rolled[-1][:3, :3], atol=1e-12)

    def test_motion_metrics_count_out_and_back_and_ignore_stationary_reversals(self):
        q = np.zeros((5, 7))
        q[:, 3] = [0., .2, .2, .1, 0.]
        q[:, 6] = [0., .4, .4, .1, 0.]
        report = self.search.joint_metrics(q)
        self.assertAlmostEqual(report["elbow_travel_rad"], .4)
        self.assertAlmostEqual(report["wrist_travel_rad"], .8)
        self.assertAlmostEqual(report["total_joint_travel_rad"], 1.2)
        self.assertEqual(report["joint_direction_reversals"], [0, 0, 0, 1, 0, 0, 1])
        self.assertEqual(report["max_joint_excursion_rad"], .4)

    def test_diverse_selection_deduplicates_posture_and_keeps_lower_motion(self):
        def candidate(q, score):
            return {"qpre": [q] * 7, "endpoint_metrics": {"wrist_travel_rad": score,
                    "elbow_travel_rad": .1, "total_joint_travel_rad": 5 + score}}
        selected = self.search.diverse_candidates([candidate(.01, 4), candidate(0., 1),
                                                   candidate(.3, 2)], limit=3)
        self.assertEqual([x["qpre"][0] for x in selected], [0., .3])

    def test_fresh_document_does_not_inherit_acceptance_or_change_provenance(self):
        source = {"world_from_left_base": np.eye(4).tolist(), "targets_file": "targets.json",
                  "audit_file": "audit.json", "provenance": {"audit_file": "audit.json"},
                  "revalidation": {"accepted": True}, "training_accepted": True,
                  "points": [{"name": "R1", "contact_success": True, "old_diagnostic": True}]}
        row = {"name": "R1", "success": True, "full_route_planned": True,
               "contact_success": False, "selected_fit": fit(10), "selected_audit_candidate_index": 1}
        result = self.search.fresh_plan_document(source, row)
        self.assertNotIn("revalidation", result)
        self.assertNotIn("training_accepted", result)
        self.assertNotIn("old_diagnostic", result["points"][0])
        self.assertEqual(result["provenance"], source["provenance"])
        self.assertFalse(result["points"][0]["contact_success"])
        row["selected_fit"]["gap_m"] = 1
        self.assertEqual(result["points"][0]["selected_fit"]["gap_m"], .005)
        with self.assertRaisesRegex(ValueError, "physics"):
            self.search.fresh_plan_document(source, {**row, "contact_success": True})

    def test_geometry_retains_detector_gap_and_separate_audited_normal(self):
        self.assertTrue(hasattr(self.search, "fit_geometry"), "audited geometry construction is missing")
        from types import SimpleNamespace
        point = SimpleNamespace(position_world=np.array([1., 2., 3.]),
                                normal_world=np.array([0., .6, .8]))
        geometry = self.search.fit_geometry(point, fit(), np.eye(4), pre_gap=.025)
        np.testing.assert_allclose(geometry["center_world"], [1., 2.003, 3.004])
        np.testing.assert_allclose(geometry["pre_world"], [1., 2., 3.025])
        np.testing.assert_allclose(geometry["free_end_world"], [1., 2., 3.015])

    def test_actual_candidate_counts_and_current_kinematics_tail(self):
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        directory = Path(__file__).parents[1]/"auscultation_expert_layout_v1/results/expert_v2/plans"
        kin = OpenArmKinematics()
        for name, expected in [("R1", 36), ("L1", 12)]:
            document = json.loads((directory/f"planned_paths_{name}.json").read_text())
            row = document["points"][0]
            audit = json.loads((directory/document["audit_file"]).read_text())
            self.assertEqual(len(self.search.matching_roll_candidates(audit, row["selected_fit"])), expected)
            tail = self.search.current_staging_tail(row, kin)
            base = np.asarray(document["world_from_left_base"])
            np.testing.assert_allclose((base@tail[0])[:3, 3], row["lower_staging_tcp_world"], atol=1e-8)

    def test_endpoint_seeds_are_finite_diverse_and_include_neighbors(self):
        self.assertTrue(hasattr(self.search, "endpoint_seeds"), "bounded redundant seed generation is missing")
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        from sim.auscultation_expert_layout_v1.posture import PostureLimits
        directory = Path(__file__).parents[1]/"auscultation_expert_layout_v1/results/expert_v2/plans"
        document = json.loads((directory/"planned_paths_R1.json").read_text())
        row = document["points"][0]
        neighbor = json.loads((directory/"planned_paths_R2.json").read_text())["points"][0]
        kin = OpenArmKinematics()
        limits = PostureLimits.from_dict(row["posture_limits"], kin)
        seeds = self.search.endpoint_seeds(kin, np.asarray(document["world_from_left_base"]), limits, row, [neighbor])
        self.assertGreaterEqual(len(seeds), 8)
        self.assertTrue(all(np.isfinite(x["q"]).all() and np.asarray(x["q"]).shape == (7,) for x in seeds))
        self.assertTrue(any(x["label"] == "neighbor_0_pre" for x in seeds))
        self.assertEqual(len([x for x in seeds if x["label"].startswith("null_")]), 4)

    def test_full_route_rejects_a_prepose_without_mesh_clearance(self):
        self.assertTrue(hasattr(self.search, "build_candidate_route"), "complete gated route construction is missing")
        from types import SimpleNamespace
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        from sim.auscultation_expert_layout_v1.posture import PostureLimits
        directory = Path(__file__).parents[1]/"auscultation_expert_layout_v1/results/expert_v2/plans"
        document = json.loads((directory/"planned_paths_R1.json").read_text())
        row = document["points"][0]
        audit = json.loads((directory/document["audit_file"]).read_text())
        kin = OpenArmKinematics()
        target = SimpleNamespace(name="R1", position_world=np.asarray(row["position_world"]),
                                 normal_world=np.asarray(row["normal_world"]))
        checker = SimpleNamespace(check=lambda *a, **k: {"clear": False, "minimum_distance_m": .0001}, samples=1)
        screened = {"qpre": row["approach_q"][0], "audit_candidate_index": row["selected_audit_candidate_index"],
                    "relative_roll_rad": 0., "relative_roll_deg": 0.}
        result = self.search.build_candidate_route(target, audit, document, screened, kin,
            np.asarray(document["world_from_left_base"]), PostureLimits.from_dict(row["posture_limits"], kin),
            checker, deadline=__import__("time").monotonic()+3.)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure"], "prepose_clearance")
        self.assertNotIn("row", result)

    def test_cli_help_does_not_import_fcl(self):
        result = subprocess.run([sys.executable, "-m", "sim.upper_pair_posture_20260924.search_cli", "--help"],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-routes-per-point", result.stdout)


if __name__ == "__main__":
    unittest.main()
