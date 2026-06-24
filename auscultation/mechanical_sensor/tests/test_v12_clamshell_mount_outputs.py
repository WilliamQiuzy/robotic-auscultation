from __future__ import annotations

import json
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "v12_clamshell"
CAD_SCRIPT = ROOT / "cad" / "thinklab_openarm_v12_clamshell_mount.py"


def load_v12_module():
    if importlib.util.find_spec("cadquery") is None:
        raise unittest.SkipTest("CadQuery is required for V12 geometry boolean validation")
    spec = importlib.util.spec_from_file_location("v12_mount_under_test", CAD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["v12_mount_under_test"] = module
    spec.loader.exec_module(module)
    return module


class V12ClamshellMountOutputsTest(unittest.TestCase):
    def test_v12_is_two_piece_clamshell_with_two_sided_m3_fasteners(self) -> None:
        metadata_path = OUT / "v12_design_metadata.json"
        self.assertTrue(metadata_path.exists(), "run cad/thinklab_openarm_v12_clamshell_mount.py first")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["recommended_mechanics"]["simulation_variant"], "v12_stepped_clamshell_two_sided_m3")
        self.assertEqual(metadata["mechanism"]["type"], "two_piece_clamshell_stepped_profile")

        profile = metadata["thinklabs_profile"]
        self.assertEqual(profile["profile_type"], "lower_large_frustum_plus_upper_small_disc")
        self.assertTrue(profile["side_profile_is_trapezoid_like"])

        clamshell = metadata["clamshell_layout"]
        self.assertEqual(clamshell["half_count"], 2)
        self.assertTrue(clamshell["left_half_printed_separately"])
        self.assertTrue(clamshell["right_half_printed_separately"])
        self.assertEqual(clamshell["split_plane"], "X=0")
        self.assertEqual(clamshell["seam_locations"], ["front", "rear"])
        self.assertTrue(clamshell["no_living_hinge"])
        self.assertTrue(clamshell["printed_plastic_primary_structure"])
        self.assertFalse(clamshell["soft_pad_required"])
        self.assertEqual(clamshell["m3_screw_count"], 4)
        self.assertEqual(clamshell["front_m3_screw_count"], 2)
        self.assertEqual(clamshell["rear_m3_screw_count"], 2)
        self.assertTrue(clamshell["front_jack_window_drawn"])
        self.assertTrue(clamshell["rear_cable_window_drawn"])
        self.assertIn("upper_retaining_blocks_removed", clamshell)
        self.assertIn("upper_retaining_pad_count", clamshell)
        self.assertTrue(clamshell["upper_retaining_blocks_removed"])
        self.assertEqual(clamshell["upper_retaining_pad_count"], 0)
        self.assertIn("rear_m3_spring_collision_free", clamshell)
        self.assertIn("rear_m3_spring_centerline_clearance_mm", clamshell)
        self.assertIn("rear_m3_spring_required_clearance_mm", clamshell)
        self.assertTrue(clamshell["rear_m3_spring_collision_free"])
        self.assertGreaterEqual(clamshell["rear_m3_spring_required_clearance_mm"], 7.75)
        self.assertGreater(clamshell["rear_m3_spring_centerline_clearance_mm"], 8.0)
        self.assertGreater(
            clamshell["rear_m3_spring_centerline_clearance_mm"],
            clamshell["rear_m3_spring_required_clearance_mm"],
        )

        contact = metadata["contact_layout"]
        self.assertTrue(contact["only_dark_bottom_contact_face_projects_below_printed_parts"])
        self.assertGreaterEqual(contact["contact_face_projects_below_printed_parts_mm"], 12.0)

        self.assertTrue(metadata["geometry_audit"]["pass"])
        self.assertGreaterEqual(metadata["geometry_audit"]["clearances"]["magnet_hall_y_gap_mm"], 1.5)
        self.assertGreaterEqual(metadata["geometry_audit"]["clearances"]["spring_end_clearance_min_mm"], 1.8)

        print_parts = metadata["print_parts"]
        self.assertIn("left_clamshell", print_parts)
        self.assertIn("right_clamshell", print_parts)
        self.assertLess(print_parts["left_clamshell"]["xlen"], 70.0)
        self.assertLess(print_parts["right_clamshell"]["xlen"], 70.0)

        required = [
            OUT / "thinklab_openarm_v12_fixed_bridge_print.step",
            OUT / "thinklab_openarm_v12_fixed_bridge_print.stl",
            OUT / "thinklab_openarm_v12_left_clamshell_print.step",
            OUT / "thinklab_openarm_v12_left_clamshell_print.stl",
            OUT / "thinklab_openarm_v12_right_clamshell_print.step",
            OUT / "thinklab_openarm_v12_right_clamshell_print.stl",
            OUT / "thinklab_openarm_v12_clamshell_assembly.step",
            OUT / "thinklab_openarm_v12_clamshell_preview.step",
            OUT / "thinklab_openarm_v12_clamshell_preview.png",
            OUT / "presentation" / "v12_cad_clean.png",
            OUT / "presentation" / "v12_clamshell_detail.png",
            OUT / "presentation" / "v12_actual_cad_motion.mp4",
            OUT / "simulation" / "v12_contact_motion.mp4",
            OUT / "simulation" / "v12_contact_summary.json",
            OUT / "v12_geometry_audit.json",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_v12_simulation_stays_inside_force_and_travel_gates(self) -> None:
        summary_path = OUT / "simulation" / "v12_contact_summary.json"
        self.assertTrue(summary_path.exists(), "run cad/thinklab_openarm_v12_clamshell_mount.py first")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertLess(summary["peak_force_n"], 5.5)
        self.assertAlmostEqual(summary["final_force_n"], 4.0, delta=0.05)
        self.assertLess(summary["max_spring_compression_mm"], 8.5)

    def test_v12_critical_hardware_has_verified_clearance(self) -> None:
        v12 = load_v12_module()
        fixed = v12.make_fixed_bridge(v12.P)
        left, right = v12.make_clamshell_halves(v12.P)
        thinklab = v12.v11.make_stepped_thinklab_proxy(v12.P)
        hardware = v12.make_hardware_proxies(v12.P)

        audit = v12.make_geometry_audit(fixed, left, right, thinklab, hardware)
        failing_checks = [check for check in audit["checks"] if not check["pass"]]
        self.assertEqual([], failing_checks)
        self.assertGreaterEqual(audit["clearances"]["magnet_hall_y_gap_mm"], 1.5)
        self.assertGreaterEqual(audit["clearances"]["spring_end_clearance_min_mm"], 1.8)


if __name__ == "__main__":
    unittest.main()
