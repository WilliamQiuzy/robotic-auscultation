from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "v9_simplified"


class V9SimplifiedMountOutputsTest(unittest.TestCase):
    def test_v9_metadata_and_artifacts_are_generated(self) -> None:
        metadata_path = OUT / "v9_design_metadata.json"
        self.assertTrue(metadata_path.exists(), "run cad/thinklab_openarm_v9_simplified_rail_mount.py first")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["recommended_mechanics"]["simulation_variant"], "v9_simplified_two_rail_spring_clamp")
        self.assertEqual(metadata["params"]["guide_rod_count"], 2)
        self.assertEqual(metadata["params"]["spring_count"], 1)
        self.assertEqual(metadata["params"]["clamp_screw_count"], 2)
        self.assertEqual(metadata["params"]["spring_free_length"], 25.0)
        self.assertEqual(metadata["params"]["spring_od"], 6.0)
        self.assertFalse(metadata["electronics_layout"]["xiao_board_drawn_in_cad"])
        self.assertTrue(metadata["electronics_layout"]["hall_sensor_drawn_in_cad"])
        self.assertTrue(metadata["electronics_layout"]["magnet_drawn_in_cad"])

        mechanism = metadata["mechanism"]
        self.assertEqual(mechanism["type"], "minimal_two_rail_spring_clamp")
        self.assertEqual(mechanism["fixed_part_role"], "small_robot_bridge_only")
        self.assertTrue(mechanism["blue_cradle_is_main_structural_part"])
        self.assertTrue(mechanism["two_side_clamp_screws"])

        contact = metadata["contact_layout"]
        self.assertTrue(contact["dark_proxy_is_stethoscope"])
        self.assertTrue(contact["only_dark_bottom_contact_face_projects_below_printed_parts"])
        self.assertGreaterEqual(contact["contact_face_projects_below_printed_parts_mm"], 12.0)
        self.assertGreaterEqual(contact["cradle_inner_diameter_mm"], 62.6)

        print_parts = metadata["print_parts"]
        self.assertLess(print_parts["fixed_bridge"]["xlen"], 100.0)
        self.assertLess(print_parts["fixed_bridge"]["ylen"], 66.0)
        self.assertLess(print_parts["fixed_bridge"]["zlen"], 18.0)
        self.assertLess(print_parts["blue_cradle"]["xlen"], 104.0)
        self.assertLess(print_parts["blue_cradle"]["ylen"], 104.0)
        self.assertLess(print_parts["blue_cradle"]["zlen"], 22.0)

        clearance = metadata["clearance_audit"]
        self.assertFalse(clearance["fixed_print_intrudes_into_thinklabs_body_envelope"])
        self.assertFalse(clearance["guide_rods_intrude_into_thinklabs_body_envelope"])
        self.assertFalse(clearance["spring_intrudes_into_thinklabs_body_envelope"])
        self.assertGreater(clearance["guide_rod_side_clearance_from_body_mm"], 5.0)
        self.assertGreater(clearance["spring_side_clearance_from_body_mm"], 5.0)
        self.assertGreater(clearance["clamp_boss_side_clearance_from_body_mm"], 2.0)

        required = [
            OUT / "thinklab_openarm_v9_fixed_bridge_print.step",
            OUT / "thinklab_openarm_v9_fixed_bridge_print.stl",
            OUT / "thinklab_openarm_v9_blue_cradle_print.step",
            OUT / "thinklab_openarm_v9_blue_cradle_print.stl",
            OUT / "thinklab_openarm_v9_simplified_assembly.step",
            OUT / "thinklab_openarm_v9_simplified_preview.step",
            OUT / "thinklab_openarm_v9_simplified_preview.png",
            OUT / "simulation" / "v9_contact_motion.mp4",
            OUT / "simulation" / "v9_contact_summary.json",
            OUT / "presentation" / "v9_cad_clean.png",
            OUT / "presentation" / "v9_simple_schematic.png",
            OUT / "presentation" / "v9_actual_cad_motion.mp4",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_v9_simulation_stays_inside_force_and_travel_gates(self) -> None:
        summary_path = OUT / "simulation" / "v9_contact_summary.json"
        self.assertTrue(summary_path.exists(), "run cad/thinklab_openarm_v9_simplified_rail_mount.py first")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertLess(summary["peak_force_n"], 5.5)
        self.assertAlmostEqual(summary["final_force_n"], 4.0, delta=0.05)
        self.assertLess(summary["max_spring_compression_mm"], 8.5)


if __name__ == "__main__":
    unittest.main()
