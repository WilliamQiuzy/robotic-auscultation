from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "v10_detailed"


class V10DetailedMountOutputsTest(unittest.TestCase):
    def test_v10_details_are_explicit_and_artifacts_are_generated(self) -> None:
        metadata_path = OUT / "v10_design_metadata.json"
        self.assertTrue(metadata_path.exists(), "run cad/thinklab_openarm_v10_detailed_clamp_mount.py first")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["recommended_mechanics"]["simulation_variant"], "v10_detailed_two_rail_visible_interfaces")
        self.assertEqual(metadata["mechanism"]["type"], "detailed_split_clamp_two_rail_spring")
        self.assertTrue(metadata["mechanism"]["blue_cradle_is_main_structural_part"])

        clamp = metadata["clamp_layout"]
        self.assertEqual(clamp["m3_screw_count"], 2)
        self.assertEqual(clamp["screw_axis"], "X")
        self.assertEqual(clamp["screw_hole_d_mm"], 3.5)
        self.assertTrue(clamp["open_split_gap_no_center_bridge"])
        self.assertTrue(clamp["nut_pockets_drawn"])
        self.assertGreaterEqual(clamp["ear_block_size_mm"][2], 12.0)
        self.assertGreater(clamp["gap_width_mm"], 8.0)

        sensor = metadata["sensor_layout"]
        self.assertTrue(sensor["magnet_drawn_in_blue_cradle"])
        self.assertTrue(sensor["magnet_pocket_drawn"])
        self.assertTrue(sensor["hall_drawn_on_fixed_side"])
        self.assertTrue(sensor["short_wire_proxy_to_connector_drawn"])
        self.assertLessEqual(sensor["nominal_hall_magnet_gap_mm"], 6.0)

        spring = metadata["spring_layout"]
        self.assertEqual(spring["spring_count"], 1)
        self.assertTrue(spring["fixed_top_seat_drawn"])
        self.assertTrue(spring["moving_bottom_seat_drawn"])
        self.assertTrue(spring["visible_coil_proxy_drawn"])
        self.assertAlmostEqual(spring["seat_face_gap_at_rest_mm"], 25.0, delta=0.05)

        interface = metadata["robot_interface_layout"]
        self.assertTrue(interface["openarm_flange_proxy_drawn"])
        self.assertEqual(interface["mount_slot_count"], 4)
        self.assertEqual(interface["bolt_proxy_count"], 4)
        self.assertTrue(interface["interface_is_on_top_fixed_bridge"])

        contact = metadata["contact_layout"]
        self.assertTrue(contact["dark_proxy_is_stethoscope"])
        self.assertTrue(contact["only_dark_bottom_contact_face_projects_below_printed_parts"])
        self.assertGreaterEqual(contact["contact_face_projects_below_printed_parts_mm"], 12.0)

        print_parts = metadata["print_parts"]
        self.assertLess(print_parts["fixed_bridge"]["zlen"], 20.0)
        self.assertLess(print_parts["blue_cradle"]["zlen"], 24.0)

        required = [
            OUT / "thinklab_openarm_v10_fixed_bridge_print.step",
            OUT / "thinklab_openarm_v10_fixed_bridge_print.stl",
            OUT / "thinklab_openarm_v10_blue_cradle_print.step",
            OUT / "thinklab_openarm_v10_blue_cradle_print.stl",
            OUT / "thinklab_openarm_v10_detailed_assembly.step",
            OUT / "thinklab_openarm_v10_detailed_preview.step",
            OUT / "thinklab_openarm_v10_detailed_preview.png",
            OUT / "presentation" / "v10_cad_clean.png",
            OUT / "presentation" / "v10_detail_layout.png",
            OUT / "presentation" / "v10_actual_cad_motion.mp4",
            OUT / "simulation" / "v10_contact_motion.mp4",
            OUT / "simulation" / "v10_contact_summary.json",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_v10_simulation_stays_inside_force_and_travel_gates(self) -> None:
        summary_path = OUT / "simulation" / "v10_contact_summary.json"
        self.assertTrue(summary_path.exists(), "run cad/thinklab_openarm_v10_detailed_clamp_mount.py first")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertLess(summary["peak_force_n"], 5.5)
        self.assertAlmostEqual(summary["final_force_n"], 4.0, delta=0.05)
        self.assertLess(summary["max_spring_compression_mm"], 8.5)


if __name__ == "__main__":
    unittest.main()
