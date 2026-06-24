from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "v8b_compact"


class V8BCompactMountOutputsTest(unittest.TestCase):
    def test_v8b_metadata_and_artifacts_are_generated(self) -> None:
        metadata_path = OUT / "v8b_design_metadata.json"
        self.assertTrue(metadata_path.exists(), "run cad/thinklab_openarm_v8b_compact_mount.py first")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["recommended_mechanics"]["simulation_variant"], "v8b_compact_mini_spring")
        self.assertEqual(metadata["params"]["guide_rod_d"], 0.0)
        self.assertEqual(metadata["params"]["travel"], 11.0)
        self.assertEqual(metadata["params"]["spring_free_length"], 25.0)
        self.assertEqual(metadata["params"]["spring_od"], 6.0)
        self.assertEqual(metadata["recommended_mechanics"]["max_screened_contact_force_n"], 5.5)
        self.assertGreaterEqual(metadata["contact_layout"]["contact_face_projects_below_printed_parts_mm"], 12.0)
        self.assertLess(metadata["print_parts"]["fixed_frame"]["xlen"], 100.0)
        self.assertLess(metadata["print_parts"]["fixed_frame"]["zlen"], 55.0)
        self.assertLess(metadata["print_parts"]["floating_carriage"]["xlen"], 100.0)

        fit = metadata["hardware_fit_audit"]
        self.assertTrue(fit["spring"]["fits"])
        self.assertAlmostEqual(fit["spring"]["seat_face_gap_at_rest_mm"], 25.0, delta=0.05)
        self.assertGreater(fit["spring"]["side_clearance_from_thinklabs_body_mm"], 5.0)
        self.assertGreater(fit["spring"]["side_clearance_from_printed_cup_od_mm"], 1.5)
        self.assertTrue(fit["magnet"]["fits"])
        self.assertGreaterEqual(fit["magnet"]["pocket_clearance_box_mm"][0], 5.5)
        self.assertGreaterEqual(fit["magnet"]["pocket_clearance_box_mm"][1], 2.2)
        self.assertGreaterEqual(fit["magnet"]["pocket_clearance_box_mm"][2], 5.5)
        self.assertTrue(fit["hall_sensor"]["fits"])
        self.assertTrue(fit["electronics"]["xiao_rp2040_external"])
        self.assertTrue(fit["electronics"]["external_wiring_drawn_as_non_print_proxy"])

        wiring = metadata["wiring_layout"]
        self.assertTrue(wiring["non_print_proxy"])
        self.assertIn("red wire", wiring["hall_vcc_wire"])
        self.assertIn("yellow wire", wiring["hall_signal_wire"])
        self.assertIn("black wire", wiring["hall_gnd_wire"])
        self.assertIn("ThinkLabs", wiring["thinklabs_audio_cable"])

        presentation = metadata["presentation_assets"]
        self.assertTrue(presentation["clean_cad_png"].endswith("presentation/v8b_cad_clean.png"))
        self.assertTrue(presentation["simple_schematic_png"].endswith("presentation/v8b_simple_schematic.png"))
        self.assertTrue(presentation["actual_cad_motion_mp4"].endswith("presentation/v8b_actual_cad_motion.mp4"))

        required = [
            OUT / "thinklab_openarm_v8b_fixed_frame_print.step",
            OUT / "thinklab_openarm_v8b_floating_carriage_print.step",
            OUT / "thinklab_openarm_v8b_clearance_checked_assembly.step",
            OUT / "thinklab_openarm_v8b_clearance_checked_preview.step",
            OUT / "thinklab_openarm_v8b_compact_mount_preview.png",
            OUT / "simulation" / "v8b_contact_motion.mp4",
            OUT / "simulation" / "v8b_contact_summary.json",
            OUT / "presentation" / "v8b_cad_clean.png",
            OUT / "presentation" / "v8b_simple_schematic.png",
            OUT / "presentation" / "v8b_actual_cad_motion.mp4",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_v8b_simulation_stays_inside_force_and_travel_gates(self) -> None:
        summary_path = OUT / "simulation" / "v8b_contact_summary.json"
        self.assertTrue(summary_path.exists(), "run cad/thinklab_openarm_v8b_compact_mount.py first")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertLess(summary["peak_force_n"], 5.5)
        self.assertAlmostEqual(summary["final_force_n"], 4.0, delta=0.05)
        self.assertLess(summary["max_spring_compression_mm"], 8.5)


if __name__ == "__main__":
    unittest.main()
