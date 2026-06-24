from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "v11_stepped_thinklabs"


class V11SteppedThinklabsMountOutputsTest(unittest.TestCase):
    def test_v11_uses_stepped_thinklabs_proxy_and_clamp(self) -> None:
        metadata_path = OUT / "v11_design_metadata.json"
        self.assertTrue(metadata_path.exists(), "run cad/thinklab_openarm_v11_stepped_thinklabs_mount.py first")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["recommended_mechanics"]["simulation_variant"], "v11_stepped_thinklabs_profile_clamp")
        self.assertEqual(metadata["mechanism"]["type"], "stepped_profile_lower_rim_upper_retainer")

        profile = metadata["thinklabs_profile"]
        self.assertEqual(profile["profile_type"], "lower_large_frustum_plus_upper_small_disc")
        self.assertGreater(profile["lower_bottom_d_mm"], profile["upper_disc_d_mm"])
        self.assertGreater(profile["lower_top_d_mm"], profile["upper_disc_d_mm"])
        self.assertTrue(profile["front_jack_pair_drawn"])
        self.assertTrue(profile["top_control_face_drawn"])
        self.assertTrue(profile["side_profile_is_trapezoid_like"])

        clamp = metadata["stepped_clamp_layout"]
        self.assertTrue(clamp["lower_rim_support_drawn"])
        self.assertTrue(clamp["upper_disc_retaining_pads_drawn"])
        self.assertEqual(clamp["upper_retaining_pad_count"], 3)
        self.assertTrue(clamp["front_jack_window_drawn"])
        self.assertTrue(clamp["rear_cable_window_drawn"])
        self.assertGreaterEqual(clamp["lower_rim_clearance_d_mm"], profile["lower_bottom_d_mm"] + 0.8)
        self.assertGreaterEqual(clamp["upper_disc_clearance_d_mm"], profile["upper_disc_d_mm"] + 0.6)
        self.assertEqual(clamp["m3_screw_count"], 2)

        contact = metadata["contact_layout"]
        self.assertTrue(contact["only_dark_bottom_contact_face_projects_below_printed_parts"])
        self.assertGreaterEqual(contact["contact_face_projects_below_printed_parts_mm"], 12.0)

        required = [
            OUT / "thinklab_openarm_v11_fixed_bridge_print.step",
            OUT / "thinklab_openarm_v11_fixed_bridge_print.stl",
            OUT / "thinklab_openarm_v11_stepped_cradle_print.step",
            OUT / "thinklab_openarm_v11_stepped_cradle_print.stl",
            OUT / "thinklab_openarm_v11_stepped_assembly.step",
            OUT / "thinklab_openarm_v11_stepped_preview.step",
            OUT / "thinklab_openarm_v11_stepped_preview.png",
            OUT / "presentation" / "v11_cad_clean.png",
            OUT / "presentation" / "v11_profile_clamp_detail.png",
            OUT / "presentation" / "v11_actual_cad_motion.mp4",
            OUT / "simulation" / "v11_contact_motion.mp4",
            OUT / "simulation" / "v11_contact_summary.json",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_v11_simulation_stays_inside_force_and_travel_gates(self) -> None:
        summary_path = OUT / "simulation" / "v11_contact_summary.json"
        self.assertTrue(summary_path.exists(), "run cad/thinklab_openarm_v11_stepped_thinklabs_mount.py first")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertLess(summary["peak_force_n"], 5.5)
        self.assertAlmostEqual(summary["final_force_n"], 4.0, delta=0.05)
        self.assertLess(summary["max_spring_compression_mm"], 8.5)


if __name__ == "__main__":
    unittest.main()
