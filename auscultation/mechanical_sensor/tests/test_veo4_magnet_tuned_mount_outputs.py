from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "veo4_magnet_tuned"
CAD_SCRIPT = ROOT / "cad" / "thinklab_openarm_veo4_magnet_tuned_mount.py"


def load_veo4_module():
    if importlib.util.find_spec("cadquery") is None:
        raise unittest.SkipTest("CadQuery is required for VEO4 geometry validation")
    spec = importlib.util.spec_from_file_location("veo4_mount_under_test", CAD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["veo4_mount_under_test"] = module
    spec.loader.exec_module(module)
    return module


class VEO4MagnetTunedMountOutputsTest(unittest.TestCase):
    def test_veo4_outputs_use_disc_magnet_design(self) -> None:
        self.assertTrue(CAD_SCRIPT.exists(), "VEO4 magnet-tuned CAD generator must live in project code")
        metadata_path = OUT / "veo4_design_metadata.json"
        self.assertTrue(metadata_path.exists(), "run cad/thinklab_openarm_veo4_magnet_tuned_mount.py first")
        assembly_path = OUT / "thinklab_openarm_veo4_assembly.step"
        self.assertGreaterEqual(
            assembly_path.stat().st_mtime_ns,
            CAD_SCRIPT.stat().st_mtime_ns,
            "VEO4 assembly STEP must be regenerated after the VEO4 CAD source changes",
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["mechanism"]["type"], "veo4_magnet_tuned_clamshell")
        self.assertTrue(metadata["validation_summary"]["pass"])

        magnet = metadata["magnet_design"]
        self.assertEqual(magnet["shape"], "d5x2_cylindrical_disc")
        self.assertEqual(magnet["axis"], "Y")
        self.assertAlmostEqual(magnet["diameter_mm"], 5.0, delta=0.01)
        self.assertAlmostEqual(magnet["thickness_mm"], 2.0, delta=0.01)
        self.assertAlmostEqual(magnet["pocket_diameter_mm"], 5.5, delta=0.01)
        self.assertAlmostEqual(magnet["pocket_depth_mm"], 2.6, delta=0.01)
        self.assertEqual(magnet["center_mm"], [43.5, -16.0, -20.0])
        self.assertEqual(metadata["fastener_layout"]["front_seam_y_mm"], -45.0)
        self.assertEqual(metadata["fastener_layout"]["rear_seam_y_mm"], 43.0)

        geometry = metadata["geometry_audit"]
        self.assertTrue(geometry["pass"])
        self.assertGreaterEqual(geometry["clearances"]["magnet_hall_y_gap_mm"], 2.0)
        self.assertLessEqual(geometry["clearances"]["magnet_hall_y_gap_mm"], 5.0)
        self.assertEqual([], [check for check in geometry["checks"] if not check["pass"]])
        for check in geometry["checks"]:
            if check["name"].startswith("magnet_vs_"):
                self.assertLessEqual(check["volume_mm3"], 0.005, check["name"])
            if check["name"].startswith("clamp_screws_vs_"):
                self.assertLessEqual(check["volume_mm3"], 0.005, check["name"])

        required = [
            OUT / "thinklab_openarm_veo4_fixed_bridge_print.step",
            OUT / "thinklab_openarm_veo4_left_clamshell_print.step",
            OUT / "thinklab_openarm_veo4_right_clamshell_print.step",
            OUT / "thinklab_openarm_veo4_assembly.step",
            OUT / "thinklab_openarm_veo4_preview.step",
            OUT / "presentation" / "veo4_cad_clean.png",
            OUT / "presentation" / "veo4_actual_cad_motion.mp4",
            OUT / "veo4_geometry_audit.json",
            OUT / "veo4_design_metadata.json",
            OUT / "VEO4_MAGNET_TUNED_REPORT.md",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_veo4_runtime_magnet_audit_passes_from_project_code(self) -> None:
        veo4 = load_veo4_module()
        model = veo4.build_veo4_model()
        self.assertEqual([name for name in model["hardware"] if "spring" in name], ["spring"])
        geometry = veo4.make_veo4_geometry_audit(model)
        mechanical = veo4.make_mechanical_audit()
        material = veo4.make_material_audit()
        summary = veo4.make_validation_summary(geometry, mechanical, material)

        self.assertTrue(summary["pass"])
        self.assertEqual([], [check for check in geometry["checks"] if not check["pass"]])
        self.assertEqual([], [check for check in mechanical["checks"] if not check["pass"]])
        self.assertEqual([], [check for check in material["checks"] if not check["pass"]])
        self.assertEqual(geometry["magnet_design"]["shape"], "d5x2_cylindrical_disc")
        for check in geometry["checks"]:
            if check["name"].startswith("clamp_screws_vs_"):
                self.assertLessEqual(check["volume_mm3"], 0.005, check["name"])


if __name__ == "__main__":
    unittest.main()
