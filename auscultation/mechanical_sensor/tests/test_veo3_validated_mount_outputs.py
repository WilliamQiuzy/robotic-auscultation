from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "veo3_validated"
CAD_SCRIPT = ROOT / "cad" / "thinklab_openarm_veo3_validated_mount.py"


def load_veo3_module():
    if importlib.util.find_spec("cadquery") is None:
        raise unittest.SkipTest("CadQuery is required for VEO3 geometry validation")
    spec = importlib.util.spec_from_file_location("veo3_mount_under_test", CAD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["veo3_mount_under_test"] = module
    spec.loader.exec_module(module)
    return module


class VEO3ValidatedMountOutputsTest(unittest.TestCase):
    def test_veo3_project_code_and_outputs_exist(self) -> None:
        self.assertTrue(CAD_SCRIPT.exists(), "VEO3 CAD generator must live in project code")
        metadata_path = OUT / "veo3_design_metadata.json"
        self.assertTrue(metadata_path.exists(), "run cad/thinklab_openarm_veo3_validated_mount.py first")
        assembly_path = OUT / "thinklab_openarm_veo3_assembly.step"
        self.assertGreaterEqual(
            assembly_path.stat().st_mtime_ns,
            CAD_SCRIPT.stat().st_mtime_ns,
            "assembly STEP must be regenerated after the VEO3 CAD source changes",
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["recommended_mechanics"]["simulation_variant"], "veo3_validated_clamshell_strength_checked")
        self.assertEqual(metadata["mechanism"]["type"], "veo3_validated_two_piece_clamshell")
        self.assertTrue(metadata["validation_summary"]["pass"])

        geometry = metadata["geometry_audit"]
        self.assertTrue(geometry["pass"])
        self.assertEqual(geometry.get("spring_visual_model"), "continuous_helix")
        for check in geometry["checks"]:
            if check["name"].startswith("spring_vs_"):
                self.assertLessEqual(check["volume_mm3"], 0.005, check["name"])
        self.assertGreaterEqual(geometry["clearances"]["magnet_hall_y_gap_mm"], 2.0)
        self.assertLessEqual(geometry["clearances"].get("spring_seating_error_max_mm", 999.0), 0.4)
        self.assertGreaterEqual(geometry["clearances"].get("wire_min_radial_clearance_from_thinklabs_body_mm", -999.0), 2.0)
        self.assertGreaterEqual(geometry["dynamic_audit"]["max_compression_mm"], 9.0)

        mechanical = metadata["mechanical_audit"]
        self.assertTrue(mechanical["pass"])
        self.assertLessEqual(mechanical["max_screened_contact_force_n"], 5.5)
        self.assertAlmostEqual(mechanical["target_compression_mm"], 8.0, delta=0.05)
        self.assertGreaterEqual(mechanical["travel_margin_mm"], 1.0)
        self.assertGreaterEqual(mechanical["spring_deflection_margin_mm"], 2.0)

        material = metadata["material_audit"]
        self.assertTrue(material["pass"])
        self.assertEqual(material["print_material"], "PETG_or_PA12_baseline")
        self.assertGreaterEqual(material["minimum_safety_factor"], 4.0)

        sensor = metadata.get("sensor_layout", {})
        self.assertEqual(sensor.get("wiring_route"), "external_short_fixed_side")
        self.assertLessEqual(sensor.get("wire_bundle_span_mm", 999.0), 18.0)
        self.assertGreater(sensor.get("connector_center_mm", [-999.0, 0.0, 0.0])[0], 45.0)
        self.assertLess(sensor.get("connector_center_mm", [0.0, 999.0, 0.0])[1], 0.0)

        required = [
            OUT / "thinklab_openarm_veo3_fixed_bridge_print.step",
            OUT / "thinklab_openarm_veo3_fixed_bridge_print.stl",
            OUT / "thinklab_openarm_veo3_left_clamshell_print.step",
            OUT / "thinklab_openarm_veo3_left_clamshell_print.stl",
            OUT / "thinklab_openarm_veo3_right_clamshell_print.step",
            OUT / "thinklab_openarm_veo3_right_clamshell_print.stl",
            OUT / "thinklab_openarm_veo3_assembly.step",
            OUT / "thinklab_openarm_veo3_preview.step",
            OUT / "presentation" / "veo3_cad_clean.png",
            OUT / "presentation" / "veo3_clamshell_detail.png",
            OUT / "presentation" / "veo3_actual_cad_motion.mp4",
            OUT / "simulation" / "veo3_contact_summary.json",
            OUT / "veo3_geometry_audit.json",
            OUT / "veo3_mechanical_audit.json",
            OUT / "veo3_material_audit.json",
            OUT / "VEO3_VALIDATED_REPORT.md",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_veo3_runtime_audits_pass_from_project_code(self) -> None:
        veo3 = load_veo3_module()
        model = veo3.build_veo3_model()
        geometry = veo3.make_veo3_geometry_audit(model)
        mechanical = veo3.make_mechanical_audit()
        material = veo3.make_material_audit()
        summary = veo3.make_validation_summary(geometry, mechanical, material)

        self.assertTrue(summary["pass"])
        self.assertEqual([], [check for check in geometry["checks"] if not check["pass"]])
        self.assertEqual(geometry.get("spring_visual_model"), "continuous_helix")
        for check in geometry["checks"]:
            if check["name"].startswith("spring_vs_"):
                self.assertLessEqual(check["volume_mm3"], 0.005, check["name"])
        self.assertLessEqual(geometry["clearances"].get("spring_seating_error_max_mm", 999.0), 0.4)
        self.assertGreaterEqual(geometry["clearances"].get("wire_min_radial_clearance_from_thinklabs_body_mm", -999.0), 2.0)
        self.assertEqual([], [check for check in mechanical["checks"] if not check["pass"]])
        self.assertEqual([], [check for check in material["checks"] if not check["pass"]])


if __name__ == "__main__":
    unittest.main()
