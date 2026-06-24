from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "veo5_coaxial_hall"
CAD_SCRIPT = ROOT / "cad" / "thinklab_openarm_veo5_coaxial_hall_mount.py"


def load_veo5_module():
    if importlib.util.find_spec("cadquery") is None:
        raise unittest.SkipTest("CadQuery is required for VEO5 geometry validation")
    spec = importlib.util.spec_from_file_location("veo5_mount_under_test", CAD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["veo5_mount_under_test"] = module
    spec.loader.exec_module(module)
    return module


class VEO5CoaxialHallMountOutputsTest(unittest.TestCase):
    def test_veo5_outputs_use_coaxial_hall_design(self) -> None:
        self.assertTrue(CAD_SCRIPT.exists(), "VEO5 coaxial-Hall CAD generator must live in project code")
        metadata_path = OUT / "veo5_design_metadata.json"
        self.assertTrue(metadata_path.exists(), "run cad/thinklab_openarm_veo5_coaxial_hall_mount.py first")
        assembly_path = OUT / "thinklab_openarm_veo5_assembly.step"
        self.assertGreaterEqual(
            assembly_path.stat().st_mtime_ns,
            CAD_SCRIPT.stat().st_mtime_ns,
            "VEO5 assembly STEP must be regenerated after the VEO5 CAD source changes",
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["mechanism"]["type"], "veo5_coaxial_hall_clamshell")
        self.assertTrue(metadata["validation_summary"]["pass"])

        magnet = metadata["magnet_design"]
        self.assertEqual(magnet["shape"], "d5x2_cylindrical_disc")
        self.assertIn("Z", magnet["axis"])
        self.assertAlmostEqual(magnet["diameter_mm"], 5.0, delta=0.01)
        self.assertAlmostEqual(magnet["thickness_mm"], 2.0, delta=0.01)
        self.assertAlmostEqual(magnet["pocket_diameter_mm"], 5.5, delta=0.01)
        self.assertAlmostEqual(magnet["pocket_depth_mm"], 2.6, delta=0.01)

        geometry = metadata["geometry_audit"]
        self.assertTrue(geometry["pass"])
        self.assertEqual([], [check for check in geometry["checks"] if not check["pass"]])
        check_names = {check["name"] for check in geometry["checks"]}

        # Regression: VEO4 defect 1 — printed halves shared a 3 mm overlap slab.
        self.assertIn("left_clamshell_vs_right_clamshell", check_names)
        self.assertGreaterEqual(geometry["clearances"]["clamshell_split_air_gap_mm"], 2.5)

        # Regression: VEO4 defect 2 — clamshells printed as loose pieces.
        for part in ("fixed_bridge", "left_clamshell", "right_clamshell"):
            self.assertIn(f"{part}_is_single_connected_solid", check_names)

        # Regression: spring/ring interpenetration — the compressed spring must
        # be audited against the moved clamshell ring at several travel points,
        # including the coil-bind hard stop.
        for compression in ("2.0", "4.0", "4.6", "5.5"):
            self.assertIn(f"compressed_{compression}mm_spring_vs_left", check_names)
            self.assertIn(f"compressed_{compression}mm_spring_vs_right", check_names)

        # Regression: 56 mm screw rods skewered the spring bridge slab with the
        # nut buried inside the 3 mm plate.
        self.assertIn("clamp_screws_clear_of_spring_bridge_slab", check_names)
        self.assertIn("clamp_screws_match_m3x30_bom", check_names)
        self.assertIn("nut_pocket_inside_right_lug", check_names)
        fasteners = metadata["fastener_layout"]
        self.assertIn("M3 x 30", fasteners["clamp_screw_spec"])
        self.assertLessEqual(fasteners["nut_pocket_center_x_mm"], 15.0, "nut must sit inside the right lug")

        # Regression: Hall proxy must model the purchased SS49E's straight leads.
        self.assertIn("hall_body_plus_leads_length", check_names)
        self.assertIn("hall_leads_clear_leg_channel", check_names)

        # Regression: 47 mm guide rods reached the patient's skin at the 4 N
        # working compression. Rods must be purchasable 5x30 dowels whose tips
        # clear the skin plane through the whole stroke.
        self.assertIn("guide_rods_match_5x25_dowel", check_names)
        self.assertIn("rod_tips_above_skin_at_4.6mm", check_names)
        self.assertIn("rod_tips_above_skin_at_5.5mm", check_names)
        self.assertIn("hard_stop_travel_is_5p5mm", check_names)
        self.assertIn("spring_centroid_on_device_axis", [c["name"] for c in metadata["mechanical_audit"]["checks"]])
        self.assertIn("rod_bushing_engagement_at_rest", check_names)

        # Regression: purchased-part dimensional fidelity (WF6-25 spring OD,
        # silicone wire OD).
        self.assertIn("spring_visual_od_matches_wf6_25", check_names)
        self.assertIn("sensor_wire_od_matches_28awg_silicone", check_names)
        self.assertIn("purchased_parts_fit", metadata)

        # Regression: the Thinklabs proxy must match the OFFICIAL 46x28 spec
        # (the old 66/48 guess oversized the whole clamp by ~20 mm), the top
        # lip must retain the shoulder, and the device must clear the fixed
        # plate at every audited compression including coil bind.
        self.assertIn("thinklab_proxy_matches_official_46x28", check_names)
        self.assertIn("cone_grip_radial_clearance", check_names)
        self.assertIn("dial_rotation_clearance_left", check_names)
        self.assertIn("audio_plug_engaged_in_socket", check_names)
        self.assertIn("audio_plug_vs_left_clamshell", check_names)
        self.assertIn("compressed_5.5mm_audio_plug_vs_fixed_print", check_names)
        self.assertIn("bushing_bottoms_above_diaphragm_plane", check_names)
        self.assertIn("compressed_5.5mm_thinklab_vs_fixed_print", check_names)

        # Regression: rods must run clear through the bushing bores (the
        # bridge blocks were never bored in any prior version).
        self.assertIn("guide_rods_vs_left_clamshell", check_names)
        self.assertIn("guide_rods_vs_right_clamshell", check_names)

        # New on-mount electronics: XIAO cradle and WidowX AI mount pattern.
        self.assertIn("xiao_vs_fixed_print", check_names)
        self.assertIn("xiao_board_matches_catalog_outline", check_names)
        self.assertIn("wxai_mount_holes_open_through_plate", check_names)

        # Regression: wires must POSITIVELY join both solder partners (a
        # v9.cyl_y displacement bug once left them floating disconnected),
        # and the round X-Series horn interface must be present and clear.
        self.assertIn("hall_signal_wire_joins_hall_lead", check_names)
        self.assertIn("hall_signal_wire_joins_xiao_pads", check_names)
        self.assertIn("horn_relief_and_m2_holes_open", check_names)
        self.assertIn("horn_and_servo_seat_without_clash", check_names)
        self.assertIn("wrist_servo_clear_of_xiao_cradle", check_names)

        # Regression: VEO4 defect 3 — non-monotonic Hall readout.
        self.assertIn("hall_output_monotonic_over_full_travel", check_names)
        sensing = metadata["sensing_model"]
        self.assertEqual(sensing["arrangement"], "coaxial_z_axial_gap_closing")
        self.assertTrue(sensing["monotonic_over_full_travel"])
        self.assertGreaterEqual(sensing["operating_span_v"], 0.25)
        # Air gap must outlast full coil-bind travel so magnet never hits the Hall.
        self.assertGreaterEqual(geometry["clearances"]["magnet_hall_gap_at_hard_stop_mm"], 1.0)
        self.assertGreaterEqual(geometry["clearances"]["magnet_hall_rest_gap_mm"], 7.0)

        required = [
            OUT / "thinklab_openarm_veo5_fixed_bridge_print.step",
            OUT / "thinklab_openarm_veo5_left_clamshell_print.step",
            OUT / "thinklab_openarm_veo5_right_clamshell_print.step",
            OUT / "thinklab_openarm_veo5_assembly.step",
            OUT / "thinklab_openarm_veo5_preview.step",
            OUT / "presentation" / "veo5_cad_clean.png",
            OUT / "presentation" / "veo5_actual_cad_motion.mp4",
            OUT / "simulation" / "veo5_hall_response.csv",
            OUT / "simulation" / "veo5_hall_response.png",
            OUT / "veo5_geometry_audit.json",
            OUT / "veo5_design_metadata.json",
            OUT / "VEO5_COAXIAL_HALL_REPORT.md",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_veo5_runtime_audit_passes_from_project_code(self) -> None:
        veo5 = load_veo5_module()
        model = veo5.build_veo5_model()
        self.assertEqual([name for name in model["hardware"] if "spring" in name], ["spring"])
        geometry = veo5.make_veo5_geometry_audit(model)
        mechanical = veo5.make_veo5_mechanical_audit()
        material = veo5.make_veo5_material_audit()
        summary = veo5.veo3.make_validation_summary(geometry, mechanical, material)

        self.assertTrue(summary["pass"])
        self.assertEqual([], [check for check in geometry["checks"] if not check["pass"]])
        self.assertEqual([], [check for check in mechanical["checks"] if not check["pass"]])
        self.assertEqual([], [check for check in material["checks"] if not check["pass"]])

        # Printed halves must be disjoint single solids with a real air gap.
        left_bb = model["left"].val().BoundingBox()
        right_bb = model["right"].val().BoundingBox()
        self.assertGreaterEqual(right_bb.xmin - left_bb.xmax, 2.5)
        for part in ("fixed", "left", "right"):
            self.assertEqual(len(model[part].val().Solids()), 1, f"{part} must print as one piece")

        # Coaxial sensing: magnet and Hall body share an XY axis (the leads
        # extend off-axis, so alignment is judged on the body only).
        magnet_bb = model["hardware"]["magnet"].val().BoundingBox()
        hall_bb = veo5.make_hall_body_proxy().val().BoundingBox()
        self.assertAlmostEqual(
            (magnet_bb.xmin + magnet_bb.xmax) / 2.0, (hall_bb.xmin + hall_bb.xmax) / 2.0, delta=0.3
        )
        self.assertAlmostEqual(
            (magnet_bb.ymin + magnet_bb.ymax) / 2.0, (hall_bb.ymin + hall_bb.ymax) / 2.0, delta=0.3
        )
        rest_gap = hall_bb.zmin - magnet_bb.zmax
        self.assertGreaterEqual(rest_gap - veo5.HARD_STOP_TRAVEL_MM, 1.0)

        # The hall proxy must include the three straight SS49E leads.
        full_hall_bb = model["hardware"]["hall"].val().BoundingBox()
        self.assertGreaterEqual(full_hall_bb.xmax - full_hall_bb.xmin, 15.5)

        # Screws must be purchasable M3x30, not full-width rods.
        screws_bb = model["hardware"]["clamp_screws"].val().BoundingBox()
        self.assertLessEqual(screws_bb.xmax - screws_bb.xmin, 34.0)

        # Guide rods must be standard 25 mm dowels well clear of the patient
        # plane even at the coil-bind hard stop.
        rods_bb = model["hardware"]["guide_rods"].val().BoundingBox()
        self.assertAlmostEqual(rods_bb.zmax - rods_bb.zmin, 25.0, delta=0.1)
        tk_bb = model["thinklab"].val().BoundingBox()
        self.assertGreaterEqual(rods_bb.zmin - (tk_bb.zmin + veo5.HARD_STOP_TRAVEL_MM), 5.0)

        # Two springs, point-symmetric about the device axis (moment-free).
        sp_bb = model["hardware"]["spring"].val().BoundingBox()
        self.assertAlmostEqual((sp_bb.xmin + sp_bb.xmax) / 2.0, 0.0, delta=0.05)
        self.assertAlmostEqual((sp_bb.ymin + sp_bb.ymax) / 2.0, 0.0, delta=0.05)

        # Thinklabs proxy must match the official 46 dia x 28 tall spec.
        self.assertAlmostEqual(tk_bb.xmax - tk_bb.xmin, 46.0, delta=0.1)
        self.assertAlmostEqual(tk_bb.zmax - tk_bb.zmin, 28.0, delta=0.1)

        # XIAO proxy present and matching the catalog outline; the JST
        # connector is gone (hall wires solder directly to the XIAO pads).
        xiao_bb = model["hardware"]["xiao"].val().BoundingBox()
        self.assertAlmostEqual(xiao_bb.ymax - xiao_bb.ymin, 21.0, delta=0.1)
        self.assertNotIn("sensor_connector", model["hardware"])

        sensing = veo5.make_sensing_model()
        self.assertTrue(sensing["monotonic_over_full_travel"])


if __name__ == "__main__":
    unittest.main()
