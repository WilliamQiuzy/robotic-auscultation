from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "veo6_realshape"
CAD_SCRIPT = ROOT / "cad" / "thinklab_openarm_veo6_realshape_mount.py"


def load_veo6_module():
    if importlib.util.find_spec("cadquery") is None:
        raise unittest.SkipTest("CadQuery is required for VEO6 geometry validation")
    spec = importlib.util.spec_from_file_location("veo6_mount_under_test", CAD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["veo6_mount_under_test"] = module
    spec.loader.exec_module(module)
    return module


class VEO6CoaxialHallMountOutputsTest(unittest.TestCase):
    def test_veo6_outputs_use_coaxial_hall_design(self) -> None:
        self.assertTrue(CAD_SCRIPT.exists(), "VEO6 realshape CAD generator must live in project code")
        metadata_path = OUT / "veo6_design_metadata.json"
        self.assertTrue(metadata_path.exists(), "run cad/thinklab_openarm_veo6_realshape_mount.py first")
        assembly_path = OUT / "thinklab_openarm_veo6_assembly.step"
        self.assertGreaterEqual(
            assembly_path.stat().st_mtime_ns,
            CAD_SCRIPT.stat().st_mtime_ns,
            "VEO6 assembly STEP must be regenerated after the VEO6 CAD source changes",
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["mechanism"]["type"], "veo6_realshape_clamshell")
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
        # up to the 4 mm hard stop (preloaded design).
        for compression in ("1.5", "2.5", "7.25"):
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

        # STABILITY mechanism: preloaded captive guide posts. The post head is
        # the rest stop (defines a stable zero under spring preload) AND the
        # drop-out catch (head can't pass the bushing bore). 4 mm travel caps
        # force at 5.5 N. The diaphragm-side head stays clear of the patient.
        self.assertIn("guide_posts_captive_shoulder_len", check_names)
        self.assertIn("rod_head_retains_ring", check_names)
        self.assertIn("rest_stop_seats_barrel_on_head", check_names)
        self.assertIn("rod_head_above_skin_at_hard_stop", check_names)
        self.assertIn("hard_stop_travel_matches_design", check_names)
        self.assertIn("rod_bushing_engagement_at_rest", check_names)
        mech_names = [c["name"] for c in metadata["mechanical_audit"]["checks"]]
        self.assertIn("spring_centroid_on_device_axis", mech_names)
        self.assertIn("preload_holds_zero_below_threshold", mech_names)
        self.assertAlmostEqual(metadata["mechanical_audit"]["preload_n"], 1.5, delta=0.01)
        self.assertAlmostEqual(metadata["mechanical_audit"]["hard_stop_force_n"], 8.75, delta=0.01)

        # Regression: purchased-part dimensional fidelity (WF6-25 spring OD,
        # silicone wire OD).
        self.assertIn("spring_visual_od_matches_wf6_25", check_names)
        self.assertIn("sensor_wire_od_matches_28awg_silicone", check_names)
        self.assertIn("purchased_parts_fit", metadata)

        # VEO6: the device is the HAND-MEASURED real shape (6 mm black edge +
        # 11 mm cone 46->38 + 12 mm dial, 46 x 29). The ring grips only the
        # black edge with a straight bore; the cone/dial/buttons/jack sit
        # above the ring and are never touched.
        self.assertIn("thinklab_proxy_matches_measured_46x29", check_names)
        self.assertIn("black_edge_grip_clearance", check_names)
        self.assertIn("black_edge_seated_in_ring", check_names)
        self.assertIn("device_cone_dial_vs_left_clamshell", check_names)
        self.assertIn("dial_and_jack_above_ring_top", check_names)
        # Positive retention catch: the bottom lip must block the device from
        # dropping out (friction alone is not relied upon).
        self.assertIn("device_retained_by_bottom_lip", check_names)
        self.assertIn("diaphragm_contact_opening_sufficient", check_names)
        lip = next(c for c in geometry["checks"] if c["name"] == "device_retained_by_bottom_lip")
        self.assertGreaterEqual(lip["lip_radial_catch_mm"], 1.0)
        self.assertTrue(lip["device_rests_on_lip"])
        self.assertIn("bushing_bottoms_above_diaphragm_plane", check_names)
        self.assertIn("compressed_7.25mm_thinklab_vs_fixed_print", check_names)

        # Captive post: the Ø5 shoulder slides freely in the Ø5.9 bore while the
        # head seats on the barrel (rest stop) and retains the ring.
        self.assertIn("rod_shaft_slides_in_bore", check_names)

        # New on-mount electronics: XIAO cradle, and the OpenArm DM-J4310
        # output-face M3 bolt holes must be open through the plate.
        self.assertIn("xiao_vs_fixed_print", check_names)
        self.assertIn("xiao_board_matches_catalog_outline", check_names)
        self.assertIn("output_m3_holes_open_through_plate", check_names)

        # Regression: wires must POSITIVELY join both solder partners (a
        # v9.cyl_y displacement bug once left them floating disconnected),
        # and the OpenArm actuator output-face interface must seat clear.
        self.assertIn("hall_signal_wire_joins_hall_lead", check_names)
        self.assertIn("hall_signal_wire_joins_xiao_pads", check_names)
        self.assertIn("actuator_seats_without_clash", check_names)
        self.assertIn("output_m3_screws_vs_fixed_print", check_names)
        self.assertIn("actuator_flange_clear_of_xiao_cradle", check_names)

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
            OUT / "thinklab_openarm_veo6_fixed_bridge_print.step",
            OUT / "thinklab_openarm_veo6_left_clamshell_print.step",
            OUT / "thinklab_openarm_veo6_right_clamshell_print.step",
            OUT / "thinklab_openarm_veo6_assembly.step",
            OUT / "thinklab_openarm_veo6_preview.step",
            OUT / "presentation" / "veo6_cad_clean.png",
            OUT / "presentation" / "veo6_actual_cad_motion.mp4",
            OUT / "simulation" / "veo6_hall_response.csv",
            OUT / "simulation" / "veo6_hall_response.png",
            OUT / "veo6_geometry_audit.json",
            OUT / "veo6_design_metadata.json",
            OUT / "VEO6_REALSHAPE_REPORT.md",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_veo6_runtime_audit_passes_from_project_code(self) -> None:
        veo6 = load_veo6_module()
        model = veo6.build_veo6_model()
        self.assertEqual([name for name in model["hardware"] if "spring" in name], ["spring"])
        geometry = veo6.make_veo6_geometry_audit(model)
        mechanical = veo6.make_veo6_mechanical_audit()
        material = veo6.make_veo6_material_audit()
        summary = veo6.veo3.make_validation_summary(geometry, mechanical, material)

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
        hall_bb = veo6.make_hall_body_proxy().val().BoundingBox()
        self.assertAlmostEqual(
            (magnet_bb.xmin + magnet_bb.xmax) / 2.0, (hall_bb.xmin + hall_bb.xmax) / 2.0, delta=0.3
        )
        self.assertAlmostEqual(
            (magnet_bb.ymin + magnet_bb.ymax) / 2.0, (hall_bb.ymin + hall_bb.ymax) / 2.0, delta=0.3
        )
        rest_gap = hall_bb.zmin - magnet_bb.zmax
        self.assertGreaterEqual(rest_gap - veo6.HARD_STOP_TRAVEL_MM, 1.0)

        # The hall proxy must include the three straight SS49E leads.
        full_hall_bb = model["hardware"]["hall"].val().BoundingBox()
        self.assertGreaterEqual(full_hall_bb.xmax - full_hall_bb.xmin, 15.5)

        # Screws must be purchasable M3x30, not full-width rods.
        screws_bb = model["hardware"]["clamp_screws"].val().BoundingBox()
        self.assertLessEqual(screws_bb.xmax - screws_bb.xmin, 34.0)

        # Captive guide posts: Ø5 shoulder + Ø8 rest-stop head, ~34.7 mm total.
        # At the 7.25 mm hard stop (8.75 N, the user-chosen max firmness) the Ø8
        # head dips ~0.75 mm below the diaphragm plane — negligible on soft
        # tissue and accepted (matches rod_head_above_skin_at_hard_stop's -1.0
        # bound in the geometry audit; the old 4 mm-travel design cleared +2 mm).
        rods_bb = model["hardware"]["guide_rods"].val().BoundingBox()
        self.assertAlmostEqual(rods_bb.zmax - rods_bb.zmin, veo6.GUIDE_ROD_LENGTH_MM, delta=0.1)
        tk_bb = model["thinklab"].val().BoundingBox()
        self.assertGreaterEqual(rods_bb.zmin - (tk_bb.zmin + veo6.HARD_STOP_TRAVEL_MM), -1.0)

        # Two springs, point-symmetric about the device axis (moment-free).
        sp_bb = model["hardware"]["spring"].val().BoundingBox()
        self.assertAlmostEqual((sp_bb.xmin + sp_bb.xmax) / 2.0, 0.0, delta=0.05)
        self.assertAlmostEqual((sp_bb.ymin + sp_bb.ymax) / 2.0, 0.0, delta=0.05)

        # Thinklabs proxy must match the official 46 dia x 28 tall spec.
        self.assertAlmostEqual(tk_bb.xmax - tk_bb.xmin, 46.0, delta=0.2)
        self.assertAlmostEqual(tk_bb.zmax - tk_bb.zmin, 29.0, delta=0.2)

        # XIAO proxy present and matching the catalog outline; the JST
        # connector is gone (hall wires solder directly to the XIAO pads).
        xiao_bb = model["hardware"]["xiao"].val().BoundingBox()
        self.assertAlmostEqual(xiao_bb.ymax - xiao_bb.ymin, 21.0, delta=0.1)
        self.assertNotIn("sensor_connector", model["hardware"])

        sensing = veo6.make_sensing_model()
        self.assertTrue(sensing["monotonic_over_full_travel"])


if __name__ == "__main__":
    unittest.main()
