"""VEO3 validated ThinkLabs One + OpenArm passive force mount.

VEO3 is a stricter successor to V12. It keeps the two-piece clamshell idea, but
the generated package is driven by three explicit audits:

- geometry: static and max-compression boolean intersection checks
- mechanics: spring force/travel gates for gentle auscultation contact
- material: first-order printed-part stress estimates with safety factors

This is still a parametric prototype until the real ThinkLabs and robot flange
are measured, but the output is no longer just visual CAD.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path


V12_PATH = Path(__file__).with_name("thinklab_openarm_v12_clamshell_mount.py")
spec = importlib.util.spec_from_file_location("v12_mount", V12_PATH)
v12 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["v12_mount"] = v12
spec.loader.exec_module(v12)

cq = v12.cq
exporters = v12.exporters
v9 = v12.v9


P = v9.SimplifiedRailMountParams(
    cup_clearance=0.9,
    cup_wall=5.0,
    cup_height=18.0,
    lip_height=3.0,
    lip_opening_d=51.0,
    cable_window_w=15.0,
    clamp_gap_w=4.0,
    guide_rod_d=5.0,
    guide_rod_clearance_d=5.9,
    guide_rod_x=47.0,
    spring_axis_x=20.0,
    spring_axis_y=58.0,
    top_bridge_x=108.0,
    top_bridge_y=44.0,
    top_bridge_t=6.0,
    guide_bushing_h=20.0,
)

RECOMMENDED_MECHANICS = {
    "simulation_variant": "veo3_validated_clamshell_strength_checked",
    "spring_rate_n_per_mm": 0.5,
    "linear_damping_n_s_per_m": 3.2,
    "target_contact_force_n": 4.0,
    "max_screened_contact_force_n": 4.6,
}

PRINT_MATERIAL = {
    "name": "PETG_or_PA12_baseline",
    "allowable_stress_mpa": 18.0,
    "allowable_bearing_mpa": 10.0,
    "minimum_required_safety_factor": 4.0,
}

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated" / "veo3_validated"
SIM_DIR = OUT_DIR / "simulation"
PRESENTATION_DIR = OUT_DIR / "presentation"
SENSOR_WIRE_NAMES = ("hall_vcc_wire", "hall_signal_wire", "hall_gnd_wire")


def configure_v12_backend() -> None:
    v12.P = P
    v12.RECOMMENDED_MECHANICS = RECOMMENDED_MECHANICS
    v12.OUT_DIR = OUT_DIR
    v12.SIM_DIR = SIM_DIR
    v12.PRESENTATION_DIR = PRESENTATION_DIR
    v12.sensor_layout = veo3_sensor_layout


def veo3_sensor_layout(p: v9.SimplifiedRailMountParams) -> dict[str, tuple[float, float, float]]:
    return {
        "magnet": (p.guide_rod_x - 3.5, -16.0, p.carriage_z + 10.0),
        "hall": (p.guide_rod_x - 3.0, -10.0, -15.0),
        "hall_backplate": (p.guide_rod_x - 3.0, -7.5, -10.5),
        "sensor_connector": (53.5, -14.8, -15.6),
    }


def _cylinder_between(radius: float, start: tuple[float, float, float], end: tuple[float, float, float]) -> cq.Solid:
    start_vec = cq.Vector(*start)
    end_vec = cq.Vector(*end)
    direction = end_vec - start_vec
    return cq.Solid.makeCylinder(radius, direction.Length, start_vec, direction.normalized())


def make_veo3_visible_spring(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    wire_r = 0.30
    visual_outer_r = p.spring_od / 2.0 - 0.40
    helix_r = visual_outer_r - wire_r
    bottom_z = p.spring_moving_face_z + wire_r + 0.39
    top_z = p.spring_fixed_face_z - wire_r
    turns = 7.75
    phase_rad = math.pi / 2.0
    segment_count = 124
    solids = []
    for i in range(segment_count):
        t0 = i / segment_count
        t1 = (i + 1) / segment_count
        a0 = phase_rad + 2.0 * math.pi * turns * t0
        a1 = phase_rad + 2.0 * math.pi * turns * t1
        start = (
            p.spring_axis_x + helix_r * math.cos(a0),
            p.spring_axis_y + helix_r * math.sin(a0),
            bottom_z + (top_z - bottom_z) * t0,
        )
        end = (
            p.spring_axis_x + helix_r * math.cos(a1),
            p.spring_axis_y + helix_r * math.sin(a1),
            bottom_z + (top_z - bottom_z) * t1,
        )
        solids.append(_cylinder_between(wire_r, start, end))
    return cq.Workplane("XY").add(cq.Compound.makeCompound(solids))


def make_veo3_fixed_bridge(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    bridge = v12.make_fixed_bridge(p)
    bridge = bridge.union(v9.box(15.0, 1.0, 2.0, (50.5, -7.8, -9.0)))
    bridge = bridge.union(v9.box(3.0, 1.0, 8.0, (58.0, -7.8, -12.0)))
    return bridge


def build_veo3_model() -> dict:
    configure_v12_backend()
    fixed = make_veo3_fixed_bridge(P)
    left, right = v12.make_clamshell_halves(P)
    thinklab = v12.v11.make_stepped_thinklab_proxy(P)
    hardware = v12.make_hardware_proxies(P)
    hardware["spring"] = make_veo3_visible_spring(P)
    return {
        "fixed": fixed,
        "left": left,
        "right": right,
        "thinklab": thinklab,
        "hardware": hardware,
    }


def _add_collision_check(checks: list[dict], name: str, a: cq.Workplane, b: cq.Workplane, max_volume_mm3: float = 0.05) -> None:
    volume = v12._intersection_volume_mm3(a, b)
    checks.append(
        {
            "name": name,
            "type": "intersection_volume",
            "volume_mm3": round(volume, 6),
            "max_volume_mm3": max_volume_mm3,
            "pass": volume <= max_volume_mm3,
        }
    )


def _bbox_min_radial_distance_xy(shape: cq.Workplane) -> float:
    bb = shape.val().BoundingBox()
    if bb.xmin <= 0.0 <= bb.xmax:
        dx = 0.0
    else:
        dx = min(abs(bb.xmin), abs(bb.xmax))
    if bb.ymin <= 0.0 <= bb.ymax:
        dy = 0.0
    else:
        dy = min(abs(bb.ymin), abs(bb.ymax))
    return math.hypot(dx, dy)


def _combined_bbox(shapes: list[cq.Workplane]) -> cq.BoundBox:
    return cq.Compound.makeCompound([shape.val() for shape in shapes]).BoundingBox()


def make_veo3_geometry_audit(model: dict) -> dict:
    fixed = model["fixed"]
    left = model["left"]
    right = model["right"]
    hardware = model["hardware"]
    checks: list[dict] = []

    spring_max_collision_mm3 = 0.005
    _add_collision_check(checks, "spring_vs_fixed_print", hardware["spring"], fixed, spring_max_collision_mm3)
    _add_collision_check(checks, "spring_vs_left_clamshell", hardware["spring"], left, spring_max_collision_mm3)
    _add_collision_check(checks, "spring_vs_right_clamshell", hardware["spring"], right, spring_max_collision_mm3)
    _add_collision_check(checks, "spring_vs_clamp_screws", hardware["spring"], hardware["clamp_screws"], spring_max_collision_mm3)
    _add_collision_check(checks, "magnet_vs_hall_sensor", hardware["magnet"], hardware["hall"])
    _add_collision_check(checks, "magnet_vs_left_clamshell", hardware["magnet"], left)
    _add_collision_check(checks, "magnet_vs_right_clamshell", hardware["magnet"], right)
    _add_collision_check(checks, "hall_sensor_vs_fixed_print", hardware["hall"], fixed)
    _add_collision_check(checks, "hall_sensor_vs_left_clamshell", hardware["hall"], left)
    _add_collision_check(checks, "hall_sensor_vs_right_clamshell", hardware["hall"], right)
    _add_collision_check(checks, "sensor_connector_vs_fixed_print", hardware["sensor_connector"], fixed)
    _add_collision_check(checks, "sensor_connector_vs_thinklab_proxy", hardware["sensor_connector"], model["thinklab"])
    _add_collision_check(checks, "sensor_connector_vs_left_clamshell", hardware["sensor_connector"], left)
    _add_collision_check(checks, "sensor_connector_vs_right_clamshell", hardware["sensor_connector"], right)
    for wire_name in SENSOR_WIRE_NAMES:
        _add_collision_check(checks, f"{wire_name}_vs_fixed_print", hardware[wire_name], fixed, max_volume_mm3=0.02)
        _add_collision_check(checks, f"{wire_name}_vs_thinklab_proxy", hardware[wire_name], model["thinklab"], max_volume_mm3=0.02)
        _add_collision_check(checks, f"{wire_name}_vs_left_clamshell", hardware[wire_name], left, max_volume_mm3=0.02)
        _add_collision_check(checks, f"{wire_name}_vs_right_clamshell", hardware[wire_name], right, max_volume_mm3=0.02)

    max_compression_mm = 9.0
    for name, shape in [
        ("left_clamshell", left),
        ("right_clamshell", right),
        ("magnet", hardware["magnet"]),
        ("clamp_screws", hardware["clamp_screws"]),
    ]:
        _add_collision_check(
            checks,
            f"max_compression_{name}_vs_fixed_print",
            shape.translate((0, 0, max_compression_mm)),
            fixed,
        )

    spring_bb = hardware["spring"].val().BoundingBox()
    spring_bottom_seating_error = abs(spring_bb.zmin - P.spring_moving_face_z)
    spring_top_seating_error = abs(P.spring_fixed_face_z - spring_bb.zmax)
    magnet_hall_y_gap = v12._bbox_axis_gap_mm(hardware["magnet"], hardware["hall"], "y")
    thinklabs_body_r = v12.THINKLABS_PROFILE["lower_bottom_d_mm"] / 2.0
    wire_radial_clearances = [
        _bbox_min_radial_distance_xy(hardware[name]) - thinklabs_body_r for name in SENSOR_WIRE_NAMES
    ]
    wire_min_radial_clearance = min(wire_radial_clearances)
    clearances = {
        "magnet_hall_y_gap_mm": round(magnet_hall_y_gap, 3),
        "spring_bottom_seating_error_mm": round(spring_bottom_seating_error, 3),
        "spring_top_seating_error_mm": round(spring_top_seating_error, 3),
        "spring_seating_error_max_mm": round(max(spring_bottom_seating_error, spring_top_seating_error), 3),
        "wire_min_radial_clearance_from_thinklabs_body_mm": round(wire_min_radial_clearance, 3),
    }
    checks.append(
        {
            "name": "magnet_hall_air_gap",
            "type": "bbox_axis_gap",
            "axis": "y",
            "gap_mm": clearances["magnet_hall_y_gap_mm"],
            "min_gap_mm": 2.0,
            "pass": magnet_hall_y_gap >= 2.0,
        }
    )
    checks.append(
        {
            "name": "spring_seated_in_sockets",
            "type": "bbox_z_seating",
            "max_error_mm": clearances["spring_seating_error_max_mm"],
            "max_allowed_error_mm": 0.4,
            "pass": clearances["spring_seating_error_max_mm"] <= 0.4,
        }
    )
    checks.append(
        {
            "name": "wires_route_outside_thinklabs_body",
            "type": "radial_bbox_clearance",
            "clearance_mm": clearances["wire_min_radial_clearance_from_thinklabs_body_mm"],
            "min_clearance_mm": 2.0,
            "pass": wire_min_radial_clearance >= 2.0,
        }
    )
    return {
        "units": "mm",
        "method": "CadQuery boolean intersection volumes plus max-compression dynamic clearance checks",
        "spring_visual_model": "continuous_helix",
        "checks": checks,
        "clearances": clearances,
        "dynamic_audit": {
            "max_compression_mm": max_compression_mm,
            "moving_parts_translated_up_relative_to_fixed": True,
        },
        "pass": all(check["pass"] for check in checks),
    }


def _beam_bending_stress_mpa(force_n: float, lever_mm: float, width_mm: float, thickness_mm: float) -> float:
    return 6.0 * force_n * lever_mm / (width_mm * thickness_mm**2)


def _safety_factor(allowable_mpa: float, stress_mpa: float) -> float:
    return allowable_mpa / max(stress_mpa, 1e-9)


def make_mechanical_audit() -> dict:
    spring_rate = RECOMMENDED_MECHANICS["spring_rate_n_per_mm"]
    target_force = RECOMMENDED_MECHANICS["target_contact_force_n"]
    max_force = RECOMMENDED_MECHANICS["max_screened_contact_force_n"]
    target_compression = target_force / spring_rate
    max_compression = max_force / spring_rate
    travel_margin = P.travel - target_compression
    spring_deflection_margin = P.spring_allowable_deflection - max_compression
    checks = [
        {
            "name": "target_force_requires_expected_compression",
            "type": "spring_force",
            "target_force_n": target_force,
            "spring_rate_n_per_mm": spring_rate,
            "compression_mm": round(target_compression, 3),
            "expected_compression_mm": 8.0,
            "pass": abs(target_compression - 8.0) <= 0.05,
        },
        {
            "name": "screened_force_below_patient_contact_limit",
            "type": "force_gate",
            "force_n": max_force,
            "limit_n": 5.5,
            "pass": max_force <= 5.5,
        },
        {
            "name": "target_travel_margin",
            "type": "travel_margin",
            "margin_mm": round(travel_margin, 3),
            "min_margin_mm": 1.0,
            "pass": travel_margin >= 1.0,
        },
        {
            "name": "spring_allowable_deflection_margin",
            "type": "spring_margin",
            "margin_mm": round(spring_deflection_margin, 3),
            "min_margin_mm": 2.0,
            "pass": spring_deflection_margin >= 2.0,
        },
    ]
    return {
        "target_contact_force_n": target_force,
        "max_screened_contact_force_n": max_force,
        "spring_rate_n_per_mm": spring_rate,
        "target_compression_mm": round(target_compression, 3),
        "max_screened_compression_mm": round(max_compression, 3),
        "travel_limit_mm": P.travel,
        "travel_margin_mm": round(travel_margin, 3),
        "spring_allowable_deflection_mm": P.spring_allowable_deflection,
        "spring_deflection_margin_mm": round(spring_deflection_margin, 3),
        "checks": checks,
        "pass": all(check["pass"] for check in checks),
    }


def make_material_audit() -> dict:
    allowable = PRINT_MATERIAL["allowable_stress_mpa"]
    max_force = RECOMMENDED_MECHANICS["max_screened_contact_force_n"]
    lower_inner_r = (v12.THINKLABS_PROFILE["lower_bottom_d_mm"] + 1.0) / 2.0
    outer_r = lower_inner_r + P.cup_wall
    fixed_tab_lever = P.spring_axis_y - P.top_bridge_y / 2.0
    moving_bridge_lever = P.spring_axis_y - outer_r

    fixed_tab_stress = _beam_bending_stress_mpa(max_force, fixed_tab_lever, 24.0, P.top_bridge_t)
    moving_bridge_stress = _beam_bending_stress_mpa(max_force, moving_bridge_lever, 18.0, 4.0)
    lug_tension_stress = 25.0 / (14.0 * 6.0)
    guide_bushing_bearing_stress = max_force / (P.guide_rod_d * P.guide_bushing_h * 2.0)

    checks = []
    for name, stress, allowable_mpa in [
        ("fixed_spring_tab_bending", fixed_tab_stress, allowable),
        ("moving_spring_bridge_bending", moving_bridge_stress, allowable),
        ("m3_lug_tension_proxy", lug_tension_stress, allowable),
        ("guide_bushing_bearing_proxy", guide_bushing_bearing_stress, PRINT_MATERIAL["allowable_bearing_mpa"]),
    ]:
        sf = _safety_factor(allowable_mpa, stress)
        checks.append(
            {
                "name": name,
                "stress_mpa": round(stress, 4),
                "allowable_mpa": allowable_mpa,
                "safety_factor": round(sf, 3),
                "min_safety_factor": PRINT_MATERIAL["minimum_required_safety_factor"],
                "pass": sf >= PRINT_MATERIAL["minimum_required_safety_factor"],
            }
        )
    return {
        "print_material": PRINT_MATERIAL["name"],
        "assumption": "First-order static stress screen for PETG/PA12-class printed parts; physical prototype still needs load testing.",
        "minimum_safety_factor": min(check["safety_factor"] for check in checks),
        "checks": checks,
        "pass": all(check["pass"] for check in checks),
    }


def make_validation_summary(geometry: dict, mechanical: dict, material: dict) -> dict:
    return {
        "pass": geometry["pass"] and mechanical["pass"] and material["pass"],
        "geometry_pass": geometry["pass"],
        "mechanical_pass": mechanical["pass"],
        "material_pass": material["pass"],
        "required_followup": [
            "Measure real ThinkLabs One diameters before printing final fit.",
            "Bench-test spring calibration with the actual Hall sensor and magnet.",
            "Do a physical load test on the printed bridge before patient-facing use.",
        ],
    }


def write_sim_outputs(metadata: dict) -> dict:
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    helpers = v12.v10.load_isaac_helpers()
    rows = helpers.simulate_step_response(
        metadata,
        duration=3.0,
        dt=1.0 / 120.0,
        approach_speed_mm_s=28.0,
        tissue_stiffness_n_per_mm=0.8,
        tissue_damping_n_s_per_m=12.0,
    )
    trace_path = SIM_DIR / "veo3_contact_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = helpers.write_contact_plot(rows, metadata, SIM_DIR)
    if plot_path is not None:
        final_plot = SIM_DIR / "veo3_contact_response.png"
        if final_plot.exists():
            final_plot.unlink()
        plot_path.rename(final_plot)
    video_path = helpers.write_motion_video(rows, metadata, SIM_DIR, fps=30, max_frames=150)
    if video_path is not None:
        final_video = SIM_DIR / "veo3_contact_motion.mp4"
        if final_video.exists():
            final_video.unlink()
        video_path.rename(final_video)
    summary = {
        "variant": RECOMMENDED_MECHANICS["simulation_variant"],
        "output_csv": str(trace_path),
        "output_plot": str(SIM_DIR / "veo3_contact_response.png"),
        "output_motion_video": str(SIM_DIR / "veo3_contact_motion.mp4"),
        "peak_force_n": max(row["contact_force_n"] for row in rows),
        "final_force_n": rows[-1]["contact_force_n"],
        "max_spring_compression_mm": max(row["spring_compression_mm"] for row in rows),
    }
    (SIM_DIR / "veo3_contact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_sim_rows() -> list[dict[str, float]]:
    with (SIM_DIR / "veo3_contact_trace.csv").open(newline="", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def export_part(part: cq.Workplane, stem: str) -> None:
    exporters.export(part, str(OUT_DIR / f"{stem}.step"))
    exporters.export(part, str(OUT_DIR / f"{stem}.stl"), tolerance=0.14, angularTolerance=0.12)


def make_sensor_layout_metadata(hardware: dict[str, cq.Workplane]) -> dict:
    layout = veo3_sensor_layout(P)
    wire_bb = _combined_bbox([hardware[name] for name in SENSOR_WIRE_NAMES])
    return {
        "wiring_route": "external_short_fixed_side",
        "magnet_center_mm": [round(value, 3) for value in layout["magnet"]],
        "hall_center_mm": [round(value, 3) for value in layout["hall"]],
        "connector_center_mm": [round(value, 3) for value in layout["sensor_connector"]],
        "wire_bundle_span_mm": round(
            max(wire_bb.xmax - wire_bb.xmin, wire_bb.ymax - wire_bb.ymin, wire_bb.zmax - wire_bb.zmin),
            3,
        ),
        "wire_bundle_bbox_mm": {
            "xmin": round(wire_bb.xmin, 3),
            "xmax": round(wire_bb.xmax, 3),
            "ymin": round(wire_bb.ymin, 3),
            "ymax": round(wire_bb.ymax, 3),
            "zmin": round(wire_bb.zmin, 3),
            "zmax": round(wire_bb.zmax, 3),
        },
    }


def make_metadata(model: dict, geometry: dict, mechanical: dict, material: dict, validation: dict, sim_summary=None, presentation_assets=None) -> dict:
    metadata = {
        "units": "mm",
        "params": asdict(P),
        "recommended_mechanics": RECOMMENDED_MECHANICS,
        "mechanism": {
            "type": "veo3_validated_two_piece_clamshell",
            "basis": "V12 clamshell with stricter geometry, force, and material audits",
            "force_feedback": "Hall sensor reads moving magnet displacement; force is spring calibrated.",
            "spring_visual_model": "continuous_helix",
        },
        "sensor_layout": make_sensor_layout_metadata(model["hardware"]),
        "print_parts": {
            "fixed_bridge": v9.bbox_dict(model["fixed"]),
            "left_clamshell": v9.bbox_dict(model["left"]),
            "right_clamshell": v9.bbox_dict(model["right"]),
        },
        "geometry_audit": geometry,
        "mechanical_audit": mechanical,
        "material_audit": material,
        "validation_summary": validation,
    }
    if sim_summary is not None:
        metadata["simulation_summary"] = sim_summary
    if presentation_assets is not None:
        metadata["presentation_assets"] = presentation_assets
    return metadata


def render_presentation_assets(model: dict, rows: list[dict[str, float]]) -> dict:
    configure_v12_backend()
    clean = v12.render_clean_cad_png(model["fixed"], model["left"], model["right"], model["thinklab"], model["hardware"])
    detail = v12.render_clamshell_detail_png(model["fixed"], model["left"], model["right"], model["thinklab"], model["hardware"])
    video = v12.write_actual_cad_motion_video(model["fixed"], model["left"], model["right"], model["thinklab"], model["hardware"], rows)
    renamed = {}
    for old_path, new_name in [
        (clean, "veo3_cad_clean.png"),
        (detail, "veo3_clamshell_detail.png"),
        (video, "veo3_actual_cad_motion.mp4" if video is not None else None),
    ]:
        if old_path is None or new_name is None:
            continue
        new_path = PRESENTATION_DIR / new_name
        if new_path.exists():
            new_path.unlink()
        old_path.rename(new_path)
        renamed[new_name] = str(new_path)
    return renamed


def write_report(metadata: dict) -> None:
    geometry = metadata["geometry_audit"]
    mechanical = metadata["mechanical_audit"]
    material = metadata["material_audit"]
    report = [
        "# VEO3 Validated Clamshell Mount",
        "",
        "VEO3 is generated from project code and includes geometry, mechanical, and material audits.",
        "",
        "## Validation",
        "",
        f"- Overall pass: `{metadata['validation_summary']['pass']}`",
        f"- Geometry pass: `{geometry['pass']}`",
        f"- Mechanical pass: `{mechanical['pass']}`",
        f"- Material pass: `{material['pass']}`",
        f"- Magnet/Hall Y air gap: `{geometry['clearances']['magnet_hall_y_gap_mm']:.3f} mm`",
        f"- Spring seating error: `{geometry['clearances']['spring_seating_error_max_mm']:.3f} mm`",
        f"- Wire radial clearance from ThinkLabs body: `{geometry['clearances']['wire_min_radial_clearance_from_thinklabs_body_mm']:.3f} mm`",
        f"- Dynamic geometry audit compression: `{geometry['dynamic_audit']['max_compression_mm']:.3f} mm`",
        f"- Target force: `{mechanical['target_contact_force_n']:.3f} N`",
        f"- Max screened force: `{mechanical['max_screened_contact_force_n']:.3f} N`",
        f"- Minimum material safety factor: `{material['minimum_safety_factor']:.3f}`",
        "",
        "## Limits",
        "",
        "- This is a CAD and first-order analytical validation package, not a substitute for physical load testing.",
        "- Final print dimensions should be updated after measuring the real ThinkLabs One and OpenArm flange.",
    ]
    (OUT_DIR / "VEO3_VALIDATED_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    model = build_veo3_model()
    geometry = make_veo3_geometry_audit(model)
    mechanical = make_mechanical_audit()
    material = make_material_audit()
    validation = make_validation_summary(geometry, mechanical, material)

    export_part(model["fixed"], "thinklab_openarm_veo3_fixed_bridge_print")
    export_part(model["left"], "thinklab_openarm_veo3_left_clamshell_print")
    export_part(model["right"], "thinklab_openarm_veo3_right_clamshell_print")
    exporters.export(model["thinklab"], str(OUT_DIR / "thinklab_veo3_stepped_profile_proxy.step"))

    assembly = cq.Assembly(name="thinklab_openarm_veo3_validated_mount")
    assembly.add(model["fixed"], name="printed_fixed_bridge", color=cq.Color(0.83, 0.80, 0.70))
    assembly.add(model["left"], name="printed_left_clamshell_half", color=cq.Color(0.05, 0.34, 0.75))
    assembly.add(model["right"], name="printed_right_clamshell_half", color=cq.Color(0.05, 0.48, 0.84))
    assembly.add(model["thinklab"], name="thinklab_one_stepped_proxy_not_printed", color=cq.Color(0.05, 0.05, 0.05, 0.40))
    for name, shape in model["hardware"].items():
        color = cq.Color(0.02, 0.02, 0.02)
        if name == "robot_flange":
            color = cq.Color(0.54, 0.56, 0.58, 0.55)
        elif name == "guide_rods":
            color = cq.Color(0.72, 0.72, 0.72)
        elif name == "spring":
            color = cq.Color(0.2, 0.8, 0.25)
        elif name == "magnet" or name == "hall_vcc_wire":
            color = cq.Color(0.84, 0.16, 0.16)
        elif name == "hall":
            color = cq.Color(0.42, 0.10, 0.60)
        elif name == "sensor_connector":
            color = cq.Color(0.06, 0.62, 0.54)
        elif name == "hall_signal_wire":
            color = cq.Color(0.95, 0.76, 0.19)
        assembly.add(shape, name=f"{name}_proxy_not_printed", color=color)
    assembly.save(str(OUT_DIR / "thinklab_openarm_veo3_assembly.step"), exportType="STEP")

    preview = cq.Compound.makeCompound(
        [model["fixed"].val(), model["left"].val(), model["right"].val(), model["thinklab"].val()]
        + [shape.val() for shape in model["hardware"].values()]
    )
    exporters.export(cq.Workplane("XY").add(preview), str(OUT_DIR / "thinklab_openarm_veo3_preview.step"))

    (OUT_DIR / "veo3_geometry_audit.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    (OUT_DIR / "veo3_mechanical_audit.json").write_text(json.dumps(mechanical, indent=2), encoding="utf-8")
    (OUT_DIR / "veo3_material_audit.json").write_text(json.dumps(material, indent=2), encoding="utf-8")

    metadata = make_metadata(model, geometry, mechanical, material, validation)
    sim_summary = write_sim_outputs(metadata)
    rows = read_sim_rows()
    presentation_assets = render_presentation_assets(model, rows)
    metadata = make_metadata(model, geometry, mechanical, material, validation, sim_summary, presentation_assets)
    (OUT_DIR / "veo3_design_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(metadata)

    print(
        json.dumps(
            {
                "validation_summary": validation,
                "mechanical": {
                    "target_force_n": mechanical["target_contact_force_n"],
                    "max_screened_force_n": mechanical["max_screened_contact_force_n"],
                },
                "material_min_sf": material["minimum_safety_factor"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
