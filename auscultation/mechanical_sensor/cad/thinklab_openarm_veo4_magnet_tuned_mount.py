"""VEO4 magnet-tuned ThinkLabs One + OpenArm passive force mount.

VEO4 keeps the VEO3 validated clamshell architecture, but replaces the
rectangular magnet proxy with the purchased D5x2 mm cylindrical disc magnet and
adds explicit magnet-pocket validation.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path


VEO3_PATH = Path(__file__).with_name("thinklab_openarm_veo3_validated_mount.py")
spec = importlib.util.spec_from_file_location("veo3_mount", VEO3_PATH)
veo3 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["veo3_mount"] = veo3
spec.loader.exec_module(veo3)

cq = veo3.cq
exporters = veo3.exporters
v9 = veo3.v9
v12 = veo3.v12

P = veo3.P
RECOMMENDED_MECHANICS = {
    **veo3.RECOMMENDED_MECHANICS,
    "simulation_variant": "veo4_magnet_tuned_d5x2_disc_pocket",
}

MAGNET_DIAMETER_MM = 5.0
MAGNET_THICKNESS_MM = 2.0
MAGNET_POCKET_DIAMETER_MM = 5.5
MAGNET_POCKET_DEPTH_MM = 2.6
NUT_CENTER_X_MM = 23.0

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated" / "veo4_magnet_tuned"
SIM_DIR = OUT_DIR / "simulation"
PRESENTATION_DIR = OUT_DIR / "presentation"


def ensure_ffmpeg_on_path() -> None:
    if shutil.which("ffmpeg") is not None:
        return
    homebrew_ffmpeg = Path("/opt/homebrew/Cellar/ffmpeg")
    if not homebrew_ffmpeg.exists():
        return
    candidates = sorted(homebrew_ffmpeg.glob("*/bin"), reverse=True)
    if candidates:
        os.environ["PATH"] = f"{candidates[0]}:{os.environ.get('PATH', '')}"


def veo4_sensor_layout(p: v9.SimplifiedRailMountParams) -> dict[str, tuple[float, float, float]]:
    return {
        "magnet": (p.guide_rod_x - 3.5, -16.0, p.carriage_z + 10.0),
        "hall": (p.guide_rod_x - 3.0, -10.0, -15.0),
        "hall_backplate": (p.guide_rod_x - 3.0, -7.5, -10.5),
        "sensor_connector": (53.5, -14.8, -15.6),
    }


def configure_backends() -> None:
    v12.P = P
    v12.RECOMMENDED_MECHANICS = RECOMMENDED_MECHANICS
    v12.OUT_DIR = OUT_DIR
    v12.SIM_DIR = SIM_DIR
    v12.PRESENTATION_DIR = PRESENTATION_DIR
    v12.sensor_layout = veo4_sensor_layout


def cyl_y_centered(radius: float, height: float, center: tuple[float, float, float]) -> cq.Workplane:
    x, y, z = center
    solid = cq.Solid.makeCylinder(radius, height, cq.Vector(x, y - height / 2.0, z), cq.Vector(0, 1, 0))
    return cq.Workplane("XY").add(solid)


def clamshell_seam_ys(p: v9.SimplifiedRailMountParams) -> tuple[float, float]:
    prof = v12.THINKLABS_PROFILE
    lower_inner_r = (prof["lower_bottom_d_mm"] + 1.0) / 2.0
    outer_r = lower_inner_r + p.cup_wall
    return (-(outer_r + 6.5), outer_r + 4.5)


def _base_stepped_ring_with_disc_magnet(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    prof = v12.THINKLABS_PROFILE
    lower_clearance_d = prof["lower_bottom_d_mm"] + 1.0
    lower_inner_r = lower_clearance_d / 2.0
    outer_r = lower_inner_r + p.cup_wall
    zc = p.carriage_z

    lower_band = v9.cyl_z(outer_r, p.cup_height, (0, 0, zc)).cut(
        v9.cyl_z(lower_inner_r, p.cup_height + 3.0, (0, 0, zc))
    )
    lip_z = zc - p.cup_height / 2.0 + p.lip_height / 2.0
    lower_lip = v9.cyl_z(outer_r, p.lip_height, (0, 0, lip_z)).cut(
        v9.cyl_z(p.lip_opening_d / 2.0, p.lip_height + 3.0, (0, 0, lip_z))
    )
    ring = lower_band.union(lower_lip)

    ring = ring.cut(v9.box(18.0, 11.0, 8.0, (0, -outer_r - 1.0, zc - 1.0)))
    ring = ring.cut(v9.box(p.cable_window_w, outer_r + 12.0, 7.0, (0, outer_r, zc + 5.0)))

    for sx in (-p.guide_rod_x, p.guide_rod_x):
        bushing = v9.cyl_z(p.guide_bushing_od / 2.0, p.guide_bushing_h, (sx, p.guide_rod_y, zc))
        bushing = bushing.cut(v9.cyl_z(p.guide_rod_clearance_d / 2.0, p.guide_bushing_h + 3.0, (sx, p.guide_rod_y, zc)))
        bridge = v9.box(10.0, 10.0, 10.0, (sx * 0.94, p.guide_rod_y, zc))
        ring = ring.union(bushing).union(bridge)

    spring_bridge_y = (outer_r + p.spring_axis_y) / 2.0
    spring_bridge_len = max(5.0, p.spring_axis_y - outer_r + 4.0)
    ring = ring.union(v9.box(18.0, spring_bridge_len, 4.0, (p.spring_axis_x, spring_bridge_y, p.spring_moving_face_z - 1.5)))
    ring = ring.union(v9.upward_spring_socket(p))

    magnet_x, magnet_y, magnet_z = veo4_sensor_layout(p)["magnet"]
    holder = v9.box(10.0, 5.2, 7.0, (magnet_x, magnet_y + 1.8, magnet_z))
    pocket = cyl_y_centered(MAGNET_POCKET_DIAMETER_MM / 2.0, MAGNET_POCKET_DEPTH_MM, (magnet_x, magnet_y, magnet_z))
    ring = ring.union(holder).cut(pocket)
    return ring


def make_clamshell_halves(p: v9.SimplifiedRailMountParams) -> tuple[cq.Workplane, cq.Workplane]:
    prof = v12.THINKLABS_PROFILE
    lower_inner_r = (prof["lower_bottom_d_mm"] + 1.0) / 2.0
    outer_r = lower_inner_r + p.cup_wall
    zc = p.carriage_z
    base = _base_stepped_ring_with_disc_magnet(p)

    split_gap_w = 3.0
    cutter_right = v9.box(120.0, 130.0, 80.0, (60.0 + split_gap_w / 2.0, 0, zc))
    cutter_left = v9.box(120.0, 130.0, 80.0, (-60.0 - split_gap_w / 2.0, 0, zc))
    left = base.cut(cutter_right)
    right = base.cut(cutter_left)

    seam_ys = list(clamshell_seam_ys(p))
    for y in seam_ys:
        for sx, target in [(-10.0, "left"), (10.0, "right")]:
            lug = v9.box(14.0, 12.0, 14.0, (sx, y, zc))
            if target == "left":
                left = left.union(lug)
            else:
                right = right.union(lug)
        for dz in (-4.0, 4.0):
            hole = v9.cyl_x(p.m3_clearance_d / 2.0, 56.0, (0, y, zc + dz))
            left = left.cut(hole)
            right = right.cut(hole)
            nut = v12.v10.hex_prism_x(3.55, 4.0, (NUT_CENTER_X_MM, y, zc + dz))
            right = right.cut(nut)
    return left, right


def make_clamp_screws_proxy(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    zc = p.carriage_z
    clamp_screws = None
    for y in clamshell_seam_ys(p):
        for dz in (-4.0, 4.0):
            screw = v9.cyl_x(1.35, 56.0, (0, y, zc + dz))
            screw = screw.union(v9.cyl_x(2.4, 3.0, (-28.5, y, zc + dz)))
            screw = screw.union(v12.v10.hex_prism_x(3.1, 2.4, (NUT_CENTER_X_MM, y, zc + dz)))
            clamp_screws = screw if clamp_screws is None else clamp_screws.union(screw)
    assert clamp_screws is not None
    return clamp_screws


def make_hardware_proxies(p: v9.SimplifiedRailMountParams) -> dict[str, cq.Workplane]:
    hardware = v12.make_hardware_proxies(p)
    magnet_x, magnet_y, magnet_z = veo4_sensor_layout(p)["magnet"]
    hardware["clamp_screws"] = make_clamp_screws_proxy(p)
    hardware["spring"] = veo3.make_veo3_visible_spring(p)
    hardware["magnet"] = cyl_y_centered(MAGNET_DIAMETER_MM / 2.0, MAGNET_THICKNESS_MM, (magnet_x, magnet_y, magnet_z))
    return hardware


def build_veo4_model() -> dict:
    configure_backends()
    fixed = veo3.make_veo3_fixed_bridge(P)
    left, right = make_clamshell_halves(P)
    thinklab = v12.v11.make_stepped_thinklab_proxy(P)
    hardware = make_hardware_proxies(P)
    return {
        "fixed": fixed,
        "left": left,
        "right": right,
        "thinklab": thinklab,
        "hardware": hardware,
    }


def make_magnet_design_metadata() -> dict:
    return {
        "shape": "d5x2_cylindrical_disc",
        "axis": "Y",
        "diameter_mm": MAGNET_DIAMETER_MM,
        "thickness_mm": MAGNET_THICKNESS_MM,
        "pocket_diameter_mm": MAGNET_POCKET_DIAMETER_MM,
        "pocket_depth_mm": MAGNET_POCKET_DEPTH_MM,
        "center_mm": [round(value, 3) for value in veo4_sensor_layout(P)["magnet"]],
        "pocket_clearance_diametral_mm": round(MAGNET_POCKET_DIAMETER_MM - MAGNET_DIAMETER_MM, 3),
        "pocket_clearance_depth_mm": round(MAGNET_POCKET_DEPTH_MM - MAGNET_THICKNESS_MM, 3),
    }


def make_fastener_layout_metadata() -> dict:
    front_y, rear_y = clamshell_seam_ys(P)
    return {
        "front_seam_y_mm": round(front_y, 3),
        "rear_seam_y_mm": round(rear_y, 3),
        "clamshell_m3_screw_count": 4,
        "m3_clearance_diameter_mm": P.m3_clearance_d,
        "nut_pocket_center_x_mm": NUT_CENTER_X_MM,
    }


def make_veo4_geometry_audit(model: dict) -> dict:
    geometry = veo3.make_veo3_geometry_audit(model)
    magnet = model["hardware"]["magnet"]
    bb = magnet.val().BoundingBox()
    clamp = model["hardware"]["clamp_screws"]
    clamp_checks = []
    for printed_name in ("left", "right", "thinklab"):
        volume = v12._intersection_volume_mm3(clamp, model[printed_name])
        clamp_checks.append(
            {
                "name": f"clamp_screws_vs_{printed_name}",
                "type": "intersection_volume",
                "volume_mm3": round(volume, 6),
                "max_volume_mm3": 0.005,
                "pass": volume <= 0.005,
            }
        )
    magnet_checks = [
        {
            "name": "magnet_disc_diameter_x",
            "type": "bbox_dimension",
            "value_mm": round(bb.xmax - bb.xmin, 3),
            "expected_mm": MAGNET_DIAMETER_MM,
            "tolerance_mm": 0.05,
            "pass": abs((bb.xmax - bb.xmin) - MAGNET_DIAMETER_MM) <= 0.05,
        },
        {
            "name": "magnet_disc_thickness_y",
            "type": "bbox_dimension",
            "value_mm": round(bb.ymax - bb.ymin, 3),
            "expected_mm": MAGNET_THICKNESS_MM,
            "tolerance_mm": 0.05,
            "pass": abs((bb.ymax - bb.ymin) - MAGNET_THICKNESS_MM) <= 0.05,
        },
        {
            "name": "magnet_disc_diameter_z",
            "type": "bbox_dimension",
            "value_mm": round(bb.zmax - bb.zmin, 3),
            "expected_mm": MAGNET_DIAMETER_MM,
            "tolerance_mm": 0.05,
            "pass": abs((bb.zmax - bb.zmin) - MAGNET_DIAMETER_MM) <= 0.05,
        },
    ]
    geometry["checks"].extend(clamp_checks)
    geometry["checks"].extend(magnet_checks)
    geometry["magnet_design"] = make_magnet_design_metadata()
    geometry["pass"] = all(check["pass"] for check in geometry["checks"])
    return geometry


def make_mechanical_audit() -> dict:
    return veo3.make_mechanical_audit()


def make_material_audit() -> dict:
    return veo3.make_material_audit()


def make_validation_summary(geometry: dict, mechanical: dict, material: dict) -> dict:
    return veo3.make_validation_summary(geometry, mechanical, material)


def export_part(part: cq.Workplane, stem: str) -> None:
    exporters.export(part, str(OUT_DIR / f"{stem}.step"))
    exporters.export(part, str(OUT_DIR / f"{stem}.stl"), tolerance=0.14, angularTolerance=0.12)


def write_sim_outputs(metadata: dict) -> dict:
    ensure_ffmpeg_on_path()
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
    trace_path = SIM_DIR / "veo4_contact_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = helpers.write_contact_plot(rows, metadata, SIM_DIR)
    if plot_path is not None:
        final_plot = SIM_DIR / "veo4_contact_response.png"
        if final_plot.exists():
            final_plot.unlink()
        plot_path.rename(final_plot)
    video_path = helpers.write_motion_video(rows, metadata, SIM_DIR, fps=30, max_frames=150)
    if video_path is not None:
        final_video = SIM_DIR / "veo4_contact_motion.mp4"
        if final_video.exists():
            final_video.unlink()
        video_path.rename(final_video)
    summary = {
        "variant": RECOMMENDED_MECHANICS["simulation_variant"],
        "output_csv": str(trace_path),
        "output_plot": str(SIM_DIR / "veo4_contact_response.png"),
        "output_motion_video": str(SIM_DIR / "veo4_contact_motion.mp4"),
        "peak_force_n": max(row["contact_force_n"] for row in rows),
        "final_force_n": rows[-1]["contact_force_n"],
        "max_spring_compression_mm": max(row["spring_compression_mm"] for row in rows),
    }
    (SIM_DIR / "veo4_contact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_sim_rows() -> list[dict[str, float]]:
    with (SIM_DIR / "veo4_contact_trace.csv").open(newline="", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def render_presentation_assets(model: dict, rows: list[dict[str, float]]) -> dict:
    ensure_ffmpeg_on_path()
    configure_backends()
    clean = v12.render_clean_cad_png(model["fixed"], model["left"], model["right"], model["thinklab"], model["hardware"])
    detail = v12.render_clamshell_detail_png(model["fixed"], model["left"], model["right"], model["thinklab"], model["hardware"])
    video = v12.write_actual_cad_motion_video(model["fixed"], model["left"], model["right"], model["thinklab"], model["hardware"], rows)
    renamed = {}
    for old_path, new_name in [
        (clean, "veo4_cad_clean.png"),
        (detail, "veo4_clamshell_detail.png"),
        (video, "veo4_actual_cad_motion.mp4" if video is not None else None),
    ]:
        if old_path is None or new_name is None:
            continue
        new_path = PRESENTATION_DIR / new_name
        if new_path.exists():
            new_path.unlink()
        old_path.rename(new_path)
        renamed[new_name] = str(new_path)
    return renamed


def make_metadata(model: dict, geometry: dict, mechanical: dict, material: dict, validation: dict, sim_summary=None, presentation_assets=None) -> dict:
    metadata = {
        "units": "mm",
        "params": asdict(P),
        "recommended_mechanics": RECOMMENDED_MECHANICS,
        "mechanism": {
            "type": "veo4_magnet_tuned_clamshell",
            "basis": "VEO3 clamshell with D5x2 mm cylindrical disc magnet and matching pocket",
            "force_feedback": "Hall sensor reads moving disc-magnet displacement; force is spring calibrated.",
            "spring_visual_model": "continuous_helix",
        },
        "magnet_design": make_magnet_design_metadata(),
        "fastener_layout": make_fastener_layout_metadata(),
        "sensor_layout": veo3.make_sensor_layout_metadata(model["hardware"]),
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


def write_report(metadata: dict) -> None:
    geometry = metadata["geometry_audit"]
    mechanical = metadata["mechanical_audit"]
    material = metadata["material_audit"]
    magnet = metadata["magnet_design"]
    report = [
        "# VEO4 Magnet-Tuned Clamshell Mount",
        "",
        "VEO4 is generated from project code and uses a D5x2 mm cylindrical disc magnet in a matching printed pocket.",
        "",
        "## Validation",
        "",
        f"- Overall pass: `{metadata['validation_summary']['pass']}`",
        f"- Geometry pass: `{geometry['pass']}`",
        f"- Mechanical pass: `{mechanical['pass']}`",
        f"- Material pass: `{material['pass']}`",
        f"- Magnet shape: `{magnet['shape']}`",
        f"- Magnet/Hall Y air gap: `{geometry['clearances']['magnet_hall_y_gap_mm']:.3f} mm`",
        f"- Spring seating error: `{geometry['clearances']['spring_seating_error_max_mm']:.3f} mm`",
        f"- Target force: `{mechanical['target_contact_force_n']:.3f} N`",
        f"- Minimum material safety factor: `{material['minimum_safety_factor']:.3f}`",
        "",
        "## Limits",
        "",
        "- This is CAD and first-order analytical validation; the magnet polarity and Hall calibration still need bench testing.",
        "- Final print dimensions should be updated after measuring the real ThinkLabs One and OpenArm flange.",
    ]
    (OUT_DIR / "VEO4_MAGNET_TUNED_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    configure_backends()
    model = build_veo4_model()
    geometry = make_veo4_geometry_audit(model)
    mechanical = make_mechanical_audit()
    material = make_material_audit()
    validation = make_validation_summary(geometry, mechanical, material)

    export_part(model["fixed"], "thinklab_openarm_veo4_fixed_bridge_print")
    export_part(model["left"], "thinklab_openarm_veo4_left_clamshell_print")
    export_part(model["right"], "thinklab_openarm_veo4_right_clamshell_print")
    exporters.export(model["thinklab"], str(OUT_DIR / "thinklab_veo4_stepped_profile_proxy.step"))

    assembly = cq.Assembly(name="thinklab_openarm_veo4_magnet_tuned_mount")
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
    assembly.save(str(OUT_DIR / "thinklab_openarm_veo4_assembly.step"), exportType="STEP")

    preview = cq.Compound.makeCompound(
        [model["fixed"].val(), model["left"].val(), model["right"].val(), model["thinklab"].val()]
        + [shape.val() for shape in model["hardware"].values()]
    )
    exporters.export(cq.Workplane("XY").add(preview), str(OUT_DIR / "thinklab_openarm_veo4_preview.step"))

    (OUT_DIR / "veo4_geometry_audit.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    (OUT_DIR / "veo4_mechanical_audit.json").write_text(json.dumps(mechanical, indent=2), encoding="utf-8")
    (OUT_DIR / "veo4_material_audit.json").write_text(json.dumps(material, indent=2), encoding="utf-8")

    metadata = make_metadata(model, geometry, mechanical, material, validation)
    sim_summary = write_sim_outputs(metadata)
    rows = read_sim_rows()
    presentation_assets = render_presentation_assets(model, rows)
    metadata = make_metadata(model, geometry, mechanical, material, validation, sim_summary, presentation_assets)
    (OUT_DIR / "veo4_design_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(metadata)

    print(
        json.dumps(
            {
                "validation_summary": validation,
                "magnet_design": make_magnet_design_metadata(),
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
