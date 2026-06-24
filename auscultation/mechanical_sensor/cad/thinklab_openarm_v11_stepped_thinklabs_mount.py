"""V11 stepped ThinkLabs One profile mount.

V11 corrects the stethoscope proxy. ThinkLabs One is modeled as a larger lower
frustum with a smaller upper disc/control face, then the printed blue cradle is
changed from a single-diameter clamp to a lower-rim support plus upper retaining
pads.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path


V10_PATH = Path(__file__).with_name("thinklab_openarm_v10_detailed_clamp_mount.py")
spec = importlib.util.spec_from_file_location("v10_mount", V10_PATH)
v10 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["v10_mount"] = v10
spec.loader.exec_module(v10)

cq = v10.cq
exporters = v10.exporters
plt = v10.plt
Poly3DCollection = v10.Poly3DCollection
v9 = v10.v9


P = v9.SimplifiedRailMountParams(
    cup_clearance=0.8,
    cup_wall=4.0,
    cup_height=17.0,
    lip_height=2.8,
    lip_opening_d=52.0,
    cable_window_w=14.0,
    clamp_gap_w=10.0,
    guide_rod_d=5.0,
    guide_rod_clearance_d=5.8,
    guide_rod_x=45.0,
    spring_axis_y=44.0,
    top_bridge_x=92.0,
    top_bridge_y=34.0,
    guide_bushing_h=18.0,
)

RECOMMENDED_MECHANICS = {
    "simulation_variant": "v11_stepped_thinklabs_profile_clamp",
    "spring_rate_n_per_mm": 0.5,
    "linear_damping_n_s_per_m": 3.2,
    "target_contact_force_n": 4.0,
    "max_screened_contact_force_n": 5.5,
}

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated" / "v11_stepped_thinklabs"
SIM_DIR = OUT_DIR / "simulation"
PRESENTATION_DIR = OUT_DIR / "presentation"

THINKLABS_PROFILE = {
    "lower_bottom_d_mm": 66.0,
    "lower_top_d_mm": 56.0,
    "lower_frustum_h_mm": 14.0,
    "upper_disc_d_mm": 48.0,
    "upper_disc_h_mm": 10.0,
    "top_control_face_d_mm": 39.0,
    "top_control_face_h_mm": 1.2,
}


def cone_z(radius_bottom: float, radius_top: float, height: float, center: tuple[float, float, float]) -> cq.Workplane:
    x, y, z = center
    solid = cq.Solid.makeCone(radius_bottom, radius_top, height)
    return cq.Workplane("XY").add(solid).translate((x, y, z - height / 2.0))


def radial_box(tangent: float, radial: float, height: float, radius: float, z: float, angle_deg: float) -> cq.Workplane:
    part = v9.box(tangent, radial, height, (0, radius, z))
    return part.rotate((0, 0, 0), (0, 0, 1), angle_deg)


def make_stepped_thinklab_proxy(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    prof = THINKLABS_PROFILE
    lower_h = prof["lower_frustum_h_mm"]
    upper_h = prof["upper_disc_h_mm"]
    lower_center_z = p.carriage_z - 1.0
    lower_top_z = lower_center_z + lower_h / 2.0
    upper_center_z = lower_top_z + upper_h / 2.0
    top_face_center_z = lower_top_z + upper_h + prof["top_control_face_h_mm"] / 2.0

    lower = cone_z(prof["lower_bottom_d_mm"] / 2.0, prof["lower_top_d_mm"] / 2.0, lower_h, (0, 0, lower_center_z))
    shoulder = v9.cyl_z(prof["lower_bottom_d_mm"] / 2.0, 1.2, (0, 0, lower_center_z - lower_h / 2.0 + 0.6))
    upper = v9.cyl_z(prof["upper_disc_d_mm"] / 2.0, upper_h, (0, 0, upper_center_z))
    top_face = v9.cyl_z(prof["top_control_face_d_mm"] / 2.0, prof["top_control_face_h_mm"], (0, 0, top_face_center_z))

    body_bottom_z = lower_center_z - lower_h / 2.0
    contact_h = 2.5
    contact_bottom_z = body_bottom_z - p.contact_face_projection
    contact_center_z = contact_bottom_z + contact_h / 2.0
    neck_h = body_bottom_z - (contact_bottom_z + contact_h)
    neck = v9.cyl_z(23.0, neck_h, (0, 0, contact_bottom_z + contact_h + neck_h / 2.0))
    contact = v9.cyl_z(26.0, contact_h, (0, 0, contact_center_z))

    # Front connector/jack pair visible in the reference photos.
    jack_y = -prof["lower_bottom_d_mm"] / 2.0 - 0.4
    jack_z = lower_center_z - 1.5
    jack_l = 6.5
    jack_pair = None
    for sx in (-6.0, 6.0):
        jack = v9.cyl_y(2.0, jack_l, (sx, jack_y, jack_z))
        jack = jack.union(v9.cyl_y(1.25, 1.6, (sx, jack_y - jack_l / 2.0 - 0.6, jack_z)))
        jack_pair = jack if jack_pair is None else jack_pair.union(jack)
    assert jack_pair is not None

    cable_stub = v9.cyl_y(1.4, 12.0, (0, prof["lower_bottom_d_mm"] / 2.0 + 4.0, lower_center_z + 1.0))
    return lower.union(shoulder).union(upper).union(top_face).union(neck).union(contact).union(jack_pair).union(cable_stub)


def make_stepped_cradle(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    prof = THINKLABS_PROFILE
    lower_clearance_d = prof["lower_bottom_d_mm"] + 1.0
    upper_clearance_d = prof["upper_disc_d_mm"] + 0.8
    lower_inner_r = lower_clearance_d / 2.0
    upper_inner_r = upper_clearance_d / 2.0
    outer_r = lower_inner_r + p.cup_wall
    zc = p.carriage_z

    lower_band = v9.cyl_z(outer_r, p.cup_height, (0, 0, zc)).cut(v9.cyl_z(lower_inner_r, p.cup_height + 3.0, (0, 0, zc)))
    lip_z = zc - p.cup_height / 2.0 + p.lip_height / 2.0
    lower_lip = v9.cyl_z(outer_r, p.lip_height, (0, 0, lip_z)).cut(v9.cyl_z(p.lip_opening_d / 2.0, p.lip_height + 3.0, (0, 0, lip_z)))
    cradle = lower_band.union(lower_lip)

    # Front split clamp plus jack window. The lower lip still supports the big
    # bottom rim; the split is only where the clamp closes.
    cradle = cradle.cut(v9.box(p.clamp_gap_w, outer_r + 14.0, p.cup_height + 5.0, (0, -outer_r, zc)))
    cradle = cradle.cut(v9.box(18.0, 11.0, 8.0, (0, -outer_r - 1.0, zc - 1.0)))
    cradle = cradle.cut(v9.box(p.cable_window_w, outer_r + 12.0, 7.0, (0, outer_r, zc + 5.0)))

    clamp_y = -outer_r - 4.5
    for sx in (-14.5, 14.5):
        cradle = cradle.union(v9.box(13.0, 12.0, 14.0, (sx, clamp_y, zc)))
    for dz in (-4.0, 4.0):
        cradle = cradle.cut(v9.cyl_x(p.m3_clearance_d / 2.0, 52.0, (0, clamp_y, zc + dz)))
        cradle = cradle.cut(v10.hex_prism_x(3.55, 4.0, (22.0, clamp_y, zc + dz)))

    for sx in (-p.guide_rod_x, p.guide_rod_x):
        bushing = v9.cyl_z(p.guide_bushing_od / 2.0, p.guide_bushing_h, (sx, p.guide_rod_y, zc))
        bushing = bushing.cut(v9.cyl_z(p.guide_rod_clearance_d / 2.0, p.guide_bushing_h + 3.0, (sx, p.guide_rod_y, zc)))
        bridge = v9.box(10.0, 10.0, 10.0, (sx * 0.94, p.guide_rod_y, zc))
        cradle = cradle.union(bushing).union(bridge)

    # Three upper retaining pads lightly capture the smaller upper disc at the
    # shoulder without covering the front jacks.
    upper_pad_z = zc + 9.0
    upper_pad_radius = upper_inner_r + 3.0
    for angle in (65.0, 180.0, 295.0):
        cradle = cradle.union(radial_box(13.0, 6.0, 5.5, upper_pad_radius, upper_pad_z, angle))

    spring_bridge_y = (outer_r + p.spring_axis_y) / 2.0
    spring_bridge_len = max(5.0, p.spring_axis_y - outer_r + 4.0)
    cradle = cradle.union(v9.box(16.0, spring_bridge_len, 4.0, (0, spring_bridge_y, p.spring_moving_face_z - 1.5)))
    cradle = cradle.union(v9.upward_spring_socket(p))

    magnet_x, magnet_y, magnet_z = p.guide_rod_x - 7.0, -7.0, p.carriage_z + 10.0
    cradle = cradle.union(v9.box(9.5, 4.5, 7.0, (magnet_x, magnet_y + 1.5, magnet_z)))
    cradle = cradle.cut(v9.box(5.8, 2.4, 5.8, (magnet_x, magnet_y - 0.7, magnet_z)))
    return cradle


def make_hardware_proxies(p: v9.SimplifiedRailMountParams) -> dict[str, cq.Workplane]:
    hardware = v10.make_hardware_proxies(p)
    magnet_x, magnet_y, magnet_z = p.guide_rod_x - 7.0, -7.0, p.carriage_z + 10.0
    hall_x, hall_y, hall_z = p.guide_rod_x - 6.5, -5.8, -15.0
    hardware["magnet"] = v9.box(5.0, 2.0, 5.0, (magnet_x, magnet_y, magnet_z))
    hardware["hall"] = v9.box(7.0, 2.2, 9.0, (hall_x, hall_y, hall_z))
    return hardware


def assembly_parts(fixed, cradle, thinklab, hardware):
    return [
        (fixed, "#d7d0bd", 1.0),
        (cradle, "#2471b5", 1.0),
        (thinklab, "#202326", 0.46),
        (hardware["robot_flange"], "#8b8f95", 0.55),
        (hardware["interface_bolts"], "#111111", 0.9),
        (hardware["guide_rods"], "#b8b8b8", 0.88),
        (hardware["spring"], "#47b567", 0.9),
        (hardware["clamp_screws"], "#111111", 0.95),
        (hardware["magnet"], "#d62828", 1.0),
        (hardware["hall"], "#6a1b9a", 1.0),
        (hardware["sensor_connector"], "#0f9d8a", 0.9),
        (hardware["hall_vcc_wire"], "#d62828", 1.0),
        (hardware["hall_signal_wire"], "#f2c230", 1.0),
        (hardware["hall_gnd_wire"], "#111111", 1.0),
    ]


def compound_bounds(shapes: list[cq.Workplane]) -> tuple[float, float, float, float, float, float]:
    comp = cq.Compound.makeCompound([shape.val() for shape in shapes])
    bb = comp.BoundingBox()
    return (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax)


def render_clean_cad_png(fixed, cradle, thinklab, hardware) -> Path:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    bounds = compound_bounds([fixed, cradle, thinklab, *hardware.values()])
    fig = plt.figure(figsize=(10.0, 8.0), dpi=240)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    for shape, color, alpha in assembly_parts(fixed, cradle, thinklab, hardware):
        v9.add_shape_to_axis(ax, shape, color, alpha, tolerance=0.95)
    v9.set_equal_axes(ax, bounds)
    ax.view_init(elev=24, azim=-42)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    path = PRESENTATION_DIR / "v11_cad_clean.png"
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


def render_preview_png(fixed, cradle, thinklab, hardware) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bounds = compound_bounds([fixed, cradle, thinklab, *hardware.values()])
    views = [("Isometric", 24, -42), ("Front clamp", 0, -90), ("Side profile", 0, 0), ("Top pads", 90, -90)]
    fig = plt.figure(figsize=(13.5, 10.0), dpi=180)
    fig.patch.set_facecolor("#f7f7f5")
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.set_facecolor("#f7f7f5")
        for shape, color, alpha in assembly_parts(fixed, cradle, thinklab, hardware):
            v9.add_shape_to_axis(ax, shape, color, alpha)
        v9.set_equal_axes(ax, bounds)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11, pad=0)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    path = OUT_DIR / "thinklab_openarm_v11_stepped_preview.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def render_profile_clamp_detail_png(fixed, cradle, thinklab, hardware) -> Path:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Corrected side profile: lower large frustum, upper small disc", 0, 0, (-44, 44, -38, 38, -54, -8)),
        ("Lower rim support: blue lip carries the larger bottom disc", 0, -90, (-44, 44, -52, -18, -42, -18)),
        ("Upper retaining pads: three blue pads capture the smaller upper disc", 70, -90, (-42, 42, -42, 42, -30, -8)),
        ("Open windows: front jacks and rear cable are not blocked", 30, -55, (-50, 50, -50, 50, -42, -8)),
    ]
    fig = plt.figure(figsize=(13.5, 10.0), dpi=180)
    fig.patch.set_facecolor("white")
    for idx, (title, elev, azim, bounds) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.set_facecolor("white")
        for shape, color, alpha in assembly_parts(fixed, cradle, thinklab, hardware):
            if idx == 1 and shape is cradle:
                alpha = 0.25
            if idx == 3 and shape is thinklab:
                alpha = 0.28
            v9.add_shape_to_axis(ax, shape, color, alpha, tolerance=0.95)
        v9.set_equal_axes(ax, bounds)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    path = PRESENTATION_DIR / "v11_profile_clamp_detail.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def write_sim_outputs(metadata: dict) -> dict:
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    helpers = v10.load_isaac_helpers()
    rows = helpers.simulate_step_response(
        metadata,
        duration=3.0,
        dt=1.0 / 120.0,
        approach_speed_mm_s=28.0,
        tissue_stiffness_n_per_mm=0.8,
        tissue_damping_n_s_per_m=12.0,
    )
    trace_path = SIM_DIR / "v11_contact_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = helpers.write_contact_plot(rows, metadata, SIM_DIR)
    if plot_path is not None:
        plot_path.rename(SIM_DIR / "v11_contact_response.png")
    video_path = helpers.write_motion_video(rows, metadata, SIM_DIR, fps=30, max_frames=150)
    if video_path is not None:
        video_path.rename(SIM_DIR / "v11_contact_motion.mp4")
    summary = {
        "variant": RECOMMENDED_MECHANICS["simulation_variant"],
        "output_csv": str(trace_path),
        "output_plot": str(SIM_DIR / "v11_contact_response.png"),
        "output_motion_video": str(SIM_DIR / "v11_contact_motion.mp4"),
        "peak_force_n": max(row["contact_force_n"] for row in rows),
        "final_force_n": rows[-1]["contact_force_n"],
        "max_spring_compression_mm": max(row["spring_compression_mm"] for row in rows),
    }
    (SIM_DIR / "v11_contact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_sim_rows() -> list[dict[str, float]]:
    with (SIM_DIR / "v11_contact_trace.csv").open(newline="", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def write_actual_cad_motion_video(fixed, cradle, thinklab, hardware, rows) -> Path | None:
    v9.PRESENTATION_DIR = PRESENTATION_DIR
    old_path = v9.write_actual_cad_motion_video(fixed, cradle, thinklab, hardware, rows, P)
    if old_path is None:
        return None
    new_path = PRESENTATION_DIR / "v11_actual_cad_motion.mp4"
    if new_path.exists():
        new_path.unlink()
    old_path.rename(new_path)
    return new_path


def export_part(part: cq.Workplane, stem: str) -> None:
    exporters.export(part, str(OUT_DIR / f"{stem}.step"))
    exporters.export(part, str(OUT_DIR / f"{stem}.stl"), tolerance=0.14, angularTolerance=0.12)


def make_metadata(fixed, cradle, sim_summary=None, presentation_assets=None) -> dict:
    prof = THINKLABS_PROFILE
    lower_clearance_d = prof["lower_bottom_d_mm"] + 1.0
    upper_clearance_d = prof["upper_disc_d_mm"] + 0.8
    metadata = {
        "units": "mm",
        "params": asdict(P),
        "recommended_mechanics": RECOMMENDED_MECHANICS,
        "print_parts": {
            "fixed_bridge": v9.bbox_dict(fixed),
            "stepped_cradle": v9.bbox_dict(cradle),
        },
        "mechanism": {
            "type": "stepped_profile_lower_rim_upper_retainer",
            "blue_cradle_is_main_structural_part": True,
            "force_feedback": "Hall sensor reads the moving red magnet displacement; force is spring calibrated.",
        },
        "thinklabs_profile": {
            "profile_type": "lower_large_frustum_plus_upper_small_disc",
            "lower_bottom_d_mm": prof["lower_bottom_d_mm"],
            "lower_top_d_mm": prof["lower_top_d_mm"],
            "lower_frustum_h_mm": prof["lower_frustum_h_mm"],
            "upper_disc_d_mm": prof["upper_disc_d_mm"],
            "upper_disc_h_mm": prof["upper_disc_h_mm"],
            "top_control_face_drawn": True,
            "front_jack_pair_drawn": True,
            "side_profile_is_trapezoid_like": True,
            "source_note": "Approximated from local reference images in mechanical_sensor/thinklab_one.",
        },
        "stepped_clamp_layout": {
            "lower_rim_support_drawn": True,
            "lower_rim_clearance_d_mm": lower_clearance_d,
            "upper_disc_retaining_pads_drawn": True,
            "upper_retaining_pad_count": 3,
            "upper_disc_clearance_d_mm": upper_clearance_d,
            "front_jack_window_drawn": True,
            "rear_cable_window_drawn": True,
            "m3_screw_count": 2,
            "clamp_strategy": "support large lower rim, lightly retain smaller upper disc shoulder, leave jacks/cable windows open",
        },
        "contact_layout": {
            "dark_proxy_is_stethoscope": True,
            "only_dark_bottom_contact_face_projects_below_printed_parts": True,
            "contact_face_projects_below_printed_parts_mm": P.contact_face_projection,
        },
        "spring_layout": {
            "spring_count": 1,
            "seat_face_gap_at_rest_mm": round(P.spring_fixed_face_z - P.spring_moving_face_z, 3),
        },
        "robot_interface_layout": {
            "openarm_flange_proxy_drawn": True,
            "mount_slot_count": 4,
            "bolt_proxy_count": 4,
        },
    }
    if sim_summary is not None:
        metadata["simulation_summary"] = sim_summary
    if presentation_assets is not None:
        metadata["presentation_assets"] = presentation_assets
    return metadata


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    fixed = v10.make_fixed_bridge(P)
    cradle = make_stepped_cradle(P)
    thinklab = make_stepped_thinklab_proxy(P)
    hardware = make_hardware_proxies(P)

    export_part(fixed, "thinklab_openarm_v11_fixed_bridge_print")
    export_part(cradle, "thinklab_openarm_v11_stepped_cradle_print")
    exporters.export(thinklab, str(OUT_DIR / "thinklab_v11_stepped_profile_proxy.step"))

    assembly = cq.Assembly(name="thinklab_openarm_v11_stepped_thinklabs_mount")
    assembly.add(fixed, name="printed_fixed_bridge_with_interface_slots", color=cq.Color(0.83, 0.80, 0.70))
    assembly.add(cradle, name="printed_blue_stepped_profile_cradle", color=cq.Color(0.05, 0.34, 0.75))
    assembly.add(thinklab, name="thinklab_one_stepped_proxy_not_printed", color=cq.Color(0.05, 0.05, 0.05, 0.40))
    assembly.add(hardware["robot_flange"], name="openarm_tool_flange_proxy_not_printed", color=cq.Color(0.54, 0.56, 0.58, 0.55))
    assembly.add(hardware["interface_bolts"], name="top_interface_bolts_proxy_not_printed", color=cq.Color(0.02, 0.02, 0.02))
    assembly.add(hardware["guide_rods"], name="two_guide_rods_proxy_not_printed", color=cq.Color(0.72, 0.72, 0.72))
    assembly.add(hardware["spring"], name="wf6_25_visible_coil_proxy_not_printed", color=cq.Color(0.2, 0.8, 0.25))
    assembly.add(hardware["clamp_screws"], name="two_m3_clamp_screws_and_nuts_proxy_not_printed", color=cq.Color(0.02, 0.02, 0.02))
    assembly.add(hardware["magnet"], name="moving_red_magnet_proxy_not_printed", color=cq.Color(0.84, 0.16, 0.16))
    assembly.add(hardware["hall"], name="fixed_purple_hall_sensor_proxy_not_printed", color=cq.Color(0.42, 0.10, 0.60))
    assembly.add(hardware["sensor_connector"], name="local_sensor_connector_proxy_not_printed", color=cq.Color(0.06, 0.62, 0.54))
    assembly.add(hardware["hall_vcc_wire"], name="hall_vcc_short_wire_proxy_not_printed", color=cq.Color(0.84, 0.16, 0.16))
    assembly.add(hardware["hall_signal_wire"], name="hall_signal_short_wire_proxy_not_printed", color=cq.Color(0.95, 0.76, 0.19))
    assembly.add(hardware["hall_gnd_wire"], name="hall_gnd_short_wire_proxy_not_printed", color=cq.Color(0.02, 0.02, 0.02))
    assembly.save(str(OUT_DIR / "thinklab_openarm_v11_stepped_assembly.step"), exportType="STEP")

    preview = cq.Compound.makeCompound([fixed.val(), cradle.val(), thinklab.val()] + [shape.val() for shape in hardware.values()])
    exporters.export(cq.Workplane("XY").add(preview), str(OUT_DIR / "thinklab_openarm_v11_stepped_preview.step"))
    render_preview_png(fixed, cradle, thinklab, hardware)

    metadata = make_metadata(fixed, cradle)
    sim_summary = write_sim_outputs(metadata)
    rows = read_sim_rows()
    presentation_assets = {
        "clean_cad_png": str(render_clean_cad_png(fixed, cradle, thinklab, hardware)),
        "profile_clamp_detail_png": str(render_profile_clamp_detail_png(fixed, cradle, thinklab, hardware)),
    }
    video = write_actual_cad_motion_video(fixed, cradle, thinklab, hardware, rows)
    if video is not None:
        presentation_assets["actual_cad_motion_mp4"] = str(video)
    metadata = make_metadata(fixed, cradle, sim_summary, presentation_assets)
    (OUT_DIR / "v11_design_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    calibration = {
        "variant": RECOMMENDED_MECHANICS["simulation_variant"],
        "measurement_type": "spring_compression_to_force",
        "spring_rate_n_per_mm": RECOMMENDED_MECHANICS["spring_rate_n_per_mm"],
        "target_force_n": RECOMMENDED_MECHANICS["target_contact_force_n"],
        "max_screened_force_n": RECOMMENDED_MECHANICS["max_screened_contact_force_n"],
        "travel_limit_mm": P.travel,
    }
    (OUT_DIR / "v11_force_feedback_calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    report = [
        "# V11 Stepped ThinkLabs Profile Mount",
        "",
        "V11 corrects the ThinkLabs One proxy from a plain cylinder to a lower large frustum plus upper small disc profile.",
        "",
        "## Clamp Strategy",
        "",
        "- Lower blue lip supports the larger bottom rim.",
        "- Three upper retaining pads capture the smaller upper disc shoulder.",
        "- Front jack pair and rear cable have open windows.",
        "- Two M3 screws still close the front split clamp.",
        "",
        "## Simulation",
        "",
        f"- Peak force: `{sim_summary['peak_force_n']:.3f} N`",
        f"- Final force: `{sim_summary['final_force_n']:.3f} N`",
        f"- Max compression: `{sim_summary['max_spring_compression_mm']:.3f} mm`",
    ]
    (OUT_DIR / "V11_STEPPED_THINKLABS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"print_parts": metadata["print_parts"], "thinklabs_profile": metadata["thinklabs_profile"], "simulation": sim_summary}, indent=2))


if __name__ == "__main__":
    main()
