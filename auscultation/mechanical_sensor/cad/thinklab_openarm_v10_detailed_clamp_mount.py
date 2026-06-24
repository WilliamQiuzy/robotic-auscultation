"""Detailed v10 mount concept for ThinkLabs One + OpenArm arm.

V10 keeps the simple V9 architecture but makes the ambiguous details explicit:
the clamp screws, nut pockets, spring seats, magnet/Hall pair, short sensor
connector, and robot-side interface are all visible in the assembly STEP.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path


V9_PATH = Path(__file__).with_name("thinklab_openarm_v9_simplified_rail_mount.py")
spec = importlib.util.spec_from_file_location("v9_mount", V9_PATH)
v9 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["v9_mount"] = v9
spec.loader.exec_module(v9)

cq = v9.cq
exporters = v9.exporters
plt = v9.plt
Poly3DCollection = v9.Poly3DCollection


P = v9.SimplifiedRailMountParams(
    cable_window_w=14.0,
    clamp_gap_w=10.0,
    guide_rod_d=5.0,
    guide_rod_clearance_d=5.8,
    guide_rod_x=43.0,
    spring_axis_y=42.0,
    top_bridge_y=34.0,
    guide_bushing_h=18.0,
)
RECOMMENDED_MECHANICS = {
    "simulation_variant": "v10_detailed_two_rail_visible_interfaces",
    "spring_rate_n_per_mm": 0.5,
    "linear_damping_n_s_per_m": 3.2,
    "target_contact_force_n": 4.0,
    "max_screened_contact_force_n": 5.5,
}

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated" / "v10_detailed"
SIM_DIR = OUT_DIR / "simulation"
PRESENTATION_DIR = OUT_DIR / "presentation"


def hex_prism_x(radius: float, height: float, center: tuple[float, float, float]) -> cq.Workplane:
    x, y, z = center
    return cq.Workplane("YZ").polygon(6, radius).extrude(height).translate((x - height / 2.0, y, z))


def make_fixed_bridge(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    bridge = v9.box(p.top_bridge_x, p.top_bridge_y, p.top_bridge_t, (0, 0, 0))
    for sx in (-18.0, 18.0):
        for sy in (-10.0, 10.0):
            bridge = bridge.cut(v9.slot_cut_z(p.slot_len, p.slot_w, p.top_bridge_t + 3.0, (sx, sy, 0)))

    for sx in (-p.guide_rod_x, p.guide_rod_x):
        socket = v9.cyl_z(p.rod_socket_od / 2.0, 7.0, (sx, p.guide_rod_y, -3.0))
        socket = socket.cut(v9.cyl_z(p.guide_rod_clearance_d / 2.0, 9.0, (sx, p.guide_rod_y, -3.0)))
        bridge = bridge.union(socket)

    tab_center_y = (p.top_bridge_y / 2.0 + p.spring_axis_y) / 2.0
    tab_y = p.spring_axis_y - p.top_bridge_y / 2.0 + p.spring_seat_d / 2.0 + 1.5
    bridge = bridge.union(v9.box(p.spring_tab_x, tab_y, p.top_bridge_t, (0, tab_center_y, 0)))
    bridge = bridge.union(v9.downward_spring_socket(p))

    # Small fixed-side Hall boss. It is connected to the top bridge and kept
    # short so the tan print does not become another large side frame.
    hall_x, hall_y = p.guide_rod_x - 4.5, -2.5
    bridge = bridge.union(v9.box(5.0, 5.0, 13.5, (hall_x, hall_y, -9.0)))
    bridge = bridge.union(v9.box(9.0, 5.5, 3.6, (hall_x, hall_y, -15.0)))
    bridge = bridge.cut(v9.cyl_x(p.m2_clearance_d / 2.0, 7.0, (hall_x, hall_y, -15.0)))
    return bridge


def make_blue_cradle(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    body_r = p.thinklab_body_d / 2.0 + p.cup_clearance
    outer_r = body_r + p.cup_wall
    zc = p.carriage_z

    ring = v9.cyl_z(outer_r, p.cup_height, (0, 0, zc)).cut(v9.cyl_z(body_r, p.cup_height + 3.0, (0, 0, zc)))
    lip_z = zc - p.cup_height / 2.0 + p.lip_height / 2.0
    lip = v9.cyl_z(outer_r, p.lip_height, (0, 0, lip_z)).cut(v9.cyl_z(p.lip_opening_d / 2.0, p.lip_height + 3.0, (0, 0, lip_z)))
    cradle = ring.union(lip)

    # One real split gap for clamp action. There is intentionally no center
    # blue bridge across this gap; the black screws are the only cross-link.
    cradle = cradle.cut(v9.box(p.clamp_gap_w, outer_r + 12.0, p.cup_height + 4.0, (0, -outer_r, zc)))

    # Rear cable exit is a shallow upper notch, not a full-height second split.
    cradle = cradle.cut(v9.box(p.cable_window_w, outer_r + 10.0, 7.0, (0, outer_r, zc + 4.5)))

    clamp_y = -outer_r - 4.5
    for sx in (-14.0, 14.0):
        cradle = cradle.union(v9.box(13.0, 12.0, 14.0, (sx, clamp_y, zc)))
    for dz in (-4.0, 4.0):
        cradle = cradle.cut(v9.cyl_x(p.m3_clearance_d / 2.0, 50.0, (0, clamp_y, zc + dz)))
        cradle = cradle.cut(hex_prism_x(3.55, 4.0, (21.0, clamp_y, zc + dz)))

    for sx in (-p.guide_rod_x, p.guide_rod_x):
        bushing = v9.cyl_z(p.guide_bushing_od / 2.0, p.guide_bushing_h, (sx, p.guide_rod_y, zc))
        bushing = bushing.cut(v9.cyl_z(p.guide_rod_clearance_d / 2.0, p.guide_bushing_h + 3.0, (sx, p.guide_rod_y, zc)))
        bridge = v9.box(10.0, 10.0, 10.0, (sx * 0.94, p.guide_rod_y, zc))
        cradle = cradle.union(bushing).union(bridge)

    spring_bridge_y = (outer_r + p.spring_axis_y) / 2.0
    spring_bridge_len = max(5.0, p.spring_axis_y - outer_r + 4.0)
    cradle = cradle.union(v9.box(16.0, spring_bridge_len, 4.0, (0, spring_bridge_y, p.spring_moving_face_z - 1.5)))
    cradle = cradle.union(v9.upward_spring_socket(p))

    magnet_x, magnet_y, magnet_z = p.guide_rod_x - 5.0, -5.5, p.carriage_z + 10.0
    cradle = cradle.union(v9.box(9.5, 4.5, 7.0, (magnet_x, magnet_y, magnet_z)))
    cradle = cradle.cut(v9.box(5.8, 2.4, 5.8, (magnet_x, magnet_y - 2.2, magnet_z)))
    return cradle


def make_thinklab_proxy(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    return v9.make_thinklab_proxy(p)


def make_visible_spring(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    top_z = p.spring_fixed_face_z
    bottom_z = p.spring_moving_face_z
    count = 11
    spring = None
    for i in range(count):
        z = bottom_z + 1.4 + i * ((top_z - bottom_z - 2.8) / (count - 1))
        ring = v9.cyl_z(p.spring_od / 2.0, 0.7, (p.spring_axis_x, p.spring_axis_y, z))
        ring = ring.cut(v9.cyl_z((p.spring_od - 1.4) / 2.0, 1.0, (p.spring_axis_x, p.spring_axis_y, z)))
        spring = ring if spring is None else spring.union(ring)
    assert spring is not None
    return spring


def make_hardware_proxies(p: v9.SimplifiedRailMountParams) -> dict[str, cq.Workplane]:
    guide_rods = None
    rod_center_z = -18.5
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        rod = v9.cyl_z(p.guide_rod_d / 2.0, p.guide_rod_length, (sx, p.guide_rod_y, rod_center_z))
        guide_rods = rod if guide_rods is None else guide_rods.union(rod)
    assert guide_rods is not None

    body_r = p.thinklab_body_d / 2.0 + p.cup_clearance
    outer_r = body_r + p.cup_wall
    clamp_y = -outer_r - 4.5
    clamp_screws = None
    for dz in (-4.0, 4.0):
        screw = v9.cyl_x(1.35, 48.0, (0, clamp_y, p.carriage_z + dz))
        screw = screw.union(v9.cyl_x(2.4, 3.0, (-24.5, clamp_y, p.carriage_z + dz)))
        screw = screw.union(hex_prism_x(3.1, 2.4, (23.4, clamp_y, p.carriage_z + dz)))
        clamp_screws = screw if clamp_screws is None else clamp_screws.union(screw)
    assert clamp_screws is not None

    magnet_x, magnet_y, magnet_z = p.guide_rod_x - 5.0, -7.0, p.carriage_z + 10.0
    hall_x, hall_y, hall_z = p.guide_rod_x - 4.5, -5.8, -15.0
    magnet = v9.box(5.0, 2.0, 5.0, (magnet_x, magnet_y, magnet_z))
    hall = v9.box(7.0, 2.2, 9.0, (hall_x, hall_y, hall_z))

    flange = v9.box(70.0, 28.0, 3.0, (0, 0, 5.2))
    interface_bolts = None
    for sx in (-18.0, 18.0):
        for sy in (-10.0, 10.0):
            bolt = v9.cyl_z(1.6, 9.0, (sx, sy, 1.0))
            head = v9.cyl_z(3.0, 1.8, (sx, sy, 5.2))
            bolt = bolt.union(head)
            interface_bolts = bolt if interface_bolts is None else interface_bolts.union(bolt)
    assert interface_bolts is not None

    connector = v9.box(10.0, 6.0, 4.0, (23.0, -14.0, -4.0))
    connector = connector.union(v9.box(4.0, 3.0, 2.0, (29.5, -14.0, -4.0)))
    wire_r = 0.38
    wires: dict[str, cq.Workplane] = {}
    for name, color_offset, dz in [
        ("hall_vcc_wire", 0.0, 1.0),
        ("hall_signal_wire", 1.2, 0.0),
        ("hall_gnd_wire", 2.4, -1.0),
    ]:
        y0 = hall_y - color_offset
        z0 = hall_z + dz
        xmid = (hall_x + 29.0) / 2.0
        segment_x = v9.cyl_x(wire_r, abs(hall_x - 29.0), (xmid, y0, z0))
        segment_y = v9.cyl_y(wire_r, abs(y0 + 14.0), (29.0, (y0 - 14.0) / 2.0, z0))
        segment_z = v9.cyl_z(wire_r, abs(z0 + 4.0), (29.0, -14.0, (z0 - 4.0) / 2.0))
        wires[name] = segment_x.union(segment_y).union(segment_z)

    return {
        "spring": make_visible_spring(p),
        "guide_rods": guide_rods,
        "clamp_screws": clamp_screws,
        "magnet": magnet,
        "hall": hall,
        "robot_flange": flange,
        "interface_bolts": interface_bolts,
        "sensor_connector": connector,
        **wires,
    }


def assembly_parts(
    fixed: cq.Workplane,
    cradle: cq.Workplane,
    thinklab: cq.Workplane,
    hardware: dict[str, cq.Workplane],
) -> list[tuple[cq.Workplane, str, float]]:
    return [
        (fixed, "#d7d0bd", 1.0),
        (cradle, "#2471b5", 1.0),
        (thinklab, "#202326", 0.45),
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


def render_preview_png(
    fixed: cq.Workplane,
    cradle: cq.Workplane,
    thinklab: cq.Workplane,
    hardware: dict[str, cq.Workplane],
) -> Path:
    preview = cq.Compound.makeCompound([fixed.val(), cradle.val(), thinklab.val()] + [shape.val() for shape in hardware.values()])
    bb = preview.BoundingBox()
    bounds = (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax)
    views = [("Isometric", 24, -42), ("Front", 0, -90), ("Right detail", 0, 0), ("Top clamp", 90, -90)]
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
    path = OUT_DIR / "thinklab_openarm_v10_detailed_preview.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def render_clean_cad_png(
    fixed: cq.Workplane,
    cradle: cq.Workplane,
    thinklab: cq.Workplane,
    hardware: dict[str, cq.Workplane],
) -> Path:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    preview = cq.Compound.makeCompound([fixed.val(), cradle.val(), thinklab.val()] + [shape.val() for shape in hardware.values()])
    bb = preview.BoundingBox()
    bounds = (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax)
    fig = plt.figure(figsize=(10.0, 8.0), dpi=240)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    for shape, color, alpha in assembly_parts(fixed, cradle, thinklab, hardware):
        v9.add_shape_to_axis(ax, shape, color, alpha, tolerance=0.95)
    v9.set_equal_axes(ax, bounds)
    ax.view_init(elev=24, azim=-42)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    path = PRESENTATION_DIR / "v10_cad_clean.png"
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


def render_detail_layout_png(
    fixed: cq.Workplane,
    cradle: cq.Workplane,
    thinklab: cq.Workplane,
    hardware: dict[str, cq.Workplane],
) -> Path:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Clamp: two M3 screws cross the open split", 76, -90, (-36, 36, -55, -22, -43, -17)),
        ("Spring: top seat on fixed bridge, bottom seat on blue cradle", 0, 0, (-16, 16, 30, 52, -36, 2)),
        ("Force feedback: red magnet faces purple Hall sensor", 5, 0, (28, 48, -15, 4, -27, -8)),
        ("Robot interface: top flange proxy and four bolt slots", 78, -42, (-42, 42, -24, 24, -6, 9)),
    ]
    fig = plt.figure(figsize=(13.5, 10.0), dpi=180)
    fig.patch.set_facecolor("white")
    for idx, (title, elev, azim, bounds) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.set_facecolor("white")
        if idx == 3:
            parts = [
                (fixed, "#d7d0bd", 0.22),
                (cradle, "#2471b5", 0.28),
                (hardware["magnet"], "#d62828", 1.0),
                (hardware["hall"], "#6a1b9a", 1.0),
                (hardware["sensor_connector"], "#0f9d8a", 0.95),
                (hardware["hall_vcc_wire"], "#d62828", 1.0),
                (hardware["hall_signal_wire"], "#f2c230", 1.0),
                (hardware["hall_gnd_wire"], "#111111", 1.0),
            ]
        else:
            parts = assembly_parts(fixed, cradle, thinklab, hardware)
        for shape, color, alpha in parts:
            v9.add_shape_to_axis(ax, shape, color, alpha, tolerance=0.95)
        if idx == 3:
            ax.text(38.0, -7.0, -20.0, "magnet", color="#d62828", fontsize=8)
            ax.text(39.0, -5.8, -15.0, "Hall", color="#6a1b9a", fontsize=8)
        v9.set_equal_axes(ax, bounds)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    path = PRESENTATION_DIR / "v10_detail_layout.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def write_actual_cad_motion_video(
    fixed: cq.Workplane,
    cradle: cq.Workplane,
    thinklab: cq.Workplane,
    hardware: dict[str, cq.Workplane],
    rows: list[dict[str, float]],
    p: v9.SimplifiedRailMountParams,
) -> Path | None:
    v9.PRESENTATION_DIR = PRESENTATION_DIR
    old_path = v9.write_actual_cad_motion_video(fixed, cradle, thinklab, hardware, rows, p)
    if old_path is None:
        return None
    new_path = PRESENTATION_DIR / "v10_actual_cad_motion.mp4"
    if new_path.exists():
        new_path.unlink()
    old_path.rename(new_path)
    return new_path


def load_isaac_helpers():
    return v9.load_isaac_helpers()


def write_sim_outputs(metadata: dict) -> dict:
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    helpers = load_isaac_helpers()
    rows = helpers.simulate_step_response(
        metadata,
        duration=3.0,
        dt=1.0 / 120.0,
        approach_speed_mm_s=28.0,
        tissue_stiffness_n_per_mm=0.8,
        tissue_damping_n_s_per_m=12.0,
    )
    trace_path = SIM_DIR / "v10_contact_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = helpers.write_contact_plot(rows, metadata, SIM_DIR)
    if plot_path is not None:
        plot_path.rename(SIM_DIR / "v10_contact_response.png")
    video_path = helpers.write_motion_video(rows, metadata, SIM_DIR, fps=30, max_frames=150)
    if video_path is not None:
        video_path.rename(SIM_DIR / "v10_contact_motion.mp4")

    summary = {
        "variant": RECOMMENDED_MECHANICS["simulation_variant"],
        "output_csv": str(trace_path),
        "output_plot": str(SIM_DIR / "v10_contact_response.png"),
        "output_motion_video": str(SIM_DIR / "v10_contact_motion.mp4"),
        "peak_force_n": max(row["contact_force_n"] for row in rows),
        "final_force_n": rows[-1]["contact_force_n"],
        "max_spring_compression_mm": max(row["spring_compression_mm"] for row in rows),
    }
    (SIM_DIR / "v10_contact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_sim_rows() -> list[dict[str, float]]:
    trace_path = SIM_DIR / "v10_contact_trace.csv"
    with trace_path.open(newline="", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def export_part(part: cq.Workplane, stem: str) -> None:
    exporters.export(part, str(OUT_DIR / f"{stem}.step"))
    exporters.export(part, str(OUT_DIR / f"{stem}.stl"), tolerance=0.14, angularTolerance=0.12)


def make_metadata(
    fixed: cq.Workplane,
    cradle: cq.Workplane,
    sim_summary: dict | None = None,
    presentation_assets: dict | None = None,
) -> dict:
    body_r = P.thinklab_body_d / 2.0 + P.cup_clearance
    outer_r = body_r + P.cup_wall
    clamp_y = -outer_r - 4.5
    magnet_center = [P.guide_rod_x - 5.0, -7.0, P.carriage_z + 10.0]
    hall_center = [P.guide_rod_x - 4.5, -5.8, -15.0]
    nominal_gap = math.dist(magnet_center, hall_center)
    metadata = {
        "units": "mm",
        "params": asdict(P),
        "recommended_mechanics": RECOMMENDED_MECHANICS,
        "print_parts": {
            "fixed_bridge": v9.bbox_dict(fixed),
            "blue_cradle": v9.bbox_dict(cradle),
        },
        "mechanism": {
            "type": "detailed_split_clamp_two_rail_spring",
            "fixed_part_role": "small_robot_bridge_with_visible_interface",
            "blue_cradle_is_main_structural_part": True,
            "force_feedback": "Hall sensor reads the moving red magnet displacement; force is spring calibrated.",
        },
        "clamp_layout": {
            "m3_screw_count": 2,
            "screw_axis": "X",
            "screw_z_offsets_mm": [-4.0, 4.0],
            "screw_hole_d_mm": P.m3_clearance_d,
            "open_split_gap_no_center_bridge": True,
            "nut_pockets_drawn": True,
            "gap_width_mm": P.clamp_gap_w,
            "ear_block_size_mm": [13.0, 12.0, 14.0],
            "clamp_center_y_mm": round(clamp_y, 3),
        },
        "sensor_layout": {
            "magnet_drawn_in_blue_cradle": True,
            "magnet_pocket_drawn": True,
            "magnet_center_mm": magnet_center,
            "hall_drawn_on_fixed_side": True,
            "hall_center_mm": hall_center,
            "nominal_hall_magnet_gap_mm": round(nominal_gap, 3),
            "short_wire_proxy_to_connector_drawn": True,
            "connector_drawn_on_fixed_bridge": True,
        },
        "spring_layout": {
            "spring_count": 1,
            "spring_part": "MISUMI WF6-25 proxy",
            "spring_axis_mm": [P.spring_axis_x, P.spring_axis_y],
            "fixed_top_seat_drawn": True,
            "moving_bottom_seat_drawn": True,
            "visible_coil_proxy_drawn": True,
            "seat_face_gap_at_rest_mm": round(P.spring_fixed_face_z - P.spring_moving_face_z, 3),
        },
        "robot_interface_layout": {
            "openarm_flange_proxy_drawn": True,
            "mount_slot_count": 4,
            "bolt_proxy_count": 4,
            "interface_is_on_top_fixed_bridge": True,
            "slot_pattern_note": "generic adjustable M3 slot pattern; replace with measured OpenArm bolt circle later",
        },
        "contact_layout": {
            "dark_proxy_is_stethoscope": True,
            "blue_part_holds_stethoscope_side_wall": True,
            "only_dark_bottom_contact_face_projects_below_printed_parts": True,
            "contact_face_projects_below_printed_parts_mm": P.contact_face_projection,
            "cradle_inner_diameter_mm": round(body_r * 2.0, 3),
            "rear_cable_window_is_upper_notch_not_second_split": True,
        },
        "clearance_audit": v9.clearance_audit(P),
        "assumptions": [
            "ThinkLabs One approximated as a 62 mm diameter, 24 mm tall circular body.",
            "The top robot flange is a non-print proxy for interface visualization.",
            "The M3 clamp holes, screw proxies, and nut pockets are drawn so the clamp load path is visible.",
            "The Hall connector is a small local connector proxy; the XIAO RP2040 remains external.",
        ],
    }
    if sim_summary is not None:
        metadata["simulation_summary"] = sim_summary
    if presentation_assets is not None:
        metadata["presentation_assets"] = presentation_assets
    return metadata


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    fixed = make_fixed_bridge(P)
    cradle = make_blue_cradle(P)
    thinklab = make_thinklab_proxy(P)
    hardware = make_hardware_proxies(P)

    export_part(fixed, "thinklab_openarm_v10_fixed_bridge_print")
    export_part(cradle, "thinklab_openarm_v10_blue_cradle_print")
    exporters.export(thinklab, str(OUT_DIR / "thinklab_v10_one_envelope_proxy.step"))

    assembly = cq.Assembly(name="thinklab_openarm_v10_detailed_mount")
    colors = {
        "fixed": cq.Color(0.83, 0.80, 0.70),
        "cradle": cq.Color(0.05, 0.34, 0.75),
        "thinklab": cq.Color(0.05, 0.05, 0.05, 0.38),
        "metal": cq.Color(0.72, 0.72, 0.72),
        "black": cq.Color(0.02, 0.02, 0.02),
        "spring": cq.Color(0.2, 0.8, 0.25),
        "magnet": cq.Color(0.84, 0.16, 0.16),
        "hall": cq.Color(0.42, 0.10, 0.60),
        "connector": cq.Color(0.06, 0.62, 0.54),
        "yellow": cq.Color(0.95, 0.76, 0.19),
    }
    assembly.add(fixed, name="printed_fixed_bridge_with_interface_slots", color=colors["fixed"])
    assembly.add(cradle, name="printed_blue_split_clamp_cradle", color=colors["cradle"])
    assembly.add(thinklab, name="thinklab_one_proxy_not_printed", color=colors["thinklab"])
    assembly.add(hardware["robot_flange"], name="openarm_tool_flange_proxy_not_printed", color=colors["metal"])
    assembly.add(hardware["interface_bolts"], name="top_interface_bolts_proxy_not_printed", color=colors["black"])
    assembly.add(hardware["guide_rods"], name="two_guide_rods_proxy_not_printed", color=colors["metal"])
    assembly.add(hardware["spring"], name="wf6_25_visible_coil_proxy_not_printed", color=colors["spring"])
    assembly.add(hardware["clamp_screws"], name="two_m3_clamp_screws_and_nuts_proxy_not_printed", color=colors["black"])
    assembly.add(hardware["magnet"], name="moving_red_magnet_proxy_not_printed", color=colors["magnet"])
    assembly.add(hardware["hall"], name="fixed_purple_hall_sensor_proxy_not_printed", color=colors["hall"])
    assembly.add(hardware["sensor_connector"], name="local_sensor_connector_proxy_not_printed", color=colors["connector"])
    assembly.add(hardware["hall_vcc_wire"], name="hall_vcc_short_wire_proxy_not_printed", color=colors["magnet"])
    assembly.add(hardware["hall_signal_wire"], name="hall_signal_short_wire_proxy_not_printed", color=colors["yellow"])
    assembly.add(hardware["hall_gnd_wire"], name="hall_gnd_short_wire_proxy_not_printed", color=colors["black"])
    assembly.save(str(OUT_DIR / "thinklab_openarm_v10_detailed_assembly.step"), exportType="STEP")

    preview = cq.Compound.makeCompound([fixed.val(), cradle.val(), thinklab.val()] + [shape.val() for shape in hardware.values()])
    exporters.export(cq.Workplane("XY").add(preview), str(OUT_DIR / "thinklab_openarm_v10_detailed_preview.step"))
    render_preview_png(fixed, cradle, thinklab, hardware)

    metadata = make_metadata(fixed, cradle)
    sim_summary = write_sim_outputs(metadata)
    rows = read_sim_rows()
    presentation_assets = {
        "clean_cad_png": str(render_clean_cad_png(fixed, cradle, thinklab, hardware)),
        "detail_layout_png": str(render_detail_layout_png(fixed, cradle, thinklab, hardware)),
    }
    video = write_actual_cad_motion_video(fixed, cradle, thinklab, hardware, rows, P)
    if video is not None:
        presentation_assets["actual_cad_motion_mp4"] = str(video)
    metadata = make_metadata(fixed, cradle, sim_summary, presentation_assets)
    (OUT_DIR / "v10_design_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    calibration = {
        "variant": RECOMMENDED_MECHANICS["simulation_variant"],
        "measurement_type": "spring_compression_to_force",
        "spring_rate_n_per_mm": RECOMMENDED_MECHANICS["spring_rate_n_per_mm"],
        "zero_compression_mm": 0.0,
        "target_force_n": RECOMMENDED_MECHANICS["target_contact_force_n"],
        "max_screened_force_n": RECOMMENDED_MECHANICS["max_screened_contact_force_n"],
        "travel_limit_mm": P.travel,
        "serial_protocol": {
            "preferred_json_line": "{\"t_ms\": 1234, \"disp_mm\": 8.0}",
            "computed_force": "force_n = spring_rate_n_per_mm * max(0, disp_mm - zero_compression_mm)",
        },
    }
    (OUT_DIR / "v10_force_feedback_calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    report = [
        "# V10 Detailed Clamp Mount",
        "",
        "V10 keeps the V9 two-rail architecture but makes the mechanical details explicit.",
        "",
        "## Detail Changes",
        "",
        "- Blue cradle has one open front split; no center blue bridge crosses the gap.",
        "- Two M3 clamp screws and nut pockets are drawn across the split.",
        "- Spring top and bottom seats are visible, with a coil-style spring proxy.",
        "- Red magnet pocket is on the moving blue cradle; purple Hall sensor is on the fixed side.",
        "- A small local connector and short Hall wires show where force-feedback wiring leaves the module.",
        "- A top flange and four bolt proxies show the robot-side interface location.",
        "",
        "## Simulation",
        "",
        f"- Peak force: `{sim_summary['peak_force_n']:.3f} N`",
        f"- Final force: `{sim_summary['final_force_n']:.3f} N`",
        f"- Max compression: `{sim_summary['max_spring_compression_mm']:.3f} mm`",
    ]
    (OUT_DIR / "V10_DETAILED_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({"print_parts": metadata["print_parts"], "clamp": metadata["clamp_layout"], "simulation": sim_summary}, indent=2))


if __name__ == "__main__":
    main()
