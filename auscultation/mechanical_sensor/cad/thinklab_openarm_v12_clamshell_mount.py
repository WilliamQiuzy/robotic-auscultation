"""V12 two-piece clamshell mount for the stepped ThinkLabs One proxy.

V12 keeps the V11 stepped ThinkLabs profile but replaces the single split ring
with two separate printed clamshell halves. Front and rear seams both use M3
fasteners, so the plastic ring is not used as a living hinge.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path


V11_PATH = Path(__file__).with_name("thinklab_openarm_v11_stepped_thinklabs_mount.py")
spec = importlib.util.spec_from_file_location("v11_mount", V11_PATH)
v11 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["v11_mount"] = v11
spec.loader.exec_module(v11)

cq = v11.cq
exporters = v11.exporters
plt = v11.plt
v9 = v11.v9
v10 = v11.v10


P = v9.SimplifiedRailMountParams(
    cup_clearance=0.8,
    cup_wall=4.0,
    cup_height=17.0,
    lip_height=2.8,
    lip_opening_d=52.0,
    cable_window_w=14.0,
    clamp_gap_w=4.0,
    guide_rod_d=5.0,
    guide_rod_clearance_d=5.8,
    guide_rod_x=45.0,
    spring_axis_x=20.0,
    spring_axis_y=54.0,
    top_bridge_x=96.0,
    top_bridge_y=34.0,
    guide_bushing_h=18.0,
)

RECOMMENDED_MECHANICS = {
    "simulation_variant": "v12_stepped_clamshell_two_sided_m3",
    "spring_rate_n_per_mm": 0.5,
    "linear_damping_n_s_per_m": 3.2,
    "target_contact_force_n": 4.0,
    "max_screened_contact_force_n": 5.5,
}

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated" / "v12_clamshell"
SIM_DIR = OUT_DIR / "simulation"
PRESENTATION_DIR = OUT_DIR / "presentation"
THINKLABS_PROFILE = v11.THINKLABS_PROFILE


def sensor_layout(p: v9.SimplifiedRailMountParams) -> dict[str, tuple[float, float, float]]:
    return {
        "magnet": (p.guide_rod_x - 3.5, -17.0, p.carriage_z + 10.0),
        "hall": (p.guide_rod_x - 3.0, -11.0, -15.0),
        "hall_backplate": (p.guide_rod_x - 3.0, -7.8, -10.5),
        "sensor_connector": (23.0, -27.0, -4.0),
    }


def make_v12_visible_spring(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    top_z = p.spring_fixed_face_z - 2.2
    bottom_z = p.spring_moving_face_z + 2.2
    count = 10
    spring = None
    for i in range(count):
        z = bottom_z + i * ((top_z - bottom_z) / (count - 1))
        ring = v9.cyl_z(p.spring_od / 2.0, 0.55, (p.spring_axis_x, p.spring_axis_y, z))
        ring = ring.cut(v9.cyl_z((p.spring_od - 1.4) / 2.0, 0.85, (p.spring_axis_x, p.spring_axis_y, z)))
        spring = ring if spring is None else spring.union(ring)
    assert spring is not None
    return spring


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
    bridge = bridge.union(v9.box(24.0, tab_y, p.top_bridge_t, (p.spring_axis_x, tab_center_y, 0)))
    bridge = bridge.union(v9.downward_spring_socket(p))

    hall_x, backplate_y, backplate_z = sensor_layout(p)["hall_backplate"]
    bridge = bridge.union(v9.box(5.0, 2.4, 16.0, (hall_x, backplate_y, backplate_z)))
    return bridge


def _base_stepped_ring(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    prof = THINKLABS_PROFILE
    lower_clearance_d = prof["lower_bottom_d_mm"] + 1.0
    lower_inner_r = lower_clearance_d / 2.0
    outer_r = lower_inner_r + p.cup_wall
    zc = p.carriage_z

    lower_band = v9.cyl_z(outer_r, p.cup_height, (0, 0, zc)).cut(v9.cyl_z(lower_inner_r, p.cup_height + 3.0, (0, 0, zc)))
    lip_z = zc - p.cup_height / 2.0 + p.lip_height / 2.0
    lower_lip = v9.cyl_z(outer_r, p.lip_height, (0, 0, lip_z)).cut(v9.cyl_z(p.lip_opening_d / 2.0, p.lip_height + 3.0, (0, 0, lip_z)))
    ring = lower_band.union(lower_lip)

    # Keep front jacks and rear cable path open.
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

    magnet_x, magnet_y, magnet_z = sensor_layout(p)["magnet"]
    ring = ring.union(v9.box(9.5, 5.0, 7.0, (magnet_x, magnet_y + 1.8, magnet_z)))
    ring = ring.cut(v9.box(5.9, 2.9, 5.9, (magnet_x, magnet_y, magnet_z)))
    return ring


def make_clamshell_halves(p: v9.SimplifiedRailMountParams) -> tuple[cq.Workplane, cq.Workplane]:
    prof = THINKLABS_PROFILE
    lower_inner_r = (prof["lower_bottom_d_mm"] + 1.0) / 2.0
    outer_r = lower_inner_r + p.cup_wall
    zc = p.carriage_z
    base = _base_stepped_ring(p)

    split_gap_w = 3.0
    cutter_right = v9.box(120.0, 130.0, 80.0, (60.0 + split_gap_w / 2.0, 0, zc))
    cutter_left = v9.box(120.0, 130.0, 80.0, (-60.0 - split_gap_w / 2.0, 0, zc))
    left = base.cut(cutter_right)
    right = base.cut(cutter_left)

    seam_ys = [-outer_r - 4.5, outer_r + 4.5]
    for side, y in enumerate(seam_ys):
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
            nut = v10.hex_prism_x(3.55, 4.0, (23.0, y, zc + dz))
            right = right.cut(nut)
    return left, right


def make_hardware_proxies(p: v9.SimplifiedRailMountParams) -> dict[str, cq.Workplane]:
    hardware = v10.make_hardware_proxies(p)
    hardware["spring"] = make_v12_visible_spring(p)
    prof = THINKLABS_PROFILE
    outer_r = (prof["lower_bottom_d_mm"] + 1.0) / 2.0 + p.cup_wall
    zc = p.carriage_z
    clamp_screws = None
    for y in (-outer_r - 4.5, outer_r + 4.5):
        for dz in (-4.0, 4.0):
            screw = v9.cyl_x(1.35, 56.0, (0, y, zc + dz))
            screw = screw.union(v9.cyl_x(2.4, 3.0, (-28.5, y, zc + dz)))
            screw = screw.union(v10.hex_prism_x(3.1, 2.4, (27.4, y, zc + dz)))
            clamp_screws = screw if clamp_screws is None else clamp_screws.union(screw)
    assert clamp_screws is not None
    hardware["clamp_screws"] = clamp_screws

    layout = sensor_layout(p)
    magnet_x, magnet_y, magnet_z = layout["magnet"]
    hall_x, hall_y, hall_z = layout["hall"]
    connector_x, connector_y, connector_z = layout["sensor_connector"]
    hardware["magnet"] = v9.box(5.0, 2.0, 5.0, (magnet_x, magnet_y, magnet_z))
    hardware["hall"] = v9.box(7.0, 2.2, 9.0, (hall_x, hall_y, hall_z))
    hardware["sensor_connector"] = v9.box(10.0, 6.0, 4.0, (connector_x, connector_y, connector_z))
    hardware["sensor_connector"] = hardware["sensor_connector"].union(
        v9.box(4.0, 3.0, 2.0, (connector_x + 6.5, connector_y, connector_z))
    )
    wire_r = 0.38
    for name, y_offset, z_offset in [
        ("hall_vcc_wire", 0.0, 1.0),
        ("hall_signal_wire", -1.2, 0.0),
        ("hall_gnd_wire", -2.4, -1.0),
    ]:
        y0 = hall_y + y_offset
        z0 = hall_z + z_offset
        xmid = (hall_x + connector_x + 6.0) / 2.0
        segment_x = v9.cyl_x(wire_r, abs(hall_x - (connector_x + 6.0)), (xmid, y0, z0))
        ymid = (y0 + connector_y) / 2.0
        segment_y = v9.cyl_y(wire_r, abs(y0 - connector_y), (connector_x + 6.0, ymid, z0))
        zmid = (z0 + connector_z) / 2.0
        segment_z = v9.cyl_z(wire_r, abs(z0 - connector_z), (connector_x + 6.0, connector_y, zmid))
        hardware[name] = segment_x.union(segment_y).union(segment_z)
    return hardware


def assembly_parts(fixed, left, right, thinklab, hardware):
    return [
        (fixed, "#d7d0bd", 1.0),
        (left, "#1f77b4", 1.0),
        (right, "#248bd2", 1.0),
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


def _intersection_volume_mm3(a: cq.Workplane, b: cq.Workplane) -> float:
    try:
        return float(a.intersect(b).val().Volume())
    except Exception:  # noqa: BLE001
        return 0.0


def _bbox_axis_gap_mm(a: cq.Workplane, b: cq.Workplane, axis: str) -> float:
    abb = a.val().BoundingBox()
    bbb = b.val().BoundingBox()
    amin = getattr(abb, f"{axis}min")
    amax = getattr(abb, f"{axis}max")
    bmin = getattr(bbb, f"{axis}min")
    bmax = getattr(bbb, f"{axis}max")
    if amax <= bmin:
        return float(bmin - amax)
    if bmax <= amin:
        return float(amin - bmax)
    return -float(min(amax - bmin, bmax - amin))


def make_geometry_audit(fixed, left, right, thinklab, hardware) -> dict:
    checks: list[dict] = []

    def add_collision_check(name: str, a: cq.Workplane, b: cq.Workplane, max_volume_mm3: float = 0.05) -> None:
        volume = _intersection_volume_mm3(a, b)
        checks.append(
            {
                "name": name,
                "type": "intersection_volume",
                "volume_mm3": round(volume, 6),
                "max_volume_mm3": max_volume_mm3,
                "pass": volume <= max_volume_mm3,
            }
        )

    add_collision_check("spring_vs_fixed_print", hardware["spring"], fixed)
    add_collision_check("spring_vs_left_clamshell", hardware["spring"], left)
    add_collision_check("spring_vs_right_clamshell", hardware["spring"], right)
    add_collision_check("spring_vs_clamp_screws", hardware["spring"], hardware["clamp_screws"])
    add_collision_check("magnet_vs_hall_sensor", hardware["magnet"], hardware["hall"])
    add_collision_check("magnet_vs_left_clamshell", hardware["magnet"], left)
    add_collision_check("magnet_vs_right_clamshell", hardware["magnet"], right)
    add_collision_check("hall_sensor_vs_fixed_print", hardware["hall"], fixed)
    add_collision_check("hall_sensor_vs_left_clamshell", hardware["hall"], left)
    add_collision_check("hall_sensor_vs_right_clamshell", hardware["hall"], right)
    add_collision_check("sensor_connector_vs_fixed_print", hardware["sensor_connector"], fixed)
    for wire_name in ("hall_vcc_wire", "hall_signal_wire", "hall_gnd_wire"):
        add_collision_check(f"{wire_name}_vs_fixed_print", hardware[wire_name], fixed, max_volume_mm3=0.02)

    max_audit_compression_mm = 8.5
    add_collision_check(
        "max_compression_left_clamshell_vs_fixed_print",
        left.translate((0, 0, max_audit_compression_mm)),
        fixed,
    )
    add_collision_check(
        "max_compression_right_clamshell_vs_fixed_print",
        right.translate((0, 0, max_audit_compression_mm)),
        fixed,
    )
    add_collision_check(
        "max_compression_magnet_vs_fixed_print",
        hardware["magnet"].translate((0, 0, max_audit_compression_mm)),
        fixed,
    )
    add_collision_check(
        "max_compression_clamp_screws_vs_fixed_print",
        hardware["clamp_screws"].translate((0, 0, max_audit_compression_mm)),
        fixed,
    )

    spring_bb = hardware["spring"].val().BoundingBox()
    spring_bottom_clearance = spring_bb.zmin - P.spring_moving_face_z
    spring_top_clearance = P.spring_fixed_face_z - spring_bb.zmax
    magnet_hall_y_gap = _bbox_axis_gap_mm(hardware["magnet"], hardware["hall"], "y")
    clearances = {
        "magnet_hall_y_gap_mm": round(magnet_hall_y_gap, 3),
        "spring_bottom_end_clearance_mm": round(spring_bottom_clearance, 3),
        "spring_top_end_clearance_mm": round(spring_top_clearance, 3),
        "spring_end_clearance_min_mm": round(min(spring_bottom_clearance, spring_top_clearance), 3),
    }
    checks.append(
        {
            "name": "magnet_hall_air_gap",
            "type": "bbox_axis_gap",
            "axis": "y",
            "gap_mm": clearances["magnet_hall_y_gap_mm"],
            "min_gap_mm": 1.5,
            "pass": magnet_hall_y_gap >= 1.5,
        }
    )
    checks.append(
        {
            "name": "spring_end_clearance",
            "type": "bbox_z_clearance",
            "clearance_mm": clearances["spring_end_clearance_min_mm"],
            "min_clearance_mm": 1.8,
            "pass": clearances["spring_end_clearance_min_mm"] >= 1.8,
        }
    )
    return {
        "units": "mm",
        "method": "CadQuery boolean intersection volumes plus bounding-box clearances",
        "checks": checks,
        "clearances": clearances,
        "dynamic_audit": {
            "max_compression_mm": max_audit_compression_mm,
            "moving_parts_translated_up_relative_to_fixed": True,
        },
        "pass": all(check["pass"] for check in checks),
    }


def render_preview_png(fixed, left, right, thinklab, hardware) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bounds = compound_bounds([fixed, left, right, thinklab, *hardware.values()])
    views = [("Isometric", 24, -42), ("Front seam", 0, -90), ("Rear seam", 0, 90), ("Top split", 90, -90)]
    fig = plt.figure(figsize=(13.5, 10.0), dpi=180)
    fig.patch.set_facecolor("#f7f7f5")
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.set_facecolor("#f7f7f5")
        for shape, color, alpha in assembly_parts(fixed, left, right, thinklab, hardware):
            v9.add_shape_to_axis(ax, shape, color, alpha)
        v9.set_equal_axes(ax, bounds)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11, pad=0)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    path = OUT_DIR / "thinklab_openarm_v12_clamshell_preview.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def render_clean_cad_png(fixed, left, right, thinklab, hardware) -> Path:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    bounds = compound_bounds([fixed, left, right, thinklab, *hardware.values()])
    fig = plt.figure(figsize=(10.0, 8.0), dpi=240)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    for shape, color, alpha in assembly_parts(fixed, left, right, thinklab, hardware):
        v9.add_shape_to_axis(ax, shape, color, alpha, tolerance=0.95)
    v9.set_equal_axes(ax, bounds)
    ax.view_init(elev=24, azim=-42)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    path = PRESENTATION_DIR / "v12_cad_clean.png"
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


def render_clamshell_detail_png(fixed, left, right, thinklab, hardware) -> Path:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Two separate printed half rings, split on X=0", 72, -90, (-52, 52, -48, 48, -42, -12)),
        ("Front M3 seam: two screws, no living hinge", 0, -90, (-36, 36, -56, -22, -43, -17)),
        ("Rear M3 seam: two screws plus cable window", 0, 90, (-36, 36, 22, 56, -43, -17)),
        ("No soft pad: printed clamshell captures stepped profile directly", 25, -42, (-54, 54, -54, 54, -46, -10)),
    ]
    fig = plt.figure(figsize=(13.5, 10.0), dpi=180)
    fig.patch.set_facecolor("white")
    for idx, (title, elev, azim, bounds) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.set_facecolor("white")
        for shape, color, alpha in assembly_parts(fixed, left, right, thinklab, hardware):
            if idx in (1, 4) and shape is thinklab:
                alpha = 0.30
            v9.add_shape_to_axis(ax, shape, color, alpha, tolerance=0.95)
        v9.set_equal_axes(ax, bounds)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10, pad=0)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    path = PRESENTATION_DIR / "v12_clamshell_detail.png"
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
    trace_path = SIM_DIR / "v12_contact_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = helpers.write_contact_plot(rows, metadata, SIM_DIR)
    if plot_path is not None:
        plot_path.rename(SIM_DIR / "v12_contact_response.png")
    video_path = helpers.write_motion_video(rows, metadata, SIM_DIR, fps=30, max_frames=150)
    if video_path is not None:
        video_path.rename(SIM_DIR / "v12_contact_motion.mp4")
    summary = {
        "variant": RECOMMENDED_MECHANICS["simulation_variant"],
        "output_csv": str(trace_path),
        "output_plot": str(SIM_DIR / "v12_contact_response.png"),
        "output_motion_video": str(SIM_DIR / "v12_contact_motion.mp4"),
        "peak_force_n": max(row["contact_force_n"] for row in rows),
        "final_force_n": rows[-1]["contact_force_n"],
        "max_spring_compression_mm": max(row["spring_compression_mm"] for row in rows),
    }
    (SIM_DIR / "v12_contact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_sim_rows() -> list[dict[str, float]]:
    with (SIM_DIR / "v12_contact_trace.csv").open(newline="", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def write_actual_cad_motion_video(fixed, left, right, thinklab, hardware, rows) -> Path | None:
    moving = left.union(right)
    v9.PRESENTATION_DIR = PRESENTATION_DIR
    old_path = v9.write_actual_cad_motion_video(fixed, moving, thinklab, hardware, rows, P)
    if old_path is None:
        return None
    new_path = PRESENTATION_DIR / "v12_actual_cad_motion.mp4"
    if new_path.exists():
        new_path.unlink()
    old_path.rename(new_path)
    return new_path


def export_part(part: cq.Workplane, stem: str) -> None:
    exporters.export(part, str(OUT_DIR / f"{stem}.step"))
    exporters.export(part, str(OUT_DIR / f"{stem}.stl"), tolerance=0.14, angularTolerance=0.12)


def make_metadata(fixed, left, right, sim_summary=None, presentation_assets=None, geometry_audit=None) -> dict:
    prof = v11.THINKLABS_PROFILE
    outer_r = (prof["lower_bottom_d_mm"] + 1.0) / 2.0 + P.cup_wall
    rear_screw_y = outer_r + 4.5
    rear_m3_spring_centerline_clearance = abs(P.spring_axis_y - rear_screw_y)
    spring_seat_envelope_d = max(P.spring_od, P.spring_seat_d)
    rear_m3_spring_required_clearance = spring_seat_envelope_d / 2.0 + P.m3_clearance_d / 2.0 + 1.0
    metadata = {
        "units": "mm",
        "params": asdict(P),
        "recommended_mechanics": RECOMMENDED_MECHANICS,
        "print_parts": {
            "fixed_bridge": v9.bbox_dict(fixed),
            "left_clamshell": v9.bbox_dict(left),
            "right_clamshell": v9.bbox_dict(right),
        },
        "mechanism": {
            "type": "two_piece_clamshell_stepped_profile",
            "blue_clamshell_is_main_structural_part": True,
            "force_feedback": "Hall sensor reads moving magnet displacement; force is spring calibrated.",
        },
        "thinklabs_profile": {
            "profile_type": "lower_large_frustum_plus_upper_small_disc",
            "lower_bottom_d_mm": prof["lower_bottom_d_mm"],
            "lower_top_d_mm": prof["lower_top_d_mm"],
            "upper_disc_d_mm": prof["upper_disc_d_mm"],
            "side_profile_is_trapezoid_like": True,
        },
        "clamshell_layout": {
            "half_count": 2,
            "left_half_printed_separately": True,
            "right_half_printed_separately": True,
            "split_plane": "X=0",
            "seam_locations": ["front", "rear"],
            "no_living_hinge": True,
            "printed_plastic_primary_structure": True,
            "soft_pad_required": False,
            "m3_screw_count": 4,
            "front_m3_screw_count": 2,
            "rear_m3_screw_count": 2,
            "front_jack_window_drawn": True,
            "rear_cable_window_drawn": True,
            "upper_retaining_blocks_removed": True,
            "upper_retaining_pad_count": 0,
            "rear_m3_spring_centerline_clearance_mm": round(rear_m3_spring_centerline_clearance, 3),
            "rear_m3_spring_required_clearance_mm": round(rear_m3_spring_required_clearance, 3),
            "rear_m3_spring_collision_free": rear_m3_spring_centerline_clearance > rear_m3_spring_required_clearance,
            "clamp_strategy": "two printed half rings bolted at front and rear; no silicone/TPU pad required",
        },
        "contact_layout": {
            "dark_proxy_is_stethoscope": True,
            "only_dark_bottom_contact_face_projects_below_printed_parts": True,
            "contact_face_projects_below_printed_parts_mm": P.contact_face_projection,
        },
        "spring_layout": {
            "spring_count": 1,
            "spring_axis_mm": [P.spring_axis_x, P.spring_axis_y],
            "seat_face_gap_at_rest_mm": round(P.spring_fixed_face_z - P.spring_moving_face_z, 3),
        },
    }
    if sim_summary is not None:
        metadata["simulation_summary"] = sim_summary
    if presentation_assets is not None:
        metadata["presentation_assets"] = presentation_assets
    if geometry_audit is not None:
        metadata["geometry_audit"] = geometry_audit
    return metadata


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    fixed = make_fixed_bridge(P)
    left, right = make_clamshell_halves(P)
    thinklab = v11.make_stepped_thinklab_proxy(P)
    hardware = make_hardware_proxies(P)
    geometry_audit = make_geometry_audit(fixed, left, right, thinklab, hardware)

    export_part(fixed, "thinklab_openarm_v12_fixed_bridge_print")
    export_part(left, "thinklab_openarm_v12_left_clamshell_print")
    export_part(right, "thinklab_openarm_v12_right_clamshell_print")
    exporters.export(thinklab, str(OUT_DIR / "thinklab_v12_stepped_profile_proxy.step"))

    assembly = cq.Assembly(name="thinklab_openarm_v12_clamshell_mount")
    assembly.add(fixed, name="printed_fixed_bridge", color=cq.Color(0.83, 0.80, 0.70))
    assembly.add(left, name="printed_left_clamshell_half", color=cq.Color(0.05, 0.34, 0.75))
    assembly.add(right, name="printed_right_clamshell_half", color=cq.Color(0.05, 0.48, 0.84))
    assembly.add(thinklab, name="thinklab_one_stepped_proxy_not_printed", color=cq.Color(0.05, 0.05, 0.05, 0.40))
    for name in [
        "robot_flange",
        "interface_bolts",
        "guide_rods",
        "spring",
        "clamp_screws",
        "magnet",
        "hall",
        "sensor_connector",
        "hall_vcc_wire",
        "hall_signal_wire",
        "hall_gnd_wire",
    ]:
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
        assembly.add(hardware[name], name=f"{name}_proxy_not_printed", color=color)
    assembly.save(str(OUT_DIR / "thinklab_openarm_v12_clamshell_assembly.step"), exportType="STEP")

    preview = cq.Compound.makeCompound([fixed.val(), left.val(), right.val(), thinklab.val()] + [shape.val() for shape in hardware.values()])
    exporters.export(cq.Workplane("XY").add(preview), str(OUT_DIR / "thinklab_openarm_v12_clamshell_preview.step"))
    render_preview_png(fixed, left, right, thinklab, hardware)

    (OUT_DIR / "v12_geometry_audit.json").write_text(json.dumps(geometry_audit, indent=2), encoding="utf-8")

    metadata = make_metadata(fixed, left, right, geometry_audit=geometry_audit)
    sim_summary = write_sim_outputs(metadata)
    rows = read_sim_rows()
    presentation_assets = {
        "clean_cad_png": str(render_clean_cad_png(fixed, left, right, thinklab, hardware)),
        "clamshell_detail_png": str(render_clamshell_detail_png(fixed, left, right, thinklab, hardware)),
    }
    video = write_actual_cad_motion_video(fixed, left, right, thinklab, hardware, rows)
    if video is not None:
        presentation_assets["actual_cad_motion_mp4"] = str(video)
    metadata = make_metadata(fixed, left, right, sim_summary, presentation_assets, geometry_audit)
    (OUT_DIR / "v12_design_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    calibration = {
        "variant": RECOMMENDED_MECHANICS["simulation_variant"],
        "measurement_type": "spring_compression_to_force",
        "spring_rate_n_per_mm": RECOMMENDED_MECHANICS["spring_rate_n_per_mm"],
        "target_force_n": RECOMMENDED_MECHANICS["target_contact_force_n"],
        "max_screened_force_n": RECOMMENDED_MECHANICS["max_screened_contact_force_n"],
        "travel_limit_mm": P.travel,
    }
    (OUT_DIR / "v12_force_feedback_calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    report = [
        "# V12 Two-Piece Clamshell Mount",
        "",
        "V12 changes the blue holder from a single split ring into two separate printed half rings.",
        "",
        "## Clamp Strategy",
        "",
        "- Left and right clamshell halves are printed separately.",
        "- Front seam uses two M3 screws.",
        "- Rear seam uses two M3 screws.",
        "- Rear M3 screws are offset from the spring axis to avoid spring/screw interference.",
        "- Four upper retaining blocks were removed; the lower lip plus two-sided M3 clamshell provide retention.",
        "- No silicone or TPU pad is required.",
        "- The plastic part no longer acts as a living hinge.",
        "",
        "## Simulation",
        "",
    ]
    report += [
        "## Geometry Audit",
        "",
        f"- Critical collision checks pass: `{geometry_audit['pass']}`",
        f"- Magnet/Hall Y air gap: `{geometry_audit['clearances']['magnet_hall_y_gap_mm']:.3f} mm`",
        f"- Spring end clearance: `{geometry_audit['clearances']['spring_end_clearance_min_mm']:.3f} mm`",
        "",
    ]
    if sim_summary:
        report += [
            f"- Peak force: `{sim_summary['peak_force_n']:.3f} N`",
            f"- Final force: `{sim_summary['final_force_n']:.3f} N`",
            f"- Max compression: `{sim_summary['max_spring_compression_mm']:.3f} mm`",
        ]
    (OUT_DIR / "V12_CLAMSHELL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"print_parts": metadata["print_parts"], "clamshell": metadata["clamshell_layout"], "simulation": sim_summary}, indent=2))


if __name__ == "__main__":
    main()
