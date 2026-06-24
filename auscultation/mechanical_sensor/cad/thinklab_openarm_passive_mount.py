"""Parametric passive force-compliance mount for ThinkLabs One + OpenArm arm.

Units are millimeters.  The generated STEP files are first-pass geometry for
mechanical review and sizing, not final clinical hardware.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("XDG_CACHE_HOME", str(tmp_root / "cadquery-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "matplotlib-cache"))

import cadquery as cq
from cadquery import exporters

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass(frozen=True)
class MountParams:
    # ThinkLabs One proxy / holder assumptions.
    thinklab_body_d: float = 62.0
    thinklab_body_h: float = 24.0
    cup_clearance: float = 0.8
    cup_wall: float = 6.0
    cup_height: float = 20.0
    lip_height: float = 3.0
    lip_opening_d: float = 52.0
    clamp_gap_w: float = 12.0
    cable_window_w: float = 18.0

    # Passive mechanism.
    travel: float = 24.0
    preload_gap: float = 42.0
    guide_rod_d: float = 6.0
    guide_clearance: float = 0.45
    guide_x: float = 42.0
    guide_y: float = 16.0
    guide_bearing_od: float = 17.0
    guide_bearing_h: float = 30.0
    spring_seat_d: float = 18.0
    spring_pilot_d: float = 7.0
    contact_face_projection: float = 16.0

    # Fixed frame / robot side.
    top_plate_x: float = 108.0
    top_plate_y: float = 70.0
    top_plate_t: float = 8.0
    side_post_x: float = 52.0
    fixed_frame_lower_z: float = -34.0
    guide_rod_bottom_z: float = -82.0
    side_post_w: float = 8.0
    side_post_d: float = 18.0

    # Moving carrier location in the assembly preview.
    carriage_z: float = -62.0

    # Fasteners / mounting.
    m3_clearance_d: float = 3.5
    m2_clearance_d: float = 2.3
    slot_len: float = 13.0
    slot_w: float = 3.7


P = MountParams()
RECOMMENDED_MECHANICS = {
    "simulation_variant": "v6_balanced_high_clearance",
    "spring_rate_n_per_mm": 0.38,
    "linear_damping_n_s_per_m": 4.2,
    "target_contact_force_n": 4.0,
    "max_screened_contact_force_n": 8.0,
}
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated"


def box(x: float, y: float, z: float, center: tuple[float, float, float]) -> cq.Workplane:
    return cq.Workplane("XY").box(x, y, z).translate(center)


def cyl_z(radius: float, height: float, center: tuple[float, float, float]) -> cq.Workplane:
    x, y, z = center
    return cq.Workplane("XY").circle(radius).extrude(height).translate((x, y, z - height / 2.0))


def cyl_x(radius: float, height: float, center: tuple[float, float, float]) -> cq.Workplane:
    x, y, z = center
    return cq.Workplane("YZ").circle(radius).extrude(height).translate((x - height / 2.0, y, z))


def cyl_y(radius: float, height: float, center: tuple[float, float, float]) -> cq.Workplane:
    x, y, z = center
    return cq.Workplane("XZ").circle(radius).extrude(height).translate((x, y - height / 2.0, z))


def slot_cut_z(
    length: float,
    width: float,
    height: float,
    center: tuple[float, float, float],
    angle_deg: float = 0.0,
) -> cq.Workplane:
    """Capsule-shaped vertical cut volume."""
    straight_len = max(0.0, length - width)
    slot = box(straight_len, width, height, (0, 0, 0))
    slot = slot.union(cyl_z(width / 2.0, height, (-straight_len / 2.0, 0, 0)))
    slot = slot.union(cyl_z(width / 2.0, height, (straight_len / 2.0, 0, 0)))
    return slot.rotate((0, 0, 0), (0, 0, 1), angle_deg).translate(center)


def make_fixed_frame(p: MountParams) -> cq.Workplane:
    """Printed stationary part: robot-side plate, guide supports, spring seat."""
    plate = box(p.top_plate_x, p.top_plate_y, p.top_plate_t, (0, 0, 0))

    # Universal robot-side mounting: M3 slots plus smaller M2 holes for gripper-style adapters.
    for sx in (-18.0, 18.0):
        for sy in (-18.0, 18.0):
            plate = plate.cut(slot_cut_z(p.slot_len, p.slot_w, p.top_plate_t + 4.0, (sx, sy, 0), 0))
    for sx in (-33.0, 33.0):
        plate = plate.cut(slot_cut_z(16.0, p.slot_w, p.top_plate_t + 4.0, (sx, 0, 0), 90))
    for sx in (-11.0, 11.0):
        for sy in (-28.0, 28.0):
            plate = plate.cut(cyl_z(p.m2_clearance_d / 2.0, p.top_plate_t + 4.0, (sx, sy, 0)))

    # Central spring preload screw / cable pass-through.
    plate = plate.cut(cyl_z(2.7, p.top_plate_t + 5.0, (0, 0, 0)))

    frame = plate

    # Top guide rod bosses.
    for sx in (-p.guide_x, p.guide_x):
        frame = frame.union(cyl_z(8.5, 10.0, (sx, p.guide_y, -9.0)))

    top_plate_bottom_z = -p.top_plate_t / 2.0
    post_h = top_plate_bottom_z - p.fixed_frame_lower_z
    post_z = (top_plate_bottom_z + p.fixed_frame_lower_z) / 2.0
    for sx in (-p.side_post_x, p.side_post_x):
        frame = frame.union(box(p.side_post_w, p.side_post_d, post_h, (sx, p.guide_y, post_z)))

    guide_cut_r = (p.guide_rod_d + p.guide_clearance) / 2.0
    for sx in (-p.guide_x, p.guide_x):
        frame = frame.cut(cyl_z(guide_cut_r, 28.0, (sx, p.guide_y, -5.0)))

    # Spring seat on the fixed side.
    frame = frame.union(cyl_z(p.spring_seat_d / 2.0, 6.0, (0, 0, -7.0)))
    frame = frame.cut(cyl_z(p.spring_pilot_d / 2.0, 18.0, (0, 0, -5.0)))

    # Side mount for a linear displacement sensor; pair with the magnet pocket on the moving carrier.
    rail = box(7.0, 12.0, 46.0, (p.side_post_x + 6.0, -22.0, -43.0))
    rail = rail.cut(cyl_x(p.m2_clearance_d / 2.0, 12.0, (p.side_post_x + 6.0, -22.0, -27.0)))
    rail = rail.cut(cyl_x(p.m2_clearance_d / 2.0, 12.0, (p.side_post_x + 6.0, -22.0, -59.0)))
    frame = frame.union(rail)
    frame = frame.union(box(7.0, 40.0, 8.0, (p.side_post_x + 4.5, -3.0, -38.0)))

    return frame


def make_floating_carriage(p: MountParams) -> cq.Workplane:
    """Printed moving part: ThinkLabs side cradle, guide bushings, spring seat."""
    body_r = p.thinklab_body_d / 2.0 + p.cup_clearance
    outer_r = body_r + p.cup_wall
    zc = p.carriage_z

    side_ring = cyl_z(outer_r, p.cup_height, (0, 0, zc)).cut(
        cyl_z(body_r, p.cup_height + 3.0, (0, 0, zc))
    )
    lip_z = zc - p.cup_height / 2.0 + p.lip_height / 2.0
    lip = cyl_z(outer_r, p.lip_height, (0, 0, lip_z)).cut(
        cyl_z(p.lip_opening_d / 2.0, p.lip_height + 3.0, (0, 0, lip_z))
    )
    carrier = side_ring.union(lip)

    # Front split for clamp compliance and a rear cable/window opening.
    carrier = carrier.cut(box(p.clamp_gap_w, outer_r + 14.0, p.cup_height + 5.0, (0, -outer_r, zc)))
    carrier = carrier.cut(box(p.cable_window_w, outer_r + 12.0, p.cup_height + 5.0, (0, outer_r, zc + 2.0)))

    # Clamp ears with one M3 pinch screw across the split.
    lug_z = zc + 2.0
    lug_y = -outer_r - 3.0
    for sx in (-12.0, 12.0):
        carrier = carrier.union(box(12.0, 14.0, 10.0, (sx, lug_y, lug_z)))
    carrier = carrier.cut(cyl_x(p.m3_clearance_d / 2.0, 44.0, (0, lug_y, lug_z)))

    # Sliding guide-bushing bosses.
    bushing_r = p.guide_bearing_od / 2.0
    guide_cut_r = (p.guide_rod_d + p.guide_clearance) / 2.0
    for sx in (-p.guide_x, p.guide_x):
        carrier = carrier.union(cyl_z(bushing_r, p.guide_bearing_h, (sx, p.guide_y, zc)))
        carrier = carrier.union(box(12.0, 12.0, 12.0, (sx * 0.92, p.guide_y * 0.86, zc)))
        carrier = carrier.cut(cyl_z(guide_cut_r, p.guide_bearing_h + 8.0, (sx, p.guide_y, zc)))

    # Moving spring seat and central post clearance.
    seat_z = zc + p.cup_height / 2.0 + 1.0
    carrier = carrier.union(box(outer_r * 2.0, 8.0, 6.0, (0, 0, seat_z - 2.0)))
    carrier = carrier.union(cyl_z(p.spring_seat_d / 2.0, 6.0, (0, 0, seat_z)))
    carrier = carrier.cut(cyl_z(p.spring_pilot_d / 2.0, 12.0, (0, 0, seat_z)))

    # Small magnet pocket for displacement/force calibration.
    pocket = box(10.0, 4.0, 8.0, (p.side_post_x + 1.0, -22.0, zc))
    pocket = pocket.cut(box(6.0, 2.4, 5.0, (p.side_post_x + 1.0, -24.1, zc)))
    carrier = carrier.union(pocket)
    carrier = carrier.union(box(14.0, 38.0, 7.0, (p.side_post_x - 5.0, -3.0, zc)))

    return carrier


def make_thinklab_proxy(p: MountParams) -> cq.Workplane:
    """Non-print proxy showing the assumed ThinkLabs One envelope."""
    zc = p.carriage_z - 1.0
    body_bottom_z = p.carriage_z - p.cup_height / 2.0 - 0.5
    contact_h = 2.5
    contact_bottom_z = body_bottom_z - p.contact_face_projection
    contact_center_z = contact_bottom_z + contact_h / 2.0
    neck_h = body_bottom_z - (contact_bottom_z + contact_h)
    neck_center_z = contact_bottom_z + contact_h + neck_h / 2.0
    lower = cyl_z(p.thinklab_body_d / 2.0, 15.0, (0, 0, zc - 2.0))
    upper = cyl_z((p.thinklab_body_d - 7.0) / 2.0, 9.0, (0, 0, zc + 8.0))
    neck = cyl_z(23.0, neck_h, (0, 0, neck_center_z))
    contact = cyl_z(26.0, contact_h, (0, 0, contact_center_z))
    side_port = cyl_y(2.8, 12.0, (0, p.thinklab_body_d / 2.0 + 0.5, zc + 2.0))
    return lower.union(upper).union(neck).union(contact).union(side_port)


def make_hardware_proxy(p: MountParams) -> tuple[cq.Workplane, cq.Workplane]:
    """Non-print rods and spring envelope."""
    rods = None
    rod_top_z = 4.0
    rod_h = rod_top_z - p.guide_rod_bottom_z
    rod_z = (rod_top_z + p.guide_rod_bottom_z) / 2.0
    for sx in (-p.guide_x, p.guide_x):
        rod = cyl_z(p.guide_rod_d / 2.0, rod_h, (sx, p.guide_y, rod_z))
        rods = rod if rods is None else rods.union(rod)

    spring = cyl_z(8.0, p.preload_gap, (0, 0, -31.0)).cut(cyl_z(4.0, p.preload_gap + 2.0, (0, 0, -31.0)))
    return rods, spring


def export_part(part: cq.Workplane, stem: str) -> None:
    exporters.export(part, str(OUT_DIR / f"{stem}.step"))
    exporters.export(part, str(OUT_DIR / f"{stem}.stl"), tolerance=0.15, angularTolerance=0.12)


def bbox_dict(shape: cq.Workplane) -> dict[str, float]:
    bb = shape.val().BoundingBox()
    return {
        "xlen": round(bb.xlen, 3),
        "ylen": round(bb.ylen, 3),
        "zlen": round(bb.zlen, 3),
        "xmin": round(bb.xmin, 3),
        "xmax": round(bb.xmax, 3),
        "ymin": round(bb.ymin, 3),
        "ymax": round(bb.ymax, 3),
        "zmin": round(bb.zmin, 3),
        "zmax": round(bb.zmax, 3),
    }


def add_shape_to_axis(
    ax,
    shape: cq.Workplane,
    color: str,
    alpha: float = 1.0,
    tolerance: float = 1.2,
) -> None:
    vertices, triangles = shape.val().tessellate(tolerance)
    pts = [(v.x, v.y, v.z) for v in vertices]
    faces = [[pts[i] for i in tri] for tri in triangles]
    mesh = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=(0.18, 0.18, 0.18, 0.08),
        linewidths=0.08,
        alpha=alpha,
    )
    mesh.set_zsort("average")
    ax.add_collection3d(mesh)


def set_equal_axes(ax, bounds: tuple[float, float, float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
    x_mid = (xmax + xmin) / 2.0
    y_mid = (ymax + ymin) / 2.0
    z_mid = (zmax + zmin) / 2.0
    pad = max_range * 0.08
    half = max_range / 2.0 + pad
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def render_preview_png(
    fixed: cq.Workplane,
    carrier: cq.Workplane,
    thinklab: cq.Workplane,
    rods: cq.Workplane,
    spring: cq.Workplane,
) -> None:
    """Render a quick multi-view PNG so CAD changes are visually inspectable."""
    preview = cq.Compound.makeCompound([fixed.val(), carrier.val(), thinklab.val(), rods.val(), spring.val()])
    bb = preview.BoundingBox()
    bounds = (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax)

    parts = [
        (fixed, "#d7d0bd", 1.0),
        (carrier, "#2471b5", 1.0),
        (thinklab, "#202326", 0.42),
        (rods, "#8e9298", 0.8),
        (spring, "#47b567", 0.42),
    ]
    views = [
        ("Isometric", 24, -42),
        ("Front", 0, -90),
        ("Side", 0, 0),
        ("Top", 90, -90),
    ]

    fig = plt.figure(figsize=(13.5, 10.0), dpi=180)
    fig.patch.set_facecolor("#f7f7f5")
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.set_facecolor("#f7f7f5")
        for shape, color, alpha in parts:
            add_shape_to_axis(ax, shape, color, alpha)
        set_equal_axes(ax, bounds)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11, pad=0)

    fig.text(0.5, 0.965, "ThinkLabs One + OpenArm passive force mount v6", ha="center", fontsize=15)
    fig.text(
        0.5,
        0.035,
        "Tan: fixed robot-side frame | Blue: floating side cradle | Dark: ThinkLabs envelope/contact face | Gray/green: rods + spring proxies",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.945))
    fig.savefig(OUT_DIR / "thinklab_openarm_passive_force_mount_preview.png")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed = make_fixed_frame(P)
    carrier = make_floating_carriage(P)
    thinklab = make_thinklab_proxy(P)
    rods, spring = make_hardware_proxy(P)

    export_part(fixed, "thinklab_openarm_fixed_frame_print")
    export_part(carrier, "thinklab_openarm_floating_carriage_print")
    exporters.export(thinklab, str(OUT_DIR / "thinklab_one_envelope_proxy.step"))

    assy = cq.Assembly(name="thinklab_openarm_passive_force_mount_v6")
    assy.add(fixed, name="printed_fixed_frame", color=cq.Color(0.83, 0.80, 0.70))
    assy.add(carrier, name="printed_floating_carriage", color=cq.Color(0.05, 0.34, 0.75))
    assy.add(thinklab, name="thinklab_one_proxy_not_printed", color=cq.Color(0.05, 0.05, 0.05, 0.35))
    assy.add(rods, name="six_mm_guide_rods_proxy_not_printed", color=cq.Color(0.55, 0.55, 0.58))
    assy.add(spring, name="compression_spring_envelope_proxy_not_printed", color=cq.Color(0.2, 0.8, 0.25, 0.35))
    assy.save(str(OUT_DIR / "thinklab_openarm_passive_force_mount_assembly.step"), exportType="STEP")

    preview = cq.Compound.makeCompound([fixed.val(), carrier.val(), thinklab.val(), rods.val(), spring.val()])
    exporters.export(cq.Workplane("XY").add(preview), str(OUT_DIR / "thinklab_openarm_passive_force_mount_preview.step"))
    render_preview_png(fixed, carrier, thinklab, rods, spring)

    metadata = {
        "units": "mm",
        "params": asdict(P),
        "recommended_mechanics": RECOMMENDED_MECHANICS,
        "print_parts": {
            "fixed_frame": bbox_dict(fixed),
            "floating_carriage": bbox_dict(carrier),
        },
        "visualization": "thinklab_openarm_passive_force_mount_preview.png",
        "contact_layout": {
            "dark_proxy_is_stethoscope": True,
            "blue_part_holds_stethoscope_side_wall": True,
            "skin_contact_is_bottom_dark_circular_face": True,
            "contact_face_projects_below_printed_parts_mm": P.contact_face_projection,
        },
        "assumptions": [
            "ThinkLabs One approximated as a 62 mm diameter, 24 mm tall circular body.",
            "Top plate uses generic M3 slots and M2 holes; replace with exact OpenArm flange pattern after measuring the end effector.",
            "Two 6 mm guide rods plus one compression spring create the passive force-compliance axis.",
            "Side sensor rail and moving magnet pocket are placeholders for a linear displacement sensor.",
        ],
    }
    (OUT_DIR / "design_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata["print_parts"], indent=2))
    print(f"Wrote CAD outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
