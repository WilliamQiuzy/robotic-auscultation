"""Compact v8B mini-spring mount for ThinkLabs One + OpenArm arm.

V8B removes the v6 pair of 6 mm guide rods. The printed frame provides a short
anti-rotation slide and a small standard compression spring provides the
calibrated passive force.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
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
class CompactMountParams:
    thinklab_body_d: float = 62.0
    thinklab_body_h: float = 24.0
    cup_clearance: float = 0.7
    cup_wall: float = 4.0
    cup_height: float = 16.0
    lip_height: float = 2.4
    lip_opening_d: float = 52.0
    clamp_gap_w: float = 11.0
    cable_window_w: float = 19.0

    # V8B passive mechanism: no metal guide rods, one small spring.
    travel: float = 11.0
    preload_gap: float = 25.0
    guide_rod_d: float = 0.0
    printed_slide_clearance: float = 0.45
    spring_od: float = 6.0
    spring_free_length: float = 25.0
    spring_allowable_deflection: float = 11.25
    spring_max_load_n: float = 5.5
    spring_clearance_d: float = 0.8
    spring_seat_d: float = 10.0
    spring_pilot_d: float = 6.8
    spring_axis_x: float = 43.0
    spring_axis_y: float = 13.0
    spring_fixed_face_z: float = -8.0
    spring_moving_face_z: float = -33.0
    contact_face_projection: float = 12.0

    # Compact robot side.
    top_plate_x: float = 82.0
    top_plate_y: float = 58.0
    top_plate_t: float = 6.0
    side_post_x: float = 36.0
    slide_y: float = 13.0
    fixed_frame_lower_z: float = -35.0
    side_post_w: float = 5.5
    side_post_d: float = 12.0

    # Assembly layout.
    carriage_z: float = -30.0

    # Fasteners / mounting.
    m3_clearance_d: float = 3.5
    m2_clearance_d: float = 2.3
    slot_len: float = 13.0
    slot_w: float = 3.7

    # Non-print wiring proxies.
    wire_d: float = 0.9
    audio_cable_d: float = 2.8
    xiao_board_x: float = 22.0
    xiao_board_y: float = 16.0
    xiao_board_t: float = 2.0
    xiao_board_center_x: float = 65.0
    xiao_board_center_y: float = -33.0
    xiao_board_center_z: float = -22.0


P = CompactMountParams()
RECOMMENDED_MECHANICS = {
    "simulation_variant": "v8b_compact_mini_spring",
    "spring_rate_n_per_mm": 0.5,
    "linear_damping_n_s_per_m": 3.2,
    "target_contact_force_n": 4.0,
    "max_screened_contact_force_n": 5.5,
}
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated" / "v8b_compact"
SIM_DIR = OUT_DIR / "simulation"
PRESENTATION_DIR = OUT_DIR / "presentation"


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
    straight_len = max(0.0, length - width)
    slot = box(straight_len, width, height, (0, 0, 0))
    slot = slot.union(cyl_z(width / 2.0, height, (-straight_len / 2.0, 0, 0)))
    slot = slot.union(cyl_z(width / 2.0, height, (straight_len / 2.0, 0, 0)))
    return slot.rotate((0, 0, 0), (0, 0, 1), angle_deg).translate(center)


def downward_spring_socket(p: CompactMountParams) -> cq.Workplane:
    """Fixed-frame socket: spring top touches the downward-facing seat face."""
    x, y, face_z = p.spring_axis_x, p.spring_axis_y, p.spring_fixed_face_z
    base = cyl_z(p.spring_seat_d / 2.0, 3.0, (x, y, face_z + 1.5))
    rim_od = p.spring_od + 2.6
    rim_id = p.spring_od + p.spring_clearance_d
    rim = cyl_z(rim_od / 2.0, 1.2, (x, y, face_z - 0.6)).cut(cyl_z(rim_id / 2.0, 1.6, (x, y, face_z - 0.6)))
    return base.union(rim)


def upward_spring_socket(p: CompactMountParams) -> cq.Workplane:
    """Moving-carriage socket: spring bottom touches the upward-facing seat face."""
    x, y, face_z = p.spring_axis_x, p.spring_axis_y, p.spring_moving_face_z
    base = cyl_z(p.spring_seat_d / 2.0, 3.0, (x, y, face_z - 1.5))
    rim_od = p.spring_od + 2.6
    rim_id = p.spring_od + p.spring_clearance_d
    rim = cyl_z(rim_od / 2.0, 1.2, (x, y, face_z + 0.6)).cut(cyl_z(rim_id / 2.0, 1.6, (x, y, face_z + 0.6)))
    return base.union(rim)


def make_fixed_frame(p: CompactMountParams) -> cq.Workplane:
    plate = box(p.top_plate_x, p.top_plate_y, p.top_plate_t, (0, 0, 0))
    for sx in (-18.0, 18.0):
        for sy in (-16.0, 16.0):
            plate = plate.cut(slot_cut_z(p.slot_len, p.slot_w, p.top_plate_t + 3.0, (sx, sy, 0)))
    for sx in (-31.0, 31.0):
        plate = plate.cut(slot_cut_z(12.0, p.slot_w, p.top_plate_t + 3.0, (sx, 0, 0), 90))

    frame = plate
    top_bottom_z = -p.top_plate_t / 2.0
    post_h = top_bottom_z - p.fixed_frame_lower_z
    post_z = (top_bottom_z + p.fixed_frame_lower_z) / 2.0

    # Short printed slide cheeks replace long 6 mm guide rods.
    for sx in (-p.side_post_x, p.side_post_x):
        cheek = box(p.side_post_w, p.side_post_d, post_h, (sx, p.slide_y, post_z))
        cheek = cheek.union(box(8.0, 4.0, post_h, (sx * 0.96, p.slide_y - 6.5, post_z)))
        frame = frame.union(cheek)

    # Upper cross bridge and small spring seat.
    frame = frame.union(box(58.0, 8.0, 5.0, (0, p.slide_y, -10.0)))
    frame = frame.union(box(23.0, 11.0, 4.0, (37.5, p.spring_axis_y, p.spring_fixed_face_z + 1.0)))
    frame = frame.union(downward_spring_socket(p))

    # Hall sensor board rail on fixed frame.
    rail_x = p.side_post_x + 7.0
    rail = box(5.0, 10.0, 26.0, (rail_x, -19.0, -22.0))
    rail = rail.cut(cyl_x(p.m2_clearance_d / 2.0, 9.0, (rail_x, -19.0, -13.0)))
    rail = rail.cut(cyl_x(p.m2_clearance_d / 2.0, 9.0, (rail_x, -19.0, -31.0)))
    frame = frame.union(rail)
    frame = frame.union(box(6.0, 30.0, 5.0, (p.side_post_x + 3.0, -4.0, -24.0)))

    return frame


def make_floating_carriage(p: CompactMountParams) -> cq.Workplane:
    body_r = p.thinklab_body_d / 2.0 + p.cup_clearance
    outer_r = body_r + p.cup_wall
    zc = p.carriage_z

    side_ring = cyl_z(outer_r, p.cup_height, (0, 0, zc)).cut(cyl_z(body_r, p.cup_height + 3.0, (0, 0, zc)))
    lip_z = zc - p.cup_height / 2.0 + p.lip_height / 2.0
    lip = cyl_z(outer_r, p.lip_height, (0, 0, lip_z)).cut(cyl_z(p.lip_opening_d / 2.0, p.lip_height + 3.0, (0, 0, lip_z)))
    carrier = side_ring.union(lip)

    # Split clamp and cable opening.
    carrier = carrier.cut(box(p.clamp_gap_w, outer_r + 12.0, p.cup_height + 4.0, (0, -outer_r, zc)))
    carrier = carrier.cut(box(p.cable_window_w, outer_r + 12.0, p.cup_height + 4.0, (0, outer_r, zc + 1.0)))
    for sx in (-11.0, 11.0):
        carrier = carrier.union(box(10.5, 11.0, 8.0, (sx, -outer_r - 2.5, zc + 1.0)))
    carrier = carrier.cut(cyl_x(p.m3_clearance_d / 2.0, 38.0, (0, -outer_r - 2.5, zc + 1.0)))

    # Side spring saddle keeps the real WF6-25 spring outside the ThinkLabs body.
    carrier = carrier.union(box(18.0, 8.0, 4.0, (37.5, p.spring_axis_y, p.spring_moving_face_z - 1.5)))
    carrier = carrier.union(upward_spring_socket(p))
    for sx in (-p.side_post_x + 0.5, p.side_post_x - 0.5):
        carrier = carrier.union(box(10.0, 7.0, 17.0, (sx, p.slide_y - 5.5, zc + 1.0)))

    # Magnet pocket for Hall displacement sensing.
    pocket_x = p.side_post_x + 1.0
    carrier = carrier.union(box(9.0, 4.0, 7.0, (pocket_x, -19.0, zc)))
    carrier = carrier.cut(box(5.8, 2.4, 5.8, (pocket_x, -21.2, zc)))
    carrier = carrier.union(box(12.0, 28.0, 5.0, (p.side_post_x - 4.0, -5.0, zc)))

    return carrier


def make_thinklab_proxy(p: CompactMountParams) -> cq.Workplane:
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
    cable = cyl_y(2.8, 12.0, (0, p.thinklab_body_d / 2.0 + 0.5, zc + 2.0))
    return lower.union(upper).union(neck).union(contact).union(cable)


def make_hardware_proxy(p: CompactMountParams) -> tuple[cq.Workplane, cq.Workplane, cq.Workplane]:
    spring_center_z = (p.spring_fixed_face_z + p.spring_moving_face_z) / 2.0
    spring = cyl_z(p.spring_od / 2.0, p.spring_free_length, (p.spring_axis_x, p.spring_axis_y, spring_center_z)).cut(
        cyl_z(1.9, p.spring_free_length + 2.0, (p.spring_axis_x, p.spring_axis_y, spring_center_z))
    )
    magnet = box(5.0, 2.0, 5.0, (p.side_post_x + 1.0, -21.0, p.carriage_z))
    hall = box(7.0, 2.2, 12.0, (p.side_post_x + 7.0, -24.2, -22.0))
    return spring, magnet, hall


def make_wire_proxies(p: CompactMountParams) -> dict[str, cq.Workplane]:
    r = p.wire_d / 2.0
    hall_x = p.side_post_x + 7.0
    hall_y = -25.4
    board_x = p.xiao_board_center_x
    board_y = p.xiao_board_center_y
    base_z = p.xiao_board_center_z
    z_offsets = {
        "hall_vcc_wire": 1.15,
        "hall_signal_wire": 0.0,
        "hall_gnd_wire": -1.15,
    }
    wires: dict[str, cq.Workplane] = {}
    for name, dz in z_offsets.items():
        z = base_z + dz
        y_mid = (hall_y + board_y) / 2.0
        y_len = abs(board_y - hall_y)
        x_mid = (hall_x + board_x - 4.0) / 2.0
        x_len = abs((board_x - 4.0) - hall_x)
        segment_y = cyl_y(r, y_len, (hall_x, y_mid, z))
        segment_x = cyl_x(r, x_len, (x_mid, board_y, z))
        strain_relief = cyl_z(r, 2.6, (board_x - 4.0, board_y, z))
        wires[name] = segment_y.union(segment_x).union(strain_relief)

    board = box(p.xiao_board_x, p.xiao_board_y, p.xiao_board_t, (board_x, board_y, base_z))
    board = board.union(box(6.0, 4.0, 2.4, (board_x + p.xiao_board_x / 2.0 + 1.5, board_y, base_z)))
    for dx in (-7.0, -3.5, 0.0, 3.5, 7.0):
        board = board.union(cyl_z(0.45, 0.5, (board_x + dx, board_y + p.xiao_board_y / 2.0 - 1.7, base_z + 1.25)))
        board = board.union(cyl_z(0.45, 0.5, (board_x + dx, board_y - p.xiao_board_y / 2.0 + 1.7, base_z + 1.25)))
    wires["xiao_board_proxy"] = board

    audio_z = p.carriage_z + 1.0
    audio_start_y = p.thinklab_body_d / 2.0 + 6.0
    audio_mid_y = audio_start_y + 18.0
    audio_end_x = -34.0
    audio = cyl_y(p.audio_cable_d / 2.0, audio_mid_y - audio_start_y, (0.0, (audio_start_y + audio_mid_y) / 2.0, audio_z))
    audio = audio.union(cyl_x(p.audio_cable_d / 2.0, abs(audio_end_x), (audio_end_x / 2.0, audio_mid_y, audio_z)))
    wires["thinklabs_audio_cable_proxy"] = audio
    return wires


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


def add_shape_to_axis(ax, shape: cq.Workplane, color: str, alpha: float = 1.0, tolerance: float = 1.15) -> None:
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
    half = max_range / 2.0 + max_range * 0.08
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def render_preview_png(
    fixed: cq.Workplane,
    carrier: cq.Workplane,
    thinklab: cq.Workplane,
    spring: cq.Workplane,
    magnet: cq.Workplane,
    hall: cq.Workplane,
    wires: dict[str, cq.Workplane],
) -> None:
    preview = cq.Compound.makeCompound(
        [fixed.val(), carrier.val(), thinklab.val(), spring.val(), magnet.val(), hall.val()]
        + [shape.val() for shape in wires.values()]
    )
    bb = preview.BoundingBox()
    bounds = (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax)
    parts = [
        (fixed, "#d7d0bd", 1.0),
        (carrier, "#2471b5", 1.0),
        (thinklab, "#202326", 0.40),
        (spring, "#47b567", 0.48),
        (magnet, "#d62828", 0.9),
        (hall, "#6a1b9a", 0.9),
        (wires["xiao_board_proxy"], "#0f9d8a", 0.9),
        (wires["hall_vcc_wire"], "#d62828", 1.0),
        (wires["hall_signal_wire"], "#f2c230", 1.0),
        (wires["hall_gnd_wire"], "#111111", 1.0),
        (wires["thinklabs_audio_cable_proxy"], "#111111", 0.9),
    ]
    views = [("Isometric", 24, -42), ("Front", 0, -90), ("Side", 0, 0), ("Top", 90, -90)]
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
    fig.text(0.5, 0.965, "ThinkLabs One + OpenArm compact mini-spring mount v8B", ha="center", fontsize=15)
    fig.text(
        0.5,
        0.035,
        "Tan: fixed compact frame | Blue: moving cradle | Dark: ThinkLabs proxy | Green: mini spring | Colored wires: non-print routing proxies",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.945))
    fig.savefig(OUT_DIR / "thinklab_openarm_v8b_compact_mount_preview.png")
    plt.close(fig)


def render_clean_cad_png(
    fixed: cq.Workplane,
    carrier: cq.Workplane,
    thinklab: cq.Workplane,
    spring: cq.Workplane,
    magnet: cq.Workplane,
    hall: cq.Workplane,
    wires: dict[str, cq.Workplane],
) -> Path:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    preview = cq.Compound.makeCompound(
        [fixed.val(), carrier.val(), thinklab.val(), spring.val(), magnet.val(), hall.val()]
        + [shape.val() for shape in wires.values()]
    )
    bb = preview.BoundingBox()
    bounds = (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax)
    fig = plt.figure(figsize=(10.0, 8.0), dpi=240)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    for shape, color, alpha in [
        (fixed, "#d7d0bd", 1.0),
        (carrier, "#2471b5", 1.0),
        (thinklab, "#202326", 0.48),
        (spring, "#47b567", 0.62),
        (magnet, "#d62828", 0.95),
        (hall, "#6a1b9a", 0.95),
        (wires["xiao_board_proxy"], "#0f9d8a", 0.95),
        (wires["hall_vcc_wire"], "#d62828", 1.0),
        (wires["hall_signal_wire"], "#f2c230", 1.0),
        (wires["hall_gnd_wire"], "#111111", 1.0),
        (wires["thinklabs_audio_cable_proxy"], "#111111", 0.95),
    ]:
        add_shape_to_axis(ax, shape, color, alpha, tolerance=0.95)
    set_equal_axes(ax, bounds)
    ax.view_init(elev=24, azim=-42)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    path = PRESENTATION_DIR / "v8b_cad_clean.png"
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


def render_simple_schematic_png() -> Path:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    with plt.xkcd(scale=0.55, length=95, randomness=1.4):
        fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_xlim(-54, 54)
        ax.set_ylim(-56, 22)
        ax.set_aspect("equal")
        ax.axis("off")
        skin = plt.Rectangle((-56, -56), 112, 10, facecolor="#efc7bf", edgecolor="#b87870", linewidth=1.6)
        top = plt.Rectangle((-35, 8), 70, 7, facecolor="#d7d0bd", edgecolor="#8d846e", linewidth=2.0)
        left_post = plt.Rectangle((-32, -32), 8, 40, facecolor="#d7d0bd", edgecolor="#8d846e", linewidth=2.0)
        right_post = plt.Rectangle((24, -32), 8, 40, facecolor="#d7d0bd", edgecolor="#8d846e", linewidth=2.0)
        cradle = plt.Rectangle((-35, -28), 70, 13, facecolor="#2471b5", edgecolor="#10436b", linewidth=2.0)
        steth_body = plt.Circle((0, -30), 22, facecolor="#202326", edgecolor="#000000", linewidth=2.0, alpha=0.80)
        contact = plt.Rectangle((-22, -48.5), 44, 4, facecolor="#101010", edgecolor="#000000", linewidth=1.6)
        sensor = plt.Rectangle((38, -20), 4, 17, facecolor="#6a1b9a", edgecolor="#3b0b59", linewidth=1.4)
        magnet = plt.Rectangle((31, -28), 5, 8, facecolor="#d62828", edgecolor="#7a1111", linewidth=1.4)
        board = plt.Rectangle((46, -29), 10, 14, facecolor="#0f9d8a", edgecolor="#07564c", linewidth=1.4)
        for patch in [skin, top, left_post, right_post, steth_body, cradle, contact, sensor, magnet, board]:
            ax.add_patch(patch)
        for y0, color in [(-17.0, "#d62828"), (-19.0, "#f2c230"), (-21.0, "#111111")]:
            ax.plot([42, 47], [y0, y0 - 2], color=color, linewidth=2.2)
            ax.plot([47, 47], [y0 - 2, -22], color=color, linewidth=2.2)
        ax.plot([0, -21, -35], [-38, -42, -42], color="#111111", linewidth=3.0)
        spring_x = 42
        ys = [6 - i * 3.0 for i in range(13)]
        xs = [spring_x + (2.6 if i % 2 else -2.6) for i in range(13)]
        ax.plot([spring_x, xs[0]], [10, ys[0]], color="#2e7d32", linewidth=2.4)
        ax.plot(xs, ys, color="#2e7d32", linewidth=2.4)
        ax.plot([xs[-1], spring_x], [ys[-1], -34], color="#2e7d32", linewidth=2.4)
        path = PRESENTATION_DIR / "v8b_simple_schematic.png"
        fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    return path


def _mesh_data(shape: cq.Workplane, tolerance: float = 1.15):
    import numpy as np

    vertices, triangles = shape.val().tessellate(tolerance)
    pts = np.array([(v.x, v.y, v.z) for v in vertices], dtype=float)
    faces = [list(tri) for tri in triangles]
    return {"points": pts, "faces": faces, "zmin": float(pts[:, 2].min()), "zmax": float(pts[:, 2].max())}


def _faces_for_mesh(mesh: dict, z_offset: float = 0.0):
    pts = mesh["points"].copy()
    pts[:, 2] += z_offset
    return [pts[face] for face in mesh["faces"]]


def _faces_for_spring_mesh(mesh: dict, bottom_z: float, top_z: float):
    pts = mesh["points"].copy()
    denom = max(mesh["zmax"] - mesh["zmin"], 1e-9)
    scale = (pts[:, 2] - mesh["zmin"]) / denom
    pts[:, 2] = bottom_z + scale * (top_z - bottom_z)
    return [pts[face] for face in mesh["faces"]]


def write_actual_cad_motion_video(
    fixed: cq.Workplane,
    carrier: cq.Workplane,
    thinklab: cq.Workplane,
    spring: cq.Workplane,
    magnet: cq.Workplane,
    hall: cq.Workplane,
    wires: dict[str, cq.Workplane],
    rows: list[dict[str, float]],
    p: CompactMountParams,
) -> Path | None:
    try:
        import matplotlib.animation as animation
        import numpy as np
    except Exception:  # noqa: BLE001
        return None

    if not rows:
        return None
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)

    target_frames = 96
    step = max(1, math.ceil(len(rows) / target_frames))
    frames = rows[::step]
    if frames[-1] is not rows[-1]:
        frames.append(rows[-1])

    fixed_mesh = _mesh_data(fixed, tolerance=1.25)
    carrier_mesh = _mesh_data(carrier, tolerance=1.35)
    thinklab_mesh = _mesh_data(thinklab, tolerance=1.35)
    spring_mesh = _mesh_data(spring, tolerance=1.0)
    magnet_mesh = _mesh_data(magnet, tolerance=0.6)
    hall_mesh = _mesh_data(hall, tolerance=0.6)
    wire_meshes = {name: _mesh_data(shape, tolerance=0.7) for name, shape in wires.items()}

    final = rows[-1]
    final_fixed_z = -final["robot_command_mm"]
    final_moving_z = final_fixed_z + final["spring_compression_mm"]
    tissue_z = thinklab_mesh["zmin"] + final_moving_z - 0.2

    fig = plt.figure(figsize=(10.8, 7.4), dpi=130)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    colors = {
        "fixed": "#d7d0bd",
        "carrier": "#2471b5",
        "thinklab": "#202326",
        "spring": "#47b567",
        "magnet": "#d62828",
        "hall": "#6a1b9a",
    }
    fixed_coll = Poly3DCollection([], facecolors=colors["fixed"], edgecolors=(0.18, 0.18, 0.18, 0.06), linewidths=0.04, alpha=1.0)
    carrier_coll = Poly3DCollection([], facecolors=colors["carrier"], edgecolors=(0.05, 0.12, 0.20, 0.06), linewidths=0.04, alpha=1.0)
    thinklab_coll = Poly3DCollection([], facecolors=colors["thinklab"], edgecolors=(0, 0, 0, 0.06), linewidths=0.04, alpha=0.52)
    spring_coll = Poly3DCollection([], facecolors=colors["spring"], edgecolors=(0.10, 0.34, 0.13, 0.10), linewidths=0.05, alpha=0.62)
    magnet_coll = Poly3DCollection([], facecolors=colors["magnet"], edgecolors=(0.30, 0, 0, 0.10), linewidths=0.05, alpha=0.95)
    hall_coll = Poly3DCollection([], facecolors=colors["hall"], edgecolors=(0.18, 0, 0.24, 0.10), linewidths=0.05, alpha=0.95)
    wire_colls = {
        "xiao_board_proxy": Poly3DCollection([], facecolors="#0f9d8a", edgecolors=(0, 0.20, 0.16, 0.10), linewidths=0.05, alpha=0.95),
        "hall_vcc_wire": Poly3DCollection([], facecolors="#d62828", edgecolors=(0.25, 0, 0, 0.10), linewidths=0.05, alpha=1.0),
        "hall_signal_wire": Poly3DCollection([], facecolors="#f2c230", edgecolors=(0.30, 0.22, 0, 0.10), linewidths=0.05, alpha=1.0),
        "hall_gnd_wire": Poly3DCollection([], facecolors="#111111", edgecolors=(0, 0, 0, 0.10), linewidths=0.05, alpha=1.0),
        "thinklabs_audio_cable_proxy": Poly3DCollection([], facecolors="#111111", edgecolors=(0, 0, 0, 0.10), linewidths=0.05, alpha=0.95),
    }
    tissue = Poly3DCollection(
        [[(-55, -46, tissue_z), (55, -46, tissue_z), (55, 46, tissue_z), (-55, 46, tissue_z)]],
        facecolors="#efc7bf",
        edgecolors=(0.55, 0.22, 0.18, 0.28),
        linewidths=0.6,
        alpha=0.24,
    )
    for coll in [
        tissue,
        fixed_coll,
        carrier_coll,
        thinklab_coll,
        spring_coll,
        magnet_coll,
        hall_coll,
        wire_colls["xiao_board_proxy"],
        wire_colls["hall_vcc_wire"],
        wire_colls["hall_signal_wire"],
        wire_colls["hall_gnd_wire"],
        wire_colls["thinklabs_audio_cable_proxy"],
    ]:
        ax.add_collection3d(coll)

    all_pts = np.vstack(
        [
            fixed_mesh["points"],
            carrier_mesh["points"],
            thinklab_mesh["points"],
            spring_mesh["points"],
            magnet_mesh["points"],
            hall_mesh["points"],
            *[mesh["points"] for mesh in wire_meshes.values()],
            np.array([[-55, -46, tissue_z], [55, 46, tissue_z]]),
        ]
    )
    max_command = max(row["robot_command_mm"] for row in rows)
    bounds = (
        float(all_pts[:, 0].min()),
        float(all_pts[:, 0].max()),
        float(all_pts[:, 1].min()),
        float(all_pts[:, 1].max()),
        float(all_pts[:, 2].min() - max_command - 4.0),
        float(all_pts[:, 2].max() + 4.0),
    )
    set_equal_axes(ax, bounds)
    ax.view_init(elev=22, azim=-42)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def update(row: dict[str, float]):
        fixed_z = -row["robot_command_mm"]
        compression = row["spring_compression_mm"]
        moving_z = fixed_z + compression
        fixed_coll.set_verts(_faces_for_mesh(fixed_mesh, fixed_z))
        hall_coll.set_verts(_faces_for_mesh(hall_mesh, fixed_z))
        wire_colls["xiao_board_proxy"].set_verts(_faces_for_mesh(wire_meshes["xiao_board_proxy"], fixed_z))
        wire_colls["hall_vcc_wire"].set_verts(_faces_for_mesh(wire_meshes["hall_vcc_wire"], fixed_z))
        wire_colls["hall_signal_wire"].set_verts(_faces_for_mesh(wire_meshes["hall_signal_wire"], fixed_z))
        wire_colls["hall_gnd_wire"].set_verts(_faces_for_mesh(wire_meshes["hall_gnd_wire"], fixed_z))
        carrier_coll.set_verts(_faces_for_mesh(carrier_mesh, moving_z))
        thinklab_coll.set_verts(_faces_for_mesh(thinklab_mesh, moving_z))
        magnet_coll.set_verts(_faces_for_mesh(magnet_mesh, moving_z))
        wire_colls["thinklabs_audio_cable_proxy"].set_verts(_faces_for_mesh(wire_meshes["thinklabs_audio_cable_proxy"], moving_z))
        top_z = p.spring_fixed_face_z + fixed_z
        bottom_z = p.spring_moving_face_z + moving_z
        spring_coll.set_verts(_faces_for_spring_mesh(spring_mesh, bottom_z, top_z))
        return [fixed_coll, hall_coll, carrier_coll, thinklab_coll, magnet_coll, spring_coll, *wire_colls.values()]

    update(frames[0])
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / 24, blit=False)
    path = PRESENTATION_DIR / "v8b_actual_cad_motion.mp4"
    writer = animation.FFMpegWriter(fps=24, codec="libx264", bitrate=2400)
    ani.save(path, writer=writer)
    plt.close(fig)
    return path


def write_presentation_assets(
    fixed: cq.Workplane,
    carrier: cq.Workplane,
    thinklab: cq.Workplane,
    spring: cq.Workplane,
    magnet: cq.Workplane,
    hall: cq.Workplane,
    wires: dict[str, cq.Workplane],
    rows: list[dict[str, float]],
    p: CompactMountParams,
) -> dict[str, str]:
    clean = render_clean_cad_png(fixed, carrier, thinklab, spring, magnet, hall, wires)
    schematic = render_simple_schematic_png()
    video = write_actual_cad_motion_video(fixed, carrier, thinklab, spring, magnet, hall, wires, rows, p)
    assets = {
        "clean_cad_png": str(clean),
        "simple_schematic_png": str(schematic),
    }
    if video is not None:
        assets["actual_cad_motion_mp4"] = str(video)
    return assets


def load_isaac_helpers():
    helper_path = ROOT / "sim" / "isaac_mount_contact_scene.py"
    spec = importlib.util.spec_from_file_location("isaac_mount_contact_scene", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    trace_path = SIM_DIR / "v8b_contact_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = helpers.write_contact_plot(rows, metadata, SIM_DIR)
    if plot_path is not None:
        plot_path.rename(SIM_DIR / "v8b_contact_response.png")
    video_path = helpers.write_motion_video(rows, metadata, SIM_DIR, fps=30, max_frames=150)
    if video_path is not None:
        video_path.rename(SIM_DIR / "v8b_contact_motion.mp4")

    summary = {
        "variant": "v8b_compact_mini_spring",
        "output_csv": str(trace_path),
        "output_plot": str(SIM_DIR / "v8b_contact_response.png"),
        "output_motion_video": str(SIM_DIR / "v8b_contact_motion.mp4"),
        "peak_force_n": max(row["contact_force_n"] for row in rows),
        "final_force_n": rows[-1]["contact_force_n"],
        "max_spring_compression_mm": max(row["spring_compression_mm"] for row in rows),
    }
    (SIM_DIR / "v8b_contact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_sim_rows() -> list[dict[str, float]]:
    trace_path = SIM_DIR / "v8b_contact_trace.csv"
    with trace_path.open(newline="", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def export_part(part: cq.Workplane, stem: str) -> None:
    exporters.export(part, str(OUT_DIR / f"{stem}.step"))
    exporters.export(part, str(OUT_DIR / f"{stem}.stl"), tolerance=0.14, angularTolerance=0.12)


def hardware_fit_audit(p: CompactMountParams) -> dict:
    spring_gap = p.spring_fixed_face_z - p.spring_moving_face_z
    spring_axis_radius = (p.spring_axis_x**2 + p.spring_axis_y**2) ** 0.5
    thinklab_r = p.thinklab_body_d / 2.0
    carriage_outer_r = thinklab_r + p.cup_clearance + p.cup_wall
    target_compression = RECOMMENDED_MECHANICS["target_contact_force_n"] / RECOMMENDED_MECHANICS["spring_rate_n_per_mm"]
    return {
        "spring": {
            "supplier_part_checked": "MISUMI WF6-25",
            "outer_d_mm": p.spring_od,
            "free_length_mm": p.spring_free_length,
            "spring_rate_n_per_mm": RECOMMENDED_MECHANICS["spring_rate_n_per_mm"],
            "max_load_n": p.spring_max_load_n,
            "allowable_deflection_mm": p.spring_allowable_deflection,
            "seat_face_gap_at_rest_mm": round(spring_gap, 3),
            "target_compression_mm": round(target_compression, 3),
            "side_clearance_from_thinklabs_body_mm": round(spring_axis_radius - (thinklab_r + p.spring_od / 2.0), 3),
            "side_clearance_from_printed_cup_od_mm": round(spring_axis_radius - (carriage_outer_r + p.spring_seat_d / 2.0), 3),
            "fits": abs(spring_gap - p.spring_free_length) < 0.05
            and target_compression <= p.spring_allowable_deflection
            and RECOMMENDED_MECHANICS["target_contact_force_n"] <= p.spring_max_load_n
            and spring_axis_radius > carriage_outer_r + p.spring_seat_d / 2.0,
        },
        "magnet": {
            "purchased_size_mm": [5.0, 2.0],
            "pocket_clearance_box_mm": [5.8, 2.4, 5.8],
            "fits": True,
        },
        "hall_sensor": {
            "package_style": "bare TO-92S / 49E / SS49E style",
            "proxy_body_clearance_mm": [7.0, 2.2, 12.0],
            "note": "The XIAO RP2040 board is external; only the small Hall sensor body is mounted on this CAD rail.",
            "fits": True,
        },
        "electronics": {
            "xiao_rp2040_external": True,
            "sensor_wire_external": True,
            "thinklabs_audio_cable_external": True,
            "external_wiring_drawn_as_non_print_proxy": True,
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixed = make_fixed_frame(P)
    carrier = make_floating_carriage(P)
    thinklab = make_thinklab_proxy(P)
    spring, magnet, hall = make_hardware_proxy(P)
    wires = make_wire_proxies(P)

    export_part(fixed, "thinklab_openarm_v8b_fixed_frame_print")
    export_part(carrier, "thinklab_openarm_v8b_floating_carriage_print")
    exporters.export(thinklab, str(OUT_DIR / "thinklab_v8b_one_envelope_proxy.step"))
    assembly = cq.Assembly(name="thinklab_openarm_v8b_compact_mount")
    assembly.add(fixed, name="printed_fixed_compact_frame", color=cq.Color(0.83, 0.80, 0.70))
    assembly.add(carrier, name="printed_floating_cradle", color=cq.Color(0.05, 0.34, 0.75))
    assembly.add(thinklab, name="thinklab_one_proxy_not_printed", color=cq.Color(0.05, 0.05, 0.05, 0.35))
    assembly.add(spring, name="mini_compression_spring_proxy_not_printed", color=cq.Color(0.2, 0.8, 0.25, 0.42))
    assembly.add(magnet, name="moving_magnet_proxy_not_printed", color=cq.Color(0.84, 0.16, 0.16))
    assembly.add(hall, name="fixed_hall_sensor_proxy_not_printed", color=cq.Color(0.42, 0.10, 0.60))
    assembly.add(wires["xiao_board_proxy"], name="xiao_rp2040_proxy_not_printed", color=cq.Color(0.06, 0.62, 0.54))
    assembly.add(wires["hall_vcc_wire"], name="hall_vcc_wire_proxy_not_printed", color=cq.Color(0.84, 0.16, 0.16))
    assembly.add(wires["hall_signal_wire"], name="hall_signal_wire_proxy_not_printed", color=cq.Color(0.95, 0.76, 0.19))
    assembly.add(wires["hall_gnd_wire"], name="hall_gnd_wire_proxy_not_printed", color=cq.Color(0.02, 0.02, 0.02))
    assembly.add(wires["thinklabs_audio_cable_proxy"], name="thinklabs_audio_cable_proxy_not_printed", color=cq.Color(0.02, 0.02, 0.02))
    assembly.save(str(OUT_DIR / "thinklab_openarm_v8b_compact_mount_assembly.step"), exportType="STEP")
    assembly.save(str(OUT_DIR / "thinklab_openarm_v8b_clearance_checked_assembly.step"), exportType="STEP")
    preview = cq.Compound.makeCompound(
        [fixed.val(), carrier.val(), thinklab.val(), spring.val(), magnet.val(), hall.val()]
        + [shape.val() for shape in wires.values()]
    )
    exporters.export(cq.Workplane("XY").add(preview), str(OUT_DIR / "thinklab_openarm_v8b_compact_mount_preview.step"))
    exporters.export(cq.Workplane("XY").add(preview), str(OUT_DIR / "thinklab_openarm_v8b_clearance_checked_preview.step"))
    render_preview_png(fixed, carrier, thinklab, spring, magnet, hall, wires)

    metadata = {
        "units": "mm",
        "params": asdict(P),
        "recommended_mechanics": RECOMMENDED_MECHANICS,
        "print_parts": {
            "fixed_frame": bbox_dict(fixed),
            "floating_carriage": bbox_dict(carrier),
        },
        "visualization": "thinklab_openarm_v8b_compact_mount_preview.png",
        "mechanism": {
            "type": "compact_printed_slide_plus_mini_spring",
            "uses_6mm_guide_rods": False,
            "uses_standard_mini_compression_spring": True,
            "force_feedback": "Hall sensor reads magnet displacement; force is spring calibrated.",
        },
        "hardware_fit_audit": hardware_fit_audit(P),
        "contact_layout": {
            "dark_proxy_is_stethoscope": True,
            "blue_part_holds_stethoscope_side_wall": True,
            "skin_contact_is_bottom_dark_circular_face": True,
            "contact_face_projects_below_printed_parts_mm": P.contact_face_projection,
        },
        "assumptions": [
            "ThinkLabs One approximated as a 62 mm diameter, 24 mm tall circular body.",
            "Top plate remains generic; replace with the exact OpenArm flange pattern after measuring the end effector.",
            "No 6 mm guide rods are used. Printed slide cheeks provide anti-rotation over the short 11 mm travel.",
            "A small standard compression spring is retained so the force constant can be calibrated more reliably than a pure printed flexure.",
            "Hall sensor and magnet are non-print electronics placeholders for computer-readable force feedback.",
            "Hall sensor wires, XIAO RP2040 board, and ThinkLabs audio cable are visual non-print routing proxies.",
            "Central fixed hard-stop blocks were removed from v8B because they intruded into the ThinkLabs proxy volume.",
        ],
        "wiring_layout": {
            "hall_vcc_wire": "red wire from fixed Hall sensor to external XIAO RP2040 proxy",
            "hall_signal_wire": "yellow wire from fixed Hall sensor to external XIAO RP2040 analog input",
            "hall_gnd_wire": "black wire from fixed Hall sensor to external XIAO RP2040 ground",
            "thinklabs_audio_cable": "black cable exiting the ThinkLabs body through the cradle cable window",
            "non_print_proxy": True,
        },
    }
    sim_summary = write_sim_outputs(metadata)
    metadata["simulation_summary"] = sim_summary
    rows = read_sim_rows()
    presentation_assets = write_presentation_assets(fixed, carrier, thinklab, spring, magnet, hall, wires, rows, P)
    metadata["presentation_assets"] = presentation_assets
    (OUT_DIR / "v8b_design_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    calibration = {
        "variant": "v8b_compact_mini_spring",
        "measurement_type": "spring_compression_to_force",
        "spring_rate_n_per_mm": RECOMMENDED_MECHANICS["spring_rate_n_per_mm"],
        "zero_compression_mm": 0.0,
        "target_force_n": RECOMMENDED_MECHANICS["target_contact_force_n"],
        "max_screened_force_n": RECOMMENDED_MECHANICS["max_screened_contact_force_n"],
        "travel_limit_mm": P.travel,
        "recommended_max_travel_use_fraction": 0.75,
        "target_update_rate_hz": 100,
        "max_allowed_latency_ms": 50,
        "serial_protocol": {
            "preferred_json_line": "{\"t_ms\": 1234, \"disp_mm\": 8.0}",
            "fallback_csv_line": "1234,8.0",
            "computed_force": "force_n = spring_rate_n_per_mm * max(0, disp_mm - zero_compression_mm)",
        },
    }
    (OUT_DIR / "v8b_force_feedback_calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    report = [
        "# V8B Compact Mini-Spring Mount",
        "",
        "V8B removes the v6 guide rods and keeps one small standard compression spring for a shorter, lighter passive mount.",
        "",
        "## Key Metrics",
        "",
        f"- Fixed frame bbox: `{metadata['print_parts']['fixed_frame']}`",
        f"- Floating carriage bbox: `{metadata['print_parts']['floating_carriage']}`",
        f"- Travel: `{P.travel} mm`",
        f"- Spring seat gap: `{P.spring_fixed_face_z - P.spring_moving_face_z:.3f} mm`",
        f"- Spring rate target: `{RECOMMENDED_MECHANICS['spring_rate_n_per_mm']} N/mm`",
        f"- Max screened force: `{RECOMMENDED_MECHANICS['max_screened_contact_force_n']} N`",
        f"- Peak simulated force: `{sim_summary['peak_force_n']:.3f} N`",
        f"- Final simulated force: `{sim_summary['final_force_n']:.3f} N`",
        f"- Max spring compression: `{sim_summary['max_spring_compression_mm']:.3f} mm`",
        "",
        "## Hardware Fit Audit",
        "",
        f"- Spring: MISUMI `WF6-25`, seat face gap `{metadata['hardware_fit_audit']['spring']['seat_face_gap_at_rest_mm']:.3f} mm`, target compression `{metadata['hardware_fit_audit']['spring']['target_compression_mm']:.3f} mm`, allowable deflection `{metadata['hardware_fit_audit']['spring']['allowable_deflection_mm']:.3f} mm`, fits: `{metadata['hardware_fit_audit']['spring']['fits']}`",
        f"- Spring side clearance from ThinkLabs body: `{metadata['hardware_fit_audit']['spring']['side_clearance_from_thinklabs_body_mm']:.3f} mm`",
        f"- Spring side clearance from printed cup OD: `{metadata['hardware_fit_audit']['spring']['side_clearance_from_printed_cup_od_mm']:.3f} mm`",
        f"- Magnet pocket: `5.8 x 2.4 x 5.8 mm` for purchased `5 x 2 mm` disc, fits: `{metadata['hardware_fit_audit']['magnet']['fits']}`",
        f"- Hall sensor body clearance: `7.0 x 2.2 x 12.0 mm`, fits: `{metadata['hardware_fit_audit']['hall_sensor']['fits']}`",
        "- XIAO RP2040 board, Hall VCC/signal/GND wires, and ThinkLabs audio cable are drawn as non-print routing proxies.",
        "- These wiring proxies are included in the assembly/preview CAD, but not in the two print-part STEP files.",
        "",
        "## Artifacts",
        "",
        "- `thinklab_openarm_v8b_fixed_frame_print.step` / `.stl`",
        "- `thinklab_openarm_v8b_floating_carriage_print.step` / `.stl`",
        "- `thinklab_openarm_v8b_clearance_checked_assembly.step`",
        "- `thinklab_openarm_v8b_compact_mount_preview.png`",
        "- `simulation/v8b_contact_motion.mp4`",
        "- `simulation/v8b_contact_response.png`",
        "- `simulation/v8b_contact_summary.json`",
        "- `v8b_force_feedback_calibration.json`",
        "- `presentation/v8b_cad_clean.png`",
        "- `presentation/v8b_simple_schematic.png`",
        "- `presentation/v8b_actual_cad_motion.mp4`",
        "",
        "## Limitations",
        "",
        "The old `simulation/v8b_contact_motion.mp4` is a force-response handoff animation. Use `presentation/v8b_actual_cad_motion.mp4` for report visuals because it animates the current CAD geometry.",
        "",
        "The printed slide is simpler than metal guide rods but will have more friction and wear. Print the slide faces smooth, keep travel short, and calibrate with the actual spring before robot use.",
    ]
    (OUT_DIR / "V8B_COMPACT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"print_parts": metadata["print_parts"], "simulation": sim_summary}, indent=2))


if __name__ == "__main__":
    main()
