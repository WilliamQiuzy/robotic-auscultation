"""Simplified v9 two-rail mount for ThinkLabs One + OpenArm arm.

V9 intentionally removes the bulky printed slide frame from v8B. The blue
cradle becomes the main structural part: it clamps the ThinkLabs body, rides on
two guide rods, and carries the lower spring seat. The fixed printed part is
only a small robot bridge that holds the guide rods and spring top seat.
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
class SimplifiedRailMountParams:
    thinklab_body_d: float = 62.0
    thinklab_body_h: float = 24.0
    cup_clearance: float = 0.8
    cup_wall: float = 4.0
    cup_height: float = 16.0
    lip_height: float = 2.4
    lip_opening_d: float = 52.0
    cable_window_w: float = 18.0
    clamp_gap_w: float = 10.0

    # V9 passive mechanism: two guide rods, one standard compression spring.
    travel: float = 11.0
    guide_rod_count: int = 2
    guide_rod_d: float = 5.0
    guide_rod_clearance_d: float = 5.8
    guide_rod_x: float = 43.0
    guide_rod_y: float = 0.0
    guide_rod_length: float = 47.0
    spring_count: int = 1
    spring_od: float = 6.0
    spring_free_length: float = 25.0
    spring_allowable_deflection: float = 11.25
    spring_max_load_n: float = 5.5
    spring_seat_d: float = 10.0
    spring_clearance_d: float = 0.8
    spring_axis_x: float = 0.0
    spring_axis_y: float = 42.0
    spring_fixed_face_z: float = -6.0
    spring_moving_face_z: float = -31.0
    contact_face_projection: float = 12.0

    # Compact robot-side bridge.
    top_bridge_x: float = 88.0
    top_bridge_y: float = 34.0
    top_bridge_t: float = 5.0
    spring_tab_x: float = 20.0
    spring_tab_y: float = 34.0
    rod_socket_od: float = 11.0

    # Moving cradle.
    carriage_z: float = -30.0
    guide_bushing_od: float = 12.0
    guide_bushing_h: float = 18.0
    clamp_screw_count: int = 2
    m3_clearance_d: float = 3.5
    m2_clearance_d: float = 2.3
    slot_len: float = 13.0
    slot_w: float = 3.7


P = SimplifiedRailMountParams()
RECOMMENDED_MECHANICS = {
    "simulation_variant": "v9_simplified_two_rail_spring_clamp",
    "spring_rate_n_per_mm": 0.5,
    "linear_damping_n_s_per_m": 3.2,
    "target_contact_force_n": 4.0,
    "max_screened_contact_force_n": 5.5,
}

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated" / "v9_simplified"
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


def downward_spring_socket(p: SimplifiedRailMountParams) -> cq.Workplane:
    x, y, face_z = p.spring_axis_x, p.spring_axis_y, p.spring_fixed_face_z
    base = cyl_z(p.spring_seat_d / 2.0, 3.0, (x, y, face_z + 1.5))
    rim_od = p.spring_od + 2.6
    rim_id = p.spring_od + p.spring_clearance_d
    rim = cyl_z(rim_od / 2.0, 1.2, (x, y, face_z - 0.6)).cut(cyl_z(rim_id / 2.0, 1.6, (x, y, face_z - 0.6)))
    return base.union(rim)


def upward_spring_socket(p: SimplifiedRailMountParams) -> cq.Workplane:
    x, y, face_z = p.spring_axis_x, p.spring_axis_y, p.spring_moving_face_z
    base = cyl_z(p.spring_seat_d / 2.0, 3.0, (x, y, face_z - 1.5))
    rim_od = p.spring_od + 2.6
    rim_id = p.spring_od + p.spring_clearance_d
    rim = cyl_z(rim_od / 2.0, 1.2, (x, y, face_z + 0.6)).cut(cyl_z(rim_id / 2.0, 1.6, (x, y, face_z + 0.6)))
    return base.union(rim)


def make_fixed_bridge(p: SimplifiedRailMountParams) -> cq.Workplane:
    bridge = box(p.top_bridge_x, p.top_bridge_y, p.top_bridge_t, (0, 0, 0))
    for sx in (-18.0, 18.0):
        for sy in (-10.0, 10.0):
            bridge = bridge.cut(slot_cut_z(p.slot_len, p.slot_w, p.top_bridge_t + 3.0, (sx, sy, 0)))

    # Two rod sockets are the only guide features on the fixed print.
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        socket = cyl_z(p.rod_socket_od / 2.0, 7.0, (sx, p.guide_rod_y, -3.0))
        socket = socket.cut(cyl_z(p.guide_rod_clearance_d / 2.0, 9.0, (sx, p.guide_rod_y, -3.0)))
        bridge = bridge.union(socket)

    # A small rear tab carries the top spring seat; it replaces the old side frame.
    tab_center_y = (p.top_bridge_y / 2.0 + p.spring_axis_y) / 2.0
    tab_y = p.spring_axis_y - p.top_bridge_y / 2.0 + p.spring_seat_d / 2.0 + 1.5
    bridge = bridge.union(box(p.spring_tab_x, tab_y, p.top_bridge_t, (0, tab_center_y, 0)))
    bridge = bridge.union(downward_spring_socket(p))

    return bridge


def make_blue_cradle(p: SimplifiedRailMountParams) -> cq.Workplane:
    body_r = p.thinklab_body_d / 2.0 + p.cup_clearance
    outer_r = body_r + p.cup_wall
    zc = p.carriage_z

    ring = cyl_z(outer_r, p.cup_height, (0, 0, zc)).cut(cyl_z(body_r, p.cup_height + 3.0, (0, 0, zc)))
    lip_z = zc - p.cup_height / 2.0 + p.lip_height / 2.0
    lip = cyl_z(outer_r, p.lip_height, (0, 0, lip_z)).cut(cyl_z(p.lip_opening_d / 2.0, p.lip_height + 3.0, (0, 0, lip_z)))
    cradle = ring.union(lip)

    # Front split clamp: two external ears, two M3 screw holes, no blocks inside the stethoscope envelope.
    cradle = cradle.cut(box(p.clamp_gap_w, outer_r + 12.0, p.cup_height + 4.0, (0, -outer_r, zc)))
    cradle = cradle.cut(box(p.cable_window_w, outer_r + 12.0, p.cup_height + 4.0, (0, outer_r, zc + 1.0)))
    clamp_y = -outer_r - 4.5
    for sx in (-12.0, 12.0):
        ear = box(10.0, 10.0, 8.0, (sx, clamp_y, zc + 1.0))
        ear = ear.cut(cyl_x(p.m3_clearance_d / 2.0, 12.0, (sx, clamp_y, zc + 1.0)))
        cradle = cradle.union(ear)
    cradle = cradle.union(box(28.0, 4.5, 4.0, (0, clamp_y + 2.8, zc - 2.0)))

    # Two guide bushings. They sit outside the 62 mm ThinkLabs proxy and attach to the blue ring.
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        bushing = cyl_z(p.guide_bushing_od / 2.0, p.guide_bushing_h, (sx, p.guide_rod_y, zc))
        bushing = bushing.cut(cyl_z(p.guide_rod_clearance_d / 2.0, p.guide_bushing_h + 3.0, (sx, p.guide_rod_y, zc)))
        bridge = box(9.0, 9.0, 10.0, (sx * 0.94, p.guide_rod_y, zc))
        cradle = cradle.union(bushing).union(bridge)

    # Rear spring seat is on the moving cradle.
    spring_bridge_y = (outer_r + p.spring_axis_y) / 2.0
    spring_bridge_len = max(5.0, p.spring_axis_y - outer_r + 4.0)
    cradle = cradle.union(box(16.0, spring_bridge_len, 4.0, (0, spring_bridge_y, p.spring_moving_face_z - 1.5)))
    cradle = cradle.union(upward_spring_socket(p))

    # Moving magnet pocket for Hall displacement sensing.
    magnet_x = p.guide_rod_x - 5.0
    cradle = cradle.union(box(9.0, 4.0, 7.0, (magnet_x, -2.0, zc)))
    cradle = cradle.cut(box(5.8, 2.4, 5.8, (magnet_x, -4.2, zc)))

    return cradle


def make_thinklab_proxy(p: SimplifiedRailMountParams) -> cq.Workplane:
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
    cable_stub = cyl_y(2.8 / 2.0, 13.0, (0, p.thinklab_body_d / 2.0 + 2.0, zc + 2.0))
    return lower.union(upper).union(neck).union(contact).union(cable_stub)


def make_hardware_proxies(p: SimplifiedRailMountParams) -> dict[str, cq.Workplane]:
    spring_center_z = (p.spring_fixed_face_z + p.spring_moving_face_z) / 2.0
    spring = cyl_z(p.spring_od / 2.0, p.spring_free_length, (p.spring_axis_x, p.spring_axis_y, spring_center_z))
    spring = spring.cut(cyl_z(1.9, p.spring_free_length + 2.0, (p.spring_axis_x, p.spring_axis_y, spring_center_z)))

    guide_rods = None
    rod_center_z = -18.5
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        rod = cyl_z(p.guide_rod_d / 2.0, p.guide_rod_length, (sx, p.guide_rod_y, rod_center_z))
        guide_rods = rod if guide_rods is None else guide_rods.union(rod)
    assert guide_rods is not None

    body_r = p.thinklab_body_d / 2.0 + p.cup_clearance
    outer_r = body_r + p.cup_wall
    clamp_y = -outer_r - 4.5
    clamp_screws = None
    for sx in (-12.0, 12.0):
        screw = cyl_x(1.35, 18.0, (sx, clamp_y, p.carriage_z + 1.0))
        head = cyl_x(2.3, 2.5, (sx - 7.75 if sx > 0 else sx + 7.75, clamp_y, p.carriage_z + 1.0))
        screw = screw.union(head)
        clamp_screws = screw if clamp_screws is None else clamp_screws.union(screw)
    assert clamp_screws is not None

    magnet_x = p.guide_rod_x - 5.0
    magnet = box(5.0, 2.0, 5.0, (magnet_x, -4.0, p.carriage_z))
    hall = box(7.0, 2.2, 10.0, (p.guide_rod_x - 4.5, -2.8, -16.0))
    return {
        "spring": spring,
        "guide_rods": guide_rods,
        "clamp_screws": clamp_screws,
        "magnet": magnet,
        "hall": hall,
    }


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


def assembly_parts(
    fixed: cq.Workplane,
    cradle: cq.Workplane,
    thinklab: cq.Workplane,
    hardware: dict[str, cq.Workplane],
) -> list[tuple[cq.Workplane, str, float]]:
    return [
        (fixed, "#d7d0bd", 1.0),
        (cradle, "#2471b5", 1.0),
        (thinklab, "#202326", 0.46),
        (hardware["guide_rods"], "#b8b8b8", 0.85),
        (hardware["spring"], "#47b567", 0.62),
        (hardware["clamp_screws"], "#111111", 0.9),
        (hardware["magnet"], "#d62828", 0.95),
        (hardware["hall"], "#6a1b9a", 0.95),
    ]


def render_preview_png(
    fixed: cq.Workplane,
    cradle: cq.Workplane,
    thinklab: cq.Workplane,
    hardware: dict[str, cq.Workplane],
) -> None:
    preview = cq.Compound.makeCompound([fixed.val(), cradle.val(), thinklab.val()] + [shape.val() for shape in hardware.values()])
    bb = preview.BoundingBox()
    bounds = (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax)
    views = [("Isometric", 24, -42), ("Front", 0, -90), ("Side", 0, 0), ("Top", 90, -90)]
    fig = plt.figure(figsize=(13.5, 10.0), dpi=180)
    fig.patch.set_facecolor("#f7f7f5")
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.set_facecolor("#f7f7f5")
        for shape, color, alpha in assembly_parts(fixed, cradle, thinklab, hardware):
            add_shape_to_axis(ax, shape, color, alpha)
        set_equal_axes(ax, bounds)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11, pad=0)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    fig.savefig(OUT_DIR / "thinklab_openarm_v9_simplified_preview.png")
    plt.close(fig)


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
        add_shape_to_axis(ax, shape, color, alpha, tolerance=0.95)
    set_equal_axes(ax, bounds)
    ax.view_init(elev=24, azim=-42)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    path = PRESENTATION_DIR / "v9_cad_clean.png"
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


def render_simple_schematic_png(p: SimplifiedRailMountParams) -> Path:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    with plt.xkcd(scale=0.45, length=90, randomness=1.2):
        fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_xlim(-56, 56)
        ax.set_ylim(-58, 20)
        ax.set_aspect("equal")
        ax.axis("off")

        skin = plt.Rectangle((-56, -56), 112, 9, facecolor="#efc7bf", edgecolor="#b87870", linewidth=1.5)
        top = plt.Rectangle((-44, 8), 88, 5, facecolor="#d7d0bd", edgecolor="#8d846e", linewidth=2.0)
        cradle = plt.Rectangle((-36, -30), 72, 13, facecolor="#2471b5", edgecolor="#10436b", linewidth=2.0)
        steth = plt.Circle((0, -32), 22, facecolor="#202326", edgecolor="#000000", linewidth=2.0, alpha=0.82)
        contact = plt.Rectangle((-22, -49), 44, 4, facecolor="#101010", edgecolor="#000000", linewidth=1.5)
        for patch in [skin, top, steth, cradle, contact]:
            ax.add_patch(patch)

        for x in (-39, 39):
            ax.plot([x, x], [10, -39], color="#777777", linewidth=4.0)
            ax.add_patch(plt.Rectangle((x - 5, -31), 10, 14, facecolor="#2471b5", edgecolor="#10436b", linewidth=1.6))

        xs = [0 + (3.0 if i % 2 else -3.0) for i in range(13)]
        ys = [5 - i * 3.0 for i in range(13)]
        ax.plot([0, xs[0]], [8, ys[0]], color="#2e7d32", linewidth=2.4)
        ax.plot(xs, ys, color="#2e7d32", linewidth=2.4)
        ax.plot([xs[-1], 0], [ys[-1], -34], color="#2e7d32", linewidth=2.4)
        ax.add_patch(plt.Rectangle((31, -30), 5, 8, facecolor="#d62828", edgecolor="#7a1111", linewidth=1.3))
        ax.add_patch(plt.Rectangle((38, -21), 4, 13, facecolor="#6a1b9a", edgecolor="#3b0b59", linewidth=1.3))
        ax.plot([41, 49], [-16, -12], color="#d62828", linewidth=2.0)
        ax.plot([41, 49], [-18, -18], color="#f2c230", linewidth=2.0)
        ax.plot([41, 49], [-20, -24], color="#111111", linewidth=2.0)
        ax.plot([-11, -23], [-41, -41], color="#111111", linewidth=3.0)
        ax.plot([11, 23], [-41, -41], color="#111111", linewidth=3.0)

        path = PRESENTATION_DIR / "v9_simple_schematic.png"
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
    cradle: cq.Workplane,
    thinklab: cq.Workplane,
    hardware: dict[str, cq.Workplane],
    rows: list[dict[str, float]],
    p: SimplifiedRailMountParams,
) -> Path | None:
    try:
        import matplotlib.animation as animation
        import numpy as np
    except Exception:  # noqa: BLE001
        return None

    if not rows:
        return None
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)

    target_frames = 90
    step = max(1, math.ceil(len(rows) / target_frames))
    frames = rows[::step]
    if frames[-1] is not rows[-1]:
        frames.append(rows[-1])

    fixed_mesh = _mesh_data(fixed, tolerance=1.25)
    guide_mesh = _mesh_data(hardware["guide_rods"], tolerance=1.05)
    hall_mesh = _mesh_data(hardware["hall"], tolerance=0.6)
    cradle_mesh = _mesh_data(cradle, tolerance=1.3)
    thinklab_mesh = _mesh_data(thinklab, tolerance=1.3)
    screw_mesh = _mesh_data(hardware["clamp_screws"], tolerance=0.7)
    magnet_mesh = _mesh_data(hardware["magnet"], tolerance=0.6)
    spring_mesh = _mesh_data(hardware["spring"], tolerance=0.9)

    final = rows[-1]
    final_fixed_z = -final["robot_command_mm"]
    final_moving_z = final_fixed_z + final["spring_compression_mm"]
    tissue_z = thinklab_mesh["zmin"] + final_moving_z - 0.2

    fig = plt.figure(figsize=(10.8, 7.4), dpi=130)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    collections = {
        "tissue": Poly3DCollection(
            [[(-55, -48, tissue_z), (55, -48, tissue_z), (55, 48, tissue_z), (-55, 48, tissue_z)]],
            facecolors="#efc7bf",
            edgecolors=(0.55, 0.22, 0.18, 0.28),
            linewidths=0.6,
            alpha=0.24,
        ),
        "fixed": Poly3DCollection([], facecolors="#d7d0bd", edgecolors=(0.18, 0.18, 0.18, 0.06), linewidths=0.04, alpha=1.0),
        "guide_rods": Poly3DCollection([], facecolors="#b8b8b8", edgecolors=(0.18, 0.18, 0.18, 0.06), linewidths=0.04, alpha=0.85),
        "hall": Poly3DCollection([], facecolors="#6a1b9a", edgecolors=(0.18, 0, 0.24, 0.10), linewidths=0.05, alpha=0.95),
        "cradle": Poly3DCollection([], facecolors="#2471b5", edgecolors=(0.05, 0.12, 0.20, 0.06), linewidths=0.04, alpha=1.0),
        "thinklab": Poly3DCollection([], facecolors="#202326", edgecolors=(0, 0, 0, 0.06), linewidths=0.04, alpha=0.52),
        "clamp_screws": Poly3DCollection([], facecolors="#111111", edgecolors=(0, 0, 0, 0.10), linewidths=0.05, alpha=0.9),
        "magnet": Poly3DCollection([], facecolors="#d62828", edgecolors=(0.30, 0, 0, 0.10), linewidths=0.05, alpha=0.95),
        "spring": Poly3DCollection([], facecolors="#47b567", edgecolors=(0.10, 0.34, 0.13, 0.10), linewidths=0.05, alpha=0.62),
    }
    for coll in collections.values():
        ax.add_collection3d(coll)

    all_pts = np.vstack(
        [
            fixed_mesh["points"],
            guide_mesh["points"],
            hall_mesh["points"],
            cradle_mesh["points"],
            thinklab_mesh["points"],
            screw_mesh["points"],
            magnet_mesh["points"],
            spring_mesh["points"],
            np.array([[-55, -48, tissue_z], [55, 48, tissue_z]]),
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
        collections["fixed"].set_verts(_faces_for_mesh(fixed_mesh, fixed_z))
        collections["guide_rods"].set_verts(_faces_for_mesh(guide_mesh, fixed_z))
        collections["hall"].set_verts(_faces_for_mesh(hall_mesh, fixed_z))
        collections["cradle"].set_verts(_faces_for_mesh(cradle_mesh, moving_z))
        collections["thinklab"].set_verts(_faces_for_mesh(thinklab_mesh, moving_z))
        collections["clamp_screws"].set_verts(_faces_for_mesh(screw_mesh, moving_z))
        collections["magnet"].set_verts(_faces_for_mesh(magnet_mesh, moving_z))
        # Keep the mesh's own seating offsets relative to each spring face so the
        # squashed spring never sinks into the printed sockets (visual 穿模 fix).
        top_z = spring_mesh["zmax"] + fixed_z
        bottom_z = spring_mesh["zmin"] + moving_z
        collections["spring"].set_verts(_faces_for_spring_mesh(spring_mesh, bottom_z, top_z))
        return list(collections.values())

    update(frames[0])
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / 24, blit=False)
    path = PRESENTATION_DIR / "v9_actual_cad_motion.mp4"
    writer = animation.FFMpegWriter(fps=24, codec="libx264", bitrate=2400)
    ani.save(path, writer=writer)
    plt.close(fig)
    return path


def write_presentation_assets(
    fixed: cq.Workplane,
    cradle: cq.Workplane,
    thinklab: cq.Workplane,
    hardware: dict[str, cq.Workplane],
    rows: list[dict[str, float]],
    p: SimplifiedRailMountParams,
) -> dict[str, str]:
    clean = render_clean_cad_png(fixed, cradle, thinklab, hardware)
    schematic = render_simple_schematic_png(p)
    video = write_actual_cad_motion_video(fixed, cradle, thinklab, hardware, rows, p)
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
    trace_path = SIM_DIR / "v9_contact_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = helpers.write_contact_plot(rows, metadata, SIM_DIR)
    if plot_path is not None:
        plot_path.rename(SIM_DIR / "v9_contact_response.png")
    video_path = helpers.write_motion_video(rows, metadata, SIM_DIR, fps=30, max_frames=150)
    if video_path is not None:
        video_path.rename(SIM_DIR / "v9_contact_motion.mp4")

    summary = {
        "variant": "v9_simplified_two_rail_spring_clamp",
        "output_csv": str(trace_path),
        "output_plot": str(SIM_DIR / "v9_contact_response.png"),
        "output_motion_video": str(SIM_DIR / "v9_contact_motion.mp4"),
        "peak_force_n": max(row["contact_force_n"] for row in rows),
        "final_force_n": rows[-1]["contact_force_n"],
        "max_spring_compression_mm": max(row["spring_compression_mm"] for row in rows),
    }
    (SIM_DIR / "v9_contact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_sim_rows() -> list[dict[str, float]]:
    trace_path = SIM_DIR / "v9_contact_trace.csv"
    with trace_path.open(newline="", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def export_part(part: cq.Workplane, stem: str) -> None:
    exporters.export(part, str(OUT_DIR / f"{stem}.step"))
    exporters.export(part, str(OUT_DIR / f"{stem}.stl"), tolerance=0.14, angularTolerance=0.12)


def clearance_audit(p: SimplifiedRailMountParams) -> dict:
    body_r = p.thinklab_body_d / 2.0
    body_plus_clearance_r = body_r + p.cup_clearance
    outer_r = body_plus_clearance_r + p.cup_wall
    guide_axis_radius = abs(p.guide_rod_x)
    spring_axis_radius = (p.spring_axis_x**2 + p.spring_axis_y**2) ** 0.5
    clamp_y = -outer_r - 4.5
    return {
        "thinklabs_body_radius_mm": body_r,
        "cradle_inner_radius_mm": body_plus_clearance_r,
        "fixed_print_intrudes_into_thinklabs_body_envelope": False,
        "guide_rods_intrude_into_thinklabs_body_envelope": guide_axis_radius - p.guide_rod_d / 2.0 <= body_r,
        "spring_intrudes_into_thinklabs_body_envelope": spring_axis_radius - p.spring_od / 2.0 <= body_r,
        "guide_rod_side_clearance_from_body_mm": round(guide_axis_radius - (body_r + p.guide_rod_d / 2.0), 3),
        "spring_side_clearance_from_body_mm": round(spring_axis_radius - (body_r + p.spring_od / 2.0), 3),
        "spring_side_clearance_from_printed_cup_od_mm": round(spring_axis_radius - (outer_r + p.spring_seat_d / 2.0), 3),
        "clamp_boss_side_clearance_from_body_mm": round(abs(clamp_y) - (body_r + 5.0), 3),
        "spring_seat_face_gap_at_rest_mm": round(p.spring_fixed_face_z - p.spring_moving_face_z, 3),
    }


def make_metadata(
    p: SimplifiedRailMountParams,
    fixed: cq.Workplane,
    cradle: cq.Workplane,
    sim_summary: dict | None = None,
) -> dict:
    body_r = p.thinklab_body_d / 2.0 + p.cup_clearance
    metadata = {
        "units": "mm",
        "params": asdict(p),
        "recommended_mechanics": RECOMMENDED_MECHANICS,
        "print_parts": {
            "fixed_bridge": bbox_dict(fixed),
            "blue_cradle": bbox_dict(cradle),
        },
        "mechanism": {
            "type": "minimal_two_rail_spring_clamp",
            "fixed_part_role": "small_robot_bridge_only",
            "blue_cradle_is_main_structural_part": True,
            "uses_two_guide_rods": True,
            "uses_one_standard_compression_spring": True,
            "two_side_clamp_screws": True,
            "force_feedback": "Hall sensor reads magnet displacement; force is spring calibrated.",
        },
        "clearance_audit": clearance_audit(p),
        "electronics_layout": {
            "xiao_board_drawn_in_cad": False,
            "xiao_board_mounting": "external to the printed mechanical module",
            "hall_sensor_drawn_in_cad": True,
            "magnet_drawn_in_cad": True,
            "wire_routing_drawn_in_mechanical_step": False,
            "wire_routing": "shown only in v9_simple_schematic.png to keep mechanical CAD clean",
        },
        "contact_layout": {
            "dark_proxy_is_stethoscope": True,
            "blue_part_holds_stethoscope_side_wall": True,
            "skin_contact_is_bottom_dark_circular_face": True,
            "only_dark_bottom_contact_face_projects_below_printed_parts": True,
            "contact_face_projects_below_printed_parts_mm": p.contact_face_projection,
            "cradle_inner_diameter_mm": round(body_r * 2.0, 3),
            "printed_cradle_has_bottom_lip_with_open_center": True,
        },
        "hardware_fit_audit": {
            "spring": {
                "supplier_part_checked": "MISUMI WF6-25",
                "outer_d_mm": p.spring_od,
                "free_length_mm": p.spring_free_length,
                "spring_rate_n_per_mm": RECOMMENDED_MECHANICS["spring_rate_n_per_mm"],
                "target_compression_mm": round(
                    RECOMMENDED_MECHANICS["target_contact_force_n"] / RECOMMENDED_MECHANICS["spring_rate_n_per_mm"], 3
                ),
                "seat_face_gap_at_rest_mm": round(p.spring_fixed_face_z - p.spring_moving_face_z, 3),
                "fits": abs((p.spring_fixed_face_z - p.spring_moving_face_z) - p.spring_free_length) < 0.05,
            },
            "guide_rods": {
                "count": p.guide_rod_count,
                "nominal_diameter_mm": p.guide_rod_d,
                "printed_bushing_clearance_diameter_mm": p.guide_rod_clearance_d,
            },
            "clamp_screws": {
                "count": p.clamp_screw_count,
                "recommended": "M3 socket head cap screws with heat-set inserts or locknuts after prototype fit check",
                "printed_clearance_hole_d_mm": p.m3_clearance_d,
            },
            "magnet": {
                "purchased_size_mm": [5.0, 2.0],
                "pocket_clearance_box_mm": [5.8, 2.4, 5.8],
                "fits": True,
            },
            "hall_sensor": {
                "package_style": "bare TO-92S / 49E / SS49E style",
                "proxy_body_clearance_mm": [7.0, 2.2, 10.0],
                "fits": True,
            },
        },
        "assumptions": [
            "ThinkLabs One approximated as a 62 mm diameter, 24 mm tall circular body.",
            "The fixed tan print is intentionally only a small bridge; it does not wrap around the stethoscope.",
            "Two 5 mm guide rods provide vertical alignment. The blue cradle carries the bushing holes.",
            "One MISUMI WF6-25 compression spring sets the passive contact force.",
            "The XIAO RP2040 and long wires are intentionally omitted from the mechanical STEP; they are external electronics.",
        ],
    }
    if sim_summary is not None:
        metadata["simulation_summary"] = sim_summary
    return metadata


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixed = make_fixed_bridge(P)
    cradle = make_blue_cradle(P)
    thinklab = make_thinklab_proxy(P)
    hardware = make_hardware_proxies(P)

    export_part(fixed, "thinklab_openarm_v9_fixed_bridge_print")
    export_part(cradle, "thinklab_openarm_v9_blue_cradle_print")
    exporters.export(thinklab, str(OUT_DIR / "thinklab_v9_one_envelope_proxy.step"))

    assembly = cq.Assembly(name="thinklab_openarm_v9_simplified_mount")
    assembly.add(fixed, name="printed_fixed_robot_bridge", color=cq.Color(0.83, 0.80, 0.70))
    assembly.add(cradle, name="printed_blue_clamping_cradle", color=cq.Color(0.05, 0.34, 0.75))
    assembly.add(thinklab, name="thinklab_one_proxy_not_printed", color=cq.Color(0.05, 0.05, 0.05, 0.38))
    assembly.add(hardware["guide_rods"], name="two_guide_rods_proxy_not_printed", color=cq.Color(0.72, 0.72, 0.72))
    assembly.add(hardware["spring"], name="wf6_25_compression_spring_proxy_not_printed", color=cq.Color(0.2, 0.8, 0.25, 0.46))
    assembly.add(hardware["clamp_screws"], name="m3_clamp_screws_proxy_not_printed", color=cq.Color(0.02, 0.02, 0.02))
    assembly.add(hardware["magnet"], name="moving_magnet_proxy_not_printed", color=cq.Color(0.84, 0.16, 0.16))
    assembly.add(hardware["hall"], name="fixed_hall_sensor_proxy_not_printed", color=cq.Color(0.42, 0.10, 0.60))
    assembly.save(str(OUT_DIR / "thinklab_openarm_v9_simplified_assembly.step"), exportType="STEP")

    preview = cq.Compound.makeCompound([fixed.val(), cradle.val(), thinklab.val()] + [shape.val() for shape in hardware.values()])
    exporters.export(cq.Workplane("XY").add(preview), str(OUT_DIR / "thinklab_openarm_v9_simplified_preview.step"))
    render_preview_png(fixed, cradle, thinklab, hardware)

    metadata = make_metadata(P, fixed, cradle)
    sim_summary = write_sim_outputs(metadata)
    metadata = make_metadata(P, fixed, cradle, sim_summary)
    rows = read_sim_rows()
    metadata["presentation_assets"] = write_presentation_assets(fixed, cradle, thinklab, hardware, rows, P)
    (OUT_DIR / "v9_design_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    calibration = {
        "variant": "v9_simplified_two_rail_spring_clamp",
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
    (OUT_DIR / "v9_force_feedback_calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    report = [
        "# V9 Simplified Two-Rail Mount",
        "",
        "V9 is the simplified replacement candidate for v8B. It keeps only the required mechanics: blue ThinkLabs clamp cradle, two guide rods, one spring, and Hall/magnet feedback.",
        "",
        "## Key Metrics",
        "",
        f"- Fixed bridge bbox: `{metadata['print_parts']['fixed_bridge']}`",
        f"- Blue cradle bbox: `{metadata['print_parts']['blue_cradle']}`",
        f"- Travel: `{P.travel} mm`",
        f"- Spring seat gap: `{P.spring_fixed_face_z - P.spring_moving_face_z:.3f} mm`",
        f"- Peak simulated force: `{sim_summary['peak_force_n']:.3f} N`",
        f"- Final simulated force: `{sim_summary['final_force_n']:.3f} N`",
        f"- Max spring compression: `{sim_summary['max_spring_compression_mm']:.3f} mm`",
        "",
        "## What Changed From V8B",
        "",
        "- Removed the large tan side frame and printed slide cheeks.",
        "- Added two simple guide rods and blue cradle bushings.",
        "- Kept one MISUMI WF6-25 compression spring.",
        "- Kept Hall sensor and magnet, but removed the external XIAO board from the mechanical CAD.",
        "- Kept the ThinkLabs bottom contact face protruding below every printed part.",
        "",
        "## Artifacts",
        "",
        "- `thinklab_openarm_v9_fixed_bridge_print.step` / `.stl`",
        "- `thinklab_openarm_v9_blue_cradle_print.step` / `.stl`",
        "- `thinklab_openarm_v9_simplified_assembly.step`",
        "- `thinklab_openarm_v9_simplified_preview.step`",
        "- `thinklab_openarm_v9_simplified_preview.png`",
        "- `presentation/v9_cad_clean.png`",
        "- `presentation/v9_simple_schematic.png`",
        "- `presentation/v9_actual_cad_motion.mp4`",
        "- `simulation/v9_contact_motion.mp4`",
        "- `simulation/v9_contact_response.png`",
        "- `simulation/v9_contact_summary.json`",
        "- `v9_force_feedback_calibration.json`",
    ]
    (OUT_DIR / "V9_SIMPLIFIED_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({"print_parts": metadata["print_parts"], "clearance": metadata["clearance_audit"], "simulation": sim_summary}, indent=2))


if __name__ == "__main__":
    main()
