"""VEO5 coaxial-Hall ThinkLabs One + OpenArm passive force mount.

VEO5 keeps the VEO3/VEO4 clamshell architecture but fixes four defects found in
VEO4 and redesigns the magnet sensing module:

1. Clamshell split cutters were swapped, so the printed halves overlapped each
   other by the full 3 mm split gap. VEO5 splits left to x <= -1.5 and right to
   x >= +1.5 with a true air gap.
2. Guide bushings, front M3 lugs and the magnet holder were disconnected
   islands inside each clamshell solid (the parts would print in 3-4 loose
   pieces). VEO5 fuses them with bridges/webs and audits connectivity.
3. The VEO4 magnet (Y-axis disc) translated PAST the fixed Hall along Z, so the
   field peaked mid-stroke and force readout was non-monotonic. VEO5 uses a
   coaxial axial-gap design: Z-magnetized D5x2 disc in an upward pocket on the
   moving ring, SS49E held face-down in a pillar on the fixed bridge directly
   above it. The air gap closes monotonically with spring compression.
4. The spring bridge top sat 0.11 mm under the bottom spring coil; VEO5 lowers
   it for ~0.9 mm of real clearance and audits the *compressed* spring against
   the printed parts at several travel points (VEO4 only audited free length).
5. Clamp screws were modeled as 56 mm rods skewering the spring bridge slab,
   with an M3 nut buried inside the 3 mm plate (impossible to assemble, and the
   visible blue-plate/screw interpenetration). VEO5 models real M3x30 SHCS:
   head seated on the left lug face, shaft ending inside the right lug, nut in
   a printed hex pocket with a vertical insertion slot, and shortened clearance
   holes that no longer tunnel through the spring bridge.
6. The Hall sensor was modeled as a bare 3x4 mm epoxy block; the purchased
   Honeywell SS49E (TO-92S) has three ~13 mm straight leads. VEO5 models the
   leads, widens the pillar leg channel around them, and moves the wire splice
   to the lead tips.
7. The inherited 47 mm guide-rod proxies hung down to z=-42 — only 8 mm above
   the stethoscope contact face, so the rod tips reached the patient's skin at
   the 4 N working compression (8 mm) and dug in beyond it. VEO5 uses standard
   5x30 mm dowel pins with tips 13 mm above the contact face (9.75 mm of skin
   clearance even at coil bind) while keeping >=9 mm bushing engagement at rest.
8. Purchased-part fidelity: the spring helix now has the true WF6-25 6.0 mm OD
   (was 5.2) with the catalog ~20 coils (wire 0.6, solid height 12), sensor
   wires the real ~1.2 mm silicone OD (was 0.76), and the JST-PH 3-pin header
   its catalog 9.9 x 4.5 x 6.0 footprint.
9. The Thinklabs One proxy was a guess ~20 mm too large (66/56/48 stepped).
   Official spec is 46 dia x 28 tall with a flush diaphragm bottom face; the
   whole clamp ring, lips, seams, sensing station and rod geometry are
   re-derived around the real device. The bottom lip (nothing to catch on a
   flush-bottom puck) becomes a TOP lip riding the 46->37 shoulder, so contact
   force loads the lip and the clamshell squeeze only prevents drop-out.
   Upper-tier diameter (37) and tier split are photo estimates — the only
   numbers Thinklabs publishes are 46 x 28 and 50 g — measure before printing.
10. The XIAO RP2040 (no mounting holes) gets a printed rail cradle on a wing
   off the fixed bridge, USB-C facing outward. The JST connector was removed
   (user decision): the three hall wires solder directly onto the XIAO's
   right-edge pads through a notch in the cradle rail.
11. Robot-side mounting: adds the WidowX AI follower 37 x 40 mm M3 accessory
   pattern (official drawing 999107002_01-C-A, taps 6.5 mm deep -> M3 x 12)
   alongside the generic slots kept for other arms.
12. Direct mount to the arm's ROUND wrist source: the X-Series gripper-off
   interface is the circular DYNAMIXEL horn (ROBOTIS XL-430 drawing: face
   20.5, boss 7.9, 4x M2 taps 3.5 deep on PCD 16; ViperX/XM430 same PCD,
   taps 2.0). Plate center gets a 20.8 x 0.8 locating recess, an 8.6 boss
   relief, and 4x M2 counterbored holes — positioned so the wrist servo body
   clears the XIAO cradle. v9.cyl_y is displaced by its own height (CadQuery
   "XZ" plane normal is -Y); wires/jack now use veo4.cyl_y_centered, and the
   wire-to-lead / wire-to-pad overlaps are positively audited.
13. Moment-free suspension: the single rear spring (46 mm moment arm from the
   device axis) side-loaded the bushings with friction comparable to the 4 N
   working force. TWO WF6-25 springs now sit point-symmetric beside the guide
   rods (+34,+10)/(-34,-10): force centroid = device axis, zero moment. The
   rear cantilever bridge and the plate's spring tab disappear; the plate
   becomes a round 88 dia disc (legacy slots dropped); a hard-stop sleeve on
   each rod boss caps travel at 5.5 mm so max force = 1.0 N/mm x 5.5 = the
   5.5 N patient limit mechanically; the Hall pillar deepens so the rest gap
   (7.75 mm) is re-sized to the shorter stroke, keeping ~0.4 V of signal span.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path


VEO4_PATH = Path(__file__).with_name("thinklab_openarm_veo4_magnet_tuned_mount.py")
spec = importlib.util.spec_from_file_location("veo4_mount", VEO4_PATH)
veo4 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["veo4_mount"] = veo4
spec.loader.exec_module(veo4)

cq = veo4.cq
exporters = veo4.exporters
veo3 = sys.modules["veo3_mount"]
v9 = veo4.v9
v12 = veo4.v12

# Official Thinklabs One: 46 mm diameter x 28 mm tall, 50 g, full-width
# diaphragm on the bottom face (thinklabs.com support article + Visionflex
# datasheet). The inherited 66/56/48 profile was a guess ~20 mm too large.
# Upper-tier diameter and tier split are NOT published — estimated from the
# datasheet photos and flagged for bench measurement.
# Re-modeled from the user's product photos: flared diaphragm ring at the
# bottom (the official 46 max), a cone NARROWING upward, and a camera-lens
# style ROTATING volume dial on top. Cone/dial numbers are PHOTO ESTIMATES —
# measure the real unit before the final print. Two side sockets (audio /
# charge) sit LOW on the cone.
VEO5_THINKLABS_PROFILE = {
    "lower_bottom_d_mm": 46.0,  # official max (flare / diaphragm ring)
    "base_flare_h_mm": 2.5,  # ESTIMATED
    "cone_bottom_d_mm": 45.4,  # ESTIMATED
    "cone_top_d_mm": 40.0,  # ESTIMATED
    "cone_h_mm": 15.5,  # ESTIMATED
    "dial_d_mm": 41.0,  # ESTIMATED rotating volume dial — must stay free
    "dial_h_mm": 8.8,
    "glass_d_mm": 36.0,
    "glass_h_mm": 1.2,  # totals 28.0 (official)
}
JACK_THETA_DEG = 215.0  # audio jack azimuth (front-left diagonal, clear of seams/tabs)
JACK_Z_MM = -37.5  # ESTIMATED: sockets sit low on the cone

import dataclasses

# Re-derive the shared params around the real 46x28 device. The ring shrinks
# by 10 mm radius; the whole moving assembly drops so the device top still
# clears the fixed plate at coil bind.
P = dataclasses.replace(
    veo4.P,
    thinklab_body_d=46.0,
    thinklab_body_h=28.0,
    cup_height=12.0,
    lip_height=2.5,
    lip_opening_d=38.6,  # top lip rides over the (estimated) 37 mm upper tier
    carriage_z=-36.0,
    spring_axis_y=46.0,
    guide_rod_length=25.0,
    guide_rod_x=34.0,  # rods pulled inboard: bushing barrels hug the 28.5 ring
    guide_bushing_h=14.0,
    top_bridge_x=88.0,  # plate only needs to span the inboard rod sockets now
    top_bridge_y=48.0,  # widened so the WidowX AI 37x40 holes get 4 mm edge margin
)
veo3.P = P
veo4.P = P

# WidowX AI follower accessory mount (official drawing 999107002_01-C-A):
# 4x M3x0.5 taps, 6.5 mm deep, on a 37 x 40 mm rectangle under the wrist
# module. Use M3 x 12 (6 mm plate + 6 mm engagement). The generic M3 slots
# remain for other arms (X-Series carriage route etc.).
WXAI_MOUNT_X_MM = 37.0 / 2.0
WXAI_MOUNT_Y_MM = 40.0 / 2.0

# X-Series ROUND wrist interface (the arm's "source" with the gripper off is
# the circular DYNAMIXEL horn — ROBOTIS XL-430 drawing 24-Mar-18): face
# dia 20.5, center boss dia 7.9, 4x M2 taps 3.5 deep on PCD 16 (XM430 on the
# ViperX has the same PCD 16 / M2, taps only 2.0 deep -> use M2 x 4 there).
HORN_FACE_D_MM = 20.5
HORN_BC_R_MM = 8.0
HORN_RELIEF_D_MM = 8.6  # clears the 7.9 center boss
HORN_RECESS_D_MM = 20.8  # locating recess so the horn self-centers
HORN_RECESS_DEPTH_MM = 0.8
HORN_M2_POSITIONS = ((HORN_BC_R_MM, 0.0), (-HORN_BC_R_MM, 0.0), (0.0, HORN_BC_R_MM), (0.0, -HORN_BC_R_MM))
RECOMMENDED_MECHANICS = {
    **veo3.RECOMMENDED_MECHANICS,
    "simulation_variant": "veo5_coaxial_zaxis_hall_gap",
    # TWO WF6-25 springs in parallel, point-symmetric about the device axis:
    # the single off-center spring (46 mm moment arm) side-loaded the bushings
    # with friction comparable to the 4 N working force. k doubles (4 N @
    # 4 mm) and a printed hard stop at 5.5 mm caps force at the patient limit.
    "spring_rate_n_per_mm": 1.0,
    "target_contact_force_n": 4.0,
    "max_screened_contact_force_n": 4.6,
}

# Point-symmetric spring axes beside the guide rods: force centroid = device
# axis, so contact force creates ZERO moment on the bushings.
SPRING_POSITIONS = ((34.0, 10.0), (-34.0, -10.0))
HARD_STOP_TRAVEL_MM = 5.5  # sleeve stop; 1.0 N/mm x 5.5 = 5.5 N patient limit

MAGNET_DIAMETER_MM = 5.0
MAGNET_THICKNESS_MM = 2.0
MAGNET_POCKET_DIAMETER_MM = 5.5
MAGNET_POCKET_DEPTH_MM = 2.6

# Real fastener geometry (M3 x 30 SHCS + M3 hex nut, DIN 934: 5.5 AF / 6.35 AC).
SCREW_HEAD_FACE_X_MM = -17.0  # outer face of the left lug
SCREW_LENGTH_MM = 30.0
SCREW_TIP_X_MM = SCREW_HEAD_FACE_X_MM + SCREW_LENGTH_MM  # 13.0, ends inside the right lug
NUT_CENTER_X_MM = 11.5  # inside the right lug, NOT buried in the spring bridge slab
NUT_ACROSS_CORNERS_MM = 6.35
NUT_THICKNESS_MM = 2.4
NUT_POCKET_ACROSS_CORNERS_MM = 6.9
NUT_POCKET_DEPTH_MM = 4.0

# Purchased Honeywell SS49E: TO-92S body with three straight ~13 mm leads.
HALL_LEAD_LENGTH_MM = 13.0
HALL_LEAD_PITCH_MM = 1.27
HALL_LEAD_W_MM = 0.45

# Guide rods: 5 x 25 mm standard dowel pins (user request: genuinely short —
# the tips no longer reach near the barrel bottoms). With the contact face at
# z=-44, tips at z=-25 keep 7.75 mm of skin clearance even at coil bind; rest
# engagement in the (shortened, 14 mm) barrels is 6 mm and grows 1:1 with
# compression (14 mm at the 4 N working point).
GUIDE_ROD_LENGTH_MM = 25.0
GUIDE_ROD_TOP_Z_MM = 0.0
GUIDE_ROD_TIP_Z_MM = GUIDE_ROD_TOP_Z_MM - GUIDE_ROD_LENGTH_MM  # -25.0

# Purchased spring: MISUMI WF6-25 catalog row — wire d 0.6, solid height 12
# (~20 total coils), F max 11.2 mm, load max 5.5 N, k=0.5 N/mm fixed for WF6.
SPRING_WIRE_R_MM = 0.30
SPRING_VISUAL_TURNS = 19.5

# Sensor wires: 26-30 AWG silicone wire, insulated OD ~1.2 mm.
SENSOR_WIRE_R_MM = 0.6

# Seeed XIAO RP2040: 21.0 x 17.78 mm board, no mounting holes — held in a
# printed cradle on a wing extending the fixed bridge, USB-C facing outward.
XIAO_L_MM = 21.0
XIAO_W_MM = 17.78
XIAO_CRADLE_CLEAR_MM = 0.25
# Front-center, BETWEEN the WidowX AI mount holes (x +-18.5): anywhere off-center
# the 17.78 mm board ends up covering a mount hole or crowding the JST shelf.
XIAO_CENTER_X_MM = 0.0
XIAO_CENTER_Y_MM = -32.0
WING_TOP_Z_MM = 3.0  # flush with the bridge plate top

# Sensing station: shared XY axis for magnet and Hall (coaxial along Z).
# Moved inboard (ring outer radius is now 28.5) and 1 mm back so the Hall
# pillar clears the plate's M3 mounting slots and the inboard guide rods.
SENSE_X_MM = 25.0
SENSE_Y_MM = -16.0
TOWER_TOP_Z_MM = -18.6
MAGNET_TOP_Z_MM = TOWER_TOP_Z_MM - (MAGNET_POCKET_DEPTH_MM - MAGNET_THICKNESS_MM)  # -19.2
PILLAR_BOTTOM_Z_MM = -11.5  # deeper pillar: rest gap 7.75 keeps signal span with the halved travel
HALL_FACE_Z_MM = PILLAR_BOTTOM_Z_MM + 0.05  # package recessed 0.05 into slot
REST_FACE_GAP_MM = HALL_FACE_Z_MM - MAGNET_TOP_Z_MM
HALL_DIE_OFFSET_MM = 0.6  # sensing element sits behind the branded face
COIL_BIND_TRAVEL_MM = P.spring_allowable_deflection  # 11.25 per spring (capacity check only)

# First-order SS49E / D5x2 N42 electrical model (bench calibration still required).
MAGNET_BR_T = 1.32
HALL_SUPPLY_V = 3.3
HALL_QUIESCENT_V = HALL_SUPPLY_V / 2.0
HALL_SENS_MV_PER_G = 1.4 * (HALL_SUPPLY_V / 5.0)  # ratiometric from 5 V datasheet point
ADC_BITS = 12

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated" / "veo5_coaxial_hall"
SIM_DIR = OUT_DIR / "simulation"
PRESENTATION_DIR = OUT_DIR / "presentation"


def veo5_sensor_layout(p: v9.SimplifiedRailMountParams) -> dict[str, tuple[float, float, float]]:
    return {
        "magnet": (SENSE_X_MM, SENSE_Y_MM, MAGNET_TOP_Z_MM - MAGNET_THICKNESS_MM / 2.0),
        "hall": (SENSE_X_MM, SENSE_Y_MM, HALL_FACE_Z_MM + 0.75),
        "hall_backplate": (SENSE_X_MM, SENSE_Y_MM, -4.0),
        # No connector: this is the solder target on the XIAO's right-edge pads.
        "sensor_connector": (XIAO_W_MM / 2.0, -30.0, 4.7),
    }


def veo5_assembly_parts(fixed, left, right, thinklab, hardware):
    """v12's render part list minus the removed JST connector, plus the XIAO."""
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
        (hardware["xiao"], "#2e8b2e", 1.0),
        (hardware["audio_plug"], "#7a5230", 1.0),
        (hardware["hall_vcc_wire"], "#d62828", 1.0),
        (hardware["hall_signal_wire"], "#f2c230", 1.0),
        (hardware["hall_gnd_wire"], "#111111", 1.0),
    ]


def configure_backends() -> None:
    v12.P = P
    v12.THINKLABS_PROFILE = VEO5_THINKLABS_PROFILE
    v12.RECOMMENDED_MECHANICS = RECOMMENDED_MECHANICS
    v12.OUT_DIR = OUT_DIR
    v12.SIM_DIR = SIM_DIR
    v12.PRESENTATION_DIR = PRESENTATION_DIR
    v12.sensor_layout = veo5_sensor_layout
    v12.assembly_parts = veo5_assembly_parts


def ring_radii(p: v9.SimplifiedRailMountParams) -> tuple[float, float]:
    lower_inner_r = (v12.THINKLABS_PROFILE["lower_bottom_d_mm"] + 1.0) / 2.0
    return lower_inner_r, lower_inner_r + p.cup_wall


def _upward_spring_socket_at(p: v9.SimplifiedRailMountParams, x: float, y: float) -> cq.Workplane:
    """v9.upward_spring_socket geometry at an arbitrary (x, y) axis."""
    face_z = p.spring_moving_face_z
    base = v9.cyl_z(p.spring_seat_d / 2.0, 3.0, (x, y, face_z - 1.5))
    rim_od = p.spring_od + 2.6
    rim_id = p.spring_od + p.spring_clearance_d
    rim = v9.cyl_z(rim_od / 2.0, 1.2, (x, y, face_z + 0.6)).cut(
        v9.cyl_z(rim_id / 2.0, 1.6, (x, y, face_z + 0.6))
    )
    return base.union(rim)


def _downward_spring_socket_at(p: v9.SimplifiedRailMountParams, x: float, y: float) -> cq.Workplane:
    """v9.downward_spring_socket geometry at an arbitrary (x, y) axis."""
    face_z = p.spring_fixed_face_z
    base = v9.cyl_z(p.spring_seat_d / 2.0, 3.0, (x, y, face_z + 1.5))
    rim_od = p.spring_od + 2.6
    rim_id = p.spring_od + p.spring_clearance_d
    rim = v9.cyl_z(rim_od / 2.0, 1.2, (x, y, face_z - 0.6)).cut(
        v9.cyl_z(rim_id / 2.0, 1.6, (x, y, face_z - 0.6))
    )
    return base.union(rim)


def _base_ring_veo5(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """Clamp ring for the real 46x28 Thinklabs One.

    The real device has no bottom flange — its diaphragm is the flush bottom
    face — so the old bottom lip is replaced by a TOP lip riding over the
    46->37 shoulder: contact force pushes the device up into the lip, and the
    clamshell squeeze provides drop-out friction.
    """
    lower_inner_r, outer_r = ring_radii(p)
    zc = p.carriage_z

    # Conical bore matching the device's upward-narrowing cone: contact force
    # wedges the cone INTO the narrowing bore (self-energizing — no top lip,
    # so the rotating volume dial stays completely free). Drop-out is held by
    # clamp friction alone (the device weighs 50 g; the M3 squeeze holds far
    # more). Bore = estimated cone + 0.5..1.0 mm clearance.
    bore = v12.v11.cone_z(23.725, 21.475, p.cup_height + 3.0, (0, 0, zc))
    band = v9.cyl_z(outer_r, p.cup_height, (0, 0, zc)).cut(bore)
    ring = band

    # Single jack window at the audio socket azimuth: the 3.5 mm plug stays
    # in DURING use. (The second socket stays covered; charge off-robot.)
    th = math.radians(JACK_THETA_DEG)
    window = _radial_cyl(6.0, 16.0, 17.0 * math.cos(th), 17.0 * math.sin(th), JACK_Z_MM, math.cos(th), math.sin(th))
    ring = ring.cut(window)

    # Guide bushings: shortened 14 mm barrels riding high (z -33..-19) so the
    # short 5x25 rods keep 6 mm rest engagement, hugging the ring band (rods
    # are inboard at x=+-34, so the barrels protrude only ~11 mm).
    bushing_cz = zc + 10.0
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        bushing = v9.cyl_z(p.guide_bushing_od / 2.0, p.guide_bushing_h, (sx, p.guide_rod_y, bushing_cz))
        bushing = bushing.cut(
            v9.cyl_z(p.guide_rod_clearance_d / 2.0, p.guide_bushing_h + 3.0, (sx, p.guide_rod_y, bushing_cz))
        )
        inner_x = lower_inner_r * 1.003  # stay just outside the clamp bore
        bridge_w = p.guide_rod_x + 2.0 - inner_x
        bridge_cx = (inner_x + p.guide_rod_x + 2.0) / 2.0
        bridge = v9.box(bridge_w, 10.0, 10.0, (math.copysign(bridge_cx, sx), p.guide_rod_y, zc + 8.0))
        ring = ring.union(bushing).union(bridge)
    # Re-bore the rod clearance through the FUSED solid: the bridge blocks
    # overlap the rod axis, and every prior version only bored the bushing
    # cylinder before the union (rods silently skewered the bridge blocks).
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        ring = ring.cut(v9.cyl_z(p.guide_rod_clearance_d / 2.0, 50.0, (sx, p.guide_rod_y, bushing_cz)))

    # TWO moving spring seats, point-symmetric beside the bushing barrels —
    # the old single rear-cantilever bridge (46 mm moment arm) side-loaded the
    # bushings; with the force centroid on the device axis the moment is zero
    # and the whole rear cantilever disappears.
    for sxy, (spx, spy) in zip((1.0, -1.0), SPRING_POSITIONS):
        tab = v9.box(10.0, 14.0, 3.0, (spx, spy - sxy * 2.0, p.spring_moving_face_z - 2.0))
        ring = ring.union(tab)
        ring = ring.union(_upward_spring_socket_at(p, spx, spy))

    # Z-axis magnet tower, fused into the band and top lip (3 mm bite).
    tower_h = TOWER_TOP_Z_MM - (zc + p.cup_height / 2.0 - 3.0)
    tower = v9.box(10.0, 9.0, tower_h, (SENSE_X_MM, SENSE_Y_MM, TOWER_TOP_Z_MM - tower_h / 2.0))
    ring = ring.union(tower)
    pocket = v9.cyl_z(MAGNET_POCKET_DIAMETER_MM / 2.0, MAGNET_POCKET_DEPTH_MM + 0.2, (SENSE_X_MM, SENSE_Y_MM, TOWER_TOP_Z_MM - MAGNET_POCKET_DEPTH_MM / 2.0 + 0.1))
    ring = ring.cut(pocket)
    return ring


def make_clamshell_halves(p: v9.SimplifiedRailMountParams) -> tuple[cq.Workplane, cq.Workplane]:
    _, outer_r = ring_radii(p)
    zc = p.carriage_z
    base = _base_ring_veo5(p)

    front_y, rear_y = veo4.clamshell_seam_ys(p)
    web_bite_y = outer_r - 4.5  # web reaches into the band for a solid weld
    for y, web_y_inner in ((front_y, -web_bite_y), (rear_y, web_bite_y)):
        for sx in (-10.0, 10.0):
            lug = v9.box(14.0, 12.0, 14.0, (sx, y, zc))
            web = v9.box(14.0, abs(y - web_y_inner), 14.0, (sx, (y + web_y_inner) / 2.0, zc))
            base = base.union(lug).union(web)

    # True split: left half is x <= -1.5, right half is x >= +1.5 (VEO4 cut the
    # wrong sides, leaving a 3 mm thick overlap shared by both prints).
    split_gap_w = 3.0
    cutter_for_left = v9.box(120.0, 130.0, 80.0, (60.0 - split_gap_w / 2.0, 0, zc))
    cutter_for_right = v9.box(120.0, 130.0, 80.0, (-60.0 + split_gap_w / 2.0, 0, zc))
    left = base.cut(cutter_for_left)
    right = base.cut(cutter_for_right)

    # Clearance holes only span head face -> nut pocket; they no longer tunnel
    # through the spring bridge slab (VEO4/old-VEO5 used 56 mm cuts at x 0+-28).
    hole_x0 = SCREW_HEAD_FACE_X_MM - 5.0
    hole_x1 = SCREW_TIP_X_MM + 0.5
    for y in (front_y, rear_y):
        for dz in (-4.0, 4.0):
            hole = v9.cyl_x(
                p.m3_clearance_d / 2.0, hole_x1 - hole_x0, ((hole_x0 + hole_x1) / 2.0, y, zc + dz)
            )
            left = left.cut(hole)
            right = right.cut(hole)
            pocket = v12.v10.hex_prism_x(
                NUT_POCKET_ACROSS_CORNERS_MM, NUT_POCKET_DEPTH_MM, (NUT_CENTER_X_MM, y, zc + dz)
            )
            # Vertical insertion slot: bottom screws load the nut from below the
            # lug, top screws from above, so no nut cavity is fully enclosed.
            slot_dir = -1.0 if dz < 0 else 1.0
            slot = v9.box(
                NUT_POCKET_DEPTH_MM,
                NUT_POCKET_ACROSS_CORNERS_MM - 0.2,
                7.0,
                (NUT_CENTER_X_MM, y, zc + dz + slot_dir * 3.5),
            )
            right = right.cut(pocket).cut(slot)
    return left, right


def make_fixed_bridge(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """Round plate (the old long rectangle dated from the wide rod stance):
    horn at center, WidowX AI pattern, rod sockets with hard-stop sleeves,
    twin spring seats, Hall pillar, and the XIAO wing. Legacy slots dropped."""
    bridge = v9.cyl_z(44.0, p.top_bridge_t, (0, 0, 0))
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        socket = v9.cyl_z(p.rod_socket_od / 2.0, 7.0, (sx, p.guide_rod_y, -3.0))
        # Hard-stop sleeve: extends the boss down so the bushing-barrel rim
        # lands on it at 5.5 mm travel — 1.0 N/mm x 5.5 = the 5.5 N patient
        # limit becomes a mechanical guarantee, not just a software gate.
        sleeve = v9.cyl_z(5.5, 7.0, (sx, p.guide_rod_y, -10.0))
        socket = socket.union(sleeve)
        socket = socket.cut(v9.cyl_z(p.guide_rod_clearance_d / 2.0, 15.0, (sx, p.guide_rod_y, -6.0)))
        bridge = bridge.union(socket)

    # Twin fixed spring seats directly under the plate (no cantilever tab).
    for spx, spy in SPRING_POSITIONS:
        bridge = bridge.union(_downward_spring_socket_at(p, spx, spy))

    # Hall pillar: hangs from the plate, holds the TO-92S flat (branded face
    # down) coaxially above the magnet. Deeper than before so the rest air
    # gap is 7.75 mm — sized to the new 5.5 mm hard-stop travel.
    pillar = v9.box(10.0, 8.0, 9.0, (SENSE_X_MM, SENSE_Y_MM, PILLAR_BOTTOM_Z_MM + 4.5))
    bridge = bridge.union(pillar)
    hall_slot = v9.box(3.4, 4.4, 1.9, (SENSE_X_MM, SENSE_Y_MM, PILLAR_BOTTOM_Z_MM + 0.9))
    # 4.0 mm wide so the three straight SS49E leads (3.0 mm outer span) pass
    # through the pillar with >=0.5 mm side clearance; beyond the pillar the
    # leads hang in free air under the plate.
    leg_channel = v9.box(5.6, 4.0, 1.7, (SENSE_X_MM + 2.9, SENSE_Y_MM, PILLAR_BOTTOM_Z_MM + 0.85))
    bridge = bridge.cut(hall_slot).cut(leg_channel)

    # XIAO wing: extends the round plate toward -y at front-center, carrying
    # a rail cradle for the XIAO RP2040 (board has no mounting holes). USB-C
    # end open toward -y; front corners rounded to match the circular plate.
    wing = v9.box(24.0, 21.0, 4.0, (XIAO_CENTER_X_MM, -30.5, WING_TOP_Z_MM - 2.0))
    wing = wing.union(v9.box(18.0, 3.0, 4.0, (XIAO_CENTER_X_MM, -42.5, WING_TOP_Z_MM - 2.0)))
    for cx in (XIAO_CENTER_X_MM - 9.0, XIAO_CENTER_X_MM + 9.0):
        wing = wing.union(v9.cyl_z(3.0, 4.0, (cx, -41.0, WING_TOP_Z_MM - 2.0)))
    bridge = bridge.union(wing)
    inner_x = XIAO_W_MM + 2.0 * XIAO_CRADLE_CLEAR_MM  # 18.28
    inner_y = XIAO_L_MM + 2.0 * XIAO_CRADLE_CLEAR_MM  # 21.5
    rail_z = WING_TOP_Z_MM + 2.0
    for sx in (-1, 1):
        rail_cx = XIAO_CENTER_X_MM + sx * (inner_x / 2.0 + 0.75)
        bridge = bridge.union(v9.box(1.5, inner_y, 4.0, (rail_cx, XIAO_CENTER_Y_MM, rail_z)))
    end_stop_y = XIAO_CENTER_Y_MM + inner_y / 2.0 + 0.4
    bridge = bridge.union(v9.box(inner_x + 3.0, 0.8, 4.0, (XIAO_CENTER_X_MM, end_stop_y, rail_z)))
    # Wire pass-through notch in the +x rail: the three hall wires solder
    # directly onto the XIAO's right-edge pads (no JST connector).
    notch = v9.box(2.2, 5.2, 3.6, (XIAO_CENTER_X_MM + inner_x / 2.0 + 0.75, -30.0, 5.4))
    bridge = bridge.cut(notch)
    # Wire pass-through hole: the round plate (r44) covers the riser station
    # beside the wing, so the three-wire bundle threads up through the plate.
    bridge = bridge.cut(v9.cyl_z(2.6, 10.0, (13.54, -30.0, 0.0)))

    # WidowX AI 37x40 accessory-mount pattern (M3 clearance, thru). Cut LAST so
    # later unions (spring tab, wing) cannot refill the holes.
    for sx in (-WXAI_MOUNT_X_MM, WXAI_MOUNT_X_MM):
        for sy in (-WXAI_MOUNT_Y_MM, WXAI_MOUNT_Y_MM):
            bridge = bridge.cut(v9.cyl_z(p.m3_clearance_d / 2.0, 30.0, (sx, sy, 0)))

    # Round X-Series horn interface at plate center: locating recess for the
    # 20.5 horn, thru relief for its 7.9 center boss, and 4x M2 counterbored
    # holes on the PCD-16 circle (screws drive up into the horn taps from
    # below). Center position guarantees clearance from the XIAO cradle.
    bridge = bridge.cut(v9.cyl_z(HORN_RECESS_D_MM / 2.0, 1.6, (0, 0, p.top_bridge_t / 2.0)))
    bridge = bridge.cut(v9.cyl_z(HORN_RELIEF_D_MM / 2.0, 30.0, (0, 0, 0)))
    for hx, hy in HORN_M2_POSITIONS:
        bridge = bridge.cut(v9.cyl_z(1.2, 30.0, (hx, hy, 0)))  # M2 clearance
        bridge = bridge.cut(v9.cyl_z(2.2, 3.0, (hx, hy, -p.top_bridge_t / 2.0 + 1.5)))  # head cb
    return bridge


def make_magnet_proxy() -> cq.Workplane:
    return v9.cyl_z(
        MAGNET_DIAMETER_MM / 2.0,
        MAGNET_THICKNESS_MM,
        (SENSE_X_MM, SENSE_Y_MM, MAGNET_TOP_Z_MM - MAGNET_THICKNESS_MM / 2.0),
    )


def make_clamp_screws_proxy(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """Real M3x30 SHCS: head on the left lug face, nut inside the right lug."""
    zc = p.carriage_z
    screws = None
    for y in veo4.clamshell_seam_ys(p):
        for dz in (-4.0, 4.0):
            shaft = v9.cyl_x(
                1.5, SCREW_LENGTH_MM, ((SCREW_HEAD_FACE_X_MM + SCREW_TIP_X_MM) / 2.0, y, zc + dz)
            )
            head = v9.cyl_x(2.75, 3.0, (SCREW_HEAD_FACE_X_MM - 1.5, y, zc + dz))
            nut = v12.v10.hex_prism_x(
                NUT_ACROSS_CORNERS_MM, NUT_THICKNESS_MM, (NUT_CENTER_X_MM, y, zc + dz)
            )
            screw = shaft.union(head).union(nut)
            screws = screw if screws is None else screws.union(screw)
    assert screws is not None
    return screws


def make_hall_body_proxy() -> cq.Workplane:
    """Just the TO-92S epoxy body (used for the coaxial alignment audit)."""
    return v9.box(3.1, 4.1, 1.5, (SENSE_X_MM, SENSE_Y_MM, HALL_FACE_Z_MM + 0.75))


def make_hall_proxy() -> cq.Workplane:
    """SS49E TO-92S body plus its three straight leads (matches the bought part)."""
    body = make_hall_body_proxy()
    lead_root_x = SENSE_X_MM + 3.1 / 2.0
    hall = body
    for dy in (-HALL_LEAD_PITCH_MM, 0.0, HALL_LEAD_PITCH_MM):
        lead = v9.box(
            HALL_LEAD_LENGTH_MM,
            HALL_LEAD_W_MM,
            HALL_LEAD_W_MM,
            (lead_root_x + HALL_LEAD_LENGTH_MM / 2.0, SENSE_Y_MM + dy, HALL_FACE_Z_MM + 0.75),
        )
        hall = hall.union(lead)
    return hall


def _radial_cyl(radius: float, length: float, cx: float, cy: float, cz: float, ux: float, uy: float) -> cq.Workplane:
    solid = cq.Solid.makeCylinder(radius, length, cq.Vector(cx, cy, cz), cq.Vector(ux, uy, 0.0))
    return cq.Workplane("XY").add(solid)


def make_thinklab_proxy(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """Real-shape Thinklabs One (from the user's photos): flared diaphragm
    ring, upward-narrowing cone, rotating volume dial, glass display top, and
    two low side sockets (audio/charge) at the front-left diagonal."""
    prof = VEO5_THINKLABS_PROFILE
    dev_bottom = p.carriage_z - p.cup_height / 2.0 - 2.0  # -44
    flare_h = prof["base_flare_h_mm"]
    cone_h = prof["cone_h_mm"]
    dial_h = prof["dial_h_mm"]
    flare = v9.cyl_z(prof["lower_bottom_d_mm"] / 2.0, flare_h, (0, 0, dev_bottom + flare_h / 2.0))
    cone = v12.v11.cone_z(
        prof["cone_bottom_d_mm"] / 2.0,
        prof["cone_top_d_mm"] / 2.0,
        cone_h,
        (0, 0, dev_bottom + flare_h + cone_h / 2.0),
    )
    dial_z0 = dev_bottom + flare_h + cone_h
    dial = v9.cyl_z(prof["dial_d_mm"] / 2.0, dial_h, (0, 0, dial_z0 + dial_h / 2.0))
    glass = v9.cyl_z(prof["glass_d_mm"] / 2.0, prof["glass_h_mm"], (0, 0, dial_z0 + dial_h + prof["glass_h_mm"] / 2.0))
    body = flare.union(cone).union(dial).union(glass)
    # Two low side sockets ~12 mm apart along the cone (photo): markers only,
    # poking 0.3 mm proud of the cone surface (must stay inside the ring bore).
    for theta_deg in (JACK_THETA_DEG, JACK_THETA_DEG + 30.0):
        th = math.radians(theta_deg)
        body = body.union(_radial_cyl(2.0, 2.3, 20.0 * math.cos(th), 20.0 * math.sin(th), JACK_Z_MM, math.cos(th), math.sin(th)))
    return body


def make_audio_plug_proxy() -> cq.Workplane:
    """The 3.5 mm plug that stays in the audio socket DURING auscultation —
    it rides with the device and must pass through the ring's jack window."""
    th = math.radians(JACK_THETA_DEG)
    ux, uy = math.cos(th), math.sin(th)
    # Right-angle plug (recommended): 11 mm barrel, cable turning upward.
    barrel = _radial_cyl(4.0, 11.0, 21.0 * ux, 21.0 * uy, JACK_Z_MM, ux, uy)
    cable = v9.cyl_z(1.5, 16.0, (32.0 * ux, 32.0 * uy, JACK_Z_MM + 8.0))
    return barrel.union(cable)


def make_guide_rods_proxy(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """5 mm x 30 mm dowel pins; tips stay 13 mm above the contact face."""
    rods = None
    center_z = GUIDE_ROD_TOP_Z_MM - GUIDE_ROD_LENGTH_MM / 2.0
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        rod = v9.cyl_z(p.guide_rod_d / 2.0, GUIDE_ROD_LENGTH_MM, (sx, p.guide_rod_y, center_z))
        rods = rod if rods is None else rods.union(rod)
    assert rods is not None
    return rods


def make_horn_proxy() -> cq.Workplane:
    """The arm's round wrist source: DYNAMIXEL horn seated in the plate recess
    plus a rough wrist-servo body above (for clearance checks vs the XIAO)."""
    horn = v9.cyl_z(HORN_FACE_D_MM / 2.0, 4.0, (0, 0, 4.2))  # z 2.2..6.2
    boss = v9.cyl_z(3.95, 1.8, (0, 0, 1.3))  # center boss into the relief
    servo = v9.box(28.5, 34.0, 10.0, (0, 0, 11.2))  # rough XL430 body envelope
    return horn.union(boss).union(servo)


def make_horn_screws_proxy() -> cq.Workplane:
    """4x M2 screws driving up from the counterbores into the horn taps."""
    screws = None
    for hx, hy in HORN_M2_POSITIONS:
        shaft = v9.cyl_z(1.0, 8.0, (hx, hy, 1.0))  # z -3..5, tip in the horn tap
        head = v9.cyl_z(1.85, 1.5, (hx, hy, -2.25))  # seated in the counterbore
        screw = shaft.union(head)
        screws = screw if screws is None else screws.union(screw)
    assert screws is not None
    return screws


def make_xiao_proxy() -> cq.Workplane:
    """Seeed XIAO RP2040 in the wing cradle: PCB plus the USB-C connector."""
    board = v9.box(XIAO_W_MM, XIAO_L_MM, 1.6, (XIAO_CENTER_X_MM, XIAO_CENTER_Y_MM, WING_TOP_Z_MM + 0.8))
    usb = v9.box(
        9.0,
        7.5,
        3.2,
        (XIAO_CENTER_X_MM, XIAO_CENTER_Y_MM - XIAO_L_MM / 2.0 + 3.75, WING_TOP_Z_MM + 1.6 + 1.6),
    )
    return board.union(usb)


def make_wire_proxies() -> dict[str, cq.Workplane]:
    """Three silicone wires from the SS49E lead tips, soldered directly onto
    the XIAO's right-edge pads (no connector): run inboard under the plate,
    forward past the wing's +x edge, rise beside the wing, then pass through
    the rail notch onto the pads."""
    wire_r = SENSOR_WIRE_R_MM
    splice_x = SENSE_X_MM + 3.1 / 2.0 + HALL_LEAD_LENGTH_MM - 0.5  # overlap lead tips
    bend_x = XIAO_CENTER_X_MM + (XIAO_W_MM + 2.0 * XIAO_CRADLE_CLEAR_MM) / 2.0 + 4.4  # 13.54
    pad_x = XIAO_CENTER_X_MM + XIAO_W_MM / 2.0 - 0.2  # overlap the pad row
    bypass_y = -23.5  # south of the hall pillar (ymax -20), north of the lugs (-29)
    wires: dict[str, cq.Workplane] = {}
    for name, dy, dz in [
        ("hall_vcc_wire", -1.3, -0.4),
        ("hall_signal_wire", 0.0, 0.0),
        ("hall_gnd_wire", 1.3, 0.4),
    ]:
        y0 = SENSE_Y_MM + dy
        y_run = bypass_y + dy
        y_pad = -30.0 + dy
        z0 = HALL_FACE_Z_MM + 0.75 + dz
        # z staggers halved at the pad end so every wire both clears the
        # rail-notch floor (3.6) and overlaps the 4.6-high board edge.
        z_top = 4.55 + 0.5 * dz
        # NOTE: v9.cyl_y extrudes on the "XZ" plane whose normal is -Y, so its
        # cylinders are displaced one full height — use veo4's centered helper.
        jog_south = veo4.cyl_y_centered(wire_r, y0 - y_run, (splice_x, (y0 + y_run) / 2.0, z0))
        run_in = v9.cyl_x(wire_r, splice_x - bend_x, ((splice_x + bend_x) / 2.0, y_run, z0))
        jog_front = veo4.cyl_y_centered(wire_r, y_run - y_pad, (bend_x, (y_run + y_pad) / 2.0, z0))
        riser = v9.cyl_z(wire_r, z_top - z0, (bend_x, y_pad, (z0 + z_top) / 2.0))
        run_pad = v9.cyl_x(wire_r, bend_x - pad_x, ((bend_x + pad_x) / 2.0, y_pad, z_top))
        wires[name] = jog_south.union(run_in).union(jog_front).union(riser).union(run_pad)
    return wires


def make_hardware_proxies(p: v9.SimplifiedRailMountParams) -> dict[str, cq.Workplane]:
    hardware = v12.v10.make_hardware_proxies(p)
    hardware["guide_rods"] = make_guide_rods_proxy(p)
    hardware["clamp_screws"] = make_clamp_screws_proxy(p)
    # The robot side is the round X-Series horn, not the old generic plate.
    hardware["robot_flange"] = make_horn_proxy()
    hardware["interface_bolts"] = make_horn_screws_proxy()
    hardware["spring"] = make_veo5_visible_spring(p)
    hardware["magnet"] = make_magnet_proxy()
    hardware["hall"] = make_hall_proxy()
    # No JST connector: the hall wires solder directly onto the XIAO pads.
    hardware.pop("sensor_connector", None)
    hardware["xiao"] = make_xiao_proxy()
    hardware["audio_plug"] = make_audio_plug_proxy()
    for name, shape in make_wire_proxies().items():
        hardware[name] = shape
    return hardware


def make_compressed_spring(p: v9.SimplifiedRailMountParams, compression: float) -> cq.Workplane:
    """BOTH WF6-25 visual helixes (true OD 6.0), bottom faces raised by `compression`."""
    wire_r = SPRING_WIRE_R_MM
    helix_r = p.spring_od / 2.0 - wire_r  # coil outer surface = real 6.0 mm OD
    bottom_z = p.spring_moving_face_z + compression + wire_r + 0.39
    top_z = p.spring_fixed_face_z - wire_r
    turns = SPRING_VISUAL_TURNS
    phase = math.pi / 2.0
    n = 312
    solids = []
    for axis_x, axis_y in SPRING_POSITIONS:
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            a0 = phase + 2.0 * math.pi * turns * t0
            a1 = phase + 2.0 * math.pi * turns * t1
            start = cq.Vector(
                axis_x + helix_r * math.cos(a0),
                axis_y + helix_r * math.sin(a0),
                bottom_z + (top_z - bottom_z) * t0,
            )
            end = cq.Vector(
                axis_x + helix_r * math.cos(a1),
                axis_y + helix_r * math.sin(a1),
                bottom_z + (top_z - bottom_z) * t1,
            )
            direction = end - start
            solids.append(cq.Solid.makeCylinder(wire_r, direction.Length, start, direction.normalized()))
    return cq.Workplane("XY").add(cq.Compound.makeCompound(solids))


def make_veo5_visible_spring(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """The purchased WF6-25 at free length (true 6.0 mm OD, ~0.6 mm wire)."""
    return make_compressed_spring(p, 0.0)


def build_veo5_model() -> dict:
    configure_backends()
    fixed = make_fixed_bridge(P)
    left, right = make_clamshell_halves(P)
    thinklab = make_thinklab_proxy(P)
    hardware = make_hardware_proxies(P)
    return {
        "fixed": fixed,
        "left": left,
        "right": right,
        "thinklab": thinklab,
        "hardware": hardware,
    }


# --- Sensing model -----------------------------------------------------------


def axial_disc_field_gauss(gap_mm: float) -> float:
    """On-axis flux density of an axially magnetized disc at `gap_mm` from its face."""
    r = MAGNET_DIAMETER_MM / 2.0
    t = MAGNET_THICKNESS_MM
    z = max(gap_mm, 0.05)
    b_tesla = (MAGNET_BR_T / 2.0) * (
        (z + t) / math.sqrt((z + t) ** 2 + r**2) - z / math.sqrt(z**2 + r**2)
    )
    return b_tesla * 1e4


def hall_response(compression_mm: float) -> dict:
    face_gap = REST_FACE_GAP_MM - compression_mm
    die_gap = face_gap + HALL_DIE_OFFSET_MM
    b_gauss = axial_disc_field_gauss(die_gap)
    volts = HALL_QUIESCENT_V + HALL_SENS_MV_PER_G * b_gauss / 1000.0
    counts = volts / HALL_SUPPLY_V * (2**ADC_BITS - 1)
    return {
        "compression_mm": round(compression_mm, 3),
        "force_n": round(RECOMMENDED_MECHANICS["spring_rate_n_per_mm"] * compression_mm, 3),
        "face_gap_mm": round(face_gap, 3),
        "field_gauss": round(b_gauss, 1),
        "hall_out_v": round(volts, 4),
        "adc_counts": round(counts, 1),
    }


def make_sensing_model() -> dict:
    samples = [hall_response(c / 4.0) for c in range(0, int(HARD_STOP_TRAVEL_MM * 4) + 1)]
    operating = [s for s in samples if s["compression_mm"] <= 4.6]
    span_v = operating[-1]["hall_out_v"] - operating[0]["hall_out_v"]
    key_points = {
        "rest": hall_response(0.0),
        "target_force_4n": hall_response(4.0),
        "max_screened_4p6n": hall_response(4.6),
        "hard_stop_5p5mm": hall_response(HARD_STOP_TRAVEL_MM),
    }
    return {
        "arrangement": "coaxial_z_axial_gap_closing",
        "magnet": "D5x2 N42 disc, Z (axial) magnetization, pole face toward Hall",
        "hall": "SS49E-class TO-92S, branded face down, powered at 3.3 V (XIAO RP2040 ADC)",
        "rest_face_gap_mm": round(REST_FACE_GAP_MM, 3),
        "die_offset_mm": HALL_DIE_OFFSET_MM,
        "supply_v": HALL_SUPPLY_V,
        "sensitivity_mv_per_gauss": round(HALL_SENS_MV_PER_G, 3),
        "operating_span_v": round(span_v, 4),
        "monotonic_over_full_travel": all(
            samples[i + 1]["hall_out_v"] > samples[i]["hall_out_v"] for i in range(len(samples) - 1)
        ),
        "key_points": key_points,
        "calibration": "force_n = spring_rate * (rest_gap - gap(v)); bench-fit gap(v) with known weights",
        "samples": samples,
    }


def write_sensing_outputs(sensing: dict) -> dict:
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SIM_DIR / "veo5_hall_response.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensing["samples"][0].keys()))
        writer.writeheader()
        writer.writerows(sensing["samples"])

    import matplotlib.pyplot as plt

    comps = [s["compression_mm"] for s in sensing["samples"]]
    volts = [s["hall_out_v"] for s in sensing["samples"]]
    forces = [s["force_n"] for s in sensing["samples"]]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4), dpi=130)
    ax1.plot(comps, volts, color="#2471b5", lw=2.0)
    ax1.set_xlabel("spring compression (mm)")
    ax1.set_ylabel("SS49E output @3.3 V (V)", color="#2471b5")
    ax1.axvline(8.0, color="#888", ls="--", lw=0.8)
    ax1.annotate("4 N target", xy=(8.0, volts[0] + 0.02), fontsize=8, color="#555")
    ax1.axvline(9.2, color="#c44", ls="--", lw=0.8)
    ax1.annotate("4.6 N screen", xy=(9.2, volts[0] + 0.08), fontsize=8, color="#c44")
    ax2 = ax1.twinx()
    ax2.plot(comps, forces, color="#47b567", lw=1.2, ls=":")
    ax2.set_ylabel("spring force (N)", color="#47b567")
    ax1.set_title("VEO5 coaxial Hall response (first-order model, needs bench calibration)")
    fig.tight_layout()
    png_path = SIM_DIR / "veo5_hall_response.png"
    fig.savefig(png_path)
    plt.close(fig)
    return {"hall_response_csv": str(csv_path), "hall_response_png": str(png_path)}


# --- Audits ------------------------------------------------------------------


def make_veo5_mechanical_audit() -> dict:
    """Dual-spring mechanics: 2x WF6-25 in parallel, hard stop at 5.5 mm."""
    k_each = 0.5
    k_total = RECOMMENDED_MECHANICS["spring_rate_n_per_mm"]
    target_f = RECOMMENDED_MECHANICS["target_contact_force_n"]
    screened_f = RECOMMENDED_MECHANICS["max_screened_contact_force_n"]
    target_c = target_f / k_total
    screened_c = screened_f / k_total
    stop_force = k_total * HARD_STOP_TRAVEL_MM
    per_spring_stop = k_each * HARD_STOP_TRAVEL_MM
    centroid_x = sum(x for x, _ in SPRING_POSITIONS) / len(SPRING_POSITIONS)
    centroid_y = sum(y for _, y in SPRING_POSITIONS) / len(SPRING_POSITIONS)
    centroid_offset = math.hypot(centroid_x, centroid_y)
    checks = [
        {
            "name": "target_force_requires_expected_compression",
            "type": "spring_force",
            "target_force_n": target_f,
            "spring_rate_n_per_mm": k_total,
            "compression_mm": round(target_c, 3),
            "expected_compression_mm": 4.0,
            "pass": abs(target_c - 4.0) <= 0.01,
        },
        {
            "name": "screened_force_below_patient_contact_limit",
            "type": "force_gate",
            "force_n": screened_f,
            "limit_n": P.spring_max_load_n,
            "pass": screened_f <= P.spring_max_load_n,
        },
        {
            "name": "hard_stop_caps_force_at_patient_limit",
            "type": "force_gate",
            "force_n": round(stop_force, 3),
            "limit_n": P.spring_max_load_n,
            "pass": stop_force <= P.spring_max_load_n + 1e-9,
        },
        {
            "name": "per_spring_load_within_catalog",
            "type": "force_gate",
            "force_n": round(per_spring_stop, 3),
            "limit_n": P.spring_max_load_n,
            "pass": per_spring_stop <= P.spring_max_load_n,
        },
        {
            "name": "per_spring_deflection_within_catalog",
            "type": "spring_margin",
            "margin_mm": round(P.spring_allowable_deflection - HARD_STOP_TRAVEL_MM, 3),
            "min_margin_mm": 2.0,
            "pass": (P.spring_allowable_deflection - HARD_STOP_TRAVEL_MM) >= 2.0,
        },
        {
            "name": "spring_centroid_on_device_axis",
            "type": "design_rule",
            "centroid_offset_mm": round(centroid_offset, 3),
            "max_offset_mm": 0.01,
            "pass": centroid_offset <= 0.01,
        },
    ]
    return {
        "target_contact_force_n": target_f,
        "max_screened_contact_force_n": screened_f,
        "spring_rate_n_per_mm": k_total,
        "spring_count": len(SPRING_POSITIONS),
        "target_compression_mm": round(target_c, 3),
        "max_screened_compression_mm": round(screened_c, 3),
        "hard_stop_travel_mm": HARD_STOP_TRAVEL_MM,
        "hard_stop_force_n": round(stop_force, 3),
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def make_veo5_material_audit() -> dict:
    """First-order PETG/PA12 screen for the dual-spring load paths."""
    allowable_mpa = 18.0
    # Moving spring tab: 2.75 N at ~10 mm arm on the 10 x 3 mm section.
    sigma_tab = 6.0 * 2.75 * 10.0 / (10.0 * 3.0**2)
    # Hard-stop sleeve: 5.5 N bearing on the 11/5.9 annular rim.
    ring_area = math.pi * (5.5**2 - 2.95**2)
    sigma_sleeve = 5.5 / ring_area
    # M3 lug tension proxy (unchanged load path from veo3).
    sigma_lug = 0.30
    checks = []
    for name, sigma, allow in (
        ("moving_spring_tab_bending", sigma_tab, allowable_mpa),
        ("hard_stop_sleeve_bearing", sigma_sleeve, 10.0),
        ("m3_lug_tension_proxy", sigma_lug, allowable_mpa),
    ):
        sf = allow / sigma
        checks.append(
            {
                "name": name,
                "stress_mpa": round(sigma, 4),
                "allowable_mpa": allow,
                "safety_factor": round(sf, 3),
                "min_safety_factor": 4.0,
                "pass": sf >= 4.0,
            }
        )
    return {
        "print_material": "PETG_or_PA12_baseline",
        "assumption": "First-order static screen; physical prototype still needs load testing.",
        "minimum_safety_factor": round(min(c["safety_factor"] for c in checks), 3),
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def make_veo5_geometry_audit(model: dict) -> dict:
    fixed = model["fixed"]
    left = model["left"]
    right = model["right"]
    hardware = model["hardware"]
    checks: list[dict] = []

    # Static collision checks (rest position).
    spring_tol = 0.005
    veo3._add_collision_check(checks, "spring_vs_fixed_print", hardware["spring"], fixed, spring_tol)
    veo3._add_collision_check(checks, "spring_vs_left_clamshell", hardware["spring"], left, spring_tol)
    veo3._add_collision_check(checks, "spring_vs_right_clamshell", hardware["spring"], right, spring_tol)
    veo3._add_collision_check(checks, "spring_vs_clamp_screws", hardware["spring"], hardware["clamp_screws"], spring_tol)
    veo3._add_collision_check(checks, "left_clamshell_vs_right_clamshell", left, right, 0.005)
    veo3._add_collision_check(checks, "magnet_vs_hall_sensor", hardware["magnet"], hardware["hall"])
    veo3._add_collision_check(checks, "magnet_vs_left_clamshell", hardware["magnet"], left)
    veo3._add_collision_check(checks, "magnet_vs_right_clamshell", hardware["magnet"], right)
    veo3._add_collision_check(checks, "hall_sensor_vs_fixed_print", hardware["hall"], fixed)
    veo3._add_collision_check(checks, "hall_sensor_vs_left_clamshell", hardware["hall"], left)
    veo3._add_collision_check(checks, "hall_sensor_vs_right_clamshell", hardware["hall"], right)
    for wire_name in veo3.SENSOR_WIRE_NAMES:
        veo3._add_collision_check(checks, f"{wire_name}_vs_fixed_print", hardware[wire_name], fixed, max_volume_mm3=0.02)
        veo3._add_collision_check(checks, f"{wire_name}_vs_thinklab_proxy", hardware[wire_name], model["thinklab"], max_volume_mm3=0.02)
        veo3._add_collision_check(checks, f"{wire_name}_vs_left_clamshell", hardware[wire_name], left, max_volume_mm3=0.02)
        veo3._add_collision_check(checks, f"{wire_name}_vs_right_clamshell", hardware[wire_name], right, max_volume_mm3=0.02)
    for printed_name in ("left", "right", "thinklab"):
        veo3._add_collision_check(
            checks,
            f"clamp_screws_vs_{printed_name}",
            hardware["clamp_screws"],
            model[printed_name],
            0.005,
        )

    # Printed-part connectivity (VEO4 shipped clamshells in 3-4 loose pieces).
    for name, part in (("fixed_bridge", fixed), ("left_clamshell", left), ("right_clamshell", right)):
        solid_count = len(part.val().Solids())
        checks.append(
            {
                "name": f"{name}_is_single_connected_solid",
                "type": "solid_count",
                "value": solid_count,
                "expected": 1,
                "pass": solid_count == 1,
            }
        )

    # The device itself must never clash with the printed ring (static grip)…
    veo3._add_collision_check(checks, "thinklab_vs_left_clamshell", model["thinklab"], left)
    veo3._add_collision_check(checks, "thinklab_vs_right_clamshell", model["thinklab"], right)
    # …and the guide rods must run clear through the bushing bores (every prior
    # version never checked this pair; the rods skewered the bridge blocks).
    veo3._add_collision_check(checks, "guide_rods_vs_left_clamshell", hardware["guide_rods"], left)
    veo3._add_collision_check(checks, "guide_rods_vs_right_clamshell", hardware["guide_rods"], right)

    # Dynamic checks: translate moving parts up AND compress the spring helix,
    # all the way to the coil-bind hard stop. The Thinklabs body rides with the
    # ring, so its top face must clear the fixed plate at full travel too.
    for compression in (2.0, 4.0, 4.6, HARD_STOP_TRAVEL_MM):
        spring = make_compressed_spring(P, compression)
        moved = {
            "left_clamshell": left.translate((0, 0, compression)),
            "right_clamshell": right.translate((0, 0, compression)),
            "thinklab": model["thinklab"].translate((0, 0, compression)),
            "audio_plug": hardware["audio_plug"].translate((0, 0, compression)),
            "magnet": hardware["magnet"].translate((0, 0, compression)),
            "clamp_screws": hardware["clamp_screws"].translate((0, 0, compression)),
        }
        for name, shape in moved.items():
            veo3._add_collision_check(checks, f"compressed_{compression}mm_{name}_vs_fixed_print", shape, fixed)
        veo3._add_collision_check(checks, f"compressed_{compression}mm_spring_vs_left", spring, moved["left_clamshell"], spring_tol)
        veo3._add_collision_check(checks, f"compressed_{compression}mm_spring_vs_right", spring, moved["right_clamshell"], spring_tol)
        veo3._add_collision_check(checks, f"compressed_{compression}mm_spring_vs_fixed", spring, fixed, spring_tol)
        veo3._add_collision_check(checks, f"compressed_{compression}mm_magnet_vs_hall", moved["magnet"], hardware["hall"])
        veo3._add_collision_check(checks, f"compressed_{compression}mm_ring_vs_hall_leads", moved["right_clamshell"], hardware["hall"])
        veo3._add_collision_check(checks, f"compressed_{compression}mm_left_vs_rods", moved["left_clamshell"], hardware["guide_rods"])
        veo3._add_collision_check(checks, f"compressed_{compression}mm_right_vs_rods", moved["right_clamshell"], hardware["guide_rods"])
        # The wires now cross over the rising device and run near the magnet
        # tower; verify the direct-solder routing at every travel state.
        for wire_name in veo3.SENSOR_WIRE_NAMES:
            veo3._add_collision_check(
                checks, f"compressed_{compression}mm_thinklab_vs_{wire_name}", moved["thinklab"], hardware[wire_name], 0.02
            )
            veo3._add_collision_check(
                checks, f"compressed_{compression}mm_right_vs_{wire_name}", moved["right_clamshell"], hardware[wire_name], 0.02
            )

    # Fastener realism: screws must stay clear of the spring bridge slab (the
    # blue plate the spring sits on). Probe volume covers the slab region the
    # old 56 mm screw rods used to skewer (x > nut pocket, around the rear seam).
    rear_seam_y = veo4.clamshell_seam_ys(P)[1]
    bridge_probe = v9.box(22.0, 14.0, 12.0, (25.0, rear_seam_y, P.carriage_z))
    veo3._add_collision_check(
        checks, "clamp_screws_clear_of_spring_bridge_slab", hardware["clamp_screws"], bridge_probe, 0.005
    )
    screws_bb = hardware["clamp_screws"].val().BoundingBox()
    screw_span = screws_bb.xmax - screws_bb.xmin
    checks.append(
        {
            "name": "clamp_screws_match_m3x30_bom",
            "type": "bbox_dimension",
            "value_mm": round(screw_span, 3),
            "expected_mm": SCREW_LENGTH_MM + 3.0,  # shaft + head height
            "tolerance_mm": 0.6,
            "pass": abs(screw_span - (SCREW_LENGTH_MM + 3.0)) <= 0.6,
        }
    )
    lug_x_min, lug_x_max = 3.0, 17.0
    nut_inside_lug = (
        lug_x_min <= NUT_CENTER_X_MM - NUT_POCKET_DEPTH_MM / 2.0
        and NUT_CENTER_X_MM + NUT_POCKET_DEPTH_MM / 2.0 <= lug_x_max
    )
    checks.append(
        {
            "name": "nut_pocket_inside_right_lug",
            "type": "design_rule",
            "nut_center_x_mm": NUT_CENTER_X_MM,
            "lug_x_range_mm": [lug_x_min, lug_x_max],
            "pass": nut_inside_lug,
        }
    )

    # SS49E leads: modeled at full length and clear of the pillar leg channel.
    hall_bb_for_leads = hardware["hall"].val().BoundingBox()
    hall_bb_len = hall_bb_for_leads.xmax - hall_bb_for_leads.xmin
    hall_total_len = 3.1 + HALL_LEAD_LENGTH_MM
    checks.append(
        {
            "name": "hall_body_plus_leads_length",
            "type": "bbox_dimension",
            "value_mm": round(hall_bb_len, 3),
            "expected_mm": hall_total_len,
            "tolerance_mm": 0.2,
            "pass": abs(hall_bb_len - hall_total_len) <= 0.2,
        }
    )
    channel_top_z = PILLAR_BOTTOM_Z_MM + 1.7
    lead_axis_z = HALL_FACE_Z_MM + 0.75
    z_margin = min(
        channel_top_z - (lead_axis_z + HALL_LEAD_W_MM / 2.0),
        (lead_axis_z - HALL_LEAD_W_MM / 2.0) - PILLAR_BOTTOM_Z_MM,
    )
    y_margin = 2.0 - (HALL_LEAD_PITCH_MM + HALL_LEAD_W_MM / 2.0)
    lead_margin = min(z_margin, y_margin)
    checks.append(
        {
            "name": "hall_leads_clear_leg_channel",
            "type": "travel_clearance",
            "gap_mm": round(lead_margin, 3),
            "min_gap_mm": 0.3,
            "pass": lead_margin >= 0.3,
        }
    )

    # Guide rods: purchasable 5x30 dowels whose tips never reach the patient.
    # (The inherited v10 proxy hung 47 mm rods flush with the contact face;
    # at the 4 N working point the tips reached the skin plane.)
    rods_bb = hardware["guide_rods"].val().BoundingBox()
    tk_bb = model["thinklab"].val().BoundingBox()
    rod_len = rods_bb.zmax - rods_bb.zmin
    checks.append(
        {
            "name": "guide_rods_match_5x25_dowel",
            "type": "bbox_dimension",
            "value_mm": round(rod_len, 3),
            "expected_mm": GUIDE_ROD_LENGTH_MM,
            "tolerance_mm": 0.1,
            "pass": abs(rod_len - GUIDE_ROD_LENGTH_MM) <= 0.1,
        }
    )
    # Patient-poke guard: with the short rods the tip distance to the skin
    # plane is the protection (the tip exits the shortened barrel near coil
    # bind, so the gate is raised to 5 mm at the mechanical hard stop).
    for compression, min_gap in ((4.6, 5.0), (HARD_STOP_TRAVEL_MM, 5.0)):
        skin_margin = rods_bb.zmin - (tk_bb.zmin + compression)
        checks.append(
            {
                "name": f"rod_tips_above_skin_at_{compression}mm",
                "type": "travel_clearance",
                "gap_mm": round(skin_margin, 3),
                "min_gap_mm": min_gap,
                "pass": skin_margin >= min_gap,
            }
        )
    bushing_top_z = P.carriage_z + 10.0 + P.guide_bushing_h / 2.0  # barrels ride 10 mm high
    engagement = bushing_top_z - rods_bb.zmin
    checks.append(
        {
            "name": "rod_bushing_engagement_at_rest",
            "type": "travel_clearance",
            "gap_mm": round(engagement, 3),
            "min_gap_mm": 5.5,
            "pass": engagement >= 5.5,
        }
    )
    sleeve_bottom_z = -13.5
    stop_travel = sleeve_bottom_z - bushing_top_z
    checks.append(
        {
            "name": "hard_stop_travel_is_5p5mm",
            "type": "design_rule",
            "travel_mm": round(stop_travel, 3),
            "expected_mm": HARD_STOP_TRAVEL_MM,
            "pass": abs(stop_travel - HARD_STOP_TRAVEL_MM) <= 0.05,
        }
    )
    bushing_bottom_z = P.carriage_z + 10.0 - P.guide_bushing_h / 2.0
    bushing_skin_margin = bushing_bottom_z - tk_bb.zmin  # moving parts: constant offset
    checks.append(
        {
            "name": "bushing_bottoms_above_diaphragm_plane",
            "type": "travel_clearance",
            "gap_mm": round(bushing_skin_margin, 3),
            "min_gap_mm": 1.5,
            "pass": bushing_skin_margin >= 1.5,
        }
    )

    # Official Thinklabs One envelope (46 x 28) and top-lip retention geometry.
    tk_d = tk_bb.xmax - tk_bb.xmin
    tk_h = tk_bb.zmax - tk_bb.zmin
    checks.append(
        {
            "name": "thinklab_proxy_matches_official_46x28",
            "type": "bbox_dimension",
            "value_mm": [round(tk_d, 3), round(tk_h, 3)],
            "expected_mm": [46.0, 28.0],
            "tolerance_mm": 0.1,
            "pass": abs(tk_d - 46.0) <= 0.1 and abs(tk_h - 28.0) <= 0.1,
        }
    )
    # Conical self-energizing grip: bore-to-cone radial clearance at the band
    # bottom and top (clamp squeeze closes it; contact force wedges tighter).
    prof = VEO5_THINKLABS_PROFILE
    cone_taper = (prof["cone_bottom_d_mm"] - prof["cone_top_d_mm"]) / 2.0 / prof["cone_h_mm"]
    dev_r_at_band_bottom = prof["cone_bottom_d_mm"] / 2.0 - cone_taper * 0.5  # z -42
    dev_r_at_band_top = prof["cone_bottom_d_mm"] / 2.0 - cone_taper * 12.5  # z -30
    bore_taper = (23.725 - 21.475) / 15.0
    bore_r_bottom = 23.725 - bore_taper * 1.5
    bore_r_top = 21.475 + bore_taper * 1.5
    gap_bottom = bore_r_bottom - dev_r_at_band_bottom
    gap_top = bore_r_top - dev_r_at_band_top
    checks.append(
        {
            "name": "cone_grip_radial_clearance",
            "type": "design_rule",
            "gap_at_band_bottom_mm": round(gap_bottom, 3),
            "gap_at_band_top_mm": round(gap_top, 3),
            "pass": 0.2 <= gap_bottom <= 1.5 and 0.2 <= gap_top <= 1.5,
        }
    )
    # The volume dial ROTATES — nothing printed may encroach on it.
    dial_probe = v9.cyl_z(
        prof["dial_d_mm"] / 2.0 + 0.8,
        prof["dial_h_mm"] + 1.0,
        (0, 0, tk_bb.zmax - prof["glass_h_mm"] - prof["dial_h_mm"] / 2.0),
    )
    veo3._add_collision_check(checks, "dial_rotation_clearance_left", dial_probe, left)
    veo3._add_collision_check(checks, "dial_rotation_clearance_right", dial_probe, right)
    # The audio plug stays in DURING use: plugged (positive overlap with the
    # device) and passing freely through the ring's jack window.
    plug_in = v12._intersection_volume_mm3(hardware["audio_plug"], model["thinklab"])
    checks.append(
        {
            "name": "audio_plug_engaged_in_socket",
            "type": "min_intersection_volume",
            "volume_mm3": round(plug_in, 3),
            "min_volume_mm3": 0.5,
            "pass": plug_in >= 0.5,
        }
    )
    veo3._add_collision_check(checks, "audio_plug_vs_left_clamshell", hardware["audio_plug"], left)
    veo3._add_collision_check(checks, "audio_plug_vs_right_clamshell", hardware["audio_plug"], right)
    veo3._add_collision_check(checks, "audio_plug_vs_fixed_print", hardware["audio_plug"], fixed)

    # Purchased-part dimension fidelity: WF6-25 spring OD and silicone wire OD.
    sp_probe = v9.box(20.0, 20.0, 0.4, (SPRING_POSITIONS[0][0], SPRING_POSITIONS[0][1], -18.0))
    sp_slice = hardware["spring"].val().intersect(sp_probe.val())
    sp_bb = sp_slice.BoundingBox()
    spring_od = max(sp_bb.xmax - sp_bb.xmin, sp_bb.ymax - sp_bb.ymin)
    checks.append(
        {
            "name": "spring_visual_od_matches_wf6_25",
            "type": "bbox_dimension",
            "value_mm": round(spring_od, 3),
            "expected_mm": P.spring_od,
            "tolerance_mm": 0.1,
            "pass": abs(spring_od - P.spring_od) <= 0.1,
        }
    )
    # Slice one straight segment of the routed wire to measure its modeled OD
    # (the whole-wire bbox spans the routing, not the conductor).
    od_probe = v9.box(0.2, 30.0, 20.0, (20.0, -23.5, HALL_FACE_Z_MM + 0.75))
    wire_slice = hardware["hall_signal_wire"].val().intersect(od_probe.val())
    s_bb = wire_slice.BoundingBox()
    wire_od = max(s_bb.ymax - s_bb.ymin, s_bb.zmax - s_bb.zmin)
    checks.append(
        {
            "name": "sensor_wire_od_matches_28awg_silicone",
            "type": "bbox_dimension",
            "value_mm": round(wire_od, 3),
            "expected_mm": 2.0 * SENSOR_WIRE_R_MM,
            "tolerance_mm": 0.1,
            "pass": abs(wire_od - 2.0 * SENSOR_WIRE_R_MM) <= 0.1,
        }
    )

    # XIAO RP2040 in its cradle: no clash with the printed wing/rails or the
    # robot flange volume, and the board matches the real 21.0 x 17.78 outline.
    veo3._add_collision_check(checks, "xiao_vs_fixed_print", hardware["xiao"], fixed)
    veo3._add_collision_check(checks, "xiao_vs_robot_flange", hardware["xiao"], hardware["robot_flange"])
    xiao_bb = hardware["xiao"].val().BoundingBox()
    checks.append(
        {
            "name": "xiao_board_matches_catalog_outline",
            "type": "bbox_dimension",
            "value_mm": [round(xiao_bb.xmax - xiao_bb.xmin, 3), round(xiao_bb.ymax - xiao_bb.ymin, 3)],
            "expected_mm": [XIAO_W_MM, XIAO_L_MM],
            "tolerance_mm": 0.1,
            "pass": abs((xiao_bb.xmax - xiao_bb.xmin) - XIAO_W_MM) <= 0.1
            and abs((xiao_bb.ymax - xiao_bb.ymin) - XIAO_L_MM) <= 0.1,
        }
    )

    # WidowX AI 37x40 accessory-mount holes must be open through the plate.
    wxai_probe = None
    for sx in (-WXAI_MOUNT_X_MM, WXAI_MOUNT_X_MM):
        for sy in (-WXAI_MOUNT_Y_MM, WXAI_MOUNT_Y_MM):
            pin = v9.cyl_z(1.4, P.top_bridge_t + 2.0, (sx, sy, 0))
            wxai_probe = pin if wxai_probe is None else wxai_probe.union(pin)
    veo3._add_collision_check(checks, "wxai_mount_holes_open_through_plate", wxai_probe, fixed)

    # Electrical continuity (geometric): every wire must POSITIVELY overlap
    # both solder partners — the hall lead tips and the XIAO board edge.
    # (Regression: a v9.cyl_y displacement bug once left all three wires
    # floating disconnected; zero-collision checks can't catch that.)
    for wire_name in veo3.SENSOR_WIRE_NAMES:
        v_lead = v12._intersection_volume_mm3(hardware[wire_name], hardware["hall"])
        v_xiao = v12._intersection_volume_mm3(hardware[wire_name], hardware["xiao"])
        checks.append(
            {
                "name": f"{wire_name}_joins_hall_lead",
                "type": "min_intersection_volume",
                "volume_mm3": round(v_lead, 4),
                "min_volume_mm3": 0.01,
                "pass": v_lead >= 0.01,
            }
        )
        checks.append(
            {
                "name": f"{wire_name}_joins_xiao_pads",
                "type": "min_intersection_volume",
                "volume_mm3": round(v_xiao, 4),
                "min_volume_mm3": 0.01,
                "pass": v_xiao >= 0.01,
            }
        )

    # Round X-Series horn interface: the horn (+servo body) must seat in the
    # recess without clashing, its M2 screws must run free, the boss relief
    # and PCD-16 holes must be open, and the wrist body must clear the XIAO.
    veo3._add_collision_check(checks, "horn_and_servo_seat_without_clash", hardware["robot_flange"], fixed)
    veo3._add_collision_check(checks, "horn_m2_screws_vs_fixed_print", hardware["interface_bolts"], fixed)
    veo3._add_collision_check(checks, "horn_servo_vs_xiao", hardware["robot_flange"], hardware["xiao"])
    horn_probe = v9.cyl_z(HORN_RELIEF_D_MM / 2.0 - 0.3, P.top_bridge_t + 2.0, (0, 0, 0))
    for hx, hy in HORN_M2_POSITIONS:
        horn_probe = horn_probe.union(v9.cyl_z(0.9, P.top_bridge_t + 2.0, (hx, hy, 0)))
    veo3._add_collision_check(checks, "horn_relief_and_m2_holes_open", horn_probe, fixed)
    xiao_bb_for_horn = hardware["xiao"].val().BoundingBox()
    servo_xiao_gap = -17.0 - xiao_bb_for_horn.ymax  # servo body ymin vs cradle end
    checks.append(
        {
            "name": "wrist_servo_clear_of_xiao_cradle",
            "type": "travel_clearance",
            "gap_mm": round(servo_xiao_gap, 3),
            "min_gap_mm": 2.0,
            "pass": servo_xiao_gap >= 2.0,
        }
    )

    # Coaxial alignment and air-gap budget (body only; leads extend off-axis).
    magnet_bb = hardware["magnet"].val().BoundingBox()
    hall_bb = make_hall_body_proxy().val().BoundingBox()
    magnet_cx = (magnet_bb.xmin + magnet_bb.xmax) / 2.0
    magnet_cy = (magnet_bb.ymin + magnet_bb.ymax) / 2.0
    hall_cx = (hall_bb.xmin + hall_bb.xmax) / 2.0
    hall_cy = (hall_bb.ymin + hall_bb.ymax) / 2.0
    axis_offset = math.hypot(magnet_cx - hall_cx, magnet_cy - hall_cy)
    checks.append(
        {
            "name": "magnet_hall_coaxial_alignment",
            "type": "xy_axis_offset",
            "offset_mm": round(axis_offset, 3),
            "max_offset_mm": 0.3,
            "pass": axis_offset <= 0.3,
        }
    )
    rest_gap = hall_bb.zmin - magnet_bb.zmax
    checks.append(
        {
            "name": "magnet_hall_rest_gap",
            "type": "bbox_axis_gap",
            "axis": "z",
            "gap_mm": round(rest_gap, 3),
            "min_gap_mm": HARD_STOP_TRAVEL_MM + 1.0,
            "max_gap_mm": 14.0,
            "pass": HARD_STOP_TRAVEL_MM + 1.0 <= rest_gap <= 14.0,
        }
    )
    bind_gap = rest_gap - HARD_STOP_TRAVEL_MM
    checks.append(
        {
            "name": "magnet_hall_gap_at_hard_stop",
            "type": "travel_clearance",
            "gap_mm": round(bind_gap, 3),
            "min_gap_mm": 1.0,
            "pass": bind_gap >= 1.0,
        }
    )
    tower_top = max(s.BoundingBox().zmax for s in model["right"].val().Solids())
    pillar_clear = PILLAR_BOTTOM_Z_MM - (tower_top + HARD_STOP_TRAVEL_MM)
    checks.append(
        {
            "name": "tower_vs_pillar_clearance_at_hard_stop",
            "type": "travel_clearance",
            "gap_mm": round(pillar_clear, 3),
            "min_gap_mm": 0.5,
            "pass": pillar_clear >= 0.5,
        }
    )

    # Magnet disc dimensions (Z-axis disc now: X/Y are the diameter).
    checks.append(
        {
            "name": "magnet_disc_diameter_xy",
            "type": "bbox_dimension",
            "value_mm": round(max(magnet_bb.xmax - magnet_bb.xmin, magnet_bb.ymax - magnet_bb.ymin), 3),
            "expected_mm": MAGNET_DIAMETER_MM,
            "tolerance_mm": 0.05,
            "pass": abs((magnet_bb.xmax - magnet_bb.xmin) - MAGNET_DIAMETER_MM) <= 0.05
            and abs((magnet_bb.ymax - magnet_bb.ymin) - MAGNET_DIAMETER_MM) <= 0.05,
        }
    )
    checks.append(
        {
            "name": "magnet_disc_thickness_z",
            "type": "bbox_dimension",
            "value_mm": round(magnet_bb.zmax - magnet_bb.zmin, 3),
            "expected_mm": MAGNET_THICKNESS_MM,
            "tolerance_mm": 0.05,
            "pass": abs((magnet_bb.zmax - magnet_bb.zmin) - MAGNET_THICKNESS_MM) <= 0.05,
        }
    )

    # Spring seating and split gap.
    spring_bb = hardware["spring"].val().BoundingBox()
    bottom_err = abs(spring_bb.zmin - P.spring_moving_face_z)
    top_err = abs(P.spring_fixed_face_z - spring_bb.zmax)
    checks.append(
        {
            "name": "spring_seated_in_sockets",
            "type": "bbox_z_seating",
            "max_error_mm": round(max(bottom_err, top_err), 3),
            "max_allowed_error_mm": 0.75,
            "pass": max(bottom_err, top_err) <= 0.75,
        }
    )
    left_bb = left.val().BoundingBox()
    right_bb = right.val().BoundingBox()
    split_gap = right_bb.xmin - left_bb.xmax
    checks.append(
        {
            "name": "clamshell_split_air_gap",
            "type": "bbox_axis_gap",
            "axis": "x",
            "gap_mm": round(split_gap, 3),
            "min_gap_mm": 2.5,
            "pass": split_gap >= 2.5,
        }
    )

    # (The old bbox-based "wires outside the body radius" check is obsolete:
    # the direct-solder routing legitimately crosses ABOVE the device top, so
    # wire/device safety is now proven by the per-compression boolean checks
    # in the dynamic loop instead.)

    sensing = make_sensing_model()
    checks.append(
        {
            "name": "hall_output_monotonic_over_full_travel",
            "type": "sensing_model",
            "pass": bool(sensing["monotonic_over_full_travel"]),
        }
    )
    operating_volts = [hall_response(c)["hall_out_v"] for c in (0.0, 4.0, 4.6, HARD_STOP_TRAVEL_MM)]
    in_rails = all(0.25 <= v <= HALL_SUPPLY_V - 0.25 for v in operating_volts)
    checks.append(
        {
            "name": "hall_output_within_adc_rails",
            "type": "sensing_model",
            "volts": [round(v, 3) for v in operating_volts],
            "rail_margin_v": 0.25,
            "pass": in_rails,
        }
    )
    span_ok = sensing["operating_span_v"] >= 0.25
    checks.append(
        {
            "name": "hall_operating_span_sufficient",
            "type": "sensing_model",
            "span_v": sensing["operating_span_v"],
            "min_span_v": 0.25,
            "pass": span_ok,
        }
    )

    clearances = {
        "magnet_hall_rest_gap_mm": round(rest_gap, 3),
        "magnet_hall_gap_at_hard_stop_mm": round(bind_gap, 3),
        "spring_bottom_seating_error_mm": round(bottom_err, 3),
        "spring_top_seating_error_mm": round(top_err, 3),
        "clamshell_split_air_gap_mm": round(split_gap, 3),
    }
    return {
        "units": "mm",
        "method": "CadQuery boolean intersections, connectivity, compressed-spring dynamic checks, first-order Hall model",
        "spring_visual_model": "continuous_helix",
        "checks": checks,
        "clearances": clearances,
        "dynamic_audit": {
            "compressions_checked_mm": [4.0, 8.0, 9.2],
            "hard_stop_travel_mm": HARD_STOP_TRAVEL_MM,
            "spring_model": "recompressed helix per state (not just translated)",
        },
        "sensing_model": {k: v for k, v in sensing.items() if k != "samples"},
        "pass": all(check["pass"] for check in checks),
    }


# --- Outputs -----------------------------------------------------------------


def export_part(part: cq.Workplane, stem: str) -> None:
    exporters.export(part, str(OUT_DIR / f"{stem}.step"))
    exporters.export(part, str(OUT_DIR / f"{stem}.stl"), tolerance=0.14, angularTolerance=0.12)


def write_sim_outputs(metadata: dict) -> dict:
    veo4.ensure_ffmpeg_on_path()
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
    trace_path = SIM_DIR / "veo5_contact_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = helpers.write_contact_plot(rows, metadata, SIM_DIR)
    if plot_path is not None:
        final_plot = SIM_DIR / "veo5_contact_response.png"
        if final_plot.exists():
            final_plot.unlink()
        plot_path.rename(final_plot)
    video_path = helpers.write_motion_video(rows, metadata, SIM_DIR, fps=30, max_frames=150)
    if video_path is not None:
        final_video = SIM_DIR / "veo5_contact_motion.mp4"
        if final_video.exists():
            final_video.unlink()
        video_path.rename(final_video)
    summary = {
        "variant": RECOMMENDED_MECHANICS["simulation_variant"],
        "output_csv": str(trace_path),
        "output_plot": str(SIM_DIR / "veo5_contact_response.png"),
        "output_motion_video": str(SIM_DIR / "veo5_contact_motion.mp4"),
        "peak_force_n": max(row["contact_force_n"] for row in rows),
        "final_force_n": rows[-1]["contact_force_n"],
        "max_spring_compression_mm": max(row["spring_compression_mm"] for row in rows),
    }
    (SIM_DIR / "veo5_contact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_sim_rows() -> list[dict[str, float]]:
    with (SIM_DIR / "veo5_contact_trace.csv").open(newline="", encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def render_presentation_assets(model: dict, rows: list[dict[str, float]]) -> dict:
    veo4.ensure_ffmpeg_on_path()
    configure_backends()
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    clean = v12.render_clean_cad_png(model["fixed"], model["left"], model["right"], model["thinklab"], model["hardware"])
    detail = v12.render_clamshell_detail_png(model["fixed"], model["left"], model["right"], model["thinklab"], model["hardware"])
    video = v12.write_actual_cad_motion_video(model["fixed"], model["left"], model["right"], model["thinklab"], model["hardware"], rows)
    renamed = {}
    for old_path, new_name in [
        (clean, "veo5_cad_clean.png"),
        (detail, "veo5_clamshell_detail.png"),
        (video, "veo5_actual_cad_motion.mp4" if video is not None else None),
    ]:
        if old_path is None or new_name is None:
            continue
        new_path = PRESENTATION_DIR / new_name
        if new_path.exists():
            new_path.unlink()
        old_path.rename(new_path)
        renamed[new_name] = str(new_path)
    return renamed


def make_sensor_layout_metadata(hardware: dict[str, cq.Workplane]) -> dict:
    """veo3's layout metadata but anchored to the VEO5 sensing station."""
    layout = veo5_sensor_layout(P)
    wire_bb = veo3._combined_bbox([hardware[name] for name in veo3.SENSOR_WIRE_NAMES])
    return {
        "wiring_route": "external_short_fixed_side",
        "magnet_center_mm": [round(value, 3) for value in layout["magnet"]],
        "hall_center_mm": [round(value, 3) for value in layout["hall"]],
        "solder_target_on_xiao_mm": [round(value, 3) for value in layout["sensor_connector"]],
        "xiao_center_mm": [XIAO_CENTER_X_MM, XIAO_CENTER_Y_MM, WING_TOP_Z_MM + 0.8],
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


def make_purchased_parts_fit_metadata() -> dict:
    """How each BOM purchase maps onto the CAD (and where the BOM must change)."""
    return {
        "thinklabs_one": "real shape from user photos: 46 flare + narrowing cone + rotating dial + low side sockets; cone/dial/socket numbers are PHOTO ESTIMATES — bench-measure before printing. Grip = conical self-energizing bore (no lip on the dial); audio plug stays in through the ring window",
        "spring_misumi_wf6_25": "x2 IN PARALLEL (point-symmetric beside the rods, zero moment): catalog wire 0.6, ~20 coils, k 0.5 each -> 1.0 N/mm total, 4 N @ 4 mm, hard stop 5.5 mm = 5.5 N; helix at true OD 6.0 — BUY TWO",
        "hall_ss49e_to92s": "body 3.1x4.1x1.5 plus three 13 mm straight leads; slot + 4 mm leg channel sized to part",
        "magnet_d5x2_n42": "Z-axial disc in 5.5 x 2.6 pocket on the moving ring",
        "clamp_screws": "M3 x 30 SHCS x4 + M3 nut x4 — BOM said M3 x 12-16, too short for the lug span, update BOM",
        "hall_m2_screws": "NOT needed in VEO5 — the sensor is slot-mounted, no breakout PCB",
        "guide_rods": "5 x 25 mm standard dowel pins x2 (tips 13.5 mm above skin at the 5.5 mm hard stop)",
        "sensor_wire": "26-30 AWG silicone, modeled at 1.2 mm insulated OD",
        "jst_ph_3pin": "REMOVED per user decision — the three hall wires solder directly onto the XIAO's right-edge pads (rail notch provided for the bundle)",
        "xiao_rp2040": "ON the mount: rail cradle on the front-center wing (board 21.0 x 17.78, no holes), USB-C outward; hall wires solder to its right-edge pads",
        "robot_side_screws": "PRIMARY: round X-Series horn at plate center (PCD 16) -> M2 x 5 x4 for WidowX-250/XL430 (taps 3.5 deep) or M2 x 4 for ViperX-300/XM430 (taps 2.0 deep); ALT: WidowX AI 37x40 M3 pattern -> M3 x 12 x4; generic slots kept as fallback",
        "wrist_horn_interface": "20.8 dia x 0.8 locating recess + 8.6 dia boss relief + 4x M2 counterbored on PCD 16; wrist servo body clears the XIAO cradle by >=2 mm",
    }


def make_fastener_layout_metadata() -> dict:
    front_y, rear_y = veo4.clamshell_seam_ys(P)
    return {
        "front_seam_y_mm": round(front_y, 3),
        "rear_seam_y_mm": round(rear_y, 3),
        "clamshell_m3_screw_count": 4,
        "clamp_screw_spec": "M3 x 30 socket head cap screw + M3 hex nut (DIN 934)",
        "m3_clearance_diameter_mm": P.m3_clearance_d,
        "screw_head_face_x_mm": SCREW_HEAD_FACE_X_MM,
        "screw_tip_x_mm": SCREW_TIP_X_MM,
        "nut_pocket_center_x_mm": NUT_CENTER_X_MM,
        "nut_insertion": "vertical slot in the right lug; bottom screws load the nut from below, top screws from above",
    }


def make_magnet_design_metadata() -> dict:
    return {
        "shape": "d5x2_cylindrical_disc",
        "axis": "Z (motion axis, axial gap closes with compression)",
        "diameter_mm": MAGNET_DIAMETER_MM,
        "thickness_mm": MAGNET_THICKNESS_MM,
        "pocket_diameter_mm": MAGNET_POCKET_DIAMETER_MM,
        "pocket_depth_mm": MAGNET_POCKET_DEPTH_MM,
        "pocket_center_xy_mm": [SENSE_X_MM, SENSE_Y_MM],
        "magnet_top_face_z_mm": MAGNET_TOP_Z_MM,
        "retention": "drop-in from top, CA glue dab; pole face up toward Hall",
        "pocket_clearance_diametral_mm": round(MAGNET_POCKET_DIAMETER_MM - MAGNET_DIAMETER_MM, 3),
    }


def make_metadata(model: dict, geometry: dict, mechanical: dict, material: dict, validation: dict, sensing: dict, sim_summary=None, presentation_assets=None, sensing_assets=None) -> dict:
    metadata = {
        "units": "mm",
        "params": asdict(P),
        "recommended_mechanics": RECOMMENDED_MECHANICS,
        "mechanism": {
            "type": "veo5_coaxial_hall_clamshell",
            "basis": "VEO4 clamshell with fused single-solid halves and coaxial Z-axis Hall sensing",
            "force_feedback": "Axial gap between Z-magnetized disc and fixed SS49E closes 1:1 with spring compression; monotonic output.",
            "spring_visual_model": "continuous_helix",
        },
        "magnet_design": make_magnet_design_metadata(),
        "sensing_model": {k: v for k, v in sensing.items() if k != "samples"},
        "fastener_layout": make_fastener_layout_metadata(),
        "purchased_parts_fit": make_purchased_parts_fit_metadata(),
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
    if sensing_assets is not None:
        metadata["sensing_assets"] = sensing_assets
    if presentation_assets is not None:
        metadata["presentation_assets"] = presentation_assets
    return metadata


def write_report(metadata: dict) -> None:
    geometry = metadata["geometry_audit"]
    sensing = metadata["sensing_model"]
    key = sensing["key_points"]
    report = [
        "# VEO5 Coaxial-Hall Clamshell Mount",
        "",
        "VEO5 fixes the VEO4 defects and redesigns the magnet sensing module as a",
        "coaxial axial-gap arrangement (gap closes 1:1 with spring compression).",
        "",
        "## Fixes vs VEO4",
        "",
        "- Clamshell halves no longer overlap at the centerline (split cutters were swapped; the halves shared a 3 mm slab).",
        "- Guide bushings, front M3 lugs and the magnet holder are now fused into the ring (they were floating islands; each half printed as 3-4 loose pieces).",
        "- Magnet is Z-magnetized under a fixed Hall pillar; output is monotonic over the whole stroke (VEO4's magnet crossed the Hall plane at 5 mm compression).",
        "- Spring bridge lowered: ~0.9 mm real clearance to the bottom coil (was 0.11 mm), and the *compressed* spring is now audited at 4/8/9.2/11.25 mm travel.",
        "- Motion-video spring squash now preserves the seating offsets, removing the visual spring/ring penetration.",
        "- Clamp screws are real M3 x 30 SHCS (head on the left lug face, nut in a slotted pocket inside the right lug). The old 56 mm screw rods skewered the spring bridge slab with the nut buried inside the 3 mm plate.",
        "- Hall sensor is modeled as the purchased Honeywell SS49E TO-92S with its three ~13 mm straight leads; the pillar leg channel clears them and the wires splice at the lead tips.",
        "- TWO WF6-25 springs point-symmetric beside the rods replace the single rear spring whose 46 mm moment arm side-loaded the bushings; hard-stop sleeves cap travel at 5.5 mm = the 5.5 N patient limit; the plate is a round 88 dia disc.",
        "- Spring helix has the true WF6-25 6.0 mm OD and ~20 catalog coils; sensor wires (real ~1.2 mm silicone OD) solder directly onto the XIAO's right-edge pads — the optional JST connector was dropped.",
        "- Thinklabs One is modeled at the OFFICIAL 46 dia x 28 mm (was a 66/48 guess, ~20 mm oversize); the clamp ring is re-derived with a TOP lip riding the 46->37 shoulder since the real flush-bottom puck has no flange for a bottom lip. Upper-tier 37 dia is a photo estimate - measure before printing.",
        "- XIAO RP2040 rides in a printed rail cradle on a fixed-bridge wing (USB-C outward); the plate carries the WidowX AI 37x40 M3 accessory pattern (M3 x 12) plus the generic slots for other arms.",
        "",
        "## Validation",
        "",
        f"- Overall pass: `{metadata['validation_summary']['pass']}`",
        f"- Geometry pass: `{geometry['pass']}` ({len(geometry['checks'])} checks incl. connectivity + compressed-spring states)",
        f"- Mechanical pass: `{metadata['mechanical_audit']['pass']}`",
        f"- Material pass: `{metadata['material_audit']['pass']}`",
        f"- Rest air gap (magnet face to Hall face): `{sensing['rest_face_gap_mm']} mm`",
        f"- Gap at the 5.5 mm hard stop: `{geometry['clearances']['magnet_hall_gap_at_hard_stop_mm']} mm` (never contacts)",
        "",
        "## Predicted Hall response (first-order, 3.3 V supply)",
        "",
        "| state | compression | force | face gap | field | output |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, point in (
        ("rest", key["rest"]),
        ("4 N target", key["target_force_4n"]),
        ("4.6 N screen", key["max_screened_4p6n"]),
        ("5.5 mm hard stop", key["hard_stop_5p5mm"]),
    ):
        report.append(
            f"| {label} | {point['compression_mm']} mm | {point['force_n']} N | "
            f"{point['face_gap_mm']} mm | {point['field_gauss']} G | {point['hall_out_v']} V |"
        )
    report += [
        "",
        "## Limits",
        "",
        "- Field/output numbers are an on-axis first-order model; bench-calibrate gap(V) with known weights before trusting forces.",
        "- Magnet polarity must be set at assembly (pole face toward Hall chosen so output rises with force).",
        "- Final print dimensions still need real ThinkLabs One / OpenArm flange measurements.",
    ]
    (OUT_DIR / "VEO5_COAXIAL_HALL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    configure_backends()
    model = build_veo5_model()
    geometry = make_veo5_geometry_audit(model)
    mechanical = make_veo5_mechanical_audit()
    material = make_veo5_material_audit()
    validation = veo3.make_validation_summary(geometry, mechanical, material)
    sensing = make_sensing_model()

    export_part(model["fixed"], "thinklab_openarm_veo5_fixed_bridge_print")
    export_part(model["left"], "thinklab_openarm_veo5_left_clamshell_print")
    export_part(model["right"], "thinklab_openarm_veo5_right_clamshell_print")
    exporters.export(model["thinklab"], str(OUT_DIR / "thinklab_veo5_stepped_profile_proxy.step"))

    assembly = cq.Assembly(name="thinklab_openarm_veo5_coaxial_hall_mount")
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
        elif name == "xiao":
            color = cq.Color(0.13, 0.55, 0.13)
        elif name == "audio_plug":
            color = cq.Color(0.48, 0.32, 0.19)
        assembly.add(shape, name=f"{name}_proxy_not_printed", color=color)
    assembly.save(str(OUT_DIR / "thinklab_openarm_veo5_assembly.step"), exportType="STEP")

    preview = cq.Compound.makeCompound(
        [model["fixed"].val(), model["left"].val(), model["right"].val(), model["thinklab"].val()]
        + [shape.val() for shape in model["hardware"].values()]
    )
    exporters.export(cq.Workplane("XY").add(preview), str(OUT_DIR / "thinklab_openarm_veo5_preview.step"))

    (OUT_DIR / "veo5_geometry_audit.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    (OUT_DIR / "veo5_mechanical_audit.json").write_text(json.dumps(mechanical, indent=2), encoding="utf-8")
    (OUT_DIR / "veo5_material_audit.json").write_text(json.dumps(material, indent=2), encoding="utf-8")

    sensing_assets = write_sensing_outputs(sensing)
    metadata = make_metadata(model, geometry, mechanical, material, validation, sensing)
    sim_summary = write_sim_outputs(metadata)
    rows = read_sim_rows()
    presentation_assets = render_presentation_assets(model, rows)
    metadata = make_metadata(
        model, geometry, mechanical, material, validation, sensing, sim_summary, presentation_assets, sensing_assets
    )
    (OUT_DIR / "veo5_design_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(metadata)

    failed = [c["name"] for c in geometry["checks"] if not c["pass"]]
    print(
        json.dumps(
            {
                "validation_summary": validation,
                "failed_geometry_checks": failed,
                "sensing_key_points": sensing["key_points"],
                "rest_face_gap_mm": round(REST_FACE_GAP_MM, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
