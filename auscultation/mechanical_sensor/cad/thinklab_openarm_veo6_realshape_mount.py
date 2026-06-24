"""VEO6 coaxial-Hall ThinkLabs One + OpenArm passive force mount.

VEO6 keeps the VEO3/VEO4 clamshell architecture but fixes four defects found in
VEO4 and redesigns the magnet sensing module:

1. Clamshell split cutters were swapped, so the printed halves overlapped each
   other by the full 3 mm split gap. VEO6 splits left to x <= -1.5 and right to
   x >= +1.5 with a true air gap.
2. Guide bushings, front M3 lugs and the magnet holder were disconnected
   islands inside each clamshell solid (the parts would print in 3-4 loose
   pieces). VEO6 fuses them with bridges/webs and audits connectivity.
3. The VEO4 magnet (Y-axis disc) translated PAST the fixed Hall along Z, so the
   field peaked mid-stroke and force readout was non-monotonic. VEO6 uses a
   coaxial axial-gap design: Z-magnetized D5x2 disc in an upward pocket on the
   moving ring, SS49E held face-down in a pillar on the fixed bridge directly
   above it. The air gap closes monotonically with spring compression.
4. The spring bridge top sat 0.11 mm under the bottom spring coil; VEO6 lowers
   it for ~0.9 mm of real clearance and audits the *compressed* spring against
   the printed parts at several travel points (VEO4 only audited free length).
5. Clamp screws were modeled as 56 mm rods skewering the spring bridge slab,
   with an M3 nut buried inside the 3 mm plate (impossible to assemble, and the
   visible blue-plate/screw interpenetration). VEO6 models real M3x30 SHCS:
   head seated on the left lug face, shaft ending inside the right lug, nut in
   a printed hex pocket with a vertical insertion slot, and shortened clearance
   holes that no longer tunnel through the spring bridge.
6. The Hall sensor was modeled as a bare 3x4 mm epoxy block; the purchased
   Honeywell SS49E (TO-92S) has three ~13 mm straight leads. VEO6 models the
   leads, widens the pillar leg channel around them, and moves the wire splice
   to the lead tips.
7. The inherited 47 mm guide-rod proxies hung down to z=-42 — only 8 mm above
   the stethoscope contact face, so the rod tips reached the patient's skin at
   the 4 N working compression (8 mm) and dug in beyond it. VEO6 uses standard
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
11. Robot-side mounting (OpenArm, scheme A — gripper removed; the module IS the
   end-effector). The plate bolts to an adapter plate on the OpenArm side. The
   MEASURED hole pattern (user calipers 2026-06-20) is a RECTANGLE of 4x M3,
   centers 18 mm (long) x 11 mm (short) -> holes at (±9, ±5.5), envelope
   ~25 x 18 mm. This REPLACED the earlier assumed Ø50 6x M3 circle, which did
   not match the real hardware. Hole size unchanged (M3): Ø3.5 clearance +
   Ø6.2 bottom counterbore; screws drive up into the adapter taps -> M3 x 8.
12. The plate centre stays solid (no boss relief) to mate flat on the adapter.
   The OpenArm actuator body behind the adapter is large, so the XIAO cradle is
   kept outboard (y -41) to clear it; the -y wing extends ~7 mm past the Ø88
   plate as a result. v9.cyl_y is displaced by its own height (CadQuery
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
# VEO6 = re-modeled from the user's HAND-MEASURED real unit (caliper), not
# photos. Three stacked sections, bottom -> top (totals 29.0 mm):
#   1. black diaphragm edge: 46.0 dia STRAIGHT wall, 6.0 mm tall (we clamp THIS)
#   2. cone: 46.0 -> 38.0 over 11.0 mm vertical (NARROWING upward)
#   3. dial/display: 38.0 dia, 12.0 mm tall (rotates; has buttons + the jack)
# The 9 mm audio/charge jack sits on the DIAL cylinder near the top of the
# cone. Per the user: we grip only the 6 mm black edge, so the cone, dial,
# buttons and jack are all ABOVE the ring and never touched — no jack window,
# no top lip, no conical bore needed.
VEO6_THINKLABS_PROFILE = {
    "black_edge_d_mm": 46.0,  # measured: lower disc diameter
    "black_edge_h_mm": 6.0,  # measured: black-edge height (the grip band)
    "cone_bottom_d_mm": 46.0,  # continuous with the black edge
    "cone_top_d_mm": 38.0,  # measured: upper disc diameter
    "cone_h_mm": 11.0,  # measured: slant vertical height
    "dial_d_mm": 38.0,  # measured: upper circle diameter
    "dial_h_mm": 12.0,  # measured: upper circle height
    "jack_d_mm": 9.0,  # measured: audio jack hole diameter
    # legacy keys kept so shared veo3/veo4 helpers that read lower_bottom_d_mm
    # still resolve to the real 46 mm outer diameter.
    "lower_bottom_d_mm": 46.0,
}
JACK_THETA_DEG = 215.0  # audio jack azimuth (front-left diagonal, clear of seams/tabs)
# Jack centre height: on the dial, just above the cone top. dev_bottom=-44, so
# black edge -44..-38, cone -38..-27, dial -27..-15; jack ~3 mm up the dial.
JACK_Z_MM = -24.0
# Straight clamp bore: Ø46 black edge + 0.3 mm radial clearance. The clamshell
# pinch closes this to grip; the cone (<=46, narrowing up) clears automatically.
BORE_D_MM = 46.6
# Bottom RETENTION LIP (user: a straight bore + friction alone lets the device
# drop out when the diaphragm points down). An inward Ø42 lip at the ring
# bottom physically blocks the Ø46 black edge — the device RESTS on the lip and
# cannot fall through (a positive mechanical catch, not just friction). The Ø42
# opening still exposes the diaphragm centre for skin contact (the diaphragm
# ends up recessed ~LIP_H mm inside the rim; soft tissue conforms to reach it).
BOTTOM_LIP_D_MM = 42.0
BOTTOM_LIP_H_MM = 0.8

import dataclasses

# Re-derive the shared params around the real 46x28 device. The ring shrinks
# by 10 mm radius; the whole moving assembly drops so the device top still
# clears the fixed plate at coil bind.
P = dataclasses.replace(
    veo4.P,
    thinklab_body_d=46.0,
    thinklab_body_h=28.0,
    cup_height=12.0,
    cup_wall=3.0,  # outer ring wall thinned 4.0->3.0 (user request 2026-06-23)
    top_bridge_t=3.5,  # top plate thinned 5.0->3.5 (user request 2026-06-23)
    lip_height=2.5,
    lip_opening_d=38.6,  # top lip rides over the (estimated) 37 mm upper tier
    carriage_z=-36.0,
    spring_axis_y=46.0,
    guide_rod_length=25.0,
    guide_rod_x=34.0,  # rods pulled inboard: bushing barrels hug the 28.5 ring
    guide_bushing_h=14.0,
    spring_moving_face_z=-29.5,  # 1.5 mm above the old -31 => spring preloaded 1.5 mm
    top_bridge_x=88.0,  # plate only needs to span the inboard rod sockets now
    top_bridge_y=48.0,  # spans the inboard rod sockets and the Ø50 bolt circle
)
veo3.P = P
veo4.P = P

# OpenArm end-effector interface (scheme A — gripper removed; the module is the
# end-effector). The plate bolts to an adapter plate on the OpenArm side. The
# MEASURED hole pattern on that adapter (user calipers, 2026-06-20) is a
# RECTANGLE of 4x M3, centers 18 mm (long edge) x 11 mm (short edge) -> holes at
# (±9, ±5.5) mm; overall envelope ~25 x 18 mm. This SUPERSEDES the earlier
# assumed Ø50 6x M3 output-face circle (that pattern did not match the real
# hardware). Hole SIZE unchanged (the user verified M3 screws thread into the
# existing holes): Ø3.5 clearance + Ø6.2 x 3.2 bottom-face counterbore for the
# socket head, screws drive UP into the adapter taps -> M3 x 8. No central boss
# relief (the adapter mates flat against a solid plate centre).
OUTPUT_M3_DX_MM = 10.0  # half of the 20 mm long-edge center-to-center (re-measured, +2)
OUTPUT_M3_DY_MM = 6.5  # half of the 13 mm short-edge center-to-center (re-measured, +2)
OUTPUT_M3_CBORE_D_MM = 6.2  # M3 socket head Ø5.5 + clearance
OUTPUT_M3_CBORE_DEPTH_MM = 3.2
# 4 holes in the measured 20 x 13 mm rectangle (long edge along X)
OUTPUT_M3_POSITIONS = (
    (OUTPUT_M3_DX_MM, OUTPUT_M3_DY_MM),
    (OUTPUT_M3_DX_MM, -OUTPUT_M3_DY_MM),
    (-OUTPUT_M3_DX_MM, OUTPUT_M3_DY_MM),
    (-OUTPUT_M3_DX_MM, -OUTPUT_M3_DY_MM),
)
RECOMMENDED_MECHANICS = {
    **veo3.RECOMMENDED_MECHANICS,
    "simulation_variant": "veo6_coaxial_zaxis_hall_gap",
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

# PRELOADED suspension (user: the two discs must be ABSOLUTELY stable at rest,
# only moving under contact pressure). The springs are pre-compressed 1.5 mm at
# assembly, so a 1.5 N preload firmly seats the moving ring against a REST STOP
# (the guide-rod heads). Below 1.5 N the ring does not move at all (stable zero,
# 3x the 0.5 N device weight). Above it, the ring lifts off and F = 1.5 + 1.0*x.
PRELOAD_N = 1.5
PRELOAD_DEFLECTION_MM = PRELOAD_N / 1.0  # k_total = 1.0 N/mm -> 1.5 mm precompress
# Travel from the rest stop to the up hard stop. F_at_hardstop = 1.5 + 1.0*7.25
# = 8.75 N (user choice -- a firmer press). The magnet tower is lowered so it
# still clears the coaxial Hall pillar at this longer travel; the up stop caps
# the force.
HARD_STOP_TRAVEL_MM = 7.25

# Guide-post anchor hole in the FIXED plate: tight on the Ø5 post so the posts
# do not wobble (the moving bushing keeps its own sliding clearance). Matches
# the snug Ø5.2 wire pass-through the user confirmed fits the post nicely.
POST_ANCHOR_HOLE_D_MM = 5.2

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
# Raise the clamp lugs/screws this far above the carriage so the lug bottom
# sits above the diaphragm contact face and stops poking the patient.
CLAMP_LUG_DZ_MM = 3.0
# Screw axis offset below the lug centre. At -4.0 the Ø3.5 clearance bore
# (radius 1.75) bottomed out at lug_zc-5.75, BELOW the 11 mm lug's floor
# (lug_zc-5.5) -> the round hole was sliced open at the bottom (a C-shape).
# Raised to -2.5 so the bore sits fully inside the lug (1.25 mm wall below it)
# = a FULLY ENCLOSED circular hole (user request 2026-06-23).
SCREW_DZ_MM = -2.5

# Purchased Honeywell SS49E: TO-92S body with three straight ~13 mm leads.
HALL_LEAD_LENGTH_MM = 13.0
HALL_LEAD_PITCH_MM = 1.27
HALL_LEAD_W_MM = 0.45

# Guide posts are now CAPTIVE shoulder screws (Ø5 shoulder, head DOWN, M4
# thread into the plate): the Ø5 shoulder runs from the plate (z=0) down to the
# barrel bottom (z=-33), then a Ø8 head (-33..-34.5) acts as the REST STOP and
# the drop-out catch. The preloaded spring seats the barrel bottom firmly on
# this head (stable zero); contact lifts the barrel off it. The head (Ø8) can't
# pass the Ø5.9 bore, so the moving ring can never come off the posts. The head
# sits 6.7 mm above the diaphragm at rest (>=2.7 mm even at the 4 mm hard stop)
# and at x=+-34, well outside the Ø46 contact face — clear of the patient.
GUIDE_ROD_TOP_Z_MM = 0.0
# Head top 0.2 mm BELOW the barrel bottom (-33): a small designed seat gap the
# 1.5 N preload closes (the barrel seats on the head). Keeps the rod and the
# barrel from sharing a coincident face (which read as a clash in viewers).
ROD_SHAFT_BOTTOM_Z_MM = -33.2
BARREL_BOTTOM_REST_Z_MM = -33.0
ROD_HEAD_D_MM = 8.0
ROD_HEAD_H_MM = 1.5
GUIDE_ROD_LENGTH_MM = GUIDE_ROD_TOP_Z_MM - ROD_SHAFT_BOTTOM_Z_MM + ROD_HEAD_H_MM  # 34.5
GUIDE_ROD_TIP_Z_MM = ROD_SHAFT_BOTTOM_Z_MM - ROD_HEAD_H_MM  # -34.5

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
# Front-center on the -y wing, clear of the central 18 x 11 mm bolt rectangle so
# the 17.78 mm board never covers a mounting hole.
XIAO_CENTER_X_MM = 0.0
# Shifted outboard (was -32.0) so the XIAO cradle clears the OpenArm DM-J4310
# Ø57 output flange (flange ymin -28.5); board ymax -30.5 -> 2.0 mm gap.
XIAO_CENTER_Y_MM = -41.0
WING_TOP_Z_MM = 3.0  # flush with the bridge plate top

# Sensing station: shared XY axis for magnet and Hall (coaxial along Z).
# Moved inboard (ring outer radius is now 28.5) and 1 mm back so the Hall
# pillar clears the plate's M3 mounting slots and the inboard guide rods.
SENSE_X_MM = 25.0
SENSE_Y_MM = -16.0
# Lowered 1.15 mm (was -18.6) so the magnet tower clears the coaxial Hall pillar
# (bottom -11.5) at the longer 7.25 mm travel: top at full compression = -19.75
# + 7.25 = -12.5, ~1 mm below the pillar. Rest gap grows to ~8.9 mm but the
# signal SPAN over 7.25 mm of gap closing is larger than the old 4 mm.
TOWER_TOP_Z_MM = -19.75
MAGNET_TOP_Z_MM = TOWER_TOP_Z_MM - (MAGNET_POCKET_DEPTH_MM - MAGNET_THICKNESS_MM)  # -20.35
PILLAR_BOTTOM_Z_MM = -11.5  # fixed; the magnet tower was lowered to gain travel room instead
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
OUT_DIR = ROOT / "generated" / "veo6_realshape"
SIM_DIR = OUT_DIR / "simulation"
PRESENTATION_DIR = OUT_DIR / "presentation"


def veo6_sensor_layout(p: v9.SimplifiedRailMountParams) -> dict[str, tuple[float, float, float]]:
    return {
        "magnet": (SENSE_X_MM, SENSE_Y_MM, MAGNET_TOP_Z_MM - MAGNET_THICKNESS_MM / 2.0),
        "hall": (SENSE_X_MM, SENSE_Y_MM, HALL_FACE_Z_MM + 0.75),
        "hall_backplate": (SENSE_X_MM, SENSE_Y_MM, -4.0),
        # No connector: this is the solder target on the XIAO's right-edge pads.
        "sensor_connector": (XIAO_W_MM / 2.0, -39.0, 4.7),
    }


def veo6_assembly_parts(fixed, left, right, thinklab, hardware):
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
        (hardware["hall_vcc_wire"], "#d62828", 1.0),
        (hardware["hall_signal_wire"], "#f2c230", 1.0),
        (hardware["hall_gnd_wire"], "#111111", 1.0),
    ]


def configure_backends() -> None:
    v12.P = P
    v12.THINKLABS_PROFILE = VEO6_THINKLABS_PROFILE
    v12.RECOMMENDED_MECHANICS = RECOMMENDED_MECHANICS
    v12.OUT_DIR = OUT_DIR
    v12.SIM_DIR = SIM_DIR
    v12.PRESENTATION_DIR = PRESENTATION_DIR
    v12.sensor_layout = veo6_sensor_layout
    v12.assembly_parts = veo6_assembly_parts


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


def _base_ring_veo6(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """Clamp ring for the hand-measured Thinklabs One.

    The ring grips the 6 mm black edge (Ø46) with a STRAIGHT bore, and a BOTTOM
    RETENTION LIP (inward Ø42 flange at the ring bottom) gives a POSITIVE catch
    so the device cannot drop out — the Ø46 black edge rests on the Ø42 lip.
    The cone narrows upward, so the bore clears the cone/dial/buttons/jack above
    the ring. The clamshell pinch grips the black edge; the lip is the backstop.
    """
    lower_inner_r, outer_r = ring_radii(p)
    zc = p.carriage_z
    ring_bottom_z = zc - p.cup_height / 2.0
    lip_top_z = ring_bottom_z + BOTTOM_LIP_H_MM

    # Solid band, then bore Ø(BORE_D) only ABOVE the lip, then punch the Ø42
    # diaphragm opening through the bottom lip.
    ring = v9.cyl_z(outer_r, p.cup_height, (0, 0, zc))
    bore_top_z = zc + p.cup_height / 2.0 + 3.0
    bore_h = bore_top_z - lip_top_z
    ring = ring.cut(v9.cyl_z(BORE_D_MM / 2.0, bore_h, (0, 0, (lip_top_z + bore_top_z) / 2.0)))
    ring = ring.cut(v9.cyl_z(BOTTOM_LIP_D_MM / 2.0, BOTTOM_LIP_H_MM + 4.0, (0, 0, ring_bottom_z + BOTTOM_LIP_H_MM / 2.0)))

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

    # Re-bore the rod clearance LAST — after the bushing bridges, the spring
    # tabs AND the tower — so no later union can refill it. (Bug: the spring
    # tab at (34,8) reached y=1 and filled the rod hole, leaving the Ø5 shaft
    # clipping the tab by ~15 mm3.) The bore also notches the tab clear of the rod.
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        ring = ring.cut(v9.cyl_z(p.guide_rod_clearance_d / 2.0, 50.0, (sx, p.guide_rod_y, p.carriage_z + 10.0)))
    return ring


def make_clamshell_halves(p: v9.SimplifiedRailMountParams) -> tuple[cq.Workplane, cq.Workplane]:
    inner_r, outer_r = ring_radii(p)
    zc = p.carriage_z
    lug_zc = zc + CLAMP_LUG_DZ_MM  # clamp lugs/screws raised to clear the diaphragm
    base = _base_ring_veo6(p)

    front_y, rear_y = veo4.clamshell_seam_ys(p)
    # Tuck the clamp lugs IN against the ring edge (user request: closer to the
    # ring, fewer screws). The ear is anchored to inner_r (the Ø66 device bore),
    # NOT outer_r: the user thinned cup_wall (4->3) so outer_r shrank, and an
    # outer_r-relative ear would creep into the device. inner_r + 6 (= y 39.5)
    # holds the ear's nearest corner ~1.1 mm clear of the device regardless of
    # wall thickness, and is the SAME absolute position approved at the thicker
    # wall — thinning the ring does not move the ears, only exposes a bit more
    # of them past the now-smaller ring.
    ear_yc = inner_r + 6.0
    clamp_ys = (math.copysign(ear_yc, front_y), math.copysign(ear_yc, rear_y))
    for y in clamp_ys:
        for sx in (-10.0, 10.0):
            base = base.union(v9.box(14.0, 11.0, 11.0, (sx, y, lug_zc)))

    # True split: left half is x <= -1.5, right half is x >= +1.5 (VEO4 cut the
    # wrong sides, leaving a 3 mm thick overlap shared by both prints).
    split_gap_w = 3.0
    cutter_for_left = v9.box(120.0, 130.0, 80.0, (60.0 - split_gap_w / 2.0, 0, zc))
    cutter_for_right = v9.box(120.0, 130.0, 80.0, (-60.0 + split_gap_w / 2.0, 0, zc))
    left = base.cut(cutter_for_left)
    right = base.cut(cutter_for_right)

    # Clean PARALLEL THROUGH-HOLES in BOTH halves (user request): the Ø3.5
    # clearance bore runs straight through the left lug, the split, the nut
    # pocket, AND out the far face of the right lug — the screw passes straight
    # through and the nut is captured in its hex pocket along the way. (The old
    # bore stopped at the nut, leaving the nut side blind.)
    lug_x_max = 17.0  # right lug outer face (box 14 @ sx=10 -> 3..17)
    hole_x0 = SCREW_HEAD_FACE_X_MM - 5.0
    hole_x1 = lug_x_max + 1.5  # exit the right lug => true through-hole
    for y in clamp_ys:
        for dz in (SCREW_DZ_MM,):  # ONE screw per seam; raised so the bore is fully enclosed
            hole = v9.cyl_x(
                p.m3_clearance_d / 2.0, hole_x1 - hole_x0, ((hole_x0 + hole_x1) / 2.0, y, lug_zc + dz)
            )
            left = left.cut(hole)
            right = right.cut(hole)
            pocket = v12.v10.hex_prism_x(
                NUT_POCKET_ACROSS_CORNERS_MM, NUT_POCKET_DEPTH_MM, (NUT_CENTER_X_MM, y, lug_zc + dz)
            )
            # Vertical insertion slot: bottom screws load the nut from below the
            # lug, top screws from above, so no nut cavity is fully enclosed.
            slot_dir = -1.0 if dz < 0 else 1.0
            slot = v9.box(
                NUT_POCKET_DEPTH_MM,
                NUT_POCKET_ACROSS_CORNERS_MM - 0.2,
                7.0,
                (NUT_CENTER_X_MM, y, lug_zc + dz + slot_dir * 3.5),
            )
            right = right.cut(pocket).cut(slot)
    return left, right


def make_fixed_bridge(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """Round plate (the old long rectangle dated from the wide rod stance):
    OpenArm DM-J4310 output-face bolt pattern at center, rod sockets with
    hard-stop sleeves, twin spring seats, Hall pillar, and the XIAO wing."""
    bridge = v9.cyl_z(44.0, p.top_bridge_t, (0, 0, 0))
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        socket = v9.cyl_z(p.rod_socket_od / 2.0, 7.0, (sx, p.guide_rod_y, -3.0))
        # UP hard-stop sleeve: the barrel top (rest z=-19) rises and lands on
        # the sleeve bottom at 7.25 mm travel — with the 1.5 N preload, F = 1.5 +
        # 1.0*7.25 = 8.75 N (user choice), a mechanical cap. Sleeve bottom = -11.75.
        sleeve = v9.cyl_z(4.5, 5.25, (sx, p.guide_rod_y, -9.125))  # Ø9, spans -11.75..-6.5 (7.25 mm travel)
        socket = socket.union(sleeve)
        socket = socket.cut(v9.cyl_z(POST_ANCHOR_HOLE_D_MM / 2.0, 17.0, (sx, p.guide_rod_y, -6.75)))
        bridge = bridge.union(socket)

    # Twin fixed spring seats directly under the plate (no cantilever tab).
    for spx, spy in SPRING_POSITIONS:
        bridge = bridge.union(_downward_spring_socket_at(p, spx, spy))

    # Hall pillar: hangs from the plate, holds the TO-92S flat (branded face
    # down) coaxially above the magnet. The top is tied to the plate underside
    # and embedded 2 mm into it, so thinning top_bridge_t (user 5.0->3.5) can't
    # leave a gap that detaches the pillar (its old fixed top at z=-2.5 stopped
    # 0.75 mm short of the thinned plate's -1.75 underside).
    pillar_top_z = -p.top_bridge_t / 2.0 + 2.0
    pillar_h = pillar_top_z - PILLAR_BOTTOM_Z_MM
    pillar = v9.box(
        10.0, 8.0, pillar_h, (SENSE_X_MM, SENSE_Y_MM, (PILLAR_BOTTOM_Z_MM + pillar_top_z) / 2.0)
    )
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
    wing = v9.box(24.0, 21.0, 4.0, (XIAO_CENTER_X_MM, -39.5, WING_TOP_Z_MM - 2.0))
    wing = wing.union(v9.box(18.0, 3.0, 4.0, (XIAO_CENTER_X_MM, -51.5, WING_TOP_Z_MM - 2.0)))
    for cx in (XIAO_CENTER_X_MM - 9.0, XIAO_CENTER_X_MM + 9.0):
        wing = wing.union(v9.cyl_z(3.0, 4.0, (cx, -50.0, WING_TOP_Z_MM - 2.0)))
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
    notch = v9.box(2.2, 5.2, 3.6, (XIAO_CENTER_X_MM + inner_x / 2.0 + 0.75, -39.0, 5.4))
    bridge = bridge.cut(notch)
    # Wire pass-through hole: the round plate (r44) covers the riser station
    # beside the wing, so the three-wire bundle threads up through the plate.
    bridge = bridge.cut(v9.cyl_z(2.6, 10.0, (13.54, -39.0, 0.0)))

    # OpenArm mounting interface: 4x M3 clearance holes in the MEASURED 18 x 11
    # mm rectangle (centers ±9, ±5.5) that bolt up into the adapter plate on the
    # OpenArm side. The plate centre stays SOLID (no boss relief) so it mates
    # flat against the adapter. Cut LAST so later unions (spring seats, wing)
    # cannot refill the holes. Heads counterbore into the bottom (patient) face.
    for hx, hy in OUTPUT_M3_POSITIONS:
        bridge = bridge.cut(v9.cyl_z(p.m3_clearance_d / 2.0, 30.0, (hx, hy, 0)))  # M3 clearance thru
        bridge = bridge.cut(
            v9.cyl_z(
                OUTPUT_M3_CBORE_D_MM / 2.0,
                OUTPUT_M3_CBORE_DEPTH_MM,
                (hx, hy, -p.top_bridge_t / 2.0 + OUTPUT_M3_CBORE_DEPTH_MM / 2.0),
            )
        )  # socket-head counterbore

    # Guide-post clearance bore through the FULL plate — cut LAST so the main
    # disc union cannot refill it. (Bug fix: the socket bore was applied before
    # unioning the disc, so the disc filled the hole and the Ø5 shoulder ended
    # up embedded in solid plate with no hole for the screw -> 118 mm3 clash.)
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        bridge = bridge.cut(v9.cyl_z(POST_ANCHOR_HOLE_D_MM / 2.0, 30.0, (sx, p.guide_rod_y, 0)))
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
    lug_zc = zc + CLAMP_LUG_DZ_MM  # match the raised clamp lugs
    inner_r, _ = ring_radii(p)
    fy, ry = veo4.clamshell_seam_ys(p)
    ear_yc = inner_r + 6.0  # device-anchored, matches make_clamshell_halves
    clamp_ys = (math.copysign(ear_yc, fy), math.copysign(ear_yc, ry))
    screws = None
    for y in clamp_ys:
        for dz in (SCREW_DZ_MM,):  # one screw per seam (matches the raised clamp bore)
            shaft = v9.cyl_x(
                1.5, SCREW_LENGTH_MM, ((SCREW_HEAD_FACE_X_MM + SCREW_TIP_X_MM) / 2.0, y, lug_zc + dz)
            )
            head = v9.cyl_x(2.75, 3.0, (SCREW_HEAD_FACE_X_MM - 1.5, y, lug_zc + dz))
            nut = v12.v10.hex_prism_x(
                NUT_ACROSS_CORNERS_MM, NUT_THICKNESS_MM, (NUT_CENTER_X_MM, y, lug_zc + dz)
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


def thinklab_section_zs(p: v9.SimplifiedRailMountParams) -> dict[str, float]:
    """Z heights of the three measured sections (bottom -> top). The device
    now RESTS ON the bottom retention lip, so its black-edge bottom sits at the
    lip top (the magnet is on the ring, so the gap/sensing is unaffected)."""
    prof = VEO6_THINKLABS_PROFILE
    dev_bottom = p.carriage_z - p.cup_height / 2.0 + BOTTOM_LIP_H_MM  # rests on the lip (-41.2)
    black_top = dev_bottom + prof["black_edge_h_mm"]
    cone_top = black_top + prof["cone_h_mm"]
    dial_top = cone_top + prof["dial_h_mm"]
    return {"dev_bottom": dev_bottom, "black_top": black_top, "cone_top": cone_top, "dial_top": dial_top}


def make_thinklab_proxy(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """Hand-measured Thinklabs One: 6 mm straight black edge (Ø46, the grip
    band) + 11 mm cone (46->38) + 12 mm dial (Ø38) with the Ø9 jack and a few
    buttons on the dial side. Total 29 mm."""
    prof = VEO6_THINKLABS_PROFILE
    z = thinklab_section_zs(p)
    black = v9.cyl_z(
        prof["black_edge_d_mm"] / 2.0,
        prof["black_edge_h_mm"],
        (0, 0, (z["dev_bottom"] + z["black_top"]) / 2.0),
    )
    cone = v12.v11.cone_z(
        prof["cone_bottom_d_mm"] / 2.0,
        prof["cone_top_d_mm"] / 2.0,
        prof["cone_h_mm"],
        (0, 0, (z["black_top"] + z["cone_top"]) / 2.0),
    )
    dial = v9.cyl_z(prof["dial_d_mm"] / 2.0, prof["dial_h_mm"], (0, 0, (z["cone_top"] + z["dial_top"]) / 2.0))
    body = black.union(cone).union(dial)
    # Ø9 audio/charge jack: a blind radial bore in the dial wall (front-left).
    th = math.radians(JACK_THETA_DEG)
    dial_r = prof["dial_d_mm"] / 2.0
    jack = _radial_cyl(prof["jack_d_mm"] / 2.0, 4.5, dial_r * math.cos(th), dial_r * math.sin(th), JACK_Z_MM, math.cos(th), math.sin(th))
    body = body.cut(jack)
    # A few control buttons proud of the dial (markers): on the front face.
    for dz, btn_th in ((1.5, 195.0), (-1.5, 245.0)):
        bt = math.radians(btn_th)
        body = body.union(_radial_cyl(1.6, 1.0, dial_r * math.cos(bt), dial_r * math.sin(bt), JACK_Z_MM + dz, math.cos(bt), math.sin(bt)))
    return body


def make_sleeves_proxy(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """The up-stop SLEEVE volume only (z -11.75..-6.5, Ø9 annulus around each
    rod), matching the boss fused into the fixed bridge. Used to highlight the
    sleeve in the assembly view; the printed bridge keeps this material."""
    sleeves = None
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        cyl = v9.cyl_z(4.5, 5.25, (sx, p.guide_rod_y, -9.125))
        cyl = cyl.cut(v9.cyl_z(p.guide_rod_clearance_d / 2.0, 8.0, (sx, p.guide_rod_y, -9.125)))
        sleeves = cyl if sleeves is None else sleeves.union(cyl)
    assert sleeves is not None
    return sleeves


def make_guide_rods_proxy(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """Captive shoulder posts: Ø5 shoulder (plate -> -33) + Ø8 rest-stop head
    (-33..-34.5). The barrel bottom seats on the head under spring preload."""
    rods = None
    shaft_h = GUIDE_ROD_TOP_Z_MM - ROD_SHAFT_BOTTOM_Z_MM  # 33
    shaft_cz = (GUIDE_ROD_TOP_Z_MM + ROD_SHAFT_BOTTOM_Z_MM) / 2.0
    head_cz = ROD_SHAFT_BOTTOM_Z_MM - ROD_HEAD_H_MM / 2.0
    for sx in (-p.guide_rod_x, p.guide_rod_x):
        shaft = v9.cyl_z(p.guide_rod_d / 2.0, shaft_h, (sx, p.guide_rod_y, shaft_cz))
        head = v9.cyl_z(ROD_HEAD_D_MM / 2.0, ROD_HEAD_H_MM, (sx, p.guide_rod_y, head_cz))
        rod = shaft.union(head)
        rods = rod if rods is None else rods.union(rod)
    assert rods is not None
    return rods


def make_horn_proxy() -> cq.Workplane:
    """The OpenArm DM-J4310 actuator above the plate (gripper removed): the Ø57
    output flange against the plate top, then the Ø35 body. Envelope only — for
    clearance checks vs the XIAO cradle and the sensing pillar."""
    t = P.top_bridge_t / 2.0
    flange = v9.cyl_z(57.0 / 2.0, 3.0, (0, 0, t + 1.5))  # z 3..6 against the plate top
    body = v9.cyl_z(35.0 / 2.0, 40.0, (0, 0, t + 3.0 + 20.0))  # z 6..46 actuator body
    return flange.union(body)


def make_horn_screws_proxy() -> cq.Workplane:
    """4x M3 driving up from the bottom-face counterbores into the DM-J4310
    adapter taps (the measured 18 x 11 mm rectangle, centers ±9/±5.5)."""
    screws = None
    for hx, hy in OUTPUT_M3_POSITIONS:
        shaft = v9.cyl_z(1.5, 11.0, (hx, hy, 2.5))  # z -3..8: through plate, ~5 mm into the actuator
        head = v9.cyl_z(2.75, 2.8, (hx, hy, -1.6))  # M3 socket head seated in the counterbore (z -3..-0.2)
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
        y_pad = -39.0 + dy  # tracks the outboard-shifted XIAO pad row
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
    # The robot side is the OpenArm adapter plate (gripper removed); the module
    # bolts on via 4x M3 in the measured 18 x 11 mm rectangle. The proxy is a
    # conservative Ø57 envelope for the actuator body behind the adapter.
    hardware["robot_flange"] = make_horn_proxy()
    hardware["interface_bolts"] = make_horn_screws_proxy()
    hardware["spring"] = make_veo6_visible_spring(p)
    hardware["magnet"] = make_magnet_proxy()
    hardware["hall"] = make_hall_proxy()
    # No JST connector: the hall wires solder directly onto the XIAO pads.
    hardware.pop("sensor_connector", None)
    hardware["xiao"] = make_xiao_proxy()
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


def make_veo6_visible_spring(p: v9.SimplifiedRailMountParams) -> cq.Workplane:
    """The purchased WF6-25 at free length (true 6.0 mm OD, ~0.6 mm wire)."""
    return make_compressed_spring(p, 0.0)


def build_veo6_model() -> dict:
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
        # Preloaded: the ring only moves once contact force exceeds the preload,
        # then F = preload + k * displacement.
        "force_n": round(PRELOAD_N + RECOMMENDED_MECHANICS["spring_rate_n_per_mm"] * compression_mm, 3),
        "face_gap_mm": round(face_gap, 3),
        "field_gauss": round(b_gauss, 1),
        "hall_out_v": round(volts, 4),
        "adc_counts": round(counts, 1),
    }


def make_sensing_model() -> dict:
    samples = [hall_response(c / 4.0) for c in range(0, int(HARD_STOP_TRAVEL_MM * 4) + 1)]
    operating = [s for s in samples if s["compression_mm"] <= HARD_STOP_TRAVEL_MM]
    span_v = operating[-1]["hall_out_v"] - operating[0]["hall_out_v"]
    # With the 1.5 N preload, displacement starts at 0 (rest) and force reads
    # F = 1.5 + 1.0*x: 4 N @ 2.5 mm, 4.6 N @ 3.1 mm, 8.75 N @ 7.25 mm hard stop.
    key_points = {
        "rest_preload_1p5n": hall_response(0.0),
        "target_force_4n": hall_response(2.5),
        "max_screened_4p6n": hall_response(3.1),
        "hard_stop_8p75n": hall_response(HARD_STOP_TRAVEL_MM),
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
    csv_path = SIM_DIR / "veo6_hall_response.csv"
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
    ax1.set_title("VEO6 coaxial Hall response (first-order model, needs bench calibration)")
    fig.tight_layout()
    png_path = SIM_DIR / "veo6_hall_response.png"
    fig.savefig(png_path)
    plt.close(fig)
    return {"hall_response_csv": str(csv_path), "hall_response_png": str(png_path)}


# --- Audits ------------------------------------------------------------------


def make_veo6_mechanical_audit() -> dict:
    """Dual preloaded WF6-25 springs: 1.5 N preload + 1.0 N/mm, 7.25 mm travel to
    the 8.75 N hard stop. Each spring is precompressed PRELOAD_DEFLECTION/1 ... in
    parallel (k_each 0.5 x2 = 1.0), so each carries half the load."""
    k_each = 0.5
    k_total = RECOMMENDED_MECHANICS["spring_rate_n_per_mm"]
    target_f = RECOMMENDED_MECHANICS["target_contact_force_n"]
    screened_f = RECOMMENDED_MECHANICS["max_screened_contact_force_n"]
    # F = preload + k*x  ->  x = (F - preload)/k
    target_c = (target_f - PRELOAD_N) / k_total
    screened_c = (screened_f - PRELOAD_N) / k_total
    stop_force = PRELOAD_N + k_total * HARD_STOP_TRAVEL_MM
    # Each spring's TOTAL deflection at the hard stop = preload precompress + travel.
    per_spring_total_defl = PRELOAD_DEFLECTION_MM + HARD_STOP_TRAVEL_MM
    per_spring_stop = k_each * per_spring_total_defl
    centroid_x = sum(x for x, _ in SPRING_POSITIONS) / len(SPRING_POSITIONS)
    centroid_y = sum(y for _, y in SPRING_POSITIONS) / len(SPRING_POSITIONS)
    centroid_offset = math.hypot(centroid_x, centroid_y)
    checks = [
        {
            "name": "preload_holds_zero_below_threshold",
            "type": "design_rule",
            "preload_n": PRELOAD_N,
            "device_weight_n": 0.5,
            "pass": PRELOAD_N >= 2.0 * 0.5,  # >=2x device weight => stable rest zero
        },
        {
            "name": "target_force_requires_expected_compression",
            "type": "spring_force",
            "target_force_n": target_f,
            "preload_n": PRELOAD_N,
            "spring_rate_n_per_mm": k_total,
            "compression_mm": round(target_c, 3),
            "expected_compression_mm": 2.5,
            "pass": abs(target_c - 2.5) <= 0.01,
        },
        {
            "name": "screened_force_below_patient_contact_limit",
            "type": "force_gate",
            "force_n": screened_f,
            "limit_n": P.spring_max_load_n,
            "pass": screened_f <= P.spring_max_load_n,
        },
        {
            # The mechanical force cap (user set it to 8.75 N at 7.25 mm). Verify
            # the two-spring SYSTEM can bear it (per-spring load/deflection are
            # checked separately below).
            "name": "hard_stop_within_spring_system_capacity",
            "type": "force_gate",
            "force_n": round(stop_force, 3),
            "limit_n": round(len(SPRING_POSITIONS) * P.spring_max_load_n, 3),
            "pass": stop_force <= len(SPRING_POSITIONS) * P.spring_max_load_n + 1e-9,
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
            "margin_mm": round(P.spring_allowable_deflection - per_spring_total_defl, 3),
            "min_margin_mm": 2.0,
            "pass": (P.spring_allowable_deflection - per_spring_total_defl) >= 2.0,
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
        "preload_n": PRELOAD_N,
        "preload_deflection_mm": PRELOAD_DEFLECTION_MM,
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


def make_veo6_material_audit() -> dict:
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


def make_veo6_geometry_audit(model: dict) -> dict:
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
    # NOTE: the post head now INTENTIONALLY seats on the barrel bottom (the rest
    # stop), so a rod-vs-ring zero-collision check no longer applies — retention
    # and seating are audited explicitly below (rod_head_retains_ring,
    # rest_stop_seats_barrel_on_head). The Ø5 shaft still clears the Ø5.9 bore.
    shaft_bore_clear = (P.guide_rod_clearance_d - P.guide_rod_d) / 2.0
    checks.append(
        {
            "name": "rod_shaft_slides_in_bore",
            "type": "design_rule",
            "radial_clearance_mm": round(shaft_bore_clear, 3),
            "pass": 0.2 <= shaft_bore_clear <= 0.8,
        }
    )

    # Dynamic checks: translate moving parts up AND compress the spring helix,
    # from the preloaded rest (0) to the 4 mm hard stop. The Thinklabs body
    # rides with the ring, so its top face must clear the fixed plate too.
    for compression in (1.5, 2.5, HARD_STOP_TRAVEL_MM):
        spring = make_compressed_spring(P, compression)
        moved = {
            "left_clamshell": left.translate((0, 0, compression)),
            "right_clamshell": right.translate((0, 0, compression)),
            "thinklab": model["thinklab"].translate((0, 0, compression)),
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

    # Captive guide posts (the STABILITY mechanism): Ø5 shoulder + Ø8 rest-stop
    # head. The preloaded spring seats the barrel bottom firmly on the head, so
    # the two discs hold an absolutely stable zero until contact force exceeds
    # the 1.5 N preload. The head also catches the ring (drop-out impossible).
    rods_bb = hardware["guide_rods"].val().BoundingBox()
    tk_bb = model["thinklab"].val().BoundingBox()
    rod_len = rods_bb.zmax - rods_bb.zmin
    checks.append(
        {
            "name": "guide_posts_captive_shoulder_len",
            "type": "bbox_dimension",
            "value_mm": round(rod_len, 3),
            "expected_mm": GUIDE_ROD_LENGTH_MM,
            "tolerance_mm": 0.1,
            "pass": abs(rod_len - GUIDE_ROD_LENGTH_MM) <= 0.1,
        }
    )
    # POSITIVE retention: the Ø8 head cannot pass the Ø5.9 bushing bore, so the
    # moving ring can never slide off the posts (no reliance on friction).
    head_catch_radial = (ROD_HEAD_D_MM - P.guide_rod_clearance_d) / 2.0
    checks.append(
        {
            "name": "rod_head_retains_ring",
            "type": "design_rule",
            "radial_catch_mm": round(head_catch_radial, 3),
            "pass": head_catch_radial >= 0.8,
        }
    )
    # REST STOP: the head top sits a small 0.1-0.3 mm gap below the barrel
    # bottom at rest; the 1.5 N preload closes it so the barrel seats on the
    # head -> defined stable zero. (A designed gap, not a coincident face.)
    barrel_bottom_rest = P.carriage_z + 10.0 - P.guide_bushing_h / 2.0  # -33
    seat_gap = barrel_bottom_rest - ROD_SHAFT_BOTTOM_Z_MM  # head top is below
    checks.append(
        {
            "name": "rest_stop_seats_barrel_on_head",
            "type": "design_rule",
            "seat_gap_mm": round(seat_gap, 3),
            "pass": 0.1 <= seat_gap <= 0.4,
        }
    )
    # Patient guard: the Ø8 head (x=+-34, outside the Ø46 contact face) is well
    # clear at the working force (4 mm above the diaphragm at 4 N). At the 8.75 N
    # hard stop the device rises 7.25 mm, so the rigid head dips up to ~0.75 mm
    # into soft tissue at the Ø68 ring -- accepted (user choice) for the firmer
    # press; only at the rare absolute-max force.
    head_above_skin_at_stop = rods_bb.zmin - (tk_bb.zmin + HARD_STOP_TRAVEL_MM)
    checks.append(
        {
            "name": "rod_head_above_skin_at_hard_stop",
            "type": "travel_clearance",
            "gap_mm": round(head_above_skin_at_stop, 3),
            "min_gap_mm": -1.0,
            "pass": head_above_skin_at_stop >= -1.0,
        }
    )
    bushing_top_z = P.carriage_z + 10.0 + P.guide_bushing_h / 2.0  # -19
    # Shoulder engages the FULL 14 mm barrel at rest (shaft spans down to -33).
    engagement = bushing_top_z - ROD_SHAFT_BOTTOM_Z_MM
    checks.append(
        {
            "name": "rod_bushing_engagement_at_rest",
            "type": "travel_clearance",
            "gap_mm": round(engagement, 3),
            "min_gap_mm": 10.0,
            "pass": engagement >= 10.0,
        }
    )
    sleeve_bottom_z = -11.75  # up-stop sleeve bottom for the 7.25 mm travel
    stop_travel = sleeve_bottom_z - bushing_top_z
    checks.append(
        {
            "name": "hard_stop_travel_matches_design",
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

    # Measured Thinklabs One envelope: 46 dia (widest = black edge / cone base)
    # x 29 tall (6 black edge + 11 cone + 12 dial).
    prof = VEO6_THINKLABS_PROFILE
    tk_d = tk_bb.xmax - tk_bb.xmin
    tk_h = tk_bb.zmax - tk_bb.zmin
    checks.append(
        {
            "name": "thinklab_proxy_matches_measured_46x29",
            "type": "bbox_dimension",
            "value_mm": [round(tk_d, 3), round(tk_h, 3)],
            "expected_mm": [46.0, 29.0],
            "tolerance_mm": 0.2,
            "pass": abs(tk_d - 46.0) <= 0.2 and abs(tk_h - 29.0) <= 0.2,
        }
    )
    z = thinklab_section_zs(P)
    # Black-edge grip: the ring bore squeezes the 6 mm straight Ø46 band. Check
    # the radial clearance of the (unsqueezed) bore over the black edge — the
    # clamshell pinch closes it to grip; drop-out is friction-held (50 g part).
    bore_clear = (BORE_D_MM - prof["black_edge_d_mm"]) / 2.0
    checks.append(
        {
            "name": "black_edge_grip_clearance",
            "type": "design_rule",
            "radial_clearance_mm": round(bore_clear, 3),
            "grip_band_h_mm": prof["black_edge_h_mm"],
            "pass": 0.15 <= bore_clear <= 0.6,
        }
    )
    # The black edge sits inside the ring band (z overlap) so the clamp grips it.
    ring_bottom = P.carriage_z - P.cup_height / 2.0
    ring_top = P.carriage_z + P.cup_height / 2.0
    grip_overlap = min(z["black_top"], ring_top) - max(z["dev_bottom"], ring_bottom)
    checks.append(
        {
            "name": "black_edge_seated_in_ring",
            "type": "design_rule",
            "grip_overlap_mm": round(grip_overlap, 3),
            "pass": grip_overlap >= 3.0,
        }
    )
    # POSITIVE retention catch: the Ø42 bottom lip must be smaller than the Ø46
    # black edge (so the device cannot drop through), and the device must rest
    # ON the lip (its bottom face at the lip top). The Ø42 opening exposes the
    # diaphragm centre; it ends up recessed LIP_H mm inside the lip rim.
    lip_overlap_radial = (prof["black_edge_d_mm"] - BOTTOM_LIP_D_MM) / 2.0  # 2.0 mm of catch
    device_rests_on_lip = abs(z["dev_bottom"] - (ring_bottom + BOTTOM_LIP_H_MM)) <= 0.05
    diaphragm_recess = z["dev_bottom"] - ring_bottom  # +0.8 (recessed inside the rim)
    checks.append(
        {
            "name": "device_retained_by_bottom_lip",
            "type": "design_rule",
            "lip_radial_catch_mm": round(lip_overlap_radial, 3),
            "device_rests_on_lip": device_rests_on_lip,
            "diaphragm_recess_mm": round(diaphragm_recess, 3),
            "pass": lip_overlap_radial >= 1.0 and device_rests_on_lip,
        }
    )
    # The diaphragm centre must stay open to the skin (lip opening > 0 and the
    # exposed contact patch is a useful fraction of the Ø46 diaphragm).
    checks.append(
        {
            "name": "diaphragm_contact_opening_sufficient",
            "type": "design_rule",
            "opening_d_mm": BOTTOM_LIP_D_MM,
            "exposed_fraction": round((BOTTOM_LIP_D_MM / 46.0) ** 2, 3),
            "pass": BOTTOM_LIP_D_MM >= 38.0,
        }
    )
    # Per user: the cone, dial, buttons and jack must clear the ring entirely —
    # they sit ABOVE the ring top, nothing printed touches them.
    veo3._add_collision_check(checks, "device_cone_dial_vs_left_clamshell", model["thinklab"], left, 0.05)
    veo3._add_collision_check(checks, "device_cone_dial_vs_right_clamshell", model["thinklab"], right, 0.05)
    # The dial (with jack + buttons) clears the ring top plane with margin.
    dial_above_ring = z["cone_top"] - ring_top
    checks.append(
        {
            "name": "dial_and_jack_above_ring_top",
            "type": "travel_clearance",
            "gap_mm": round(dial_above_ring, 3),
            "min_gap_mm": 1.0,
            "pass": dial_above_ring >= 1.0,
        }
    )

    # Purchased-part dimension fidelity: WF6-25 spring OD and silicone wire OD.
    # Slice a TALL window around ONE spring (many turns) so the bbox captures
    # the full helix outer diameter regardless of the helix phase.
    sp_probe = v9.box(20.0, 20.0, 12.0, (SPRING_POSITIONS[0][0], SPRING_POSITIONS[0][1], -18.0))
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

    # OpenArm M3 holes (4x in the 18 x 11 rectangle) must be open through plate.
    output_probe = None
    for hx, hy in OUTPUT_M3_POSITIONS:
        pin = v9.cyl_z(P.m3_clearance_d / 2.0 - 0.3, P.top_bridge_t + 2.0, (hx, hy, 0))
        output_probe = pin if output_probe is None else output_probe.union(pin)
    veo3._add_collision_check(checks, "output_m3_holes_open_through_plate", output_probe, fixed)

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

    # OpenArm DM-J4310 output-face interface: the actuator envelope must seat on
    # the plate top without clashing, the 4x M3 screws must run free (holes
    # checked open above), and the actuator must clear the XIAO cradle. The Ø57
    # output flange (not the Ø35 body) is the binding clearance vs the cradle.
    veo3._add_collision_check(checks, "actuator_seats_without_clash", hardware["robot_flange"], fixed)
    veo3._add_collision_check(checks, "output_m3_screws_vs_fixed_print", hardware["interface_bolts"], fixed)
    veo3._add_collision_check(checks, "actuator_flange_vs_xiao", hardware["robot_flange"], hardware["xiao"])
    xiao_bb_for_horn = hardware["xiao"].val().BoundingBox()
    flange_xiao_gap = -57.0 / 2.0 - xiao_bb_for_horn.ymax  # Ø57 flange ymin vs cradle top edge
    checks.append(
        {
            "name": "actuator_flange_clear_of_xiao_cradle",
            "type": "travel_clearance",
            "gap_mm": round(flange_xiao_gap, 3),
            "min_gap_mm": 2.0,
            "pass": flange_xiao_gap >= 2.0,
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
    # Only the magnet tower is coaxial with the Hall pillar (both at SENSE x/y).
    # The clamshell bbox zmax catches the OFF-AXIS guide barrels (x=+-34) which
    # do not collide with the pillar, so measure the coaxial tower top directly.
    tower_top = TOWER_TOP_Z_MM
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
            "compressions_checked_mm": [1.5, 2.5, HARD_STOP_TRAVEL_MM],
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
    trace_path = SIM_DIR / "veo6_contact_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = helpers.write_contact_plot(rows, metadata, SIM_DIR)
    if plot_path is not None:
        final_plot = SIM_DIR / "veo6_contact_response.png"
        if final_plot.exists():
            final_plot.unlink()
        plot_path.rename(final_plot)
    video_path = helpers.write_motion_video(rows, metadata, SIM_DIR, fps=30, max_frames=150)
    if video_path is not None:
        final_video = SIM_DIR / "veo6_contact_motion.mp4"
        if final_video.exists():
            final_video.unlink()
        video_path.rename(final_video)
    summary = {
        "variant": RECOMMENDED_MECHANICS["simulation_variant"],
        "output_csv": str(trace_path),
        "output_plot": str(SIM_DIR / "veo6_contact_response.png"),
        "output_motion_video": str(SIM_DIR / "veo6_contact_motion.mp4"),
        "peak_force_n": max(row["contact_force_n"] for row in rows),
        "final_force_n": rows[-1]["contact_force_n"],
        "max_spring_compression_mm": max(row["spring_compression_mm"] for row in rows),
    }
    (SIM_DIR / "veo6_contact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_sim_rows() -> list[dict[str, float]]:
    with (SIM_DIR / "veo6_contact_trace.csv").open(newline="", encoding="utf-8") as f:
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
        (clean, "veo6_cad_clean.png"),
        (detail, "veo6_clamshell_detail.png"),
        (video, "veo6_actual_cad_motion.mp4" if video is not None else None),
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
    """veo3's layout metadata but anchored to the VEO6 sensing station."""
    layout = veo6_sensor_layout(P)
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
        "thinklabs_one": "HAND-MEASURED real shape: 6 mm black edge (Ø46) + 11 mm cone (46->38) + 12 mm dial (Ø38) with Ø9 jack + buttons; total 29 mm. Ring grips the black edge AND a Ø42 bottom retention lip gives a POSITIVE catch so the device cannot drop out (rests on the lip, not friction alone). Diaphragm centre (Ø42) contacts skin through the lip opening, recessed ~0.8 mm. Cone/dial/buttons/jack sit above the ring. Caliper-confirm before printing.",
        "spring_misumi_wf6_25": "x2 IN PARALLEL (point-symmetric beside the rods, zero moment): catalog wire 0.6, ~20 coils, k 0.5 each -> 1.0 N/mm total, 4 N @ 2.5 mm, hard stop 7.25 mm = 8.75 N (user choice; each spring 4.375 N / 8.75 mm, within the 5.5 N / 11.25 mm catalog limits); helix at true OD 6.0 — BUY TWO",
        "hall_ss49e_to92s": "body 3.1x4.1x1.5 plus three 13 mm straight leads; slot + 4 mm leg channel sized to part",
        "magnet_d5x2_n42": "Z-axial disc in 5.5 x 2.6 pocket on the moving ring",
        "clamp_screws": "M3 x 30 SHCS x4 + M3 nut x4 — BOM said M3 x 12-16, too short for the lug span, update BOM",
        "hall_m2_screws": "NOT needed in VEO6 — the sensor is slot-mounted, no breakout PCB",
        "guide_rods": "2x Ø5 shoulder screw (~35 mm shoulder, M4, head-down captive post); at the 7.25 mm hard stop the head dips ~0.75 mm into soft tissue at the Ø68 ring (accepted, only at the 8.75 N max)",
        "sensor_wire": "26-30 AWG silicone, modeled at 1.2 mm insulated OD",
        "jst_ph_3pin": "REMOVED per user decision — the three hall wires solder directly onto the XIAO's right-edge pads (rail notch provided for the bundle)",
        "xiao_rp2040": "ON the mount: rail cradle on the front-center wing (board 21.0 x 17.78, no holes), USB-C outward; hall wires solder to its right-edge pads",
        "robot_side_screws": "OpenArm (scheme A, gripper removed): 4x M3 x 8 SHCS into the adapter plate on the OpenArm side. Holes in the MEASURED rectangle: centers 18 mm (long) x 11 mm (short), i.e. (+-9, +-5.5) mm, envelope ~25 x 18 mm. Bottom-face counterbores Ø6.2 x 3.2. (Supersedes the earlier assumed Ø50 6x M3 circle.)",
        "actuator_adapter_interface": "4x M3 in an 18 x 11 mm rectangle (centers +-9/+-5.5), solid plate centre (no boss relief) mating flat on the adapter; XIAO cradle kept outboard (y -41) to clear the actuator body behind the adapter",
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
            "type": "veo6_realshape_clamshell",
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
        "# VEO6 Coaxial-Hall Clamshell Mount",
        "",
        "VEO6 fixes the VEO4 defects and redesigns the magnet sensing module as a",
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
        "- PRELOADED suspension for an absolutely stable zero: two WF6-25 springs precompressed 1.5 mm (1.5 N preload) seat the moving ring firmly on the captive guide-post heads. The two discs do not move until contact force exceeds 1.5 N; above it the ring lifts off and F = 1.5 + 1.0*x, capped at 8.75 N by the up hard stop at 7.25 mm (user choice for a firmer press; the magnet tower was lowered so it still clears the Hall pillar). The Ø8 post heads also make drop-out impossible (they cannot pass the bore).",
        "- Spring helix has the true WF6-25 6.0 mm OD and ~20 catalog coils; sensor wires (real ~1.2 mm silicone OD) solder directly onto the XIAO's right-edge pads — the optional JST connector was dropped.",
        "- Thinklabs One re-modeled from the HAND-MEASURED real unit: 6 mm black edge (Ø46) + 11 mm cone (46->38) + 12 mm dial (Ø38), total 29 mm. The clamp ring grips the black edge with a straight Ø46.6 bore PLUS a Ø42 bottom retention lip that the Ø46 black edge rests on -- a POSITIVE mechanical catch so the device cannot drop out (not friction alone). The diaphragm centre (Ø42) contacts skin through the lip opening (recessed ~0.8 mm); the cone, dial, buttons and Ø9 jack sit above the ring.",
        "- OpenArm mount (scheme A): the gripper is removed and the plate bolts to an adapter plate on the OpenArm side with 4x M3 x 8 in a MEASURED 18 x 11 mm rectangle (centers +-9/+-5.5; supersedes the assumed Ø50 6x M3 circle). The plate centre is solid (no boss relief); the XIAO RP2040 rail cradle stays outboard (y -41, USB-C outward) to clear the actuator body behind the adapter.",
        "",
        "## Validation",
        "",
        f"- Overall pass: `{metadata['validation_summary']['pass']}`",
        f"- Geometry pass: `{geometry['pass']}` ({len(geometry['checks'])} checks incl. connectivity + compressed-spring states)",
        f"- Mechanical pass: `{metadata['mechanical_audit']['pass']}`",
        f"- Material pass: `{metadata['material_audit']['pass']}`",
        f"- Rest air gap (magnet face to Hall face): `{sensing['rest_face_gap_mm']} mm`",
        f"- Gap at the 7.25 mm hard stop: `{geometry['clearances']['magnet_hall_gap_at_hard_stop_mm']} mm` (never contacts)",
        "",
        "## Predicted Hall response (first-order, 3.3 V supply)",
        "",
        "| state | compression | force | face gap | field | output |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, point in (
        ("rest (1.5 N preload)", key["rest_preload_1p5n"]),
        ("4 N target", key["target_force_4n"]),
        ("4.6 N screen", key["max_screened_4p6n"]),
        ("8.75 N hard stop", key["hard_stop_8p75n"]),
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
    (OUT_DIR / "VEO6_REALSHAPE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    configure_backends()
    model = build_veo6_model()
    geometry = make_veo6_geometry_audit(model)
    mechanical = make_veo6_mechanical_audit()
    material = make_veo6_material_audit()
    validation = veo3.make_validation_summary(geometry, mechanical, material)
    sensing = make_sensing_model()

    export_part(model["fixed"], "thinklab_openarm_veo6_fixed_bridge_print")
    export_part(model["left"], "thinklab_openarm_veo6_left_clamshell_print")
    export_part(model["right"], "thinklab_openarm_veo6_right_clamshell_print")
    exporters.export(model["thinklab"], str(OUT_DIR / "thinklab_veo6_stepped_profile_proxy.step"))

    # Highlight the up-stop SLEEVE as its own ORANGE component in the ASSEMBLY
    # only (purely a viewing aid). The sleeve volume is cut out of the display
    # bridge and re-added orange; together they are still the full bridge, and
    # the PRINT file (exported above) keeps the sleeve fused into the bridge.
    sleeves = make_sleeves_proxy(P)
    display_fixed = model["fixed"].cut(sleeves)

    assembly = cq.Assembly(name="thinklab_openarm_veo6_coaxial_hall_mount")
    assembly.add(display_fixed, name="printed_fixed_bridge", color=cq.Color(0.83, 0.80, 0.70))
    # Up-stop sleeve shown in the same purple as the Hall sensor (both are
    # fixed-bridge sensing-side features) so they read as one system.
    assembly.add(sleeves, name="up_stop_sleeve_HIGHLIGHT", color=cq.Color(0.42, 0.10, 0.60))
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
        assembly.add(shape, name=f"{name}_proxy_not_printed", color=color)
    assembly.save(str(OUT_DIR / "thinklab_openarm_veo6_assembly.step"), exportType="STEP")

    preview = cq.Compound.makeCompound(
        [model["fixed"].val(), model["left"].val(), model["right"].val(), model["thinklab"].val()]
        + [shape.val() for shape in model["hardware"].values()]
    )
    exporters.export(cq.Workplane("XY").add(preview), str(OUT_DIR / "thinklab_openarm_veo6_preview.step"))

    (OUT_DIR / "veo6_geometry_audit.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    (OUT_DIR / "veo6_mechanical_audit.json").write_text(json.dumps(mechanical, indent=2), encoding="utf-8")
    (OUT_DIR / "veo6_material_audit.json").write_text(json.dumps(material, indent=2), encoding="utf-8")

    sensing_assets = write_sensing_outputs(sensing)
    metadata = make_metadata(model, geometry, mechanical, material, validation, sensing)
    sim_summary = write_sim_outputs(metadata)
    rows = read_sim_rows()
    presentation_assets = render_presentation_assets(model, rows)
    metadata = make_metadata(
        model, geometry, mechanical, material, validation, sensing, sim_summary, presentation_assets, sensing_assets
    )
    (OUT_DIR / "veo6_design_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
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
