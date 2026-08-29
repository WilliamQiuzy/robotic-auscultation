"""Hand-eye calibration tip cap V1 for the ThinkLabs One puck.

Why this part exists (2026-08-29): the commissioned tip-touch protocol solves
camera<->base with the tool tip as the correspondence point. R4 ran it with the
bare stethoscope: a flush Ø46 diaphragm gives ~±20 mm of contact/click
ambiguity per sample, and the run landed at training RMS 9.26 mm — three times
the 3.0 mm validated-tier gate. This cap replaces the disc contact with a
single sharp apex.

Two properties matter; nothing else does:
  * RIGID seat on the puck (no wobble between clicks) — the calibration solver
    estimates ``tip_offset_tcp_m`` itself, so the printed dimensions only need
    to be repeatable, not accurate;
  * a unique, camera-visible apex — the operator clicks the apex pixel, and
    depth unprojects it; a Ø1.5 spherical tip keeps that click within ±1-2 px.

Geometry (ThinkLabs One official envelope: Ø46 x 28 mm, flush bottom face —
see thinklab_openarm_veo6_realshape_mount.py item 9):
  * friction sleeve ID 46.30 (print fit), wall 3 mm, engagement 12 mm, with
    three 2.5 mm relief slits so the sleeve springs onto the puck;
  * 3 mm solid base disc under the diaphragm (also protects it);
  * 60-degree cone, 12 mm tall, apex finished as an r0.75 sphere;
  * apex sits ~15 mm below the diaphragm plane on the device axis (recorded in
    the audit for the runbook; the solver re-estimates it regardless).

Print: apex UP (sleeve opening down on the bed needs no roof bridging when
printed this way — the disc becomes the roof of the sleeve tube; use standard
internal supports inside the sleeve bore and remove them; cosmetics inside the
bore are irrelevant). PETG or PLA, 0.2 mm layers, 4 walls.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cadquery as cq

# ── ThinkLabs One official envelope ─────────────────────────────────────────
PUCK_DIAMETER_MM = 46.0
SLEEVE_FIT_CLEARANCE_MM = 0.30       # diametral print fit
SLEEVE_ID_MM = PUCK_DIAMETER_MM + SLEEVE_FIT_CLEARANCE_MM
SLEEVE_WALL_MM = 3.0
SLEEVE_OD_MM = SLEEVE_ID_MM + 2.0 * SLEEVE_WALL_MM
SLEEVE_ENGAGEMENT_MM = 12.0          # grip height on the puck's lower cylinder
SLIT_COUNT = 3
SLIT_WIDTH_MM = 2.5
SLIT_HEIGHT_MM = 10.0                # from the rim, leaves a solid collar band

BASE_DISC_THICKNESS_MM = 3.0

CONE_HEIGHT_MM = 12.0
CONE_INCLUDED_ANGLE_DEG = 60.0
TIP_SPHERE_RADIUS_MM = 0.75

# z=0 is the diaphragm plane (puck bottom face); +z into the puck.
# The base disc spans z in [-BASE_DISC_THICKNESS_MM, 0]; the cone hangs below.
APEX_Z_MM = -(BASE_DISC_THICKNESS_MM + CONE_HEIGHT_MM)


def build() -> cq.Workplane:
    cone_base_radius = CONE_HEIGHT_MM * math.tan(
        math.radians(CONE_INCLUDED_ANGLE_DEG / 2.0)
    )

    sleeve = (
        cq.Workplane("XY")
        .circle(SLEEVE_OD_MM / 2.0)
        .circle(SLEEVE_ID_MM / 2.0)
        .extrude(SLEEVE_ENGAGEMENT_MM)
    )
    # relief slits from the rim downward (rim is at +SLEEVE_ENGAGEMENT_MM)
    for index in range(SLIT_COUNT):
        angle = math.radians(index * 360.0 / SLIT_COUNT)
        direction = (math.cos(angle), math.sin(angle))
        slit = (
            cq.Workplane("XY")
            .center(
                direction[0] * (SLEEVE_ID_MM / 2.0 + SLEEVE_WALL_MM / 2.0),
                direction[1] * (SLEEVE_ID_MM / 2.0 + SLEEVE_WALL_MM / 2.0),
            )
            .rect(SLIT_WIDTH_MM, SLEEVE_OD_MM)  # oversized; cut is local
            .extrude(SLIT_HEIGHT_MM)
            .translate((0, 0, SLEEVE_ENGAGEMENT_MM - SLIT_HEIGHT_MM))
            .rotate((0, 0, 0), (0, 0, 1), math.degrees(angle))
        )
        sleeve = sleeve.cut(slit)

    base = (
        cq.Workplane("XY")
        .circle(SLEEVE_OD_MM / 2.0)
        .extrude(-BASE_DISC_THICKNESS_MM)
    )

    cone = (
        cq.Workplane("XY")
        .workplane(offset=-BASE_DISC_THICKNESS_MM)
        .circle(cone_base_radius)
        .workplane(offset=-CONE_HEIGHT_MM + TIP_SPHERE_RADIUS_MM)
        .circle(TIP_SPHERE_RADIUS_MM)
        .loft()
    )
    tip = (
        cq.Workplane("XY")
        .sphere(TIP_SPHERE_RADIUS_MM)
        .translate((0, 0, APEX_Z_MM + TIP_SPHERE_RADIUS_MM))
    )
    return sleeve.union(base).union(cone).union(tip)


def audit(part: cq.Workplane) -> dict:
    solid = part.val()
    bb = solid.BoundingBox()
    checks = {
        "single_solid": {
            "observed": len(part.solids().vals()),
            "expected": 1,
            "pass": len(part.solids().vals()) == 1,
        },
        "overall_height_mm": {
            "observed": round(bb.zmax - bb.zmin, 3),
            "expected": SLEEVE_ENGAGEMENT_MM
            + BASE_DISC_THICKNESS_MM
            + CONE_HEIGHT_MM,
            "pass": abs(
                (bb.zmax - bb.zmin)
                - (
                    SLEEVE_ENGAGEMENT_MM
                    + BASE_DISC_THICKNESS_MM
                    + CONE_HEIGHT_MM
                )
            )
            <= 0.05,
        },
        "apex_z_mm": {
            "observed": round(bb.zmin, 3),
            "expected": APEX_Z_MM,
            "pass": abs(bb.zmin - APEX_Z_MM) <= 0.05,
        },
        "apex_on_axis": {
            "observed": [round(bb.xmin + bb.xmax, 3), round(bb.ymin + bb.ymax, 3)],
            "expected": [0.0, 0.0],
            "pass": abs(bb.xmin + bb.xmax) <= 0.1 and abs(bb.ymin + bb.ymax) <= 0.1,
        },
        "sleeve_od_mm": {
            "observed": round(bb.xmax - bb.xmin, 3),
            "expected": SLEEVE_OD_MM,
            "pass": abs((bb.xmax - bb.xmin) - SLEEVE_OD_MM) <= 0.05,
        },
    }
    return {
        "part": "handeye_calibration_tip_cap_v1",
        "puck": {"diameter_mm": PUCK_DIAMETER_MM, "fit_clearance_mm": SLEEVE_FIT_CLEARANCE_MM},
        "tip": {
            "apex_below_diaphragm_mm": -APEX_Z_MM,
            "sphere_radius_mm": TIP_SPHERE_RADIUS_MM,
            "cone_included_angle_deg": CONE_INCLUDED_ANGLE_DEG,
            "note": "tip_offset_tcp_m is re-estimated by the calibration solver; "
                    "these numbers are informational (runbook / sanity).",
        },
        "print": {
            "orientation": "apex up; internal supports inside the sleeve bore",
            "material": "PETG or PLA, 0.2 mm layers, 4 walls",
        },
        "checks": checks,
        "all_pass": all(item["pass"] for item in checks.values()),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "generated" / "handeye_tip_cap"
    out_dir.mkdir(parents=True, exist_ok=True)
    part = build()
    report = audit(part)
    cq.exporters.export(part, str(out_dir / "handeye_calibration_tip_cap_v1.stl"))
    cq.exporters.export(part, str(out_dir / "handeye_calibration_tip_cap_v1.step"))
    (out_dir / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["checks"], indent=2))
    print("all_pass:", report["all_pass"])
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
