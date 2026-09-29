#!/usr/bin/env python3
"""Export current VEO6 CAD into independent, metre-authored triangle meshes.

Run with CadQuery installed; no CAD main(), render, audit or legacy output
pipeline is invoked. The original CAD files are read only. The manifest is the
interface for USD creation: vertices already share the CAD assembly frame and
must not be scaled or individually recentered by the importer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parents[1] / "auscultation/mechanical_sensor/cad/thinklab_openarm_veo6_realshape_mount.py"
MM_TO_M = 0.001


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest_atomically(path: Path, manifest: dict) -> None:
    """Publish a complete validated manifest, never a partially written JSON."""
    contents = json.dumps(manifest, indent=2) + "\n"
    pending = None
    try:
        # Same directory keeps os.replace on one filesystem.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                delete=False) as stream:
            pending = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def load_cad(source: Path):
    spec = importlib.util.spec_from_file_location("real2sim_veo6_mount", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CAD: {source}")
    cad = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cad
    spec.loader.exec_module(cad)
    cad.configure_backends()
    return cad


def shape_of(part):
    return part.val() if hasattr(part, "val") else part


def bbox(shape, scale=MM_TO_M):
    # CadQuery's normal bounds can expand by cached tessellation deflection.
    # Exact analytic bounds are needed to check 46/88 mm dimensions reliably.
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape_of(shape).wrapped, b, False, False)
    bounds = b.Get()
    lo = np.array(bounds[:3]) * scale
    hi = np.array(bounds[3:]) * scale
    return {"min": lo.tolist(), "max": hi.tolist(), "size": (hi - lo).tolist()}


def slice_z(cad, shape, z0, z1):
    return shape.intersect(cad.v9.box(250, 250, z1-z0, (0, 0, (z0+z1)/2)))


def components(cad):
    """Build only the requested parts, deliberately skipping helix construction."""
    p = cad.P
    built = []

    def add(name, shape, group, material, color, density, *, mass=None,
            install=True, note="", contact="hardware"):
        built.append({"name": name, "shape": shape_of(shape), "group": group,
                      "material": material, "display_color_srgb": color,
                      "density_kg_m3_estimate": density, "mass_override_kg": mass,
                      "install_default": install, "note": note,
                      "contact_category": contact})

    printed = ("printed_petg", [0.025, 0.028, 0.030], 1270 * 0.55)
    steel = ("steel", [0.62, 0.66, 0.70], 7850)
    black_steel = ("black_steel", [0.025, 0.028, 0.031], 7850)
    print_note = "Mass estimate uses PETG density 1270 kg/m3 times 0.55 effective fill; not weighed. Concave visual mesh; do not use a single convex hull."
    print("Building fixed bridge and clamshells", flush=True)
    fixed = cad.make_fixed_bridge(p)
    left, right = cad.make_clamshell_halves(p)
    add("fixed_bridge", fixed, "fixed", *printed, note=print_note, contact="fixed_frame")
    add("left_clamshell", left, "moving", *printed, note=print_note, contact="lip_clamp")
    add("right_clamshell", right, "moving", *printed, note=print_note, contact="lip_clamp")

    print("Building device and purchased parts", flush=True)
    device = cad.make_thinklab_proxy(p)
    zs = cad.thinklab_section_zs(p)
    device_volume = shape_of(device).Volume()
    for name, z0, z1, color, material in [
        ("thinklabs_black_edge", zs["dev_bottom"], zs["black_top"], [0.025, 0.028, 0.031], "device_black"),
        ("thinklabs_cone", zs["black_top"], zs["cone_top"], [0.040, 0.044, 0.048], "device_black"),
        ("thinklabs_dial", zs["cone_top"], zs["dial_top"], [0.63, 0.66, 0.70], "device_metallic_dial"),
    ]:
        region = slice_z(cad, device, z0, z1)
        fraction = shape_of(region).Volume() / device_volume
        add(name, region, "moving", material, color, None, mass=0.050 * fraction,
            note="Exact z-slice of CAD device; 50 g nominal total apportioned by CAD volume, inertia/distribution estimated.",
            contact="diaphragm" if name == "thinklabs_black_edge" else "device_body")

    rods = sorted(shape_of(cad.make_guide_rods_proxy(p)).Solids(), key=lambda s: s.Center().x)
    for name, part in zip(("guide_rod_left", "guide_rod_right"), rods, strict=True):
        add(name, part, "fixed", *steel, contact="post_head")
    screws = sorted(shape_of(cad.make_clamp_screws_proxy(p)).Solids(), key=lambda s: s.Center().y)
    for i, part in enumerate(screws):
        add(f"clamp_screw_{i}", part, "moving", *black_steel)
    bolts = sorted(shape_of(cad.make_horn_screws_proxy()).Solids(), key=lambda s: (s.Center().x, s.Center().y))
    for i, part in enumerate(bolts):
        add(f"interface_bolt_{i}", part, "fixed", *black_steel)
    add("robot_flange", cad.make_horn_proxy(), "fixed", *steel, install=False,
        note="CAD clearance envelope only (57 mm flange, 35 mm actuator body); not verified actual wrist geometry. Do not overlay on an existing robot wrist.")
    add("magnet", cad.make_magnet_proxy(), "moving", "nickel_magnet", [0.48, 0.50, 0.53], 7500)
    hall_body = cad.make_hall_body_proxy()
    add("hall_body", hall_body, "fixed", "epoxy", [0.020, 0.022, 0.024], 1200)
    add("hall_leads", cad.make_hall_proxy().cut(hall_body), "fixed", *steel)
    xiao = cad.make_xiao_proxy()
    board_top = cad.WING_TOP_Z_MM + 1.6
    add("xiao_board", slice_z(cad, xiao, cad.WING_TOP_Z_MM, board_top), "fixed",
        "pcb", [0.035, 0.28, 0.095], 1850)
    add("xiao_usb", slice_z(cad, xiao, board_top, board_top + 3.2), "fixed", *steel,
        note="Solid USB enclosure proxy; density overestimates the real hollow connector.")
    colors = {"hall_vcc_wire": [0.72, 0.025, 0.025],
              "hall_signal_wire": [0.88, 0.88, 0.84],
              "hall_gnd_wire": [0.015, 0.015, 0.018]}
    for name, part in cad.make_wire_proxies().items():
        add(name, part, "fixed", "silicone_wire", colors[name], 2200,
            note="Rigid routing visualization; effective density estimate combines conductor and insulation.")
    return built, {"fixed": fixed, "left": left, "right": right, "device": device}


def export_one(item, out, tolerance, angular_tolerance):
    shape = item["shape"]
    if not shape.isValid() or shape.Volume() <= 0:
        raise ValueError(f"Invalid CAD solid: {item['name']}")
    verts, triangles = shape.tessellate(tolerance, angular_tolerance)
    vertices = np.array([v.toTuple() for v in verts], dtype=np.float64) * MM_TO_M
    faces = np.array(triangles, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Malformed mesh: {item['name']}")
    if not len(faces) or not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Invalid mesh indices/coordinates: {item['name']}")
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    area2 = np.linalg.norm(np.cross(b-a, c-a), axis=1)
    # Remove only numerical zero-area triangles, retaining small lead features.
    faces = faces[area2 > 1e-18]
    rel = Path("components") / item["name"]
    npz = out / rel.with_suffix(".npz")
    obj = out / rel.with_suffix(".obj")
    np.savez_compressed(npz, vertices=vertices, faces=faces)
    with obj.open("w") as stream:
        stream.write("# Coordinates in metres; common CAD assembly frame; +Z compression\n")
        stream.write(f"o {item['name']}\n")
        for v in vertices:
            stream.write("v %.10g %.10g %.10g\n" % tuple(v))
        for f in faces + 1:
            stream.write("f %d %d %d\n" % tuple(f))
    with np.load(npz, allow_pickle=False) as decoded:
        assert np.array_equal(decoded["vertices"], vertices)
        assert np.array_equal(decoded["faces"], faces)
    volume_m3 = shape.Volume() * 1e-9
    mass = item["mass_override_kg"]
    if mass is None:
        mass = volume_m3 * item["density_kg_m3_estimate"]
    center = shape.Center()
    # OCCT inertia is about the CAD centre of mass at unit density (mm^5).
    # Scale by kg/mm^3 and then mm^2 -> m^2 for a uniform-density estimate.
    inertia = np.asarray(shape.matrixOfInertia(shape)) * (mass / shape.Volume()) * 1e-6
    record = {k: v for k, v in item.items() if k not in ("shape", "mass_override_kg")}
    record.update({"npz": rel.with_suffix(".npz").as_posix(),
                   "obj": rel.with_suffix(".obj").as_posix(),
                   "vertices": len(vertices), "triangles": len(faces),
                   "bbox_m": bbox(shape), "volume_m3": volume_m3,
                   "solid_count": len(shape.Solids()), "mass_kg_estimate": mass,
                   "center_of_mass_m_estimate": [center.x*.001, center.y*.001, center.z*.001],
                   "inertia_about_com_kg_m2_estimate": inertia.tolist(),
                   "mesh_sha256": sha256(npz), "obj_sha256": sha256(obj),
                   "decoded_ok": True, "cad_valid": True})
    mesh_error = np.max(np.abs(np.array([vertices.min(axis=0), vertices.max(axis=0)]) -
                               np.array([record["bbox_m"]["min"], record["bbox_m"]["max"]])))
    record["tessellation_bbox_max_error_m"] = float(mesh_error)
    if mesh_error > tolerance * .001 * 2:
        raise ValueError(f"Mesh boundary differs from CAD by {mesh_error} m: {item['name']}")
    print(f"  {item['name']}: {len(faces)} triangles", flush=True)
    return record


def dimensions_and_validation(cad, geometry, records):
    p, z = cad.P, cad.thinklab_section_zs(cad.P)
    lip_bottom = p.carriage_z - p.cup_height/2
    device_bbox = bbox(geometry["device"], scale=1)
    fixed_bbox = bbox(geometry["fixed"], scale=1)
    bushing_top = p.carriage_z + 10 + p.guide_bushing_h/2
    checks = []

    def check(name, actual, expected, tolerance=1e-5):
        checks.append({"name": name, "actual": actual, "expected": expected,
                       "tolerance": tolerance, "pass": bool(np.max(np.abs(np.asarray(actual)-expected)) <= tolerance)})

    check("device_diameter_x_mm", device_bbox["size"][0], 46.)
    check("device_diameter_y_mm", device_bbox["size"][1], 46.)
    check("device_height_mm", device_bbox["size"][2], 29.)
    check("fixed_plate_diameter_x_mm", fixed_bbox["size"][0], 88.)
    check("retention_recess_mm", z["dev_bottom"]-lip_bottom, .8)
    check("mount_hole_centers_mm", sorted(cad.OUTPUT_M3_POSITIONS), sorted([(-10.5,-7),(-10.5,7),(10.5,-7),(10.5,7)]))
    check("clamshell_gap_mm", bbox(geometry["right"], 1)["min"][0]-bbox(geometry["left"], 1)["max"][0], 3.)
    # Positive-length centre probes verify holes exist through all plate/boss material.
    hole_volumes = []
    for x, y in cad.OUTPUT_M3_POSITIONS:
        probe = cad.v9.cyl_z(p.m3_clearance_d/2-.05, 8, (x, y, 0))
        hole_volumes.append(shape_of(geometry["fixed"].intersect(probe)).Volume())
    check("mount_holes_are_open_intersection_mm3", hole_volumes, [0.,0.,0.,0.], 1e-6)
    opening_probe = cad.v9.cyl_z(cad.BOTTOM_LIP_D_MM/2-.05, cad.BOTTOM_LIP_H_MM+.2,
                               (0, 0, lip_bottom+cad.BOTTOM_LIP_H_MM/2))
    lip_vol = sum(shape_of(geometry[key].intersect(opening_probe)).Volume() for key in ("left", "right"))
    check("central_retention_opening_is_clear_mm3", lip_vol, 0., 1e-6)
    device_total_volume = sum(r["volume_m3"] for r in records if r["name"].startswith("thinklabs_"))
    check("device_material_slices_preserve_volume_m3", device_total_volume, shape_of(geometry["device"]).Volume()*1e-9, 1e-11)
    if not all(c["pass"] for c in checks):
        raise ValueError(f"Dimension checks failed: {[c for c in checks if not c['pass']]}")
    return {
        "device_profile_mm": cad.VEO6_THINKLABS_PROFILE,
        "device_bbox_mm": device_bbox, "fixed_bridge_bbox_mm": fixed_bbox,
        "plate_nominal_diameter_mm": 88., "plate_thickness_mm": p.top_bridge_t,
        "mount_hole_centers_mm": cad.OUTPUT_M3_POSITIONS,
        "mount_clearance_hole_diameter_mm": p.m3_clearance_d,
        "mount_counterbore_diameter_mm": cad.OUTPUT_M3_CBORE_D_MM,
        "mount_counterbore_depth_mm": cad.OUTPUT_M3_CBORE_DEPTH_MM,
        "cad_z_surfaces_mm": {**z, "lip_bottom": lip_bottom,
            "adapter_mating_plane": p.top_bridge_t/2,
            "washer_boss_bottom": -p.top_bridge_t/2-cad.OUTPUT_M3_BOSS_H_MM,
            "guide_post_tip": cad.GUIDE_ROD_TIP_Z_MM,
            "guide_post_head_top": cad.ROD_SHAFT_BOTTOM_Z_MM,
            "barrel_bottom": cad.BARREL_BOTTOM_REST_Z_MM,
            "barrel_top": bushing_top, "up_stop_bottom": bushing_top+cad.HARD_STOP_TRAVEL_MM,
            "magnet_top": cad.MAGNET_TOP_Z_MM, "hall_face": cad.HALL_FACE_Z_MM,
            "spring_moving_seat": p.spring_moving_face_z, "spring_fixed_seat": p.spring_fixed_face_z},
        "diaphragm_recess_mm": z["dev_bottom"]-lip_bottom,
        "retention_opening_diameter_mm": cad.BOTTOM_LIP_D_MM,
        "guide_post_xy_mm": [[-p.guide_rod_x,p.guide_rod_y],[p.guide_rod_x,p.guide_rod_y]],
        "guide_post_diameter_mm": p.guide_rod_d,
        "guide_post_head_diameter_mm": cad.ROD_HEAD_D_MM,
        "guide_post_head_height_mm": cad.ROD_HEAD_H_MM,
        "rod_head_design_seat_gap_mm": cad.BARREL_BOTTOM_REST_Z_MM-cad.ROD_SHAFT_BOTTOM_Z_MM,
        "rod_tip_clearance_above_diaphragm_at_rest_mm": cad.GUIDE_ROD_TIP_Z_MM-z["dev_bottom"],
        "rod_tip_clearance_above_diaphragm_at_max_travel_mm": cad.GUIDE_ROD_TIP_Z_MM-(z["dev_bottom"]+cad.HARD_STOP_TRAVEL_MM),
        "validation": {"pass": True, "checks": checks}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=HERE/"assets")
    parser.add_argument("--tolerance-mm", type=float, default=.06)
    parser.add_argument("--angular-tolerance-rad", type=float, default=.10)
    args = parser.parse_args()
    if args.tolerance_mm <= 0 or args.angular_tolerance_rad <= 0:
        parser.error("Tessellation tolerances must be positive")
    start = time.monotonic()
    source, out = args.source.resolve(), args.out.resolve()
    sys.dont_write_bytecode = True
    # Freeze bytes before import. Check again after export to detect concurrent edits.
    source_bytes = {p: p.read_bytes() for p in source.parent.glob("*.py")}
    cad = load_cad(source)
    imported_sources = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values())
        if getattr(m, "__file__", None) and Path(m.__file__).resolve().parent == source.parent})
    # Invalidate prior success before changing any published assets. A failed
    # re-export may leave partial meshes, but no manifest may certify them.
    # The independent validation report also describes the previous meshes.
    for name in ("component_manifest.json", "export_validation.json"):
        (out/name).unlink(missing_ok=True)
    (out/"components").mkdir(parents=True, exist_ok=True)
    (out/"source_snapshot").mkdir(parents=True, exist_ok=True)
    source_records = []
    for path in imported_sources:
        raw = source_bytes[path]
        snapshot = out/"source_snapshot"/path.name
        snapshot.write_bytes(raw)
        source_records.append({"path": str(path), "snapshot": str(snapshot.relative_to(out)),
                               "sha256": hashlib.sha256(raw).hexdigest()})
    built, geometry = components(cad)
    records = [export_one(c, out, args.tolerance_mm, args.angular_tolerance_rad) for c in built]
    dims = dimensions_and_validation(cad, geometry, records)
    for path in imported_sources:
        if path.read_bytes() != source_bytes[path]:
            raise RuntimeError(f"Source changed during export: {path}")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_convention": {"units": "metres", "cad_to_mesh_scale": .001,
            "common_origin": "centre of the 88 mm circular fixed plate at its mid-thickness",
            "compression_axis": [0,0,1], "contact_direction": [0,0,-1],
            "moving_component_rest_offset_m": [0,0,0],
            "mount_origin_cad_m": [0,0,cad.P.top_bridge_t*.0005],
            "transform_rule": "p_world = mount_world + R @ (p_mesh - mount_origin_cad_m + group_compression_m); group_compression_m=(0,0,q) for moving, zero for fixed",
            "warning": "Vertices are already metres in the common assembly frame; do not recenter, individually rotate, or apply a second 0.001 scale."},
        "mechanics_nominal": {"preload_n": cad.PRELOAD_N,
            "stiffness_n_per_m": cad.RECOMMENDED_MECHANICS["spring_rate_n_per_mm"]*1000,
            "travel_m": [0,cad.HARD_STOP_TRAVEL_MM*.001],
            "prismatic_drive_target_m": -cad.PRELOAD_DEFLECTION_MM*.001,
            "spring_force_at_stop_n": cad.PRELOAD_N+cad.RECOMMENDED_MECHANICS["spring_rate_n_per_mm"]*cad.HARD_STOP_TRAVEL_MM,
            "note": "CAD design values; not bench calibration. A travel hard stop does not limit external contact force."},
        "springs": {"group": "spring", "geometry_exported": False,
            "reason": "Scene builder generates deforming helix visuals; no expensive legacy union exported.",
            "axes_xy_m": (np.asarray(cad.SPRING_POSITIONS)*.001).tolist(),
            "outer_diameter_m": cad.P.spring_od*.001,
            "wire_radius_m": cad.SPRING_WIRE_R_MM*.001,
            "turns": cad.SPRING_VISUAL_TURNS,
            "moving_seat_z_m": cad.P.spring_moving_face_z*.001,
            "fixed_seat_z_m": cad.P.spring_fixed_face_z*.001},
        "dimensions": dims,
        "mass_assumptions": "All masses estimated. Printed volume uses 55% PETG effective fill; nominal 50 g Thinklabs split by region volume; metal parts use solid steel or nickel proxy density. Electronics/wires approximate; flange envelope excluded.",
        "mass_totals_kg_estimate": {group: sum(r["mass_kg_estimate"] for r in records if r["group"] == group and r["install_default"]) for group in ("fixed", "moving")},
        "collision_guidance": "Meshes are visuals and CAD solids, not validated convex decomposition. Keep ring aperture open using compound convex shapes; never one whole-ring convex hull. Include fixed post heads. All internal fixed/moving non-contact pairs require deliberate filtering.",
        "components": records, "source_files": source_records,
        "exporter_sha256": sha256(Path(__file__)),
        "runtime": {"python": platform.python_version(), "cadquery": importlib.metadata.version("cadquery"),
                    "numpy": np.__version__, "tolerance_mm": args.tolerance_mm,
                    "angular_tolerance_rad": args.angular_tolerance_rad},
        "validation": {"pass": True, "all_meshes_decoded": True,
                       "source_bytes_unchanged": True, "component_count": len(records)},
        "elapsed_seconds": round(time.monotonic()-start, 3)}
    write_manifest_atomically(out/"component_manifest.json", manifest)
    print(json.dumps({"manifest": str(out/"component_manifest.json"),
        "components": len(records), "triangles": sum(r["triangles"] for r in records),
        "mass_totals_kg_estimate": manifest["mass_totals_kg_estimate"],
        "elapsed_seconds": manifest["elapsed_seconds"], "validation": "passed"}, indent=2))


if __name__ == "__main__":
    main()
