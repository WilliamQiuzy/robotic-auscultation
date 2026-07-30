"""Author an Isaac/Omniverse USD physics handoff scene for the v6 mount.

The generated USD is a physics proxy, not a mesh conversion of the printable
STEP files. Isaac should use these simple colliders for stable contact tests
and keep the STEP/STL files as CAD references.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[3]
GEN_DIR = ROOT / "auscultation" / "mechanical_sensor" / "generated"
ACCEPT_DIR = GEN_DIR / "acceptance"
MUJOCO_DIR = GEN_DIR / "mujoco"
OUT_DIR = GEN_DIR / "isaac_usd"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def add_translate(prim, xyz: tuple[float, float, float]) -> None:
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*xyz))


def add_scale(prim, xyz: tuple[float, float, float]) -> None:
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*xyz))


def set_display_color(prim, rgb: tuple[float, float, float]) -> None:
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def create_cube(
    stage: Usd.Stage,
    path: str,
    center_m: tuple[float, float, float],
    size_m: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = True,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    add_translate(prim, center_m)
    add_scale(prim, size_m)
    set_display_color(prim, color)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    return prim


def create_cylinder(
    stage: Usd.Stage,
    path: str,
    center_m: tuple[float, float, float],
    radius_m: float,
    height_m: float,
    color: tuple[float, float, float],
    collision: bool = True,
) -> Usd.Prim:
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(radius_m)
    cyl.CreateHeightAttr(height_m)
    prim = cyl.GetPrim()
    add_translate(prim, center_m)
    set_display_color(prim, color)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    return prim


def create_capsule(
    stage: Usd.Stage,
    path: str,
    center_m: tuple[float, float, float],
    radius_m: float,
    height_m: float,
    color: tuple[float, float, float],
) -> Usd.Prim:
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.CreateAxisAttr("X")
    capsule.CreateRadiusAttr(radius_m)
    capsule.CreateHeightAttr(height_m)
    prim = capsule.GetPrim()
    add_translate(prim, center_m)
    set_display_color(prim, color)
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    return prim


def custom_attr(prim: Usd.Prim, name: str, value, type_name) -> None:
    prim.CreateAttribute(name, type_name).Set(value)


def author_usd(meta: dict, gates: dict, mujoco_summary: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    usd_path = OUT_DIR / "thinklabs_v6_passive_mount_physics.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(240)

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(0.0)

    material_prim = stage.DefinePrim("/World/Materials/SoftTissueContact", "Material")
    mat_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    mat_api.CreateStaticFrictionAttr(0.8)
    mat_api.CreateDynamicFrictionAttr(0.55)
    mat_api.CreateRestitutionAttr(0.0)

    tissue = create_cube(stage, "/World/TissuePad", (0.0, 0.0, -0.003), (0.16, 0.16, 0.006), (0.78, 0.50, 0.46))
    rib = create_capsule(stage, "/World/RibBumpProxy", (0.0, 0.0, 0.0045), 0.0055, 0.10, (0.55, 0.34, 0.30))

    fixed = UsdGeom.Xform.Define(stage, "/World/FixedFrame").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(fixed).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI(fixed).CreateKinematicEnabledAttr(True)
    create_cube(stage, "/World/FixedFrame/RobotPlate", (0.0, 0.0, 0.080), (0.108, 0.070, 0.008), (0.83, 0.80, 0.70), collision=False)
    create_cylinder(stage, "/World/FixedFrame/LeftGuideRod", (-0.042, 0.016, 0.035), 0.003, 0.090, (0.55, 0.55, 0.58), collision=False)
    create_cylinder(stage, "/World/FixedFrame/RightGuideRod", (0.042, 0.016, 0.035), 0.003, 0.090, (0.55, 0.55, 0.58), collision=False)

    carrier = UsdGeom.Xform.Define(stage, "/World/FloatingCarrier").GetPrim()
    add_translate(carrier, (0.0, 0.0, 0.030))
    rb = UsdPhysics.RigidBodyAPI.Apply(carrier)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(carrier).CreateMassAttr(0.32)
    create_cube(stage, "/World/FloatingCarrier/CradleProxy", (0.0, 0.0, 0.022), (0.108, 0.085, 0.014), (0.05, 0.34, 0.75), collision=False)
    create_cylinder(stage, "/World/FloatingCarrier/ThinkLabsContactDisk", (0.0, 0.0, 0.0), 0.026, 0.0025, (0.05, 0.05, 0.05), collision=True)
    create_cylinder(stage, "/World/FloatingCarrier/SpringEnvelope", (0.0, 0.0, 0.030), 0.008, 0.042, (0.20, 0.80, 0.25), collision=False)

    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/PassiveAxisJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/FixedFrame")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/FloatingCarrier")])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.030))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLowerLimitAttr(-float(meta["params"]["travel"]) / 1000.0)
    joint.CreateUpperLimitAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(gates["spring_rate_n_per_mm"]) * 1000.0)
    drive.CreateDampingAttr(float(gates["linear_damping_n_s_per_m"]))
    drive.CreateTargetPositionAttr(0.0)

    root = stage.GetDefaultPrim()
    custom_attr(root, "wam:selectedVariant", meta["recommended_mechanics"]["simulation_variant"], Sdf.ValueTypeNames.String)
    custom_attr(root, "wam:targetContactForceN", float(gates["target_contact_force_n"]), Sdf.ValueTypeNames.Float)
    custom_attr(root, "wam:maxSafeForceN", float(gates["max_safe_force_n"]), Sdf.ValueTypeNames.Float)
    custom_attr(root, "wam:targetSpringCompressionMm", float(gates["target_spring_compression_mm"]), Sdf.ValueTypeNames.Float)
    custom_attr(root, "wam:contactFaceProjectionMm", float(meta["params"]["contact_face_projection"]), Sdf.ValueTypeNames.Float)
    custom_attr(root, "wam:mujocoMaxPeakForceN", float(mujoco_summary["max_peak_contact_force_n"]), Sdf.ValueTypeNames.Float)
    custom_attr(root, "wam:mujocoUnsafeForceRate", float(mujoco_summary["unsafe_force_rate"]), Sdf.ValueTypeNames.Float)

    cad_refs = stage.DefinePrim("/World/CADReferences", "Scope")
    custom_attr(cad_refs, "wam:fixedFrameStep", rel(GEN_DIR / "thinklab_openarm_fixed_frame_print.step"), Sdf.ValueTypeNames.String)
    custom_attr(cad_refs, "wam:floatingCarrierStep", rel(GEN_DIR / "thinklab_openarm_floating_carriage_print.step"), Sdf.ValueTypeNames.String)
    custom_attr(cad_refs, "wam:assemblyStep", rel(GEN_DIR / "thinklab_openarm_passive_force_mount_assembly.step"), Sdf.ValueTypeNames.String)
    custom_attr(cad_refs, "wam:mujocoReport", rel(MUJOCO_DIR / "MUJOCO_REPORT.md"), Sdf.ValueTypeNames.String)

    stage.GetRootLayer().Save()
    return usd_path


def render_usd_handoff_preview(meta: dict, gates: dict, mujoco_summary: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "thinklabs_v6_usd_handoff_preview.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), dpi=170)
    side = axes[0]
    side.add_patch(Rectangle((-54, 74), 108, 8, facecolor="#d4cdb9", edgecolor="#3b3b3b", linewidth=0.8))
    side.add_patch(Rectangle((-54, 10), 108, 14, facecolor="#0d57a1", edgecolor="#1f1f1f", linewidth=0.8, alpha=0.9))
    side.add_patch(Circle((0, 0), 26, facecolor="#1f2225", edgecolor="#0f0f0f", alpha=0.95))
    side.add_patch(Rectangle((-80, -9), 160, 6, facecolor="#c98075", edgecolor="#6d413d", linewidth=0.8, alpha=0.8))
    side.add_patch(Rectangle((-8, 26), 16, 42, facecolor="#46b565", edgecolor="#2e7440", alpha=0.65))
    for x in (-42, 42):
        side.add_patch(Rectangle((x - 3, 0), 6, 82, facecolor="#8e9298", edgecolor="#4f5257", alpha=0.75))
    side.set_title("USD physics proxy side view")
    side.set_aspect("equal")
    side.set_xlim(-90, 90)
    side.set_ylim(-24, 90)
    side.set_xlabel("x (mm)")
    side.set_ylabel("z (mm)")
    side.grid(True, alpha=0.2)

    ax = axes[1]
    labels = ["target", "MuJoCo p95", "MuJoCo max", "abort"]
    values = [
        float(gates["target_contact_force_n"]),
        float(mujoco_summary["p95_peak_contact_force_n"]),
        float(mujoco_summary["max_peak_contact_force_n"]),
        float(gates["max_safe_force_n"]),
    ]
    colors = ["#3f6fa7", "#648f54", "#b2794c", "#b83f37"]
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("force (N)")
    ax.set_title(f"{meta['recommended_mechanics']['simulation_variant']} evidence carried into USD")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def write_report(usd_path: Path, preview_path: Path, meta: dict, gates: dict, mujoco_summary: dict) -> Path:
    report_path = OUT_DIR / "ISAAC_USD_HANDOFF_REPORT.md"
    lines = [
        "# Isaac USD Handoff Package",
        "",
        "## Scope",
        "",
        "This package authors a simplified USD physics proxy for the selected ThinkLabs v6 passive mount. It is meant for Isaac/Omniverse import on the repo's Vast.ai Linux GPU environment.",
        "",
        "It is not a full SimReady validation pass and does not replace the pending Isaac Lab rigid-body/contact-sensor run.",
        "",
        "## Artifacts",
        "",
        f"- USD scene: `{rel(usd_path)}`",
        f"- Preview image: `{rel(preview_path)}`",
        "- Physics proxy: fixed tissue pad, optional rib bump proxy, kinematic fixed frame, dynamic sliding carrier, ThinkLabs contact disk collider, guide rods, spring envelope, and a prismatic passive-axis joint.",
        "",
        "## Candidate Metadata",
        "",
        f"- Variant: `{meta['recommended_mechanics']['simulation_variant']}`",
        f"- Contact-face projection: `{meta['params']['contact_face_projection']} mm`",
        f"- Spring: `{gates['spring_rate_n_per_mm']} N/mm`",
        f"- Damping: `{gates['linear_damping_n_s_per_m']} N s/m`",
        f"- Target compression at 4 N: `{gates['target_spring_compression_mm']} mm`",
        "",
        "## Carried MuJoCo Evidence",
        "",
        f"- Random sweep samples: `{mujoco_summary['sample_count']}`",
        f"- P95 / max peak force: `{mujoco_summary['p95_peak_contact_force_n']} N` / `{mujoco_summary['max_peak_contact_force_n']} N`",
        f"- Unsafe-force / travel-gate / contact-loss rates: `{100.0 * mujoco_summary['unsafe_force_rate']:.1f}%` / `{100.0 * mujoco_summary['travel_gate_violation_rate']:.1f}%` / `{100.0 * mujoco_summary['contact_loss_rate']:.1f}%`",
        "",
        "## Suggested Isaac Next Step",
        "",
        "Open `thinklabs_v6_passive_mount_physics.usda` in Isaac Sim, attach a contact sensor to `/World/FloatingCarrier/ThinkLabsContactDisk`, drive `/World/PassiveAxisJoint` with the same compression-gated controller, and compare peak force/travel/contact persistence against the MuJoCo gates.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((GEN_DIR / "design_metadata.json").read_text(encoding="utf-8"))
    gates = json.loads((ACCEPT_DIR / "acceptance_gates.json").read_text(encoding="utf-8"))
    mujoco_summary = json.loads((MUJOCO_DIR / "mujoco_random_sweep_summary.json").read_text(encoding="utf-8"))
    usd_path = author_usd(meta, gates, mujoco_summary)
    preview_path = render_usd_handoff_preview(meta, gates, mujoco_summary)
    report_path = write_report(usd_path, preview_path, meta, gates, mujoco_summary)
    payload = {
        "usd_scene": str(usd_path),
        "preview": str(preview_path),
        "report": str(report_path),
        "variant": meta["recommended_mechanics"]["simulation_variant"],
        "mujoco_random_sweep": mujoco_summary,
    }
    (OUT_DIR / "isaac_usd_handoff_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
