"""Run a continuous, audited expert sequence in Isaac Sim, never hardware.

Example (Linux): python -m sim.auscultation_training.run_sequence --points R1 L1
Every requested plan and source fingerprint is validated before motion. The
default runs all ten points in one environment and returns home between them.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from .controller import ExpertController, create_snapshot, digest, timing_variation, validate_plan, verify_snapshot
from .kinematics import DEFAULT_URDF_PATH, OpenArmKinematics
from .scene import ROOT, physical_sources

DEFAULT_POINTS = [side + str(level) for level in range(1, 6) for side in ("R", "L")]


def source_paths(root):
    """Local sources used to build physics; flattened runtime stage is also saved."""
    paths = {"code/" + p.name: p for p in root.glob("*.py")}
    paths["urdf"] = root / "openarm_v10_follower_stetho.urdf"
    _, physical = physical_sources()
    paths.update({'scene/' + name: path for name, path in physical.items()})
    return paths


def main(argv=None, *, controller_class=ExpertController, result_auditor=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", nargs="+", default=DEFAULT_POINTS)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario-id", default="rigid_seated_rgbd_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-scale", type=float, default=1.,
                        help="motion timing scale in [0.7, 1.0]; force criteria and 3-second hold stay fixed")
    args = parser.parse_args(argv)
    try:
        variation = timing_variation(args.speed_scale)
    except ValueError as error:
        parser.error(str(error))
    if len(set(args.points)) != len(args.points) or any(name not in DEFAULT_POINTS for name in args.points):
        parser.error("points must be distinct names R1/L1 through R5/L5")
    # This stage has no Isaac import, GPU use, or movement on validation failure.
    bundles = [validate_plan(args.input_dir / ("planned_paths_" + point + ".json"),
                             args.input_dir, args.runtime_dir) for point in args.points]
    if any(bundle.target.name != point for bundle, point in zip(bundles, args.points)):
        raise ValueError('plan point does not match requested point')
    bases = [np.asarray(bundle.plan["world_from_left_base"]) for bundle in bundles]
    if any(not np.allclose(base, bases[0], atol=1e-9, rtol=0) for base in bases[1:]):
        raise ValueError("sequence plans disagree on robot base")
    common_labels = ("targets", "runtime", "collision_metadata", "collision_arrays", "kinematics_urdf")
    for label in common_labels:
        if len({digest(bundle.inputs[label][1]) for bundle in bundles}) != 1:
            raise ValueError("sequence plans use inconsistent " + label)
    if digest(DEFAULT_URDF_PATH.read_bytes()) != digest(bundles[0].inputs["kinematics_urdf"][1]):
        raise ValueError("runtime kinematics URDF differs from pinned plan")
    from .revalidate_plans import validate_geometry_binding
    binding_sha = validate_geometry_binding(ROOT)
    for bundle in bundles:
        record = bundle.plan.get('revalidation', {})
        if record.get('accepted') is not True or record.get('collision_regeneration_sha256') != binding_sha:
            raise ValueError('Plan has no current geometry revalidation: '+bundle.target.name)
    output = args.output or ROOT / "results" / ("sequence_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = source_paths(ROOT)
    sources['collision_regeneration'] = ROOT/'results/collision_regeneration_verification.json'
    snapshot = create_snapshot(bundles, output / "input_snapshot", sources)
    snapshot_sha = digest((snapshot / "manifest.json").read_bytes())
    verify_snapshot(snapshot, verify_sources=True)
    # FCL is installed only in this isolated dependency directory on Linux.
    dependency_dir = ROOT / "python_deps"
    if dependency_dir.is_dir():
        sys.path.insert(0, str(dependency_dir))
    from .collision import CollisionChecker
    kin = OpenArmKinematics()
    # The checker consumes the pinned bytes, not subsequently mutable exports.
    with tempfile.TemporaryDirectory(prefix="ausc_checked_geometry_") as geometry_dir:
        geometry_dir = Path(geometry_dir)
        for label, filename in (("collision_metadata", "collision_geometry.json"),
                                ("collision_arrays", "collision_geometry.npz")):
            (geometry_dir / filename).write_bytes(bundles[0].inputs[label][1])
        checker = CollisionChecker(kin, bases[0], directory=geometry_dir)
    app = None
    env = None
    controller = None
    outcome = {"success": False, "training_accepted": False, "points": [],
               "scope": "simulation contact mechanics only; no acoustic or clinical success"}
    try:
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True})
        from .env import IsaacContactEnv
        env = IsaacContactEnv(output)
        verify_snapshot(snapshot, verify_sources=True)
        if not np.allclose(env.base, bases[0], atol=1e-7, rtol=0):
            raise ValueError("live simulation base differs from pinned plan")
        controller = controller_class(env, kin, checker, speed_scale=args.speed_scale)
        outcome = controller.run([(bundle.target, bundle.row) for bundle in bundles])
        verify_snapshot(snapshot, verify_sources=True)
        if validate_geometry_binding(ROOT) != binding_sha:
            raise ValueError('Collision source binding changed during execution')
    except Exception as error:
        outcome.update(success=False, training_accepted=False, failure_reason=str(error))
        if controller is not None:
            for row in controller.control_rows:
                row["training_accepted"] = False
        if env is not None:
            for row in env.rows:
                row["training_accepted"] = False
        for point in outcome.get("points", []):
            point["training_accepted"] = False
    finally:
        outcome.update(requested_points=args.points, input_snapshot="input_snapshot/manifest.json",
                       scenario_id=args.scenario_id, seed=args.seed,
                       seed_semantics="metadata only; changing seed alone does not perturb this deterministic expert",
                       speed_scale=args.speed_scale, variation=variation,
                       input_snapshot_manifest_sha256=snapshot_sha,
                       source_binding="regenerated collider arrays and metadata match; source hashes verified before and after execution",
                       collision_regeneration_sha256=binding_sha)
        if env is not None:
            env.save("expert")
            outcome["training_scene_sha256"] = digest((output / "training_scene.usdc").read_bytes())
        if controller is not None:
            (output / "control_commands.json").write_text(json.dumps(controller.control_rows) + "\n")
        (output / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
        if result_auditor is not None:
            result_auditor(output)
        if app is not None:
            app.close()
    print("SEQUENCE", "ACCEPTED" if outcome["training_accepted"] else "REJECTED", str(output), flush=True)
    return 0 if outcome["training_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
