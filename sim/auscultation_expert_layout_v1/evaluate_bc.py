"""Evaluate frozen BC in closed-loop Isaac Sim; default smoke target is R1.

The model manifest SHA must be supplied explicitly. No optimizer, hardware
interface, demonstration playback, or fallback expert trajectory is used.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from .controller import create_snapshot, digest, timing_variation, validate_plan, verify_snapshot
from .kinematics import DEFAULT_URDF_PATH, OpenArmKinematics
from .run_sequence import DEFAULT_POINTS, source_paths
from .scene import ROOT
from .student import StudentController, load_actor


def finalize_evaluation(output, app, env, controller, outcome, audit_function=None):
    """Persist and audit while Isaac is alive; teardown is always the last step.

    Isaac teardown can unload Python/native services needed by the auditor.
    The outcome is written exactly once before its independent hash binding;
    evaluation_summary is a separate artifact and never invalidates that hash.
    """
    output = Path(output)
    try:
        if env is not None:
            env.save("expert")  # Shared trace schema; outcome identifies student evaluation.
            outcome["training_scene_sha256"] = digest((output / "training_scene.usdc").read_bytes())
        if controller is not None:
            (output / "control_commands.json").write_text(json.dumps(controller.control_rows) + "\n")
        (output / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
        audit = {"accepted": False, "reasons": ["simulation did not initialize"]}
        if env is not None and controller is not None:
            try:
                if audit_function is None:
                    from .audit_episode import audit_directory
                    audit_function = audit_directory
                audit = audit_function(output)
            except Exception as error:
                audit = {"accepted": False, "mechanical_accepted": None,
                         "reasons": ["audit_exception:" + str(error)],
                         "artifact_sha256": {name: digest((output / name).read_bytes()) for name in
                             ("expert_trace.jsonl", "control_commands.json", "outcome.json")}}
                (output / "independent_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        accepted = bool(outcome["success"] and audit["accepted"])
        summary = {"closed_loop_accepted": accepted, "controller_success": outcome["success"],
                   "independent_audit_accepted": audit["accepted"], "audit_reasons": audit["reasons"],
                   "controller_type": "learned_bc", "contact_mode": outcome["contact_mode"],
                   "checkpoint_sha256": outcome["policy"]["checkpoint_sha256"], "output": str(output)}
        (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("STUDENT_EVALUATION", json.dumps(summary), flush=True)
        return 0 if accepted else 1
    finally:
        if app is not None:
            try:
                app.close()
            except SystemExit as error:
                # Some Isaac hosts request exit(0) during close. A Python exit
                # request must not turn a rejected evaluation into shell success.
                # Native process termination cannot be intercepted here; all
                # authoritative artifacts were persisted before teardown.
                code = error.code
                status = {"system_exit_suppressed": True,
                          "requested_exit_code": code if code is None or isinstance(code, (str, int)) else repr(code),
                          "reason": "preserve evaluation return status and any pending exception"}
                (output / "teardown_status.json").write_text(json.dumps(status, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True,
                        help="expected SHA-256 of the trained model run_manifest.json")
    parser.add_argument("--points", nargs="+", default=["R1"])
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contact-mode", choices=("pure_bc", "hybrid_normal"), default="hybrid_normal")
    parser.add_argument("--waypoint-timeout", type=float)
    parser.add_argument("--speed-scale", type=float, default=1.)
    parser.add_argument("--scenario-id", default="frozen_bc_closed_loop_v1")
    args = parser.parse_args(argv)
    if len(set(args.points)) != len(args.points) or any(name not in DEFAULT_POINTS for name in args.points):
        parser.error("points must be distinct names R1/L1 through R5/L5")
    try:
        variation = timing_variation(args.speed_scale)
    except ValueError as error:
        parser.error(str(error))
    if args.waypoint_timeout is not None and (not np.isfinite(args.waypoint_timeout) or not 0 < args.waypoint_timeout <= 180):
        parser.error("waypoint-timeout must be in (0, 180] seconds")
    torch.set_num_threads(1)
    actor = load_actor(args.model_dir, args.manifest_sha256)
    bundles = [validate_plan(args.input_dir / ("planned_paths_" + point + ".json"),
                             args.input_dir, args.runtime_dir) for point in args.points]
    if any(bundle.target.name != point for bundle, point in zip(bundles, args.points)):
        raise ValueError("plan point does not match requested point")
    bases = [np.asarray(bundle.plan["world_from_left_base"]) for bundle in bundles]
    if any(not np.allclose(base, bases[0], atol=1e-9, rtol=0) for base in bases[1:]):
        raise ValueError("sequence plans disagree on robot base")
    for label in ("targets", "runtime", "collision_metadata", "collision_arrays", "kinematics_urdf"):
        if len({digest(bundle.inputs[label][1]) for bundle in bundles}) != 1:
            raise ValueError("sequence plans use inconsistent " + label)
    if digest(DEFAULT_URDF_PATH.read_bytes()) != digest(bundles[0].inputs["kinematics_urdf"][1]):
        raise ValueError("runtime kinematics URDF differs from pinned plan")
    from .revalidate_plans import validate_geometry_binding
    binding_sha = validate_geometry_binding(ROOT)
    for bundle in bundles:
        record = bundle.plan.get("revalidation", {})
        if record.get("accepted") is not True or record.get("collision_regeneration_sha256") != binding_sha:
            raise ValueError("plan has no current geometry revalidation: " + bundle.target.name)
    output = (args.output or ROOT / "results" / ("student_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = source_paths(ROOT)
    sources["collision_regeneration"] = ROOT / "results/collision_regeneration_verification.json"
    for name in actor.artifacts:
        sources["model/" + name] = args.model_dir / name
    snapshot = create_snapshot(bundles, output / "input_snapshot", sources)
    snapshot_sha = digest((snapshot / "manifest.json").read_bytes())
    manifest = verify_snapshot(snapshot, verify_sources=True)
    for name, data in actor.artifacts.items():
        if manifest["files"]["source/model/" + name]["sha256"] != digest(data):
            raise ValueError("model changed between validated load and snapshot")
    dependency_dir = ROOT / "python_deps"
    if dependency_dir.is_dir():
        sys.path.insert(0, str(dependency_dir))
    from .collision import CollisionChecker
    kin = OpenArmKinematics()
    with tempfile.TemporaryDirectory(prefix="ausc_student_geometry_") as directory:
        directory = Path(directory)
        for label, filename in (("collision_metadata", "collision_geometry.json"),
                                ("collision_arrays", "collision_geometry.npz")):
            (directory / filename).write_bytes(bundles[0].inputs[label][1])
        checker = CollisionChecker(kin, bases[0], directory=directory)
    app = env = controller = None
    outcome = {"success": False, "training_accepted": False, "points": []}
    try:
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        from isaacsim import SimulationApp
        # Run through run_checked_evaluation: Kit fast shutdown may exit 0
        # even after rejection; the parent verifies saved evidence separately.
        app = SimulationApp({"headless": True})
        from .env import IsaacContactEnv
        env = IsaacContactEnv(output)
        verify_snapshot(snapshot, verify_sources=True)
        if not np.allclose(env.base, bases[0], atol=1e-7, rtol=0):
            raise ValueError("live simulation base differs from pinned plan")
        controller = StudentController(env, kin, checker, actor=actor, contact_mode=args.contact_mode,
            waypoint_timeout_s=args.waypoint_timeout, speed_scale=args.speed_scale)
        outcome = controller.run([(bundle.target, bundle.row) for bundle in bundles])
        verify_snapshot(snapshot, verify_sources=True)
        if validate_geometry_binding(ROOT) != binding_sha:
            raise ValueError("collision source binding changed during execution")
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
            input_snapshot_manifest_sha256=snapshot_sha, collision_regeneration_sha256=binding_sha,
            source_binding="regenerated geometry and global/model source hashes verified before and after execution",
            controller_type="learned_bc", dataset_role="student_evaluation", expert_demonstration=False,
            policy=actor.metadata, scenario_id=args.scenario_id, speed_scale=args.speed_scale, variation=variation,
            contact_mode=args.contact_mode, weight_updates=False, hardware_commands=False,
            scope="closed-loop simulation-mechanics evaluation; no acoustic or clinical result")
        result = finalize_evaluation(output, app, env, controller, outcome)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
