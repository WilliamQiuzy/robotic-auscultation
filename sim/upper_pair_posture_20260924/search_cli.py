"""Run the finite R1/L1 alternative search in a new directory (Linux FCL)."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .search import SCOPE, fresh_plan_document, search_point


PACKAGE = Path(__file__).resolve().parent
CANONICAL_ROOT = PACKAGE.parent/"auscultation_expert_layout_v1"
CANONICAL_PLANS = CANONICAL_ROOT/"results/expert_v2/plans"
ALL_POINTS = [f"{side}{number}" for number in range(1, 6) for side in ("R", "L")]


def _write_json(path, document):
    path.write_text(json.dumps(document, indent=2, allow_nan=False)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing directories are rejected")
    parser.add_argument("--input-dir", type=Path, default=CANONICAL_PLANS)
    parser.add_argument("--runtime-dir", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--points", nargs="+", choices=("R1", "L1"), default=["R1", "L1"])
    parser.add_argument("--max-seconds", type=float, default=1200., help="global search wall budget")
    parser.add_argument("--max-routes-per-point", type=int, default=6)
    args = parser.parse_args(argv)
    if (not np.isfinite(args.max_seconds) or args.max_seconds < 5.
            or not 1 <= args.max_routes_per_point <= 24 or len(set(args.points)) != len(args.points)):
        parser.error("invalid finite search budgets or duplicated points")
    if args.output.exists():
        parser.error("output already exists; immutable source/results must not be overwritten")
    from sim.auscultation_expert_layout_v1.controller import validate_plan
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    from sim.auscultation_expert_layout_v1.replan_layout import PinnedInputs, SafetyChecker, sha
    from sim.auscultation_expert_layout_v1.targets import load_targets

    pinned = PinnedInputs()
    documents, audits, bundles = {}, {}, {}
    for name in ALL_POINTS:
        path = args.input_dir/f"planned_paths_{name}.json"
        documents[name] = json.loads(pinned.read(path))
        bundles[name] = validate_plan(path, args.input_dir, args.runtime_dir)
        for _, (source, data) in bundles[name].inputs.items():
            if pinned.read(source) != data:
                raise ValueError("source changed while validating input plan")
        audits[name] = json.loads(bundles[name].inputs["audit"][1])
    base = np.asarray(documents["R1"]["world_from_left_base"], float)
    kin = OpenArmKinematics(bundles["R1"].inputs["kinematics_urdf"][0])
    limits_bytes = pinned.read(args.input_dir/"posture_limits.json")
    limits_dict = json.loads(limits_bytes)
    limits = PostureLimits.from_dict(limits_dict, kin)
    for name, document in documents.items():
        if document["points"][0]["posture_limits"] != limits_dict:
            raise ValueError("all ten current posture envelopes must be identical: "+name)
        if not np.array_equal(np.asarray(document["world_from_left_base"]), base):
            raise ValueError("all ten current base transforms must be identical: "+name)
    for path in (PACKAGE/"search.py", Path(__file__), CANONICAL_ROOT/"pose_ik.py",
                 CANONICAL_ROOT/"replan_layout.py", CANONICAL_ROOT/"plan_low_elbow.py",
                 CANONICAL_ROOT/"collision.py", CANONICAL_ROOT/"kinematics.py",
                 CANONICAL_ROOT/"posture.py", CANONICAL_ROOT/"tool_frames.py",
                 CANONICAL_ROOT/"controller.py", PACKAGE.parent/"curobo_expert_v1/fast_collision.py"):
        pinned.read(path)
    # The only FCL import on this module's command path. --help and CPU tests
    # work without native FCL; a real search never substitutes a fake checker.
    from sim.curobo_expert_v1.fast_collision import FastCollisionChecker
    collision = FastCollisionChecker(kin, base, directory=args.input_dir)
    checker = SafetyChecker(collision, base, limits, margin_deg=3.)
    targets = {point.name: point for point in load_targets(base, args.input_dir/"contact_targets.json")}
    pinned.verify()
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {str(path): {"sha256": sha(data), "bytes": len(data)} for path, data in pinned.files.items()}
    summary = {"schema": "upper_pair_posture_search_v1", "scope": SCOPE,
               "input_dir": str(args.input_dir.resolve()), "runtime_dir": str(args.runtime_dir.resolve()),
               "requested_points": args.points, "source_hashes": source_hashes,
               "max_seconds": args.max_seconds, "max_routes_per_point": args.max_routes_per_point,
               "fixed_scene": True, "physics_accepted": False, "candidates": [], "points": []}
    _write_json(args.output/"input_manifest.json", source_hashes)
    _write_json(args.output/"search_summary.json", summary)
    started = time.monotonic()
    deadline = started+args.max_seconds

    def progress(document):
        print(json.dumps(document, separators=(",", ":"), allow_nan=False), flush=True)

    try:
        for number, name in enumerate(args.points):
            remaining = deadline-time.monotonic()
            if remaining <= 0.:
                summary["stopped"] = "global_time_budget"
                break
            neighbors = [documents[other]["points"][0] for other in ("R1", "L1", "R2") if other != name]
            preceding = np.zeros(7) if name == "R1" else np.asarray(documents["R1"]["points"][0]["approach_q"][0])
            following = np.asarray(documents["L1" if name == "R1" else "R2"]["points"][0]["approach_q"][0])
            result = search_point(targets[name], audits[name], documents[name], kin, base, limits, checker,
                neighbors=neighbors, anchors=(preceding, following), max_seconds=remaining/(len(args.points)-number),
                max_routes=args.max_routes_per_point, seed=265+ALL_POINTS.index(name), progress=progress)
            pinned.verify()
            for candidate in result["candidates"]:
                directory = args.output/candidate["id"]
                directory.mkdir(exist_ok=False)
                # Only this point's candidate and exact source inputs are copied.
                # Complete ten-point candidate sets are integration's responsibility.
                for label, (source, data) in bundles[name].inputs.items():
                    if label != "plan":
                        (directory/source.name).write_bytes(data)
                (directory/"posture_limits.json").write_bytes(limits_bytes)
                plan_path = directory/f"planned_paths_{name}.json"
                document = fresh_plan_document(documents[name], candidate["row"])
                _write_json(plan_path, document)
                # Verify this freshly written standalone point's exact bindings.
                validate_plan(plan_path, directory, directory)
                record = {key: value for key, value in candidate.items() if key != "row"}
                record.update(plan_path=str(plan_path.relative_to(args.output)),
                              absolute_plan_path=str(plan_path.resolve()),
                              plan_sha256=sha(plan_path.read_bytes()),
                              fit=document["points"][0]["selected_fit"],
                              staging_roll_mode=document["points"][0]["staging_roll_mode"])
                summary["candidates"].append(record)
            summary["points"].append({key: value for key, value in result.items() if key != "candidates"})
            summary["elapsed_seconds"] = time.monotonic()-started
            _write_json(args.output/"search_summary.json", summary)
        pinned.verify()
        summary["source_inputs_unchanged"] = True
        summary["completed"] = True
    except Exception as error:
        summary.update(completed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic()-started
        summary["conclusion_scope"] = "Finite candidate search; absent or rejected candidates do not establish impossibility."
        _write_json(args.output/"search_summary.json", summary)
    progress({"phase": "search_finished", "candidates": len(summary["candidates"]),
              "summary": str((args.output/"search_summary.json").resolve())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
