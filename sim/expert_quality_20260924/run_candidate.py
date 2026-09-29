"""Run a candidate with the frozen simulation and independent saved audits.

Example on the physics host:
 python -m sim.expert_quality_20260924.run_candidate --policy staged_continuous \
   --input-dir sim/auscultation_expert_layout_v1/results/expert_v2/plans \
   --output sim/expert_quality_20260924/results/physics_staged

Only this new package and new output directories are written. Candidate code
is added to the original input snapshot and verified before and after motion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .candidate_planner import POLICIES, ROOT


def candidate_source_paths():
    return {"candidate/"+name: ROOT/name for name in
            ("candidate_controller.py", "candidate_planner.py", "run_candidate.py")}


def bind_candidate_manifest(output, policy):
    """Bind derived implementation to the already captured source objects."""
    output = Path(output)
    snapshot = json.loads((output/"input_snapshot/manifest.json").read_text())
    records = {}
    for key in candidate_source_paths():
        record = snapshot["files"]["source/"+key]
        records[Path(key).name] = {"path": "input_snapshot/objects/"+record["sha256"],
                                  "sha256": record["sha256"]}
    manifest = {"schema": "wam_expert_candidate_sources_v1", "candidate_policy": policy,
                "derived_controller": "sim.expert_quality_20260924.candidate_controller.CandidateController",
                "files": records}
    data = (json.dumps(manifest, indent=2)+"\n").encode()
    (output/"candidate_sources_manifest.json").write_bytes(data)
    outcome_path = output/"outcome.json"
    outcome = json.loads(outcome_path.read_text())
    outcome.update(candidate_sources_manifest="candidate_sources_manifest.json",
                   candidate_sources_manifest_sha256=hashlib.sha256(data).hexdigest())
    outcome_path.write_text(json.dumps(outcome, indent=2)+"\n")
    return manifest


def worker(argv, policy):
    from functools import partial
    from sim.auscultation_expert_layout_v1 import run_sequence
    from sim.auscultation_expert_layout_v1.run_low_elbow import audit_saved
    from .candidate_controller import CandidateController
    original = run_sequence.source_paths

    def bound_sources(root):
        return {**original(root), **candidate_source_paths()}

    def audit_candidate(output):
        bind_candidate_manifest(output, policy)
        return audit_saved(output)

    run_sequence.source_paths = bound_sources
    try:
        return run_sequence.main(argv, controller_class=partial(CandidateController, policy=policy),
                                 result_auditor=audit_candidate)
    finally:
        run_sequence.source_paths = original


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, default="staged_continuous")
    parser.add_argument("--worker", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    if args.worker:
        return worker(forwarded, args.policy)
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output_parser.add_argument("--input-dir", type=Path, required=True)
    paths, _ = output_parser.parse_known_args(forwarded)
    output = paths.output.resolve()
    if output.exists():
        parser.error("output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    command = [sys.executable, "-m", __package__+".run_candidate", "--worker", "--policy", args.policy, *forwarded]
    with log.open("x") as stream:
        try:
            child = subprocess.run(command, cwd=ROOT.parents[1], stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=3600)
            code = child.returncode
        except subprocess.TimeoutExpired:
            code = None
    from sim.auscultation_expert_layout_v1.run_low_elbow import verify_saved
    from sim.auscultation_expert_layout_v1.controller import verify_snapshot
    report = verify_saved(output, code)
    report["candidate_policy"] = args.policy
    try:
        manifest = verify_snapshot(output/"input_snapshot", verify_sources=True)
        if any("source/"+key not in manifest["files"] for key in candidate_source_paths()):
            raise ValueError("candidate code absent from input snapshot")
        outcome = json.loads((output/"outcome.json").read_text())
        if outcome.get("candidate_policy") != args.policy:
            raise ValueError("candidate policy differs from request")
        candidate_manifest_bytes = (output/"candidate_sources_manifest.json").read_bytes()
        if (outcome.get("candidate_sources_manifest") != "candidate_sources_manifest.json"
                or hashlib.sha256(candidate_manifest_bytes).hexdigest() != outcome.get("candidate_sources_manifest_sha256")):
            raise ValueError("candidate source manifest differs from outcome")
        candidate_manifest = json.loads(candidate_manifest_bytes)
        for record in candidate_manifest["files"].values():
            if hashlib.sha256((output/record["path"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("candidate source object differs from manifest")
        report["candidate_sources_manifest"] = candidate_manifest
        report["candidate_sources_manifest_sha256"] = outcome["candidate_sources_manifest_sha256"]
    except (OSError, ValueError, KeyError) as error:
        report["accepted"] = False
        report["reasons"].append("candidate_source_binding:"+str(error))
    destination = output/"runner_summary.json" if output.exists() else log.with_suffix(".runner.json")
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
