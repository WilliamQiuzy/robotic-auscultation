"""Run the low-elbow expert and verify saved audits in an outer process."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def audit_saved(output):
    from .audit_episode import audit_directory as mechanical_audit
    from .audit_low_elbow import audit_directory as posture_audit
    output = Path(output)
    results = {}
    for key, function in (("mechanics", mechanical_audit), ("posture", posture_audit)):
        try:
            results[key] = function(output)
        except Exception as error:
            results[key] = {"accepted": False, "reasons": [str(error)]}
    report = {"accepted": all(value.get("accepted") is True for value in results.values()),
              "audits": results, "scope": "simulation mechanics and low-elbow posture"}
    (output / "verification_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_saved(output, returncode):
    output = Path(output)
    errors, hashes, contents, paths = [], {}, {}, {}

    def capture(name, path):
        if name not in contents:
            contents[name] = path.read_bytes()
            paths[name] = path
            hashes[name] = hashlib.sha256(contents[name]).hexdigest()
        return contents[name]

    try:
        data = capture("verification_summary.json", output / "verification_summary.json")
        report = json.loads(data)
        if not isinstance(report, dict):
            raise ValueError("verification summary must be an object")
        if report.get("accepted") is not True:
            errors.append("audits_rejected")
        audits = report.get("audits")
        if not isinstance(audits, dict) or set(audits) != {"mechanics", "posture"}:
            raise ValueError("missing_audit")
        required = {"mechanics": {"expert_trace.jsonl", "control_commands.json", "outcome.json"},
                    "posture": {"expert_trace.jsonl", "expert_replay.npz", "outcome.json"}}
        for kind, audit in audits.items():
            if not isinstance(audit, dict):
                raise ValueError("malformed_audit:" + kind)
            if audit.get("accepted") is not True:
                errors.append("audit_rejected")
            evidence = audit.get("artifact_sha256")
            if not isinstance(evidence, dict) or not required[kind].issubset(evidence):
                raise ValueError("missing_audit_artifacts:" + kind)
            for name, expected in evidence.items():
                if Path(name).name != name:
                    raise ValueError("invalid_artifact_name:" + name)
                capture(name, output / name)
                if hashes[name] != expected:
                    errors.append("changed_artifact:" + name)
        sources = audits["posture"].get("source_sha256")
        if not isinstance(sources, dict) or set(sources) != {"kinematics_urdf", "world_from_left_base"}:
            raise ValueError("missing_posture_sources")
        for name, record in sources.items():
            if not isinstance(record, dict):
                raise ValueError("malformed_posture_source:" + name)
            path = Path(record["path"])
            if name == "world_from_left_base" and path.resolve() != (output / "world_from_left_base.json").resolve():
                raise ValueError("posture_base_path_mismatch")
            key = "source/" + name
            capture(key, path)
            if hashes[key] != record["sha256"]:
                errors.append("changed_source:" + name)
        # Parse the same bytes matched to both independent audit records.
        outcome = json.loads(contents["outcome.json"])
        if not isinstance(outcome, dict):
            raise ValueError("outcome must be an object")
        points = outcome.get("points")
        if (outcome.get("success") is not True or outcome.get("training_accepted") is not True
                or outcome.get("controller_type") != "algorithmic_expert_low_elbow"
                or not isinstance(points, list) or not points):
            errors.append("controller_rejected")
        if not isinstance(points, list) or any(not isinstance(row, dict) for row in points):
            raise ValueError("malformed_points")
        if any(row.get("success") is not True or row.get("training_accepted") is not True for row in points):
            errors.append("point_rejected")
        if [row.get("target") for row in points] != outcome.get("requested_points"):
            errors.append("incomplete_sequence")
        for name, expected in hashes.items():
            if hashlib.sha256(paths[name].read_bytes()).hexdigest() != expected:
                errors.append("changed_after_read:" + name)
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append("missing_or_invalid_evidence:" + str(error))
    if returncode != 0:
        errors.append("child_process_failed")
    return {"accepted": not errors, "reasons": errors, "child_returncode": returncode,
            "evidence_sha256": hashes}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        argv.remove("--worker")
        from .run_sequence import main as run_sequence
        from .low_elbow_controller import LowElbowController
        return run_sequence(argv, controller_class=LowElbowController, result_auditor=audit_saved)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    output = args.output.resolve()
    if output.exists():
        parser.error("output must be a new directory")
    log = output.with_suffix(".log")
    command = [sys.executable, "-m", __package__ + ".run_low_elbow", "--worker", *argv]
    with log.open("x") as stream:
        try:
            child = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
            returncode = child.returncode
        except subprocess.TimeoutExpired:
            returncode = None
    report = verify_saved(output, returncode)
    destination = output / "runner_summary.json" if output.exists() else log.with_suffix(".runner.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
