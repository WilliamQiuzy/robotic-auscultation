"""Compare recorded expert movement; these descriptive metrics are not safety proof."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np

from .audit_low_elbow import elbow_positions_world
from .kinematics import OpenArmKinematics, DEFAULT_URDF_PATH


def recorded_metrics(directory, require_posture=False):
    directory = Path(directory)
    names = ["outcome.json", "expert_trace.jsonl", "control_commands.json", "independent_audit.json",
             "world_from_left_base.json", "input_snapshot/manifest.json"]
    if require_posture:
        names += ["posture_audit.json", "verification_summary.json", "runner_summary.json", "expert_replay.npz"]
    captured = {name: (directory/name).read_bytes() for name in names}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in captured.items()}
    outcome = json.loads(captured["outcome.json"])
    audit = json.loads(captured["independent_audit.json"])
    if (outcome.get("success") is not True or outcome.get("training_accepted") is not True
            or audit.get("accepted") is not True):
        raise ValueError("comparison requires completed mechanical runs")
    for name in ("outcome.json", "expert_trace.jsonl", "control_commands.json"):
        if audit["artifact_sha256"][name] != hashes[name]:
            raise ValueError("mechanical evidence fingerprint mismatch")
    if hashes["input_snapshot/manifest.json"] != outcome.get("input_snapshot_manifest_sha256"):
        raise ValueError("comparison snapshot fingerprint mismatch")
    snapshot = json.loads(captured["input_snapshot/manifest.json"])

    def snapshot_object(label):
        expected = snapshot["files"][label]["sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("invalid comparison snapshot object fingerprint")
        name = "input_snapshot/objects/" + expected
        if name not in captured:
            captured[name] = (directory/name).read_bytes()
            hashes[name] = hashlib.sha256(captured[name]).hexdigest()
        if hashes[name] != expected:
            raise ValueError("comparison snapshot object changed: " + label)
        return captured[name]

    urdf = snapshot_object("source/urdf")
    urdf_sha = hashlib.sha256(urdf).hexdigest()
    if urdf_sha != hashlib.sha256(DEFAULT_URDF_PATH.read_bytes()).hexdigest():
        raise ValueError("comparison FK differs from recorded URDF")
    base_value = json.loads(captured["world_from_left_base.json"])
    base = np.asarray(base_value.get("world_from_left_base", base_value) if isinstance(base_value, dict) else base_value)
    if base.shape != (4, 4) or not np.isfinite(base).all():
        raise ValueError("invalid recorded base")
    # Older mechanical runs do not carry posture-source hashes. Their immutable
    # plan/runtime snapshot still binds the base used to recompute elbow height.
    for point in outcome["requested_points"]:
        runtime = json.loads(snapshot_object(point + "/runtime"))
        recorded_base = np.asarray(runtime["world_from_left_base"], float)
        if recorded_base.shape != (4, 4) or not np.allclose(base, recorded_base, atol=1e-7, rtol=0):
            raise ValueError("comparison base differs from recorded runtime")
    if require_posture:
        posture = json.loads(captured["posture_audit.json"])
        verification = json.loads(captured["verification_summary.json"])
        runner = json.loads(captured["runner_summary.json"])
        if (posture.get("accepted") is not True or verification.get("accepted") is not True
                or runner.get("accepted") is not True or runner.get("child_returncode") != 0):
            raise ValueError("new expert has not passed posture/outer verification")
        for name in ("outcome.json", "expert_trace.jsonl", "expert_replay.npz"):
            if posture["artifact_sha256"][name] != hashes[name]:
                raise ValueError("posture evidence fingerprint mismatch")
        sources = posture["source_sha256"]
        if (sources["world_from_left_base"]["sha256"] != hashes["world_from_left_base.json"]
                or sources["kinematics_urdf"]["sha256"] != urdf_sha):
            raise ValueError("posture source fingerprint mismatch")
        if ("world_from_left_base.json" in posture["artifact_sha256"]
                and posture["artifact_sha256"]["world_from_left_base.json"] != hashes["world_from_left_base.json"]):
            raise ValueError("posture base artifact fingerprint mismatch")
        if verification.get("audits") != {"mechanics": audit, "posture": posture}:
            raise ValueError("comparison independent reports differ from verified reports")
        runner_hashes = runner["evidence_sha256"]
        for name in ("outcome.json", "expert_trace.jsonl", "control_commands.json", "expert_replay.npz", "verification_summary.json"):
            if runner_hashes.get(name) != hashes[name]:
                raise ValueError("outer evidence fingerprint mismatch")
        if (runner_hashes.get("source/world_from_left_base") != hashes["world_from_left_base.json"]
                or runner_hashes.get("source/kinematics_urdf") != urdf_sha):
            raise ValueError("outer source fingerprint mismatch")
    samples = []
    for line in captured["expert_trace.jsonl"].splitlines():
        row = json.loads(line)
        samples.append((row["q"], row["time_s"], row["target"]))
    q = np.asarray([row[0] for row in samples], float)
    times = np.asarray([row[1] for row in samples], float)
    if not np.isfinite(q).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("invalid recorded sequence")
    # Parse exactly the immutable URDF bytes whose snapshot/source hashes were
    # checked above, even if the current checkout changes during computation.
    with tempfile.NamedTemporaryFile(suffix=".urdf") as captured_urdf:
        captured_urdf.write(urdf)
        captured_urdf.flush()
        heights = elbow_positions_world(OpenArmKinematics(captured_urdf.name), base, q)[:, 2]
    points = {}
    for point in outcome["points"]:
        start, end = point["start_physics_row"], point["end_physics_row"]
        if end <= start or any(row[2] != point["target"] for row in samples[start:end]):
            raise ValueError("invalid recorded point coverage")
        points[point["target"]] = {
            "max_elbow_z_m": float(heights[start:end].max()),
            "joint_travel_rad": float(np.abs(np.diff(q[start:end], axis=0)).sum()),
            "simulation_duration_s": float(times[end-1]-times[start]+outcome["physics_dt_s"]),
            "contact_hold_s": point["verified_front_contact_hold_s"]}
    for name, original in captured.items():
        if (directory/name).read_bytes() != original:
            raise ValueError("comparison input changed")
    return {"directory": str(directory.resolve()), "input_sha256": hashes, "points": points,
            "max_elbow_z_m": float(heights.max()),
            "total_joint_travel_rad": float(np.abs(np.diff(q, axis=0)).sum()),
            "simulation_duration_s": float(times[-1]-times[0]+outcome["physics_dt_s"]),
            "requested_points": outcome["requested_points"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("comparison output must be new")
    before = recorded_metrics(args.before)
    after = recorded_metrics(args.after, require_posture=True)
    if before["requested_points"] != after["requested_points"]:
        raise ValueError("compare identical requested point sequences")
    report = {"before": before, "after": after,
              "peak_elbow_reduction_m": before["max_elbow_z_m"]-after["max_elbow_z_m"],
              "joint_travel_reduction_fraction": 1-after["total_joint_travel_rad"]/before["total_joint_travel_rad"],
              "scope": "recorded movement comparison, not motor-load or hardware validation"}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("before", "after")}))


if __name__ == "__main__":
    main()
