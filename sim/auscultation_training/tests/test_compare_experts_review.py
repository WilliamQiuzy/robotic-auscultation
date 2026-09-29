"""Independent regression checks for comparison evidence bindings."""
import hashlib
import json

import numpy as np
import pytest

from sim.auscultation_training.compare_experts import recorded_metrics
from sim.auscultation_training.kinematics import DEFAULT_URDF_PATH


def comparison_fixture(path):
    def write(name, value):
        destination = path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value).encode()
        destination.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    urdf_sha = hashlib.sha256(DEFAULT_URDF_PATH.read_bytes()).hexdigest()
    runtime = {"world_from_left_base": np.eye(4).tolist()}
    runtime_sha = hashlib.sha256(json.dumps(runtime).encode()).hexdigest()
    write("input_snapshot/objects/" + runtime_sha, runtime)
    (path / "input_snapshot/objects" / urdf_sha).write_bytes(DEFAULT_URDF_PATH.read_bytes())
    snapshot_sha = write("input_snapshot/manifest.json", {"files": {
        "source/urdf": {"sha256": urdf_sha}, "R1/runtime": {"sha256": runtime_sha}}})
    base_sha = write("world_from_left_base.json", np.eye(4).tolist())
    outcome_sha = write("outcome.json", {
        "success": True, "training_accepted": True, "controller_type": "algorithmic_expert_low_elbow",
        "requested_points": ["R1"], "physics_dt_s": 1 / 240,
        "input_snapshot_manifest_sha256": snapshot_sha,
        "points": [{"target": "R1", "success": True, "training_accepted": True,
                    "start_physics_row": 0, "end_physics_row": 2,
                    "verified_front_contact_hold_s": 3.}]})
    rows = [{"q": [0., -.4, 0., .6 + index * .01, 0., 0., 0.],
             "time_s": (index + 1) / 240, "target": "R1"} for index in range(2)]
    trace_data = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    (path / "expert_trace.jsonl").write_bytes(trace_data)
    hashes = {"outcome.json": outcome_sha, "expert_trace.jsonl": hashlib.sha256(trace_data).hexdigest(),
              "world_from_left_base.json": base_sha}
    hashes["control_commands.json"] = write("control_commands.json", [])
    (path / "expert_replay.npz").write_bytes(b"bound comparison fixture")
    hashes["expert_replay.npz"] = hashlib.sha256((path / "expert_replay.npz").read_bytes()).hexdigest()
    mechanics = {"accepted": True, "artifact_sha256": {name: hashes[name] for name in
        ("outcome.json", "expert_trace.jsonl", "control_commands.json")}}
    posture = {"accepted": True, "artifact_sha256": {name: hashes[name] for name in
        ("outcome.json", "expert_trace.jsonl", "expert_replay.npz", "world_from_left_base.json")},
        "source_sha256": {"kinematics_urdf": {"path": str(DEFAULT_URDF_PATH), "sha256": urdf_sha},
            "world_from_left_base": {"path": str(path / "world_from_left_base.json"), "sha256": base_sha}}}
    write("independent_audit.json", mechanics)
    write("posture_audit.json", posture)
    verification_sha = write("verification_summary.json", {
        "accepted": True, "audits": {"mechanics": mechanics, "posture": posture}})
    write("runner_summary.json", {"accepted": True, "child_returncode": 0,
        "evidence_sha256": {**hashes, "verification_summary.json": verification_sha,
                            "source/kinematics_urdf": urdf_sha, "source/world_from_left_base": base_sha}})


def test_comparison_retains_complete_bound_input(tmp_path):
    comparison_fixture(tmp_path)
    result = recorded_metrics(tmp_path, require_posture=True)
    assert result["simulation_duration_s"] == pytest.approx(2 / 240)
    assert result["total_joint_travel_rad"] == pytest.approx(.01)


@pytest.mark.parametrize("require_posture", [False, True])
def test_comparison_rejects_base_changed_after_accepted_posture_audit(tmp_path, require_posture):
    comparison_fixture(tmp_path)
    base = np.eye(4)
    base[2, 3] = .5
    (tmp_path / "world_from_left_base.json").write_text(json.dumps(base.tolist()))
    with pytest.raises(ValueError):
        recorded_metrics(tmp_path, require_posture=require_posture)


def test_comparison_rejects_unbound_snapshot_manifest(tmp_path):
    comparison_fixture(tmp_path)
    path = tmp_path / "input_snapshot/manifest.json"
    snapshot = json.loads(path.read_bytes())
    snapshot["unrecorded_metadata"] = "changed after run"
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError):
        recorded_metrics(tmp_path, require_posture=True)


def test_comparison_rejects_runner_from_a_different_episode(tmp_path):
    comparison_fixture(tmp_path)
    path = tmp_path / "runner_summary.json"
    runner = json.loads(path.read_bytes())
    runner["evidence_sha256"]["outcome.json"] = "0" * 64
    path.write_text(json.dumps(runner))
    with pytest.raises(ValueError):
        recorded_metrics(tmp_path, require_posture=True)


def test_comparison_rejects_nested_verification_disagreement(tmp_path):
    comparison_fixture(tmp_path)
    path = tmp_path / "verification_summary.json"
    verification = json.loads(path.read_bytes())
    verification["audits"]["posture"]["accepted"] = False
    data = json.dumps(verification).encode()
    path.write_bytes(data)
    runner_path = tmp_path / "runner_summary.json"
    runner = json.loads(runner_path.read_bytes())
    runner["evidence_sha256"]["verification_summary.json"] = hashlib.sha256(data).hexdigest()
    runner_path.write_text(json.dumps(runner))
    with pytest.raises(ValueError):
        recorded_metrics(tmp_path, require_posture=True)


def test_comparison_rejects_replay_changed_after_posture_acceptance(tmp_path):
    comparison_fixture(tmp_path)
    (tmp_path / "expert_replay.npz").write_bytes(b"changed after audit")
    with pytest.raises(ValueError):
        recorded_metrics(tmp_path, require_posture=True)
