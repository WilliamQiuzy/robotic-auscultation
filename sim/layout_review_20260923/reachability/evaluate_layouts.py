"""Bounded static IK comparison; does not authorize collisions or trajectories."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sim.auscultation_training.kinematics import OpenArmKinematics, DEFAULT_URDF_PATH
from sim.auscultation_training.pose_ik import IKBudgetExpired, solve_posture_ik
from sim.auscultation_training.posture import PostureLimits

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODULE = ROOT / "sim/auscultation_training"
PLANS = MODULE / "results/low_elbow_release_v1/plans"
NAMES = [f"{side}{i}" for i in range(1, 6) for side in ("R", "L")]
VARIANTS = [
    ("baseline", [0., 0., 0.]),
    ("farther_5cm_lower_5cm", [0., .05, -.05]),
    ("farther_10cm_lower_5cm", [0., .10, -.05]),
    ("farther_15cm_lower_5cm", [0., .15, -.05]),
    ("farther_10cm_only", [0., .10, 0.]),
    ("lower_5cm_only", [0., 0., -.05]),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(kin, base, limits, q, contact):
    links = kin._chain(q)[0]
    elbow = (base @ links["openarm_left_link4"])[:3, 3]
    wrist = (base @ links["openarm_left_link7"])[:3, 3]
    return {
        "q_rad": q.tolist(),
        "q_deg": np.rad2deg(q).tolist(),
        "elbow_world_m": elbow.tolist(),
        "wrist_world_m": wrist.tolist(),
        "elbow_above_shoulder_m": float(elbow[2] - base[2, 3]),
        "elbow_below_wrist_m": float(wrist[2] - elbow[2]),
        "min_urdf_joint_margin_deg": float(np.rad2deg(np.min(np.r_[q - kin.lower_limits, kin.upper_limits - q]))),
        "min_posture_joint_margin_deg": float(np.rad2deg(np.min(np.r_[q - limits.joint_lower_rad, limits.joint_upper_rad - q]))),
        "joint2_upper_margin_deg": float(np.rad2deg(kin.upper_limits[1] - q[1])),
        "posture": limits.evaluate(kin, base, q, contact=contact),
    }


def main():
    began = time.monotonic()
    deadline = began + 360.
    kin = OpenArmKinematics()
    source_files = [Path(__file__), DEFAULT_URDF_PATH, MODULE / "runtime_inspection.json", PLANS / "posture_limits.json", PLANS / "collision_geometry.json"]
    source_files += [MODULE / n for n in ["pose_ik.py", "kinematics.py", "posture.py", "tool_frames.py", "targets.py"]]
    documents = {}
    for name in NAMES:
        path = PLANS / f"planned_paths_{name}.json"
        source_files.append(path)
        documents[name] = json.loads(path.read_text())
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    base = np.asarray(json.loads((MODULE / "runtime_inspection.json").read_text())["world_from_left_base"], float)
    limits = PostureLimits.from_dict(json.loads((PLANS / "posture_limits.json").read_text()), kin)
    for document in documents.values():
        assert np.allclose(base, document["world_from_left_base"], rtol=0, atol=1e-10)
        assert document["points"][0]["posture_limits"] == limits.to_dict()
    out = {
        "schema": "layout_static_ik_review_v1",
        "scope": "Static exact pose IK only. No FCL, moved patient collision geometry, trajectories, actuator dynamics, force, acoustic coupling or PhysX checks performed. Failed bounded search is not proof of mathematical unreachability.",
        "assumptions": {
            "patient_targets_translate_with_body": True,
            "robot_base_and_table_unchanged": True,
            "tool_orientation_and_surface_normal_unchanged_relative_to_body": True,
            "stool_seat_and_body_vertical_translation_m": -.05,
            "table_patient_side_world_y_m": .5421375153481023,
            "farther_direction_world": [0., 1., 0.],
            "poses": "lower staging, 25mm precontact, and loaded 2mm compression; exact reference TCP FK translated with body",
            "contact_gap_constraint_applies": ["precontact", "loaded"],
            "planning_elbow_buffer_m": .005,
            "planning_joint_buffer_deg": .2,
            "seeds_per_pose": 2,
            "solver_max_iterations": 120,
            "per_attempt_wall_budget_s": 2.,
            "global_wall_budget_s": 360.,
            "selection": "accepted solution with largest minimum URDF joint margin among two fixed deterministic seeds; otherwise smallest pose residual",
        },
        "world_from_left_base": base.tolist(),
        "posture_limits": limits.to_dict(),
        "source_sha256": source_hashes,
        "variants": [],
    }
    for variant, offset in VARIANTS:
        delta = np.asarray(offset, float)
        record = {"name": variant, "offset_world_m": offset, "points": []}
        for name in NAMES:
            row = documents[name]["points"][0]
            poses = [("staging", row["lower_staging_q"], 0., False), ("precontact", row["approach_q"][0], 0., True), ("loaded", row["loaded_q"], row["loaded_compression_m"], True)]
            point = {"name": name, "target_world_m": (np.asarray(row["position_world"]) + delta).tolist(), "poses": {}}
            for pose_name, original, compression, contact in poses:
                original = np.asarray(original, float)
                reference = kin.tcp(original, compression)
                desired_position = reference[:3, 3] + base[:3, :3].T @ delta
                rotation = reference[:3, :3]
                # The same deterministic seed choices are used for every layout.
                seeds = [original, original + np.array([-.12, -.18, .08, -.10, .08, -.03, -.06])]
                attempts = []
                for seed_index, seed in enumerate(seeds):
                    if time.monotonic() >= deadline:
                        attempts.append({"seed_index": seed_index, "accepted": False, "budget_expired": True})
                        break
                    try:
                        q, accepted, pe, re = solve_posture_ik(kin, base, limits, desired_position, rotation, seed,
                            compression=compression, maxiter=120, deadline=min(deadline, time.monotonic()+2.), contact=contact)
                        attempts.append({"seed_index": seed_index, "accepted": bool(accepted), "position_error_m": pe,
                            "orientation_error_rad": re, **metrics(kin, base, limits, q, contact)})
                    except IKBudgetExpired:
                        attempts.append({"seed_index": seed_index, "accepted": False, "budget_expired": True})
                accepted_attempts = [a for a in attempts if a["accepted"]]
                candidates = accepted_attempts or [a for a in attempts if "position_error_m" in a]
                selected = (max(candidates, key=lambda a:a["min_urdf_joint_margin_deg"]) if accepted_attempts else
                    min(candidates, key=lambda a:a["position_error_m"]+.1*a["orientation_error_rad"])) if candidates else None
                point["poses"][pose_name] = {"accepted": bool(accepted_attempts), "compression_m": compression,
                    "target_tcp_world_m": (base[:3, :3] @ desired_position + base[:3, 3]).tolist(),
                    "rotation_world": (base[:3, :3] @ rotation).tolist(), "attempts": attempts, "selected": selected}
            point["all_three_poses_accepted"] = all(p["accepted"] for p in point["poses"].values())
            point["precontact_and_loaded_accepted"] = all(point["poses"][p]["accepted"] for p in ["precontact", "loaded"])
            record["points"].append(point)
        record["all_three_poses_accepted_points"] = sum(p["all_three_poses_accepted"] for p in record["points"])
        record["precontact_and_loaded_accepted_points"] = sum(p["precontact_and_loaded_accepted"] for p in record["points"])
        record["accepted_pose_count"] = sum(pose["accepted"] for p in record["points"] for pose in p["poses"].values())
        selected_accepted = [pose["selected"] for p in record["points"] for pose in p["poses"].values() if pose["accepted"]]
        record["min_urdf_joint_margin_deg_for_accepted_poses"] = min(s["min_urdf_joint_margin_deg"] for s in selected_accepted) if selected_accepted else None
        out["variants"].append(record)
        print(json.dumps({k:v for k,v in record.items() if k!="points"}), flush=True)
        (HERE / "layout_ik_review.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
    out["elapsed_seconds"] = time.monotonic()-began
    out["source_unchanged"] = all(sha(ROOT/p)==h for p,h in source_hashes.items())
    assert out["source_unchanged"]
    (HERE / "layout_ik_review.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
