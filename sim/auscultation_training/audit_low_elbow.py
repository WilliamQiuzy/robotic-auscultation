"""Independent recorded-motion posture audit; simulation evidence only.

The 240 Hz measured joints and commands are checked separately. Recorded rigid
body positions supply an additional actual elbow-height gate at every saved
30 Hz pose. Neither collision avoidance nor motor/cable safety follows from
these posture checks; the existing mechanical audit remains required.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .kinematics import DEFAULT_URDF_PATH, OpenArmKinematics
from .posture import PostureLimits
from .targets import _base_transform


BOUND_EPS = 1e-9
REPLAY_POSITION_TOLERANCE_M = .002
REVERSAL_THRESHOLD_RAD = np.deg2rad(1.)
CONTACT_PHASES = ("CONTACT", "HOLD", "RETRACT")
KNOWN_PHASES = ("APPROACH", *CONTACT_PHASES, "DONE", "ABORT")
ARTIFACTS = ("outcome.json", "expert_trace.jsonl", "expert_replay.npz",
             "world_from_left_base.json")


def landmark_positions_world(kin, base, q, link):
    """Batch URDF FK without clipping joints; out-of-bounds rows remain visible."""
    q = np.asarray(q, float)
    base = _base_transform(base)
    if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
        raise ValueError("invalid joint array")
    transform = np.broadcast_to(base, (len(q), 4, 4)).copy()
    for index, joint in enumerate(kin._joints):
        transform = transform @ joint.origin
        axis = joint.axis
        skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                         [-axis[1], axis[0], 0.]])
        angle = q[:, index, None, None]
        rotation = (np.eye(3) + np.sin(angle)*skew
                    + (1.-np.cos(angle))*(skew @ skew))
        motion = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        motion[:, :3, :3] = rotation
        transform = transform @ motion
        if joint.child == link:
            return transform[:, :3, 3]
    raise ValueError("landmark absent from URDF")


def elbow_positions_world(kin, base, q):
    return landmark_positions_world(kin, base, q, "openarm_left_link4")


def direction_reversals(q, threshold_rad=REVERSAL_THRESHOLD_RAD):
    """Count reversals only after a one-degree excursion from a running extreme.

    Many tiny steps may accumulate into a real reversal. Subthreshold jitter
    around an extreme does not count, unlike counting signs of each difference.
    """
    q = np.asarray(q, float)
    if (q.ndim != 2 or q.shape[1] != 7 or len(q) < 1
            or not np.isfinite(q).all() or not np.isfinite(threshold_rad)
            or threshold_rad <= 0):
        raise ValueError("invalid reversal input")
    direction = np.zeros(7, dtype=int)
    extreme = q[0].copy()
    count = np.zeros(7, dtype=int)
    for value in q[1:]:
        for j in range(7):
            if direction[j] == 0:
                if abs(value[j]-extreme[j]) > threshold_rad:
                    direction[j] = 1 if value[j] > extreme[j] else -1
                    extreme[j] = value[j]
            elif direction[j]*(value[j]-extreme[j]) > 0:
                extreme[j] = value[j]
            elif direction[j]*(extreme[j]-value[j]) > threshold_rad:
                count[j] += 1
                direction[j] *= -1
                extreme[j] = value[j]
    return count.tolist()


def _joint_metrics(q, lower, upper, kin, tolerance_rad):
    margin = np.minimum(q-lower, upper-q)
    urdf_margin = np.minimum(q-kin.lower_limits, kin.upper_limits-q)
    bad = np.any(margin < -tolerance_rad-BOUND_EPS, axis=1)
    urdf_bad = np.any(urdf_margin < -tolerance_rad-BOUND_EPS, axis=1)
    travel = np.sum(np.abs(np.diff(q, axis=0)), axis=0)
    return {
        "joint_min_rad": q.min(axis=0).tolist(),
        "joint_max_rad": q.max(axis=0).tolist(),
        "minimum_envelope_margin_rad": float(margin.min()),
        "minimum_urdf_limit_margin_rad": float(urdf_margin.min()),
        "observed_solver_tolerance_rad": tolerance_rad,
        "envelope_violation_rows": int(bad.sum()),
        "first_envelope_violation_row": int(np.flatnonzero(bad)[0]) if bad.any() else None,
        "urdf_violation_rows": int(urdf_bad.sum()),
        "cumulative_joint_travel_rad": travel.tolist(),
        "total_cumulative_joint_travel_rad": float(travel.sum()),
        "direction_reversals": direction_reversals(q),
        "reversal_excursion_threshold_deg": 1.,
    }


def audit_rows(rows, outcome, base, replay, *, kin=None, diagnostic_limits=None):
    """Audit parsed exact evidence; a diagnostic override can never accept a run."""
    report = {"schema": "low_elbow_recorded_audit_v1", "accepted": False,
              "reasons": [], "scope": "simulation posture only; separate contact/collision audit required; real hardware safety unverified"}
    reasons = report["reasons"]
    kin = OpenArmKinematics() if kin is None else kin
    config = outcome.get("posture_limits") if isinstance(outcome, dict) else None
    if config is None:
        reasons.append("missing_posture_limits")
    if diagnostic_limits is not None:
        reasons.append("diagnostic_limits_override")
        config = diagnostic_limits
    try:
        if not isinstance(config, dict):
            raise ValueError("posture_limits must be an object")
        limits = PostureLimits.from_dict(config, kin)
        base = _base_transform(base)
    except (ValueError, TypeError, KeyError, AttributeError):
        reasons.append("invalid_posture_limits_or_base")
        return report
    report["posture_limits"] = limits.to_dict()
    report["numerical_bound_epsilon"] = BOUND_EPS
    shoulder_z = float((base @ kin.link_transforms(np.zeros(7))["openarm_left_link1"])[2, 3])
    allowance = getattr(limits, "max_elbow_above_shoulder_m", 0.)
    contact_gap = getattr(limits, "min_contact_elbow_below_wrist_m", 0.)
    contact_gate_enabled = "min_contact_elbow_below_wrist_m" in config
    report["shoulder_z_m"] = shoulder_z
    report["declared_above_shoulder_allowance_m"] = allowance
    if limits.max_elbow_z_m > shoulder_z + allowance + BOUND_EPS:
        reasons.append("ceiling_exceeds_declared_shoulder_allowance")
    if not isinstance(rows, list) or len(rows) < 8:
        reasons.append("insufficient_trace")
        return report
    try:
        q = np.asarray([r["q"] for r in rows], float)
        commands = np.asarray([r["q_command"] for r in rows], float)
        times = np.asarray([r["time_s"] for r in rows], float)
        if q.shape != (len(rows), 7) or commands.shape != q.shape or times.shape != (len(rows),):
            raise ValueError("joint/time shape")
    except (ValueError, KeyError, TypeError):
        reasons.append("malformed_trace")
        return report
    if not all(np.isfinite(x).all() for x in (q, commands, times)):
        reasons.append("nonfinite_trace")
        return report
    if not np.allclose(times, np.arange(1, len(rows)+1)/240., atol=1e-7, rtol=0):
        reasons.append("timestamp_gap")
    contact = np.zeros(len(rows), dtype=bool)
    report["contact_elbow_wrist_gate"] = {"enabled": contact_gate_enabled,
        "phases": list(CONTACT_PHASES), "required_height_gap_m": contact_gap,
        "legacy_configuration_without_gate": not contact_gate_enabled}
    if contact_gate_enabled:
        phases = [row.get("phase") for row in rows]
        if any(value not in KNOWN_PHASES for value in phases):
            reasons.append("missing_or_invalid_phase_for_contact_posture")
        contact = np.asarray([value in CONTACT_PHASES for value in phases])
        if not contact.any():
            reasons.append("no_contact_phase_samples")
        report["contact_elbow_wrist_gate"]["physics_sample_count"] = int(contact.sum())
    positions = {}
    wrists = {}
    for label, values in (("measured", q), ("commanded", commands)):
        tolerance = limits.observed_joint_tolerance_rad if label == "measured" else 0.
        metric = _joint_metrics(values, np.asarray(limits.joint_lower_rad),
                                np.asarray(limits.joint_upper_rad), kin, tolerance)
        positions[label] = elbow_positions_world(kin, base, values)
        height = positions[label][:, 2]
        bad = height > limits.max_elbow_z_m + BOUND_EPS
        metric.update(max_elbow_z_m=float(height.max()),
                      max_elbow_row=int(height.argmax()),
                      elbow_height_violation_rows=int(bad.sum()),
                      first_elbow_height_violation_row=int(np.flatnonzero(bad)[0]) if bad.any() else None)
        report[label] = metric
        if metric["envelope_violation_rows"]:
            reasons.append(label+"_joint_envelope")
        if metric["urdf_violation_rows"]:
            reasons.append(label+"_urdf_joint_limits")
        if bad.any():
            reasons.append(label+"_elbow_height")
        if contact_gate_enabled:
            wrists[label] = landmark_positions_world(kin, base, values, "openarm_left_link7")
            gaps = wrists[label][:, 2] - height
            bad_contact = contact & (gaps < contact_gap - BOUND_EPS)
            metric.update(minimum_contact_elbow_below_wrist_m=float(gaps[contact].min()) if contact.any() else None,
                          contact_elbow_wrist_violation_rows=int(bad_contact.sum()),
                          first_contact_elbow_wrist_violation_row=int(np.flatnonzero(bad_contact)[0]) if bad_contact.any() else None)
            if bad_contact.any():
                reasons.append(label+"_contact_elbow_above_wrist_limit")
    try:
        poses = np.asarray(replay["poses"])
        indices = np.asarray(replay["row_indices"])
        paths = np.asarray(replay["paths"])
        if (poses.shape != (len(indices), len(paths), 7) or paths.ndim != 1
                or paths.dtype.kind not in "US" or indices.ndim != 1
                or indices.dtype.kind not in "iu" or not np.isfinite(poses).all()
                or not np.array_equal(indices, np.arange(7, len(rows), 8))):
            raise ValueError("replay shape, finiteness or exact recorded coverage")
        matches = [i for i, p in enumerate(paths.tolist())
                   if p.endswith("/"+limits.elbow_link)]
        if len(matches) != 1 or len(set(paths.tolist())) != len(paths):
            raise ValueError("missing or ambiguous elbow body")
        actual = poses[:, matches[0], :3]
        error = np.linalg.norm(actual-positions["measured"][indices], axis=1)
        z = actual[:, 2]
        bad = z > limits.max_elbow_z_m + BOUND_EPS
        report["actual_replay"] = {
            "sample_count": len(indices), "elbow_body_path": str(paths[matches[0]]),
            "max_elbow_z_m": float(z.max()),
            "max_elbow_physics_row": int(indices[z.argmax()]),
            "elbow_height_violation_samples": int(bad.sum()),
            "max_fk_position_error_m": float(error.max()),
            "fk_position_tolerance_m": REPLAY_POSITION_TOLERANCE_M,
            "sampling": "actual body world poses every 8 physics rows; all rows additionally checked by measured-joint FK",
        }
        if bad.any():
            reasons.append("actual_replay_elbow_height")
        if error.max() > REPLAY_POSITION_TOLERANCE_M:
            reasons.append("replay_fk_disagreement")
        if contact_gate_enabled:
            wrist_matches = [i for i, p in enumerate(paths.tolist())
                             if p.endswith("/openarm_left_link7")]
            if len(wrist_matches) != 1:
                raise ValueError("missing or ambiguous wrist body")
            actual_wrist = poses[:, wrist_matches[0], :3]
            wrist_error = np.linalg.norm(actual_wrist-wrists["measured"][indices], axis=1)
            recorded_contact = contact[indices]
            gaps = actual_wrist[:, 2] - z
            bad_contact = recorded_contact & (gaps < contact_gap-BOUND_EPS)
            report["actual_replay"].update(
                wrist_body_path=str(paths[wrist_matches[0]]),
                max_wrist_fk_position_error_m=float(wrist_error.max()),
                contact_sample_count=int(recorded_contact.sum()),
                minimum_contact_elbow_below_wrist_m=float(gaps[recorded_contact].min()) if recorded_contact.any() else None,
                contact_elbow_wrist_violation_samples=int(bad_contact.sum()))
            if contact.any() and not recorded_contact.any():
                reasons.append("no_actual_replay_contact_samples")
            if bad_contact.any():
                reasons.append("actual_replay_contact_elbow_above_wrist_limit")
            if wrist_error.max() > REPLAY_POSITION_TOLERANCE_M:
                reasons.append("replay_wrist_fk_disagreement")
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        reasons.append("malformed_or_incomplete_replay")
    if not isinstance(outcome, dict) or outcome.get("success") is not True or outcome.get("training_accepted") is not True:
        reasons.append("controller_did_not_accept")
    report["physics_steps"] = len(rows)
    report["accepted"] = not reasons
    return report


def audit_directory(path, *, diagnostic_limits=None, write=True):
    """Read/hash identical bytes, audit, recheck bindings, save posture_audit.json."""
    path = Path(path)
    artifacts = {name: (path/name).read_bytes() for name in ARTIFACTS}
    urdf_bytes = DEFAULT_URDF_PATH.read_bytes()
    kin = OpenArmKinematics()
    if DEFAULT_URDF_PATH.read_bytes() != urdf_bytes:
        raise ValueError("URDF changed while constructing kinematics")
    with np.load(io.BytesIO(artifacts["expert_replay.npz"]), allow_pickle=False) as loaded:
        replay = {name: loaded[name] for name in loaded.files}
    # Retain only posture inputs, not large unrelated contact-point records.
    # Iterating captured bytes avoids another full-file splitlines allocation.
    rows = []
    for line in io.BytesIO(artifacts["expert_trace.jsonl"]):
        row = json.loads(line)
        rows.append({**{key: row[key] for key in ("q", "q_command", "time_s")},
                     "phase": row.get("phase")})
    report = audit_rows(rows,
                        json.loads(artifacts["outcome.json"]),
                        json.loads(artifacts["world_from_left_base.json"]), replay,
                        kin=kin, diagnostic_limits=diagnostic_limits)
    report["artifact_sha256"] = {name: hashlib.sha256(artifacts[name]).hexdigest()
                                  for name in ARTIFACTS[:3]}
    report["source_sha256"] = {
        "kinematics_urdf": {"path": str(DEFAULT_URDF_PATH.resolve()),
                             "sha256": hashlib.sha256(urdf_bytes).hexdigest()},
        "world_from_left_base": {"path": str((path/"world_from_left_base.json").resolve()),
                                  "sha256": hashlib.sha256(artifacts["world_from_left_base.json"]).hexdigest()},
    }
    for name, data in artifacts.items():
        if (path/name).read_bytes() != data:
            report["reasons"].append("artifact_changed_during_audit:"+name)
    if DEFAULT_URDF_PATH.read_bytes() != urdf_bytes:
        report["reasons"].append("kinematics_changed_during_audit")
    report["accepted"] = not report["reasons"]
    if write:
        (path/"posture_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--diagnostic-limits", type=Path,
                        help="Compare an old run without accepting it or modifying its artifacts")
    parser.add_argument("--output", type=Path, help="Optional separate diagnostic report path")
    args = parser.parse_args()
    limits = json.loads(args.diagnostic_limits.read_text()) if args.diagnostic_limits else None
    report = audit_directory(args.episode, diagnostic_limits=limits,
                             write=limits is None and args.output is None)
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"accepted": report["accepted"], "reasons": report["reasons"],
                      "physics_steps": report.get("physics_steps")}))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
