import hashlib
import json

import numpy as np
import pytest

from sim.auscultation_training.audit_low_elbow import (
    ARTIFACTS, audit_directory, audit_rows, direction_reversals, elbow_positions_world,
    landmark_positions_world,
)
from sim.auscultation_training.kinematics import OpenArmKinematics
from sim.auscultation_training.posture import PostureLimits


def evidence(n=24):
    kin = OpenArmKinematics()
    base = np.eye(4)
    base[:3, :3] = np.diag([1., -1., -1.])
    base[:3, 3] = [.2, .1, .8]
    q = np.zeros((n, 7))
    q[:, 3] = .1
    limits = PostureLimits(.73, tuple(kin.lower_limits), tuple(kin.upper_limits))
    rows = [dict(q=values.tolist(), q_command=values.tolist(), time_s=(i+1)/240., target="R1")
            for i, values in enumerate(q)]
    indices = np.arange(7, n, 8)
    poses = np.zeros((len(indices), 1, 7))
    poses[:, 0, :3] = elbow_positions_world(kin, base, q)[indices]
    poses[:, 0, 3] = 1.
    replay = dict(poses=poses, row_indices=indices, paths=np.array(["/World/Robot/openarm_left_link4"]))
    outcome = dict(posture_limits=limits.to_dict(), success=True, training_accepted=True)
    # Exercise legacy traces that predate the contact wrist gate. Newly emitted
    # profiles explicitly carry the field, including when its value is zero.
    outcome["posture_limits"].pop("min_contact_elbow_below_wrist_m")
    return rows, outcome, base, replay


def test_batch_fk_matches_original_kinematics():
    kin = OpenArmKinematics()
    rng = np.random.default_rng(4)
    q = rng.uniform(kin.lower_limits, kin.upper_limits, (20, 7))
    base = np.eye(4)
    base[:3, 3] = [.3, -.8, .4]
    expected = [(base @ kin.link_transforms(x)["openarm_left_link4"])[:3, 3] for x in q]
    np.testing.assert_allclose(elbow_positions_world(kin, base, q), expected, atol=1e-12)


def test_accepts_complete_matching_low_elbow_evidence():
    report = audit_rows(*evidence())
    assert report["accepted"] and not report["reasons"]
    assert report["actual_replay"]["sample_count"] == 3
    assert report["measured"]["direction_reversals"] == [0]*7


def test_diagnostic_config_cannot_upgrade_missing_run_config():
    rows, outcome, base, replay = evidence()
    config = outcome.pop("posture_limits")
    report = audit_rows(rows, outcome, base, replay, diagnostic_limits=config)
    assert not report["accepted"]
    assert "missing_posture_limits" in report["reasons"]
    assert "diagnostic_limits_override" in report["reasons"]
    assert "actual_replay" in report


def test_override_cannot_hide_original_config():
    rows, outcome, base, replay = evidence()
    report = audit_rows(rows, outcome, base, replay, diagnostic_limits=outcome["posture_limits"])
    assert report["reasons"] == ["diagnostic_limits_override"]


@pytest.mark.parametrize("label,key", [("measured", "q"), ("commanded", "q_command")])
def test_one_unsampled_row_joint_violation_rejected(label, key):
    rows, outcome, base, replay = evidence()
    rows[10][key][0] = 2.
    report = audit_rows(rows, outcome, base, replay)
    assert not report["accepted"]
    assert label+"_joint_envelope" in report["reasons"]
    assert report[label]["first_envelope_violation_row"] == 10


def test_one_unsampled_height_violation_rejected():
    rows, outcome, base, replay = evidence()
    rows[10]["q"][1] = .15
    report = audit_rows(rows, outcome, base, replay)
    assert "measured_elbow_height" in report["reasons"]
    assert report["measured"]["elbow_height_violation_rows"] == 1


def test_actual_replay_height_gate_even_when_fk_passes():
    rows, outcome, base, replay = evidence()
    outcome["posture_limits"]["max_elbow_z_m"] = .71
    replay["poses"][0, 0, 2] = .7105
    report = audit_rows(rows, outcome, base, replay)
    assert "actual_replay_elbow_height" in report["reasons"]
    assert "measured_elbow_height" not in report["reasons"]


def test_replay_must_match_measured_fk_in_full_position():
    rows, outcome, base, replay = evidence()
    replay["poses"][0, 0, 0] += .003
    assert "replay_fk_disagreement" in audit_rows(rows, outcome, base, replay)["reasons"]


@pytest.mark.parametrize("bad", ["missing_frame", "duplicate_index", "missing_elbow", "nan", "noninteger_index"])
def test_replay_evidence_cannot_be_thinned_or_forged(bad):
    rows, outcome, base, replay = evidence()
    if bad == "missing_frame":
        replay["row_indices"] = replay["row_indices"][:-1]
        replay["poses"] = replay["poses"][:-1]
    elif bad == "duplicate_index":
        replay["row_indices"][1] = replay["row_indices"][0]
    elif bad == "missing_elbow":
        replay["paths"] = np.array(["/World/Robot/openarm_left_link5"])
    elif bad == "nan":
        replay["poses"][0, 0, 2] = np.nan
    else:
        replay["row_indices"] = replay["row_indices"].astype(float)
    assert "malformed_or_incomplete_replay" in audit_rows(rows, outcome, base, replay)["reasons"]


@pytest.mark.parametrize("key", ["q", "q_command", "time_s"])
def test_nonfinite_trace_fails_closed(key):
    rows, outcome, base, replay = evidence()
    if key == "time_s":
        rows[0][key] = np.nan
    else:
        rows[0][key][2] = np.inf
    assert audit_rows(rows, outcome, base, replay)["reasons"] == ["nonfinite_trace"]


def test_time_gap_and_failed_controller_reject():
    rows, outcome, base, replay = evidence()
    rows[10]["time_s"] += .01
    outcome["success"] = False
    report = audit_rows(rows, outcome, base, replay)
    assert "timestamp_gap" in report["reasons"]
    assert "controller_did_not_accept" in report["reasons"]


def test_reversals_accumulate_small_steps_but_ignore_small_jitter():
    degrees = np.r_[np.linspace(0, 3, 31), [2.9, 3.0, 2.8, 3.1], np.linspace(3.1, 0, 32), np.linspace(0, 2, 21)]
    q = np.zeros((len(degrees), 7))
    q[:, 0] = np.deg2rad(degrees)
    assert direction_reversals(q) == [2, 0, 0, 0, 0, 0, 0]


def test_directory_hashes_original_bytes_and_writes_report(tmp_path):
    rows, outcome, base, replay = evidence()
    (tmp_path/"expert_trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    (tmp_path/"outcome.json").write_text(json.dumps(outcome))
    (tmp_path/"world_from_left_base.json").write_text(json.dumps(base.tolist()))
    np.savez_compressed(tmp_path/"expert_replay.npz", **replay)
    report = audit_directory(tmp_path)
    assert report["accepted"]
    assert json.loads((tmp_path/"posture_audit.json").read_text()) == report
    assert report["artifact_sha256"] == {
        name: hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in ARTIFACTS[:3]}
    assert report["source_sha256"]["world_from_left_base"]["sha256"] == hashlib.sha256(
        (tmp_path/"world_from_left_base.json").read_bytes()).hexdigest()


def test_observed_solver_tolerance_is_disclosed_without_clipping():
    rows, outcome, base, replay = evidence()
    rows[10]["q"][3] = -.003
    report = audit_rows(rows, outcome, base, replay)
    assert report["accepted"]
    assert report["measured"]["minimum_urdf_limit_margin_rad"] == -.003
    assert report["measured"]["observed_solver_tolerance_rad"] == .005
    assert report["commanded"]["observed_solver_tolerance_rad"] == 0.
    rows[10]["q_command"][3] = -.003
    assert "commanded_joint_envelope" in audit_rows(rows, outcome, base, replay)["reasons"]


def test_observed_values_beyond_solver_tolerance_rejected():
    rows, outcome, base, replay = evidence()
    rows[10]["q"][3] = -.0051
    assert "measured_joint_envelope" in audit_rows(rows, outcome, base, replay)["reasons"]


def test_missing_config_without_override_fails_closed():
    rows, outcome, base, replay = evidence()
    outcome.pop("posture_limits")
    report = audit_rows(rows, outcome, base, replay)
    assert not report["accepted"] and "missing_posture_limits" in report["reasons"]


def test_config_cannot_increase_observed_tolerance():
    rows, outcome, base, replay = evidence()
    outcome["posture_limits"]["observed_joint_tolerance_rad"] = .1
    assert "invalid_posture_limits_or_base" in audit_rows(rows, outcome, base, replay)["reasons"]


def test_changed_base_during_audit_rejected(tmp_path, monkeypatch):
    from sim.auscultation_training import audit_low_elbow as module
    rows, outcome, base, replay = evidence()
    (tmp_path/"expert_trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    (tmp_path/"outcome.json").write_text(json.dumps(outcome))
    base_path = tmp_path/"world_from_left_base.json"
    base_path.write_text(json.dumps(base.tolist()))
    np.savez_compressed(tmp_path/"expert_replay.npz", **replay)
    original = module.audit_rows
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        base_path.write_text("[]")
        return result
    monkeypatch.setattr(module, "audit_rows", changed)
    report = module.audit_directory(tmp_path)
    assert not report["accepted"]
    assert "artifact_changed_during_audit:world_from_left_base.json" in report["reasons"]


def contact_evidence():
    rows, outcome, base, replay = evidence()
    kin = OpenArmKinematics()
    base[:3, :3] = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
    base[2, 3] = 1.376
    q = np.tile([0., 0., 0., 2.2, 0., 0., 0.], (len(rows), 1))
    for i, row in enumerate(rows):
        row.update(q=q[i].tolist(), q_command=q[i].tolist(),
                   phase=["CONTACT", "HOLD", "RETRACT"][i//8])
    outcome["posture_limits"].update(max_elbow_z_m=1.451,
        max_elbow_above_shoulder_m=.075, min_contact_elbow_below_wrist_m=.03)
    poses = np.zeros((3, 2, 7)); poses[:, :, 3] = 1.
    for j, link in enumerate(["openarm_left_link4", "openarm_left_link7"]):
        poses[:, j, :3] = landmark_positions_world(kin, base, q, link)[replay["row_indices"]]
    replay.update(poses=poses, paths=np.array(["/World/Robot/openarm_left_link4", "/World/Robot/openarm_left_link7"]))
    return rows, outcome, base, replay


def test_new_profile_checks_actual_contact_elbow_below_wrist():
    report = audit_rows(*contact_evidence())
    assert report["accepted"], report["reasons"]
    assert report["actual_replay"]["contact_sample_count"] == 3
    assert report["measured"]["minimum_contact_elbow_below_wrist_m"] >= .03


@pytest.mark.parametrize("phase", ["CONTACT", "HOLD", "RETRACT"])
@pytest.mark.parametrize("key,label", [("q", "measured"), ("q_command", "commanded")])
def test_one_contact_phase_row_low_wrist_rejected(phase, key, label):
    rows, outcome, base, replay = contact_evidence()
    rows[10][key][3] = .1
    rows[10]["phase"] = phase
    report = audit_rows(rows, outcome, base, replay)
    assert label+"_contact_elbow_above_wrist_limit" in report["reasons"]


def test_actual_contact_wrist_can_reject_even_if_joint_fk_passes():
    rows, outcome, base, replay = contact_evidence()
    # Set the required gap just below FK, then move only the recorded wrist
    # by 1 mm (below the separate 2 mm consistency tolerance).
    gap = replay["poses"][0, 1, 2]-replay["poses"][0, 0, 2]
    outcome["posture_limits"]["min_contact_elbow_below_wrist_m"] = gap-.0005
    replay["poses"][0, 1, 2] -= .001
    report = audit_rows(rows, outcome, base, replay)
    assert "actual_replay_contact_elbow_above_wrist_limit" in report["reasons"]
    assert "measured_contact_elbow_above_wrist_limit" not in report["reasons"]


def test_new_profile_cannot_omit_contact_phases_or_actual_wrist():
    rows, outcome, base, replay = contact_evidence()
    rows[0].pop("phase")
    replay["paths"][1] = "/World/Robot/openarm_left_link6"
    report = audit_rows(rows, outcome, base, replay)
    assert "missing_or_invalid_phase_for_contact_posture" in report["reasons"]
    assert "malformed_or_incomplete_replay" in report["reasons"]


def test_declared_above_shoulder_allowance_cannot_be_exceeded():
    rows, outcome, base, replay = contact_evidence()
    outcome["posture_limits"]["max_elbow_z_m"] += .001
    assert "ceiling_exceeds_declared_shoulder_allowance" in audit_rows(rows, outcome, base, replay)["reasons"]


def test_explicit_zero_gap_still_requires_contact_elbow_below_wrist():
    rows, outcome, base, replay = contact_evidence()
    outcome["posture_limits"]["min_contact_elbow_below_wrist_m"] = 0.
    rows[10]["q"][3] = .1
    replay["poses"][0, 1, 2] = replay["poses"][0, 0, 2]-.001
    report = audit_rows(rows, outcome, base, replay)
    assert report["contact_elbow_wrist_gate"]["enabled"]
    assert "measured_contact_elbow_above_wrist_limit" in report["reasons"]
    assert "actual_replay_contact_elbow_above_wrist_limit" in report["reasons"]


def test_absent_contact_gap_is_explicitly_marked_legacy():
    report = audit_rows(*evidence())
    assert report["accepted"]
    assert not report["contact_elbow_wrist_gate"]["enabled"]
    assert report["contact_elbow_wrist_gate"]["legacy_configuration_without_gate"]
