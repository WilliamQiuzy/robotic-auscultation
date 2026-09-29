"""CPU-only contract checks for truthful continuous-episode replay."""
from copy import deepcopy
import numpy as np
import pytest

from sim.auscultation_training.render_expert import (
    accepted_frame, camera_pose, playback_frames, point_dwell_manifest,
    point_records, recorded_pose_digest, validate_replay_timing,
)


def sequence():
    rows, points = [], []
    for name, position in [("R1", [.1, .8, 1.4]), ("L5", [-.1, .9, 1.0])]:
        point = {"target": name, "world_target_m": position,
                 "command_normal_world": [0., -1., .4], "success": True,
                 "training_accepted": True, "verified_front_contact_hold_s": 3.,
                 "observed_hold_s": 3., "start_physics_row": len(rows)}
        for phase, subphase, count, force, gap in [
            ("APPROACH", "transfer", 1920, 0., .06),
            ("CONTACT", "contact_cycle", 960, 0., .02),
            ("CONTACT", "contact_cycle", 240, 3.8, .005),
            ("HOLD", "contact_cycle", 720, 4., .005),
            ("RETRACT", "contact_cycle", 240, 2., .007),
            ("DONE", "return_home", 1920, 0., .06),
        ]:
            for tick in range(count):
                rows.append({"target": name, "time_s": (len(rows) + 1) / 240,
                             "phase": phase, "subphase": subphase, "lip_N": force,
                             "diaphragm_N": 0., "contact_rim_N": 0., "gap_m": gap,
                             "hold_elapsed_s": (tick + 1) / 240 if phase == "HOLD" else 0.,
                             "training_accepted": True})
        point["end_physics_row"] = len(rows)
        points.append(point)
    return rows, {"points": points, "training_accepted": True, "success": True}


def rendered_manifest(rows, plan):
    return [{"video_frame": i, "measurement_row": spec["measurement_row"],
             "target": spec["target"], "phase": rows[spec["measurement_row"]]["phase"],
             "playback_speed": spec["speed"],
             "simulation_time_s": rows[spec["measurement_row"]]["time_s"],
             "hold_elapsed_s": rows[spec["measurement_row"]]["hold_elapsed_s"]}
            for i, spec in enumerate(plan)]


def test_every_point_gets_its_own_exact_normal_speed_hold():
    rows, outcome = sequence()
    indices = np.arange(7, len(rows), 8)
    validate_replay_timing(rows, indices)
    plan = playback_frames(rows, int(indices[0]))
    manifests = point_dwell_manifest(rows, rendered_manifest(rows, plan), point_records(rows, outcome))
    assert set(manifests) == {"R1", "L5"}
    for name, point in manifests.items():
        assert point["hold_physics_steps"] == 720
        assert point["hold_trace_duration_s"] == 3.
        assert point["hold_frames_at_1x"] == 45
        assert point["hold_video_duration_s"] == 3.
        assert point["hold_elapsed_s"][-1] == 3.
        assert np.diff(point["hold_simulation_timestamps_s"]) == pytest.approx(np.full(44, 1 / 15))
        assert all(rows[i]["target"] == name for i in point["hold_measurement_rows"])
        assert point["hold_intervals"][0]["duration_s"] == 3.
    selected = [spec["measurement_row"] for spec in plan]
    assert all(b > a for a, b in zip(selected, selected[1:]))
    assert plan[-1]["measurement_row"] == len(rows) - 1
    for spec in plan:
        row = rows[spec["measurement_row"]]
        assert spec["target"] == row["target"]
        if row["subphase"] in ("transfer", "return_home"):
            assert spec["speed"] == 8.
        if row["phase"] in ("HOLD", "RETRACT"):
            assert spec["speed"] == 1.


def test_rejected_episode_cannot_claim_accepted_earlier_point():
    rows, outcome = sequence()
    point = outcome["points"][0]
    assert accepted_frame(rows[0], point, outcome)
    outcome["training_accepted"] = False
    assert not accepted_frame(rows[0], point, outcome)
    assert not point_records(rows, outcome)["R1"]["training_accepted"]
    outcome["training_accepted"] = True
    assert not accepted_frame(dict(rows[0], recovery=True), point, outcome)
    assert not accepted_frame(dict(rows[0], training_accepted=False), point, outcome)


@pytest.mark.parametrize("cap", [1., 2.])
def test_slow_posture_review_preserves_actual_hold_duration(cap):
    rows, outcome = sequence()
    plan = playback_frames(rows, 7, max_motion_speed=cap)
    assert max(frame["speed"] for frame in plan) <= cap
    records = point_dwell_manifest(rows, rendered_manifest(rows, plan), point_records(rows, outcome))
    assert all(point["hold_video_duration_s"] == 3. for point in records.values())
    assert all(frame["speed"] == 1. for frame in plan
               if rows[frame["measurement_row"]]["phase"] == "HOLD")


def test_motion_playback_cap_rejects_invalid_values():
    rows, _ = sequence()
    for cap in (0., .5, 9., float("nan")):
        with pytest.raises(ValueError, match="playback cap"):
            playback_frames(rows, 7, max_motion_speed=cap)


def test_single_point_outcome_and_camera_are_backward_compatible():
    rows, outcome = sequence()
    point = outcome["points"][0]
    single_rows = [row for row in rows if row["target"] == "R1"]
    points = point_records(single_rows, point)
    assert list(points) == ["R1"]
    assert accepted_frame({"target": "R1"}, point, point)
    before = deepcopy(point)
    eye, aim, focal = camera_pose("contact", point)
    assert np.isfinite(eye).all() and np.isfinite(aim).all() and focal == 28.
    shifted = dict(point, world_target_m=(np.asarray(point["world_target_m"]) + [1., 2., 3.]).tolist())
    shifted_eye, shifted_aim, _ = camera_pose("contact", shifted)
    assert shifted_eye - eye == pytest.approx([1., 2., 3.])
    assert shifted_aim - aim == pytest.approx([1., 2., 3.])
    assert point == before
    for normal in ([0., 0., 1.], [0., 0., -1.]):
        vertical_eye, vertical_aim, _ = camera_pose("contact", dict(point, command_normal_world=normal))
        assert np.isfinite(vertical_eye).all() and np.isfinite(vertical_aim).all()


def test_unknown_trace_target_and_bad_geometry_fail_closed():
    rows, outcome = sequence()
    with pytest.raises(ValueError, match="no outcome metadata"):
        point_records([dict(rows[0], target="R4")], outcome)
    invalid = deepcopy(outcome)
    invalid["points"][0]["world_target_m"] = [float("nan"), 0., 0.]
    with pytest.raises(ValueError, match="camera metadata"):
        point_records(rows, invalid)
    invalid = deepcopy(outcome)
    invalid["points"][0]["command_normal_world"] = [0., 0., 0.]
    with pytest.raises(ValueError, match="Zero camera normal"):
        point_records(rows, invalid)


def test_short_failed_hold_is_not_extended_to_three_seconds():
    rows, outcome = sequence()
    start = next(i for i, row in enumerate(rows) if row["phase"] == "HOLD")
    rows = rows[:start + 240]
    outcome["points"] = outcome["points"][:1]
    outcome["points"][0].update(success=False, training_accepted=False,
                                verified_front_contact_hold_s=1., observed_hold_s=1.)
    outcome.update(success=False, training_accepted=False)
    plan = playback_frames(rows, 7)
    summary = point_dwell_manifest(rows, rendered_manifest(rows, plan), point_records(rows, outcome))["R1"]
    assert summary["hold_frames_at_1x"] == 15
    assert summary["hold_video_duration_s"] == 1.
    assert summary["hold_elapsed_s"][-1] == 1.
    assert not summary["training_accepted"]


def test_recovery_is_not_accelerated_even_during_return_motion():
    rows, _ = sequence()
    rows = [dict(row, phase="RECOVERY", recovery=True) for row in rows[:240]]
    plan = playback_frames(rows, 7)
    assert all(spec["speed"] == 1 and spec["segment"] == "RECOVERY" for spec in plan)


def test_pose_hash_binds_exact_recorded_samples():
    poses = np.arange(5 * 2 * 7, dtype=np.float32).reshape(5, 2, 7)
    baseline = recorded_pose_digest(poses, [1, 3, 1])
    assert baseline["recorded_pose_indices"] == [1, 3]
    assert recorded_pose_digest(poses, [3, 1]) == baseline
    modified = poses.copy()
    modified[0, 0, 0] += 1
    assert recorded_pose_digest(modified, [1, 3]) == baseline
    modified[3, 0, 0] += 1
    assert recorded_pose_digest(modified, [1, 3]) != baseline


def learned_outcome(mode="hybrid_normal"):
    rows, outcome = sequence()
    outcome.update(controller_type="learned_bc", dataset_role="student_evaluation",
        expert_demonstration=False, contact_mode=mode,
        action_source_counts={"learned_bc":100,"supervisor_stationary_hold":20})
    return rows, outcome


def test_legacy_and_explicit_expert_are_never_labeled_learned():
    from sim.auscultation_training.render_expert import replay_presentation
    rows,outcome=sequence()
    for metadata in ({}, {"controller_type":"algorithmic_expert","dataset_role":"expert_demonstration","expert_demonstration":True}):
        presentation=replay_presentation(dict(outcome,**metadata),rows)
        assert presentation["replay_role"]=="expert"
        assert presentation["title"]=="ALGORITHMIC EXPERT"
        assert "learned" not in presentation["control_banner"].lower()
        assert presentation["filename_prefix"]=="expert"


def test_hybrid_learned_replay_discloses_normal_force_assistance_and_supervision():
    from sim.auscultation_training.render_expert import replay_presentation
    rows,outcome=learned_outcome()
    presentation=replay_presentation(outcome,rows)
    assert presentation["title"]=="LEARNED POLICY"
    assert "analytic normal-force assistance" in presentation["control_banner"]
    assert "stationary holds supervised" in presentation["control_banner"]
    assert "recovery" not in presentation["control_banner"]
    assert presentation["filename_prefix"]=="learned_bc"
    rows[0].update(recovery=True,phase="RECOVERY")
    assert "analytic fault recovery recorded" in replay_presentation(outcome,rows)["control_banner"]


def test_pure_bc_does_not_claim_normal_force_assistance():
    from sim.auscultation_training.render_expert import replay_presentation
    rows,outcome=learned_outcome("pure_bc")
    presentation=replay_presentation(outcome,rows)
    assert "pure BC joint actions" in presentation["control_banner"]
    assert "normal-force assistance" not in presentation["control_banner"]


@pytest.mark.parametrize("metadata", [
    {"controller_type":"learned_bc","dataset_role":"expert_demonstration","expert_demonstration":True},
    {"controller_type":"algorithmic_expert","dataset_role":"student_evaluation","expert_demonstration":False},
    {"controller_type":"unknown_controller"},
])
def test_inconsistent_controller_identity_cannot_be_rebranded(metadata):
    from sim.auscultation_training.render_expert import replay_presentation
    rows,outcome=sequence()
    with pytest.raises(ValueError,match="controller identity"):
        replay_presentation(dict(outcome,**metadata),rows)


def test_full_student_export_requires_bound_acceptance_and_preview_is_diagnostic():
    from sim.auscultation_training.render_expert import validate_student_export
    rows,outcome=learned_outcome()
    hashes={name:name for name in ("expert_trace.jsonl","control_commands.json","outcome.json")}
    audit={"accepted":True,"artifact_sha256":hashes.copy()}
    summary={"closed_loop_accepted":True,"controller_success":True,"independent_audit_accepted":True}
    assert validate_student_export(outcome,rows,False,audit,summary,hashes)
    audit["artifact_sha256"]["outcome.json"]="changed"
    with pytest.raises(ValueError,match="student recording is not accepted"):
        validate_student_export(outcome,rows,False,audit,summary,hashes)
    assert not validate_student_export(outcome,rows,True,audit,summary,hashes)
    audit["artifact_sha256"]=hashes.copy()
    outcome.update(success=False,training_accepted=False)
    with pytest.raises(ValueError,match="student recording is not accepted"):
        validate_student_export(outcome,rows,False,audit,summary,hashes)
    assert not validate_student_export(outcome,rows,True,audit,summary,hashes)


def test_student_recovery_or_negative_audit_cannot_export_success_even_with_true_outcome():
    from sim.auscultation_training.render_expert import validate_student_export
    rows,outcome=learned_outcome()
    hashes={name:name for name in ("expert_trace.jsonl","control_commands.json","outcome.json")}
    audit={"accepted":False,"artifact_sha256":hashes.copy()}
    summary={"closed_loop_accepted":True,"controller_success":True,"independent_audit_accepted":True}
    with pytest.raises(ValueError,match="student recording is not accepted"):
        validate_student_export(outcome,rows,False,audit,summary,hashes)
    audit["accepted"]=True; rows[0]["recovery"]=True
    with pytest.raises(ValueError,match="student recording is not accepted"):
        validate_student_export(outcome,rows,False,audit,summary,hashes)


def test_rejected_student_cli_refuses_export_before_loading_isaac(monkeypatch,tmp_path):
    import json
    import sys
    from sim.auscultation_training.render_expert import main
    rows,outcome=learned_outcome()
    outcome.update(success=False,training_accepted=False)
    episode=tmp_path/"rejected_student"; episode.mkdir()
    output=tmp_path/"video"
    for name in ("training_scene.usdc","expert_replay.npz"):
        (episode/name).write_bytes(b"placeholder; must never be loaded into simulator")
    (episode/"expert_trace.jsonl").write_text(json.dumps(rows[0])+"\n")
    (episode/"outcome.json").write_text(json.dumps(outcome))
    monkeypatch.setattr(sys,"argv",["render_expert","--episode",str(episode),"--output",str(output)])
    with pytest.raises(ValueError,match="student recording is not accepted"):
        main()
    assert not output.exists()


@pytest.mark.parametrize("filename,lines", [
    ("expert_trace.jsonl", True), ("outcome.json", False),
    ("independent_audit.json", False), ("evaluation_summary.json", False),
])
def test_json_labels_and_hash_use_the_same_captured_bytes(monkeypatch, tmp_path, filename, lines):
    import hashlib
    import json
    from pathlib import Path
    from sim.auscultation_training.render_expert import read_hashed_json
    path = tmp_path / filename
    original = json.dumps({"label": "captured"}).encode() + b"\n"
    replacement = json.dumps({"label": "later"}).encode() + b"\n"
    path.write_bytes(original)
    read_bytes = Path.read_bytes
    calls = []

    def capture_then_replace(current):
        data = read_bytes(current)
        if current == path:
            calls.append(current)
            path.write_bytes(replacement)
        return data

    monkeypatch.setattr(Path, "read_bytes", capture_then_replace)
    parsed, checksum = read_hashed_json(path, lines=lines)
    assert parsed == ([{"label": "captured"}] if lines else {"label": "captured"})
    assert checksum == hashlib.sha256(original).hexdigest()
    assert len(calls) == 1


def test_transient_expert_to_learned_relabel_cannot_borrow_expert_audit(monkeypatch, tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import sys
    from sim.auscultation_training.render_expert import main
    rows, expert = sequence()
    _, learned = learned_outcome()
    episode = tmp_path / "expert_source"
    episode.mkdir()
    output = tmp_path / "video"
    for name in ("training_scene.usdc", "expert_replay.npz"):
        (episode / name).write_bytes(b"placeholder; simulator must not load")
    (episode / "expert_trace.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (episode / "control_commands.json").write_text("[]")
    (episode / "outcome.json").write_text(json.dumps(expert))
    hashes = {name: hashlib.sha256((episode/name).read_bytes()).hexdigest()
              for name in ("expert_trace.jsonl", "control_commands.json", "outcome.json")}
    (episode / "independent_audit.json").write_text(json.dumps({"accepted": True, "artifact_sha256": hashes}))
    (episode / "evaluation_summary.json").write_text(json.dumps({
        "closed_loop_accepted": True, "controller_success": True, "independent_audit_accepted": True}))
    replacement = json.dumps(learned).encode()
    read_bytes, read_text = Path.read_bytes, Path.read_text
    # The file hashed by the old reader remains expert; a transient second
    # read supplies learned labels, then final disk hashes again look unchanged.
    monkeypatch.setattr(Path, "read_bytes", lambda path: replacement if path == episode/"outcome.json" else read_bytes(path))
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw: replacement.decode() if path == episode/"outcome.json" else read_text(path, *a, **kw))
    monkeypatch.setattr(sys, "argv", ["render_expert", "--episode", str(episode), "--output", str(output)])
    monkeypatch.setitem(sys.modules, "isaacsim", None)
    with pytest.raises(ValueError, match="student recording is not accepted"):
        main()
    assert not output.exists()


def test_input_change_after_parsing_is_rejected_before_output_or_isaac(monkeypatch, tmp_path):
    import json
    import sys
    from sim.auscultation_training import render_expert
    rows, outcome = sequence()
    episode = tmp_path / "expert_source"
    episode.mkdir()
    output = tmp_path / "video"
    for name in ("training_scene.usdc", "expert_replay.npz"):
        (episode / name).write_bytes(b"placeholder; simulator must not load")
    (episode / "expert_trace.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (episode / "outcome.json").write_text(json.dumps(outcome))
    point_records = render_expert.point_records

    def mutate_after_labels(captured_rows, captured_outcome):
        points = point_records(captured_rows, captured_outcome)
        (episode / "outcome.json").write_text(json.dumps(dict(outcome, training_accepted=False)))
        return points

    monkeypatch.setattr(render_expert, "point_records", mutate_after_labels)
    monkeypatch.setattr(sys, "argv", ["render_expert", "--episode", str(episode), "--output", str(output)])
    monkeypatch.setitem(sys.modules, "isaacsim", None)
    with pytest.raises(RuntimeError, match="Replay inputs changed"):
        render_expert.main()
    assert not output.exists()


@pytest.mark.parametrize("mode", ["hybrid_normal", "pure_bc"])
def test_v3_banner_discloses_planner_route_and_learned_progress(mode):
    from sim.auscultation_training.render_expert import replay_presentation
    rows, outcome = learned_outcome(mode)
    outcome.update(policy_version="v3", motion_parameterization="checked_route_learned_progress_v3",
        route_planning_learned=False, route_direction_source="checked planner waypoint minus previous command",
        route_progress_min_fraction=0.)
    presentation = replay_presentation(outcome, rows)
    assert "Planner route; learned progress; learned contact actions" in presentation["control_banner"]
    assert ("analytic normal-force assistance" in presentation["control_banner"]) == (mode == "hybrid_normal")
    assert presentation["route_planning_learned"] is False
    outcome["route_planning_learned"] = True
    with pytest.raises(ValueError, match="route/progress identity"):
        replay_presentation(outcome, rows)


def test_v3_missing_explicit_route_identity_cannot_use_v2_label():
    from sim.auscultation_training.render_expert import replay_presentation
    rows, outcome = learned_outcome()
    outcome["policy_version"] = "v3"
    with pytest.raises(ValueError, match="motion parameterization"):
        replay_presentation(outcome, rows)


def low_elbow_evidence():
    rows, outcome = sequence()
    outcome.update(controller_type="algorithmic_expert_low_elbow",
                   requested_points=[point["target"] for point in outcome["points"]])
    hashes = {name: name for name in ("expert_trace.jsonl", "control_commands.json", "outcome.json",
                                     "expert_replay.npz", "verification_summary.json")}
    mechanics = {"accepted": True, "artifact_sha256": {
        name: hashes[name] for name in ("expert_trace.jsonl", "control_commands.json", "outcome.json")}}
    posture = {"accepted": True, "artifact_sha256": {
        name: hashes[name] for name in ("expert_trace.jsonl", "expert_replay.npz", "outcome.json")}}
    evidence = {"independent_audit.json": mechanics, "posture_audit.json": posture,
                "verification_summary.json": {"accepted": True,
                    "audits": {"mechanics": deepcopy(mechanics), "posture": deepcopy(posture)}},
                "runner_summary.json": {"accepted": True, "child_returncode": 0,
                                         "evidence_sha256": hashes.copy()}}
    return rows, outcome, evidence, hashes


def test_low_elbow_is_clearly_algorithmic_expert_approaching_from_below():
    from sim.auscultation_training.render_expert import replay_presentation
    rows, outcome, _, _ = low_elbow_evidence()
    presentation = replay_presentation(outcome, rows)
    assert presentation["title"] == "LOW-ELBOW EXPERT"
    assert "APPROACH FROM BELOW" in presentation["control_banner"]
    assert "Algorithmic expert" in presentation["control_banner"]
    assert "learned" not in presentation["control_banner"].lower()
    assert presentation["replay_role"] == "expert"
    assert presentation["expert_demonstration"] is True
    assert presentation["filename_prefix"] == "low_elbow_expert"


def test_low_elbow_export_requires_both_audits_nested_verification_and_outer_runner():
    from sim.auscultation_training.render_expert import validate_low_elbow_export
    rows, outcome, evidence, hashes = low_elbow_evidence()
    assert validate_low_elbow_export(outcome, rows, False, evidence, hashes)
    for name in evidence:
        missing = deepcopy(evidence)
        del missing[name]
        with pytest.raises(ValueError, match="low-elbow recording is not accepted"):
            validate_low_elbow_export(outcome, rows, False, missing, hashes)
        assert not validate_low_elbow_export(outcome, rows, True, missing, hashes)
        rejected = deepcopy(evidence)
        rejected[name]["accepted"] = False
        with pytest.raises(ValueError, match="low-elbow recording is not accepted"):
            validate_low_elbow_export(outcome, rows, False, rejected, hashes)


@pytest.mark.parametrize("name", ["expert_trace.jsonl", "outcome.json", "expert_replay.npz",
                                  "control_commands.json", "verification_summary.json"])
def test_low_elbow_export_binds_acceptance_to_exact_loaded_artifacts(name):
    from sim.auscultation_training.render_expert import validate_low_elbow_export
    rows, outcome, evidence, hashes = low_elbow_evidence()
    hashes[name] = "different captured bytes"
    with pytest.raises(ValueError, match="low-elbow recording is not accepted"):
        validate_low_elbow_export(outcome, rows, False, evidence, hashes)


@pytest.mark.parametrize("kind", ["mechanics", "posture"])
def test_low_elbow_separate_audit_must_match_exact_nested_verified_report(kind):
    from sim.auscultation_training.render_expert import validate_low_elbow_export
    rows, outcome, evidence, hashes = low_elbow_evidence()
    evidence["verification_summary.json"]["audits"][kind]["accepted"] = False
    with pytest.raises(ValueError, match="low-elbow recording is not accepted"):
        validate_low_elbow_export(outcome, rows, False, evidence, hashes)


@pytest.mark.parametrize("failure", ["point", "sequence", "row", "recovery", "posture", "child"])
def test_low_elbow_failed_record_cannot_be_exported_as_accepted(failure):
    from sim.auscultation_training.render_expert import validate_low_elbow_export
    rows, outcome, evidence, hashes = low_elbow_evidence()
    if failure == "point":
        outcome["points"][0]["success"] = False
    elif failure == "sequence":
        outcome["requested_points"].append("R2")
    elif failure == "row":
        rows[0]["training_accepted"] = False
    elif failure == "recovery":
        rows[0]["recovery"] = True
    elif failure == "posture":
        rows[0]["posture_violations"] = ["elbow_height"]
    else:
        evidence["runner_summary.json"]["child_returncode"] = 1
    assert not validate_low_elbow_export(outcome, rows, True, evidence, hashes)
    with pytest.raises(ValueError, match="low-elbow recording is not accepted"):
        validate_low_elbow_export(outcome, rows, False, evidence, hashes)


def test_low_elbow_cli_rejects_missing_audits_before_loading_isaac(monkeypatch, tmp_path):
    import json
    import sys
    from sim.auscultation_training.render_expert import main
    rows, outcome, _, _ = low_elbow_evidence()
    episode = tmp_path / "low_elbow_episode"
    episode.mkdir()
    output = tmp_path / "video"
    for name in ("training_scene.usdc", "expert_replay.npz"):
        (episode / name).write_bytes(b"placeholder; must never load simulator")
    (episode / "expert_trace.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (episode / "outcome.json").write_text(json.dumps(outcome))
    monkeypatch.setattr(sys, "argv", ["render_expert", "--episode", str(episode), "--output", str(output)])
    monkeypatch.setitem(sys.modules, "isaacsim", None)
    with pytest.raises(ValueError, match="low-elbow recording is not accepted"):
        main()
    assert not output.exists()
