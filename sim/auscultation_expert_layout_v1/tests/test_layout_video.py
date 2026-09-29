"""CPU-only contract checks for truthful continuous-episode replay."""
from copy import deepcopy
import numpy as np
import pytest

from sim.auscultation_expert_layout_v1.render_expert import (
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
            assert spec["speed"] == 2.
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
    from sim.auscultation_expert_layout_v1.render_expert import replay_presentation
    rows,outcome=sequence()
    for metadata in ({}, {"controller_type":"algorithmic_expert","dataset_role":"expert_demonstration","expert_demonstration":True}):
        presentation=replay_presentation(dict(outcome,**metadata),rows)
        assert presentation["replay_role"]=="expert"
        assert presentation["title"]=="ALGORITHMIC EXPERT"
        assert "learned" not in presentation["control_banner"].lower()
        assert presentation["filename_prefix"]=="expert"


def test_hybrid_learned_replay_discloses_normal_force_assistance_and_supervision():
    from sim.auscultation_expert_layout_v1.render_expert import replay_presentation
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
    from sim.auscultation_expert_layout_v1.render_expert import replay_presentation
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
    from sim.auscultation_expert_layout_v1.render_expert import replay_presentation
    rows,outcome=sequence()
    with pytest.raises(ValueError,match="controller identity"):
        replay_presentation(dict(outcome,**metadata),rows)


def test_full_student_export_requires_bound_acceptance_and_preview_is_diagnostic():
    from sim.auscultation_expert_layout_v1.render_expert import validate_student_export
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
    from sim.auscultation_expert_layout_v1.render_expert import validate_student_export
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
    from sim.auscultation_expert_layout_v1.render_expert import main
    rows,outcome=learned_outcome()
    outcome.update(success=False,training_accepted=False)
    episode=tmp_path/"rejected_student"; episode.mkdir()
    output=tmp_path/"video"
    for name in ("training_scene.usdc","expert_replay.npz"):
        (episode/name).write_bytes(b"placeholder; must never be loaded into simulator")
    (episode/"expert_trace.jsonl").write_text(json.dumps(rows[0])+"\n")
    (episode/"outcome.json").write_text(json.dumps(outcome))
    monkeypatch.setattr(sys,"argv",["render_expert","--episode",str(episode),"--output",str(output)])
    with pytest.raises(ValueError,match="independently audited algorithmic low-elbow expert"):
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
    from sim.auscultation_expert_layout_v1.render_expert import read_hashed_json
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
    from sim.auscultation_expert_layout_v1.render_expert import main
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
    with pytest.raises(ValueError, match="independently audited algorithmic low-elbow expert"):
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
    from sim.auscultation_expert_layout_v1.render_expert import replay_presentation
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
    from sim.auscultation_expert_layout_v1.render_expert import replay_presentation
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
    from sim.auscultation_expert_layout_v1.render_expert import replay_presentation
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
    from sim.auscultation_expert_layout_v1.render_expert import validate_low_elbow_export
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
    from sim.auscultation_expert_layout_v1.render_expert import validate_low_elbow_export
    rows, outcome, evidence, hashes = low_elbow_evidence()
    hashes[name] = "different captured bytes"
    with pytest.raises(ValueError, match="low-elbow recording is not accepted"):
        validate_low_elbow_export(outcome, rows, False, evidence, hashes)


@pytest.mark.parametrize("kind", ["mechanics", "posture"])
def test_low_elbow_separate_audit_must_match_exact_nested_verified_report(kind):
    from sim.auscultation_expert_layout_v1.render_expert import validate_low_elbow_export
    rows, outcome, evidence, hashes = low_elbow_evidence()
    evidence["verification_summary.json"]["audits"][kind]["accepted"] = False
    with pytest.raises(ValueError, match="low-elbow recording is not accepted"):
        validate_low_elbow_export(outcome, rows, False, evidence, hashes)


@pytest.mark.parametrize("failure", ["point", "sequence", "row", "recovery", "posture", "child"])
def test_low_elbow_failed_record_cannot_be_exported_as_accepted(failure):
    from sim.auscultation_expert_layout_v1.render_expert import validate_low_elbow_export
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
    from sim.auscultation_expert_layout_v1.render_expert import main
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


def test_camera_view_options_are_mutually_exclusive_and_speed_cap_is_two():
    from sim.auscultation_expert_layout_v1.render_expert import argument_parser
    parser = argument_parser()
    assert parser.parse_args([]).motion_speed == 2
    for args in (['--robotcam-view', '--posture-view'], ['--motion-speed', '4']):
        with pytest.raises(SystemExit):
            parser.parse_args(args)


@pytest.mark.parametrize('sized_default', [True, False])
def test_font_fallback_survives_missing_linux_fonts(sized_default):
    from sim.auscultation_expert_layout_v1.render_expert import annotation_font

    class MissingFonts:
        def truetype(self, path, size):
            raise OSError('not installed')

        def load_default(self, **kwargs):
            if not sized_default and kwargs:
                raise TypeError('old Pillow')
            return 'font object'

    font, source = annotation_font(MissingFonts(), 18)
    assert font == 'font object'
    assert 'Pillow bundled' in source


def accepted_video_fixture(tmp_path, view_mode, *, scene_binding=True):
    """Create synthetic evidence, never a physics run or rendered demonstration."""
    import hashlib
    import io
    import json
    from sim.auscultation_expert_layout_v1.render_expert import (
        front_force, recorded_schedule_digest,
    )
    template_rows, template_outcome, _, _ = low_elbow_evidence()
    rows, points = [], []
    for level in range(1, 6):
        for side in ('R', 'L'):
            name = side + str(level)
            point = dict(template_outcome['points'][0], target=name)
            points.append(point)
            for row in template_rows:
                if row['target'] == 'R1':
                    rows.append(dict(row, target=name, time_s=(len(rows) + 1) / 240))
    outcome = dict(template_outcome, points=points, requested_points=[point['target'] for point in points])
    scene_bytes = b'synthetic test fixture; not USD'
    if scene_binding:
        outcome['training_scene_sha256'] = hashlib.sha256(scene_bytes).hexdigest()
    episode = tmp_path / view_mode
    episode.mkdir()
    objects = episode / 'input_snapshot/objects'
    objects.mkdir(parents=True)
    camera = {'world_from_optical': np.eye(4).tolist(), 'focal_length': 3.25,
              'horizontal_aperture': 5.45, 'vertical_aperture': 3.1,
              'horizontal_aperture_offset': 0., 'vertical_aperture_offset': 0.,
              'clipping_range': [.1, 100000.], 'fx': 3.25 / 5.45 * 1920,
              'fy': 3.25 / 3.1 * 1080, 'cx': 960., 'cy': 540., 'width': 1920, 'height': 1080}
    reference = json.dumps({'camera': camera}).encode()
    object_sha = hashlib.sha256(reference).hexdigest()
    (objects / object_sha).write_bytes(reference)
    snapshot = json.dumps({'files': {'source/scene/layout_inputs': {'sha256': object_sha}}}).encode()
    (episode / 'input_snapshot/manifest.json').write_bytes(snapshot)
    snapshot_sha = hashlib.sha256(snapshot).hexdigest()
    outcome['input_snapshot_manifest_sha256'] = snapshot_sha
    indices = np.arange(7, len(rows), 8)
    poses = np.zeros((len(indices), 1, 7), dtype=np.float32)
    poses[:, 0, 0] = np.arange(len(indices)) / 10000
    poses[:, 0, 3] = 1.
    archive = io.BytesIO()
    np.savez(archive, poses=poses, row_indices=indices, paths=np.array(['/World/TestBody']))
    raw = {'expert_trace.jsonl': ('\n'.join(json.dumps(row) for row in rows) + '\n').encode(),
           'expert_replay.npz': archive.getvalue(), 'outcome.json': json.dumps(outcome).encode(),
           'control_commands.json': b'[]', 'training_scene.usdc': scene_bytes}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    mechanics = {'accepted': True, 'artifact_sha256': {name: hashes[name] for name in (
        'expert_trace.jsonl', 'control_commands.json', 'outcome.json')}}
    posture = {'accepted': True, 'artifact_sha256': {name: hashes[name] for name in (
        'expert_trace.jsonl', 'expert_replay.npz', 'outcome.json')}}
    evidence = {'independent_audit.json': mechanics, 'posture_audit.json': posture,
                'verification_summary.json': {'accepted': True, 'audits': {'mechanics': mechanics, 'posture': posture}}}
    for name, value in evidence.items():
        raw[name] = json.dumps(value).encode()
        hashes[name] = hashlib.sha256(raw[name]).hexdigest()
    evidence['runner_summary.json'] = {'accepted': True, 'child_returncode': 0, 'evidence_sha256': hashes.copy()}
    raw['runner_summary.json'] = json.dumps(evidence['runner_summary.json']).encode()
    hashes['runner_summary.json'] = hashlib.sha256(raw['runner_summary.json']).hexdigest()
    for name, data in raw.items():
        (episode / name).write_bytes(data)
    plan = ([{'measurement_row': int(indices[0]), 'speed': 0, 'shot': 'overview'}] * 30
            + playback_frames(rows, int(indices[0]), max_motion_speed=2)
            + [{'measurement_row': len(rows) - 1, 'speed': 0, 'shot': 'overview'}] * 30)
    frames = []
    for index, spec in enumerate(plan):
        r = spec['measurement_row']
        row = rows[r]
        k = int(np.searchsorted(indices, r, side='right') - 1)
        frames.append({'video_frame': index, 'measurement_row': r, 'playback_speed': spec['speed'],
            'shot': view_mode if view_mode != 'report' else spec['shot'], 'pose_index': k,
            'pose_row': int(indices[k]), 'pose_age_s': row['time_s'] - rows[int(indices[k])]['time_s'],
            'target': row['target'], 'phase': row['phase'], 'simulation_time_s': row['time_s'],
            'hold_elapsed_s': row['hold_elapsed_s'], 'physical_front_N': front_force(row),
            'membrane_N': row['diaphragm_N'], 'training_accepted': True})
    point_manifest = point_dwell_manifest(rows, frames, point_records(rows, outcome), 15)
    for target, point in point_manifest.items():
        point.update(recorded_pose_digest(poses, [f['pose_index'] for f in frames if f['target'] == target]))
        point['source_trace_sha256'] = hashes['expert_trace.jsonl']
        point['source_replay_sha256'] = hashes['expert_replay.npz']
    manifest = {'preview': False, 'inputs_sha256': hashes, 'camera_reference': camera,
        'camera_reference_inputs_sha256': {'input_snapshot/manifest.json': snapshot_sha,
                                           'input_snapshot/objects/' + object_sha: object_sha},
        'view_mode': view_mode, 'robotcam_view': view_mode == 'robotcam', 'posture_view': view_mode == 'posture',
        'render_camera_path': '/World/RobotCam' if view_mode == 'robotcam' else '/World/RecordedReplayReportCamera',
        'resolution': [1920, 1080] if view_mode == 'robotcam' else [1280, 720], 'fps': 15,
        'frame_count': len(frames), 'frames': frames, 'motion_playback_speed_cap': 2,
        'physics_timeline_mode': 'stopped_static_pose_replay',
        'pose_application': 'recorded_raw_rigid_body_transforms_no_interpolation',
        'playback_schedule_sha256': recorded_schedule_digest(frames),
        'recorded_poses': recorded_pose_digest(poses, [f['pose_index'] for f in frames]),
        'points': point_manifest, 'hold_frames_at_1x': 450}
    for flag in ('training_accepted', 'source_layers_unchanged', 'physics_timeline_stopped',
                 'original_rgbd_camera_unchanged', 'camera_pose_optics_verified_every_frame'):
        manifest[flag] = True
    return episode, manifest


@pytest.mark.parametrize('view_mode', ['robotcam', 'posture'])
def test_both_views_validate_complete_exact_ten_point_recordings(tmp_path, view_mode):
    from sim.auscultation_expert_layout_v1.validate_video import validate_manifest
    episode, manifest = accepted_video_fixture(tmp_path, view_mode)
    points, _, _ = validate_manifest(episode, manifest)
    assert len(points) == 10
    assert sum(p['hold_video_duration_s'] for p in points.values()) == 30.
    for key, value in [('render_camera_path', '/World/WrongCamera'), ('resolution', [640, 480]),
                       ('playback_schedule_sha256', 'incorrect'), ('training_accepted', False)]:
        altered = dict(manifest, **{key: value})
        with pytest.raises(ValueError):
            validate_manifest(episode, altered)
    for frame_key, value in [('pose_index', 9), ('playback_speed', 8), ('physical_front_N', 100),
                              ('hold_elapsed_s', 10), ('shot', 'other_view')]:
        altered = deepcopy(manifest)
        altered['frames'][0][frame_key] = value
        with pytest.raises(ValueError):
            validate_manifest(episode, altered)
    reference = next(name for name in manifest['camera_reference_inputs_sha256'] if '/objects/' in name)
    (episode / reference).write_text('{}')
    with pytest.raises(ValueError, match='Camera reference object changed'):
        validate_manifest(episode, manifest)


def test_validator_uses_actual_codec_metadata_and_decodes_all_frames(tmp_path):
    import shutil
    import subprocess
    from sim.auscultation_expert_layout_v1.validate_video import probe_and_decode
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('media CLI dependencies unavailable')
    # A two-frame solid-color codec fixture; no Isaac/scene rendering occurs.
    video = tmp_path / 'codec_fixture.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', '16x16', '-r', '15', '-i', '-', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', str(video)], input=bytes(16 * 16 * 3 * 2), check=True)
    manifest = {'frame_count': 2, 'resolution': [16, 16], 'fps': 15}
    assert probe_and_decode(video, manifest)['nb_read_frames'] == '2'
    with pytest.raises(ValueError, match='frame count mismatch'):
        probe_and_decode(video, dict(manifest, frame_count=3))


@pytest.mark.parametrize('failure', ['scene_replaced', 'binding_missing'])
def test_renderer_and_validator_refuse_scene_provenance_gaps_before_isaac(tmp_path, monkeypatch, failure):
    import hashlib
    import sys
    from sim.auscultation_expert_layout_v1.render_expert import main
    from sim.auscultation_expert_layout_v1.validate_video import validate_manifest
    episode, manifest = accepted_video_fixture(tmp_path, 'robotcam', scene_binding=failure != 'binding_missing')
    if failure == 'scene_replaced':
        replacement = b'old patient and stool geometry, with unchanged RobotCam'
        (episode / 'training_scene.usdc').write_bytes(replacement)
        # A new renderer/video manifest can hash the replaced file consistently;
        # only the accepted outcome's original scene hash detects this substitution.
        manifest['inputs_sha256']['training_scene.usdc'] = hashlib.sha256(replacement).hexdigest()
    with pytest.raises(ValueError, match='audited outcome training_scene_sha256'):
        validate_manifest(episode, manifest)
    output = tmp_path / 'rejected_video'
    monkeypatch.setattr(sys, 'argv', ['render_expert', '--episode', str(episode),
                                    '--output', str(output), '--robotcam-view'])
    monkeypatch.setitem(sys.modules, 'isaacsim', None)
    with pytest.raises(ValueError, match='audited outcome training_scene_sha256'):
        main()
    assert not output.exists()
