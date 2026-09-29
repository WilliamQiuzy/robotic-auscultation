"""Software fixtures only: none of these samples establish physical success."""
import copy
import hashlib
import importlib.util
import json

import numpy as np
import pytest


def test_control_stage_must_match_audited_physics(tmp_path):
    from sim.auscultation_training.dataset import load_episode
    episode = fixture_episode(tmp_path/'episode')
    mutate_controls(episode, lambda rows: rows[0].update(phase='HOLD',subphase='contact_cycle'))
    with pytest.raises(ValueError,match='stage'):
        load_episode(episode)


def test_passing_student_evaluation_is_not_expert_data(tmp_path):
    from sim.auscultation_training.dataset import load_episode
    episode = fixture_episode(tmp_path/'episode')
    path = episode/'outcome.json'
    value = json.loads(path.read_text())
    value['dataset_role'] = 'student_evaluation'
    path.write_text(json.dumps(value))
    bind_audit(episode)
    with pytest.raises(ValueError,match='expert demonstration'):
        load_episode(episode)


def test_nonfinite_final_optimizer_result_rejected(monkeypatch):
    import torch
    from sim.auscultation_training.policy import BoundedJointPolicy
    from sim.auscultation_training.train_bc import fit_supervised
    torch.manual_seed(1)
    policy = BoundedJointPolicy(4)
    original_step = torch.optim.Adam.step
    def corrupt_final_update(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            optimizer.param_groups[0]['params'][0].fill_(float('nan'))
        return result
    monkeypatch.setattr(torch.optim.Adam,'step',corrupt_final_update)
    x = np.ones((4,4),dtype=np.float32)
    y = np.full((4,7),.01,dtype=np.float32)
    with pytest.raises(ValueError,match='nonfinite'):
        fit_supervised(policy,x,y,epochs=1,batch_size=4,learning_rate=.001)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_episode(path, scenario="fixture", seed=0, count=8):
    path.mkdir()
    rows = []
    previous = np.zeros(7)
    for i in range(count):
        action = np.array([.001 * (seed + 1), 0, 0, 0, 0, 0, 0])
        rows.append({"time_s": i * .05, "target": "R1", "phase": "APPROACH",
            "subphase": "transfer", "recovery": False, "training_accepted": True,
            "physics_row": i * 12, "physics_row_end_exclusive": (i+1)*12,
            "executed_physics_steps": 12, "command_block_complete": True,
            "action_duration_s": .05, "previous_q_command": previous.tolist(),
            "q_command": (previous+action).tolist(), "action_joint_delta_rad": action.tolist(),
            "action_interpolation": "physics-rate command rows are authoritative",
            "observation": {"q": previous.tolist(), "dq": (action/.05 if i else np.zeros(7)).tolist(),
                "tcp_base_m": [0., 0., 1.], "goal_relative_base_m": [i*.001, 0, 0],
                "normal_base": [0., 0., -1.], "hall_N": 0., "compression_m": 0.,
                "tcp_rotation_base": np.eye(3).tolist(),
                "target_rotation_base": np.eye(3).tolist()}})
        previous += action
    (path / "control_commands.json").write_text(json.dumps(rows))
    trace = []
    for row in rows:
        for step in range(1, 13):
            command = np.asarray(row["previous_q_command"]) + step/12 * np.asarray(row["action_joint_delta_rad"])
            trace.append({"time_s": row["time_s"] + step/240, "target": row["target"],
                "phase": row["phase"], "subphase": row["subphase"], "q_command": command.tolist(),
                "q": command.tolist(), "dq": (np.asarray(row["action_joint_delta_rad"])/.05).tolist(),
                "hall_N": 0., "compression_m": 0., "tcp_base_m": [0., 0., 1.],
                "tcp_rotation_base": np.eye(3).tolist(), "software_fixture": True})
    write_trace(path, trace)
    (path / "outcome.json").write_text(json.dumps({"success": True, "training_accepted": True,
        "scenario_id": scenario, "seed": seed, "control_dt_s": .05,
        "points": [{"target": "R1", "success": True, "training_accepted": True}]}))
    bind_audit(path)
    return path


def write_trace(path, trace):
    (path / "expert_trace.jsonl").write_text("".join(json.dumps(row)+"\n" for row in trace))


def read_trace(path):
    return [json.loads(line) for line in (path / "expert_trace.jsonl").read_text().splitlines()]


def fixture_metadata(path, value):
    trace = read_trace(path)
    trace[0]["fixture_id"] = value
    write_trace(path, trace)
    bind_audit(path)


def bind_audit(path):
    (path / "independent_audit.json").write_text(json.dumps({"accepted": True,
        "artifact_sha256": {name: sha(path / name) for name in
            ("control_commands.json", "outcome.json", "expert_trace.jsonl")}}))


def mutate_controls(path, change):
    rows = json.loads((path / "control_commands.json").read_text())
    change(rows)
    (path / "control_commands.json").write_text(json.dumps(rows))
    bind_audit(path)


def test_bc_modules_exist():
    for name in ("dataset", "policy", "train_bc"):
        assert importlib.util.find_spec("sim.auscultation_training." + name) is not None


def test_episode_rejects_unbound_or_rejected_independent_audit(tmp_path):
    from sim.auscultation_training.dataset import load_episode
    episode = fixture_episode(tmp_path / "episode")
    load_episode(episode)
    (episode / "expert_trace.jsonl").write_text("changed trace\n")
    with pytest.raises(ValueError, match="fingerprint"):
        load_episode(episode)
    bind_audit(episode)
    report = json.loads((episode / "independent_audit.json").read_text())
    report["accepted"] = False
    (episode / "independent_audit.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="independent audit"):
        load_episode(episode)


@pytest.mark.parametrize("change,reason", [
    (lambda r: r[0]["observation"].pop("target_rotation_base"), "target_rotation_base"),
    (lambda r: r[0].update(recovery=True), "recovery"),
    (lambda r: r[0].update(action_joint_delta_rad=[.02]*7), "bound"),
    (lambda r: r[0].update(command_block_complete=False), "incomplete"),
    (lambda r: r[0].update(action_duration_s=.1), "duration"),
    (lambda r: r[1].update(previous_q_command=[0.]*7), "continuity"),
    (lambda r: r[0]["observation"].update(hall_N=float("nan")), "nonfinite"),
    (lambda r: r[0].update(subphase="unknown"), "stage"),
])
def test_episode_fails_closed_on_invalid_actor_rows(tmp_path, change, reason):
    from sim.auscultation_training.dataset import load_episode
    episode = fixture_episode(tmp_path / "episode")
    mutate_controls(episode, change)
    with pytest.raises(ValueError, match=reason):
        load_episode(episode)


def test_actor_has_no_contact_truth_and_history_resets_on_target_change(tmp_path):
    from sim.auscultation_training.dataset import load_episode, encode_frame, FRAME_FEATURES
    episode = fixture_episode(tmp_path / "episode")
    def change(rows):
        rows[3]["target"] = "L1"
        for row in rows[3:]:
            row["target"] = "L1"
        rows[0]["observation"]["physx_force_N"] = 99999
        rows[0]["diaphragm_N"] = 99999
    mutate_controls(episode, change)
    trace = read_trace(episode)
    for row in trace[3*12:]:
        row["target"] = "L1"
    write_trace(episode, trace)
    bind_audit(episode)
    loaded = load_episode(episode)
    rows = json.loads((episode / "control_commands.json").read_text())
    clean = copy.deepcopy(rows[0]); clean["observation"].pop("physx_force_N"); clean.pop("diaphragm_N")
    np.testing.assert_array_equal(encode_frame(rows[0]), encode_frame(clean))
    assert not any("physx" in x or "diaphragm" in x for x in FRAME_FEATURES)
    assert loaded.observations.shape == (8, 4, len(FRAME_FEATURES))
    np.testing.assert_array_equal(loaded.observations[3], np.tile(encode_frame(rows[3]), (4, 1)))


def test_split_keeps_entire_scenario_seed_group_and_scaler_uses_training_only(tmp_path):
    from sim.auscultation_training.dataset import build_dataset
    episodes = [fixture_episode(tmp_path / str(i), seed=i//2) for i in range(6)]
    # Equal scenario/seed repetitions have different exact trace bytes to avoid duplicate data.
    for i, path in enumerate(episodes):
        fixture_metadata(path, i)
    output = tmp_path / "dataset"
    manifest = build_dataset(episodes, output, split_seed=12, evaluation_fraction=.34)
    arrays = np.load(output / "dataset.npz")
    train, evaluation = arrays["split"] == 0, arrays["split"] == 1
    assert train.any() and evaluation.any()
    for episode_id in np.unique(arrays["episode_index"]):
        assert len(set(arrays["split"][arrays["episode_index"] == episode_id])) == 1
    groups = {}
    for item in manifest["episodes"]:
        groups.setdefault(item["group_id"], set()).add(item["split"])
    assert all(len(x) == 1 for x in groups.values())
    np.testing.assert_allclose(manifest["scaler"]["mean"], arrays["observations"][train].reshape(train.sum(), -1).mean(0), atol=1e-7)
    assert manifest["action_interpolation"]["student"] == "linear over 12 physics steps"


def test_one_group_or_duplicate_episode_cannot_make_heldout_set(tmp_path):
    from sim.auscultation_training.dataset import build_dataset
    episode = fixture_episode(tmp_path / "episode")
    with pytest.raises(ValueError, match="two.*group"):
        build_dataset([episode], tmp_path / "one")
    with pytest.raises(ValueError, match="duplicate"):
        build_dataset([episode, episode], tmp_path / "duplicate")


def test_shared_policy_is_bounded_and_optimizer_can_overfit_micro_fixture():
    import torch
    from sim.auscultation_training.policy import BoundedJointPolicy
    from sim.auscultation_training.train_bc import fit_supervised
    torch.set_num_threads(1)
    torch.manual_seed(3)
    x = torch.randn(12, 24)
    y = .003 * torch.tanh(x[:, :7])
    policy = BoundedJointPolicy(24, hidden=(32, 32))
    result = fit_supervised(policy, x.numpy(), y.numpy(), epochs=150, batch_size=12,
                            learning_rate=.01, seed=3)
    assert result[-1] < result[0] * .02
    assert torch.max(torch.abs(policy(torch.randn(40, 24)*100))).item() <= .0125 + 1e-9


def test_training_cli_writes_reproducible_manifest_and_checkpoint(tmp_path):
    from sim.auscultation_training.dataset import build_dataset
    from sim.auscultation_training.train_bc import main
    episodes = [fixture_episode(tmp_path / str(i), seed=i) for i in range(3)]
    for i, path in enumerate(episodes):
        fixture_metadata(path, i)
    build_dataset(episodes, tmp_path / "data")
    assert main([str(tmp_path / "data"), "--output", str(tmp_path / "train"), "--epochs", "2", "--seed", "5"]) == 0
    report = json.loads((tmp_path / "train" / "run_manifest.json").read_text())
    assert report["device"] == "cpu" and report["seed"] == 5
    assert report["closed_loop_evaluated"] is False
    assert report["train_episode_ids"] and report["evaluation_episode_ids"]
    assert len(report["training_loss"]) == 2
    assert report["checkpoint_sha256"] == sha(tmp_path / "train" / "policy.pt")
    assert report["evaluation_action_metrics"]["sample_count"] > 0


def test_audit_acceptance_cannot_override_controller_failure(tmp_path):
    from sim.auscultation_training.dataset import load_episode
    episode = fixture_episode(tmp_path / "episode")
    outcome = json.loads((episode / "outcome.json").read_text())
    outcome["success"] = False
    (episode / "outcome.json").write_text(json.dumps(outcome))
    bind_audit(episode)
    with pytest.raises(ValueError, match="controller"):
        load_episode(episode)


def test_identical_actor_trajectory_cannot_cross_seed_groups(tmp_path):
    from sim.auscultation_training.dataset import build_dataset
    first = fixture_episode(tmp_path / "first")
    second = fixture_episode(tmp_path / "second")
    outcome = json.loads((second / "outcome.json").read_text())
    outcome["seed"] = 10
    (second / "outcome.json").write_text(json.dumps(outcome))
    fixture_metadata(second, "different trace metadata")
    with pytest.raises(ValueError, match="identical actor trajectory"):
        build_dataset([first, second], tmp_path / "data")


def test_actor_rejects_nonfinite_inputs():
    import torch
    from sim.auscultation_training.policy import BoundedJointPolicy
    x = torch.zeros(1, 12)
    x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        BoundedJointPolicy(12)(x)


def test_training_recomputes_training_scaler_to_detect_leaked_statistics(tmp_path):
    from sim.auscultation_training.dataset import build_dataset
    from sim.auscultation_training.train_bc import main
    episodes = [fixture_episode(tmp_path / str(i), seed=i) for i in range(2)]
    for i, path in enumerate(episodes):
        fixture_metadata(path, i)
    build_dataset(episodes, tmp_path / "data")
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scaler"]["mean"][0] += .1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="scaler"):
        main([str(tmp_path / "data"), "--output", str(tmp_path / "train"), "--epochs", "1"])


def test_full_864_row_trace_cannot_be_accepted_with_one_control_block(tmp_path):
    from sim.auscultation_training.dataset import load_episode
    episode = fixture_episode(tmp_path / "episode", count=72)
    assert len(read_trace(episode)) == 864
    # Regression: a real accepted trace was paired with just one unrelated label.
    def shorten(rows):
        del rows[1:]
        rows[0]["q_command"][0] = .002
        rows[0]["action_joint_delta_rad"][0] = .002
    mutate_controls(episode, shorten)
    with pytest.raises(ValueError, match="cover|endpoint"):
        load_episode(episode)


@pytest.mark.parametrize("change,reason", [
    (lambda rows: rows[1].update(physics_row=13, physics_row_end_exclusive=25), "coverage"),
    (lambda rows: rows[1].update(physics_row=11, physics_row_end_exclusive=23), "coverage"),
    (lambda rows: rows[0].update(target="L1"), "target"),
    (lambda rows: rows[0].update(q_command=[.002, 0, 0, 0, 0, 0, 0], action_joint_delta_rad=[.002, 0, 0, 0, 0, 0, 0]), "endpoint"),
    (lambda rows: rows[0].update(time_s=.1), "timestamp"),
    (lambda rows: rows[0]["observation"].update(q=[.01]*7), "initial"),
    (lambda rows: rows[1]["observation"].update(q=[.01]*7), "observation.*q"),
    (lambda rows: rows[1]["observation"].update(dq=[.01]*7), "observation.*dq"),
    (lambda rows: rows[1]["observation"].update(hall_N=.1), "observation.*hall_N"),
    (lambda rows: rows[1]["observation"].update(compression_m=.001), "observation.*compression_m"),
    (lambda rows: rows[1]["observation"].update(tcp_base_m=[0, 0, 1.1]), "observation.*tcp_base_m"),
    (lambda rows: rows[1]["observation"].update(tcp_rotation_base=np.diag([-1., -1., 1.]).tolist()), "observation.*tcp_rotation_base"),
])
def test_control_trace_binding_rejects_mismatched_labels(tmp_path, change, reason):
    from sim.auscultation_training.dataset import validate_control_trace
    episode = fixture_episode(tmp_path / "episode")
    rows = json.loads((episode / "control_commands.json").read_text())
    change(rows)
    with pytest.raises(ValueError, match=reason):
        validate_control_trace(rows, read_trace(episode))


def test_binding_records_unavailable_pose_fields_without_inventing_a_transform(tmp_path):
    from sim.auscultation_training.dataset import validate_control_trace
    episode = fixture_episode(tmp_path / "episode")
    trace = read_trace(episode)
    for row in trace:
        row.pop("tcp_base_m")
        row.pop("tcp_rotation_base")
        row["tcp_world"] = [0., 5., 2.]
        row["axis_world"] = [0., 0., 1.]
    result = validate_control_trace(json.loads((episode / "control_commands.json").read_text()), trace)
    assert result["physics_rows_covered"] == len(trace)
    assert result["state_field_comparisons"]["q"] == 7
    assert result["state_field_comparisons"]["tcp_base_m"] == 0
    assert "tcp_base_m" in result["unverified_state_fields"]
