import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.auscultation_training.controller import (
    ExpertController, create_snapshot, validate_plan, verify_snapshot,
)
from sim.auscultation_training.targets import Target


def fixture_plan(tmp_path, name="R1"):
    base = np.eye(4).tolist()
    normal = [0, 0, 1]
    rotation = np.diag([1, -1, -1]).tolist()
    candidate = {"normal_world": normal, "wrist_rotation_world": rotation,
                 "gap_m": .005}
    data = {
        "audit": {"name": name, "compression_m": .002, "candidates": [candidate]},
        "targets": {"scope": "candidate only", "frames": {"raised": {"points": [
            {"name": name, "depth_valid": True, "world_xyz_m": [0, 0, 0],
             "normal_world_from_depth": normal}]}}},
        "runtime": {"world_from_left_base": base},
        "collision_metadata": {"colliders": [{"path": "/World/Patient/surface"}],
                               "allowed_body_pairs": []},
        "collision_arrays": {"fixture": True},
        "kinematics_urdf": {"fixture_urdf": True},
    }
    provenance = {}
    for key, value in data.items():
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(value))
        provenance[key + "_file"] = path.name
        provenance[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {"name": name, "success": True, "position_world": [0, 0, 0],
           "normal_world": normal, "command_normal_world": normal,
           "command_rotation_world": rotation, "selected_audit_candidate_index": 0,
           "selected_fit": candidate, "loaded_compression_m": .002,
           "transfer_q": [[0.] * 7, [0, 0, .24, 0, 0, 0, 0]],
           "approach_q": [[0, 0, .24, 0, 0, 0, 0], [0, 0, .285, 0, 0, 0, 0]],
           "geometric_clearance_m": .003, "joint_edge_sample_step_rad": .015}
    plan = {"world_from_left_base": base, "targets_file": "targets.json",
            "audit_file": "audit.json", "provenance": provenance, "points": [row]}
    path = tmp_path / (name + ".json")
    path.write_text(json.dumps(plan))
    return path


def test_changed_input_rejected_before_any_motion(tmp_path):
    path = fixture_plan(tmp_path)
    (tmp_path / "collision_arrays.json").write_text("changed")
    with pytest.raises(ValueError, match="fingerprint.*collision_arrays"):
        validate_plan(path, tmp_path, tmp_path)


def test_old_plan_without_urdf_fingerprint_cannot_be_accepted(tmp_path):
    path = fixture_plan(tmp_path)
    plan = json.loads(path.read_text())
    del plan["provenance"]["kinematics_urdf_file"]
    del plan["provenance"]["kinematics_urdf_sha256"]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="kinematics_urdf"):
        validate_plan(path, tmp_path, tmp_path)


def test_target_metadata_cannot_disagree_with_hashed_input(tmp_path):
    path = fixture_plan(tmp_path)
    plan = json.loads(path.read_text())
    plan["points"][0]["position_world"] = [.001, 0, 0]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="target coordinates"):
        validate_plan(path, tmp_path, tmp_path)


def test_snapshot_remains_bound_to_exact_bytes_and_detects_tampering(tmp_path):
    bundle = validate_plan(fixture_plan(tmp_path), tmp_path, tmp_path)
    snapshot = create_snapshot([bundle], tmp_path / "snapshot", {})
    (tmp_path / "targets.json").write_text("mutated source after snapshot")
    verify_snapshot(snapshot)
    first = next((snapshot / "objects").iterdir())
    first.chmod(0o644)
    first.write_bytes(b"tampered snapshot")
    with pytest.raises(ValueError, match="snapshot fingerprint"):
        verify_snapshot(snapshot)


class FakeKin:
    lower_limits = np.full(7, -3.)
    upper_limits = np.full(7, 3.)

    def tcp(self, q):
        pose = np.eye(4)
        pose[:3, 3] = [q[0], q[1], .3 - q[2]]
        return pose

    def solve_ik(self, position, **kw):
        q = np.asarray(kw["seed"]).copy()
        q[:3] = [position[0], position[1], .3 - position[2]]
        return SimpleNamespace(success=True, q=q)


class FakeEnv:
    """A deterministic spring fixture; no reset or pose-write API exists."""
    def __init__(self, failure=False):
        self.base = np.eye(4)
        self.q = np.zeros(7)
        self.t = 0.
        self.rows = []
        self.last_contacts = {}
        self.last_pairs = {}
        self.failure = failure
        self.previous_tcp = None

    def state(self):
        commanded_gap = .3 - self.q[2]
        compression = max(0., .003 - commanded_gap)
        hall = 0. if compression < .0001 else 1.5 + 1000 * compression
        tcp = np.array([self.q[0], self.q[1], commanded_gap + compression])
        return dict(q=self.q.copy(), dq=np.zeros(7), tcp_world=tcp, tcp_base=tcp,
                    rotation_world=np.diag([1., -1., -1.]), axis_world=np.array([0., 0., -1.]),
                    compression_m=compression, hall_N=hall, time_s=self.t)

    def step(self, q, phase, target, meta):
        self.q = np.asarray(q).copy()
        self.t += 1 / 240
        s = self.state()
        self.last_contacts = {"lip": s["hall_N"]}
        if self.failure and s["compression_m"] > .0005:
            self.last_contacts["frame"] = .6
        speed = 0. if self.previous_tcp is None else np.linalg.norm(s["tcp_world"] - self.previous_tcp) * 240
        self.previous_tcp = s["tcp_world"].copy()
        self.rows.append({"time_s": self.t, "tcp_speed_m_s": speed, "phase": phase,
                          "target": target, "q_command": self.q.tolist(), **meta})
        return s


class FakeChecker:
    def edge(self, start, end):
        return {"clear": True, "minimum_distance_m": .003}


def target(name):
    return Target(name, np.zeros(3), np.array([0., 0., 1.]), np.zeros(3),
                  np.array([0., 0., 1.]), {"scope": "synthetic test"})


def row(name):
    return {"name": name, "success": True, "command_normal_world": [0, 0, 1],
            "command_rotation_world": np.diag([1, -1, -1]).tolist(),
            "transfer_q": [np.zeros(7).tolist(), [0, 0, .24, 0, 0, 0, 0]]}


def test_two_points_return_home_dynamically_and_record_bounded_transfer_actions():
    env = FakeEnv()
    controller = ExpertController(env, FakeKin(), checker=FakeChecker())
    outcome = controller.run([(target("R1"), row("R1")), (target("L1"), row("L1"))])
    assert outcome["training_accepted"]
    assert [p["target"] for p in outcome["points"]] == ["R1", "L1"]
    assert all(p["verified_front_contact_hold_s"] >= 3 - 2 / 240 for p in outcome["points"])
    assert np.max(np.abs(env.q)) < 1e-6
    assert {r["subphase"] for r in controller.control_rows} >= {"transfer", "return_home", "withdraw_to_prepose"}
    assert all(np.max(np.abs(r["action_joint_delta_rad"])) <= .25 * r["action_duration_s"] + 1e-9
               for r in controller.control_rows)
    assert all(b["time_s"] > a["time_s"] for a, b in zip(env.rows, env.rows[1:]))


def test_contact_failure_attempts_outward_retreat_and_never_runs_next_point():
    env = FakeEnv(failure=True)
    controller = ExpertController(env, FakeKin(), checker=FakeChecker())
    outcome = controller.run([(target("R1"), row("R1")), (target("L1"), row("L1"))])
    assert not outcome["training_accepted"]
    assert len(outcome["points"]) == 1
    assert outcome["points"][0]["recovery"]["attempted"]
    assert outcome["points"][0]["recovery"]["released"]
    assert any(r["subphase"] == "fault_retract" for r in env.rows)
    assert not any(r["target"] == "L1" for r in env.rows)
    assert not any(r["training_accepted"] for r in controller.control_rows)


def test_starting_away_from_home_rejects_without_teleport_or_motion():
    env = FakeEnv()
    env.q[0] = .2
    controller = ExpertController(env, FakeKin(), checker=FakeChecker())
    outcome = controller.run([(target("R1"), row("R1"))])
    assert not outcome["training_accepted"]
    assert not env.rows
    assert env.q[0] == .2


def test_recovery_monitors_transient_contact_at_physics_rate():
    class TransientEnv(FakeEnv):
        recovery_steps = 0

        def step(self, q, phase, target, meta):
            state = super().step(q, phase, target, meta)
            if meta["subphase"] == "fault_retract":
                self.recovery_steps += 1
                if self.recovery_steps == 1:
                    self.last_contacts["frame"] = 2.
            return state
    env = TransientEnv(failure=True)
    result = ExpertController(env, FakeKin(), FakeChecker()).run([(target("R1"), row("R1"))])
    assert not result["success"]
    assert env.recovery_steps == 1
    assert not result["points"][0]["recovery"]["released"]
    assert "worsened" in result["points"][0]["recovery"]["reason"]


def test_closer_checked_prepose_changes_only_standoff_not_hold_or_force():
    close_row = row("R3")
    close_row["free_approach_start_gap_m"] = .025
    close_row["transfer_q"][-1][2] = .275
    result = ExpertController(FakeEnv(), FakeKin(), FakeChecker()).run([(target("R3"), close_row)])
    assert result["success"]
    config = result["points"][0]["phase_config"]
    assert config["precontact_gap_m"] == .025
    assert config["retract_clearance_m"] == .020
    assert config["hold_duration_s"] == 3.
    assert config["target_force_N"] == 4.
    assert config["hard_force_limit_N"] == 5.5


def test_later_point_failure_invalidates_entire_sequence_for_training():
    class SecondPointFault(FakeEnv):
        def step(self, q, phase, target, meta):
            self.failure = target == "L1"
            return super().step(q, phase, target, meta)
    controller = ExpertController(SecondPointFault(), FakeKin(), FakeChecker())
    result = controller.run([(target("R1"), row("R1")), (target("L1"), row("L1"))])
    assert result["points"][0]["success"]
    assert not result["points"][1]["success"]
    assert not any(p["training_accepted"] for p in result["points"])
    assert not any(r["training_accepted"] for r in controller.control_rows)


def test_redundant_ik_jump_is_rejected_before_command_execution():
    class JumpingKin(FakeKin):
        def solve_ik(self, position, **kw):
            solution = super().solve_ik(position, **kw)
            solution.q[0] = 1.
            return solution
    env = FakeEnv()
    result = ExpertController(env, JumpingKin(), FakeChecker()).run([(target("R1"), row("R1"))])
    assert not result["success"]
    assert "speed limit" in result["points"][0]["failure_reason"]
    assert all(abs(r["q_command"][0]) < 1e-9 for r in env.rows)


def test_invalid_later_prepose_rejected_before_first_point_moves():
    env = FakeEnv()
    invalid = row('L1')
    invalid['free_approach_start_gap_m'] = .07
    result = ExpertController(env, FakeKin(), FakeChecker()).run([
        (target('R1'), row('R1')), (target('L1'), invalid)])
    assert not result['training_accepted']
    assert not env.rows
    assert not any(p['training_accepted'] for p in result['points'])


def test_unsupported_plan_prepose_rejected_before_initialization(tmp_path):
    path = fixture_plan(tmp_path)
    plan = json.loads(path.read_text())
    plan['points'][0]['free_approach_start_gap_m'] = .07
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='prepose'):
        validate_plan(path, tmp_path, tmp_path)


def test_requested_point_cannot_be_substituted_by_file_contents(tmp_path):
    from sim.auscultation_training.run_sequence import main
    path = fixture_plan(tmp_path, 'L1')
    path.rename(tmp_path/'planned_paths_R1.json')
    with pytest.raises(ValueError, match='requested point'):
        main(['--points','R1','--input-dir',str(tmp_path),'--runtime-dir',str(tmp_path)])


@pytest.mark.parametrize("scale", [.69, 1.01, float("nan"), float("inf"), True, "0.8"])
def test_invalid_speed_scale_is_rejected_before_motion(scale):
    env = FakeEnv()
    with pytest.raises(ValueError, match="speed_scale"):
        ExpertController(env, FakeKin(), FakeChecker(), speed_scale=scale)
    assert not env.rows


def test_timing_variation_changes_motion_but_preserves_contact_criteria():
    runs = []
    for scale in (1., .7):
        env = FakeEnv()
        controller = ExpertController(env, FakeKin(), FakeChecker(), speed_scale=scale)
        result = controller.run([(target("R1"), row("R1"))])
        assert result["training_accepted"]
        assert result["speed_scale"] == scale
        assert result["variation"]["kind"] == "deterministic_motion_timing"
        assert result["variation"]["free_approach_far_speed_m_s"] == .012 * scale
        assert result["variation"]["free_approach_near_speed_m_s"] == .002 * scale
        assert result["variation"]["force_feedback_gain_scale"] == 1.
        assert result["joint_speed_limit_rad_s"] == .25 * scale
        assert result["points"][0]["observed_hold_s"] == 3.
        first_contact = next(r for r in controller.control_rows if r["subphase"] == "contact_cycle")
        assert first_contact["command_gap_m"] == pytest.approx(.06 - .012 * scale * .05)
        assert all(np.max(np.abs(r["action_joint_delta_rad"])) <= .25 * scale * r["action_duration_s"] + 1e-9
                   for r in controller.control_rows)
        assert np.max(np.abs(env.q)) < 1e-6
        runs.append((env, controller, result))
    nominal, slowed = runs
    assert slowed[0].t > nominal[0].t + 5.
    nominal_config = nominal[2]["points"][0]["phase_config"]
    slower_config = slowed[2]["points"][0]["phase_config"]
    travel_timeouts = {"approach_timeout_s", "contact_timeout_s", "retract_timeout_s", "episode_timeout_s"}
    for name, value in nominal_config.items():
        assert slower_config[name] == pytest.approx(value / .7 if name in travel_timeouts else value)
    assert slower_config["target_force_N"] == 4.
    assert slower_config["force_tolerance_N"] == .5
    assert slower_config["hard_force_limit_N"] == 5.5


def test_cli_invalid_speed_scale_fails_before_loading_plans(capsys):
    from sim.auscultation_training.run_sequence import main
    with pytest.raises(SystemExit) as error:
        main(["--speed-scale", "0.69", "--input-dir", "/no/such/directory"])
    assert error.value.code == 2
    assert "speed_scale" in capsys.readouterr().err
