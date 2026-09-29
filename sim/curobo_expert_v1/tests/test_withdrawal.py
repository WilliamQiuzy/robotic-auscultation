"""CPU verification of normal unloading and command-boundary continuity."""
import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest


def module():
    name = "sim.curobo_expert_v1.withdrawal"
    assert importlib.util.find_spec(name) is not None, "normal withdrawal implementation is missing"
    return importlib.import_module(name)


def simple_path():
    return np.linspace(np.zeros(7), np.array([.03, -.01, .02, 0., .005, 0., 0.]), 17)


def test_normal_path_retains_endpoint_and_both_dynamic_boundaries():
    m = module()
    path = simple_path()
    history = np.zeros((3, 7))
    # The prior contact controller is still moving slightly at phase entry.
    history[:, 0] = [-2e-6, -1e-6, 0.]
    commands, fractions, report = m.normal_path_commands(path, .008, .002, history)
    np.testing.assert_array_equal(commands[-1], path[-1])
    assert len(commands) % 12 == 0
    assert fractions[-1] == 1. and np.all(np.diff(fractions) >= 0)
    padded = np.vstack([history, commands, np.repeat(commands[-1:], 3, axis=0)])
    for order, limit in ((1, .25), (2, .5), (3, 8.)):
        assert np.max(np.abs(np.diff(padded, n=order, axis=0))) * 240**order <= limit+1e-8
    incoming = history[-1]-history[-2]
    np.testing.assert_allclose(commands[0]-history[-1], incoming, atol=2e-8)
    assert report["boundary_history_samples"] == 3


def test_normal_profile_does_not_erase_retained_command_mismatch():
    path = simple_path()
    with pytest.raises(ValueError, match="retained command"):
        module().normal_path_commands(path, .008, .002, np.full((3, 7), .001))


def test_normal_profile_rejects_dynamics_that_exhaust_the_original_time_budget():
    with pytest.raises(RuntimeError, match="timing budget"):
        module().normal_path_commands(simple_path(), .008, .002,
                                      np.zeros((3, 7)), maximum_seconds=.1)


def controller(monkeypatch, env=None):
    from sim.expert_quality_20260924.candidate_controller import CandidateController
    from sim.auscultation_expert_layout_v1 import pose_ik
    from sim.auscultation_training.tests.test_controller import FakeEnv, target
    from sim.auscultation_training.tests.test_low_elbow_runtime import PostureKin, PostureCollisionChecker, profile
    from sim.auscultation_training.tests.test_low_elbow_sequence import fake_constrained_ik

    class Revised(module().NormalWithdrawalMixin, CandidateController):
        pass

    kin, env = PostureKin(), env or FakeEnv()
    # FakeKin.tcp originally omits its fixed rotation; supply the actual frame
    # represented by the spring fixture rather than weakening geometry checks.
    original_tcp = kin.tcp
    def tcp(q):
        result = original_tcp(q)
        result[:3, :3] = np.diag([1., -1., -1.])
        return result
    kin.tcp = tcp
    value = Revised(env, kin, PostureCollisionChecker(kin), policy="checked_shortcut")
    monkeypatch.setattr(pose_ik, "solve_posture_ik", fake_constrained_ik)
    row = {"name": "R5", "success": True, "command_normal_world": [0., 0., 1.],
           "command_rotation_world": np.diag([1., -1., -1.]).tolist(),
           "posture_limits": profile(), "lower_staging_q": [0., 0., .18, 0., 0., 0., 0.],
           "transfer_q": [np.zeros(7).tolist(), [0., 0., .18, 0., 0., 0., 0.],
                          [0., 0., .275, 0., 0., 0., 0.]],
           "free_approach_start_gap_m": .025}
    return value, [(target("R5"), row)]


def test_complete_contact_keeps_hold_force_settings_and_smooth_unloading(monkeypatch):
    value, points = controller(monkeypatch)
    result = value.run(points)
    assert result["success"], result
    point = result["points"][0]
    assert point["verified_front_contact_hold_s"] >= 3.-2/240
    assert point["phase_config"]["target_force_N"] == 4.
    assert point["phase_config"]["hold_duration_s"] == 3.
    reports = result["normal_withdrawal_segments"]
    assert [r["kind"] for r in reports] == ["unload", "released_retreat"]
    assert all(r["motion_complete"] for r in reports)
    assert reports[0]["normal_speed_limit_m_s"] == .002
    for report in reports:
        assert report["peaks"]["acceleration_rad_s2"] <= .5+1e-8
        assert report["peaks"]["jerk_rad_s3"] <= 8.+1e-8
    assert value.released_prepose["target"] == "R5"
    assert value.released_prepose["physics_row_end_exclusive"] < len(value.env.rows)


@pytest.mark.parametrize("kind", ["physical", "hall", "nonfinite"])
def test_withdrawal_cannot_publish_release_on_bad_measurements(monkeypatch, kind):
    value, points = controller(monkeypatch)
    result = value.run(points)
    assert result["success"]
    value._hold = lambda *args: None
    value.q = np.array([0., 0., .275, 0., 0., 0., 0.])
    value.env.q = value.q.copy()
    state = value.env.state()
    if kind == "hall": state["hall_N"] = .2
    elif kind == "nonfinite": state["hall_N"] = float("nan")
    else: value.env.last_contacts = {"lip": .3}
    value.env.state = lambda: state
    value.released_prepose = None
    with pytest.raises(RuntimeError, match="release|withdrawal"):
        value._withdraw_to_prepose(None)
    assert value.released_prepose is None


def test_contact_force_failure_still_fails_without_next_point(monkeypatch):
    from sim.auscultation_training.tests.test_controller import FakeEnv
    value, points = controller(monkeypatch, FakeEnv(failure=True))
    result = value.run(points)
    assert not result["success"] and not result["training_accepted"]
    assert result["points"][0]["recovery"]["attempted"]
    assert not result["normal_withdrawal_segments"]


def test_hall_release_cannot_start_faster_retreat_while_physical_force_remains(monkeypatch):
    from sim.auscultation_training.tests.test_controller import FakeEnv

    class StuckContact(FakeEnv):
        def step(self, q, phase, target, meta):
            state = super().step(q, phase, target, meta)
            if meta["subphase"] == "normal_unload" and state["hall_N"] < .2:
                self.last_contacts["lip"] = .3
            return state

    value, points = controller(monkeypatch, StuckContact())
    result = value.run(points)
    assert not result["success"], "physical release must be checked before faster retreat"
    assert "physical" in result["points"][0]["failure_reason"]
    assert [r["kind"] for r in result["normal_withdrawal_segments"]] == ["unload"]


def released_actor(compression):
    value = module().NormalWithdrawalMixin()
    value.target = SimpleNamespace(name="R5")
    value._normal_release_target = "R5"
    value.q = np.zeros(7)
    value.pre_gap = .025
    value._hold = lambda *args: None
    value.metrics = lambda state: {"gap_m": .025}
    state = {"q": np.zeros(7), "time_s": 1., "hall_N": 0., "compression_m": compression}
    value.env = SimpleNamespace(state=lambda: state, last_contacts={}, rows=[{}])
    return value


@pytest.mark.parametrize("compression", [.0001501, float("nan"), float("inf")])
def test_release_rejects_residual_compression_even_with_zero_hall_and_force(compression):
    value = released_actor(compression)
    with pytest.raises(RuntimeError, match="compression"):
        value._withdraw_to_prepose(None)
    assert value.released_prepose is None


@pytest.mark.parametrize("compression", [0., .00015])
def test_release_records_compression_at_or_below_original_free_threshold(compression):
    value = released_actor(compression)
    value._withdraw_to_prepose(None)
    assert value.released_prepose["compression_m"] == compression
