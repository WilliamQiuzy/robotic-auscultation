import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sim.auscultation_training.dataset import FRAME_FEATURES as V1_FEATURES
from sim.auscultation_training.controller import CONTROL_DT, CONTROL_STEPS, DT, EpisodeRejected
from sim.auscultation_training.phase_manager import PhaseManager
from sim.auscultation_training.policy import BoundedJointPolicy
from sim.auscultation_training.tests.test_controller import FakeEnv, FakeKin, FakeChecker, target, row
from sim.auscultation_training.tests.test_student import ZeroActor


def test_v2_training_statistics_preserve_large_constant_columns():
    from sim.auscultation_training.dataset_v2 import training_statistics
    # Matches the 15,666-row expert training episode and four 69-D frames.
    values = np.full((15666, 4*69), .85, dtype=np.float32)
    values[:, 0] = np.linspace(-2., 2., len(values), dtype=np.float32)
    mean, std = training_statistics(values)
    assert np.array_equal(mean[1:], values[0, 1:].astype(np.float64))
    assert np.array_equal(std[1:], np.ones(values.shape[1]-1))
    assert mean.dtype == std.dtype == np.float64
    assert np.isclose(mean[0], values[:, 0].astype(np.float64).mean(), atol=1e-15, rtol=0)
    assert np.isclose(std[0], values[:, 0].astype(np.float64).std(), atol=1e-12, rtol=0)


def segment(start, goal, phase="transfer", scale=.85, offset=0):
    start, goal = np.array(start, float), np.array(goal, float)
    blocks = int(np.ceil(max(1 / scale, 1.875 * np.max(abs(goal-start)) / (.25*scale)) / CONTROL_DT))
    steps = blocks * CONTROL_STEPS
    u = np.arange(1, steps+1) / steps
    q = start + (10*u**3-15*u**4+6*u**5)[:, None] * (goal-start)
    rows, trace = [], []
    for k in range(blocks):
        a, b = k*CONTROL_STEPS, (k+1)*CONTROL_STEPS
        prior = start if a == 0 else q[a-1]
        rows.append({"target": "R1", "phase": "APPROACH", "subphase": phase,
            "speed_scale": scale, "physics_row": offset+a, "physics_row_end_exclusive": offset+b,
            "previous_q_command": prior.tolist(), "q_command": q[b-1].tolist(),
            "action_joint_delta_rad": (q[b-1]-prior).tolist(), "action_duration_s": CONTROL_DT,
            "observation": {"q": prior.tolist(), "dq": [0.]*7, "tcp_base_m": [0.,0.,.3],
                "goal_relative_base_m": [0.,0.,-.3], "normal_base": [0.,0.,1.],
                "hall_N": 0., "compression_m": 0., "tcp_rotation_base": np.eye(3).tolist(),
                "target_rotation_base": np.eye(3).tolist()}})
    trace = [{"q_command": value.tolist()} for value in q]
    return rows, trace


def route_fixture():
    zero = np.zeros(7); middle = zero.copy(); middle[0] = .1
    goal = middle.copy(); goal[2] = .2
    a, ta = segment(zero, middle)
    b, tb = segment(middle, goal, offset=len(ta))
    plans = {"R1": {"name": "R1", "success": True, "transfer_q": [zero.tolist(), middle.tolist(), goal.tolist()]}}
    return a+b, ta+tb, plans, len(a)


def test_v2_assigns_verified_active_waypoint_without_changing_teacher_actions():
    from sim.auscultation_training.dataset_v2 import assign_waypoints, encode_frame, FRAME_FEATURES
    rows, trace, plans, cut = route_fixture()
    original = copy.deepcopy(rows)
    annotated, segments = assign_waypoints(rows, plans, trace)
    assert rows == original
    assert len(segments) == 2
    assert annotated[0]["local_waypoint_q"] == plans["R1"]["transfer_q"][1]
    assert annotated[cut]["local_waypoint_q"] == plans["R1"]["transfer_q"][2]
    assert annotated[-1]["action_joint_delta_rad"] == rows[-1]["action_joint_delta_rad"]
    x = encode_frame(annotated[cut])
    assert len(FRAME_FEATURES) == len(V1_FEATURES)+9 == len(x)
    assert np.allclose(x[-9:-2], np.array(plans["R1"]["transfer_q"][2])-rows[cut]["observation"]["q"])
    assert x[-2] == 1 and np.isclose(x[-1], .85)


@pytest.mark.parametrize("mutation", ["endpoint", "interior", "speed", "plan", "missing_segment"])
def test_v2_rejects_unverifiable_waypoint_assignment(mutation):
    from sim.auscultation_training.dataset_v2 import assign_waypoints
    rows, trace, plans, cut = route_fixture()
    if mutation == "endpoint": rows[2]["q_command"][0] += .001
    if mutation == "interior": trace[3]["q_command"][0] += .001
    if mutation == "speed": rows[2]["speed_scale"] = 1.
    if mutation == "plan": plans["R1"]["transfer_q"][1][0] += .01
    if mutation == "missing_segment": rows = rows[:cut]
    with pytest.raises(ValueError): assign_waypoints(rows, plans, trace)


def test_contact_stage_requires_explicit_absent_waypoint_and_finite_speed():
    from sim.auscultation_training.dataset_v2 import encode_frame
    rows, *_ = route_fixture()
    r = dict(rows[0], subphase="contact_cycle", local_waypoint_valid=False, local_waypoint_q=None)
    assert np.all(encode_frame(r)[-9:-1] == 0)
    with pytest.raises(ValueError): encode_frame(dict(r, local_waypoint_valid=True, local_waypoint_q=[0.]*7))
    with pytest.raises(ValueError): encode_frame(dict(r, speed_scale=float("nan")))


def controller(actor=None, checker=None):
    from sim.auscultation_training.student_v2 import StudentControllerV2
    c = StudentControllerV2(FakeEnv(), FakeKin(), checker or FakeChecker(), actor=actor or ZeroActor(),
                            contact_mode="pure_bc", speed_scale=.85, waypoint_timeout_s=.1)
    c.target = target("R1"); c.normal = c.normal_base = np.array([0.,0.,1.])
    c.rotation_base = np.diag([1.,-1.,-1.]); c.manager = PhaseManager(c.config)
    c.face_filtered = c.total_filtered = c.audit_hold = c.best_audit_hold = 0.
    return c


def test_v2_online_waypoint_is_actual_supervisor_goal_and_matches_offline_encoder():
    from sim.auscultation_training.dataset_v2 import encode_frame
    actor = ZeroActor(); c = controller(actor)
    goal = np.array([.1,0.,0.,0.,0.,0.,0.])
    with pytest.raises(EpisodeRejected, match="waypoint timeout"): c._joint_segment(goal, "transfer")
    assert actor.histories and actor.histories[0].shape == (4,69)
    assert np.allclose(actor.histories[0][-1], encode_frame(c.control_rows[0]))
    assert np.allclose(actor.histories[0][-1][-9:-2], goal)
    assert c.current_waypoint is None
    assert np.max(abs(c.env.q)) == 0.  # No nominal analytic rescue.


def test_position_projection_reports_raw_and_applied_and_preserves_legal_components():
    c = controller(); c.kin.lower_limits = np.zeros(7); c.kin.upper_limits = np.ones(7)
    raw = np.array([-.001,.002,0.,0.,0.,0.,0.])
    applied, meta = c.compose_action(raw, c.env.state())
    assert np.array_equal(applied, [0.,.002,0.,0.,0.,0.,0.])
    assert meta["joint_position_projection_applied"] is True
    assert meta["raw_policy_action_rad"] == raw.tolist()
    assert meta["applied_action_rad"] == applied.tolist()
    assert meta["joint_position_projection_delta_rad"] == [.001,0.,0.,0.,0.,0.,0.]
    assert np.max(abs(applied)) <= .25*.85*CONTROL_DT


def test_projected_action_still_cannot_bypass_collision_preflight():
    class Collision(FakeChecker):
        def edge(self, *args, **kwargs): return {"clear": False}
    class Moving(ZeroActor):
        def predict(self, history): return np.array([-.001,.002,0.,0.,0.,0.,0.])
    c = controller(Moving(), Collision()); c.kin.lower_limits = np.zeros(7)
    c.current_waypoint = np.ones(7)*.1
    with pytest.raises(EpisodeRejected, match="collision preflight"): c._learned_step("transfer")
    assert not c.env.rows


def test_v2_loader_pins_new_encoder_and_rejects_v1_checkpoint(tmp_path):
    from sim.auscultation_training.dataset_v2 import FRAME_FEATURES
    from sim.auscultation_training.student_v2 import load_actor, MODEL_CODE_FILES
    policy = BoundedJointPolicy(4*len(FRAME_FEATURES), hidden=(8,))
    checkpoint = {"schema":"auscultation_bc_policy_v2", "state_dict":policy.state_dict(),
        "input_dim":policy.input_dim,"hidden":[8],"history_steps":4,"frame_features":list(FRAME_FEATURES),
        "action_limit_rad":.0125,"dataset_manifest_sha256":"a"*64}
    torch.save(checkpoint, tmp_path/"policy.pt")
    sha = lambda data: hashlib.sha256(data).hexdigest()
    root=Path(__file__).parents[1]
    manifest={"schema":"auscultation_bc_training_v2", "checkpoint_sha256":sha((tmp_path/"policy.pt").read_bytes()),
        "dataset_manifest_sha256":"a"*64, "architecture":{"type":"shared_waypoint_conditioned_MLP",
            "input_dim":276,"hidden":[8],"output_dim":7,"output":"tanh times 0.0125 rad"},
        "scaler":{"mean":policy.input_mean.tolist(),"std":policy.input_std.tolist()},
        "training_speed_scales":[.85], "code_sha256":{n:sha((root/n).read_bytes()) for n in MODEL_CODE_FILES}}
    data=json.dumps(manifest).encode(); (tmp_path/"run_manifest.json").write_bytes(data)
    actor=load_actor(tmp_path,sha(data))
    assert actor.predict(np.zeros((4,69),np.float32)).shape == (7,)
    assert actor.metadata["training_speed_scales"] == [.85]
    assert not any(p.requires_grad for p in actor.policy.parameters())
    checkpoint["schema"]="auscultation_bc_policy_v1"; torch.save(checkpoint,tmp_path/"policy.pt")
    manifest["checkpoint_sha256"]=sha((tmp_path/"policy.pt").read_bytes())
    data=json.dumps(manifest).encode(); (tmp_path/"run_manifest.json").write_bytes(data)
    with pytest.raises(ValueError,match="schema"): load_actor(tmp_path,sha(data))


def test_v2_training_input_requires_bound_annotation_sidecars(tmp_path):
    from sim.auscultation_training.train_bc_v2 import load_training_data
    from sim.auscultation_training.dataset_v2 import FRAME_FEATURES
    arrays = io.BytesIO()
    x=np.zeros((2,4,69),np.float32); x[:,:,-1]=.85; y=np.zeros((2,7),np.float32)
    np.savez(arrays, observations=x, actions=y, split=np.array([0,1]), episode_index=np.array([0,1]))
    blob=arrays.getvalue(); (tmp_path/"dataset.npz").write_bytes(blob)
    sidecar=json.dumps({"schema":"auscultation_waypoint_annotations_v2", "source_artifact_sha256":{"trace":"fixture"}, "rows":[{
        "control_row":0,"target":"R1","local_waypoint_valid":False,"local_waypoint_q":None,"speed_scale":.85}]}).encode()
    (tmp_path/"labels.json").write_bytes(sidecar)
    manifest={"schema":"auscultation_bc_dataset_v2", "frame_features":list(FRAME_FEATURES),
        "history_steps":4, "action_limit_rad":.0125,"arrays_sha256":hashlib.sha256(blob).hexdigest(),
        "scaler":{"fit_split":"train_only","mean":x[0].reshape(-1).tolist(),"std":[1.]*276},
        "episodes":[{"split":s,"samples":1,"group_id":s,"episode_id":s,"speed_scales":[.85],
            "artifact_sha256":{"trace":"fixture"},
            "waypoint_annotation_file":"labels.json","waypoint_annotation_sha256":hashlib.sha256(sidecar).hexdigest()}
            for s in ("train","evaluation")]}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    result=load_training_data(tmp_path)
    assert result[2]["observations"].shape == (2,4,69)
    manifest["episodes"][0]["speed_scales"]=[1.]
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="speed metadata differs"): load_training_data(tmp_path)
    manifest["episodes"][0]["speed_scales"]=[.85]
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    (tmp_path/"labels.json").write_bytes(b"changed")
    with pytest.raises(ValueError,match="annotation fingerprint"): load_training_data(tmp_path)


def test_v2_annotation_conditioning_cannot_disagree_with_actor_arrays():
    from sim.auscultation_training.train_bc_v2 import verify_annotation_features
    rows=[{"control_row":0,"target":"R1","local_waypoint_valid":True,"local_waypoint_q":[.1]*7,"speed_scale":.85}]
    x=np.zeros((1,4,69),np.float32)
    x[:,:,-9:-2]=.1; x[:,:,-2]=1; x[:,:,-1]=.85
    verify_annotation_features(x,{"rows":rows})
    x[0,0,-9]=.2
    with pytest.raises(ValueError,match="conditioning differs"): verify_annotation_features(x,{"rows":rows})


def test_v2_eval_requires_training_speed_before_simulator_or_plans(monkeypatch, tmp_path):
    from sim.auscultation_training import evaluate_bc_v2
    from types import SimpleNamespace
    monkeypatch.setattr(evaluate_bc_v2,"load_actor",lambda *_:SimpleNamespace(metadata={"training_speed_scales":[.85]}))
    def unreachable(*args,**kwargs): raise AssertionError("plan validation should not be reached")
    monkeypatch.setattr(evaluate_bc_v2,"validate_plan",unreachable)
    with pytest.raises(ValueError,match="training speed"):
        evaluate_bc_v2.main(["--model-dir",str(tmp_path),"--manifest-sha256","a"*64,"--speed-scale","1"])


@pytest.mark.parametrize("change", ["manifest", "object"])
def test_v2_annotation_rejects_changed_archived_plan_binding(tmp_path, change):
    from sim.auscultation_training.dataset_v2 import annotate_episode
    from sim.auscultation_training.dataset import ARTIFACT_NAMES
    rows,trace,plans,_=route_fixture()
    snap=tmp_path/"input_snapshot"; objects=snap/"objects"; objects.mkdir(parents=True)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    plan=json.dumps({"points":[plans["R1"]]}).encode(); digest=sha(plan)
    (objects/digest).write_bytes(plan)
    manifest=json.dumps({"files":{"R1/plan":{"sha256":digest}}}).encode()
    (snap/"manifest.json").write_bytes(manifest)
    (tmp_path/"outcome.json").write_text(json.dumps({"input_snapshot_manifest_sha256":sha(manifest)}))
    (tmp_path/"control_commands.json").write_text(json.dumps(rows))
    (tmp_path/"expert_trace.jsonl").write_text("\n".join(json.dumps(r) for r in trace))
    hashes={name:sha((tmp_path/name).read_bytes()) for name in ARTIFACT_NAMES}
    assert annotate_episode(tmp_path,hashes)[0][0]["local_waypoint_valid"]
    if change=="manifest": (snap/"manifest.json").write_bytes(manifest+b" ")
    else: (objects/digest).write_bytes(plan+b" ")
    with pytest.raises(ValueError,match="fingerprint"): annotate_episode(tmp_path,hashes)
