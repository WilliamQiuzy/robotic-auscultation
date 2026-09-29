import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sim.auscultation_training.controller import CONTROL_DT, EpisodeRejected
from sim.auscultation_training.dataset_v2 import FRAME_FEATURES, HISTORY_STEPS
from sim.auscultation_training.phase_manager import PhaseManager
from sim.auscultation_training.tests.test_controller import FakeEnv, FakeKin, FakeChecker, target


def history(stage="transfer", scale=.85):
    values=np.zeros((1,HISTORY_STEPS,len(FRAME_FEATURES)),np.float32)
    values[:,:,FRAME_FEATURES.index("phase_APPROACH")]=1
    values[:,:,FRAME_FEATURES.index("subphase_"+stage)]=1
    values[:,:,-2]=float(stage in ("initial_home","transfer","return_bridge","return_home"))
    values[:,:,-1]=scale
    return torch.from_numpy(values)


def test_route_policy_preserves_planner_direction_and_loss_reaches_scalar_head():
    from sim.auscultation_training.policy_v3 import RouteProgressPolicy
    policy=RouteProgressPolicy(hidden=(8,))
    for parameter in policy.parameters(): parameter.data.zero_()
    remaining=torch.tensor([[-.2,.1,0.,.3,0.,0.,.04]])
    prediction=policy(history(),remaining)
    base=remaining*(.25*.85*CONTROL_DT/.3)
    assert torch.allclose(prediction,base*.5)
    assert torch.max(torch.abs(prediction))<=.25*.85*CONTROL_DT
    loss=((prediction-base*.8)**2).mean();loss.backward()
    assert policy.network[-1].bias.grad[0].abs()>0
    assert torch.all(policy.network[-1].bias.grad[1:]==0)


def test_joint_head_remains_learned_for_contact_and_zero_route_never_moves():
    from sim.auscultation_training.policy_v3 import RouteProgressPolicy
    policy=RouteProgressPolicy(hidden=(8,))
    for parameter in policy.parameters(): parameter.data.zero_()
    policy.network[-1].bias.data[1:]=.5
    contact=policy(history("contact_cycle"),torch.zeros((1,7)))
    assert torch.allclose(contact,torch.full((1,7),.0125*np.tanh(.5)))
    assert torch.count_nonzero(policy(history(),torch.zeros((1,7))))==0
    policy.network[-1].bias.data[0]=-1000.
    assert torch.count_nonzero(policy(history(),torch.ones((1,7))))==0


@pytest.mark.parametrize("invalid",["shape","speed","mask","nonfinite"])
def test_policy_rejects_unsupported_route_inputs(invalid):
    from sim.auscultation_training.policy_v3 import RouteProgressPolicy
    policy=RouteProgressPolicy(hidden=(8,));x=history();remaining=torch.ones((1,7))
    if invalid=="shape": remaining=torch.ones((1,6))
    if invalid=="speed": x[:,:,-1]=.5
    if invalid=="mask": x[:,:,-2]=0.
    if invalid=="nonfinite": remaining[0,0]=float("nan")
    with pytest.raises(ValueError): policy(x,remaining)


def test_exact_certified_route_labels_are_validated_before_training():
    from sim.auscultation_training.train_bc_v3 import verify_route_action
    remaining=np.array([-.2,.1,0.,.3,0.,0.,.04]);scale=.85
    base=remaining*min(1.,.25*scale*CONTROL_DT/max(abs(remaining)))
    fraction,residual=verify_route_action(remaining,base*.4,scale)
    assert fraction==pytest.approx(.4) and residual<1e-14
    for action in (-base*.1,base*1.1,base*.4+np.array([0.,0.,.0001,0.,0.,0.,0.])):
        with pytest.raises(ValueError): verify_route_action(remaining,action,scale)
    with pytest.raises(ValueError): verify_route_action(np.zeros(7),np.ones(7)*1e-4,scale)


class ComponentsActor:
    metadata={"checkpoint_sha256":"fake", "scope":"unit test"}
    def __init__(self,fraction): self.fraction=fraction;self.histories=[]
    def predict_components(self,values):
        self.histories.append(values.copy())
        return {"progress_fraction":self.fraction,"joint_action_rad":[-.001]*7}


def controller(fraction,checker=None):
    from sim.auscultation_training.student_v3 import StudentControllerV3
    c=StudentControllerV3(FakeEnv(),FakeKin(),checker or FakeChecker(),actor=ComponentsActor(fraction),
        contact_mode="pure_bc",speed_scale=.85,waypoint_timeout_s=.1)
    c.target=target("R1");c.normal=c.normal_base=np.array([0.,0.,1.]);c.rotation_base=np.diag([1.,-1.,-1.])
    c.manager=PhaseManager(c.config);c.face_filtered=c.total_filtered=c.audit_hold=c.best_audit_hold=0.
    return c


def test_live_route_uses_exact_waypoint_and_prior_command_not_float32_reconstruction():
    c=controller(.4);c.q=np.array([.1234567890123,0.,0.,0.,0.,0.,0.])
    c.env.q=c.q.copy();c.current_waypoint=c.q+np.array([1e-9,.1,0.,0.,0.,0.,0.])
    remaining=c.current_waypoint-c.q;base=remaining*(.25*.85*CONTROL_DT/.1)
    c._learned_step("transfer")
    row=c.control_rows[-1]
    assert np.asarray(row["action_joint_delta_rad"])==pytest.approx(base*.4,abs=1e-15)
    assert row["route_constraint_applied"] is True
    assert row["predicted_route_fraction"]==.4
    assert row["applied_route_fraction"]==pytest.approx(.4)
    assert row["controller_source"]=="learned_progress_on_checked_route"
    assert row["route_direction_source"]=="checked planner waypoint minus previous command"


def test_zero_fraction_stalls_and_rejects_without_minimum_speed_or_expert_rescue():
    c=controller(0.)
    with pytest.raises(EpisodeRejected,match="waypoint timeout"):
        c._joint_segment(np.ones(7)*.1,"transfer")
    assert np.count_nonzero(c.env.q)==0
    assert all(row["predicted_route_fraction"]==0 for row in c.control_rows)


def test_route_constraint_cannot_bypass_collision_or_invalid_fraction():
    class Blocked(FakeChecker):
        def edge(self,*args,**kwargs):return {"clear":False}
    c=controller(.5,Blocked());c.current_waypoint=np.ones(7)*.1
    with pytest.raises(EpisodeRejected,match="collision preflight"):c._learned_step("transfer")
    assert not c.env.rows
    c=controller(-.1);c.current_waypoint=np.ones(7)*.1
    with pytest.raises(EpisodeRejected,match="progress fraction"):c._learned_step("transfer")
    assert not c.env.rows


def test_v3_checkpoint_schema_architecture_and_sources_are_strict(tmp_path):
    from sim.auscultation_training.policy_v3 import RouteProgressPolicy,architecture
    from sim.auscultation_training.student_v3 import load_actor
    from sim.auscultation_training.train_bc_v3 import MODEL_CODE_FILES
    p=RouteProgressPolicy(hidden=(8,))
    checkpoint={"schema":"auscultation_bc_policy_v3","state_dict":p.state_dict(),"input_dim":276,
        "hidden":[8],"history_steps":4,"frame_features":list(FRAME_FEATURES),"action_limit_rad":.0125,
        "dataset_manifest_sha256":"a"*64,"motion_parameterization":"checked_route_learned_progress_v3"}
    torch.save(checkpoint,tmp_path/"policy.pt")
    sha=lambda x:hashlib.sha256(x).hexdigest();root=Path(__file__).parents[1]
    manifest={"schema":"auscultation_bc_training_v3","checkpoint_sha256":sha((tmp_path/"policy.pt").read_bytes()),
        "dataset_manifest_sha256":"a"*64,"architecture":architecture([8]),
        "scaler":{"mean":p.input_mean.tolist(),"std":p.input_std.tolist()},"training_speed_scales":[.85],
        "motion_parameterization":"checked_route_learned_progress_v3",
        "code_sha256":{name:sha((root/name).read_bytes()) for name in MODEL_CODE_FILES}}
    data=json.dumps(manifest).encode();(tmp_path/"run_manifest.json").write_bytes(data)
    actor=load_actor(tmp_path,sha(data));prediction=actor.predict_components(history().numpy()[0])
    assert 0<=prediction["progress_fraction"]<=1
    assert not any(parameter.requires_grad for parameter in actor.policy.parameters())
    checkpoint["schema"]="auscultation_bc_policy_v2";torch.save(checkpoint,tmp_path/"policy.pt")
    manifest["checkpoint_sha256"]=sha((tmp_path/"policy.pt").read_bytes())
    data=json.dumps(manifest).encode();(tmp_path/"run_manifest.json").write_bytes(data)
    with pytest.raises(ValueError,match="schema"):load_actor(tmp_path,sha(data))


def test_exact_route_context_is_bound_to_original_control_hash(monkeypatch,tmp_path):
    from sim.auscultation_training import train_bc_v3
    remaining=np.array([-.2,.1,0.,.3,0.,0.,.04]);base=remaining*(.0125*.85/.3)
    controls=[{"training_accepted":True,"recovery":False,"target":"R1","speed_scale":.85,
        "previous_q_command":[0.]*7,"action_joint_delta_rad":(base*.4).tolist()}]
    source=tmp_path/"episode";source.mkdir();blob=json.dumps(controls).encode()
    (source/"control_commands.json").write_bytes(blob)
    sidecar={"rows":[{"target":"R1","speed_scale":.85,"local_waypoint_valid":True,"local_waypoint_q":remaining.tolist()}]}
    sidecar_bytes=json.dumps(sidecar).encode();(tmp_path/"labels.json").write_bytes(sidecar_bytes)
    episode={"source":str(source),"artifact_sha256":{"control_commands.json":hashlib.sha256(blob).hexdigest()},
        "waypoint_annotation_file":"labels.json","waypoint_annotation_sha256":hashlib.sha256(sidecar_bytes).hexdigest(),"episode_id":"fixture"}
    arrays={"actions":np.array([base*.4],np.float32),"episode_index":np.array([0])}
    monkeypatch.setattr(train_bc_v3,"load_training_data",lambda _ :({"episodes":[episode]},b"manifest",arrays.copy()))
    _,_,result,proof=train_bc_v3.load_route_training_data(tmp_path)
    assert result["route_remaining"][0]==pytest.approx(remaining)
    assert proof["route_rows"]==1 and proof["progress_fraction_max"]==pytest.approx(.4)
    (source/"control_commands.json").write_bytes(blob+b" ")
    with pytest.raises(ValueError,match="control fingerprint"):train_bc_v3.load_route_training_data(tmp_path)


def test_interrupted_route_records_only_the_applied_progress():
    c=controller(.4);c.env.failure=True;c.q[2]=.3;c.env.q=c.q.copy()
    c.current_waypoint=c.q+np.array([0.,0.,.01,0.,0.,0.,0.])
    with pytest.raises(EpisodeRejected):c._learned_step("transfer")
    row=c.control_rows[-1]
    assert row["executed_physics_steps"]==1 and not row["command_block_complete"]
    assert row["predicted_route_fraction"]==.4
    assert row["applied_route_fraction"]==pytest.approx(.4/12)
    assert row["training_accepted"] is False


def test_v3_evaluator_rejects_unseen_speed_before_plan_or_simulator(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from sim.auscultation_training import evaluate_bc_v3
    monkeypatch.setattr(evaluate_bc_v3,"load_actor",lambda *_:SimpleNamespace(metadata={"training_speed_scales":[.85]}))
    def unreachable(*args,**kwargs):raise AssertionError("must reject before plan validation")
    monkeypatch.setattr(evaluate_bc_v3,"validate_plan",unreachable)
    with pytest.raises(ValueError,match="training speed"):
        evaluate_bc_v3.main(["--model-dir",str(tmp_path),"--manifest-sha256","a"*64,"--speed-scale","1"])
