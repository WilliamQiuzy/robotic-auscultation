"""CPU-only route-constrained supervised BC using certified v2 precision labels.

The loss compares the resulting seven-joint action directly with the original
expert action. No reward or auxiliary progress target replaces those labels.
Exact route context is recovered only from hash-bound original controls and
verified archived waypoint annotations, never from demonstration time/index.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import torch

from .dataset_v2 import ACTION_LIMIT_RAD,FRAME_FEATURES,HISTORY_STEPS,sha256
from .policy_v3 import RouteProgressPolicy,architecture,MOTION_PARAMETERIZATION,ROUTE_DIRECTION_SOURCE
from .train_bc_v2 import load_training_data

MODEL_CODE_FILES=("dataset.py","dataset_v2.py","policy.py","policy_v3.py","train_bc_v2.py","train_bc_v3.py")


def verify_route_action(remaining,action,speed_scale):
    remaining,action=np.asarray(remaining,float),np.asarray(action,float)
    if (remaining.shape!=(7,) or action.shape!=(7,) or not np.isfinite(remaining).all()
            or not np.isfinite(action).all() or type(speed_scale) not in (int,float) or not .7<=speed_scale<=1.):
        raise ValueError("invalid certified route label")
    limit=ACTION_LIMIT_RAD*speed_scale
    maximum=float(np.max(np.abs(remaining)))
    base=remaining*min(1.,limit/max(maximum,1e-300))
    denominator=float(base@base)
    if denominator==0:
        if np.max(np.abs(action))>1e-12:
            raise ValueError("nonzero action at zero route remainder")
        return 0.,float(np.max(np.abs(action)))
    fraction=float((action@base)/denominator)
    residual=float(np.max(np.abs(action-base*fraction)))
    if residual>1e-10 or fraction < -1e-8 or fraction > 1.+1e-8 or np.max(np.abs(action))>limit+1e-10:
        raise ValueError("certified waypoint action is not collinear forward bounded progression")
    return float(np.clip(fraction,0.,1.)),residual


def load_route_training_data(directory):
    """Add exact planner context; require original certified controls to exist."""
    directory=Path(directory)
    manifest,manifest_bytes,arrays=load_training_data(directory)
    context=np.zeros((len(arrays["actions"]),7),dtype=np.float64)
    fractions=[];residuals=[];bindings=[]
    for episode_index,episode in enumerate(manifest["episodes"]):
        source=Path(episode["source"])/"control_commands.json"
        data=source.read_bytes()
        expected=episode["artifact_sha256"]["control_commands.json"]
        if sha256(data)!=expected:
            raise ValueError("original certified control fingerprint mismatch")
        controls=json.loads(data)
        sidecar_bytes=(directory/episode["waypoint_annotation_file"]).read_bytes()
        if sha256(sidecar_bytes)!=episode["waypoint_annotation_sha256"]:
            raise ValueError("waypoint sidecar changed during route validation")
        sidecar=json.loads(sidecar_bytes)
        indices=np.flatnonzero(arrays["episode_index"]==episode_index)
        if len(controls)!=len(indices) or len(sidecar["rows"])!=len(indices):
            raise ValueError("original route label count mismatch")
        count=0
        for local,index in enumerate(indices):
            row,label=controls[local],sidecar["rows"][local]
            action=np.asarray(row.get("action_joint_delta_rad"),float)
            if (row.get("training_accepted") is not True or row.get("recovery") is not False
                    or row.get("target")!=label.get("target") or row.get("speed_scale")!=label.get("speed_scale")
                    or action.shape!=(7,) or not np.isfinite(action).all()
                    or not np.array_equal(action.astype(np.float32),arrays["actions"][index])):
                raise ValueError("original certified action differs from bound dataset")
            if label["local_waypoint_valid"]:
                prior=np.asarray(row["previous_q_command"],float)
                goal=np.asarray(label["local_waypoint_q"],float)
                if prior.shape!=(7,) or goal.shape!=(7,) or not np.isfinite(prior).all() or not np.isfinite(goal).all():
                    raise ValueError("invalid exact route command context")
                remaining=goal-prior
                fraction,residual=verify_route_action(remaining,action,row["speed_scale"])
                context[index]=remaining;fractions.append(fraction);residuals.append(residual);count+=1
        bindings.append({"episode_id":episode["episode_id"],"control_commands_sha256":expected,
            "waypoint_annotation_sha256":episode["waypoint_annotation_sha256"],"verified_route_rows":count})
    if not fractions:
        raise ValueError("no certified route progression labels")
    proof={"verification":"original double-precision command labels; forward collinearity residual <=1e-10 rad; no relabeling",
        "route_rows":len(fractions),"progress_fraction_min":min(fractions),"progress_fraction_max":max(fractions),
        "maximum_collinearity_residual_rad":max(residuals),"source_bindings":bindings,
        "route_remaining_float64_sha256":sha256(context.astype("<f8").tobytes())}
    arrays["route_remaining"]=context
    return manifest,manifest_bytes,arrays,proof


def fit_supervised(policy,observations,actions,remaining,epochs=200,batch_size=256,learning_rate=.001,seed=0):
    if epochs<=0 or batch_size<=0 or not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError("positive training settings required")
    x=torch.as_tensor(observations,dtype=torch.float32,device="cpu")
    y=torch.as_tensor(actions,dtype=torch.float32,device="cpu")
    context=torch.as_tensor(remaining,dtype=torch.float32,device="cpu")
    if len(x)==0 or y.shape!=(len(x),7) or context.shape!=y.shape or not torch.isfinite(y).all() or torch.max(torch.abs(y))>ACTION_LIMIT_RAD+1e-8:
        raise ValueError("invalid bounded supervised labels")
    policy.to("cpu").train()
    generator=torch.Generator(device="cpu").manual_seed(seed)
    optimizer=torch.optim.Adam(policy.parameters(),lr=learning_rate)
    losses=[]
    for _ in range(epochs):
        permutation=torch.randperm(len(x),generator=generator)
        for start in range(0,len(x),batch_size):
            indices=permutation[start:start+batch_size];optimizer.zero_grad()
            loss=torch.mean(((policy(x[indices],context[indices])-y[indices])/ACTION_LIMIT_RAD)**2)
            if not torch.isfinite(loss):raise ValueError("nonfinite training loss")
            loss.backward();optimizer.step()
            if any(not torch.isfinite(p).all() for p in policy.parameters()):raise ValueError("nonfinite trained parameter")
        with torch.no_grad():
            total,count=0.,0
            for start in range(0,len(x),batch_size):
                error=(policy(x[start:start+batch_size],context[start:start+batch_size])-y[start:start+batch_size])/ACTION_LIMIT_RAD
                total+=torch.sum(error**2).item();count+=error.numel()
            if not np.isfinite(total/count):raise ValueError("nonfinite final loss")
            losses.append(total/count)
    return losses


def action_metrics(policy,observations,actions,remaining,batch_size=1024):
    policy.eval();predictions=[]
    with torch.inference_mode():
        for start in range(0,len(actions),batch_size):
            x=torch.as_tensor(observations[start:start+batch_size],dtype=torch.float32)
            context=torch.as_tensor(remaining[start:start+batch_size],dtype=torch.float32)
            predictions.append(policy(x,context).numpy())
    error=np.concatenate(predictions)-actions
    return {"sample_count":len(actions),"mae_rad":float(np.abs(error).mean()),
        "rmse_rad":float(np.sqrt(np.mean(error**2))),"per_joint_mae_rad":np.abs(error).mean(0).tolist(),
        "max_abs_error_rad":float(np.max(np.abs(error))),"scope":"held-out action error only; no closed-loop result"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",type=Path);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=200);parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--learning-rate",type=float,default=.001);parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--threads",type=int,default=1);args=parser.parse_args(argv)
    if not 1<=args.epochs<=500 or not 1<=args.batch_size<=4096 or not 1<=args.threads<=4:
        parser.error("CPU limits: epochs1..500, batch1..4096, threads1..4")
    if args.output.exists():raise ValueError("output exists; prior models remain immutable")
    manifest,manifest_bytes,arrays,proof=load_route_training_data(args.dataset)
    x,y,split,remaining=[arrays[k] for k in ("observations","actions","split","route_remaining")]
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);torch.use_deterministic_algorithms(True)
    policy=RouteProgressPolicy(mean=manifest["scaler"]["mean"],std=manifest["scaler"]["std"])
    losses=fit_supervised(policy,x[split==0],y[split==0],remaining[split==0],args.epochs,args.batch_size,args.learning_rate,args.seed)
    metrics=action_metrics(policy,x[split==1],y[split==1],remaining[split==1])
    args.output.mkdir(parents=True,exist_ok=False)
    checkpoint={"schema":"auscultation_bc_policy_v3","state_dict":policy.state_dict(),"input_dim":policy.input_dim,
        "hidden":list(policy.hidden),"history_steps":HISTORY_STEPS,"frame_features":list(FRAME_FEATURES),
        "action_limit_rad":ACTION_LIMIT_RAD,"dataset_manifest_sha256":sha256(manifest_bytes),
        "motion_parameterization":MOTION_PARAMETERIZATION}
    torch.save(checkpoint,args.output/"policy.pt")
    report={"schema":"auscultation_bc_training_v3","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "device":"cpu","seed":args.seed,"threads":args.threads,"architecture":architecture(policy.hidden),
        "motion_parameterization":MOTION_PARAMETERIZATION,"route_planning_learned":False,
        "route_direction_source":ROUTE_DIRECTION_SOURCE,"route_progress_min_fraction":0.,
        "scope":"planner route direction; learned progress; learned contact/withdraw joint actions; no rewards or acoustics",
        "epochs":args.epochs,"batch_size":args.batch_size,"learning_rate":args.learning_rate,"optimizer":"Adam",
        "loss":"MSE of parameterized seven-joint action versus unchanged certified action, divided by action_limit_rad squared",
        "training_loss":losses,"evaluation_action_metrics":metrics,"closed_loop_evaluated":False,
        "scaler":manifest["scaler"],"action_interpolation":manifest["action_interpolation"],"route_label_verification":proof,
        "train_episode_ids":[e["episode_id"] for e in manifest["episodes"] if e["split"]=="train"],
        "evaluation_episode_ids":[e["episode_id"] for e in manifest["episodes"] if e["split"]=="evaluation"],
        "training_speed_scales":sorted({s for e in manifest["episodes"] if e["split"]=="train" for s in e["speed_scales"]}),
        "evaluation_speed_scales":sorted({s for e in manifest["episodes"] if e["split"]=="evaluation" for s in e["speed_scales"]}),
        "speed_generalization_claim":False,"dataset_manifest_sha256":sha256(manifest_bytes),
        "dataset_arrays_sha256":manifest["arrays_sha256"],"checkpoint_sha256":sha256((args.output/"policy.pt").read_bytes()),
        "code_sha256":{name:sha256(Path(__file__).with_name(name).read_bytes()) for name in MODEL_CODE_FILES},
        "runtime":{"numpy":np.__version__,"torch":torch.__version__}}
    (args.output/"run_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"final_training_loss":losses[-1],"held_out_action_mae_rad":metrics["mae_rad"],"closed_loop_evaluated":False}))
    return 0


if __name__=="__main__":raise SystemExit(main())
