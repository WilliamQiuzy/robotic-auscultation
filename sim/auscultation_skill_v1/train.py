"""Actual PyTorch scalar BC; CUDA required by default, validation-only selection."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import load_dataset, sha, SAMPLE_SEMANTICS, SAMPLE_WEIGHTING, decision_counts
from .observations import ACTION_LIMIT_M_S, ACTION_INWARD_LIMIT_M_S, ACTION_SCALING, FRAME_FEATURES, HISTORY_STEPS
from .policy import CODE_FILES, NormalVelocityPolicy


def resolve_device(request):
    device=torch.device(request)
    if device.type not in ("cpu","cuda"):
        raise ValueError("only explicit CPU smoke or CUDA training is supported")
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    return device


def fit(policy, train_x, train_y, validation_x, validation_y, *, device="cuda", epochs=150,
        batch_size=256, learning_rate=.001, seed=0, resume_state=None):
    """One equal-weight sample per issued decision; validation-only selection."""
    selected=resolve_device(device)
    if epochs<1 or batch_size<1 or not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError("invalid optimization configuration")
    torch.manual_seed(seed)
    if selected.type=="cuda": torch.cuda.manual_seed_all(seed)
    tensors=[]
    for x,y in ((train_x,train_y),(validation_x,validation_y)):
        x=np.asarray(x,np.float32); y=np.asarray(y,np.float32)
        if x.shape!=(len(y),HISTORY_STEPS,len(FRAME_FEATURES)) or y.shape!=(len(x),1) or not len(x):
            raise ValueError("invalid supervised dimensions")
        speed=x[:,-1,FRAME_FEATURES.index("speed_scale")]
        if (not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(y[:,0]<-ACTION_INWARD_LIMIT_M_S*speed-1e-9)
                or np.any(y[:,0]>ACTION_LIMIT_M_S+1e-9)):
            raise ValueError("invalid supervised values")
        tensors.extend((torch.as_tensor(x,device=selected),torch.as_tensor(y,device=selected)))
    x,y,vx,vy=tensors; policy.to(selected).train()
    optimizer=torch.optim.Adam(policy.parameters(),lr=learning_rate)
    generator=torch.Generator(device="cpu").manual_seed(seed)
    history=[]; best=float("inf"); best_epoch=0; best_state=None; start=0
    if resume_state is not None:
        if (resume_state.get("schema")!="auscultation_scalar_resume_v1"
                or resume_state.get("action_scaling")!=ACTION_SCALING
                or resume_state.get("action_limit_m_s")!=ACTION_LIMIT_M_S
                or resume_state.get("action_inward_limit_m_s")!=ACTION_INWARD_LIMIT_M_S):
            raise ValueError("resume checkpoint action contract mismatch")
        if resume_state.get("sample_semantics")!=SAMPLE_SEMANTICS or resume_state.get("sample_weighting")!=SAMPLE_WEIGHTING:
            raise ValueError("resume checkpoint sample semantics mismatch")
        policy.load_state_dict(resume_state["last_state_dict"],strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        generator.set_state(resume_state["shuffle_generator_state"].cpu())
        start=resume_state["epochs_completed"]; best=resume_state["best_validation_loss"]
        best_epoch=resume_state["best_epoch"]; best_state=resume_state["best_state_dict"]
        history=copy.deepcopy(resume_state["history"])
        if start>=epochs: raise ValueError("resume must request additional epochs")
    for epoch in range(start,epochs):
        policy.train(); order=torch.randperm(len(x),generator=generator)
        for offset in range(0,len(x),batch_size):
            indices=order[offset:offset+batch_size].to(selected)
            optimizer.zero_grad(set_to_none=True)
            loss=torch.mean(((policy(x[indices])-y[indices])/ACTION_LIMIT_M_S)**2)
            if not torch.isfinite(loss): raise ValueError("nonfinite training loss")
            loss.backward(); optimizer.step()
        policy.eval()
        with torch.inference_mode():
            metrics=[]
            for xx,yy in ((x,y),(vx,vy)):
                total=0.
                for offset in range(0,len(xx),batch_size):
                    error=(policy(xx[offset:offset+batch_size])-yy[offset:offset+batch_size])/ACTION_LIMIT_M_S
                    total+=float(torch.sum(error**2).cpu())
                metrics.append(total/len(xx))
        if not np.isfinite(metrics).all() or any(not torch.isfinite(p).all() for p in policy.parameters()):
            raise ValueError("nonfinite optimization state")
        history.append({"epoch":epoch+1,"train_mse_scaled":metrics[0],"validation_mse_scaled":metrics[1]})
        if metrics[1]<best:
            best=metrics[1]; best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    resume={"schema":"auscultation_scalar_resume_v1","action_scaling":ACTION_SCALING,
            "sample_semantics":SAMPLE_SEMANTICS,"sample_weighting":SAMPLE_WEIGHTING,
            "action_limit_m_s":ACTION_LIMIT_M_S,"action_inward_limit_m_s":ACTION_INWARD_LIMIT_M_S,
            "last_state_dict":{k:v.detach().cpu().clone() for k,v in policy.state_dict().items()},
            "optimizer_state_dict":optimizer.state_dict(),"shuffle_generator_state":generator.get_state(),
            "epochs_completed":epochs,"best_validation_loss":best,"best_epoch":best_epoch,
            "best_state_dict":best_state,"history":history}
    policy.load_state_dict(best_state,strict=True); policy.eval()
    return {"best_epoch":best_epoch,"best_validation_loss":best,"history":history,"resume":resume}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True,type=Path); parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--device",default="cuda"); parser.add_argument("--epochs",type=int,default=150)
    parser.add_argument("--batch-size",type=int,default=256); parser.add_argument("--learning-rate",type=float,default=.001)
    parser.add_argument("--seed",type=int,default=0); parser.add_argument("--resume",type=Path)
    args=parser.parse_args(argv); device=resolve_device(args.device)
    arrays,manifest=load_dataset(args.dataset)
    code_hashes={name:sha(Path(__file__).with_name(name).read_bytes()) for name in CODE_FILES}
    if any(code_hashes.get(name)!=digest for name,digest in manifest["executor_code_sha256"].items()):
        raise ValueError("training executor source differs from qualified data")
    dataset_bytes=(args.dataset/"manifest.json").read_bytes(); dataset_sha=sha(dataset_bytes)
    torch.manual_seed(args.seed)
    model=NormalVelocityPolicy(mean=manifest["scaler"]["mean"],std=manifest["scaler"]["std"])
    resume=None
    if args.resume:
        previous=json.loads((args.resume/"run_manifest.json").read_text())
        if (previous.get("dataset_manifest_sha256")!=dataset_sha or previous.get("seed")!=args.seed
                or previous.get("schema")!="auscultation_scalar_training_v1"
                or previous.get("action_scaling")!=ACTION_SCALING
                or previous.get("action_limit_m_s")!=ACTION_LIMIT_M_S
                or previous.get("action_inward_limit_m_s")!=ACTION_INWARD_LIMIT_M_S
                or previous.get("sample_semantics")!=SAMPLE_SEMANTICS or previous.get("sample_weighting")!=SAMPLE_WEIGHTING
                or previous.get("batch_size")!=args.batch_size or previous.get("learning_rate")!=args.learning_rate
                or previous.get("code_sha256")!=code_hashes
                or previous.get("executor_code_sha256")!=manifest["executor_code_sha256"]):
            raise ValueError("resume changes dataset or optimization contract")
        resume_bytes=(args.resume/"resume.pt").read_bytes()
        if sha(resume_bytes)!=previous.get("resume_sha256"):
            raise ValueError("resume checkpoint hash mismatch")
        resume=torch.load(args.resume/"resume.pt",map_location=device,weights_only=True)
    x,y,split=arrays["observations"],arrays["actions"],arrays["split"]
    result=fit(model,x[split==0],y[split==0],x[split==1],y[split==1],device=args.device,
               epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,seed=args.seed,resume_state=resume)
    # Detect collection/evidence changes during optimization before publishing.
    load_dataset(args.dataset)
    if sha((args.dataset/"manifest.json").read_bytes())!=dataset_sha:
        raise ValueError("dataset manifest changed during training")
    if any(sha(Path(__file__).with_name(name).read_bytes())!=digest for name,digest in code_hashes.items()):
        raise ValueError("learning source changed during training")
    args.output.mkdir(parents=True,exist_ok=False)
    checkpoint={"schema":"auscultation_scalar_policy_v1","state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "hidden":list(model.hidden),"history_steps":HISTORY_STEPS,"frame_features":list(FRAME_FEATURES),
                "action_limit_m_s":ACTION_LIMIT_M_S,"action_inward_limit_m_s":ACTION_INWARD_LIMIT_M_S,
                "action_scaling":ACTION_SCALING,
                "sample_semantics":SAMPLE_SEMANTICS,"sample_weighting":SAMPLE_WEIGHTING,
                "dataset_manifest_sha256":dataset_sha}
    checkpoint["executor_code_sha256"]=manifest["executor_code_sha256"]
    torch.save(checkpoint,args.output/"policy.pt"); torch.save(result["resume"],args.output/"resume.pt")
    run={"schema":"auscultation_scalar_training_v1","scope":"state BC desired normal speed; shared checked executor; no RGB-D trained encoder",
         "device":str(device),"cuda_device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
         "torch_version":str(torch.__version__),"seed":args.seed,"epochs":args.epochs,"batch_size":args.batch_size,
         "learning_rate":args.learning_rate,"best_epoch":result["best_epoch"],"history":result["history"],
         "selection":"validation MSE only","test_evaluated":False,"closed_loop_evaluated":False,
         "contract_sha256":manifest["contract_sha256"],"dataset_manifest_sha256":dataset_sha,
         "executor_code_sha256":manifest["executor_code_sha256"],
         "conditions_sha256":manifest["conditions_sha256"],"scaler":manifest["scaler"],
         "action_limit_m_s":ACTION_LIMIT_M_S,"action_inward_limit_m_s":ACTION_INWARD_LIMIT_M_S,
         "action_scaling":ACTION_SCALING,
         "sample_semantics":SAMPLE_SEMANTICS,"sample_weighting":SAMPLE_WEIGHTING,
         "decision_counts_by_split":{name:decision_counts({key:value[split==code] for key,value in arrays.items()})
                                     for name,code in (("train",0),("validation",1))},
         "architecture":{"type":"scalar_normal_velocity_MLP","input_dim":model.input_dim,"hidden":list(model.hidden),"output_dim":1},
         "split_episode_ids":{name:[e["episode_id"] for e in manifest["episodes"] if e["split"]==name] for name in ("train","validation","test")},
         "checkpoint_sha256":sha((args.output/"policy.pt").read_bytes()),"resume_sha256":sha((args.output/"resume.pt").read_bytes()),
         "code_sha256":code_hashes}
    (args.output/"run_manifest.json").write_text(json.dumps(run,indent=2)+"\n")
    (args.output/"dataset_manifest.json").write_bytes(dataset_bytes)
    print(json.dumps({"output":str(args.output),"device":str(device),"best_epoch":result["best_epoch"],"test_evaluated":False}))
    return 0


if __name__=="__main__": raise SystemExit(main())
