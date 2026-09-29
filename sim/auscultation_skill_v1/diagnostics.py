"""Bound evidence and phase/point coverage diagnostics; never a success gate.

Test metrics are withheld unless explicitly requested. These are teacher-state
action diagnostics, not closed-loop, RGB-D, acoustic or hardware validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import TARGETS
from .dataset import (load_dataset, sha, SPLITS, EXECUTOR_CODE_FILES, SAMPLE_SEMANTICS, SAMPLE_WEIGHTING,
                      SAMPLE_ARRAYS, decision_samples, validate_decision_arrays)
from .observations import (ACTION_LIMIT_M_S, ACTION_INWARD_LIMIT_M_S, ACTION_SCALING, FRAME_FEATURES,
                           HISTORY_STEPS, PHYSICS_STEPS, PHASES)

SOURCE_FILES=tuple(dict.fromkeys(("diagnostics.py","dataset.py","observations.py","policy.py","train.py",
                                 "quality.py","run.py","provenance.py","conditions.json")+EXECUTOR_CODE_FILES))
TARGET_FORCE_N=4.
FORCE_DEADBAND_N=.02
NEAR_BOUND_FRACTION=.95


def _hashes(paths):
    return {str(Path(path).resolve()):sha(Path(path).read_bytes()) for path in paths}


def bound_points(arrays, manifest):
    """Recover point names from the exact recorded decisions behind each label."""
    n=len(arrays["actions"]); indices=np.asarray(arrays["episode_index"])
    validate_decision_arrays(arrays)
    ranges=np.asarray(arrays["physics_ranges"]); times=np.asarray(arrays["sample_time_s"])
    if (indices.shape!=(n,) or not np.issubdtype(indices.dtype,np.integer)
            or ranges.shape!=(n,2) or not np.issubdtype(ranges.dtype,np.integer)
            or times.shape!=(n,) or not np.isfinite(times).all()
            or np.any(indices<0) or np.any(indices>=len(manifest["episodes"]))):
        raise ValueError("invalid sample episode or range binding")
    points=[None]*n
    for episode_index,episode in enumerate(manifest["episodes"]):
        path=(Path(episode["episode_dir"])/"control_commands.json").resolve()
        contents=path.read_bytes()
        if episode["source_sha256"].get(str(path))!=sha(contents):
            raise ValueError("point source changed")
        groups=[]; group=None
        for row in json.loads(contents):
            if "skill_observation" not in row:
                group=None
                continue
            if row.get("policy_update") is True:
                group=[]; groups.append(group)
            if group is None: raise ValueError("recorded skill row lacks issued decision")
            group.append(row)
        outcome_path=(Path(episode["episode_dir"])/"outcome.json").resolve(); outcome=outcome_path.read_bytes()
        if episode["source_sha256"].get(str(outcome_path))!=sha(outcome):
            raise ValueError("decision terminal source changed")
        actual=decision_samples(groups,json.loads(outcome))
        decisions={int(row[0]):i for i,row in enumerate(actual["physics_ranges"])}
        for index in np.flatnonzero(indices==episode_index):
            start,end=ranges[index]; source_index=decisions.get(int(start))
            if source_index is None:
                raise ValueError("sample range does not begin at a recorded decision")
            row=groups[source_index][0]
            if (row.get("target") not in TARGETS
                    or any(not np.array_equal(actual[key][source_index],arrays[key][index]) for key in SAMPLE_ARRAYS)):
                raise ValueError("sample observation, range, duration or action differs from bound decision")
            points[index]=row["target"]
    return points


def _quantiles(values):
    if not len(values): return None
    return {"p"+str(q):float(v) for q,v in zip((0,5,50,95,100),np.quantile(values,[0,.05,.5,.95,1]))}


def _error(predictions, labels):
    if predictions is None or not len(labels): return None
    error=predictions-labels
    return dict(mae_m_s=float(np.mean(np.abs(error))),rmse_m_s=float(np.sqrt(np.mean(error**2))),
                bias_m_s=float(np.mean(error)),max_abs_m_s=float(np.max(np.abs(error))),
                mse_scaled_by_0_004=float(np.mean((error/ACTION_LIMIT_M_S)**2)),
                label_match_fraction_at_1e_9=float(np.mean(np.abs(error)<=1e-9)))


def summarize_arrays(arrays, points, predictions=None, *, include_test=False):
    """Pure statistics. Production callers must use diagnose for qualification."""
    x=np.asarray(arrays["observations"]); y=np.asarray(arrays["actions"],float)
    split=np.asarray(arrays["split"]); n=len(y); points=np.asarray(points)
    if (x.shape!=(n,HISTORY_STEPS,len(FRAME_FEATURES)) or y.shape!=(n,1)
            or split.shape!=(n,) or points.shape!=(n,) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isin(split,list(SPLITS.values())).all()
            or not np.isin(points,TARGETS).all()):
        raise ValueError("invalid diagnostic sample arrays")
    validate_decision_arrays(arrays)
    steps=np.asarray(arrays["decision_steps"]); reasons=np.asarray(arrays["decision_reason"])
    first_decisions=np.asarray(arrays["is_first_decision"])
    frame=x[:,-1]; y=y[:,0]
    column=lambda key:frame[:,FRAME_FEATURES.index(key)]
    phase_values=frame[:,[FRAME_FEATURES.index("phase_"+phase) for phase in PHASES]]
    if not np.isin(phase_values,[0.,1.]).all() or not np.all(phase_values.sum(1)==1):
        raise ValueError("invalid phase encoding")
    phases=np.asarray(PHASES)[np.argmax(phase_values,axis=1)]
    speeds=column("speed_scale")
    inward=ACTION_INWARD_LIMIT_M_S*speeds; outward=np.full(n,ACTION_LIMIT_M_S)
    if np.any((speeds<.7)|(speeds>1.)) or np.any(y < -inward-1e-9) or np.any(y>outward+1e-9):
        raise ValueError("label outside declared action bounds")
    included=np.ones(n,dtype=bool) if include_test else split!=SPLITS["test"]
    predicted=None if predictions is None else np.asarray(predictions,float)
    if predicted is not None and (predicted.shape!=(n,) or not np.isfinite(predicted[included]).all()
            or np.any(predicted[included]<-inward[included]-1e-9)
            or np.any(predicted[included]>outward[included]+1e-9)):
        raise ValueError("nonfinite prediction or prediction outside action bounds")
    previous=column("previous_desired_normal_velocity_m_s").astype(float)
    hall=column("hall_N").astype(float); force_error=hall-TARGET_FORCE_N
    feedback=np.isin(phases,["APPROACH","CONTACT","HOLD"]) & ((hall>.3)|(column("compression_m")>.00015))
    units=lambda action,idx:np.where(action<0,action/inward[idx],action/outward[idx])

    def cell(mask):
        idx=np.flatnonzero(mask & included); labels=y[idx]
        prediction=None if predicted is None else predicted[idx]
        active=idx[feedback[idx]]; err=force_error[active]; action=y[active]
        first=idx[first_decisions[idx]]; first_prediction=None if predicted is None else predicted[first]
        directions=lambda values:dict(inward=int(np.sum(values<0)),zero=int(np.sum(values==0)),outward=int(np.sum(values>0)))
        first_check=dict(samples=len(first),label_direction_counts=directions(y[first]),
            label_m_s_quantiles=_quantiles(y[first]),
            model_direction_counts=None if first_prediction is None else directions(first_prediction),
            prediction_m_s_quantiles=None if first_prediction is None else _quantiles(first_prediction),
            model_error=_error(first_prediction,y[first]),
            outward_start_risk_count=None if first_prediction is None else int(np.sum(first_prediction>0)))
        feedback_counts=dict(samples=len(active),positive_force_error=int(np.sum(err>0)),
            negative_force_error=int(np.sum(err<0)),zero_force_error=int(np.sum(err==0)),
            above_deadband=int(np.sum(err>FORCE_DEADBAND_N)),below_deadband=int(np.sum(err < -FORCE_DEADBAND_N)),
            within_deadband=int(np.sum(np.abs(err)<=FORCE_DEADBAND_N)),
            outward_labels=int(np.sum(action>1e-9)),inward_labels=int(np.sum(action < -1e-9)),
            zero_labels=int(np.sum(np.abs(action)<=1e-9)),
            positive_error_outward_label=int(np.sum((err>FORCE_DEADBAND_N)&(action>1e-9))),
            negative_error_inward_label=int(np.sum((err < -FORCE_DEADBAND_N)&(action < -1e-9))),
            positive_error_without_outward_label=int(np.sum((err>FORCE_DEADBAND_N)&(action<=1e-9))),
            negative_error_without_inward_label=int(np.sum((err < -FORCE_DEADBAND_N)&(action>=-1e-9))))
        bounds={}
        for name,near in (("inward",labels<=-NEAR_BOUND_FRACTION*inward[idx]),
                          ("outward",labels>=NEAR_BOUND_FRACTION*outward[idx])):
            selected=idx[near]; p=None if predicted is None else predicted[selected]
            bounds[name]=dict(samples=len(selected),previous_action_error=_error(previous[selected],y[selected]),
                model_error=_error(p,y[selected]),prediction_unit_quantiles=None if p is None else _quantiles(units(p,selected)))
        return dict(samples=len(idx),hall_N_quantiles=_quantiles(hall[idx]),
            decision_coverage=dict(full_period=int(np.sum(steps[idx]==PHYSICS_STEPS)),partial=int(np.sum(steps[idx]<PHYSICS_STEPS)),
                phase_boundary=int(np.sum(reasons[idx]=="phase_boundary")),successful_terminal=int(np.sum(reasons[idx]=="successful_terminal")),
                duration_s_quantiles=_quantiles(np.asarray(arrays["decision_duration_s"])[idx])),
            first_target_decisions=first_check,
            force_error_N_quantiles=_quantiles(force_error[idx]),
            force_error_sign_counts=dict(positive=int(np.sum(force_error[idx]>0)),
                negative=int(np.sum(force_error[idx]<0)),zero=int(np.sum(force_error[idx]==0))),
            force_feedback=feedback_counts,
            action_sign_counts=dict(outward=int(np.sum(labels>1e-9)),inward=int(np.sum(labels < -1e-9)),
                                    zero=int(np.sum(np.abs(labels)<=1e-9))),
            previous_action_error=_error(previous[idx],labels),model_error=_error(prediction,labels),
            label_unit_quantiles=_quantiles(units(labels,idx)),
            prediction_unit_quantiles=None if prediction is None else _quantiles(units(prediction,idx)),near_bounds=bounds)

    names=list(SPLITS) if include_test else ["train","validation"]
    result=dict(by_split={},by_split_phase={},by_split_point={},by_split_point_phase={},by_split_decision_extent={},
                test_samples_withheld=0 if include_test else int(np.sum(split==SPLITS["test"])),
                heldout_diagnostics_inspected=bool(include_test and np.any(split==SPLITS["test"])))
    for name in names:
        mask=split==SPLITS[name]
        result["by_split"][name]=cell(mask)
        result["by_split_decision_extent"][name]={"full_period":cell(mask & (steps==PHYSICS_STEPS)),
                                                  "partial":cell(mask & (steps<PHYSICS_STEPS))}
        result["by_split_phase"][name]={phase:cell(mask & (phases==phase)) for phase in PHASES}
        result["by_split_point"][name]={point:cell(mask & (points==point)) for point in TARGETS}
        result["by_split_point_phase"][name]={point:{phase:cell(mask & (points==point)&(phases==phase))
                                                      for phase in PHASES} for point in TARGETS}
    result["first_decision_checks"]=[dict(sample_index=int(i),split=next(name for name,code in SPLITS.items() if code==split[i]),
        episode_index=None if "episode_index" not in arrays else int(arrays["episode_index"][i]),
        target=str(points[i]),phase=str(phases[i]),decision_steps=int(steps[i]),decision_reason=str(reasons[i]),
        physics_range=np.asarray(arrays["physics_ranges"])[i].tolist(),sample_time_s=float(arrays["sample_time_s"][i]),
        label_m_s=float(y[i]),prediction_m_s=None if predicted is None else float(predicted[i]),
        outward_start_risk=None if predicted is None else bool(predicted[i]>0))
        for i in np.flatnonzero(first_decisions & included)]
    return result


def diagnose(dataset_dir, model_dir=None, output=None, *, include_test=False):
    """Refresh qualified evidence before/after diagnostics; never change inputs."""
    directory=Path(dataset_dir).resolve()
    paths={name:directory/name for name in ("manifest.json","dataset.npz","conditions.json")}
    bindings=_hashes(paths.values())
    source_paths={name:Path(__file__).with_name(name) for name in SOURCE_FILES}
    source_hashes=_hashes(source_paths.values())
    arrays,manifest=load_dataset(directory)
    points=bound_points(arrays,manifest)
    predictions=None; model_record=None
    if model_dir is not None:
        from .policy import load_policy
        model_dir=Path(model_dir).resolve()
        model_paths={name:model_dir/name for name in ("run_manifest.json","policy.pt")}
        model_hashes=_hashes(model_paths.values()); bindings.update(model_hashes)
        actor=load_policy(model_dir,model_hashes[str(model_paths["run_manifest.json"])],manifest["contract_sha256"],device="cpu")
        if actor.metadata.get("dataset_manifest_sha256")!=bindings[str(directory/"manifest.json")]:
            raise ValueError("frozen model was trained against a different dataset manifest")
        if actor.metadata.get("checkpoint_sha256")!=model_hashes[str(model_paths["policy.pt"])]:
            raise ValueError("frozen model checkpoint changed")
        selected=np.arange(len(arrays["actions"])) if include_test else np.flatnonzero(arrays["split"]!=SPLITS["test"])
        predictions=np.full(len(arrays["actions"]),np.nan)
        # Batch inference over saved histories is exactly the stateless online MLP.
        import torch
        with torch.inference_mode():
            for start in range(0,len(selected),1024):
                index=selected[start:start+1024]
                values=torch.as_tensor(arrays["observations"][index],device=actor.device)
                predictions[index]=actor.policy(values).cpu().numpy()[:,0]
        model_record=dict(directory=str(model_dir),checkpoint_sha256=actor.metadata["checkpoint_sha256"],
                          model_manifest_sha256=model_hashes[str(model_paths["run_manifest.json"])],
                          training_dataset_manifest_sha256=actor.metadata["dataset_manifest_sha256"],device="cpu")
    result=summarize_arrays(arrays,points,predictions,include_test=include_test)
    for first in result["first_decision_checks"]:
        episode=manifest["episodes"][first["episode_index"]]
        first.update(episode_id=episode["episode_id"],condition_id=episode["condition_id"],family_id=episode["family_id"])
    result.update(schema="auscultation_scalar_diagnostics_v1",dataset=str(directory),model=model_record,
        sample_semantics=SAMPLE_SEMANTICS,sample_weighting=SAMPLE_WEIGHTING,
        action_scaling=ACTION_SCALING,action_limit_m_s=ACTION_LIMIT_M_S,action_inward_limit_m_s=ACTION_INWARD_LIMIT_M_S,
        dataset_sha256={name:bindings[str(path)] for name,path in paths.items()},
        diagnostic_source_sha256={name:source_hashes[str(path.resolve())] for name,path in source_paths.items()},
        episode_evidence_sha256={e["episode_id"]:e["source_sha256"] for e in manifest["episodes"]},
        contract_sha256=manifest["contract_sha256"],executor_code_sha256=manifest["executor_code_sha256"],
        dataset_evidence_verified=True,closed_loop_success_established=False,
        definitions=dict(target_force_N=TARGET_FORCE_N,force_error="actor Hall observation minus target; not simulator truth",
            force_deadband_N=FORCE_DEADBAND_N,near_action_bound_fraction=NEAR_BOUND_FRACTION,
            action_sign="positive outward; negative inward",unit_action="negative divided by 0.002*speed_scale, positive by fixed 0.004",
            feedback_subset="APPROACH/CONTACT/HOLD with Hall > 0.3 N or compression > 0.00015 m; excludes RETRACT/DONE",
            empty_cell="zero samples and null errors/quantiles; not a passed cell",
            decision_samples="equal-weight issued decisions; actual 1..12 steps, short only at a bound phase refresh or successful terminal",
            full_period_view="decision_steps == 12, reported separately from partial decisions",
            first_decision_risk="positive output at initial maximum executor gap risks immediate domain rejection; report actual output without forcing its sign",
            previous_action_baseline="copy the current observation's previous desired action; no fitting"),
        scope="Qualified teacher-state coverage and action prediction diagnostics only. Per-phase/per-point errors and closed-loop validation are required; aggregate validation MSE is not task success. No RGB-D, acoustic or hardware claim.")
    load_dataset(directory)  # Recheck physical eligibility and all episode bindings.
    for path,digest in {**bindings,**source_hashes}.items():
        if sha(Path(path).read_bytes())!=digest: raise ValueError("diagnostic input or source changed: "+path)
    if output is not None:
        destination=Path(output).resolve()
        protected=[directory,*[Path(e["episode_dir"]).resolve() for e in manifest["episodes"]]]
        if model_dir is not None: protected.append(model_dir)
        if any(destination==root or root in destination.parents for root in protected):
            raise ValueError("diagnostics output must be outside immutable inputs")
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open("x") as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True,type=Path); parser.add_argument("--model",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--include-test",action="store_true",help="Explicitly inspect heldout metrics; never use them for model selection")
    args=parser.parse_args(argv)
    result=diagnose(args.dataset,args.model,args.output,include_test=args.include_test)
    print(json.dumps(dict(output=str(args.output),model_evaluated=result["model"] is not None,
                         heldout_diagnostics_inspected=result["heldout_diagnostics_inspected"])))
    return 0


if __name__=="__main__": raise SystemExit(main())
