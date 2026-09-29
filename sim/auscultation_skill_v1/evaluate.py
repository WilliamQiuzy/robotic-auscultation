"""Independent frozen-network replay and complete held-out trial accounting."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from .config import TARGETS
from .dataset import ARTIFACTS, sha
from .observations import History
from .policy import load_policy
from .provenance import STUDENT_CLASS, verify_candidate_sources
from .student import load_negative_control


def verify_action_rows(rows, actor, mode, tolerance_m_s=2e-8):
    """Recompute each real decision, then bind ALL held physical commands to it."""
    history=History(); expected=None; group_id=None; input_sha=None
    failures=[]; decisions=steps=0; network_errors=[]; interventions=0
    for index,row in enumerate(rows):
        if "skill_observation" not in row and "policy_update" not in row:
            if row.get("recovery") or row.get("expert_normal_rescue_used"):
                failures.append("recovery_or_rescue:"+str(index))
            continue
        steps+=1
        try:
            if row.get("policy_update") is True:
                values=history.push(row["skill_observation"],row["target"])
                expected=actor.predict(values); input_sha=sha(values.tobytes())
                group_id=row["policy_group_id"]; decisions+=1
            elif row.get("policy_update") is not False or expected is None:
                raise ValueError("missing decision")
            if (row.get("policy_group_id")!=group_id or row.get("policy_input_sha256")!=input_sha
                    or row.get("policy_checkpoint_sha256")!=actor.metadata["checkpoint_sha256"]
                    or row.get("policy_mode")!=mode or row.get("controller_source")!="learned_normal_policy"):
                raise ValueError("action source or observation mismatch")
            values=np.asarray([row.get("policy_prediction_m_s"),row.get("desired_normal_velocity_m_s")],float)
            if not np.isfinite(values).all(): raise ValueError("nonfinite actor action")
            error=float(np.max(np.abs(values-expected))); network_errors.append(error)
            if error>tolerance_m_s: raise ValueError("saved action differs from frozen policy")
            if mode=="zero":
                if row.get("network_prediction_m_s") is not None or np.any(values!=0.):
                    raise ValueError("zero control produced a moving request")
            else:
                actual=float(row.get("network_prediction_m_s",float("nan")))
                if not np.isfinite(actual) or abs(actual-expected)>tolerance_m_s:
                    raise ValueError("network prediction mismatch")
            if row.get("hard_clamp_intervened") is not False:
                interventions+=1; raise ValueError("hard intervention")
            if row.get("recovery") is not False or row.get("expert_normal_rescue_used") is True:
                raise ValueError("expert rescue or recovery")
        except (ValueError,KeyError,TypeError) as error:
            failures.append(str(index)+":"+str(error))
    if not decisions: failures.append("no_network_decisions")
    return {"action_attribution_passed":not failures,"decisions_replayed":decisions,
            "physics_skill_records_checked":steps,"maximum_prediction_error_m_s":max(network_errors,default=0.),
            "tolerance_m_s":tolerance_m_s,"hard_intervention_records":interventions,"failures":failures,
            "scope":"Every decision and held request verified; common executor dynamics are explicit, not teacher force substitution"}


def verify_process_evidence(source, quality):
    """Re-read the parent and post-Kit-close receipt, bound to the quality audit."""
    from .quality import verify_runner_lifecycle
    lifecycle=verify_runner_lifecycle(source)
    if quality.get("worker_lifecycle")!=lifecycle:
        raise ValueError("quality worker lifecycle differs from executed files")
    for name in ("runner_summary.json","worker_completion.json"):
        if quality.get("artifact_sha256",{}).get(name)!=sha((Path(source)/name).read_bytes()):
            raise ValueError("quality process evidence hash mismatch: "+name)
    return lifecycle


def assess_evaluation(outcome, quality, rows, attribution, mode, lifecycle):
    """Task evidence and a normal worker exit are both required for success."""
    points=outcome.get("requested_points",[])
    complete=(bool(points) and [p.get("target") for p in outcome.get("points",[])]==points
              and all(p.get("success") is True for p in outcome.get("points",[])))
    identity=(outcome.get("executed_controller_class")==STUDENT_CLASS and outcome.get("candidate_method")==STUDENT_CLASS
              and outcome.get("dataset_role")=="student_evaluation" and outcome.get("expert_demonstration") is False
              and outcome.get("policy_mode")==mode)
    normal=(lifecycle.get("status")=="passed" and lifecycle.get("verified_from_bound_files") is True
            and type(lifecycle.get("child_returncode")) is int)
    process=bool(normal and lifecycle["child_returncode"]==0 and lifecycle.get("runner_accepted") is True)
    hard=bool(quality.get("hard_simulation_gates_passed") and outcome.get("success") is True and complete)
    clean=(not any(outcome.get(k) for k in ("failure_latched","recovery_used","expert_fallback_used","expert_normal_rescue_used"))
           and not any(row.get("failure") for row in rows))
    success=bool(mode=="policy" and identity and process and hard and quality.get("episode_simulation_gates_passed")
                 and attribution["action_attribution_passed"] and clean)
    # Preserve task completion even when another gate fails: a negative control
    # that completes the task is not evidence of learned-policy necessity.
    negative_complete=bool(mode!="policy" and outcome.get("success") is True and complete)
    negative_pass=bool(mode!="policy" and identity and normal and lifecycle["child_returncode"]==1
                       and lifecycle.get("runner_accepted") is False and outcome.get("success") is False
                       and attribution["action_attribution_passed"] and not negative_complete)
    return {"requested_points":points,"complete_requested_sequence":complete,
            "learned_task_success":success,"assisted_system_success":bool(identity and process and hard),
            "negative_control_task_completed":negative_complete,"negative_control_diagnostic_passed":negative_pass,
            "student_identity_valid":identity,"worker_process_gate_passed":process,"worker_lifecycle":lifecycle}


def audit_episode(episode_dir, quality_report, model_dir, mode="policy", output=None):
    source=Path(episode_dir).resolve(); report_path=Path(quality_report).resolve()
    from .quality import _read_bound_quality
    quality_sha=sha(report_path.read_bytes())
    quality=_read_bound_quality(report_path)
    if Path(quality["episode"]).resolve()!=source:
        raise ValueError("quality report belongs to another episode")
    lifecycle=verify_process_evidence(source,quality)
    data={name:(source/name).read_bytes() for name in (*ARTIFACTS,"runner_summary.json","worker_completion.json")}
    hashes={name:sha(value) for name,value in data.items()}
    if any(quality["artifact_sha256"].get(name)!=value for name,value in hashes.items()):
        raise ValueError("quality artifact hash mismatch")
    outcome=json.loads(data["outcome.json"]); rows=json.loads(data["control_commands.json"])
    model_dir=Path(model_dir); model_manifest=(model_dir/"run_manifest.json").read_bytes()
    loader=load_policy if mode=="policy" else load_negative_control
    args=(model_dir,sha(model_manifest),quality["contract_sha256"])
    actor=loader(*args,device="cpu") if mode=="policy" else loader(*args,mode,device="cpu")
    provenance=verify_candidate_sources(source,outcome)
    if provenance["status"]!="passed": raise ValueError("student provenance failed: "+str(provenance["reasons"]))
    snapshot=json.loads((source/outcome["input_snapshot"]).read_text())
    files=snapshot["files"]
    for key,digest in (("learning/checkpoint",actor.metadata["checkpoint_sha256"]),("learning/manifest",sha(model_manifest))):
        if files.get("source/contact_skill/"+key,{}).get("sha256")!=digest:
            raise ValueError("evaluated model differs from executed captured model")
    attribution=verify_action_rows(rows,actor,mode)
    status=assess_evaluation(outcome,quality,rows,attribution,mode,lifecycle)
    result={"schema":"auscultation_scalar_evaluation_v1","episode":str(source),"condition_id":outcome.get("condition_id"),
            "mode":mode,**status,"action_attribution":attribution,
            "checkpoint_sha256":actor.metadata["checkpoint_sha256"],"model_manifest_sha256":sha(model_manifest),
            "artifact_sha256":hashes,"quality_report":{"path":str(report_path),"sha256":quality_sha},
            "scope":"closed-loop saved physics plus frozen state-policy action attribution; no RGB-D or acoustic result"}
    # Do not let an inference pass certify evidence that changed during replay.
    if any(sha((source/name).read_bytes())!=value for name,value in hashes.items()):
        raise ValueError("episode changed during policy audit")
    if sha(report_path.read_bytes())!=quality_sha:
        raise ValueError("quality report changed during policy audit")
    if output is not None:
        destination=Path(output).resolve()
        if destination==source or source in destination.parents: raise ValueError("evaluation output must be outside immutable episode")
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open("x") as stream: stream.write(json.dumps(result,indent=2)+"\n")
    return result


def summarize_trials(results, expected_conditions, required_points=TARGETS, expected_model=None):
    """Missing, aborted and failed requested trials all stay in the denominator."""
    if len(set(expected_conditions))!=len(expected_conditions) or not expected_conditions:
        raise ValueError("expected condition IDs must be distinct and nonempty")
    model_keys=("checkpoint_sha256","model_manifest_sha256")
    def model_identity(value):
        if not isinstance(value,dict) or any(not isinstance(value.get(key),str)
                or re.fullmatch("[0-9a-f]{64}",value[key]) is None for key in model_keys):
            raise ValueError("missing or invalid frozen model identity")
        return {key:value[key] for key in model_keys}
    pinned=model_identity(expected_model) if expected_model is not None else None
    by_condition={}
    for result in results:
        name=result.get("condition_id")
        if name not in expected_conditions or name in by_condition:
            raise ValueError("unexpected or duplicate trial condition")
        if (result.get("learned_task_success") is True or result.get("assisted_system_success") is True
                or any(key in result for key in model_keys)):
            model=model_identity(result)
            if pinned is None: pinned=model
            elif model!=pinned: raise ValueError("trial model differs from frozen campaign model")
        by_condition[name]=result
    learned=assisted=0
    for result in by_condition.values():
        complete=(result.get("requested_points")==list(required_points)
                  and result.get("complete_requested_sequence") is True)
        learned+=int(complete and result.get("learned_task_success") is True)
        assisted+=int(complete and result.get("assisted_system_success") is True)
    return {"schema":"auscultation_scalar_trial_summary_v1","requested_trials":len(expected_conditions),
            "learned_successes":learned,"learned_success_rate":learned/len(expected_conditions),
            "assisted_successes":assisted,"missing_conditions":[name for name in expected_conditions if name not in by_condition],
            "required_points":list(required_points),"results":results,"pinned_model":pinned,
            "scope":"reserved condition trials; missing or incomplete evidence counts as failure"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode",required=True,type=Path); parser.add_argument("--quality-report",required=True,type=Path)
    parser.add_argument("--model",required=True,type=Path); parser.add_argument("--mode",choices=("policy","zero","untrained"),default="policy")
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    result=audit_episode(args.episode,args.quality_report,args.model,args.mode,args.output)
    print(json.dumps({key:result[key] for key in ("learned_task_success","assisted_system_success","mode")}))
    return 0 if result["learned_task_success"] or result["negative_control_diagnostic_passed"] else 1


if __name__=="__main__": raise SystemExit(main())
