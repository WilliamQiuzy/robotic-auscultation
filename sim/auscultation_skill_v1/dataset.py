"""Hash-bound, explicitly eligible scalar imitation data with family splits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .observations import (ACTION_LIMIT_M_S, ACTION_INWARD_LIMIT_M_S, ACTION_SCALING, CONTROL_DT_S, PHYSICS_DT_S,
                           PHYSICS_STEPS, FRAME_FEATURES, HISTORY_STEPS,
                           History, SensorCondition, encode, finite)

SCHEMA = "auscultation_scalar_dataset_v2"
SAMPLE_SEMANTICS = "issued_decisions_max12_phase_or_success_terminal_v1"
SAMPLE_WEIGHTING = "equal_issued_decision"
DECISION_REASONS = ("full_period", "phase_boundary", "successful_terminal")
SAMPLE_ARRAYS = ("observations", "actions", "sample_time_s", "physics_ranges", "decision_steps",
                 "decision_duration_s", "decision_reason", "is_first_decision")
SPLITS = {"train":0, "validation":1, "test":2}
ARTIFACTS = ("expert_trace.jsonl", "control_commands.json", "outcome.json")
DEFAULT_CONDITIONS = Path(__file__).with_name("conditions.json")
# run.py selects the actual collision checker; qualification must bind it too.
EXECUTOR_CODE_FILES = ("controller.py", "curve.py", "executor.py", "config.py", "contract.py", "observations.py",
                       "run.py", "certified_collision.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def current_executor_inventory():
    return {name:sha(Path(__file__).with_name(name).read_bytes()) for name in EXECUTOR_CODE_FILES}


def verify_executor_inventory(inventory):
    if not isinstance(inventory,dict) or inventory!=current_executor_inventory():
        raise ValueError("executor source differs from qualified demonstration inventory")
    return inventory


def captured_executor_inventory(directory, outcome, quality):
    """Read content-addressed objects actually captured by the expert run.

    Only paths inside the episode are used; source-machine paths in a snapshot
    are deliberately irrelevant. The quality report must bind the manifest and
    each captured object, and each qualified object must match the live code.
    """
    directory=Path(directory).resolve()
    relative=outcome.get("input_snapshot")
    if not isinstance(relative,str):
        raise ValueError("missing executor source snapshot")
    snapshot=(directory/relative).resolve()
    if directory not in snapshot.parents:
        raise ValueError("unsafe executor source snapshot path")
    data=snapshot.read_bytes(); snapshot_sha=sha(data)
    relative=str(snapshot.relative_to(directory))
    artifacts=quality.get("artifact_sha256",{})
    if snapshot_sha!=outcome.get("input_snapshot_manifest_sha256") or artifacts.get(relative)!=snapshot_sha:
        raise ValueError("executor source snapshot is not quality-bound")
    files=json.loads(data).get("files",{}); inventory={}; sources={str(snapshot):snapshot_sha}
    for name in EXECUTOR_CODE_FILES:
        digest=files.get("source/contact_skill/"+name,{}).get("sha256","")
        if len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("missing captured executor source: "+name)
        obj=(snapshot.parent/"objects"/digest).resolve()
        if directory not in obj.parents or sha(obj.read_bytes())!=digest or artifacts.get(str(obj.relative_to(directory)))!=digest:
            raise ValueError("executor source object is not quality-bound: "+name)
        inventory[name]=digest; sources[str(obj)]=digest
    verify_executor_inventory(inventory)
    return inventory,sources


def verify_eligibility(entry):
    """Refresh the independent audit, source snapshots and nominal repetitions."""
    from .quality import _read_bound_quality, verify_repetitions
    report=_read_bound_quality(entry["quality_report"])
    if (report.get("sim_control_imitation_eligible") is not True
            or Path(report["episode"]).resolve()!=Path(entry["episode_dir"]).resolve()):
        raise ValueError("fresh quality report is not eligible for this episode")
    repetition=report.get("repetition_report")
    if not isinstance(repetition,dict) or sha(Path(repetition["path"]).read_bytes())!=repetition.get("sha256"):
        raise ValueError("missing or changed repetition eligibility evidence")
    verify_repetitions(repetition["path"])
    return report


def conditions_by_id(document):
    if document.get("schema") != "auscultation_skill_conditions_v1":
        raise ValueError("unsupported conditions schema")
    result, families = {}, {}
    for item in document.get("conditions",[]):
        identifier, family, split = (item.get(k) for k in ("condition_id","family_id","split"))
        if not identifier or identifier in result or not family or split not in SPLITS:
            raise ValueError("invalid or duplicate condition")
        if family in families and families[family] != split:
            raise ValueError("condition family crosses splits")
        speed=float(item.get("speed_scale",0.))
        if not .7 <= speed <= 1.:
            raise ValueError("unsupported condition speed")
        SensorCondition(item)  # Validate supported sensor parameter domains.
        result[identifier]=item; families[family]=split
    if not result:
        raise ValueError("empty condition inventory")
    return result


def _evidence(entry):
    directory, report_path = Path(entry["episode_dir"]).resolve(), Path(entry["quality_report"]).resolve()
    report_bytes=report_path.read_bytes(); report=json.loads(report_bytes)
    if (report.get("schema") != "auscultation_skill_quality_v1"
            or report.get("sim_control_imitation_eligible") is not True):
        raise ValueError("episode is not explicitly imitation eligible")
    contract=report.get("contract_sha256","")
    if len(contract)!=64 or any(c not in "0123456789abcdef" for c in contract):
        raise ValueError("missing source contract hash")
    data={name:(directory/name).read_bytes() for name in ARTIFACTS}
    if any(report.get("artifact_sha256",{}).get(name)!=sha(data[name]) for name in ARTIFACTS):
        raise ValueError("source artifact hash mismatch")
    verify_eligibility(entry)
    sources={str(directory/name):sha(value) for name,value in data.items()}
    sources[str(report_path)]=sha(report_bytes)
    inventory,captured_sources=captured_executor_inventory(directory,json.loads(data["outcome.json"]),report)
    sources.update(captured_sources)
    return data, report, sources, inventory


def _bind_sensor(row, trace, condition):
    obs=row["skill_observation"]
    now=float(finite(row["time_s"],(),"decision timestamp"))
    sampled=float(finite(row.get("observation_time_s"),(),"observation timestamp"))
    hall_time=float(finite(row.get("hall_sample_time_s"),(),"Hall timestamp"))
    noise=float(finite(row.get("hall_noise_N"),(),"Hall noise"))
    if (sampled>now+1e-8 or hall_time>sampled+1e-8 or sampled<0 or hall_time<0
            or now-hall_time > condition["observation_delay_steps"]*CONTROL_DT_S+1e-7
            or abs(noise)>condition["hall_noise_clip_N"]+1e-9):
        raise ValueError("invalid realized sensor delay/noise")
    if not np.isclose(obs["speed_scale"],condition["speed_scale"],atol=1e-12):
        raise ValueError("condition speed was not realized")
    # Initial state has no independent pre-step row. Later samples bind exactly
    # to the selected physics row, not to a future or current undelayed state.
    for timestamp, keys in ((sampled,("q","dq","compression_m","gap_m")),(hall_time,("hall_N",))):
        if np.isclose(timestamp,0.,atol=1e-9):
            continue
        index=int(round(timestamp/PHYSICS_DT_S))-1
        if index<0 or index>=len(trace) or not np.isclose(trace[index]["time_s"],timestamp,atol=1e-7,rtol=0):
            raise ValueError("sensor timestamp has no physics sample")
        source=trace[index]
        for key in keys:
            expected=np.asarray(source[key],float)+(noise if key=="hall_N" else 0.)
            if not np.allclose(np.asarray(obs[key],float),expected,atol=1e-7,rtol=0):
                raise ValueError("actor sensor differs from selected physics sample: "+key)


def decision_samples(groups, outcome):
    """Extract actual issued decisions after row validation, not qualification.

    A short group needs a contiguous phase refresh or an independently recorded
    successful contact endpoint. Durations/reasons are audit metadata, never
    policy inputs. This helper does not make a failed episode training-eligible.
    """
    history=History(); observations=[]; actions=[]; times=[]; ranges=[]; steps=[]; reasons=[]; firsts=[]
    previous=None; seen=set(); ids=set()
    for index,group in enumerate(groups):
        if not group or len(group)>PHYSICS_STEPS:
            raise ValueError("decision must have 1..12 executed physics steps")
        first,last=group[0],group[-1]; target=first["target"]; phase=first["skill_observation"]["phase"]
        start,end=first["physics_row"],last["physics_row_end_exclusive"]
        if (end-start!=len(group) or first.get("policy_update") is not True
                or type(first.get("policy_group_id")) is not int
                or first.get("policy_group_id") in ids
                or any(r["physics_row"]!=start+i or r["physics_row_end_exclusive"]!=start+i+1
                       or r.get("executed_physics_steps")!=1 for i,r in enumerate(group))):
            raise ValueError("invalid issued decision coverage or identity")
        ids.add(first["policy_group_id"])
        following=groups[index+1][0] if index+1<len(groups) else None
        phase_boundary=(following is not None and following["target"]==target
                        and following["physics_row"]==end and following.get("policy_update") is True
                        and following["skill_observation"]["phase"]!=phase)
        reason="full_period"
        if len(group)<PHYSICS_STEPS:
            if phase_boundary:
                reason="phase_boundary"
            else:
                points=[p for p in outcome.get("points",[]) if p.get("target")==target]
                curves=[c for c in outcome.get("contact_curves",[]) if c.get("target")==target]
                terminal=(phase=="DONE" and len(points)==len(curves)==1
                          and points[0].get("success") is True and points[0].get("final_phase")=="DONE"
                          and points[0].get("contact_end_physics_row")==end
                          and curves[0].get("motion_complete") is True
                          and curves[0].get("end_physics_row_exclusive")==end
                          and not any(g[0]["target"]==target for g in groups[index+1:]))
                if not terminal:
                    raise ValueError("short decision lacks a real phase boundary or successful terminal")
                reason="successful_terminal"
        if previous is not None and start!=previous:
            history.reset()
        observations.append(history.push(first["skill_observation"],target))
        actions.append([first["desired_normal_velocity_m_s"]]); times.append(first["time_s"])
        ranges.append([start,end]); steps.append(len(group)); reasons.append(reason); firsts.append(target not in seen)
        seen.add(target); previous=end
    if not actions:
        raise ValueError("no executed skill decisions")
    return dict(observations=np.asarray(observations,np.float32),actions=np.asarray(actions,np.float32),
                sample_time_s=np.asarray(times,np.float64),physics_ranges=np.asarray(ranges,np.int64),
                decision_steps=np.asarray(steps,np.int16),decision_duration_s=np.asarray(steps,np.float64)*PHYSICS_DT_S,
                decision_reason=np.asarray(reasons,dtype="U20"),is_first_decision=np.asarray(firsts,bool))


def validate_decision_arrays(arrays):
    n=len(arrays["actions"])
    steps=np.asarray(arrays["decision_steps"]); durations=np.asarray(arrays["decision_duration_s"])
    ranges=np.asarray(arrays["physics_ranges"]); reasons=np.asarray(arrays["decision_reason"])
    firsts=np.asarray(arrays["is_first_decision"]); times=np.asarray(arrays["sample_time_s"])
    if (steps.shape!=(n,) or not np.issubdtype(steps.dtype,np.integer) or np.any((steps<1)|(steps>PHYSICS_STEPS))
            or ranges.shape!=(n,2) or not np.issubdtype(ranges.dtype,np.integer) or np.any(ranges[:,0]<0)
            or not np.array_equal(ranges[:,1]-ranges[:,0],steps)
            or durations.shape!=(n,) or not np.isfinite(durations).all()
            or not np.allclose(durations,steps*PHYSICS_DT_S,atol=1e-12,rtol=0)
            or reasons.shape!=(n,) or not np.isin(reasons,DECISION_REASONS).all()
            or not np.array_equal(reasons=="full_period",steps==PHYSICS_STEPS)
            or firsts.shape!=(n,) or firsts.dtype!=np.dtype(bool)
            or times.shape!=(n,) or not np.isfinite(times).all()):
        raise ValueError("invalid decision duration, range, reason or first-decision metadata")


def decision_counts(arrays):
    return {"issued":len(arrays["actions"]),"full_period":int(np.sum(arrays["decision_steps"]==PHYSICS_STEPS)),
            "partial":int(np.sum(arrays["decision_steps"]<PHYSICS_STEPS)),
            "first_per_target":int(np.sum(arrays["is_first_decision"]))}


def load_episode(entry, conditions):
    """Require separate quality evidence, exact condition realization and labels.

    Executor v/a/jerk shaping is the declared actuator. Hard guard intervention,
    teacher rescue, recovery and failed episodes never supply training labels.
    Every actually executed decision is supervised once, including justified
    short phase-boundary/terminal decisions. Never pad a short hold to 12 steps.
    """
    allowed=conditions_by_id(conditions)
    data,report,sources,executor_inventory=_evidence(entry)
    outcome=json.loads(data["outcome.json"])
    if (outcome.get("success") is not True or outcome.get("dataset_role")!="expert_demonstration"
            or outcome.get("expert_demonstration") is not True):
        raise ValueError("not a successful explicit expert demonstration")
    condition=allowed.get(outcome.get("condition_id"))
    if condition is None or outcome.get("condition_parameters")!=condition or outcome.get("family_id")!=condition["family_id"]:
        raise ValueError("episode condition/family differs from frozen inventory")
    if not isinstance(outcome.get("reset_id"),str) or not outcome["reset_id"]:
        raise ValueError("missing independent reset identity")
    rows=json.loads(data["control_commands.json"])
    trace=[json.loads(line) for line in data["expert_trace.jsonl"].splitlines()]
    if not rows or not trace:
        raise ValueError("empty episode")
    times=np.asarray([r["time_s"] for r in trace],float)
    if not np.isfinite(times).all() or not np.allclose(np.diff(times),PHYSICS_DT_S,atol=1e-7,rtol=0):
        raise ValueError("invalid physics timestamps")
    cursor=0; groups=[]; group=None
    sensor_times=[]; sensor_target=None
    noise_rng=np.random.default_rng(condition["seed"])
    for row in rows:
        start,end=row.get("physics_row"),row.get("physics_row_end_exclusive")
        if (type(start) is not int or type(end) is not int or start!=cursor or end<=start or end>len(trace)
                or row.get("executed_physics_steps")!=end-start or row.get("command_block_complete") is not True
                or not np.isclose(row.get("action_duration_s",0),(end-start)*PHYSICS_DT_S,atol=1e-9)
                or not np.isclose(row.get("time_s",-1),times[start]-PHYSICS_DT_S,atol=1e-7,rtol=0)):
            raise ValueError("control/physics coverage or timestamp mismatch")
        cursor=end
        if row.get("recovery") is not False or row.get("hard_clamp_intervened",False) or row.get("failure",False):
            raise ValueError("failure, recovery or hard intervention in demonstration")
        is_skill="skill_observation" in row or "policy_update" in row
        if not is_skill:
            group=None
            continue
        if (end-start!=1 or type(row.get("policy_update")) is not bool
                or row.get("controller_source")!="analytic_skill_expert"
                or row.get("hard_clamp_intervened") is not False or row.get("failure") is not False):
            raise ValueError("invalid skill action source or execution record")
        encode(row["skill_observation"])
        if (row.get("phase")!=row["skill_observation"]["phase"]
                or trace[start].get("phase")!=row["phase"] or trace[start].get("target")!=row["target"]):
            raise ValueError("actor phase differs from executed phase trace")
        if row["policy_update"]:
            _bind_sensor(row,trace,condition)
            if row["target"]!=sensor_target:
                sensor_times=[]; sensor_target=row["target"]
            sensor_times.append(row["time_s"])
            expected_time=sensor_times[max(0,len(sensor_times)-1-condition["observation_delay_steps"])]
            sigma,clip=condition["hall_noise_std_N"],condition["hall_noise_clip_N"]
            expected_noise=float(np.clip(noise_rng.normal(0.,sigma),-clip,clip)) if sigma else 0.
            if (not np.isclose(row["observation_time_s"],expected_time,atol=1e-7,rtol=0)
                    or not np.isclose(row["hall_sample_time_s"],expected_time,atol=1e-7,rtol=0)
                    or not np.isclose(row["hall_noise_N"],expected_noise,atol=1e-12,rtol=0)):
                raise ValueError("declared sensor condition was not realized")
            group=[]; groups.append(group)
        if group is None:
            raise ValueError("skill execution without decision")
        group.append(row)
        first=group[0]
        desired=float(finite(row.get("desired_normal_velocity_m_s"),(),"desired action"))
        applied=float(finite(row.get("applied_normal_velocity_m_s"),(),"applied velocity"))
        lower=-ACTION_INWARD_LIMIT_M_S*condition["speed_scale"]
        upper=ACTION_LIMIT_M_S
        if desired<lower-1e-9 or desired>upper+1e-9 or applied<lower-1e-9 or applied>upper+1e-9:
            raise ValueError("scalar action exceeds declared action space")
        if (row.get("policy_group_id")!=first.get("policy_group_id") or row.get("target")!=first.get("target")
                or row["skill_observation"]!=first["skill_observation"]
                or not np.isclose(desired,first["desired_normal_velocity_m_s"],atol=1e-12,rtol=0)):
            raise ValueError("held action or target changes inside decision")
    if cursor!=len(trace):
        raise ValueError("uncovered physics rows")
    samples=decision_samples(groups,outcome)
    validate_decision_arrays(samples)
    x,y=samples["observations"],samples["actions"]
    return {"episode_id":sha(data["expert_trace.jsonl"]),"reset_id":outcome["reset_id"],
            "episode_dir":str(Path(entry["episode_dir"]).resolve()),
            "quality_report":str(Path(entry["quality_report"]).resolve()),
            "family_id":condition["family_id"],"condition_id":condition["condition_id"],"split":condition["split"],
            "contract_sha256":report["contract_sha256"],"source_sha256":sources,
            "executor_code_sha256":executor_inventory,
            "action_scaling":ACTION_SCALING,"sample_semantics":SAMPLE_SEMANTICS,
            **samples,"decision_counts":decision_counts(samples),
            "actor_trajectory_sha256":sha(x.tobytes()+y.tobytes())}


def validate_episode_inventory(episodes):
    """The builder and every loader preserve family isolation and one reset once."""
    if not episodes:
        raise ValueError("empty episode inventory")
    for key in ("episode_id","reset_id"):
        if len({e[key] for e in episodes})!=len(episodes):
            raise ValueError("duplicate "+key)
    if len({e["contract_sha256"] for e in episodes})!=1:
        raise ValueError("mixed executor/source contracts")
    if any(e["executor_code_sha256"]!=episodes[0]["executor_code_sha256"] for e in episodes):
        raise ValueError("mixed qualified executor source inventories")
    trajectories={}
    for e in episodes:
        trajectories.setdefault(e["actor_trajectory_sha256"],set()).add(e["split"])
    if any(len(splits)>1 for splits in trajectories.values()):
        raise ValueError("identical actor trajectory crosses splits")
    if not {"train","validation"}.issubset({e["split"] for e in episodes}):
        raise ValueError("independent train and validation families required")


def build_dataset(entries, output, conditions_path=DEFAULT_CONDITIONS):
    condition_bytes=Path(conditions_path).read_bytes(); conditions=json.loads(condition_bytes)
    episodes=[load_episode(entry,conditions) for entry in entries]
    validate_episode_inventory(episodes)
    arrays={key:np.concatenate([e[key] for e in episodes]) for key in SAMPLE_ARRAYS}
    arrays["episode_index"]=np.concatenate([np.full(len(e["actions"]),i,np.int32) for i,e in enumerate(episodes)])
    arrays["split"]=np.concatenate([np.full(len(e["actions"]),SPLITS[e["split"]],np.uint8) for e in episodes])
    train=arrays["observations"][arrays["split"]==0].reshape(int((arrays["split"]==0).sum()),-1)
    mean,std=train.mean(0,dtype=np.float64),train.std(0,dtype=np.float64)
    std=np.where(std<1e-6,1.,std)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/"dataset.npz",**arrays)
    (output/"conditions.json").write_bytes(condition_bytes)
    records=[{k:v for k,v in e.items() if not isinstance(v,np.ndarray)} for e in episodes]
    for record,e in zip(records,episodes): record["samples"]=len(e["actions"])
    manifest={"schema":SCHEMA,"scope":"fixed-scene state BC scalar contact skill; no RGB-D encoder, routes or acoustic rewards",
              "frame_features":list(FRAME_FEATURES),"history_steps":HISTORY_STEPS,
              "control_dt_s":CONTROL_DT_S,"maximum_physics_steps_per_decision":PHYSICS_STEPS,
              "sample_semantics":SAMPLE_SEMANTICS,"sample_weighting":SAMPLE_WEIGHTING,"phase_boundary_refresh":True,
              "full_period_view":"decision_steps == 12","decision_counts":decision_counts(arrays),
              "action":"issued desired outward normal velocity m/s; 1..12 actual physics steps before phase refresh or successful terminal",
              "action_limit_m_s":ACTION_LIMIT_M_S,"action_inward_limit_m_s":ACTION_INWARD_LIMIT_M_S,
              "action_scaling":ACTION_SCALING,
              "split_labels":SPLITS,"episodes":records,
              "contract_sha256":episodes[0]["contract_sha256"],"conditions_sha256":sha(condition_bytes),
              "executor_code_sha256":episodes[0]["executor_code_sha256"],
              "scaler":{"fit_split":"train_only","mean":mean.tolist(),"std":std.tolist()},
              "arrays_sha256":sha((output/"dataset.npz").read_bytes()),"sample_count":len(arrays["actions"])}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def load_dataset(directory, verify_sources=True):
    directory=Path(directory); manifest=json.loads((directory/"manifest.json").read_bytes())
    data=(directory/"dataset.npz").read_bytes()
    if (manifest.get("schema")!=SCHEMA or manifest.get("arrays_sha256")!=sha(data)
            or manifest.get("sample_semantics")!=SAMPLE_SEMANTICS or manifest.get("sample_weighting")!=SAMPLE_WEIGHTING
            or manifest.get("maximum_physics_steps_per_decision")!=PHYSICS_STEPS
            or manifest.get("phase_boundary_refresh") is not True
            or manifest.get("frame_features")!=list(FRAME_FEATURES)
            or manifest.get("history_steps")!=HISTORY_STEPS
            or manifest.get("action_limit_m_s")!=ACTION_LIMIT_M_S
            or manifest.get("action_inward_limit_m_s")!=ACTION_INWARD_LIMIT_M_S
            or manifest.get("action_scaling")!=ACTION_SCALING
            or manifest.get("conditions_sha256")!=sha((directory/"conditions.json").read_bytes())):
        raise ValueError("dataset schema or artifact mismatch")
    verify_executor_inventory(manifest.get("executor_code_sha256"))
    validate_episode_inventory(manifest["episodes"])
    if any(e.get("contract_sha256")!=manifest.get("contract_sha256") for e in manifest["episodes"]):
        raise ValueError("episode contract differs from dataset")
    if any(e.get("executor_code_sha256")!=manifest["executor_code_sha256"] for e in manifest["episodes"]):
        raise ValueError("episode qualified executor source differs from dataset")
    if any(e.get("action_scaling")!=ACTION_SCALING or e.get("sample_semantics")!=SAMPLE_SEMANTICS for e in manifest["episodes"]):
        raise ValueError("episode action scaling or sample semantics differs from dataset")
    reconstructed=[]
    if verify_sources:
        conditions=json.loads((directory/"conditions.json").read_bytes())
        for e in manifest["episodes"]:
            for path,expected in e["source_sha256"].items():
                if sha(Path(path).read_bytes())!=expected:
                    raise ValueError("source evidence changed: "+path)
            fresh=load_episode(e,conditions)
            if (fresh["source_sha256"]!=e["source_sha256"] or fresh["executor_code_sha256"]!=manifest["executor_code_sha256"]
                    or any(fresh[key]!=e[key] for key in ("episode_id","reset_id","condition_id","family_id","split","contract_sha256",
                                                         "actor_trajectory_sha256","decision_counts"))):
                raise ValueError("qualified episode identity or decision coverage changed")
            reconstructed.append(fresh)
        validate_episode_inventory(reconstructed)
    with np.load(directory/"dataset.npz",allow_pickle=False) as archive:
        arrays={k:archive[k].copy() for k in archive.files}
    if set(arrays)!=set(SAMPLE_ARRAYS)|{"episode_index","split"}:
        raise ValueError("dataset decision array schema mismatch")
    validate_decision_arrays(arrays)
    expected_indices=np.concatenate([np.full(e["samples"],i,np.int32) for i,e in enumerate(manifest["episodes"])])
    expected_splits=np.concatenate([np.full(e["samples"],SPLITS[e["split"]],np.uint8) for e in manifest["episodes"]])
    if (not np.array_equal(arrays["episode_index"],expected_indices) or not np.array_equal(arrays["split"],expected_splits)
            or manifest.get("sample_count")!=len(arrays["actions"])):
        raise ValueError("episode/sample split order binding mismatch")
    if manifest.get("decision_counts")!=decision_counts(arrays):
        raise ValueError("dataset decision coverage count mismatch")
    if reconstructed:
        for key in SAMPLE_ARRAYS:
            if not np.array_equal(arrays[key],np.concatenate([e[key] for e in reconstructed])):
                raise ValueError("dataset sample differs from actual issued decision: "+key)
    x,y,split=arrays["observations"],arrays["actions"],arrays["split"]
    indices=arrays["episode_index"]
    if (x.shape!=(len(y),HISTORY_STEPS,len(FRAME_FEATURES)) or y.shape!=(len(x),1) or split.shape!=(len(x),)
            or indices.shape!=(len(x),) or not np.issubdtype(indices.dtype,np.integer)
            or np.any(indices<0) or np.any(indices>=len(manifest["episodes"]))):
        raise ValueError("invalid dataset array dimensions")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(split,list(SPLITS.values())).all():
        raise ValueError("invalid dataset values")
    speeds=x[:,-1,FRAME_FEATURES.index("speed_scale")]
    if (not (split==0).any() or not (split==1).any()
            or np.any(y[:,0]<-ACTION_INWARD_LIMIT_M_S*speeds-1e-9)
            or np.any(y[:,0]>ACTION_LIMIT_M_S+1e-9)):
        raise ValueError("invalid splits or action bounds")
    family_splits={}
    for index,e in enumerate(manifest["episodes"]):
        mask=arrays["episode_index"]==index
        if mask.sum()!=e["samples"] or not np.all(split[mask]==SPLITS[e["split"]]):
            raise ValueError("episode split binding mismatch")
        if e.get("decision_counts")!=decision_counts({key:value[mask] for key,value in arrays.items()}):
            raise ValueError("episode decision coverage count mismatch")
        family_splits.setdefault(e["family_id"],set()).add(e["split"])
    if any(len(value)>1 for value in family_splits.values()):
        raise ValueError("family crosses splits")
    train=x[split==0].reshape(int((split==0).sum()),-1)
    mean,std=train.mean(0,dtype=np.float64),train.std(0,dtype=np.float64); std=np.where(std<1e-6,1.,std)
    if manifest["scaler"].get("fit_split")!="train_only" or any(not np.allclose(manifest["scaler"][k],v,atol=1e-9,rtol=0) for k,v in (("mean",mean),("std",std))):
        raise ValueError("scaler is not training-only")
    return arrays,manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries",required=True,type=Path,help="JSON list of episode_dir and quality_report associations")
    parser.add_argument("--conditions",type=Path,default=DEFAULT_CONDITIONS)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    manifest=build_dataset(json.loads(args.entries.read_text()),args.output,args.conditions)
    print(json.dumps({"samples":manifest["sample_count"],"output":str(args.output)}))


if __name__=="__main__": main()
