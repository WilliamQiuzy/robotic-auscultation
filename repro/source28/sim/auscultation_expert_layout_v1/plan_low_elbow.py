"""Bounded, scored lower-approach planning using existing audited contact fits.

The ten detector targets and audited tool poses are inputs, never modified.
Every free edge is sampled for full-arm clearance and the explicit posture
envelope. A successful file is geometry evidence only; physics must follow.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .collision import CollisionChecker
from .kinematics import OpenArmKinematics, DEFAULT_URDF_PATH
from .plan_contact_pose import candidate_geometry, bounded_rrt_path, _BudgetExpired, _TimedChecker
from .posture import PostureLimits, PostureChecker
from .pose_ik import solve_posture_ik, IKBudgetExpired
from .scene import ROOT
from .targets import load_targets


def default_limits(kin, base):
    lower, upper = kin.lower_limits.copy(), kin.upper_limits.copy()
    lower[:2] = np.maximum(lower[:2], np.deg2rad(-140.))
    upper[0] = min(upper[0], np.deg2rad(70.))
    shoulder = (base @ kin.link_transforms(np.zeros(7))["openarm_left_link1"])[2, 3]
    return PostureLimits(float(shoulder), tuple(lower), tuple(upper))


def lower_staging(geometry, base, height=.06, outward=.035):
    """Lower pose with an absolute clearance from the target's tangent plane."""
    if not np.isfinite([height, outward]).all() or height < .04 or outward < .03:
        raise ValueError("staging requires at least 40 mm below and 30 mm target-plane clearance")
    normal = np.asarray(geometry["normal_world"], float)
    horizontal_squared = float(normal[0]**2 + normal[1]**2)
    if horizontal_squared < .01:
        raise ValueError("near-vertical surface normal has no bounded lower staging construction")
    # Keep the requested vertical drop exact. Merely adding normal*outward
    # then subtracting height would lift the staging pose on upper-back fits.
    pre_gap = float(np.dot(geometry["pre_world"]-geometry["free_end_world"],normal)+.015)
    horizontal_scale = max(0., (outward-pre_gap+height*normal[2]) / horizontal_squared)
    world = geometry["pre_world"] + np.r_[normal[:2]*horizontal_scale, -height]
    return world, base[:3, :3].T @ (world - base[:3, 3])


def path_score(path, kin, base):
    """Rank all completed candidates, favoring short paths and lower elbows."""
    path = np.asarray(path, float)
    travel = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    shoulder_travel = float(np.abs(np.diff(path[:, :2], axis=0)).sum())
    max_elbow = max(float((base @ kin.link_transforms(q)["openarm_left_link4"])[2, 3]) for q in path)
    return travel + .5 * shoulder_travel + 2. * max_elbow


def candidate_indices(count, budget):
    """Cover the audit's full pose range before spending the remaining budget."""
    if count < 1 or budget < 1:
        raise ValueError("candidate count and budget must be positive")
    spread = np.linspace(0, count-1, min(24, count, budget), dtype=int).tolist()
    return list(dict.fromkeys(spread + list(range(count))))[:budget]


def checked_transfer(checker, limits, home, goal, kin, base, rng, deadline):
    """Try scored elbow-down bridges, then RRT restricted to the posture box."""
    proxy = SimpleNamespace(check=checker.check, kinematics=SimpleNamespace(
        lower_limits=np.asarray(limits.joint_lower_rad), upper_limits=np.asarray(limits.joint_upper_rad)))
    timed = _TimedChecker(proxy, deadline)
    started = time.monotonic(); candidates = []
    # These are only search proposals. Every edge, including the initial bend,
    # receives the identical full-mesh clearance and posture checks.
    ready = []
    for shoulder in (-70., -85., -100.):
        for elbow in (40., 60., 80.):
            ready.append(np.deg2rad([0., shoulder, 0., elbow, 0., 0., 0.]))
    for group in ((0,1,2),(3,4,5,6),(1,3),(0,3)):
        anchor = home.copy(); anchor[list(group)] = goal[list(group)]; ready.append(anchor)
    try:
        if timed.edge(home,goal)["clear"]:
            return [home,goal], {"reason":"direct_checked", "edge_samples":timed.samples,
                                 "seconds":time.monotonic()-started,"scored_bridges":0}
        for anchor in ready:
            if not timed.edge(home,anchor)["clear"]: continue
            if not timed.edge(anchor,goal)["clear"]: continue
            path=[home,anchor,goal]
            candidates.append((path_score(path,kin,base),path))
    except _BudgetExpired:
        pass
    if candidates:
        return min(candidates,key=lambda entry:entry[0])[1], {
            "reason":"scored_structured_bridge", "edge_samples":timed.samples,
            "seconds":time.monotonic()-started,"scored_bridges":len(candidates)}
    if time.monotonic() >= deadline:
        return None,{"reason":"time_budget_exhausted","edge_samples":timed.samples,
                     "seconds":time.monotonic()-started,"scored_bridges":0}
    path, report = bounded_rrt_path(proxy,home,goal,rng,deadline=deadline,max_iterations=1000)
    report["edge_samples"] += timed.samples
    report["scored_bridges"] = 0
    report["seconds"] = time.monotonic()-started
    return path,report


def solve_pose(kin, base, limits, position, rotation, seed, *, compression=0., max_nfev=100,
               deadline=None, minimize_elbow=False):
    """Bounded exact pose equality IK; continuous stages stay near their seed."""
    try:
        return solve_posture_ik(kin,base,limits,position,rotation,seed,compression=compression,
                maxiter=max_nfev,deadline=deadline,minimize_elbow=minimize_elbow,contact=compression>0)
    except IKBudgetExpired as error:
        raise _BudgetExpired from error


def checked_cartesian(kin, base, limits, checker, start_q, start, end, rotation, deadline):
    """Continuous local branch; reject rather than accept an IK posture flip."""
    count = max(2, int(np.ceil(np.linalg.norm(end-start)/.002)))
    route = [np.asarray(start_q).copy()]
    for fraction in np.linspace(0., 1., count+1)[1:]:
        q, good, _, _ = solve_pose(kin, base, limits, start + fraction*(end-start), rotation,
                                  route[-1], max_nfev=80, deadline=deadline)
        if not good or np.max(np.abs(q-route[-1])) > .15:
            return None, "cartesian_ik_or_branch_jump"
        check = checker.edge(route[-1], q)
        if not check["clear"]:
            return None, "cartesian_clearance_or_posture"
        route.append(q)
    return route, None


def plan_target(target, audit, base, kin, checker, limits, *, max_candidates=186,
                seeds_per_candidate=4, max_seconds=300., max_routes=4, pre_gap=.06,
                reference_q=None, hints=None, rng=None):
    if (audit.get("name") != target.name or not 1 <= max_candidates <= 1000
            or not 1 <= seeds_per_candidate <= 12 or not 2 <= max_routes <= 12
            or not np.isfinite(max_seconds) or max_seconds <= 0
            or not np.isfinite(pre_gap) or not .025 <= pre_gap <= .06):
        raise ValueError("invalid bounded planning inputs")
    rng = np.random.default_rng(65) if rng is None else rng
    compression = float(audit["compression_m"])
    if not 0 < compression <= .005:
        raise ValueError("unsupported audit compression")
    started = time.monotonic(); deadline = started + max_seconds
    home = np.zeros(7)
    row = {"name": target.name, "success": False, "contact_success": False,
           "position_world": target.position_world.tolist(), "normal_world": target.normal_world.tolist(),
           "loaded_compression_m": compression, "free_approach_start_gap_m": pre_gap,
           "free_approach_end_gap_m": .015, "geometric_clearance_m": .003,
           "loaded_noncontact_clearance_m": .0003, "joint_edge_sample_step_rad": .015,
           "posture_limits": limits.to_dict(), "approach_design": "lower_outward_staging_then_surface_normal",
           "scope": "sampled posture and collision geometry only; physical contact and dynamics unverified",
           "attempts": [], "limits": {"candidate_attempts": max_candidates,
             "seeds_per_candidate": seeds_per_candidate, "seconds": max_seconds,
             "maximum_scored_routes": max_routes, "ik_method":"analytic_pose_equality_slsqp",
             "ik_max_iterations":100,"planning_joint_buffer_rad":float(np.deg2rad(.2)),
             "planning_elbow_buffer_m":.005}}
    if not checker.check(home)["clear"]:
        row.update(failure_reason="home_collision_or_posture", seconds=0.)
        return row
    routes = []
    lower, upper = np.asarray(limits.joint_lower_rad), np.asarray(limits.joint_upper_rad)
    previous_seed = home if reference_q is None else np.asarray(reference_q,float)
    if previous_seed.shape != (7,) or not np.isfinite(previous_seed).all():
        raise ValueError("invalid reference pose seed")
    hints = [] if hints is None else hints
    if not isinstance(hints,list) or any(not isinstance(hint,dict) for hint in hints):
        raise ValueError("search hints must be a list of candidate records")
    hint_by_candidate = {}
    for hint in hints:
        index = hint.get("candidate_index")
        if not isinstance(index,int) or isinstance(index,bool) or not 0 <= index < len(audit["candidates"]):
            raise ValueError("invalid audited candidate hint")
        for key in ("loaded_q","pre_q"):
            if key not in hint: continue
            q=np.asarray(hint[key],float)
            if q.shape!=(7,) or not np.isfinite(q).all(): raise ValueError("invalid hinted pose seed")
        if "loaded_q" not in hint: raise ValueError("hint requires loaded pose seed")
        hint_by_candidate.setdefault(index,hint)
    row["search_hints"] = hints
    order = list(dict.fromkeys(list(hint_by_candidate)+candidate_indices(len(audit["candidates"]),max_candidates)))[:max_candidates]
    try:
        for ci in order:
            candidate = audit["candidates"][ci]
            geometry = candidate_geometry(target, candidate, base, pre_gap=pre_gap)
            hint=hint_by_candidate.get(ci,{})
            for si in range(seeds_per_candidate):
                if time.monotonic() >= deadline: raise _BudgetExpired
                attempt = {"audit_candidate_index": ci, "seed_index": si}
                row["attempts"].append(attempt)
                seed = np.asarray(hint.get("loaded_q",previous_seed)) if si == 0 else rng.uniform(lower, upper)
                loaded, good, pe, re = solve_pose(kin, base, limits, geometry["center_base"],
                    geometry["rotation_base"], seed, compression=compression, deadline=deadline,minimize_elbow=True)
                if not good:
                    attempt["failure"] = "loaded_ik_or_posture"; continue
                loaded_check = checker.check(loaded, clearance=.0003, allow_patient_contact=True, compression=compression)
                if not loaded_check["clear"]:
                    attempt["failure"] = "loaded_clearance"; continue
                previous_seed = loaded.copy()
                pre, good, _, _ = solve_pose(kin, base, limits, geometry["pre_base"], geometry["rotation_base"],
                    hint.get("pre_q",loaded), deadline=deadline)
                if not good or not checker.check(pre)["clear"]:
                    attempt["failure"] = "prepose_ik_clearance_or_posture"; continue
                stage_world, stage_base = lower_staging(geometry, base)
                descent, failure = checked_cartesian(kin, base, limits, checker, pre,
                    geometry["pre_base"], stage_base, geometry["rotation_base"], deadline)
                if failure:
                    attempt["failure"] = "staging_"+failure; continue
                stage_to_pre = descent[::-1]
                approach, failure = checked_cartesian(kin, base, limits, checker, pre,
                    geometry["pre_base"], geometry["free_end_base"], geometry["rotation_base"], deadline)
                if failure:
                    attempt["failure"] = "approach_"+failure; continue
                loaded, good, pe, re = solve_pose(kin, base, limits, geometry["center_base"], geometry["rotation_base"],
                    approach[-1], compression=compression, deadline=deadline)
                if not good or np.max(np.abs(loaded-approach[-1])) > .4:
                    attempt["failure"] = "loaded_branch_ik_or_discontinuity"; continue
                loaded_check = checker.check(loaded, clearance=.0003, allow_patient_contact=True, compression=compression)
                if not loaded_check["clear"]:
                    attempt["failure"] = "loaded_branch_clearance_or_posture"; continue
                transfer, search = checked_transfer(checker, limits, home, stage_to_pre[0], kin, base, rng,
                    deadline=min(deadline, time.monotonic()+30.))
                attempt["transfer_search"] = search
                if transfer is None:
                    attempt["failure"] = "transfer_"+search["reason"]; continue
                transfer = list(transfer)+stage_to_pre[1:]
                score = path_score(transfer+approach[1:]+[loaded], kin, base)
                result = {"transfer_q": np.asarray(transfer).tolist(), "approach_q": np.asarray(approach).tolist(),
                    "loaded_q": loaded.tolist(), "loaded_endpoint_check": loaded_check,
                    "selected_audit_candidate_index": ci, "selected_fit": candidate,
                    "command_normal_world": geometry["normal_world"].tolist(),
                    "command_rotation_world": geometry["rotation_world"].tolist(),
                    "expected_contact_center_world": geometry["center_world"].tolist(),
                    "expected_contact_gap_m": float(candidate["gap_m"]),
                    "final_free_tcp_world": geometry["free_end_world"].tolist(),
                    "loaded_position_error_m": pe, "loaded_orientation_error_rad": re,
                    "lower_staging_tcp_world": stage_world.tolist(), "lower_staging_vertical_offset_m": .06,
                    "lower_staging_min_target_plane_gap_m": .035,
                    "lower_staging_target_plane_gap_m": float(np.dot(stage_world-(geometry["free_end_world"]-
                        geometry["normal_world"]*.015),geometry["normal_world"])),
                    "lower_staging_q": stage_to_pre[0].tolist(),
                    "route_score": score}
                routes.append(result); attempt["route_score"] = score
                print("LOW_ROUTE",target.name,len(routes),"candidate",ci,"score",round(score,4),flush=True)
                break
            if len(routes) >= max_routes: break
    except _BudgetExpired:
        row["search_stopped"] = "time_budget_exhausted"
    row["seconds"] = time.monotonic()-started
    row["failure_counts"] = dict(Counter(a["failure"] for a in row["attempts"] if "failure" in a))
    row["scored_route_count"] = len(routes)
    row["candidate_scores"] = [{"candidate_index":r["selected_audit_candidate_index"],"score":r["route_score"]} for r in routes]
    # Exhaust the configured search/time/route-count budget before selection.
    # A lone feasible route is still geometrically valid; report its count.
    if routes:
        row.update(min(routes, key=lambda r:r["route_score"])); row["success"] = True
    else:
        row["failure_reason"] = "no_complete_low_routes"
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", nargs="+", default=[side+str(i) for i in range(1,6) for side in ("R","L")])
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/low_elbow_v1/plans")
    parser.add_argument("--max-candidates", type=int, default=186)
    parser.add_argument("--max-seconds", type=float, default=300.)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--max-routes", type=int, default=4)
    parser.add_argument("--posture-limits", type=Path,
                        help="explicit alternative reviewed envelope; default is shoulder-height ceiling")
    parser.add_argument("--pre-gap", type=float,
                        help="explicit 25–60 mm prepose override; otherwise retain each prior checked distance")
    parser.add_argument("--hints", type=Path,
                        help="source-hashed audited candidate IDs and seeds; every result is solved and checked anew")
    args = parser.parse_args(argv)
    base_path = ROOT/"runtime_inspection.json"
    common_paths = {"targets":ROOT/"results/contact_targets.json", "runtime":base_path,
        "kinematics_urdf":DEFAULT_URDF_PATH, "collision_metadata":ROOT/"results/collision_geometry.json",
        "collision_arrays":ROOT/"results/collision_geometry.npz"}
    common_bytes = {label:path.read_bytes() for label,path in common_paths.items()}
    base = np.asarray(json.loads(common_bytes["runtime"])["world_from_left_base"])
    limits_bytes = args.posture_limits.read_bytes() if args.posture_limits else None
    hints_bytes = args.hints.read_bytes() if args.hints else None
    hints_data = json.loads(hints_bytes) if hints_bytes else {}
    if not isinstance(hints_data,dict): raise ValueError("hints must map point names to candidate lists")
    kin = OpenArmKinematics()
    limits = PostureLimits.from_dict(json.loads(limits_bytes),kin) if limits_bytes else default_limits(kin, base)
    checker = PostureChecker(CollisionChecker(kin, base), base, limits)
    targets = {t.name:t for t in load_targets(base, path=ROOT/"results/contact_targets.json")}
    for label,path in common_paths.items():
        if path.read_bytes() != common_bytes[label]:
            raise ValueError("input changed while constructing planner: "+label)
    if args.posture_limits and args.posture_limits.read_bytes() != limits_bytes:
        raise ValueError("posture limits changed while constructing planner")
    if len(set(args.points)) != len(args.points) or any(n not in targets for n in args.points):
        parser.error("points must be distinct detector point names")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = args.output_dir/"posture_limits.json"
    config_bytes = (json.dumps(limits.to_dict(),indent=2)+"\n").encode()
    if config.exists() and config.read_bytes() != config_bytes: raise ValueError("output posture config differs")
    if not config.exists(): config.write_bytes(config_bytes)
    if hints_bytes:
        hint_destination=args.output_dir/"search_hints.json"
        if hint_destination.exists() and hint_destination.read_bytes()!=hints_bytes:
            raise ValueError("output hints differ")
        if not hint_destination.exists(): hint_destination.write_bytes(hints_bytes)
    succeeded = True
    for name in args.points:
        output = args.output_dir/("planned_paths_"+name+".json")
        if output.exists(): raise FileExistsError(output)
        inputs = {"audit": ROOT/"results"/("tool_fit_"+name+"_2.0mm.json"), **common_paths}
        captured = {"audit":inputs["audit"].read_bytes(), **common_bytes}
        hints = hints_data.get(name,[])
        if not isinstance(hints,list) or any(not isinstance(hint,dict) for hint in hints):
            raise ValueError("search hints must be a list of candidate records")
        for hint in hints:
            if hint.get("audit_sha256") != hashlib.sha256(captured["audit"]).hexdigest():
                raise ValueError("hint audit source differs: "+name)
        if args.hints and args.hints.read_bytes()!=hints_bytes: raise ValueError("hints changed before planning")
        reference_path = ROOT/"results"/("planned_paths_"+name+".json")
        reference_bytes = reference_path.read_bytes()
        reference_row = json.loads(reference_bytes)["points"][0]
        if reference_row["name"] != name: raise ValueError("reference prepose point mismatch")
        reference_gap = float(reference_row["free_approach_start_gap_m"])
        pre_gap = reference_gap if args.pre_gap is None else args.pre_gap
        for label,path in inputs.items():
            if path.read_bytes() != captured[label]: raise ValueError("input changed before planning: "+label)
        row = plan_target(targets[name],json.loads(captured["audit"]),base,kin,checker,limits,
             max_candidates=args.max_candidates,seeds_per_candidate=args.seeds,max_seconds=args.max_seconds,
             max_routes=args.max_routes,pre_gap=pre_gap,reference_q=reference_row["loaded_q"],hints=hints)
        if reference_path.read_bytes() != reference_bytes: raise ValueError("reference prepose changed during planning")
        if args.posture_limits and args.posture_limits.read_bytes() != limits_bytes:
            raise ValueError("posture limits changed during planning")
        if args.hints and args.hints.read_bytes()!=hints_bytes: raise ValueError("hints changed during planning")
        if hints_bytes: row["search_hints_file_sha256"] = hashlib.sha256(hints_bytes).hexdigest()
        row["reference_prepose"] = {"file":reference_path.name,"sha256":hashlib.sha256(reference_bytes).hexdigest(),
                                    "reference_free_gap_m":reference_gap,"chosen_free_gap_m":pre_gap,
                                    "override_requested":args.pre_gap is not None,
                                    "initial_ik_seed":"reference loaded pose, clipped to current constraints"}
        provenance = {}
        for label,path in inputs.items():
            provenance[label+"_file"] = path.name
            provenance[label+"_sha256"] = hashlib.sha256(captured[label]).hexdigest()
            if path.read_bytes() != captured[label]: raise ValueError("input changed during planning: "+label)
            if label not in ("runtime","kinematics_urdf"):
                destination = args.output_dir/path.name
                if destination.exists() and destination.read_bytes() != captured[label]: raise ValueError("output input differs")
                if not destination.exists(): destination.write_bytes(captured[label])
        result = {"world_from_left_base":base.tolist(),"targets_file":inputs["targets"].name,
            "audit_file":inputs["audit"].name,"provenance":provenance,"points":[row],
            "scope":"low-elbow sampled geometric plan; no physical success claim"}
        output.write_text(json.dumps(result,indent=2)+"\n")
        print("LOW_PLAN",name,row["success"],row["scored_route_count"],row["seconds"],flush=True)
        succeeded &= row["success"]
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
