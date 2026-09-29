"""Finite upper-pair pose search on fixed scene inputs; no physics acceptance.

FCL imports occur only in the CLI/runtime path. Candidate screening and full
route construction retain the existing analytic IK and sampled mesh gates.
"""
import copy
import time

import numpy as np
from scipy.spatial.transform import Rotation


SCOPE = ("Finite alternative-posture search; sampled full-mesh geometry only. "
         "Force, actual withdrawal, measured tracking, and physics acceptance require a new run.")


def matching_roll_candidates(audit, selected_fit):
    """Return unchanged audit entries differing only by wrist local-Z rotation."""
    reference = np.asarray(selected_fit["wrist_rotation_world"], float)
    normal = np.asarray(selected_fit["normal_world"], float)
    if reference.shape != (3, 3) or normal.shape != (3,):
        raise ValueError("invalid reference fit")
    found = []
    for index, fit in enumerate(audit.get("candidates", [])):
        if (not np.allclose(fit["normal_world"], normal, atol=1e-12, rtol=0)
                or not np.isclose(fit["gap_m"], selected_fit["gap_m"], atol=1e-12, rtol=0)):
            continue
        rotation = np.asarray(fit["wrist_rotation_world"], float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-9, rtol=0)
                or not np.allclose(rotation[:, 2], -normal, atol=1e-9, rtol=0)):
            continue
        relative = reference.T @ rotation
        roll = float(np.arctan2(relative[1, 0], relative[0, 0]))
        if not np.allclose(relative, Rotation.from_rotvec([0., 0., roll]).as_matrix(), atol=1e-9, rtol=0):
            continue
        degrees = float(np.rad2deg(roll))
        if abs(abs(degrees)-180.) < 1e-8:
            degrees = 180.
        found.append({"audit_candidate_index": index, "relative_roll_deg": degrees,
                      "relative_roll_rad": float(np.deg2rad(degrees)), "fit": copy.deepcopy(fit)})
    return sorted(found, key=lambda c: (abs(c["relative_roll_deg"]) > 50. + 1e-8,
                                       abs(c["relative_roll_deg"]), c["audit_candidate_index"]))


def current_staging_tail(row, kin):
    """Use current FK positions directly: this intentionally has no OFFSET."""
    route = np.asarray(row["transfer_q"], float)
    stage = np.asarray(row["lower_staging_q"], float)
    if route.ndim != 2 or route.shape[1:] != (7,) or stage.shape != (7,):
        raise ValueError("invalid latest staging route")
    matches = np.flatnonzero(np.all(np.isclose(route, stage, atol=1e-9, rtol=0), axis=1))
    if len(matches) != 1 or not 0 < matches[0] < len(route)-1:
        raise ValueError("latest plan requires one unique interior lower-staging waypoint")
    return [np.asarray(kin.tcp(q), float).copy() for q in route[matches[0]:]]


def rolled_staging_tail(tail, roll_rad, mode="constant"):
    """Rotate each local tool axis, retaining every latest staging position."""
    if mode not in ("constant", "ramp") or len(tail) < 2 or not np.isfinite(roll_rad):
        raise ValueError("invalid staging roll mode or route")
    result = []
    for index, original in enumerate(tail):
        fraction = 1. if mode == "constant" else index/(len(tail)-1)
        # Smooth ramp has zero endpoint rate with respect to waypoint progress.
        if mode == "ramp":
            fraction = fraction*fraction*(3.-2.*fraction)
        pose = np.asarray(original, float).copy()
        pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec([0., 0., roll_rad*fraction]).as_matrix()
        result.append(pose)
    return result


def joint_metrics(route):
    """Raw joint variation, with q4 elbow and q5-q7 wrist reported separately."""
    q = np.asarray(route, float)
    if q.ndim != 2 or q.shape[1:] != (7,) or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("joint route requires at least two finite seven-joint poses")
    delta = np.diff(q, axis=0)
    travel = np.abs(delta).sum(axis=0)
    reversals = []
    for column in delta.T:
        signs = np.sign(column[np.abs(column) > 1e-5])
        reversals.append(int(np.count_nonzero(np.diff(signs))))
    return {"joint_travel_rad": travel.tolist(), "total_joint_travel_rad": float(travel.sum()),
            "elbow_travel_rad": float(travel[3]), "wrist_travel_rad": float(travel[4:].sum()),
            "joint_l2_path_rad": float(np.linalg.norm(delta, axis=1).sum()),
            "joint_excursion_rad": np.ptp(q, axis=0).tolist(),
            "max_joint_excursion_rad": float(np.ptp(q, axis=0).max()),
            "joint_direction_reversals": reversals, "direction_reversals": sum(reversals),
            "joint_peak_abs_rad": np.abs(q).max(axis=0).tolist()}


def motion_rank(metrics):
    return (float(metrics["wrist_travel_rad"])+float(metrics["elbow_travel_rad"]),
            float(metrics["total_joint_travel_rad"]))


def diverse_candidates(candidates, limit=6, minimum_distance_rad=.04):
    if limit < 1 or minimum_distance_rad <= 0:
        raise ValueError("invalid candidate retention budget")
    selected = []
    for row in sorted(candidates, key=lambda r: motion_rank(r["endpoint_metrics"])):
        q = np.asarray(row["qpre"], float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("invalid screened qpre")
        if all(np.max(np.abs(q-np.asarray(other["qpre"]))) >= minimum_distance_rad for other in selected):
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected


def fresh_plan_document(reference_document, row):
    """Copy only source bindings, never old route or physics acceptance fields."""
    if (row.get("contact_success") is not False or row.get("training_accepted", False)
            or "revalidation" in row):
        raise ValueError("new geometry must not claim inherited physics acceptance")
    result = {key: copy.deepcopy(reference_document[key]) for key in
              ("world_from_left_base", "targets_file", "audit_file", "provenance")}
    result.update(points=[copy.deepcopy(row)], scope=SCOPE)
    return result


def fit_geometry(target, fit, base, pre_gap=.025):
    """The retained candidate_geometry equations, available without importing FCL."""
    from sim.auscultation_expert_layout_v1.targets import _base_transform
    base = _base_transform(base)
    normal = np.asarray(fit["normal_world"], float)
    rotation = np.asarray(fit["wrist_rotation_world"], float)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1., atol=1e-9, rtol=0)
            or rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation[:, 2], -normal, atol=1e-9, rtol=0)
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-9, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-9, rtol=0)):
        raise ValueError("invalid audited tool frame")
    gap = float(fit["gap_m"])
    if not 0. <= gap < .015 or not .025 <= pre_gap <= .06:
        raise ValueError("invalid fixed audit or prepose gap")
    center = np.asarray(target.position_world)+np.asarray(target.normal_world)*gap
    pre = np.asarray(target.position_world)+normal*pre_gap
    free = np.asarray(target.position_world)+normal*.015
    result = {"normal_world": normal, "rotation_world": rotation,
              "rotation_base": base[:3, :3].T@rotation,
              "center_world": center, "pre_world": pre, "free_end_world": free}
    for label, point in (("center_base", center), ("pre_base", pre), ("free_end_base", free)):
        result[label] = base[:3, :3].T@(point-base[:3, 3])
    return result


def endpoint_seeds(kin, base, limits, reference, neighbors=()):
    """Finite existing, adjacent, neutral-distal, and Jacobian-nullspace seeds."""
    from sim.auscultation_expert_layout_v1.pose_ik import PoseConstraint
    pre = np.asarray(reference["approach_q"][0], float)
    seeds = [("reference_pre", pre), ("reference_loaded", np.asarray(reference["loaded_q"], float))]
    for index, neighbor in enumerate(neighbors):
        seeds += [(f"neighbor_{index}_pre", np.asarray(neighbor["approach_q"][0], float)),
                  (f"neighbor_{index}_loaded", np.asarray(neighbor["loaded_q"], float))]
    pose = kin.tcp(pre)
    constraint = PoseConstraint(kin, base, pose[:3, 3], pose[:3, :3])
    _, _, vectors = np.linalg.svd(constraint.derivative(pre), full_matrices=True)
    null = vectors[-1]
    null /= np.max(np.abs(null))
    for amount in (.2, -.2, .45, -.45):
        seeds.append((f"null_{amount:+.2f}", pre+amount*null))
    for scale in (.5, 0.):
        q = pre.copy()
        q[4:] *= scale
        seeds.append((f"distal_scale_{scale:.1f}", q))
    lower, upper = np.asarray(limits.joint_lower_rad), np.asarray(limits.joint_upper_rad)
    result = []
    for label, q in seeds:
        q = np.clip(q, lower, upper)
        if all(np.max(np.abs(q-np.asarray(old["q"]))) > 1e-6 for old in result):
            result.append({"label": label, "q": q.tolist()})
    return result


def _pose_error(kin, q, position, rotation, compression=0.):
    actual = kin.tcp(q, compression)
    return {"position_error_m": float(np.linalg.norm(actual[:3, 3]-position)),
            "rotation_error_rad": float(Rotation.from_matrix(actual[:3, :3]@rotation.T).magnitude())}


def _pose_accepted(error):
    return error["position_error_m"] <= 2e-5 and error["rotation_error_rad"] <= 2e-4


def screen_endpoints(target, audit, reference_document, kin, base, limits, checker,
                     *, neighbors=(), anchors=None, deadline, progress=None):
    """Solve each discrete roll before spending additional seeds on that roll."""
    from sim.auscultation_expert_layout_v1.replan_layout import solve
    from sim.auscultation_expert_layout_v1.pose_ik import IKBudgetExpired
    reference = reference_document["points"][0]
    rolls = matching_roll_candidates(audit, reference["selected_fit"])
    seeds = endpoint_seeds(kin, base, limits, reference, neighbors)
    if anchors is None:
        anchors = (np.zeros(7), np.asarray(neighbors[-1]["approach_q"][0]) if neighbors else np.zeros(7))
    attempts, candidates = [], []
    # Breadth first: all 36/12 audited roll entries receive the reference seed
    # before deeper redundant-branch exploration consumes the finite budget.
    for seed_index, seed in enumerate(seeds):
        for roll in rolls:
            if time.monotonic() >= deadline:
                return {"candidates": candidates, "attempts": attempts, "rolls": rolls,
                        "seed_count": len(seeds), "stopped": "endpoint_time_budget_exhausted"}
            geometry = fit_geometry(target, roll["fit"], base, reference["free_approach_start_gap_m"])
            attempt = {"audit_candidate_index": roll["audit_candidate_index"], "seed_index": seed_index,
                       "seed_label": seed["label"], "relative_roll_deg": roll["relative_roll_deg"]}
            attempts.append(attempt)
            try:
                q, good, pe, re = solve(kin, base, limits, geometry["pre_base"], geometry["rotation_base"],
                                       seed["q"], deadline, contact=True)
                attempt.update(position_error_m=pe, rotation_error_rad=re)
                if not good:
                    attempt["failure"] = "prepose_ik_or_posture"
                    continue
                check = checker.check(q, contact=True)
                if not check["clear"]:
                    attempt.update(failure="prepose_clearance", check=check)
                    continue
                metrics = joint_metrics([anchors[0], q, anchors[1]])
                entry = {**copy.deepcopy(roll), "qpre": q.tolist(), "seed_label": seed["label"],
                         "seed_index": seed_index, "endpoint_metrics": metrics, "prepose_check": check,
                         "pose_error": {"position_error_m": pe, "rotation_error_rad": re}}
                attempt["accepted"] = True
                candidates.append(entry)
            except IKBudgetExpired:
                attempt["failure"] = "ik_attempt_or_endpoint_budget"
        if progress is not None:
            progress({"phase": "endpoint_screen", "point": target.name, "seed_index": seed_index,
                      "attempts": len(attempts), "feasible": len(candidates)})
    return {"candidates": candidates, "attempts": attempts, "rolls": rolls,
            "seed_count": len(seeds), "stopped": "finite_seed_and_roll_set_exhausted"}


def build_candidate_route(target, audit, reference_document, screened, kin, base, limits, checker,
                          *, deadline, rng=None, transfer_planner=None):
    """New pre/free/contact/staging/home and reverse routes, with fresh evidence."""
    from sim.auscultation_expert_layout_v1.replan_layout import (
        solve, follow_poses, coupled_contact_edge, validate_coupled_endpoint,
        route_metrics, FREE_CLEARANCE, CONTACT_CLEARANCE, JOINT_STEP, ELBOW_BUFFER, SOLVER_JOINT_BUFFER,
    )
    from sim.auscultation_expert_layout_v1.pose_ik import IKBudgetExpired
    from sim.auscultation_expert_layout_v1.controller import validate_contact_ready_route
    reference = reference_document["points"][0]
    index = int(screened["audit_candidate_index"])
    matches = {c["audit_candidate_index"]: c for c in matching_roll_candidates(audit, reference["selected_fit"])}
    if index not in matches:
        raise ValueError("candidate changes the retained normal/gap or lacks exact audit entry")
    roll = matches[index]
    fit = audit["candidates"][index]
    geometry = fit_geometry(target, fit, base, reference["free_approach_start_gap_m"])
    qpre = np.asarray(screened["qpre"], float)
    compression = float(reference["loaded_compression_m"])
    if audit["name"] != target.name or audit["compression_m"] != compression:
        raise ValueError("fixed audit identity or compression changed")
    evidence = {"scope": SCOPE, "staging_attempts": []}
    started = time.monotonic()

    def fail(reason, detail=None):
        result = {"success": False, "failure": reason, "geometric_checks": evidence,
                  "seconds": time.monotonic()-started}
        if detail is not None:
            result["detail"] = detail
        return result

    try:
        pre_error = _pose_error(kin, qpre, geometry["pre_base"], geometry["rotation_base"])
        if not _pose_accepted(pre_error):
            return fail("prepose_identity", pre_error)
        evidence["prepose_check"] = checker.check(qpre, contact=True)
        if not evidence["prepose_check"]["clear"]:
            return fail("prepose_clearance")
        pre = np.eye(4)
        pre[:3, :3], pre[:3, 3] = geometry["rotation_base"], geometry["pre_base"]
        free = pre.copy()
        free[:3, 3] = geometry["free_end_base"]
        approach, problem = follow_poses(kin, base, limits, checker, qpre, [pre, free], deadline, contact=True)
        if problem:
            return fail("free_approach", problem)
        loaded, good, pe, re = solve(kin, base, limits, geometry["center_base"], geometry["rotation_base"],
                                   approach[-1], deadline, contact=True, compression=compression)
        if not good or np.max(np.abs(loaded-approach[-1])) > .4:
            return fail("loaded_branch_ik", {"position_error_m": pe, "rotation_error_rad": re})
        coupling = {"center_base": geometry["center_base"].tolist(),
                    "normal_base": (base[:3, :3].T@geometry["normal_world"]).tolist(),
                    "compression_m": compression,
                    "law": "c(q)=clip(dot(C-tcp(q,compression=0),n_outward),0,loaded_compression)",
                    "scope": "nominal audited-center tangent-plane closure; contact onset and dynamics unverified",
                    "noncontact_clearance_m": CONTACT_CLEARANCE, "joint_edge_sample_step_rad": JOINT_STEP,
                    "tcp_endpoint_chord_sample_step_m": .0005}
        evidence["initial_loaded_endpoint_identity"] = validate_coupled_endpoint(kin, loaded, coupling)
        corridor_checks = []
        corridor, problem = follow_poses(kin, base, limits, checker, approach[-1], [free, kin.tcp(loaded)], deadline,
                                        contact=True, designated_contact=True, coupling=coupling,
                                        contact_evidence=corridor_checks)
        evidence["contact_corridor_edge_checks"] = corridor_checks
        if problem:
            return fail("contact_corridor", problem)
        loaded = corridor[-1]
        evidence["loaded_endpoint_identity"] = validate_coupled_endpoint(kin, loaded, coupling)
        evidence["loaded_check"] = checker.check(loaded, clearance=CONTACT_CLEARANCE, allow_patient_contact=True,
                                                  compression=compression, contact=True)
        loaded_error = _pose_error(kin, loaded, geometry["center_base"], geometry["rotation_base"], compression)
        if not evidence["loaded_check"]["clear"] or not _pose_accepted(loaded_error):
            return fail("loaded_endpoint", loaded_error)
        reverse_corridor = corridor[::-1]
        evidence["nominal_withdrawal_edge_checks"] = []
        for a, b in zip(reverse_corridor[:-1], reverse_corridor[1:]):
            check = coupled_contact_edge(kin, checker, a, b, coupling)
            evidence["nominal_withdrawal_edge_checks"].append(check)
            if not check["clear"]:
                return fail("nominal_withdrawal", check)
        evidence["reverse_free_approach_checks"] = []
        for a, b in zip(approach[:0:-1], approach[-2::-1]):
            check = checker.edge(a, b, contact=True)
            evidence["reverse_free_approach_checks"].append(check)
            if not check["clear"]:
                return fail("reverse_free_approach", check)
        original_tail = current_staging_tail(reference, kin)
        transfer, chosen_tail, selected_mode, selected_search = None, None, None, None
        for mode in ("constant", "ramp"):
            if mode == "ramp" and abs(roll["relative_roll_rad"]) < 1e-9:
                continue
            tail = rolled_staging_tail(original_tail, roll["relative_roll_rad"], mode)
            tail[-1] = pre.copy()
            attempt = {"roll_mode": mode}
            evidence["staging_attempts"].append(attempt)
            descent, problem = follow_poses(kin, base, limits, checker, qpre, tail[::-1], deadline)
            if problem:
                attempt.update(failure="staging_continuation", detail=problem)
                continue
            ascent = descent[::-1]
            if transfer_planner is None:
                # plan_low_elbow imports FCL; keep CPU helpers/import checks independent.
                from sim.auscultation_expert_layout_v1.plan_low_elbow import checked_transfer
                transfer_planner = checked_transfer
            bridge, report = transfer_planner(checker, limits, np.zeros(7), ascent[0], kin, base,
                np.random.default_rng(265) if rng is None else rng, min(deadline, time.monotonic()+30.))
            attempt["home_bridge"] = report
            if bridge is None:
                attempt["failure"] = "home_bridge"
                continue
            transfer = list(bridge)+ascent[1:]
            chosen_tail, selected_mode, selected_search = tail, mode, report
            break
        if transfer is None:
            return fail("no_complete_staging_home_route")
        validate_contact_ready_route({"posture_limits": limits.to_dict(), "approach_q": np.asarray(approach).tolist(),
                                      "loaded_q": loaded.tolist()}, base, kin)
        # Store independently sampled evidence for forward AND return paths.
        # Reversed geometry does not certify actual physical release.
        for key, path in (("outbound_free_edge_checks", transfer), ("return_free_edge_checks", transfer[::-1])):
            checks = evidence[key] = []
            for a, b in zip(path[:-1], path[1:]):
                check = checker.edge(a, b)
                checks.append(check)
                if not check["clear"]:
                    return fail(key, check)
        outbound = transfer+approach[1:]+corridor[1:]
        complete = outbound+reverse_corridor[1:]+approach[-2::-1]+transfer[-2::-1]
        measured = joint_metrics(complete)
        stage_world = base@chosen_tail[0]
        row = {"name": target.name, "success": True, "contact_success": False, "full_route_planned": True,
               "position_world": np.asarray(target.position_world).tolist(),
               "normal_world": np.asarray(target.normal_world).tolist(), "posture_limits": limits.to_dict(),
               "reference_posture_limits": copy.deepcopy(reference.get("reference_posture_limits", reference["posture_limits"])),
               "solver_posture_limits": limits.to_dict(), "geometric_clearance_m": FREE_CLEARANCE,
               "loaded_noncontact_clearance_m": CONTACT_CLEARANCE, "joint_edge_sample_step_rad": JOINT_STEP,
               "planning_elbow_buffer_m": ELBOW_BUFFER, "planning_joint_buffer_rad": float(SOLVER_JOINT_BUFFER),
               "loaded_compression_m": compression, "free_approach_start_gap_m": reference["free_approach_start_gap_m"],
               "free_approach_end_gap_m": reference["free_approach_end_gap_m"],
               "selected_audit_candidate_index": index, "selected_fit": copy.deepcopy(fit),
               "command_normal_world": geometry["normal_world"].tolist(),
               "command_rotation_world": geometry["rotation_world"].tolist(),
               "expected_contact_center_world": geometry["center_world"].tolist(),
               "expected_contact_gap_m": float(fit["gap_m"]), "final_free_tcp_world": geometry["free_end_world"].tolist(),
               "transfer_q": np.asarray(transfer).tolist(), "approach_q": np.asarray(approach).tolist(),
               "loaded_q": loaded.tolist(), "contact_corridor_posture_q": np.asarray(corridor).tolist(),
               "nominal_withdrawal_q": np.asarray(reverse_corridor).tolist(),
               "return_approach_q": np.asarray(approach[::-1]).tolist(), "return_transfer_q": np.asarray(transfer[::-1]).tolist(),
               "loaded_endpoint_check": evidence["loaded_check"],
               "loaded_position_error_m": loaded_error["position_error_m"],
               "loaded_orientation_error_rad": loaded_error["rotation_error_rad"],
               "contact_assembly_model": coupling, "contact_corridor_scope": coupling["scope"],
               "lower_staging_q": np.asarray(transfer[len(transfer)-len(descent)]).tolist(),
               "lower_staging_tcp_world": stage_world[:3, 3].tolist(),
               "lower_staging_rotation_world": stage_world[:3, :3].tolist(),
               "lower_staging_vertical_offset_m": float(geometry["pre_world"][2]-stage_world[2, 3]),
               "lower_staging_min_target_plane_gap_m": reference["lower_staging_min_target_plane_gap_m"],
               "lower_staging_target_plane_gap_m": float((stage_world[:3, 3]-target.position_world)@geometry["normal_world"]),
               "approach_design": "new_exact_ik_on_current_staging_positions_with_audited_axial_roll",
               "staging_roll_mode": selected_mode, "relative_roll_deg": roll["relative_roll_deg"],
               "layout_offset_world_m": copy.deepcopy(reference.get("layout_offset_world_m", [0., .05, -.02])),
               "known_pose_policy": "fixed current targets, normal and gap; existing audited axial roll; new redundant IK",
               "route_metrics": route_metrics(outbound, kin, base, limits), "motion_metrics": measured,
               "outbound_motion_metrics": joint_metrics(outbound), "route_score": motion_rank(measured)[0],
               "transfer_search": selected_search, "geometric_checks": evidence,
               "geometry_samples": checker.samples, "scope": SCOPE,
               "joint4_zero_home_exception": True, "seconds": time.monotonic()-started}
        return {"success": True, "row": row, "metrics": measured, "seconds": time.monotonic()-started}
    except IKBudgetExpired:
        return fail("ik_attempt_or_route_time_budget")
    except ValueError as error:
        return fail("pose_or_contact_route_gate", str(error))


def search_point(target, audit, reference_document, kin, base, limits, checker, *, neighbors=(), anchors=None,
                 max_seconds=600., max_routes=6, seed=265, progress=None):
    """Screen finite poses, then spend the remaining budget on complete routes."""
    if not np.isfinite(max_seconds) or max_seconds <= 0. or not 1 <= max_routes <= 24:
        raise ValueError("invalid bounded point search budget")
    started = time.monotonic()
    deadline = started+max_seconds
    prior_deadline = getattr(checker, "deadline", None)
    checker.deadline = deadline
    try:
        screen = screen_endpoints(target, audit, reference_document, kin, base, limits, checker, neighbors=neighbors,
            anchors=anchors, deadline=min(deadline, started+max_seconds*.4), progress=progress)
        pool = diverse_candidates(screen["candidates"], limit=max(18, max_routes*4))
        complete, attempts = [], []
        rng = np.random.default_rng(seed)
        for number, endpoint in enumerate(pool):
            if time.monotonic() >= deadline or len(complete) >= max_routes:
                break
            result = build_candidate_route(target, audit, reference_document, endpoint, kin, base, limits, checker,
                                           deadline=min(deadline, time.monotonic()+180.), rng=rng)
            attempt = {"endpoint_pool_index": number, "audit_candidate_index": endpoint["audit_candidate_index"],
                       "relative_roll_deg": endpoint["relative_roll_deg"],
                       **{key: value for key, value in result.items() if key not in ("row", "geometric_checks")}}
            attempts.append(attempt)
            if result["success"]:
                identifier = f"{target.name}_a{endpoint['audit_candidate_index']:04d}_s{endpoint['seed_index']:02d}"
                complete.append({"id": identifier, "name": target.name, "audit_candidate_index": endpoint["audit_candidate_index"],
                                 "relative_roll_deg": endpoint["relative_roll_deg"], "qpre": endpoint["qpre"],
                                 "lower_staging_q": result["row"]["lower_staging_q"],
                                 "loaded_q": result["row"]["loaded_q"], "metrics": result["metrics"],
                                 "endpoint_metrics": endpoint["endpoint_metrics"], "row": result["row"]})
            if progress is not None:
                progress({"phase": "complete_route", "point": target.name, "attempt": number,
                          "success": result["success"], "complete_candidates": len(complete),
                          "failure": result.get("failure")})
        return {"name": target.name, "candidates": complete, "route_attempts": attempts,
                "endpoint_screen": screen, "retained_endpoint_pool": pool, "seconds": time.monotonic()-started,
                "budget": {"max_seconds": max_seconds, "max_routes": max_routes, "ik_attempt_seconds": 2.,
                           "ik_iterations": 120, "route_attempt_seconds": 180., "bridge_attempt_seconds": 30.},
                "search_exhaustion_scope": "No improving route found is only a bounded-search result, never impossibility.",
                "stopped": "max_routes" if len(complete) >= max_routes else
                           "time_budget" if time.monotonic() >= deadline else "retained_endpoint_pool_exhausted"}
    finally:
        checker.deadline = prior_deadline
