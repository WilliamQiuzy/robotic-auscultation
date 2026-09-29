"""Re-solve the approved +Y50/-Z20 mm layout using audited expert poses.

No Isaac import is needed. FCL is loaded only for actual planning. Old joint
paths and static IK are search hints, never collision evidence. Every free
edge is newly sampled on exported full meshes, including inter-point routes.
Success remains geometry evidence only; the controller must verify physics.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .kinematics import OpenArmKinematics, DEFAULT_URDF_PATH
from .pose_ik import IKBudgetExpired, solve_posture_ik
from .posture import PostureLimits
from .scene import ROOT
from .targets import load_targets, _base_transform

OFFSET = np.array([0., .05, -.02])
NAMES = [f"{side}{i}" for i in range(1, 6) for side in ("R", "L")]
FREE_CLEARANCE = .003
CONTACT_CLEARANCE = .0003
JOINT_STEP = .015
ELBOW_BUFFER = .005
SOLVER_JOINT_BUFFER = np.deg2rad(.2)
SCOPE = ("New-layout sampled mesh geometry; contact corridor uses nominal coupled wrist/slider geometry only. "
         "True contact onset, jammed-slider states, force and dynamics remain unverified.")


def sha(data):
    return hashlib.sha256(data).hexdigest()


class PinnedInputs:
    """Capture exact source bytes and reject changes before publishing results."""
    def __init__(self):
        self.files = {}

    def read(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        if path in self.files and self.files[path] != data:
            raise ValueError("input changed: " + str(path))
        self.files[path] = data
        return data

    def verify(self):
        for path, data in self.files.items():
            if path.read_bytes() != data:
                raise ValueError("input changed: " + str(path))

    def manifest(self):
        return {str(path): sha(data) for path, data in self.files.items()}


def tightened_bounds(limits, margin_deg, kin=None):
    """Intersect published bounds with a URDF inset, retaining zero elbow home.

    The published distal bounds already contain an approximately 3-degree
    inset; another 3 degrees must not accidentally be added on top of those.
    The canonical straight elbow home lies exactly on joint 4's lower limit.
    """
    if not np.isfinite(margin_deg) or not 3. <= margin_deg <= 10.:
        raise ValueError("URDF margin must be 3 to 10 degrees; no automatic relaxation")
    source_lower = np.asarray(limits.joint_lower_rad if kin is None else kin.lower_limits, float)
    source_upper = np.asarray(limits.joint_upper_rad if kin is None else kin.upper_limits, float)
    margin = np.deg2rad(margin_deg)
    lower = np.maximum(limits.joint_lower_rad, source_lower + margin)
    upper = np.minimum(limits.joint_upper_rad, source_upper - margin)
    if source_lower[3] == 0. and limits.joint_lower_rad[3] == 0.:
        lower[3] = 0.
    if np.any(lower >= upper):
        raise ValueError("URDF margin removes posture range")
    return lower, upper


def validate_translation(target, reference):
    if (not np.allclose(target.position_world, np.asarray(reference["position_world"]) + OFFSET,
                        atol=1e-9, rtol=0)
            or not np.allclose(target.normal_world, reference["normal_world"], atol=1e-9, rtol=0)):
        raise ValueError("target differs from exact approved translation or reference normal")


class SafetyChecker:
    """Mesh checks plus fixed posture, URDF margin and contact-ready buffers."""
    def __init__(self, collision, base, limits, margin_deg=3., deadline=None):
        self.checker, self.base, self.limits = collision, base, limits
        self.kinematics = collision.kinematics
        kin = self.kinematics if hasattr(self.kinematics, "lower_limits") else None
        lower, upper = tightened_bounds(limits, margin_deg, kin)
        self.lower, self.upper = lower, upper
        self.deadline = deadline
        self.samples = 0

    def check(self, q, *, clearance=FREE_CLEARANCE, allow_patient_contact=False,
              compression=0., contact=False, **kwargs):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise IKBudgetExpired
        q = np.asarray(q, float)
        posture = self.limits.evaluate(self.kinematics, self.base, q, contact=contact)
        reason = None
        if not posture["clear"]:
            reason = "published_posture"
        elif np.any(q < self.lower - 1e-9) or np.any(q > self.upper + 1e-9):
            reason = "urdf_planning_margin"
        elif posture["elbow_z_m"] > self.limits.max_elbow_z_m - ELBOW_BUFFER + 1e-8:
            reason = "planning_elbow_buffer"
        elif contact and posture["elbow_below_wrist_m"] < self.limits.min_contact_elbow_below_wrist_m + ELBOW_BUFFER - 1e-8:
            reason = "contact_wrist_buffer"
        if reason:
            return {**posture, "clear": False, "minimum_distance_m": 0., "reason": reason}
        result = self.checker.check(q, clearance=clearance,
            allow_patient_contact=allow_patient_contact, compression=compression, **kwargs)
        self.samples += 1
        return {**result, **{k: v for k, v in posture.items() if k != "clear"}}

    def edge(self, start, end, **kwargs):
        start, end = np.asarray(start, float), np.asarray(end, float)
        count = max(1, int(np.ceil(np.max(np.abs(end-start)) / JOINT_STEP)))
        minimum = float(kwargs.get("clearance", FREE_CLEARANCE))
        for fraction in np.linspace(0., 1., count + 1):
            result = self.check(start + fraction * (end-start), **kwargs)
            minimum = min(minimum, result["minimum_distance_m"])
            if not result["clear"]:
                return {**result, "fraction": float(fraction), "samples": count + 1}
        return {"clear": True, "minimum_distance_m": minimum, "samples": count + 1}


def solve(kin, base, solver_limits, position, rotation, seed, deadline, *, contact=False, compression=0.):
    return solve_posture_ik(kin, base, solver_limits, position, rotation, seed,
        compression=compression, contact=contact, maxiter=120,
        planning_margin_m=ELBOW_BUFFER, joint_margin_rad=SOLVER_JOINT_BUFFER,
        deadline=min(deadline, time.monotonic() + 2.))


def interpolate_poses(start, end):
    """At most 2 mm translation and 2 degrees rotation per local IK proposal."""
    distance = np.linalg.norm(end[:3, 3] - start[:3, 3])
    angle = (Rotation.from_matrix(start[:3, :3]).inv() * Rotation.from_matrix(end[:3, :3])).magnitude()
    count = max(1, int(np.ceil(distance / .002)), int(np.ceil(angle / np.deg2rad(2.))))
    fractions = np.linspace(0., 1., count + 1)[1:]
    rotations = Slerp([0., 1.], Rotation.from_matrix(np.stack([start[:3, :3], end[:3, :3]])))(fractions)
    for fraction, rotation in zip(fractions, rotations):
        pose = np.eye(4)
        pose[:3, :3] = rotation.as_matrix()
        pose[:3, 3] = start[:3, 3] + fraction * (end[:3, 3] - start[:3, 3])
        yield pose


def nominal_compression(kin, q, model):
    """Nominal assembly closure, not a contact-onset or spring/force prediction.

    n points outward. A zero-compression TCP advanced inward beyond the plane
    through the audited loaded diaphragm center produces positive compression.
    This pairs each wrist pose with one assembly displacement, rather than
    requiring all independently chosen wrist/slider combinations to be safe.
    """
    center = np.asarray(model["center_base"], float)
    normal = np.asarray(model["normal_base"], float)
    maximum = float(model["compression_m"])
    if (center.shape != (3,) or normal.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1., atol=1e-8, rtol=0)
            or not np.isfinite(maximum) or not 0. < maximum <= .005):
        raise ValueError("invalid nominal compression geometry")
    position = np.asarray(kin.tcp(q)[:3, 3], float)
    advance = float(np.dot(center-position, normal))
    if not np.isfinite(advance):
        raise ValueError("nonfinite nominal compression advance")
    return float(np.clip(advance, 0., maximum)), advance


def validate_coupled_endpoint(kin, q, model):
    compression, advance = nominal_compression(kin, q, model)
    if abs(advance-float(model["compression_m"])) > 2e-5:
        raise ValueError("nominal coupled endpoint does not match audited loaded compression")
    return {"compression_m": compression, "unclamped_normal_advance_m": advance,
            "expected_compression_m": float(model["compression_m"]), "tolerance_m": 2e-5}


def coupled_contact_edge(kin, checker, start, end, model):
    """Sample the joint pose and its nominal compression together on each edge."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    tcp_distance = float(np.linalg.norm(kin.tcp(end)[:3, 3]-kin.tcp(start)[:3, 3]))
    count = max(1, int(np.ceil(np.max(np.abs(end-start))/JOINT_STEP)), int(np.ceil(tcp_distance/.0005)))
    minimum, evidence = CONTACT_CLEARANCE, []
    for fraction in np.linspace(0., 1., count+1):
        q = start + fraction*(end-start)
        compression, advance = nominal_compression(kin, q, model)
        check = checker.check(q, clearance=CONTACT_CLEARANCE, allow_patient_contact=True,
                              compression=compression, contact=True)
        minimum = min(minimum, check["minimum_distance_m"])
        sample = {"fraction": float(fraction), "q_rad": q.tolist(), "compression_m": compression,
                  "unclamped_normal_advance_m": advance, "minimum_distance_m": check["minimum_distance_m"],
                  "clear": bool(check["clear"])}
        evidence.append(sample)
        if not check["clear"]:
            return {**check, "fraction": float(fraction), "compression_m": compression,
                    "unclamped_normal_advance_m": advance, "samples": len(evidence), "sample_evidence": evidence}
    return {"clear": True, "minimum_distance_m": minimum, "samples": len(evidence),
            "compression_start_m": evidence[0]["compression_m"], "compression_end_m": evidence[-1]["compression_m"],
            "compression_min_m": min(s["compression_m"] for s in evidence),
            "compression_max_m": max(s["compression_m"] for s in evidence), "sample_evidence": evidence}


def follow_poses(kin, base, solver_limits, checker, start_q, poses, deadline, *, contact=False,
                 designated_contact=False, coupling=None, contact_evidence=None):
    """Preserve a local branch; reject an IK jump or unsafe interior sample."""
    if designated_contact and (not contact or coupling is None):
        raise ValueError("designated contact route requires contact posture and an explicit coupling model")
    route = [np.asarray(start_q, float)]
    for start, end in zip(poses[:-1], poses[1:]):
        for pose in interpolate_poses(start, end):
            q, good, pe, re = solve(kin, base, solver_limits, pose[:3, 3], pose[:3, :3],
                                     route[-1], deadline, contact=contact)
            if not good or np.max(np.abs(q-route[-1])) > .15:
                return None, {"reason": "local_ik_or_branch_jump", "position_error_m": pe, "rotation_error_rad": re}
            if designated_contact:
                check = coupled_contact_edge(kin, checker, route[-1], q, coupling)
                if contact_evidence is not None:
                    contact_evidence.append(check)
            else:
                check = checker.edge(route[-1], q, contact=contact, clearance=FREE_CLEARANCE,
                                     allow_patient_contact=False, compression=0.)
            if not check["clear"]:
                return None, {"reason": "sampled_clearance_or_posture", "check": check}
            route.append(q)
    return route, None


def reference_tail(reference, kin, base):
    route = np.asarray(reference["transfer_q"], float)
    stage = np.asarray(reference["lower_staging_q"], float)
    indices = np.flatnonzero(np.all(np.isclose(route, stage, atol=1e-9, rtol=0), axis=1))
    if len(indices) != 1 or not 0 < indices[0] < len(route)-1:
        raise ValueError("reference has no unique lower-staging tail")
    tail = []
    for q in route[indices[0]:]:
        pose = kin.tcp(q).copy()
        pose[:3, 3] += base[:3, :3].T @ OFFSET
        tail.append(pose)
    return tail


def static_seeds(document, name):
    matches = [variant for variant in document["variants"]
               if np.allclose(variant["offset_world_m"], OFFSET, atol=1e-12, rtol=0)]
    if len(matches) != 1:
        raise ValueError("static review must have exactly one approved layout variant")
    points = [point for point in matches[0]["points"] if point["name"] == name]
    if len(points) != 1:
        raise ValueError("static review point missing or duplicated")
    return points[0]["poses"]


def route_metrics(route, kin, base, limits):
    q = np.asarray(route, float)
    samples = []
    for start, end in zip(q[:-1], q[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(end-start)) / JOINT_STEP)))
        samples.extend(start + np.linspace(0., 1., count+1)[:, None] * (end-start))
    q = np.asarray(samples)
    urdf = np.minimum(q - kin.lower_limits, kin.upper_limits-q)
    other = urdf.copy()
    # q4 lower=0 is the explicit canonical-home exception, never hide it.
    other[:, 3] = kin.upper_limits[3] - q[:, 3]
    return {"sample_count": len(q), "min_urdf_joint_margin_deg_including_home": float(np.rad2deg(urdf.min())),
        "min_urdf_margin_deg_except_joint4_lower": float(np.rad2deg(other.min())),
        "joint2_upper_margin_deg": float(np.rad2deg((kin.upper_limits[1]-q[:, 1]).min())),
        "joint4_lower_margin_deg": float(np.rad2deg((q[:, 3]-kin.lower_limits[3]).min())),
        "max_elbow_z_m": max(limits.evaluate(kin, base, x)["elbow_z_m"] for x in q),
        "joint_travel_rad": float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())}


def plan_point(target, audit, reference, seeds, kin, base, limits, solver_limits, checker,
               *, max_seconds=180., max_attempts=6, rng_seed=265, diagnostic_only=False):
    from .plan_contact_pose import candidate_geometry, _BudgetExpired
    from .plan_low_elbow import checked_transfer
    validate_translation(target, reference)
    index = reference["selected_audit_candidate_index"]
    candidate = audit["candidates"][index]
    if audit["name"] != target.name:
        raise ValueError("audit point identity differs")
    for key in ("normal_world", "wrist_rotation_world", "gap_m"):
        if not np.allclose(candidate[key], reference["selected_fit"][key], atol=1e-12, rtol=0):
            raise ValueError("known accepted tool pose changed: " + key)
    compression = float(audit["compression_m"])
    if compression != reference["loaded_compression_m"] or not 0. < compression <= .005:
        raise ValueError("audit compression differs from accepted fit")
    geometry = candidate_geometry(target, candidate, base, pre_gap=.025)
    tail = reference_tail(reference, kin, base)
    pre = np.eye(4); pre[:3, :3] = geometry["rotation_base"]; pre[:3, 3] = geometry["pre_base"]
    tail[-1] = pre  # Exact translated target geometry, rather than old IK residual.
    free = pre.copy(); free[:3, 3] = geometry["free_end_base"]
    started = time.monotonic(); deadline = started + max_seconds
    checker.deadline = deadline
    coupling = {"center_base": geometry["center_base"].tolist(),
        "normal_base": (base[:3, :3].T@geometry["normal_world"]).tolist(), "compression_m": compression,
        "law": "c(q)=clip(dot(C-tcp(q,compression=0),n_outward),0,loaded_compression)",
        "sign": "positive normal advance is inward wrist motion; positive compression retracts carriage outward",
        "scope": "nominal audited-center tangent-plane closure only; not true contact onset, force, dynamics or jammed-slider coverage",
        "noncontact_clearance_m": CONTACT_CLEARANCE, "joint_edge_sample_step_rad": JOINT_STEP,
        "tcp_endpoint_chord_sample_step_m": .0005}
    row = {"name": target.name, "success": False, "contact_success": False, "full_route_planned": False,
        "position_world": target.position_world.tolist(), "normal_world": target.normal_world.tolist(),
        "posture_limits": solver_limits.to_dict(), "reference_posture_limits": limits.to_dict(),
        "geometric_clearance_m": FREE_CLEARANCE,
        "loaded_noncontact_clearance_m": CONTACT_CLEARANCE, "joint_edge_sample_step_rad": JOINT_STEP,
        "loaded_compression_m": compression, "free_approach_start_gap_m": .025,
        "free_approach_end_gap_m": .015, "selected_audit_candidate_index": index,
        "selected_fit": candidate, "command_normal_world": geometry["normal_world"].tolist(),
        "command_rotation_world": geometry["rotation_world"].tolist(),
        "expected_contact_center_world": geometry["center_world"].tolist(),
        "expected_contact_gap_m": float(candidate["gap_m"]), "final_free_tcp_world": geometry["free_end_world"].tolist(),
        "approach_design": "newly_solved_translated_progressive_lower_staging_then_normal_approach",
        "scope": SCOPE, "attempts": [], "layout_offset_world_m": OFFSET.tolist(),
        "solver_posture_limits": solver_limits.to_dict(), "planning_joint_buffer_rad": float(SOLVER_JOINT_BUFFER),
        "planning_elbow_buffer_m": ELBOW_BUFFER, "contact_assembly_model": coupling,
        "known_pose_policy": "preserve accepted fit rotation, normal, gap and progressive staging rotations; solve joints anew"}
    seed_list = [np.asarray(seeds["precontact"]["selected"]["q_rad"]),
                 np.asarray(seeds["loaded"]["selected"]["q_rad"]), np.asarray(reference["approach_q"][0])]
    perturbations = [np.array([-.12, -.18, .08, -.10, .08, -.03, -.06]),
                     np.array([.12, -.12, -.08, .08, -.08, .03, .06]), np.array([0., -.2, .1, 0., 0., 0., -.1])]
    seed_list += [seed_list[0] + change for change in perturbations]
    rng = np.random.default_rng(rng_seed)
    try:
        home_check = checker.check(np.zeros(7))
        if not home_check["clear"]:
            row["failure_reason"] = "home_clearance_or_posture"; row["home_check"] = home_check
            return row
        for seed_index, seed in enumerate(seed_list[:max_attempts]):
            attempt = {"seed_index": seed_index}; row["attempts"].append(attempt)
            try:
                qpre, good, pe, re = solve(kin, base, solver_limits, pre[:3, 3], pre[:3, :3], seed, deadline, contact=True)
                if not good:
                    attempt.update(failure="precontact_ik", position_error_m=pe, orientation_error_rad=re); continue
                check = checker.check(qpre, contact=True)
                if not check["clear"]:
                    attempt.update(failure="precontact_geometry", check=check); continue
                approach, failure = follow_poses(kin, base, solver_limits, checker, qpre, [pre, free], deadline, contact=True)
                if failure:
                    attempt.update(failure="approach", detail=failure); continue
                loaded, good, pe, re = solve(kin, base, solver_limits, geometry["center_base"], geometry["rotation_base"],
                    approach[-1], deadline, contact=True, compression=compression)
                if not good or np.max(np.abs(loaded-approach[-1])) > .4:
                    attempt.update(failure="loaded_branch_ik", position_error_m=pe, orientation_error_rad=re); continue
                # Solve the uncompressed commanded TCP continuously through the
                # loaded-equivalent wrist pose, as used by contact feedback.
                end = kin.tcp(loaded).copy()
                endpoint_identity = validate_coupled_endpoint(kin, loaded, coupling)
                comparison = [{"compression_m": c, "check": checker.check(loaded, clearance=CONTACT_CLEARANCE,
                    allow_patient_contact=True, compression=c, contact=True)} for c in (0., compression/2., compression)]
                evidence = []
                diagnostic = {"loaded_q_rad": loaded.tolist(), "loaded_endpoint_identity": endpoint_identity,
                    "fixed_loaded_wrist_comparison": comparison,
                    "comparison_scope": "0 and half compression at loaded wrist are counterfactual diagnostics, not acceptance gates; audited loaded compression remains mandatory",
                    "nominal_coupled_corridor_edge_checks": evidence}
                row["contact_coupling_diagnostic"] = diagnostic
                corridor, failure = follow_poses(kin, base, solver_limits, checker, approach[-1], [free, end], deadline,
                    contact=True, designated_contact=True, coupling=coupling, contact_evidence=evidence)
                diagnostic["nominal_coupled_corridor_success"] = corridor is not None
                if failure:
                    attempt.update(failure="contact_corridor", detail=failure)
                    if diagnostic_only:
                        row["failure_reason"] = "nominal_coupled_contact_corridor_rejected"
                        return row
                    continue
                loaded = corridor[-1]
                diagnostic["resolved_endpoint_identity"] = validate_coupled_endpoint(kin, loaded, coupling)
                loaded_check = checker.check(loaded, clearance=CONTACT_CLEARANCE, allow_patient_contact=True,
                                             compression=compression, contact=True)
                diagnostic["resolved_loaded_endpoint_check"] = loaded_check
                if not loaded_check["clear"]:
                    attempt.update(failure="loaded_geometry", check=loaded_check)
                    if diagnostic_only:
                        row["failure_reason"] = "loaded_geometry_rejected"
                        return row
                    continue
                if diagnostic_only:
                    row.update(success=True, scope="Contact coupling diagnostic only; home/staging/full route are not planned or accepted.",
                               contact_corridor_posture_q=np.asarray(corridor).tolist())
                    return row
                descent, failure = follow_poses(kin, base, solver_limits, checker, qpre, tail[::-1], deadline)
                if failure:
                    attempt.update(failure="lower_staging", detail=failure); continue
                ascent = descent[::-1]
                bridge, search = checked_transfer(checker, solver_limits, np.zeros(7), ascent[0], kin, base,
                    rng, min(deadline, time.monotonic()+30.))
                attempt["transfer_search"] = search
                if bridge is None:
                    attempt.update(failure="home_transfer"); continue
                transfer = list(bridge) + ascent[1:]
                # Controller preflight also samples the direct free-to-loaded
                # joint interval. Keep this gate even with a finer corridor.
                from .controller import validate_contact_ready_route
                try:
                    validate_contact_ready_route({"posture_limits": solver_limits.to_dict(),
                        "approach_q": np.asarray(approach).tolist(), "loaded_q": loaded.tolist()}, base, kin)
                except ValueError as error:
                    attempt.update(failure="controller_contact_route_gate", detail=str(error)); continue
                stage_world = base @ tail[0]
                actual = kin.tcp(loaded, compression)
                pe = float(np.linalg.norm(actual[:3, 3]-geometry["center_base"]))
                re = float((Rotation.from_matrix(actual[:3, :3]).inv()*Rotation.from_matrix(geometry["rotation_base"])).magnitude())
                if pe > 2e-5 or re > 2e-4:
                    attempt.update(failure="loaded_corridor_endpoint_error", position_error_m=pe, orientation_error_rad=re); continue
                metrics = route_metrics(transfer + approach[1:] + corridor[1:], kin, base, limits)
                row.update(success=True, full_route_planned=True, transfer_q=np.asarray(transfer).tolist(), approach_q=np.asarray(approach).tolist(),
                    loaded_q=loaded.tolist(), loaded_endpoint_check=loaded_check,
                    contact_corridor_posture_q=np.asarray(corridor).tolist(),
                    contact_corridor_scope="continuous local IK; q and nominal c(q) mesh configuration sampled together at <=0.015rad; only designated patient surfaces exempt; contact onset and dynamics unverified",
                    loaded_position_error_m=pe, loaded_orientation_error_rad=re,
                    lower_staging_q=ascent[0].tolist(), lower_staging_tcp_world=stage_world[:3, 3].tolist(),
                    lower_staging_rotation_world=stage_world[:3, :3].tolist(),
                    lower_staging_vertical_offset_m=float(geometry["pre_world"][2]-stage_world[2, 3]),
                    lower_staging_min_target_plane_gap_m=.035,
                    lower_staging_target_plane_gap_m=float(np.dot(stage_world[:3, 3]-target.position_world, geometry["normal_world"])),
                    route_metrics=metrics, route_score=metrics["joint_travel_rad"], transfer_search=search,
                    scored_route_count=1, geometry_samples=checker.samples)
                break
            except IKBudgetExpired:
                attempt["failure"] = "ik_attempt_or_wall_budget"
                if time.monotonic() >= deadline:
                    break
    except (_BudgetExpired, IKBudgetExpired):
        row["search_stopped"] = "time_budget_exhausted"
    finally:
        checker.deadline = None
        row["seconds"] = time.monotonic()-started
        row["failure_counts"] = dict(Counter(a["failure"] for a in row["attempts"] if "failure" in a))
    if not row["success"]:
        row.setdefault("failure_reason", "no_complete_new_layout_route")
    return row


def write_json(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", nargs="+", default=NAMES)
    parser.add_argument("--input-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/expert_v2/plans")
    parser.add_argument("--reference-dir", type=Path, default=ROOT.parent/"auscultation_training/results/low_elbow_release_v1/plans")
    parser.add_argument("--static-review", type=Path, default=ROOT.parent/"layout_review_20260923/reachability/layout_ik_refinement.json")
    parser.add_argument("--urdf-margin-deg", type=float, default=3.)
    parser.add_argument("--max-seconds", type=float, default=180.)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--diagnose-contact-coupling", action="store_true",
                        help="compare loaded wrist at 0/half/full compression and nominal coupled corridor only; not executable plans")
    args = parser.parse_args(argv)
    if (len(set(args.points)) != len(args.points) or not args.points or any(n not in NAMES for n in args.points)
            or not np.isfinite(args.max_seconds) or not 5. <= args.max_seconds <= 900.
            or not 1 <= args.max_attempts <= 6):
        parser.error("invalid point names or bounded search limits")
    if args.output_dir.resolve() == args.reference_dir.resolve():
        raise ValueError("old release is immutable")
    pinned = PinnedInputs()
    common = {"runtime": ROOT/"runtime_inspection.json", "kinematics_urdf": DEFAULT_URDF_PATH,
        "targets": args.input_dir/"contact_targets.json", "collision_metadata": args.input_dir/"collision_geometry.json",
        "collision_arrays": args.input_dir/"collision_geometry.npz"}
    data = {label: pinned.read(path) for label, path in common.items()}
    base = _base_transform(json.loads(data["runtime"])["world_from_left_base"])
    kin = OpenArmKinematics(DEFAULT_URDF_PATH)
    limits_bytes = pinned.read(args.reference_dir/"posture_limits.json")
    limits = PostureLimits.from_dict(json.loads(limits_bytes), kin)
    lower, upper = tightened_bounds(limits, args.urdf_margin_deg, kin)
    solver_limits = replace(limits, joint_lower_rad=tuple(lower), joint_upper_rad=tuple(upper))
    static = json.loads(pinned.read(args.static_review))
    pinned.read(Path(__file__))
    for name in ("collision.py", "kinematics.py", "posture.py", "pose_ik.py", "tool_frames.py", "plan_low_elbow.py", "plan_contact_pose.py", "controller.py"):
        pinned.read(ROOT/name)
    from .collision import CollisionChecker
    collision = CollisionChecker(kin, base, directory=args.input_dir)
    checker = SafetyChecker(collision, base, limits, args.urdf_margin_deg)
    targets = {t.name: t for t in load_targets(base, common["targets"])}
    pinned.verify()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/"posture_limits.json", solver_limits.to_dict())
    write_json(args.output_dir/"reference_posture_limits.json", limits.to_dict())
    write_json(args.output_dir/"solver_posture_limits.json", solver_limits.to_dict())
    for label, path in common.items():
        if label not in ("runtime", "kinematics_urdf"):
            (args.output_dir/path.name).write_bytes(data[label])
    rows = []
    for name in args.points:
        reference_path = args.reference_dir/f"planned_paths_{name}.json"
        reference_bytes = pinned.read(reference_path)
        reference_document = json.loads(reference_bytes)
        reference = reference_document["points"][0]
        if (reference.get("name") != name or reference["posture_limits"] != limits.to_dict()
                or not np.allclose(reference_document["world_from_left_base"], base, atol=1e-10, rtol=0)):
            raise ValueError("reference point, base or fixed posture envelope differs")
        audit_name = reference_document["provenance"]["audit_file"]
        if Path(audit_name).name != audit_name:
            raise ValueError("unsafe audit filename")
        audit_path = args.input_dir/audit_name
        audit_bytes = pinned.read(audit_path)
        row = plan_point(targets[name], json.loads(audit_bytes), reference, static_seeds(static, name),
            kin, base, limits, solver_limits, checker, max_seconds=args.max_seconds,
            max_attempts=args.max_attempts, rng_seed=265+NAMES.index(name), diagnostic_only=args.diagnose_contact_coupling)
        row.update(reference_plan_sha256=sha(reference_bytes), static_review_sha256=sha(pinned.read(args.static_review)),
                   planner_script_sha256=sha(pinned.read(Path(__file__))), requested_urdf_margin_deg=args.urdf_margin_deg,
                   joint4_zero_home_exception=True)
        pinned.verify()
        (args.output_dir/audit_name).write_bytes(audit_bytes)
        provenance = {}
        for label, path in {**common, "audit": audit_path}.items():
            provenance[label+"_file"] = path.name
            provenance[label+"_sha256"] = sha(pinned.read(path))
        result = {"world_from_left_base": base.tolist(), "targets_file": common["targets"].name,
                  "audit_file": audit_name, "provenance": provenance, "points": [row], "scope": SCOPE}
        prefix = "contact_coupling" if args.diagnose_contact_coupling else "planned_paths"
        output = args.output_dir/f"{prefix}_{name}.json"
        write_json(output, result)
        if row["success"] and not args.diagnose_contact_coupling:
            from .controller import validate_plan
            validate_plan(output, args.output_dir, ROOT)
        rows.append(row)
        print("LAYOUT_PLAN", name, row["success"], row.get("failure_counts", {}), round(row["seconds"], 3), flush=True)
    bridges = []
    if all(row["success"] for row in rows) and not args.diagnose_contact_coupling:
        from .plan_low_elbow import checked_transfer
        for previous, current in zip(rows[:-1], rows[1:]):
            route, search = checked_transfer(checker, solver_limits, np.asarray(previous["approach_q"][0]),
                np.asarray(current["lower_staging_q"]), kin, base, np.random.default_rng(365+len(bridges)),
                time.monotonic()+30.)
            bridges.append({"from": previous["name"], "to": current["name"], "success": route is not None,
                            "q": None if route is None else np.asarray(route).tolist(), "search": search,
                            "scope": "nominal released prepose to next lower staging; runtime replans from measured release"})
    passed = all(row["success"] for row in rows) and all(bridge["success"] for bridge in bridges)
    pinned.verify()
    write_json(args.output_dir/"planning_report.json", {"success": passed,
        "scope": "Contact coupling diagnostics only; no full route accepted." if args.diagnose_contact_coupling else SCOPE,
        "diagnostic_only": args.diagnose_contact_coupling,
        "points": [{"name": row["name"], "success": row["success"], "seconds": row["seconds"],
                    "route_metrics": row.get("route_metrics"), "failure_counts": row.get("failure_counts")} for row in rows],
        "nominal_inter_point_bridges": bridges, "source_sha256": pinned.manifest(),
        "urdf_margin_deg": args.urdf_margin_deg, "joint4_zero_home_exception": True,
        "original_posture_envelope_never_relaxed": True, "runtime_joint_envelope_tightened": True,
        "observed_joint_tolerance_rad": solver_limits.observed_joint_tolerance_rad,
        "approved_offset_world_m": OFFSET.tolist()})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
