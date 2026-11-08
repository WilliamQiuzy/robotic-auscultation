"""Small CPU-only planning/timing alternatives for the frozen expert.

Neither a geometric report nor the command generator is physics validation.
The checker is injected; importing this module never imports FCL or Isaac Sim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT.parent / "auscultation_expert_layout_v1"
POLICIES = ("staged_continuous", "checked_shortcut")


def joint_path(path):
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError("path must contain at least two finite seven-joint poses")
    return path


def path_metrics(path):
    path = joint_path(path)
    delta = np.diff(path, axis=0)
    per_joint = np.abs(delta).sum(axis=0)
    return {"waypoint_count": len(path), "joint_travel_rad": per_joint.tolist(),
            "total_joint_travel_rad": float(per_joint.sum()),
            "joint_l2_path_length_rad": float(np.linalg.norm(delta, axis=1).sum()),
            "joint_linf_path_length_rad": float(np.max(np.abs(delta), axis=1).sum())}


def checked_edge(start, end, checker, deadline):
    start, end = joint_path([start, end])
    count = max(1, int(np.ceil(np.max(np.abs(end-start))/.015)))
    minimum = float("inf")
    for i, fraction in enumerate(np.linspace(0., 1., count+1)):
        if time.monotonic() >= deadline:
            raise TimeoutError("candidate planning deadline exceeded")
        result = checker.check(start+fraction*(end-start), clearance=.003,
                               allow_patient_contact=False, compression=0.)
        if time.monotonic() >= deadline:
            raise TimeoutError("candidate planning deadline exceeded")
        distance = float(result.get("minimum_distance_m", float("nan")))
        if result.get("clear") is not True or not np.isfinite(distance) or distance < .003-1e-9:
            return {**result, "clear": False, "fraction": float(fraction), "samples": i+1}
        minimum = min(minimum, distance)
    return {"clear": True, "minimum_distance_m": minimum, "samples": count+1}


def checked_shortcut(path, checker, deadline):
    """Accept a direct free-space edge only after checking the entire baseline.

    Keeping the checked baseline as a fallback makes infeasibility explicit;
    there is no unbounded IK or RRT search hidden inside this comparison.
    """
    path = joint_path(path)
    checks = [checked_edge(a, b, checker, deadline) for a, b in zip(path, path[1:])]
    if any(check["clear"] is not True for check in checks):
        raise ValueError("baseline route lacks sampled clearance or posture validity")
    direct = checked_edge(path[0], path[-1], checker, deadline)
    selected = path[[0, -1]].copy() if direct["clear"] else path.copy()
    return selected, {"shortcut_accepted": direct["clear"], "direct_check": direct,
                      "baseline_edge_checks": checks, "baseline_metrics": path_metrics(path),
                      "selected_metrics": path_metrics(selected),
                      "scope": "3 mm mesh clearance and posture sampled at <=0.015 rad; physical execution required"}


def discrete_dynamic_peaks(path, commands, dt=1/240):
    """Forward differences with three stationary samples on both boundaries.

    These are sample-level command bounds, not continuous C2 smoothness or
    measured joint dynamics. Bounding the forward differences also bounds the
    centered-gradient estimates used by the independent numerical audit.
    """
    padded = np.vstack([np.repeat(path[:1], 3, axis=0), commands,
                        np.repeat(path[-1:], 3, axis=0)])
    return {name: float(np.max(np.abs(np.diff(padded, n=order, axis=0)))/dt**order)
            for name, order in (("velocity_rad_s", 1), ("acceleration_rad_s2", 2), ("jerk_rad_s3", 3))}


def path_commands(path, *, speed_limit=.25, minimum_seconds=1., dt=1/240, block_size=12,
                  acceleration_limit=2., jerk_limit=20., maximum_seconds=120.,
                  maximum_iterations=24, maximum_compute_seconds=5., return_report=False):
    """Retimed original polyline with verified finite-difference command bounds.

    A quintic arclength time law alone does not remove tangent discontinuities
    at IK knots. Generate actual physics-rate commands, check velocity,
    acceleration and jerk including stationary endpoints, then lengthen the
    complete segment and recheck. Reject on either bounded budget exhaustion.
    """
    path = joint_path(path)
    if (not np.isfinite([speed_limit, minimum_seconds, dt, acceleration_limit, jerk_limit, maximum_seconds, maximum_compute_seconds]).all()
            or speed_limit <= 0 or minimum_seconds <= 0 or dt <= 0
            or acceleration_limit <= 0 or jerk_limit <= 0 or maximum_seconds <= 0 or maximum_compute_seconds <= 0
            or not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1
            or not isinstance(maximum_iterations, int) or isinstance(maximum_iterations, bool) or maximum_iterations < 1):
        raise ValueError("invalid continuous path timing")
    original = path.copy()
    started = time.monotonic()
    lengths = np.max(np.abs(np.diff(path, axis=0)), axis=1)
    keep = np.r_[True, lengths > 1e-14]
    path = path[keep]
    lengths = np.max(np.abs(np.diff(path, axis=0)), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    seconds = max(minimum_seconds, 1.875*cumulative[-1]/speed_limit)
    initial_seconds = seconds
    last_peaks = None
    for attempt in range(1, maximum_iterations+1):
        if time.monotonic()-started >= maximum_compute_seconds:
            break
        steps = int(np.ceil(seconds/(dt*block_size)))*block_size
        duration_s = steps*dt
        if duration_s > maximum_seconds+1e-9:
            break
        u = np.arange(1, steps+1)/steps
        distance = (10*u**3-15*u**4+6*u**5)*cumulative[-1]
        commands = np.column_stack([np.interp(distance, cumulative, path[:, j]) for j in range(7)])
        commands[-1] = path[-1]
        peaks = discrete_dynamic_peaks(original, commands, dt)
        last_peaks = peaks
        if time.monotonic()-started >= maximum_compute_seconds:
            break
        if (peaks["velocity_rad_s"] <= speed_limit+1e-9
                and peaks["acceleration_rad_s2"] <= acceleration_limit+1e-9
                and peaks["jerk_rad_s3"] <= jerk_limit+1e-9):
            report = {"duration_s": duration_s, "attempts": attempt, "peaks": peaks,
                      "velocity_limit_rad_s": speed_limit, "acceleration_limit_rad_s2": acceleration_limit,
                      "jerk_limit_rad_s3": jerk_limit, "maximum_seconds": maximum_seconds,
                      "maximum_iterations": maximum_iterations, "initial_speed_only_seconds": initial_seconds,
                      "maximum_compute_seconds": maximum_compute_seconds, "compute_seconds": time.monotonic()-started,
                      "endpoint_stationary_padding_samples": 3, "physics_dt_s": dt,
                      "differentiation": "forward finite differences through order 3; includes start/stop stationary padding",
                      "geometry": "original piecewise-linear joint path; no C2 claim",
                      "scope": "free-motion commands only; contact commands and measured dynamics independently audited"}
            return (commands, report) if return_report else commands
        # Cubic scaling is only a proposal: knot crossings can dominate and
        # peaks need not decrease monotonically. Every attempt is remeasured.
        scale = max(1.2, peaks["velocity_rad_s"]/speed_limit,
                    np.sqrt(peaks["acceleration_rad_s2"]/acceleration_limit),
                    np.cbrt(peaks["jerk_rad_s3"]/jerk_limit))
        seconds = duration_s*scale*1.02
    raise RuntimeError("dynamic timing budget exhausted: "+json.dumps({
        "maximum_seconds": maximum_seconds, "maximum_iterations": maximum_iterations,
        "maximum_compute_seconds": maximum_compute_seconds,
        "next_requested_seconds": seconds, "last_peaks": last_peaks}, allow_nan=False))


def duration(path, continuous=False):
    path = joint_path(path)
    if continuous:
        return len(path_commands(path))/240
    return float(sum(np.ceil(max(1., 1.875*np.max(np.abs(b-a))/.25)/.05)*.05
                     for a, b in zip(path, path[1:])))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=FROZEN/"results/expert_v2/plans")
    parser.add_argument("--baseline", type=Path, default=FROZEN/"results/expert_v2/physics_all10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds-per-route", type=float, default=30.)
    parser.add_argument("--timing-only", action="store_true", help="no collision claim; no FCL dependency")
    args = parser.parse_args(argv)
    if not np.isfinite(args.seconds_per_route) or args.seconds_per_route <= 0:
        parser.error("seconds-per-route must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    outcome_path = args.baseline/"outcome.json"
    outcome = json.loads(outcome_path.read_text())
    if not outcome.get("success") or not outcome.get("training_accepted"):
        raise ValueError("baseline must be an accepted continuous sequence")
    checker = None
    if not args.timing_only:
        from sim.auscultation_expert_layout_v1.collision import CollisionChecker
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        from sim.auscultation_expert_layout_v1.posture import PostureChecker, PostureLimits
        from sim.auscultation_expert_layout_v1.controller import validate_plan
        from sim.auscultation_expert_layout_v1.revalidate_plans import validate_geometry_binding
        binding = validate_geometry_binding(FROZEN)
        bundles = [validate_plan(args.input_dir/("planned_paths_"+name+".json"), args.input_dir, FROZEN)
                   for name in outcome["requested_points"]]
        if any(b.plan["revalidation"]["collision_regeneration_sha256"] != binding for b in bundles):
            raise ValueError("geometry binding mismatch")
        kin = OpenArmKinematics()
        base = np.asarray(bundles[0].plan["world_from_left_base"])
        limits = PostureLimits.from_dict(bundles[0].row["posture_limits"], kin)
        checker = PostureChecker(CollisionChecker(kin, base, directory=args.input_dir), base, limits)
    records = []
    for route in outcome["inter_point_routes"]:
        path = joint_path(route["joint_waypoints_q"])
        index = route["lower_staging_waypoint_index"]
        _, tail_timing = path_commands(path[index:], return_report=True)
        record = {"from_target": route["from_target"], "to_target": route["to_target"],
                  "baseline_metrics": path_metrics(path), "baseline_transfer_seconds": duration(path),
                  "staged_continuous_transfer_seconds": duration(path[:index+1])+tail_timing["duration_s"],
                  "staged_tail_timing": tail_timing,
                  "staged_continuous_geometry_unchanged": True,
                  "direct_endpoint_lower_bound_rad": float(np.abs(path[-1]-path[0]).sum())}
        if checker is not None:
            started = time.monotonic()
            try:
                chosen, report = checked_shortcut(path, checker, started+args.seconds_per_route)
                record.update(report, joint_waypoints_q=chosen.tolist(),
                              checked_shortcut_transfer_seconds=(duration(chosen) if report["shortcut_accepted"]
                              else record["staged_continuous_transfer_seconds"]))
            except (ValueError, TimeoutError) as error:
                record.update(feasible=False, failure_reason=str(error))
            record["planning_seconds"] = time.monotonic()-started
        records.append(record)
        print("CANDIDATE", record["from_target"], record["to_target"],
              record.get("shortcut_accepted", "timing_only"), flush=True)
    report = {"scope": "offline geometry/timing only; saved old observed endpoints are diagnostic, runtime replans from fresh measured release",
              "physics_validated": False, "policy_names": list(POLICIES), "records": records,
              "baseline_outcome_sha256": sha(outcome_path), "candidate_planner_sha256": sha(__file__),
              "collision_checks_performed": checker is not None,
              "free_motion_dynamic_limits": {"velocity_rad_s": .25, "acceleration_rad_s2": 2.,
                  "jerk_rad_s3": 20., "differentiation": "padded forward differences at 240 Hz",
                  "scope": "commanded free motion only; measured dynamics and unchanged contact commands not certified"},
              "baseline_transfer_seconds": sum(r["baseline_transfer_seconds"] for r in records),
              "staged_continuous_transfer_seconds": sum(r["staged_continuous_transfer_seconds"] for r in records),
              "baseline_transfer_joint_travel_rad": sum(r["baseline_metrics"]["total_joint_travel_rad"] for r in records),
              "direct_endpoint_lower_bound_rad": sum(r["direct_endpoint_lower_bound_rad"] for r in records)}
    if checker is not None:
        report["geometry_feasible"] = all("selected_metrics" in r for r in records)
        if report["geometry_feasible"]:
            report["checked_shortcut_transfer_joint_travel_rad"] = sum(r["selected_metrics"]["total_joint_travel_rad"] for r in records)
            report["checked_shortcut_transfer_seconds"] = sum(r["checked_shortcut_transfer_seconds"] for r in records)
            report["direct_shortcuts_accepted"] = sum(r["shortcut_accepted"] for r in records)
    (args.output/"planning_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)
    return 0 if checker is None or report["geometry_feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
