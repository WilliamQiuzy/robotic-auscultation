"""Bounded full-pose paths to finite-head contact hypotheses, for simulation.

Success means a sampled geometric route from zero home to a free approach pose
and a separate loaded-pose clearance check. The last free-to-loaded interval,
contact forces, dynamics, and anatomical acceptance remain uncertified.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .kinematics import OpenArmKinematics, DEFAULT_URDF_PATH
from .scene import ROOT
from .targets import _base_transform, _vector, load_targets
from .tool_frames import MAX_COMPRESSION_M


class _BudgetExpired(Exception):
    pass


def rrt_path(checker, start, goal, rng, max_iterations=600):
    """Bidirectional RRT-Connect; every extension and shortcut is sampled."""
    if checker.edge(start, goal)['clear']:
        return [start, goal]
    trees = [([np.asarray(start).copy()], [-1]), ([np.asarray(goal).copy()], [-1])]

    def extend(tree, target):
        nodes, parents = tree
        nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes)-target, axis=1)))
        delta = target-nodes[nearest]
        distance = np.linalg.norm(delta)
        candidate = nodes[nearest]+delta*min(1., .22/max(distance, 1e-12))
        if not checker.edge(nodes[nearest], candidate)['clear']:
            return None, False
        nodes.append(candidate)
        parents.append(nearest)
        return len(nodes)-1, distance <= .22

    def branch(tree, index):
        nodes, parents = tree
        result = []
        while index >= 0:
            result.append(nodes[index])
            index = parents[index]
        return result[::-1]

    for iteration in range(max_iterations):
        a = iteration % 2
        b = 1-a
        target = (trees[b][0][0] if iteration % 5 == 0 else
                  rng.uniform(checker.kinematics.lower_limits, checker.kinematics.upper_limits))
        first, _ = extend(trees[a], target)
        if first is None:
            continue
        for _ in range(100):
            second, connected = extend(trees[b], trees[a][0][first])
            if second is None:
                break
            if not connected:
                continue
            path = branch(trees[a], first) + branch(trees[b], second)[-2::-1]
            if a == 1:
                path = path[::-1]
            for _ in range(12):
                if len(path) < 3:
                    break
                left, right = sorted(rng.choice(len(path), 2, replace=False))
                if right > left+1 and checker.edge(path[left], path[right])['clear']:
                    path = path[:left+1]+path[right:]
            return path
    return None


def candidate_geometry(target, candidate, world_from_base, pre_gap=.06):
    """Keep target coordinates fixed and use the explicitly declared gap axis.

    Historical audits default to the original target normal. New surface fits
    declare command_normal so their loaded pose and the contact curve lie on
    the same line through the unchanged target.

    The audit records wrist rotation. The diaphragm frame has the same rotation,
    so it is also the full TCP orientation required by ``solve_ik``.
    """
    base = _base_transform(world_from_base)
    normal = _vector(candidate["normal_world"], "candidate normal_world")
    if not np.isclose(np.linalg.norm(normal), 1., atol=1e-6, rtol=0):
        raise ValueError("candidate normal must be unit length")
    rotation = np.asarray(candidate["wrist_rotation_world"], dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
        raise ValueError("candidate wrist rotation must be a proper 3 x 3 rotation")
    if not np.allclose(rotation[:, 2], -normal, atol=1e-6, rtol=0):
        raise ValueError("candidate wrist +Z must oppose its outward normal")
    gap = float(candidate["gap_m"])
    if not np.isfinite(gap) or not 0 <= gap < .015:
        raise ValueError("candidate gap must be finite, nonnegative, and below the free approach gap")
    gap_axis = candidate.get('gap_axis', 'surface_normal')
    if gap_axis not in ('surface_normal', 'command_normal'):
        raise ValueError('unknown candidate gap axis')
    center = target.position_world + (normal if gap_axis == 'command_normal' else target.normal_world) * gap
    if not np.isfinite(pre_gap) or not .025 <= pre_gap <= .06:
        raise ValueError('pre_gap must be between 25 and 60 mm')
    pre = target.position_world + normal * pre_gap
    free_end = target.position_world + normal * .015
    return {
        "normal_world": normal, "rotation_world": rotation,
        "rotation_base": base[:3, :3].T @ rotation,
        "center_world": center, "pre_world": pre, "free_end_world": free_end,
        "center_base": base[:3, :3].T @ (center - base[:3, 3]),
        "pre_base": base[:3, :3].T @ (pre - base[:3, 3]),
        "free_end_base": base[:3, :3].T @ (free_end - base[:3, 3]),
    }


class _TimedChecker:
    """Check the clock at each sample, including RRT shortcuts and long edges.

    The wall budget is cooperative: a single FCL distance call cannot be
    interrupted, but no further sample starts once the deadline has expired.
    """

    def __init__(self, checker, deadline):
        self.checker = checker
        self.kinematics = checker.kinematics
        self.deadline = deadline
        self.samples = 0

    def edge(self, start, end):
        start, end = np.asarray(start), np.asarray(end)
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / .015)))
        minimum = .003
        for fraction in np.linspace(0, 1, count + 1):
            if time.monotonic() >= self.deadline:
                raise _BudgetExpired
            result = self.checker.check(start + fraction * (end - start), clearance=.003)
            self.samples += 1
            if time.monotonic() >= self.deadline:
                raise _BudgetExpired
            minimum = min(minimum, result["minimum_distance_m"])
            if not result["clear"]:
                return {**result, "fraction": float(fraction), "samples": count + 1}
        return {"clear": True, "minimum_distance_m": minimum, "samples": count + 1}


def bounded_rrt_path(checker, start, goal, rng, *, deadline, max_iterations=600):
    """Run the existing checked RRT with iteration and per-sample time limits."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    timed = _TimedChecker(checker, deadline)
    started = time.monotonic()
    try:
        path = rrt_path(timed, start, goal, rng, max_iterations=max_iterations)
        if time.monotonic() >= deadline:
            raise _BudgetExpired
        reason = "route_found" if path is not None else "iteration_budget_exhausted"
    except _BudgetExpired:
        path, reason = None, "time_budget_exhausted"
    return path, {"reason": reason, "seconds": time.monotonic() - started,
                  "max_iterations": max_iterations, "edge_samples": timed.samples}


def plan_target(target, audit, base, kin, checker, *, max_attempts=30,
                max_seconds=180., rrt_seconds=12., rrt_iterations=600, rng=None, pre_gap=.06):
    """Try at most 30 finite-head hypotheses and report explicit failure reasons."""
    if not 1 <= max_attempts <= 30:
        raise ValueError("max_attempts must be between 1 and 30")
    if (not np.isfinite([max_seconds, rrt_seconds]).all()
            or min(max_seconds, rrt_seconds) <= 0 or rrt_iterations < 1):
        raise ValueError("time and iteration budgets must be finite and positive")
    if audit.get("name") != target.name:
        raise ValueError("audit point name does not match target")
    compression = float(audit["compression_m"])
    if not np.isfinite(compression) or not 0 < compression <= MAX_COMPRESSION_M:
        raise ValueError("audit requires a positive physical compression")
    rng = np.random.default_rng(14) if rng is None else rng
    started = time.monotonic()
    deadline = started + max_seconds
    home = np.zeros(7)
    row = {
        "name": target.name, "success": False, "contact_success": False,
        "position_world": target.position_world.tolist(),
        "normal_world": target.normal_world.tolist(),
        "attempts": [], "loaded_compression_m": compression,
        "free_approach_start_gap_m": pre_gap, "free_approach_end_gap_m": .015,
        "geometric_clearance_m": .003, "loaded_noncontact_clearance_m": .0003,
        "joint_edge_sample_step_rad": .015,
        "limits": {"candidate_attempts": max_attempts, "seconds": max_seconds,
                   "rrt_seconds_per_candidate": rrt_seconds, "rrt_iterations": rrt_iterations,
                   "ik_max_nfev_per_start": 120, "ik_max_starts": 2},
        "scope": "sampled geometric free route and separate loaded endpoint; no validated contact or dynamics",
    }
    home_check = checker.check(home, clearance=.003)
    if not home_check["clear"]:
        row.update(failure_reason="home_collision", home_check=home_check, seconds=time.monotonic() - started)
        return row
    indexed = sorted(enumerate(audit["candidates"]), key=lambda item: (
        item[1]["angle_change_deg"], -min(item[1]["shell_clearance_m"], item[1]["fixed_clearance_m"]), item[0]))
    for original_index, candidate in indexed[:max_attempts]:
        if time.monotonic() >= deadline:
            row["failure_reason"] = "point_time_budget_exhausted"
            break
        attempt = {"audit_candidate_index": original_index, "angle_change_deg": candidate["angle_change_deg"]}
        row["attempts"].append(attempt)
        geometry = candidate_geometry(target, candidate, base, pre_gap=pre_gap)

        def solve(position, seed, loaded=False, starts=1):
            if time.monotonic() >= deadline:
                raise _BudgetExpired
            return kin.solve_ik(position, orientation_base=geometry["rotation_base"], seed=seed,
                                compression=compression if loaded else 0., max_starts=starts, max_nfev=120,
                                position_tolerance=2e-5, orientation_tolerance=2e-4)

        try:
            seed = home if len(row["attempts"]) == 1 else rng.uniform(kin.lower_limits, kin.upper_limits)
            loaded = solve(geometry["center_base"], seed, loaded=True, starts=2)
            attempt["loaded_ik"] = {"position_error_m": loaded.position_error_m,
                                     "orientation_error_rad": loaded.orientation_error_rad, "nfev": loaded.nfev}
            if not loaded.success:
                attempt["failure"] = "loaded_full_pose_ik"
                continue
            loaded_check = checker.check(loaded.q, clearance=.0003, allow_patient_contact=True, compression=compression)
            if not loaded_check["clear"]:
                attempt.update(failure="loaded_collision", collision=loaded_check)
                continue
            pre = solve(geometry["pre_base"], loaded.q)
            if not pre.success:
                attempt["failure"] = "prepose_full_pose_ik"
                continue
            pre_check = checker.check(pre.q, clearance=.003)
            if not pre_check["clear"]:
                attempt.update(failure="prepose_collision", collision=pre_check)
                continue
            q = pre.q
            approach = [q.copy()]
            timed = _TimedChecker(checker, deadline)
            for fraction in np.linspace(0, 1, 24)[1:]:
                position = geometry["pre_base"] + fraction * (geometry["free_end_base"] - geometry["pre_base"])
                solution = solve(position, q)
                if not solution.success or np.max(np.abs(solution.q - q)) > .2:
                    attempt["failure"] = "approach_ik_or_discontinuity"
                    break
                check = timed.edge(q, solution.q)
                if not check["clear"]:
                    attempt.update(failure="approach_collision", collision=check)
                    break
                q = solution.q
                approach.append(q.copy())
            if "failure" in attempt:
                continue
            # Check that the approach branch also reaches the loaded full pose.
            # This is still only an endpoint test; dynamics owns the intervening motion.
            loaded = solve(geometry["center_base"], q, loaded=True)
            if not loaded.success or np.max(np.abs(loaded.q - q)) > .4:
                attempt["failure"] = "loaded_branch_ik_or_discontinuity"
                continue
            loaded_check = checker.check(loaded.q, clearance=.0003, allow_patient_contact=True, compression=compression)
            if not loaded_check["clear"]:
                attempt.update(failure="loaded_branch_collision", collision=loaded_check)
                continue
            path, transfer = bounded_rrt_path(checker, home, pre.q, rng,
                                              deadline=min(deadline, time.monotonic() + rrt_seconds),
                                              max_iterations=rrt_iterations)
            attempt["transfer"] = transfer
            if path is None:
                attempt["failure"] = "transfer_" + transfer["reason"]
                continue
            if time.monotonic() >= deadline:
                raise _BudgetExpired
            row.update(
                success=True, transfer_q=np.asarray(path).tolist(), approach_q=np.asarray(approach).tolist(),
                loaded_q=loaded.q.tolist(), loaded_endpoint_check=loaded_check,
                command_rotation_world=geometry["rotation_world"].tolist(),
                command_normal_world=geometry["normal_world"].tolist(),
                expected_contact_center_world=geometry["center_world"].tolist(),
                expected_contact_gap_m=float(candidate["gap_m"]),
                selected_audit_candidate_index=original_index, selected_fit=candidate,
                final_free_tcp_world=geometry["free_end_world"].tolist(),
                loaded_position_error_m=loaded.position_error_m, loaded_orientation_error_rad=loaded.orientation_error_rad,
            )
            break
        except _BudgetExpired:
            attempt["failure"] = "point_time_budget_exhausted"
            row["failure_reason"] = "point_time_budget_exhausted"
            break
        finally:
            print("CANDIDATE", target.name, len(row["attempts"]), original_index,
                  attempt.get("failure", "route_found" if row["success"] else "evaluated"), flush=True)
    row["seconds"] = time.monotonic() - started
    row["candidates"] = len(row["attempts"])
    row["failure_counts"] = dict(Counter(a["failure"] for a in row["attempts"] if "failure" in a))
    if not row["success"]:
        row.setdefault("failure_reason", "candidate_attempt_limit" if len(indexed) >= max_attempts else "all_audit_candidates_failed")
    return row


def main():
    from .collision import CollisionChecker
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", default="R1")
    parser.add_argument("--targets", type=Path, default=ROOT / "results/contact_targets.json")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=180.)
    parser.add_argument("--rrt-seconds", type=float, default=12.)
    parser.add_argument("--rrt-iterations", type=int, default=600)
    parser.add_argument("--pre-gap", type=float, default=.06)
    args = parser.parse_args()
    audit_path = args.audit or ROOT / "results" / ("tool_fit_" + args.point + "_2.0mm.json")
    output_path = args.output or ROOT / "results" / ("planned_paths_" + args.point + ".json")
    base_path = ROOT / "runtime_inspection.json"
    inputs = {"audit": audit_path, "targets": args.targets, "runtime": base_path,
              "kinematics_urdf": DEFAULT_URDF_PATH,
              "collision_metadata": ROOT / "results/collision_geometry.json",
              "collision_arrays": ROOT / "results/collision_geometry.npz"}
    hashes_before = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in inputs.items()}
    base = _base_transform(json.loads(base_path.read_text())["world_from_left_base"])
    targets = load_targets(base, path=args.targets)
    target = next((target for target in targets if target.name == args.point), None)
    if target is None:
        parser.error("requested point is absent from target file")
    audit = json.loads(audit_path.read_text())
    kin = OpenArmKinematics()
    checker = CollisionChecker(kin, base)
    row = plan_target(target, audit, base, kin, checker, max_attempts=args.max_attempts,
                      max_seconds=args.max_seconds, rrt_seconds=args.rrt_seconds, rrt_iterations=args.rrt_iterations,
                      pre_gap=args.pre_gap)
    provenance = {}
    changed = []
    for label, path in inputs.items():
        provenance[label + "_file"] = path.name
        provenance[label + "_sha256"] = hashes_before[label]
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes_before[label]:
            changed.append(label)
    if changed:
        row.update(success=False, failure_reason="inputs_changed_during_planning", changed_inputs=changed)
    result = {"world_from_left_base": base.tolist(), "targets_file": args.targets.name,
              "audit_file": audit_path.name, "provenance": provenance,
              "scope": "simulation-only sampled geometry planning; no validated contact or training",
              "points": [row]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print("PLAN", target.name, row["success"], "attempts", len(row["attempts"]),
          "seconds", round(row["seconds"], 3), "output", output_path, flush=True)
    return 0 if row["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
