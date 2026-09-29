"""Fresh upper-pair full-mesh route evidence, preserving sealed input bindings.

FCL is imported only by revalidate_set. Pure geometry/receipt tests run on Mac;
the CLI is for the Linux FCL runtime. Geometry acceptance is not physics,
actual withdrawal, force, training-data, acoustic, or clinical acceptance.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from . import contract as fixed


def _array(value, shape, label):
    result = np.asarray(value, float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("invalid " + label)
    return result


def _route(value, label):
    result = np.asarray(value, float)
    if result.ndim != 2 or result.shape[1:] != (7,) or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("invalid " + label + " route")
    return result


def _same(a, b):
    return np.allclose(a, b, atol=1e-9, rtol=0)


def _pose_check(kin, base, q, position, rotation, label, compression=0.):
    actual = base @ kin.tcp(q, compression=compression)
    position_error = float(np.linalg.norm(actual[:3, 3] - position))
    angle_error = float(Rotation.from_matrix(actual[:3, :3].T @ rotation).magnitude())
    if position_error > 2e-5 or angle_error > 2e-4:
        raise ValueError(label + " pose does not match pinned target/kinematics")
    return {"position_error_m": position_error, "angle_error_rad": angle_error}


def _sample_route(label, route, kin, checker, *, contact, coupled, model, joint_step):
    from sim.auscultation_expert_layout_v1.replan_layout import nominal_compression
    clearance = .0003 if coupled else .003
    edges = []
    max_step = 0.
    for index, (start, end) in enumerate(zip(route[:-1], route[1:])):
        extent = float(np.max(np.abs(end - start)))
        count = max(1, int(np.ceil(extent / joint_step)))
        if coupled:
            chord = float(np.linalg.norm(kin.tcp(end)[:3, 3] - kin.tcp(start)[:3, 3]))
            count = max(count, int(np.ceil(chord / .0005)))
        max_step = max(max_step, extent / count)
        samples = []
        for fraction in np.linspace(0., 1., count + 1):
            q = start + fraction * (end - start)
            compression, advance = nominal_compression(kin, q, model) if coupled else (0., None)
            result = checker.check(q, clearance=clearance, allow_patient_contact=coupled,
                                   compression=compression, contact=contact)
            if result.get("clear") is not True:
                raise ValueError(f"{label} edge {index} sample {len(samples)} rejected: {result}")
            if not np.isfinite(result["minimum_distance_m"]):
                raise ValueError(label + " checker returned nonfinite clearance")
            samples.append({"fraction": float(fraction), "q_rad": q.tolist(),
                            "compression_m": compression, "normal_advance_m": advance, "check": result})
        edges.append({"edge_index": index, "samples": len(samples), "sample_evidence": samples,
                      "minimum_distance_m": min(s["check"]["minimum_distance_m"] for s in samples)})
    return {"edge_count": len(edges), "checked_samples": sum(e["samples"] for e in edges),
            "max_sample_joint_step_rad": max_step, "clearance_m": clearance,
            "contact_posture_required": contact, "nominal_coupled_geometry": coupled, "edges": edges}


def check_plan_geometry(plan, kin, checker):
    """Recompute all path evidence; production supplies the full-mesh SafetyChecker."""
    fixed.require("revalidation" not in plan, "inherited revalidation must be removed before a new check")
    fixed.require(len(plan.get("points", [])) == 1, "one point required")
    row = plan["points"][0]
    fixed.require(row.get("name") in ("R1", "L1") and row.get("success") is True
                  and row.get("full_route_planned") is True, "complete successful upper route required")
    fixed.require(row.get("contact_success") is not True and row.get("training_accepted") is not True,
                  "geometry cannot inherit physics/training acceptance")
    base = _array(plan["world_from_left_base"], (4, 4), "base")
    normal = _array(row["command_normal_world"], (3,), "normal")
    rotation = _array(row["command_rotation_world"], (3, 3), "rotation")
    fixed.require(np.isclose(np.linalg.norm(normal), 1., atol=1e-9, rtol=0)
                  and _same(rotation.T @ rotation, np.eye(3))
                  and np.isclose(np.linalg.det(rotation), 1., atol=1e-9, rtol=0)
                  and _same(rotation[:, 2], -normal), "invalid command rotation/normal")
    transfer = _route(row.get("transfer_q"), "transfer")
    approach = _route(row.get("approach_q"), "approach")
    corridor = _route(row.get("contact_corridor_posture_q"), "contact corridor")
    loaded = _array(row.get("loaded_q"), (7,), "loaded joints")
    stage = _array(row.get("lower_staging_q"), (7,), "lower staging joints")
    fixed.require(_same(transfer[0], np.zeros(7)) and _same(transfer[-1], approach[0]),
                  "disconnected home/transfer/approach route")
    matches = np.flatnonzero(np.all(np.isclose(transfer, stage, atol=1e-9, rtol=0), axis=1))
    fixed.require(len(matches) == 1 and 0 < matches[0] < len(transfer)-1,
                  "transfer requires unique interior lower-staging waypoint")
    fixed.require(_same(corridor[0], approach[-1]) and _same(corridor[-1], loaded),
                  "contact corridor is disconnected from free/loaded endpoints")
    joint_step = float(row["joint_edge_sample_step_rad"])
    fixed.require(0 < joint_step <= .015 and row["geometric_clearance_m"] == .003
                  and row["loaded_noncontact_clearance_m"] == .0003, "weakened or unsupported sampled gates")
    position = _array(row["position_world"], (3,), "target position")
    center = _array(row["expected_contact_center_world"], (3,), "loaded center")
    compression = float(row["loaded_compression_m"])
    fixed.require(np.isfinite(compression) and 0 < compression <= .005, "invalid fixed compression")
    model = copy.deepcopy(row["contact_assembly_model"])
    expected_center = base[:3, :3].T @ (center - base[:3, 3])
    expected_normal = base[:3, :3].T @ normal
    fixed.require(_same(_array(model["center_base"], (3,), "coupling center"), expected_center),
                  "coupling center differs from audited loaded center")
    fixed.require(_same(_array(model["normal_base"], (3,), "coupling normal"), expected_normal)
                  and model["compression_m"] == compression, "coupling normal/compression changed")
    pre = position + normal * row["free_approach_start_gap_m"]
    free = position + normal * row["free_approach_end_gap_m"]
    endpoints = {"prepose": _pose_check(kin, base, approach[0], pre, rotation, "prepose"),
                 "free_end": _pose_check(kin, base, approach[-1], free, rotation, "free endpoint"),
                 "loaded": _pose_check(kin, base, loaded, center, rotation, "loaded", compression)}
    stage_pose = base @ kin.tcp(stage)
    endpoints["lower_staging"] = _pose_check(kin, base, stage,
        _array(row["lower_staging_tcp_world"], (3,), "staging position"),
        _array(row["lower_staging_rotation_world"], (3, 3), "staging rotation"), "lower staging")
    gap = float((stage_pose[:3, 3] - position) @ normal)
    fixed.require(gap >= max(.035, float(row["lower_staging_min_target_plane_gap_m"])) - 2e-5,
                  "lower staging target-plane clearance weakened")
    from sim.auscultation_expert_layout_v1.replan_layout import validate_coupled_endpoint
    from sim.auscultation_expert_layout_v1.controller import validate_contact_ready_route
    endpoints["nominal_coupled_endpoint"] = validate_coupled_endpoint(kin, loaded, model)
    validate_contact_ready_route(row, base, kin)
    routes = {"transfer": (transfer, False, False), "approach": (approach, True, False),
              "contact_corridor": (corridor, True, True)}
    for label, default, contact, coupled in (
        ("nominal_withdrawal", corridor[::-1], True, True),
        ("return_approach", approach[::-1], True, False),
        ("return_transfer", transfer[::-1], False, False)):
        path = _route(row.get(label + "_q", default), label)
        fixed.require(_same(path[0], default[0]) and _same(path[-1], default[-1]),
                      label + " is disconnected")
        routes[label] = (path, contact, coupled)
    results = {label: _sample_route(label, path, kin, checker, contact=contact, coupled=coupled,
                                   model=model, joint_step=joint_step)
               for label, (path, contact, coupled) in routes.items()}
    return {"accepted": True, "independent_upper_pair_check": True,
            "schema": "upper_pair_independent_geometry_v1", "point": row["name"],
            "lower_staging_index": int(matches[0]), "endpoint_checks": endpoints, "routes": results,
            "edge_count": sum(r["edge_count"] for r in results.values()),
            "checked_samples": sum(r["checked_samples"] for r in results.values()),
            "scope": "Fresh full-mesh/posture samples of home, staging, free approach, nominal coupled contact, "
                     "and reverse routes. Actual contact, HOLD, withdrawal, inter-point transfer and dynamics "
                     "require the new Isaac run; no hardware, training-data or clinical acceptance."}


def write_verified_plan(path, original, evidence):
    """Keep prevalidation bytes verbatim and never alter the source provenance."""
    path = Path(path)
    fixed.require(path.read_bytes() == original, "plan changed during revalidation")
    plan = json.loads(original)
    fixed.require("revalidation" not in plan, "inherited revalidation cannot be overwritten")
    fixed.require(evidence.get("accepted") is True and evidence.get("independent_upper_pair_check") is True,
                  "incomplete independent geometry evidence")
    before = path.with_suffix(".before_revalidation.json")
    fixed.require(not before.exists(), "prior revalidation bytes are immutable; use a new candidate directory")
    receipt = copy.deepcopy(evidence)
    receipt["input_plan_sha256"] = hashlib.sha256(original).hexdigest()
    plan["revalidation"] = receipt
    encoded = (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode()
    with before.open("xb") as stream:
        stream.write(original)
    fixed.require(path.read_bytes() == original, "plan changed during revalidation")
    path.write_bytes(encoded)


def revalidate_set(input_dir, contract_path, runtime_dir):
    """Use explicit copied inputs, pinned runtime, and Linux's complete FCL meshes."""
    from sim.auscultation_expert_layout_v1.controller import validate_plan
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    from sim.auscultation_expert_layout_v1.replan_layout import SafetyChecker
    from sim.auscultation_expert_layout_v1.revalidate_plans import validate_geometry_binding
    input_dir, contract_path, runtime_dir = map(lambda p: Path(p).resolve(),
                                              (input_dir, contract_path, runtime_dir))
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    before_hashes = fixed.validate_materialized_set(input_dir, contract, require_revalidation=False,
                                                    contract_path=contract_path)
    bundles = {name: validate_plan(input_dir / f"planned_paths_{name}.json", input_dir, runtime_dir)
               for name in fixed.TARGETS}
    for name in ("R1", "L1"):
        fixed.require("revalidation" not in bundles[name].plan, "inherited revalidation must be removed: " + name)
        fixed.require(not bundles[name].plan_path.with_suffix(".before_revalidation.json").exists(),
                      "prior revalidation is immutable: " + name)
    binding = validate_geometry_binding(runtime_dir)
    captured = {path.resolve(): hashlib.sha256(data).hexdigest()
                for bundle in bundles.values() for path, data in bundle.inputs.values()}
    source_files = (Path(__file__), Path(fixed.__file__),
        fixed.PROJECT / "sim/curobo_expert_v1/fast_collision.py",
        *(fixed.PROJECT / "sim/auscultation_expert_layout_v1" / name for name in
          ("collision.py", "kinematics.py", "tool_frames.py", "posture.py", "replan_layout.py",
           "controller.py", "revalidate_plans.py", "targets.py", "scene.py")))
    sources = {str(p.relative_to(fixed.PROJECT)): fixed.sha(p) for p in source_files}
    # Import FCL only after every copied input and baseline fingerprint passes.
    dependency_dir = runtime_dir / "python_deps"
    if dependency_dir.is_dir() and str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))
    import fcl
    from sim.curobo_expert_v1.fast_collision import FastCollisionChecker
    evidence = {}
    for name in ("R1", "L1"):
        bundle = bundles[name]
        kin = OpenArmKinematics(bundle.inputs["kinematics_urdf"][0])
        base = np.asarray(bundle.plan["world_from_left_base"], float)
        limits = PostureLimits.from_dict(bundle.row["posture_limits"], kin)
        collision = FastCollisionChecker(kin, base, directory=input_dir)
        checker = SafetyChecker(collision, base, limits, margin_deg=3.)
        evidence[name] = check_plan_geometry(bundle.plan, kin, checker)
        evidence[name]["collision_regeneration_sha256"] = binding
        evidence[name]["collider_receipt"] = {
            "implementation": "sim.curobo_expert_v1.fast_collision.FastCollisionChecker",
            "policy": "All exported triangle meshes, allowed pairs and original FCL contact exemptions; no reduced proxy.",
            "fcl_version": str(getattr(fcl, "__version__", "unavailable")),
            "collider_count": len(collision.entries), "pair_count": len(collision.pairs),
            "dynamic_collider_count": len(collision.dynamic), "urdf_planning_margin_deg": 3.,
            "sources_sha256": sources,
            "inputs_sha256": {label: hashlib.sha256(data).hexdigest()
                              for label, (_, data) in bundle.inputs.items()},
            "contract_sha256": hashlib.sha256(contract_bytes).hexdigest()}
        print("CHECKED", name, evidence[name]["checked_samples"], "fresh samples", flush=True)
    fixed.require(contract_path.read_bytes() == contract_bytes, "contract changed during revalidation")
    for path, digest in captured.items():
        fixed.require(fixed.sha(path) == digest, "input changed during revalidation: " + str(path))
    for relative, digest in sources.items():
        fixed.require(fixed.sha(fixed.PROJECT / relative) == digest, "checker source changed: " + relative)
    fixed.require(validate_geometry_binding(runtime_dir) == binding, "collision binding changed during revalidation")
    fixed.require(fixed.validate_materialized_set(input_dir, contract, require_revalidation=False,
                                                  contract_path=contract_path) == before_hashes,
                  "materialized set changed during revalidation")
    for name in ("R1", "L1"):
        bundle = bundles[name]
        write_verified_plan(bundle.plan_path, bundle.inputs["plan"][1], evidence[name])
    result = fixed.validate_materialized_set(input_dir, contract, contract_path=contract_path)
    return {"status": "geometry_passed", "plan_sha256": result,
            "changed_points": ["R1", "L1"], "physics_accepted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(revalidate_set(args.input_dir, args.contract, args.runtime_dir), indent=2))


if __name__ == "__main__":
    main()
