"""Read-only USD dynamics contracts. Importing this module never starts Isaac.

Two deliberately different compatibility claims are available:
``effective_physics`` requires explicit resolved mass/inertia/material values;
``simulator_recipe`` permits identical auto-computation inputs, but requires exact
scene/recipe/runtime bindings. Neither scope certifies physical calibration or
replaces trajectory, collision, Hall, or hold qualification.

Register the installed USD/PhysX schema plugins *before* opening the stage. USD
schema fallback values then come from Get(), never from handwritten defaults.
Historical inspection versions must not be passed as historical run versions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SCHEMA = "wam_dynamics_contract_v1"
SCOPES = ("effective_physics", "simulator_recipe")
VERSION_KEYS = ("isaac_sim", "isaac_sim_build", "usd", "physx_schema")


class DynamicsContractError(ValueError):
    def __init__(self, message, partial_contract=None):
        super().__init__(message)
        self.partial_contract = partial_contract


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supervisor_contract():
    """Declared gentle timing matches acquired experts; dwell/safety gates stay fixed."""
    return dict(target_force_N=4.,force_tolerance_N=.5,hard_force_limit_N=5.5,
        min_contact_force_N=.2,min_contact_compression_m=.0001,max_compression_m=.006,
        precontact_gap_m=.04,precontact_tolerance_m=.002,approach_tangential_tolerance_m=.003,
        contact_normal_tolerance_m=.006,hold_tangential_tolerance_m=.003,angle_tolerance_deg=5.,
        approach_speed_limit_m_s=.02,hold_speed_limit_m_s=.005,settle_duration_s=.3,
        hold_duration_s=3.,retract_clearance_m=.04,release_force_limit_N=.2,
        release_confirm_duration_s=.15,max_observation_age_s=.2,max_update_gap_s=.2,
        approach_timeout_s=120/.7,contact_timeout_s=20/.7,hold_timeout_s=20.,
        retract_timeout_s=20/.7,episode_timeout_s=180/.7)


def execution_contract(*, ruckig_version="0.12.2"):
    if ruckig_version != "0.12.2":
        raise DynamicsContractError("Ruckig must be exactly 0.12.2")
    return {
        "controller": "continuous_ruckig_position", "ruckig_version": ruckig_version,
        "dof": 7, "physics_period_s": 1 / 240, "observation_period_s": .05,
        "maximum_open_loop_s": .05, "velocity_rad_s": [.175] * 7,
        "acceleration_rad_s2": [.5] * 7, "jerk_rad_s3": [8.] * 7,
        "target_velocity_rad_s": [0.] * 7, "target_acceleration_rad_s2": [0.] * 7,
        "state_update": "output.pass_to_input; retain velocity and acceleration across queries",
        "initial_position": "last q_command; initialized once before learned segment",
        "initial_velocity_rad_s": [0.] * 7, "initial_acceleration_rad_s2": [0.] * 7,
        "action_semantics": "absolute_joint_command_v1: q_command(t+50ms), radians, seven ordered joints",
        "target_composition": "first_raw_action in fixed URDF joint-zero convention",
        "raw_joint_lower_rad": [-3.490659,-3.3161253267948965,-1.570796,0.,-1.570796,-.785398,-1.570796],
        "raw_joint_upper_rad": [1.3962629999999998,.17453267320510335,1.570796,2.443461,1.570796,.785398,1.570796],
        "clipping": False, "ik": False, "expert_fallback": False,
        "parent_timing_scale": .7, "supervisor_base_config": supervisor_contract(),
        "observed_supervisor_version": "observed_adc_fk_rgbd_supervisor_v1",
        "observed_supervisor_sha256": sha256_file(Path(__file__).with_name('observed_supervisor.py')),
        "hall_interpretation_sha256": sha256_file(Path(__file__).with_name('supervision.py')),
        "supervisor_observation": "decoded Hall ADC, measured FK and RGB-D transported initial task frame",
        "point_supervisor_parameters": "precontact_gap=min(.04, checked_prepose_gap); retract_clearance=min(.04, checked_prepose_gap-.005)",
    }


def _usd_value(value):
    """Retain USD infinity sentinels as explicit valid-JSON values, never NaN."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise DynamicsContractError("NaN USD setting")
        if math.isinf(value):
            return {"usd_infinity": "+" if value > 0 else "-"}
        return value
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        return [_usd_value(float(value.GetReal())),
                *[_usd_value(float(x)) for x in value.GetImaginary()]]
    try:
        return [_usd_value(x) for x in value]
    except TypeError as exc:
        raise DynamicsContractError(f"Unsupported USD value {type(value).__name__}") from exc


def snapshot_stage(stage):
    """Extract resolved static schema attributes and effective material bindings.

    Dynamic q/dq and drive targets for controlled joints are removed by
    build_contract; passive spring drive targets remain physical settings.
    Geometry/placement is bound separately by the recipe's exact USD hashes.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    prims, origins, issues = {}, {}, []
    dynamic = {"physics:velocity", "physics:angularVelocity", "physics:startsAsleep"}
    for prim in stage.Traverse():
        attrs, attr_origins = {}, {}
        for attr in prim.GetAttributes():
            name = str(attr.GetName())
            if not name.startswith(("physics:", "physx", "drive:")) or name in dynamic:
                continue
            if name.startswith("physxContactReport:"):
                continue
            value = attr.Get()
            if value is None:
                continue
            if attr.GetNumTimeSamples():
                issues.append(f"time-varying static setting: {prim.GetPath()}.{name}")
            attrs[name] = _usd_value(value)
            attr_origins[name] = "authored" if attr.HasAuthoredValueOpinion() else "schema_fallback"
        apis = sorted(str(x) for x in prim.GetAppliedSchemas()
                      if x.startswith(("Physics", "Physx")))
        if not attrs and not apis:
            continue
        path = str(prim.GetPath())
        entry = {"type": str(prim.GetTypeName()), "apis": apis, "attributes": attrs,
                 "relationships": {str(rel.GetName()): [str(x) for x in rel.GetTargets()]
                                   for rel in prim.GetRelationships()
                                   if rel.GetName().startswith(("physics:", "physx"))}}
        if "PhysicsCollisionAPI" in apis:
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            entry["physics_material"] = str(material.GetPath()) if material else None
            if "PhysxCollisionAPI" not in apis:
                definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("PhysxCollisionAPI")
                if definition:
                    # Bind available engine schema inputs, but do not pretend
                    # an unapplied API has resolved effective contact values.
                    entry["unapplied_api_defaults"] = {
                        name: _usd_value(definition.GetAttributeFallbackValue(name))
                        for name in definition.GetPropertyNames()
                    }
        prims[path] = entry
        origins[path] = attr_origins
    return {
        "units": {"meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
                  "kilograms_per_unit": float(UsdPhysics.GetStageKilogramsPerUnit(stage)),
                  "up_axis": str(UsdGeom.GetStageUpAxis(stage))},
        "prims": prims, "resolution_evidence": origins, "capture_issues": issues,
    }


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def build_contract(snapshot, ordered_joint_paths, *, runtime_versions,
                   recipe=None, evidence=None, diagnostic_only=False,
                   scope="effective_physics", executor=None):
    """Build a signed-value document; missing requirements raise by default.

    diagnostic_only permits writing an explicitly incomplete historical sidecar.
    It never permits require_compatible to accept an incomplete chosen scope.
    recipe needs baseline/effective USD SHA256, canonical physics_profile and
    runtime_settings mappings. Those settings must be captured at execution,
    not inferred from whichever software happens to inspect old evidence.
    """
    if scope not in SCOPES:
        raise DynamicsContractError(f"Unknown compatibility scope {scope}")
    # Copy and normalize to JSON so callers cannot mutate a finished contract.
    physical = json.loads(canonical_json({"units": snapshot.get("units", {}),
                                        "prims": snapshot.get("prims", {}),
                                        "ordered_joint_paths": list(ordered_joint_paths)}))
    prims = physical["prims"]
    missing = list(snapshot.get("capture_issues", []))
    unresolved = []

    def require(path, names):
        attrs = prims.get(path, {}).get("attributes", {})
        for name in names:
            if attrs.get(name) is None:
                missing.append(f"missing {path}.{name}")
        return attrs

    units = physical["units"]
    for name in ("meters_per_unit", "kilograms_per_unit"):
        if not _finite_number(units.get(name)) or units[name] <= 0:
            missing.append(f"missing/invalid units.{name}")
    if units.get("up_axis") not in ("X", "Y", "Z"):
        missing.append("missing/invalid units.up_axis")
    scenes = [p for p, entry in prims.items() if entry.get("type") == "PhysicsScene"]
    if len(scenes) != 1:
        missing.append("exactly one PhysicsScene is required")
        physical["dt_s"] = None
    else:
        attrs = require(scenes[0], ("physics:gravityDirection", "physics:gravityMagnitude",
                                    "physxScene:timeStepsPerSecond", "physxScene:solverType"))
        rate = attrs.get("physxScene:timeStepsPerSecond")
        physical["dt_s"] = 1 / rate if _finite_number(rate) and rate > 0 else None
        if physical["dt_s"] != 1 / 240:
            missing.append("physics DT must equal explicit execution period 1/240 s")
    if len(ordered_joint_paths) != 7 or len(set(ordered_joint_paths)) != 7:
        missing.append("exactly seven distinct ordered joint paths are required")
    for path in ordered_joint_paths:
        attrs = require(path, ("physxJoint:armature", "drive:angular:physics:type",
                               "drive:angular:physics:stiffness", "drive:angular:physics:damping",
                               "drive:angular:physics:maxForce"))
        for name in ("drive:angular:physics:targetPosition", "drive:angular:physics:targetVelocity"):
            attrs.pop(name, None)
    counts = {"rigid_bodies": 0, "articulations": 0, "colliders": 0, "materials": 0}
    for path, entry in prims.items():
        apis = entry.get("apis", [])
        if "PhysicsRigidBodyAPI" in apis:
            counts["rigid_bodies"] += 1
            attrs = require(path, ("physics:mass", "physics:diagonalInertia", "physics:centerOfMass",
                                    "physics:principalAxes", "physxRigidBody:solverPositionIterationCount",
                                    "physxRigidBody:solverVelocityIterationCount"))
            for name, size in (("physics:mass", 1), ("physics:diagonalInertia", 3),
                               ("physics:centerOfMass", 3), ("physics:principalAxes", 4)):
                val = attrs.get(name)
                if val is None:
                    continue
                vals = [val] if size == 1 else val
                valid = isinstance(vals, list) and len(vals) == size and all(_finite_number(x) for x in vals)
                if valid and name in ("physics:mass", "physics:diagonalInertia"):
                    valid = all(x > 0 for x in vals)
                if valid and name == "physics:principalAxes":
                    valid = any(x != 0 for x in vals)
                if not valid:
                    unresolved.append(f"auto/invalid effective value: {path}.{name}")
        if "PhysicsArticulationRootAPI" in apis:
            counts["articulations"] += 1
            require(path, ("physxArticulation:solverPositionIterationCount",
                           "physxArticulation:solverVelocityIterationCount"))
        if "PhysicsMaterialAPI" in apis:
            counts["materials"] += 1
            require(path, ("physics:staticFriction", "physics:dynamicFriction", "physics:restitution"))
        if "PhysicsCollisionAPI" in apis:
            counts["colliders"] += 1
            attrs = require(path, ("physics:collisionEnabled",))
            for name in ("physxCollision:contactOffset", "physxCollision:restOffset"):
                if name not in attrs:
                    defaults = entry.get("unapplied_api_defaults", {})
                    if "PhysxCollisionAPI" not in apis and defaults.get(name) is not None:
                        unresolved.append(f"unapplied API; effective engine value unverified: {path}.{name}")
                    else:
                        missing.append(f"missing {path}.{name}")
                elif not _finite_number(attrs[name]):
                    unresolved.append(f"auto effective value: {path}.{name}")
            material = entry.get("physics_material")
            if not material or "PhysicsMaterialAPI" not in prims.get(material, {}).get("apis", []):
                unresolved.append(f"unresolved effective physics material: {path}")
    for name, count in counts.items():
        if count == 0:
            missing.append(f"missing {name}")
    version_issues = []
    if isinstance(runtime_versions, dict):
        runtime_versions = {k: v for k, v in runtime_versions.items() if k != "ruckig"}
    if not isinstance(runtime_versions, dict) or any(
            not runtime_versions.get(k) or runtime_versions[k] in ("unknown", "unverified")
            for k in VERSION_KEYS):
        version_issues.append("runtime_versions must bind Isaac package/build, USD and PhysX schema from the run")
    recipe_issues = []
    if not isinstance(recipe, dict):
        recipe_issues.append("missing simulator recipe bindings")
    else:
        for key in ("baseline_scene_sha256", "effective_scene_sha256"):
            if not _valid_hash(recipe.get(key)):
                recipe_issues.append(f"missing/invalid recipe.{key}")
        for key in ("physics_profile", "runtime_settings"):
            if not isinstance(recipe.get(key), dict) or not recipe[key]:
                recipe_issues.append(f"missing recipe.{key}")
        if not evidence or evidence.get("source_usd_sha256") != recipe.get("effective_scene_sha256"):
            recipe_issues.append("recipe effective scene hash does not match inspected source USD")
    execution = json.loads(canonical_json(executor))
    scopes = {
        "effective_physics": {"issues": missing + version_issues + unresolved},
        "simulator_recipe": {"issues": missing + version_issues + recipe_issues},
    }
    for details in scopes.values():
        details["complete"] = not details["issues"]
    document = {
        "schema": SCHEMA, "physics": physical, "physics_sha256": sha256_json(physical),
        "execution": execution, "execution_sha256": sha256_json(execution),
        "runtime_versions": runtime_versions, "recipe": recipe,
        "scopes": scopes, "complete": scopes[scope]["complete"],
        "issues": scopes[scope]["issues"], "unresolved_effective_values": unresolved,
        "selected_scope": scope, "evidence": evidence or {},
        "resolution_evidence": snapshot.get("resolution_evidence", {}),
        "counts": counts,
    }
    document["binding_sha256"] = sha256_json({k: document[k] for k in (
        "schema", "physics_sha256", "execution_sha256", "runtime_versions", "recipe", "scopes")})
    document["recipe_sha256"] = sha256_json({k: document[k] for k in (
        "schema", "physics_sha256", "runtime_versions", "recipe")})
    if not document["complete"] and not diagnostic_only:
        raise DynamicsContractError("; ".join(document["issues"]), document)
    return document


def _differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            result.extend(_differences(a.get(key), b.get(key), f"{path}/{key}"))
        return result
    if a != b:
        return [{"path": path, "left": a, "right": b}]
    return []


def compare_contracts(left, right, *, scope="effective_physics", compare_execution=False):
    if scope not in SCOPES:
        raise DynamicsContractError(f"Unknown compatibility scope {scope}")
    issues = []
    for label, doc in (("left", left), ("right", right)):
        if doc.get("schema") != SCHEMA:
            issues.append(f"{label}: unsupported contract schema")
        if not doc.get("scopes", {}).get(scope, {}).get("complete", False):
            issues.append(f"{label}: incomplete {scope}")
        for name in ("physics", "execution"):
            if doc.get(f"{name}_sha256") != sha256_json(doc.get(name)):
                issues.append(f"{label}: {name} integrity hash mismatch")
        binding = {k: doc.get(k) for k in (
            "schema", "physics_sha256", "execution_sha256", "runtime_versions", "recipe", "scopes")}
        if doc.get("binding_sha256") != sha256_json(binding):
            issues.append(f"{label}: binding integrity hash mismatch")
        recipe_binding = {k: doc.get(k) for k in (
            "schema", "physics_sha256", "runtime_versions", "recipe")}
        if doc.get("recipe_sha256") != sha256_json(recipe_binding):
            issues.append(f"{label}: recipe integrity hash mismatch")
        if compare_execution and not doc.get("execution"):
            issues.append(f"{label}: missing executor contract")
    fields = ["physics", "runtime_versions"]
    if compare_execution:
        fields.append("execution")
    if scope == "simulator_recipe":
        fields.append("recipe")
    differences = []
    for name in fields:
        differences.extend(_differences(left.get(name), right.get(name), name))
    return {"scope": scope, "compare_execution": compare_execution,
            "compatible": not issues and not differences,
            "issues": issues, "differences": differences,
            "claim": "matching simulator inputs; effective auto values and physical calibration unverified"
            if scope == "simulator_recipe" else "matching extracted resolved static physical settings"}


def require_compatible(left, right, *, scope="effective_physics", compare_execution=False):
    result = compare_contracts(left, right, scope=scope, compare_execution=compare_execution)
    if not result["compatible"]:
        raise DynamicsContractError("incompatible dynamics contracts: " + canonical_json(result))
    return result
