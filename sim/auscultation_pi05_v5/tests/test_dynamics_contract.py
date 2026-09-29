"""Fail-closed contract tests; no simulator or GPU is started."""
from copy import deepcopy

import pytest

from sim.auscultation_pi05_v5.dynamics_contract import (
    DynamicsContractError, build_contract, canonical_json, compare_contracts,
    execution_contract, require_compatible, sha256_json,
)


@pytest.fixture
def snapshot():
    prims = {
        "/scene": {"type": "PhysicsScene", "apis": ["PhysxSceneAPI"],
                   "attributes": {"physics:gravityDirection": [0, 0, -1],
                                  "physics:gravityMagnitude": 9.81,
                                  "physxScene:timeStepsPerSecond": 240,
                                  "physxScene:solverType": "TGS"}},
        "/body": {"type": "Xform", "apis": ["PhysicsRigidBodyAPI"],
                  "attributes": {"physics:mass": 1., "physics:diagonalInertia": [.1]*3,
                                 "physics:centerOfMass": [0., 0., 0.],
                                 "physics:principalAxes": [1., 0., 0., 0.],
                                 "physxRigidBody:solverPositionIterationCount": 32,
                                 "physxRigidBody:solverVelocityIterationCount": 8}},
        "/root": {"type": "PhysicsFixedJoint", "apis": ["PhysicsArticulationRootAPI"],
                  "attributes": {"physxArticulation:solverPositionIterationCount": 32,
                                 "physxArticulation:solverVelocityIterationCount": 8}},
        "/material": {"type": "Material", "apis": ["PhysicsMaterialAPI"],
                      "attributes": {"physics:staticFriction": .5,
                                     "physics:dynamicFriction": .4,
                                     "physics:restitution": 0.}},
        "/collider": {"type": "Cube", "apis": ["PhysicsCollisionAPI"],
                      "physics_material": "/material",
                      "attributes": {"physics:collisionEnabled": True,
                                     "physxCollision:contactOffset": .00003,
                                     "physxCollision:restOffset": 0.}},
    }
    for i in range(7):
        prims[f"/joint{i}"] = {
            "type": "PhysicsRevoluteJoint", "apis": ["PhysxJointAPI"],
            "attributes": {"physxJoint:armature": 0.,
                           "drive:angular:physics:type": "acceleration",
                           "drive:angular:physics:stiffness": 1600.,
                           "drive:angular:physics:damping": 80.,
                           "drive:angular:physics:maxForce": 40.},
        }
    return {"units": {"meters_per_unit": 1., "kilograms_per_unit": 1., "up_axis": "Z"},
            "prims": prims}


def contract(snapshot, **kwargs):
    return build_contract(snapshot, [f"/joint{i}" for i in range(7)],
                          runtime_versions={"isaac_sim": "5.1.0.0", "isaac_sim_build": "5.1.0-rc.19+release.26219.9c81211b.gl", "usd": "0.24.11",
                                            "physx_schema": "107.3.26", "ruckig": "0.12.2"},
                          **kwargs)


def test_canonical_hash_and_nonfinite_rejection():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})
    assert canonical_json({"a": 1}) == '{"a":1}'
    with pytest.raises(ValueError):
        canonical_json({"a": float("nan")})


def test_equal_effective_values_ignore_file_evidence(snapshot):
    a = contract(snapshot, evidence={"source_usd_sha256": "a" * 64})
    b = contract(snapshot, evidence={"source_usd_sha256": "b" * 64})
    assert a["physics_sha256"] == b["physics_sha256"]
    assert require_compatible(a, b)["compatible"] is True
    assert a["physics"]["dt_s"] == 1 / 240


def test_known_old_to_live_mismatch(snapshot):
    old = contract(snapshot)
    live = deepcopy(snapshot)
    live["prims"]["/body"]["attributes"]["physxRigidBody:solverVelocityIterationCount"] = 1
    live["prims"]["/root"]["attributes"]["physxArticulation:solverVelocityIterationCount"] = 1
    for i in range(7):
        live["prims"][f"/joint{i}"]["attributes"]["physxJoint:armature"] = .01
    result = compare_contracts(old, contract(live))
    assert result["compatible"] is False
    assert len(result["differences"]) == 9
    assert any("armature" in x["path"] for x in result["differences"])
    with pytest.raises(DynamicsContractError, match="incompatible"):
        require_compatible(old, contract(live))


@pytest.mark.parametrize("path,attribute", [
    ("/scene", "physxScene:timeStepsPerSecond"),
    ("/joint2", "physxJoint:armature"),
    ("/joint4", "drive:angular:physics:damping"),
    ("/body", "physics:diagonalInertia"),
    ("/root", "physxArticulation:solverVelocityIterationCount"),
    ("/material", "physics:dynamicFriction"),
    ("/collider", "physxCollision:contactOffset"),
])
def test_missing_required_values_fail_closed(snapshot, path, attribute):
    del snapshot["prims"][path]["attributes"][attribute]
    with pytest.raises(DynamicsContractError, match=attribute):
        contract(snapshot)


def test_auto_mass_and_unbound_material_cannot_claim_compatibility(snapshot):
    snapshot["prims"]["/body"]["attributes"]["physics:mass"] = 0.
    snapshot["prims"]["/collider"]["physics_material"] = None
    a = contract(snapshot, diagnostic_only=True)
    assert not a["complete"]
    assert len(a["issues"]) == 2
    assert not compare_contracts(a, a)["compatible"]
    with pytest.raises(DynamicsContractError):
        contract(snapshot)


def test_joint_order_is_part_of_contract(snapshot):
    a = contract(snapshot)
    b = deepcopy(a)
    b["physics"]["ordered_joint_paths"].reverse()
    b["physics_sha256"] = sha256_json(b["physics"])
    assert not compare_contracts(a, b)["compatible"]


def test_mutated_contract_hash_is_rejected(snapshot):
    a = contract(snapshot)
    b = deepcopy(a)
    b["physics"]["dt_s"] = 1 / 120
    assert not compare_contracts(a, b)["compatible"]
    assert any("integrity" in x for x in compare_contracts(a, b)["issues"])


def test_execution_contract_is_explicit_and_version_pinned(snapshot):
    e = execution_contract()
    assert e["ruckig_version"] == "0.12.2"
    assert e["velocity_rad_s"] == [.25] * 7
    assert e["acceleration_rad_s2"] == [.5] * 7
    assert e["jerk_rad_s3"] == [8.] * 7
    assert e["physics_period_s"] == 1 / 240
    assert e["maximum_open_loop_s"] == .05
    assert e["observation_period_s"] == .05
    a = contract(snapshot, executor=execution_contract())
    b = deepcopy(a)
    b["execution"]["jerk_rad_s3"][0] = 9.
    b["execution_sha256"] = sha256_json(b["execution"])
    assert not compare_contracts(a, b, compare_execution=True)["compatible"]
    with pytest.raises(DynamicsContractError):
        execution_contract(ruckig_version="0.13.0")


def test_historical_runtime_unknown_is_incomplete(snapshot):
    with pytest.raises(DynamicsContractError, match="runtime_versions"):
        build_contract(snapshot, [f"/joint{i}" for i in range(7)], runtime_versions=None)


def test_identical_recipe_permits_auto_values_without_effective_physics_claim(snapshot):
    snapshot["prims"]["/body"]["attributes"]["physics:mass"] = 0.
    recipe = {"baseline_scene_sha256": "a"*64, "effective_scene_sha256": "b"*64,
              "physics_profile": {"position_iterations": 32, "velocity_iterations": 8},
              "runtime_settings": {"device": "cpu", "physics_dt_s": 1/240}}
    a = contract(snapshot, scope="simulator_recipe", recipe=recipe,
                 evidence={"source_usd_sha256": "b"*64})
    assert a["complete"]
    assert a["unresolved_effective_values"]
    assert require_compatible(a, a, scope="simulator_recipe")["compatible"]
    assert not compare_contracts(a, a, scope="effective_physics")["compatible"]
    changed = deepcopy(snapshot)
    changed["prims"]["/joint0"]["attributes"]["physxJoint:armature"] = .01
    b = contract(changed, scope="simulator_recipe", recipe=recipe,
                 evidence={"source_usd_sha256": "b"*64})
    assert not compare_contracts(a, b, scope="simulator_recipe")["compatible"]


def test_recipe_needs_exact_scene_and_settings_bindings(snapshot):
    with pytest.raises(DynamicsContractError, match="recipe"):
        contract(snapshot, scope="simulator_recipe")
    recipe = {"baseline_scene_sha256": "a"*64, "effective_scene_sha256": "b"*64,
              "physics_profile": {"velocity_iterations": 8},
              "runtime_settings": {"device": "cpu"}}
    with pytest.raises(DynamicsContractError, match="inspected source"):
        contract(snapshot, scope="simulator_recipe", recipe=recipe,
                 evidence={"source_usd_sha256": "c"*64})


def test_teacher_and_deployment_executors_are_separate(snapshot):
    teacher = contract(snapshot, executor={"controller": "NormalExecutor"})
    learner = contract(snapshot, executor=execution_contract())
    assert teacher["recipe_sha256"] == learner["recipe_sha256"]
    assert compare_contracts(teacher, learner)["compatible"]
    assert not compare_contracts(teacher, learner, compare_execution=True)["compatible"]


def test_unapplied_collision_schema_defaults_are_recipe_inputs_not_effective_values(snapshot):
    collider = snapshot["prims"]["/collider"]
    del collider["attributes"]["physxCollision:contactOffset"]
    collider["unapplied_api_defaults"] = {"physxCollision:contactOffset": {"usd_infinity": "-"}}
    recipe = {"baseline_scene_sha256": "a"*64, "effective_scene_sha256": "b"*64,
              "physics_profile": {"velocity_iterations": 8}, "runtime_settings": {"device": "cpu"}}
    a = contract(snapshot, scope="simulator_recipe", recipe=recipe,
                 evidence={"source_usd_sha256": "b"*64})
    assert a["complete"]
    assert not a["scopes"]["effective_physics"]["complete"]


def test_usd_schema_default_and_time_rate(tmp_path):
    Usd = pytest.importorskip("pxr.Usd")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    from sim.auscultation_pi05_v5.dynamics_contract import snapshot_stage
    stage = Usd.Stage.CreateInMemory()
    material = UsdPhysics.MaterialAPI.Apply(stage.DefinePrim("/material", "Material"))
    fallback = material.GetStaticFrictionAttr().Get()
    assert fallback is not None
    snap = snapshot_stage(stage)
    assert snap["prims"]["/material"]["attributes"]["physics:staticFriction"] == fallback
    assert snap["resolution_evidence"]["/material"]["physics:staticFriction"] == "schema_fallback"
    # USD display frame rate is deliberately not interpreted as physics DT.
    stage.SetTimeCodesPerSecond(24)
    with pytest.raises(DynamicsContractError, match="PhysicsScene"):
        contract(snap)
