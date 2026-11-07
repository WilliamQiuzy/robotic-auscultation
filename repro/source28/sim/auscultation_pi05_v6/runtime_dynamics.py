"""Capture simulator identity in the running Isaac process before robot motion."""
import importlib.metadata
import json
from pathlib import Path
import sys
from .dynamics_contract import snapshot_stage,build_contract,sha256_file,sha256_json,DynamicsContractError


def capture_runtime_contract(env, *, executor=None):
    import isaacsim
    from pxr import Usd,PhysxSchema
    import sim.auscultation_expert_layout_v1.env as environment
    source=env.out/'training_scene.usdc'
    profile=json.loads((env.out/'physics_profile.json').read_text())
    source_sha=sha256_file(source)
    if source_sha!=profile['effective_scene_sha256']:
        raise ValueError('Effective simulator scene changed before contract capture')
    native={}
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name.startswith(('pxr.PhysxSchema','omni.physx.bindings')) and path and path.endswith('.so'):
            native[name]=sha256_file(path)
    if not any(n.startswith('pxr.PhysxSchema') for n in native) or not any(n.startswith('omni.physx.bindings') for n in native):
        raise ValueError('Loaded PhysX native bindings not identifiable')
    build=(Path(isaacsim.__file__).parent/'VERSION').read_text().strip()
    versions=dict(isaac_sim=importlib.metadata.version('isaacsim'),isaac_sim_build=build,
        usd='.'.join(map(str,Usd.GetVersion())),physx_schema='native-bindings-sha256:'+sha256_json(native))
    dt=float(env.world.get_physics_dt())
    if abs(dt-1/240)>1e-12:raise ValueError('Unexpected actual physics timestep')
    settings=dict(physics_period_s=dt,backend=env.world.backend,device=env.world.device,
        environment_source_sha256=sha256_file(environment.__file__))
    recipe=dict(baseline_scene_sha256=profile['baseline_scene_sha256'],effective_scene_sha256=source_sha,
        physics_profile=profile['profile'],runtime_settings=settings)
    paths=sorted(profile['joint_paths'],key=lambda p:int(p.rsplit('joint',1)[1]))
    evidence=dict(source_usd_sha256=source_sha,native_bindings_sha256=native,
        capture='actual running Isaac process before controlled motion',
        world_time_s=float(env.world.current_time),recipe_source_sha256=sha256_file(__file__))
    try:
        contract=build_contract(snapshot_stage(env.stage),paths,runtime_versions=versions,
            recipe=recipe,evidence=evidence,scope='simulator_recipe',executor=executor)
    except DynamicsContractError as error:
        if error.partial_contract is not None:
            (env.out/'dynamics_contract_incomplete.json').write_text(json.dumps(error.partial_contract,indent=2)+'\n')
        raise
    (env.out/'dynamics_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    return contract
