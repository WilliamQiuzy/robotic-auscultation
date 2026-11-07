"""Explicit simulation candidate. Armature is an uncalibrated engineering prior."""
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import math


@dataclass(frozen=True)
class PhysicsProfile:
    position_iterations:int=128
    velocity_iterations:int=1
    armature_kg_m2:float=.01

    def validate(self):
        if type(self.position_iterations) is not int or not 1<=self.position_iterations<=255:
            raise ValueError('Invalid position iterations')
        if type(self.velocity_iterations) is not int or not 0<=self.velocity_iterations<=255:
            raise ValueError('Invalid velocity iterations')
        if not math.isfinite(self.armature_kg_m2) or not 0<=self.armature_kg_m2<=.02:
            raise ValueError('Armature outside declared experiment domain')


def apply_profile(stage,profile):
    from pxr import UsdPhysics,PhysxSchema
    profile.validate();joint_paths=[];before={}
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            api=PhysxSchema.PhysxRigidBodyAPI.Apply(p)
            api.CreateSolverPositionIterationCountAttr(profile.position_iterations)
            api.CreateSolverVelocityIterationCountAttr(profile.velocity_iterations)
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            api=PhysxSchema.PhysxArticulationAPI.Apply(p)
            api.CreateSolverPositionIterationCountAttr(profile.position_iterations)
            api.CreateSolverVelocityIterationCountAttr(profile.velocity_iterations)
        if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName().startswith('openarm_left_joint'):
            api=PhysxSchema.PhysxJointAPI.Apply(p)
            before[str(p.GetPath())]=api.GetArmatureAttr().Get()
            api.CreateArmatureAttr(profile.armature_kg_m2);joint_paths.append(str(p.GetPath()))
    if len(joint_paths)!=7:raise ValueError('Expected exactly seven left-arm joints')
    return dict(profile=asdict(profile),armature_before=before,joint_paths=joint_paths,
        armature_status='uncalibrated engineering prior; not measured rotor inertia',
        contact_surface='unchanged rigid patient',collision_geometry='unchanged',drive_gains='unchanged',
        claim='simulation candidate only; real actuator identification and sensitivity testing still required')


def make_env(output,profile=PhysicsProfile()):
    """Scoped stage-factory injection into frozen V1, in an exclusive Isaac process."""
    import sim.auscultation_expert_layout_v1.env as runtime
    original=runtime.prepare_stage
    def prepare(destination):
        stage=original(destination)
        before=hashlib.sha256(Path(destination).read_bytes()).hexdigest()
        receipt=apply_profile(stage,profile);stage.GetRootLayer().Export(str(destination))
        receipt.update(baseline_scene_sha256=before,
            effective_scene_sha256=hashlib.sha256(Path(destination).read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        Path(destination).with_name('physics_profile.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return stage
    runtime.prepare_stage=prepare
    try:return runtime.IsaacContactEnv(output)
    finally:runtime.prepare_stage=original
