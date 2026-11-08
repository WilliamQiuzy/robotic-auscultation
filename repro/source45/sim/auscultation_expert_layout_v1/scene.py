"""Load the hash-bound approved physical layout without repeating CAD repairs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
ROBOT = '/World/StethoArmMount/OpenArmStetho'
WRIST = ROBOT + '/openarm_left_link7'
FIXED = '/World/Stethoscope/Fixed'
MOVING = '/World/Stethoscope/Carriage'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_sources():
    manifest = ROOT / 'layout_inputs.json'
    data = json.loads(manifest.read_text())
    paths = {'layout_inputs': manifest}
    for relative, expected in data['source_files'].items():
        path = (PROJECT / relative).resolve()
        if not path.is_relative_to(PROJECT.resolve()) or sha(path) != expected:
            raise ValueError('Approved physical input changed: ' + relative)
        paths[relative] = path
    return data, paths


def prepare_stage(destination):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    data, paths = physical_sources()
    before = {name: sha(path) for name, path in paths.items()}
    source = Usd.Stage.Open(str(PROJECT / data['scene_relative_to_project']))
    stage = Usd.Stage.Open(source.Flatten())
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.
    assert str(UsdGeom.GetStageUpAxis(stage)) == 'Z'
    for path in ['/Render', '/Replicator', '/Orchestrator']:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
    cache = UsdGeom.XformCache()
    camera = UsdGeom.Camera(stage.GetPrimAtPath('/World/RobotCam'))
    matrix = np.asarray(cache.GetLocalToWorldTransform(camera.GetPrim())).T
    optical = matrix @ np.diag([1., -1., -1., 1.])
    np.testing.assert_allclose(optical, data['camera']['world_from_optical'], atol=1e-10, rtol=0)
    for key, attr in [('focal_length', camera.GetFocalLengthAttr()),
                      ('horizontal_aperture', camera.GetHorizontalApertureAttr()),
                      ('vertical_aperture', camera.GetVerticalApertureAttr())]:
        assert attr.Get() == data['camera'][key], key
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    for name in ['Patient', 'Stool', 'Table']:
        r = bounds.ComputeWorldBound(stage.GetPrimAtPath('/World/' + name)).ComputeAlignedRange()
        np.testing.assert_allclose([r.GetMin(), r.GetMax()], data['approved_variant']['bounds_world_m'][name], atol=1e-6, rtol=0)
    body_count = joint_count = patient_colliders = 0
    for p in stage.Traverse():
        path = str(p.GetPath())
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and (path.startswith(ROBOT) or path in (FIXED, MOVING)):
            assert UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get(), path
            assert p.HasAPI(PhysxSchema.PhysxContactReportAPI), path
            body_count += 1
        if p.IsA(UsdPhysics.Joint):
            assert UsdPhysics.Joint(p).GetJointEnabledAttr().Get(), path
            joint_count += 1
        if path.startswith('/World/Patient/') and p.HasAPI(UsdPhysics.CollisionAPI):
            patient_colliders += int(bool(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()))
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            assert PhysxSchema.PhysxArticulationAPI(p).GetEnabledSelfCollisionsAttr().Get(), path
    assert body_count >= 17 and joint_count >= 15 and patient_colliders > 0
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(destination))
    for name, path in paths.items():
        assert sha(path) == before[name], name
    (destination.parent / 'scene_repairs.json').write_text(json.dumps({
        'scope': 'Exact approved layout; previous CAD and physics settings retained without reapplication.',
        'physical_input_sha256': before, 'patient_translation_world_m': data['translation_world_m'],
        'enabled_robot_module_bodies': body_count, 'enabled_joints': joint_count,
        'enabled_patient_colliders': patient_colliders, 'camera_and_layout_verified': True,
        'source_layers_unchanged': True, 'exported_scene_sha256': sha(destination),
    }, indent=2) + '\n')
    return stage
