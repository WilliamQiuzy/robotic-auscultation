"""Prepare a reversible raised-seat scene and RGB-D inputs. No robot commands."""
import os
from pathlib import Path
import json
import hashlib

os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width': 1920, 'height': 1080,
                     'renderer': 'RayTracedLighting', 'multi_gpu': False})
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSkel, Vt
import omni.usd
import omni.replicator.core as rep

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'real2sim_20260922/auscultation_real2sim.usdc'
ROOT.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def move_z(prim, dz):
    xf = UsdGeom.Xformable(prim)
    matrix = xf.GetLocalTransformation()
    position = matrix.ExtractTranslation()
    matrix.SetTranslateOnly(Gf.Vec3d(position[0], position[1], position[2] + dz))
    xf.ClearXformOpOrder()
    xf.AddTransformOp(opSuffix='seatAdjustment').Set(matrix)

source_hash = sha(SOURCE)
baseline = Usd.Stage.Open(str(SOURCE))
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
seat_top = float(bbox.ComputeWorldBound(baseline.GetPrimAtPath('/World/Stool/seat')).ComputeAlignedRange().GetMax()[2])
delta = seat_top * .15
layer = Sdf.Layer.CreateAnonymous()
layer.subLayerPaths = ['../real2sim_20260922/auscultation_real2sim.usdc']
layer.Export(str(ROOT / 'raised_seat.usda'))
stage = Usd.Stage.Open(str(ROOT / 'raised_seat.usda'))
layer = stage.GetRootLayer()
stage.SetEditTarget(layer)
# Stage-level metadata does not inherit the source layer's unit fallback.
UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(baseline))
UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(baseline))
assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
for patient_prim in list(Usd.PrimRange(stage.GetPrimAtPath('/World/Patient'))):
    if patient_prim.IsInstance():
        patient_prim.SetInstanceable(False)
move_z(stage.GetPrimAtPath('/World/Patient'), delta)
move_z(stage.GetPrimAtPath('/World/Stool/seat'), delta)
for child in stage.GetPrimAtPath('/World/Stool').GetChildren():
    if child.GetName().startswith('leg'):
        cylinder = UsdGeom.Cylinder(child)
        cylinder.GetHeightAttr().Set(float(cylinder.GetHeightAttr().Get()) + delta)
        move_z(child, delta / 2)
    elif child.GetName().startswith('rung'):
        z = UsdGeom.Xformable(child).GetLocalTransformation().ExtractTranslation()[2]
        move_z(child, float(z) * .15)
fixture = stage.GetPrimAtPath('/World/ContactFixture')
if fixture:
    fixture.SetActive(False)
layer.Save()

# This seated asset has authored mesh vertices and retained SkelBinding attributes,
# but no SkelRoot. Export the authored geometry without applying a second skin warp.
# A separate RGB-D/mesh comparison audits contact-surface correspondence.
mesh_rows = []
arrays = {}
xf_cache = UsdGeom.XformCache()
assert not any(p.IsA(UsdSkel.Root) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Patient')))
for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Patient')):
    if not prim.IsA(UsdGeom.Mesh):
        continue
    mesh = UsdGeom.Mesh(prim)
    world = xf_cache.GetLocalToWorldTransform(prim)
    vertices = np.array([world.Transform(Gf.Vec3d(*map(float, p))) for p in mesh.GetPointsAttr().Get()], dtype=np.float32)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    triangles = []
    offset = 0
    for n in counts:
        face = indices[offset:offset+n]
        triangles.extend([[face[0], face[i], face[i+1]] for i in range(1, n-1)])
        offset += n
    faces = np.asarray(triangles, dtype=np.int32)
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        faces = faces[:, [0, 2, 1]]
    name = prim.GetName()
    arrays[name + '_vertices_world_m'] = vertices
    arrays[name + '_triangles'] = faces
    collision = UsdPhysics.CollisionAPI(prim)
    mesh_rows.append({'name': name, 'source_prim': str(mesh.GetPath()),
                      'vertices': len(vertices), 'triangles': len(faces),
                      'existing_collision': bool(collision and collision.GetCollisionEnabledAttr().Get()),
                      'subdivision_scheme': str(mesh.GetSubdivisionSchemeAttr().Get()),
                      'export_method': 'authored seated control vertices transformed to world; no additional skin warp',
                      'bounds_world_m': [vertices.min(0).tolist(), vertices.max(0).tolist()]})
assert len(mesh_rows) == 7, mesh_rows
np.savez_compressed(ROOT / 'patient_seated_surface_assets.npz', **arrays)

camera = UsdGeom.Camera(stage.GetPrimAtPath('/World/RobotCam'))
f, ha, va = [float(x.Get()) for x in [camera.GetFocalLengthAttr(), camera.GetHorizontalApertureAttr(), camera.GetVerticalApertureAttr()]]
ho, vo = float(camera.GetHorizontalApertureOffsetAttr().Get()), float(camera.GetVerticalApertureOffsetAttr().Get())
intr = {'width': 1920, 'height': 1080, 'fx': f / ha * 1920, 'fy': f / va * 1080,
        'cx': 960 - ho / ha * 1920, 'cy': 540 + vo / va * 1080, 'depth_scale': 1.0}
cam_usd = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim())).T
manifest = {'source_sha256': source_hash, 'source_scene': str(SOURCE),
            'meters_per_unit': UsdGeom.GetStageMetersPerUnit(stage),
            'up_axis': str(UsdGeom.GetStageUpAxis(stage)),
            'seat_height_before_m': seat_top, 'height_fraction': .15, 'patient_translation_z_m': delta,
            'seat_height_after_m': seat_top + delta, 'person_scale_changed': False,
            'camera_pose_and_optics_changed': False, 'intrinsics_rendered': intr,
            'world_from_camera_optical': (cam_usd @ np.diag([1, -1, -1, 1])).tolist(),
            'surface_meshes': mesh_rows,
            'surface_status': 'baked seated control mesh; subdivision/render correspondence requires audit before physical training',
            'deformation_status': 'not implemented or mechanically calibrated',
            'acoustics_status': 'absent', 'demonstrations_generated': False, 'training_started': False}
(ROOT / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2))

def render(path, stem):
    # All freezes happen in a disposable flattened render copy, never the candidate.
    src = Usd.Stage.Open(str(path))
    flat = src.Flatten()
    audit = Usd.Stage.Open(flat)
    for pth in ['/Render', '/Replicator', '/Orchestrator']:
        if audit.GetPrimAtPath(pth):
            audit.RemovePrim(pth)
    for prim in audit.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
    if audit.GetPrimAtPath('/World/ContactFixture'):
        audit.GetPrimAtPath('/World/ContactFixture').SetActive(False)
    render_path = Path('/tmp') / ('wam_asset_' + stem + '.usdc')
    audit.GetRootLayer().Export(str(render_path))
    omni.usd.get_context().open_stage(str(render_path))
    for _ in range(35): app.update()
    product = rep.create.render_product('/World/RobotCam', (1920, 1080))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
    rgb.attach([product]); depth.attach([product])
    for _ in range(20): app.update()
    rep.orchestrator.step(rt_subframes=8, delta_time=0.)
    Image.fromarray(rgb.get_data()[:, :, :3]).save(ROOT / (stem + '_rgb.png'))
    z = np.asarray(depth.get_data(), dtype=np.float32).copy()
    # This camera is about 0.4 m from the torso. Catch cm/m stage errors before inference.
    assert .15 < float(z[540, 960]) < 2., ('unexpected metric torso depth', stem, z[540, 960])
    np.save(ROOT / (stem + '_depth_m.npy'), z)
    valid = np.isfinite(z) & (z > .05) & (z < 10.)
    mm = np.where(valid, np.clip(z * 1000, 0, 65535), 0).astype(np.uint16)
    Image.fromarray(mm).save(ROOT / (stem + '_depth_mm.png'))
    rgb.detach([product.path]); depth.detach([product.path]); product.destroy()
    print('RENDERED', stem, flush=True)

render(SOURCE, 'before')
render(ROOT / 'raised_seat.usda', 'raised')
assert sha(SOURCE) == source_hash
manifest['raised_scene_sha256'] = sha(ROOT / 'raised_seat.usda')
manifest['surface_assets_sha256'] = sha(ROOT / 'patient_seated_surface_assets.npz')
(ROOT / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2))
print('ASSETS_PREPARED', delta, flush=True)
app.close()
