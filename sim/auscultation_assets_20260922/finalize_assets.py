"""Build a static contact-asset candidate and audit geometry. No motion/training."""
import os
import json
import hashlib
from pathlib import Path
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'asset_manifest.json').read_text())
predictions = json.loads((ROOT / 'model_predictions.json').read_text())
surfaces = np.load(ROOT / 'patient_seated_surface_assets.npz')

def new_overlay(filename, base):
    layer = Sdf.Layer.CreateAnonymous()
    layer.subLayerPaths = [base]
    layer.Export(str(ROOT / filename))
    stage = Usd.Stage.Open(str(ROOT / filename))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    return stage

stage = new_overlay('contact_static.usda', 'raised_seat.usda')
material = UsdShade.Material.Define(stage, '/World/TrainingContactMaterial')
physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
physics_material.CreateStaticFrictionAttr(.5)
physics_material.CreateDynamicFrictionAttr(.4)
physics_material.CreateRestitutionAttr(0.)
material.GetPrim().SetCustomDataByKey('parameter_status', 'nominal engineering values; no tissue calibration')
triangles = []
owners = []
for row in manifest['surface_meshes']:
    prim = stage.GetPrimAtPath(row['source_prim'])
    visible = UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
    row['visible'] = bool(visible)
    # Skin and clothing surfaces are obstacles. Hidden props remain excluded.
    contact_surface = visible and any(t in row['name'] for t in ['skin', 'fabric', 'sneakers'])
    row['static_contact_enabled'] = bool(contact_surface)
    if not contact_surface:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        continue
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
    physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx.CreateContactOffsetAttr(.001)
    physx.CreateRestOffsetAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
    vertices = surfaces[row['name'] + '_vertices_world_m']
    faces = surfaces[row['name'] + '_triangles']
    triangles.append(vertices[faces])
    owners.extend([row['name']] * len(faces))
stage.GetRootLayer().Save()
triangles = np.concatenate(triangles).astype(np.float64)
owners = np.asarray(owners)

def first_intersection(origin, direction):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = b-a, c-a
    h = np.cross(np.broadcast_to(direction, e2.shape), e2)
    determinant = np.einsum('ij,ij->i', e1, h)
    valid = np.abs(determinant) > 1.e-10
    inv = np.zeros_like(determinant)
    inv[valid] = 1. / determinant[valid]
    s = origin-a
    u = np.einsum('ij,ij->i', s, h) * inv
    q = np.cross(s, e1)
    v = (q @ direction) * inv
    t = np.einsum('ij,ij->i', e2, q) * inv
    valid &= (u >= 0) & (v >= 0) & (u+v <= 1) & (t > .001)
    if not np.any(valid): return None
    t[~valid] = np.inf
    index = int(np.argmin(t))
    return {'ray_z_m': float(t[index]), 'mesh': str(owners[index]),
            'triangle_index_in_contact_meshes': index, 'barycentric': [float(1-u[index]-v[index]), float(u[index]), float(v[index])]}

T = np.asarray(manifest['world_from_camera_optical'])
K = manifest['intrinsics_rendered']
depth = np.load(ROOT / 'raised_depth_m.npy')
point_checks = []
for point in predictions['frames']['raised']['points']:
    u, v = point['uv_px']
    ray = np.array([(u-K['cx'])/K['fx'], (v-K['cy'])/K['fy'], 1.])
    hit = first_intersection(T[:3, 3], T[:3, :3] @ ray)
    point_checks.append({'name': point['name'], 'hit': hit,
                         'control_mesh_vs_rgbd_depth_error_m': abs(hit['ray_z_m']-point['camera_xyz_m'][2]) if hit else None})

source = Usd.Stage.Open(manifest['source_scene'])
source_xf, new_xf = UsdGeom.XformCache(), UsdGeom.XformCache()
unchanged = {}
for path in ['/World/RobotCam', '/World/StethoArmMount/OpenArmStetho/openarm_left_link7']:
    original = np.asarray(source_xf.GetLocalToWorldTransform(source.GetPrimAtPath(path)))
    current = np.asarray(new_xf.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
    unchanged[path] = bool(np.array_equal(original, current))
assert all(unchanged.values())
leg_bottoms = []
for child in stage.GetPrimAtPath('/World/Stool').GetChildren():
    if child.GetName().startswith('leg'):
        height = float(UsdGeom.Cylinder(child).GetHeightAttr().Get())
        center = new_xf.GetLocalToWorldTransform(child).ExtractTranslation()[2]
        leg_bottoms.append(float(center-height/2))
assert max(abs(z) for z in leg_bottoms) < 1.e-5
assert hashlib.sha256(Path(manifest['source_scene']).read_bytes()).hexdigest() == manifest['source_sha256']

# Independent display-only layer: these candidate markers must never enter actor inputs.
debug = new_overlay('candidate_points_review.usda', 'contact_static.usda')
group = UsdGeom.Xform.Define(debug, '/World/DebugCandidatePoints')
group.GetPrim().SetCustomDataByKey('training_input', False)
group.GetPrim().SetCustomDataByKey('accepted_targets', False)
for point in predictions['frames']['raised']['points']:
    if not point['depth_valid']: continue
    sphere = UsdGeom.Sphere.Define(debug, '/World/DebugCandidatePoints/' + point['name'])
    sphere.CreateRadiusAttr(.0035)
    p = np.asarray(point['world_xyz_m'])
    n = np.asarray(point.get('normal_world_from_depth', [0, 0, 0]))
    sphere.AddTranslateOp().Set(Gf.Vec3d(*(p + .002*n)))
    color = (.95, .22, .08) if point['name'].startswith('R') else (.05, .5, .95)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    sphere.GetPrim().SetCustomDataByKey('candidate_only', True)
debug.GetRootLayer().Save()

props = {'schema_version': 1,
         'status': 'static contact asset candidate, not a dynamically validated tissue environment',
         'patient': {'geometry': 'current authored seated mesh, separate clothing/skin',
                     'friction': {'static': .5, 'dynamic': .4, 'status': 'nominal unmeasured'},
                     'restitution': {'value': 0., 'status': 'nominal unmeasured'},
                     'contact_offset_m': .001, 'rest_offset_m': 0.,
                     'tissue_stiffness_n_per_m': None, 'tissue_damping_n_s_per_m': None,
                     'tissue_hysteresis': None, 'maximum_indentation_m': None,
                     'deformation': 'not implemented', 'breathing': 'not implemented'},
         'tool': {'spring_rate_n_per_mm': 1., 'preload_n': 1.5, 'hard_stop_travel_mm': 7.25,
                  'parameter_source': 'existing real2sim module and live Hall calibration force_model',
                  'hall_model_status': 'endpoint-anchored estimate, no independent bench load validation',
                  'nominal_sim_goal_n': 4., 'nominal_sim_band_n': .5},
         'audio': {'enabled': False, 'missing': True, 'reward_enabled': False},
         'required_measurements': ['reference-force versus Hall/displacement including hysteresis and hard stop',
                                   'force-indentation and recovery for phantom materials',
                                   'friction versus angle/speed', 'sensor latency/noise and current RGB-D calibration'],
         'version_1_scope': {'human_surface': 'rigid', 'compliance': 'existing tool spring',
                             'soft_tissue_deformation_required': False, 'simulation_rewards_enabled': False,
                             'proposed_learning': 'goal-conditioned single-point behavior cloning; not started',
                             'camera_and_point_visualization_user_review': 'accepted as initial development view; not new anatomical ground truth'},
         'before_training': ['confirm 10-point pose IK and collision-free approach/retract paths',
                             'module-versus-rigid-surface dynamic contact/force test',
                             'check subdivision versus collision at contact patches']}
(ROOT / 'contact_properties.json').write_text(json.dumps(props, indent=2))
errors = [r['control_mesh_vs_rgbd_depth_error_m'] for r in point_checks if r['hit']]
audit = {'source_unchanged': True, 'camera_and_wrist_unchanged': unchanged,
         'leg_bottom_world_z_m': leg_bottoms, 'stage_meters_per_unit': UsdGeom.GetStageMetersPerUnit(stage),
         'point_ray_checks': point_checks, 'points_intersecting_static_surface': len(errors),
         'maximum_control_mesh_depth_error_m': max(errors) if errors else None,
         'method': 'independent triangle-ray geometry check; not a PhysX force/contact validation',
         'model_targets_accepted': False, 'training_started': False, 'robot_motion_sent': False}
(ROOT / 'contact_geometry_audit.json').write_text(json.dumps(audit, indent=2))
(ROOT / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2))
print('CONTACT_ASSETS_READY', len(errors), 'max depth error', max(errors) if errors else None, flush=True)
app.close()
