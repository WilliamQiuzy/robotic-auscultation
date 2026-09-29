"""Read-only PhysX ray audit of the prepared static-contact asset."""
import os
import json
from pathlib import Path
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import numpy as np
from pxr import Usd, UsdPhysics
import omni.usd
import omni.timeline
import omni.physx

ROOT = Path(__file__).resolve().parent
stage = Usd.Stage.Open(str(ROOT / 'contact_static.usda'))
probe = Usd.Stage.Open(stage.Flatten())
for prim in probe.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    if prim.IsA(UsdPhysics.Joint):
        UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
temporary = '/tmp/wam_static_collision_probe.usdc'
probe.GetRootLayer().Export(temporary)
omni.usd.get_context().open_stage(temporary)
for _ in range(8): app.update()
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(12): app.update()
meta = json.loads((ROOT / 'asset_manifest.json').read_text())
pred = json.loads((ROOT / 'model_predictions.json').read_text())
T = np.asarray(meta['world_from_camera_optical'])
origin = T[:3, 3]
query = omni.physx.get_physx_scene_query_interface()
rows = []
for point in pred['frames']['raised']['points']:
    surface = np.asarray(point['world_xyz_m'])
    direction = surface-origin
    reference_distance = float(np.linalg.norm(direction))
    direction /= reference_distance
    hit = query.raycast_closest(tuple(origin), tuple(direction), reference_distance+.04)
    found = bool(hit.get('hit', False))
    row = {'name': point['name'], 'hit': found}
    if found:
        row.update(collider=str(hit['collision']), distance_m=float(hit['distance']),
                   rendered_surface_distance_m=reference_distance,
                   distance_error_m=abs(float(hit['distance'])-reference_distance))
    rows.append(row)
timeline.stop()
result = {'test': 'PhysX cooked collision raycasts, frozen simulated bodies; no force/deformation validation',
          'points': rows, 'all_hit_patient': all(r['hit'] and r['collider'].startswith('/World/Patient/') for r in rows),
          'max_distance_error_m': max((r.get('distance_error_m', 0.) for r in rows), default=None)}
(ROOT / 'physx_collision_probe.json').write_text(json.dumps(result, indent=2))
assert result['all_hit_patient'], result
assert result['max_distance_error_m'] < .005, result
print('PHYSX_RAYCAST_AUDIT_PASS', result['max_distance_error_m'], flush=True)
app.close()
