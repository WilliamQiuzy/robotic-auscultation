"""Inspect a disposable physical scene; never imports the real robot runtime."""
import os
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'multi_gpu': False})
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
import omni.usd
import carb
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
ROBOT = '/World/StethoArmMount/OpenArmStetho'
s = Usd.Stage.Open(str(ROOT.parent / 'auscultation_assets_20260922/contact_static.usda'))
s = Usd.Stage.Open(s.Flatten())
for path in ['/Render', '/Replicator', '/Orchestrator', '/World/ContactFixture', '/World/ContactFixtureGuide']:
    if s.GetPrimAtPath(path): s.RemovePrim(path)
cache = UsdGeom.XformCache()
report = {'roots': [], 'joints': [], 'bodies': []}
for p in s.Traverse():
    if p.HasAPI(UsdPhysics.ArticulationRootAPI): report['roots'].append(str(p.GetPath()))
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        report['bodies'].append({'path':str(p.GetPath()), 'world_from_body':np.array(cache.GetLocalToWorldTransform(p)).T.tolist()})
    if p.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(p)
        report['joints'].append({'path':str(p.GetPath()), 'type':p.GetTypeName(),
            'body0':[str(x) for x in j.GetBody0Rel().GetTargets()], 'body1':[str(x) for x in j.GetBody1Rel().GetTargets()],
            'pos0':list(j.GetLocalPos0Attr().Get()), 'pos1':list(j.GetLocalPos1Attr().Get())})
report['world_from_left_base'] = np.array(cache.GetLocalToWorldTransform(s.GetPrimAtPath(ROBOT+'/openarm_left_link0'))).T.tolist()
s.GetRootLayer().Export(str(OUT/'runtime_inspection.usdc'))
omni.usd.get_context().open_stage(str(OUT/'runtime_inspection.usdc'))
for _ in range(8): app.update()
carb.settings.get_settings().set('/physics/updateToUsd', True)
world = World(physics_dt=1/240, rendering_dt=1/30, stage_units_in_meters=1., physics_prim_path='/World/PhysicsScene')
world.reset(); world.play()
robot = SingleArticulation(ROBOT, name='inspection_robot')
robot.initialize()
report['dof_names'] = robot.dof_names
report['initial_q'] = robot.get_joint_positions().tolist()
paths = [ROBOT+'/openarm_left_link0', ROBOT+'/openarm_left_link7', '/World/Stethoscope/Fixed', '/World/Stethoscope/Carriage']
bodies = [SingleRigidPrim(p, name=f'body_{i}') for i,p in enumerate(paths)]
for b in bodies: b.initialize()
def states():
    return {p: [np.asarray(x).tolist() for x in b.get_world_pose()] for p,b in zip(paths,bodies)}
report['initial_poses'] = states()
for _ in range(240): world.step(render=False)
report['settled_q'] = robot.get_joint_positions().tolist()
report['settled_poses'] = states()
(OUT/'runtime_inspection.json').write_text(json.dumps(report, indent=2))
print('INSPECT_DONE', json.dumps({k:report[k] for k in ['roots','dof_names','initial_q','settled_q','initial_poses','settled_poses']}), flush=True)
app.close()
