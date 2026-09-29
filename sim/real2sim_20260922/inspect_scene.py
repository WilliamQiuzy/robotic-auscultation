"""Read-only USD scene geometry/physics inventory, without starting Kit."""
import json
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics

root = Path('/home/robot/Desktop/stethoscope/sim')
stage = Usd.Stage.Open(str(root/'auscultation_scene.usd'))
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
xf = UsdGeom.XformCache()
rows = []
for p in stage.Traverse():
    path = str(p.GetPath())
    if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Joint) or path.count('/') <= 3 or 'StethoModule' in path:
        row = {'path':path, 'type':p.GetTypeName(), 'apis':p.GetAppliedSchemas()}
        if p.IsA(UsdGeom.Xformable):
            row['transform'] = str(xf.GetLocalToWorldTransform(p))
            b=cache.ComputeWorldBound(p).ComputeAlignedRange()
            if not b.IsEmpty(): row['bounds'] = [list(b.GetMin()),list(b.GetMax())]
        if p.IsA(UsdPhysics.Joint):
            row['body0']=[str(x) for x in UsdPhysics.Joint(p).GetBody0Rel().GetTargets()]
            row['body1']=[str(x) for x in UsdPhysics.Joint(p).GetBody1Rel().GetTargets()]
        rows.append(row)
print(json.dumps(rows,indent=2))
