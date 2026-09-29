import os
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
from pxr import Usd, UsdGeom, UsdPhysics
from pathlib import Path
import json
root=Path(__file__).resolve().parent
s=Usd.Stage.Open(str(root.parent/'auscultation_assets_20260922/contact_static.usda'))
rows=[]
for p in Usd.PrimRange.Stage(s,Usd.TraverseInstanceProxies()):
    path=str(p.GetPath())
    if '/collisions/' in path or 'Table' in path or 'table' in path:
        row={'path':path,'type':str(p.GetTypeName()),'instance':p.IsInstanceProxy(),'schemas':list(p.GetAppliedSchemas())}
        if p.IsA(UsdGeom.Mesh): row['vertices']=len(UsdGeom.Mesh(p).GetPointsAttr().Get())
        rows.append(row)
(root/'results/geometry_structure.json').write_text(json.dumps(rows,indent=2))
print('STRUCTURE_EXPORTED',len(rows),flush=True)
app.close()
