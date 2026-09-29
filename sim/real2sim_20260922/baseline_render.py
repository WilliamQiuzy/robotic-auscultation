import os
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':1280,'height':720,'renderer':'RayTracedLighting','anti_aliasing':3})
import json
from pathlib import Path
import numpy as np
import omni.usd
import omni.replicator.core as rep
from pxr import Usd,UsdGeom,Gf,UsdLux
root=Path(__file__).resolve().parent
out=root/'results';out.mkdir(exist_ok=True)
omni.usd.get_context().open_stage('/home/robot/Desktop/stethoscope/sim/auscultation_scene.usd')
for _ in range(45):app.update()
stage=omni.usd.get_context().get_stage()
cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
for name in ['/World/Table','/World/Patient','/World/StethoArmMount/OpenArmStetho/openarm_left_link7']:
 p=stage.GetPrimAtPath(name)
 print('BASELINE_BOUNDS',name,cache.ComputeWorldBound(p).ComputeAlignedRange(),flush=True)
cam=rep.create.camera(position=(2.1,2.5,1.8),look_at=(0,0.55,1.0),clipping_range=(0.01,100),focal_length=34)
product=rep.create.render_product(cam,(1280,720))
ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product])
for _ in range(45):app.update()
rep.orchestrator.step(rt_subframes=16)
from PIL import Image
Image.fromarray(ann.get_data()[:,:,:3]).save(out/'baseline_overview.png')
robot=stage.GetPrimAtPath('/World/StethoArmMount/OpenArmStetho')
inventory=[]
xf=UsdGeom.XformCache()
for p in Usd.PrimRange(robot):
 if any(t in str(p.GetPath()) for t in ['left_link7','left_link6']):
  d={'path':str(p.GetPath()),'type':p.GetTypeName(),'apis':p.GetAppliedSchemas()}
  if p.IsA(UsdGeom.Xformable):
   d['world']=str(xf.GetLocalToWorldTransform(p))
   d['bounds']=str(cache.ComputeWorldBound(p).ComputeAlignedRange())
  inventory.append(d)
(out/'baseline_wrist.json').write_text(json.dumps(inventory,indent=2))
stage.Flatten().Export(str(out/'baseline_flattened.usdc'))
print('BASELINE_COMPLETE',flush=True)
app.close()
