import os
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':1280,'height':720,'renderer':'RayTracedLighting','anti_aliasing':3})
from pathlib import Path
import omni.usd,omni.replicator.core as rep
from pxr import UsdGeom,Gf
from PIL import Image
root=Path(__file__).resolve().parent
from pxr import Usd
s=Usd.Stage.Open(str(root/'auscultation_real2sim.usdc'))
if s.GetPrimAtPath('/Orchestrator'):s.RemovePrim('/Orchestrator')
s.GetRootLayer().Export(str(root/'results/render_clean.usdc'));s=None
omni.usd.get_context().open_stage(str(root/'results/render_clean.usdc'))
for _ in range(45):app.update()
stage=omni.usd.get_context().get_stage()
stage.GetPrimAtPath('/World/ContactFixture').SetActive(False)
cam=rep.create.camera(position=(.22,.1,1.02),look_at=(0,.4535,.83),clipping_range=(.002,100),focal_length=48.)
product=rep.create.render_product(cam,(1280,720));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product])
for _ in range(45):app.update()
print('STATIC_RENDER_READY',flush=True)
rep.orchestrator.step(rt_subframes=8)
Image.fromarray(ann.get_data()[:,:,:3]).save(root/'results/static_mount.png')
print('STATIC_RENDER_COMPLETE',flush=True)
app.close()
