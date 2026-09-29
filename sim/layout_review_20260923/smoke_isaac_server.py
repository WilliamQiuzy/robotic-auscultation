"""Verify headless RGB and metric depth on a real RTX render product."""
import json
import os
from pathlib import Path
import time

os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runtime_smoke'
OUT.mkdir(exist_ok=False)
began=time.time()
app=SimulationApp({'headless':True,'width':640,'height':360,
                   'renderer':'RayTracedLighting','multi_gpu':False})
try:
    import numpy as np
    from PIL import Image
    from pxr import UsdGeom,UsdLux,Gf
    import omni.usd
    import omni.replicator.core as rep
    import omni.timeline

    context=omni.usd.get_context()
    context.new_stage()
    stage=context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,'Z')
    cube=UsdGeom.Cube.Define(stage,'/World/Box')
    cube.CreateSizeAttr(.5)
    cube.CreateDisplayColorAttr([(0.12,0.45,0.8)])
    camera=UsdGeom.Camera.Define(stage,'/World/Camera')
    camera.CreateFocalLengthAttr(24.)
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateVerticalApertureAttr(36.*360/640)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100.))
    UsdGeom.Xformable(camera).AddTranslateOp().Set(Gf.Vec3d(0.,0.,2.))
    light=UsdLux.DomeLight.Define(stage,'/World/Light')
    light.CreateIntensityAttr(1500.)
    rep.orchestrator.set_capture_on_play(False)
    product=rep.create.render_product('/World/Camera',(640,360))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb')
    depth=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
    rgb.attach([product]);depth.attach([product])
    for _ in range(30):app.update()
    rep.orchestrator.step(rt_subframes=4,delta_time=0.,pause_timeline=True)
    color=np.asarray(rgb.get_data())[:,:,:3].copy()
    z=np.asarray(depth.get_data()).copy()
    assert color.shape==(360,640,3) and np.std(color)>3
    assert z.shape==(360,640) and np.isfinite(z[180,320])
    assert abs(float(z[180,320])-1.75)<.01, float(z[180,320])
    assert not omni.timeline.get_timeline_interface().is_playing()
    Image.fromarray(color).save(OUT/'rgb.png')
    np.save(OUT/'depth_m.npy',z)
    (OUT/'result.json').write_text(json.dumps({'accepted':True,'elapsed_seconds':time.time()-began,
        'scope':'Headless RTX RGB and depth test only; no task contact dynamics validation.',
        'resolution':[640,360],'center_depth_m':float(z[180,320]),'expected_center_depth_m':1.75,
        'rgb_std':float(np.std(color))},indent=2)+'\n')
    print('ISAAC_RGBD_SMOKE_COMPLETE',flush=True)
finally:
    app.close()
