"""Render sampled PhysX poses with physics disabled; never synthesize compression."""
import os,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');a=p.parse_args()
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':1280,'height':720,'renderer':'RayTracedLighting','anti_aliasing':3,'multi_gpu':False})
import csv,json,math,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pxr import Usd,UsdGeom,UsdPhysics,Gf
import omni.usd,omni.replicator.core as rep
import build_scene as bs
root=Path(__file__).resolve().parent;out=root/'results'
rows=list(csv.DictReader((out/'contact_trace.csv').open()))
for row in rows:
 for key in row:
  if key not in ('phase','probe'):row[key]=float(row[key])
record=np.load(out/'physics_replay.npz');poses=record['poses'];indices=record['row_indices'];paths=record['paths']
# Remove the capture graph saved in the original stage. A stale graph blocks new RGB captures.
s=Usd.Stage.Open(str(root/'auscultation_real2sim.usdc'))
for path in ['/Render','/Replicator','/Orchestrator']:
 if s.GetPrimAtPath(path):s.RemovePrim(path)
for prim in s.Traverse():
 if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
 if prim.IsA(UsdPhysics.Joint):UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
s.GetRootLayer().Export(str(out/'replay_render.usdc'));s=None
omni.usd.get_context().open_stage(str(out/'replay_render.usdc'))
for _ in range(45):app.update()
stage=omni.usd.get_context().get_stage()
# World poses are converted to local coordinates once relative to each static parent.
cache=UsdGeom.XformCache();ops=[]
for path in paths:
 prim=stage.GetPrimAtPath(str(path));parent_inv=cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
 xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();op=xf.AddTransformOp(opSuffix='replay')
 ops.append((op,parent_inv))
cam=rep.create.camera(position=(2.1,2.5,1.8),look_at=(0,.55,1.),clipping_range=(.001,100),focal_length=34.)
camroot=stage.GetPrimAtPath(str(cam.get_output_prims()['prims'][0].GetPath()))
cp=next(p for p in Usd.PrimRange(camroot) if p.IsA(UsdGeom.Camera))
# Reset the parent created by Replicator so the authored camera matrix is a world transform.
cr=UsdGeom.Xformable(camroot);cr.ClearXformOpOrder()
xf=UsdGeom.Xformable(cp);xf.ClearXformOpOrder();camop=xf.AddTransformOp()
product=rep.create.render_product(cam,(1280,720));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product])
for _ in range(30):app.update()
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
f32=ImageFont.truetype(font,30);f22=ImageFont.truetype(font,22);f18=ImageFont.truetype(font,17)
encoder=None;frame_count=0;manifest=[]
if not a.preview:
 import imageio_ffmpeg
 encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','15','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'real2sim_demo.mp4')],stdin=subprocess.PIPE)
def set_camera(eye,target,focal=38):
 camop.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
 UsdGeom.Camera(cp).GetFocalLengthAttr().Set(focal)
def apply(k,probe=True):
 row=rows[int(indices[k])]
 for (op,inv),pose in zip(ops,poses[k]):
  m=Gf.Matrix4d(1.);m.SetRotate(Gf.Quatd(float(pose[3]),Gf.Vec3d(*[float(v) for v in pose[4:]])));m.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in pose[:3]]));op.Set(m*inv)
 for name,kind in [('CentralProbe','central'),('BroadPad','broad'),('FrameProbe','frame')]:
  show=probe and (row['probe']==kind or (name=='FrameProbe' and row['probe']=='post'))
  UsdGeom.Imageable(stage.GetPrimAtPath('/World/ContactFixture/'+name)).GetVisibilityAttr().Set('inherited' if show else 'invisible')
 bs.springs(stage,row['compression_mm']/1000.)
 return row

def render(k,title,subtitle,metrics=False,probe=False,save=None):
 global frame_count
 row=apply(k,probe)
 rep.orchestrator.step(rt_subframes=3,delta_time=0.)
 frame=ann.get_data()
 if frame.ndim!=3:raise RuntimeError('No RGB frame from Isaac')
 im=Image.fromarray(frame[:,:,:3]);d=ImageDraw.Draw(im,'RGBA')
 d.rectangle((0,0,1280,93),fill=(13,21,31,240));d.text((25,15),title,font=f32,fill='white');d.text((27,57),subtitle,font=f18,fill=(166,199,218))
 if metrics:
  d.rounded_rectangle((20,491,414,673),radius=13,fill=(13,21,31,239))
  d.text((38,506),f"Compression  {row['compression_mm']:.2f} mm",font=f22,fill='white')
  d.rounded_rectangle((39,542,388,550),radius=4,fill=(63,75,86))
  width=max(0,min(349,349*row['compression_mm']/7.25));d.rounded_rectangle((39,542,39+width,550),radius=4,fill=(69,211,204))
  d.text((38,563),f"Contact force  {row['total_contact_N']:.2f} N",font=f22,fill=(108,228,226))
  d.text((38,601),f"Diaphragm {row['diaphragm_N']:.2f}  |  Clamp {row['clamp_N']:.2f} N",font=f18,fill='white')
  d.text((38,632),f"Frame {row['frame_N']:.2f}  |  Guide {row['post_N']:.2f} N",font=f18,fill='white')
 d.rectangle((0,686,1280,720),fill=(13,21,31,236))
 d.text((25,693),'Isaac Sim 5.1 / PhysX   |   recorded physics poses   |   spring and adapter parameters estimated',font=f18,fill=(174,195,205))
 if save:im.save(out/save)
 if encoder:encoder.stdin.write(np.ascontiguousarray(np.asarray(im)).tobytes())
 manifest.append({'video_frame':frame_count,'trace_row':int(indices[k]),'phase':row['phase']});frame_count+=1

def matching(prefix):return [i for i,j in enumerate(indices) if rows[int(j)]['phase'].startswith(prefix)]
rest=matching('settle')[-1]
fixed_index=list(paths).index(bs.FIXED);center=poses[rest,fixed_index,:3]+np.array([0,0,-.023])
# Overview from the patient's front side shows the complete table, stool and two-arm assembly.
for f in range(1 if a.preview else 60):
 t=f/60.;set_camera((3.1-.2*t,3.8,2.15),(0,.65,.85),24.)
 render(rest,'REAL2SIM / OpenArm auscultation scene','Table, seated patient and mounted passive stethoscope module',save='overview.png' if f==0 else None)
print('OVERVIEW_DONE',flush=True)
for f in range(1 if a.preview else 75):
 t=f/75.;angle=-1.03+.4*t
 set_camera(center+np.array([.40*math.cos(angle),.40*math.sin(angle),.13]),center+np.array([0,0,.018]),43.)
 render(rest,'01 / Assembly and scale','46 mm device  /  88 mm fixed plate  /  21 x 14 mm mounting pattern',save='mount_closeup.png' if f==0 else None)
print('ASSEMBLY_DONE',flush=True)
segments=[('broad_','02 / Broad surface contact','The retaining lip contacts first; the diaphragm is recessed by 0.8 mm',1),('central_','03 / Diaphragm contact','Press, hold and release: carriage motion is solved by PhysX',1),('central_stop_','04 / Travel stop','7.25 mm travel limit; test actuator capped at 20 N, not a clinical force setting',1),('frame_stop_','05 / Exterior frame collision','Offset probe tests the rigid frame; normal-speed physics, 2x video playback',2),('post_stop_','06 / Guide-head collision','Offset probe tests the fixed guide head; 2x video playback',2)]
for prefix,title,subtitle,speed in segments:
 selected=[k for k in matching(prefix) if (prefix!='central_' or not rows[int(indices[k])]['phase'].startswith('central_stop')) and (prefix!='broad_' or not rows[int(indices[k])]['phase'].startswith('broad_stop'))]
 if a.preview:selected=[next(k for k in selected if rows[int(indices[k])]['phase'].endswith('hold'))]
 else:selected=selected[::2*speed]
 for f,k in enumerate(selected):
  # Side view leaves clear space for the readout on the left.
  set_camera(center+np.array([.20,-.37,.055]),center+np.array([-.018,0,-.003]),46.)
  render(k,title,subtitle,metrics=True,probe=True,save=prefix+'hold.png' if rows[int(indices[k])]['phase'].endswith('hold') and (a.preview or f==60) else None)
 print('SEGMENT_DONE',prefix,flush=True)
if encoder:
 encoder.stdin.close()
 if encoder.wait()!=0:raise RuntimeError('ffmpeg failed')
(out/('preview_frames.json' if a.preview else 'video_frames.json')).write_text(json.dumps(manifest))
print('RENDER_COMPLETE',frame_count,flush=True)
app.close()
