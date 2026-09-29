"""Build, simulate and record the actual Isaac/PhysX scene (no robot hardware access)."""
import argparse,os,faulthandler,signal
faulthandler.register(signal.SIGUSR1,all_threads=False)
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--smoke',action='store_true')
parser.add_argument('--build-only',action='store_true')
parser.add_argument('--reuse-scene',action='store_true')
parser.add_argument('--physics-only',action='store_true')
args=parser.parse_args()
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':1280,'height':720,'renderer':'RayTracedLighting','anti_aliasing':3,'multi_gpu':False})
import csv,json,math,subprocess
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import carb,omni.usd,omni.replicator.core as rep
from omni.physx import get_physx_simulation_interface
from pxr import Gf,Usd,UsdGeom,UsdPhysics,PhysicsSchemaTools
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
import build_scene as bs

root=Path(__file__).resolve().parent;out=root/'results';out.mkdir(exist_ok=True)
stagepath=root/'auscultation_real2sim.usdc'
if not args.reuse_scene:
    # Never write through to the original stage or its referenced layers.
    from pxr import Sdf
    Sdf.Layer.FindOrOpen(str(out/'baseline_flattened.usdc')).Export(str(stagepath))
    clean=Usd.Stage.Open(str(stagepath))
    for p in ['/Render','/Replicator','/Orchestrator']:
        if clean.GetPrimAtPath(p):clean.RemovePrim(p)
    clean.GetRootLayer().Save();clean=None
omni.usd.get_context().open_stage(str(stagepath))
for _ in range(8):app.update()
stage=omni.usd.get_context().get_stage()
if not args.reuse_scene:meta=bs.build(stage,root)
else:meta=json.loads((out/'build_metadata.json').read_text())
stage.GetRootLayer().Save()
print('BUILD_COMPLETE',flush=True)
if args.build_only:
    app.close();raise SystemExit
carb.settings.get_settings().set('/physics/updateToUsd',True)
carb.settings.get_settings().set('/physics/updateVelocitiesToUsd',True)
world=World(physics_dt=1/240,rendering_dt=1/30,stage_units_in_meters=1.,physics_prim_path='/World/PhysicsScene')
world.reset()
world.play()
print("PHYSICS_PLAYING",world.is_playing(),flush=True)
fixed=SingleRigidPrim(bs.FIXED,name='fixed_frame');fixed.initialize()
moving=SingleRigidPrim(bs.MOVING,name='passive_carriage');moving.initialize()
wrist=SingleRigidPrim(bs.WRIST,name='arm_wrist');wrist.initialize()
fixture=SingleRigidPrim('/World/ContactFixture',name='fixture');fixture.initialize()


def pose_matrix(body):
    p,q=body.get_world_pose();m=Gf.Matrix4d(1.)
    m.SetRotate(Gf.Quatd(float(q[0]),Gf.Vec3d(*[float(x) for x in q[1:]])));m.SetTranslateOnly(Gf.Vec3d(*[float(x) for x in p]));return m

contacts=defaultdict(float);separations=[];contact_pairs=set();carriage_impulse=np.zeros(3)
def callback(headers,data):
    for h in headers:
        a,b=str(PhysicsSchemaTools.intToSdfPath(h.collider0)),str(PhysicsSchemaTools.intToSdfPath(h.collider1))
        module=a if '/Stethoscope/' in a else b if '/Stethoscope/' in b else ''
        if not module or '/ContactFixture/' not in a+b:continue
        category=('diaphragm' if '/Diaphragm' in module else 'post' if 'PostHead' in module or 'GuidePost' in module else 'frame' if '/Fixed/' in module else 'clamp')
        contact_pairs.add((a,b))
        for idx in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
            impulse=data[idx].impulse
            v=np.array([impulse.x,impulse.y,impulse.z],dtype=float)
            contacts[category]+=float(np.linalg.norm(v))*240.
            if '/Carriage/' in module:
                carriage_impulse[:]+=v*(1. if '/Carriage/' in a else -1.)
            separations.append(float(data[idx].separation))
sub=get_physx_simulation_interface().subscribe_contact_report_events(callback)

if not args.physics_only:
    cam=rep.create.camera(position=(1.8,-2,1.6),look_at=(0,.5,1.05),clipping_range=(.002,100),focal_length=35.)
    product=rep.create.render_product(cam,(1280,720));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product])
    camroot=stage.GetPrimAtPath(str(cam.get_output_prims()['prims'][0].GetPath()))
    camprim=next(p for p in Usd.PrimRange(camroot) if p.IsA(UsdGeom.Camera))
    def camera(eye,target,focal=35.):
        matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
        x=UsdGeom.Xformable(camprim);x.ClearXformOpOrder();x.AddTransformOp().Set(matrix)
        UsdGeom.Camera(camprim).GetFocalLengthAttr().Set(focal)
else:
    def camera(*args,**kwargs):pass

initial=Gf.Matrix4d(meta['fixed_world_matrix'])
normal=np.array(initial.TransformDir(Gf.Vec3d(0,0,1)))
start=np.array(initial.Transform(Gf.Vec3d(0,0,bs.CONTACT_Z)))
orient=initial.ExtractRotationQuat();orientation=np.array([orient.GetReal(),*orient.GetImaginary()])
local=Gf.Matrix4d(1.);local.SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0),180));local.SetTranslateOnly(Gf.Vec3d(0,0,bs.MOUNT_Z))
probe_kind=None
def set_probe(kind,distance):
    global probe_kind
    offset=initial.TransformDir(Gf.Vec3d(.037,-.015,0) if kind=='frame' else Gf.Vec3d(.034,0,0) if kind=='post' else Gf.Vec3d(0))
    if kind!=probe_kind:
        for name in ['CentralProbe','BroadPad','FrameProbe']:
            p=stage.GetPrimAtPath('/World/ContactFixture/'+name);enabled=(kind=='central' and name=='CentralProbe') or (kind=='broad' and name=='BroadPad') or (kind in ('frame','post') and name=='FrameProbe')
            UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(enabled)
            UsdGeom.Imageable(p).GetVisibilityAttr().Set('inherited' if enabled else 'invisible')
        probe_kind=kind
        world.pause()
        UsdPhysics.PrismaticJoint(stage.GetPrimAtPath('/World/ContactFixtureGuide')).GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in start+np.array(offset)]))
        fixture.set_world_pose(start+np.array(offset)-normal*.014,orientation)
        fixture.set_linear_velocity(np.zeros(3));fixture.set_angular_velocity(np.zeros(3))
        app.update();world.play()
    offset=initial.TransformDir(Gf.Vec3d(.037,-.015,0) if kind=='frame' else Gf.Vec3d(.034,0,0) if kind=='post' else Gf.Vec3d(0))
    position=start+normal*distance+np.array(offset)
    current,q=fixture.get_world_pose()
    force=normal*(2000.*float(np.dot(position-current,normal))-30.*float(np.dot(fixture.get_linear_velocity(),normal)))
    norm=float(np.linalg.norm(force))
    if norm>20.:force*=20./norm
    # Force-controlled guided press; the physical contacts stop its motion.
    fixture._rigid_prim_view.apply_forces(forces=np.asarray([force],dtype=np.float32),is_global=True)

font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(font_path,23);small=ImageFont.truetype(font_path,18);titlefont=ImageFont.truetype(font_path,31)
def overlay(frame,title,subtitle,values=None):
    im=Image.fromarray(frame[:,:,:3]);d=ImageDraw.Draw(im,'RGBA')
    d.rectangle((0,0,1280,102),fill=(12,21,32,237))
    d.text((28,18),'REAL2SIM  /  '+title,font=titlefont,fill='white')
    d.text((28,64),subtitle,font=small,fill=(178,204,222))
    if values:
        d.rounded_rectangle((25,532,590,698),radius=12,fill=(12,21,32,237))
        d.text((45,549),f'Compression: {values["compression_mm"]:5.2f} / 7.25 mm',font=font,fill='white')
        d.text((45,586),f'PhysX contact: {values["total_contact_N"]:5.2f} N',font=font,fill=(105,220,233))
        d.text((45,623),f'Diaphragm {values["diaphragm_N"]:.2f} N  |  Clamp {values["clamp_N"]:.2f} N',font=small,fill='white')
        d.text((45,658),f'Frame / post: {values["frame_N"]+values["post_N"]:.2f} N',font=small,fill=(195,200,210))
    else:
        d.rectangle((0,678,1280,720),fill=(12,21,32,220))
        d.text((28,688),'46 mm device  |  88 mm plate  |  21 x 14 mm bolt pattern  |  SI metres',font=small,fill='white')
    return np.asarray(im)

encoder=None
if not args.smoke and not args.physics_only:
    import imageio_ffmpeg
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    encoder=subprocess.Popen([exe,'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'real2sim_demo.mp4')],stdin=subprocess.PIPE)

rows=[];clock=0.;last=None;frameidx=0;pose_frames=[];pose_indices=[]
pose_bodies=[]
for p in stage.Traverse():
    path=str(p.GetPath())
    if p.HasAPI(UsdPhysics.RigidBodyAPI) and (path.startswith(bs.ROBOT) or path in [bs.FIXED,bs.MOVING,'/World/ContactFixture']):
        body=SingleRigidPrim(path,name='record_'+str(len(pose_bodies)));body.initialize();pose_bodies.append((path,body))
def step(phase,kind,distance):
    global clock,last
    set_probe(kind,distance);contacts.clear();separations.clear();carriage_impulse[:]=0.
    world.step(render=False);clock+=1/240
    fm=pose_matrix(fixed);mm=pose_matrix(moving);wm=pose_matrix(wrist)
    axis=np.array(fm.TransformDir(Gf.Vec3d(0,0,1)));fp=np.array(fm.ExtractTranslation());mp=np.array(mm.ExtractTranslation())
    q=float(np.dot(mp-fp,axis));expected=(local*wm).ExtractTranslation()
    last={'time_s':clock,'phase':phase,'probe':kind,'command_m':distance,'compression_mm':q*1000,
          'closure_mm':float(np.linalg.norm(fp-np.array(expected)))*1000,
          'spring_inferred_N':1.5+1000*q,'gravity_axis_N':-meta['moving_mass_kg']*9.81*float(axis[2]),
          'diaphragm_N':contacts['diaphragm'],'clamp_N':contacts['clamp'],'frame_N':contacts['frame'],'post_N':contacts['post'],
          'carriage_axial_N':float(np.dot(carriage_impulse*240.,axis)),
          'carriage_force_x_N':carriage_impulse[0]*240.,'carriage_force_y_N':carriage_impulse[1]*240.,'carriage_force_z_N':carriage_impulse[2]*240.,
          'total_contact_N':sum(contacts.values()),'min_separation_m':min(separations) if separations else 0.}
    rows.append(last)
    if len(rows)%8==0:
        pose_frames.append([np.r_[body.get_world_pose()[0],body.get_world_pose()[1]] for path,body in pose_bodies]);pose_indices.append(len(rows)-1)
    return q

def render(title,subtitle,values=None,save=None):
    global frameidx
    if args.physics_only:return
    world.pause()
    bs.springs(stage,max(0,min(bs.TRAVEL,(last or {}).get('compression_mm',0)/1000)))
    rep.orchestrator.step(rt_subframes=2)
    frame=ann.get_data()
    if frame is None or not frame.size:raise RuntimeError('Empty rendered RGB frame')
    rgb=overlay(frame,title,subtitle,values)
    if save:Image.fromarray(rgb).save(out/save)
    if encoder:encoder.stdin.write(np.ascontiguousarray(rgb).tobytes())
    frameidx+=1
    world.play()

for _ in range(240):step('settle','none',-.014)
print('SETTLED',json.dumps(last),'PHYSICS_TIME',world.current_time,flush=True)
camera((1.55,-1.5,1.65),(0,.54,1.07),31.)
print('OVERVIEW_RENDER_START',flush=True)
render('Auscultation scene','OpenArm + seated patient + physically articulated stethoscope',save='overview.png')
print('OVERVIEW_RENDER_DONE',flush=True)
if not args.smoke:
    for f in range(90):
        theta=-.8+.35*f/90;camera((1.8*math.cos(theta),.35+1.8*math.sin(theta),1.55),(0,.52,1.02),30.)
        for _ in range(8):step('overview','none',-.014)
        render('Auscultation scene','Original environment retained; left gripper replaced with passive module')
center=np.array(pose_matrix(fixed).Transform(Gf.Vec3d(0,0,-.022)))
camera(center+np.array([.20,-.31,.125]),center,48.)
render('Mounted module','Black printed frame, independent carriage, guide posts, springs and wiring',save='mount_closeup.png')
if not args.smoke:
    for f in range(90):
        angle=-1.1+.60*f/90;camera(center+np.array([.35*math.cos(angle),.35*math.sin(angle),.12]),center,48.)
        for _ in range(8):step('closeup','none',-.014)
        render('Mounting and scale','Wrist adapter estimate is documented; meshes exported directly in metres')

def cycle(kind,peak,title,hard=False):
    # Slow prescribed test-surface motion; all carriage displacement is solved by PhysX.
    schedule=[('approach',-.006,1.),('contact',peak,3.),('hold',peak,1.),('release',-.012,3.),('return',-.012,1.)]
    previous=-.012
    for phase,target,duration in schedule:
        frames=round(duration*30)
        for f in range(frames):
            t=(f+1)/frames;s=t*t*(3-2*t);cmd=previous+(target-previous)*s
            for _ in range(8):step(kind+('_stop' if hard else '')+'_'+phase,kind,cmd)
            if not args.smoke or f in (frames-1,):
                caption=('Rigid stop limits travel, NOT external force' if hard else 'Contact impulses measured independently from the spring-displacement model')
                render(title,caption,last,save=(kind+'_'+phase+'.png') if f==frames-1 else None)
        previous=target
    print('CYCLE_COMPLETE',kind,flush=True)

camera(center+np.array([.18,-.29,.06]),center+np.array([0,0,-.014]),55.)
cycle('broad',.0040,'Broad surface / clamp-first contact')
cycle('central',.0050,'Central probe / diaphragm contact')
cycle('central',.022,'Travel stop / overload demonstration',hard=True)
cycle('broad',.022,'Broad pad / fixed guide-head contact',hard=True)
cycle('frame',.049,'Offset probe / frame collision',hard=True)
cycle('post',.016,'Offset probe / guide-head collision',hard=True)
if encoder:
    encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError('Video encoding failed')
np.savez_compressed(out/'physics_replay.npz',poses=np.asarray(pose_frames),row_indices=np.asarray(pose_indices),paths=np.asarray([p for p,b in pose_bodies]))
with (out/'contact_trace.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
validation={'frames':frameidx,'physics_samples':len(rows),'minimum_compression_mm':min(r['compression_mm'] for r in rows),
            'maximum_compression_mm':max(r['compression_mm'] for r in rows),'maximum_wrist_closure_mm':max(r['closure_mm'] for r in rows),
            'maximum_contact_N':max(r['total_contact_N'] for r in rows),'released_compression_mm':last['compression_mm'],
            'diaphragm_contact_observed':any(r['diaphragm_N']>.1 for r in rows),
            'clamp_contact_observed':any(r['clamp_N']>.1 for r in rows),'contact_pairs':sorted(contact_pairs)}
(out/'validation.json').write_text(json.dumps(validation,indent=2))
print('DEMO_COMPLETE',json.dumps(validation),flush=True)
world.stop();app.close()
