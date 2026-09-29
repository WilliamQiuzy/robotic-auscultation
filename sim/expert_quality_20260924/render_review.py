"""Render blind 1x review clips from actual recorded body poses on Linux Isaac.

No policy rerun, no interpolation, no invented force, no scores burned into video.
The user-approved RGB-D viewpoint and a separate full-arm report camera are used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CAMERA_CONFIG = PROJECT/'sim/camera_framing_20260923/recommended_camera.json'
FIELDS = ('time_s','target','phase','subphase','hall_N','lip_N','diaphragm_N','contact_rim_N','hold_elapsed_s')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def schedule(times, pose_rows, start, end, fps=15):
    if fps not in (10,15,30): raise ValueError('unsupported frame rate')
    if not 0 <= start < end <= len(times): raise ValueError('invalid review interval')
    if not np.allclose(np.diff(times),1/240,atol=1e-7,rtol=0): raise ValueError('trace time gap')
    if len(pose_rows)<2 or np.any(np.diff(pose_rows)<=0) or pose_rows[0]<0 or pose_rows[-1]>=len(times): raise ValueError('invalid pose indices')
    if np.max(np.diff(times[pose_rows]))>1/30+1e-7: raise ValueError('pose time gap')
    count=int(np.floor((end-start)/240*fps+1e-7))
    if count<1: raise ValueError('interval too short')
    # End-of-frame sampling; clip duration equals this many 1/fps intervals.
    origin=float(times[start]-1/240)
    frame_times=origin+np.arange(1,count+1)/fps
    rows=np.searchsorted(times,frame_times+1e-8,side='right')-1
    poses=np.searchsorted(pose_rows,rows,side='right')-1
    if np.any(poses<0) or np.any(rows<start) or np.any(rows>=end): raise ValueError('frame has no preceding pose')
    age=times[rows]-times[pose_rows[poses]]
    if np.any(age<0) or np.max(age)>1/30+1e-7: raise ValueError('stale pose')
    return rows,poses,frame_times


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--points',nargs='+',default=['R1','R3','R5']); parser.add_argument('--fps',type=int,default=15)
    parser.add_argument('--views',nargs='+',choices=['posture','robotcam'],default=['posture','robotcam'])
    args=parser.parse_args(); episode=args.episode.resolve(); out=args.output.resolve()
    if out==episode or episode in out.parents: raise ValueError('output must be outside immutable episode')
    if len(set(args.points))!=len(args.points) or len(set(args.views))!=len(args.views): raise ValueError('duplicate selection')
    source=episode/'training_scene.usdc'; outcome=json.loads((episode/'outcome.json').read_text())
    if sha(source)!=outcome.get('training_scene_sha256'): raise ValueError('scene provenance mismatch')
    inputs={str(p.relative_to(PROJECT)):sha(p) for p in [source,episode/'outcome.json',episode/'expert_replay.npz',episode/'expert_trace.jsonl',CAMERA_CONFIG,Path(__file__)]}
    rows=[]
    with (episode/'expert_trace.jsonl').open() as f:
        for line in f:
            r=json.loads(line); rows.append({k:r.get(k) for k in FIELDS})
    times=np.asarray([r['time_s'] for r in rows],float)
    if not np.isfinite(times).all(): raise ValueError('nonfinite time')
    record=np.load(episode/'expert_replay.npz',allow_pickle=False); poses=record['poses']; indices=record['row_indices']; paths=record['paths']
    if poses.shape!=(len(indices),len(paths),7) or not np.isfinite(poses).all(): raise ValueError('malformed poses')
    points={x['target']:x for x in outcome.get('points',[])}
    clips=[]
    for name in args.points:
        if name not in points: raise ValueError('requested point absent: '+name)
        p=points[name]; start,end=p['start_physics_row'],p['end_physics_row']
        if any(r['target']!=name for r in rows[start:end]): raise ValueError('point interval identity mismatch')
        rr,kk,tt=schedule(times,indices,start,end,args.fps)
        clips.append((name,rr,kk,tt))
    config=json.loads(CAMERA_CONFIG.read_text())
    assert config['downward_pitch_delta_deg']==7 and config['camera_height_delta_m']==.05
    out.mkdir(parents=True,exist_ok=False)
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'; sys.argv=[sys.argv[0]]
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':1280,'height':720,'renderer':'RayTracedLighting','multi_gpu':False})
    encoders=[]; attached=[]
    try:
        from pxr import Usd,UsdGeom,UsdPhysics,Gf
        from PIL import Image
        import omni.usd,omni.timeline
        import omni.replicator.core as rep
        import imageio_ffmpeg
        stage=Usd.Stage.Open(Usd.Stage.Open(str(source)).Flatten())
        for path in ('/Render','/Replicator','/Orchestrator'):
            if stage.GetPrimAtPath(path):stage.RemovePrim(path)
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            if p.IsA(UsdPhysics.Joint):UsdPhysics.Joint(p).CreateJointEnabledAttr(False)
        stage.GetRootLayer().Export(str(out/'disposable_render.usdc'))
        omni.usd.get_context().open_stage(str(out/'disposable_render.usdc'))
        timeline=omni.timeline.get_timeline_interface();timeline.stop()
        for _ in range(35):app.update()
        stage=omni.usd.get_context().get_stage();cache=UsdGeom.XformCache();transforms=[]
        for path in paths:
            prim=stage.GetPrimAtPath(str(path));assert prim
            parent=prim.GetParent()
            while parent and not parent.IsPseudoRoot():
                if str(parent.GetPath()) in set(map(str,paths)):raise ValueError('dynamic parent in replay')
                parent=parent.GetParent()
            inv=cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();transforms.append((xf.AddTransformOp(opSuffix='reviewReplay'),inv))
        camera_world=np.asarray(config['world_from_camera_usd'])
        for path in ('/World/RobotCam','/World/RobotCamBody'):
            prim=stage.GetPrimAtPath(path);inv=cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();xf.AddTransformOp(opSuffix='approved7deg').Set(Gf.Matrix4d(camera_world.T)*inv)
        reportcam=UsdGeom.Camera.Define(stage,'/World/QualityReviewCamera')
        reportcam.CreateClippingRangeAttr(Gf.Vec2f(.001,100.));reportcam.CreateHorizontalApertureAttr(20.955);reportcam.CreateFocalLengthAttr(22.)
        xf=UsdGeom.Xformable(reportcam);xf.ClearXformOpOrder();xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.45,-1.,1.90),Gf.Vec3d(-.12,.50,1.24),Gf.Vec3d(0,0,1)).GetInverse())
        products={}
        for view in args.views:
            path='/World/RobotCam' if view=='robotcam' else '/World/QualityReviewCamera'
            product=rep.create.render_product(path,(1280,720));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product]);attached.append((ann,product));products[view]=(ann,path)
        rep.orchestrator.set_capture_on_play(False)
        for _ in range(15):app.update()
        manifest={'schema':'blind_recorded_review_v1','source_sha256':inputs,'camera_config':str(CAMERA_CONFIG.relative_to(PROJECT)),'playback_speed':1.,'resolution':[1280,720],'fps':args.fps,'source_physics_dt_s':1/240,'source_pose_dt_s':1/30,'no_interpolation':True,'no_acceptance_or_model_scores_in_pixels':True,'camera_view_scope':'Approved7deg RGB view downsampled to1280x720; separate whole-arm report view. RGB only review, not a synchronized RGB-D training export.','source_outcome_accepted':outcome.get('training_accepted') is True,'clips':[]}
        for name,rr,kk,tt in clips:
            print('REVIEW_CLIP_START',name,'frames',len(rr),flush=True)
            active={}
            for view in args.views:
                path=out/f'{name}_{view}_1x.mp4'
                proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(args.fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
                encoders.append(proc);active[view]=(proc,path)
            fm=[]
            for f,(r,k,t) in enumerate(zip(rr,kk,tt)):
                for (op,inv),pose in zip(transforms,poses[k]):
                    mat=Gf.Matrix4d(1.).SetRotate(Gf.Quatd(float(pose[3]),Gf.Vec3d(*map(float,pose[4:]))));mat.SetTranslateOnly(Gf.Vec3d(*map(float,pose[:3])));op.Set(mat*inv)
                rep.orchestrator.step(rt_subframes=3,delta_time=0.,pause_timeline=True)
                if timeline.is_playing() or abs(timeline.get_current_time())>1e-12:raise RuntimeError('physics advanced during static replay')
                for view,(ann,path) in products.items():
                    pixels=np.asarray(ann.get_data())[:,:,:3].copy()
                    if pixels.shape!=(720,1280,3) or pixels.std()<5:raise RuntimeError('invalid RGB')
                    active[view][0].stdin.write(pixels.tobytes())
                    if f in (0,len(rr)//2,len(rr)-1):Image.fromarray(pixels).save(out/f'{name}_{view}_{f:05d}.png')
                row=rows[int(r)];fm.append({'frame_index':f,'video_time_s':f/args.fps,'simulation_interval_end_s':float(t),'measurement_row':int(r),'measurement_time_s':float(times[r]),'pose_index':int(k),'pose_row':int(indices[k]),'pose_time_s':float(times[indices[k]]),'phase':row['phase'],'subphase':row['subphase'],'hall_N':row['hall_N'],'front_N':sum(float(row.get(key) or 0) for key in ('lip_N','diaphragm_N','contact_rim_N'))})
            for view,(proc,path) in active.items():
                proc.stdin.close()
                if proc.wait()!=0:raise RuntimeError('encoder failed')
                encoders.remove(proc)
                manifest['clips'].append({'clip_id':f'{name}_{view}','target':name,'view':view,'video':path.name,'video_sha256':sha(path),'frame_count':len(rr),'duration_s':len(rr)/args.fps,'point_start_physics_row':points[name]['start_physics_row'],'point_end_physics_row':points[name]['end_physics_row'],'frames':fm})
            print('REVIEW_CLIP_DONE',name,flush=True)
        if any(sha(PROJECT/rel)!=digest for rel,digest in inputs.items()):raise RuntimeError('source changed during render')
        manifest['source_files_unchanged']=True
        (out/'review_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n');print('REVIEW_RENDER_COMPLETE',flush=True)
    finally:
        for proc in encoders:
            if proc.stdin and not proc.stdin.closed:proc.stdin.close()
            proc.wait()
        for ann,product in attached:ann.detach([product.path]);product.destroy()
        app.close()


if __name__=='__main__':main()
