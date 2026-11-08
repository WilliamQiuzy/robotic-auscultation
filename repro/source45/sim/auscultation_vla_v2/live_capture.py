"""Synchronized Isaac sensor capture. Simulation rendering only, no hardware."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .body_tracking import BodyTracker
from .hall import HallSimulator,default_calibration
from .observations import Observation,RobotState,encode
from .multimodal import INSTRUCTIONS


class LiveCapture:
    def __init__(self,env,kinematics,camera_path,annotation_path,output,*,width=640):
        from pxr import UsdGeom,Gf
        import omni.replicator.core as rep
        self.rep=rep;self.env=env;self.kin=kinematics;self.out=Path(output);self.out.mkdir()
        self.rows=[];self.annotators=[];self.products=[]
        self.cal=default_calibration();self.hall=HallSimulator(self.cal)
        camera=json.loads(Path(camera_path).read_text());annotation=json.loads(Path(annotation_path).read_text())
        original=camera['intrinsics'];scale=width/original['width'];height=round(original['height']*scale)
        self.K=np.array([[original['fx']*scale,0,original['cx']*scale],
                         [0,original['fy']*scale,original['cy']*scale],[0,0,1.]])
        self.camera_to_base=np.linalg.inv(env.base)@np.array(camera['world_from_optical'])
        self.seeds={name:np.array(uv)*scale for name,uv in annotation['target_pixels'].items()}
        self.polygon=np.array(annotation['support_polygon_px'])*scale
        cache=UsdGeom.XformCache()
        for path in (camera['camera_path'],'/World/RobotCamBody'):
            prim=env.stage.GetPrimAtPath(path)
            if not prim:raise ValueError('approved camera missing')
            inv=cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder()
            xf.AddTransformOp(opSuffix='approved7deg').Set(Gf.Matrix4d(np.array(camera['world_from_camera_usd']).T)*inv)
        product=rep.create.render_product(camera['camera_path'],(width,height));self.products.append(product)
        for name in ('rgb','distance_to_image_plane'):
            ann=rep.AnnotatorRegistry.get_annotator(name);self.annotators.append(ann);ann.attach([product])
        rep.orchestrator.set_capture_on_play(False)
        self.tracker=BodyTracker(self.K,self.camera_to_base)
        self._initialized=False
        self.metadata=dict(schema='wam_live_rgbd_capture_v1',width=width,height=height,
            intrinsics=self.K.tolist(),camera_to_base=self.camera_to_base.tolist(),
            camera_sha256=hashlib.sha256(Path(camera_path).read_bytes()).hexdigest(),
            annotation_sha256=hashlib.sha256(Path(annotation_path).read_bytes()).hexdigest(),
            calibration_sha256=self.cal.sha256,audio='absent',depth_semantics='metric_optical_z',
            target_source='rgbd_surface_tracking',physics_render_synchronization='zero-delta render with measured clock/joint invariance checks')
        (self.out/'camera.json').write_text(json.dumps(self.metadata,indent=2)+'\n')

    def tick(self,state):
        self.hall.update(state['time_s'],state['compression_m'])

    def capture(self,state,phase,target,physics_row,*,eligible,subphase):
        clock=float(self.env.world.current_time);q=state['q'].copy()
        self.rep.orchestrator.step(rt_subframes=2,delta_time=0.,pause_timeline=False)
        if abs(float(self.env.world.current_time)-clock)>1e-8 or np.max(abs(self.env.state()['q']-q))>1e-8:
            raise ValueError('render advanced unaccounted physics')
        rgb=np.asarray(self.annotators[0].get_data())[...,:3].copy()
        depth=np.asarray(self.annotators[1].get_data()).copy();valid=np.isfinite(depth)&(depth>0)
        stamp=state['time_s'];times=dict(rgb_time_s=stamp,depth_time_s=stamp,now_s=stamp)
        index=len(self.rows);file='%05d.npz'%index
        np.savez_compressed(self.out/file,rgb=rgb,depth_m=depth,depth_valid=valid)
        if not self._initialized:
            estimates=self.tracker.initialize(rgb,depth,valid,self.seeds,self.polygon,**times);self._initialized=True
        else:estimates=self.tracker.update(rgb,depth,valid,**times)
        packet=self.hall.read(stamp);wrist=self.kin.fk(q)
        robot=RobotState(q,state['dq'],wrist[:3,3],wrist[:3,:3],packet,stamp)
        obs=Observation(rgb,depth,valid,self.K,self.camera_to_base,stamp,stamp,robot,estimates[target],INSTRUCTIONS[phase])
        encoded=encode(obs,now_s=stamp,hall_calibration=self.cal)
        raw_hash=hashlib.sha256((self.out/file).read_bytes()).hexdigest()
        row=dict(index=index,time_s=stamp,physics_row=physics_row,phase=phase,target=target,
            eligible=bool(eligible),subphase=subphase,
            raw_file=file,raw_sha256=raw_hash,state=encoded['state'].tolist(),q=q.tolist(),dq=state['dq'].tolist(),
            hall=asdict(packet),target_estimate=asdict(estimates[target]),
            prompt=INSTRUCTIONS[phase],render_physics_delta_s=float(self.env.world.current_time)-clock)
        row['tracking_quality']=getattr(self.tracker,'last_quality',None)
        # Convert NumPy arrays only for the explicit sensor/evidence ledger.
        for key,value in row['target_estimate'].items():
            if isinstance(value,np.ndarray):row['target_estimate'][key]=value.tolist()
        self.rows.append(row)
        with (self.out/'frames.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        if index%100==0:print('LIVE_EXPERT_FRAME',index,phase,target,round(stamp,2),flush=True)

    def release(self,errors,released):
        from sim.auscultation_skill_v1.media_teardown import release_owned_render_resources
        released.update(release_owned_render_resources(self.rep,self.annotators,self.products,errors))
        self.annotators.clear();self.products.clear()
