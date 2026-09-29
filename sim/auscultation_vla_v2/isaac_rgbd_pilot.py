"""Live, passive RGB-D/Hall/joint pilot. Never a qualified expert dataset.

The arm holds its initial joints. Only the simulated patient's common parent
is translated a few millimetres to test sensor-derived target motion. No robot
hardware APIs are imported. Simulator patient motion is an evaluation ledger,
not a tracker input. The approved camera pose is restored unchanged.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys


def _capture(app,errors,released,args):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from PIL import Image
    from .body_tracking import BodyTracker
    from .hall import HallSimulator,default_calibration
    from .observations import Observation,RobotState,encode
    result=dict(schema='wam_live_multimodal_pilot_v1',passed=False,frames=[],
        task_training_data=False,expert_actions=False,hardware_access=False,audio='absent',errors=[])
    product=rgb_ann=depth_ann=None;status=1
    try:
        from pxr import UsdGeom,Gf
        import omni.replicator.core as rep
        from sim.auscultation_expert_layout_v1.env import IsaacContactEnv
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        env=IsaacContactEnv(args.output/'scene');kin=OpenArmKinematics()
        camera=json.loads((args.source_project/'sim/camera_framing_20260923/recommended_camera.json').read_text())
        annotation=json.loads(args.annotation.read_text())
        optical=np.asarray(camera['world_from_optical']);k=camera['intrinsics']
        scale=args.width/k['width']
        k={name:(int(round(value*scale)) if name in ('width','height') else value*scale) for name,value in k.items()}
        annotation=dict(annotation,target_pixels={name:(np.array(pixel)*scale).tolist() for name,pixel in annotation['target_pixels'].items()},
                        support_polygon_px=(np.array(annotation['support_polygon_px'])*scale).tolist())
        K=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1.]])
        cache=UsdGeom.XformCache()
        for path in (camera['camera_path'],'/World/RobotCamBody'):
            prim=env.stage.GetPrimAtPath(path)
            if not prim:raise ValueError('approved camera missing')
            inv=cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder()
            xf.AddTransformOp(opSuffix='approved7deg').Set(Gf.Matrix4d(np.asarray(camera['world_from_camera_usd']).T)*inv)
        product=rep.create.render_product(camera['camera_path'],(k['width'],k['height']))
        rgb_ann=rep.AnnotatorRegistry.get_annotator('rgb');depth_ann=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
        rgb_ann.attach([product]);depth_ann.attach([product]);rep.orchestrator.set_capture_on_play(False)
        cal=default_calibration();hall=HallSimulator(cal);hold=env.state()['q'].copy()
        for _ in range(480):
            s=env.step(hold,label='PASSIVE_SENSOR_PILOT',target='none');hall.update(s['time_s'],s['compression_m'])
        patient=UsdGeom.Xformable(env.stage.GetPrimAtPath('/World/Patient'))
        patient_initial=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(patient.GetPrim())).T
        shift=patient.AddTranslateOp(opSuffix='sensorPilotTranslation')
        tracker=BodyTracker(K,np.linalg.inv(env.base)@optical)
        initial=None
        for index,dx in enumerate((0.,.002,.004,.002,0.,-.002,0.)):
            shift.Set(Gf.Vec3d(dx,0.,0.))
            for _ in range(12):
                s=env.step(hold,label='PASSIVE_SENSOR_PILOT',target='none');hall.update(s['time_s'],s['compression_m'])
            clock_before=float(env.world.current_time);q_before=s['q'].copy()
            rep.orchestrator.step(rt_subframes=4,delta_time=0.,pause_timeline=False)
            clock_after=float(env.world.current_time);s_after=env.state()
            if abs(clock_after-clock_before)>1e-8 or np.max(abs(s_after['q']-q_before))>1e-8:
                raise ValueError('render capture advanced unaccounted physics')
            rgb=np.asarray(rgb_ann.get_data())[...,:3].copy();depth=np.asarray(depth_ann.get_data()).copy()
            valid=np.isfinite(depth)&(depth>0);t=s['time_s'];packet=hall.read(t)
            times=dict(rgb_time_s=t,depth_time_s=t,now_s=t)
            if index==0:
                targets=tracker.initialize(rgb,depth,valid,annotation['target_pixels'],annotation['support_polygon_px'],**times)
                initial=targets
            else:targets=tracker.update(rgb,depth,valid,**times)
            wrist=kin.fk(s['q']);robot=RobotState(s['q'],s['dq'],wrist[:3,3],wrist[:3,:3],packet,t)
            states={}
            for name,target in targets.items():
                obs=Observation(rgb,depth,valid,K,np.linalg.inv(env.base)@optical,t,t,robot,target,
                                'Observe the designated back target; keep the arm stationary.')
                states[name]=encode(obs,now_s=t,hall_calibration=cal)['state'].tolist()
            # Appended USD translate is in the patient's local frame. Evaluate
            # actual world displacement separately; never feed it to tracking.
            patient_now=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(patient.GetPrim())).T
            actual_world_shift=patient_now[:3,3]-patient_initial[:3,3]
            expected=env.base[:3,:3].T@actual_world_shift
            errors={name:float(np.linalg.norm(target.position_base_m-initial[name].position_base_m-expected)) for name,target in targets.items()}
            stem='%03d'%index;Image.fromarray(rgb).save(args.output/(stem+'_rgb.png'))
            np.savez_compressed(args.output/(stem+'_depth.npz'),depth_m=depth,valid=valid)
            row=dict(index=index,time_s=t,physics_clock_s=clock_before,render_physics_delta_s=clock_after-clock_before,
                measured_q=s['q'].tolist(),measured_dq=s['dq'].tolist(),hall=asdict(packet),
                states=states,evaluation_only_patient_translation_world_m=actual_world_shift.tolist(),
                relative_target_error_m=errors)
            result['frames'].append(row)
            print('LIVE_SENSOR_FRAME',index,'max_motion_error_m',max(errors.values()),flush=True)
        result.update(camera_config_sha256=hashlib.sha256(json.dumps(camera,sort_keys=True).encode()).hexdigest(),
            annotation_sha256=hashlib.sha256(args.annotation.read_bytes()).hexdigest(),calibration_sha256=cal.sha256,
            max_relative_target_error_m=max(max(r['relative_target_error_m'].values()) for r in result['frames']),
            scope='Seven live passive frames, small rigid torso translation only; no contact or VLA control')
        result['passed']=result['max_relative_target_error_m']<.003 and len(result['frames'])==7
        status=0 if result['passed'] else 1
    except Exception as error:
        result['errors'].append(type(error).__name__+': '+str(error))
        print('LIVE_SENSOR_FAILURE',result['errors'][-1],flush=True)
    finally:
        from sim.auscultation_skill_v1.media_teardown import release_owned_render_resources
        released.update(release_owned_render_resources(rep,
            [a for a in (rgb_ann,depth_ann) if a is not None],
            [] if product is None else [product],errors))
        (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-project',type=Path,required=True)
    parser.add_argument('--annotation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=1920)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sys.path.append(str(args.source_project.resolve()))
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'fast_shutdown':False})
    from sim.auscultation_skill_v1.media_teardown import run_scoped_capture,require_complete_teardown
    status=1
    try:
        result,receipt=run_scoped_capture(app,_capture,args)
        require_complete_teardown(receipt,annotators=2,products=1)
        result['teardown']=receipt;result['close_returned']=True
        status=0 if result['passed'] else 1
    except Exception as error:
        path=args.output/'result.json'
        result=json.loads(path.read_text()) if path.exists() else dict(passed=False,errors=[])
        result['passed']=False;result['errors'].append(type(error).__name__+': '+str(error))
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'}),flush=True)
    sys.stdout.flush();sys.stderr.flush();os._exit(status)


if __name__=='__main__':main()
