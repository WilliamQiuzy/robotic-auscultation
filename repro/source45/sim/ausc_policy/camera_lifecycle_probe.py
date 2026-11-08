"""Twelve-step/three-frame owned camera lifecycle probe, never physics acceptance.

Compose a new config with make_config(existing_compact_config, ...), freeze it,
then invoke the ORIGINAL owner:
python -m sim.ausc_policy.run --config NEW_PROBE_JSON --run-id UNIQUE_RUN_ID
The existing locked review wrapper gives the native child at most 120 seconds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import scene_store as store
from .native_lifecycle import shared_app_settings,close_shared_runtime
from .m2_capture import M2CameraSampler,APPROVED_CAMERA_SHA256,COMPACT_IMAGE_SIZE

SCHEMA='m2_camera_lifecycle_probe_config_v1'
STEPS=12
FRAMES=3
CHILD_TIMEOUT_SECONDS=120


def need(ok,message):
    if not ok:raise ValueError(message)


def make_config(runtime_config,*,base,asset_root,isaac_python,ld_library_path):
    """CPU configuration intent only; no preparation or native launch."""
    path=store.real_path(runtime_config)
    return dict(schema=SCHEMA,experiment='camera_lifecycle_probe',base=str(base),
        asset_root=str(asset_root),isaac_python=str(isaac_python),ld_library_path=str(ld_library_path),
        time_limit_seconds=180,child_timeout_seconds=CHILD_TIMEOUT_SECONDS,
        runtime_config=dict(file=str(path),sha256=store.sha(path)))


def preflight(config):
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1','Original Linux experiment owner required')
    need(config.get('schema')==SCHEMA and config.get('experiment')=='camera_lifecycle_probe' and
         config.get('child_timeout_seconds')==CHILD_TIMEOUT_SECONDS and config.get('time_limit_seconds')==180,
         'Exact bounded camera lifecycle probe configuration required')
    ref=config['runtime_config']
    need(isinstance(ref,dict) and set(ref)=={'file','sha256'},'Hash-bound compact runtime config required')
    path=store.real_path(ref['file']);payload=path.read_bytes()
    need(hashlib.sha256(payload).hexdigest()==ref['sha256'],'Compact runtime config changed')
    runtime=json.loads(payload)
    need(runtime.get('compact_scene_storage',{}).get('schema')==store.SDK_SCHEMA and
         'm2_act_deployment' not in runtime,'SDK v2 compact expert config required; no learned policy')
    store.validate_config(runtime)
    return runtime


def read_camera_bytes(output,runtime):
    """Authenticate the definition and its separate prepared overlay receipt."""
    from .approved_camera import load_approved_camera,PATHS
    capture=runtime['fresh_expert']['raw_multimodal_capture']
    need(capture.get('camera_sha256')==APPROVED_CAMERA_SHA256,'Runtime camera definition hash differs from approval')
    payload=store.real_path(capture['camera_file']).read_bytes()
    definition=load_approved_camera(payload)
    reference=runtime['compact_scene_storage']['sources']['approved_camera']
    receipt=json.loads(store.read_source(dict(reference,
        file=str(store.real_path(output)/'approved_camera_restoration.json'))))
    need(receipt.get('camera_sha256')==APPROVED_CAMERA_SHA256 and
         receipt.get('changed_paths')==list(PATHS) and
         all(receipt.get(key) is True for key in ('non_camera_attributes_unchanged',
             'physical_geometry_unchanged','camera_optics_unchanged','restored_user_approved_pose')) and
         receipt.get('physics_executed') is False and receipt.get('training_eligible') is False,
         'Invalid authenticated camera restoration receipt')
    after=receipt.get('after',{})
    need(isinstance(after,dict) and set(after)==set(PATHS),'Incomplete camera restoration transforms')
    desired=np.asarray(definition['world_from_camera_usd'],float)
    need(all(np.asarray(after[path],float).shape==(4,4) and
             np.allclose(after[path],desired,atol=1e-9,rtol=0) for path in PATHS),
         'Camera restoration differs from approved definition pose')
    return payload


def create_environment(stage,mass,output):
    from sim.auscultation_expert_layout_v1.env import IsaacContactEnv
    from .fresh_expert import scene_factory
    ref=json.loads((Path(output)/'scene_ref.json').read_bytes())
    return IsaacContactEnv(Path(output)/'fresh_expert',stage_factory=scene_factory(stage,ref),
        contact_offset_m=.132,moving_mass_kg=float(mass['mass_kg']),scene_reference=ref)


def stop_started_timeline():
    """Constructor failure can leave a playing World without returning it."""
    import omni.timeline
    omni.timeline.get_timeline_interface().stop()


def finalize_environment(env,output):
    """Use the same finalizer as full fresh runs, after sampler finalization."""
    from .fresh_expert import finalize_probe
    _,errors=finalize_probe(env,None,None,output)
    need(not errors,'Full fresh finalizer failed: '+repr(errors))
    return dict(errors=errors,stage_cache_lease=env.private_scene_cache_lease_receipt)


def exercise_camera(env,camera_bytes,*,sampler_factory=M2CameraSampler,finalizer=None):
    """Observe every post-initialization physics callback, including render/cleanup."""
    count=[0];sampler=None;shapes=[];receipt=None;error=None;full_finalizer=None
    def physics_step(_dt):count[0]+=1
    try:
        env.world.add_physics_callback('wam_camera_lifecycle_probe',physics_step)
        sampler=sampler_factory(env,camera_bytes=camera_bytes,native_image_size=COMPACT_IMAGE_SIZE)
        need(count[0]==0,'Camera setup advanced physics')
        for index in range(STEPS):
            env.world.step(render=False);env.t+=1/240
            need(count[0]==index+1,'Unexpected physics callback count')
            if (index+1)%(STEPS//FRAMES):continue
            before=count[0];rgb,depth,valid=sampler.capture(env.state())
            need(count[0]==before,'Camera render advanced physics')
            need(rgb.dtype==np.uint8 and rgb.shape in ((144,256,3),(144,256,4)) and
                 depth.shape==(144,256) and depth.dtype.kind=='f' and valid.shape==(144,256) and
                 valid.dtype==bool,'Unexpected approved compact RGB-D shape/type')
            shapes.append(dict(rgb=list(rgb.shape),depth=list(depth.shape),valid=list(valid.shape)))
    except BaseException as caught:error=caught
    finally:
        try:
            if sampler is not None:
                receipt=sampler.finalize(error)
                need(receipt.get('completed') is True,'Camera finalization failed: '+repr(receipt))
        except BaseException as caught:
            if error is None:error=caught
        finally:
            try:
                if finalizer is None:env.world.stop()
                else:full_finalizer=finalizer()
            except BaseException as caught:
                if error is None:error=caught
    if error is not None:raise error
    need(count[0]==STEPS,'Camera finalization/world stop advanced physics')
    return dict(physics_callback_count=count[0],explicit_physics_steps=STEPS,
        zero_step_render_count=len(shapes),frame_shapes=shapes,camera=receipt,full_finalizer=full_finalizer,
        count_scope='All callbacks after environment initialization, including sampler setup and cleanup'),lambda:count[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_bytes());runtime=preflight(config)
    output=store.real_path(args.output,exists=False);output.mkdir(parents=True,exist_ok=False)
    (output/'probe_config.json').write_bytes(args.config.read_bytes())
    (output/'runtime_config.json').write_text(json.dumps(runtime,sort_keys=True)+'\n')
    app=None;env=None;closing=False;constructing_environment=False
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(shared_app_settings())
        stage,mass=store.prepared_runtime(runtime,output)
        camera_bytes=read_camera_bytes(output,runtime)
        constructing_environment=True
        env=create_environment(stage,mass,output)
        constructing_environment=False
        # All argument authentication is complete; exercise owns world.stop
        # immediately, including callback registration failure.
        observed,counter=exercise_camera(env,camera_bytes,finalizer=lambda:finalize_environment(env,output))
        env=None  # Match full run_probe scope: StageCache lease, not this local, owns the attached stage.
        (output/'lifecycle_observed.json').write_text(json.dumps(observed,sort_keys=True,allow_nan=False)+'\n')
        closing=True;closing_app,app=app,None
        shutdown=close_shared_runtime(closing_app,progress_path=output/'runtime_shutdown_progress.jsonl')
        need(counter()==STEPS,'Native cleanup/close advanced physics')
        result=dict(schema='m2_camera_lifecycle_probe_result_v1',success=True,completed=True,
            lifecycle_qualification=True,physical_validation=False,training_eligible=False,
            model_executed=False,compact_capture_complete=False,raw_capture_complete=False,
            observed=observed,runtime_shutdown=shutdown,
            training_scene_sha256=runtime['compact_scene_storage']['sources']['scene']['sha256'],
            scope='Camera lifecycle and normal shutdown only; no motion/contact/data qualification')
        store.finalize_job(output,result)
        need(result.get('success') is True,'Lifecycle post-close integrity/byte finalization failed')
        with (output/'COMPLETE.json').open('rb') as stream:os.fsync(stream.fileno())
        return True
    except BaseException as error:
        details=dict(success=False,completed=False,lifecycle_qualification=False,
            physical_validation=False,training_eligible=False,error=repr(error))
        if constructing_environment:
            try:stop_started_timeline()
            except BaseException as stop_error:details['timeline_stop_error']=repr(stop_error)
        env=None
        if app is not None and not closing:
            closing_app,app=app,None
            try:close_shared_runtime(closing_app,progress_path=output/'runtime_shutdown_progress.jsonl')
            except BaseException as close_error:details['shutdown_error']=repr(close_error)
        (output/'FAILED.json').write_text(json.dumps(details,sort_keys=True)+'\n')
        # Never promote SystemExit(0) from an SDK routine into lifecycle success.
        raise RuntimeError('Camera lifecycle probe failed: '+repr(error)) from error


def native_entrypoint():
    if main() is True:
        sys.stdout.flush();sys.stderr.flush();os._exit(0)


if __name__=='__main__':native_entrypoint()
