"""Command-identical PhysX diagnostic. No physical robot interfaces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def run(args,app):
    import numpy as np
    from pxr import Usd,UsdPhysics,PhysxSchema,UsdShade
    import sim.auscultation_expert_layout_v1.env as runtime
    from .metrics import motion_metrics
    inputs=dict(trace_sha256=sha(args.trace),scene_sha256=sha(args.scene),
                script_sha256=sha(__file__),mode=args.mode,solver=args.solver,
                velocity_iterations=args.velocity_iterations,substeps=args.substeps,
                contact_stiffness=args.contact_stiffness,contact_damping=args.contact_damping,armature=args.armature)
    rows=[json.loads(s) for s in args.trace.open()]
    original_prepare=runtime.prepare_stage
    def prepare(path):
        stage=Usd.Stage.Open(str(args.scene));stage.GetRootLayer().Export(str(path))
        stage=Usd.Stage.Open(str(path))
        if args.armature:
            joints=[]
            for p in stage.Traverse():
                if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName().startswith('openarm_left_joint'):
                    api=PhysxSchema.PhysxJointAPI.Apply(p);api.CreateArmatureAttr(args.armature)
                    joints.append(str(p.GetPath()))
            if len(joints)!=7:raise ValueError('Expected seven left revolute joints')
            inputs['armature_joints']=joints
        inputs['patient_materials_before']={}
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith('/World/Patient/') and prim.HasAPI(UsdPhysics.CollisionAPI):
                mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
                inputs['patient_materials_before'][str(prim.GetPath())]=str(mat.GetPath()) if mat else None
        if args.contact_stiffness:
            # Engineering contact-compliance diagnostic, not measured skin data.
            # Preserve existing material parameters by copying the bound material.
            from pxr import Sdf
            cache={}
            for path_name,old in inputs['patient_materials_before'].items():
                if old is None:raise ValueError('Patient collider has no explicit physics material; cannot preserve friction')
                if old not in cache:
                    target='/World/RepairContactMaterial'+str(len(cache))
                    Sdf.CopySpec(stage.GetRootLayer(),old,stage.GetRootLayer(),target)
                    mat=UsdShade.Material(stage.GetPrimAtPath(target))
                    api=PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
                    api.CreateCompliantContactStiffnessAttr(args.contact_stiffness)
                    api.CreateCompliantContactDampingAttr(args.contact_damping)
                    cache[old]=mat
                UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path_name)).Bind(cache[old],UsdShade.Tokens.strongerThanDescendants,'physics')
            if not cache:raise ValueError('No patient collider received compliance')
        if args.solver or args.velocity_iterations is not None:
            for p in stage.Traverse():
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    rb=PhysxSchema.PhysxRigidBodyAPI.Apply(p)
                    if args.solver:rb.CreateSolverPositionIterationCountAttr(args.solver)
                    if args.velocity_iterations is not None:rb.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
                if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                    ar=PhysxSchema.PhysxArticulationAPI.Apply(p)
                    if args.solver:ar.CreateSolverPositionIterationCountAttr(args.solver)
                    if args.velocity_iterations is not None:ar.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
        stage.GetRootLayer().Save()
        return stage
    runtime.prepare_stage=prepare
    old_dt=runtime.DT;runtime.DT=old_dt/args.substeps
    try:env=runtime.IsaacContactEnv(args.output)
    finally:runtime.prepare_stage=original_prepare
    inputs['effective_scene_sha256']=sha(args.output/'training_scene.usdc')
    inputs['solver_settings']={str(p.GetPath()):{a.GetName():a.Get() for a in p.GetAttributes()
        if 'solverPositionIterationCount' in a.GetName() or 'solverVelocityIterationCount' in a.GetName()}
        for p in env.stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.ArticulationRootAPI)}
    (args.output/'inputs.json').write_text(json.dumps(inputs,indent=2))
    render=[];resources=[];rep=None
    if args.mode=='render':
        import omni.replicator.core as rep
        product=rep.create.render_product('/World/RobotCam',(640,360));resources.append(product)
        for name in ('rgb','distance_to_image_plane'):
            a=rep.AnnotatorRegistry.get_annotator(name);a.attach([product]);resources.append(a)
        rep.orchestrator.set_capture_on_play(False)
    try:
        for i,row in enumerate(rows):
            for substep in range(args.substeps):
                s=env.step(row['q_command'],row['phase'],row['target'],
                    {'subphase':row.get('subphase',''),'source_row':i,'substep':substep})
            if rep is not None and (i+1)%12==0:
                clock=float(env.world.current_time)
                allq=env.robot.get_joint_positions().copy();alldq=env.robot.get_joint_velocities().copy()
                rep.orchestrator.step(rt_subframes=2,delta_time=0.,pause_timeline=False)
                afterq=env.robot.get_joint_positions();afterdq=env.robot.get_joint_velocities()
                render.append(dict(row=i,time_s=s['time_s'],
                    delta_clock=float(env.world.current_time)-clock,
                    max_delta_q=float(np.max(abs(afterq-allq))),
                    max_delta_dq=float(np.max(abs(afterdq-alldq))),
                    dq_before=alldq.tolist(),dq_after=afterdq.tolist()))
            if i%2400==0:print('REPLAY',i,'/',len(rows),flush=True)
        native=env.rows
        native_metrics=motion_metrics([r['time_s'] for r in native],[r['q'] for r in native],[r['dq'] for r in native])
        if args.substeps>1:
            np.savez_compressed(args.output/'native_measurements.npz',
                time_s=[r['time_s'] for r in native],q=[r['q'] for r in native],dq=[r['dq'] for r in native],
                compression_m=[r['compression_m'] for r in native],front_N=[r['contact_rim_N']+r['diaphragm_N']+r['lip_N'] for r in native],
                forbidden_N=[r['forbidden_N'] for r in native])
            env.rows=native[args.substeps-1::args.substeps]
            # Replay pose archive is omitted: its indices refer to native substeps.
            with (args.output/'replay_trace.jsonl').open('w') as f:
                for row in env.rows:f.write(json.dumps(row)+'\n')
        else:env.save('replay')
        observed=env.rows;t=[r['time_s'] for r in observed]
        report=dict(inputs=inputs,native_measured=native_metrics,measured=motion_metrics(t,[r['q'] for r in observed],[r['dq'] for r in observed]),
            original_measured=motion_metrics(t,[r['q'] for r in rows],[r['dq'] for r in rows]),
            tracking_error_rad=float(np.max(abs(np.array([r['q'] for r in observed])-np.array([r['q_command'] for r in rows])))),
            max_forbidden_N=max(r['forbidden_N'] for r in native),
            max_front_N=max(r['contact_rim_N']+r['diaphragm_N']+r['lip_N'] for r in native),
            original_q_difference_rad=float(np.max(abs(np.array([r['q'] for r in observed])-np.array([r['q'] for r in rows])))),
            render_events=len(render),render_max_delta_q=max((r['max_delta_q'] for r in render),default=0.),
            render_max_delta_dq=max((r['max_delta_dq'] for r in render),default=0.))
        (args.output/'render_invariance.json').write_text(json.dumps(render))
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('PROBE_RESULT',json.dumps(report),flush=True)
    finally:
        if rep is not None:
            for ann in resources[1:]:ann.detach([resources[0]])
            resources[0].destroy()
        env.contact_subscription=None
        env.world.stop()
        runtime.DT=old_dt


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('physics','render'),default='physics');p.add_argument('--solver',type=int,default=0)
    p.add_argument('--velocity-iterations',type=int)
    p.add_argument('--substeps',type=int,choices=(1,2,4,8),default=1)
    p.add_argument('--contact-stiffness',type=float,default=0.)
    p.add_argument('--contact-damping',type=float,default=10.)
    p.add_argument('--armature',type=float,default=0.)
    args=p.parse_args()
    if args.solver<0 or args.solver>255 or (args.velocity_iterations is not None and not 0<=args.velocity_iterations<=255):
        raise ValueError('invalid solver iterations')
    import math
    if any(not math.isfinite(v) or v<0 for v in [args.contact_stiffness,args.contact_damping,args.armature]):
        raise ValueError('Invalid nonnegative physical parameter')
    if args.output.exists():raise ValueError('Use a new output directory')
    args.output.mkdir(parents=True)
    for name in ('replay_probe.py','metrics.py'):
        (args.output/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'fast_shutdown':False})
    status=1
    try:run(args,app);status=0
    except Exception as error:
        import traceback
        traceback.print_exc();(args.output/'failure.json').write_text(json.dumps(dict(error=repr(error))))
    finally:app.close()
    sys.stdout.flush();sys.stderr.flush();os._exit(status)


if __name__=='__main__':main()
