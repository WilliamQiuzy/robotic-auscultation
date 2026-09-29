"""Fresh V2 live sensor collection with the frozen, checked route-B teacher.

This producer does not seal a training manifest. Independent contact/posture,
sensor pairing and action-label audits must qualify each completed rollout.
Teacher geometry is privileged; it never enters BodyTracker or encoded state.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def capture(app,errors,released,args,bundle,snapshot_sha,binding_sha):
    import numpy as np
    from sim.auscultation_repair_v3.physics_profile import make_env,PhysicsProfile
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.controller import verify_snapshot,digest
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from sim.auscultation_skill_v1.controller import ExpertController
    from sim.auscultation_vla_v2.live_capture import LiveCapture
    env=sensor=controller=None
    outcome=dict(success=False,training_accepted=False,points=[],failure_reason='construction incomplete')
    try:
        kin=OpenArmKinematics();base=np.array(bundle.plan['world_from_left_base'])
        with tempfile.TemporaryDirectory(prefix='wam_vla_geometry_') as temporary:
            folder=Path(temporary)
            for label,name in [('collision_metadata','collision_geometry.json'),('collision_arrays','collision_geometry.npz')]:
                (folder/name).write_bytes(bundle.inputs[label][1])
            checker=CertifiedCollisionChecker(kin,base,directory=folder)
        env=make_env(args.output,PhysicsProfile(args.position_iterations,1,args.armature))
        if not np.allclose(env.base,base,atol=1e-7,rtol=0):raise ValueError('live robot base differs from plan')
        if args.sensors:
            sensor=LiveCapture(env,kin,args.source_project/'sim/camera_framing_20260923/recommended_camera.json',
                               args.annotation,args.output/'sensors',width=640)
        class ObservedTeacher(ExpertController):
            def desired_normal_velocity(self,state):
                desired=super().desired_normal_velocity(state)
                if not args.gentle:return desired
                from sim.auscultation_repair_v3.gentle_contact import limit_near_contact
                center=self.contact_curves[-1]['nominal_contact_center_gap_m']
                return limit_near_contact(self.manager.state.phase.value,desired,self.metrics(state)['gap_m'],center)

            def _tick(self,command,subphase,active=False,recovery=False):
                state=super()._tick(command,subphase,active=active,recovery=recovery)
                if sensor is not None:sensor.tick(state)
                if sensor is not None and len(env.rows)%12==0:
                    eligible=subphase not in {'settle','initial_home','return_bridge','return_home','home_settle'} and not recovery
                    if subphase in {'transfer','inter_point_transfer','align','settle','initial_home'}:phase='FREE'
                    elif subphase.startswith('normal_'):phase='RETRACT'
                    else:phase=self.manager.state.phase.value
                    if phase not in {'FREE','APPROACH','CONTACT','HOLD','RETRACT'}:
                        phase='FREE';eligible=False
                    sensor.capture(state,phase,self.target.name,len(env.rows)-1,eligible=eligible,subphase=subphase)
                return state
        controller=ObservedTeacher(env,kin,checker,generator_policy='revised_shortcut',speed_scale=args.speed_scale)
        outcome=controller.run([(bundle.target,bundle.row)])
        verify_snapshot(args.output/'input_snapshot',verify_sources=True)
    except Exception as error:
        outcome.update(success=False,training_accepted=False,failure_reason=type(error).__name__+': '+str(error))
        print('LIVE_EXPERT_FAILURE',outcome['failure_reason'],flush=True)
    finally:
        outcome.update(requested_points=[args.point],input_snapshot='input_snapshot/manifest.json',
            input_snapshot_manifest_sha256=snapshot_sha,collision_regeneration_sha256=binding_sha,
            scope='Fresh route-B expert; repaired physics candidate; audio absent',
            dataset_role='unqualified_live_demonstration' if sensor else 'physics_only_diagnostic_not_training_data',
            sensor_frames=0 if sensor is None else len(sensor.rows),
            body_source='existing_seated_patient_asset_v1',layout_family='approved_farther_lower_seat_camera7deg',
            motion_family='independent_target_plan_'+args.point,source_project=str(args.source_project))
        outcome['repair_contact_policy']=dict(enabled=args.gentle,
            inward_near_contact_limit_m_s=.001,loaded_withdrawal_limit_m_s=.0005,
            normal_executor='unchanged acceleration/jerk bounded executor',hold_duration_s=3.,
            supersedes_parent_policy_timing_description=args.gentle)
        if env is not None:env.save('expert')
        if controller is not None:
            (args.output/'control_commands.json').write_text(json.dumps(controller.control_rows,allow_nan=False)+'\n')
        if sensor is not None:sensor.release(errors,released)
        (args.output/'outcome.json').write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n')
    return outcome


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--annotation',type=Path,required=True)
    parser.add_argument('--point',required=True,choices=[s+str(i) for i in range(1,6) for s in 'RL'])
    parser.add_argument('--speed-scale',type=float,default=1.)
    parser.add_argument('--sensors',action='store_true')
    parser.add_argument('--gentle',action='store_true')
    parser.add_argument('--armature',type=float,default=.01)
    parser.add_argument('--position-iterations',type=int,default=32)
    args=parser.parse_args();args.source_project=args.source_project.resolve();args.output=args.output.resolve()
    if args.output.exists():raise ValueError('new output required')
    reserve=3*2**30
    if shutil.disk_usage(args.output.parent).free<reserve:raise RuntimeError('insufficient collection reserve')
    sys.path.append(str(args.source_project))
    runtime=args.source_project/'sim/auscultation_expert_layout_v1'
    sys.path.insert(0,str(runtime/'python_deps'))
    from sim.auscultation_expert_layout_v1.controller import validate_plan,create_snapshot,verify_snapshot,digest
    from sim.auscultation_expert_layout_v1.run_sequence import source_paths
    from sim.auscultation_expert_layout_v1.revalidate_plans import validate_geometry_binding
    plans=args.source_project/'sim/upper_pair_posture_20260924/results/candidate_b_plans'
    bundle=validate_plan(plans/('planned_paths_'+args.point+'.json'),plans,runtime)
    binding_sha=validate_geometry_binding(runtime)
    if bundle.plan.get('revalidation',{}).get('collision_regeneration_sha256')!=binding_sha:
        raise ValueError('plan geometry binding mismatch')
    args.output.mkdir();sources=source_paths(runtime)
    sources['collision_regeneration']=runtime/'results/collision_regeneration_verification.json'
    for name in ('collect_expert.py','live_capture.py','body_tracking.py','rigid_registration.py','tracking.py','observations.py','hall.py','multimodal.py'):
        sources['vla/'+name]=Path(__file__).parents[1]/'auscultation_vla_v2'/name
    for name in ('collect_repaired_expert.py','physics_profile.py','metrics.py','teardown.py','gentle_contact.py'):
        sources['repair/'+name]=Path(__file__).parents[1]/'auscultation_repair_v3'/name
    sources['pi05_v4/collect_demo.py']=Path(__file__)
    sources['vla/annotation']=args.annotation
    snapshot=create_snapshot([bundle],args.output/'input_snapshot',sources);verify_snapshot(snapshot,verify_sources=True)
    snapshot_sha=digest((snapshot/'manifest.json').read_bytes())
    import fcl  # Fail before Kit if the explicit collision dependency path is absent.
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'fast_shutdown':False})
    from sim.auscultation_skill_v1.media_teardown import run_scoped_capture,require_complete_teardown
    status=1;report=dict(passed=False,errors=[])
    try:
        outcome,receipt=run_scoped_capture(app,capture,args,bundle,snapshot_sha,binding_sha)
        (args.output/'teardown.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if args.sensors:require_complete_teardown(receipt,annotators=2,products=1)
        else:
            from sim.auscultation_repair_v3.teardown import require_no_sensor_teardown
            require_no_sensor_teardown(receipt)
        from sim.auscultation_skill_v1.audits import audit_saved
        audit=audit_saved(args.output)
        from sim.auscultation_repair_v3.metrics import motion_metrics
        rows=[json.loads(line) for line in (args.output/'expert_trace.jsonl').open()]
        dynamics=motion_metrics([r['time_s'] for r in rows],[r['q'] for r in rows],[r['dq'] for r in rows])
        (args.output/'measured_dynamics.json').write_text(json.dumps(dynamics,indent=2)+'\n')
        report.update(passed=outcome['success'] and audit['accepted'],teardown=receipt,audit=audit,
                      sensor_frames=outcome['sensor_frames'],qualification='physics candidate; not sealed for training',
                      measured_dynamics=dynamics,
                      smoothness_target_met=dynamics['centered']['acceleration_rad_s2']['max']<=2 and dynamics['centered']['jerk_rad_s3']['max']<=20)
        report['mechanics_passed']=report['passed']
        report['passed']=report['passed'] and report['smoothness_target_met']
        status=0 if report['passed'] else 1
    except Exception as error:report['errors'].append(type(error).__name__+': '+str(error))
    (args.output/'collection_result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('LIVE_EXPERT_RESULT',json.dumps(report),flush=True)
    sys.stdout.flush();sys.stderr.flush();os._exit(status)


if __name__=='__main__':main()
