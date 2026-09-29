"""Live multimodal pi05 physics evaluation with independent safety audits.

No learned rollout is qualified as an expert demonstration. Initial task points
are provided explicitly; subsequent body tracking uses the RGB-D observation.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import importlib


def validate_sequence_bundles(bundles,points,binding_sha):
    """Every point must use the same verified scene geometry and robot base."""
    import numpy as np
    if not bundles or [bundle.target.name for bundle in bundles]!=list(points):
        raise ValueError('validated plan order differs from requested points')
    base=np.asarray(bundles[0].plan['world_from_left_base'],float)
    if base.shape!=(4,4) or not np.isfinite(base).all():raise ValueError('invalid sequence base')
    for bundle in bundles:
        current=np.asarray(bundle.plan['world_from_left_base'],float)
        if current.shape!=(4,4) or not np.allclose(current,base,atol=1e-9,rtol=0):
            raise ValueError('sequence plan robot bases differ')
        if bundle.plan.get('revalidation',{}).get('collision_regeneration_sha256')!=binding_sha:
            raise ValueError('plan geometry binding mismatch')
        for label in ('collision_metadata','collision_arrays'):
            if bundle.inputs[label][1]!=bundles[0].inputs[label][1]:
                raise ValueError('sequence plan collision geometry differs: '+label)


def capture(app,errors,released,args,bundles,snapshot_sha,binding_sha):
    import numpy as np
    from sim.auscultation_repair_v3.physics_profile import make_env,PhysicsProfile
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.controller import verify_snapshot,digest
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from .learned_controller import LearnedController
    from .policy_client import PolicyClient
    from .live_capture import LiveCapture
    env=sensor=controller=None
    outcome=dict(success=False,training_accepted=False,points=[],failure_reason='construction incomplete')
    try:
        bundle=bundles[0]
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
                               args.annotation,args.output/'sensors',width=640,hall_condition=args.hall_condition_data)
            sensor.warm_up()
        client=PolicyClient(args.client_config)
        controller=LearnedController(env,kin,checker,sensor=sensor,client=client,task_scope=args.task_scope,
            generator_policy='revised_shortcut',speed_scale=args.speed_scale)
        outcome=controller.run([(item.target,item.row) for item in bundles])
        verify_snapshot(args.output/'input_snapshot',verify_sources=True)
    except Exception as error:
        outcome.update(success=False,training_accepted=False,failure_reason=type(error).__name__+': '+str(error))
        print('LEARNED_POLICY_FAILURE',outcome['failure_reason'],flush=True)
    finally:
        outcome.update(requested_points=args.points,input_snapshot='input_snapshot/manifest.json',
            input_snapshot_manifest_sha256=snapshot_sha,collision_regeneration_sha256=binding_sha,
            scope='Learned pi05 policy physics evaluation; no teacher action substitution; audio absent',
            dataset_role='learned_policy_evaluation_not_expert_data',expert_demonstration=False,training_dataset_eligible=False,
            sensor_frames=0 if sensor is None else len(sensor.rows),
            body_source='existing_seated_patient_asset_v1',layout_family='approved_farther_lower_seat_camera7deg',
            motion_family=('independent_target_plan_'+args.points[0] if len(args.points)==1 else
                           'continuous_learned_target_sequence_'+'_'.join(args.points)),
            source_project=str(args.source_project),hall_condition=args.hall_condition_data,
            hall_condition_scope='Explicit engineering sensor perturbations; not measured distributions or pose/motion generalization',
            sequence_execution=dict(requested_point_count=len(args.points),scene_instances=1,
                resets_between_points=0,task_scope=args.task_scope,
                inter_point_motion='learned FREE/APPROACH; no classical transfer' if len(args.points)>1 else 'not applicable',
                final_home='classical after final successful learned release, outside learned task interval'))
        if env is not None:env.save('expert')
        if controller is not None:
            (args.output/'control_commands.json').write_text(json.dumps(controller.control_rows,allow_nan=False)+'\n')
        if sensor is not None:sensor.release(errors,released)
        (args.output/'outcome.json').write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n')
    return outcome


def parse_args(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--source-project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--annotation',type=Path,required=True)
    selection=parser.add_mutually_exclusive_group(required=True)
    choices=[s+str(i) for i in range(1,6) for s in 'RL']
    selection.add_argument('--point',choices=choices)
    selection.add_argument('--points',nargs='+',choices=choices)
    parser.add_argument('--speed-scale',type=float,default=1.)
    parser.add_argument('--sensors',action='store_true')
    parser.add_argument('--client-config',type=Path,required=True)
    parser.add_argument('--hall-condition',type=Path)
    parser.add_argument('--task-scope',choices=['full','contact'],default='full')
    parser.add_argument('--armature',type=float,default=.01)
    parser.add_argument('--position-iterations',type=int,default=32)
    args=parser.parse_args(argv);args.points=[args.point] if args.point is not None else args.points
    if len(set(args.points))!=len(args.points):parser.error('points must be unique and ordered')
    if args.task_scope=='contact' and len(args.points)>1:
        parser.error('contact component scope accepts one point; continuous sequences require full learned scope')
    return args


def main():
    args=parse_args();args.source_project=args.source_project.resolve();args.output=args.output.resolve();args.sensors=True
    args.hall_condition_data=None
    if args.hall_condition is not None:
        args.hall_condition=args.hall_condition.resolve()
        args.hall_condition_data=json.loads(args.hall_condition.read_text())
        if not isinstance(args.hall_condition_data,dict):raise ValueError('Hall condition must be a JSON object')
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
    bundles=[validate_plan(plans/('planned_paths_'+point+'.json'),plans,runtime) for point in args.points]
    binding_sha=validate_geometry_binding(runtime)
    validate_sequence_bundles(bundles,args.points,binding_sha)
    args.output.mkdir();sources=source_paths(runtime)
    # These parent packages implement the actual execution and audit chain.
    # Resolve via Python's active import roots, not guessed mirror locations.
    for package in ('auscultation_skill_v1','curobo_expert_v1','expert_quality_20260924'):
        imported=importlib.import_module('sim.'+package)
        for directory in imported.__path__:
            for path in Path(directory).glob('*.py'):
                sources['runtime/'+package+'/'+path.name]=path
    sources['collision_regeneration']=runtime/'results/collision_regeneration_verification.json'
    for name in ('collect_expert.py','live_capture.py','body_tracking.py','rigid_registration.py','tracking.py','observations.py','hall.py','multimodal.py'):
        sources['vla/'+name]=Path(__file__).parents[1]/'auscultation_vla_v2'/name
    calibration_module=importlib.import_module('auscultation.mechanical_sensor.force_feedback.force_sensor')
    sources['vla/force_sensor.py']=Path(calibration_module.__file__)
    sources['vla/hall_calibration.json']=Path(__file__).parents[1]/'auscultation_vla_v2/calibration/veo6_tared_snapshot.json'
    for name in ('physics_profile.py','metrics.py','teardown.py'):
        sources['repair/'+name]=Path(__file__).parents[1]/'auscultation_repair_v3'/name
    for name in ('run_isaac.py','learned_controller.py','execution.py','policy_client.py','action_contract.py','live_capture.py','hall_conditions.py','supervision.py','render_warmup.py'):
        sources['pi05_v5/'+name]=Path(__file__).with_name(name)
    sources['vla/annotation']=args.annotation
    if args.hall_condition is not None:sources['pi05_v5/hall_condition']=args.hall_condition
    snapshot=create_snapshot(bundles,args.output/'input_snapshot',sources);verify_snapshot(snapshot,verify_sources=True)
    snapshot_sha=digest((snapshot/'manifest.json').read_bytes())
    import fcl  # Fail before Kit if the explicit collision dependency path is absent.
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'fast_shutdown':False})
    from sim.auscultation_skill_v1.media_teardown import run_scoped_capture,require_complete_teardown
    status=1;report=dict(passed=False,errors=[])
    try:
        outcome,receipt=run_scoped_capture(app,capture,args,bundles,snapshot_sha,binding_sha)
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
                      sensor_frames=outcome['sensor_frames'],qualification='learned policy component/full test as explicitly declared; no generalization attestation',
                      measured_dynamics=dynamics,
                      smoothness_target_met=dynamics['centered']['acceleration_rad_s2']['max']<=2 and dynamics['centered']['jerk_rad_s3']['max']<=20)
        report['smoothness_gate_definition']={'estimator':'historical centered np.gradient three times, unfiltered q',
            'acceleration_rad_s2':2.,'jerk_rad_s3':20.,
            'native_forward_also_reported':True,'not_a_claim_all_differentiation_estimators_pass':True}
        report['mechanics_passed']=report['passed']
        report['passed']=report['passed'] and report['smoothness_target_met']
        status=0 if report['passed'] else 1
    except Exception as error:report['errors'].append(type(error).__name__+': '+str(error))
    (args.output/'collection_result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('LIVE_EXPERT_RESULT',json.dumps(report),flush=True)
    sys.stdout.flush();sys.stderr.flush();os._exit(status)


if __name__=='__main__':main()
