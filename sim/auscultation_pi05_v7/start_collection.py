"""Independent V7 nonzero-start collector; never qualifies or seals data.

The capture and post-close audit flow are a versioned copy of V6 collect_demo.
Only the teacher factory, bound start/profile inputs and truthful provenance
change. Simulator/physics imports remain inside capture, after SimulationApp.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

CLASS='sim.auscultation_pi05_v7.start_teacher.StartObservedTeacher'
SOURCE='analytic_expert_shared_measured_supervisor_with_bound_start'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate JSON field: '+key)
            result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def collector_source_paths():
    base=Path(importlib.util.find_spec('sim.auscultation_pi05_v6').submodule_search_locations[0])
    names=('collect_demo.py','collect_demo.sh','teacher_timing.py','observed_supervisor.py',
           'supervision.py','live_capture.py','render_warmup.py','hall_conditions.py',
           'dynamics_contract.py','runtime_dynamics.py')
    sources={'pi05_v6/'+name:base/name for name in names}
    sources.update({'pi05_v7/'+name:Path(__file__).with_name(name)
                    for name in ('start_collection.py','start_teacher.py')})
    return sources


def argument_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--annotation',type=Path,required=True)
    parser.add_argument('--point',required=True,choices=[s+str(i) for i in range(1,6) for s in 'RL'])
    parser.add_argument('--episode-id',required=True)
    parser.add_argument('--start-manifest',type=Path,required=True)
    parser.add_argument('--start-manifest-sha256',required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    return parser


def bound_start(args,bundle):
    for path,expected in ((args.start_manifest,args.start_manifest_sha256),(args.protocol,args.protocol_sha256)):
        if not re.fullmatch('[0-9a-f]{64}',expected) or path.is_symlink() or sha(path)!=expected:
            raise ValueError('Explicit start/protocol SHA mismatch')
    if not re.fullmatch('[A-Za-z0-9_-]+',args.episode_id):raise ValueError('Simple unique episode identity required')
    start=read_json(args.start_manifest);protocol=read_json(args.protocol)
    if (start.get('schema')!='wam_v7_robot_start_geometry_manifest_v1'
            or start.get('geometry_preflight_passed') is not True
            or start.get('actual_acquisition_started') is not False
            or start.get('physics_accepted') is not False or start.get('training_accepted') is not False
            or start.get('actual_measured_start_q_rad') is not None or start.get('target')!=args.point
            or start.get('parent_plan_sha256')!=sha(bundle.plan_path)
            or start.get('requested_physics_profile')!={'position_iterations':32,'velocity_iterations':1,'armature_kg_m2':.02}):
        raise ValueError('A passed, still unacquired original-plan .02 geometry case is required')
    q=start.get('requested_start_q_rad')
    if (not isinstance(q,list) or len(q)!=7 or any(type(v) not in (int,float) or not math.isfinite(v) for v in q)
            or max(map(abs,q))<=.06):raise ValueError('Finite nonzero requested start required')
    if (protocol.get('schema')!='wam_v7_frozen_start_family_protocol_v1'
            or protocol.get('actual_start_tolerance_rad')!=.03
            or start.get('proposed_starts_sha256')!=protocol.get('start_spec_sha256')):
        raise ValueError('Frozen start-family experiment protocol differs')
    canonical=(json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if args.protocol.read_bytes()!=canonical:raise ValueError('Canonical frozen protocol bytes required')
    key=start['start_family']+'::'+args.point;case=protocol.get('cases',{}).get(key,{})
    families=[f for f in protocol.get('families',[]) if f.get('family_id')==start['start_family']]
    if (case.get('start_manifest_sha256')!=args.start_manifest_sha256
            or case.get('source_plan_sha256')!=start['parent_plan_sha256']
            or case.get('target')!=args.point or case.get('start_family')!=start['start_family']
            or len(families)!=1 or families[0].get('requested_q_rad')!=q
            or families[0].get('split')!=start.get('split')):
        raise ValueError('Actual start case is not the frozen family/target binding')
    inventory=args.start_manifest.parent/'source_sha256.json'
    if sha(inventory)!=start['source_sha256']:raise ValueError('Geometry source inventory differs')
    # Remap source project files to the actual runtime checkout; original helper
    # paths are also rehashed. No qualification is inferred from this check.
    args.geometry_source_pins={}
    for old,expected in read_json(inventory).items():
        actual=args.source_project/('sim/'+old.split('/sim/',1)[1]) if '/sim/' in old else Path(old)
        if actual.is_symlink() or sha(actual)!=expected:raise ValueError('Geometry source drift: '+str(actual))
        if '/sim/' in old:args.geometry_source_pins['sim/'+old.split('/sim/',1)[1]]=expected
    curve_name=start.get('original_contact_curve_file','')
    if not curve_name or Path(curve_name).name!=curve_name:raise ValueError('Contained contact-curve evidence required')
    curve=args.start_manifest.parent/curve_name
    if sha(curve)!=start['original_contact_curve_sha256']:raise ValueError('Original contact-curve evidence differs')
    limits={'free_clearance_m':.003,'contact_clearance_m':.0003,'joint_edge_sample_step_rad':.015,
        'urdf_planning_margin_deg':3.,'command_velocity_rad_s':.175,'command_acceleration_rad_s2':1.,
        'command_jerk_rad_s3':1.5,'raw_measured_acceleration_rad_s2':2.,'raw_measured_jerk_rad_s3':20.}
    if start.get('limits')!=limits:raise ValueError('Original geometry and dynamics limits differ')
    return start,{'v7/start_manifest':args.start_manifest,'v7/start_protocol':args.protocol,
                 'v7/geometry_source_inventory':inventory,'v7/original_contact_curve':curve}



def verify_imported_geometry_sources(pins):
    """An alternate source-project path cannot mask drift in imported modules."""
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if not name.startswith('sim.') or not path or '/sim/' not in str(path):continue
        relative='sim/'+str(path).split('/sim/',1)[1]
        if relative in pins and sha(path)!=pins[relative]:
            raise ValueError('Actually imported geometry source drift: '+name)


def capture(app,errors,released,args,bundle,snapshot_sha,binding_sha):
    import numpy as np
    from sim.auscultation_repair_v3.physics_profile import make_env,PhysicsProfile
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.controller import verify_snapshot,digest
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from sim.auscultation_pi05_v6.live_capture import LiveCapture
    from sim.auscultation_pi05_v6.collect_demo import teacher_policy_description, SUPERVISOR_VERSION
    from .start_teacher import StartObservedTeacher, CLASS, SOURCE
    verify_imported_geometry_sources(args.geometry_source_pins)
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
        from sim.auscultation_pi05_v6.runtime_dynamics import capture_runtime_contract
        dynamics=capture_runtime_contract(env)
        if args.sensors:
            sensor=LiveCapture(env,kin,args.source_project/'sim/camera_framing_20260923/recommended_camera.json',
                               args.annotation,args.output/'sensors',width=640)
            sensor.warm_up()
        controller=StartObservedTeacher(env,kin,checker,sensor=sensor,
            generator_policy='revised_shortcut',speed_scale=args.speed_scale,
            start_manifest=args.bound_start,start_manifest_sha256=args.start_manifest_sha256,
            experiment_protocol_sha256=args.protocol_sha256,episode_id=args.episode_id)
        outcome=controller.run([(bundle.target,bundle.row)])
        outcome['dynamics_recipe_sha256']=dynamics['recipe_sha256']
        verify_snapshot(args.output/'input_snapshot',verify_sources=True)
        verify_imported_geometry_sources(args.geometry_source_pins)
    except Exception as error:
        outcome.update(success=False,training_accepted=False,failure_reason=type(error).__name__+': '+str(error))
        print('LIVE_EXPERT_FAILURE',outcome['failure_reason'],flush=True)
    finally:
        outcome.update(requested_points=[args.point],input_snapshot='input_snapshot/manifest.json',
            input_snapshot_manifest_sha256=snapshot_sha,collision_regeneration_sha256=binding_sha,
            scope='V7 real measured-start preparation; unchanged V6 ADC/FK/RGB-D teacher; independent qualification pending; audio absent',
            executed_controller_class=CLASS,controller_source=SOURCE,episode_id=args.episode_id,
            start_family=args.bound_start['start_family'],experiment_protocol_sha256=args.protocol_sha256,
            qualification_passed=False,dataset_sealed=False,
            dataset_role='unqualified_live_demonstration' if sensor else 'physics_only_diagnostic_not_training_data',
            sensor_frames=0 if sensor is None else len(sensor.rows),
            body_source='existing_seated_patient_asset_v1',layout_family='approved_farther_lower_seat_camera7deg',
            motion_family='independent_target_plan_'+args.point,source_project=str(args.source_project))
        outcome['repair_contact_policy']=teacher_policy_description()
        outcome['collector_source_hashes']={name:digest(path.read_bytes())
            for name,path in collector_source_paths().items()}
        outcome['observed_supervisor_version']=SUPERVISOR_VERSION
        if env is not None:env.save('expert')
        if controller is not None:
            (args.output/'control_commands.json').write_text(json.dumps(controller.control_rows,allow_nan=False)+'\n')
        if sensor is not None:sensor.release(errors,released)
        if controller is not None:
            try:
                receipt=controller.finalize_start_receipt(args.output/'sensors/frames.jsonl')
                receipt['input_snapshot_manifest_sha256']=snapshot_sha
                receipt['collector_source_hashes']=outcome['collector_source_hashes']
                path=args.output/'measured_start_receipt.json'
                with path.open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
                outcome['measured_start_receipt']=dict(path=path.name,sha256=sha(path))
            except Exception as error:
                outcome.update(success=False,failure_reason='start receipt: '+type(error).__name__+': '+str(error))
                (args.output/'start_preparation_failure.json').write_text(json.dumps(
                    controller.start_receipt,indent=2,allow_nan=False)+'\n')
        (args.output/'outcome.json').write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n')
    return outcome


def main():
    args=argument_parser().parse_args()
    args.source_project=args.source_project.resolve();args.output=args.output.resolve()
    args.start_manifest=args.start_manifest.resolve();args.protocol=args.protocol.resolve()
    args.sensors=True;args.speed_scale=.7;args.position_iterations=32;args.armature=.02
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
    args.bound_start,start_sources=bound_start(args,bundle)
    args.output.mkdir();sources=source_paths(runtime);sources.update(start_sources)
    sources['collision_regeneration']=runtime/'results/collision_regeneration_verification.json'
    for name in ('collect_expert.py','live_capture.py','body_tracking.py','rigid_registration.py','tracking.py','observations.py','hall.py','multimodal.py'):
        sources['vla/'+name]=Path(__file__).parents[1]/'auscultation_vla_v2'/name
    for name in ('collect_repaired_expert.py','physics_profile.py','metrics.py','teardown.py','gentle_contact.py'):
        sources['repair/'+name]=Path(__file__).parents[1]/'auscultation_repair_v3'/name
    sources.update(collector_source_paths())
    for package in ('auscultation_skill_v1','curobo_expert_v1','expert_quality_20260924'):
        imported=importlib.import_module('sim.'+package)
        for directory in imported.__path__:
            for path in Path(directory).glob('*.py'):
                sources['runtime/'+package+'/'+path.name]=path
    calibration_module=importlib.import_module('auscultation.mechanical_sensor.force_feedback.force_sensor')
    sources['vla/force_sensor.py']=Path(calibration_module.__file__)
    sources['vla/hall_calibration.json']=Path(__file__).parents[1]/'auscultation_vla_v2/calibration/veo6_tared_snapshot.json'
    sources['vla/annotation']=args.annotation
    snapshot=create_snapshot([bundle],args.output/'input_snapshot',sources);verify_snapshot(snapshot,verify_sources=True)
    snapshot_sha=digest((snapshot/'manifest.json').read_bytes())
    verify_imported_geometry_sources(args.geometry_source_pins)
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
                      sensor_frames=outcome['sensor_frames'],qualification='V7 physics candidate; independent start/ADC/label qualification pending; not sealed for training',
                      qualification_passed=False,dataset_sealed=False,
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
