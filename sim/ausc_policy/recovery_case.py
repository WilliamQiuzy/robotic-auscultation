"""One isolated actual Isaac stationary-recovery diagnostic, never a VLA test."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .isaac_case import approved_assets, prepare, write


def geometry_preflight(bundle, output):
    """Prospective states only; no measured-state or physics qualification claim."""
    import numpy as np
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureChecker, PostureLimits
    from sim.auscultation_expert_layout_v1.plan_low_elbow import checked_transfer
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from sim.curobo_expert_v1.static_world import load_static_world, install_fcl_static_world
    from .recovery_oracle import StationaryRecoveryOracle
    from .recovery_run import CASES
    kin = OpenArmKinematics(); base = np.asarray(bundle.plan['world_from_left_base'])
    checker = CertifiedCollisionChecker(kin,base,directory=output/'geometry')
    install_fcl_static_world(checker,load_static_world(output/'static_world'),geometry_dir=output/'geometry')
    limits = PostureLimits.from_dict(bundle.row['posture_limits'],kin)
    checker = PostureChecker(checker,base,limits)
    oracle = StationaryRecoveryOracle(checker,lambda start,goal,rng,deadline:
        checked_transfer(checker,limits,start,goal,kin,base,rng,deadline),
        lower=limits.joint_lower_rad,upper=limits.joint_upper_rad)
    goal = np.asarray(bundle.row['transfer_q'][-1]); cases = []
    for name,offset in CASES:
        q = goal.copy(); q[2] += offset
        query = dict(q=q.tolist(),dq=[0.]*7,command_history=[q.tolist()]*3,hall_load_N=0.,
            physical_contact_N=0.,compression_m=0.,time_s=0.,physics_row=0)
        result = oracle.query(query,dict(target='R4',phase='APPROACH',revision=0,q_goal=goal.tolist()))
        write(output/(name+'-prospective-query.json'),result)
        cases.append(dict(case_id=name,kind=result['kind'],reason=result.get('reason'),wall_time_s=result['wall_time_s']))
    report = dict(passed=all(c['kind']=='trajectory' for c in cases),cases=cases,
        scope='Prospective stationary geometry only, no acquired observations or physical success')
    write(output/'recovery_geometry_preflight.json',report)
    if not report['passed']: raise ValueError('Prospective recovery geometry refused; see case receipts')


def capture(app, errors, released, config, output, bundle, snapshot_sha, binding):
    import numpy as np
    from sim.auscultation_repair_v3.physics_profile import make_env, PhysicsProfile
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.controller import verify_snapshot
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from sim.auscultation_pi05_v6.live_capture import LiveCapture
    from sim.auscultation_pi05_v6.runtime_dynamics import capture_runtime_contract
    from sim.curobo_expert_v1.static_world import load_static_world
    from .recovery_teacher import RecoveryTeacher
    code = Path(__file__).resolve().parents[2]
    env = sensor = controller = None
    outcome = dict(success=False,training_accepted=False,points=[],failure_reason='construction incomplete')
    try:
        kin = OpenArmKinematics(); base = np.asarray(bundle.plan['world_from_left_base'])
        with tempfile.TemporaryDirectory(prefix='ausc_recovery_geometry_') as temporary:
            folder = Path(temporary)
            for label,name in [('collision_metadata','collision_geometry.json'),('collision_arrays','collision_geometry.npz')]:
                (folder/name).write_bytes(bundle.inputs[label][1])
            checker = CertifiedCollisionChecker(kin,base,directory=folder)
        with approved_assets(Path(config['asset_root'])):
            env = make_env(output,PhysicsProfile(**config['physics']))
        if not np.allclose(env.base,base,atol=1e-7,rtol=0):
            raise ValueError('Actual robot base differs from approved plan')
        dynamics = capture_runtime_contract(env)
        sensor = LiveCapture(env,kin,code/'sim/camera_framing_20260923/recommended_camera.json',
            code/'sim/auscultation_vla_v2/sensor_pilot_annotation.json',output/'sensors',width=640)
        sensor.warm_up()
        controller = RecoveryTeacher(env,kin,checker,sensor=sensor,perturbation_rad=config['perturbation_rad'],
            generator_policy='revised_shortcut',speed_scale=.7,
            static_world_bundle=load_static_world(output/'static_world'),static_geometry_dir=output/'geometry')
        outcome = controller.run([(bundle.target,bundle.row)])
        outcome['dynamics_recipe_sha256'] = dynamics['recipe_sha256']
        verify_snapshot(output/'input_snapshot',verify_sources=True)
    except Exception as error:
        outcome.update(success=False,training_accepted=False,failure_reason=type(error).__name__+': '+str(error))
        print('RECOVERY_FAILURE',outcome['failure_reason'],flush=True)
    finally:
        outcome.update(requested_points=['R4'],input_snapshot='input_snapshot/manifest.json',
            input_snapshot_manifest_sha256=snapshot_sha,collision_regeneration_sha256=binding,
            expert_demonstration=True,learned_policy=False,training_dataset_eligible=False,
            training_accepted=False,m0_passed=False,sensor_frames=0 if sensor is None else len(sensor.rows),
            layout_family='approved_farther_lower_seat_camera7deg',case_id=config['case_id'])
        if controller is not None:
            outcome['oracle_receipt'] = controller.oracle_receipt
        try:
            if env is not None: env.save('expert')
            if controller is not None: write(output/'control_commands.json',controller.control_rows)
            write(output/'outcome.json',outcome)
        finally:
            if sensor is not None: sensor.release(errors,released)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or os.environ.get('AUSC_OWNED_RUN') != '1':
        raise RuntimeError('Original guarded Linux owner required')
    config = json.loads(args.config.read_text())
    from .recovery_run import validate_case
    validate_case(config)
    bundle,snapshot_sha,binding = prepare(config,args.output)
    from .recovery_teacher import RecoveryTeacher  # import before Kit; no construction
    if args.preflight_only:
        geometry_preflight(bundle,args.output)
        print('RECOVERY_PREFLIGHT_OK',flush=True); return 0
    from isaacsim import SimulationApp
    app = SimulationApp({'headless':True,'fast_shutdown':False})
    from sim.auscultation_skill_v1.media_teardown import run_scoped_capture,require_complete_teardown
    report = dict(execution_completed=False,case_passed=False,m0_passed=False,training_dataset_eligible=False)
    try:
        outcome,receipt = run_scoped_capture(app,capture,config,args.output,bundle,snapshot_sha,binding)
        write(args.output/'teardown.json',receipt)
        require_complete_teardown(receipt,annotators=2,products=1)
        from sim.auscultation_skill_v1.audits import audit_saved
        from .evaluation_audit import interpret
        from sim.auscultation_repair_v3.metrics import motion_metrics
        original = audit_saved(args.output)
        audit = interpret(original,outcome,expert_evaluation=True)
        write(args.output/'evaluation_audit.json',audit)
        from sim.auscultation_pi05_v6.audit_observed_supervisor import audit_episode
        replay = audit_episode(args.output)
        write(args.output/'observed_supervisor_replay.json',replay)
        rows = [json.loads(line) for line in (args.output/'expert_trace.jsonl').open()]
        if len(rows)<9: raise RuntimeError('Insufficient executed physics')
        dynamics = motion_metrics([r['time_s'] for r in rows],[r['q'] for r in rows],[r['dq'] for r in rows])
        write(args.output/'measured_dynamics.json',dynamics)
        smooth = dynamics['centered']['acceleration_rad_s2']['max'] <= 2 and dynamics['centered']['jerk_rad_s3']['max'] <= 20
        report.update(execution_completed=True,case_passed=bool(outcome['success'] and audit['accepted'] and smooth and replay['passed']),
            physics_audit_accepted=audit['accepted'],smoothness_target_met=smooth,
            observed_supervisor_replay_passed=replay['passed'],
            failure_reason=outcome.get('failure_reason'),oracle=outcome.get('oracle_receipt'),
            sim_duration_s=rows[-1]['time_s']-rows[0]['time_s'],physics_steps=len(rows),sensor_frames=outcome['sensor_frames'],
            qualification='Stationary expert pilot only; raw modality replay and dataset qualification pending')
    except Exception as error:
        report.update(error=type(error).__name__,reason=str(error))
    write(args.output/'result.json',report)
    print('RECOVERY_RESULT',json.dumps(report),flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if report['execution_completed'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
