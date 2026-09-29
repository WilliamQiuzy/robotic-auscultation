"""Single R4 ACT D0 diagnostic. Approved assets are explicit, code is versioned.

The ACT supplies L2 commands; measured analytic L3 is called only if its unchanged
entry contract accepts. No expert transfer, local alignment or fallback is added.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


@contextmanager
def approved_assets(asset_root):
    # Scoped asset location injection; no editing/symlinking approved source files.
    from sim.auscultation_expert_layout_v1 import scene
    original = scene.ROOT, scene.PROJECT
    scene.ROOT = asset_root/'sim/auscultation_expert_layout_v1'
    scene.PROJECT = asset_root
    try:
        yield scene
    finally:
        scene.ROOT, scene.PROJECT = original


def prepare(config, output):
    """No Kit import or GPU work. Validate original assets, dependencies and plan."""
    asset = Path(config['asset_root']).resolve()
    code = Path(__file__).resolve().parents[2]
    for relative, expected in config['core_assets'].items():
        if sha(code/relative) != expected or sha(asset/relative) != expected:
            raise ValueError('Approved core asset differs: '+relative)
    for name, expected in config['dependency_files'].items():
        if sha(name) != expected:
            raise ValueError('Physical dependency differs: '+name)
    sys.path.append(str(asset))
    for folder in reversed(config['dependency_paths']):
        sys.path.insert(0, folder)
    from sim.auscultation_expert_layout_v1.controller import validate_plan,create_snapshot,verify_snapshot
    from sim.auscultation_expert_layout_v1.revalidate_plans import validate_geometry_binding
    runtime = asset/'sim/auscultation_expert_layout_v1'
    plans = asset/'sim/upper_pair_posture_20260924/results/candidate_b_plans'
    with approved_assets(asset) as scene:
        scene.physical_sources()
        binding = validate_geometry_binding(runtime)
    bundle = validate_plan(plans/f'planned_paths_{config["point"]}.json', plans, runtime)
    if bundle.plan.get('revalidation',{}).get('collision_regeneration_sha256') != binding:
        raise ValueError('Plan collision geometry binding differs')
    if sha(code/'sim/auscultation_expert_layout_v1/openarm_v10_follower_stetho.urdf') != hashlib.sha256(bundle.inputs['kinematics_urdf'][1]).hexdigest():
        raise ValueError('New runtime URDF differs from the approved plan')
    output.mkdir(parents=True, exist_ok=False)
    sources = dict(collision_regeneration=runtime/'results/collision_regeneration_verification.json')
    packages = ('ausc_policy','auscultation_expert_layout_v1','auscultation_skill_v1',
                'auscultation_pi05_v6','auscultation_vla_v2','auscultation_repair_v3',
                'curobo_expert_v1','expert_quality_20260924')
    for package in packages:
        for path in (code/'sim'/package).glob('*.py'):
            sources['code/'+package+'/'+path.name] = path
    for relative in config['core_assets']:
        sources[relative] = code/relative
    calibration = importlib.import_module('auscultation.mechanical_sensor.force_feedback.force_sensor')
    sources['calibration/force_sensor.py'] = Path(calibration.__file__)
    snapshot = create_snapshot([bundle], output/'input_snapshot', sources)
    verify_snapshot(snapshot, verify_sources=True)
    # The legacy monitor has its own static-arm/containment geometry source.
    # Bind that source to the same captured bytes as the dynamic checker.
    geometry = output/'geometry'; geometry.mkdir()
    for label,name in [('collision_metadata','collision_geometry.json'),('collision_arrays','collision_geometry.npz')]:
        value=bundle.inputs[label][1]; source=snapshot/'objects'/hashlib.sha256(value).hexdigest()
        os.link(source,geometry/name)
    import fcl, ruckig
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from sim.curobo_expert_v1.static_world import build_static_world,save_static_world,install_fcl_static_world
    static=build_static_world(geometry_dir=geometry,urdf_path=code/'sim/auscultation_expert_layout_v1/openarm_v10_follower_stetho.urdf')
    (output/'static_world').mkdir(); save_static_world(output/'static_world',static)
    # Exercise both native collision construction and containment source loading before Kit.
    checker=CertifiedCollisionChecker(OpenArmKinematics(),bundle.plan['world_from_left_base'],directory=geometry)
    installation=install_fcl_static_world(checker,static,geometry_dir=geometry)
    from .controller import M0DirectController  # catch pure import defects before starting Kit
    write(output/'preflight.json', dict(passed=True, code_commit=(code/'CODE_COMMIT').read_text().strip(),
        asset_root=str(asset), collision_regeneration_sha256=binding, core_assets=config['core_assets'],
        dependency_files=config['dependency_files'], static_world_installation=installation,m0_passed=False))
    return bundle, sha(snapshot/'manifest.json'), binding


def capture(app, errors, released, config, output, ready, bundle, snapshot_sha, binding):
    import numpy as np
    from sim.auscultation_repair_v3.physics_profile import make_env,PhysicsProfile
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.controller import verify_snapshot
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from sim.auscultation_pi05_v6.live_capture import LiveCapture
    from sim.auscultation_pi05_v6.runtime_dynamics import capture_runtime_contract
    from .act_runtime import ACTClient
    from .controller import M0DirectController
    from sim.curobo_expert_v1.static_world import load_static_world
    code = Path(__file__).resolve().parents[2]
    env = sensor = controller = None
    outcome = dict(success=False,training_accepted=False,points=[],failure_reason='construction incomplete')
    try:
        kin = OpenArmKinematics(); base = np.asarray(bundle.plan['world_from_left_base'])
        with tempfile.TemporaryDirectory(prefix='ausc_m0_geometry_') as temporary:
            folder = Path(temporary)
            for label,name in [('collision_metadata','collision_geometry.json'),('collision_arrays','collision_geometry.npz')]:
                (folder/name).write_bytes(bundle.inputs[label][1])
            checker = CertifiedCollisionChecker(kin,base,directory=folder)
        with approved_assets(Path(config['asset_root'])):
            env = make_env(output, PhysicsProfile(**config['physics']))
        if not np.allclose(env.base,base,atol=1e-7,rtol=0):
            raise ValueError('Actual robot base differs from approved plan')
        dynamics = capture_runtime_contract(env)
        sensor = LiveCapture(env,kin,code/'sim/camera_framing_20260923/recommended_camera.json',
            code/'sim/auscultation_vla_v2/sensor_pilot_annotation.json',output/'sensors',width=640)
        sensor.warm_up()
        client = ACTClient(ready['port'], ready['receipt'], timeout_s=5.)
        controller = M0DirectController(env,kin,checker,sensor=sensor,client=client,
            task_id=output.parent.parent.name,episode_id=output.parent.parent.name+'-R4',
            generator_policy='revised_shortcut',speed_scale=.7,
            motion_velocity=config.get('motion_velocity','zero'),
            posture_hint_preview=config.get('posture_hint_preview',False),
            posture_hint_selection=config.get('posture_hint_selection','first_feasible'),
            static_world_bundle=load_static_world(output/'static_world'),static_geometry_dir=output/'geometry')
        outcome = controller.run([(bundle.target,bundle.row)])
        outcome['dynamics_recipe_sha256'] = dynamics['recipe_sha256']
        verify_snapshot(output/'input_snapshot',verify_sources=True)
    except Exception as error:
        outcome.update(success=False,training_accepted=False,failure_reason=type(error).__name__+': '+str(error))
        print('ACT_D0_FAILURE',outcome['failure_reason'],flush=True)
    finally:
        outcome.update(requested_points=[config['point']],input_snapshot='input_snapshot/manifest.json',
            input_snapshot_manifest_sha256=snapshot_sha,collision_regeneration_sha256=binding,
            scope='state ACT L2 + measured analytic L3; single approved body; no acoustics or generalization claim',
            expert_demonstration=False,training_dataset_eligible=False,training_accepted=False,m0_passed=False,
            sensor_frames=0 if sensor is None else len(sensor.rows),
            closed_loop_evaluated=bool(controller is not None and controller.l2_queries),
            checkpoint_receipt=ready['receipt'],layout_family='approved_farther_lower_seat_camera7deg')
        try:
            if env is not None:
                env.save('expert')  # frozen audit reader filename, not an expert-policy claim
            if controller is not None:
                write(output/'control_commands.json',controller.control_rows)
            write(output/'outcome.json',outcome)
        finally:
            if sensor is not None:
                sensor.release(errors,released)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--ready',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or os.environ.get('AUSC_OWNED_RUN') != '1':
        raise RuntimeError('Isaac entry requires the guarded Linux owner')
    config = json.loads(args.config.read_text())
    from .isaac_run import validate_config
    expected = validate_config(config)
    bundle,snapshot_sha,binding = prepare(config,args.output)
    if args.preflight_only:
        print('ACT_D0_PREFLIGHT_OK',flush=True); return 0
    ready = json.loads(args.ready.read_text())
    if ready['host'] != '127.0.0.1' or ready['receipt'] != expected:
        raise ValueError('Runtime service does not match CPU reload evidence')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless':True,'fast_shutdown':False})
    from sim.auscultation_skill_v1.media_teardown import run_scoped_capture,require_complete_teardown
    report = dict(execution_completed=False,case_passed=False,m0_passed=False,closed_loop_evaluated=False)
    try:
        outcome,receipt = run_scoped_capture(app,capture,config,args.output,ready,bundle,snapshot_sha,binding)
        write(args.output/'teardown.json',receipt)
        require_complete_teardown(receipt,annotators=2,products=1)
        from sim.auscultation_skill_v1.audits import audit_saved
        from .evaluation_audit import interpret
        from sim.auscultation_repair_v3.metrics import motion_metrics
        original_audit = audit_saved(args.output)
        audit = interpret(original_audit,outcome)
        write(args.output/'evaluation_audit.json',audit)
        rows = [json.loads(line) for line in (args.output/'expert_trace.jsonl').open()]
        if len(rows)<9:
            raise RuntimeError('Insufficient executed physics; original construction/controller failure: '+str(outcome.get('failure_reason')))
        dynamics = motion_metrics([r['time_s'] for r in rows],[r['q'] for r in rows],[r['dq'] for r in rows])
        write(args.output/'measured_dynamics.json',dynamics)
        smooth = dynamics['centered']['acceleration_rad_s2']['max'] <= 2 and dynamics['centered']['jerk_rad_s3']['max'] <= 20
        report.update(execution_completed=True,case_passed=bool(outcome['success'] and audit['accepted'] and smooth),
            closed_loop_evaluated=outcome['closed_loop_evaluated'],failure_reason=outcome.get('failure_reason'),
            physics_audit_accepted=audit['accepted'],smoothness_target_met=smooth,
            sensor_frames=outcome['sensor_frames'],qualification='single-body D0 diagnostic only')
    except Exception as error:
        report.update(error=type(error).__name__,reason=str(error))
    write(args.output/'result.json',report)
    print('ACT_D0_RESULT',json.dumps(report),flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    # Kit's explicit cleanup already returned; avoid a second atexit native teardown.
    os._exit(0 if report['execution_completed'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
