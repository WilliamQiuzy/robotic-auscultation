"""Isolated parent/worker Isaac entry point for the versioned contact skill."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .provenance import ROOT, PROJECT, implementation_sources, bind_manifest, verify_candidate_sources, sha256_file

PLANS = PROJECT/'sim/upper_pair_posture_20260924/results/candidate_b_plans'
RUNTIME = PROJECT/'sim/auscultation_expert_layout_v1'
POINTS = [side+str(level) for level in range(1,6) for side in ('R','L')]


@contextmanager
def graceful_simulation_shutdown(isaac_module):
    """Kit's fast shutdown exits Python before post-close evidence can be saved."""
    original=isaac_module.SimulationApp
    lifecycle={'apps_created':0}
    def create(launch_config=None, *args, **kwargs):
        configuration=dict(launch_config or {})
        configuration['fast_shutdown']=False
        app=original(configuration,*args,**kwargs)
        lifecycle['apps_created']+=1
        return app
    isaac_module.SimulationApp=create
    try:
        yield lifecycle
    finally:
        isaac_module.SimulationApp=original


def bind_actual_timing(outcome):
    """The inherited entry point overwrites variation after controller.run()."""
    policy = outcome.get('actual_contact_policy')
    if policy is None: return outcome  # Incomplete construction remains a failed run.
    variation = dict(outcome.get('variation', {}))
    approach = abs(float(policy['teacher_approach_speed_m_s']))
    variation.update(free_approach_far_speed_m_s=approach,
        free_approach_near_speed_m_s=approach,
        retraction_speed_m_s=float(policy['teacher_unload_speed_m_s']),
        fault_response='simulation stopped without alternate motion path',
        scope='fixed free-space timing and declared shared contact executor; actor conditions recorded separately')
    variation.pop('fault_retraction_speed_m_s', None)
    outcome['variation'] = variation
    return outcome


def verify_worker_completion(output, returncode):
    """Saved trajectories alone do not prove the worker exited normally."""
    output=Path(output); reasons=[]; hashes={}
    try:
        path=output/'worker_completion.json'
        expected=sha256_file(path); receipt=json.loads(path.read_text())
        outcome=json.loads((output/'outcome.json').read_text())
        if (receipt.get('schema')!='auscultation_skill_worker_completion_v1'
                or type(receipt.get('returncode')) is not int or type(returncode) is not int
                or returncode not in (0,1) or receipt['returncode']!=returncode):
            raise ValueError('worker did not return normally with its recorded code')
        if (receipt.get('close_returned') is not True
                or receipt.get('completion_stage')!='post_app_close_pre_process_exit'
                or receipt.get('exit_policy')!='controlled_os_exit_after_close'):
            raise ValueError('worker lacks explicit post-close controlled-exit evidence')
        if receipt.get('reset_id')!=outcome.get('reset_id') or not outcome.get('reset_id'):
            raise ValueError('worker reset identity mismatch')
        if receipt.get('outcome_sha256')!=sha256_file(output/'outcome.json'):
            raise ValueError('worker outcome hash mismatch')
        snapshot=output/'input_snapshot/manifest.json'
        if (receipt.get('input_snapshot_manifest_sha256')!=sha256_file(snapshot)
                or receipt['input_snapshot_manifest_sha256']!=outcome.get('input_snapshot_manifest_sha256')):
            raise ValueError('worker snapshot hash mismatch')
        if sha256_file(path)!=expected: raise ValueError('worker completion changed during verification')
        hashes['worker_completion.json']=expected
    except (OSError,ValueError,KeyError,TypeError) as error:
        reasons.append(str(error))
    return dict(status='failed' if reasons else 'passed', returncode=returncode,
                evidence_sha256=hashes, reasons=reasons)


def selected_condition(path, condition_id, speed_scale):
    if condition_id is None:
        return dict(condition_id='nominal_qualification', family_id='nominal_qualification',
            split='qualification', speed_scale=speed_scale, hall_noise_std_N=0.,
            hall_noise_clip_N=0., observation_delay_steps=0, seed=0)
    document = json.loads(Path(path).read_text())
    records = document['conditions'] if isinstance(document, dict) else document
    if isinstance(records, dict):
        if condition_id not in records: raise ValueError('unknown condition: '+condition_id)
        condition = dict(records[condition_id])
        condition.setdefault('condition_id', condition_id)
    else:
        matches = [r for r in records if r['condition_id']==condition_id]
        if len(matches)!=1: raise ValueError('unknown or repeated condition: '+condition_id)
        condition = dict(matches[0])
    return condition


def worker(args, condition):
    from .contract import verify_contract
    verify_contract(args.contract)
    from sim.upper_pair_posture_20260924.run import verify_selection
    selection_path, selection = verify_selection(args.input_dir)
    from sim.auscultation_expert_layout_v1 import run_sequence, collision
    from .audits import audit_saved
    from .certified_collision import CertifiedCollisionChecker
    from .controller import ExpertController
    controller_factory = partial(ExpertController, generator_policy='revised_shortcut', condition=condition)
    extras = {'run_contract.json': args.contract, 'selection.json': selection_path}
    if args.condition is not None: extras['run_conditions.json'] = args.conditions
    if args.mode != 'expert':
        from .student import StudentController, load_negative_control
        from .policy import load_policy
        if args.model is None: raise ValueError('student and negative controls require an explicit captured model')
        model_manifest = sha256_file(args.model/'run_manifest.json')
        loader = load_policy if args.mode == 'policy' else load_negative_control
        actor = loader(args.model, expected_manifest_sha256=model_manifest,
                       expected_contract_sha256=sha256_file(args.contract), device='cpu')
        extras.update({'learning/checkpoint': args.model/'policy.pt',
                       'learning/manifest': args.model/'run_manifest.json'})
        controller_factory = partial(StudentController, mode=args.mode, actor=actor,
                                     generator_policy='revised_shortcut', condition=condition)
    for i, relative in enumerate(sorted(selection['source_sha256'])):
        extras['selection/input_'+str(i)] = PROJECT/relative
    original_sources, original_checker = run_sequence.source_paths, collision.CollisionChecker
    sources = implementation_sources(extras)
    run_sequence.source_paths = lambda root: {**original_sources(root), **sources}
    collision.CollisionChecker = CertifiedCollisionChecker
    selected_factory = controller_factory
    runtime_checker = {}

    def checked_controller_factory(env, kin, checker, **options):
        if type(checker) is not CertifiedCollisionChecker:
            raise ValueError('runtime collision checker differs from the qualified implementation')
        runtime_checker['instance'] = checker
        return selected_factory(env, kin, checker, **options)
    reset_id = str(uuid.uuid4())

    def audit(output):
        path = Path(output)/'outcome.json'
        outcome = json.loads(path.read_text())
        bind_actual_timing(outcome)
        actual_checker = runtime_checker.get('instance')
        if actual_checker is not None:
            outcome['collision_backend'] = dict(
                implementation=type(actual_checker).__module__+'.'+type(actual_checker).__name__,
                numeric_guard_m=actual_checker.numeric_guard_m,
                native_distance_calls=actual_checker.total_exact_distance_calls,
                conservative_distance_reuses=actual_checker.total_certificate_reuses,
                geometry_policy='frozen source meshes with the original static-envelope installer and containment wrapper',
                scope='all requested discrete AABB checks retained; only sufficiently separated native mesh-distance calls reused; no continuous swept-volume claim')
        outcome.update(reset_id=reset_id, worker_pid=os.getpid(),
            simulation_lifecycle_config={'fast_shutdown':False,'receipt_after_close':True},
            dataset_role='expert_demonstration' if args.mode=='expert' else 'student_evaluation',
            expert_demonstration=args.mode=='expert', run_mode=args.mode,
            condition_id=condition['condition_id'], family_id=condition['family_id'],
            condition_parameters=condition,
            seed_semantics='actor observation perturbation RNG only; fresh process/environment reset',
            controller_scope='learnable scalar contact motion; checked free-space planner and phase supervisor')
        path.write_text(json.dumps(outcome, indent=2, allow_nan=False)+'\n')
        bind_manifest(output)
        return audit_saved(output)

    forwarded = ['--points', *args.points, '--input-dir', str(args.input_dir),
        '--runtime-dir', str(RUNTIME), '--output', str(args.output),
        '--speed-scale', str(condition['speed_scale']), '--seed', str(condition['seed'])]
    try:
        import isaacsim
        with graceful_simulation_shutdown(isaacsim) as lifecycle:
            code=run_sequence.main(forwarded, controller_class=checked_controller_factory, result_auditor=audit)
        if type(code) is not int or code not in (0,1) or lifecycle['apps_created']!=1:
            raise RuntimeError('worker did not return through exactly one initialized SimulationApp')
        completion=dict(schema='auscultation_skill_worker_completion_v1',returncode=code,
            close_returned=True,completion_stage='post_app_close_pre_process_exit',
            exit_policy='controlled_os_exit_after_close',
            reset_id=reset_id,outcome_sha256=sha256_file(args.output/'outcome.json'),
            input_snapshot_manifest_sha256=sha256_file(args.output/'input_snapshot/manifest.json'),
            completed_utc=datetime.now(timezone.utc).isoformat(),
            meaning='run_sequence returned after audit save and SimulationApp.close; wrapper exits with recorded OS code after flushing streams',
            process_exit_policy='explicit os._exit after Kit close and final receipt; skips hanging Python interpreter finalizers')
        with (args.output/'worker_completion.json').open('x') as stream:
            stream.write(json.dumps(completion,indent=2,allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        return code
    finally:
        run_sequence.source_paths, collision.CollisionChecker = original_sources, original_checker


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, default=ROOT/'contract.json')
    parser.add_argument('--input-dir', type=Path, default=PLANS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--points', nargs='+', default=POINTS)
    parser.add_argument('--conditions', type=Path, default=ROOT/'conditions.json')
    parser.add_argument('--condition')
    parser.add_argument('--mode', choices=('expert','policy','zero','untrained'), default='expert')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--speed-scale', type=float, default=1.)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--timeout', type=float, default=7200.)
    args = parser.parse_args(argv)
    for key in ('contract','input_dir','output','conditions'):
        setattr(args, key, getattr(args,key).resolve())
    if args.model is not None: args.model = args.model.resolve()
    if args.mode!='expert' and args.model is None: parser.error('--model is required for policy and negative controls')
    if not args.points or len(set(args.points))!=len(args.points) or any(p not in POINTS for p in args.points):
        parser.error('points must be distinct R1/L1 through R5/L5')
    condition = selected_condition(args.conditions, args.condition, args.speed_scale)
    from .contract import verify_contract, validate_outcome_contract
    verify_contract(args.contract)
    if args.worker:
        code=worker(args,condition)
        from .runtime import exit_after_cleanup
        exit_after_cleanup(code)
    if args.output.exists(): parser.error('output must be a new directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix('.log')
    with log.open('x') as stream:
        child=subprocess.Popen([sys.executable, '-B', '-m', 'sim.auscultation_skill_v1.run',
            '--worker', *argv], cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code=child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            from .runtime import stop_process_tree
            stop_process_tree(child); code=None
    from sim.auscultation_expert_layout_v1.run_low_elbow import verify_saved
    from sim.auscultation_expert_layout_v1.controller import verify_snapshot
    report = verify_saved(args.output, code)
    lifecycle=verify_worker_completion(args.output,code)
    report['worker_lifecycle']=lifecycle
    report.setdefault('evidence_sha256',{}).update(lifecycle['evidence_sha256'])
    if lifecycle['status']!='passed':
        report['accepted']=False
        report.setdefault('reasons',[]).append('worker_lifecycle:'+str(lifecycle['reasons']))
    try:
        verify_snapshot(args.output/'input_snapshot', verify_sources=True)
        verify_contract(args.contract)
        outcome = json.loads((args.output/'outcome.json').read_text())
        contract = json.loads(args.contract.read_text())
        validate_outcome_contract(outcome, contract, args.points, speed_scale=condition['speed_scale'])
        binding = verify_candidate_sources(args.output, outcome)
        report['new_source_binding'] = binding
        if binding['status']!='passed': raise ValueError(str(binding['reasons']))
        if outcome['condition_parameters']!=condition: raise ValueError('condition differs from request')
    except (OSError, ValueError, KeyError, TypeError) as error:
        report['accepted'] = False
        report.setdefault('reasons', []).append('contact_skill_contract:'+str(error))
    destination = args.output/'runner_summary.json' if args.output.exists() else log.with_suffix('.runner.json')
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(accepted=report['accepted'], reasons=report.get('reasons',[]),
                         report=str(destination))), flush=True)
    return 0 if report['accepted'] else 1


if __name__=='__main__': raise SystemExit(main())
