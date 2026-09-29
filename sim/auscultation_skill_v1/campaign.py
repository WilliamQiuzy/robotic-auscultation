"""Sequential, fail-closed simulation campaigns on the authorized Linux host."""
import argparse
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
import subprocess

from .config import TARGETS, PROJECT, ROOT
from .contract import verify_contract
from .run import selected_condition
from .runtime import stop_process_tree
from .provenance import sha256_file


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def trial(directory, name, launcher, *, mode='expert', condition=None, model=None,
          repetition_report=None, points=TARGETS, timeout=7200):
    """Fresh worker per episode. A failed trial is preserved and never resumed."""
    episode=directory/'episodes'/name
    quality_path=directory/'quality'/(name+'.json')
    episode.parent.mkdir(parents=True,exist_ok=True)
    command=['bash', str(launcher), '-m', 'sim.auscultation_skill_v1.run', '--output',str(episode),
             '--points',*points,'--mode',mode,'--timeout',str(timeout-30)]
    if condition is not None: command.extend(('--condition',condition))
    if model is not None: command.extend(('--model',str(model)))
    receipt=dict(name=name, mode=mode, condition_id=condition or 'nominal_qualification',
        requested_points=list(points), episode=str(episode), quality_report=str(quality_path),
        started_utc=datetime.now(timezone.utc).isoformat(), command=command)
    receipt_path=directory/(name+'.trial.json'); write(receipt_path,receipt)
    with (directory/(name+'.launcher.log')).open('x') as log:
        child=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
        try:
            code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process_tree(child); code=None
    receipt['returncode']=code
    try:
        runner=json.loads((episode/'runner_summary.json').read_text())
        lifecycle=runner.get('worker_lifecycle',{})
        lifecycle_passed=lifecycle.get('status')=='passed' and lifecycle.get('returncode')==runner.get('child_returncode')
        receipt.update(runner_accepted=runner.get('accepted') is True,
                       worker_lifecycle_passed=lifecycle_passed)
    except (OSError,ValueError,TypeError):
        receipt.update(runner_accepted=False,worker_lifecycle_passed=False)
    try:
        from .quality import audit_episode
        quality=audit_episode(episode,quality_path,repetition_report=repetition_report)
        receipt.update(episode_simulation_gates_passed=quality['episode_simulation_gates_passed'],
            sim_control_imitation_eligible=quality['sim_control_imitation_eligible'],
            hard_failures=quality['hard_failures'], quality_failures=quality['failures'])
        if mode!='expert':
            from .evaluate import audit_episode as evaluate
            evaluation_path=directory/'evaluations'/(name+'.json')
            result=evaluate(episode,quality_path,model,mode=mode,output=evaluation_path)
            receipt.update(evaluation_path=str(evaluation_path),evaluation=result)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,pickle.UnpicklingError,EOFError) as error:
        receipt.update(episode_simulation_gates_passed=False,sim_control_imitation_eligible=False,
                       evidence_error=str(error))
    receipt['completed_utc']=datetime.now(timezone.utc).isoformat()
    write(receipt_path,receipt)
    print(json.dumps({key:receipt.get(key) for key in
        ('name','returncode','episode_simulation_gates_passed','quality_failures','evidence_error')}),flush=True)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('qualify','collect','validate','evaluate','negative'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--launcher',type=Path,default=Path('/home/william/wam-sim/run_isaac_l40s.sh'))
    parser.add_argument('--repetition-report',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--conditions',nargs='+')
    parser.add_argument('--points',nargs='+',default=list(TARGETS))
    parser.add_argument('--mode',choices=('zero','untrained'),default='zero')
    parser.add_argument('--timeout-per-trial',type=int,default=7200)
    args=parser.parse_args(argv)
    policy_stage=args.stage in ('validate','evaluate')
    verify_contract()
    if args.stage=='collect' and args.repetition_report is None:
        parser.error('collection requires a verified repetition report')
    if (policy_stage or args.stage=='negative') and args.model is None:
        parser.error('evaluation requires a frozen model or negative-control model')
    if args.stage in ('qualify','collect') and args.model is not None:
        parser.error('expert qualification and collection do not accept a learned model')
    if args.stage=='qualify' and args.points!=list(TARGETS):
        parser.error('qualification requires full ten points')
    if args.stage=='collect':
        from .quality import verify_repetitions
        verify_repetitions(args.repetition_report)
    pinned_model=None
    if args.model is not None:
        args.model=args.model.resolve()
        pinned_model=dict(checkpoint_sha256=sha256_file(args.model/'policy.pt'),
                          model_manifest_sha256=sha256_file(args.model/'run_manifest.json'))
    frozen_conditions=json.loads((ROOT/'conditions.json').read_text())['conditions']
    condition_splits={'collect':('train','validation'),'validate':('validation',)}.get(args.stage,('test',))
    conditions=args.conditions or [r['condition_id'] for r in frozen_conditions if r['split'] in condition_splits]
    if args.stage in ('collect','validate','evaluate'):
        if not conditions or len(conditions)!=len(set(conditions)):
            parser.error('requested conditions must be distinct and nonempty')
        for identifier in conditions:
            record=selected_condition(ROOT/'conditions.json',identifier,1.)
            if record['split'] not in condition_splits: parser.error('condition belongs to another experiment split')
    directory=args.output.resolve(); directory.mkdir(parents=True,exist_ok=False)
    receipts=[]
    if args.stage=='qualify': cases=[('nominal_'+str(i).zfill(2),None) for i in range(1,4)]
    elif args.stage=='negative': cases=[('negative_'+args.mode,conditions[0])]
    else: cases=[(condition,condition) for condition in conditions]
    for name,condition in cases:
        if pinned_model is not None and (
                sha256_file(args.model/'policy.pt')!=pinned_model['checkpoint_sha256']
                or sha256_file(args.model/'run_manifest.json')!=pinned_model['model_manifest_sha256']):
            write(directory/'input_binding_failure.json',dict(reason='frozen model changed between trials',
                                                             next_trial=name,pinned_model=pinned_model))
            break
        receipt=trial(directory,name,args.launcher,mode=args.mode if args.stage=='negative' else
            'policy' if policy_stage else 'expert',condition=condition,model=args.model,
            repetition_report=args.repetition_report,points=args.points,timeout=args.timeout_per_trial)
        receipts.append(receipt); write(directory/'progress.json',dict(stage=args.stage,trials=receipts,pinned_model=pinned_model))
        if args.stage in ('qualify','collect') and (receipt['returncode']!=0 or not receipt['runner_accepted']
                or not receipt['worker_lifecycle_passed'] or not receipt['episode_simulation_gates_passed']):
            return 1  # Do not spend further episodes on an unqualified expert.
    model_unchanged=pinned_model is None or (
        sha256_file(args.model/'policy.pt')==pinned_model['checkpoint_sha256']
        and sha256_file(args.model/'run_manifest.json')==pinned_model['model_manifest_sha256'])
    if args.stage=='qualify':
        from .quality import qualify_repetitions
        qualify_repetitions([r['quality_report'] for r in receipts],directory/'repetitions.json')
    elif args.stage=='collect':
        from .dataset import build_dataset
        entries=[{'episode_dir':r['episode'],'quality_report':r['quality_report']} for r in receipts]
        write(directory/'entries.json',entries)
        build_dataset(entries,directory/'dataset')
    elif policy_stage:
        from .evaluate import summarize_trials
        results=[r['evaluation'] for r in receipts if 'evaluation' in r
                 and r['returncode']==0 and r['runner_accepted'] and r['worker_lifecycle_passed']]
        rejected_model_conditions=[r.get('condition_id') for r in results
            if any(r.get(key)!=value for key,value in pinned_model.items())]
        results=[r for r in results if all(r.get(key)==value for key,value in pinned_model.items())]
        summary=summarize_trials(results,conditions,required_points=args.points,expected_model=pinned_model)
        summary.update(stage=args.stage,evaluation_split=condition_splits[0],condition_ids=conditions,
            evaluation_role='development_validation' if args.stage=='validate' else 'heldout_test',
            heldout_test_evaluation=args.stage=='evaluate',
            scope=('development validation conditions; results may guide model development; '
                   if args.stage=='validate' else 'heldout test conditions; ') +
                  'missing, rejected or incomplete requested trials remain failures in the fixed denominator')
        summary['rejected_model_conditions']=rejected_model_conditions
        if not model_unchanged:
            summary.update(input_binding_passed=False,learned_successes=0,learned_success_rate=0.,
                           assisted_successes=0,assisted_success_rate=0.,
                           invalidation_reason='frozen model changed during campaign')
        else: summary['input_binding_passed']=True
        write(directory/'summary.json',summary)
        return 0 if summary['learned_successes']==summary['requested_trials'] else 1
    elif args.stage=='negative':
        if not receipts: return 1
        result=receipts[0].get('evaluation',{})
        process=receipts[0]['returncode']==1 and receipts[0]['worker_lifecycle_passed']
        return 0 if model_unchanged and process and result.get('negative_control_diagnostic_passed') is True else 1
    return 0


if __name__=='__main__': raise SystemExit(main())
