"""Build a compact experiment table from existing run receipts; never run a model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = ('run_id', 'experiment', 'mode', 'action_representation', 'seed',
          'code_commit', 'config_sha256', 'data_manifest_sha256', 'train_episodes',
          'completed', 'exit_code', 'gpu_released', 'optimizer_updates',
          'median_sample_mae_rad', 'hold_median_sample_mae_rad',
          'cv_median_sample_mae_rad', 'case_passed', 'm0_passed', 'closed_loop_evaluated',
          'expert_cases', 'expert_cases_passed', 'learned_policy', 'training_dataset_eligible',
          'started_utc', 'ended_utc', 'failure')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def record(folder):
    result = read(folder / 'result.json')
    run = result or read(folder / 'run.json')
    if not run:
        return None
    config = run['config']
    output = folder / 'output'
    complete = read(output / 'COMPLETE.json')
    stage = result.get('stage', {})
    step = complete.get('checkpoint_step')
    metrics = read(output / f'checkpoint_{step:04d}' / 'metrics.json') if isinstance(step, int) else {}
    m = metrics.get('metrics', {})
    failure = read(output / 'FAILED.json')
    expert_cases = complete.get('cases', []) if config.get('experiment') == 'recovery_run' else None
    case_passed = complete.get('case_passed')
    if expert_cases is not None:
        case_passed = complete.get('all_cases_passed')
    case_failures = '|'.join(f"{case['case_id']}: {case['failure_reason']}"
        for case in (expert_cases or []) if case.get('failure_reason'))
    return dict(run_id=run['run_id'], experiment=config.get('experiment'), mode=config.get('mode'),
        action_representation=config.get('action_representation'), seed=config.get('seed'),
        code_commit=run.get('code_commit'), config_sha256=run.get('config_sha256'),
        data_manifest_sha256=config.get('manifest_sha256'),
        train_episodes='|'.join(config.get('train_episode_ids', [])),
        completed=result.get('completed', False), exit_code=stage.get('exit_code'),
        gpu_released=result.get('gpu_released'),
        optimizer_updates=complete.get('steps', complete.get('optimizer_updates')),
        median_sample_mae_rad=m.get('act', {}).get('median_sample_mae_rad'),
        hold_median_sample_mae_rad=m.get('hold', {}).get('median_sample_mae_rad'),
        cv_median_sample_mae_rad=m.get('constant_velocity', {}).get('median_sample_mae_rad'),
        case_passed=case_passed,m0_passed=complete.get('m0_passed'), closed_loop_evaluated=complete.get('closed_loop_evaluated'),
        expert_cases=len(expert_cases) if expert_cases is not None else None,
        expert_cases_passed=sum(case.get('case_passed') is True for case in expert_cases) if expert_cases is not None else None,
        learned_policy=complete.get('learned_policy'),training_dataset_eligible=complete.get('training_dataset_eligible'),
        started_utc=run.get('started_utc'), ended_utc=result.get('ended_utc'),
        failure=failure.get('reason') or complete.get('failure_reason') or case_failures or '|'.join(map(str, stage.get('errors', []))))


def build(root):
    rows = [record(folder) for folder in sorted(root.iterdir()) if folder.is_dir()]
    rows = sorted((row for row in rows if row), key=lambda row: row['started_utc'] or '')
    temporary = root / 'summary.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / 'summary.csv')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.runs)
    print(json.dumps({'runs': len(rows), 'summary': str(args.runs / 'summary.csv')}))


if __name__ == '__main__':
    main()
