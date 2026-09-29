"""Single GPU experiment owner, with existing resource and process-group guard."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


GUARD_SHA = '48c47cf0d6d8f3d770028931a61d2882c5281684053a6149dacfcbe7b2483ff2'
MODULES = {'seed_probe': 'sim.ausc_policy.seed_probe',
           'act_baseline': 'sim.ausc_policy.act_baseline',
           'act_runtime': 'sim.ausc_policy.act_runtime',
           'executor_probe': 'sim.ausc_policy.executor_probe',
           'motion_preview_probe': 'sim.ausc_policy.motion_preview_probe',
           'path_probe': 'sim.ausc_policy.path_probe',
           'feedback_probe': 'sim.ausc_policy.feedback_probe',
           'recovery_run': 'sim.ausc_policy.recovery_run',
           'recovery_sensor_audit': 'sim.ausc_policy.recovery_sensor_audit',
           'moving_brake_probe': 'sim.ausc_policy.moving_brake_probe',
           'capture_probe': 'sim.ausc_policy.capture_probe',
           'history_prepare': 'sim.ausc_policy.history_prepare',
           'history_fit': 'sim.ausc_policy.history_fit',
           'history_reload_probe': 'sim.ausc_policy.history_reload_probe',
           'history_runtime_prepare': 'sim.ausc_policy.history_runtime_prepare',
           'history_runtime': 'sim.ausc_policy.history_runtime',
           'history_path_probe': 'sim.ausc_policy.history_path_probe',
           'isaac_run': 'sim.ausc_policy.isaac_run'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('GPU experiments run on Linux only')
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.run_id):
        raise ValueError('Invalid run ID')
    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    module = MODULES[config['experiment']]
    base = Path(config['base'])
    path = base / 'diagnostics/coverage_pilot_cloud_launcher/process_guard.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != GUARD_SHA:
        raise ValueError('Original process/resource guard changed')
    spec = importlib.util.spec_from_file_location('_ausc_process_guard', path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.STAGE_LIMIT_S = config.get('time_limit_seconds', 3600)
    guard.GUARD = dict(guard.GUARD, stage_timeout_s=guard.STAGE_LIMIT_S)
    folder = base / 'runs/ausc_policy' / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    evidence = folder / 'lifecycle'
    evidence.mkdir()
    output = folder / 'output'
    root = Path(__file__).resolve().parents[2]
    commit_file = root / 'CODE_COMMIT'
    commit = commit_file.read_text().strip() if commit_file.exists() else subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    command = [sys.executable, '-u', '-B', '-m', module, '--config', str(config_path), '--output', str(output)]
    env = dict(os.environ, AUSC_OWNED_RUN='1', CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false')
    for key in ['PYTHONPATH', 'HF_TOKEN', 'JAX_PLATFORMS']:
        env.pop(key, None)
    if config['experiment'] in {'executor_probe','motion_preview_probe','path_probe','feedback_probe','recovery_sensor_audit','moving_brake_probe','capture_probe','history_prepare','history_runtime_prepare','history_runtime','history_path_probe'}:
        env['CUDA_VISIBLE_DEVICES'] = ''
    os.chdir(root)
    record = dict(run_id=args.run_id, code_commit=commit, config=config,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(), command=command,
        started_utc=guard.now(), completed=False)
    guard.write_new(folder / 'run.json', record)
    try:
        stage = guard.run_stage(config['experiment'], command,
            SimpleNamespace(output=output, evidence=evidence), {'runtime_project': str(root)}, env=env)
        record['stage'] = stage
        record['gpu_released'] = guard.wait_gpu_release()
        record['completed'] = bool(stage['passed'] and record['gpu_released'] and (output / 'COMPLETE.json').is_file())
    finally:
        record['ended_utc'] = guard.now()
        guard.write_new(folder / 'result.json', record)
    print(json.dumps({k: record.get(k) for k in ['run_id', 'completed', 'gpu_released']}), flush=True)
    return 0 if record['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
