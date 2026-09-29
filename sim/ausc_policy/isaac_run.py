"""Own one CPU ACT service and one Isaac diagnostic in the guarded process group."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def validate_posture_preview(config):
    enabled=config.get('posture_hint_preview',False)
    if type(enabled) is not bool or (enabled and config.get('motion_velocity')!='quadratic5'):
        raise ValueError('Invalid posture preview configuration')
    if not enabled:
        return
    probe=Path(config['motion_preview_probe_run'])
    owner=read(probe/'result.json');complete=read(probe/'output/COMPLETE.json')
    data=(probe/'output/metrics.json').read_bytes();metrics=json.loads(data)
    if (owner.get('completed') is not True or owner.get('gpu_released') is not True
            or complete.get('preview_passed') is not True
            or complete.get('metrics_sha256')!=hashlib.sha256(data).hexdigest()
            or metrics.get('motion_preview_source_sha256')!=hashlib.sha256(Path(__file__).with_name('motion_preview.py').read_bytes()).hexdigest()
            or metrics.get('original_failure_reproduced') is not True
            or metrics.get('active_state_unchanged') is not True
            or metrics['preview_vs_generation_max_abs_rad']>1e-12
            or metrics['limits']!=dict(dt=1/240,velocity=.175,acceleration=.5,jerk=7.99999)
            or metrics['factors']!=[1.,.5,.25,.125,0.] or metrics['ticks']!=12):
        raise ValueError('Actual source-bound native posture preview proof is required')


def validate_config(config):
    if (config['experiment'] != 'isaac_run' or config['mode'] != 'm0_direct'
            or config['point'] != 'R4' or config['checkpoint_step'] != 2000
            or config['physics'] != dict(position_iterations=32, velocity_iterations=1, armature_kg_m2=.02)
            or config['speed_scale'] != .7 or config['time_limit_seconds'] != 3600
            or config.get('motion_velocity','zero') not in {'zero','quadratic5'}):
        raise ValueError('Only the finite, unchanged-physics M0 R4 direct diagnostic is declared')
    validate_posture_preview(config)
    if config.get('motion_velocity','zero') == 'quadratic5':
        probe = Path(config['motion_probe_run'])
        owner = read(probe/'result.json'); complete = read(probe/'output/COMPLETE.json')
        data = (probe/'output/metrics.json').read_bytes(); metrics = json.loads(data)
        if (owner.get('completed') is not True or owner.get('gpu_released') is not True
                or complete.get('comparison_completed') is not True
                or complete.get('metrics_sha256') != hashlib.sha256(data).hexdigest()
                or metrics.get('velocity_estimator') != 'quadratic5'
                or set(metrics['branches']) != {'zero_terminal_velocity','quadratic5_target_velocity'}
                or not all(branch['completed'] for branch in metrics['branches'].values())
                or metrics['schedule']['limits'] != dict(dt=1/240,velocity=.175,acceleration=.5,jerk=7.99999)
                or metrics['schedule']['horizon'] != 10 or metrics['schedule']['execute_steps'] != 5
                or metrics['schedule']['period_s'] != .05 or metrics['schedule']['ticks_per_action'] != 12):
            raise ValueError('Actual unchanged-limit native quadratic5 probe is required')
    reload_run = Path(config['reload_run'])
    result = read(reload_run/'result.json'); receipt = read(reload_run/'output/COMPLETE.json')
    if (result.get('completed') is not True or receipt.get('passed') is not True
            or receipt['receipt']['checkpoint_sha256'] != config['checkpoint_sha256']
            or receipt['max_reference_difference_rad'] > 1e-6
            or receipt['max_reload_difference_rad'] != 0.):
        raise ValueError('Actual native CPU reload gate is absent or mismatched')
    return receipt['receipt']


def service_config(config):
    # Deliberately omit all training arrays and the reload reference files.
    return {**{key: config[key] for key in ('base','run_root','checkpoint_step','checkpoint_sha256')},
            'experiment':'act_runtime', 'mode':'serve', 'port':0, 'request_timeout_s':5.}


def stop_child(child):
    if child is None:
        return None
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait(timeout=10)
    return child.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); config = read(args.config)
    if sys.platform != 'linux' or os.environ.get('AUSC_OWNED_RUN') != '1':
        raise RuntimeError('An original guarded Linux owner is required')
    receipt = validate_config(config)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    write(output/'configuration.json', config)
    service = child = None
    report = dict(completed=False, m0_passed=False, closed_loop_evaluated=False,
                  expert_demonstration=False, training_dataset_eligible=False)
    with (Path(config['base'])/'locks/isaac_gpu.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            env = dict(os.environ, LD_LIBRARY_PATH=config['ld_library_path'], OMNI_KIT_ACCEPT_EULA='YES')
            env.pop('PYTHONPATH', None)
            with (output/'preflight.log').open('w') as preflight_log:
                child = subprocess.Popen([config['isaac_python'],'-u','-B','-m','sim.ausc_policy.isaac_case',
                    '--config',str(args.config.resolve()),'--output',str(output/'preflight'),'--preflight-only'],
                    env=env, stdout=preflight_log, stderr=subprocess.STDOUT)
                if child.wait(timeout=120) != 0:
                    raise RuntimeError('Isaac CPU asset/import preflight failed; see preflight.log')
            ready_path = output/'service/ready.json'
            native_config = output/'service_config.json'; write(native_config, service_config(config))
            with (output/'service.log').open('w') as service_log, (output/'isaac.log').open('w') as case_log:
                service = subprocess.Popen([sys.executable,'-u','-B','-m','sim.ausc_policy.act_runtime',
                    '--config',str(native_config),'--output',str(output/'service')],
                    stdout=service_log, stderr=subprocess.STDOUT)
                deadline = time.monotonic()+120
                while not ready_path.exists():
                    if service.poll() is not None:
                        raise RuntimeError('ACT CPU service exited before ready; see service.log')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('ACT CPU readiness deadline')
                    time.sleep(.1)
                ready = read(ready_path)
                if ready['host'] != '127.0.0.1' or ready['receipt'] != receipt:
                    raise ValueError('Ready service differs from the actually reloaded checkpoint')
                command = [config['isaac_python'],'-u','-B','-m','sim.ausc_policy.isaac_case',
                    '--config',str(args.config.resolve()),'--ready',str(ready_path),'--output',str(output/'case')]
                child = subprocess.Popen(command, env=env, stdout=case_log, stderr=subprocess.STDOUT)
                # Neither child creates a new process group; the original guard owns all descendants.
                code = child.wait(timeout=config['time_limit_seconds']-150)
                case = read(output/'case/result.json')
                if code != 0 or case.get('execution_completed') is not True:
                    raise RuntimeError('Isaac execution/cleanup failed; preserve isaac.log and case evidence')
                report.update(completed=True, closed_loop_evaluated=case['closed_loop_evaluated'],
                    case_passed=case['case_passed'], failure_reason=case.get('failure_reason'),
                    case_result_sha256=hashlib.sha256((output/'case/result.json').read_bytes()).hexdigest())
        except Exception as error:
            report.update(error=type(error).__name__, reason=str(error))
            raise
        finally:
            report.update(isaac_exit_code=stop_child(child), service_exit_code=stop_child(service))
            write(output/('COMPLETE.json' if report['completed'] else 'FAILED.json'), report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
