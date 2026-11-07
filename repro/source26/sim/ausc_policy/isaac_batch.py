"""Finite parallel Isaac workers under ONE existing owner session and GPU lock.

Children deliberately inherit the session: process_guard can identify, monitor,
and drain every worker. A completed batch is not a physical qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def validate_jobs(config):
    n = config.get('parallelism')
    profile=config.get('resource_profile')
    if profile is None:maximum=4
    elif (profile=='wam_l40s_diagnostic_eight_v1' and config.get('diagnostic_only') is True
          and config.get('formal_acceptance') is False):maximum=8
    else:raise ValueError('Explicit truthful diagnostic resource profile required')
    if type(n) is not int or not 1 <= n <= maximum:
        raise ValueError('parallelism must be an integer from 1 to '+str(maximum))
    jobs = config.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Explicit finite jobs required')
    seen = set()
    result = []
    for job in jobs:
        name = job.get('id', '')
        if (not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name)
                or name in seen):
            raise ValueError('Invalid or duplicate job id')
        seen.add(name)
        path = Path(job['config']).resolve()
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != job['sha256']:
            raise ValueError('Child configuration hash differs')
        settings = json.loads(data)
        if (settings['experiment'] != 'geometry_qualification'
                or Path(settings['base']).resolve() != Path(config['base']).resolve()
                or not 60 < settings['time_limit_seconds'] <= config['time_limit_seconds']):
            raise ValueError('Unbounded or incompatible child configuration')
        result.append(dict(job, config=str(path), settings=settings))
    if 'matched_scene_evaluation' in config:
        from .m2_rebuild_paired_eval import validate_batch
        validate_batch(config,result)
    if 'validation_scenarios' in config:
        from .m2_given_contact_expanded_val_scenarios import validate_batch
        validate_batch(config,result)
    return result


def child_launch(job, output):
    settings = job['settings']
    command = [settings['isaac_python'], '-u', '-B', '-m',
               'sim.ausc_policy.geometry_qualification', '--config', job['config'],
               '--output', str(output)]
    env = dict(os.environ, AUSC_OWNED_RUN='1', CUDA_VISIBLE_DEVICES='0',
               LD_LIBRARY_PATH=settings['ld_library_path'], OMNI_KIT_ACCEPT_EULA='YES')
    for key in tuple(env):
        if key.startswith('AUSC_VAL_SCENARIO_'):env.pop(key)
    if 'validation_scenario' in job:
        scenario=job['validation_scenario'];ref=scenario['protocol']
        env.update(AUSC_VAL_SCENARIO_PROTOCOL=ref['path'],AUSC_VAL_SCENARIO_PROTOCOL_BYTES=str(ref['bytes']),
            AUSC_VAL_SCENARIO_PROTOCOL_SHA256=ref['sha256'],AUSC_VAL_SCENARIO_ARM=scenario['arm'],
            AUSC_VAL_SCENARIO_ID=scenario['scenario_id'],AUSC_VAL_SCENARIO_SEED=str(scenario['seed']))
    return command, dict(env=env, start_new_session=False)


def physical_pass(report):
    return (report.get('success') is True and report.get('physical_validation') is True
            and not report.get('errors'))


def main():
    import fcntl
    from . import process_guard as guard
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if os.environ.get('AUSC_OWNED_RUN') != '1' or config['experiment'] != 'isaac_batch':
        raise RuntimeError('Existing Linux experiment owner required')
    jobs = validate_jobs(config)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = []
    active = []
    queue = list(jobs)
    with (Path(config['base']) / 'locks/isaac_gpu.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Both checks occur while holding the shared lease. The live parent
        # guard continues checking for unrelated GPU owners after startup.
        for index in range(2):
            if guard.gpu_pids():
                raise RuntimeError('GPU already occupied')
            if index == 0:
                time.sleep(2)
        try:
            while queue or active:
                if time.monotonic() - started >= config['time_limit_seconds'] - 60:
                    raise TimeoutError('Finite batch deadline reached')
                while queue and len(active) < config['parallelism']:
                    job = queue.pop(0)
                    if hashlib.sha256(Path(job['config']).read_bytes()).hexdigest() != job['sha256']:
                        raise ValueError('Child configuration changed before launch')
                    output = args.output / job['id']
                    log = (args.output / (job['id'] + '.log')).open('xb')
                    command, kwargs = child_launch(job, output)
                    try:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **kwargs)
                    except BaseException:
                        log.close()
                        raise
                    row = dict(id=job['id'], config_sha256=job['sha256'], pid=child.pid,
                        started_utc=guard.now(), output=str(output))
                    # Register for local reaping before any fallible /proc or
                    # receipt I/O. The GPU lease must outlive every live child.
                    active.append((child, job, log, row, time.monotonic()))
                    row['identity'] = guard.process_identity(child.pid)
                    guard.write_new(args.output / (job['id'] + '.launch.json'), row)
                for item in list(active):
                    child, job, log, row, child_started = item
                    if child.poll() is None:
                        if time.monotonic() - child_started >= job['settings']['time_limit_seconds']:
                            raise TimeoutError('Child deadline: ' + job['id'])
                        continue
                    log.close()
                    output = Path(row['output'])
                    report_path = output / 'COMPLETE.json'
                    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
                    row.update(exit_code=child.returncode, ended_utc=guard.now(),
                        wall_s=time.monotonic() - child_started,
                        physical_pass=child.returncode == 0 and physical_pass(report),
                        worker_complete=child.returncode == 0 and bool(report),
                        result_sha256=guard.sha(report_path) if report_path.is_file() else None)
                    guard.write_new(args.output / (job['id'] + '.result.json'), row)
                    rows.append(row)
                    active.remove(item)
                    print(json.dumps(row), flush=True)
                if active:
                    time.sleep(.5)
        finally:
            # Outer process_guard handles the entire verified session. Also
            # reap direct children before releasing this lock on local errors.
            for child, _, _, _, _ in active:
                if child.poll() is None:
                    child.terminate()
            for child, _, log, _, _ in active:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
                log.close()
        summary = dict(completed=all(row['worker_complete'] for row in rows),
            physical_successes=sum(row['physical_pass'] for row in rows),
            attempted=len(rows), parallelism=config['parallelism'],
            wall_s=time.monotonic() - started, jobs=rows,
            learned_policy=False, training_eligible=False)
        if 'resource_profile' in config:summary['resource_profile']=config['resource_profile']
        if 'validation_scenarios' in config:
            descriptor=config['validation_scenarios'];arm=descriptor['arm']
            summary.update(validation_scenarios=descriptor,fixed_denominator=30,
                learned_policy=arm=='learned',expert_demonstration=arm=='expert',diagnostic_only=True,
                formal_acceptance=False,selection_validation_used=True,independent_test_claim=False,
                test_payloads_opened=False,no_replacements=True)
        guard.write_new(args.output / 'BATCH_RESULTS.json', summary)
        if not summary['completed']:
            raise RuntimeError('Worker infrastructure failed; see BATCH_RESULTS.json')
        guard.write_new(args.output / 'COMPLETE.json', summary)


if __name__ == '__main__':
    main()
