"""Own a finite sequential three-case expert pilot under the original GPU lock."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from .isaac_run import stop_child, write, read
from .moving_origin import selected_origin


CASES = [('nominal',0.),('joint3-plus',.02),('joint3-minus',-.02)]


def selected_cases(config):
    cases=CASES
    if config.get('mode')=='moving_oracle_pilot':
        # Preserve the old case-list API; full source loading still requires
        # its explicit index. New protocols may not use an implicit origin.
        selection=dict(config)
        if 'moving_origin' not in selection:selection.setdefault('source_action_index',609)
        cases=[(selected_origin(selection)[2],0.)]
    if config.get('mode')=='capture_component':
        cases=[('entry-tangent-plus',0.),('entry-normal-plus',0.)]
    names = config.get('case_ids',[name for name,_ in cases])
    if (not isinstance(names,list) or not names or any(not isinstance(n,str) for n in names)
            or names != [name for name,_ in cases if name in names]):
        raise ValueError('Require a nonempty unique subset in the predeclared case order')
    return [(name,offset) for name,offset in cases if name in names]


def validate_case(config):
    if (config['experiment'] != 'recovery_run' or config['mode'] not in ('stationary_oracle_pilot','moving_oracle_pilot','capture_component')
            or config['point'] != 'R4' or config['speed_scale'] != .7
            or config['physics'] != dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02)
            or config['time_limit_seconds'] != 7200):
        raise ValueError('Only the finite unchanged-physics stationary R4 pilot is declared')
    expected = {name:[0.,0.,offset,0.,0.,0.,0.] for name,offset in selected_cases(config)}
    if config['case_id'] not in dict(selected_cases(config)):
        raise ValueError('Case is outside declared subset')
    if config['case_id'] not in expected or config['perturbation_rad'] != expected[config['case_id']]:
        raise ValueError('Case differs from predeclared symmetric perturbation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); config = read(args.config)
    if sys.platform != 'linux' or os.environ.get('AUSC_OWNED_RUN') != '1':
        raise RuntimeError('Original guarded Linux owner required')
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    write(output/'configuration.json',config)
    report = dict(completed=False,m0_passed=False,learned_policy=False,training_dataset_eligible=False,cases=[])
    child = None
    with (Path(config['base'])/'locks/isaac_gpu.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            env = dict(os.environ,LD_LIBRARY_PATH=config['ld_library_path'],OMNI_KIT_ACCEPT_EULA='YES')
            env.pop('PYTHONPATH',None)
            for index,(name,offset) in enumerate(selected_cases(config)):
                case = dict(config,case_id=name,perturbation_rad=[0.,0.,offset,0.,0.,0.,0.])
                validate_case(case)
                path = output/(name+'-config.json'); write(path,case)
                command = [config['isaac_python'],'-u','-B','-m','sim.ausc_policy.recovery_case','--config',str(path)]
                if index == 0:
                    with (output/'preflight.log').open('w') as log:
                        child = subprocess.Popen(command+['--output',str(output/'preflight'),'--preflight-only'],env=env,stdout=log,stderr=subprocess.STDOUT)
                        if child.wait(timeout=120) != 0:
                            raise RuntimeError('Recovery asset/native import preflight failed')
                with (output/(name+'.log')).open('w') as log:
                    child = subprocess.Popen(command+['--output',str(output/name)],env=env,stdout=log,stderr=subprocess.STDOUT)
                    code = child.wait(timeout=2200)
                case_result = read(output/name/'result.json')
                report['cases'].append(dict(case_id=name,**case_result))
                write(output/'progress.json',report)
                if code != 0 or not case_result.get('execution_completed'):
                    raise RuntimeError('Case execution/cleanup failed: '+name)
            report.update(completed=True,all_cases_passed=all(row['case_passed'] for row in report['cases']))
        except Exception as error:
            report.update(error=type(error).__name__,reason=str(error))
            raise
        finally:
            report['last_child_exit_code'] = stop_child(child)
            write(output/('COMPLETE.json' if report['completed'] else 'FAILED.json'),report)


if __name__ == '__main__':
    main()
