"""Own one finite, locked Isaac static model review."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    c = json.loads(args.config.read_text())
    if os.environ.get('AUSC_OWNED_RUN') != '1' or c['experiment'] not in ('tool_geometry_review', 'geometry_qualification','m2_geometry_plan','retainer_geometry_review','contact_semantics_oracle','camera_lifecycle_probe'):
        raise RuntimeError('Existing experiment owner required')
    if c['experiment']=='camera_lifecycle_probe' and (c.get('child_timeout_seconds')!=120 or c['time_limit_seconds']!=180):
        raise ValueError('Camera lifecycle probe requires a 120 second native child and 180 second owner stage')
    formal='m2_given_contact_heldout_planning_aggregate' in c
    if formal:
        from . import m2_given_contact_evaluation as evaluation
        evaluation.planning_aggregate_preflight(c,output=args.output/'review')
    args.output.mkdir(parents=True, exist_ok=False)
    from .isaac_case import approved_assets
    with approved_assets(Path(c['asset_root'])) as scene:
        scene.physical_sources()
    with (Path(c['base'])/'locks/isaac_gpu.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU already occupied')
        env = dict(os.environ, LD_LIBRARY_PATH=c['ld_library_path'], OMNI_KIT_ACCEPT_EULA='YES')
        if c['experiment']=='m2_geometry_plan':env['CUDA_VISIBLE_DEVICES']=''
        with (args.output/'isaac.log').open('w') as log:
            worker = 'sim.ausc_policy.geometry_qualification' if c['experiment']=='geometry_qualification' else 'sim.ausc_policy.tool_geometry_render'
            if c['experiment']=='retainer_geometry_review':worker='sim.ausc_policy.retainer_geometry_review'
            if c['experiment']=='m2_geometry_plan':worker='sim.ausc_policy.geometry_plan'
            if formal:worker=evaluation.MODULE
            if c['experiment']=='contact_semantics_oracle':worker='sim.ausc_policy.contact_semantics_oracle'
            if c['experiment']=='camera_lifecycle_probe':worker='sim.ausc_policy.camera_lifecycle_probe'
            command=[c['isaac_python'], '-u', '-B', '-m', worker,
                '--config', str(args.config.resolve()), '--output', str(args.output/'review')]
            child = subprocess.Popen(command,
                stdout=log, stderr=subprocess.STDOUT, env=env)
            timed_out=False
            identity=None
            try:
                if formal:
                    from . import process_guard
                    identity=process_guard.process_identity(child.pid)
                    if identity is None:raise RuntimeError('Formal CPU planning child identity unavailable')
                code = child.wait(timeout=c['time_limit_seconds']-60)
            except subprocess.TimeoutExpired:
                timed_out=True
                raise
            finally:
                try:
                    if child.poll() is None:
                        child.terminate()
                        try: child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait(timeout=10)
                finally:
                    status=dict(worker=worker,pid=child.pid,returncode=child.poll(),
                        timed_out=timed_out,clean_exit=child.returncode==0,
                        complete_file_present=(args.output/'review/COMPLETE.json').is_file())
                    if formal:
                        status.update(identity=identity,command=command,configuration=evaluation.runtime.reference(args.config.resolve()),
                            output=str(args.output/'review'),source_inventory=evaluation.planning_sources(),
                            complete=evaluation.runtime.reference(args.output/'review/COMPLETE.json') if status['complete_file_present'] else None)
                    (args.output/'native_worker_exit.json').write_text(json.dumps(status,indent=2)+'\n')
        if code != 0 or not (args.output/'review/COMPLETE.json').is_file():
            raise RuntimeError('Isaac geometry review failed; see preserved log')
        if formal:
            evaluation.planning_aggregate_preflight(c,output=args.output/'review')
            evaluation.runtime.bound_path(status['configuration'])
        report = json.loads((args.output/'review/COMPLETE.json').read_text())
        report['isaac_exit_code'] = code
        (args.output/'COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
