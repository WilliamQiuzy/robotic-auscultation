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
    if os.environ.get('AUSC_OWNED_RUN') != '1' or c['experiment'] not in ('tool_geometry_review', 'geometry_qualification','m2_geometry_plan','retainer_geometry_review'):
        raise RuntimeError('Existing experiment owner required')
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
            child = subprocess.Popen([c['isaac_python'], '-u', '-B', '-m', worker,
                '--config', str(args.config.resolve()), '--output', str(args.output/'review')],
                stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                code = child.wait(timeout=c['time_limit_seconds']-60)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait(timeout=10)
        if code != 0 or not (args.output/'review/COMPLETE.json').is_file():
            raise RuntimeError('Isaac geometry review failed; see preserved log')
        report = json.loads((args.output/'review/COMPLETE.json').read_text())
        report['isaac_exit_code'] = code
        (args.output/'COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
