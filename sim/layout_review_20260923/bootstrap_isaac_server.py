"""Install an isolated Isaac Sim 5.1 runtime on the user-authorized L40S host."""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT = Path.home() / 'wam-sim'


def main():
    assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
    ROOT.mkdir(exist_ok=True)
    marker = ROOT / 'isaac_install_status.json'
    env_path = ROOT / 'env_isaacsim51'
    if env_path.exists() and not marker.exists():
        raise RuntimeError('Refusing to change an existing unmarked environment')
    assert shutil.disk_usage(ROOT).free > 60*1024**3
    state = {'schema':'wam_isaac_install_v1','isaac_version':'5.1.0',
             'environment':str(env_path),'started_at_epoch':time.time(),
             'status':'installing','steps':[]}
    marker.write_text(json.dumps(state,indent=2)+'\n')
    env = os.environ.copy()
    env.update(UV_PYTHON_INSTALL_DIR=str(ROOT/'python-runtimes'),
               UV_PYTHON_BIN_DIR=str(ROOT/'bin'), UV_CACHE_DIR=str(ROOT/'uv-cache'),
               PIP_CACHE_DIR=str(ROOT/'pip-cache'),
               PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

    def run(name,args):
        print('INSTALL_STEP',name,flush=True)
        began=time.time()
        result=subprocess.run(list(map(str,args)),env=env,check=True)
        state['steps'].append({'name':name,'seconds':round(time.time()-began,2),'returncode':result.returncode})
        marker.write_text(json.dumps(state,indent=2)+'\n')

    try:
        uv=Path.home()/'.local/bin/uv'
        assert uv.is_file(), 'Expected previously verified uv executable'
        run('install_task_python311',[uv,'python','install','3.11'])
        candidates=sorted((ROOT/'python-runtimes').glob('cpython-3.11.*-linux-x86_64-gnu/bin/python3.11'))
        assert len(candidates)==1, candidates
        if not env_path.exists():
            run('create_isolated_venv',[candidates[0],'-m','venv',env_path])
        python=env_path/'bin/python'
        run('upgrade_task_pip',[python,'-m','pip','install','--upgrade','pip'])
        run('install_isaacsim51',[python,'-m','pip','install','isaacsim[all,extscache]==5.1.0',
                                '--extra-index-url','https://pypi.nvidia.com'])
        run('install_preview_image_dependency',[python,'-m','pip','install','Pillow'])
        installed=subprocess.run([python,'-c','import importlib.metadata as m; print(m.version("isaacsim"))'],
                                 env=env,check=True,capture_output=True,text=True).stdout.strip()
        # NVIDIA's wheel metadata uses the equivalent four-part release number.
        assert installed in ('5.1.0', '5.1.0.0')
        state.update(status='installed_not_yet_runtime_tested',finished_at_epoch=time.time(),installed_version=installed)
        marker.write_text(json.dumps(state,indent=2)+'\n')
        print('ISAAC_INSTALL_COMPLETE',installed,flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),finished_at_epoch=time.time())
        marker.write_text(json.dumps(state,indent=2)+'\n')
        raise


if __name__=='__main__':
    main()
