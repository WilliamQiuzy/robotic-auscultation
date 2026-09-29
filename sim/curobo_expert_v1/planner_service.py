"""File-based, isolated cuRobo calls; no Isaac or GPU imports here."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):h.update(part)
    return h.hexdigest()


def validate_proposal(result, start, goal, tolerance=.002):
    if not isinstance(result,dict) or result.get('success') is not True:
        raise ValueError('cuRobo proposal failed: '+str(result.get('failure') if isinstance(result,dict) else result))
    q=np.asarray(result.get('path'),float);times=np.asarray(result.get('timestamps'),float)
    start=np.asarray(start,float);goal=np.asarray(goal,float)
    if q.ndim!=2 or q.shape[1]!=7 or len(q)<2 or not np.isfinite(q).all():
        raise ValueError('proposal must contain finite seven-joint poses')
    if times.shape!=(len(q),) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('proposal timestamps missing or nonmonotonic')
    if start.shape!=(7,) or goal.shape!=(7,) or not np.isfinite([start,goal]).all():
        raise ValueError('requested endpoints malformed')
    if max(np.max(np.abs(q[0]-start)),np.max(np.abs(q[-1]-goal)))>tolerance:
        raise ValueError('proposal endpoint residual exceeds tolerance')
    # Never overwrite an approximate endpoint: the caller must verify explicit
    # connectors to both actual retained command and exact requested goal.
    return q


def request_plan(start, goal, directory, bridge_dir, launcher, *, timeout=240):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    bridge=Path(bridge_dir).resolve()
    from .robot_bridge import load_bridge
    robot,world,_=load_bridge(bridge)
    request={'operation':'plan','robot_config':robot,
             'world':world,'start7':list(map(float,start)),
             'end7':list(map(float,goal)),
             'options':{'graph_seed':0,'ik_seed':0,'enable_graph':True,'enable_opt':False,'timeout_s':60.}}
    request_path=directory/'request.json';result_path=directory/'result.json'
    request_path.write_text(json.dumps(request,indent=2,allow_nan=False)+'\n')
    command=[sys.executable,'-m','sim.curobo_expert_v1.guarded_run',
             '--timeout-seconds',str(timeout),'--record',str((directory/'resources.json').resolve()),
             '--',str(Path(launcher).resolve()),'--request',str(request_path.resolve()),
             '--output',str(result_path.resolve())]
    # The launcher belongs to the isolated backend environment, not Isaac's
    # bundled Python. Remove inherited module search paths at this boundary.
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_LIBRARY_PATH'):env.pop(key,None)
    started=time.monotonic();status=None;error=None
    with (directory/'backend.log').open('x') as stream:
        try:status=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                                  cwd=Path(__file__).resolve().parents[2],env=env,timeout=timeout+30).returncode
        except subprocess.TimeoutExpired:error='planner subprocess timeout'
        except OSError as exc:error='planner launch failed: '+str(exc)
    evidence={'command':command,'returncode':status,'error':error,
              'wall_time_s':time.monotonic()-started,
              'request_sha256':sha(request_path),'launcher_sha256':sha(launcher),
              'result_sha256':sha(result_path) if result_path.exists() else None}
    (directory/'call.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if status!=0:raise ValueError('cuRobo backend failed; see '+str(directory/'backend.log'))
    result=json.loads(result_path.read_text())
    from .curobo_backend import EXPECTED_COMMIT
    metadata=result.get('metadata',{})
    if (metadata.get('backend')!='curobo' or metadata.get('fallback_used') is not False
            or metadata.get('curobo_commit')!=EXPECTED_COMMIT
            or metadata.get('curobo_source_dirty') not in (False,'')
            or metadata.get('request_sha256')!=evidence['request_sha256']
            or metadata.get('backend_source_sha256')!=sha(Path(__file__).with_name('curobo_backend.py'))):
        raise ValueError('cuRobo backend identity or request binding mismatch')
    from .preflight import expected_assets,verify_assets
    verify_assets(metadata,expected_assets(robot,world))
    if (metadata.get('solver_mode')!='native_graph_only'
            or any(metadata.get('options',{}).get(k)!=v for k,v in request['options'].items())):
        raise ValueError('backend solver mode/options differ from explicit graph-only request')
    return validate_proposal(result,start,goal),result,evidence
