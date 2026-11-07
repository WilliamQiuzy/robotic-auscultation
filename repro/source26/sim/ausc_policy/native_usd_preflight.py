"""Full layered preflight in a bounded CPU child, keeping Kit's parent USD-clean.

The parent imports only the standard library. No validation is skipped: the
same scene_store.validate_config runs before Kit in the child and after Kit in
the original prepared_runtime route. This is startup evidence, never physics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import sys
import time

MODULE='sim.ausc_policy.native_usd_preflight'
SCHEMA='m2_isolated_native_usd_preflight_v1'
NATIVE_PYTHON='/opt/wam-runtime/envs/isaacsim60/bin/python'
REQUEST_LIMIT=2*1024*1024
STDOUT_LIMIT=8192
STDERR_LIMIT=65536
TIMEOUT_S=180.


def need(ok,message):
    if not ok:raise ValueError(message)


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(encoded(value)).hexdigest()


def exact(left,right):return encoded(left)==encoded(right)


def decode(raw):
    def pairs(items):
        result={}
        for key,value in items:
            need(key not in result,'Duplicate preflight JSON key');result[key]=value
        return result
    def invalid(value):raise ValueError('Nonfinite preflight JSON: '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)


def read_file(path):
    path=Path(path)
    need(path.is_absolute() and '..' not in path.parts and str(path)==str(path.resolve()) and
        not any(p.is_symlink() for p in (path,*path.parents)), 'Canonical preflight input required')
    meta=path.lstat()
    need(stat.S_ISREG(meta.st_mode) and meta.st_nlink==1,'Single regular preflight input required')
    data=path.read_bytes();after=path.lstat()
    need((meta.st_dev,meta.st_ino,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and
        len(data)==meta.st_size,'Preflight input changed during read')
    return dict(file=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),data


def file_ref(path):return read_file(path)[0]


def source_snapshot():
    helper=file_ref(Path(__file__));root=Path(helper['file']).parents[2]
    commit_ref,commit_data=read_file(root/'CODE_COMMIT');manifest_ref,manifest_data=read_file(root/'SOURCE_MANIFEST.json')
    commit=commit_data.decode().strip()
    need(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Frozen collector commit required')
    rows=decode(manifest_data);need(isinstance(rows,list) and rows,'Frozen source inventory required')
    names=[]
    for row in rows:
        need(isinstance(row,dict) and set(row)=={'file','bytes','sha256'} and type(row['bytes']) is int and
            row['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',str(row['sha256'])) is not None,'Exact frozen source file entry required')
        name=row.get('file');need(isinstance(name,str) and not Path(name).is_absolute() and
            '..' not in Path(name).parts and name not in names,'Unique closed source role required');names.append(name)
    actual={}
    for path in root.rglob('*'):
        meta=path.lstat();need(not stat.S_ISLNK(meta.st_mode),'Source tree symlink')
        if stat.S_ISDIR(meta.st_mode):continue
        ref=file_ref(path);actual[path.relative_to(root).as_posix()]=ref
    need(set(actual)==set(names)|{'CODE_COMMIT','SOURCE_MANIFEST.json'},'Source tree is not closed')
    for row in rows:
        ref=actual[row['file']]
        need(ref['bytes']==row.get('bytes') and ref['sha256']==row.get('sha256'),'Frozen preflight source changed')
    need(actual.get('sim/ausc_policy/native_usd_preflight.py')==helper,'Preflight helper outside source freeze')
    return dict(root=str(root),code_commit=commit,code_commit_file=commit_ref,source_manifest=manifest_ref,
        helper=helper,closed_files=len(actual),closed_inventory_sha256=digest(actual))


def inspect_usd(modules,maps_text):
    paths=[]
    for line in maps_text.splitlines():
        if any(token in line for token in ('/usd_exchange.libs/','/usd_core.libs/','/site-packages/pxr/')):
            paths.append(line.split()[-1])
    return dict(pxr_modules=sorted(name for name in modules if name=='pxr' or name.startswith('pxr.')),
        standalone_libraries=sorted(set(paths)))


def usd_state():return inspect_usd(sys.modules,Path('/proc/self/maps').read_text())


def clean_parent():
    state=usd_state()
    need(not state['pxr_modules'] and not state['standalone_libraries'],'Native parent must remain USD-clean before Kit')
    return state


def native_identity(*,worker=False):
    need(sys.platform=='linux' and sys.executable==NATIVE_PYTHON and sys.dont_write_bytecode and
        not sys.flags.optimize and os.environ.get('AUSC_OWNED_RUN')=='1','Owned unoptimized native60 -B preflight required')
    if worker:need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Isolated preflight must hide CUDA')
    text=Path('/proc/self/stat').read_text();fields=text[text.rfind(')')+2:].split()
    return dict(pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgrp(),sid=os.getsid(0),start_ticks=fields[19])


def configuration(config,path):
    ref,data=read_file(path);need(exact(decode(data),config),'Preflight config bytes differ from native object')
    need(config.get('compact_scene_storage',{}).get('schema')=='m2_compact_scene_storage_v3',
        'Isolated preflight is restricted to explicit layered v3')
    return ref


def bounded_cpu_call(command,payload,environment,*,cwd=None,timeout_s=TIMEOUT_S,stdout_limit=STDOUT_LIMIT,stderr_limit=STDERR_LIMIT):
    need(isinstance(payload,bytes) and len(payload)<=REQUEST_LIMIT,'Bounded preflight request required')
    child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env=environment,cwd=cwd,start_new_session=False)
    deadline=time.monotonic()+timeout_s;offset=0;outputs={};streams=(child.stdin,child.stdout,child.stderr)
    try:
        with selectors.DefaultSelector() as select:
            for stream,label,event in ((child.stdin,'stdin',selectors.EVENT_WRITE),(child.stdout,'stdout',selectors.EVENT_READ),
                                      (child.stderr,'stderr',selectors.EVENT_READ)):
                os.set_blocking(stream.fileno(),False);select.register(stream,event,label)
            while select.get_map():
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Isolated preflight deadline reached')
                events=select.select(remaining)
                if not events:raise TimeoutError('Isolated preflight deadline reached')
                for key,_ in events:
                    stream=key.fileobj;label=key.data
                    if label=='stdin':
                        if offset<len(payload):offset+=os.write(stream.fileno(),payload[offset:offset+65536])
                        if offset==len(payload):select.unregister(stream);stream.close()
                        continue
                    chunk=os.read(stream.fileno(),65536)
                    if not chunk:select.unregister(stream);continue
                    value=outputs.setdefault(label,bytearray());value.extend(chunk)
                    need(len(value)<=(stdout_limit if label=='stdout' else stderr_limit),'Isolated preflight output limit exceeded')
            child.wait(timeout=max(.001,deadline-time.monotonic()))
        need(child.returncode==0,'Isolated full preflight failed: '+bytes(outputs.get('stderr',b''))[-4096:].decode(errors='replace'))
        return dict(pid=child.pid,exit_code=child.returncode,reaped=child.poll() is not None,
            stdout=bytes(outputs.get('stdout',b'')),stderr=bytes(outputs.get('stderr',b'')))
    finally:
        if child.poll() is None:
            try:child.kill()
            except ProcessLookupError:pass
        child.wait(timeout=5)
        for stream in streams:
            if not stream.closed:stream.close()


def worker(request):
    identity=native_identity(worker=True)
    need(set(request)=={'schema','config','config_ref','source','parent'} and request['schema']==SCHEMA,
        'Exact isolated preflight request required')
    source=source_snapshot();need(exact(source,request['source']),'Child preflight source binding changed')
    need(exact(configuration(request['config'],request['config_ref']['file']),request['config_ref']), 'Child raw config binding changed')
    parent=request['parent']
    need(identity['ppid']==parent['pid'] and all(identity[k]==parent[k] for k in ('pgid','sid')),'Preflight detached from native owner group')
    need('omni.kit.app' not in sys.modules,'CPU preflight cannot enter Kit')
    from . import scene_store
    need(file_ref(scene_store.__file__)['file']==str(Path(source['root'])/'sim/ausc_policy/scene_store.py'),'Wrong isolated validator source')
    scene_store.validate_config(request['config'])  # Every original metadata, source and USD check; no bypass.
    need('omni.kit.app' not in sys.modules,'Full CPU validator unexpectedly entered Kit')
    need(exact(source_snapshot(),source) and exact(configuration(request['config'],request['config_ref']['file']),request['config_ref']),
        'Child inputs changed during full validation')
    return dict(schema=SCHEMA,request_sha256=digest(request),source=source,config_ref=request['config_ref'],
        identity=identity,passed=True,full_original_validator=True,kit_started=False,physics_executed=False,
        training_eligible=False)


def validate_before_kit(config,config_file):
    before=clean_parent();parent=native_identity();source=source_snapshot();config_ref=configuration(config,config_file)
    request=dict(schema=SCHEMA,config=config,config_ref=config_ref,source=source,parent=parent)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=source['root'],PYTHONDONTWRITEBYTECODE='1',
        PYTHONOPTIMIZE='0',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.monotonic()
    result=bounded_cpu_call([NATIVE_PYTHON,'-B','-m',MODULE,'--worker'],encoded(request),environment,cwd=source['root'])
    need(result.get('exit_code')==0 and result.get('reaped') is True,'CPU preflight child must exit and be reaped')
    value=decode(result['stdout']);need(isinstance(value,dict),'Complete CPU preflight JSON object required')
    identity=value.get('identity',{})
    need(isinstance(identity,dict) and set(identity)=={'pid','ppid','pgid','sid','start_ticks'} and
        all(type(identity[k]) is int and identity[k]>0 for k in ('pid','ppid','pgid','sid')) and
        isinstance(identity['start_ticks'],str) and identity['start_ticks'].isdigit(),'Exact CPU child process identity required')
    expected=dict(schema=SCHEMA,request_sha256=digest(request),source=source,config_ref=config_ref,
        identity=identity,passed=True,full_original_validator=True,kit_started=False,physics_executed=False,training_eligible=False)
    need(exact(value,expected) and identity.get('pid')==result['pid'] and identity.get('ppid')==parent['pid'] and
        all(identity.get(k)==parent[k] for k in ('sid','pgid')) and value.get('passed') is True and
        value.get('full_original_validator') is True and all(value.get(k) is False for k in
        ('kit_started','physics_executed','training_eligible')),'Incomplete or foreign CPU preflight response')
    need(exact(clean_parent(),before) and exact(native_identity(),parent) and exact(source_snapshot(),source) and exact(configuration(config,config_file),config_ref),
        'Native parent/config/source changed before Kit startup')
    return dict(value,parent_identity=parent,parent_USD_clean=True,wall_s=time.monotonic()-started,
        child_exit_code=result['exit_code'],child_reaped=True,stderr_bytes=len(result['stderr']),
        stderr_sha256=hashlib.sha256(result['stderr']).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true',required=True)
    parser.parse_args();native_identity(worker=True)
    raw=sys.stdin.buffer.read(REQUEST_LIMIT+1);need(len(raw)<=REQUEST_LIMIT,'Oversized preflight request')
    print(encoded(worker(decode(raw))).decode(),flush=True)


if __name__=='__main__':main()
