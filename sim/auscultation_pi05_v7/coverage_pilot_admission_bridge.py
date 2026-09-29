"""Bound CPU subprocess admission in the unchanged training interpreter.

No selection algorithm is copied. The original selected_admission reconstructs
all real evidence in OpenPI; the physical interpreter rehashes every artifact.
"""
import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time

BASE=Path('/home/william/wam-vla-v2')
OPENPI_PREFIX=BASE/'vendor/openpi/.venv'
OPENPI_PYTHON=OPENPI_PREFIX/'bin/python'
PYTHON_SHA='ec579de7512e84d960bc642a840de69c785c52d6adf89e3a15f51717b29be40f'
NUMPY_SHA='22cd1535fa14d74ef6f457cca149ffdc80875f460be313b8f895273f78bc402e'
SCIPY_SHA='185913aa107512fafd5cf89dfd452a8719b45a6eba833e2dcee2cbfd18b4bbe5'
PARENT_INVENTORY=BASE/'cloud-runtime-v7/coverage-pilot-isaac-20260927-v1.source_inventory.json'
PARENT_SHA='5ea45af1940551957b0745ccfe8275d42113a1f8c85d1c5c44bb24f332c67a6a'
GUARD_PATH=BASE/'diagnostics/headroom_cloud_launcher/process_guard.py'
GUARD_SHA='9d4bffe8e1e06c76075a649049809ad30432404c42089ae40fddb9377cd85c27'
ORIGINAL_CONTRACT_SHA='4399fb7170e77c80d5e2926883a86750085ff6709d032b96b9306708fbb0c4d4'
NAMES=('coverage_pilot_admission_bridge.py','coverage_pilot_bridge_runtime_contract.py','run_coverage_pilot_bridge_headroom_isaac.py',
       'audit_coverage_pilot_bridge_headroom_actions.py','assess_coverage_pilot_bridge_headroom_trials.py')
REQUEST_SCHEMA='wam_v7_coverage_pilot_cpu_admission_request_v1'
RESPONSE_SCHEMA='wam_v7_coverage_pilot_cpu_admission_response_v1'
RECEIPT_SCHEMA='wam_v7_coverage_pilot_cpu_admission_bridge_receipt_v1'
TIMEOUT_S=1800
STDOUT_LIMIT=4*2**20
STDERR_LIMIT=128*2**10
CHILD_ENV_KEYS={'PATH','LANG','HOME','CUDA_VISIBLE_DEVICES','JAX_PLATFORMS','PYTHONPATH','PYTHONNOUSERSITE',
                'PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'}


def need(ok,message):
    if not ok:raise ValueError(message)


class AdmissionFailure(ValueError):
    """Public bounded process evidence; caller retains this in its own log."""
    def __init__(self,message,receipt):
        super().__init__(message)
        self.receipt=receipt


class AdmissionInterrupted(KeyboardInterrupt):
    """Parent cancellation must stop all subsequent audit/lifecycle dispatch."""
    def __init__(self,message,receipt):
        super().__init__(message)
        self.receipt=receipt


def file_bytes_sha(raw):return hashlib.sha256(raw).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()

def json_bytes(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def digest(value):return file_bytes_sha(json_bytes(value))


def read_json_bytes(raw):
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,'Duplicate JSON key');out[k]=v
        return out
    return json.loads(raw.decode('utf8'),object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def regular(path):
    p=Path(path)
    need(p.is_absolute() and not p.is_symlink() and p.resolve()==p and p.is_file(),'Canonical regular input required: '+str(p))
    return p


def root_path():return Path(__file__).resolve().parents[2]


def source_contract():
    root=root_path();need(file_sha(regular(PARENT_INVENTORY))==PARENT_SHA,'Original337 inventory changed')
    parent=read_json_bytes(PARENT_INVENTORY.read_bytes());files=parent['source_files']
    need(parent.get('source_count')==337 and len(files)==337,'Original337 inventory required')
    for name,expected in files.items():need(file_sha(regular(root/name))==expected,'Inherited337 source changed: '+name)
    extension={'v7/'+name:file_sha(regular(Path(__file__).with_name(name))) for name in NAMES}
    expected_names=set(files)|{'sim/auscultation_pi05_'+n for n in extension}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    need(actual==expected_names,'Exact independent342 source tree required')
    need(file_sha(OPENPI_PYTHON)==PYTHON_SHA,'Bound original OpenPI interpreter changed')
    site=OPENPI_PREFIX/'lib/python3.11/site-packages'
    env_files={str(site/'numpy/__init__.py'):NUMPY_SHA,str(site/'scipy/__init__.py'):SCIPY_SHA}
    for path,expected in env_files.items():need(file_sha(regular(path))==expected,'Training numerical package binding changed')
    need(file_sha(regular(GUARD_PATH))==GUARD_SHA,'Reviewed owned-process guard changed')
    return dict(schema='wam_v7_coverage_pilot_cpu_admission_bridge_contract_v1',source_project=str(root),
        parent337_inventory=dict(path=str(PARENT_INVENTORY),sha256=PARENT_SHA),
        original_contract_sha256=ORIGINAL_CONTRACT_SHA,original_pilot_source_sha256=parent['pilot_source_sha256'],
        original_connection_source_sha256=parent['connection_source_sha256'],source_sha256=extension,
        worker_python=dict(path=str(OPENPI_PYTHON),sha256=PYTHON_SHA,prefix=str(OPENPI_PREFIX)),
        worker_numpy_version='1.26.4',worker_scipy_version='1.15.3',numerical_package_files=env_files,
        owned_process_guard=dict(path=str(GUARD_PATH),sha256=GUARD_SHA),timeout_s=TIMEOUT_S,
        cleanup_grace_s=dict(interrupt=120,terminate=60,kill=30),
        cancellation='AdmissionInterrupted(KeyboardInterrupt) propagates after owned cleanup; no subsequent dispatch',
        stdout_limit_bytes=STDOUT_LIMIT,stderr_limit_bytes=STDERR_LIMIT,
        selection_algorithm='original frozen coverage_pilot_runtime_contract.selected_admission',
        physical_numpy_unchanged=True,selection_algorithm_copied=False,version_relabelled=False,
        model_initialized=False,private_configuration_used=False)


def child_environment(root):
    return dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',HOME='/home/william',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
        PYTHONPATH=str(root),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')


def guard_module():
    need(file_sha(regular(GUARD_PATH))==GUARD_SHA,'Bound CPU cleanup helper changed')
    spec=importlib.util.spec_from_file_location('wam_bridge_owned_guard',GUARD_PATH);m=importlib.util.module_from_spec(spec)
    exec(compile(GUARD_PATH.read_bytes(),str(GUARD_PATH),'exec'),m.__dict__);return m


def run_bounded(command,payload,*,env,cwd,timeout_s=TIMEOUT_S,stdout_limit=STDOUT_LIMIT,stderr_limit=STDERR_LIMIT):
    """Owned Linux session; bound streams/time, no GPU/lock/resource probes."""
    need(sys.platform=='linux','CPU admission ownership requires Linux')
    need(timeout_s>0 and stdout_limit>0 and stderr_limit>0,'Positive subprocess bounds required')
    g=guard_module();process=None;identity=None;buffers={'stdout':bytearray(),'stderr':bytearray()}
    result=dict(exit_code=None,process_group_drained=False,errors=[],signals=[],cancelled=False)
    prior={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)};deferred=[]
    def defer(signum,frame):deferred.append(signal.Signals(signum).name)
    def interrupt(signum,frame):deferred.append(signal.Signals(signum).name);raise KeyboardInterrupt('Admission parent interrupted')
    selector=selectors.DefaultSelector();started=time.monotonic();input_stream=None
    try:
        for sig in prior:signal.signal(sig,defer)
        need(len(payload)<=65536,'Admission request exceeds bound')
        input_stream=tempfile.TemporaryFile();input_stream.write(payload);input_stream.seek(0)
        process=subprocess.Popen(command,stdin=input_stream,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            env=env,cwd=cwd,start_new_session=True)
        identity=g.process_identity(process.pid);need(identity['pgid']==process.pid and identity['sid']==process.pid,'Owned child session differs')
        result['process_identity']=dict(identity);g.remember_group(identity)
        for sig in prior:signal.signal(sig,interrupt)
        if deferred:raise KeyboardInterrupt('Admission interrupted at child startup')
        for name in buffers:
            stream=getattr(process,name);os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,name)
        deadline=started+timeout_s
        while selector.get_map():
            if time.monotonic()>=deadline:raise TimeoutError('CPU admission wall-clock limit exceeded')
            g.remember_group(identity)
            for key,_ in selector.select(timeout=min(.2,max(0,deadline-time.monotonic()))):
                raw=os.read(key.fileobj.fileno(),8192)
                if not raw:selector.unregister(key.fileobj);continue
                name=key.data;buffers[name].extend(raw)
                need(len(buffers[name])<=(stdout_limit if name=='stdout' else stderr_limit),'CPU admission '+name+' limit exceeded')
        remaining=max(.001,deadline-time.monotonic());process.wait(timeout=remaining)
        if not g.drain(process,identity,2):raise RuntimeError('CPU admission descendants remain')
    except BaseException as error:result['errors'].append(type(error).__name__+': '+str(error))
    finally:
        for sig in prior:signal.signal(sig,defer)
        try:
            if process is not None:
                if identity is None:
                    try:identity=g.process_identity(process.pid)
                    except (OSError,ValueError):identity=None
                if identity is not None and identity.get('pgid')==process.pid and identity.get('sid')==process.pid:
                    try:result['process_group_drained']=g.stop_owned(process,identity,result['signals'])
                    except BaseException as error:result['errors'].append('CPU cleanup: '+type(error).__name__+': '+str(error))
                else:
                    result['errors'].append('Unverified child session; direct unreaped child cleanup only')
                    for sig,seconds in ((signal.SIGTERM,10),(signal.SIGKILL,10)):
                        if process.poll() is not None:break
                        process.send_signal(sig)
                        try:process.wait(timeout=seconds);break
                        except subprocess.TimeoutExpired:pass
                result['exit_code']=process.poll()
                for name in ('stdin','stdout','stderr'):
                    stream=getattr(process,name,None)
                    if stream is not None and not stream.closed:stream.close()
            else:result['process_group_drained']=True
            result.update(stdout=bytes(buffers['stdout']),stderr=bytes(buffers['stderr']),elapsed_s=time.monotonic()-started,
                cancelled=bool(deferred),deferred_signals=list(deferred))
        finally:
            selector.close()
            if input_stream is not None:input_stream.close()
            for sig,handler in prior.items():signal.signal(sig,handler)
    # Signals deferred during stream/selector close must also cancel admission.
    result.update(cancelled=bool(deferred) or any(e.startswith('KeyboardInterrupt:') for e in result['errors']),deferred_signals=list(deferred))
    return result


def verify_admission_artifacts(admission):
    artifacts=admission.get('artifact_sha256')
    need(isinstance(artifacts,dict) and artifacts,'Nonempty original admission artifacts required')
    for path,expected in artifacts.items():need(file_sha(regular(path))==expected,'Bound admission artifact changed: '+path)
    return dict(artifacts)


def verify_worker_environment():
    need(Path(sys.prefix)==OPENPI_PREFIX and file_sha(sys.executable)==PYTHON_SHA,'Actual original OpenPI interpreter required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('JAX_PLATFORMS')=='cpu','CPU-hidden admission worker required')
    import numpy,scipy
    need(numpy.__version__=='1.26.4' and scipy.__version__=='1.15.3','Actual training numerical versions differ')
    need(file_sha(numpy.__file__)==NUMPY_SHA and file_sha(scipy.__file__)==SCIPY_SHA,'Actual numerical imports changed')
    return dict(python=sys.executable,prefix=sys.prefix,numpy=numpy.__version__,scipy=scipy.__version__,
        cuda_visible_devices='',jax_platforms='cpu',pid=os.getpid(),parent_pid=os.getppid())


def verify_request(request):
    contract=source_contract()
    need(set(request)=={'schema','source_contract','source_project','selection_lock','selection_lock_sha256','manifest','manifest_sha256'},'Exact public admission request schema required')
    need(request['schema']==REQUEST_SCHEMA and request['source_contract']==contract and request['source_project']==str(root_path()),'Admission request source binding differs')
    need(file_sha(regular(request['manifest']))==request['manifest_sha256'],'Requested original manifest changed')
    lock=Path(request['selection_lock']);need(lock.is_absolute() and lock.resolve()==lock and not lock.is_symlink(),'Canonical actual selection path required')
    if lock.exists():need(file_sha(regular(lock))==request['selection_lock_sha256'],'Actual selected lock changed')
    else:need(request['selection_lock_sha256'] is None,'Cannot claim a hash for absent selected lock')
    return contract


def worker_response(request):
    environment=verify_worker_environment();contract=verify_request(request)
    result=dict(schema=RESPONSE_SCHEMA,request_sha256=digest(request),source_contract=contract,environment=environment,
        status='not_ready',admission=None,selection_reconstructed=False,model_initialized=False,service_contacted=False)
    if not Path(request['selection_lock']).is_file():return result
    original=importlib.import_module('sim.auscultation_pi05_v7.coverage_pilot_runtime_contract')
    need(Path(original.__file__).resolve()==root_path()/'sim/auscultation_pi05_v7/coverage_pilot_runtime_contract.py'
        and file_sha(original.__file__)==ORIGINAL_CONTRACT_SHA,'Original selector source escaped bound tree')
    admission=original.selected_admission(request['selection_lock'],request['selection_lock_sha256'],request['manifest'])
    need(source_contract()==contract,'Admission source changed during CPU reconstruction')
    verify_admission_artifacts(admission)
    need(not any(n=='fcl' or n.startswith('fcl.') or n in {'isaacsim','pxr','omni'} or n.startswith(('isaacsim.','pxr.','omni.')) for n in sys.modules),'CPU selector imported physical SDK/FCL')
    result.update(status='verified',admission=admission,selection_reconstructed=True);return result


def verify_receipt(receipt,admission):
    need(receipt.get('schema')==RECEIPT_SCHEMA,'New bridge receipt required')
    need(type(receipt.get('process_exit_code')) is int and receipt['process_exit_code']==0
        and receipt.get('process_group_drained') is True and receipt.get('cancelled') is False,'CPU admission exit/drain/cancellation invalid')
    response=receipt['worker_response'];contract=source_contract()
    need(response.get('schema')==RESPONSE_SCHEMA and response.get('status')=='verified' and response.get('selection_reconstructed') is True,'Real full CPU selection reconstruction required')
    need(response.get('source_contract')==receipt.get('source_contract')==contract,'Bridge source/interpreter contract changed')
    request=receipt['request'];verify_request(request)
    need(response.get('request_sha256')==receipt.get('request_sha256')==digest(request),'Bridge request binding differs')
    need(request['selection_lock']==admission.get('selection_lock_path') and
        request['selection_lock_sha256']==admission.get('selection_lock_sha256') and
        request['manifest']==admission.get('manifest_path') and request['manifest_sha256']==admission.get('manifest_sha256'),
        'Requested selection/manifest differs from original result')
    identity=receipt['process_identity'];parent=receipt['parent_pid']
    need(type(parent) is int and parent>1 and type(identity.get('pid')) is int and identity['pid']>1
        and identity.get('pgid')==identity.get('sid')==identity['pid'] and
        isinstance(identity.get('start_ticks'),str) and identity['start_ticks'].isdecimal(),'Owned worker identity invalid')
    need(receipt.get('command')==[str(OPENPI_PYTHON),'-B',str(Path(__file__).resolve()),'--worker','--parent-pid',str(parent)],
        'Bound worker argv differs')
    need(response.get('environment')==dict(python=str(OPENPI_PYTHON),prefix=str(OPENPI_PREFIX),numpy='1.26.4',scipy='1.15.3',
        cuda_visible_devices='',jax_platforms='cpu',pid=identity['pid'],parent_pid=parent),'Actual CPU worker environment differs')
    need(response.get('model_initialized') is False and response.get('service_contacted') is False and
        receipt.get('sdk_numpy_unchanged') is True and receipt.get('private_configuration_used') is False,
        'Admission execution scope differs')
    elapsed=receipt.get('elapsed_s')
    need(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=TIMEOUT_S+212,'CPU process duration invalid')
    need(receipt.get('process_errors')==[] and receipt.get('cleanup_signals')==[],'Admission process required cleanup or failed')
    need(response.get('admission')==admission and receipt.get('admission_sha256')==digest(admission),'Bridge selected admission differs')
    need(receipt.get('stdout_sha256')==digest(response) and receipt.get('stdout_bytes')==len(json_bytes(response))<=STDOUT_LIMIT,'Bound worker output changed')
    need(type(receipt.get('stderr_bytes')) is int and 0<=receipt['stderr_bytes']<=STDERR_LIMIT,'Worker stderr bound differs')
    actual=verify_admission_artifacts(admission)
    need(receipt.get('artifact_sha256_reverified')==actual,'Saved SDK artifact verification differs')
    return True


def raise_process_failure(error,contract,request,command,run):
    failure=dict(schema='wam_v7_coverage_pilot_cpu_admission_failure_v1',passed=False,
        reason=type(error).__name__+': '+str(error),source_contract=contract,request_sha256=digest(request),
        command=command,parent_pid=os.getpid(),process_identity=run.get('process_identity'),
        process_exit_code=run['exit_code'],process_group_drained=run['process_group_drained'],
        process_errors=run['errors'],cancelled=run['cancelled'],cleanup_signals=run['signals'],
        elapsed_s=run['elapsed_s'],stdout_sha256=file_bytes_sha(run['stdout']),stdout_bytes=len(run['stdout']),
        stderr_sha256=file_bytes_sha(run['stderr']),stderr_bytes=len(run['stderr']),
            stderr_excerpt_utf8=run['stderr'][:8192].decode('utf8',errors='replace'),
            stderr_excerpt_truncated=len(run['stderr'])>8192,private_configuration_used=False)
    try:
        response=read_json_bytes(run['stdout'])
        if isinstance(response,dict) and response.get('schema')==RESPONSE_SCHEMA:
            failure['worker_response_unaccepted']=response
    except (ValueError,UnicodeError):pass
    kind=AdmissionInterrupted if run['cancelled'] else AdmissionFailure
    raise kind(failure['reason'],failure) from error


def selected_admission_with_receipt(lock_path,lock_sha,manifest):
    contract=source_contract();root=root_path();lock=Path(lock_path).resolve();manifest=regular(Path(manifest).resolve())
    request=dict(schema=REQUEST_SCHEMA,source_contract=contract,source_project=str(root),selection_lock=str(lock),
        selection_lock_sha256=lock_sha,manifest=str(manifest),manifest_sha256=file_sha(manifest))
    verify_request(request)
    command=[str(OPENPI_PYTHON),'-B',str(Path(__file__).resolve()),'--worker','--parent-pid',str(os.getpid())]
    run=run_bounded(command,json_bytes(request),env=child_environment(root),cwd=root)
    try:
        need(not run['errors'] and not run['cancelled'] and run['process_group_drained'],'CPU admission process failed: '+'; '.join(run['errors']))
        need(run['exit_code'] in (0,2),'Original CPU admission returned nonzero exit: '+str(run['exit_code']))
        response=read_json_bytes(run['stdout'])
        if run['exit_code']==2 and response.get('status')=='not_ready':raise ValueError('Actual selected lock is not ready; no admission created')
        need(run['exit_code']==0,'Original CPU admission did not verify selection')
        admission=response.get('admission');need(isinstance(admission,dict),'Missing original admission result')
    except (ValueError,KeyError,TypeError,UnicodeError) as error:
        raise_process_failure(error,contract,request,command,run)
    try:
        receipt=dict(schema=RECEIPT_SCHEMA,process_exit_code=run['exit_code'],process_group_drained=run['process_group_drained'],
            process_identity=run['process_identity'],cancelled=run['cancelled'],elapsed_s=run['elapsed_s'],
            command=command,parent_pid=os.getpid(),request=request,request_sha256=digest(request),source_contract=contract,worker_response=response,
            process_errors=run['errors'],cleanup_signals=run['signals'],
            stdout_sha256=file_bytes_sha(run['stdout']),stdout_bytes=len(run['stdout']),stderr_sha256=file_bytes_sha(run['stderr']),stderr_bytes=len(run['stderr']),
            admission_sha256=digest(admission),artifact_sha256_reverified=verify_admission_artifacts(admission),
            sdk_numpy_unchanged=True,private_configuration_used=False)
        verify_receipt(receipt,admission);return admission,receipt
    except (ValueError,KeyError,TypeError,OSError) as error:
        raise_process_failure(error,contract,request,command,run)


def _parent_death_guard(expected_parent):
    need(sys.platform=='linux' and expected_parent==os.getppid(),'Worker parent identity differs')
    libc=ctypes.CDLL(None,use_errno=True)
    need(libc.prctl(1,signal.SIGKILL,0,0,0)==0,'Cannot arm CPU worker parent-death cleanup')
    need(expected_parent==os.getppid(),'Worker parent changed during cleanup setup')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--parent-pid',type=int,required=True);args=parser.parse_args()
    need(args.worker,'Explicit internal CPU worker mode required');_parent_death_guard(args.parent_pid)
    raw=sys.stdin.buffer.read(65537);need(len(raw)<=65536,'Admission request exceeds bound')
    response=worker_response(read_json_bytes(raw));sys.stdout.buffer.write(json_bytes(response));sys.stdout.buffer.flush()
    return 0 if response['status']=='verified' else 2


if __name__=='__main__':raise SystemExit(main())
