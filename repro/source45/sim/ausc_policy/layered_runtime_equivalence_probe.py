"""Finite original-owner Kit loading check. Software evidence, never physics.

The owner51 wrapper imports only stdlib modules and metadata validators. Its
one native60 child runs the original isolated preflights, then eight fresh
prepared_runtime_stage calls in a single Kit registry. No stage is attached.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import traceback

from . import native_usd_preflight as pre

EXPERIMENT='layered_runtime_equivalence_probe'
MODULE='sim.ausc_policy.layered_runtime_equivalence_probe'
SCHEMA='m2_layered_runtime_equivalence_probe_config_v1'
RESULT_SCHEMA='m2_layered_runtime_equivalence_probe_result_v1'
REQUEST_SCHEMA='m2_layered_runtime_equivalence_probe_owner_request_v1'
POLICY='hash_bound_static_usd_current_raw_and_sdk_v4'
OWNER_SECONDS=1800
NATIVE_SECONDS=1740
OWNER_PYTHON='/opt/wam-runtime/envs/isaacsim51/bin/python'
CASES=(('given-p001005-l3','p001005','validation','L3'),
       ('given-p000449-r3','p000449','train','R3'),
       ('given-p000465-r3','p000465','train','R3'),
       ('given-p000547-l4','p000547','train','L4'),
       ('given-p000945-l5','p000945','train','L5'),
       ('reg001-p000829-r1','p000829','train','R1'),
       ('reg004-p001197-l3','p001197','train','L3'),
       ('reg022-p001191-l4','p001191','train','L4'))
FALSE_FLAGS=dict(physics_executed=False,physical_validation=False,model_qualification=False,training_eligible=False)
HEADER_ROLES=('preparation_config','prepared_complete','patient_receipt','scene_authoring_receipt','layered_runtime_equivalence')


def need(ok,message):
    if not ok:raise ValueError(message)


def canonical(value,*,directory=False,exists=True):
    need(isinstance(value,(str,Path)),'Canonical absolute path required')
    path=Path(value)
    need(path.is_absolute() and '..' not in path.parts and str(value)==str(path) and
        str(path)==str(path.resolve()) and not any(p.is_symlink() for p in (path,*path.parents)),
        'Canonical path without aliases required')
    if exists:need(path.is_dir() if directory else path.is_file(),'Missing canonical input: '+str(path))
    return path


def reference_shape(ref):
    need(isinstance(ref,dict) and set(ref)=={'file','bytes','sha256'} and type(ref['bytes']) is int and
        ref['bytes']>=0 and isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,
        'Exact typed file/bytes/SHA reference required')
    return canonical(ref['file'])


def signature(ref,*,sdk=False):
    path=reference_shape(ref);meta=path.lstat()
    if sdk:
        from .sdk_mdl import APPROVED_ROOT
        need(APPROVED_ROOT in path.parents,'Hardlink allowance is restricted to approved SDK payload paths')
    need(stat.S_ISREG(meta.st_mode) and (meta.st_nlink in (1,2) if sdk else meta.st_nlink==1) and meta.st_size==ref['bytes'],
        'Single regular bound file required')
    return [meta.st_dev,meta.st_ino,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns,meta.st_nlink]


def read_ref(ref,*,sdk=False):
    reference_shape(ref)
    if sdk:
        before=signature(ref,sdk=True);data=Path(ref['file']).read_bytes()
        need(pre.exact(signature(ref,sdk=True),before),'Approved SDK input changed during read')
        actual=dict(file=ref['file'],bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    else:actual,data=pre.read_file(ref['file'])
    need(pre.exact(actual,ref),'Bound input bytes/SHA changed: '+ref['file'])
    return data


def json_ref(ref):return pre.decode(read_ref(ref))


def closed_inventory(root,*,sdk=False):
    root=canonical(root,directory=True);rows={}
    for path in root.rglob('*'):
        canonical(path,exists=False);meta=path.lstat()
        need(not stat.S_ISLNK(meta.st_mode),'Input inventory symlink')
        if stat.S_ISDIR(meta.st_mode):continue
        need(stat.S_ISREG(meta.st_mode) and (meta.st_nlink in (1,2) if sdk else meta.st_nlink==1),
            'Single regular closed inventory input required')
        rows[str(path)]=[meta.st_dev,meta.st_ino,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns,meta.st_nlink]
    return rows


def full_reference(value):
    need(isinstance(value,dict) and set(value) in ({'file','sha256'},{'file','bytes','sha256'}),
        'Exact historical metadata reference required')
    ref=dict(value)
    if 'bytes' not in ref:ref['bytes']=canonical(ref['file']).stat().st_size
    signature(ref);return ref


def sdk_role(role):return re.fullmatch('sdk_mdl_[0-9]{4}',role) is not None


def _runtime_header(runtime,case):
    need(isinstance(runtime,dict) and not any(key in runtime for key in
        ('m2_act_deployment','m2_given_contact_deployment','m2_given_contact_evaluation','m2_given_contact_heldout_planning_aggregate')),
        'Expert TRAIN/VAL config required; learned/TEST deployment forbidden')
    binding=runtime.get('compact_scene_storage',{})
    need(isinstance(binding,dict) and set(binding)=={'schema','store_root','batch_id','dependency_policy','sources'} and
        binding['schema']=='m2_compact_scene_storage_v3' and binding['dependency_policy']==POLICY,
        'Explicit current RAW and SDK v4 policy required')
    refs=binding['sources'];need(isinstance(refs,dict) and set(HEADER_ROLES)<=set(refs),'All layered metadata roles required')
    for role,ref in refs.items():signature(ref,sdk=sdk_role(role))
    sidecar=refs['layered_runtime_equivalence']
    need(Path(sidecar['file']).suffix=='.json' and all(sidecar['file']!=ref['file'] for role,ref in refs.items()
        if role!='layered_runtime_equivalence'),'Sidecar role confusion forbidden')
    route=runtime.get('m2_given_contact_control',{});fresh=runtime.get('fresh_expert',{})
    raw=fresh.get('raw_multimodal_capture',{}) if isinstance(fresh,dict) else {}
    metadata=route.get('metadata',{}) if isinstance(route,dict) else {}
    need(isinstance(raw,dict) and isinstance(metadata,dict) and route.get('schema')=='m2_given_contact_pose_control_v1' and
        metadata.get('subject_id')==case['subject_id'] and metadata.get('split')==case['split'] and
        metadata.get('target_id')==case['site'] and metadata.get('target_source')=='given_contact_pose' and
        isinstance(route.get('episode_id'),str) and bool(route['episode_id']) and
        metadata.get('episode_id')==route['episode_id']==raw.get('episode_id') and raw.get('enabled') is True and
        raw.get('profile')=='compact' and pre.exact(raw.get('compact',{}).get('metadata'),metadata),
        'Exact prescribed expert identity/capture headers required')
    return binding


def load_inputs(config_file):
    """Authenticate the entire population's JSON headers without USD/body I/O."""
    path=canonical(config_file);config_ref,data=pre.read_file(path);config=pre.decode(data)
    keys={'schema','experiment','base','runtime_project','isaac_python','ld_library_path','time_limit_seconds',
          'child_timeout_seconds','source','cases'}
    need(isinstance(config,dict) and set(config)==keys and config['schema']==SCHEMA and config['experiment']==EXPERIMENT and
        type(config['time_limit_seconds']) is int and config['time_limit_seconds']==OWNER_SECONDS and
        type(config['child_timeout_seconds']) is int and config['child_timeout_seconds']==NATIVE_SECONDS,
        'Exact finite software probe config required')
    canonical(config['base'],directory=True);canonical(config['runtime_project'],directory=True)
    need(config['isaac_python']==pre.NATIVE_PYTHON,'Fixed native60 interpreter required')
    need(isinstance(config['ld_library_path'],str) and bool(config['ld_library_path']),'Exact native library path required')
    for part in config['ld_library_path'].split(':'):canonical(part,directory=True,exists=False)
    source=pre.source_snapshot()
    need(pre.exact(config['source'],source) and config['runtime_project']==source['root'],'Frozen source/config root changed')
    need(pre.file_ref(__file__)['file']==str(Path(source['root'])/'sim/ausc_policy/layered_runtime_equivalence_probe.py'),
        'Probe module must come from the frozen source root')
    cases=config['cases'];need(isinstance(cases,list) and len(cases)==len(CASES),'Exactly eight ordered cases required')
    # All case identities/reference shapes precede even the first runtime JSON.
    for case,expected in zip(cases,CASES):
        need(isinstance(case,dict) and set(case)=={'case_id','subject_id','split','site','runtime_config'} and
            pre.exact([case[k] for k in ('case_id','subject_id','split','site')],list(expected)),
            'Exact ordered TRAIN/VAL case identity required')
        signature(case['runtime_config'])
    need(len({c['runtime_config']['file'] for c in cases})==8,'Unique runtime config files required')
    admitted=[];baselines={}
    for case in cases:
        runtime=json_ref(case['runtime_config']);binding=_runtime_header(runtime,case)
        admitted.append(dict(case, runtime=runtime))
        for role,ref in [('runtime_config',case['runtime_config']),*binding['sources'].items()]:
            old=baselines.get(ref['file'])
            sdk=sdk_role(role);row=dict(reference=ref,signature=signature(ref,sdk=sdk),sdk=sdk)
            need(old is None or pre.exact(old,row),'Conflicting input file roles')
            baselines[ref['file']]=row
    from . import scene_store
    inventories={};header_sets=[]
    for case in admitted:
        binding=case['runtime']['compact_scene_storage']
        headers={role:json_ref(binding['sources'][role]) for role in HEADER_ROLES}
        prep,complete,patient=scene_store._binding_metadata(binding)
        need(patient.get('identity')==case['subject_id'] and patient.get('split')==case['split'] and
            patient.get('non_patient_scene_unchanged') is True and prep.get('prepare_only') is True and
            complete.get('completed') is True and pre.exact(complete.get('patient'),patient),
            'Prepared TRAIN/VAL metadata identity mismatch')
        header_sets.append(headers)
    # Population authentication above is complete. Gather only schema-known
    # provenance roles, never arbitrary dictionaries from asset/SDK manifests.
    def retain(value):
        ref=full_reference(value);row=dict(reference=ref,signature=signature(ref),sdk=False);old=baselines.get(ref['file'])
        need(old is None or pre.exact(old,row),'Conflicting provenance file references')
        baselines[ref['file']]=row
        if Path(ref['file']).name=='COMPLETE.json':
            root=str(Path(ref['file']).parent)
            inventories[root]=dict(sdk=False,files=closed_inventory(root))
    for case,headers in zip(admitted,header_sets):
        receipt=headers['scene_authoring_receipt'];sidecar=headers['layered_runtime_equivalence']
        for group in receipt.get('input_bindings',[]):
            for ref in group:retain(ref)
        for selection in receipt.get('original_selected',[]):
            for role in ('preparation_config','prepared_complete','patient_receipt'):
                if role in selection:retain(selection[role])
        for ref in sidecar.get('historical_prepared_sources',{}).values():retain(ref)
        if 'historical_prepared_complete' in sidecar:retain(sidecar['historical_prepared_complete'])
        # Historical authoring receipts bind the complete original prepared trees.
        binding=case['runtime']['compact_scene_storage'];root=str(Path(binding['sources']['prepared_complete']['file']).parent)
        inventories[root]=dict(sdk=False,files=closed_inventory(root))
        if any(sdk_role(role) for role in binding['sources']):
            from .sdk_mdl import APPROVED_ROOT
            root=str(APPROVED_ROOT);inventories[root]=dict(sdk=True,files=closed_inventory(root,sdk=True))
    return dict(config=config,config_ref=config_ref,source=source,cases=admitted,baselines=baselines,inventories=inventories)


def rebind(inputs,*,full=False):
    need(pre.exact(pre.file_ref(inputs['config_ref']['file']),inputs['config_ref']) and
        pre.exact(pre.source_snapshot(),inputs['source']),'Source or raw configuration changed')
    for row in inputs['baselines'].values():
        need(pre.exact(signature(row['reference'],sdk=row['sdk']),row['signature']),'Input identity/stat changed')
        if full:read_ref(row['reference'],sdk=row['sdk'])
    for root,before in inputs['inventories'].items():
        need(pre.exact(closed_inventory(root,sdk=before['sdk']),before['files']),'Closed provenance input inventory changed')
    return True


def write_new(path,value):
    with Path(path).open('xb') as stream:
        stream.write(pre.encoded(value)+b'\n');stream.flush();os.fsync(stream.fileno())


def publish_complete(path,value):
    """Never expose a COMPLETE marker before its bytes and directory are durable."""
    path=Path(path);pending=path.with_name('.'+path.name+'.pending')
    need(not path.exists(),'Refusing to replace an existing COMPLETE')
    try:
        write_new(pending,value)
        os.rename(pending,path)
        descriptor=os.open(str(path.parent),os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    except BaseException:
        pending.unlink(missing_ok=True);path.unlink(missing_ok=True)
        raise


def process_identity(pid):
    from .process_guard import process_identity as original_identity
    value=original_identity(pid)
    return {key:value[key] for key in ('pid','pgid','sid','start_ticks')}


def require_owner(inputs,output,*,native=False):
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and sys.dont_write_bytecode and
        not sys.flags.optimize and sys.executable==(pre.NATIVE_PYTHON if native else OWNER_PYTHON),
        'Original owned unoptimized -B interpreter required')
    output=canonical(output,exists=False);folder=output.parent if not native else output.parent.parent
    need(folder.parent==Path(inputs['config']['base'])/'runs/ausc_policy' and
        re.fullmatch('[a-z0-9_-]+',folder.name) is not None and output==folder/'output'/('native' if native else ''),
        'Original owner output namespace required')
    run=pre.decode(pre.read_file(folder/'run.json')[1]);root=Path(inputs['source']['root'])
    expected=[OWNER_PYTHON,'-u','-B','-m',MODULE,'--config',inputs['config_ref']['file'],'--output',str(folder/'output')]
    need(pre.exact(run.get('config'),inputs['config']) and run.get('config_sha256')==inputs['config_ref']['sha256'] and
        run.get('code_commit')==inputs['source']['code_commit'] and run.get('command')==expected and
        run.get('completed') is False,'Original owner run/config/source command binding required')
    guard=run.get('resource_guard',{})
    need(guard.get('path')==str(root/'sim/ausc_policy/process_guard.py') and
        guard.get('sha256')==pre.file_ref(guard['path'])['sha256'],'Original source-bound process guard required')
    identity=process_identity(os.getpid())
    if not native:need(identity['pid']==identity['pgid']==identity['sid'],'Original wrapper session leader required')
    return identity


def gpu_idle():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    need(not value.strip(),'GPU occupied at live locked idle check')
    return dict(pids=[],checked_monotonic=time.monotonic())


def run_native_child(command,environment,folder,*,timeout_s=NATIVE_SECONDS):
    """One inherited-group direct child, bounded TERM/KILL and actual reaping."""
    need(0<timeout_s<=NATIVE_SECONDS,'Fixed bounded native child timeout required')
    identity=None;child=None;timed_out=False;error=None;started=time.monotonic();deferred=[];status=None
    prior={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def defer(signum,frame):deferred.append(signal.Signals(signum).name)
    def interrupt(signum,frame):raise InterruptedError('Owner wrapper received '+signal.Signals(signum).name)
    def handlers(fn):
        for sig in prior:signal.signal(sig,fn)
    with (Path(folder)/'isaac.log').open('xb') as log:
        try:
            handlers(defer)
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,
                cwd=environment.get('PYTHONPATH'),start_new_session=False)
            identity=process_identity(child.pid)
            need(identity['pgid']==os.getpgrp() and identity['sid']==os.getsid(0),'Native worker escaped inherited owner group')
            handlers(interrupt)
            if deferred:raise InterruptedError('Owner wrapper interrupted during native startup')
            try:child.wait(timeout=max(.001,timeout_s-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:timed_out=True
        except BaseException as caught:error=caught
        finally:
            handlers(defer)
            try:
                if child is not None:
                    try:
                        if child.poll() is None:
                            # Popen retains the direct child; never signal the owner's whole group.
                            child.terminate()
                            try:child.wait(timeout=15)
                            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
                        else:child.wait(timeout=5)
                    except BaseException as caught:
                        if error is None:error=caught
                    if deferred and error is None:error=InterruptedError('Owner wrapper interrupted during native lifecycle')
                    status=dict(pid=child.pid,identity=identity,command=command,returncode=child.poll(),
                        timed_out=timed_out,reaped=child.poll() is not None,wall_s=time.monotonic()-started,
                        started_monotonic=started,deferred_signals=deferred,
                        error=None if error is None else dict(type=type(error).__name__,message=str(error)),**FALSE_FLAGS)
                    try:write_new(Path(folder)/'native_worker_exit.json',status)
                    except BaseException as caught:
                        if error is None:error=caught
                    if deferred and error is None:error=InterruptedError('Owner wrapper interrupted during terminal receipt')
            finally:
                for sig,handler in prior.items():signal.signal(sig,handler)
    if error is not None:raise error
    return status


def remaining_descendants(identity):
    from .process_guard import group_members
    return [member for member in group_members(identity) if member['pid']!=identity['pid']]


def validate_native_receipt(report,inputs,status,parent):
    keys={'schema','completed','config_ref','source','identity','cases','cpu_preflights','failed_case','registry',
          'normal_close','error','cleanup_error',*FALSE_FLAGS}
    need(isinstance(report,dict) and set(report)==keys and report.get('schema')==RESULT_SCHEMA and
        report.get('completed') is True and pre.exact(report.get('config_ref'),inputs['config_ref']) and
        pre.exact(report.get('source'),inputs['source']) and report.get('error') is None and
        report.get('cleanup_error') is None and report.get('failed_case') is None and
        all(report.get(k) is False for k in FALSE_FLAGS),'Exact successful software-only receipt required')
    identity=report.get('identity',{})
    need(isinstance(identity,dict) and set(identity)=={'pid','ppid','pgid','sid','start_ticks'} and
        pre.exact({k:identity[k] for k in ('pid','pgid','sid','start_ticks')},status.get('identity')) and
        type(identity['ppid']) is int and identity['ppid']==parent['pid'] and
        all(identity[k]==parent[k] for k in ('pgid','sid')),'Native receipt PID/start/group differs from actual child')
    close=report.get('normal_close',{})
    from .native_lifecycle import shared_app_settings
    need(isinstance(close,dict) and set(close)=={'app_close_returned','fast_shutdown','software_only','stage_cache_leases','settings'} and
        close.get('app_close_returned') is True and close.get('fast_shutdown') is False and close.get('software_only') is True and
        pre.exact(close.get('stage_cache_leases'),[]) and pre.exact(close.get('settings'),shared_app_settings()),
        'Normal software-only shared-settings close evidence required')
    rows=report.get('cases');proofs=report.get('cpu_preflights')
    need(isinstance(rows,list) and len(rows)==8 and isinstance(proofs,list) and len(proofs)==8,'Eight stage returns and CPU proofs required')
    for case,row,proof in zip(inputs['cases'],rows,proofs):
        need(isinstance(row,dict) and set(row)=={'case_id','runtime_config','returned','private_anonymous','layer_identifiers'} and
            row.get('case_id')==case['case_id'] and pre.exact(row.get('runtime_config'),case['runtime_config']) and
            row.get('returned') is True and row.get('private_anonymous') is True and isinstance(row.get('layer_identifiers'),list) and
            bool(row['layer_identifiers']) and all(isinstance(v,str) and v.startswith('anon:') for v in row['layer_identifiers']),
            'Exact private stage return evidence required')
        need(isinstance(proof,dict) and all(proof.get(k) is True for k in
            ('passed','full_original_validator','parent_USD_clean','child_reaped')) and
            type(proof.get('child_exit_code')) is int and proof['child_exit_code']==0 and
            all(proof.get(k) is False for k in ('physics_executed','kit_started','training_eligible')) and
            pre.exact(proof.get('config_ref'),case['runtime_config']) and pre.exact(proof.get('source'),inputs['source']) and
            pre.exact(proof.get('parent_identity'),identity),'Exact full original isolated preflight receipt required')
    return report


def run_owned(config_file,output):
    started=time.monotonic();inputs=load_inputs(config_file);identity=require_owner(inputs,output)
    output=canonical(output,exists=False);output.mkdir(exist_ok=False)
    config=inputs['config'];command=[pre.NATIVE_PYTHON,'-u','-B','-m',MODULE,'--native-worker',
        '--config',inputs['config_ref']['file'],'--output',str(output/'native')]
    request=dict(schema=REQUEST_SCHEMA,parent=identity,command=command,config_ref=inputs['config_ref'],source=inputs['source'],
        deadline_monotonic=started+NATIVE_SECONDS)
    write_new(output/'native_worker_request.json',request)
    lock_path=canonical(Path(config['base'])/'locks/isaac_gpu.lock',exists=False)
    canonical(lock_path.parent,directory=True)
    if lock_path.exists():
        meta=lock_path.lstat();need(stat.S_ISREG(meta.st_mode) and meta.st_nlink==1,'Original lock must be one regular file')
    status=None;idle=[];descendants=None
    with os.fdopen(os.open(str(lock_path),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o644),'a+') as lock:
        meta=os.fstat(lock.fileno())
        need(stat.S_ISREG(meta.st_mode) and meta.st_nlink==1 and
            (meta.st_dev,meta.st_ino)==(lock_path.stat().st_dev,lock_path.stat().st_ino),'Original lock identity changed')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            idle=[gpu_idle()];rebind(inputs);idle.append(gpu_idle())
            environment=dict(os.environ,LD_LIBRARY_PATH=config['ld_library_path'],OMNI_KIT_ACCEPT_EULA='YES',
                PYTHONPATH=inputs['source']['root'],CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',
                PYTHONOPTIMIZE='0',PYTHONNOUSERSITE='1')
            remaining=request['deadline_monotonic']-time.monotonic();need(remaining>0,'Native deadline expired before launch')
            status=run_native_child(command,environment,output,timeout_s=remaining)
            need(status.get('returncode')==0 and status.get('reaped') is True and status.get('timed_out') is False and
                status.get('error') is None,'Native software probe failed or timed out; preserved child evidence')
            descendants=remaining_descendants(identity);need(not descendants,'Native descendants remain for original owner cleanup')
            report=validate_native_receipt(pre.decode(pre.read_file(output/'native/COMPLETE.json')[1]),inputs,status,identity)
            rebind(inputs,full=True)
            write_new(output/'wrapper_status.json',dict(completed=True,config_ref=inputs['config_ref'],source=inputs['source'],
                parent=identity,native=status,idle_checks=idle,native_descendants_drained=True,
                terminal_group_drain_owner='sim.ausc_policy.process_guard',error=None,**FALSE_FLAGS))
            publish_complete(output/'COMPLETE.json',dict(report,native_exit=status,owner_wrapper=identity,idle_checks=idle))
        except BaseException as caught:
            failure=dict(completed=False,config_ref=inputs['config_ref'],source=inputs['source'],parent=identity,
                native=status,idle_checks=idle,native_descendants_drained=descendants==[],
                terminal_group_drain_owner='sim.ausc_policy.process_guard',
                error=dict(type=type(caught).__name__,message=str(caught)),**FALSE_FLAGS)
            try:write_new(output/'wrapper_failure.json',failure)
            except BaseException:pass  # The original launch/cleanup failure always wins.
            raise
    return report


def simulation_app_class():
    from isaacsim import SimulationApp
    return SimulationApp


def registry_diagnostics():
    from pxr import Plug,Usd
    modules={name:getattr(module,'__file__',None) for name,module in sorted(sys.modules.items())
        if name=='pxr' or name.startswith('pxr.')}
    plugins=[dict(name=p.name,path=p.path,resource_path=p.resourcePath,is_loaded=p.isLoaded)
        for p in Plug.Registry().GetAllPlugins()]
    return dict(version=list(Usd.GetVersion()),modules=modules,registry=dict(plugins=sorted(plugins,key=lambda p:(p['name'],p['path']))))


def exercise(inputs,output,deadline):
    """Actual registry loop; all inputs are authenticated before app startup."""
    from . import native_lifecycle,scene_store
    output=canonical(output,exists=False);output.mkdir(exist_ok=False)
    app=None;error=None;error_record=None;cleanup_errors=[];close=None;failed_case=None;rows=[];proofs=[];diagnostics=None;stage=None;layers=None
    identity=pre.native_identity()
    def mark(event,**values):
        with (output/'progress.jsonl').open('ab') as stream:
            stream.write(pre.encoded(dict(event=event,monotonic=time.monotonic(),**values))+b'\n');stream.flush();os.fsync(stream.fileno())
    def check_deadline():need(time.monotonic()<deadline,'Native software probe deadline reached')
    def exception_record(caught):
        return dict(type=type(caught).__name__,message=str(caught),traceback=''.join(traceback.format_exception(type(caught),caught,caught.__traceback__)))
    def cleanup_failure(caught):
        nonlocal error,error_record
        cleanup_errors.append(exception_record(caught))
        if error is None:error=caught;error_record=cleanup_errors[-1]
    try:
        pre.clean_parent();rebind(inputs)
        for case in inputs['cases']:
            check_deadline();failed_case=case['case_id'];mark('cpu_preflight_enter',case_id=failed_case)
            proof=pre.validate_before_kit(case['runtime'],case['runtime_config']['file'])
            need(all(proof.get(k) is True for k in ('passed','full_original_validator','parent_USD_clean','child_reaped')) and
                proof.get('child_exit_code')==0 and type(proof.get('child_exit_code')) is int and
                all(proof.get(k) is False for k in ('physics_executed','kit_started','training_eligible')) and
                pre.exact(proof.get('config_ref'),case['runtime_config']) and pre.exact(proof.get('parent_identity'),identity),
                'Full original isolated preflight proof required')
            proofs.append(proof);mark('cpu_preflight_return',case_id=failed_case)
        rebind(inputs,full=True);check_deadline()
        need(scene_store.runtime_stage_leases()==[],'Software probe cannot own stage-cache leases')
        pre.clean_parent()
        app=simulation_app_class()(native_lifecycle.shared_app_settings());mark('simulation_app_return')
        diagnostics=registry_diagnostics()
        for case in inputs['cases']:
            check_deadline();failed_case=case['case_id'];mark('current_raw_and_sdk_enter',case_id=failed_case)
            stage=scene_store.prepared_runtime_stage(case['runtime']['compact_scene_storage'])
            need(stage is not None and stage.GetRootLayer().anonymous,'Private anonymous returned root required')
            layers=stage.GetUsedLayers();need(bool(layers) and all(layer.anonymous for layer in layers),'Every returned layer must be private anonymous')
            rows.append(dict(case_id=failed_case,runtime_config=case['runtime_config'],returned=True,private_anonymous=True,
                layer_identifiers=[layer.identifier for layer in layers]))
            stage=None;layers=None;gc.collect();mark('current_raw_and_sdk_return',case_id=failed_case)
            need(scene_store.runtime_stage_leases()==[],'Software stage call acquired a cache lease')
        failed_case=None;rebind(inputs,full=True);check_deadline()
    except BaseException as caught:
        error=caught;error_record=exception_record(caught)
        traceback.clear_frames(caught.__traceback__)
    finally:
        stage=None;layers=None
        try:gc.collect()
        except BaseException as caught:cleanup_failure(caught)
        if app is not None:
            try:
                need(scene_store.runtime_stage_leases()==[],'Software probe acquired cache leases before close')
            except BaseException as caught:cleanup_failure(caught)
            try:mark('normal_close_enter')
            except BaseException as caught:cleanup_failure(caught)
            try:
                # No World, attached stage, viewport or owned capture products exist
                # to quiesce. The normal fast_shutdown=False close is invoked once.
                app.close()
                close=dict(app_close_returned=True,fast_shutdown=False,software_only=True,stage_cache_leases=[],
                    settings=native_lifecycle.shared_app_settings())
            except BaseException as caught:cleanup_failure(caught)
            else:
                try:mark('normal_close_return')
                except BaseException as caught:cleanup_failure(caught)
            # An empty-map check has no USD API calls after app.close.
            try:need(not scene_store._RUNTIME_STAGE_LEASES,'Unexpected cache leases after normal software close')
            except BaseException as caught:cleanup_failure(caught)
        try:rebind(inputs,full=True);check_deadline()
        except BaseException as caught:
            if error is None:error=caught;error_record=exception_record(caught)
    report=dict(schema=RESULT_SCHEMA,completed=error is None,config_ref=inputs['config_ref'],source=inputs['source'],
        identity=identity,cases=rows,cpu_preflights=proofs,failed_case=failed_case,registry=diagnostics,normal_close=close,
        error=error_record,cleanup_error=cleanup_errors or None,**FALSE_FLAGS)
    if error is None:
        try:publish_complete(output/'COMPLETE.json',report)
        except BaseException as caught:
            error=caught;report=dict(report,completed=False,error=exception_record(caught))
    if error is not None:
        try:write_new(output/'FAILURE.json',report)
        except BaseException:pass  # Preserve an oracle failure even if its journal cannot be fsynced.
    if error is not None:raise error
    return report


def native_worker(config_file,output):
    pre.clean_parent();pre.native_identity();inputs=load_inputs(config_file)
    identity=require_owner(inputs,output,native=True)
    request=pre.decode(pre.read_file(Path(output).parent/'native_worker_request.json')[1])
    need(isinstance(request,dict) and set(request)=={'schema','parent','command','config_ref','source','deadline_monotonic'} and
        request['schema']==REQUEST_SCHEMA and pre.exact(request['config_ref'],inputs['config_ref']) and
        pre.exact(request['source'],inputs['source']) and type(request['deadline_monotonic']) is float and
        0<request['deadline_monotonic']-time.monotonic()<=NATIVE_SECONDS,'Original bounded owner request required')
    parent=request['parent'];need(parent.get('pid')==os.getppid() and all(parent.get(k)==identity[k] for k in ('pgid','sid')) and
        pre.exact(process_identity(os.getppid()),parent),'Original native parent PID/start/group required')
    command=[pre.NATIVE_PYTHON,'-u','-B','-m',MODULE,'--native-worker','--config',inputs['config_ref']['file'],'--output',str(output)]
    need(request['command']==command,'Exact original native worker command required')
    def stop(signum,frame):raise InterruptedError('Native software probe received '+signal.Signals(signum).name)
    prior=signal.signal(signal.SIGTERM,stop)
    try:return exercise(inputs,output,request['deadline_monotonic'])
    finally:signal.signal(signal.SIGTERM,prior)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-worker',action='store_true');args=parser.parse_args()
    try:
        if args.native_worker:
            native_worker(args.config,args.output)
            # The receipt is fsynced only after normal close and all postchecks.
            sys.stdout.flush();sys.stderr.flush();os._exit(0)
        run_owned(args.config,args.output)
    except SystemExit as caught:
        # Preserve each owned route's original failure evidence and chaining.
        # Argument parsing above retains its normal --help exit behavior.
        if caught.code is None or caught.code==0:raise SystemExit(1) from caught
        raise


if __name__=='__main__':main()
