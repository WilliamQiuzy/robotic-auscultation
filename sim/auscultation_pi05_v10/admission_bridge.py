"""Owned OpenPI CPU verification at the original external scientific producer.

Physical/service processes never import the V10 trainer, evaluator or parity.
The wrapper arms parent-death cleanup, then execs the pinned producer unchanged.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from sim.auscultation_pi05_v7 import coverage_pilot_admission_bridge as owned

BASE=owned.BASE
PRODUCER=BASE/'cloud-runtime-v10/recovery-B7300-20260928-v1'
PARENT_INVENTORY=PRODUCER.with_name(PRODUCER.name+'.source_inventory.json')
PARENT_SHA='1caaec3789715bd3798a902bc1ea84819b482e999c93fe0c39fb8cd490cab2f5'
ENTRY=BASE/'diagnostics/pi05_v10_postlock_runtime_v1/postlock.py'
OWNED_SHA='95e39b8dd739ba3772d91892bf7c729ee8fd3a97d5b9eedc59afdbc13e05f10d'
NORM_SHA='0d374775d5b85b9b22a833b1b20afc69540b8b7c2f7ddd80e63de16d089e6794'
OBJECT_SHA='313b47e59585a4d248102ca2babe1cffd19c7375e224702267d34ebdee756170'
EXTENSIONS=('actor.py','policy_contract.py','policy_client.py','policy_service.py','admission_bridge.py','runtime_contract.py','run_headroom_isaac.py','audit_actions.py','assess_trials.py')
DESCRIPTOR_SCHEMA='wam_pi05_v10_admission_verifier_descriptor_v1'
ADMISSION_SCHEMA='wam_pi05_v10_recovery_runtime_admission_v1'
REQUEST_SCHEMA='wam_pi05_v10_recovery_cpu_admission_request_v1'
RECEIPT_SCHEMA='wam_pi05_v10_recovery_cpu_admission_bridge_receipt_v1'
AdmissionFailure=owned.AdmissionFailure
AdmissionInterrupted=owned.AdmissionInterrupted
need,file_sha,digest,json_bytes,read_json_bytes=owned.need,owned.file_sha,owned.digest,owned.json_bytes,owned.read_json_bytes


def regular(path):
    p=Path(path);need(p.is_absolute() and p==p.resolve() and all(not x.is_symlink() for x in (p,*p.parents)) and p.is_file(),'Canonical regular input required: '+str(p));return p


def read_bound(path,expected_sha):
    raw=regular(path).read_bytes();need(hashlib.sha256(raw).hexdigest()==expected_sha,'Admitted bytes changed before parsing: '+str(path));return read_json_bytes(raw)


def read_binding(b):
    need(type(b) is dict and set(b)=={'path','sha256'},'Exact file binding required');return read_bound(b['path'],b['sha256'])


def add_artifacts(artifacts,more):
    for p,h in more.items():
        need(p not in artifacts or artifacts[p]==h,'Cannot rebind consumed artifact: '+p)
        need(file_sha(regular(p))==h,'Consumed artifact changed: '+p);artifacts[p]=h


def verify_tree(root,files):
    root=Path(root);need(root.is_absolute() and root==root.resolve() and root.is_dir(),'Canonical source project required')
    tree=[root,*root.rglob('*')]
    need(all(not p.is_symlink() and (p.is_file() or p.is_dir()) and not p.stat().st_mode&0o222 for p in tree),'Read-only regular source tree required')
    need({str(p.relative_to(root)) for p in tree if p.is_file()}==set(files),'Exact source tree required')
    for name,h in files.items():
        rel=Path(name);need(not rel.is_absolute() and '..' not in rel.parts,'Unsafe relative source')
        p=regular(root/rel);need(p.stat().st_nlink==1 and file_sha(p)==h,'Independent source bytes changed')


def root_path():return Path(__file__).resolve().parents[2]


def source_contract(verifier_binding=None):
    if verifier_binding is None:
        root=root_path();ip=root.with_name(root.name+'.source_inventory.json')
        verifier_binding=read_bound(ip,file_sha(regular(ip)))['admission_verifier_binding']
    descriptor=read_binding(verifier_binding)
    need(set(descriptor)=={'schema','entry_binding','producer_source_project','producer_inventory_binding','interpreter_binding'} and descriptor['schema']==DESCRIPTOR_SCHEMA,'Exact verifier descriptor required')
    need(descriptor['producer_source_project']==str(PRODUCER) and descriptor['producer_inventory_binding']==dict(path=str(PARENT_INVENTORY),sha256=PARENT_SHA),'Original376 scientific source path required')
    need(descriptor['entry_binding']['path']==str(ENTRY),'Unique external producer entry required')
    entry=descriptor['entry_binding'];need(file_sha(regular(entry['path']))==entry['sha256'],'External producer source changed')
    original=read_bound(PARENT_INVENTORY,PARENT_SHA);old=original['source_files']
    need(original.get('source_project')==str(PRODUCER) and len(old)==376,'Exact original376 inventory required');verify_tree(PRODUCER,old)
    runtime=root_path();need(runtime.parent==BASE/'cloud-runtime-v10' and runtime!=PRODUCER,'Independent physical385 required')
    ip=runtime.with_name(runtime.name+'.source_inventory.json');ih=file_sha(regular(ip));inventory=read_bound(ip,ih);files=inventory['source_files']
    extension={'sim/auscultation_pi05_v10/'+n for n in EXTENSIONS}
    need(inventory.get('schema')=='wam_pi05_v10_recovery_physical_runtime_inventory_v1' and inventory.get('source_project')==str(runtime) and inventory.get('parent_inventory')==descriptor['producer_inventory_binding'] and inventory.get('admission_verifier_binding')==verifier_binding,'Physical inventory identity differs')
    need(set(files)==set(old)|extension and len(files)==385 and all(files[k]==v for k,v in old.items()),'Exact376+9 source ancestry required');verify_tree(runtime,files)
    interpreter=descriptor['interpreter_binding']
    need(set(interpreter)=={'path','resolved_path','sha256','prefix'} and interpreter['path']==str(owned.OPENPI_PYTHON) and interpreter['prefix']==str(owned.OPENPI_PREFIX) and interpreter['resolved_path']==str(owned.OPENPI_PYTHON.resolve(strict=True)) and interpreter['sha256']==owned.PYTHON_SHA and file_sha(owned.OPENPI_PYTHON)==owned.PYTHON_SHA,'Requested/resolved official interpreter differs')
    need(file_sha(owned.__file__)==OWNED_SHA and file_sha(regular(owned.GUARD_PATH))==owned.GUARD_SHA,'Original owned CPU process implementation differs')
    artifacts={verifier_binding['path']:verifier_binding['sha256'],str(PARENT_INVENTORY):PARENT_SHA,str(ip):ih,entry['path']:entry['sha256'],interpreter['resolved_path']:interpreter['sha256'],str(owned.GUARD_PATH):owned.GUARD_SHA}
    add_artifacts(artifacts,{str(PRODUCER/k):h for k,h in old.items()});add_artifacts(artifacts,{str(runtime/k):h for k,h in files.items()})
    site=owned.OPENPI_PREFIX/'lib/python3.11/site-packages'
    add_artifacts(artifacts,{str(site/'numpy/__init__.py'):owned.NUMPY_SHA,str(site/'scipy/__init__.py'):owned.SCIPY_SHA})
    return dict(schema='wam_pi05_v10_recovery_cpu_bridge_contract_v1',verifier_binding=copy.deepcopy(verifier_binding),entry_binding=entry,producer_source_project=str(PRODUCER),producer_inventory_binding=descriptor['producer_inventory_binding'],worker_python=interpreter,runtime_inventory_binding=dict(path=str(ip),sha256=ih),source_project=str(runtime),owned_process_implementation_sha256=OWNED_SHA,owned_process_guard=dict(path=str(owned.GUARD_PATH),sha256=owned.GUARD_SHA),timeout_s=owned.TIMEOUT_S,stdout_limit_bytes=owned.STDOUT_LIMIT,stderr_limit_bytes=owned.STDERR_LIMIT,artifact_sha256=artifacts,scientific_algorithm='external V10 postlock.verify_admission',model_initialized=False,physical_numpy_unchanged=True)


def worker_command(contract,path,sha):return [contract['worker_python']['path'],'-B',contract['entry_binding']['path'],'--verify-admission',str(path),'--admission-sha256',sha]


def no_marker(path):
    p=Path(path);need(not p.exists() and not p.is_symlink(),'Late cancellation/failure marker: '+str(p))


def verify_artifacts(admission):
    a=admission['artifact_sha256'];need(type(a) is dict and a,'Nonempty consumed artifact map required');add_artifacts({},a)
    scientific=admission['scientific_admission'];no_marker(Path(admission['admission_binding']['path']).with_suffix('.invalid.json'))
    cp=admission['checkpoint_binding'];run=Path(cp['checkpoint_path']).parent.parent
    def pinned(p):return read_bound(p,a[str(p)])
    terminal=pinned(run/'training_run.json');segment=Path(terminal['segment']);need(segment.parent==run/'training_segments' and pinned(segment)==terminal,'Current terminal differs');no_marker(segment.with_suffix('.cancelled.json'))
    owner=pinned(run/'TRAINING_OWNER.json');life=Path(owner['lifecycle']['path']);pinned(life)
    for folder in [life.parent]:
        need(not any(p.name=='late_interrupt.json' or p.name.endswith(('.late_interrupt.json','.cancelled.json')) for p in folder.iterdir()),'Owner interrupted')
    for b in (scientific['probes_binding'],scientific['test_binding']):
        folder=Path(b['path']).parent
        for name in ('failure.json','publication_failure.json','cancelled.json'):no_marker(folder/name)
        execution=pinned(folder/'execution.json');guard=Path(execution['guard_evidence_binding']['path']);pinned(guard)
        no_marker(guard.with_name(guard.name[:-len('.process.json')]+'.late_interrupt.json'));no_marker(guard.parent/'late_interrupt.json')
    add_artifacts({},a);return dict(a)


def build_runtime_admission(v,path,expected_sha,contract):
    need(v.get('schema')=='wam_pi05_v10_recovery_admission_v1' and v.get('candidate_admitted') is True and v.get('service_eligible') is True,'Positive reconstructed V10 admission required')
    need(v.get('physical_acceptance') is False and v.get('population_generalization') is False and v.get('no_reselection') is True,'Scientific scope differs')
    rb=dict(path=str(path),sha256=expected_sha);need(v.get('receipt_binding')==rb,'Early scientific receipt differs')
    raw=read_bound(path,expected_sha);expected=copy.deepcopy(raw);expected['receipt_binding']=rb;expected['artifact_sha256'][str(path)]=expected_sha
    need(v==expected,'Scientific CLI result differs from bound receipt')
    a=dict(v['artifact_sha256']);add_artifacts(a,contract['artifact_sha256']);cp=v['admitted_checkpoint_binding'];d=v['dataset_binding']
    metadata_path=Path(cp['checkpoint_path'])/'metadata.json';metadata=read_bound(metadata_path,cp['metadata_sha256']);c=metadata['config'];run=Path(c['training_run_path']);saved=read_bound(run/'configuration.json',a[str(run/'configuration.json')])
    need(cp['schema']=='wam_pi05_v10_recovery_checkpoint_binding_v1' and digest(c)==cp['config_sha256'] and c['training_schema']=='wam_pi05_v10_recovery_training_v1','V10 checkpoint/config differs')
    need(saved['config']==c and saved['normalization']==metadata['normalization'] and saved['data_hash']==cp['data_hash'] and c['dataset_binding']==d and c['source_sha256']==cp['source_sha256'],'Admitted model/data configuration differs')
    norm=metadata['normalization'];need(digest(norm)==cp['normalization_sha256']==NORM_SHA and hashlib.sha256(json_bytes(norm).rstrip(b'\n')).hexdigest()==d['normalization_object_sha256']==OBJECT_SHA,'Original fixed-B normalization differs')
    nb=d['normalization_file_binding'];need(nb['sha256']==NORM_SHA and read_binding(nb)==norm,'Exact saved normalization file required');mb=d['manifest_binding'];read_binding(mb)
    add_artifacts(a,{str(metadata_path):cp['metadata_sha256'],mb['path']:mb['sha256'],nb['path']:nb['sha256']})
    need(all(str(PRODUCER/k) in v['source_binding'] and v['source_binding'][str(PRODUCER/k)]==h for k,h in cp['source_sha256'].items()),'Original training producer source paths differ')
    need(v['source_binding'].get(contract['entry_binding']['path'])==contract['entry_binding']['sha256'],'Original external producer differs')
    result=dict(schema=ADMISSION_SCHEMA,candidate_admitted=True,service_eligible=True,admission_binding=rb,scientific_admission=v,admission_verifier_binding=contract['verifier_binding'],admission_bridge_contract=contract,checkpoint_binding=cp,selected_checkpoint=cp['checkpoint_path'],checkpoint_metadata_sha256=cp['metadata_sha256'],metadata_binding=dict(path=str(metadata_path),sha256=cp['metadata_sha256']),configuration_binding=dict(path=str(run/'configuration.json'),sha256=a[str(run/'configuration.json')]),manifest_path=mb['path'],manifest_sha256=mb['sha256'],normalization_file_binding=nb,normalization_object_sha256=OBJECT_SHA,dataset_binding=d,selection_lock_binding=v['lock_binding'],training_source_sha256=cp['source_sha256'],postlock_producer_source_binding=v['source_binding'],runtime_inventory_binding=contract['runtime_inventory_binding'],data_hash=cp['data_hash'],artifact_sha256=a,scientific_admission_reconstructed=True,physical_accepted=False,generalization_validated=False,**{k:c[k] for k in ('dynamics_recipe_sha256','dynamics_contract_sha256','execution_contract_sha256')})
    result['dataset_codec_binding']=dict(dataset_binding=d,normalization_sha256=NORM_SHA,normalization_object_sha256=OBJECT_SHA,action_encoding_binding=norm['action_encoding_binding'])
    verify_artifacts(result);return result


def verify_receipt(receipt,admission):
    need(receipt.get('schema')==RECEIPT_SCHEMA and type(receipt.get('process_exit_code')) is int and receipt['process_exit_code']==0 and receipt.get('process_group_drained') is True and receipt.get('cancelled') is False,'Actual CPU process did not complete')
    request=receipt['request'];contract=source_contract(request['verifier_binding']);need(contract==receipt['source_contract']==admission['admission_bridge_contract'],'Source contract changed')
    need(request['schema']==REQUEST_SCHEMA and request['source_contract_sha256']==digest(contract) and receipt['request_sha256']==digest(request) and request['admission_binding']==admission['admission_binding'],'CPU request differs')
    parent=receipt['parent_pid'];identity=receipt['process_identity'];need(type(parent) is int and parent>1 and all(type(identity.get(k)) is int for k in ('pid','pgid','sid')) and identity['pid']>1 and identity.get('pgid')==identity.get('sid')==identity['pid'] and type(identity.get('start_ticks')) is str and identity['start_ticks'].isascii() and identity['start_ticks'].isdecimal(),'Actual CPU session identity required')
    expected=[str(owned.OPENPI_PYTHON),'-B',str(Path(__file__).resolve()),'--worker','--parent-pid',str(parent)]
    need(receipt['guarded_command']==expected and receipt['verifier_exec_command']==worker_command(contract,request['admission_binding']['path'],request['admission_binding']['sha256']),'Guarded wrapper/exec commands differ')
    stdout=(json.dumps(receipt['worker_response'],allow_nan=False)+'\n').encode()
    need(receipt['worker_response']==admission['scientific_admission'] and receipt['stdout_sha256']==hashlib.sha256(stdout).hexdigest() and receipt['stdout_bytes']==len(stdout),'Actual scientific stdout differs')
    need(receipt.get('sdk_numpy_unchanged') is True and receipt.get('private_configuration_used') is False,'CPU/SDK isolation differs')
    need(receipt['process_errors']==receipt['cleanup_signals']==receipt['deferred_signals']==[] and receipt['child_environment']==owned.child_environment(PRODUCER),'CPU cleanup/environment differs')
    need(type(receipt['elapsed_s']) in (int,float) and math.isfinite(receipt['elapsed_s']) and 0<=receipt['elapsed_s']<=owned.TIMEOUT_S+212,'CPU time bound exceeded')
    for key,limit in (('stdout_bytes',owned.STDOUT_LIMIT),('stderr_bytes',owned.STDERR_LIMIT)):need(type(receipt[key]) is int and 0<=receipt[key]<=limit,'CPU output bound exceeded')
    need(receipt['admission_sha256']==digest(admission) and receipt['artifact_sha256_reverified']==verify_artifacts(admission),'CPU/physical artifact closure differs')
    return True


def selected_admission_with_receipt(path,expected_sha,*,verifier_binding):
    contract=source_contract(verifier_binding);regular(path);read_bound(path,expected_sha)
    request=dict(schema=REQUEST_SCHEMA,admission_binding=dict(path=str(path),sha256=expected_sha),verifier_binding=copy.deepcopy(verifier_binding),source_contract_sha256=digest(contract))
    command=[str(owned.OPENPI_PYTHON),'-B',str(Path(__file__).resolve()),'--worker','--parent-pid',str(os.getpid())]
    env=owned.child_environment(PRODUCER);run=owned.run_bounded(command,json_bytes(request),env=env,cwd=PRODUCER)
    failure=dict(schema='wam_pi05_v10_recovery_cpu_admission_failure_v1',request=request,guarded_command=command,verifier_exec_command=worker_command(contract,path,expected_sha),source_contract=contract,**{k:run.get(k) for k in ('exit_code','process_group_drained','errors','signals','cancelled','deferred_signals','elapsed_s','process_identity')},stdout_sha256=hashlib.sha256(run['stdout']).hexdigest(),stderr_sha256=hashlib.sha256(run['stderr']).hexdigest())
    try:
        need(not run['cancelled'] and run['exit_code']==0 and run['process_group_drained'] and not run['errors'] and not run['signals'] and not run['deferred_signals'],'CPU verification failed or cancelled')
        response=read_json_bytes(run['stdout']);admission=build_runtime_admission(response,path,expected_sha,contract)
        receipt=dict(schema=RECEIPT_SCHEMA,process_exit_code=run['exit_code'],process_group_drained=run['process_group_drained'],process_identity=run['process_identity'],cancelled=run['cancelled'],elapsed_s=run['elapsed_s'],guarded_command=command,verifier_exec_command=worker_command(contract,path,expected_sha),parent_pid=os.getpid(),request=request,request_sha256=digest(request),source_contract=contract,worker_response=response,child_environment=env,process_errors=run['errors'],cleanup_signals=run['signals'],deferred_signals=run['deferred_signals'],stdout_sha256=hashlib.sha256(run['stdout']).hexdigest(),stdout_bytes=len(run['stdout']),stderr_sha256=hashlib.sha256(run['stderr']).hexdigest(),stderr_bytes=len(run['stderr']),admission_sha256=digest(admission),artifact_sha256_reverified=verify_artifacts(admission),sdk_numpy_unchanged=True,private_configuration_used=False)
        verify_receipt(receipt,admission);return admission,receipt
    except (ValueError,KeyError,TypeError,OSError,UnicodeError) as error:
        failure['reason']=type(error).__name__+': '+str(error);kind=AdmissionInterrupted if run['cancelled'] else AdmissionFailure;raise kind(failure['reason'],failure) from error


def selected_admission(path,expected_sha,*,verifier_binding):return selected_admission_with_receipt(path,expected_sha,verifier_binding=verifier_binding)[0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--parent-pid',type=int,required=True);args=p.parse_args()
    need(args.worker,'Explicit internal CPU wrapper required');owned._parent_death_guard(args.parent_pid);owned.verify_worker_environment()
    raw=sys.stdin.buffer.read(65537);need(len(raw)<=65536,'CPU request too large');request=read_json_bytes(raw)
    need(set(request)=={'schema','admission_binding','verifier_binding','source_contract_sha256'} and request['schema']==REQUEST_SCHEMA,'Exact V10 CPU request required')
    contract=source_contract(request['verifier_binding']);need(digest(contract)==request['source_contract_sha256'],'CPU source contract changed');b=request['admission_binding'];read_binding(b)
    command=worker_command(contract,b['path'],b['sha256']);os.execve(command[0],command,owned.child_environment(PRODUCER))


if __name__=='__main__':main()
