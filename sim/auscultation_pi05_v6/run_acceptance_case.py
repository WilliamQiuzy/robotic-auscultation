"""Mac orchestration of one real learned Isaac case and its independent audits.

The default prints arguments only. Execution uses existing authenticated SSH,
retains failed trials, snapshots server receipts, and archives complete bytes.
No inference token is printed or copied by this orchestration.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

PROJECT='/home/robot/wam-sim-media/hall-modality-v1/project'
SOURCE='/home/robot/wam-sim-media/skill_v4/project'
PYTHON='/home/robot/miniconda3/envs/env_isaacsim/bin/python'
MODULE='sim.auscultation_pi05_v6.'


def remote(host,code):
    return subprocess.check_output(['ssh','-o','BatchMode=yes',host,'python3 -B -'],input=code,text=True)


def linux_python(code):
    command=shlex.join(['env','PYTHONDONTWRITEBYTECODE=1',
        'PYTHONPATH='+':'.join((PROJECT,SOURCE,'/home/robot/wam-sim-media/pi05-v4-deps',
                             '/home/robot/Desktop/stethoscope/sim/auscultation_training/python_deps')),
        PYTHON,'-B','-'])
    return subprocess.check_output(['ssh','linbox',command],input=code,text=True)


def run(args):
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,100}',args.case):raise ValueError('Invalid case name')
    if not re.fullmatch('[a-f0-9]{64}',args.checkpoint_sha256):raise ValueError('Exact selected checkpoint required')
    if len(set(args.points))!=len(args.points) or (args.scope=='contact' and len(args.points)!=1):
        raise ValueError('Unique ordered points; contact is a single component only')
    source_root=Path(args.source_root)
    if not source_root.is_absolute():raise ValueError('Absolute remote output root required')
    trial=str(source_root/args.case);mirror=Path(args.mirror).resolve()
    if mirror.exists():raise ValueError('New Mac evidence directory required')
    spec=dict(case=args.case,points=args.points,scope=args.scope,remote_trial=trial,
        checkpoint_metadata_sha256=args.checkpoint_sha256,source_root=str(source_root),
        service_directory=args.service_directory,mirror=str(mirror))
    if not args.execute:return dict(prepared_only=True,**spec)
    mirror.mkdir(parents=True)
    receipt=dict(schema='wam_pi05_learned_case_lifecycle_v1',**spec,
        started_utc=datetime.now(timezone.utc).isoformat(),completed=False,policy_acceptance=False)
    def save():(mirror/'lifecycle.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    try:
        # Verify private configuration through the client; expose only public binding.
        binding=json.loads(linux_python(f'''
from sim.auscultation_pi05_v6.policy_client import PolicyClient
import json
c=PolicyClient({args.client_config!r})
if c.binding['checkpoint_metadata_sha256']!={args.checkpoint_sha256!r}:raise ValueError('Selected checkpoint mismatch')
print(json.dumps(c.binding))
'''))
        (mirror/'selected_policy_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
        preflight=json.loads(remote('linbox',f'''
from pathlib import Path
import json,shutil
p=Path({str(source_root)!r})
if not p.is_dir() or p.is_symlink():raise ValueError('Existing regular output root required')
if (p/{args.case!r}).exists():raise ValueError('No reuse or replacement of a previous trial')
free=shutil.disk_usage(p).free
minimum={25 if len(args.points)>1 else 3}*2**30
if free<minimum:raise ValueError('Insufficient episode storage reserve')
print(json.dumps(dict(free_bytes=free,minimum_bytes=minimum)))
'''))
        receipt['preflight']=preflight;save()
        command=shlex.join(['bash',PROJECT+'/sim/auscultation_pi05_v6/run_isaac.sh',
            '--output',trial,'--points',*args.points,'--task-scope',args.scope,
            '--client-config',args.client_config,'--dynamics-contract',args.dynamics_contract,
            '--sensors','--speed-scale','.7','--position-iterations','32','--armature','.01'])
        receipt['isaac_exit_code']=subprocess.call(['ssh','linbox',command+' > '+shlex.quote(trial+'.log')+' 2>&1']);save()
        status=json.loads(remote('linbox',f'''
from pathlib import Path
import json
p=Path({trial!r});c=json.loads((p/'collection_result.json').read_text());t=json.loads((p/'teardown.json').read_text())
if t.get('close_returned') is not True or t.get('errors')!=[]:raise ValueError('Isaac teardown incomplete; preserve live data')
learned=p/'learned_actions.jsonl'
ids=[r['inference']['request_id'] for r in map(json.loads,learned.read_text().splitlines()) if r.get('inference')] if learned.exists() else []
if len(set(ids))!=len(ids):raise ValueError('Duplicate learned request identities')
print(json.dumps(dict(physics_passed=c.get('passed') is True,request_ids=ids)))
'''))
        ids=status.pop('request_ids');receipt.update(status,recorded_requests=len(ids));save()
        # Read the server's completed receipts once and retain exactly this trial's IDs.
        records=json.loads(remote('qwen-agent-l40s',f'''
from pathlib import Path
import json
p=Path({args.service_directory!r})/'requests.jsonl';wanted=set({ids!r})
rows=[r for r in map(json.loads,p.read_text().splitlines()) if r['request_id'] in wanted] if wanted else []
if len(rows)!=len(wanted) or {{r['request_id'] for r in rows}}!=wanted:raise ValueError('Missing/duplicate server request evidence')
print(json.dumps(rows))
'''))
        requests=mirror/'server_requests.jsonl'
        with requests.open('x') as stream:stream.writelines(json.dumps(row)+'\n' for row in records)
        subprocess.run(['scp',str(requests),'linbox:'+trial+'/server_requests.jsonl'],check=True)
        # On partial failures, a failed audit is evidence to retain, never acceptance.
        audit_status=json.loads(linux_python(f'''
from pathlib import Path
import json
from sim.auscultation_pi05_v6.audit_learned_actions import audit_episode as actions
from sim.auscultation_pi05_v6.audit_observed_supervisor import audit_episode as contact
p=Path({trial!r});status={{}}
for name,function,kwargs in [
 ('learned_actions_audit.json',actions,dict(expected_checkpoint_metadata_sha256={args.checkpoint_sha256!r},server_requests_path=p/'server_requests.jsonl')),
 ('observed_supervisor_audit.json',contact,{{}})]:
 try:report=function(p,**kwargs)
 except Exception as error:report=dict(passed=False,errors=[type(error).__name__+': '+str(error)],scope='Audit could not complete; no acceptance')
 with (p/name).open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\\n')
 status[name]=dict(passed=report.get('passed') is True,errors=report.get('errors'))
print(json.dumps(status))
'''))
        receipt['audits']=audit_status;save()
        # Reuse the byte-verified transport with an explicitly scoped output root.
        from sim.auscultation_pi05_v5 import archive_trial
        previous=sys.argv
        try:
            archive_trial.SRC_ROOT=str(source_root)
            sys.argv=['archive_trial','--episode',args.case,'--mirror',str(mirror)]
            archive_trial.main()
        finally:sys.argv=previous
        receipt['completed']=True
        receipt['case_passed']=bool(receipt['isaac_exit_code']==0 and receipt['physics_passed']
            and ids and all(value['passed'] for value in audit_status.values()))
        receipt['scope_note']='One independently audited declared case; complete protocol requires all declared cases and sequence'
    except BaseException as error:
        receipt['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        receipt['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('case','checkpoint-sha256','client-config','dynamics-contract','service-directory','mirror'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--source-root',default='/home/robot/wam-sim-media/pi05-v6-results')
    parser.add_argument('--points',nargs='+',choices=[s+str(i) for i in range(1,6) for s in 'RL'],required=True)
    parser.add_argument('--scope',choices=['full','contact'],default='full')
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    result=run(args);print(json.dumps(result,indent=2))
    if args.execute and not result.get('case_passed'):raise SystemExit(1)
