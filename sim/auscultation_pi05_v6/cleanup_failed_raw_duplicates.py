"""Explicit R5 failed-trial housekeeping; no qualification or acceptance claim."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
import subprocess

EPISODE='R5_expert_observed_v1'
RECEIPT_SHA='69a5df2abf95ad6694292cf623ec9da6349758cfd20721e173900fb0ac97312e'
SOURCE='/home/robot/wam-sim-media/pi05-v6-results/'+EPISODE
DESTINATION='/home/william/wam-vla-v2/isaac-trials-v5/'+EPISODE
MIRROR=Path('/Users/williamqiu/Desktop/Harvard/WAM/results/pi05_v6_pilot')/EPISODE


def validate_archive(data,source,destination):
    if (data.get('verified_identical') is not True or data.get('acceptance_passed') is not False
            or data.get('source')!='linbox:'+source or data.get('archive')!='qwen-agent-l40s:'+destination):
        raise ValueError('Exact completed-failure archive required')
    selected={n:h for n,h in data['files'].items() if re.fullmatch(r'sensors/[0-9]+\.npz',n)}
    if not selected or any(not re.fullmatch('[a-f0-9]{64}',h) for h in selected.values()):
        raise ValueError('Exact numbered sensor hashes required')
    return selected


def validate_completed_failure(collection,teardown):
    if collection.get('passed') is not False:
        raise ValueError('Only this failed completed collection is authorized')
    if teardown.get('close_returned') is not True or teardown.get('errors')!=[]:
        raise ValueError('Completed managed teardown required')


def safe_file(root,name):
    if not re.fullmatch(r'sensors/[0-9]+\.npz',name):
        raise ValueError('Only numbered sensor NPZ paths may be removed')
    root=Path(root)
    if root.resolve()!=root or root.is_symlink():raise ValueError('Unsafe trial root')
    path=root/name
    if path.parent.is_symlink() or path.parent.resolve()!=root/'sensors':raise ValueError('Unsafe sensor directory')
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('Independent regular source file required')
    return path


def remote(host,script):
    return subprocess.check_output(['ssh',host,'python3 -B -'],input=script,text=True)


def main():
    path=MIRROR/'archive_receipt.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=RECEIPT_SHA:raise ValueError('Pinned archive receipt differs')
    receipt=json.loads(path.read_text());selected=validate_archive(receipt,SOURCE,DESTINATION)
    common='import hashlib,json,os,re,stat,shutil\nfrom pathlib import Path\n'+inspect.getsource(safe_file)+inspect.getsource(validate_completed_failure)
    checks=f'''
p=Path(ROOT);r=p/'archive_receipt.json'
if hashlib.sha256(r.read_bytes()).hexdigest()!={RECEIPT_SHA!r}:raise ValueError('Remote receipt differs')
receipt=json.loads(r.read_text())
for name in ('collection_result.json','teardown.json'):
 if hashlib.sha256((p/name).read_bytes()).hexdigest()!=receipt['files'][name]:raise ValueError('Terminal metadata differs')
collection=json.loads((p/'collection_result.json').read_text());teardown=json.loads((p/'teardown.json').read_text())
validate_completed_failure(collection,teardown)
selected={selected!r}
if collection['sensor_frames']!=len(selected):raise ValueError('Sensor count mismatch')
'''
    confirmation=json.loads(remote('qwen-agent-l40s',common+'\nROOT='+repr(DESTINATION)+checks+'''
for name,digest in selected.items():
 f=safe_file(p,name)
 if hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('Archived raw differs')
print(json.dumps(dict(verified=True,files=len(selected))))
'''))
    if confirmation!={'verified':True,'files':len(selected)}:raise ValueError('Archive reread missing')
    result=json.loads(remote('linbox',common+'\nROOT='+repr(SOURCE)+checks+f'''
destination=p/'failed_raw_duplicate_cleanup.json'
if destination.exists():raise ValueError('Cleanup receipt already exists; no automatic resume')
records=[]
for name,digest in selected.items():
 f=safe_file(p,name)
 if hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('Source raw differs')
 info=f.stat();records.append(dict(relative_path=name,sha256=digest,bytes=info.st_size))
before=shutil.disk_usage(p).free
record=dict(schema='wam_verified_failed_trial_raw_cleanup_v1',source=str(p),archive={('qwen-agent-l40s:'+DESTINATION)!r},
 archive_receipt_sha256={RECEIPT_SHA!r},archive_raw_reread_verified=True,qualification_claimed=False,
 failed_collection_preserved=True,teardown_completed=True,status='prepared',removed_paths=[],selected_files=records,
 free_bytes_before=before,scope='Only this completed failed R5 original numbered NPZ duplicates; complete exact raw archive retained; no qualification claim')
destination.write_text(json.dumps(record,indent=2)+'\\n')
for item in records:
 f=safe_file(p,item['relative_path']);before_stat=f.stat()
 if hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Source changed immediately before removal')
 after_stat=f.stat()
 if (before_stat.st_dev,before_stat.st_ino,before_stat.st_size,before_stat.st_mtime_ns,before_stat.st_ctime_ns)!=(after_stat.st_dev,after_stat.st_ino,after_stat.st_size,after_stat.st_mtime_ns,after_stat.st_ctime_ns):raise ValueError('Source identity changed')
 f.unlink();record['removed_paths'].append(item['relative_path'])
record.update(status='complete',removed_count=len(records),removed_bytes=sum(x['bytes'] for x in records),free_bytes_after=shutil.disk_usage(p).free)
destination.write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''))
    output=MIRROR/'failed_raw_duplicate_cleanup.json';output.write_text(json.dumps(result,indent=2)+'\n')
    subprocess.run(['scp',str(output),'qwen-agent-l40s:'+DESTINATION+'/'+output.name],check=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ('removed_paths','selected_files')},indent=2))


if __name__=='__main__':main()
