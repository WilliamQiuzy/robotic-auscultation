"""Mac orchestration: remove only exact archived qualification NPZ duplicates."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess
from .operational_identity import qualification_id

BASE='/home/robot/wam-sim-media'
MIRROR=Path('/Users/williamqiu/Desktop/Harvard/WAM/results/pi05_v6_pilot')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def qualified_array_paths(episode,receipt):
    qid=qualification_id(episode)
    names=(qid+'.npz','parent_v2/'+qid+'.npz')
    if receipt.get('verified_identical') is not True or receipt.get('episode_id')!=qid:
        raise ValueError('Exact qualified copy required')
    selected={name:receipt['files_sha256'][name] for name in names}
    if any(value!=receipt['arrays_sha256'] for value in selected.values()):
        raise ValueError('Parent and child arrays must remain byte-identical')
    return qid,selected


def remote(host,code):
    return json.loads(subprocess.check_output(['ssh',host,'python3 -B -'],input=code,text=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--episode',required=True);args=parser.parse_args()
    folder=MIRROR/args.episode
    receipt=json.loads((folder/'qualification_copy_receipt.json').read_text())
    qid,selected=qualified_array_paths(args.episode,receipt)
    for name,digest in selected.items():
        path=folder/'qualified'/name
        if path.is_symlink() or not path.is_file() or sha(path)!=digest:raise ValueError('Mac array copy differs')
    destination='/home/william/wam-vla-v2/qualified-v6/'+qid
    remote('qwen-agent-l40s',f'''
from pathlib import Path
import json,hashlib
root=Path({destination!r});expected={selected!r}
if hashlib.sha256((root/{qid+'.qualification.json'!r}).read_bytes()).hexdigest()!={receipt['qualification_sha256']!r}:raise ValueError('L40S qualification report differs')
for name,digest in expected.items():
 p=root/name
 if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('L40S array copy differs')
print(json.dumps(dict(verified=True)))
''')
    record=remote('linbox',f'''
from pathlib import Path
import hashlib,json,shutil
root=Path({BASE+'/pi05-v6-qualified/'+qid!r});expected={selected!r}
receipt=root/'qualified_array_duplicate_cleanup.json'
if receipt.exists():raise ValueError('Cleanup receipt already exists')
report=root/{qid+'.qualification.json'!r}
if hashlib.sha256(report.read_bytes()).hexdigest()!={receipt['qualification_sha256']!r}:raise ValueError('Qualified report changed')
qualified=json.loads(report.read_text())
if qualified.get('qualified') is not True or qualified['checks'].get('observed_contact_supervision_verified') is not True:raise ValueError('Passing V6 qualification required')
files=[]
for name,digest in expected.items():
 p=root/name
 if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Local array differs from copies')
 st=p.stat();files.append((p,st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,digest))
before=shutil.disk_usage(root).free;removed=[]
for p,dev,ino,size,mtime,digest in files:
 st=p.stat()
 if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(dev,ino,size,mtime):raise ValueError('Array changed before unlink')
 p.unlink();removed.append(dict(relative_path=str(p.relative_to(root)),sha256=digest,bytes=size))
record=dict(schema='wam_verified_qualified_array_duplicate_cleanup_v1',source=str(root),
 exact_copies=[{str(folder/'qualified')!r},{'qwen-agent-l40s:'+destination!r}],removed=removed,
 qualified_report_preserved=True,free_bytes_before=before,free_bytes_after=shutil.disk_usage(root).free,
 scope='Only qualified NPZ duplicates removed; immutable exact arrays remain on Mac and L40S')
receipt.write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
''')
    path=folder/'qualified_array_duplicate_cleanup.json';path.write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run(['scp',str(path),'qwen-agent-l40s:'+destination+'/qualified_array_duplicate_cleanup.json'],check=True)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
