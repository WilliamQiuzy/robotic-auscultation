"""Delete only explicitly authorized, reverified archive duplicates after qualification."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess
from .operational_identity import qualification_id
BASE='/home/robot/wam-sim-media'
MAC=Path('/Users/williamqiu/Desktop/Harvard/WAM/results')

def remote(host,code):
 return json.loads(subprocess.check_output(['ssh',host,'python3 -B -'],input=code,text=True))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--episode',required=True);a=ap.parse_args()
 qid=qualification_id(a.episode);workings=[BASE+'/pi05-v6-audits/'+qid+'_working']
 mirror=MAC/'pi05_v6_pilot'/a.episode
 src=BASE+'/pi05-v6-results/'+a.episode;archive='/home/william/wam-vla-v2/isaac-trials-v5/'+a.episode
 copied=json.loads((mirror/'qualification_copy_receipt.json').read_text())
 qualified_path=mirror/'qualified'/(qid+'.qualification.json')
 qualified=json.loads(qualified_path.read_text())
 if (copied.get('verified_identical') is not True or copied['qualification_sha256']!=sha(qualified_path)
     or qualified.get('qualified') is not True or qualified['checks'].get('observed_contact_supervision_verified') is not True):
  raise ValueError('Exact passing V6 qualification copy required before evidence cleanup')
 receipt=json.loads((mirror/'archive_receipt.json').read_text())
 if receipt.get('verified_identical') is not True or receipt['archive']!='qwen-agent-l40s:'+archive:raise ValueError('verified archive required')
 choices=remote('qwen-agent-l40s',f'''from pathlib import Path
import json,hashlib,re
p=Path({archive!r});r=json.loads((p/'archive_receipt.json').read_text());result={{}}
for rel,digest in r['files'].items():
 f=p/rel
 if rel in ('expert_trace.jsonl','control_commands.json','supervisor_decisions.jsonl'):
  if not f.is_file() or f.is_symlink():raise ValueError('not regular archive file')
  actual=hashlib.sha256(f.read_bytes()).hexdigest()
  if actual!=digest:raise ValueError('archive differs from recorded manifest')
  result[rel]=dict(sha256=digest,bytes=f.stat().st_size)
print(json.dumps(result))
''')
 for rel,item in choices.items():
  if receipt['files'].get(rel)!=item['sha256']:raise ValueError('archive manifest disagreement')
  local=mirror/rel;local.parent.mkdir(parents=True,exist_ok=True)
  if not local.exists():subprocess.run(['scp','linbox:'+src+'/'+rel,str(local)],check=True)
  if not local.is_file() or local.is_symlink() or sha(local)!=item['sha256']:raise ValueError('Mac copy mismatch')
 # Repeat actual archive readback immediately before the local unlink transaction.
 verified=remote('qwen-agent-l40s',f'''from pathlib import Path
import hashlib,json
p=Path({archive!r});expected={choices!r}
for rel,item in expected.items():
 f=p/rel
 if hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('archive changed before cleanup')
print(json.dumps(expected))
''')
 record=remote('linbox',f'''from pathlib import Path
import hashlib,json,os,time
roots=[Path({src!r}),*[Path(p) for p in {workings!r}]]
expected={verified!r};to_remove=[];mismatches=[];missing=[]
if any((root/'evidence_duplicate_cleanup.json').exists() for root in roots):raise ValueError('cleanup receipt already exists')
for root in roots:
 for rel,item in expected.items():
  p=root/rel
  if not p.exists():missing.append(str(p));continue
  if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
   mismatches.append(str(p));continue
  st=p.stat();to_remove.append((p,st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,rel,item))
# Do not unlink any mismatch; each matching pathname is checked again. Shared
# inode ctime changes from earlier unlink, so compare inode/size/mtime per link.
removed=[]
for p,dev,ino,size,mtime,ctime,rel,item in to_remove:
 st=p.stat()
 if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(dev,ino,size,mtime):raise ValueError('input changed before unlink')
 p.unlink();removed.append(dict(path=str(p),relative_path=rel,sha256=item['sha256'],bytes=size))
record=dict(schema='wam_verified_evidence_duplicate_cleanup_v1',archive={'qwen-agent-l40s:'+archive!r},
 scope='only expert_trace.jsonl, control_commands.json and supervisor_decisions.jsonl; exact complete copies on L40S and Mac',
 archive_and_mac_files=expected,removed=removed,mismatches_preserved=mismatches,already_missing=missing,
 source_metadata_and_qualification_preserved=True,full_trace_and_decisions_preserved_on_Mac_and_L40S=True)
for root in roots:
 out=root/'evidence_duplicate_cleanup.json'
 if out.exists():raise ValueError('cleanup receipt already exists')
 out.write_text(json.dumps(record,indent=2)+chr(10))
print(json.dumps(record))
''')
 out=mirror/'evidence_duplicate_cleanup.json';out.write_text(json.dumps(record,indent=2)+'\n')
 subprocess.run(['scp',str(out),'qwen-agent-l40s:'+archive+'/evidence_duplicate_cleanup.json'],check=True)
 print(json.dumps({'episode':a.episode,'selected_files':len(choices),'removed_paths':len(record['removed']),
                   'unique_bytes':sum(v['bytes'] for v in choices.values()),'mismatches_preserved':record['mismatches_preserved'],
                   'receipt_sha256':sha(out)}),flush=True)
 if record['mismatches_preserved'] or record['already_missing']:
  raise RuntimeError('Evidence duplicate mismatch or missing path preserved; stop orchestration for review')

if __name__=='__main__':main()
