"""Verify exact standalone qualification copy to the V6 assembly host."""
from pathlib import Path
import argparse,hashlib,json,re,shlex,subprocess,os
from .operational_identity import qualification_id
ROOT=Path('/Users/williamqiu/Desktop/Harvard/WAM/results/pi05_v6_pilot')

def remote(code):
 return subprocess.check_output(['ssh','qwen-agent-l40s','python3 -B -'],input=code,text=True)

def inventory(p):
 return {str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--episode',required=True);a=p.parse_args()
 expected_qid=qualification_id(a.episode)
 folder=ROOT/a.episode;src=folder/'qualified'
 archived=json.loads((folder/'archive_receipt.json').read_text())
 if archived.get('verified_identical') is not True:raise ValueError('source archive not verified')
 reports=list(src.glob('*.qualification.json'))
 if len(reports)!=1:raise ValueError('exactly one qualified report required')
 report=json.loads(reports[0].read_text());qid=report['episode_id']
 if qid!=expected_qid:raise ValueError('qualification ID mismatch')
 if report.get('qualified') is not True:raise ValueError('only passing qualification copied for V6 assembly')
 before=inventory(src);dst='/home/william/wam-vla-v2/qualified-v6/'+qid
 created=json.loads(remote(f'''from pathlib import Path
import json
p=Path({dst!r})
exists=p.exists()
p.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps(dict(exists=exists)))
'''))
 if not created['exists']:
  subprocess.run(['rsync','-az',str(src)+'/', 'qwen-agent-l40s:'+dst+'/'],check=True)
 actual=json.loads(remote(f'''from pathlib import Path
import json,hashlib
p=Path({dst!r})
print(json.dumps({{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file()}}))
'''))
 if before!=inventory(src) or before!=actual:raise ValueError('qualification copy mismatch')
 receipt={'schema':'wam_qualification_copy_receipt_v1','episode_id':qid,'source':str(src),
          'destination':'qwen-agent-l40s:'+dst,'files_sha256':before,'verified_identical':True,
          'source_episode_archive':archived['archive'],'qualification_sha256':before[reports[0].name],
          'arrays_sha256':report['arrays_sha256'],'runtime_binding_sha256':before['runtime_recipe_binding.json']}
 path=folder/'qualification_copy_receipt.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
 subprocess.run(['scp',str(path),'qwen-agent-l40s:/home/william/wam-vla-v2/qualified-v6/'+qid+'.copy_receipt.json'],check=True)
 event={'event':'qualification_copy_verified','episode_id':qid,'destination':dst,'files':len(before),'arrays_sha256':report['arrays_sha256'],
        'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 fd=os.open(ROOT/'batch_events.jsonl',os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
 try:os.write(fd,(json.dumps(event)+'\n').encode())
 finally:os.close(fd)
 print(json.dumps(event),flush=True)
if __name__=='__main__':main()
