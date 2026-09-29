"""Mac orchestration only: verified Isaac evidence archive, no local rendering.

Only completed, torn-down simulation trials are eligible. Full bytes stay on
L40S; optional cleanup removes only duplicate numbered sensor NPZs on linbox.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

SRC_ROOT='/home/robot/wam-sim-media/pi05-v5-results'
DST_ROOT='/home/william/wam-vla-v2/isaac-trials-v5'
DIRECT_HELPER='/home/robot/.local/share/wam-direct-archive-20260927-c8e712/transfer.py'
DIRECT_HELPER_SHA256='c38ce98fb7b97c8a0a557e3481f0cedee341415476f5c908620c5a235e7276e5'


def remote(host,script):
    return subprocess.check_output(['ssh','-o','BatchMode=yes',host,'python3 -'],input=script,text=True)


def inventory(host,path):
    return json.loads(remote(host,f'''
from pathlib import Path
import hashlib,json
p=Path({path!r})
print(json.dumps({{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file()}}))
'''))


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True)
    p.add_argument('--mirror',type=Path,required=True);p.add_argument('--remove-sensor-duplicates',action='store_true')
    p.add_argument('--transfer',choices=('direct','relay'),default='direct',
                   help='Verified temporary direct sender, or explicit legacy Mac relay')
    a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,100}',a.episode):raise ValueError('Invalid trial name')
    src=SRC_ROOT+'/'+a.episode;dst=DST_ROOT+'/'+a.episode
    status=json.loads(remote('linbox',f'''
from pathlib import Path
import json
p=Path({src!r})
result=json.loads((p/'collection_result.json').read_text());t=json.loads((p/'teardown.json').read_text())
print(json.dumps(dict(completed=True,closed=t.get('close_returned') is True,errors=t.get('errors'),passed=result['passed'])))
'''))
    if not status['closed'] or status['errors']:raise ValueError('Trial teardown not complete')
    before=inventory('linbox',src)
    remote('qwen-agent-l40s',f'''
from pathlib import Path
p=Path({dst!r})
if p.exists():raise ValueError('Archive destination already exists')
p.parent.mkdir(parents=True,exist_ok=True)
''')
    print('ARCHIVE_TRANSFER',a.episode,len(before),flush=True)
    transfer_receipt=None
    if a.transfer=='direct':
        transfer_receipt=json.loads(remote('linbox',f'''
from pathlib import Path
import hashlib,subprocess
p=Path({DIRECT_HELPER!r})
if hashlib.sha256(p.read_bytes()).hexdigest()!={DIRECT_HELPER_SHA256!r}:
 raise ValueError('Direct archive helper differs from verified source')
result=subprocess.run(['python3','-B',str(p),'send',{src!r},{a.episode!r}],check=True,capture_output=True,text=True)
print(result.stdout)
'''))
        if (transfer_receipt.get('status')!='verified_and_published'
                or transfer_receipt.get('written_file_readback_verified') is not True
                or transfer_receipt.get('destination')!=dst
                or transfer_receipt.get('source')!=src
                or transfer_receipt.get('sender_script_sha256')!=DIRECT_HELPER_SHA256
                or transfer_receipt.get('receiver_script_sha256')!=DIRECT_HELPER_SHA256):
            raise RuntimeError('Direct archive receipt mismatch; source preserved')
    else:
        producer=subprocess.Popen(['ssh','linbox','tar -C '+shlex.quote(SRC_ROOT)+' -czf - -- '+shlex.quote(a.episode)],stdout=subprocess.PIPE)
        consumer=subprocess.Popen(['ssh','qwen-agent-l40s','tar -C '+shlex.quote(DST_ROOT)+' -xzf -'],stdin=producer.stdout)
        producer.stdout.close()
        consumer_code=consumer.wait();producer_code=producer.wait()
        if consumer_code or producer_code:raise RuntimeError('Archive transfer incomplete; source preserved')
    archived=inventory('qwen-agent-l40s',dst);after=inventory('linbox',src)
    if before!=after or before!=archived:raise ValueError('Archive/source byte mismatch; source preserved')
    receipt=dict(source='linbox:'+src,archive='qwen-agent-l40s:'+dst,files=before,verified_identical=True,
        transfer_method=a.transfer,direct_transfer_receipt=transfer_receipt,
        acceptance_passed=status['passed'],scope='Evidence archive only; no new model acceptance')
    a.mirror.mkdir(parents=True,exist_ok=True);receipt_path=a.mirror/'archive_receipt.json'
    receipt_path.write_text(json.dumps(receipt,sort_keys=True)+'\n')
    subprocess.run(['scp',str(receipt_path),'linbox:'+src+'/archive_receipt.json'],check=True)
    subprocess.run(['scp',str(receipt_path),'qwen-agent-l40s:'+dst+'/archive_receipt.json'],check=True)
    subprocess.run(['rsync','-az','--exclude=sensors/*.npz','--exclude=training_scene.usdc',
                    'linbox:'+src+'/',str(a.mirror)+'/'],check=True)
    if a.remove_sensor_duplicates:
        print(remote('linbox',f'''
from pathlib import Path
import re,json,hashlib
p=Path({src!r});r=json.loads((p/'archive_receipt.json').read_text())
files=[(p/n,h) for n,h in r['files'].items() if re.fullmatch(r'sensors/[0-9]+\\.npz',n)]
for f,h in files:
 if hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed before cleanup')
size=sum(f.stat().st_size for f,h in files)
for f,h in files:f.unlink()
record=dict(removed_verified_duplicates=len(files),bytes_recovered=size,archive=r['archive'])
(p/'duplicate_cleanup.json').write_text(json.dumps(record)+'\\n');print(json.dumps(record))
''').strip())
        if re.fullmatch('[RL][1-5]_expert_recipe_v1',a.episode):
            identity=a.episode.replace('_expert_','_')
            qualified=json.loads(remote('linbox',f'''
from pathlib import Path
import json
p=Path('/home/robot/wam-sim-media/pi05-v5-qualified')/{identity!r}/({identity!r}+'.qualification.json')
print(json.dumps(p.is_file() and json.loads(p.read_text()).get('qualified') is True))
'''))
            if qualified:
                subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('cleanup_scene_duplicates.py')),
                                '--episode',a.episode],check=True)
    print('ARCHIVE_VERIFIED',a.episode,flush=True)


if __name__=='__main__':main()
