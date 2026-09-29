"""Remove only duplicate numbered raw files after exact dual archive/qualification copies."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
from .operational_identity import qualification_id
ROOT=Path('/Users/williamqiu/Desktop/Harvard/WAM/results/pi05_v6_pilot')
def remote(host,code):
    return subprocess.check_output(['ssh',host,'python3 -B -'],input=code,text=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--episode',required=True)
    args=parser.parse_args();episode=args.episode;qid=qualification_id(episode)
    mirror=ROOT/episode;archive_path=mirror/'archive_receipt.json';archive=json.loads(archive_path.read_text())
    copied=json.loads((mirror/'qualification_copy_receipt.json').read_text())
    if not archive.get('verified_identical') or not copied.get('verified_identical') or copied['source_episode_archive']!=archive['archive']:
        raise ValueError('Both exact archives required before cleanup')
    dst='/home/william/wam-vla-v2/isaac-trials-v5/'+episode
    expected_receipt=hashlib.sha256(archive_path.read_bytes()).hexdigest()
    confirmation=json.loads(remote('qwen-agent-l40s',f'''
from pathlib import Path
import hashlib,json,re
p=Path({dst!r});receipt=p/'archive_receipt.json'
if hashlib.sha256(receipt.read_bytes()).hexdigest()!={expected_receipt!r}:raise ValueError('Archive receipt changed')
r=json.loads(receipt.read_text());selected={{n:h for n,h in r['files'].items() if re.fullmatch(r'sensors/[0-9]+\\.npz',n)}}
for n,h in selected.items():
 if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h:raise ValueError('Archive raw changed')
print(json.dumps(dict(files=len(selected),archive_raw_reread_verified=True)))
'''))
    if not confirmation['archive_raw_reread_verified']:raise ValueError('Archive verification absent')
    result=remote('linbox',f'''
from pathlib import Path
import hashlib,json,re,shutil
p=Path('/home/robot/wam-sim-media/pi05-v6-results')/{episode!r}
w=Path('/home/robot/wam-sim-media/pi05-v6-audits')/{(qid+'_working')!r}
qual=Path('/home/robot/wam-sim-media/pi05-v6-qualified')/{qid!r}/{(qid+'.qualification.json')!r}
if hashlib.sha256(qual.read_bytes()).hexdigest()!={copied['qualification_sha256']!r}:raise ValueError('Qualified report changed')
q=json.loads(qual.read_text())
if q.get('qualified') is not True or q['checks'].get('observed_contact_supervision_verified') is not True:raise ValueError('Verified qualification required')
r=json.loads((p/'archive_receipt.json').read_text())
selected={{n:h for n,h in r['files'].items() if re.fullmatch(r'sensors/[0-9]+\\.npz',n)}}
if len(selected)!={confirmation['files']!r} or len(selected)!=q['evidence']['raw_frames_verified']:raise ValueError('Raw count mismatch')
unique={{}};paths=[]
for root in (p,w):
 for name,h in selected.items():
  f=root/name
  if f.is_symlink() or not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise ValueError('Source/working raw differs from verified archive')
  stat=f.stat();unique[(stat.st_dev,stat.st_ino)]=stat.st_size;paths.append(f)
before=shutil.disk_usage(p).free
for f in paths:f.unlink()
after=shutil.disk_usage(p).free
record=dict(schema='wam_verified_raw_duplicate_cleanup_v1',source=str(p),working_copy=str(w),archive=r['archive'],
 numbered_frames=len(selected),unlinked_paths=len(paths),unique_raw_inodes=len(unique),unique_raw_file_bytes=sum(unique.values()),
 free_bytes_before=before,free_bytes_after=after,archive_raw_reread_verified=True,
 qualification_sha256={copied['qualification_sha256']!r},verified_raw_sha256=selected,
 scope='Only numbered raw duplicates removed after independent replay, qualification and exact full archive/qualified copies; all raw remains archived')
for root in (p,w):(root/'duplicate_cleanup.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
''')
    (mirror/'duplicate_cleanup.json').write_text(json.dumps(json.loads(result),indent=2)+'\n')
    subprocess.run(['scp',str(mirror/'duplicate_cleanup.json'),'qwen-agent-l40s:'+dst+'/duplicate_cleanup.json'],check=True)
    record=json.loads(result);record.pop('verified_raw_sha256')
    print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
