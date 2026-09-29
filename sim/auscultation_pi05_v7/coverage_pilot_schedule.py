"""Pure, immutable train-window schedule for the original-eight coverage pilot.

No model, dataset qualification, normalization fit, or simulator is performed.
Callers must supply the unchanged loader's complete qualified train indices.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION='wam_pi05_v7_original8_coverage_schedule_v1'
PHASES=('FREE','APPROACH','CONTACT','HOLD','RETRACT')
SEED=61
BATCH_SIZE=4
COMPLETE_DECK_MINIMUM=2
BOUNDARY_SLOTS=112
COLUMNS=('train_index','episode_slot','start','phase_slot','kind','deck','deck_position')
SOURCE_NAMES=('coverage_pilot_schedule.py','coverage_pilot_train.py','coverage_pilot_actor.py',
              'coverage_pilot_policy_service.py','coverage_pilot_policy_client.py',
              'coverage_pilot_evaluation.py','coverage_pilot_selection.py')

def json_bytes(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(json_bytes(value)).hexdigest()
def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def source_paths():return {'v7/'+n:Path(__file__).with_name(n) for n in SOURCE_NAMES}
def source_hashes():return {key:file_sha(path) for key,path in source_paths().items()}
def _integer(value):return isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_))
def _records(records,ids):
 if (not isinstance(ids,(list,tuple)) or len(ids)!=8 or len(set(ids))!=8
     or any(not isinstance(e,str) or not e for e in ids)):
  raise ValueError('Exactly eight explicit unique training identities required')
 if not isinstance(records,(list,tuple)) or not records:raise ValueError('Complete train index inventory required')
 out=[];origins=set();indices=set();episode_indices={}
 for row in records:
  if not isinstance(row,dict) or set(row)!={'train_index','episode_id','episode_index','start','phase','split'}:
   raise ValueError('Exact train index record schema required')
  if row['split']!='train' or row['episode_id'] not in ids or row['phase'] not in PHASES:
   raise ValueError('Only declared train identities and known phases allowed')
  for key in ('train_index','episode_index','start'):
   if not _integer(row[key]) or row[key]<0:raise ValueError('Nonnegative integer index required')
  if row['train_index'] in indices or (row['episode_id'],row['start']) in origins:
   raise ValueError('Duplicate qualified index/origin')
  indices.add(row['train_index']);origins.add((row['episode_id'],row['start']))
  previous=episode_indices.setdefault(row['episode_id'],int(row['episode_index']))
  if previous!=row['episode_index']:raise ValueError('Episode index changed')
  out.append({k:int(v) if k in ('train_index','episode_index','start') else v for k,v in row.items()})
 if indices!=set(range(len(out))) or set(episode_indices)!=set(ids) or len(set(episode_indices.values()))!=8:
  raise ValueError('Complete contiguous train inventory and unique episodes required')
 out.sort(key=lambda x:x['train_index'])
 for e in ids:
  for p in PHASES:
   if not any(r['episode_id']==e and r['phase']==p for r in out):raise ValueError('Empty episode/phase bucket')
 return out

def build_schedule(records,expected_episode_ids):
 records=_records(records,expected_episode_ids);ids=list(expected_episode_ids)
 buckets={(ei,pi):sorted((r for r in records if r['episode_id']==e and r['phase']==p),key=lambda r:r['start'])
          for ei,e in enumerate(ids) for pi,p in enumerate(PHASES)}
 maximum=max(map(len,buckets.values()));base=COMPLETE_DECK_MINIMUM*maximum;quota=base+BOUNDARY_SLOTS
 streams={};details=[]
 for (ei,pi),bucket in buckets.items():
  rng=np.random.default_rng(np.random.SeedSequence([SEED,1,ei,pi]));regular=[];deck=0
  while len(regular)<base:
   for position,i in enumerate(rng.permutation(len(bucket))):
    r=bucket[int(i)];regular.append([r['train_index'],ei,r['start'],pi,0,deck,position])
    if len(regular)==base:break
   deck+=1
  boundary=[]
  for r in bucket[:3]+bucket[-3:]:
   if r['train_index'] not in [v['train_index'] for v in boundary]:boundary.append(r)
  bonus=[r for r in boundary for _ in range(32 if r is bucket[0] else 16)]
  for i in range(BOUNDARY_SLOTS-len(bonus)):bonus.append(boundary[i%len(boundary)])
  order=np.random.default_rng(np.random.SeedSequence([SEED,2,ei,pi])).permutation(BOUNDARY_SLOTS)
  bonus=[[bonus[int(i)]['train_index'],ei,bonus[int(i)]['start'],pi,1,-1,j] for j,i in enumerate(order)]
  stream=[];oi=bi=0
  for t in range(quota):
   is_bonus=((t+1)*BOUNDARY_SLOTS)//quota>(t*BOUNDARY_SLOTS)//quota
   if is_bonus:stream.append(bonus[bi]);bi+=1
   else:stream.append(regular[oi]);oi+=1
  assert oi==base and bi==BOUNDARY_SLOTS
  streams[(ei,pi)]=stream
  details.append(dict(episode_id=ids[ei],phase=PHASES[pi],valid_windows=len(bucket),ordinary_draws=base,boundary_draws=BOUNDARY_SLOTS,
    boundary_train_indices=[r['train_index'] for r in boundary],first_train_index=bucket[0]['train_index']))
 rows=[];order_rng=np.random.default_rng(np.random.SeedSequence([SEED,3]))
 for offset in range(0,quota,4):
  for ei in order_rng.permutation(8):
   for t in range(offset,min(offset+4,quota)):
    for pi in range(5):rows.append(streams[(int(ei),pi)][t])
 assert len(rows)==40*quota and len(rows)%BATCH_SIZE==0
 steps=len(rows)//BATCH_SIZE
 return dict(schema=VERSION,seed=SEED,batch_size=BATCH_SIZE,columns=list(COLUMNS),episode_ids=ids,phases=list(PHASES),
  records_sha256=digest(records),valid_window_count=len(records),maximum_bucket_windows=maximum,bucket_draws=quota,
  ordinary_draws=40*base,boundary_draws=40*BOUNDARY_SLOTS,total_steps=steps,total_draws=len(rows),
  additional_window_equivalents=len(rows)/len(records),ordinary_window_equivalents=40*base/len(records),
  candidate_steps=[0]+[(steps*j+11)//12 for j in range(1,13)],numpy_version=np.__version__,
  policy='two complete ordinary decks minimum; equal 40 bucket quotas; independent 112 boundary slots; shuffled eight-episode sweeps with 20-sample balanced-phase blocks',
  rows=rows,buckets=details)

def verify_schedule(schedule,records,expected_episode_ids):
 expected=build_schedule(records,expected_episode_ids)
 if json_bytes(schedule)!=json_bytes(expected):raise ValueError('Frozen schedule differs from complete deterministic train inventory')
 return schedule

def batch_indices(schedule,local_step):
 if (schedule.get('schema')!=VERSION or not _integer(local_step)
     or not 0<=local_step<schedule['total_steps']):raise ValueError('Invalid schedule/local step')
 rows=schedule['rows'][4*int(local_step):4*(int(local_step)+1)]
 if len(rows)!=4:raise ValueError('Incomplete schedule batch')
 return np.asarray([row[0] for row in rows],dtype=np.int64)

def save_schedule(path,schedule):
 path=Path(path)
 if schedule.get('schema')!=VERSION:raise ValueError('Invalid schedule schema')
 with path.open('xb') as stream:stream.write(json_bytes(schedule))
 return dict(schema=VERSION,path=str(path.resolve()),sha256=file_sha(path),records_sha256=schedule['records_sha256'],
             total_steps=schedule['total_steps'],total_draws=schedule['total_draws'])

def load_schedule(path,binding):
 path=Path(path)
 if (binding.get('schema')!=VERSION or str(path.resolve())!=binding.get('path') or file_sha(path)!=binding.get('sha256')):
  raise ValueError('Schedule binding/file changed')
 schedule=json.loads(path.read_text())
 for key in ('schema','records_sha256','total_steps','total_draws'):
  if schedule.get(key)!=binding.get(key):raise ValueError('Schedule metadata changed')
 return schedule
