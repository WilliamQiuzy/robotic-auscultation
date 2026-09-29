"""Pure schedule invariants; fixtures are synthetic and never qualifications."""
import importlib.util
from pathlib import Path
from collections import Counter,defaultdict
import copy
import numpy as np
import pytest

PATH=Path(__file__).parents[1]/'coverage_pilot_schedule.py'
def module():
 assert PATH.is_file(),'New schedule implementation does not exist yet'
 spec=importlib.util.spec_from_file_location('pilot_schedule_test',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def inputs():
 ids=['episode'+str(i) for i in range(8)];records=[]
 for e,name in enumerate(ids):
  for p,phase in enumerate(('FREE','APPROACH','CONTACT','HOLD','RETRACT')):
   for s in range(2+(e+p)%7):
    records.append(dict(train_index=len(records),episode_id=name,episode_index=e,start=100*p+2*s,phase=phase,split='train'))
 return records,ids

def test_coverage_decks_and_bonus_are_separate():
 m=module();records,ids=inputs();s=m.build_schedule(records,ids)
 assert s['total_steps']==40*(2*8+112)//4
 rows=np.asarray(s['rows']);assert rows.shape==(s['total_steps']*4,7)
 ordinary=rows[rows[:,4]==0];bonus=rows[rows[:,4]==1]
 assert len(bonus)==40*112
 counts=Counter(ordinary[:,0]);assert min(counts.values())>=2
 assert len(counts)==len(records)
 for ei in range(8):
  for pi in range(5):
   bucket=[r for r in records if r['episode_id']==ids[ei] and r['phase']==m.PHASES[pi]]
   own=ordinary[(ordinary[:,1]==ei)&(ordinary[:,3]==pi)]
   assert len(own)==2*8
   for deck in set(own[:,5]):
    part=own[own[:,5]==deck]
    assert len(set(part[:,0]))==len(part)
    assert list(part[:,6])==list(range(len(part)))
   b=bonus[(bonus[:,1]==ei)&(bonus[:,3]==pi)];cnt=Counter(b[:,0]);boundary=bucket[:3]+bucket[-3:]
   assert set(cnt)=={r['train_index'] for r in boundary}
   assert cnt[bucket[0]['train_index']]>=32
   assert all(cnt[r['train_index']]>=16 for r in boundary)

def test_every_bucket_equal_and_batch_cursor_has_exact_suffix():
 m=module();records,ids=inputs();s=m.build_schedule(records,ids)
 counts=Counter((r[1],r[3]) for r in s['rows']);assert len(set(counts.values()))==1
 all_indices=np.asarray(s['rows'])[:,0].reshape(-1,4)
 for step in (0,1,7,s['total_steps']-1,s['total_steps']):
  tail=[m.batch_indices(s,t) for t in range(step,s['total_steps'])]
  assert np.array_equal(np.asarray(tail,dtype=np.int64).reshape(-1,4),all_indices[step:])
 with pytest.raises(ValueError):m.batch_indices(s,-1)
 with pytest.raises(ValueError):m.batch_indices(s,s['total_steps'])
 with pytest.raises(ValueError):m.batch_indices(s,True)

def test_deterministic_and_input_order_canonical():
 m=module();records,ids=inputs();a=m.build_schedule(records,ids);b=m.build_schedule(list(reversed(records)),ids)
 assert m.json_bytes(a)==m.json_bytes(b)
 assert m.digest(a)==m.digest(b)
 assert len(a['candidate_steps'])==13 and a['candidate_steps'][0]==0 and a['candidate_steps'][-1]==a['total_steps']

@pytest.mark.parametrize('mode',['test','empty','duplicate_index','duplicate_origin','unknown_phase','wrong_ids','boolean_index'])
def test_ineligible_input_fails(mode):
 m=module();r,ids=inputs();r=copy.deepcopy(r)
 if mode=='test':r[0]['split']='test'
 elif mode=='empty':r=[x for x in r if not(x['episode_id']==ids[0] and x['phase']=='FREE')]
 elif mode=='duplicate_index':r[1]['train_index']=0
 elif mode=='duplicate_origin':r[1]['start']=r[0]['start']
 elif mode=='unknown_phase':r[0]['phase']='DONE'
 elif mode=='wrong_ids':ids[0]='other'
 elif mode=='boolean_index':r[0]['train_index']=True
 with pytest.raises(ValueError):m.build_schedule(r,ids)

def test_freeze_load_and_rehash_tamper(tmp_path):
 m=module();r,ids=inputs();s=m.build_schedule(r,ids);path=tmp_path/'schedule.json'
 binding=m.save_schedule(path,s);assert m.load_schedule(path,binding)==s
 with pytest.raises(FileExistsError):m.save_schedule(path,s)
 x=copy.deepcopy(s);x['rows'][0][0]+=1;path.write_bytes(m.json_bytes(x))
 with pytest.raises(ValueError):m.load_schedule(path,binding)

def test_rehashed_structural_tamper_rejected():
 m=module();r,ids=inputs();s=m.build_schedule(r,ids)
 s['rows'][0][0]=s['rows'][1][0]
 with pytest.raises(ValueError):m.verify_schedule(s,r,ids)

def test_episode_blocks_bound_image_reload_frequency():
 m=module();r,ids=inputs();s=m.build_schedule(r,ids);a=np.asarray(s['rows']);n=s['bucket_draws']
 cursor=0
 for offset in range(0,n,4):
  size=min(4,n-offset)*5
  seen=[]
  for e in range(8):
   block=a[cursor:cursor+size];cursor+=size
   assert len(set(block[:,1]))==1;seen.append(block[0,1])
   assert Counter(block[:,3])=={p:size//5 for p in range(5)}
  assert set(seen)==set(range(8))
 assert cursor==len(a)
