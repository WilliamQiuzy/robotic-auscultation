"""Synthetic qualified-view schedule tests; no actual acquisitions or model."""
import copy,hashlib,importlib,json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

SPEC=Path(__file__).resolve().parents[3]/'results/diagnostics/pi05_recovery_data_training_design_v1/machine_spec_v2.json'

def api():
    try:return importlib.import_module('sim.auscultation_pi05_v10.schedule')
    except ModuleNotFoundError as e:
        if e.name not in ('sim.auscultation_pi05_v10','sim.auscultation_pi05_v10.schedule'):raise
        return SimpleNamespace(build_schedule=lambda *a,**k:pytest.fail('V10 schedule not implemented'))

@pytest.fixture
def sample(monkeypatch):
    S=api();design=json.loads(SPEC.read_text());ids=sum([design['data'][k] for k in ['train_original8','train_existing5','train_new3']],[])
    eps=[];indices=[]
    for i,eid in enumerate(ids):
        phases=np.repeat(['FREE','APPROACH','CONTACT','HOLD','RETRACT'],7)
        eps.append(dict(episode_id=eid,split='train',phases=phases))
        indices.extend((i,start) for start in range(35))
    d=SimpleNamespace(episodes=eps,indices={'train':indices,'validation':[],'test':[]},loaded_scopes=('train',),qualification_scope_binding={'synthetic':True})
    if hasattr(S,'D'):monkeypatch.setattr(S.D,'dataset_binding',lambda ds:{'schema':'synthetic_only','data_hash':'a'*64})
    binding={'path':str(SPEC),'sha256':hashlib.sha256(SPEC.read_bytes()).hexdigest()}
    return S,d,binding

def test_exact_integer_budget_and_cell_balance(sample):
    S,d,b=sample;s=S.build_schedule(d,design_binding=b)
    assert s['schema']=='wam_pi05_v10_recovery_schedule_v1' and s['total_draws']==16000 and len(s['draws'])==16000
    assert s['candidate_steps']==[500,1000,1500,2000,2500,3000,3500,4000]
    for block in range(8):assert Counter(r['group'] for r in s['draws'][2000*block:2000*(block+1)])=={'original8':1100,'existing5':400,'new3_precontact':500}
    counts=Counter((r['group'],r['phase'],r['episode_id']) for r in s['draws'])
    assert {v for (g,_,_),v in counts.items() if g=='original8'}=={220}
    assert {v for (g,_,_),v in counts.items() if g=='existing5'}=={128}
    assert {v for (g,_,_),v in counts.items() if g=='new3_precontact'}=={666,667}
    assert all(r['phase'] in ('FREE','APPROACH') for r in s['draws'] if r['group']=='new3_precontact')
    assert S.verify_schedule(s,d,design_binding=b)==s
    for step in [0,499,500,3999]:np.testing.assert_array_equal(S.batch_indices(s,step),[r['train_index'] for r in s['draws'][step*4:step*4+4]])

def test_decks_have_no_repeated_start_before_all_cell_starts(sample):
    S,d,b=sample;s=S.build_schedule(d,design_binding=b);cells={}
    for r in s['draws']:cells.setdefault((r['group'],r['phase'],r['episode_id'],r['deck']),[]).append(r)
    assert all(len({x['start'] for x in rows})==len(rows) for rows in cells.values())
    assert all(sorted(x['deck_position'] for x in rows)==list(range(len(rows))) for rows in cells.values())
    assert s==S.build_schedule(d,design_binding=b)

def test_real_empty_phase_not_fabricated(sample):
    S,d,b=sample;d.indices['train']=[(ei,st) for ei,st in d.indices['train'] if not(ei==15 and st<7)]
    s=S.build_schedule(d,design_binding=b)
    assert len(s['draws'])==16000 and not any(r['episode_index']==15 and r['phase']=='FREE' for r in s['draws'])

@pytest.mark.parametrize('case',['missing_episode','duplicate_origin','heldout_origin','invalid_bool','unloaded','wrong_design'])
def test_bad_inventory_or_binding_rejected(sample,case):
    S,d,b=sample
    if case=='missing_episode':d.indices['train']=[r for r in d.indices['train'] if r[0]!=15]
    elif case=='duplicate_origin':d.indices['train'].append(d.indices['train'][0])
    elif case=='heldout_origin':d.episodes[0]['split']='validation'
    elif case=='invalid_bool':d.indices['train'][0]=(False,0)
    elif case=='unloaded':d.loaded_scopes=()
    else:b['sha256']='f'*64
    with pytest.raises(ValueError):S.build_schedule(d,design_binding=b)

@pytest.mark.parametrize('field,value',[('train_index',1),('start',999),('group','test')])
def test_modified_schedule_not_accepted(sample,field,value):
    S,d,b=sample;s=S.build_schedule(d,design_binding=b);s['draws'][0][field]=value
    with pytest.raises(ValueError):S.verify_schedule(s,d,design_binding=b)

def test_invalid_clock_and_boolean_rejected(sample):
    S,d,b=sample;s=S.build_schedule(d,design_binding=b)
    for step in [-1,4000,True,0.0]:
        with pytest.raises(ValueError):S.batch_indices(s,step)
