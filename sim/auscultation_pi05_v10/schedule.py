"""Pure fixed 55/20/25 recovery sampling; only already qualified TRAIN indices."""
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from . import training_data as D

SCHEMA='wam_pi05_v10_recovery_schedule_v1'
TOTAL_STEPS=4000
BATCH_SIZE=4
CANDIDATE_STEPS=(500,1000,1500,2000,2500,3000,3500,4000)
GROUPS=('original8','existing5','new3_precontact')
QUOTAS=(1100,400,500)
PHASES=D.PHASES
file_sha=D.file_sha
digest=D.digest
json_bytes=D.json_bytes


def _integer(v):return isinstance(v,(int,np.integer)) and not isinstance(v,(bool,np.bool_))


def _design(binding):
    D.need(type(binding) is dict and set(binding)=={'path','sha256'} and binding['sha256']==D.DESIGN_SHA256,'Exact approved schedule design required')
    return D._read_pinned(binding['path'],binding['sha256'],{})


def _inventory(dataset,design):
    identity=D.dataset_binding(dataset)
    D.need('train' in dataset.loaded_scopes,'Schedule requires qualified TRAIN')
    ids=[design['data'][name] for name in ('train_original8','train_existing5','train_new3')]
    assignment={eid:g for g,own in zip(GROUPS,ids) for eid in own}
    D.need(len(assignment)==16 and len([e for e in dataset.episodes if e['split']=='train'])==16,'Exact16 qualified training episodes required')
    rows=[];seen=set();present=set();cells=defaultdict(list)
    for index,pair in enumerate(dataset.indices['train']):
        D.need(isinstance(pair,(tuple,list)) and len(pair)==2 and all(_integer(x) and x>=0 for x in pair),'Integer training origin required')
        ei,start=map(int,pair);D.need(ei<len(dataset.episodes),'Episode index outside inventory')
        ep=dataset.episodes[ei];eid=ep['episode_id']
        D.need(ep['split']=='train' and eid in assignment and start<len(ep['phases']),'Only fixed TRAIN sources allowed')
        D.need((ei,start) not in seen,'Duplicate qualified origin');seen.add((ei,start));present.add(eid)
        phase=str(ep['phases'][start]);D.need(phase in PHASES,'Unknown qualified phase')
        row=dict(train_index=index,episode_index=ei,episode_id=eid,start=start,phase=phase,split='train',group=assignment[eid]);rows.append(row)
        if row['group']!='new3_precontact' or phase in ('FREE','APPROACH'):cells[(row['group'],phase,eid)].append(row)
    D.need(present==set(assignment) and {k[2] for k in cells}==set(assignment),'Every planned episode needs actual permitted starts')
    for values in cells.values():values.sort(key=lambda r:r['start'])
    return identity,rows,cells


def _sources():return dict(D.core_sources(),**{'sim/auscultation_pi05_v10/schedule.py':file_sha(__file__)})


def build_schedule(dataset,*,design_binding):
    design=_design(design_binding);identity,inventory,cells=_inventory(dataset,design)
    pc=Counter();ec=Counter();consumed=Counter();decks={};draws=[];blocks=[]
    for block in range(8):
        block_rows=[]
        for group,quota in zip(GROUPS,QUOTAS):
            phases=[p for p in PHASES if any(g==group and phase==p for g,phase,eid in cells)]
            D.need(phases,'Empty training group')
            for _ in range(quota):
                phase=phases[pc[group]%len(phases)];pc[group]+=1
                ids=sorted(eid for g,p,eid in cells if g==group and p==phase)
                eid=ids[ec[group,phase]%len(ids)];ec[group,phase]+=1;key=(group,phase,eid)
                cycle,position=divmod(consumed[key],len(cells[key]));consumed[key]+=1
                if (key,cycle) not in decks:
                    decks[key,cycle]=sorted(cells[key],key=lambda r:(D.object_digest(dict(seed=61,namespace='cell',group=group,phase=phase,episode_id=eid,cycle=cycle,start=r['start'])),r['start']))
                row=dict(decks[key,cycle][position],deck=cycle,deck_position=position,block=block,logical_draw_ordinal=len(block_rows));block_rows.append(row)
        block_rows.sort(key=lambda r:(D.object_digest(dict(seed=61,namespace='block',block=block,logical_draw_ordinal=r['logical_draw_ordinal'])),r['logical_draw_ordinal']))
        for row in block_rows:draws.append(dict(row,draw_index=len(draws)))
        blocks.append(dict(block=block,draw_start=block*2000,draw_end_exclusive=(block+1)*2000,group_counts=dict(Counter(r['group'] for r in block_rows))))
    coverage=[]
    for key,values in sorted(cells.items()):
        own=[r for r in draws if (r['group'],r['phase'],r['episode_id'])==key];counts=Counter(r['start'] for r in own)
        coverage.append(dict(group=key[0],phase=key[1],episode_id=key[2],valid_starts=len(values),draws=len(own),unique_starts=len(counts),minimum_draws_per_valid_start=min(counts.get(r['start'],0) for r in values),maximum_draws_per_valid_start=max(counts.values()),complete_decks=len(own)//len(values)))
    result=dict(schema=SCHEMA,mode=D.MODE,design_binding=dict(design_binding),dataset_binding=identity,source_sha256=_sources(),seed=61,batch_size=4,total_steps=4000,total_draws=16000,candidate_steps=list(CANDIDATE_STEPS),valid_window_count=len(inventory),train_inventory_sha256=digest(inventory),inventory=inventory,qualification_scopes_used=['train'],blocks=blocks,coverage=coverage,draws=draws)
    result['schedule_sha256']=digest(result)
    D.need(D.dataset_binding(dataset)==identity,'Dataset changed during scheduling')
    _design(design_binding);return result


def verify_schedule(schedule,dataset,*,design_binding):
    expected=build_schedule(dataset,design_binding=design_binding)
    D.need(type(schedule) is dict and schedule==expected,'Frozen schedule differs from qualified origins/design')
    return expected


def batch_indices(schedule,local_step):
    """Slice a schedule already verified/frozen by its owner; no full refit/rebuild."""
    D.need(_integer(local_step) and 0<=int(local_step)<TOTAL_STEPS,'Training clock outside fixed budget')
    D.need(schedule.get('schema')==SCHEMA and schedule.get('total_steps')==TOTAL_STEPS and schedule.get('batch_size')==4 and schedule.get('total_draws')==16000 and len(schedule.get('draws',[]))==16000,'Fixed schedule shape required')
    start=4*int(local_step);rows=schedule['draws'][start:start+4]
    D.need(all(r['draw_index']==start+i and r['split']=='train' and _integer(r['train_index']) and 0<=r['train_index']<schedule['valid_window_count'] for i,r in enumerate(rows)),'Batch no longer matches fixed TRAIN cursor')
    return np.asarray([r['train_index'] for r in rows],np.int64)
