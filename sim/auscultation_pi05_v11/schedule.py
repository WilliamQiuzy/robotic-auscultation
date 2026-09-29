"""Declared TRAIN home coverage floor over the immutable V10 55/20/25 draws."""
from collections import Counter
from pathlib import Path
import copy
import numpy as np
from sim.auscultation_pi05_v10 import schedule as P
from . import training_data as D

SCHEMA='wam_pi05_v11_bounded_schedule_v1'
TOTAL_STEPS=4000
BATCH_SIZE=4
CANDIDATE_STEPS=P.CANDIDATE_STEPS
GROUPS=P.GROUPS
QUOTAS=P.QUOTAS
PHASES=D.PHASES
file_sha=D.file_sha
digest=D.digest
json_bytes=D.json_bytes


def replace_home_slots(parent_draws,inventory,original_ids):
    D.need(len(parent_draws)==16000 and len(original_ids)==8 and len(set(original_ids))==8,'Exact parent denominator required')
    lookup={(r['episode_id'],r['start']):r for r in inventory}
    D.need(len(lookup)==len(inventory),'Unique prequalified TRAIN inventory required')
    draws=copy.deepcopy(parent_draws);reserved=[]
    for block in range(8):
        starts=[0]+[1+((3*block+k)%12) for k in range(3)]
        for eid in original_ids:
            slots=[i for i,r in enumerate(draws) if r['block']==block and r['episode_id']==eid and r['phase']=='FREE' and r['group']=='original8'][:4]
            D.need(len(slots)==4,'Four original FREE slots required in every block')
            for i,s in zip(slots,starts):
                before=parent_draws[i];after=lookup.get((eid,s));D.need(after is not None and after['phase']=='FREE' and after['group']=='original8','Reserved home origin must be prequalified FREE')
                D.need(before['draw_index']==i and before['block']==i//2000,'Original final draw order required')
                draws[i].update(after)
                reserved.append(dict(draw_index=i,block=block,episode_id=eid,parent_start=before['start'],reserved_start=s,changed_origin=before['start']!=s))
    reserved_by_index={r['draw_index']:r for r in reserved}
    for i,(old,new) in enumerate(zip(parent_draws,draws)):
        D.need(new['draw_index']==old['draw_index']==i and new['block']==old['block'],'Slot clock changed')
        D.need(all(new[k]==old[k] for k in ('group','phase','episode_id','episode_index','split')),'Cell quota changed')
        if i not in reserved_by_index:D.need(new==old,'Unreserved origin changed')
        new['parent_origin']=dict(train_index=old['train_index'],episode_index=old['episode_index'],episode_id=old['episode_id'],start=old['start'])
        new['home_coverage_reserved']=i in reserved_by_index
        # Parent deck bookkeeping no longer describes the modified sampling.
        new['parent_deck_position']={k:new.pop(k) for k in ('deck','deck_position') if k in new}
    for block in range(8):
        counts=Counter(r['group'] for r in draws[2000*block:2000*(block+1)])
        D.need(counts==dict(zip(GROUPS,QUOTAS)),'55/20/25 block quota changed')
    return draws,dict(reserved_slots=len(reserved),changed_origin_count=sum(r['changed_origin'] for r in reserved),reservations=reserved,
        unreserved_origins_unchanged=True,parent_group_phase_episode_counts_unchanged=True,
        parent_deck_uniqueness_claim_applies_to_modified_draws=False)


def build_schedule(dataset,*,design_binding):
    D.need(design_binding==dict(path=str(D.DESIGN),sha256=D.DESIGN_SHA256),'Approved V11 design required')
    spec=D.design();identity=D.dataset_binding(dataset);b=spec['schedule']['parent']
    parent=D._read_pinned(b['path'],b['sha256'],{})
    P.verify_schedule(parent,dataset.parent,design_binding=spec['data']['qualified_parent_dataset_binding']['design_binding'])
    D.need(len(dataset.indices['train'])==parent['valid_window_count']==24397,'Complete unchanged TRAIN inventory required')
    draws,audit=replace_home_slots(parent['draws'],parent['inventory'],D.original_ids(dataset))
    sources=D.core_sources();sources['sim/auscultation_pi05_v10/schedule.py']=file_sha(P.__file__)
    sources['sim/auscultation_pi05_v11/schedule.py']=file_sha(__file__)
    result=dict(schema=SCHEMA,mode=D.MODE,design_binding=design_binding,dataset_binding=identity,parent_schedule_binding=b,source_sha256=sources,
        seed=61,batch_size=4,total_steps=4000,total_draws=16000,candidate_steps=list(CANDIDATE_STEPS),
        valid_window_count=parent['valid_window_count'],train_inventory_sha256=parent['train_inventory_sha256'],inventory=parent['inventory'],
        qualification_scopes_used=['train'],blocks=parent['blocks'],home_coverage=audit,draws=draws)
    result['schedule_sha256']=digest(result)
    D.need(D.dataset_binding(dataset)==identity,'Dataset changed during schedule construction');return result


def verify_schedule(schedule,dataset,*,design_binding):
    expected=build_schedule(dataset,design_binding=design_binding);D.need(schedule==expected,'Frozen V11 schedule changed');return expected


def batch_indices(schedule,local_step):
    D.need(isinstance(local_step,(int,np.integer)) and not isinstance(local_step,(bool,np.bool_)) and 0<=int(local_step)<4000,'Invalid local optimizer step')
    D.need(schedule.get('schema')==SCHEMA and schedule.get('total_draws')==16000 and schedule.get('batch_size')==4 and len(schedule.get('draws',[]))==16000,'Fixed V11 schedule shape required')
    start=4*int(local_step);rows=schedule['draws'][start:start+4]
    D.need(all(r['draw_index']==start+i and r['split']=='train' and type(r['train_index']) is int and 0<=r['train_index']<schedule['valid_window_count'] for i,r in enumerate(rows)),'Batch outside fixed TRAIN schedule')
    return np.asarray([r['train_index'] for r in rows],np.int64)
