"""Verify the declared coverage replacement over a synthetic parent schedule."""
from collections import Counter
import copy
import pytest
from sim.auscultation_pi05_v11 import schedule as S


def fixture():
    ids=['R1','L1','R2','L2','R3','L3','R4','L4'];inventory=[]
    for ei,eid in enumerate(ids):
        for start in range(20):inventory.append(dict(train_index=len(inventory),episode_index=ei,episode_id=eid,start=start,phase='FREE',split='train',group='original8'))
    extra=[]
    for group in ('existing5','new3_precontact'):
        r=dict(train_index=len(inventory),episode_index=len(ids)+len(extra),episode_id=group,start=20,phase='APPROACH',split='train',group=group);inventory.append(r);extra.append(r)
    draws=[]
    for block in range(8):
        rows=[inventory[(i%8)*20+19] for i in range(1100)]+[extra[0]]*400+[extra[1]]*500
        for r in rows:draws.append(dict(r,draw_index=len(draws),block=block,deck=0,deck_position=len(draws),logical_draw_ordinal=len(draws)%2000))
    return ids,inventory,draws


def test_exact_home_floor_without_changing_cells():
    ids,inventory,parent=fixture();before=copy.deepcopy(parent)
    actual,audit=S.replace_home_slots(parent,inventory,ids)
    assert parent==before and audit['reserved_slots']==256 and audit['changed_origin_count']==256
    for eid in ids:
        starts=Counter(r['reserved_start'] for r in audit['reservations'] if r['episode_id']==eid)
        assert starts==dict([(0,8)]+[(k,2) for k in range(1,13)])
    slots={r['draw_index'] for r in audit['reservations']}
    for i,(a,b) in enumerate(zip(actual,parent)):
        assert 'deck' not in a and 'deck_position' not in a
        assert a['parent_origin']['start']==b['start']
        assert (a['group'],a['phase'],a['episode_id'])==(b['group'],b['phase'],b['episode_id'])
        if i not in slots:assert a['start']==b['start'] and a['train_index']==b['train_index']


def test_missing_reserved_origin_rejected():
    ids,inventory,parent=fixture()
    with pytest.raises(ValueError,match='prequalified'):S.replace_home_slots(parent,inventory[1:],ids)


def test_absent_cell_rejected():
    ids,inventory,parent=fixture()
    for r in parent:
        if r['block']==0 and r['episode_id']=='R1':r['phase']='HOLD'
    with pytest.raises(ValueError,match='Four'):S.replace_home_slots(parent,inventory,ids)
