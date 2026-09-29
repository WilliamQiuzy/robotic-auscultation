"""Decision-boundary tests; synthetic predictions are never model evidence."""
import numpy as np
import pytest
from sim.auscultation_pi05_v11 import evaluation as E


def test_phase_macro_does_not_reward_long_hold():
    rows=[{'phase':p,'target':'R4'} for p,n in zip(E.PHASES,[1,1,1,100,1]) for _ in range(n)]
    ref=np.zeros((len(rows),10,7));pred=ref.copy()
    for i,r in enumerate(rows):pred[i]+=0 if r['phase']=='HOLD' else .01
    result=E.phase_metrics(pred,ref,np.zeros((len(rows),7)),rows)
    assert result['macro_first_mae_rad']==pytest.approx(.008)
    assert result['macro_full_mae_rad']==pytest.approx(.008)


def test_posture_uses_command_tolerance_without_observed_slack():
    rows=[{'phase':'FREE','target':'R4'}]
    plans={'R4':{'joint_lower_rad':[-1]*7,'joint_upper_rad':[1]*7}}
    p=np.zeros((1,10,7));p[0,9,4]=1+2e-9
    assert E.raw_gates(p,rows,plans)['posture']['violating_coordinates']==1
    p[0,9,4]=1+0.5e-9
    assert E.raw_gates(p,rows,plans)['posture']['all_passed']


def test_full_horizon_urdf_never_clipped():
    p=np.zeros((1,10,7));p[0,9,0]=E.UPPER[0]+.01
    plans={'R4':{'joint_lower_rad':[-99]*7,'joint_upper_rad':[99]*7}}
    before=p.tobytes();r=E.raw_gates(p,[{'target':'R4'}],plans)
    assert not r['urdf']['all_passed'] and r['urdf']['violating_coordinates']==1
    assert p.tobytes()==before


def test_nonfinite_is_completed_negative_not_rankable():
    p=np.zeros((5,10,7));p[0,0,0]=np.nan
    rows=[{'target':'R4','phase':x} for x in E.PHASES]
    assert E.phase_metrics(p,np.zeros_like(p),np.zeros((5,7)),rows)['rankable'] is False


def test_infinite_raw_is_json_accountable_without_invented_finite_value():
    import json
    p=np.zeros((1,10,7));p[0,0,0]=np.inf
    plans={'R4':{'joint_lower_rad':[-1]*7,'joint_upper_rad':[1]*7}}
    result=E.raw_gates(p,[{'target':'R4'}],plans)
    json.dumps(result,allow_nan=False)
    assert not result['urdf']['all_passed']
    assert result['urdf']['violations'][0]['proposed_rad'] is None


def test_selection_full_denominator_and_positive_tie_break():
    candidates=[{'local_step':s,'eligible':s>0,'scores':{'macro_first_mae_rad':.01,'macro_full_mae_rad':.02}} for s in E.STEPS]
    assert E.choose_candidate(candidates)['local_step']==500
    with pytest.raises(ValueError):E.choose_candidate(candidates[:-1])
    candidates[0]['eligible']=True
    assert E.choose_candidate(candidates)['local_step']==500
    for r in candidates:r['eligible']=False
    assert E.choose_candidate(candidates) is None


def test_decoder_identity_errors_are_not_scientific_negatives(monkeypatch):
    from types import SimpleNamespace
    data=SimpleNamespace(episodes=[{'episode_id':'R4_x','state':np.zeros((1,41)),'targets':['R4'],'phases':['FREE']}]);rows=[dict(episode_id='R4_x',episode_index=0,start=0,phase='FREE',target='R4')]
    def changed(*args):raise ValueError('Dataset identity changed')
    monkeypatch.setattr(E.D,'decode_prediction',changed)
    with pytest.raises(ValueError,match='identity'):E.decode_accounted(data,np.zeros((1,10,7)),rows)


def test_report_schedule_keeps_full_heldout_and_original120(monkeypatch):
    from types import SimpleNamespace
    ids=[s+str(i)+'_original' for i in range(1,5) for s in ('R','L')]
    def episode(eid,target,split):
        return dict(episode_id=eid,target=target,split=split,phases=np.repeat(E.PHASES,40),targets=np.array([target]*200),state=np.zeros((200,41)),actions=np.zeros((200,7)))
    episodes=[episode(eid,eid.split('_')[0],'train') for eid in ids]
    train=[(ei,40*p+s) for ei in range(8) for p in range(5) for s in range(21)]
    norm={'residual_actions':{'fit_episode_ids':ids+['R4_new'],'bounds_contract':{}}}
    view=SimpleNamespace(episodes=episodes,indices={'train':train},normalization={'residual_actions':{'fit_episode_ids':ids}})
    monkeypatch.setattr(E.D,'original_train_view',lambda _:view)
    monkeypatch.setattr(E.D,'original_ids',lambda _:ids)
    monkeypatch.setattr(E.D.codec,'inverse_coordinates',lambda *a:None)
    val=episode('R5_original','R5','validation');added=episode('R4_new','R4','train')
    data=SimpleNamespace(episodes=episodes+[val,added],indices={'train':train+[(9,s) for s in (0,1,2,40,80,120,160)],'validation':[(8,40*p+s) for p in range(5) for s in range(21)]},loaded_scopes=('train','validation'),normalization=norm)
    rows=E.build_report_schedule(data,'train')
    assert len(rows)==127 and [r['group'] for r in rows[:120]]==['original120']*120
    assert [(r['episode_id'],r['start']) for r in rows[:120]]==[(r['episode_id'],r['start']) for r in E.original.build_diagnostic_schedule(view)]
    assert all(r['episode_index']==9 for r in rows[120:])
    full=E.build_report_schedule(data,'validation')
    assert len(full)==105 and len({r['split_index'] for r in full})==105
    ref,anchor=E.references(data,full)
    assert ref.shape==(105,10,7) and anchor.shape==(105,7)
    assert len(E.build_probe_schedule(data))==164


@pytest.mark.parametrize('changed',['none','marker','selection_bytes','lock','marker_symlink','selection_symlink'])
def test_selection_verifier_retains_early_three_file_identity(tmp_path,monkeypatch,changed):
    from sim.auscultation_pi05_v11 import train as T
    run=tmp_path.resolve();folder=run/'selection';folder.mkdir();sp=folder/'selection.json';lp=folder/'selection.lock.json';marker=run/E.MARKER
    selection=dict(run=str(run),selected=None,status='rejected_no_eligible_candidate')
    E.write_new(sp,selection)
    lock=dict(schema=E.LOCK_SCHEMA,selection_path=str(sp),selection_sha256=E.file_sha(sp),selection=selection)
    E.write_new(lp,lock);E.write_new(marker,dict(selection_lock=str(lp),selection_lock_sha256=E.file_sha(lp)))
    original=E.file_sha(lp)
    def context(_):
        if changed=='marker':marker.write_text('{}\n')
        elif changed=='selection_bytes':sp.write_bytes(sp.read_bytes()+b' ')
        elif changed=='lock':lp.write_bytes(lp.read_bytes()+b' ')
        elif changed.endswith('_symlink'):
            p=marker if changed.startswith('marker') else sp;other=p.with_suffix('.copied');other.write_bytes(p.read_bytes());p.unlink();p.symlink_to(other)
        return {'synthetic_only':True}
    monkeypatch.setattr(T,'verified_run_context',context);monkeypatch.setattr(E,'build_selection',lambda _:selection)
    if changed=='none':assert E.verify_selection_lock(lp,original)==lock
    else:
        with pytest.raises(ValueError):E.verify_selection_lock(lp,original)
