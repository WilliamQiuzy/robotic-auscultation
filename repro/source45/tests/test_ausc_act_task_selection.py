"""Synthetic validation arrays and legal URDF poses, never actual datasets."""
from copy import deepcopy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def module():
    name='sim.ausc_policy.m2_act_task_selection'
    assert importlib.util.find_spec(name), 'V9 task-space selection API missing'
    return importlib.import_module(name)


def fixture(tmp_path,monkeypatch):
    m=module();rt=m.rt
    from test_ausc_act_training_v6 import cache_fixture
    p,_,_=cache_fixture(tmp_path,monkeypatch)
    samples=deepcopy(p['train']);n=len(samples['states'])
    samples.update(schema=rt.given.MASKED_DATASET_SCHEMA,label_protocol=rt.shared.MASKED_LABEL_PROTOCOL)
    # Deliberately unequal per-window valid cells and two gap values.
    samples['action_is_pad']=np.arange(20)[None,:]>=np.asarray([1,3,2,1,2,3,1,1,1])[:,None]
    samples['actions']=np.zeros((n,20,7));samples['origins']=np.zeros((n,7))
    samples['states'][:,:7]=0;samples['histories'][:,:,:7]=0
    episodes=[];items=[]
    for i in range(3):
        target='R3' if i==0 else 'L3';gap=.04+.003*i;eid='selection-fixture-'+str(i)
        meta=dict(subject_id='p000923',split='validation',target_id=target,episode_id=eid,
            geometry_id=rt.geometry.GEOMETRY_ID,geometry_source_sha256=rt.geometry.GEOMETRY_SOURCE_SHA256,freshly_generated=True)
        config=rt.write_new(tmp_path/('val-config-'+str(i)+'.json'),dict(m2_given_contact_control=dict(metadata=meta,episode_id=eid)))
        report=rt.write_new(tmp_path/('val-report-'+str(i)+'.json'),dict(points=[dict(target=target,checked_prepose_gap_m=gap,phase_config=dict(precontact_gap_m=gap))]))
        episodes.append(dict(meta,config=config,run_result=report,split_sample_range=[3*i,3*i+3],windows=3))
        samples['states'][3*i:3*i+3,7:10]=[[0,0,-gap],[.029,0,-gap],[.031,0,-gap]]
        samples['histories'][3*i:3*i+3,:,7:10]=samples['states'][3*i:3*i+3,None,7:10]
        items.extend(dict(split='validation',sample_index=j,episode_id=eid,
            window=dict(valid_future_cells=int((~samples['action_is_pad'][j]).sum()))) for j in range(3*i,3*i+3))
    archive=tmp_path/'selection-validation.npz'
    with archive.open('xb') as f:np.savez(f,**{k:samples[k] for k in m.ARRAY_KEYS})
    normalization=rt.write_new(tmp_path/'selection-normalization.json',p['normalization'])
    header=dict(counts=dict(train=8373,validation=n),episodes=episodes,normalization=normalization,
        arrays=dict(validation=rt.reference(archive)),index=rt.write_new(tmp_path/'selection-index.json',dict(samples=items)))
    ref=rt.write_new(tmp_path/'selection-dataset.json',header)
    monkeypatch.setattr(m,'BASE29',ref);monkeypatch.setattr(m,'BASE_COUNTS',header['counts'])
    monkeypatch.setattr(m,'NORMALIZATION',normalization)
    monkeypatch.setattr(m.expanded,'load_dataset',lambda r:dict(validation=deepcopy(samples),preparation=header))
    return m,ref,samples,header


class AnalyticKin:
    """Invented FK with independent metres and radians for scale assertions."""
    lower_limits=np.full(7,-3.);upper_limits=np.full(7,3.)
    contact_frame=None
    def tcp(self,q,compression=0.):
        assert compression==0.
        result=np.eye(4);result[0,3]=q[0];result[:3,:3]=Rotation.from_rotvec([0,0,q[1]]).as_matrix()
        return result


def analytic(m,monkeypatch):
    class BoundAnalyticKin(AnalyticKin):
        urdf_path=m.kinematics.DEFAULT_URDF_PATH
        contact_frame=staticmethod(m.rt.geometry.contact_frame)
    monkeypatch.setattr(m.kinematics,'OpenArmKinematics',BoundAnalyticKin)
    monkeypatch.setattr(m,'_kinematics',lambda:BoundAnalyticKin())
    return BoundAnalyticKin


def test_freeze_exact_observed_gap_subset_order_and_teacher_cache_once(tmp_path,monkeypatch):
    m,ref,samples,header=fixture(tmp_path,monkeypatch)
    subset=m.freeze_subset(ref,tmp_path/'subset');value=m.rt.read_bound(subset)
    assert value['indices']==[0,1,3,4,6,7] and value['target_window_counts']==dict(R3=2,L3=4)
    assert value['selection_rule']=='observed_actual_POINT_prepose_error_le_30mm'
    assert value['normalization']==header['normalization'] and set(value['array_proofs'])==set(m.ARRAY_KEYS)
    assert value['teacher_cache'] and value['geometry_sources']
    good=m.score(np.zeros_like(samples['actions']),samples,subset)
    assert good['task_score']==0. and good['valid_cells']==dict(R3=4,L3=5)
    bad=deepcopy(samples);bad['action_is_pad'][0,0]=True
    with pytest.raises(ValueError):m.score(np.zeros_like(samples['actions']),bad,subset)
    bad=m.rt.read_bound(subset);bad['indices'].reverse()
    bad_ref=m.rt.write_new(tmp_path/'reordered-subset.json',bad)
    with pytest.raises(ValueError):m.score(np.zeros_like(samples['actions']),samples,bad_ref)


def test_each_pose_once_valid_cell_means_max_branches_scales_and_padding(tmp_path,monkeypatch):
    m,ref,samples,_=fixture(tmp_path,monkeypatch)
    Kin=analytic(m,monkeypatch)
    subset=m.freeze_subset(ref,tmp_path/'subset');prediction=np.zeros_like(samples['actions'])
    prediction[0,0,0]=.001 # R3: sum1 /4 cells
    prediction[3,0,1]=np.deg2rad(4) # L3: sum1 /5 cells
    prediction[samples['action_is_pad']]=np.nan
    result=m.score(prediction,samples,subset)
    assert result['target_means']==pytest.approx(dict(R3=.25,L3=.2)) and result['task_score']==pytest.approx(.25)
    prediction[3,0,1]=np.deg2rad(8)
    assert m.score(prediction,samples,subset)['task_score']==pytest.approx(.4)
    prediction[3,0,0]=.003
    assert m.score(prediction,samples,subset)['task_score']==pytest.approx(.6)
    prediction[0,0,0]=np.nan
    with pytest.raises(ValueError):m.score(prediction,samples,subset)


def test_same_joint_mae_different_tcp_ranking_and_exact_earliest_tie(tmp_path,monkeypatch):
    m,ref,samples,_=fixture(tmp_path,monkeypatch);Kin=analytic(m,monkeypatch)
    subset=m.freeze_subset(ref,tmp_path/'subset');a=np.zeros_like(samples['actions']);b=a.copy()
    a[:,:,0]=.001;b[:,:,6]=.001
    sa=m.score(a,samples,subset);sb=m.score(b,samples,subset)
    assert sa['raw_joint_mae_rad']==sb['raw_joint_mae_rad'] and sa['task_score']>sb['task_score']
    rows=[dict(step=s,**deepcopy(sb if s in (1000,1500) else sa)) for s in m.CANDIDATE_STEPS]
    assert m.select_candidate(rows)['step']==1000
    rows[3]['task_score']=np.nan
    with pytest.raises(ValueError):m.select_candidate(rows)
    with pytest.raises(ValueError):m.select_candidate(rows[:-1])


@pytest.mark.parametrize('fault',['gap','indices','geometry','kernel','cache','omitted','out_of_range','target'])
def test_subset_geometry_prediction_and_cache_corruption_rejects(tmp_path,monkeypatch,fault):
    m,ref,samples,header=fixture(tmp_path,monkeypatch);subset=m.freeze_subset(ref,tmp_path/'subset')
    value=m.rt.read_bound(subset);pred=np.zeros_like(samples['actions'])
    if fault=='gap':value['sources'][0]['actual_gap_m']+=.001
    elif fault=='indices':value['indices'][0]=2
    elif fault=='geometry':value['sources'][0]['geometry']['geometry_id']='old'
    elif fault=='kernel':value['geometry_sources']['geometry']['sha256']='0'*64
    elif fault=='cache':
        from pathlib import Path
        Path(value['teacher_cache']['path']).write_bytes(b'bad')
    elif fault=='omitted':pred=pred[:-1]
    elif fault=='out_of_range':pred[0,0,0]=1000.
    else:value['target_window_counts']['R3']=0
    if fault not in ('cache','omitted','out_of_range'):subset=m.rt.write_new(tmp_path/'corrupted-subset.json',value)
    with pytest.raises((ValueError,OSError)):m.score(pred,samples,subset)


def test_candidates_are_rescored_from_same_stored_predictions_exactly(tmp_path,monkeypatch):
    m,ref,samples,_=fixture(tmp_path,monkeypatch);subset=m.freeze_subset(ref,tmp_path/'subset');rows=[]
    for step in m.CANDIDATE_STEPS:
        prediction=np.zeros_like(samples['actions']);path=tmp_path/(str(step)+'.npy')
        with path.open('xb') as stream:np.save(stream,prediction,allow_pickle=False)
        rows.append(dict(step=step,predictions=m.rt.reference(path),score=m.score(prediction,samples,subset)))
    selection=dict(schema=m.SELECTION_SCHEMA,subset=subset,dataset=ref,candidates=rows,selected_step=500)
    assert m.verify_candidates(selection)['selected_step']==500
    selection['candidates'][1]['score']['task_score']=1e-15
    with pytest.raises(ValueError):m.verify_candidates(selection)


def test_only_exact_bound_kinematics_and_teacher_fk_is_not_recomputed(tmp_path,monkeypatch):
    m,ref,samples,_=fixture(tmp_path,monkeypatch);subset=m.freeze_subset(ref,tmp_path/'subset')
    with pytest.raises(ValueError):m.score(np.zeros_like(samples['actions']),samples,subset,kin=AnalyticKin())
    calls=[];original=m.kinematics.OpenArmKinematics.tcp
    def recorded(self,q,compression=0.):
        calls.append(q.copy());return original(self,q,compression)
    monkeypatch.setattr(m.kinematics.OpenArmKinematics,'tcp',recorded)
    count=len(m.rt.read_bound(subset)['pose_cells'])
    for _ in range(2):m.score(np.zeros_like(samples['actions']),samples,subset)
    assert len(calls)==2*count # predictions only; cached teacher FK is never run again.


def test_equal_joint_mae_has_distinct_ranking_with_actual_float64_urdf_fk(tmp_path,monkeypatch):
    m,ref,samples,_=fixture(tmp_path,monkeypatch);subset=m.freeze_subset(ref,tmp_path/'subset')
    first=np.zeros_like(samples['actions']);last=first.copy();first[:,:,0]=.001;last[:,:,6]=.001
    a=m.score(first,samples,subset);b=m.score(last,samples,subset)
    assert a['raw_joint_mae_rad']==b['raw_joint_mae_rad']
    assert a['task_score']!=b['task_score']
    kin=m.kinematics.OpenArmKinematics()
    with pytest.raises(ValueError,match='External kinematics'):m.score(last,samples,subset,kin=kin)


def test_official_float32_values_and_exact_float64_storage_have_same_score_proof(tmp_path,monkeypatch):
    m,ref,samples,_=fixture(tmp_path,monkeypatch);subset=m.freeze_subset(ref,tmp_path/'subset')
    original=np.zeros_like(samples['actions'],dtype=np.float32);original[:,:,0]=.001
    stored=m.rt.canonical_predictions(original)
    assert m.score(original,samples,subset)==m.score(stored,samples,subset)


def test_omitted_target_and_label_protocol_cannot_define_subset(tmp_path,monkeypatch):
    m,ref,samples,header=fixture(tmp_path,monkeypatch)
    samples['states'][:3,7]=.031;samples['histories'][:3,:,7]=.031
    with pytest.raises(ValueError):m.freeze_subset(ref,tmp_path/'missing-target')
    samples['states'][:3,7]=0;samples['histories'][:3,:,7]=0;samples['label_protocol']='invented'
    with pytest.raises(ValueError):m.freeze_subset(ref,tmp_path/'wrong-labels')


@pytest.mark.parametrize('change',['tcp','joint_state','limits'])
def test_genuine_kinematics_with_modified_instance_cannot_change_selection(tmp_path,monkeypatch,change):
    m,ref,samples,_=fixture(tmp_path,monkeypatch);subset=m.freeze_subset(ref,tmp_path/'subset')
    prediction=np.zeros_like(samples['actions']);prediction[:,:,0]=.01
    assert m.score(prediction,samples,subset)['task_score']>0
    kin=m._kinematics()
    if change=='tcp':
        teacher=kin.tcp(np.zeros(7));kin.tcp=lambda q,compression=0.:teacher.copy()
    elif change=='joint_state':kin._joints[0].origin[:3,3]+=.1
    else:kin.lower_limits=np.full(7,-100.);kin.upper_limits=np.full(7,100.)
    with pytest.raises(ValueError):m.score(prediction,samples,subset,kin=kin)


def test_inline_subset_cannot_rebind_teacher_cache_even_with_matching_hashes(tmp_path,monkeypatch):
    m,ref,samples,_=fixture(tmp_path,monkeypatch);subset=m.freeze_subset(ref,tmp_path/'subset')
    prediction=np.zeros_like(samples['actions']);assert m.score(prediction,samples,subset)['task_score']==0.
    value=m.rt.read_bound(subset);teacher=np.load(m.rt.bound_path(value['teacher_cache']),allow_pickle=False)
    teacher[:,:3,3]+=.1
    path=tmp_path/'substituted-teacher.npy'
    with path.open('xb') as stream:np.save(stream,teacher,allow_pickle=False)
    value['teacher_cache']=m.rt.reference(path);value['teacher_cache_proof']=m.measured.array_proof(teacher)
    with pytest.raises(ValueError):m.score(prediction,samples,value)


@pytest.mark.parametrize('change',['inline','symlink','hardlink','bytes_missing','sha_changed'])
def test_selection_requires_canonical_full_frozen_subset_reference(tmp_path,monkeypatch,change):
    from pathlib import Path
    import os
    m,ref,samples,_=fixture(tmp_path,monkeypatch);subset=m.freeze_subset(ref,tmp_path/'subset');bad=deepcopy(subset)
    if change=='inline':bad=m.rt.read_bound(subset)
    elif change in ('symlink','hardlink'):
        path=tmp_path/'subset-alias.json'
        if change=='symlink':path.symlink_to(Path(subset['path']))
        else:os.link(subset['path'],path)
        bad['path']=str(path)
    elif change=='bytes_missing':bad.pop('bytes')
    else:bad['sha256']='0'*64
    with pytest.raises(ValueError):m.score(np.zeros_like(samples['actions']),samples,bad)
