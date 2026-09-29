"""Synthetic coverage/selection isolation; no real held-out arrays or GPU."""
import importlib
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest


def module(name):
    try:
        result = importlib.import_module('sim.auscultation_pi05_v7.expanded_'+name)
    except ModuleNotFoundError:
        result = None
    assert result is not None, 'Independent expanded '+name+' entry point required'
    return result


PHASES = ('FREE', 'APPROACH', 'CONTACT', 'HOLD', 'RETRACT')
TARGETS = [side+str(i) for i in range(1, 6) for side in ('R', 'L')]


class Unopened:
    def __getitem__(self, key):
        raise AssertionError('Held-out test arrays were accessed during development')


def dataset():
    episodes = []; indices = dict(train=[], validation=[], test=[])
    for split, families in [('train', 3), ('validation', 1), ('test', 1)]:
        for family in range(families):
            for target in TARGETS:
                e = len(episodes); n = 150
                state = np.zeros((n, 41), np.float32); state[:, 3] = 1
                actions = state[:, :7].copy()
                phases = np.repeat(PHASES, 30)
                episodes.append(dict(episode_id=f'{split}-{family}-{target}', split=split,
                    phases=phases, targets=np.full(n, target), state=state, actions=actions))
                indices[split].extend((e, s) for s in range(n) if s % 30 <= 20)
    return SimpleNamespace(episodes=episodes, indices=indices)


def test_validation_covers_all_fifty_bins_first_last_unique_without_test_access():
    d = dataset()
    for e in d.episodes:
        if e['split'] == 'test':
            for key in ('actions', 'state', 'phases', 'targets'): e[key] = Unopened()
    m = module('evaluation'); rows = m.build_validation_schedule(d)
    assert len(rows) == 1000
    for e in [x for x in d.episodes if x['split'] == 'validation']:
        for pi, phase in enumerate(PHASES):
            selected = [r['frame'] for r in rows if r['episode_id'] == e['episode_id'] and r['phase'] == phase]
            assert len(selected) == len(set(selected)) == 20
            assert selected[0] == pi*30 and selected[-1] == pi*30+20
    sentinels = m.build_train_sentinels(d)
    assert len(sentinels) == 30 and all(r['phase'] == 'FREE' and r['frame'] == 0 for r in sentinels)


def test_sparse_bins_not_padded_and_empty_bin_rejected():
    d = dataset(); first = d.indices['validation'][0][0]
    d.indices['validation'] = [(e,s) for e,s in d.indices['validation'] if e != first or s >= 30 or s in (0, 20)]
    rows = module('evaluation').build_validation_schedule(d)
    assert len([r for r in rows if r['episode_index'] == first and r['phase']=='FREE']) == 2
    d.indices['validation'] = [(e,s) for e,s in d.indices['validation'] if not(e == first and s < 30)]
    with pytest.raises(ValueError, match='empty'): module('evaluation').build_validation_schedule(d)


def test_development_sampler_rejects_test_before_prediction():
    d=dataset(); m=module('evaluation'); called=[]
    row=dict(split='test', episode_index=40, dataset_index=0, frame=0,
        episode_id=d.episodes[40]['episode_id'], phase='FREE', target='R1')
    with pytest.raises(ValueError, match='test|development'):
        m.evaluate_development_rows(d, [row], lambda *args: called.append(args))
    assert not called


def test_macro_equal_bucket_not_global_weight_and_sentinel_not_ranked():
    m=module('evaluation'); rows=[]; p=[]; ref=[]; anchor=[]
    for bucket,n,error in [('R1',20,.1),('R2',1,.3)]:
        for i in range(n):
            rows.append(dict(episode_id=bucket,phase='HOLD',target=bucket,split='validation',frame=i))
            ref.append(np.tile([0,0,0,1,0,0,0],(10,1))); p.append(ref[-1]+error); anchor.append(ref[-1][:1])
    s=m.score_arrays(np.array(p),np.array(ref),np.array(anchor),rows)
    assert s['macro_first_action_mae_rad'] == pytest.approx(.2)
    assert s['macro_full_chunk_mae_rad'] == pytest.approx(.2)
    assert s['micro_first_action_mae_rad'] != pytest.approx(.2)
    veto=m.bounds_report(np.array(p))
    assert 'mae' not in json.dumps(veto)


def test_selector_feasible_first_and_no_feasible_refuses():
    m=module('selection')
    rows=[dict(step=1000,feasible=False,macro_first_action_mae_rad=.001,macro_full_chunk_mae_rad=.001),
          dict(step=2000,feasible=True,macro_first_action_mae_rad=.1,macro_full_chunk_mae_rad=.2),
          dict(step=3000,feasible=True,macro_first_action_mae_rad=.1,macro_full_chunk_mae_rad=.2)]
    assert m.pick_feasible(rows)['step']==2000
    with pytest.raises(ValueError,match='feasible'):m.pick_feasible(rows[:1])


def test_known_start_sentinel_raw_bounds_veto_even_if_validation_passes():
    m=module('evaluation'); a=np.zeros((30,10,7)); a[:,:,3]=1; a[7,9,3]=-1e-12
    result=m.bounds_report(a)
    assert result['all_passed'] is False
    assert result['violations']==[dict(record_index=7,horizon_index=9,joint_index=3,proposed_rad=-1e-12)]


def test_numeric_report_tampering_is_recomputed(tmp_path):
    m=module('evaluation'); d=dataset(); rows=m.build_validation_schedule(d)
    refs,anchors=m.reference_arrays(d,rows); pred=refs+.0001
    m.save_numeric_report(tmp_path/'result',pred,refs,anchors,rows,kind='validation',binding={'fixture':1})
    report=tmp_path/'result/report.json'; obj=json.loads(report.read_text())
    obj['scores']['macro_first_action_mae_rad']=0; report.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='numeric|score'):
        m.verify_numeric_report(report,d,rows,kind='validation',binding={'fixture':1})


def test_numeric_references_cannot_be_changed_with_rehashed_npz(tmp_path):
    m=module('evaluation'); d=dataset(); rows=m.build_validation_schedule(d); ref,anc=m.reference_arrays(d,rows)
    m.save_numeric_report(tmp_path/'result',ref,ref,anc,rows,kind='validation',binding={})
    path=tmp_path/'result/predictions.npz'; np.savez_compressed(path,prediction=ref,reference=ref+.01,measured_anchor=anc)
    report=tmp_path/'result/report.json'; obj=json.loads(report.read_text()); obj['predictions_sha256']=m.sha(path); report.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='reference|anchor'):
        m.verify_numeric_report(report,d,rows,kind='validation',binding={})


def test_final_test_once_claim_survives_failure_and_other_output_path(tmp_path):
    m=module('final_test'); lock=tmp_path/'selection.lock.json'; lock.write_text('{}')
    with pytest.raises(RuntimeError,match='injected'):
        with m.claim_test_once(lock,'a'*64,tmp_path/'attempt1'):
            raise RuntimeError('injected')
    assert (tmp_path/'final_test_started.json').is_file()
    with pytest.raises((ValueError,FileExistsError)):
        with m.claim_test_once(lock,'a'*64,tmp_path/'attempt2'): pass
    assert not (tmp_path/'attempt2').exists()


def test_final_test_lock_digest_mismatch_rejects_before_claim(tmp_path):
    m=module('final_test'); lock=tmp_path/'selection.lock.json'; lock.write_text('{}')
    with pytest.raises(ValueError,match='SHA|digest'):
        m.read_selection_lock(lock,'0'*64)
    assert not (tmp_path/'final_test_started.json').exists()


def complete_evaluation_fixture(tmp_path,monkeypatch):
    m=module('evaluation');d=dataset();b={'fixture':'synthetic'}
    context=dict(dataset=d,binding=b,checkpoints=[],metadata=[])
    validation=m.build_validation_schedule(d);sentinels=m.build_train_sentinels(d)
    reports=[];inventory={}
    for step in m.STEPS:
        cp=tmp_path/'model'/('step-%06d'%step);context['checkpoints'].append(cp);context['metadata'].append(dict(step=step))
        inventory[str(cp)]={'fixture':'not an actual checkpoint'}
    b['checkpoint_inventory']=inventory
    for step,cp in zip(m.STEPS,context['checkpoints']):
        binding=dict(b,step=step,checkpoint=str(cp))
        for kind,rows in [('validation',validation),('train_sentinel',sentinels)]:
            ref,anc=m.reference_arrays(d,rows);prediction=ref.copy()
            # First candidate has smallest validation MAE but one illegal train sentinel.
            if kind=='validation':prediction+=step/1e7
            elif step==1000:prediction[0,0,3]=-1e-7
            path=tmp_path/'evaluation'/('step_%06d'%step)/kind
            m.save_numeric_report(path,prediction,ref,anc,rows,kind=kind,binding=binding)
            reports.append(dict(step=step,kind=kind,path=str(path/'report.json'),sha256=m.sha(path/'report.json')))
    root=tmp_path/'evaluation/evaluation.json'
    m.write_new(root,dict(schema='wam_pi05_v7_expanded_evaluation_completed_v1',binding=b,
        validation_schedule=validation,train_sentinel_schedule=sentinels,reports=reports,
        evaluation_complete=True,test_predictions_generated=False,closed_loop_validated=False))
    monkeypatch.setattr(m,'verify_context_unchanged',lambda context:None)
    return m,context,root


def test_complete_twelve_numeric_reports_sentinel_veto_and_test_content_invariance(tmp_path,monkeypatch):
    m,context,path=complete_evaluation_fixture(tmp_path,monkeypatch)
    selector=module('selection'); result=selector.build_selection(context,path)
    assert result['selected']['step']==2000
    assert result['candidates'][0]['filter_reasons']==['known_start_train_raw_urdf_violation']
    for e in context['dataset'].episodes:
        if e['split']=='test':
            for key in ('state','actions','phases','targets'):e[key]=Unopened()
    assert selector.build_selection(context,path)==result


def test_complete_selection_rejects_missing_candidate_and_report_hash_tamper(tmp_path,monkeypatch):
    m,context,path=complete_evaluation_fixture(tmp_path,monkeypatch);r=m.read(path)
    r['reports'][-1]['sha256']='0'*64;path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Report list'):module('selection').build_selection(context,path)


def test_row_identity_and_sparse_target_coverage_cannot_be_forged():
    m=module('evaluation');d=dataset();rows=m.build_validation_schedule(d)
    rows[0]['dataset_index']=1
    with pytest.raises(ValueError,match='Row differs'):m.reference_arrays(d,rows)
    d=dataset();d.episodes[30]['targets'][:]='L1'
    with pytest.raises(ValueError,match='ten targets'):m.build_validation_schedule(d)


def test_nonfinite_predictions_fail_closed_without_numeric_pass():
    m=module('evaluation');a=np.zeros((1,10,7));a[:,:,3]=1;a[0,5,4]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):m.bounds_report(a)


def test_final_test_marker_is_atomic_and_shared_across_copied_locks(tmp_path):
    m=module('final_test');marker=tmp_path/'trained_run/FINAL_TEST_STARTED.json';marker.parent.mkdir()
    locks=[]
    for name in ('a','b'):
        parent=tmp_path/name;parent.mkdir();lock=parent/'selection.lock.json';lock.write_text('{}');locks.append(lock)
    with m.claim_test_once(locks[0],'a'*64,tmp_path/'test1',marker_path=marker):pass
    with pytest.raises(FileExistsError):
        with m.claim_test_once(locks[1],'b'*64,tmp_path/'test2',marker_path=marker):pass
    assert not (tmp_path/'test2').exists()


def test_fresh_output_cannot_be_nested_in_frozen_run_or_data(tmp_path):
    m=module('evaluation');run=tmp_path/'run';data=tmp_path/'data';side=tmp_path/'sidecar';source=tmp_path/'source'
    for root in (run,data,side,source):root.mkdir()
    binding=dict(run=str(run),manifest_path=str(data/'manifest.json'),experiment_path=str(side/'experiment.json'))
    for destination in (run/'new',data/'new',side/'new',source/'new',tmp_path):
        with pytest.raises(ValueError,match='protected|input'):
            m.require_fresh_output(destination,binding,source_root=source)
    m.require_fresh_output(tmp_path/'new',binding,source_root=source)


def test_final_lock_without_feasible_candidate_cannot_be_published(tmp_path):
    m=module('selection');path=tmp_path/'selection.json';path.write_text('{}')
    with pytest.raises(ValueError,match='feasible'):m.lock_payload({'selected':None},path)


def test_selection_marker_binds_exact_lock_and_cannot_be_replaced(tmp_path):
    s=module('selection');e=module('evaluation');run=tmp_path/'run';run.mkdir()
    lock=tmp_path/'selection.lock.json';lock.write_text(json.dumps({'selection':{'binding':{'run':str(run)}}}))
    s.bind_selection_marker(lock)
    original=(run/'SELECTION_LOCKED.json').read_bytes()
    copy=tmp_path/'copy.lock.json';copy.write_bytes(lock.read_bytes())
    with pytest.raises(FileExistsError):s.bind_selection_marker(copy)
    assert (run/'SELECTION_LOCKED.json').read_bytes()==original
    with pytest.raises(ValueError,match='selection marker'):s.verify_selection_marker(copy,e.read(copy))
    s.verify_selection_marker(lock,e.read(lock))


def test_final_run_checks_lock_before_model_or_test_access(tmp_path,monkeypatch):
    f=module('final_test');s=module('selection');lock=tmp_path/'bad.lock';lock.write_text('{}')
    calls=[]
    monkeypatch.setattr(s,'verify_selection_lock',lambda *a:calls.append('unexpected'))
    with pytest.raises(ValueError):f.run(SimpleNamespace(lock=lock,lock_sha256='0'*64,output=tmp_path/'test'))
    assert calls==[] and not (tmp_path/'test').exists()


def test_final_run_claims_before_model_init_and_failure_stays_consumed(tmp_path,monkeypatch):
    import sys
    f=module('final_test');s=module('selection');e=module('evaluation');run=tmp_path/'run';run.mkdir()
    checkpoint=run/'model/step-002000';checkpoint.mkdir(parents=True)
    for name in ('metadata.json','arrays.npz','COMPLETE.json'):(checkpoint/name).write_text('synthetic')
    data=tmp_path/'data';data.mkdir();side=tmp_path/'sidecar';side.mkdir()
    binding=dict(run=str(run),manifest_path=str(data/'manifest'),experiment_path=str(side/'experiment'))
    lock_obj=dict(schema=s.LOCK_SCHEMA,selection=dict(selected=dict(step=2000,checkpoint=str(checkpoint),
        checkpoint_sha256=e.checkpoint_inventory(checkpoint))),final_test_marker=str(run/'FINAL_TEST_STARTED.json'))
    lock=tmp_path/'selection.lock.json';lock.write_text(json.dumps(lock_obj))
    context=dict(dataset=dataset(),binding=binding)
    monkeypatch.setattr(s,'verify_selection_lock',lambda *args:(lock_obj,context))
    def fail_model(*args):
        assert (run/'FINAL_TEST_STARTED.json').exists()
        raise RuntimeError('injected model initialization failure')
    # Only the heavyweight model initializer is replaced; claim/path/lock flow is real.
    package=importlib.import_module('sim.auscultation_pi05_v7')
    fake=SimpleNamespace(initialize=fail_model)
    monkeypatch.setattr(package,'model',fake,raising=False)
    monkeypatch.setitem(sys.modules,'sim.auscultation_pi05_v7.model',fake)
    args=SimpleNamespace(lock=lock,lock_sha256=e.sha(lock),output=tmp_path/'test')
    with pytest.raises(RuntimeError,match='injected'):f.run(args)
    assert (tmp_path/'test/failed.json').exists()
    args.output=tmp_path/'different'
    with pytest.raises(FileExistsError):f.run(args)
    assert not args.output.exists()


def test_copied_completed_run_cannot_escape_original_marker_namespace(tmp_path):
    e=module('evaluation');original=tmp_path/'original';original.mkdir();copied=tmp_path/'copy';copied.mkdir()
    config={'training_run_path':str(original.resolve())}
    e.verify_training_run_identity(config,original)
    with pytest.raises(ValueError,match='original|training.run|copied'):
        e.verify_training_run_identity(config,copied)
    with pytest.raises(ValueError):e.verify_training_run_identity({},original)
