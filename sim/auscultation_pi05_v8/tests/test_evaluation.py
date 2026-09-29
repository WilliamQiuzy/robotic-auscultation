"""Synthetic numeric tests only; no model, GPU, or qualified data is fabricated."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.auscultation_pi05_v7.tests.test_coverage_pilot_evaluation import fixture
from sim.auscultation_pi05_v8 import action_encoding as C


def api():
    return importlib.import_module('sim.auscultation_pi05_v8.evaluation')


def sample(mode=C.B):
    e=api(); old=fixture()
    records=[ep for ep in old.episodes if ep['split']=='train']
    stats=C.fit_stats(records,mode)
    data=SimpleNamespace(parent=old,mode=mode,data_hash=mode,
                         normalization={'residual_actions':stats})
    rows=e.build_diagnostic_schedule(data); ref,anchor=e.references(data,rows)
    z=np.stack([C.encode_chunk(r,q,row['phase'],stats,mode=mode) for r,q,row in zip(ref,anchor,rows)])
    p=np.stack([C.decode_chunk(v,q,row['phase'],stats,mode=mode) for v,q,row in zip(z,anchor,rows)])
    return e,data,rows,ref,anchor,p,z


def test_same_diagnostic_origins_physical_metrics_and_strict_bounds():
    e,d,rows,ref,anchor,p,z=sample()
    from sim.auscultation_pi05_v7 import coverage_pilot_evaluation as old
    assert rows==old.build_diagnostic_schedule(d.parent)
    assert e.metrics(p,ref,anchor,rows)==old.metrics(p,ref,anchor,rows)
    assert e.metrics(p,ref,anchor,rows)['moving_buckets']['passed']==16
    p[0,0,3]=-1e-12
    assert not e.metrics(p,ref,anchor,rows)['bounds']['all_passed']


@pytest.mark.parametrize('mode',C.MODES)
def test_numeric_report_reconstructs_and_rejects_tampering(tmp_path,monkeypatch,mode):
    e,d,rows,ref,anchor,p,z=sample(mode)
    b=dict(schema=e.BINDING_SCHEMA,mode=mode,checkpoint_path=str(tmp_path/'model/step-000000001217'),
           local_step=1217,optimizer_step=1217,data_hash=d.data_hash,
           normalization_sha256=e.digest(d.normalization))
    monkeypatch.setattr(e,'verify_checkpoint_binding',lambda value:None)
    dest=tmp_path/'diagnostics/local_001217'
    saved=e.save_checkpoint_report(dest,d,p,z,b)
    assert e.verify_checkpoint_report(dest,d,b)==saved
    assert saved['mode']==mode and saved['optimizer_step']==1217
    assert saved['initialization_baseline_only'] is False
    report=dest/'report.json';obj=json.loads(report.read_text())
    obj['metrics']['scores']['macro_first_mae_rad']=0.1;report.write_text(json.dumps(obj))
    with pytest.raises(ValueError):e.verify_checkpoint_report(dest,d,b)


def test_cross_arm_or_wrong_decode_rejected_before_report_creation(tmp_path,monkeypatch):
    e,d,rows,ref,anchor,p,z=sample()
    b=dict(mode=C.A,checkpoint_path=str(tmp_path/'model/step-000000001217'),local_step=1217,
           data_hash=d.data_hash,normalization_sha256=e.digest(d.normalization))
    monkeypatch.setattr(e,'verify_checkpoint_binding',lambda value:None)
    dest=tmp_path/'diagnostics/local_001217'
    with pytest.raises(ValueError):e.save_checkpoint_report(dest,d,p,z,b)
    b['mode']=C.B
    with pytest.raises(ValueError):e.save_checkpoint_report(dest,d,p,z+1,b)
    assert not dest.exists()


def candidate(step,worst,macro,eligible=True):
    return dict(local_step=step,eligible=eligible,scores=dict(
        worst_initial_first_max_abs_rad=worst,macro_first_mae_rad=macro,macro_full_mae_rad=macro))


def test_original_ranking_no_step0_or_failed_candidate_fallback():
    e=api()
    rows=[candidate(0,0,0),candidate(1217,.1,.001),candidate(2434,.01,.1),candidate(14600,.001,.001,False)]
    assert e.pick_candidate(rows)['local_step']==2434
    with pytest.raises(ValueError):e.pick_candidate([rows[0],rows[-1]])


def test_selection_summary_keeps_negative_control_non_deployable():
    e=api();rows=[candidate(s,.01,.001,False) for s in e.EXPECTED_STEPS]
    a=e.choose_result(rows,C.A)
    assert a['selected'] is None and a['descriptive_only_local_step']==14600
    assert a['status']=='rejected_no_eligible_candidate' and not a['service_eligible']
    b=e.choose_result(rows,C.B)
    assert b['selected'] is None and b['descriptive_only_local_step'] is None
    assert not b['service_eligible']
    rows[4]['eligible']=True
    assert e.choose_result(rows,C.B)['selected']['local_step']==4867
    with pytest.raises(ValueError):e.choose_result(rows[:-1],C.B)
    with pytest.raises(ValueError):e.choose_result(rows+[rows[0]],C.B)


@pytest.mark.parametrize('update',[{'optimizer_step':5217},{'mode':'legacy'}, {'schema':'old'}, {'local_step':True}])
def test_binding_rejects_wrong_fresh_clock_mode_and_schema(update):
    e=api()
    b={key:'f'*64 for key in e.BINDING_KEYS}
    b.update(schema=e.BINDING_SCHEMA,mode=C.B,checkpoint_path='/unused',local_step=1217,optimizer_step=1217)
    b.update(update)
    with pytest.raises(ValueError):e.verify_checkpoint_binding(b)


def prepared_run(tmp_path,monkeypatch,mode,eligible):
    """Real reports and locks; mock only checkpoint and acquisition boundaries."""
    import sys
    e,d,rows,ref,anchor,p,z=sample(mode)
    run=tmp_path/'run';run.mkdir()
    if not eligible:
        hold=np.repeat(anchor[:,None,:],10,axis=1)
        # Unambiguous raw violation; avoid near-zero codec rounding making a
        # nominal hold microscopically outperform the original hold criterion.
        hold[0,0,3]=-.01
        z=np.stack([C.encode_chunk(r,q,row['phase'],d.normalization['residual_actions'],mode=mode)
                    for r,q,row in zip(hold,anchor,rows)])
        p=np.stack([C.decode_chunk(v,q,row['phase'],d.normalization['residual_actions'],mode=mode)
                    for v,q,row in zip(z,anchor,rows)])
    config=dict(training_run_path=str(run),mode=mode,source_sha256={'synthetic':'f'*64})
    context=dict(dataset=d,config=config,schedule=dict(candidate_steps=list(e.EXPECTED_STEPS),total_steps=14600),
                 completed=True,candidate_paths=[])
    bindings={}
    monkeypatch.setitem(sys.modules,'sim.auscultation_pi05_v8.train',SimpleNamespace(
        checkpoint_binding=lambda path:bindings[str(Path(path).resolve())]))
    monkeypatch.setattr(e,'verify_checkpoint_binding',lambda value:None)
    monkeypatch.setattr(e,'load_context',lambda path:context)
    for n,v in [('configuration.json',dict(config=config)),('training_run.json',dict(training_complete=True)),
                ('diagnostic_schedule.json',dict(data_hash=d.data_hash,rows=rows)),('training_schedule.json',context['schedule'])]:
        e.write_new(run/n,v)
    for step in e.EXPECTED_STEPS:
        path=run/'model'/('step-%012d'%step);path.mkdir(parents=True)
        b=dict(schema=e.BINDING_SCHEMA,mode=mode,checkpoint_path=str(path),local_step=step,optimizer_step=step,
               data_hash=d.data_hash,normalization_sha256=e.digest(d.normalization),config_sha256=e.digest(config),
               source_sha256=config['source_sha256'])
        bindings[str(path)]=b;context['candidate_paths'].append(path)
        e.save_checkpoint_report(run/'diagnostics'/('local_%06d'%step),d,p,z,b)
    return e,run,context,bindings


@pytest.mark.parametrize('mode,eligible',[(C.A,False),(C.B,False),(C.B,True)])
def test_full_numeric_selection_lock_reconstruction_and_no_second_selection(tmp_path,monkeypatch,mode,eligible):
    e,run,context,bindings=prepared_run(tmp_path,monkeypatch,mode,eligible)
    result=e.select_run(run)
    assert not result['service_eligible']
    assert (result['selected_local_step']==1217) if eligible else (result['selected_local_step'] is None)
    assert result['descriptive_only_local_step']==(14600 if mode==C.A and not eligible else None)
    lock=e.verify_selection_lock(result['selection_lock'],result['selection_lock_sha256'])
    assert lock['selection']['candidate_count']==13
    assert len(lock['selection']['artifact_sha256'])==30
    with pytest.raises(ValueError):e.select_run(run)
    p=run/'diagnostics/local_014600/report.json';obj=json.loads(p.read_text())
    obj['metrics']['scores']['macro_first_mae_rad']=999;p.write_text(json.dumps(obj))
    with pytest.raises(ValueError):e.verify_selection_lock(result['selection_lock'])


def test_incomplete_budget_cannot_select(tmp_path,monkeypatch):
    e,run,context,_=prepared_run(tmp_path,monkeypatch,C.B,True)
    context['completed']=False
    with pytest.raises(ValueError):e.build_selection(context)
    context['completed']=True;context['candidate_paths'].pop()
    with pytest.raises(ValueError):e.build_selection(context)
