"""Synthetic numeric tests only; no model, GPU, or qualified data is fabricated."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.auscultation_pi05_v7.tests.test_coverage_pilot_evaluation import fixture
from sim.auscultation_pi05_v9 import action_encoding as C
from sim.auscultation_pi05_v8 import action_encoding as B


def api():
    return importlib.import_module('sim.auscultation_pi05_v9.evaluation')


def sample(mode=C.C):
    e=api(); old=fixture()
    records=[ep for ep in old.episodes if ep['split']=='train']
    stats=C.fit_stats(records,mode)
    bdata=SimpleNamespace(parent=old,mode=B.B)
    data=SimpleNamespace(parent=bdata,mode=mode,data_hash=mode,
                         normalization={'residual_actions':stats})
    rows=e.build_diagnostic_schedule(data); ref,anchor=e.references(data,rows)
    z=np.stack([C.encode_chunk(r,q,row['phase'],stats,mode=mode) for r,q,row in zip(ref,anchor,rows)])
    p=np.stack([C.decode_chunk(v,q,row['phase'],stats,mode=mode) for v,q,row in zip(z,anchor,rows)])
    return e,data,rows,ref,anchor,p,z


def test_same_diagnostic_origins_physical_metrics_and_strict_bounds():
    e,d,rows,ref,anchor,p,z=sample()
    from sim.auscultation_pi05_v7 import coverage_pilot_evaluation as old
    assert rows==old.build_diagnostic_schedule(d.parent.parent)
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
    assert saved['representation_metrics']['descriptive_only'] is True
    assert saved['representation_metrics']['affects_eligibility'] is False
    assert saved['representation_metrics']['q4']['coordinate_count']==1200
    report=dest/'report.json';obj=json.loads(report.read_text())
    obj['metrics']['scores']['macro_first_mae_rad']=0.1;report.write_text(json.dumps(obj))
    with pytest.raises(ValueError):e.verify_checkpoint_report(dest,d,b)


def test_cross_arm_or_wrong_decode_rejected_before_report_creation(tmp_path,monkeypatch):
    e,d,rows,ref,anchor,p,z=sample()
    b=dict(mode="B_horizon_free_approach",checkpoint_path=str(tmp_path/'model/step-000000001217'),local_step=1217,
           data_hash=d.data_hash,normalization_sha256=e.digest(d.normalization))
    monkeypatch.setattr(e,'verify_checkpoint_binding',lambda value:None)
    dest=tmp_path/'diagnostics/local_001217'
    with pytest.raises(ValueError):e.save_checkpoint_report(dest,d,p,z,b)
    b['mode']=C.C
    with pytest.raises(ValueError):e.save_checkpoint_report(dest,d,p,z+1,b)
    assert not dest.exists()


def candidate(step,worst,macro,eligible=True):
    return dict(local_step=step,eligible=eligible,scores=dict(
        worst_initial_first_max_abs_rad=worst,macro_first_mae_rad=macro,macro_full_mae_rad=macro))


def terminal_fixture(e,run):
    segment=run/'training_segments/current.json';segment.parent.mkdir(exist_ok=True)
    receipt={'training_complete':True,'current_segment_path':str(segment)}
    (run/'training_run.json').write_text(json.dumps(receipt));segment.write_text(json.dumps(receipt))
    return dict(path=str(segment),sha256=e.file_sha(segment),cancellation_path=str(segment.with_suffix('.cancelled.json')))


def test_original_ranking_no_step0_or_failed_candidate_fallback():
    e=api()
    rows=[candidate(0,0,0),candidate(1217,.1,.001),candidate(2434,.01,.1),candidate(14600,.001,.001,False)]
    assert e.pick_candidate(rows)['local_step']==2434
    with pytest.raises(ValueError):e.pick_candidate([rows[0],rows[-1]])


def test_selection_summary_rejects_negative_C_and_preserves_original_ranking():
    e=api();rows=[candidate(s,.01,.001,False) for s in e.EXPECTED_STEPS]
    c=e.choose_result(rows,C.C)
    assert c['selected'] is None and c['descriptive_only_local_step'] is None
    assert c['status']=='rejected_no_eligible_candidate' and not c['service_eligible']
    rows[4]['eligible']=True
    assert e.choose_result(rows,C.C)['selected']['local_step']==4867
    assert e.choose_result(rows,C.C)['status']=='train_fit_candidate_requires_comparison_and_fixed160'
    with pytest.raises(ValueError):e.choose_result(rows[:-1],C.C)
    with pytest.raises(ValueError):e.choose_result(rows+[rows[0]],C.C)
    with pytest.raises(ValueError):e.choose_result(rows,B.B)


@pytest.mark.parametrize('update',[{'optimizer_step':5217},{'mode':'legacy'}, {'schema':'old'}, {'local_step':True}])
def test_binding_rejects_wrong_fresh_clock_mode_and_schema(update):
    e=api()
    b={key:'f'*64 for key in e.BINDING_KEYS}
    b.update(schema=e.BINDING_SCHEMA,mode=C.C,checkpoint_path='/unused',local_step=1217,optimizer_step=1217)
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
        hold[0,0,0]=-100.
        z=np.stack([C.encode_chunk(r,q,row['phase'],d.normalization['residual_actions'],mode=mode)
                    for r,q,row in zip(hold,anchor,rows)])
        p=np.stack([C.decode_chunk(v,q,row['phase'],d.normalization['residual_actions'],mode=mode)
                    for v,q,row in zip(z,anchor,rows)])
    config=dict(training_run_path=str(run),mode=mode,source_sha256={'synthetic':'f'*64})
    context=dict(dataset=d,config=config,schedule=dict(candidate_steps=list(e.EXPECTED_STEPS),total_steps=14600),
                 completed=True,candidate_paths=[])
    bindings={}
    monkeypatch.setitem(sys.modules,'sim.auscultation_pi05_v9.train',SimpleNamespace(
        checkpoint_binding=lambda path:bindings[str(Path(path).resolve())]))
    monkeypatch.setattr(e,'verify_checkpoint_binding',lambda value:None)
    monkeypatch.setattr(e,'load_context',lambda path:context)
    for n,v in [('configuration.json',dict(config=config)),('training_run.json',dict(training_complete=True)),
                ('diagnostic_schedule.json',dict(data_hash=d.data_hash,rows=rows)),('training_schedule.json',context['schedule']),
                ('initialization.json',{'synthetic':'initialization'})]:
        e.write_new(run/n,v)
    context['terminal_segment_binding']=terminal_fixture(e,run)
    context['verified_input_sha256']={str(run/n):e.file_sha(run/n) for n in e.CONTEXT_FILES}
    for step in e.EXPECTED_STEPS:
        path=run/'model'/('step-%012d'%step);path.mkdir(parents=True)
        b=dict(schema=e.BINDING_SCHEMA,mode=mode,checkpoint_path=str(path),local_step=step,optimizer_step=step,
               data_hash=d.data_hash,normalization_sha256=e.digest(d.normalization),config_sha256=e.digest(config),
               source_sha256=config['source_sha256'])
        bindings[str(path)]=b;context['candidate_paths'].append(path)
        e.save_checkpoint_report(run/'diagnostics'/('local_%06d'%step),d,p,z,b)
    return e,run,context,bindings


@pytest.mark.parametrize('mode,eligible',[(C.C,False),(C.C,True)])
def test_full_numeric_selection_lock_reconstruction_and_no_second_selection(tmp_path,monkeypatch,mode,eligible):
    e,run,context,bindings=prepared_run(tmp_path,monkeypatch,mode,eligible)
    result=e.select_run(run)
    assert not result['service_eligible']
    assert (result['selected_local_step']==1217) if eligible else (result['selected_local_step'] is None)
    assert result['descriptive_only_local_step']==None
    lock=e.verify_selection_lock(result['selection_lock'],result['selection_lock_sha256'])
    assert lock['selection']['candidate_count']==13
    assert len(lock['selection']['artifact_sha256'])==32
    with pytest.raises(ValueError):e.select_run(run)
    p=run/'diagnostics/local_014600/report.json';obj=json.loads(p.read_text())
    obj['metrics']['scores']['macro_first_mae_rad']=999;p.write_text(json.dumps(obj))
    with pytest.raises(ValueError):e.verify_selection_lock(result['selection_lock'])


def test_incomplete_budget_cannot_select(tmp_path,monkeypatch):
    e,run,context,_=prepared_run(tmp_path,monkeypatch,C.C,True)
    context['completed']=False
    with pytest.raises(ValueError):e.build_selection(context)
    context['completed']=True;context['candidate_paths'].pop()
    with pytest.raises(ValueError):e.build_selection(context)


def test_representation_slopes_endpoint_caveats_and_no_eligibility_change():
    e,d,rows,ref,anchor,p,z=sample()
    before=e.metrics(p,ref,anchor,rows)
    description=e.representation_metrics(d,p,z,ref,anchor,rows)
    assert description['q4']['slope']['undefined_count']==0
    assert 0 <= description['q4']['slope']['min'] <= description['q4']['slope']['max'] <= 1
    assert len(description['per_joint']['first_mae_rad'])==7
    assert e.metrics(p,ref,anchor,rows)==before
    r=np.array([-1.,0.,1.,0.,0.])
    a=np.array([0.,0.,C.U_SAFE_RAD,C.U_SAFE_RAD,1.])
    slopes=e.q4_slopes(r,a)
    assert slopes['undefined_count']==2 and slopes['finite_count']==3
    assert slopes['zero_at_lower_endpoint_count']==1 and slopes['zero_at_upper_endpoint_count']==1
    assert slopes['min']==0 and slopes['max']==1
    assert slopes['lower_endpoint_zero_left_right_limits']==[0.,1.]
    assert slopes['upper_endpoint_zero_left_right_limits']==[1.,0.]


@pytest.mark.parametrize('joint',range(7))
def test_any_of_seven_decoded_coordinates_cannot_be_forged(tmp_path,monkeypatch,joint):
    e,d,rows,ref,anchor,p,z=sample()
    binding=dict(mode=C.C,data_hash=d.data_hash,normalization_sha256=e.digest(d.normalization))
    monkeypatch.setattr(e,'verify_checkpoint_binding',lambda _:None)
    p=p.copy();p[0,9,joint]=np.nextafter(p[0,9,joint],np.float32(np.inf))
    with pytest.raises(ValueError,match='decode'):e._numeric(d,p,z,binding)


def test_report_keeps_consumed_json_identity_until_numeric_verification_finishes(tmp_path,monkeypatch):
    e,d,rows,ref,anchor,p,z=sample()
    binding=dict(schema=e.BINDING_SCHEMA,mode=C.C,checkpoint_path=str(tmp_path/'model/step-000000001217'),
                 local_step=1217,optimizer_step=1217,data_hash=d.data_hash,normalization_sha256=e.digest(d.normalization))
    monkeypatch.setattr(e,'verify_checkpoint_binding',lambda _:None)
    dest=tmp_path/'diagnostics/local_001217';e.save_checkpoint_report(dest,d,p,z,binding)
    original=e._numeric
    def mutate_after_json_read(*args):
        result=original(*args)
        path=dest/'report.json';obj=json.loads(path.read_text())
        obj['metrics']['scores']['macro_first_mae_rad']=987.;path.write_text(json.dumps(obj))
        return result
    monkeypatch.setattr(e,'_numeric',mutate_after_json_read)
    with pytest.raises(ValueError,match='evidence differs'):
        e.verify_checkpoint_report(dest,d,binding)


@pytest.mark.parametrize('name',['training_run.json','training_schedule.json','initialization.json'])
def test_context_input_identity_is_captured_before_trainer_and_kept_for_selection(tmp_path,monkeypatch,name):
    import sys
    e=api();run=tmp_path.resolve()
    for n in e.CONTEXT_FILES:e.write_new(run/n,{'synthetic':n})
    identity={str(run/n):e.file_sha(run/n) for n in e.CONTEXT_FILES}
    context=dict(config=dict(training_run_path=str(run),mode=C.C),
                 dataset=SimpleNamespace(mode=C.C),completed=True,verified_input_sha256=identity)
    def mutate_after_trainer_verification(path):
        (run/name).write_text('{"synthetic":"unverified replacement"}')
        return context
    monkeypatch.setitem(sys.modules,'sim.auscultation_pi05_v9.train',SimpleNamespace(
        verified_run_context=mutate_after_trainer_verification,checkpoint_binding=lambda _:None))
    with pytest.raises(ValueError,match='context input changed'):e.load_context(run)
    with pytest.raises(ValueError,match='context input changed'):e.build_selection(context)


def test_context_without_preverification_identity_cannot_select(tmp_path):
    e=api();context=dict(config=dict(training_run_path=str(tmp_path.resolve()),mode=C.C),
                        dataset=SimpleNamespace(mode=C.C),completed=True)
    with pytest.raises(ValueError,match='pre-context input identities'):e.build_selection(context)


@pytest.mark.parametrize('name',['report.json','predictions.npz'])
def test_selection_cannot_adopt_replacement_bytes_after_report_verifier(tmp_path,monkeypatch,name):
    """Synthetic acquisition/checkpoint stubs; actual closure and temporary I/O."""
    import sys
    e=api();run=tmp_path.resolve();data=SimpleNamespace(mode=C.C,data_hash='synthetic')
    config=dict(training_run_path=str(run),mode=C.C,source_sha256={})
    rows=[]
    for n in e.CONTEXT_FILES:
        e.write_new(run/n,dict(data_hash=data.data_hash,rows=rows) if n=='diagnostic_schedule.json' else {})
    paths=[run/'model'/('step-%012d'%s) for s in e.EXPECTED_STEPS]
    for path in paths:path.mkdir(parents=True)
    context=dict(config=config,dataset=data,completed=True,candidate_paths=paths,
            schedule=dict(candidate_steps=list(e.EXPECTED_STEPS),total_steps=14600),
            verified_input_sha256={str(run/n):e.file_sha(run/n) for n in e.CONTEXT_FILES})
    context['terminal_segment_binding']=terminal_fixture(e,run)
    context['verified_input_sha256']={str(run/n):e.file_sha(run/n) for n in e.CONTEXT_FILES}
    report=run/'diagnostics/local_000000/report.json';report.parent.mkdir(parents=True)
    report.write_text('{}');report.with_name('predictions.npz').write_bytes(b'synthetic original')
    binding=dict(checkpoint_path=str(paths[0]),local_step=0,optimizer_step=0,mode=C.C,
                 config_sha256=e.digest(config),source_sha256={})
    monkeypatch.setitem(sys.modules,'sim.auscultation_pi05_v9.train',SimpleNamespace(checkpoint_binding=lambda _:binding))
    monkeypatch.setattr(e,'build_diagnostic_schedule',lambda _:rows)
    def mutate_after_valid_report(*args):
        report.with_name(name).write_bytes(b'synthetic replacement')
        return {'metrics':{}}
    monkeypatch.setattr(e,'verify_checkpoint_report',mutate_after_valid_report)
    with pytest.raises(ValueError,match='Report identity changed'):e.build_selection(context)


@pytest.mark.parametrize('mutation',['current_cancel','dangling_cancel','segment_change','same_json_new_bytes','wrong_current','missing_binding'])
def test_current_terminal_segment_is_retained_and_cancelled_segment_rejected(tmp_path,mutation):
    e=api();run=tmp_path.resolve();binding=terminal_fixture(e,run)
    identities={str(run/'training_run.json'):e.file_sha(run/'training_run.json')}
    assert e.verify_terminal_segment(run,binding,identities)==binding['sha256']
    # A previous interrupted segment does not prevent a later clean resume.
    (run/'training_segments/previous.cancelled.json').write_text('{}')
    e.verify_terminal_segment(run,binding,identities)
    if mutation=='current_cancel':Path(binding['cancellation_path']).write_text('{}')
    elif mutation=='dangling_cancel':Path(binding['cancellation_path']).symlink_to(run/'absent')
    elif mutation=='segment_change':Path(binding['path']).write_text('{}')
    elif mutation=='same_json_new_bytes':
        path=Path(binding['path']);path.write_text(path.read_text()+'\n');binding['sha256']=e.file_sha(path)
    elif mutation=='wrong_current':
        receipt={'training_complete':True,'current_segment_path':str(run/'training_segments/previous.json')}
        (run/'training_run.json').write_text(json.dumps(receipt));identities[str(run/'training_run.json')]=e.file_sha(run/'training_run.json')
    else:binding=None
    with pytest.raises(ValueError):e.verify_terminal_segment(run,binding,identities)


@pytest.mark.parametrize('mutation',['late_cancel','late_segment','late_main'])
def test_terminal_identity_and_cancel_are_rechecked_after_consumption(tmp_path,monkeypatch,mutation):
    e=api();run=tmp_path.resolve();binding=terminal_fixture(e,run)
    identities={str(run/'training_run.json'):e.file_sha(run/'training_run.json')}
    original=e.read_pinned
    def change_after_read(path,*args):
        result=original(path,*args)
        if Path(path)==run/'training_run.json':
            if mutation=='late_cancel':Path(binding['cancellation_path']).write_text('{}')
            else:
                target=Path(binding['path']) if mutation=='late_segment' else Path(path)
                target.write_text(target.read_text()+'\n')
        return result
    monkeypatch.setattr(e,'read_pinned',change_after_read)
    with pytest.raises(ValueError,match='changed or was cancelled'):
        e.verify_terminal_segment(run,binding,identities)
