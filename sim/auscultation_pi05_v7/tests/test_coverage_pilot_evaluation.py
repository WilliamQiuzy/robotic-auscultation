"""Synthetic CPU-only diagnostic contracts; no qualified data/model is created."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

def module(name):
    try:result=importlib.import_module('sim.auscultation_pi05_v7.coverage_pilot_'+name)
    except ModuleNotFoundError:result=None
    assert result is not None,'Independent pilot diagnostic module required'
    return result

def fixture():
    from sim.auscultation_pi05_v7.action_encoding import fit_stats
    phases=('FREE','APPROACH','CONTACT','HOLD','RETRACT');episodes=[];indices=[];train=[]
    for i in range(8):
        anchor=np.array([0,-.5,0,1,0,0,0],np.float32);state=np.zeros((75,41),np.float32);state[:,:7]=anchor
        actions=np.tile(anchor,(75,1));labels=np.repeat(phases,15);times=np.arange(75)*.05
        target=('R' if i<4 else 'L')+str(i%4+1)
        for pi,phase in enumerate(phases):
            delta=np.arange(15,dtype=np.float32)[:,None]*np.full((1,7),.0001,np.float32)
            if phase=='HOLD':delta[:]=1e-7
            actions[pi*15:(pi+1)*15]+=delta
        starts=[pi*15+j for pi in range(5) for j in range(6)]
        ep=dict(episode_id='synthetic_'+target,split='train',state=state,actions=actions,phases=labels,
            targets=np.repeat(target,75),times=times,starts=starts)
        episodes.append(ep);train.append(ep);indices.extend((i,s) for s in starts)
    class Forbidden(dict):
        def __getitem__(self,key):
            if key not in ('episode_id','split'):raise AssertionError('Held-out sample access')
            return super().__getitem__(key)
    episodes.append(Forbidden(episode_id='forbidden',split='test'))
    return SimpleNamespace(episodes=episodes,indices={'train':indices},data_hash='a'*64,
        normalization={'residual_actions':fit_stats(train)})

def samples():
    e=module('evaluation');d=fixture();rows=e.build_diagnostic_schedule(d);ref,anchors=e.references(d,rows)
    return e,d,rows,ref,anchors

def test_exact120_train_first_middle_last_and_empty_rejection():
    e,d,rows,ref,anchors=samples()
    assert len(rows)==120 and len({(r['episode_id'],r['phase']) for r in rows})==40
    assert [r['start'] for r in rows[:3]]==[0,3,5]
    assert ref.shape==(120,10,7) and anchors.shape==(120,7)
    d.indices['train']=[p for p in d.indices['train'] if not(p[0]==0 and p[1]<15)]
    with pytest.raises(ValueError,match='three'):e.build_diagnostic_schedule(d)

def test_perfect_labels_pass_all_bounds_and16_progress_gates():
    e,d,rows,ref,a=samples();m=e.metrics(ref,ref,a,rows)
    assert m['eligible'] is True and m['bounds']['all_passed'] is True
    assert m['moving_buckets']['required']==16 and m['moving_buckets']['passed']==16
    assert m['scores']['worst_initial_first_max_abs_rad']==0
    assert len(m['initial_free'])==8 and len(m['buckets'])==40
    assert any(b['near_zero_count']==3 for b in m['buckets'] if b['phase']=='HOLD')

def test_hold_baseline_cannot_pass_movement_gate():
    e,d,rows,ref,a=samples();pred=np.repeat(a[:,None,:],10,axis=1);m=e.metrics(pred,ref,a,rows)
    assert m['bounds']['all_passed'] and not m['eligible']
    assert m['moving_buckets']['passed']==0
    for b in m['buckets']:
        if b['phase'] in ('FREE','APPROACH'):
            assert b['moving']['projection_progress']==0 and b['moving']['full_mae_rad']==b['moving']['hold_full_mae_rad']

def test_one_bad_point_not_hidden_by_macro_and_no_tiny_sign_gate():
    e,d,rows,ref,a=samples();p=ref.copy();p[0,0,0]-=.005;m=e.metrics(p,ref,a,rows)
    assert m['scores']['worst_initial_first_max_abs_rad']>=.004999
    assert m['initial_free'][0]['max_abs_error_rad']>=.004999
    p=ref.copy();p[0,0,0]=-1e-9;m=e.metrics(p,ref,a,rows)
    assert m['eligible'] is True  # q1 label near zero: tiny sign is diagnostic only
    p=ref.copy();p[0,0,3]=-1e-12
    assert e.metrics(p,ref,a,rows)['bounds']['all_passed'] is False  # exact raw q4 bound stays strict

def test_missing_moving_support_unassessable_never_pass():
    e,d,rows,ref,a=samples();ref=ref.copy()
    mask=[i for i,r in enumerate(rows) if r['episode_id']==rows[0]['episode_id'] and r['phase']=='FREE']
    ref[mask]=a[mask,None,:]
    m=e.metrics(ref,ref,a,rows);assert not m['eligible']
    own=next(b for b in m['buckets'] if b['episode_id']==rows[0]['episode_id'] and b['phase']=='FREE')
    assert own['moving']['status']=='unassessable_no_moving_origin'

def test_selection_excludes_initialization_and_prioritizes_worst_start():
    s=module('selection')
    def candidate(step,worst,macro,ok=True):return dict(local_step=step,eligible=ok,scores=dict(
        worst_initial_first_max_abs_rad=worst,macro_first_mae_rad=macro,macro_full_mae_rad=macro))
    rows=[candidate(0,0,0),candidate(1217,.1,.0001),candidate(2434,.01,.1),candidate(3650,.001,.001,False)]
    assert s.pick_candidate(rows)['local_step']==2434
    with pytest.raises(ValueError):s.pick_candidate([candidate(0,0,0),candidate(1,0,0,False)])

def test_report_fresh_numeric_reconstruction_rejects_tamper(tmp_path,monkeypatch):
    e,d,rows,ref,a=samples()
    from sim.auscultation_pi05_v7.action_encoding import encode_chunk,decode_chunk
    norm=np.stack([encode_chunk(v,x,r['phase'],d.normalization['residual_actions']) for v,x,r in zip(ref,a,rows)])
    pred=np.stack([decode_chunk(v,x,r['phase'],d.normalization['residual_actions']) for v,x,r in zip(norm,a,rows)])
    binding=dict(schema=e.BINDING_SCHEMA,checkpoint_path=str(tmp_path/'model/step-000000001217'),local_step=1217,
        data_hash=d.data_hash,normalization_sha256=e.digest(d.normalization))
    monkeypatch.setattr(e,'verify_checkpoint_binding',lambda b:None)
    out=tmp_path/'diagnostics/local_001217'
    e.save_checkpoint_report(out,d,pred,norm,binding)
    e.verify_checkpoint_report(out/'report.json',d,binding)
    p=out/'report.json';value=json.loads(p.read_text());value['metrics']['scores']['macro_first_mae_rad']=999;p.write_text(json.dumps(value))
    with pytest.raises(ValueError):e.verify_checkpoint_report(p,d,binding)

def test_report_cannot_change_references_or_normalized_decode(tmp_path,monkeypatch):
    e,d,rows,ref,a=samples()
    from sim.auscultation_pi05_v7.action_encoding import encode_chunk
    norm=np.stack([encode_chunk(v,x,r['phase'],d.normalization['residual_actions']) for v,x,r in zip(ref,a,rows)])
    binding=dict(schema=e.BINDING_SCHEMA,checkpoint_path=str(tmp_path/'model/step-000000001217'),local_step=1217,
        data_hash=d.data_hash,normalization_sha256=e.digest(d.normalization))
    monkeypatch.setattr(e,'verify_checkpoint_binding',lambda b:None)
    with pytest.raises(ValueError,match='decode'):
        e.save_checkpoint_report(tmp_path/'diagnostics/local_001217',d,ref,norm+1,binding)
    assert not (tmp_path/'diagnostics/local_001217').exists()


def prepared_run(tmp_path,monkeypatch):
    """Real synthetic numeric reports, mocked qualified/checkpoint source boundary."""
    import sys
    from sim.auscultation_pi05_v7.action_encoding import encode_chunk,decode_chunk
    e,d,rows,ref,a=samples();s=module('selection');run=tmp_path/'run';run.mkdir()
    config=dict(training_run_path=str(run));sources={'synthetic_source':'f'*64};pilot={'scope':'synthetic'}
    context=dict(dataset=d,config=config,schedule=dict(candidate_steps=list(s.EXPECTED_STEPS),total_steps=14600),
        pilot_binding=pilot,pilot_source_sha256=sources,completed=True,candidate_paths=[])
    bindings={}
    def bound(path):return bindings[str(Path(path).resolve())]
    monkeypatch.setitem(sys.modules,'sim.auscultation_pi05_v7.coverage_pilot_train',SimpleNamespace(checkpoint_binding=bound))
    monkeypatch.setattr(e,'verify_checkpoint_binding',lambda b:None)
    monkeypatch.setattr(s,'source_hashes',lambda:sources)
    monkeypatch.setattr(s,'load_context',lambda path:context)
    for name,obj in [('configuration.json',dict(config=config)),('training_run.json',dict(training_complete=True,local_step=14600)),
                     ('diagnostic_schedule.json',dict(data_hash=d.data_hash,rows=rows))]:e.write_new(run/name,obj)
    norm=np.stack([encode_chunk(v,x,r['phase'],d.normalization['residual_actions']) for v,x,r in zip(ref,a,rows)])
    pred=np.stack([decode_chunk(v,x,r['phase'],d.normalization['residual_actions']) for v,x,r in zip(norm,a,rows)])
    for step in s.EXPECTED_STEPS:
        path=run/'model'/('step-%012d'%step);path.mkdir(parents=True)
        binding=dict(schema=e.BINDING_SCHEMA,checkpoint_path=str(path),local_step=step,optimizer_step=4000+step,
            data_hash=d.data_hash,normalization_sha256=e.digest(d.normalization),config_sha256=e.digest(config),
            pilot_binding=pilot,pilot_source_sha256=sources)
        bindings[str(path)]=binding;context['candidate_paths'].append(path)
        e.save_checkpoint_report(run/'diagnostics'/('local_%06d'%step),d,pred,norm,binding)
    return e,s,run,context,bindings


def test_full_budget_all13_reports_required_before_selection(tmp_path,monkeypatch):
    e,s,run,context,bindings=prepared_run(tmp_path,monkeypatch)
    result=s.build_selection(context)
    assert result['candidate_count']==13 and result['selected']['local_step']==1217
    assert result['closed_loop_validated'] is False and result['generalization_validated'] is False
    context['completed']=False
    with pytest.raises(ValueError,match='complete'):s.build_selection(context)
    context['completed']=True;context['candidate_paths']=context['candidate_paths'][:-1]
    with pytest.raises(ValueError,match='namespace'):s.build_selection(context)


def test_selector_rejects_updated_hash_with_raw_reference_or_metrics_tamper(tmp_path,monkeypatch):
    e,s,run,context,bindings=prepared_run(tmp_path,monkeypatch)
    path=run/'diagnostics/local_002434/report.json';report=e.read(path);report['metrics']['eligible']=False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='fresh numeric'):s.build_selection(context)
    report['metrics']['eligible']=True;path.write_text(json.dumps(report))
    numeric=path.with_name('predictions.npz')
    with np.load(numeric,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    arrays['reference']=arrays['reference'].copy();arrays['reference'][0,0,0]+=.001
    np.savez_compressed(numeric,**arrays);report['numeric_sha256']=e.file_sha(numeric);path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Raw source'):s.build_selection(context)


def test_selection_lock_requires_original_marker_and_replays_all_reports(tmp_path,monkeypatch):
    e,s,run,context,bindings=prepared_run(tmp_path,monkeypatch)
    result=s.select_run(run);path=Path(result['selection_lock'])
    locked=s.verify_selection_lock(path,result['selection_lock_sha256'])
    assert locked['selection']['selected']['local_step']==1217
    with pytest.raises(ValueError,match='locked'):s.select_run(run)
    (run/s.MARKER).write_text('{}')
    with pytest.raises(ValueError,match='marker'):s.verify_selection_lock(path)


def test_step0_only_feasible_leaves_rejection_evidence_and_blocks_resume(tmp_path,monkeypatch):
    e,s,run,context,bindings=prepared_run(tmp_path,monkeypatch)
    original=e.verify_checkpoint_report
    def verify(path,d,binding):
        report=copy.deepcopy(original(path,d,binding))
        if binding['local_step']>0:
            report['metrics']['eligible']=False;report['metrics']['moving_buckets']['passed']=0
        return report
    monkeypatch.setattr(e,'verify_checkpoint_report',verify)
    with pytest.raises(ValueError,match='No eligible'):s.select_run(run)
    assert (run/s.MARKER).is_file() and (run/'pilot_selection/selection.json').is_file()
    assert not (run/'pilot_selection/selection.lock.json').exists()
    assert e.read(run/'pilot_selection/selection.json')['selected'] is None
