"""Buffered neutral pilot observer; no physical/native execution claims."""
from copy import deepcopy
import gzip
import importlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy import m2_given_contact_pilot_deployment as pilot
from sim.ausc_policy import m2_given_contact_runtime as rt
from sim.ausc_policy import fresh_expert


def module():
    name='sim.ausc_policy.m2_given_contact_pilot_native_audit'
    assert importlib.util.find_spec(name) is not None,'Buffered neutral pilot auditor missing'
    return importlib.import_module(name)


def setup(tmp_path,monkeypatch,*,original_error=None,metadata_extra=None):
    m=module();state=dict(time_s=0.,q=np.zeros(7),tcp_base=np.array([0.,0.,.1]),rotation_world=np.eye(3))
    events=[];controller=pilot.PilotGivenACT.__new__(pilot.PilotGivenACT)
    controller.env=SimpleNamespace(rows=[],base=np.eye(4),state=lambda:deepcopy(state),out=tmp_path,
        world=SimpleNamespace(stop=lambda:events.append('stop')),save=lambda name:events.append('bulk_save'))
    controller.metadata=dict(reference_sha256='a'*64,**(metadata_extra or {}));controller.reference=SimpleNamespace(sha256='a'*64)
    controller.episode_id='pilot-native';controller.trajectory=None;controller.control_owner='PREPARATION';controller.actor_phase=None
    controller.manager=SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='FREE')))
    controller._current_intent_audit=None;controller.control_rows=[{'q_command':[0.]*7}]
    controller.policy_receipt={'model':{'checkpoint':{'sha256':'b'*64}}}
    def original(command,subphase,active=False,recovery=False):
        events.append(('tick',list(command),active,recovery));state['time_s']+=1/240.
        controller.env.rows.append(dict(time_s=state['time_s'],q=state['q'].tolist(),q_command=list(command),
            phase='FREE',subphase=subphase,physical_violations=['original gate'] if original_error else [],
            diaphragm_N=4.,retainer_front_audit={'actual':True},target='L1'))
        if original_error:raise original_error
        return 'original-return'
    controller._tick=original;dep=dict(metadata=deepcopy(controller.metadata),model=deepcopy(controller.policy_receipt['model']),
        episode_id=controller.episode_id,source_config={'sha256':'c'*64})
    source=tmp_path/'source.py';source.write_text('# immutable observer binding\n')
    dep['source_inventory']={'source':rt.reference(source)}
    monkeypatch.setattr(pilot,'preflight',lambda *a,**k:dict(deployment=deepcopy(dep)))
    monkeypatch.setattr(pilot,'sources',lambda:{'source':rt.reference(source)})
    audit=m.attach(controller,{pilot.ROUTE_KEY:dep},tmp_path,adapter=pilot)
    return m,audit,controller,events,original,source


def test_original_tick_once_full_neutral_evidence_and_buffer_flush_240_then_close(tmp_path,monkeypatch):
    m,audit,c,events,original,source=setup(tmp_path,monkeypatch);metadata=deepcopy(c.metadata)
    before_source=source.read_bytes()
    for _ in range(239):assert c._tick([0.]*7,'staging',active=True)=='original-return'
    assert audit.rows==239 and audit.stream.batch_flushes==0 and audit.stream.pending_rows==239
    c._tick([0.]*7,'staging',recovery=True)
    assert audit.rows==240 and audit.stream.batch_flushes==1 and audit.stream.pending_rows==0
    c._tick([0.]*7,'staging')
    from sim.ausc_policy.m2_given_contact_evaluation import close_native_audit,NativeAudit
    assert type(audit)._append is NativeAudit._append
    receipt=close_native_audit(c)
    assert receipt['rows']==241 and receipt['complete'] is True and audit.stream.batch_flushes==2
    assert c._tick is original and m.installed(c) and source.read_bytes()==before_source and c.metadata==metadata
    assert len([row for row in events if isinstance(row,tuple)])==241
    with gzip.open(receipt['trace']['path'],'rt') as stream:rows=[json.loads(line) for line in stream]
    assert len(rows)==241 and rows[-1]['physics_row']==240
    assert rows[0]['q_command']==[0.]*7 and rows[0]['observed_before']['pose_base']
    assert rows[0]['native_row']['diaphragm_N']==4. and rows[0]['retainer_front_audit']=={'actual':True}
    assert receipt['metadata']==metadata and receipt['pilot_binding']['episode_id']=='pilot-native'


def test_append_failure_revokes_next_operation_and_preserves_negative_receipt(tmp_path,monkeypatch):
    m,audit,c,events,original,source=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(audit.stream,'write',lambda value:(_ for _ in ()).throw(OSError('disk append failure')))
    with pytest.raises(OSError,match='disk append'):c._tick([0.]*7,'staging')
    with pytest.raises(ValueError,match='revoked'):c._tick([0.]*7,'staging')
    assert len(events)==1 and len(c.env.rows)==1
    receipt=audit.close()
    assert receipt['complete'] is False and receipt['errors'] and receipt['trace']
    assert rt.read_bound(receipt['index'])['complete'] is False


def test_original_physical_error_identity_survives_and_failed_audit_is_not_complete(tmp_path,monkeypatch):
    error=RuntimeError('original physical gate')
    m,audit,c,events,original,source=setup(tmp_path,monkeypatch,original_error=error)
    with pytest.raises(RuntimeError) as caught:c._tick([0.]*7,'staging')
    assert caught.value is error
    receipt=audit.close();assert receipt['complete'] is False and receipt['rows']==1
    with gzip.open(receipt['trace']['path'],'rt') as stream:row=json.loads(stream.readline())
    assert row['tick_error']==repr(error) and row['physical_violations']==['original gate']


def test_explicit_installed_pilot_skips_bulk_duplicates_keeps_video_diagnostics_world_stop(tmp_path,monkeypatch):
    m,audit,c,events,original,source=setup(tmp_path,monkeypatch)
    c._tick([0.]*7,'staging');c.control_rows=[{'nonfinite_duplicate':float('nan')}]
    recorder=SimpleNamespace(close=lambda:events.append('video') or {'file':'actual10hz640x360.mp4'})
    video,errors=fresh_expert.finalize_probe(c.env,c,recorder,tmp_path)
    assert not errors and 'bulk_save' not in events and events[-2:]==['video','stop']
    assert video['file']=='actual10hz640x360.mp4'
    assert not (tmp_path/'fresh_expert_control.jsonl').exists()
    assert (tmp_path/'fresh_expert_controller_diagnostics.json').exists()
    assert c.neutral_native_audit['complete'] and m.trace_path(c)==audit.path


def test_final_close_failure_revokes_acceptance_but_does_not_skip_video_or_world_stop(tmp_path,monkeypatch):
    m,audit,c,events,original,source=setup(tmp_path,monkeypatch);c._tick([0.]*7,'staging')
    monkeypatch.setattr(audit.stream,'close',lambda:(_ for _ in ()).throw(OSError('final close failure')))
    video,errors=fresh_expert.finalize_probe(c.env,c,SimpleNamespace(close=lambda:events.append('video')),tmp_path)
    assert any(row['stage']=='neutral_native_audit_finalize' for row in errors)
    assert c.neutral_native_audit['complete'] is False and c.neutral_native_audit['errors']
    assert events[-2:]==['video','stop'] and 'bulk_save' not in events


def test_default_finalizer_keeps_bulk_behavior_and_false_marker_cannot_opt_out(tmp_path):
    m=module();calls=[]
    env=SimpleNamespace(rows=[],save=lambda name:calls.append('bulk'),world=SimpleNamespace(stop=lambda:calls.append('stop')))
    controller=SimpleNamespace(control_rows=[{'q_command':[0.]*7}],_pilot_light_native_audit=True)
    assert not m.installed(controller)
    _,errors=fresh_expert.finalize_probe(env,controller,None,tmp_path)
    assert not errors and calls==['bulk','stop'] and (tmp_path/'fresh_expert_control.jsonl').exists()


def test_bound_source_change_or_controller_metadata_mutation_revokes_audit(tmp_path,monkeypatch):
    m,audit,c,events,original,source=setup(tmp_path,monkeypatch)
    c.metadata['reference_sha256']='d'*64
    with pytest.raises(ValueError,match='binding'):c._tick([0.]*7,'staging')
    assert not events
    c.metadata['reference_sha256']='a'*64;source.write_text('# changed source\n')
    receipt=audit.close()
    assert receipt['complete'] is False and any(row['stage']=='binding' for row in receipt['errors'])


def closed_run(tmp_path,monkeypatch):
    m,audit,c,events,original,source=setup(tmp_path,monkeypatch,metadata_extra={'split':'train','subject_id':'p000449'})
    dep=dict(metadata=deepcopy(c.metadata),model=deepcopy(c.policy_receipt['model']),episode_id=c.episode_id,
        source_config={'sha256':'c'*64},source_inventory=pilot.sources())
    config={pilot.ROUTE_KEY:dep};c._tick([0.]*7,'staging')
    _,errors=fresh_expert.finalize_probe(c.env,c,None,tmp_path);assert not errors
    run=dict(success=True,physical_validation=True,errors=[],learned_policy=True,expert_l2_fallback=False,physics_steps=1,
        neutral_native_audit=c.neutral_native_audit,trace_file=str(audit.path),trace_sha256=rt.reference(audit.path)['sha256'],
        learned_policy_receipt=dict(model=dep['model'],metadata=c.metadata,source_inventory=pilot.sources()),
        cpu_child_exit={'normal_exit':False},points=[{'force_audit':{'passed':True}}])
    for name in ('fresh_expert_result.json','COMPLETE.json'):rt.write_new(tmp_path/name,run)
    return m,c,config,run


def test_closed_qualifier_calls_original_physical_gate_without_formal_population_or_test_route(tmp_path,monkeypatch):
    m,c,config,run=closed_run(tmp_path,monkeypatch);called=[]
    monkeypatch.setattr(m,'_ipc',lambda *a:{'normal_exit':True,'request_count':1})
    monkeypatch.setattr(rt.dataset,'_physical',lambda actual,events,meta:called.append((actual,events,meta)))
    receipt=rt.read_bound(m.qualify(tmp_path,config))
    assert receipt['passed'] is True and receipt['checks']['unchanged_physical_gate_passed'] is True
    assert len(called)==1 and called[0][0]==run and called[0][2]==c.metadata
    assert called[0][1][0]['native_row']['diaphragm_N']==4.
    assert receipt['checks']['physical']['maximum_force_N']=={'diaphragm_N':4.}
    assert receipt['formal_acceptance'] is False and receipt['raw_capture_complete'] is False


def test_qualifier_rejects_abnormal_actual_cpu_exit_before_physical_acceptance(tmp_path,monkeypatch):
    m,c,config,run=closed_run(tmp_path,monkeypatch)
    monkeypatch.setattr(rt.dataset,'_physical',lambda *a:pytest.fail('physical gate called before CPU exit evidence'))
    receipt=rt.read_bound(m.qualify(tmp_path,config))
    assert receipt['passed'] is False and 'normal CPU child exit' in receipt['error']
