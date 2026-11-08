"""Measured supplement contracts; all fixtures are explicitly synthetic."""
from copy import deepcopy
import gzip
import importlib
import json
import numpy as np
import pytest


def module():
    name='sim.ausc_policy.m2_measured_recovery_data'
    assert importlib.util.find_spec(name) is not None, 'Typed measured Native supplement missing'
    return importlib.import_module(name)


def test_teacher_capture_keeps_real_gzip_transport_and_hash_codec_binding(tmp_path):
    from sim.ausc_policy import m2_given_contact_dataset as d,m2_given_contact_runtime as rt
    module();p=tmp_path/'native_audit.jsonl.gz'
    with gzip.open(p,'wt') as f:f.write(json.dumps(dict(physics_row=0))+'\n')
    ref=rt.reference(p);transport=dict(reference=ref,codec='gzip')
    assert d.read_native_capture_audit(tmp_path,dict(native_audit_sha256=ref['sha256']),transport=transport)==[dict(physics_row=0)]
    with pytest.raises(ValueError,match='gzip'):d.read_native_capture_audit(tmp_path,dict(native_audit_sha256=ref['sha256']),transport=dict(transport,codec='xz'))
    with pytest.raises(ValueError):d.read_native_capture_audit(tmp_path,dict(native_audit_sha256='f'*64),transport=transport)


def test_native_decoder_fixes_exact_verifier_and_default_remains_legacy(monkeypatch):
    from sim.ausc_policy import m2_given_contact_dataset as d,m2_stop_residual_trajectory as stop,m2_given_contact_recovery as recovery
    module();calls=[]
    monkeypatch.setattr(stop,'verify_native_physical',lambda *a,**k:calls.append((a,k)))
    monkeypatch.setattr(d,'_physical',lambda *a:(_ for _ in ()).throw(ValueError('legacy rejects native adapter')))
    config=dict(stop_residual_trajectory=stop.CONFIG,m2_given_contact_recovery=dict(schema=recovery.QUERY_SCHEMA))
    d._capture_physical({},[],{},config,native_query_recovery=True)
    assert calls==[(({},[],{}),dict(config=config))]
    with pytest.raises(ValueError,match='legacy'):d._capture_physical({},[],{},config,native_query_recovery=False)
    with pytest.raises(ValueError):d._capture_physical({},[],{},dict(config,m2_given_contact_recovery=dict(schema='unknown')),native_query_recovery=True)


def qualified_ledger(m):
    return [dict(slot_id=slot,target_id='R3' if i<2 else 'L3',status='qualified',windows=2,
        identity=dict(config=dict(path='/original/'+str(i//2),sha256=str(i//2)*64),physics_row=24+12*i))
        for i,slot in enumerate(m.query.SLOT_IDS)]


@pytest.mark.parametrize('fault',['missing','duplicate','failed','empty','target','short','order'])
def test_all_four_distinct_attempts_and_both_targets_required_for_admission(fault):
    m=module();ledger=qualified_ledger(m)
    assert m.admission(ledger,profile='production_source30')['measured_native_optimization_eligible'] is True
    if fault=='missing':ledger[1]['status']='missing'
    elif fault=='duplicate':ledger[1]['identity']=deepcopy(ledger[0]['identity'])
    elif fault=='failed':ledger[1]['status']='failed'
    elif fault=='empty':ledger[1]['windows']=0
    elif fault=='target':ledger[2]['target_id']='R3'
    elif fault=='short':ledger.pop()
    else:ledger.reverse()
    assert m.admission(ledger,profile='production_source30')['measured_native_optimization_eligible'] is False


def test_synthetic_qualification_never_becomes_production_or_population_evidence():
    m=module();value=m.admission(qualified_ledger(m),profile='local_synthetic_fixture')
    assert value['measured_native_optimization_eligible'] is False
    assert value['population_generalization_evidence'] is False
    with pytest.raises(ValueError):m.admission([],profile='unknown')


def test_positive_labels_exclude_prefix_failed_stop_align_and_use_actual_masked_tail(tmp_path):
    from sim.ausc_policy import m2_given_contact_act as given,m2_given_contact_recovery_dataset as recovery
    from test_ausc_given_contact_capture import new_inputs
    from test_ausc_given_contact_recovery import ledger
    m=module();_,records,audit,meta=new_inputs(tmp_path,n=34);sidecar=ledger(audit,meta)
    samples=given.assemble_samples(records,audit,meta,allow_truncated=True)
    selected=recovery.filter_recovery_samples(samples,audit,meta,sidecar)
    m.validate_samples(selected)
    assert len(selected['states'])>5 and (~selected['action_is_pad']).sum(1)[-1]==1
    assert all(w['inclusive_native_ticks'][0]>=119 for w in selected['derived_index']['windows'])
    corrupt=deepcopy(selected);corrupt['actions'][-1,-1]=1.
    with pytest.raises(ValueError):m.validate_samples(corrupt)
    sidecar['events'][10]['kind']='excluded'
    excluded=recovery.filter_recovery_samples(samples,audit,meta,sidecar)
    assert len(excluded['states'])<len(selected['states'])


def test_nontrain_last_attempt_fails_before_any_native_capture_or_array_reads(tmp_path,monkeypatch):
    m=module();rows=[dict(slot_id=s,episode=None) for s in m.query.SLOT_IDS]
    rows[-1]['episode']=dict(split='validation',subject_id='p000923')
    ref=m.rt.write_new(tmp_path/'manifest.json',dict(schema=m.MANIFEST_SCHEMA,profile='production_source30',attempts=rows,
        prefix_producer={},base_dataset={},normalization={}))
    monkeypatch.setattr(m,'read_episode',lambda *a:pytest.fail('Capture opened before all TRAIN identities'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Arrays opened before identity gate'))
    with pytest.raises(ValueError,match='TRAIN'):m.export_dataset(ref,tmp_path/'out')


@pytest.mark.parametrize('fault',['none','inventory','camera','native_count','reference','home','confidence'])
def test_native_optin_still_uses_full_original_capture_fidelity_decoder(tmp_path,monkeypatch,fault):
    from test_ausc_given_contact_dataset import decoder_fixture
    from sim.ausc_policy import m2_given_contact_dataset as d,m2_stop_residual_trajectory as stop,m2_given_contact_recovery as r
    m=module();row,context,run=decoder_fixture(tmp_path,monkeypatch)
    row['capture_audit']=dict(reference=m.rt.reference(context[0]/'native_audit.jsonl.gz'),codec='gzip')
    config=dict(stop_residual_trajectory=stop.CONFIG,m2_given_contact_recovery=dict(schema=r.QUERY_SCHEMA))
    calls=[];monkeypatch.setattr(stop,'verify_native_physical',lambda *a,**k:calls.append((a,k)))
    monkeypatch.setattr(d,'_physical',lambda *a:pytest.fail('Typed native recovery entered legacy physical gate'))
    if fault=='inventory':row['artifacts'].pop('given_reference.json')
    elif fault=='camera':run['raw_multimodal_capture']['receipts']['camera']['frames']+=1
    elif fault=='native_count':run['physics_steps']+=1
    elif fault=='reference':context[2]['reference_sha256']='0'*64
    elif fault in ('home','confidence'):
        original=d.iter_training_records
        def changed(path):
            records=list(original(path))
            if fault=='home':records[-1]['measured_q']=np.full(7,.031)
            else:records[0]['target_confidence']=.5
            yield from records
        monkeypatch.setattr(d,'iter_training_records',changed)
    if fault=='none':
        samples,receipt,_=d.decode_native_query_recovery_capture(row,tmp_path,context,run,config,None)
        assert samples['action_is_pad'].dtype==bool and len(samples['states'])>0
        assert calls[0][1]['config']==config and receipt['capture_bytes']<=100_000_000
    else:
        with pytest.raises(ValueError):d.decode_native_query_recovery_capture(row,tmp_path,context,run,config,None)


def test_measured_batch_uses_only_original_stats_and_cannot_bypass_bank_admission(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_act as g
    from test_ausc_given_contact_capture import new_inputs
    from test_ausc_given_contact_recovery import ledger
    m=module();_,records,audit,meta=new_inputs(tmp_path,n=34)
    sample=m.roles.filter_recovery_samples(g.assemble_samples(records,audit,meta,allow_truncated=True),audit,meta,ledger(audit,meta))
    stats=g.fit_normalization(sample,split='train');norm=m.rt.write_new(tmp_path/'BASE29-normalization.json',stats)
    before=open(norm['path'],'rb').read();sample['normalization_reference']=norm
    monkeypatch.setattr(g,'fit_normalization',lambda *a,**k:pytest.fail('Measured supplement refit stats'))
    with pytest.raises(ValueError,match='admission'):m.pack_batch(sample,norm,[0])
    sample.update(bank_schema=m.SCHEMA,optimization_admission=m.admission(qualified_ledger(m),profile='production_source30'))
    got=m.pack_batch(sample,norm,[0,len(sample['states'])-1])
    expected=g.pack_batch(sample,stats,[0,len(sample['states'])-1],training=False)
    assert all(np.array_equal(got[k],expected[k]) for k in got) and open(norm['path'],'rb').read()==before
    with pytest.raises(ValueError):m.pack_batch(sample,dict(norm,sha256='f'*64),[0])


def test_missing_failed_and_duplicate_attempts_close_fixed_ineligible_artifact_without_refit(tmp_path,monkeypatch):
    m=module();attempts=[dict(slot_id=s,episode=None) for s in m.query.SLOT_IDS]
    producer=dict(slots=[dict(slot_id=s,target_id='R3' if i<2 else 'L3',physics_row=None,identity=None,source=None,
        status='duplicate' if i==1 else 'missing') for i,s in enumerate(m.query.SLOT_IDS)])
    manifest=m.rt.write_new(tmp_path/'MANIFEST.json',dict(schema=m.MANIFEST_SCHEMA,profile='local_synthetic_fixture',attempts=attempts,
        prefix_producer={},base_dataset={},normalization={}))
    monkeypatch.setattr(m,'_headers',lambda *a:(attempts,producer,{}))  # synthetic accounting only
    monkeypatch.setattr(m.given,'fit_normalization',lambda *a,**k:pytest.fail('Stats refitted'))
    monkeypatch.setattr(m,'read_episode',lambda *a:pytest.fail('Missing/duplicate attempt opened capture'))
    complete=m.rt.read_bound(m.export_dataset(manifest,tmp_path/'out'));meta=m.rt.read_bound(complete['metadata'])
    assert meta['fixed_denominator']==4 and len(meta['attempt_ledger'])==4 and complete['arrays'] is None
    assert complete['optimization_admission']['measured_native_optimization_eligible'] is False
    assert meta['normalization_refitted'] is False and not (tmp_path/'out/normalization.json').exists()
    assert [e['status'] for e in meta['attempt_ledger']]==['missing','duplicate','missing','missing']


@pytest.mark.parametrize('fault',['none','clock','history','delta','pose','dq','missing_anchor','source','remaining'])
def test_replay_terminal_binds_actual_captured_history_and_unchanged_source_cut(tmp_path,fault):
    from test_ausc_given_contact_capture import new_inputs
    from test_ausc_given_contact_recovery import ledger
    m=module();_,records,audit,meta=new_inputs(tmp_path,n=34)
    sample=m.roles.filter_recovery_samples(m.given.assemble_samples(records,audit,meta,allow_truncated=True),audit,meta,ledger(audit,meta))
    t=float(sample['observation_times'][0]);row=sample['derived_index']['windows'][0]['inclusive_native_ticks'][0]+1
    actual=dict(measured_q=sample['origins'][0].tolist(),measured_dq=[.01]*7,history=sample['histories'][0].tolist(),
        history_times=[t-.05,t],time_s=t,physics_row=row,metadata=meta,context=[meta['episode_id'],meta['target_id'],0,'APPROACH'],
        measured_pose_base=np.eye(4).tolist(),reference_sha256=meta['reference_sha256'])
    original=deepcopy(actual);original['measured_q']=[.2]*7;original['measured_dq']=[.02]*7
    prefix=dict(source_cut_query=original,l2_started_at_s=0.,l2_deadline_s=120.)
    terminal=dict(kind='measured_query_replay_terminal',query=actual,l2_started_at_s=0.,l2_deadline_s=120.,
        remaining_l2_budget_s=120.-t,source_cut_query_sha256=m.rt.digest(original),saved_state_injected=False,
        replay_minus_source_q=(sample['origins'][0]-.2).tolist(),replay_minus_source_dq=[-.01]*7,
        replay_minus_source_tcp_pose=np.zeros((4,4)).tolist())
    if fault=='clock':terminal['query']['physics_row']+=12
    elif fault=='history':terminal['query']['history'][0][0]+=.01
    elif fault=='delta':terminal['replay_minus_source_q'][0]=0.
    elif fault=='pose':terminal['replay_minus_source_tcp_pose'][0][3]=1.
    elif fault=='dq':terminal['replay_minus_source_dq'][0]=1.
    elif fault=='missing_anchor':sample['observation_times']+=1.
    elif fault=='source':terminal['source_cut_query_sha256']='f'*64
    elif fault=='remaining':terminal['remaining_l2_budget_s']=120.
    if fault=='none':m.validate_replay_terminal(terminal,prefix,sample,meta)
    else:
        with pytest.raises(ValueError):m.validate_replay_terminal(terminal,prefix,sample,meta)


def test_native_decoder_seals_real_takeover_h2_independent_of_label_admission(tmp_path,monkeypatch):
    from test_ausc_given_contact_dataset import decoder_fixture
    from sim.ausc_policy import m2_given_contact_dataset as d,m2_given_contact_recovery as r
    m=module();row,context,run=decoder_fixture(tmp_path,monkeypatch)
    row['capture_audit']=dict(reference=m.rt.reference(context[0]/'native_audit.jsonl.gz'),codec='gzip')
    config=dict(stop_residual_trajectory=m.residual.CONFIG,m2_given_contact_recovery=dict(schema=r.QUERY_SCHEMA))
    monkeypatch.setattr(m.residual,'verify_native_physical',lambda *a,**k:None)
    _,receipt,_=d.decode_native_query_recovery_capture(row,tmp_path,context,run,config,None,replay_cut_row=24)
    proof=receipt['replay_cut_observations']
    assert proof['physics_row']==24 and proof['history_times']==pytest.approx([.05,.1])
    records=list(d.iter_training_records(context[0]/'compact'))
    np.testing.assert_array_equal(proof['history'],[records[0]['state17'],records[1]['state17']])
    np.testing.assert_array_equal(proof['measured_q'],records[1]['measured_q'])


def test_v2_admission_is_explicit_versioned_and_still_four_distinct():
    m=module()
    assert hasattr(m,'SCHEMA_V2'),'Explicit measured bank V2 schema missing'
    value=m.admission(qualified_ledger(m),profile=m.PROFILE_V2)
    assert value['schema']==m.ADMISSION_SCHEMA_V2 and value['measured_native_optimization_eligible'] is True
    ledger=qualified_ledger(m);ledger[-1]['identity']=deepcopy(ledger[0]['identity'])
    assert m.admission(ledger,profile=m.PROFILE_V2)['measured_native_optimization_eligible'] is False
    assert m.admission(qualified_ledger(m),profile='local_synthetic_fixture_v2')['measured_native_optimization_eligible'] is False
    assert m.admission(qualified_ledger(m),profile='production_source30')['schema']==m.ADMISSION_SCHEMA


@pytest.mark.parametrize('fault',['no_profile','old_manifest','unknown_source'])
def test_v2_explicit_manifest_source_gates_precede_any_payload(tmp_path,monkeypatch,fault):
    m=module()
    assert hasattr(m,'MANIFEST_SCHEMA_V2'),'Explicit V2 manifest missing'
    from sim.ausc_policy import m2_teacher_stop_reader as teacher
    value=dict(schema=m.MANIFEST_SCHEMA_V2,profile=m.PROFILE_V2,evidence_profile=teacher.EVIDENCE_PROFILE,
        recorded_source=deepcopy(teacher.SOURCE32),attempts=[dict(slot_id=s,episode=None) for s in m.query.SLOT_IDS],
        prefix_producer={},base_dataset={},normalization={})
    if fault=='no_profile':del value['evidence_profile']
    elif fault=='old_manifest':value['schema']=m.MANIFEST_SCHEMA
    else:value['recorded_source']['root']='/unknown'
    monkeypatch.setattr(m,'_base',lambda *a:pytest.fail('Base/payload opened before explicit teacher source gate'))
    monkeypatch.setattr(m,'_prefix_producer',lambda *a:pytest.fail('Prefix opened before explicit teacher source gate'))
    with pytest.raises(ValueError):m._headers(value,tmp_path)


def test_teacher_decoder_is_fixed_explicit_and_v1_has_no_format_fallback(tmp_path,monkeypatch):
    from test_ausc_given_contact_dataset import decoder_fixture
    from sim.ausc_policy import m2_teacher_stop_reader as teacher
    m=module();d=m.data;row,context,run=decoder_fixture(tmp_path,monkeypatch)
    assert hasattr(d,'decode_teacher_query_recovery_capture'),'Explicit teacher compact decoder missing'
    row['capture_audit']=dict(reference=m.rt.reference(context[0]/'native_audit.jsonl.gz'),codec='gzip')
    calls=[]
    monkeypatch.setattr(teacher,'validate_source32_header',lambda *a,**k:calls.append('source'))
    monkeypatch.setattr(teacher,'verify_teacher_physical',lambda *a,**k:calls.append('teacher_physical'))
    monkeypatch.setattr(m.residual,'verify_native_physical',lambda *a,**k:(_ for _ in ()).throw(ValueError('neutral strict')))
    config=dict(stop_residual_trajectory=m.residual.CONFIG,m2_given_contact_recovery=dict(schema=m.recovery.QUERY_SCHEMA))
    teacher_context=dict(evidence_profile=teacher.EVIDENCE_PROFILE,recorded_source=teacher.SOURCE32,source_freeze={})
    samples,receipt,_=d.decode_teacher_query_recovery_capture(row,tmp_path,context,run,config,{},teacher_context=teacher_context,replay_cut_row=24)
    proof=receipt['replay_cut_observations'];records=list(d.iter_training_records(context[0]/'compact'))
    assert proof['physics_row']==24 and proof['history_times']==pytest.approx([.05,.1])
    np.testing.assert_array_equal(proof['history'],[records[0]['state17'],records[1]['state17']])
    np.testing.assert_array_equal(proof['measured_q'],records[1]['measured_q'])
    assert len(samples['states'])>0 and 'teacher_physical' in calls
    with pytest.raises(ValueError,match='neutral strict'):d.decode_native_query_recovery_capture(row,tmp_path,context,run,config,{})


def test_v2_synthetic_export_retains_all_four_masks_order_and_norm_without_production_admission(tmp_path,monkeypatch):
    from test_ausc_given_contact_capture import new_inputs
    from test_ausc_given_contact_recovery import ledger
    m=module();_,records,audit,meta=new_inputs(tmp_path,n=34)
    selected=m.roles.filter_recovery_samples(m.given.assemble_samples(records,audit,meta,allow_truncated=True),audit,meta,ledger(audit,meta))
    stats=m.given.fit_normalization(selected,split='train');norm=m.rt.write_new(tmp_path/'unchanged-norm.json',stats)
    norm_bytes=open(norm['path'],'rb').read();producer={'slots':qualified_ledger(m)};attempts=[]
    for i,slot in enumerate(m.query.SLOT_IDS):
        attempts.append(dict(slot_id=slot,episode=dict(split='train',subject_id='p001120',pose_id='p000474_a001230_f0000',
            target_id='R3' if i<2 else 'L3',episode_id='synthetic-'+str(i),fixture_reset=i)))
        producer['slots'][i]['status']='selected'
    manifest=m.rt.write_new(tmp_path/'fixture-v2-manifest.json',dict(schema=m.MANIFEST_SCHEMA_V2,profile='local_synthetic_fixture_v2',
        attempts=attempts,normalization=norm,base_dataset=m.BASE29,prefix_producer={},
        evidence_profile='local_synthetic_fixture_v2',recorded_source={'fixture':True}))
    monkeypatch.setattr(m,'_headers',lambda *a:(attempts,producer,{})) # Accounting/arrays only; production headers reject fixture.
    calls=[]
    def read(row,*a,**kw):
        i=row['fixture_reset'];calls.append(i)
        receipt=dict(native_reset_identity=dict(pid=i+1,sid=i+1,start_ticks=i+2))
        return deepcopy(selected),receipt,dict(meta,episode_id=row['episode_id'],target_id=row['target_id'])
    monkeypatch.setattr(m,'read_teacher_episode',read)
    monkeypatch.setattr(m,'read_episode',lambda *a:pytest.fail('V2 used V1 decoder'))
    monkeypatch.setattr(m.given,'fit_normalization',lambda *a,**k:pytest.fail('Normalization refit'))
    ref=m.export_dataset(manifest,tmp_path/'new-synthetic-v2');complete=m.rt.read_bound(ref);metadata=m.rt.read_bound(complete['metadata'])
    assert calls==list(range(4)) and complete['schema']==m.SCHEMA_V2
    assert metadata['normalization']==norm and open(norm['path'],'rb').read()==norm_bytes
    assert len(metadata['attempt_ledger'])==len(metadata['sources'])==4
    assert complete['optimization_admission']['measured_native_optimization_eligible'] is False
    with np.load(complete['arrays']['path'],allow_pickle=False) as archive:
        for key in m.ARRAY_KEYS:np.testing.assert_array_equal(archive[key],np.concatenate([selected[key]]*4))
    index=m.rt.read_bound(metadata['index'])
    assert [w['sample_index'] for w in index['samples']]==list(range(len(selected['states'])*4))
    assert all(w['window']['label_role']=='actual_clean_recovery_teacher_raw_intents' for w in index['samples'])
    with pytest.raises(ValueError):m.metadata(ref) # A synthetic bank cannot enter the production loader.


def test_direct_v2_reader_rejects_nontrain_before_source_or_capture_metadata(tmp_path,monkeypatch):
    m=module();row=dict(split='validation',subject_id='p000923',run_id='act9-query-recovery-20261107-v1')
    monkeypatch.setattr(m,'_teacher_context',lambda *a,**k:pytest.fail('Non-TRAIN opened source metadata'))
    with pytest.raises(ValueError,match='TRAIN'):
        m.read_teacher_episode(row,tmp_path,evidence_profile='local_synthetic_fixture_v2',recorded_source={})


@pytest.mark.parametrize('route',['none','mixed','unknown_teacher'])
def test_replay_cut_never_enables_unknown_or_mixed_physical_routes(tmp_path,monkeypatch,route):
    from test_ausc_given_contact_dataset import decoder_fixture
    from sim.ausc_policy import m2_teacher_stop_reader as teacher
    m=module();d=m.data;row,context,run=decoder_fixture(tmp_path,monkeypatch)
    row['capture_audit']=dict(reference=m.rt.reference(context[0]/'native_audit.jsonl.gz'),codec='gzip')
    config=dict(stop_residual_trajectory=m.residual.CONFIG,m2_given_contact_recovery=dict(schema=m.recovery.QUERY_SCHEMA))
    if route=='unknown_teacher':
        monkeypatch.setattr(d,'_scene_integrity',lambda *a:pytest.fail('Unknown profile opened capture payload'))
        with pytest.raises(ValueError):d.decode_teacher_query_recovery_capture(row,tmp_path,context,run,config,{},
            teacher_context=dict(evidence_profile='unknown',recorded_source={},source_freeze={}),replay_cut_row=24)
    else:
        kwargs=dict(_replay_cut_row=24)
        if route=='mixed':kwargs.update(_native_query_recovery=True,_teacher_stop_context={})
        with pytest.raises(ValueError):d._decode_capture(row,tmp_path,context,run,config,{},**kwargs)
