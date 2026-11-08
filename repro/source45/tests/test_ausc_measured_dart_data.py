"""Synthetic contract tests; no actual assets, physics qualification or models."""
from copy import deepcopy
import importlib

import numpy as np
import pytest


def module():
    name='sim.ausc_policy.m2_measured_dart_data'
    assert importlib.util.find_spec(name) is not None, 'Separate measured DART bank missing'
    return importlib.import_module(name)


def manifest(m,*,source37=False):
    return dict(schema=m.MANIFEST_SCHEMA,profile=m.PROFILE_SOURCE37 if source37 else m.PROFILE,
        evidence_profile=m.teacher.DART37_EVIDENCE_PROFILE if source37 else m.teacher.DART_EVIDENCE_PROFILE,
        recorded_source=deepcopy(m.teacher.SOURCE37 if source37 else m.teacher.SOURCE34),base_dataset=deepcopy(m.BASE29),normalization=deepcopy(m.NORMALIZATION),
        precollection=dict(path='/synthetic/precollection.json',sha256='a'*64),
        attempts=[dict(request=r,nominal_source_config=None,episode=None,outcome='cpu_unavailable',
            reason='Synthetic unavailable metadata only',evidence={}) for r in m.planned_requests()])


@pytest.mark.parametrize('fault',['none','old_evidence','old_source','old_bank','unknown'])
def test_source37_manifest_requires_exact_explicit_triplet_before_payloads(tmp_path,monkeypatch,fault):
    m=module();assert hasattr(m,'PROFILE_SOURCE37'),'Separate Source37 bank profile missing'
    value=manifest(m,source37=True)
    if fault=='old_evidence':value['evidence_profile']=m.teacher.DART_EVIDENCE_PROFILE
    elif fault=='old_source':value['recorded_source']=deepcopy(m.teacher.SOURCE34)
    elif fault=='old_bank':value['profile']=m.PROFILE
    elif fault=='unknown':value['profile']='unknown'
    monkeypatch.setattr(m.rt,'read_bound',lambda *a,**k:pytest.fail('Profile metadata opened payload'))
    if fault=='none':assert len(m.validate_manifest(value,tmp_path))==20
    else:
        with pytest.raises(ValueError):m.validate_manifest(value,tmp_path)


def test_exact_twenty_requests_are_deterministic_independent_and_train_only():
    m=module();requests=m.planned_requests()
    expected=[(s,p,t) for s in ('p000758','p001031','p000833','p000961','p000742')
        for p in ('p000474_a001230_f0000','p000476_a001231_f0023') for t in ('R3','L3')]
    assert [(r['subject_id'],r['pose_id'],r['target_id']) for r in requests]==expected
    assert [r['seed'] for r in requests]==list(range(20261107,20261127))
    assert len({r['slot_id'] for r in requests})==20
    assert all(r['split']=='train' and r['amplitude_rad']==.015 and r['trigger_distance_m']==.035 for r in requests)
    requests[0]['subject_id']='p001120'
    assert m.planned_requests()[0]['subject_id']=='p000758'


@pytest.mark.parametrize('fault',['validation','fresh','old_body','pose','target','seed','amplitude','trigger',
    'duplicate','missing','order','unknown','source32','base','norm','episode_identity'])
def test_all_metadata_gates_precede_any_payload(tmp_path,monkeypatch,fault):
    m=module();value=manifest(m);last=value['attempts'][-1]
    if fault=='validation':last['request'].update(split='validation',subject_id='p000923')
    elif fault=='fresh':last['request'].update(split='test',subject_id='p000438')
    elif fault=='old_body':last['request']['subject_id']='p001120'
    elif fault in ('pose','target','seed','amplitude','trigger'):
        key={'pose':'pose_id','target':'target_id','seed':'seed','amplitude':'amplitude_rad','trigger':'trigger_distance_m'}[fault]
        last['request'][key]='changed'
    elif fault=='duplicate':value['attempts'][-1]=deepcopy(value['attempts'][0])
    elif fault=='missing':value['attempts'].pop()
    elif fault=='order':value['attempts'].reverse()
    elif fault=='unknown':value['schema']='unknown'
    elif fault=='source32':value['recorded_source']=deepcopy(m.teacher.SOURCE32)
    elif fault=='base':value['base_dataset']['sha256']='0'*64
    elif fault=='norm':value['normalization']['sha256']='0'*64
    else:last['episode']=dict(split='validation',subject_id='p000923')
    monkeypatch.setattr(m.rt,'read_bound',lambda *a,**k:pytest.fail('Payload read before all metadata'))
    with pytest.raises(ValueError):m.validate_manifest(value,tmp_path)


def qualified(m):
    return [dict(request=r,status='qualified',windows=2,infrastructure_contamination=False,
        native_reset_identity=dict(pid=i+1,sid=77,start_ticks=str(i+100))) for i,r in enumerate(m.planned_requests())]


def preparation(m,unavailable=()):
    return dict(schema='m2_measured_dart_precollection_cpu_plan_v1',frozen_utc='2026-11-07T16:00:00Z',
        frozen_before_first_native=True,slots=[dict(request=r,status='cpu_unavailable' if i in unavailable else 'prepared',
            reason='Finite original CPU plan unavailable' if i in unavailable else '',evidence={})
            for i,r in enumerate(m.planned_requests())])


@pytest.mark.parametrize('fault',['none','one_native_negative','too_few','reason','unfrozen','missing','relabel'])
def test_prefrozen_prepared_denominator_separates_cpu_unavailability_from_native_outcomes(fault):
    m=module();unavailable=(1,5,13);plan=preparation(m,unavailable);rows=qualified(m)
    for i in unavailable:rows[i].update(status='cpu_unavailable',windows=0)
    if fault=='one_native_negative':rows[0].update(status='physical_negative',windows=0)
    elif fault=='too_few':
        plan=preparation(m,(1,3,5,9,13))
        for i in (1,3,5,9,13):rows[i].update(status='cpu_unavailable',windows=0)
    elif fault=='reason':plan['slots'][1]['reason']=''
    elif fault=='unfrozen':plan['frozen_before_first_native']=False
    elif fault=='missing':plan['slots'].pop()
    elif fault=='relabel':rows[0].update(status='cpu_unavailable',windows=0)
    if fault in ('reason','unfrozen','missing','relabel'):
        with pytest.raises(ValueError):m.admission(rows,precollection=plan)
    else:
        result=m.admission(rows,precollection=plan)
        assert result['pilot_optimization_eligible'] is (fault=='none')
        assert result['prepared_episode_denominator']==(15 if fault=='too_few' else 17)
        assert result['fixed_attempt_denominator']==20


@pytest.mark.parametrize('fault',['none','last_config_validation','last_compact_validation'])
def test_every_config_and_compact_identity_is_checked_before_first_asset_preflight(tmp_path,monkeypatch,fault):
    m=module();value=manifest(m);plan=preparation(m);rows=[]
    for i,a in enumerate(value['attempts']):
        metadata={k:a['request'][k] for k in ('split','subject_id','target_id')}
        a.update(outcome='collected',nominal_source_config=dict(path='/nominal/'+str(i),sha256='a'*64),
            episode=dict(**metadata,pose_id=a['request']['pose_id'],episode_id=str(i),reset_id=str(i),run_id='owner',job_id=str(i),
                capture_root='/capture/'+str(i),config=dict(path='/config/'+str(i),sha256='a'*64),
                owner_terminal=dict(path='/owner',sha256='a'*64),run_result=dict(path='/run/'+str(i),sha256=str(i%10)*64),
                artifacts={'compact/manifest.json':dict(sha256='a'*64)}))
        rows.append(metadata)
    monkeypatch.setattr(m,'validate_manifest',lambda *a:value['attempts'])
    monkeypatch.setattr(m,'_precollection',lambda *a:plan)
    monkeypatch.setattr(m,'_base',lambda *a:None)
    def bound(ref,base):
        path=ref['path'];index=int(path.split('/')[2]) if path.startswith('/capture/') else int(path.split('/')[-1]) if path.split('/')[-1].isdigit() else 19
        meta=deepcopy(rows[index])
        if index==19 and ((fault=='last_config_validation' and path.startswith('/config/')) or
                (fault=='last_compact_validation' and path.startswith('/capture/'))):meta.update(split='validation',subject_id='p000923')
        if path.startswith('/config/') or path.startswith('/nominal/'):return dict(m2_given_contact_control=dict(metadata=meta)),None
        if path.startswith('/capture/'):return dict(metadata=meta),None
        return dict(started_utc='2026-11-07T17:00:00Z',config=dict(measured_dart_precollection=value['precollection'],jobs=[])),None
    monkeypatch.setattr(m.data,'_bound',bound)
    called=[]
    def config(*a):called.append(1);raise ValueError('Synthetic asset preflight intentionally stops')
    monkeypatch.setattr(m,'_config',config)
    if fault=='none':m._headers(value,tmp_path);assert called
    else:
        with pytest.raises(ValueError,match='TRAIN'):m._headers(value,tmp_path)
        assert not called


@pytest.mark.parametrize('fault',['none','one_physical','two_physical','infra','zero','duplicate','order','coverage','unknown'])
def test_fixed_denominator_threshold_coverage_and_infrastructure(fault):
    m=module();rows=qualified(m)
    if fault in ('one_physical','two_physical'):
        for i in range(1 if fault=='one_physical' else 2):rows[i].update(status='physical_negative',windows=0)
    elif fault=='infra':rows[-1].update(status='infrastructure_error',windows=0,infrastructure_contamination=True)
    elif fault=='zero':rows[-1]['windows']=0
    elif fault=='duplicate':rows[-1]['native_reset_identity']=deepcopy(rows[0]['native_reset_identity'])
    elif fault=='order':rows.reverse()
    elif fault=='coverage':
        for row in rows[-4:]:row.update(status='physical_negative',windows=0)
    elif fault=='unknown':rows[-1]['status']='unknown'
    result=m.admission(rows,precollection=preparation(m))
    assert result['pilot_optimization_eligible'] is (fault in ('none','one_physical'))
    assert result['fixed_attempt_denominator']==20 and result['minimum_qualified_episodes']==19
    assert result['bulk_expert_qualification'] is False and result['model_qualified'] is False
    if fault=='duplicate':assert result['no_infrastructure_contamination'] is False


def test_clean_masks_keep_real_postperturb_history_and_no_fictional_horizon(tmp_path):
    from test_ausc_given_contact_capture import new_inputs
    from test_ausc_given_contact_recovery import ledger
    m=module();_,records,audit,meta=new_inputs(tmp_path,n=34)
    samples=m.given.assemble_samples(records,audit,meta,allow_truncated=True)
    sidecar=ledger(audit,meta)
    result=m.roles.filter_recovery_samples(samples,audit,meta,sidecar);m.validate_samples(result)
    assert len(result['states'])>0 and (~result['action_is_pad']).sum(1)[-1]==1
    # H2 is observed history, so an observation from the perturbation is valid.
    assert result['derived_index']['windows'][0]['inclusive_native_ticks'][0]==119
    for row in sidecar['events'][11:]:row['kind']='excluded'
    short=m.roles.filter_recovery_samples(samples,audit,meta,sidecar);m.validate_samples(short)
    assert len(short['states'])==0  # Cross-boundary futures are never labels.
    short_audit=audit[:144];short_sidecar=ledger(short_audit,meta)
    prefix_records=deepcopy(records[:12]);prefix_records[-1].pop('q_command');prefix_records[-1]['record_kind']='terminal_observation'
    prefix=m.roles.filter_recovery_samples(m.given.assemble_samples(prefix_records,short_audit,meta,
        allow_truncated=True),short_audit,meta,short_sidecar)
    m.validate_samples(prefix)
    assert len(prefix['states'])>0 and not (~prefix['action_is_pad'])[:,19].any()
    for fault in ('origin','history','padding','velocity','clock','role','native_boundary','window_order'):
        bad=deepcopy(result)
        if fault=='origin':bad['origins'][0,0]+=.1
        elif fault=='history':bad['histories'][0,1,0]+=.1
        elif fault=='padding':bad['actions'][-1,-1,0]=.1
        elif fault=='velocity':bad['velocities'][0,0]+=.1
        elif fault=='clock':bad['observation_times'][0]+=.01
        elif fault=='role':bad['derived_index']['windows'][0]['actual_control_owner']='ACT_L2'
        elif fault=='native_boundary':bad['derived_index']['windows'][0]['raw_intent_native_ticks'][1]+=1
        else:bad['derived_index']['windows'].reverse()
        with pytest.raises(ValueError):m.validate_samples(bad)


def test_unavailable_twenty_close_terminal_ineligible_bank_without_labels_or_refit(tmp_path,monkeypatch):
    m=module();value=manifest(m);ref=m.rt.write_new(tmp_path/'manifest.json',value)
    monkeypatch.setattr(m,'_base',lambda *a:None)
    plan=preparation(m,range(20));value['precollection']=m.rt.write_new(tmp_path/'precollection.json',plan)
    ref=m.rt.write_new(tmp_path/'closed-manifest.json',value)
    monkeypatch.setattr(m,'read_episode',lambda *a,**k:pytest.fail('Unavailable capture opened'))
    monkeypatch.setattr(m.given,'fit_normalization',lambda *a,**k:pytest.fail('Base29 normalization refit'))
    result=m.rt.read_bound(m.export_dataset(ref,tmp_path/'bank'));meta=m.rt.read_bound(result['metadata'])
    assert result['complete'] is True and result['arrays'] is None and result['windows']==0
    assert result['optimization_admission']['pilot_optimization_eligible'] is False
    assert len(meta['attempt_ledger'])==20 and all(e['status']=='cpu_unavailable' for e in meta['attempt_ledger'])
    assert meta['normalization']==m.NORMALIZATION and meta['normalization_refitted'] is False
    assert meta['horizon_diagnostics']['20']['mean_absolute_action_rad'] is None
    assert not (tmp_path/'bank/normalization.json').exists()
    with pytest.raises(ValueError,match='eligible'):m.load_arrays(m.rt.reference(tmp_path/'bank/COMPLETE.json'))


def synthetic_export(tmp_path,monkeypatch,*,fault=None,source37=False):
    """Only the private evidence reader is replaced for artifact/accounting tests."""
    from test_ausc_given_contact_capture import new_inputs
    from test_ausc_given_contact_recovery import ledger
    m=module();_,records,audit,meta=new_inputs(tmp_path,n=14)
    sample=m.roles.filter_recovery_samples(m.given.assemble_samples(records,audit,meta,allow_truncated=True),audit,meta,ledger(audit,meta))
    value=manifest(m,source37=source37)
    for i,a in enumerate(value['attempts']):
        a.update(outcome='collected',episode=dict(**{k:a['request'][k] for k in ('split','subject_id','pose_id','target_id')},
            episode_id='synthetic-'+str(i),reset_id='reset-'+str(i),fixture_index=i),
            nominal_source_config=dict(path='/synthetic/config-'+str(i),sha256=str(i%10)*64))
    ref=m.rt.write_new(tmp_path/'manifest.json',value)
    plan=preparation(m)
    monkeypatch.setattr(m,'_headers',lambda *a:(value['attempts'],{},plan))
    monkeypatch.setattr(m,'validate_manifest',lambda *a:value['attempts'])
    monkeypatch.setattr(m.given,'fit_normalization',lambda *a,**k:pytest.fail('Normalization refit'))
    def read(attempt,base,manifest):
        i=attempt['episode']['fixture_index']
        if fault=='infra' and i==19:raise OSError('Synthetic worker interruption')
        if fault=='clock_negative' and i==19:raise m.PhysicalNegative('Synthetic independently observed cumulative L2 overrun')
        selected=deepcopy(sample)
        if fault=='zero' and i==19:
            for k in m.ARRAY_KEYS:selected[k]=selected[k][:0]
            selected['derived_index']['windows']=[]
        reset=dict(pid=i+1,pgid=77,sid=77,start_ticks=str(i+100))
        if fault=='reset' and i==19:reset=dict(pid=1,pgid=77,sid=77,start_ticks='100')
        return selected,dict(native_reset_identity=reset,evidence_profile=value['evidence_profile'],
            recorded_source=deepcopy(value['recorded_source'])),dict(meta,**{k:attempt['episode'][k] for k in ('split','subject_id','target_id','episode_id')})
    monkeypatch.setattr(m,'read_episode',read)
    return m,m.export_dataset(ref,tmp_path/'bank'),sample


@pytest.mark.parametrize('fault',[None,'infra'])
def test_source37_bank_keeps_full_twenty_admission_arrays_and_normalization(tmp_path,monkeypatch,fault):
    m=module();assert hasattr(m,'PROFILE_SOURCE37'),'Separate Source37 bank profile missing'
    m,ref,sample=synthetic_export(tmp_path,monkeypatch,fault=fault,source37=True)
    complete=m.rt.read_bound(ref);meta=m.rt.read_bound(complete['metadata'])
    assert complete['profile']==meta['profile']==m.PROFILE_SOURCE37
    assert len(meta['attempt_ledger'])==20 and meta['normalization']==m.NORMALIZATION and meta['normalization_refitted'] is False
    assert complete['optimization_admission']['pilot_optimization_eligible'] is (fault is None)
    if fault is None:
        arrays,_=m.load_arrays(ref)
        for key in m.ARRAY_KEYS:np.testing.assert_array_equal(arrays[key],np.concatenate([sample[key]]*20))
    else:
        with pytest.raises(ValueError,match='eligible'):m.load_arrays(ref)


@pytest.mark.parametrize('fault',['none','old_source','physical','owner','trace','clock'])
def test_source37_standalone_episode_uses_full_gates_without_whole_bank_admission(tmp_path,monkeypatch,fault):
    from test_ausc_given_contact_capture import new_inputs
    from test_ausc_given_contact_recovery import ledger
    m=module();assert hasattr(m,'PROFILE_SOURCE37'),'Separate Source37 bank profile missing'
    _,records,audit,meta=new_inputs(tmp_path,n=14);samples=m.given.assemble_samples(records,audit,meta,allow_truncated=True)
    request=m.planned_requests()[2];row=dict(**{k:request[k] for k in ('split','subject_id','pose_id','target_id')},
        episode_id=meta['episode_id'],run_id='synthetic-owner',run_result=dict(path=str(tmp_path/'COMPLETE.json'),sha256='a'*64),
        child_result=dict(path=str(tmp_path/'child.json'),sha256='b'*64),capture_audit=dict(codec='gzip'))
    attempt=dict(request=request,episode=row);config={};settings=dict(schema=m.recovery.LATE_SCHEMA)
    sidecar=ledger(audit,meta);sidecar.update(config_sha256=m.rt.digest(config),declaration=settings)
    run=dict(recovery_collection=dict(path=str(tmp_path/'fresh_expert/given_recovery.json'),sha256='c'*64))
    freeze={'synthetic_original_source37':True};calls=[]
    profile={k:v for k,v in manifest(m,source37=True).items() if k in ('profile','evidence_profile','recorded_source')}
    if fault=='old_source':profile['recorded_source']=deepcopy(m.teacher.SOURCE34)
    monkeypatch.setattr(m,'validate_manifest',lambda *a:pytest.fail('Standalone requires terminal whole-bank manifest'))
    monkeypatch.setattr(m,'admission',lambda *a,**k:pytest.fail('Standalone claims whole-bank eligibility'))
    monkeypatch.setattr(m,'_config',lambda *a:(config,dict(settings=settings,reference=object())))
    monkeypatch.setattr(m.data,'_metadata',lambda *a:(tmp_path,{},meta))
    monkeypatch.setattr(m.data,'_native_evidence',lambda *a,**k:(run,config,freeze))
    monkeypatch.setattr(m.data,'_bound',lambda *a:({'physical_pass':True},None))
    monkeypatch.setattr(m.rt,'read_bound',lambda *a:sidecar)
    monkeypatch.setattr(m.roles,'require_sidecar_location',lambda *a:None)
    monkeypatch.setattr(m.rt.shared,'_json',lambda *a:{})
    monkeypatch.setattr(m.data,'read_native_capture_audit',lambda *a,**k:audit)
    monkeypatch.setattr(m.data,'iter_training_records',lambda *a:iter(records))
    monkeypatch.setattr(m.teacher,'validate_source37_header',lambda *a,**k:calls.append('source37 header'))
    monkeypatch.setattr(m.teacher,'validate_source34_header',lambda *a,**k:pytest.fail('Source37 relabeled Source34'))
    def sources(*a,**k):
        calls.append('source37 owner/source')
        if fault=='owner':raise ValueError('original owner/source failure')
    def decode(*a,**k):
        calls.append('full source37 physical decoder')
        if fault=='physical':raise ValueError('original physical failure')
        return samples,{},meta
    def trace(*a,**k):
        calls.append('actual DART trace')
        if fault=='trace':raise ValueError('retained trajectory failure')
        return {}
    def clock(*a):
        calls.append('actual cumulative clock')
        if fault=='clock':raise m.PhysicalNegative('measured cumulative clock failure')
        return {}
    monkeypatch.setattr(m.teacher,'_source37_configuration_and_sources',sources)
    monkeypatch.setattr(m.teacher,'_source34_configuration_and_sources',lambda *a,**k:pytest.fail('Source37 used old source checks'))
    monkeypatch.setattr(m.teacher,'decode_source37_dart_capture',decode)
    monkeypatch.setattr(m.teacher,'decode_source34_dart_capture',lambda *a,**k:pytest.fail('Source37 used old decoder'))
    monkeypatch.setattr(m,'_validate_dart_trace',trace);monkeypatch.setattr(m,'_cumulative_l2',clock)
    if fault=='none':
        selected,receipt,_=m.read_episode(attempt,tmp_path,profile)
        assert len(selected['states'])>0 and receipt['bank_optimization_eligible'] is False
        assert receipt['qualification_scope']=='single_episode_only' and receipt['recorded_source']==m.teacher.SOURCE37
        assert calls==['source37 header','source37 owner/source','full source37 physical decoder','actual DART trace','actual cumulative clock']
    else:
        with pytest.raises(ValueError):m.read_episode(attempt,tmp_path,profile)


@pytest.mark.parametrize('fault',[None,'infra','zero','reset','clock_negative'])
def test_arrays_masks_order_denominator_and_reader_errors(tmp_path,monkeypatch,fault):
    m,ref,sample=synthetic_export(tmp_path,monkeypatch,fault=fault)
    complete=m.rt.read_bound(ref);meta=m.rt.read_bound(complete['metadata'])
    assert len(meta['attempt_ledger'])==20
    assert complete['optimization_admission']['pilot_optimization_eligible'] is (fault in (None,'clock_negative'))
    assert meta['horizon_diagnostics']['20']['valid_cells']==0
    assert meta['horizon_diagnostics']['20']['mean_absolute_action_rad'] is None
    if fault is None:
        arrays,loaded=m.load_arrays(ref)
        for key in m.ARRAY_KEYS:np.testing.assert_array_equal(arrays[key],np.concatenate([sample[key]]*20))
        index=m.rt.read_bound(loaded['index'])
        assert [e['sample_index'] for e in index['samples']]==list(range(complete['windows']))
    elif fault=='clock_negative':
        assert meta['attempt_ledger'][-1]['status']=='physical_negative'
        assert meta['attempt_ledger'][-1]['infrastructure_contamination'] is False
        arrays,_=m.load_arrays(ref);assert len(arrays['states'])==19*len(sample['states'])
    else:
        assert meta['attempt_ledger'][-1]['infrastructure_contamination'] is True
        with pytest.raises(ValueError,match='eligible'):m.load_arrays(ref)


@pytest.mark.parametrize('fault',['array_bytes','index_order','missing_key','unknown_schema','source_order'])
def test_corrupt_or_reordered_bank_never_opens_optimization_arrays(tmp_path,monkeypatch,fault):
    m,ref,_=synthetic_export(tmp_path,monkeypatch);complete=m.rt.read_bound(ref);meta=m.rt.read_bound(complete['metadata'])
    if fault=='array_bytes':
        with open(complete['arrays']['path'],'ab') as stream:stream.write(b'corruption')
    elif fault=='unknown_schema':
        complete['schema']='unknown';ref=m.rt.write_new(tmp_path/'bad-complete.json',complete)
    else:
        if fault=='index_order':
            index=m.rt.read_bound(meta['index']);index['samples'].reverse();index['order_sha256']=m.rt.digest(index['samples'])
            meta['index']=m.rt.write_new(tmp_path/'changed-index.json',index)
        elif fault=='source_order':meta['sources'].reverse()
        else:meta.pop('array_proofs')
        complete['metadata']=m.rt.write_new(tmp_path/'changed-meta.json',meta)
        ref=m.rt.write_new(tmp_path/'changed-complete.json',complete)
    with pytest.raises((ValueError,KeyError)):m.load_arrays(ref)


def dart_trace(tmp_path):
    from types import SimpleNamespace
    from test_ausc_given_contact_recovery import query
    from test_ausc_given_contact_handoff import FixtureTrajectory
    from sim.ausc_policy.m2_given_contact_intent import IntentDispatcher
    from sim.ausc_policy.retainer_contact import ALL_FORCE_GROUPS
    m=module();reference,pre=query(tmp_path);meta=pre['metadata'];h=m.rt.handoff
    settings=dict(schema=m.recovery.LATE_SCHEMA,seed=20261107,amplitude_rad=.015,trigger_distance_m=.035)
    rng=np.random.default_rng(settings['seed']);offset=rng.normal(size=7);offset*=.015/max(abs(offset))
    trajectory=FixtureTrajectory(np.zeros(7),dt=1/240.,**h.PLANNING_LIMITS)
    snapshot=h.input_snapshot(trajectory);dispatcher=IntentDispatcher(trajectory,reference_sha256=reference.sha256)
    audit=[];events=[];records=[];post=deepcopy(pre);post.update(physics_row=360,time_s=1.5,history_times=[1.45,1.5])
    for i in range(384):
        if i%12==0:
            kind='nominal_teacher' if i<240 else 'perturbation' if i<360 else 'recovery_teacher'
            target=np.zeros(7) if i<240 else offset if i<360 else np.full(7,.02)
            setter=dispatcher.issue(target,time_s=i/240.,physics_row=i,context=pre['context'],owner='TEACHER_L2',role='given_l2_motion')
            events.append(dict(intent_id=setter['intent_id'],setter_sha256=setter['setter_sha256'],physics_row_start=i,
                physics_row_end_exclusive=i+12,kind=kind,completed=True))
        raw=dispatcher.snapshot(time_s=i/240.,physics_row=i);raw['live_target_after_step_q']=deepcopy(raw['live_target_q'])
        audit.append(dict(physics_row=i,time_s=(i+1)/240.,q_command=trajectory.step().tolist(),
            actual_control_owner='TEACHER_L2',raw_intent=raw,measurements=dict(hall_N=0.,compression_m=0.,
                max_abs_joint_velocity_rad_s=0.,**{k+'_N':0. for k in ALL_FORCE_GROUPS})))
        if (i+1)%12==0:records.append(dict(observed_at_s=(i+1)/240.,state17=deepcopy(pre['history'][0]),measured_q=[0.]*7))
    pert_preview=dict(input_state=snapshot,input_sha256=m.rt.digest(snapshot),steps=120,
        commands_sha256=m.measured.hashlib.sha256(np.zeros((120,7)).tobytes()).hexdigest(),
        all_240hz_commands_verified=True,physics_executed=False)
    retained=deepcopy(snapshot);retained['fields']['target_position']=offset.tolist()
    proposal_preview=dict(pert_preview,input_state=retained,input_sha256=m.rt.digest(retained),steps=12,
        commands_sha256=m.measured.hashlib.sha256(np.zeros((12,7)).tobytes()).hexdigest())
    sidecar=dict(schema=m.recovery.SIDECAR_SCHEMA,completed=True,episode_id=meta['episode_id'],reference_sha256=reference.sha256,
        perturbations_executed=1,native_steps=len(audit),events=events,declaration=settings,queries=[
            dict(kind='perturbation_preview',query=pre,offset_rad=offset.tolist(),target_q=offset.tolist(),preview=pert_preview),
            dict(kind='recovery_proposal',query=post,query_sha256=m.rt.digest(post),route=[[0.]*7,[.02]*7],
                destination_q=[.02]*7,preview=proposal_preview,physics_executed=False)])
    pose=reference.pose_base.copy();pose[:3,3]+=[.034,0,0]
    return sidecar,audit,records,meta,reference,settings,FixtureTrajectory,SimpleNamespace(tcp=lambda q:pose)


@pytest.mark.parametrize('reported_total',[.01,0.,.02])
def test_dart_query_contact_sum_includes_actual_nonzero_retainer_force(tmp_path,reported_total):
    m=module();sidecar,audit,records,meta,reference,settings,factory,kin=dart_trace(tmp_path)
    for event in sidecar['queries']:
        query=event['query'];query['physical_contact_N']=reported_total
        audit[query['physics_row']-1]['measurements']['retainer_N']=.01
    sidecar['queries'][1]['query_sha256']=m.rt.digest(sidecar['queries'][1]['query'])
    if reported_total==.01:
        proof=m._validate_dart_trace(sidecar,audit,records,meta,reference,settings,trajectory_factory=factory,kin=kin)
        assert proof['measured_h2_capture_bound'] is True
    else:
        with pytest.raises(ValueError,match='scalar witness'):
            m._validate_dart_trace(sidecar,audit,records,meta,reference,settings,trajectory_factory=factory,kin=kin)


def test_default_dart_trigger_uses_genuine_canonical_contact_frame_and_records_geometry_proof(tmp_path,monkeypatch):
    from sim.auscultation_expert_layout_v1 import kinematics
    from sim.auscultation_expert_layout_v1.tool_frames import wrist_from_diaphragm
    m=module();sidecar,audit,records,meta,reference,settings,factory,_=dart_trace(tmp_path)
    # A purely synthetic wrist pose, with genuine approved and legacy TCP
    # transforms. No observed Linux joint vector or asset is used.
    wrist=reference.pose_base.copy()
    wrist[:3,3]-=wrist[:3,2]*(float(m.rt.geometry.contact_frame()[2,3])+.034)
    callbacks=[]
    class SyntheticWristKinematics(kinematics.OpenArmKinematics):
        def __init__(self,*,contact_frame=wrist_from_diaphragm):
            callbacks.append(contact_frame);self.contact_frame=contact_frame
        def fk(self,q):return wrist.copy()
    legacy=SyntheticWristKinematics();canonical=SyntheticWristKinematics(contact_frame=m.rt.geometry.contact_frame)
    legacy_distance=float(np.linalg.norm(legacy.tcp([0.]*7)[:3,3]-reference.pose_base[:3,3]))
    canonical_distance=float(np.linalg.norm(canonical.tcp([0.]*7)[:3,3]-reference.pose_base[:3,3]))
    assert canonical_distance<settings['trigger_distance_m']<legacy_distance
    callbacks.clear();monkeypatch.setattr(kinematics,'OpenArmKinematics',SyntheticWristKinematics)
    proof=m._validate_dart_trace(sidecar,audit,records,meta,reference,settings,trajectory_factory=factory)
    assert callbacks==[m.rt.geometry.contact_frame]
    assert proof['trigger_skin_target_distance_m']==pytest.approx(canonical_distance,abs=1e-12)
    assert proof['trigger_geometry']==dict(geometry_id=meta['geometry_id'],geometry_source_sha256=meta['geometry_source_sha256'],
        contact_frame='sim.ausc_policy.geometry_epoch.contact_frame',contact_frame_source=m.rt.reference(m.rt.geometry.__file__))
    with pytest.raises(ValueError,match='3D skin-target distance'):
        m._validate_dart_trace(sidecar,audit,records,meta,reference,settings,trajectory_factory=factory,kin=legacy)


@pytest.mark.parametrize('fault',['geometry_id','geometry_source_sha256','freshly_generated'])
def test_dart_trigger_rejects_wrong_geometry_before_fk_construction(tmp_path,monkeypatch,fault):
    from sim.auscultation_expert_layout_v1 import kinematics
    m=module();sidecar,audit,records,meta,reference,settings,factory,_=dart_trace(tmp_path)
    meta[fault]=False if fault=='freshly_generated' else 'retired-or-unbound'
    def forbidden(*a,**k):pytest.fail('Wrong geometry reached trigger FK')
    monkeypatch.setattr(kinematics,'OpenArmKinematics',forbidden)
    with pytest.raises(ValueError,match='geometry'):
        m._validate_dart_trace(sidecar,audit,records,meta,reference,settings,trajectory_factory=factory)


@pytest.mark.parametrize('fault',['none','query_replay','h2','measured','dq','query_hash','perturb_target','seed','retained',
    'command','unknown_role','preview_only','wrong_trigger','setter_route'])
def test_actual_dart_trace_authenticates_measured_queries_retained_state_and_real_setters(tmp_path,fault):
    m=module();args=list(dart_trace(tmp_path));sidecar,audit,records,meta,reference,settings,factory,kin=args
    if fault=='query_replay':sidecar['queries'][0]['kind']='measured_query_replay_terminal'
    elif fault=='h2':sidecar['queries'][1]['query']['history'][0][0]=.1
    elif fault=='measured':records[29]['measured_q'][0]=.1
    elif fault=='dq':audit[359]['measurements']['max_abs_joint_velocity_rad_s']=.1
    elif fault=='query_hash':sidecar['queries'][1]['query_sha256']='0'*64
    elif fault=='perturb_target':sidecar['queries'][0]['target_q'][0]+=.1
    elif fault=='seed':settings['seed']+=1
    elif fault=='retained':
        preview=sidecar['queries'][1]['preview'];preview['input_state']['fields']['current_velocity'][0]=.1
        preview['input_sha256']=m.rt.digest(preview['input_state'])
    elif fault=='command':audit[250]['q_command'][0]=.1
    elif fault=='unknown_role':sidecar['events'][30]['kind']='unknown'
    elif fault=='preview_only':sidecar['events'][30]['kind']='excluded'
    elif fault=='wrong_trigger':kin.tcp=lambda q:np.eye(4)
    elif fault=='setter_route':sidecar['queries'][1]['route'][1]=[.03]*7
    if fault=='none':
        proof=m._validate_dart_trace(sidecar,audit,records,meta,reference,settings,trajectory_factory=factory,kin=kin)
        assert proof['postperturb_physics_row']==360 and proof['actual_perturbation_steps']==120
    else:
        with pytest.raises((ValueError,KeyError)):m._validate_dart_trace(sidecar,audit,records,meta,reference,settings,
            trajectory_factory=factory,kin=kin)


def test_cumulative_l2_uses_actual_first_interval_through_parking_without_invented_clock():
    m=module();audit=[dict(actual_control_owner='PREPARATION') for _ in range(240)]
    audit += [dict(actual_control_owner='TEACHER_L2') for _ in range(28800)]
    run=dict(l2_handoffs=[dict(parking_end_physics_row=len(audit))])
    assert m._cumulative_l2(run,audit)['duration_s']==120.
    audit.append(dict(actual_control_owner='L2_STOP'));run['l2_handoffs'][0]['parking_end_physics_row']+=1
    with pytest.raises(ValueError,match='120'):m._cumulative_l2(run,audit)


@pytest.mark.parametrize('elapsed',[-.01,120.,120.01])
def test_source34_late_dart_inherits_the_original_cumulative_deadline_before_every_tick(elapsed):
    from types import SimpleNamespace
    m=module();controller=m.recovery.GivenContactRecoveryExpert.__new__(m.recovery.GivenContactRecoveryExpert)
    controller._l2_started_at=1.;controller._recovery_events=[]
    controller.recovery_binding=dict(settings=dict(schema=m.recovery.LATE_SCHEMA))
    controller.env=SimpleNamespace(state=lambda:dict(time_s=1.+elapsed))
    with pytest.raises(ValueError,match='120 second deadline'):controller._advance_tick('given_l2_motion')
    assert controller._l2_started_at==1.
