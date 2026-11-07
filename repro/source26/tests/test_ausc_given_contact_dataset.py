"""CPU contracts only. These fixtures never establish native qualification."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from sim.ausc_policy import m2_act as shared, m2_given_contact_act as given
from sim.ausc_policy import m2_given_contact_pose as pose
from sim.ausc_policy.m2_given_contact_handoff import POLICY_ID
from test_ausc_given_contact_capture import new_inputs


def module():
    name = 'sim.ausc_policy.m2_given_contact_dataset'
    assert importlib.util.find_spec(name) is not None, 'Task5A given dataset route is missing'
    return importlib.import_module(name)


def test_old_handoff_policy_capture_refuses_before_payload(tmp_path,monkeypatch):
    m=module();row,*_=metadata_row(tmp_path)
    rewrite_meta(row,lambda x:x['metadata'].pop('handoff_policy_id'))
    monkeypatch.setattr(m,'_read_payload',lambda *a:pytest.fail('Old handoff reached capture payload'))
    with pytest.raises(ValueError,match='handoff policy'):m.read_episode(row,tmp_path)


@pytest.mark.parametrize('old_policy',['m2_given_interior_parking_handoff_v1','m2_given_interior_parking_handoff_v2'])
def test_new_stop_profile_never_promotes_historical_policy_metadata(tmp_path,monkeypatch,old_policy):
    m=module();row,*_=metadata_row(tmp_path)
    rewrite_meta(row,lambda value:value['metadata'].update(handoff_policy_id=old_policy))
    monkeypatch.setattr(m,'_read_payload',lambda *a:pytest.fail('Historical policy reached new STOP reader payload'))
    with pytest.raises(ValueError,match='handoff policy'):m.read_episode(row,tmp_path)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
    return dict(path=str(path), sha256=shared.sha256(path))


def metadata_row(tmp_path, *, name='one', split='train', subject='p000439'):
    folder = tmp_path/name
    folder.mkdir()
    ref, records, audit, meta = new_inputs(folder, 24)
    if subject!='p000439' or split!='train':
        snapshot=ref.snapshot();patient=pose.bound_json(snapshot['sources']['patient_receipt'])
        patient.update(identity=subject,split=split)
        saved=save(Path(snapshot['sources']['patient_receipt']['file']),patient)
        patient_ref=dict(file=saved['path'],sha256=saved['sha256'])
        plan=pose.bound_json(snapshot['sources']['fresh_plan'])
        plan['patient_provenance'].update(identity=subject,binding=patient_ref)
        saved=save(Path(snapshot['sources']['fresh_plan']['file']),plan)
        plan_ref=dict(file=saved['path'],sha256=saved['sha256'])
        target=patient['targets'][0]
        ref=pose.build_given_reference(dict(target,manifest_ref=patient_ref),plan_ref,patient_ref,np.eye(4),0.)
        meta=pose.route_metadata(meta,ref)
    meta.update(episode_id=name, subject_id=subject, split=split, source_kind='isaac_live_physics', replay=False,handoff_policy_id=POLICY_ID)
    meta.update(patient_asset_sha256=pose.bound_json(ref.snapshot()['sources']['patient_receipt'])['asset_sha256'],
        installed_patient_receipt_sha256=ref.snapshot()['sources']['patient_receipt']['sha256'])
    root = folder/'capture'
    path = root/'compact/manifest.json'
    from sim.ausc_policy.training_capture import GIVEN_ADAPTIVE_SCHEMA
    save(path, dict(schema=GIVEN_ADAPTIVE_SCHEMA, metadata=meta,
                    observation_contract=pose.observation_contract(), action_semantics_id=given.ACTION_SEMANTICS_ID))
    row = dict(episode_id=name, subject_id=subject, split=split, reset_id='reset-'+name,
               run_id='owner-'+name, job_id='job-'+name, target_id=meta['target_id'],capture_root=str(root),
               artifacts={'compact/manifest.json':dict(bytes=path.stat().st_size, sha256=shared.sha256(path))},
               run_result=dict(path=str(folder/'native/COMPLETE.json'), sha256=hashlib.sha256(name.encode()).hexdigest()))
    row['config']=save(folder/'config.json',dict(m2_given_contact_control=dict(schema=pose.TASK_ID,metadata=meta)))
    return row, meta, records, audit


def manifest(tmp_path, rows):
    return save(tmp_path/'dataset.json', dict(schema=module().MANIFEST_SCHEMA, task_id=pose.TASK_ID,
        action_semantics_id=given.ACTION_SEMANTICS_ID,handoff_policy_id=POLICY_ID, episodes=rows))


def rewrite_meta(row, transform):
    path = Path(row['capture_root'])/'compact/manifest.json'
    value = json.loads(path.read_text()); transform(value)
    save(path, value)
    row['artifacts']['compact/manifest.json'] = dict(bytes=path.stat().st_size, sha256=shared.sha256(path))


@pytest.mark.parametrize('defect', ['test', 'quarantine', 'old_schema', 'old_observation', 'cpu', 'changed_reference'])
def test_all_metadata_preflight_precedes_any_capture_payload(tmp_path, monkeypatch, defect):
    m = module()
    one, *_ = metadata_row(tmp_path, name='first')
    two, *_ = metadata_row(tmp_path, name='last')
    if defect == 'test': two.update(split='test', subject_id='p000438')
    elif defect == 'quarantine': two['subject_id'] = 'p000452'
    elif defect == 'old_schema': rewrite_meta(two, lambda x:x.update(schema='m2_compact_single_target_rgbd256x144_state17_hall20_v4'))
    elif defect == 'old_observation': rewrite_meta(two, lambda x:x['metadata'].update(observation_contract=shared.contract.observation_contract()))
    elif defect == 'cpu': rewrite_meta(two, lambda x:x['metadata'].update(source_kind='cpu_synthetic_test'))
    else: rewrite_meta(two, lambda x:x['metadata'].update(reference_sha256='0'*64))
    ref = manifest(tmp_path, [one,two])
    monkeypatch.setattr(m, '_read_payload', lambda *a:pytest.fail('Payload opened before ALL metadata passed'))
    with pytest.raises(ValueError): m.prepare_dataset(ref['path'], ref['sha256'])


@pytest.mark.parametrize('key', ['episode_id', 'reset_id', 'run_result', 'capture_root'])
def test_duplicate_episode_reset_receipt_or_source_is_rejected_before_payload(tmp_path, monkeypatch, key):
    m = module(); one,*_ = metadata_row(tmp_path, name='first'); two,*_ = metadata_row(tmp_path, name='second')
    two[key] = deepcopy(one[key])
    ref = manifest(tmp_path, [one,two])
    monkeypatch.setattr(m, '_read_payload', lambda *a:pytest.fail('Duplicate payload opened'))
    with pytest.raises(ValueError): m.prepare_dataset(ref['path'], ref['sha256'])


def native_metadata(tmp_path, row, meta):
    """Receipt layout of isaac_batch; fabricated solely to test refusals."""
    folder = tmp_path/'native'; job = folder/'output'/row['job_id']
    source = tmp_path/'archive/m2_given_contact_handoff.py'; source.parent.mkdir(); source.write_text('frozen CPU source fixture\n')
    commit = tmp_path/'archive/CODE_COMMIT'; commit.write_text('a'*40+'\n')
    source_manifest = save(tmp_path/'archive/SOURCE_MANIFEST.json', [dict(file=source.name,sha256=shared.sha256(source))])
    freeze = save(tmp_path/'freeze.json', dict(schema=module().SOURCE_FREEZE_SCHEMA, code_commit='a'*40,
        code_commit_file=dict(path=str(commit),sha256=shared.sha256(commit)), source_manifest=source_manifest,
        files={str(source):dict(path=str(source),sha256=shared.sha256(source),bytes=source.stat().st_size)}))
    row['source_freeze'] = freeze
    from test_ausc_policy_retainer_witness import contract_fixture
    (tmp_path/'witness').mkdir()
    native_contract,_=contract_fixture(tmp_path/'witness')
    meta['native_retainer_audit']=native_contract
    rewrite_meta(row,lambda x:x.update(metadata=meta))
    ref = pose.require_route(meta)
    binding = dict(reference_sha256=ref.sha256, bound_at_s=0., physics_row=0)
    reference = dict(reference=ref.snapshot(), reference_sha256=ref.sha256, availability_binding=binding,
                     task_id=pose.TASK_ID)
    config = dict(experiment='geometry_qualification', base='/workspace/wam', isaac_python='/native/bin/python',
        time_limit_seconds=300, m2_given_contact_control=dict(schema=pose.TASK_ID,metadata=meta,handoff_policy_id=POLICY_ID))
    row['config'] = save(tmp_path/'config.json', config)
    row['owner_terminal'] = save(folder/'result.json', dict(run_id=row['run_id'],code_commit='a'*40,
        completed=True,gpu_released=True,ended_utc='2026-10-27T01:00:00Z',config=dict(experiment='isaac_batch',
        base='/workspace/wam',parallelism=1,jobs=[dict(id=row['job_id'],config=row['config']['path'],sha256=row['config']['sha256'])]),
        stage=dict(passed=True,exit_code=0,errors=[],signals=[],process_group_drained=True,
                   process_identity=dict(pgid=11,sid=11,members={'12':'123'}))))
    run = dict(episode_id=row['episode_id'],native_retainer_audit=native_contract,handoff_policy_id=POLICY_ID,
        task_id=pose.TASK_ID,target_source=pose.TARGET_SOURCE,action_semantics_id=given.ACTION_SEMANTICS_ID,
        observation_contract=pose.observation_contract(),given_reference=reference,geometry_id=meta['geometry_id'],
        geometry_source_sha256=meta['geometry_source_sha256'],success=True,physical_validation=True,
        controller_motion_complete=True,learned_policy=False,errors=[],compact_capture_complete=True,
        executed_controller_class='sim.ausc_policy.m2_given_contact_controller.GivenContactExpert',
        runtime_sources_sha256={str(source):shared.sha256(source)},
        runtime_shutdown=dict(app_close_returned=True,fast_shutdown=False,
            interpreter_exit_policy='success_only_os_exit_after_native_close_and_fsynced_receipt'))
    from sim.ausc_policy.m2_given_contact_controller import GIVEN_TRAJECTORY_LIMITS,GIVEN_VELOCITY_NUMERICAL_RESERVE_RAD_S,GIVEN_JERK_NUMERICAL_RESERVE_RAD_S3
    run.update(raw_target_interpolation_limits=GIVEN_TRAJECTORY_LIMITS,
        planning_velocity_numerical_reserve_rad_s=GIVEN_VELOCITY_NUMERICAL_RESERVE_RAD_S,
        planning_jerk_numerical_reserve_rad_s3=GIVEN_JERK_NUMERICAL_RESERVE_RAD_S3)
    row['run_result'] = save(job/'COMPLETE.json',run)
    row['child_result'] = save(folder/'output'/(row['job_id']+'.result.json'), dict(id=row['job_id'],
        config_sha256=row['config']['sha256'],output=str(job),exit_code=0,worker_complete=True,physical_pass=True,
        result_sha256=row['run_result']['sha256'],pid=12,identity=dict(pid=12,pgid=11,sid=11,start_ticks='123')))
    return run


def mutate_ref(row, key, change):
    ref = row[key]; value = json.loads(Path(ref['path']).read_text()); change(value)
    row[key] = save(Path(ref['path']),value)


@pytest.mark.parametrize('defect', ['no_owner','failed_owner','undrained','gpu_held','child_exit','wrong_owner',
    'wrong_job','changed_config','no_close','late_reference','reference_receipt','changed_source','cpu_witness'])
def test_native_evidence_refuses_missing_failed_or_misbound_receipts(tmp_path, monkeypatch, defect):
    m=module(); row,meta,*_ = metadata_row(tmp_path); native_metadata(tmp_path,row,meta)
    monkeypatch.setattr(m.witness,'load_contract',lambda _:dict(cpu_diagnostic_only=False))
    if defect=='no_owner': row.pop('owner_terminal')
    elif defect=='failed_owner': mutate_ref(row,'owner_terminal',lambda x:x.update(completed=False))
    elif defect=='undrained': mutate_ref(row,'owner_terminal',lambda x:x['stage'].update(process_group_drained=False))
    elif defect=='gpu_held': mutate_ref(row,'owner_terminal',lambda x:x.update(gpu_released=False))
    elif defect=='child_exit': mutate_ref(row,'child_result',lambda x:x.update(exit_code=-11))
    elif defect=='wrong_owner': mutate_ref(row,'child_result',lambda x:x['identity'].update(pgid=77))
    elif defect=='wrong_job': mutate_ref(row,'child_result',lambda x:x.update(id='another'))
    elif defect=='changed_config': Path(row['config']['path']).write_text('{}')
    elif defect=='no_close': mutate_ref(row,'run_result',lambda x:x.pop('runtime_shutdown'))
    elif defect=='late_reference': mutate_ref(row,'run_result',lambda x:x['given_reference']['availability_binding'].update(bound_at_s=.1,physics_row=24))
    elif defect=='reference_receipt': mutate_ref(row,'run_result',lambda x:x['given_reference'].update(reference_sha256='0'*64))
    elif defect=='changed_source': (tmp_path/'archive/m2_given_contact_handoff.py').write_text('changed')
    else:
        monkeypatch.setattr(m.witness,'load_contract',lambda _:dict(cpu_diagnostic_only=True))
    # Never reach archive decoding for a broken native metadata chain.
    monkeypatch.setattr(m,'_decode_capture',lambda *a:pytest.fail('Invalid receipt reached capture decoding'))
    with pytest.raises(ValueError): m.read_episode(row,tmp_path)


def test_native_metadata_chain_reaches_payload_only_with_exact_bound_job(tmp_path,monkeypatch):
    m=module();row,meta,*_=metadata_row(tmp_path);run=native_metadata(tmp_path,row,meta)
    monkeypatch.setattr(m.witness,'load_contract',lambda _:dict(cpu_diagnostic_only=False))
    assert m._native_evidence(row,tmp_path,meta)[0]==run


@pytest.mark.parametrize('defect,reason',[('child_exit','child'),('wrong_owner','process group'),
    ('late_reference','before first'),('reference_receipt','reference receipt'),('changed_source','hash changed')])
def test_native_refusals_are_specific_not_incidental(tmp_path,monkeypatch,defect,reason):
    m=module();row,meta,*_=metadata_row(tmp_path);native_metadata(tmp_path,row,meta)
    monkeypatch.setattr(m.witness,'load_contract',lambda _:dict(cpu_diagnostic_only=False))
    if defect=='child_exit':mutate_ref(row,'child_result',lambda x:x.update(exit_code=-11))
    elif defect=='wrong_owner':mutate_ref(row,'child_result',lambda x:x['identity'].update(pgid=77))
    elif defect=='late_reference':mutate_ref(row,'run_result',lambda x:x['given_reference']['availability_binding'].update(bound_at_s=.1,physics_row=24))
    elif defect=='reference_receipt':mutate_ref(row,'run_result',lambda x:x['given_reference'].update(reference_sha256='0'*64))
    else:(tmp_path/'archive/m2_given_contact_handoff.py').write_text('changed')
    # Keep the outer job's COMPLETE hash truthful when testing inner content.
    if defect in ('late_reference','reference_receipt'):
        mutate_ref(row,'child_result',lambda x:x.update(result_sha256=row['run_result']['sha256']))
    with pytest.raises(ValueError,match=reason):m._native_evidence(row,tmp_path,meta)


def test_reference_sources_are_reauthenticated_before_payload(tmp_path,monkeypatch):
    m=module(); row,meta,*_=metadata_row(tmp_path); native_metadata(tmp_path,row,meta)
    Path(meta['given_reference']['sources']['fresh_plan']['file']).write_text('{}')
    monkeypatch.setattr(m,'_decode_capture',lambda *a:pytest.fail('Changed reference reached payload'))
    with pytest.raises(ValueError,match='source hash'):m.read_episode(row,tmp_path)


def test_reference_roundtrip_preserves_selected_full_rotation_and_original_surface_normal(tmp_path):
    from scipy.spatial.transform import Rotation
    m=module();row,meta,*_=metadata_row(tmp_path)
    snapshot=meta['given_reference'];source=snapshot['sources']['fresh_plan'];plan=pose.bound_json(source)
    rotation=Rotation.from_euler('xz',[.015,.7]).as_matrix()
    plan.update(command_rotation_world=rotation.tolist(),command_normal_world=(-rotation[:,2]).tolist())
    bound=save(Path(source['file']),plan);plan_ref=dict(file=bound['path'],sha256=bound['sha256'])
    patient_ref=snapshot['sources']['patient_receipt'];patient=pose.bound_json(patient_ref)
    reference=pose.build_given_reference(dict(patient['targets'][0],manifest_ref=patient_ref),
        plan_ref,patient_ref,np.eye(4),0.)
    changed=pose.route_metadata(meta,reference)
    assert not np.allclose(reference.normal_base,patient['targets'][0]['normal_world'])
    assert m._reference_metadata(changed).sha256==reference.sha256


def test_last_row_frozen_config_membership_is_checked_before_first_payload(tmp_path,monkeypatch):
    m=module();one,*_=metadata_row(tmp_path,name='first');two,*_=metadata_row(tmp_path,name='last')
    mutate_ref(two,'config',lambda x:x['m2_given_contact_control']['metadata'].update(split='test',subject_id='p000438'))
    ref=manifest(tmp_path,[one,two])
    monkeypatch.setattr(m,'_read_payload',lambda *a:pytest.fail('Payload before all config metadata passed'))
    with pytest.raises(ValueError):m.prepare_dataset(ref['path'],ref['sha256'])


def test_old_reader_does_not_accept_new_capture_metadata(tmp_path):
    row,*_=metadata_row(tmp_path)
    with pytest.raises(ValueError): shared._read_episode(row,tmp_path)


def test_given_windows_exclude_boundary_and_history_transients(tmp_path):
    _,records,audit,meta=new_inputs(tmp_path,22)
    assert len(given.assemble_samples(records,audit,meta)['states'])==1
    for role in ('given_l2_park','given_l2_history_wait','L3_ALIGN','contact','home_settle'):
        changed=deepcopy(audit)
        # A truthful natural transition excludes a sample, without fake padding.
        changed[17]['controller_after']['subphase']=role
        assert len(given.assemble_samples(records,changed,meta)['states'])==0


def synthetic_read(row, base, context):
    meta=context[2]
    state=np.r_[np.full(7,1. if row['split']=='train' else 100.),np.zeros(10)]
    n=2
    data=dict(schema=given.DATASET_SCHEMA,task_id=pose.TASK_ID,observation_contract=pose.observation_contract(),
        action_semantics_id=given.ACTION_SEMANTICS_ID,states=np.tile(state,(n,1)),histories=np.tile(state,(n,2,1)),
        actions=np.full((n,20,7),.2),origins=np.tile(state[:7],(n,1)),velocities=np.zeros((n,7)),
        observation_times=np.array([.1,.15]),derived_index=dict(episode_id=row['episode_id'],windows=[dict(observation_indices=[0,1]),dict(observation_indices=[1,2])]))
    receipt=dict(**{k:row[k] for k in ('episode_id','subject_id','split','reset_id','run_id','job_id','run_result')},
        windows=n,capture_bytes=123,reference_sha256=meta['reference_sha256'])
    return data,receipt,meta


def test_preparation_preserves_manifest_order_many_references_and_train_only_stats(tmp_path,monkeypatch):
    m=module(); train,*_=metadata_row(tmp_path,name='train'); val,*_=metadata_row(tmp_path,name='val',split='validation',subject='p000441')
    ref=manifest(tmp_path,[val,train]); monkeypatch.setattr(m,'_read_payload',synthetic_read)
    first=m.prepare_dataset(ref['path'],ref['sha256']); second=m.prepare_dataset(ref['path'],ref['sha256'])
    assert first['training_eligible'] is False and first['test_payloads_opened'] is False
    assert [r['episode_id'] for r in first['episodes']]==['val','train']
    assert len(first['reference_inventory'])==2
    assert first['normalization']['state_mean'][:7]==[1.]*7
    assert first['normalization']==second['normalization']
    assert first['derived_index']==second['derived_index']
    for split in ('train','validation'):
        for key in shared.ARRAY_KEYS:np.testing.assert_array_equal(first[split][key],second[split][key])
    assert first['derived_index']['samples'][0]['episode_id']=='val'
    assert first['derived_index']['samples'][2]['episode_id']=='train'
    assert first['task_contract']['history_steps']==2
    assert first['task_contract']['handoff_policy_id']==POLICY_ID
    assert 'reference_sha256' not in first['task_contract'] and 'given_reference' not in first['task_contract']


def population(tmp_path, monkeypatch, failures=5, later_failures=0):
    """Population math fixture; native evidence checker is separately tested."""
    m=module(); splits=shared._json(shared.contract.IDENTITY_SPLIT_PATH)['splits']
    planned=[];executed=[];outcomes=[];episodes=[];allocations={'train':[],'validation':[]};slots=[]
    for split,n,ids in [('train',200,[x for x in splits['train'] if x not in shared.contract.QUARANTINED_IDENTITIES][:40]),
                        ('validation',40,[x for x in splits['validation'] if x not in shared.contract.QUARANTINED_IDENTITIES][:10])]:
        for i in range(n):
            subject=ids[i%len(ids)];aid=f'{split}-{i:03d}';slot=f'slot-{aid}'
            allocated=dict(subject_id=subject,slot_id=slot,target='L2')
            allocations[split].append(allocated)
            attempt=dict(attempt_id=aid,episode_id=aid,subject_id=subject,split=split,
                reset_id='reset-'+aid,run_id='owner-'+aid,job_id='job-'+aid,slot_id=slot,target_id='L2',
                preclassified_reachable=True,config=dict(path=str(tmp_path/(aid+'.config.json')),sha256='a'*64))
            planned.append(attempt)
            success=not(split=='train' and (i<failures or 100<=i<100+later_failures))
            native=save(tmp_path/(aid+'.native.json'),dict(episode_id=aid,success=success,physical_validation=success,
                learned_policy=False,errors=[] if success else ['native failure'],task_id=pose.TASK_ID,
                geometry_id=geometry_id(),geometry_source_sha256=m.geometry.GEOMETRY_SOURCE_SHA256))
            entry=dict(attempt_id=aid,launched=True,run_result=native,
                launch_receipt=dict(path=str(tmp_path/(aid+'.launch.json')),sha256='b'*64),
                owner_terminal=dict(path=str(tmp_path/(aid+'.owner.json')),sha256='c'*64),
                child_result=dict(path=str(tmp_path/(aid+'.child.json')),sha256='d'*64))
            executed.append(entry)
            outcomes.append(dict(attempt,**entry,native_success=success,outcome='eligible' if success else 'native_failure'))
            if success:episodes.append(dict(attempt,run_result=native,windows=2,manifest_sha256='f'*64,
                reference_sha256=hashlib.sha256(aid.encode()).hexdigest(),capture_bytes=10))
        # Retain every planning slot, including those outside the finite allocation.
        slots.extend(dict(slot_id=f'coverage-{split}-{i}',split=split,subject_id=ids[i%len(ids)],geometry_path_found=i<n)
                     for i in range(800 if split=='train' else 200))
    mapping=save(tmp_path/'mapping.json',dict(all_original_slots=slots,ordered_allocations=allocations,
        summaries=dict(train=dict(seed=2026102701),validation=dict(seed=2026102702))))
    proposal=save(tmp_path/'proposal.json',dict(mapping=dict(file=mapping['path'],sha256=mapping['sha256']),
        train_original_slots_retained=800,validation_original_slots_retained=200,allocation_feasible=True))
    monkeypatch.setattr(m,'ALLOCATION_PROPOSAL_SHA256',proposal['sha256'])
    monkeypatch.setattr(m,'ALLOCATION_MAPPING_SHA256',mapping['sha256'])
    plan=save(tmp_path/'plan.json',dict(schema=m.PLAN_SCHEMA,task_id=pose.TASK_ID,
        action_semantics_id=given.ACTION_SEMANTICS_ID,frozen_before_launch=True,handoff_policy_id=POLICY_ID,
        frozen_utc='2026-10-26T00:00:00Z',allocation_proposal=proposal,allocation_mapping=mapping,attempts=planned))
    execution=save(tmp_path/'execution.json',dict(schema=m.EXECUTION_SCHEMA,finalized=True,
        attempt_plan_sha256=plan['sha256'],attempts=executed,
        first100_gate=dict(attempt_ids=[a['attempt_id'] for a in planned[:100]],native_successes=100-failures,
            passed=failures<=5,decided_utc='2026-10-27T00:00:00Z'),
        accounting=dict(all_attempt_and_operational_bytes=2400,shared_scene_bytes=10,sdk_bytes=10,
                        source_bytes=10,config_bytes=10,capture_bytes=2400,failed_attempt_bytes=0,
                        tier_b_episodes=0,tier_b_bytes=0)))
    ledger=save(tmp_path/'ledger.json',dict(schema=m.LEDGER_SCHEMA,attempts=outcomes))
    original=manifest(tmp_path,deepcopy(episodes))
    for row in episodes:row['manifest_sha256']=original['sha256']
    receipt=save(tmp_path/'qualification.json',dict(schema=m.POPULATION_SCHEMA,task_id=pose.TASK_ID,
        action_semantics_id=given.ACTION_SEMANTICS_ID,passed=True,handoff_policy_id=POLICY_ID,manifest=original,manifest_sha256=original['sha256'],
        minimum_episodes=150,minimum_identities=20,minimum_success_rate=.95,
        attempt_plan=plan,execution_inventory=execution,attempt_ledger=ledger))
    def native(attempt,entry,base):
        i=int(attempt['attempt_id'].rsplit('-',1)[1]);later=attempt['split']=='validation' or i>=100
        return dict(shared._json(entry['run_result']['path']),
            _started_utc='2026-10-27T01:00:00Z' if later else '2026-10-26T01:00:00Z',
            _ended_utc='2026-10-27T02:00:00Z' if later else '2026-10-26T02:00:00Z')
    monkeypatch.setattr(m,'_population_native',native)
    monkeypatch.setattr(m,'_accounting',lambda *a:dict(total_accounted_bytes=2400,all_accounted_jobs_within_budget=True))
    monkeypatch.setattr(m,'_plan_metadata',lambda *a:None)
    monkeypatch.setattr(m,'_launch_state',lambda plan,attempt,entry,base:entry['launched'])
    return receipt,episodes


def geometry_id():
    from sim.ausc_policy.geometry_epoch import GEOMETRY_ID
    return GEOMETRY_ID


def change_population(ref,key,transform):
    receipt=shared._json(ref['path']);target=receipt[key];value=shared._json(target['path']);transform(value)
    receipt[key]=save(Path(target['path']),value)
    return save(Path(ref['path']),receipt)


def test_population_math_keeps_failed_attempts_and_fixed_first100(tmp_path,monkeypatch):
    m=module();ref,episodes=population(tmp_path,monkeypatch)
    result=m._verify_population(ref,episodes)
    assert result['reachable_denominator']==200 and result['reachable_successes']==195
    assert result['first100_native_successes']==95
    assert result['original_train_slots']==800 and result['original_validation_slots']==200
    assert result['qualified_train_episodes']==195 and result['qualified_train_identities']==40


@pytest.mark.parametrize('document',['qualification','attempt_plan'])
def test_population_preflight_refuses_old_policy_before_capture_payload(tmp_path,monkeypatch,document):
    m=module();ref,episodes=population(tmp_path,monkeypatch)
    if document=='attempt_plan':ref=change_population(ref,document,lambda x:x.pop('handoff_policy_id'))
    else:
        value=shared._json(ref['path']);value.pop('handoff_policy_id');ref=save(Path(ref['path']),value)
    with pytest.raises(ValueError,match='handoff policy'):
        m._population_preflight(ref,tmp_path,episodes[0]['manifest_sha256'])


@pytest.mark.parametrize('entry_point',['private','preflight','public'])
@pytest.mark.parametrize('defect',['missing','sha','changed','extra_key'])
def test_every_population_path_requires_actual_original_manifest(tmp_path,monkeypatch,entry_point,defect):
    m=module();ref,episodes=population(tmp_path,monkeypatch)
    receipt=shared._json(ref['path']);original=receipt['manifest']
    if defect=='missing':receipt.pop('manifest')
    elif defect=='sha':receipt['manifest']['sha256']='0'*64
    elif defect=='changed':Path(original['path']).write_text('{}')
    else:receipt['manifest']['bytes']=Path(original['path']).stat().st_size
    ref=save(Path(ref['path']),receipt)
    monkeypatch.setattr(m,'_read_payload',lambda *a:pytest.fail('Capture payload opened before manifest binding'))
    with pytest.raises(ValueError,match='manifest|artifact'):
        if entry_point=='private':m._verify_population(ref,episodes)
        elif entry_point=='preflight':m._population_preflight(ref,tmp_path,episodes[0]['manifest_sha256'])
        else:m.verify_population(ref,episodes)


def test_preparer_refuses_missing_original_population_manifest_before_any_payload(tmp_path,monkeypatch):
    m=module();one,*_=metadata_row(tmp_path,name='train');two,*_=metadata_row(tmp_path,name='val',split='validation',subject='p000441')
    original=manifest(tmp_path,[one,two])
    qualification=save(tmp_path/'qualification.json',dict(manifest_sha256=original['sha256'],handoff_policy_id=POLICY_ID))
    monkeypatch.setattr(m,'_read_payload',lambda *a:pytest.fail('Capture opened before original population manifest validation'))
    with pytest.raises(ValueError,match='original immutable manifest'):
        m.prepare_dataset(original['path'],original['sha256'],qualification)


@pytest.mark.parametrize('defect',['first100','final_rate','positive_only','missing_unlaunched','zero_windows',
    'old_schema','weakened_cut','wrong_manifest','reclassified','duplicate_reset','duplicate_receipt','reordered_plan'])
def test_population_refuses_selection_rate_identity_and_schema_bypasses(tmp_path,monkeypatch,defect):
    m=module();ref,episodes=population(tmp_path,monkeypatch,failures=6 if defect=='first100' else 5,
        later_failures=6 if defect=='final_rate' else 0)
    if defect=='positive_only':ref=change_population(ref,'attempt_ledger',lambda x:x.update(attempts=x['attempts'][5:]))
    elif defect=='missing_unlaunched':ref=change_population(ref,'execution_inventory',lambda x:x['attempts'][150].update(launched=False))
    elif defect=='zero_windows':
        for row in episodes[:100]:row['windows']=0
    elif defect in ('old_schema','weakened_cut','wrong_manifest'):
        value=shared._json(ref['path'])
        value.update(**({'schema':'m2_expert_population_qualification_v1'} if defect=='old_schema' else
            {'minimum_success_rate':.9} if defect=='weakened_cut' else {'manifest_sha256':'0'*64}))
        ref=save(Path(ref['path']),value)
    elif defect=='reclassified':ref=change_population(ref,'attempt_ledger',lambda x:x['attempts'][0].update(preclassified_reachable=False))
    elif defect=='duplicate_reset':ref=change_population(ref,'attempt_plan',lambda x:x['attempts'][1].update(reset_id=x['attempts'][0]['reset_id']))
    elif defect=='duplicate_receipt':ref=change_population(ref,'attempt_ledger',lambda x:x['attempts'][1].update(run_result=x['attempts'][0]['run_result']))
    elif defect=='reordered_plan':ref=change_population(ref,'attempt_plan',lambda x:x['attempts'].reverse())
    with pytest.raises(ValueError):m._verify_population(ref,episodes)


def test_public_population_reconstructs_original_capture_rows_before_counts(tmp_path,monkeypatch):
    m=module();one,*_=metadata_row(tmp_path,name='train');two,*_=metadata_row(tmp_path,name='val',split='validation',subject='p000441')
    ref=manifest(tmp_path,[one,two]);monkeypatch.setattr(m,'_read_payload',synthetic_read)
    actual=m.prepare_dataset(ref['path'],ref['sha256'])['episodes']
    qualification=save(tmp_path/'qualification.json',dict(manifest=ref,manifest_sha256=ref['sha256']))
    monkeypatch.setattr(m,'_population_preflight',lambda *a:None)
    monkeypatch.setattr(m,'_verify_population',lambda ref,rows:dict(passed=True,windows=sum(r['windows'] for r in rows)))
    invented=deepcopy(actual);invented[0]['windows']=10000
    with pytest.raises(ValueError,match='reconstruct'):m.verify_population(qualification,invented)
    assert m.verify_population(qualification,actual)['windows']==4


def test_authoritative_launch_file_cannot_be_called_unlaunched_when_result_missing(tmp_path):
    m=module();owner=tmp_path/'run';output=owner/'output';output.mkdir(parents=True)
    launch=save(output/'job.launch.json',dict(id='job'))
    attempt=dict(run_id='run',job_id='job');plan=dict(run_root=str(tmp_path))
    with pytest.raises(ValueError,match='launch'):m._launch_state(plan,attempt,dict(launched=False),tmp_path)
    assert m._launch_state(plan,attempt,dict(launched=True,launch_receipt=launch),tmp_path) is True


def test_same_native_process_cannot_be_reused_as_two_resets(tmp_path,monkeypatch):
    m=module();folders=[tmp_path/'one',tmp_path/'two']
    rows=[]
    for i,folder in enumerate(folders):
        folder.mkdir();row,meta,*_=metadata_row(folder,name=f'case{i}');native_metadata(folder,row,meta);rows.append(row)
    ref=manifest(tmp_path,rows)
    monkeypatch.setattr(m,'_read_payload',lambda *a:pytest.fail('Reused native reset reached payload'))
    with pytest.raises(ValueError,match='process|reset'):m.prepare_dataset(ref['path'],ref['sha256'])


def test_launched_aborted_child_retains_partial_safety_failure_without_success(tmp_path):
    m=module();row,meta,*_=metadata_row(tmp_path);native_metadata(tmp_path,row,meta)
    child=shared._json(row['child_result']['path']);output=Path(child['output'])
    launch=save(output.parent/(row['job_id']+'.launch.json'),dict(child,started_utc='2026-10-27T00:58:44Z'))
    partial=save(output/'fresh_expert_result.json',dict(physical_validation=False,success=False,
        errors=[dict(stage='execution',error='Native adjacent-command velocity exceeds .175')]))
    Path(row['child_result']['path']).unlink();Path(row['run_result']['path']).unlink()
    entry=dict(launched=True,owner_terminal=row['owner_terminal'],launch_receipt=launch,partial_result=partial)
    result=m._population_native(row,entry,tmp_path)
    assert result['success'] is False and result['physical_validation'] is False
    assert result['_primary_failure_kind']=='resource_abort'
    assert set(result['_reason_flags'])=={'lifecycle_failure','launched_aborted_missing_result','native_safety_failure'}
    omitted=dict(entry);omitted.pop('partial_result')
    with pytest.raises(ValueError,match='partial'):m._population_native(row,omitted,tmp_path)


def physical_fixture(tmp_path,monkeypatch,*,gap=.025,release_ticks=48,plan_change=None,patient_change=None):
    m=module();ref,_,_,meta=new_inputs(tmp_path)
    # Original physical gates are isolated here; complete new parking evidence
    # is reconstructed independently in test_ausc_given_contact_handoff.py.
    monkeypatch.setattr(m,'_interior_parking',lambda *a,**k:None)
    snapshot=ref.snapshot();sources=snapshot['sources']
    patient=pose.bound_json(sources['patient_receipt'])
    plane=dict(contact_plane_normal_world=[0.,0.,-1.],contact_plane_radius_m=.023,
        plane_tilt_limit_deg=3.,tilt_relaxed=False)
    patient['targets'][0].update(plane)
    if patient_change:patient_change(patient)
    saved=save(Path(sources['patient_receipt']['file']),patient)
    patient_ref=dict(file=saved['path'],sha256=saved['sha256'])
    plan=pose.bound_json(sources['fresh_plan'])
    plan.update(free_approach_start_gap_m=gap,selected_fit=dict(first_contact=deepcopy(plane)))
    plan['patient_provenance']['binding']=patient_ref
    if plan_change:plan_change(plan)
    saved=save(Path(sources['fresh_plan']['file']),plan)
    plan_ref=dict(file=saved['path'],sha256=saved['sha256'])
    ref=pose.build_given_reference(dict(patient['targets'][0],manifest_ref=patient_ref),plan_ref,patient_ref,np.eye(4),0.)
    meta=pose.route_metadata(meta,ref)
    meta.update(patient_asset_sha256=patient['asset_sha256'],installed_patient_receipt_sha256=patient_ref['sha256'])
    meta['native_retainer_audit']=dict(schema='retainer_front_audit_v3',qualification={})
    monkeypatch.setattr(m.witness,'load_contract',lambda _:dict(cpu_diagnostic_only=False))
    from sim.ausc_policy.m2_given_contact_controller import BOUNDS
    audit=[]
    for i in range(1200):
        role=('given_l2_motion' if i<288 else 'given_l2_park' if i<300 else 'L3_ALIGN' if i<312 else
              'contact' if i<324 else 'hold' if i<1044 else 'retract' if i<1056 else
              'normal_release_confirm' if i<1056+release_ticks else 'home_settle')
        phase='HOLD' if role=='hold' else 'RETRACT' if role in ('retract','normal_release_confirm') else 'DONE' if role=='home_settle' else 'APPROACH'
        values={k+'_N':0. for k in ('diaphragm','retainer','lip','contact_rim','carriage_other','frame','forbidden')}
        values.update(hall_N=4. if role=='hold' else 0.,compression_m=.002 if role=='hold' else 0.,
            gap_m=min(.04,gap) if i>=287 else .06,tangent_m=0.,angle_deg=0.,tcp_speed_m_s=0.,axis_world=[0.,0.,1.])
        values['diaphragm_N']=4. if role=='hold' else 0.
        actor=dict(owner='L2' if i<288 else 'PREPARATION' if i<300 else 'L3',phase='APPROACH',reference_sha256=meta['reference_sha256'])
        audit.append(dict(physics_row=i,time_s=(i+1)/240.,state_audit='completed',physical_violations=[],tick_error=None,
            controller_after=dict(phase=phase,subphase=role,actor=actor),controller_before=dict(phase=phase,subphase=role,actor=actor),
            measurements=values,native_retainer_audit=meta['native_retainer_audit'],
            retainer_front_audit=dict(schema='retainer_front_audit_v3',native_retainer_audit=meta['native_retainer_audit'])))
    audit_channels={kind:dict(samples=13,max_translation_m=.001,path_length_m=.001,max_full_rotation_rad=.01,
        rotation_path_rad=.01,max_local_plane_deg=1.) for kind in ('command','observed')}
    handoff=dict(target=meta['target_id'],l2_coarse_reached=True,l2_handoff_success=True,alignment_success=True,
        l3_started=True,l3_completed=True,expert_l2_fallback=False,coarse_entry=dict(passed=True,physics_row=288,time_s=1.2),
        parked_entry=dict(passed=True,physics_row=300,time_s=1.25),parking_trigger=dict(physics_row=288),alignment_admission=dict(passed=True),
        numerical_curve_entry=dict(passed=True),alignment_actual_audit=audit_channels,
        alignment_plan=dict(schema='m2_given_contact_l3_align_v1',bounds=BOUNDS,reference_sha256=meta['reference_sha256'],
            reference_revision=0,all_240hz_commands_verified=True,final_stationary_stop_included=True,duration_s=.05,planning_cpu_s=.1))
    run=dict(command_collision_samples=1200,observed_collision_samples=1200,
        posture_command_checks='every commanded physics sample',posture_observation_checks='every observed physics sample',
        phase_failure_diagnostics=[],collision_failure_diagnostics=[],l2_handoffs=[handoff],
        static_world_verification=dict(checked_physics_steps=1200,last_report=dict(clear=True,violations=[]),
            containment_gate=dict(source_npz_sha256=meta['collision_arrays_sha256'])),
        phase_config=dict(precontact_gap_m=.04,precontact_tolerance_m=.002,approach_tangential_tolerance_m=.003,
            angle_tolerance_deg=5.,approach_speed_limit_m_s=.02),
        points=[dict(target=meta['target_id'],success=True,final_phase='DONE',checked_prepose_gap_m=gap,
            phase_config=dict(precontact_gap_m=min(.04,gap),precontact_tolerance_m=.002,approach_tangential_tolerance_m=.003,
                angle_tolerance_deg=5.,approach_speed_limit_m_s=.02),force_audit=dict(passed=True),contact_plane_audit=dict(passed=True))],
        contact_plane_contracts={meta['target_id']:deepcopy(plane)})
    return run,audit,meta


def test_physical_reader_consumes_actual_two_channel_align_report(tmp_path,monkeypatch):
    run,audit,meta=physical_fixture(tmp_path,monkeypatch)
    module()._physical(run,audit,meta)


@pytest.mark.parametrize('gap',[.025,.03,.04,.06])
def test_original_base_and_point_coarse_gaps_follow_checked_plan(tmp_path,monkeypatch,gap):
    run,audit,meta=physical_fixture(tmp_path,monkeypatch,gap=gap)
    run['phase_config']['precontact_gap_m']=.04
    module()._physical(run,audit,meta)


@pytest.mark.parametrize('defect',['point_gap','checked_gap','command_summary','observed_summary','observed_axis',
    'command_q','relaxed_run','relaxed_plan','plan_patient_plane','plan_radius'])
def test_plane_and_gap_limits_are_authenticated_independently(tmp_path,monkeypatch,defect):
    plan_change=None;patient_change=None
    if defect=='relaxed_plan':
        plan_change=lambda p:p['selected_fit']['first_contact'].update(plane_tilt_limit_deg=10.,tilt_relaxed=True)
    elif defect=='plan_radius':plan_change=lambda p:p['selected_fit']['first_contact'].update(contact_plane_radius_m=.024)
    elif defect=='plan_patient_plane':
        patient_change=lambda p:p['targets'][0].update(contact_plane_normal_world=[0.,-.1,-np.sqrt(.99)])
    run,audit,meta=physical_fixture(tmp_path,monkeypatch,plan_change=plan_change,patient_change=patient_change)
    if defect=='point_gap':run['points'][0]['phase_config']['precontact_gap_m']=.03
    elif defect=='checked_gap':run['points'][0]['checked_prepose_gap_m']=.03
    elif defect in ('command_summary','observed_summary'):
        kind=defect.split('_')[0];run['l2_handoffs'][0]['alignment_actual_audit'][kind]['max_local_plane_deg']=100.
    elif defect=='observed_axis':audit[305]['measurements']['axis_world']=[0.,np.sin(np.deg2rad(8)),np.cos(np.deg2rad(8))]
    elif defect=='command_q':audit[305]['q_command']=[0.]*7  # Native FK axis is +Y, 90 degrees from this plane.
    elif defect=='relaxed_run':
        run['contact_plane_contracts'][meta['target_id']].update(plane_tilt_limit_deg=10.,tilt_relaxed=True)
        for event in audit[324:1044]:event['measurements']['axis_world']=[0.,np.sin(np.deg2rad(8)),np.cos(np.deg2rad(8))]
    with pytest.raises(ValueError):module()._physical(run,audit,meta)


@pytest.mark.parametrize('defect',['36_ticks','continuous_break','hall','compression','force','late_home','wrong_phase','before_hold','after_home'])
def test_native_release_needs_48_continuous_original_confirm_ticks(tmp_path,monkeypatch,defect):
    run,audit,meta=physical_fixture(tmp_path,monkeypatch,release_ticks=36 if defect=='36_ticks' else 48)
    if defect=='continuous_break':audit[1080]['controller_after']['subphase']='normal_unload'
    elif defect=='hall':audit[1080]['measurements']['hall_N']=.2
    elif defect=='compression':audit[1080]['measurements']['compression_m']=.00016
    elif defect=='force':audit[1080]['measurements']['diaphragm_N']=.2
    elif defect=='late_home':
        for event in audit[1056:1104]:event['controller_after']['subphase']='home_settle'
    elif defect=='wrong_phase':
        for event in audit[1056:1104]:event['controller_after']['phase']='APPROACH'
    elif defect=='before_hold':
        for event in audit[312:360]:
            event['controller_after'].update(subphase='normal_release_confirm',phase='RETRACT')
            event['measurements'].update(diaphragm_N=0.,hall_N=0.,compression_m=0.)
        for event in audit[360:1080]:
            event['controller_after'].update(subphase='hold',phase='HOLD')
            event['measurements'].update(diaphragm_N=4.,hall_N=4.,compression_m=.002)
        for event in audit[1080:1104]:event['controller_after']['subphase']='normal_unload'
    elif defect=='after_home':audit[1055]['controller_after'].update(subphase='home_settle',phase='DONE')
    with pytest.raises(ValueError,match='release'):module()._physical(run,audit,meta)


@pytest.mark.parametrize('defect',['short_hold','force','plane','alignment_path','alignment_samples','early_coarse',
    'no_park','no_release','no_home','native_violation','missing_tick','containment'])
def test_original_native_physical_gates_cannot_be_replaced_by_success_flag(tmp_path,monkeypatch,defect):
    m=module();run,audit,meta=physical_fixture(tmp_path,monkeypatch)
    if defect=='short_hold':audit[500]['measurements']['diaphragm_N']=0.
    elif defect=='force':audit[500]['measurements']['lip_N']=.001
    elif defect=='plane':audit[500]['measurements']['axis_world']=[1.,0.,0.]
    elif defect=='alignment_path':run['l2_handoffs'][0]['alignment_actual_audit']['observed']['path_length_m']=.0031
    elif defect=='alignment_samples':run['l2_handoffs'][0]['alignment_actual_audit']['observed']['samples']=1
    elif defect=='early_coarse':run['l2_handoffs'][0]['coarse_entry']['physics_row']=287
    elif defect=='no_park':audit[290]['controller_after']['subphase']='given_l2_motion'
    elif defect=='no_release':audit[-10]['measurements']['diaphragm_N']=.3
    elif defect=='no_home':audit[-1]['controller_after']['subphase']='retract'
    elif defect=='native_violation':audit[20]['physical_violations']=['collision']
    elif defect=='missing_tick':audit[20]['physics_row']=21
    else:run['static_world_verification']['containment_gate']['source_npz_sha256']='0'*64
    with pytest.raises(ValueError):m._physical(run,audit,meta)


def accounting_fixture(tmp_path):
    from test_ausc_policy_scene_store import bound
    from sim.ausc_policy import scene_store
    binding,config=bound(tmp_path)
    rows=[];entries={};roots=[]
    for i in range(2):
        owner=tmp_path/f'owner{i}';root=owner/'output'/f'job{i}'
        scene_store.prepare_job(config,root)
        (owner/'owner.log').write_bytes(b'operational log')
        ref=save(owner/'result.json',dict(run_id=f'owner{i}'))
        cfg=save(tmp_path/f'config{i}.json',config)
        row=dict(attempt_id=f'a{i}',config=cfg,job_id=f'job{i}',source_freeze=None)
        rows.append(row);entries[row['attempt_id']]=dict(launched=True,owner_terminal=ref)
        roots.append(str(owner))
    # An immutable external config source shared by all phases must be charged once.
    extra=tmp_path/'shared_source.py';extra.write_bytes(b'x'*100)
    ref=dict(path=str(extra),sha256=shared.sha256(extra),bytes=100)
    scope=dict(owner_roots=roots,extra_input_refs=[dict(category='source',reference=ref),
        *[dict(category='config',reference=dict(r['config'],bytes=Path(r['config']['path']).stat().st_size)) for r in rows]],
        tier_b_attempt_ids=[],storage_roots=[dict(category=kind,path=str(tmp_path/'prepared'))
            for kind in ('complete_cpu_bank','backup_archive','provisioned_runtime')])
    return dict(accounting_scope=scope),rows,entries,extra


def test_population_accounts_unique_union_and_all_owner_overhead(tmp_path):
    m=module();plan,rows,entries,extra=accounting_fixture(tmp_path)
    result=m._accounting(plan,rows,entries,tmp_path)
    union=result['unique_union']; owners=result['per_owner']
    assert len(owners)==2 and union['observed_jobs']==2
    assert result['additional_input_bytes']==100+sum(Path(r['config']['path']).stat().st_size for r in rows)
    assert result['owner_operational_bytes']==sum(r['batch_overhead_bytes'] for r in owners)
    assert result['total_accounted_bytes']==union['total_accounted_bytes']+result['owner_operational_bytes']+result['additional_input_bytes']
    assert result['all_accounted_jobs_within_budget'] is True
    assert {r['category'] for r in result['storage_reports']}=={'complete_cpu_bank','backup_archive','provisioned_runtime'}


def test_provisioned_storage_stats_count_links_without_following_or_decoding(tmp_path):
    m=module();plan,rows,entries,_=accounting_fixture(tmp_path)
    runtime=tmp_path/'runtime';runtime.mkdir()
    source=runtime/'python';source.write_bytes(b'fake-runtime'*1000)
    os.link(source,runtime/'python-hardlink')
    outside=tmp_path/'outside';outside.mkdir();(outside/'unrelated').write_bytes(b'x'*20000)
    (runtime/'venv-python').symlink_to(source)
    (runtime/'directory-link').symlink_to(outside,target_is_directory=True)
    (runtime/'loop').symlink_to(runtime,target_is_directory=True)
    (runtime/'dangling').symlink_to(runtime/'missing')
    plan['accounting_scope']['storage_roots'][-1]['path']=str(runtime)
    report=m._accounting(plan,rows,entries,tmp_path)['storage_reports'][-1]
    links=sum(p.lstat().st_size for p in runtime.iterdir() if p.is_symlink())
    assert report['regular_file_bytes']==2*source.stat().st_size and report['files']==2
    assert report['symlink_bytes']==links and report['symlinks']==4
    assert report['total_bytes']==2*source.stat().st_size+links
    unique={}
    for path in [runtime,*runtime.iterdir()]:
        st=path.lstat();unique[(st.st_dev,st.st_ino)]=st.st_blocks*512
    assert report['inode_unique_physical_bytes']==sum(unique.values())


def test_separate_storage_symlinks_do_not_relax_native_job_budget_inventory(tmp_path):
    m=module();plan,rows,entries,_=accounting_fixture(tmp_path)
    job=Path(entries['a0']['owner_terminal']['path']).parent/'output'/rows[0]['job_id']
    (job/'hidden-source').symlink_to(tmp_path/'shared_source.py')
    with pytest.raises(ValueError,match='symlink'):m._accounting(plan,rows,entries,tmp_path)


@pytest.mark.parametrize('defect',['omitted_owner','missing_config','changed_extra','tier_b','missing_storage_scope'])
def test_population_accounting_refuses_partial_or_mutated_cost_scope(tmp_path,defect):
    m=module();plan,rows,entries,extra=accounting_fixture(tmp_path)
    scope=plan['accounting_scope']
    if defect=='omitted_owner':scope['owner_roots'].pop()
    elif defect=='missing_config':scope['extra_input_refs'].pop()
    elif defect=='changed_extra':extra.write_bytes(b'changed')
    elif defect=='tier_b':scope['tier_b_attempt_ids']=['a0']
    else:scope['storage_roots']=[]
    with pytest.raises(ValueError):m._accounting(plan,rows,entries,tmp_path)


def test_immutable_export_reconstructs_twice_and_never_rewrites_source(tmp_path,monkeypatch):
    m=module(); train,*_=metadata_row(tmp_path,name='train');val,*_=metadata_row(tmp_path,name='val',split='validation',subject='p000441')
    ref=manifest(tmp_path,[train,val]);before=Path(ref['path']).read_bytes()
    monkeypatch.setattr(m,'_read_payload',synthetic_read)
    output=tmp_path/'new-output'
    result=m.export_dataset(ref['path'],ref['sha256'],output)
    assert Path(ref['path']).read_bytes()==before
    assert result['training_eligible'] is False
    assert set(p.name for p in output.iterdir())=={'manifest.json','index.json','normalization.json','qualification.json','reconstruction.json'}
    with pytest.raises(ValueError,match='new'):m.export_dataset(ref['path'],ref['sha256'],output)


def decoder_fixture(tmp_path,monkeypatch):
    """Real compact CPU writer/fidelity reader; public live qualification refuses."""
    from sim.ausc_policy import expert_capture,training_capture
    from test_ausc_policy_compact_expert_capture import compact_binding
    from test_ausc_policy_expert_capture import GEOMETRY,Env,Backend
    monkeypatch.setattr(module(),'_scene_integrity',lambda *a:None)
    ref,_,audit,meta=new_inputs(tmp_path,24)
    meta['handoff_policy_id']=POLICY_ID
    meta.update(training_capture.QUALIFICATIONS)
    binding=compact_binding();binding['episode_id']=meta['episode_id'];settings=binding['compact'];settings.update(metadata=meta,max_frames=30,
        estimated_target_base=ref.pose_base.tolist(),target_confidence=1.,target_estimate_source_id='given-contact-'+ref.sha256)
    capture=expert_capture.create_expert_capture(tmp_path/'capture',binding=binding,geometry_metadata=GEOMETRY)
    env=Env();capture.start(env,backend_factory=Backend)
    for i,event in enumerate(audit):
        if i>=len(audit)-12:
            event['q_command']=[0.]*7;event['raw_intent']=None
            for side in ('controller_before','controller_after'):
                event[side].update(subphase='home_settle',actual_control_owner='PREPARATION')
                event[side]['actor'].update(phase=None,owner='PREPARATION',role='PREPARATION')
        env.advance();q=np.asarray(event['q_command'])
        env.rows[-1].update(target='L1',subphase=event['controller_after']['subphase'],q_command=q.tolist())
        capture.after_physics(env,i)
        before=deepcopy(event['controller_before']);after=deepcopy(event['controller_after'])
        before['raw_intent']=deepcopy(event['raw_intent']);after['raw_intent']=deepcopy(event['raw_intent'])
        capture.finish_tick(env,i,before,after,clean_command=dict(q_command=q,issued_at_s=i/240.,
            endpoint_at_s=(i+1)/240.,readback_at_s=(i+1)/240.,source_id=f'native-{i}'),control_owner='L2')
    capture.close();outer=capture.result(len(audit),execution_completed=True)
    assert outer['compact_capture_complete']
    root=capture.output;manifest=shared._json(root/'compact/manifest.json');meta=manifest['metadata']
    source=tmp_path/'archive/source.py';source.parent.mkdir();source.write_text('diagnostic archive')
    commit=tmp_path/'archive/CODE_COMMIT';commit.write_text('a'*40+'\n')
    code_manifest=save(tmp_path/'archive/SOURCE_MANIFEST.json',[dict(file='source.py',sha256=shared.sha256(source))])
    sources={key:dict(path=key,bytes=Path(key).stat().st_size,sha256=digest) for key,digest in capture.sources.items()}
    freeze=save(tmp_path/'freeze.json',dict(schema=module().SOURCE_FREEZE_SCHEMA,code_commit='a'*40,
        code_commit_file=dict(path=str(commit),sha256=shared.sha256(commit)),source_manifest=code_manifest,files=sources))
    row=dict(episode_id=meta['episode_id'],subject_id=meta['subject_id'],split=meta['split'],reset_id='cpu-reset',
        run_id='cpu-run',job_id='cpu-job',capture_root=str(root),source_freeze=freeze,
        config={},owner_terminal={},child_result=save(tmp_path/'CPU-child.json',dict(identity=dict(pid=12,pgid=11,sid=11,start_ticks='123'))),
        run_result=dict(path=str(tmp_path/'CPU-COMPLETE.json'),sha256='a'*64),
        artifacts={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=shared.sha256(p))
            for p in root.rglob('*') if p.is_file()})
    run=dict(raw_multimodal_capture=outer,physics_steps=len(audit),runtime_sources_sha256=capture.sources)
    return row,(root,manifest,meta),run


def test_new_decoder_exhausts_real_compact_fidelity_and_retains_raw_labels(tmp_path,monkeypatch):
    m=module();row,context,run=decoder_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_physical',lambda *a:None)  # Native acceptance is independently tested; this is CPU decoding only.
    samples,receipt,_=m._decode_capture(row,tmp_path,context,run,{},None)
    assert samples['training_eligible'] is False and len(samples['states'])==2
    assert samples['actions'][0,0,0]==pytest.approx(.704)
    assert receipt['capture_bytes']==run['raw_multimodal_capture']['total_bytes']
    assert samples['derived_index']['source_row_range']==[0,24]
    with pytest.raises(ValueError,match='CPU|Live'):m.read_episode(row,tmp_path)


@pytest.mark.parametrize('defect',['confidence','pose'])
def test_decoder_cannot_change_prescribed_pose_or_availability_confidence(tmp_path,monkeypatch,defect):
    m=module();row,context,run=decoder_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_physical',lambda *a:None)
    original=m.iter_training_records
    def changed(path):
        for i,value in enumerate(original(path)):
            if i==1:
                if defect=='confidence':value['target_confidence']=.5
                else:value['estimated_target_base'][0,3]+=.001
                value['state17']=shared.contract.encode_state(value['measured_q'],value['estimated_target_base'],
                    float(value['target_confidence']),metadata=context[2])
            yield value
    monkeypatch.setattr(m,'iter_training_records',changed)
    with pytest.raises(ValueError,match='prescribed|confidence'):m._decode_capture(row,tmp_path,context,run,{},None)


@pytest.mark.parametrize('defect',['inventory','bytes','camera','native_count','reference','archived_source'])
def test_new_decoder_rejects_exact_inventory_frame_reference_and_source_defects(tmp_path,monkeypatch,defect):
    m=module();row,context,run=decoder_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_physical',lambda *a:None)
    if defect=='inventory':row['artifacts'].pop('given_reference.json')
    elif defect=='bytes':row['artifacts']['native_audit.jsonl.gz']['bytes']+=1
    elif defect=='camera':run['raw_multimodal_capture']['receipts']['camera']['frames']+=1
    elif defect=='native_count':run['physics_steps']+=1
    elif defect=='reference':context[2]['reference_sha256']='0'*64
    else:
        value=shared._json(row['source_freeze']['path']);value['files'].pop(next(iter(value['files'])))
        row['source_freeze']=save(Path(row['source_freeze']['path']),value)
    with pytest.raises(ValueError):m._decode_capture(row,tmp_path,context,run,{},None)


def test_full_scene_integrity_uses_existing_cas_and_whole_job_ledger(tmp_path,monkeypatch):
    from test_ausc_given_contact_scene_store import given_config
    from sim.ausc_policy import scene_store
    config,_=given_config(tmp_path,monkeypatch)
    job=tmp_path/'owner/output/job';scene_store.prepare_job(config,job)
    meta=config['m2_given_contact_control']['metadata']
    run=dict(geometry_artifacts={k:meta[k] for k in ('collision_geometry_sha256','collision_arrays_sha256')})
    scene_store.finalize_job(job,run)
    row=dict(run_result=dict(path=str(job/'COMPLETE.json'),sha256=shared.sha256(job/'COMPLETE.json')),
             capture_root=str(job/'compact_multimodal'))
    module()._scene_integrity(row,tmp_path,run,config,meta)
    (job/'unaccounted.bin').write_bytes(b'not declared at native completion')
    with pytest.raises(ValueError,match='job|bytes'):module()._scene_integrity(row,tmp_path,run,config,meta)
