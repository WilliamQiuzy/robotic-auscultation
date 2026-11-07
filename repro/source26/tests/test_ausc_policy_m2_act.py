"""M2 ACT CPU contract tests; synthetic inputs never establish physics acceptance."""
import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sim.ausc_policy import m2_contract as contract
from test_ausc_policy_training_capture import metadata


def module():
    assert importlib.util.find_spec('sim.ausc_policy.m2_act') is not None, 'new M2 ACT baseline missing'
    return importlib.import_module('sim.ausc_policy.m2_act')


def inputs(n=24):
    meta = metadata()
    rows = []
    for i in range(n):
        q = np.full(7, .1 + i*.001)
        t = (i+1)*.05
        rows.append(dict(index=i, observed_at_s=t, measured_q=q,
            state17=contract.encode_state(q,np.eye(4),.8,metadata=meta),
            estimated_target_base=np.eye(4),target_confidence=.8,
            target_source=meta['target_source'],episode_id=meta['episode_id'],target_id='R3',
            target_estimate_time_s=t,observation_source_id=f'native-{12*(i+1)-1}',
            phase='APPROACH',control_owner='L2',record_kind='paired' if i<n-1 else 'terminal_observation'))
        if i<n-1:
            rows[-1].update(q_command=np.full(7,.4+i*.003),action_endpoint_at_s=t+.05,
                action_issued_at_s=t+.049,action_readback_at_s=t+.05,action_source_id=f'cmd-{i}')
    audit = [dict(physics_row=i,time_s=(i+1)/240.,
        controller_before=dict(target='R3',phase='APPROACH',subphase='transfer',active=False,recovery=False,
            actor=dict(phase='APPROACH',owner='L2',sampled_at_s=i/240.,
                rule_id='m2_estimated_tcp_distance_latch_9cm_v1',source='native_measured_q_estimated_target_v1',estimated_tcp_distance_m=.08)),
        controller_after=dict(target='R3',phase='APPROACH',subphase='transfer',active=False,recovery=False,
            actor=dict(phase='APPROACH',owner='L2',sampled_at_s=(i+1)/240.,
                rule_id='m2_estimated_tcp_distance_latch_9cm_v1',source='native_measured_q_estimated_target_v1',estimated_tcp_distance_m=.08)),control_owner='L2',
        state_audit='completed',physical_violations=[],tick_error=None) for i in range(n*12)]
    return rows,audit,meta


def test_full_twenty_future_actual_commands_fixed_measured_origin():
    rows,audit,meta=inputs()
    samples=module().assemble_samples(rows,audit,meta)
    assert samples['states'].shape==(3,17)
    assert samples['actions'].shape==(3,20,7)
    np.testing.assert_allclose(samples['actions'][0,:,0],.4+np.arange(1,21)*.003-.101)
    np.testing.assert_allclose(samples['origins'][0],.101)
    assert samples['training_eligible'] is False
    assert samples['action_representation']=='block_relative_joint'


@pytest.mark.parametrize('direction', [-np.inf, np.inf])
def test_causality_uses_validated_observed_clock_not_ideal_grid(direction):
    rows,audit,meta=inputs()
    for row in rows:
        actual=float(np.nextafter(row['observed_at_s'], direction))
        row['observed_at_s']=actual
        row['target_estimate_time_s']=actual
        if 'q_command' in row:
            row['action_issued_at_s']=actual
            row['action_endpoint_at_s']=actual+.05
            row['action_readback_at_s']=actual+.05
    samples=module().assemble_samples(rows,audit,meta)
    assert len(samples['states'])==3
    np.testing.assert_array_equal(samples['observation_times'],
                                  [r['observed_at_s'] for r in rows[1:4]])


@pytest.mark.parametrize('kind', ['future_estimate', 'early_command', 'bad_cadence'])
def test_observed_clock_fix_preserves_strict_causality_and_cadence(kind):
    rows,audit,meta=inputs()
    row=rows[2]
    row['observed_at_s']=float(np.nextafter(row['observed_at_s'], np.inf))
    row['target_estimate_time_s']=row['observed_at_s']
    if kind=='future_estimate':
        row['target_estimate_time_s']=float(np.nextafter(row['observed_at_s'],np.inf))
    elif kind=='early_command':
        row['action_issued_at_s']=float(np.nextafter(row['observed_at_s'],-np.inf))
    else:
        row['observed_at_s']+=2e-9
    with pytest.raises(ValueError):
        module().assemble_samples(rows,audit,meta)


@pytest.mark.parametrize('side,key,value',[('controller_before','phase','FREE'),
    ('controller_after','phase','CONTACT'),('controller_before','target','L3'),
    ('controller_after','episode_id','other')])
def test_any_intervening_native_boundary_excludes_chunk(side,key,value):
    rows,audit,meta=inputs(21)
    audit[16][side][key]=value
    assert len(module().assemble_samples(rows,audit,meta)['states'])==0


def test_owner_transient_and_incomplete_never_pad():
    rows,audit,meta=inputs(21)
    audit[16]['control_owner']='L3'
    assert len(module().assemble_samples(rows,audit,meta)['states'])==0
    rows,audit,meta=inputs(20)
    assert len(module().assemble_samples(rows,audit,meta)['states'])==0


@pytest.mark.parametrize('kind',['missing_tick','clock','tick_error','violation','measured_action','nan'])
def test_corrupt_stream_fails_closed(kind):
    rows,audit,meta=inputs()
    if kind=='missing_tick': audit.pop(4)
    elif kind=='clock': audit[4]['time_s']+=.0001
    elif kind=='tick_error': audit[4]['tick_error']='failed'
    elif kind=='violation': audit[4]['physical_violations']=['collision']
    elif kind=='measured_action': del rows[0]['q_command']
    else: rows[0]['q_command'][0]=np.nan
    with pytest.raises(ValueError): module().assemble_samples(rows,audit,meta)


def test_frozen_membership_quarantine_and_test_rejected():
    for split,subject in [('train','p000452'),('validation','p000471'),('train','p000441'),('test','p000438')]:
        with pytest.raises(ValueError): module().require_membership(split,subject)
    module().require_membership('train','p000439')
    module().require_membership('validation','p000441')


def test_test_manifest_path_is_never_opened(tmp_path):
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(schema='m2_act_dataset_v1',episodes=[dict(split='test',
        subject_id='p000438',episode_id='do-not-open',capture_root='/absent/test-data')])) )
    with pytest.raises(ValueError,match='TRAIN|VALIDATION|train|validation'):
        module().prepare_dataset(path,module().sha256(path))


def test_normalization_only_train_and_q_augmentation_never_reanchors_labels():
    rows,audit,meta=inputs()
    samples=module().assemble_samples(rows,audit,meta)
    stats=module().fit_normalization(samples,split='train')
    with pytest.raises(ValueError,match='TRAIN'): module().fit_normalization(samples,split='validation')
    clean=module().pack_batch(samples,stats,np.arange(2),training=False)
    noisy=module().pack_batch(samples,stats,np.arange(2),training=True,rng=np.random.default_rng(9),q_noise_std=.1,q_dropout=1.)
    np.testing.assert_array_equal(noisy['observation.state'],np.zeros((2,7)))
    np.testing.assert_array_equal(noisy['action'],clean['action'])
    np.testing.assert_array_equal(noisy['observation.environment_state'][:,:10],clean['observation.environment_state'][:,:10])
    assert not clean['action_is_pad'].any()


def test_baselines_and_all_twenty_horizons_are_reported():
    rows,audit,meta=inputs()
    data=module().assemble_samples(rows,audit,meta)
    report=module().evaluate_predictions(data['actions'],data)
    assert report['act']['full_chunk_mae_rad']==0
    assert len(report['act']['per_horizon_mae_rad'])==20
    assert report['hold']['full_chunk_mae_rad']>0
    assert report['constant_velocity']['full_chunk_mae_rad']>0


def test_fixture_data_cannot_enter_fitting(tmp_path):
    rows,audit,meta=inputs()
    data=module().assemble_samples(rows,audit,meta)
    with pytest.raises(ValueError,match='qualification'):
        module().fit(dict(train=data,validation=data,training_eligible=False),{},tmp_path/'fit')


def test_native_config_is_new_h20_e10_state17_architecture():
    cfg=module().native_spec()
    assert cfg['chunk_size']==20 and cfg['n_action_steps']==10
    assert cfg['dim_model']==256 and cfg['n_heads']==4
    assert cfg['n_encoder_layers']==2 and cfg['n_decoder_layers']==1
    assert cfg['use_vae'] is False


def test_real_compact_cpu_fixture_cannot_be_promoted_even_with_claimed_run_gate(tmp_path):
    from test_ausc_policy_compact_expert_capture import make, tick
    env,capture=make(tmp_path,max_frames=24)
    for _ in range(24*12): tick(env,capture)
    capture.close();outer=capture.result(24*12,execution_completed=True)
    root=capture.output
    run=tmp_path/'run.json'
    run.write_text(json.dumps(dict(success=True,physical_validation=True,errors=[],learned_policy=False,
                                  compact_capture_complete=True,raw_multimodal_capture=outer)))
    inventory={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=module().sha256(p))
               for p in root.rglob('*') if p.is_file()}
    meta=json.loads((root/'compact/manifest.json').read_text())['metadata']
    row=dict(split=meta['split'],subject_id=meta['subject_id'],episode_id=meta['episode_id'],
             capture_root=str(root),run_result=dict(path=str(run),sha256=module().sha256(run)),artifacts=inventory)
    with pytest.raises(ValueError,match='CPU fixtures|live Isaac'):
        module()._read_episode(row,tmp_path)


def test_inventory_requires_every_byte_and_rejects_tamper(tmp_path):
    root=tmp_path/'capture';root.mkdir();(root/'a').write_bytes(b'a');(root/'b').write_bytes(b'b')
    inv={name:dict(bytes=1,sha256=module().sha256(root/name)) for name in ('a','b')}
    assert module()._verify_inventory(root,inv)==2
    with pytest.raises(ValueError,match='exhaustive'):module()._verify_inventory(root,{'a':inv['a']})
    (root/'a').write_bytes(b'x')
    with pytest.raises(ValueError,match='hash'):module()._verify_inventory(root,inv)


def test_population_gate_cannot_accept_an_incomplete_positive_only_pool(tmp_path):
    receipt=tmp_path/'qualification.json'
    receipt.write_text(json.dumps(dict(schema='m2_expert_population_qualification_v1',passed=True,
        manifest_sha256='a'*64,minimum_episodes=1,minimum_identities=1,minimum_success_rate=.1)))
    with pytest.raises(ValueError,match='threshold'):
        module()._qualification(dict(path=receipt.name,sha256=module().sha256(receipt)),tmp_path,'a'*64,[])


def qualification_fixture(tmp_path, failures):
    from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
    ids=[v for v in json.loads(contract.IDENTITY_SPLIT_PATH.read_text())['splits']['train']
         if v not in contract.QUARANTINED_IDENTITIES][:20]
    attempts=[];episodes=[]
    for i in range(150+failures):
        success=i<150
        result=tmp_path/('run-'+str(i)+'.json')
        result.write_text(json.dumps(dict(episode_id='e'+str(i),success=success,physical_validation=success,
            errors=[] if success else ['failed contact'],learned_policy=False,
            geometry_id=GEOMETRY_ID,geometry_source_sha256=GEOMETRY_SOURCE_SHA256)))
        reference=dict(path=result.name,sha256=module().sha256(result))
        attempt=dict(attempt_id='attempt-'+str(i),split='train',subject_id=ids[i%20],episode_id='e'+str(i),
                     preclassified_reachable=True,success=success,run_result=reference)
        attempts.append(attempt)
        if success: episodes.append(dict(attempt,windows=1))
    ledger=tmp_path/'attempts.json';ledger.write_text(json.dumps(dict(attempts=attempts)))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(schema='m2_expert_attempt_plan_v1',
        frozen_before_launch=True,attempts=[{k:a[k] for k in ('attempt_id','episode_id','subject_id','split',
        'preclassified_reachable')} for a in attempts])))
    execution=tmp_path/'execution.json';execution.write_text(json.dumps(dict(schema='m2_expert_execution_inventory_v1',
        finalized=True,attempt_plan_sha256=module().sha256(plan),attempts=[dict(attempt_id=a['attempt_id'],
        launched=True,run_result=a['run_result']) for a in attempts])))
    receipt=tmp_path/'qualification.json'
    receipt.write_text(json.dumps(dict(schema='m2_expert_population_qualification_v1',passed=True,
        manifest_sha256='a'*64,minimum_episodes=150,minimum_identities=20,minimum_success_rate=.95,
        attempt_ledger=dict(path=ledger.name,sha256=module().sha256(ledger)),
        attempt_plan=dict(path=plan.name,sha256=module().sha256(plan)),
        execution_inventory=dict(path=execution.name,sha256=module().sha256(execution)))))
    return dict(path=receipt.name,sha256=module().sha256(receipt)),episodes


def test_population_denominator_retains_reachable_failures(tmp_path):
    ref,episodes=qualification_fixture(tmp_path,9)
    with pytest.raises(ValueError,match='denominator'):
        module()._qualification(ref,tmp_path,'a'*64,episodes)


def test_population_cut_counts_only_train_with_complete_h20(tmp_path):
    ref,episodes=qualification_fixture(tmp_path,7)
    result=module()._qualification(ref,tmp_path,'a'*64,episodes)
    assert result['reachable_denominator']==157 and result['reachable_successes']==150
    assert result['qualified_train_identities']==20
    episodes[-1]['windows']=0
    with pytest.raises(ValueError,match='counts'):
        module()._qualification(ref,tmp_path,'a'*64,episodes)


def test_cpu_checkpoint_requires_explicit_fixture_route_and_never_promotes():
    checkpoint=dict(schema=module().SCHEMA,upstream_commit=module().UPSTREAM_COMMIT,
        model=module().native_spec(),selection_split='validation',action_representation='block_relative_joint',
        state_only=True,cpu_fixture_only=True,metadata=metadata(),observation_contract=contract.observation_contract())
    with pytest.raises(ValueError,match='fixture'):
        module().validate_checkpoint(checkpoint)
    module().validate_checkpoint(checkpoint,allow_cpu_fixture=True)
    checkpoint['cpu_fixture_only']=False
    with pytest.raises(ValueError,match='qualification'):
        module().validate_checkpoint(checkpoint)


def rewrite_qualification_inputs(tmp_path, ref, transform):
    receipt=json.loads((tmp_path/ref['path']).read_text())
    ledger=json.loads((tmp_path/receipt['attempt_ledger']['path']).read_text())
    plan=json.loads((tmp_path/receipt['attempt_plan']['path']).read_text())
    inventory=json.loads((tmp_path/receipt['execution_inventory']['path']).read_text())
    transform(ledger,plan,inventory)
    for key,value in [('attempt_ledger',ledger),('attempt_plan',plan)]:
        path=tmp_path/receipt[key]['path'];path.write_text(json.dumps(value))
        receipt[key]['sha256']=module().sha256(path)
    inventory['attempt_plan_sha256']=receipt['attempt_plan']['sha256']
    path=tmp_path/receipt['execution_inventory']['path'];path.write_text(json.dumps(inventory))
    receipt['execution_inventory']['sha256']=module().sha256(path)
    path=tmp_path/ref['path'];path.write_text(json.dumps(receipt))
    return dict(path=path.name,sha256=module().sha256(path))


def test_duplicate_success_alias_cannot_rescue_failed_population_rate(tmp_path):
    ref,episodes=qualification_fixture(tmp_path,9)
    def duplicate(ledger,plan,inventory):
        for i in range(30):
            clone=deepcopy(ledger['attempts'][0]);clone['attempt_id']='alias-'+str(i)
            ledger['attempts'].append(clone)
            plan['attempts'].append({k:clone[k] for k in ('attempt_id','episode_id','subject_id','split','preclassified_reachable')})
            inventory['attempts'].append(dict(attempt_id=clone['attempt_id'],launched=True,run_result=clone['run_result']))
    ref=rewrite_qualification_inputs(tmp_path,ref,duplicate)
    with pytest.raises(ValueError,match='duplicate|Duplicate|identity'):
        module()._qualification(ref,tmp_path,'a'*64,episodes)


@pytest.mark.parametrize('fault',[{'learned_policy':True},{'geometry_id':'retired'},{'geometry_source_sha256':'f'*64}])
def test_nonexpert_or_retired_geometry_cannot_inflate_population_success(tmp_path,fault):
    ref,episodes=qualification_fixture(tmp_path,9)
    def append_distinct(ledger,plan,inventory):
        for i in range(30):
            clone=deepcopy(ledger['attempts'][0]);clone.update(attempt_id='extra-'+str(i),episode_id='other-'+str(i))
            source=json.loads((tmp_path/clone['run_result']['path']).read_text())
            source.update(episode_id=clone['episode_id'],**fault)
            path=tmp_path/('extra-run-'+str(i)+'.json');path.write_text(json.dumps(source))
            clone['run_result']=dict(path=path.name,sha256=module().sha256(path))
            ledger['attempts'].append(clone)
            plan['attempts'].append({k:clone[k] for k in ('attempt_id','episode_id','subject_id','split','preclassified_reachable')})
            inventory['attempts'].append(dict(attempt_id=clone['attempt_id'],launched=True,run_result=clone['run_result']))
    ref=rewrite_qualification_inputs(tmp_path,ref,append_distinct)
    with pytest.raises(ValueError,match='expert on the current geometry'):
        module()._qualification(ref,tmp_path,'a'*64,episodes)


def test_dropped_failure_rejected_against_separate_execution_inventory(tmp_path):
    ref,episodes=qualification_fixture(tmp_path,9)
    ref=rewrite_qualification_inputs(tmp_path,ref,
        lambda ledger,plan,inventory:ledger['attempts'].__delitem__(slice(-2,None)))
    with pytest.raises(ValueError,match='coverage|launched'):
        module()._qualification(ref,tmp_path,'a'*64,episodes)


def test_changed_reachability_rejected_against_frozen_plan(tmp_path):
    ref,episodes=qualification_fixture(tmp_path,9)
    def change(ledger,plan,inventory):
        for attempt in ledger['attempts'][-2:]:attempt['preclassified_reachable']=False
    ref=rewrite_qualification_inputs(tmp_path,ref,change)
    with pytest.raises(ValueError,match='plan|reachability'):
        module()._qualification(ref,tmp_path,'a'*64,episodes)


def test_bound_run_episode_identity_cannot_be_reassigned(tmp_path):
    ref,episodes=qualification_fixture(tmp_path,7)
    receipt=json.loads((tmp_path/ref['path']).read_text())
    ledger=json.loads((tmp_path/receipt['attempt_ledger']['path']).read_text())
    result_path=tmp_path/ledger['attempts'][0]['run_result']['path']
    result=json.loads(result_path.read_text());result['episode_id']='different-native-episode'
    result_path.write_text(json.dumps(result))
    def change(ledger,plan,inventory):
        ledger['attempts'][0]['run_result']['sha256']=module().sha256(result_path)
        inventory['attempts'][0]['run_result']['sha256']=module().sha256(result_path)
    ref=rewrite_qualification_inputs(tmp_path,ref,change)
    episodes[0]['run_result']['sha256']=module().sha256(result_path)
    with pytest.raises(ValueError,match='identity'):
        module()._qualification(ref,tmp_path,'a'*64,episodes)


def test_metadata_split_preflight_precedes_any_payload_hash(tmp_path,monkeypatch):
    from sim.ausc_policy.training_capture import ADAPTIVE_SCHEMA
    root=tmp_path/'test-capture';(root/'compact').mkdir(parents=True)
    meta=metadata();meta.update(split='test',subject_id='p000438')
    path=root/'compact/manifest.json';path.write_text(json.dumps(dict(schema=ADAPTIVE_SCHEMA,metadata=meta)))
    (root/'compact/numeric.bin').write_bytes(b'held-out payload must not be hashed or read')
    (root/'compact/receipt.json').write_text('{}')
    (root/'receipt.json').write_text('{}')
    row=dict(split='train',subject_id='p000439',episode_id=meta['episode_id'],capture_root=str(root),
        artifacts={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=module().sha256(p))
                   for p in root.rglob('*') if p.is_file()})
    opened=[];original=module().sha256
    def track(path):
        opened.append(str(path))
        return original(path)
    monkeypatch.setattr(module(),'sha256',track)
    with pytest.raises(ValueError,match='TRAIN|VALIDATION|identity'):
        module()._read_episode(row,tmp_path)
    assert not any(path.endswith('numeric.bin') for path in opened)
