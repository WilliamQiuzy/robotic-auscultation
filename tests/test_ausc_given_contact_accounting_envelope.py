"""Finite metadata envelopes; CPU fixtures never qualify native collection."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from sim.ausc_policy import m2_given_contact_dataset as dataset
from test_ausc_given_contact_dataset import accounting_fixture, population


def envelope(tmp_path):
    root=tmp_path/'metadata';root.mkdir()
    (root/'plan.json').write_text('{}\n')
    return dict(version='m2_given_operational_metadata_envelope_v1',
        owner_roots=[str(tmp_path/'owner')],extra_input_refs=[],
        operational_metadata=dict(root=str(root),allowed_files=['plan.json','export/index.json'],
            limit_bytes=2147483648,charged_bytes=2147483648),
        final_seal=dict(path=str(tmp_path/'external-seal.json'),limit_bytes=1048576,charged_bytes=1048576))


def test_metadata_envelope_charges_future_budget_separately_from_observed(tmp_path):
    scope=envelope(tmp_path)
    report=dataset.metadata_accounting(scope,tmp_path,snapshot_phase='pre_export')
    assert report['charged_bytes']==2147483648+1048576
    assert report['observed_bytes']==3 and report['seal_observed_bytes']==0
    assert report['snapshot_phase']=='pre_export' and report['observed_at']
    assert report['inode_unique_physical_bytes']>=0


@pytest.mark.parametrize('fault',['cap','seal_cap','charge','unknown','traversal','absolute','duplicate',
    'symlink','hardlink','special','owner_overlap','input_overlap','seal_overlap','overrun','seal_overrun'])
def test_metadata_envelope_refuses_changed_or_unaccountable_scope(tmp_path,fault):
    scope=envelope(tmp_path);meta=scope['operational_metadata'];root=Path(meta['root'])
    if fault=='cap':meta['limit_bytes']+=1
    elif fault=='seal_cap':scope['final_seal']['limit_bytes']+=1
    elif fault=='charge':meta['charged_bytes']-=1
    elif fault=='unknown':(root/'unexpected.log').write_text('x')
    elif fault=='traversal':meta['allowed_files'].append('../out.json')
    elif fault=='absolute':meta['allowed_files'].append(str(tmp_path/'out.json'))
    elif fault=='duplicate':meta['allowed_files'].append('plan.json')
    elif fault=='symlink':(root/'plan.json').unlink();(root/'plan.json').symlink_to(tmp_path/'outside')
    elif fault=='hardlink':os.link(root/'plan.json',tmp_path/'hardlink.json')
    elif fault=='special':(root/'plan.json').unlink();os.mkfifo(root/'plan.json')
    elif fault=='owner_overlap':scope['owner_roots']=[str(root/'owner')]
    elif fault=='input_overlap':scope['extra_input_refs']=[dict(reference=dict(path=str(root/'plan.json')))]
    elif fault=='seal_overlap':scope['final_seal']['path']=str(root/'seal.json')
    elif fault=='overrun':
        with (root/'plan.json').open('r+b') as stream:stream.truncate(2147483649)
    else:
        with Path(scope['final_seal']['path']).open('wb') as stream:stream.truncate(1048577)
    with pytest.raises(ValueError):dataset.metadata_accounting(scope,tmp_path,snapshot_phase='post_export')


def test_population_adds_one_fixed_envelope_to_unique_union(tmp_path):
    plan,rows,inventory,_=accounting_fixture(tmp_path)
    old=dataset._accounting(plan,rows,inventory,tmp_path)
    scope=envelope(tmp_path);scope['owner_roots']=plan['accounting_scope']['owner_roots']
    plan['accounting_scope'].update({k:v for k,v in scope.items() if k not in ('owner_roots','extra_input_refs')})
    new=dataset._accounting(plan,rows,inventory,tmp_path)
    assert new['total_accounted_bytes']-old['total_accounted_bytes']==2148532224
    assert new['metadata_envelope']['observed_bytes']==3
    assert new['all_accounted_jobs_within_budget'] is False


def test_population_rejects_validation_launched_before_first100_decision(tmp_path,monkeypatch):
    ref,episodes=population(tmp_path,monkeypatch)
    original=dataset._population_native
    def early(attempt,entry,base):
        result=original(attempt,entry,base)
        if attempt['split']=='validation':
            result.update(_started_utc='2026-10-26T01:00:00Z',_ended_utc='2026-10-26T02:00:00Z')
        return result
    monkeypatch.setattr(dataset,'_population_native',early)
    with pytest.raises(ValueError,match='first100'):dataset._verify_population(ref,episodes)


def test_failed_collection_accounts_launched_subset_and_all_frozen_inputs(tmp_path):
    plan,rows,inventory,_=accounting_fixture(tmp_path)
    absent=tmp_path/'never-launched-owner'
    plan['accounting_scope']['owner_roots'].append(str(absent))
    plan['accounting_scope']['version']=dataset.ACCOUNTING_SCHEMA
    scope=envelope(tmp_path)
    plan['accounting_scope'].update({k:v for k,v in scope.items() if k not in ('owner_roots','extra_input_refs')})
    inventory[rows[1]['attempt_id']]=dict(launched=False)
    # An unlaunched owner's existing files are never silently treated as zero.
    __import__('shutil').rmtree(tmp_path/'owner1')
    result=dataset._accounting(plan,rows,inventory,tmp_path,snapshot_phase='reconcile')
    assert result['unique_union']['observed_jobs']==1
    assert result['planned_owner_count']==3 and result['launched_owner_count']==1
    assert result['additional_input_bytes']>=sum(Path(r['config']['path']).stat().st_size for r in rows)
    absent.mkdir();(absent/'unexpected.log').write_text('not launched')
    with pytest.raises(ValueError,match='Unlaunched owner'):dataset._accounting(plan,rows,inventory,tmp_path,snapshot_phase='reconcile')


def test_zero_job_terminal_failure_reports_cost_without_fake_ledger_or_denominator(tmp_path,monkeypatch):
    from sim.ausc_policy import scene_store
    from test_ausc_given_contact_dataset import save
    owner=tmp_path/'owner';output=owner/'output';output.mkdir(parents=True)
    (owner/'owner.log').write_bytes(b'operational cost')
    terminal=save(owner/'result.json',dict(gpu_released=True,ended_utc='2026-10-27T01:00:00Z',
        stage=dict(passed=False,exit_code=-2,process_group_drained=True)))
    launch=save(output/'job.launch.json',dict(id='job',cpu_fixture_only=True))
    cfg=save(tmp_path/'config.json',dict(cpu_fixture_only=True));cfg['bytes']=Path(cfg['path']).stat().st_size
    runtime=tmp_path/'existing-runtime.so';runtime.write_bytes(b'explicit CPU runtime fixture')
    __import__('os').link(runtime,tmp_path/'package-cache-copy.so')
    runtime_ref=dict(path=str(runtime),bytes=runtime.stat().st_size,sha256=dataset.sha256(runtime))
    sdk=tmp_path/'existing-sdk.mdl';sdk.write_bytes(b'authenticated CPU SDK fixture')
    __import__('os').link(sdk,tmp_path/'sdk-package-cache-copy.mdl')
    sdk_ref=dict(path=str(sdk),bytes=sdk.stat().st_size,sha256=dataset.sha256(sdk))
    rows=[dict(attempt_id='a',config=cfg,source_freeze=None,job_id='job')]
    entries=dict(a=dict(launched=True,owner_terminal=terminal,launch_receipt=launch))
    scope=envelope(tmp_path);scope['owner_roots']=[str(owner)]
    scope['extra_input_refs']=[dict(category='config',reference=cfg),dict(category='other_runtime',reference=runtime_ref),
        dict(category='sdk',reference=sdk_ref)]
    scope.update(tier_b_attempt_ids=[],storage_roots=[dict(category=c,path=str(owner))
        for c in ('complete_cpu_bank','backup_archive','provisioned_runtime')])
    original=scene_store.batch_ledger
    monkeypatch.setattr(scene_store,'batch_ledger',lambda *a,**kw:pytest.fail('Invented an observed job for the original ledger'))
    report=dataset._accounting(dict(accounting_scope=scope),rows,entries,tmp_path,snapshot_phase='reconcile')
    assert report['observed_jobs']==0 and report['accounted_bytes_per_observed_job'] is None
    assert report['owner_operational_bytes']==sum(p.stat().st_size for p in owner.rglob('*') if p.is_file())
    assert report['additional_input_bytes']==cfg['bytes']+runtime_ref['bytes']+sdk_ref['bytes']
    assert report['all_accounted_jobs_within_budget'] is False
    assert report['qualification'] is False and report['complete_shared_union_available'] is False
    # A crash can also leave a bare job directory before either scene_ref or
    # FAILED is written. Keep the old ledger refusal and report known costs.
    (output/'job').mkdir();monkeypatch.setattr(scene_store,'batch_ledger',original)
    report=dataset._accounting(dict(accounting_scope=scope),rows,entries,tmp_path,snapshot_phase='reconcile')
    assert report['observed_jobs']==1 and report['qualification'] is False
    assert 'scene_ref' in report['original_scene_ledger_error']
    scope['extra_input_refs'][1]['category']='source'
    with pytest.raises(ValueError,match='failure-accounting input'):
        dataset._accounting(dict(accounting_scope=scope),rows,entries,tmp_path,snapshot_phase='reconcile')
    scope['extra_input_refs'][1]['category']='other_runtime';sdk.write_bytes(b'changed installed SDK bytes')
    with pytest.raises(ValueError,match='failure-accounting input'):
        dataset._accounting(dict(accounting_scope=scope),rows,entries,tmp_path,snapshot_phase='reconcile')
