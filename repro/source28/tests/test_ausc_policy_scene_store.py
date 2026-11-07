"""CPU storage/provenance tests; synthetic TRAIN metadata, no simulator."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import stat

import pytest


def module():
    assert importlib.util.find_spec('sim.ausc_policy.scene_store') is not None, 'Missing versioned scene store'
    return importlib.import_module('sim.ausc_policy.scene_store')


def fixture(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    def save(name,data):
        path=prepared/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data if isinstance(data,bytes) else json.dumps(data).encode())
        return dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    scene=save('candidate_physics.usdc',b'synthetic flattened scene')
    patient=save('patient_receipt.json',dict(identity='synthetic_train',split='train',non_patient_scene_unchanged=True))
    geometry=save('geometry/collision_geometry.json',{})
    arrays=save('geometry/collision_geometry.npz',b'synthetic collision arrays')
    artifacts=dict(collision_geometry_sha256=geometry['sha256'],collision_arrays_sha256=arrays['sha256'])
    prep_config=dict(prepare_only=True,geometry_source='original.usdc',geometry_source_sha256='a'*64,
        patient_asset=dict(expected_split='train'),physics=dict(gravity=9.81),component_priors={},
        posture_limits={},fresh_expert=dict(passive_preload_N=1.5))
    prep=save('prepare.json',prep_config)
    complete=save('COMPLETE.json',dict(completed=True,geometry_id='synthetic',source_sha256='a'*64,
        patient=json.loads(Path(patient['file']).read_text()),geometry_artifacts=artifacts,
        effective_scene_sha256=scene['sha256'],world_from_left_base=[]))
    save('mass_estimates.json',dict(mass_kg=.067))
    save('physics_profile.json',{})
    save('approved_camera_restoration.json',{})
    config=deepcopy(prep_config);config.pop('prepare_only')
    config['fresh_expert'].update(plans=[],raw_multimodal_capture=dict(enabled=True,profile='compact',episode_id='episode1'))
    return dict(preparation_config=prep,prepared_complete=complete,patient_receipt=patient),config


def bound(tmp_path):
    refs,config=fixture(tmp_path)
    binding=module().build_binding(**refs,store_root=tmp_path/'cas',batch_id='batch1')
    config['compact_scene_storage']=binding
    return binding,config


def test_publish_deduplicates_scene_bytes_and_writes_only_small_refs(tmp_path):
    binding,config=bound(tmp_path);m=module()
    first=m.prepare_job(config,tmp_path/'job1');second=m.prepare_job(config,tmp_path/'job2')
    assert first['scene']==second['scene']
    source=m.resolve_scene(first)
    assert source.read_bytes()==b'synthetic flattened scene'
    assert source.stat().st_mode & 0o222 == 0
    assert source.stat().st_nlink==1
    assert not (tmp_path/'job1/candidate_physics.usdc').exists()
    assert not (tmp_path/'job1/fresh_expert/training_scene.usdc').exists()
    assert (tmp_path/'job1/scene_ref.json').stat().st_size<10_000
    assert list((tmp_path/'cas/objects').rglob('*.usdc'))==[source]
    assert json.loads((tmp_path/'job1/patient_receipt.json').read_text())['split']=='train'


def test_concurrent_publish_never_uses_hardlink_aliases(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import os
    binding,config=bound(tmp_path);m=module()
    def forbidden(*args):raise AssertionError('Publishing shared scenes through hardlink aliases is forbidden')
    monkeypatch.setattr(os,'link',forbidden)
    with ThreadPoolExecutor(max_workers=8) as workers:
        refs=list(workers.map(lambda index:m.prepare_job(config,tmp_path/f'job{index}'),range(16)))
    assert len({str(m.resolve_scene(ref)) for ref in refs})==1
    assert m.resolve_scene(refs[0]).stat().st_nlink==1


@pytest.mark.parametrize('kind',['config','patient','geometry','scene','physics','preload','unknown_schema','full_raw'])
def test_binding_rejects_source_mutation_and_runtime_authoring_drift(tmp_path,kind):
    binding,config=bound(tmp_path);m=module()
    if kind in ('config','patient','geometry','scene'):
        name={'config':'preparation_config','patient':'patient_receipt','geometry':'collision_geometry','scene':'scene'}[kind]
        Path(binding['sources'][name]['file']).write_bytes(b'changed')
    elif kind=='physics':config['physics']['gravity']=0
    elif kind=='preload':config['fresh_expert']['passive_preload_N']=2
    elif kind=='unknown_schema':binding['schema']='unknown'
    else:config['fresh_expert']['raw_multimodal_capture']['profile']='full_raw'
    with pytest.raises(ValueError):m.prepare_job(config,tmp_path/'job')


@pytest.mark.parametrize('kind',['hash','size','traversal','absolute','symlink','writable','mutation'])
def test_resolver_rejects_corrupt_or_escaping_shared_scene(tmp_path,kind):
    binding,config=bound(tmp_path);m=module();ref=m.prepare_job(config,tmp_path/'job')
    source=m.resolve_scene(ref)
    if kind=='hash':ref['scene']['sha256']='0'*64
    elif kind=='size':ref['scene']['bytes']+=1
    elif kind=='traversal':ref['scene']['path']='objects/../outside.usdc'
    elif kind=='absolute':ref['scene']['path']=str(source)
    elif kind=='symlink':
        source.chmod(0o644);source.unlink();source.symlink_to(Path(binding['sources']['scene']['file']))
    elif kind=='writable':source.chmod(0o644)
    else:
        source.chmod(0o644);source.write_bytes(b'mutated');source.chmod(0o444)
    with pytest.raises(ValueError):m.resolve_scene(ref)


def test_batch_ledger_counts_actual_job_trees_and_shared_scene_once(tmp_path):
    binding,config=bound(tmp_path);m=module();jobs=[]
    for index in range(2):
        job=tmp_path/f'job{index}';m.prepare_job(config,job);(job/'extra.bin').write_bytes(b'x'*101);jobs.append(job)
    ledger=m.batch_ledger(jobs)
    actual=sum(p.stat().st_size for root in jobs for p in root.rglob('*') if p.is_file())
    assert ledger['job_directory_bytes']==actual
    assert ledger['shared_scene_bytes']==len(b'synthetic flattened scene')
    assert ledger['retained_prepared_scene_bytes']==len(b'synthetic flattened scene')
    assert ledger['total_accounted_bytes']==sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())
    assert ledger['observed_jobs']==2 and ledger['shared_scene_amortized_bytes_per_job']==25.
    assert ledger['all_accounted_jobs_within_budget'] is True
    assert all(row['accounted_bytes_with_batch_amortization']==row['total_bytes']+ledger['shared_input_bytes']/2 for row in ledger['jobs'])
    (jobs[0]/'hidden.bin').write_bytes(b'x'*100_000_000)
    ledger=m.batch_ledger(jobs)
    assert not ledger['jobs'][0]['within_budget']
    assert ledger['jobs'][0]['total_bytes']>100_000_000


def test_batch_ledger_counts_owner_logs_and_receipts_outside_job_trees(tmp_path):
    _,config=bound(tmp_path);m=module();batch=tmp_path/'batch';job=batch/'job1'
    m.prepare_job(config,job);(batch/'job1.log').write_bytes(b'captured owner log')
    ledger=m.batch_ledger([job],batch_root=batch)
    assert ledger['batch_overhead_bytes']==len(b'captured owner log')
    assert ledger['batch_directory_bytes']==sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())
    assert ledger['total_accounted_bytes']==sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())


def test_batch_ledger_charges_every_external_bound_input_once_by_real_path(tmp_path):
    binding,config=bound(tmp_path);m=module();batch=tmp_path/'batch';jobs=[]
    for index in range(2):
        job=batch/f'job{index}';m.prepare_job(config,job);jobs.append(job)
    ledger=m.batch_ledger(jobs,batch_root=batch)
    physical=sum(path.stat().st_size for path in tmp_path.rglob('*') if path.is_file())
    assert ledger['total_accounted_bytes']==physical
    unique={ref['file']:ref['bytes'] for ref in binding['sources'].values()}
    assert ledger['retained_prepared_input_bytes']==sum(unique.values())
    assert ledger['shared_input_bytes']==sum(unique.values())+ledger['shared_scene_bytes']
    assert len({row['file'] for row in ledger['shared_inputs']})==len(ledger['shared_inputs'])
    assert ledger['shared_input_inventory_complete'] is True


def test_batch_ledger_keeps_early_failed_attempt_without_scene_reference(tmp_path):
    _,config=bound(tmp_path);m=module();batch=tmp_path/'batch';good=batch/'good';failed=batch/'failed'
    m.prepare_job(config,good);failed.mkdir();(failed/'FAILED.json').write_text('{"error":"publish failed"}')
    ledger=m.batch_ledger([good,failed],batch_root=batch)
    assert ledger['observed_jobs']==2
    assert ledger['jobs'][1]['scene_reference_status']=='absent_early_failure'
    assert ledger['jobs'][1]['total_bytes']==(failed/'FAILED.json').stat().st_size
    assert ledger['shared_input_inventory_complete'] is False
    assert ledger['all_accounted_jobs_within_budget'] is False
    assert ledger['total_accounted_bytes']==sum(path.stat().st_size for path in tmp_path.rglob('*') if path.is_file())


def test_all_early_failed_attempts_return_honest_partial_inventory(tmp_path):
    m=module();batch=tmp_path/'batch';failed=batch/'failed';failed.mkdir(parents=True)
    (failed/'FAILED.json').write_text('{"error":"no scene published"}')
    ledger=m.batch_ledger([failed],batch_root=batch)
    assert ledger['observed_jobs']==1 and ledger['batch_id'] is None
    assert ledger['shared_input_inventory_complete'] is False
    assert ledger['shared_input_bytes']==0
    assert ledger['total_accounted_bytes']==(failed/'FAILED.json').stat().st_size
    assert ledger['all_accounted_jobs_within_budget'] is False


def test_final_ledger_counts_its_own_bytes_and_detects_mutation(tmp_path):
    _,config=bound(tmp_path);m=module();job=tmp_path/'job';ref=m.prepare_job(config,job)
    result=dict(success=True,compact_capture_complete=True)
    m.finalize_job(job,result)
    actual=sum(p.stat().st_size for p in job.rglob('*') if p.is_file())
    assert result['scene_storage']['job_directory_bytes']==actual
    assert result['scene_storage']['cas_unchanged'] is True
    assert result['scene_storage']['batch_amortized_budget_evaluated'] is False
    assert json.loads((job/'COMPLETE.json').read_text())==result
    source=m.resolve_scene(ref);source.chmod(0o644);source.write_bytes(b'changed');source.chmod(0o444)
    with pytest.raises(ValueError):m.finalize_job(job,result)


def test_finalization_replaces_stale_success_receipt_after_cas_mutation(tmp_path):
    _,config=bound(tmp_path);m=module();job=tmp_path/'job';ref=m.prepare_job(config,job)
    result=dict(success=True,compact_capture_complete=True,physical_validation=True)
    m.finalize_job(job,result)
    source=m.resolve_scene(ref);source.chmod(0o644);source.write_bytes(b'changed');source.chmod(0o444)
    with pytest.raises(ValueError):m.finalize_job(job,result)
    stored=json.loads((job/'COMPLETE.json').read_text())
    assert stored['success'] is False and stored['compact_capture_complete'] is False
    assert stored['scene_storage']['cas_unchanged'] is False
    assert stored['scene_storage']['job_directory_bytes']==sum(p.stat().st_size for p in job.rglob('*') if p.is_file())


def test_byte_cap_covers_job_siblings_outside_compact_capture(tmp_path):
    _,config=bound(tmp_path);m=module();job=tmp_path/'job';m.prepare_job(config,job)
    with (job/'large_sibling.bin').open('wb') as stream:stream.truncate(100_000_001)
    result=dict(success=True,compact_capture_complete=True,physical_validation=True)
    m.finalize_job(job,result)
    assert result['compact_capture_complete'] is False
    assert result['scene_storage']['within_budget'] is False
    assert result['scene_storage']['job_directory_bytes']>100_000_000


def test_ledger_does_not_count_retained_prepared_input_twice_when_it_is_in_job_tree(tmp_path):
    _,config=bound(tmp_path);m=module();job=tmp_path/'job';ref=m.prepare_job(config,job)
    # Model a retained input under the measured job tree without changing its bytes.
    original=Path(ref['binding']['sources']['scene']['file']);inside=job/'retained_input.usdc';inside.write_bytes(original.read_bytes())
    ref['binding']['sources']['scene']['file']=str(inside)
    (job/'scene_ref.json').write_text(json.dumps(ref))
    ledger=m.batch_ledger([job])
    assert ledger['retained_prepared_scene_bytes']==0
    assert ledger['total_accounted_bytes']==ledger['job_directory_bytes']+ledger['shared_input_bytes']


def test_replay_source_resolves_legacy_and_explicit_reference(tmp_path):
    m=module();legacy=tmp_path/'legacy';legacy.mkdir();(legacy/'training_scene.usdc').write_bytes(b'old')
    assert m.replay_scene(legacy)==legacy/'training_scene.usdc'
    _,config=bound(tmp_path);ref=m.prepare_job(config,tmp_path/'job')
    assert m.replay_scene(tmp_path/'job')==m.resolve_scene(ref)
    (tmp_path/'job/training_scene.usdc').write_bytes(b'conflicting')
    with pytest.raises(ValueError,match='ambiguous'):m.replay_scene(tmp_path/'job')


def test_anonymous_workers_edit_export_and_save_without_touching_cas(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Sdf,Usd
    refs,config=fixture(tmp_path);source=Path(refs['prepared_complete']['file']).parent/'candidate_physics.usdc'
    layer=Sdf.Layer.CreateNew(str(source.with_suffix('.usda')))
    stage=Usd.Stage.Open(layer);stage.DefinePrim('/World')
    stage.GetPrimAtPath('/World').CreateAttribute('mdl',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('OmniPBR.mdl'))
    stage.GetRootLayer().Export(str(source))
    complete_path=Path(refs['prepared_complete']['file']);complete=json.loads(complete_path.read_text())
    complete['effective_scene_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    complete_path.write_text(json.dumps(complete));refs['prepared_complete']['sha256']=hashlib.sha256(complete_path.read_bytes()).hexdigest()
    m=module();config['compact_scene_storage']=m.build_binding(**refs,store_root=tmp_path/'cas',batch_id='batch1')
    ref=m.prepare_job(config,tmp_path/'job');cas=m.resolve_scene(ref);initial=cas.read_bytes()
    first=m.runtime_stage(ref);second=m.runtime_stage(ref)
    assert first.GetPrimAtPath('/World').GetAttribute('mdl').Get().path=='OmniPBR.mdl'
    assert first.GetRootLayer().anonymous and first.GetSessionLayer().anonymous
    assert first.GetRootLayer().identifier!=second.GetRootLayer().identifier
    first.DefinePrim('/WorkerPrivate');first.GetRootLayer().Export(str(tmp_path/'disposable.usda'))
    with pytest.raises(Exception):first.GetRootLayer().Save()
    assert not second.GetPrimAtPath('/WorkerPrivate')
    assert cas.read_bytes()==initial
    assert m.resolve_scene(ref)==cas


@pytest.mark.parametrize('method',['default_search_path','resolver_context'])
def test_runtime_rejects_builtin_mdl_resolved_from_unbound_search_path(tmp_path,method):
    pytest.importorskip('pxr')
    from pxr import Ar,Sdf,Usd
    refs,config=fixture(tmp_path);directory=Path(refs['prepared_complete']['file']).parent
    authored=Usd.Stage.CreateInMemory();world=authored.DefinePrim('/World')
    world.CreateAttribute('mdl',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('OmniPBR.mdl'))
    source=directory/'candidate_physics.usdc';authored.GetRootLayer().Export(str(source))
    complete_path=Path(refs['prepared_complete']['file']);complete=json.loads(complete_path.read_text())
    complete['effective_scene_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    complete_path.write_text(json.dumps(complete));refs['prepared_complete']['sha256']=hashlib.sha256(complete_path.read_bytes()).hexdigest()
    m=module();config['compact_scene_storage']=m.build_binding(**refs,store_root=tmp_path/'cas',batch_id='shadow')
    ref=m.prepare_job(config,tmp_path/'job');initial=source.read_bytes();cas=m.resolve_scene(ref).read_bytes()
    other=tmp_path/'unrelated-search-directory';other.mkdir();shadow=other/'OmniPBR.mdl'
    shadow.write_bytes(b'unbound resolver-search-path shader')
    def rejected():
        assert str(Ar.GetResolver().Resolve('OmniPBR.mdl'))==str(shadow)
        with pytest.raises(ValueError,match='dependency'):m.runtime_stage(ref)
    if method=='default_search_path':
        Ar.DefaultResolver.SetDefaultSearchPath([str(other)])
        try:rejected()
        finally:Ar.DefaultResolver.SetDefaultSearchPath([])
    else:
        with Ar.ResolverContextBinder(Ar.DefaultResolverContext([str(other)])):rejected()
    assert source.read_bytes()==initial and m.resolve_scene(ref).read_bytes()==cas


@pytest.mark.parametrize('kind',['reference','sublayer','texture','local_builtin_shadow'])
def test_runtime_rejects_unbound_file_dependencies_before_composing(tmp_path,kind):
    pytest.importorskip('pxr')
    from pxr import Sdf,Usd,UsdGeom
    refs,config=fixture(tmp_path);directory=Path(refs['prepared_complete']['file']).parent
    assets=directory/'assets';assets.mkdir();texture=assets/'surface.png';texture.write_bytes(b'synthetic texture bytes')
    model=Usd.Stage.CreateNew(str(assets/'model.usda'));UsdGeom.Cube.Define(model,'/Model')
    model.GetPrimAtPath('/Model').CreateAttribute('texture',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('surface.png'))
    model.GetRootLayer().Save()
    sub=Usd.Stage.CreateNew(str(assets/'sub.usda'));UsdGeom.Sphere.Define(sub,'/SubLayerSphere');sub.GetRootLayer().Save()
    authored=Usd.Stage.CreateNew(str(directory/'author.usda'));world=authored.DefinePrim('/World')
    if kind=='reference':world.GetReferences().AddReference('assets/model.usda','/Model')
    elif kind=='sublayer':authored.GetRootLayer().subLayerPaths=['assets/sub.usda']
    elif kind=='texture':world.CreateAttribute('texture',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('assets/surface.png'))
    else:
        (directory/'OmniPBR.mdl').write_bytes(b'unbound local shader')
        world.CreateAttribute('mdl',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('OmniPBR.mdl'))
    source=directory/'candidate_physics.usdc';authored.GetRootLayer().Export(str(source))
    complete_path=Path(refs['prepared_complete']['file']);complete=json.loads(complete_path.read_text())
    complete['effective_scene_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    complete_path.write_text(json.dumps(complete));refs['prepared_complete']['sha256']=hashlib.sha256(complete_path.read_bytes()).hexdigest()
    m=module();config['compact_scene_storage']=m.build_binding(**refs,store_root=tmp_path/'cas',batch_id='anchored')
    ref=m.prepare_job(config,tmp_path/'job');initial=source.read_bytes();cas=m.resolve_scene(ref).read_bytes()
    with pytest.raises(ValueError,match='dependency|dependencies'):m.runtime_stage(ref)
    if kind=='reference':
        UsdGeom.Cube(model.GetPrimAtPath('/Model')).GetSizeAttr().Set(2.)
        model.GetRootLayer().Save()
        with pytest.raises(ValueError,match='dependency|dependencies'):m.runtime_stage(ref)
    assert source.read_bytes()==initial and m.resolve_scene(ref).read_bytes()==cas


def test_dependency_policy_is_explicit_in_scene_reference(tmp_path):
    binding,config=bound(tmp_path);m=module();ref=m.prepare_job(config,tmp_path/'job')
    assert binding['dependency_policy']=='builtin_omnipbr_only_v1'
    ref['binding']['dependency_policy']='allow_unbound_files'
    with pytest.raises(ValueError):m.resolve_scene(ref)


@pytest.mark.parametrize('replacement',[{},None,dict(scene={}),dict(scene={'file':'relative','sha256':'z','bytes':-1})])
def test_resolver_rejects_malformed_source_inventory_with_value_error(tmp_path,replacement):
    _,config=bound(tmp_path);m=module();ref=m.prepare_job(config,tmp_path/'job')
    ref['binding']['sources']=replacement
    with pytest.raises(ValueError):m.resolve_scene(ref)


def test_native_attach_waits_for_verified_callback_and_same_anonymous_root(tmp_path,monkeypatch):
    pytest.importorskip('pxr')
    from pxr import Usd
    from types import SimpleNamespace
    m=module();stage=Usd.Stage.CreateInMemory();context=SimpleNamespace()
    pending=[]
    def attach(stage_id,callback):
        from pxr import UsdUtils
        assert UsdUtils.StageCache.Get().Find(Usd.StageCache.Id.FromLongInt(stage_id))==stage
        pending.append(callback);return True
    context.attach_stage_with_callback=attach;context.get_stage=lambda:stage
    def update():pending.pop()(True,'')
    assert m.attach_runtime_stage(stage,context,update)==stage
    context.get_stage=lambda:Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError,match='root'):m.attach_runtime_stage(stage,context,update)
    def fail_update():pending.pop()(False,'fixture attach failure')
    with pytest.raises(ValueError,match='failure'):m.attach_runtime_stage(stage,context,fail_update)
    from pxr import UsdUtils
    assert UsdUtils.StageCache.Get().Contains(stage)
    m.release_runtime_stage(stage)


def test_prepared_branch_enters_no_usd_export_or_authoring(tmp_path,monkeypatch):
    _,config=bound(tmp_path);m=module()
    sentinel=object();monkeypatch.setattr(m,'runtime_stage',lambda ref:sentinel)
    stage,mass=m.prepared_runtime(config,tmp_path/'job')
    assert stage is sentinel and mass['mass_kg']==.067
    assert not (tmp_path/'job/candidate_physics.usdc').exists()


def test_geometry_entrypoint_uses_prepared_opt_in_and_final_whole_job_inventory(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace,ModuleType
    from sim.ausc_policy import geometry_qualification as entry,fresh_expert,native_lifecycle
    _,config=bound(tmp_path);m=module();sentinel=object();closed=[]
    monkeypatch.setattr(m,'runtime_stage',lambda ref:sentinel)
    monkeypatch.setattr(entry,'require_current',lambda value:None)
    monkeypatch.setattr(entry,'preparation_backend',lambda *args:'fixture_owned')
    fake=ModuleType('isaacsim')
    monkeypatch.setattr(native_lifecycle,'quiesce_before_close',lambda app,**kwargs:closed.append('quiesced') or {})
    def create_app(cfg):
        assert cfg==dict(headless=True,fast_shutdown=False,disable_viewport_updates=True,limit_cpu_threads=6)
        return SimpleNamespace(close=lambda:closed.append(True))
    fake.SimulationApp=create_app
    monkeypatch.setitem(sys.modules,'isaacsim',fake)
    def probe(app,stage,output,actual,mass):
        assert stage is sentinel and actual==config and mass['mass_kg']==.067
        (output/'probe_diagnostic.json').write_text('{}')
        return dict(success=True,compact_capture_complete=True)
    monkeypatch.setattr(fresh_expert,'run_probe',probe)
    path=tmp_path/'runtime.json';path.write_text(json.dumps(config));output=tmp_path/'job'
    monkeypatch.setattr(sys,'argv',['geometry_qualification','--config',str(path),'--output',str(output)])
    assert entry.main() is True
    result=json.loads((output/'COMPLETE.json').read_text())
    assert result['scene_storage']['job_directory_bytes']==sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    assert result['training_scene_sha256']==config['compact_scene_storage']['sources']['scene']['sha256']
    assert result['runtime_shutdown']['app_close_returned'] is True
    assert result['runtime_shutdown']['configured_limit_cpu_threads']==6
    assert not (output/'candidate_physics.usdc').exists() and closed==['quiesced',True]


@pytest.mark.parametrize('shared_complete',[False,True])
def test_native_entrypoint_bypasses_finalizers_only_after_shared_finalization(monkeypatch,shared_complete):
    from sim.ausc_policy import geometry_qualification as entry
    calls=[]
    monkeypatch.setattr(entry,'main',lambda:calls.append('main_returned') or shared_complete)
    monkeypatch.setattr(entry.os,'_exit',lambda code:calls.append(('exit',code)))
    entry.native_entrypoint()
    assert calls==(['main_returned',('exit',0)] if shared_complete else ['main_returned'])


def test_native_entrypoint_does_not_turn_main_exception_into_success(monkeypatch):
    from sim.ausc_policy import geometry_qualification as entry
    calls=[]
    def fail():raise RuntimeError('worker failed')
    monkeypatch.setattr(entry,'main',fail)
    monkeypatch.setattr(entry.os,'_exit',lambda code:calls.append(code))
    with pytest.raises(RuntimeError,match='worker failed'):entry.native_entrypoint()
    assert not calls


def test_geometry_failure_receipt_counts_failed_json_after_shutdown(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace,ModuleType
    from sim.ausc_policy import geometry_qualification as entry,fresh_expert,native_lifecycle
    _,config=bound(tmp_path);m=module();closed=[]
    monkeypatch.setattr(m,'runtime_stage',lambda ref:object())
    monkeypatch.setattr(entry,'require_current',lambda value:None)
    monkeypatch.setattr(entry,'preparation_backend',lambda *args:'fixture_owned')
    fake=ModuleType('isaacsim')
    monkeypatch.setattr(native_lifecycle,'quiesce_before_close',lambda app,**kwargs:{})
    def create_app(cfg):
        assert cfg==dict(headless=True,fast_shutdown=False,disable_viewport_updates=True,limit_cpu_threads=6)
        return SimpleNamespace(close=lambda:closed.append(True))
    fake.SimulationApp=create_app
    monkeypatch.setitem(sys.modules,'isaacsim',fake)
    def fail(*args):raise ValueError('fixture runtime failure')
    monkeypatch.setattr(fresh_expert,'run_probe',fail)
    path=tmp_path/'runtime.json';path.write_text(json.dumps(config));output=tmp_path/'job'
    monkeypatch.setattr(sys,'argv',['geometry_qualification','--config',str(path),'--output',str(output)])
    with pytest.raises(ValueError,match='fixture runtime failure'):entry.main()
    assert (output/'FAILED.json').exists() and closed==[True]
    receipt=json.loads((output/'COMPLETE.json').read_text())
    assert receipt['success'] is False and receipt['compact_capture_complete'] is False
    assert receipt['scene_storage']['job_directory_bytes']==sum(p.stat().st_size for p in output.rglob('*') if p.is_file())


@pytest.mark.parametrize('failure',['quiesce','close'])
def test_shared_cleanup_failure_never_writes_complete_or_retries_close(tmp_path,monkeypatch,failure):
    import sys
    from types import SimpleNamespace,ModuleType
    from sim.ausc_policy import geometry_qualification as entry,fresh_expert,native_lifecycle
    _,config=bound(tmp_path);m=module();calls=[]
    monkeypatch.setattr(m,'runtime_stage',lambda ref:object())
    monkeypatch.setattr(entry,'require_current',lambda value:None)
    monkeypatch.setattr(entry,'preparation_backend',lambda *args:'fixture_owned')
    def quiesce(app,**kwargs):
        calls.append('quiesce')
        if failure=='quiesce':raise RuntimeError('quiesce failed')
        return {}
    def close():
        calls.append('close')
        if failure=='close':raise RuntimeError('close failed')
    monkeypatch.setattr(native_lifecycle,'quiesce_before_close',quiesce)
    fake=ModuleType('isaacsim');fake.SimulationApp=lambda cfg:SimpleNamespace(close=close)
    monkeypatch.setitem(sys.modules,'isaacsim',fake)
    monkeypatch.setattr(fresh_expert,'run_probe',lambda *args:dict(success=True,compact_capture_complete=True))
    path=tmp_path/'runtime.json';path.write_text(json.dumps(config));output=tmp_path/'job'
    monkeypatch.setattr(sys,'argv',['geometry_qualification','--config',str(path),'--output',str(output)])
    with pytest.raises(RuntimeError,match=failure):entry.main()
    assert calls==['quiesce','close'] and (output/'FAILED.json').is_file()
    assert not (output/'COMPLETE.json').exists()


def test_scene_factory_preserves_fullraw_and_uses_independent_compact_root(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from sim.ausc_policy import fresh_expert
    m=module();_,config=bound(tmp_path);ref=m.prepare_job(config,tmp_path/'job');exports=[];clone=object()
    stage=SimpleNamespace(GetRootLayer=lambda:SimpleNamespace(Export=lambda name:exports.append(name)))
    monkeypatch.setattr(m,'runtime_stage',lambda value:clone)
    factory=fresh_expert.scene_factory(stage,None)
    assert factory(tmp_path/'old.usdc') is stage and exports==[str(tmp_path/'old.usdc')]
    exports.clear();factory=fresh_expert.scene_factory(stage,ref);destination=tmp_path/'job/fresh/training_scene.usdc'
    destination.parent.mkdir()
    assert factory(destination) is clone and exports==[] and not destination.exists()
    assert m.replay_scene(destination.parent)==m.resolve_scene(ref)


@pytest.mark.parametrize('name',['sim.auscultation_expert_layout_v1.render_expert','sim.auscultation_training.render_expert'])
def test_renderers_use_scene_reference_resolver_and_track_reference_bytes(tmp_path,name,monkeypatch):
    renderer=importlib.import_module(name);m=module();_,config=bound(tmp_path)
    ref=m.prepare_job(config,tmp_path/'job')
    assert renderer.replay_scene(tmp_path/'job')==m.resolve_scene(ref)
    sentinel=object();monkeypatch.setattr(m,'runtime_stage',lambda value:sentinel)
    assert renderer.replay_stage(tmp_path/'job') is sentinel
    files=[tmp_path/'job/scene_ref.json',m.resolve_scene(ref)]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    renderer.verify_replay_inputs(files,hashes)
    files[0].write_text('{}')
    with pytest.raises(RuntimeError):renderer.verify_replay_inputs(files,hashes)
