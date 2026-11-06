"""Explicit v2 SDK dependency opt-in; synthetic TRAIN scene and CPU USD only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from test_ausc_policy_scene_store import fixture as scene_fixture
from test_ausc_policy_sdk_mdl import sdk_fixture
from sim.ausc_policy import scene_store as store


def bound(tmp_path,monkeypatch,*,usd=False,asset='OmniPBR.mdl',dependency=None):
    sdk,root,_=sdk_fixture(tmp_path,monkeypatch)
    manifest=sdk.prepare_manifest(tmp_path/'sdk-manifest.json')
    refs,config=scene_fixture(tmp_path)
    if usd:
        from pxr import Sdf,Usd
        authored=Usd.Stage.CreateInMemory();world=authored.DefinePrim('/World')
        world.CreateAttribute('mdl',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(asset))
        scene=Path(refs['prepared_complete']['file']).parent/'candidate_physics.usdc'
        if dependency is not None:
            external=scene.parent/'unbound-model.usda';model=Usd.Stage.CreateNew(str(external));model.DefinePrim('/Model')
            model.GetRootLayer().Save()
            if dependency=='reference':world.GetReferences().AddReference(str(external),'/Model')
            else:authored.GetRootLayer().subLayerPaths=[str(external)]
        authored.GetRootLayer().Export(str(scene));path=Path(refs['prepared_complete']['file'])
        complete=json.loads(path.read_bytes());complete['effective_scene_sha256']=hashlib.sha256(scene.read_bytes()).hexdigest()
        path.write_text(json.dumps(complete));refs['prepared_complete']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    config['compact_scene_storage']=store.build_binding(**refs,store_root=tmp_path/'cas',batch_id='v2',sdk_mdl_manifest=manifest)
    return sdk,root,manifest,config


def test_v2_opt_in_binds_every_library_file_and_manifest_byte_in_shared_ledger(tmp_path,monkeypatch):
    sdk,root,manifest,config=bound(tmp_path,monkeypatch);binding=config['compact_scene_storage']
    assert binding['schema']=='m2_compact_scene_storage_v2'
    assert binding['dependency_policy']=='hash_bound_isaacsim60_mdl_v2'
    sdk_sources={key:value for key,value in binding['sources'].items() if key.startswith('sdk_mdl_')}
    assert len(sdk_sources)==sdk.APPROVED_FILE_COUNT+1
    assert sum(row['bytes'] for row in sdk_sources.values())==sdk.APPROVED_TOTAL_BYTES+Path(manifest['file']).stat().st_size
    batch=tmp_path/'batch';jobs=[batch/'first',batch/'second']
    for job in jobs:store.prepare_job(config,job)
    ledger=store.batch_ledger(jobs,batch_root=batch)
    assert ledger['total_accounted_bytes']==sum(path.stat().st_size for path in tmp_path.rglob('*') if path.is_file())
    assert {str(path) for path in root.rglob('*') if path.is_file()}<={row['file'] for row in ledger['shared_inputs']}
    assert not any(path for job in jobs for path in job.rglob('*.usdc'))


@pytest.mark.parametrize('kind',['library_drift','manifest_drift','incomplete_bound_files','bound_file_substitution','extra_library_file'])
def test_v2_preflight_and_postcheck_reject_library_or_bound_inventory_drift(tmp_path,monkeypatch,kind):
    sdk,root,manifest,config=bound(tmp_path,monkeypatch);job=tmp_path/'job';ref=store.prepare_job(config,job)
    if kind=='library_drift':(root/'rtx/helper.mdl').write_bytes(b'mutated SDK dependency')
    elif kind=='manifest_drift':
        path=Path(manifest['file']);path.chmod(0o644);path.write_text('{}')
    elif kind=='incomplete_bound_files':config['compact_scene_storage']['sources'].pop('sdk_mdl_0000')
    elif kind=='bound_file_substitution':
        key=next(k for k in config['compact_scene_storage']['sources'] if k.startswith('sdk_mdl_0'))
        config['compact_scene_storage']['sources'][key]=store.source_ref(root/sdk.ENTRY_RELATIVE)
    else:(root/'unbound-extra.mdl').write_bytes(b'extra')
    with pytest.raises(ValueError):store.validate_config(config)
    if kind in ('library_drift','manifest_drift','extra_library_file'):
        result=dict(success=True,compact_capture_complete=True)
        with pytest.raises(ValueError):store.finalize_job(job,result)
        assert result['compact_capture_complete'] is False
        assert result['scene_storage']['source_bindings_unchanged'] is False


def test_v2_rechecks_sdk_after_final_receipt_publication(tmp_path,monkeypatch):
    _,root,_,config=bound(tmp_path,monkeypatch);job=tmp_path/'job';store.prepare_job(config,job)
    publish=store._inventory_receipts;called=[]
    def publish_then_drift(output,result,storage):
        publish(output,result,storage)
        if not called:
            called.append(True);path=root/'rtx/helper.mdl';data=path.read_bytes();path.write_bytes(b'X'+data[1:])
    monkeypatch.setattr(store,'_inventory_receipts',publish_then_drift)
    result=dict(success=True,compact_capture_complete=True)
    with pytest.raises(ValueError,match='SDK MDL'):store.finalize_job(job,result)
    receipt=json.loads((job/'COMPLETE.json').read_bytes())
    assert receipt['success'] is False and receipt['compact_capture_complete'] is False
    assert receipt['scene_storage']['cas_unchanged'] is True
    assert receipt['scene_storage']['source_bindings_unchanged'] is False
    assert receipt['scene_storage']['job_directory_bytes']==sum(p.stat().st_size for p in job.rglob('*') if p.is_file())


def test_v2_private_stage_rewrites_only_exact_authenticated_entry(tmp_path,monkeypatch):
    pytest.importorskip('pxr')
    from pxr import Ar,Usd,UsdUtils
    sdk,root,manifest,config=bound(tmp_path,monkeypatch,usd=True);ref=store.prepare_job(config,tmp_path/'job')
    cas=store.resolve_scene(ref);original=Path(ref['binding']['sources']['scene']['file'])
    before=(cas.read_bytes(),original.read_bytes());entry=root/sdk.ENTRY_RELATIVE
    cache=UsdUtils.StageCache.Get();cached=list(cache.GetAllStages())
    with Ar.ResolverContextBinder(Ar.DefaultResolverContext([str(entry.parent)])):
        stage=store.runtime_stage(ref)
    value=stage.GetPrimAtPath('/World').GetAttribute('mdl').Get()
    assert value.path==value.resolvedPath==str(entry)
    assert all(layer.anonymous for layer in stage.GetUsedLayers())
    stage.DefinePrim('/WorkerPrivate');stage.GetRootLayer().Export(str(tmp_path/'diagnostic.usda'))
    with pytest.raises(Exception):stage.GetRootLayer().Save()
    assert not Usd.Stage.Open(str(original)).GetPrimAtPath('/WorkerPrivate')
    assert (cas.read_bytes(),original.read_bytes())==before
    assert list(cache.GetAllStages())==cached
    sdk.validate_manifest(manifest)


def test_v2_stage_context_bare_shadow_cannot_replace_selected_absolute_entry(tmp_path,monkeypatch):
    pytest.importorskip('pxr')
    from pxr import Ar,Usd,UsdUtils
    sdk,root,_,config=bound(tmp_path,monkeypatch,usd=True);ref=store.prepare_job(config,tmp_path/'job')
    other=tmp_path/'stage-context-shadow';other.mkdir();shadow=other/'OmniPBR.mdl';shadow.write_bytes(b'unbound stage shader')
    open_stage=Usd.Stage.Open
    def open_with_shadow(layer,*args,**kwargs):
        kwargs['pathResolverContext']=Ar.ResolverContext(Ar.DefaultResolverContext([str(other)]))
        return open_stage(layer,*args,**kwargs)
    monkeypatch.setattr(Usd.Stage,'Open',staticmethod(open_with_shadow))
    cache=UsdUtils.StageCache.Get();cached=list(cache.GetAllStages());cas=store.resolve_scene(ref);before=cas.read_bytes()
    with Ar.ResolverContextBinder(Ar.DefaultResolverContext([str((root/sdk.ENTRY_RELATIVE).parent)])):
        stage=store.runtime_stage(ref)
    assert stage.GetPrimAtPath('/World').GetAttribute('mdl').Get().path==str(root/sdk.ENTRY_RELATIVE)
    assert cas.read_bytes()==before and list(cache.GetAllStages())==cached


def test_v2_rechecks_full_bundle_after_private_stage_creation(tmp_path,monkeypatch):
    pytest.importorskip('pxr')
    from pxr import Ar,Usd
    sdk,root,_,config=bound(tmp_path,monkeypatch,usd=True);ref=store.prepare_job(config,tmp_path/'job')
    open_stage=Usd.Stage.Open
    def drift_during_open(layer,*args,**kwargs):
        stage=open_stage(layer,*args,**kwargs)
        path=root/'rtx/helper.mdl';data=path.read_bytes();path.write_bytes(b'X'+data[1:])
        return stage
    monkeypatch.setattr(Usd.Stage,'Open',staticmethod(drift_during_open))
    with Ar.ResolverContextBinder(Ar.DefaultResolverContext([str((root/sdk.ENTRY_RELATIVE).parent)])):
        with pytest.raises(ValueError,match='SDK MDL'):store.runtime_stage(ref)


@pytest.mark.parametrize('resolved',['','OmniPBR.mdl','omniverse://unapproved/OmniPBR.mdl','file:///unapproved/OmniPBR.mdl'])
def test_v2_rejects_unresolved_relative_and_uri_resolver_results(tmp_path,monkeypatch,resolved):
    pytest.importorskip('pxr')
    from pxr import Ar
    from types import SimpleNamespace
    _,_,_,config=bound(tmp_path,monkeypatch,usd=True);ref=store.prepare_job(config,tmp_path/'job')
    # URI providers are absent in CPU usd-core; inject only their returned path.
    monkeypatch.setattr(Ar,'GetResolver',lambda:SimpleNamespace(Resolve=lambda token:resolved))
    with pytest.raises(ValueError,match='resolution mismatch') as caught:store.runtime_stage(ref)
    assert 'actual_resolved='+repr(resolved) in str(caught.value)


@pytest.mark.parametrize('kind',['default_search','resolver_context','same_hash_test_copy'])
def test_v2_bare_search_provider_is_not_loaded_or_claimed_equivalent(tmp_path,monkeypatch,kind):
    pytest.importorskip('pxr')
    from pxr import Ar,UsdUtils
    sdk,root,_,config=bound(tmp_path,monkeypatch,usd=True);ref=store.prepare_job(config,tmp_path/'job')
    entry=root/sdk.ENTRY_RELATIVE;other=tmp_path/'unapproved-sdk-test-copy';other.mkdir()
    shadow=other/'OmniPBR.mdl';shadow.write_bytes(entry.read_bytes() if kind=='same_hash_test_copy' else b'unapproved')
    cas=store.resolve_scene(ref);before=cas.read_bytes();cache=UsdUtils.StageCache.Get();cached=list(cache.GetAllStages())
    def selected():
        assert str(Ar.GetResolver().Resolve('OmniPBR.mdl'))==str(shadow)
        stage=store.runtime_stage(ref)
        value=stage.GetPrimAtPath('/World').GetAttribute('mdl').Get()
        assert value.path==value.resolvedPath==str(entry)
    if kind=='default_search':
        Ar.DefaultResolver.SetDefaultSearchPath([str(other)])
        try:selected()
        finally:Ar.DefaultResolver.SetDefaultSearchPath([])
    else:
        with Ar.ResolverContextBinder(Ar.DefaultResolverContext([str(other)])):selected()
    assert cas.read_bytes()==before and list(cache.GetAllStages())==cached


@pytest.mark.parametrize('kind',['source_shadow','cwd_shadow'])
def test_v2_still_rejects_local_authored_search_shadows(tmp_path,monkeypatch,kind):
    pytest.importorskip('pxr')
    sdk,root,_,config=bound(tmp_path,monkeypatch,usd=True);ref=store.prepare_job(config,tmp_path/'job')
    if kind=='source_shadow':directory=Path(ref['binding']['sources']['scene']['file']).parent
    else:
        directory=tmp_path/'shadow-cwd';directory.mkdir();monkeypatch.chdir(directory)
    (directory/'OmniPBR.mdl').write_bytes(b'local authored-search shadow')
    with pytest.raises(ValueError,match='local file dependency'):store.runtime_stage(ref)


@pytest.mark.parametrize('asset,dependency',[
    ('textures/surface.png',None),('core/Base/OmniPBR_ClearCoat.mdl',None),
    ('omniverse://unknown/OmniPBR.mdl',None),('/tmp/unbound/OmniPBR.mdl',None),
    ('OmniPBR.mdl','reference'),('OmniPBR.mdl','sublayer')])
def test_v2_does_not_expand_authored_asset_or_composition_policy(tmp_path,monkeypatch,asset,dependency):
    pytest.importorskip('pxr')
    _,_,_,config=bound(tmp_path,monkeypatch,usd=True,asset=asset,dependency=dependency)
    ref=store.prepare_job(config,tmp_path/'job');cas=store.resolve_scene(ref);before=cas.read_bytes()
    with pytest.raises(ValueError,match='dependency|dependencies'):store.runtime_stage(ref)
    assert cas.read_bytes()==before


def test_v2_uri_bare_provider_diagnostic_cannot_override_selected_entry(tmp_path,monkeypatch):
    pytest.importorskip('pxr')
    from pxr import Ar
    from types import SimpleNamespace
    sdk,root,_,config=bound(tmp_path,monkeypatch,usd=True);ref=store.prepare_job(config,tmp_path/'job')
    resolver=Ar.GetResolver()
    monkeypatch.setattr(Ar,'GetResolver',lambda:SimpleNamespace(
        Resolve=lambda path:'omniverse://unknown-provider/OmniPBR.mdl' if path=='OmniPBR.mdl' else resolver.Resolve(path),
        GetCurrentContext=resolver.GetCurrentContext))
    stage=store.runtime_stage(ref)
    assert stage.GetPrimAtPath('/World').GetAttribute('mdl').Get().path==str(root/sdk.ENTRY_RELATIVE)


def test_v2_rejects_selected_entry_redirect_in_final_private_stage_context(tmp_path,monkeypatch):
    pytest.importorskip('pxr')
    from pxr import Ar,Usd
    from types import SimpleNamespace
    sdk,root,_,config=bound(tmp_path,monkeypatch,usd=True);ref=store.prepare_job(config,tmp_path/'job')
    resolver=Ar.GetResolver();open_stage=Usd.Stage.Open;opened=[]
    entry=str(root/sdk.ENTRY_RELATIVE);redirect=str(tmp_path/'unapproved.mdl')
    def open_then_redirect(layer,*args,**kwargs):
        stage=open_stage(layer,*args,**kwargs);opened.append(stage)
        return stage
    def resolved(path):
        if path==entry and len(opened)==2:return redirect
        return resolver.Resolve(path)
    monkeypatch.setattr(Usd.Stage,'Open',staticmethod(open_then_redirect))
    monkeypatch.setattr(Ar,'GetResolver',lambda:SimpleNamespace(
        Resolve=resolved,GetCurrentContext=resolver.GetCurrentContext))
    with pytest.raises(ValueError,match='resolution mismatch') as caught:store.runtime_stage(ref)
    assert 'actual_resolved='+repr(redirect) in str(caught.value)


def test_v2_records_explicit_selection_diagnostics_for_prepared_and_fresh_workers(tmp_path,monkeypatch):
    pytest.importorskip('pxr')
    from pxr import Ar
    from sim.ausc_policy import fresh_expert
    sdk,root,manifest,config=bound(tmp_path,monkeypatch,usd=True);job=tmp_path/'batch/job'
    other=tmp_path/'diagnostic-provider';other.mkdir();shadow=other/'OmniPBR.mdl';shadow.write_bytes(b'unbound provider')
    with Ar.ResolverContextBinder(Ar.DefaultResolverContext([str(other)])):
        initial,_=store.prepared_runtime(config,job)
        ref=json.loads((job/'scene_ref.json').read_bytes());fresh=job/'fresh_expert';fresh.mkdir()
        worker=fresh_expert.scene_factory(initial,ref)(fresh/'training_scene.usdc')
    assert worker.GetRootLayer()!=initial.GetRootLayer()
    for directory in (job,fresh):
        record=json.loads((directory/'sdk_mdl_selection.json').read_bytes())
        assert record['schema']=='m2_sdk_mdl_selection_v2'
        assert record['material_selection']=='explicit_approved_sdk_entry'
        assert record['original_render_equivalence_claim'] is False
        assert record['manifest']['sha256']==manifest['sha256']
        assert record['entry']==store.source_ref(root/sdk.ENTRY_RELATIVE)
        assert record['authored_token']=='OmniPBR.mdl'
        assert record['asset_token_count']==1
        assert record['bare_token_resolved']==dict(caller=str(shadow),composed_stage=str(shadow),private_stage=str(shadow))
        assert record['selected_entry_resolved']==dict.fromkeys(record['bare_token_resolved'],str(root/sdk.ENTRY_RELATIVE))
        assert not list(directory.glob('*.usdc'))
    result=dict(success=True,compact_capture_complete=True);store.finalize_job(job,result)
    assert len(result['scene_storage']['sdk_mdl_selections'])==2
    assert result['scene_storage']['job_directory_bytes']==sum(p.stat().st_size for p in job.rglob('*') if p.is_file())
    ledger=store.batch_ledger([job],batch_root=job.parent)
    assert str(shadow) not in {row['file'] for row in ledger['shared_inputs']}
    assert ledger['total_accounted_bytes']==sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file() and p!=shadow)


def test_v1_stays_closed_even_when_approved_sdk_entry_resolves(tmp_path,monkeypatch):
    pytest.importorskip('pxr')
    from pxr import Ar
    sdk,root,_,config=bound(tmp_path,monkeypatch,usd=True)
    # V1 intent is explicit and contains no SDK input inventory.
    binding=config['compact_scene_storage'];binding['schema']=store.SCHEMA;binding['dependency_policy']=store.DEPENDENCY_POLICY
    binding['sources']={k:v for k,v in binding['sources'].items() if not k.startswith('sdk_mdl_')}
    ref=store.prepare_job(config,tmp_path/'legacy-v1')
    with Ar.ResolverContextBinder(Ar.DefaultResolverContext([str((root/sdk.ENTRY_RELATIVE).parent)])):
        with pytest.raises(ValueError):store.runtime_stage(ref)
