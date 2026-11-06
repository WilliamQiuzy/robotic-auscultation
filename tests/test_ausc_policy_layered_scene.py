"""Lossless layered USD CPU contracts; no Kit, physics, render or admission."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def module():
    return importlib.import_module('sim.ausc_policy.layered_scene_authoring')


def scene(path, body=1, *, iterations=64):
    from pxr import Sdf, Usd, UsdPhysics, Vt
    stage=Usd.Stage.CreateInMemory()
    stage.DefinePrim('/World','Xform')
    mount=stage.DefinePrim('/World/StethoArmMount','Mesh')
    mount.CreateAttribute('points',Sdf.ValueTypeNames.Point3fArray).Set(Vt.Vec3fArray.FromNumpy(
        np.random.default_rng(4).normal(size=(15000,3)).astype(np.float32)))
    mount.CreateAttribute('mdl',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('OmniPBR.mdl'))
    for i in range(24):
        p=stage.DefinePrim('/World/Robot/R'+str(i),'Xform');UsdPhysics.RigidBodyAPI.Apply(p)
        p.CreateAttribute('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Int).Set(iterations)
        p.CreateAttribute('physxRigidBody:solverVelocityIterationCount',Sdf.ValueTypeNames.Int).Set(1)
        p.CreateAttribute('physxJoint:armature',Sdf.ValueTypeNames.Float).Set(.02)
    stage.DefinePrim('/World/Patient','Mesh').CreateAttribute('body',Sdf.ValueTypeNames.Int).Set(body)
    c=stage.CreateClassPrim('/Flattened_Prototype_0');c.CreateAttribute('points',Sdf.ValueTypeNames.Point3fArray).Set(Vt.Vec3fArray.FromNumpy(np.asarray(mount.GetAttribute('points').Get())*2))
    stage.DefinePrim('/World/Instance').GetInherits().AddInherit('/Flattened_Prototype_0')
    stage.GetRootLayer().Export(str(path))
    return stage


def refs(tmp_path):
    from sim.ausc_policy.scene_store import source_ref
    directory=tmp_path/'inputs';directory.mkdir()
    scene(directory/'one.usdc',1);scene(directory/'two.usdc',2)
    return [source_ref(directory/name) for name in ('one.usdc','two.usdc')]


def test_value_hash_preserves_types_bulk_arrays_and_one_bit():
    from pxr import Vt
    m=module();a=np.arange(12,dtype=np.float32).reshape(4,3);b=a.copy();b.view(np.uint32)[1,1]^=1
    assert m.typed_value(Vt.Vec3fArray.FromNumpy(a))!=m.typed_value(Vt.Vec3fArray.FromNumpy(b))
    assert m.typed_value(Vt.FloatArray([1]))!=m.typed_value(Vt.DoubleArray([1]))
    assert m.typed_value(1)!=m.typed_value(1.0)


@pytest.mark.parametrize('mutation',['one_bit','type','api','metadata','children','prototype','mdl','targets','connections','samples','listop'])
def test_full_snapshot_detects_safety_or_composition_mutations(tmp_path,mutation):
    from pxr import Sdf,UsdPhysics
    m=module();stage=scene(tmp_path/'source.usdc');before=m.scene_snapshot(stage)
    p=stage.GetPrimAtPath('/World/StethoArmMount');a=p.GetAttribute('points')
    if mutation=='one_bit':
        data=np.asarray(a.Get()).copy();data.view(np.uint32)[0,0]^=1;a.Set(data)
    elif mutation=='type':
        p.RemoveProperty('points');p.CreateAttribute('points',Sdf.ValueTypeNames.Double3Array).Set([(1,2,3)])
    elif mutation=='api':stage.GetPrimAtPath('/World/Robot/R0').RemoveAPI(UsdPhysics.RigidBodyAPI)
    elif mutation=='metadata':p.SetMetadata('documentation','changed safety metadata')
    elif mutation=='children':stage.GetPrimAtPath('/World').SetChildrenReorder(['Patient','StethoArmMount'])
    elif mutation=='prototype':stage.GetPrimAtPath('/Flattened_Prototype_0').GetAttribute('points').Set([(3,2,1)])
    elif mutation=='mdl':p.GetAttribute('mdl').Set(Sdf.AssetPath('other.mdl'))
    elif mutation=='targets':p.CreateRelationship('r').SetTargets(['/World/Patient','/World/Robot'])
    elif mutation=='connections':p.CreateAttribute('connected',Sdf.ValueTypeNames.Float).AddConnection('/World/Robot/R0.physxJoint:armature')
    elif mutation=='samples':a.Set([(2,1,0)],1)
    else:p.GetReferences().AddInternalReference('/Flattened_Prototype_0')
    assert m.scene_snapshot(stage)!=before


def test_factor_saved_readback_and_private_runtime_keep_complete_hierarchy(tmp_path):
    from pxr import Usd
    m=module();inputs=refs(tmp_path);before=[Path(r['file']).read_bytes() for r in inputs]
    result=m.factor_scene_layers(inputs,output_root=tmp_path/'new')
    assert result['saved_bytes']<result['original_bytes']*.9
    assert '/Flattened_Prototype_0' in result['static_subtrees']
    assert '/World/StethoArmMount' in result['static_subtrees']
    assert '/World/Patient' not in result['static_subtrees']
    for source,item in zip(inputs,result['scenes']):
        inventory=item['dependency_inventory'];saved=Usd.Stage.Open(inventory['primary']['file'])
        assert m.scene_snapshot(saved)==m.scene_snapshot(Usd.Stage.Open(source['file']))
        from sim.ausc_policy.scene_store import validate_dependency_inventory
        assert validate_dependency_inventory(inventory)==inventory
        runtime=m.private_layered_stage(inventory)
        assert m.scene_snapshot(runtime)==m.scene_snapshot(saved)
        assert all(x.anonymous for x in runtime.GetUsedLayers())
        runtime.DefinePrim('/PrivateOnly')
        assert not saved.GetPrimAtPath('/PrivateOnly')
    assert [Path(r['file']).read_bytes() for r in inputs]==before


@pytest.mark.parametrize('mutation',['stale','hardlink','symlink','lexical_alias','unknown','cycle','extra','missing'])
def test_dependency_inventory_rejects_open_or_changed_closure(tmp_path,mutation):
    import os
    from pxr import Sdf
    from sim.ausc_policy import scene_store as store
    result=module().factor_scene_layers(refs(tmp_path),output_root=tmp_path/'new')
    inv=deepcopy(result['scenes'][0]['dependency_inventory']);base=Path(inv['base']['file'])
    if mutation=='stale':base.chmod(0o644);base.write_bytes(base.read_bytes()+b'x')
    elif mutation=='hardlink':os.link(base,tmp_path/'alias.usdc')
    elif mutation=='symlink':
        alias=tmp_path/'alias.usdc';alias.symlink_to(base);inv['base']['file']=str(alias)
    elif mutation=='lexical_alias':inv['base']['file']=str(base.parent)+'/./'+base.name
    elif mutation=='unknown':inv['schema']='unbound'
    elif mutation=='extra':inv['other']=inv['base']
    elif mutation=='missing':base.unlink()
    else:
        primary=Path(inv['primary']['file']);layer=Sdf.Layer.FindOrOpen(str(primary));primary.chmod(0o644)
        layer.subLayerPaths=[str(primary)];layer.Save();primary.chmod(0o444);inv['primary']=store.source_ref(primary)
    with pytest.raises(ValueError):store.validate_dependency_inventory(inv)


def test_qualified_iterations_is_only_24_existing_int_defaults(tmp_path):
    m=module();stage=scene(tmp_path/'source.usdc',iterations=32);before=m.scene_snapshot(stage)
    selected,proof=m.apply_qualified_iterations(stage)
    assert m.scene_snapshot(stage)==before and len(proof['only_iteration_changes'])==24
    assert {x['before'] for x in proof['only_iteration_changes']}=={32}
    assert {x['after'] for x in proof['only_iteration_changes']}=={64}
    assert selected.GetPrimAtPath('/World/Robot/R0').GetAttribute('physxJoint:armature').Get()==np.float32(.02)


def test_qualified_iterations_preserves_real_unauthored_schema_armature_fallback(tmp_path):
    import subprocess
    import sys
    # USD caches schema definitions on first stage creation. Register this
    # resource-only schema in a fresh process, independent of suite order.
    plugin=tmp_path/'schema';plugin.mkdir()
    (plugin/'plugInfo.json').write_text(json.dumps(dict(Plugins=[dict(
        Type='resource',Name='wamTestArmature',Root='.',ResourcePath='.',Info=dict(Types=dict(
            WamTestUsdArmatureAPI=dict(alias=dict(UsdSchemaBase='WamTestArmatureAPI'),
                bases=['UsdAPISchemaBase'],schemaIdentifier='WamTestArmatureAPI',
                schemaKind='singleApplyAPI',autoGenerated=True))))])))
    (plugin/'generatedSchema.usda').write_text(
        '#usda 1.0\nclass "WamTestArmatureAPI" {\n float physxJoint:armature = 0\n}\n')
    result=subprocess.run([sys.executable,'-B','-c',r'''
import runpy
import sys
from pathlib import Path
from pxr import Plug, Sdf
root=Path(sys.argv[1])
Plug.Registry().RegisterPlugins(str(root/'schema/plugInfo.json'))
fixture=runpy.run_path(sys.argv[2])
m=fixture['module']()
source=root/'source.usdc';stage=fixture['scene'](source,iterations=32)
joint=stage.DefinePrim('/World/UnauthoredSchemaJoint')
assert joint.ApplyAPI('WamTestArmatureAPI')
assert stage.GetPrimAtPath('/World/Robot/R0').ApplyAPI('WamTestArmatureAPI')
a=joint.GetAttribute('physxJoint:armature')
assert a.GetTypeName()==Sdf.ValueTypeNames.Float and a.Get()==0.0
assert not a.HasAuthoredValueOpinion() and not a.GetPropertyStack()
assert a in joint.GetAttributes() and not joint.GetAuthoredAttributes()
assert stage.GetRootLayer().Export(str(source))
source_bytes=source.read_bytes()
before=m.scene_snapshot(stage);raw_before=m._layer_snapshot(stage.GetRootLayer())
authored=sorted(str(a.GetPath()) for p in stage.TraverseAll()
    for a in p.GetAuthoredAttributes() if a.GetName().endswith(':armature'))
selected,proof=m.apply_qualified_iterations(stage)
assert sorted(proof['float32_armature_attributes'])==authored and len(authored)==24
assert len(proof['only_iteration_changes'])==len(proof['rigid_articulation_roles'])==24
for change in proof['only_iteration_changes']:
    assert change['before']==32 and change['after']==64
    p=selected.GetPrimAtPath(change['path'])
    assert p.GetAttribute(change['attribute']).Get()==64
    assert p.GetAttribute('physxRigidBody:solverVelocityIterationCount').Get()==1
fallback=selected.GetPrimAtPath(joint.GetPath()).GetAttribute(a.GetName())
assert fallback.Get()==0.0 and not fallback.HasAuthoredValueOpinion()
assert not fallback.GetPropertyStack()
after=m.scene_snapshot(selected);raw_after=m._layer_snapshot(selected.GetRootLayer())
assert proof['before_stage_semantics_sha256']==m.digest(before)
assert proof['after_stage_semantics_sha256']==m.digest(after)
assert proof['before_authored_sha256']==m.digest(raw_before)
assert proof['after_authored_sha256']==m.digest(raw_after)
for change in proof['only_iteration_changes']:
    path,name=change['path'],change['attribute']
    after['prims'][path]['attributes'][name]['default']=before['prims'][path]['attributes'][name]['default']
    raw_after[path+'.'+name]['info']['default']=raw_before[path+'.'+name]['info']['default']
assert after==before and raw_after==raw_before
assert proof['armature_authoring_changed'] is False
assert proof['exhaustive_metadata_API_relationship_connection_time_sample_check'] is True
assert m.scene_snapshot(stage)==before and m._layer_snapshot(stage.GetRootLayer())==raw_before
assert source.read_bytes()==source_bytes
''',str(tmp_path),str(Path(__file__).resolve())],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('mutation',['value','one_bit','type','no_default','blocked','sample','missing'])
def test_qualified_iterations_rejects_malformed_authored_armature(tmp_path,mutation):
    from pxr import Sdf
    m=module();stage=scene(tmp_path/'source.usdc',iterations=32)
    p=stage.GetPrimAtPath('/World/Robot/R0');a=p.GetAttribute('physxJoint:armature')
    if mutation=='value':a.Set(.021)
    elif mutation=='one_bit':a.Set(float(np.nextafter(np.float32(.02),np.float32(np.inf))))
    elif mutation=='type':
        p.RemoveProperty(a.GetName());p.CreateAttribute(a.GetName(),Sdf.ValueTypeNames.Double).Set(.02)
    elif mutation=='no_default':a.Clear()
    elif mutation=='blocked':a.Block()
    elif mutation=='sample':a.Set(.02,1)
    else:
        for p in stage.TraverseAll():p.RemoveProperty('physxJoint:armature')
    before=m.scene_snapshot(stage);raw_before=m._layer_snapshot(stage.GetRootLayer())
    with pytest.raises(ValueError,match='armature'):
        m.apply_qualified_iterations(stage)
    assert m.scene_snapshot(stage)==before and m._layer_snapshot(stage.GetRootLayer())==raw_before


@pytest.mark.parametrize('mutation',['count','velocity','armature','type','sample'])
def test_qualified_iterations_rejects_any_unqualified_profile(tmp_path,mutation):
    from pxr import Sdf
    m=module();stage=scene(tmp_path/'source.usdc',iterations=32);p=stage.GetPrimAtPath('/World/Robot/R0')
    if mutation=='count':stage.RemovePrim('/World/Robot/R0')
    elif mutation=='velocity':p.GetAttribute('physxRigidBody:solverVelocityIterationCount').Set(2)
    elif mutation=='armature':p.GetAttribute('physxJoint:armature').Set(.021)
    elif mutation=='type':
        p.RemoveProperty('physxRigidBody:solverPositionIterationCount');p.CreateAttribute('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Float).Set(32.)
    else:p.GetAttribute('physxRigidBody:solverPositionIterationCount').Set(32,1)
    with pytest.raises(ValueError):m.apply_qualified_iterations(stage)


def prepared(tmp_path,monkeypatch,*,identity='p000439',split='train',body=1):
    from test_ausc_policy_m2_pilot_config import fixture
    from sim.ausc_policy import m2_pilot_config as config
    tmp_path.mkdir();selection,_,_=fixture(tmp_path,monkeypatch,identity=identity,split=split)
    authored=scene(tmp_path/'candidate_physics.usdc',body,iterations=32)
    authored.Flatten().Export(str(tmp_path/'candidate_physics.usdc'))
    complete=Path(selection['prepared_complete']['file']);data=json.loads(complete.read_text())
    data['effective_scene_sha256']=config.hash_file(tmp_path/'candidate_physics.usdc');complete.write_text(json.dumps(data))
    selection['prepared_complete']=config.reference(complete)
    for name in ('mass_estimates.json','physics_profile.json','approved_camera_restoration.json'):(tmp_path/name).write_text('{}')
    return selection


def test_population_metadata_all_precedes_payload_and_rejects_test(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_pilot_config as config
    selected=[prepared(tmp_path/'one',monkeypatch),prepared(tmp_path/'bad',monkeypatch,identity='p000839',split='test')]
    opened=[];real=config.hash_file
    monkeypatch.setattr(config,'hash_file',lambda path:opened.append(Path(path).name) or real(path))
    with pytest.raises(ValueError,match='TRAIN|VALIDATION|split'):
        module().author_population(selected=selected,output_root=tmp_path/'new',iteration_spec=module().ITERATIONS)
    assert not any(name in ('candidate_physics.usdc','body.npz','collision_geometry.npz') for name in opened)
    assert not (tmp_path/'new').exists()


def test_population_direct_32_to_layered_64_keeps_originals_and_separate_proofs(tmp_path,monkeypatch):
    from sim.ausc_policy import scene_store as store
    selected=[prepared(tmp_path/'one',monkeypatch,body=1),prepared(tmp_path/'two',monkeypatch,body=2)]
    before={str(p):p.read_bytes() for root in ('one','two') for p in (tmp_path/root).rglob('*') if p.is_file()}
    from test_ausc_policy_sdk_mdl import sdk_fixture
    sdk,_,_=sdk_fixture(tmp_path,monkeypatch);manifest=sdk.prepare_manifest(tmp_path/'sdk-manifest.json')
    m=module();result=m.author_population(selected=selected,output_root=tmp_path/'new',iteration_spec=m.ITERATIONS)
    assert len(result['prepared'])==2
    assert len(list((tmp_path/'new').rglob('*.usdc')))==3
    for original,item in zip(selected,result['prepared']):
        complete=json.loads(store.read_source(store.source_ref(item['prepared_complete']['file'])))
        assert complete['patient']==json.loads(Path(original['patient_receipt']['file']).read_text())
        assert complete['scene_dependency_inventory']==item['dependency_inventory']
        profile=json.loads(Path(item['physics_profile']['file']).read_text())
        assert len(profile['only_iteration_changes'])==24
        assert item['original']==original and item['input_bindings']
        binding=store.build_binding(**{k:item[k] for k in original},store_root=tmp_path/'cas',batch_id='new',sdk_mdl_manifest=manifest)
        # Explicit layered route requires its exact approved SDK bundle.
        assert binding['schema']==store.SDK_LAYER_SCHEMA
        config=json.loads(Path(item['preparation_config']['file']).read_text());config.pop('prepare_only')
        config.setdefault('fresh_expert',{}).update(raw_multimodal_capture=dict(enabled=True,profile='compact'))
        config['compact_scene_storage']=binding
        runtime=store.runtime_stage(store.prepare_job(config,tmp_path/('job'+str(len(binding['sources']['scene']['file']))+Path(item['prepared_complete']['file']).parent.name)))
        assert all(layer.anonymous for layer in runtime.GetUsedLayers())
        assert len(runtime.GetUsedLayers())==2
        old_binding=store.build_binding(**original,store_root=tmp_path/'old-cas',batch_id='old',sdk_mdl_manifest=manifest)
        old_runtime=store.prepared_runtime_stage(old_binding)
        old_selected,_=m.apply_qualified_iterations(old_runtime)
        assert m.scene_snapshot(old_selected)==m.scene_snapshot(runtime)
        import os
        alias=tmp_path/'sidecar-alias';os.link(binding['sources']['mass_estimates']['file'],alias)
        with pytest.raises(ValueError,match='single-link|alias'):store._validate_binding(binding)
        alias.unlink()
    read_counts={};read=store.read_source
    def counted(ref,**kwargs):
        read_counts[ref['file']]=read_counts.get(ref['file'],0)+1
        return read(ref,**kwargs)
    monkeypatch.setattr(store,'read_source',counted)
    ledger=store.batch_ledger([p.parent for p in tmp_path.glob('job*/scene_ref.json')])
    for name in ('one','two'):
        assert read_counts[str(tmp_path/name/'candidate_physics.usdc')]==1
        assert read_counts[str(tmp_path/name/'body.npz')]==1
    inputs={r['file'] for r in ledger['shared_inputs']}
    assert {str(tmp_path/name/'candidate_physics.usdc') for name in ('one','two')}<=inputs
    assert sum(r['file']==result['base']['file'] for r in ledger['shared_inputs'])==1
    assert all(Path(p).read_bytes()==value for p,value in before.items())


@pytest.mark.parametrize('kind',['external_reference','internal_cycle','mdl','extra_sublayer','malformed_ref'])
def test_inventory_validates_closed_graph_even_with_refreshed_hash(tmp_path,kind):
    from pxr import Sdf
    from sim.ausc_policy import scene_store as store
    result=module().factor_scene_layers(refs(tmp_path),output_root=tmp_path/'new')
    inv=deepcopy(result['scenes'][0]['dependency_inventory'])
    if kind=='malformed_ref':inv['base']={}
    else:
        path=Path(inv['base']['file']);path.chmod(0o644);layer=Sdf.Layer.OpenAsAnonymous(str(path))
        if kind=='external_reference':
            p=layer.GetPrimAtPath('/World/StethoArmMount');p.referenceList.prependedItems=[Sdf.Reference('/unbound/file.usdc')]
        elif kind=='internal_cycle':
            layer.GetPrimAtPath('/World/StethoArmMount').referenceList.prependedItems=[Sdf.Reference('', '/World/StethoArmMount')]
        elif kind=='extra_sublayer':layer.subLayerPaths=[inv['primary']['file']]
        else:layer.GetAttributeAtPath('/World/StethoArmMount.mdl').default=Sdf.AssetPath('/unbound/OmniPBR.mdl')
        layer.Export(str(path));path.chmod(0o444);inv['base']=store.source_ref(path)
    with pytest.raises(ValueError):store.validate_dependency_inventory(inv)


def test_static_factor_keeps_one_bit_varying_complete_mesh_in_primary(tmp_path):
    from pxr import Sdf,Usd
    inputs=refs(tmp_path);path=Path(inputs[1]['file']);stage=Usd.Stage.Open(str(path))
    attr=stage.GetPrimAtPath('/World/StethoArmMount').GetAttribute('points')
    points=np.asarray(attr.Get()).copy();points.view(np.uint32)[0,0]^=1;attr.Set(points);stage.GetRootLayer().Save()
    from sim.ausc_policy.scene_store import source_ref
    inputs[1]=source_ref(path)
    result=module().factor_scene_layers(inputs,output_root=tmp_path/'new')
    assert '/World/StethoArmMount' not in result['static_subtrees']
    assert '/Flattened_Prototype_0' in result['static_subtrees']
    for item in result['scenes']:
        primary=Sdf.Layer.OpenAsAnonymous(item['dependency_inventory']['primary']['file'])
        assert primary.GetAttributeAtPath('/World/StethoArmMount.points').HasInfo('default')


def test_prepared_private_equivalence_route_creates_no_cas_or_dense_copy(tmp_path,monkeypatch):
    from sim.ausc_policy import scene_store as store
    from test_ausc_policy_scene_sdk_mdl import bound
    _,_,_,config=bound(tmp_path,monkeypatch,usd=True);binding=config['compact_scene_storage']
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(store,'_publish',lambda *args,**kwargs:pytest.fail('Prepared equivalence must not publish a dense CAS'))
    stage=store.prepared_runtime_stage(binding)
    assert all(layer.anonymous for layer in stage.GetUsedLayers())
    assert not Path(binding['store_root']).exists()
    assert all(Path(p).read_bytes()==data for p,data in before.items())


def test_saving_gate_counts_each_original_actual_path_once(tmp_path):
    inputs=refs(tmp_path)
    with pytest.raises(ValueError,match='save at least ten percent'):
        module().factor_scene_layers([inputs[0],deepcopy(inputs[0])],output_root=tmp_path/'new')


def test_snapshot_and_factoring_preserve_actual_instance_prototypes(tmp_path):
    from pxr import Usd
    inputs=refs(tmp_path)
    from sim.ausc_policy.scene_store import source_ref
    for i,ref in enumerate(inputs):
        stage=Usd.Stage.Open(ref['file']);stage.GetPrimAtPath('/World/Instance').SetInstanceable(True)
        stage.GetRootLayer().Save();inputs[i]=source_ref(ref['file'])
    original=Usd.Stage.Open(inputs[0]['file']);before=module().scene_snapshot(original)
    assert before['prototypes']
    result=module().factor_scene_layers(inputs,output_root=tmp_path/'new')
    private=module().private_layered_stage(result['scenes'][0]['dependency_inventory'])
    assert module().scene_snapshot(private)==before
    original.GetPrimAtPath('/Flattened_Prototype_0').GetAttribute('points').Set([(1,2,3)])
    assert module().scene_snapshot(original)!=before


@pytest.mark.parametrize('mutation',['omitted','wrong_ref','test_header','during_ledger_mutation'])
def test_ledger_authenticates_complete_authoring_headers_and_closure(tmp_path,monkeypatch,mutation):
    from sim.ausc_policy import scene_store as store
    from test_ausc_policy_sdk_mdl import sdk_fixture
    selected=[prepared(tmp_path/'one',monkeypatch,body=1),prepared(tmp_path/'two',monkeypatch,body=2)]
    sdk,_,_=sdk_fixture(tmp_path,monkeypatch);manifest=sdk.prepare_manifest(tmp_path/'sdk-manifest.json')
    result=module().author_population(selected=selected,output_root=tmp_path/'new',iteration_spec=module().ITERATIONS)
    item=result['prepared'][0]
    binding=store.build_binding(**{k:item[k] for k in selected[0]},store_root=tmp_path/'cas',batch_id='new',sdk_mdl_manifest=manifest)
    config=json.loads(Path(item['preparation_config']['file']).read_text());config.pop('prepare_only')
    config['fresh_expert'].update(raw_multimodal_capture=dict(enabled=True,profile='compact'))
    config['compact_scene_storage']=binding
    job=tmp_path/'job';store.prepare_job(config,job)
    ref=json.loads((job/'scene_ref.json').read_text());receipt_path=Path(result['receipt']['file'])
    receipt=json.loads(receipt_path.read_text())
    if mutation=='omitted':receipt['input_bindings'][0].pop()
    elif mutation=='wrong_ref':
        next(r for r in receipt['input_bindings'][0] if Path(r['file']).name=='candidate_physics.usdc')['sha256']='0'*64
    elif mutation=='test_header':
        receipt['original_selected'][1]=prepared(tmp_path/'bad',monkeypatch,identity='p000839',split='test')
    if mutation!='during_ledger_mutation':
        receipt_path.chmod(0o644);receipt_path.write_text(json.dumps(receipt));receipt_path.chmod(0o444)
        receipt_ref=store.source_ref(receipt_path)
        complete_path=Path(item['prepared_complete']['file']);complete=json.loads(complete_path.read_text())
        complete['scene_authoring_receipt']=receipt_ref
        complete_path.chmod(0o644);complete_path.write_text(json.dumps(complete));complete_path.chmod(0o444)
        ref['binding']['sources']['prepared_complete']=store.source_ref(complete_path)
        ref['binding']['sources']['scene_authoring_receipt']=receipt_ref
        (job/'scene_ref.json').write_text(json.dumps(ref))
    read=store.read_source;payloads=[];changed=[]
    original=tmp_path/'one/candidate_physics.usdc'
    def observe(source,**kwargs):
        name=Path(source['file']).name
        if name in ('body.npz','candidate_physics.usdc','collision_geometry.npz'):payloads.append(source['file'])
        if mutation=='during_ledger_mutation' and Path(source['file'])==Path(binding['sources']['scene']['file']) and not changed:
            original.write_bytes(original.read_bytes()+b'changed after cached source verification');changed.append(True)
        return read(source,**kwargs)
    monkeypatch.setattr(store,'read_source',observe)
    with pytest.raises(ValueError):store.batch_ledger([job])
    if mutation!='during_ledger_mutation':assert not payloads
    else:assert changed


def test_population_rejects_metadata_lexical_alias_before_payload(tmp_path,monkeypatch):
    selected=[prepared(tmp_path/'one',monkeypatch),prepared(tmp_path/'two',monkeypatch)]
    path=Path(selected[1]['preparation_config']['file'])
    selected[1]['preparation_config']['file']=str(path.parent)+'/./'+path.name
    from sim.ausc_policy import m2_pilot_config as config
    real=config.hash_file;payloads=[]
    def observe(path):
        if Path(path).name in ('body.npz','candidate_physics.usdc','collision_geometry.npz'):payloads.append(str(path))
        return real(path)
    monkeypatch.setattr(config,'hash_file',observe)
    with pytest.raises(ValueError,match='alias|Canonical'):
        module().author_population(selected=selected,output_root=tmp_path/'new',iteration_spec=module().ITERATIONS)
    assert not payloads


def historical_alias_fixture(tmp_path,monkeypatch):
    m=module();target_root=tmp_path/'workspace/wam';target=target_root/'runs/review/candidate.usdc'
    target.parent.mkdir(parents=True);target.write_bytes(b'canonical historical geometry')
    link=tmp_path/'home/william/wam-vla-v2';link.parent.mkdir(parents=True);link.symlink_to(target_root,target_is_directory=True)
    raw=link/'runs/review/candidate.usdc'
    monkeypatch.setattr(m,'HISTORICAL_GEOMETRY_RAW',raw)
    monkeypatch.setattr(m,'HISTORICAL_GEOMETRY_LINK',link)
    monkeypatch.setattr(m,'HISTORICAL_GEOMETRY_TARGET',target)
    monkeypatch.setattr(m,'HISTORICAL_GEOMETRY_LINK_TARGET',str(target_root))
    return m,raw,link,target


def test_only_exact_preexisting_historical_geometry_alias_resolves_canonical_target(tmp_path,monkeypatch):
    m,raw,link,target=historical_alias_fixture(tmp_path,monkeypatch)
    assert m.geometry_source_path(dict(geometry_source=str(raw)))==target
    assert m.geometry_source_path(dict(geometry_source=str(target)))==target
    assert str(m.HISTORICAL_GEOMETRY_LINK_TARGET)==str(target.parents[2])


@pytest.mark.parametrize('kind',['fake_link','wrong_target','unexpected_parent','other_alias'])
def test_historical_alias_rule_rejects_any_other_link_boundary(tmp_path,monkeypatch,kind):
    m,raw,link,target=historical_alias_fixture(tmp_path,monkeypatch)
    if kind=='fake_link':
        link.unlink();link.mkdir()
    elif kind=='wrong_target':
        link.unlink();link.symlink_to(tmp_path/'elsewhere',target_is_directory=True)
    elif kind=='unexpected_parent':
        directory=target.parent;directory.rename(directory.with_name('actual'));directory.symlink_to(directory.with_name('actual'),target_is_directory=True)
    else:
        extra=tmp_path/'other';extra.symlink_to(target.parent,target_is_directory=True);raw=extra/target.name
    with pytest.raises(ValueError):m.geometry_source_path(dict(geometry_source=str(raw)))


def test_historical_geometry_alias_constants_are_exact_and_not_generic():
    m=module()
    assert str(m.HISTORICAL_GEOMETRY_RAW)=='/home/william/wam-vla-v2/runs/ausc_policy/retainer-5mm-recess1mm-review-20260930-v2/output/review/candidate.usdc'
    assert str(m.HISTORICAL_GEOMETRY_LINK)=='/home/william/wam-vla-v2'
    assert m.HISTORICAL_GEOMETRY_LINK_TARGET=='/workspace/wam'
    assert str(m.HISTORICAL_GEOMETRY_TARGET)=='/workspace/wam/runs/ausc_policy/retainer-5mm-recess1mm-review-20260930-v2/output/review/candidate.usdc'


@pytest.mark.parametrize('role,prim_path',[
    ('base','/World/StethoArmMount'),
    ('primary','/World/Patient'),
])
def test_saved_authored_proof_rejects_lost_fallback_default(tmp_path,monkeypatch,role,prim_path):
    from pxr import Sdf,Usd,UsdGeom
    from sim.ausc_policy.scene_store import source_ref
    inputs=refs(tmp_path)
    for i,ref in enumerate(inputs):
        stage=Usd.Stage.Open(ref['file'])
        UsdGeom.Imageable(stage.GetPrimAtPath(prim_path)).CreateVisibilityAttr('inherited')
        stage.GetRootLayer().Save();inputs[i]=source_ref(ref['file'])
    export=Sdf.Layer.Export;removed=[]
    filename='static_base.usdc' if role=='base' else 'candidate_physics.usdc'
    def drop_saved_default(layer,path,*args,**kwargs):
        result=export(layer,path,*args,**kwargs)
        if Path(path).name==filename:
            saved=Sdf.Layer.OpenAsAnonymous(str(path))
            attr=saved.GetAttributeAtPath(prim_path+'.visibility')
            assert attr is not None and attr.HasInfo('default')
            attr.ClearInfo('default')
            assert export(saved,path)
            removed.append(str(path))
        return result
    monkeypatch.setattr(Sdf.Layer,'Export',drop_saved_default)
    with pytest.raises(ValueError,match='Layered authored fields/list operations changed'):
        module().factor_scene_layers(inputs,output_root=tmp_path/'new')
    assert len(removed)==1
    saved=Sdf.Layer.OpenAsAnonymous(removed[0])
    assert not saved.GetAttributeAtPath(prim_path+'.visibility').HasInfo('default')
    assert UsdGeom.Imageable(Usd.Stage.Open(saved).GetPrimAtPath(prim_path)).GetVisibilityAttr().Get()=='inherited'


@pytest.mark.parametrize('fault',['base_ancestor_metadata','primary_static_shell_default'])
def test_saved_authored_proof_rejects_masked_extra_opinions(tmp_path,monkeypatch,fault):
    from pxr import Sdf,Usd,UsdGeom
    from sim.ausc_policy.scene_store import source_ref
    inputs=refs(tmp_path)
    for i,ref in enumerate(inputs):
        stage=Usd.Stage.Open(ref['file'])
        stage.GetPrimAtPath('/World').SetMetadata('documentation','approved original')
        UsdGeom.Imageable(stage.GetPrimAtPath('/World/StethoArmMount')).CreateVisibilityAttr('inherited')
        stage.GetRootLayer().Save();inputs[i]=source_ref(ref['file'])
    export=Sdf.Layer.Export;injected=[]
    filename='static_base.usdc' if fault=='base_ancestor_metadata' else 'candidate_physics.usdc'
    def add_masked_saved_opinion(layer,path,*args,**kwargs):
        result=export(layer,path,*args,**kwargs)
        if Path(path).name==filename and not injected:
            saved=Sdf.Layer.OpenAsAnonymous(str(path))
            if fault=='base_ancestor_metadata':
                assert not saved.GetPrimAtPath('/World').HasInfo('documentation')
                saved.GetPrimAtPath('/World').SetInfo('documentation','UNAPPROVED SAVED BASE OPINION')
            else:
                attr_path='/World/StethoArmMount.points'
                assert saved.GetAttributeAtPath(attr_path) is None
                base=Sdf.Layer.OpenAsAnonymous(str(Path(path).parent.parent/'static_base.usdc'))
                assert Sdf.CopySpec(base,attr_path,saved,attr_path)
                assert saved.GetAttributeAtPath(attr_path).HasInfo('default')
            assert export(saved,path)
            injected.append(str(path))
        return result
    monkeypatch.setattr(Sdf.Layer,'Export',add_masked_saved_opinion)
    with pytest.raises(ValueError,match='Layered authored fields/list operations changed'):
        module().factor_scene_layers(inputs,output_root=tmp_path/'new')
    assert len(injected)==1
    composed=Usd.Stage.Open(str(tmp_path/'new/scene_0000/candidate_physics.usdc'))
    assert composed.GetPrimAtPath('/World').GetMetadata('documentation')=='approved original'
    assert UsdGeom.Imageable(composed.GetPrimAtPath('/World/StethoArmMount')).GetVisibilityAttr().Get()=='inherited'
    assert module().scene_snapshot(composed)==module().scene_snapshot(Usd.Stage.Open(inputs[0]['file']))


def sdk_layered_pair(tmp_path,monkeypatch,*,already64=False):
    from pxr import Usd
    from sim.ausc_policy import scene_store as store,m2_pilot_config as builder
    from test_ausc_policy_sdk_mdl import sdk_fixture
    selected=[prepared(tmp_path/'one',monkeypatch,body=1),prepared(tmp_path/'two',monkeypatch,body=2)]
    if already64:
        for selection in selected:
            root=Path(selection['prepared_complete']['file']).parent
            stage=Usd.Stage.Open(str(root/'candidate_physics.usdc'))
            stage,_=module().apply_qualified_iterations(stage)
            stage.GetRootLayer().Export(str(root/'candidate_physics.usdc'))
            complete=json.loads((root/'COMPLETE.json').read_text());complete['effective_scene_sha256']=store.sha(root/'candidate_physics.usdc')
            (root/'COMPLETE.json').write_text(json.dumps(complete));selection['prepared_complete']=builder.reference(root/'COMPLETE.json')
            prep_path=Path(selection['preparation_config']['file']);prep=json.loads(prep_path.read_text());prep['physics']=deepcopy(module().ITERATIONS)
            prep_path.write_text(json.dumps(prep));selection['preparation_config']=builder.reference(prep_path)
    sdk,_,_=sdk_fixture(tmp_path,monkeypatch);manifest=sdk.prepare_manifest(tmp_path/'sdk-manifest.json')
    authored=module().author_population(selected=selected,output_root=tmp_path/'new',
        iteration_spec=None if already64 else module().ITERATIONS)
    item=authored['prepared'][0]
    old=store.build_binding(**selected[0],store_root=tmp_path/'old-cas',batch_id='old',sdk_mdl_manifest=manifest)
    new=store.build_binding(**{k:item[k] for k in selected[0]},store_root=tmp_path/'new-cas',batch_id='new',sdk_mdl_manifest=manifest)
    return selected,authored,old,new


@pytest.mark.parametrize('already64',[False,True])
@pytest.mark.parametrize('bare_provider',['approved_entry','diagnostic_only'])
def test_approved_sdk_resolver_change_compares_complete_current_context_stages(tmp_path,monkeypatch,already64,bare_provider):
    from pxr import Ar
    from sim.ausc_policy import scene_store as store
    Ar.DefaultResolver.SetDefaultSearchPath([])
    try:
        _,authored,old,new=sdk_layered_pair(tmp_path,monkeypatch,already64=already64)
        receipt=Path(authored['receipt']['file']);before=receipt.read_bytes()
        store.prepared_runtime_stage(new)
        entry=store._sdk_bundle(new)['entry']['file']
        provider=Path(entry)
        if bare_provider=='diagnostic_only':
            provider=tmp_path/'other-provider/OmniPBR.mdl';provider.parent.mkdir();provider.write_bytes(b'unbound bare diagnostic provider')
        Ar.DefaultResolver.SetDefaultSearchPath([str(provider.parent)])
        assert str(Ar.GetResolver().Resolve('OmniPBR.mdl'))==str(provider)
        expected=store.prepared_runtime_stage(old)
        if not already64:expected,_=module().apply_qualified_iterations(expected)
        actual=store.prepared_runtime_stage(new)
        assert module().scene_snapshot(actual)==module().scene_snapshot(expected)
        mdl=actual.GetPrimAtPath('/World/StethoArmMount').GetAttribute('mdl').Get()
        assert mdl.path==mdl.resolvedPath==entry
        assert all(layer.anonymous for layer in actual.GetUsedLayers())
        assert receipt.read_bytes()==before
        assert not Path(old['store_root']).exists() and not Path(new['store_root']).exists()
    finally:Ar.DefaultResolver.SetDefaultSearchPath([])


def rewrite_layered_receipt(authored,new,mutate):
    from sim.ausc_policy import scene_store as store
    path=Path(authored['receipt']['file']);receipt=json.loads(path.read_text());mutate(receipt)
    path.chmod(0o644);path.write_text(json.dumps(receipt));path.chmod(0o444)
    receipt_ref=store.source_ref(path)
    complete_path=Path(new['sources']['prepared_complete']['file']);complete=json.loads(complete_path.read_text())
    complete['scene_authoring_receipt']=receipt_ref
    complete_path.chmod(0o644);complete_path.write_text(json.dumps(complete));complete_path.chmod(0o444)
    new['sources']['prepared_complete']=store.source_ref(complete_path);new['sources']['scene_authoring_receipt']=receipt_ref
    return complete


@pytest.mark.parametrize('route',['prepared','binding','resolve','ledger'])
@pytest.mark.parametrize('fault',['unknown_policy','late_test_header'])
def test_layered_all_headers_and_policy_precede_payloads(tmp_path,monkeypatch,route,fault):
    from sim.ausc_policy import scene_store as store
    selected,authored,_,new=sdk_layered_pair(tmp_path,monkeypatch)
    reference=store._publish(new)
    bad=prepared(tmp_path/'bad',monkeypatch,identity='p000839',split='test') if fault=='late_test_header' else None
    def corrupt(receipt):
        if fault=='unknown_policy':receipt['runtime_equivalence_policy']='unbound_sdk_equivalence'
        else:receipt['original_selected'][1]=bad
    complete=rewrite_layered_receipt(authored,new,corrupt);reference['binding']=deepcopy(new)
    job=tmp_path/'job';job.mkdir();(job/'scene_ref.json').write_text(json.dumps(reference))
    payloads=[];read=store.read_source;hash_file=store.sha
    def observed_read(ref,**kwargs):
        if Path(ref['file']).suffix in ('.usdc','.npz'):payloads.append(ref['file'])
        return read(ref,**kwargs)
    def observed_hash(path):
        if Path(path).suffix in ('.usdc','.npz'):payloads.append(str(path))
        return hash_file(path)
    monkeypatch.setattr(store,'read_source',observed_read);monkeypatch.setattr(store,'sha',observed_hash)
    with pytest.raises(ValueError):
        if route=='prepared':store.validate_prepared_dependency_inventory(complete)
        elif route=='resolve':store.resolve_scene(reference)
        elif route=='ledger':store.batch_ledger([job])
        else:
            item=authored['prepared'][0];refs={k:item[k] for k in selected[0]};refs['prepared_complete']={k:new['sources']['prepared_complete'][k] for k in ('file','sha256')}
            store.build_binding(**refs,store_root=new['store_root'],batch_id='new',sdk_mdl_manifest=new['sources']['sdk_mdl_manifest'])
    assert not payloads


@pytest.mark.parametrize('fault',['missing_policy','missing_composed','missing_authored','duplicate_scene','wrong_mode','old_reference'])
def test_layered_receipt_requires_complete_unique_bound_runtime_proof(tmp_path,monkeypatch,fault):
    from sim.ausc_policy import scene_store as store
    _,authored,_,new=sdk_layered_pair(tmp_path,monkeypatch)
    def corrupt(receipt):
        if fault=='missing_policy':receipt.pop('runtime_equivalence_policy',None)
        elif fault=='missing_composed':receipt['scenes'][0].pop('composed_semantics_sha256')
        elif fault=='missing_authored':receipt['scenes'][0].pop('authored_semantics_sha256')
        elif fault=='duplicate_scene':receipt['scenes'][1]=deepcopy(receipt['scenes'][0])
        elif fault=='wrong_mode':receipt['iteration_proofs'][0]['source_already_qualified64']=True
        else:receipt['original_selected'][0]['prepared_complete']['sha256']='0'*64
    complete=rewrite_layered_receipt(authored,new,corrupt)
    with pytest.raises(ValueError):store.validate_prepared_dependency_inventory(complete)


@pytest.mark.parametrize('fault',['cwd_shadow','prepared_shadow','sdk_entry_change','sdk_manifest_change','external_reference'])
def test_layered_current_context_preserves_sdk_and_closed_dependency_rejections(tmp_path,monkeypatch,fault):
    from pxr import Sdf
    from sim.ausc_policy import scene_store as store
    _,_,_,new=sdk_layered_pair(tmp_path,monkeypatch)
    if fault=='cwd_shadow':
        directory=tmp_path/'cwd';directory.mkdir();(directory/'OmniPBR.mdl').write_bytes(b'local shadow');monkeypatch.chdir(directory)
    elif fault=='prepared_shadow':(Path(new['sources']['scene']['file']).parent/'OmniPBR.mdl').write_bytes(b'local shadow')
    elif fault=='sdk_entry_change':Path(store._sdk_bundle(new)['entry']['file']).write_bytes(b'altered approved SDK entry')
    elif fault=='sdk_manifest_change':
        path=Path(new['sources']['sdk_mdl_manifest']['file']);path.chmod(0o644);path.write_text('{}')
    else:
        path=Path(new['sources']['scene_base']['file']);path.chmod(0o644)
        layer=Sdf.Layer.OpenAsAnonymous(str(path));layer.GetPrimAtPath('/World/StethoArmMount').referenceList.prependedItems=[Sdf.Reference('/unbound/foreign.usdc')]
        layer.Export(str(path));path.chmod(0o444)
    with pytest.raises(ValueError):store.prepared_runtime_stage(new)


@pytest.mark.parametrize('fault',['mdl','metadata','safety_api'])
def test_current_sdk_full_snapshot_gate_detects_private_runtime_mutation(tmp_path,monkeypatch,fault):
    from pxr import Sdf,UsdPhysics
    from sim.ausc_policy import scene_store as store
    _,_,_,new=sdk_layered_pair(tmp_path,monkeypatch)
    original=store._runtime_stage_clone_impl
    def mutate(binding,source,recheck):
        stage=original(binding,source,recheck)
        if binding['schema']==store.SDK_LAYER_SCHEMA:
            if fault=='mdl':stage.GetPrimAtPath('/World/StethoArmMount').GetAttribute('mdl').Set(Sdf.AssetPath('/unbound/other.mdl'))
            elif fault=='metadata':stage.GetPrimAtPath('/World').SetMetadata('documentation','altered after SDK restoration')
            else:stage.GetPrimAtPath('/World/Robot/R0').RemoveAPI(UsdPhysics.RigidBodyAPI)
        return stage
    monkeypatch.setattr(store,'_runtime_stage_clone_impl',mutate)
    with pytest.raises(ValueError,match='full current-context approved SDK dense reference'):store.prepared_runtime_stage(new)
