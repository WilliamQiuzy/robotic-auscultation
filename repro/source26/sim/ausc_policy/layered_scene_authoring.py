"""CPU-only, closed, lossless static USD factoring. Never starts Kit or physics.

The original 32/1 -> 64/1 proof and the subsequent storage representation
proof are independent. No dense derived scene is ever persisted.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from .scene_store import need, real_path, read_source, source_ref, sha

INVENTORY_SCHEMA='m2_static_usd_dependency_inventory_v1'
RECEIPT_SCHEMA='m2_lossless_static_usd_authoring_v1'
RUNTIME_EQUIVALENCE_POLICY='m2_current_context_approved_sdk_dense_equivalence_v1'
CURRENT_RAW_SIDECAR_SCHEMA='m2_layered_current_raw_equivalence_v1'
ITERATIONS=dict(position_iterations=64,velocity_iterations=1,armature_kg_m2=.02)
HISTORICAL_GEOMETRY_RAW=Path('/home/william/wam-vla-v2/runs/ausc_policy/retainer-5mm-recess1mm-review-20260930-v2/output/review/candidate.usdc')
HISTORICAL_GEOMETRY_LINK=Path('/home/william/wam-vla-v2')
HISTORICAL_GEOMETRY_LINK_TARGET='/workspace/wam'
HISTORICAL_GEOMETRY_TARGET=Path('/workspace/wam/runs/ausc_policy/retainer-5mm-recess1mm-review-20260930-v2/output/review/candidate.usdc')


def typed_value(value):
    """Keep USD/Python types and hash numeric buffers without vertex iteration."""
    kind=type(value).__module__+'.'+type(value).__name__
    if value is None:return dict(type=kind,value=None)
    if isinstance(value,(bool,str,int)):return dict(type=kind,value=value)
    if isinstance(value,float):return dict(type=kind,bits=struct.pack('>d',value).hex())
    if isinstance(value,dict):
        return dict(type=kind,items=[[typed_value(k),typed_value(v)] for k,v in sorted(value.items(),key=lambda x:str(x[0]))])
    if isinstance(value,(list,tuple)):return dict(type=kind,items=[typed_value(v) for v in value])
    if hasattr(value,'path') and hasattr(value,'resolvedPath'):
        return dict(type=kind,path=value.path,resolved=value.resolvedPath)
    if hasattr(value,'pathString'):return dict(type=kind,path=value.pathString)
    if hasattr(value,'isExplicit') and hasattr(value,'explicitItems'):
        return dict(type=kind,is_explicit=value.isExplicit,**{name:typed_value(list(getattr(value,name)))
            for name in ('explicitItems','prependedItems','appendedItems','addedItems','deletedItems','orderedItems')})
    try:
        buffer=memoryview(value)
        return dict(type=kind,format=buffer.format,shape=list(buffer.shape),
            sha256=hashlib.sha256(buffer.tobytes()).hexdigest())
    except TypeError:pass
    if isinstance(value,np.generic):return dict(type=kind,bytes=value.tobytes().hex())
    # Non-buffer Vt token/string arrays are small; numeric CAD took the bulk path.
    if kind.startswith('pxr.Vt.') and kind.endswith('Array'):
        return dict(type=kind,items=[typed_value(v) for v in value])
    text=repr(value)
    need(' at 0x' not in text,'Unsupported nondeterministic USD value: '+kind)
    return dict(type=kind,representation=text)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _layer_snapshot(layer):
    from pxr import Sdf
    rows={}
    def collect(path):
        spec=layer.GetObjectAtPath(path)
        if spec is None:
            need(path.IsTargetPath(),'Unsupported missing Sdf object')
            rows[str(path)]=dict(type='target')
            return
        row=dict(type=type(spec).__name__,info={k:typed_value(spec.GetInfo(k)) for k in spec.ListInfoKeys()})
        if isinstance(spec,Sdf.PrimSpec):
            row['children']=[x.name for x in spec.nameChildren]
            row['properties']=[x.name for x in spec.properties]
        rows[str(path)]=row
    layer.Traverse(Sdf.Path.absoluteRootPath,collect)
    return rows


def scene_snapshot(stage):
    """Complete composed semantic readback, including classes and prototypes."""
    from pxr import Usd
    need(stage is not None and not stage.GetCompositionErrors(),'USD composition errors')
    def listops(stack):
        return [{k:typed_value(s.GetInfo(k)) for k in s.ListInfoKeys()
            if k in ('references','payload','inheritPaths','specializes','apiSchemas','connectionPaths','targetPaths')}
            for s in stack if any(k in s.ListInfoKeys() for k in
                ('references','payload','inheritPaths','specializes','apiSchemas','connectionPaths','targetPaths'))]
    def prim(p):
        attributes={}
        for a in p.GetAttributes():
            metadata=a.GetAllMetadata();metadata.pop('default',None);metadata.pop('timeSamples',None)
            attributes[a.GetName()]=dict(type=str(a.GetTypeName()),metadata=typed_value(metadata),
                default=typed_value(a.Get()),samples=[[typed_value(t),typed_value(a.Get(t))] for t in a.GetTimeSamples()],
                connections=[str(x) for x in a.GetConnections()],listops=listops(a.GetPropertyStack()))
        return dict(type=p.GetTypeName(),metadata=typed_value(p.GetAllMetadata()),
            applied_schemas=list(p.GetAppliedSchemas()),listops=listops(p.GetPrimStack()),children=[x.GetName() for x in p.GetAllChildren()],
            properties=[x.GetName() for x in p.GetProperties()],attributes=attributes,
            relationships={r.GetName():dict(metadata=typed_value(r.GetAllMetadata()),
                targets=[str(x) for x in r.GetTargets()],listops=listops(r.GetPropertyStack())) for r in p.GetRelationships()},
            is_instance=p.IsInstance(),is_prototype=p.IsPrototype(),
            prototype=str(p.GetPrototype().GetPath()) if p.IsInstance() else None)
    roots=list(stage.GetPrototypes())
    return dict(stage=typed_value(stage.GetPseudoRoot().GetAllMetadata()),
        root_order=[p.GetName() for p in stage.GetPseudoRoot().GetAllChildren()],
        prims={str(p.GetPath()):prim(p) for p in stage.TraverseAll()},
        prototypes=[dict(path=str(root.GetPath()),prims={str(p.GetPath()):prim(p)
            for p in Usd.PrimRange(root,Usd.PrimAllPrimsPredicate)}) for root in roots])


def _private_source(ref):
    from pxr import Sdf,Usd
    _single_file(ref,immutable=False);read_source(ref)
    root=Sdf.Layer.OpenAsAnonymous(ref['file'])
    need(root is not None,'Cannot read source USD')
    _closed_layer(root,())
    stage=Usd.Stage.Open(root)
    need(stage is not None and not stage.GetCompositionErrors(),'Source USD composition failed')
    return stage


def apply_qualified_iterations(source):
    """Only 24 existing int position defaults change; velocity and f32 armature stay."""
    from pxr import Sdf,Usd,UsdPhysics
    need(source and not source.GetRootLayer().subLayerPaths,'Dense source without sublayers required')
    before=scene_snapshot(source);raw_before=_layer_snapshot(source.GetRootLayer())
    layer=Sdf.Layer.CreateAnonymous('qualified-iteration-private.usda');layer.TransferContent(source.GetRootLayer())
    stage=Usd.Stage.Open(layer);allowed=[];roles=[];armatures=[]
    for p in stage.TraverseAll():
        for api,prefix in ((UsdPhysics.RigidBodyAPI,'physxRigidBody'),(UsdPhysics.ArticulationRootAPI,'physxArticulation')):
            if not p.HasAPI(api):continue
            roles.append([str(p.GetPath()),prefix])
            for name,old in ((prefix+':solverPositionIterationCount',32),(prefix+':solverVelocityIterationCount',1)):
                a=p.GetAttribute(name)
                need(a and a.GetTypeName()==Sdf.ValueTypeNames.Int and a.HasAuthoredValueOpinion()
                    and type(a.Get()) is int and a.Get()==old and not a.GetTimeSamples(),
                    'Only existing unsampled int 32/1 iteration defaults supported')
                if old==32:
                    need(a.Set(64),'Iteration attribute authoring failed')
                    allowed.append(dict(path=str(p.GetPath()),attribute=name,before=32,after=64))
        # Loaded schemas also expose unauthored armature fallbacks. Validate
        # every authored property; the full snapshots still include fallbacks.
        for a in p.GetAuthoredAttributes():
            if a.GetName().endswith(':armature'):
                need(a.GetTypeName()==Sdf.ValueTypeNames.Float and a.HasAuthoredValueOpinion()
                    and not a.GetTimeSamples() and a.Get()==float(np.float32(.02)),
                    'Only unchanged authored float32 armature .02 supported')
                armatures.append(str(a.GetPath()))
    need(len(roles)==len(allowed)==24 and armatures,'Unexpected rigid/articulation role, change count or missing armature')
    after=scene_snapshot(stage);raw_after=_layer_snapshot(layer)
    reverted=deepcopy(after);raw_reverted=deepcopy(raw_after)
    for change in allowed:
        path,name=change['path'],change['attribute']
        old=before['prims'][path]['attributes'][name]['default']
        reverted['prims'][path]['attributes'][name]['default']=old
        raw_reverted[path+'.'+name]['info']['default']=raw_before[path+'.'+name]['info']['default']
    need(reverted==before and raw_reverted==raw_before,'Undeclared physics/API/metadata/property change')
    need(scene_snapshot(source)==before and _layer_snapshot(source.GetRootLayer())==raw_before,'Original USD changed')
    return stage,dict(schema='m2_solver_iteration_only_authoring_v2',profile=deepcopy(ITERATIONS),
        only_iteration_changes=allowed,rigid_articulation_roles=roles,float32_armature_attributes=armatures,
        before_stage_semantics_sha256=digest(before),after_stage_semantics_sha256=digest(after),
        before_authored_sha256=digest(raw_before),after_authored_sha256=digest(raw_after),
        armature_authoring_changed=False,geometry_camera_drives_mass_contact_unchanged=True,
        exhaustive_metadata_API_relationship_connection_time_sample_check=True,
        physics_executed=False,physical_validation=False,training_eligible=False)


def replay_qualified_iterations(source,proof,*,already64):
    """Fresh typed proof; historical registry-dependent hashes are provenance only."""
    from pxr import Sdf,Usd,UsdPhysics
    if already64:
        before=scene_snapshot(source);raw_before=_layer_snapshot(source.GetRootLayer())
        layer=Sdf.Layer.CreateAnonymous('qualified64-inverse-private.usda')
        layer.TransferContent(source.GetRootLayer());temporary=Usd.Stage.Open(layer);positions=[]
        for prim in temporary.TraverseAll():
            for api,prefix in ((UsdPhysics.RigidBodyAPI,'physxRigidBody'),(UsdPhysics.ArticulationRootAPI,'physxArticulation')):
                if not prim.HasAPI(api):continue
                attr=prim.GetAttribute(prefix+':solverPositionIterationCount')
                spec=layer.GetAttributeAtPath(attr.GetPath()) if attr else None
                need(attr and attr.GetTypeName()==Sdf.ValueTypeNames.Int and
                    spec is not None and spec.HasInfo('default') and type(spec.default) is int and
                    spec.default==64 and attr.HasAuthoredValueOpinion() and
                    type(attr.Get()) is int and attr.Get()==64 and not attr.GetTimeSamples(),
                    'Exact authored unsampled int qualified64 defaults required')
                positions.append(attr)
        need(len(positions)==24,'Exact original qualified64 role count required')
        for attr in positions:need(attr.Set(32),'Private qualified64 proof inversion failed')
        selected,actual=apply_qualified_iterations(temporary)
        need(_layer_snapshot(selected.GetRootLayer())==raw_before and scene_snapshot(selected)==before,
            'Already-qualified64 proof changed complete RAW or stage semantics')
        need(_layer_snapshot(source.GetRootLayer())==raw_before and scene_snapshot(source)==before,
            'Original qualified64 source changed')
    else:selected,actual=apply_qualified_iterations(source)
    if proof is not None:
        from .scene_store import _exact_json
        for name in ('profile','only_iteration_changes','rigid_articulation_roles','float32_armature_attributes',
                     'armature_authoring_changed','geometry_camera_drives_mass_contact_unchanged',
                     'exhaustive_metadata_API_relationship_connection_time_sample_check'):
            need(_exact_json(actual[name],proof[name]),'Current typed iteration proof differs from bound CPU proof: '+name)
    return selected,actual


def _single_file(ref,*,immutable):
    from .scene_store import _source_shape
    _source_shape(ref)
    path=real_path(ref['file']);s=path.stat()
    need(isinstance(ref['file'],str) and str(path)==ref['file'],'Canonical scene source path required; lexical aliases are forbidden')
    import stat
    need(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'Scene layers require single-link regular files, without aliases')
    if immutable:need(s.st_mode&0o222==0,'Scene dependency must be immutable')
    return path


def _closed_layer(layer,sublayers):
    """Inspect before composition; only internal references and approved MDL token."""
    from pxr import Sdf,UsdUtils
    need(list(layer.subLayerPaths)==list(sublayers),'Unknown or cyclic USD sublayer dependency')
    def check(path):
        spec=layer.GetObjectAtPath(path)
        if spec is None:return
        for name in ('references','payload'):
            if name in spec.ListInfoKeys():
                op=spec.GetInfo(name)
                for field in ('explicitItems','prependedItems','appendedItems','addedItems','deletedItems','orderedItems'):
                    need(all(not item.assetPath for item in getattr(op,field)), 'Unknown external USD reference dependency')
    layer.Traverse(Sdf.Path.absoluteRootPath,check)
    def asset(path):
        need(not path or path=='OmniPBR.mdl' or path in sublayers,'Unsupported unbound asset dependency: '+path)
        return path
    # A temporary copy makes path checking independent of immutable layers.
    private=Sdf.Layer.CreateAnonymous();private.TransferContent(layer);UsdUtils.ModifyAssetPaths(private,asset)


def validate_inventory(inventory):
    from pxr import Sdf,Usd
    need(isinstance(inventory,dict) and set(inventory)=={'schema','primary','base'} and
        inventory['schema']==INVENTORY_SCHEMA,'Explicit closed USD dependency inventory required')
    paths=[];inodes=[]
    for role in ('primary','base'):
        ref=inventory[role];path=_single_file(ref,immutable=True);read_source(ref)
        paths.append(path);inodes.append((path.stat().st_dev,path.stat().st_ino))
    need(len(set(paths))==len(set(inodes))==2,'Layer dependency aliases or cycles are forbidden')
    primary=Sdf.Layer.OpenAsAnonymous(str(paths[0]));base=Sdf.Layer.OpenAsAnonymous(str(paths[1]))
    need(primary is not None and base is not None,'Missing or invalid USD dependencies')
    _closed_layer(primary,[str(paths[1])]);_closed_layer(base,())
    primary.subLayerPaths=[base.identifier]
    composed=Usd.Stage.Open(primary)
    need(composed is not None and not composed.GetCompositionErrors(),
        'Invalid or cyclic internal USD reference dependencies')
    need(all(layer.anonymous for layer in composed.GetUsedLayers()),'Dependency validation must use only private anonymous layers')
    return inventory


def private_layered_stage(inventory):
    from pxr import Sdf,Usd
    validate_inventory(inventory)
    primary=Sdf.Layer.OpenAsAnonymous(inventory['primary']['file'])
    base=Sdf.Layer.OpenAsAnonymous(inventory['base']['file'])
    primary.subLayerPaths=[base.identifier]
    stage=Usd.Stage.Open(primary)
    need(stage and not stage.GetCompositionErrors() and all(x.anonymous for x in stage.GetUsedLayers()),
        'All layered runtime USD layers must be private anonymous layers')
    return stage


def _subtree_signature(spec):
    def row(s):
        return dict(info={k:typed_value(s.GetInfo(k)) for k in s.ListInfoKeys()},
            properties=[dict(name=p.name,type=type(p).__name__,info={k:typed_value(p.GetInfo(k)) for k in p.ListInfoKeys()}) for p in s.properties],
            children=[dict(name=c.name,content=row(c)) for c in s.nameChildren])
    return digest(row(spec))


def _factor(stages,inputs,output_root):
    from pxr import Sdf,Usd
    output_root=real_path(output_root,exists=False)
    need(not output_root.exists(),'Exclusive new output root required')
    need(all(output_root!=Path(r['file']).parent and output_root not in Path(r['file']).parents
        and Path(r['file']).parent not in output_root.parents for r in inputs),'Output overlaps immutable inputs')
    snapshots=[scene_snapshot(s) for s in stages]
    layers=[s.GetRootLayer() for s in stages]
    static=[]
    def select(spec):
        peers=[l.GetPrimAtPath(spec.path) for l in layers]
        if all(p is not None for p in peers) and len({_subtree_signature(p) for p in peers})==1:
            static.append(str(spec.path));return
        for child in spec.nameChildren:select(child)
    for spec in layers[0].rootPrims:select(spec)
    need(static,'No fully identical complete static USD subtree')
    output_root.mkdir(parents=True,exist_ok=False)
    base=Sdf.Layer.CreateAnonymous('static-base.usda')
    # USD merges sibling order from weak layers. Empty sibling shells in the
    # base retain the original order without authoring new primOrder metadata.
    ancestors=set()
    for path in static:
        parent=Sdf.Path(path).GetParentPath()
        while parent!=Sdf.Path.absoluteRootPath:
            ancestors.add(str(parent));parent=parent.GetParentPath()
    def base_skeleton(spec):
        Sdf.CreatePrimInLayer(base,spec.path)
        if str(spec.path) in ancestors:
            for child in spec.nameChildren:base_skeleton(child)
    for spec in layers[0].rootPrims:base_skeleton(spec)
    for path in static:
        need(Sdf.CopySpec(layers[0],path,base,path),'Static subtree copy failed')
    base_path=output_root/'static_base.usdc';need(base.Export(str(base_path)),'Static base export failed');base_path.chmod(0o444)
    base_ref=source_ref(base_path);scenes=[]
    for i,(stage,source,expected) in enumerate(zip(stages,inputs,snapshots)):
        primary=Sdf.Layer.CreateAnonymous('body-primary.usda');primary.TransferContent(stage.GetRootLayer())
        for path in static:
            shell=Sdf.Layer.CreateAnonymous();Sdf.CreatePrimInLayer(shell,path)
            need(Sdf.CopySpec(shell,path,primary,path),'Primary static shell replacement failed')
        primary.subLayerPaths=[str(base_path)]
        directory=output_root/('scene_'+str(i).zfill(4));directory.mkdir()
        path=directory/'candidate_physics.usdc';need(primary.Export(str(path)),'Primary export failed');path.chmod(0o444)
        inventory=dict(schema=INVENTORY_SCHEMA,primary=source_ref(path),base=deepcopy(base_ref))
        validate_inventory(inventory)
        # Independently re-open saved file bytes and the anonymous runtime route.
        saved_primary=Sdf.Layer.OpenAsAnonymous(str(path))
        saved_base=Sdf.Layer.OpenAsAnonymous(str(base_path))
        need(saved_primary is not None and saved_base is not None,'Cannot reopen saved authored USD layers')
        def primary_authored(layer):
            need(list(layer.subLayerPaths)==[str(base_path)] and
                list(layer.subLayerOffsets)==[Sdf.LayerOffset()],
                'Exact declared base path and identity sublayer offset required')
            raw=Sdf.Layer.CreateAnonymous();raw.TransferContent(layer)
            for field in ('subLayers','subLayerOffsets'):raw.pseudoRoot.ClearInfo(field)
            return _layer_snapshot(raw)
        need(primary_authored(saved_primary)==primary_authored(primary),
            'Layered authored fields/list operations changed in saved primary')
        need(_layer_snapshot(saved_base)==_layer_snapshot(base),
            'Layered authored fields/list operations changed in saved base')
        saved=Usd.Stage.Open(saved_primary)
        need(scene_snapshot(saved)==expected,'Saved layered USD semantics differ from dense selected source')
        private=private_layered_stage(inventory)
        need(scene_snapshot(private)==expected,'Private layered runtime semantics differ')
        # Reconstruct from both saved layers, so an export cannot lose authored
        # opinions hidden by schema fallbacks. Only the declared sublayer fields
        # differ; every original authored field/list operation remains exact.
        rebuilt=Sdf.Layer.CreateAnonymous();rebuilt.TransferContent(saved_primary);rebuilt.pseudoRoot.ClearInfo('subLayers');rebuilt.pseudoRoot.ClearInfo('subLayerOffsets')
        for subtree in static:need(Sdf.CopySpec(saved_base,subtree,rebuilt,subtree),'Authoring reconstruction failed')
        need(_layer_snapshot(rebuilt)==_layer_snapshot(stage.GetRootLayer()),'Layered authored fields/list operations changed')
        scenes.append(dict(original=deepcopy(source),dependency_inventory=inventory,
            composed_semantics_sha256=digest(expected),authored_semantics_sha256=digest(_layer_snapshot(stage.GetRootLayer())),
            saved_USD_readback_exact=True,private_runtime_readback_exact=True))
    unique={}
    for ref in inputs:
        need(ref['file'] not in unique or unique[ref['file']]==ref,'Conflicting original source bindings')
        unique[ref['file']]=ref
    original_bytes=sum(r['bytes'] for r in unique.values());saved_bytes=base_ref['bytes']+sum(x['dependency_inventory']['primary']['bytes'] for x in scenes)
    need(saved_bytes<original_bytes*.9,'Lossless factoring did not save at least ten percent of selected scene bytes')
    for ref in inputs:read_source(ref)
    return dict(schema=RECEIPT_SCHEMA,base=base_ref,scenes=scenes,static_subtrees=static,
        original_bytes=original_bytes,saved_bytes=saved_bytes,representation_change_only=True,
        physics_executed=False,physical_validation=False,training_eligible=False)


def factor_scene_layers(inputs,*,output_root):
    """Low-level exact-file CPU factorer; population callers use author_population."""
    need(isinstance(inputs,list) and len(inputs)>=2,'At least two selected scenes required')
    return _factor([_private_source(r) for r in inputs],inputs,output_root)


def _population_metadata(selected):
    """Authenticate ALL identities/manifest headers before any payload hash/read."""
    from . import m2_pilot_config as config,m2_contract as contract
    from .geometry_epoch import require_current,GEOMETRY_ID,GEOMETRY_SOURCE_SHA256
    from .patient_scene import layout_translation
    need(isinstance(selected,list) and len(selected)>=2,'Ordered selected population of at least two required')
    rows=[]
    for selection in selected:
        need(isinstance(selection,dict) and set(selection)=={'preparation_config','prepared_complete','patient_receipt'},
            'Exact selected-scene metadata references required')
        for ref in selection.values():
            need(isinstance(ref,dict) and set(ref)=={'file','sha256'},'Exact file/SHA metadata reference required')
            _single_file(dict(ref,bytes=real_path(ref['file']).stat().st_size),immutable=False)
        prep,complete,patient=[config.bound_json(selection[k]) for k in ('preparation_config','prepared_complete','patient_receipt')]
        spec=prep.get('patient_asset',{});require_current(prep);require_current(spec)
        need(prep.get('prepare_only') is True and complete.get('completed') is True and
            complete.get('geometry_id')==GEOMETRY_ID and complete.get('source_sha256')==GEOMETRY_SOURCE_SHA256 and
            complete.get('patient')==patient and patient.get('non_patient_scene_unchanged') is True,
            'Completed current unchanged-patient preparation required')
        split,subject=patient.get('split'),patient.get('identity')
        need(split in ('train','validation') and spec.get('expected_split','train')==split and
            subject not in contract.QUARANTINED_IDENTITIES,'TRAIN/VALIDATION split only; quarantined identities forbidden')
        splits=config.bound_json(dict(file=spec['identity_split'],sha256=contract.IDENTITY_SPLIT_SHA256))['splits']
        need([name for name,ids in splits.items() if subject in ids]==[split] and
            patient.get('identity_split_sha256')==contract.IDENTITY_SPLIT_SHA256,'Frozen identity split mismatch')
        directory=real_path(spec['directory'])
        manifest=config.bound_json(dict(file=str(directory/'asset_manifest.json'),sha256=spec['manifest_sha256']))
        summary=config.bound_json(dict(file=str(directory/'summary.json'),sha256=spec['summary_sha256']))
        need(summary.get('identity_split_sha256')==contract.IDENTITY_SPLIT_SHA256 and
            summary.get('schema')=='m2_observed_seated_retarget_v1','Observed identity provenance required')
        body=[r for r in manifest if r.get('file')==spec.get('file')]
        need(len(body)==1 and body[0].get('identity')==subject and body[0].get('split')==split and
            body[0].get('sha256')==patient.get('asset_sha256') and body[0].get('mesh_scale')==1. and
            body[0].get('numeric_candidate_screen_passed') is True,'Unique manifest body identity/receipt mismatch')
        need(Path(spec['file']).name==spec['file'] and
            np.array_equal(np.asarray(patient.get('layout_translation_world_m')),np.asarray(layout_translation(spec))),
            'Authenticated body path or layout mismatch')
        root=real_path(selection['prepared_complete']['file']).parent
        need(real_path(selection['patient_receipt']['file'])==root/'patient_receipt.json','Exact installed patient receipt path required')
        need('scene_dependency_inventory' not in complete,'Author from original dense preparation only')
        rows.append((prep,complete,patient,root))
    return rows


def _write_new(path,value):
    path=real_path(path,exists=False)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n')
    path.chmod(0o444)
    return source_ref(path)


def author_population(*,selected,output_root,iteration_spec=None):
    """Authenticate an ordered population, derive privately, export one base + N primaries.

    `selected` holds exact preparation_config/prepared_complete/patient_receipt
    file+SHA references. All outputs occupy a new exclusive root. Failed output
    trees are retained and can never be retried in place.
    """
    from . import m2_pilot_config as config
    from .scene_store import FILES,OPTIONAL
    metadata=_population_metadata(selected)
    need(iteration_spec is None or iteration_spec==ITERATIONS,'Only fixed qualified 64/1/.02 iteration specification supported')
    output_root=real_path(output_root,exists=False);need(not output_root.exists(),'Exclusive new output root required')
    geometry_paths=[geometry_source_path(prep) for prep,_,_,_ in metadata]
    stages=[];inputs=[];proofs=[];bindings=[]
    # Only after the complete population's identity preflight may _prepared
    # authenticate body/scene/collision payload bytes.
    for selection,(prep,complete,patient,root) in zip(selected,metadata):
        config._prepared(**selection)
        files=[]
        for path in sorted(root.rglob('*')):
            real_path(path)
            if path.is_dir():continue
            ref=source_ref(path);_single_file(ref,immutable=False);files.append(ref)
        for ref in selection.values():
            current=source_ref(ref['file']);need(current['sha256']==ref['sha256'],'Original metadata changed')
            if current not in files:files.append(current)
        spec=prep['patient_asset'];files.extend(source_ref(p) for p in (
            Path(spec['directory'])/spec['file'],Path(spec['directory'])/'asset_manifest.json',
            Path(spec['directory'])/'summary.json',spec['identity_split'],geometry_source_path(prep)))
        for ref in files:_single_file(ref,immutable=False)
        source=source_ref(root/'candidate_physics.usdc');stage=_private_source(source)
        if iteration_spec is not None:
            need(prep.get('physics')==dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02),'Original 32/1/.02 physics required')
            stage,proof=apply_qualified_iterations(stage)
        else:
            need(prep.get('physics')==ITERATIONS,'Already qualified 64/1/.02 physics required')
            stage,proof=replay_qualified_iterations(stage,None,already64=True)
            proof['source_already_qualified64']=True
        unique={}
        for ref in files:
            need(ref['file'] not in unique or unique[ref['file']]==ref,'Conflicting original input bindings')
            unique[ref['file']]=ref
        files=[unique[name] for name in sorted(unique)]
        stages.append(stage);inputs.append(source);proofs.append(proof);bindings.append(files)
    result=_factor(stages,inputs,output_root);prepared=[]
    receipt_ref=_write_new(output_root/'scene_authoring_receipt.json',dict(result,iteration_proofs=proofs,
        runtime_equivalence_policy=RUNTIME_EQUIVALENCE_POLICY,
        original_selected=deepcopy(selected),input_bindings=bindings))
    for i,(selection,meta,item,proof,files) in enumerate(zip(selected,metadata,result['scenes'],proofs,bindings)):
        prep,complete,patient,root=meta;directory=Path(item['dependency_inventory']['primary']['file']).parent
        copied={}
        for name,filename in {**FILES,**OPTIONAL}.items():
            if name in ('scene','physics_profile') or not (root/filename).exists():continue
            old=source_ref(root/filename);target=directory/filename;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(read_source(old))
            target.chmod(0o444);copied[name]=source_ref(target)
        profile=dict(proof,original_scene=inputs[i],inherited_profile=source_ref(root/'physics_profile.json'),
            new_scene=item['dependency_inventory']['primary'],representation_lineage=receipt_ref,
            saved_USD_readback_exact=True)
        profile_ref=_write_new(directory/'physics_profile.json',profile)
        new_complete=deepcopy(complete);new_complete.update(
            effective_scene_sha256=item['dependency_inventory']['primary']['sha256'],
            scene_dependency_inventory=deepcopy(item['dependency_inventory']),scene_authoring_receipt=receipt_ref,
            solver_isolation_lineage={k:profile_ref[k] for k in ('file','sha256')},
            preparation_backend='cpu_lossless_static_layer_and_exact_existing_iteration_authoring')
        complete_ref=_write_new(directory/'COMPLETE.json',new_complete)
        new_prep=deepcopy(prep);new_prep['physics']=deepcopy(ITERATIONS)
        prep_ref=_write_new(directory/'prepare.json',new_prep)
        short=lambda ref:{k:ref[k] for k in ('file','sha256')}
        prepared.append(dict(preparation_config=short(prep_ref),prepared_complete=short(complete_ref),
            patient_receipt=short(copied['patient_receipt']),physics_profile=short(profile_ref),original=deepcopy(selection),
            input_bindings=files,dependency_inventory=deepcopy(item['dependency_inventory']),
            baseline_scene_sha256=inputs[i]['sha256'],iteration_proof=deepcopy(proof),
            composed_semantics_sha256=item['composed_semantics_sha256'],scene_authoring_receipt=receipt_ref))
    for files in bindings:
        for ref in files:read_source(ref)
    for (prep,_,_,_),expected in zip(metadata,geometry_paths):
        need(geometry_source_path(prep)==expected,'Original geometry source boundary changed')
    return dict(schema=RECEIPT_SCHEMA,base=result['base'],prepared=prepared,receipt=receipt_ref,
        original_bytes=result['original_bytes'],saved_bytes=result['saved_bytes'],static_subtrees=result['static_subtrees'])


def authoring_input_inventory(receipt,metadata):
    """Validate canonical authoring closure using headers/stat only, not payloads."""
    selected=receipt.get('original_selected');groups=receipt.get('input_bindings')
    scenes=receipt.get('scenes')
    need(receipt.get('schema')==RECEIPT_SCHEMA and isinstance(groups,list) and
        isinstance(scenes,list) and len(groups)==len(selected)==len(metadata)==len(scenes),
        'Closed original authoring input inventory required')
    union={}
    for selection,group,meta,item in zip(selected,groups,metadata,scenes):
        prep,complete,patient,root=meta
        need(isinstance(group,list),'Authoring input group must be a list')
        names=[];rows={}
        for ref in group:
            path=_single_file(ref,immutable=False)
            need(path.stat().st_size==ref['bytes'],'Authoring source size changed')
            names.append(ref['file']);rows[ref['file']]=ref
            need(ref['file'] not in union or union[ref['file']]==ref,'Conflicting authoring source binding')
            union[ref['file']]=ref
        need(names==sorted(set(names)),'Canonical unique authoring source paths required')
        expected=set()
        for path in root.rglob('*'):
            real_path(path)
            if path.is_dir():continue
            expected.add(str(path))
        expected.update(ref['file'] for ref in selection.values())
        spec=prep['patient_asset']
        expected.update(str(real_path(p)) for p in (Path(spec['directory'])/spec['file'],
            Path(spec['directory'])/'asset_manifest.json',Path(spec['directory'])/'summary.json',
            spec['identity_split'],geometry_source_path(prep)))
        need(set(names)==expected,'Missing or unknown original authoring input path')
        for ref in selection.values():
            need(rows[ref['file']]['sha256']==ref['sha256'],'Original metadata input reference changed')
        scene=rows[str(root/'candidate_physics.usdc')]
        need(scene==item.get('original') and scene['sha256']==complete.get('effective_scene_sha256'),
            'Original scene input reference changed')
        need(rows[str(Path(spec['directory'])/spec['file'])]['sha256']==patient['asset_sha256'],
            'Original body input reference changed')
        if prep['geometry_source']==str(HISTORICAL_GEOMETRY_RAW):
            from .geometry_epoch import GEOMETRY_SOURCE_SHA256
            need(rows[str(geometry_source_path(prep))]['sha256']==GEOMETRY_SOURCE_SHA256,
                'Approved canonical historical geometry binding changed')
    return [union[name] for name in sorted(union)]


def geometry_source_path(preparation):
    """Resolve only the existing approved historical role; new layers forbid aliases.

    _prepared authenticates this role's frozen geometry SHA before body use.
    Its old literal stays in the configuration; accounting binds the canonical
    target file, exactly as the immutable iteration helper did.
    """
    raw=preparation.get('geometry_source')
    if raw!=str(HISTORICAL_GEOMETRY_RAW):
        path=real_path(raw)
        need(isinstance(raw,str) and str(path)==raw,'Canonical geometry source required; aliases forbidden')
        return path
    link=HISTORICAL_GEOMETRY_LINK
    need(link.is_symlink() and os.readlink(link)==HISTORICAL_GEOMETRY_LINK_TARGET,
        'Only the exact pre-existing historical geometry alias is supported')
    target=real_path(HISTORICAL_GEOMETRY_TARGET)
    need(HISTORICAL_GEOMETRY_RAW.resolve()==target and
        not any(path.is_symlink() for path in (HISTORICAL_GEOMETRY_RAW,*HISTORICAL_GEOMETRY_RAW.parents) if path!=link),
        'Unexpected historical geometry alias target or parent')
    return target


def _layered_authored_snapshot(inventory,static_subtrees):
    """Read every authored value without resolving AssetPath values through USD."""
    from pxr import Sdf
    validate_inventory(inventory)
    primary=Sdf.Layer.OpenAsAnonymous(inventory['primary']['file'])
    base=Sdf.Layer.OpenAsAnonymous(inventory['base']['file'])
    need(isinstance(static_subtrees,list) and static_subtrees and
        all(isinstance(path,str) and Sdf.Path(path).IsAbsolutePath() and
            Sdf.Path(path).IsPrimPath() for path in static_subtrees) and
        len(set(static_subtrees))==len(static_subtrees),'Exact static subtree inventory required')
    for field in ('subLayers','subLayerOffsets'):primary.pseudoRoot.ClearInfo(field)
    for path in static_subtrees:
        need(base.GetPrimAtPath(path) is not None and Sdf.CopySpec(base,path,primary,path),
            'Layered authored reconstruction failed')
    return _layer_snapshot(primary)


def current_raw_sidecar_record(prepared_complete,authoring_receipt,prepared_sources,receipt,item,index):
    """Strict immutable provenance, deliberately without a new COMPLETE/config cycle."""
    from .scene_store import SDK_LAYER_CURRENT_RAW_POLICY
    return dict(schema=CURRENT_RAW_SIDECAR_SCHEMA,dependency_policy=SDK_LAYER_CURRENT_RAW_POLICY,
        historical_prepared_complete=deepcopy(prepared_complete),authoring_receipt=deepcopy(authoring_receipt),
        historical_prepared_sources=deepcopy(prepared_sources),
        scene_index=index,dependency_inventory=deepcopy(item['dependency_inventory']),
        original_selected=deepcopy(receipt['original_selected'][index]),original_dense=deepcopy(item['original']),
        original_iteration_proof=deepcopy(receipt['iteration_proofs'][index]),
        historical_authored_semantics_sha256=item['authored_semantics_sha256'],
        historical_digest_scope='original_authoring_cpu_sdf_registry_only',
        physics_executed=False,physical_validation=False,training_eligible=False,USD_written=False)


def historical_prepared_sources(prepared_complete,complete,receipt,item,index,metadata):
    """Authenticate historical metadata first; derive payload refs without opening them."""
    from .scene_store import FILES,OPTIONAL,_strict_json,_exact_json
    root=Path(prepared_complete['file']).parent;original_root=metadata[index][3]
    originals={ref['file']:ref for ref in receipt['input_bindings'][index]}
    sources={}
    for name,filename in {**FILES,**OPTIONAL}.items():
        original=originals.get(str(original_root/filename));path=root/filename
        if name in OPTIONAL:
            need(path.exists()==(original is not None),'Historical optional prepared inventory changed')
            if original is None:continue
        if name=='scene':ref=deepcopy(item['dependency_inventory']['primary'])
        elif name=='physics_profile':
            short=complete.get('solver_isolation_lineage')
            need(isinstance(short,dict) and set(short)=={'file','sha256'} and short['file']==str(path),
                'Exact historical prepared physics lineage required')
            ref=dict(short,bytes=real_path(path).stat().st_size)
        else:
            need(original is not None,'Missing original copied prepared source')
            ref=dict(original,file=str(path))
        _single_file(ref,immutable=True);sources[name]=ref
    prep_ref=source_ref(root/'prepare.json');_single_file(prep_ref,immutable=True)
    expected=deepcopy(metadata[index][0]);expected['physics']=deepcopy(ITERATIONS)
    need(_exact_json(_strict_json(read_source(prep_ref)),expected),
        'Historical prepared configuration differs from exact original fixed64 authoring')
    sources['preparation_config']=prep_ref
    for name,ref in sources.items():
        if name not in ('scene','collision_geometry','collision_arrays'):_strict_json(read_source(ref))
    return sources


def issue_current_raw_equivalence(*,prepared_complete,output_file):
    """CPU-only JSON issuer. Every consumer must independently execute the RAW proof."""
    from .scene_store import (_strict_json,_layered_metadata,_layered_policy,SDK_LAYER_DEPENDENCY_POLICY,
        _selected_raw_expected)
    _single_file(prepared_complete,immutable=True)
    complete=_strict_json(read_source(prepared_complete))
    need(_layered_policy(complete)==SDK_LAYER_DEPENDENCY_POLICY,'Issue only from an immutable historical v3 COMPLETE')
    _strict_json(read_source(complete['scene_authoring_receipt']))
    receipt,item,index,metadata=_layered_metadata(complete)
    output_file=real_path(output_file,exists=False)
    original_roots=[meta[3] for meta in metadata]
    original_roots.extend((Path(prepared_complete['file']).parent,Path(complete['scene_authoring_receipt']['file']).parent))
    need(all(root not in output_file.parents for root in original_roots),'Current RAW sidecar output overlaps historical source tree')
    need(prepared_complete['file']==str(Path(item['dependency_inventory']['primary']['file']).parent/'COMPLETE.json'),
        'Historical prepared COMPLETE must belong to the exact primary parent')
    prepared_sources=historical_prepared_sources(prepared_complete,complete,receipt,item,index,metadata)
    selected,recheck=_selected_raw_expected(receipt,item,index,metadata)
    need(_layer_snapshot(selected.GetRootLayer())==_layered_authored_snapshot(item['dependency_inventory'],receipt['static_subtrees']),
        'Layered runtime differs from the full current-context RAW dense reference')
    record=current_raw_sidecar_record(prepared_complete,complete['scene_authoring_receipt'],prepared_sources,receipt,item,index)
    recheck()
    for ref in item['dependency_inventory'].values():
        if isinstance(ref,dict):read_source(ref)
    read_source(prepared_complete);read_source(complete['scene_authoring_receipt'])
    for ref in prepared_sources.values():read_source(ref)
    return _write_new(output_file,record)
