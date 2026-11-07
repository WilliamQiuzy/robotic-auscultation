"""Metadata-only lossless deployment lineage; no USD, NPZ or native execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_ausc_given_contact_deployment import module, scene_metadata_fixture, frozen_reference_fixture
from test_ausc_given_contact_pose import reference, metadata


def population(tmp_path, monkeypatch, mutate=None, *, dense_alias=False, scene_counts=(20,4), targets=None, ordered_layers=False):
    m=module();r=m.runtime
    from sim.ausc_policy import layered_scene_authoring as layers, scene_store as store
    refs={}
    def save(path,value):
        ref=r.write_new(path,value);refs[ref['path']]=ref;return ref
    def short(ref):return dict(file=ref.get('file',ref.get('path')),sha256=ref['sha256'])
    def full(ref):
        saved=refs[ref.get('file',ref.get('path'))]
        return dict(file=saved['path'],sha256=saved['sha256'],bytes=saved['bytes'])
    def path_ref(ref):
        value=full(ref);return dict(path=value['file'],sha256=value['sha256'],bytes=value['bytes'])
    def source(path,digest='d'*64):
        ref=dict(file=str(path),sha256=digest,bytes=1)
        refs[str(path)]=dict(path=str(path),sha256=digest,bytes=1)
        return ref
    def dense_source(root):return source(root/('same-content-alias.usdc' if dense_alias else 'candidate_physics.usdc'),'1'*64)
    splits=json.loads(r.shared.contract.IDENTITY_SPLIT_PATH.read_text())['splits']
    prototype=tmp_path/'prototype';prototype.mkdir();snapshot=reference(prototype).snapshot()
    scenes=[];selected=[];prepared=[];originals={};allocations=dict(train=[],validation=[])
    for split,count in zip(('train','validation'),scene_counts):
        for i,subject in enumerate([s for s in splits[split] if s not in r.shared.contract.QUARANTINED_IDENTITIES][:count]):
            proposal=split+'-'+str(i);oldroot=tmp_path/'original'/proposal
            newroot=tmp_path/'derived'/('scene_'+str(len(scenes)).zfill(4) if ordered_layers else proposal)
            patient=dict(identity=subject,split=split,asset_sha256='a'*64,non_patient_scene_unchanged=True)
            oldpatient=save(oldroot/'patient_receipt.json',patient);newpatient=save(newroot/'patient_receipt.json',patient)
            prep=dict(patient_asset=dict(directory=str(oldroot),file='body.npz',expected_split=split),
                physics=dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02),prepare_only=True)
            oldprep=save(oldroot/'prepare.json',prep);newprep=save(newroot/'prepare.json',dict(prep,physics=layers.ITERATIONS))
            oldcomplete=save(oldroot/'COMPLETE.json',dict(completed=True,patient=patient,effective_scene_sha256='1'*64))
            scene=dict(proposal_id=proposal,subject_id=subject,split=split,template=tuple(sorted(m.POSE_TEMPLATES))[i%2],
                preparation_config=short(oldprep),prepared_complete=short(oldcomplete),patient_receipt=short(oldpatient))
            scenes.append(scene);selection={key:scene[key] for key in ('preparation_config','prepared_complete','patient_receipt')}
            selected.append(selection)
            inv=dict(schema=layers.INVENTORY_SCHEMA,primary=source(newroot/'candidate_physics.usdc'),base=source(tmp_path/'derived/static_base.usdc'))
            proof=dict(schema='m2_solver_iteration_only_authoring_v2',profile=layers.ITERATIONS,
                rigid_articulation_roles=[[str(j),'physxRigidBody'] for j in range(24)],
                only_iteration_changes=[dict(path=str(j),attribute='physxRigidBody:solverPositionIterationCount',before=32,after=64) for j in range(24)],
                float32_armature_attributes=['armature'],armature_authoring_changed=False,geometry_camera_drives_mass_contact_unchanged=True,
                exhaustive_metadata_API_relationship_connection_time_sample_check=True,physics_executed=False,physical_validation=False,training_eligible=False,
                **{k:'e'*64 for k in ('before_stage_semantics_sha256','after_stage_semantics_sha256','before_authored_sha256','after_authored_sha256')})
            prepared.append(dict(original=selection,preparation_config=short(newprep),patient_receipt=short(newpatient),
                dependency_inventory=inv,input_bindings=[dict(file=x['path'],sha256=x['sha256'],bytes=x['bytes']) for x in (oldprep,oldcomplete,oldpatient)],
                baseline_scene_sha256='1'*64,iteration_proof=proof,composed_semantics_sha256='f'*64))
            originals[proposal]=(prep,oldcomplete,newroot)
            for j,target in enumerate(targets or [side+str(n) for side in 'LR' for n in range(1,6)]):
                oldplan=dict(name=target,success=True,patient_provenance=dict(identity=subject,binding=short(oldpatient)),numeric_plan=[j,1,2])
                plan=save(oldroot/'plans'/(target+'.json'),oldplan)
                allocations[split].append(dict(slot_id=proposal+'-'+target,subject_id=subject,target=target,proposal_id=proposal,
                    scene_ref=proposal,fresh_plan=short(plan),seed=2026102700+j))
    authorreceipt=save(tmp_path/'derived/scene_authoring_receipt.json',dict(schema=layers.RECEIPT_SCHEMA,original_selected=selected,
        representation_change_only=True,runtime_equivalence_policy=layers.RUNTIME_EQUIVALENCE_POLICY,
        physics_executed=False,physical_validation=False,training_eligible=False,
        base=prepared[0]['dependency_inventory']['base'],static_subtrees=['/Static'],original_bytes=1000,saved_bytes=100,
        scenes=[dict(original=dense_source(Path(s['prepared_complete']['file']).parent),
            dependency_inventory=p['dependency_inventory'],composed_semantics_sha256=p['composed_semantics_sha256'],
            saved_USD_readback_exact=True,private_runtime_readback_exact=True) for s,p in zip(scenes,prepared)],
        iteration_proofs=[p['iteration_proof'] for p in prepared],input_bindings=[p['input_bindings'] for p in prepared]))
    lineages=[]
    for scene,p in zip(scenes,prepared):
        prep,oldcomplete,newroot=originals[scene['proposal_id']]
        p['scene_authoring_receipt']=dict(file=authorreceipt['path'],bytes=authorreceipt['bytes'],sha256=authorreceipt['sha256'])
        profile=save(newroot/'physics_profile.json',dict(p['iteration_proof'],original_scene=dense_source(Path(oldcomplete['path']).parent),
            inherited_profile=source(Path(oldcomplete['path']).parent/'physics_profile.json'),new_scene=p['dependency_inventory']['primary'],
            representation_lineage=p['scene_authoring_receipt'],saved_USD_readback_exact=True))
        p['physics_profile']=short(profile)
        complete=json.loads(Path(oldcomplete['path']).read_text());complete.update(effective_scene_sha256='d'*64,
            scene_dependency_inventory=p['dependency_inventory'],scene_authoring_receipt=p['scene_authoring_receipt'],
            solver_isolation_lineage=p['physics_profile'],preparation_backend='cpu_lossless_static_layer_and_exact_existing_iteration_authoring')
        p['prepared_complete']=short(save(newroot/'COMPLETE.json',complete))
        derived=dict(scene,**{key:p[key] for key in ('preparation_config','prepared_complete','patient_receipt')},derivation=p)
        lineages.append(dict(proposal_id=scene['proposal_id'],original=scene,derived=derived))
    def scene_sources(scene,preparation):
        root=Path(scene['prepared_complete']['file']).parent
        sources=dict(preparation_config=full(preparation),prepared_complete=full(scene['prepared_complete']))
        for role,name in store.FILES.items():
            if role=='patient_receipt':sources[role]=full(scene[role])
            elif role=='physics_profile' and 'derivation' in scene:sources[role]=full(scene['derivation'][role])
            elif role=='scene' and 'derivation' in scene:sources[role]=deepcopy(scene['derivation']['dependency_inventory']['primary'])
            else:sources[role]=source(root/name,'1'*64 if role=='scene' else 'd'*64)
        sources.update(sdk_mdl_manifest=source('/fixed/sdk/manifest.json','6'*64),sdk_mdl_0000=source('/fixed/sdk/first.mdl','7'*64))
        if 'derivation' in scene:
            sources['scene_base']=deepcopy(scene['derivation']['dependency_inventory']['base'])
            sources['scene_authoring_receipt']=deepcopy(scene['derivation']['scene_authoring_receipt'])
        return sources
    by_id={entry['proposal_id']:entry for entry in lineages};attempts=[];plans=[];rows=[];baselines={}
    for i,a in enumerate(allocations['train']+allocations['validation']):
        lineage=by_id[a['proposal_id']];scene=lineage['original'];derived=lineage['derived'];split=scene['split']
        oldplan=json.loads(Path(a['fresh_plan']['file']).read_text());newplan=deepcopy(oldplan)
        newplan['patient_provenance']['binding']=derived['patient_receipt']
        newplanref=save(tmp_path/'plans'/(str(i+1).zfill(3)+'.json'),newplan)
        metas=[]
        for patient,plan in ((scene['patient_receipt'],a['fresh_plan']),(derived['patient_receipt'],short(newplanref))):
            snap=deepcopy(snapshot);snap['target_id']=a['target'];snap['sources']=dict(task_manifest=patient,patient_receipt=patient,fresh_plan=plan)
            ref=r.pose.GivenReference.from_snapshot(snap)
            metas.append(dict(metadata(ref),subject_id=a['subject_id'],split=split,target_id=a['target'],episode_id='episode-'+str(i),handoff_policy_id=r.handoff.POLICY_ID))
        preparations=[]
        for kind,prepared_scene in (('baseline',scene),('runtime',derived)):
            runtimeprep=json.loads(Path(prepared_scene['preparation_config']['file']).read_text());runtimeprep.update(base='/workspace/wam',time_limit_seconds=2400)
            preparations.append(save(tmp_path/kind/(str(i+1).zfill(3)+'.prepare.json'),runtimeprep))
        configs=[]
        for meta,physics,prepared_scene,preparation in zip(metas,(dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02),layers.ITERATIONS),(scene,derived),preparations):
            configs.append(dict(base='/workspace/wam',time_limit_seconds=2400,physics=physics,seed=a['seed'],m2_given_contact_control=dict(metadata=meta),
                fresh_expert=dict(plans=[meta['given_reference']['sources']['fresh_plan']]),
                compact_scene_storage=dict(schema=store.SDK_SCHEMA if len(configs)==0 else store.SDK_LAYER_SCHEMA,
                    dependency_policy=store.SDK_DEPENDENCY_POLICY if len(configs)==0 else store.SDK_LAYER_DEPENDENCY_POLICY,
                    store_root='/workspace/wam/scene_store',batch_id='fixed-lossless',sources=scene_sources(prepared_scene,preparation))))
        baseline=save(tmp_path/'baseline'/(str(i+1).zfill(3)+'.json'),configs[0]);baselines.setdefault(a['proposal_id'],baseline)
        config=save(tmp_path/'runtime'/(str(i+1).zfill(3)+'.json'),configs[1])
        builder=save(tmp_path/'runtime'/(str(i+1).zfill(3)+'.provenance.json'),dict(runtime_preparation=preparations[1]))
        binding=dict(original_reference_sha256=metas[0]['reference_sha256'],derived_reference_sha256=metas[1]['reference_sha256'],
            original_sources=metas[0]['given_reference']['sources'],derived_sources=metas[1]['given_reference']['sources'])
        plans.append(dict(slot_id=a['slot_id'],proposal_id=a['proposal_id'],original=a['fresh_plan'],derived=short(newplanref),
            baseline_config=baseline,runtime=config,builder_receipt=builder,reference_binding=binding))
        attempts.append(dict(slot_id=a['slot_id'],subject_id=a['subject_id'],split=split,target_id=a['target'],config=config,preclassified_reachable=True))
        rows.append(dict(subject_id=a['subject_id'],split=split,target_id=a['target'],pose_template=scene['template'],metadata=metas[1],
            pose_binding=dict(patient_receipt=derived['patient_receipt'],preparation_config=derived['preparation_config'])))
    request=dict(selected=selected,output_root=str(tmp_path/'derived'),iteration_spec=layers.ITERATIONS,
        baseline_configs=[baselines[s['proposal_id']] for s in scenes])
    collector=dict(root='/frozen/collector',commit='c'*40,
        code_commit_file=path_ref(source('/frozen/collector/CODE_COMMIT','8'*64)),
        source_manifest=path_ref(source('/frozen/collector/SOURCE_MANIFEST.json','9'*64)))
    provider=path_ref(source(Path(collector['root'])/'sim/ausc_policy/layered_scene_authoring.py'))
    cpu=dict(executable=dict(invoked_path='/native/bin/python',**path_ref(source('/native/python','b'*64))),collector_root=collector['root'],collector=collector,provider=provider,
        script_sha256='c'*64,request_sha256=r.digest(request),gpu_work=False,physics_advanced=False)
    authored=dict(schema=layers.RECEIPT_SCHEMA,receipt=dict(file=authorreceipt['path'],bytes=authorreceipt['bytes'],sha256=authorreceipt['sha256']),
        base=prepared[0]['dependency_inventory']['base'],prepared=prepared,static_subtrees=['/Static'],original_bytes=1000,saved_bytes=100,
        runtime_equivalence=[dict(schema='m2_lossless_sdk_private_runtime_equivalence_v1',SDK_selected_runtime_readback_exact=True,
            all_layers_private_anonymous=True,dense_copy_created=False,physics_executed=False,training_eligible=False,
            original_sources=json.loads(Path(baselines[s['proposal_id']]['path']).read_text())['compact_scene_storage']['sources'],
            derived_sources=scene_sources(by_id[s['proposal_id']]['derived'],by_id[s['proposal_id']]['derived']['preparation_config']),
            selected_iteration_proof=deepcopy(p['iteration_proof']),composed_semantics_sha256='f'*64) for s,p in zip(scenes,prepared)])
    parity=save(tmp_path/'SDK_RUNTIME_EQUIVALENCE.json',dict(schema='m2_given_lossless_sdk_cpu_receipt_v1',authoring_request=request,
        cpu_execution=cpu,scene_authoring_receipt=authored['receipt'],proofs=authored['runtime_equivalence']))
    author=path_ref(source('/frozen/author.py','a'*64))
    profile=dict(file='/workspace/wam/configs/m2-original022-solver-isolation-20261027-v1/prepared/position64_velocity1/physics_profile.json',
        sha256='3ae2247e5c2f6ad9ac8f8b4c9474a96e97a97dac5cd6bc4fa2d8b58935fd982d')
    source(profile['file'],profile['sha256'])
    report=dict(schema='m2_given_lossless_derivation_map_v1',author_helper=author,selected_profile=profile,profile=deepcopy(layers.ITERATIONS),
        unique_original_scenes=len(scenes),scenes=lineages,plans=plans,numerical_plans_unchanged=True,authoring_request=request,
        layered_authoring=authored,cpu_execution=cpu,runtime_equivalence_receipt=parity,physics_executed=False,training_eligible=False)
    recipe=save(tmp_path/'recipe.json',dict(schema='m2_given_qualified_source_recipe_lossless_v1',author_helper=author,selected_profile=profile,
        qualification_helper=path_ref(source('/frozen/qualification.py','e'*64)),
        numeric_container_codec='framed_npz_gzip_single_member_v1',collector_conditions=dict(physics=layers.ITERATIONS,
            scene_dependencies=dict(schema=store.SDK_LAYER_SCHEMA,dependency_policy=store.SDK_LAYER_DEPENDENCY_POLICY))))
    mapping=dict(scenes=scenes,ordered_allocations=allocations,summaries=dict(train=dict(seed=2026102701),validation=dict(seed=2026102702)))
    mappingref=save(tmp_path/'original_mapping.json',mapping)
    if mutate:mutate(report,rows,refs)
    reportref=save(tmp_path/'DERIVATION_MAP.json',report)
    planref=save(tmp_path/'plan.json',dict(allocation_mapping=mappingref,derivation=reportref,source_recipe=recipe,
        collector=collector,attempts=attempts,accounting_scope=dict(extra_input_refs=[dict(category='config',reference=ref) for ref in refs.values()])))
    monkeypatch.setattr(r.dataset,'ALLOCATION_MAPPING_SHA256',mappingref['sha256'])
    return m,rows,mapping,mappingref,dict(plan=planref),report


def test_public_mapping_bridge_preserves_original_permit_role_and_authenticates_all_derived_rows(tmp_path,monkeypatch):
    m,rows,original,ref,processing,_=population(tmp_path,monkeypatch)
    read=m.runtime.read_bound
    def read_json_only(ref):
        assert Path(ref['path']).suffix=='.json', 'Projection opened a non-JSON source or payload'
        return read(ref)
    monkeypatch.setattr(m.runtime,'read_bound',read_json_only)
    mapping,canonical=m._allocation_mapping(ref,processing)
    assert canonical==ref
    assert original['scenes'][0]['patient_receipt']!=mapping['scenes'][0]['patient_receipt']
    for row in rows:assert m._reference_roles(row,mapping).sha256==row['metadata']['reference_sha256']
    assert not list(tmp_path.rglob('*.usdc')) and not list(tmp_path.rglob('*.npz'))


@pytest.mark.parametrize('fault,role',[(fault,role) for fault in ('missing','sha256','bytes') for role in
    ('primary','base','original_dense','provider','executable','collector_commit','collector_manifest','author_helper',
     'selected_profile','qualification_helper','baseline_source','runtime_source','parity_original_source','parity_derived_source')
    if (fault,role)!=('bytes','selected_profile')])  # A fixed shortref supplies no byte count to contradict.
def test_projection_binds_every_consumed_source_to_frozen_accounting_without_payload(tmp_path,monkeypatch,role,fault):
    def mutate(report,rows,refs):
        authored=report['layered_authoring'];result=authored['prepared'][-1]
        if role=='primary':ref=result['dependency_inventory']['primary']
        elif role=='base':ref=authored['base']
        elif role=='original_dense':ref=json.loads(Path(authored['receipt']['file']).read_text())['scenes'][-1]['original']
        elif role in ('provider','executable'):ref=report['cpu_execution'][role]
        elif role.startswith('collector_'):ref=report['cpu_execution']['collector']['code_commit_file' if role=='collector_commit' else 'source_manifest']
        elif role in ('author_helper','selected_profile'):ref=report[role]
        elif role=='qualification_helper':ref=json.loads((tmp_path/'recipe.json').read_text())[role]
        elif role in ('baseline_source','runtime_source'):
            config=report['plans'][-1]['baseline_config' if role=='baseline_source' else 'runtime']
            ref=json.loads(Path(config['path']).read_text())['compact_scene_storage']['sources']['collision_arrays']
        else:
            sources=authored['runtime_equivalence'][-1]['original_sources' if role=='parity_original_source' else 'derived_sources']
            ref=sources['sdk_mdl_0000']
        path=ref.get('file',ref.get('path'))
        if fault=='missing':refs.pop(path)
        else:
            # The entire report and its JSON receipts remain self-consistent;
            # only the frozen accounting declaration disagrees with this role.
            refs[path]=dict(refs[path],**{fault:'0'*64 if fault=='sha256' else refs[path]['bytes']+1})
    m,_,_,ref,processing,_=population(tmp_path,monkeypatch,mutate)
    with pytest.raises(ValueError,match='frozen plan input inventory'):m._allocation_mapping(ref,processing)
    assert not list(tmp_path.rglob('*.usdc')) and not list(tmp_path.rglob('*.npz'))


@pytest.mark.parametrize('consumer',['baseline','runtime','parity_original','parity_derived'])
def test_each_source_consumer_independently_refuses_an_unregistered_leaf(tmp_path,monkeypatch,consumer):
    def mutate(report,rows,refs):
        alias=dict(file='/unregistered/'+consumer+'.npz',sha256='0'*64,bytes=1)
        if consumer in ('baseline','runtime'):
            ref=report['plans'][-1]['baseline_config' if consumer=='baseline' else 'runtime']
            path=Path(ref['path']);value=json.loads(path.read_text());value['compact_scene_storage']['sources']['collision_arrays']=alias
        else:
            proof=report['layered_authoring']['runtime_equivalence'][-1]
            proof['original_sources' if consumer=='parity_original' else 'derived_sources']['collision_arrays']=alias
            ref=report['runtime_equivalence_receipt'];path=Path(ref['path']);value=json.loads(path.read_text())
            value['proofs']=report['layered_authoring']['runtime_equivalence']
        path.write_text(json.dumps(value));ref.update(module().runtime.reference(path));refs[str(path)]=ref
    m,_,_,ref,processing,_=population(tmp_path,monkeypatch,mutate)
    with pytest.raises(ValueError,match='frozen plan input inventory'):m._allocation_mapping(ref,processing)


def test_registered_same_content_dense_alias_cannot_replace_original_scene_role(tmp_path,monkeypatch):
    m,_,_,ref,processing,_=population(tmp_path,monkeypatch,dense_alias=True)
    with pytest.raises(ValueError,match='original dense scene source role'):m._allocation_mapping(ref,processing)


@pytest.mark.parametrize('fault',['late_identity','late_patient_identity','aliased_role','shuffled_plan','unknown_schema','numerical_plan','unknown_profile','unknown_recipe','builder_preparation','inherited_digest'])
def test_public_bridge_rejects_complete_population_metadata_faults_without_payload(tmp_path,monkeypatch,fault):
    def mutate(report,rows,refs):
        if fault=='late_identity':report['scenes'][-1]['derived']['subject_id']='p000001'
        elif fault=='aliased_role':report['scenes'][-1]['derived']['patient_receipt']=report['scenes'][0]['derived']['patient_receipt']
        elif fault=='shuffled_plan':report['plans'][-2:]=reversed(report['plans'][-2:])
        elif fault=='unknown_schema':report['schema']='unreviewed_derivation_v2'
        elif fault in ('numerical_plan','late_patient_identity'):
            ref=report['plans'][-1]['derived'] if fault=='numerical_plan' else report['scenes'][-1]['derived']['patient_receipt']
            path=Path(ref['file']);value=json.loads(path.read_text());value.update(numeric_plan=[999]) if fault=='numerical_plan' else value.update(identity='p000001')
            path.write_text(json.dumps(value));ref['sha256']=module().runtime.reference(path)['sha256']
            refs[str(path)]=module().runtime.reference(path)  # Rebound metadata still cannot change numerical/identity semantics.
        elif fault=='unknown_profile':report['profile']['position_iterations']=65
        elif fault=='inherited_digest':
            result=report['scenes'][-1]['derived']['derivation'];ref=result['physics_profile'];path=Path(ref['file'])
            value=json.loads(path.read_text());value['inherited_profile']['sha256']='9'*64;path.write_text(json.dumps(value))
            ref['sha256']=module().runtime.reference(path)['sha256'];refs[str(path)]=module().runtime.reference(path)
            ref=result['prepared_complete'];path=Path(ref['file']);value=json.loads(path.read_text())
            value['solver_isolation_lineage']=result['physics_profile'];path.write_text(json.dumps(value))
            ref['sha256']=module().runtime.reference(path)['sha256'];refs[str(path)]=module().runtime.reference(path)
        else:
            ref=refs[str(tmp_path/'recipe.json')] if fault=='unknown_recipe' else report['plans'][-1]['builder_receipt']
            path=Path(ref['path']);value=json.loads(path.read_text())
            if fault=='unknown_recipe':value['schema']='unreviewed_recipe_v2'
            else:value['runtime_preparation']=dict(path=report['scenes'][0]['derived']['preparation_config']['file'],sha256=report['scenes'][0]['derived']['preparation_config']['sha256'])
            path.write_text(json.dumps(value));ref.update(module().runtime.reference(path));refs[str(path)]=ref
    m,rows,_,ref,processing,_=population(tmp_path,monkeypatch,mutate)
    with pytest.raises(ValueError):m._allocation_mapping(ref,processing)
    assert not list(tmp_path.rglob('*.usdc')) and not list(tmp_path.rglob('*.npz'))


def test_public_legacy_mapping_then_roles_remains_unchanged(tmp_path,monkeypatch):
    m,row,original=frozen_reference_fixture(tmp_path);ref=m.runtime.write_new(tmp_path/'mapping.json',original)
    monkeypatch.setattr(m.runtime.dataset,'ALLOCATION_MAPPING_SHA256',ref['sha256'])
    plan=m.runtime.write_new(tmp_path/'legacy-plan.json',dict(allocation_mapping=ref))
    mapping,canonical=m._allocation_mapping(ref,dict(plan=plan))
    assert mapping==original and canonical==ref
    assert m._reference_roles(row,mapping).sha256==row['metadata']['reference_sha256']


def test_module_inventory_binds_lossless_bridge_and_layered_provider():
    inventory=module().module_inventory()
    assert 'sim.ausc_policy.m2_given_contact_lossless_mapping' in inventory
    assert 'sim.ausc_policy.layered_scene_authoring' in inventory


@pytest.mark.parametrize('fault',['valid','schema','recipe','physics','base','authoring'])
def test_scene_metadata_accepts_only_reviewed_layered_conditions(tmp_path,fault):
    from sim.ausc_policy import scene_store as store
    m,row,config,mapping,recipe=scene_metadata_fixture(tmp_path)
    binding=config['compact_scene_storage'];binding['schema']=store.SDK_LAYER_SCHEMA
    binding['dependency_policy']=store.SDK_LAYER_DEPENDENCY_POLICY
    sources=binding['sources'];sources['scene_base']=dict(file='/derived/static_base.usdc',bytes=1,sha256='7'*64)
    sources['scene_authoring_receipt']=dict(file='/derived/scene_authoring_receipt.json',bytes=1,sha256='8'*64)
    recipe['collector_conditions']['scene_dependencies'].update(schema=binding['schema'],dependency_policy=binding['dependency_policy'])
    from sim.ausc_policy import layered_scene_authoring as layers
    config['physics']=deepcopy(layers.ITERATIONS);recipe['collector_conditions']['physics']=deepcopy(layers.ITERATIONS)
    recipe['schema']='m2_given_qualified_source_recipe_lossless_v1'
    scene=mapping['scenes'][0];scene['derivation']=dict(dependency_inventory=dict(schema=layers.INVENTORY_SCHEMA,
        primary=deepcopy(sources['scene']),base=deepcopy(sources['scene_base'])),scene_authoring_receipt=deepcopy(sources['scene_authoring_receipt']))
    if fault=='schema':binding['schema']='unreviewed_v4'
    elif fault=='recipe':recipe['schema']='unreviewed_v2'
    elif fault=='physics':config['physics']['position_iterations']=32
    elif fault=='base':sources['scene_base']['file']='/other/static_base.usdc'
    elif fault=='authoring':sources['scene_authoring_receipt']['sha256']='9'*64
    if fault!='valid':
        with pytest.raises(ValueError):m._scene_metadata(config,row,mapping,recipe)
    else:m._scene_metadata(config,row,mapping,recipe)


def current_population(tmp_path,monkeypatch,**population_options):
    """Producer-shaped JSON only; use its pure metadata constructors, never USD."""
    from sim.ausc_policy import layered_scene_authoring as layers,scene_store as store,sdk_mdl
    from scripts.experiments.wam_runpod_20261027 import given_formal_collection_lossless_v1 as producer
    m,rows,mapping,mappingref,processing,report=population(tmp_path,monkeypatch,**population_options)
    r=m.runtime;plan=json.loads(Path(processing['plan']['path']).read_text())
    refs={entry['reference']['path']:entry['reference'] for entry in plan['accounting_scope']['extra_input_refs']}
    def put(path,value):
        path=Path(path);path.write_text(json.dumps(value));ref=r.reference(path);refs[str(path)]=ref;return ref
    def full(ref):
        bound=refs[ref.get('file',ref.get('path'))];return dict(file=bound['path'],sha256=bound['sha256'],bytes=bound['bytes'])
    def short(ref):return dict(file=ref.get('file',ref.get('path')),sha256=ref['sha256'])
    def source(path,digest='d'*64):
        ref=dict(path=str(path),bytes=1,sha256=digest);refs[str(path)]=ref;return full(ref)
    receipt=json.loads(Path(report['layered_authoring']['receipt']['file']).read_text());authored=report['layered_authoring']
    selected=[];raw=[];baselines=[];provenance=[];records=[];byid={}
    collector_root=tmp_path/'collector';collector_root.mkdir()
    code_names=dict(native_backend='m2_capture.py',state_adapter='m2_contract.py',target_estimator='m2_given_contact_pose.py',
        action_source='m2_given_contact_controller.py',raw_intent_dispatcher='m2_given_contact_intent.py',interior_handoff='m2_given_contact_handoff.py',
        native_runner='geometry_qualification.py')
    code_sources={role:source(collector_root/'sim/ausc_policy'/name,format(1000+i,'064x')) for i,(role,name) in enumerate(code_names.items())}
    provider=source(collector_root/'sim/ausc_policy/layered_scene_authoring.py')
    manifest=put(collector_root/'SOURCE_MANIFEST.json',[dict(file=str(Path(ref['file']).relative_to(collector_root)),sha256=ref['sha256'],bytes=ref['bytes']) for ref in [*code_sources.values(),provider]])
    collector=dict(root=str(collector_root),commit='c'*40,code_commit_file=dict(path=str(collector_root/'CODE_COMMIT'),sha256='8'*64,bytes=1),source_manifest=manifest)
    refs[collector['code_commit_file']['path']]=collector['code_commit_file']
    report['cpu_execution'].update(collector_root=str(collector_root),collector=collector,
        provider=dict(path=str(collector_root/'sim/ausc_policy/layered_scene_authoring.py'),sha256='d'*64,bytes=1))
    refs[report['cpu_execution']['provider']['path']]=report['cpu_execution']['provider'];plan['collector']=collector
    native=short(source('/fixed/native-qualified.json',format(9001,'064x')))
    def capture_config(config,*,current):
        meta=config['m2_given_contact_control']['metadata'];meta['native_retainer_audit']=dict(schema='native_retainer_witness_audit_v1',qualification=native)
        config.update(experiment='geometry_qualification',isaac_python='/native/bin/python')
        compact=dict(metadata=deepcopy(meta),native_image_size=[256,144],max_frames=5000,limit_bytes=100000000,target_confidence=1.,
            code_bindings={role:ref['sha256'] for role,ref in code_sources.items() if role!='native_runner'},
            estimated_target_base=[0.,0.,0.],target_estimate_source_id='given-contact-'+meta['reference_sha256'])
        compact['code_bindings']['body']='a'*64
        if current:compact['numeric_container_codec']=producer.NUMERIC_CODEC
        config['fresh_expert'].update(collision_profile='full',native_retainer_audit=deepcopy(meta['native_retainer_audit']),raw_multimodal_capture=dict(enabled=True,profile='compact',episode_id=meta['episode_id'],
            camera_file='/fixed/camera.json',camera_sha256='d'*64,calibration_file='/fixed/calibration.json',calibration_sha256='d'*64,compact=compact))
    for index,(lineage,result,item) in enumerate(zip(report['scenes'],authored['prepared'],receipt['scenes'])):
        scene=lineage['original'];oldroot=Path(scene['prepared_complete']['file']).parent;root=Path(result['prepared_complete']['file']).parent
        result['dependency_inventory']['primary']['sha256']='2'*64
        refs[result['dependency_inventory']['primary']['file']]['sha256']='2'*64
        item['dependency_inventory']=deepcopy(result['dependency_inventory'])
        prep=json.loads(Path(scene['preparation_config']['file']).read_text());prep.update(base='/raw',time_limit_seconds=480)
        prep['geometry_source']='/fixed/geometry.usdc';prep['geometry_source_sha256']='d'*64;source(prep['geometry_source'])
        spec=prep['patient_asset'];spec['identity_split']=str(r.shared.contract.IDENTITY_SPLIT_PATH)
        splitref=r.reference(spec['identity_split']);refs[splitref['path']]=splitref
        for name,key in (('asset_manifest.json','manifest_sha256'),('summary.json','summary_sha256')):
            p=oldroot/name;ref=put(p,dict(identity=scene['subject_id'],split=scene['split']));spec[key]=ref['sha256']
        source(oldroot/'body.npz','a'*64)
        scene['preparation_config']=short(put(scene['preparation_config']['file'],prep))
        raw_selection={k:short(scene[k]) for k in ('preparation_config','prepared_complete','patient_receipt')};raw.append(raw_selection)
        first=next(e for e in report['plans'] if e['proposal_id']==scene['proposal_id'])
        baseline=json.loads(Path(first['baseline_config']['path']).read_text());normref=baseline['compact_scene_storage']['sources']['preparation_config']
        normalized=dict(prep,base=baseline['base'],time_limit_seconds=2400);normref=put(normref['file'],normalized)
        selection=dict(raw_selection,preparation_config=short(normref));selected.append(selection)
        result['original']=selection;result['preparation_config']=short(put(result['preparation_config']['file'],dict(normalized,physics=deepcopy(layers.ITERATIONS))))
        proof=result['iteration_proof'];proof['rigid_articulation_roles']=[['/World/R'+str(i),'physxRigidBody'] for i in range(24)]
        proof['only_iteration_changes']=[dict(path=p,attribute=role+':solverPositionIterationCount',before=32,after=64) for p,role in proof['rigid_articulation_roles']]
        proof['float32_armature_attributes']=['/World/R0.physxJoint:armature'];item['authored_semantics_sha256']='e'*64
        receipt['iteration_proofs'][index]=deepcopy(proof)
        authored['runtime_equivalence'][index]['selected_iteration_proof']=deepcopy(proof)
        for role,name in store.FILES.items():
            if role in ('scene','patient_receipt','collision_geometry','collision_arrays'):continue
            original=put(oldroot/name,dict(role=role,value=1));put(root/name,dict(role=role,value=1))
        inputs=[full(scene[k]) for k in ('preparation_config','prepared_complete','patient_receipt')]+[full(normref)]
        inputs.extend(full(refs[str(oldroot/name)]) for name in store.FILES.values())
        inputs.extend(full(refs[str(oldroot/name)]) for name in ('asset_manifest.json','summary.json','body.npz'))
        inputs.extend((full(splitref),full(refs[prep['geometry_source']])))
        inputs.extend(full(a['fresh_plan']) for a in mapping['ordered_allocations'][scene['split']] if a['proposal_id']==scene['proposal_id'])
        result['input_bindings']=[v for _,v in sorted({v['file']:v for v in inputs}.items())]
        receipt['input_bindings'][index]=result['input_bindings'];receipt['original_selected'][index]=selection
        lineage['derived']['raw_preparation_config']=refs[scene['preparation_config']['file']]
        byid[scene['proposal_id']]=(lineage,result,index)
    def builder_receipt(scene,entry,config,runtimeprep,*,current=False):
        # Closed receipt emitted by _prepared/build_given_contact_expert and the
        # public producer's CPU wrapper; unrelated numerical content stays opaque.
        fields=('preparation_config prepared_complete patient_receipt subject_id split identity_split_sha256 body effective_scene geometry_artifacts '
            'payload_identity_preflight test_payloads_opened test_metadata_path_nonaccess_claim physics_executed physical_validation physics_qualified '
            'training_eligible data_accepted learned_policy schema fresh_plan observation_contract source_bindings camera hall_calibration '
            'metadata_is_acquisition_intent_not_execution_evidence cli_module collision_profile compact_max_frames total_byte_cap compact_scene_storage '
            'task_id action_semantics_id given_reference reference_sha256 handoff_policy_id teacher_boundary translation_noise physical_qualification '
            'production_act_ready collector_cpu_builder runtime_preparation').split()
        value={key:None for key in fields}
        selected=scene['derived']['derivation'] if current else scene['original']
        value.update({key:short(selected[key]) for key in ('preparation_config','prepared_complete','patient_receipt')})
        meta=config['m2_given_contact_control']['metadata'];body=source(Path(scene['original']['prepared_complete']['file']).parent/'body.npz','a'*64)
        value.update(subject_id=scene['original']['subject_id'],split=scene['original']['split'],identity_split_sha256=r.shared.contract.IDENTITY_SPLIT_SHA256,
            schema='m2_given_contact_compact_expert_config_v2',task_id=r.pose.TASK_ID,body=short(body),effective_scene=short(config['compact_scene_storage']['sources']['scene']),
            fresh_plan=entry['derived' if current else 'original'],given_reference=deepcopy(meta['given_reference']),reference_sha256=meta['reference_sha256'],
            compact_scene_storage=deepcopy(config['compact_scene_storage']),runtime_preparation=deepcopy(runtimeprep),payload_identity_preflight=True,
            compact_max_frames=5000,total_byte_cap=100000000,
            source_bindings={key:short(body) if key=='body' else short(code_sources[key]) for key in
                ('native_backend','state_adapter','target_estimator','action_source','body','raw_intent_dispatcher','interior_handoff')},
            camera=short(source('/fixed/camera.json')),hall_calibration=short(source('/fixed/calibration.json')),
            collector_cpu_builder=dict(collector_root=report['cpu_execution']['collector_root'],executable=deepcopy(report['cpu_execution']['executable']),
                script_sha256='c'*64,gpu_work=False,physics_advanced=False))
        for key in ('physics_executed','physical_validation','physics_qualified','training_eligible','data_accepted','learned_policy',
            'test_payloads_opened','test_metadata_path_nonaccess_claim','physical_qualification','production_act_ready'):value[key]=False
        value['native_retainer_audit']=deepcopy(meta['native_retainer_audit'])
        if current:value['current_raw_prepared_complete']=scene['derived']['prepared_complete']
        return value
    # The actual public producer's authoring request authenticates exact baseline
    # provenance. Baseline configs below retain those exact named preparations.
    for entry in report['plans']:
        lineage,result,index=byid[entry['proposal_id']];scene=lineage['original'];config=json.loads(Path(entry['baseline_config']['path']).read_text())
        prepref=selected[index]['preparation_config'];config['compact_scene_storage']['sources']['preparation_config']=full(prepref)
        for role,name in store.FILES.items():config['compact_scene_storage']['sources'][role]=full(refs[str(Path(scene['prepared_complete']['file']).parent/name)])
        for j in range(sdk_mdl.APPROVED_FILE_COUNT):config['compact_scene_storage']['sources']['sdk_mdl_'+str(j).zfill(4)]=source('/fixed/sdk/'+str(j)+'.mdl',format(j+100,'064x'))
        capture_config(config,current=False)
        entry['baseline_config']=put(entry['baseline_config']['path'],config)
        prov=builder_receipt(lineage,entry,config,refs[prepref['file']])
        entry['baseline_provenance']=put(Path(entry['baseline_config']['path']).with_suffix('.provenance.json'),prov)
        if index==len(baselines):baselines.append(entry['baseline_config']);provenance.append(entry['baseline_provenance'])
    request=producer._authoring_request(raw,baselines,provenance,tmp_path);report['authoring_request']=request
    authored['receipt']=full(put(authored['receipt']['file'],receipt))
    for lineage,result,item,index in ((l,p,s,i) for i,(l,p,s) in enumerate(zip(report['scenes'],authored['prepared'],receipt['scenes']))):
        root=Path(result['prepared_complete']['file']).parent;result['scene_authoring_receipt']=authored['receipt']
        profile=dict(result['iteration_proof'],original_scene=item['original'],inherited_profile=full(refs[str(Path(lineage['original']['prepared_complete']['file']).parent/'physics_profile.json')]),
            new_scene=result['dependency_inventory']['primary'],representation_lineage=authored['receipt'],saved_USD_readback_exact=True)
        result['physics_profile']=short(put(root/'physics_profile.json',profile))
        complete=json.loads(Path(lineage['original']['prepared_complete']['file']).read_text());complete.update(effective_scene_sha256=result['dependency_inventory']['primary']['sha256'],
            scene_dependency_inventory=result['dependency_inventory'],scene_authoring_receipt=authored['receipt'],solver_isolation_lineage=result['physics_profile'],
            preparation_backend='cpu_lossless_static_layer_and_exact_existing_iteration_authoring')
        result['prepared_complete']=short(put(root/'COMPLETE.json',complete))
        historical=full(result['prepared_complete']);sources={role:full(refs[str(root/name)]) for role,name in store.FILES.items()}
        sources['scene']=deepcopy(result['dependency_inventory']['primary']);sources['preparation_config']=full(result['preparation_config'])
        sidecar=layers.current_raw_sidecar_record(historical,authored['receipt'],sources,receipt,item,index)
        directory=tmp_path/'current_raw'/str(index).zfill(3);directory.mkdir(parents=True)
        sidecarref=full(put(directory/'layered_runtime_equivalence.json',sidecar))
        current=short(put(directory/'COMPLETE.json',dict(complete,scene_dependency_policy=store.SDK_LAYER_CURRENT_RAW_POLICY,layered_runtime_equivalence=sidecarref)))
        record=dict(historical_prepared_complete=historical,layered_runtime_equivalence=sidecarref,prepared_complete=current);records.append(record)
        derived=lineage['derived'];derived.update({k:deepcopy(result[k]) for k in ('preparation_config','patient_receipt')})
        derived.update(prepared_complete=current,current_raw=record,derivation=result)
    for entry in report['plans']:
        lineage,result,index=byid[entry['proposal_id']];baseline=json.loads(Path(entry['baseline_config']['path']).read_text())
        config=json.loads(Path(entry['runtime']['path']).read_text());sources=deepcopy(baseline['compact_scene_storage']['sources'])
        sources.update({role:full(result[role]) for role in ('preparation_config','patient_receipt','physics_profile')})
        root=Path(result['prepared_complete']['file']).parent
        for role,name in store.FILES.items():
            if role not in ('scene','physics_profile','patient_receipt'):sources[role]=full(refs[str(root/name)])
        sources.update(scene=deepcopy(result['dependency_inventory']['primary']),scene_base=deepcopy(result['dependency_inventory']['base']),
            scene_authoring_receipt=authored['receipt'],layered_runtime_equivalence=records[index]['layered_runtime_equivalence'],prepared_complete=full(records[index]['prepared_complete']))
        config['compact_scene_storage']['sources']=sources;config['compact_scene_storage']['dependency_policy']=store.SDK_LAYER_CURRENT_RAW_POLICY
        capture_config(config,current=True)
        entry['runtime']=put(entry['runtime']['path'],config)
        entry['builder_receipt']=put(entry['builder_receipt']['path'],builder_receipt(lineage,entry,config,refs[result['preparation_config']['file']],current=True))
        pairs={key:dict(original=baseline['compact_scene_storage']['sources'].get(key),derived=value) for key,value in sources.items()}
        assert producer._runtime_rebinding(baseline,config,pairs)==entry['reference_binding']
        attempt=next(a for a in plan['attempts'] if a['slot_id']==entry['slot_id']);attempt['config']=entry['runtime']
    report['current_raw_prepared']=records;report['cpu_execution'].update(request_sha256=r.digest(request),current_raw_equivalence=True,dependency_policy=store.SDK_LAYER_CURRENT_RAW_POLICY)
    for proof,baseline,result,record in zip(authored['runtime_equivalence'],baselines,authored['prepared'],records):
        proof['original_sources']=json.loads(Path(baseline['path']).read_text())['compact_scene_storage']['sources']
        proof['derived_sources']=deepcopy(proof['original_sources']);proof['derived_sources'].update({k:full(result[k]) for k in ('preparation_config','patient_receipt','physics_profile')})
        root=Path(result['prepared_complete']['file']).parent
        for role,name in store.FILES.items():
            if role not in ('scene','physics_profile','patient_receipt'):proof['derived_sources'][role]=full(refs[str(root/name)])
        proof['derived_sources'].update(scene=result['dependency_inventory']['primary'],scene_base=authored['base'],scene_authoring_receipt=authored['receipt'],
            prepared_complete=full(record['prepared_complete']),layered_runtime_equivalence=record['layered_runtime_equivalence'])
    report['runtime_equivalence_receipt']=put(tmp_path/'SDK_RUNTIME_EQUIVALENCE.json',dict(schema='m2_given_lossless_sdk_cpu_receipt_v1',authoring_request=request,cpu_execution=report['cpu_execution'],scene_authoring_receipt=authored['receipt'],proofs=authored['runtime_equivalence']))
    recipe=json.loads(Path(plan['source_recipe']['path']).read_text());recipe.update(current_raw_equivalence=True,scene_dependency_policy=store.SDK_LAYER_CURRENT_RAW_POLICY)
    inventories=[];conditions=[];(tmp_path/'episodes').mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(producer,'reference',lambda path:deepcopy(refs[str(path)]))
        for i,(entry,attempt) in enumerate(zip(report['plans'],plan['attempts'])):
            config=json.loads(Path(entry['runtime']['path']).read_text());meta=config['m2_given_contact_control']['metadata']
            roles=meta['given_reference']['sources'];fixed=producer._fixed_sources(meta,config)
            union={}
            for ref in [*code_sources.values(),*roles.values(),*fixed.values()]:
                path=ref['file'];assert path not in union or union[path]==ref['sha256'];union[path]=ref['sha256']
            inventory=producer._recipe_inventory(union,collector_root,roles,fixed)
            inventories.append(inventory);conditions.append(producer._conditions(config));files={}
            for original,digest in union.items():
                original_ref=refs[original];archive=tmp_path/'archive'/digest
                archived=dict(path=str(archive),sha256=digest,bytes=original_ref['bytes']);refs[str(archive)]=archived;files[original]=archived
            attempt['attempt_id']='fixture-'+str(i+1).zfill(3)
            attempt['source_freeze']=put(tmp_path/'episodes'/(attempt['attempt_id']+'.source-freeze.json'),
                dict(schema=r.dataset.SOURCE_FREEZE_SCHEMA,code_commit=collector['commit'],code_commit_file=collector['code_commit_file'],
                    source_manifest=collector['source_manifest'],files=files))
    assert all(inventory==inventories[0] for inventory in inventories) and all(condition==conditions[0] for condition in conditions)
    recipe['collector_conditions']=conditions[0]
    recipe.update(qualification=source('/fixed/qualification.json'),qualification_all_eight=source('/fixed/eight.json'),inventory=inventories[0])
    plan['current_raw_equivalence']=True
    plan['source_recipe']=put(plan['source_recipe']['path'],recipe);plan['derivation']=put(tmp_path/'DERIVATION_MAP.json',report)
    mappingref=put(mappingref['path'],mapping);plan['allocation_mapping']=mappingref
    monkeypatch.setattr(r.dataset,'ALLOCATION_MAPPING_SHA256',mappingref['sha256'])
    plan['accounting_scope']['extra_input_refs']=[dict(category='config',reference=ref) for ref in refs.values()]
    processing['plan']=put(processing['plan']['path'],plan)
    for row in rows:
        scene=next(s for s in report['scenes'] if s['original']['subject_id']==row['subject_id'])
        row['pose_binding']['preparation_config']=deepcopy(scene['derived']['preparation_config'])
    return m,rows,mapping,mappingref,processing,report


def test_current_raw_mapping_accepts_producer_metadata_without_payload(tmp_path,monkeypatch):
    m,rows,original,ref,processing,report=current_population(tmp_path,monkeypatch)
    read_bytes=Path.read_bytes
    def json_only(path):
        assert path.suffix=='.json' and path.name!='collision_geometry.json', 'Payload access in metadata projection'
        return read_bytes(path)
    monkeypatch.setattr(Path,'read_bytes',json_only)
    path_open=Path.open
    def json_open(path,*args,**kwargs):
        assert path.suffix=='.json' and path.name!='collision_geometry.json', 'Payload open in metadata projection'
        return path_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',json_open)
    from sim.ausc_policy import scene_store,layered_scene_authoring
    def forbidden(*args,**kwargs):raise AssertionError('Payload-capable core helper invoked by metadata projection')
    for name in ('read_source','source_ref','_layered_metadata','_validate_binding','prepared_runtime_stage'):
        monkeypatch.setattr(scene_store,name,forbidden)
    for name in ('issue_current_raw_equivalence','author_population','authoring_input_inventory','historical_prepared_sources'):
        monkeypatch.setattr(layered_scene_authoring,name,forbidden)
    mapping,canonical=m._allocation_mapping(ref,processing)
    assert canonical==ref and mapping['ordered_allocations']['train']!=original['ordered_allocations']['train']
    assert mapping['summaries']==original['summaries']
    for row in rows:assert m._reference_roles(row,mapping).sha256==row['metadata']['reference_sha256']
    assert len(report['plans'])==240 and len(report['scenes'])==24
    assert not list(tmp_path.rglob('*.usdc')) and not list(tmp_path.rglob('*.npz'))


@pytest.mark.parametrize('data',[b'{"value":1,"value":2}',b'{"value":NaN}',b'{"value":Infinity}',b'{"value":1e999}'])
def test_current_reader_rejects_duplicate_and_nonfinite_raw_json(tmp_path,data):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    path=tmp_path/'registered.json';path.write_bytes(data);ref=module().runtime.reference(path)
    reader=bridge._CurrentMetadata(dict(accounting_scope=dict(extra_input_refs=[dict(reference=ref)])))
    with pytest.raises(ValueError):reader.read(ref)


def test_current_builder_camera_receipt_must_match_actual_capture(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    _,_,mapping,_,processing,report=current_population(tmp_path,monkeypatch)
    plan=json.loads(Path(processing['plan']['path']).read_text());reader=bridge._CurrentMetadata(plan)
    entry=report['plans'][-1];scene=next(s for s in report['scenes'] if s['proposal_id']==entry['proposal_id'])
    config=reader.read(entry['runtime']);receipt=reader.read(entry['builder_receipt'])
    receipt['camera']=deepcopy(receipt['hall_calibration'])
    with pytest.raises(ValueError,match='capture|camera'):
        bridge._current_builder(reader,receipt,scene,entry,config,plan['collector']['root'],current=True)


def test_checkpoint_headers_route_historical_outputs_and_new_plans_to_distinct_roots(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    _,_,mapping,_,processing,report=current_population(tmp_path,monkeypatch)
    plan=json.loads(Path(processing['plan']['path']).read_text());reader=bridge._CurrentMetadata(plan)
    assembly=tmp_path/'assembly-v4';assembly.mkdir()
    def copy(ref,directory):
        path=directory/Path(ref.get('path',ref.get('file'))).name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(Path(ref.get('path',ref.get('file'))).read_bytes());bound=bridge.runtime.reference(path)
        reader.inputs[str(path)]=bound;return bound
    plan['derivation']=dict(plan['derivation'],path=str(assembly/'DERIVATION_MAP.json'))
    for entry in report['plans']:entry['derived']=bridge._file(copy(entry['derived'],assembly/'plans'))
    for attempt in plan['attempts']:attempt['source_freeze']=copy(attempt['source_freeze'],assembly/'episodes')
    selected,allocated,_=bridge._population(mapping);recipe=reader.read(plan['source_recipe'])
    with pytest.raises(ValueError):bridge._current_headers(report,recipe,selected,allocated,plan,reader,assembly)
    report['completed_authoring_checkpoint']=dict(path='/registered/checkpoint.json',bytes=1,sha256='a'*64)
    historical,normalized=bridge._current_headers(report,recipe,selected,allocated,plan,reader,assembly,checkpoint=dict(source_root=str(tmp_path)))
    bridge._current_authored(report,historical,normalized,reader,tmp_path)
    assert report['authoring_request']['output_root']==str(tmp_path/'derived')


def checkpoint_population(tmp_path,monkeypatch):
    """Real metadata consumer fixture: 48 scenes, 240 saved and copied plans."""
    import shutil
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    from scripts.experiments.wam_runpod_20261027 import completed_authoring_checkpoint_v1 as recovery
    historical=tmp_path/'historical';historical.mkdir()
    _,_,mapping,_,processing,report=current_population(historical,monkeypatch,
        scene_counts=(40,8),targets=['L'+str(i) for i in range(1,6)],ordered_layers=True)
    plan=json.loads(Path(processing['plan']['path']).read_text());assembly=tmp_path/'assembly';assembly.mkdir()
    refs={row['reference']['path']:row['reference'] for row in plan['accounting_scope']['extra_input_refs']}
    def put(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        ref=recovery.reference(path);refs[str(path)]=ref;return ref
    def replace(value,strings):
        if isinstance(value,dict):return {k:replace(v,strings) for k,v in value.items()}
        if isinstance(value,list):return [replace(v,strings) for v in value]
        return strings.get(value,value) if isinstance(value,str) else value
    core=tmp_path/'core';qualified=tmp_path/'qualified';captures={};roles={}
    for name in ('camera','calibration'):
        default=put(core/(name+'.json'),{});approved=put(qualified/(name+'.json'),{})
        captures[name]=dict(default=default,qualified=approved);roles[name]=(name+'.json',default['bytes'],default['sha256'])
    native51='/opt/wam-runtime/envs/isaacsim51/bin/python';native60='/opt/wam-runtime/envs/isaacsim60/bin/python'
    owner=dict(report['cpu_execution']['executable'],invoked_path=native51)
    cpu=dict(owner,invoked_path=native60,path=str(tmp_path/'native60'))
    refs[cpu['path']]={k:v for k,v in cpu.items() if k!='invoked_path'};report['cpu_execution']['executable']=cpu
    for entry,attempt in zip(report['plans'],plan['attempts']):
        saved=Path(entry['derived']['file']);copied=assembly/'plans'/saved.name;copied.parent.mkdir(parents=True,exist_ok=True)
        copied.write_bytes(saved.read_bytes());copied_ref=recovery.reference(copied);refs[str(copied)]=copied_ref
        entry['derived']=bridge._file(copied_ref)
        baseline=json.loads(Path(entry['baseline_config']['path']).read_text());config=json.loads(Path(entry['runtime']['path']).read_text())
        prior=json.loads(Path(entry['baseline_provenance']['path']).read_text());receipt=json.loads(Path(entry['builder_receipt']['path']).read_text())
        for value,builder,current in ((baseline,prior,False),(config,receipt,True)):
            value['isaac_python']=native51;builder['collector_cpu_builder']['executable']=cpu if current else owner
            for name,role in (('camera','camera'),('calibration','hall_calibration')):
                ref=captures[name]['qualified' if current else 'default']
                value['fresh_expert']['raw_multimodal_capture'].update({name+'_file':ref['path'],name+'_sha256':ref['sha256']})
                builder[role]=bridge._file(ref)
        before=config['m2_given_contact_control']['metadata']['reference_sha256'];paths={str(saved):str(copied)}
        config=replace(config,paths);reference=bridge.runtime.pose.GivenReference.from_snapshot(config['m2_given_contact_control']['metadata']['given_reference'])
        substitutions=dict(paths);substitutions.update({before:reference.sha256,'given-contact-'+before:'given-contact-'+reference.sha256})
        config=replace(config,substitutions);receipt=replace(receipt,substitutions)
        entry['reference_binding'].update(derived_reference_sha256=reference.sha256,derived_sources=reference.snapshot()['sources'])
        entry['baseline_config']=put(entry['baseline_config']['path'],baseline);entry['baseline_provenance']=put(entry['baseline_provenance']['path'],prior)
        entry['runtime']=put(assembly/'episodes'/Path(entry['runtime']['path']).name,config)
        entry['builder_receipt']=put(assembly/'episodes'/Path(entry['builder_receipt']['path']).name,receipt);attempt['config']=entry['runtime']
        freeze=json.loads(Path(attempt['source_freeze']['path']).read_text())
        changed_paths={str(saved):str(copied),'/fixed/camera.json':captures['camera']['qualified']['path'],
            '/fixed/calibration.json':captures['calibration']['qualified']['path']}
        files={}
        for name,ref in freeze['files'].items():
            if name in ('/fixed/camera.json','/fixed/calibration.json'):
                ref=captures['camera' if name.endswith('/camera.json') else 'calibration']['qualified']
            archived=dict(path=str(assembly/'archive'/ref['sha256']),bytes=ref['bytes'],sha256=ref['sha256'])
            refs[archived['path']]=archived;files[changed_paths.get(name,name)]=archived
        freeze['files']=files;attempt['source_freeze']=put(assembly/'episodes'/Path(attempt['source_freeze']['path']).name,freeze)
    first=[next(e for e in report['plans'] if e['proposal_id']==s['proposal_id']) for s in report['scenes']]
    report['authoring_request'].update(baseline_configs=[e['baseline_config'] for e in first],baseline_provenance=[e['baseline_provenance'] for e in first])
    report['cpu_execution']['request_sha256']=bridge.runtime.digest(report['authoring_request'])
    report['runtime_equivalence_receipt']=put(historical/'SDK_RUNTIME_EQUIVALENCE.json',dict(schema='m2_given_lossless_sdk_cpu_receipt_v1',
        authoring_request=report['authoring_request'],cpu_execution=report['cpu_execution'],scene_authoring_receipt=report['layered_authoring']['receipt'],proofs=report['layered_authoring']['runtime_equivalence']))
    recipe=json.loads(Path(plan['source_recipe']['path']).read_text());recipe['collector_conditions']['isaac_python']=native51
    for name in captures:
        ref=captures[name]['qualified'];recipe['collector_conditions']['capture'].update({name+'_file':ref['path'],name+'_sha256':ref['sha256']})
        recipe['inventory']['fixed:'+name]['reference']=ref
    plan.update(source_recipe=put(plan['source_recipe']['path'],recipe),owner_executable=owner,config_executable=cpu)
    for name in ('DERIVATION_MAP.json','plan.json'):(historical/name).unlink()
    shutil.rmtree(historical/'episodes');support=tmp_path/'support';e4=tmp_path/'e4'
    bindings={key:put(support/(key+'.json'),{}) for key in recovery.PINNED}
    bindings.update(original_settings=put(support/'original.json',dict(base=str(tmp_path),run_ids=dict(first100='old-v3'),
        metadata_root=str(tmp_path/'old-meta'),seal_path=str(tmp_path/'old-seal'))),original_recipe=plan['source_recipe'],
        sdk_receipt=report['runtime_equivalence_receipt'],scene_receipt=recovery.reference(report['layered_authoring']['receipt']['file']),
        historical_source_manifest=put(e4/'SOURCE_MANIFEST.json',[]))
    constants=dict(SOURCE_ROOT=str(historical),ASSEMBLY_ROOT=str(assembly),E4_ROOT=str(e4),CORE_ROOT=str(core),QUALIFIED_ROOT=str(qualified),
        CAPTURE_ROLES=roles,PINNED={k:(v['bytes'],v['sha256']) for k,v in bindings.items()},PINNED_PATHS={k:v['path'] for k,v in bindings.items()})
    for name,value in constants.items():monkeypatch.setattr(recovery,name,value)
    roots=[dict(path=str(p),category='prepared',kind='directory') for p in (historical,support,e4,core,qualified)]
    monkeypatch.setattr(recovery,'REQUIRED_RETAINED_ROOTS',deepcopy(roots))
    retained=sorted([dict(category=row['category'],reference=recovery.reference(p)) for row in roots for p in Path(row['path']).rglob('*') if p.is_file()],key=lambda row:row['reference']['path'])
    negative=put(tmp_path/'negative.json',dict(schema=recovery.NEGATIVE_SCHEMA,retained_roots=roots,retained_inputs=retained,retained_file_count=len(retained),
        total_retained_bytes=sum(row['reference']['bytes'] for row in retained),physical_denominator=0,model_denominator=0,physical_launched_count=0,
        valid_plan_issued=False,remote_operator_exit_status=1,test_payloads_opened=False,training_eligible=False));bindings['failed_negative']=negative
    selected,allocated,_=bridge._population(mapping);scenes=[];plans=[]
    for scene,row in zip(selected,report['scenes']):
        result=row['derived']['derivation'];current=row['derived']['current_raw']
        scenes.append(dict({k:scene[k] for k in ('proposal_id','subject_id','split')},original={k:bridge._file(scene[k]) for k in recovery.SCENE_ROLES},
            files={k:recovery.reference(result[k]['file']) for k in ('preparation_config','prepared_complete','patient_receipt','physics_profile')},
            current_raw={k:recovery.reference(current[k]['file']) for k in ('prepared_complete','layered_runtime_equivalence')}))
    for a,entry in zip(allocated,report['plans']):
        plans.append(dict({k:a[k] for k in ('slot_id','proposal_id','subject_id','split','target')},original=bridge._file(entry['original']),
            derived=recovery.reference(historical/'plans'/Path(entry['derived']['file']).name),baseline_config=entry['baseline_config'],baseline_provenance=entry['baseline_provenance']))
    value=dict(schema=recovery.SCHEMA,source_root=str(historical),assembly_root=str(assembly),bindings=bindings,scenes=scenes,plans=plans,
        retained_roots=[*roots,dict(path=negative['path'],category='prepared',kind='file')],
        retained_inputs=sorted([*retained,dict(category='prepared',reference=negative)],key=lambda row:row['reference']['path']),capture_paths=captures,
        physical_launched_count=0,physical_denominator=0,model_denominator=0,test_payloads_opened=False,training_eligible=False)
    certificate=put(tmp_path/'checkpoint.json',value);report['completed_authoring_checkpoint']=certificate;plan['completed_authoring_checkpoint']=certificate
    plan['derivation']=put(assembly/'DERIVATION_MAP.json',report)
    for row in value['retained_inputs']:refs[row['reference']['path']]=row['reference']
    plan['accounting_scope']['extra_input_refs']=[dict(category='prepared',reference=ref) for ref in refs.values()]
    return mapping,plan,value


def test_checkpoint_full_consumer_rejects_last_baseline_alias_and_saved_plan_damage(tmp_path,monkeypatch):
    import re
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    from scripts.experiments.wam_runpod_20261027 import completed_authoring_checkpoint_v1 as recovery
    mapping,plan,checkpoint=checkpoint_population(tmp_path,monkeypatch);original_read=Path.read_bytes
    def json_only(path):
        assert path.suffix=='.json' and path.name!='collision_geometry.json', 'Metadata consumer opened an opaque payload'
        return original_read(path)
    monkeypatch.setattr(Path,'read_bytes',json_only)
    assert len(bridge.authenticated_view(mapping,plan)['ordered_allocations']['validation'])==40
    report=json.loads(Path(plan['derivation']['path']).read_text())
    copied=Path(report['plans'][-1]['derived']['file']);before=copied.read_bytes()
    damaged=re.sub(rb'("numeric_plan":\s*\[\s*)[0-9]',lambda match:match[1]+b'9',before,count=1)
    assert damaged!=before and len(damaged)==len(before)
    copied.write_bytes(damaged);rejected=[]
    try:bridge.authenticated_view(mapping,plan)
    except ValueError:rejected.append('copied_plan')
    copied.write_bytes(before)
    saved=Path(checkpoint['plans'][-1]['derived']['path']);before=saved.read_bytes()
    damaged=re.sub(rb'("numeric_plan":\s*\[\s*)[0-9]',lambda match:match[1]+b'9',before,count=1)
    assert damaged!=before and len(damaged)==len(before)
    saved.write_bytes(damaged)
    try:bridge.authenticated_view(mapping,plan)
    except ValueError:rejected.append('saved_plan')
    saved.write_bytes(before)
    report=json.loads(Path(plan['derivation']['path']).read_text());entry=report['plans'][-1]
    baseline=json.loads(Path(entry['baseline_config']['path']).read_text());config=json.loads(Path(entry['runtime']['path']).read_text())
    baseline['seed']=config['seed']=987654321
    alias=tmp_path/'assembly/foreign-baseline.json';alias.write_text(json.dumps(baseline))
    provenance=alias.with_suffix('.provenance.json');provenance.write_bytes(Path(entry['baseline_provenance']['path']).read_bytes())
    entry['baseline_config']=recovery.reference(alias);entry['baseline_provenance']=recovery.reference(provenance)
    runtime=Path(entry['runtime']['path']);runtime.write_text(json.dumps(config));entry['runtime']=recovery.reference(runtime);plan['attempts'][-1]['config']=entry['runtime']
    report_path=Path(plan['derivation']['path']);report_path.write_text(json.dumps(report));plan['derivation']=recovery.reference(report_path)
    inputs={row['reference']['path']:row for row in plan['accounting_scope']['extra_input_refs']}
    for ref in (entry['baseline_config'],entry['baseline_provenance'],entry['runtime'],plan['derivation']):inputs[ref['path']]=dict(category='config',reference=ref)
    plan['accounting_scope']['extra_input_refs']=list(inputs.values())
    try:bridge.authenticated_view(mapping,plan)
    except ValueError:rejected.append('baseline_alias')
    assert rejected==['copied_plan','saved_plan','baseline_alias']


@pytest.mark.parametrize('value',[True,64.0])
def test_current_proof_rejects_typed_profile_aliases(tmp_path,monkeypatch,value):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    _,_,_,_,_,report=population(tmp_path,monkeypatch)
    proof=deepcopy(report['layered_authoring']['prepared'][0]['iteration_proof'])
    proof['profile']['position_iterations']=value
    with pytest.raises(ValueError):bridge._current_proof(proof)


def test_current_headers_refuse_late_identity_alias_and_unknown_policy_before_leaf_json(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    m,_,mapping,_,processing,original=current_population(tmp_path,monkeypatch)
    plan=json.loads(Path(processing['plan']['path']).read_text());reader=bridge._CurrentMetadata(plan)
    recipe=reader.read(plan['source_recipe']);selected,allocated,_=bridge._population(mapping)
    root=Path(plan['derivation']['path']).parent
    faults={
        'late_identity':lambda report,recipe:report['scenes'][-1]['derived'].update(subject_id='p000001'),
        'late_alias':lambda report,recipe:report['scenes'][-1]['derived'].update(patient_receipt=report['scenes'][0]['derived']['patient_receipt']),
        'late_plan':lambda report,recipe:report['plans'][-1].update(slot_id=report['plans'][0]['slot_id']),
        'unknown_report':lambda report,recipe:report.update(unknown=False),
        'unknown_recipe':lambda report,recipe:recipe.update(unknown=False),
        'unknown_request':lambda report,recipe:report['authoring_request'].update(unknown=False),
        'unknown_policy':lambda report,recipe:recipe['collector_conditions']['scene_dependencies'].update(dependency_policy='unknown'),
        'mixed_policy':lambda report,recipe:recipe.update(scene_dependency_policy=bridge.scene_store.SDK_LAYER_DEPENDENCY_POLICY),
        'missing_opt_in':lambda report,recipe:report['authoring_request'].pop('current_raw_equivalence'),
        'count_alias':lambda report,recipe:report.update(unique_original_scenes=float(report['unique_original_scenes'])),
        'baseline_clone':lambda report,recipe:report['plans'][-1]['baseline_provenance'].update(path=report['plans'][0]['baseline_provenance']['path']),
    }
    def no_leaf(ref):raise AssertionError('Invalid complete headers reached a leaf JSON')
    monkeypatch.setattr(reader,'read',no_leaf)
    for fault,change in faults.items():
        report=deepcopy(original);current_recipe=deepcopy(recipe);change(report,current_recipe)
        with pytest.raises(ValueError):
            bridge._current_headers(report,current_recipe,selected,allocated,plan,reader,root)


def test_current_projection_rejects_self_bound_sidecar_source_normalization_and_control_faults(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    m,_,mapping,_,processing,base=current_population(tmp_path,monkeypatch)
    frozen=json.loads(Path(processing['plan']['path']).read_text())
    def rewrite_refs(value,path,ref):
        if isinstance(value,dict):
            if value.get('file',value.get('path'))==str(path) and 'sha256' in value:
                value['sha256']=ref['sha256']
                if 'bytes' in value:value['bytes']=ref['bytes']
            for item in value.values():rewrite_refs(item,path,ref)
        elif isinstance(value,list):
            for item in value:rewrite_refs(item,path,ref)
    for fault in ('sidecar_index','sidecar_unknown','sidecar_scope','source_missing','raw_inventory_missing','baseline_normalization','builder_unknown','builder_count_alias','duplicate_report','runtime_control'):
        report=deepcopy(base);plan=deepcopy(frozen);changed=[]
        def edit(ref,change):
            path=Path(ref.get('file',ref.get('path')));data=path.read_bytes();value=json.loads(data);change(value)
            path.write_text(json.dumps(value));new=m.runtime.reference(path);changed.append((path,data))
            rewrite_refs(report,path,new);rewrite_refs(plan,path,new)
        try:
            if fault.startswith('sidecar_'):
                ref=report['current_raw_prepared'][-1]['layered_runtime_equivalence']
                edit(ref,lambda value:value.update(**({'scene_index':23.0} if fault=='sidecar_index' else
                    {'unknown':False} if fault=='sidecar_unknown' else {'historical_digest_scope':'executed_oracle'})))
            elif fault=='source_missing':
                path=report['current_raw_prepared'][-1]['layered_runtime_equivalence']['file']
                plan['accounting_scope']['extra_input_refs']=[e for e in plan['accounting_scope']['extra_input_refs'] if e['reference']['path']!=path]
            elif fault=='raw_inventory_missing':report['layered_authoring']['prepared'][-1]['input_bindings'].pop()
            elif fault=='baseline_normalization':
                ref=report['plans'][-1]['baseline_provenance'];edit(ref,lambda value:value.update(runtime_preparation=report['scenes'][-1]['derived']['raw_preparation_config']))
            elif fault=='builder_unknown':edit(report['plans'][-1]['builder_receipt'],lambda value:value.update(unknown=False))
            elif fault=='builder_count_alias':edit(report['plans'][-1]['builder_receipt'],lambda value:value.update(compact_max_frames=5000.0))
            elif fault=='runtime_control':edit(report['plans'][-1]['runtime'],lambda value:value.update(time_limit_seconds=2401))
            reportpath=Path(plan['derivation']['path']);data=reportpath.read_bytes();reportpath.write_text(json.dumps(report) if fault!='duplicate_report' else json.dumps(report)[:-1]+',"training_eligible":false}');changed.append((reportpath,data))
            new=m.runtime.reference(reportpath);rewrite_refs(plan,reportpath,new)
            with pytest.raises(ValueError):bridge.authenticated_view(mapping,plan)
        finally:
            for path,data in reversed(changed):path.write_bytes(data)


def rebind_current_json(plan,ref,value):
    """Rehash a real on-disk JSON role and update its frozen plan capability."""
    path=Path(ref.get('file',ref.get('path')));path.write_text(json.dumps(value));new=module().runtime.reference(path)
    def update(item):
        if isinstance(item,dict):
            if item.get('file',item.get('path'))==str(path) and 'sha256' in item:
                item['sha256']=new['sha256']
                if 'bytes' in item:item['bytes']=new['bytes']
            for child in item.values():update(child)
        elif isinstance(item,list):
            for child in item:update(child)
    update(plan)


@pytest.mark.parametrize('fault',['alias','unknown','unregistered'])
def test_current_prepared_headers_reject_json_roundtrip_fault_before_any_leaf(tmp_path,monkeypatch,fault):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    _,_,mapping,_,processing,_=current_population(tmp_path,monkeypatch)
    plan=json.loads(Path(processing['plan']['path']).read_text())
    report=json.loads(Path(plan['derivation']['path']).read_text());prepared=report['layered_authoring']['prepared']
    # JSON reload detaches scene.derivation from layered_authoring.prepared.
    assert prepared[-1] is not report['scenes'][-1]['derived']['derivation']
    if fault=='alias':prepared[-1]['patient_receipt']=deepcopy(prepared[0]['patient_receipt'])
    elif fault=='unknown':prepared[-1]['unknown']=False
    else:prepared[-1]['patient_receipt']=dict(file='/unregistered/last-patient.json',sha256='a'*64)
    rebind_current_json(plan,plan['derivation'],report)
    leaves=[];read=bridge._CurrentMetadata.read;top={plan['derivation']['path'],plan['source_recipe']['path']}
    def count_leaf(reader,ref):
        path=bridge._reference(ref)['path']
        if path not in top:leaves.append(path)
        return read(reader,ref)
    monkeypatch.setattr(bridge._CurrentMetadata,'read',count_leaf)
    with pytest.raises(ValueError):bridge.authenticated_view(mapping,plan)
    assert leaves==[], 'Rejected complete prepared header only after leaf reads: '+str(len(leaves))


def test_current_recipe_rejects_rehashed_unapproved_collector_conditions(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    _,_,mapping,_,processing,_=current_population(tmp_path,monkeypatch)
    plan=json.loads(Path(processing['plan']['path']).read_text());recipe=json.loads(Path(plan['source_recipe']['path']).read_text())
    recipe['collector_conditions']['fresh']=dict(collision_profile='none')
    recipe['collector_conditions']['capture']=dict(enabled=False,profile='unapproved',compact=dict(numeric_container_codec='lossy',limit_bytes=999999999))
    rebind_current_json(plan,plan['source_recipe'],recipe)
    with pytest.raises(ValueError):bridge.authenticated_view(mapping,plan)


def test_current_recipe_rejects_rehashed_foreign_code_and_given_inventory(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    _,_,mapping,_,processing,_=current_population(tmp_path,monkeypatch)
    plan=json.loads(Path(processing['plan']['path']).read_text());recipe=json.loads(Path(plan['source_recipe']['path']).read_text())
    recipe['inventory']['code:../../foreign.py']=dict(kind='code',relative_path='../../foreign.py',sha256='not-a-sha')
    recipe['inventory']['given:foreign']=dict(kind='given',roles=['not-a-given-role'])
    rebind_current_json(plan,plan['source_recipe'],recipe)
    with pytest.raises(ValueError):bridge.authenticated_view(mapping,plan)


def test_current_condition_record_matches_public_producer_and_preserves_every_field(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    from scripts.experiments.wam_runpod_20261027 import given_formal_collection_lossless_v1 as producer
    _,_,_,_,processing,report=current_population(tmp_path,monkeypatch)
    config=json.loads(Path(report['plans'][0]['runtime']['path']).read_text())
    expected=producer._conditions(config)
    assert bridge._same(bridge._current_conditions(config),expected)
    for field,value in (('isaac_python','/foreign/python'),('collector_extra',False)):
        changed=json.loads(json.dumps(config));changed[field]=value
        assert not bridge._same(bridge._current_conditions(changed),expected)
    for change in (
        lambda value:value['fresh_expert'].update(collision_profile='none'),
        lambda value:value['fresh_expert']['raw_multimodal_capture'].update(enabled=False),
        lambda value:value['fresh_expert']['raw_multimodal_capture'].update(profile='unapproved'),
        lambda value:value['fresh_expert']['raw_multimodal_capture']['compact'].update(numeric_container_codec='lossy'),
        lambda value:value['fresh_expert']['raw_multimodal_capture']['compact'].update(limit_bytes=100000000.0),
        lambda value:value['fresh_expert']['raw_multimodal_capture']['compact'].update(native_image_size=[256.,144]),
    ):
        changed=json.loads(json.dumps(config));change(changed)
        with pytest.raises(ValueError):bridge._current_conditions(changed)


def test_current_source_inventory_authenticates_headers_aliases_and_full_given_coverage(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    from scripts.experiments.wam_runpod_20261027 import given_formal_collection_lossless_v1 as producer
    _,_,_,_,processing,report=current_population(tmp_path,monkeypatch)
    base=json.loads(Path(processing['plan']['path']).read_text());entry=report['plans'][0]
    config=json.loads(Path(entry['runtime']['path']).read_text());root=Path(base['derivation']['path']).parent
    freeze_ref=base['attempts'][0]['source_freeze'];path=Path(freeze_ref['path']);raw=path.read_bytes();original=json.loads(raw)
    roles=config['m2_given_contact_control']['metadata']['given_reference']['sources']
    bindings=json.loads(Path(entry['builder_receipt']['path']).read_text())['source_bindings']
    recipe=json.loads(Path(base['source_recipe']['path']).read_text())
    reader=bridge._CurrentMetadata(base)
    assert bridge._same(bridge._current_inventory(reader,base['attempts'][0],entry,config,base['collector'],root),recipe['inventory'])
    fixed_roles={name for item in recipe['inventory'].values() if item['kind'] in ('fixed','prepared') for name in item['roles']}
    assert fixed_roles==set(producer._fixed_sources(config['m2_given_contact_control']['metadata'],config))
    assert recipe['inventory']['given:patient_receipt,task_manifest']==dict(kind='given',roles=['patient_receipt','task_manifest'])
    assert original['files']['/fixed/camera.json']==original['files']['/fixed/calibration.json']
    # The public producer intentionally deduplicates byte-identical originals,
    # including code files, at the one exact archive/<digest> capability.
    manifest_path=Path(base['collector']['source_manifest']['path']);manifest_raw=manifest_path.read_bytes()
    try:
        plan=json.loads(json.dumps(base));freeze=json.loads(raw);manifest=json.loads(manifest_raw)
        code=bindings['native_backend']['file'];archive=deepcopy(freeze['files'][code])
        relative='sim/ausc_policy/identical_capture.py';duplicate=str(Path(base['collector']['root'])/relative)
        manifest.append(dict(file=relative,bytes=archive['bytes'],sha256=archive['sha256']))
        plan['accounting_scope']['extra_input_refs'].append(dict(category='source',reference=dict(path=duplicate,bytes=archive['bytes'],sha256=archive['sha256'])))
        rebind_current_json(plan,plan['collector']['source_manifest'],manifest)
        freeze['source_manifest']=deepcopy(plan['collector']['source_manifest']);freeze['files'][duplicate]=archive
        rebind_current_json(plan,plan['attempts'][0]['source_freeze'],freeze)
        shared=bridge._current_inventory(bridge._CurrentMetadata(plan),plan['attempts'][0],entry,config,plan['collector'],root)
        assert freeze['files'][code]==freeze['files'][duplicate]
        assert shared['code:'+relative]==dict(kind='code',relative_path=relative,sha256=archive['sha256'])
    finally:
        manifest_path.write_bytes(manifest_raw);path.write_bytes(raw)
    for fault in ('traversal','foreign_code','foreign_fixed','missing_given','missing_capture_code','archive_alias','archive_bytes_alias','unknown_freeze'):
        plan=json.loads(json.dumps(base));freeze=json.loads(raw)
        code=bindings['target_estimator']['file'];archive=deepcopy(freeze['files'][code])
        if fault in ('traversal','foreign_code','foreign_fixed'):
            foreign=(base['collector']['root']+'/sim/../../foreign.py' if fault=='traversal' else
                base['collector']['root']+'/sim/foreign.py' if fault=='foreign_code' else '/foreign/fixed.json')
            freeze['files'][foreign]=archive
            plan['accounting_scope']['extra_input_refs'].append(dict(category='source',reference=dict(path=foreign,sha256=archive['sha256'],bytes=archive['bytes'])))
        elif fault=='missing_given':freeze['files'].pop(roles['fresh_plan']['file'])
        elif fault=='missing_capture_code':freeze['files'].pop(code)
        elif fault=='archive_alias':
            archive['path']=str(root/'archive/same-byte-clone');freeze['files'][code]=archive
            plan['accounting_scope']['extra_input_refs'].append(dict(category='source',reference=archive))
        elif fault=='archive_bytes_alias':freeze['files'][code]['bytes']=float(archive['bytes'])
        else:freeze['unknown']=False
        rebind_current_json(plan,plan['attempts'][0]['source_freeze'],freeze)
        with pytest.raises(ValueError):bridge._current_inventory(bridge._CurrentMetadata(plan),plan['attempts'][0],entry,config,plan['collector'],root)
    path.write_bytes(raw)


def test_current_recipe_rejects_self_consistent_camera_calibration_omission_in_all_240_freezes(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    _,_,mapping,_,processing,report=current_population(tmp_path,monkeypatch)
    plan=json.loads(Path(processing['plan']['path']).read_text());recipe=json.loads(Path(plan['source_recipe']['path']).read_text())
    recipe['inventory'].pop('fixed:camera');recipe['inventory'].pop('fixed:calibration')
    rebind_current_json(plan,plan['source_recipe'],recipe)
    for attempt,entry in zip(plan['attempts'],report['plans']):
        config=json.loads(Path(entry['runtime']['path']).read_text());capture=config['fresh_expert']['raw_multimodal_capture']
        freeze=json.loads(Path(attempt['source_freeze']['path']).read_text())
        freeze['files'].pop(capture['camera_file']);freeze['files'].pop(capture['calibration_file'])
        rebind_current_json(plan,attempt['source_freeze'],freeze)
    assert len(plan['attempts'])==240
    with pytest.raises(ValueError,match='fixed/capture source role coverage'):bridge.authenticated_view(mapping,plan)


def test_current_inventory_requires_each_configured_fixed_role_in_last_attempt(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as bridge
    from scripts.experiments.wam_runpod_20261027 import given_formal_collection_lossless_v1 as producer
    _,_,_,_,processing,report=current_population(tmp_path,monkeypatch)
    base=json.loads(Path(processing['plan']['path']).read_text());entry=report['plans'][-1]
    config=json.loads(Path(entry['runtime']['path']).read_text());root=Path(base['derivation']['path']).parent
    fixed=producer._fixed_sources(config['m2_given_contact_control']['metadata'],config)
    original=json.loads(Path(base['attempts'][-1]['source_freeze']['path']).read_text())
    assert len(base['attempts'])==240
    for role in ('calibration','native_qualification','sdk_mdl_0000'):
        plan=json.loads(json.dumps(base));freeze=json.loads(json.dumps(original));freeze['files'].pop(fixed[role]['file'])
        rebind_current_json(plan,plan['attempts'][-1]['source_freeze'],freeze)
        with pytest.raises(ValueError,match='fixed/capture source role coverage'):
            bridge._current_inventory(bridge._CurrentMetadata(plan),plan['attempts'][-1],entry,config,plan['collector'],root)
