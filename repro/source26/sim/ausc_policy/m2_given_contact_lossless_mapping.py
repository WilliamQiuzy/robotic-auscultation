"""Authenticated metadata projection of the fixed allocation onto lossless refs.

This reader never opens scene/body/array payloads and never authors a mapping.
The caller retains the original mapping path/SHA as its permit capability. Only
the three prepared JSON roles and each plan's patient provenance can change.
"""
from copy import deepcopy
from pathlib import Path
import re
import hashlib
import json

from . import layered_scene_authoring as layers, m2_given_contact_runtime as runtime, scene_store

need=runtime.need
SCHEMA='m2_given_lossless_derivation_map_v1'
RECIPE_SCHEMA='m2_given_qualified_source_recipe_lossless_v1'
NUMERIC_CODEC='framed_npz_gzip_single_member_v1'
SCENE_ROLES=('preparation_config','prepared_complete','patient_receipt')
POSE_TEMPLATES=frozenset(('p000474_a001230_f0000','p000476_a001231_f0023'))
SELECTED_PROFILE=dict(file='/workspace/wam/configs/m2-original022-solver-isolation-20261027-v1/prepared/position64_velocity1/physics_profile.json',
    sha256='3ae2247e5c2f6ad9ac8f8b4c9474a96e97a97dac5cd6bc4fa2d8b58935fd982d')


def _reference(value):
    need(isinstance(value,dict) and set(value) in ({'file','sha256'},{'file','sha256','bytes'},
        {'path','sha256'},{'path','sha256','bytes'}), 'Exact lossless source path/SHA role required')
    ref=dict(path=value.get('path',value.get('file')),sha256=value['sha256'])
    if 'bytes' in value:
        need(type(value['bytes']) is int and value['bytes']>=0, 'Exact lossless source bytes required')
        ref['bytes']=value['bytes']
    runtime._reference_shape(ref)
    need(str(Path(ref['path']))==ref['path'] and '..' not in Path(ref['path']).parts,
        'Canonical absolute lossless source role required')
    return ref


def _file(value):
    ref=_reference(value);return dict(file=ref['path'],sha256=ref['sha256'])


class _Metadata:
    """Bind declared sources without opening them; read only registered JSON."""
    def __init__(self,plan):
        self.inputs={};self.cache={}
        for entry in plan.get('accounting_scope',{}).get('extra_input_refs',[]):
            ref=_reference(entry.get('reference'));path=ref['path']
            need('bytes' in ref and path not in self.inputs, 'Unique complete lossless retained-input roles required')
            self.inputs[path]=ref

    def bind(self,value):
        ref=_reference(value);registered=self.inputs.get(ref['path'])
        need(registered is not None and all(registered[k]==v for k,v in ref.items()),
            'Lossless source role outside the frozen plan input inventory')
        return registered

    def read(self,value):
        ref=self.bind(value);path=ref['path']
        need(Path(path).suffix=='.json', 'Lossless bridge reads JSON metadata only; payload access forbidden')
        if path not in self.cache:self.cache[path]=runtime.read_bound(ref)
        return deepcopy(self.cache[path])


def _source_inventory(reader,sources):
    need(isinstance(sources,dict) and bool(sources), 'Complete lossless source-role inventory required')
    for ref in sources.values():reader.bind(ref)


def _execution_sources(report,recipe,reader):
    for value in (report,recipe):
        for key in ('author_helper','selected_profile'):reader.bind(value.get(key))
    reader.bind(recipe.get('qualification_helper'))
    cpu=report['cpu_execution'];collector=cpu['collector'];root=Path(cpu['collector_root'])
    for key,name in (('code_commit_file','CODE_COMMIT'),('source_manifest','SOURCE_MANIFEST.json')):
        ref=reader.bind(collector.get(key))
        need(ref['path']==str(root/name), 'Exact frozen collector source role required')
    reader.bind(cpu['provider'])
    executable=cpu.get('executable',{})
    need(isinstance(executable,dict) and set(executable)=={'invoked_path','path','sha256','bytes'},
        'Exact invoked interpreter and real binary reference required')
    _reference(dict(path=executable['invoked_path'],sha256=executable['sha256']))
    reader.bind({key:value for key,value in executable.items() if key!='invoked_path'})


def _population(mapping):
    scenes={};roles=set();selected=[];seen=set();allocated=[]
    for scene in mapping.get('scenes',[]):
        runtime.shared.require_membership(scene.get('split'),scene.get('subject_id'))
        key=scene.get('proposal_id')
        need(isinstance(key,str) and key not in scenes, 'Unique original scene proposal required')
        scenes[key]=scene
    for split,n,seed in (('train',200,2026102701),('validation',40,2026102702)):
        rows=mapping.get('ordered_allocations',{}).get(split,[])
        need(len(rows)==n and mapping.get('summaries',{}).get(split,{}).get('seed')==seed,
            'Fixed 200 TRAIN/40 VAL allocation and original seeds required')
        for a in rows:
            runtime.shared.require_membership(split,a.get('subject_id'));scene=scenes.get(a.get('proposal_id'),{})
            need(scene.get('split')==split and scene.get('subject_id')==a['subject_id'] and
                a.get('scene_ref')==scene.get('proposal_id') and scene.get('template') in POSE_TEMPLATES and
                a.get('target') in {side+str(j) for side in 'LR' for j in range(1,6)},
                'Original allocation identity/template/target/proposal changed')
            _file(a.get('fresh_plan'));allocated.append(dict(a,split=split))
            if a['proposal_id'] in seen:continue
            for role in SCENE_ROLES:
                path=_reference(scene.get(role))['path']
                need(path not in roles, 'Original scene role alias across identities or roles')
                roles.add(path)
            seen.add(a['proposal_id']);selected.append(scene)
    need(len({a.get('slot_id') for a in allocated})==240 and
        len({(a['split'],a['subject_id'],a['target']) for a in allocated})==240,
        'Unique fixed slot and subject/target allocation required')
    return selected,allocated,roles


def _iteration(proof):
    roles=proof.get('rigid_articulation_roles',[]);changes=proof.get('only_iteration_changes',[])
    need(proof.get('schema')=='m2_solver_iteration_only_authoring_v2' and proof.get('profile')==layers.ITERATIONS and
        len(roles)==len(changes)==24 and len({tuple(row) for row in roles})==24 and
        all(type(c.get('before')) is int and c['before']==32 and type(c.get('after')) is int and c['after']==64 and
            (c.get('path'),c.get('attribute'))==(role[0],role[1]+':solverPositionIterationCount') for c,role in zip(changes,roles)) and
        bool(proof.get('float32_armature_attributes')) and proof.get('armature_authoring_changed') is False and
        proof.get('geometry_camera_drives_mass_contact_unchanged') is True and
        proof.get('exhaustive_metadata_API_relationship_connection_time_sample_check') is True and
        all(proof.get(k) is False for k in ('physics_executed','physical_validation','training_eligible')) and
        all(re.fullmatch('[0-9a-f]{64}',str(proof.get(k))) for k in
            ('before_stage_semantics_sha256','after_stage_semantics_sha256','before_authored_sha256','after_authored_sha256')),
        'Exact reviewed 32-to64 iteration-only metadata proof required')


def _scene_headers(report,selected,allocated,plan,reader):
    """Validate the entire ordered identity/role inventory before leaf JSON."""
    rows=report.get('scenes',[]);entries=report.get('plans',[]);attempts=plan.get('attempts',[])
    need([row.get('original') for row in rows]==selected and len(entries)==len(attempts)==240,
        'Complete ordered original scenes and 240 plan lineages required')
    original_paths={_reference(scene[role])['path'] for scene in selected for role in SCENE_ROLES}
    derived_paths=set();scenes={}
    for row,original in zip(rows,selected):
        need(set(row)=={'proposal_id','original','derived'} and row['proposal_id']==original['proposal_id'],
            'Derived scene proposal/order changed')
        derived=row['derived'];result=derived.get('derivation',{});restored=deepcopy(derived)
        restored.pop('derivation',None)
        for role in SCENE_ROLES:restored[role]=original[role]
        need(restored==original and result.get('original')=={k:_file(original[k]) for k in SCENE_ROLES},
            'Derived scene identity/template or original source roles changed')
        for role in SCENE_ROLES:
            need(derived.get(role)==result.get(role), 'Derived scene/provider output role differs')
            ref=_reference(derived[role]);path=ref['path']
            need(path not in original_paths|derived_paths, 'Derived scene role alias across identities or roles')
            derived_paths.add(path);reader.bind(ref)
        scenes[row['proposal_id']]=row
    plan_paths=set()
    for i,(entry,a,attempt) in enumerate(zip(entries,allocated,attempts)):
        need(set(entry)=={'slot_id','proposal_id','original','derived','baseline_config','runtime','builder_receipt','reference_binding'} and
            entry['slot_id']==a['slot_id'] and entry['proposal_id']==a['proposal_id'] and entry['original']==_file(a['fresh_plan']) and
            all(attempt.get(k)==a[k] for k in ('slot_id','subject_id','split')) and attempt.get('target_id')==a['target'] and
            attempt.get('preclassified_reachable') is True and entry['runtime']==attempt.get('config'),
            'Derived plan order/identity/target/runtime differs from fixed allocation')
        path=_reference(entry['derived'])['path']
        need(path==str(Path(_reference(plan['derivation'])['path']).parent/'plans'/(str(i+1).zfill(3)+'.json')) and
            path not in original_paths|derived_paths|plan_paths and path!=_reference(entry['original'])['path'],
            'Derived plan exact role or alias relation changed')
        plan_paths.add(path)
        for key in ('original','derived','baseline_config','runtime','builder_receipt'):reader.bind(entry[key])
    return scenes


def _authored(report,selected,reader,root):
    authored=report.get('layered_authoring',{});receipt=reader.read(authored.get('receipt'))
    need(authored.get('schema')==receipt.get('schema')==layers.RECEIPT_SCHEMA and
        _reference(authored['receipt'])['path']==str(root/'derived/scene_authoring_receipt.json') and
        receipt.get('original_selected')==report['authoring_request']['selected'] and
        receipt.get('representation_change_only') is True and receipt.get('runtime_equivalence_policy')==layers.RUNTIME_EQUIVALENCE_POLICY and
        all(receipt.get(k) is False for k in ('physics_executed','physical_validation','training_eligible')) and
        all(authored.get(k)==receipt.get(k) for k in ('base','static_subtrees','original_bytes','saved_bytes')) and
        bool(receipt.get('static_subtrees')) and
        len(authored.get('prepared',[]))==len(receipt.get('scenes',[]))==len(receipt.get('iteration_proofs',[]))==len(receipt.get('input_bindings',[]))==len(selected),
        'Complete bound layered authoring receipt required')
    reader.bind(authored['base'])
    for scene,result,item,proof,inputs in zip(report['scenes'],authored['prepared'],receipt['scenes'],receipt['iteration_proofs'],receipt['input_bindings']):
        need(scene['derived']['derivation']==result and result.get('original')=={k:_file(scene['original'][k]) for k in SCENE_ROLES} and
            result.get('input_bindings')==inputs and result.get('scene_authoring_receipt')==authored['receipt'] and
            result.get('iteration_proof')==proof and result.get('dependency_inventory')==item.get('dependency_inventory') and
            result.get('baseline_scene_sha256')==item.get('original',{}).get('sha256') and
            result.get('composed_semantics_sha256')==item.get('composed_semantics_sha256') and
            item.get('saved_USD_readback_exact') is True and item.get('private_runtime_readback_exact') is True,
            'Derived scene/provider/input/readback lineage changed')
        _iteration(proof);inventory=result['dependency_inventory']
        need(set(inventory)=={'schema','primary','base'} and inventory['schema']==layers.INVENTORY_SCHEMA and inventory['base']==authored['base'] and
            _reference(inventory['base'])['path']==str(root/'derived/static_base.usdc'), 'Exact closed layered dependency roles required')
        directory=Path(_reference(inventory['primary'])['path']).parent
        need(directory.parent==root/'derived' and inventory['primary']['file']==str(directory/'candidate_physics.usdc'),
            'Exact derived scene primary role required')
        for key in ('primary','base'):reader.bind(inventory[key])
        dense=reader.bind(item.get('original'));oldroot=Path(scene['original']['prepared_complete']['file']).parent
        need(dense['path']==str(oldroot/'candidate_physics.usdc'), 'Exact original dense scene source role required')
        for key,name in (('preparation_config','prepare.json'),('prepared_complete','COMPLETE.json'),('patient_receipt','patient_receipt.json'),('physics_profile','physics_profile.json')):
            need(_reference(result[key])['path']==str(directory/name), 'Exact per-scene derived JSON role required')
        refs={}
        for ref in inputs:
            bound=reader.bind(ref);need(bound['path'] not in refs, 'Duplicate original authoring source role');refs[bound['path']]=bound
        for role in SCENE_ROLES:
            ref=_reference(scene['original'][role]);need(ref['path'] in refs and refs[ref['path']]['sha256']==ref['sha256'],
                'Original scene metadata omitted from authoring inputs')
        oldprep,oldcomplete,patient=[reader.read(scene['original'][key]) for key in SCENE_ROLES]
        newprep,newcomplete,newpatient=[reader.read(result[key]) for key in SCENE_ROLES]
        need(patient==newpatient and oldcomplete.get('patient')==patient and oldcomplete.get('completed') is True and
            dense['sha256']==oldcomplete.get('effective_scene_sha256') and
            patient.get('identity')==scene['original']['subject_id'] and patient.get('split')==scene['original']['split'] and
            patient.get('non_patient_scene_unchanged') is True and
            oldprep.get('patient_asset',{}).get('expected_split','train')==scene['original']['split'],
            'Original or derived patient identity differs before payload access')
        expected=deepcopy(oldprep);expected['physics']=deepcopy(layers.ITERATIONS)
        need(oldprep.get('physics')==dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02) and newprep==expected,
            'Preparation changed beyond exact reviewed iteration profile')
        expected=deepcopy(oldcomplete);expected.update(effective_scene_sha256=inventory['primary']['sha256'],
            scene_dependency_inventory=inventory,scene_authoring_receipt=authored['receipt'],solver_isolation_lineage=result['physics_profile'],
            preparation_backend='cpu_lossless_static_layer_and_exact_existing_iteration_authoring')
        need(newcomplete==expected, 'Complete metadata changed beyond authenticated layered provenance')
        profile=reader.read(result['physics_profile']);oldroot=Path(scene['original']['prepared_complete']['file']).parent
        reader.bind(profile.get('inherited_profile'))
        need(profile==dict(proof,original_scene=item['original'],inherited_profile=profile.get('inherited_profile'),
            new_scene=inventory['primary'],representation_lineage=authored['receipt'],saved_USD_readback_exact=True) and
            _reference(profile['inherited_profile'])['path']==str(oldroot/'physics_profile.json'),
            'Iteration profile provenance differs from exact authoring receipt')
    parity=reader.read(report['runtime_equivalence_receipt']);proofs=authored.get('runtime_equivalence',[])
    need(_reference(report['runtime_equivalence_receipt'])['path']==str(root/'SDK_RUNTIME_EQUIVALENCE.json') and parity==dict(
        schema='m2_given_lossless_sdk_cpu_receipt_v1',authoring_request=report['authoring_request'],cpu_execution=report['cpu_execution'],
        scene_authoring_receipt=authored['receipt'],proofs=proofs) and len(proofs)==len(selected) and
        all(p.get('schema')=='m2_lossless_sdk_private_runtime_equivalence_v1' and
            all(p.get(k) is True for k in ('SDK_selected_runtime_readback_exact','all_layers_private_anonymous')) and
            all(p.get(k) is False for k in ('dense_copy_created','physics_executed','training_eligible')) for p in proofs),
        'Exact bound offline SDK runtime equivalence metadata required')
    for proof in proofs:
        for key in ('original_sources','derived_sources'):_source_inventory(reader,proof.get(key))


def authenticated_view(mapping,plan):
    """Return a copy only after exhaustive JSON lineage validation succeeds."""
    if plan.get('current_raw_equivalence') is True:return _current_view(mapping,plan)
    selected,allocated,_=_population(mapping);reader=_Metadata(plan);report=reader.read(plan['derivation'])
    root=Path(_reference(plan['derivation'])['path']).parent;recipe=reader.read(plan.get('source_recipe'))
    if ('current_raw_prepared' in report or 'current_raw_equivalence' in recipe or 'scene_dependency_policy' in recipe or
        recipe.get('collector_conditions',{}).get('scene_dependencies',{}).get('dependency_policy')==scene_store.SDK_LAYER_CURRENT_RAW_POLICY):
        return _current_view(mapping,plan)
    need(set(report)=={'schema','author_helper','selected_profile','profile','unique_original_scenes','scenes','plans',
        'numerical_plans_unchanged','authoring_request','layered_authoring','cpu_execution','runtime_equivalence_receipt','physics_executed','training_eligible'} and
        report.get('schema')==SCHEMA and report.get('unique_original_scenes')==len(selected) and report.get('profile')==layers.ITERATIONS and
        report.get('numerical_plans_unchanged') is True and report.get('physics_executed') is False and report.get('training_eligible') is False and
        recipe.get('schema')==RECIPE_SCHEMA and recipe.get('numeric_container_codec')==NUMERIC_CODEC and
        report.get('author_helper')==recipe.get('author_helper') and report.get('selected_profile')==recipe.get('selected_profile')==SELECTED_PROFILE and
        recipe.get('collector_conditions',{}).get('physics')==layers.ITERATIONS and
        all(recipe['collector_conditions'].get('scene_dependencies',{}).get(k)==v for k,v in
            (('schema',scene_store.SDK_LAYER_SCHEMA),('dependency_policy',scene_store.SDK_LAYER_DEPENDENCY_POLICY))),
        'Exact reviewed lossless derivation schema/profile/source recipe required')
    scenes=_scene_headers(report,selected,allocated,plan,reader)
    expected=dict(selected=[{k:_file(scene[k]) for k in SCENE_ROLES} for scene in selected],output_root=str(root/'derived'),
        iteration_spec=deepcopy(layers.ITERATIONS),baseline_configs=[next(e['baseline_config'] for e in report['plans'] if e['proposal_id']==s['proposal_id']) for s in selected])
    cpu=report.get('cpu_execution',{})
    need(report.get('authoring_request')==expected and cpu.get('collector')==plan.get('collector') and
        cpu.get('collector_root')==plan.get('collector',{}).get('root') and
        _reference(cpu.get('provider'))['path']==str(Path(cpu['collector_root'])/'sim/ausc_policy/layered_scene_authoring.py') and
        cpu.get('request_sha256')==runtime.digest(expected) and re.fullmatch('[0-9a-f]{64}',str(cpu.get('script_sha256'))) and
        cpu.get('gpu_work') is False and cpu.get('physics_advanced') is False,
        'Exact ordered frozen CPU authoring request/provider binding required')
    _execution_sources(report,recipe,reader)
    _authored(report,selected,reader,root)
    for entry,a in zip(report['plans'],allocated):
        scene=scenes[a['proposal_id']];old=reader.read(entry['original']);new=reader.read(entry['derived'])
        restored=deepcopy(new);restored.setdefault('patient_provenance',{})['binding']=old.get('patient_provenance',{}).get('binding')
        need(restored==old and old.get('patient_provenance',{}).get('binding')==_file(scene['original']['patient_receipt']) and
            new.get('patient_provenance',{}).get('binding')==_file(scene['derived']['patient_receipt']) and
            old.get('patient_provenance',{}).get('identity')==a['subject_id'] and old.get('name')==a['target'] and old.get('success') is True,
            'Numerical plan or exact original-to-derived patient provenance changed')
        baseline=reader.read(entry['baseline_config']);config=reader.read(entry['runtime']);receipt=reader.read(entry['builder_receipt'])
        for value in (baseline,config):_source_inventory(reader,value.get('compact_scene_storage',{}).get('sources'))
        runtimeprep=reader.read(receipt.get('runtime_preparation'));expectedprep=reader.read(scene['derived']['preparation_config'])
        for key in ('base','time_limit_seconds'):
            need(config.get(key)==baseline.get(key), 'Runtime preparation changed original base/deadline')
            expectedprep[key]=baseline.get(key)
        need(runtimeprep==expectedprep and baseline.get('physics')==dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02) and
            config.get('physics')==layers.ITERATIONS, 'Builder preparation or selected runtime physics differs from exact derived scene')
        refs=[]
        for value,patient,planref in ((baseline,scene['original']['patient_receipt'],entry['original']),
                                     (config,scene['derived']['patient_receipt'],entry['derived'])):
            meta=value.get('m2_given_contact_control',{}).get('metadata',{});ref=runtime.pose.require_route(meta)
            need(all(meta.get(k)==a[k] for k in ('subject_id','split')) and meta.get('target_id')==a['target'] and
                ref.snapshot()['sources']==dict(task_manifest=_file(patient),patient_receipt=_file(patient),fresh_plan=_file(planref)),
                'Fixed runtime identity/target/reference roles changed')
            refs.append(ref)
        before,after=[ref.snapshot() for ref in refs];sources=deepcopy(after['sources']);after['sources']=before['sources']
        need(after==before and entry['reference_binding']==dict(original_reference_sha256=refs[0].sha256,derived_reference_sha256=refs[1].sha256,
            original_sources=before['sources'],derived_sources=sources) and baseline.get('seed')==config.get('seed'),
            'Prescribed numerical pose/availability/seed or exact reference lineage changed')
    view=deepcopy(mapping)
    for i,scene in enumerate(view['scenes']):
        if scene['proposal_id'] in scenes:view['scenes'][i]=deepcopy(scenes[scene['proposal_id']]['derived'])
    for entry,a in zip(report['plans'],[a for split in ('train','validation') for a in view['ordered_allocations'][split]]):
        a['fresh_plan']=deepcopy(entry['derived'])
    return view


def _same(left,right):
    """JSON equality preserves bool/int/float distinctions in provenance."""
    return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)


def _keys(value,keys):
    need(isinstance(value,dict) and set(value)==set(keys.split()), 'Exact current RAW metadata fields required')


class _CurrentMetadata(_Metadata):
    """CURRENT-only parser: authenticate raw bytes before losing JSON evidence."""
    def bind(self,value):
        ref=_reference(value);registered=self.inputs.get(ref['path'])
        need(registered is not None and all(registered[k]==v for k,v in ref.items()),
            'Current RAW source outside the frozen plan input inventory: '+ref['path'])
        return registered

    def read(self,value):
        ref=self.bind(value);path=Path(ref['path'])
        need(path.suffix=='.json' and path.name!='collision_geometry.json',
            'Current RAW bridge reads registered provenance JSON only')
        if str(path) not in self.cache:
            need(not any(p.is_symlink() for p in (path,*path.parents)) and path.is_file(),
                'Canonical current RAW JSON source required')
            data=path.read_bytes()
            need(len(data)==ref['bytes'] and hashlib.sha256(data).hexdigest()==ref['sha256'],
                'Current RAW JSON source hash/bytes changed')
            self.cache[str(path)]=scene_store._strict_json(data)
        return deepcopy(self.cache[str(path)])


def _current_proof(proof):
    _keys(proof,'schema profile rigid_articulation_roles only_iteration_changes float32_armature_attributes '
        'armature_authoring_changed geometry_camera_drives_mass_contact_unchanged '
        'exhaustive_metadata_API_relationship_connection_time_sample_check physics_executed physical_validation '
        'training_eligible before_stage_semantics_sha256 after_stage_semantics_sha256 before_authored_sha256 after_authored_sha256')
    _iteration(proof)
    need(_same(proof['profile'],layers.ITERATIONS), 'Typed current RAW iteration profile required')
    roles=proof['rigid_articulation_roles']
    need(isinstance(roles,list) and all(isinstance(row,list) and len(row)==2 and
        isinstance(row[0],str) and row[0].startswith('/') and row[1] in ('physxRigidBody','physxArticulation') for row in roles),
        'Exact current RAW rigid/articulation roles required')
    expected=[dict(path=p,attribute=r+':solverPositionIterationCount',before=32,after=64) for p,r in roles]
    need(_same(proof['only_iteration_changes'],expected), 'Typed exact iteration changes required')
    armatures=proof['float32_armature_attributes']
    need(isinstance(armatures,list) and armatures and len(set(armatures))==len(armatures) and
        all(isinstance(p,str) and p.startswith('/') and p.endswith(':armature') for p in armatures),
        'Exact current RAW float32 armature inventory required')


def _current_sources(reader,binding):
    # This shape validator only checks declared references; it never opens files.
    scene_store._binding_shape(binding)
    for ref in binding['sources'].values():
        need(set(ref)=={'file','bytes','sha256'}, 'Complete current RAW source reference required')
        reader.bind(ref)
    return binding['sources']


def _current_headers(report,recipe,selected,allocated,plan,reader,root,*,checkpoint=None):
    """All ordered identities and role declarations precede leaf JSON reads."""
    _keys(report,'schema author_helper selected_profile profile unique_original_scenes scenes plans numerical_plans_unchanged '
        'authoring_request layered_authoring current_raw_prepared cpu_execution runtime_equivalence_receipt physics_executed training_eligible'+
        ('' if checkpoint is None else ' completed_authoring_checkpoint'))
    source_root=root if checkpoint is None else Path(checkpoint['source_root'])
    _keys(recipe,'schema qualification qualification_all_eight author_helper qualification_helper numeric_container_codec '
        'current_raw_equivalence scene_dependency_policy selected_profile inventory collector_conditions')
    request=report['authoring_request'];_keys(request,'selected raw_selected baseline_provenance baseline_configs output_root '
        'current_raw_output_root current_raw_equivalence iteration_spec')
    conditions=recipe['collector_conditions']
    _keys(conditions,'top_keys physics fresh capture isaac_python scene_dependencies')
    dependencies=conditions['scene_dependencies'];_keys(dependencies,'schema dependency_policy sources')
    _source_inventory(reader,dependencies['sources'])
    for key in ('qualification','qualification_all_eight'):reader.bind(recipe[key])
    need(recipe['schema']==RECIPE_SCHEMA and recipe['numeric_container_codec']==NUMERIC_CODEC and
        dependencies['schema']==scene_store.SDK_LAYER_SCHEMA and dependencies['dependency_policy']==scene_store.SDK_LAYER_CURRENT_RAW_POLICY and
        recipe['scene_dependency_policy']==scene_store.SDK_LAYER_CURRENT_RAW_POLICY and recipe['current_raw_equivalence'] is True and
        request['current_raw_equivalence'] is True and report['schema']==SCHEMA and
        type(report['unique_original_scenes']) is int and report['unique_original_scenes']==len(selected) and
        _same(report['profile'],layers.ITERATIONS) and _same(recipe['collector_conditions']['physics'],layers.ITERATIONS) and
        report['numerical_plans_unchanged'] is True and report['physics_executed'] is False and report['training_eligible'] is False and
        report['author_helper']==recipe['author_helper'] and report['selected_profile']==recipe['selected_profile']==SELECTED_PROFILE,
        'Explicit exact current RAW derivation policy/profile required')
    need(request['raw_selected']==[{k:_file(s[k]) for k in SCENE_ROLES} for s in selected] and
        request['output_root']==str(source_root/'derived') and request['current_raw_output_root']==str(source_root/'current_raw') and
        _same(request['iteration_spec'],layers.ITERATIONS) and isinstance(request['selected'],list) and
        len(request['selected'])==len(selected)==len(report['current_raw_prepared']) and
        [row['original'] for row in report['scenes']]==selected,
        'Complete ordered current RAW selection/normalization headers required')
    # Validate new fields first; only then construct the historical header view.
    historical=deepcopy(report);normalized=[];aliases=set()
    for i,(row,scene,selection,record) in enumerate(zip(report['scenes'],selected,request['selected'],report['current_raw_prepared'])):
        _keys(row,'proposal_id original derived');_keys(selection,'preparation_config prepared_complete patient_receipt')
        _keys(record,'historical_prepared_complete layered_runtime_equivalence prepared_complete')
        derived=row['derived'];result=derived['derivation']
        need(set(derived)==set(scene)|{'derivation','current_raw','raw_preparation_config'} and
            derived['current_raw']==record and derived['raw_preparation_config']==reader.bind(scene['preparation_config']) and
            derived['prepared_complete']==record['prepared_complete'] and result['original']==selection and
            selection['prepared_complete']==_file(scene['prepared_complete']) and selection['patient_receipt']==_file(scene['patient_receipt']) and
            all(derived[k]==result[k] for k in ('preparation_config','patient_receipt')),
            'Exact current RAW scene identity/source declaration required')
        _keys(result,'preparation_config prepared_complete patient_receipt physics_profile original input_bindings dependency_inventory '
            'baseline_scene_sha256 iteration_proof composed_semantics_sha256 scene_authoring_receipt')
        _keys(result['dependency_inventory'],'schema primary base')
        for ref in result['input_bindings']:reader.bind(ref)
        for key in ('primary','base'):reader.bind(result['dependency_inventory'][key])
        for key in ('physics_profile','scene_authoring_receipt'):reader.bind(result[key])
        directory=source_root/'current_raw'/str(i).zfill(3)
        for key,path in (('prepared_complete',directory/'COMPLETE.json'),('layered_runtime_equivalence',directory/'layered_runtime_equivalence.json')):
            ref=reader.bind(record[key]);need(ref['path']==str(path) and ref['path'] not in aliases,
                'Exact unique current RAW metadata output role required');aliases.add(ref['path'])
        need(set(record['layered_runtime_equivalence'])=={'file','bytes','sha256'} and
            set(record['historical_prepared_complete'])=={'file','bytes','sha256'} and
            set(record['prepared_complete'])=={'file','sha256'} and
            _reference(record['historical_prepared_complete'])==reader.bind(result['prepared_complete']),
            'Exact historical COMPLETE and full sidecar source required')
        reader.bind(selection['preparation_config']);reader.bind(derived['raw_preparation_config'])
        normalized_scene=dict(scene,preparation_config=selection['preparation_config']);normalized.append(normalized_scene)
        h=historical['scenes'][i];h['original']=normalized_scene
        h['derived'].pop('current_raw');h['derived'].pop('raw_preparation_config')
        h['derived']['prepared_complete']=deepcopy(result['prepared_complete'])
    need(len(report['plans'])==240, 'All current RAW plan declarations required')
    authored=report['layered_authoring']
    _keys(authored,'schema base prepared receipt original_bytes saved_bytes static_subtrees runtime_equivalence')
    reader.bind(authored['receipt']);reader.bind(authored['base'])
    need(len(authored['prepared'])==len(authored['runtime_equivalence'])==len(selected), 'Complete authored source declarations required')
    need(all(_same(result,row['derived']['derivation']) for result,row in zip(authored['prepared'],report['scenes'])),
        'Every prepared declaration must equal its authenticated scene header before leaf JSON')
    for proof in authored['runtime_equivalence']:
        for key in ('original_sources','derived_sources'):_source_inventory(reader,proof[key])
    for entry in report['plans']:
        _keys(entry,'slot_id proposal_id original derived baseline_config baseline_provenance runtime builder_receipt reference_binding')
        provenance=reader.bind(entry['baseline_provenance']);baseline=reader.bind(entry['baseline_config'])
        need(provenance['path']==str(Path(baseline['path']).with_suffix('.provenance.json')),
            'Exact sibling baseline provenance source required')
    for entry in historical['plans']:entry.pop('baseline_provenance')
    _scene_headers(historical,normalized,allocated,plan,reader)
    freezes=set()
    for attempt in plan['attempts']:
        aid=attempt.get('attempt_id')
        need(isinstance(aid,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',aid),
            'Exact current RAW attempt/source-freeze identity required')
        ref=reader.bind(attempt.get('source_freeze'))
        need(ref['path']==str(root/'episodes'/(aid+'.source-freeze.json')) and ref['path'] not in freezes,
            'Unique exact per-attempt source-freeze metadata role required')
        freezes.add(ref['path'])
    first=[next(e for e in report['plans'] if e['proposal_id']==s['proposal_id']) for s in selected]
    need(request['baseline_configs']==[e['baseline_config'] for e in first] and
        request['baseline_provenance']==[e['baseline_provenance'] for e in first],
        'Exact first-use baseline normalization lineage required')
    return historical,normalized


def _current_authored(report,historical,normalized,reader,root):
    authored=report['layered_authoring'];receipt=reader.read(authored['receipt'])
    _keys(authored,'schema base prepared receipt original_bytes saved_bytes static_subtrees runtime_equivalence')
    _keys(receipt,'schema base scenes static_subtrees original_bytes saved_bytes representation_change_only physics_executed '
        'physical_validation training_eligible iteration_proofs runtime_equivalence_policy original_selected input_bindings')
    for value in (authored,receipt):
        need(all(type(value[k]) is int and value[k]>=0 for k in ('original_bytes','saved_bytes')),
            'Typed authored byte counts required')
    for i,(row,result,item,proof,record) in enumerate(zip(report['scenes'],authored['prepared'],receipt['scenes'],receipt['iteration_proofs'],report['current_raw_prepared'])):
        _keys(item,'original dependency_inventory composed_semantics_sha256 authored_semantics_sha256 saved_USD_readback_exact private_runtime_readback_exact')
        _current_proof(proof)
        need(_same(row['derived']['derivation'],result) and re.fullmatch('[0-9a-f]{64}',item['authored_semantics_sha256']),
            'Exact closed current RAW authoring scene proof required')
        sidecar=reader.read(record['layered_runtime_equivalence'])
        _keys(sidecar,'schema dependency_policy historical_prepared_complete authoring_receipt historical_prepared_sources scene_index '
            'dependency_inventory original_selected original_dense original_iteration_proof historical_authored_semantics_sha256 '
            'historical_digest_scope physics_executed physical_validation training_eligible USD_written')
        need(type(sidecar['scene_index']) is int and sidecar['scene_index']==i, 'Typed exact current RAW sidecar index required')
        sources=sidecar['historical_prepared_sources'];directory=Path(result['prepared_complete']['file']).parent
        expected_roles={'preparation_config',*scene_store.FILES}
        optional=set(sources)&set(scene_store.OPTIONAL)
        need(set(sources)==expected_roles|optional, 'Closed historical prepared FILES inventory required')
        for role,name in {**scene_store.FILES,**scene_store.OPTIONAL,'preparation_config':'prepare.json'}.items():
            if role not in sources:continue
            ref=reader.bind(sources[role]);need(set(sources[role])=={'file','bytes','sha256'} and ref['path']==str(directory/name),
                'Historical prepared FILES must remain at the factored root')
            if role not in ('scene','collision_geometry','collision_arrays'):reader.read(ref)
        profile=reader.read(result['physics_profile']);inherited=reader.bind(dict(path=str(Path(row['original']['prepared_complete']['file']).parent/'physics_profile.json'),sha256=profile['inherited_profile']['sha256']))
        expected_profile=dict(proof,original_scene=item['original'],inherited_profile=dict(file=inherited['path'],bytes=inherited['bytes'],sha256=inherited['sha256']),
            new_scene=item['dependency_inventory']['primary'],representation_lineage=authored['receipt'],saved_USD_readback_exact=True)
        need(_same(profile,expected_profile), 'Typed exact iteration profile provenance required')
        need(sources['scene']==result['dependency_inventory']['primary'] and
            all(_file(sources[k])==result[k] for k in ('preparation_config','patient_receipt','physics_profile')),
            'Exact historical prepared source bindings required')
        expected=dict(schema=layers.CURRENT_RAW_SIDECAR_SCHEMA,dependency_policy=scene_store.SDK_LAYER_CURRENT_RAW_POLICY,
            historical_prepared_complete=record['historical_prepared_complete'],authoring_receipt=authored['receipt'],
            historical_prepared_sources=sources,scene_index=i,dependency_inventory=item['dependency_inventory'],
            original_selected=receipt['original_selected'][i],original_dense=item['original'],original_iteration_proof=proof,
            historical_authored_semantics_sha256=item['authored_semantics_sha256'],historical_digest_scope='original_authoring_cpu_sdf_registry_only',
            physics_executed=False,physical_validation=False,training_eligible=False,USD_written=False)
        need(_same(sidecar,expected), 'Exact current RAW sidecar provenance required')
        old=reader.read(record['historical_prepared_complete']);new=reader.read(record['prepared_complete'])
        need(not ({'scene_dependency_policy','layered_runtime_equivalence'} & set(old)) and
            _same(new,dict(old,scene_dependency_policy=scene_store.SDK_LAYER_CURRENT_RAW_POLICY,
                layered_runtime_equivalence=record['layered_runtime_equivalence'])),
            'Current COMPLETE differs beyond its two declared provenance fields')
        paths=[reader.bind(ref)['path'] for ref in (record['historical_prepared_complete'],record['layered_runtime_equivalence'],
            record['prepared_complete'],authored['receipt'],item['original'],item['dependency_inventory']['primary'],item['dependency_inventory']['base'],
            *receipt['original_selected'][i].values())]
        need(len(paths)==len(set(paths)), 'Current RAW provenance roles cannot alias')
        inputs=result['input_bindings']
        need(all(isinstance(ref,dict) and set(ref)=={'file','bytes','sha256'} for ref in inputs),
            'Full original authoring input bindings required')
        names=[reader.bind(ref)['path'] for ref in inputs]
        need(names==sorted(set(names)), 'Canonical complete authoring source inventory required')
        original=row['original'];prep=reader.read(original['preparation_config']);patient=reader.read(original['patient_receipt'])
        spec=prep['patient_asset'];oldroot=Path(original['prepared_complete']['file']).parent
        required={path for path in reader.inputs if oldroot in Path(path).parents}
        required.update(reader.bind(ref)['path'] for ref in receipt['original_selected'][i].values())
        required.add(reader.bind(original['preparation_config'])['path'])
        body=str(Path(spec['directory'])/spec['file']);required.add(body)
        need(reader.bind(dict(path=body,sha256=patient['asset_sha256']))['path']==body, 'Exact original body source binding required')
        for name,key in (('asset_manifest.json','manifest_sha256'),('summary.json','summary_sha256')):
            ref=dict(path=str(Path(spec['directory'])/name),sha256=spec[key]);required.add(reader.bind(ref)['path']);reader.read(ref)
        split=dict(path=spec['identity_split'],sha256=runtime.shared.contract.IDENTITY_SPLIT_SHA256)
        required.add(reader.bind(split)['path']);reader.read(split)
        geometry=prep['geometry_source']
        if geometry==str(layers.HISTORICAL_GEOMETRY_RAW):geometry=str(layers.HISTORICAL_GEOMETRY_TARGET)
        required.add(reader.bind(dict(path=geometry,sha256=prep['geometry_source_sha256']))['path'])
        need(set(names)==required, 'Complete authoring inventory omitted or added a foreign source')
    # The retained historical JSON is read without replacing any registered data.
    _authored(historical,normalized,reader,root)
    for proof in authored['runtime_equivalence']:
        _keys(proof,'schema original_sources derived_sources selected_iteration_proof composed_semantics_sha256 '
            'SDK_selected_runtime_readback_exact all_layers_private_anonymous dense_copy_created physics_executed training_eligible')
        _current_proof(proof['selected_iteration_proof'])
        need(re.fullmatch('[0-9a-f]{64}',proof['composed_semantics_sha256']), 'Exact offline SDK proof digest required')


BUILDER_FIELDS=frozenset(('preparation_config prepared_complete patient_receipt subject_id split identity_split_sha256 body '
    'effective_scene geometry_artifacts payload_identity_preflight test_payloads_opened test_metadata_path_nonaccess_claim '
    'physics_executed physical_validation physics_qualified training_eligible data_accepted learned_policy schema fresh_plan '
    'observation_contract source_bindings camera hall_calibration metadata_is_acquisition_intent_not_execution_evidence cli_module '
    'collision_profile compact_max_frames total_byte_cap compact_scene_storage task_id action_semantics_id given_reference '
    'reference_sha256 handoff_policy_id teacher_boundary translation_noise physical_qualification production_act_ready '
    'collector_cpu_builder runtime_preparation').split())


def _current_builder(reader,receipt,scene,entry,config,collector_root,*,current):
    keys=set(BUILDER_FIELDS)
    if 'native_retainer_audit' in receipt:keys.add('native_retainer_audit')
    if current:keys.add('current_raw_prepared_complete')
    need(set(receipt)==keys and receipt['schema']=='m2_given_contact_compact_expert_config_v2' and
        receipt['task_id']==runtime.pose.TASK_ID and receipt['subject_id']==scene['original']['subject_id'] and
        receipt['split']==scene['original']['split'] and receipt['identity_split_sha256']==runtime.shared.contract.IDENTITY_SPLIT_SHA256 and
        all(receipt[k] is False for k in ('physics_executed','physical_validation','physics_qualified','training_eligible',
            'data_accepted','learned_policy','test_payloads_opened','test_metadata_path_nonaccess_claim','physical_qualification','production_act_ready')) and
        receipt['payload_identity_preflight'] is True and
        type(receipt['compact_max_frames']) is int and receipt['compact_max_frames']==5000 and
        type(receipt['total_byte_cap']) is int and receipt['total_byte_cap']==100000000,
        'Exact closed metadata-only baseline/current builder receipt required')
    selected=scene['derived']['derivation'] if current else scene['original']
    need(all(receipt[k]==_file(selected[k]) for k in SCENE_ROLES) and
        receipt['fresh_plan']==entry['derived' if current else 'original'] and
        _same(receipt['compact_scene_storage'],config['compact_scene_storage']) and
        (not current or receipt['current_raw_prepared_complete']==scene['derived']['prepared_complete']),
        'Exact original/factored builder source lineage required')
    meta=config['m2_given_contact_control']['metadata']
    need(_same(receipt['given_reference'],meta['given_reference']) and receipt['reference_sha256']==meta['reference_sha256'],
        'Builder given-reference lineage differs from its actual config')
    _keys(receipt['source_bindings'],'native_backend state_adapter target_estimator action_source body raw_intent_dispatcher interior_handoff')
    for ref in [*receipt['source_bindings'].values(),receipt['camera'],receipt['hall_calibration'],receipt['body'],receipt['effective_scene']]:reader.bind(ref)
    capture=config['fresh_expert']['raw_multimodal_capture']
    need(reader.bind(receipt['camera'])==reader.bind(dict(file=capture['camera_file'],sha256=capture['camera_sha256'])) and
         reader.bind(receipt['hall_calibration'])==reader.bind(dict(file=capture['calibration_file'],sha256=capture['calibration_sha256'])),
         'Builder camera/Hall receipt differs from actual capture config')
    builder=receipt['collector_cpu_builder'];_keys(builder,'collector_root executable script_sha256 gpu_work physics_advanced')
    executable=builder['executable'];_keys(executable,'invoked_path path sha256 bytes')
    reader.bind({k:v for k,v in executable.items() if k!='invoked_path'})
    need(builder['collector_root']==collector_root and builder['gpu_work'] is False and builder['physics_advanced'] is False and
        re.fullmatch('[0-9a-f]{64}',builder['script_sha256']), 'Bound CPU-only builder metadata required')


def _current_runtime(reader,entry,scene,allocation,collector_root,*,capture_paths=None):
    old=reader.read(entry['original']);new=reader.read(entry['derived'])
    restored=deepcopy(new);restored['patient_provenance']['binding']=old['patient_provenance']['binding']
    need(_same(restored,old) and old['patient_provenance']['binding']==_file(scene['original']['patient_receipt']) and
        new['patient_provenance']['binding']==_file(scene['derived']['patient_receipt']) and old['patient_provenance']['identity']==allocation['subject_id'] and
        old['name']==allocation['target'] and old['success'] is True, 'Numerical plan or patient lineage changed')
    baseline=reader.read(entry['baseline_config']);config=reader.read(entry['runtime']);provenance=reader.read(entry['baseline_provenance'])
    result=scene['derived']['derivation'];record=scene['derived']['current_raw']
    need(all(provenance[k]==_file(scene['original'][k]) for k in SCENE_ROLES), 'Exact raw baseline builder provenance required')
    prepref=provenance['runtime_preparation'];need(set(prepref)=={'path','bytes','sha256'}, 'Full baseline normalized preparation source required')
    raw=reader.read(scene['original']['preparation_config']);normalized=reader.read(prepref)
    need(_same(raw['physics'],dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02)) and
        _same(reader.read(result['preparation_config']),dict(normalized,physics=deepcopy(layers.ITERATIONS))),
        'Typed exact normalized32-to-factored64 preparation required')
    need(type(raw['time_limit_seconds']) is int and raw['time_limit_seconds']==480 and
        baseline['base']=='/workspace/wam' and type(baseline['time_limit_seconds']) is int and baseline['time_limit_seconds']==2400 and
        _same(normalized,dict(raw,base=baseline['base'],time_limit_seconds=2400)), 'Preparation changed beyond exact base/deadline normalization')
    before=_current_sources(reader,baseline['compact_scene_storage']);after=_current_sources(reader,config['compact_scene_storage'])
    need(_reference(before['preparation_config'])==reader.bind(prepref) and
        result['original']==dict(preparation_config=_file(prepref),prepared_complete=_file(scene['original']['prepared_complete']),patient_receipt=_file(scene['original']['patient_receipt'])),
        'Baseline config must retain its exact named normalized preparation')
    oldroot=Path(scene['original']['prepared_complete']['file']).parent;directory=Path(result['prepared_complete']['file']).parent
    for role,name in {**scene_store.FILES,**scene_store.OPTIONAL}.items():
        if role in before:need(before[role]['file']==str(oldroot/name), 'Exact original prepared FILES source path required')
    need(all(_file(before[k])==result['original'][k] for k in SCENE_ROLES) and
        before['scene']['sha256']==result['baseline_scene_sha256'], 'Exact baseline author input source roles required')
    expected=deepcopy(before)
    for key in ('preparation_config','patient_receipt','physics_profile'):expected[key]=dict(file=reader.bind(result[key])['path'],sha256=reader.bind(result[key])['sha256'],bytes=reader.bind(result[key])['bytes'])
    for key,name in {**scene_store.FILES,**scene_store.OPTIONAL}.items():
        if key in before and key not in ('scene','patient_receipt','physics_profile'):
            expected[key]=dict(before[key],file=str(directory/name));reader.bind(expected[key])
            if key not in ('collision_geometry','collision_arrays'):
                need(_same(reader.read(before[key]),reader.read(expected[key])), 'Copied prepared JSON changed')
    expected.update(prepared_complete=dict(file=reader.bind(record['prepared_complete'])['path'],sha256=reader.bind(record['prepared_complete'])['sha256'],bytes=reader.bind(record['prepared_complete'])['bytes']),
        layered_runtime_equivalence=record['layered_runtime_equivalence'],scene=result['dependency_inventory']['primary'],
        scene_base=result['dependency_inventory']['base'],scene_authoring_receipt=result['scene_authoring_receipt'])
    need(_same(after,expected), 'Actual current runtime source inventory differs from exact author outputs')
    historical_sources=reader.read(record['layered_runtime_equivalence'])['historical_prepared_sources']
    need(_same(historical_sources,{k:v for k,v in after.items() if k in {'preparation_config',*scene_store.FILES,*scene_store.OPTIONAL}}),
        'Sidecar must inventory every actual historical prepared FILES role')
    receipt=reader.read(entry['builder_receipt']);runtimeprep=receipt['runtime_preparation']
    _current_builder(reader,provenance,scene,entry,baseline,collector_root,current=False)
    _current_builder(reader,receipt,scene,entry,config,collector_root,current=True)
    need(_reference(runtimeprep)==reader.bind(result['preparation_config']) and
        _same(reader.read(runtimeprep),reader.read(result['preparation_config'])), 'Runtime must bind exact factored preparation at 2400 seconds')
    refs=[]
    for value,patient,planref in ((baseline,scene['original']['patient_receipt'],entry['original']),
                                 (config,scene['derived']['patient_receipt'],entry['derived'])):
        meta=value['m2_given_contact_control']['metadata'];ref=runtime.pose.require_route(meta)
        need(all(meta[k]==allocation[k] for k in ('subject_id','split')) and meta['target_id']==allocation['target'] and
            ref.snapshot()['sources']==dict(task_manifest=_file(patient),patient_receipt=_file(patient),fresh_plan=_file(planref)),
            'Exact fixed current runtime identity/reference roles required');refs.append(ref)
    before_ref,after_ref=[r.snapshot() for r in refs];after_sources=deepcopy(after_ref['sources']);after_ref['sources']=before_ref['sources']
    binding=dict(original_reference_sha256=refs[0].sha256,derived_reference_sha256=refs[1].sha256,
        original_sources=before_ref['sources'],derived_sources=after_sources)
    need(_same(before_ref,after_ref) and _same(entry['reference_binding'],binding), 'Numerical pose/availability or reference lineage changed')
    a=baseline['compact_scene_storage'];b=config['compact_scene_storage'];new_roles={'scene_base','scene_authoring_receipt','layered_runtime_equivalence'}
    need(a['schema']==scene_store.SDK_SCHEMA and a['dependency_policy']==scene_store.SDK_DEPENDENCY_POLICY and
        b['schema']==scene_store.SDK_LAYER_SCHEMA and b['dependency_policy']==scene_store.SDK_LAYER_CURRENT_RAW_POLICY and
        all(a[k]==b[k] for k in ('store_root','batch_id')) and not (set(before)&new_roles) and set(after)==set(before)|new_roles and
        _same(baseline['physics'],dict(position_iterations=32,velocity_iterations=1,armature_kg_m2=.02)) and _same(config['physics'],layers.ITERATIONS),
        'Exact dense32-to-layered64 storage transition required')
    reverted=deepcopy(config);compact=reverted['fresh_expert']['raw_multimodal_capture']['compact']
    need('numeric_container_codec' not in baseline['fresh_expert']['raw_multimodal_capture']['compact'] and
        compact.pop('numeric_container_codec')==NUMERIC_CODEC, 'Explicit reviewed lossless codec transition required')
    objects=[];strings={}
    for original,derived in [(before[k],after[k]) for k in before]+[(binding['original_sources'][k],binding['derived_sources'][k]) for k in binding['original_sources']]:
        objects.append((derived,original))
        for key in ('file','sha256'):
            need(derived[key] not in strings or strings[derived[key]]==original[key], 'Ambiguous runtime source alias substitution')
            strings[derived[key]]=original[key]
    strings[refs[1].sha256]=refs[0].sha256;strings['given-contact-'+refs[1].sha256]='given-contact-'+refs[0].sha256
    object_replacements={json.dumps(derived,sort_keys=True,allow_nan=False):original for derived,original in objects}
    def revert(value):
        if isinstance(value,dict):
            if set(value) in ({'file','sha256'},{'file','bytes','sha256'}):
                original=object_replacements.get(json.dumps(value,sort_keys=True,allow_nan=False))
                if original is not None:return deepcopy(original)
            return {k:revert(v) for k,v in value.items()}
        if isinstance(value,list):return [revert(v) for v in value]
        return strings.get(value,value) if isinstance(value,str) else value
    reverted['physics']=deepcopy(baseline['physics']);storage=reverted['compact_scene_storage']
    for key in new_roles:storage['sources'].pop(key)
    storage['schema']=a['schema'];storage['dependency_policy']=a['dependency_policy']
    comparison=deepcopy(baseline)
    if capture_paths is not None:
        need(set(capture_paths)=={'camera','calibration'}, 'Only two named checkpoint capture path substitutions permitted')
        for name,pair in capture_paths.items():
            a=comparison['fresh_expert']['raw_multimodal_capture'];b=reverted['fresh_expert']['raw_multimodal_capture']
            need(a[name+'_file']==pair['default']['path'] and b[name+'_file']==pair['qualified']['path'] and
                 a[name+'_sha256']==b[name+'_sha256']==pair['default']['sha256']==pair['qualified']['sha256'] and
                 pair['default']['bytes']==pair['qualified']['bytes'], 'Checkpoint capture substitution differs from exact roles')
            a[name+'_file']=b[name+'_file']
    need(_same(revert(reverted),comparison), 'Unlisted current runtime control/seed/time/geometry/camera change')
    return baseline,config


def _current_conditions(config):
    """Reconstruct the producer's complete approved collector-condition record."""
    need(config.get('experiment')=='geometry_qualification' and 'm2_act_deployment' not in config and
        config.get('prepare_only') is not True and _same(config.get('physics'),layers.ITERATIONS),
        'Only the approved live given expert collector conditions are supported')
    fresh=deepcopy(config['fresh_expert']);capture=fresh.pop('raw_multimodal_capture');fresh.pop('plans',None)
    need(fresh.get('collision_profile')=='full' and capture.get('enabled') is True and capture.get('profile')=='compact',
        'Full collision and enabled compact native capture required')
    compact=deepcopy(capture['compact'])
    need(_same(compact.get('native_image_size'),[256,144]) and type(compact.get('max_frames')) is int and
        compact['max_frames']==5000 and type(compact.get('limit_bytes')) is int and compact['limit_bytes']==100000000 and
        compact.get('numeric_container_codec')==NUMERIC_CODEC, 'Exact approved image/frame/byte/codec conditions required')
    for key in ('metadata','estimated_target_base','target_estimate_source_id'):compact.pop(key,None)
    compact['code_bindings'].pop('body',None)
    capture['compact']=compact;capture.pop('episode_id',None)
    store=config['compact_scene_storage'];_keys(store,'schema dependency_policy store_root batch_id sources')
    need(store['schema']==scene_store.SDK_LAYER_SCHEMA and store['dependency_policy']==scene_store.SDK_LAYER_CURRENT_RAW_POLICY,
        'Exact current RAW collector storage conditions required')
    sdk={key:ref for key,ref in store['sources'].items() if key=='sdk_mdl_manifest' or scene_store._sdk_file_name(key)}
    return dict(top_keys=sorted(config),physics=deepcopy(config['physics']),fresh=fresh,capture=capture,
        isaac_python=config.get('isaac_python'),scene_dependencies=dict(schema=store['schema'],dependency_policy=store['dependency_policy'],sources=sdk))


def _current_code_manifest(reader,collector):
    """Authenticate code headers once; code files themselves are bind-only."""
    source=reader.bind(collector['source_manifest']);cache=getattr(reader,'code_manifests',{})
    if source['path'] in cache:return cache[source['path']]
    root=Path(collector['root']);manifest=reader.read(source)
    need(isinstance(manifest,list) and bool(manifest), 'Complete frozen collector source manifest required')
    result={}
    for row in manifest:
        need(isinstance(row,dict) and set(row) in ({'file','sha256'},{'file','sha256','bytes'}),
            'Exact frozen collector source header fields required')
        relative=row['file']
        need(isinstance(relative,str) and relative and not Path(relative).is_absolute() and
            str(Path(relative))==relative and '..' not in Path(relative).parts,
            'Canonical collector-relative source name required')
        ref=reader.bind(dict(path=str(root/relative),**{key:row[key] for key in ('sha256','bytes') if key in row}))
        need(ref['path'] not in result, 'Unique frozen collector source name required')
        result[ref['path']]=ref
    cache[source['path']]=result;reader.code_manifests=cache
    return result


def _current_inventory(reader,attempt,entry,config,collector,root):
    """Normalize every authenticated source-freeze header without payload reads."""
    freeze=reader.read(attempt['source_freeze'])
    _keys(freeze,'schema code_commit code_commit_file source_manifest files')
    need(freeze['schema']==runtime.dataset.SOURCE_FREEZE_SCHEMA and
        isinstance(freeze['code_commit'],str) and re.fullmatch('[0-9a-f]{40}',freeze['code_commit']) and
        freeze['code_commit']==collector['commit'] and _same(freeze['code_commit_file'],collector['code_commit_file']) and
        _same(freeze['source_manifest'],collector['source_manifest']), 'Exact per-attempt frozen collector source headers required')
    reader.bind(freeze['code_commit_file']);reader.bind(freeze['source_manifest'])
    manifest=_current_code_manifest(reader,collector);code=Path(collector['root'])
    meta=config['m2_given_contact_control']['metadata'];roles=meta['given_reference']['sources']
    _keys(roles,'task_manifest patient_receipt fresh_plan')
    capture=config['fresh_expert']['raw_multimodal_capture']
    fixed={name:dict(file=capture[name+'_file'],sha256=capture[name+'_sha256']) for name in ('camera','calibration')}
    fixed['native_qualification']=meta['native_retainer_audit']['qualification']
    fixed.update({name:_file(ref) for name,ref in config['compact_scene_storage']['sources'].items()
        if name=='sdk_mdl_manifest' or scene_store._sdk_file_name(name) or name=='layered_runtime_equivalence'})
    given_paths={};fixed_paths={}
    for name,ref in roles.items():
        bound=reader.bind(ref);given_paths.setdefault(bound['path'],[]).append(name)
    for name,ref in fixed.items():
        bound=reader.bind(ref);fixed_paths.setdefault((bound['path'],bound['sha256']),[]).append(name)
    files=freeze['files'];need(isinstance(files,dict) and bool(files), 'Complete frozen original/archive source union required')
    result={};covered=set();fixed_covered=set()
    for original,archive in files.items():
        _keys(archive,'path bytes sha256')
        archived=reader.bind(archive);bound=reader.bind(dict(path=original,sha256=archived['sha256'],bytes=archived['bytes']))
        need(archived['path']==str(root/'archive'/archived['sha256']), 'Exact immutable source archive role required')
        fixed_names=fixed_paths.get((original,bound['sha256']),[]);fixed_covered.update(fixed_names)
        path=Path(original)
        if code in path.parents:
            need(original in manifest and _same(bound,manifest[original]), 'Frozen runtime code differs from the collector source manifest')
            relative=path.relative_to(code).as_posix();key='code:'+relative
            item=dict(kind='code',relative_path=relative,sha256=bound['sha256'])
        else:
            names=sorted(given_paths.get(original,[]))
            if names:
                need(all(roles[name]['sha256']==bound['sha256'] for name in names),
                    'Given source union digest or alias relation changed')
                covered.update(names);key='given:'+','.join(names);item=dict(kind='given',roles=names)
            else:
                names=sorted(fixed_names)
                need(bool(names), 'Unknown non-code source in the frozen runtime union')
                key='fixed:'+','.join(names)
                item=(dict(kind='prepared',roles=names) if names==['layered_runtime_equivalence'] else
                    dict(kind='fixed',roles=names,reference=deepcopy(bound)))
        need(key not in result, 'Duplicate normalized frozen source identity')
        result[key]=item
    need(covered==set(roles), 'Complete exact given-source role coverage required')
    need(fixed_covered==set(fixed), 'Complete exact fixed/capture source role coverage required')
    bindings=reader.read(entry['builder_receipt'])['source_bindings'];codes=capture['compact']['code_bindings']
    need(set(codes)==set(bindings) and all(codes[name]==ref['sha256'] for name,ref in bindings.items()),
        'Capture code bindings differ from the authenticated builder source headers')
    need(all(_reference(ref)['path'] in files and code in Path(_reference(ref)['path']).parents
        for name,ref in bindings.items() if name!='body') and
        'code:sim/ausc_policy/m2_given_contact_handoff.py' in result,
        'Complete authenticated capture code source union and interior handoff helper required')
    need(result.get('fixed:layered_runtime_equivalence')==dict(kind='prepared',roles=['layered_runtime_equivalence']),
        'Frozen runtime union must declare the current RAW prepared source role')
    return result


def _current_view(mapping,plan):
    selected,allocated,_=_population(mapping);reader=_CurrentMetadata(plan)
    need(all(type(mapping['summaries'][split]['seed']) is int for split in ('train','validation')) and
        all(type(a['seed']) is int for a in allocated), 'Typed fixed allocation seeds required')
    report=reader.read(plan['derivation']);recipe=reader.read(plan['source_recipe']);root=Path(_reference(plan['derivation'])['path']).parent
    checkpoint=None;source_root=root
    if report.get('completed_authoring_checkpoint') is not None or plan.get('completed_authoring_checkpoint') is not None:
        from scripts.experiments.wam_runpod_20261027 import completed_authoring_checkpoint_v1 as recovery
        binding=report.get('completed_authoring_checkpoint')
        need(_same(binding,plan.get('completed_authoring_checkpoint')), 'Derivation/plan checkpoint capabilities differ')
        checkpoint=reader.read(binding)
        recovery.validate_headers(checkpoint,selected,allocated,root,bind=reader.bind)
        recovery.validate_plan_headers(checkpoint,report.get('plans'),root,bind=reader.bind)
        historical,normalized=_current_headers(report,recipe,selected,allocated,plan,reader,root,checkpoint=checkpoint)
        recovery.validate_copied_plans(checkpoint,report['plans'],read_json=reader.read)
        recovered=recovery.reconstruct(checkpoint,selected,allocated,root,bind=reader.bind,read_json=reader.read,payloads=False)
        need(_same(report['layered_authoring'],recovered['authored']) and _same(report['cpu_execution'],recovered['cpu_execution']) and
             _same(report['current_raw_prepared'],recovered['current_raw_prepared']) and
             _same(report['authoring_request'],recovered['authoring_request']) and
             _same(report['runtime_equivalence_receipt'],recovered['runtime_equivalence_receipt']) and
             _same(plan['source_recipe'],checkpoint['bindings']['original_recipe']), 'Historical completed-authoring metadata changed')
        source_root=Path(checkpoint['source_root'])
        need(plan['owner_executable']['invoked_path']=='/opt/wam-runtime/envs/isaacsim51/bin/python' and
             plan['config_executable']['invoked_path']=='/opt/wam-runtime/envs/isaacsim60/bin/python', 'Checkpoint Native51 owner/Native60 CPU routing changed')
    else:historical,normalized=_current_headers(report,recipe,selected,allocated,plan,reader,root)
    cpu=report['cpu_execution'];_keys(cpu,'executable collector_root collector provider script_sha256 request_sha256 '
        'current_raw_equivalence dependency_policy gpu_work physics_advanced')
    need(cpu['current_raw_equivalence'] is True and cpu['dependency_policy']==scene_store.SDK_LAYER_CURRENT_RAW_POLICY and
        cpu['collector']==plan['collector'] and cpu['collector_root']==plan['collector']['root'] and
        _reference(cpu['provider'])['path']==str(Path(cpu['collector_root'])/'sim/ausc_policy/layered_scene_authoring.py') and
        cpu['request_sha256']==runtime.digest(report['authoring_request']) and re.fullmatch('[0-9a-f]{64}',cpu['script_sha256']) and
        cpu['gpu_work'] is False and cpu['physics_advanced'] is False, 'Exact current RAW CPU request/provider capability required')
    _execution_sources(report,recipe,reader)
    _current_authored(report,historical,normalized,reader,source_root)
    scenes={s['proposal_id']:s for s in report['scenes']};first={}
    for entry,allocation,attempt in zip(report['plans'],allocated,plan['attempts']):
        baseline,config=_current_runtime(reader,entry,scenes[allocation['proposal_id']],allocation,cpu['collector_root'],
            **({} if checkpoint is None else dict(capture_paths=checkpoint['capture_paths'])))
        if checkpoint is not None:
            builder=reader.read(entry['builder_receipt'])['collector_cpu_builder'];prior=reader.read(entry['baseline_provenance'])['collector_cpu_builder']
            need(config.get('isaac_python')==plan['owner_executable']['invoked_path'] and
                 _same(builder['executable'],plan['config_executable']) and builder['script_sha256']==prior['script_sha256'] and
                 prior['executable']['invoked_path']==plan['owner_executable']['invoked_path'],
                 'Native owner or actual CPU builder provenance changed')
        first.setdefault(allocation['proposal_id'],(baseline,config))
        need(_same(recipe['collector_conditions'],_current_conditions(config)), 'Complete actual collector conditions differ from the approved recipe')
        need(_same(recipe['inventory'],_current_inventory(reader,attempt,entry,config,plan['collector'],root)),
            'Complete normalized runtime source inventory differs from the approved recipe')
    for scene,proof in zip(report['scenes'],report['layered_authoring']['runtime_equivalence']):
        baseline,_=first[scene['proposal_id']]
        sources=proof['derived_sources'];runtime_sources=first[scene['proposal_id']][1]['compact_scene_storage']['sources']
        need(_same(proof['original_sources'],baseline['compact_scene_storage']['sources']) and _same(sources,runtime_sources),
            'Offline SDK proof omitted or substituted a current RAW source role')
    view=deepcopy(mapping)
    for i,scene in enumerate(view['scenes']):
        if scene['proposal_id'] in scenes:view['scenes'][i]=deepcopy(scenes[scene['proposal_id']]['derived'])
    for entry,a in zip(report['plans'],[a for s in ('train','validation') for a in view['ordered_allocations'][s]]):a['fresh_plan']=deepcopy(entry['derived'])
    return view
