"""Versioned compact initial-scene CAS. No simulator, training or data admission.

The CAS is an explicitly charged shared input, never a writable runtime layer.
Legacy jobs do not enter this module unless resolving a legacy replay pathname.
"""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from .storage_policy import EPISODE_LIMIT_BYTES, episode_size

SCHEMA='m2_compact_scene_storage_v1'
SDK_SCHEMA='m2_compact_scene_storage_v2'
SDK_LAYER_SCHEMA='m2_compact_scene_storage_v3'
REF_SCHEMA='m2_compact_scene_ref_v1'
HELDOUT_REF_SCHEMA='m2_given_contact_sealed_test_scene_ref_v1'
LEDGER_SCHEMA='m2_compact_scene_byte_ledger_v1'
DEPENDENCY_POLICY='builtin_omnipbr_only_v1'
SDK_DEPENDENCY_POLICY='hash_bound_isaacsim60_mdl_v2'
SDK_LAYER_DEPENDENCY_POLICY='hash_bound_static_usd_and_isaacsim60_mdl_v3'
SDK_LAYER_CURRENT_RAW_POLICY='hash_bound_static_usd_current_raw_and_sdk_v4'
SDK_SELECTION_KEY='wamSdkMdlSelectionJson'
_RUNTIME_STAGE_LEASES={}
DEFAULT_ROOT='/workspace/wam/scene_store'
FILES=dict(scene='candidate_physics.usdc',patient_receipt='patient_receipt.json',
    collision_geometry='geometry/collision_geometry.json',collision_arrays='geometry/collision_geometry.npz',
    mass_estimates='mass_estimates.json',physics_profile='physics_profile.json',
    approved_camera='approved_camera_restoration.json')
OPTIONAL_RUNTIME_SOURCES={'native_retainer_independent_analysis'}
RUNTIME_SOURCES={'native_retainer_qualification',*(
    'native_retainer_evidence_'+name for name in ('owner_terminal','owner_exit','complete','cases','native','residuals'))}
OPTIONAL=dict(preload_authoring='passive_preload_authoring.json',
    preload_readback='passive_preload_saved_usd_readback.json',stool='stool_layout_receipt.json')


def need(ok,message):
    if not ok:raise ValueError(message)


def real_path(value, *, exists=True):
    need(isinstance(value,(str,Path)),'A real absolute path is required')
    path=Path(value)
    need(path.is_absolute() and '..' not in path.parts,'Absolute path without traversal required')
    need(not any(p.is_symlink() for p in (path,*path.parents)),'Symlink scene aliases are forbidden')
    if exists:need(path.exists(),'Missing scene storage source: '+str(path))
    return path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
    return digest.hexdigest()


def source_ref(path):
    path=real_path(path)
    need(path.is_file(),'Regular source file required')
    return dict(file=str(path),sha256=sha(path),bytes=path.stat().st_size)


def _source_shape(ref,*,allow_empty=False):
    need(isinstance(ref,dict) and set(ref)=={'file','sha256','bytes'} and
         isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None and
         type(ref['bytes']) is int and ref['bytes']>=(0 if allow_empty else 1),'Exact file/bytes/SHA256 source binding required')
    real_path(ref['file'],exists=False)


def read_source(ref,*,allow_empty=False):
    _source_shape(ref,allow_empty=allow_empty)
    path=real_path(ref['file']);need(path.is_file(),'Regular source required')
    data=path.read_bytes()
    need(len(data)==ref['bytes'] and hashlib.sha256(data).hexdigest()==ref['sha256'],
         'Prepared source hash/bytes changed: '+str(path))
    return data


def _json(data):return json.loads(data)


def _strict_json(data):
    """Provenance JSON has no duplicate keys or nonfinite values."""
    def pairs(rows):
        value={}
        for key,item in rows:
            need(key not in value,'Duplicate provenance JSON field: '+key)
            value[key]=item
        return value
    def constant(value):raise ValueError('Nonfinite provenance JSON value: '+value)
    value=json.loads(data,object_pairs_hook=pairs,parse_constant=constant)
    json.dumps(value,allow_nan=False)
    return value


def _exact_json(left,right):
    return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)


def _layered_policy(complete):
    policy=complete.get('scene_dependency_policy',SDK_LAYER_DEPENDENCY_POLICY)
    need(policy in (SDK_LAYER_DEPENDENCY_POLICY,SDK_LAYER_CURRENT_RAW_POLICY),
         'Unknown layered scene dependency policy')
    need(('layered_runtime_equivalence' in complete)==(policy==SDK_LAYER_CURRENT_RAW_POLICY),
         'Layered runtime equivalence role requires its explicit v4 policy')
    return policy


def _prepared_files_root(complete,default):
    if _layered_policy(complete)!=SDK_LAYER_CURRENT_RAW_POLICY:return default
    # _layered_metadata has already authenticated this exact historical COMPLETE.
    record=_strict_json(read_source(complete['layered_runtime_equivalence']))
    return Path(record['historical_prepared_complete']['file']).parent


def _write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')


def build_binding(*,preparation_config,prepared_complete,patient_receipt,
                  store_root=DEFAULT_ROOT,batch_id,retainer_witness_contract=None,sdk_mdl_manifest=None,formal_preparation=None):
    """Build intent only. Caller authenticates identity before reading payloads."""
    if formal_preparation is not None:
        from .m2_given_contact_evaluation import authorize_prepared
        authorize_prepared(formal_preparation,preparation_config,prepared_complete,patient_receipt)
    sources={}
    for name,ref in [('preparation_config',preparation_config),('prepared_complete',prepared_complete),
                     ('patient_receipt',patient_receipt)]:
        need(isinstance(ref,dict) and set(ref)=={'file','sha256'},'Expected file/SHA256 reference')
        current=source_ref(ref['file']);need(current['sha256']==ref['sha256'],'Prepared metadata hash changed')
        sources[name]=current
    root=Path(sources['prepared_complete']['file']).parent
    complete=_json(read_source(sources['prepared_complete']))
    if 'scene_dependency_inventory' in complete:
        need(sdk_mdl_manifest is not None,'Layered scene requires exact approved SDK MDL manifest')
        _layered_metadata(complete)  # All original identity headers precede any scene/body payload.
        root=_prepared_files_root(complete,root)
    for name,filename in FILES.items():
        value=source_ref(root/filename)
        if name in sources:need(value==sources[name],'Patient receipt must belong to prepared scene')
        sources[name]=value
    for name,filename in OPTIONAL.items():
        if (root/filename).exists():sources[name]=source_ref(root/filename)
    if retainer_witness_contract is not None:
        from .retainer_witness import load_contract
        qualification=load_contract(retainer_witness_contract)
        sources['native_retainer_qualification']=source_ref(retainer_witness_contract['qualification']['file'])
        for key,ref in qualification['evidence'].items():
            sources['native_retainer_evidence_'+key]=source_ref(ref['file'])
        if 'independent_analysis' in qualification:
            sources['native_retainer_independent_analysis']=source_ref(qualification['independent_analysis']['file'])
    schema,policy=SCHEMA,DEPENDENCY_POLICY
    if sdk_mdl_manifest is not None:
        from .sdk_mdl import validate_manifest
        bundle=validate_manifest(sdk_mdl_manifest)
        sources.update(_sdk_sources(bundle));schema,policy=SDK_SCHEMA,SDK_DEPENDENCY_POLICY
    if 'scene_dependency_inventory' in complete:
        inventory=validate_prepared_dependency_inventory(complete)
        need(inventory['primary']==sources['scene'],'Layered primary differs from prepared source')
        sources['scene_base']=deepcopy(inventory['base'])
        sources['scene_authoring_receipt']=deepcopy(complete.get('scene_authoring_receipt'))
        schema,policy=SDK_LAYER_SCHEMA,_layered_policy(complete)
        if policy==SDK_LAYER_CURRENT_RAW_POLICY:
            sources['layered_runtime_equivalence']=deepcopy(complete['layered_runtime_equivalence'])
    binding=dict(schema=schema,store_root=str(real_path(store_root,exists=False)),batch_id=batch_id,
                 dependency_policy=policy,sources=sources)
    _validate_binding(binding,**({'formal_preparation':formal_preparation} if formal_preparation is not None else {}))
    return binding


def _binding_shape(binding):
    need(isinstance(binding,dict) and set(binding)=={'schema','store_root','batch_id','dependency_policy','sources'} and
         (binding['schema'],binding['dependency_policy']) in ((SCHEMA,DEPENDENCY_POLICY),(SDK_SCHEMA,SDK_DEPENDENCY_POLICY),
             (SDK_LAYER_SCHEMA,SDK_LAYER_DEPENDENCY_POLICY),(SDK_LAYER_SCHEMA,SDK_LAYER_CURRENT_RAW_POLICY)),
         'Explicit versioned compact scene storage dependency policy required')
    real_path(binding['store_root'],exists=False)
    need(isinstance(binding['batch_id'],str) and
         re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',binding['batch_id']) is not None,'Invalid frozen batch identity')
    sources=binding['sources'];required={'preparation_config','prepared_complete',*FILES}
    if binding['schema'] in (SDK_SCHEMA,SDK_LAYER_SCHEMA):
        from .sdk_mdl import APPROVED_FILE_COUNT
        required|={'sdk_mdl_manifest',*('sdk_mdl_'+str(i).zfill(4) for i in range(APPROVED_FILE_COUNT))}
    if binding['schema']==SDK_LAYER_SCHEMA:required|={'scene_base','scene_authoring_receipt'}
    if binding['dependency_policy']==SDK_LAYER_CURRENT_RAW_POLICY:required|={'layered_runtime_equivalence'}
    need(isinstance(sources,dict) and required<=set(sources)<=required|set(OPTIONAL)|RUNTIME_SOURCES|OPTIONAL_RUNTIME_SOURCES,'Invalid prepared source inventory')
    for name,ref in sources.items():_source_shape(ref,allow_empty=_sdk_file_name(name))


def _sdk_file_name(name):return re.fullmatch('sdk_mdl_[0-9]{4}',name) is not None


def _sdk_sources(bundle):
    return dict(sdk_mdl_manifest=bundle['manifest'],**{
        'sdk_mdl_'+str(i).zfill(4):ref for i,ref in enumerate(bundle['files'])})


def _sdk_bundle(binding):
    if binding['schema'] not in (SDK_SCHEMA,SDK_LAYER_SCHEMA):return None
    from .sdk_mdl import validate_manifest
    bundle=validate_manifest(binding['sources']['sdk_mdl_manifest'])
    need(all(binding['sources'][name]==ref for name,ref in _sdk_sources(bundle).items()),
         'SDK MDL bound input inventory differs from exact manifest')
    return bundle


def _binding_metadata(binding,*,authorization=None,formal_preparation=None):
    if authorization is not None:
        from .m2_given_contact_evaluation import verify_scene_authorization
        verify_scene_authorization(binding,authorization)
    elif formal_preparation is not None:
        from .m2_given_contact_evaluation import verify_prepared_binding
        verify_prepared_binding(binding,formal_preparation)
    _binding_shape(binding);sources=binding['sources']
    root=Path(sources['prepared_complete']['file']).parent
    if binding['dependency_policy']==SDK_LAYER_CURRENT_RAW_POLICY:
        complete=_strict_json(read_source(sources['prepared_complete']))
        _bound_layered_metadata(binding,complete)
        root=_prepared_files_root(complete,root)
    need(Path(sources['prepared_complete']['file']).name=='COMPLETE.json' and
         Path(sources['patient_receipt']['file'])==root/'patient_receipt.json', 'Exact prepared JSON metadata roles required')
    # Identity metadata is checked before scene/collision payload bytes.
    parse=_strict_json if binding['dependency_policy']==SDK_LAYER_CURRENT_RAW_POLICY else _json
    prep=parse(read_source(sources['preparation_config']))
    complete=parse(read_source(sources['prepared_complete']))
    patient=parse(read_source(sources['patient_receipt']))
    need(prep.get('prepare_only') is True and complete.get('completed') is True and
         (patient.get('split') in ('train','validation') if authorization is None and formal_preparation is None else patient.get('split')=='test') and complete.get('patient')==patient and
         patient.get('non_patient_scene_unchanged') is True,'Authenticated TRAIN/VALIDATION prepared patient required')
    return prep,complete,patient


def _validate_binding(binding,*,authorization=None,formal_preparation=None):
    prep,complete,patient=_binding_metadata(binding,authorization=authorization,formal_preparation=formal_preparation)
    sources=binding['sources']
    root=Path(sources['prepared_complete']['file']).parent
    if binding['dependency_policy']==SDK_LAYER_CURRENT_RAW_POLICY:root=_prepared_files_root(complete,root)
    for name,filename in {**FILES,**OPTIONAL}.items():
        if name in sources:need(Path(sources[name]['file'])==root/filename,'Prepared source path substitution')
    expected=complete.get('geometry_artifacts',{})
    need(sources['scene']['sha256']==complete.get('effective_scene_sha256') and
         sources['collision_geometry']['sha256']==expected.get('collision_geometry_sha256') and
         sources['collision_arrays']['sha256']==expected.get('collision_arrays_sha256'),
         'Prepared scene/geometry exact hashes differ from COMPLETE')
    runtime=set(sources)&(RUNTIME_SOURCES|OPTIONAL_RUNTIME_SOURCES)
    if runtime:
        from .retainer_witness import SCHEMA as AUDIT_SCHEMA,load_contract
        need(RUNTIME_SOURCES<=runtime,'Incomplete native retainer qualification input inventory')
        ref=sources['native_retainer_qualification']
        receipt=load_contract(dict(schema=AUDIT_SCHEMA,qualification={k:ref[k] for k in ('file','sha256')}))
        analysis=receipt.get('independent_analysis')
        stored_analysis=sources.get('native_retainer_independent_analysis')
        need((analysis is None and stored_analysis is None) or
             (analysis is not None and stored_analysis is not None and
              all(analysis[k]==stored_analysis[k] for k in ('file','sha256'))),
             'Independent native retainer analysis input substitution')
        for key,value in receipt['evidence'].items():
            need(all(sources['native_retainer_evidence_'+key][k]==value[k] for k in ('file','sha256')),
                 'Native retainer evidence source substitution')
    _sdk_bundle(binding)
    if binding['schema']==SDK_LAYER_SCHEMA:_layered_binding(binding,complete)
    else:need('scene_dependency_inventory' not in complete,'Layered prepared scene needs explicit layered schema')
    for name,ref in sources.items():read_source(ref,allow_empty=_sdk_file_name(name))
    return prep,complete,patient


def _validate_given_teacher_route(config):
    """Only this exact bound teacher intent is independent of USD authoring."""
    route=config.get('m2_given_contact_control')
    fresh=config.get('fresh_expert',{})
    raw=fresh.get('raw_multimodal_capture',{})
    compact=raw.get('compact',{}) if isinstance(raw,dict) else {}
    captured_metadata=compact.get('metadata',{}) if isinstance(compact,dict) else {}
    given_capture=isinstance(captured_metadata,dict) and captured_metadata.get('target_source')=='given_contact_pose'
    if route is None:
        need(not given_capture,'Given-contact capture requires its explicit teacher route')
        return False
    from .m2_given_contact_pose import TASK_ID,require_route
    from .m2_contract import require_observation_contract
    from .expert_capture import capture_settings
    need(isinstance(route,dict) and set(route) in ({'schema','metadata','episode_id'},
         {'schema','metadata','episode_id','handoff_policy_id'}) and
         route['schema']==TASK_ID and isinstance(route['metadata'],dict) and 'm2_act_deployment' not in config,
         'Exact given-contact teacher route required before stripping runtime intent')
    from .m2_given_contact_handoff import POLICY_ID
    need(route.get('handoff_policy_id')==route['metadata'].get('handoff_policy_id') and
         route.get('handoff_policy_id') in (None,POLICY_ID), 'Given runtime/capture handoff policy differs')
    episode=route['episode_id']
    need(isinstance(episode,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',episode) is not None,
         'Invalid given-contact teacher episode')
    metadata=route['metadata'];require_observation_contract(metadata)
    reference=require_route(metadata);reference.verify_sources()
    capture=capture_settings(fresh)
    need(capture is not None and capture.get('profile')=='compact' and given_capture and
         capture['episode_id']==episode==metadata.get('episode_id') and
         capture['compact']['metadata']==metadata,
         'Given-contact route/capture metadata or episode differs')
    sources=reference.snapshot()['sources']
    need(fresh.get('plans')==[sources['fresh_plan']], 'Given-contact route/fresh plan binding differs')
    binding=config.get('compact_scene_storage',{})
    bound=binding.get('sources',{})
    patient_ref=bound.get('patient_receipt',{})
    need(all(patient_ref.get(k)==sources['patient_receipt'][k] for k in ('file','sha256')),
         'Given-contact route/prepared patient binding differs')
    patient=_json(read_source(patient_ref))
    complete=_json(read_source(bound.get('prepared_complete',{})))
    need(metadata.get('subject_id')==patient.get('identity') and metadata.get('split')==patient.get('split') and
         metadata.get('patient_asset_sha256')==patient.get('asset_sha256') and
         metadata.get('installed_patient_receipt_sha256')==patient_ref['sha256'] and
         metadata.get('identity_split_sha256')==patient.get('identity_split_sha256') and
         all(metadata.get(k)==v for k,v in complete.get('geometry_artifacts',{}).items()),
         'Given-contact route/prepared patient or geometry metadata differs')
    need(reference.snapshot()['world_from_base']==complete.get('world_from_left_base'),
         'Given-contact route/prepared robot base differs')
    return True


def _authoring(config):
    result=deepcopy(config)
    if 'matched_scene_eval' in result:
        from .m2_rebuild_paired_eval import preflight_arm,ROUTE_KEY,DESCRIPTOR
        preflight_arm(result)
        for key in (ROUTE_KEY,DESCRIPTOR,'m2_given_contact_control','astra_supervisor'):result.pop(key,None)
    if 'm2_given_contact_expanded_val_scenarios' in result:
        from .m2_given_contact_expanded_val_scenarios import preflight
        preflight(result)
        result.pop('m2_given_contact_expanded_val_scenarios')
    if 'm2_given_contact_recovery' in result:
        from .m2_given_contact_recovery import preflight
        preflight(result)
        result.pop('m2_given_contact_recovery')
    if 'm2_given_contact_expanded_pilot_deployment' in result:
        from .m2_given_contact_expanded_pilot_deployment import preflight
        preflight(result)
        result.pop('m2_given_contact_expanded_pilot_deployment')
    if 'm2_given_contact_evaluation' in result:
        from .m2_given_contact_evaluation import preflight
        preflight(result)
        result.pop('m2_given_contact_evaluation')
        result.pop('m2_given_contact_control',None)
        result.pop('m2_given_contact_deployment',None)
    elif 'm2_given_contact_deployment' in result:
        from .m2_given_contact_deployment import preflight
        preflight(result)  # Authenticate the exact formal permit before stripping runtime intent.
        result.pop('m2_given_contact_deployment')
    elif 'm2_given_contact_pilot_deployment' in result:
        from .m2_given_contact_pilot_deployment import preflight
        preflight(result)  # Authenticate diagnostic intent; original USD authoring comparison stays exact.
        result.pop('m2_given_contact_pilot_deployment')
    if _validate_given_teacher_route(result):
        result.pop('m2_given_contact_control')
    if 'stop_residual_trajectory' in result:
        from .m2_stop_residual_trajectory import CONFIG_KEY,native_options
        # The route preflights above authenticate the original metadata before
        # any runtime intent is removed. Never synthesize a handoff identity.
        routes=[config[key] for key in ('m2_given_contact_control',
            'm2_given_contact_pilot_deployment','m2_given_contact_expanded_pilot_deployment',
            'm2_given_contact_deployment','m2_given_contact_evaluation',
            'm2_given_contact_expanded_val_scenarios','m2_rebuild_paired_eval') if key in config]
        need(routes and all(isinstance(route,dict) and isinstance(route.get('metadata'),dict)
             for route in routes),'STOP residual authoring projection requires actual route metadata')
        for route in routes:native_options(config,metadata=route['metadata'])
        result.pop(CONFIG_KEY)
    for key in ('prepare_only','targets','raw_target_provenance','compact_scene_storage',
                'patient_preview','m2_sensor_probe','m2_render_settling_probe','contact_probe'):
        result.pop(key,None)
    if 'native_retainer_audit' in result.get('fresh_expert',{}):
        from .retainer_witness import settings
        settings(result['fresh_expert'])  # Validate before stripping this runtime-only key.
        result['fresh_expert'].pop('native_retainer_audit')
    for key in ('plans','raw_multimodal_capture','collision_profile'):
        result.get('fresh_expert',{}).pop(key,None)
    return result


def _object_name(digest):return 'objects/'+digest[:2]+'/'+digest+'.usdc'


def resolve_scene(ref):
    heldout=isinstance(ref,dict) and ref.get('schema')==HELDOUT_REF_SCHEMA
    need(isinstance(ref,dict) and ((set(ref)=={'schema','binding','scene'} and ref['schema']==REF_SCHEMA) or
         (set(ref)=={'schema','binding','scene','authorization'} and heldout)),
         'Explicit versioned scene_ref required')
    binding=ref['binding']
    _,complete,_=_binding_metadata(binding,authorization=ref.get('authorization'))
    if binding['schema']==SDK_LAYER_SCHEMA:_bound_layered_metadata(binding,complete)
    scene=ref['scene']
    need(isinstance(scene,dict) and set(scene)=={'path','sha256','bytes'} and
         isinstance(scene['sha256'],str) and re.fullmatch('[0-9a-f]{64}',scene['sha256']) is not None and
         type(scene['bytes']) is int and scene['bytes']>0 and
         scene['path']==_object_name(scene['sha256']),'Invalid hash-addressed scene path/bytes')
    need(scene['sha256']==binding['sources']['scene']['sha256'] and
         scene['bytes']==binding['sources']['scene']['bytes'],'Scene differs from prepared binding')
    root=real_path(binding['store_root']);need(root.is_dir(),'CAS root must be a directory')
    path=real_path(root/scene['path'])
    need(path.is_file() and stat.S_ISREG(path.stat().st_mode) and path.stat().st_nlink==1 and
         path.stat().st_mode & 0o222==0,'CAS scene must be one immutable regular file')
    need(path.stat().st_size==scene['bytes'] and sha(path)==scene['sha256'],'CAS scene hash/bytes changed')
    if binding['schema']==SDK_LAYER_SCHEMA:_layered_binding(binding)
    return path


def _publish(binding,*,authorization=None):
    if authorization is not None:
        from .m2_given_contact_evaluation import verify_scene_authorization
        verify_scene_authorization(binding,authorization)
    source=binding['sources']['scene'];root=real_path(binding['store_root'],exists=False)
    destination=root/_object_name(source['sha256']);destination.parent.mkdir(parents=True,exist_ok=True)
    real_path(destination.parent)
    ref=dict(schema=REF_SCHEMA,binding=deepcopy(binding),
             scene=dict(path=_object_name(source['sha256']),sha256=source['sha256'],bytes=source['bytes']))
    if authorization is not None:
        ref.update(schema=HELDOUT_REF_SCHEMA,authorization=deepcopy(authorization))
    # Serialize first publication without ever exposing a hardlink alias.
    lock_path=destination.parent/'.publish.lock';real_path(lock_path,exists=False)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not destination.exists():
            fd,name=tempfile.mkstemp(prefix='.publish-'+binding['batch_id']+'-',dir=str(destination.parent))
            try:
                with os.fdopen(fd,'wb') as stream:
                    stream.write(read_source(source));stream.flush();os.fsync(stream.fileno())
                os.chmod(name,0o444)
                os.replace(name,destination)
            finally:Path(name).unlink(missing_ok=True)
    resolve_scene(ref)
    return ref


def validate_config(config):
    """Validate the opt-in before constructing any SimulationApp."""
    if 'matched_scene_eval' in config:
        from .m2_rebuild_paired_eval import preflight_arm
        preflight_arm(config)  # Exact new scenario/model admission before native payload access.
    heldout='m2_given_contact_evaluation' in config;authorization=None
    formal='m2_given_contact_deployment' in config or heldout
    validation_scenario='m2_given_contact_expanded_val_scenarios' in config
    expanded_pilot='m2_given_contact_expanded_pilot_deployment' in config
    rebuilt_paired='m2_rebuild_paired_eval' in config
    pilot='m2_given_contact_pilot_deployment' in config or expanded_pilot or validation_scenario or rebuilt_paired
    if heldout:
        from .m2_given_contact_evaluation import scene_authorization
        authorization=scene_authorization(config)
    elif formal:
        from .m2_given_contact_deployment import preflight
        preflight(config)  # Before _validate_binding/read_source can hash a patient/USD/NPZ payload.
    elif pilot:
        if rebuilt_paired:
            from .m2_rebuild_paired_eval import preflight
        elif validation_scenario:
            from .m2_given_contact_expanded_val_scenarios import preflight
        elif expanded_pilot:
            from .m2_given_contact_expanded_pilot_deployment import preflight
        else:
            from .m2_given_contact_pilot_deployment import preflight
        preflight(config)  # Distinct diagnostic route; retain every original binding/authoring check.
    binding=config.get('compact_scene_storage');prep,complete,_=_validate_binding(binding,**(
        {'authorization':authorization} if heldout else {}))
    capture=config.get('fresh_expert',{}).get('raw_multimodal_capture',{})
    if formal or pilot or 'matched_scene_eval' in config:
        need('raw_multimodal_capture' not in config.get('fresh_expert',{}), 'Learned scene cannot carry expert capture labels')
    else:
        need(capture.get('enabled') is True and capture.get('profile')=='compact' and
             'm2_act_deployment' not in config,'Shared scene store requires explicit compact expert capture')
    from .retainer_witness import settings
    contract=settings(config.get('fresh_expert',{}))
    source=binding['sources'].get('native_retainer_qualification')
    need((contract is None and source is None) or
         (contract is not None and source is not None and
          all(contract['qualification'][k]==source[k] for k in ('file','sha256'))),
         'Runtime native retainer qualification differs from shared binding')
    need(_authoring(config)==_authoring(prep),'Runtime authoring differs from exact prepared config')
    return binding,complete


def prepare_job(config,output):
    """Reuse an authenticated prepared scene without exporting another USD."""
    binding,complete=validate_config(config)
    output=real_path(output,exists=False);output.mkdir(parents=True,exist_ok=True)
    need(not (output/'scene_ref.json').exists(),'Compact job already has scene_ref')
    authorization=None
    if 'm2_given_contact_evaluation' in config:
        from .m2_given_contact_evaluation import scene_authorization
        authorization=scene_authorization(config)
    ref=_publish(binding,**({'authorization':authorization} if authorization is not None else {}))
    for name,filename in {**FILES,**OPTIONAL}.items():
        if name=='scene' or name not in binding['sources']:continue
        path=output/filename;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(read_source(binding['sources'][name]))
    _write(output/'world_from_left_base.json',complete['world_from_left_base'])
    _write(output/'scene_ref.json',ref)
    return ref


def write_scene_ref(directory,ref):
    resolve_scene(ref)
    directory=real_path(directory);need(not (directory/'training_scene.usdc').exists(),'Ambiguous saved scene')
    _write(directory/'scene_ref.json',ref)


def runtime_stage(ref):
    """Read into a new anonymous root; never load CAS as a writable USD layer."""
    source=resolve_scene(ref)
    return _runtime_stage_clone(ref['binding'],source,lambda:resolve_scene(ref))


def prepared_runtime_stage(binding):
    """CPU equivalence clone of authenticated TRAIN/VAL input; writes no CAS.

    This returns an unattached private stage and never starts physics, obtains
    an owner lease, or substitutes for prepare_job/runtime_stage at execution.
    """
    _validate_binding(binding)
    from .layered_scene_authoring import _single_file
    source=_single_file(binding['sources']['scene'],immutable=False)
    def recheck():
        _validate_binding(binding)
        read_source(binding['sources']['scene'])
    return _runtime_stage_clone(binding,source,recheck)


def _runtime_stage_clone_impl(binding,source,recheck):
    from pxr import Ar,Sdf,Usd,UsdUtils
    layered=binding['schema']==SDK_LAYER_SCHEMA
    raw_private=None;private_ids=set();raw_recheck=None
    if layered:
        from .layered_scene_authoring import private_layered_stage,_layered_authored_snapshot,_layer_snapshot,_single_file,digest
        complete=_strict_json(read_source(binding['sources']['prepared_complete']))
        receipt,item,index,metadata=_bound_layered_metadata(binding,complete)
        actual=source_ref(source);_single_file(actual,immutable=True)
        prepared=item['dependency_inventory']['primary']
        need(all(actual[k]==prepared[k] for k in ('bytes','sha256')),
             'Layered actual primary differs from the exact prepared primary bytes/SHA')
        inventory=deepcopy(item['dependency_inventory']);inventory['primary']=actual
        if binding['dependency_policy']==SDK_LAYER_CURRENT_RAW_POLICY:
            expected,raw_recheck=_selected_raw_expected(receipt,item,index,metadata)
            need(_layer_snapshot(expected.GetRootLayer())==_layered_authored_snapshot(inventory,receipt['static_subtrees']),
                 'Layered runtime differs from the full current-context RAW dense reference')
            raw_recheck()
        else:
            # Historical policy deliberately retains its original digest gate.
            need(digest(_layered_authored_snapshot(inventory,receipt['static_subtrees']))==item['authored_semantics_sha256'],
                 'Layered runtime authored semantics changed')
        raw_private=private_layered_stage(inventory)
        root=raw_private.GetRootLayer();private_ids={layer.identifier for layer in raw_private.GetUsedLayers()}
    else:root=Sdf.Layer.OpenAsAnonymous(str(source))
    need(root is not None and root.anonymous,'Cannot create private anonymous scene root')
    # The initial scene must be self-contained. Merely anchoring references or
    # textures would let unbound files change composed geometry under one root
    # SHA. V1 accepts only the production built-in MDL search token.
    need(layered or (not root.subLayerPaths and not root.GetExternalReferences()),
         'Unsupported external USD dependencies in compact initial scene')
    original=binding['sources']['scene'];read_source(original)
    bundle=_sdk_bundle(binding);asset_tokens=[];bare_resolutions={};selected_resolutions={}
    def check_resolution(context):
        token='OmniPBR.mdl';actual=str(Ar.GetResolver().Resolve(token))
        diagnostic=' token='+repr(token)+' actual_resolved='+repr(actual)
        need(not (Path(original['file']).parent/token).exists() and not (Path.cwd()/token).exists(),
             'Unsupported local file dependency shadows built-in MDL search asset;'+diagnostic)
        if bundle is None:
            need(not actual,'Unsupported resolved file dependency for built-in MDL search asset;'+diagnostic)
            return token
        # V2 explicitly selects one authenticated SDK material. The original
        # search token's provider is diagnostic only, never loaded or claimed
        # equivalent. Only the selected absolute entry may reach composition.
        expected=bundle['entry']['file']
        selected=str(Ar.GetResolver().Resolve(expected))
        need(selected==expected,'SDK MDL resolution mismatch; selected_asset='+repr(expected)+
             ' actual_resolved='+repr(selected)+' bare_token_resolved='+repr(actual))
        real_path(selected)
        bare_resolutions[context]=actual;selected_resolutions[context]=selected
        return expected
    def checked_asset(path):
        if layered and path in private_ids:return path
        need(not path or path=='OmniPBR.mdl','Unsupported unbound file dependency: '+path)
        if path:
            asset_tokens.append(path)
            return check_resolution('caller')
        return path
    if layered:
        for layer in raw_private.GetUsedLayers():UsdUtils.ModifyAssetPaths(layer,checked_asset)
    else:UsdUtils.ModifyAssetPaths(root,checked_asset)
    # V2 pins the caller's resolver context on the private stage; an
    # anonymous stage otherwise creates a new context and can lose that search
    # path. The context is still independently checked below, never bypassed.
    stage_options={} if bundle is None else dict(pathResolverContext=Ar.GetResolver().GetCurrentContext())
    composed=Usd.Stage.Open(root,**stage_options)
    need(composed is not None and not composed.GetCompositionErrors(),'Scene clone has unresolved composition dependencies')
    # A stage can have its own resolver context. V1 keeps the bare token closed;
    # V2 verifies that its selected absolute entry remains the same bound file.
    with Ar.ResolverContextBinder(composed.GetPathResolverContext()):
        if asset_tokens:check_resolution('composed_stage')
    # Retain the original final private flatten for both representations.
    # Equivalence compares its actual complete output, without ignoring the
    # documentation, list operations or other metadata USD authors here.
    stage=Usd.Stage.Open(composed.Flatten(),**stage_options)
    need(stage is not None and all(layer.anonymous for layer in stage.GetUsedLayers()),
         'Private runtime stage must have no writable external USD layers')
    need(stage.GetEditTarget().GetLayer()==stage.GetRootLayer() and stage.GetSessionLayer().anonymous,
         'Runtime edits must target private anonymous layers')
    if bundle is not None:
        with Ar.ResolverContextBinder(stage.GetPathResolverContext()):
            if asset_tokens:check_resolution('private_stage')
    recheck()
    if raw_recheck is not None:
        raw_recheck()
        for ref in (actual,prepared,inventory['base'],binding['sources']['scene_authoring_receipt'],
                    binding['sources']['layered_runtime_equivalence']):
            _single_file(ref,immutable=True);read_source(ref)
        _bound_layered_metadata(binding,_strict_json(read_source(binding['sources']['prepared_complete'])))
    _sdk_bundle(binding)
    if bundle is not None:
        selection=dict(schema='m2_sdk_mdl_selection_v2',material_selection='explicit_approved_sdk_entry',
            original_render_equivalence_claim=False,manifest=bundle['manifest'],entry=bundle['entry'],
            authored_token='OmniPBR.mdl',asset_token_count=len(asset_tokens),
            bare_token_resolved=bare_resolutions,selected_entry_resolved=selected_resolutions)
        data=dict(stage.GetRootLayer().customLayerData)
        data[SDK_SELECTION_KEY]=json.dumps(selection,sort_keys=True,allow_nan=False)
        stage.GetRootLayer().customLayerData=data
    return stage


def write_sdk_selection(stage,directory):
    """Retain private-stage material selection diagnostics, without a USD copy."""
    record=_json(stage.GetRootLayer().customLayerData[SDK_SELECTION_KEY])
    output=real_path(real_path(directory)/'sdk_mdl_selection.json',exists=False)
    need(not output.exists(),'SDK MDL selection receipt already exists')
    _write(output,record)


def attach_runtime_stage(stage,context,update):
    """Use the installed omni.usd callback API, then verify the attached root.

    `update` is Kit app.update supplied by the already-owned caller. This
    function never constructs SimulationApp and never falls back to open_stage.
    """
    from pxr import UsdUtils
    need(stage.GetRootLayer().anonymous and all(layer.anonymous for layer in stage.GetUsedLayers()),
         'Only private anonymous runtime stages may be attached')
    cache=UsdUtils.StageCache.Get();already_cached=cache.Contains(stage)
    root_id=stage.GetRootLayer().identifier
    if already_cached:
        need(root_id in _RUNTIME_STAGE_LEASES and
             cache.GetId(stage).ToLongInt()==_RUNTIME_STAGE_LEASES[root_id]['stage_cache_id'],
             'Cannot claim an unowned pre-existing StageCache lease')
    stage_id=cache.Insert(stage)
    completed=[]
    try:
        accepted=context.attach_stage_with_callback(stage_id.ToLongInt(),lambda ok,error:completed.append((ok,error)))
        need(accepted is not False,'Native anonymous stage attach was rejected')
        for _ in range(240):
            if completed:break
            update()
        need(completed and completed[0][0],'Native anonymous stage attach failure: '+
             (str(completed[0][1]) if completed else 'callback did not finish'))
        attached=context.get_stage()
        need(attached is not None and attached.GetRootLayer()==stage.GetRootLayer() and
             attached.GetRootLayer().anonymous,'Native attached root differs from private root')
        _RUNTIME_STAGE_LEASES[root_id]=dict(stage_cache_id=stage_id.ToLongInt(),release_deferred=False)
        return attached
    except BaseException:
        if not already_cached:cache.Erase(stage_id)
        raise


def release_runtime_stage(stage):
    """Explicit isolated-test/failed-attach disposal; never shared finalization."""
    from pxr import UsdUtils
    _RUNTIME_STAGE_LEASES.pop(stage.GetRootLayer().identifier,None)
    UsdUtils.StageCache.Get().Erase(stage)


def runtime_stage_leases():
    """Validate original owned IDs without insertion or global cache mutation."""
    if not _RUNTIME_STAGE_LEASES:return []
    from pxr import Usd,UsdUtils
    cache=UsdUtils.StageCache.Get();result=[]
    for root_id,lease in sorted(_RUNTIME_STAGE_LEASES.items()):
        stage=cache.Find(Usd.StageCache.Id.FromLongInt(lease['stage_cache_id']))
        need(stage and stage.GetRootLayer().identifier==root_id and stage.GetRootLayer().anonymous,
             'Owned private StageCache lease is missing or changed')
        result.append(dict(schema='m2_private_stage_cache_lease_v1',**lease,
            stage_cache_id_valid=True,anonymous_root=root_id,release_policy='owned_process_exit'))
    return result


def defer_runtime_stage_release(stage):
    need(stage,'Owned private StageCache stage is no longer valid')
    root_id=stage.GetRootLayer().identifier
    need(root_id in _RUNTIME_STAGE_LEASES,'Private StageCache entry is not owned by this attach')
    runtime_stage_leases()
    _RUNTIME_STAGE_LEASES[root_id]['release_deferred']=True
    return next(row for row in runtime_stage_leases() if row['anonymous_root']==root_id)


def prepared_runtime(config,output):
    ref=prepare_job(config,output)
    stage=runtime_stage(ref)
    if ref['binding']['schema'] in (SDK_SCHEMA,SDK_LAYER_SCHEMA):write_sdk_selection(stage,output)
    return stage,_json((Path(output)/'mass_estimates.json').read_bytes())


def replay_scene(directory):
    """Resolve explicit references; an absent reference retains the legacy path."""
    directory=Path(directory);reference=directory/'scene_ref.json';legacy=directory/'training_scene.usdc'
    if reference.exists():
        need(not legacy.exists(),'Scene reference and saved USD are ambiguous')
        return resolve_scene(_json(reference.read_bytes()))
    return legacy


def replay_stage(directory):
    """Use the closed-dependency runtime loader for explicit CAS references."""
    reference=Path(directory)/'scene_ref.json'
    if reference.exists():
        replay_scene(directory)  # Reject ambiguous legacy/reference inputs too.
        return runtime_stage(_json(reference.read_bytes()))
    from pxr import Usd
    return Usd.Stage.Open(str(replay_scene(directory)))


def batch_ledger(job_roots,*,batch_root=None):
    """Count observed attempts and every external bound input by unique path.

    The denominator is actual supplied job trees, never a planned job count.
    This CPU reconciliation is separate from physical/data acceptance.
    """
    roots=[real_path(p) for p in job_roots]
    need(roots and len(set(roots))==len(roots),'Unique observed job roots required')
    need(not any(a in b.parents for a in roots for b in roots if a!=b),'Overlapping job trees')
    batch=None if batch_root is None else real_path(batch_root)
    if batch is not None:need(all(batch in root.parents for root in roots),'Every job must belong to batch_root')
    shared={};retained={};retained_inputs={};inputs={};rows=[];batch_ids=set();missing=[]
    measured_roots=roots if batch is None else [batch]
    def outside_measured(path):return not any(root in path.parents for root in measured_roots)
    def add_input(ref,kind):
        key=str(real_path(ref['file']))
        if key in inputs:
            need(inputs[key]['sha256']==ref['sha256'] and inputs[key]['bytes']==ref['bytes'],
                 'Conflicting shared input bindings: '+key)
            inputs[key]['kinds']=sorted(set(inputs[key]['kinds'])|{kind})
        else:inputs[key]=dict(file=key,sha256=ref['sha256'],bytes=ref['bytes'],kinds=[kind])
        return key
    # New v3 authoring receipts carry historical dense/body prerequisites.
    # Authenticate every receipt's full identity inventory before resolving a
    # scene or reading any authoring payload; cache only within this ledger.
    authoring_receipts={};authoring_sources={};authoring_stats={};historical_completes={}
    def signature(path):
        value=real_path(path).stat()
        return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,
                value.st_ctime_ns,value.st_nlink,value.st_mode)
    for root in roots:
        if not (root/'scene_ref.json').exists():continue
        reference=_json((root/'scene_ref.json').read_bytes())
        binding=reference.get('binding',{})
        if binding.get('schema')!=SDK_LAYER_SCHEMA:continue
        _,complete,_=_binding_metadata(binding,authorization=reference.get('authorization'))
        receipt,_,_,metadata=_bound_layered_metadata(binding,complete)
        if binding['dependency_policy']==SDK_LAYER_CURRENT_RAW_POLICY:
            record=_strict_json(read_source(binding['sources']['layered_runtime_equivalence']))
            historical=record['historical_prepared_complete'];name=historical['file']
            need(name not in authoring_sources or authoring_sources[name]==historical,
                 'Conflicting historical prepared COMPLETE bindings')
            historical_completes[name]=historical;authoring_sources[name]=historical
            authoring_stats[name]=signature(name)
        source=binding['sources']['scene_authoring_receipt']
        key=(source['file'],source['sha256'],source['bytes'])
        if key in authoring_receipts:continue
        from .layered_scene_authoring import _single_file,authoring_input_inventory
        _single_file(source,immutable=True)
        original_inputs=authoring_input_inventory(receipt,metadata)
        authoring_receipts[key]=(receipt,original_inputs)
        for original in original_inputs:
            name=original['file']
            need(name not in authoring_sources or authoring_sources[name]==original,
                 'Conflicting original authoring input bindings')
            authoring_sources[name]=original
            authoring_stats[name]=signature(name)
    for name,source in authoring_sources.items():
        read_source(source)
        need(signature(name)==authoring_stats[name],'Original authoring input changed during hashing')
    for root in roots:
        row=dict(root=str(root),**episode_size(root));rows.append(row)
        if not (root/'scene_ref.json').exists():
            need((root/'FAILED.json').is_file(),'Observed attempt lacks both scene_ref.json and FAILED.json')
            row['scene_reference_status']='absent_early_failure';missing.append(str(root))
            continue
        row['scene_reference_status']='verified'
        ref=_json((root/'scene_ref.json').read_bytes());path=resolve_scene(ref)
        need(not any(job in path.parents for job in roots),'Shared CAS must be outside job trees')
        if batch is not None:need(batch not in path.parents,'Persistent CAS must be outside the measured batch tree')
        batch_ids.add(ref['binding']['batch_id'])
        shared[str(path)]=dict(file=str(path),sha256=ref['scene']['sha256'],bytes=ref['scene']['bytes'])
        add_input(shared[str(path)],'cas_scene')
        _sdk_bundle(ref['binding'])
        for name,source in ref['binding']['sources'].items():
            read_source(source,allow_empty=_sdk_file_name(name))
            source_path=real_path(source['file'])
            if source_path!=path and outside_measured(source_path):
                key=add_input(source,name);retained_inputs[key]=deepcopy(source)
                if name=='scene':retained[key]=deepcopy(source)
        if ref['binding']['schema']==SDK_LAYER_SCHEMA:
            source=ref['binding']['sources']['scene_authoring_receipt']
            receipt,original_inputs=authoring_receipts[(source['file'],source['sha256'],source['bytes'])]
            original_scenes={item['original']['file']:item['original'] for item in receipt['scenes']}
            for original in original_inputs:
                source_path=real_path(original['file'])
                if outside_measured(source_path):
                    key=add_input(original,'layer_authoring_source');retained_inputs[key]=deepcopy(original)
                    if key in original_scenes:retained[key]=deepcopy(original)
    for source in historical_completes.values():
        if outside_measured(real_path(source['file'])):
            key=add_input(source,'layered_historical_prepared_complete');retained_inputs[key]=deepcopy(source)
    need(len(batch_ids)<=1,'Ledger must cover one explicit batch')
    total=sum(row['total_bytes'] for row in rows);initial=sum(row['bytes'] for row in shared.values())
    batch_total=total if batch is None else episode_size(batch)['total_bytes']
    overhead=batch_total-total
    original_bytes=sum(row['bytes'] for row in retained.values())
    retained_bytes=sum(row['bytes'] for row in retained_inputs.values())
    input_bytes=sum(row['bytes'] for row in inputs.values());shared_bytes=input_bytes+overhead
    complete=not missing
    for row in rows:
        row['accounted_bytes_with_batch_amortization']=row['total_bytes']+shared_bytes/len(rows)
        row['within_budget_with_shared_amortization']=row['accounted_bytes_with_batch_amortization']<=EPISODE_LIMIT_BYTES
    for name,expected in authoring_stats.items():
        need(signature(name)==expected,'Original authoring input changed during ledger reconciliation')
    return dict(schema=LEDGER_SCHEMA,batch_id=batch_ids.pop() if batch_ids else None,observed_jobs=len(rows),jobs=rows,
        shared_scenes=list(shared.values()),job_directory_bytes=total,shared_scene_bytes=initial,
        batch_root=None if batch is None else str(batch),batch_directory_bytes=batch_total,
        batch_overhead_bytes=overhead,batch_overhead_evaluated=batch is not None,
        retained_prepared_scenes=list(retained.values()),retained_prepared_scene_bytes=original_bytes,
        retained_prepared_inputs=list(retained_inputs.values()),retained_prepared_input_bytes=retained_bytes,
        shared_inputs=list(inputs.values()),shared_input_bytes=input_bytes,
        other_shared_input_bytes=retained_bytes-original_bytes,
        shared_input_inventory_complete=complete,attempts_without_scene_reference=missing,
        total_accounted_bytes_complete=complete,
        shared_scene_amortized_bytes_per_job=(initial+original_bytes)/len(rows),
        shared_input_amortized_bytes_per_job=input_bytes/len(rows),
        batch_overhead_amortized_bytes_per_job=overhead/len(rows),total_accounted_bytes=total+shared_bytes,
        accounted_bytes_per_observed_job=(total+shared_bytes)/len(rows),
        all_job_directories_within_budget=all(row['within_budget'] for row in rows),
        all_accounted_jobs_within_budget=complete and all(row['within_budget_with_shared_amortization'] for row in rows),
        initial_scene_is_shared_input=True,training_eligible=False,physical_validation=False)


def finalize_job(output,result):
    """Final CAS readback and self-inclusive entire-job st_size inventory."""
    output=real_path(output);ref=_json((output/'scene_ref.json').read_bytes());integrity_error=None;cas_ok=True
    try:resolve_scene(ref)
    except Exception as error:integrity_error=error;cas_ok=False
    try:_validate_binding(ref['binding'],**({'authorization':ref['authorization']} if 'authorization' in ref else {}))
    except Exception as error:integrity_error=error
    storage=dict(schema=LEDGER_SCHEMA,cas_unchanged=cas_ok,source_bindings_unchanged=integrity_error is None,
        dependency_policy=ref['binding']['dependency_policy'],
        external_file_dependencies_supported=False,
        sdk_mdl_manifest=deepcopy(ref['binding']['sources'].get('sdk_mdl_manifest')),
        sdk_mdl_selections=[source_ref(path) for path in sorted(output.rglob('sdk_mdl_selection.json'))]
            if ref['binding']['schema'] in (SDK_SCHEMA,SDK_LAYER_SCHEMA) else [],
        scene=deepcopy(ref['scene']),
        job_directory_bytes=0,limit_bytes=EPISODE_LIMIT_BYTES,within_budget=True,
        shared_scene_bytes=ref['scene']['bytes'],
        batch_amortized_budget_evaluated=False,
        shared_scene_accounting='Use batch_ledger on all observed job roots; charge each CAS object once',
        batch_id=ref['binding']['batch_id'])
    result['scene_storage']=storage
    if integrity_error is not None:
        result.update(success=False,compact_capture_complete=False,physical_validation=False,
            failure_reason='Scene storage integrity failure: '+str(integrity_error))
    _inventory_receipts(output,result,storage)
    if integrity_error is not None:raise ValueError(result['failure_reason']) from integrity_error
    try:resolve_scene(ref)
    except Exception as error:
        storage['cas_unchanged']=False
        result.update(success=False,compact_capture_complete=False,physical_validation=False,
            failure_reason='Scene storage changed during final readback: '+str(error))
        _inventory_receipts(output,result,storage)
        raise
    try:_sdk_bundle(ref['binding'])
    except Exception as error:
        storage['source_bindings_unchanged']=False
        result.update(success=False,compact_capture_complete=False,physical_validation=False,
            failure_reason='SDK MDL storage changed during final readback: '+str(error))
        _inventory_receipts(output,result,storage)
        raise
    return result


def _inventory_receipts(output,result,storage):
    for _ in range(32):
        _write(output/'COMPLETE.json',result)
        if (output/'fresh_expert_result.json').exists():_write(output/'fresh_expert_result.json',result)
        measured=episode_size(output)
        if storage['job_directory_bytes']==measured['total_bytes']:break
        storage['job_directory_bytes']=measured['total_bytes'];storage['within_budget']=measured['within_budget']
        if not measured['within_budget']:
            result.update(success=False,compact_capture_complete=False,physical_validation=False,
                failure_reason='Entire compact job directory exceeds 100000000 bytes')
    else:raise ValueError('Whole-job byte ledger failed to converge')


def validate_dependency_inventory(inventory):
    """Authenticate the exact primary + one static base before any composition."""
    from .layered_scene_authoring import validate_inventory
    return validate_inventory(inventory)


def _layered_binding(binding,complete=None):
    """Validate metadata first, then every closed layer and immutable source byte."""
    if complete is None:complete=_json(read_source(binding['sources']['prepared_complete']))
    receipt,item,_,_=_bound_layered_metadata(binding,complete)
    validate_dependency_inventory(complete['scene_dependency_inventory'])
    sources=binding['sources']
    from .layered_scene_authoring import _single_file
    for name,ref in sources.items():
        if not _sdk_file_name(name):_single_file(ref,immutable=name in ('scene','scene_base','scene_authoring_receipt'))
    return receipt,item


def _runtime_proof_mode(proof,preparation):
    """Validate the bound fixed profile without comparing resolver-dependent digests."""
    from .layered_scene_authoring import ITERATIONS
    need(isinstance(proof,dict) and proof.get('schema')=='m2_solver_iteration_only_authoring_v2' and
         proof.get('profile')==ITERATIONS and
         all(type(proof['profile'][k]) is type(v) for k,v in ITERATIONS.items()),
         'Exact qualified iteration proof required')
    already='source_already_qualified64' in proof
    need(not already or proof['source_already_qualified64'] is True,'Unknown original iteration proof mode')
    original=dict(ITERATIONS,position_iterations=64 if already else 32)
    need(preparation.get('physics')==original and
         all(type(preparation['physics'][k]) is type(v) for k,v in original.items()),
         'Original physics differs from the explicit iteration proof mode')
    roles=proof.get('rigid_articulation_roles')
    need(isinstance(roles,list) and len(roles)==24 and all(isinstance(row,list) and len(row)==2 and
         isinstance(row[0],str) and row[0].startswith('/') and
         row[1] in ('physxRigidBody','physxArticulation') for row in roles) and
         len({tuple(row) for row in roles})==24,'Exact 24 rigid/articulation roles required')
    expected=[dict(path=path,attribute=prefix+':solverPositionIterationCount',before=32,after=64)
              for path,prefix in roles]
    need(proof.get('only_iteration_changes')==expected and
         proof.get('armature_authoring_changed') is False and
         proof.get('geometry_camera_drives_mass_contact_unchanged') is True and
         proof.get('exhaustive_metadata_API_relationship_connection_time_sample_check') is True,
         'Only the fixed 24 position defaults may differ')
    armatures=proof.get('float32_armature_attributes')
    need(isinstance(armatures,list) and armatures and all(isinstance(path,str) and path.startswith('/') and
         path.endswith(':armature') for path in armatures) and len(set(armatures))==len(armatures),
         'Exact authored float32 armature inventory required')
    for name in ('before_stage_semantics_sha256','after_stage_semantics_sha256',
                 'before_authored_sha256','after_authored_sha256'):
        need(isinstance(proof.get(name),str) and re.fullmatch('[0-9a-f]{64}',proof[name]) is not None,
             'Complete original CPU iteration proof digests required')
    return already


def _layered_metadata(complete):
    """Authenticate the whole original population before opening any USD/NPZ."""
    from .layered_scene_authoring import (INVENTORY_SCHEMA,RECEIPT_SCHEMA,RUNTIME_EQUIVALENCE_POLICY,
        _single_file,_population_metadata,authoring_input_inventory)
    need(isinstance(complete,dict),'Explicit prepared COMPLETE metadata required')
    policy=_layered_policy(complete)
    inventory=complete.get('scene_dependency_inventory')
    need(isinstance(inventory,dict) and set(inventory)=={'schema','primary','base'} and
         inventory['schema']==INVENTORY_SCHEMA,'Explicit closed USD dependency inventory required')
    for role in ('primary','base'):_source_shape(inventory[role])
    source=complete.get('scene_authoring_receipt');_single_file(source,immutable=True)
    receipt=(_strict_json if policy==SDK_LAYER_CURRENT_RAW_POLICY else _json)(read_source(source))
    need(isinstance(receipt,dict) and receipt.get('schema')==RECEIPT_SCHEMA and
         receipt.get('runtime_equivalence_policy')==RUNTIME_EQUIVALENCE_POLICY and
         receipt.get('representation_change_only') is True and receipt.get('base')==inventory['base'],
         'Explicit versioned approved SDK runtime equivalence policy required')
    metadata=_population_metadata(receipt.get('original_selected'))
    authoring_input_inventory(receipt,metadata)  # Headers/stat only; no authoring payload is opened here.
    scenes=receipt['scenes'];proofs=receipt.get('iteration_proofs')
    need(isinstance(proofs,list) and len(proofs)==len(metadata),'Exact ordered original iteration proofs required')
    matches=[i for i,item in enumerate(scenes) if item.get('dependency_inventory')==inventory]
    need(len(matches)==1,'Unique current scene to original prepared reference binding required')
    for i,proof in enumerate(proofs):_runtime_proof_mode(proof,metadata[i][0])
    index=matches[0];item=scenes[index]
    need(item.get('saved_USD_readback_exact') is True and item.get('private_runtime_readback_exact') is True and
         complete.get('effective_scene_sha256')==inventory['primary']['sha256'] and
         complete.get('patient')==metadata[index][2],'Exact saved scene/patient proof binding required')
    for name in ('composed_semantics_sha256','authored_semantics_sha256'):
        need(isinstance(item.get(name),str) and re.fullmatch('[0-9a-f]{64}',item[name]) is not None,
             'Complete original CPU authored and composed snapshots required')
    if policy==SDK_LAYER_CURRENT_RAW_POLICY:
        _current_raw_sidecar(complete,receipt,item,index,metadata)
    return receipt,item,index,metadata


def _current_raw_sidecar(complete,receipt,item,index,metadata):
    from .layered_scene_authoring import (_single_file,current_raw_sidecar_record,authoring_input_inventory,historical_prepared_sources)
    source=complete['layered_runtime_equivalence'];_single_file(source,immutable=True)
    record=_strict_json(read_source(source))
    need(isinstance(record,dict) and type(record.get('scene_index')) is int,'Exact current RAW sidecar index required')
    historical=record.get('historical_prepared_complete');_single_file(historical,immutable=True)
    need(historical['file']==str(Path(item['dependency_inventory']['primary']['file']).parent/'COMPLETE.json'),
         'Historical prepared COMPLETE must belong to the exact primary parent')
    old_complete=_strict_json(read_source(historical))
    need(_layered_policy(old_complete)==SDK_LAYER_DEPENDENCY_POLICY,'Historical COMPLETE cannot refer to a v4 sidecar')
    derived=deepcopy(complete)
    derived.pop('scene_dependency_policy');derived.pop('layered_runtime_equivalence')
    need(_exact_json(derived,old_complete),'Derived COMPLETE differs from exact historical prepared COMPLETE')
    prepared_sources=historical_prepared_sources(historical,old_complete,receipt,item,index,metadata)
    expected=current_raw_sidecar_record(historical,complete['scene_authoring_receipt'],prepared_sources,receipt,item,index)
    need(_exact_json(record,expected),'Exact versioned current RAW sidecar provenance required')
    refs=[source,historical,complete['scene_authoring_receipt'],item['original'],
          item['dependency_inventory']['primary'],item['dependency_inventory']['base']]
    for ref in receipt['original_selected'][index].values():
        refs.append(dict(ref,bytes=real_path(ref['file']).stat().st_size))
    paths=[];inodes=[]
    for ref in refs:
        path=_single_file(ref,immutable=False);paths.append(path);s=path.stat();inodes.append((s.st_dev,s.st_ino))
    need(len(set(paths))==len(set(inodes))==len(refs),'Current RAW provenance roles cannot alias')
    # Every original header/reference is authenticated before a USD/body payload.
    authoring_input_inventory(receipt,metadata)
    _strict_original_headers(receipt,metadata)
    for proof,meta in zip(receipt['iteration_proofs'],metadata):_strict_runtime_proof(proof,meta[0])
    return record


def _strict_original_headers(receipt,metadata):
    """Only identity/preparation metadata; no geometry/body payload reads."""
    for selection,meta in zip(receipt['original_selected'],metadata):
        prep=meta[0];spec=prep['patient_asset']
        refs=list(selection.values())+[dict(file=str(Path(spec['directory'])/name),sha256=spec[key])
            for name,key in (('asset_manifest.json','manifest_sha256'),('summary.json','summary_sha256'))]
        from .m2_contract import IDENTITY_SPLIT_SHA256
        refs.append(dict(file=spec['identity_split'],sha256=IDENTITY_SPLIT_SHA256))
        for ref in refs:
            _strict_json(read_source(dict(ref,bytes=real_path(ref['file']).stat().st_size)))


def _strict_runtime_proof(proof,preparation):
    already=_runtime_proof_mode(proof,preparation)
    keys={'schema','profile','only_iteration_changes','rigid_articulation_roles','float32_armature_attributes',
          'before_stage_semantics_sha256','after_stage_semantics_sha256','before_authored_sha256','after_authored_sha256',
          'armature_authoring_changed','geometry_camera_drives_mass_contact_unchanged',
          'exhaustive_metadata_API_relationship_connection_time_sample_check','physics_executed',
          'physical_validation','training_eligible'}
    if already:keys.add('source_already_qualified64')
    need(set(proof)==keys and all(proof[k] is False for k in ('physics_executed','physical_validation','training_eligible')),
         'Exact closed original iteration proof required')
    roles=proof['rigid_articulation_roles']
    changes=[dict(path=path,attribute=prefix+':solverPositionIterationCount',before=32,after=64) for path,prefix in roles]
    need(_exact_json(proof['only_iteration_changes'],changes),'Typed original iteration changes required')
    return already


def _selected_raw_expected(receipt,item,index,metadata):
    """Exact bound original dense scene before SDK asset selection or Flatten."""
    from .layered_scene_authoring import (_private_source,replay_qualified_iterations,_single_file,authoring_input_inventory)
    proof=receipt['iteration_proofs'][index]
    already=_strict_runtime_proof(proof,metadata[index][0])
    for original_proof,meta in zip(receipt['iteration_proofs'],metadata):_strict_runtime_proof(original_proof,meta[0])
    originals=authoring_input_inventory(receipt,metadata)
    def recheck_originals():
        authoring_input_inventory(receipt,metadata)
        for ref in originals:_single_file(ref,immutable=False);read_source(ref)
    # Authenticate ALL original population metadata before the first body read.
    _strict_original_headers(receipt,metadata)
    recheck_originals()
    original=item['original']
    need(original['file']==str(metadata[index][3]/'candidate_physics.usdc') and
         original['sha256']==metadata[index][1]['effective_scene_sha256'],'Exact original dense reference required')
    stage=_private_source(original)
    read_source(original)  # OpenAsAnonymous must have read the same bound bytes.
    selected,_=replay_qualified_iterations(stage,proof,already64=already)
    recheck_originals()
    return selected,recheck_originals


def validate_prepared_dependency_inventory(complete):
    """Preflight all original identity headers/policy, then authenticate both layers."""
    _layered_metadata(complete)
    return validate_dependency_inventory(complete['scene_dependency_inventory'])


def _selected_runtime_expected(binding,receipt,item,index,metadata):
    """Same approved SDK/current context, bound dense original, RAM only."""
    from .layered_scene_authoring import apply_qualified_iterations,scene_snapshot,_single_file
    selection=receipt['original_selected'][index];proof=receipt['iteration_proofs'][index]
    already=_runtime_proof_mode(proof,metadata[index][0])
    originals=receipt['input_bindings'][index]
    def recheck_originals():
        for original in originals:
            _single_file(original,immutable=False);read_source(original)
        _sdk_bundle(binding)
    recheck_originals()
    original=item['original'];root=metadata[index][3]
    need(original['file']==str(root/'candidate_physics.usdc') and
         original['sha256']==metadata[index][1]['effective_scene_sha256'],
         'Exact dense original scene path/hash required')
    old=build_binding(**selection,store_root=binding['store_root'],batch_id=binding['batch_id'],
                      sdk_mdl_manifest=binding['sources']['sdk_mdl_manifest'])
    need(old['schema']==SDK_SCHEMA and old['sources']['scene']==original,
         'Dense original must use the exact non-recursive v2 SDK route')
    stage=_runtime_stage_clone_impl(old,real_path(original['file']),lambda:_validate_binding(old))
    if already:
        from pxr import Sdf,Usd,UsdPhysics
        before=scene_snapshot(stage)
        layer=Sdf.Layer.CreateAnonymous();layer.TransferContent(stage.GetRootLayer());temporary=Usd.Stage.Open(layer)
        positions=[]
        for prim in temporary.TraverseAll():
            for api,prefix in ((UsdPhysics.RigidBodyAPI,'physxRigidBody'),(UsdPhysics.ArticulationRootAPI,'physxArticulation')):
                if prim.HasAPI(api):
                    attr=prim.GetAttribute(prefix+':solverPositionIterationCount')
                    need(attr and attr.GetTypeName()==Sdf.ValueTypeNames.Int and type(attr.Get()) is int and
                         attr.Get()==64 and not attr.GetTimeSamples(),'Exact original qualified64 defaults required')
                    positions.append(attr)
        need(len(positions)==24,'Exact original qualified64 role count required')
        for attr in positions:need(attr.Set(32),'Private qualified64 proof inversion failed')
        selected,actual=apply_qualified_iterations(temporary)
        need(scene_snapshot(selected)==before,'Already-qualified64 proof changed complete stage semantics')
    else:selected,actual=apply_qualified_iterations(stage)
    for name in ('profile','only_iteration_changes','rigid_articulation_roles','float32_armature_attributes',
                 'armature_authoring_changed','geometry_camera_drives_mass_contact_unchanged',
                 'exhaustive_metadata_API_relationship_connection_time_sample_check'):
        need(actual[name]==proof[name],'Current typed iteration proof differs from bound CPU proof: '+name)
    recheck_originals()
    return selected,recheck_originals


def _runtime_stage_clone(binding,source,recheck):
    if binding['schema']!=SDK_LAYER_SCHEMA:return _runtime_stage_clone_impl(binding,source,recheck)
    from pxr import Ar
    from .layered_scene_authoring import scene_snapshot
    context=Ar.GetResolver().GetCurrentContext()
    with Ar.ResolverContextBinder(context),Ar.ResolverScopedCache():
        complete=_json(read_source(binding['sources']['prepared_complete']))
        receipt,item,index,metadata=_bound_layered_metadata(binding,complete)
        if binding['dependency_policy']==SDK_LAYER_CURRENT_RAW_POLICY:
            # The compulsory low-level RAW gate precedes even the SDK oracle's
            # reference transformation. It is never a cached sidecar assertion.
            actual=_runtime_stage_clone_impl(binding,source,recheck)
            expected,recheck_originals=_selected_runtime_expected(binding,receipt,item,index,metadata)
        else:
            expected,recheck_originals=_selected_runtime_expected(binding,receipt,item,index,metadata)
            actual=_runtime_stage_clone_impl(binding,source,recheck)
        need(scene_snapshot(actual)==scene_snapshot(expected),
             'Layered runtime differs from the full current-context approved SDK dense reference')
        recheck_originals();recheck()
        return actual


def _bound_layered_metadata(binding,complete):
    _binding_shape(binding)
    need(binding['dependency_policy']==_layered_policy(complete),'Layered binding/COMPLETE policy mismatch')
    result=_layered_metadata(complete)
    inventory=complete['scene_dependency_inventory'];sources=binding['sources']
    need(inventory['primary']==sources['scene'] and inventory['base']==sources['scene_base'],
        'Layered dependencies differ from exact prepared source inventory')
    need(complete['scene_authoring_receipt']==sources['scene_authoring_receipt'],
        'Layered authoring receipt substitution')
    if binding['dependency_policy']==SDK_LAYER_CURRENT_RAW_POLICY:
        need(_exact_json(complete['layered_runtime_equivalence'],sources['layered_runtime_equivalence']),
             'Layered runtime equivalence sidecar substitution')
        record=_strict_json(read_source(sources['layered_runtime_equivalence']))
        need(all(_exact_json(sources.get(name),ref) for name,ref in record['historical_prepared_sources'].items()) and
             set(sources)&set(OPTIONAL)==set(record['historical_prepared_sources'])&set(OPTIONAL),
             'Prepared body/physics/camera sources differ from exact historical prepared inputs')
    return result
