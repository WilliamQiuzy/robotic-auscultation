"""Closed, read-only capability for the completed V3 CPU authoring stage.

This module never authors a scene, issues a current-RAW sidecar, or launches a
native job. Consumers may authenticate the same capability with a JSON-only
reader; payload verification belongs to the CPU preparation boundary.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import stat

from sim.ausc_policy import scene_store

need=scene_store.need
SCHEMA='m2_given_completed_authoring_checkpoint_v1'
NEGATIVE_SCHEMA='m2_given_completed_authoring_prepare_negative_v1'
SOURCE_ROOT='/workspace/wam/configs/m2-given-formal-lossless-cohort-20261028-v3'
ASSEMBLY_ROOT='/workspace/wam/configs/m2-given-formal-lossless-cohort-20261029-v4'
E4_ROOT='/opt/wam-runtime/code/m2-given-lossless-current-raw-early-v4'
E4_COMMIT='9eb93222d6456e80868f8dad7f7b03270b3e5dbd'
CORE_ROOT='/opt/wam-runtime/code/m2-lossless-current-raw-core-v4'
QUALIFIED_ROOT='/opt/wam-runtime/code/m2-interior-tail-06bc874'
CAPTURE_ROLES={
    'camera':('sim/camera_framing_20260923/recommended_camera.json',8126,'aeac00eb5a963520e869c60efff52a300484bd1c4a49232067a669c525ad07f0'),
    'calibration':('sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json',3374,'ce6bcda3bef2282e07be0f44686dc007756ede57cdfcd732d35e28c1d469d054')}
PINNED={
    'original_settings':(1028565,'3441a0eb7ba74785e74ecbcd542bc12ab369ef441ddea9167667d97cf679635b'),
    'original_execution_freeze':(329934,'e920f0b0f4f05ac0ac01058258d72aeb975fbc93145a661b66b36e8f9a44100d'),
    'original_recipe':(86933,'6d1f32ecb21257065a865dcea96d8f0bd9861ae764aa525ce3acfe1b1d8fa3e1'),
    'failed_terminal':(2880,'6859ae2ece1127c94558871ae5f026916290329ef9dba1234fb478e87305fb33'),
    'failed_log':(1258,'8d7b1aedd0b2e051925126c001b0e3129302c51e16d0094a293e2c624a738a63'),
    'historical_source_manifest':(192240,'bced8df1f1a2b02a29324c163abc81db605a25d79e1f580c5bd7b3703a473a99'),
    'historical_archive':(3407663,'ebbca691cc23649e9c98cf203b73f895d135c9d488af0e78dceb0c1d8b1d2dd0'),
    'historical_archive_receipt':(968,'73ce5027ef2fa47f8957e9e55d99355d0661680bc1616208409cf058e065b9a2'),
    'sdk_receipt':(7914143,'ad11d37477461dbf99c7db9a4f255379b47caadefe368fd73b3d109c7ba2a750'),
    'scene_receipt':(621723,'2e6050a5b10d45e246a568c7d544c1591000fbf90969a0b75f77a83f94787bd1')}
PINNED_PATHS={
    **{key:'/workspace/wam/configs/m2-given-formal-lossless-support-20261028-v3/'+name for key,name in
       (('original_settings','settings.json'),('original_execution_freeze','execution_freeze.json'),('original_recipe','recipe.json'))},
    'failed_terminal':'/workspace/wam/diagnostics/m2-given-formal-lossless-cpu-prepare-20261028-v3/terminal.json',
    'failed_log':'/workspace/wam/diagnostics/m2-given-formal-lossless-cpu-prepare-20261028-v3/operation.log',
    'historical_source_manifest':E4_ROOT+'/SOURCE_MANIFEST.json',
    'historical_archive':'/workspace/wam/source_archives/m2-given-lossless-current-raw-early-v4/m2-given-lossless-current-raw-early-v4-full-source.tar.gz',
    'historical_archive_receipt':'/workspace/wam/bootstrap/m2-given-lossless-current-raw-early-v4-full-source.tar.gz.receipt.json',
    'sdk_receipt':SOURCE_ROOT+'/SDK_RUNTIME_EQUIVALENCE.json',
    'scene_receipt':SOURCE_ROOT+'/derived/scene_authoring_receipt.json'}
SCENE_ROLES=('preparation_config','prepared_complete','patient_receipt')
REQUIRED_RETAINED_ROOTS=[
    dict(path=E4_ROOT,category='source',kind='directory'),
    dict(path='/workspace/wam/source_archives/m2-given-lossless-current-raw-early-v4',category='source',kind='directory'),
    dict(path='/workspace/wam/bootstrap/m2-given-lossless-current-raw-early-v4-full-source.tar.gz.receipt.json',category='source',kind='file'),
    *[dict(path='/workspace/wam/configs/'+name,category='prepared',kind='directory') for name in
        ('m2-given-formal-lossless-support-20261028-v3','m2-given-formal-lossless-cohort-20261028-v3')],
    *[dict(path='/workspace/wam/diagnostics/'+name,category='prepared',kind='directory') for name in
        ('m2-given-formal-lossless-cpu-support-20261028-v3','m2-given-formal-lossless-cpu-prepare-20261028-v3',
         'm2-given-cpu-builder-repro-20261029-v1','m2-given-cpu-builder-native60-20261029-v1',
         'm2-given-cpu-builder-full-config-20261029-v2','m2-given-cpu-builder-full-config-20261029-v3',
         'm2-given-cpu-builder-qualified-refs-20261029-v1')],
    dict(path='/workspace/wam/bootstrap/m2-given-cpu-builder-diagnostics-20261029-v1',category='source',kind='directory'),
    dict(path='/workspace/wam/bootstrap/m2-given-cpu-builder-diagnostics-20261029-v1.tar.gz',category='source',kind='file'),
    *[dict(path='/workspace/wam/diagnostics/'+name+'/negative.json',category='prepared',kind='file') for name in
        ('m2-formal-cpu-prepare-20261028-v1-negative','m2-formal-cpu-support-20261028-v2-negative')],
    *[dict(path=str(Path(root)/relative),category='prepared',kind='file') for root in (CORE_ROOT,QUALIFIED_ROOT)
        for relative,_,_ in CAPTURE_ROLES.values()]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(raw):
    need(type(raw) is str and Path(raw).is_absolute() and str(Path(raw))==raw and '..' not in Path(raw).parts,
         'Canonical absolute checkpoint path required')
    path=Path(raw)
    need(not any(p.is_symlink() for p in (path,*path.parents)) and str(path.resolve())==raw,
         'Checkpoint path aliases and symbolic links forbidden')
    return path


def signature(path):
    st=Path(path).lstat()
    return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_nlink,st.st_mode)


def header(ref):
    need(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['bytes']) is int and ref['bytes']>=0 and
         type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,
         'Full typed checkpoint reference required')
    path=canonical(ref['path']);st=path.lstat()
    need(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_size==ref['bytes'],
         'Checkpoint regular single-link file header changed')
    return path,signature(path)


def reference(path):
    path=canonical(str(path));st=path.lstat()
    ref=dict(path=str(path),bytes=st.st_size,sha256=sha(path));header(ref)
    return ref


def short(ref):
    return dict(file=ref.get('path',ref.get('file')),sha256=ref['sha256'])


def full_file(ref):
    return dict(file=ref['path'],bytes=ref['bytes'],sha256=ref['sha256'])


def authenticate_inventory(roots,rows,fresh,*,payloads=True,bind=None):
    """Close every path/inode/size before hashing the first opaque payload."""
    need(type(roots) is list and roots and type(rows) is list and rows, 'Closed retained checkpoint inventory required')
    paths=[];actual={}
    for row in roots:
        need(type(row) is dict and set(row)=={'path','category','kind'} and row['category'] in ('source','prepared') and
             row['kind'] in ('file','directory'), 'Exact retained-root role required')
        root=canonical(row['path']);paths.append(root)
        need(root.exists() and (root.is_file() if row['kind']=='file' else root.is_dir()), 'Retained root missing')
        for p in ([root] if row['kind']=='file' else root.rglob('*')):
            st=p.lstat();need(not stat.S_ISLNK(st.st_mode), 'Retained symbolic link forbidden')
            if stat.S_ISDIR(st.st_mode):continue
            need(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and str(p) not in actual, 'Retained special/alias file forbidden')
            actual[str(p)]=row['category']
    need(all(a!=b and a not in b.parents and b not in a.parents for i,a in enumerate(paths) for b in paths[i+1:]),
         'Retained roots overlap')
    future=[canonical(str(p)) for p in fresh]
    need(all(a!=b and a not in b.parents and b not in a.parents for a in paths for b in future), 'Source and successor roots overlap')
    seen=set();inodes=set();snapshots={}
    for row in rows:
        need(type(row) is dict and set(row)=={'category','reference'}, 'Exact retained input row required')
        ref=row['reference'];path,snapshot=header(ref)
        need(str(path) in actual and row['category']==actual[str(path)] and str(path) not in seen and snapshot[:2] not in inodes,
             'Retained category/path/inode alias or duplicate')
        seen.add(str(path));inodes.add(snapshot[:2]);snapshots[str(path)]=snapshot
        if bind is not None:need(bind(ref)==ref, 'Checkpoint file outside retained accounting scope')
    need([row['reference']['path'] for row in rows]==sorted(actual) and seen==set(actual), 'Unlisted or unordered retained files')
    if payloads:
        for row in rows:
            ref=row['reference'];need(sha(ref['path'])==ref['sha256'], 'Retained checkpoint payload changed')
    need(all(signature(path)==snapshot for path,snapshot in snapshots.items()), 'Checkpoint file changed during authentication')
    return deepcopy(rows)


def capture_paths(pairs,*,pinned=False,bind=None,read_json=None):
    need(type(pairs) is dict and set(pairs)==set(CAPTURE_ROLES), 'Only camera and calibration substitutions permitted')
    result={};refs=[]
    for name,pair in pairs.items():
        need(type(pair) is dict and set(pair)=={'default','qualified'}, 'Exact capture path pair required')
        for ref in pair.values():header(ref);refs.append(ref)
        a,b=pair['default'],pair['qualified']
        need(a['path']!=b['path'] and a['bytes']==b['bytes'] and a['sha256']==b['sha256'], 'Capture substitution must preserve exact bytes')
        if pinned:
            relative,size,digest=CAPTURE_ROLES[name]
            need(a==dict(path=str(Path(CORE_ROOT)/relative),bytes=size,sha256=digest) and
                 b==dict(path=str(Path(QUALIFIED_ROOT)/relative),bytes=size,sha256=digest), 'Capture pair differs from exact qualified/default roles')
        result[name+'_file']=b['path']
    for ref in refs:
        if bind is not None:need(bind(ref)==ref, 'Capture role not retained')
        if read_json is None:need(sha(ref['path'])==ref['sha256'], 'Capture file contents changed')
    if read_json is not None:
        for pair in pairs.values():need(read_json(pair['default'])==read_json(pair['qualified']), 'Capture JSON contents differ')
    return result


def validate_headers(value,selected,allocated,assembly_root,*,bind=None):
    keys={'schema','source_root','assembly_root','bindings','scenes','plans','retained_roots','retained_inputs',
          'capture_paths','physical_launched_count','physical_denominator','model_denominator','test_payloads_opened','training_eligible'}
    need(type(value) is dict and set(value)==keys and value['schema']==SCHEMA and
         value['source_root']==SOURCE_ROOT and value['assembly_root']==ASSEMBLY_ROOT==str(assembly_root),
         'Exact versioned source-V3/assembly-V4 checkpoint required')
    for key in ('physical_launched_count','physical_denominator','model_denominator'):
        need(type(value[key]) is int and value[key]==0, 'Completed CPU checkpoint has zero physical/model denominators')
    need(value['test_payloads_opened'] is False and value['training_eligible'] is False, 'CPU recovery cannot qualify data/model')
    bindings=value['bindings'];need(type(bindings) is dict and set(bindings)==set(PINNED)|{'failed_negative'}, 'Closed historical binding roles required')
    need(value['retained_roots']==[*REQUIRED_RETAINED_ROOTS,dict(path=bindings['failed_negative']['path'],category='prepared',kind='file')],
         'Checkpoint retained roots differ from the finite historical/diagnostic scope')
    inventory={row['reference']['path']:row['reference'] for row in value['retained_inputs']}
    def registered(ref):
        header(ref);need(inventory.get(ref['path'])==ref, 'Checkpoint role omitted from complete costs')
        if bind is not None:need(bind(ref)==ref, 'Checkpoint role outside plan accounting')
    for key,ref in bindings.items():
        registered(ref)
        if key in PINNED:need(ref['path']==PINNED_PATHS[key] and (ref['bytes'],ref['sha256'])==PINNED[key], 'Historical '+key+' path/bytes/SHA changed')
    root=canonical(value['source_root']);canonical(value['assembly_root'])
    need(bindings['sdk_receipt']['path']==str(root/'SDK_RUNTIME_EQUIVALENCE.json') and
         bindings['scene_receipt']['path']==str(root/'derived/scene_authoring_receipt.json') and
         bindings['historical_source_manifest']['path']==str(Path(E4_ROOT)/'SOURCE_MANIFEST.json'), 'Historical source role paths changed')
    need(len(selected)==48 and len(allocated)==240 and type(value['scenes']) is list and len(value['scenes'])==48 and
         type(value['plans']) is list and len(value['plans'])==240, 'Complete 48-scene/240-slot checkpoint required')
    for i,(row,scene) in enumerate(zip(value['scenes'],selected)):
        need(set(row)=={'proposal_id','subject_id','split','original','files','current_raw'} and
             all(row[k]==scene[k] for k in ('proposal_id','subject_id','split')) and
             row['original']=={k:short(scene[k]) for k in SCENE_ROLES}, 'Ordered original checkpoint membership changed')
        need(set(row['files'])=={'preparation_config','prepared_complete','patient_receipt','physics_profile'} and
             set(row['current_raw'])=={'prepared_complete','layered_runtime_equivalence'}, 'Closed prepared/current-RAW roles required')
        for role,name in (('preparation_config','prepare.json'),('prepared_complete','COMPLETE.json'),('patient_receipt','patient_receipt.json'),('physics_profile','physics_profile.json')):
            ref=row['files'][role];registered(ref)
            need(ref['path']==str(root/'derived'/('scene_'+str(i).zfill(4))/name), 'Exact saved authoring JSON role required')
        for role,name in (('prepared_complete','COMPLETE.json'),('layered_runtime_equivalence','layered_runtime_equivalence.json')):
            ref=row['current_raw'][role];registered(ref)
            need(ref['path']==str(root/'current_raw'/str(i).zfill(3)/name), 'Exact saved current-RAW JSON role required')
    for i,(row,a) in enumerate(zip(value['plans'],allocated)):
        need(set(row)=={'slot_id','proposal_id','subject_id','split','target','original','derived','baseline_config','baseline_provenance'} and
             all(row[k]==a[k] for k in ('slot_id','proposal_id','subject_id','split','target')) and row['original']==short(a['fresh_plan']),
             'Ordered fixed checkpoint allocation changed')
        for role,directory,suffix in (('derived','plans','.json'),('baseline_config','baseline','.json'),('baseline_provenance','baseline','.provenance.json')):
            ref=row[role];registered(ref)
            need(ref['path']==str(root/directory/(str(i+1).zfill(3)+suffix)), 'Exact saved baseline/rebound-plan role required')
    # Existence of any issued plan/native roots invalidates this capability.
    need(not any((root/p).exists() for p in ('episodes','DERIVATION_MAP.json','CONFIG_INVENTORY.json','first100.batch.json')),
         'Checkpoint is not an unlaunched completed-authoring failure')
    return inventory


def validate_plan_headers(value,entries,assembly_root,*,bind=None):
    """Bind every assembly row to its exact ordered saved checkpoint row.

    This pass only checks declarations and file headers. No plan leaf or opaque
    authoring payload is read before the last row has been authenticated.
    """
    need(type(entries) is list and len(entries)==len(value['plans'])==240 and
         str(assembly_root)==value['assembly_root'], 'Complete checkpoint report plan linkage required')
    for i,(entry,saved) in enumerate(zip(entries,value['plans'])):
        need(type(entry) is dict and all(scene_store._exact_json(entry.get(key),saved[key]) for key in
             ('slot_id','proposal_id','original','baseline_config','baseline_provenance')),
             'Report plan differs from its exact ordered checkpoint baseline/provenance/original row')
        copied=dict(path=str(Path(assembly_root)/'plans'/(str(i+1).zfill(3)+'.json')),
            bytes=saved['derived']['bytes'],sha256=saved['derived']['sha256'])
        need(entry.get('derived')==short(copied), 'Copied plan must preserve checkpoint SHA and exact new assembly path')
        header(copied)
        if bind is not None:need(bind(copied)==copied, 'Copied plan bytes/SHA differ from checkpoint retained role')


def validate_saved_plans(value,allocated,read_json=None):
    """Authenticate all saved JSON leaves and the sole patient-binding change."""
    read_json=_metadata_json if read_json is None else read_json
    scenes={row['proposal_id']:row for row in value['scenes']}
    for row,a in zip(value['plans'],allocated):
        original=read_json(row['original']);derived=read_json(row['derived']);scene=scenes[row['proposal_id']]
        need(type(original) is dict and type(derived) is dict and
             type(original.get('patient_provenance')) is dict and type(derived.get('patient_provenance')) is dict,
             'Saved checkpoint plans require exact patient provenance')
        restored=deepcopy(derived);restored['patient_provenance']['binding']=original['patient_provenance'].get('binding')
        need(scene_store._exact_json(restored,original) and original.get('success') is True and original.get('name')==a['target'] and
             original['patient_provenance'].get('identity')==a['subject_id'] and
             original['patient_provenance'].get('binding')==scene['original']['patient_receipt'] and
             derived['patient_provenance'].get('binding')==short(scene['files']['patient_receipt']),
             'Saved checkpoint rebound plan changed numerical/source/patient provenance')


def _metadata_json(ref):
    path=canonical(ref.get('path',ref.get('file')))
    need(path.suffix=='.json' and path.name!='collision_geometry.json', 'Checkpoint reconstruction reads metadata JSON only')
    before=signature(path);raw=path.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==ref['sha256'] and ('bytes' not in ref or len(raw)==ref['bytes']) and
         signature(path)==before, 'Checkpoint JSON reference changed')
    return scene_store._strict_json(raw)


def validate_copied_plans(value,entries,*,read_json=None):
    """Read exact copied JSON only after the complete ordered header pass."""
    read_json=_metadata_json if read_json is None else read_json
    for entry,saved in zip(entries,value['plans']):
        copied=dict(path=entry['derived']['file'],bytes=saved['derived']['bytes'],sha256=saved['derived']['sha256'])
        need(scene_store._exact_json(read_json(copied),read_json(saved['derived'])),
             'Copied assembly plan changed saved checkpoint numerical/source provenance')


def reconstruct(value,selected,allocated,assembly_root,*,bind=None,read_json=None,payloads=True):
    """Reconstruct actual original provider output without calling its issuer."""
    validate_headers(value,selected,allocated,assembly_root,bind=bind)
    authenticate_inventory(value['retained_roots'],value['retained_inputs'],[assembly_root],payloads=False,bind=bind)
    read_json=_metadata_json if read_json is None else read_json
    validate_saved_plans(value,allocated,read_json)
    paths=capture_paths(value['capture_paths'],pinned=True,bind=bind,read_json=read_json)
    b=value['bindings'];negative=read_json(b['failed_negative'])
    need(type(negative) is dict and set(negative)=={'schema','retained_roots','retained_inputs','retained_file_count','total_retained_bytes',
         'physical_denominator','model_denominator','physical_launched_count','valid_plan_issued','remote_operator_exit_status','test_payloads_opened','training_eligible'} and
         negative.get('schema')==NEGATIVE_SCHEMA and negative.get('retained_roots')==REQUIRED_RETAINED_ROOTS and
         negative.get('retained_inputs')==[r for r in value['retained_inputs'] if r['reference']!=b['failed_negative']] and
         type(negative.get('physical_denominator')) is int and negative['physical_denominator']==0 and
         type(negative.get('model_denominator')) is int and negative['model_denominator']==0 and
         type(negative.get('physical_launched_count')) is int and negative['physical_launched_count']==0 and negative.get('valid_plan_issued') is False and
         type(negative.get('remote_operator_exit_status')) is int and negative['remote_operator_exit_status']==1 and negative.get('test_payloads_opened') is False and
         negative.get('training_eligible') is False, 'Exact complete failed V3 negative inventory required')
    need(type(negative.get('retained_file_count')) is int and negative['retained_file_count']==len(negative['retained_inputs']) and
         type(negative.get('total_retained_bytes')) is int and negative['total_retained_bytes']==sum(r['reference']['bytes'] for r in negative['retained_inputs']),
         'Complete negative cost totals required')
    sdk=read_json(b['sdk_receipt']);receipt=read_json(b['scene_receipt']);request=sdk['authoring_request']
    settings=read_json(b['original_settings'])
    need(not any((Path(settings['base'])/'runs/ausc_policy'/run).exists() for run in settings['run_ids'].values()) and
         not (Path(settings['metadata_root'])/'plan.json').exists() and not Path(settings['seal_path']).exists(),
         'Original V3 native jobs/plan/seal must never have been issued')
    need(set(sdk)=={'schema','authoring_request','cpu_execution','scene_authoring_receipt','proofs'} and
         sdk['schema']=='m2_given_lossless_sdk_cpu_receipt_v1' and sdk['scene_authoring_receipt']==full_file(b['scene_receipt']) and
         request['raw_selected']==[row['original'] for row in value['scenes']] and len(request['selected'])==48 and
         request['output_root']==str(Path(SOURCE_ROOT)/'derived') and request['current_raw_output_root']==str(Path(SOURCE_ROOT)/'current_raw') and
         request['selected']==receipt['original_selected'] and len(receipt['scenes'])==len(receipt['input_bindings'])==len(receipt['iteration_proofs'])==len(sdk['proofs'])==48,
         'Original author request/receipt complete ordered population changed')
    need(sdk['cpu_execution']['request_sha256']==hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
         'Original historical request digest changed')
    prepared=[];current=[]
    for i,(row,item,proof,inputs,selection) in enumerate(zip(value['scenes'],receipt['scenes'],receipt['iteration_proofs'],receipt['input_bindings'],request['selected'])):
        f=row['files'];prepared.append(dict({k:short(v) for k,v in f.items()},original=selection,input_bindings=inputs,
            dependency_inventory=item['dependency_inventory'],baseline_scene_sha256=item['original']['sha256'],iteration_proof=proof,
            composed_semantics_sha256=item['composed_semantics_sha256'],scene_authoring_receipt=full_file(b['scene_receipt'])))
        current.append(dict(historical_prepared_complete=full_file(f['prepared_complete']),
            layered_runtime_equivalence=full_file(row['current_raw']['layered_runtime_equivalence']),prepared_complete=short(row['current_raw']['prepared_complete'])))
    authored=dict(schema=receipt['schema'],base=receipt['base'],prepared=prepared,receipt=full_file(b['scene_receipt']),
        original_bytes=receipt['original_bytes'],saved_bytes=receipt['saved_bytes'],static_subtrees=receipt['static_subtrees'],runtime_equivalence=sdk['proofs'])
    result=dict(authored=authored,current_raw_prepared=current,authoring_request=request,cpu_execution=sdk['cpu_execution'],
        runtime_equivalence_receipt=b['sdk_receipt'],capture_paths=paths)
    if payloads:
        from scripts.experiments.wam_runpod_20261027 import given_formal_collection_lossless_v1 as op
        op._checkpoint_metadata(value,selected,allocated,result)
        authenticate_inventory(value['retained_roots'],value['retained_inputs'],[assembly_root],payloads=True,bind=bind)
    return result


def _original_population():
    from scripts.experiments.wam_runpod_20261027 import given_formal_collection_lossless_v1 as op
    binding=dict(path=PINNED_PATHS['original_settings'],bytes=PINNED['original_settings'][0],sha256=PINNED['original_settings'][1])
    need(reference(binding['path'])==binding, 'Exact original V3 settings required')
    settings=scene_store._strict_json(Path(binding['path']).read_bytes())
    mapping=op.data._allocation(settings,Path.cwd());selected=op._selected_scenes(mapping)
    allocated=[dict(a,split=s) for s in ('train','validation') for a in mapping['ordered_allocations'][s]]
    need(len(selected)==48 and len(allocated)==240, 'Exact original population required')
    # This is deliberately before any retained USD/NPZ hashing in either issuer.
    op._original_headers(mapping,allocated)
    return settings,selected,allocated


def _write_new(path,value):
    import os
    path=canonical(str(path));need(not path.exists(), 'New immutable checkpoint/negative path required')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    path.chmod(0o444)
    return reference(path)


def issue_negative(output):
    _original_population();rows=[]
    # Inspect the complete tree first, including unknown small diagnostics.
    for row in REQUIRED_RETAINED_ROOTS:
        root=canonical(row['path'])
        paths=[root] if row['kind']=='file' else root.rglob('*')
        for path in paths:
            st=path.lstat();need(not stat.S_ISLNK(st.st_mode), 'Retained symbolic link forbidden')
            if stat.S_ISDIR(st.st_mode):continue
            need(stat.S_ISREG(st.st_mode) and st.st_nlink==1, 'Retained special/linked file forbidden')
            rows.append(dict(category=row['category'],reference=dict(path=str(path),bytes=st.st_size,sha256='0'*64)))
    rows.sort(key=lambda r:r['reference']['path'])
    authenticate_inventory(REQUIRED_RETAINED_ROOTS,rows,[output,ASSEMBLY_ROOT],payloads=False)
    for row in rows:row['reference']['sha256']=sha(row['reference']['path'])
    authenticate_inventory(REQUIRED_RETAINED_ROOTS,rows,[output,ASSEMBLY_ROOT])
    value=dict(schema=NEGATIVE_SCHEMA,retained_roots=deepcopy(REQUIRED_RETAINED_ROOTS),retained_inputs=rows,
        retained_file_count=len(rows),total_retained_bytes=sum(r['reference']['bytes'] for r in rows),
        physical_denominator=0,model_denominator=0,physical_launched_count=0,valid_plan_issued=False,
        remote_operator_exit_status=1,test_payloads_opened=False,training_eligible=False)
    return _write_new(output,value)


def issue_checkpoint(negative_ref,output):
    _,selected,allocated=_original_population()
    need(reference(negative_ref['path'])==negative_ref, 'Bound ROOT-issued negative required')
    negative=scene_store._strict_json(Path(negative_ref['path']).read_bytes())
    need(negative.get('schema')==NEGATIVE_SCHEMA and negative.get('retained_roots')==REQUIRED_RETAINED_ROOTS, 'Exact negative inventory required')
    roots=[*deepcopy(REQUIRED_RETAINED_ROOTS),dict(path=negative_ref['path'],category='prepared',kind='file')]
    rows=sorted([*deepcopy(negative['retained_inputs']),dict(category='prepared',reference=negative_ref)],key=lambda r:r['reference']['path'])
    authenticate_inventory(roots,rows,[output,ASSEMBLY_ROOT],payloads=False)
    refs={r['reference']['path']:r['reference'] for r in rows}
    bindings={k:deepcopy(refs[path]) for k,path in PINNED_PATHS.items()};bindings['failed_negative']=deepcopy(negative_ref)
    scenes=[];plans=[];root=Path(SOURCE_ROOT)
    for i,scene in enumerate(selected):
        directory=root/'derived'/('scene_'+str(i).zfill(4));current=root/'current_raw'/str(i).zfill(3)
        scenes.append(dict({k:scene[k] for k in ('proposal_id','subject_id','split')},original={k:short(scene[k]) for k in SCENE_ROLES},
            files={k:deepcopy(refs[str(directory/name)]) for k,name in
                (('preparation_config','prepare.json'),('prepared_complete','COMPLETE.json'),('patient_receipt','patient_receipt.json'),('physics_profile','physics_profile.json'))},
            current_raw={k:deepcopy(refs[str(current/name)]) for k,name in
                (('prepared_complete','COMPLETE.json'),('layered_runtime_equivalence','layered_runtime_equivalence.json'))}))
    for i,a in enumerate(allocated):
        n=str(i+1).zfill(3)
        plans.append(dict({k:a[k] for k in ('slot_id','proposal_id','subject_id','split','target')},original=short(a['fresh_plan']),
            derived=deepcopy(refs[str(root/'plans'/(n+'.json'))]),baseline_config=deepcopy(refs[str(root/'baseline'/(n+'.json'))]),
            baseline_provenance=deepcopy(refs[str(root/'baseline'/(n+'.provenance.json'))])))
    captures={name:dict(default=deepcopy(refs[str(Path(CORE_ROOT)/relative)]),qualified=deepcopy(refs[str(Path(QUALIFIED_ROOT)/relative)])) for name,(relative,_,_) in CAPTURE_ROLES.items()}
    value=dict(schema=SCHEMA,source_root=SOURCE_ROOT,assembly_root=ASSEMBLY_ROOT,bindings=bindings,scenes=scenes,plans=plans,
        retained_roots=roots,retained_inputs=rows,capture_paths=captures,physical_launched_count=0,physical_denominator=0,model_denominator=0,
        test_payloads_opened=False,training_eligible=False)
    reconstruct(value,selected,allocated,ASSEMBLY_ROOT)
    return _write_new(output,value)


def main():
    import argparse,os,sys
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('negative','checkpoint'))
    parser.add_argument('--output',required=True);parser.add_argument('--negative');parser.add_argument('--negative-sha256');args=parser.parse_args()
    need(sys.executable=='/opt/wam-runtime/envs/isaacsim60/bin/python' and sys.dont_write_bytecode and not sys.flags.optimize and
         os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Native60 CPU-only -B unoptimized issuer required')
    if args.mode=='negative':result=issue_negative(args.output)
    else:
        need(args.negative is not None and args.negative_sha256 is not None, 'Checkpoint requires exact negative path/SHA')
        binding=reference(args.negative);need(binding['sha256']==args.negative_sha256, 'ROOT-issued negative hash changed')
        result=issue_checkpoint(binding,args.output)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
