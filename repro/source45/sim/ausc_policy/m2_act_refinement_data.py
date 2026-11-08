"""Typed TRAIN supervision, retaining raw setters and original normalization.

Enrichment re-audits existing Source37 captures. On-policy collection is a
separate data producer and can never constitute ACT benchmark success.
"""
from copy import deepcopy
from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import numpy as np

from . import m2_given_contact_runtime as rt, m2_given_contact_dataset as data
from . import m2_given_contact_act as given, m2_measured_dart_data as dart
from . import m2_given_contact_recovery_dataset as roles
from . import m2_measured_recovery_data as query, m2_act_refinement_warm_start as warm
from .m2_given_contact_intent import validate_native_intents, raw_cell

need=rt.need
ARRAY_KEYS=dart.ARRAY_KEYS
ENRICHED_SCHEMA='m2_act_refinement_enriched_dart_bank_v1'
ONPOLICY_SCHEMA='m2_act_refinement_onpolicy_teacher_bank_v1'
ENRICHED_PROFILE='source37_all_postperturb_completed_teacher_v1'
ONPOLICY_PROFILE='source41_live_train_teacher_intervention_v1'
ROLE_SCHEMA='m2_refinement_completed_teacher_roles_v1'
_B='/opt/wam-runtime/fast-wam'
SOURCE37_QUALIFICATION=dict(path=_B+'/diagnostics/act10-dart-source37-pilot-qualification-20261107-v3/COMPLETE.json',
    bytes=6052,sha256='0e138c4da506b310e5758c27a675f5a12eb54ae8ae670c84e4c91a426d57ad69')
SOURCE37_BANK=dict(path=_B+'/datasets/act10-source37-measured-dart-bank-20261107-v3/COMPLETE.json',
    bytes=1462,sha256='4b3175b8dbdf9927504855f85fb26f2994d5392ade72565944767366d135ad20')
SOURCE41_CONSUMER=dict(path=_B+'/datasets/act10-diverse-v8-dart37-20261107-v1/COMPLETE.json',
    bytes=1442,sha256='fac4f86802c6e472e43792aedea3a5713ddda9df34a2ba937311376443b38fab')
SOURCE40=dict(root='/opt/wam-runtime/code/wam-act-recovery-20261106-v40',code_commit='95f8ef6ece094666875c3f041ae129a487eda397',
    files=1699,manifest=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v40/SOURCE_MANIFEST.json',bytes=294116,
        sha256='99e2b558687791fe8669407d59c1f7a14481fc43071128997b3c5464c1b81087'))
SDK_CONFIGURATION=dict(path=_B+'/datasets/rebuilt-expert-capture-20261105-v2-prepared/configs/p001120-p000474_a001230_f0000-r3.json',
    bytes=108923,sha256='618a99561839ffe9e88845eb6a7876f2baa539ee91e2fbe77e50bd83a8af2294')
TRANSPORT_SCHEMA='m2_act_refinement_original40_transport_v1'
QUERY_SCHEMA='m2_act_refinement_original41_query32_consumer_v1'
QUERY_PROFILE='unchanged_source32_query58_original41_reader_current_consumer_v1'
QUERY_TRANSPORT_SCHEMA='m2_act_refinement_original41_query32_transport_v1'
QUERY_EXIT_SCHEMA='m2_act_refinement_original41_query32_reader_exit_v1'
SOURCE41_QUERY32=dict(path=_B+'/datasets/act10-diverse-v8-query32-20261107-v1/COMPLETE.json',
    bytes=2809,sha256='579b0fefa027509c4077040429b3c32a02624c2a4b33f291d4de0a48ea6b4319')

NORMALIZATION=deepcopy(dart.NORMALIZATION)


def completed_roles(audit,events,*,boundary_row):
    """Bind every role to the actual setter hash and completed tick interval."""
    need(type(boundary_row) is int and boundary_row>=12 and boundary_row%12==0 and boundary_row<len(audit),
        'Certified grid boundary required')
    actual=[];ends={}
    for e in audit:
        raw=e.get('raw_intent')
        if raw is not None:
            s=raw['setter'];ends[s['intent_id']]=e['physics_row']+1
            if raw['native_tick_in_cell']==0:actual.append(s)
    need(isinstance(events,list) and len(events)==len(actual),'Complete actual setter role ledger required')
    by_row={}
    for e,s in zip(events,actual):
        start=s['physics_row_start'];end=ends[s['intent_id']]
        need(set(e)=={'intent_id','setter_sha256','physics_row_start','physics_row_end_exclusive','kind','completed'} and
            e['kind'] in ('nominal_teacher','recovery_teacher','perturbation','learner_prefix','excluded') and
            type(e['completed']) is bool and e['intent_id']==s['intent_id'] and e['setter_sha256']==s['setter_sha256'] and
            e['physics_row_start']==start and e['physics_row_end_exclusive']==end and end>start and
            e['completed']==(end-start==12),'Role proof differs from actual completed native setter')
        if e['kind'] in ('nominal_teacher','recovery_teacher'):
            need(s['control_owner']=='TEACHER_L2' and s['execution_role']=='given_l2_motion' and not s['offgrid_stop'],
                'Only actual teacher motion can provide labels')
        if e['kind']=='learner_prefix':need(start<boundary_row and s['control_owner']=='ACT_L2','Learner attribution differs')
        by_row[start]=e
    return by_row


def teacher_windows(records,audit,metadata,*,boundary_row,role_proof):
    """Real H2 may precede takeover; every future cell must be completed teacher.

    The shared assembler performs all global state/clock/native checks. This
    typed assembler then uses the original records directly, without changing
    owners or normalizing history roles to make a legacy window pass.
    """
    need(metadata.get('split')=='train','Refinement teacher data is TRAIN only')
    ref=rt.pose.require_route(metadata);records=list(records)
    validate_native_intents(audit,metadata)
    need(isinstance(role_proof,dict) and set(role_proof)=={'schema','boundary_row','events'} and
        role_proof.get('schema')==ROLE_SCHEMA and role_proof.get('boundary_row')==boundary_row,'Typed certified role proof required')
    by_row=completed_roles(audit,role_proof['events'],boundary_row=boundary_row)
    # This is validation and numerical assembly on unmodified inputs, not a
    # relabelled copy of the audit. It intentionally retains the old API.
    checked=rt.shared.assemble_samples(records,audit,metadata,phase_source=rt.pose.NATIVE_SOURCE,
        route=rt.pose.TASK_ID,allow_truncated=True)
    for i,row in enumerate(records):
        need(rt.shared._scalar(row.get('target_estimate_source_id'))=='given-contact-'+ref.sha256,
            'Given-reference observation source mismatch')
        if 'q_command' in row:
            endpoint=12*(i+2)-1
            need(endpoint<len(audit) and np.array_equal(row['q_command'],audit[endpoint]['q_command']),
                'Actual native applied endpoint differs')
    out={k:[] for k in ARRAY_KEYS};windows=[]
    for start in range(1,len(records)-1):
        first=12*(start+1)
        if first<boundary_row:continue
        origin_event=audit[first-1];actor=origin_event['controller_after']['actor'];row=records[start]
        # History consists of two actual L2 observations. Earlier actual owners
        # may be ACT or perturbation, but contexts and clocks remain causal.
        history_span=audit[first-13:first]
        if any(e['actual_control_owner'] not in ('ACT_L2','TEACHER_L2') for e in history_span):continue
        phase=actor['phase']
        valid_history=True
        for e in history_span:
            for side in ('controller_before','controller_after'):
                c=e[side];a=c['actor']
                if (c.get('control_owner','L2')!='L2' or c['phase']!=rt.shared._scalar(row['phase']) or
                    c['target']!=metadata['target_id'] or c['active'] or c['recovery'] or a['owner']!='L2' or
                    a['phase']!=phase or c['subphase'] not in ('given_l2_motion','given_l2_history_wait')):
                    valid_history=False
        if not valid_history:continue
        cells=[]
        for k in range(min(20,len(records)-start-1)):
            cell_start=first+12*k;e=by_row.get(cell_start)
            if e is None or e['kind'] not in ('nominal_teacher','recovery_teacher') or not e['completed']:break
            span=audit[cell_start:cell_start+12]
            if len(span)!=12 or any(x['actual_control_owner']!='TEACHER_L2' for x in span):break
            if any(not rt.shared._context_matches(x,row,metadata,actor,'given_l2_motion') for x in span):break
            value=raw_cell(audit,cell_start,metadata)
            if value is None:break
            cells.append(value)
        if not cells:continue
        history=np.asarray([records[start-1]['state17'],row['state17']],dtype=np.float64)
        origin=history[-1,:7].copy();action=np.zeros((20,7),dtype=np.float64)
        action[:len(cells)]=np.asarray(cells)-origin
        for key,value in dict(states=history[-1],histories=history,actions=action,origins=origin,
            velocities=(history[1,:7]-history[0,:7])/.05,observation_times=float(row['observed_at_s']),
            action_is_pad=np.arange(20)>=len(cells)).items():out[key].append(value)
        windows.append(dict(observation_indices=[start-1,start],action_record_indices=list(range(start,start+len(cells))),
            inclusive_native_ticks=[first-1,first+12*len(cells)-1],history_native_ticks=[first-13,first-1],
            phase=phase,execution_role='given_l2_motion',phase_source=rt.pose.NATIVE_SOURCE,
            valid_future_cells=len(cells),label_protocol=rt.shared.MASKED_LABEL_PROTOCOL,
            action_semantics_id=given.ACTION_SEMANTICS_ID,actual_control_owner='TEACHER_L2',
            raw_intent_native_ticks=[first,first+12*len(cells)-1],future_label_times='cell end t+(k+1)*.05',
            label_role='actual_clean_recovery_teacher_raw_intents',
            history_actual_owners=[audit[first-13]['actual_control_owner'],audit[first-1]['actual_control_owner']]))
    shapes=dict(states=(-1,17),histories=(-1,2,17),actions=(-1,20,7),origins=(-1,7),
        velocities=(-1,7),observation_times=(-1,),action_is_pad=(-1,20))
    for k in ARRAY_KEYS:out[k]=np.asarray(out[k],dtype=bool if k=='action_is_pad' else np.float64).reshape(shapes[k])
    out.update(schema=given.MASKED_DATASET_SCHEMA,task_id=rt.pose.TASK_ID,observation_contract=rt.pose.observation_contract(),
        reference_sha256=ref.sha256,action_semantics_id=given.ACTION_SEMANTICS_ID,label_protocol=rt.shared.MASKED_LABEL_PROTOCOL,
        derived_index=dict(checked['derived_index'],windows=windows),training_eligible=False,physical_qualification=False,
        normalization_reference=deepcopy(NORMALIZATION))
    dart.validate_samples(out)
    return out


def parent_subsequence(parent,enriched):
    """Prove ordered original samples, all seven arrays and window semantics."""
    old=parent['derived_index']['windows'];new=enriched['derived_index']['windows'];mapping=[];cursor=0
    for i,w in enumerate(old):
        found=None
        for j in range(cursor,len(new)):
            if all(new[j].get(k)==v for k,v in w.items()) and all(np.array_equal(parent[k][i],enriched[k][j]) for k in ARRAY_KEYS):
                found=j;break
        need(found is not None,'Original parent arrays/mask/window is not an exact ordered subsequence')
        mapping.append(found);cursor=found+1
    return mapping


def _qualification(qualification_ref,parent_bank_ref):
    need(qualification_ref==SOURCE37_QUALIFICATION and parent_bank_ref==SOURCE37_BANK,
        'Exact original Source37 qualification and parent bank required')
    value=rt.read_bound(qualification_ref)
    need(value.get('schema')=='act10_source40_independent_dart_pilot_v1' and value.get('complete') is True and
        value.get('bank_qualified') is True and value.get('bank')==parent_bank_ref and value.get('windows')==274 and
        value.get('planned')==value.get('fixed_denominator')==20 and value.get('prepared')==17 and
        value.get('seven_arrays_masks_loaded') is True and value.get('optimization_admission',{}).get('qualified_episodes')==17 and
        value['optimization_admission'].get('pilot_optimization_eligible') is True,
        'Original all17 Source37 physical qualification required')
    return value


def _regular_unaliased(path):
    path=Path(path)
    need(path.is_absolute() and all(not p.is_symlink() for p in (path,*path.parents)),
        'Canonical regular historical reader path required, no aliases')
    info=path.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'Historical source hardlinks/nonregular files forbidden')


def _authenticate_original40(qualification):
    """Rehash all1699 original files before and after every finite child."""
    reader=qualification.get('reader_source')
    need(reader==SOURCE40,'Exact pinned original40 reader descriptor required')
    root=Path(reader['root']);need(root.resolve()==root,'Canonical original40 root required')
    manifest=reader['manifest'];_regular_unaliased(manifest['path']);rows=rt.read_bound(manifest)
    commit=root/'CODE_COMMIT';_regular_unaliased(commit)
    need(commit.read_text()==reader['code_commit']+'\n' and isinstance(rows,list) and len(rows)==reader['files'],
        'Original40 commit/full1699 manifest differs')
    names=set()
    for row in rows:
        name=row.get('file');relative=Path(name) if isinstance(name,str) else None
        need(set(row)=={'file','bytes','sha256'} and relative is not None and name==relative.as_posix() and
            not relative.is_absolute() and relative.parts and '..' not in relative.parts and name not in names and
            name not in ('SOURCE_MANIFEST.json','CODE_COMMIT'),'Unique bounded original40 inventory path required')
        path=root/relative;_regular_unaliased(path);rt.bound_path(dict(path=str(path),bytes=row['bytes'],sha256=row['sha256']));names.add(name)
    actual=set()
    for path in root.rglob('*'):
        need(not path.is_symlink(),'Original40 archive symlink forbidden');mode=path.lstat().st_mode
        need(stat.S_ISDIR(mode) or stat.S_ISREG(mode),'Unknown original40 archive file type')
        if stat.S_ISREG(mode):_regular_unaliased(path);actual.add(path.relative_to(root).as_posix())
    need(actual==names|{'SOURCE_MANIFEST.json','CODE_COMMIT'},'Unknown/omitted original40 archive files')
    return deepcopy(reader)


# This finite CPU process imports exclusively the authenticated original40
# package. Its original full kernel remains unchanged, including legacy BOUNDS.
_ORIGINAL40_PROGRAM=r'''import gzip,json,os,sys
from pathlib import Path
payload=json.load(sys.stdin);root=Path(payload['physical_reader']['root']);sys.path.insert(0,str(root))
from sim.ausc_policy import m2_given_contact_runtime as rt,m2_measured_dart_data as dart,m2_given_contact_dataset as data
assert all(Path(m.__file__).resolve().is_relative_to(root) for m in (rt,dart,data))
assert sys.executable==payload['python'] and sys.prefix=='/opt/wam-runtime/envs/isaacsim60'
assert sys.flags.isolated and sys.dont_write_bytecode and os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert 'AUSC_OWNED_RUN' not in os.environ
manifest=rt.read_bound(payload['manifest']);base=Path(payload['manifest']['path']).parent
attempts,errors,precollection=dart._headers(manifest,base)
parent,meta=dart.load_arrays(payload['parent_bank'])
assert len(parent['states'])==274 and len(attempts)==20 and sum(s['status']=='prepared' for s in precollection['slots'])==17
assert meta['source_manifest']==payload['manifest'] and meta['profile']==dart.PROFILE_SOURCE37
qualified=[e['request']['slot_id'] for e in meta['attempt_ledger'] if e['status']=='qualified']
assert len(qualified)==17 and qualified==payload['qualified_slots']
out=Path(payload['output']);out.mkdir(parents=True,exist_ok=False);sources=[];cursor=0
for attempt,old in zip(attempts,meta['attempt_ledger']):
    if old['status']!='qualified':continue
    assert attempt['request']['slot_id'] not in errors
    samples,receipt,metadata=dart.read_episode(attempt,base,manifest)
    count=len(samples['states']);assert count==old['windows'] and count>0
    for key in dart.ARRAY_KEYS:assert dart.measured.array_proof(samples[key])==dart.measured.array_proof(parent[key][cursor:cursor+count])
    assert samples['derived_index']['windows']==parent['derived_index']['windows'][cursor:cursor+count]
    cursor+=count;row=attempt['episode'];context=data._metadata(row,base)
    outer=rt.shared._json(context[0]/'receipt.json')
    audit=data.read_native_capture_audit(context[0],outer,transport=row['capture_audit'])
    records=[{k:v for k,v in r.items() if k not in ('rgb','depth_q','depth_code','depth_valid')} for r in data.iter_training_records(context[0]/'compact')]
    sidecar=rt.read_bound(receipt['recovery_collection'])
    body=dict(request=attempt['request'],declaration=row,records=records,audit=audit,metadata=metadata,receipt=receipt,sidecar=sidecar,
        original_samples=dict(**{k:samples[k].tolist() for k in dart.ARRAY_KEYS},derived_index=samples['derived_index']))
    path=out/(attempt['request']['slot_id']+'.json.gz')
    with gzip.open(path,'xt') as stream:json.dump(body,stream,sort_keys=True,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else x)
    sources.append(dict(slot_id=attempt['request']['slot_id'],payload=rt.reference(path)))
assert cursor==274 and len(sources)==17
assert not any(n=='torch' or n.startswith(('torch.','isaacsim','omni.')) for n in sys.modules)
import platform,ruckig,numpy,scipy
from sim.ausc_policy import m2_teacher_stop_reader as teacher
assert all(Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()) for m in (numpy,scipy,ruckig))
fingerprint=dict(executable=sys.executable,python=rt.reference(sys.executable),prefix=sys.prefix,python_version=platform.python_version(),
    numpy_version=numpy.__version__,numpy=rt.reference(numpy.__file__),scipy_version=scipy.__version__,scipy=rt.reference(scipy.__file__),
    ruckig_version=ruckig.__version__,ruckig=rt.reference(ruckig.__file__),numeric_engine_sha256=teacher.numeric_engine_fingerprint(),
    isolated=True,bytecode_disabled=True,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
    ld_library_path=os.environ.get('LD_LIBRARY_PATH'),no_owner='AUSC_OWNED_RUN' not in os.environ,no_torch_world=True)
rt.write_new(out/'COMPLETE.json',dict(schema=payload['schema'],complete=True,manifest=payload['manifest'],parent_bank=payload['parent_bank'],
    qualification=payload['qualification'],physical_reader=payload['physical_reader'],fixed_denominator=20,
    qualified_slots=qualified,sources=sources,windows=cursor,full_original_physical_kernel=True,processing_environment=fingerprint))
'''


def _original40_transport(manifest_ref,parent_bank_ref,qualification_ref,output):
    qualification=_qualification(qualification_ref,parent_bank_ref);reader=_authenticate_original40(qualification)
    need(qualification['manifest']==manifest_ref,'Original40 source manifest differs')
    output=Path(output).resolve();need(not output.exists(),'Fresh Linux-only original40 transport output required')
    need(sys.platform=='linux','Actual raw Original40 transport stays Linux only')
    original=rt.read_bound(SDK_CONFIGURATION);ld=original.get('ld_library_path')
    need(isinstance(ld,str) and ld,'Authenticated original Native SDK LD_LIBRARY_PATH required verbatim')
    # Hash-bound metadata determines all17, never a caller-selected subset.
    parent=rt.read_bound(rt.read_bound(parent_bank_ref)['metadata'])
    slots=[e['request']['slot_id'] for e in parent['attempt_ledger'] if e['status']=='qualified']
    need(len(slots)==17,'All17 original physically qualified captures required')
    output.mkdir(parents=True);transport=output/'payloads';payload=dict(schema=TRANSPORT_SCHEMA,manifest=manifest_ref,parent_bank=parent_bank_ref,
        qualification=qualification_ref,physical_reader=reader,qualified_slots=slots,output=str(transport),python=recovery_python())
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMNI_KIT_ACCEPT_EULA='YES',LD_LIBRARY_PATH=ld,
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    for key in ('AUSC_OWNED_RUN','PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','HF_TOKEN'):environment.pop(key,None)
    code=None;error=None;archive_after=False
    try:
        with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
            result=subprocess.run([payload['python'],'-I','-B','-c',_ORIGINAL40_PROGRAM],input=json.dumps(payload).encode(),
                stdout=stdout,stderr=stderr,env=environment,cwd=reader['root'],check=False)
        code=result.returncode;need(code==0,'Original40 full physical reader failed; labels not admitted')
    except BaseException as caught:
        error=type(caught).__name__+': '+str(caught)
        raise
    finally:
        try:archive_after=_authenticate_original40(qualification)==reader
        finally:
            receipt=dict(schema='m2_act_refinement_original40_reader_exit_v1',physical_reader=reader,
                archive_authenticated_before=True,archive_authenticated_after=archive_after,exit_code=code,
                normal_exit=code==0 and error is None and archive_after,error=error,python=payload['python'],
                command_flags=['-I','-B'],sdk_configuration=SDK_CONFIGURATION,stdout=rt.reference(output/'stdout.log') if (output/'stdout.log').exists() else None,
                stderr=rt.reference(output/'stderr.log') if (output/'stderr.log').exists() else None)
            rt.write_new(output/'reader-exit.json',receipt)
            if not receipt['normal_exit'] and transport.exists():shutil.rmtree(transport)
    complete=rt.reference(transport/'COMPLETE.json')
    _read_original40_transport(complete,manifest_ref,slots)
    return complete,rt.reference(output/'reader-exit.json')


def recovery_python():
    return '/opt/wam-runtime/envs/isaacsim60/bin/python'


def _read_original40_transport(ref,manifest_ref,qualified_slots):
    value=rt.read_bound(ref)
    need(value.get('schema')==TRANSPORT_SCHEMA and value.get('complete') is True and value.get('manifest')==manifest_ref and
        value.get('parent_bank')==SOURCE37_BANK and value.get('qualification')==SOURCE37_QUALIFICATION and
        value.get('physical_reader')==SOURCE40 and value.get('fixed_denominator')==20 and value.get('windows')==274 and
        value.get('full_original_physical_kernel') is True and value.get('qualified_slots')==qualified_slots and len(qualified_slots)==17 and
        [s.get('slot_id') for s in value.get('sources',[])]==qualified_slots,'Exact all17 full original40 transport required')
    fingerprint=value.get('processing_environment',{})
    need(fingerprint.get('prefix')=='/opt/wam-runtime/envs/isaacsim60' and fingerprint.get('executable')==recovery_python() and fingerprint.get('python_version')=='3.12.15' and
        fingerprint.get('numpy_version')=='2.3.1' and fingerprint.get('numeric_engine_sha256')==dart.teacher.NUMERIC_ENGINE_SHA256 and
        fingerprint.get('isolated') is True and fingerprint.get('bytecode_disabled') is True and fingerprint.get('cuda_visible_devices')=='' and
        fingerprint.get('no_owner') is True and fingerprint.get('no_torch_world') is True and fingerprint.get('ruckig_version')=='0.15.3',
        'Original Native isolated CPU processing fingerprint required')
    prefix=Path(fingerprint['prefix']).resolve()
    for key in ('numpy','scipy','ruckig'):
        module_ref=fingerprint.get(key,{})
        need(Path(module_ref.get('path','')).is_relative_to(prefix),'Native CPU module origin outside actual environment')
        rt.bound_path(module_ref)
    rt.bound_path(fingerprint['python'])
    root=Path(ref['path']).parent
    need(set(p.name for p in root.iterdir())=={'COMPLETE.json',*(s['slot_id']+'.json.gz' for s in value['sources'])},
        'Unknown/omitted historical transport files')
    for source in value['sources']:
        path=Path(source['payload']['path']);need(path==root/(source['slot_id']+'.json.gz'),'Transport must remain in fresh Linux-only directory')
        _regular_unaliased(path);rt.bound_path(source['payload'])
    return value


def _historical_parent(parent_bank_ref,qualification_ref):
    """Only the pinned immutable Source40 reader's Source37 bank, no aliases.

    The legacy reader deliberately rejects changed current module bindings.
    Authenticate its original archive and outputs rather than dropping those
    bindings or relabelling historical bytes as current-reader production.
    """
    q=_qualification(qualification_ref,parent_bank_ref);reader=_authenticate_original40(q)
    complete=rt.read_bound(parent_bank_ref);meta=rt.read_bound(complete['metadata']);root=Path(reader['root'])
    inventory=rt.read_bound(reader['manifest'])
    warm_module=__import__(__package__+'.m2_act_refinement_warm_start',fromlist=['_regular_unaliased'])
    listed={r['file']:r for r in inventory}
    expected=dict(bank='m2_measured_dart_data.py',capture_decoder='m2_given_contact_dataset.py',
        role_filter='m2_given_contact_recovery_dataset.py',teacher_reader='m2_teacher_stop_reader.py',
        sample_validator='m2_measured_recovery_data.py',collector='m2_given_contact_recovery.py')
    need(set(meta.get('data_code',{}))==set(expected),'Complete original reader code bindings required')
    for key,name in expected.items():
        ref=meta['data_code'][key];relative='sim/ausc_policy/'+name;row=listed.get(relative,{})
        need(ref==dict(path=str(root/relative),bytes=row.get('bytes'),sha256=row.get('sha256')),
            'Original reader module path/bytes/hash differs')
        warm_module._regular_unaliased(Path(ref['path']));rt.bound_path(ref)
    need(complete.get('schema')==meta.get('schema')==dart.SCHEMA and complete.get('complete') is True and
        complete.get('profile')==meta.get('profile')==dart.PROFILE_SOURCE37 and
        meta.get('source_manifest')==q['manifest'] and meta.get('windows')==complete.get('windows')==q['windows']==274 and
        complete.get('normalization')==meta.get('normalization')==NORMALIZATION and meta.get('normalization_refitted') is False and
        meta.get('base_dataset')==dart.BASE29 and meta.get('base_counts')==dart.BASE_COUNTS and
        meta.get('fixed_denominator')==20 and meta.get('array_proofs')==q['array_proofs'] and
        complete.get('arrays')==meta.get('arrays') and complete.get('optimization_admission')==meta.get('optimization_admission')==q['optimization_admission'],
        'Original immutable Source37 parent graph/arrays/normalization differs')
    manifest=rt.read_bound(meta['source_manifest']);base=Path(meta['source_manifest']['path']).parent
    attempts,errors,precollection=dart._headers(manifest,base)
    need(meta.get('precollection_plan')==precollection and len(meta.get('attempt_ledger',[]))==20 and
        meta.get('optimization_admission')==dart.admission(meta['attempt_ledger'],precollection=precollection),
        'Original fixed20 ledger/prepared17 admission differs')
    sources=meta.get('sources');qualified=[e for e in meta['attempt_ledger'] if e['status']=='qualified']
    need(len(qualified)==17 and isinstance(sources,list) and [s['slot_id'] for s in sources]==[e['request']['slot_id'] for e in qualified],
        'Every original qualified capture must remain declared')
    for a,e in zip(attempts,meta['attempt_ledger']):
        need(all(e.get(k)==a[k] for k in ('request','episode','nominal_source_config','reason','evidence')) and
            (e['status']=='cpu_unavailable' if a['outcome']=='cpu_unavailable' else e['status']=='qualified') and
            (e['status']!='qualified' or e['request']['slot_id'] not in errors),'Original terminal outcome/owner/header differs')
    index=rt.read_bound(meta['index']);items=index.get('samples');cursor=0;resets=set()
    need(isinstance(items,list) and len(items)==274 and index.get('order_sha256')==rt.digest(items) and
        [w.get('sample_index') for w in items]==list(range(274)),'Original parent window order/index differs')
    for source,entry in zip(sources,qualified):
        n=source['windows'];reset=dart._reset(source['receipt'])
        need(source['declaration']==entry['episode'] and source['receipt']==entry['receipt'] and n==entry['windows'] and
            source['split_sample_range']==[cursor,cursor+n] and reset not in resets and
            all(w['slot_id']==source['slot_id'] for w in items[cursor:cursor+n]),'Original independent parent source ranges differ')
        resets.add(reset);cursor+=n
    need(cursor==274,'Original parent all-source window count differs')
    with np.load(rt.bound_path(complete['arrays']),allow_pickle=False) as archive:
        need(set(archive.files)==set(ARRAY_KEYS),'Exactly seven original parent arrays required');arrays={k:archive[k].copy() for k in ARRAY_KEYS}
    need({k:dart.measured.array_proof(v) for k,v in arrays.items()}==q['array_proofs'],'Original parent all7 raw/mask array bytes differ')
    arrays.update(schema=given.MASKED_DATASET_SCHEMA,task_id=rt.pose.TASK_ID,action_semantics_id=given.ACTION_SEMANTICS_ID,
        observation_contract=rt.pose.observation_contract(),label_protocol=rt.shared.MASKED_LABEL_PROTOCOL,
        derived_index=dict(windows=[w['window'] for w in items]))
    dart.validate_samples(arrays)
    return arrays,meta


def _read_enriched_episode(attempt,payload_ref,parent):
    path=rt.bound_path(payload_ref)
    with gzip.open(path,'rt') as stream:body=json.load(stream)
    need(body.get('request')==attempt['request'] and body.get('declaration')==attempt['episode'],
        'Original40 actual capture declaration/slot differs')
    audit=body['audit'];records=body['records'];meta=body['metadata'];receipt=body['receipt'];sidecar=body['sidecar']
    original={k:np.asarray(body['original_samples'][k],dtype=bool if k=='action_is_pad' else np.float64) for k in ARRAY_KEYS}
    original['derived_index']=body['original_samples']['derived_index']
    need(parent_subsequence(parent,original)==list(range(len(parent['states']))),'Original40 all7 decoded parent arrays/windows differ')
    roles.validate_sidecar(sidecar,audit,meta)
    boundary=[e for e in sidecar['events'] if e['kind']=='perturbation'][-1]['physics_row_end_exclusive']
    proof=dict(schema=ROLE_SCHEMA,boundary_row=boundary,events=sidecar['events'])
    samples=teacher_windows(records,audit,meta,boundary_row=boundary,role_proof=proof)
    mapping=parent_subsequence(parent,samples)
    receipt=dict(schema='m2_act_refinement_enriched_dart_receipt_v1',original_physical_receipt=deepcopy(receipt),
        native_reset_identity=deepcopy(receipt['native_reset_identity']),original_windows=len(parent['states']),windows=len(samples['states']),parent_sample_map=mapping,
        certified_boundary_row=boundary,new_native_capture=False,physical_reader=deepcopy(SOURCE40),label_reader=rt.reference(__file__),
        reaudited_transport=payload_ref)
    return samples,receipt,meta


def code_binding():
    return dict(refinement=rt.reference(__file__),decoder=rt.reference(data.__file__),
        collection=rt.reference(Path(__file__).with_name('m2_act_onpolicy_collection.py')),
        neutral_native_snapshots=rt.reference(Path(__file__).with_name('m2_given_contact_evaluation.py')))


def _write_bank(output,*,schema,profile,manifest_ref,ledger,sources,arrays,windows,admission,extra=None):
    output=Path(output).resolve();need(not output.exists(),'Fresh separate typed data bank output required');output.mkdir(parents=True)
    packed={k:np.concatenate([a[k] for a in arrays]) for k in ARRAY_KEYS} if arrays else {}
    array_ref=None
    if packed:
        with (output/'TRAIN.npz').open('xb') as stream:np.savez_compressed(stream,**packed)
        array_ref=rt.reference(output/'TRAIN.npz')
    index=rt.write_new(output/'INDEX.json',dict(samples=windows,order_sha256=rt.digest(windows),label_protocol=rt.shared.MASKED_LABEL_PROTOCOL))
    value=dict(schema=schema,profile=profile,source_manifest=manifest_ref,attempt_ledger=ledger,sources=sources,
        arrays=array_ref,array_proofs={k:dart.measured.array_proof(v) for k,v in packed.items()},index=index,windows=len(windows),
        normalization=deepcopy(NORMALIZATION),normalization_refitted=False,optimization_admission=admission,
        label_protocol=rt.shared.MASKED_LABEL_PROTOCOL,code_binding=code_binding(),
        model_qualified=False,benchmark_act_eligible=False,training_started=False,validation_payloads_opened=False,fresh_payloads_opened=False)
    if extra:value.update(extra)
    meta_ref=rt.write_new(output/'METADATA.json',value)
    return rt.write_new(output/'COMPLETE.json',dict(schema=schema,profile=profile,complete=True,metadata=meta_ref,
        arrays=array_ref,windows=len(windows),normalization=deepcopy(NORMALIZATION),optimization_admission=admission))


def export_enriched_dart(manifest_ref,parent_bank_ref,qualification_ref,output):
    qualification=_qualification(qualification_ref,parent_bank_ref)
    parent_arrays,parent_meta=_historical_parent(parent_bank_ref,qualification_ref)
    need(parent_meta['source_manifest']==manifest_ref and parent_meta['profile']==dart.PROFILE_SOURCE37 and
        parent_meta['windows']==274 and qualification['manifest']==manifest_ref and
        qualification['array_proofs']==parent_meta['array_proofs'],'Exact Source37 original274/manifest required')
    manifest=rt.read_bound(manifest_ref);base=Path(manifest_ref['path']).parent
    attempts,errors,precollection=dart._headers(manifest,base)
    need(sum(s['status']=='prepared' for s in precollection['slots'])==17 and
        sum(e['status']=='qualified' for e in parent_meta['attempt_ledger'])==17,'Original fixed20/prepared17/qualified17 required')
    output=Path(output).resolve();need(not output.exists(),'Fresh enriched DART output required')
    transport_ref,reader_exit=_original40_transport(manifest_ref,parent_bank_ref,qualification_ref,output/'ORIGINAL40_REAUDIT')
    slots=[e['request']['slot_id'] for e in parent_meta['attempt_ledger'] if e['status']=='qualified']
    transport=_read_original40_transport(transport_ref,manifest_ref,slots)
    by_slot={s['slot_id']:s['payload'] for s in transport['sources']}
    ledger=[];sources=[];arrays=[];windows=[];offset=0;mapping=[];parent_offset=0
    for attempt,old in zip(attempts,parent_meta['attempt_ledger']):
        entry=deepcopy(old)
        if old['status']=='qualified':
            need(attempt['request']['slot_id'] not in errors,'Original capture header no longer qualifies')
            parent={k:parent_arrays[k][parent_offset:parent_offset+old['windows']] for k in ARRAY_KEYS}
            parent['derived_index']=dict(windows=parent_arrays['derived_index']['windows'][parent_offset:parent_offset+old['windows']])
            sample,receipt,meta=_read_enriched_episode(attempt,by_slot[attempt['request']['slot_id']],parent);count=len(sample['states'])
            parent_offset+=old['windows']
            need(count>0,'Original qualified capture cannot be omitted')
            mapping.extend(offset+i for i in receipt['parent_sample_map'])
            sources.append(dict(slot_id=attempt['request']['slot_id'],declaration=attempt['episode'],metadata=meta,
                receipt=receipt,windows=count,split_sample_range=[offset,offset+count]))
            arrays.append(sample);windows.extend(dict(slot_id=attempt['request']['slot_id'],sample_index=offset+i,window=deepcopy(w))
                for i,w in enumerate(sample['derived_index']['windows']))
            offset+=count;entry.update(windows=count,receipt=receipt)
        elif attempt['outcome'] in ('physical_negative','preview_rejected'):dart._read_negative(attempt,base,manifest)
        ledger.append(entry)
    packed={k:np.concatenate([a[k] for a in arrays]) for k in ARRAY_KEYS}
    packed['derived_index']=dict(windows=[w['window'] for w in windows])
    need(parent_subsequence(parent_arrays,packed)==mapping and len(mapping)==274,'Global original274 ordered subsequence differs')
    gate=dart.admission(ledger,precollection=precollection)
    need(gate['pilot_optimization_eligible'] and gate['qualified_episodes']==17,'Original95percent/17 physical qualification must remain')
    gate=dict(gate,schema='m2_act_refinement_enriched_dart_admission_v1',data_optimization_eligible=bool(gate['pilot_optimization_eligible'] and gate['qualified_episodes']==17),benchmark_act_eligible=False)
    return _write_bank(output/'BANK',schema=ENRICHED_SCHEMA,profile=ENRICHED_PROFILE,manifest_ref=manifest_ref,ledger=ledger,
        sources=sources,arrays=arrays,windows=windows,admission=gate,extra=dict(parent_bank=parent_bank_ref,
        original_qualification=qualification_ref,parent_sample_map=mapping,precollection=precollection,new_native_captures=0,
        original40_transport=transport_ref,original40_reader_exit=reader_exit,physical_reader=deepcopy(SOURCE40),label_reader=rt.reference(__file__)))


def export_onpolicy(manifest_ref,output):
    from . import m2_act_onpolicy_collection as collection
    manifest=rt.read_bound(manifest_ref);attempts=collection.validate_manifest(manifest)
    base=Path(manifest_ref['path']).parent
    # Every declaration and configuration metadata is checked before payloads.
    collection.manifest_headers(manifest,base)
    ledger=[];sources=[];arrays=[];windows=[];offset=0;resets=set()
    for attempt in attempts:
        entry=dict(request=deepcopy(attempt['request']),episode=deepcopy(attempt['episode']),status=attempt['outcome'],
            reason=attempt['reason'],windows=0,infrastructure_contamination=False)
        try:
            if attempt['outcome'] in ('physical_negative','preview_rejected'):
                entry['negative_receipt']=collection.read_negative(attempt,base,manifest)
                ledger.append(entry);continue
            need(attempt['outcome']=='collected','Fixed collection attempt did not complete: '+attempt['reason'])
            sample,receipt,meta=collection.read_episode(attempt,base,manifest)
            reset=dart._reset(receipt);need(reset not in resets,'Duplicate native collection reset');resets.add(reset)
            count=len(sample['states']);need(count>0,'Completed collection has no actual teacher labels')
            sources.append(dict(slot_id=attempt['request']['slot_id'],declaration=attempt['episode'],metadata=meta,
                receipt=receipt,windows=count,split_sample_range=[offset,offset+count]))
            arrays.append(sample);windows.extend(dict(slot_id=attempt['request']['slot_id'],sample_index=offset+i,window=deepcopy(w))
                for i,w in enumerate(sample['derived_index']['windows']))
            offset+=count;entry.update(status='qualified',windows=count,receipt=receipt,native_reset_identity=receipt['native_reset_identity'])
        except (ValueError,KeyError,TypeError,OSError,RuntimeError) as error:
            entry.update(status='physical_negative' if attempt['outcome'] in ('physical_negative','preview_rejected') else 'infrastructure_error',
                infrastructure_contamination=attempt['outcome'] not in ('physical_negative','preview_rejected'),error=type(error).__name__+': '+str(error))
        ledger.append(entry)
    gate=collection.admission(ledger)
    return _write_bank(output,schema=ONPOLICY_SCHEMA,profile=ONPOLICY_PROFILE,manifest_ref=manifest_ref,ledger=ledger,
        sources=sources,arrays=arrays,windows=windows,admission=gate,extra=dict(prepared=manifest['prepared'],
        initialization=manifest['initialization'],fixed_denominator=8,producer_source=manifest['producer_source']))


def metadata(complete_ref):
    complete=rt.read_bound(complete_ref);schema=complete.get('schema')
    if schema==QUERY_SCHEMA:return _query_metadata(complete_ref)
    need(schema in (ENRICHED_SCHEMA,ONPOLICY_SCHEMA) and complete.get('complete') is True,'Distinct complete refinement bank required')
    meta=rt.read_bound(complete['metadata']);profile=ENRICHED_PROFILE if schema==ENRICHED_SCHEMA else ONPOLICY_PROFILE
    need(meta.get('schema')==schema and complete.get('profile')==meta.get('profile')==profile and
        all(complete.get(k)==meta.get(k) for k in ('arrays','windows','normalization','optimization_admission')) and
        meta.get('normalization')==NORMALIZATION and meta.get('normalization_refitted') is False and
        meta.get('code_binding')==code_binding() and
        meta.get('benchmark_act_eligible') is False,'Typed bank/source/normalization binding differs')
    if schema==ENRICHED_SCHEMA:
        _qualification(meta['original_qualification'],meta['parent_bank'])
        _,parent=_historical_parent(meta['parent_bank'],meta['original_qualification'])
        need(parent['source_manifest']==meta['source_manifest'] and len(meta.get('parent_sample_map',[]))==274 and
            meta.get('new_native_captures')==0,'Original Source37 parent/qualification differs')
        need(len(meta['attempt_ledger'])==20 and all(all(e.get(k)==old.get(k) for k in
            ('request','episode','nominal_source_config','status','reason','evidence','infrastructure_contamination'))
            for e,old in zip(meta['attempt_ledger'],parent['attempt_ledger'])),
            'Enrichment changed original fixed20 outcomes/source captures')
        exit=rt.read_bound(meta['original40_reader_exit'])
        need(exit.get('schema')=='m2_act_refinement_original40_reader_exit_v1' and exit.get('normal_exit') is True and
            exit.get('exit_code')==0 and exit.get('archive_authenticated_before') is True and exit.get('archive_authenticated_after') is True and
            exit.get('physical_reader')==SOURCE40 and meta.get('physical_reader')==SOURCE40 and meta.get('label_reader')==rt.reference(__file__),
            'Normally closed authenticated original40 physical reader required')
        slots=[e['request']['slot_id'] for e in parent['attempt_ledger'] if e['status']=='qualified']
        transport=_read_original40_transport(meta['original40_transport'],meta['source_manifest'],slots)
        transport_by_slot={s['slot_id']:s['payload'] for s in transport['sources']}
        original_gate=dart.admission(meta['attempt_ledger'],precollection=meta['precollection'])
        gate=dict(original_gate,schema='m2_act_refinement_enriched_dart_admission_v1',
            data_optimization_eligible=bool(original_gate['pilot_optimization_eligible'] and original_gate['qualified_episodes']==17),benchmark_act_eligible=False)
    else:
        from . import m2_act_onpolicy_collection as collection
        manifest=rt.read_bound(meta['source_manifest']);collection.validate_manifest(manifest)
        collection.manifest_headers(manifest,Path(meta['source_manifest']['path']).parent)
        need(len(meta['attempt_ledger'])==8 and all(e.get('request')==a['request'] and e.get('episode')==a['episode']
            for e,a in zip(meta['attempt_ledger'],manifest['attempts'])),'Original fixed8 outcome declarations differ')
        gate=collection.admission(meta['attempt_ledger'])
    need(gate==meta['optimization_admission'],'Typed fixed-denominator admission changed')
    index=rt.read_bound(meta['index']);items=index.get('samples');cursor=0;seen=set()
    need(isinstance(items,list) and index.get('order_sha256')==rt.digest(items) and
        [w.get('sample_index') for w in items]==list(range(meta['windows'])),'Ordered native windows changed')
    qualified=[e for e in meta['attempt_ledger'] if e['status']=='qualified']
    need([s['slot_id'] for s in meta['sources']]==[e['request']['slot_id'] for e in qualified],'Omitted/duplicate qualified captures')
    for source,entry in zip(meta['sources'],qualified):
        n=source['windows'];slot=source['slot_id'];m=source['metadata'];declaration=source['declaration']
        need(all(m.get(k)==entry['request'][k] for k in ('subject_id','split','target_id')) and
            m.get('episode_id')==declaration['episode_id'],'Actual TRAIN source metadata differs from fixed slot')
        if schema==ENRICHED_SCHEMA:
            need(source['receipt'].get('physical_reader')==SOURCE40 and source['receipt'].get('label_reader')==rt.reference(__file__) and
                source['receipt'].get('reaudited_transport')==transport_by_slot.get(slot),
                'Truthful original40 physical/current label reader origins required')
        else:
            need(source['receipt'].get('schema')=='m2_act_onpolicy_physical_data_receipt_v1' and
                source['receipt'].get('qualification_scope')=='actual_collection_teacher_data_only' and
                source['receipt'].get('benchmark_act_eligible') is False and source['receipt'].get('learner_prefix_executed') is True and
                source['receipt'].get('expert_intervention') is True,'Collection physics cannot be benchmark ACT evidence')
        need(source['declaration']==entry['episode'] and source['receipt']==entry['receipt'] and n==entry['windows'] and
            source['split_sample_range']==[cursor,cursor+n] and all(w['slot_id']==slot for w in items[cursor:cursor+n]),
            'Per-slot chronological contiguous source range differs')
        ticks=[w['window']['inclusive_native_ticks'][0] for w in items[cursor:cursor+n]]
        need(ticks==sorted(set(ticks)),'Duplicate/reordered native windows')
        reset=dart._reset(source['receipt']);need(reset not in seen,'Duplicate source reset');seen.add(reset);cursor+=n
    need(cursor==meta['windows']==len(items),'Typed bank source count differs')
    return complete,meta


def load_arrays(complete_ref):
    complete,meta=metadata(complete_ref)
    need(meta['optimization_admission'].get('data_optimization_eligible') is True,'Unqualified fixed-denominator data cannot load')
    with np.load(rt.bound_path(complete['arrays']),allow_pickle=False) as archive:
        need(set(archive.files)==set(ARRAY_KEYS),'Exact seven raw/mask arrays required');arrays={k:archive[k].copy() for k in ARRAY_KEYS}
    need({k:dart.measured.array_proof(v) for k,v in arrays.items()}==meta['array_proofs'],'Exact ordered typed array bytes differ')
    index=rt.read_bound(meta['index'])
    arrays.update(schema=given.MASKED_DATASET_SCHEMA,task_id=rt.pose.TASK_ID,observation_contract=rt.pose.observation_contract(),
        action_semantics_id=given.ACTION_SEMANTICS_ID,label_protocol=rt.shared.MASKED_LABEL_PROTOCOL,
        derived_index=dict(windows=[w['window'] for w in index['samples']]),normalization_reference=deepcopy(NORMALIZATION),bank_schema=meta['schema'])
    (query.validate_samples if meta['schema']==QUERY_SCHEMA else dart.validate_samples)(arrays)
    if meta['schema']==ENRICHED_SCHEMA:
        parent,_=_historical_parent(meta['parent_bank'],meta['original_qualification'])
        need(parent_subsequence(parent,arrays)==meta['parent_sample_map'],'Original274 ordered parent map differs')
    return arrays,meta


def _authenticate_original41_query():
    """Source-only authentication; never import a model or read tensor files."""
    warm.require_archive()
    return deepcopy(warm.SOURCE41)


def _query_parent(parent_bank_ref,*,authenticate=True):
    need(parent_bank_ref==SOURCE41_QUERY32,'Exact original41 unchanged Source32 query58 parent required')
    if authenticate:_authenticate_original41_query()
    _regular_unaliased(parent_bank_ref['path'])
    complete=rt.read_bound(parent_bank_ref);meta=rt.read_bound(complete['metadata'])
    query.teacher.validate_profile(meta.get('evidence_profile'),meta.get('recorded_source'))
    gate=query.admission(meta.get('attempt_ledger'),profile=query.PROFILE_V2)
    need(complete.get('schema')==meta.get('schema')==query.SCHEMA_V2 and complete.get('complete') is True and
        complete.get('profile')==meta.get('profile')==query.PROFILE_V2 and
        meta.get('base_dataset')==query.BASE29 and meta.get('base_counts')==query.BASE_COUNTS and
        complete.get('windows')==meta.get('windows')==58 and meta.get('normalization_refitted') is False and
        complete.get('normalization')==meta.get('normalization')==NORMALIZATION and
        gate['measured_native_optimization_eligible'] is True and
        complete.get('optimization_admission')==meta.get('optimization_admission')==gate and
        complete.get('arrays')==meta.get('arrays') and
        all(meta.get(k) is True for k in ('actions_executed','physics_qualified','native_eligible')),
        'Original41 query58 full four-episode source/admission/statistics required')
    names=dict(bank='m2_measured_recovery_data.py',capture_decoder='m2_given_contact_dataset.py',
        role_filter='m2_given_contact_recovery_dataset.py',native_physical='m2_stop_residual_trajectory.py',
        query_prefix='m2_query_recovery_prefix.py',teacher_reader='m2_teacher_stop_reader.py')
    root=Path(warm.SOURCE41['root']);listed={r['file']:r for r in rt.read_bound(warm.SOURCE41['source_manifest'])}
    need(set(meta.get('data_code',{}))==set(names),'All original41 pure module code references required')
    for key,name in names.items():
        relative='sim/ausc_policy/'+name;row=listed.get(relative,{})
        need(meta['data_code'][key]==dict(path=str(root/relative),bytes=row.get('bytes'),sha256=row.get('sha256')),
            'Exact original41 module origin required; no source aliases')
        _regular_unaliased(meta['data_code'][key]['path']);rt.bound_path(meta['data_code'][key])
    index=rt.read_bound(meta['index']);items=index.get('samples');cursor=0;episodes=set()
    need(isinstance(items,list) and len(items)==58 and index.get('order_sha256')==rt.digest(items) and
        [w.get('sample_index') for w in items]==list(range(58)),'Exact original query ordered window index required')
    need([s.get('slot_id') for s in meta.get('sources',[])]==list(query.query.SLOT_IDS),'All FOUR original query sources required')
    for i,(source,entry) in enumerate(zip(meta['sources'],meta['attempt_ledger'])):
        m=source['metadata'];declaration=source['declaration'];n=source['windows'];episode=m.get('episode_id')
        need(type(n) is int and n>0 and source.get('split_sample_range')==[cursor,cursor+n] and
            m.get('split')=='train' and m.get('subject_id')=='p001120' and
            declaration.get('pose_id')=='p000474_a001230_f0000' and m.get('pose_id',declaration['pose_id'])==declaration['pose_id'] and
            m.get('target_id')==('R3' if i<2 else 'L3') and isinstance(episode,str) and episode not in episodes and
            all(m.get(k)==declaration.get(k) for k in ('episode_id','split','subject_id','target_id')) and
            entry.get('episode')==declaration and entry.get('receipt')==source['receipt'] and entry.get('windows')==n and
            all(w.get('slot_id')==source['slot_id'] for w in items[cursor:cursor+n]),
            'Original four query source/receipt/range identities differ')
        episodes.add(episode);cursor+=n
    need(cursor==58 and set(meta.get('array_proofs',{}))==set(ARRAY_KEYS),'All58 seven-array query proofs required')
    return complete,meta,index


# Import ONLY the authenticated original41 package. Its metadata/load_arrays
# enforce Source32 headers, original full physical reader and native windows.
# No old policy, optimizer, Torch, World or GPU is involved.
_ORIGINAL41_QUERY_PROGRAM=r'''import json,os,sys
from pathlib import Path
payload=json.load(sys.stdin);root=Path(payload['reader_source']['root']);sys.path.insert(0,str(root))
from sim.ausc_policy import m2_given_contact_runtime as rt,m2_measured_recovery_data as query
import numpy as np
assert sys.platform=='linux' and sys.executable==payload['python'] and sys.prefix=='/opt/wam-runtime/envs/isaacsim60'
assert sys.flags.isolated and sys.dont_write_bytecode and os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert 'AUSC_OWNED_RUN' not in os.environ
assert all(Path(m.__file__).resolve().is_relative_to(root) for m in (rt,query))
complete,meta=query.metadata(payload['parent_bank']);samples,again=query.load_arrays(payload['parent_bank'])
assert again==meta and len(samples['states'])==58 and len(meta['sources'])==4
assert meta['data_code']==query.code_binding() and meta['profile']==query.PROFILE_V2
assert meta['optimization_admission']['measured_native_optimization_eligible']
index=rt.read_bound(meta['index']);assert len(index['samples'])==58
assert {k:query.array_proof(samples[k]) for k in query.ARRAY_KEYS}==meta['array_proofs']
assert samples['derived_index']['windows']==[w['window'] for w in index['samples']]
assert not any(n=='torch' or n.startswith(('torch.','isaacsim','omni.')) for n in sys.modules)
out=Path(payload['output']);out.mkdir(parents=True,exist_ok=False)
with (out/'TRAIN.npz').open('xb') as stream:np.savez_compressed(stream,**{k:samples[k] for k in query.ARRAY_KEYS})
fingerprint=dict(executable=sys.executable,prefix=sys.prefix,isolated=True,bytecode_disabled=True,cuda_visible_devices='',
    no_owner=True,no_torch_world=True,linux=True)
rt.write_new(out/'COMPLETE.json',dict(schema=payload['schema'],complete=True,parent_bank=payload['parent_bank'],reader_source=payload['reader_source'],
    original_metadata=complete['metadata'],original_index=meta['index'],original_code=meta['data_code'],arrays=rt.reference(out/'TRAIN.npz'),
    array_proofs=meta['array_proofs'],windows=58,processing_environment=fingerprint))
'''


def _original41_query_transport(parent_bank_ref,output):
    need(parent_bank_ref==SOURCE41_QUERY32,'Exact unchanged query parent required before archive/payload reads')
    reader=_authenticate_original41_query();_query_parent(parent_bank_ref,authenticate=False)
    need(sys.platform=='linux','Original41 query transport is Linux Native CPU only')
    output=Path(output).resolve();need(not output.exists(),'Fresh original41 query transport output required')
    ld=rt.read_bound(SDK_CONFIGURATION).get('ld_library_path')
    need(isinstance(ld,str) and ld,'Authenticated original Native SDK LD_LIBRARY_PATH required verbatim')
    output.mkdir(parents=True);transport=output/'payloads'
    payload=dict(schema=QUERY_TRANSPORT_SCHEMA,parent_bank=parent_bank_ref,reader_source=reader,python=recovery_python(),output=str(transport))
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMNI_KIT_ACCEPT_EULA='YES',LD_LIBRARY_PATH=ld,
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    for key in ('AUSC_OWNED_RUN','PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','HF_TOKEN'):environment.pop(key,None)
    code=None;error=None;after=False
    try:
        with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
            result=subprocess.run([payload['python'],'-I','-B','-c',_ORIGINAL41_QUERY_PROGRAM],input=json.dumps(payload).encode(),
                stdout=stdout,stderr=stderr,env=environment,cwd=reader['root'],check=False)
        code=result.returncode;need(code==0,'Original41 query full data reader failed; no consumer admitted')
    except BaseException as caught:
        error=type(caught).__name__+': '+str(caught);raise
    finally:
        try:after=_authenticate_original41_query()==reader
        finally:
            receipt=dict(schema=QUERY_EXIT_SCHEMA,reader_source=reader,parent_bank=parent_bank_ref,
                archive_authenticated_before=True,archive_authenticated_after=after,exit_code=code,
                normal_exit=code==0 and error is None and after,error=error,python=payload['python'],command_flags=['-I','-B'],
                sdk_configuration=SDK_CONFIGURATION,
                stdout=rt.reference(output/'stdout.log') if (output/'stdout.log').exists() else None,
                stderr=rt.reference(output/'stderr.log') if (output/'stderr.log').exists() else None)
            rt.write_new(output/'reader-exit.json',receipt)
            if not receipt['normal_exit'] and transport.exists():shutil.rmtree(transport)
    ref=rt.reference(transport/'COMPLETE.json');_read_original41_query_transport(ref,parent_bank_ref)
    return ref,rt.reference(output/'reader-exit.json')


def _read_original41_query_transport(ref,parent_bank_ref):
    complete,meta,index=_query_parent(parent_bank_ref)
    _regular_unaliased(ref['path']);value=rt.read_bound(ref)
    need(value.get('schema')==QUERY_TRANSPORT_SCHEMA and value.get('complete') is True and
        value.get('parent_bank')==parent_bank_ref and value.get('reader_source')==warm.SOURCE41 and value.get('windows')==58 and
        value.get('original_metadata')==complete['metadata'] and value.get('original_index')==meta['index'] and
        value.get('original_code')==meta['data_code'] and value.get('array_proofs')==meta['array_proofs'],
        'Exact original41 query source/module/metadata/index/array transport required')
    need(value.get('processing_environment')==dict(executable=recovery_python(),prefix='/opt/wam-runtime/envs/isaacsim60',
        isolated=True,bytecode_disabled=True,cuda_visible_devices='',no_owner=True,no_torch_world=True,linux=True),
        'Exact isolated CUDA-hidden Native CPU query reader required')
    root=Path(ref['path']).parent;array_path=Path(value['arrays']['path'])
    need(array_path==root/'TRAIN.npz' and set(p.name for p in root.iterdir())=={'COMPLETE.json','TRAIN.npz'},
        'Unknown/omitted or aliased query transport files')
    _regular_unaliased(array_path);rt.bound_path(value['arrays'])
    # Original41's child admitted all seven numerical arrays. Current metadata
    # binds that sealed transport without loading its NPZ; load_arrays performs
    # the independent numerical proof and native-label validation at consumption.
    return value


def _query_consumer_metadata(parent,transport,reader_exit):
    complete,original,_=_query_parent(parent)
    packet=_read_original41_query_transport(transport,parent);exit=rt.read_bound(reader_exit)
    need(exit.get('schema')==QUERY_EXIT_SCHEMA and exit.get('normal_exit') is True and exit.get('exit_code')==0 and
        exit.get('archive_authenticated_before') is True and exit.get('archive_authenticated_after') is True and
        exit.get('reader_source')==warm.SOURCE41 and exit.get('parent_bank')==parent and exit.get('command_flags')==['-I','-B'] and
        exit.get('python')==recovery_python() and exit.get('sdk_configuration')==SDK_CONFIGURATION,
        'Normally closed authenticated original41 query reader required')
    # Keep original source paths, receipts, data_code and every semantic field
    # verbatim. The new typed schema identifies this consumer, not a new capture.
    value=deepcopy(original)
    value.update(schema=QUERY_SCHEMA,profile=QUERY_PROFILE,parent_bank=deepcopy(parent),reader_source=deepcopy(warm.SOURCE41),
        producer_source=deepcopy(original['recorded_source']),original_metadata=deepcopy(complete['metadata']),
        original41_transport=deepcopy(transport),original41_reader_exit=deepcopy(reader_exit),arrays=packet['arrays'],
        code_binding=code_binding(),normalization_refitted=False,new_native_captures=0,benchmark_act_eligible=False,
        optimization_admission=dict(original['optimization_admission'],schema='m2_act_refinement_query32_admission_v1',
            data_optimization_eligible=True,benchmark_act_eligible=False))
    return value


def export_query32(parent_bank_ref,output):
    """New typed consumer; unchanged query58 decoded ONLY by original41."""
    need(parent_bank_ref==SOURCE41_QUERY32,'Exact unchanged query parent required before any payload')
    output=Path(output).resolve();need(not output.exists(),'Fresh typed query consumer output required')
    transport,exit=_original41_query_transport(parent_bank_ref,output/'ORIGINAL41_QUERY')
    meta=_query_consumer_metadata(parent_bank_ref,transport,exit)
    meta_ref=rt.write_new(output/'METADATA.json',meta)
    return rt.write_new(output/'COMPLETE.json',dict(schema=QUERY_SCHEMA,profile=QUERY_PROFILE,complete=True,metadata=meta_ref,
        **{k:deepcopy(meta[k]) for k in ('arrays','windows','normalization','optimization_admission')}))


def _query_metadata(ref):
    complete=rt.read_bound(ref);meta=rt.read_bound(complete['metadata'])
    expected=_query_consumer_metadata(meta.get('parent_bank'),meta.get('original41_transport'),meta.get('original41_reader_exit'))
    need(meta==expected and complete==dict(schema=QUERY_SCHEMA,profile=QUERY_PROFILE,complete=True,metadata=complete['metadata'],
        **{k:deepcopy(meta[k]) for k in ('arrays','windows','normalization','optimization_admission')}),
        'Exact current consumer/original41 reader/Source32 producer provenance required')
    return complete,meta
