"""Explicit Source30 query-cut replay inputs, sealed once by an isolated CPU child.

The loader reads only this bounded sealed proof and prefix cells. It never
loads historical factories/models or source native/CPU traces into a World.
"""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from . import m2_given_contact_runtime as rt
from . import m2_query_recovery_child as child

need=rt.need
SCHEMA='m2_actual_student_query_recovery_prefix_v2'
REQUEST_SCHEMA='m2_source30_query_recovery_request_v1'
BRIDGE_SCHEMA='m2_source30_query_recovery_cpu_bridge_v1'
ARCHIVE=child.ARCHIVE
SLOT_IDS=child.SLOT_IDS
NATIVE_PYTHON=child.NATIVE_PYTHON
CHILD_ENTRY=Path(child.__file__).resolve()
CHILD_TIMEOUT_S=1800
THREAD_VARIABLES=child.THREAD_VARIABLES
select_rows=child.select_rows
slot_ledger=child.slot_ledger


def source_metadata(source):
    return child.source_metadata(rt,source)


def require_archive():
    child.verify_archive()


def child_command(request_path,output):
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    for key in THREAD_VARIABLES:environment[key]='1'
    for key in ('AUSC_OWNED_RUN','PYTHONPATH'):environment.pop(key,None)
    return [NATIVE_PYTHON,'-I','-B',str(CHILD_ENTRY),str(request_path),str(output)],environment


def validate_producer(value):
    """Authenticate the recorded new producer independently of this decoder."""
    need(isinstance(value,dict) and set(value)=={'root','code_commit','code_commit_file','source_manifest','file_count','files'},
        'Recorded query-prefix producer source closure required')
    root=Path(value['root']);commit=value['code_commit']
    need(root.is_absolute() and root.resolve()==root and type(commit) is str and len(commit)==40 and
        all(c in '0123456789abcdef' for c in commit) and
        Path(value['source_manifest']['path'])==root/'SOURCE_MANIFEST.json' and
        Path(value['code_commit_file']['path'])==root/'CODE_COMMIT' and
        rt.bound_path(value['code_commit_file']).read_text().strip()==commit,'Recorded producer root/commit/manifest paths differ')
    rows=rt.read_bound(value['source_manifest'])
    need(isinstance(rows,list) and type(value['file_count']) is int and len(rows)==value['file_count']>=2 and
        len({r['file'] for r in rows})==len(rows) and
        all(isinstance(r.get('file'),str) and not Path(r['file']).is_absolute() and '..' not in Path(r['file']).parts for r in rows),
        'Complete unique safe recorded producer manifest required')
    inventory={r['file']:r for r in rows};files=value['files']
    need(set(files)=={'m2_query_recovery_prefix.py','m2_query_recovery_child.py'},'Exact producer parent/entry source binding required')
    # Approval follows content; the original recorded paths remain authoritative
    # when an identical decoder is relocated to a later immutable source root.
    approved={'m2_query_recovery_prefix.py':rt.reference(__file__),
        'm2_query_recovery_child.py':rt.reference(CHILD_ENTRY)}
    for name,ref in files.items():
        relative='sim/ausc_policy/'+name
        need(Path(ref['path'])==root/relative and relative in inventory and
            all(ref.get(k)==inventory[relative].get(k) for k in ('bytes','sha256')),
            'Recorded producer module path/manifest bytes differ')
        rt.bound_path(ref)
        need(all(ref.get(k)==approved[name][k] for k in ('bytes','sha256')),
            'Recorded module differs from approved producer content')
    return value


def producer_source():
    root=Path(__file__).resolve().parents[2];manifest=rt.reference(root/'SOURCE_MANIFEST.json')
    return validate_producer(dict(root=str(root),code_commit=(root/'CODE_COMMIT').read_text().strip(),
        code_commit_file=rt.reference(root/'CODE_COMMIT'),source_manifest=manifest,file_count=len(rt.read_bound(manifest)),
        files={Path(__file__).name:rt.reference(__file__),CHILD_ENTRY.name:rt.reference(CHILD_ENTRY)}))


def _namespace(origins):
    root=Path(ARCHIVE['root'])
    need(isinstance(origins,dict) and {'sim.ausc_policy.m2_student_recovery_prefix',
        'sim.ausc_policy.m2_student_oracle_proposals'}<=set(origins),'Original authentication namespace proof required')
    for name,origin in origins.items():
        need(name.split('.')[0] in ('sim','scripts'),'Unknown project namespace')
        need(origin.get('file') is None or origin.get('file')==origin.get('spec_origin'),
            'Recorded project file/spec origins differ')
        for path in [origin.get('file'),origin.get('spec_origin'),*origin.get('package_paths',[])]:
            if path not in (None,'namespace'):need(Path(path).is_absolute() and Path(path).resolve()==Path(path) and Path(path).is_relative_to(root),
                'Historical source namespace escaped original30')


def validate_bridge(ref):
    """Bound JSON/hash proof only; never reauthenticate multi-GB historical traces."""
    proof=rt.read_bound(ref);need(proof.get('schema')==BRIDGE_SCHEMA and proof.get('complete') is True and
        proof.get('profile')=='production_source30' and proof.get('source30')==ARCHIVE and
        proof.get('archive_verified_before') is True and proof.get('archive_verified_after') is True,
        'Explicit complete original30 production CPU proof required')
    execution=proof.get('execution',{});entry=proof.get('child_entry')
    producer=validate_producer(proof.get('producer_source'))
    rt.bound_path(entry)
    need(rt.bound_path(proof['child_log']).stat().st_size<=65536,'Bounded scalar/reference child log required')
    need(entry==producer['files']['m2_query_recovery_child.py'] and
        execution.get('exit_code')==0 and type(execution.get('exit_code')) is int and
        execution.get('normal_exit') is True and execution.get('timeout_s')==CHILD_TIMEOUT_S and
        execution.get('cwd')==ARCHIVE['root'] and execution.get('command')==[NATIVE_PYTHON,'-I','-B',entry['path'],
            proof['request']['path'],str(Path(proof['result']['path']).parent)] and
        execution.get('environment')==dict(cuda_visible_devices='',owned_run_present=False,pythonpath_present=False,
            thread_variables={k:'1' for k in THREAD_VARIABLES}), 'Exact isolated child entry/exit/environment required')
    request=rt.read_bound(proof['request'])
    need(request.get('schema')==REQUEST_SCHEMA and request.get('profile')=='production_source30' and
        set(request.get('sources',{}))=={'R3','L3'},'Exact sealed TRAIN source request required')
    for target,source in request['sources'].items():
        _,meta=source_metadata(source);need(meta['target_id']==target,'Sealed source target differs')
        for item in source.values():rt.bound_path(item)  # bounded metadata; never source trace payloads
    result=rt.read_bound(proof['result'])
    need(result.get('schema')==child.RESULT_SCHEMA and result.get('complete') is True and result.get('source30')==ARCHIVE and
        result.get('request')==proof['request'] and result.get('child_entry')==entry and result.get('profile')=='production_source30' and
        result.get('cwd')==ARCHIVE['root'] and result.get('executable')==NATIVE_PYTHON and
        result.get('completion_gate_verified_before') is True and result.get('completion_gate_verified_after') is True and
        result.get('source_episode_verdicts_unchanged') is True,'Sealed original CPU output manifest differs')
    _namespace(result.get('namespace_origins'))
    slots=result.get('slots');need(isinstance(slots,list) and [s.get('slot_id') for s in slots]==list(SLOT_IDS),
        'Complete FOUR fixed slot ledger required')
    expected=slot_ledger({target:dict(source=request['sources'][target],rows=[s['physics_row'] for s in slots if s['target_id']==target])
        for target in ('R3','L3')})
    need(slots==expected and set(result.get('prefixes',{}))=={s['slot_id'] for s in slots if s['status']=='selected'},
        'Missing/duplicate/query identity ledger differs')
    return proof,result


def _payload(value):
    need(value.get('schema')==child.PAYLOAD_SCHEMA and value.get('profile')=='production_source30' and value.get('split')=='train' and
        value.get('subject_id')=='p001120' and value.get('target_id') in ('R3','L3') and value.get('slot_id') in SLOT_IDS and
        value.get('positive_teacher_labels') is False and value.get('learned_policy_executed') is False and
        value.get('actual_student_command_prefix_replayed') is True and value.get('source_episode_verdict_unchanged') is True and
        value.get('initial_context_phase') in ('FREE','APPROACH') and value.get('source_kind')=='retained_actual_student_raw_targets',
        'Explicit authenticated TRAIN replay perturbation payload required')
    targets=np.asarray(value.get('raw_targets'),float);roles=value.get('source_roles');start=value.get('start_row');cut=value.get('end_row_exclusive')
    need(targets.ndim==2 and targets.shape[1:]==(7,) and 1<=len(targets)<=2400 and np.isfinite(targets).all() and
        type(start) is int and start>=12 and start%12==0 and type(cut) is int and cut==start+12*len(targets) and
        isinstance(roles,list) and len(roles)==len(targets) and set(roles)<={'given_l2_motion','given_l2_history_wait'} and
        roles[0]=='given_l2_motion' and value.get('source_policy_motion_cells')==roles.count('given_l2_motion') and
        value.get('source_history_wait_cells')==roles.count('given_l2_history_wait'),'Exact bounded completed source command cells/roles required')
    rt.shared.contract._array(value.get('initial_measured_q'),(7,),'initial actual measured q')
    query=value.get('source_cut_query',{});select_rows([query],gap_m=value.get('actual_gap_m'),target=value['target_id'])
    need(query['physics_row']==cut and query.get('metadata',{}).get('split')=='train' and
        query.get('reference_sha256')==value.get('reference_sha256') and
        query.get('metadata',{}).get('subject_id')==value['subject_id'] and query.get('metadata',{}).get('target_id')==value['target_id'],
        'Actual authenticated cut state/reference differs')
    rt.shared.contract._array(query.get('measured_q'),(7,),'source cut actual q')
    rt.shared.contract._array(query.get('measured_dq'),(7,),'source cut actual dq')
    rt.shared.contract._pose(query.get('measured_pose_base'),'source cut actual native TCP full pose')
    seam=rt.shared.contract._array(value.get('source_cut_command_history'),(3,7),'source retained command seam')
    snapshot=value.get('source_cut_command_input',{})
    need(np.array_equal(seam[-1],snapshot.get('fields',{}).get('current_position')),'Source cut retained command input differs')
    for key in ('current_velocity','current_acceleration'):rt.shared.contract._array(snapshot.get('fields',{}).get(key),(7,),key)
    started=value.get('l2_started_at_s');deadline=value.get('l2_deadline_s')
    need(type(started) in (int,float) and np.isfinite(started) and 0<=started<=start/240. and deadline==started+120. and
        query['time_s']<deadline and value.get('source_cut_remaining_budget_s')==deadline-query['time_s'],
        'Original cumulative120s L2 clock/deadline required')
    if value['slot_id']=='l3_last_45s':need(query['time_s']<=45.,'Late cut exceeds original45s physics clock')
    for key in ('source_cpu_trace','source_native_trace'):rt._reference_shape(value.get(key))
    return value


def load(ref,source_config):
    saved=rt.read_bound(ref)
    need(set(saved)=={'schema','profile','payload','bridge','slot_id'} and saved.get('schema')==SCHEMA and
        saved.get('profile')=='production_source30','Explicit query-prefix v2 wrapper required')
    proof,result=validate_bridge(saved['bridge']);slot=saved['slot_id']
    need(slot in SLOT_IDS and result['prefixes'].get(slot)==saved['payload'],'Prefix is outside sealed output manifest')
    value=_payload(rt.read_bound(saved['payload']));declared=result['slots'][SLOT_IDS.index(slot)]
    config,meta=source_metadata(value['source']);dep=config['m2_given_contact_expanded_pilot_deployment']
    need(value['slot_id']==slot and value['source_config']==source_config and value['source']==declared['source'] and
        value['end_row_exclusive']==declared['physics_row'] and declared['status']=='selected' and
        value['source_model']==dep.get('model') and value['source_config']==dep.get('source_config') and
        value['reference_sha256']==meta.get('reference_sha256'),'Same-scene original nominal/query/model source required')
    return value


def export(request_ref,output):
    """Linux CPU producer only; never launches Native, a GPU owner or inference."""
    request=rt.read_bound(request_ref)
    # TRAIN/config and whole owner gates precede archive and trace work.
    child.require_gate(rt,request)
    need(sys.platform=='linux','Production historical bridge requires Linux CPU environment')
    require_archive();producer=producer_source();entry=rt.reference(CHILD_ENTRY)
    output=Path(output).resolve();need(not output.exists() and not output.is_relative_to(ARCHIVE['root']),
        'New producer directory outside immutable Source30 required');output.mkdir(parents=True)
    child_output=output/'original30';child_output.mkdir()
    command,environment=child_command(request_ref['path'],child_output)
    # Files bound memory use even if an imported original accidentally logs data.
    log=output/'child.log'
    with log.open('xb') as stream:
        try:result=subprocess.run(command,cwd=ARCHIVE['root'],env=environment,stdout=stream,stderr=subprocess.STDOUT,
            check=False,timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            require_archive();raise ValueError('Original30 CPU bridge exceeded finite deadline') from error
    require_archive();validate_producer(producer);rt.bound_path(entry)
    need(result.returncode==0,'Original30 CPU bridge failed (exit '+str(result.returncode)+'); see bound child log')
    need(log.stat().st_size<=65536,'Child logs exceeded bounded scalar/reference envelope')
    result_ref=rt.reference(child_output/'RESULT.json')
    proof=rt.write_new(output/'BRIDGE.json',dict(schema=BRIDGE_SCHEMA,complete=True,profile='production_source30',source30=ARCHIVE,
        producer_source=producer,request=request_ref,result=result_ref,child_entry=entry,child_log=rt.reference(log),
        archive_verified_before=True,archive_verified_after=True,
        execution=dict(command=command,cwd=ARCHIVE['root'],exit_code=result.returncode,normal_exit=True,timeout_s=CHILD_TIMEOUT_S,
            environment=dict(cuda_visible_devices='',owned_run_present=False,pythonpath_present=False,thread_variables={k:'1' for k in THREAD_VARIABLES}))))
    _,manifest=validate_bridge(proof);prefixes={}
    for slot,payload in manifest['prefixes'].items():
        prefixes[slot]=rt.write_new(output/(slot+'.json'),dict(schema=SCHEMA,profile='production_source30',slot_id=slot,payload=payload,bridge=proof))
        load(prefixes[slot],rt.read_bound(payload)['source_config'])
    return rt.write_new(output/'COMPLETE.json',dict(schema=BRIDGE_SCHEMA,complete=True,bridge=proof,slots=manifest['slots'],prefixes=prefixes,
        fixed_denominator=4,collection_eligible=all(s['status']=='selected' for s in manifest['slots']),
        physics_executed=False,training_eligible=False))
