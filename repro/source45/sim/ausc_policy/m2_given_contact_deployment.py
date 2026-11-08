"""Finite, simulation-only given ACT permit and same-session CPU pipe bridge.

The native parent alone advances physics and owns persistent interpolation,
E10 execution, parking and analytic contact. The child receives observations
only; protocol failure ends the learned attempt without expert substitution.

CPU metadata freeze: --freeze-smoke --admission/--admission-sha256,
--allocation-mapping/--mapping-sha256, --attempts/--attempts-sha256,
--output NEW_PERMIT.json. Attempts contain exactly four frozen native job
identities, full given metadata, seed, runtime_config_sha256 (excluding the
formal route), template and original patient/preparation refs. Native config
uses m2_given_contact_deployment with schema, production_ready=true,
simulation_only=true, permit ref, episode_id and metadata. TEST has no permit
or scene-store capability in this version. Native execution stays under the
original isaac_batch owner; --serve is its same-session CPU child only.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

import numpy as np

from . import m2_given_contact_runtime as runtime

need=runtime.need
MODULE='sim.ausc_policy.m2_given_contact_deployment'
PERMIT_SCHEMA='m2_given_contact_formal_simulation_permit_v1'
SMOKE_PLAN_SCHEMA='m2_given_contact_trainval_bridge_smoke_plan_v1'
STARTUP_S=120.
REQUEST_S=10.
REQUEST_BYTES=32768
RESPONSE_BYTES=131072


def protocol():
    return dict(schema=runtime.given.PROTOCOL_SCHEMA,task_contract=runtime.task_contract(),
        startup_timeout_s=STARTUP_S,shared_write_read_timeout_s=REQUEST_S,request_bytes=REQUEST_BYTES,
        response_bytes=RESPONSE_BYTES,outstanding_requests=1,history_spacing_s=.05,history_steps=2,
        horizon=20,execution_prefix=10,transport='newline JSON over owned same-session stdin/stdout pipes',
        observation_only=True,expert_l2_fallback=False,cpu_threads=2,interop_threads=1,
        physics_during_inference='unchanged authoritative native timestamp/row/q',real_hardware_validated=False)


def wire_decode(raw):
    def pairs(items):
        value={}
        for key,item in items:
            need(key not in value, 'Duplicate wire JSON key');value[key]=item
        return value
    def constant(value):raise ValueError('Nonfinite wire JSON value: '+value)
    try:value=json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
    except (UnicodeDecodeError,json.JSONDecodeError,RecursionError) as error:raise ValueError('Malformed bounded wire JSON') from error
    need(isinstance(value,dict), 'Wire JSON object required');return value


def _attempt(dep,permit):
    rows=[r for r in permit['attempts'] if r['episode_id']==dep['episode_id']]
    need(len(rows)==1, 'Episode outside exact frozen simulation permit')
    return rows[0]


def make_request(dep,permit,sequence,measured_q,target,confidence,*,metadata,history,history_times,context):
    row=_attempt(dep,permit);ref=runtime.pose.require_route(metadata)
    binding=dict(permit_sha256=dep['permit']['sha256'],admission_sha256=permit['admission']['sha256'],
        checkpoint_sha256=permit['model']['checkpoint']['sha256'],episode_id=row['episode_id'],attempt_id=row['attempt_id'],
        subject_id=row['subject_id'],split=row['split'],target_id=row['target_id'],
        reference_sha256=ref.sha256,reference_revision=0,reference_pose_base=ref.pose_base.tolist())
    return dict(schema=runtime.given.PROTOCOL_SCHEMA,protocol_sha256=runtime.digest(protocol()),request_id=sequence,
        binding=binding,metadata=deepcopy(metadata),measured_q=np.asarray(measured_q).tolist(),
        estimated_target_base=np.asarray(target).tolist(),confidence=float(confidence),
        history=np.asarray(history).tolist(),history_times=np.asarray(history_times).tolist(),context=deepcopy(context))


def validate_request(value,dep,permit,*,previous):
    need(set(value)=={'schema','protocol_sha256','request_id','binding','metadata','measured_q','estimated_target_base',
        'confidence','history','history_times','context'} and value.get('schema')==runtime.given.PROTOCOL_SCHEMA and
        value.get('protocol_sha256')==runtime.digest(protocol()) and permit.get('protocol')==protocol() and
        permit.get('task_contract')==runtime.task_contract(), 'Exact formal observation-only protocol required')
    row=_attempt(dep,permit);meta=value['metadata']
    need(meta==dep['metadata']==row['metadata'], 'Request changed frozen full-reference/episode metadata')
    if permit.get('schema')=='m2_given_contact_sealed_test_capability_v1':
        from .m2_given_contact_evaluation import require_permitted_request
        require_permitted_request(meta,dep,permit,row)
    else:runtime.shared.require_membership(meta.get('split'),meta.get('subject_id'))
    ref=runtime.pose.require_route(meta)
    need(meta.get('handoff_policy_id')==runtime.handoff.POLICY_ID and all(meta.get(key)==row[key] for key in
         ('episode_id','subject_id','split','target_id')), 'Request identity/common handoff contract differs')
    q=runtime.shared._vector(value['measured_q'],(7,),'same-time measured q')
    history=runtime.shared._vector(value['history'],(2,17),'two real causal history frames')
    times=runtime.shared._vector(value['history_times'],(2,),'two real history times')
    target=runtime.shared._vector(value['estimated_target_base'],(4,4),'full prescribed target pose')
    need(value['confidence']==1 and np.array_equal(target,ref.pose_base), 'Given full R/revision/availability changed')
    state=runtime.shared.contract.encode_state(q,target,1,metadata=meta)
    need(np.array_equal(history[1],state) and times[0]>=0 and abs(times[1]-times[0]-.05)<=1e-8,
         'Inference origin/current/history/20 Hz spacing differs')
    context=value['context'];phase_context=context.get('context') if isinstance(context,dict) else None
    stamp=context.get('observed_at_s') if isinstance(context,dict) else None
    need(set(context)=={'observed_at_s','physics_row','source_id','context'} and
         isinstance(phase_context,list) and phase_context[:3]==[row['episode_id'],row['target_id'],0] and
         len(phase_context)==4 and phase_context[3] in ('FREE','APPROACH') and
         type(stamp) in (int,float) and np.isfinite(stamp) and stamp>=0 and
         type(context['physics_row']) is int and context['physics_row']>=0 and
         context['source_id']=='given-contact-'+ref.sha256 and abs(times[-1]-stamp)<=1e-8 and
         ref.snapshot()['available_at_s']<=times[0], 'Invalid/stale same-time native provider context')
    expected=make_request(dep,permit,value['request_id'],q,target,1,metadata=meta,history=history,history_times=times,context=context)
    need(value['binding']==expected['binding'], 'Frozen permit/model/full reference binding differs')
    need(type(value['request_id']) is int and value['request_id']==(1 if previous is None else previous['request_id']+1),
         'Nonconsecutive or duplicate inference request')
    if previous is not None:
        need(stamp>previous['stamp'] and context['physics_row']>previous['physics_row'] and
             not(previous['phase']=='APPROACH' and phase_context[3]=='FREE'), 'Reordered/stale clock or phase regression')
    return dict(request_id=value['request_id'],stamp=stamp,physics_row=context['physics_row'],phase=phase_context[3])


def response_for(request,relative):
    relative=runtime.shared._vector(relative,(20,7),'complete finite relative H20')
    return dict(schema=runtime.given.PROTOCOL_SCHEMA,protocol_sha256=runtime.digest(protocol()),request_id=request['request_id'],
        request_sha256=runtime.digest(request),binding=deepcopy(request['binding']),context=deepcopy(request['context']),
        origin=deepcopy(request['measured_q']),relative=relative.tolist(),
        metadata=dict(deepcopy(request['metadata']),checkpoint_policy=runtime.given.CHECKPOINT_SCHEMA))


def decode_response(value,request):
    need(set(value)=={'schema','protocol_sha256','request_id','request_sha256','binding','context','origin','relative','metadata'} and
         value.get('schema')==runtime.given.PROTOCOL_SCHEMA and value.get('protocol_sha256')==runtime.digest(protocol()) and
         value.get('request_id')==request['request_id'] and value.get('request_sha256')==runtime.digest(request) and
         value.get('binding')==request['binding'] and value.get('context')==request['context'] and
         value.get('metadata')==dict(request['metadata'],checkpoint_policy=runtime.given.CHECKPOINT_SCHEMA),
         'Inference response protocol/sequence/model/reference/same-time echo differs')
    origin=runtime.shared._vector(value['origin'],(7,),'response measured origin')
    need(np.array_equal(origin,np.asarray(request['measured_q'])), 'Inference changed fixed measured block origin')
    return runtime.shared.contract.JointActionBlock('block_relative_joint',value['relative'],origin,value['metadata'])


class FramedPipe:
    def __init__(self,reader,writer):self.reader=reader;self.writer=writer;self.buffer=b''

    def _read(self,deadline):
        with selectors.DefaultSelector() as select:
            select.register(self.reader,selectors.EVENT_READ)
            while b'\n' not in self.buffer:
                remaining=deadline-time.monotonic()
                need(remaining>0 and bool(select.select(max(0.,remaining))), 'Inference response deadline reached')
                chunk=os.read(self.reader.fileno(),65536)
                need(bool(chunk), 'Inference child EOF/truncated response')
                self.buffer+=chunk;need(len(self.buffer)<=RESPONSE_BYTES, 'Oversized inference response')
        line,self.buffer=self.buffer.split(b'\n',1)
        need(not self.buffer, 'Unsolicited duplicate inference response')
        return wire_decode(line)

    def read(self,*,timeout):return self._read(time.monotonic()+timeout)

    def exchange(self,value,*,timeout):
        payload=json.dumps(value,allow_nan=False,separators=(',',':')).encode()+b'\n'
        need(len(payload)<=REQUEST_BYTES, 'Oversized inference request')
        deadline=time.monotonic()+timeout;offset=0
        with selectors.DefaultSelector() as select:
            select.register(self.reader,selectors.EVENT_READ)
            need(not select.select(0), 'Unsolicited response before request')
        descriptor=self.writer.fileno();os.set_blocking(descriptor,False)
        with selectors.DefaultSelector() as select:
            select.register(self.writer,selectors.EVENT_WRITE)
            while offset<len(payload):
                remaining=deadline-time.monotonic()
                need(remaining>0 and bool(select.select(max(0.,remaining))), 'Inference shared write/read deadline reached')
                try:written=os.write(descriptor,payload[offset:])
                except (BlockingIOError,InterruptedError):continue
                need(written>0, 'Inference child stopped reading');offset+=written
        return self._read(deadline)


POSE_TEMPLATES=frozenset(('p000474_a001230_f0000','p000476_a001231_f0023'))


def _validate_permit(value):
    need(set(value)=={'schema','plan_schema','protocol','task_contract','production_ready','simulation_only','scope',
        'test_payloads_opened','physical_acceptance','real_hardware_validated','closed_loop_evaluated',
        'admission','model','processing','source_inventory','allocation_mapping','attempts'} and
        value.get('schema')==PERMIT_SCHEMA and value.get('plan_schema')==SMOKE_PLAN_SCHEMA and
        value.get('protocol')==protocol() and value.get('task_contract')==runtime.task_contract() and
        value.get('production_ready') is True and value.get('simulation_only') is True and
        value.get('scope')=='trainval_smoke_only' and all(value.get(k) is False for k in
        ('test_payloads_opened','physical_acceptance','real_hardware_validated','closed_loop_evaluated')),
        'Only explicit finite simulation smoke permit is admitted; no old/pilot/fixture/hardware/TEST route')
    need(set(value.get('model',{}))=={'checkpoint','complete'}, 'Exact selected weights and complete run required')
    for ref in [value['admission'],value['allocation_mapping'],*value['model'].values()]:runtime._reference_shape(ref)
    need(value['allocation_mapping']['sha256']==runtime.dataset.ALLOCATION_MAPPING_SHA256, 'Original frozen pose allocation mapping required')
    rows=value.get('attempts');need(isinstance(rows,list) and len(rows)==4, 'Exactly four fixed native TRAIN/VAL attempts required')
    need(len({r.get('episode_id') for r in rows})==len({r.get('attempt_id') for r in rows})==
         len({r.get('target_id') for r in rows})==len({r.get('output') for r in rows})==4,
         'Four unique episodes/attempts/distinct targets/output paths required; no replacements')
    for split in ('train','validation'):
        selected=[r for r in rows if r.get('split')==split]
        need(len(selected)==2 and len({r.get('subject_id') for r in selected})==2 and
             {r.get('pose_template') for r in selected}==POSE_TEMPLATES,
             'Two distinct TRAIN and two distinct VAL identities must each cover both fixed known poses')
    for row in rows:
        need(set(row)=={'episode_id','attempt_id','job_id','owner_run_id','output','subject_id','split','target_id',
            'metadata','seed','runtime_config_sha256','pose_template','pose_binding'}, 'Exact frozen smoke attempt fields required')
        runtime.shared.require_membership(row['split'],row['subject_id'])
        need(type(row['seed']) is int and 0<=row['seed']<2**32 and all(isinstance(row[k],str) and row[k] and
            all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in row[k]) for k in
            ('episode_id','attempt_id','job_id','owner_run_id')), 'Fixed integer seed and safe attempt/owner identities required')
        need(row['output']=='/workspace/wam/runs/ausc_policy/'+row['owner_run_id']+'/output/'+row['job_id'],
             'Smoke output must be the declared original batch owner job')
        runtime._reference_shape(dict(path=row['output'],sha256=row['runtime_config_sha256']))
        meta=row['metadata'];ref=runtime.pose.require_route(meta)
        need(meta.get('handoff_policy_id')==runtime.handoff.POLICY_ID and all(meta.get(k)==row[k] for k in
            ('episode_id','subject_id','split','target_id')) and row['target_id'] in {s+str(i) for s in 'LR' for i in range(1,6)},
             'Immutable fresh runtime reference/identity/handoff metadata differs')
        need(set(row['pose_binding'])=={'patient_receipt','preparation_config'} and
            row['pose_binding']['patient_receipt']==ref.snapshot()['sources']['patient_receipt'],
             'Known pose must bind the authenticated patient receipt')
    return value


def runtime_config_sha256(config):
    return runtime.digest({k:v for k,v in config.items() if k!='m2_given_contact_deployment'})


def module_inventory():
    from . import m2_given_contact_train, m2_given_contact_admission, m2_given_contact_controller
    from . import m2_given_contact_intent, fresh_expert, geometry_qualification, run, scene_store
    from . import m2_given_contact_lossless_mapping, layered_scene_authoring
    inventory={m.__name__:runtime.reference(m.__file__) for m in (runtime,runtime.given,runtime.pose,runtime.handoff,
        m2_given_contact_train,m2_given_contact_admission,m2_given_contact_controller,m2_given_contact_intent,
        fresh_expert,geometry_qualification,run,scene_store,m2_given_contact_lossless_mapping,layered_scene_authoring)}
    inventory[MODULE]=runtime.reference(__file__)
    return inventory


def _training_owner(admission,complete):
    """Portable metadata verification: native Python never imports official ACT."""
    from . import process_guard as guard
    ref=admission['cuda_owner_terminal'];terminal=runtime.read_bound(ref);start=complete['owned_stage']
    run=runtime.read_bound(start['run']);stage_config=runtime.read_bound(start['config'])
    folder=runtime.bound_path(ref).parent;stage=terminal.get('stage',{})
    expected=[runtime.VENDOR_ROOT+'/.venv/bin/python','-u','-B','-m','sim.ausc_policy.m2_given_contact_train',
        '--config',start['config']['path'],'--output',str(folder/'output')]
    need(runtime.bound_path(ref)==folder/'result.json' and runtime.bound_path(start['run'])==folder/'run.json' and
        runtime.bound_path(admission['model']['complete'])==folder/'output/COMPLETE.json' and
        stage_config.get('schema')==runtime.OWNED_STAGE_SCHEMA and stage_config.get('experiment')=='m2_given_contact_train' and
        stage_config.get('time_limit_seconds')==21600 and stage_config.get('input')==complete['inputs']['config'] and
        start.get('input')==stage_config['input'] and start.get('output')==str(folder/'output') and
        start.get('run_id')==folder.name and stage_config.get('base')==str(folder.parents[2]),
        'Original formal training owner/config/input/output path differs')
    need(terminal.get('completed') is True and terminal.get('gpu_released') is True and
        all(terminal.get(k)==run.get(k) for k in ('run_id','code_commit','config','config_sha256','command','resource_guard','started_utc')) and
        terminal.get('command')==expected and run.get('config')==stage_config and run.get('config_sha256')==start['config']['sha256'] and
        stage.get('passed') is True and stage.get('exit_code')==0 and stage.get('process_group_drained') is True and
        stage.get('errors')==stage.get('signals')==stage.get('deferred_signals')==[],
        'Original normal training exit, group drain and GPU release are required')
    process=runtime.reference(folder/'lifecycle/m2_given_contact_train.process.json')
    resources=runtime.read_bound(complete['resources']);identity=resources['identity'];ownership=resources['owner']
    source=runtime.reference(guard.__file__)
    need(source['sha256']==runtime.GUARD_SHA256 and start.get('guard')==source and
        terminal.get('resource_guard')==dict(path=source['path'],sha256=source['sha256'],gpu_exit_grace_s=guard.EXIT_GPU_GRACE_S) and
        runtime.read_bound(process)==stage and runtime._same_process(stage.get('process_identity'),identity) and
        runtime._same_process(start.get('child_identity'),identity) and ownership.get('path')==runtime.GPU_LOCK and
        (ownership.get('device'),ownership.get('inode'))==runtime.GPU_LOCK_ID and ownership.get('pid')==identity['pid'] and
        ownership.get('checks')==dict(gpu_pids_before=[]), 'Original guard, child identity or one-lock evidence differs')
    expected_audit=dict(passed=True,terminal=ref,run=start['run'],process=process,complete=admission['model']['complete'],
        child_identity={k:identity[k] for k in ('pid','pgid','sid','start_ticks')},ownership=ownership)
    need(admission.get('cuda_owner_audit')==expected_audit, 'Admission original owner proof changed')
    return expected_audit


def _admission_metadata(admission_ref):
    """Rebind sealed numerical evidence; do not decode 240 episodes or infer 40 models."""
    from . import m2_given_contact_train as train
    value=runtime.read_bound(admission_ref)
    need(value.get('schema')==runtime.given.ADMISSION_SCHEMA and value.get('formal_run_schema')==runtime.FORMAL_RUN_SCHEMA and
        value.get('task_contract')==runtime.task_contract() and value.get('passed') is True and
        value.get('test_payloads_opened') is False and value.get('test_sealed') is True and
        value.get('production_ready') is False and value.get('production_bridge_pending') is True and
        value.get('physical_acceptance') is False and value.get('cpu_fixture_only',False) is False and
        value.get('comparison',{}).get('passed') is True and value['comparison'].get('cpu_byte_identical') is True and
        isinstance(value.get('model'),dict) and set(value['model'])=={'checkpoint','complete'} and
        isinstance(value.get('cpu_results'),list) and len(value['cpu_results'])==2,
        'Passed complete formal admission and two fresh CPUs required; old/pilot/fixture receipts refused')
    complete=runtime.validate_complete(runtime.read_bound(value['model']['complete']))
    need(value['model']['checkpoint']==complete['best'] and value.get('inputs')==complete['inputs'] and
        value.get('processing')==complete['processing'] and value.get('normalization')==complete['inputs']['normalization'] and
        value.get('reference_inventory')==complete['inputs']['reference_inventory'] and
        value.get('source_inventory')==complete['inputs']['software'], 'Immutable selected model/stats/source binding differs')
    runtime.bound_path(value['model']['checkpoint'])
    config=train.validate_config(runtime.read_bound(complete['inputs']['config']))
    need(config['dataset']==complete['inputs']['dataset'] and config['processing']==complete['processing'], 'Sealed training input/config differs')
    stats=runtime.validate_normalization(runtime.read_bound(value['normalization']))
    runtime.read_bound(value['reference_inventory'])
    software=runtime.read_bound(value['source_inventory'])
    need(software.get('schema')==runtime.RUNTIME_SCHEMA and software.get('prefix')==runtime.dataset.OFFICIAL_ACT_PREFIX and
        software.get('vendor',{}).get('root')==runtime.VENDOR_ROOT and software['vendor'].get('upstream_commit')==runtime.UPSTREAM_COMMIT,
        'Original full official software inventory required')
    runtime.bound_path(software['python'])
    runtime.validate_execution_settings(complete.get('execution',{}),device='cuda')
    workers=[runtime.read_bound(ref) for ref in value['cpu_results']]
    need(workers[0]['identity']!=workers[1]['identity'] and workers[0]['identity']['pid']!=workers[1]['identity']['pid'], 'Two distinct CPU model loads required')
    for worker in workers:
        need(worker.get('schema')=='m2_given_contact_formal_cpu_reconstruction_v1' and worker.get('model')==value['model'] and
            worker.get('inputs')==complete['inputs'] and worker.get('window_order')==complete['window_order'] and
            worker.get('source_audit_passed') is True and worker.get('test_payloads_opened') is False,
            'Immutable CPU reconstruction record differs')
        runtime.validate_execution_settings(worker.get('execution',{}),device='cpu')
    need(runtime.earliest_argmin(workers[0]['evaluations'])==complete['selected_step'] and
        all(a['checkpoint']==b['checkpoint'] for a,b in zip(workers[0]['evaluations'],complete['evaluations'])) and
        workers[0]['selected_metrics']==workers[1]['selected_metrics']==value['comparison']['metrics'] and
        runtime.admission_gate(workers[0]['selected_metrics']['train'])==value['comparison']['train_gate'] and
        runtime.admission_gate(complete['selected_metrics']['train'])==value['comparison']['cuda_train_gate'] and
        value['comparison']['train_gate']['passed'] is True and value['comparison']['cuda_train_gate']['passed'] is True,
        'Original 40-snapshot selection, complete selected metrics or exact dual TRAIN gate differs')
    owner=_training_owner(value,complete)
    for ref in (value['input_rebinding'],complete['input_rebinding']):
        need(runtime.read_bound(ref).get('passed') is True, 'Exact terminal input rebinding receipt required')
    return dict(admission=value,complete=complete,config=config,normalization=stats,software=software,owner=owner)


def _sources(binding):
    plan=runtime.read_bound(binding['plan']);base=runtime.bound_path(binding['plan']).parent;scope=plan['accounting_scope']
    runtime.dataset.metadata_accounting(scope,base,snapshot_phase='pre_export')
    report=runtime.dataset.processing_source_accounting(scope['processing_sources'],base,snapshot_phase='pre_export',
        excluded_paths=[*scope['owner_roots'],scope['operational_metadata']['root'],scope['final_seal']['path'],
            *(e['reference']['path'] for e in scope['extra_input_refs'])])
    need(report.get('pending') is False and report.get('descriptor')==binding['processing_freeze'], 'Final immutable processing source descriptor required')
    descriptor=runtime.read_bound(binding['processing_freeze']);inventory=module_inventory()
    root=Path(scope['processing_sources']['live_root'])
    for ref in inventory.values():
        path=runtime.bound_path(ref)
        need(root in path.parents and descriptor['files'].get(path.relative_to(root).as_posix())==ref,
            'Bridge/native module outside its complete final processing source freeze')
    return inventory


def _reference_roles(row,mapping):
    """Authenticate every leaf role using original allocation JSON, without I/O."""
    runtime.shared.require_membership(row.get('split'),row.get('subject_id'))
    def file_ref(ref):
        need(isinstance(ref,dict) and {'file','sha256'}<=set(ref)<= {'file','sha256','bytes'},
            'Original frozen allocation file/SHA role required')
        runtime._reference_shape(dict(path=ref['file'],sha256=ref['sha256'],**({'bytes':ref['bytes']} if 'bytes' in ref else {})))
        return {key:ref[key] for key in ('file','sha256')}
    allocations=[a for a in mapping.get('ordered_allocations',{}).get(row['split'],[]) if
        a.get('subject_id')==row['subject_id'] and a.get('target')==row['target_id']]
    need(len(allocations)==1, 'Original frozen allocation must authenticate the exact subject/target before source reads')
    allocation=allocations[0]
    scenes=[s for s in mapping.get('scenes',[]) if s.get('subject_id')==row['subject_id'] and s.get('split')==row['split'] and
        s.get('proposal_id')==allocation.get('proposal_id')==allocation.get('scene_ref') and
        s.get('template')==row['pose_template']]
    need(len(scenes)==1, 'Original frozen allocation must authenticate the exact scene/template before source reads')
    scene=scenes[0];patient=file_ref(scene.get('patient_receipt'));prep=file_ref(scene.get('preparation_config'))
    # The reviewed given builder uses the installed patient receipt as its task
    # surface manifest; caller-supplied alias/content equivalence is insufficient.
    expected=dict(task_manifest=patient,patient_receipt=patient,fresh_plan=file_ref(allocation.get('fresh_plan')))
    ref=runtime.pose.require_route(row['metadata'])
    need(ref.snapshot()['sources']==expected and row.get('pose_binding',{}).get('patient_receipt')==patient and
        file_ref(row.get('pose_binding',{}).get('preparation_config'))==prep,
        'Original frozen allocation reference/preparation source roles differ before source reads')
    return ref


def _pose_binding(row,mapping):
    _reference_roles(row,mapping)
    ref=runtime.dataset._reference_metadata(row['metadata'])
    from .fresh_expert import contact_plane_contract
    plane=contact_plane_contract(runtime.pose.bound_json(ref.snapshot()['sources']['fresh_plan']))
    need(plane is not None and plane['plane_tilt_limit_deg']==3. and plane['tilt_relaxed'] is False and
        plane['contact_plane_radius_m']==.023, 'Original nonrelaxed 23 mm/3 degree local plane required')
    return ref


def _allocation_mapping(ref,processing):
    """Bind the original metadata role, including its absolute source path."""
    def normalize(value):
        need(isinstance(value,dict) and set(value) in ({'path','sha256'},{'path','sha256','bytes'},
            {'file','sha256'},{'file','sha256','bytes'}), 'Exact original allocation mapping reference required')
        result=dict(path=value.get('path',value.get('file')),sha256=value['sha256'])
        if 'bytes' in value:result['bytes']=value['bytes']
        runtime._reference_shape(result);return result
    plan=runtime.read_bound(processing['plan']);original=normalize(plan.get('allocation_mapping'))
    supplied=normalize(ref)
    need(all(supplied[k]==original[k] for k in ('path','sha256')) and
        original['sha256']==runtime.dataset.ALLOCATION_MAPPING_SHA256,
        'Permit allocation mapping must use the exact original admitted path/SHA role')
    mapping=runtime.read_bound(original);size=runtime.bound_path(original).stat().st_size
    need(all('bytes' not in value or type(value['bytes']) is int and value['bytes']==size for value in (supplied,original)),
        'Original allocation mapping declared bytes changed')
    if 'derivation' in plan:
        from .m2_given_contact_lossless_mapping import authenticated_view
        mapping=authenticated_view(mapping,plan)
    return mapping,dict(path=original['path'],sha256=original['sha256'],bytes=size)


def verify_permit(ref):
    value=runtime.read_bound(ref)
    if value.get('schema')=='m2_given_contact_sealed_test_capability_v1':
        from .m2_given_contact_evaluation import verify_capability
        return verify_capability(ref,stage='native')
    value=_validate_permit(value);bundle=_admission_metadata(value['admission'])
    need(value['model']==bundle['admission']['model'] and value['processing']==bundle['config']['processing'] and
        value['source_inventory']==_sources(value['processing']), 'Permit model/source/common processing binding differs')
    mapping,_=_allocation_mapping(value['allocation_mapping'],value['processing'])
    for row in value['attempts']:_reference_roles(row,mapping)
    for row in value['attempts']:_pose_binding(row,mapping)
    return value


def freeze_smoke_permit(admission_ref,allocation_mapping,attempts,output):
    """CPU metadata freeze only. Four actual native smokes remain required."""
    bundle=_admission_metadata(admission_ref);processing=bundle['config']['processing']
    value=dict(schema=PERMIT_SCHEMA,plan_schema=SMOKE_PLAN_SCHEMA,protocol=protocol(),task_contract=runtime.task_contract(),
        production_ready=True,simulation_only=True,scope='trainval_smoke_only',test_payloads_opened=False,
        physical_acceptance=False,real_hardware_validated=False,closed_loop_evaluated=False,
        admission=admission_ref,model=bundle['admission']['model'],processing=processing,
        source_inventory=_sources(processing),allocation_mapping=allocation_mapping,attempts=deepcopy(attempts))
    mapping,value['allocation_mapping']=_allocation_mapping(allocation_mapping,processing)
    _validate_permit(value)
    for row in attempts:_reference_roles(row,mapping)
    for row in attempts:
        _pose_binding(row,mapping)
        need(not Path(row['output']).exists() and not Path(row['output']).parent.parent.exists(), 'Smoke owner/output already exists')
    return runtime.write_new(output,value)


def _current_scene_metadata(config,row,mapping,recipe,plan):
    """Authenticate CURRENT provenance before resolving historical prepared files."""
    from . import m2_given_contact_lossless_mapping as lossless,scene_store
    policy=scene_store.SDK_LAYER_CURRENT_RAW_POLICY;binding=config.get('compact_scene_storage')
    conditions=recipe.get('collector_conditions');dependencies=conditions.get('scene_dependencies') if isinstance(conditions,dict) else None
    need(isinstance(plan,dict) and plan.get('current_raw_equivalence') is True and
        recipe.get('schema')==lossless.RECIPE_SCHEMA and recipe.get('current_raw_equivalence') is True and
        recipe.get('scene_dependency_policy')==policy and
        isinstance(dependencies,dict) and dependencies.get('dependency_policy')==policy and
        isinstance(binding,dict) and binding.get('schema')==scene_store.SDK_LAYER_SCHEMA and binding.get('dependency_policy')==policy,
        'Explicit current RAW plan/recipe/runtime policy required')
    reader=lossless._CurrentMetadata(plan);sources=lossless._current_sources(reader,binding)
    original_ref=reader.bind(plan['allocation_mapping'])
    need(original_ref['sha256']==runtime.dataset.ALLOCATION_MAPPING_SHA256, 'Original admitted current RAW mapping required')
    original=reader.read(original_ref);selected,allocated,_=lossless._population(original)
    report=reader.read(plan['derivation']);bound_recipe=reader.read(plan['source_recipe'])
    root=Path(lossless._reference(plan['derivation'])['path']).parent
    lossless._current_headers(report,bound_recipe,selected,allocated,plan,reader,root)
    need(lossless._same(recipe,bound_recipe), 'Runtime recipe differs from its registered current RAW source')
    # Reject every copied allocation/scene header before opening any leaf JSON.
    expected=deepcopy(original);scenes={s['proposal_id']:s['derived'] for s in report['scenes']}
    for i,scene in enumerate(expected['scenes']):
        if scene['proposal_id'] in scenes:expected['scenes'][i]=deepcopy(scenes[scene['proposal_id']])
    for entry,allocation in zip(report['plans'],[a for split in ('train','validation') for a in expected['ordered_allocations'][split]]):
        allocation['fresh_plan']=deepcopy(entry['derived'])
    need(lossless._same(mapping,expected), 'Complete current RAW allocation/source headers differ')
    receipt=_reference_roles(row,mapping).snapshot()['sources']['patient_receipt']
    matches=[s for s in mapping['scenes'] if s['subject_id']==row['subject_id'] and s['split']==row['split'] and
        lossless._file(s['patient_receipt'])==receipt]
    need(len(matches)==1, 'Exactly one current RAW TRAIN/VAL prepared scene required')
    scene=matches[0];index=next(i for i,s in enumerate(report['scenes']) if s['proposal_id']==scene['proposal_id'])
    proof=report['layered_authoring']['runtime_equivalence'][index]
    need(lossless._same(sources,proof['derived_sources']), 'Actual current RAW runtime source roles differ before leaf JSON')
    # The accepted adapter validates every registered JSON role, normalization,
    # source freeze, sidecar and typed proof across all 240 attempts. No payload
    # reader, USD validator or oracle issuer is involved in this authentication.
    need(lossless._same(mapping,lossless.authenticated_view(original,plan)), 'Authenticated current RAW copied view differs')
    result=scene['derivation'];record=scene['current_raw'];conditions=bound_recipe['collector_conditions']
    prepared=reader.bind(scene['preparation_config']);prep=reader.read(prepared);complete=reader.read(sources['prepared_complete'])
    patient=reader.read(sources['patient_receipt']);historical=reader.read(record['historical_prepared_complete'])
    sidecar=reader.read(record['layered_runtime_equivalence'])
    need(lossless._same(sources['preparation_config'],dict(file=prepared['path'],bytes=prepared['bytes'],sha256=prepared['sha256'])) and
        lossless._file(sources['prepared_complete'])==scene['prepared_complete']==record['prepared_complete'] and
        lossless._file(sources['patient_receipt'])==receipt and
        lossless._same(sources['layered_runtime_equivalence'],record['layered_runtime_equivalence']) and
        lossless._same(sidecar['historical_prepared_complete'],record['historical_prepared_complete']) and
        lossless._same(complete,dict(historical,scene_dependency_policy=policy,layered_runtime_equivalence=record['layered_runtime_equivalence'])),
        'Exact current COMPLETE/sidecar/historical prepared lineage required')
    inventory=result['dependency_inventory']
    need(inventory['schema']==lossless.layers.INVENTORY_SCHEMA and
        lossless._same(config.get('physics'),conditions['physics']) and lossless._same(config.get('physics'),lossless.layers.ITERATIONS) and
        lossless._same(prep.get('physics'),lossless.layers.ITERATIONS) and
        lossless._same(sources['scene'],inventory['primary']) and lossless._same(sources['scene_base'],inventory['base']) and
        lossless._same(sources['scene_authoring_receipt'],result['scene_authoring_receipt']),
        'Exact current RAW layered scene/profile/source conditions required')
    need(lossless._same(config.get('patient_asset'),prep.get('patient_asset')) and
        config.get('patient_asset',{}).get('expected_split','train')==row['split'] and
        all(lossless._same(config.get(k),prep.get(k)) for k in ('geometry_id','geometry_source','geometry_source_sha256','posture_limits','freshly_generated')) and
        lossless._same({k:v for k,v in config.get('fresh_expert',{}).items() if k!='plans'},conditions['fresh']) and
        config.get('isaac_python')==conditions['isaac_python'], 'Original prepared patient/geometry/posture/native gates changed')
    need(complete.get('completed') is True and lossless._same(complete.get('patient'),patient) and
        patient.get('identity')==row['subject_id'] and patient.get('split')==row['split'] and
        patient.get('asset_sha256')==row['metadata'].get('patient_asset_sha256') and
        patient.get('non_patient_scene_unchanged') is True, 'Runtime body identity differs before payload access')
    directory=Path(record['historical_prepared_complete']['file']).parent
    need(lossless._file(record['historical_prepared_complete'])==result['prepared_complete'] and
        sources['preparation_config']['file']==str(directory/'prepare.json'), 'Exact historical factored prepared root required')
    for name,filename in {**scene_store.FILES,**scene_store.OPTIONAL}.items():
        if name in sources:need(sources[name]['file']==str(directory/filename), 'Prepared body payload path substitution')
    hashes=complete.get('geometry_artifacts',{})
    need(sources['scene']['sha256']==complete.get('effective_scene_sha256') and
        sources['collision_geometry']['sha256']==hashes.get('collision_geometry_sha256') and
        sources['collision_arrays']['sha256']==hashes.get('collision_arrays_sha256'), 'Prepared body payload hash substitution')
    sdk={k:v for k,v in sources.items() if k=='sdk_mdl_manifest' or scene_store._sdk_file_name(k)}
    need(lossless._same(dict(schema=binding['schema'],dependency_policy=binding['dependency_policy'],sources=sdk),conditions['scene_dependencies']),
        'Original qualified SDK source/schema/policy changed')


def _scene_metadata(config,row,mapping,recipe,*,plan=None):
    """Bind identity and every body payload name using JSON only, before USD/NPZ."""
    from . import scene_store
    current_binding=config.get('compact_scene_storage')
    if ('current_raw_equivalence' in recipe or 'scene_dependency_policy' in recipe or
        (isinstance(current_binding,dict) and current_binding.get('dependency_policy')==scene_store.SDK_LAYER_CURRENT_RAW_POLICY) or
        any('current_raw' in s for s in mapping.get('scenes',[]))):
        return _current_scene_metadata(config,row,mapping,recipe,plan)
    def file_ref(ref):return {k:ref[k] for k in ('file','sha256')}
    def read(ref):return runtime.read_bound(dict(path=ref['file'],sha256=ref['sha256']))
    receipt=runtime.pose.require_route(row['metadata']).snapshot()['sources']['patient_receipt']
    scenes=[s for s in mapping.get('scenes',[]) if s.get('subject_id')==row['subject_id'] and s.get('split')==row['split'] and
        s.get('patient_receipt',{}).get('file')==receipt['file'] and s.get('patient_receipt',{}).get('sha256')==receipt['sha256']]
    need(len(scenes)==1, 'Exactly one authenticated TRAIN/VAL prepared scene required')
    scene=scenes[0];prep=read(scene['preparation_config']);binding=config.get('compact_scene_storage')
    scene_store._binding_shape(binding);sources=binding['sources'];conditions=recipe['collector_conditions']
    if 'derivation' in scene:
        from . import m2_given_contact_lossless_mapping as lossless
        result=scene['derivation'];inventory=result.get('dependency_inventory',{})
        need(recipe.get('schema')==lossless.RECIPE_SCHEMA and
            binding['schema']==scene_store.SDK_LAYER_SCHEMA and binding['dependency_policy']==scene_store.SDK_LAYER_DEPENDENCY_POLICY and
            config.get('physics')==conditions.get('physics')==lossless.layers.ITERATIONS and
            inventory.get('schema')==lossless.layers.INVENTORY_SCHEMA and
            sources['scene']==inventory.get('primary') and sources['scene_base']==inventory.get('base') and
            sources['scene_authoring_receipt']==result.get('scene_authoring_receipt'),
            'Exact reviewed layered scene/profile/source conditions required')
    need(file_ref(sources['patient_receipt'])==receipt and
        file_ref(sources['prepared_complete'])==file_ref(scene['prepared_complete']) and
        config.get('patient_asset')==prep.get('patient_asset') and
        config.get('patient_asset',{}).get('expected_split','train')==row['split'] and
        all(config.get(k)==prep.get(k) for k in ('geometry_id','geometry_source','geometry_source_sha256','posture_limits','freshly_generated')) and
        {k:v for k,v in config.get('fresh_expert',{}).items() if k!='plans'}==conditions['fresh'] and
        config.get('isaac_python')==conditions['isaac_python'], 'Original prepared patient/geometry/posture/native gates changed')
    complete=read(sources['prepared_complete']);patient=read(sources['patient_receipt'])
    need(complete.get('completed') is True and complete.get('patient')==patient and
        patient.get('identity')==row['subject_id'] and patient.get('split')==row['split'] and
        patient.get('asset_sha256')==row['metadata'].get('patient_asset_sha256') and
        patient.get('non_patient_scene_unchanged') is True, 'Runtime body identity differs before payload access')
    root=Path(sources['prepared_complete']['file']).parent
    for name,filename in {**scene_store.FILES,**scene_store.OPTIONAL}.items():
        if name in sources:need(sources[name]['file']==str(root/filename), 'Prepared body payload path substitution')
    hashes=complete.get('geometry_artifacts',{})
    need(sources['scene']['sha256']==complete.get('effective_scene_sha256') and
        sources['collision_geometry']['sha256']==hashes.get('collision_geometry_sha256') and
        sources['collision_arrays']['sha256']==hashes.get('collision_arrays_sha256'), 'Prepared body payload hash substitution')
    sdk={k:v for k,v in sources.items() if k=='sdk_mdl_manifest' or scene_store._sdk_file_name(k)}
    need(dict(schema=binding['schema'],dependency_policy=binding['dependency_policy'],sources=sdk)==conditions['scene_dependencies'],
        'Original qualified SDK source/schema/policy changed')


def preflight(config,*,output=None):
    if 'm2_given_contact_evaluation' in config:
        from .m2_given_contact_evaluation import preflight as heldout_preflight
        binding=heldout_preflight(config,output=output)
        need(binding['stage']=='native' and binding['row']['arm']=='act' and
             'm2_given_contact_control' not in config and 'm2_act_deployment' not in config and
             'raw_multimodal_capture' not in config.get('fresh_expert',{}), 'Exact capture-free sealed ACT arm required')
        return dict(deployment=deepcopy(config['m2_given_contact_deployment']),permit=binding['capability'],attempt=binding['row'])
    dep=config.get('m2_given_contact_deployment')
    need(isinstance(dep,dict) and set(dep)=={'schema','production_ready','simulation_only','permit','episode_id','metadata'} and
        dep.get('schema')==runtime.given.DEPLOYMENT_SCHEMA and dep.get('production_ready') is True and dep.get('simulation_only') is True,
        'Explicit admitted formal simulation deployment required')
    need('m2_act_deployment' not in config and 'm2_given_contact_control' not in config and
        'raw_multimodal_capture' not in config.get('fresh_expert',{}), 'Formal learned route cannot mix teacher, pilot or expert capture labels')
    permit=verify_permit(dep['permit']);row=_attempt(dep,permit)
    need(dep['metadata']==row['metadata'] and runtime_config_sha256(config)==row['runtime_config_sha256'],
        'Original predeclared smoke config/reference differs')
    if output is not None:need(str(runtime.dataset._path(str(output),Path.cwd()))==row['output'], 'Original frozen native smoke output path differs')
    plans=config.get('fresh_expert',{}).get('plans',[]);ref=runtime.pose.require_route(dep['metadata'])
    need(len(plans)==1 and {k:plans[0].get(k) for k in ('file','sha256')}==ref.snapshot()['sources']['fresh_plan'],
        'Runtime fresh plan differs from prescribed full pose source')
    plan=runtime.read_bound(permit['processing']['plan'])
    prep_ref=config.get('compact_scene_storage',{}).get('sources',{}).get('preparation_config',{})
    need(any(e.get('category')=='config' and all(e.get('reference',{}).get(k)==prep_ref.get('file' if k=='path' else k)
        for k in ('path','sha256','bytes')) for e in plan['accounting_scope']['extra_input_refs']),
        'Runtime preparation must be an exact authenticated original config input before any payload read')
    mapping,_=_allocation_mapping(permit['allocation_mapping'],permit['processing'])
    _scene_metadata(config,row,mapping,runtime.read_bound(plan['source_recipe']),plan=plan)
    return dict(deployment=deepcopy(dep),permit=permit,attempt=row)


def _official_python(software):
    python=Path(runtime.VENDOR_ROOT)/'.venv/bin/python'
    need(python.is_file() and runtime.reference(python.resolve())==software['python'], 'Exact official CPU interpreter binary required')
    return str(python)


def _ready_binding(dep,permit,software_ref,software):
    row=_attempt(dep,permit)
    return dict(ready=True,schema=runtime.given.PROTOCOL_SCHEMA,protocol_sha256=runtime.digest(protocol()),
        permit=dep['permit'],model=permit['model'],episode_id=row['episode_id'],runtime_config_sha256=row['runtime_config_sha256'],
        metadata_sha256=runtime.digest(row['metadata']),source_inventory=permit['source_inventory'],
        software=software_ref,python=software['python'],prefix=runtime.dataset.OFFICIAL_ACT_PREFIX)


def _same_live_process(left,right):
    """Scheduler state can change between samples; process identity cannot."""
    for value in (left,right):
        if not(isinstance(value,dict) and {'pid','pgid','sid','start_ticks','state'}<=set(value) and
            all(type(value[key]) is int and value[key]>0 for key in ('pid','pgid','sid')) and
            isinstance(value['start_ticks'],str) and value['start_ticks'].isdigit() and
            isinstance(value['state'],str) and len(value['state'])==1 and value['state'] in 'RSDTtWPIK'):
            return False
    return {k:v for k,v in left.items() if k!='state'}=={k:v for k,v in right.items() if k!='state'}


class CPUBridge:
    """One original native-session child, with no listening service or fallback."""
    def __init__(self,deployment,*,output):
        from . import process_guard as guard
        need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1', 'Original owned Linux native parent required')
        self.dep=deepcopy(deployment);self.permit=verify_permit(self.dep['permit']);self.process=None
        self.identity=None;self.context=None;self.sequence=0;self.previous=None;self.busy=False
        self.exit_receipt=None;self.stderr=None;self.trace=None;self.output=Path(output)
        admission=runtime.read_bound(self.permit['admission']);software_ref=admission['source_inventory']
        software=runtime.read_bound(software_ref);python=_official_python(software)
        self.parent=guard.process_identity(os.getpid());need(self.parent is not None, 'Native parent identity unavailable')
        root=Path(__file__).resolve().parents[2];self.output.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(root),
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        for key in ('PYTHONHOME','PYTHONUSERBASE','HF_TOKEN'):env.pop(key,None)
        ref=self.dep['permit'];command=[python,'-u','-B','-m',MODULE,'--serve','--permit',ref['path'],
            '--permit-sha256',ref['sha256'],'--episode-id',self.dep['episode_id']]
        try:
            self.stderr=(self.output/'given_act_cpu_stderr.log').open('xb')
            self.trace=(self.output/'given_act_ipc.jsonl').open('x')
            self.process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=self.stderr,start_new_session=False,close_fds=True,bufsize=0)
            self.identity=guard.process_identity(self.process.pid)
            need(self.identity is not None and self.identity['pid']==self.process.pid and
                all(self.identity[k]==self.parent[k] for k in ('pgid','sid')), 'CPU child left original native process group/session')
            self.pipe=FramedPipe(self.process.stdout,self.process.stdin);ready=self.pipe.read(timeout=STARTUP_S)
            expected=_ready_binding(self.dep,self.permit,software_ref,software)
            need(set(ready)==set(expected)|{'identity','parent_identity','execution'} and
                all(ready.get(k)==v for k,v in expected.items()) and self.process.poll() is None and
                _same_live_process(ready['identity'],self.identity) and _same_live_process(ready['parent_identity'],self.parent),
                'CPU child ready/model/module/source/full-reference/identity binding differs')
            runtime.validate_execution_settings(ready['execution'],device='cpu')
            self.runtime=ready;self._trace(dict(event='ready',value=ready))
        except BaseException:
            self.close();raise

    def _trace(self,value):
        if self.trace is not None:
            self.trace.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');self.trace.flush()

    def __call__(self,measured_q,estimated_target_base,confidence,*,metadata,history,history_times):
        need(not self.busy and self.process is not None and self.process.poll() is None and isinstance(self.context,dict),
             'One live child and one explicit same-time request required')
        self.busy=True;self.sequence+=1;started=time.monotonic();row=dict(event='inference',request_id=self.sequence,accepted=False)
        try:
            request=make_request(self.dep,self.permit,self.sequence,measured_q,estimated_target_base,confidence,
                metadata=metadata,history=history,history_times=history_times,context=self.context)
            state=validate_request(request,self.dep,self.permit,previous=self.previous)
            row['request']=request
            response=self.pipe.exchange(request,timeout=REQUEST_S);row['response']=response
            block=decode_response(response,request);self.previous=state;row['accepted']=True
            return block
        except BaseException as error:
            row['error']=type(error).__name__+': '+str(error)
            raise
        finally:
            self.busy=False;row['wall_latency_s']=time.monotonic()-started
            row['simulated_time_pause_s']=0.
            try:self._trace(row)
            except BaseException:self.close();raise
            finally:
                if not row['accepted']:self.close()

    def close(self):
        if self.exit_receipt is not None:return self.exit_receipt
        child=self.process;signals=[];errors=[];code=None
        try:
            if child is not None:
                if child.stdin:
                    try:child.stdin.close()
                    except OSError as error:errors.append(str(error))
                try:code=child.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    # Popen's still-unreaped direct child is owned even if its
                    # startup handshake failed. Never signal the shared group.
                    for action,label in ((child.terminate,'SIGTERM'),(child.kill,'SIGKILL')):
                        if child.poll() is not None:break
                        action();signals.append(label)
                        try:code=child.wait(timeout=2.);break
                        except subprocess.TimeoutExpired:pass
                need(child.poll() is not None, 'Owned CPU child could not be reaped')
                code=child.returncode
        finally:
            if child is not None and child.stdout:child.stdout.close()
            for stream in (self.trace,self.stderr):
                if stream is not None:stream.close()
            self.trace=None;self.stderr=None
        receipt=dict(schema='m2_given_contact_cpu_child_exit_v1',identity=self.identity,parent_identity=self.parent,
            started=child is not None,reaped=child is None or child.poll() is not None,exit_code=code,
            normal_exit=code==0,signals=signals,errors=errors,shared_native_group_signaled=False,
            request_count=self.sequence,simulation_only=True)
        if child is not None:
            receipt.update(stderr=runtime.reference(self.output/'given_act_cpu_stderr.log'),
                trace=runtime.reference(self.output/'given_act_ipc.jsonl'))
            runtime.write_new(self.output/'given_act_cpu_exit.json',receipt)
        self.exit_receipt=receipt;return receipt


def _child_permit(ref):
    # The native parent's public preflight reconstructs the fresh reference
    # against patient/task/planner JSON before any scene payload. The child
    # rebinds that immutable permit and common source/model evidence without
    # opening the privileged fresh plan or exposing waypoints to inference.
    value=runtime.read_bound(ref)
    if value.get('schema')=='m2_given_contact_sealed_test_capability_v1':
        from .m2_given_contact_evaluation import verify_capability
        return verify_capability(ref,stage='native',child_only=True)
    value=_validate_permit(value);bundle=_admission_metadata(value['admission'])
    need(value['model']==bundle['admission']['model'] and value['processing']==bundle['config']['processing'] and
        value['source_inventory']==_sources(value['processing']), 'Child permit model/source binding differs')
    return value


def serve(permit_ref,episode_id):
    need(sys.platform=='linux' and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
        os.environ.get('PYTHONDONTWRITEBYTECODE')=='1', 'Owned official CPU child environment required')
    permit=_child_permit(permit_ref);rows=[r for r in permit['attempts'] if r['episode_id']==episode_id]
    need(len(rows)==1, 'CPU child episode outside fixed native smoke permit')
    row=rows[0];dep=dict(permit=permit_ref,episode_id=episode_id,metadata=row['metadata'])
    need(row.get('arm','act')=='act', 'A sealed expert arm cannot start a learned CPU model child')
    bundle=_admission_metadata(permit['admission'])
    from . import process_guard as guard
    identity=guard.process_identity(os.getpid());parent=guard.process_identity(os.getppid())
    need(identity is not None and parent is not None and all(identity[k]==parent[k] for k in ('pgid','sid')),
        'CPU child must retain native parent group and session')
    with redirect_stdout(sys.stderr):
        torch,policy,checkpoint,complete=runtime.load_checkpoint(permit['model'],device='cpu')
        need(checkpoint['normalization']==bundle['normalization'] and complete==bundle['complete'], 'Loaded model differs from immutable admission')
        predictor=runtime.ChunkPredictor(torch,policy,checkpoint,row['metadata'],device='cpu')
        execution=runtime.execution_settings(torch,policy,device='cpu')
    ready=dict(_ready_binding(dep,permit,bundle['admission']['source_inventory'],bundle['software']),
        identity=identity,parent_identity=parent,execution=execution)
    def send(value):
        payload=json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()+b'\n'
        need(len(payload)<=RESPONSE_BYTES, 'Oversized CPU child response');sys.stdout.buffer.write(payload);sys.stdout.buffer.flush()
    send(ready);previous=None
    while True:
        raw=sys.stdin.buffer.readline(REQUEST_BYTES+1)
        if not raw:break
        need(len(raw)<=REQUEST_BYTES and raw.endswith(b'\n'), 'Oversized or truncated CPU child request')
        request=wire_decode(raw);state=validate_request(request,dep,permit,previous=previous)
        with redirect_stdout(sys.stderr):
            block=predictor(request['measured_q'],request['estimated_target_base'],request['confidence'],metadata=request['metadata'],
                history=request['history'],history_times=request['history_times'])
        send(response_for(request,block.values));previous=state


from .m2_given_contact_controller import GivenContactACT as _GivenContactACT


class FormalGivenACT(_GivenContactACT):
    """IPC context and audit only; the original native controller executes E10."""
    def _tick(self,command,subphase,active=False,recovery=False):
        start=len(self.env.rows)
        try:return super()._tick(command,subphase,active=active,recovery=recovery)
        finally:
            if len(self.env.rows)>start:
                need(len(self.env.rows)==start+1, 'Formal action attribution requires exactly one original native tick')
                state=self.env.state();row=self.env.rows[-1]
                need(state['time_s']==row['time_s'], 'Formal actual pose must use this POST-step row/time')
                pose=np.eye(4);pose[:3,:3]=self.env.base[:3,:3].T@np.asarray(state['rotation_world'])
                pose[:3,3]=state['tcp_base']
                row['formal_act_attribution']=dict(schema='m2_given_contact_native_action_attribution_v1',
                    physics_row=len(self.env.rows),time_s=float(state['time_s']),subphase=subphase,
                    actual_control_owner=self.control_owner,reference_sha256=self.reference.sha256,
                    raw_intent=deepcopy(getattr(self,'_current_intent_audit',None)),
                    observed_tcp_pose_base=runtime.shared.contract._pose(pose,'actual POST-step TCP full pose').tolist())

    def _query(self,observation):
        self.predictor.context=dict(observed_at_s=float(observation['observed_at_s']),physics_row=len(self.env.rows),
            context=list(observation['context']),source_id=observation['source_id'])
        try:return super()._query(observation)
        finally:self.predictor.context=None

    def close_bridge(self):
        receipt=self.predictor.close();self.policy_receipt['cpu_child_exit']=receipt
        return receipt

    def run(self,points):
        try:return super().run(points)
        finally:self.close_bridge()


def make_controller(*args,deployment,estimate_provider,output,**kwargs):
    need(isinstance(deployment,dict) and deployment.get('schema')==runtime.given.DEPLOYMENT_SCHEMA and
        deployment.get('production_ready') is True and deployment.get('simulation_only') is True,
        'Qualified formal simulation permit required')
    permit=verify_permit(deployment['permit']);row=_attempt(deployment,permit)
    need(row.get('arm','act')=='act', 'Expert-arm capability cannot acquire the learned controller')
    need(deployment.get('metadata')==row['metadata'] and isinstance(estimate_provider,runtime.pose.GivenPoseProvider) and
        estimate_provider.reference.sha256==runtime.pose.require_route(row['metadata']).sha256,
        'Exact fresh prescribed runtime reference provider required')
    need(str(runtime.dataset._path(str(output),Path.cwd()))==row['output'] and args, 'Exact frozen native job output/environment required')
    env=args[0];before=env.state();rows=len(env.rows);q=np.asarray(before['q']).copy();stamp=before['time_s']
    bridge=CPUBridge(deployment,output=output)
    try:
        after=env.state()
        need(len(env.rows)==rows and after['time_s']==stamp and np.array_equal(after['q'],q),
             'Physics/feedback advanced during CPU child startup')
    except BaseException:bridge.close();raise
    receipt=dict(schema=runtime.given.DEPLOYMENT_SCHEMA,permit=deployment['permit'],admission=permit['admission'],model=permit['model'],
        metadata=deepcopy(row['metadata']),child_runtime=bridge.runtime,protocol=protocol(),state_only=True,
        production_ready=True,simulation_only=True,scope=permit['scope'],physical_acceptance=False,
        real_hardware_validated=False,learned_policy=True,expert_demonstration=False,expert_l2_fallback=False,
        training_eligible=False,task_contract=runtime.task_contract(),source_inventory=permit['source_inventory'],
        rollout_seed=row['seed'],seed_scope='Python/NumPy native control; no task/estimator noise',
        startup_observation=dict(physics_row=rows,time_s=stamp,measured_q=q.tolist()))
    try:
        return FormalGivenACT(*args,predictor=bridge,metadata=row['metadata'],episode_id=row['episode_id'],
            estimate_provider=estimate_provider,policy_receipt=receipt,**kwargs)
    except BaseException:bridge.close();raise


def tag_result(result,deployment,controller=None):
    """Preserve learned failure identity after the old expert result defaults."""
    from .m2_act_controller import tag_learned_result
    receipt=getattr(controller,'policy_receipt',dict(schema=runtime.given.DEPLOYMENT_SCHEMA,
        configured_deployment=deepcopy(deployment),controller_constructed=False,simulation_only=True))
    handoffs=getattr(controller,'l2_handoffs',[]) or [dict(target=deployment.get('metadata',{}).get('target_id'),
        l2_coarse_reached=False,l2_handoff_success=False,l3_started=False,l3_completed=False,expert_l2_fallback=False,
        failure_reason=result.get('failure_reason','Formal learned attempt did not construct a controller'))]
    tag_learned_result(result,receipt,handoffs)
    result.update(task_id=runtime.pose.TASK_ID,target_source=runtime.pose.TARGET_SOURCE,
        action_semantics_id=runtime.given.ACTION_SEMANTICS_ID,handoff_policy_id=runtime.handoff.POLICY_ID,
        executed_controller_class=MODULE+'.FormalGivenACT',m2_given_contact_deployment=deepcopy(deployment),
        l2_queries=deepcopy(getattr(controller,'l2_queries',[])),l2_actions=deepcopy(getattr(controller,'l2_actions',[])),
        raw_capture_complete=False,compact_capture_complete=False,raw_multimodal_training_capture=False,
        simulation_only=True,real_hardware_validated=False,bridge_smoke_qualified=False,
        observation_contract=runtime.pose.observation_contract(),
        scope='Finite TRAIN/VAL prescribed full-pose learned control; no heldout or sensor robustness claim')
    if controller is not None:
        result['given_reference']=controller.estimate_provider.snapshot()
        exit_receipt=controller.close_bridge();result['cpu_child_exit']=exit_receipt
        if exit_receipt.get('normal_exit') is not True or exit_receipt.get('reaped') is not True:
            result.update(success=False,physical_validation=False,failure_reason='Formal CPU child did not exit and reap normally')
    return result


def run_native_probe(app,stage,output,config,mass):
    from . import fresh_expert
    lifecycle={};deployment=config.get('m2_given_contact_deployment')
    try:
        binding=preflight(config,output=output)
        import random
        seed=binding['attempt']['seed'];random.seed(seed);np.random.seed(seed)
        return fresh_expert._run_probe(app,stage,output,config,mass,_learned_lifecycle=lifecycle)
    except BaseException as error:
        result=dict(success=False,physical_validation=False,diagnostic_completed=False,points=[],
            physics_steps=len(lifecycle['env'].rows) if lifecycle.get('env') is not None else 0,
            video=lifecycle.get('video'),geometry_id=runtime.geometry.GEOMETRY_ID,
            geometry_source_sha256=config.get('geometry_source_sha256'),failure_reason=str(error),
            errors=[dict(stage='formal_learned_preflight_or_finalization',error=repr(error))])
        tag_result(result,deployment if isinstance(deployment,dict) else {},lifecycle.get('controller'))
        try:runtime.write_new(Path(output)/'fresh_expert_result.json',result)
        except Exception as write_error:result['errors'].append(dict(stage='result_save',error=repr(write_error)))
        if not isinstance(error,Exception):raise
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve',action='store_true');parser.add_argument('--permit');parser.add_argument('--permit-sha256')
    parser.add_argument('--episode-id');parser.add_argument('--freeze-smoke',action='store_true')
    parser.add_argument('--admission');parser.add_argument('--admission-sha256')
    parser.add_argument('--allocation-mapping');parser.add_argument('--mapping-sha256')
    parser.add_argument('--attempts');parser.add_argument('--attempts-sha256');parser.add_argument('--output')
    args=parser.parse_args()
    def ref(path,sha):
        need(path and sha, 'Explicit immutable CLI path and SHA256 required')
        return dict(path=str(Path(path).resolve()),sha256=sha)
    need(args.serve is not args.freeze_smoke, 'Exactly one finite bridge operation required')
    if args.serve:
        need(args.episode_id and not any((args.output,args.admission,args.attempts,args.allocation_mapping)), 'Exact owned CPU serve arguments required')
        serve(ref(args.permit,args.permit_sha256),args.episode_id)
    else:
        need(args.output and not any((args.permit,args.episode_id)), 'New immutable finite smoke permit output required')
        result=freeze_smoke_permit(ref(args.admission,args.admission_sha256),ref(args.allocation_mapping,args.mapping_sha256),
            runtime.read_bound(ref(args.attempts,args.attempts_sha256)),args.output)
        print(json.dumps(dict(permit=result,physical_qualification=False,test_payloads_opened=False)))


if __name__=='__main__':main()
