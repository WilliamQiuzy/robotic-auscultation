"""Authenticate historical TRAIN student raw targets for fresh recovery collection.

This reads closed actual evidence. It neither invokes a policy nor qualifies a
failed episode. Retained student targets remain perturbations, never labels.
"""
import argparse
from pathlib import Path
import os
import sys

import numpy as np

from . import m2_given_contact_runtime as rt
from . import m2_given_contact_expanded_pilot_deployment as adapter
from . import m2_given_contact_evaluation as neutral

SCHEMA='m2_actual_student_raw_target_prefix_v1'
need=rt.need


def select_cut(requests,distance_m):
    need(type(distance_m) in (int,float) and distance_m in (.080,.060),
        'Predeclared first80/60mm query cut required')
    for request in requests:
        state=rt.shared.contract._array(request['history'],(2,17),'actual student history')[-1]
        if np.linalg.norm(state[7:10])<=distance_m:
            return request['context']['physics_row']
    raise ValueError('Student did not reach the predeclared query distance')


def extract_cells(events,blocks,*,cut_row):
    need(type(cut_row) is int and 0<cut_row<=len(events) and cut_row%12==0,
        'Actual complete source query boundary required')
    need(all(e.get('physics_row')==i for i,e in enumerate(events)), 'Missing or reordered native source tick')
    motion=[i for i,e in enumerate(events[:cut_row]) if e.get('actual_control_owner')=='ACT_L2' and
        e.get('raw_intent') is not None and e['raw_intent'].get('native_tick_in_cell')==0 and
        e['raw_intent'].get('setter',{}).get('execution_role')=='given_l2_motion']
    need(motion and 0<cut_row-motion[0]<=2400*12 and (cut_row-motion[0])%12==0,
        'Bounded contiguous actually executed student prefix required')
    starts=list(range(motion[0],cut_row,12));targets=[];roles=[]
    for start in starts:
        setter=(events[start].get('raw_intent') or {}).get('setter',{});role=setter.get('execution_role')
        if role=='given_l2_motion':
            candidates=[r for r in blocks if r<=start and (start-r)%12==0 and 0<=(start-r)//12<10]
            need(candidates,'Actual fixed-origin E10 CPU block required for each source motion cell')
            # Context changes can discard the old unexecuted chunk suffix.
            row=max(candidates);absolute=rt.shared.contract._array(blocks[row],(20,7),'actual returned H20')
            target=absolute[(start-row)//12]
        else:
            need(role=='given_l2_history_wait' and targets,
                'Only authenticated internal native history waits may accompany policy motion')
            target=rt.shared.contract._array(setter.get('target_q'),(7,),'actual held history target')
            need(np.array_equal(target,targets[-1]),'Native history wait changed the retained live target')
        for index in range(12):
            e=events[start+index];raw=e.get('raw_intent');setter={} if raw is None else raw.get('setter',{})
            need(e.get('actual_control_owner')=='ACT_L2' and e.get('tick_error') is None and
                e.get('state_audit')=='completed' and e.get('physical_violations')==[] and raw is not None and
                raw.get('native_tick_in_cell')==index and setter.get('physics_row_start')==start and
                setter.get('control_owner')=='ACT_L2' and setter.get('execution_role')==role and
                setter.get('offgrid_stop') is False and np.array_equal(setter.get('target_q'),target) and
                np.array_equal(raw.get('live_target_q'),target) and np.array_equal(raw.get('live_target_after_step_q'),target),
                'Source raw setter differs from completed actual CPU/native action or includes a failed tick')
        targets.append(target.tolist());roles.append(role)
    return dict(start_row=starts[0],end_row_exclusive=cut_row,raw_targets=targets,
        source_roles=roles,source_policy_motion_cells=roles.count('given_l2_motion'),
        source_history_wait_cells=roles.count('given_l2_history_wait'),
        source_kind='retained_actual_student_raw_targets',positive_teacher_labels=False)


def _normal_owner(value):
    stage=value.get('stage',{})
    return (value.get('completed') is True and value.get('gpu_released') is True and
        stage.get('passed') is True and type(stage.get('exit_code')) is int and stage['exit_code']==0 and
        stage.get('process_group_drained') is True and all(stage.get(k)==[] for k in ('errors','signals','deferred_signals')))


def authenticate(config_ref,owner_ref,child_ref,native_ref):
    # Read metadata first: validation/test trajectories never become data inputs.
    config=rt.read_bound(config_ref);dep=config.get(adapter.ROUTE_KEY,{})
    meta=dep.get('metadata',{})
    need(meta.get('split')=='train','Student recovery sources must be TRAIN before native payload access')
    rt.shared.require_membership('train',meta.get('subject_id'))
    owner=rt.read_bound(owner_ref);child=rt.read_bound(child_ref)
    need(_normal_owner(owner),'Original student owner must close normally, drain and release')
    jobs=[j for j in owner.get('config',{}).get('jobs',[]) if
        j.get('config')==config_ref['path'] and j.get('sha256')==config_ref['sha256']]
    need(owner.get('config',{}).get('experiment')=='isaac_batch' and len(jobs)==1,
        'Exact original student batch/config required')
    root=Path(owner_ref['path']).parent/'output'/jobs[0]['id']
    identity=child.get('identity',{});group=owner['stage'].get('process_identity',{})
    need(Path(child_ref['path'])==root.parent/(jobs[0]['id']+'.result.json') and
        Path(native_ref['path'])==root/'COMPLETE.json' and child.get('id')==jobs[0]['id'] and
        child.get('output')==str(root) and child.get('config_sha256')==config_ref['sha256'] and
        child.get('result_sha256')==native_ref['sha256'] and type(child.get('exit_code')) is int and
        child['exit_code']==0 and child.get('worker_complete') is True and child.get('pid')==identity.get('pid') and
        identity.get('pgid')==identity.get('sid')==group.get('pgid')==group.get('sid') and
        group.get('members',{}).get(str(child.get('pid')))==identity.get('start_ticks'),
        'Exact closed source worker/result/owner process identity required')
    model,complete=adapter.model_binding(dep['model']['complete'])  # Original numerical/CPU admission; no model load.
    need(model==dep['model'],'Actual selected student model changed')
    nominal=rt.read_bound(dep['source_config'])
    need(nominal.get('m2_given_contact_control',{}).get('metadata',{}).get('split')=='train',
        'Original nominal source must also be TRAIN')
    current=adapter.model_sources(model) if model.get('checkpoint_schema') in ('m2_act_auxiliary_act_pilot_v4','m2_act_auxiliary_act_pilot_v5','m2_act_auxiliary_act_pilot_v6','m2_act_auxiliary_act_pilot_v7') else adapter.sources()
    inventory=dep['source_inventory'];oldroot=Path(inventory['root'])
    need(set(inventory['files'])==set(current['files']) and all(
        {k:r[k] for k in ('bytes','sha256')}=={k:current['files'][name][k] for k in ('bytes','sha256')}
        for name,r in inventory['files'].items()),'Historical student numerical/native source bytes changed')
    need(inventory['code_commit']==owner['code_commit']==(oldroot/'CODE_COMMIT').read_text().strip() and
        inventory['source_manifest']==rt.reference(oldroot/'SOURCE_MANIFEST.json'),
        'Actual archived source/owner commit differs')
    manifest={r['file']:r for r in rt.read_bound(inventory['source_manifest'])}
    for name,ref in inventory['files'].items():
        need(Path(ref['path'])==oldroot/'sim/ausc_policy'/name and
            rt.reference(ref['path'])==ref and
            all(manifest['sim/ausc_policy/'+name][k]==ref[k] for k in ('bytes','sha256')),
            'Historical student source archive binding differs')
    options={'stop_residual_trajectory':config['stop_residual_trajectory']} if 'stop_residual_trajectory' in config else {}
    expected=adapter.derive_config(nominal,source_config=dep['source_config'],model=model,run_id=owner['run_id'],
        source_inventory=inventory,scope=complete['scope'],**options)
    need(config==expected,'Historical student altered its original qualified scene/reference')
    run=rt.read_bound(native_ref)
    need(run.get('learned_policy') is True and run.get('expert_l2_fallback') is False and
        run.get('learned_policy_receipt',{}).get('model')==model and
        run.get('learned_policy_receipt',{}).get('metadata')==meta,
        'Actual student motor attribution required; no teacher substitution')
    receipt=run.get('neutral_native_audit',{});bound=receipt.get('pilot_binding',{})
    need(receipt.get('source_inventory')==inventory and receipt.get('metadata')==meta and
        bound.get('configuration_sha256')==rt.digest(config) and bound.get('model')==model,
        'Original actual native observer/model/configuration binding differs')
    events=neutral._read_native_events(receipt,meta)
    need(len(events)==run.get('physics_steps'),'Actual source native count differs')
    for path,sha in run.get('runtime_sources_sha256',{}).items():
        need(rt.reference(path)['sha256']==sha,'Actual executed student source changed')
    need(bool(run.get('runtime_sources_sha256')),'Actual native runtime source inventory required')
    cpu=run.get('cpu_child_exit',{})
    need(cpu.get('normal_exit') is True and cpu.get('reaped') is True and cpu.get('exit_code')==0 and
        cpu.get('signals')==[] and cpu.get('started') is True and
        run.get('learned_policy_receipt',{}).get('cpu_child_exit')==cpu,
        'Actual CPU child must finish normally')
    records=[neutral.deployment.wire_decode(line) for line in rt.bound_path(cpu['trace']).read_text().splitlines()]
    need(records and records[0].get('event')=='ready','Actual CPU handshake required')
    ready=records[0]['value'];deployment_ref=rt.reference(root/'pilot_cpu_deployment.json')
    need(rt.read_bound(deployment_ref)==dep and ready.get('ready') is True and
        ready.get('schema')==adapter.PROTOCOL_SCHEMA and ready.get('deployment_sha256')==deployment_ref['sha256'] and
        ready.get('checkpoint')==model['checkpoint'] and ready.get('source_inventory')==inventory and
        ready.get('pid')==cpu['identity']['pid'] and ready.get('parent_pid')==cpu['parent_identity']['pid'] and
        all(cpu['identity'][k]==cpu['parent_identity'][k] for k in ('pgid','sid')),
        'Actual CPU handshake/checkpoint/source/session differs')
    rt.validate_execution_settings(ready['execution'],device='cpu')
    records=records[1:]
    need(len(records)==cpu.get('request_count')==len(run.get('l2_queries',[])),
        'Original CPU request count differs')
    blocks={};requests=[];previous=None
    for i,record in enumerate(records,1):
        need(record.get('event')=='inference' and record.get('request_id')==i and record.get('accepted') is True,
            'Actual successful CPU exchange required')
        request=record['request'];adapter.validate_request(request,dep,sequence=i,previous=previous)
        block=adapter.decode_response(record['response'],request);row=request['context']['physics_row']
        need(0<=row<len(events) and request['context']['observed_at_s']==events[row]['observed_before']['time_s'] and
            request['measured_q']==events[row]['observed_before']['q'],
            'Source CPU query must bind to the same-time actual measured state')
        blocks[row]=np.asarray(request['measured_q'])+block.values
        requests.append(request);previous=request
    return dict(metadata=meta,model=model,source_config=dep['source_config'],events=events,
        requests=requests,blocks=blocks)


def value(binding,source,cut_distance_m,generator):
    cut=select_cut(binding['requests'],cut_distance_m)
    cells=extract_cells(binding['events'],binding['blocks'],cut_row=cut)
    return dict(schema=SCHEMA,source=source,source_config=binding['source_config'],
        subject_id=binding['metadata']['subject_id'],split='train',target_id=binding['metadata']['target_id'],
        source_model=binding['model'],reference_sha256=binding['metadata']['reference_sha256'],
        cut_distance_m=cut_distance_m,initial_measured_q=binding['events'][cells['start_row']]['observed_before']['q'],
        initial_context_phase=binding['requests'][0]['context']['context'][-1],
        generator=generator,**cells)


def export(config_ref,owner_ref,child_ref,native_ref,cut_distance_m,output):
    source=dict(config=config_ref,owner=owner_ref,child=child_ref,native=native_ref)
    binding=authenticate(config_ref,owner_ref,child_ref,native_ref)
    return rt.write_new(output,value(binding,source,cut_distance_m,rt.reference(__file__)))


def load(ref,source_config):
    saved=rt.read_bound(ref)
    if saved.get('schema')=='m2_actual_student_query_recovery_prefix_v2':
        from . import m2_query_recovery_prefix as query
        return query.load(ref,source_config)
    need(saved.get('schema')==SCHEMA and saved.get('split')=='train' and saved.get('source_config')==source_config,
        'Explicit same-scene TRAIN prefix required')
    rt.bound_path(saved['generator'])
    source=saved['source'];binding=authenticate(source['config'],source['owner'],source['child'],source['native'])
    need(saved==value(binding,source,saved['cut_distance_m'],saved['generator']),
        'Retained student targets or cut differ from actual closed native evidence')
    return saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','owner','child','native','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--cut-distance',required=True,type=float,choices=(.080,.060));args=p.parse_args()
    need(sys.platform=='linux' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Linux CPU-only export required')
    export(*(rt.reference(getattr(args,k)) for k in ('config','owner','child','native')),args.cut_distance,args.output)


if __name__=='__main__':main()
