"""Buffered full neutral native evidence for diagnostic learned-pilot routes.

Reuse the original NativeAudit observer, never its formal attach/capability.
No command, physical threshold, camera, capture label or native tick changes.
"""
import argparse
from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path
from types import MethodType

from . import m2_given_contact_runtime as runtime
from . import m2_given_contact_evaluation as neutral

need=runtime.need
FLUSH_ROWS=240
FLAGS=dict(diagnostic_only=True,formal_acceptance=False,test_payloads_opened=False,
    training_eligible=False,raw_capture_complete=False,compact_capture_complete=False)


class BufferedGzipText:
    """NativeAudit flushes each row; only drain complete 240-row batches."""
    def __init__(self,path):
        self.raw=gzip.open(path,'xt',compresslevel=6)
        self.pending=[];self.batch_flushes=0;self.written_rows=0;self.failed=False;self.closed=False

    @property
    def pending_rows(self):return len(self.pending)

    def write(self,value):
        need(not self.closed and not self.failed,'Buffered native writer revoked or closed')
        self.pending.append(value)

    def _drain(self):
        if not self.pending:return
        need(not self.failed,'Buffered native writer revoked')
        try:self.raw.write(''.join(self.pending));self.raw.flush()
        except BaseException:self.failed=True;raise
        self.written_rows+=len(self.pending);self.pending.clear();self.batch_flushes+=1

    def flush(self):
        if self.pending_rows>=FLUSH_ROWS:self._drain()

    def abort_close(self):
        try:self.raw.close()
        finally:self.closed=True

    def close(self):
        if self.closed:return
        try:
            need(not self.failed,'Buffered native writer revoked')
            self._drain()
        finally:self.abort_close()


class BufferedNativeAudit(neutral.NativeAudit):
    """Full original observer/event schema, buffered transport and bound owner."""
    def __init__(self,path,controller,adapter,config):
        self.path=Path(path);self.stream=BufferedGzipText(self.path);self.metadata=deepcopy(controller.metadata)
        self.rows=0;self.errors=[];self.receipt=None;self.controller=controller;self.adapter=adapter
        self.sources=deepcopy(adapter.sources());self.original_class=type(controller)
        dep=config[adapter.ROUTE_KEY]
        self.binding=dict(configuration_sha256=runtime.digest(config),model=deepcopy(dep['model']),
            episode_id=controller.episode_id,source_config=deepcopy(dep['source_config']),
            controller_class=type(controller).__module__+'.'+type(controller).__name__,
            controller_source=runtime.reference(inspect.getfile(type(controller))),
            task_contract=runtime.task_contract(),flush_rows=FLUSH_ROWS,full_native_row=True,**FLAGS)

    def _identity(self,controller):
        need(controller is self.controller and type(controller) is self.original_class and controller.metadata==self.metadata and
            controller.episode_id==self.binding['episode_id'] and controller.reference.sha256==self.metadata['reference_sha256'] and
            controller.policy_receipt['model']==self.binding['model'],'Pilot native observer/controller/model/metadata binding changed')

    def tick(self,controller,*args,**kwargs):
        need(not self.errors,'Pilot native audit revoked after an earlier failure')
        try:self._identity(controller)
        except BaseException as error:self.errors.append(dict(stage='binding',error=repr(error)));raise
        return super().tick(controller,*args,**kwargs)

    def close(self):
        if self.receipt is not None:return self.receipt
        try:
            self._identity(self.controller)
            need(self.adapter.sources()==self.sources,'Pilot native observer source bytes changed')
        except BaseException as error:self.errors.append(dict(stage='binding',error=repr(error)))
        try:self.stream.close()
        except BaseException as error:
            self.errors.append(dict(stage='close',error=repr(error)))
            try:self.stream.abort_close()
            except BaseException as closing:self.errors.append(dict(stage='abort_close',error=repr(closing)))
        trace=None
        try:trace=runtime.reference(self.path)
        except BaseException as error:self.errors.append(dict(stage='trace_reference',error=repr(error)))
        value=dict(schema='m2_given_contact_neutral_native_audit_v1',metadata=self.metadata,
            rows=self.rows,trace=trace,complete=not self.errors,errors=deepcopy(self.errors),source_inventory=self.sources,
            pilot_binding=deepcopy(self.binding),buffer=dict(flush_rows=FLUSH_ROWS,batch_flushes=self.stream.batch_flushes,
                written_rows=self.stream.written_rows,pending_rows=self.stream.pending_rows),**FLAGS)
        try:index=runtime.write_new(self.path.with_suffix('.index.json'),value)
        except BaseException as error:
            self.errors.append(dict(stage='index_write',error=repr(error)))
            value.update(complete=False,errors=deepcopy(self.errors));index=None
        self.receipt=dict(value,**({'index':index} if index is not None else {}));return self.receipt


def adapter_for(config):
    from . import m2_given_contact_pilot_deployment as pilot
    from . import m2_given_contact_expanded_pilot_deployment as expanded
    from . import m2_given_contact_expanded_val_scenarios as validation
    from . import m2_rebuild_paired_eval as rebuilt
    matches=[module for module in (pilot,expanded,validation,rebuilt) if module.ROUTE_KEY in config]
    need(len(matches)==1 and not any(key in config for key in ('m2_given_contact_evaluation','m2_given_contact_deployment',
        'm2_given_contact_control','m2_act_deployment')),'Only one explicit diagnostic learned-pilot route may attach or qualify')
    return matches[0]


def attach(controller,config,output,*,adapter=None):
    from .m2_given_contact_pilot_deployment import PilotGivenACT
    adapter=adapter_for(config) if adapter is None else adapter
    need(isinstance(controller,PilotGivenACT) and not hasattr(controller,'_neutral_native_audit'),
        'Actual pilot GivenContactACT and one observer attachment required')
    binding=adapter.preflight(config,output=output);dep=binding['deployment']
    need(controller.metadata==dep['metadata'] and controller.episode_id==dep['episode_id'] and
        controller.policy_receipt['model']==dep['model'],'Pilot native controller differs from authenticated deployment')
    original=controller._tick
    audit=BufferedNativeAudit(Path(output)/'given_control_native.jsonl.gz',controller,adapter,config)
    def tick(self,command,subphase,*args,**kwargs):
        return audit.tick(self,command,subphase,lambda:original(command,subphase,*args,**kwargs))
    controller._neutral_native_audit=audit;controller._neutral_original_tick=original
    controller._pilot_light_native_audit=audit;controller._tick=MethodType(tick,controller)
    return audit


def installed(controller):
    audit=getattr(controller,'_pilot_light_native_audit',None)
    return isinstance(audit,BufferedNativeAudit) and getattr(controller,'_neutral_native_audit',None) is audit and audit.controller is controller


def trace_path(controller):
    need(installed(controller),'Explicit installed pilot native observer required')
    return controller._pilot_light_native_audit.path


def close_checked(controller):
    receipt=neutral.close_native_audit(controller)
    need(receipt is not None and receipt.get('complete') is True and receipt.get('errors')==[],
        'Pilot neutral native audit incomplete; physical acceptance revoked')
    return receipt


def _ipc(output,run,dep,adapter,events):
    receipt=run.get('cpu_child_exit',{})
    need(receipt.get('normal_exit') is True and receipt.get('reaped') is True and receipt.get('exit_code')==0 and
        receipt.get('started') is True and receipt.get('signals')==[],'Actual normal CPU child exit required')
    identity=receipt['identity'];parent=receipt['parent_identity']
    need(all(identity[k]==parent[k] for k in ('pgid','sid')),'Actual CPU child left its original owner session')
    runtime.bound_path(receipt['stderr']);path=runtime.bound_path(receipt['trace'])
    with path.open() as stream:records=[neutral.deployment.wire_decode(line) for line in stream]
    need(records and records[0].get('event')=='ready','Actual official CPU ready record required')
    ready=records[0]['value'];deployment_ref=runtime.reference(Path(output)/'pilot_cpu_deployment.json')
    need(runtime.read_bound(deployment_ref)==dep and ready.get('ready') is True and ready.get('schema')==adapter.PROTOCOL_SCHEMA and
        ready.get('deployment_sha256')==deployment_ref['sha256'] and ready.get('checkpoint')==dep['model']['checkpoint'] and
        ready.get('source_inventory')==dep['source_inventory'] and ready.get('pid')==identity['pid'] and
        ready.get('parent_pid')==parent['pid'],'Actual CPU handshake/model/source/parent binding differs')
    runtime.validate_execution_settings(ready['execution'],device='cpu')
    queries=run.get('l2_queries',[]);records=records[1:]
    need(records and len(records)==receipt.get('request_count')==len(queries) and
        run.get('learned_policy_receipt',{}).get('cpu_child_exit')==receipt,'Actual CPU inference/query/exit count binding differs')
    previous=None
    for i,(record,query) in enumerate(zip(records,queries),1):
        need(record.get('event')=='inference' and record.get('request_id')==i and
            record.get('accepted') is True and query.get('accepted') is True,'Successful actual CPU exchange and native query required')
        request=record['request'];adapter.validate_request(request,dep,sequence=i,previous=previous)
        adapter.decode_response(record['response'],request)
        row=request['context']['physics_row']
        need(0<=row<len(events) and request['context']['observed_at_s']==events[row]['observed_before']['time_s'] and
            request['measured_q']==events[row]['observed_before']['q'],'Actual inference must bind to same-time measured native state')
        previous=request
    actions=run.get('l2_actions',[])
    need(actions and all(type(action.get('query_index')) is int and 0<=action['query_index']<len(queries) and
        type(action.get('physics_row_start')) is int and type(action.get('physics_row_end_exclusive')) is int and
        0<=action['physics_row_start']<action['physics_row_end_exclusive']<=len(events) for action in actions),
        'Actual learned L2 actions must have executed native tick ranges')
    return dict(request_count=len(records),executed_actions=len(actions),normal_exit=True,trace=receipt['trace'])


def qualify(output,config,*,receipt_path=None):
    """Recheck actual closed evidence with the unchanged dataset physical gate."""
    output=Path(output);inputs={};checks={};passed=False;error=None
    try:
        if not isinstance(config,dict):
            inputs['configuration']=runtime.reference(config);config=runtime.read_bound(inputs['configuration'])
        adapter=adapter_for(config);dep=config[adapter.ROUTE_KEY];meta=dep['metadata']
        need(meta.get('split') in ('train','validation'),'Diagnostic qualification cannot open TEST payloads')
        runtime.shared.require_membership(meta['split'],meta.get('subject_id'));adapter.preflight(config,output=output)
        inputs['run']=runtime.reference(output/'fresh_expert_result.json');run=runtime.read_bound(inputs['run'])
        inputs['complete']=runtime.reference(output/'COMPLETE.json');complete=runtime.read_bound(inputs['complete'])
        need(runtime.digest(complete)==runtime.digest(run) and run.get('success') is True and run.get('physical_validation') is True and
            run.get('errors')==[] and run.get('learned_policy') is True and run.get('expert_l2_fallback') is False,
            'Actual closed successful learned native result without errors required')
        receipt=run.get('neutral_native_audit',{})
        need(receipt.get('complete') is True and receipt.get('errors')==[] and receipt.get('metadata')==meta and
            receipt.get('source_inventory')==adapter.sources() and receipt.get('pilot_binding',{}).get('configuration_sha256')==runtime.digest(config) and
            receipt['pilot_binding'].get('model')==dep['model'],'Actual closed pilot native audit/model/source binding differs')
        events=neutral.read_native_trace(receipt,meta)
        need(len(events)==run.get('physics_steps') and run.get('trace_file')==receipt['trace']['path'] and
            run.get('trace_sha256')==receipt['trace']['sha256'],'Actual complete native tick count/trace hash differs')
        learned=run.get('learned_policy_receipt',{})
        need(learned.get('model')==dep['model'] and learned.get('metadata')==meta and
            learned.get('source_inventory')==dep['source_inventory'],'Actual learned controller/model/reference attribution differs')
        checks['ipc']=_ipc(output,run,dep,adapter,events)
        runtime.dataset._physical(run,events,meta)
        checks['unchanged_physical_gate_passed']=True
        force_keys=sorted({key for event in events for key in event['measurements'] if key.endswith('_N')})
        checks['physical']=dict(physics_steps=len(events),
            maximum_force_N={key:max(event['measurements'].get(key,0.) for event in events) for key in force_keys},
            hold_evidence=deepcopy(run['points'][0].get('force_audit',{})),
            normal_release_confirm_ticks=sum(event['controller_after']['subphase']=='normal_release_confirm' for event in events),
            home_settle_ticks=sum(event['controller_after']['subphase']=='home_settle' for event in events),
            final_subphase=events[-1]['controller_after']['subphase'],final_measured_q=events[-1]['observed_after']['q'])
        passed=True
    except Exception as failure:error=type(failure).__name__+': '+str(failure)
    value=dict(schema='m2_given_contact_diagnostic_native_qualification_v1',passed=passed,inputs=inputs,checks=checks,error=error,
        configuration_sha256=runtime.digest(config) if isinstance(config,dict) else None,
        scope='Actual selected-policy diagnostic simulation evidence; no raw training labels or formal/held-out TEST acceptance.',**FLAGS)
    return runtime.write_new(output/'PILOT_NATIVE_QUALIFICATION.json' if receipt_path is None else receipt_path,value)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--qualify',action='store_true',required=True)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--receipt')
    args=parser.parse_args();ref=qualify(args.output,args.config,receipt_path=args.receipt);value=runtime.read_bound(ref)
    print(json.dumps(dict(receipt=ref,passed=value['passed'],error=value['error'])),flush=True)
    return 0 if value['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
